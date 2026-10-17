"""Fully Unsupervised Score Ensembling (paper Algorithm 1).

    1. tau*  = argmin_tau S-hat(g_tau(V))                       (minimise TCI violation)
    2. V~    = g_tau*(V);  (psi, eta, b) = MoM(V~)               (Jaffe et al., 2015)
       drop verifiers whose estimated balanced accuracy < 1/2    (Appendix D)
    3. p-hat = mean over triplets of the TCI posterior            (eq. 9)
       theta* = argmax_theta sum_i (2 p-hat_i - 1) f_theta(V_i)   (eq. 7)
       return argmax_i f_theta*(V_i)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from sklearn.linear_model import LogisticRegression

from .clustering import cluster_verifiers
from .moments import MoMEstimate, mom_estimate
from .posterior import averaged_triplet_posterior
from .preprocess import minmax_to_unit_interval
from .tci import binarize, minimize_tci

Predictor = Literal["logistic", "posterior"]


@dataclass
class FUSEResult:
    scores: np.ndarray  # predicted probability of correctness per response, (N,)
    posterior: np.ndarray | None  # p-hat per response, (N,)
    kept: np.ndarray  # columns used for the posterior (cluster indices if clustered)
    tau: np.ndarray | None
    estimate: MoMEstimate | None
    tci: float | None
    fallback: str | None = None
    info: dict = field(default_factory=dict)

    def select(self) -> int:
        return int(np.argmax(self.scores))


@dataclass
class FUSE:
    """Unsupervised verifier ensemble.

    Parameters mirror the design choices in the paper: `n_sweeps`/`n_quantiles` control the
    coordinate descent over thresholds, `drop_below_random` implements the balanced-accuracy
    dropping heuristic, and `predictor` chooses between the paper's default logistic
    regression fitted to pseudo-labels ("logistic") and the parameter-free rule of ranking by
    p-hat directly ("posterior"). `cluster_threshold`, when set, is not in the paper: it
    averages blocks of verifiers whose |Spearman correlation| exceeds it into one
    meta-verifier before thresholding (see `clustering.py`).
    """

    n_sweeps: int = 3
    n_quantiles: int = 19
    tci_clip: float = 1e-3
    tci_standardize: bool = False
    drop_below_random: bool = True
    refit_after_drop: bool = True
    predictor: Predictor = "logistic"
    logistic_C: float = 1.0
    max_triplets: int = 20000
    cluster_threshold: float | None = None
    seed: int = 0

    def fit_scores(self, V: np.ndarray, rescale: bool = True) -> FUSEResult:
        """Run FUSE on one score matrix V (N responses x m verifiers).

        With rescale=True each verifier column is min-max mapped to [-1, 1] first
        (Appendix A.2, footnote 3). NaNs must be imputed before calling.
        """
        V = np.asarray(V, dtype=float)
        if np.isnan(V).any():
            raise ValueError("impute missing scores before running FUSE")
        Vs = minmax_to_unit_interval(V) if rescale else V
        info: dict = {}
        Vm = Vs
        if self.cluster_threshold is not None:
            clusters = cluster_verifiers(Vs, self.cluster_threshold)
            Vm = clusters.collapse(Vs)
            info["clusters"] = clusters.groups

        thr = minimize_tci(
            Vm,
            n_sweeps=self.n_sweeps,
            n_quantiles=self.n_quantiles,
            clip=self.tci_clip,
            standardize=self.tci_standardize,
        )
        active = np.flatnonzero(thr.active)
        if active.size < 3:
            return self._fallback(Vs, "fewer than three non-constant verifiers")

        Vt = binarize(Vm[:, active], thr.tau[active])
        est = mom_estimate(Vt)
        kept = np.arange(active.size)
        if self.drop_below_random:
            good = np.flatnonzero(est.balanced_accuracy >= 0.5)
            kept = good if good.size >= 3 else np.argsort(-est.balanced_accuracy)[:3]
            if self.refit_after_drop and kept.size < active.size:
                est = mom_estimate(Vt[:, kept])
            else:
                est = _subset(est, kept)

        post = averaged_triplet_posterior(
            Vt[:, kept], est, max_triplets=self.max_triplets, rng=np.random.default_rng(self.seed)
        )
        scores = self._predict(Vs, post)
        return FUSEResult(
            scores=scores,
            posterior=post,
            kept=active[kept],
            tau=thr.tau,
            estimate=est,
            tci=thr.statistic,
            info=info,
        )

    def _predict(self, Vs: np.ndarray, post: np.ndarray) -> np.ndarray:
        if self.predictor == "posterior":
            return post
        # Maximising sum_i (2 p_i - 1) yhat_i is weighted 0-1 classification with labels
        # sign(2 p_i - 1) and weights |2 p_i - 1|; logistic loss is its convex surrogate.
        labels = (post > 0.5).astype(int)
        weights = np.abs(2.0 * post - 1.0)
        if labels.min() == labels.max() or weights.sum() <= 0:
            return post
        clf = LogisticRegression(C=self.logistic_C, max_iter=1000)
        clf.fit(Vs, labels, sample_weight=weights)
        return clf.predict_proba(Vs)[:, 1]

    @staticmethod
    def _fallback(Vs: np.ndarray, reason: str) -> FUSEResult:
        return FUSEResult(
            scores=Vs.mean(axis=1),
            posterior=None,
            kept=np.arange(Vs.shape[1]),
            tau=None,
            estimate=None,
            tci=None,
            fallback=f"naive ensemble ({reason})",
        )


def _subset(est: MoMEstimate, idx: np.ndarray) -> MoMEstimate:
    return MoMEstimate(psi=est.psi[idx], eta=est.eta[idx], b=est.b, u=est.u[idx], mu=est.mu[idx])


def fuse_batched(
    matrices: list[np.ndarray], model: FUSE | None = None
) -> tuple[list[np.ndarray], FUSEResult]:
    """Batched mode (Appendix D): rescale each query's matrix, stack them into one tall
    matrix, learn a single ensemble, and split the scores back per query."""
    model = model or FUSE()
    scaled = [minmax_to_unit_interval(np.asarray(V, dtype=float)) for V in matrices]
    result = model.fit_scores(np.vstack(scaled), rescale=False)
    bounds = np.cumsum([0] + [len(V) for V in scaled])
    return [result.scores[bounds[i] : bounds[i + 1]] for i in range(len(scaled))], result
