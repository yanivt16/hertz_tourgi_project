"""Baselines from the paper (Appendix B, E.1, E.8).

Each selector returns a score per response; the response with the highest score is
selected and ties are handled by `metrics.selection_accuracy`.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.linear_model import LogisticRegression

from .metrics import balanced_accuracy
from .moments import mom_estimate
from .preprocess import minmax_to_unit_interval


# ---- unsupervised --------------------------------------------------------------------


def random_scores(n: int) -> np.ndarray:
    """All-tied scores: selection accuracy equals the mean correctness, i.e. Pass@1."""
    return np.zeros(n)


def majority_vote_scores(answers: list) -> np.ndarray:
    """Score each response by how many responses share its extracted answer."""
    keys = [None if a is None else str(a).strip() for a in answers]
    counts = Counter(k for k in keys if k not in (None, ""))
    if not counts:
        return np.zeros(len(keys))
    return np.array([counts.get(k, 0) if k not in (None, "") else -1 for k in keys], dtype=float)


def naive_ensemble_scores(V: np.ndarray) -> np.ndarray:
    return minmax_to_unit_interval(V).mean(axis=1)


def jaffe2015_scores(V: np.ndarray) -> np.ndarray:
    """MoM estimates plugged into the JCI maximum-likelihood rule (Appendix B).

    The log-likelihood ratio for a {±1} verifier is
        v/2 * log(psi eta / ((1 - psi)(1 - eta))) + 1/2 * log(psi (1 - psi) / (eta (1 - eta))),
    applied linearly to median-binarised scores.
    """
    Vs = minmax_to_unit_interval(V)
    B = np.where(Vs > np.median(Vs, axis=0), 1.0, -1.0)
    keep = B.std(axis=0) > 0
    if keep.sum() < 3:
        return Vs.mean(axis=1)
    B = B[:, keep]
    est = mom_estimate(B)
    psi, eta = est.psi, est.eta
    slope = 0.5 * np.log(psi * eta / ((1 - psi) * (1 - eta)))
    return B @ slope


def dawid_skene_scores(V: np.ndarray, n_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    """One-coin-per-class Dawid & Skene (1979) EM on median-binarised scores."""
    Vs = minmax_to_unit_interval(V)
    B = Vs > np.median(Vs, axis=0)
    keep = B.std(axis=0) > 0
    if keep.sum() == 0:
        return np.zeros(len(V))
    B = B[:, keep].astype(float)
    q = B.mean(axis=1).clip(1e-3, 1 - 1e-3)  # initialise with a soft majority vote
    for _ in range(n_iter):
        prior = q.mean().clip(1e-3, 1 - 1e-3)
        psi = ((q[:, None] * B).sum(0) + 1) / (q.sum() + 2)
        eta = (((1 - q)[:, None] * (1 - B)).sum(0) + 1) / ((1 - q).sum() + 2)
        log_pos = np.log(prior) + B @ np.log(psi) + (1 - B) @ np.log(1 - psi)
        log_neg = np.log(1 - prior) + B @ np.log(1 - eta) + (1 - B) @ np.log(eta)
        q_new = 1 / (1 + np.exp(log_neg - log_pos))
        if np.max(np.abs(q_new - q)) < tol:
            q = q_new
            break
        q = q_new
    return q


# ---- supervised (fit on labelled queries, applied per query) -------------------------


class SupervisedLogistic:
    """Logistic regression on rescaled verifier scores using sklearn defaults."""

    def fit(self, matrices: list[np.ndarray], labels: list[np.ndarray]) -> "SupervisedLogistic":
        X = np.vstack([minmax_to_unit_interval(V) for V in matrices])
        y = np.concatenate(labels).astype(int)
        self.clf = LogisticRegression(max_iter=1000)
        if y.min() == y.max():
            self.constant = float(y[0])
        else:
            self.constant = None
            self.clf.fit(X, y)
        return self

    def scores(self, V: np.ndarray) -> np.ndarray:
        if self.constant is not None:
            return np.full(len(V), self.constant)
        return self.clf.predict_proba(minmax_to_unit_interval(V))[:, 1]


class SupervisedNaiveBayes:
    """Naive Bayes on per-query median-binarised scores with Laplace smoothing."""

    def fit(self, matrices: list[np.ndarray], labels: list[np.ndarray]) -> "SupervisedNaiveBayes":
        B = np.vstack([_median_binarize(V) for V in matrices])
        y = np.concatenate(labels).astype(bool)
        self.prior = (y.sum() + 1) / (len(y) + 2)
        self.p_pos = (B[y].sum(0) + 1) / (y.sum() + 2)
        self.p_neg = (B[~y].sum(0) + 1) / ((~y).sum() + 2)
        return self

    def scores(self, V: np.ndarray) -> np.ndarray:
        B = _median_binarize(V)
        lp = np.log(self.prior) + B @ np.log(self.p_pos) + (1 - B) @ np.log(1 - self.p_pos)
        ln = np.log(1 - self.prior) + B @ np.log(self.p_neg) + (1 - B) @ np.log(1 - self.p_neg)
        return lp - ln


def _median_binarize(V: np.ndarray) -> np.ndarray:
    Vs = minmax_to_unit_interval(V)
    return (Vs > np.median(Vs, axis=0)).astype(float)


# ---- oracles -------------------------------------------------------------------------


def oracle_best_verifier(matrices: list[np.ndarray], labels: list[np.ndarray]) -> int:
    """Index of the verifier with the highest pooled balanced accuracy (uses labels)."""
    X = np.vstack([minmax_to_unit_interval(V) for V in matrices])
    y = np.concatenate(labels)
    accs = [balanced_accuracy(X[:, j], y) for j in range(X.shape[1])]
    return int(np.nanargmax(accs))
