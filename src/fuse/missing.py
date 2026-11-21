"""Missing-aware FUSE: every step uses observed ratings only (see docs/missing_data_plan.md).

The paper's convention (missing -> 0 after mapping to [-1, 1]) turns every missing rating into
a vote once the column is binarised at tau. Here a missing rating stays missing:

* moments are pairwise / triplet-complete: mu_j over the items j rated, Sigma_jk over the
  items both rated, T_jkl over the items all three rated (centred with the per-verifier means);
* pairs / triplets with fewer than `min_overlap` co-rated items are treated as unobserved by
  the rank-one fit, the class-imbalance fit and the TCI statistic;
* the posterior of an item uses only the verifiers that rated it.

With no missing entries every function here returns exactly what its counterpart in
moments.py / tci.py / posterior.py returns. The estimates are consistent when which items a
verifier rates does not depend on the label (random or batch assignment). When assignment
depends on the label, the prevalence differs between pairs and the rank-one model is off.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .moments import (
    MomentTensors,
    MoMEstimate,
    _distinct_triplet_mask,
    orient_by_majority,
    rank_one_offdiagonal,
    sensitivities_specificities,
)
from .tci import ThresholdResult, candidate_thresholds


@dataclass(frozen=True)
class MaskedMoments(MomentTensors):
    pair_ok: np.ndarray  # (m, m) bool, off-diagonal pairs with enough co-rated items
    triplet_ok: np.ndarray  # (m, m, m) bool, distinct triplets with enough co-rated items


def minmax_observed(V: np.ndarray) -> np.ndarray:
    """Map each column's observed values to [-1, 1]; NaN stays NaN."""
    V = np.asarray(V, dtype=float)
    lo, hi = np.nanmin(V, axis=0), np.nanmax(V, axis=0)
    span = hi - lo
    out = np.where(np.isnan(V), np.nan, 0.0)
    nz = span > 0
    out[:, nz] = 2.0 * (V[:, nz] - lo[nz]) / span[nz] - 1.0
    return out


def binarize_observed(V: np.ndarray, tau: np.ndarray) -> np.ndarray:
    return np.where(np.isnan(V), np.nan, np.where(V > tau[None, :], 1.0, -1.0))


def _third(X: np.ndarray, Y: np.ndarray | None = None, chunk: int = 4096) -> np.ndarray:
    """sum_i X_ij Y_ik Y_il as an (m, m, m) array, chunked like moments.moment_tensors."""
    Y = X if Y is None else Y
    n, m = X.shape
    T = np.zeros((m, m * m))
    for s in range(0, n, chunk):
        Yc = Y[s : s + chunk]
        T += X[s : s + chunk].T @ (Yc[:, :, None] * Yc[:, None, :]).reshape(len(Yc), m * m)
    return T.reshape(m, m, m)


def _counts(O: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return O.T @ O, _third(O)


def masked_moments(V: np.ndarray, min_overlap: int = 20, counts=None) -> MaskedMoments:
    """Pairwise/triplet-complete mu, Sigma, T of a matrix with NaN for missing entries."""
    V = np.asarray(V, dtype=float)
    obs = ~np.isnan(V)
    O = obs.astype(float)
    C2, C3 = counts if counts is not None else _counts(O)
    mu = np.where(obs.any(0), np.nanmean(np.where(obs, V, np.nan), axis=0), 0.0) if obs.any() else np.zeros(V.shape[1])
    X = np.where(obs, V - mu, 0.0)
    sigma = (X.T @ X) / np.maximum(C2, 1)
    T = _third(X) / np.maximum(C3, 1)
    m = V.shape[1]
    pair_ok = (C2 >= min_overlap) & ~np.eye(m, dtype=bool)
    triplet_ok = (C3 >= min_overlap) & _distinct_triplet_mask(m)
    return MaskedMoments(mu=mu, sigma=sigma, T=T, pair_ok=pair_ok, triplet_ok=triplet_ok)


def _class_imbalance(T: np.ndarray, u: np.ndarray, mask: np.ndarray) -> float:
    """moments.estimate_class_imbalance restricted to the triplets in `mask`."""
    P = u[:, None, None] * u[None, :, None] * u[None, None, :]
    denom = np.sum((P**2)[mask])
    if denom < 1e-18:
        return 0.0
    c = np.sum((T * P)[mask]) / denom
    return float(np.clip(-c / np.sqrt(4.0 + c**2), -1 + 1e-4, 1 - 1e-4))


def masked_mom_estimate(V: np.ndarray, min_overlap: int = 20, moments: MaskedMoments | None = None) -> MoMEstimate:
    mm = moments or masked_moments(V, min_overlap)
    full = mm.pair_ok | np.eye(len(mm.mu), dtype=bool)
    u = orient_by_majority(rank_one_offdiagonal(mm.sigma, observed=None if full.all() else mm.pair_ok))
    b = _class_imbalance(mm.T, u, mm.triplet_ok)
    psi, eta = sensitivities_specificities(mm.mu, u, b)
    return MoMEstimate(psi=psi, eta=eta, b=b, u=u, mu=mm.mu)


def masked_tci_statistic(sigma, T, pair_ok, triplet_ok, clip: float = 1e-3) -> float:
    """tci.tci_statistic over the observed triplets only."""
    m = sigma.shape[0]
    denom = np.maximum(sigma, clip)
    total = 0.0
    for j3 in range(2, m):
        j1, j2 = np.triu_indices(j3, k=1)
        keep = triplet_ok[j1, j2, j3] & pair_ok[j1, j2]
        if keep.sum() > 1:
            total += float(np.var(T[j1[keep], j2[keep], j3] / denom[j1[keep], j2[keep]], ddof=1))
    return total


def minimize_tci_masked(V: np.ndarray, n_sweeps: int = 3, n_quantiles: int = 19, clip: float = 1e-3,
                        min_overlap: int = 20) -> ThresholdResult:
    """tci.minimize_tci with missing entries. Which entries are observed never changes with
    tau, so the co-rating counts are computed once and only the numerators are updated."""
    V = np.asarray(V, dtype=float)
    n, m = V.shape
    obs = ~np.isnan(V)
    candidates = [candidate_thresholds(V[obs[:, j], j], n_quantiles) for j in range(m)]
    active = np.array([c.size > 0 for c in candidates])
    tau = np.full(m, np.inf)
    for j in np.flatnonzero(active):
        med = np.median(V[obs[:, j], j])
        tau[j] = candidates[j][np.argmin(np.abs(candidates[j] - med))]
    idx = np.flatnonzero(active)
    if idx.size < 3:
        return ThresholdResult(tau=tau, statistic=0.0, active=active)

    Va, Oa = V[:, idx], obs[:, idx].astype(float)
    C2, C3 = _counts(Oa)
    tau_a = tau[idx].copy()
    mm = masked_moments(binarize_observed(Va, tau_a), min_overlap, counts=(C2, C3))
    pair_ok, triplet_ok = mm.pair_ok, mm.triplet_ok
    X = np.where(Oa > 0, binarize_observed(Va, tau_a) - mm.mu, 0.0)
    sigma, T = mm.sigma.copy(), mm.T.copy()
    best = masked_tci_statistic(sigma, T, pair_ok, triplet_ok, clip)

    for _ in range(n_sweeps):
        improved = False
        for k in range(idx.size):
            ok = Oa[:, k] > 0
            for t in candidates[idx[k]]:
                if t == tau_a[k]:
                    continue
                col = np.where(Va[ok, k] > t, 1.0, -1.0)
                xk = np.zeros(n)
                xk[ok] = col - col.mean()
                Xn = X.copy()
                Xn[:, k] = xk
                s_new = sigma.copy()
                row = (Xn.T @ xk) / np.maximum(C2[k], 1)
                s_new[k, :] = row
                s_new[:, k] = row
                Tk = ((Xn * xk[:, None]).T @ Xn) / np.maximum(C3[k], 1)
                T_new = T.copy()
                T_new[k, :, :] = Tk
                T_new[:, k, :] = Tk
                T_new[:, :, k] = Tk
                stat = masked_tci_statistic(s_new, T_new, pair_ok, triplet_ok, clip)
                if stat < best - 1e-12:
                    best, tau_a[k], X, sigma, T = stat, t, Xn, s_new, T_new
                    improved = True
        if not improved:
            break
    tau[idx] = tau_a
    return ThresholdResult(tau=tau, statistic=best, active=active)


def _log_factors(Vb: np.ndarray, est: MoMEstimate):
    """Log likelihood factors per (item, verifier); a missing entry contributes 0 (factor 1)."""
    V = np.nan_to_num(Vb, nan=0.0)
    f_pos = np.clip(1.0 - V + 2.0 * V * est.psi[None, :], 1e-12, None)
    f_neg = np.clip(1.0 + V - 2.0 * V * est.eta[None, :], 1e-12, None)
    return np.log(f_pos), np.log(f_neg)


def product_posterior(Vb: np.ndarray, est: MoMEstimate) -> np.ndarray:
    """P(y=1 | all observed votes) under conditional independence: one Dawid-Skene E-step
    with the moment estimates (variant B in the plan)."""
    lp, ln = _log_factors(Vb, est)
    logit = np.log1p(est.b) - np.log1p(-est.b) + lp.sum(1) - ln.sum(1)
    return 1.0 / (1.0 + np.exp(-np.clip(logit, -500, 500)))


def observed_triplet_posterior(Vb: np.ndarray, est: MoMEstimate, max_triplets: int = 20000,
                               chunk: int = 512, rng: np.random.Generator | None = None) -> np.ndarray:
    """Equation (9) restricted, per item, to the triplets whose three verifiers all rated it
    (variant A). Items with no fully observed triplet fall back to the product posterior."""
    n, m = Vb.shape
    obs = ~np.isnan(Vb)
    lp, ln = _log_factors(Vb, est)
    triplets = np.array(list(combinations(range(m), 3)))
    if len(triplets) > max_triplets:
        rng = rng or np.random.default_rng(0)
        triplets = triplets[rng.choice(len(triplets), max_triplets, replace=False)]
    prior = np.log1p(est.b) - np.log1p(-est.b)
    total, cnt = np.zeros(n), np.zeros(n)
    for s in range(0, len(triplets), chunk):
        tr = triplets[s : s + chunk]
        valid = obs[:, tr].all(axis=2)
        logit = prior + lp[:, tr].sum(axis=2) - ln[:, tr].sum(axis=2)
        total += (valid / (1.0 + np.exp(-logit))).sum(1)
        cnt += valid.sum(1)
    out = product_posterior(Vb, est)
    has = cnt > 0
    out[has] = total[has] / cnt[has]
    return out


def em_from(Vb: np.ndarray, q: np.ndarray, n_iter: int = 200, tol: float = 1e-8) -> np.ndarray:
    """Binary one-coin-per-class Dawid-Skene EM on {+1, -1, NaN} votes, started from the item
    posteriors q (e.g. FUSE's) instead of the majority-vote share."""
    M = (~np.isnan(Vb)).astype(float)
    Bp = np.where(M > 0, (Vb > 0).astype(float), 0.0)
    q = np.clip(q, 1e-3, 1 - 1e-3)
    for _ in range(n_iter):
        prior = np.clip(q.mean(), 1e-3, 1 - 1e-3)
        psi = ((q[:, None] * Bp).sum(0) + 1) / ((q[:, None] * M).sum(0) + 2)
        eta = (((1 - q)[:, None] * (M - Bp)).sum(0) + 1) / (((1 - q)[:, None] * M).sum(0) + 2)
        lp = np.log(prior) + Bp @ np.log(psi) + (M - Bp) @ np.log(1 - psi)
        lq = np.log(1 - prior) + Bp @ np.log(1 - eta) + (M - Bp) @ np.log(eta)
        q_new = 1 / (1 + np.exp(np.clip(lq - lp, -500, 500)))
        done = np.max(np.abs(q_new - q)) < tol
        q = q_new
        if done:
            break
    return q
