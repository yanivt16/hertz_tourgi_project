"""Label-free measure of triplet-conditional-independence violations and the
threshold search that minimises it (paper Section 2.2, Proposition 2.4).

Under TCI, T_{j1 j2 j3} / Sigma_{j1 j2} = alpha^3 u_{j3} for every pair j1 < j2 < j3, so
for each j3 the ratio is constant across pairs. The statistic sums, over j3, the sample
variance of that ratio; it is zero under TCI and large when TCI is violated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .moments import moment_tensors


def tci_statistic(sigma: np.ndarray, T: np.ndarray, clip: float = 1e-3) -> float:
    """Empirical S-hat of equation (4). The denominator is clipped to a small positive
    number for stability, as in the paper's footnote 1."""
    m = sigma.shape[0]
    denom = np.maximum(sigma, clip)
    total = 0.0
    for j3 in range(2, m):
        j1, j2 = np.triu_indices(j3, k=1)
        ratios = T[j1, j2, j3] / denom[j1, j2]
        if ratios.size > 1:
            total += float(np.var(ratios, ddof=1))
    return total


def binarize(V: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """g_tau(V)_{ij} = sign(v_ij - tau_j), with ties sent to -1."""
    return np.where(V > tau[None, :], 1.0, -1.0)


def candidate_thresholds(column: np.ndarray, n_quantiles: int = 19) -> np.ndarray:
    """Thresholds for one verifier that leave both classes non-empty.

    Real-valued verifiers get a quantile grid; discrete verifiers (few unique values) get
    every midpoint between consecutive unique values.
    """
    uniq = np.unique(column)
    if uniq.size < 2:
        return np.empty(0)
    midpoints = (uniq[:-1] + uniq[1:]) / 2.0
    if midpoints.size <= n_quantiles:
        return midpoints
    levels = np.linspace(0.05, 0.95, n_quantiles)
    qs = np.unique(np.quantile(column, levels))
    return qs[(qs >= uniq[0]) & (qs < uniq[-1])]


@dataclass(frozen=True)
class ThresholdResult:
    tau: np.ndarray
    statistic: float
    active: np.ndarray  # boolean mask of verifiers that are non-constant after binarisation


class _IncrementalMoments:
    """Keeps mu, Sigma and T for a binarised matrix and updates a single column in
    O(N m^2) instead of recomputing everything in O(N m^3)."""

    def __init__(self, B: np.ndarray):
        self.B = B.copy()
        m = moment_tensors(self.B)
        self.mu, self.sigma, self.T = m.mu.copy(), m.sigma.copy(), m.T.copy()
        self.X = self.B - self.mu

    def column_update(self, j: int, new_col: np.ndarray):
        """Return (sigma, T) with column j replaced, without mutating state."""
        n = len(new_col)
        xj = new_col - new_col.mean()
        X = self.X.copy()
        X[:, j] = xj
        sigma = self.sigma.copy()
        row = X.T @ xj / n
        sigma[j, :] = row
        sigma[:, j] = row
        Tj = (X * xj[:, None]).T @ X / n
        T = self.T.copy()
        T[j, :, :] = Tj
        T[:, j, :] = Tj
        T[:, :, j] = Tj
        return sigma, T, X, new_col.mean()

    def commit(self, j: int, new_col: np.ndarray, sigma, T, X, mu_j):
        self.B[:, j] = new_col
        self.sigma, self.T, self.X = sigma, T, X
        self.mu[j] = mu_j


def minimize_tci(
    V: np.ndarray,
    n_sweeps: int = 3,
    n_quantiles: int = 19,
    clip: float = 1e-3,
) -> ThresholdResult:
    """Coordinate descent over per-verifier thresholds tau to minimise S-hat(g_tau(V))."""
    V = np.asarray(V, dtype=float)
    n, m = V.shape
    candidates = [candidate_thresholds(V[:, j], n_quantiles) for j in range(m)]
    active = np.array([c.size > 0 for c in candidates])
    tau = np.full(m, np.inf)
    for j in np.flatnonzero(active):
        median = np.median(V[:, j])
        tau[j] = candidates[j][np.argmin(np.abs(candidates[j] - median))]

    idx = np.flatnonzero(active)
    if idx.size < 3:
        return ThresholdResult(tau=tau, statistic=0.0, active=active)

    Va = V[:, idx]
    cands = [candidates[j] for j in idx]
    tau_a = tau[idx].copy()
    state = _IncrementalMoments(binarize(Va, tau_a))
    best = tci_statistic(state.sigma, state.T, clip)

    for _ in range(n_sweeps):
        improved = False
        for k in range(idx.size):
            for t in cands[k]:
                if t == tau_a[k]:
                    continue
                col = np.where(Va[:, k] > t, 1.0, -1.0)
                sigma, T, X, mu_k = state.column_update(k, col)
                stat = tci_statistic(sigma, T, clip)
                if stat < best - 1e-12:
                    best = stat
                    tau_a[k] = t
                    state.commit(k, col, sigma, T, X, mu_k)
                    improved = True
        if not improved:
            break

    tau[idx] = tau_a
    return ThresholdResult(tau=tau, statistic=best, active=active)
