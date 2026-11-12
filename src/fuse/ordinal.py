"""Dawid-Skene for scores: each rater's scores are cut into K quantile bins and the bins are
treated as categorical answers with a full per-rater, per-class distribution (Dawid & Skene
1979, binary truth, K answer levels). This is the natural Dawid-Skene baseline when raters
give continuous or ordinal scores instead of yes/no. Missing entries (NaN) are skipped."""

from __future__ import annotations

import numpy as np


def quantile_bins(V: np.ndarray, k: int = 5) -> np.ndarray:
    """Per column, map observed scores to bin indices 0..k-1 (fewer if ties); NaN -> -1."""
    out = np.full(V.shape, -1, dtype=int)
    for j in range(V.shape[1]):
        ok = ~np.isnan(V[:, j])
        x = V[ok, j]
        edges = np.unique(np.quantile(x, np.linspace(0, 1, k + 1)[1:-1]))
        out[ok, j] = np.searchsorted(edges, x, side="right")
    return out


def dawid_skene_ordinal(V: np.ndarray, k: int = 5, n_iter: int = 300, tol: float = 1e-8) -> np.ndarray:
    """P(y = 1 | scores). Higher scores are assumed to lean positive only for the
    initialisation (mean of per-rater bin ranks); EM learns each rater's bin distributions."""
    L = quantile_bins(V, k)
    n, m = L.shape
    K = L.max() + 1
    onehot = np.zeros((n, m, K))
    ii, jj = np.nonzero(L >= 0)
    onehot[ii, jj, L[ii, jj]] = 1.0
    obs = (L >= 0).astype(float)
    rank = np.where(L >= 0, L / np.maximum(K - 1, 1), np.nan)
    with np.errstate(invalid="ignore"):
        r = np.nanmean(rank, 1)
    q = np.clip(np.nan_to_num(r, nan=0.5), 1e-3, 1 - 1e-3)
    q = (q > np.median(q)).astype(float) * 0.8 + 0.1
    for _ in range(n_iter):
        prior = np.clip(q.mean(), 1e-3, 1 - 1e-3)
        th1 = (np.einsum("i,ijk->jk", q, onehot) + 0.5) / (np.einsum("i,ij->j", q, obs)[:, None] + 0.5 * K)
        th0 = (np.einsum("i,ijk->jk", 1 - q, onehot) + 0.5) / (np.einsum("i,ij->j", 1 - q, obs)[:, None] + 0.5 * K)
        l1 = np.log(prior) + np.einsum("ijk,jk->i", onehot, np.log(th1))
        l0 = np.log(1 - prior) + np.einsum("ijk,jk->i", onehot, np.log(th0))
        q_new = 1 / (1 + np.exp(np.clip(l0 - l1, -500, 500)))
        done = np.max(np.abs(q_new - q)) < tol
        q = q_new
        if done:
            break
    return q
