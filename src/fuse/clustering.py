"""Collapse blocks of mutually correlated verifiers before the moment estimate.

The method of moments assumes verifiers are conditionally independent given correctness.
A block of verifiers that share a nuisance factor (response length, a common base model)
violates that, and the rank-one fit of the covariance reads the shared factor as the label.
Averaging each block into one meta-verifier leaves a set that is closer to conditionally
independent, and each block then gets one vote in the moment estimate instead of many.

Blocks are found without labels: average-linkage clustering on 1 - |Spearman correlation|.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


@dataclass
class VerifierClusters:
    groups: list[np.ndarray]  # column indices of each cluster
    signs: np.ndarray  # (m,) +-1 orientation of each column inside its cluster

    def collapse(self, V: np.ndarray) -> np.ndarray:
        """One column per cluster: the mean of its sign-aligned members."""
        Vo = V * self.signs[None, :]
        return np.column_stack([Vo[:, g].mean(axis=1) for g in self.groups])


def spearman_matrix(V: np.ndarray) -> np.ndarray:
    """Spearman correlation between columns; constant columns correlate 0 with everything."""
    R = np.argsort(np.argsort(V, axis=0, kind="stable"), axis=0).astype(float)
    R -= R.mean(axis=0)
    norm = np.sqrt((R**2).sum(axis=0))
    nz = norm > 0
    C = np.zeros((V.shape[1], V.shape[1]))
    C[np.ix_(nz, nz)] = (R[:, nz].T @ R[:, nz]) / np.outer(norm[nz], norm[nz])
    np.fill_diagonal(C, 1.0)
    return C


def cluster_verifiers(V: np.ndarray, threshold: float, min_clusters: int = 3) -> VerifierClusters:
    """Merge verifiers whose average |Spearman correlation| exceeds `threshold`.

    If that leaves fewer than `min_clusters` clusters (FUSE needs triplets), the dendrogram
    is cut at exactly `min_clusters` instead.
    """
    m = V.shape[1]
    C = spearman_matrix(V)
    if m < 2:
        return VerifierClusters([np.arange(m)], np.ones(m))
    D = np.clip(1.0 - np.abs(C), 0.0, None)
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(D, checks=False), method="average")
    labels = fcluster(Z, t=1.0 - threshold, criterion="distance")
    if labels.max() < min(min_clusters, m):
        labels = fcluster(Z, t=min(min_clusters, m), criterion="maxclust")

    groups = [np.flatnonzero(labels == k) for k in np.unique(labels)]
    signs = np.ones(m)
    for g in groups:
        # Orient members toward the member most correlated with the rest of the cluster.
        anchor = g[np.argmax(np.abs(C[np.ix_(g, g)]).sum(axis=1))]
        signs[g] = np.where(C[anchor, g] < 0, -1.0, 1.0)
    return VerifierClusters(groups, signs)


# --- Residual (conditional) correlation: separate shared errors from shared signal -------------
# Total correlation cannot tell raters who agree because they are accurate from raters who agree
# because they share errors (cluster_verifiers above merges both). Under conditional independence
# the correlation matrix is rank one off the diagonal, C_ij = u_i u_j, with u_i the rater's
# correlation with the latent label. A least-trimmed-squares rank-one fit lets a minority of pairs
# that share errors be trimmed instead of pulling the fit; the partial correlation left after it,
# (C_ij - u_i u_j) / sqrt((1 - u_i^2)(1 - u_j^2)), is high only for pairs that share errors.

def trimmed_rank_one(C: np.ndarray, keep: float = 0.7, iters: int = 40) -> np.ndarray:
    """u minimising the squared residual of C_ij - u_i u_j over the `keep` share of off-diagonal pairs
    that fit best (least trimmed squares), from several starts; the best trimmed loss wins.
    A minority of pairs that share errors is trimmed instead of pulling the fit."""
    m = len(C)
    off = ~np.eye(m, dtype=bool)
    # starts: diagonal-completed leading eigenvector, and each rater's own correlation row
    D = C.copy()
    for _ in range(20):
        w, v = np.linalg.eigh(D); u = v[:, -1] * np.sqrt(max(w[-1], 0))
        np.fill_diagonal(D, u ** 2)
    starts = [u * np.sign(u.sum() or 1)]
    for i in range(m):
        u0 = np.clip(C[i], 0.01, 0.99).copy(); u0[i] = np.median(u0[off[i]])
        starts.append(u0)
    best, best_loss = None, np.inf
    for u in starts:
        u = np.clip(u, 1e-3, 0.999).astype(float)
        for _ in range(iters):
            Rz = np.abs(C - np.outer(u, u))
            cut = np.quantile(Rz[off], keep)
            W = off & (Rz <= cut)
            num = (W * C) @ u
            den = W @ (u ** 2)
            u = np.clip(np.where(den > 0, num / np.maximum(den, 1e-12), u), 1e-3, 0.999)
        Rz = (C - np.outer(u, u))[off]
        loss = np.sort(Rz ** 2)[: int(keep * Rz.size)].sum()
        if loss < best_loss:
            best, best_loss = u, loss
    return best


def residual_corr(V: np.ndarray, keep: float = 0.7):
    """(P, u, C): partial correlation given the fitted latent, loadings u, Spearman matrix C."""
    C = spearman_matrix(V)
    u = trimmed_rank_one(C, keep)
    P = (C - np.outer(u, u)) / np.sqrt(np.outer(1 - u ** 2, 1 - u ** 2))
    np.fill_diagonal(P, 1.0)
    return P, u, C


def residual_groups(V: np.ndarray, threshold: float = 0.4, keep: float = 0.7):
    """(groups, P, u): average-linkage clusters of raters whose residual (partial) correlation
    exceeds `threshold` (0.4 was safe on every panel in docs/residual_blocks.md; 0.3 catches weaker
    blocks but can merge the only two experts in a panel of near-chance raters)."""
    P, u, C = residual_corr(V, keep)
    m = len(P)
    D = np.clip(1 - P, 0, None); np.fill_diagonal(D, 0)
    lab = fcluster(linkage(squareform(D, checks=False), "average"), t=1 - threshold, criterion="distance")
    groups = [np.flatnonzero(lab == k) for k in np.unique(lab)]
    return groups, P, u


def collapse_groups(V: np.ndarray, groups: list[np.ndarray]) -> np.ndarray:
    """One column per group: the mean of its members' z-scores (a single member is kept as it is)."""
    sd = V.std(0); Z = (V - V.mean(0)) / np.where(sd > 0, sd, 1)
    return np.column_stack([V[:, g[0]] if len(g) == 1 else Z[:, g].mean(1) for g in groups])


# --- Baseline: Jaffe, Fetaya, Nadler, Jiang & Kluger (AISTATS 2016), arXiv:1510.05830 ----------
# Raters fall into groups; each group depends on the label only through its own latent variable.
# Off the diagonal the covariance is then rank one in two pieces: r_ij = v_on_i v_on_j within a
# group and v_off_i v_off_j across groups (their Lemma 1). Their Algorithm 1: a score matrix of
# 2x2 minors, s_ij = sum_{k,l not in {i,j}} |r_ij r_kl - r_il r_kj| (Eq. 15), spectral clustering
# on its Laplacian for every K in 2..m-1, and the K whose two rank-one fits (Eq. 14) leave the
# smallest residual. Written from the paper's main text; their code was not used.

def jaffe_score_matrix(R: np.ndarray) -> np.ndarray:
    """Eq. 15: large s_ij means raters i and j are conditionally dependent."""
    m = len(R)
    S = np.zeros((m, m))
    idx = np.arange(m)
    for i in range(m):
        for j in range(i + 1, m):
            keep = (idx != i) & (idx != j)
            M = np.abs(R[i, j] * R[np.ix_(keep, keep)] - np.outer(R[keep, j], R[i, keep]))
            S[i, j] = S[j, i] = M.sum() - np.trace(M)  # k != l: the diagonal is not rank one
    return S


def _masked_rank_one(R: np.ndarray, W: np.ndarray, iters: int = 300) -> np.ndarray:
    """v minimising sum over W of (v_i v_j - r_ij)^2 (damped alternating least squares)."""
    w, vec = np.linalg.eigh(np.where(W, R, 0.0))
    v = vec[:, -1] * np.sqrt(max(w[-1], 1e-12))
    Wf = W.astype(float)
    for _ in range(iters):
        den = Wf @ (v ** 2)
        new = np.where(den > 1e-12, ((Wf * R) @ v) / np.maximum(den, 1e-12), v)
        v = 0.5 * v + 0.5 * new
    return v


def jaffe_residual(R: np.ndarray, labels: np.ndarray) -> float:
    """Eq. 14 for a given assignment: on-group and off-group pairs each get a rank-one fit."""
    same = labels[:, None] == labels[None, :]
    off = ~np.eye(len(R), dtype=bool)
    total = 0.0
    for W in (same & off, ~same):
        if W.any():
            v = _masked_rank_one(R, W)
            total += float((((np.outer(v, v) - R) ** 2)[W]).sum())
    return total


def jaffe_groups(R: np.ndarray, seed: int = 0):
    """Algorithm 1 of Jaffe et al. (2016) on a rater covariance (or correlation) matrix R.
    Returns (groups, K, S, residual by K)."""
    from sklearn.cluster import KMeans

    m = len(R)
    S = jaffe_score_matrix(R)
    d = S.sum(1)
    dinv = 1 / np.sqrt(np.where(d > 0, d, 1))
    L = np.eye(m) - dinv[:, None] * S * dinv[None, :]
    _, vec = np.linalg.eigh(L)  # one eigendecomposition, reused for every K
    best, res = None, {}
    for K in range(2, m):
        Y = vec[:, :K] / np.maximum(np.linalg.norm(vec[:, :K], axis=1, keepdims=True), 1e-12)
        lab = KMeans(K, n_init=10, random_state=seed).fit_predict(Y)
        res[K] = jaffe_residual(R, lab)
        if best is None or res[K] < res[best[0]]:
            best = (K, lab)
    if best is None:  # m < 3: nothing to cluster
        return [np.array([i]) for i in range(m)], m, S, res
    K, lab = best
    return [np.flatnonzero(lab == k) for k in np.unique(lab)], K, S, res
