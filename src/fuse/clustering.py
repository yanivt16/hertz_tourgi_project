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
