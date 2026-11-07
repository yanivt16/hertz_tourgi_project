"""Choose between the FUSE variants without labels.

Clustering helps when a correlated block shares a nuisance factor and hurts when the block
is correlated because its members are right. Under the model FUSE fits, verifiers interact
only through the latent label, so the correlation matrix is rank one off the diagonal. That
gives a label-free test for a candidate block: fit the rank-one factor u on the
cross-cluster entries only, then ask how much of the within-cluster correlation u explains,

    label share R = sum_{j<k same cluster} u_j u_k / sum_{j<k same cluster} C_jk.

R near 1: the block agrees because of the label (keep its members as separate votes).
R near 0: the block agrees for some other reason (collapse it). The rule picks the most
aggressive threshold whose blocks are still nuisance-dominated (R < 0.5), else no clustering.

The majority-vote (MV) column is added when it fits the same rank-one model at least as well
as the median verifier (relative residual of its row of C).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .clustering import cluster_verifiers
from .moments import rank_one_offdiagonal
from .preprocess import minmax_to_unit_interval

THRESHOLDS = (0.3, 0.5, 0.7)
MAX_LABEL_SHARE = 0.5


def correlation(V: np.ndarray) -> np.ndarray:
    """Pearson correlation of columns; constant columns correlate 0 with everything."""
    X = V - V.mean(axis=0)
    sd = np.sqrt((X**2).mean(axis=0))
    nz = sd > 0
    C = np.zeros((V.shape[1], V.shape[1]))
    C[np.ix_(nz, nz)] = (X[:, nz].T @ X[:, nz]) / len(V) / np.outer(sd[nz], sd[nz])
    np.fill_diagonal(C, 1.0)
    return C


def row_residuals(C: np.ndarray) -> np.ndarray:
    """Relative residual of each row of C under the rank-one off-diagonal fit."""
    u = rank_one_offdiagonal(C)
    R = C - np.outer(u, u)
    np.fill_diagonal(R, 0.0)
    off = C - np.diag(np.diag(C))
    norm = np.linalg.norm(off, axis=1)
    return np.where(norm > 0, np.linalg.norm(R, axis=1) / np.maximum(norm, 1e-12), np.inf)


def label_share(V: np.ndarray, threshold: float) -> float | None:
    """Fraction of within-cluster correlation explained by the latent label (None: no merges)."""
    clusters = cluster_verifiers(V, threshold)
    m = V.shape[1]
    lab = np.empty(m, dtype=int)
    for k, g in enumerate(clusters.groups):
        lab[g] = k
    same = lab[:, None] == lab[None, :]
    within = same & ~np.eye(m, dtype=bool)
    if not within.any():
        return None
    C = correlation(V) * np.outer(clusters.signs, clusters.signs)
    u = rank_one_offdiagonal(C, n_iter=2000, observed=~same)
    observed_sum = C[within].sum()
    if observed_sum <= 0:
        return None
    return float(np.outer(u, u)[within].sum() / observed_sum)


@dataclass
class Selection:
    cluster_threshold: float | None
    majority_vote: bool
    label_shares: dict[float, float | None]
    mv_pass_rate: float | None
    info: dict = field(default_factory=dict)

    @property
    def variant(self) -> str:
        """Name as used by scripts/run_variants.py ('base', 'mv', 'cluster@0.5+mv', ...)."""
        name = "base" if self.cluster_threshold is None else f"cluster@{self.cluster_threshold}"
        if self.majority_vote:
            name = "mv" if self.cluster_threshold is None else name + "+mv"
        return name


def select_variant(
    matrices: list[np.ndarray],
    mv_columns: list[np.ndarray] | None = None,
    thresholds: tuple[float, ...] = THRESHOLDS,
    max_label_share: float = MAX_LABEL_SHARE,
) -> Selection:
    """Pick the FUSE variant for one setting from its score matrices alone.

    `matrices` are the imputed per-query matrices for query mode, or a single stacked matrix
    of per-query-rescaled scores for batched mode. `mv_columns` holds the matching
    majority-vote columns, or None when answers are unavailable.
    """
    mats = [minmax_to_unit_interval(np.asarray(V, dtype=float)) for V in matrices]

    mv_rate = None
    use_mv = False
    if mv_columns is not None:
        passes = []
        for V, mv in zip(mats, mv_columns):
            W = np.column_stack([V, minmax_to_unit_interval(np.asarray(mv, dtype=float)[:, None])])
            r = row_residuals(correlation(W))
            passes.append(bool(np.isfinite(r[-1]) and r[-1] <= np.median(r[:-1])))
        mv_rate = float(np.mean(passes))
        use_mv = mv_rate >= 0.5
        if use_mv:
            mats = [
                np.column_stack([V, minmax_to_unit_interval(np.asarray(mv, dtype=float)[:, None])])
                for V, mv in zip(mats, mv_columns)
            ]

    shares: dict[float, float | None] = {}
    for t in thresholds:
        vals = [s for s in (label_share(V, t) for V in mats) if s is not None]
        shares[t] = float(np.median(vals)) if vals else None
    eligible = [t for t in sorted(thresholds) if shares[t] is not None and shares[t] < max_label_share]
    return Selection(
        cluster_threshold=eligible[0] if eligible else None,
        majority_vote=use_mv,
        label_shares=shares,
        mv_pass_rate=mv_rate,
    )
