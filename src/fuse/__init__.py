"""FUSE: Fully Unsupervised Score Ensembling (Lee, Ma, Zhao et al., arXiv:2604.18547)."""

from .ensemble import FUSE, FUSEResult, fuse_batched
from .metrics import balanced_accuracy, selection_accuracy
from .moments import MoMEstimate, mom_estimate, moment_tensors
from .posterior import averaged_triplet_posterior, triplet_posterior
from .tci import binarize, minimize_tci, tci_statistic

__all__ = [
    "FUSE",
    "FUSEResult",
    "MoMEstimate",
    "averaged_triplet_posterior",
    "balanced_accuracy",
    "binarize",
    "fuse_batched",
    "minimize_tci",
    "mom_estimate",
    "moment_tensors",
    "selection_accuracy",
    "tci_statistic",
    "triplet_posterior",
]
