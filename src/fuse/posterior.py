"""Posterior label probabilities from estimated verifier quality (paper Appendix C).

For binarised verifiers and any triplet (j1, j2, j3), under TCI (Proposition C.1):

    P(y = y | v_j1, v_j2, v_j3) ∝ (1 + b y) prod_l [1 - y v_jl + v_jl ((1 + y) psi_jl - (1 - y) eta_jl)]

The default estimate of the full posterior averages these over all triplets (eq. 9).
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from .moments import MoMEstimate


def _factors(V: np.ndarray, psi: np.ndarray, eta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-verifier likelihood factors for y = +1 and y = -1 (each up to the common 1/2)."""
    f_pos = 1.0 - V + 2.0 * V * psi[None, :]
    f_neg = 1.0 + V - 2.0 * V * eta[None, :]
    return np.clip(f_pos, 1e-12, None), np.clip(f_neg, 1e-12, None)


def triplet_posterior(V: np.ndarray, est: MoMEstimate, triplet: tuple[int, int, int]) -> np.ndarray:
    f_pos, f_neg = _factors(V, est.psi, est.eta)
    t = list(triplet)
    a = (1.0 + est.b) * np.prod(f_pos[:, t], axis=1)
    c = (1.0 - est.b) * np.prod(f_neg[:, t], axis=1)
    return a / (a + c)


def averaged_triplet_posterior(
    V: np.ndarray,
    est: MoMEstimate,
    max_triplets: int = 20000,
    chunk: int = 512,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """p-hat(r_i) of equation (9): the mean of triplet posteriors over all m-choose-3 triplets.

    When m-choose-3 exceeds max_triplets a uniform random subset is used, which keeps the
    estimator unbiased for the full average.
    """
    n, m = V.shape
    if m < 3:
        raise ValueError("posterior averaging needs at least three verifiers")
    f_pos, f_neg = _factors(V, est.psi, est.eta)
    log_pos, log_neg = np.log(f_pos), np.log(f_neg)
    triplets = np.array(list(combinations(range(m), 3)))
    if len(triplets) > max_triplets:
        rng = rng or np.random.default_rng(0)
        triplets = triplets[rng.choice(len(triplets), max_triplets, replace=False)]

    log_prior = np.log1p(est.b) - np.log1p(-est.b)
    total = np.zeros(n)
    for start in range(0, len(triplets), chunk):
        tr = triplets[start : start + chunk]
        logit = log_prior + log_pos[:, tr].sum(axis=2) - log_neg[:, tr].sum(axis=2)
        total += (1.0 / (1.0 + np.exp(-logit))).sum(axis=1)
    return total / len(triplets)
