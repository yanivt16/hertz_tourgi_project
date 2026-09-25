"""Method-of-moments estimation of verifier quality (Jaffe et al., 2015; paper Section 2.1).

Given a score matrix V (N responses x m verifiers, entries in [-1, 1]), the second and
third order covariance tensors have rank-one off-diagonal structure under triplet
conditional independence (Theorem 2.3):

    Sigma_{jk}  = u_j u_k              u = sqrt(1 - b^2) (2 pi - 1)
    T_{jkl}     = w_j w_k w_l          w = (-2 b (1 - b^2))^{1/3} (2 pi - 1)

from which the class imbalance b, sensitivities psi and specificities eta follow.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-6


@dataclass(frozen=True)
class MomentTensors:
    mu: np.ndarray  # (m,)
    sigma: np.ndarray  # (m, m)
    T: np.ndarray  # (m, m, m)


@dataclass(frozen=True)
class MoMEstimate:
    psi: np.ndarray  # sensitivities, (m,)
    eta: np.ndarray  # specificities, (m,)
    b: float  # class imbalance P(y=1) - P(y=-1)
    u: np.ndarray  # rank-one factor of Sigma, (m,)
    mu: np.ndarray  # verifier means, (m,)

    @property
    def balanced_accuracy(self) -> np.ndarray:
        return (self.psi + self.eta) / 2.0

    @property
    def prevalence(self) -> float:
        return (1.0 + self.b) / 2.0


def moment_tensors(V: np.ndarray) -> MomentTensors:
    V = np.asarray(V, dtype=float)
    n, m = V.shape
    mu = V.mean(axis=0)
    X = V - mu
    sigma = X.T @ X / n
    # T_{jkl} = E[x_j x_k x_l], computed as chunked BLAS matmuls so batched mode
    # (tens of thousands of rows) never materialises the full N x m^2 outer product.
    T = np.zeros((m, m * m))
    for start in range(0, n, 4096):
        Xc = X[start : start + 4096]
        T += Xc.T @ (Xc[:, :, None] * Xc[:, None, :]).reshape(len(Xc), m * m)
    return MomentTensors(mu=mu, sigma=sigma, T=T.reshape(m, m, m) / n)


def rank_one_offdiagonal(S: np.ndarray, n_iter: int = 200, tol: float = 1e-10) -> np.ndarray:
    """Fit u such that the off-diagonal entries of S match those of u u^T.

    Uses the iterative diagonal-completion scheme of Parisi et al. (2014): replace the
    diagonal with the current rank-one estimate and take the leading eigenpair again.
    """
    S = np.asarray(S, dtype=float)
    M = S.copy()
    u = np.zeros(S.shape[0])
    for _ in range(n_iter):
        eigvals, eigvecs = np.linalg.eigh(M)
        lam = max(eigvals[-1], 0.0)
        u_new = np.sqrt(lam) * eigvecs[:, -1]
        np.fill_diagonal(M, u_new**2)
        if np.max(np.abs(np.abs(u_new) - np.abs(u))) < tol:
            u = u_new
            break
        u = u_new
    return u


def orient_by_majority(u: np.ndarray) -> np.ndarray:
    """Resolve the sign ambiguity of u using Assumption 2.1 (most verifiers beat random)."""
    return -u if np.sum(u > 0) < np.sum(u < 0) else u


def _distinct_triplet_mask(m: int) -> np.ndarray:
    idx = np.arange(m)
    return (
        (idx[:, None, None] != idx[None, :, None])
        & (idx[:, None, None] != idx[None, None, :])
        & (idx[None, :, None] != idx[None, None, :])
    )


def estimate_class_imbalance(T: np.ndarray, u: np.ndarray) -> float:
    """Recover b from the scale between w and u.

    w = alpha * u with alpha^3 = -2 b (1 - b^2) / (1 - b^2)^{3/2} = -2 b / sqrt(1 - b^2),
    so with c := alpha^3 we get b = -c / sqrt(4 + c^2). c is fit by least squares over
    the entries of T with three distinct indices.
    """
    m = len(u)
    if m < 3:
        return 0.0
    mask = _distinct_triplet_mask(m)
    P = u[:, None, None] * u[None, :, None] * u[None, None, :]
    denom = np.sum((P**2)[mask])
    if denom < 1e-18:
        return 0.0
    c = np.sum((T * P)[mask]) / denom
    b = -c / np.sqrt(4.0 + c**2)
    return float(np.clip(b, -1 + 1e-4, 1 - 1e-4))


def sensitivities_specificities(
    mu: np.ndarray, u: np.ndarray, b: float
) -> tuple[np.ndarray, np.ndarray]:
    """Theorem 2.3 (iii).

    psi = (1 + mu + u sqrt((1 - b) / (1 + b))) / 2
    eta = (1 - mu + u sqrt((1 + b) / (1 - b))) / 2
    """
    psi = 0.5 * (1.0 + mu + u * np.sqrt((1.0 - b) / (1.0 + b)))
    eta = 0.5 * (1.0 - mu + u * np.sqrt((1.0 + b) / (1.0 - b)))
    return np.clip(psi, EPS, 1 - EPS), np.clip(eta, EPS, 1 - EPS)


def mom_estimate(V: np.ndarray, moments: MomentTensors | None = None) -> MoMEstimate:
    """Estimate (psi, eta, b) from a score matrix with zero labels (paper Section 2.1)."""
    if moments is None:
        moments = moment_tensors(V)
    u = orient_by_majority(rank_one_offdiagonal(moments.sigma))
    b = estimate_class_imbalance(moments.T, u)
    psi, eta = sensitivities_specificities(moments.mu, u, b)
    return MoMEstimate(psi=psi, eta=eta, b=b, u=u, mu=moments.mu)
