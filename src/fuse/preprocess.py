from __future__ import annotations

import numpy as np


def minmax_to_unit_interval(V: np.ndarray) -> np.ndarray:
    """Map each column to [-1, 1] via x -> 2 (x - min) / (max - min) - 1 (paper footnote 5).

    A constant column carries no information for ranking and maps to 0.
    """
    V = np.asarray(V, dtype=float)
    lo = V.min(axis=0)
    span = V.max(axis=0) - lo
    out = np.zeros_like(V)
    nz = span > 0
    out[:, nz] = 2.0 * (V[:, nz] - lo[nz]) / span[nz] - 1.0
    return out


def impute_missing(V: np.ndarray, value: float = 0.0) -> np.ndarray:
    """The paper imputes missing rubric scores as 0, i.e. 'fully incorrect' (Appendix E.4/E.5)."""
    V = np.array(V, dtype=float)
    V[np.isnan(V)] = value
    return V
