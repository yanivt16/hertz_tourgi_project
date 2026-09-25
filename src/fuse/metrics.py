from __future__ import annotations

import numpy as np


def selection_accuracy(scores: np.ndarray, correct: np.ndarray, atol: float = 1e-12) -> float:
    """Accuracy of picking argmax(scores); ties score the fraction of tied responses that
    are correct (paper Appendix E, 'Tie-breaking')."""
    scores = np.asarray(scores, dtype=float)
    correct = np.asarray(correct, dtype=float)
    tied = scores >= scores.max() - atol
    return float(correct[tied].mean())


def balanced_accuracy(scores: np.ndarray, correct: np.ndarray) -> float:
    """Real-valued balanced accuracy of Appendix A.2 for scores already in [-1, 1]:
    psi = E[(1 + v)/2 | y = 1], eta = E[(1 - v)/2 | y = -1]."""
    scores = np.asarray(scores, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    if correct.all() or (~correct).all():
        return float("nan")
    psi = np.mean((1 + scores[correct]) / 2)
    eta = np.mean((1 - scores[~correct]) / 2)
    return float((psi + eta) / 2)
