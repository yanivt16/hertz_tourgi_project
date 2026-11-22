"""Like-for-like evaluation of one rater panel (DICES-350), shared by the panel experiments.

Inputs are per-rater ratings in {-1 (No), 0 (Unsure), 1 (Yes)}. Every method sees the same
raters and, where it can, the same three answers:

* Majority vote: share of Yes with Unsure as 1/2 (and, for the information check only, with
  Unsure as No).
* Dawid–Skene, three answers: Yes / Unsure / No as categories (the fair Dawid–Skene).
* Dawid–Skene, Yes only: Unsure folded into No (what the first panel run used).
* Dawid–Skene, Unsure as Yes.
* FUSE: repo defaults on the -1/0/1 matrix.

Each method's natural rule (score > 0.5) differs in how many conversations it calls unsafe,
and with a balanced gold label whoever calls more wins accuracy. So accuracy is also scored at
two shared cut-offs, where every method calls the SAME number of conversations unsafe:
k = FUSE's estimated prevalence x N, and k = three-answer Dawid–Skene's estimated prevalence x N.
Ties in a score are broken by one fixed random order of the conversations, the same for every
method, so that no method gains from the order of the item ids.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fuse import FUSE, dices  # noqa: E402

METHODS = ["Majority vote", "Dawid–Skene, 3 answers", "Dawid–Skene, Yes only", "Dawid–Skene, Unsure as Yes", "FUSE"]
CUTS = ["natural", "at FUSE est.", "at DS-3 est."]
TIE = np.random.default_rng(350).random(350)  # one fixed tie-break order for the 350 conversations


def top_k(score: np.ndarray, k: int) -> np.ndarray:
    order = np.lexsort((TIE[: len(score)], -score))
    pred = np.zeros(len(score), dtype=bool)
    pred[order[:k]] = True
    return pred


def balanced_acc(pred: np.ndarray, y: np.ndarray) -> float:
    return float((pred[y].mean() + (~pred[~y]).mean()) / 2)


def _rho(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return None
    return float(spearmanr(a, b).statistic)


def evaluate_panel(V: np.ndarray, y: np.ndarray) -> dict:
    """V: (items, raters) in {-1, 0, 1}, no NaN. Returns hits per method and cut, AUCs,
    calls, estimates, and rank correlations of estimated with true rater quality."""
    n = len(y)
    share = ((V + 1) / 2).mean(1)
    share_no = (V > 0).mean(1)
    q3, th1, th0 = dices.dawid_skene_3class(V, return_params=True)
    qy = dices.dawid_skene((V > 0).astype(float))
    qu = dices.dawid_skene((V >= 0).astype(float))
    f = FUSE().fit_scores(V, rescale=False)
    est_f = None if f.estimate is None else float(f.estimate.prevalence)
    est_d = float(q3.mean())
    scores = {"Majority vote": share, "Dawid–Skene, 3 answers": q3, "Dawid–Skene, Yes only": qy,
              "Dawid–Skene, Unsure as Yes": qu, "FUSE": f.scores}
    kf = int(round((est_f if est_f is not None else est_d) * n))
    kd = int(round(est_d * n))
    out = {"n_raters": V.shape[1], "fuse_est_prev": est_f, "ds3_est_prev": est_d, "fuse_fallback": f.fallback,
           "auc_mv_unsure_no": 100 * roc_auc_score(y, share_no), "hits": {}, "auc": {}, "called": {}}
    for m, s in scores.items():
        preds = {"natural": s > 0.5, "at FUSE est.": top_k(s, kf), "at DS-3 est.": top_k(s, kd)}
        out["hits"][m] = {c: (p == y) for c, p in preds.items()}
        out["called"][m] = 100 * float(preds["natural"].mean())
        out["auc"][m] = 100 * float(roc_auc_score(y, s))
    # Does the method find the good raters? True quality is measured on the same vote the
    # method models: FUSE's thresholded vote for the raters it keeps, the Yes vote for DS-3.
    true_yes = np.array([balanced_acc(V[:, j] > 0, y) for j in range(V.shape[1])])
    est_d3 = (th1[:, 2] + th0[:, 0] + th0[:, 1]) / 2
    out["rho_ds3"] = _rho(est_d3, true_yes)
    if f.estimate is not None and f.tau is not None:
        kept = np.asarray(f.kept)
        true_f = np.array([balanced_acc(V[:, j] > f.tau[j], y) for j in kept])
        out["rho_fuse"] = _rho(f.estimate.balanced_accuracy, true_f)
    else:
        out["rho_fuse"] = None
    out["best_rater_acc"] = 100 * float(max(((V[:, j] > 0) == y).mean() for j in range(V.shape[1])))
    return out


def summarize(rows: list[dict], mask: np.ndarray | None = None) -> dict:
    """Mean accuracy per method and cut (on `mask` conversations), AUC and calls."""
    out = {}
    for m in METHODS:
        out[m] = {"auc": float(np.mean([r["auc"][m] for r in rows])), "called": float(np.mean([r["called"][m] for r in rows]))}
        for c in CUTS:
            v = np.array([100 * r["hits"][m][c][mask if mask is not None else slice(None)].mean() for r in rows])
            out[m][c] = float(v.mean())
            out[m][c + " sd"] = float(v.std())
    return out


def paired(rows: list[dict], a: str, b: str, cut: str, mask: np.ndarray | None = None, seed: int = 0) -> dict:
    """a minus b. Across panels: mean difference and share of panels each side wins.
    Across conversations: each conversation's hit rate averaged over the panels, then a paired
    bootstrap over conversations (95% interval)."""
    sl = mask if mask is not None else slice(None)
    d = np.array([100 * (r["hits"][a][cut][sl].mean() - r["hits"][b][cut][sl].mean()) for r in rows])
    ha = np.mean([r["hits"][a][cut][sl] for r in rows], axis=0)
    hb = np.mean([r["hits"][b][cut][sl] for r in rows], axis=0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ha), (5000, len(ha)))
    bs = 100 * (ha[idx].mean(1) - hb[idx].mean(1))
    return {"a": a, "b": b, "cut": cut, "diff": float(d.mean()), "a_wins": 100 * float((d > 0).mean()),
            "tie": 100 * float((d == 0).mean()), "b_wins": 100 * float((d < 0).mean()),
            "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5))}


def paired_auc(rows: list[dict], a: str, b: str) -> dict:
    d = np.array([r["auc"][a] - r["auc"][b] for r in rows])
    return {"a": a, "b": b, "diff": float(d.mean()), "a_wins": 100 * float((d > 0).mean()),
            "p2.5": float(np.percentile(d, 2.5)), "p97.5": float(np.percentile(d, 97.5))}


def rho_summary(rows: list[dict]) -> dict:
    f = [r["rho_fuse"] for r in rows if r["rho_fuse"] is not None]
    d = [r["rho_ds3"] for r in rows if r["rho_ds3"] is not None]
    both = [(r["rho_fuse"], r["rho_ds3"]) for r in rows if r["rho_fuse"] is not None and r["rho_ds3"] is not None]
    return {"fuse": float(np.mean(f)) if f else None, "ds3": float(np.mean(d)) if d else None,
            "fuse_higher_share": 100 * float(np.mean([a > b for a, b in both])) if both else None, "n": len(both)}
