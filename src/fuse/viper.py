"""FUSE on VIPER (FDA DIDSR mammography reader study): radiologists as verifiers.

VIPER (github.com/DIDSR/viperData): 43 radiologists read DMIST screening mammograms, once on
full-field digital (FFDM) and once on screen-film (SFM). Each read has a binary recall
decision and a 0-201 suspicion score (0-100 = no recall, 101-201 = recall, higher = more
suspicious). The gold label is cancer status at 455-day follow-up, independent of the readers.

Per modality the matrix is about 715 cases x 43 readers with 67% missing. The images are not
public (DMIST/ACRIN 6652, available only on request from ECOG-ACRIN).

Missing reads are NOT at random. The study enriched cancers and assigned them to more readers
and to specific reader groups, so which radiologist read a case predicts cancer with AUC
about 0.94 without looking at a single score (`assignment_leak`). Any method that turns a
missing read into a value (zero-imputation, as in the paper) can pick this up. Evaluate on
fully dense blocks (`dense_blocks`) or with a method that skips missing reads.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from .dices import dawid_skene
from .ensemble import FUSE
from .preprocess import minmax_to_unit_interval

URL = "https://raw.githubusercontent.com/DIDSR/viperData/master/data/{name}.rda"
CACHE = Path(__file__).resolve().parents[2] / ".cache" / "viper"


def load(cache_dir: Path = CACHE):
    """Return (reads, gold): one row per (case, reader, modality) read, and gold cancer
    status per case (pandas Series indexed by caseID as str). Needs `pip install rdata`."""
    import rdata

    frames = {}
    for name, key in [("viperObs455", "viperObs455"), ("casesViper", "casesViper")]:
        path = Path(cache_dir) / f"{name}.rda"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(URL.format(name=name), path)
        frames[key] = rdata.read_rda(str(path))[key]
    reads, cases = frames["viperObs455"], frames["casesViper"]
    reads = reads.assign(caseID=reads.caseID.astype(str), readerID=reads.readerID.astype(str),
                         modalityID=reads.modalityID.astype(str))
    reads = reads.drop_duplicates(["caseID", "readerID", "modalityID"])
    gold = cases.drop_duplicates("caseID").assign(caseID=lambda d: d.caseID.astype(str))
    return reads, gold.set_index("caseID")["cancerStatus455"].astype(int)


def build(reads, gold):
    """(S, B, y): suspicion scores and recall decisions (1 = recall), NaN where unread."""
    items, ws = sorted(reads.caseID.unique()), sorted(reads.readerID.unique())
    ii = {x: k for k, x in enumerate(items)}
    jj = {x: k for k, x in enumerate(ws)}
    r, c = reads.caseID.map(ii).to_numpy(int), reads.readerID.map(jj).to_numpy(int)
    S = np.full((len(items), len(ws)), np.nan)
    S[r, c] = reads.score.astype(float).to_numpy()
    B = np.full_like(S, np.nan)
    B[r, c] = (reads.recall.astype(str) == "1").astype(float).to_numpy()
    return S, B, gold.reindex(items).to_numpy()


def dense_blocks(reads, min_readers: int = 5, min_cases_per_reader: int = 380) -> dict[int, tuple[list, list]]:
    """Fully dense (readers x cases) blocks. Greedy: start from the readers with the most
    cases and drop, one at a time, the reader whose removal most enlarges the set of cases
    read by all remaining readers. Returns {n_cases: (readers, cases)}; blocks are nested."""
    import pandas as pd

    P = pd.crosstab(reads.caseID, reads.readerID) > 0
    Rs = list(P.columns[P.sum(0) >= min_cases_per_reader])
    out = {}
    while len(Rs) >= min_readers:
        cs = P.index[P[Rs].all(1)]
        if len(cs):
            out[len(cs)] = (list(Rs), list(cs))
        Rs.remove(max(Rs, key=lambda r: P.index[P[[x for x in Rs if x != r]].all(1)].size))
    return out


def assignment_leak(reads, gold, folds: int = 5) -> float:
    """Cross-validated AUC of predicting cancer from WHICH readers read the case (no scores)."""
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict

    P = pd.crosstab(reads.caseID, reads.readerID).gt(0).astype(float)
    y = gold.reindex(P.index).to_numpy()
    p = cross_val_predict(LogisticRegression(max_iter=2000), P.to_numpy(), y, cv=folds, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def _auc_diff(a, b, y, n=2000, seed=0) -> str:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), (n, len(y)))
    d = [roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]) for i in idx if 0 < y[i].sum() < len(i)]
    lo, hi = np.percentile(d, [2.5, 97.5])
    return f"{100 * (roc_auc_score(y, a) - roc_auc_score(y, b)):+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"


def evaluate(S, B, y) -> list[dict]:
    """AUC against cancer status for MV, mean score, Dawid-Skene and FUSE. FUSE gets each
    reader's column mapped to [-1, 1] with missing reads set to 0, as in the DICES runs."""
    mask = ~np.isnan(S)
    share = np.nanmean(B, 1)
    rows = []

    def add(name, s, pred=None):
        r = {"Method": name, "AUC": 100 * roc_auc_score(y, s)}
        if pred is not None:
            pred = np.asarray(pred, bool)
            r["Sens."] = 100 * pred[y == 1].mean()
            r["Spec."] = 100 * (~pred[y == 0]).mean()
        r["AUC vs MV (95% CI)"] = "" if not rows else _auc_diff(s, share, y)
        rows.append(r)

    add("Majority vote (share recalling)", share, share > 0.5)
    Z = (S - np.nanmean(S, 0)) / np.nanstd(S, 0)
    add("Mean suspicion score (per-reader z)", np.nanmean(Z, 1))
    q = dawid_skene(np.where(mask, 2 * B - 1, np.nan))
    add("Dawid-Skene (recall)", q, q > 0.5)
    Bx = np.where(mask, 2 * B - 1, 0.0)
    Sx = np.where(mask, minmax_to_unit_interval(np.where(mask, S, np.nanmin(S, 0))), 0.0)
    for view, X in [("recall", Bx), ("suspicion score", Sx)]:
        for pred in ("logistic", "posterior"):
            r = FUSE(predictor=pred).fit_scores(X, rescale=False)
            add(f"FUSE {pred} ({view})", r.scores, r.scores > 0.5)
    ok = [j for j in range(S.shape[1]) if mask[:, j].sum() >= 20 and 0 < y[mask[:, j]].sum() < mask[:, j].sum()]
    aucs = np.array([roc_auc_score(y[mask[:, j]], S[mask[:, j], j]) for j in ok])
    rows.append({"Method": "Median single reader", "AUC": 100 * np.median(aucs)})
    rows.append({"Method": "Best single reader (picked with labels)", "AUC": 100 * aucs.max()})
    return rows
