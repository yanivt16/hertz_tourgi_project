"""FUSE on DICES (Aroyo et al., 2023, arXiv:2306.11247): human raters as verifiers.

DICES-350 has 350 adversarial conversations, each rated by all 123 raters for whether the
final chatbot response is unsafe, plus an expert gold label (`safety_gold`, 175 Yes / 175 No).
That is FUSE's setting with humans in place of reward models: an (items x verifiers) score
matrix and a hidden binary label. There is no Best-of-N here; each conversation is its own
item, so FUSE runs once on the whole matrix (batched mode) and is scored as a binary
classifier against the gold label.

DICES-990 has no gold label (only `degree_of_harm` on 306 items, which on DICES-350 does not
track `safety_gold`), so only DICES-350 can be evaluated.

Verifier views of the same ratings:

* `overall`  one verifier per rater, Q_overall encoded Yes = 1, Unsure = 0, No = -1.
* `granular` one verifier per rater, the number of granular Q2-Q6 reasons marked Yes
             (Unsure counts half), so a rater who flags more reasons scores higher.
* `groups`   one verifier per demographic group, the share of the group's raters who said
             Yes (Unsure counts half). Few continuous verifiers, closest to the paper's setup.
"""

from __future__ import annotations

import math
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .ensemble import FUSE
from .preprocess import minmax_to_unit_interval

URL = "https://raw.githubusercontent.com/google-research-datasets/dices-dataset/main/{v}/diverse_safety_adversarial_dialog_{v}.csv"
CACHE = Path(__file__).resolve().parents[2] / ".cache" / "dices"

# Raters the DICES authors drop for failing their quality checks (dataset READMEs).
REMOVED_RATERS = {
    "350": [
        "297514565398139", "297515609163939", "297515750682315", "297515617432733", "297541515566649",
        "297541515769980", "297515629971478", "297059995361243", "297541522412126", "297540556928761",
        "297541321453321", "297540562350921", "297540983991638", "297060365288109", "297514543980607",
        "297515729806999", "297541271027233", "296709611112092", "296709543131761",
    ],
}

ENCODE = {"Yes": 1.0, "Unsure": 0.0, "No": -1.0}


@dataclass
class DicesMatrix:
    name: str
    V: np.ndarray  # (items, verifiers); NaN where a rater did not rate an item
    verifiers: list[str]
    items: np.ndarray
    gold: np.ndarray | None  # True = unsafe
    majority: np.ndarray  # per-item share of raters answering Yes to Q_overall (Unsure = 1/2)
    rescale: bool  # whether FUSE should min-max each column (False when already in [-1, 1])

    def summary(self) -> str:
        missing = np.isnan(self.V).mean()
        prev = "n/a" if self.gold is None else f"{self.gold.mean():.2f}"
        return (
            f"{self.name}: {self.V.shape[0]} items, {self.V.shape[1]} verifiers, "
            f"{100 * missing:.0f}% missing, gold prevalence {prev}"
        )


def load(variant: str = "350", cache_dir: Path = CACHE):
    import pandas as pd

    path = Path(cache_dir) / f"dices_{variant}.csv"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL.format(v=variant), path)
    # The conversation text is never needed.
    df = pd.read_csv(path, usecols=lambda c: c not in ("context", "response"), dtype={"rater_id": str})
    df.attrs["variant"] = variant
    return df


def _granular_columns(df) -> list[str]:
    return [
        c for c in df.columns
        if c[:3] in ("Q2_", "Q3_", "Q4_", "Q5_", "Q6_") and not c.endswith("_overall")
    ]


def build(df, view: str = "overall", drop_removed: bool = False, groups=("rater_race", "rater_gender")) -> DicesMatrix:
    variant = df.attrs.get("variant", "350")
    if drop_removed:
        df = df[~df.rater_id.isin(REMOVED_RATERS.get(variant, []))]
    df = df.copy()
    df["_yes"] = df.Q_overall.map({"Yes": 1.0, "Unsure": 0.5, "No": 0.0})
    items = np.sort(df.item_id.unique())
    majority = df.groupby("item_id")["_yes"].mean().reindex(items).to_numpy()

    if view == "overall":
        df["_s"] = df.Q_overall.map(ENCODE)
        wide = df.pivot(index="item_id", columns="rater_id", values="_s")
        rescale = False
    elif view == "granular":
        cols = _granular_columns(df)
        df["_s"] = sum(df[c].map({"Yes": 1.0, "Unsure": 0.5, "No": 0.0}).fillna(0.0) for c in cols)
        wide = df.pivot(index="item_id", columns="rater_id", values="_s")
        rescale = True
    elif view == "groups":
        key = df[list(groups)].astype(str).agg("/".join, axis=1)
        wide = df.assign(_g=key).groupby(["item_id", "_g"])["_yes"].mean().unstack()
        rescale = True
    else:
        raise ValueError(view)
    wide = wide.reindex(items)

    gold = None
    if "safety_gold" in df.columns:
        g = df.groupby("item_id").safety_gold.first().reindex(items)
        if g.notna().all():
            gold = (g == "Yes").to_numpy()
    suffix = "-filtered" if drop_removed else ""
    return DicesMatrix(
        name=f"dices{variant}-{view}{suffix}",
        V=wide.to_numpy(dtype=float),
        verifiers=[str(c) for c in wide.columns],
        items=items,
        gold=gold,
        majority=majority,
        rescale=rescale,
    )


# ---- label-free aggregators --------------------------------------------------------------


def dawid_skene(V: np.ndarray, n_iter: int = 200, tol: float = 1e-8) -> np.ndarray:
    """Binary one-coin-per-class Dawid & Skene (1979) EM that skips missing ratings.
    A rating is positive when it is above the rater's own midpoint (Yes for `overall`)."""
    mask = ~np.isnan(V)
    Vs = minmax_to_unit_interval(np.where(mask, V, np.nanmin(V, axis=0)))
    B = np.where(mask, (Vs > 0).astype(float), 0.0)
    M = mask.astype(float)
    q = (B.sum(1) / np.maximum(M.sum(1), 1)).clip(1e-3, 1 - 1e-3)
    for _ in range(n_iter):
        prior = q.mean().clip(1e-3, 1 - 1e-3)
        psi = ((q[:, None] * B).sum(0) + 1) / ((q[:, None] * M).sum(0) + 2)
        eta = (((1 - q)[:, None] * (M - B)).sum(0) + 1) / (((1 - q)[:, None] * M).sum(0) + 2)
        lp = np.log(prior) + B @ np.log(psi) + (M - B) @ np.log(1 - psi)
        ln = np.log(1 - prior) + B @ np.log(1 - eta) + (M - B) @ np.log(eta)
        q_new = 1 / (1 + np.exp(np.clip(ln - lp, -500, 500)))
        done = np.max(np.abs(q_new - q)) < tol
        q = q_new
        if done:
            break
    return q


# ---- evaluation ------------------------------------------------------------------------


def _metrics(score: np.ndarray, pred: np.ndarray | None, y: np.ndarray, ref: np.ndarray | None = None) -> dict[str, float]:
    out = {"AUC": 100 * roc_auc_score(y, score)}
    if pred is not None:
        pred = np.asarray(pred, dtype=bool)
        out["Accuracy"] = 100 * float((pred == y).mean())
        out["Balanced Acc."] = 100 * float((pred[y].mean() + (~pred[~y]).mean()) / 2)
        out["Predicted prevalence"] = 100 * float(pred.mean())
        if ref is not None:
            out["vs MV (95% CI)"] = paired_bootstrap(pred, ref, y)
    return out


def paired_bootstrap(pred: np.ndarray, ref: np.ndarray, y: np.ndarray, n: int = 5000, seed: int = 0) -> str:
    """Accuracy of `pred` minus accuracy of `ref` with a paired bootstrap 95% interval."""
    rng = np.random.default_rng(seed)
    hit_p, hit_r = (pred == y).astype(float), (ref == y).astype(float)
    idx = rng.integers(0, len(y), (n, len(y)))
    d = 100 * (hit_p[idx].mean(1) - hit_r[idx].mean(1))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return f"{100 * (hit_p.mean() - hit_r.mean()):+.1f} [{lo:+.1f}, {hi:+.1f}]"


def _prepared(m: DicesMatrix) -> np.ndarray:
    """Missing ratings become the rater's neutral value (0 after mapping to [-1, 1])."""
    Vs = minmax_to_unit_interval(np.where(np.isnan(m.V), np.nanmin(m.V, axis=0), m.V)) if m.rescale else m.V
    return np.where(np.isnan(m.V), 0.0, Vs)


def evaluate(m: DicesMatrix, fuse: FUSE | None = None, label_fracs=(0.05, 0.10), seeds: int = 20) -> dict:
    if m.gold is None:
        raise ValueError(f"{m.name} has no gold label")
    y = m.gold
    X = _prepared(m)
    res: dict[str, dict] = {}

    mv = m.majority > 0.5
    res["Majority vote (Q_overall)"] = _metrics(m.majority, mv, y)
    naive = X.mean(axis=1)
    res["Naive ensemble"] = _metrics(naive, None, y)
    q = dawid_skene(m.V)
    res["Dawid-Skene"] = _metrics(q, q > 0.5, y, mv)

    fuse = fuse or FUSE()
    info = {}
    for predictor in ("logistic", "posterior"):
        model = FUSE(**{**asdict(fuse), "predictor": predictor})
        r = model.fit_scores(X, rescale=False)
        res[f"FUSE ({predictor})"] = _metrics(r.scores, r.scores > 0.5, y, mv)
        if predictor == "logistic":
            info = {
                "fallback": r.fallback,
                "n_verifiers": int(X.shape[1]),
                "n_kept": int(len(r.kept)),
                "estimated_prevalence": None if r.estimate is None else float(r.estimate.prevalence),
                "kept": [m.verifiers[i] for i in r.kept] if r.fallback is None and not fuse.cluster_threshold else None,
                "tau": None if r.tau is None else [float(t) for t in r.tau],
            }

    # Oracle: the single verifier with the best AUC against the gold label.
    aucs = np.array([roc_auc_score(y, X[:, j]) if X[:, j].std() > 0 else 0.5 for j in range(X.shape[1])])
    best = int(np.argmax(aucs))
    res["Oracle best verifier"] = _metrics(X[:, best], X[:, best] > 0, y, mv)
    # Ceiling for anything that thresholds the crowd's Yes-share: the threshold tuned on gold.
    ths = np.unique(m.majority)
    t = ths[int(np.argmax([((m.majority > t) == y).mean() for t in ths]))]
    res["Crowd share, gold-tuned threshold"] = _metrics(m.majority, m.majority > t, y, mv)
    info["crowd_oracle_threshold"] = float(t)
    info["best_verifier"] = m.verifiers[best]
    info["verifier_auc_quartiles"] = [float(v) for v in np.percentile(100 * aucs, [0, 25, 50, 75, 100])]

    for frac in label_fracs:
        n_lab = max(2, math.ceil(frac * len(y)))
        rows = []
        for seed in range(seeds):
            rng = np.random.default_rng(seed)
            while True:  # need both classes in the labelled split
                lab = rng.choice(len(y), n_lab, replace=False)
                if 0 < y[lab].sum() < n_lab:
                    break
            test = np.setdiff1d(np.arange(len(y)), lab)
            clf = LogisticRegression(max_iter=2000).fit(X[lab], y[lab])
            p = clf.predict_proba(X[test])[:, 1]
            rows.append(_metrics(p, p > 0.5, y[test]))
        res[f"Logistic regression ({int(100 * frac)}% labels)"] = {
            k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "vs MV (95% CI)"
        }
    return {"setting": m.name, "summary": m.summary(), "methods": res, "fuse_info": info}


def format_table(result: dict) -> str:
    cols = ["Accuracy", "Balanced Acc.", "AUC", "Predicted prevalence", "vs MV (95% CI)"]
    lines = ["| Method | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]
    for name, met in result["methods"].items():
        cells = ["" if c not in met else met[c] if isinstance(met[c], str) else f"{met[c]:.1f}" for c in cols]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
