"""FUSE on DICES-990, block by block: conversations grouped so each block shares its raters.

    python scripts/dices990_blocks.py --labels dices990_v4 --k 5

DICES-990 is 58% empty as a whole, but it was collected in batches: groups of conversations
were rated by largely the same raters. Conversations are clustered by the Jaccard overlap of
their rater sets (average linkage); `--k` is the number of blocks, and the default 5 is the
largest split in which every block keeps at least 100 conversations. In each block the
"core" raters are those who rated at least 90% of its conversations, which leaves an almost
complete matrix. Every method in a block, majority vote included, sees the same raters.
Each block is fitted separately; block predictions are then pooled over all 990 conversations.

Scored against an LLM label (DICES-990 has no expert verdict), so accuracies are agreement
with that label.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fuse import FUSE, dices  # noqa: E402

RESULTS = ROOT / "results"
METHODS = ["Majority vote", "Dawid-Skene", "FUSE (logistic)", "FUSE (posterior)"]


def blocks(df: pd.DataFrame, k: int) -> pd.Series:
    """Block id per conversation from average-linkage clustering of rater-set Jaccard distance."""
    A = pd.crosstab(df.item_id, df.rater_id) > 0
    X = A.to_numpy(float)
    inter = X @ X.T
    size = X.sum(1)
    D = 1 - inter / (size[:, None] + size[None, :] - inter)
    np.fill_diagonal(D, 0)
    lab = fcluster(linkage(squareform(D, checks=False), method="average"), k, criterion="maxclust")
    # Number blocks by size, largest first, so ids are stable across runs.
    order = {c: i + 1 for i, c in enumerate(pd.Series(lab).value_counts().index)}
    return pd.Series([order[c] for c in lab], index=A.index, name="block")


def core_raters(df_block: pd.DataFrame, min_share: float) -> list[str]:
    n = df_block.item_id.nunique()
    share = df_block.groupby("rater_id").item_id.nunique() / n
    return share.index[share >= min_share].tolist()


def predictions(m: dices.DicesMatrix) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """(score, label-free prediction) per method for one block."""
    X = dices._prepared(m)
    out = {"Majority vote": (m.majority, m.majority > 0.5)}
    q = dices.dawid_skene(m.V)
    out["Dawid-Skene"] = (q, q > 0.5)
    for predictor in ("logistic", "posterior"):
        r = FUSE(predictor=predictor).fit_scores(X, rescale=False)
        out[f"FUSE ({predictor})"] = (r.scores, r.scores > 0.5)
    return out


def metrics(score, pred, y) -> dict:
    return {
        "Accuracy": round(100 * float((pred == y).mean()), 1),
        "AUC": round(100 * roc_auc_score(y, score), 1) if 0 < y.mean() < 1 else None,
        "Predicted unsafe %": round(100 * float(pred.mean()), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="dices990_v4")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--core", type=float, default=0.9, help="min share of the block's conversations a core rater rated")
    args = ap.parse_args()

    df = dices.load("990")
    lab = pd.read_csv(RESULTS / "llm_labels" / f"{args.labels}.csv").set_index("item_id").unsafe.astype(bool)
    block = blocks(df, args.k)
    pooled = {view: {m: {} for m in METHODS} for view in ("core raters", "all raters")}
    rows, per_block = [], []
    for b in sorted(block.unique()):
        items = block.index[block == b]
        dfb = df[df.item_id.isin(items)]
        dfb.attrs["variant"] = "990"
        core = core_raters(dfb, args.core)
        for view, d in (("core raters", dfb[dfb.rater_id.isin(core)]), ("all raters", dfb)):
            d = d.copy()
            d.attrs["variant"] = "990"
            m = dices.build(d, "overall")
            y = lab.reindex(m.items).to_numpy()
            preds = predictions(m)
            info = {"block": int(b), "view": view, "conversations": len(m.items), "raters": m.V.shape[1],
                    "missing %": round(100 * float(np.isnan(m.V).mean()), 1),
                    "ratings per conversation": int(np.median((~np.isnan(m.V)).sum(1))),
                    "label unsafe %": round(100 * float(y.mean()), 1)}
            for name, (s, p) in preds.items():
                info[name] = metrics(s, p, y)
                pooled[view][name].update({int(i): (bool(pi), bool(yi)) for i, pi, yi in zip(m.items, p, y)})
            mv = preds["Majority vote"][1]
            for name in METHODS[1:]:
                info[name]["vs MV"] = dices.paired_bootstrap(preds[name][1], mv, y)
            per_block.append(info)

    # Pooled over all 990 conversations: each block's own fit, predictions stacked.
    for view in pooled:
        ids = sorted(pooled[view]["Majority vote"])
        y = np.array([pooled[view]["Majority vote"][i][1] for i in ids])
        mv = np.array([pooled[view]["Majority vote"][i][0] for i in ids])
        for name in METHODS:
            p = np.array([pooled[view][name][i][0] for i in ids])
            rows.append({"view": view, "method": name, "conversations": len(ids),
                         "Accuracy": round(100 * float((p == y).mean()), 1),
                         "Predicted unsafe %": round(100 * float(p.mean()), 1),
                         "vs MV (95% CI)": "" if name == "Majority vote" else dices.paired_bootstrap(p, mv, y)})

    md = [
        "# FUSE on DICES-990, block by block (generated by scripts/dices990_blocks.py)", "",
        f"Reference: LLM label `{args.labels}`. Conversations clustered into {args.k} blocks by shared raters; "
        f"core raters rated at least {args.core:.0%} of their block's conversations. Each block is fitted separately.", "",
        "## Pooled over all 990 conversations", "",
        "| Raters used | Method | Conversations | Accuracy | Predicted unsafe % | vs majority vote (95% CI) |",
        "|---|---|---:|---:|---:|---|",
    ]
    md += [f"| {r['view']} | {r['method']} | {r['conversations']} | {r['Accuracy']} | {r['Predicted unsafe %']} | {r['vs MV (95% CI)']} |" for r in rows]
    md += ["", "## Per block", "",
           "| Block | Raters used | Conversations | Raters | Missing % | Ratings per conversation | Label unsafe % | MV | Dawid-Skene | FUSE logistic | FUSE posterior | FUSE logistic − MV (95% CI) |",
           "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for i in per_block:
        md.append(
            f"| {i['block']} | {i['view']} | {i['conversations']} | {i['raters']} | {i['missing %']} | {i['ratings per conversation']} | "
            f"{i['label unsafe %']} | {i['Majority vote']['Accuracy']} | {i['Dawid-Skene']['Accuracy']} | "
            f"{i['FUSE (logistic)']['Accuracy']} | {i['FUSE (posterior)']['Accuracy']} | {i['FUSE (logistic)']['vs MV']} |"
        )
    text = "\n".join(md)
    print(text)
    (RESULTS / "DICES990_BLOCKS.md").write_text(text + "\n")
    (RESULTS / "dices990_blocks.json").write_text(json.dumps({"pooled": rows, "per_block": per_block,
                                                              "block_of_item": {int(k): int(v) for k, v in block.items()}}, indent=2))


if __name__ == "__main__":
    main()
