"""Do the aggregation methods make the same mistakes on DICES-350?

    python scripts/dices_error_overlap.py      # writes results/DICES_ERRORS.md and results/dices350_errors.json

Uses the per-rater view (123 raters, Q_overall Yes/Unsure/No -> 1/0/-1) and the expert
`safety_gold` label. For every method it keeps the per-conversation prediction, then measures
how far the error sets overlap, which conversations every method misses, and whether any
subset of raters catches those conversations.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fuse import FUSE, dices  # noqa: E402
from fuse import baselines as bl  # noqa: E402

RESULTS = ROOT / "results"
LABEL_FREE = ["MV", "Naive", "DS", "Jaffe", "FUSE-L", "FUSE-P"]
NAMES = {
    "MV": "Majority vote",
    "Naive": "Naive ensemble (mean > 0)",
    "DS": "Dawid-Skene",
    "Jaffe": "Jaffe et al. (2015), sign of the LLR",
    "FUSE-L": "FUSE (logistic)",
    "FUSE-P": "FUSE (posterior)",
    "BestRater": "Best single rater (uses labels)",
    "LR10": "Logistic regression, 10% labels, cross-fitted (uses labels)",
    "CrowdOpt": "Crowd share, gold-tuned threshold (uses labels)",
}
REASON = {
    "acceptance_safety_bias_inheritedattributes": "Bias: inherited attributes",
    "acceptance_policy_polarizing": "Policy: polarizing topics",
    "acceptance_safety_harmful_advice": "Harmful advice",
    "acceptance_safety_bias_gendersexuality": "Bias: gender and sexuality",
    "acceptance_safety_harmful_dangerous": "Dangerous content",
    "acceptance_safety_misinformation": "Misinformation",
    "acceptance_other": "Other",
    "acceptance_safety_politicalaffiliation": "Political affiliation",
    "acceptance_safety_harmful_personalinformation": "Personal information",
    "acceptance_safety_bias_status": "Bias: status",
}


def predictions(m: dices.DicesMatrix, X: np.ndarray, y: np.ndarray) -> tuple[dict, dict]:
    scores, preds = {}, {}

    def add(key, s, p):
        scores[key], preds[key] = np.asarray(s, float), np.asarray(p, bool)

    add("MV", m.majority, m.majority > 0.5)
    naive = X.mean(1)
    add("Naive", naive, naive > 0)
    q = dices.dawid_skene(m.V)
    add("DS", q, q > 0.5)
    j = bl.jaffe2015_scores(X)
    add("Jaffe", j, j > 0)
    for key, predictor in (("FUSE-L", "logistic"), ("FUSE-P", "posterior")):
        r = FUSE(predictor=predictor).fit_scores(X, rescale=False)
        add(key, r.scores, r.scores > 0.5)
    aucs = np.array([roc_auc_score(y, X[:, k]) if X[:, k].std() > 0 else 0.5 for k in range(X.shape[1])])
    best = int(np.argmax(aucs))
    add("BestRater", X[:, best], X[:, best] > 0)
    # 10 folds x 5 repeats: train on one fold (10% of labels), predict the other nine, then
    # average each conversation's probability over the 45 models that never saw its label.
    rng = np.random.default_rng(0)
    n = len(y)
    prob, cnt = np.zeros(n), np.zeros(n)
    for _ in range(5):
        for fold in np.array_split(rng.permutation(n), 10):
            rest = np.setdiff1d(np.arange(n), fold)
            clf = LogisticRegression(max_iter=3000).fit(X[fold], y[fold])
            prob[rest] += clf.predict_proba(X[rest])[:, 1]
            cnt[rest] += 1
    add("LR10", prob / cnt, prob / cnt > 0.5)
    ths = np.unique(m.majority)
    t = ths[int(np.argmax([((m.majority > t) == y).mean() for t in ths]))]
    add("CrowdOpt", m.majority, m.majority > t)
    return scores, preds


def md_table(df: pd.DataFrame, index_name: str = "") -> str:
    cols = [index_name] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|---|" + "---:|" * len(df.columns)]
    for idx, row in zip(df.index, df.astype(object).itertuples(index=False)):  # iterrows would upcast ints
        lines.append(f"| {idx} | " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def main():
    df = dices.load("350")
    m = dices.build(df, "overall")
    y, X = m.gold, dices._prepared(m)
    scores, preds = predictions(m, X, y)
    keys = list(preds)

    perf = pd.DataFrame(
        {
            NAMES[k]: {
                "Accuracy": round(100 * float((p == y).mean()), 1),
                "AUC": round(100 * roc_auc_score(y, scores[k]), 1),
                "Predicted unsafe %": round(100 * float(p.mean()), 1),
                "Missed unsafe (FN)": int((y & ~p).sum()),
                "False alarm (FP)": int((~y & p).sum()),
            }
            for k, p in preds.items()
        }
    ).T
    for c in ("Missed unsafe (FN)", "False alarm (FP)"):
        perf[c] = perf[c].astype(int)

    E = pd.DataFrame({k: preds[k] != y for k in keys})
    def overlap(a, b):
        union = (E[a] | E[b]).sum()
        return int(round(100 * (E[a] & E[b]).sum() / union)) if union else 100

    jac = pd.DataFrame({b: {a: overlap(a, b) for a in keys} for b in keys}).loc[keys, keys]
    agree = pd.DataFrame(
        {b: {a: int(round(100 * (preds[a] == preds[b]).mean())) for a in keys} for b in keys}
    ).loc[keys, keys]

    k_all, k_lf = E.sum(1).to_numpy(), E[LABEL_FREE].sum(1).to_numpy()
    all_wrong = k_all == len(keys)
    hard = all_wrong & y  # expert-unsafe conversations that every method misses

    it = df.groupby("item_id").agg(reason=("safety_gold_reason", "first"), doh=("degree_of_harm", "first")).reindex(m.items)
    it["share"], it["gold"], it["hard"] = m.majority, y, hard
    unsafe = it[it.gold].copy()
    unsafe["r"] = unsafe.reason.astype(str).apply(lambda s: re.findall(r"acceptance_[a-z_]+", s))
    by_reason = unsafe.explode("r").groupby("r").agg(n=("hard", "size"), missed=("hard", "sum"))
    by_reason["share missed by all"] = (100 * by_reason.missed / by_reason.n).round(0).astype(int).astype(str) + "%"
    by_reason = by_reason.sort_values("n", ascending=False).head(10)
    by_reason.index = [REASON.get(r, r) for r in by_reason.index]

    # Rater-level view of the hard set: is there a subset of raters that sees what experts see?
    said_yes = (X > 0.5).astype(float)
    r_hard, r_safe = said_yes[hard].mean(0), said_yes[~y].mean(0)
    rng = np.random.default_rng(0)
    idx = np.flatnonzero(hard)
    split_half = float(np.mean([
        np.corrcoef(said_yes[p[: len(p) // 2]].mean(0), said_yes[p[len(p) // 2 :]].mean(0))[0, 1]
        for p in (rng.permutation(idx) for _ in range(200))
    ]))
    strict = []
    for k in (10, 20, 30):
        top = np.argsort(-said_yes.mean(0))[:k]
        share = said_yes[:, top].mean(1)
        strict.append({
            "raters": k,
            "hard caught": f"{int((share[hard] > 0.5).sum())}/{int(hard.sum())}",
            "accuracy": round(100 * float(((share > 0.5) == y).mean()), 1),
            "predicted unsafe %": round(100 * float((share > 0.5).mean()), 0),
        })
    raters = df.drop_duplicates("rater_id").set_index("rater_id").reindex(m.verifiers)
    demo = []
    for col in ("rater_gender", "rater_race", "rater_age"):
        g = pd.DataFrame({"group": raters[col].values, "hard": r_hard, "safe": r_safe}).groupby("group").mean()
        for name, row in g.iterrows():
            demo.append({"attribute": col, "group": name, "says unsafe on hard set %": round(100 * row.hard, 1),
                         "says unsafe on expert-safe %": round(100 * row.safe, 1)})

    hist = {int(k): int(v) for k, v in pd.Series(k_all).value_counts().sort_index().items()}
    facts = {
        "label_free_wrong_by_all_6": int((k_lf == 6).sum()),
        "label_free_right_by_all_6": int((k_lf == 0).sum()),
        "label_free_split": int(((k_lf > 0) & (k_lf < 6)).sum()),
        "wrong_by_all_9": int(all_wrong.sum()),
        "wrong_by_all_9_expert_unsafe": int(hard.sum()),
        "hard_crowd_share_range": [round(float(m.majority[hard].min()), 3), round(float(m.majority[hard].max()), 3)],
        "other_expert_unsafe_mean_share": round(float(m.majority[y & ~hard].mean()), 3),
        "expert_safe_all_methods_unsafe_shares": [round(float(s), 3) for s in m.majority[all_wrong & ~y]],
        "corr_rater_rate_hard_vs_safe": round(float(np.corrcoef(r_hard, r_safe)[0, 1]), 2),
        "split_half_corr_hard": round(split_half, 2),
        "raters_flag_hard_10pts_more_than_safe": int((r_hard - r_safe >= 0.10).sum()),
        "mean_rater_rate_hard": round(100 * float(r_hard.mean()), 1),
        "mean_rater_rate_safe": round(100 * float(r_safe.mean()), 1),
    }

    md = [
        "# Do the methods make the same mistakes? (DICES-350, 123 raters)",
        "",
        "Generated by `scripts/dices_error_overlap.py`. Per-rater view, scored against the expert `safety_gold` label.",
        "",
        "## Results per method",
        "",
        md_table(perf, "Method"),
        "",
        "The 10%-label logistic row averages 45 models (10 folds × 5 repeats), each trained on 10% of conversations; "
        "each conversation is predicted only by models that never saw its label. Averaging is why it beats the "
        "single-model figure in DICES.md (67.6).",
        "",
        "## Overlap of errors",
        "",
        "Jaccard overlap of the two methods' error sets (%):",
        "",
        md_table(jac, ""),
        "",
        "Share of conversations on which the two methods predict the same label (%):",
        "",
        md_table(agree, ""),
        "",
        f"Number of methods (of 9) wrong per conversation: {hist}.",
        "",
        f"Across the six label-free methods, {facts['label_free_wrong_by_all_6']} conversations are wrong under all six, "
        f"{facts['label_free_right_by_all_6']} right under all six and {facts['label_free_split']} depend on the method. "
        f"{facts['wrong_by_all_9']} conversations are wrong under all nine methods, including those that use labels: "
        f"{facts['wrong_by_all_9_expert_unsafe']} expert-unsafe ones and "
        f"{facts['wrong_by_all_9'] - facts['wrong_by_all_9_expert_unsafe']} expert-safe ones "
        f"(crowd unsafe share {facts['expert_safe_all_methods_unsafe_shares']}).",
        "",
        "## The conversations every method misses",
        "",
        f"On the {facts['wrong_by_all_9_expert_unsafe']} expert-unsafe conversations every method misses, "
        f"{100 * facts['hard_crowd_share_range'][0]:.0f}–{100 * facts['hard_crowd_share_range'][1]:.0f}% of raters say unsafe, "
        f"against {100 * facts['other_expert_unsafe_mean_share']:.0f}% on average for the other expert-unsafe conversations.",
        "",
        md_table(by_reason[["n", "missed"]].rename(columns={"n": "Expert-unsafe", "missed": "Missed by every method"}).assign(Share=by_reason["share missed by all"]), "Expert reason"),
        "",
        "## Does any group of raters catch them?",
        "",
        f"- A rater's unsafe rate on these conversations correlates {facts['corr_rater_rate_hard_vs_safe']} with the same "
        "rater's unsafe rate on expert-safe conversations: the raters who flag them are the ones who flag everything.",
        f"- On average raters flag them less ({facts['mean_rater_rate_hard']}%) than expert-safe conversations "
        f"({facts['mean_rater_rate_safe']}%).",
        f"- {facts['raters_flag_hard_10pts_more_than_safe']} of 123 raters flag them at least 10 points more often than they flag expert-safe ones.",
        f"- Split-half consistency of which raters flag them: {facts['split_half_corr_hard']}, so the pattern is stable, and it is leniency.",
        "",
        "Majority among the strictest raters:",
        "",
        md_table(pd.DataFrame(strict).set_index("raters"), "Strictest raters"),
        "",
        "By demographic group (mean over the group's raters):",
        "",
        md_table(pd.DataFrame(demo).set_index("group"), "Group"),
        "",
    ]
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "DICES_ERRORS.md").write_text("\n".join(md))
    out = {
        "performance": perf.reset_index(names="method").to_dict("records"),
        "jaccard": jac.to_dict(),
        "agreement": agree.to_dict(),
        "methods_wrong_histogram": hist,
        "facts": facts,
        "by_reason": by_reason.reset_index(names="reason").to_dict("records"),
        "strict_raters": strict,
        "demographics": demo,
        "hard_item_ids": [int(i) for i in m.items[hard]],
    }
    (RESULTS / "dices350_errors.json").write_text(json.dumps(out, indent=2))
    print("\n".join(md))


if __name__ == "__main__":
    main()
