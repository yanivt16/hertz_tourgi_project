"""FUSE vs Dawid–Skene on random rater groups, over many random splits (DICES-350).

Each split shuffles the 123 raters and cuts them into k near-equal groups (k = 123 // size, so
size 20 gives 6 groups of 20–21). A group's score for a conversation is the share of its raters
saying unsafe (Unsure = 1/2). On each split:

* FUSE defaults: FUSE on the group shares, unsafe when its score > 0.5.
* FUSE tuned: the same fit, cut at FUSE's own estimated prevalence (the cut-off adopted in
  scripts/tune_dices350.py, chosen on the dev half; so compare it on the test half).
* Dawid–Skene: one vote per group (unsafe when its share > 1/2), EM, unsafe when q > 0.5.
* Dawid–Skene cut at its own estimated prevalence (the same label-free cut-off as FUSE tuned).
* Majority of group votes, for reference.

Like for like (added after review). Cutting each method at its OWN estimate is not one cut-off:
both under-call, and whichever calls more wins accuracy. So FUSE and Dawid–Skene are also scored
calling the SAME number of conversations unsafe, at FUSE's estimate and at Dawid–Skene's. And
Dawid–Skene on group votes sees less than FUSE (one yes/no per group against each group's share),
so two fuller Dawid–Skene baselines are added: Dawid–Skene on the group shares cut into 5 bins
(fuse.ordinal), and three-answer Dawid–Skene on the individual raters, which needs no groups.
Ties are broken by one fixed random order of the conversations (scripts/panel_eval.py).

Neither FUSE defaults nor Dawid–Skene uses a label, so they can be compared on all 350.

Two kinds of uncertainty are reported. Across splits: how often one method beats the other and
how much the result moves with the grouping. Across conversations: each conversation's hit
rate averaged over all splits, then a paired bootstrap over conversations, which says whether
the difference would hold on new conversations rather than on new groupings.

    python scripts/dices_random_groups.py --splits 1000          # ~minutes on 10 cores
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import panel_eval as PE  # noqa: E402
import tune_dices350 as T  # noqa: E402
from fuse import FUSE, dices  # noqa: E402
from fuse.ordinal import dawid_skene_ordinal  # noqa: E402
from fuse.preprocess import minmax_to_unit_interval  # noqa: E402

METHODS = ["FUSE defaults", "FUSE tuned (cut at its prevalence)", "FUSE cut at DS est.",
           "Dawid–Skene", "Dawid–Skene cut at its prevalence", "Dawid–Skene cut at FUSE est.",
           "Dawid–Skene, 5-bin shares", "Dawid–Skene, 5-bin shares cut at FUSE est.", "Majority of group votes",
           "Dawid–Skene per rater, 3 answers", "Dawid–Skene per rater, 3 answers cut at FUSE est."]
FIXED = "Dawid–Skene per rater (no groups)"  # same on every split; the strongest Dawid–Skene here
PAIRS = [("FUSE defaults", "Dawid–Skene", "all"),
         ("FUSE defaults", "Dawid–Skene", "test"),
         ("FUSE tuned (cut at its prevalence)", "Dawid–Skene cut at FUSE est.", "all"),
         ("FUSE tuned (cut at its prevalence)", "Dawid–Skene cut at FUSE est.", "test"),
         ("FUSE cut at DS est.", "Dawid–Skene cut at its prevalence", "all"),
         ("FUSE cut at DS est.", "Dawid–Skene cut at its prevalence", "test"),
         ("FUSE tuned (cut at its prevalence)", "Dawid–Skene, 5-bin shares cut at FUSE est.", "test"),
         ("FUSE tuned (cut at its prevalence)", "Dawid–Skene per rater, 3 answers cut at FUSE est.", "test"),
         ("FUSE tuned (cut at its prevalence)", FIXED, "test"),
         ("FUSE defaults", FIXED, "all")]

_R = _Q3 = None  # (items, raters) share of Yes; per-rater three-answer Dawid–Skene; set in each worker


def _init(R, q3=None):
    global _R, _Q3
    _R, _Q3 = R, q3


def top_share(score, share):
    return PE.top_k(score, int(round(share * len(score))))


def one_split(args):
    size, seed = args
    n_raters = _R.shape[1]
    perm = np.random.default_rng(10_000 + seed).permutation(n_raters)
    groups = np.array_split(perm, n_raters // size)
    S = np.column_stack([_R[:, g].mean(axis=1) for g in groups])
    r = FUSE().fit_scores(minmax_to_unit_interval(S), rescale=False)
    est = None if r.estimate is None else float(r.estimate.prevalence)
    votes = S > 0.5
    q = dices.dawid_skene(votes.astype(float))
    qo = dawid_skene_ordinal(S, k=5)
    ef = est if est is not None else 0.5
    if est is None:
        print(f"warning: FUSE gave no prevalence estimate (size {size}, split {seed}); its cut uses 0.5", flush=True)
    preds = {
        "FUSE defaults": (r.scores, r.scores > 0.5),
        "FUSE tuned (cut at its prevalence)": (r.scores, top_share(r.scores, est) if est is not None else r.scores > 0.5),
        "Dawid–Skene": (q, q > 0.5),
        "Dawid–Skene cut at its prevalence": (q, top_share(q, float(q.mean()))),
        "FUSE cut at DS est.": (r.scores, top_share(r.scores, float(q.mean()))),
        "Dawid–Skene cut at FUSE est.": (q, top_share(q, ef)),
        "Dawid–Skene, 5-bin shares": (qo, qo > 0.5),
        "Dawid–Skene, 5-bin shares cut at FUSE est.": (qo, top_share(qo, ef)),
        "Majority of group votes": (votes.mean(1), votes.mean(1) > 0.5),
        "Dawid–Skene per rater, 3 answers": (_Q3, _Q3 > 0.5),
        "Dawid–Skene per rater, 3 answers cut at FUSE est.": (_Q3, top_share(_Q3, ef)),
    }
    return {"size": size, "seed": seed, "k": len(groups), "fuse_est_prev": est, "ds_est_prev": float(q.mean()),
            "fallback": r.fallback, "pred": {m: p for m, (_, p) in preds.items()},
            "score": {m: s for m, (s, _) in preds.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=1000)
    ap.add_argument("--sizes", type=int, nargs="*", default=[5, 10, 20])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()

    d = T.Data()
    y, dev = d.gold, d.dev
    masks = {"dev": dev, "test": ~dev, "all": np.ones_like(dev)}
    df = d.df.assign(_yes=d.df.Q_overall.map({"Yes": 1.0, "Unsure": 0.5, "No": 0.0}))
    R = df.pivot(index="item_id", columns="rater_id", values="_yes").reindex(d.items).to_numpy()
    assert not np.isnan(R).any()
    ds_r = dices.dawid_skene(d.raters_V)
    fixed_score, fixed_pred = ds_r, ds_r > 0.5

    out = {"splits": a.splits, "sizes": {}, "n": {k: int(v.sum()) for k, v in masks.items()}}
    md = ["# FUSE vs Dawid–Skene on random rater groups (generated by scripts/dices_random_groups.py)", "",
          f"{a.splits} random splits of the 123 raters per group size. Accuracy (%) against the expert label. "
          f"Dev {out['n']['dev']} / test {out['n']['test']} conversations are the halves used for tuning; "
          "FUSE tuned's cut-off was chosen on dev, so compare it on test.", ""]
    for size in a.sizes:
        t0 = time.time()
        hits = {m: np.zeros(len(y)) for m in METHODS + [FIXED]}  # per-conversation hit counts over splits
        per = []
        with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(R, dices.dawid_skene_3class(d.raters_V))) as pool:
            for i, res in enumerate(pool.map(one_split, [(size, s) for s in range(a.splits)], chunksize=8)):
                row = {"seed": res["seed"], "k": res["k"], "fuse_est_prev": res["fuse_est_prev"],
                       "ds_est_prev": res["ds_est_prev"], "fallback": res["fallback"]}
                res["pred"][FIXED], res["score"][FIXED] = fixed_pred, fixed_score
                for m in METHODS + [FIXED]:
                    p, s = res["pred"][m], res["score"][m]
                    ok = p == y
                    hits[m] += ok
                    for split, mk in masks.items():
                        row[f"{m}|{split}"] = round(100 * float(ok[mk].mean()), 2)
                    row[f"{m}|auc"] = round(100 * float(roc_auc_score(y, s)), 2)
                    row[f"{m}|unsafe"] = round(100 * float(p.mean()), 1)
                per.append(row)
                if (i + 1) % 100 == 0:
                    print(f"size {size}: {i + 1}/{a.splits} splits ({time.time() - t0:.0f}s)", flush=True)

        summ = {}
        for m in METHODS + [FIXED]:
            summ[m] = {}
            for split in ["dev", "test", "all", "auc", "unsafe"]:
                v = np.array([r[f"{m}|{split}"] for r in per])
                summ[m][split] = {"mean": round(float(v.mean()), 2), "sd": round(float(v.std()), 2),
                                  "p2.5": round(float(np.percentile(v, 2.5)), 2), "p97.5": round(float(np.percentile(v, 97.5)), 2)}
        pairs = []
        rng = np.random.default_rng(0)
        for m1, m2, split in PAIRS:
            diff = np.array([r[f"{m1}|{split}"] - r[f"{m2}|{split}"] for r in per])
            mk = masks[split]
            # Conversation-level: average hit rate over splits, paired bootstrap over conversations.
            h1, h2 = hits[m1][mk] / a.splits, hits[m2][mk] / a.splits
            idx = rng.integers(0, mk.sum(), (5000, mk.sum()))
            bs = 100 * (h1[idx].mean(1) - h2[idx].mean(1))
            pairs.append({"a": m1, "b": m2, "split": split,
                          "mean_diff": round(float(diff.mean()), 2),
                          "share_a_wins": round(100 * float((diff > 0).mean()), 1),
                          "share_tie": round(100 * float((diff == 0).mean()), 1),
                          "share_b_wins": round(100 * float((diff < 0).mean()), 1),
                          "diff_p2.5": round(float(np.percentile(diff, 2.5)), 2),
                          "diff_p97.5": round(float(np.percentile(diff, 97.5)), 2),
                          "conv_diff": round(float(100 * (h1.mean() - h2.mean())), 2),
                          "conv_ci": [round(float(np.percentile(bs, 2.5)), 2), round(float(np.percentile(bs, 97.5)), 2)]})
        k = per[0]["k"]
        fe = np.array([r["fuse_est_prev"] for r in per if r["fuse_est_prev"] is not None])
        out["sizes"][str(size)] = {"k": k, "summary": summ, "pairs": pairs,
                                   "fuse_est_prev_mean": round(float(fe.mean()), 3),
                                   "n_fallback": sum(r["fallback"] is not None for r in per)}
        print(f"size {size} done in {time.time() - t0:.0f}s", flush=True)

        md += [f"## Groups of about {size} ({k} groups per split)", "",
               f"FUSE's estimated unsafe share averages {fe.mean():.2f} over splits; Dawid–Skene's "
               f"{np.mean([r['ds_est_prev'] for r in per]):.2f} (experts 0.50).", "",
               "| Method | Test, mean ± sd | Test, middle 95% of splits | All 350, mean | AUC (all), mean | Called unsafe % |",
               "|---|---:|---:|---:|---:|---:|"]
        for m in METHODS + [FIXED]:
            s = summ[m]
            md.append(f"| {m} | {s['test']['mean']:.1f} ± {s['test']['sd']:.1f} | {s['test']['p2.5']:.1f}–{s['test']['p97.5']:.1f} | "
                      f"{s['all']['mean']:.1f} | {s['auc']['mean']:.1f} | {s['unsafe']['mean']:.0f} |")
        md += ["", "| Comparison | Half | Mean difference | A wins / tie / B wins (% of splits) | Middle 95% of splits | "
               "Over conversations (95% CI) |", "|---|---|---:|---|---|---|"]
        for p in pairs:
            md.append(f"| {p['a']} − {p['b']} | {p['split']} | {p['mean_diff']:+.1f} | "
                      f"{p['share_a_wins']:.0f} / {p['share_tie']:.0f} / {p['share_b_wins']:.0f} | "
                      f"{p['diff_p2.5']:+.1f} to {p['diff_p97.5']:+.1f} | {p['conv_diff']:+.1f} [{p['conv_ci'][0]:+.1f}, {p['conv_ci'][1]:+.1f}] |")
        md.append("")
        (ROOT / "results" / "DICES_RANDOM_GROUPS.md").write_text("\n".join(md) + "\n")
        (ROOT / "results" / "dices_random_groups.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
