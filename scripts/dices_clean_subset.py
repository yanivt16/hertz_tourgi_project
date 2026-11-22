"""DICES-350 without the conversations where the crowd clearly disagrees with the expert label.

    python scripts/dices_clean_subset.py --reps 100

Contested conversations: with all raters, every one of four aggregators (majority vote, three-answer
Dawid-Skene, Yes-only Dawid-Skene, FUSE) puts the conversation on the opposite side of the expert label when
each calls exactly 50% unsafe (the experts' prevalence). Two other definitions are reported for size:
each method's own rule (> 0.5), and both rules together.

On the remaining conversations ("clean subset") everything is refitted from scratch, label-free:
(a) all raters as one panel, (b) small random panels of about 5, 10 and 20 raters (disjoint, 100 reshuffles,
as in scripts/dices_small_panels.py), scored at each method's own rule and at shared cut-offs
(scripts/panel_eval.py), and (c) the eigenvalues of the class-conditional residual correlation
(scripts/dices_error_blocks.py) before and after the removal.

The removal uses the expert label on every conversation, so accuracy on the clean subset is optimistic for
every method; the comparison between methods is the point, not the level.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import dices_error_blocks as EB  # noqa: E402
import panel_eval as PE  # noqa: E402
import tune_dices350 as T  # noqa: E402
from fuse import FUSE, dices  # noqa: E402

OUT_JSON = ROOT / "results" / "dices_clean_subset.json"
OUT_MD = ROOT / "results" / "DICES_CLEAN_SUBSET.md"
FILTER_METHODS = ["Majority vote", "Dawid–Skene, 3 answers", "Dawid–Skene, Yes only", "FUSE"]

_V = _Y = None


def _init(V, y):
    global _V, _Y
    _V, _Y = V, y


def one_rep(args):
    size, rep = args
    perm = np.random.default_rng(20_000 + rep).permutation(_V.shape[1])
    return [PE.evaluate_panel(_V[:, cols], _Y) for cols in np.array_split(perm, _V.shape[1] // size)]


def full_scores(V):
    f = FUSE().fit_scores(V, rescale=False)
    return {"Majority vote": ((V + 1) / 2).mean(1), "Dawid–Skene, 3 answers": dices.dawid_skene_3class(V),
            "Dawid–Skene, Yes only": dices.dawid_skene((V > 0).astype(float)), "FUSE": f.scores}


def contested(V, y):
    n = len(y)
    S = full_scores(V)
    half = np.array([PE.top_k(s, n // 2) != y for s in S.values()]).all(0)
    natural = np.array([(s > 0.5) != y for s in S.values()]).all(0)
    return half, natural, S


def eig_info(V, y, dev):
    """Residual-correlation eigenvalues on the dev part of the given conversations (raters with variance)."""
    X, yy = V[dev], y[dev]
    keep = X.std(0) > 0
    C = EB.corr0(EB.residuals(X[:, keep], yy))
    w = EB.eig_desc(C)[0]
    N, T_ = int(keep.sum()), int(dev.sum())
    lam = EB.mp_edge(N, T_)
    return {"N": N, "T": T_, "top5": w[:5].tolist(), "mp_edge": float(lam), "n_above": int((w > lam).sum()),
            "global_share": float(w[0] / N), "mean_pair_corr": float(EB.offdiag(C).mean())}


def fmt_pair(p):
    return (f"| {p['a']} − {p['b']} | {p['cut']} | {p['diff']:+.2f} | {p['a_wins']:.0f} / {p['tie']:.0f} / {p['b_wins']:.0f} | "
            f"[{p['lo']:+.2f}, {p['hi']:+.2f}] |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--sizes", type=int, nargs="*", default=[5, 10, 20])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    t0 = time.time()

    d = T.Data()
    V, y, items, dev = d.raters_V, d.gold, d.items, d.dev
    half, natural, S_all = contested(V, y)
    agree = np.where(y, (V > 0).mean(1), (V < 0).mean(1))
    keep = ~half
    Vc, yc = V[keep], y[keep]
    out = {"n_all": len(y), "contested": {
        "definition": "all four methods wrong when each calls 50% unsafe (all raters, all 350 conversations)",
        "n": int(half.sum()), "unsafe": int(half[y].sum()), "safe": int(half[~y].sum()),
        "n_natural_rule": int(natural.sum()), "unsafe_natural_rule": int(natural[y].sum()),
        "n_both_rules": int((half & natural).sum()),
        "rater_agreement_with_expert_median": {"contested": float(np.median(agree[half])), "rest": float(np.median(agree[keep]))},
        "by_harm_degree": {}, "item_ids": [str(i) for i in items[half]]},
        "n_clean": int(keep.sum()), "clean_prevalence": float(yc.mean())}
    item_df = d.df.drop_duplicates("item_id").set_index("item_id").reindex(items)
    for deg in sorted(item_df.degree_of_harm.astype(str).unique()):
        m = (item_df.degree_of_harm.astype(str) == deg).to_numpy()
        out["contested"]["by_harm_degree"][deg] = {"n": int(m.sum()), "contested": int((m & half).sum()),
                                                  "contested_unsafe": int((m & half & y).sum()), "contested_safe": int((m & half & ~y).sum())}

    # residual eigenvalues before / after (dev part only, as in dices_error_blocks)
    out["eigen"] = {"all_350": eig_info(V, y, dev), "clean": eig_info(Vc, yc, dev[keep])}

    # all raters on the clean subset
    full = PE.evaluate_panel(Vc, yc)
    out["all_raters"] = {m: {c: 100 * float(full["hits"][m][c].mean()) for c in PE.CUTS} | {"auc": full["auc"][m], "called": full["called"][m]}
                         for m in PE.METHODS}
    out["all_raters_est"] = {"fuse": full["fuse_est_prev"], "ds3": full["ds3_est_prev"]}
    full350 = PE.evaluate_panel(V, y)
    out["all_raters_350"] = {m: {c: 100 * float(full350["hits"][m][c].mean()) for c in PE.CUTS} | {"auc": full350["auc"][m]} for m in PE.METHODS}
    rows1 = [full]
    rng = np.random.default_rng(0)
    out["all_raters_pairs"] = []
    for cut in PE.CUTS:
        h = rows1[0]["hits"]
        dd = h["FUSE"][cut].astype(float) - h["Dawid–Skene, 3 answers"][cut]
        bs = 100 * dd[rng.integers(0, len(dd), (5000, len(dd)))].mean(1)
        out["all_raters_pairs"].append({"cut": cut, "diff": float(100 * dd.mean()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5))})
    print(f"contested {half.sum()}, clean {keep.sum()} ({time.time() - t0:.0f}s)", flush=True)

    out["sizes"] = {}
    for size in a.sizes:
        with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(Vc, yc)) as pool:
            rows = [r for res in pool.map(one_rep, [(size, r) for r in range(a.reps)], chunksize=4) for r in res]
        pairs = [PE.paired(rows, "FUSE", "Dawid–Skene, 3 answers", c) for c in PE.CUTS]
        pairs += [PE.paired(rows, "FUSE", "Majority vote", "at FUSE est."), PE.paired(rows, "Dawid–Skene, 3 answers", "Majority vote", "at DS-3 est.")]
        aucs = [PE.paired_auc(rows, "FUSE", b) for b in ["Dawid–Skene, 3 answers", "Dawid–Skene, Yes only", "Majority vote"]]
        out["sizes"][str(size)] = {"panels": len(rows), "methods": PE.summarize(rows), "pairs": pairs, "auc_pairs": aucs, "rho": PE.rho_summary(rows),
                                   "est_prev": {"fuse": float(np.nanmean([np.nan if r["fuse_est_prev"] is None else r["fuse_est_prev"] for r in rows])),
                                                "ds3": float(np.mean([r["ds3_est_prev"] for r in rows])),
                                                "fallbacks": int(sum(r["fuse_fallback"] is not None for r in rows))}}
        print(f"size {size}: {len(rows)} panels ({time.time() - t0:.0f}s)", flush=True)
    # same panels on all 350 for reference, read from the existing small-panels run
    sp = ROOT / "results" / "dices_small_panels.json"
    if sp.exists():
        ref = json.loads(sp.read_text())["sizes"]
        out["reference_350"] = {s: [p for p in ref[s]["pairs"] if p["a"] == "FUSE" and p["b"] == "Dawid–Skene, 3 answers" and "test" not in p["cut"]]
                                for s in ref if s in out["sizes"]}
    OUT_JSON.write_text(json.dumps(out, indent=1))
    write_md(out)
    print(f"done ({time.time() - t0:.0f}s)")


def write_md(o):
    c = o["contested"]
    md = ["# DICES-350 without the contested conversations (generated by scripts/dices_clean_subset.py)", "",
          f"Raw numbers and the removed conversation ids: [dices_clean_subset.json](dices_clean_subset.json).", "",
          "## Which conversations are contested", "",
          f"**Contested** = {c['definition']}. That is **{c['n']} of {o['n_all']}** conversations ({100 * c['n'] / o['n_all']:.0f}%): "
          f"{c['unsafe']} the experts call unsafe and {c['safe']} they call safe. On them a median of "
          f"{100 * c['rater_agreement_with_expert_median']['contested']:.0f}% of raters agree with the expert, against "
          f"{100 * c['rater_agreement_with_expert_median']['rest']:.0f}% on the rest. Other definitions: every method wrong at its own rule "
          f"(> 0.5): {c['n_natural_rule']} ({c['unsafe_natural_rule']} unsafe, because the methods under-call); wrong under both: {c['n_both_rules']}.", "",
          "| Degree of harm (DICES) | Conversations | Contested | of which unsafe by experts | safe by experts |", "|---|---:|---:|---:|---:|"]
    md += [f"| {k} | {v['n']} | {v['contested']} | {v['contested_unsafe']} | {v['contested_safe']} |" for k, v in c["by_harm_degree"].items()]
    md += ["", f"Clean subset: {o['n_clean']} conversations, {100 * o['clean_prevalence']:.0f}% unsafe by the experts. Everything below is "
           "refitted on it without labels. The removal used the expert label, so accuracy here is optimistic for every method: compare "
           "methods, not levels.", "",
           "## Shared errors before and after (dev part, class-conditional residual correlation)", "",
           "| Conversations | Raters × conversations | Mean pair correlation | Top 5 eigenvalues | MP edge | Above edge | Global mode share |",
           "|---|---|---:|---|---:|---:|---:|"]
    for k, nm in [("all_350", "All 350"), ("clean", "Clean subset")]:
        e = o["eigen"][k]
        md.append(f"| {nm} | {e['N']} × {e['T']} | {e['mean_pair_corr']:.3f} | {', '.join(f'{x:.2f}' for x in e['top5'])} | {e['mp_edge']:.2f} | "
                  f"{e['n_above']} | {100 * e['global_share']:.0f}% |")
    md += ["", "## All raters as one panel", "", "| Method | Natural | At FUSE est. | At DS-3 est. | AUC | Called unsafe % | (all 350: at DS-3 est. / AUC) |",
           "|---|---:|---:|---:|---:|---:|---|"]
    for m, v in o["all_raters"].items():
        r = o["all_raters_350"][m]
        md.append(f"| {m} | {v['natural']:.1f} | {v['at FUSE est.']:.1f} | {v['at DS-3 est.']:.1f} | {v['auc']:.1f} | {v['called']:.0f} | "
                  f"{r['at DS-3 est.']:.1f} / {r['auc']:.1f} |")
    md += ["", f"Estimated unsafe share: FUSE {o['all_raters_est']['fuse']:.2f}, three-answer Dawid–Skene {o['all_raters_est']['ds3']:.2f} "
           f"(experts {o['clean_prevalence']:.2f}). FUSE − three-answer Dawid–Skene: " +
           "; ".join(f"{p['cut']} {p['diff']:+.2f} [{p['lo']:+.2f}, {p['hi']:+.2f}]" for p in o["all_raters_pairs"]) + ".", ""]
    for s, v in o["sizes"].items():
        md += [f"## Small random panels of about {s} raters ({v['panels']} panels)", "",
               f"Estimated unsafe share, mean over panels: FUSE {v['est_prev']['fuse']:.2f}, three-answer Dawid–Skene {v['est_prev']['ds3']:.2f}; "
               f"FUSE fell back to a plain average in {v['est_prev']['fallbacks']} panels.", "",
               "| Method | Natural | At FUSE est. | At DS-3 est. | AUC | Called unsafe % |", "|---|---:|---:|---:|---:|---:|"]
        md += [f"| {m} | {x['natural']:.1f} | {x['at FUSE est.']:.1f} | {x['at DS-3 est.']:.1f} | {x['auc']:.1f} | {x['called']:.0f} |" for m, x in v["methods"].items()]
        md += ["", "| Comparison | Cut-off | Mean difference | A wins / tie / B wins (% of panels) | Over conversations (95% CI) |", "|---|---|---:|---|---|"]
        md += [fmt_pair(p) for p in v["pairs"]]
        if "reference_350" in o and s in o["reference_350"]:
            md += ["", "Same panels on all 350 conversations (results/DICES_SMALL_PANELS.md), FUSE − three-answer Dawid–Skene: " +
                   "; ".join(f"{p['cut']} {p['diff']:+.2f} [{p['lo']:+.2f}, {p['hi']:+.2f}]" for p in o["reference_350"][s]) + "."]
        md += ["", "| AUC comparison | Mean difference | A higher in % of panels | Middle 95% of panels |", "|---|---:|---:|---|"]
        md += [f"| {p['a']} − {p['b']} | {p['diff']:+.2f} | {p['a_wins']:.0f} | {p['p2.5']:+.1f} to {p['p97.5']:+.1f} |" for p in v["auc_pairs"]]
        r = v["rho"]
        md += ["", f"Spearman between estimated and true rater quality, mean over panels: FUSE {r['fuse']:+.2f}, three-answer Dawid–Skene "
               f"{r['ds3']:+.2f}; FUSE higher in {r['fuse_higher_share']:.0f}% of {r['n']} panels.", ""]
    OUT_MD.write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
