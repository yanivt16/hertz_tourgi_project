"""Inside each demographic group: majority vote vs Dawid–Skene vs FUSE on that group's raters (DICES-350).

Hypothesis: 123 raters dilute the good raters' signal; inside a smaller, more homogeneous crowd
FUSE finds them better than Dawid–Skene. Unlike the grouped views (one verifier per group), every
rater stays a separate verifier: for each group of one attribute (gender, race, education, age)
the methods run on only that group's raters, on all 350 conversations, scored against the expert
label. Intersections are too small to use.

Control: for each group, 100 random panels of the SAME size drawn from all 123 raters. The
difference "demographic minus size-matched random" separates "these raters are alike" from
"the panel is small".

The comparison is like for like (scripts/panel_eval.py): the main Dawid–Skene uses all three
answers, as FUSE does, and accuracy is also scored with every method calling the same number of
conversations unsafe (at FUSE's estimated prevalence, and at three-answer Dawid–Skene's).
Intervals are paired bootstraps over conversations (5000 draws); with 13 groups and several
comparisons each, a few intervals will exclude 0 by chance.

    python scripts/dices_demographic_panels.py --reps 100
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

import panel_eval as PE  # noqa: E402
import tune_dices350 as T  # noqa: E402

ATTRS = ["rater_gender", "rater_race", "rater_education", "rater_age"]
SMALL = 10  # groups smaller than this are flagged as unreliable
A, B = "FUSE", "Dawid–Skene, 3 answers"
SHOW = ["Majority vote", "Dawid–Skene, 3 answers", "Dawid–Skene, Yes only", "FUSE"]

_V = _Y = None


def _init(V, y):
    global _V, _Y
    _V, _Y = V, y


def random_panels(args):
    key, size, reps = args
    rng = np.random.default_rng(key)
    return [PE.evaluate_panel(_V[:, rng.choice(_V.shape[1], size, replace=False)], _Y) for _ in range(reps)]


def boot(diff_per_conv: np.ndarray, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff_per_conv), (5000, len(diff_per_conv)))
    bs = 100 * diff_per_conv[idx].mean(1)
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def ci(x, lo_hi):
    return f"{x:+.1f} [{lo_hi[0]:+.1f}, {lo_hi[1]:+.1f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()

    d = T.Data()
    V, y = d.raters_V, d.gold
    raters = np.array(sorted(d.df.rater_id.unique()))
    m = T.dices.build(d.df, view="overall")
    assert list(m.verifiers) == list(raters), "rater order must match the overall view's columns"
    info = d.df.drop_duplicates("rater_id").set_index("rater_id")

    groups = []
    for attr in ATTRS:
        for g, ids in info.groupby(attr).groups.items():
            cols = np.flatnonzero(np.isin(raters, list(ids)))
            groups.append({"attr": attr.replace("rater_", ""), "group": str(g), "cols": cols, "n": len(cols)})

    t0 = time.time()
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(V, y)) as pool:
        ctrl = list(pool.map(random_panels, [(5000 + i, g["n"], a.reps) for i, g in enumerate(groups)]))
    _init(V, y)
    print(f"controls done ({time.time() - t0:.0f}s)", flush=True)

    rows, out = [], {"reps": a.reps, "groups": []}
    for i, (g, rnd) in enumerate(zip(groups, ctrl)):
        r = PE.evaluate_panel(V[:, g["cols"]], y)
        rec = {"attr": g["attr"], "group": g["group"], "n": g["n"], "small": g["n"] < SMALL,
               "fuse_est_prev": r["fuse_est_prev"], "ds3_est_prev": r["ds3_est_prev"], "fuse_fallback": r["fuse_fallback"],
               "rho_fuse": r["rho_fuse"], "rho_ds3": r["rho_ds3"], "rho_random": PE.rho_summary(rnd),
               "acc": {}, "auc": {}, "random_acc": {}, "random_auc": {}, "demog_minus_random": {}}
        for meth in PE.METHODS:
            rec["auc"][meth] = r["auc"][meth]
            rec["random_auc"][meth] = float(np.mean([x["auc"][meth] for x in rnd]))
            for cut in PE.CUTS:
                h = r["hits"][meth][cut].astype(float)
                hr = np.mean([x["hits"][meth][cut] for x in rnd], axis=0)
                rec["acc"].setdefault(meth, {})[cut] = 100 * float(h.mean())
                rec["random_acc"].setdefault(meth, {})[cut] = 100 * float(hr.mean())
                rec["demog_minus_random"].setdefault(meth, {})[cut] = {"diff": 100 * float((h - hr).mean()), "ci": boot(h - hr, i)}
        gaps = {}
        for cut in PE.CUTS:
            hd = r["hits"][A][cut].astype(float) - r["hits"][B][cut]
            hr = np.mean([x["hits"][A][cut].astype(float) - x["hits"][B][cut] for x in rnd], axis=0)
            gaps[cut] = {"demog": 100 * float(hd.mean()), "demog_ci": boot(hd, i),
                         "random": 100 * float(hr.mean()), "random_ci": boot(hr, i),
                         "demog_minus_random": 100 * float((hd - hr).mean()), "dmr_ci": boot(hd - hr, i)}
        rec["fuse_minus_ds3"] = gaps
        rec["auc_fuse_minus_ds3"] = r["auc"][A] - r["auc"][B]
        rec["auc_fuse_minus_ds3_random"] = float(np.mean([x["auc"][A] - x["auc"][B] for x in rnd]))
        out["groups"].append(rec)
        rows.append(rec)

    # Pooled over groups: the average gap, and how many groups each side leads.
    def pooled(cut):
        v = [x["fuse_minus_ds3"][cut]["demog"] for x in rows]
        ok = [x for x in rows if not x["small"]]
        return {"mean": float(np.mean(v)), "fuse_ahead": sum(x > 0 for x in v), "ds_ahead": sum(x < 0 for x in v), "groups": len(v),
                "ci_excludes_0": sum(1 for x in rows if x["fuse_minus_ds3"][cut]["demog_ci"][0] > 0 or x["fuse_minus_ds3"][cut]["demog_ci"][1] < 0),
                "mean_large_groups": float(np.mean([x["fuse_minus_ds3"][cut]["demog"] for x in ok]))}
    out["pooled"] = {cut: pooled(cut) for cut in PE.CUTS}
    out["pooled"]["demog_minus_random_gap"] = {cut: float(np.mean([x["fuse_minus_ds3"][cut]["demog_minus_random"] for x in rows])) for cut in PE.CUTS}
    out["pooled"]["rho"] = {"fuse": float(np.mean([x["rho_fuse"] for x in rows if x["rho_fuse"] is not None])),
                            "ds3": float(np.mean([x["rho_ds3"] for x in rows if x["rho_ds3"] is not None])),
                            "fuse_higher": sum(1 for x in rows if x["rho_fuse"] is not None and x["rho_ds3"] is not None and x["rho_fuse"] > x["rho_ds3"])}

    md = ["# Inside each demographic group, DICES-350 (generated by scripts/dices_demographic_panels.py)", "",
          "Each group's raters stay separate verifiers; the methods see only that group's raters, on all 350 conversations. "
          f"Control: {a.reps} random panels of the same size from all 123 raters. Accuracy (%) against the expert label. "
          "**Natural**: each method's own rule (score > 0.5). **Shared cut-off**: every method calls the same number of "
          "conversations unsafe, set by FUSE's estimated prevalence (the like-for-like comparison). DS-3 is Dawid–Skene with "
          "three answers (Yes / Unsure / No). Intervals: paired bootstrap over conversations. "
          f"Groups under {SMALL} raters are marked *small* and should not be read on their own. With 13 groups and several "
          "comparisons each, a few intervals will exclude 0 by chance.", "",
          "## Accuracy and AUC per group", "",
          "| Attribute | Group | Raters | MV natural | DS-3 natural | FUSE natural | MV shared | DS-3 shared | FUSE shared | AUC MV | AUC DS-3 | AUC FUSE | FUSE est. | DS-3 est. |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    cutS = "at FUSE est."
    for x in rows:
        acc, auc = x["acc"], x["auc"]
        fe = "—" if x["fuse_est_prev"] is None else f"{x['fuse_est_prev']:.2f}"
        md.append(f"| {x['attr']} | {x['group']}{' (small)' if x['small'] else ''} | {x['n']} | {acc['Majority vote']['natural']:.1f} | "
                  f"{acc[B]['natural']:.1f} | {acc[A]['natural']:.1f} | {acc['Majority vote'][cutS]:.1f} | {acc[B][cutS]:.1f} | {acc[A][cutS]:.1f} | "
                  f"{auc['Majority vote']:.1f} | {auc[B]:.1f} | {auc[A]:.1f} | {fe} | {x['ds3_est_prev']:.2f} |")
    md += ["", "## FUSE minus three-answer Dawid–Skene, inside the group and in size-matched random panels", "",
           "| Group | Raters | Shared cut: demographic | Shared cut: random, same size | Demographic − random | Natural: demographic | AUC: demographic | AUC: random |",
           "|---|---:|---|---|---|---|---:|---:|"]
    for x in rows:
        g, gn = x["fuse_minus_ds3"][cutS], x["fuse_minus_ds3"]["natural"]
        md.append(f"| {x['attr']}: {x['group']}{' (small)' if x['small'] else ''} | {x['n']} | {ci(g['demog'], g['demog_ci'])} | "
                  f"{ci(g['random'], g['random_ci'])} | {ci(g['demog_minus_random'], g['dmr_ci'])} | {ci(gn['demog'], gn['demog_ci'])} | "
                  f"{x['auc_fuse_minus_ds3']:+.1f} | {x['auc_fuse_minus_ds3_random']:+.1f} |")
    md += ["", "## Demographic minus size-matched random, per method (shared cut-off)", "",
           "| Group | Raters | Majority vote | DS-3 | DS, Yes only | FUSE |", "|---|---:|---|---|---|---|"]
    for x in rows:
        dm = x["demog_minus_random"]
        md.append(f"| {x['attr']}: {x['group']}{' (small)' if x['small'] else ''} | {x['n']} | " +
                  " | ".join(ci(dm[meth][cutS]["diff"], dm[meth][cutS]["ci"]) for meth in SHOW) + " |")
    md += ["", "## Does each method find the good raters?", "",
           "Spearman between estimated and true rater quality (FUSE: on the raters it keeps, true quality of its thresholded "
           "vote; DS-3: true quality of the Yes vote).", "",
           "| Group | Raters | FUSE | DS-3 | Random panels: FUSE | Random panels: DS-3 |", "|---|---:|---:|---:|---:|---:|"]
    for x in rows:
        f = "—" if x["rho_fuse"] is None else f"{x['rho_fuse']:+.2f}"
        dd = "—" if x["rho_ds3"] is None else f"{x['rho_ds3']:+.2f}"
        rr = x["rho_random"]
        md.append(f"| {x['attr']}: {x['group']}{' (small)' if x['small'] else ''} | {x['n']} | {f} | {dd} | "
                  f"{rr['fuse']:+.2f} | {rr['ds3']:+.2f} |")
    p = out["pooled"]
    md += ["", "## Pooled over the 13 groups", "",
           f"- Shared cut-off at FUSE's estimate: FUSE − DS-3 averages {p[cutS]['mean']:+.2f} points; FUSE ahead in "
           f"{p[cutS]['fuse_ahead']} groups, DS-3 in {p[cutS]['ds_ahead']}; intervals exclude 0 in {p[cutS]['ci_excludes_0']}. "
           f"Without the small groups: {p[cutS]['mean_large_groups']:+.2f}.",
           f"- Shared cut-off at DS-3's estimate: {p['at DS-3 est.']['mean']:+.2f}; FUSE ahead in {p['at DS-3 est.']['fuse_ahead']}, "
           f"DS-3 in {p['at DS-3 est.']['ds_ahead']}.",
           f"- Natural cut-offs: {p['natural']['mean']:+.2f}; FUSE ahead in {p['natural']['fuse_ahead']}, DS-3 in {p['natural']['ds_ahead']}.",
           f"- Demographic minus random in the FUSE − DS-3 gap (shared cut at FUSE's estimate): {p['demog_minus_random_gap'][cutS]:+.2f} on average.",
           f"- Rater-quality correlation, mean over groups: FUSE {p['rho']['fuse']:+.2f}, DS-3 {p['rho']['ds3']:+.2f}; "
           f"FUSE higher in {p['rho']['fuse_higher']} of 13 groups.", ""]
    (ROOT / "results" / "DICES_DEMOGRAPHIC_PANELS.md").write_text("\n".join(md) + "\n")
    (ROOT / "results" / "dices_demographic_panels.json").write_text(json.dumps(out, indent=1))
    print(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
