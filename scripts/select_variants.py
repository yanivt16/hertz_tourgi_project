"""Pick a FUSE variant per setting without labels and score the pick against the cached grid.

    python scripts/select_variants.py            # diagnostics -> results/selection.json
                                                 # table       -> results/SELECTION.md

The choice (`fuse.selection.select_variant`) sees only score matrices and extracted answers.
Accuracies come from results/variants_sweep.json (scripts/run_variants.py); nothing is re-run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fuse import baselines as bl  # noqa: E402
from fuse.experiment import has_answers  # noqa: E402
from fuse.preprocess import impute_missing, minmax_to_unit_interval  # noqa: E402
from fuse.selection import select_variant  # noqa: E402
from run_variants import SETTINGS, load  # noqa: E402
from summarize_variants import ORDER  # noqa: E402

RESULTS = ROOT / "results"
CONFIGS = [(m, p) for m in ("query", "batch") for p in ("logistic", "posterior")]


def diagnostics(name: str) -> dict:
    bench = load(name)
    mats = [impute_missing(q.scores) for q in bench.queries]
    mvs = [bl.majority_vote_scores(q.answers) for q in bench.queries] if has_answers(bench) else None
    out = {}
    for mode in ("query", "batch"):
        if mode == "query":
            sel = select_variant(mats, mvs)
        else:
            stacked = [np.vstack([minmax_to_unit_interval(V) for V in mats])]
            mv = None if mvs is None else [np.concatenate([minmax_to_unit_interval(np.asarray(c, float)[:, None])[:, 0] for c in mvs])]
            sel = select_variant(stacked, mv)
        out[mode] = {
            "variant": sel.variant,
            "label_shares": {str(t): s for t, s in sel.label_shares.items()},
            "mv_pass_rate": sel.mv_pass_rate,
        }
        print(f"{name:18s} {mode:5s} -> {sel.variant:14s} R={out[mode]['label_shares']} mv={sel.mv_pass_rate}", flush=True)
    return out


def fmt(x) -> str:
    return "" if x is None else f"{x:.1f}"


def write_table(sel: dict) -> str:
    grid = json.loads((RESULTS / "variants_sweep.json").read_text())
    acc = lambda n, m, p, v: grid.get(f"{n}|{m}|{p}|{v}", {}).get("FUSE")  # noqa: E731
    variants = sorted({k.split("|")[3] for k in grid})
    names = [n for n in ORDER if n in sel]

    lines = ["# Label-free variant selection", ""]
    lines.append("The rule was committed (fba0117) before it was run on these settings or joined with")
    lines.append("accuracy; its constants (R < 0.5, median) were not tuned. The per-setting accuracies in")
    lines.append("VARIANTS.md were already known when it was designed.")
    lines.append("")
    lines.append("Selected variant per setting and mode (both predictors use the same pick), from")
    lines.append("`fuse.selection.select_variant`. R_t is the label share at clustering threshold t")
    lines.append("(cluster when < 0.5); MV pass is the fraction of queries (or 1/0 in batched mode)")
    lines.append("where the majority-vote column fits the rank-one model at least as well as the median verifier.")
    lines.append("")
    lines.append("| Setting | Mode | R@0.3 | R@0.5 | R@0.7 | MV pass | Selected |")
    lines.append("|---|---|---:|---:|---:|---:|---|")
    for n in names:
        for m in ("query", "batch"):
            d = sel[n][m]
            r = [d["label_shares"].get(t) for t in ("0.3", "0.5", "0.7")]
            mv = d["mv_pass_rate"]
            lines.append(
                f"| {n} | {m} | " + " | ".join("" if x is None else f"{x:.2f}" for x in r)
                + f" | {'' if mv is None else f'{mv:.2f}'} | {d['variant']} |"
            )

    lines += ["", "## Mean accuracy over the 12 settings", ""]
    lines.append("Best-in-hindsight picks the best variant per setting using the labels (an upper bound,")
    lines.append("not a method). Best fixed variant is the one variant with the highest mean over all")
    lines.append("settings, also chosen with hindsight.")
    lines.append("")
    lines.append("| Mode | Predictor | base | selected | best fixed variant | best in hindsight | selected vs base (wins/ties/losses) | selected − base without math500-8b |")
    lines.append("|---|---|---:|---:|---:|---:|---|---:|")
    summary = {}
    for m, p in CONFIGS:
        base = [acc(n, m, p, "base") for n in names]
        chosen = [acc(n, m, p, sel[n][m]["variant"]) for n in names]
        oracle = [max(a for v in variants if (a := acc(n, m, p, v)) is not None) for n in names]
        fixed = {v: [acc(n, m, p, v) for n in names] for v in variants}
        fixed = {v: np.mean(xs) for v, xs in fixed.items() if None not in xs}
        best_fixed = max(fixed, key=fixed.get)
        diff = np.array(chosen) - np.array(base)
        w, t, l = int((diff > 0.05).sum()), int((np.abs(diff) <= 0.05).sum()), int((diff < -0.05).sum())
        summary[(m, p)] = (base, chosen, oracle)
        lines.append(
            f"| {m} | {p} | {np.mean(base):.1f} | {np.mean(chosen):.1f} | {fixed[best_fixed]:.1f} ({best_fixed}) "
            f"| {np.mean(oracle):.1f} | {w}/{t}/{l} | {np.mean(np.delete(diff, names.index('math500-8b'))):+.2f} |"
        )

    for m, p in CONFIGS:
        base, chosen, oracle = summary[(m, p)]
        lines += ["", f"## {m} mode, {p} predictor", ""]
        lines.append("| Setting | base | selected | variant | best in hindsight | best variant |")
        lines.append("|---|---:|---:|---|---:|---|")
        for n, b, c, o in zip(names, base, chosen, oracle):
            best_v = max((v for v in variants if acc(n, m, p, v) is not None), key=lambda v: acc(n, m, p, v))
            lines.append(f"| {n} | {fmt(b)} | {fmt(c)} | {sel[n][m]['variant']} | {fmt(o)} | {best_v} |")
        lines.append(f"| **mean** | {np.mean(base):.1f} | {np.mean(chosen):.1f} | | {np.mean(oracle):.1f} | |")

    out = "\n".join(lines) + "\n"
    (RESULTS / "SELECTION.md").write_text(out)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--settings", nargs="*", default=SETTINGS)
    args = p.parse_args()
    path = RESULTS / "selection.json"
    sel = json.loads(path.read_text()) if path.exists() else {}
    for name in args.settings:
        if name not in sel:
            sel[name] = diagnostics(name)
            path.write_text(json.dumps(sel, indent=1))
    print(write_table(sel))


if __name__ == "__main__":
    main()
