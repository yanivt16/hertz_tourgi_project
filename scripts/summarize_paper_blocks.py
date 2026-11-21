"""results/paper_blocks.json -> results/PAPER_BLOCKS.md (see scripts/run_paper_blocks.py)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = json.loads((ROOT / "results" / "paper_blocks.json").read_text())
SETTINGS = [f"{b}-{s}" for s in ("70b", "8b") for b in ["gpqa", "gpqa-diamond", "math500", "mmlu", "mmlu-pro"]] + ["hle", "imo"]

COLS = [
    ("FUSE paper", "FUSE paper"),
    ("D&S", "D&S"),
    ("D&S → logistic", "D&S -> logistic"),
    ("Mean z", "Mean z"),
    ("Collapse (ties 0.4) → D&S → logistic", "Collapse ties@0.4 -> D&S -> logistic"),
    ("Collapse (ties 0.4) → FUSE", "Collapse ties@0.4 -> FUSE"),
    ("Collapse (ties 0.3) → D&S → logistic", "Collapse ties@0.3 -> D&S -> logistic"),
    ("Collapse (repo 0.4) → D&S → logistic", "Collapse repo@0.4 -> D&S -> logistic"),
    ("FUSE, median thresholds", "FUSE median thresholds"),
    ("D&S on FUSE's thresholds → logistic", "D&S on FUSE thresholds -> logistic"),
]
GAPS = [
    ("D&S → logistic, no collapsing", "D&S -> logistic"),
    ("Mean z", "Mean z"),
    ("Collapse (ties 0.4) → D&S → logistic", "Collapse ties@0.4 -> D&S -> logistic"),
    ("Collapse (ties 0.3) → D&S → logistic", "Collapse ties@0.3 -> D&S -> logistic"),
    ("Collapse (repo 0.4) → D&S → logistic", "Collapse repo@0.4 -> D&S -> logistic"),
    ("Collapse (repo 0.3) → D&S → logistic", "Collapse repo@0.3 -> D&S -> logistic"),
    ("Collapse (ties 0.4) → FUSE", "Collapse ties@0.4 -> FUSE"),
    ("FUSE, median thresholds", "FUSE median thresholds"),
    ("D&S on FUSE's thresholds → logistic", "D&S on FUSE thresholds -> logistic"),
]


def table(mode: str) -> list[str]:
    keys = [s for s in SETTINGS if f"{s}|{mode}" in RES]
    out = ["| Setting | m | " + " | ".join(c for c, _ in COLS) + " |", "|---|---:|" + "---:|" * len(COLS)]
    for s in keys:
        r = RES[f"{s}|{mode}"]
        a = r["accuracy"]
        best = max(a[k] for _, k in COLS)
        cells = [f"**{a[k]:.1f}**" if a[k] >= best - 0.05 else f"{a[k]:.1f}" for _, k in COLS]
        out.append(f"| {s} | {r['m']} | " + " | ".join(cells) + " |")
    out += ["", f"Gap to FUSE paper ({mode} mode), in points of selection accuracy, over {len(keys)} settings:", "",
            "| Method | Mean gap | Settings ahead by > 1 | Within ±1 | Behind by > 1 | Worst |", "|---|---:|---:|---:|---:|---:|"]
    for label, k in GAPS:
        d = np.array([RES[f"{s}|{mode}"]["accuracy"][k] - RES[f"{s}|{mode}"]["accuracy"]["FUSE paper"] for s in keys])
        out.append(f"| {label} | {d.mean():+.1f} | {(d > 1).sum()} | {(abs(d) <= 1).sum()} | {(d < -1).sum()} | {d.min():+.1f} |")
    return out


def groups() -> list[str]:
    out = ["| Setting | Detector | Columns after | Merged groups |", "|---|---|---:|---|"]
    for s in SETTINGS:
        r = RES.get(f"{s}|batch")
        if not r:
            continue
        for tag in ("ties@0.4", "ties@0.3", "repo@0.4"):
            gs = r["groups"][tag]
            desc = "; ".join("{" + ", ".join(v.replace("_scores", "").replace("_verdicts", " (verdict)") for v in g) + "}" for g in gs) or "none"
            out.append(f"| {s} | {tag} | {r['columns'][tag]} / {r['m']} | {desc} |")
    return out


def query_merges() -> list[str]:
    out = ["| Setting | m | ties 0.4: mean columns | ties 0.4: queries with a merge | repo 0.4: mean columns | repo 0.4: queries with a merge |",
           "|---|---:|---:|---:|---:|---:|"]
    for s in SETTINGS:
        r = RES.get(f"{s}|query")
        if not r:
            continue
        mc, sh = r["mean_columns"], r["share_of_queries_with_a_merge"]
        out.append(f"| {s} | {r['m']} | {mc['ties@0.4']:.1f} | {100 * sh['ties@0.4']:.0f}% | {mc['repo@0.4']:.1f} | {100 * sh['repo@0.4']:.0f}% |")
    return out


lines = [
    "# Shared-error blocks on the paper's Best-of-N benchmarks (generated from results/paper_blocks.json)",
    "",
    "Best-of-N selection accuracy (%), all methods label-free. Script: `scripts/run_paper_blocks.py`. "
    "Write-up: `docs/paper_blocks.md`. Bold: best in the row (within 0.05). `m`: verifiers (paper Table 6). "
    "'ties' uses a tie-aware Spearman in the residual detector; 'repo' uses `fuse.clustering.spearman_matrix` as it is.",
    "",
    "## Batched mode (one ensemble per setting)",
    "",
    *table("batch"),
    "",
    "## Per-query mode (one ensemble per question)",
    "",
    *table("query"),
    "",
    "## What the detector merged (batched mode)",
    "",
    *groups(),
    "",
    "## How often it merged (per-query mode)",
    "",
    *query_merges(),
    "",
]
(ROOT / "results" / "PAPER_BLOCKS.md").write_text("\n".join(lines))
print("\n".join(lines))
