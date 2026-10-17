"""Turn results/variants_sweep.json (from run_variants.py) into results/VARIANTS.md."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fuse.experiment import PAPER_RESULTS  # noqa: E402

RESULTS = ROOT / "results"
ORDER = [
    "gpqa-70b", "gpqa-diamond-70b", "math500-70b", "mmlu-70b", "mmlu-pro-70b",
    "gpqa-8b", "gpqa-diamond-8b", "math500-8b", "mmlu-8b", "mmlu-pro-8b", "hle", "imo",
]
CONFIGS = [(m, p) for m in ("query", "batch") for p in ("logistic", "posterior")]


def fmt(x) -> str:
    return "" if x is None else f"{x:.1f}"


def mean(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def main():
    data = json.loads((RESULTS / "variants_sweep.json").read_text())
    get = lambda name, mode, pred, var: data.get(f"{name}|{mode}|{pred}|{var}", {}).get("FUSE")  # noqa: E731
    variants = sorted({k.split("|")[3] for k in data}, key=lambda v: (v != "base", "mv" in v, v))
    names = [n for n in ORDER if any(k.startswith(n + "|") for k in data)]

    lines = ["## Mean over settings, by configuration", ""]
    lines.append("Means over the settings where every variant ran (MV needs extracted answers).")
    lines.append("")
    lines.append("| Mode | Predictor | " + " | ".join(variants) + " |")
    lines.append("|---|---|" + "---:|" * len(variants))
    for mode, pred in CONFIGS:
        common = [n for n in names if all(get(n, mode, pred, v) is not None for v in variants)]
        cells = [fmt(mean(get(n, mode, pred, v) for n in common)) for v in variants]
        lines.append(f"| {mode} | {pred} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Means over all settings for the variants that need no answers:")
    lines.append("")
    no_mv = [v for v in variants if "mv" not in v]
    lines.append("| Mode | Predictor | " + " | ".join(no_mv) + " |")
    lines.append("|---|---|" + "---:|" * len(no_mv))
    for mode, pred in CONFIGS:
        cells = [fmt(mean(get(n, mode, pred, v) for n in names)) for v in no_mv]
        lines.append(f"| {mode} | {pred} | " + " | ".join(cells) + " |")

    for mode, pred in CONFIGS:
        lines += ["", f"## {mode} mode, {pred} predictor", ""]
        lines.append("| Setting | " + " | ".join(variants) + " | FUSE paper |")
        lines.append("|---|" + "---:|" * (len(variants) + 1))
        for n in names:
            cells = [fmt(get(n, mode, pred, v)) for v in variants]
            lines.append(f"| {n} | " + " | ".join(cells) + f" | {fmt(PAPER_RESULTS.get(n, {}).get('FUSE'))} |")

    out = "\n".join(lines) + "\n"
    (RESULTS / "VARIANTS.md").write_text(out)
    print(out)


if __name__ == "__main__":
    main()
