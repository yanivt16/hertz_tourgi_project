"""Collect results/*.json into results/SUMMARY.md, side by side with the paper."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fuse.experiment import PAPER_RESULTS  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
ORDER = [
    "gpqa-70b", "gpqa-diamond-70b", "math500-70b", "mmlu-70b", "mmlu-pro-70b",
    "gpqa-8b", "gpqa-diamond-8b", "math500-8b", "mmlu-8b", "mmlu-pro-8b", "hle", "imo",
]


def load(name: str, mode: str, predictor: str = "logistic") -> dict:
    path = RESULTS / f"{name}_{mode}_{predictor}.json"
    return json.loads(path.read_text()) if path.exists() else {}


def fmt(x) -> str:
    return "" if x is None else f"{x:.1f}"


def main():
    rows = [
        "| Benchmark | Pass@1 | Naive Ens. | Oracle Best Verifier | Logistic (5% labels) "
        "| FUSE query | FUSE batch | FUSE batch + posterior | FUSE paper | Weaver paper | Pass@k |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ORDER:
        q, b, p = load(name, "query"), load(name, "batch"), PAPER_RESULTS.get(name, {})
        bp = load(name, "batch", "posterior")
        if not q:
            continue
        rows.append(
            f"| {name} | {fmt(q.get('Pass@1'))} | {fmt(q.get('Naive Ensemble'))} | {fmt(q.get('Oracle Best Verifier'))} "
            f"| {fmt(q.get('Logistic Regression'))} | {fmt(q.get('FUSE'))} | {fmt(b.get('FUSE'))} | {fmt(bp.get('FUSE'))} "
            f"| {fmt(p.get('FUSE'))} | {fmt(p.get('Weaver'))} | {fmt(q.get('Pass@k'))} |"
        )
    out = "\n".join(rows) + "\n"
    (RESULTS / "SUMMARY.md").write_text(out)
    print(out)


if __name__ == "__main__":
    main()
