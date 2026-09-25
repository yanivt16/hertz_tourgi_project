"""Reproduce the paper's Best-of-N tables.

    python scripts/run_experiment.py imo
    python scripts/run_experiment.py hle
    python scripts/run_experiment.py weaver --benchmark mmlu-pro --size 70b
    python scripts/run_experiment.py all
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fuse import FUSE  # noqa: E402
from fuse.data import WEAVER_BENCHMARKS, download_fuse, download_weaver  # noqa: E402
from fuse.experiment import format_table, run_supervised, run_unsupervised  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def run(bench, args) -> dict:
    print(bench.summary(), flush=True)
    start = time.time()
    fuse = FUSE(predictor=args.predictor, n_sweeps=args.sweeps)
    res = run_unsupervised(bench, fuse, mode=args.mode, workers=args.workers)
    res.update(run_supervised(bench, seeds=args.seeds))
    print(format_table(bench.name, res))
    print(f"best verifier: {res['_best_verifier']}  ({time.time() - start:.0f}s)\n", flush=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"{bench.name}_{args.mode}_{args.predictor}.json").write_text(json.dumps(res, indent=2))
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset", choices=["hle", "imo", "weaver", "all"])
    p.add_argument("--benchmark", choices=list(WEAVER_BENCHMARKS), default="mmlu-pro")
    p.add_argument("--size", choices=["8b", "70b"], default="70b")
    p.add_argument("--mode", choices=["query", "batch"], default="query")
    p.add_argument("--predictor", choices=["logistic", "posterior"], default="logistic")
    p.add_argument("--sweeps", type=int, default=3)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    if args.dataset in ("hle", "imo"):
        run(download_fuse(args.dataset), args)
    elif args.dataset == "weaver":
        run(download_weaver(args.benchmark, args.size), args)
    else:
        for key in ("imo", "hle"):
            run(download_fuse(key), args)
        for size in ("70b", "8b"):
            for b in WEAVER_BENCHMARKS:
                run(download_weaver(b, size), args)


if __name__ == "__main__":
    main()
