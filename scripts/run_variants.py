"""Compare FUSE with and without the two label-free extensions on all 12 settings.

    python scripts/run_variants.py                    # full grid, writes results/VARIANTS.md
    python scripts/run_variants.py --thresholds 0.3 0.5 0.7

Variants: `base` (the paper's FUSE), `cluster` (correlated verifiers averaged into one
meta-verifier), `mv` (answer agreement added as a verifier) and `cluster+mv`. Each runs in
per-query and batched mode with both final predictors. Loaded benchmarks are cached under
.cache/benchmarks so the Weaver parquet shards are read once.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fuse import FUSE  # noqa: E402
from fuse.data import WEAVER_BENCHMARKS, download_fuse, download_weaver  # noqa: E402
from fuse.experiment import has_answers, run_unsupervised  # noqa: E402

RESULTS = ROOT / "results"
CACHE = ROOT / ".cache" / "benchmarks"
SETTINGS = [f"{b}-{s}" for s in ("70b", "8b") for b in WEAVER_BENCHMARKS] + ["hle", "imo"]


def load(name: str):
    path = CACHE / f"{name}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    if name in ("hle", "imo"):
        bench = download_fuse(name)
    else:
        bench_name, size = name.rsplit("-", 1)
        bench = download_weaver(bench_name, size)
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(bench))
    return bench


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--settings", nargs="*", default=SETTINGS)
    p.add_argument("--thresholds", nargs="*", type=float, default=[0.5])
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", default="variants.json")
    args = p.parse_args()

    out_path = RESULTS / args.out
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    for name in args.settings:
        bench = load(name)
        mv_ok = has_answers(bench)
        print(bench.summary(), "| answers:", mv_ok, flush=True)
        variants = [("base", None, False)]
        for t in args.thresholds:
            variants.append((f"cluster@{t}", t, False))
        if mv_ok:
            variants.append(("mv", None, True))
            for t in args.thresholds:
                variants.append((f"cluster@{t}+mv", t, True))
        for mode in ("query", "batch"):
            for predictor in ("logistic", "posterior"):
                for label, thr, mv in variants:
                    key = f"{name}|{mode}|{predictor}|{label}"
                    if key in results:
                        continue
                    start = time.time()
                    fuse = FUSE(predictor=predictor, cluster_threshold=thr)
                    res = run_unsupervised(bench, fuse, mode=mode, workers=args.workers, majority_vote_verifier=mv)
                    results[key] = res
                    print(f"  {key:55s} FUSE={res['FUSE']:.1f}  ({time.time() - start:.0f}s)", flush=True)
                    out_path.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
