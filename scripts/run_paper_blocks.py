"""Does collapsing shared-error blocks erase FUSE's lead on the paper's own Best-of-N benchmarks?

    python scripts/run_paper_blocks.py --cache /path/to/.cache            # all 12 settings, both modes
    python scripts/run_paper_blocks.py --settings imo hle --modes batch

The claim under test (docs/residual_blocks.md): FUSE beats Dawid-Skene only where a block of
verifiers shares errors, and once a label-free detector collapses that block, "D&S -> logistic"
matches FUSE. Every method here is label-free and scored by Best-of-N selection accuracy.

Detector: fuse.clustering.residual_groups at thresholds 0.3 and 0.4, run with the repo's
`spearman_matrix` ("repo") and with a tie-aware Spearman ("ties"). The repo version ranks tied
values by row order, which gives two independent binary columns a correlation of about 0.2;
the 70B settings include binary LM-judge verdicts, so both are reported.

Writes results/paper_blocks.json; scripts/summarize_paper_blocks.py turns it into a table.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import fuse.clustering as cl  # noqa: E402
from fuse import FUSE  # noqa: E402
from fuse.data import PAPER_VERIFIERS, WEAVER_BENCHMARKS, download_fuse, download_weaver  # noqa: E402
from fuse.dices import dawid_skene  # noqa: E402
from fuse.metrics import selection_accuracy  # noqa: E402
from fuse.preprocess import impute_missing, minmax_to_unit_interval  # noqa: E402
from fuse.tci import binarize as tci_binarize  # noqa: E402

SETTINGS = [f"{b}-{s}" for s in ("70b", "8b") for b in WEAVER_BENCHMARKS] + ["hle", "imo"]
REPO_SPEARMAN = cl.spearman_matrix
DETECTORS = [("repo", 0.4), ("repo", 0.3), ("ties", 0.4), ("ties", 0.3)]


def tie_spearman(V: np.ndarray) -> np.ndarray:
    """Spearman with average ranks for ties; constant columns correlate 0 with everything."""
    R = rankdata(V, axis=0)
    R -= R.mean(axis=0)
    norm = np.sqrt((R**2).sum(axis=0))
    nz = norm > 0
    C = np.zeros((V.shape[1], V.shape[1]))
    C[np.ix_(nz, nz)] = (R[:, nz].T @ R[:, nz]) / np.outer(norm[nz], norm[nz])
    np.fill_diagonal(C, 1.0)
    return C


def binarize(V):
    """As in run_residual_blocks.py: two-valued columns split at their minimum, others at the median."""
    two = np.array([len(np.unique(V[:, j])) <= 2 for j in range(V.shape[1])])
    return np.where(two[None, :], np.where(V > V.min(0), 1.0, -1.0), np.where(V > np.median(V, 0), 1.0, -1.0))


def zmean(V):
    sd = V.std(0)
    return ((V - V.mean(0)) / np.where(sd > 0, sd, 1)).mean(1)


def methods(V: np.ndarray, rescaled: bool) -> tuple[dict[str, np.ndarray], dict]:
    """Score every response with every label-free method; also return the groups each detector found."""
    Vs = V if rescaled else minmax_to_unit_interval(V)
    out, groups_out = {}, {}
    fit = FUSE().fit_scores(V, rescale=not rescaled)
    paper = fit.scores
    ds = dawid_skene(binarize(V))
    out["FUSE paper"] = paper
    out["D&S"] = ds
    out["D&S -> logistic"] = FUSE()._predict(Vs, ds)
    out["Mean z"] = zmean(V)
    # Diagnostics: which part of FUSE carries its lead, the threshold search or the estimator?
    out["FUSE median thresholds"] = FUSE(n_sweeps=0).fit_scores(V, rescale=not rescaled).scores
    if fit.tau is not None:
        act = np.isfinite(fit.tau)
        out["D&S on FUSE thresholds -> logistic"] = FUSE()._predict(Vs, dawid_skene(tci_binarize(Vs[:, act], fit.tau[act])))
    else:
        out["D&S on FUSE thresholds -> logistic"] = paper
    seen: dict[tuple, dict[str, np.ndarray]] = {}
    for kind, t in DETECTORS:
        cl.spearman_matrix = REPO_SPEARMAN if kind == "repo" else tie_spearman
        try:
            groups, _, _ = cl.residual_groups(V, t)
        finally:
            cl.spearman_matrix = REPO_SPEARMAN
        tag = f"{kind}@{t}"
        groups_out[tag] = [g.tolist() for g in groups]
        sig = tuple(tuple(g) for g in groups)
        if sig not in seen:
            if len(groups) == V.shape[1]:  # nothing merged: identical to no collapsing
                res = {"D&S -> logistic": out["D&S -> logistic"], "FUSE": paper, "Mean z": out["Mean z"]}
            else:
                Vc = cl.collapse_groups(V, groups)
                res = {"D&S -> logistic": FUSE()._predict(Vs, dawid_skene(binarize(Vc))), "Mean z": zmean(Vc)}
                res["FUSE"] = (
                    FUSE()._predict(Vs, FUSE(predictor="posterior").fit_scores(Vc).scores)
                    if Vc.shape[1] >= 3 else res["D&S -> logistic"]
                )
            seen[sig] = res
        for k, s in seen[sig].items():
            out[f"Collapse {tag} -> {k}"] = s
    return out, groups_out


def _query(args):
    V, y = args
    scores, groups = methods(impute_missing(V), rescaled=False)
    return {k: selection_accuracy(s, y) for k, s in scores.items()}, {k: len(g) for k, g in groups.items()}


def load(name: str, cache: Path):
    path = cache / f"{name}.pkl"
    if path.exists():
        bench = pickle.loads(path.read_bytes())
    elif name in ("hle", "imo"):
        bench = download_fuse(name)
    else:
        b, size = name.rsplit("-", 1)
        bench = download_weaver(b, size)
    if name not in ("hle", "imo"):  # the cache may hold a few extra columns; keep the paper's Table 6 set
        size = name.rsplit("-", 1)[1]
        bench = bench.filter_verifiers([v for v in bench.verifiers if v in PAPER_VERIFIERS[size]])
    return bench


def run(name: str, bench, mode: str, workers: int) -> dict:
    ys = [q.correct for q in bench.queries]
    t0 = time.time()
    if mode == "query":
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_query, [(q.scores, q.correct) for q in bench.queries], chunksize=4))
        acc = {k: 100 * float(np.mean([r[0][k] for r in rows])) for k in rows[0][0]}
        ncols = {k: float(np.mean([r[1][k] for r in rows])) for k in rows[0][1]}
        merged = {k: float(np.mean([r[1][k] < len(bench.verifiers) for r in rows])) for k in rows[0][1]}
        detail = {"mean_columns": ncols, "share_of_queries_with_a_merge": merged}
    else:
        mats = [minmax_to_unit_interval(impute_missing(q.scores)) for q in bench.queries]
        bounds = np.cumsum([0] + [len(V) for V in mats])
        scores, groups = methods(np.vstack(mats), rescaled=True)
        acc = {
            k: 100 * float(np.mean([selection_accuracy(s[bounds[i]:bounds[i + 1]], y) for i, y in enumerate(ys)]))
            for k, s in scores.items()
        }
        names = bench.verifiers
        detail = {
            "groups": {k: [[names[i] for i in g] for g in gs if len(g) > 1] for k, gs in groups.items()},
            "columns": {k: len(gs) for k, gs in groups.items()},
        }
    print(f"## {name} {mode}  ({len(bench.queries)} queries, {len(bench.verifiers)} verifiers, {time.time() - t0:.0f}s)", flush=True)
    for k, v in acc.items():
        print(f"  {k:42s} {v:5.1f}", flush=True)
    return {"accuracy": acc, "m": len(bench.verifiers), "queries": len(bench.queries), **detail}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--settings", nargs="*", default=SETTINGS)
    p.add_argument("--modes", nargs="*", default=["batch", "query"])
    p.add_argument("--cache", type=Path, default=ROOT / ".cache" / "benchmarks")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", type=Path, default=ROOT / "results" / "paper_blocks.json")
    args = p.parse_args()
    results = json.loads(args.out.read_text()) if args.out.exists() else {}
    for name in args.settings:
        bench = load(name, args.cache)
        for mode in args.modes:
            key = f"{name}|{mode}"
            if key in results:
                continue
            results[key] = run(name, bench, mode, args.workers)
            args.out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
