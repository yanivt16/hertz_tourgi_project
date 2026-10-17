"""Best-of-N evaluation of FUSE and baselines on a `Benchmark` (paper Section 3)."""

from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict

import numpy as np

from . import baselines as bl
from .data import Benchmark
from .ensemble import FUSE, fuse_batched
from .metrics import selection_accuracy
from .preprocess import impute_missing

# Accuracies (%) reported in the paper, used only for side-by-side comparison.
PAPER_RESULTS = {
    "gpqa-70b": {"Pass@1": 42.9, "Majority Vote": 47.4, "Naive Ensemble": 60.1, "FUSE": 66.8, "Oracle Best Verifier": 59.1, "Logistic Regression": 69.3, "Naive Bayes": 60.5, "Pass@k": 81.0},
    "math500-70b": {"Pass@1": 78.0, "Majority Vote": 82.4, "Naive Ensemble": 92.4, "FUSE": 92.8, "Oracle Best Verifier": 87.2, "Logistic Regression": 96.2, "Naive Bayes": 94.4, "Pass@k": 98.6},
    "mmlu-70b": {"Pass@1": 82.6, "Majority Vote": 84.1, "Naive Ensemble": 92.4, "FUSE": 94.1, "Oracle Best Verifier": 88.4, "Logistic Regression": 93.1, "Naive Bayes": 93.7, "Pass@k": 96.0},
    "mmlu-pro-70b": {"Pass@1": 69.9, "Majority Vote": 74.4, "Naive Ensemble": 87.0, "FUSE": 91.4, "Oracle Best Verifier": 85.6, "Logistic Regression": 91.8, "Naive Bayes": 91.4, "Pass@k": 92.0},
    "gpqa-8b": {"Pass@1": 28.3, "Majority Vote": 30.5, "Naive Ensemble": 37.8, "FUSE": 40.5, "Oracle Best Verifier": 41.9, "Logistic Regression": 35.9, "Naive Bayes": 37.6, "Pass@k": 95.2},
    "math500-8b": {"Pass@1": 49.9, "Majority Vote": 69.6, "Naive Ensemble": 75.2, "FUSE": 83.9, "Oracle Best Verifier": 82.8, "Logistic Regression": 88.0, "Naive Bayes": 80.8, "Pass@k": 99.2},
    "mmlu-8b": {"Pass@1": 64.1, "Majority Vote": 72.7, "Naive Ensemble": 81.9, "FUSE": 84.9, "Oracle Best Verifier": 83.6, "Logistic Regression": 80.4, "Naive Bayes": 77.8, "Pass@k": 98.5},
    "mmlu-pro-8b": {"Pass@1": 46.6, "Majority Vote": 56.4, "Naive Ensemble": 67.2, "FUSE": 69.8, "Oracle Best Verifier": 66.5, "Logistic Regression": 65.2, "Naive Bayes": 66.0, "Pass@k": 96.8},
    "gpqa-diamond-70b": {"Pass@1": 42.3, "Majority Vote": 49.5, "Naive Ensemble": 57.6, "FUSE": 64.4, "Oracle Best Verifier": 50.8, "Logistic Regression": 63.3, "Naive Bayes": 57.1, "Pass@k": 75.3},
    "gpqa-diamond-8b": {"Pass@1": 28.3, "Majority Vote": 32.3, "Naive Ensemble": 38.4, "FUSE": 42.7, "Oracle Best Verifier": 40.6, "Logistic Regression": 34.3, "Naive Bayes": 37.5, "Pass@k": 95.0},
    "hle": {"Pass@1": 52.1, "Naive Ensemble": 51.4, "Oracle Best Verifier": 53.5, "Naive Bayes": 52.0, "Logistic Regression": 53.4, "FUSE": 54.3},
    "imo": {"Pass@1": 53.3, "Majority Vote": 57.7, "Naive Ensemble": 63.8, "Oracle Best Verifier": 59.7, "Naive Bayes": 59.1, "Logistic Regression": 60.2, "FUSE": 63.8},
}


PAPER_WEAVER = {
    "gpqa-70b": 66.4, "gpqa-diamond-70b": 64.1, "math500-70b": 93.4, "mmlu-70b": 94.9, "mmlu-pro-70b": 90.2,
    "gpqa-8b": 47.1, "gpqa-diamond-8b": 46.5, "math500-8b": 74.8, "mmlu-8b": 85.7, "mmlu-pro-8b": 67.2,
    "hle": 51.2, "imo": 62.1,
}
for _name, _acc in PAPER_WEAVER.items():
    PAPER_RESULTS[_name]["Weaver"] = _acc


def _fuse_one(args):
    V, config = args
    return FUSE(**config).fit_scores(V).scores


def has_answers(bench: Benchmark) -> bool:
    return all(q.answers is not None and len(q.answers) == len(q.correct) for q in bench.queries)


def with_majority_vote(mats: list[np.ndarray], bench: Benchmark) -> list[np.ndarray]:
    """Append answer agreement (how many responses share each response's extracted answer)
    as one more verifier column. It needs no model, and its errors come from the generator
    rather than from any reward model, so it is plausibly independent of RM nuisance factors."""
    return [
        np.column_stack([V, bl.majority_vote_scores(q.answers)]) for V, q in zip(mats, bench.queries)
    ]


def run_unsupervised(
    bench: Benchmark,
    fuse: FUSE,
    mode: str = "query",
    workers: int = 0,
    majority_vote_verifier: bool = False,
) -> dict[str, float]:
    mats = [impute_missing(q.scores) for q in bench.queries]
    ys = [q.correct for q in bench.queries]
    out: dict[str, list[float]] = {
        "Pass@1": [y.mean() for y in ys],
        "Pass@k": [float(y.any()) for y in ys],
        "Naive Ensemble": [selection_accuracy(bl.naive_ensemble_scores(V), y) for V, y in zip(mats, ys)],
        "Jaffe et al. (2015)": [selection_accuracy(bl.jaffe2015_scores(V), y) for V, y in zip(mats, ys)],
        "Dawid-Skene": [selection_accuracy(bl.dawid_skene_scores(V), y) for V, y in zip(mats, ys)],
    }
    if has_answers(bench):
        out["Majority Vote"] = [
            selection_accuracy(bl.majority_vote_scores(q.answers), q.correct) for q in bench.queries
        ]

    if all(q.weaver_scores is not None for q in bench.queries):
        # Weaver's own released output (Saad-Falcon et al., best configuration), read from
        # the dataset rather than re-run.
        out["Weaver"] = [
            selection_accuracy(np.nan_to_num(q.weaver_scores, nan=-np.inf), q.correct) for q in bench.queries
        ]

    if majority_vote_verifier:
        if not has_answers(bench):
            raise ValueError(f"{bench.name} has no extracted answers for a majority-vote verifier")
        mats = with_majority_vote(mats, bench)

    if mode == "batch":
        scores, _ = fuse_batched(mats, fuse)
    elif workers and workers > 1:
        config = asdict(fuse)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            scores = list(pool.map(_fuse_one, [(V, config) for V in mats], chunksize=4))
    else:
        scores = [fuse.fit_scores(V).scores for V in mats]
    out["FUSE"] = [selection_accuracy(s, y) for s, y in zip(scores, ys)]
    return {k: 100 * float(np.mean(v)) for k, v in out.items()}


def run_supervised(bench: Benchmark, label_frac: float = 0.05, seeds: int = 5) -> dict[str, float]:
    """Semi-supervised baselines get full labels for label_frac of queries and are scored
    on the remaining queries; results are averaged over random label splits."""
    mats = [impute_missing(q.scores) for q in bench.queries]
    ys = [q.correct for q in bench.queries]
    n = len(mats)
    n_lab = max(1, math.ceil(label_frac * n))
    res: dict[str, list[float]] = {"Logistic Regression": [], "Naive Bayes": []}
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        lab = rng.choice(n, n_lab, replace=False)
        test = np.setdiff1d(np.arange(n), lab)
        train_V, train_y = [mats[i] for i in lab], [ys[i] for i in lab]
        lr = bl.SupervisedLogistic().fit(train_V, train_y)
        nb = bl.SupervisedNaiveBayes().fit(train_V, train_y)
        res["Logistic Regression"].append(np.mean([selection_accuracy(lr.scores(mats[i]), ys[i]) for i in test]))
        res["Naive Bayes"].append(np.mean([selection_accuracy(nb.scores(mats[i]), ys[i]) for i in test]))
    best = bl.oracle_best_verifier(mats, ys)
    out = {k: 100 * float(np.mean(v)) for k, v in res.items()}
    out["Oracle Best Verifier"] = 100 * float(
        np.mean([selection_accuracy(bl.minmax_to_unit_interval(V)[:, best], y) for V, y in zip(mats, ys)])
    )
    out["_best_verifier"] = bench.verifiers[best]
    return out


def format_table(name: str, results: dict[str, float]) -> str:
    paper = PAPER_RESULTS.get(name, {})
    order = [
        "Pass@1", "Majority Vote", "Naive Ensemble", "Dawid-Skene", "Jaffe et al. (2015)",
        "FUSE", "Oracle Best Verifier", "Weaver", "Logistic Regression", "Naive Bayes", "Pass@k",
    ]
    lines = ["| Method | Ours | Paper |", "|---|---:|---:|"]
    for k in order:
        if k in results or k in paper:
            ours, p = results.get(k), paper.get(k)
            lines.append(f"| {k} | {'' if ours is None else f'{ours:.1f}'} | {'' if p is None else f'{p:.1f}'} |")
    return "\n".join(lines)
