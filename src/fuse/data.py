"""Loaders for the three data sources used in the paper.

* FUSE-verifiers/HLE-Verifications         649 HLE questions, 50 Gemini 3 Pro responses, 7 verifiers
* FUSE-verifiers/IMO-Shortlist-Verifications 123 IMO questions, 50 Qwen3-30B responses, 9 verifiers
* hazyresearch/<BENCH>_with_Llama_3.1_<SIZE>_Instruct_v1  Weaver data (Saad-Falcon et al., 2025)

Everything is reduced to a `Benchmark`: one (N x m) score matrix per query plus the
ground-truth correctness vector, which is all FUSE and the baselines consume.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FUSE_DATASETS = {
    "hle": ("FUSE-verifiers/HLE-Verifications", "FUSE-hle-data.jsonl"),
    "imo": ("FUSE-verifiers/IMO-Shortlist-Verifications", "FUSE-imo-shortlist-data.jsonl"),
}

# Repo names copied from DATASET_TO_HF in HazyResearch/scaling-verification (weaver/constants.py).
# GPQA Diamond is not its own dataset: it is the GPQA rows whose `type` is "diamond".
WEAVER_REPOS = {
    ("gpqa", "8b"): "hazyresearch/GPQA_with_Llama_3.1_8B_Instruct_v1",
    ("gpqa", "70b"): "hazyresearch/GPQA_with_Llama_3.1_70B_Instruct_v1",
    ("math500", "8b"): "hazyresearch/MATH-500_with_Llama_3.1_8B_Instruct_v1",
    ("math500", "70b"): "hazyresearch/MATH500_with_Llama_3.1_70B_Instruct_v1",
    ("mmlu", "8b"): "hazyresearch/MMLU_with_Llama_3.1_8B_Instruct_v1",
    ("mmlu", "70b"): "hazyresearch/MMLU_with_Llama_3.1_70B_Instruct_v1",
    ("mmlu-pro", "8b"): "hazyresearch/MMLU-Pro_with_Llama_3.1_8B_Instruct_v1",
    ("mmlu-pro", "70b"): "hazyresearch/MMLU-Pro_with_Llama_3.1_70B_Instruct_v1",
}
WEAVER_BENCHMARKS = ["gpqa", "gpqa-diamond", "math500", "mmlu", "mmlu-pro"]

# Paper Table 6: the verifiers used for each generator size. The datasets carry a few more
# score columns (e.g. EurusPRMStage1_min for 8B, GRMLlama32 for 70B) that the paper excludes.
_BOTH = [
    "ArmorRM_scores", "EurusPRMStage1_max_scores", "EurusPRMStage2_min_scores", "GPM_scores",
    "GRMGemma_scores", "GRM_scores", "InternLM2Reward7B_scores", "OffsetBias_scores",
    "QRM_scores", "QwenPRM_avg_scores", "QwenPRM_min_scores", "Skyworks_scores", "URM_scores",
]
PAPER_VERIFIERS = {
    "8b": _BOTH + ["EurusPRMStage1_avg_scores", "EurusPRMStage2_max_scores", "GRMLlama32_scores"],
    "70b": _BOTH + [
        "DecisionTreeReward27B_scores", "DecisionTreeReward8B_scores", "EurusPRMStage1_min_scores",
        "EurusPRMStage2_avg_scores", "INFORM_scores", "InternLM2RewardModel_scores",
        "LDLRewardGemma_scores", "QRMGemma_scores", "Qwen72B_scores", "QwenPRM_max_scores",
        "SkyworksGemma_scores",
        "DeepSeekLlama70B_verdicts", "DeepSeekQwen32B_verdicts", "Llama-3.3-70B-Instruct_verdicts",
        "Meta-Llama-3.1-405B-Instruct-quantized.w8a16_verdicts", "Mixtral-8x22B-Instruct-v0.1_verdicts",
        "Qwen/Qwen2.5-72B-Instruct_verdicts", "SkyT1_verdicts", "WizardLM-2-8x22B_verdicts",
    ],
}


@dataclass
class Query:
    qid: str
    scores: np.ndarray  # (N, m), NaN where a verifier produced no score
    correct: np.ndarray  # (N,) bool
    answers: list | None = None
    weaver_scores: np.ndarray | None = None  # Weaver's released selection scores, when present


@dataclass
class Benchmark:
    name: str
    verifiers: list[str]
    queries: list[Query]

    def filter_verifiers(self, keep: list[str]) -> "Benchmark":
        idx = [self.verifiers.index(v) for v in keep]
        return Benchmark(
            name=self.name,
            verifiers=keep,
            queries=[Query(q.qid, q.scores[:, idx], q.correct, q.answers, q.weaver_scores) for q in self.queries],
        )

    def summary(self) -> str:
        n = [len(q.correct) for q in self.queries]
        acc = np.mean([q.correct.mean() for q in self.queries])
        return (
            f"{self.name}: {len(self.queries)} queries, {min(n)}-{max(n)} responses, "
            f"{len(self.verifiers)} verifiers, Pass@1={acc:.3f}"
        )


def _to_float(v) -> float:
    """Scores arrive as scalars or as lists of repeated verifications ([4.0], [], [None])."""
    if isinstance(v, (list, tuple)):
        vals = [float(x) for x in v if x is not None]
        return float(np.mean(vals)) if vals else np.nan
    return np.nan if v is None else float(v)


def _to_float_array(values) -> np.ndarray:
    return np.array([_to_float(v) for v in values], dtype=float)


def rows_to_benchmark(name: str, rows: list[dict], correct_key: str) -> Benchmark:
    verifier_cols = sorted(
        k
        for k in rows[0]
        if (k.endswith("_scores") or k.endswith("_verdicts")) and k != "weaver_scores"
    )
    queries = []
    for i, row in enumerate(rows):
        correct = np.array([bool(c) for c in row[correct_key]])
        n = len(correct)
        cols = []
        for k in verifier_cols:
            vals = row.get(k)
            col = _to_float_array(vals) if vals is not None else np.full(n, np.nan)
            if len(col) != n:
                col = np.resize(col, n) if len(col) else np.full(n, np.nan)
            cols.append(col)
        answers = row.get("extracted_answers")
        weaver = row.get("weaver_scores")
        queries.append(
            Query(
                qid=str(row.get("id", i)),
                scores=np.column_stack(cols),
                correct=correct,
                answers=list(answers) if answers is not None else None,
                weaver_scores=_to_float_array(weaver) if weaver is not None and len(weaver) == n else None,
            )
        )
    return Benchmark(name=name, verifiers=verifier_cols, queries=queries)


def load_fuse_jsonl(path: str | Path, name: str) -> Benchmark:
    keep_suffixes = ("_scores", "is_correct", "extracted_answers", "id")
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                rows.append({k: v for k, v in row.items() if k.endswith(keep_suffixes)})
    return rows_to_benchmark(name, rows, correct_key="is_correct")


def download_fuse(key: str, cache_dir: str | Path | None = None) -> Benchmark:
    from huggingface_hub import hf_hub_download

    repo_id, filename = FUSE_DATASETS[key]
    path = hf_hub_download(repo_id, filename, repo_type="dataset", cache_dir=cache_dir)
    return load_fuse_jsonl(path, name=key)


def download_weaver(
    benchmark: str, size: str, include_verdicts: bool | None = None, paper_verifiers: bool = True
) -> Benchmark:
    """Read only the score and label columns from the auto-converted parquet shards, so the
    (very large) response text is never downloaded.

    For the 8B generator the paper excludes the binary LM judges (Appendix E.2). The
    `weaver_scores` column is Weaver's own output and per-step PRM lists are not verifiers
    in the paper's Table 6, so both are always excluded.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    if include_verdicts is None:
        include_verdicts = size.lower() == "70b"
    size = size.lower()
    diamond = benchmark == "gpqa-diamond"
    repo = WEAVER_REPOS[("gpqa" if diamond else benchmark, size)]
    fs = HfFileSystem()
    root = f"datasets/{repo}@~parquet"
    shards = sorted(p for p in fs.glob(f"{root}/**/*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no parquet shards under {root}")
    schema = pq.read_schema(shards[0], filesystem=fs)
    cols = [
        c
        for c in schema.names
        if c in ("answer_correct", "extracted_answers", "type", "weaver_scores")
        or (c.endswith("_scores") and not c.endswith("_step_scores"))
        or (include_verdicts and c.endswith("_verdicts"))
    ]
    rows = []
    for shard in shards:
        rows.extend(pq.read_table(shard, columns=cols, filesystem=fs).to_pylist())
    if diamond:
        rows = [r for r in rows if r.get("type") == "diamond"]
    bench = rows_to_benchmark(f"{benchmark}-{size}", rows, correct_key="answer_correct")
    if paper_verifiers:
        bench = bench.filter_verifiers([v for v in bench.verifiers if v in PAPER_VERIFIERS[size]])
    return bench
