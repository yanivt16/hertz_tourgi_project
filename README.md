# FUSE — Ensembling Verifiers with Zero Labeled Data

A from-scratch reimplementation of **FUSE (Fully Unsupervised Score Ensembling)**,
Lee, Ma, Zhao et al., [arXiv:2604.18547](https://arxiv.org/abs/2604.18547), with a
reproduction of the paper's Best-of-N experiments.

The authors have not released code (the paper says the pipeline "will be open-sourced").
Everything here is built from the paper's equations and appendices. They did release
their data, and this repo runs on it directly.

## What FUSE does

Given N candidate responses to a question and m imperfect verifiers (LLM judges, reward
models), pick the response most likely to be correct **without any ground-truth labels**.

| Step | Paper | Code |
|---|---|---|
| 1. Binarise each verifier at the threshold that minimises a label-free measure of triplet-conditional-independence violation | §2.2, Prop. 2.4 | `src/fuse/tci.py` |
| 2. Estimate each verifier's sensitivity/specificity and the class balance by method of moments | §2.1, Thm. 2.3 (Jaffe et al., 2015) | `src/fuse/moments.py` |
| 3. Drop verifiers estimated worse than random | App. D | `src/fuse/ensemble.py` |
| 4. Posterior per response, averaged over all verifier triplets | App. C, eq. 9 | `src/fuse/posterior.py` |
| 5. Fit logistic regression to the pseudo-labels, maximising estimated accuracy; select argmax | eq. 7 | `src/fuse/ensemble.py` |

Baselines (`src/fuse/baselines.py`): Pass@1, Pass@k, majority vote, naive ensemble,
Jaffe et al. (2015), Dawid–Skene, oracle best verifier, and supervised logistic regression
and naive Bayes trained on 5% of questions.

## Usage

```python
from fuse import FUSE

result = FUSE().fit_scores(V)   # V: (N responses x m verifiers), no labels
best = result.select()          # index of the chosen response
```

Batched mode pools many questions into one ensemble: `fuse_batched([V1, V2, ...])`.

## Reproducing the paper

```bash
python -m venv .venv
.venv/bin/pip install -e ".[data,dev]"
.venv/bin/python -m pytest                                          # 11 tests on synthetic data
.venv/bin/python scripts/run_experiment.py imo --mode query
.venv/bin/python scripts/run_experiment.py weaver --benchmark gpqa --size 70b --mode batch
.venv/bin/python scripts/summarize.py                               # writes results/SUMMARY.md
```

Data is pulled from HuggingFace: `FUSE-verifiers/HLE-Verifications`,
`FUSE-verifiers/IMO-Shortlist-Verifications`, and the Weaver datasets `hazyresearch/*`
(only the score columns are read, never the response text).

## Results

Best-of-N selection accuracy (%). Every run's full per-method table is in `results/all_runs.log`.
Weaver-data settings use exactly the verifiers of the paper's Table 6 (16 for 8B, 32 for 70B).

| Benchmark | Pass@1 | Naive Ens. | Oracle Best Verifier | Logistic (5% labels) | FUSE query | FUSE batch | FUSE batch + posterior | FUSE paper | Weaver paper | Pass@k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gpqa-70b | 42.9 | 59.9 | 59.1 | 73.7 | 60.5 | 66.9 | 67.0 | 66.8 | 66.4 | 81.0 |
| gpqa-diamond-70b | 42.3 | 56.1 | 58.1 | 68.2 | 55.1 | 59.6 | 64.4 | 64.4 | 64.1 | 75.3 |
| math500-70b | 78.0 | 92.4 | 87.2 | 97.1 | 91.2 | 94.1 | 91.3 | 92.8 | 93.4 | 98.6 |
| mmlu-70b | 82.6 | 93.2 | 87.6 | 94.3 | 92.8 | 93.7 | 93.9 | 94.1 | 94.9 | 96.0 |
| mmlu-pro-70b | 69.9 | 87.0 | 82.3 | 92.0 | 88.2 | 91.6 | 90.4 | 91.4 | 90.2 | 92.0 |
| gpqa-8b | 28.3 | 38.9 | 40.0 | 37.0 | 38.7 | 37.9 | 37.3 | 40.5 | 47.1 | 95.2 |
| gpqa-diamond-8b | 28.3 | 40.9 | 44.0 | 35.2 | 40.9 | 31.7 | 42.9 | 42.7 | 46.5 | 94.9 |
| math500-8b | 49.9 | 74.6 | 82.8 | 89.9 | 66.0 | 55.2 | 60.2 | 83.9 | 74.8 | 99.2 |
| mmlu-8b | 64.1 | 82.9 | 83.6 | 89.1 | 80.5 | 70.6 | 86.7 | 84.9 | 85.7 | 98.5 |
| mmlu-pro-8b | 46.6 | 68.6 | 66.5 | 70.7 | 69.0 | 60.3 | 61.5 | 69.8 | 67.2 | 96.8 |
| hle | 51.4 | 51.1 | 53.5 | 52.2 | 53.7 | 52.2 | 52.1 | 54.3 | 51.2 | 100.0 |
| imo | 53.2 | 62.1 | 59.6 | 57.9 | 57.1 | 62.1 | 60.8 | 63.8 | 62.1 | 85.4 |

**What reproduces.** Pass@1, Pass@k and the oracle best verifier match the paper on the
Weaver benchmarks and on IMO, which validates the data and evaluation. HLE's Pass@1 is
0.7 points below the paper's.

- **70B, batched mode**: GPQA, MATH500, MMLU and MMLU-Pro within 1.5 points of the paper.
  GPQA-Diamond-70B needs the posterior as the final predictor (last FUSE column), which
  matches exactly (64.4 vs 64.4) and keeps the other 70B settings within 1.5 points.
- **8B, per-question mode**: GPQA, GPQA-Diamond and MMLU-Pro within 2 points. MMLU-8B
  reaches 86.7 with batch + posterior (paper 84.9).
- **HLE** 53.7 vs 54.3 (per-question); **IMO** 62.1 vs 63.8 (batched).

No single configuration wins everywhere. The paper doesn't state which mode or final
predictor it used per table, and the numbers above are reported per configuration, not
tuned per benchmark.

**What does not reproduce: MATH500-8B** (best 66.0 vs 83.9). Nine of its 16 reward models
are near-random (selection accuracy about 54 against Pass@1's 50) but strongly
correlated with each other (|corr| ≈ 0.5). The two best verifiers, QwenPRM min and avg,
are weakly correlated with everything. The moment estimator's rank-one fit locks onto the
correlated block and reads it as the correctness signal. Estimated on all questions at
once, it even ranks the junk block best (0.86–0.93) and QwenPRM worst (0.50). A shared
factor unrelated to correctness (response length, for example) is indistinguishable from
the label to a label-free moment method, and the TCI statistic cannot flag it because a
single shared factor is itself nearly rank-one. Nothing the paper describes handles this.

**Things tried that did not help** (evaluated on all settings, not just the failing ones):
a scale-free TCI statistic (`tci_standardize=True`; it stops the threshold search pushing
70–77% of verifiers to near-constant splits, but lowers accuracy overall), stronger
logistic regularisation (`logistic_C=0.1`), global instead of per-question rescaling in
batched mode, and deciding verifier drops from the pooled matrix.

## Extensions beyond the paper: verifier clustering and a majority-vote verifier

Two label-free additions aimed at the MATH500-8B failure, both off by default:

- **Verifier clustering** (`FUSE(cluster_threshold=t)`, `src/fuse/clustering.py`): average-linkage
  clustering on 1 − |Spearman correlation|; each cluster is averaged (sign-aligned) into one
  meta-verifier before the threshold search, so a correlated block gets one vote in the
  moment estimate. The logistic predictor still sees every original verifier.
- **Majority-vote verifier** (`run_unsupervised(..., majority_vote_verifier=True)`): answer
  agreement appended as one more verifier column. HLE has no extracted answers, so it is skipped there.

`scripts/run_variants.py` runs every combination (thresholds 0.3/0.5/0.7, both modes, both
predictors) and `scripts/summarize_variants.py` writes [results/VARIANTS.md](results/VARIANTS.md).
The threshold 0.5 was fixed as the default before the grid ran.

**Result: clustering fixes the failure it targets but is not a better default.**

| Setting (per-question, logistic) | base | cluster@0.5 | cluster@0.5 + MV | FUSE paper |
|---|---:|---:|---:|---:|
| math500-8b | 66.0 | **79.3** | 78.5 | 83.9 |
| mmlu-8b | 80.5 | 82.6 | 83.2 | 84.9 |
| imo | 57.1 | 57.2 | **61.9** | 63.8 |
| gpqa-70b | 60.5 | 58.0 | 57.2 | 66.8 |
| gpqa-diamond-70b | 55.1 | 53.5 | 53.3 | 64.4 |
| mean of 11 settings with answers | 67.3 | 67.9 | 68.5 | |

- MATH500-8B gains 13 points (per-question) and up to 20 (batched, threshold 0.3: 55.2 → 75.1).
  Batched MMLU-8B and MMLU-Pro-8B gain up to 13 and 10 points.
- GPQA-70B/Diamond-70B lose 2–10 points depending on configuration. The 70B sets include
  eight LM-judge verdicts that agree with each other *because they are right*. Correlation
  alone cannot tell that block apart from one that shares a nuisance factor, so clustering
  collapses informative judges as well as junk.
- The majority-vote verifier helps IMO (+4–5) and slightly helps the 8B settings. Elsewhere it is roughly neutral.
- Averaged over settings, no variant beats the base by more than about 2 points in any
  mode/predictor combination, and some lose up to 3.6. Batched + logistic gains most
  (64.7 → 66.3 over all 12 at threshold 0.5).

The next step is therefore choosing between variants per setting without labels, not a new default.

## Choices the paper leaves open

These are the likely sources of the remaining gaps, and each is configurable on `FUSE`:

- **Threshold candidates**: a 19-point quantile grid, or midpoints for discrete scores, searched by 3 sweeps of coordinate descent (`n_quantiles`, `n_sweeps`).
- **After dropping verifiers**: re-estimate on the kept subset (`refit_after_drop`).
- **Surrogate for eq. 7**: weighted logistic regression with labels `p > 0.5` and weights `|2p − 1|` (`predictor`, `logistic_C`). `predictor="posterior"` ranks by p̂ directly.
- **Verifier sets**: Table 6 of the paper (`paper_verifiers=True` in `download_weaver`); the datasets carry 3–4 extra score columns per setting.
- **Rescaling**: per-question min-max to [−1, 1]; missing scores imputed as 0 before rescaling.
- **Supervised baselines**: evaluated on the 95% of questions they were not trained on, averaged over 5 splits.

## About the Weaver baseline

The paper produced its Weaver numbers by running
[HazyResearch/scaling-verification](https://github.com/HazyResearch/scaling-verification).
Its `selection/run.py` depends on torch, hydra and the `metal-ama` weak-supervision
library, and was not run here. The table quotes the paper's Weaver numbers. The
`weaver_scores` column shipped in the Weaver datasets gives much lower accuracy (e.g.
43.4 on GPQA-70B), so it evidently comes from a different configuration and is not used
for comparison.
