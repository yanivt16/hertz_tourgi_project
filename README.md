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
.venv/bin/python -m pytest                                          # 10 tests on synthetic data
.venv/bin/python scripts/run_experiment.py imo --mode query
.venv/bin/python scripts/run_experiment.py weaver --benchmark gpqa --size 70b --mode batch
.venv/bin/python scripts/summarize.py                               # writes results/SUMMARY.md
```

Data is pulled from HuggingFace: `FUSE-verifiers/HLE-Verifications`,
`FUSE-verifiers/IMO-Shortlist-Verifications`, and the Weaver datasets `hazyresearch/*`
(only the score columns are read, never the response text).

## Results

Best-of-N selection accuracy (%). Full per-method tables are in `results/*.log`.

| Benchmark | Pass@1 | Naive Ens. | Oracle Best Verifier | Logistic (5% labels) | FUSE query | FUSE batch | FUSE paper | Weaver paper | Pass@k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gpqa-70b | 42.9 | 59.3 | 59.1 | 73.7 | 59.6 | **65.3** | 66.8 | 66.4 | 81.0 |
| gpqa-diamond-70b | 42.3 | 56.6 | 58.1 | 68.5 | 55.6 | **60.6** | 64.4 | 64.1 | 75.3 |
| math500-70b | 78.0 | 93.0 | 87.2 | 97.0 | 90.8 | **94.2** | 92.8 | 93.4 | 98.6 |
| mmlu-70b | 82.6 | 93.0 | 87.6 | 94.3 | 92.8 | **93.6** | 94.1 | 94.9 | 96.0 |
| mmlu-pro-70b | 69.9 | 87.2 | 82.3 | 92.0 | 87.8 | **90.0** | 91.4 | 90.2 | 92.0 |
| gpqa-8b | 28.3 | 39.8 | 40.0 | 37.2 | 39.9 | **40.2** | 40.5 | 47.1 | 95.2 |
| gpqa-diamond-8b | 28.3 | 44.4 | 44.0 | 34.6 | **41.9** | 37.2 | 42.7 | 46.5 | 94.9 |
| math500-8b | 49.9 | 73.8 | 82.8 | 91.2 | **70.0** | 64.0 | 83.9 | 74.8 | 99.2 |
| mmlu-8b | 64.1 | 82.8 | 83.6 | 89.2 | **80.8** | 70.7 | 84.9 | 85.7 | 98.5 |
| mmlu-pro-8b | 46.6 | 68.4 | 66.5 | 71.5 | **68.6** | 51.8 | 69.8 | 67.2 | 96.8 |
| hle | 51.4 | 51.1 | 53.5 | 52.2 | **53.7** | 52.2 | 54.3 | 51.2 | 100.0 |
| imo | 53.2 | 62.1 | 59.6 | 57.9 | 57.1 | **62.1** | 63.8 | 62.1 | 85.4 |

**What reproduces.** Pass@1, Pass@k and the oracle best verifier match the paper on the
Weaver benchmarks and on IMO, which validates the data loading and evaluation. HLE's Pass@1
is 0.7 points below the paper's. FUSE lands within about 1.5 points of the paper on 8 of
the 12 settings (bold = the mode closer to the paper).

**What does not.** MATH500-8B (70.0 vs 83.9), GPQA-Diamond-70B (60.6 vs 64.4) and MMLU-8B
(80.8 vs 84.9) remain below the paper.

**Query vs batched mode matters and the paper does not say which it used per table.**
With 35 verifiers and 100 responses per question (70B), per-question third-order moment
estimates are too noisy, and pooling questions helps. With 19 verifiers (8B), pooling
questions of very different difficulty breaks conditional independence and hurts.

## Choices the paper leaves open

These are the likely sources of the remaining gaps, and each is configurable on `FUSE`:

- **Threshold candidates**: a 19-point quantile grid, or midpoints for discrete scores, searched by 3 sweeps of coordinate descent (`n_quantiles`, `n_sweeps`).
- **After dropping verifiers**: re-estimate on the kept subset (`refit_after_drop`).
- **Surrogate for eq. 7**: weighted logistic regression with labels `p > 0.5` and weights `|2p − 1|` (`predictor`, `logistic_C`). `predictor="posterior"` ranks by p̂ directly.
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
