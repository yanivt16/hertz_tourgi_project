# FUSE — Ensembling Verifiers with Zero Labeled Data

A from-scratch reimplementation of **FUSE (Fully Unsupervised Score Ensembling)**,
Lee, Ma, Zhao et al., [arXiv:2604.18547](https://arxiv.org/abs/2604.18547), with a
reproduction of the paper's Best-of-N experiments.

The authors have not released code (the paper says the pipeline "will be open-sourced").
Everything here is built from the paper's equations and appendices. They did release
their data, and this repo runs on it directly.

## Deliverables

| Deliverable | Where |
|---|---|
| FUSE implementation | [src/fuse/](src/fuse/) (usage under [Usage](#usage)), tests in [tests/test_fuse.py](tests/test_fuse.py) |
| Paper reproduction, 12 Best-of-N settings | Table: [results/SUMMARY.md](results/SUMMARY.md) · every method per setting: [results/all_runs.log](results/all_runs.log) · raw numbers: `results/<setting>_<mode>_<predictor>.json` · script: [scripts/run_experiment.py](scripts/run_experiment.py) · write-up: [Results](#results) |
| Extensions: verifier clustering and majority-vote verifier | Table: [results/VARIANTS.md](results/VARIANTS.md) · raw: [results/variants_sweep.json](results/variants_sweep.json), [results/variants.log](results/variants.log) · script: [scripts/run_variants.py](scripts/run_variants.py) · write-up: [Extensions](#extensions-beyond-the-paper-verifier-clustering-and-a-majority-vote-verifier) |
| Label-free choice between those variants per setting | Table: [results/SELECTION.md](results/SELECTION.md) · raw: [results/selection.json](results/selection.json) · code: [src/fuse/selection.py](src/fuse/selection.py), [scripts/select_variants.py](scripts/select_variants.py) · write-up: [Choosing a variant](#choosing-a-variant-per-setting-without-labels) |
| FUSE on DICES (human safety raters as verifiers) | Tables: [results/DICES.md](results/DICES.md) · raw: [results/dices350.json](results/dices350.json), [results/dices.log](results/dices.log) · code: [src/fuse/dices.py](src/fuse/dices.py), [scripts/run_dices.py](scripts/run_dices.py) · write-up: [DICES](#dices-human-safety-raters-as-verifiers) |
| DICES error overlap: do the methods miss the same conversations? | Tables: [results/DICES_ERRORS.md](results/DICES_ERRORS.md) · raw (incl. the missed conversation ids): [results/dices350_errors.json](results/dices350_errors.json) · script: [scripts/dices_error_overlap.py](scripts/dices_error_overlap.py) |
| LLM safety label for DICES: prompt development on DICES-350, held-out test, labels for all of DICES-990 | Write-up: [results/DICES_LLM_LABELS.md](results/DICES_LLM_LABELS.md) · labels and run metadata: [results/llm_labels/](results/llm_labels/) · prompts: [scripts/dices_prompts/](scripts/dices_prompts/) · script: [scripts/label_dices_llm.py](scripts/label_dices_llm.py) |
| FUSE on DICES-990 against the LLM label (missing ratings) | Tables: [results/DICES990.md](results/DICES990.md) · raw: [results/dices990.json](results/dices990.json), [results/dices990.log](results/dices990.log) · script: [scripts/run_dices990.py](scripts/run_dices990.py) · write-up: [DICES-990](#dices-990-against-an-llm-safety-label) |
| FUSE on DICES-990 block by block (conversations grouped by shared raters) | Tables: [results/DICES990_BLOCKS.md](results/DICES990_BLOCKS.md) · raw (incl. each conversation's block): [results/dices990_blocks.json](results/dices990_blocks.json) · script: [scripts/dices990_blocks.py](scripts/dices990_blocks.py) |
| FUSE on VIPER (radiologists reading mammograms, gold = cancer at follow-up); missingness leaks the label, so dense blocks are the clean result | Tables: [results/VIPER.md](results/VIPER.md) · code: [src/fuse/viper.py](src/fuse/viper.py), [scripts/run_viper.py](scripts/run_viper.py) |
| Stage 1 dataset survey and recommendation (RTE, TEMP, D_PosSent, D_Product, Bluebird, VIPER) | [docs/dataset_survey.md](docs/dataset_survey.md) |
| Missing ratings in FUSE: problem, missing-aware FUSE (`FUSE(missing="skip")`), first results | Plan and results: [docs/missing_data_plan.md](docs/missing_data_plan.md) · code: [src/fuse/missing.py](src/fuse/missing.py), tests: [tests/test_missing.py](tests/test_missing.py) · simulation: [results/MISSING_SIM.md](results/MISSING_SIM.md), [scripts/sim_missing.py](scripts/sim_missing.py) · real data (RTE, D_PosSent, D_Product, Bluebird, DICES, VIPER): [results/MISSING_REAL.md](results/MISSING_REAL.md), [scripts/run_missing_real.py](scripts/run_missing_real.py), loader [src/fuse/crowd.py](src/fuse/crowd.py) |
| Does FUSE beat Dawid-Skene on continuous scores? Simulation (Gaussian raters, correlated junk raters) and real scores (VIPER dense blocks, DICES-350 views) against Dawid-Skene on binned scores | Write-up: [docs/continuous_scores.md](docs/continuous_scores.md) · tables: [results/CONTINUOUS.md](results/CONTINUOUS.md) · script: [scripts/run_continuous.py](scripts/run_continuous.py) · baseline: [src/fuse/ordinal.py](src/fuse/ordinal.py) |
| Pilot: humans + local LLM raters (Llama 3.1 8B, Qwen3 8B, Gemma 3 12B via MLX) on DICES-350, which aggregator handles mixed panels | Write-up: [docs/mixed_panel.md](docs/mixed_panel.md) · tables: [results/MIXED_PANEL.md](results/MIXED_PANEL.md) · LLM scores: [results/llm_scores/](results/llm_scores/) · scripts: [scripts/label_dices_local.py](scripts/label_dices_local.py), [scripts/run_mixed_panel.py](scripts/run_mixed_panel.py) |
| Second pilot: crowd + the same local LLMs on RTE, where LLMs are strong; do they help, and which aggregator uses them best | Write-up: [docs/mixed_panel_rte.md](docs/mixed_panel_rte.md) · tables: [results/MIXED_PANEL_RTE.md](results/MIXED_PANEL_RTE.md) · scripts: [scripts/label_rte_local.py](scripts/label_rte_local.py), [scripts/run_mixed_panel_rte.py](scripts/run_mixed_panel_rte.py) |
| Simulation map: in which regimes does FUSE beat Dawid-Skene (panel size, difficulty, quality spread, prevalence, scores vs votes, missing, correlated raters, many non-experts on a hard question) | Write-up: [docs/regime_map.md](docs/regime_map.md) · tables: [results/REGIME_MAP.md](results/REGIME_MAP.md) · script: [scripts/sim_regime_map.py](scripts/sim_regime_map.py) |
| Real non-expert crowds on hard questions (deepfake videos, news fact-checking): FUSE vs Dawid-Skene on individuals, demographic groups and random groups | Write-up: [docs/hard_crowds.md](docs/hard_crowds.md) · tables: [results/HARD_CROWDS.md](results/HARD_CROWDS.md) · script: [scripts/run_hard_crowds.py](scripts/run_hard_crowds.py) · loader: [src/fuse/hard_crowds.py](src/fuse/hard_crowds.py) |
| The regime where the simulation puts FUSE ahead, on real labels: 10 RTE human columns plus a correlated block of small Qwen models (and a check of the implementation) | Write-up: [docs/correlated_block_rte.md](docs/correlated_block_rte.md) · tables: [results/CORRELATED_BLOCK_RTE.md](results/CORRELATED_BLOCK_RTE.md) · script: [scripts/run_correlated_block_rte.py](scripts/run_correlated_block_rte.py) |
| Can a change to the threshold search keep FUSE's correlated-block win without its losses? (narrower search, scale-free statistic, clustering; simulation and all real panels) | Write-up: [docs/threshold_search_fixes.md](docs/threshold_search_fixes.md) · tables: [results/TCI_FIXES.md](results/TCI_FIXES.md) · scripts: [scripts/tci_fix_sim.py](scripts/tci_fix_sim.py), [scripts/tci_fix_real.py](scripts/tci_fix_real.py) |
| Finding raters who share errors without labels (residual correlation after a trimmed rank-one fit) and collapsing them before any aggregator, with Jaffe et al. (2016) as a baseline detector | Write-up: [docs/residual_blocks.md](docs/residual_blocks.md) · tables: [results/RESIDUAL_BLOCKS.md](results/RESIDUAL_BLOCKS.md), [results/JAFFE_BASELINE.md](results/JAFFE_BASELINE.md) · script: [scripts/run_residual_blocks.py](scripts/run_residual_blocks.py) |
| Candidate datasets for the next tests (per-annotator labels + independent gold) | [docs/dataset_candidates.md](docs/dataset_candidates.md) |
| Daily summary of the human-annotator work (7 Nov 2026): DICES, LLM label, block-by-block FUSE, scope decision, advisor message | [docs/daily_summaries/2026-11-07.md](docs/daily_summaries/2026-11-07.md) |
| DICES dataset explainer page | [claude.ai/artifact/771KB6KMA3VUGRKyS3WUHD](https://claude.ai/artifact/771KB6KMA3VUGRKyS3WUHD) (private until shared from the page's Share menu) |

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

### Choosing a variant per setting without labels

`fuse.selection.select_variant` picks base, cluster@t and/or the majority-vote column from the
score matrices alone. FUSE's model says verifiers interact only through the latent label, so
the correlation matrix is rank one off the diagonal. For each threshold t the selector fits
that rank-one factor on **cross-cluster** entries only and measures the *label share*
R_t = (within-cluster correlation predicted by the label) / (observed within-cluster correlation).
A block whose members agree because they are right has R ≈ 1. A block that agrees for another
reason has R ≈ 0. It clusters at the smallest t with R_t < 0.5. MV is added when its row of the
correlation matrix fits the rank-one model at least as well as the median verifier. The rule was
committed before it was run on real data, and its constants were not tuned. The per-setting
accuracies were already known when it was designed. `scripts/select_variants.py` writes
[results/SELECTION.md](results/SELECTION.md).

| Mode | Predictor | base | selected | best in hindsight | selected − base, without math500-8b |
|---|---|---:|---:|---:|---:|
| query | logistic | 66.1 | 66.6 | 68.4 | −0.1 |
| query | posterior | 66.4 | 66.1 | 68.1 | −0.2 |
| batch | logistic | 64.7 | 66.3 | 69.1 | −0.0 |
| batch | posterior | 67.4 | 68.2 | 70.2 | −0.4 |

- **It separates the case it was built for.** GPQA-70B and GPQA-Diamond-70B get R = 0.59–0.85
  and stay on base in every configuration, so they lose nothing. MATH500-8B gets R@0.3 = 0.37
  (per question) and 0.05 (batched) and is clustered in every configuration: +20 and +15 points
  batched, +6 per question with the logistic predictor, but −2.5 with the posterior predictor.
- **It beats base on average in three of four configurations (+0.5 to +1.6), but only because of
  MATH500-8B.** Without that setting it is slightly worse than base everywhere (−0.0 to −0.4).
  The wrong calls are clustering HLE per question (−2), MMLU-8B per question with the posterior
  predictor (−3.4) and the batched 70B MATH/MMLU-Pro sets (up to −2.7). It also misses the large
  batched gains on MMLU-8B (R = 1.4–1.5) and MMLU-Pro-8B (R ≥ 0.65).
- **The MV test almost never passes** (at most 49% of IMO questions), so the selector never adds
  MV and gives up its IMO gain of 4–5 points.
- Best in hindsight is 2–3 points above the selector, so most of the available gain is still unclaimed.
- A natural next step is deciding per cluster rather than per setting: collapse only the blocks
  whose own label share is low and keep the rest as separate verifiers. That is a new variant, not
  in the grid, so it has not been evaluated.

## DICES: human safety raters as verifiers

[DICES](https://github.com/google-research-datasets/dices-dataset) (Aroyo et al., 2023)
has 350 adversarial chatbot conversations, each rated by all 123 crowd raters for whether
the final response is unsafe, plus an expert gold label (`safety_gold`, 175 unsafe / 175 safe).
This is FUSE's setting with humans instead of reward models: conversations are items,
raters are verifiers, the gold label is hidden. There is no Best-of-N, so FUSE runs once on
the full 350 × m matrix and is scored as a binary classifier (threshold 0.5 on its output).
DICES-990 has no gold label; it is scored against an LLM label instead (see
[DICES-990 against an LLM safety label](#dices-990-against-an-llm-safety-label) below).

```bash
.venv/bin/pip install -e ".[data]"
.venv/bin/python scripts/run_dices.py      # ~1 min, writes results/DICES.md and results/dices350.json
```

Three views of the same ratings (`src/fuse/dices.py`): **per rater** (`Q_overall`
Yes/Unsure/No → 1/0/−1), **per rater, granular** (number of the 17 specific harm reasons
marked Yes), and **per demographic group** (share of the group's raters saying Yes).

Accuracy (%) against the expert label; the last column is the paired difference from
majority vote with a bootstrap 95% interval.

| View | Majority vote | Dawid–Skene | FUSE logistic | FUSE posterior | FUSE − MV | Logistic, 10% labels | Crowd share, gold-tuned threshold |
|---|---:|---:|---:|---:|---:|---:|---:|
| 123 raters | 65.1 | 68.6 | 65.1 | 65.1 | +0.0 [−0.9, +0.9] | 67.6 | 72.0 |
| 104 raters (DICES quality filter) | 66.0 | 69.4 | 66.6 | 66.3 | +0.6 [−0.9, +2.3] | 67.1 | 71.4 |
| 123 raters, granular | 65.1 | 57.4 | **50.0** | **50.0** | −15.1 [−20.0, −10.6] | 67.6 | 72.0 |
| same, `tci_standardize=True` | 65.1 | 57.4 | 64.9 | 65.4 | +0.3 [−0.6, +1.4] | 67.6 | 72.0 |
| 10 race × gender groups | 65.1 | 68.0 | 70.0 | **70.9** | +5.7 [+0.6, +10.9] | 68.0 | 72.0 |
| 30 race × gender × age groups | 65.1 | 66.3 | 64.0 | 64.0 | −1.1 [−2.6, +0.0] | 66.5 | 72.0 |

AUC is 76–79 for every working method, including plain majority vote (79.0). Full tables,
including AUC and the oracle best single rater, are in [results/DICES.md](results/DICES.md).

- **The crowd caps every label-free method.** Even with the threshold tuned on the gold labels,
  the share of raters saying "unsafe" reaches only 72.0%. Individual raters range from 46 to 72 AUC
  (median 62). Crowd and experts disagree on what is unsafe: raters call 23–33% of
  conversations unsafe, the experts 50%. A label-free method can only recover the crowd's
  latent class, and FUSE estimates its prevalence at 0.33, which matches the crowd, not the experts.
- **Per rater, FUSE reduces to majority vote** (identical predictions, ±0.9 CI). With 123
  roughly exchangeable raters the posterior is dominated by the vote count. Dawid–Skene is
  3.4 points better, only because it calls more conversations unsafe (33% vs 23%). FUSE's
  drop-below-random step removed 1 rater, and only 1 of the 19 raters DICES removes for quality.
- **Granular counts break FUSE** through the failure noted under "Things tried" above: the TCI search pushes
  thresholds to near-constant splits, the moment estimate puts prevalence at 1.5%, and every
  conversation is called safe. The scale-free statistic (`tci_standardize=True`) fixes it and
  brings it back to majority vote.
- **10 demographic groups is the only view where FUSE beats majority vote** (+5.7 points,
  CI excludes 0), close to the 72.0 ceiling. Its prevalence estimate came out 0.48, so it
  thresholds lower than majority vote does. With 30 finer groups the estimate falls back to
  0.26 and the gain disappears, so treat the 10-group result as one favourable configuration,
  not a robust improvement.
- Logistic regression with 5–10% of labels (18–35 conversations) is no better than the
  label-free methods: 123 features is too many for that few labels.

**The methods make the same mistakes** ([results/DICES_ERRORS.md](results/DICES_ERRORS.md),
`scripts/dices_error_overlap.py`). On the 123-rater view, majority vote, the naive mean and
both FUSE predictors agree on 99–100% of conversations. All six label-free methods rank the
conversations almost identically (AUC 78–79) and differ mainly in how many they call unsafe.
89 conversations are wrong under all six. 31 expert-unsafe conversations are missed by all
nine methods, including the three that use labels: only 11–26% of raters flag them, and they
cluster in gender/sexuality bias (10 of 22) and personal information (5 of 7). No subset of
raters sees them. A rater's flag rate on them correlates 0.94 with the same rater's rate on
expert-safe conversations, and raters flag them less often than expert-safe ones (17% vs 23%),
in every demographic group. Better aggregation of these ratings cannot recover them.

### DICES-990 against an LLM safety label

DICES-990 (990 conversations, 172 raters, 58% of the rater × conversation matrix empty) has no
expert verdict. We labelled it with `gpt-6.1-terra` and a written safety policy, after
developing the policy on half of DICES-350 and checking it once on the other half
([results/DICES_LLM_LABELS.md](results/DICES_LLM_LABELS.md)). The final prompt (v4) agrees with
the DICES-350 experts on **69.9%** of the 176 held-out conversations, against 65.3% for the
crowd majority (+4.5, 95% interval −2.8 to +11.9). It calls 40.8% of DICES-990 unsafe, the
crowd majority 15.7%.

```bash
.venv/bin/pip install -e ".[data,llm]"     # OPENAI_API_KEY in an untracked .env
.venv/bin/python scripts/label_dices_llm.py run --split 990 --prompt v4 --name dices990_v4 --direct
.venv/bin/python scripts/run_dices990.py --labels dices990_v4 --labels350 dev_v4 test_v4
```

Agreement (%) with the LLM label on DICES-990; the last column is FUSE (logistic) minus majority
vote with a paired bootstrap 95% interval. Full tables: [results/DICES990.md](results/DICES990.md).

| Verifiers | Missing | Majority vote | Dawid–Skene | FUSE logistic | FUSE posterior | FUSE − MV | Crowd share, threshold tuned on the label |
|---|---:|---:|---:|---:|---:|---:|---:|
| 172 raters, missing imputed as 0 | 58% | 71.4 | **77.5** | 71.0 | 70.9 | −0.4 [−2.1, +1.4] | 77.6 |
| 157 raters with ≥ 50 ratings | 54% | 71.4 | **77.6** | 71.5 | 71.3 | +0.1 [−1.7, +1.9] | 77.6 |
| 4 locale × gender groups | 0% | 71.7 | 74.3 | 73.0 | 72.4 | +1.3 [−2.5, +5.1] | 77.6 |
| 6 locale × age groups | 0% | 71.4 | 74.8 | 74.5 | 73.3 | +3.1 [−0.5, +6.9] | 77.6 |
| 7 locale × race groups | 0% | 71.4 | 74.0 | **77.5** | 74.9 | **+6.1 [+2.9, +9.2]** | 77.6 |

- **On sparse individual raters FUSE is majority vote again**, and imputing the 58% missing
  ratings as 0 does not help it: its prevalence estimate (17%) follows the crowd. Dawid–Skene,
  which skips missing ratings instead of imputing them, gains 6 points. Missing data is where
  FUSE needs the pairwise-complete moments planned in the proposal.
- **On demographic groups, which have no missing entries, FUSE beats majority vote**, most on
  locale × race (+6.1, CI excludes 0), where it reaches the label-tuned crowd threshold (77.6).
  This repeats the DICES-350 pattern: FUSE helps with a few aggregated verifiers and does
  nothing with many interchangeable raters.
- **The LLM label shifts the level, not the ranking.** On DICES-350 the same runs scored
  against the experts and against the LLM label order the methods the same way (per rater
  FUSE = majority vote under both; on race × gender groups FUSE ahead of majority vote by
  4.0–4.9 against the experts and 2.6–2.9 against the LLM label), while every label-free method agrees
  more with the LLM label (77–81%) than with the experts (65–72%). The LLM label sits between
  the crowd and the experts.

**Running FUSE block by block fixes the missing-data problem** ([results/DICES990_BLOCKS.md](results/DICES990_BLOCKS.md),
`scripts/dices990_blocks.py`). DICES-990 was collected in batches, so conversations can be
grouped by shared raters: average-linkage clustering of their rater sets (Jaccard) gives 5
blocks of 159–330 conversations, the largest split in which every block keeps at least 100.
In each block, 41–56 "core" raters rated at least 90% of its conversations, so the block's
matrix is under 1% empty instead of 58%. Each block is fitted separately; every method in a
block, majority vote included, sees the same raters; predictions are pooled over all 990.

| Pooled over 990, agreement with the LLM label | Whole sparse matrix | Blocks, core raters | Blocks: − majority vote (95% CI) |
|---|---:|---:|---|
| Majority vote | 71.4 | 70.5 | |
| Dawid–Skene | 77.5 | 76.3 | +5.8 [+2.8, +8.7] |
| FUSE logistic | 71.0 | 74.4 | +3.9 [+2.1, +5.8] |
| FUSE posterior | 70.9 | **75.6** | **+5.1 [+3.1, +7.1]** |

- FUSE beats majority vote in all 5 blocks (+1.3 to +6.7; significant in 3) and is level with
  Dawid–Skene (FUSE posterior − Dawid–Skene: −0.7 [−3.0, +1.5]), which was 6 points ahead on the
  sparse matrix.
- The number of blocks barely matters: with 3 or 4 blocks FUSE posterior gains +5.3 and +4.9.
- Density is what helps, not the split: the same blocks with all their raters (21–37% empty
  inside the block) give FUSE only +0.5 to +1.8.

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
