# Shared-error blocks on the paper's own benchmarks, 21 Nov 2026

Tables: [results/PAPER_BLOCKS.md](../results/PAPER_BLOCKS.md) · raw: [results/paper_blocks.json](../results/paper_blocks.json),
[results/paper_blocks.log](../results/paper_blocks.log) · scripts:
[scripts/run_paper_blocks.py](../scripts/run_paper_blocks.py),
[scripts/summarize_paper_blocks.py](../scripts/summarize_paper_blocks.py).

**The question.** [residual_blocks.md](residual_blocks.md) ends with a candidate thesis claim: FUSE
beats Dawid–Skene in the paper only because some verifiers share errors, and once a label-free
detector collapses those blocks, "D&S → logistic" matches FUSE. That was tested on simulations
and human panels, but not on the paper's 12 Best-of-N settings, because the data was not cached
on the machine that ran it. This note runs that test.

## Setup

- **Data.** The 12 settings of the reproduction (README): Weaver GPQA, GPQA Diamond, MATH-500, MMLU
  and MMLU-Pro with 8B and 70B generators (16 and 32 verifiers, paper Table 6), plus HLE (7) and
  IMO (9). Missing scores are set to 0, as in the paper.
- **Metric.** Best-of-N selection accuracy, ties split as in the paper. Every method is label-free.
- **Modes.** Batched (one ensemble per setting, rows of all questions stacked) and per-query
  (one ensemble per question).
- **Methods.**
  - FUSE as in the paper (threshold search, logistic predictor).
  - Dawid–Skene on median-binarized scores, alone and → logistic (the FUSE logistic step fed with
    D&S posteriors), and the mean z-score.
  - Residual block detector (`residual_groups`, thresholds 0.3 and 0.4), collapse each block to
    its mean z-score, then D&S → logistic, mean z, or FUSE.
  - Two diagnostics: FUSE with median thresholds (no search), and D&S on the thresholds FUSE's
    search picked, → logistic.
- **Spearman with ties.** `fuse.clustering.spearman_matrix` ranks tied values by row order. That
  gives two independent binary columns a correlation of about 0.22, and two sparse binary columns
  0.67 (scipy: about 0). The 70B settings include binary LLM-judge verdicts, so the detector was
  run with the repo function ("repo") and with a tie-aware Spearman ("ties").

## Results

Best of the two modes for each method (all numbers per mode are in the results file):

| Setting | FUSE (ours) | FUSE (paper) | Collapse blocks → D&S → logistic | D&S on FUSE's thresholds → logistic | Mean z |
|---|---:|---:|---:|---:|---:|
| gpqa-70b | 66.9 | 66.8 | 56.8 | 67.3 | 59.1 |
| gpqa-diamond-70b | 59.6 | 64.4 | 53.5 | 58.6 | 55.1 |
| math500-70b | 94.1 | 92.8 | 86.4 | 94.4 | 92.6 |
| mmlu-70b | 93.7 | 94.1 | 89.3 | 94.2 | 93.0 |
| mmlu-pro-70b | 91.6 | 91.4 | 83.6 | 91.6 | 87.0 |
| gpqa-8b | 38.7 | 40.5 | 35.9 | 38.3 | 40.2 |
| gpqa-diamond-8b | 40.9 | 42.7 | 37.0 | 41.8 | 42.9 |
| math500-8b | 66.0 | 83.9 | 56.6 | 63.9 | 74.6 |
| mmlu-8b | 80.5 | 84.9 | 78.2 | 80.3 | 82.9 |
| mmlu-pro-8b | 69.0 | 69.8 | 64.2 | 66.4 | 68.8 |
| hle | 53.4 | 54.3 | 53.2 | 53.4 | 52.2 |
| imo | 62.1 | 63.8 | 59.6 | 62.1 | 62.1 |
| **Mean** | **68.0** | 70.8 | 62.9 | 67.7 | 67.6 |

Collapse uses the tie-aware detector at 0.4. Gaps to FUSE, per mode:

| Method | Batched: mean gap | Batched: within ±1 | Per-query: mean gap | Per-query: within ±1 |
|---|---:|---:|---:|---:|
| D&S → logistic, no collapsing | −2.8 | 1 of 12 | −4.1 | 1 of 12 |
| Collapse (ties 0.4) → D&S → logistic | −3.0 | 1 of 12 | −4.0 | 1 of 12 |
| Collapse (ties 0.3) → D&S → logistic | −2.9 | 1 of 12 | −3.5 | 1 of 12 |
| FUSE, median thresholds | −3.2 | 3 of 12 | −2.7 | 1 of 12 |
| **D&S on FUSE's thresholds → logistic** | **+1.2** | **7 of 12** | **−0.5** | **8 of 12** |

## What this says

1. **The block claim does not hold on the paper's benchmarks.**
   - The detector finds plausible blocks in all 10 Weaver settings: variants of one model
     (QwenPRM avg/min, EurusPRM stages), reward models from one family (DecisionTreeReward8B with
     QRM; the Gemma-based RMs), and groups of LLM-judge verdicts. In batched mode it merges 32
     verifiers into 22–26 columns for 70B and 16 into 13–14 for 8B. It merges nothing on HLE and IMO.
   - Collapsing them barely moves D&S → logistic (mean −2.8 → −3.0 batched, −4.1 → −4.0
     per-query). FUSE stays 3–5 points ahead.
   - Collapsing also does not help FUSE itself (−1.2 batched, −0.8 per-query).
2. **FUSE's lead is its threshold search.**
   - Give Dawid–Skene the thresholds FUSE's search picked, and it matches FUSE: within one point
     in 7 of 12 settings (batched) and 8 of 12 (per-query), mean +1.2 and −0.5.
   - Take the search away (median thresholds) and FUSE falls back to Dawid–Skene level (−3.2, −2.7).
   - So the method-of-moments estimator and the triplet posterior add nothing over Dawid–Skene
     here. This matches the simulation finding in
     [correlated_block_rte.md](correlated_block_rte.md) that the search drives both FUSE's wins
     and its losses.
3. **This also explains the human-annotator results.** Human votes are binary, so there is no
   threshold to search, and FUSE ties Dawid–Skene on every crowd dataset (RTE, D_PosSent,
   DICES). The threshold search is FUSE's contribution, and it matters only for continuous scores
   with different scales and skews, such as reward models.
4. **A plain mean z-score is a strong baseline.** It ties our FUSE on average (67.6 vs 68.0) and
   is ahead on several 8B settings. It is 3.2 points below the FUSE numbers the paper reports,
   though, and most of that gap is on the settings where our FUSE also falls short of the paper
   (math500-8b: ours 66.0, paper 83.9). Do not read it as "mean z beats FUSE" until the 8B gap in
   the reproduction is explained.
5. **The Spearman tie bug changes which verdicts merge, not the accuracy here.** With the repo
   function the detector puts up to 6 binary verdicts in one group, against 3–4 with the
   tie-aware version. Accuracy moves by at most 0.3 points on average in these settings. Panels
   made mostly of binary votes (RTE human slots, demographic vote shares) may be affected more,
   so the residual-block results in PR #10 should be rerun with a fixed `spearman_matrix`.

## For the thesis

The story moves from "shared-error blocks" to the threshold search:
- FUSE = a threshold search (label-free, by minimizing the triplet-independence violation) +
  an estimator that is no better than Dawid–Skene.
- The search is what wins on LLM verifiers (up to +12 points on GPQA-70B) and what loses with
  many near-chance raters (residual_blocks.md, threshold_search_fixes.md).
- On binary human votes there is nothing to search, which is why FUSE ties Dawid–Skene there.

Natural next tests: Dawid–Skene with FUSE's thresholds on the crowd datasets with continuous
scores (deepfakes, DICES group shares, VIPER), and a closer look at why the search fails with
near-chance raters.

## Caveats

- One run per setting, no confidence intervals. Per-query results average over 123–719 questions.
- Detector thresholds (0.3, 0.4) and the trim share (70%) are fixed, as in residual_blocks.md.
- D&S binarizes at the median (two-valued columns at their minimum).
- Our FUSE reproduction is 2.8 points below the paper on average, mostly on 8B settings
  (math500-8b, mmlu-8b). The conclusions compare methods within our code, so they do not depend on
  that gap, but the gap itself is still unexplained.
