# A correlated block of small LLMs next to humans (RTE), 11 Nov 2026

Tables: [results/CORRELATED_BLOCK_RTE.md](../results/CORRELATED_BLOCK_RTE.md) · script:
[scripts/run_correlated_block_rte.py](../scripts/run_correlated_block_rte.py) · LLM scores:
[scripts/label_rte_local.py](../scripts/label_rte_local.py).

**Why this test.** The simulation map ([regime_map.md](regime_map.md)) puts FUSE ahead of
Dawid–Skene in one regime only: a dense panel with a minority block of weak raters who share
errors. Every real panel we tried before was either near-independent (humans) or had a block
that was correlated but strong (the 8B LLMs on RTE). This test builds the missing case on real
labels.

## Before the test: the implementation was checked

We checked whether a bug was behind FUSE's weak results.
- The code matches the paper's equations (eq. 4, Theorem 2.3, Proposition C.1, Appendix D).
- It reproduces the paper within 2 points in 10 of 12 settings (README).
- A step-by-step ablation on the simulation, with the same thresholds for both methods:
  - The moment estimates are off by 1–2 points of balanced accuracy.
  - With independent raters, the posterior matches the posterior with the true parameters to within
    about 2 AUC points.
  - With median thresholds, FUSE equals Dawid–Skene (96.1 vs 96.1; 86.9 vs 86.4).

So no bug. FUSE's wins and losses both come from the paper's threshold search (TCI step).
- With many near-chance raters, the search pushes 95% of the columns to extreme splits and loses
  17 AUC points.
- With a correlated weak block, the same search is what puts FUSE ahead (76 vs 61–64).

The check also showed that the logistic step is not specific to FUSE. Fed with Dawid–Skene
posteriors instead of FUSE's, it does as well or better in most simulated settings. That is
why this test adds a "D&S → logistic" baseline.

## Setup

**Data.** RTE (Snow et al. 2008): 800 entailment pairs, 10 MTurk labels per pair, gold label,
50% positive.

**Humans.** Each pair's 10 labels are shuffled into 10 slot columns. That gives a dense matrix
of yes/no votes, with a different worker behind each vote. One human label: AUC 72.9.

**The block.** Small models from one family: Qwen2.5-0.5B, Qwen3-0.6B, Qwen2.5-1.5B and
Qwen3-1.7B (4-bit MLX). Each model runs with the same two prompts as the earlier RTE pilot.
- Tiny models (0.5B and 0.6B): AUC 74–77, so about as good as one human. They are poorly
  calibrated, with accuracy 50–64% at P(Yes) = 0.5.
- 1.5B and 1.7B: AUC 88–90.

**Conditional dependence.** Within-class correlation:
- human with human: 0.04;
- tiny LLM with tiny LLM: 0.66;
- human with tiny LLM: 0.09.

This is the regime the simulation asked for.

**Statistics.** Mean over 20 slot shuffles. The 95% intervals resample both the shuffle and the
items.

## Results (AUC)

| Panel | Dawid–Skene (5 bins) | D&S → logistic | Mean z | FUSE logistic | FUSE − D&S 5 bins | FUSE − D&S → logistic | FUSE − mean z |
|---|---:|---:|---:|---:|---|---|---|
| 10 humans | 96.5 | 96.5 | 96.5 | 96.5 | +0.0 | +0.0 | −0.0 [−0.5, +0.4] |
| 10 humans + 2 tiny (17%) | 94.9 | 96.3 | 96.3 | 96.6 | +1.6 [+0.9, +2.5] | +0.3 [−0.1, +0.6] | +0.3 [−0.1, +0.7] |
| 10 humans + 4 tiny (29%) | 91.1 | 95.0 | 95.9 | **96.7** | +5.6 [+4.1, +7.1] | +1.7 [+1.0, +2.4] | +0.8 [+0.3, +1.3] |
| 10 humans + 8 small Qwen (44%) | 93.8 | 94.8 | 96.3 | **96.4** | +2.7 [+1.7, +3.9] | +1.6 [+0.8, +2.7] | +0.1 [−0.4, +0.9] |
| 5 humans | 92.1 | 92.1 | 92.2 | 92.1 | +0.0 | +0.0 | −0.1 |
| 5 humans + 4 tiny (44%) | 84.8 | 84.9 | **92.3** | 92.1 | +7.3 [+3.4, +10.7] | +7.2 [+3.4, +11.7] | −0.2 [−3.8, +2.1] |
| 3 humans | 86.8 | 86.8 | 87.3 | 86.8 | +0.0 | +0.0 | −0.4 |
| 3 humans + 4 tiny (57%) | 82.3 | 79.7 | **89.1** | 83.6 | +1.3 [−2.0, +9.0] | +3.9 [+1.3, +11.5] | −5.5 [−9.3, +1.7] |
| 8 small Qwen only | 89.9 | **90.6** | 89.2 | 87.3 | −2.7 [−3.6, −1.8] | −3.4 [−4.3, −2.5] | −2.0 [−2.7, −1.3] |

All methods and panels are in the results file.

## What this says

1. **This is the first real-data panel where FUSE beats Dawid–Skene with an interval that
   excludes zero.** When a correlated block of tiny LLMs is 17–44% of the panel, FUSE logistic
   is 1.6 to 7.3 AUC points above binned D&S. It also beats D&S → logistic, so the gain is not
   only the logistic step. The mechanism is the one the simulation predicted: D&S reads the
   block's shared errors as signal, and binned D&S drops from 96.5 to 91.1 when 4 tiny models
   join 10 humans.
2. **But the gain is robustness, not added value.**
   - With 10 or 5 humans, no method gets more out of humans plus the block than out of the
     humans alone. The best method on humans alone gets 96.6 and 92.5; the best with the block
     gets 96.7 and 92.3. FUSE's win is that it is not hurt by the block, while D&S is.
   - With 3 humans the block does add information, and the plain average of z-scores uses it
     best (best on 3 humans alone 87.5, mean z with the block 89.1). FUSE loses there (83.6).
3. **The plain average of z-scores is about as robust.** It is within 1 point of FUSE in every
   panel with 5 or 10 humans. It is ahead with 3 humans, but that interval crosses zero, so not
   significant.
4. **The TCI search earns part of the win.** With median thresholds instead of the search,
   FUSE logistic drops from 96.7 to 95.6 (10 + 4 tiny) and from 92.1 to 88.5 (5 + 4 tiny). On
   the 8 small Qwen models alone, the same search costs 3 points.

## Caveats

- The human slots mix different workers, so a slot is not one person. That keeps the slots
  close to independent, which is the point, but it is not a fixed panel of annotators.
- RTE is public (part of GLUE), so the models may have seen it. That matters less here, since
  the block is meant to be weak.
- One task and one model family. A second task with a natural correlated block would show
  whether this holds beyond RTE.
