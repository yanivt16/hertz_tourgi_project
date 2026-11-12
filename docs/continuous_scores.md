# Does FUSE beat Dawid-Skene on continuous scores? (10 Nov 2026)

Tables: [results/CONTINUOUS.md](../results/CONTINUOUS.md) · script:
[scripts/run_continuous.py](../scripts/run_continuous.py) · new baseline: Dawid-Skene on 5
quantile bins per rater ([src/fuse/ordinal.py](../src/fuse/ordinal.py)).

On binary votes, missing-aware FUSE only matched Dawid-Skene
([missing_data_plan.md](missing_data_plan.md)). The hope was that continuous scores are where
FUSE's extra machinery pays off: label-free per-rater thresholds (TCI) and moment estimates
that work on any score scale. They mostly do not. The fair baseline for scores is Dawid-Skene
on binned scores, not on a median split.

## Simulation (10 raters, Gaussian scores, 10 seeds)

| Scenario | D&S, 5 bins | Mean z-score | FUSE logistic (paper) | FUSE product posterior | Oracle |
|---|---:|---:|---:|---:|---:|
| Similar raters, balanced | 99.6 | 99.7 | 99.6 | 96.8 | 99.7 |
| Raters differ in quality, balanced | 99.7 | 98.8 | 99.7 | 97.3 | 99.8 |
| Differ in quality, 15% positive | 99.5 | 98.9 | 96.7 | 94.6 | 99.7 |
| ... + skewed score scales | 99.5 | 97.9 | 95.3 | 94.6 | 99.7 |
| ... + 3 correlated junk raters | **48.7** | 94.1 | 93.4 | 93.3 | 99.7 |

AUC against the true label.

1. **Under conditional independence, Dawid-Skene on 5 bins is essentially at the oracle.**
   FUSE ties it with balanced classes and falls behind with 15% positives: AUC 96.7 vs 99.5,
   accuracy 82 vs 98. Binarising each rater at one threshold throws away information, which
   the binned model keeps.
2. **With correlated raters, Dawid-Skene collapses to chance.** It locks onto the junk
   cluster. FUSE stays at AUC 93, but a plain average of z-scores does just as well (94.1).
   So FUSE's robustness here is relative to Dawid-Skene, not to simple averaging.

## Real scores (dense, no missing entries)

- **VIPER suspicion score** (4 dense blocks, 93-203 cases, 19-47 cancers). No method is
  reliably better. Every difference from binned Dawid-Skene has a CI that crosses 0, except
  readers' own recall decision (-3.4) and the mean z-score (-1.5, upper bound -0.0) on the
  FFDM 8×201 block.
- **DICES-350 demographic-group shares** (10 or 30 groups). FUSE logistic ties binned
  Dawid-Skene (+0.2 and -0.0 AUC). The posterior variants are 1-2.5 points lower.
- **DICES-350 granular reason counts.** FUSE predicts every conversation safe (AUC 59), as in
  the earlier run in [DICES.md](../results/DICES.md); every other method is at 77-79.

## What this means for the project

- On human annotators, in every setting we have tried, FUSE ties Dawid-Skene at best:
  binary, sparse, and continuous.
- The one place it wins is correlated raters, a known Dawid-Skene failure mode, but there a
  plain average does as well.
- Candidate framing for the midterm: report this as a negative result, with the
  missing-data fix as the methodological contribution.
- Candidate next step: test whether correlated groups of raters exist in real crowds (for
  example DICES demographic groups). If they do, a Dawid-Skene that uses FUSE's TCI
  statistic to detect and merge correlated raters is a natural hybrid. This is untested; to
  be discussed with Regev.
