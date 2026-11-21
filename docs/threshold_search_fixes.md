# Fixing FUSE's threshold search, 11 Nov 2026

Tables: [results/TCI_FIXES.md](../results/TCI_FIXES.md) · scripts:
[scripts/tci_fix_sim.py](../scripts/tci_fix_sim.py) (simulation) and
[scripts/tci_fix_real.py](../scripts/tci_fix_real.py) (every real panel we have).

**The question.** The implementation check ([correlated_block_rte.md](correlated_block_rte.md))
showed that FUSE's wins and losses both come from the paper's threshold search (TCI step):
- With many near-chance raters it pushes columns to extreme splits and loses up to 17 AUC points.
- With a correlated weak block, the same search is what puts FUSE ahead of Dawid–Skene.

Can a change to the search keep the win without the losses?

## Variants tested

All are label-free.
- **Restricting the search:**
  - median thresholds with no search (`n_sweeps=0`);
  - a narrower quantile range, 20–80% or 30–70% (new option `quantile_range`; the paper uses 5–95%);
  - the scale-free TCI statistic (`tci_standardize=True`, already in the code).
- **Collapsing correlated raters before FUSE** (`cluster_threshold=0.5`, already in the code; not
  in the paper). Raters whose average |Spearman| exceeds 0.5 are averaged into one column. We ran
  it with and without the search.
- **Baselines on the clustered columns:** Dawid–Skene → logistic and mean z-score. These show
  whether a gain belongs to FUSE or to the clustering.

## Results (AUC)

| Setting | D&S → logistic | Mean z | FUSE paper | FUSE median | FUSE 30–70% | FUSE clustered, median | Clustered D&S → logistic |
|---|---:|---:|---:|---:|---:|---:|---:|
| Sim: 20 near-chance + 2 experts | 91.8 | 75.2 | 72.2 | 89.8 | 85.9 | 89.8 | 91.8 |
| Sim: 40% weak correlated block | 59.9 | 86.2 | 76.3 | 60.7 | 62.9 | 95.1 | 95.6 |
| RTE 5 humans + 4 tiny | 85.0 | 92.3 | 92.6 | 88.5 | 88.4 | 93.6 | 93.7 |
| RTE 3 humans + 4 tiny | 79.8 | 89.1 | 83.4 | 84.3 | 84.1 | 90.6 | 90.8 |
| DICES-350 granular | 77.8 | 77.7 | 59.2 | 77.9 | 78.1 | 79.0 | 78.1 |
| Fact-checking, political groups | 90.6 | 91.2 | 90.9 | 90.9 | 91.0 | 79.1 | 87.2 |
| RTE 8 small Qwen only | 90.6 | 89.2 | 87.3 | 90.4 | 90.5 | 86.3 | 86.3 |

The other 20 settings are in the results file.

## What this says

1. **Restricting the search fixes the losses but gives up the win.** Median thresholds, the
   narrow range and the scale-free statistic all recover the near-chance and DICES losses (72 →
   80–90, 59 → 77–78). But they also drop FUSE back to Dawid–Skene level on a correlated block
   (RTE 5 + 4 tiny: 92.6 → 88.5; simulated 40% block: 76 → 61–63). The extreme splits are how
   the paper's search discounts a block, so a search that avoids them cannot keep that win.
2. **Collapsing the block works much better than the search.**
   - FUSE with clustering and median thresholds is the best FUSE in every block setting:
     - simulated 40% block: 95.1, against 76.3 for the paper and 86.2 for mean z;
     - RTE 3 humans + 4 tiny: 90.6, against 83.4 for the paper; this is the panel where mean z
       (89.1) had been best.
   - It also does not break with many near-chance raters (89.8), because independent raters do
     not cluster.
3. **But the gain belongs to the clustering, not to FUSE.** Dawid–Skene → logistic on the same
   clustered columns does as well or slightly better in every block setting (95.6, 93.7, 90.8),
   and so does mean z on the clustered columns on RTE (93.7, 90.9). This
   repeats the earlier finding about the logistic step: the part that helps is a generic step
   any aggregator can use.
4. **Clustering has its own failure mode.** It cannot tell raters who agree because they share
   errors from raters who agree because they are accurate:
   - On political groups in fact-checking, the groups correlate mainly through the truth, get
     merged, and FUSE drops from 90.9 to 79.1.
   - On 8 small Qwen models alone it drops from 90.4 to 86.3.
   - With threshold 0.5 there is no single setting that is safe everywhere.

## So what

There is no threshold-search change that turns FUSE into a better method than
"cluster, then Dawid–Skene → logistic". The open problem has moved. It is no longer the
threshold search but detecting a block of shared errors without labels, as distinct from
agreement that comes from accuracy. A detector that clusters on correlation left after
removing the shared signal (the rank-one part the moment method fits) is the natural next
test, and it would help every aggregator, not only FUSE.

Caveats:
- One clustering threshold (0.5).
- 6 seeds in the simulation and 10 shuffles on RTE, with no intervals here (the earlier RTE
  intervals were ±1–6 points).
- The paper's own 12 verifier benchmarks were not rerun. They are not cached locally, and earlier
  runs there found that the scale-free statistic lowers accuracy and that clustering moves the
  average by up to 2 points either way.
