# Order-free FUSE (24 Nov 2026)

Tables: [results/FUSE_ORDER_FREE.md](../results/FUSE_ORDER_FREE.md) (rater panels),
[results/ORDER_FREE_BON.md](../results/ORDER_FREE_BON.md) (the paper's Best-of-N data) · scripts:
[scripts/run_fuse_order_free.py](../scripts/run_fuse_order_free.py),
[scripts/run_order_free_bon.py](../scripts/run_order_free_bon.py) · code:
[src/fuse/tci.py](../src/fuse/tci.py) · test: `test_symmetric_tci_and_greedy_search_do_not_depend_on_column_order`
in [tests/test_fuse.py](../tests/test_fuse.py).

**The problem.** FUSE's result changes when the rater columns are reordered ([fuse_order.md](fuse_order.md)).
It has two sources, and both are in the paper's method:
1. **The statistic.** Equation (4) sums, for each verifier j3, the variance over pairs j1 < j2 < j3
   only. Verifiers late in the column order enter many more terms than early ones. On one panel
   the statistic ranged from 41 to 537 depending only on the order.
2. **The search.** Coordinate descent visits the verifiers in column order and accepts every
   improving threshold at once.

## The fix (not in the paper, off by default)

- `FUSE(tci_symmetric=True)`: for each j3, the variance is over every pair of the other verifiers.
  It is still zero under TCI, and it does not depend on the order.
- `FUSE(tci_search="greedy")`: at each step, evaluate every verifier and every candidate threshold
  and apply only the single best move. Stop when no move improves the statistic, or after
  `n_sweeps * m` moves.

Together they give the same result in any column order (tested). The paper's defaults are
unchanged. The greedy search costs more: about m times as many evaluations per sweep.

## Results

**The paper's Best-of-N data** (10 settings; GPQA-70B and GPQA-Diamond-70B are no longer on
Hugging Face). Selection accuracy, against paper FUSE averaged over 3 random column orders:

| Configuration | Order-free minus paper FUSE (random order), mean of 10 settings | Spread of paper FUSE across 3 orders |
|---|---:|---|
| Per question, logistic | −0.1 | up to 3 points (GPQA-Diamond-8B) |
| Batched, posterior | +0.4 | up to 10 points (MATH500-8B: 57.7–67.8) |

The fix costs nothing on average where FUSE was built, and in batched mode it removes a large
order lottery. Its worst setting is batched GPQA-8B (34.5, against 36.8 averaged over orders).

**Rater panels** (AUC; the fair comparison is against plain Dawid–Skene on 5 bins):

| Panel | Order-free FUSE | Plain D&S | Mean z | Order-free minus D&S |
|---|---:|---:|---:|---:|
| DICES 10 humans + 8 LLMs | 74.8 | 71.1 | 72.6 | +3.6 ± 0.8 |
| DICES 10 humans + 6 local LLMs | 73.6 | 67.5 | 71.2 | +6.1 ± 1.0 |
| DICES 5 humans + 8 LLMs | 72.3 | 68.7 | 70.6 | +3.6 ± 0.5 |
| DICES 10 humans, no LLMs | 75.4 | 74.7 | 74.3 | +0.7 ± 1.2 |
| RTE 10 humans + 4 tiny Qwen | 95.9 | 91.1 | 95.9 | +4.8 ± 0.1 |
| RTE 5 humans + 4 tiny Qwen | 88.5 | 84.8 | 92.3 | +3.6 ± 0.2 |
| RTE 3 humans + 4 tiny Qwen | 84.2 | 82.3 | 89.1 | +1.9 ± 0.3 |
| RTE 10 humans, no LLMs | 96.5 | 96.5 | 96.5 | 0.0 |

- On every mixed panel, order-free FUSE beats plain Dawid–Skene, by 1.9 to 6.1 points.
- On human-only panels it ties.
- Against mean z it wins on DICES (by 1.7 to 2.4) and loses on small RTE panels (by 3.8 and 4.9).
- Accuracy is close to paper FUSE in a random order. Two exceptions: RTE 5 + 4 (88.5 vs 90.5) and
  the simulation with no shared noise (87.9 vs 91.0).
- In the simulation with strong shared noise among the LLMs (0.6), every FUSE stays near 72. Mean z
  reaches 90.8 there, and the detector followed by D&S reaches 98.2.

"Detector + D&S" (merge correlated raters, then Dawid–Skene) is in the results file. It is our
addition to D&S, not a plain baseline, so it explains where FUSE's edge comes from rather than
being the fair comparison.

## The gap from the paper's numbers

It is there with or without the fix, and it was documented in the README before this work. The
largest is MATH500-8B: 60–66 for us against 83.9 in the paper. The README's explanation: nine of
its 16 reward models are near-random but correlated with each other, and the moment estimator
reads that block as the correctness signal.

**Could the order explain it?** Partly at most ([results/MATH500_ORDERS.md](../results/MATH500_ORDERS.md),
[scripts/run_math500_orders.py](../scripts/run_math500_orders.py)). Over 30 random orders, batched
paper FUSE ranges from 52 to 80, with a median of 55 (logistic) and 60 (posterior). Only one order
in 30 reaches 75, and none reaches 83.9. Per question the range is narrow, 63.8 to 68.2 over 8
orders. So a lucky order can close most of the gap in batched mode, but it is rare, and we cannot
tell which order the authors used. Other possible causes, none of them checked: preprocessing
details the paper does not give, or a different version of the Weaver data on Hugging Face.

## Caveats

- 3 random orders per Best-of-N setting, 4 per rater panel. The spreads would grow with more.
- The paper does not say which configuration produced each of its numbers.
- The authors' code is not released, so we cannot check whether it has the same order dependence.
