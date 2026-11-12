# Finding raters who share errors without labels, 12 Nov 2026

Tables: [results/RESIDUAL_BLOCKS.md](../results/RESIDUAL_BLOCKS.md) · script:
[scripts/run_residual_blocks.py](../scripts/run_residual_blocks.py) · detector:
`residual_groups` in [src/fuse/clustering.py](../src/fuse/clustering.py).

**Why.** The threshold-search test ([threshold_search_fixes.md](threshold_search_fixes.md))
showed that collapsing a correlated block helps every aggregator. The clustering we had works on
total correlation, though. It cannot tell raters who agree because they are accurate from raters
who agree because they share errors, so it also merged accurate raters (fact-checking groups:
90.9 → 79.1).

**The detector.** Under conditional independence the rater correlation matrix is rank one off the
diagonal: C_ij = u_i u_j, where u_i is rater i's correlation with the latent label.
- We fit u by least trimmed squares, keeping the 70% of rater pairs that fit best, from several
  starting points. A minority of pairs that share errors is then trimmed instead of pulling the fit.
- The partial correlation left after the fit, (C_ij − u_i u_j) / √((1 − u_i²)(1 − u_j²)), is
  high only for pairs that share errors.
- Raters are clustered on it (average linkage, threshold t), and each cluster is averaged into
  one column (mean z-score).
- The aggregator then runs on the collapsed columns. The logistic step is still fitted on all
  the original columns.

## Does it find the right raters?

On the simulation, at t = 0.3, seeds 0–2:
- **A block of 2 or 4 raters with shared noise ρ = 0.6, weak or of normal quality:** it finds
  the exact block in all 9 runs.
- **The base setting with independent raters:** it finds nothing.
- **A weaker shared noise (ρ = 0.3):** it finds the whole block in 1 run of 3 and half of it in
  the other 2.
- **One false positive.** In "20 near-chance raters + 2 experts" it sometimes merges the two
  experts. When everyone else is noise, the two experts' agreement looks like a block.

## Results (AUC)

| Setting | No collapsing: D&S → logistic | Total-correlation clusters: D&S → logistic | Residual 0.3: D&S → logistic | Residual 0.4: D&S → logistic | Residual 0.4: FUSE median | FUSE paper |
|---|---:|---:|---:|---:|---:|---:|
| Sim: 40% weak block | 59.9 | 95.6 | 95.6 | 95.6 | 95.2 | 76.3 |
| Sim: 40% normal-quality block | 85.0 | 97.0 | 97.0 | 97.0 | 96.9 | 91.0 |
| Sim: 40% weak block, ρ = 0.3 | 76.5 | | 95.4 | 76.5 | 78.3 | 94.2 |
| Sim: 20 near-chance + 2 experts | 91.8 | 91.8 | 69.0 | 91.8 | 89.8 | 72.2 |
| RTE 5 humans + 4 tiny | 85.0 | 93.7 | 93.7 | 93.7 | 93.6 | 92.6 |
| RTE 3 humans + 4 tiny | 79.8 | 90.8 | 86.3 | 86.7 | 88.4 | 83.4 |
| RTE 10 humans + 8 small Qwen | 94.8 | 97.4 | 97.4 | 97.4 | 97.4 | 96.4 |
| RTE 8 small Qwen only | 90.6 | 86.3 | 90.9 | 90.6 | 90.3 | 87.3 |
| Fact-checking, political groups | 90.6 | 87.2 | 90.6 | 90.6 | 90.9 | 90.9 |
| VIPER SFM, 15 readers | 82.3 | 80.4 | 84.0 | 84.0 | 83.4 | 80.7 |

All 27 settings are in the results file.

Across the 26 settings run in both tests:

| Method | Mean AUC | Mean gap to the best method | Worst gap |
|---|---:|---:|---:|
| Residual 0.4, then D&S → logistic | 89.8 | 0.8 | 10.2 |
| Residual 0.4, then FUSE median | 89.8 | 0.8 | 8.4 |
| Total-correlation clusters, then D&S → logistic | 89.2 | 1.3 | 10.2 |
| Mean z-score | 88.7 | 1.8 | 16.6 |
| D&S → logistic, no collapsing | 87.1 | 3.5 | 35.7 |
| FUSE paper | 86.6 | 3.9 | 19.9 |

The worst gap of the residual methods is the hard setting (d = 0.3), where mean z (74.5) beats
every other method in this table by 8 or more points. It has nothing to do with blocks.

## What this says

1. **At threshold 0.4 the detector costs nothing and gains a lot where there is a block.**
   - In none of the 27 settings is D&S → logistic after collapsing more than 0.05 points worse
     than without it.
   - Where a block hurts the panel it adds 1.8 to 35.7 points: RTE 5 + 4 tiny 85.0 → 93.7;
     simulated 40% weak block 59.9 → 95.6.
2. **It fixes the failure of total-correlation clustering.** Accurate raters are no longer merged:
   fact-checking groups 87.2 → 90.6, 8 small Qwen 86.3 → 90.6.
3. **After collapsing, FUSE and Dawid–Skene are the same.** FUSE median minus D&S → logistic
   averages +0.06 points (range −1.9 to +1.8). Once shared-error blocks are removed, FUSE has no
   advantage left. Its only win came from handling a block, and a generic pre-step handles blocks
   better.
4. **The threshold is a real trade-off.**
   - At 0.4 it misses a block with weaker shared noise (ρ = 0.3: 76.5, against 95.4 at 0.3).
   - At 0.3 it catches that block, but merges the only two experts in a panel of near-chance
     raters (91.8 → 69.0).
   - On RTE with 3 humans + 4 tiny, total-correlation clustering is still better (90.8 against
     86.7).

   This is partly an identifiability limit. With only two informative raters, agreement from
   accuracy and agreement from shared error look the same without labels.

## For the thesis

**The detector itself is not new.** Prior work, checked 12 Nov 2026 (full text read for the
first two; the last two from abstracts or a delegated read only):

| Paper | Same as ours | Different from ours |
|---|---|---|
| Jaffe, Fetaya, Nadler, Jiang, Kluger, AISTATS 2016, [arXiv:1510.05830](https://arxiv.org/abs/1510.05830) | Same model: raters in groups, each group depends on the label only through a latent variable. Groups are found without labels from the deviation of the covariance from rank one, then aggregated in two stages (L-SML). | Binary predictions only. Detection uses a score matrix of 2×2 minors with spectral clustering, and picks the number of groups by residual. It is O(m⁴). They note it degrades as the group grows (one group of 7+ out of 20). |
| Traganitis and Giannakis, Asilomar 2022, DOI 10.1109/IEEECONF56349.2022.10052052 | Same model for crowdsourcing annotators, groups found before aggregation from second-order moments, then group-aware Dawid–Skene. | Multiclass. Uses robust PCA (low rank + sparse) on co-occurrence matrices, then subspace clustering. The number of groups is assumed known. |
| Bach et al., ICML 2017; Varma et al., ICML 2019 | The same problem in weak supervision: learning which labeling functions are dependent without labels. | Structure learning in a graphical model (Varma: sparse + low-rank decomposition of the inverse covariance). |
| Zhao et al., CARE, arXiv:2603.00039 (2026) | LLM judges with shared confounders, handled without labels. | A factor model (sparse + low-rank precision matrix). It does not cluster judges into groups. |

Our version differs only in details:
- continuous scores via Spearman;
- a trimmed rank-one fit with a partial-correlation threshold, so the number of groups is not set;
- collapsing to the mean z-score, followed by any aggregator.

These are engineering choices, not a new method.

**What may still be ours:**
1. The FUSE paper's wins over Dawid–Skene are explained by blocks of shared errors. A
   dependency-aware pre-step of the Jaffe 2016 kind removes them, and after it FUSE and
   Dawid–Skene are equal.
2. The evaluation on mixed human + LLM panels and on real annotator panels (RTE, VIPER, DICES,
   demographic groups). The papers above use classifier ensembles (UCI, DREAM), one crowd dataset
   (Bluebird), or LLM-judge benchmarks.

## Jaffe et al. (2016) as a baseline

Table: [results/JAFFE_BASELINE.md](../results/JAFFE_BASELINE.md). Code: `jaffe_groups` in
[src/fuse/clustering.py](../src/fuse/clustering.py), written from the paper's main text (Algorithm 1,
Eqs. 14–15), not from their code.

**Sanity check.** On the paper's own synthetic setup (20 binary classifiers, one dependent group),
our implementation recovers the exact group of size 2, 4 and 6 in 3 of 3 seeds and of size 8 in 2 of
3. The paper reports good recovery up to a group of 6 out of 20.

**Head to head.** Same downstream step (collapse, then D&S → logistic); only the detector differs.

| Setting | No collapsing | Ours 0.4 | Jaffe (binary, as in paper) | Jaffe (Spearman) | Jaffe two-stage D&S |
|---|---:|---:|---:|---:|---:|
| Sim: 40% weak block | 59.9 | 95.6 | 94.2 | 95.4 | 87.3 |
| Sim: 40% weak block, ρ = 0.3 | 76.5 | 76.5 | 95.4 | 94.7 | 88.5 |
| Sim: 20 near-chance + 2 experts | 91.8 | 91.8 | 70.2 | 78.7 | 46.4 |
| RTE 5 humans + 4 tiny | 85.0 | 93.7 | 81.2 | 81.2 | 84.5 |
| RTE 3 humans + 4 tiny | 79.8 | 86.7 | 79.5 | 79.7 | 81.9 |
| RTE 10 humans + 8 small Qwen | 94.8 | 97.4 | 94.9 | 94.6 | 94.7 |
| **Mean over 27 settings** | 86.7 | **89.3** | 88.2 | 88.7 | 85.4 |
| Settings > 1 point below no collapsing | 0 | **0** | 2 | 2 | 12 |

1. **Ours is better on average and never hurts. Jaffe is better on weaker shared noise.**
   - Jaffe catches the ρ = 0.3 block that ours misses at 0.4 (95.4 against 76.5).
   - It always forms at least one group (its K runs 2..m−1), so it merges raters with no block
     too. With 20 near-chance raters it split the panel into two halves in one seed, and it loses
     21.5 points there.
2. **On the real LLM blocks Jaffe finds only part of the block.** On RTE 5 humans + 4 tiny, it
   merged only the two Qwen3 prompts in all 3 shuffles we inspected; ours merged all four tiny
   columns. The residual of Eq. 14 picks K = 8 of 9 there. As a result Jaffe is below no collapsing
   on that panel (81.2 against 85.0).
3. **Their two-stage aggregation is weak here.** D&S inside groups and then across them is
   below no collapsing in 12 of 27 settings. Groups of 2–3 raters give D&S too little to work
   with. Our stand-in may be weaker than their full maximum-likelihood rule.
4. **Binary vs Spearman input barely matters** (88.2 against 88.7 mean). The gap to ours is in the
   group search (score matrix, spectral clustering, residual-based K), not in binarizing.

**Where this leaves the contribution:**
- The detector idea is Jaffe's.
- On these panels our variant (trimmed rank-one fit, partial-correlation threshold) is safer: it
  never hurts, and it finds the full LLM block on RTE.
- Jaffe's is more sensitive to weak blocks.
- A combined rule, or a data-driven threshold, is the open question.
- Not tested: Traganitis and Giannakis (multiclass, robust PCA, needs the number of groups).

Caveats:
- Threshold and trim share (70%) are fixed, not tuned.
- 6 seeds in the simulation and 10 shuffles on RTE, with no intervals.
- The paper's 12 verifier benchmarks were not rerun (not cached locally).
