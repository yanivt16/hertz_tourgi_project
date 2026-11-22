# VIPER: ranking radiologists by skill from scores, and blocks among radiologists (22 Nov 2026)

Tables: [results/VIPER_SKILL_BLOCKS.md](../results/VIPER_SKILL_BLOCKS.md) · script:
[scripts/viper_skill_and_blocks.py](../scripts/viper_skill_and_blocks.py). Follows
[viper_recall.md](viper_recall.md).

## A. Does seeing the scores help rank readers by skill?

[viper_recall.md](viper_recall.md) found that, on recall decisions, label-free methods mostly pick up
how often each reader recalls rather than how well they discriminate. Here true skill is each reader's
AUC of the 0–201 suspicion score against cancer, which does not depend on the recall threshold, and
the methods see the scores.

**No, not reliably.**
- On the full matrix the best label-free estimate (binned Dawid–Skene) correlates +0.29 (FFDM) and
  +0.48 (SFM) with true per-reader AUC. Agreement with the other readers gives +0.11 and +0.26.
- On dense blocks the correlations range from −0.60 to +0.98 and no method is consistently ahead.
  The leave-one-out soft-label AUC from binned Dawid–Skene is the most stable (SFM, 13–15 readers:
  +0.63 to +0.80). FUSE's is erratic (+0.74 on one block, 0.00 on the next, nested one).
- One reason it is hard: readers differ little in skill. True AUCs span 0.57–0.78 (FFDM) and
  0.58–0.80 (SFM), while their recall rates span a much wider range. With 5–15 readers per block,
  a narrow spread of skill is hard to rank.

## B. Do radiologists form blocks?

**Weak evidence, only on SFM.**
- SFM, at threshold 0.4: two pairs, reader08 + reader11 (residual correlation 0.40–0.43) and
  reader03 + reader12 (0.44–0.45), found in the 15- and 14-reader blocks. Merging them is what raised
  binned Dawid–Skene → logistic from 82.3 to 84.0 in the detector runs.
- FFDM: nothing at 0.4. At 0.3, weak pairs (residual 0.30–0.35), mostly reader09 + reader14.
- The SFM pairs are not merged on FFDM, so they are not stable across modalities, and each pair is on
  different reading tracks (4 and 3; 3 and 4). Reading track is the only reader-level field in the
  data, so nothing in it explains the pairs.

## Caveats

- The 15- and 14-reader blocks share almost all their cases, so finding the same pair in both is not
  independent confirmation.
- With about 93 cases, a residual correlation of 0.4 is within reach of chance across the 105 reader
  pairs of a 15-reader block. We ran no permutation test.
- Spearman correlations over 5–15 readers are very noisy; no confidence intervals.
