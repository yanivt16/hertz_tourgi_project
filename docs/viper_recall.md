# VIPER: recall decisions against cancer, and ranking readers without labels (22 Nov 2026)

Tables: [results/VIPER_RECALL.md](../results/VIPER_RECALL.md) · script:
[scripts/viper_recall_vs_cancer.py](../scripts/viper_recall_vs_cancer.py) · loader: `fuse.viper.load`.

**Question.** Every VIPER read has a recall decision (send the woman for further work-up or not), and
the gold label is cancer at 455-day follow-up. How large is the gap between "recalled" and "cancer",
and can a label-free method tell which radiologists are good at the recall decision?

## Findings

1. **A quarter of cancer reads are not recalled.** 27% (FFDM) and 25% (SFM) of reads of cancer cases
   end in "no recall". 26 and 16 of the 115 cancer cases are recalled by fewer than half of their
   readers. None is missed by every reader (each cancer case has 10–25 readers).
2. **Readers differ a lot.** Per-reader sensitivity ranges from 35–46% to 96–100%; specificity from
   13–14% to 76–78% (medians 73–76% and 52%).
3. **Specificity is low by design.** About half of the non-cancer cases were recalled in the original
   screening (BI-RADS 0) and are hard negatives. Readers recall 60% of those, against 38–40% of
   non-cancer cases that were not recalled originally (BI-RADS 1–3) and 73–75% of cancers.
4. **Dawid–Skene, fitted on recall decisions only, ranks readers well on sensitivity and specificity
   (Spearman +0.86 to +0.97), but that mostly reflects how often each reader recalls.** Recall rate
   alone correlates +0.85 to +0.87 with sensitivity and −0.96 to −0.97 with specificity. On skill
   (sensitivity + specificity − 1), the ranking is moderate: +0.53 (FFDM) and +0.63 (SFM) on the full
   matrix, +0.20 to +0.83 on dense blocks. Readers who recall more are less skilled (−0.53, −0.62).
5. **Dawid–Skene overestimates every reader**: mean estimated sensitivity 0.81–0.82 against a true
   0.72–0.74, and specificity 0.65–0.68 against 0.50–0.51.

## What it means

This is the first check in the project of whether a label-free method identifies the good raters,
rather than only whether its final labels are accurate. On expert radiologists the answer is "partly":
Dawid–Skene separates readers who recall a lot from readers who recall little, but ranks them only
moderately on actual skill. The same check can be run for FUSE and for mixed human–LLM panels.

## Caveats

- `Ctype` is our reading of the case-type field (BI-RADS 0 = recalled in the original screening,
  BI-RADS 1–3 = not recalled). Cancer cases carry no original-screening outcome.
- The full matrix is 67% missing. Dawid–Skene skips missing reads, so it does not see the assignment
  leak directly, but the dense blocks are the cleaner check. They are small (5–15 readers), so their
  correlations are noisy.
- No confidence intervals on the correlations.
