# Stage 1 dataset survey (10 Nov 2026)

Candidate binary datasets with per-annotator labels and an independent gold label. Profiled
with a one-off script: binary Dawid-Skene with 100 EM iterations, majority-vote ties broken
at random, seed 0. The script is not in the repo; the VIPER numbers come from
[scripts/run_viper.py](../scripts/run_viper.py). FUSE needs third-order moments, so what
matters most is how many rater triplets share enough items.

| | RTE | TEMP | D_PosSent | D_Product | Bluebird/Duck | VIPER (per modality) |
|---|---:|---:|---:|---:|---:|---:|
| Items | 800 | 462 | 1000 | 8315 | 108 | 712-715 |
| Annotators | 164 | 76 | 85 | 176 | 39 | 43 radiologists |
| Labels per item | 10 | 10 | 20 | 3 | 39 | median 15 |
| Missing | 94% | 87% | 76% | 98% | 0% | 67% |
| Rater pairs sharing at least 20 items | 9.4% | 8.7% | 44.3% | 1.4% | 100% | 88% |
| Triplets sharing at least 20 items | 4150 / 721764 | 708 / 8436 | 14550 / 98770 | 29 / 147440 | all 9139 | 6990 / 12341 |
| Positive rate (gold) | 50% | 44% | 47% | 12% | 44% | 16% (cancer) |
| Majority-vote accuracy | 89.0 | 93.9 | 93.7 | 89.7 | 75.9 | AUC 72.9 / 73.8 |
| Dawid-Skene accuracy | 93.0 | 94.2 | 95.9 | 93.8 | 88.9 | AUC 74.4 / 76.7 |
| Item text available | yes | no | no | no | no | no (images on request only) |

Sources:

- RTE and TEMP (Snow et al. 2008): `tutorial/data/*.standardized.csv` in
  [CrowdTruth/CrowdTruth-core](https://github.com/CrowdTruth/CrowdTruth-core).
- D_PosSent, D_Product and Bluebird/Duck (Zheng et al. 2017): the
  [crowd truth inference benchmark](https://zhydhkcws.github.io/crowd_truth_inference/datasets.zip).
- VIPER: [DIDSR/viperData](https://github.com/DIDSR/viperData).

## Recommendation

- **Keep RTE and D_PosSent.** They are sparse, real crowds, and RTE has text for Stage 2.
  They are where missing-data handling matters ([missing_data_plan.md](missing_data_plan.md)).
- **Add Bluebird/Duck** as a dense sanity check. FUSE works there without any missing-data
  machinery, and there is a large Dawid-Skene vs. majority-vote gap to close.
- **Add VIPER** as the radiology dataset. Its gold label is independent of the readers, scores
  are continuous, and FUSE's triplets are well covered. Its missingness depends on the label,
  so evaluate it on dense blocks or with missing-aware methods only. See
  [results/VIPER.md](../results/VIPER.md).
- **Drop TEMP.** Majority vote is already at 94%, so there is no headroom.
- **D_Product only as a stress test.** Too few triplets share items.

Open issue: Stage 2 still lacks a large text dataset with gold.
