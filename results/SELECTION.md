# Label-free variant selection

The rule was committed (fba0117) before it was run on these settings or joined with
accuracy; its constants (R < 0.5, median) were not tuned. The per-setting accuracies in
VARIANTS.md were already known when it was designed.

Selected variant per setting and mode (both predictors use the same pick), from
`fuse.selection.select_variant`. R_t is the label share at clustering threshold t
(cluster when < 0.5); MV pass is the fraction of queries (or 1/0 in batched mode)
where the majority-vote column fits the rank-one model at least as well as the median verifier.

| Setting | Mode | R@0.3 | R@0.5 | R@0.7 | MV pass | Selected |
|---|---|---:|---:|---:|---:|---|
| gpqa-70b | query | 0.78 | 0.69 | 0.62 | 0.23 | base |
| gpqa-70b | batch | 0.62 | 0.78 | 0.85 | 0.00 | base |
| gpqa-diamond-70b | query | 0.73 | 0.67 | 0.59 | 0.23 | base |
| gpqa-diamond-70b | batch | 0.69 | 0.72 | 0.79 | 0.00 | base |
| math500-70b | query | 0.81 | 0.86 | 0.87 | 0.03 | base |
| math500-70b | batch | 0.38 | 0.45 | 0.86 | 0.00 | cluster@0.3 |
| mmlu-70b | query | 0.76 | 0.93 | 0.95 | 0.01 | base |
| mmlu-70b | batch | 0.69 | 0.47 | 0.81 | 0.00 | cluster@0.5 |
| mmlu-pro-70b | query | 0.66 | 0.86 | 0.90 | 0.03 | base |
| mmlu-pro-70b | batch | 0.20 | 0.53 | 0.63 | 0.00 | cluster@0.3 |
| gpqa-8b | query | 0.70 | 0.79 | 0.92 | 0.01 | base |
| gpqa-8b | batch | 1.03 | 1.19 | 0.86 | 0.00 | base |
| gpqa-diamond-8b | query | 0.68 | 0.83 | 0.96 | 0.02 | base |
| gpqa-diamond-8b | batch | 1.13 | 1.13 | 0.94 | 0.00 | base |
| math500-8b | query | 0.37 | 0.87 | 0.89 | 0.00 | cluster@0.3 |
| math500-8b | batch | 0.05 | 1.92 | 1.08 | 0.00 | cluster@0.3 |
| mmlu-8b | query | 0.28 | 0.97 | 0.93 | 0.00 | cluster@0.3 |
| mmlu-8b | batch | 1.38 | 1.48 | 0.88 | 0.00 | base |
| mmlu-pro-8b | query | 0.64 | 0.83 | 0.82 | 0.02 | base |
| mmlu-pro-8b | batch | 0.65 | 0.94 | 0.89 | 0.00 | base |
| hle | query | 0.23 | 0.32 | 0.31 |  | cluster@0.3 |
| hle | batch | 1.03 |  |  |  | base |
| imo | query | 0.46 | 0.47 | 0.55 | 0.49 | cluster@0.3 |
| imo | batch | 1.42 |  |  | 0.00 | base |

## Mean accuracy over the 12 settings

Best-in-hindsight picks the best variant per setting using the labels (an upper bound,
not a method). Best fixed variant is the one variant with the highest mean over all
settings, also chosen with hindsight.

| Mode | Predictor | base | selected | best fixed variant | best in hindsight | selected vs base (wins/ties/losses) | selected − base without math500-8b |
|---|---|---:|---:|---:|---:|---|---:|
| query | logistic | 66.1 | 66.6 | 66.5 (cluster@0.5) | 68.4 | 3/8/1 | -0.09 |
| query | posterior | 66.4 | 66.1 | 66.9 (cluster@0.7) | 68.1 | 1/8/3 | -0.15 |
| batch | logistic | 64.7 | 66.3 | 66.6 (cluster@0.3) | 69.1 | 2/8/2 | -0.04 |
| batch | posterior | 67.4 | 68.2 | 67.4 (base) | 70.2 | 1/8/3 | -0.43 |

## query mode, logistic predictor

| Setting | base | selected | variant | best in hindsight | best variant |
|---|---:|---:|---|---:|---|
| gpqa-70b | 60.5 | 60.5 | base | 61.5 | cluster@0.7+mv |
| gpqa-diamond-70b | 55.1 | 55.1 | base | 57.1 | cluster@0.7+mv |
| math500-70b | 91.2 | 91.2 | base | 91.5 | cluster@0.5+mv |
| mmlu-70b | 92.8 | 92.8 | base | 94.0 | cluster@0.5+mv |
| mmlu-pro-70b | 88.2 | 88.2 | base | 88.4 | cluster@0.7 |
| gpqa-8b | 38.7 | 38.7 | base | 38.7 | base |
| gpqa-diamond-8b | 40.9 | 40.9 | base | 41.6 | cluster@0.7 |
| math500-8b | 66.0 | 72.3 | cluster@0.3 | 79.3 | cluster@0.5 |
| mmlu-8b | 80.5 | 80.6 | cluster@0.3 | 83.2 | cluster@0.5+mv |
| mmlu-pro-8b | 69.0 | 69.0 | base | 69.3 | mv |
| hle | 53.4 | 51.3 | cluster@0.3 | 53.4 | base |
| imo | 57.1 | 58.1 | cluster@0.3 | 62.8 | cluster@0.7+mv |
| **mean** | 66.1 | 66.6 | | 68.4 | |

## query mode, posterior predictor

| Setting | base | selected | variant | best in hindsight | best variant |
|---|---:|---:|---|---:|---|
| gpqa-70b | 61.4 | 61.4 | base | 62.0 | mv |
| gpqa-diamond-70b | 55.8 | 55.8 | base | 57.1 | mv |
| math500-70b | 91.8 | 91.8 | base | 91.8 | base |
| mmlu-70b | 92.9 | 92.9 | base | 93.8 | cluster@0.7 |
| mmlu-pro-70b | 88.2 | 88.2 | base | 89.5 | cluster@0.7 |
| gpqa-8b | 38.4 | 38.4 | base | 39.8 | mv |
| gpqa-diamond-8b | 39.4 | 39.4 | base | 40.2 | cluster@0.7+mv |
| math500-8b | 70.0 | 67.5 | cluster@0.3 | 76.8 | cluster@0.7+mv |
| mmlu-8b | 82.3 | 78.9 | cluster@0.3 | 83.5 | cluster@0.5+mv |
| mmlu-pro-8b | 68.9 | 68.9 | base | 69.1 | cluster@0.7 |
| hle | 52.4 | 50.8 | cluster@0.3 | 52.4 | base |
| imo | 55.7 | 59.0 | cluster@0.3 | 61.8 | cluster@0.5+mv |
| **mean** | 66.4 | 66.1 | | 68.1 | |

## batch mode, logistic predictor

| Setting | base | selected | variant | best in hindsight | best variant |
|---|---:|---:|---|---:|---|
| gpqa-70b | 66.9 | 66.9 | base | 66.9 | base |
| gpqa-diamond-70b | 59.6 | 59.6 | base | 61.1 | cluster@0.7 |
| math500-70b | 94.1 | 94.6 | cluster@0.3 | 94.6 | cluster@0.3 |
| mmlu-70b | 93.7 | 93.3 | cluster@0.5 | 94.3 | cluster@0.7 |
| mmlu-pro-70b | 91.6 | 91.0 | cluster@0.3 | 91.6 | base |
| gpqa-8b | 37.9 | 37.9 | base | 37.9 | base |
| gpqa-diamond-8b | 31.7 | 31.7 | base | 36.4 | cluster@0.3+mv |
| math500-8b | 55.2 | 75.1 | cluster@0.3 | 76.4 | cluster@0.3+mv |
| mmlu-8b | 70.6 | 70.6 | base | 84.0 | cluster@0.3 |
| mmlu-pro-8b | 60.3 | 60.3 | base | 70.4 | cluster@0.3 |
| hle | 52.2 | 52.2 | base | 52.4 | cluster@0.3 |
| imo | 62.1 | 62.1 | base | 62.9 | cluster@0.3 |
| **mean** | 64.7 | 66.3 | | 69.1 | |

## batch mode, posterior predictor

| Setting | base | selected | variant | best in hindsight | best variant |
|---|---:|---:|---|---:|---|
| gpqa-70b | 67.0 | 67.0 | base | 67.0 | base |
| gpqa-diamond-70b | 64.4 | 64.4 | base | 64.4 | base |
| math500-70b | 91.3 | 88.6 | cluster@0.3 | 93.7 | cluster@0.3+mv |
| mmlu-70b | 93.9 | 93.4 | cluster@0.5 | 94.1 | cluster@0.7 |
| mmlu-pro-70b | 90.4 | 89.0 | cluster@0.3 | 90.8 | cluster@0.7 |
| gpqa-8b | 37.3 | 37.3 | base | 37.3 | base |
| gpqa-diamond-8b | 42.9 | 42.9 | base | 44.6 | mv |
| math500-8b | 60.2 | 75.1 | cluster@0.3 | 76.4 | cluster@0.3+mv |
| mmlu-8b | 86.7 | 86.7 | base | 86.7 | base |
| mmlu-pro-8b | 61.5 | 61.5 | base | 71.1 | cluster@0.7+mv |
| hle | 52.1 | 52.1 | base | 53.9 | cluster@0.3 |
| imo | 60.8 | 60.8 | base | 62.1 | cluster@0.5+mv |
| **mean** | 67.4 | 68.2 | | 70.2 | |
