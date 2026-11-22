# Shared-error blocks on the paper's Best-of-N benchmarks (generated from results/paper_blocks.json)

Best-of-N selection accuracy (%), all methods label-free. Script: `scripts/run_paper_blocks.py`. Write-up: `docs/paper_blocks.md`. Bold: best in the row (within 0.05). `m`: verifiers (paper Table 6). 'ties' uses a tie-aware Spearman in the residual detector; 'repo' uses `fuse.clustering.spearman_matrix` as it is.

## Batched mode (one ensemble per setting)

| Setting | m | FUSE paper | D&S | D&S → logistic | Mean z | Collapse (ties 0.4) → D&S → logistic | Collapse (ties 0.4) → FUSE | Collapse (ties 0.3) → D&S → logistic | Collapse (repo 0.4) → D&S → logistic | FUSE, median thresholds | D&S on FUSE's thresholds → logistic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gpqa-70b | 32 | 66.9 | 56.6 | 55.2 | 58.8 | 56.2 | 55.3 | 55.1 | 56.0 | 55.6 | **67.3** |
| gpqa-diamond-70b | 32 | 59.6 | 54.0 | 52.6 | 55.1 | 52.9 | **61.1** | 52.0 | 52.9 | 53.1 | 58.6 |
| math500-70b | 32 | 94.1 | 86.7 | 87.0 | 92.6 | 86.4 | 92.5 | 86.6 | 86.6 | 88.2 | **94.4** |
| mmlu-70b | 32 | 93.7 | 85.7 | 89.6 | 93.0 | 89.3 | 90.8 | 89.3 | 89.3 | 90.3 | **94.2** |
| mmlu-pro-70b | 32 | **91.6** | 77.0 | 83.8 | 86.8 | 83.6 | 89.2 | 82.8 | 82.8 | 85.0 | **91.6** |
| gpqa-8b | 16 | 37.9 | 32.1 | 35.2 | **39.8** | 35.4 | 37.3 | 35.2 | 35.4 | 34.8 | 36.8 |
| gpqa-diamond-8b | 16 | 31.7 | 32.5 | 36.8 | **42.4** | 37.0 | 31.6 | 37.0 | 37.0 | 37.7 | 37.3 |
| math500-8b | 16 | 55.2 | 32.8 | 53.5 | **73.2** | 53.1 | 53.0 | 53.7 | 53.1 | 53.1 | 54.6 |
| mmlu-8b | 16 | 70.6 | 76.6 | 70.5 | **82.6** | 70.2 | 69.7 | 70.0 | 70.2 | 69.9 | 74.2 |
| mmlu-pro-8b | 16 | 60.3 | 55.0 | 65.8 | **68.8** | 63.4 | 67.0 | 66.2 | 63.4 | 55.3 | 66.4 |
| hle | 7 | 52.2 | 51.6 | **53.2** | 51.2 | **53.2** | 52.2 | **53.2** | **53.2** | 52.3 | 52.6 |
| imo | 9 | **62.1** | 59.6 | 59.6 | **62.1** | 59.6 | **62.1** | 59.6 | 59.6 | **62.1** | **62.1** |

Gap to FUSE paper (batch mode), in points of selection accuracy, over 12 settings:

| Method | Mean gap | Settings ahead by > 1 | Within ±1 | Behind by > 1 | Worst |
|---|---:|---:|---:|---:|---:|
| D&S → logistic, no collapsing | -2.8 | 3 | 1 | 8 | -11.7 |
| Mean z | +2.5 | 5 | 2 | 5 | -8.0 |
| Collapse (ties 0.4) → D&S → logistic | -3.0 | 3 | 1 | 8 | -10.7 |
| Collapse (ties 0.3) → D&S → logistic | -2.9 | 3 | 1 | 8 | -11.7 |
| Collapse (repo 0.4) → D&S → logistic | -3.0 | 3 | 1 | 8 | -10.9 |
| Collapse (repo 0.3) → D&S → logistic | -2.9 | 3 | 1 | 8 | -12.1 |
| Collapse (ties 0.4) → FUSE | -1.2 | 2 | 5 | 5 | -11.6 |
| FUSE, median thresholds | -3.2 | 1 | 3 | 8 | -11.3 |
| D&S on FUSE's thresholds → logistic | +1.2 | 3 | 7 | 2 | -1.1 |

## Per-query mode (one ensemble per question)

| Setting | m | FUSE paper | D&S | D&S → logistic | Mean z | Collapse (ties 0.4) → D&S → logistic | Collapse (ties 0.4) → FUSE | Collapse (ties 0.3) → D&S → logistic | Collapse (repo 0.4) → D&S → logistic | FUSE, median thresholds | D&S on FUSE's thresholds → logistic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gpqa-70b | 32 | **60.5** | 59.5 | 56.8 | 59.1 | 56.8 | 59.6 | 56.5 | 55.7 | 57.1 | 59.4 |
| gpqa-diamond-70b | 32 | 55.1 | **56.6** | 52.5 | 54.5 | 53.5 | 55.6 | 53.0 | 52.5 | 54.0 | 54.5 |
| math500-70b | 32 | 91.2 | 84.0 | 83.6 | **92.4** | 85.2 | 88.4 | 85.4 | 84.0 | 87.0 | 90.3 |
| mmlu-70b | 32 | 92.8 | 87.0 | 88.9 | 92.5 | 89.0 | **93.2** | 89.6 | 89.3 | 90.1 | 92.6 |
| mmlu-pro-70b | 32 | **88.2** | 79.6 | 82.8 | 87.0 | 83.4 | 87.0 | 84.0 | 83.0 | 84.4 | 87.4 |
| gpqa-8b | 16 | 38.7 | 30.2 | 35.8 | **40.2** | 35.9 | 37.9 | 36.5 | 36.1 | 35.3 | 38.3 |
| gpqa-diamond-8b | 16 | 40.9 | 31.6 | 36.9 | **42.9** | 34.8 | 40.9 | 37.4 | 34.8 | 37.9 | 41.8 |
| math500-8b | 16 | 66.0 | 59.7 | 56.4 | **74.6** | 56.6 | 64.2 | 56.0 | 56.8 | 59.8 | 63.9 |
| mmlu-8b | 16 | 80.5 | 76.3 | 79.3 | **82.9** | 78.2 | 80.5 | 78.7 | 78.0 | 79.6 | 80.3 |
| mmlu-pro-8b | 16 | **69.0** | 58.0 | 63.8 | 68.8 | 64.2 | 66.0 | 65.8 | 65.0 | 65.8 | 66.1 |
| hle | 7 | **53.4** | 50.1 | 49.9 | 52.2 | 51.2 | **53.4** | 50.9 | 53.2 | 51.4 | 53.4 |
| imo | 9 | 57.1 | 56.2 | 57.6 | **62.1** | 56.8 | 56.5 | 57.7 | 58.5 | 59.0 | 59.2 |

Gap to FUSE paper (query mode), in points of selection accuracy, over 12 settings:

| Method | Mean gap | Settings ahead by > 1 | Within ±1 | Behind by > 1 | Worst |
|---|---:|---:|---:|---:|---:|
| D&S → logistic, no collapsing | -4.1 | 0 | 1 | 11 | -9.6 |
| Mean z | +1.3 | 6 | 3 | 3 | -1.4 |
| Collapse (ties 0.4) → D&S → logistic | -4.0 | 0 | 1 | 11 | -9.4 |
| Collapse (ties 0.3) → D&S → logistic | -3.5 | 0 | 1 | 11 | -10.0 |
| Collapse (repo 0.4) → D&S → logistic | -3.9 | 1 | 1 | 10 | -9.2 |
| Collapse (repo 0.3) → D&S → logistic | -3.5 | 1 | 1 | 10 | -9.4 |
| Collapse (ties 0.4) → FUSE | -0.8 | 0 | 8 | 4 | -3.0 |
| FUSE, median thresholds | -2.7 | 1 | 1 | 10 | -6.2 |
| D&S on FUSE's thresholds → logistic | -0.5 | 1 | 8 | 3 | -2.9 |

## What the detector merged (batched mode)

| Setting | Detector | Columns after | Merged groups |
|---|---|---:|---|
| gpqa-70b | ties@0.4 | 26 / 32 | {DecisionTreeReward8B, QRM}; {QwenPRM_avg, QwenPRM_min}; {EurusPRMStage2_avg, EurusPRMStage2_min}; {QRMGemma, SkyworksGemma}; {DeepSeekQwen32B (verdict), Llama-3.3-70B-Instruct (verdict), Meta-Llama-3.1-405B-Instruct-quantized.w8a16 (verdict)} |
| gpqa-70b | ties@0.3 | 21 / 32 | {DecisionTreeReward8B, QRM}; {QwenPRM_avg, QwenPRM_max, QwenPRM_min}; {EurusPRMStage2_avg, EurusPRMStage2_min}; {EurusPRMStage1_max, EurusPRMStage1_min}; {DecisionTreeReward27B, LDLRewardGemma, QRMGemma, SkyworksGemma}; {DeepSeekQwen32B (verdict), Llama-3.3-70B-Instruct (verdict), Meta-Llama-3.1-405B-Instruct-quantized.w8a16 (verdict), Mixtral-8x22B-Instruct-v0.1 (verdict)} |
| gpqa-70b | repo@0.4 | 22 / 32 | {DecisionTreeReward8B, QRM}; {QRMGemma, SkyworksGemma}; {QwenPRM_avg, QwenPRM_min}; {EurusPRMStage2_avg, EurusPRMStage2_min}; {DeepSeekLlama70B (verdict), WizardLM-2-8x22B (verdict)}; {DeepSeekQwen32B (verdict), Llama-3.3-70B-Instruct (verdict), Meta-Llama-3.1-405B-Instruct-quantized.w8a16 (verdict), Mixtral-8x22B-Instruct-v0.1 (verdict), Qwen/Qwen2.5-72B-Instruct (verdict), SkyT1 (verdict)} |
| gpqa-diamond-70b | ties@0.4 | 26 / 32 | {DecisionTreeReward8B, QRM}; {QRMGemma, SkyworksGemma}; {DeepSeekQwen32B (verdict), Llama-3.3-70B-Instruct (verdict), Meta-Llama-3.1-405B-Instruct-quantized.w8a16 (verdict)}; {QwenPRM_avg, QwenPRM_min}; {EurusPRMStage2_avg, EurusPRMStage2_min} |
| gpqa-diamond-70b | ties@0.3 | 21 / 32 | {DecisionTreeReward8B, QRM}; {DecisionTreeReward27B, LDLRewardGemma, QRMGemma, SkyworksGemma}; {Mixtral-8x22B-Instruct-v0.1 (verdict), WizardLM-2-8x22B (verdict)}; {DeepSeekQwen32B (verdict), Llama-3.3-70B-Instruct (verdict), Meta-Llama-3.1-405B-Instruct-quantized.w8a16 (verdict)}; {QwenPRM_avg, QwenPRM_max, QwenPRM_min}; {EurusPRMStage2_avg, EurusPRMStage2_min}; {EurusPRMStage1_max, EurusPRMStage1_min} |
| gpqa-diamond-70b | repo@0.4 | 22 / 32 | {DecisionTreeReward8B, QRM}; {QwenPRM_avg, QwenPRM_min}; {EurusPRMStage2_avg, EurusPRMStage2_min}; {DecisionTreeReward27B, QRMGemma}; {DeepSeekLlama70B (verdict), WizardLM-2-8x22B (verdict)}; {DeepSeekQwen32B (verdict), Llama-3.3-70B-Instruct (verdict), Meta-Llama-3.1-405B-Instruct-quantized.w8a16 (verdict), Mixtral-8x22B-Instruct-v0.1 (verdict), Qwen/Qwen2.5-72B-Instruct (verdict), SkyT1 (verdict)} |
| math500-70b | ties@0.4 | 26 / 32 | {DecisionTreeReward27B, QRMGemma}; {DecisionTreeReward8B, QRM}; {Llama-3.3-70B-Instruct (verdict), Meta-Llama-3.1-405B-Instruct-quantized.w8a16 (verdict)}; {EurusPRMStage2_avg, EurusPRMStage2_min}; {DeepSeekLlama70B (verdict), DeepSeekQwen32B (verdict)}; {QwenPRM_avg, QwenPRM_min} |
| math500-70b | ties@0.3 | 19 / 32 | {EurusPRMStage1_max, LDLRewardGemma}; {DecisionTreeReward27B, QRMGemma}; {GRMGemma, SkyworksGemma}; {DecisionTreeReward8B, QRM}; {InternLM2Reward7B, InternLM2RewardModel}; {Llama-3.3-70B-Instruct (verdict), Meta-Llama-3.1-405B-Instruct-quantized.w8a16 (verdict), Mixtral-8x22B-Instruct-v0.1 (verdict)}; {EurusPRMStage2_avg, EurusPRMStage2_min}; {Qwen/Qwen2.5-72B-Instruct (verdict), WizardLM-2-8x22B (verdict)}; {DeepSeekLlama70B (verdict), DeepSeekQwen32B (verdict), Qwen72B}; {QwenPRM_avg, QwenPRM_min, SkyT1 (verdict)} |
| math500-70b | repo@0.4 | 20 / 32 | {InternLM2Reward7B, InternLM2RewardModel}; {QwenPRM_avg, QwenPRM_min}; {EurusPRMStage2_avg, EurusPRMStage2_min}; {Llama-3.3-70B-Instruct (verdict), Meta-Llama-3.1-405B-Instruct-quantized.w8a16 (verdict), Mixtral-8x22B-Instruct-v0.1 (verdict)}; {DeepSeekLlama70B (verdict), DeepSeekQwen32B (verdict), Qwen/Qwen2.5-72B-Instruct (verdict), SkyT1 (verdict), WizardLM-2-8x22B (verdict)}; {DecisionTreeReward27B, QRMGemma}; {GRMGemma, SkyworksGemma}; {DecisionTreeReward8B, QRM} |
| mmlu-70b | ties@0.4 | 23 / 32 | {InternLM2Reward7B, InternLM2RewardModel}; {DeepSeekLlama70B (verdict), Meta-Llama-3.1-405B-Instruct-quantized.w8a16 (verdict)}; {EurusPRMStage1_min, EurusPRMStage2_avg, EurusPRMStage2_min}; {DeepSeekQwen32B (verdict), Llama-3.3-70B-Instruct (verdict)}; {QwenPRM_avg, QwenPRM_min}; {DecisionTreeReward8B, QRM}; {DecisionTreeReward27B, QRMGemma, SkyworksGemma} |
| mmlu-70b | ties@0.3 | 20 / 32 | {InternLM2Reward7B, InternLM2RewardModel}; {DeepSeekLlama70B (verdict), Meta-Llama-3.1-405B-Instruct-quantized.w8a16 (verdict), Qwen/Qwen2.5-72B-Instruct (verdict)}; {EurusPRMStage1_min, EurusPRMStage2_avg, EurusPRMStage2_min}; {DeepSeekQwen32B (verdict), Llama-3.3-70B-Instruct (verdict), SkyT1 (verdict), WizardLM-2-8x22B (verdict)}; {QwenPRM_avg, QwenPRM_min}; {DecisionTreeReward8B, QRM}; {DecisionTreeReward27B, QRMGemma, SkyworksGemma} |
| mmlu-70b | repo@0.4 | 20 / 32 | {DecisionTreeReward8B, QRM}; {DecisionTreeReward27B, QRMGemma, SkyworksGemma}; {InternLM2Reward7B, InternLM2RewardModel}; {EurusPRMStage1_min, EurusPRMStage2_avg, EurusPRMStage2_min}; {QwenPRM_avg, QwenPRM_min}; {DeepSeekQwen32B (verdict), Llama-3.3-70B-Instruct (verdict), SkyT1 (verdict), WizardLM-2-8x22B (verdict)}; {DeepSeekLlama70B (verdict), Meta-Llama-3.1-405B-Instruct-quantized.w8a16 (verdict), Qwen/Qwen2.5-72B-Instruct (verdict)} |
| mmlu-pro-70b | ties@0.4 | 22 / 32 | {EurusPRMStage1_min, EurusPRMStage2_avg, EurusPRMStage2_min}; {QwenPRM_avg, QwenPRM_min}; {DeepSeekLlama70B (verdict), DeepSeekQwen32B (verdict), Meta-Llama-3.1-405B-Instruct-quantized.w8a16 (verdict), WizardLM-2-8x22B (verdict)}; {InternLM2Reward7B, InternLM2RewardModel}; {DecisionTreeReward27B, INFORM, QRMGemma}; {DecisionTreeReward8B, QRM} |
| mmlu-pro-70b | ties@0.3 | 18 / 32 | {EurusPRMStage1_min, EurusPRMStage2_avg, EurusPRMStage2_min}; {QwenPRM_avg, QwenPRM_max, QwenPRM_min}; {DeepSeekLlama70B (verdict), DeepSeekQwen32B (verdict), GPM, Meta-Llama-3.1-405B-Instruct-quantized.w8a16 (verdict), WizardLM-2-8x22B (verdict)}; {InternLM2Reward7B, InternLM2RewardModel}; {DecisionTreeReward27B, INFORM, LDLRewardGemma, QRMGemma, SkyworksGemma}; {DecisionTreeReward8B, QRM} |
| mmlu-pro-70b | repo@0.4 | 18 / 32 | {DecisionTreeReward8B, QRM}; {DecisionTreeReward27B, INFORM, LDLRewardGemma, QRMGemma, SkyworksGemma}; {EurusPRMStage1_min, EurusPRMStage2_avg, EurusPRMStage2_min}; {Mixtral-8x22B-Instruct-v0.1 (verdict), Qwen/Qwen2.5-72B-Instruct (verdict), SkyT1 (verdict)}; {DeepSeekLlama70B (verdict), DeepSeekQwen32B (verdict), Meta-Llama-3.1-405B-Instruct-quantized.w8a16 (verdict), WizardLM-2-8x22B (verdict)}; {QwenPRM_avg, QwenPRM_min}; {InternLM2Reward7B, InternLM2RewardModel} |
| gpqa-8b | ties@0.4 | 14 / 16 | {QwenPRM_avg, QwenPRM_min}; {EurusPRMStage1_avg, EurusPRMStage1_max} |
| gpqa-8b | ties@0.3 | 13 / 16 | {QwenPRM_avg, QwenPRM_min}; {EurusPRMStage1_avg, EurusPRMStage1_max, EurusPRMStage2_max} |
| gpqa-8b | repo@0.4 | 14 / 16 | {EurusPRMStage1_avg, EurusPRMStage1_max}; {QwenPRM_avg, QwenPRM_min} |
| gpqa-diamond-8b | ties@0.4 | 14 / 16 | {EurusPRMStage1_avg, EurusPRMStage1_max}; {QwenPRM_avg, QwenPRM_min} |
| gpqa-diamond-8b | ties@0.3 | 14 / 16 | {EurusPRMStage1_avg, EurusPRMStage1_max}; {QwenPRM_avg, QwenPRM_min} |
| gpqa-diamond-8b | repo@0.4 | 14 / 16 | {QwenPRM_avg, QwenPRM_min}; {EurusPRMStage1_avg, EurusPRMStage1_max} |
| math500-8b | ties@0.4 | 13 / 16 | {QwenPRM_avg, QwenPRM_min}; {EurusPRMStage1_avg, EurusPRMStage1_max, EurusPRMStage2_min} |
| math500-8b | ties@0.3 | 9 / 16 | {EurusPRMStage1_avg, EurusPRMStage1_max, EurusPRMStage2_max, EurusPRMStage2_min, QwenPRM_avg, QwenPRM_min}; {QRM, Skyworks}; {OffsetBias, URM} |
| math500-8b | repo@0.4 | 13 / 16 | {QwenPRM_avg, QwenPRM_min}; {EurusPRMStage1_avg, EurusPRMStage1_max, EurusPRMStage2_min} |
| mmlu-8b | ties@0.4 | 13 / 16 | {ArmorRM, OffsetBias}; {EurusPRMStage1_avg, EurusPRMStage1_max}; {QwenPRM_avg, QwenPRM_min} |
| mmlu-8b | ties@0.3 | 11 / 16 | {ArmorRM, OffsetBias}; {EurusPRMStage1_avg, EurusPRMStage1_max, EurusPRMStage2_max}; {QwenPRM_avg, QwenPRM_min}; {GRM, InternLM2Reward7B} |
| mmlu-8b | repo@0.4 | 13 / 16 | {ArmorRM, OffsetBias}; {EurusPRMStage1_avg, EurusPRMStage1_max}; {QwenPRM_avg, QwenPRM_min} |
| mmlu-pro-8b | ties@0.4 | 13 / 16 | {EurusPRMStage1_avg, EurusPRMStage1_max}; {QwenPRM_avg, QwenPRM_min}; {GRM, InternLM2Reward7B} |
| mmlu-pro-8b | ties@0.3 | 11 / 16 | {GRMLlama32, Skyworks}; {EurusPRMStage1_avg, EurusPRMStage1_max, EurusPRMStage2_min}; {QwenPRM_avg, QwenPRM_min}; {GRM, InternLM2Reward7B} |
| mmlu-pro-8b | repo@0.4 | 13 / 16 | {EurusPRMStage1_avg, EurusPRMStage1_max}; {QwenPRM_avg, QwenPRM_min}; {GRM, InternLM2Reward7B} |
| hle | ties@0.4 | 7 / 7 | none |
| hle | ties@0.3 | 7 / 7 | none |
| hle | repo@0.4 | 7 / 7 | none |
| imo | ties@0.4 | 9 / 9 | none |
| imo | ties@0.3 | 9 / 9 | none |
| imo | repo@0.4 | 9 / 9 | none |

## How often it merged (per-query mode)

| Setting | m | ties 0.4: mean columns | ties 0.4: queries with a merge | repo 0.4: mean columns | repo 0.4: queries with a merge |
|---|---:|---:|---:|---:|---:|
| gpqa-70b | 32 | 25.2 | 100% | 19.4 | 100% |
| gpqa-diamond-70b | 32 | 25.3 | 99% | 19.6 | 100% |
| math500-70b | 32 | 25.7 | 98% | 20.8 | 100% |
| mmlu-70b | 32 | 21.9 | 100% | 16.4 | 100% |
| mmlu-pro-70b | 32 | 22.7 | 100% | 18.0 | 100% |
| gpqa-8b | 16 | 12.2 | 99% | 12.2 | 99% |
| gpqa-diamond-8b | 16 | 12.3 | 99% | 12.2 | 99% |
| math500-8b | 16 | 10.9 | 100% | 10.9 | 100% |
| mmlu-8b | 16 | 12.2 | 100% | 12.1 | 100% |
| mmlu-pro-8b | 16 | 12.8 | 100% | 12.8 | 100% |
| hle | 7 | 6.6 | 33% | 6.1 | 66% |
| imo | 9 | 8.3 | 46% | 7.3 | 83% |
