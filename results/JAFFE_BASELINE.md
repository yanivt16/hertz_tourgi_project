# Jaffe et al. (2016) as a baseline detector (generated from results/residual_blocks_*.json)

AUC. Bold: best in the row (within 0.05). All columns except the two-stage ones are Dawid–Skene → logistic on the collapsed columns; only the detector differs. 'Jaffe groups': Algorithm 1 of Jaffe et al. (AISTATS 2016) via fuse.clustering.jaffe_groups, on the covariance of binarized columns (as in the paper) or on the Spearman matrix. 'Two-stage D&S': D&S inside each Jaffe group, then D&S on the group labels (a stand-in for their L-SML, whose final step is a full maximum-likelihood rule). Columns: number of columns after collapsing (ours 0.4 / Jaffe binary). Simulation: 6 seeds; RTE: 10 slot shuffles.

| Setting | Columns | No collapsing | Ours 0.4 | Ours 0.3 | Jaffe groups (binary, as in paper) | Jaffe groups (Spearman) | Jaffe two-stage D&S | Jaffe two-stage D&S → logistic |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| base | 10 / 8 | **98.4** | **98.4** | **98.4** | 98.4 | 98.4 | 94.6 | 98.2 |
| 5 raters | 5 / 4 | **93.5** | **93.5** | **93.5** | **93.5** | 93.4 | 86.5 | 93.1 |
| hard: d=0.3 | 10 / 7 | 64.3 | 64.3 | 64.3 | 65.2 | **71.1** | 59.9 | 60.8 |
| high quality spread | 10 / 8 | **98.5** | **98.5** | **98.5** | 98.3 | **98.4** | 94.0 | 97.5 |
| 20 near-chance + 2 experts | 20 / 10 | **91.8** | **91.8** | 69.0 | 70.2 | 78.7 | 46.4 | 45.1 |
| 20% weak corr block | 9 / 9 | 97.2 | **97.3** | **97.3** | **97.3** | **97.3** | 94.2 | **97.3** |
| 40% weak corr block | 7 / 6 | 59.9 | **95.6** | **95.6** | 94.2 | 95.4 | 87.3 | 92.4 |
| 40% weak corr block, rho 0.3 | 10 / 6 | 76.5 | 76.5 | 95.4 | **95.4** | 94.7 | 88.5 | 95.2 |
| 40% normal-quality corr block | 7 / 5 | 85.0 | **97.0** | **97.0** | 96.4 | 96.6 | 90.7 | 95.9 |
| binary votes | 10 / 8 | **95.9** | **95.9** | **95.9** | 95.2 | 95.6 | 93.0 | 94.4 |
| RTE 10 humans | 10 / 9 | **96.5** | **96.5** | **96.5** | **96.5** | 96.4 | 96.1 | 96.4 |
| RTE 10 humans + 4 tiny | 11 / 13 | 95.0 | **96.8** | **96.8** | 96.2 | 96.4 | 95.1 | 96.2 |
| RTE 10 humans + 8 small Qwen | 11 / 17 | 94.8 | **97.4** | **97.4** | 94.9 | 94.6 | 94.7 | 95.0 |
| RTE 5 humans + 4 tiny | 6 / 8 | 85.0 | **93.7** | **93.7** | 81.2 | 81.2 | 84.5 | 81.1 |
| RTE 3 humans + 4 tiny | 5 / 6 | 79.8 | **86.7** | 86.3 | 79.5 | 79.7 | 81.9 | 79.5 |
| RTE 10 humans + 6 strong LLM | 11 / 15 | 98.5 | **98.6** | **98.6** | 98.4 | **98.6** | 98.2 | 98.4 |
| RTE 3 humans + 6 strong LLM | 6 / 8 | 98.0 | **98.0** | **98.0** | 97.9 | 97.9 | 97.3 | 97.9 |
| RTE 8 small Qwen only | 4 / 7 | 90.6 | 90.6 | **90.9** | 90.8 | 90.3 | 88.2 | **90.8** |
| VIPER FFDM 15 readers x 93 cases | 15 / 13 | 77.6 | 77.6 | 77.9 | 77.3 | 76.7 | **78.3** | 77.5 |
| VIPER FFDM 8 readers x 201 cases | 8 / 7 | **80.4** | **80.4** | **80.4** | 80.1 | 80.1 | **80.4** | 80.3 |
| VIPER SFM 15 readers x 93 cases | 13 / 13 | 82.3 | 84.0 | 84.0 | 82.7 | 83.0 | **84.2** | 83.2 |
| VIPER SFM 7 readers x 203 cases | 7 / 6 | 83.3 | 83.3 | 83.3 | 82.9 | 82.7 | 79.5 | **83.4** |
| DICES-350 granular | 105 / 120 | 77.8 | **78.3** | 77.6 | 77.9 | 78.1 | 77.9 | 78.2 |
| DICES-350 race x gender groups | 10 / 9 | 79.0 | 79.0 | 79.0 | **79.1** | 78.8 | 77.5 | 79.0 |
| DICES-350 race x gender x age groups | 29 / 28 | 78.5 | 78.5 | 78.7 | **78.8** | 78.5 | 76.8 | **78.8** |
| deepfake demographic groups | 12 / 11 | **91.8** | **91.8** | 91.6 | **91.8** | **91.8** | 90.3 | 91.5 |
| factcheck demographic groups | 10 / 9 | 90.6 | 90.6 | 90.6 | **90.7** | 90.5 | 89.7 | **90.7** |

| Summary over 27 settings | | No collapsing | Ours 0.4 | Ours 0.3 | Jaffe groups (binary, as in paper) | Jaffe groups (Spearman) | Jaffe two-stage D&S | Jaffe two-stage D&S → logistic |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Mean AUC | | 86.7 | 89.3 | 89.1 | 88.2 | 88.7 | 85.4 | 86.9 |
| Mean gap to best | | 3.6 | 1.0 | 1.2 | 2.1 | 1.6 | 4.9 | 3.4 |
| Worst gap | | 35.7 | 19.0 | 22.7 | 21.5 | 13.1 | 45.3 | 46.6 |
| Settings > 1 point below no collapsing | | 0 | 0 | 1 | 2 | 2 | 12 | 4 |
