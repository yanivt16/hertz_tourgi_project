# Collapsing raters who share errors (generated from results/residual_blocks_*.json and results/tci_fix_*.json)

AUC. Bold: best in the row (within 0.05). 'Residual t': raters clustered on residual correlation at threshold t (fuse.clustering.residual_groups), each cluster averaged into one column. 'Total-corr clusters': the existing |Spearman| > 0.5 clustering. Columns after collapsing at 0.2/0.3/0.4 are in the JSON files (key 'n columns').
Simulation: 6 seeds; RTE: 10 slot shuffles. The rho 0.3 block scenario was run only in this test, so the total-correlation column is empty there.

| Setting | Columns at 0.3 / 0.4 | D&S 5 bins | Mean z | D&S → logistic | FUSE paper | Total-corr clusters: D&S → logistic | Residual 0.3: D&S → logistic | Residual 0.4: D&S → logistic | Residual 0.4: mean z | Residual 0.4: FUSE median | Residual 0.4: FUSE paper |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 10 → 10 / 10 | 98.0 | 98.2 | **98.4** | 98.0 | **98.4** | **98.4** | **98.4** | 98.2 | **98.5** | 98.0 |
| 5 raters | 5 → 5 / 5 | 92.4 | 93.4 | **93.5** | 91.8 | **93.5** | **93.5** | **93.5** | 93.4 | **93.5** | 91.8 |
| hard: d=0.3 | 10 → 10 / 10 | 57.4 | **74.5** | 64.3 | 62.3 | 64.3 | 64.3 | 64.3 | **74.5** | 66.0 | 62.3 |
| high quality spread | 10 → 10 / 10 | 98.1 | 96.9 | **98.5** | 97.0 | **98.5** | **98.5** | **98.5** | 96.9 | 98.4 | 97.0 |
| 20 near-chance + 2 experts | 20 → 20 / 20 | 89.9 | 75.2 | **91.8** | 72.2 | **91.8** | 69.0 | **91.8** | 75.2 | 89.8 | 72.2 |
| 20% weak corr block | 10 → 9 / 9 | 96.5 | 95.0 | 97.2 | 96.9 | **97.3** | **97.3** | **97.3** | 96.4 | 97.2 | 96.3 |
| 40% weak corr block | 10 → 7 / 7 | 59.8 | 86.2 | 59.9 | 76.3 | **95.6** | **95.6** | **95.6** | 94.3 | 95.2 | 94.7 |
| 40% weak corr block, rho 0.3 | 10 → 8 / 10 |  | 89.1 | 76.5 | 94.2 |  | **95.4** | 76.5 | 89.1 | 78.3 | 94.2 |
| 40% normal-quality corr block | 10 → 7 / 7 | 85.0 | 94.7 | 85.0 | 91.0 | **97.0** | **97.0** | **97.0** | 96.5 | 96.9 | 96.7 |
| binary votes | 10 → 10 / 10 |  | 95.0 | **95.9** | 95.8 | 95.8 | **95.9** | **95.9** | 95.0 | 95.8 | 95.8 |
| RTE 10 humans | 10 → 10 / 10 |  | **96.6** | 96.5 | 96.5 | 96.5 | 96.5 | 96.5 | **96.6** | 96.5 | 96.5 |
| RTE 10 humans + 4 tiny | 14 → 11 / 11 | 91.1 | 95.9 | 95.0 | 96.7 | **96.8** | **96.8** | **96.8** | **96.8** | **96.8** | 96.7 |
| RTE 10 humans + 8 small Qwen | 18 → 11 / 11 | 93.8 | 96.3 | 94.8 | 96.4 | **97.4** | **97.4** | **97.4** | 97.2 | **97.4** | 97.2 |
| RTE 5 humans + 4 tiny | 9 → 6 / 6 | 84.9 | 92.3 | 85.0 | 92.6 | 93.7 | 93.7 | 93.7 | **93.8** | 93.6 | 93.6 |
| RTE 3 humans + 4 tiny | 7 → 5 / 5 | 82.4 | 89.1 | 79.8 | 83.4 | **90.8** | 86.3 | 86.7 | **90.7** | 88.4 | 89.5 |
| RTE 10 humans + 6 strong LLM | 16 → 11 / 11 | 98.4 | **98.7** | 98.5 | 98.6 | 98.5 | 98.6 | 98.6 | 97.7 | 98.5 | 98.1 |
| RTE 3 humans + 6 strong LLM | 9 → 6 / 6 | 97.9 | **98.2** | 98.0 | 97.8 | 98.0 | 98.0 | 98.0 | 97.7 | 98.1 | 97.9 |
| RTE 8 small Qwen only | 8 → 3 / 4 | 89.9 | 89.2 | 90.6 | 87.3 | 86.3 | **90.9** | 90.6 | 89.2 | 90.3 | 90.4 |
| VIPER FFDM 15 readers x 93 cases | 15 → 13 / 15 | 77.3 | **78.4** | 77.6 | 78.0 | 74.7 | 77.9 | 77.6 | **78.4** | 77.5 | 78.0 |
| VIPER FFDM 8 readers x 201 cases | 8 → 8 / 8 | **81.4** | 79.8 | 80.4 | 80.2 | 79.8 | 80.4 | 80.4 | 79.8 | 80.1 | 80.2 |
| VIPER SFM 15 readers x 93 cases | 15 → 13 / 13 | 83.5 | 81.7 | 82.3 | 80.7 | 80.4 | **84.0** | **84.0** | 82.3 | 83.4 | 81.5 |
| VIPER SFM 7 readers x 203 cases | 7 → 7 / 7 | 81.8 | **83.9** | 83.3 | 83.4 | 83.0 | 83.3 | 83.3 | **83.9** | 83.0 | 83.4 |
| DICES-350 granular | 123 → 90 / 105 | **79.1** | 77.7 | 77.8 | 59.2 | 78.1 | 77.6 | 78.3 | 78.4 | 78.2 | 77.5 |
| DICES-350 race x gender groups | 10 → 10 / 10 | 78.8 | 78.9 | 79.0 | **79.1** | 78.8 | 79.0 | 79.0 | 78.9 | 79.0 | **79.1** |
| DICES-350 race x gender x age groups | 30 → 27 / 29 | 78.4 | 78.1 | 78.5 | 78.4 | 76.3 | 78.7 | 78.5 | 78.3 | **78.9** | 78.8 |
| deepfake demographic groups | 12 → 11 / 12 | 90.5 | 91.4 | 91.8 | **92.5** | 90.9 | 91.6 | 91.8 | 91.4 | 92.0 | **92.5** |
| factcheck demographic groups | 10 → 10 / 10 | 90.3 | **91.2** | 90.6 | 90.9 | 87.2 | 90.6 | 90.6 | **91.2** | 90.9 | 90.9 |
