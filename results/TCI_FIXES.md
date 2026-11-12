# Fixing FUSE's threshold search (generated from results/tci_fix_sim.json and results/tci_fix_real.json)

AUC. Bold: best in the row (within 0.05). 'Clustered': raters whose average |Spearman| exceeds 0.5 are averaged into one column first (`cluster_threshold=0.5`). 'FUSE median': median thresholds, no search (`n_sweeps=0`). 'FUSE 30–70%': search restricted to the 30–70% quantiles (`quantile_range`). 'Scale-free stat': `tci_standardize=True`. The 20–80% range and scale-free + 20–80% are in the JSON files.

## Simulation (scripts/tci_fix_sim.py), mean over 6 seeds

Base: 1000 items, 10 continuous raters, separation d on [0.5, 1.5], 30% positive. Weak block: d = 0.2, shared noise ρ = 0.6.

| Scenario | D&S 5 bins | D&S | Mean z | D&S → logistic | Clustered D&S → logistic | Clustered mean z | FUSE paper | FUSE median | FUSE 30–70% | FUSE scale-free stat | FUSE clustered | FUSE clustered, median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base (10 raters, d 0.5-1.5) | 98.0 | 96.1 | 98.2 | **98.4** | **98.4** | 98.2 | 98.0 | **98.5** | **98.4** | 98.2 | 98.3 | **98.5** |
| 5 raters | 92.4 | 88.4 | 93.4 | 93.5 | 93.5 | 93.4 | 91.8 | 93.5 | **94.0** | 93.5 | 93.5 | 93.5 |
| hard: d=0.3 | 57.4 | 63.3 | **74.5** | 64.3 | 64.3 | **74.5** | 62.3 | 66.0 | 69.3 | 67.9 | 63.7 | 66.0 |
| high quality spread | 98.1 | 96.1 | 96.9 | **98.5** | **98.5** | 96.9 | 97.0 | 98.4 | **98.5** | 98.0 | 98.4 | 98.4 |
| 20 near-chance + 2 experts | 89.9 | 86.4 | 75.2 | **91.8** | **91.8** | 75.2 | 72.2 | 89.8 | 85.9 | 80.3 | 61.8 | 89.8 |
| 50 near-chance + 2 experts | 78.3 | 85.9 | 76.9 | **89.4** | **89.4** | 76.9 | 68.1 | 86.5 | 85.9 | 84.5 | 63.0 | 86.5 |
| 20% weak corr block | 96.5 | 94.1 | 95.0 | 97.2 | **97.3** | 96.4 | 96.9 | 92.1 | 96.8 | 87.1 | 96.7 | 97.2 |
| 40% weak corr block | 59.8 | 63.1 | 86.2 | 59.9 | **95.6** | 94.3 | 76.3 | 60.7 | 62.9 | 63.2 | 94.9 | 95.1 |
| 40% normal-quality corr block | 85.0 | 84.9 | 94.7 | 85.0 | **97.0** | 96.5 | 91.0 | 86.0 | 86.8 | 87.0 | 96.8 | 96.9 |
| binary votes |  | **95.8** | 95.0 | **95.9** | 95.8 | 95.0 | 95.8 | 95.8 | 95.8 | 95.8 | 95.8 | 95.7 |

## Real panels (scripts/tci_fix_real.py)

RTE panels: mean over 10 shuffles of each pair's 10 labels into human columns. Tiny: Qwen2.5-0.5B and Qwen3-0.6B; small Qwen: those plus the 1.5B and 1.7B; strong: Llama-3.1-8B, Qwen3-8B, Gemma-3-12B. Two prompts per model.

| Panel | D&S 5 bins | D&S | Mean z | D&S → logistic | Clustered D&S → logistic | Clustered mean z | FUSE paper | FUSE median | FUSE 30–70% | FUSE scale-free stat | FUSE clustered | FUSE clustered, median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RTE 10 humans |  | 96.5 | **96.6** | 96.5 | 96.5 | **96.6** | 96.5 | 96.5 | 96.5 | 96.5 | 96.5 | 96.5 |
| RTE 10 humans + 4 tiny | 91.1 | 94.0 | 95.9 | 95.0 | **96.8** | **96.8** | 96.7 | 95.5 | 96.0 | 95.9 | 96.8 | **96.8** |
| RTE 10 humans + 8 small Qwen | 93.8 | 94.3 | 96.3 | 94.8 | **97.4** | 97.2 | 96.4 | 95.6 | 95.6 | 95.6 | 97.4 | **97.4** |
| RTE 5 humans + 4 tiny | 84.9 | 84.8 | 92.3 | 85.0 | 93.7 | **93.7** | 92.6 | 88.5 | 88.4 | 88.5 | 93.6 | 93.6 |
| RTE 3 humans + 4 tiny | 82.4 | 81.3 | 89.1 | 79.8 | 90.8 | **90.9** | 83.4 | 84.3 | 84.1 | 83.8 | 90.4 | 90.6 |
| RTE 10 humans + 6 strong LLM | 98.4 | 98.2 | **98.7** | 98.5 | 98.5 | 97.7 | 98.6 | 98.5 | 98.5 | 98.5 | 98.3 | 98.5 |
| RTE 3 humans + 6 strong LLM | 97.9 | 97.3 | **98.2** | 98.0 | 98.0 | 94.9 | 97.8 | 98.0 | 98.0 | 98.0 | 96.4 | 98.0 |
| RTE 8 small Qwen only | 89.9 | 87.9 | 89.2 | **90.6** | 86.3 | 87.2 | 87.3 | 90.4 | 90.5 | 90.2 | 86.3 | 86.3 |
| VIPER FFDM 15 readers x 93 cases | 77.3 | **79.2** | 78.4 | 77.6 | 74.7 | 75.7 | 78.0 | 77.5 | 78.1 | 78.1 | 77.7 | 76.0 |
| VIPER FFDM 8 readers x 201 cases | **81.4** | 80.4 | 79.8 | 80.4 | 79.8 | 78.1 | 80.2 | 80.1 | 80.0 | 80.1 | 79.8 | 79.8 |
| VIPER SFM 15 readers x 93 cases | **83.5** | 83.4 | 81.7 | 82.3 | 80.4 | 79.1 | 80.7 | 82.7 | 80.8 | 80.8 | 79.7 | 81.4 |
| VIPER SFM 7 readers x 203 cases | 81.8 | 79.0 | **83.9** | 83.3 | 83.0 | 83.9 | 83.4 | 83.0 | 83.4 | 83.3 | 81.9 | 83.1 |
| DICES-350 granular | **79.1** | 77.8 | 77.7 | 77.8 | 78.1 | 77.2 | 59.2 | 77.9 | 78.1 | 77.2 | **79.0** | **79.0** |
| DICES-350 race x gender groups | 78.8 | 77.5 | 78.9 | 79.0 | 78.8 | 77.3 | **79.1** | 79.0 | **79.0** | **79.0** | 78.9 | 78.9 |
| DICES-350 race x gender x age groups | 78.4 | 76.9 | 78.1 | 78.5 | 76.3 | 74.4 | 78.4 | **78.9** | 78.8 | 78.3 | 76.0 | 77.8 |
| deepfake demographic groups | 90.5 | 91.7 | 91.4 | 91.8 | 90.9 | 86.6 | **92.5** | 92.0 | 92.2 | 92.2 | 91.6 | 91.6 |
| factcheck demographic groups | 90.3 | 90.2 | **91.2** | 90.6 | 87.2 | 85.1 | 90.9 | 90.9 | 91.0 | 91.0 | 79.1 | 79.1 |
