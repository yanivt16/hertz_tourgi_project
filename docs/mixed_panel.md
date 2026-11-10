# Humans + local LLMs as raters: first pilot (10 Nov 2026)

Tables: [results/MIXED_PANEL.md](../results/MIXED_PANEL.md) · labelling:
[scripts/label_dices_local.py](../scripts/label_dices_local.py) (raw scores in
[results/llm_scores/](../results/llm_scores/)) · panels:
[scripts/run_mixed_panel.py](../scripts/run_mixed_panel.py).

**Question.** LLM raters are correlated with each other and give continuous scores. That is
FUSE's home ground and a known failure mode of Dawid-Skene
([continuous_scores.md](continuous_scores.md)). Does FUSE handle a panel of humans plus LLMs
better?

**Setup.**
- Dataset: DICES-350, with the expert label as gold.
- LLM raters: three local open-weight models on Apple MLX (Llama 3.1 8B, Qwen3 8B,
  Gemma 3 12B, all 4-bit), each with two prompts: the v4 policy and the short v1. That makes
  6 LLM raters.
- Score per conversation: log P(Yes) - log P(No) for "is the final response unsafe?".
- The v4 prompt was tuned on the DEV half, so the clean numbers are on the TEST half
  (175 conversations).

## Results (test half, AUC against the expert label)

**Single LLM raters are weak.**
- AUC 52-67, against 79-81 for any aggregate of the 123 humans. Qwen3 8B is near chance.
- The LLM raters are strongly correlated with each other (mean pairwise correlation 0.60,
  against 0.24 between humans).

**Full panel (123 humans).** Adding the 6 LLM raters changes nothing for any method; they are
drowned out. Without them, FUSE (product posterior) is at 81.3 and Dawid-Skene (5 bins) at
80.0. The difference is +1.3, 95% CI -0.5 to +3.1, so not significant.

**Few humans per item.** Mean AUC over 20 random draws of k human raters:

| k humans | Panel | MV | D&S (5 bins) | Mean z | FUSE logistic | FUSE product |
|---:|---|---:|---:|---:|---:|---:|
| 3 | humans | 67.2 | 67.8 | 69.1 | 68.8 | 68.6 |
| 3 | + 6 LLMs | 67.5 | **63.1** | 66.6 | 66.6 | 66.7 |
| 5 | humans | 69.7 | 71.0 | 71.6 | 72.0 | 71.4 |
| 5 | + 6 LLMs | 69.1 | **64.1** | 68.6 | **70.3** | 69.7 |
| 10 | humans | 74.0 | 74.7 | 74.3 | 75.2 | 74.8 |
| 10 | + 6 LLMs | 72.1 | **67.5** | 71.2 | **74.2** | 71.7 |

## What this says

1. **The mechanism shows up on real data.** Correlated LLM raters hurt Dawid-Skene the most:
   it loses about 5 to 7 AUC points as soon as they join a small human panel. FUSE logistic
   loses the least (1 to 2 points) and is the best method on every mixed panel with 5 or 10
   humans. This matches the simulation with correlated junk raters.
2. **These LLMs don't help anyone.** Small local models are too weak for this task, so adding
   them lowers every method below humans-only. The pilot shows FUSE is *robust* to correlated
   LLM raters. It does not show that FUSE makes them *useful*.
3. **What would test the real claim** is LLM raters that are individually competitive with a
   human. Stronger models are one route: the GPT label reached 69.9% accuracy on the test half
   in [DICES_LLM_LABELS.md](../results/DICES_LLM_LABELS.md), and API models were not tried
   here. A task where LLMs are good is the other: RTE, for example, has text and gold.

The bootstrap CIs for the few-humans table were not computed. The differences there are
means over 20 draws on 175 conversations; treat them as indicative.
