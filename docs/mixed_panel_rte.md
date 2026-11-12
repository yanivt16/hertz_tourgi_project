# Crowd + local LLMs on RTE: second pilot (10 Nov 2026)

Tables: [results/MIXED_PANEL_RTE.md](../results/MIXED_PANEL_RTE.md) · labelling:
[scripts/label_rte_local.py](../scripts/label_rte_local.py) (shared scorer:
[scripts/local_llm.py](../scripts/local_llm.py)) · panels:
[scripts/run_mixed_panel_rte.py](../scripts/run_mixed_panel_rte.py).

**Why RTE.** On DICES the local LLMs were too weak to help anyone
([mixed_panel.md](mixed_panel.md)). RTE (800 textual-entailment pairs, 164 MTurk workers,
10 labels per pair, gold label, 50% positive) is a task where small LLMs are good. That makes it
a test of the stronger claim: does an aggregator turn LLM raters into a gain over the crowd?

**Setup.**
- LLM raters: the same three local models (Llama 3.1 8B, Qwen3 8B, Gemma 3 12B, 4-bit), each
  with two prompts: a one-line definition and a paraphrase of the RTE annotation guidelines.
  That makes 6 LLM raters.
- Score: log P(Yes) - log P(No) for "does the text entail the hypothesis?".
- No prompt was tuned on RTE, so all 800 pairs are test items.
- The crowd matrix is sparse (94% missing), so every FUSE variant here is missing-aware FUSE
  (`missing="skip"`).
- "k workers" keeps k of each pair's 10 labels at random, a different subset per pair. Results
  are the mean over 10 draws.

## Results

**The LLMs are strong and very correlated.**
- Each LLM rater reaches AUC 94-97 and accuracy 83-90%. Each one alone is roughly as good as the
  majority vote of all 10 workers (90.1%).
- The average worker label is right 73% of the time.
- Mean pairwise correlation between the LLM raters is 0.87.
- Caveat: RTE is a public benchmark (it is part of GLUE), so these models may have seen it in
  training. We cannot rule that out, so treat the LLM accuracy as an upper bound.

**Adding LLMs helps a lot when workers are few.** Accuracy, mean over 10 draws:

| k workers | Panel | MV | D&S | D&S (5 bins) | Mean z | FUSE skip, product | FUSE skip + EM |
|---:|---|---:|---:|---:|---:|---:|---:|
| 3 | workers | 80.7 | 77.9 | 77.3 | 78.0 | 54.8 | 59.8 |
| 3 | + 6 LLMs | 91.8 | 92.1 | 92.5 | **92.8** | 71.5 | 77.2 |
| 5 | workers | 84.6 | 89.0 | 89.2 | 84.2 | 58.9 | 76.2 |
| 5 | + 6 LLMs | 92.3 | 92.5 | 92.8 | **93.2** | 76.5 | 89.5 |
| 10 | workers | 90.1 | 93.0 | 92.9 | 89.9 | 92.0 | 93.0 |
| 10 | + 6 LLMs | 92.9 | 93.4 | 94.2 | **94.5** | 91.9 | 93.1 |
| 0 | LLMs only | 91.0 | 91.5 | 92.1 | 92.0 | 89.0 | |

The AUC table in the results file shows the same picture.

## What this says

1. **The "humans + LLMs" panel works in principle.** Three workers plus six LLMs (92-93%)
   match the full 10-worker crowd with Dawid-Skene (93.0%). On the full crowd, adding the LLMs
   gives +1.2 points with binned D&S (95% CI -0.5 to +3.1), so not significant.
2. **The method that delivers it is not FUSE.**
   - The simple methods do best: binned D&S and a plain average of z-scores.
   - Missing-aware FUSE breaks when the crowd is thinned out. With 3-5 labels per pair, few
     worker pairs share the 20 items its moment estimates need. With the product posterior it
     lands at 55-77%; EM refinement recovers part of the gap, but not all of it.
     This is a real limitation of the missing-data fix, not visible in the earlier runs, which
     always used all 10 labels.
   - On the dense LLM-only block, the paper's FUSE (logistic) reaches 90.6% (AUC 97.2), against
     92.1% for binned D&S. It is close, but not better.
3. **Correlation was not the problem here.** The LLMs are correlated at 0.87, yet D&S does
   well. When correlated raters are also *accurate*, D&S's independence assumption costs
   little. The DICES failure needed correlated *and* weak raters.

## Combined with the DICES pilot

| Setting | LLM raters vs one human | Do LLMs help? | Best method on the mixed panel |
|---|---|---|---|
| DICES-350 (safety, subjective) | weaker (AUC 52-70) | no | FUSE logistic: loses least |
| RTE (entailment, objective) | stronger (83-90% vs 73%) | yes, a lot when workers are few | binned D&S / mean z |

The honest reading is that FUSE's advantage shows up only when weak raters are correlated. In
that setting the LLMs aren't worth adding anyway. When they are worth adding, simple methods
already handle them.

Open questions for Regev:
- Is there a setting between these two, with LLMs about as good as one human and correlated?
  That is where FUSE should matter. A harder task, or weaker or noisier prompts, could produce it.
- Is the contamination caveat on RTE acceptable, or do we need a task the models cannot have
  seen?
