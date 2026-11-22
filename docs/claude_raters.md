# Claude raters on DICES-350, and filling holes with LLMs (22 Nov 2026)

Tables: [results/CLAUDE_RATERS.md](../results/CLAUDE_RATERS.md) · script:
[scripts/run_claude_raters.py](../scripts/run_claude_raters.py) · how the scores were made:
[scripts/make_claude_rater_batches.py](../scripts/make_claude_rater_batches.py) · scores:
[results/claude_scores/](../results/claude_scores/).

**Question.** Our local LLM verifiers on DICES were weak (test-half AUC 52–67), so the mixed panels
never had a strong LLM. Do two stronger LLM raters change the picture? And can LLMs "fill holes":
each item rated by only a few humans, with LLMs rating every item?

**Setup.** Claude Sonnet and Claude Haiku, run as agents, read the same v4 safety policy and gave each
of the 350 conversations a 0–100 probability that experts would call the final response unsafe (four
batches per model, one agent per batch). Gold = expert label. All AUCs are on the test half (175
conversations), since the v4 policy was tuned on the dev half. Few-human results are means over 20
random draws of humans.

## Findings

1. **The Claude raters are the best single raters we have, but far below the crowd.** Sonnet 70.4,
   Haiku 72.1, against 52–67 for the local LLMs, a median single human of 62.2 (best 73.8), and about
   80 for all 123 humans aggregated.
2. **They form a natural block, and the detector finds it without labels.** Sonnet and Haiku
   correlate 0.84. At threshold 0.4 the residual detector merges Sonnet + Haiku and the two Qwen
   prompts, and nothing else.
3. **Dense panels: FUSE wins when weak LLMs are mixed in.** With the same 10 humans on every item plus
   all 8 LLMs, FUSE logistic gets 75.0 against 71.1 for binned Dawid–Skene and 72.6 for mean z
   (ahead of D&S in 20/20 draws, of mean z in 19/20). With 5 humans, 72.0 vs 68.7 and 70.6. The weak,
   correlated local LLMs cost D&S 5.5 points and mean z 3.7, against 1.3 for FUSE.
4. **The gap closes once the blocks are merged.** Merging each detected block into one verifier
   before aggregating gives D&S 74.6 and FUSE 75.5 (10 humans). Same pattern as on RTE: FUSE reaches,
   with no detector and no tuning, what D&S reaches with the detector.
5. **With only good LLMs there is no FUSE advantage.** Humans + Claude: D&S 76.6, FUSE 76.3 (10
   humans). With all 123 humans, FUSE product posterior is 81.3 against 79.6 for D&S (+1.7, CI
   −0.3 to +3.8), with or without Claude.
6. **Filling holes works, and mean z is the best way to do it.** When each item gets 3 humans and
   Claude rates everything, mean z reaches 74–75 against about 68 for the 3 humans alone, in every
   setting tried: a large pool (3 of 123 humans per item) and small fixed teams (6, 10, 15 or 20
   humans, 3 or 5 per item).
7. **FUSE does not help with holes.** Missing-aware FUSE estimates rater quality from items that
   raters rated together. In the large pool two humans share about 0.1 items, so FUSE on humans alone
   is at chance (51). In small teams it is usable, but with Claude it stays at 68.9–71.5, below D&S
   (72.5–73.9) and mean z (74.3–75.4). Only when the weak local LLMs are in the panel does it match or
   beat mean z, by up to a point in 3 of the 5 team settings, and that panel is itself 4–5 points worse
   than humans + Claude.

## What it means

FUSE's advantage on real data is robustness: when one cannot tell in advance which LLM raters are
good and adds all of them, FUSE loses least. When the LLMs are chosen well, or when ratings are
sparse, simple averaging of z-scores is as good or better. For "filling holes" with LLMs, the useful
result is that a strong LLM adds about 6–7 AUC points to 3 humans per item; FUSE is not the tool that
gets it.

## Caveats

- One dataset. Claude may have seen DICES (public since 2023) in training.
- The agents give verbal probabilities, not log-probabilities, and a rerun would differ somewhat.
- No proper confidence intervals for the few-human results: all draws share the same 175 items, so
  we report how often one method beats another instead.
- Why missing-aware FUSE does badly with Claude is not checked. Two guesses: the logistic predictor
  is not available with missing ratings, and the product posterior counts Sonnet and Haiku as two
  independent votes although they share errors.
