<!-- Mirror of the Google Doc "FUSE on Human Annotators: Progress Update".
     The Google Doc is the canonical version: https://docs.google.com/document/d/1Zsr_Y5Iax8SfreIJF5TE5SUnH0ADE3DeWreIkypeWz0/edit
     Regenerate with scripts/gdoc_to_md.py (see CLAUDE.md). Last synced: 2026-11-21.
     AHEAD OF THE GOOGLE DOC: the 21 Nov paper-benchmark result (new section "The paper's own
     benchmarks", plus one sentence each in Summary and "Where this leaves FUSE", one Next steps
     bullet, one question) is in this mirror only. So is the 22 Nov DICES-350 result (new section
     "DICES-350: tuning FUSE and small rater panels", plus one sentence each in Summary and
     "Where this leaves FUSE", and one Next steps bullet). Copy both into the Google Doc, then regenerate. -->

# FUSE on Human Annotators: Progress Update

Yaniv & Ohad · MLDS M.Sc. project · Mentor: Dr. Regev Cohen · 11 November 2026

Code and all result files: [github.com/yanivt16/hertz_tourgi_project](https://github.com/yanivt16/hertz_tourgi_project). The project proposal (Reichman format) is shared alongside this document.

## Summary

On human annotators, the baseline FUSE has to beat is Dawid–Skene, not majority vote. With many interchangeable raters FUSE reduces to majority vote, and on sparse data, zero-imputation of missing ratings (as in the paper) costs it about 6 points against Dawid–Skene. When the matrix is made dense, by running FUSE on blocks of conversations that share raters, FUSE beats majority vote by 5.1 points and ties Dawid–Skene. We propose to focus the project on binary tasks and to make missing-data handling the main methodological work of Stage 1.

Since the first draft we implemented missing-aware FUSE and tested continuous scores, a radiology reader study (VIPER), panels that mix humans with local LLM raters, a simulation map of where FUSE should win, two real crowds of non-experts on hard questions, and a correlated block of small LLMs next to humans. Missing-aware FUSE closes the gap to Dawid–Skene on sparse data. FUSE beats Dawid–Skene reliably in one setting only, a weak correlated block added to a human panel, and there it avoids damage rather than adding accuracy. The new sections start at "Missing-aware FUSE" below. A new test on the paper's own Best-of-N benchmarks shows where FUSE's lead over Dawid–Skene comes from: its threshold search, not its estimator. On DICES-350, tuning FUSE and running it inside small or demographic rater panels does not change the picture: with the same information and the same cut-off, FUSE and Dawid–Skene make the same calls.

## Paper reproduction

We reimplemented FUSE from scratch (the authors have not released code) and ran it on the paper's released Best-of-N data. Pass@1, Pass@k and the oracle best verifier match the paper, which validates the data and the evaluation. On the 70B settings (GPQA, MATH500, MMLU, MMLU-Pro) we are within 1.5 points of the paper's FUSE numbers; GPQA-Diamond-70B matches exactly (64.4) with the posterior as the final predictor. The paper does not say which mode (per-question or batched) or final predictor it used per table, so no single configuration of ours wins everywhere.

The one setting that does not reproduce is MATH500-8B (our best 66.0 vs. 83.9 in the paper). Nine of its 16 reward models are near-random but strongly correlated with each other (|corr| ≈ 0.5), while the two best verifiers are weakly correlated with everything. The moment estimator's rank-one fit locks onto the correlated block and reads it as the correctness signal. A label-free extension we added, clustering correlated verifiers into one meta-verifier, brings MATH500-8B to 79.3, but it costs 2 to 10 points on GPQA-70B, where the correlated block is correlated because it is right. It is therefore not a better default.

## First experiment: DICES

DICES (Google, Aroyo et al. 2023) has safety ratings of adversarial chatbot conversations, on a Yes / Unsure / No scale. Conversations are items, raters are verifiers, and the expert label is hidden from every method.

|  | **DICES-350** | **DICES-990** |
|---|---|---|
| Conversations | 350 | 990 |
| Raters | 123 | 172 |
| Missing ratings | 0% | 58% |
| Expert label | Yes (175 unsafe / 175 safe) | No |

## Findings

- **Per rater, FUSE is identical to majority vote.** On DICES-350 both reach 65.1% against the experts (difference +0.0, 95% CI −0.9 to +0.9). With 123 similar raters there is little to weight. With 10 race × gender groups as verifiers, FUSE beats majority vote by 5.7 points (95% CI +0.6 to +10.9), but with 30 finer groups the gain disappears (−1.1), so we treat it as one favourable configuration.
- **The crowd sets a ceiling.** Even with a threshold tuned on the expert labels, the share of raters saying "unsafe" reaches only 72.0%. Raters call 23–33% of conversations unsafe, the experts 50%. 31 expert-unsafe conversations are missed by all nine methods, including those that use labels, mostly gender bias and personal information. No demographic subgroup of raters flags them.
- **An LLM label for DICES-990.** DICES-990 has no expert label, so we built one with an LLM and a written safety policy, developed on half of DICES-350 and tested once on the other half. It agrees with the experts on 69.9% of the 176 held-out conversations, against 65.3% for majority vote (+4.5, 95% CI −2.8 to +11.9, so not significant). The LLM and the crowd are right on different conversations.
- **On sparse data, FUSE equals majority vote and Dawid–Skene is 6 points ahead.** On DICES-990 as a whole, with missing ratings filled with zeros as in the paper, FUSE reaches 71.0% agreement with the LLM label and majority vote 71.4%, while Dawid–Skene reaches 77.5% because it skips missing ratings instead of imputing them.
- **Main result: on dense blocks, FUSE beats majority vote and ties Dawid–Skene.** DICES-990 was collected in batches, so we split it into 5 blocks of 159–330 conversations, each rated by the same 41–56 core raters (under 1% missing). Pooled over all 990, against the LLM label, FUSE (posterior) beats majority vote by 5.1 points (95% CI +3.1 to +7.1) and is level with Dawid–Skene (−0.7, 95% CI −3.0 to +1.5). FUSE wins in all 5 blocks. The result holds with 3 and 4 blocks, and the same blocks with all their raters (21–37% missing) give only +0.5 to +1.8.

| **DICES-990, agreement with the LLM label (%)** | **Whole sparse matrix** | **Dense blocks** |
|---|---|---|
| Majority vote | 71.4 | 70.5 |
| Dawid–Skene | 77.5 | 76.3 |
| FUSE (logistic) | 71.0 | 74.4 |
| **FUSE (posterior)** | **70.9** | **75.6** |

## What we learned

On human annotators the method to beat is Dawid–Skene. In the paper and in our reproduction FUSE beats it in all 12 Best-of-N settings (taking the best FUSE configuration per setting), but there the verifiers are few, give continuous scores and have no missing data. Sparse crowd votes are the setting Dawid–Skene was built for.

On DICES-990, missing data explains most of FUSE's gap: once the matrix is dense, FUSE ties Dawid–Skene. Density alone does not close the gap everywhere, though. On DICES-350, which is fully dense, Dawid–Skene is still 3.4 points ahead per rater (68.6 vs. 65.1), mostly because it calls more conversations unsafe (33% vs. 23%).

## Missing-aware FUSE

We implemented the pairwise-complete version of FUSE proposed in the first draft. Moments are computed only over items both raters labelled, thresholds are searched over observed ratings only, and each item's posterior uses only the raters who labelled it. A missing rating no longer counts as a vote.

On six datasets with a reference label, this closes the gap to Dawid–Skene. FUSE now ties it, but does not beat it (accuracy, %):

| **Dataset** | **Majority vote** | **Dawid–Skene** | **FUSE, missing = 0 (paper)** | **FUSE, missing-aware** |
|---|---|---|---|---|
| RTE | 90.1 | 93.0 | 69.1 | 92.0 |
| D_PosSent | 93.3 | 95.9 | 84.0 | 95.6 |
| D_Product | 89.7 | 93.8 | 87.8 | 92.6 |
| Bluebird | 75.9 | 88.9 | 86.1 | 88.9 |
| DICES-350 (expert label) | 65.4 | 68.6 | 65.1 | 68.0 |
| DICES-990 (LLM label) | 71.8 | 77.5 | 70.9 | 77.7 |

Majority vote here breaks ties at random, so its DICES numbers differ slightly from the tables above. In simulation, zero-imputation drops to 73% accuracy once half the ratings are missing, while missing-aware FUSE stays level with Dawid–Skene down to about 7 ratings per item. The method has two limits. It assumes that which rater labels an item is unrelated to the item's label, and it needs at least 20 items in common for each pair of raters. When RTE is thinned to 3–5 labels per item, the second condition fails and FUSE drops to 55–59%, against 78–89% for Dawid–Skene.

## Continuous scores

FUSE was designed for continuous verifier scores, so we tested whether scores are where it beats Dawid–Skene. The fair baseline is Dawid–Skene on each rater's scores cut into 5 quantile bins. In simulation, binned Dawid–Skene is at the oracle when raters are conditionally independent. FUSE ties it with balanced classes and trails it with 15% positives (AUC 96.7 vs. 99.5). When three correlated low-quality raters are added, Dawid–Skene falls to chance (AUC 48.7) while FUSE stays at 93.4, but a plain average of z-scored ratings does as well (94.1). On real scores (VIPER suspicion scores, DICES demographic-group shares) no method is reliably better than another.

## Radiology: VIPER

VIPER (FDA DIDSR) is a mammography reader study: radiologists read screening cases, and cancer status at follow-up is the gold label. The images are not public. In the full case × reader matrix, which radiologists read a case predicts cancer with AUC 94.2 from the assignment alone, so any method that uses the missingness pattern looks better than it is there. On dense blocks where the same readers read every case (93–201 cases, 7–15 readers), FUSE ties majority vote and Dawid–Skene, on both the recall decision and the suspicion score.

## Humans plus LLM raters

We added small open-weight LLMs, run locally, as extra raters next to the humans: Llama 3.1 8B, Qwen3 8B and Gemma 3 12B, each with two prompts, which gives 6 LLM raters. Each scores an item as log P(Yes) − log P(No). LLM raters are correlated with each other, a known weakness of Dawid–Skene, so this is where we expected FUSE to help. Two pilots gave opposite outcomes:

|  | **DICES-350 (safety)** | **RTE (entailment)** |
|---|---|---|
| One LLM rater | AUC 52–67; the crowd reaches about 80 | 83–90% accuracy; one worker label 73% |
| Correlation between LLM raters | 0.60 (between humans: 0.24) | 0.87 |
| Dawid–Skene, 5 humans → plus LLMs | 71.0 → 64.1 (AUC, binned) | 89.0 → 92.5 (%) |
| FUSE, 5 humans → plus LLMs | 72.0 → 70.3 (AUC, logistic) | 58.9 → 76.5 (%, missing-aware) |
| Best method on the mixed panel | FUSE logistic | Binned Dawid–Skene, mean z-score |

On DICES the LLMs are too weak to help any method, and FUSE loses the least when they are added. On RTE they help a lot: 3 workers plus the LLMs reach 92–93%, the same as all 10 workers with Dawid–Skene (93.0%). There the simple methods use them best, and missing-aware FUSE fails because the thinned crowd leaves too few shared items per rater pair. On the 6 LLMs alone, a dense matrix, FUSE reaches 90.6% against 92.1% for binned Dawid–Skene. RTE is a public benchmark (part of GLUE), so the models may have seen it in training.

## Where FUSE can win: a simulation map

To find where FUSE should win, we simulated raters of known quality and changed one thing at a time: panel size, difficulty, spread of rater quality, prevalence, scores versus yes/no votes, missing ratings, and a block of raters who share errors. We also tested the hypothesis that FUSE helps when many non-experts face a hard question alongside a few experts. Each cell is the best FUSE variant minus the best Dawid–Skene variant, in AUC points, averaged over 4–6 runs.

| **Setting** | **FUSE minus Dawid–Skene (AUC)** | **Note** |
|---|---|---|
| Independent raters (any panel size, difficulty, prevalence, scores or votes) | −1.1 to +0.2 | Ties |
| Missing ratings (50% or 20% observed) | −7.6, −6.6 | Too few co-rated items |
| Many near-chance raters plus 2 experts, continuous scores | −1.7 to −25.5 | Threshold search fails; a median split recovers it |
| Same, yes/no votes | −0.2 to +1.9 | Ties |
| 20% of raters weak and sharing errors | +4.0 | FUSE logistic best (96.6) |
| 40% of raters weak and sharing errors | +15.0 | Mean z-score better still (87.7 vs 82.3) |

With independent raters FUSE ties Dawid–Skene everywhere. It loses with missing ratings, and loses clearly when many near-chance raters give continuous scores. The cause is its label-free threshold search: a plain median split recovers the loss. FUSE wins only when a block of weak raters shares errors, and once that block is 40% of the panel a plain average of z-scores is as good or better. The many-non-experts hypothesis is not supported.

## Real crowds of non-experts on hard questions

We then tested the same hypothesis on two public crowds whose gold labels do not come from the crowd: deepfake videos (Groh et al. 2022; 54 videos, 304 people, 0–100 confidence that a video is fake) and news fact-checking (Allen et al. 2021; 189 articles, 1,128 laypeople, gold from 3 professional fact-checkers). One rating alone has AUC 65–67. Two people share too few items for FUSE to estimate anything (a median of 7 and 1), so we ran it on demographic groups, with each group's mean score as one rater, and compared them with random groups of the same sizes.

| **AUC** | **Deepfakes** | **Fact-checking** |
|---|---|---|
| Crowd mean of individuals | 90.0 | 90.1 |
| Binned Dawid–Skene, groups | 90.5 | 90.3 |
| Mean z-score, groups | 91.4 | 91.2 |
| FUSE logistic, groups | 92.5 | 90.9 |
| FUSE product posterior, groups | 89.9 | 89.9 |

Every method lands within noise of the crowd mean. FUSE logistic is 1.9 AUC points above binned Dawid–Skene on deepfakes (95% CI 0.0 to +5.1, with only 54 videos) and 0.6 above on fact-checking (CI −0.5 to +1.9). Political groups do share a bias: within each class, groups on the same side correlate 0.45 and groups on opposite sides 0.23. But the two sides are of similar size and accuracy, so there is no bad block to down-weight, and FUSE gains no more with political groups than with random ones (+0.6 against 0.0).

## A correlated block on real labels

Before this test we checked whether a bug was behind FUSE's weak results. The code matches the paper's equations and reproduces its results within 2 points in 10 of 12 settings. With the same thresholds, FUSE's moment estimates and posterior give the same AUC as Dawid–Skene. So FUSE's wins and its losses both come from the paper's threshold search: it pushes near-chance raters to extreme splits, and it is also what lets FUSE discount a correlated weak block. The same check showed that FUSE's final logistic step works just as well on Dawid–Skene's posteriors, so we added that as a baseline.

We then built the regime the simulation asked for from real labels. RTE has 10 crowd labels per entailment pair; we shuffled them into 10 dense human columns (within-class correlation 0.04). Next to them we put small Qwen models (0.5B to 1.7B, two prompts each). The two smallest are about as good as one human (AUC 74–77 against 73) and correlate 0.66 with each other. AUC, mean over 20 shuffles, with 95% intervals:

| **Panel** | **Binned Dawid–Skene** | **Mean z-score** | **FUSE logistic** | **FUSE minus binned D&S** |
|---|---|---|---|---|
| 10 humans | 96.5 | 96.5 | 96.5 | 0.0 |
| 10 humans + 4 tiny LLMs (29%) | 91.1 | 95.9 | 96.7 | +5.6 [+4.1, +7.1] |
| 10 humans + 8 small LLMs (44%) | 93.8 | 96.3 | 96.4 | +2.7 [+1.7, +3.9] |
| 5 humans | 92.1 | 92.2 | 92.1 | 0.0 |
| 5 humans + 4 tiny LLMs (44%) | 84.8 | 92.3 | 92.1 | +7.3 [+3.4, +10.7] |
| 3 humans + 4 tiny LLMs (57%) | 82.3 | 89.1 | 83.6 | +1.3 [−2.0, +9.0] |

This is the first real panel where FUSE beats Dawid–Skene with an interval that excludes zero, and it also beats Dawid–Skene followed by the same logistic step. The mechanism is the one the simulation predicted: Dawid–Skene reads the block's shared errors as signal. But the gain is robustness, not added value. With 5 or 10 humans, no method does better with the block than without it, and the plain average of z-scores is within a point of FUSE. With 3 humans the block does add information, and the z-score average uses it best (89.1, against 87.5 for the humans alone) while FUSE falls behind. So far this is one task and one model family.

## The paper's own benchmarks: what carries FUSE's lead

The simulation and the RTE test point to the threshold search as the part of FUSE that matters. We checked this on the paper's own 12 Best-of-N settings, where FUSE beats Dawid–Skene. Two label-free changes isolate the search: FUSE with every verifier split at its median (no search), and Dawid–Skene run on the yes/no scores that FUSE's search produces, followed by the same logistic step. We also tested whether blocks of verifiers that share errors explain the lead: a label-free detector merges each block into one column before Dawid–Skene runs. Best-of-N accuracy, mean over the 12 settings (best of per-question and batched mode):

| **Method** | **Mean accuracy (%)** |
|---|---|
| FUSE (our reproduction) | 68.0 |
| Dawid–Skene on FUSE's thresholds, then logistic | 67.7 |
| Mean z-score | 67.6 |
| FUSE, median thresholds (no search) | 64.0 |
| Dawid–Skene, then logistic | 63.1 |
| Blocks merged, then Dawid–Skene, then logistic | 62.9 |

Dawid–Skene given FUSE's thresholds matches FUSE, within one point in 7 of 12 settings in batched mode and 8 of 12 per question. Without the search, FUSE falls to Dawid–Skene's level. Merging blocks does not help: the detector finds sensible groups (variants of one reward model, reward models from one family, groups of LLM-judge verdicts), but Dawid–Skene stays 3–5 points behind FUSE after merging them. So on LLM verifiers FUSE's lead is its label-free threshold search, and its moment estimator and posterior add nothing over Dawid–Skene. This also explains the human results: yes/no votes leave nothing to search, so FUSE ties Dawid–Skene there. A plain average of z-scores comes close (67.6), but mostly on the 8B settings where our FUSE falls short of the paper (the paper reports 70.8 on average; see Paper reproduction). One run per setting, no confidence intervals. Results: results/PAPER_BLOCKS.md; write-up: docs/paper_blocks.md.

While doing this we found a bug in our rater-clustering code: its Spearman correlation breaks ties by row order, so two unrelated yes/no columns correlate at about 0.2. On these benchmarks it only changes which LLM-judge verdicts are merged (accuracy moves by at most 0.3 points), but results that cluster yes/no votes need a rerun once it is fixed.

## DICES-350: tuning FUSE and small rater panels

We tuned FUSE on DICES-350 one setting at a time (how raters become verifiers, parts of the algorithm, numeric settings), choosing on a fixed dev half (174 conversations) and scoring once on the test half (176). Two changes were adopted: demographic groups (race × education) as verifiers, and a cut-off at FUSE's own estimated prevalence. Dev accuracy rose from 64.9% to 75.9%, but the test half gives 69.9%, against 65.3% for majority vote (+4.5, 95% CI −3.4 to +12.5) and 71.0% for Dawid–Skene on the individual raters. Grouping raters is the only large lever, and random groups do about as well as demographic ones.

We then tested whether small panels, where good raters weigh more, let FUSE find them better than Dawid–Skene: 100 reshuffles of the 123 raters into disjoint panels of about 5, 10 and 20, and the raters of each demographic group (gender, race, education, age: 13 groups of 7–75 raters), each against 100 random panels of the same size. Every rater stays a separate verifier. A first version showed a small FUSE lead; an independent review found it came from two unfair comparisons, now fixed. Each method had been cut at its own estimated prevalence (both under-call, so whichever calls more wins), and Dawid–Skene had been given only Yes votes while FUSE also saw Unsure. With a three-answer Dawid–Skene (Yes / Unsure / No) and every method calling the same number of conversations unsafe (accuracy points, 95% intervals over conversations):

| **Panels** | **FUSE minus Dawid–Skene, same cut-off** | **FUSE minus Dawid–Skene, each at its own rule** |
|---|---|---|
| About 5 raters (2,400 panels) | +0.0 [−0.1, +0.1] | −1.3 [−1.9, −0.8] |
| About 10 raters (1,200 panels) | +0.1 [−0.1, +0.2] | −2.1 [−3.3, −1.0] |
| About 20 raters (600 panels) | +0.1 [−0.0, +0.3] | −2.8 [−4.5, −1.1] |
| 13 demographic groups | −0.1 on average; no interval excludes 0 | −2.9 on average; Dawid–Skene ahead in all 13 |

The hypothesis is not supported. At a shared cut-off FUSE, Dawid–Skene and even majority vote make almost the same calls. FUSE's earlier AUC edge was the Unsure information Dawid–Skene had been denied (against three-answer Dawid–Skene it is 0). Both methods rank raters by quality equally well (Spearman 0.70–0.85 for each), and demographic panels behave like random panels of the same size (−0.2 points on average in the FUSE − Dawid–Skene gap). The same correction removes FUSE's lead on random groups of raters (1,000 groupings per size, −0.3 to +0.5 points at a shared cut-off), and Dawid–Skene on the individual raters beats tuned FUSE on the test half in 80–92% of groupings. On DICES the differences between methods come from how many conversations each calls unsafe, not from how they weigh raters. Results: results/DICES_TUNING.md, DICES_SMALL_PANELS.md, DICES_RANDOM_GROUPS.md, DICES_DEMOGRAPHIC_PANELS.md.

## Where this leaves FUSE

In almost every setting we have tried (binary, sparse, continuous, and mixed panels of humans and LLMs), FUSE at best ties Dawid–Skene. Its one advantage is robustness to raters that are both correlated and weak, now confirmed on real labels, and there a plain average of z-scores does about as well. The simulation map and the two real crowds above narrow this further: FUSE is ahead only when a minority block of weak raters shares errors, and on real crowds of non-experts every method lands within noise of the crowd mean. On RTE with a block of small LLMs, FUSE wins by not being hurt by the block, not by getting more out of it. The honest midterm result may be a negative one, with missing-aware FUSE as the methodological contribution. On the paper's own Best-of-N benchmarks the picture is the same from the other side: FUSE's lead over Dawid–Skene there comes from its threshold search, and Dawid–Skene given the same thresholds matches it. So FUSE's real contribution is a label-free way to pick thresholds for continuous scores, which has nothing to work on when raters give yes/no votes. Small and demographic panels on DICES-350 give the same answer: with like-for-like inputs and a shared cut-off FUSE and Dawid–Skene are level, and the small leads we first saw came from cut-off differences.

## Proposed scope change

Compared with the attached proposal, we would like to focus on binary tasks, as in the original paper, and move multi-class to an optional extension (one-vs-rest, if time allows). The core question, whether FUSE helps on human annotators, can be answered fully on binary data, and the gap DICES exposed is missing data, not the number of classes. Stage 2 would then use a pretrained text encoder instead of a CNN on images.

## Next steps

- Stage 1 on 2–3 binary datasets with gold labels: DICES-350, tweet sentiment (D_PosSent, Zheng et al. 2017) and textual entailment (RTE, Snow et al. 2008).
- Done: missing-aware FUSE (see above). Still open: raters who share fewer than 20 items, as in a thinned crowd.
- Stage 2: train a text model on FUSE labels and compare it with one trained on a single annotator's labels.
- Done: the one regime where the simulation puts FUSE ahead, tested on real labels (see above). Next: repeat it on a second task with a natural correlated block, to see whether it holds beyond RTE. Multi-class stays an optional extension.
- Done: tuning, small panels and demographic panels on DICES-350 (see above); the result is negative. For every comparison from now on: score methods at a shared cut-off as well as at their own rule, and give Dawid–Skene the same answer scale as FUSE (Unsure, score bins).
- Done: what carries FUSE's lead on the paper's own benchmarks (see above). Next: give Dawid–Skene FUSE's thresholds on the crowds with continuous scores (deepfakes, fact-checking, DICES group shares, VIPER), and fix the tie bug in the clustering code before rerunning results that use it.

## Questions for you

- Is DICES a good fit as one of the Stage 1 datasets, or is the subjectivity of the expert label a problem?
- Is an LLM label a legitimate reference where there is no gold label, or should we stick to datasets with gold?
- Should Dawid–Skene, rather than majority vote, be the main baseline to beat in the midterm criterion?
- Are you OK with focusing on binary tasks and moving multi-class to an optional extension?
- Is a careful negative result (FUSE ties Dawid–Skene on human annotators) plus missing-aware FUSE an acceptable outcome for the midterm?
- Do you see a setting, beyond correlated weak raters, where FUSE should beat Dawid–Skene on human or mixed panels?
- Is "FUSE's contribution is a label-free way to pick thresholds for continuous scores" a framing you would accept for the thesis, given that its estimator adds nothing over Dawid–Skene on the paper's benchmarks?
