# Non-expert crowds on hard questions: deepfakes and fact-checking

Tables: [results/HARD_CROWDS.md](../results/HARD_CROWDS.md). Script: [scripts/run_hard_crowds.py](../scripts/run_hard_crowds.py). Loader: [src/fuse/hard_crowds.py](../src/fuse/hard_crowds.py) (downloads public files into `.cache/hard_crowds`).

## Why these two datasets

The regime map ([regime_map.md](regime_map.md)) left one open question: on real crowds of many non-experts facing a hard question, with gold that does not come from the crowd, does FUSE beat Dawid-Skene? And do demographic groups form the "correlated block" that the simulation says FUSE handles better?

- **Deepfakes** (Groh et al. 2022, PNAS, experiment 1, Prolific). 54 videos, half deepfakes, 304 people, each rating about 20 videos on a 0-100 "confidence it is fake" slider. Gold is known by construction. We keep the first guess per person and video and drop the attention-check video. A single rating has AUC 67; the crowd mean has AUC 90.
- **Fact-checking** (Allen et al. 2021, Science Advances). 189 news articles, 1,128 MTurk laypeople, each rating about 17 articles on seven 1-7 accuracy items (we average them). Gold is the majority verdict of 3 professional fact-checkers from the same file (positive = False or Misleading; articles with no majority or a "Not sure" majority are dropped, 207 → 189). A single rating has AUC 65; the crowd mean has AUC 90.

Not used: Groh's `pilot_treat.csv`, which contains participant IP addresses.

## Design

1. **Individuals as raters.** The matrix is very sparse. Two people share a median of 7 videos (deepfakes) or 1 article (fact-checking), and no pair shares 20. FUSE cannot estimate pairwise or triplet moments here: it returns noise, and for 1,128 raters its triplet tensor would need about 11 GB. So this section compares baselines only.
2. **Demographic groups as raters.** Each group's mean score per item gives a dense matrix of 10-12 continuous raters, the setting FUSE was designed for. Deepfakes: sex × age band × White/not White (12 groups, 12-59 people). Fact-checking: political preference on 6 levels × sex (10 groups with full coverage, 61-146 people).
3. **Random groups of the same sizes**, drawn from the same people, 20 draws. If demographic groups share a bias, FUSE's gain over Dawid-Skene should be larger with demographic groups than with random ones.

## Results

**Everything lands at AUC 89-92, and the simple crowd mean is as good as anything.**

| | Deepfakes | Fact-checking |
|---|---:|---:|
| Crowd mean of individuals (raw scores) | 90.0 | 90.1 |
| Best method on individuals | 91.1 (majority vote) | 90.1 (crowd mean) |
| D&S 5 bins on groups | 90.5 | 90.3 |
| FUSE logistic on groups | 92.5 | 90.9 |
| Mean z-score on groups | 91.4 | 91.2 |
| FUSE product posterior on groups | 89.9 | 89.9 |

1. **On groups, FUSE logistic is in the top two on both datasets but never clearly ahead.** On deepfakes it is +1.9 AUC over binned D&S with a 95% CI of [+0.0, +5.1], with 54 items and 6 comparisons, so it is not a result. On fact-checking it is +0.6 [-0.5, +1.9], behind a plain mean z-score. The FUSE posteriors (triplet and product) are at or below D&S.
2. **The political block is real, but it does not help FUSE.** Within-class correlation is 0.45 between groups on the same political side and 0.23 across sides. That is a shared bias by political side, the structure we were looking for. But FUSE's gain over binned D&S is +0.6 with political groups and +0.0 with random groups, which is within noise. The simulation showed FUSE wins when a minority of weak raters shares an error. Here the two blocks are of similar size and similar accuracy, so there is no bad block to down-weight.
3. **Binned D&S is the worst choice on sparse individuals** (83.8 on fact-checking, 5-6 points below everything else, CI excludes 0). With 17-20 ratings per person, per-rater quantile bins are noise. Median-split D&S or the plain mean is better there.
4. **Demographic groups beat random groups by about 1.5 AUC on deepfakes for every method**, including majority vote, which learns no weights. The cause is not established. It may be how the group mean weights small groups, or real diversity across groups. On fact-checking the difference is under 1 point.

## What it means for the project

This is the third test of the hypothesis "FUSE helps with many non-experts on a hard question," and the third negative. With wisdom-of-crowds data like this, averaging is close to the ceiling, and no label-free reweighting has much room to help. Together with the regime map, the defensible scope for FUSE on human raters is narrow: dense panels with a minority block of correlated weak raters (for example several LLMs from the same family, or a bot cluster). Even there, a z-score average is a strong competitor.

## Caveats

- Small item counts (54 and 189), so differences under about 2-3 AUC are not distinguishable.
- Group means of 12-146 people are much less noisy than individuals. That is why every method lands at about 90.
- Fact-checking gold is 3 fact-checkers, who disagree on some articles. It is closer to "expert consensus" than to ground truth.
- The random-group baseline fills the rare missing item in a small random group with the mean of the other groups. Dropping the column instead would give random panels fewer raters, which happened in a first run and inflated the demographic advantage on fact-checking.
