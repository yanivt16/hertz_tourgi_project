# Where does FUSE beat Dawid-Skene? Simulation map (11 Nov 2026)

Tables: [results/REGIME_MAP.md](../results/REGIME_MAP.md) · script:
[scripts/sim_regime_map.py](../scripts/sim_regime_map.py).

**Question.** Rather than asking "does FUSE work on human annotators", map the regimes where it should
beat Dawid-Skene. One question from the team in particular: does it win with many non-expert raters on
a hard question?

**Model.** Each rater j gives a latent score s_ij = shift_j + d_j y_i + noise. d_j is the rater's
separation: 0.1 is near chance (single-rater AUC ~0.53), 1.5 is an expert (~0.86). On top of that, the
script can make raters correlated (a cluster shares part of its noise), turn scores into yes/no votes,
and drop ratings at random. Base setting: 1000 items, 10 raters, continuous scores, dense, 30% positive.
Each table changes one thing. All methods are label-free.

## Results

| Regime | Best FUSE - best D&S (AUC) | Best method |
|---|---:|---|
| Independent raters: any panel size, difficulty, quality spread, prevalence, scores or votes | -1 to +0.2 | ties; mean z-score is as good or better |
| Hard question for everyone (d = 0.3) | -1.1 | mean z-score (74.5, vs. 62-68 for the rest) |
| Missing ratings (continuous, 50% or 20% observed) | -7.6, -6.6 | mean z-score |
| 20% of raters weak and correlated | +4.0 ± 3.7 | FUSE logistic (96.6; mean z 95.4; D&S binned 92.2) |
| 40% of raters weak and correlated | **+15.0** | mean z-score (87.7; FUSE 82.3; D&S 67.4) |
| 40% of raters correlated junk (no signal) | +8.2 | mean z-score (82.9; FUSE 60.5; D&S 51.5) |
| Many non-experts + 2 experts, hard question, **continuous scores** | **-2 to -25** | binned D&S (~90) |
| Same, **yes/no votes** | about 0 (see the check below) | ties |

## Checks on the two surprising cells

- **FUSE losing up to 25 points on continuous scores (many near-chance raters plus 2 experts).** The
  loss comes from FUSE's label-free threshold search (TCI). On the same data, cutting every rater at the
  median instead gives FUSE AUC 87 instead of 60. When nearly all raters carry almost no signal, the
  threshold search is unstable; in one seed it also dropped one of the two experts. Our TCI
  reproduces the paper on its Best-of-N data, so we read this as a limitation of the method in this
  regime, not an implementation bug, but it is worth a second look.
- **FUSE +8.9 at 100 raters with yes/no votes and d = 0.05 (4 seeds).** This was noise. With 8 seeds
  FUSE gets 76-77 and Dawid-Skene 75.3. Both methods are bimodal across seeds: they either find the
  experts (about 85) or miss them (about 60).

## What this says

1. **The hypothesis "many non-experts on a hard question" does not favour FUSE.** With continuous
   scores FUSE loses clearly, and with votes it ties Dawid-Skene. More raters help every method equally,
   and when everyone is weak a plain average of z-scores is the most reliable choice.
2. **The one regime where FUSE beats Dawid-Skene is correlated weak raters**: a block of raters who
   share an error, making up 20-40% of the panel. Dawid-Skene locks onto the block and collapses.
   With a small block (20%) FUSE is the best method overall, but the gap is noisy (± 3.7). With a
   larger block, a plain average does better than both.
3. **Candidate problem definition for the project**: aggregation when part of the panel shares a
   bias. Examples are a demographic subgroup with the same blind spot, raters trained by the same
   guideline, or several LLMs from the same family. Real data with that structure is the next test.

Limits: Gaussian scores; one correlation structure (a single cluster, rho = 0.6); 6 seeds (4 in the
grids) and 1000 items per cell.
