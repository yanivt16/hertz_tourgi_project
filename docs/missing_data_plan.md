# Missing ratings in FUSE: problem and plan

Draft for discussion (Ohad, 10 Nov 2026). Stage 1's main methodological work.

## The problem

The paper's setting has no missing data: every verifier scores every response. With human
annotators most entries are missing (DICES-990 58%, VIPER 67%, D_PosSent 76%, RTE 94%).
Following the paper's convention we set missing entries to 0 after mapping each column to
[-1, 1]. That is not neutral. FUSE binarises every column at a threshold tau, so **each
missing rating becomes a vote**: +1 if tau < 0, -1 otherwise. On VIPER FFDM, 63% of the
missing recall decisions become "recall" votes and the rest "no recall"
(30% / 70% for the suspicion score).

Consequences we have already seen:

1. **Biased moments.** Two raters who skipped the same items look like they agree. The
   rank-one fit then reads shared missingness as shared skill.
2. **Weak results on sparse data.** On DICES-990 as a whole FUSE equals majority vote and
   Dawid-Skene, which skips missing ratings, is 6 points ahead. On dense blocks of the same
   data FUSE ties Dawid-Skene ([DICES990_BLOCKS.md](../results/DICES990_BLOCKS.md)).
3. **Label leakage when missingness depends on the label.** In VIPER, cancers were assigned to
   more readers and to particular reader groups, so *which* readers read a case predicts
   cancer with AUC 94 using no scores. Zero-imputed FUSE reaches AUC 81-84 on the full
   matrix; on dense blocks it does no better than averaging the scores
   ([VIPER.md](../results/VIPER.md)).

## Two kinds of missingness

| Kind | Example | What is safe |
|---|---|---|
| Assignment independent of the label (random or by batch) | Crowdsourcing: RTE, D_PosSent, most of DICES-990 | Skipping missing entries (pairwise-complete moments) is unbiased |
| Assignment depends on the label (enriched designs) | VIPER, many reader studies | Skipping is still not leaking, but per-pair prevalence differs, so the rank-one structure breaks. Estimate per design stratum (sub-study) or on dense blocks |

Diagnostic to run on every dataset before any result: cross-validated AUC of predicting the
gold label from the missingness pattern alone (`fuse.viper.assignment_leak`, to be made
generic). Near 50 means kind 1; well above 50 means kind 2.

## Proposal: missing-aware FUSE

Keep FUSE's steps, never turn a missing rating into a value.

1. **Binarise observed entries only.** `B` in {+1, -1, NaN}.
2. **Pairwise-complete moments.**
   - Mean of verifier j: over items j rated.
   - Covariance (j, k): over items both rated, centred with the means on those items.
   - Third moment (j, k, l): over items all three rated.
   - An entry with fewer than `n_min` co-rated items (start with 20) is treated as unobserved.
     `rank_one_offdiagonal` already accepts an `observed` mask. `estimate_class_imbalance`
     and the TCI statistic use only the triplets that are observed.
3. **Threshold search.** Same coordinate descent, with the masked TCI statistic. The
   incremental moment update needs a masked version. This is the main engineering cost.
4. **Posterior per item from observed raters only.** Two variants, report both:
   - A. Average over the triplets fully observed on that item. This is the paper's eq. 9
     restricted to what exists.
   - B. Product over all observed raters (naive Bayes with the MoM parameters). This equals
     one Dawid-Skene E-step and is sharper when an item has many ratings.
5. **Final predictor.** The logistic step needs complete features, so on sparse data use the
   posterior as the final predictor.
6. **Optional: FUSE as the initialiser of Dawid-Skene EM.** Spectral or moment estimates as
   EM initialisation is the approach of Zhang, Chen, Zhou and Jordan (2014), "Spectral methods
   meet EM". It gives a natural "FUSE + EM" variant to compare with plain Dawid-Skene.

Sanity property: with no missing entries, missing-aware FUSE must reproduce current FUSE
exactly. This becomes a unit test.

## Alternatives considered

| Option | Why not (as the main method) |
|---|---|
| Impute with each rater's mean | After binarisation it is still a vote, unless it lands exactly on tau |
| Low-rank matrix completion | Fills gaps with consensus, so raters look more accurate than they are |
| Dense blocks only (what DICES-990 does) | Clean, but drops data and needs a design with blocks. Keep it as the reference result |

## How to evaluate

1. **Simulation (first, cheapest, most convincing).**
   - Setup: known sensitivity and specificity, 20 to 100 raters, missing rate from 0% to 95%,
     assignment random or label-dependent.
   - Measure: parameter recovery and accuracy against the true labels.
   - Compare: zero-imputed FUSE, missing-aware FUSE (A, B), FUSE + EM, Dawid-Skene, majority vote.
2. **Real data.**
   - DICES-990: target is the dense-block result, 75.6 against the LLM label, with
     Dawid-Skene at 77.5.
   - RTE (94% missing) and D_PosSent (76% missing), both with gold.
   - VIPER: per sub-study and on dense blocks, with the leakage diagnostic.
3. **Success criterion.** On sparse data, missing-aware FUSE beats zero-imputed FUSE and closes
   most of the gap to Dawid-Skene. On dense data, nothing changes.

## Order of work

1. Generic leakage diagnostic.
2. Masked moments and masked TCI, with the unit test (no missing gives identical output).
3. Posteriors A and B.
4. Simulation.
5. Real datasets.
6. FUSE + EM.
