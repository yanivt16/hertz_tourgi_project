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

## First results (10 Nov 2026)

Implemented as `FUSE(missing="skip")` in [src/fuse/missing.py](../src/fuse/missing.py). The
default (`missing="zero"`) is unchanged. With no missing entries, skip mode reproduces the
current FUSE exactly; this is unit-tested in `tests/test_missing.py`.

Simulation, 30 conditionally independent raters ([results/MISSING_SIM.md](../results/MISSING_SIM.md)):

- **Zero-imputation breaks quickly.** At 50% observed it gets 73% accuracy against
  97% for the other methods. Its parameter error is 0.25, against 0.02 with skip.
- **Skip with the product posterior matches Dawid-Skene** down to about 7 ratings per item.
- **At 3 or fewer ratings per item** too few rater pairs share items, so the moments fail.

Real data ([results/MISSING_REAL.md](../results/MISSING_REAL.md)), accuracy against gold:

| Dataset | MV | Dawid-Skene | FUSE missing = 0 | FUSE skip, triplet | FUSE skip, product |
|---|---:|---:|---:|---:|---:|
| RTE (94% missing) | 90.1 | 93.0 | 69.1 | 90.1 | 92.0 |
| D_PosSent (76%) | 93.3 | 95.9 | 84.0 | 95.6 | 95.6 |
| D_Product (98%) | 89.7 | 93.8 | 87.8 | 92.6 | 92.6 |
| DICES-990 (58%, LLM label) | 71.8 | 77.5 | 70.9 | 75.2 | **77.7** |
| DICES-350 (dense, expert) | 65.4 | 68.6 | 65.1 | 65.1 | 68.0 |
| Bluebird (dense) | 75.9 | 88.9 | 86.1 | 86.1 | 88.9 |

What this says:

1. **Most of FUSE's gap to Dawid-Skene on human data was missing-data handling.**
   - On sparse crowds, zero-imputed FUSE is far below even majority vote (RTE: 21 points
     below). Skipping missing ratings recovers it.
   - On DICES-990 the full sparse matrix now ties Dawid-Skene (+0.2) and beats MV by 5.9.
     The dense-block workaround gave 75.6, and this needs no blocks.
2. **The other part is the triplet-averaged posterior (eq. 9).**
   - With many raters, averaging posteriors from 3 raters at a time throws away evidence.
     On dense DICES-350 it gives 65.1, the same as MV. The product over all raters gives 68.0.
   - This explains the earlier finding that "per rater, FUSE = MV".
3. **FUSE + EM ends at the same fixed point as Dawid-Skene**, so it adds nothing. Dropped
   as a method.
4. **On VIPER, skip mode is worse than MV** (AUC 68-71 vs 73). This is consistent with the
   warning above: when assignment depends on the label, prevalence differs between rater
   pairs and the rank-one moment model is wrong. Not further investigated.
5. **Honest bottom line.** On binary human labels under conditional independence,
   missing-aware FUSE matches Dawid-Skene but does not beat it. Dawid-Skene is the MLE for
   exactly this model. Where FUSE could still add something is continuous scores
   (thresholds chosen by TCI) and few, correlated verifiers. That is the next question for
   Stage 1.
