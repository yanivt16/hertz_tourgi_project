# FUSE: Ensembling Verifiers with Zero Labeled Data

**Full text:** [PDF in this repo](pdf/fuse_lee_2026_arxiv-2604.18547.pdf) (arXiv licence: CC BY 4.0) · https://arxiv.org/pdf/2604.18547

**Lee, J., Ma, V., Zhao, S., Nair, Y., Spector, A., Cohen, R., Candès, E. J. (2026).**
arXiv:2604.18547 [stat.ML], v1, submitted 20 Apr 2026. Stanford University (Cohen: Google).
Link: https://arxiv.org/abs/2604.18547 · Full text: https://arxiv.org/html/2604.18547 · Data: https://huggingface.co/FUSE-verifiers

- **Cited in the proposal as [1].** Dr. Regev Cohen (our mentor) is a co-author.
- **Read status:** the full HTML text was read by a subagent in two passes and summarised here. The numbers below were transcribed from that summary. The headline numbers for the Weaver benchmarks, HLE and IMO agree with the "FUSE paper" column of this repo's README (a check against an independent copy of the same tables). A copy of the PDF is kept in `pdf/` (the arXiv page states CC BY 4.0). The supplementary material was not read.
- **No code released.** App. G: "We will open-source a detailed version of our experimental pipeline." Raw HLE and IMO verifier data are public.

## What it does

Given N candidate responses to one question and m imperfect verifiers (LLM judges, reward
models), pick a correct response **without any ground-truth labels**.

| Step | Paper | In this repo |
|---|---|---|
| 1. Binarise each verifier as sign(v_j − τ_j); choose τ by coordinate descent to minimise Ŝ, a label-free measure of triplet-conditional-independence (TCI) violation | Sec. 2.2, Prop. 2.4, Eq. 4 | `src/fuse/tci.py` |
| 2. Estimate each verifier's sensitivity ψ_j, specificity η_j and the class balance b by the method of moments of Jaffe et al. (2015) | Sec. 2.1, Thm 2.3, App. A/D | `src/fuse/moments.py` |
| 3. Drop verifiers with estimated balanced accuracy ≤ 1/2 | App. D | `src/fuse/ensemble.py` |
| 4. Per-response posterior: computed for each verifier triplet, then averaged | App. C, Prop. C.1, Eq. 9 | `src/fuse/posterior.py` |
| 5. Fit a logistic regression on V by maximising Σ (2p̂(r_i) − 1) f_θ(V_i) (Eq. 7); return argmax_i f_θ*(V_i) | Sec. 2.3, Eqs. 6–7, Alg. 1 | `src/fuse/ensemble.py` |

## Setting and assumptions

- Query q fixed; responses r_1..r_N i.i.d.; correctness y_i ∈ {±1}; score matrix V (N × m). Multi-query modes ("batching", "query-conditional") are described.
- **Assumption 2.1:** more than m/2 verifiers have balanced accuracy π_j > 1/2 (fixes the sign ambiguity).
- **Assumption 2.2 (TCI):** each triplet of verifiers is conditionally independent given y. This is weaker than joint conditional independence (JCI), the assumption of Jaffe et al. (2015), which the paper calls "empirically untestable and unlikely to hold in practice" (Sec. 1.2).
- **Theorem 2.3** (restating Jaffe et al.): off-diagonal Σ entries equal those of uuᵀ with u = √(1−b²)(2π−1); off-diagonal third-order tensor entries equal those of w⊗w⊗w; b is recovered from the ratio of the two, then ψ, η follow from the mean vector μ, u and b. O_p(1/√N) consistency.
- **Prop. 2.4:** under TCI a sum of sample variances of tensor/covariance ratios is zero; a large empirical Ŝ "certifies that TCI does not hold." The threshold search minimises Ŝ.
- Real-valued scores are supported (App. A.2, rescaled to [−1, 1]).

## Experiments (Best-of-N selection accuracy, %)

Weaver data: 100 generations per question (Llama 3.1 8B, Llama 3.3 70B), up to 33 open-source reward models and LM judges; semi-supervised baselines get labels for 5% of queries.

| Setting | Pass@1 | Majority vote* | Naive ensemble | FUSE | Weaver | Logistic (5%) | Pass@100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 70B GPQA | 42.9 | 47.4 | 60.1 | 66.8 | 66.4 | 69.3 | 81.0 |
| 70B MATH500 | 78.0 | 82.4 | 92.4 | 92.8 | 93.4 | 96.2 | 98.6 |
| 70B MMLU | 82.6 | 84.1 | 92.4 | 94.1 | 94.9 | 93.1 | 96.0 |
| 8B GPQA | 28.3 | 30.5 | 37.8 | 40.5 | 47.1 | 35.9 | 95.2 |
| 8B MATH500 | 49.9 | 69.6 | 75.2 | 83.9 | 74.8 | 88.0 | 99.2 |

\* **Majority vote here is over the generated responses, not over verifiers.** The paper's verifier-averaging baseline is the *naive ensemble* (equal-weight mean of min-max-normalised scores).

Also: HLE (649 questions, Gemini 3 Pro Preview, 7 verifiers): FUSE 54.3, best other 53.5. IMO Shortlist (123 questions, Qwen3-30B-A3B-Thinking, 9 verifiers): FUSE 63.8, tied with naive ensemble 63.8. A mixed-data ablation (Table 2) widens the FUSE–Weaver gap at 8B from 4.3 to 6.8 points. Supervised baselines beat FUSE in several cells (for example 8B GPQA: Weaver 47.1 vs 40.5). The text's "27 of 40" supervised-win count could not be reconciled with the printed table.

- **Jaffe et al. (2015)** "under-performs a simple naive ensemble in 7 out of 10 settings" (Sec. 2.1). Extra baselines (Dawid–Skene, Jaffe et al. 2016, a Gaussian mixture) are in App. E.8, Table 9, and "uniformly fall below FUSE".
- Ablations (App. D) are described in words only: dropping verifiers is "robustly performance-enhancing"; "simple averaging" of posteriors beats merging verifiers unless m is large; cross-fitting did not help.

## Limits the paper states or shows

- Sec. 5: the primary limitation is "computational and logistical efficiency" (needs several distinct verifier models).
- Conditional dependence is expected to fail for LLM verifiers (Sec. 1.2, 4.2); TCI is still assumed for identification. App. E.2 reports large correlations among reward models; Sec. 3.3 notes near-conditional independence on IMO.
- **Binary labels only** (y ∈ {±1}); no multi-class treatment.
- **Missing scores are imputed as 0** (App. E.4, E.5: 8.63% of HLE scores, 4.15% of IMO scores). No subset-scoring model.
- **No human annotators as data.** Sec. 5 only suggests "FUSE could be used to rank sources of ground-truth such as expert human labelers" and to diagnose benchmark label errors.
- Pseudo-labels are used to fit the paper's own selection rule (Eq. 7); the paper does not train a separate downstream model on them (App. D says benefits would be regularisation, inductive bias and covariates, none of which it uses).

## Why it matters for the proposal

The proposal's Stage 1 and Stage 2 both sit in the gap the paper leaves: human annotators, multi-class, missing entries, and training a model on FUSE labels. The proposal's baselines (naive majority vote over annotators, Dawid–Skene, Jaffe et al. 2015 and 2016) overlap with the paper's own baselines, except that the paper never reports a verifier-level majority vote.
