# Estimating the accuracies of multiple classifiers without labeled data

**Jaffe, A., Nadler, B., Kluger, Y. (2015).** AISTATS 2015, PMLR vol. 38, pp. 407–415.
Links: https://proceedings.mlr.press/v38/jaffe15.html · arXiv:1407.7644 (v2, 30 Oct 2014): https://arxiv.org/abs/1407.7644

- **Cited in the proposal as [2].** This is the method of moments that FUSE builds on (FUSE's Theorem 2.3 restates its Lemmas 1, 2 and 4).
- **Read status:** a subagent read the arXiv v2 HTML. About the first 40% of the text (through Sec. 4.1 and Lemmas 1–5) was read verbatim; the rest (restricted likelihood, experiments, limitations) is **summary only**. The PMLR PDF could not be parsed. We did not verify that arXiv v2 matches the AISTATS camera-ready, and did not read the supplement.
- **Code/data:** no code or data release statement found.

## Setting

- **Binary** labels in {−1, +1}. Observed: an m × n matrix Z of predictions of m classifiers on n unlabeled instances; **m ≥ 3**. Multi-class appears only as a partial result (Sec. 5).
- Unknown per classifier: sensitivity ψ_i, specificity η_i; and the class imbalance b = Pr(Y=1) − Pr(Y=−1). Balanced accuracy π_i = (ψ_i + η_i)/2.
- Goal: estimate ψ_i, η_i consistently and build a more accurate unsupervised ensemble.

## Assumptions

1. Instances i.i.d.
2. **Conditional independence** of the classifiers given the label.
3. More than half of the classifiers are better than random (π_i > 0.5), to fix a sign ambiguity.
4. To recover v uniquely, at least three classifiers have different balanced accuracies, all different from 1/2.
5. The covariance step needs b known; the tensor method (triplet-wise conditional independence) and the restricted likelihood (joint conditional independence; known bounds on b and on ψ, η) remove that need.

## Method

1. Sample mean μ̂ and covariance R̂ of the predictions.
2. **Lemma 1** (from Parisi et al., 2014): off-diagonal entries of R equal those of vvᵀ with v = √(1−b²)(2π−1), up to sign. Estimate v by low-rank matrix completion.
3. **Lemma 2:** closed-form ψ = ½(1 + μ + v√((1−b)/(1+b))), η = ½(1 − μ + v√((1+b)/(1−b))).
4. Unknown b: (a) **tensor method** (Lemma 4: T_ijk = −2b(1−b²)(2π_i−1)(2π_j−1)(2π_k−1)), fitted by least squares; (b) **restricted likelihood**, a 1-D grid scan over b (summary only).
5. **Decision (Eqs. 8–9):** y_ML = sign(Σ_i f_i(x) ln α_i + ln β_i) with α_i = ψ_iη_i / ((1−ψ_i)(1−η_i)). The proposed *i-SML* plugs the estimated ψ̂, η̂ into it. It is intended as a starting point for EM.

## Results

- **Lemma 3:** with known b, ψ̂ and η̂ are consistent with error O_P(1/√n). **Lemma 5:** the tensor-based b̂ has the same O_P(1/√n) rate, which the authors call optimal.
- Experiments (summary only): artificial data with 10 classifiers; real UCI/MNIST datasets with 10 Weka classifiers trained on 200 instances each. Baselines: majority voting, SML (Parisi et al.), oracle ML. i-SML beat SML by about 2% balanced accuracy on Magic; majority voting was "significantly worse" on MNIST. No exact numbers were verified.

## Limits and relation to other work

- Conditional independence is an idealisation; instance difficulty is suggested as future work. The multi-class solution is partial: the text says not all confusion-matrix entries can be estimated from first and second moments.
- **Dawid–Skene (1979):** same sensitivity/specificity model; EM-based methods "whose only guarantee is convergence to a local maxima". This paper is a spectral, non-EM alternative with consistency results.
- **Majority vote:** the framing baseline ("Beyond the simplest approach of majority voting").

## Why it matters for the proposal

FUSE's moment step is this paper's method. Its requirements (≥ 3 classifiers, conditional independence, binary) are why the proposal flags independence of human annotators as a risk, restricts the first version to binary tasks (multi-class via one-vs-rest), and needs enough annotations per item.
