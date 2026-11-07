# Maximum likelihood estimation of observer error-rates using the EM algorithm

**Dawid, A. P., Skene, A. M. (1979).** Journal of the Royal Statistical Society, Series C (Applied Statistics) 28(1), 20–28.

- **Cited in the proposal as [3].**
- **Read status: not read.** No full text was fetched. The only things recorded here come from how other papers describe it.

## What other papers say about it

- Jaffe et al. (2015) call it the first to define and address the problem of estimating classifier accuracies without labels, and describe such EM-based methods as having "only" local-maximum convergence guarantees (see `02_jaffe_nadler_kluger_2015.md`).
- The FUSE paper (App. E.8) describes it as "an Expectation-Maximization algorithm that assumes conditional independence," and reports it as a baseline, below FUSE on all ten Weaver settings in Table 9 (for example 8B MATH500: 0.5513). Sec. 2.1 notes that conditional-independence assumptions of this kind are common in the crowdsourcing literature.
- The proposal uses it as the stronger standard baseline after naive majority vote.

## To do

Read the paper itself before writing anything about its model beyond the above (annotator error matrices, the multi-class form, initialisation).
