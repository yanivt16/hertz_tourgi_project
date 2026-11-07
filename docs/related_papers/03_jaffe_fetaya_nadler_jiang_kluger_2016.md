# Unsupervised ensemble learning with dependent classifiers

**Full text:** https://proceedings.mlr.press/v51/jaffe16.pdf

**Jaffe, A., Fetaya, E., Nadler, B., Jiang, T., Kluger, Y. (2016).** AISTATS 2016 (Cadiz), PMLR vol. 51, pp. 351–360.
Link: https://proceedings.mlr.press/v51/jaffe16.html

- **Cited in the proposal as [8].** Listed in the FUSE paper as an extra baseline (App. E.8; "an extension of Jaffe et al. (2015) to settings with structured conditional dependence"). The FUSE paper's Table 9 has a column labelled "DCL" that the text never defines; it may refer to this method, which is unconfirmed.
- **Read status: abstract only.** The PMLR PDF returned binary data and could not be parsed. Nothing below the abstract has been read, so the method, theory and experiments are **not summarised here**. Open the PDF directly before relying on this paper.
- No arXiv link was found on the PMLR page; do not guess one.

## What the abstract says

- Most earlier work assumed "perfect diversity" (conditional independence) between sources, which is often violated in realistic scenarios, and ensembles built on it "can be severely sub-optimal."
- The paper introduces a statistical model that allows dependencies between classifiers, and develops unsupervised methods to **detect strongly dependent classifiers**, **better estimate their accuracies**, and **construct an improved meta-learner**.
- Tested on artificial and real datasets, showing "the importance of taking classifier dependencies into account and the competitive performance of our approach."

## Why it matters for the proposal

It is the one baseline in the FUSE paper that addresses dependent classifiers, which is the risk the proposal flags for human annotators who share guidelines and biases. What dependence structure it assumes, and how it handles it, still has to be read from the PDF.
