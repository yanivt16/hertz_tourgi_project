# Shrinking the generation-verification gap with weak verifiers (Weaver)

**Saad-Falcon, J., Buchanan, E. K., Chen, M. F., Huang, T., McLaughlin, B., Bhathal, T., Zhu, S., Athiwaratkun, B., Sala, F., Linderman, S., Mirhoseini, A., Ré, C., et al. (2025).** arXiv:2506.18203.
Link: https://arxiv.org/abs/2506.18203 · Code: https://github.com/HazyResearch/scaling-verification

- **Not cited in the human-annotator proposal; included because FUSE is built on and compared against it.** (It is cited in `docs/sample_proposal/main.tex`.)
- **Read status: not read.** Everything here comes from the FUSE paper's description of it.

## What the FUSE paper says about it

- The Weaver datasets (100 generations per question from Llama 3.1 8B and Llama 3.3 70B; GPQA, GPQA Diamond, MATH500, MMLU, MMLU Pro; up to 33 reward models and LM judges) are the main benchmark FUSE is evaluated on (Sec. 3, App. E.2, Table 6).
- Weaver is the main **semi-supervised** baseline: it "uses a held-out set of queries with ground truth labels" to weight the verifiers; FUSE reaches "essentially exact parity" with it at 70B using no labels (Sec. 3.1).
- In this repo, the Weaver datasets are loaded from HuggingFace (`hazyresearch/*`), and the README's "Weaver paper" column reports its numbers.
- The FUSE authors report running the Weaver replication code and finding "several typos in Tables 1 and 3" of the Weaver paper, which they amended (App. E.2, footnote 6). The FUSE paper cites two entries for Weaver (2025a and 2025b) with the same title and arXiv id; use 2506.18203.
