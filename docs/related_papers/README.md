# Related papers

One note per paper cited by the FUSE human-annotator proposal (`docs/fuse_crowd_proposal/`), plus Weaver, which FUSE builds on. Each note starts with a **Full text** line (a direct PDF link; the two papers whose licence allows it are also stored in [pdf/](pdf/)) and says how much of the paper was actually read; **a note is only as reliable as its "Read status" line.** Notes were written from a subagent's reading of the web pages, not from our own reading of the PDFs, and have not been proofread against the papers.

| # | Paper | Role | Read status |
|---|---|---|---|
| 1 | [FUSE, Lee et al. 2026](01_fuse_lee_2026.md) · [PDF](pdf/fuse_lee_2026_arxiv-2604.18547.pdf) | The method we extend; mentor is a co-author | Full HTML text, summarised in detail |
| 2 | [Jaffe, Nadler, Kluger 2015](02_jaffe_nadler_kluger_2015.md) | The moment estimator inside FUSE | arXiv HTML, first ~40% verbatim, rest summary only |
| 3 | [Jaffe et al. 2016](03_jaffe_fetaya_nadler_jiang_kluger_2016.md) | Baseline for dependent classifiers | **Abstract only** |
| 4 | [Dawid & Skene 1979](04_dawid_skene_1979.md) | Classic EM baseline | **Not read** (described from other papers) |
| 5 | [Peterson et al. 2019, CIFAR-10H](05_peterson_2019_cifar10h.md) | Candidate multi-class dataset | Paper and repository read; zip not downloaded |
| 6 | [Rodrigues & Pereira 2018, LabelMe](06_rodrigues_pereira_2018_labelme.md) | Candidate multi-class dataset | Paper and author pages read; archive not opened |
| 7 | [Snow et al. 2008, RTE](07_snow_2008_rte.md) | Candidate binary dataset | **Metadata page only** |
| 8 | [Zheng et al. 2017](08_zheng_2017_truth_inference.md) · [PDF](pdf/zheng_2017_pvldb-10-5.pdf) | Candidate binary datasets and method comparison | PDF read in part (datasets, methods, conclusions) |
| 9 | [Weaver, Saad-Falcon et al. 2025](09_weaver_saad_falcon_2025.md) | FUSE's main benchmark and semi-supervised baseline | **Not read** (described from the FUSE paper) |

## Dataset availability summary (for the proposal's "no data collection" requirement)

| Dataset | Per-annotator labels | Raw items | Gold | Status |
|---|---|---|---|---|
| CIFAR-10H | yes (`cifar10h-raw.csv`) | from CIFAR-10 (public) | yes (CIFAR-10 labels) | usable as is |
| LabelMe (crowd) | unconfirmed | unconfirmed | unconfirmed | download and inspect |
| RTE (Snow et al.) | unconfirmed | unconfirmed | unconfirmed | locate the data first |
| Zheng et al. (D_PosSent: 1,000 binary tasks, 20 answers each, gold) | paper implies yes | unconfirmed | yes | the paper's dataset page returns 404; find the files first |

## Open items

- Read the PDFs of papers 2–4 (the PMLR PDFs did not parse for the subagent).
- Find where the Zheng et al. datasets are hosted (the page cited in the paper returns 404).
- Open the LabelMe archive, and locate the Snow et al. RTE data.
