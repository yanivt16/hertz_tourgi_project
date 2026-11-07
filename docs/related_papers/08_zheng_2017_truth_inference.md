# Truth inference in crowdsourcing: is the problem solved?

**Full text:** [PDF in this repo](pdf/zheng_2017_pvldb-10-5.pdf) (the paper states CC BY-NC-ND 4.0: share unchanged, with attribution, non-commercially) · http://www.vldb.org/pvldb/vol10/p541-zheng.pdf

**Zheng, Y., Li, G., Li, Y., Shan, C., Cheng, R. (2017).** PVLDB 10(5). Tsinghua University and The University of Hong Kong.

- **Cited in the proposal as [5].** A source of binary crowd datasets and a comparison of truth-inference methods.
- **Read status:** the PDF text was extracted and read in part (setup, dataset table, methods table, headline conclusions); the full experimental sections were not read in detail. An earlier note guessed an arXiv id (1611.01247, which is an unrelated physics paper) and a dataset page; neither is used.

## What it is

An experimental survey of 17 existing truth-inference methods (Table 4), evaluated on 5 real crowdsourced datasets. Methods include majority voting (MV) and Dawid–Skene (D&S) and many of D&S's extensions. Its conclusions: MV is the simple baseline; D&S is recommended as the robust classical method; and there is "no best method that beats others" (paraphrased, apart from the quoted words). In its tables D&S beats MV on both binary datasets, for example D_Product accuracy 93.66% vs 89.66% and F1 71.59% vs 59.05%, and D_PosSent accuracy 96.00% vs 93.31%.

## The five datasets (Table 5)

| Dataset | Type | Tasks | Gold ("truth") | Answers \|V\| | Answers per task | Workers |
|---|---|---:|---:|---:|---:|---:|
| D_Product | binary (entity resolution) | 8,315 | 8,315 | 24,945 | 3 | 176 |
| **D_PosSent** | binary (tweet sentiment) | 1,000 | 1,000 | 20,000 | 20 | 85 |
| S_Rel | single-label (relevance, several choices) | 20,232 | 4,460 | 98,453 | 4.9 | 766 |
| S_Adult | single-label (website adult-content level) | 11,040 | 1,517 | 92,721 | 8.4 | 825 |
| N_Emotion | numeric | 700 | 700 | 7,000 | 10 | 38 |

D_Product has a strongly skewed truth (0.12 true : 0.88 false), so the paper reports F1 as well as accuracy there.

## Availability: unresolved

- The paper says (Sec. 6) "We have made all our used datasets and codes available" at its reference [40], http://dbgroup.cs.tsinghua.edu.cn/ligl/crowd_truth_inference/. **That page now returns 404** (checked over http and https), as does the earlier guess `/crowd_data/`.
- The released files themselves, and whether they keep the per-worker answers (the paper's counts \|V\| and \|W\| imply they exist), were **not verified**. Also note that the datasets originate from other works cited in the paper ([50] for D_Product, [9] S_Rel, [4] S_Adult, [44] N_Emotion; D_PosSent from [45]); the original sources may still host them.

## Why it matters for the proposal

**D_PosSent is the best fit for the binary setting:** 1,000 tasks, all with gold labels, 20 answers per task from 85 workers. It also gives a published reference point (MV 93.31% and D&S 96.00% accuracy, from the paper's table). Since D&S beats MV there, D&S is the right strong baseline to include next to majority vote, and FUSE has to beat both. The data must still be located before relying on it.
