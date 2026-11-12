# Candidate datasets for the next tests (11 Nov 2026)

Found by a web search (Hugging Face Hub, OSF, GitHub, Harvard Dataverse). Each entry was opened and
its schema checked during the search; we spot-checked the downloaded samples of #2 and #5 afterwards.
Requirements: per-annotator labels with stable annotator IDs, a gold label that does not come from
those annotators, and a binary or binarisable task. Already in use, so not listed: RTE, D_PosSent,
D_Product, Bluebird, DICES, VIPER.

| # | Dataset | Size | Ratings | Gold | Fit |
|---|---|---|---|---|---|
| 1 | Allen et al. 2021, laypeople vs. fact-checkers ([OSF](https://osf.io/hts3w)) | 207 articles × 1,128 laypeople, median 101 per article | 7 Likert items (1-7) | modal verdict of 3 professional fact-checkers | many non-experts, ordinal, hard; the code mapping for the `mis` field is unverified |
| 2 | Groh et al. 2022, deepfake detection ([GitHub](https://github.com/mattgroh/cognitive-science-detecting-deepfakes)) | 54 videos × 304 Prolific participants, median 114 per video | 0-100 confidence slider | real/fake, known by construction | many non-experts, continuous, hard. Do not ingest `pilot_treat.csv` (it contains IP addresses) |
| 3 | Russell et al. 2025, Human Detectors ([GitHub](https://github.com/jenna-russell/human_detectors)) | 300 articles × 5 annotators, fully dense | yes/no + confidence 1-5 | human vs. AI-written, known | small dense panel, different quality (accuracy 0.79-0.96); near-ceiling, so report the harder subsets too. MIT |
| 4 | Nightingale & Farid 2022, real vs. synthetic faces ([OSF](https://osf.io/ru36d)) | 800 faces × 315 participants, ~50 per face | binary | real vs. StyleGAN2, known | very hard (near-chance accuracy), many raters, binary only |
| 5 | Snow et al. 2008 affect, Zheng benchmark copy ([GitHub](https://github.com/zhydhkcws/crowd_truth_infer/tree/master/datasets/f201_Emotion_FULL)) | 700 items × 38 workers, 10 per item | 0-100 | expert scores (threshold to binarise) | continuous, small |
| 6 | Good Judgment Project ([Dataverse](https://doi.org/10.7910/DVN/BPCDH5)) | hundreds of questions × thousands of forecasters | probabilities | resolved real-world outcome | continuous with outcome gold; needs filtering (binary questions, first forecast per user). CC0 |
| 7 | CIFAR-10H raw ([GitHub](https://github.com/jcpeterson/cifar-10h)) | 10,000 images, ~50 labels each | 10-class → one-vs-rest | original CIFAR-10 labels | mostly easy; a control |

Excluded after checking: ChaosNLI (counts only, no annotator IDs); Measuring Hate Speech (gold
fitted from the same annotators); the audio-deepfake perception set on Hugging Face (median one label
per clip); Yandex Relevance-5 and NetEaseCrowd (gold provenance unclear). Not found publicly: the
Prelec et al. 2017 "surprisingly popular" study data.

**Which to run first.**
- #2 (deepfakes) is the real-data version of the "many non-experts, hard question, continuous scores"
  regime, where the simulation predicts Dawid-Skene ≥ FUSE.
- #1 (fact-checking) has political subgroups of raters, which is a natural place to look for the
  correlated-block structure where FUSE should win.
- #3 is the clean small-panel case.
