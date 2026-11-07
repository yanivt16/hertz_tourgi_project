# Human uncertainty makes classification more robust (CIFAR-10H)

**Peterson, J. C., Battleday, R. M., Griffiths, T. L., Russakovsky, O. (2019).** ICCV 2019. arXiv:1908.07086.
Links: https://arxiv.org/abs/1908.07086 · Data: https://github.com/jcpeterson/cifar-10h

- **Cited in the proposal as [6].** A candidate multi-class dataset for both stages.
- **Read status:** the arXiv HTML and the dataset repository (README, licence) were read by a subagent. The raw-data zip was **not downloaded**, so the column list comes from the README.

## The dataset

| | |
|---|---|
| Items | 10,000 images (the CIFAR-10 test set), 10 classes |
| Judgments | 511,400 in total; 51 per image on average (range 47–63) |
| Annotators | 2,571 Amazon Mechanical Turk participants (14 removed for failing attention checks); 210 trials each (200 normal, 10 attention checks) |
| **Per-annotator labels** | **Yes.** `data/cifar10h-raw.zip` → `cifar10h-raw.csv` with `annotator_id`, `chosen_label`, `true_label`, `cifar10_test_set_idx`, `is_attn_check`, `reaction_time` |
| Aggregated files | `cifar10h-counts.npy` (10000×10 counts), `cifar10h-probs.npy` (normalised, the paper's training labels) |
| Gold label | The CIFAR-10 label, recorded as `true_label` per trial (airplane=0 … truck=9) |
| Raw items | The images are not in the repo; they come from CIFAR-10 (https://www.cs.toronto.edu/~kriz/cifar.html) |
| Licence | CC BY-NC-SA 4.0 (non-commercial, share-alike, attribution) |

## Caveats

- Each annotator labels about 200 of the 10,000 images, so the annotator-by-item matrix is sparse and FUSE needs a missing-data treatment (the proposal's Stage 1 implementation step).
- The README's TODO list is unfinished (no loader examples or dataset statistics).
- The gold label is the CIFAR-10 label, which has known errors; a separate verified gold file was not confirmed.

## Why it matters for the proposal

The strongest candidate of the four: many annotators per item, per-annotator labels, a gold label, and the images are obtainable. Nothing needs to be collected.
