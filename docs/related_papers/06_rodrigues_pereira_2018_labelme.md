# Deep learning from crowds (LabelMe crowd dataset)

**Full text:** https://arxiv.org/pdf/1709.01779

**Rodrigues, F., Pereira, F. C. (2018).** AAAI 2018. arXiv:1709.01779 (v2, 25 Dec 2017).
Links: https://arxiv.org/abs/1709.01779 · Code and data: http://www.fprodrigues.com/ , https://fprodrigues.com/software/crowdlayer/

- **Cited in the proposal as [7].** A candidate multi-class dataset.
- **Read status:** the arXiv HTML and the author pages were read by a subagent. **The data archive was not downloaded or opened**, so nothing below about its contents is verified.

## What the paper reports

- Datasets: Dogs vs. Cats (simulated annotators, 5), **LabelMe**, MovieReviews, CoNLL-2003 NER. "Source code, datasets and demos for all experiments are provided" on the author's site.
- **LabelMe:** 2,688 images, 8 classes (highway, inside city, tall building, street, forest, coast, mountain, open country). 1,000 images were labelled by Mechanical Turk workers, with **2.547 workers per image on average** and a mean worker accuracy of 69.2%; the other 1,688 images are for evaluation. The paper uses pre-trained VGG-16 features.
- Download listed on the CrowdLayer page: https://fprodrigues.com//deep_LabelMe.tar.gz ("multi-class classification; annotations from Amazon Mechanical Turk").
- The paper contains no music-genre dataset (a subagent searched for one and found none).

## Not confirmed

- Whether the archive keeps **per-annotator** labels or only aggregated ones.
- Whether it contains the **raw images** (or only VGG features), and the gold labels for the evaluation images.
- Licence: not stated on any page read.

## Why it matters for the proposal

Only about 2.5 crowd labels per image on average, with many annotators each labelling a subset: this is the hard case for a moment estimator that needs at least three annotators per triplet, and so a good test of the missing-data extension, but it may be too sparse for FUSE to work at all. Download the archive and inspect it before choosing this dataset.
