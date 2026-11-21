"""FUSE vs Dawid-Skene on two crowds of non-experts on hard questions (src/fuse/hard_crowds.py).

    python scripts/run_hard_crowds.py      # writes results/HARD_CROWDS.md

For each dataset:
1. Individuals as raters. The matrix is sparse (each person rated ~20 items) and two people share only
   a few items (median 7 for deepfakes, 1 for fact-checking; no pair shares 20), so FUSE has no pairs
   or triplets to estimate moments from, and its triplet tensor (m^3) is ~11 GB for 1,128 raters.
   Only the baselines are run here.
2. Demographic groups as raters: mean score of a group's members per item, a dense matrix of ~12
   continuous raters (the setting FUSE was built for). Deepfakes: sex x age x White / not White.
   Fact-checking: political preference (6 levels, strongly Democratic .. strongly Republican) x sex.
3. Random groups with the same sizes, 20 draws. If demographic groups share a bias (a correlated
   block, the one regime where the simulation says FUSE beats Dawid-Skene), FUSE should gain relative
   to Dawid-Skene on demographic groups compared with random groups.
Also reported: within-class correlation between group columns (conditional dependence), split into
same-side and cross-side pairs for political groups.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fuse import hard_crowds  # noqa: E402
from fuse.dices import dawid_skene  # noqa: E402
from fuse.ordinal import dawid_skene_ordinal  # noqa: E402
from run_continuous import boot, methods  # noqa: E402

warnings.filterwarnings("ignore")
DRAWS = 20
GROUPS = {"deepfake": lambda L: L.sex + "|" + L.age + "|" + np.where(L.ethnicity == "White", "White", "nonWhite"),
          "factcheck": lambda L: L.polpref.astype("Int64").astype(str) + "|" + L.sex}
TITLES = {"deepfake": "Deepfake videos (Groh et al. 2022), 0-100 confidence it is fake",
          "factcheck": "News articles (Allen et al. 2021), 1-7 accuracy ratings vs. fact-checkers"}
REF = "Dawid-Skene (5 bins)"


def sparse_methods(V):
    obs = ~np.isnan(V)
    med = np.nanmedian(V, 0)
    B = np.where(obs, np.where(V > med, 1.0, -1.0), np.nan)
    sd = np.nanstd(V, 0)
    Z = (V - np.nanmean(V, 0)) / np.where(sd > 0, sd, 1)
    out = {"Majority vote (median split)": np.nansum(B, 1) / obs.sum(1),
           "Dawid-Skene (median split)": dawid_skene(B),
           REF: dawid_skene_ordinal(V, 5),
           "Mean z-score": np.nanmean(Z, 1),
           "Crowd mean (raw scores)": np.nanmean(V, 1)}
    return out


def within_class_corr(V, y, pairs=None):
    vals = []
    for c in (0, 1):
        C = np.corrcoef(V[y == c].T)
        iu = np.triu_indices(len(C), 1)
        keep = np.ones(len(iu[0]), bool) if pairs is None else pairs[iu]
        vals.append(np.nanmean(C[iu][keep]))
    return float(np.mean(vals))


def table(md, ms, y, extra_cols=None):
    ref = ms[REF]
    md += ["| Method | AUC | AUC vs D&S 5 bins (95% CI) |", "|---|---:|---|"]
    for k, s in ms.items():
        md.append(f"| {k} | {100 * roc_auc_score(y, s):.1f} | {'' if k == REF else boot(s, ref, y)} |")
    md.append("")


def run(md, name):
    L, gold = hard_crowds.load(name)
    V, y, _ = hard_crowds.matrix(L, gold)
    obs = ~np.isnan(V)
    md += [f"## {TITLES[name]}", "",
           f"{V.shape[0]} items ({y.sum()} positive), {V.shape[1]} raters, median {np.median(obs.sum(1)):.0f} "
           f"ratings per item and {np.median(obs.sum(0)):.0f} per rater.", ""]
    print(f"\n{name}: {V.shape}", flush=True)

    O = obs.astype(float)
    C = (O.T @ O)[np.triu_indices(V.shape[1], 1)]
    md += ["### 1. Individuals as raters (sparse)", "",
           f"Two raters share a median of {np.median(C):.0f} items and no pair shares 20 or more "
           f"(max {C.max():.0f}), so FUSE has no moments to estimate here; baselines only.", ""]
    table(md, sparse_methods(V), y)

    L = L.assign(group=GROUPS[name](L))
    L = L[~L.group.str.contains("nan|<NA>")]
    G, y, cols = hard_crowds.matrix(L, gold, by="group")
    keep = ~np.isnan(G).any(0)
    G, cols = G[:, keep], [c for c, k in zip(cols, keep) if k]
    L = L[L.group.isin(cols)]  # random groups are drawn from the same people
    sizes = L.groupby("group").rater.nunique().reindex(cols).to_numpy()
    md += ["### 2. Demographic groups as raters (dense)", "",
           f"{len(cols)} groups ({sizes.min()}-{sizes.max()} people each), mean score per group and item.", ""]
    ms = {k: s for k, (s, _) in methods(G).items()}
    table(md, ms, y)
    demo_auc = {k: 100 * roc_auc_score(y, s) for k, s in ms.items()}
    note = f"Within-class correlation between group columns: {within_class_corr(G, y):.2f}"
    if name == "factcheck":
        side = np.array([int(c.split("|")[0]) <= 3 for c in cols])
        same = side[:, None] == side[None, :]
        note += (f" (same political side {within_class_corr(G, y, same):.2f}, "
                 f"opposite sides {within_class_corr(G, y, ~same):.2f})")
    md += [note + ".", ""]

    # random groups with the same sizes
    raters = L.drop_duplicates("rater")[["rater"]].reset_index(drop=True)
    rnd, corr = [], []
    for seed in range(DRAWS):
        rng = np.random.default_rng(seed)
        lab = np.repeat(np.arange(len(sizes)), sizes)[rng.permutation(sizes.sum())]
        mp = dict(zip(raters.rater.to_numpy()[: len(lab)], lab))
        R, yr, _ = hard_crowds.matrix(L.assign(group=L.rater.map(mp)), gold, by="group")
        # a small random group can miss an item; fill it with that item's mean over the other groups
        # (dropping the column instead would give random panels fewer raters than demographic ones)
        R = np.where(np.isnan(R), np.nanmean(R, 1, keepdims=True), R)
        rnd.append({k: 100 * roc_auc_score(yr, s) for k, (s, _) in methods(R).items()})
        corr.append(within_class_corr(R, yr))
    md += ["### 3. Demographic vs. random groups of the same sizes", "",
           f"Random: mean over {DRAWS} draws (in brackets, the share of draws where the method beats D&S 5 bins). "
           f"Within-class correlation, random groups: {np.mean(corr):.2f}.", "",
           "| Method | Demographic groups | Random groups | FUSE/other minus D&S 5 bins: demographic | random |",
           "|---|---:|---:|---:|---:|"]
    for k in demo_auc:
        r = [d[k] for d in rnd]
        wins = np.mean([d[k] > d[REF] for d in rnd])
        md.append(f"| {k} | {demo_auc[k]:.1f} | {np.mean(r):.1f} ({100 * wins:.0f}%) | "
                  f"{demo_auc[k] - demo_auc[REF]:+.1f} | {np.mean(r) - np.mean([d[REF] for d in rnd]):+.1f} |")
    md.append("")
    print("\n".join(md[-12:]), flush=True)


def main():
    md = ["# Non-expert crowds on hard questions (generated by scripts/run_hard_crowds.py)", "",
          "AUC against gold that does not come from the crowd. Differences are paired bootstraps "
          "(2000 resamples) against Dawid-Skene on 5 quantile bins.", ""]
    for name in hard_crowds.NAMES:
        run(md, name)
    (ROOT / "results" / "HARD_CROWDS.md").write_text("\n".join(md))


if __name__ == "__main__":
    main()
