"""Does FUSE beat Dawid-Skene when raters give continuous scores?

    python scripts/run_continuous.py      # writes results/CONTINUOUS.md

FUSE's extra machinery over Dawid-Skene is the label-free choice of a per-rater threshold
(TCI) and moment estimates that work on any score scale. On binary votes there is nothing to
threshold, and missing-aware FUSE only matched Dawid-Skene (results/MISSING_REAL.md). Here
every method sees the same score matrix, dense, no missing entries:

* Majority vote / Dawid-Skene: scores binarised at each rater's median;
* Dawid-Skene (5 bins): each rater's scores cut into 5 quantile bins, categorical EM
  (src/fuse/ordinal.py), the strongest Dawid-Skene variant for scores;
* Mean z-score: per-rater standardised scores, averaged (unweighted);
* FUSE: paper defaults (logistic and triplet posterior) and the product posterior.

Part 1 simulates Gaussian raters (known truth, Bayes-optimal oracle as ceiling).
Part 2 uses the real score data we have: VIPER dense blocks (suspicion score 0-201, gold =
cancer) and DICES-350 (granular reason counts and demographic-group shares, expert gold).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fuse import FUSE, dices, viper  # noqa: E402
from fuse.dices import dawid_skene  # noqa: E402
from fuse.ordinal import dawid_skene_ordinal  # noqa: E402

SEEDS = 10


def methods(V: np.ndarray) -> dict[str, tuple[np.ndarray, bool]]:
    """name -> (score, is_probability). V: items x raters, higher = more positive."""
    B = np.where(V > np.median(V, 0), 1.0, -1.0)
    share = (B > 0).mean(1)
    Z = (V - V.mean(0)) / np.where(V.std(0) > 0, V.std(0), 1)
    out = {
        "Majority vote (median split)": (share, True),
        "Dawid-Skene (median split)": (dawid_skene(B), True),
        "Dawid-Skene (5 bins)": (dawid_skene_ordinal(V, 5), True),
        "Mean z-score": (Z.mean(1), False),
        "FUSE logistic (paper)": (FUSE().fit_scores(V).scores, True),
        "FUSE triplet posterior (paper)": (FUSE(predictor="posterior").fit_scores(V).scores, True),
        "FUSE product posterior": (FUSE(predictor="posterior", missing="skip", posterior_mode="product").fit_scores(V).scores, True),
    }
    return out


def score(y, s, prob):
    auc = 100 * roc_auc_score(y, s)
    acc = 100 * float(((s > 0.5) == y).mean()) if prob else np.nan
    return auc, acc


# ---- part 1: simulation ------------------------------------------------------------------

SCENARIOS = [
    ("Similar raters, balanced", dict(d=(1.0, 1.5), prev=0.5)),
    ("Raters differ in quality, balanced", dict(d=(0.1, 2.5), prev=0.5)),
    ("Raters differ in quality, 15% positive", dict(d=(0.1, 2.5), prev=0.15)),
    ("Differ in quality, 15% positive, skewed score scales", dict(d=(0.1, 2.5), prev=0.15, skew=True)),
    ("As above + 3 correlated junk raters", dict(d=(0.1, 2.5), prev=0.15, skew=True, junk=3)),
]


def simulate(rng, d, prev, n=1000, m=10, skew=False, junk=0):
    y = rng.random(n) < prev
    dj = rng.uniform(*d, m)
    shift = rng.normal(0, 1, m)
    latent = y[:, None] * dj + rng.normal(0, 1, (n, m))
    llr = (dj * (latent - dj / 2)).sum(1) + np.log(prev / (1 - prev))
    V = latent + shift
    if skew:
        V = np.exp(rng.uniform(0.3, 1.2, m) * V)
    if junk:
        common = rng.normal(0, 1, n)
        V = np.c_[V, common[:, None] * 1.5 + rng.normal(0, 0.5, (n, junk))]
    return V, y, llr


def part1(md):
    md += ["## Part 1: simulated raters", "",
           f"1000 items, 10 raters. Rater j's score = shift_j + d_j * y + N(0, 1), d_j uniform on the given "
           f"range (0.1-2.5 means some raters are near-useless and some excellent). 'Skewed' applies a "
           f"different exp transform per rater. Mean over {SEEDS} seeds. Oracle = Bayes-optimal rule with the "
           "true parameters.", ""]
    for name, kw in SCENARIOS:
        res = []
        for seed in range(SEEDS):
            V, y, llr = simulate(np.random.default_rng(seed), **kw)
            row = {k: score(y, s, p) for k, (s, p) in methods(V).items()}
            row["Oracle (true parameters)"] = score(y, 1 / (1 + np.exp(-llr)), True)
            res.append(row)
        keys = list(res[0])
        auc = {k: np.mean([r[k][0] for r in res]) for k in keys}
        acc = {k: np.nanmean([r[k][1] for r in res]) if not np.isnan(res[0][k][1]) else np.nan for k in keys}
        print(f"\n{name}"); [print(f"  {k:34s} AUC {auc[k]:5.1f}  acc {acc[k]:5.1f}") for k in keys]
        md += [f"### {name}", "", "| Method | AUC | Accuracy |", "|---|---:|---:|"]
        md += [f"| {k} | {auc[k]:.1f} | {'' if np.isnan(acc[k]) else f'{acc[k]:.1f}'} |" for k in keys] + [""]


# ---- part 2: real scores -----------------------------------------------------------------

def boot(a, b, y, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), (n, len(y)))
    d = [roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]) for i in idx if 0 < y[i].sum() < len(i)]
    lo, hi = np.percentile(d, [2.5, 97.5])
    return f"{100 * (roc_auc_score(y, a) - roc_auc_score(y, b)):+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"


def real(md, name, V, y, extra=None):
    y = np.asarray(y).astype(int)
    ms = methods(V)
    if extra:
        ms = {**extra, **ms}
    ref = ms["Dawid-Skene (5 bins)"][0]
    head = f"{V.shape[0]} items ({y.sum()} positive), {V.shape[1]} raters"
    print(f"\n{name}: {head}")
    md += [f"### {name}", "", head, "", "| Method | AUC | Accuracy | AUC vs Dawid-Skene 5 bins (95% CI) |", "|---|---:|---:|---|"]
    for k, (s, p) in ms.items():
        auc, acc = score(y, s, p)
        diff = "" if k == "Dawid-Skene (5 bins)" else boot(s, ref, y)
        print(f"  {k:36s} AUC {auc:5.1f}  acc {acc:5.1f}  {diff}")
        md.append(f"| {k} | {auc:.1f} | {'' if np.isnan(acc) else f'{acc:.1f}'} | {diff} |")
    md.append("")


def part2(md):
    md += ["## Part 2: real score data", "",
           "Dense matrices only (no missing entries), so no method can use missingness. Differences are "
           "paired bootstraps against Dawid-Skene with 5 bins.", ""]
    reads, gold = viper.load()
    for mod in ["FFDM", "SFM"]:
        r = reads[reads.modalityID == mod]
        blocks = viper.dense_blocks(r)
        for lo in (90, 200):
            n, (Rs, cs) = max(((n, b) for n, b in blocks.items() if n >= lo), key=lambda t: len(t[1][0]))
            S, Bv, y = viper.build(r[r.readerID.isin(Rs) & r.caseID.isin(cs)], gold)
            extra = {"Dawid-Skene (readers' own recall decision)": (dawid_skene(2 * Bv - 1), True)}
            real(md, f"VIPER {mod} suspicion score, {len(Rs)} readers x {n} cases", S, y, extra)
    df = dices.load("350")
    for view, groups in [("granular", None), ("groups", ("rater_race", "rater_gender")), ("groups", ("rater_race", "rater_gender", "rater_age"))]:
        m = dices.build(df, view, groups=groups or ("rater_race", "rater_gender"))
        if np.isnan(m.V).any():
            keep = ~np.isnan(m.V).any(0)
            m.V = m.V[:, keep]
        label = view if not groups else "groups: " + " x ".join(g.replace("rater_", "") for g in groups)
        real(md, f"DICES-350 {label} (expert label)", m.V, m.gold)


def main():
    md = ["# FUSE on continuous scores (generated by scripts/run_continuous.py)", ""]
    part1(md)
    part2(md)
    (ROOT / "results" / "CONTINUOUS.md").write_text("\n".join(md))


if __name__ == "__main__":
    main()
