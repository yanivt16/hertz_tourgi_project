"""Where does FUSE beat Dawid-Skene? A simulation map over the regimes we care about.

    python scripts/sim_regime_map.py            # writes results/REGIME_MAP.md
    python scripts/sim_regime_map.py --quick    # 2 seeds, for a smoke test

Generative model (one latent Gaussian score per rater and item):
    s_ij = shift_j + d_j * y_i + e_ij,   e_ij ~ N(0, 1)
d_j is rater j's separation (d = 0.3: barely better than chance, single-rater AUC ~ 0.58;
d = 2: an expert, AUC ~ 0.92). Optional pieces:
* correlated raters: a cluster of raters shares part of its noise, e_ij = sqrt(rho) c_i + sqrt(1-rho) u_ij;
* binary votes: each rater says yes when s_ij is above their own threshold (a random quantile);
* missing ratings: each rating is observed with probability `obs`, independently of y.

Methods (all label-free): majority vote, Dawid-Skene (binary, at each rater's median), Dawid-Skene on
5 quantile bins (scores only), mean z-score, FUSE logistic (paper; dense only) and missing-aware FUSE
with the product posterior. AUC against y, mean over seeds.

Each table varies one axis from a base setting; the last two tables are 2-D grids for the hypothesis
"many non-expert raters on a hard question, plus a few experts".
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fuse import FUSE  # noqa: E402
from fuse.dices import dawid_skene  # noqa: E402
from fuse.ordinal import dawid_skene_ordinal  # noqa: E402

warnings.filterwarnings("ignore")
BASE = dict(n=1000, m=10, d_mean=1.0, spread="medium", n_experts=0, prev=0.3, binary=False, obs=1.0,
            corr_frac=0.0, corr_rho=0.0, corr_d=None)
SPREADS = {"none": 0.0, "medium": 0.5, "high": 0.9}  # d_j uniform on d_mean * [1 - s, 1 + s]


def simulate(rng, n, m, d_mean, spread, n_experts, prev, binary, obs, corr_frac, corr_rho, corr_d):
    y = rng.random(n) < prev
    s = SPREADS[spread]
    d = rng.uniform(d_mean * (1 - s), d_mean * (1 + s), m)
    if n_experts:
        d[:n_experts] = 1.5  # a few experts among the non-experts (single-rater AUC ~ 0.86)
    noise = rng.normal(0, 1, (n, m))
    k = int(round(corr_frac * m))
    if k >= 2:
        idx = np.arange(m - k, m)  # the correlated cluster (never the experts)
        noise[:, idx] = np.sqrt(corr_rho) * rng.normal(0, 1, (n, 1)) + np.sqrt(1 - corr_rho) * noise[:, idx]
        if corr_d is not None:
            d[idx] = corr_d
    V = rng.normal(0, 1, m) + y[:, None] * d + noise
    if binary:
        tau = np.array([np.quantile(V[:, j], rng.uniform(0.3, 0.8)) for j in range(m)])
        V = np.where(V > tau, 1.0, -1.0)
    if obs < 1:
        mask = rng.random((n, m)) < obs
        mask[np.arange(n), rng.integers(0, m, n)] = True  # every item has at least one rating
        V = np.where(mask, V, np.nan)
    return V, y.astype(int)


def methods(V, binary):
    dense = not np.isnan(V).any()
    med = np.nanmedian(V, 0)
    B = np.where(np.isnan(V), np.nan, np.where(V > med, 1.0, -1.0)) if not binary else V
    obs = ~np.isnan(B)
    tie = np.random.default_rng(0).random(len(V)) * 1e-6
    sd = np.nanstd(V, 0)
    Z = (V - np.nanmean(V, 0)) / np.where(sd > 0, sd, 1)
    out = {"MV": np.nansum(B, 1) / np.maximum(obs.sum(1), 1) + tie,
           "D&S": dawid_skene(B),
           "D&S 5 bins": None if binary else dawid_skene_ordinal(V, 5),
           "Mean z": np.nanmean(Z, 1) + tie,
           "FUSE logistic": FUSE().fit_scores(V).scores if dense else None,
           "FUSE product": FUSE(predictor="posterior", missing="skip", posterior_mode="product").fit_scores(V).scores}
    return out


KEYS = ["MV", "D&S", "D&S 5 bins", "Mean z", "FUSE logistic", "FUSE product"]


def run_cell(cfg, seeds):
    res = {k: [] for k in KEYS}
    gap = []
    for seed in range(seeds):
        V, y = simulate(np.random.default_rng(seed), **cfg)
        ms = methods(V, cfg["binary"])
        auc = {k: (100 * roc_auc_score(y, s) if s is not None else np.nan) for k, s in ms.items()}
        for k in KEYS:
            res[k].append(auc[k])
        best_f = np.nanmax([auc["FUSE logistic"], auc["FUSE product"]])
        best_d = np.nanmax([auc["D&S"], auc["D&S 5 bins"]])
        gap.append(best_f - best_d)
    mean = {k: np.nanmean(v) if not np.all(np.isnan(v)) else np.nan for k, v in res.items()}
    return mean, float(np.mean(gap)), float(np.std(gap) / np.sqrt(len(gap)))


def fmt(x):
    return "" if np.isnan(x) else f"{x:.1f}"


def axis_table(md, title, note, key, values, seeds, base=None, label=None):
    base = dict(BASE if base is None else base)
    md += [f"### {title}", "", note, "",
           f"| {label or key} | " + " | ".join(KEYS) + " | best FUSE - best D&S (± s.e.) |",
           "|---|" + "---:|" * (len(KEYS) + 1)]
    for v in values:
        cfg = dict(base, **(v if isinstance(v, dict) else {key: v}))
        name = cfg.pop("_name", v)
        t0 = time.time()
        mean, g, se = run_cell(cfg, seeds)
        md.append(f"| {name} | " + " | ".join(fmt(mean[k]) for k in KEYS) + f" | **{g:+.1f}** ± {se:.1f} |")
        print(f"{title} | {name}: gap {g:+.1f} ({time.time() - t0:.0f}s)", flush=True)
    md.append("")


def grid(md, title, note, base, seeds, ms=(5, 20, 50, 100), ds=(0.05, 0.1, 0.2, 0.35)):
    md += [f"### {title}", "", note, "",
           "Cell = best FUSE AUC - best Dawid-Skene AUC (positive = FUSE wins); in brackets, best D&S AUC.", "",
           "| raters \\ non-expert d | " + " | ".join(str(d) for d in ds) + " |", "|---|" + "---:|" * len(ds)]
    for m in ms:
        row = []
        for d in ds:
            mean, g, se = run_cell(dict(base, m=m, d_mean=d), seeds)
            best_d = np.nanmax([mean["D&S"], mean["D&S 5 bins"]])
            row.append(f"{g:+.1f} ({best_d:.0f})")
            print(f"{title} | m={m} d={d}: gap {g:+.1f} ± {se:.1f}", flush=True)
        md.append(f"| {m} | " + " | ".join(row) + " |")
    md.append("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    seeds = 2 if args.quick else 6
    md = ["# Where does FUSE beat Dawid-Skene? (generated by scripts/sim_regime_map.py)", "",
          f"AUC against the true label, mean over {seeds} seeds. Base setting: 1000 items, 10 raters, "
          "continuous scores, dense, 30% positive, rater separation d uniform on [0.5, 1.5] "
          "(single-rater AUC 0.64-0.86), no correlation. Each table changes one thing. "
          "'best FUSE - best D&S' compares the better FUSE variant with the better Dawid-Skene variant "
          "in each run.", ""]
    md += ["## One axis at a time", ""]
    axis_table(md, "Number of raters", "", "m", [3, 5, 10, 20, 50], seeds)
    axis_table(md, "Difficulty (mean separation d)", "d = 0.3 is a hard question for every rater.", "d_mean", [0.3, 0.5, 1.0, 1.5], seeds)
    axis_table(md, "Spread of rater quality", "none: all raters equal; high: d from 0.1 d_mean to 1.9 d_mean.", "spread", ["none", "medium", "high"], seeds)
    axis_table(md, "Prevalence", "", "prev", [0.5, 0.3, 0.1], seeds)
    axis_table(md, "Scores vs. yes/no votes", "", "binary", [False, True], seeds, label="binary votes")
    axis_table(md, "Missing ratings (continuous)", "Share of ratings observed, at random.", "obs", [1.0, 0.5, 0.2], seeds, label="observed")
    axis_table(md, "Correlated raters", "A cluster of raters shares part of its noise (rho = 0.6). "
               "'weak' cluster: d = 0.3; 'junk' cluster: d = 0 (no signal at all).", "corr", [
                   {"_name": "none"},
                   {"_name": "20% weak, correlated", "corr_frac": 0.2, "corr_rho": 0.6, "corr_d": 0.3},
                   {"_name": "40% weak, correlated", "corr_frac": 0.4, "corr_rho": 0.6, "corr_d": 0.3},
                   {"_name": "40% junk, correlated", "corr_frac": 0.4, "corr_rho": 0.6, "corr_d": 0.0},
                   {"_name": "60% weak, correlated", "corr_frac": 0.6, "corr_rho": 0.6, "corr_d": 0.3}], seeds, label="cluster")
    md += ["## Hypothesis: many non-experts on a hard question", "",
           "Non-experts have separation d (all equal, spread none; d = 0.1 means single-rater AUC ~ 0.53); "
           "2 of the raters are experts with d = 1.5, whatever the panel size. "
           "Rows: number of raters; columns: how hard the question is for the non-experts.", ""]
    base = dict(BASE, spread="none", n_experts=2)
    gseeds = max(2, seeds - 2)
    grid(md, "Continuous scores", "", base, gseeds)
    grid(md, "Yes/no votes", "", dict(base, binary=True), gseeds)
    grid(md, "Yes/no votes, 40% of non-experts in a correlated cluster (rho = 0.6)", "",
         dict(base, binary=True, corr_frac=0.4, corr_rho=0.6), gseeds)
    out = ROOT / "results" / ("REGIME_MAP_quick.md" if args.quick else "REGIME_MAP.md")
    out.write_text("\n".join(md))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
