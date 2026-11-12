"""Can a change to FUSE's threshold search keep its win on a correlated weak block without its losses?

    python scripts/tci_fix_sim.py [seeds]   # writes results/tci_fix_sim.json (see docs/threshold_search_fixes.md)

Simulated raters (scripts/sim_regime_map.py), 10 regimes, 6 seeds. FUSE variants: the paper's search
(5-95% quantiles), median thresholds (no search), narrowed ranges, the scale-free TCI statistic, and
clustering of correlated raters (cluster_threshold) with and without the search. Baselines include
Dawid-Skene, mean z-score and the FUSE logistic step fed with Dawid-Skene posteriors, each also on the
clustered columns, to see whether a gain belongs to FUSE or to the clustering.
"""
import sys, warnings, json, time
warnings.filterwarnings("ignore")
from pathlib import Path
R = str(Path(__file__).resolve().parents[1]); sys.path.insert(0, R + "/src"); sys.path.insert(0, R + "/scripts")
import numpy as np
from sklearn.metrics import roc_auc_score as auc
from fuse import FUSE
from fuse.dices import dawid_skene
from fuse.ordinal import dawid_skene_ordinal
from fuse.preprocess import minmax_to_unit_interval
from fuse.clustering import cluster_verifiers
from sim_regime_map import simulate, BASE

SCEN = {
    "base (10 raters, d 0.5-1.5)": dict(BASE),
    "5 raters": dict(BASE, m=5),
    "hard: d=0.3": dict(BASE, d_mean=0.3),
    "high quality spread": dict(BASE, spread="high"),
    "20 near-chance + 2 experts": dict(BASE, m=20, d_mean=0.1, spread="none", n_experts=2),
    "50 near-chance + 2 experts": dict(BASE, m=50, d_mean=0.1, spread="none", n_experts=2),
    "20% weak corr block": dict(BASE, corr_frac=0.2, corr_rho=0.6, corr_d=0.2),
    "40% weak corr block": dict(BASE, corr_frac=0.4, corr_rho=0.6, corr_d=0.2),
    "40% normal-quality corr block": dict(BASE, corr_frac=0.4, corr_rho=0.6),
    "binary votes": dict(BASE, binary=True),
}
VARIANTS = {
    "paper (5-95%)": dict(),
    "median only": dict(n_sweeps=0),
    "range 20-80%": dict(quantile_range=(0.2, 0.8)),
    "range 30-70%": dict(quantile_range=(0.3, 0.7)),
    "standardized stat": dict(tci_standardize=True),
    "std + 20-80%": dict(tci_standardize=True, quantile_range=(0.2, 0.8)),
    "cluster@0.5": dict(cluster_threshold=0.5),
    "cluster@0.5 + median": dict(cluster_threshold=0.5, n_sweeps=0),
}


def refs(V):
    two = np.array([len(np.unique(V[:, j])) <= 2 for j in range(V.shape[1])])
    B = np.where(two[None, :], np.where(V > V.min(0), 1.0, -1.0), np.where(V > np.median(V, 0), 1.0, -1.0))
    sd = V.std(0); Z = (V - V.mean(0)) / np.where(sd > 0, sd, 1)
    ds = dawid_skene(B)
    out = {"D&S median": ds, "Mean z": Z.mean(1), "D&S -> logistic": FUSE()._predict(minmax_to_unit_interval(V), ds)}
    if len(np.unique(V)) > 2:
        out["D&S 5 bins"] = dawid_skene_ordinal(V, 5)
    Vs = minmax_to_unit_interval(V)
    cl = cluster_verifiers(Vs, 0.5).collapse(Vs)
    # a cluster that is one binary column keeps its two values; otherwise split at the median
    twoc = np.array([len(np.unique(cl[:, j])) <= 2 for j in range(cl.shape[1])])
    Bc = np.where(twoc[None, :], np.where(cl > cl.min(0), 1.0, -1.0), np.where(cl > np.median(cl, 0), 1.0, -1.0))
    dsc = dawid_skene(Bc)
    out["D&S median, clustered@0.5"] = dsc
    out["D&S clustered -> logistic"] = FUSE()._predict(Vs, dsc)
    out["Mean z, clustered@0.5"] = (lambda Z: Z.mean(1))((cl - cl.mean(0)) / np.where(cl.std(0) > 0, cl.std(0), 1))
    return out


if __name__ == "__main__":
    SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    res = {}
    for name, c in SCEN.items():
        t = time.time(); r = {}
        for s in range(SEEDS):
            V, y = simulate(np.random.default_rng(s), **c)
            for k, sc in refs(V).items():
                r.setdefault(k, []).append(auc(y, sc))
            for k, kw in VARIANTS.items():
                r.setdefault("FUSE " + k, []).append(auc(y, FUSE(**kw).fit_scores(V).scores))
        res[name] = {k: list(map(float, v)) for k, v in r.items()}
        print(f"## {name} ({time.time()-t:.0f}s)", flush=True)
        for k, v in r.items():
            print(f"  {k:32s} {100*np.mean(v):6.1f}", flush=True)
    json.dump(res, open(R + "/results/tci_fix_sim.json", "w"))
