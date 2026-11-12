"""Collapse raters who share errors, found label-free from residual correlation, then aggregate.

    python scripts/run_residual_blocks.py sim    # writes results/residual_blocks_sim.json
    python scripts/run_residual_blocks.py real   # writes results/residual_blocks_real.json (VIPER needs rdata)

Detector: fuse.clustering.residual_groups (least-trimmed-squares rank-one fit of the Spearman matrix,
average-linkage clusters on the partial correlation left after it) at thresholds 0.2, 0.3, 0.4. Each
cluster becomes one column (mean z-score). Aggregators on the collapsed columns: Dawid-Skene -> the
FUSE logistic step, mean z-score, FUSE with median thresholds and FUSE with the paper's search; the
logistic step is always fitted on all original columns. See docs/residual_blocks.md.

Baseline detector: Jaffe et al. (AISTATS 2016), fuse.clustering.jaffe_groups, on the covariance of
the binarized columns as in the paper ('jaffe') and on the Spearman matrix ('jaffe-spearman'). Its
groups go through the same collapse and aggregators, and through a two-stage Dawid-Skene (D&S inside
each group, then D&S on the group labels), a stand-in for their L-SML.
"""
import sys, os, json, time, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
R = str(Path(__file__).resolve().parents[1]); sys.path.insert(0, R + "/src"); sys.path.insert(0, R + "/scripts")
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score as auc
from fuse import FUSE
from fuse.dices import dawid_skene
from fuse.preprocess import minmax_to_unit_interval
from fuse.clustering import collapse_groups as collapse, jaffe_groups, residual_groups, spearman_matrix
from sim_regime_map import simulate, BASE

THR = (0.2, 0.3, 0.4)


def binarize(V):
    two = np.array([len(np.unique(V[:, j])) <= 2 for j in range(V.shape[1])])
    return np.where(two[None, :], np.where(V > V.min(0), 1.0, -1.0), np.where(V > np.median(V, 0), 1.0, -1.0))


def zmean(V):
    sd = V.std(0); return ((V - V.mean(0)) / np.where(sd > 0, sd, 1)).mean(1)


def two_stage_ds(B, groups):
    """D&S inside each group gives a latent label per group; D&S on those labels gives the posterior."""
    lat = np.column_stack([(B[:, g[0]] + 1) / 2 if len(g) == 1 else dawid_skene(B[:, g]) for g in groups])
    return dawid_skene(np.where(lat > 0.5, 1.0, -1.0))


def jaffe_methods(V, Vs, out):
    B = binarize(V)
    for tag, Rm in [("jaffe", np.cov(B.T)), ("jaffe-spearman", spearman_matrix(V))]:
        g, K, _, _ = jaffe_groups(Rm)
        Vc = collapse(V, g)
        out[f"D&S -> logistic | {tag}"] = FUSE()._predict(Vs, dawid_skene(binarize(Vc)))
        out[f"Mean z | {tag}"] = zmean(Vc)
        if Vc.shape[1] >= 3:
            out[f"FUSE median | {tag}"] = FUSE()._predict(Vs, FUSE(predictor="posterior", n_sweeps=0).fit_scores(Vc).scores)
        ts = two_stage_ds(B, g)
        out[f"Two-stage D&S | {tag}"] = ts
        out[f"Two-stage D&S -> logistic | {tag}"] = FUSE()._predict(Vs, ts)
        out[f"n columns | {tag}"] = len(g)


def methods(V):
    Vs = minmax_to_unit_interval(V)
    out = {}
    jaffe_methods(V, Vs, out)
    for t in (None,) + THR:
        if t is None:
            Vc, tag, ng = V, "none", V.shape[1]
        else:
            g, _, _ = residual_groups(V, t); Vc = collapse(V, g); tag = f"resid@{t}"; ng = len(g)
        ds = dawid_skene(binarize(Vc))
        out[f"D&S -> logistic | {tag}"] = FUSE()._predict(Vs, ds)
        out[f"Mean z | {tag}"] = zmean(Vc)
        if Vc.shape[1] >= 3:
            for lab, kw in [("FUSE median", dict(n_sweeps=0)), ("FUSE paper", {})]:
                post = FUSE(predictor="posterior", **kw).fit_scores(Vc).scores
                out[f"{lab} | {tag}"] = FUSE()._predict(Vs, post) if Vc.shape[1] < V.shape[1] else FUSE(**kw).fit_scores(V).scores
        out[f"n columns | {tag}"] = ng
    return out


def evaluate(name, mats, y):
    t = time.time(); r = {}
    for V in mats:
        for k, s in methods(V).items():
            r.setdefault(k, []).append(float(s) if np.isscalar(s) else auc(y, s))
    res[name] = r
    print(f"## {name} ({time.time()-t:.0f}s)", flush=True)
    for k, v in r.items():
        print(f"  {k:32s} {np.mean(v) if k.startswith('n col') else 100*np.mean(v):6.1f}", flush=True)


res = {}
part = sys.argv[1] if len(sys.argv) > 1 else "sim"
if part == "sim":
    SC = {"base": dict(BASE), "5 raters": dict(BASE, m=5), "hard: d=0.3": dict(BASE, d_mean=0.3),
          "high quality spread": dict(BASE, spread="high"),
          "20 near-chance + 2 experts": dict(BASE, m=20, d_mean=0.1, spread="none", n_experts=2),
          "20% weak corr block": dict(BASE, corr_frac=0.2, corr_rho=0.6, corr_d=0.2),
          "40% weak corr block": dict(BASE, corr_frac=0.4, corr_rho=0.6, corr_d=0.2),
          "40% weak corr block, rho 0.3": dict(BASE, corr_frac=0.4, corr_rho=0.3, corr_d=0.2),
          "40% normal-quality corr block": dict(BASE, corr_frac=0.4, corr_rho=0.6),
          "binary votes": dict(BASE, binary=True)}
    for name, c in SC.items():
        mats, ys = zip(*[simulate(np.random.default_rng(s), **c) for s in range(6)])
        t = time.time(); r = {}
        for V, y in zip(mats, ys):
            for k, s in methods(V).items():
                r.setdefault(k, []).append(float(s) if np.isscalar(s) else auc(y, s))
        res[name] = r
        print(f"## {name} ({time.time()-t:.0f}s)", flush=True)
        for k, v in r.items():
            print(f"  {k:32s} {np.mean(v) if k.startswith('n col') else 100*np.mean(v):6.1f}", flush=True)
else:
    from fuse import crowd, dices, viper, hard_crowds

    def llm(model, prompt):
        return pd.read_csv(f"{R}/results/llm_scores/rte_{model}_{prompt}.csv").sort_values("item_id").logit_yes.to_numpy()

    def slots(H, rng):
        out = np.empty((len(H), 10))
        for i in range(len(H)):
            v = H[i, ~np.isnan(H[i])]; out[i] = rng.permutation(v)[:10]
        return out

    H, gold = crowd.load("RTE"); y = gold.astype(int)
    TINY = ["Qwen2.5-0.5B-Instruct-4bit", "Qwen3-0.6B-4bit"]
    SMALL = TINY + ["Qwen2.5-1.5B-Instruct-4bit", "Qwen3-1.7B-4bit"]
    STRONG = ["Meta-Llama-3.1-8B-Instruct-4bit", "Qwen3-8B-4bit", "gemma-3-12b-it-qat-4bit"]
    X = {m: np.column_stack([llm(m, p) for p in ("short", "guide")]) for m in SMALL + STRONG}
    for name, k, models in [("RTE 10 humans", 10, []), ("RTE 10 humans + 4 tiny", 10, TINY),
                            ("RTE 10 humans + 8 small Qwen", 10, SMALL), ("RTE 5 humans + 4 tiny", 5, TINY),
                            ("RTE 3 humans + 4 tiny", 3, TINY), ("RTE 10 humans + 6 strong LLM", 10, STRONG),
                            ("RTE 3 humans + 6 strong LLM", 3, STRONG)]:
        L = np.column_stack([X[m] for m in models]) if models else np.empty((len(y), 0))
        evaluate(name, [np.c_[slots(H, np.random.default_rng(d))[:, :k], L] for d in range(10)], y)
    evaluate("RTE 8 small Qwen only", [np.column_stack([X[m] for m in SMALL])], y)
    reads, vg = viper.load()
    for mod in ["FFDM", "SFM"]:
        r = reads[reads.modalityID == mod]
        blocks = viper.dense_blocks(r)
        for lo in (90, 200):
            n, (Rs, cs) = max(((n, b) for n, b in blocks.items() if n >= lo), key=lambda t: len(t[1][0]))
            S, Bv, yv = viper.build(r[r.readerID.isin(Rs) & r.caseID.isin(cs)], vg)
            evaluate(f"VIPER {mod} {len(Rs)} readers x {n} cases", [S], np.asarray(yv).astype(int))
    df = dices.load("350")
    for label, view, groups in [("granular", "granular", ("rater_race", "rater_gender")),
                                ("race x gender groups", "groups", ("rater_race", "rater_gender")),
                                ("race x gender x age groups", "groups", ("rater_race", "rater_gender", "rater_age"))]:
        m = dices.build(df, view, groups=groups)
        V = m.V[:, ~np.isnan(m.V).any(0)] if np.isnan(m.V).any() else m.V
        evaluate(f"DICES-350 {label}", [V], np.asarray(m.gold).astype(int))
    GROUPS = {"deepfake": lambda L: L.sex + "|" + L.age + "|" + np.where(L.ethnicity == "White", "White", "nonWhite"),
              "factcheck": lambda L: L.polpref.astype("Int64").astype(str) + "|" + L.sex}
    for name in ("deepfake", "factcheck"):
        L, g = hard_crowds.load(name)
        L = L.assign(group=GROUPS[name](L)); L = L[~L.group.str.contains("nan|<NA>")]
        G, yg, cols = hard_crowds.matrix(L, g, by="group")
        evaluate(f"{name} demographic groups", [G[:, ~np.isnan(G).any(0)]], np.asarray(yg).astype(int))
json.dump(res, open(f"{R}/results/residual_blocks_{part}.json", "w"))
