"""Groups of DICES-350 raters who share errors: do they exist, and does FUSE gain when one is a minority?

    python scripts/dices_error_blocks.py groups    # steps 1-4, writes results/dices_error_blocks.json
    python scripts/dices_error_blocks.py panels    # step 5 (needs the groups), adds to the same JSON
    python scripts/dices_error_blocks.py report    # writes results/DICES_ERROR_BLOCKS.md from the JSON

Step 1. Shared errors, not shared agreement. Rating Yes = 1, Unsure = 0, No = -1. A rater's residual on a
conversation is the rating minus that rater's mean rating over conversations with the same expert label
(class-conditional centring). Residuals are correlated between raters, pooled over both classes. A high
residual correlation means two raters move together beyond what the label explains: conditional dependence.
Groups and every choice are made on the DEV half (174 conversations, the fixed split of
scripts/tune_dices350.py); the TEST half (176) is only used to check the groups and to score the panels.
Rater 61 answers No on all 350 conversations (zero variance) and is left out everywhere (122 raters).

Step 2. Eigenvalues of the residual correlation matrix against the Marchenko-Pastur edge
lambda+ = (1 + sqrt(N/T))^2 and two nulls: (a) each rater's residuals shuffled within class (the null asked
for: no dependence at all), (b) a one-factor Gaussian null with the fitted global mode (everyone shares one
error factor, nothing else), which is the right null for group modes once a global mode is present.

Step 3. Four clusterings of the dev residual correlation matrix:
* MacMahon & Garlaschelli (2015, PRX 5, 021006): keep only the group modes (eigenvalues above the null, minus
  the global mode), and maximise the modularity Q = sum over same-community pairs of the filtered matrix by
  Louvain-style local moves from many starts. The number of communities is chosen by the optimisation.
* Spectral clustering on the positive part of the residual correlation (sklearn, precomputed affinity).
* Average-linkage hierarchical clustering on 1 - r, cut at k clusters.
* Greedy rule: seed with the most-correlated remaining pair, add the rater that raises the group's mean
  within-group correlation most, stop when none does; repeat on the rest (5 groups; the rest stays unassigned).

Step 4. Bootstrap stability (100 stratified resamples of dev conversations), a shuffled-residual null for the
within-group correlation, the held-out check (within-group residual correlation of the dev-built groups on
the test half, against random groups of the same sizes), and agreement with demographics.

Step 5. Panels of 10, 15, 20 raters where one dev-built group (the highest dev within-group correlation with
at least 8 raters, per clustering method) is 20 / 30 / 40% of the panel and the rest are drawn from raters
outside it; 100 draws per setting; size-matched random panels as control. Every method is fitted on the test
half only (176 conversations, no labels) and scored there: fitting on the dev half would include the very
conversations the groups were selected on and overstate their shared errors. Methods: majority vote,
three-answer Dawid-Skene, FUSE (repo defaults), mean z-score, and three-answer Dawid-Skene after the
label-free detector (fuse.clustering.residual_groups at 0.4; each detected group collapsed to one
three-answer column: Yes if the members' mean rating > 1/3, No if < -1/3, else Unsure). Accuracy at each
method's own rule and at a shared cut-off (every method calls the same number unsafe; scripts/panel_eval.py),
AUC, paired bootstrap 95% intervals over conversations.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import panel_eval as PE  # noqa: E402
import tune_dices350 as T  # noqa: E402
from fuse import FUSE, dices  # noqa: E402
from fuse.clustering import residual_groups  # noqa: E402

OUT_JSON = ROOT / "results" / "dices_error_blocks.json"
OUT_MD = ROOT / "results" / "DICES_ERROR_BLOCKS.md"
N_BOOT = 100
N_NULL = 100
N_DRAWS = 100
SIZES = (10, 15, 20)
FRACS = (0.2, 0.3, 0.4)
MIN_BLOCK = 8  # 40% of a 20-rater panel
GREEDY_GROUPS = 5
DEMOS = ["rater_gender", "rater_race", "rater_education", "rater_age"]
METHODS = ["Majority vote", "Dawid–Skene, 3 answers", "FUSE", "Mean z-score", "Dawid–Skene after detector"]
CUTS = ["natural", "shared (DS-3 est.)", "shared (FUSE est.)"]


# ---- data and residual correlation -----------------------------------------------------------


def load():
    d = T.Data()
    m = dices.build(d.df, view="overall")
    assert (m.items == d.items).all()
    V, y = d.raters_V, d.gold
    keep = V.std(0) > 0
    raters = np.array(m.verifiers)[keep]
    demo = d.df.drop_duplicates("rater_id").set_index("rater_id")[DEMOS].astype(str)
    demo.index = demo.index.astype(str)
    return V[:, keep], y, d.dev, raters, demo.loc[raters]


def residuals(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    R = X.astype(float).copy()
    for c in (True, False):
        R[y == c] -= X[y == c].mean(0)
    return R


def corr(R: np.ndarray) -> np.ndarray:
    """Pearson correlation of columns; a column with no variance correlates NaN (callers decide)."""
    Z = R - R.mean(0)
    sd = np.sqrt((Z ** 2).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        C = (Z.T @ Z) / np.outer(sd, sd)
    np.fill_diagonal(C, 1.0)
    return C


def corr0(R):
    C = corr(R)
    return np.nan_to_num(C, nan=0.0)


def shuffle_within_class(R, y, rng):
    S = R.copy()
    for c in (True, False):
        idx = np.flatnonzero(y == c)
        order = np.argsort(rng.random((len(idx), R.shape[1])), axis=0)
        S[idx] = np.take_along_axis(R[idx], order, axis=0)
    return S


def mp_edge(N, T_):
    return (1 + np.sqrt(N / T_)) ** 2


def offdiag(C):
    return C[np.triu_indices(len(C), 1)]


def pct(a, qs=(1, 5, 25, 50, 75, 95, 99)):
    a = a[~np.isnan(a)]
    return {str(q): float(np.percentile(a, q)) for q in qs}


# ---- step 2: eigenvalues and nulls --------------------------------------------------------------


def eig_desc(C):
    w, v = np.linalg.eigh(C)
    return w[::-1], v[:, ::-1]


def subspace_cos(A, B, k):
    """Cosines of the principal angles between the top-k eigenspaces of A and B (1 = same direction)."""
    _, va = eig_desc(A)
    _, vb = eig_desc(B)
    return np.linalg.svd(va[:, :k].T @ vb[:, :k], compute_uv=False).tolist()


def one_factor_null(C, T_, n, rng, top=10):
    w, v = eig_desc(C)
    u = np.clip(np.sqrt(w[0]) * v[:, 0] * np.sign(v[:, 0].sum()), -0.99, 0.99)
    L = np.column_stack([u, np.diag(np.sqrt(1 - u ** 2))])
    out = []
    for _ in range(n):
        X = rng.standard_normal((T_, L.shape[1])) @ L.T
        out.append(eig_desc(corr0(X))[0][:top])
    return np.array(out)


def shuffle_null_eigs(R, y, n, rng, top=10):
    return np.array([eig_desc(corr0(shuffle_within_class(R, y, rng)))[0][:top] for _ in range(n)])


# ---- step 3: clusterings --------------------------------------------------------------------


def mg_filtered(C, n_modes):
    """MacMahon-Garlaschelli 'group' component: the eigenmodes 2..n_modes+1 (global mode and bulk removed)."""
    w, v = eig_desc(C)
    idx = np.arange(1, 1 + n_modes)
    return (v[:, idx] * w[idx]) @ v[:, idx].T


def modularity(Cg, lab, norm):
    same = lab[:, None] == lab[None, :]
    np.fill_diagonal(same, False)
    return float((Cg * same).sum() / norm)


def local_moves(Cg, lab, rng, max_sweeps=100):
    """Louvain phase 1 on a signed matrix: move each node to the community (or a new one) that raises
    sum_{same community, i != j} Cg_ij most."""
    _, lab = np.unique(lab, return_inverse=True)
    A = Cg.copy()
    np.fill_diagonal(A, 0.0)
    N = len(A)
    for _ in range(max_sweeps):
        moved = False
        for i in rng.permutation(N):
            gain = np.bincount(lab, weights=A[i], minlength=N)  # labels stay below N
            own = gain[lab[i]]
            occupied = np.bincount(lab, minlength=N) > 0
            gain[~occupied] = -np.inf
            best = int(np.argmax(gain))
            if gain[best] <= 0:
                if own < 0:  # better alone: move to an empty label
                    lab[i] = int(np.flatnonzero(~occupied)[0])
                    moved = True
                continue
            if best != lab[i] and gain[best] > own + 1e-12:
                lab[i] = best
                moved = True
        if not moved:
            break
    _, lab = np.unique(lab, return_inverse=True)
    return lab


def mg_cluster(C, n_modes, seed=0, restarts=40):
    if n_modes < 1:
        return np.zeros(len(C), int)
    Cg = mg_filtered(C, n_modes)
    norm = np.abs(C).sum()
    rng = np.random.default_rng(seed)
    best, best_q = None, -np.inf
    w, v = eig_desc(Cg)
    starts = [np.arange(len(C))]  # singletons
    # sign pattern of the group eigenvectors (spectral start)
    s = (v[:, :n_modes] > 0).astype(int) @ (2 ** np.arange(n_modes))
    starts.append(s)
    for r in range(restarts):
        starts.append(rng.integers(0, 2 + r % 5, len(C)) if r % 2 else np.arange(len(C)))
    for st in starts:
        lab = local_moves(Cg, np.asarray(st), rng)
        q = modularity(Cg, lab, norm)
        if q > best_q + 1e-12:
            best, best_q = lab, q
    return canon(best)


def canon(lab):
    """Relabel by decreasing size so labels are comparable across runs (0 = largest)."""
    lab = np.asarray(lab)
    u, c = np.unique(lab, return_counts=True)
    order = u[np.argsort(-c, kind="stable")]
    m = {g: i for i, g in enumerate(order)}
    return np.array([m[g] for g in lab])


def spectral_cluster(C, k, seed=0):
    A = np.clip(C, 0, None)
    np.fill_diagonal(A, 0.0)
    sc = SpectralClustering(n_clusters=k, affinity="precomputed", random_state=seed, n_init=10)
    return canon(sc.fit_predict(A))


def hier_cluster(C, k):
    D = np.clip(1 - C, 0, None)
    np.fill_diagonal(D, 0.0)
    return canon(fcluster(linkage(squareform(D, checks=False), "average"), t=k, criterion="maxclust"))


def greedy_cluster(C, n_groups=GREEDY_GROUPS, min_size=2):
    """Yaniv's rule. Unassigned raters get label -1. With min_size > 2 the best rater is added even when it
    lowers the mean until the group has min_size members; after that the stopping rule applies."""
    N = len(C)
    lab = -np.ones(N, int)
    rem = list(range(N))
    for g in range(n_groups):
        if len(rem) < 2:
            break
        sub = C[np.ix_(rem, rem)].copy()
        np.fill_diagonal(sub, -np.inf)
        i, j = np.unravel_index(np.argmax(sub), sub.shape)
        G = [rem[i], rem[j]]
        tot, npairs = C[G[0], G[1]], 1
        while True:
            cand = [r for r in rem if r not in G]
            if not cand:
                break
            add = C[np.ix_(cand, G)].sum(1)
            new_mean = (tot + add) / (npairs + len(G))
            b = int(np.argmax(new_mean))
            if new_mean[b] <= tot / npairs and len(G) >= min_size:
                break
            tot += add[b]
            npairs += len(G)
            G.append(cand[b])
        lab[G] = g
        rem = [r for r in rem if r not in G]
    return lab


def run_method(name, C, k, n_modes, seed=0):
    if name == "MacMahon–Garlaschelli":
        return mg_cluster(C, n_modes["mp"], seed)
    if name == "MacMahon–Garlaschelli, modes vs one-factor null":
        return mg_cluster(C, n_modes["one_factor"], seed)
    if name == "Greedy, grown to 8":
        return greedy_cluster(C, min_size=MIN_BLOCK)
    if name == "Spectral (positive part)":
        return spectral_cluster(C, k, seed)
    if name == "Hierarchical (average, 1 − r)":
        return hier_cluster(C, k)
    if name == "Greedy (Yaniv's rule)":
        return greedy_cluster(C)
    raise ValueError(name)


CLUSTERERS = ["MacMahon–Garlaschelli", "Spectral (positive part)", "Hierarchical (average, 1 − r)", "Greedy (Yaniv's rule)",
              "MacMahon–Garlaschelli, modes vs one-factor null", "Greedy, grown to 8"]


def groups_of(lab, min_size=2):
    return [np.flatnonzero(lab == g) for g in np.unique(lab) if g >= 0 and (lab == g).sum() >= min_size]


def within_between(C, lab):
    """Pooled mean correlation over pairs in the same group (groups of 2+), over pairs in different groups
    (unassigned raters, label -1, count as outside every group), and the per-group means."""
    N = len(C)
    iu = np.triu_indices(N, 1)
    a, b = lab[iu[0]], lab[iu[1]]
    r = C[iu]
    same = (a == b) & (a >= 0)
    diff = a != b
    per = {}
    for g in np.unique(lab):
        if g < 0:
            continue
        m = np.flatnonzero(lab == g)
        if len(m) >= 2:
            per[int(g)] = {"size": int(len(m)), "within": float(np.nanmean(offdiag(C[np.ix_(m, m)])))}
        else:
            per[int(g)] = {"size": 1, "within": None}
    return {"within": float(np.nanmean(r[same])) if same.any() else None,
            "between": float(np.nanmean(r[diff])) if diff.any() else None,
            "all_pairs": float(np.nanmean(r)), "per_group": per}


def random_group_null(C, lab, rng, n=2000):
    """Pooled within-group mean for random relabellings of the same group sizes."""
    vals = []
    for _ in range(n):
        vals.append(within_between(C, rng.permutation(lab))["within"])
    return np.array(vals)


# ---- steps 1-4 ----------------------------------------------------------------------------------


def step_groups():
    t0 = time.time()
    rng = np.random.default_rng(2026)
    V, y, dev, raters, demo = load()
    N = V.shape[1]
    Xd, yd, Xt, yt = V[dev], y[dev], V[~dev], y[~dev]
    Rd, Rt = residuals(Xd, yd), residuals(Xt, yt)
    Cd, Ct = corr0(Rd), corr(Rt)
    Pd = corr0(Xd)
    out = {"n_raters": int(N), "dropped": ["rater 61 (index in the 123-rater matrix; No on all 350)"],
           "T_dev": int(dev.sum()), "T_test": int((~dev).sum())}

    # step 1
    cd, pdv, ctv = offdiag(Cd), offdiag(Pd), offdiag(Ct)
    out["step1"] = {
        "residual_dev": pct(cd) | {"mean": float(cd.mean()), "share_gt_0.3": float((cd > 0.3).mean()), "share_lt_0": float((cd < 0).mean())},
        "plain_dev": pct(pdv) | {"mean": float(pdv.mean()), "share_gt_0.3": float((pdv > 0.3).mean()), "share_lt_0": float((pdv < 0).mean())},
        "residual_test": pct(ctv) | {"mean": float(np.nanmean(ctv))},
        "plain_minus_residual_mean": float(pdv.mean() - cd.mean()),
        "corr_plain_vs_residual_pairs": float(np.corrcoef(pdv, cd)[0, 1]),
        "residual_dev_vs_test_pairs": float(np.corrcoef(cd[~np.isnan(ctv)], ctv[~np.isnan(ctv)])[0, 1]),
        "top_pairs_dev": [{"a": raters[i], "b": raters[j], "r_dev": float(Cd[i, j]), "r_test": float(Ct[i, j])}
                          for i, j in zip(*np.unravel_index(np.argsort(-np.triu(Cd, 1), axis=None)[:10], Cd.shape))],
    }
    print("step 1 done", flush=True)

    # step 2
    T_ = int(dev.sum())
    wd, vd = eig_desc(Cd)
    wt, _ = eig_desc(np.nan_to_num(Ct))
    lam = mp_edge(N, T_)
    lam_adj = (1 - wd[0] / N) * lam  # bulk variance left after the global mode (Laloux et al. 1999)
    sh = shuffle_null_eigs(Rd, yd, N_NULL, rng)
    of = one_factor_null(Cd, T_, N_NULL, rng)
    of95 = np.percentile(of, 95, axis=0)
    n_modes = 0
    for j in range(1, 10):
        if wd[j] > of95[j]:
            n_modes += 1
        else:
            break
    out["step2"] = {
        "eig_dev_top10": wd[:10].tolist(), "eig_test_top10": wt[:10].tolist(), "plain_eig_dev_top5": eig_desc(Pd)[0][:5].tolist(),
        "mp_edge": float(lam), "mp_edge_test": float(mp_edge(N, int((~dev).sum()))),
        "n_above_mp_dev": int((wd > lam).sum()), "n_above_mp_test": int((wt > mp_edge(N, int((~dev).sum()))).sum()),
        "mp_edge_adjusted_for_global": float(lam_adj), "n_above_adjusted_dev": int((wd > lam_adj).sum()),
        "global_mode_share_of_variance": float(wd[0] / N),
        "global_mode_loadings": pct(vd[:, 0] * np.sign(vd[:, 0].sum()) * np.sqrt(wd[0])),
        "shuffle_null_top": {"mean": sh.mean(0)[:5].tolist(), "p95": np.percentile(sh, 95, axis=0)[:5].tolist(), "max": sh.max(0)[:5].tolist()},
        "one_factor_null": {"mean": of.mean(0)[:10].tolist(), "p95": of95[:10].tolist()},
        "n_eig_above_shuffle_null_max": int((wd > sh[:, 0].max()).sum()),
        "n_group_modes_one_factor": int(n_modes),
        "n_group_modes_mp": int(max(0, (wd > lam).sum() - 1)),
        "dev_test_subspace_cos": subspace_cos(Cd, np.nan_to_num(Ct), 3),
    }
    modes = {"mp": out["step2"]["n_group_modes_mp"], "one_factor": n_modes}
    print(f"step 2: eig {wd[:5].round(2)}, edge {lam:.2f}, group modes {modes} ({time.time() - t0:.0f}s)", flush=True)

    # step 3
    bacc = np.array([PE.balanced_acc(Xd[:, j] > 0, yd) for j in range(N)])
    yes, uns = (Xd > 0).mean(0), (Xd == 0).mean(0)
    gload = vd[:, 0] * np.sign(vd[:, 0].sum()) * np.sqrt(wd[0])
    out["rater_profile_all"] = {"bal_acc_dev": float(bacc.mean()), "yes_rate": float(yes.mean()), "unsure_rate": float(uns.mean()),
                                "global_loading": float(gload.mean())}
    out["mode_vs_profile"] = {f"mode {j + 1}": {"bal_acc": float(np.corrcoef(vd[:, j], bacc)[0, 1]), "yes_rate": float(np.corrcoef(vd[:, j], yes)[0, 1]),
                                                "unsure_rate": float(np.corrcoef(vd[:, j], uns)[0, 1])} for j in range(4)}
    labs = {}
    labs["MacMahon–Garlaschelli"] = mg_cluster(Cd, modes["mp"])
    labs["MacMahon–Garlaschelli, modes vs one-factor null"] = mg_cluster(Cd, modes["one_factor"])
    labs["Greedy, grown to 8"] = greedy_cluster(Cd, min_size=MIN_BLOCK)
    k_mg = len(groups_of(labs["MacMahon–Garlaschelli"], 1))
    k = k_mg if 4 <= k_mg <= 5 else (4 if k_mg < 4 else k_mg)
    out["k_rule"] = (f"MacMahon–Garlaschelli finds {k_mg} communities from {modes['mp']} group mode(s); spectral and hierarchical "
                     f"are cut at k = {k} (the brief's 4–5 when the modularity gives fewer), with k = 4 and 5 as a sensitivity check")
    labs["Spectral (positive part)"] = spectral_cluster(Cd, k)
    labs["Hierarchical (average, 1 − r)"] = hier_cluster(Cd, k)
    labs["Greedy (Yaniv's rule)"] = greedy_cluster(Cd)
    sens = {}
    for kk in (4, 5):
        for nm, f in [("Spectral (positive part)", spectral_cluster), ("Hierarchical (average, 1 − r)", hier_cluster)]:
            lab = f(Cd, kk)
            wb = within_between(Cd, lab)
            sens[f"{nm}, k={kk}"] = {"sizes": sorted([int(s) for s in np.bincount(lab)], reverse=True),
                                     "within_dev": wb["within"], "between_dev": wb["between"],
                                     "within_test": within_between(Ct, lab)["within"]}
    out["step3_sensitivity"] = sens

    res = {}
    for nm in CLUSTERERS:
        lab = labs[nm]
        t1 = time.time()
        wb_d, wb_t = within_between(Cd, lab), within_between(Ct, lab)
        # step 4a: bootstrap stability
        aris, co = [], np.zeros((N, N))
        pos, neg = np.flatnonzero(yd), np.flatnonzero(~yd)
        for b in range(N_BOOT):
            idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
            Cb = corr0(residuals(Xd[idx], yd[idx]))
            lb = run_method(nm, Cb, k, modes, seed=b)
            aris.append(ari(lab, lb))
            co += (lb[:, None] == lb[None, :]) & (lb[:, None] >= 0)
        co /= N_BOOT
        iu = np.triu_indices(N, 1)
        same = (lab[iu[0]] == lab[iu[1]]) & (lab[iu[0]] >= 0)
        # step 4b: shuffled-residual null for within-group correlation (same method, same k / modes)
        null_w, null_c = [], []
        for s in range(N_NULL):
            Cs = corr0(shuffle_within_class(Rd, yd, rng))
            ls = run_method(nm, Cs, k, modes, seed=s)
            wbs = within_between(Cs, ls)
            if wbs["within"] is not None:
                null_w.append(wbs["within"])
                null_c.append(wbs["within"] - (wbs["between"] if wbs["between"] is not None else wbs["all_pairs"]))
        # step 4c: held-out, against random groups of the same sizes on the test half
        rnd = random_group_null(Ct, lab, rng)
        contrast_d = wb_d["within"] - wb_d["between"] if wb_d["between"] is not None else None
        contrast_t = wb_t["within"] - wb_t["between"] if wb_t["between"] is not None else None
        # step 4d: demographics
        dem = {}
        for col in DEMOS:
            dv = demo[col].to_numpy()
            ct = {}
            for g in np.unique(lab):
                vals, cnt = np.unique(dv[lab == g], return_counts=True)
                ct[str(int(g))] = {str(a): int(c) for a, c in zip(vals, cnt)}
            dem[col] = {"ari": float(ari(dv, lab)), "crosstab": ct}
        prof = {}
        for g in np.unique(lab):
            mm = lab == g
            prof[str(int(g))] = {"bal_acc_dev": float(bacc[mm].mean()), "yes_rate": float(yes[mm].mean()), "unsure_rate": float(uns[mm].mean()),
                                 "global_loading": float(gload[mm].mean())}
        res[nm] = {
            "profile": prof,
            "labels": {r: int(g) for r, g in zip(raters, lab)},
            "sizes": sorted([int((lab == g).sum()) for g in np.unique(lab) if g >= 0], reverse=True),
            "n_unassigned": int((lab < 0).sum()),
            "dev": wb_d, "test": wb_t, "contrast_dev": contrast_d, "contrast_test": contrast_t,
            "boot_ari_mean": float(np.mean(aris)), "boot_ari_p5": float(np.percentile(aris, 5)), "boot_ari_p95": float(np.percentile(aris, 95)),
            "coassign_same_pairs": float(co[iu][same].mean()) if same.any() else None,
            "coassign_diff_pairs": float(co[iu][~same].mean()),
            "per_group_coassign": {str(int(g)): float(offdiag(co[np.ix_(m, m)]).mean()) for g in np.unique(lab) if g >= 0
                                   for m in [np.flatnonzero(lab == g)] if len(m) >= 2},
            "null_within_mean": float(np.mean(null_w)) if null_w else None, "null_within_p95": float(np.percentile(null_w, 95)) if null_w else None,
            "null_contrast_mean": float(np.mean(null_c)) if null_c else None, "null_contrast_p95": float(np.percentile(null_c, 95)) if null_c else None,
            "test_random_groups_within_mean": float(rnd.mean()), "test_random_groups_within_p95": float(np.percentile(rnd, 95)),
            "test_within_p_value": float((1 + (rnd >= wb_t["within"]).sum()) / (1 + len(rnd))),
            "demographics": dem,
        }
        print(f"step 3-4 {nm}: sizes {res[nm]['sizes']} within dev {wb_d['within']:.3f} test {wb_t['within']:.3f} "
              f"(random {rnd.mean():.3f}) ARI {np.mean(aris):.2f} ({time.time() - t1:.0f}s)", flush=True)
    out["step3_4"] = res
    out["k"] = int(k)
    data = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {}
    data["groups"] = out
    OUT_JSON.write_text(json.dumps(data, indent=1))
    print(f"groups written ({time.time() - t0:.0f}s)")


# ---- step 5: panels -----------------------------------------------------------------------------

_V = _Y = None


def _init(V, y):
    global _V, _Y
    _V, _Y = V, y


def collapse3(V, groups):
    cols = []
    for g in groups:
        if len(g) == 1:
            cols.append(V[:, g[0]])
        else:
            m = V[:, g].mean(1)
            cols.append(np.where(m > 1 / 3, 1.0, np.where(m < -1 / 3, -1.0, 0.0)))
    return np.column_stack(cols)


def zmean(V):
    sd = V.std(0)
    return ((V - V.mean(0)) / np.where(sd > 0, sd, 1)).mean(1)


def eval_panel(cols, planted):
    V, y = _V[:, cols], _Y
    n = len(y)
    share = ((V + 1) / 2).mean(1)
    q3 = dices.dawid_skene_3class(V)
    f = FUSE().fit_scores(V, rescale=False)
    groups, _, _ = residual_groups(V, 0.4)
    qd = dices.dawid_skene_3class(collapse3(V, groups))
    z = zmean(V)
    est_f = None if f.estimate is None else float(f.estimate.prevalence)
    est_d = float(q3.mean())
    kd = int(round(est_d * n))
    kf = int(round((est_f if est_f is not None else est_d) * n))
    scores = {"Majority vote": share, "Dawid–Skene, 3 answers": q3, "FUSE": f.scores, "Mean z-score": z, "Dawid–Skene after detector": qd}
    natural = {"Majority vote": share > 0.5, "Dawid–Skene, 3 answers": q3 > 0.5, "FUSE": f.scores > 0.5, "Mean z-score": z > 0,
               "Dawid–Skene after detector": qd > 0.5}
    merged = np.zeros(len(cols), bool)
    for g in groups:
        if len(g) > 1:
            merged[g] = True
    pl = np.isin(cols, planted)
    pos, neg = np.flatnonzero(y), np.flatnonzero(~y)
    out = {"hits": {}, "auc": {}, "called": {}, "G": {}, "est_f": est_f, "est_d": est_d, "fallback": f.fallback,
           "n_cols_detector": len(groups), "n_merged": int(merged.sum()), "tp": int((merged & pl).sum()), "n_planted": int(pl.sum()),
           "planted_one_group": bool(pl.sum() >= 2 and any(pl[g].all() and len(g) >= pl.sum() for g in groups)),
           "best_jaccard": float(max(((pl[g].sum()) / len(set(g) | set(np.flatnonzero(pl))) for g in groups), default=0.0)) if pl.any() else None}
    for m, s in scores.items():
        out["hits"][m] = {"natural": natural[m] == y, "shared (DS-3 est.)": PE.top_k(s, kd) == y, "shared (FUSE est.)": PE.top_k(s, kf) == y}
        out["called"][m] = 100 * float(natural[m].mean())
        out["auc"][m] = 100 * float(roc_auc_score(y, s))
        sp, sn = s[pos][:, None], s[neg][None, :]
        out["G"][m] = ((sp > sn) + 0.5 * (sp == sn)).astype(np.float32)
    return out


def one_setting(args):
    kind, size, frac, block, others, draw = args
    rng = np.random.default_rng([size, int(100 * frac), draw, zlib.crc32(kind.encode())])
    if kind == "random":
        cols = rng.choice(_V.shape[1], size, replace=False)
        planted = np.array([], int)
    else:
        nb = int(np.floor(frac * size + 0.5))
        b = rng.choice(block, nb, replace=False)
        o = rng.choice(others, size - nb, replace=False)
        cols = np.concatenate([b, o])
        planted = b
    r = eval_panel(cols, planted)
    r["cols"] = cols.tolist()
    return r


def boot_acc(rows_a, rows_b, m1, m2, cut, rng, B=5000):
    """(m1 - m2) accuracy, mean over panels; 95% paired bootstrap over conversations. With rows_b, the
    difference-in-differences (block panels minus random panels)."""
    def per_conv(rows):
        return np.mean([r["hits"][m1][cut].astype(float) - r["hits"][m2][cut] for r in rows], axis=0)
    d = per_conv(rows_a) - (per_conv(rows_b) if rows_b is not None else 0)
    idx = rng.integers(0, len(d), (B, len(d)))
    bs = 100 * d[idx].mean(1)
    return {"diff": float(100 * d.mean()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5))}


def boot_auc(rows_a, rows_b, m1, m2, rng, B=2000):
    """(m1 - m2) AUC, mean over panels; stratified 95% bootstrap over conversations (positives and negatives
    resampled separately) using each panel's pairwise win matrix."""
    G = np.mean([r["G"][m1] - r["G"][m2] for r in rows_a], axis=0)
    if rows_b is not None:
        G = G - np.mean([r["G"][m1] - r["G"][m2] for r in rows_b], axis=0)
    npos, nneg = G.shape
    Wp = np.stack([np.bincount(rng.integers(0, npos, npos), minlength=npos) for _ in range(B)]).astype(float)
    Wn = np.stack([np.bincount(rng.integers(0, nneg, nneg), minlength=nneg) for _ in range(B)]).astype(float)
    bs = 100 * ((Wp @ G) * Wn).sum(1) / (npos * nneg)
    return {"diff": float(100 * G.mean()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5))}


def summarize_rows(rows):
    s = {}
    for m in METHODS:
        s[m] = {"auc": float(np.mean([r["auc"][m] for r in rows])), "called": float(np.mean([r["called"][m] for r in rows]))}
        for c in CUTS:
            s[m][c] = float(np.mean([100 * r["hits"][m][c].mean() for r in rows]))
    return s


def step_panels(workers):
    t0 = time.time()
    data = json.loads(OUT_JSON.read_text())
    G = data["groups"]
    V, y, dev, raters, _ = load()
    Vt, yt = V[~dev], y[~dev]
    # one block per clustering method: its group with the highest DEV within-group correlation, with at least
    # MIN_BLOCK raters and enough raters outside it to fill 80% of the largest panel
    max_size = V.shape[1] - int(np.ceil((1 - min(FRACS)) * max(SIZES)))
    blocks = {}
    for nm in CLUSTERERS:
        res = G["step3_4"][nm]
        lab = np.array([res["labels"][r] for r in raters])
        cand = [(v["within"], int(g)) for g, v in res["dev"]["per_group"].items()
                if MIN_BLOCK <= v["size"] <= max_size and v["within"] is not None]
        if not cand:
            print(f"{nm}: no group with {MIN_BLOCK}–{max_size} raters, skipped", flush=True)
            continue
        w, g = max(cand)
        members = np.flatnonzero(lab == g)
        key = tuple(sorted(members.tolist()))
        if any(tuple(sorted(b["members"])) == key for b in blocks.values()):
            continue
        blocks[nm] = {"group": g, "members": members.tolist(), "size": len(members), "within_dev": w,
                      "within_test": res["test"]["per_group"][str(g)]["within"], "profile": res["profile"][str(g)]}
    # closest to the simulated regime where FUSE wins (a WEAK block): the grown-greedy group with the lowest dev accuracy
    res = G["step3_4"]["Greedy, grown to 8"]
    lab = np.array([res["labels"][r] for r in raters])
    g = min((v["bal_acc_dev"], int(k)) for k, v in res["profile"].items() if int(k) >= 0 and (lab == int(k)).sum() >= MIN_BLOCK)[1]
    members = np.flatnonzero(lab == g)
    if all(sorted(b["members"]) != members.tolist() for b in blocks.values()):
        blocks["Greedy, grown to 8, least accurate group"] = {
            "group": g, "members": members.tolist(), "size": len(members), "within_dev": res["dev"]["per_group"][str(g)]["within"],
            "within_test": res["test"]["per_group"][str(g)]["within"], "profile": res["profile"][str(g)]}
    jobs = []
    for size in SIZES:
        jobs += [("random", size, 0.0, None, None, d) for d in range(N_DRAWS)]
        for nm, b in blocks.items():
            members = np.array(b["members"])
            others = np.setdiff1d(np.arange(V.shape[1]), members)
            for f in FRACS:
                jobs += [(nm, size, f, members, others, d) for d in range(N_DRAWS)]
    print(f"{len(jobs)} panels, blocks: {[(k, b['size'], round(b['within_dev'], 3)) for k, b in blocks.items()]}", flush=True)
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(Vt, yt)) as pool:
        rows = list(pool.map(one_setting, jobs, chunksize=8))
    print(f"panels done ({time.time() - t0:.0f}s)", flush=True)
    by = {}
    for j, r in zip(jobs, rows):
        by.setdefault((j[0], j[1], j[2]), []).append(r)
    rng = np.random.default_rng(5)
    settings = []
    for (kind, size, frac), rs in by.items():
        rnd = by[("random", size, 0.0)]
        e = {"block_method": kind, "size": size, "frac": frac, "n_block": 0 if kind == "random" else int(np.floor(frac * size + 0.5)),
             "methods": summarize_rows(rs),
             "fuse_est_prev": float(np.nanmean([np.nan if r["est_f"] is None else r["est_f"] for r in rs])),
             "ds3_est_prev": float(np.mean([r["est_d"] for r in rs])), "fuse_fallbacks": int(sum(r["fallback"] is not None for r in rs)),
             "detector": {"cols_mean": float(np.mean([r["n_cols_detector"] for r in rs])), "merged_mean": float(np.mean([r["n_merged"] for r in rs])),
                          "precision": float(np.sum([r["tp"] for r in rs]) / max(1, np.sum([r["n_merged"] for r in rs]))) if kind != "random" else None,
                          "recall": float(np.sum([r["tp"] for r in rs]) / max(1, np.sum([r["n_planted"] for r in rs]))) if kind != "random" else None,
                          "planted_whole_in_one_group": float(np.mean([r["planted_one_group"] for r in rs])) if kind != "random" else None,
                          "best_jaccard_mean": float(np.mean([r["best_jaccard"] for r in rs])) if kind != "random" else None},
             "pairs": {}}
        for (a, b) in [("FUSE", "Dawid–Skene, 3 answers"), ("FUSE", "Majority vote"), ("Dawid–Skene after detector", "Dawid–Skene, 3 answers"),
                       ("FUSE", "Dawid–Skene after detector"), ("Mean z-score", "Dawid–Skene, 3 answers")]:
            p = {c: boot_acc(rs, None, a, b, c, rng) for c in CUTS}
            p["auc"] = boot_auc(rs, None, a, b, rng)
            if kind != "random":
                p["did_vs_random"] = {c: boot_acc(rs, rnd, a, b, c, rng) for c in CUTS[:2]} | {"auc": boot_auc(rs, rnd, a, b, rng)}
            e["pairs"][f"{a} − {b}"] = p
        settings.append(e)
    data["panels"] = {"blocks": {k: {kk: vv for kk, vv in b.items()} | {"raters": [raters[i] for i in b["members"]]} for k, b in blocks.items()},
                      "settings": settings, "draws": N_DRAWS, "sizes": list(SIZES), "fracs": list(FRACS), "seconds": time.time() - t0}
    OUT_JSON.write_text(json.dumps(data, indent=1))
    print(f"panels written ({time.time() - t0:.0f}s)")


# ---- report -------------------------------------------------------------------------------------


def _ci(x, nd=2):
    return f"{x['diff']:+.{nd}f} [{x['lo']:+.{nd}f}, {x['hi']:+.{nd}f}]"


def _f(x, nd=3):
    return "–" if x is None else f"{x:.{nd}f}"


def write_report():
    data = json.loads(OUT_JSON.read_text())
    G, P = data["groups"], data.get("panels")
    s1, s2, res = G["step1"], G["step2"], G["step3_4"]
    md = ["# DICES-350: groups of raters who share errors (generated by scripts/dices_error_blocks.py)", "",
          "Raw numbers: [dices_error_blocks.json](dices_error_blocks.json). Groups are built on the dev half (174 conversations) "
          "only; the test half (176) is used to check them and to score the panels. Ratings coded Yes = 1, Unsure = 0, No = −1; "
          "a rater's **residual** is the rating minus that rater's mean over conversations with the same expert label, so "
          "residual correlation measures shared errors beyond the label (conditional dependence), not agreement. Rater 61 "
          f"answers No on all 350 conversations and is left out ({G['n_raters']} raters).", ""]

    # step 1
    q = ["1", "5", "25", "50", "75", "95", "99"]
    md += ["## 1. Shared-error (residual) correlation vs plain correlation", "",
           "| Pairs of raters | mean | " + " | ".join(f"p{x}" for x in q) + " | share > 0.3 | share < 0 |", "|---|" + "---:|" * (len(q) + 3)]
    for nm, k in [("Residual, dev", "residual_dev"), ("Plain, dev", "plain_dev")]:
        d = s1[k]
        md.append(f"| {nm} | {d['mean']:.3f} | " + " | ".join(f"{d[x]:.3f}" for x in q) + f" | {d['share_gt_0.3']:.3f} | {d['share_lt_0']:.3f} |")
    d = s1["residual_test"]
    md.append(f"| Residual, test | {d['mean']:.3f} | " + " | ".join(f"{d[x]:.3f}" for x in q) + " | | |")
    md += ["", f"Residual correlation is on average {s1['plain_minus_residual_mean']:.3f} lower than plain correlation, and the two "
           f"rank the pairs almost the same way (correlation over pairs {s1['corr_plain_vs_residual_pairs']:.2f}). Almost every pair "
           "of raters has a POSITIVE residual correlation: DICES raters share errors broadly, not in a few cliques. A pair's residual "
           f"correlation on dev predicts it on test only moderately (correlation over pairs {s1['residual_dev_vs_test_pairs']:.2f}).", "",
           "Ten most correlated pairs on dev, and the same pair on test:", "", "| Rater A | Rater B | r dev | r test |", "|---|---|---:|---:|"]
    md += [f"| {t['a']} | {t['b']} | {t['r_dev']:.2f} | {_f(t['r_test'], 2)} |" for t in s1["top_pairs_dev"]]

    # step 2
    md += ["", "## 2. Do groups exist? Eigenvalues against random-matrix nulls", "",
           f"Marchenko–Pastur edge λ+ = (1 + √(N/T))² = {s2['mp_edge']:.2f} for N = {G['n_raters']}, T = {G['T_dev']} "
           f"(test: {s2['mp_edge_test']:.2f}). The first eigenvalue is a **global mode**: {100 * s2['global_mode_share_of_variance']:.0f}% of "
           "all residual variance, every rater loading positively (median loading "
           f"{s2['global_mode_loadings']['50']:.2f}). It is conversations that everyone gets wrong together, not a group. Once it is "
           "there the noise bulk shrinks, so a second null is used: a one-factor Gaussian model with the fitted global mode and "
           "nothing else (100 draws), next to the shuffle null asked for (each rater's residuals shuffled within class, 100 times).", "",
           "| # | Dev | Test | Shuffle null mean / p95 / max | One-factor null p95 |", "|---:|---:|---:|---|---:|"]
    for j in range(10):
        sh = (f"{s2['shuffle_null_top']['mean'][j]:.2f} / {s2['shuffle_null_top']['p95'][j]:.2f} / {s2['shuffle_null_top']['max'][j]:.2f}"
              if j < 5 else "")
        md.append(f"| {j + 1} | {s2['eig_dev_top10'][j]:.2f} | {s2['eig_test_top10'][j]:.2f} | {sh} | {s2['one_factor_null']['p95'][j]:.2f} |")
    cos = s2["dev_test_subspace_cos"]
    mvp = G["mode_vs_profile"]
    md += ["", f"* Above the MP edge: **{s2['n_above_mp_dev']}** on dev, {s2['n_above_mp_test']} on test, i.e. the global mode plus "
           f"**{s2['n_group_modes_mp']} group modes**; all {s2['n_eig_above_shuffle_null_max']} lie above the largest of 100 shuffle-null "
           f"eigenvalues. Against the one-factor null (the stricter comparison of mode j with null mode j) {s2['n_group_modes_one_factor']} "
           f"modes beyond the global one stand out, but past the third they are small ({s2['eig_dev_top10'][9]:.1f}–{s2['eig_dev_top10'][3]:.1f} "
           f"against null {s2['one_factor_null']['p95'][9]:.1f}–{s2['one_factor_null']['p95'][3]:.1f}). With the edge scaled "
           f"down for the global mode ({s2['mp_edge_adjusted_for_global']:.2f}) {s2['n_above_adjusted_dev']} eigenvalues exceed it.",
           f"* The modes replicate on the held-out half: cosines of the principal angles between the dev and test top-3 eigenspaces "
           f"{cos[0]:.2f}, {cos[1]:.2f}, {cos[2]:.2f}.",
           f"* What mode 2 is: its loadings correlate with a rater's Yes rate ({mvp['mode 2']['yes_rate']:+.2f}) and dev accuracy "
           f"({mvp['mode 2']['bal_acc']:+.2f}), so it separates strict raters (who share false alarms on the same borderline-safe "
           "conversations) from lenient ones (who share misses on the same borderline-unsafe ones). That is what a continuous "
           "conversation severity with rater-specific thresholds produces; it is a gradient more than a set of cliques.", "",
           "**Answer to step 2: yes, but weak.** DICES-350 residuals have one strong global mode and two clear group modes that "
           "survive both nulls and replicate on test; the group structure carries far less variance than the global mode "
           f"(λ2 = {s2['eig_dev_top10'][1]:.1f}, λ3 = {s2['eig_dev_top10'][2]:.1f} against λ1 = {s2['eig_dev_top10'][0]:.1f}).", ""]

    # step 3-4
    md += ["## 3–4. Clusterings and whether the groups are real", "", G["k_rule"] + ".", "",
           "Within / between = mean residual correlation over pairs in the same group / in different groups. Contrast = within − between. "
           "Held-out = the dev-built groups scored on the test half; random = random groups of the same sizes on the test half (2000 "
           "draws, one-sided p). Bootstrap = 100 stratified resamples of dev conversations, clustering rerun: adjusted Rand index (ARI) "
           "to the full-dev clustering, and how often pairs in the same / different full-dev groups land together. Shuffle null = the "
           "same clustering on within-class-shuffled dev residuals (100 times).", "",
           "| Method | Group sizes | Dev within / between (contrast) | Test within / between (contrast) | Test random groups mean / p95, p | "
           "Bootstrap ARI mean [p5–p95] | Co-assigned same / diff | Shuffle-null contrast mean / p95 |", "|---|---|---|---|---|---|---|---|"]
    for nm, r in res.items():
        sizes = ", ".join(map(str, r["sizes"])) + (f" (+{r['n_unassigned']} unassigned)" if r["n_unassigned"] else "")
        md.append(f"| {nm} | {sizes} | {_f(r['dev']['within'])} / {_f(r['dev']['between'])} ({_f(r['contrast_dev'])}) | "
                  f"{_f(r['test']['within'])} / {_f(r['test']['between'])} ({_f(r['contrast_test'])}) | "
                  f"{r['test_random_groups_within_mean']:.3f} / {r['test_random_groups_within_p95']:.3f}, "
                  f"{'< 0.001' if r['test_within_p_value'] < 0.001 else format(r['test_within_p_value'], '.3f')} | "
                  f"{r['boot_ari_mean']:.2f} [{r['boot_ari_p5']:.2f}–{r['boot_ari_p95']:.2f}] | "
                  f"{_f(r['coassign_same_pairs'], 2)} / {_f(r['coassign_diff_pairs'], 2)} | {_f(r['null_contrast_mean'])} / {_f(r['null_contrast_p95'])} |")
    md += ["", "Per group (dev-built). Accuracy = balanced accuracy of the Yes vote on dev; overall mean over raters "
           f"{G['rater_profile_all']['bal_acc_dev']:.3f}, Yes rate {G['rater_profile_all']['yes_rate']:.2f}.", "",
           "| Method | Group | Size | Within dev | Within test | Bootstrap co-assignment | Accuracy (dev) | Yes rate | Unsure rate |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for nm, r in res.items():
        for g, v in r["dev"]["per_group"].items():
            if v["size"] < 2:
                continue
            pr = r["profile"][g]
            md.append(f"| {nm} | {g} | {v['size']} | {_f(v['within'])} | {_f(r['test']['per_group'][g]['within'])} | "
                      f"{_f(r['per_group_coassign'].get(g), 2)} | {pr['bal_acc_dev']:.3f} | {pr['yes_rate']:.2f} | {pr['unsure_rate']:.2f} |")
    md += ["", "Sensitivity, spectral and hierarchical at k = 4 and 5:", "", "| Clustering | Sizes | Within dev | Between dev | Within test |", "|---|---|---:|---:|---:|"]
    md += [f"| {k} | {', '.join(map(str, v['sizes']))} | {_f(v['within_dev'])} | {_f(v['between_dev'])} | {_f(v['within_test'])} |"
           for k, v in G["step3_sensitivity"].items()]
    md += ["", "Agreement with demographics (adjusted Rand index; 0 = no more than chance):", "",
           "| Method | " + " | ".join(c.replace("rater_", "") for c in DEMOS) + " |", "|---|" + "---:|" * len(DEMOS)]
    md += [f"| {nm} | " + " | ".join(f"{r['demographics'][c]['ari']:+.3f}" for c in DEMOS) + " |" for nm, r in res.items()]
    mg = res["MacMahon–Garlaschelli"]["demographics"]
    md += ["", "Cross-tabs, MacMahon–Garlaschelli groups (rows: group; counts of raters):", ""]
    for c in DEMOS:
        cats = sorted({k for v in mg[c]["crosstab"].values() for k in v})
        md += [f"*{c.replace('rater_', '')}*", "", "| Group | " + " | ".join(cats) + " |", "|---|" + "---:|" * len(cats)]
        md += [f"| {g} | " + " | ".join(str(v.get(k, 0)) for k in cats) + " |" for g, v in mg[c]["crosstab"].items()]
        md.append("")

    if P:
        md += ["## 5. Does FUSE gain when a shared-error group is a minority of the panel?", "",
               f"{P['draws']} draws per setting; panels of {', '.join(map(str, P['sizes']))} raters; the block is "
               f"{', '.join(str(int(100 * f)) + '%' for f in P['fracs'])} of the panel, the rest drawn from raters outside it. Every method "
               "is fitted on the test half only and scored there (176 conversations, 88 unsafe). **Shared cut-off**: every method calls the "
               "same number unsafe (three-answer Dawid–Skene's estimated prevalence × 176; scripts/panel_eval.py). **DiD**: the "
               "difference in the block panels minus the same difference in size-matched random panels, the part due to the block. "
               "Intervals: paired bootstrap over conversations (AUC: stratified), 95%. Detector: fuse.clustering.residual_groups at "
               "0.4, label-free; precision = share of merged raters that belong to the planted block, recall = share of the planted "
               "block that got merged.", "", "Blocks (each chosen on dev):", "",
               "| Block from | Size | Within dev | Within test | Accuracy (dev) | Yes rate |", "|---|---:|---:|---:|---:|---:|"]
        md += [f"| {k} | {b['size']} | {b['within_dev']:.3f} | {b['within_test']:.3f} | {b['profile']['bal_acc_dev']:.3f} | {b['profile']['yes_rate']:.2f} |"
               for k, b in P["blocks"].items()]
        md += ["", "Hierarchical clustering (one group of 118) and the plain greedy rule (pairs) have no group of 8–106 raters and give no block.", "",
               "### FUSE − three-answer Dawid–Skene (accuracy points; AUC points)", "",
               "| Block | Panel | Block share | Shared cut-off | DiD vs random | Own rule (> 0.5) | AUC | AUC DiD |", "|---|---:|---:|---|---|---|---|---|"]
        CS = "shared (DS-3 est.)"
        did_acc, did_auc = [], []
        for e in P["settings"]:
            p = e["pairs"]["FUSE − Dawid–Skene, 3 answers"]
            if e["block_method"] == "random":
                md.append(f"| random panel | {e['size']} | – | {_ci(p[CS])} | – | {_ci(p['natural'])} | {_ci(p['auc'])} | – |")
            else:
                did_acc.append(p["did_vs_random"][CS])
                did_auc.append(p["did_vs_random"]["auc"])
                md.append(f"| {e['block_method']} | {e['size']} | {int(100 * e['frac'])}% | {_ci(p[CS])} | {_ci(p['did_vs_random'][CS])} | "
                          f"{_ci(p['natural'])} | {_ci(p['auc'])} | {_ci(p['did_vs_random']['auc'])} |")
        n = len(did_acc)
        up_a = sum(d["lo"] > 0 for d in did_acc)
        dn_a = sum(d["hi"] < 0 for d in did_acc)
        up_u = sum(d["lo"] > 0 for d in did_auc)
        dn_u = sum(d["hi"] < 0 for d in did_auc)
        md += ["", f"Across the {n} block settings, the accuracy DiD at the shared cut-off averages {np.mean([d['diff'] for d in did_acc]):+.2f} "
               f"points (range {min(d['diff'] for d in did_acc):+.2f} to {max(d['diff'] for d in did_acc):+.2f}); its interval excludes 0 "
               f"upward in {up_a} and downward in {dn_a}. The AUC DiD averages {np.mean([d['diff'] for d in did_auc]):+.2f} (excludes 0 upward "
               f"in {up_u}, downward in {dn_u}). With {n} uncorrected intervals, about {0.05 * n:.0f} would exclude 0 by chance.", "",
               "### Dawid–Skene after the label-free detector, and what the detector finds", "",
               "| Block | Panel | Block share | DS after detector − DS (shared cut-off) | FUSE − DS after detector (shared) | DS after detector − DS (AUC) | "
               "Columns after detector | Raters merged | Precision | Recall | Block in one group |", "|---|---:|---:|---|---|---|---:|---:|---:|---:|---:|"]
        for e in P["settings"]:
            q = e["pairs"]["Dawid–Skene after detector − Dawid–Skene, 3 answers"]
            q2 = e["pairs"]["FUSE − Dawid–Skene after detector"]
            dd = e["detector"]
            md.append(f"| {e['block_method']} | {e['size']} | {'–' if e['block_method'] == 'random' else str(int(100 * e['frac'])) + '%'} | "
                      f"{_ci(q[CS])} | {_ci(q2[CS])} | {_ci(q['auc'])} | {dd['cols_mean']:.1f} | {dd['merged_mean']:.1f} | "
                      f"{_f(dd['precision'], 2)} | {_f(dd['recall'], 2)} | {_f(dd['planted_whole_in_one_group'], 2)} |")
        md += ["", "### All methods: accuracy at the shared cut-off / at own rule, and AUC (mean over panels)", "",
               "| Block | Panel | Share | " + " | ".join(METHODS) + " |", "|---|---:|---:|" + "---|" * len(METHODS)]
        for e in P["settings"]:
            m = e["methods"]
            md.append(f"| {e['block_method']} | {e['size']} | {'–' if e['block_method'] == 'random' else str(int(100 * e['frac'])) + '%'} | " +
                      " | ".join(f"{m[x][CS]:.1f} / {m[x]['natural']:.1f} / {m[x]['auc']:.1f}" for x in METHODS) + " |")
        md += ["", "Mean z-score's own rule is z > 0 (it calls about half unsafe by construction)."]
    OUT_MD.write_text("\n".join(md) + "\n")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["groups", "panels", "report"])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    if a.step == "groups":
        step_groups()
    elif a.step == "panels":
        step_panels(a.workers)
    else:
        write_report()
