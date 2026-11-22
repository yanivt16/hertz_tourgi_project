"""VIPER follow-up: (A) can a label-free method rank radiologists by skill when it sees their
suspicion scores rather than their recall decisions? (B) do radiologists form blocks of shared errors?

    pip install rdata
    python scripts/viper_skill_and_blocks.py      # writes results/VIPER_SKILL_BLOCKS.md

(A) True skill = each reader's AUC of the 0-201 suspicion score against cancer, which does not depend
on where the reader puts the recall threshold. Label-free estimates, compared by Spearman correlation
across readers:
  - Binned Dawid-Skene, model-based: each reader's scores in 5 quantile bins; the AUC implied by the
    fitted per-class bin distributions.
  - Soft-label AUC: the reader's AUC against the posterior of an aggregator fitted WITHOUT that reader
    (leave-one-out), with the posterior used as a soft label. Aggregators: binned Dawid-Skene, FUSE.
  - Agreement: Spearman correlation of the reader's scores with the mean z-score of the other readers.
  - For reference, the decision-based estimate from viper_recall_vs_cancer.py: Dawid-Skene skill (J)
    from recall decisions.
(B) Our residual detector (fuse.clustering.residual_groups) on the scores of each dense block, at
thresholds 0.4 and 0.3. For every merged group: the residual correlation, the readers' reading track
(the only reader-level field in the data) and their true AUCs.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from fuse import viper  # noqa: E402
from fuse.clustering import residual_groups  # noqa: E402
from fuse.ensemble import FUSE  # noqa: E402
from fuse.ordinal import dawid_skene_ordinal, quantile_bins  # noqa: E402

np.seterr(all="ignore")


def rho(a, b) -> str:
    return f"{spearmanr(a, b, nan_policy='omit').correlation:+.2f}"


def soft_auc(s: np.ndarray, q: np.ndarray) -> float:
    """P(score of a positive > score of a negative), with q as the probability that an item is positive."""
    ok = ~np.isnan(s)
    s, q = s[ok], q[ok]
    order = np.argsort(s)
    s, q = s[order], q[order]
    w1, w0 = q, 1 - q
    num, cum0, i = 0.0, 0.0, 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        a1, a0 = w1[i:j].sum(), w0[i:j].sum()
        num += a1 * cum0 + 0.5 * a1 * a0
        cum0 += a0
        i = j
    return num / (w1.sum() * w0.sum())


def ds_model_auc(S: np.ndarray, k: int = 5) -> np.ndarray:
    """Per-reader AUC implied by binned Dawid-Skene's per-class bin distributions."""
    q = dawid_skene_ordinal(S, k)
    L = quantile_bins(S, k)
    K = L.max() + 1
    out = np.full(S.shape[1], np.nan)
    for j in range(S.shape[1]):
        ok = L[:, j] >= 0
        p1 = np.array([(q[ok] * (L[ok, j] == b)).sum() for b in range(K)]) + 0.5
        p0 = np.array([((1 - q[ok]) * (L[ok, j] == b)).sum() for b in range(K)]) + 0.5
        p1, p0 = p1 / p1.sum(), p0 / p0.sum()
        out[j] = sum(p1[a] * p0[b] for a in range(K) for b in range(K) if a > b) + 0.5 * (p1 * p0).sum()
    return out


def loo_soft_auc(S: np.ndarray, fit) -> np.ndarray:
    out = np.full(S.shape[1], np.nan)
    for j in range(S.shape[1]):
        q = fit(np.delete(S, j, 1))
        out[j] = soft_auc(S[:, j], q)
    return out


def agreement(S: np.ndarray) -> np.ndarray:
    Z = (S - np.nanmean(S, 0)) / np.nanstd(S, 0)
    out = np.full(S.shape[1], np.nan)
    for j in range(S.shape[1]):
        others = np.nanmean(np.delete(Z, j, 1), 1)
        ok = ~np.isnan(S[:, j]) & ~np.isnan(others)
        out[j] = spearmanr(S[ok, j], others[ok]).correlation
    return out


def ds_decision_J(B: np.ndarray) -> np.ndarray:
    """Dawid-Skene on recall decisions (same model as viper_recall_vs_cancer.ds_rates); returns se+sp-1."""
    M = (~np.isnan(B)).astype(float)
    X = np.nan_to_num(B)
    q = (X.sum(1) / np.maximum(M.sum(1), 1)).clip(1e-3, 1 - 1e-3)
    for _ in range(200):
        prior = q.mean().clip(1e-3, 1 - 1e-3)
        se = ((q[:, None] * X).sum(0) + 1) / ((q[:, None] * M).sum(0) + 2)
        sp = (((1 - q)[:, None] * (M - X)).sum(0) + 1) / (((1 - q)[:, None] * M).sum(0) + 2)
        l1 = np.log(prior) + X @ np.log(se) + (M - X) @ np.log(1 - se)
        l0 = np.log(1 - prior) + (M - X) @ np.log(sp) + X @ np.log(1 - sp)
        q = 1 / (1 + np.exp(np.clip(l0 - l1, -50, 50)))
    return se + sp - 1


def fuse_posterior(S: np.ndarray) -> np.ndarray:
    r = FUSE(predictor="posterior", drop_below_random=False).fit_scores(S)
    return r.posterior if r.posterior is not None else r.scores


def true_auc(S: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.array([roc_auc_score(y[~np.isnan(S[:, j])], S[~np.isnan(S[:, j]), j]) for j in range(S.shape[1])])


def main():
    reads, gold = viper.load()
    track = reads.groupby("readerID").Track.first().astype(str)
    out = ["# VIPER: ranking readers by skill from scores, and blocks among readers "
           "(generated by scripts/viper_skill_and_blocks.py)", "",
           "True skill = each reader's AUC of the suspicion score against cancer at 455 days. Cells in part A are "
           "Spearman correlations across readers between a label-free estimate and the true skill. Soft-label "
           "AUCs are leave-one-out: the reader's own scores are not used to fit the aggregator.", ""]
    for mod in ["FFDM", "SFM"]:
        r = reads[reads.modalityID == mod]
        S, B, y = viper.build(r, gold)
        y = np.asarray(y).astype(int)
        auc = true_auc(S, y)
        out += [f"## {mod}", "", f"Per-reader true AUC: median {np.median(auc):.3f}, range {auc.min():.3f}-{auc.max():.3f}.", "",
                "### A. Label-free estimates of skill vs true per-reader AUC", "",
                "| Panel | Readers | Binned D&S (model) | Binned D&S soft-label AUC (LOO) | FUSE soft-label AUC (LOO) | Agreement with others (LOO) | D&S skill from recall decisions |",
                "|---|---:|---:|---:|---:|---:|---:|"]
        est_model = ds_model_auc(S)
        est_ds = loo_soft_auc(S, lambda X: dawid_skene_ordinal(X))
        out.append(f"| Full matrix ({np.isnan(S).mean():.0%} missing) | {S.shape[1]} | {rho(est_model, auc)} | {rho(est_ds, auc)} | "
                   f"not run (needs a dense matrix) | {rho(agreement(S), auc)} | {rho(ds_decision_J(B), auc)} |")
        blocks = viper.dense_blocks(r)
        block_rows, det_rows = [], []
        for n_cases, (rs, cs) in sorted(blocks.items()):
            Sb, Bb, yb = viper.build(r[r.readerID.isin(rs) & r.caseID.isin(cs)], gold)
            yb = np.asarray(yb).astype(int)
            ab = true_auc(Sb, yb)
            ids = sorted(r[r.readerID.isin(rs)].readerID.unique())
            block_rows.append(f"| Dense {len(ids)} readers x {n_cases} cases ({yb.sum()} cancers) | {len(ids)} | "
                              f"{rho(ds_model_auc(Sb), ab)} | {rho(loo_soft_auc(Sb, lambda X: dawid_skene_ordinal(X)), ab)} | "
                              f"{rho(loo_soft_auc(Sb, fuse_posterior), ab)} | {rho(agreement(Sb), ab)} | {rho(ds_decision_J(Bb), ab)} |")
            for t in (0.4, 0.3):
                groups, P, _ = residual_groups(Sb, threshold=t)
                for g in groups:
                    if len(g) > 1:
                        pc = np.mean([P[a, b] for a in g for b in g if a < b])
                        det_rows.append(f"| {len(ids)} x {n_cases} | {t} | {', '.join(ids[k] for k in g)} | {pc:.2f} | "
                                        f"{', '.join(track[ids[k]] for k in g)} | {', '.join(f'{ab[k]:.2f}' for k in g)} |")
                if not any(len(g) > 1 for g in groups):
                    det_rows.append(f"| {len(ids)} x {n_cases} | {t} | none merged | | | |")
        out += block_rows + ["", "### B. Blocks among readers (residual detector on dense blocks)", "",
                             "| Block | Threshold | Readers merged | Mean residual correlation | Reading track | True AUC |",
                             "|---|---:|---|---:|---|---|"] + det_rows + [""]
        # how often readers on the same track have high residual correlation, full set of dense blocks
    path = ROOT / "results" / "VIPER_SKILL_BLOCKS.md"
    path.write_text("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
