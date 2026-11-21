"""Small random rater panels: majority vote vs Dawid–Skene vs FUSE inside each panel (DICES-350).

Hypothesis: with all 123 raters the few good raters are drowned out by the crowd, and in a
small panel they weigh more, so a method that learns rater quality without labels (FUSE)
should gain most there.

Each repetition shuffles the 123 raters and cuts them into disjoint panels of about `size`
(123 // size panels, e.g. 12 panels of 10–11). Inside each panel, every method sees only that
panel's raters, one verifier per rater (Yes = 1, Unsure = 0, No = -1), on all 350 conversations:

* Majority vote: unsafe when the panel's share of Yes (Unsure = 1/2) is above 1/2.
* Dawid–Skene: one-coin-per-class EM on the panel's Yes votes, unsafe when q > 0.5.
* Dawid–Skene cut at its own estimated prevalence (the mean of q).
* FUSE: repo defaults, unsafe when its score > 0.5.
* FUSE cut at its own estimated prevalence (the cut-off adopted by the dev-half tuning).

No method uses a label, so all are scored on all 350 conversations against the expert label.
Besides accuracy and AUC, each panel records whether a method finds the good raters: the
Spearman correlation between each rater's quality as the method estimates it (balanced accuracy
for FUSE, from psi/eta implied by q for Dawid–Skene) and the rater's true balanced accuracy.

    python scripts/dices_small_panels.py --reps 100 --sizes 5 10 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fuse import FUSE, dices  # noqa: E402

METHODS = ["Majority vote", "Dawid–Skene", "Dawid–Skene cut at its prevalence", "FUSE", "FUSE cut at its prevalence"]
PAIRS = [("FUSE", "Majority vote"), ("FUSE", "Dawid–Skene"),
         ("FUSE cut at its prevalence", "Dawid–Skene cut at its prevalence"), ("Dawid–Skene", "Majority vote")]

_V = _Y = None  # (items, raters) in {-1, 0, 1}, gold labels; set in each worker


def _init(V, y):
    global _V, _Y
    _V, _Y = V, y


def top_share(score, share):
    k = int(round(share * len(score)))
    pred = np.zeros(len(score), dtype=bool)
    pred[np.argsort(-score, kind="stable")[:k]] = True
    return pred


def balanced_acc(pred, y):
    return float((pred[y].mean() + (~pred[~y]).mean()) / 2)


def ds_rater_quality(V, q):
    """Balanced accuracy of each rater implied by Dawid–Skene's posterior q."""
    B = (V > 0).astype(float)
    psi = (q[:, None] * B).sum(0) / q.sum()
    eta = ((1 - q)[:, None] * (1 - B)).sum(0) / (1 - q).sum()
    return (psi + eta) / 2


def spearman(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(spearmanr(a, b).statistic)


def one_rep(args):
    size, rep = args
    n_raters = _V.shape[1]
    perm = np.random.default_rng(20_000 + rep).permutation(n_raters)
    out = []
    for p, cols in enumerate(np.array_split(perm, n_raters // size)):
        V = _V[:, cols]
        y = _Y
        true_q = np.array([balanced_acc(V[:, j] > 0, y) for j in range(V.shape[1])])
        share = ((V + 1) / 2).mean(1)  # Yes = 1, Unsure = 1/2, No = 0
        q = dices.dawid_skene((V > 0).astype(float))  # Yes votes; Unsure and No are "safe"
        r = FUSE().fit_scores(V, rescale=False)
        est = None if r.estimate is None else float(r.estimate.prevalence)
        preds = {
            "Majority vote": (share, share > 0.5),
            "Dawid–Skene": (q, q > 0.5),
            "Dawid–Skene cut at its prevalence": (q, top_share(q, float(q.mean()))),
            "FUSE": (r.scores, r.scores > 0.5),
            "FUSE cut at its prevalence": (r.scores, top_share(r.scores, est) if est is not None else r.scores > 0.5),
        }
        row = {"rep": rep, "panel": p, "n": len(cols), "fallback": r.fallback,
               "fuse_est_prev": est, "ds_est_prev": float(q.mean()),
               "best_rater_acc": round(100 * float(max(((V[:, j] > 0) == y).mean() for j in range(V.shape[1]))), 2),
               "mean_rater_acc": round(100 * float(np.mean([((V[:, j] > 0) == y).mean() for j in range(V.shape[1])])), 2),
               "rater_ba_spread": round(100 * float(true_q.max() - true_q.min()), 2)}
        for m, (s, pr) in preds.items():
            row[f"{m}|acc"] = round(100 * float((pr == y).mean()), 2)
            row[f"{m}|auc"] = round(100 * float(roc_auc_score(y, s)), 2)
            row[f"{m}|unsafe"] = round(100 * float(pr.mean()), 1)
            row[f"{m}|hits"] = (pr == y)
        # Does the method find the good raters? FUSE estimates only the raters it kept.
        if r.estimate is not None:
            kept = np.asarray(r.kept)
            row["fuse_quality_rho"] = spearman(r.estimate.balanced_accuracy, true_q[kept])
            row["fuse_n_kept"] = int(len(kept))
        else:
            row["fuse_quality_rho"], row["fuse_n_kept"] = None, 0
        row["ds_quality_rho"] = spearman(ds_rater_quality(V, q), true_q)
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--sizes", type=int, nargs="*", default=[5, 10, 20])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()

    m = dices.build(dices.load("350"), view="overall")
    V, y = m.V, m.gold
    assert not np.isnan(V).any()
    n = len(y)
    # Full-crowd reference: the same methods on all 123 raters.
    q_all = dices.dawid_skene(V)
    f_all = FUSE().fit_scores(V, rescale=False)
    ref = {"Majority vote, 123 raters": 100 * float(((m.majority > 0.5) == y).mean()),
           "Dawid–Skene, 123 raters": 100 * float(((q_all > 0.5) == y).mean()),
           "FUSE, 123 raters": 100 * float(((f_all.scores > 0.5) == y).mean())}

    out = {"reps": a.reps, "reference_123": {k: round(v, 2) for k, v in ref.items()}, "sizes": {}}
    md = ["# Small random rater panels on DICES-350 (generated by scripts/dices_small_panels.py)", "",
          f"{a.reps} reshuffles of the 123 raters per panel size; each reshuffle cuts them into disjoint panels. "
          "Inside each panel every method sees only that panel's raters. No method uses a label; accuracy (%) is "
          "against the expert label on all 350 conversations.", "",
          "With all 123 raters: " + ", ".join(f"{k} {v:.1f}" for k, v in ref.items()) + ".", ""]
    rng = np.random.default_rng(0)
    for size in a.sizes:
        t0 = time.time()
        rows = []
        with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(V, y)) as pool:
            for i, res in enumerate(pool.map(one_rep, [(size, r) for r in range(a.reps)], chunksize=4)):
                rows.extend(res)
                if (i + 1) % 25 == 0:
                    print(f"size {size}: {i + 1}/{a.reps} reshuffles ({time.time() - t0:.0f}s)", flush=True)
        P = len(rows)
        summ = {}
        for meth in METHODS:
            summ[meth] = {}
            for k in ["acc", "auc", "unsafe"]:
                v = np.array([r[f"{meth}|{k}"] for r in rows])
                summ[meth][k] = {"mean": round(float(v.mean()), 2), "sd": round(float(v.std()), 2),
                                 "p2.5": round(float(np.percentile(v, 2.5)), 2), "p97.5": round(float(np.percentile(v, 97.5)), 2)}
        # Per-conversation hit rate averaged over all panels, for a bootstrap over conversations.
        rate = {meth: np.mean([r[f"{meth}|hits"] for r in rows], axis=0) for meth in METHODS}
        idx = rng.integers(0, n, (5000, n))
        pairs = []
        for m1, m2 in PAIRS:
            for k in ["acc", "auc"]:
                d = np.array([r[f"{m1}|{k}"] - r[f"{m2}|{k}"] for r in rows])
                pr = {"a": m1, "b": m2, "metric": k, "mean_diff": round(float(d.mean()), 2),
                      "a_wins": round(100 * float((d > 0).mean()), 1), "tie": round(100 * float((d == 0).mean()), 1),
                      "b_wins": round(100 * float((d < 0).mean()), 1),
                      "p2.5": round(float(np.percentile(d, 2.5)), 2), "p97.5": round(float(np.percentile(d, 97.5)), 2)}
                if k == "acc":
                    bs = 100 * (rate[m1][idx].mean(1) - rate[m2][idx].mean(1))
                    pr["conv_ci"] = [round(float(np.percentile(bs, 2.5)), 2), round(float(np.percentile(bs, 97.5)), 2)]
                pairs.append(pr)
        fr = np.array([r["fuse_quality_rho"] for r in rows if r["fuse_quality_rho"] is not None])
        dr = np.array([r["ds_quality_rho"] for r in rows if r["ds_quality_rho"] is not None])
        best = np.array([r["best_rater_acc"] for r in rows])
        mean_r = np.array([r["mean_rater_acc"] for r in rows])
        # Where good raters stand out most: panels in the top quarter of rater spread.
        spread = np.array([r["rater_ba_spread"] for r in rows])
        hi = spread >= np.percentile(spread, 75)
        lo = spread <= np.percentile(spread, 25)
        by_spread = {meth: {"high": round(float(np.mean([r[f"{meth}|acc"] for r, h in zip(rows, hi) if h])), 2),
                            "low": round(float(np.mean([r[f"{meth}|acc"] for r, l_ in zip(rows, lo) if l_])), 2)}
                     for meth in METHODS}
        diag = {"panels": P, "panel_sizes": sorted({r["n"] for r in rows}),
                "fuse_quality_rho_mean": round(float(fr.mean()), 3), "fuse_quality_rho_n": int(len(fr)),
                "ds_quality_rho_mean": round(float(dr.mean()), 3), "ds_quality_rho_n": int(len(dr)),
                "fuse_kept_mean": round(float(np.mean([r["fuse_n_kept"] for r in rows])), 2),
                "best_rater_acc_mean": round(float(best.mean()), 2), "mean_rater_acc_mean": round(float(mean_r.mean()), 2),
                "fallbacks": sum(r["fallback"] is not None for r in rows),
                "fuse_est_prev_mean": round(float(np.nanmean([r["fuse_est_prev"] or np.nan for r in rows])), 3),
                "ds_est_prev_mean": round(float(np.mean([r["ds_est_prev"] for r in rows])), 3),
                "by_spread": by_spread}
        out["sizes"][str(size)] = {"summary": summ, "pairs": pairs, "diagnostics": diag}
        print(f"size {size} done in {time.time() - t0:.0f}s ({P} panels)", flush=True)

        md += [f"## Panels of about {size} raters ({P} panels: {a.reps} reshuffles × {P // a.reps})", "",
               f"Average single rater in a panel: {diag['mean_rater_acc_mean']:.1f}; best rater in the panel (picked with "
               f"the labels): {diag['best_rater_acc_mean']:.1f}. Estimated unsafe share: FUSE {diag['fuse_est_prev_mean']:.2f}, "
               f"Dawid–Skene {diag['ds_est_prev_mean']:.2f} (experts 0.50). FUSE keeps {diag['fuse_kept_mean']:.1f} raters "
               f"on average; {diag['fallbacks']} panels fell back to a plain average.", "",
               "| Method | Accuracy, mean ± sd | Middle 95% of panels | AUC, mean | Called unsafe % |",
               "|---|---:|---:|---:|---:|"]
        for meth in METHODS:
            s = summ[meth]
            md.append(f"| {meth} | {s['acc']['mean']:.1f} ± {s['acc']['sd']:.1f} | {s['acc']['p2.5']:.1f}–{s['acc']['p97.5']:.1f} | "
                      f"{s['auc']['mean']:.1f} | {s['unsafe']['mean']:.0f} |")
        md += ["", "| Comparison | Metric | Mean difference | A wins / tie / B wins (% of panels) | Middle 95% of panels | "
               "Over conversations (95% CI) |", "|---|---|---:|---|---|---|"]
        for p in pairs:
            ci = f"[{p['conv_ci'][0]:+.1f}, {p['conv_ci'][1]:+.1f}]" if "conv_ci" in p else ""
            md.append(f"| {p['a']} − {p['b']} | {p['metric']} | {p['mean_diff']:+.1f} | {p['a_wins']:.0f} / {p['tie']:.0f} / "
                      f"{p['b_wins']:.0f} | {p['p2.5']:+.1f} to {p['p97.5']:+.1f} | {ci} |")
        md += ["", f"**Does the method find the good raters?** Spearman between estimated and true rater quality, "
               f"mean over panels: FUSE {diag['fuse_quality_rho_mean']:+.2f} (over the raters it keeps, {diag['fuse_quality_rho_n']} panels), "
               f"Dawid–Skene {diag['ds_quality_rho_mean']:+.2f} ({diag['ds_quality_rho_n']} panels).", "",
               "Accuracy in the quarter of panels where raters differ most in quality vs the quarter where they differ least:", "",
               "| Method | Most spread | Least spread |", "|---|---:|---:|"]
        md += [f"| {meth} | {v['high']:.1f} | {v['low']:.1f} |" for meth, v in by_spread.items()]
        md.append("")
        (ROOT / "results" / "DICES_SMALL_PANELS.md").write_text("\n".join(md) + "\n")
        (ROOT / "results" / "dices_small_panels.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
