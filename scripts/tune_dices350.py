"""Coarse-to-fine, one-change-at-a-time tuning of FUSE on DICES-350.

FUSE never sees a label: it is always fitted on all 350 conversations. The expert label is
used only to choose settings, by accuracy on a fixed "dev" half (the same stratified split
as the LLM-label work), and the chosen configuration is scored once on the "test" half.

Search: coordinate descent over one setting at a time, starting from the repo defaults on
the per-rater view. A step sweeps one setting with everything else at the incumbent; the
best value is adopted only if it gains at least GAIN points of dev accuracy and does not
lower dev AUC by more than AUC_TOL. Stage 1 is the verifier view, stage 2 switches parts of
the algorithm on and off, stage 3 tunes numeric values. Then all three are swept again
(pass 2, 3, ...) until a whole pass adopts nothing.

    python scripts/tune_dices350.py next      # run the next step, save state
    python scripts/tune_dices350.py stage     # run steps until the stage (or pass) changes
    python scripts/tune_dices350.py all       # run every remaining step
    python scripts/tune_dices350.py final     # seed stability + one score on the test half
    python scripts/tune_dices350.py report    # write results/DICES_TUNING.md

State is kept in results/dices_tuning.json; `--export PATH` also writes the run table as
{rows, meta} JSON for the live dashboard.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fuse import FUSE  # noqa: E402
from fuse import dices  # noqa: E402

STATE = ROOT / "results" / "dices_tuning.json"
REPORT = ROOT / "results" / "DICES_TUNING.md"
GAIN = 2.0  # points of dev accuracy needed to adopt a change (1 conversation = 0.57)
AUC_TOL = 0.5  # a change may not lower dev AUC by more than this
MAX_PASSES = 4
RANDOM_SEEDS = 5

FUSE_DEFAULTS = {f.name: f.default for f in fields(FUSE)}
DEFAULT_CONFIG = {"view": "raters", "cutoff": "0.5", **FUSE_DEFAULTS}

# Verifier views. `eligible=False` marks controls that are reported but never adopted.
VIEWS: dict[str, dict] = {
    "raters": dict(kind="overall"),
    "raters, Unsure=Yes": dict(kind="overall", unsure="Yes"),
    "raters, Unsure=No": dict(kind="overall", unsure="No"),
    "raters, quality-filtered": dict(kind="overall", drop_removed=True),
    "granular counts": dict(kind="granular"),
    "groups: gender": dict(kind="groups", groups=("rater_gender",)),
    "groups: age": dict(kind="groups", groups=("rater_age",)),
    "groups: education": dict(kind="groups", groups=("rater_education",)),
    "groups: race": dict(kind="groups", groups=("rater_race",)),
    "groups: race×gender": dict(kind="groups", groups=("rater_race", "rater_gender")),
    "groups: race×age": dict(kind="groups", groups=("rater_race", "rater_age")),
    "groups: gender×age": dict(kind="groups", groups=("rater_gender", "rater_age")),
    "groups: race×education": dict(kind="groups", groups=("rater_race", "rater_education")),
    "groups: race×gender×age": dict(kind="groups", groups=("rater_race", "rater_gender", "rater_age")),
    "random groups of 5": dict(kind="random", size=5, eligible=False),
    "random groups of 10": dict(kind="random", size=10, eligible=False),
    "random groups of 20": dict(kind="random", size=20, eligible=False),
}

# (stage, setting, values, applies-when)
STEPS = [
    (1, "view", list(VIEWS), None),
    (2, "predictor", ["logistic", "posterior"], None),
    (2, "drop_below_random", [True, False], None),
    (2, "refit_after_drop", [True, False], lambda c: c["drop_below_random"]),
    (2, "tci_standardize", [False, True], None),
    (2, "cluster_threshold", [None, 0.5], None),
    (2, "cutoff", ["0.5", "prevalence"], None),
    (3, "n_quantiles", [5, 9, 19, 39], None),
    (3, "n_sweeps", [1, 3, 5, 10], None),
    (3, "logistic_C", [0.01, 0.1, 1.0, 10.0, 100.0], lambda c: c["predictor"] == "logistic"),
    (3, "cluster_threshold", [0.3, 0.4, 0.5, 0.6, 0.7, 0.8], lambda c: c["cluster_threshold"] is not None),
    (3, "tci_clip", [1e-4, 1e-3, 1e-2], None),
]
STAGE_NAME = {1: "Views", 2: "Switches", 3: "Numbers"}


# ---- data ------------------------------------------------------------------------------


def split_350(items: np.ndarray, gold: np.ndarray) -> np.ndarray:
    """Boolean dev mask; same stratified split as scripts/label_dices_llm.py (seed 350)."""
    rng = np.random.default_rng(350)
    dev = []
    for g in (True, False):
        dev.extend(rng.permutation(items[gold == g])[: (gold == g).sum() // 2])
    return np.isin(items, dev)


class Data:
    def __init__(self):
        self.df = dices.load("350")
        base = dices.build(self.df, view="overall")
        self.items, self.gold = base.items, base.gold
        self.dev = split_350(self.items, self.gold)
        self.majority = base.majority
        self.raters_V = base.V
        self._cache: dict = {}

    def matrix(self, view: str, seed: int = 0) -> dices.DicesMatrix:
        key = (view, seed)
        if key in self._cache:
            return self._cache[key]
        spec = VIEWS[view]
        df = self.df
        if spec.get("unsure"):
            df = df.copy()
            df["Q_overall"] = df.Q_overall.replace("Unsure", spec["unsure"])
            df.attrs["variant"] = "350"
        if spec["kind"] == "random":
            raters = np.sort(df.rater_id.unique())
            perm = np.random.default_rng(1000 + seed).permutation(raters)
            # Near-equal groups (123 // size of them), as in scripts/dices_random_groups.py.
            gid = {r: f"g{g:02d}" for g, part in enumerate(np.array_split(perm, len(raters) // spec["size"])) for r in part}
            df = df.assign(rater_rand=df.rater_id.map(gid))
            df.attrs["variant"] = "350"
            m = dices.build(df, view="groups", groups=("rater_rand",))
        elif spec["kind"] == "groups":
            m = dices.build(df, view="groups", groups=spec["groups"])
        else:
            m = dices.build(df, view=spec["kind"], drop_removed=spec.get("drop_removed", False))
        assert (m.items == self.items).all()
        self._cache[key] = m
        return m


# ---- one run ---------------------------------------------------------------------------


def predict(data: Data, config: dict, seed: int = 0) -> dict:
    m = data.matrix(config["view"], seed)
    X = dices._prepared(m)
    model = FUSE(**{k: config[k] for k in FUSE_DEFAULTS})
    r = model.fit_scores(X, rescale=False)
    est = None if r.estimate is None else float(r.estimate.prevalence)
    if config["cutoff"] == "prevalence" and est is None:
        print(f"warning: FUSE gave no prevalence estimate for {config['view']}; cut at 0.5 instead", flush=True)
    if config["cutoff"] == "prevalence" and est is not None:
        k = int(round(est * len(r.scores)))
        pred = np.zeros(len(r.scores), dtype=bool)
        pred[np.argsort(-r.scores, kind="stable")[:k]] = True
    else:
        pred = r.scores > 0.5
    return {"scores": r.scores, "pred": pred, "est_prev": est, "tci": r.tci,
            "n_kept": int(len(r.kept)), "n_verifiers": int(X.shape[1]), "fallback": r.fallback}


def metrics(score, pred, y) -> dict:
    pred = np.asarray(pred, dtype=bool)
    return {
        "acc": 100 * float((pred == y).mean()),
        "bal_acc": 100 * float((pred[y].mean() + (~pred[~y]).mean()) / 2),
        "auc": 100 * float(roc_auc_score(y, score)),
        "pred_prev": 100 * float(pred.mean()),
    }


def evaluate(data: Data, config: dict) -> dict:
    t0 = time.time()
    spec = VIEWS[config["view"]]
    seeds = range(RANDOM_SEEDS) if spec["kind"] == "random" else [0]
    outs = []
    for s in seeds:
        p = predict(data, config, s)
        d = data.dev
        outs.append({**metrics(p["scores"][d], p["pred"][d], data.gold[d]),
                     "est_prev": p["est_prev"], "tci": p["tci"], "n_kept": p["n_kept"],
                     "n_verifiers": p["n_verifiers"], "fallback": p["fallback"]})
    res = {}
    for k in outs[0]:
        vals = [o[k] for o in outs]
        if all(isinstance(v, (int, float)) and v is not None for v in vals):
            res[k] = float(np.mean(vals))
        else:
            res[k] = vals[0]
    if len(outs) > 1:
        res["acc_sd"] = float(np.std([o["acc"] for o in outs]))
    res["seconds"] = round(time.time() - t0, 1)
    return res


# ---- search state ----------------------------------------------------------------------


def key(config: dict) -> str:
    return json.dumps(config, sort_keys=True, default=str)


def diff(config: dict) -> str:
    parts = [] if config["view"] == DEFAULT_CONFIG["view"] else [config["view"]]
    parts += [f"{k}={config[k]}" for k in config if k != "view" and config[k] != DEFAULT_CONFIG[k]]
    return ", ".join(parts) or "defaults"


def fmt(v) -> str:
    return "off" if v is None else str(v)


def new_state(data: Data) -> dict:
    d = data.dev
    y = data.gold[d]
    mv = data.majority > 0.5
    ds = dices.dawid_skene(data.raters_V)
    ths = np.unique(data.majority[d])
    t = ths[int(np.argmax([((data.majority[d] > t) == y).mean() for t in ths]))]
    baselines = {
        "Majority vote": metrics(data.majority[d], mv[d], y),
        "Dawid–Skene": metrics(ds[d], ds[d] > 0.5, y),
        "Crowd share, threshold tuned on dev (ceiling)": metrics(data.majority[d], data.majority[d] > t, y),
    }
    return {"incumbent": dict(DEFAULT_CONFIG), "incumbent_metrics": None, "pass": 1,
            "step_index": 0, "pass_adopted": 0, "runs": [], "steps": [], "baselines": baselines,
            "done": False, "final": None, "n_dev": int(d.sum()), "n_test": int((~d).sum())}


def steps_for_pass(p: int) -> list:
    return STEPS


def plan_values(state: dict, step) -> list:
    stage, param, values, when = step
    inc = state["incumbent"]
    if when is not None and not when(inc):
        return []
    if param == "cluster_threshold" and stage == 2:
        # On/off switch: "on" means the current threshold if one is set, else 0.5.
        on = inc["cluster_threshold"] if inc["cluster_threshold"] is not None else 0.5
        return [None, on]
    return values


def run_step(data: Data, state: dict) -> bool:
    """Run one step. Returns False when the search is finished."""
    if state["done"]:
        return False
    cache = {r["key"]: r for r in state["runs"] if r.get("status") != "queued"}
    if state["incumbent_metrics"] is None:
        m = evaluate(data, state["incumbent"])
        state["incumbent_metrics"] = m
        state["runs"].append(run_row(state, 0, "start", "start", "defaults", state["incumbent"], m, "incumbent"))
        cache[state["runs"][-1]["key"]] = state["runs"][-1]

    steps = steps_for_pass(state["pass"])
    step = steps[state["step_index"]]
    stage, param, _, _ = step
    values = plan_values(state, step)
    inc, inc_m = state["incumbent"], state["incumbent_metrics"]
    step_id = f"P{state['pass']}·{state['step_index'] + 1:02d}"
    rows = []
    for v in values:
        cfg = {**inc, param: v}
        k = key(cfg)
        if k in cache and cache[k].get("dev_acc") is not None:
            m = {"acc": cache[k]["dev_acc"], "auc": cache[k]["dev_auc"], "bal_acc": cache[k]["dev_bal_acc"],
                 "pred_prev": cache[k]["dev_pred_prev"], "est_prev": cache[k]["est_prev"], "tci": cache[k]["tci"],
                 "n_kept": cache[k]["n_kept"], "n_verifiers": cache[k]["n_verifiers"], "fallback": cache[k]["fallback"],
                 "seconds": 0.0}
            reused = True
        else:
            m = evaluate(data, cfg)
            reused = False
        eligible = VIEWS[cfg["view"]].get("eligible", True)
        row = run_row(state, stage, step_id, param, fmt(v), cfg, m, "control" if not eligible else "tried")
        row["reused"] = reused
        rows.append((v, m, eligible, row))
        print(f"  {step_id} {param}={fmt(v):<28} acc {m['acc']:.1f}  auc {m['auc']:.1f}  "
              f"pred {m['pred_prev']:.0f}%  est {m['est_prev'] if m['est_prev'] is None else round(m['est_prev'], 2)}"
              f"  ({m['seconds']}s)", flush=True)

    cands = [(v, m, row) for v, m, e, row in rows if e and v != inc[param]]
    decision, winner = "kept", None
    if cands:
        v, m, row = max(cands, key=lambda c: (c[1]["acc"], c[1]["auc"]))
        if m["acc"] >= inc_m["acc"] + GAIN and m["auc"] >= inc_m["auc"] - AUC_TOL:
            decision, winner = "adopted", (v, m, row)
    for v, m, e, row in rows:
        if v == inc[param]:
            row["decision"] = "incumbent"
        elif winner and row is winner[2]:
            row["decision"] = "adopted"
        elif e:
            row["decision"] = "rejected"
        else:
            row["decision"] = "control"
        row["run"] = len(state["runs"]) + 1
        state["runs"].append(row)

    best = max((r for r in rows if r[2]), key=lambda c: (c[1]["acc"], c[1]["auc"]), default=None)
    state["steps"].append({
        "step": step_id, "pass": state["pass"], "stage": stage, "stage_name": STAGE_NAME[stage], "param": param,
        "n_values": len(values), "skipped": not values,
        "before": fmt(inc[param]), "before_acc": inc_m["acc"], "before_auc": inc_m["auc"],
        "best_value": None if best is None else fmt(best[0]), "best_acc": None if best is None else best[1]["acc"],
        "decision": "skipped (not applicable)" if not values else decision,
        "after": fmt(winner[0]) if winner else fmt(inc[param]),
    })
    if winner:
        state["incumbent"] = {**inc, param: winner[0]}
        state["incumbent_metrics"] = winner[1]
        state["pass_adopted"] += 1
        print(f"  -> adopted {param}={fmt(winner[0])}: dev acc {inc_m['acc']:.1f} -> {winner[1]['acc']:.1f}")
    elif values:
        print(f"  -> kept {param}={fmt(inc[param])}")

    state["step_index"] += 1
    if state["step_index"] >= len(steps):
        if state["pass_adopted"] == 0 or state["pass"] >= MAX_PASSES:
            state["done"] = True
        else:
            state["pass"] += 1
            state["step_index"] = 0
            state["pass_adopted"] = 0
    return not state["done"]


def run_row(state, stage, step_id, param, value, cfg, m, decision) -> dict:
    return {
        "run": len(state["runs"]) + 1, "pass": state["pass"], "stage": stage,
        "stage_name": STAGE_NAME.get(stage, "Start"), "step": step_id, "param": param, "value": value,
        "view": cfg["view"], "config": diff(cfg), "key": key(cfg), "status": "done", "decision": decision,
        "dev_acc": round(m["acc"], 2), "dev_auc": round(m["auc"], 2), "dev_bal_acc": round(m["bal_acc"], 2),
        "dev_pred_prev": round(m["pred_prev"], 1),
        "est_prev": None if m["est_prev"] is None else round(m["est_prev"], 3),
        "tci": None if m["tci"] is None else float(f"{m['tci']:.4g}"),
        "n_kept": m["n_kept"], "n_verifiers": m["n_verifiers"], "fallback": m["fallback"],
        "acc_sd": round(m["acc_sd"], 2) if "acc_sd" in m else None, "seconds": m["seconds"],
    }


# ---- final -----------------------------------------------------------------------------


def paired(pred, ref, y, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = (pred == y).astype(float), (ref == y).astype(float)
    idx = rng.integers(0, len(y), (n, len(y)))
    d = 100 * (a[idx].mean(1) - b[idx].mean(1))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"diff": round(100 * (a.mean() - b.mean()), 1), "lo": round(float(lo), 1), "hi": round(float(hi), 1)}


def final(data: Data, state: dict) -> None:
    inc = state["incumbent"]
    # Seed stability: only the triplet sample depends on the seed.
    seed_rows = []
    for s in range(5):
        m = evaluate(data, {**inc, "seed": s})
        seed_rows.append({"seed": s, "dev_acc": round(m["acc"], 2), "dev_auc": round(m["auc"], 2)})
        print(f"  seed {s}: dev acc {m['acc']:.1f}")
    t = ~data.dev
    y = data.gold[t]
    mv = data.majority > 0.5
    ds = dices.dawid_skene(data.raters_V)
    default = predict(data, dict(DEFAULT_CONFIG))
    rg = predict(data, {**DEFAULT_CONFIG, "view": "groups: race×gender"})
    tuned = predict(data, inc)
    methods = {
        "Tuned FUSE": (tuned["scores"], tuned["pred"]),
        "FUSE defaults, per rater": (default["scores"], default["pred"]),
        "FUSE defaults, race×gender groups": (rg["scores"], rg["pred"]),
        "Majority vote": (data.majority, mv),
        "Dawid–Skene": (ds, ds > 0.5),
    }
    rows = []
    for name, (s, p) in methods.items():
        r = {"method": name, **{f"test_{k}": round(v, 2) for k, v in metrics(s[t], p[t], y).items()}}
        if name != "Majority vote":
            c = paired(p[t], mv[t], y)
            r.update({"vs_mv": c["diff"], "vs_mv_lo": c["lo"], "vs_mv_hi": c["hi"]})
        if name != "Tuned FUSE":
            c = paired(tuned["pred"][t], p[t], y)
            r.update({"tuned_minus": c["diff"], "tuned_minus_lo": c["lo"], "tuned_minus_hi": c["hi"]})
        rows.append(r)
        print(f"  test {name:<36} acc {r['test_acc']:.1f}  auc {r['test_auc']:.1f}")
    state["final"] = {"config": diff(inc), "seeds": seed_rows, "test": rows,
                      "seed_sd": round(float(np.std([r["dev_acc"] for r in seed_rows])), 2)}


# ---- output ----------------------------------------------------------------------------


def label_free_check(state: dict) -> list[dict]:
    """Within each step, does the TCI statistic rank values like dev accuracy does?"""
    from scipy.stats import spearmanr

    out = []
    for st in state["steps"]:
        rs = [r for r in state["runs"] if r["step"] == st["step"] and r["tci"] is not None]
        if len(rs) >= 3 and st["param"] != "view":
            rho = spearmanr([-r["tci"] for r in rs], [r["dev_acc"] for r in rs]).statistic
            out.append({"step": st["step"], "param": st["param"], "rho": None if np.isnan(rho) else round(float(rho), 2)})
    return out


def export(state: dict, path: Path) -> None:
    """One JSON file for the live dashboard: each top-level key is one table of rows."""
    runs = [{k: v for k, v in r.items() if k != "key"} for r in state["runs"]]
    base = [{"method": k, **{f"dev_{kk}": round(vv, 2) for kk, vv in v.items()}} for k, v in state["baselines"].items()]
    start = state["runs"][0] if state["runs"] else None
    inc_m = state["incumbent_metrics"] or {}
    if state["final"]:
        phase = "finished"
    elif state["done"]:
        phase = "search finished, test pending"
    else:
        phase = "searching"
    nxt = None if state["done"] else STEPS[state["step_index"]]
    status = [{
        "phase": phase, "pass": state["pass"],
        "next_step": None if nxt is None else f"P{state['pass']}·{state['step_index'] + 1:02d}",
        "next_setting": None if nxt is None else nxt[1],
        "next_stage": None if nxt is None else STAGE_NAME[nxt[0]],
        "steps_in_pass": len(STEPS), "step_in_pass": None if nxt is None else state["step_index"] + 1,
        "best_config": diff(state["incumbent"]),
        "best_acc": None if not inc_m else round(inc_m["acc"], 2),
        "best_auc": None if not inc_m else round(inc_m["auc"], 2),
        "start_acc": None if start is None else start["dev_acc"],
        "gain": None if not inc_m or start is None else round(inc_m["acc"] - start["dev_acc"], 2),
        "runs_done": len(state["runs"]), "steps_done": len(state["steps"]),
        "n_dev": state["n_dev"], "n_test": state["n_test"], "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }]
    fin = state["final"] or {}
    out = {
        "runs": runs, "steps": state["steps"], "baselines": base, "status": status,
        "test": fin.get("test", []), "seeds": fin.get("seeds", []),
    }
    path.write_text(json.dumps(out, default=str))


def report(state: dict) -> None:
    L = ["# FUSE tuning on DICES-350 (generated by scripts/tune_dices350.py)", "",
         f"Dev half {state['n_dev']} conversations (choosing), test half {state['n_test']} (scored once). "
         f"A change is adopted when dev accuracy gains at least {GAIN} points and dev AUC drops at most {AUC_TOL}.", "",
         "## Baselines on dev", "", "| Method | Accuracy | AUC | Predicted unsafe % |", "|---|---:|---:|---:|"]
    for k, v in state["baselines"].items():
        L.append(f"| {k} | {v['acc']:.1f} | {v['auc']:.1f} | {v['pred_prev']:.0f} |")
    L += ["", "## Steps", "", "| Step | Setting | Before | Dev acc before | Best value | Best dev acc | Decision |",
          "|---|---|---|---:|---|---:|---|"]
    for s in state["steps"]:
        ba = "" if s["best_acc"] is None else f"{s['best_acc']:.1f}"
        L.append(f"| {s['step']} | {s['param']} | {s['before']} | {s['before_acc']:.1f} | {s['best_value'] or ''} | {ba} | {s['decision']} |")
    L += ["", "## Every run", "", "| Run | Step | Setting = value | Config | Dev acc | Dev AUC | Pred. unsafe % | Est. prevalence | Decision |",
          "|---:|---|---|---|---:|---:|---:|---:|---|"]
    for r in state["runs"]:
        est = "" if r["est_prev"] is None else f"{r['est_prev']:.2f}"
        sd = f" ± {r['acc_sd']:.1f}" if r.get("acc_sd") else ""
        L.append(f"| {r['run']} | {r['step']} | {r['param']} = {r['value']} | {r['config']} | {r['dev_acc']:.1f}{sd} | "
                 f"{r['dev_auc']:.1f} | {r['dev_pred_prev']:.0f} | {est} | {r['decision']} |")
    lf = label_free_check(state)
    if lf:
        L += ["", "## Label-free check: does a lower TCI statistic go with higher dev accuracy within a step?", "",
              "| Step | Setting | Spearman (−TCI, dev acc) |", "|---|---|---:|"]
        L += [f"| {x['step']} | {x['param']} | {'' if x['rho'] is None else x['rho']} |" for x in lf]
    if state["final"]:
        f = state["final"]
        L += ["", f"## Final: `{f['config']}`", "",
              f"Seed stability on dev (5 seeds): sd {f['seed_sd']:.2f} points "
              f"({', '.join(f'{r['dev_acc']:.1f}' for r in f['seeds'])}).", "",
              "| Method (test half) | Accuracy | AUC | Pred. unsafe % | − Majority vote (95% CI) | Tuned − this (95% CI) |",
              "|---|---:|---:|---:|---|---|"]
        for r in f["test"]:
            mvc = "" if "vs_mv" not in r else f"{r['vs_mv']:+.1f} [{r['vs_mv_lo']:+.1f}, {r['vs_mv_hi']:+.1f}]"
            tm = "" if "tuned_minus" not in r else f"{r['tuned_minus']:+.1f} [{r['tuned_minus_lo']:+.1f}, {r['tuned_minus_hi']:+.1f}]"
            L.append(f"| {r['method']} | {r['test_acc']:.1f} | {r['test_auc']:.1f} | {r['test_pred_prev']:.0f} | {mvc} | {tm} |")
    REPORT.write_text("\n".join(L) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["next", "stage", "all", "final", "report", "reset"])
    p.add_argument("--export", type=Path, default=None)
    a = p.parse_args()
    if a.cmd == "reset":
        STATE.unlink(missing_ok=True)
        return
    data = Data()
    state = json.loads(STATE.read_text()) if STATE.exists() else new_state(data)
    if a.cmd == "next":
        run_step(data, state)
    elif a.cmd == "stage":
        stage = (state["pass"], STEPS[state["step_index"]][0])
        while run_step(data, state) and (state["pass"], STEPS[state["step_index"]][0]) == stage:
            pass
    elif a.cmd == "all":
        while run_step(data, state):
            pass
    elif a.cmd == "final":
        final(data, state)
    STATE.parent.mkdir(exist_ok=True)
    STATE.write_text(json.dumps(state, indent=1, default=str))
    if a.export:
        export(state, a.export)
    report(state)


if __name__ == "__main__":
    main()
