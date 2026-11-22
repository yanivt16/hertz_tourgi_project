"""The threshold-search variants of scripts/tci_fix_sim.py on every real panel we have.

    python scripts/tci_fix_real.py   # writes results/tci_fix_real.json; VIPER needs `pip install rdata`

RTE human slots with blocks of small Qwen models or strong 8B-12B LLMs (10 slot shuffles), VIPER dense
blocks, DICES-350 (granular and demographic groups) and the two hard crowds as demographic groups.
"""
import sys, warnings, json, time
warnings.filterwarnings("ignore")
from pathlib import Path
R = str(Path(__file__).resolve().parents[1]); sys.path.insert(0, R + "/src"); sys.path.insert(0, R + "/scripts")
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score as auc
from fuse import FUSE, crowd, dices, viper, hard_crowds
from tci_fix_sim import VARIANTS, refs  # noqa: same variants and baselines as the simulation

DRAWS = 10
ROOT = R


def llm(model, prompt):
    return pd.read_csv(f"{ROOT}/results/llm_scores/rte_{model}_{prompt}.csv").sort_values("item_id").logit_yes.to_numpy()


def slots(H, rng):
    out = np.empty((len(H), 10))
    for i in range(len(H)):
        v = H[i, ~np.isnan(H[i])]
        out[i] = rng.permutation(v)[:10]
    return out


def all_methods(V):
    out = dict(refs(V))
    for k, kw in VARIANTS.items():
        out["FUSE " + k] = FUSE(**kw).fit_scores(V).scores
    return out


def evaluate(name, mats, y):
    t = time.time(); r = {}
    for V in mats:
        for k, s in all_methods(V).items():
            r.setdefault(k, []).append(auc(y, s))
    res[name] = {k: list(map(float, v)) for k, v in r.items()}
    print(f"## {name} ({time.time()-t:.0f}s)", flush=True)
    for k, v in r.items():
        print(f"  {k:32s} {100*np.mean(v):6.1f}", flush=True)


res = {}
import os
if not os.environ.get("SKIP_RTE"):
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
      evaluate(name, [np.c_[slots(H, np.random.default_rng(d))[:, :k], L] for d in range(DRAWS)], y)
  evaluate("RTE 8 small Qwen only", [np.column_stack([X[m] for m in SMALL])], y)

# VIPER (dense blocks), DICES-350, hard crowds (demographic groups)
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
    G = G[:, ~np.isnan(G).any(0)]
    evaluate(f"{name} demographic groups", [G], np.asarray(yg).astype(int))
json.dump(res, open(R + "/results/tci_fix_real.json", "w"))
