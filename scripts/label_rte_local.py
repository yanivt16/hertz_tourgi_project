"""Score the 800 RTE pairs (Snow et al. 2008) with local open-weight LLMs as extra verifiers.

Same setup as scripts/label_dices_local.py: each (model, prompt) pair is one LLM "rater",
score = log P(Yes) - log P(No) for "does the text entail the hypothesis?". No prompt was tuned
on RTE, so every item is a clean test item.

    python scripts/label_rte_local.py --model mlx-community/Qwen3-8B-4bit --prompt short
    python scripts/label_rte_local.py --model ... --prompt guide --limit 20 --check

Output: results/llm_scores/rte_<model>_<prompt>.csv (item_id = orig_id, p_yes, logit_yes,
p_yes_or_no).
"""

from __future__ import annotations

import argparse
import html
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fuse import crowd  # noqa: E402
from local_llm import YesNoScorer  # noqa: E402

OUT = ROOT / "results" / "llm_scores"
ANSWER = "Answer with exactly one word: Yes or No."
PROMPTS = {
    # one-line definition
    "short": ("You will see a text and a hypothesis. Does the text entail the hypothesis, that is, "
              "would a typical person who reads the text conclude that the hypothesis is most likely true? "
              + ANSWER),
    # paraphrase of the PASCAL RTE annotation guidelines
    "guide": ("You are annotating textual entailment.\n"
              "The text T entails the hypothesis H if a typical human reading T would infer that H is "
              "most likely true. Rules:\n"
              "- Use T together with common background knowledge; do not rely on facts that are not in T "
              "and are not common knowledge.\n"
              "- H must be supported by T. If H adds information that T does not state or clearly imply, "
              "the answer is No.\n"
              "- If H contradicts T, or is merely plausible or possible given T, the answer is No.\n"
              "- Small differences in wording, tense or paraphrase do not matter.\n\n"
              "Does T entail H? " + ANSWER),
}


def pairs() -> pd.DataFrame:
    df = pd.read_csv(crowd._fetch(crowd.RTE_URL, crowd.CACHE / "rte.standardized.csv"))
    return df.groupby("orig_id")[["text", "hypothesis", "gold"]].first().sort_index()


def user_message(text: str, hypothesis: str) -> str:
    return (f"Text: {html.unescape(text)}\n\nHypothesis: {html.unescape(hypothesis)}\n\n"
            "Does the text entail the hypothesis? Answer Yes or No.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="short", choices=list(PROMPTS))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--check", action="store_true", help="compare cached-prefix logits with a full forward pass")
    args = ap.parse_args()

    items = pairs()
    if args.limit:
        items = items.iloc[: args.limit]
    sc = YesNoScorer(args.model, PROMPTS[args.prompt])
    print(f"{args.model} / {args.prompt}: {len(items)} items, prefix {sc.n_prefix} tokens", flush=True)
    if args.check:
        print(sc.check(user_message(items.iloc[0].text, items.iloc[0].hypothesis)), flush=True)

    rows, t0 = [], time.time()
    for k, (iid, row) in enumerate(items.iterrows()):
        ly, ln = sc.score(user_message(row.text, row.hypothesis))
        rows.append({"item_id": iid, "p_yes": 1 / (1 + np.exp(ln - ly)), "logit_yes": ly - ln,
                     "p_yes_or_no": float(np.exp(ly) + np.exp(ln))})
        if k % 100 == 0:
            print(f"  {k}/{len(items)} {time.time() - t0:.0f}s", flush=True)

    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"rte_{args.model.split('/')[-1]}_{args.prompt}{'_limit' + str(args.limit) if args.limit else ''}.csv"
    out.to_csv(path, index=False)
    g = items.gold.to_numpy() == 1
    msg = f"wrote {path.name}: {len(out)} items in {time.time() - t0:.0f}s, mean P(Yes|No mass) {out.p_yes_or_no.mean():.2f}"
    if 0 < g.sum() < len(g):
        from sklearn.metrics import roc_auc_score
        msg += f", AUC vs gold {100 * roc_auc_score(g, out.logit_yes):.1f}, acc@0.5 {100 * np.mean((out.p_yes > .5) == g):.1f}"
    print(msg, flush=True)


if __name__ == "__main__":
    main()
