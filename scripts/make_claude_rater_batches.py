"""Inputs for the Claude rater agents on DICES-350 (see docs/claude_raters.md).

    python scripts/make_claude_rater_batches.py      # writes .cache/claude_raters/

Writes the v4 policy (scripts/dices_prompts/v4.txt without its answer-format paragraph) and the
350 conversations in four interleaved batches (items 0, 4, 8, ... in batch 0, and so on). Each
batch went to one agent per model (Sonnet, Haiku) with scripts/dices_prompts/claude_agent_brief.txt
(D = the output folder, K = batch number, MODEL = sonnet or haiku). Each agent read the policy and
its batch, gave every item a 0-100 probability that experts would label the final response unsafe,
and wrote item_id,p_unsafe. The merged scores are results/claude_scores/350_{sonnet,haiku}.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import label_dices_llm as L  # noqa: E402

OUT = ROOT / ".cache" / "claude_raters"


def main():
    (OUT / "batches").mkdir(parents=True, exist_ok=True)
    (OUT / "out").mkdir(exist_ok=True)
    items = L.conversations("350")
    (OUT / "policy.txt").write_text(L.PROMPTS["v4"].rsplit("\n\nAnswer with", 1)[0])
    ids = items.index.to_numpy()
    for k in range(4):
        parts = []
        for i in ids[k::4]:
            c, r = L.normalize(items.loc[i, "context"], items.loc[i, "response"])
            parts.append(f"=== ITEM {i} ===\n<conversation>\n{c}\n</conversation>\n"
                         f"<final_response>\nCHATBOT: {r}\n</final_response>\n")
        (OUT / "batches" / f"batch_{k}.txt").write_text("\n".join(parts))
        print(f"batch {k}: {len(parts)} items")


if __name__ == "__main__":
    main()
