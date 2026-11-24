# FUSE depends on the order of the rater columns (24 Nov 2026)

Tables: [results/FUSE_ORDER.md](../results/FUSE_ORDER.md),
[results/FUSE_ORDER_SORTED.md](../results/FUSE_ORDER_SORTED.md) · scripts:
[scripts/run_fuse_order.py](../scripts/run_fuse_order.py),
[scripts/run_fuse_order_sorted.py](../scripts/run_fuse_order_sorted.py).

**How we found it.** In a simulation of soft blocks (LLM raters whose errors are only partly shared),
FUSE beat every baseline by 2–5 AUC points with 5 humans and 6 weak LLMs. Running "detector + FUSE"
with nothing merged should have reproduced FUSE exactly, and it did not: the detector returns the
columns in a different order. FUSE's threshold search (coordinate descent on TCI) visits the raters
one at a time, so the same panel in a different column order can end with different thresholds and a
different AUC. Every FUSE number we reported before used one fixed order: humans first, then LLMs.

## Findings

1. **On panels with many weak raters, the order moves FUSE by 5–14 AUC points.** Over 8 random orders
   of the same simulated panel, max minus min AUC has a median of 5–14 points. On real panels it is
   1–2.6 points on DICES, about 1 on RTE with 10 humans, and 5–7 on RTE with 3–5 humans.
2. **The simulation win came from the order.** Humans first was 3–9 points better than a random order.
   With a random order FUSE trails the best baseline in every simulated setting.
3. **The real-data wins mostly survive.** On DICES, FUSE with a random order still beats mean z and
   Dawid–Skene: 74.6 vs 72.6 and 71.1 with 10 humans + 8 LLMs (ahead of mean z in 20/20 draws).
   With 5 humans + 8 LLMs, order-averaged FUSE (72.0) beats the best of four baselines (mean z, 70.6)
   by 1.4 ± 0.6, the only panel where it beats all four with an interval that excludes zero. "Detector + D&S" ties FUSE with
   10 humans (74.6) and falls to 67.4 with 5. On RTE with 10 humans + 4 tiny Qwen, the gain over mean z
   shrinks from 0.8 to 0.5, and "detector + D&S" is ahead (96.7 vs 96.4). With 5 humans, FUSE drops
   from 92.1 (humans first) to 90.4 (random order), below mean z (92.3) and "detector + D&S" (93.6).
4. **Three label-free fixes do not help.**
   - Averaging FUSE's scores over 8 orders recovers 0–2 points, and usually under 1.
   - Sorting the columns by the detector's robust quality estimate, strongest first, matches the
     humans-first order when there is no block. With a soft block it loses 5–10 points, because the
     correlated LLMs get high quality estimates and are put first.
   - Choosing the order with the lowest TCI does not help either: across orders TCI and AUC are
     unrelated (mean Spearman −0.3 to +0.1). That run is not in the scripts above.

## What it means

The order sensitivity comes from the paper's coordinate-descent threshold search, which updates one
rater at a time. We cannot check whether the authors' own code behaves the same; it is not released.
Their Best-of-N verifiers are stronger, and in our runs the spread is small on strong panels (about 1
point on RTE with 10 humans), so the effect may be smaller there. It is largest on panels with many weak
raters, which is the setting of this project. The robust results so far: FUSE beats plain Dawid–Skene
and mean z on DICES mixed panels in any order, and the detector followed by Dawid–Skene matches it
with 10 humans.

## Caveats

- 8 orders per panel; the spread would grow with more.
- Simulation: 10 seeds per setting. Real data: 20 draws of humans, same items in every draw.
- "Best baseline" is chosen after the fact, which favours the baselines.
