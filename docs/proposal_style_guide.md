# Research-Proposal Style Guide (MLDS Project Proposal format)

Derived from the "MLDS Projects" Drive folder (Efi Arazi School of CS, MLDS M.Sc.,
Reichman University): 37 PDFs, Mar–Oct 2026. Eight were read in full, spanning the
whole range from 2-page stubs to the 4-page flagship proposals; the rest were
identified by title and mentor only, so the rules reflect the sample, not all 37. These are my working rules for writing such a
document.

## 1. What the genre is

A proposal is a **short, mentor-authored offer of a research problem to M.Sc. students**.
It is not a paper. It must let a student decide in 5 minutes whether to take it, and
let a mentor judge in 5 minutes whether the scope is realistic. It states the problem,
the scientific hook, the concrete work, and what "done" looks like. Results are never
reported; only methods, controls and deliverables.

## 2. Observed formats (pick one, don't mix)

| Tier | Length | Seen in | Use when |
|---|---|---|---|
| **A. Stub** | 1–2 pp | clinical/applied proposals (CME, melanoma, lung audio) | Problem and data are given; method is open. Background + one objective + resources. |
| **B. Standard** | 3 pp | Kipnis, Assa, Caplan, Galili-Log-rank | Method idea exists. Background (+ worked example or statistical concept) → objectives → requirements → resources → refs. |
| **C. Flagship** | 4 pp | Backtest-Overfitting, Transfer-Success, Player-DNA, PPG-to-Hormones | Full study design. Adds preliminaries/notation, core-vs-extension split, evaluation discipline, deliverables list. |

Default for anything new that is research-heavy: **Tier C**, ~3.5–4 pages, 2-column
layout is NOT used; proposals are single-column, ~11 pt, with a running header.

## 3. Fixed skeleton

```
Header (every page):  <School> — <Program>            "MLDS Project Proposal  Page N"
Title                 "MLDS Project Proposal <Descriptive title>"   (a claim or question, not a topic)
Date: <Month YYYY>
Mentors: <Dr./Prof. names; clinical/industry mentors with affiliation>

Background
  [Preliminaries]            (Tier C: notation, definitions, formulas, labels)
  [Statistical / Scientific concept]   (Tier B: the one idea the project rests on)
  [Core: ...]                (Tier C: numbered movements a)–h)
  [Extension: ...]           (Tier C: the higher-risk half, gated)
  Research question          (one sentence, in bold or its own line)

Objectives & Deliverables
  Objectives    (numbered or bulleted; "Build / Formalize / Quantify / Compare")
  Deliverables  (code, benchmark, notebook, report)

Requirements   (courses, languages, workload e.g. "1–2 days/week for an academic year")
Resources      (data, compute, access, IP clause)
References     (numbered, full citations, DOI/URL)
```

Rules: use exactly these section names; "Requirements" is optional in Tier C,
"Intellectual Property" is included only for clinical/industry data; Figures are
optional and used only when they carry the design (network map, trade-off curve,
pipeline diagram), always captioned "Figure N. (A) … (B) …".

## 4. Depth rules (what separates Tier C from a stub)

1. **Open with a stake, not a definition.** First sentence is the practitioner's
   question or a widely held belief the project will test ("Find me a player like this
   one, but cheaper"; "good backtest results are taken as evidence...").
2. **State why the naïve approach fails** in 2–3 specific mechanisms, each with a
   number or formula where possible (e.g. E[max SR] ≈ √(2 log N); SE of Sharpe;
   3 years of daily data CI includes 0).
3. **Define the object before analysing it.** Preliminaries fix notation, the unit of
   analysis, the labels/outcomes, and inclusion rules *before* any result. Pre-registered
   mappings (axis → columns) are listed explicitly.
4. **One research question**, falsifiable, phrased "To what extent… / Does… / Which…".
5. **Core vs. extension.** The core must be doable and valuable on its own. The
   extension is the high-risk half, opens only after a stated checkpoint, and a null
   result is declared in advance to be valid.
6. **Tiered analyses.** Label each movement mandatory / secondary / exploratory.
   Exploratory items are "reported as an open problem, not a result."
7. **Evaluation discipline paragraph.** Name the ways the result could look better than
   it is (leakage, circularity, multiple testing, selection bias, survivorship,
   confounding) and the control for each. Observational data → "association, not causal."
8. **Baselines are honest.** Always include the transparent floor ("the honest floor
   rather than a strawman"), and make the proposed method beat it jointly, not alone.
9. **Proper scoring and out-of-sample protocol.** Name the metrics (Brier, log loss,
   calibration, RMSE/MAE, size and power) and the split (grouped, time-respecting,
   context-held-out) in the proposal itself.
10. **Worked toy example** (Tier B) when the contribution is a phenomenon: a 4–5 row
    table where the claim can be verified by hand (FFSI example, accuracy matrix).

## 4b. Voice and style

- Third person / "we", present and future tense ("The project will…", "This is deliberately the higher-risk half").
- Dense but plain. No hype words (novel, groundbreaking) except where literally true.
- Hedge exactly once and precisely: "the sporting and financial definitions can be redefined".
- Every claim of effect size or bound is either cited or derived on the page.
- State limitations inside the sentence that introduces the thing ("with unsold players censored rather than counted as failures").
- Terms of art in italics/bold at first definition only; consistent symbol set (x_i, z_c, λ).
- Lists: objectives are verbs; deliverables are nouns (artefacts you can hand over).

## 5. Objectives & Deliverables rules

- 4–6 objectives. First = infrastructure/formalization, middle = the central
  contribution (say so: "This comparison is the project's central contribution"),
  last = real-data illustration or extension.
- Deliverables are concrete artefacts: pipeline, library with unit tests, benchmark
  table, instructional notebook, final report. The **final report is always last and
  states results "falsifiably"** (which signal is real, which is artefact).
- "The work can include several components out of the following list; the student is
  expected to choose one or two" is acceptable for Tier B open-ended directions.

## 6. Resources, requirements, IP

- Resources: data source + access route + compute (laptop / single GPU / secured cloud).
  If data is proprietary add the secured-environment sentence and IP clause
  ("Academic project. Any developed IP will be addressed when such occurs.").
- Requirements: courses (Advanced Statistics, ML), languages (Python), workload.
- Only cite what is real. Free fallbacks for gated data are listed with their known bias.

## 7. References

Numbered, full-form, DOI where available. 3–6 for stubs, 6–12 for Tier C: foundational
theory, the method papers, the dataset paper, one or two recent works. Never cite a
paper I have not verified exists.

## 8. Self-check before delivering

- [ ] Title is a claim or question; Date and Mentors present
- [ ] First paragraph states a stake; naïve-approach failure given with numbers
- [ ] Notation and labels defined before use; inclusion rules stated
- [ ] Exactly one research question
- [ ] Core vs. extension separated; checkpoint and null-result clause present
- [ ] Evaluation-discipline paragraph with a control per risk
- [ ] Baseline floor + proper metrics + split protocol named
- [ ] Objectives = verbs, deliverables = artefacts, final report last
- [ ] Resources/requirements/IP consistent with the data
- [ ] Every reference verified; every number derivable
- [ ] Length within tier (≤ 4 pp), running header and page numbers

## 9. LaTeX conventions (when produced as .tex)

`article`, 11pt, 1 column, `geometry` margin 2.2 cm, `fancyhdr` header
"Efi Arazi School … / MLDS Project Proposal — Page \thepage", `amsmath`, `booktabs`,
`graphicx` for figures, `natbib` or numeric refs, section titles unnumbered
(`\section*`). Final deliverable is a PDF plus the `.tex` source.
