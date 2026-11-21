# Shared Claude instructions — FUSE on Human Annotators

Shared by everyone on the project (Yaniv & Ohad). Put project-wide guidelines here;
keep personal preferences in your own `~/.claude/CLAUDE.md` or `CLAUDE.local.md`.

## 1. Keep the rolling progress document up to date

The project's living record is the Google Doc
**"FUSE on Human Annotators: Progress Update"**:
https://docs.google.com/document/d/1Zsr_Y5Iax8SfreIJF5TE5SUnH0ADE3DeWreIkypeWz0/edit

It is what the mentor reads, and it is the canonical version. A Markdown mirror lives in
the repo at [docs/progress/progress_update.md](docs/progress/progress_update.md).

**Every meaningful piece of progress or valuable insight goes into this document**, in the
same session it happens. That includes:

- a new experiment result (with the numbers and where the results file is),
- a finding that changes the story (a method wins/loses, a hypothesis is refuted),
- a bug or pitfall that explains earlier numbers,
- a change of scope, plan or next steps,
- a new question for the mentor.

Routine work with nothing new to report (refactors, reruns that match) does not need an entry.

How to update:

1. Read the current Google Doc first and fit the new material into its existing structure:
   update the Summary, the relevant section, "Where this leaves FUSE", "Next steps" and
   "Questions for you" as needed. Edit numbers in place rather than appending contradictions.
2. Match the doc's style: plain, precise prose; every number traceable to a results file in
   the repo; confidence intervals where we have them; say plainly when a result is negative
   or not significant.
3. Write the change to the Google Doc (Google Docs connector), then refresh the repo mirror
   with [scripts/gdoc_to_md.py](scripts/gdoc_to_md.py), update the "Last synced" date in the
   mirror's header, and commit the mirror together with the code/results it describes.
4. If you cannot reach the Google Doc in a session, update the repo mirror and say clearly
   in your reply that the Google Doc still needs the same change.
