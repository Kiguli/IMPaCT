# IMPaCT v2.0 — Local Issue Tracker

Lightweight, file-based issue tracking (no GitHub remote required for now). One
markdown file per issue under `issues/`, indexed in [`INDEX.md`](INDEX.md).
When/if this repo gets a GitHub remote, these can be migrated to GitHub Issues
(the frontmatter maps cleanly to labels/state).

## Filing an issue
1. Copy [`TEMPLATE.md`](TEMPLATE.md) to `issues/NNNN-short-slug.md` (next free
   4-digit id).
2. Fill the frontmatter and body.
3. Add a one-line row to [`INDEX.md`](INDEX.md).
4. Commit.

## Frontmatter fields
- `id`: ISSUE-NNNN
- `title`: short summary
- `status`: `open` | `in-progress` | `resolved` | `wontfix` | `under-verification`
- `severity`: `low` | `medium` | `high` | `critical`
- `labels`: comma list (see below)
- `created` / `updated`: YYYY-MM-DD
- `related`: files, tests, refs (BibTeX keys from `paper/References.bib`)

## Labels
`correctness`, `soundness`, `performance`, `scalability`, `tool-v1` (issue in the
inherited v1 code), `reference`, `build`, `methodology`, and the two special
literature labels below.

## Literature-counterexample protocol (READ BEFORE using these labels)
Claiming a counterexample to a **published** method is a strong, falsifiable
claim and is *usually wrong* (the paper is right; our reading/transcription is
wrong). To avoid crying wolf, any such candidate is tracked with escalating
labels and may NOT be called confirmed until every box is checked:

- `candidate-literature-counterexample` — a *suspected* failure of a published
  method. Requires, before escalation:
  1. **Exact citation** — the specific paper + algorithm/theorem number + the
     pseudocode/statement transcribed verbatim (with DOI in `paper/References.bib`).
  2. **Minimal reproducible counterexample** — smallest instance, fully spelled out.
  3. **Independent verification** — the failure shown by a method independent of
     our implementation (e.g. definition-based brute force, a second tool).
  4. **Adversarial review** — an independent agent/person tries to *refute* the
     claim (most often by finding that we mis-transcribed the algorithm).
- `confirmed-literature-counterexample` — only after all four pass AND the
  refutation attempt fails. This is paper-worthy; notify the user.
- `naive-strawman` — the failing rule is a simplification *we* invented, not what
  the paper actually states. (This is the common, honest outcome — record it so
  the trap is documented, but it is NOT a counterexample to the literature.)
- `our-bug` — the failure is in our implementation, not the method.
- `misreading` — we mis-stated the published method; corrected.

Default classification for a new candidate is `candidate-literature-counterexample`;
downgrade to `naive-strawman` / `our-bug` / `misreading` or upgrade to
`confirmed-literature-counterexample` only with the evidence above.
