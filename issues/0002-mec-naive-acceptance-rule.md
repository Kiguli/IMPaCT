---
id: ISSUE-0002
title: MEC — naive "staying-action-exists" acceptance rule is unsound
status: resolved
severity: low
labels: methodology, naive-strawman
created: 2026-06-25
updated: 2026-06-25
related:
  - src/graph_utils.cpp
  - tests/unit/test_graph.cpp
  - deAlfaro1997formal
  - ChatterjeeHenzinger2011
  - WeiningerMeggendorferKretinsky2019BMDP
---

## Summary
While implementing MEC decomposition (Phase 1b) a tempting *naive* acceptance rule
was identified as INCORRECT, with a minimal counterexample. Honest classification:
this is a counterexample to a **naive simplification we considered**, NOT (so far
as established) to any published algorithm. Tracked under the literature-
counterexample protocol so the trap and its resolution are documented. The
implemented algorithm avoids it.

## The naive (incorrect) rule
"Compute SCCs of the full set's staying-action graph once. Accept an SCC C as a MEC
if every state in C has *some* action whose successors stay in C (i.e. the set of
states with no staying action is empty)."

## Minimal reproducible counterexample
MDP, states {0,1,2}: `0: a->{1,2}, b->{0}` ; `1: a->{0}` ; `2: a->{2}`.
- In the full staying-action graph the edge `0->1` exists (via `a`), so `{0,1}` is
  an SCC. State 0 has `b` staying in `{0,1}`, state 1 has `a` staying in `{0,1}`, so
  the naive "no state lacks a staying action" test holds -> naive rule accepts `{0,1}`.
- But `{0,1}` is NOT an end component: 0's only `{0,1}`-staying action is the
  self-loop `b`, so 0 cannot reach 1 inside `{0,1}` -> not strongly connected.
- True MECs: `{0}` (via self-loop `b`) and `{2}`.

## Independent verification
- Definition-based brute-force oracle (`tests/unit/test_graph.cpp` `brute_mecs`)
  returns `{0},{2}`; a dedicated test pins this case.
- Randomized differential test: `mecs()` == `brute_mecs()` on 1500 random MDPs.
- The implemented algorithm recomputes SCCs under each candidate's OWN staying
  actions and accepts only single-SCC fixpoints, so it returns `{0},{2}` correctly.

## Cited method (VERIFIED — primary sources read directly)
- **Baier & Katoen, *Principles of Model Checking* (MIT Press, 2008), Algorithm 47
  (pp. 878–879), Def. 10.116/10.117 (p. 870), Notation 10.124 (p. 876), Lemma 10.126.**
  The algorithm recomputes the nontrivial SCCs of G_(T, A|T) — the digraph induced by
  the CURRENT candidate restricted to its surviving (staying) actions — prunes actions
  whose successors leave the freshly-computed SCC (`A(s) := {α | Post(s,α) ⊆ Ti}`),
  cascades state removal via `Pre`, and accepts a candidate as a MEC only at a FIXPOINT
  (`repeat … until MEC = MECnew`). p. 877 states explicitly: "Due to the removal of
  actions … the strong connectivity of Ti … might be lost. We therefore have to repeat
  the whole procedure." (= our `naive` rule's exact failure; the loop exists to avoid it.)
- de Alfaro, PhD thesis 1997, Sec. 3.3 (End Components); Chatterjee–Henzinger,
  JACM 61(3) 2014, Sec. 3 — same recompute-until-fixpoint structure (CH adds a faster
  lock-step/“collapse” variant). Bibkeys: deAlfaro1997formal, ChatterjeeHenzinger2011.

## Adversarial review / audit outcome
`mec-correctness-audit` workflow (agents read the Baier-Katoen PDF, de Alfaro thesis,
and CH JACM directly). Conclusion: **our example refutes ONLY the naive one-pass
strawman, NOT any published algorithm** — Baier-Katoen's own text anticipates exactly
this failure and the refinement loop prevents it. Our implementation
(`src/graph_utils.cpp::mecs`) is a clean variant equivalent to Algorithm 47:
recompute SCCs under the candidate's own staying actions; accept iff the candidate is a
single nontrivial SCC (multi-state, or singleton with a self-loop); else refine into
SCCs and re-process. (One of three adversarial-verdict agents hit a StructuredOutput
retry-cap tooling failure — not a substantive refutation; the direct primary-source
reads are conclusive.)

## Classification
**`naive-strawman`** — confirmed. NOT a literature counterexample. The published
algorithms are correct and our implementation matches them.

## Resolution
RESOLVED. (1) Implemented the correct (published) algorithm in Phase 1b. (2) Pinned the
trap with an explicit test + a definition-based brute-force differential test (1500
random MDPs). (3) Audit confirms equivalence to Baier-Katoen Alg. 47 / de Alfaro / CH.
Confidence basis: matches a proven published algorithm AND passes an independent
definition-based oracle.
