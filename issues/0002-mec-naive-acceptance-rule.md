---
id: ISSUE-0002
title: MEC — naive "staying-action-exists" acceptance rule is unsound
status: under-verification
severity: low
labels: methodology, candidate-literature-counterexample
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

## Cited method (under verification)
The standard published MEC algorithms (de Alfaro 1997; Baier-Katoen Alg. 47;
Chatterjee-Henzinger SODA 2011 / JACM 2014) recompute SCCs under candidate-staying
actions and are correct. A `mec-correctness-audit` workflow is confirming their
exact structure + whether any source actually uses the naive rule.

## Adversarial review
`mec-correctness-audit` workflow (adversarial verifiers on the claims "our example
refutes only a naive strawman, not the literature" and "our implementation matches
a correct published algorithm"). Outcome to be recorded here.

## Classification
**`naive-strawman`** (expected). Will be finalized to `naive-strawman` once the
audit confirms no published algorithm uses the naive rule; if — unexpectedly — a
published algorithm does, this escalates to `candidate-literature-counterexample`
and then `confirmed-...` only after the full protocol passes.

## Resolution / plan
Implemented the correct algorithm (Phase 1b) + brute-force differential guard.
Close as `resolved`/`naive-strawman` after the audit + dev-log wording correction.
