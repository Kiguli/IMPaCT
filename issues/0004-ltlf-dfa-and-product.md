---
id: ISSUE-0004
title: Phase 2 — LTLf→DFA construction + IMDP×DFA product + co-safe reachability
status: open
severity: medium
labels: enhancement, phase-2
created: 2026-06-25
updated: 2026-06-25
related:
  - src/ltl.cpp
  - src/solve.cpp
  - tests/TEST_PLAN.md
  - degiacomo2021compositional
  - duretlutz2016spot
---

## Summary
Phase 2 part 1 delivered the LTLf front-end as a parser + finite-trace membership
evaluator (`src/ltl.cpp`). To actually synthesize against co-safe-LTL specs we
need: (1) an LTLf→DFA construction, (2) the IMDP×DFA product, (3) reduction to
robust reachability (reuse `solve`). This unlocks Package Delivery (PD/PDx) at
full spec (currently "reduced" in ARCH-COMP).

## Decision (recorded)
Build a **self-contained DFA** via formula progression / Brzozowski-style
derivatives over the (finite) AP alphabet — NO external dependency — and validate
it differentially against the existing membership evaluator (the evaluator is the
ground-truth oracle: DFA-accepts(trace) must equal evaluator-accepts(trace) for
random formulas × random traces). Rationale: keeps the build dependency-free and
testable; consistent with not adding heavy deps unilaterally.
**Spot** (duretlutz2016spot) remains a future option if we need industrial-scale
or minimized automata / ω-automata performance; revisit if the homemade DFA is a
bottleneck.

## Plan
1. New contract: `impact::product` (IMDP×DFA) in contracts.h + TEST_PLAN §2.
2. DFA: progression of the parsed LTLf AST over letters; dedup states by
   normalized formula; accepting = formula satisfied by empty continuation.
3. Differential test: DFA membership == `ltl::acceptsFinite` on random cases.
4. Product + absorbing accepting states → `solve::maxReach*`; cross-check PD
   probability vs oracle/SySCoRe on a small instance.

## Status
Open — next Phase 2 step.
