---
id: ISSUE-0004
title: Phase 2 — LTLf→DFA construction + IMDP×DFA product + co-safe reachability
status: resolved
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
In progress. Step (2) DONE: `ltl::toDFA` builds a finite minimal DFA via semantic
(truth-table-over-anchors) canonicalization, validated differentially against the
evaluator (see ISSUE-0005 for the explosion bug that motivated the semantic
approach). Remaining: the `impact::product` contract (IMDP×DFA), absorbing
accepting states, reduction to `solve::maxReach*`, and the PD benchmark.

## RESOLVED (2026-07-02)
All three pieces shipped and are end-to-end validated: LTLf->DFA via formula progression
/ derivatives (src/ltl.cpp; Brzozowski JACM 1964 lineage; LTLf semantics De Giacomo-Vardi
IJCAI 2013), IMDP x DFA product (src/product.cpp), reduction to robust reachability
(solve.cpp). benchmarks/validate_cosafe.cpp verifies the FULL pipeline on a continuous
Package-Delivery-pattern system: sparse abstraction -> DFA product (400 cells x 4 DFA
states = 1608 product states) -> robust lower bound, then Monte-Carlo simulation of the
closed loop with LTLf trace evaluation: 4/4 start states satisfy empirical >= robust
lower bound. The DFA construction is differentially validated against the membership
evaluator as planned. PD at full co-safe spec is thereby unlocked.
