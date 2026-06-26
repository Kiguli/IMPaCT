---
id: ISSUE-0003
title: Pessimistic interval iteration does not converge on nature-confinable end components
status: resolved
severity: medium
labels: soundness, correctness, robust-ec
created: 2026-06-25
updated: 2026-06-25
related:
  - src/solve.cpp
  - tests/unit/test_interval_iteration.cpp
  - HaddadMonmege2018IntervalIteration
  - Baier2017IntervalIteration
  - Asadi2026qualitative
---

## Summary
`solve::maxReachPessimistic` collapses end components using the SUPPORT graph
(transitions with hi>0), which captures *controller* end components. For the
robust (adversarial-nature) value this is insufficient: nature can **confine**
the play inside a set by sending a leaving edge to its lower bound 0. Such
"nature-confinable" end components create a NON-UNIQUE fixpoint, so the upper
bound of interval iteration does not converge to the true value (the gap stays
large). The upper bound is still a *valid* upper bound (so results are not
unsound), but the contract `gap <= 2*eps` is violated for these instances.

## Minimal reproducible counterexample
States {0,1,2}, target = {2}:
- `0: a -> {0:[0.5,1.0], 1:[0.0,0.5]}`
- `1: b -> {2:[1,1]}`   (state 1 reaches the target w.p. 1)
- `2: -> {2:[1,1]}`     (target)

True robust value: V*(0)=0 — nature sets p(0->1)=0 (its lower bound) and
p(0->0)=1, trapping the controller at 0 forever. V*(1)=1, V*(2)=1.

Interval iteration: the lower bound converges correctly to 0. The upper bound
sticks at U(0)=1: F_pess(U)(0) = min_p [p0·U0 + p1·U1] = U1 + p0·(U0−U1); with
U0=U1=1 this is 1 for every nature choice, and since U0 starts equal to U1=1 it
never decreases. Any value in [0,1] (with U1=1) is a fixpoint → non-unique.

## Independent verification
The `VI-from-below` oracle (`tests/unit/test_interval_iteration.cpp`) gives the
true value 0; the randomized interval differential test surfaced this (2 failing
assertions: gap = 1, midpoint 0.5 vs oracle 0).

## Impact / current scope
`maxReachPessimistic` is sound AND convergent for:
- point-probability MDPs (lo == hi: nature has no choice; support EC = real EC), and
- the hand contracts Models 1–6 (no nature-trap structure).
`maxReachOptimistic` is sound + convergent on intervals (support EC = optimistic EC).
The gap is **pessimistic + interval with lo=0 leaving edges enabling nature-traps**.

## Root cause
The end-component notion for robust (2.5-player) interval MDPs differs from the
optimistic support-graph MEC. Nature-confinable ECs must also be collapsed (or
handled by a robust-EC algorithm) for the pessimistic upper bound to converge.

## Resolution (RESOLVED)
Added **Optimistic Value Iteration** (Hartmanns & Kaminski, CAV 2020 — verified in
`paper/References.bib`) as the default solver (`solve::Method::OptimisticVI`). OVI
computes the lower bound by VI-from-below and a guessed upper bound U=min(1,L+eps)
that is VERIFIED inductive (F(U) <= U); by Knaster-Tarski V* <= U, so [L,U] is sound
with gap <= eps — with NO end-component handling, so it converges on nature-confinable
ECs. The counterexample now converges (V*(0)=0); the randomized differential covers
pessimistic AND optimistic on point AND interval MDPs (45k+ assertions).

This is offered as a selectable method (toolbox): `OptimisticVI` (default, general,
sound+convergent) vs `MECCollapse` (interval iteration with EC collapse — faster on
controller ECs but does NOT converge on pessimistic interval nature-traps; valid for
point MDPs / optimistic). A future option is the robust-EC interval iteration
(Dutreix-Coogan / Weininger game reduction) that is both fast and trap-correct
(ROADMAP / Phase 3). References: Hartmanns-Kaminski (CAV 2020); Haddad-Monmege
(TCS 2018); Baier et al. (CAV 2017).

## Classification
`our-bug` / known limitation. NOT a literature counterexample — the literature
(Haddad-Monmege) handles this; our implementation took an optimistic-support
shortcut for MEC collapse.
