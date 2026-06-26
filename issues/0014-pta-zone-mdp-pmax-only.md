---
id: ISSUE-0014
title: PTA forward zone-MDP computes maximum reachability exactly; minimum needs the backward/game construction
status: resolved
severity: low
labels: design-decision, timed-automata, scope
created: 2026-06-26
updated: 2026-06-26
related:
  - src/pta.cpp
  - KNSW2007SymbolicPTA
  - KNSS2002PTA
---

## Summary
`pta::build` constructs the forward zone graph of a probabilistic timed automaton as
a finite MDP and `pta::maxReachLocation` solves it with `solve::maxReach`. The
forward zone-MDP yields the EXACT MAXIMUM probability of reaching a target location
(Kwiatkowska-Norman-Sproston-Wang, Inf. & Comp. 2007). It does NOT, in general, give
the exact MINIMUM reachability probability: forward zones can over-approximate the
behaviour relevant to Pmin, which requires the backward zone computation / the
game-based (or digital-clocks) construction.

## Design decision
Expose Pmax now (the common controller-synthesis objective: best strategy to reach a
goal). Symbolic states are identified by canonical-zone EQUALITY (not inclusion) so
the branch probabilities remain exact; extrapolation (Behrmann et al. 2006) keeps the
graph finite. Empty/blocked branches and no-edge locations route to an absorbing
deadlock sink so each action's distribution still sums to 1.

## Resolution (2026-06-26)
Implemented option (b): a **digital-clocks** engine (`pta::buildDigital`,
`maxReachLocationDigital`, `minReachLocationDigital`) for closed, diagonal-free PTAs.
Dense clocks become bounded integers (saturating above the max constant) and time
elapse is an explicit "tick" action, giving a finite MDP that is exact for BOTH Pmin
and Pmax (Kwiatkowska-Norman-Sproston, FMSD 2006). Pmin = 1 - maxSafety(avoid=target)
on that MDP. REQUIREMENT: kmax[i] >= every constant clock i is compared against in any
guard OR invariant. The zone engine remains the efficient Pmax path; the two engines
are cross-checked (digital Pmax must equal zone Pmax) in tests/unit/test_pta.cpp,
which also covers Pmin (invariant-forced=1, wait-out=0, lower-of-two=0.3). Both
engines are now selectable (the "toolbox of literature approaches" principle).

## Verification
Hand-case PTAs with known Pmax (timing-gated probabilistic edge, controller choice,
sequential resets, invariant-blocked unreachability) in tests/unit/test_pta.cpp; the
induced MDP is solved by the already-verified `solve::maxReach` (point distributions
=> exact).
