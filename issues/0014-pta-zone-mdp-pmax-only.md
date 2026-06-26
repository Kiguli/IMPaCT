---
id: ISSUE-0014
title: PTA forward zone-MDP computes maximum reachability exactly; minimum needs the backward/game construction
status: open
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

## Plan
Add Pmin via either (a) the backward zone construction of KNSW 2007, or (b) the
digital-clocks semantics (Kwiatkowska-Norman-Sproston, "Performance analysis of PTAs
using digital clocks") which reduces a (closed, bounded) PTA to a finite MDP exact
for both Pmin and Pmax. Until then, `maxReachLocation` is documented as Pmax-only.

## Verification
Hand-case PTAs with known Pmax (timing-gated probabilistic edge, controller choice,
sequential resets, invariant-blocked unreachability) in tests/unit/test_pta.cpp; the
induced MDP is solved by the already-verified `solve::maxReach` (point distributions
=> exact).
