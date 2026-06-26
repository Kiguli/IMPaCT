---
id: ISSUE-0008
title: Grid-unaligned target double-counted -> unsound (too-high) reach lower bound
status: resolved
severity: high
labels: correctness, soundness, bug, abstraction
created: 2026-06-25
updated: 2026-06-25
related:
  - src/abstraction.cpp
  - benchmarks/validate_vp.cpp
---

## Summary
The abstraction added BOTH a transition to the target REGION [tlo,thi] (an
aggregate) AND transitions to each grid cell in the kernel window (skipping only
cells FULLY inside the target). When the target region is not grid-aligned, cells
that PARTIALLY overlap it are not skipped, so the overlap mass is counted twice:
once in the target aggregate and once in the partial cell. This inflates reach and
makes the (pessimistic) lower bound exceed the true value -> unsound.

## How it was found
Van der Pol nonlinear end-to-end verification (`benchmarks/validate_vp.cpp`): from
start (-1.0,-2.0) the synthesized lower bound was 0.354 but the empirical reach of
the real nonlinear stochastic system was 0.236 < 0.354 — a lower-bound violation.
(VP's target [-1.2,-0.9]x[-2.9,-2.0] is not aligned to the eta=0.2 grid.)

## Fix
Remove the separate target-region aggregate. Transitions go to the DISJOINT grid
cells in the window; a cell that is a target cell routes its mass to the absorbing
TARGET. Disjoint cells cannot double-count. Applied in buildSparseReachGeneral
(used by nD + nonlinear) and buildSparseReach1D. The target is thus the union of
fully-inside target cells (a sound under-approximation of the region; the simulator
uses the same target-cell criterion, keeping the comparison consistent).

## Verification of the fix
- VP end-to-end: 6/6 start states have empirical reach within [lower,upper] (the
  fixed cell now has lower 0.106, empirical 0.236 in [0.106, 0.689]).
- Unit suite still 44/44 (aligned-target tests unchanged: aggregate == window-routing
  when the target is grid-aligned).

## Classification
`our-bug`, resolved. Second soundness bug caught by end-to-end Monte-Carlo
validation (after ISSUE-0007); both were invisible to the internal differential
tests. Reinforces keeping closed-loop validation in the verification toolbox.
