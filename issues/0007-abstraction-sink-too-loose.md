---
id: ISSUE-0007
title: Abstraction SINK (outside-grid) bound too loose -> severe reach under-estimation
status: resolved
severity: high
labels: correctness, soundness, bug, abstraction
created: 2026-06-25
updated: 2026-06-25
related:
  - src/abstraction.cpp
  - benchmarks/validate_montecarlo.cpp
---

## Summary
The first sparse abstraction set the SINK (outside-grid, value 0) interval to
`[1 - sumHi, 1 - sumLo]`, where sumLo/sumHi are the sums of the per-successor
lower/upper bounds. Because each successor lower bound is a loose per-region
minimum, `sumLo` is tiny, so `sinkHi = 1 - sumLo` ≈ 1. For pessimistic (robust)
reachability nature then routes almost all mass to the value-0 sink, making the
synthesized reach ≈ 0 even where the true reach ≈ 1.

## How it was found
Monte-Carlo closed-loop validation (`benchmarks/validate_montecarlo.cpp`):
synthesize the robust controller, simulate the continuous system under it, compare
the empirical reach to the synthesized bounds. The empirical reach was ~0.99 from
every start while the synthesized upper bound was ~0.01–0.76 — a gross
under-estimate that exposed the loose sink. (Internal tests passed because dense
and sparse builds shared the same wrong sink.)

## Fix
Bound the outside-grid probability TIGHTLY as the complement of the whole-grid-box
probability: `P(outside) ∈ [1 - gridHi, 1 - gridLo]`, where `[gridLo,gridHi]` is
the transition bound to the entire state-space box (product of per-dimension
in-grid probabilities). This is a valid and tight bound (P(outside)=1-P(in grid)).
Applied in both `buildSparseReach1D` and `buildSparseReachND`.

## Verification of the fix
- Unit suite still green (lossless pruning / sparse==dense / target=1 / O(N)).
- Monte-Carlo: empirical reach >= synthesized robust lower bound for all tested
  start states, in BOTH a saturated case (value≈1) and a discriminating
  high-noise case (value≈0.32, empirical≈0.36). For a robust controller the real
  (benign) noise meets-or-exceeds the worst-case lower bound; empirical may exceed
  the robust upper (which bounds the worst case, not the actual).

## Classification
`our-bug`, resolved. Notable as a soundness bug caught by end-to-end Monte-Carlo
validation that internal differential tests missed (both builds shared the error) —
motivates keeping the closed-loop validation in the verification toolbox.
