---
id: ISSUE-0010
title: OVI is slow on large strongly-recurrent IMDPs (VI-from-below mixes slowly)
status: open
severity: low
labels: performance, solver
created: 2026-06-25
updated: 2026-06-25
related:
  - src/solve.cpp
---

## Summary
Optimistic Value Iteration (the default solver) computes its lower bound by value
iteration from below. On a large, strongly-recurrent IMDP (e.g. a full-dynamics
grid abstraction with no absorbing target, where the value approaches 1 almost
everywhere), VI-from-below converges slowly (slow mixing), so OVI can take many
iterations. Observed: `maxReachOptimistic` on a 144-cell full-dynamics 2-D grid
took long enough to dominate the unit suite; shrinking the test grid fixed it.

## Impact
Correctness unaffected (still sound). Only runtime, and only for large recurrent
models / the optimistic sense. The pessimistic product runs in the co-safe
benchmark were fast.

## Resolution / plan
- For the optimistic sense, `Method::MECCollapse` collapses the large end component
  and converges much faster — offer/auto-select it for optimistic on recurrent models.
- The robust-EC interval iteration (Dutreix-Coogan / Weininger, ISSUE-0009) would
  also collapse ECs and be fast for both senses.
- Possible: add a relative-stopping / Gauss-Seidel sweep or topological ordering to
  speed VI-from-below. Track here.
