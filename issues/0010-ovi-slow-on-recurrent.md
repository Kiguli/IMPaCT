---
id: ISSUE-0010
title: OVI is slow on large strongly-recurrent IMDPs (VI-from-below mixes slowly)
status: resolved
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

## UPDATE (2026-07-02) — plan grounded, verified citations
Empirically re-confirmed during the ISSUE-0013 sweep: robust BA safety at eps 1e-6 needs
>180 s (eps 1e-4: 41 iters, fast); the peers' sparse engines are faster on such strongly
recurrent models. Grounded remedies (citations verified): (a) topological decomposition —
solve SCCs in reverse topological order (Dai, Mausam, Weld, Goldsmith, "Topological Value
Iteration Algorithms", JAIR 42:181-209, 2011, DOI 10.1613/jair.3390; graph::sccs already
exists); (b) Gauss-Seidel sweeps (Puterman 1994 Sec 6.3.3); (c) Sound Value Iteration
(Quatmann-Katoen, CAV 2018, DOI 10.1007/978-3-319-96145-3_37) as an alternative certified
engine. Severity low (workaround: eps 1e-4, or ValueIteration/II engines); remains open
as a performance enhancement with this implementation plan.

## RESOLVED (2026-07-02) — topological Gauss-Seidel value iteration
Implemented in solve.cpp::solveOVI: the from-below phase now (i) decomposes the support
graph into SCCs and processes them in reverse topological order — each component iterates
with its successor components already converged (Dai, Mausam, Weld, Goldsmith, "Topological
Value Iteration Algorithms", JAIR 42:181-209, 2011, DOI 10.1613/jair.3390; acyclic
singletons take a single pass, self-loop singletons iterate) — and (ii) sweeps in place
(Gauss-Seidel, Puterman 1994 Sec. 6.3.3). Both preserve monotone convergence from below to
the same least fixpoint, and OVI's inductive certificate is checked globally afterwards,
so soundness is independent of the sweep schedule.
MEASURED: BA robust safety at eps 1e-6 — previously >180 s (timeout) — now **7.8 s**
(>23x). Reference values unchanged (chain 0.25, choice 0.4, fork 0.4, robot 0.8946629826).
