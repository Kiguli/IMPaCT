---
id: ISSUE-0011
title: Robust infinite-horizon omega-regular value is 0 under unbounded noise with an absorbing sink
status: resolved
severity: medium
labels: design-decision, abstraction, omega-regular, soundness
created: 2026-06-26
updated: 2026-06-26
related:
  - src/abstraction.cpp
  - src/omega.cpp
  - benchmarks/validate_omega.cpp
---

## Summary
Abstracting an UNBOUNDED-noise system (Gaussian) with an absorbing off-grid SINK
yields a transition system with **no non-trivial end components**: the per-cell
support window (hi>0 successors) is a loose over-approximation spanning several
cells, so every support-closed set must be "thick", but boundary cells leak to the
SINK and the looseness erodes any candidate region — the SINK is the only recurrent
class, and every support path reaches it. Consequently EVERY infinite-horizon
ω-regular value is **0**: robustly (nature drives to the SINK almost surely) AND
even optimistically (there is no support-closed set other than the SINK to cycle
in). This is mathematically correct, not a solver bug — and stronger than first
thought (the optimistic value is 0 too, not just the robust one).

## Details / evidence
`benchmarks/validate_omega.cpp` on a 2-D affine Gaussian system (144 cells):
robust persistence and robust Büchi are 0 at every start, while the OPTIMISTIC
values are non-zero. `omega::robustClosure(safeCells)` is empty because no action's
hi>0 support is ⊆ safeCells (the SINK is always a hi>0 successor).

## Impact
The robust ω-regular SOLVERS are exact and non-trivial on explicit IMDPs (validated:
12k-check strategy-enumeration differential, ISSUE-0009). The TRIVIALITY is specific
to robust + infinite-horizon + unbounded-noise + absorbing-sink abstractions. It does
NOT affect: robust reach-avoid / co-safe / finite-horizon (the ARCH stochastic
objectives), the optimistic ω-regular value, or robust ω-regular on bounded-
disturbance systems.

## Resolution / design decisions
1. **Document it** (this issue; note in `benchmarks/validate_omega.cpp`).
2. **Demonstrate the robust ω-regular contribution on explicit IMDPs**, where the
   values are non-zero and independently verifiable — added ω-regular models +
   reference values to `benchmarks/crosstool/` (Büchi recurrence, generalized-Büchi
   patrol, persistence), checked by `compare.py`, and `tools/imdp_solve` extended
   with `buchi` / `patrol` / `persist` properties.
3. **Continuous case study**: `validate_omega.cpp` reports robust (lower) vs
   optimistic (upper) and Monte-Carlo-validates that the real closed loop lies within
   the abstraction's two-sided bracket `[robust_lower, optimistic_upper]` — a sound
   soundness statement even when the robust lower bound is 0.
4. **Bounded disturbance closes the continuous gap (done).** Added the bounded
   uniform-noise kernel `abstraction::transitionInterval1DUniform` (TDD vs a
   fine mu-scan oracle, `test_abstraction.cpp`). With BOUNDED support, a cell well
   inside the domain has its whole next-state window in-domain — NO sink edge — so
   genuine robust end components exist. `benchmarks/validate_omega_bounded.cpp`:
   a 1-D system `x' = 0.9x + 0.5u + Uniform[-W,W]` with recurrence `G F region`
   gives robust value 1 at all 12 cells, and the continuous closed loop is
   Monte-Carlo validated (empirical recurrence ≥ robust lower bound at every start).
   So robust continuous ω-regular synthesis IS non-trivial once the disturbance is
   bounded — the unbounded-Gaussian triviality is specific to unbounded noise.
   Future: more bounded/truncated noise types and an n-D bounded builder.
