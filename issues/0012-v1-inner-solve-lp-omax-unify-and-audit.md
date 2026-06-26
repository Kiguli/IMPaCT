---
id: ISSUE-0012
title: Unify v1 inner-solve on O-maximization (LP names -> aliases) + v1 code audit
status: in-progress
severity: medium
labels: refactor, cleanup, performance, tool-v1
created: 2026-06-26
updated: 2026-06-26
related:
  - src/IMDP.cpp
  - src/GPU_synthesis.cpp
  - src/omaximization.cpp
---

## Summary
The original IMPaCT (v1) computed the robust (controller-vs-nature) Bellman update
two equivalent ways: a per-state **GLPK linear program**
(`infiniteHorizonControllerImpl` / `finiteHorizonControllerImpl`) and a closed-form
**O-maximization "sort-and-assign"** (the `*Sorted` variants). They return the same
robust extremal value, but O-maximization is O(n log n) per state vs an LP solve, so
it is always faster, and its core is verified against brute-force vertex enumeration
(v2 `test_omaximization`). Per user request, O-maximization is now the single
default and the LP-named public methods are backwards-compatible aliases.

## Change made (this commit)
`src/IMDP.cpp`: the four public entry points now delegate to the O-maximization
implementations (no API/signature change — pure backwards-compatible redirect):
- `infiniteHorizonReachController`  -> `infiniteHorizonReachControllerSorted`
- `infiniteHorizonSafeController`   -> `infiniteHorizonSafeControllerSorted`
- `finiteHorizonReachController`    -> `finiteHorizonReachControllerSorted`
- `finiteHorizonSafeController`     -> `finiteHorizonSafeControllerSorted`
The GLPK LP implementations are now dead code, marked DEPRECATED in place. Removed
the stale "DEPRECATED ... delete after verification" comment blocks around the
entry points.

## Audit findings (v1 IMDP.cpp / GPU_synthesis.cpp)
Verified by reading the source (could NOT compile/run here — needs the
AdaptiveCpp + Armadillo + GLPK + HDF5 toolchain, absent in this environment):

- **REAL — finite-vs-infinite SAFETY inconsistency (LP path).** In
  `finiteHorizonControllerImpl` the safe branch sets the Avoid objective coef to 0.0
  (IMDP.cpp ~4937/4945) and never accumulates `glp_get_col_prim(lp, n+1)` (only does
  so for reach, ~4972), whereas `infiniteHorizonControllerImpl` uses coef 1.0 (~3247)
  and DOES accumulate the Avoid column for safe (~3281). One of the two is wrong.
  **Mooted** by the unification (the LP path is no longer used); to be deleted.
- **FALSE POSITIVE — "GLPK sparse-matrix indexing bug" (`&ja[0]`/`&ar[0]`).** This is
  the standard GLPK 1-based idiom: `ja`/`ar` are sized `n+extra+1`, populated at
  indices 1..len, and `glp_set_mat_row(lp,1,len,&ja[0],&ar[0])` reads indices 1..len
  (index 0 ignored). Correct, NOT a bug. (Recorded to avoid re-flagging — cf. the
  ISSUE-0002 lesson on over-claiming.)
- **Minor.** Dead local `vector<int> ia = {0};` in the LP kernels (unused);
  `glp_simplex` return status is ignored at all call sites (silent-failure risk);
  two hardcoded tolerances (inner 1e-8 vs outer epsilon) — all in the now-dead LP
  path. `//TODO: throw an error here.` stubs in the Sorted path
  (GPU_synthesis.cpp ~3223 / ~8640) where residual mass cannot fully cover the
  target — should raise instead of silently allocating partial mass (live code;
  worth a follow-up). GLPK problem objects ARE freed (`glp_delete_prob`).
- The v1 Sorted path uses tolerance stopping, not sound interval iteration
  (pre-existing ISSUE-0001); the v2 solver (OVI) is the sound path.

## Plan / remaining
1. (done) Redirect public entry points to O-maximization.
2. (pending, needs toolchain) compile-verify, then DELETE the LP implementations and
   drop the GLPK dependency from the build; address the Sorted `//TODO` error stubs.
3. Re-run the example case studies to confirm identical outputs and a speedup.

## Note
Edits here are NOT compile-verified in this environment (no SYCL/GLPK toolchain).
They are mechanical, signature-preserving redirects; verification on a build machine
is the next step before deleting the LP code.
