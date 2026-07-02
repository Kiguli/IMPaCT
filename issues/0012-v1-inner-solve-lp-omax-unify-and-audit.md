---
id: ISSUE-0012
title: Unify v1 inner-solve on O-maximization (LP names -> aliases) + v1 code audit
status: resolved
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

- **RETRACTED over-claim (was "REAL finite-vs-infinite SAFETY inconsistency").**
  An earlier version of this audit called the LP finite/infinite avoid handling a
  genuine bug ("one of the two is wrong"). That was an OVER-CLAIM (same pattern as
  the ISSUE-0002 lesson) and is withdrawn — the user correctly pushed back. Verified
  against the IMPaCT theory paper (Wooding-Lavaei, arXiv:2401.03555v2, Sec 5.1,
  Algorithm 3, Remark 5.1) and by an adversarially-checked multi-agent review: the
  two paths compute the SAME safety probability via two equivalent encodings —
  finite uses direct stay-safe VI (avoid = value 0, no complement), infinite uses
  reach-avoid VI (avoid = value-1 affine term) then a 1-complement. Proof (per fixed
  policy/LP vertex, avoid lumping all non-safe mass so p_S + a = 1): with W=stay-safe
  (W0=1) and V=reach-avoid (V0=0), 1 - V_{k+1} = sum_S T (1 - V_k) = W_{k+1}, so
  W_k = 1 - V_k for every finite k. The infinite path keeps the affine avoid term for
  CONVERGENCE (without it V0 sticks at 0n), not as a different quantity. The user's
  conclusion ("finite does not need unsafe-state values") is therefore right; their
  stated reason ("because we don't check convergence") is not the precise mechanism
  (the direct encoding works at any horizon — it isn't horizon-specific). NOT a bug;
  no action needed. (Genuine but separate bugs this review surfaced in the now-LIVE
  finite-safe Sorted path are tracked in ISSUE-0013.)
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

## RESOLVED (2026-07-02)
The unification shipped (LP-named entry points delegate to the O-maximization Sorted
implementations; O-max verified against brute-force vertex enumeration). The audit did its
job: it surfaced the genuine upper-kernel bugs now tracked+fixed as ISSUE-0013. The
unified path is compile-verified on the full toolchain and cross-validated against
IntervalMDP.jl / Storm / PRISM on every shared model (TOOL_COMPARISONS.md; finite-horizon
BA/IC values match to ~1e-6). Remaining v1 cleanup is cosmetic, not correctness.
