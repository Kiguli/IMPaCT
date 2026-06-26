---
id: ISSUE-0013
title: Bugs in the finite-horizon safety Sorted (O-maximization) UPPER-bound kernels
status: in-progress
severity: high
labels: correctness, tool-v1, omega-regular, safety, needs-toolchain-verification
created: 2026-06-26
updated: 2026-06-26
related:
  - src/GPU_synthesis.cpp
  - issues/0012-v1-inner-solve-lp-omax-unify-and-audit.md
---

## Summary
While verifying the ISSUE-0012 safety question (against the IMPaCT theory paper and
via an adversarially-checked multi-agent review), genuine bugs were found in
`IMDP::finiteHorizonSafeControllerSorted` (src/GPU_synthesis.cpp). These are the
UPPER-bound (`second1`) kernels — the LOWER-bound kernels are correct and were used
as the consistency oracle. This matters because ISSUE-0012 made the Sorted path the
LIVE default for `finiteHorizonSafeController`, so these defects are now on the
default safety path.

## Bugs (all in finiteHorizonSafeControllerSorted, no-input/no-disturbance branch)
1. **Over-count (IMDP_lower == false, upper loop).** The kernel added the avoid mass
   to the VALUE accumulator: `temp1 += accminAT[i];` then `s += accminAT[i];`. For
   finite (direct stay-safe) VI the avoid set is value 0, so it must go to `s`
   (normalization) ONLY — as the matching lower-bound loop does (it has just
   `s = s + accminAT[i];`). The extra term over-counts the upper safety bound by
   ~the per-state min-avoid mass each iteration. **Fix applied:** removed the
   `temp1 += accminAT[i];` line.
2. **Uninitialized value (IMDP_lower == true, upper loop).** `double temp1;` was
   declared and then read-modified (`temp1 += ...`) without initialization —
   undefined behaviour on the upper-bound value. **Fix applied:** `temp1 = 0;`.
3. **Missing base term (IMDP_lower == true, upper loop).** The first accumulation
   loop did `s += accminT[...]` but omitted `temp1 += accminT[...]*accs1[col]` (the
   minimum-transition contribution to the value), which all sibling loops include
   (e.g. the lower-bound loop of the same branch). Result would be systematically
   too low. **Fix applied:** added the missing `temp1 += accminT*accs1` term.
4. **Missing avoid-set residual block (IMDP_lower == true, upper loop).** The
   `if ((1.0-s) <= accdAT[i]) s = 1.0; else s += accdAT[i];` block (present in the
   matching lower-bound loop) was absent, so the residual normalization before the
   O-maximization redistribution was wrong. **Fix applied:** added the block.

The fixes make the upper-bound kernels mirror their verified lower-bound siblings.

## Independent verification
- Multi-agent review (4 readers: theory paper + LP-infinite + LP-finite + Sorted)
  plus a Reconcile synthesis plus 3 adversarial refuters; bugs 1 and 2 were called
  out by the refuters and the synthesis, and bugs 1–4 were then re-confirmed by
  direct reading of the four sibling kernels (the lower-bound loops at
  GPU_synthesis.cpp ~6157 and ~6328 are the consistency oracle; the buggy upper
  loops are at ~6242 and ~6426).
- Theory: IMPaCT paper (arXiv:2401.03555v2) Sec 5.1 / Algorithm 3 / Remark 5.1 — the
  finite case is "Algorithm 3 with while→for"; the implementation's equivalent direct
  stay-safe encoding must accumulate base + redistributed transition value, with
  avoid as value 0.

## IMPORTANT caveat
These fixes are **NOT compile-verified** in this environment (no AdaptiveCpp /
Armadillo / SYCL toolchain here). They are faithful, minimal edits that make each
buggy upper-bound kernel match its already-correct lower-bound sibling. **Verify on
the Windows/Linux toolchain** before relying on finite-safety synthesis: build a
small IMDP, run `finiteHorizonSafeController`, and check the upper bound equals the
LP-path result and brackets a Monte-Carlo estimate. The v2 `solve::maxSafety*`
(separately verified) and the cross-tool `.imdp` safety reference values can serve
as oracles.

## Guidance for the ISSUE-0012 unification
If toolchain verification of these fixes fails, hold back ONLY the finite-safety
redirect (keep `finiteHorizonSafeController` on the LP path) while leaving the reach
and infinite-safety redirects in place; re-enable once verified. (That the upper
kernel was using an uninitialized variable suggests this finite-safe Sorted path may
have been under-exercised in v1.)
