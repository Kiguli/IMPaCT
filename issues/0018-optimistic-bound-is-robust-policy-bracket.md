---
id: ISSUE-0018
title: IMPaCT's optimistic value is a bracket of the robust controller, not the cooperative optimum
status: resolved
severity: low
labels: semantics, documentation, cross-tool
created: 2026-06-27
updated: 2026-06-27
related:
  - src/GPU_synthesis.cpp        # controller.col(dim_x+dim_u+1) = second1 (upper) under the fixed lower policy
  - TOOL_COMPARISONS.md
---

## Summary
When IMPaCT synthesises a robust controller with `IMDP_lower = true`, the controller
file stores **two** value columns: a lower (pessimistic) and an upper (optimistic)
bound. The upper bound is the best-case value of the **fixed pessimistic-optimal
policy** (nature cooperating), *not* the value of the separately-optimised
cooperative policy. A peer tool's "optimistic + maximize" query re-optimises the
policy, so the two optimistic numbers legitimately differ. This is correct by design
but is easy to misread as a discrepancy.

## Details / evidence
* `infiniteHorizonReachControllerSorted(true)` / the finite-horizon `Sorted` variants
  compute the lower-bound iterate `first0` and, for the **same fixed input policy**
  (the with-input upper loop uses a reduced matrix "where input is fixed"), the upper
  iterate `second1`. The controller stores `col(dim_x+dim_u)=first0`,
  `col(dim_x+dim_u+1)=second1`.
* Cross-tool measurement on the exported IMDP (robust controller, IMDP_lower=true):
  the **pessimistic** column matches IntervalMDP.jl `Pessimistic,Maximize` to ≤ 7e-7
  on every ARCH benchmark, while the **optimistic** column differs from
  `Optimistic,Maximize`: AS 0.996, IC_safe 0.108, IC_reach 0.095, BA 4.7e-3, because
  IntervalMDP re-optimises the policy for the cooperative objective.

## Reproduction
```
# robust controller from IMPaCT (IMDP_lower=true) -> extract both value columns
python3 benchmarks/crosstool/peers/extract_impact_value.py run/AS/controller.h5 3 2
# peer optimistic re-optimises the policy:
julia benchmarks/crosstool/peers/intervalmdp_runner.jl models/AS.imdp reach --horizon 10
# pess columns agree; opt columns differ (expected).
```

## Independent verification
The pessimistic value (the one that carries the guarantee) is confirmed identical by
IntervalMDP.jl and Storm. The optimistic gap disappears if IMPaCT is run with
`IMDP_lower = false` (synthesise the optimistic controller) and compared to
`Optimistic, Maximize`.

## Impact
No soundness impact. Purely a reporting/semantics clarification: the `[lower, upper]`
pair is the value **interval of the synthesised robust policy**, which is the useful
quantity for a controller, but is not the same as the two-sided value of the
*optimal* policy in each sense.

## Resolution / plan
Document in TOOL_COMPARISONS.md and the controller-output docs (done). Optionally
expose both controllers (robust and cooperative) or label the two columns explicitly
as "robust-policy lower / robust-policy upper" to avoid confusion.

## RESOLVED (2026-07-02)
The v2 portable solver's `maxReachOptimistic` (and the dense path's OptimisticVI
dispatch in the `false`/optimistic sense) computes the TRUE cooperative optimum —
controller and nature jointly maximizing — not merely the robust controller's upper
bracket. This is validated against Storm `--uncertainty-resolution cooperative` and
IntervalMDP.jl `Optimistic/Maximize` on every shared model (equal to 1e-6 or better,
TOOL_COMPARISONS.md). The legacy dense `second1` bound remains what it always was — the
upper bracket of the ROBUST controller — and the distinction is now documented in the
docs (manual/specifications.md) and the comparison pages. Users wanting the cooperative
optimum use the portable solver / OVI dispatch; semantics ambiguity eliminated.
