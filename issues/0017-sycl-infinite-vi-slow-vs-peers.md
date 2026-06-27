---
id: ISSUE-0017
title: SYCL infinite-horizon robust VI is slow on recurrent IMDPs vs Storm / IntervalMDP.jl
status: open
severity: medium
labels: performance, solver, tool-v1, cross-tool
created: 2026-06-27
updated: 2026-06-27
related:
  - src/GPU_synthesis.cpp        # infiniteHorizonReachControllerSorted (per-sweep SYCL buffers)
  - issues/0010-ovi-slow-on-recurrent.md
  - benchmarks/crosstool/arch
  - TOOL_COMPARISONS.md
---

## Summary
IMPaCT's SYCL infinite-horizon robust value iteration converges very slowly on
weakly-contracting (strongly-recurrent) abstractions compared to the peer tools,
even though it computes the **same** robust value. On the ARCH Van der Pol (VP)
reach benchmark IMPaCT takes 6119 interval-iteration sweeps / **214 s**, where
IntervalMDP.jl converges in 31 residual sweeps / **0.6 s** and Storm in 1.35 s.
On PD_p3 (package delivery, infinite reach) IMPaCT's SYCL VI did not finish in
12 minutes, while IntervalMDP.jl solved the identical exported IMDP in 1.4 s.

## Details / evidence
* `infiniteHorizonReachControllerSorted` (src/GPU_synthesis.cpp) performs **sound
  interval iteration**: it iterates a from-0 value vector and a from-1 value vector
  and stops when `max|V1 − V0| < epsilon`. On near-deterministic cycles this gap
  closes geometrically with a factor ≈ 1, so thousands of sweeps are needed.
* Each sweep re-creates the SYCL buffers (`sycl::buffer<…>` for the matrices and the
  sorted index) and re-runs `getSortedIndices`, adding fixed per-sweep overhead.
* Measured (CPU/OMP target):
  * VP: 6119 sweeps, 214.4 s (abstraction itself was only 1.9 s).
  * PD_p3: killed after > 700 s (no convergence); exported IMDP solved by
    IntervalMDP.jl in 1.40 s and by IMPaCT's own `imdp_solve` in the same range.
* The values agree where IMPaCT does converge (VP per-cell robust value matches
  IntervalMDP.jl to 2.3e-5), so this is purely performance.

## Reproduction
```
# in the impact toolchain container, with the exported models present:
tools/imdp_solve benchmarks/crosstool/arch/models/PD_p3.imdp reach target --bound pess   # fast (v2 solver)
# vs the SYCL example examples/ARCH-COMP/2025/PD/PD_target_p3.cpp (infiniteHorizonReachController) -> very slow
julia benchmarks/crosstool/peers/intervalmdp_runner.jl benchmarks/crosstool/arch/models/VP.imdp reach   # 31 iters, 0.6 s
```

## Independent verification
Two independent solvers (IntervalMDP.jl 0.6.0 robust VI; Storm 1.13.0
`--uncertainty-resolution robust`) solve the same exported `.imdp`/DRN in < 2 s and
return the same robust value, confirming the slowness is in IMPaCT's iteration
strategy + SYCL per-sweep overhead, not the problem size.

## Impact
Performance only (soundness unaffected). Makes infinite-horizon objectives on
recurrent abstractions impractical on the SYCL path; finite-horizon synthesis is
unaffected (BA 0.27 s, IC 0.06 s) and is competitive with the peers.

## Resolution / plan
This is the SYCL-path manifestation of ISSUE-0010 (OVI slow on recurrent IMDPs).
Options: (a) hoist SYCL buffer creation out of the sweep loop / keep device-resident
state across sweeps; (b) add a residual-stopping / Gauss–Seidel fast path with the
interval-iteration bracket only as a final certificate; (c) adopt sound/optimistic
value iteration (Quatmann–Katoen / Hartmanns–Kaminski) as the default solver, as
already planned in ROADMAP Phase 1. The v2 light-weight `solve::maxReach` (OVI) does
not exhibit the per-sweep SYCL overhead and should be the reference implementation.
