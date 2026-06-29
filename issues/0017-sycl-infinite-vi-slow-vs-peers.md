---
id: ISSUE-0017
title: SYCL infinite-horizon robust VI is slow on recurrent IMDPs vs Storm / IntervalMDP.jl
status: in-progress
severity: low
labels: performance, solver, tool-v1, cross-tool
created: 2026-06-27
updated: 2026-06-28
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
**Fixed (per-sweep waste), 2026-06-28.** The dominant cost was loop-INVARIANT work
recomputed every sweep inside the value-iteration loops of the two infinite-horizon
synthesis functions:
* `mat diffT = maxTransitionM - minTransitionM;` (and the target/avoid analogues and
  the fixed-input `tempTmax-tempTmin` variants) — a full `(state*input) x state` dense
  matrix **reallocated + recomputed on every sweep** (for the large with-input cases,
  e.g. PD, that is a ~1.1 GB allocation per sweep); and
* a fresh `sycl::queue` constructed every sweep.
Both are now hoisted once-per-loop (each loop wrapped in a `{ }` block so the
once-computed diffs do not clash across sibling loops — see
`benchmarks/crosstool/peers/hoist_diffs.py` / `hoist_queue.py`, 32 loops each).
This is loop-invariant code motion: the synthesised values are **bit-for-bit
identical** (verified: VP `[0,0.229697]`, PD_p1 `[0.999789,1]` unchanged), and the
robust VI is much faster:

| case | before | after | peers (IntervalMDP.jl) |
|---|---|---|---|
| VP (infinite reach) | 214.4 s / 6119 sweeps | **34.6 s** (6.2x) | 0.6 s |
| PD_p1 (infinite reach) | 6.5 s | **1.29 s** (5x, ~= peer) | 1.33 s |

**Remaining (kept open, low severity):** VP is still slower than the peers because of
(a) the per-sweep construction of the ~11 SYCL buffers (the *constant* matrix buffers
could also be hoisted — more invasive, needs GPU re-verification), and (b) the
iteration count itself: IMPaCT does **sound interval iteration** (iterate from 0 and
from 1, stop when the gap < ε), which is more rigorous but slower-converging than the
peers' residual-stopping VI. Adopting OVI (Quatmann–Katoen / Hartmanns–Kaminski) as
the SYCL default (ROADMAP Phase 1) would close both gaps.

**Note on PD_p3 (re-diagnosed).** PD_p3's non-termination is NOT this performance
issue — it is the **interval-iteration nature-trap (ISSUE-0003)**: with an end
component the from-1 iterate converges to the greatest fixed point while the from-0
iterate converges to the (correct, smaller) least fixed point, so the gap never closes
and the loop runs to the timeout (observed: gap stuck at 0.9997 after 15 380 sweeps).
The v2 light-weight `solve::maxReach` (OVI + MEC handling) returns the correct value
(1.0, = IntervalMDP.jl / Storm); porting that MEC-collapse to the SYCL path is the fix
for PD_p3 and is tracked under ISSUE-0003 / ISSUE-0010, separate from this item.
