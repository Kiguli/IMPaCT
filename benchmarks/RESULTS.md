# IMPaCT v2.0 — Benchmark Results (running log)

Append a results block per run via `run_archcomp.py`. Baseline first; then one
block per phase showing the interleaved progression (same answers + speedup in
Phase 1; new spec classes in Phases 2–3).

Spec class per ARCH-COMP Stochastic-Models case (verify against the ARCH-COMP25
report before publication):

| Case | Dyn / dims | Spec class | v1 status | v2 target phase |
|---|---|---|---|---|
| AS | anaesthesia, 3D nonlinear | finite-horizon safety | full | P0/P1 |
| BA | building automation, 4D | finite-horizon safety | full | P0/P1 |
| VP | van der Pol, 2D | infinite-horizon synthesis | full | P0/P1 |
| IC | integrator chain, ≤? D | infinite-horizon reach/safety | reduced (scale) | P1 |
| AV_minimal | autonomous vehicle (reduced) | reach-while-avoid | full | P0/P1 |
| PR_minimal | patrol robot (reduced) | reach-while-avoid | full | P0/P1 → P3 recurrence |
| LM | (confirm) | (confirm) | — | P0/P1 |
| PD | package delivery, 2D | **co-safe LTL (DFA)** | **reduced** | **P2** |

> Note: the on-disk PD example is currently `PD_target_p1/p3.cpp` — i.e. the
> co-safe spec collapsed into separate reach-avoid targets. Phase 2 replaces this
> with a real DFA product (no reduction).

## v1 baseline note (ISSUE-0006)
The v1 dense path OOMs even on the smallest shipped example: 2D-robot-RA builds a
dense (state×input)×state matrix of 698103×1583 ≈ **8.84 GB** → killed under a 4 GB
cap. Confirms the need for sparse storage before larger benchmarks.

## Sparse abstraction scaling (v2.0, `benchmarks/sparse_scaling.cpp`)
1-D reach IMDP, domain grown with grid step (eta=0.1) and noise (sigma=0.3) FIXED, so
the kernel spans a constant #cells ⇒ nnz is O(N) while a dense (min+max) matrix is
O(N²). Synthesis = optimistic VI (eps=1e-6).

| domain | cells | nnz | nnz/cell | sparse (MB) | dense (MB) | synth (s) |
|---|---|---|---|---|---|---|
| 20 | 200 | 18,005 | 89 | 0.36 | 0.6 | 0.01 |
| 100 | 1,000 | 99,980 | 100 | 2.0 | 16 | 0.00 |
| 500 | 5,000 | 507,980 | 102 | 10.2 | 400 | 0.01 |
| 2,500 | 25,000 | 2,547,980 | 102 | 51 | 10,000 | 0.08 |
| 12,500 | 125,000 | 12,747,980 | 102 | **255** | **250,000** | 0.42 |

At 125k states the sparse model is ~255 MB and synthesizes in 0.42 s, where the dense
matrix would be ~250 GB (infeasible). This is the IntervalMDP.jl-style memory win.

## ARCH runs on the big machine (2026-06-27, AdaptiveCpp/OMP, 56 GB host)

Full cross-tool comparison vs PRISM / Storm / IntervalMDP.jl is in the top-level
[`TOOL_COMPARISONS.md`](../TOOL_COMPARISONS.md). Summary of the IMPaCT side
(abstraction = build the IMDP from continuous dynamics; synth = robust VI):

| case | spec | cells | abstraction (s) | synth (s) | robust value matches IntervalMDP.jl? |
|---|---|---|---|---|---|
| AS | finite reach H=10 | 2460 | 27.0 | 4.3 | yes (Δ≤4.8e-7) |
| BA | finite safety H=6 | 1225 | 7.6 | 0.27 | yes (Δ≤4.3e-7) |
| VP | infinite reach | 2591 | 1.9 | 214 (6119 iters) | yes (Δ≤2.3e-5) |
| IC_reach | finite reach H=5 | 592 | 1.8 | 0.06 | yes (Δ≤6.3e-7) |
| IC_safe | finite safety H=5 | 1681 | 2.7 | 0.47 | yes (Δ≤6.3e-7) |
| PD_p1 | infinite reach | 571 | 37 | 6.5 | yes (Δ≤7.1e-7) |
| PD_p3 | infinite reach | 571 | ~37 | (SYCL VI did not converge) | yes via imdp_solve/peers (1.0) |
| PR_minimal | infinite reach | 1583 | ~533 (8.84 GB×2) | (SYCL VI did not converge) | abstraction OK (was OOM @4 GB) |
| AV_minimal | infinite reach | 5562 | (7.42 GB×2) | — | abstraction fits (14.8 GB) |
| LM | finite reach H=200 | 36450 | dense-infeasible (~96 GB/matrix) | — | needs sparse storage |

Findings: robust values agree with the peers everywhere applicable; IMPaCT's
infinite-horizon SYCL VI is slow on recurrent models (ISSUE-0017); robust ω-regular
is IMPaCT-only (Storm/IntervalMDP.jl unsupported); the dense abstraction is the
memory wall (ISSUE-0006), cleared by the big machine except for the 6-D LM case.
