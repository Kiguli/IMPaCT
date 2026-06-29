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
| VP | infinite reach | 2591 | 1.9 | 29.1 II / 0.59 VI (was 214) | yes (Δ≤2.3e-5) |
| IC_reach | finite reach H=5 | 592 | 1.8 | 0.06 | yes (Δ≤6.3e-7) |
| IC_safe | finite safety H=5 | 1681 | 2.7 | 0.47 | yes (Δ≤6.3e-7) |
| PD_p1 | infinite reach | 571 | 37 | 1.29 II / 1.34 VI (was 6.5) | yes (Δ≤7.1e-7) |
| PD_p3 | infinite reach | 571 | ~37 | 1.36 VI (II non-convergent, ISSUE-0003) | yes (1.0 = peers) |
| PR_minimal | infinite reach | 1583 | ~533 (8.84 GB×2) | (SYCL VI did not converge) | abstraction OK (was OOM @4 GB) |
| AV_minimal | infinite reach | 5562 | (7.42 GB×2) | — | abstraction fits (14.8 GB) |
| LM | finite reach H=200 | 36450 | dense-infeasible (~96 GB/matrix) | — | needs sparse storage |

## Sparse abstraction rerun (2026-06-29, `benchmarks/sparse_arch.cpp`)

Re-ran the AFFINE-in-state ARCH cases (mean range EXACT → no abstraction loss) through
the v2 **sparse** abstraction (`src/abstraction.cpp`, O(nnz)) + sparse OVI solver
(`src/solve.cpp`), instead of the dense SYCL `IMDP` class. Pure std C++, single thread.

| bench | cells×act | nnz | sparse | dense | build+solve | sparse interior mean | dense (nlopt) |
|---|---|---|---|---|---|---|---|
| AS (fin reach H10) | 2000×16 | 0.83M | 19.9 MB | 1.02 GB | 0.11+0.07 s | 0.0003 | 0.0045 |
| BA (fin safe H6) | 576×4 | 1.09M | 26.2 MB | 0.02 GB | 0.25+0.22 s | 0.107 | 0.471 |
| IC_reach (fin H5) | 1600×5 | 0.04M | 1.0 MB | 0.20 GB | 0.01+0.00 s | 0.057 | 0.072 |
| IC_safe (fin H5) | 1600×5 | 0.13M | 3.0 MB | 0.20 GB | 0.02+0.01 s | 0.750 | 0.761 |
| PD_p1 (inf r-avoid) | 576×441 | 5.1M | 122 MB | 2.34 GB | 0.8+15 s | 1.000 | 1.000 |
| PD_p3 (inf r-avoid) | 576×441 | 5.1M | 122 MB | 2.34 GB | 0.7+1.1 s | 1.000 | 1.0 (peers) |
| **PR_minimal** (inf) | 1600×441 | **161M** | **3.86 GB** | **18.1 GB** | 28.5+814 s | ~1.0 | (dense OOM) |

Wins: (1) **PR_minimal — the ISSUE-0006 dense-OOM case — now completes** (3.86 GB
vs 18 GB; would not run on the dense path). (2) Memory 5–51× smaller. (3) AS solves in
**0.07 s vs the dense SYCL 4.3 s** (the dense finite path still recomputes the big
`diff` matrix each sweep — the loop-invariant hoist was applied only to the infinite
functions). PR_minimal's 814 s solve is its very wide noise kernel (σ≈0.87 ≫ η=0.5 →
~217 nnz/row) under single-threaded OVI.

Accuracy: IC_safe and PD match the dense values; **AS and BA differ markedly** (BA
safety 0.11 vs 0.47). Likely cause: the sparse abstraction is **closed-form EXACT for
affine** dynamics, whereas the dense path uses `nlopt` (LN_SBPLX, a *local* optimiser)
per source cell, which can miss the adversarial worst case → looser/optimistic bounds
(BA's noise window ~7 cells exceeds its 4-cell domain, so low safety is plausible).
Needs Monte-Carlo to declare a winner (cf. abstraction-soundness ISSUE-0007/0008).
Nonlinear VP/AV/LM (interval-arithmetic mean enclosures) are the next pass.

## ISSUE-0003 nature-trap — MEC-collapse does NOT fix it (empirical, `models/naturetrap.imdp`)

On the canonical nature-trap (true robust V*(0)=0): **OVI** → `lower=0, upper=1e-6`
in **2** iters; **MEC-collapse** (support-graph) → `lower=0, upper=1` stuck, **2 000 000**
iters. Support-graph MEC misses nature-confinable ECs (state 0's action has a positive
*upper*-bound leaving edge, so it is not a support EC, yet nature can zero that edge's
*lower* bound and confine). The fix is OVI (no EC needed) or robust-EC deflation — not
plain MEC preprocessing.

Findings: robust values agree with the peers everywhere applicable; IMPaCT now offers
two infinite-horizon solvers (`setIterationMethod`) — sound interval iteration
(default; sped up ~7× by hoisting per-sweep loop-invariant work) and peer-style pure
value iteration (VP 0.59 s ≤ peers; converges on PD_p3 where interval iteration cannot,
ISSUE-0017/0003); robust ω-regular is IMPaCT-only (Storm/IntervalMDP.jl unsupported);
the dense abstraction is the memory wall (ISSUE-0006), cleared by the big machine
except for the 6-D LM case.
