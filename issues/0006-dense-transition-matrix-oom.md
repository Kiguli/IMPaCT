---
id: ISSUE-0006
title: Dense transition matrix causes memory overflow even on "small" cases
status: resolved
severity: high
labels: scalability, performance, tool-v1, memory
created: 2026-06-25
updated: 2026-06-25
related:
  - src/IMDP.cpp
  - src/MDP.cpp
  - benchmarks/RESULTS.md
---

## Summary
v1 stores transition probabilities as DENSE Armadillo matrices of size
(state×input) × state. For the 2D-robot reach-avoid example (state 1583 after
target/avoid removal, input 441) this is 698103 × 1583 ≈ 1.1e9 doubles ≈ **8.84 GB**,
which OOM-kills the run under a 4 GB container cap. So even the smallest shipped
example is not "small" in memory; truly tiny cases or sparse storage are required.

## Evidence
Docker run (`impact:baseline`, `--memory=4g`, sorted/CPU, 2D-robot-RA):
abstraction vectors computed (is/ss/ts/min*/max* h5 produced), then:
`maximum transition Matrix dimensions: 698103 x 1583 ... 8840.78Mb` → `Killed`
(real 13m6s before OOM). controller.h5 NOT produced.

## Impact
- Regression goldens can't be generated from this example under tight memory.
- Confirms the plan's dense→sparse concern; products (IMDP×automaton, Phase 2/3)
  multiply state count and will hit this much sooner.

## Resolution / progress
- **DONE — sparse synthesis path.** The v2.0 solver works on a sparse model
  (`solve::IMDPModel` = per-state lists of only nonzero successor intervals), so
  synthesis memory is O(nnz), not O(states²) — same model as IntervalMDP.jl.
- **DONE — sparse abstraction (`src/abstraction.{h,cpp}`).** Builds the sparse IMDP
  directly for affine, axis-decoupled, diagonal-Gaussian systems (1-D end-to-end),
  storing only kernel-window successors. Verified: kernel vs brute-force; lossless
  pruning (sparse synthesis == dense-window synthesis); O(N) scaling
  (`benchmarks/sparse_scaling.cpp`): 125k states ≈ 255 MB + 0.42 s vs dense ≈ 250 GB.
  **Caveat:** the O(N) win holds with grid step + noise FIXED and the domain growing
  (constant kernel-in-cells). Refining eta at fixed sigma widens the kernel-in-cells
  (window ≈ 6σ/eta), giving a constant-factor (not asymptotic) saving — documented.
- **TODO:** n-D (diagonal) and coupled dynamics (mixed-monotone / NLopt bounds, as v1);
  sparse for IMDP×automaton products; and either refactor v1 `IMDP.cpp` to sparse or
  drive ARCH benchmarks through the new sparse pipeline. v1's dense path is unchanged
  for now (still OOMs); use the sparse pipeline for large runs.

## RESOLVED (2026-07-02)
Closed by the v2 sparse path (src/abstraction.cpp O(nnz) + portable solver): the
formerly-OOM 2D-robot/PR_minimal case COMPLETES at 3.86 GB (vs 17.7 GB dense);
AV_minimal builds+solves at 7.78 GB (dense OOM); the 6-D LM case (~96 GB/matrix dense,
infeasible) builds+solves at 196 MB / ~40 s (benchmarks/RESULTS.md). Automaton products
(Phase 2/3) run on the portable sparse solver, avoiding the multiplied dense blow-up.
Dense SYCL matrices remain as the GPU option only (ISSUE-0019 design decision).
