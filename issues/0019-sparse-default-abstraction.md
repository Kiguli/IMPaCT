---
id: ISSUE-0019
title: Make the sparse abstraction the default everywhere (design decision)
status: in-progress
severity: medium
labels: design-decision, scalability, abstraction, tool-v1
created: 2026-06-29
updated: 2026-06-29
related:
  - src/abstraction.cpp
  - src/solve.cpp
  - benchmarks/sparse_arch.cpp
  - issues/0006-dense-transition-matrix-oom.md
  - issues/0020-sparse-vs-dense-abstraction-discrepancy.md
---

## Decision
Use IMPaCT's **sparse** interval-MDP abstraction (`src/abstraction.cpp`, O(nnz),
IntervalMDP.jl-style per-(state,action) successor lists) + the sparse robust solver
(`src/solve.cpp`, OVI default) as the DEFAULT abstraction/solve path for the examples
and benchmarks, replacing the dense SYCL `IMDP`-class matrices (`minTransitionM`, a
`(state*input) x state` dense `mat`) where possible without accuracy loss. The dense
SYCL path is retained as an option (GPU-parallel abstraction).

## Rationale (measured)
- **Scalability (ISSUE-0006):** the dense `(state*input) x state` matrix OOMs even on
  shipped cases. PR_minimal is `698103 x 1583 ≈ 8.84 GB` dense (×2 = 17.7 GB) and was
  killed; through the sparse path it COMPLETES at 3.86 GB. Memory is 5–51× smaller
  across the ARCH cases (`benchmarks/RESULTS.md`).
- **Speed:** finite cases solve far faster (AS 0.07 s sparse vs 4.3 s dense; even after
  the dense finite-hoist, 1.57 s). Sparse is single-threaded but avoids the dense
  per-sweep matrix work.
- **Soundness on end components:** the sparse solver defaults to OVI, which converges on
  nature-confinable ECs (ISSUE-0003) where the dense interval iteration does not.
- **Accuracy:** for AFFINE-in-state dynamics the sparse abstraction's per-cell mean
  range is EXACT, so there is no abstraction loss (and possibly a soundness GAIN over
  the dense nlopt path — see ISSUE-0020).

## Status / plan
- [x] Affine ARCH cases (AS, BA, IC_reach, IC_safe, PD_p1, PD_p3, PR_minimal) rerun via
  the sparse path (`benchmarks/sparse_arch.cpp`).
- [ ] **Gate:** resolve the AS/BA sparse-vs-dense value discrepancy (ISSUE-0020) by
  Monte-Carlo before flipping the default, so we do not silently change results.
- [ ] Nonlinear ARCH cases (VP, AV_minimal, LM) via interval-arithmetic mean enclosures.
- [ ] Make the example/recommended workflow default to the sparse path; keep the dense
  SYCL path as an explicit option.

## Accuracy resolved (ISSUE-0020)
The sparse and dense abstractions AGREE for diagonal (decoupled) A (PD, PR exact, and
interior BA cells exact) — once the sparse grid is aligned to the dense point-grid
convention (`benchmarks/sparse_arch.cpp` now does this). For COUPLED A the sparse box
bound (product of per-dimension intervals) is a SOUND OVER-approximation (more
conservative; AS/BA), where the dense joint `nlopt` is tighter. So flipping to sparse
is SOUND everywhere and a memory/scalability win; on coupled systems it is slightly
more conservative until a tight JOINT mean enclosure is added to the sparse path.

## Open risk / remaining
- Coupled-A conservatism: add a joint per-cell enclosure to `buildSparseReachGeneral`
  so sparse == dense (tight) on coupled systems too (then "no accuracy loss" is literal).
- Architectural: route the examples / IMDP-class default through the sparse path; keep
  the dense SYCL path as an explicit (GPU) option.
