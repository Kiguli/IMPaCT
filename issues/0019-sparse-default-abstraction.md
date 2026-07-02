---
id: ISSUE-0019
title: Make the sparse abstraction the default everywhere (design decision)
status: resolved
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

## Decision refinement (user, 2026-06-29): default to the FASTER path; expose BOTH.
Realized in `benchmarks/sparse_arch.cpp`:
- **FAST (default):** sparse joint enclosure with corner-min (exact joint min, log-concave)
  + per-dimension max (sound). Very fast; sound; slightly conservative on coupled A.
- **EXACT (`exact` / `exactK=N`):** K-per-dimension sub-grid min AND max over each source
  cell -> the tight joint abstraction. Matches the dense for noise ~ grid (BA finite-safety
  0.378 -> **0.468 ~ dense 0.471**), at higher cost (BA 8 s -> 224 s).
- **dense SYCL nlopt** remains a backend, and is genuinely the best for the TINY-NOISE
  regime (sigma << eta, e.g. AS: sigma 0.03, eta 1): there a fixed sub-grid under-samples
  the sharp kernel, so the dense's CONTINUOUS optimiser is more accurate (AS sparse 0.0019
  vs dense 0.0045 even in exact mode). So all three are useful; default = fastest = sparse.

## Status / plan
- [x] Affine ARCH cases rerun via the sparse path; grid aligned to the dense convention.
- [x] Joint enclosure (fast + exact) implemented; exact matches dense on noise~grid (BA).
- [x] Accuracy gate (ISSUE-0020) resolved: agree for diagonal A and noise~grid; coupled-A
  conservatism removed by exact mode; tiny-noise is a dense-nlopt strength.
- [x] Nonlinear ARCH cases wired through the sparse joint enclosure (point-sampling, K-per-dim):
      VP (FAST 0.1985 → EXACT 0.2143, K-stable, ≈ dense 0.2297), LM (6-D, dense-INFEASIBLE,
      builds+solves at 196 MB), AV_minimal (builds where dense OOMs). The point-sampling mean
      range is a HEURISTIC for non-affine `f` (so is the dense nlopt) — certified
      interval-arithmetic enclosure tracked in **ISSUE-0021**.
- [ ] Wire the example/recommended workflow to default to the sparse path with a
  fast/exact/dense switch; keep dense SYCL (GPU + tiny-noise) as an explicit option.

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

## RESOLVED (2026-07-02)
The design decision is realized: the sparse abstraction + portable solver is the
documented default/recommended path (benchmarks, crosstool harness, docs manual), with
FAST (corner-min + per-dim max) default and EXACT (K-subgrid) / dense-SYCL (GPU,
tiny-noise) as explicit options. Nonlinear cases wired (VP/AV/LM); certified enclosures
tracked separately (ISSUE-0021). Moving the sparse builder INSIDE the legacy IMDP class
is an enhancement, not part of this decision — the class gained exportIMDP + OVI dispatch
instead, which routes any dense abstraction through the same verified solver.
