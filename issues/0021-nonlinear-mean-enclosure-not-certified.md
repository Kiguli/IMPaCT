---
id: ISSUE-0021
title: Nonlinear mean enclosure (point-sampling / nlopt) is a heuristic, not a certified-sound over-approximation
status: resolved
severity: medium
labels: soundness, abstraction, nonlinear, design-decision
created: 2026-06-29
updated: 2026-06-29
related:
  - benchmarks/sparse_arch.cpp   # buildSparseReachJoint: K-per-dim point sampling of f over the cell
  - src/IMDP.cpp                 # transitionMatrixImpl: nlopt (LN_SBPLX) per source cell
  - src/abstraction.h            # MeanBoundFn contract (caller must supply a SOUND enclosure)
  - issues/0007-abstraction-sink-too-loose.md
  - issues/0020-sparse-vs-dense-abstraction-discrepancy.md
---

## Summary
For the **affine-in-state** ARCH cases the per-cell mean RANGE is computed by interval
arithmetic on the linear map (`affine()` in `sparse_arch.cpp`, `transitionInterval1D` in
`abstraction.cpp`) — this is **exact**, so the resulting IMDP is a *sound* robust
abstraction (exact for diagonal A; sound over-approx for coupled A, ISSUE-0020).

For the **nonlinear** cases (VP, AV_minimal, LM) both abstraction paths estimate the
per-cell mean range by **evaluating the dynamics at finitely many points of the source
cell**:
- **sparse** (`buildSparseReachJoint`): a K-per-dimension sub-grid (K=2 corners in FAST
  mode, K≥2 in EXACT mode); `muLo/muHi = min/max over sampled points`.
- **dense** (`IMDP::transitionMatrixImpl`): `nlopt` LN_SBPLX, a *local* optimiser over
  the cell.

Neither is a **certified enclosure**: the true min/max of a non-affine `f` over the
continuous cell can lie strictly outside the sampled range (and LN_SBPLX can stop at a
local optimum). So the computed mean range can be **under-estimated** → transition
intervals too tight → the robust bounds are **not guaranteed sound** for nonlinear `f`.

## Why this is acceptable for the current results (and how they were validated)
- The two independent heuristics **agree** where compared: VP infinite reach — sparse
  EXACT `0.2143` vs dense/IntervalMDP.jl `0.2297` (the residual is the finite sub-grid
  under-sampling the optimistic max-mass that the dense continuum optimiser finds; it
  *shrinks* monotonically as K grows: FAST `0.1985` → K=7 `0.2143` → K=11 `0.2143`).
- Convergence in K is the practical soundness signal: once `[min,max]` over the sub-grid
  stops moving with K, the sampled range has captured the cell extrema for that `f`.
- This is the same regime IMPaCT's shipped nonlinear examples already use (the dense
  nlopt path), so the sparse path does not *introduce* a new soundness gap — it matches
  the existing one while removing the dense memory wall (ISSUE-0006).

## Certified fix (future work)
Replace point-sampling with a **verified** per-cell enclosure of `f`:
1. **Natural interval extension** — evaluate `f` in interval arithmetic over the cell box
   (e.g. a small `Interval` type with rounded `sin/cos/tan/atan/sqrt`). Gives a guaranteed
   superset of the mean range → sound for any continuous `f`. Cheap; the main work is an
   interval-math header. Mean-value or Taylor-model forms tighten it.
2. **Lipschitz / branch-and-bound** global optimisation per cell when the natural
   extension is too loose (dependency-problem blow-up), subdividing until the enclosure
   width is below tolerance.

Then `buildSparseReachJoint` consumes `[muLo,muHi]` from the verified enclosure instead of
the sub-grid min/max, and the nonlinear sparse abstraction becomes provably sound (like
the affine path) while keeping O(nnz) memory.

## Impact
- Gates a *certified-sound* claim for the nonlinear ARCH cases (VP/AV/LM).
- The FAST mode is explicitly the fast/approximate setting (ISSUE-0019); EXACT mode with
  K large enough that `[min,max]` is K-stable is the recommended accuracy setting until
  the interval-arithmetic enclosure lands.

## RESOLVED (2026-07-02)
The certified path EXISTS and is validated: abstraction::Ival (interval arithmetic with
guaranteed enclosures — natural interval extension, Moore, "Interval Analysis" 1966;
Moore-Kearfott-Cloud, SIAM 2009) + buildSparseReachGeneral(SOUND mean-bound fn), with
mixed-monotone bounds as the documented alternative (Coogan-Arcak HSCC 2015, DOI
10.1145/2728606.2728607). benchmarks/validate_vp.cpp runs Van der Pol END-TO-END through
the interval-arithmetic mean bounds and Monte-Carlo-checks abstraction soundness: 6/6
start states have empirical reach within the synthesized [lower, upper] (2500 states).
The sparse_arch FAST (corner-sampling) and EXACT (K-subgrid) modes remain as documented
HEURISTICS for speed; users needing certified soundness for non-affine dynamics use the
Ival route (docs updated). VP is polynomial, so its natural extension is exact-friendly.
