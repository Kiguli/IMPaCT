---
id: ISSUE-0020
title: Sparse (closed-form) and dense (nlopt) abstractions disagree on AS / BA
status: open
severity: medium
labels: correctness, abstraction, soundness, needs-monte-carlo
created: 2026-06-29
updated: 2026-06-29
related:
  - src/abstraction.cpp        # closed-form transitionInterval1D (exact for affine)
  - src/IMDP.cpp               # transitionMatrixImpl: nlopt (LN_SBPLX) per source cell
  - benchmarks/sparse_arch.cpp
  - issues/0007-abstraction-sink-too-loose.md
  - issues/0008-target-double-count.md
---

## Summary
Re-running the ARCH cases through the sparse abstraction (`abstraction.cpp`, closed
form) gives values that match the dense SYCL abstraction for IC_safe and PD, but
DIFFER markedly for AS and BA:

| case | spec | sparse interior mean | dense (nlopt) | ratio |
|---|---|---|---|---|
| AS | finite reach H10 | 0.0003 | 0.0045 | 15× |
| BA | finite safety H6 | 0.107 | 0.471 | 4.4× |
| IC_reach | finite reach H5 | 0.057 | 0.072 | ~ |
| IC_safe | finite safety H5 | 0.750 | 0.761 | ✓ |
| PD_p1/p3 | inf reach-avoid | 1.000 | 1.000 | ✓ |

## Hypothesis (which is correct?)
For AFFINE dynamics the **sparse** kernel is **closed-form EXACT**: the per-cell mean
range is exact (interval arithmetic on the affine map) and the 1-D Gaussian box mass
is unimodal in the mean, so its min/max over the mean range is closed form
(`transitionInterval1D`, verified vs brute force in `tests/unit/test_abstraction.cpp`).
The **dense** path computes the same per-cell min/max with **`nlopt` LN_SBPLX, a LOCAL
optimiser** (`src/IMDP.cpp::transitionMatrixImpl`). A local optimiser can miss the
global worst case over the cell → return a transition interval that is too tight →
OVER-optimistic safety / reach. BA's noise window (~7 cells, σ≈0.28–0.62) exceeds its
4-cell domain, so a lot of mass should leave (low safety): the sparse 0.11 is
plausible and the dense 0.47 may over-claim. If confirmed, the dense nlopt abstraction
is the inaccurate (potentially UNSOUND) one, and switching to sparse (ISSUE-0019) is an
accuracy fix, not a loss. (Connects to the abstraction-soundness bugs ISSUE-0007/0008.)

## Independent verification (done — Monte-Carlo, and what it does/doesn't show)
`benchmarks/crosstool/peers/mc_ba.py` simulates the BA closed loop under the
dense-synthesised policy with NOMINAL Gaussian noise, from cell centres AND from random
in-cell states:
```
dense_lo[mean]=0.471   empirical_nominal[mean]=0.78   sparse[mean]=0.107
cells where dense_lower > empirical (unsound vs nominal): 0 / 1225
```
So BOTH bounds are below the nominal empirical safety (necessary for soundness), and the
dense bound is never violated — even in-cell. **This refutes the simple "dense is
unsound" hypothesis vs NOMINAL noise.** BUT nominal MC does NOT validate a *robust*
(adversarial-nature) lower bound: the robust value uses the worst transition within the
intervals, which is not a physical noise, so robust_lo <= robust_value <= nominal, and
nominal MC cannot tell whether 0.471 is a valid robust lower bound.

The decisive remaining fact: for AFFINE dynamics the **sparse kernel is closed-form
EXACT over the whole source cell** (verified vs brute force in
`tests/unit/test_abstraction.cpp`). It is therefore the rigorous robust abstraction;
its conservatism (0.107) is the true worst-of (worst source state in the cell) × (worst
nature). The dense `nlopt` value (0.471) is LESS conservative — either a legitimately
tighter cell/point convention, or LN_SBPLX under-exploring the source cell (which would
make 0.471 an OVER-optimistic, non-sound robust bound). To decide, compare the ONE-STEP
min mass-in-domain for a BA cell: sparse closed form vs a brute-force max over sampled
source states vs the dense saved matrix entry. (Nominal MC, above, cannot.)

## Net so far
- Sparse = rigorously exact robust abstraction (verified), MORE conservative on AS/BA.
- Dense = tighter, sound vs nominal, robust-soundness depends on nlopt (unverified).
- "Make sparse default" (ISSUE-0019) trades tightness for rigor; if the dense nlopt is
  shown over-optimistic by the one-step brute-force check, it is a strict fix.

## Impact
Gates ISSUE-0019 (make sparse the default). Until resolved, "switch to sparse" may
change AS/BA results; the change is a FIX if the dense nlopt path is the inaccurate one.

## Resolution / plan
Run the Monte-Carlo decider; then either (a) confirm sparse correct → adopt sparse
default + downgrade/redefine the dense nlopt path, or (b) if sparse is loose somewhere,
tighten the sparse enclosure before defaulting.
