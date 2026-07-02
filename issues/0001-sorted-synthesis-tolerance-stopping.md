---
id: ISSUE-0001
title: v1 "sorted" synthesis uses tolerance stopping, not sound interval iteration
status: resolved
severity: medium
labels: soundness, tool-v1, correctness
created: 2026-06-25
updated: 2026-06-25
related:
  - src/GPU_synthesis.cpp
  - src/IMDP.cpp
  - tests/unit/test_interval_iteration.cpp
  - HaddadMonmege2018IntervalIteration
  - Baier2017IntervalIteration
---

## Summary
The inherited v1 infinite-horizon synthesis (sorted and LP variants) iterates
value iteration with a difference-based stopping test (`while max_diff > epsilon`)
rather than sound interval iteration. On end components / absorbing structure the
two bounds can converge to each other without converging to the true value; the
code detects "both bounds converged but not to each other" and prints a warning
suggesting the user switch to a finite-horizon run. This means infinite-horizon
results are not guaranteed sound in general.

## Details / evidence
- `src/GPU_synthesis.cpp` ~L186–188: after the lower/upper VI loops, an
  `approx_equal` check prints: "Bounds both converged ... but they did not
  converge to each other. It is likely there is an absorbing state ... try ...
  the finite Horizon solution."
- The stopping condition is `max_diff > epsilon` on the change between iterates,
  which does not bound distance to the true fixpoint (classic VI unsoundness).

## Reproduction
End-component-with-no-exit / absorbing structures; captured abstractly by
`tests/unit/test_interval_iteration.cpp` Models 4 and 5 (the contracts that
require Prob0/Prob1 + MEC handling).

## Independent verification
Oracle reachability values for the affected models are computed independently
(numpy MC solve / hand derivation) in `tests/oracles/oracles.py` and the
interval-iteration contracts.

## Impact
Soundness of infinite-horizon reach/safe synthesis. Not a regression we
introduced — it is inherited v1 behaviour we are fixing.

## Resolution / plan
Phase 1c: implement sound robust interval iteration (lower from 0 / upper from 1,
robust Bellman on both, stop at gap < 2*eps) with Prob0/Prob1 precomputation and
MEC collapsing (`src/graph_utils.cpp` provides the MEC infra). Wire into the
synthesis path and confirm via the golden regression net once goldens exist.

## RESOLVED (2026-07-02)
The dense path now has THREE selectable infinite-horizon engines (`setIterationMethod`):
IntervalIteration (sound two-sided bracket, the ISSUE-0017 rework), ValueIteration
(peer-style residual stopping, explicitly labelled as certificate-free), and
**OptimisticVI** (`infiniteHorizonOVIDispatch`, Hartmanns-Kaminski CAV 2020) which returns
a CERTIFIED bracket [L, U] with a verified inductive upper bound — including on the
nature-trap/absorbing structures that defeated the v1 tolerance stopping (PD_p3: certified
[0.9998, 1.0], gap 1e-5, where the v1-style loop printed the "did not converge to each
other" warning). Docs (manual/engines.md) tell users which engine gives which guarantee.
The unsound tolerance-stop is no longer the only option anywhere; soundness gap closed.
