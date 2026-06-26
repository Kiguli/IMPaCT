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

## Baseline (v1, branch main) — TODO: fill via Docker/CI
```
# paste run_archcomp.py output here
```

## Phase 1 (sound II + O-maximization) — TODO
## Phase 2 (co-safe LTL) — TODO
## Phase 3 (full ω-regular) — TODO
