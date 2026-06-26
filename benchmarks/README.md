# IMPaCT v2.0 — Benchmark harness (ARCH-COMP Stochastic Models)

Interleaved benchmarking: every capability we add ships with the ARCH-COMP
Stochastic-Models benchmark(s) that exercise it, so the paper's evaluation
assembles itself with a clear feature→benchmark progression.

- `run_archcomp.py` — builds + runs the `examples/ARCH-COMP/2025/*` cases,
  records wall-clock and output presence, appends rows to `RESULTS.md`.
  Skips cleanly if the `acpp` toolchain is unavailable.
- `RESULTS.md` — the running results table (baseline first, then per-phase).

## Feature → benchmark progression

| Phase | Capability | Benchmarks that demonstrate it |
|---|---|---|
| 0 | baseline reproduction | AS, BA, HT, VP, AV(min), PR(min), IC, LM |
| 1 | sound II + O-maximization (speed+rigor) | same set: same results within ε, report speedup vs GLPK |
| 2 | co-safe LTL / LTLf (DFA product) | **PD, PDx at full scLTL** (remove "reduced-spec") |
| 3 | full ω-regular (robust accepting ECs) | **new variants**: patrol recurrence □◇, reach-then-stay ◇□ |

Reference: ARCH-COMP25 Stochastic Models report (verified entry in
`../paper/References.bib`). Benchmark sources: gitlab.com/goranf/ARCH-COMP.

## Usage
```bash
python3 benchmarks/run_archcomp.py            # run all available ARCH cases
python3 benchmarks/run_archcomp.py --only VP  # single case
```

## v2.0 verification & scaling benchmarks (self-contained C++)
Each links a few `src/*.cpp` files (no SYCL/HDF5 toolchain needed) and exits 0 on pass.

| File | What it does / verifies |
|---|---|
| `sparse_scaling.cpp` | O(nnz) memory: 125k-state synthesis ~255 MB / 0.4 s vs dense ~250 GB |
| `validate_montecarlo.cpp` | reach: robust controller, empirical reach >= robust lower bound (affine 2-D) |
| `validate_vp.cpp` | nonlinear ARCH Van der Pol: empirical reach within [lower,upper] (interval-MC) |
| `validate_cosafe.cpp` | co-safe LTL F(pickup & F deliver) over a continuous robot; empirical >= lower |
| `validate_safety.cpp` | safety (never enter avoid box); empirical safety >= robust lower bound |

Build one, e.g.:
```bash
c++ -std=c++17 -O2 benchmarks/validate_vp.cpp \
    src/abstraction.cpp src/solve.cpp src/omaximization.cpp src/graph_utils.cpp -o /tmp/vp && /tmp/vp
```
The Monte-Carlo validators simulate the *continuous* closed loop and check the
empirical outcome against the synthesized bounds — this is how ISSUE-0007 and
ISSUE-0008 (abstraction soundness bugs) were found. Note: an infinite-horizon
objective needs a long simulation horizon to validate (truncation under-reports).

