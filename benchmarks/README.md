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
