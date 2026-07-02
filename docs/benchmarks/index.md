# ARCH-COMP Benchmarks

IMPaCT ships the ARCH-COMP 2025 stochastic-models benchmark suite under
`examples/ARCH-COMP/2025/`. Each case abstracts a continuous stochastic system into an
interval MDP (IMDP) and synthesizes a robust controller; the abstracted IMDP is exported
so that peer tools (IntervalMDP.jl, PRISM, Storm) can solve the identical model for
cross-tool validation. This section documents each case study, the code used to solve
it, the measured results, and the design decisions behind them.

## Benchmark overview

| Case | Dim | Specification | Cells | Key result | Page |
|---|---|---|---|---|---|
| AS (anaesthesia) | 3 | finite-horizon reach, H=10 | 2460 | matches IntervalMDP.jl to Δ≤4.8e-7 | [AS](AS.md) |
| BA (building automation) | 4 | finite-horizon safety, H=6 | 1225 | init safety 0.4533 in IMPaCT / IntervalMDP.jl / Storm | [BA](BA.md) |
| VP (Van der Pol) | 2 | infinite-horizon reach | 2591 | max reach 0.2297; VI solve 0.59 s (was 214 s) | [VP](VP.md) |
| IC (integrator chain) | 2 | finite reach / safety, H=5 | 592 / 1681 | matches IntervalMDP.jl to Δ≤6.3e-7 | [IC](IC.md) |
| PD (package delivery) | 2 | infinite reach-avoid (p1, p3) | 571 | robust value 1.0 in all four tools | [PD](PD.md) |
| PR_minimal (patrol robot) | 2 | infinite reach-avoid | 1583 | former dense-OOM case completes via sparse path (3.86 GB) | [PR_minimal](PR_minimal.md) |
| AV_minimal (autonomous vehicle) | 3 | infinite reach-avoid | 7938 (sparse) | builds where dense path OOMs; interior max reach 0.3398 | [AV_minimal](AV_minimal.md) |
| LM (lane merging) | 6 | finite-horizon reach, H=200 | 12150 (sparse) | dense-infeasible (~96 GB/matrix); sparse 196 MB, ~40 s | [LM](LM.md) |

All robust (pessimistic) values agree across IMPaCT, IntervalMDP.jl, PRISM 4.8.1 and
Storm 1.13.0 wherever a comparison is applicable; details are on each page and in the
top-level `TOOL_COMPARISONS.md`.

## Building and running the dense examples

Each benchmark directory contains a Makefile that compiles the example against the
IMPaCT sources with AdaptiveCpp (SYCL):

```bash
cd examples/ARCH-COMP/2025/AS   # or BA, VP, IC, PD, PR_minimal, AV_minimal, LM
make
./AS                            # executable named after the .cpp file
```

The Makefile uses `acpp` with an OpenMP target by default and links NLopt, HDF5, GLPK,
GSL and Armadillo:

```makefile
CC = acpp # or syclcc (then remove --acpp-targets="omp")
CFLAGS = --acpp-targets="omp" -O3 -lnlopt -lm -I/usr/include/hdf5/serial \
         -L/usr/lib/$(ARCH)-linux-gnu/hdf5/serial -lhdf5 -lglpk -lgsl -lgslcblas \
         -DH5_USE_110_API -larmadillo
```

Each example exports its abstracted IMDP (`mdp.exportIMDP("<name>.imdp")`) before
synthesis, which is the input used for the cross-tool comparison.

## Building and running the sparse driver

`benchmarks/sparse_arch.cpp` re-runs all cases through IMPaCT's sparse abstraction
(`src/abstraction.cpp`, O(nnz) memory) and the sparse OVI solver (`src/solve.cpp`). It
is pure standard C++ — no SYCL or Armadillo required:

```bash
c++ -std=c++17 -O2 benchmarks/sparse_arch.cpp \
    src/abstraction.cpp src/solve.cpp src/omaximization.cpp src/graph_utils.cpp \
    -o /tmp/sparse_arch
```

Run modes:

```bash
/tmp/sparse_arch              # FAST enclosure (corner-min + per-dimension max)
/tmp/sparse_arch exact        # EXACT/TIGHT enclosure (K=5-per-dim sub-grid)
/tmp/sparse_arch exactK=7     # custom sub-grid density
/tmp/sparse_arch only=VP      # run a single benchmark by name
```

The driver aligns the sparse grid to the dense point-grid convention (cells centred on
`lb + k*eta`, `(ub-lb)/eta + 1` cells per dimension), so cell counts match the dense
abstraction exactly and values are directly comparable. For diagonal (uncoupled)
dynamics matrices the sparse mean enclosure is exact; for coupled dynamics it is a
sound over-approximation (see the BA page). For nonlinear dynamics the FAST/EXACT
modes control the tightness of the sampled mean enclosure (see the VP page).

## Cross-tool validation

Only IMPaCT can abstract continuous dynamics into an IMDP; the peers solve IMPaCT's
exported model. The pipeline is: `exportIMDP` → neutral `.imdp` file →
`imdp_to_drn.py` (Storm), `imdp_to_prism_interval.py` (PRISM),
`intervalmdp_runner.jl` (IntervalMDP.jl), driven end-to-end by
`benchmarks/crosstool/peers/sweep_compare.py`. All four tools agree on every ARCH
robust value; under the native finite horizons IMPaCT's per-cell robust values match
IntervalMDP.jl to ≤ 2.3e-5 (≤ 7e-7 for the finite-horizon cases).
