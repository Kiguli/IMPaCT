# PD — Package Delivery (2D)

The PD benchmark is a two-dimensional package-delivery robot with stable diagonal
affine dynamics, two inputs and additive Gaussian noise. The ARCH-COMP specification
is co-safe LTL; the shipped examples collapse it into two infinite-horizon
reach-avoid problems with different targets — **PD_p1** (deliver to the first drop-off)
and **PD_p3** (deliver to the second) — around a common obstacle. Sources:
`examples/ARCH-COMP/2025/PD/PD_target_p1.cpp` and `PD_target_p3.cpp`.

## System

- **State** (2D): position, gridded over `[-6,6] x [-6,6]` with step `0.5`.
- **Input** (2D): velocity commands over `[-1,1] x [-1,1]` with step `0.1`
  (441 actions).
- **Noise**: additive Gaussian, standard deviation `sqrt(0.02)` per dimension.
- **Specifications** (infinite-horizon, robust reach-avoid):
  - *PD_p1*: reach `[5,6] x [-1,1]`, avoid `[0,1] x [-5,1]`;
  - *PD_p3*: reach `[-4,-2] x [-4,-3]`, avoid `[0,1] x [-5,1]`.

The dynamics are diagonal affine (each coordinate contracts independently):

```cpp
auto dynamics = [](const vec& x, const vec& u) -> vec {
    vec xx(dim_x);
    xx[0] = 0.9*x[0] + 1.4*u[0];
    xx[1] = 0.8*x[1] + 1.4*u[1];
    return xx;
};
```

## Code walkthrough

The reach-avoid specification is set in one call, which labels target and avoid
cells and makes them absorbing:

```cpp
IMDP mdp(dim_x, dim_u, dim_w);
mdp.setStateSpace(ss_lb, ss_ub, ss_eta);
mdp.setInputSpace(is_lb, is_ub, is_eta);
mdp.setTargetAvoidSpace(target_condition, avoid_condition, true);
mdp.setDynamics(dynamics);
mdp.setNoise(NoiseType::NORMAL);
mdp.setStdDev(sigma);
```

Abstraction and export follow the standard pattern:

```cpp
mdp.targetTransitionVectorBounds();
mdp.minAvoidTransitionVector();
mdp.maxAvoidTransitionVector();
mdp.transitionMatrixBounds();
mdp.exportIMDP("PD_p1.imdp");     // "PD_p3.imdp" in the p3 variant
```

Infinite-horizon synthesis. On PD_p3 the default interval iteration does **not**
converge (see Notes); switching to `ValueIteration` (or the OVI dispatch) does:

```cpp
//mdp.setIterationMethod(IterationMethod::ValueIteration);   // needed for PD_p3
mdp.infiniteHorizonReachController(true);                    // true = pessimistic
mdp.saveController();
```

Build and run with `make && ./PD_target_p1` (or `./PD_target_p3`) in
`examples/ARCH-COMP/2025/PD`.

## Sparse-abstraction route

PD's `A` is diagonal, so the sparse abstraction in `benchmarks/sparse_arch.cpp`
(`only=PD_p1` / `only=PD_p3`) is **exact** — sparse equals dense (grid-aligned cell
count 571, matching the dense abstraction).

| bench | cells x act | nnz | sparse | dense | build + solve | interior mean |
|---|---|---|---|---|---|---|
| PD_p1 (inf reach-avoid) | 576 x 441 | 5.1M | 122 MB | 2.34 GB | 0.8 + 15 s | 1.000 (dense 1.000) |
| PD_p3 (inf reach-avoid) | 576 x 441 | 5.1M | 122 MB | 2.34 GB | 0.7 + 1.1 s | 1.000 (peers 1.0) |

## Results

Dense SYCL path (AdaptiveCpp/OMP, 56 GB host):

| case | cells | abstraction (s) | synthesis (s) |
|---|---|---|---|
| PD_p1 | 571 | 37 | 1.29 (interval iteration) / 1.34 (value iteration; was 6.5) |
| PD_p3 | 571 | ~37 | 1.36 (value iteration; interval iteration non-convergent) |

## Cross-tool validation

The exported IMDPs were solved by IntervalMDP.jl, PRISM and Storm on the identical
models. The robust reach value is **1.0 in all four tools** for both PD_p1 and PD_p3;
for PD_p1 the per-cell robust values match IntervalMDP.jl to **Δ ≤ 7.1e-7**. Solve
times (IMPaCT / IntervalMDP.jl / PRISM / Storm): PD_p1 3.4 / 1.4 / 1.7 / 2.8 s;
PD_p3 3.6 / 1.6 / 2.3 / 2.2 s.

## Notes

- **ISSUE-0003 (nature trap):** PD_p3 exhibits a nature-confinable end component on
  which interval iteration's upper bound never contracts, so the default solver does
  not converge. Two remedies ship in IMPaCT: `IterationMethod::ValueIteration`
  (converges, residual stopping) and **OptimisticVI** (OVI — default on the sparse
  path, also available on the dense path via `infiniteHorizonOVIDispatch`), which is
  both convergent and certified: dense-OVI on PD_p3 returns the bracket
  [0.9998, 1.0] with gap 1e-5 in 3.3 s. Empirically, MEC-collapse preprocessing does
  *not* fix this — support-graph MECs miss nature-confinable end components; OVI or
  robust-EC deflation is required.
- The on-disk examples are the co-safe LTL specification collapsed into separate
  reach-avoid targets (p1/p3); replacing this with a true DFA-product co-safe run is
  the planned Phase-2 form of the benchmark.
