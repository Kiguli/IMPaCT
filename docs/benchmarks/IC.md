# IC — Integrator Chain (2D)

The IC benchmark is a discrete-time double integrator (a two-dimensional integrator
chain) with one bounded input and additive Gaussian noise. It comes in two variants
sharing the same dynamics: **IC_reach** (finite-horizon reachability of a central box)
and **IC_safe** (finite-horizon safety). Sources:
`examples/ARCH-COMP/2025/IC/IC_reach_2d.cpp` and `IC_safe_2d.cpp`.

## System

- **State** (2D): position and velocity, gridded over `[-10,10] x [-10,10]` with step
  `0.5` per dimension.
- **Input** (1D): acceleration over `[-1,1]` with step `0.5`.
- **Noise**: additive Gaussian, standard deviation `sqrt(0.01)` per dimension.
- **Specifications** (both robust/pessimistic, H = 5):
  - *IC_reach*: reach the box `[-8,8] x [-8,8]` within 5 steps;
  - *IC_safe*: remain within the state grid for 5 steps.

The dynamics are the sampled double integrator with sampling time `Ns = 0.1`:

```cpp
auto dynamics = [](const vec& x, const vec& u) -> vec {
    vec xx(dim_x);
    float Ns = 0.1;
    xx[0] = x[0] + Ns*x[1] + (Ns*Ns)/2.0*u[0];
    xx[1] = x[1] + Ns*u[0];
    return xx;
};
```

## Code walkthrough

Both variants share the setup:

```cpp
IMDP mdp(dim_x, dim_u, dim_w);
mdp.setStateSpace(ss_lb, ss_ub, ss_eta);
mdp.setInputSpace(is_lb, is_ub, is_eta);
mdp.setDynamics(dynamics);
mdp.setNoise(NoiseType::NORMAL);
mdp.setStdDev(sigma);
```

**IC_reach** labels the target region, abstracts target/avoid vectors and the
transition matrices, then synthesizes a robust finite-horizon reach controller:

```cpp
mdp.setTargetSpace(target_condition, true);
mdp.targetTransitionVectorBounds();
mdp.minAvoidTransitionVector();
mdp.maxAvoidTransitionVector();
mdp.transitionMatrixBounds();
mdp.exportIMDP("IC_reach.imdp");
mdp.finiteHorizonReachControllerSorted(true, 5);
mdp.saveController();
```

**IC_safe** needs no target set — only the avoid vectors (probability of leaving the
grid) and the transition matrices — and calls the safety synthesizer:

```cpp
mdp.minAvoidTransitionVector();
mdp.maxAvoidTransitionVector();
mdp.transitionMatrixBounds();
mdp.exportIMDP("IC_safe.imdp");
mdp.finiteHorizonSafeControllerSorted(true, 5);
mdp.saveController();
```

Build and run with `make && ./IC_reach_2d` (or `./IC_safe_2d`) in
`examples/ARCH-COMP/2025/IC`.

## Sparse-abstraction route

The integrator is affine with dynamics matrix `A = [[1, 0.1], [0, 1]]`, so the sparse
path in `benchmarks/sparse_arch.cpp` (`only=IC_reach` / `only=IC_safe`) computes the
per-dimension mean range exactly. With the grid aligned to the dense point-grid
convention the sparse cell counts match the dense exactly (592 reach / 1681 safe).

| bench | cells x act | nnz | sparse | dense | build + solve | sparse interior mean | dense (nlopt) |
|---|---|---|---|---|---|---|---|
| IC_reach (fin H5) | 1600 x 5 | 0.04M | 1.0 MB | 0.20 GB | 0.01 + 0.00 s | 0.057 | 0.072 |
| IC_safe (fin H5) | 1600 x 5 | 0.13M | 3.0 MB | 0.20 GB | 0.02 + 0.01 s | 0.750 | 0.761 |

IC_safe matches the dense values closely; the small IC_reach gap follows the coupled-A
analysis on the BA page (the sparse per-dimension product bound is sound but slightly
conservative when `A` is not diagonal).

## Results

Dense SYCL path (AdaptiveCpp/OMP, 56 GB host):

| case | cells | abstraction (s) | synthesis (s) |
|---|---|---|---|
| IC_reach | 592 | 1.8 | 0.06 |
| IC_safe | 1681 | 2.7 | 0.47 |

## Cross-tool validation

The exported IMDPs were solved by IntervalMDP.jl, PRISM and Storm on the identical
models. IMPaCT's per-cell robust values match IntervalMDP.jl to **Δ ≤ 6.3e-7** for
both variants. On the infinite-horizon reductions all four tools agree: robust reach
and safety are 0 at the initial state, and even the tiny optimistic reach value
agrees to high precision (IC_reach opt ≈ 1.20e-26 in all four). Solve times
(IMPaCT / IntervalMDP.jl / PRISM / Storm): IC_reach 0.1 / 0.4 / 0.2 / 0.05 s;
IC_safe 0.15 / 0.5 / 0.06 / 0.02 s.

## Notes

- The v1 status of IC was "reduced (scale)"; the v2 runs above cover the 2D
  configuration under both its native finite-horizon specs.
- PRISM required flooring `[0,hi]` interval edges to `[1e-9,hi]` (it rejects a zero
  lower bound), so its values are an ε-perturbation of the exported model.
