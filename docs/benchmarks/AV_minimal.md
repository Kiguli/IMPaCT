# AV_minimal — Autonomous Vehicle (3D)

The AV_minimal benchmark is a reduced three-dimensional autonomous vehicle — a
kinematic bicycle model with position and heading states, speed and steering inputs,
and wide additive Gaussian noise — under an infinite-horizon reach-avoid
specification. Like PR_minimal it is primarily an abstraction-scalability case: the
dense matrices are 7.42 GB each, while the sparse path builds and solves the model.
Source: `examples/ARCH-COMP/2025/AV_minimal/AV_minimal.cpp`.

## System

- **State** (3D): planar position and heading, gridded over
  `[-5,5] x [-5,5] x [-3.4,3.4]` with steps `{0.5, 0.5, 0.4}`.
- **Input** (2D): speed `[-1,4]` (step 1) and steering angle `[-0.4,0.4]` (step 0.2)
  — 30 discrete actions.
- **Noise**: additive Gaussian, standard deviation `sqrt(1/1.5)` (≈ 0.82) per
  dimension — wide relative to the 0.5 grid step.
- **Specification**: infinite-horizon reach-avoid — reach the lane region
  `[-5.75,0.25] x [-0.25,5.75] x [-3.45,3.45]` while avoiding the adjacent strip
  `[-5.75,0.25] x [-0.75,-0.25] x [-3.45,3.45]` (robust/pessimistic).

The dynamics are the nonlinear kinematic bicycle, sampled at `Ts = 0.1`:

```cpp
const auto dynamics = [](const vec& x, const vec& u) -> vec {
    float Ts = 0.100;
    vec f(3);
    vec xx = x;
    f[0] = u[0]*cos(atan((float)(tan(u[1])/2.0))+xx[2])/cos((float)atan((float)(tan(u[1])/2.0)));
    f[1] = u[0]*sin(atan((float)(tan(u[1])/2.0))+xx[2])/cos((float)atan((float)(tan(u[1])/2.0)));
    f[2] = u[0]*tan(u[1]);
    xx = xx + Ts*f;
    return xx;
};
```

## Code walkthrough

Standard reach-avoid pipeline — spaces, specification labelling, dynamics/noise,
abstraction, export, synthesis:

```cpp
IMDP mdp(dim_x, dim_u, dim_w);
mdp.setStateSpace(ss_lb, ss_ub, ss_eta);
mdp.setInputSpace(is_lb, is_ub, is_eta);
mdp.setTargetAvoidSpace(target_condition, avoid_condition, true);
mdp.setDynamics(dynamics);
mdp.setNoise(NoiseType::NORMAL);
mdp.setStdDev(sigma);

mdp.targetTransitionVectorBounds();
mdp.minAvoidTransitionVector();
mdp.maxAvoidTransitionVector();
mdp.transitionMatrixBounds();
mdp.exportIMDP("AV_minimal.imdp");

mdp.infiniteHorizonReachControllerSorted(true);   // true = pessimistic
mdp.saveController();
```

Build and run with `make && ./AV_minimal` in `examples/ARCH-COMP/2025/AV_minimal`.

## Sparse-abstraction route

AV_minimal is nonlinear in the state (through the heading), so the sparse path in
`benchmarks/sparse_arch.cpp` (`only=AV_minimal`) uses the sampled mean enclosure:
FAST mode evaluates the dynamics at the source-cell corners; EXACT mode
(`exact` / `exactK=K`) uses a K-per-dimension sub-grid for a tighter joint enclosure.
For non-affine dynamics this enclosure is heuristic (ISSUE-0021), validated by
K-convergence on the VP benchmark.

## Results

Dense SYCL path (56 GB host): 5562 cells; the two dense matrices are 7.42 GB each, so
the abstraction fits in 14.8 GB on the big machine (it OOMs on ordinary hosts); no
dense synthesis result is reported.

Sparse path (FAST mode, single-threaded):

| cells | actions | nnz | sparse | dense est. | build + solve | interior reach |
|---|---|---|---|---|---|---|
| 7938 | 30 | 324M | 7784 MB | 30.3 GB | 420 + 1161 s | [0, **0.3398**, mean 0.031] |

The abstraction **builds where the dense path OOMs** (7.78 GB versus an estimated
30 GB dense) and solves robustly to an interior maximum reach of 0.3398.

## Cross-tool validation

AV_minimal appears in the cross-tool study as an *abstraction-scalability* case:
building the IMDP from continuous dynamics is outside the scope of IntervalMDP.jl,
PRISM and Storm, and this model is not part of the four-tool solve sweep. The
exported `.imdp` uses the same neutral format validated on the smaller ARCH cases,
where all four tools agree on every robust value.

## Notes

- The wide noise (σ ≈ 0.82 against η = 0.5) gives a kernel of roughly 10 cells per
  dimension, hence ~10³ successors per row; that makes the single-threaded OVI solve
  heavy (1161 s), similar to PR_minimal.
- The modest robust reach value reflects the noise magnitude relative to the grid:
  with noise this wide it is genuinely hard to *robustly* steer into the target, so
  a low guaranteed lower bound is the sound answer, not an artefact.
- ISSUE-0021 (certified interval-arithmetic mean enclosure) applies to this case: the
  FAST corner sampling is a heuristic enclosure for non-affine dynamics.
