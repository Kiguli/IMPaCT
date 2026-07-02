# LM — Lane Merging Vehicle (6D)

The LM benchmark is a six-dimensional nonlinear lane-merging vehicle with two wheel
inputs, mostly deterministic dynamics (noise in only two of six dimensions), and a
long-horizon reachability specification (H = 200). It is the headline scalability
case of the suite: the dense abstraction is infeasible at roughly 96 GB per matrix,
while the sparse path builds and solves the model in about 40 s at 196 MB. Source:
`examples/ARCH-COMP/2025/LM/LM.cpp`.

## System

- **State** (6D): yaw rate, lateral velocity, planar position and two angle states,
  gridded over `[-0.5,0.5] x [-0.5,0.5] x [-10,4] x [-5,4] x [-3.14,3.14] x
  [-3.14,3.14]` with steps `{0.5, 0.5, 1, 1, 3.14, 3.14}`.
- **Input** (2D): wheel speeds over `[0.5,3.5] x [0.5,3.5]` with step `1.5` per
  dimension.
- **Noise**: Gaussian with standard deviation `{0, 0, 0, 0, 0.001, 0.001}` — four of
  the six dimensions are deterministic.
- **Specification**: finite-horizon reach-avoid with H = 200 — reach the merge point
  (a box collapsed to the corner `x2 = -10`, `x3 = -5`) while avoiding the occupied
  lane region `[-0.5,0.5] x ... x [-4,4] x [2.5,4] x [-3.14,3.14] x [-3.14,3.14]`.

The dynamics are a nonlinear single-track vehicle model sampled at `Ts = 0.1`
(excerpt; see the source for all six equations and constants):

```cpp
const auto dynamics = [](const vec& x, const vec& u) -> vec {
    float Ts = 0.1; float M = 700; /* D, a, b, d, Afy, Izz, Cgamma ... */
    vec xx = x;
    xx[0] = x[0] + (Ts/Izz)*(Afy*sin(x[5]+(pi/2))*a
                 + 2*Cgamma*tan((2*(x[1]-b*x[0])/(u[0]+u[1])))*b);
    // xx[1] .. xx[5]: lateral force balance, planar kinematics, angle updates
    xx[2] = x[2] + (Ts*(u[0]+u[1])/2)*cos(x[4]);
    xx[3] = x[3] + (Ts*(u[0]+u[1])/2)*sin(x[4]);
    return xx;
};
```

## Code walkthrough

The example follows the standard reach-avoid pipeline; the only difference from the
smaller cases is the long horizon in the synthesis call:

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
mdp.transitionMatrixBounds();          // infeasible on the dense path (~96 GB/matrix)
mdp.exportIMDP("LM.imdp");

mdp.finiteHorizonReachControllerSorted(true, 200);   // true = pessimistic
mdp.saveController();
```

In practice the dense build above cannot complete — use the sparse route below.

## Sparse-abstraction route

LM is only feasible through `benchmarks/sparse_arch.cpp` (`only=LM`). The dynamics
are nonlinear, so the sparse path samples the mean over each source cell: FAST mode
uses the 2^6 cell corners, EXACT mode (`exact` / `exactK=K`) a K-per-dimension
sub-grid (a heuristic enclosure for non-affine dynamics, ISSUE-0021). The driver
grid-aligns to the dense point-grid convention, applies the avoid region as dead
absorbing cells, and runs a 200-step robust value iteration.

## Results

Dense SYCL path: **infeasible** — 36450 cells, ~96 GB per dense matrix.

Sparse path (FAST mode, single-threaded):

| cells | actions | nnz | sparse | dense est. | build + solve | interior max |
|---|---|---|---|---|---|---|
| 12150 | 9 | 8.19M | **196 MB** | 21.3 GB | 24.8 + 14.9 s | 0.0000 |

The 6-D case that is dense-infeasible builds **and** solves through the sparse path
in about 40 s at 196 MB total.

## Cross-tool validation

LM appears in the cross-tool study as an *abstraction-scalability* case: no peer tool
(IntervalMDP.jl, PRISM, Storm) can abstract continuous dynamics, and the dense model
never existed to export, so LM is not part of the four-tool solve sweep. The sparse
`.imdp` export uses the same neutral format validated on the smaller ARCH cases.

## Notes

- **The robust value 0.0000 is sound but vacuous.** Four of six dimensions have
  σ = 0 (deterministic), so on this coarse grid the corner means of a cell scatter
  into different destination cells, producing transition intervals `[0,1]`; the
  pessimistic (robust) lower bound of any such interval chain is 0. A finer grid
  and/or the certified interval-arithmetic enclosure of ISSUE-0021 is needed to
  obtain a non-trivial robust value on near-deterministic systems.
- LM demonstrates why sparse storage is a prerequisite for higher-dimensional
  benchmarks: nnz grows linearly in the number of cells (8.19M here), while the
  dense min/max matrices grow quadratically (the driver's dense estimate is
  21.3 GB, and the dense IMDP class would need ~96 GB per matrix on its grid).
