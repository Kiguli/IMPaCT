# AS — Anaesthesia Delivery (3D)

The AS benchmark models automated anaesthesia delivery as a three-compartment
pharmacokinetic system with affine (linear) dynamics, two control inputs and additive
Gaussian noise. The specification is finite-horizon reachability: steer the drug
concentrations into a target box within 10 steps. Source:
`examples/ARCH-COMP/2025/AS/AS.cpp`.

## System

- **State** (3D): compartment concentrations, gridded over
  `[1,6] x [0,10] x [0,10]` with steps `{0.25, 1, 1}`.
- **Input** (2D): infusion rates over `[0,7] x [0,30]` with steps `{1, 30}`.
- **Noise**: additive Gaussian, standard deviation `sqrt(0.001)` in each dimension.
- **Specification**: reach the target box `[4,6] x [8,10] x [8,10]` within horizon
  H = 10 (robust/pessimistic).

The dynamics are affine in the state and inputs:

```cpp
auto dynamics = [](const vec& x, const vec& u) -> vec {
    vec xx(dim_x);
    xx[0] = 0.8192*x[0] + 0.03412*x[1] + 0.01265*x[2] + 0.01883*(u[0] + u[1]);
    xx[1] = 0.01646*x[0] + 0.9822*x[1] + 0.0001*x[2]  + 0.0002*(u[0] + u[1]);
    xx[2] = 0.0009*x[0]  + 0.00002*x[1] + 0.9989*x[2] + 0.00001*(u[0] + u[1]);
    return xx;
};
```

## Code walkthrough

Create the IMDP object and discretize the state and input spaces:

```cpp
IMDP mdp(dim_x, dim_u, dim_w);
mdp.setStateSpace(ss_lb, ss_ub, ss_eta);
mdp.setInputSpace(is_lb, is_ub, is_eta);
mdp.setTargetSpace(target_condition, true);   // label target cells, make absorbing
```

Attach the dynamics and the noise model:

```cpp
mdp.setDynamics(dynamics);
mdp.setNoise(NoiseType::NORMAL);
mdp.setStdDev(sigma);
```

Build the abstraction — transition-probability intervals to the target region, to the
avoid region (here: leaving the grid), and between all cell pairs:

```cpp
mdp.targetTransitionVectorBounds();
mdp.minAvoidTransitionVector();
mdp.maxAvoidTransitionVector();
mdp.transitionMatrixBounds();
```

Export the abstracted IMDP for cross-tool comparison, then synthesize the
finite-horizon controller (`true` = pessimistic/robust) and save it:

```cpp
mdp.exportIMDP("AS.imdp");
mdp.finiteHorizonReachControllerSorted(true, 10);
mdp.saveController();
```

Build and run with `make && ./AS` in `examples/ARCH-COMP/2025/AS`.

## Sparse-abstraction route

AS is affine in the state, so the per-cell mean range is exact and the sparse
abstraction (`benchmarks/sparse_arch.cpp`, run with `only=AS`) has no accuracy loss
per dimension. The driver aligns the sparse grid to the dense point-grid convention
so the cell counts match the dense abstraction exactly. Note that AS has a
*coupled* (non-diagonal) dynamics matrix: the sparse per-dimension product bound is
sound but slightly conservative compared with the dense joint NLopt optimisation
(see the BA page and ISSUE-0020 for the analysis).

## Results

Dense SYCL path (AdaptiveCpp/OMP, 56 GB host):

| cells | abstraction (s) | synthesis (s) |
|---|---|---|
| 2460 | 27.0 | 4.3 |

Sparse path (single-threaded, pure C++): 0.83M nonzeros, **19.9 MB** versus a
1.02 GB dense matrix, building in 0.11 s and solving in **0.07 s** (versus 4.3 s
dense SYCL). After grid alignment the sparse interior mean is 0.0019 against the
dense (NLopt) 0.0045 — sound but conservative, as expected for coupled dynamics.

## Cross-tool validation

The exported `AS.imdp` was solved by IntervalMDP.jl, PRISM and Storm on the identical
model. Under the native finite horizon (H = 10) IMPaCT's per-cell robust values match
IntervalMDP.jl to **Δ ≤ 4.8e-7**. On the infinite-horizon reachability reduction all
four tools return robust value 0 at the initial state (nothing is robustly reached in
the limit), with solve times 0.4 / 0.5 / 0.5 / 0.05 s
(IMPaCT / IntervalMDP.jl / PRISM / Storm).

## Notes

- The dense finite-horizon path recomputed a large `diff` matrix each sweep; the
  loop-invariant hoist was applied to the infinite-horizon functions, which is why
  the sparse solve (0.07 s) is much faster than the dense SYCL solve (4.3 s) here.
- PRISM rejects zero lower-bound interval edges, so its run used intervals floored to
  `[1e-9, hi]` — an ε-perturbation of the exported model.
