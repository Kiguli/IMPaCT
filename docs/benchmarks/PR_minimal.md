# PR_minimal — Patrol Robot (2D)

The PR_minimal benchmark is a reduced two-dimensional patrol robot commanded by speed
and heading, with wide additive Gaussian noise, under an infinite-horizon reach-avoid
specification. It is the tool's canonical *memory-wall* case: the dense abstraction
matrices reach 8.84 GB (an out-of-memory failure under a 4 GB cap, ISSUE-0006), and it
is the case that motivated the sparse abstraction path. Source:
`examples/ARCH-COMP/2025/PR_minimal/PR_minimal.cpp`.

## System

- **State** (2D): position, gridded over `[-10,10] x [-10,10]` with step `0.5`.
- **Input** (2D): speed and heading over `[-1,1] x [-1,1]` with step `0.1`
  (441 actions).
- **Noise**: additive Gaussian, standard deviation `sqrt(1/1.3333)` (≈ 0.87) per
  dimension — very wide relative to the 0.5 grid step.
- **Specification**: infinite-horizon reach-avoid — reach `[5,8] x [5,8]` while
  avoiding `[-1.5,1.5] x [-1.5,1.5]` (robust/pessimistic).

The dynamics are affine in the state (identity) with a nonlinear input map:

```cpp
auto dynamics = [](const vec& x, const vec& u) -> vec {
    vec xx(dim_x);
    xx[0] = x[0] + 2*u[0]*cos(u[1]);
    xx[1] = x[1] + 2*u[0]*sin(u[1]);
    return xx;
};
```

## Code walkthrough

Standard reach-avoid setup, abstraction and export:

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
mdp.exportIMDP("PR_minimal.imdp");

mdp.infiniteHorizonReachControllerSorted(true);   // true = pessimistic
mdp.saveController();
```

Build and run with `make && ./PR_minimal` in `examples/ARCH-COMP/2025/PR_minimal`.
Note that the dense path needs a large-memory host (see Results).

## Sparse-abstraction route

Because the state part of the dynamics is the identity (diagonal), the sparse mean
enclosure in `benchmarks/sparse_arch.cpp` (`only=PR_minimal`) is **exact** — the
sparse abstraction matches the dense with no accuracy loss, at a fraction of the
memory. The nonlinearity lives entirely in the input map, which is evaluated per
discrete action. The driver skips PR_minimal in `fast` mode (its wide kernel makes it
the slowest case); run it explicitly with `only=PR_minimal`.

## Results

Dense SYCL path (AdaptiveCpp/OMP, 56 GB host): 1583 cells, abstraction ~533 s using
two 8.84 GB dense matrices; the SYCL value iteration did not converge on this model.
Under a 4 GB memory cap the dense path is killed outright — the v1 baseline builds a
dense (state x input) x state matrix of 698103 x 1583 ≈ 8.84 GB (ISSUE-0006).

Sparse path (single-threaded, pure C++):

| cells x act | nnz | sparse | dense | build + solve | interior value |
|---|---|---|---|---|---|
| 1600 x 441 | 161M | **3.86 GB** | 18.1 GB | 28.5 + 814 s | ~1.0 |

This is the headline of the sparse rerun: **the ISSUE-0006 dense-OOM case now
completes** (3.86 GB versus 18 GB; it would not run at all on the dense path). The
814 s solve time reflects the very wide noise kernel — σ ≈ 0.87 against η = 0.5
yields ~217 nonzeros per row — under single-threaded optimistic value iteration.

## Cross-tool validation

PR_minimal appears in the cross-tool study as an *abstraction-scalability* case: the
peer tools (IntervalMDP.jl, PRISM, Storm) cannot abstract continuous dynamics at all,
so building this IMDP is outside their scope, and it is not part of the four-tool
solve sweep. The exported `.imdp` uses the same neutral format validated on the other
ARCH cases, where all four tools agree on every robust value.

## Notes

- **ISSUE-0006:** the dense abstraction stores a (state x input) x state matrix for
  each of the min/max bounds; PR_minimal was the smallest shipped example that made
  this infeasible on ordinary hardware, confirming the need for sparse storage.
- The grid-aligned sparse abstraction is exact here (diagonal state dynamics), so the
  sparse route is strictly preferable: identical model, 5x less memory, and it
  completes the synthesis the dense SYCL solver could not.
