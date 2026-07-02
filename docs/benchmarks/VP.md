# VP — Van der Pol Oscillator (2D)

The VP benchmark is an autonomous (input-free) two-dimensional Van der Pol oscillator,
discretized in time with step 0.1, under additive Gaussian noise. The specification is
infinite-horizon reachability of a small target box, making this a verification (rather
than control-synthesis) problem and the primary validation case for IMPaCT's nonlinear
abstraction. Source: `examples/ARCH-COMP/2025/VP/VP.cpp`.

## System

- **State** (2D): gridded over `[-5,5] x [-5,5]` with step `0.2` per dimension.
- **Input**: none (`dim_u = 0`).
- **Noise**: additive Gaussian, standard deviation `sqrt(0.04)` per dimension.
  (A custom uniform PDF is included in the source, commented out, as an example of
  `NoiseType::CUSTOM`.)
- **Specification**: infinite-horizon reachability of
  `[-1.2,-0.9] x [-2.9,-2.0]` (robust/pessimistic).

The dynamics are the Euler-discretized Van der Pol equations — genuinely nonlinear:

```cpp
auto dynamics = [](const vec& x) -> vec {
    vec xx(dim_x);
    xx[0] = x[0] + 0.1*x[1];
    xx[1] = x[1] + (-x[0] + (1 - x[0]) * (1 - x[0]) * x[1]) * 0.1;
    return xx;
};
```

## Code walkthrough

No input space is set — this is verification of an autonomous system:

```cpp
IMDP mdp(dim_x, dim_u, dim_w);
mdp.setStateSpace(ss_lb, ss_ub, ss_eta);
mdp.setDynamics(dynamics);
mdp.setNoise(NoiseType::NORMAL);
mdp.setStdDev(sigma);
mdp.setTargetSpace(target_condition, true);
```

Abstraction — because the dynamics are nonlinear, the per-cell transition bounds are
computed by NLopt (a local optimiser, LN_SBPLX by default; `setAlgorithm` can change
it):

```cpp
mdp.targetTransitionVectorBounds();
mdp.minAvoidTransitionVector();
mdp.maxAvoidTransitionVector();
mdp.transitionMatrixBounds();
mdp.exportIMDP("VP.imdp");
```

Infinite-horizon synthesis. The default solver is sound interval iteration; the
example notes that `ValueIteration` (peer-style pure VI) uses far fewer sweeps:

```cpp
//mdp.setIterationMethod(IterationMethod::ValueIteration);   // optional
mdp.infiniteHorizonReachControllerSorted(true);              // true = pessimistic
mdp.saveController();
```

Build and run with `make && ./VP` in `examples/ARCH-COMP/2025/VP`.

## Sparse-abstraction route

VP is the validation case for the sparse *nonlinear* path in
`benchmarks/sparse_arch.cpp` (`only=VP`). The mean range over each source cell is
estimated by evaluating the dynamics on a K-per-dimension sub-grid:

- **FAST** (default): evaluate at the 2^dim cell corners;
- **EXACT** (`exact`, K=5, or `exactK=K`): a K-per-dimension sub-grid giving a tight
  joint min/max.

| mode | cells | nnz | sparse (MB) | build + solve | interior max | dense ref |
|---|---|---|---|---|---|---|
| FAST | 2601 | 0.20M | 4.8 | 0.1 + 0.5 s | 0.1985 | 0.2297 |
| EXACT K=7/11 | 2601 | 0.20M | 4.8 | 0.6 + 0.5 s | **0.2143** | 0.2297 |

The enclosure is K-stable at K = 7 (unchanged at K = 11) and close to the
dense/IntervalMDP.jl reference 0.2297. The residual gap is the finite sub-grid
under-sampling the optimum the dense continuum optimiser finds — expected for
non-affine dynamics (ISSUE-0021: for nonlinear `f` both the sampled sparse enclosure
and the dense NLopt bound are heuristic; the validation is K-convergence plus
agreement with the dense/peer value).

## Results

Dense SYCL path (AdaptiveCpp/OMP, 56 GB host):

| cells | abstraction (s) | synthesis (s) |
|---|---|---|
| 2591 | 1.9 | 29.1 (interval iteration) / **0.59** (value iteration; was 214) |

The maximum reachability value over the grid is **0.2297** (matched by the peer
tools). The value-iteration solve at 0.59 s is at or below the peers' solve times.

## Cross-tool validation

The exported `VP.imdp` was solved by IntervalMDP.jl, PRISM and Storm on the identical
model. IMPaCT's per-cell robust values match IntervalMDP.jl to **Δ ≤ 2.3e-5**. The
robust value at the initial state is 0 in all four tools, with solve times
1.8 / 0.55 / 1.4 / 0.04 s (IMPaCT / IntervalMDP.jl / PRISM / Storm).

## Notes

- VP was the motivating case for adding `IterationMethod::ValueIteration`: interval
  iteration is a sound two-sided bracket but took 214 s before the loop-invariant
  hoist (now 29.1 s); pure VI converges in 0.59 s with residual-based stopping.
- ISSUE-0021 tracks a certified interval-arithmetic mean enclosure for nonlinear
  dynamics, which would replace the sampled (heuristic) enclosure on the sparse path.
