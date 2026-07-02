# BA — Building Automation (4D)

The BA benchmark is a four-dimensional building-automation (temperature-control)
system with affine dynamics, one heating input and additive Gaussian noise. The
specification is finite-horizon safety: remain inside the temperature envelope for 6
steps. Source: `examples/ARCH-COMP/2025/BA/BA_4d.cpp`.

## System

- **State** (4D): two zone temperatures and two heater states, gridded over
  `[19,21] x [19,21] x [30,36] x [30,36]` with steps `{0.5, 0.5, 1, 1}`.
- **Input** (1D): supply temperature over `[17,20]` with step `1`.
- **Noise**: additive Gaussian with per-dimension standard deviations
  `sqrt(0.0774), sqrt(0.0774), sqrt(0.3872), sqrt(0.3098)`.
- **Specification**: finite-horizon safety, H = 6 — stay within the state grid
  (leaving the grid is the avoid event).

The dynamics are affine, `x' = A x + B u + Q`:

```cpp
auto dynamics = [](const vec& x, const vec& u) -> vec {
    vec xx(dim_x);
    const mat A = {{0.6682, 0.0, 0.02632, 0.0},
                   {0.0, 0.6830, 0.0, 0.02096},
                   {1.0005, 0.0, -0.000499, 0.0},
                   {0.0, 0.8004, 0.0, 0.1996}};
    const vec B = {0.1320, 0.1402, 0.0, 0.0};
    const vec Q = {3.4378, 2.9272, 13.0207, 10.4166};
    xx = A*x + B*u + Q;
    return xx;
};
```

## Code walkthrough

Set up the spaces, dynamics and noise. A safety specification needs no target set —
only the probability intervals of *leaving* the safe grid:

```cpp
IMDP mdp(dim_x, dim_u, dim_w);
mdp.setStateSpace(ss_lb, ss_ub, ss_eta);
mdp.setInputSpace(is_lb, is_ub, is_eta);
mdp.setDynamics(dynamics);
mdp.setNoise(NoiseType::NORMAL);
mdp.setStdDev(sigma);
```

Abstract the avoid vectors (probability of exiting the grid) and the cell-to-cell
transition matrices:

```cpp
mdp.minAvoidTransitionVector();
mdp.maxAvoidTransitionVector();
mdp.transitionMatrixBounds();
```

Export the IMDP and synthesize the robust finite-horizon safety controller:

```cpp
mdp.exportIMDP("BA.imdp");
mdp.finiteHorizonSafeControllerSorted(true, 6);   // true = pessimistic
mdp.saveController();
```

Build and run with `make && ./BA_4d` in `examples/ARCH-COMP/2025/BA`.

## Sparse-abstraction route

BA is affine, so it runs through `benchmarks/sparse_arch.cpp` (`only=BA`) with an
exact per-dimension mean range. With the grid aligned to the dense point-grid
convention the sparse cell count matches the dense abstraction exactly (1225 cells).

BA is the canonical *coupled-dynamics* case: because `A` is non-diagonal, the sparse
box bound (product of per-dimension intervals) is a **sound over-approximation** —
the per-dimension worst cases are not jointly attainable — while the dense joint
NLopt optimisation is tighter. A one-step brute-force check on a corner cell gives a
true value of 0.2610 versus the sparse bound 0.2664 (+0.0054): both sound, sparse
slightly conservative. Grid alignment moved the sparse interior safety value from
0.107 to 0.348, against the dense (NLopt) 0.471.

## Results

Dense SYCL path (AdaptiveCpp/OMP, 56 GB host):

| cells | abstraction (s) | synthesis (s) |
|---|---|---|
| 1225 | 7.6 | 0.27 |

Sparse path (logged run): 1.09M nonzeros, 26.2 MB (dense 0.02 GB), build + solve
0.25 + 0.22 s.

## Cross-tool validation

The exported `BA.imdp` was solved identically by IntervalMDP.jl, PRISM and Storm.
Under the native finite horizon the initial-state robust safety value is **0.4533 in
all three tools** (IMPaCT, IntervalMDP.jl, Storm), and IMPaCT's per-cell robust
values match IntervalMDP.jl to **Δ ≤ 4.3e-7**. On the infinite-horizon safety
reduction the robust value is 0, matching PRISM and Storm (verified at eps 1e-4);
IntervalMDP.jl reports 1.9e-4 at its defaults. Peer solve times: 4.9 / 0.3 / 0.7 s
(IntervalMDP.jl / PRISM / Storm).

## Notes

- IMPaCT's *optimistic* infinite-horizon safety OVI did not converge within 180 s on
  this model — a large safe end component (ISSUE-0010). The robust (pessimistic)
  value, which is the headline guarantee, is unambiguous and agrees across tools.
- Optimistic safety is convention-dependent across tools (how the cooperative sense
  resolves `[0,hi]` leak edges); only the robust value should be compared directly.
- BA's noise window spans roughly 7 cells against a 4-cell domain, so low
  finite-horizon safety values are expected on this coarse grid.
