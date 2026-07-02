# IMPaCT vs IntervalMDP.jl

[IntervalMDP.jl](https://github.com/Zinoex/IntervalMDP.jl) (compared at version 0.6.0 on
Julia 1.10) is a Julia library dedicated to interval MDPs: robust value iteration built
on O-maximization — the same inner solve IMPaCT uses — with CUDA acceleration and
factored (orthogonal and mixture) IMDP representations for scalability. Of the three
peers it is semantically closest to IMPaCT: both solve `Pessimistic/Optimistic ×
Maximize` queries, which map one-to-one to IMPaCT's `--bound pess/opt`.

## Feature overlap

| IntervalMDP.jl has (IMPaCT does not) | Both | IMPaCT has (IntervalMDP.jl does not) |
|---|---|---|
| CUDA-native value iteration | Robust reachability / safety, bounded and unbounded | Abstraction of continuous nonlinear stochastic dynamics → IMDP (6 noise models) |
| Expected exit-time queries | Discounted expected reward | Robust ω-regular (Büchi/persistence/patrol) and arbitrary det.-Büchi LTL (IntervalMDP.jl: DFA/co-safe reachability only) |
| netCDF / bmdp-tool input formats | Orthogonal and mixture IMDPs (IMPaCT `.odimdp`, added in v2.0) | Reachability rewards, long-run average, steady-state, multi-objective (all robust) |
| DFA-product (co-safe) specifications | GPU acceleration (CUDA vs any-vendor SYCL) | CTMC + CSL, interval CTMC, POMDP/PTA/TA, PRISM-subset input, statistical MC |
| — | Exact / rational arithmetic; strategy synthesis | Certified OVI bracket; interval iteration (sound two-sided bound) |

## Same results on IntervalMDP.jl's own example

`multiObj_robotIMDP` is the one interval-MDP example model shipped by a peer tool: a
207-state robot IMDP (828 state-action rows, 2784 interval transitions, init state 0,
target state 206), converted to `.imdp` with `peers/tra_to_imdp.py`. Query: robust
reachability, `Pessimistic, Maximize` = IMPaCT `reach --bound pess`.

| solver | init value | iters | note |
|---|---|---|---|
| IntervalMDP.jl (default threshold 1e-6) | 0.8946617847 | 83 | stops 1.2e-6 below the fixpoint |
| IntervalMDP.jl (threshold 1e-9) | 0.8946629814 | 99 | |
| IntervalMDP.jl (threshold 1e-12) | **0.8946629826** | 128 | converged |
| IMPaCT `imdp_solve` OVI (eps 1e-6) | **0.8946629826** | 129 | certified `[L, L+eps]` |
| IMPaCT interval iteration (eps 1e-9) | 0.8946629826 | 42177 | sound bracket [0.8946629826, 0.8946629838] |

The values agree to ~1e-10 once thresholds match. The apparent 1.2e-6 gap at defaults is
purely IntervalMDP.jl's 1e-6 residual threshold terminating short of the fixpoint on
this slowly-mixing chain — a methodological note, not a disagreement. IMPaCT's OVI
returns the converged value *and* a certified upper bound already at eps 1e-6. Both
tools give the optimistic value 1.0 (IMPaCT/Storm 0.9999979999, IntervalMDP.jl/PRISM
0.9999969998).

## Factored IMDPs: orthogonal and mixture

IMPaCT v2.0 adopts IntervalMDP.jl's flagship scalability representations, validated
against it as the oracle (`peers/odimdp_oracle.jl`, `peers/mixture_oracle.jl`):

- **Orthogonal IMDP** (`od_demo.odimdp`): IMPaCT matches
  `OrthogonalIntervalMarkovDecisionProcess` to 10 digits in both senses (state 0 pess
  0.3714285709 / opt 0.8536585360; state 1 pess 0.7285714278 / opt 0.9999999995).
- **Mixture IMDP** (`mix_demo.odimdp`): matches
  `MixtureIntervalMarkovDecisionProcess` to 10 digits both senses (pess 0.4756756752 /
  opt 0.9619450314).
- **Reduction-order semantics (ISSUE-0022).** The robust value of a factored IMDP
  depends on the order in which nature resolves the per-dimension marginals. IMPaCT's
  first implementation reduced the highest dimension innermost and got pess 0.3478 vs
  IntervalMDP.jl's 0.3714 (optimistic values already matched). Reading
  IntervalMDP.jl's `state_action_bellman` identified its convention — dimension 1
  innermost — which IMPaCT now follows; agreement is only meaningful once this
  convention is pinned.

**Discounted reward** (γ=0.9, `reward_chain`): IMPaCT pess 2.8 / opt 4.272726893 vs
IntervalMDP.jl 2.8 / 4.272726893272438 — IntervalMDP.jl is the oracle here since its
`InfiniteTimeReward` requires a discount < 1 and Storm's interval rewards ignore the
resolution mode.

## Differences and caveats

- **Specification scope.** IntervalMDP.jl covers reachability, safety, DFA/co-safe
  reachability, discounted reward and exit time; no Büchi/persistence, LRA,
  multi-objective, CTMC or statistical MC. IMPaCT's robust ω-regular, LTL, LRA and
  multi-objective queries have no IntervalMDP.jl counterpart.
- **Optimistic-safety convention.** On BA's `[0,hi]` leak edges IntervalMDP.jl's
  cooperative resolution reports opt-safety ≈ 0.997 where the PRISM/Storm `1−Pmin`
  encodings give 0 (its BA robust safety, 1.9e-4, still effectively agrees with the
  others' 0). The robust value is unambiguous; compare that.
- **Thresholds.** Match convergence thresholds before comparing digits (see the robot
  example above).

## Performance

The two tools' solve times are close — expected, since both run robust VI over
O-maximization. From the 2026-06-27 cross-tool run (seconds):

| benchmark | IMPaCT robust solve (II / VI) | IntervalMDP.jl VI |
|---|---|---|
| AS | 4.3 (finite) | 0.40 |
| BA | **0.27** (finite) | 1.14 |
| VP | 29.1 / **0.59** | 0.61 |
| IC_reach | 0.06 (finite) | 0.38 |
| IC_safe | 0.47 (finite) | 0.42 |
| PD_p1 | 1.29 / 1.34 | 1.33 |
| PD_p3 | ∞ (II) / **1.36** | 1.40 |

VP 0.59 s vs 0.61 s and PD_p1 1.34 s vs 1.33 s are effectively ties; IntervalMDP.jl is
clearly faster on AS (0.40 s vs the dense finite path's 4.3 s — IMPaCT's sparse path
later cut this to 0.07 s, see `benchmarks/RESULTS.md`). In the sweep TSVs,
IntervalMDP.jl's first query per model carries ~0.4 s of Julia runner start-up (its
second query on the same 4-state model returns in ~0.001 s), so small-model timings
favour the C++ tools. On the 207-state robot, IntervalMDP.jl solved in 1.2 ms vs
IMPaCT's ~10 ms. The sparse-memory story is shared: IMPaCT's sparse path is "the
IntervalMDP.jl-style memory win" — at 125k states the sparse model is ~255 MB where a
dense matrix would be ~250 GB, and the 6-D LM case (dense-infeasible at ~96 GB/matrix)
builds and solves at 196 MB.

## Reproduce

```bash
# run IntervalMDP.jl on any exported .imdp (inside the impact_work container)
julia benchmarks/crosstool/peers/intervalmdp_runner.jl MODEL.imdp

# native robot model runs (default + tight threshold), factored oracles
julia benchmarks/crosstool/peers/imdp_native_robot.jl
julia benchmarks/crosstool/peers/imdp_native_robot_tight.jl
julia benchmarks/crosstool/peers/odimdp_oracle.jl

# full four-tool sweep
python3 benchmarks/crosstool/peers/sweep_compare.py all
```

Raw results: `benchmarks/crosstool/sweep_all.tsv`, `sweep_arch.tsv`; per-model measured
values: `benchmarks/crosstool/models/*.ref.json` (see `od_demo.ref.json` for the
factored-semantics note).
