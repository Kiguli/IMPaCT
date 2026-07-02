# Orthogonal and Mixture IMDPs

Factored interval MDPs whose transition ambiguity decomposes per state dimension —
IntervalMDP.jl's scalability representation, now solved natively by IMPaCT.

## Model class

An **orthogonally-decoupled IMDP (odIMDP)** has a product state space
`S = S_1 × ... × S_n`; the ambiguity set of each (state, action) factors as a product
of **per-dimension marginal interval distributions**. Storing n marginals of size
`|S_d|` instead of one joint distribution of size `Π|S_d|` is the memory/compute
scalability feature — and matches the per-dimension transition bounds IMPaCT's own
sparse abstraction produces for decoupled dynamics (`src/abstraction.cpp`).

A **mixture IMDP** generalises this: one action is a *mixture* of orthogonal
components, where nature picks both a weight vector within interval weights and each
component's marginals within theirs.

## What is computed

`reach` (the only property supported on `.odimdp` inputs): robust max-reachability with
the controller maximising and nature choosing factored distributions adversarially
(pess) or cooperatively (opt).

**Reduction-order convention (ISSUE-0022).** The factored ambiguity set is
*reduction-order sensitive*: nature's inner marginal choices may condition on the
destination coordinates already realised by the outer dimensions (per-prefix
rectangular semantics), so different reduction orders define different nature classes
and can give different pessimistic values. IMPaCT reduces **dimension 1 innermost**
(then 2, ..., n outermost) — exactly IntervalMDP.jl's convention (its
`state_action_bellman` reduces dimension 1 first per combination of the remaining
coordinates). When citing or comparing factored values, always state the convention.

## Algorithm and literature

Robust Bellman backup by **recursive per-dimension O-maximization**
(`src/odimdp.{h,cpp}`): reduce the value hyper-rectangle one dimension at a time,
optimising the dim-d marginal per prefix of the outer destination coordinates. For
mixtures: per-component factored O-max, then O-max over the interval mixture weights.

- Mathiesen, Haesaert, Laurenti, *Scalable control synthesis for stochastic systems
  via structural IMDP abstractions*, arXiv:2411.11803 (odIMDP and mixture ambiguity
  sets).
- Givan, Leach, Dean, *Bounded-parameter MDPs*, AIJ 2000 (the per-marginal inner
  solve).

## The `.odimdp` format

```text
odimdp
dims n1 n2 ...                        # per-dimension sizes; states linearised, dim 0 fastest
init s
label NAME s ...
otran  <s> <a> <d> to:lo:hi ...       # dim-d marginal of action a (single component)
mtran  <s> <a> <k> <d> to:lo:hi ...   # dim-d marginal of mixture component k
mweight <s> <a> k:lo:hi ...           # interval mixture weights over components
```

See `benchmarks/crosstool/models/od_demo.odimdp` and `mix_demo.odimdp` for complete
commented examples.

## CLI examples (validated)

```bash
# Orthogonal IMDP, 2x2 product space, interval marginals per dimension
tools/imdp_solve benchmarks/crosstool/models/od_demo.odimdp reach target --bound both
# result  prop=reach  bound=pess  state=0  lower=0.3714280537  upper=0.3714280537  iters=17
# result  prop=reach  bound=opt   state=0  lower=0.8536579107  upper=0.8536579107  iters=20

# Mixture-of-orthogonal IMDP with interval mixture weights
tools/imdp_solve benchmarks/crosstool/models/mix_demo.odimdp reach target --bound both
# result  prop=reach  bound=pess  state=0  lower=0.47567519    upper=0.47567519    iters=17
# result  prop=reach  bound=opt   state=0  lower=0.9619447054  upper=0.9619447054  iters=19
```

(Default `--eps 1e-6` VI; the converged references are 0.3714285709 / 0.8536585360 and
0.4756756752 / 0.9619450314.)

## Peer tools and validation

The oracle is **IntervalMDP.jl** on the *identical* models:
`OrthogonalIntervalMarkovDecisionProcess` (`benchmarks/crosstool/peers/odimdp_oracle.jl`)
and `MixtureIntervalMarkovDecisionProcess` (`peers/mixture_oracle.jl`), with
`InfiniteTimeReachability` and `Pessimistic/Optimistic × Maximize`. At tight
thresholds **IMPaCT matches IntervalMDP.jl to 10 digits in both senses** on both
models (`od_demo.ref.json`; `FEATURE_PARITY.md`). PRISM and Storm have no factored
IMDP representation.

The first implementation reduced the *highest* dimension innermost and disagreed with
IntervalMDP.jl in the pessimistic sense (0.3478 vs 0.3714 on `od_demo`) while agreeing
optimistically — the discriminating experiment that pinned the convention
(ISSUE-0022).

## Limitations

- Only `reach` is dispatched for `.odimdp` inputs (no safety/ω-regular/reward on the
  factored representation yet); `--bound`, `--eps`, `--state` apply as usual.
- The `.odimdp` solver runs plain VI with residual stopping (single value reported as
  both `lower` and `upper`), not the certified OVI bracket of the joint solver.
- Cross-tool factored values are only comparable under a pinned reduction order
  (ISSUE-0022).
