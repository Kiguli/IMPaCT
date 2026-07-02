# Multi-Objective Verification

Weighted-sum multi-objective robust reachability and Pareto frontier tracing on
interval MDPs.

## What is computed

Given k target label sets `T_1, ..., T_k` (all made absorbing, so each objective is
"the first target hit is `T_i`") and a weight vector `w`, `multi` maximises

```text
w · ( P(reach T_1), ..., P(reach T_k) )
```

over controller policies, with nature adversarial (`--bound pess`, default) or
cooperative (`--bound opt`). The output is the weighted optimum at the initial state
**and** the per-objective vector `P(reach T_i)` achieved by the weight-optimal policy —
one achievable point on the (robust) Pareto frontier. With no `--weights` argument and
exactly two objectives, the weight simplex is swept
(`λ ∈ {0, 0.25, 0.5, 0.75, 1}`) to trace the frontier.

## Algorithm and literature

The achievable set of a multi-objective MDP is convex, so its Pareto frontier is traced
by the weighted-sum optima over the weight simplex:

- Etessami, Kwiatkowska, Vardi, Yannakakis, *Multi-Objective Model Checking of Markov
  Decision Processes*, TACAS 2007 (convexity; memoryless-randomised strategies
  suffice; ε-approximable Pareto curve).
- Forejt, Kwiatkowska, Norman, Parker, Qu, *Quantitative Multi-Objective Verification
  for Probabilistic Systems*, TACAS 2011 (the PRISM implementation this parallels).

Implementation (`solve::multiReach`, `src/solve.{h,cpp}`): the robust VI is the
ordinary O-maximization backup with the value promoted to a vector and scalarised
(`vv[i] = Σ_j w_j V_j[to]`); the policy recorded at the weighted optimum is then
evaluated per objective.

## CLI examples (validated)

The `multi_demo` point MDP offers action A with `(P(F t1), P(F t2)) = (0.9, 0.1)` and
action B with `(0.2, 0.8)`:

```bash
# Sweep the weight simplex (2 objectives) — traces the Pareto frontier
tools/imdp_solve benchmarks/crosstool/models/multi_demo.imdp multi t1,t2
# pareto  bound=pess  state=0  weights=0,1        objectives=0.2,0.8  weighted=0.8
# pareto  bound=pess  state=0  weights=0.25,0.75  objectives=0.2,0.8  weighted=0.65
# pareto  bound=pess  state=0  weights=0.5,0.5    objectives=0.9,0.1  weighted=0.5
# pareto  bound=pess  state=0  weights=0.75,0.25  objectives=0.9,0.1  weighted=0.7
# pareto  bound=pess  state=0  weights=1,0        objectives=0.9,0.1  weighted=0.9

# One point for an explicit weight vector (required for k > 2 objectives)
tools/imdp_solve benchmarks/crosstool/models/multi_demo.imdp multi t1,t2 --weights 0.5,0.5
# pareto  bound=pess  state=0  weights=0.5,0.5  objectives=0.9,0.1  weighted=0.5
```

The sweep recovers exactly the two Pareto vertices `(0.9, 0.1)` and `(0.2, 0.8)`
(`multi_demo.ref.json`).

## Peer tools and validation

- **Storm**: `multi(Pmax=?[F t1], Pmax=?[F t2])` reports "2 Pareto optimal points" on
  the same model, matching IMPaCT's vertices; the corners match Storm's
  single-objective `Pmax=?[F t1] = 0.9` and `Pmax=?[F t2] = 0.8`.
- **PRISM**: supports `multi(...)` Pareto queries — non-interval models only.
- **IntervalMDP.jl**: no multi-objective support.

Multi-objective on **interval** MDPs is IMPaCT-only (`TOOL_COMPARISONS.md` §1b,
`FEATURE_PARITY.md`); peers were compared on the degenerate (point) model.

## Robust-Pareto semantics

For interval models the reported point is what the weight-optimal policy **guarantees
robustly per objective**: nature is re-resolved adversarially for each objective
evaluation. This is a sound achievable point; the fully convex robust-Pareto set under
a *single* adversarial nature across all objectives is a documented refinement
(`ROADMAP_UPGRADES.md` §3, `multi_demo.ref.json`).

## Limitations

- Objectives are **reachability probabilities only** — reward objectives are a planned
  refinement (`ROADMAP_UPGRADES.md` §3).
- The automatic sweep works for exactly 2 objectives; for k > 2 supply `--weights`
  vectors yourself.
- Weighted-sum scalarisation enumerates the *vertices* of the Pareto frontier;
  points on non-vertex faces are convex combinations of the reported vertices
  (randomised policies), not printed explicitly.
