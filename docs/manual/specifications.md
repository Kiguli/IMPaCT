# Probabilistic Specifications

Reachability, safety, reach-avoid, step-bounded reachability, and the PCTL path
operators on interval MDPs (IMDPs).

## What is computed

For an IMDP, the controller maximises the probability of the path property while
*nature* picks, at every step, a transition distribution within the per-successor
probability intervals:

- **pessimistic** (`--bound pess`, default): nature is adversarial. The result is the
  *robust* value — a guarantee that holds for every distribution in the intervals.
- **optimistic** (`--bound opt`): nature is cooperative — the best-case value.

All unbounded properties return a sound sandwich `lower <= V*(s) <= upper` with gap
`<= 2*eps`; finite-horizon (`--horizon k`) properties are computed by exact dynamic
programming (no gap). Properties:

| Property | Semantics |
|---|---|
| `reach L` | max P(F L) — eventually reach a state labelled `L` |
| `safety L` | max P(G !L) = 1 − min P(F L) — never reach the avoid set |
| `until a,b` | max P(a U b) — reach `b` while remaining in `a` (**reach-avoid**: let `a` be the complement of the obstacle set) |
| `reach L --horizon k` | P[F<=k L] = true U<=k L (step-bounded) |
| `until a,b --horizon k` | a U<=k b (step-bounded until) |
| `next L` | P(X L) — one robust Bellman step |

## Algorithm and literature

The inner robust Bellman solve `opt_{p in [lo,hi]} Σ p(s')V(s')` is **O-maximization**
(closed-form sort-and-assign), from Givan, Leach, Dean, *Bounded-parameter Markov
decision processes*, Artificial Intelligence, 2000, and Lahijanian, Andersson, Belta,
IEEE Trans. Automatic Control, 2015 (`src/omaximization.h`, verified against
brute-force vertex enumeration).

The outer fixpoint (`src/solve.h`) is solved by one of two methods (see {doc}`engines`):

- **Optimistic value iteration** (default) — Hartmanns & Kaminski, CAV 2020: VI from
  below plus a verified inductive (pre-fixpoint) upper bound.
- **Interval iteration with end-component collapse** — Baier et al., CAV 2017;
  Haddad & Monmege, TCS 2018.

PCTL operators (`next`, `until`, bounded until; `src/pctl.h`) follow Hansson & Jonsson
(Formal Aspects of Computing, 1994); the robust interval-MDP reading follows Puggelli,
Li, Sangiovanni-Vincentelli, Seshia, CAV 2013, and Baier & Katoen, *Principles of Model
Checking*, Ch. 10.

## CLI examples (validated)

```bash
# Point chain: P(F target) = 0.25 exactly (all four tools agree)
tools/imdp_solve benchmarks/crosstool/models/chain_point.imdp reach target --bound both
# result  prop=reach  bound=pess  state=0  lower=0.25  upper=0.250001  iters=3

# Safety: P(never reach avoid) = 0.72
tools/imdp_solve benchmarks/crosstool/models/safety_point.imdp safety avoid --bound pess
# result  prop=safety  bound=pess  state=0  lower=0.719999  upper=0.72  iters=3

# Interval MDP with a controller choice: robust 0.4 (pick action B), optimistic 0.5
tools/imdp_solve benchmarks/crosstool/models/choice_interval.imdp reach target --bound both
# result  prop=reach  bound=pess  state=0  lower=0.4  upper=0.400001  iters=2
# result  prop=reach  bound=opt   state=0  lower=0.5  upper=0.500001  iters=2

# Step-bounded reachability P[F<=3 goal] = 1 - 0.5^3 = 0.875 (exact DP, == Storm)
tools/imdp_solve benchmarks/crosstool/models/bounded_demo.imdp reach goal --horizon 3
# result  prop=reach  bound=pess  state=0  lower=0.875  upper=0.875  iters=3

# PCTL next: P(X target) from state 1 of the chain = 0.5
tools/imdp_solve benchmarks/crosstool/models/chain_point.imdp next target --state 1
# result  prop=next  bound=pess  state=1  lower=0.5  upper=0.5  iters=1
```

## Peer tools and validation

Robust reachability/safety is the one query all four tools support. On the shared
models (identical `.imdp` exported to each tool's native format) **IMPaCT ==
IntervalMDP.jl == PRISM == Storm**: `chain_point` 0.25, `safety_point` 0.72,
`choice_interval` 0.4/0.5, `fork_interval` 0.4/0.6, and the 207-state
`multiObj_robotIMDP` 0.894663 (agreement ~1e-6); see `TOOL_COMPARISONS.md` §1 and the
ARCH-COMP 2025 sweep (§2), where all four agree on every robust value. Step-bounded
`P[F<=k]` was validated against Storm `Pmax=?[F<=k "goal"]` for k = 1, 2, 3
(`bounded_demo.ref.json`). PRISM peers require the `Pmaxmin`/`Pmaxmax` query form and
reject interval edges with a 0 lower bound (see {doc}`formats`).

## Limitations

- Pessimistic interval iteration alone does not converge on *nature-confinable* end
  components (ISSUE-0003, resolved by the OVI default; see {doc}`engines`).
- OVI can be slow on large strongly-recurrent IMDPs, mainly in the optimistic sense
  (ISSUE-0010); `--method mec` is faster there.
- The optimistic sense is convention-dependent across tools for `[0,hi]` "leak" edges
  (ISSUE-0018, `TOOL_COMPARISONS.md` §4); the robust value is unambiguous and is the
  headline number.
- `imdp_solve` exposes controller-*maximising* queries (`Pmax`); `Pmin` flavours are
  future work (`src/pctl.h`).
