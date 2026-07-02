# Long-Run Average and Steady-State

Robust long-run average (mean-payoff) reward and steady-state probabilities on
interval MDPs — the interval/robust versions are unique to IMPaCT.

## What is computed

- **`lra`** — the robust long-run average reward
  `lim inf (1/T) Σ_{t<T} r(s_t)`, controller maximising, nature resolving the
  intervals adversarially (pess) or cooperatively (opt) at every step.
- **`ss LABEL`** — the steady-state (long-run fraction of time) probability of the
  states labelled `LABEL`. IMPaCT computes it as the **long-run average of the
  indicator reward** `1_LABEL`, the classical identity (Baier & Katoen, *Principles of
  Model Checking*, Ch. 10; Puterman 1994) — this is PRISM's `S` operator, generalised
  to robust interval MDPs.

## Algorithm and literature

Two-phase value iteration (`solve::longRunAverage`, `src/solve.{h,cpp}`):

1. **MEC gain**: the robust average reward (gain) of each maximal end component is
   computed by *relative value iteration* with Puterman's aperiodicity transform
   `L_τ h = (1-τ)h + τ(r + opt_a opt_p Σ p·h)`, restricted to each MEC's staying
   actions (support ⊆ MEC — exactly why nature cannot escape it). The interval
   ambiguity set is resolved by O-maximization.
2. **Cash-out**: a max-reachability VI funnels every state to the best
   robustly-reachable MEC gain.

Literature (verified in `ROADMAP_UPGRADES.md` §2):

- Ashok, Chatterjee, Daca, Křetínský, Meggendorfer, *Value Iteration for Long-Run
  Average Reward in MDPs*, CAV 2017 (arXiv:1705.02326) — the two-phase VI.
- Chatterjee, Goharshady, Karrabi, Novotný, Žikelić, *Solving Long-run Average Reward
  Robust MDPs via Stochastic Games*, IJCAI 2024 (arXiv:2312.13912) — the
  robust/interval setting (that work uses policy iteration/games; IMPaCT takes the VI
  route).
- Puterman, *Markov Decision Processes*, 1994, Ch. 8 (average-reward optimality,
  relative VI, aperiodicity transform); Iyengar 2005 (robust inner solve).

## CLI examples (validated)

```bash
# Point 2-cycle, r(0)=1, r(1)=3: mean payoff (1+3)/2 = 2 (== Storm R{"r"}max=?[LRA])
tools/imdp_solve benchmarks/crosstool/models/lra_cycle.imdp lra - --bound both
# result  prop=lra  bound=pess  state=0  lower=2  upper=2  iters=0

# Interval MEC, r(0)=4, r(1)=0, stay [0.5,1]/switch [0,0.5]: nature traps the play in
# state 1 (robust gain 0) or keeps it in state 0 (optimistic gain 4). IMPaCT-only.
tools/imdp_solve benchmarks/crosstool/models/lra_interval.imdp lra - --bound both
# result  prop=lra  bound=pess  state=0  lower=2.265286626e-06  upper=2.265286626e-06  iters=0
# result  prop=lra  bound=opt   state=0  lower=4                upper=4                iters=0

# Steady-state: DTMC with stationary distribution (4/7, 3/7); S(g={1}) = 3/7
tools/imdp_solve benchmarks/crosstool/models/ss_dtmc.imdp ss g
# result  prop=ss  bound=pess  state=0  lower=0.4285698197  upper=0.4285698197  iters=0
```

The `lra` property takes no target label — pass `-` as the LABEL placeholder. Values
match the analytic references 2, 0 / 4, and 3/7 = 0.4285714 within the VI tolerance
(`lra_cycle.ref.json`, `lra_interval.ref.json`, `ss_dtmc.ref.json`).

## Peer tools and validation

- **Storm** computes point-model LRA (`R{"r"}max=? [LRA]` = 2 on `lra_cycle`, matching
  IMPaCT) and steady-state (`S=? [g]` = 0.4285714286 on `ss_dtmc`, matching IMPaCT),
  but reports *"LRA with intervals not implemented"* for interval models.
- **PRISM**'s `S` operator is DTMC/CTMC-only (no MDP steady-state reward, no interval
  LRA).
- **IntervalMDP.jl** has no LRA / steady-state objective.

Robust interval LRA and robust steady-state were therefore validated against the
analytic hand values (`lra_interval` 0 / 4) — **an IMPaCT-only capability**
(`FEATURE_PARITY.md`, `TOOL_COMPARISONS.md` §1b).

## Limitations

- Convergence of the gain phase is controlled by the same `--eps`; the robust
  pessimistic value on `lra_interval` converges to ~2e-6 rather than exactly 0
  (residual of the relative VI) — tighten `--eps` for sharper values.
- The steady-state of a *multichain* interval MDP is the reachability-weighted best
  MEC gain (phase 2), i.e. the controller-optimal long-run frequency; per-initial
  distribution steady-state vectors are not exposed.
- An exact policy-iteration LRA (as in Chatterjee et al., IJCAI 2024) is a documented
  refinement (`ROADMAP_UPGRADES.md`).
