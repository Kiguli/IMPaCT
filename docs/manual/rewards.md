# Expected Rewards

Robust expected total reward on interval MDPs: **reachability reward** (accumulate
until a target is reached) and **discounted reward**.

## What is computed

State rewards are given by `reward <s> <v>` lines in the `.imdp` file (missing states
default to 0). The controller maximises the expected accumulated reward; nature
resolves the transition intervals adversarially (pess) or cooperatively (opt) on the
*continuation value* via O-maximization.

- **Reachability reward** (`reward LABEL`, discount γ = 1): expected total state reward
  accumulated until the target set is first reached. The target is absorbing and
  reward stops there. Bellman recursion (`src/solve.h`):

  ```text
  V(s) = reward[s] + max_a  opt_{p in [lo,hi]} Σ_s' p(s') V(s'),    V(target) = 0
  ```

  **+∞ semantics:** if the target is not reached almost surely and rewards are
  positive, the value diverges; IMPaCT reports `HUGE_VAL` (prints `inf`), matching
  PRISM/Storm's `Infinity` convention for stochastic-shortest-path divergence.

- **Discounted reward** (`reward LABEL --discount g`, γ ∈ (0,1)):
  `V(s) = reward[s] + γ · max_a opt_p Σ p V`. This is a contraction, so it always
  converges and needs no target (the label argument is still required syntactically).
  This is the objective IntervalMDP.jl natively supports.

**Expected time / expected steps to reach** is the special case of reachability reward
with unit state rewards (`reward s 1` for every non-target state), as noted in
`FEATURE_PARITY.md`.

## Algorithm and literature

Robust dynamic programming over the interval ambiguity set, with the inner
minimisation/maximisation solved in closed form by O-maximization:

- Iyengar, *Robust Dynamic Programming*, Mathematics of Operations Research, 2005.
- Nilim & El Ghaoui, *Robust control of MDPs with uncertain transition matrices*,
  Operations Research, 2005.
- Givan, Leach, Dean, *Bounded-parameter MDPs*, AIJ 2000 (the interval inner solve).
- Puterman, *Markov Decision Processes*, 1994 (SSP finiteness for the γ = 1 case).
- Convergence certification: interval iteration / OVI (Baier et al. CAV 2017;
  Hartmanns & Kaminski CAV 2020), as for reachability.

Implementation: `solve::expReachReward`, `solve::expDiscountedReward` and the
`maxReachReward{Pessimistic,Optimistic}` wrappers in `src/solve.{h,cpp}`.

## CLI examples (validated)

The `reward_chain` model has r(s0)=1, r(s1)=2; from s1 nature chooses to move to the
target with probability in [0.5, 1.0] or stay with [0.0, 0.5]:

```bash
# Reachability reward: pess (nature leaves s1 asap) V0 = 3; opt (nature stays) V0 = 5
tools/imdp_solve benchmarks/crosstool/models/reward_chain.imdp reward target --bound both
# result  prop=reward  bound=pess  state=0  lower=3            upper=3            iters=3
# result  prop=reward  bound=opt   state=0  lower=4.999999046  upper=4.999999046  iters=23

# Discounted reward, gamma = 0.9: pess 2.8, opt 2/(1-0.45)*0.9+1 = 4.2727...
tools/imdp_solve benchmarks/crosstool/models/reward_chain.imdp reward target --discount 0.9 --bound both
# result  prop=reward  bound=pess  state=0  lower=2.8          upper=2.8          iters=3
# result  prop=reward  bound=opt   state=0  lower=4.272726893  upper=4.272726893  iters=21
```

Both match the analytic hand values in `reward_chain.ref.json` (3 / 5 and
2.8 / 4.272727).

## Peer tools and validation

- **PRISM 4.8.1**: interval reachability reward via `Rmaxmin` / `Rmaxmax` — matches
  IMPaCT (3.000000002 / 4.999996 on `reward_chain`).
- **IntervalMDP.jl 0.6**: discounted reward only (`InfiniteTimeReward` requires
  discount < 1) — matches IMPaCT to 9+ digits (2.8 / 4.272726893).
- **Storm 1.13.0**: its interval reward **ignores** `--uncertainty-resolution` and
  returns only the nature-max value (4.999996 = the optimistic value); `Rmin` on
  intervals is unsupported. Documented in `reward_chain.ref.json` and
  `TOOL_COMPARISONS.md` §1b.

IMPaCT is the only tool providing *both* robust senses for *both* reward objectives on
interval MDPs.

## Limitations

- Rewards are **state-based** only (no action or transition rewards in the `.imdp`
  format).
- Reward objectives are not yet available in the multi-objective engine
  ({doc}`multiobjective`) — a documented refinement in `ROADMAP_UPGRADES.md` §3.
- Divergent (+∞) reachability-reward cases are reported but not distinguished from
  very large finite values until convergence is abandoned; use the discounted variant
  when the target is not reached almost surely.
