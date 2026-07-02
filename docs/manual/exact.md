# Exact Rational Solving

`reach --exact`: robust reachability computed **exactly over the rationals**, with a
machine-checked fixpoint certificate — including on *interval* models, where the peer
tools' exact modes do not work at all.

## What is computed

The exact robust value `P(F target)` at one queried state, as a fraction (e.g. `2/5`),
for a point or interval `.imdp` model (decimal inputs are parsed exactly as
fractions). `--bound pess` is adversarial nature, `opt` cooperative. The result line
reports the fraction, a decimal approximation, the policy-iteration round count, and
whether the **certificate** passed:

```text
result  prop=reach-exact  bound=pess  state=0  exact=2/5  approx=0.4  certified=yes  iters=2
```

Rationals are `long long` numerator/denominator with `__int128` intermediates and gcd
normalisation; **overflow throws** rather than silently losing exactness.

## Algorithm and literature

Policy iteration in the Hoffman-Karp style over the controller policy *and* nature's
O-maximization vertex — both spaces are finite because the O-max vertex depends only on
the *ordering* of the value vector (`src/exact.{h,cpp}`):

1. **Qualitative preprocessing**: compute the exact robust prob-0 set P0 (the largest
   target-free set nature can confine the play to) and pin it to 0.
2. **Exact policy evaluation**: rational Gaussian elimination on the induced point
   chain (chain-level unreachability pinned to 0).
3. **Improvement**: better controller action / nature vertex by exact O-maximization
   (which is division-free — sort plus additive mass assignment — so nature's vertex
   is computed exactly); repeat until stable.
4. **Certificate**: verify the result is an exact fixpoint of the robust Bellman
   operator. Fixpoints vanishing on P0 are unique for reachability, so the certified
   fixpoint *is* the robust value.

Literature (from `src/exact.h` and `FEATURE_PARITY.md`):

- Iyengar, *Robust Dynamic Programming*, Math. OR 2005, §3.3 (robust policy
  iteration).
- Puterman, *Markov Decision Processes*, 1994, Ch. 6-7 (classical policy iteration).
- Givan, Leach, Dean, AIJ 2000 (division-free O-maximization).
- Uniqueness of the pinned fixpoint: Baier & Katoen, *Principles of Model Checking*,
  Thm 10.100ff; Haddad & Monmege, TCS 2018 (interval/MDP setting).

## CLI examples (validated)

```bash
# Point chain: exact 1/4 (== PRISM -exact == Storm --exact)
tools/imdp_solve benchmarks/crosstool/models/chain_point.imdp reach target --exact --bound both
# result  prop=reach-exact  bound=pess  state=0  exact=1/4  approx=0.25  certified=yes  iters=2

# INTERVAL models — exact robust values, certified (peers cannot do this at all)
tools/imdp_solve benchmarks/crosstool/models/choice_interval.imdp reach target --exact --bound both
# result  prop=reach-exact  bound=pess  state=0  exact=2/5  approx=0.4  certified=yes  iters=2
# result  prop=reach-exact  bound=opt   state=0  exact=1/2  approx=0.5  certified=yes  iters=2

tools/imdp_solve benchmarks/crosstool/models/fork_interval.imdp reach target --exact --bound both
# result  prop=reach-exact  bound=pess  state=0  exact=2/5  approx=0.4  certified=yes  iters=3
# result  prop=reach-exact  bound=opt   state=0  exact=3/5  approx=0.6  certified=yes  iters=2
```

## Point vs interval, and the peers

PRISM's `-exact` and Storm's `--exact` support **point models only** — verified
empirically: both error on interval inputs (`FEATURE_PARITY.md` §B). On the point
chain all three agree on exactly `1/4`. The certified interval values `2/5`, `1/2`,
`3/5` above make **exact robust interval solving unique to IMPaCT**.

## Infeasible-row strictness (ISSUE-0023)

Exact arithmetic makes model-validity checking unavoidable. Rows whose upper bounds
sum to slightly below 1 (e.g. truncated decimals like `0.999997` in the
IntervalMDP.jl-shipped robot model) admit **no** probability distribution, and exact
mode aborts:

```bash
tools/imdp_solve benchmarks/crosstool/models/multiObj_robotIMDP.imdp reach reach --exact
# exact error: exact: infeasible interval row (sum hi < 1)
```

Floating-point solvers (IMPaCT's own O-max, and evidently PRISM / Storm /
IntervalMDP.jl) silently tolerate such rows, effectively giving the missing ~3e-6 mass
to the highest-value successor — so cross-tool comparisons on such models compare
*silent-repair conventions*. The strict abort is deliberate (the same strictness that
certifies fixpoints certifies input sanity); a possible future `--exact-repair` option
would scale the upper bounds to feasibility and report the perturbation (ISSUE-0023).

## Limitations

- `--exact` supports the `reach` property only (safety = 1 − reach to avoid can be
  obtained by complementing by hand; other properties are floating-point only).
- Fixed-width rationals: models whose exact solution needs very large
  numerators/denominators throw on overflow rather than approximating (switch to the
  floating-point solver in that case).
- Values are reported for a single queried state (`--state`, default init), not the
  whole vector.
