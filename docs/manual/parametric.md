# Parametric Region Verification

Verify a *parametric* Markov model over a whole parameter region in one solve, by
reducing region verification to robust interval-MDP reachability (Parameter Lifting).

## What is computed

Given a Markov model whose transition probabilities are expressions in a parameter `p`
and a region `p ∈ [lo, hi]`, the lifted interval MDP's robust `[pess, opt]`
reachability values **bound the parametric value over the entire region**:

```text
pess  <=  min_{p in [lo,hi]} P_p(F target)   and   max_{p in [lo,hi]} P_p(F target)  <=  opt
```

The bound is **exact** when each parameter occurs once per transition row, and a
**sound over-approximation** otherwise (each parameter *occurrence* is relaxed
independently — precisely the Parameter Lifting relaxation).

## Algorithm and literature

Parameter Lifting: relax parameter dependencies so each occurrence ranges over the
region independently; the result is an interval MDP, and robust reachability on it
bounds the parametric solution function over the region.

- Quatmann, Dehnert, Jansen, Junges, Katoen, *Parameter Synthesis for Markov Models:
  Faster Than Ever*, ATVA 2016 (DOI 10.1007/978-3-319-46520-3_4).

This is a natural fit: parametric region checking ≡ interval/robust solving, which is
IMPaCT's native engine (`FEATURE_PARITY.md` §B). The lifting itself is the standalone
script `benchmarks/crosstool/peers/param_lift.py`; the solve is the ordinary
{doc}`specifications` machinery.

## The `.pimdp` format

Like `.imdp` (see {doc}`formats`) with two changes:

- a region directive `param <name> <lo> <hi>`;
- `tran <s> <a> <to>:<expr> ...` where `<expr>` is **affine in the single parameter**:
  `c`, `p`, `1-p`, or `a*p+b`.

```text
# param_demo.pimdp — P(F target) = p; region p in [0.3, 0.7]
param p 0.3 0.7
states 3
init 0
label target 1
tran 0 0 1:p 2:1-p
tran 1 0 1:1
tran 2 0 2:1
```

## CLI example (validated)

```bash
# 1. Lift the parametric model over its region to an interval MDP
python3 benchmarks/crosstool/peers/param_lift.py \
    benchmarks/crosstool/models/param_demo.pimdp > /tmp/param_demo.imdp

# 2. Robust reachability on the lifted IMDP = region bounds on P_p(F target)
tools/imdp_solve /tmp/param_demo.imdp reach target --bound both
# result  prop=reach  bound=pess  state=0  lower=0.3  upper=0.300001  iters=2
# result  prop=reach  bound=opt   state=0  lower=0.7  upper=0.700001  iters=2
```

Here `P_p(F target) = p`, so the region `[0.3, 0.7]` maps exactly to the value range
`[0.3, 0.7]` — the lifting is exact because `p` occurs once per row
(`param_demo.ref.json`).

## Peer tools and validation

- **Storm** (`storm-pars`) is the reference implementation of Parameter Lifting region
  checking. In the recorded validation the lifted IMPaCT bounds `[0.3, 0.7]` were
  confirmed against **Storm corner instantiations** (`p=0.3 → 0.3`, `p=0.7 → 0.7`) and
  the analytic value (`param_demo.ref.json`; `storm-pars` itself was not built in the
  comparison container — the corners confirm the endpoints, while the lifting covers
  the whole region in one interval-MDP solve).
- **PRISM** supports parametric models via its param engine (solution functions);
  region verification per se is Storm's feature.
- **IntervalMDP.jl**: no parametric support.

## Limitations

- Expressions must be affine in **one** parameter per transition expression
  (`param_lift.py`); multi-parameter models need one `param` region but each edge
  expression references a single parameter. Rational-function transition
  probabilities (parametric *solution functions*, Daws ICTAC 2004 state elimination)
  are a planned item, not implemented (`FEATURE_PARITY.md` §B).
- When a parameter occurs multiple times in one distribution row, the lifted bound is
  sound but not tight (the relaxation decouples the occurrences) — the standard
  Parameter Lifting trade-off; refine by splitting the region.
- Region *refinement loops* (automatic splitting until a threshold is decided) are not
  provided; run the lift-and-solve per subregion yourself.
