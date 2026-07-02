# ω-Regular and LTL Objectives

Robust Büchi, persistence, generalized-Büchi ("patrol"), a built-in LTL fragment, and
arbitrary LTL via Spot — on interval MDPs, where no peer tool offers any ω-regular
capability.

## What is computed

| Property | Objective |
|---|---|
| `buchi L` | max P(G F L) — visit `L` infinitely often |
| `persist L` | max P(F G L) — eventually remain in `L` forever |
| `patrol a,b,...` | max P(G F a ∧ G F b ∧ ...) — generalized Büchi |
| `ltl "FORMULA"` | built-in fragment: atoms, `! & \| ->`, X, F, G, U, G F, F G, conjunctions of G F |
| `ltlx "FORMULA"` | arbitrary LTL whose deterministic automaton is (generalized) Büchi, via Spot |

Pessimistic = the controller wins against *every* nature resolution; optimistic =
nature cooperates.

## Algorithm and literature

**Büchi** (`src/omega.h`) reduces to reachability of accepting end components — the
classical reduction (de Alfaro, PhD thesis, 1997; Baier & Katoen, *Principles of Model
Checking*, Ch. 10). The two senses differ structurally:

- *Optimistic*: support-graph accepting MECs (exact for that sense and point MDPs).
- *Pessimistic (robust)*: support MECs are **unsound** — inside a support EC nature can
  route around the accepting state using `lo=0` edges (ISSUE-0009). IMPaCT computes the
  robust almost-sure-Büchi winning region by a nested fixpoint of robust EC closure and
  robust almost-sure reachability — the qualitative 2.5-player Büchi computation
  (Chatterjee-Henzinger graph games; the IMDP reading follows Dutreix-Coogan permanent
  components and Asadi et al. force attractors) — then robust reachability of it.

**Persistence** F G p is robust reachability of the largest controllably-invariant
sub-region of `p`; **patrol** is degeneralized to Büchi by a round-robin counter
product.

**Arbitrary LTL** (`ltlx`, `src/ltl_spot.h`): the pipeline is

1. translate the formula with Spot's `ltl2tgba -D` (Duret-Lutz et al., *Spot 2.0*);
2. parse the HOA automaton; require a **deterministic (generalized) Büchi** automaton;
3. synchronous product with the IMDP (a deterministic automaton is always good-for-MDPs
   — Hahn, Perez, Schewe, Somenzi, Trivedi, Wojtczak, *Good-for-MDPs Automata*, TACAS
   2020 — so the product value is the LTL value);
4. solve robustly with `omega::maxGenBuchi{Pessimistic,Optimistic}`.

The LDBA/product route follows Sickert, Esparza, Jaax, Křetínský,
*Limit-Deterministic Büchi Automata for LTL*, CAV 2016. Formulas whose deterministic
automaton needs `Fin` acceptance (co-Büchi F G, Rabin/parity) are **rejected soundly**
with an error naming ISSUE-0016 — use the built-in `ltl` fragment (which covers
persistence) for those.

Set the environment variable `IMPACT_LTL2TGBA` to the `ltl2tgba` command if it is not
on `PATH` (it may embed `LD_LIBRARY_PATH=... .../ltl2tgba`).

## CLI examples (validated)

```bash
# Robust vs optimistic Büchi where nature can route around the accepting state
tools/imdp_solve benchmarks/crosstool/models/buchi_routearound.imdp buchi acc --bound both
# result  prop=buchi  bound=pess  state=0  lower=0  upper=1e-06  iters=1
# result  prop=buchi  bound=opt   state=0  lower=1  upper=1      iters=1

# Persistence with a [0,hi] leak edge: robust 0.5, optimistic 1.0
tools/imdp_solve benchmarks/crosstool/models/persist_leak.imdp persist safe --bound both

# Patrol two regions on a forced cycle: probability 1
tools/imdp_solve benchmarks/crosstool/models/patrol_cycle.imdp patrol r0,r2

# Arbitrary LTL via Spot on an interval MDP (IMPaCT-only): robust 0.3 / optimistic 0.7
export IMPACT_LTL2TGBA=ltl2tgba
tools/imdp_solve benchmarks/crosstool/models/ltl_interval.imdp ltlx "G F a" --bound both
# result  prop=ltlx  bound=pess  state=0  lower=0.3  upper=0.300001  iters=2
# result  prop=ltlx  bound=opt   state=0  lower=0.7  upper=0.700001  iters=2

# Sound rejection of a nondeterministic-automaton formula (ISSUE-0016)
tools/imdp_solve benchmarks/crosstool/models/ltl_demo.imdp ltlx "F G a"
# solve error: ltl_spot: 'F G a' compiles to a NON-deterministic automaton (e.g.
# co-Buchi / F G-type) — sound only via an LDBA (ISSUE-0016). Use the `ltl` fragment
# solver (it handles F/G/U/X/GF/FG/persistence) for such formulas.
```

Reference values: `buchi_reach` 0.5, `buchi_routearound` 0 / 1, `persist_leak`
0.5 / 1.0, `patrol_cycle` 1.0; `ltl_demo` point MDP `F a` / `G F a` / `X X a` = 0.5 and
`G (a -> X a)` = 1.0 (all `*.ref.json`).

## Peer tools and validation

No peer does robust ω-regular on intervals: Storm reports *"LTL with intervals not
implemented"*, PRISM *"Computation not implemented yet"*, IntervalMDP.jl supports only
DFA/co-safe reachability. Validation is therefore two-legged (`TOOL_COMPARISONS.md`
§1, §1b): on **point** MDPs, `ltlx` values equal Storm `Pmax=?[φ]` for `F a`, `G F a`,
`X X a`, `G(a→X a)`; on **interval** models the values match the exact analytic
references above. The robust-Büchi core (ISSUE-0009) is oracle-validated separately.

## Limitations

- `ltlx` covers the deterministic-(generalized-)Büchi class; co-Büchi/Rabin formulas
  require an LDBA (Owl) or a robust parity solver — open as ISSUE-0016. The `ltl`
  fragment handles the common persistence case natively.
- `ltlx` requires an external Spot installation (`ltl2tgba`); the `ltl` fragment has no
  external dependency.
- On unbounded-noise abstractions the infinite-horizon ω-regular value is often 0
  because no non-trivial robust ECs survive abstraction (ISSUE-0011, a modelling fact,
  not a solver bug).
