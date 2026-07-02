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
| `ltlx "FORMULA"` | arbitrary LTL: Spot (deterministic class) with automatic Owl LDBA fallback |

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

**Arbitrary LTL** (`ltlx`, `src/ltl_spot.h`): a **two-stage** pipeline covering *all of
LTL* (ISSUE-0016, resolved 2026-07-02):

1. translate the formula with Spot's `ltl2tgba -D` (Duret-Lutz et al., *Spot 2.0*);
2. if the result is a **deterministic (generalized) Büchi** automaton: synchronous
   product with the IMDP (a deterministic automaton is always good-for-MDPs — Hahn,
   Perez, Schewe, Somenzi, Trivedi, Wojtczak, *Good-for-MDPs Automata*, TACAS 2020 —
   so the product value is the LTL value), solved robustly with
   `omega::maxGenBuchi{Pessimistic,Optimistic}`;
3. otherwise (co-Büchi `F G`-type and beyond) `ltlx` **falls back automatically** to a
   **limit-deterministic Büchi automaton** from Owl's `ltl2ldba` (Křetínský,
   Meggendorfer, Sickert, *Owl: A Library for ω-Words, Automata, and LTL*, ATVA 2018,
   DOI 10.1007/978-3-030-01090-4_34; LDBA theory: Sickert, Esparza, Jaax, Křetínský,
   CAV 2016). The LDBA's ε-jumps to its accepting component appear as same-letter
   nondeterminism, which the product resolves by the **controller** — each matching
   automaton edge multiplies the product action space (sound because such LDBAs are
   good-for-MDPs, Hahn et al. TACAS 2020); transition-based Büchi marks become
   state-based accepting *entry copies*, and the same robust Büchi engine applies
   unchanged.

The Owl fallback is enabled by pointing `IMPACT_LTL2LDBA` at Owl's native binary
(no JRE needed for the Linux amd64 release):

```bash
export IMPACT_LTL2LDBA="/opt/owl/owl-linux-musl-amd64-21.0/bin/owl ltl2ldba"
```

Without it, nondeterministic-automaton formulas are rejected soundly with an error
suggesting the `ltl` fragment (which covers persistence) as the dependency-free
alternative.

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

# Nondeterministic-automaton formula via the Owl LDBA fallback (ISSUE-0016, resolved):
export IMPACT_LTL2LDBA="/opt/owl/owl-linux-musl-amd64-21.0/bin/owl ltl2ldba"
tools/imdp_solve benchmarks/crosstool/models/ltl_demo.imdp ltlx "F G a" --bound opt
# result  prop=ltlx  bound=opt  state=0  lower=0.5  upper=0.500001  iters=18
#   (== the built-in persistence solver == Storm Pmax=?[F G "a"] = 0.5)

# The same on the INTERVAL model — robust full LTL, unique to IMPaCT:
tools/imdp_solve benchmarks/crosstool/models/ltl_interval.imdp ltlx "F G a" --bound both
# result  prop=ltlx  bound=pess  state=0  lower=0.3  upper=0.300001  iters=13
# result  prop=ltlx  bound=opt   state=0  lower=0.7  upper=0.700001  iters=14

# Beyond-fragment formula, LDBA route, validated against Storm (0.5):
tools/imdp_solve benchmarks/crosstool/models/ltl_demo.imdp ltlx "F (a & X a)" --bound opt
# result  prop=ltlx  bound=opt  state=0  lower=0.5  upper=0.500001  iters=13
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

- `ltlx` covers **all of LTL**: the deterministic-(generalized-)Büchi class via Spot,
  and the nondeterministic class (co-Büchi `F G` etc.) via Owl's limit-deterministic
  automata with the controller-resolved jump product (ISSUE-0016, resolved; validated
  `F G a` = persistence solver = Storm on the point model, and 0.3/0.7 analytically on
  the interval model). LDBAs with *generalized* (multi-set) acceptance are not yet
  handled by the fallback (Owl's `ltl2ldba` emits single-set Büchi).
- `ltlx` requires external translators (Spot's `ltl2tgba`; Owl's `ltl2ldba` for the
  fallback); the `ltl` fragment has no external dependency.
- On unbounded-noise abstractions the infinite-horizon ω-regular value is often 0
  because no non-trivial robust ECs survive abstraction (ISSUE-0011, a modelling fact,
  not a solver bug).
