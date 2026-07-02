# CLI Reference: `imdp_solve`

The dependency-free command-line solver over the neutral model formats. Source:
`tools/imdp_solve.cpp`.

## Build

```bash
c++ -std=c++17 -O2 tools/imdp_solve.cpp \
    src/imdp_io.cpp src/prism.cpp src/solve.cpp src/omaximization.cpp \
    src/graph_utils.cpp src/omega.cpp src/pctl.cpp src/ltlspec.cpp \
    src/ltl_spot.cpp src/ctmc.cpp src/odimdp.cpp src/smc.cpp src/exact.cpp \
    -o tools/imdp_solve
```

`ltlx` shells out to Spot's `ltl2tgba`; set `IMPACT_LTL2TGBA` to its command if not on
`PATH`.

## Usage

```text
imdp_solve MODEL(.imdp|.prism|.pm|.nm|.odimdp) PROP LABEL [flags]
```

Output: one tab-separated `result` line per requested bound,
`result  prop=...  bound=pess|opt  state=S  lower=L  upper=U  iters=N`
(`multi` prints `pareto` lines; `smc` prints `smc`/`smc-sprt` lines; `--exact` prints
`exact=`/`approx=`/`certified=`). Exit codes: 0 success, 1 parse/solve error, 2 usage
error. All examples below use models in `benchmarks/crosstool/models/`.

## Properties

| PROP | Meaning | Example |
|---|---|---|
| `reach L` | max P(F L); `--horizon k` gives P[F<=k L]; `--exact` gives rational value | `imdp_solve chain_point.imdp reach target --bound both` → pess/opt 0.25 |
| `safety L` | max P(G !L), L = avoid set | `imdp_solve safety_point.imdp safety avoid` → 0.72 |
| `until a,b` | max P(a U b) (reach-avoid); `--horizon k` gives a U<=k b | `imdp_solve patrol_cycle.imdp until r0,r2` → 0 |
| `next L` | P(X L) | `imdp_solve chain_point.imdp next target --state 1` → 0.5 |
| `reward L` | expected reward to L (γ=1) or discounted (`--discount g<1`; L still required) | `imdp_solve reward_chain.imdp reward target --bound both` → 3 / 5 |
| `lra -` | long-run average reward (no label; pass `-`) | `imdp_solve lra_cycle.imdp lra -` → 2 |
| `ss L` | steady-state probability of L (= LRA of indicator) | `imdp_solve ss_dtmc.imdp ss g` → 0.42857 |
| `csl L` | CTMC P(F<=`--time` L); model entries are rates | `imdp_solve ctmc_demo.imdp csl goal --time 1.0` → 0.6321205588 |
| `buchi L` | max P(G F L) | `imdp_solve buchi_routearound.imdp buchi acc --bound both` → 0 / 1 |
| `persist L` | max P(F G L) | `imdp_solve persist_leak.imdp persist safe --bound both` → 0.5 / 1 |
| `patrol a,b,..` | generalized Büchi: every set infinitely often | `imdp_solve patrol_cycle.imdp patrol r0,r2` → 1 |
| `ltl "φ"` | built-in fragment (X F G U, GF, FG, patrol conjunctions) | `imdp_solve ltl_demo.imdp ltl "G F a"` → 0.5 |
| `ltlx "φ"` | arbitrary LTL via Spot + Owl LDBA fallback (all of LTL) | `imdp_solve ltl_interval.imdp ltlx "G F a" --bound both` → 0.3 / 0.7 |
| `multi t1,t2,..` | multi-objective reachability; sweep or `--weights` | `imdp_solve multi_demo.imdp multi t1,t2` → vertices (0.9,0.1), (0.2,0.8) |
| `smc L` | Monte-Carlo estimate of P(F<=`--horizon` L), point chains | `imdp_solve bounded_demo.imdp smc goal --horizon 3` → estimate 0.87471 |

`LABEL` is a label name declared in the model; for `patrol`/`until`/`multi` it is
comma-separated; for `ltl`/`ltlx` it is the LTL formula (quote it); for `lra` it is
unused (pass `-`).

`.odimdp` inputs support `reach` only:
`imdp_solve od_demo.odimdp reach target --bound both` → 0.37143 / 0.85366.

## Flags

| Flag | Default | Meaning | Example |
|---|---|---|---|
| `--bound pess\|opt\|both` | `pess` | nature adversarial / cooperative / print both | `... reach target --bound both` |
| `--eps E` | `1e-6` | convergence tolerance (bracket gap <= 2E) | `... reach target --eps 1e-9` |
| `--method ovi\|mec` | `ovi` | OVI vs MEC-collapse interval iteration ({doc}`engines`) | `... reach target --bound opt --method mec` |
| `--state S` | model `init` | report the value of state S | `... next target --state 1` |
| `--horizon k` | 0 (unbounded); `smc`: 1000 | step bound for `reach`/`until` (exact DP) and `smc` | `... reach goal --horizon 3` → 0.875 |
| `--discount g` | 1.0 | `reward`: γ<1 switches to discounted reward | `... reward target --discount 0.9` → 2.8 / 4.2727 |
| `--time t` | 1.0 | `csl`: the CSL time bound | `... csl goal --time 2.0` → 0.8646647168 |
| `--weights w1,..,wk` | sweep (k=2) | `multi`: one weight vector instead of the sweep | `... multi t1,t2 --weights 0.5,0.5` |
| `--samples N` | 100000 | `smc`: number of simulated paths | `... smc goal --samples 1000000` |
| `--threshold p` | off | `smc`: add a Wald SPRT verdict for P >= p | `... smc goal --threshold 0.8` → `accept(P>=theta)` |
| `--exact` | off | `reach` only: rational robust policy iteration + certificate | `... reach target --exact` → `exact=1/4 certified=yes` |
| `--json` | off | emit model structure + per-state `[lower,upper]` values as JSON (web app) | `... reach target --json` |
| `--trace` | off | JSON plus the per-iteration VI trace (`reach`/`safety`; implies `--json`) | `... reach target --trace` |

## Worked example

```bash
cd benchmarks/crosstool/models
../../../tools/imdp_solve choice_interval.imdp reach target --bound both --eps 1e-6
# result  prop=reach  bound=pess  state=0  lower=0.4  upper=0.400001  iters=2
# result  prop=reach  bound=opt   state=0  lower=0.5  upper=0.500001  iters=2
```

Every example value above is a live output validated against the model's `.ref.json`
and, where a peer supports the query, against PRISM / Storm / IntervalMDP.jl
(`TOOL_COMPARISONS.md`).

## Notes

- The `usage:` banner lists the most common properties; `ss`, `csl`, `multi`, `smc`
  and the `--exact`/`.odimdp` modes are dispatched as documented here (source of truth:
  the dispatch in `tools/imdp_solve.cpp`).
- `reward` requires a valid label argument even in discounted mode.
- `--method mec` does not converge for the pessimistic sense on interval nature-traps
  (ISSUE-0003); keep the OVI default there.
- `--json`/`--trace` are intended for small models (the web app draws every edge).
