# File Formats and Converters

The neutral exchange formats read by `imdp_solve`, and the converters that move models
between IMPaCT and PRISM / Storm / IntervalMDP.jl.

## `.imdp` — explicit (interval) MDP

Whitespace-separated text, `#` comments, blank lines ignored (`src/imdp_io.h`):

```text
states <N>                       # required, first
init <s>                         # optional, default 0
label <NAME> <s> <s> ...         # repeatable; accumulates state sets
reward <s> <value>               # optional; per-state reward (expected-reward / LRA props)
tran <s> <a> <to>:<lo>:<hi> ...  # one line = one action of state <s>: an interval distribution
```

- **Point MDPs / DTMCs** use `lo == hi` on every edge.
- **Rewards**: states without a `reward` line have reward 0. Rewards are state-based.
- **Rates (CTMCs)**: a CTMC is a plain `.imdp` whose single action per state carries
  transition **rates** in the `to:r:r` entries (interval rates `to:rlo:rhi` give an
  interval CTMC). The entries are interpreted as rates only by the `csl` property —
  see {doc}`ctmc-csl`. Example: `benchmarks/crosstool/models/ctmc_demo.imdp`.
- The parser validates state ranges and rejects unknown directives; `io::write`
  round-trips `io::parse`.

## `.odimdp` — orthogonal / mixture IMDP

Factored per-dimension marginals over a product state space (see {doc}`odimdp`):

```text
odimdp
dims <n1> <n2> ...                     # nStates = n1*n2*...; linearised, dim 0 fastest
init <s>
label <NAME> <s> ...
otran  <s> <a> <d> to:lo:hi ...        # dim-d marginal interval distribution (1 component)
mtran  <s> <a> <k> <d> to:lo:hi ...    # dim-d marginal of mixture component k
mweight <s> <a> k:lo:hi ...            # interval mixture weights over components
```

`to` in `otran`/`mtran` is a **dimension-d destination coordinate** `0..dims[d]-1`,
not a global state. Examples: `od_demo.odimdp`, `mix_demo.odimdp`.

## `.pimdp` — parametric MDP

Like `.imdp` plus a region directive; solved by lifting to an interval MDP (see
{doc}`parametric`):

```text
param <name> <lo> <hi>             # the parameter region
tran <s> <a> <to>:<expr> ...       # <expr> affine in ONE parameter: c, p, 1-p, a*p+b
```

## PRISM-language subset

`imdp_solve` also reads `.prism` / `.pm` / `.nm` files through a PRISM-language
front-end (`src/prism.h`) targeting the same `Problem` struct, so both front ends feed
one solver path. Validated: `choice.prism` gives the identical robust values (0.4 /
0.5) as its `.imdp` twin (`choice.prism.ref.json`).

## Exporting an abstraction: `IMDP::exportIMDP`

The abstraction front end writes its in-memory interval matrices to the neutral format
with `exportIMDP("model.imdp")` (`src/GPU_synthesis.cpp`) — this is how the cross-tool
comparisons run peers on IMPaCT's abstractions (`TOOL_COMPARISONS.md`). Note: the
export ignores the disturbance dimension (`dim_w > 0` prints a warning).

## Converters (`benchmarks/crosstool/peers/`)

| Script | Direction | Notes |
|---|---|---|
| `imdp_to_drn.py MODEL.imdp OUT.drn` | `.imdp` → Storm DRN | `@type: MDP`, `@value_type: double-interval`; preserves state order and labels, carries `reward` lines |
| `imdp_to_prism_interval.py MODEL.imdp OUTBASE [--floor EPS]` | `.imdp` → PRISM explicit (`.tra/.sta/.lab`) | PRISM ≥ 4.8 auto-detects the IMDP; PRISM **rejects `[0,hi]` edges** ("lower bound of 0"), so pass `--floor 1e-9` for abstracted models — the PRISM value is then an ε-perturbation (marked `**` in comparisons) |
| `imdp_to_prism.py` | `.imdp` → PRISM explicit, point-only | for degenerate models |
| `tra_to_imdp.py BASE > out.imdp` | PRISM-explicit interval `.tra`+`.lab` → `.imdp` | e.g. IntervalMDP.jl's shipped `multiObj_robotIMDP` |
| `param_lift.py MODEL.pimdp > MODEL.imdp` | `.pimdp` → `.imdp` | Parameter Lifting ({doc}`parametric`) |
| `intervalmdp_runner.jl` | runs IntervalMDP.jl on `.imdp`-derived models | peer oracle |

```bash
# Round-trip example: solve IMPaCT's model in Storm
python3 benchmarks/crosstool/peers/imdp_to_drn.py \
    benchmarks/crosstool/models/choice_interval.imdp /tmp/choice.drn
storm --explicit-drn /tmp/choice.drn --uncertainty-resolution robust \
    --prop 'Pmax=? [F "target"]'
```

## Interoperability caveats

- **PRISM zero lower bounds**: fixed positive support graph required; floor `[0,hi]`
  edges first (above).
- **Infeasible rows**: some third-party models contain rows with `sum(hi) < 1`
  (truncated decimals); floating-point solvers silently repair these, IMPaCT's exact
  mode rejects them (ISSUE-0023, see {doc}`exact`).
- **Thresholds**: IntervalMDP.jl's default 1e-6 residual can stop ~1e-6 short of the
  fixpoint on slowly-mixing chains — tighten it when comparing digits
  (`TOOL_COMPARISONS.md` §4).
