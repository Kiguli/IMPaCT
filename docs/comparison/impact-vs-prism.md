# IMPaCT vs PRISM

[PRISM](https://www.prismmodelchecker.org/) (compared at version 4.8.1, official Linux
binary) is the classic probabilistic model checker: PRISM-language models, full
PCTL*/LTL/CSL on DTMCs/CTMCs/MDPs, and — since v4.8 (2024) — interval MDPs. It
auto-detects `Type: IMDP` on explicit models and solves `Pmaxmin/Pmaxmax=?[F ...]`,
which is exactly IMPaCT's robust/optimistic reachability.

## Feature overlap

| PRISM has (IMPaCT does not) | Both | IMPaCT has (PRISM does not) |
|---|---|---|
| Full PCTL*/LTL/CSL (non-interval) | Interval-MDP robust reachability / safety (`Pmaxmin`/`Pmaxmax` = `--bound pess/opt`) | Abstraction of continuous nonlinear stochastic dynamics → IMDP (6 noise models) |
| Symbolic MTBDD engine | Bounded reachability `F<=k` | Robust ω-regular and arbitrary det.-Büchi LTL **on IMDPs** |
| IDTMC model class | Reachability rewards on intervals (PRISM `Rmaxmin/Rmaxmax`) | Robust interval long-run average, steady-state, multi-objective |
| Multi-objective / Pareto (non-interval) | Exact / rational arithmetic (PRISM `-exact`, IMPaCT `--exact`) | Exact arithmetic **on interval models** (PRISM is point-only) |
| Steady-state `S` operator (DTMC/CTMC only) | Statistical model checking (PRISM `-sim`, IMPaCT `smc`) | Interval CTMC (uncertain rates) with robust CSL |
| PEPA input, GUI, simulator tooling | CTMC + CSL time-bounded reachability; POMDP; PTA; strategy synthesis | GPU (SYCL) solving; orthogonal/mixture IMDPs; parametric region verification |

## Same results on shared models

All values below are from live runs recorded in `TOOL_COMPARISONS.md` and the
`benchmarks/crosstool/models/*.ref.json` measured blocks.

**Interval reference models.** PRISM reproduces every interval reference value exactly:
chain_point 0.25, safety_point 0.72, choice_interval 0.40/0.50, fork_interval 0.40/0.60.
On the 207-state `multiObj_robotIMDP`, PRISM returns 0.8946624 (pess) — agreeing with
IMPaCT/Storm's 0.8946630 to ~1e-6 at default tolerances — and 1.0 (opt).

**PRISM-language front end.** IMPaCT parses a PRISM-language subset directly:
`choice.prism` and the equivalent `choice_interval.imdp` solve to identical values
(pess 0.4 / opt 0.5) through one solver path.

**Robust rewards.** On `reward_chain` (interval MDP with rewards), IMPaCT's reachability
reward is pess 3.0 / opt 4.999999 against PRISM's `Rmaxmin` 3.000000002 / `Rmaxmax`
4.999996 — the analytic values are 3 and 5. PRISM is the oracle here because Storm's
interval rewards ignore the resolution mode.

**Exact arithmetic, 1/4.** On the point chain, IMPaCT's `imdp_solve reach --exact`
returns exactly 1/4, matching PRISM `-exact` (and Storm's exact mode). On *interval*
models IMPaCT certifies exact robust values (2/5, 1/2, 3/5 on the reference models)
where PRISM errors — its exact engine is point-only (verified).

**Statistical model checking.** IMPaCT's `imdp_solve smc` estimate 0.87471 (Wilson CI
containing the analytic 0.875) matches PRISM `-sim` 0.87479 on the same point chain.

**ARCH abstracted IMDPs.** After flooring (below), PRISM agrees with the other three
tools on every ARCH robust value (AS/IC/VP reach 0, PD 1.0, BA safety 0, IC_reach opt
≈ 1.20e-26).

## Differences and caveats

- **Zero lower bounds are rejected.** PRISM needs a fixed, strictly positive support
  graph and aborts with *"Transition probability has lower bound of 0"*. Abstracted
  IMDPs are full of `[0,hi]` edges, so the sweep floors them to `[1e-9,hi]` for PRISM
  only; its ARCH values are therefore an ε-perturbation of the true robust value.
- **No robust ω-regular on intervals.** LTL on interval models fails with *"Computation
  not implemented yet"*; the Büchi/persistence/patrol reference models are PRISM-N/A.
- **Initial-state reporting.** `prism` reports only the initial-state value; per-state
  comparisons (e.g. `chain_point` state 1 = 0.5) need the other tools.
- **No abstraction front end.** PRISM consumes a finite model; the entire
  continuous-dynamics-to-IMDP phase is IMPaCT-only.

## Performance

From the four-tool sweep on the ARCH abstracted IMDPs (solve phase only; times in
seconds, IMPaCT / IntervalMDP.jl / PRISM / Storm):

| benchmark | spec | time (s) |
|---|---|---|
| AS (anaesthesia 3D) | reach | 0.4 / 0.5 / 0.5 / 0.05 |
| IC_reach (integrator 2D) | reach | 0.1 / 0.4 / 0.2 / 0.05 |
| IC_safe (integrator 2D) | safety | 0.15 / 0.5 / 0.06 / 0.02 |
| PD_p1 (pkg delivery 2D) | reach | 3.4 / 1.4 / 1.7 / 2.8 |
| PD_p3 (pkg delivery 2D) | reach | 3.6 / 1.6 / 2.3 / 2.2 |
| VP (Van der Pol 2D) | reach | 1.8 / 0.55 / 1.4 / 0.04 |

PRISM's solve times are in the same band as IMPaCT's on these models — faster on some
(IC_safe 0.06 s vs 0.15 s, PD_p1 1.7 s vs 3.4 s), slower on others (small reference
models: ~0.01 s vs IMPaCT's ~0.002–0.006 s). Note that the peers are handed a pre-built
IMDP: the abstraction cost (e.g. AS 27 s dense / 0.11 s sparse; PD ~37 s dense) is an
IMPaCT-side cost with no PRISM equivalent, because PRISM cannot perform this phase at
all.

## Reproduce

```bash
# convert IMPaCT's exported IMDP to PRISM explicit interval format
python3 benchmarks/crosstool/peers/imdp_to_prism_interval.py MODEL.imdp

# run the whole four-tool sweep (handles the [1e-9,hi] floor for PRISM)
python3 benchmarks/crosstool/peers/sweep_compare.py all
```

Raw results: `benchmarks/crosstool/sweep_all.tsv`, `sweep_arch.tsv`; per-model measured
values: `benchmarks/crosstool/models/*.ref.json`.
