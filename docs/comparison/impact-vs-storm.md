# IMPaCT vs Storm

[Storm](https://www.stormchecker.org/) (compared at version 1.13.0, built from source in
the `impact_work` container via `peers/build_storm.sh`) is a high-performance
probabilistic model checker with the broadest model-class coverage of the peers:
DTMC/CTMC/MDP/MA/POMDP, parametric models (`storm-pars`), DFTs and GSPNs. It solves
interval MDPs with `--uncertainty-resolution robust|cooperative`, which maps directly to
IMPaCT's `--bound pess|opt`.

## Feature overlap

| Storm has (IMPaCT does not) | Both | IMPaCT has (Storm does not) |
|---|---|---|
| Markov automata (MA) | Interval-MDP robust reachability / safety and step-bounded `F<=k` | Abstraction of continuous nonlinear stochastic dynamics → IMDP (6 noise models) |
| Parametric solution functions; DFT / GSPN front ends | CTMC + CSL time-bounded reachability | Robust ω-regular and arbitrary det.-Büchi LTL **on IMDPs** (Storm: *"LTL with intervals not implemented"*) |
| JANI input; symbolic dd engine (CUDD/Sylvan) | Long-run average, steady-state, multi-objective (Storm: non-interval only) | Robust **interval** LRA, steady-state, and multi-objective |
| Full LTL via Spot (non-interval) | Expected rewards (see the interval-reward caveat below) | Robust interval rewards in both senses (reachability + discounted) |
| Sound-VI family (SVI etc.), LP solvers | Exact arithmetic; statistical model checking; strategy synthesis | Exact arithmetic **on interval models** (Storm is point-only); interval CTMC robust CSL |
| — | Parametric region verification (Storm `storm-pars`; IMPaCT via parameter lifting) | GPU (SYCL) solving; orthogonal/mixture IMDPs |

## Same results on Storm-validated queries

Storm is the oracle for most of IMPaCT's v2.0 capability additions; each value below is
from a live run recorded in the model's `.ref.json` measured block.

| model / query | IMPaCT | Storm | analytic |
|---|---|---|---|
| `bounded_demo`: `P[F<=k goal]`, k=1,2,3 | 0.5 / 0.75 / 0.875 | 0.5 / 0.75 / 0.875 (`Pmax=?[F<=k "goal"]`) | 1−0.5^k ✅ |
| `ctmc_demo`: CSL `P(F<=t goal)`, t=1, 2, 0.5 | 0.6321205588 / 0.8646647168 / 0.3934693403 | identical to 10 digits | 1−e^{−t} ✅ |
| `lra_cycle`: long-run average reward | 2.0 / 2.0 | 2.0 (`R{"r"}max=?[LRA]`) | 2 ✅ |
| `ltl_demo`: `F a`, `G F a`, `X X a` | 0.5 / 0.5 / 0.5 (`ltlx`) | 0.5 / 0.5 / 0.5 (`Pmax=?[φ]`) | ✅ |
| `multi_demo`: Pareto for `(F t1, F t2)` | vertices (0.9, 0.1), (0.2, 0.8) | 2 Pareto points; corners `Pmax[F t1]`=0.9, `Pmax[F t2]`=0.8 | ✅ |
| `ss_dtmc`: steady-state of `g` | 0.4285698 | 0.4285714286 (`S=?[g]`) | 3/7 ✅ |
| `param_demo`: region p ∈ [0.3, 0.7] | [0.3, 0.7] via lifting | corner instantiations 0.3 and 0.7 | ✅ |

On the shared interval reference models and the ARCH abstracted IMDPs, Storm agrees with
IMPaCT on every robust value (multiObj_robotIMDP pess 0.8946629814 vs IMPaCT
0.8946629826; optimistic reach agrees to high precision, e.g. IC_reach opt ≈ 1.20e-26).

## Differences and caveats

- **Interval rewards ignore the resolution mode.** On `reward_chain` Storm returns only
  the nature-max reward 4.999996 (= the optimistic value) regardless of
  `--uncertainty-resolution`, and `Rmin` is unsupported — so robust interval rewards
  were cross-validated against PRISM and IntervalMDP.jl instead.
- **No robust ω-regular / LRA on intervals.** Storm refuses LTL on interval models
  (*"We have not yet implemented LTL with intervals"*,
  `SparseMdpPrctlModelChecker.cpp:239`) and likewise interval LRA; those capabilities
  are IMPaCT-only.
- **Point-only exact and CSL.** Storm's exact engine and CSL operate on point models;
  IMPaCT additionally certifies exact robust values on interval models and solves
  interval CTMCs (uncertain rates), which Storm does not support.
- **No abstraction front end.** Storm consumes DRN/PRISM/JANI models; the
  continuous-dynamics-to-IMDP phase is IMPaCT-only.

## Performance

Storm's solver is very fast once the model is in memory, but it pays a
model-construction cost to ingest the dense DRN. From the 2026-06-27 cross-tool run
(seconds; IMPaCT II = interval iteration, VI = value iteration):

| benchmark | IMPaCT robust solve (II / VI) | Storm VI | Storm build |
|---|---|---|---|
| AS | 4.3 (finite) | 0.21 | 0.50 |
| BA | 0.27 (finite) | 5.86 | 6.05 |
| VP | 29.1 / **0.59** | 1.35 | 0.37 |
| IC_reach | 0.06 (finite) | 0.03 | 0.05 |
| IC_safe | 0.47 (finite) | 0.09 | 0.18 |
| PD_p1 | 1.29 / 1.34 | 8.31 | 7.26 |
| PD_p3 | ∞ (II) / **1.36** | 8.30 | 7.27 |

IMPaCT's finite-horizon robust VI is competitive or faster (BA 0.27 s vs Storm's
5.86 s + 6.05 s build), and its pure-VI mode beats Storm on VP (0.59 s vs 1.35 s) and
PD (1.34–1.36 s vs ~8.3 s). Honest flip side: in the later sweep re-run
(`sweep_arch.tsv`), Storm's end-to-end times on several ARCH models are the best of all
four tools (VP 0.04 s, AS 0.05 s, vs IMPaCT 1.8 s / 0.4 s) — and the abstraction that
produced these IMDPs (e.g. AS 27 s dense / 0.11 s sparse) is an IMPaCT-side cost Storm
never pays because it cannot abstract.

## Reproduce

```bash
# build Storm 1.13.0 from source inside the impact_work container
bash benchmarks/crosstool/peers/build_storm.sh

# convert the exported IMDP to Storm DRN and run the sweep
python3 benchmarks/crosstool/peers/imdp_to_drn.py MODEL.imdp
python3 benchmarks/crosstool/peers/sweep_compare.py all
```

Raw results: `benchmarks/crosstool/sweep_all.tsv`, `sweep_arch.tsv`; per-model measured
values: `benchmarks/crosstool/models/*.ref.json`.
