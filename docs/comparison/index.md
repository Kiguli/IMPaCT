# Tool comparison: IMPaCT vs PRISM vs Storm vs IntervalMDP.jl

This section compares IMPaCT against the three peer probabilistic model checkers with
interval-MDP (IMDP) support, on identical models with aligned robust semantics. It
summarises the repository's cross-tool study (`TOOL_COMPARISONS.md`, `FEATURE_PARITY.md`,
`benchmarks/crosstool/`); pairwise pages give more detail per tool:

- [IMPaCT vs PRISM](impact-vs-prism.md)
- [IMPaCT vs Storm](impact-vs-storm.md)
- [IMPaCT vs IntervalMDP.jl](impact-vs-intervalmdp.md)

## Versions compared

| Tool | Version | Obtained |
|---|---|---|
| **IMPaCT** | v2.0 (`IMPaCT-v2.0`) | built from source (AdaptiveCpp/SYCL, Armadillo, GLPK, HDF5, GSL, NLopt) |
| **PRISM** | 4.8.1 | official Linux binary (interval-MDP support since v4.8, 2024) |
| **Storm** | 1.13.0 | built from source in the `impact_work` container (`peers/build_storm.sh`) |
| **IntervalMDP.jl** | 0.6.0 | Julia 1.10, in the `impact_work` container |

## Methodology

Only IMPaCT *abstracts* a continuous stochastic system into an IMDP; the peers consume a
finite IMDP and only *solve* it. For a fair comparison, IMPaCT exports its abstracted
IMDP to a neutral `.imdp` file, and every tool solves the **identical model**:

- IMPaCT reads `.imdp` directly (`tools/imdp_solve`);
- Storm reads a DRN produced by `peers/imdp_to_drn.py`;
- PRISM reads explicit interval `.tra/.sta/.lab` from `peers/imdp_to_prism_interval.py`;
- IntervalMDP.jl reads the model via `peers/intervalmdp_runner.jl`.

One driver runs the whole sweep (`peers/sweep_compare.py`); raw output is in
`benchmarks/crosstool/sweep_all.tsv` and `sweep_arch.tsv`. Robust semantics are aligned
across tools before any number is compared:

| Sense | IMPaCT | PRISM | Storm | IntervalMDP.jl |
|---|---|---|---|---|
| **pess** (robust: controller maximises, nature adversarial) | `--bound pess` | `Pmaxmin=?[...]` | `--uncertainty-resolution robust` | `Pessimistic, Maximize` |
| **opt** (nature cooperative) | `--bound opt` | `Pmaxmax=?[...]` | `--uncertainty-resolution cooperative` | `Optimistic, Maximize` |

## Feature matrix

Audited from source and documentation, 2026-06-30, and updated with the v2.0 additions
tracked in `FEATURE_PARITY.md`. ✅ supported, ❌ not supported, N/A out of scope.

| Capability | IMPaCT | IntervalMDP.jl 0.6 | PRISM 4.8.1 | Storm 1.13.0 |
|---|---|---|---|---|
| **Abstract continuous stochastic dynamics → IMDP** (*IMPaCT-only*) | ✅ nonlinear (NLopt + sparse) | ❌ | ❌ | ❌ |
| Noise models (for abstraction) | Gaussian, multivariate, uniform, triangular, Laplace, custom PDF | N/A | N/A | N/A |
| Interval / robust (I)MDP solving | ✅ | ✅ | ✅ (since v4.8) | ✅ |
| Robust reachability / safety | ✅ | ✅ | ✅ | ✅ |
| Robust bounded / finite-horizon | ✅ | ✅ | ✅ `F<=k` | ✅ step-bounded |
| **Robust ω-regular on IMDPs** (Büchi/persistence/LTL) (*IMPaCT-only*) | ✅ | ❌ (DFA/co-safe only) | ❌ | ❌ |
| Expected reward / cost | ✅ reachability + discounted (robust) | ✅ discounted, exit-time | ✅ (+interval reach-reward) | ✅ |
| Long-run average / mean-payoff | ✅ robust interval (*interval LRA IMPaCT-only*) | ❌ | ✅ (non-interval) | ✅ (non-interval) |
| Steady-state (S operator) | ✅ robust interval | ❌ | ✅ (DTMC/CTMC only) | ✅ |
| Multi-objective / Pareto | ✅ robust reachability | ❌ | ✅ (non-interval) | ✅ (non-interval) |
| CTMC + CSL time-bounded reachability | ✅ | ❌ | ✅ | ✅ |
| **Interval CTMC (uncertain rates), robust CSL** (*IMPaCT-only*) | ✅ | ❌ | ❌ (point rates) | ❌ (point rates) |
| Orthogonal / mixture IMDPs (factored) | ✅ | ✅ | ❌ | ❌ |
| Parametric region verification | ✅ (parameter lifting → interval MDP) | ❌ | ❌ | ✅ (`storm-pars`) |
| Statistical MC / simulation | ✅ (point chains) | ❌ | ✅ `-sim` | ✅ |
| Controller / strategy synthesis | ✅ | ✅ | ✅ | ✅ |
| GPU acceleration | ✅ SYCL (any vendor) | ✅ CUDA | ❌ | ❌ |
| Exact / rational arithmetic | ✅ `--exact`, incl. certified interval-exact values | ✅ Rational | ✅ `-exact` (errors on intervals) | ✅ (errors on intervals) |
| Symbolic / BDD engine | ❌ | ❌ | ✅ MTBDD | ✅ dd (CUDD/Sylvan) |
| Full (non-robust) LTL / PCTL* / CSL | fragment + arbitrary det.-Büchi LTL via Spot | ❌ | ✅ | ✅ (Spot) |
| Other model classes | POMDP, PTA, TA | orthogonal-IMDP, mixture-IMDP, IMC | DTMC/CTMC/MDP/PTA/POMDP/IDTMC | DTMC/CTMC/MDP/MA/POMDP, parametric, DFT, GSPN |
| Solver methods | II, VI, OVI, MEC, O-max | robust VI (O-max) | VI/GS/PI/II/topological/symbolic | VI/II/OVI/SVI/PI/LP/topological/symbolic |
| Input formats | C++ dynamics, `.imdp`, PRISM-subset, HDF5 | netCDF, PRISM-explicit, bmdp-tool | PRISM lang, explicit, PEPA | PRISM, JANI, DRN, explicit, GSPN, DFT |

## Agreement: shared IMDP models, all four tools

Robust value at the initial state, `pess / opt`; ✅ = matches the analytic reference.

| model | spec | IMPaCT | IntervalMDP.jl | PRISM | Storm | ref |
|---|---|---|---|---|---|---|
| chain_point | reach | 0.25 / 0.25 | 0.25 / 0.25 | 0.25 / 0.25 | 0.25 / 0.25 | 0.25 ✅ |
| safety_point | safety | 0.72 / 0.72 | 0.72 / 0.72 | 0.72 / 0.72 | 0.72 / 0.72 | 0.72 ✅ |
| choice_interval | reach | 0.40 / 0.50 | 0.40 / 0.50 | 0.40 / 0.50 | 0.40 / 0.50 | 0.40 / 0.50 ✅ |
| fork_interval | reach | 0.40 / 0.60 | 0.40 / 0.60 | 0.40 / 0.60 | 0.40 / 0.60 | 0.40 / 0.60 ✅ |
| multiObj_robotIMDP (207-state) | reach | 0.894663 / 1.0 | 0.894663† / 1.0 | 0.894662 / 1.0 | 0.894663 / 1.0 | agree ~1e-6 ✅ |
| buchi_reach | `GF acc` | 0.5 / 0.5 | N/A | N/A | N/A | 0.5 ✅ |
| buchi_routearound | `GF acc` | 0 / 1 | N/A | N/A | N/A | 0 / 1 ✅ |
| persist_leak | `FG safe` | 0.5 / 1.0 | N/A | N/A | N/A | 0.5 ✅ |
| patrol_cycle | `GF r0 ∧ GF r2` | 1.0 / 1.0 | N/A | N/A | N/A | 1.0 ✅ |

All four tools agree on every reachability/safety model. The four ω-regular models are
IMPaCT-only: Storm reports *"LTL with intervals not implemented"*, PRISM *"Computation
not implemented yet"*, and IntervalMDP.jl offers DFA/co-safe reachability only.
† IntervalMDP.jl's default 1e-6 threshold stops at 0.894662 (1.2e-6 low); at a tight
threshold it reaches 0.894663, equal to the others.

On the ARCH-COMP 2025 abstracted IMDPs (AS, BA, IC, PD, VP — IMPaCT abstracts, the peers
solve the exported model), **all four tools agree on every robust value**; optimistic
reach also agrees to high precision (e.g. IC_reach opt ≈ 1.20e-26 in all four). See the
pairwise pages for the timing breakdowns.

## Cross-tool findings

Every observed difference is a documented semantic distinction or a performance
characteristic — no tool returned a different correct value on the same model.

- **PRISM rejects zero lower bounds.** PRISM 4.8.1 needs a strictly positive support
  graph and aborts with *"Transition probability has lower bound of 0"* on abstracted
  IMDPs; the sweep floors `[0,hi]` edges to `[1e-9,hi]`, so PRISM's ARCH values are an
  ε-perturbation. PRISM also reports only the initial-state value.
- **Convergence thresholds must match.** IntervalMDP.jl's default reachability threshold
  (1e-6) stops ~1.2e-6 short of the fixpoint on slowly-mixing chains. IMPaCT's OVI
  returns the converged value *and* a certified upper bound at the same tolerance.
- **Storm's interval rewards ignore the resolution mode.** On interval reward models
  Storm returns only the nature-max value regardless of `--uncertainty-resolution`
  (`Rmin` unsupported), so robust interval rewards were validated against PRISM's
  `Rmaxmin/Rmaxmax` and IntervalMDP.jl instead.
- **Factored (odIMDP) semantics are reduction-order sensitive** (ISSUE-0022). The robust
  value of an orthogonally-decoupled IMDP depends on which dimension nature resolves
  innermost; IMPaCT pins IntervalMDP.jl's dimension-1-innermost convention and then
  matches it to 10 digits in both senses.
- **Exact mode surfaces infeasible interval rows** (ISSUE-0023). `imdp_solve reach
  --exact` rejects rows of the IntervalMDP.jl-shipped robot model whose upper bounds sum
  to slightly below 1 (e.g. 0.999997); all floating-point solvers, IMPaCT's included,
  silently tolerate this and assign the missing ~3e-6 mass to the best successor.
- **Optimistic safety is convention-dependent** across tools (how the cooperative sense
  resolves `[0,hi]` leak edges); the robust value is unambiguous and agrees, so robust is
  the headline number everywhere.

## Reproducing the comparison

```bash
# in the impact_work container (repo mounted at /work):
python3 benchmarks/crosstool/peers/sweep_compare.py all     # 4-tool sweep -> TSV
tools/imdp_solve MODEL.imdp reach|safety|buchi|persist|patrol LABEL --bound both
```

Converters and runners live in `benchmarks/crosstool/peers/` (`tra_to_imdp.py`,
`imdp_to_drn.py`, `imdp_to_prism_interval.py`, `intervalmdp_runner.jl`,
`build_storm.sh`). Per-model reference values with measured per-tool numbers are the
`benchmarks/crosstool/models/*.ref.json` files; the IMPaCT-side benchmark log is
`benchmarks/RESULTS.md`; design decisions and discrepancies are filed under `issues/`.
