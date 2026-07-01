# IMPaCT vs PRISM / Storm / IntervalMDP.jl — cross-tool comparison

A like-for-like comparison of **IMPaCT** against three peer probabilistic model checkers
— **PRISM**, **Storm**, **IntervalMDP.jl** — on (1) the **ARCH-COMP 2025 stochastic
benchmarks** ([`examples/ARCH-COMP/2025`](examples/ARCH-COMP/2025)) and (2) the shared
**cross-tool reference / example IMDPs** ([`benchmarks/crosstool/models`](benchmarks/crosstool/models)),
plus a **feature matrix** of everything the tools can do beyond what was measured.

**The one structural difference:** only IMPaCT *abstracts* a continuous stochastic system
into an interval MDP (IMDP). PRISM/Storm/IntervalMDP.jl consume a finite IMDP and only
*solve* it. So for a fair comparison IMPaCT exports its abstracted IMDP
([`IMDP::exportIMDP`](src/GPU_synthesis.cpp) → neutral `.imdp`), and the peers solve the
identical model (`imdp_to_drn.py` → Storm DRN; `imdp_to_prism_interval.py` → PRISM
explicit; `intervalmdp_runner.jl` → IntervalMDP.jl). One driver runs everything:
[`peers/sweep_compare.py`](benchmarks/crosstool/peers/sweep_compare.py).

## Tools & versions

| Tool | Version | Obtained |
|---|---|---|
| **IMPaCT** | v2.0 (`IMPaCT-v2.0`) | built from source (AdaptiveCpp/SYCL, Armadillo, GLPK, HDF5, GSL, NLopt) |
| **IntervalMDP.jl** | 0.6.0 | Julia 1.10, in the `impact_work` container |
| **PRISM** | 4.8.1 | official Linux binary (interval-MDP support since v4.8, 2024) |
| **Storm** | 1.13.0 | **built from source** in `impact_work` ([`build_storm.sh`](benchmarks/crosstool/peers/build_storm.sh)) |

## How to read the tables

- Values are the **robust value at the initial state**, `pess / opt`:
  **pess** = robust = controller maximises, nature adversarial (IMPaCT `--bound pess` /
  PRISM `Pmaxmin` / Storm `--uncertainty-resolution robust` / IntervalMDP.jl
  `Pessimistic,Maximize`); **opt** = nature cooperative (`Pmaxmax` / `cooperative` / `Optimistic`).
- `*` = the tool **cannot abstract**; it solved *IMPaCT's* exported IMDP.
- `**` = PRISM additionally required flooring `[0,hi]` interval edges to `[1e-9,hi]`
  (it rejects a 0 lower bound), so its ARCH values are an ε-perturbation.
- **N/A** = the property is outside the tool's scope. ✅ = agrees with the analytic reference.

---

## 1. Reference & example IMDP models (all four tools read the identical model)

| model | spec | IMPaCT | IntervalMDP.jl | PRISM | Storm | ref |
|---|---|---|---|---|---|---|
| chain_point | reach | 0.25 / 0.25 | 0.25 / 0.25 | 0.25 / 0.25 | 0.25 / 0.25 | 0.25 ✅ |
| safety_point | safety | 0.72 / 0.72 | 0.72 / 0.72 | 0.72 / 0.72 | 0.72 / 0.72 | 0.72 ✅ |
| choice_interval | reach | 0.40 / 0.50 | 0.40 / 0.50 | 0.40 / 0.50 | 0.40 / 0.50 | 0.40 / 0.50 ✅ |
| fork_interval | reach | 0.40 / 0.60 | 0.40 / 0.60 | 0.40 / 0.60 | 0.40 / 0.60 | 0.40 / 0.60 ✅ |
| multiObj_robotIMDP (207-state) | reach | 0.894663 / 1.0 | 0.894663† / 1.0 | 0.894662 / 1.0 | 0.894663 / 1.0 | agree ~1e-6 ✅ |
| buchi_reach | `GF acc` | 0.5 / 0.5 | **N/A** | **N/A** | **N/A** | 0.5 ✅ |
| buchi_routearound | `GF acc` | 0 / 1 | **N/A** | **N/A** | **N/A** | 0 / 1 ✅ |
| persist_leak | `FG safe` | 0.5 / 1.0 | **N/A** | **N/A** | **N/A** | 0.5 ✅ |
| patrol_cycle | `GF r0 ∧ GF r2` | 1.0 / 1.0 | **N/A** | **N/A** | **N/A** | 1.0 ✅ |

All four tools agree on every reachability/safety model. The four **ω-regular** models are
**IMPaCT-only** — Storm: *"LTL with intervals not implemented"*; PRISM: *"Computation not
implemented yet"*; IntervalMDP.jl: DFA/co-safe reachability only, no Büchi/persistence.
† IntervalMDP.jl's default 1e-6 threshold stops at 0.894662 (1.2e-6 low); at a tight
threshold it reaches 0.894663 = the others.

---

### 1b. New IMPaCT capabilities (four §10 upgrades, each literature-grounded + validated)

All reuse the O-maximization robust backup; each was validated against the analytic value
AND a peer tool on its native query. Papers are verified (DOIs/arXiv ids); details +
roadmap in [`ROADMAP_UPGRADES.md`](ROADMAP_UPGRADES.md).

| upgrade | model | IMPaCT (pess / opt) | oracle | ref | paper |
|---|---|---|---|---|---|
| **Reachability reward** | reward_chain | **3 / 5** | PRISM `Rmaxmin/Rmaxmax` 3 / 5 | 3 / 5 ✅ | Iyengar 2005 |
| **Discounted reward** (γ=0.9) | reward_chain | **2.8 / 4.2727** | IntervalMDP.jl 2.8 / 4.2727 | ✅ | Iyengar 2005 |
| **Long-run average** | lra_cycle (point) | **2 / 2** | Storm `R{"r"}max=?[LRA]` 2 | 2 ✅ | Ashok et al. CAV 2017 |
| **Long-run average** (interval) | lra_interval | **0 / 4** | analytic (no peer does interval LRA) | 0 / 4 ✅ | Chatterjee et al. IJCAI 2024 |
| **Arbitrary LTL** (det-Büchi) | ltl_demo (point) | `F a`/`GF a`/`XX a`/`G(a→Xa)` = 0.5/0.5/0.5/1 | Storm `Pmax=?[φ]` (all match) | ✅ | Sickert et al. CAV 2016 |
| **Arbitrary LTL** (interval) | ltl_interval | `GF a` **0.3 / 0.7** | analytic (no peer does interval LTL) | ✅ | Hahn et al. TACAS 2020 |
| **Multi-objective** (Pareto) | multi_demo (point) | vertices **(0.9,0.1),(0.2,0.8)** | Storm `multi(...)` 2 Pareto pts | ✅ | Etessami et al. TACAS 2007 |

Highlights: IMPaCT is the **only** tool that does reward + discounted (PRISM/Storm reach-only
on intervals; IntervalMDP.jl discounted-only), and the only tool doing **robust interval**
long-run-average and **robust interval LTL** at all (Storm/PRISM: "not implemented" on
intervals). Storm's interval reward ignores `--uncertainty-resolution` (returns only the
nature-max value). CLIs: `imdp_solve MODEL {reward|lra|ltlx|multi} LABEL [--discount g |
--weights w1,w2]`. Nondeterministic-automaton LTL (co-Büchi `FG`, Rabin) stays ISSUE-0016
(the built-in `ltl` fragment covers persistence).

## 2. ARCH-COMP 2025 benchmarks (IMPaCT abstracts; `*`peers solve the exported IMDP)

Infinite-horizon robust reach/safety on the abstracted IMDP — the one query all four tools
run identically. Value at init `pess`; solve time in seconds `(IMPaCT / IMDP.jl / PRISM / Storm)`.

| benchmark (dyn) | spec | IMPaCT | IntervalMDP.jl | PRISM `*,**` | Storm `*` | time (s) |
|---|---|---|---|---|---|---|
| AS (anaesthesia 3D) | reach | 0 | 0 | 0 | 0 | 0.4 / 0.5 / 0.5 / 0.05 |
| BA (building 4D) | safety | 0‡ | 1.9e-4 | 0 | 0 | –‡ / 4.9 / 0.3 / 0.7 |
| IC_reach (integrator 2D) | reach | 0 | 0 | ~0 | 0 | 0.1 / 0.4 / 0.2 / 0.05 |
| IC_safe (integrator 2D) | safety | 0 | 0 | 0 | 0 | 0.15 / 0.5 / 0.06 / 0.02 |
| PD_p1 (pkg-delivery 2D) | reach | 1.0 | 1.0 | 1.0 | 1.0 | 3.4 / 1.4 / 1.7 / 2.8 |
| PD_p3 (pkg-delivery 2D) | reach | 1.0 | 1.0 | 1.0 | 1.0 | 3.6 / 1.6 / 2.3 / 2.2 |
| VP (Van der Pol 2D) | reach | 0 | 0 | 0 | 0 | 1.8 / 0.55 / 1.4 / 0.04 |

**All four tools agree on every ARCH robust value** (optimistic-reach also agrees to high
precision, e.g. IC_reach opt ≈ 1.20e-26 in all four).

- **Native ARCH specs (finite-horizon for AS/BA/IC) — separately validated:** under the
  actual ARCH horizons, IMPaCT's per-cell robust value matches IntervalMDP.jl to ≤ **2.3e-5**
  (≤ 7e-7 for finite cases) and Storm's all-state aggregates, with the richer values (e.g.
  BA finite-safety init **0.4533** in all three tools; VP reach max 0.2297). The infinite-horizon
  reductions above give AS/IC/VP = 0 (nothing is robustly reached in the limit).
- **Scalability (abstraction phase, IMPaCT-only):** the large cases are an *abstraction*
  challenge the peers cannot even attempt. `AV_minimal` (7938 cells, 324M nnz) builds where
  the dense path OOMs; `LM` (6-D) is dense-infeasible (~96 GB/matrix) yet the sparse path
  builds+solves at 196 MB; `PR_minimal` (dense-OOM at 4 GB) completes at 3.86 GB. Details in
  [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md).
- ‡ IMPaCT's **BA optimistic-safety** OVI did not converge in 180 s (a large *safe* end
  component — ISSUE-0010); its robust (pess) BA-safety is 0, matching the others (verified at
  eps 1e-4). IMPaCT used eps 1e-4 on the large ARCH IMDPs; peers at their defaults.

**Abstraction is IMPaCT-only.** PRISM/Storm/IntervalMDP.jl have **N/A** for the whole
front end — they require a pre-built (I)MDP. IMPaCT builds it from the continuous dynamics
(nonlinear, via NLopt dense + sparse), fast on CPU/GPU (e.g. AS 0.11 s, PD 0.7 s sparse).
This is the reason an end-to-end comparison is impossible and §1–§2 compare only the *solve*.

---

## 3. Feature matrix (tools × capabilities)

✅ supported · ❌ not · **N/A** out of scope. Audited from source/docs, 2026-06-30.

| Capability | IMPaCT | IntervalMDP.jl 0.6 | PRISM 4.8.1 | Storm 1.13.0 |
|---|---|---|---|---|
| **Abstract continuous stochastic dynamics → IMDP** | ✅ nonlinear (NLopt + sparse) | ❌ | ❌ | ❌ |
| Noise models (for abstraction) | Gaussian, multivariate, uniform, triangular, Laplace, custom PDF | N/A | N/A | N/A |
| Interval / robust (I)MDP solving | ✅ | ✅ | ✅ (since v4.8) | ✅ |
| Robust reachability / safety | ✅ | ✅ | ✅ | ✅ |
| Robust bounded / finite-horizon | ✅ | ✅ | ✅ `F<=k` | ✅ step-bounded |
| **Robust ω-regular on IMDPs** (Büchi/persistence/LTL) | ✅ **only tool** | ❌ (DFA/co-safe only) | ❌ | ❌ |
| Expected reward / cost | ✅ reachability + discounted (robust, §1b) | ✅ discounted, exit-time | ✅ (+interval reach-reward) | ✅ |
| Long-run average / mean-payoff | ✅ robust interval (§1b) | ❌ | ✅ (non-interval) | ✅ (non-interval) |
| Multi-objective / Pareto | ✅ robust reachability (§1b) | ❌ | ✅ (non-interval) | ✅ (non-interval) |
| Controller / strategy synthesis | ✅ | ✅ | ✅ | ✅ |
| GPU acceleration | ✅ SYCL (any vendor) | ✅ CUDA | ❌ | ❌ |
| Exact / rational arithmetic | ❌ | ✅ | ✅ `-exact` | ✅ |
| Symbolic / BDD engine | ❌ | ❌ | ✅ MTBDD | ✅ dd (CUDD/Sylvan) |
| Full (non-robust) LTL / PCTL* / CSL | fragment + arbitrary det.-Büchi LTL via Spot (§1b) | ❌ | ✅ | ✅ (Spot) |
| **Robust LTL on IMDPs** (arbitrary, not just Büchi/persist) | ✅ det.-Büchi class, only tool | ❌ | ❌ | ❌ |
| Other model classes | POMDP, PTA, TA | orthogonal-IMDP, mixture-IMDP, IMC | DTMC/CTMC/MDP/PTA/POMDP/IDTMC | DTMC/CTMC/MDP/MA/POMDP, parametric, DFT, GSPN |
| Solver methods | II, VI, **OVI**, MEC, O-max | robust VI (O-max) | VI/GS/PI/II/topological/symbolic | VI/II/OVI/SVI/PI/LP/topological/symbolic |
| Input formats | C++ dynamics, `.imdp`, PRISM-subset, HDF5 | netCDF, PRISM-explicit, bmdp-tool | PRISM lang, explicit, PEPA | PRISM, JANI, DRN, explicit, GSPN, DFT |

**Present in a tool but not exercised above:**
- **IMPaCT** — the abstraction front end + 6 noise models, GPU/SYCL, POMDP/PTA/TA,
  PCTL/CTL/STL, and the II/VI/MEC solvers (only OVI was measured). **Four §10 gaps now
  closed** (§1b): robust expected reward (reachability + discounted), long-run average,
  arbitrary det-Büchi LTL, and multi-objective Pareto — see [`ROADMAP_UPGRADES.md`](ROADMAP_UPGRADES.md).
  Remaining: nondeterministic-LDBA LTL (ISSUE-0016) and reward-based multi-objective.
- **IntervalMDP.jl** — orthogonal & mixture IMDPs (its scalability feature), CUDA VI,
  discounted-reward / expected-exit-time, exact Rational arithmetic, DFA-product specs.
- **PRISM** — DTMC/CTMC/MDP/PTA/POMDP, full PCTL*/LTL/CSL (non-interval), rewards +
  steady-state, multi-objective/Pareto (non-interval), simulator, `-exact`, MTBDD.
- **Storm** — DTMC/CTMC/MDP/MA/POMDP, full LTL (Spot, non-interval), LRA, multi-objective
  (non-interval), parametric (`storm-pars`), DFT/GSPN, sound-VI family, JANI, symbolic dd.

---

## 4. Key findings & caveats (each verified)

- **PRISM 4.8.1 supports interval MDPs** (auto-detects `Type: IMDP`, solves
  `Pmaxmin/Pmaxmax=?[F ...]`); an earlier version of this doc wrongly said it did not.
  Two limits: it **rejects 0-lower-bound interval edges** (needs a fixed positive support
  graph — hence the `**` floor on abstracted IMDPs) and does **no robust ω-regular**.
- **Robust ω-regular on IMDPs is IMPaCT-only** — Storm/PRISM explicitly "not implemented",
  IntervalMDP.jl DFA/co-safe only. IMPaCT matches the analytic references exactly.
- **Cross-tool thresholds must match:** IntervalMDP.jl's default reachability threshold
  (1e-6) stops ~1.2e-6 short of the fixpoint on slowly-mixing chains; at a tight threshold
  it equals the others. IMPaCT's OVI returns the converged value *and* a certified upper
  bound at the same tolerance.
- **Optimistic safety is convention-dependent** across tools (how the cooperative sense
  resolves `[0,hi]` leak edges); the **robust** (guarantee) value is unambiguous and agrees —
  hence robust is the headline.
- **IMPaCT solver notes:** three infinite-horizon methods (`setIterationMethod`): interval
  iteration (sound bracket), value iteration (peer-style), and OVI (default; converges on
  nature-traps, ISSUE-0003). OVI can be slow on large recurrent models (ISSUE-0010).

## 5. Reproduce

```bash
# in the impact_work container (repo mounted at /work):
python3 benchmarks/crosstool/peers/sweep_compare.py all     # 4-tool sweep -> TSV
tools/imdp_solve MODEL.imdp reach|safety|buchi|persist|patrol LABEL --bound both
```
Converters + runners: [`benchmarks/crosstool/peers/`](benchmarks/crosstool/peers)
(`tra_to_imdp.py`, `imdp_to_drn.py`, `imdp_to_prism_interval.py`, `intervalmdp_runner.jl`,
`imdp_native_robot*.jl`, `build_storm.sh`). Raw sweep output:
[`sweep_all.tsv`](benchmarks/crosstool/sweep_all.tsv),
[`sweep_arch.tsv`](benchmarks/crosstool/sweep_arch.tsv). IMPaCT-side benchmark log:
[`benchmarks/RESULTS.md`](benchmarks/RESULTS.md). Design decisions & discrepancies:
[`issues/`](issues).
