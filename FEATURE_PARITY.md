# IMPaCT feature-parity tracker — goal: subsume PRISM + Storm + IntervalMDP.jl

**Goal.** Make IMPaCT a strict superset of the three peer probabilistic model checkers:
every checkable model class, specification, and analysis they offer should be available in
IMPaCT — and, because IMPaCT's niche is *robust/interval* MDPs and *abstraction from
continuous stochastic dynamics*, each feature should also have its **robust/interval**
generalisation (which the peers largely lack). Every feature is implemented from a **verified**
paper (DOI/arXiv checked — no hallucinated citations) and cross-validated against a peer tool
on a shared model plus an analytic value.

Legend: ✅ done+validated · 🔶 partial · ⬜ planned (grounded) · 🔬 research-hard.
Details of the shipped upgrades: [`ROADMAP_UPGRADES.md`](ROADMAP_UPGRADES.md); comparison
tables: [`TOOL_COMPARISONS.md`](TOOL_COMPARISONS.md).

## A. Already in IMPaCT (pre-existing or shipped this session)

| Feature | Status | Note / paper | Oracle |
|---|---|---|---|
| Interval/robust (I)MDP, DTMC, MDP (degenerate) | ✅ | O-maximization (Givan-Leach-Dean 2000; Lahijanian et al. TAC 2015) | PRISM/Storm/IntervalMDP.jl |
| **Abstraction of continuous nonlinear stochastic dynamics → IMDP** | ✅ (IMPaCT-only) | dense NLopt + sparse; 6 noise models | — (no peer does it) |
| Robust reachability / safety, bounded & unbounded | ✅ | interval iteration / OVI (Haddad-Monmege CAV2017; Hartmanns-Kaminski CAV2020) | all four |
| Robust ω-regular: Büchi / persistence / gen-Büchi | ✅ (IMPaCT-only on intervals) | de Alfaro 1997; ISSUE-0009 robust ECs | analytic |
| **Expected reward** (reachability + discounted) | ✅ | Iyengar 2005 robust DP | PRISM, IntervalMDP.jl |
| **Long-run average / mean-payoff** | ✅ (robust interval IMPaCT-only) | Ashok et al. CAV2017; Chatterjee et al. IJCAI2024 | Storm |
| **Steady-state (S operator)** | ✅ (robust interval IMPaCT-only) | = LRA of indicator (Baier-Katoen Ch.10) | Storm |
| **Arbitrary LTL** (deterministic-Büchi class) | ✅ (robust interval IMPaCT-only) | Spot; Sickert et al. CAV2016; Hahn et al. TACAS2020 | Storm |
| **Multi-objective / Pareto** (reachability) | ✅ | Etessami et al. TACAS2007; Forejt et al. TACAS2011 | Storm |
| **CTMC** model class + **CSL** time-bounded reachability | ✅ | uniformisation + Fox-Glynn (BHHK TSE2003; FG CACM1988) | Storm, analytic |
| **Robust interval-CTMC** CSL (uncertain rates) | ✅ (IMPaCT-only) | interval uniformisation + O-max (Katoen et al. CAV2007; Badings et al. CAV2022) | analytic |
| POMDP (finite-horizon), PTA, TA | ✅ | belief MDP; zone/digital-clocks | — |
| Controller/strategy synthesis; GPU (SYCL) | ✅ | — | — |

## B. Remaining gaps to close (the parity work-list)

| Feature | Peer(s) | Status | Planned approach + paper (to verify) | Robust/interval angle |
|---|---|---|---|---|
| **CTMC + CSL** time-bounded reachability `P(F<=t g)` | PRISM, Storm | ✅ | uniformisation + Fox-Glynn (Baier-Haverkort-Hermanns-Katoen TSE 2003 DOI 10.1109/TSE.2003.1205180; Fox-Glynn CACM 1988) — `src/ctmc.cpp`, `imdp_solve csl --time t`; validated == Storm == 1-e^{-t} | interval-CTMC (Katoen et al. CAV 2007) planned |
| **Nondeterministic-LDBA full LTL** (co-Büchi FG, Rabin) | Storm, PRISM | 🔶 (ISSUE-0016) | LTL→LDBA (Owl `ltl2ldba`), controller-resolved GFM product → existing robust Büchi | robust LTL on IMDPs (all of LTL) |
| **Exact / rational arithmetic** | PRISM, Storm, IntervalMDP.jl | ✅ | robust policy iteration over rationals (Iyengar 2005 §3.3; Puterman PI; O-max is division-free) + exact Gaussian elimination + exact P0 + fixpoint certificate — `src/exact.cpp`, `imdp_solve reach --exact`. Point chain 1/4 == PRISM/Storm `-exact`; **interval exact 2/5, 1/2, 3/5 certified — peers are point-only (verified: both error on intervals)**. Exposed infeasible rows in the robot model (ISSUE-0023) | exact robust interval value — IMPaCT-only |
| **Symbolic / BDD (MTBDD) engine** | PRISM, Storm | ⬜ | MTBDD reachability (de Alfaro-Kwiatkowska-Norman-Parker-Segala TACAS2000) | symbolic robust IMDP |
| **Parametric REGION verification** | Storm (storm-pars) | ✅ | Parameter Lifting → interval-MDP reachability (Quatmann-Dehnert-Jansen-Junges-Katoen ATVA2016, DOI 10.1007/978-3-319-46520-3_4) — `peers/param_lift.py` + existing solve; region [0.3,0.7]→[0.3,0.7] == Storm corners | parametric ≡ interval/robust (native) |
| Parametric **solution functions** | Storm | ⬜ | state elimination + multivariate rational-function arithmetic (Daws ICTAC2004) — needs a rational-function type | — |
| **Markov automata (MA)** | Storm | 🔬 | CT + nondeterminism (Guck et al.; Eisentraut-Hermanns-Zhang) | robust MA |
| Step-bounded reachability `P[F<=k]` / `a U<=k b` | PRISM, Storm | ✅ | finite-horizon exact DP (`pctl::boundedUntil`, exposed via `reach/until --horizon k`) == Storm | robust bounded |
| Expected time / steps to reach | PRISM, Storm | ✅ | = expected reward with unit state rewards (`reward` with all-1 reward) | robust |
| **Full PCTL\* nesting / conditional prob** | PRISM, Storm | ⬜ | nested P-operators as state predicates over pctl.cpp (satStates already returns the sub-formula state set) | robust nested |
| **Statistical MC / simulation** (SPRT, CI) | PRISM, Storm | ✅ | Monte-Carlo path sampling + Wilson CI + APMC/Chernoff (Hérault et al. VMCAI 2004) + Wald SPRT (Younes-Simmons CAV 2002; Wald 1945) — `src/smc.cpp`, `imdp_solve smc [--threshold p]`. Estimate 0.87471 (CI ∋ 0.875 analytic) == PRISM `-sim` 0.87479; SPRT correct both directions. Point chains (like PRISM's simulator) | — |
| **Orthogonal IMDPs** (factored) | IntervalMDP.jl | ✅ | recursive per-dimension O-max, dim-1-innermost convention (Mathiesen-Haesaert-Laurenti arXiv:2411.11803) — `src/odimdp.cpp`, `imdp_solve *.odimdp reach`; == IntervalMDP.jl to 10 digits both senses (ISSUE-0022) | native fit to IMPaCT's per-dim abstraction bounds |
| **Mixture IMDPs** | IntervalMDP.jl | ✅ | per-component factored O-max + O-max over interval mixture weights (Mathiesen et al. arXiv:2411.11803) — `mtran`/`mweight` in `.odimdp`. == IntervalMDP.jl MixtureIntervalMarkovDecisionProcess to 10 digits both senses (pess 0.4756756752 / opt 0.9619450314) | — |
| **DFT / GSPN front-ends** | Storm | ⬜ | Galileo/PNPRO → MA/MDP translation | robust DFT |

> Grounded plans (papers verified, algorithm, IMPaCT reuse, oracle) are being filled in
> per row from the parity-research pass; each row becomes a ROADMAP_UPGRADES.md section as it
> is implemented, then flips to ✅ here. Issues are logged in [`issues/`](issues) as they arise.

## C. SySCoRe comparison + benchmark implementation (planned, user 2026-07-02)

**SySCoRe** (van Huijgevoort, Schön, Soudjani, Haesaert, HSCC 2023, DOI
10.1145/3575870.3587123; v2.0: NAHS 58:101607, 2025) is a MATLAB toolbox computing
(ε,δ)-stochastic simulation relations — a *different sound lower bound* on satisfaction
than IMDP robust values (global (ε,δ) error vs per-transition intervals; supports
model-order reduction; no GPU; MATLAB-only, so comparison is against its PUBLISHED
numbers). Repo: github.com/BirgitVanHuijgevoort/SySCoRe-software (BSD-3).

Plan (definitions + published targets verified 2026-07-02, research log in session
scratchpad; ARCH-COMP25 report is the closest shared-benchmark venue):
- [ ] Implement SySCoRe's benchmarks in IMPaCT: CarPark1D, CarPark2D (running example;
      `!p2 U p1`, targets 0.60/0.52/0.42 at three init states, 7.94 s/27.5 MB),
      PackageDelivery scLTL `F(p1 & (!p2 U p3))` (peak 0.663, 11.0 s), Van der Pol
      variant (X=[-3,3]^2, no published scalar — compare maps/runtime vs 3191.6 s),
      BAS 7D→2D MOR (≥0.9035, 122 s/5.4 GB) and BAS-4D-KF (ARCH-25: peak 0.973,
      δ1=0.0054; IMPaCT already ships the 4D BAS = common ground).
- [ ] Comparison page (docs/comparison/impact-vs-syscore.md): semantics difference
      ((ε,δ)-coupling vs interval abstraction), feature matrix row (scLTL via ltl2ba;
      MOR; PWA nonlinear; Gaussian-only noise; no GPU; ≤2D direct gridding), and the
      published-numbers table with hardware caveats.
- [ ] Watch paper-vs-code discrepancies: VdP domain ±4 (paper) vs ±3 (code/ARCH25);
      CarPark regions ±4 vs ±3.25; lu=3 vs 7. Benchmark against the repo/ARCH-25 values.
