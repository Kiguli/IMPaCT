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
| POMDP (finite-horizon), PTA, TA | ✅ | belief MDP; zone/digital-clocks | — |
| Controller/strategy synthesis; GPU (SYCL) | ✅ | — | — |

## B. Remaining gaps to close (the parity work-list)

| Feature | Peer(s) | Status | Planned approach + paper (to verify) | Robust/interval angle |
|---|---|---|---|---|
| **CTMC + CSL** time-bounded reachability `P(F<=t g)` | PRISM, Storm | ✅ | uniformisation + Fox-Glynn (Baier-Haverkort-Hermanns-Katoen TSE 2003 DOI 10.1109/TSE.2003.1205180; Fox-Glynn CACM 1988) — `src/ctmc.cpp`, `imdp_solve csl --time t`; validated == Storm == 1-e^{-t} | interval-CTMC (Katoen et al. CAV 2007) planned |
| **Nondeterministic-LDBA full LTL** (co-Büchi FG, Rabin) | Storm, PRISM | 🔶 (ISSUE-0016) | LTL→LDBA (Owl `ltl2ldba`), controller-resolved GFM product → existing robust Büchi | robust LTL on IMDPs (all of LTL) |
| **Exact / rational arithmetic** | PRISM, Storm, IntervalMDP.jl | ⬜ | templatise omax/VI on the numeric type; rational Gaussian elimination (Haddad-Monmege) | exact robust value, rational interval endpoints |
| **Symbolic / BDD (MTBDD) engine** | PRISM, Storm | ⬜ | MTBDD reachability (de Alfaro-Kwiatkowska-Norman-Parker-Segala TACAS2000) | symbolic robust IMDP |
| **Parametric model checking** | Storm (storm-pars) | 🔬 | solution functions / region check (Daws ICTAC2004; Junges et al.) | note: parametric ≈ interval/robust |
| **Markov automata (MA)** | Storm | 🔬 | CT + nondeterminism (Guck et al.; Eisentraut-Hermanns-Zhang) | robust MA |
| **Full PCTL* nesting / conditional prob** | PRISM, Storm | ⬜ | nested P-operators as state predicates over pctl.cpp | robust nested |
| **Statistical MC / simulation** (SPRT, CI) | PRISM, Storm | ⬜ | discrete-event simulator + hypothesis tests | — |
| **Orthogonal / mixture IMDPs** | IntervalMDP.jl | ⬜ | factored per-dimension O-max (Mathiesen-Haesaert-Laurenti arXiv:2411.11803) | scalability of robust abstraction |
| **DFT / GSPN front-ends** | Storm | ⬜ | Galileo/PNPRO → MA/MDP translation | robust DFT |

> Grounded plans (papers verified, algorithm, IMPaCT reuse, oracle) are being filled in
> per row from the parity-research pass; each row becomes a ROADMAP_UPGRADES.md section as it
> is implemented, then flips to ✅ here. Issues are logged in [`issues/`](issues) as they arise.
