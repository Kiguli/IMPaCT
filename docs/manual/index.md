# Feature Manual (v2.0)

This manual documents every verification and synthesis feature of IMPaCT v2.0 beyond
the abstraction front end: what each feature computes (with precise robust semantics),
the literature algorithm it implements, a validated CLI example, and how it compares to
the peer tools **PRISM 4.8.1**, **Storm 1.13.0**, and **IntervalMDP.jl 0.6**.

Everything here runs through `tools/imdp_solve`, the dependency-free CLI over the
neutral `.imdp` exchange format (see {doc}`formats` and {doc}`cli`). Every numeric value
quoted in this manual was produced by a live run against the models in
`benchmarks/crosstool/models/` and cross-checked against the reference values in the
matching `*.ref.json` files; cross-tool methodology is in `TOOL_COMPARISONS.md` and
`FEATURE_PARITY.md` at the repository root.

## Conventions

- **pess** (pessimistic / robust): the controller maximises the objective while nature
  resolves every transition-probability interval *adversarially* — the value a
  controller can *guarantee* for all distributions within the intervals.
- **opt** (optimistic): nature resolves the intervals *cooperatively* — the best case.
- Solvers return a **sound bracket** `[lower, upper]` with gap `<= 2*eps` (default
  `eps = 1e-6`); finite-horizon dynamic programming is exact.
- The shared inner solve for all robust Bellman backups is **O-maximization**
  (Givan, Leach, Dean, *Bounded-parameter MDPs*, AIJ 2000; Lahijanian, Andersson,
  Belta, IEEE TAC 2015), `src/omaximization.h`.

## Feature overview

| Feature | Page | Algorithm | Key literature | Peer support |
|---|---|---|---|---|
| Reachability, safety, reach-avoid; bounded `P[F<=k]`; PCTL next/until | {doc}`specifications` | OVI / interval iteration over O-max | Hartmanns-Kaminski CAV 2020; Baier et al. CAV 2017; Haddad-Monmege TCS 2018; Hansson-Jonsson FAC 1994; Puggelli et al. CAV 2013 | all four tools (validated equal) |
| Expected reward (reachability, discounted) | {doc}`rewards` | robust dynamic programming | Iyengar Math. OR 2005; Nilim & El Ghaoui Oper. Res. 2005 | PRISM (reach), IntervalMDP.jl (discounted); Storm partial |
| Long-run average, steady-state | {doc}`lra-steadystate` | two-phase VI (MEC gain + cash-out) | Ashok et al. CAV 2017; Chatterjee et al. IJCAI 2024; Puterman 1994 Ch. 8 | Storm/PRISM point-only; **interval LRA is IMPaCT-only** |
| ω-regular (Büchi, persistence, patrol) and arbitrary LTL | {doc}`omega-ltl` | robust accepting-EC + Spot product | de Alfaro 1997; Sickert et al. CAV 2016; Hahn et al. TACAS 2020; Duret-Lutz et al. (Spot 2.0) | **IMPaCT-only on intervals** |
| Multi-objective / Pareto | {doc}`multiobjective` | weighted-sum scalarised VI | Etessami et al. TACAS 2007; Forejt et al. TACAS 2011 | PRISM/Storm point-only |
| CTMC + CSL time-bounded; interval CTMC | {doc}`ctmc-csl` | uniformisation + Fox-Glynn | Baier et al. IEEE TSE 2003; Fox-Glynn CACM 1988; Katoen et al. CAV 2007 | PRISM/Storm (point rates); **interval CTMC is IMPaCT-only** |
| Parametric region verification | {doc}`parametric` | parameter lifting → interval MDP | Quatmann et al. ATVA 2016 | Storm (storm-pars) |
| Orthogonal and mixture IMDPs | {doc}`odimdp` | recursive per-dimension O-max | Mathiesen, Haesaert, Laurenti arXiv:2411.11803 | IntervalMDP.jl (validated to 10 digits) |
| Exact rational solving with certificate | {doc}`exact` | robust policy iteration over rationals | Iyengar 2005 §3.3; Puterman 1994; Haddad-Monmege TCS 2018 | PRISM/Storm exact are **point-only**; interval exact is IMPaCT-only |
| Statistical model checking | {doc}`smc` | Monte-Carlo + Wilson CI + SPRT | Younes-Simmons CAV 2002; Hérault et al. VMCAI 2004; Wald 1945 | PRISM (`-sim`), Storm |
| Solver engines and method selection | {doc}`engines` | II / VI / OVI; MEC collapse | Baier et al. CAV 2017; Haddad-Monmege TCS 2018; Hartmanns-Kaminski CAV 2020 | — |
| File formats and converters | {doc}`formats` | `.imdp`, `.odimdp`, `.pimdp`, DRN/PRISM/tra converters | — | interchange with all peers |
| Complete CLI reference | {doc}`cli` | `imdp_solve` — every property and flag | — | — |

## Scope notes

- The **abstraction** of continuous stochastic dynamics into an IMDP (the core of the
  IMPaCT tool paper, QEST+FORMATS 2024) is documented in the User Guide; this manual
  covers the *solving* layer that the abstraction feeds (`IMDP::exportIMDP` bridges the
  two, see {doc}`formats`).
- Known limitations are tracked as numbered issues in `issues/INDEX.md` and are cited
  inline on each page (e.g. ISSUE-0003, ISSUE-0010, ISSUE-0016, ISSUE-0022, ISSUE-0023).
