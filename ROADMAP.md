# IMPaCT v2.0 — Roadmap & Design Principles

## Design principle: a *toolbox* of literature-backed, selectable approaches
Where the literature offers more than one correct approach to a sub-problem, IMPaCT
should expose them as **user-selectable options**, not bake in a single choice. Every
option must be (a) grounded in a **verified** published algorithm (real DOI/arXiv/DBLP
in [`paper/References.bib`](paper/References.bib)) and (b) cross-validated by a
differential oracle in the test suite. This applies retroactively to Phases 1–2 and
forward to Phase 3+.

## Current state (implemented & verified on branch IMPaCT-v2.0)
- **Solver core** (`src/`): O-maximization (`omaximization`), SCC+MEC (`graph_utils`),
  sound robust reachability — OVI (default) + MEC-collapse (`solve`); **safety**.
- **ω-regular** (`omega`): exact robust accepting ECs / a.s.-Büchi (ISSUE-0009,
  oracle-validated), generalized Büchi (patrol), persistence/co-Büchi.
- **Specifications/logics**: LTLf→DFA + IMDP×DFA product (co-safe, `ltl`/`product`);
  **PCTL/CTL** (`pctl`); **bounded STL** (`stl`); **LTL fragment dispatcher**
  (`ltlspec`) routing F/G/U/X/GF/FG/patrol to the verified solvers.
- **Abstraction** (`abstraction`): sparse affine n-D + nonlinear (interval-arithmetic)
  diagonal-Gaussian, O(nnz); **pluggable noise** (Gaussian/uniform/triangular/Laplace);
  bounded-disturbance kernel → non-trivial robust continuous ω-regular (ISSUE-0011).
- **Other models**: **timed automata** (DBM zones + zone-graph reachability, `dbm`/`ta`);
  **probabilistic timed automata** (`pta`, zone Pmax + digital-clocks Pmin); **POMDPs**
  (`pomdp`, exact finite-horizon belief-state reachability).
- **Tooling**: neutral `.imdp` format + PRISM-language input + `imdp_solve` CLI;
  cross-tool reference benchmarks (`benchmarks/crosstool`); **web app** (`webapp`,
  small-model IMDP + satisfaction-probability visualization).
- **Verification**: doctest suite **128 cases / ~643k assertions**, all differential/
  brute-force-oracle-backed; Monte-Carlo end-to-end validators in `benchmarks/`.
  Every algorithm linked to a Crossref-verified paper in `paper/References.bib`.
- **Issues**: resolved 0001–0003,0005,0007–0009,0011,0014; in-progress 0006,0010,
  0012,0013; open/transfer 0004, 0015 (robust interval-POMDP, research),
  0016 (full-LTL→LDBA, external Spot/Owl). See the big-machine queue below.
- **Needs the big machine** (compile/external): v1 LP→OMax compile-verify + GLPK
  removal (0012), finite-safe Sorted fix verification (0013), live peer comparisons,
  robust interval-POMDP (0015), arbitrary-LTL→LDBA (0016).

## Near-term (current plan)
- **Phase 1** sound robust interval iteration — *done* (MEC-collapse variant). Add
  alternative solvers as options: **interval iteration** (Haddad-Monmege/Baier),
  **sound value iteration** (Quatmann-Katoen), **optimistic value iteration**
  (Hartmanns-Kaminski). Inner robust solve: O-maximization (done) and an LP variant.
- **Phase 2** co-safe LTL via DFA product — *done*. Options to add: external automata
  back-ends (Spot/Owl/Rabinizer/Lydia) alongside the self-contained DFA.
- **Phase 3** robust ω-regular synthesis — selectable pipelines (see below).
- **Phase 4** hostable web app (last).

## Phase 3 — selectable ω-regular pipelines (all literature-backed)
1. **Rabin-product + permanent-winning-components** (Dutreix-Coogan, CDC 2018 +
   NAHS 2022) — the only *published* end-to-end robust-IMDP ω-regular synthesis;
   quantitative lower+upper bounds; leaky-state pruning extends `graph::mecs` and
   *fixes ISSUE-0003*; reuses `solve::maxReach`.
2. **BMDP-as-stochastic-game + interval iteration** (Weininger-Meggendorfer-Křetínský,
   CDC 2019) — cleanest semantic cure for the non-unique-fixpoint (game value = robust
   value, positional strategies); anchors the ISSUE-0003 fix theoretically.
3. **Qualitative force-oracle mode** (Asadi et al., AAAI 2026) — almost-sure/positive
   winning via `force_A`/`force_E` attractors; fast, most general rectangular uncertainty.
4. *(research/new)* LDBA/GFM-product + robust accepting EC — established for MDPs
   (Sickert CAV 2016; Hahn TACAS 2020) but a NEW theorem for robust IMDPs.

## Later phases — breadth of approaches (per user direction)
- **Alternative Phase-1/2 implementations** added as selectable options (the solver
  and inner-solve variants above; external automata back-ends).
- **Automata types**: beyond DFA/Rabin — **Büchi, generalized Büchi, Streett, parity,
  LDBA/limit-deterministic, good-for-MDP** — as interchangeable automaton back-ends,
  with the product/EC analysis specialized per acceptance condition.

## User-requested extensions (added 2026-06-26)
- **(a) More noise types.** Support as many disturbance models as possible. The
  per-cell transition kernel is parameterized by the noise model; each needs the 1-D
  mass-in-box and its min/max over the source-cell mean range. Done: Gaussian
  (`transitionInterval1D`), bounded uniform (`transitionInterval1DUniform`,
  enables non-trivial robust ω-regular — ISSUE-0011). Next: truncated Gaussian,
  triangular, Laplace, and a unified `NoiseModel` interface so builders accept any.
- **(b) Audit + clean up the original IMPaCT (v1) code.** Review `src/IMDP.cpp` and
  `src/GPU_synthesis.cpp` for correctness and sloppy code; file issues for anything
  found. Unify the two inner-solve approaches: the **GLPK LP** and the **O-maximization
  sort-and-assign** compute the same robust extremal value, but OMax is always faster
  (O(n log n) vs an LP solve) — make **OMax the single default** and keep the LP-named
  entry points as **backwards-compatible aliases** that call the faster OMax routine.
  (OMax already verified vs brute-force vertex enumeration in `test_omaximization`.)
- **(web app) Visualize small models.** For models under a configurable state cap,
  render the IMDP (states/edges) and overlay the synthesized **satisfaction
  probabilities** (per-state value heat map). Part of the Phase-4 web app (last).
- **Timed automata** — stretch, AFTER STL/CTL: real-time specs / timed-automaton
  models (region/zone abstraction) on top of the stochastic core.

## Stretch / long-term targets (per user direction)
- **Sparse transition representation** (CSR) — required to scale (see ISSUE-0006);
  prerequisite for large products.
- **Other robust-MDP models**: s-rectangular and general (convex) robust MDPs;
  **POMDPs** / partially observable robust models.
- **PRISM / Storm feature parity and beyond**: match the modelling + property coverage
  of PRISM and Storm, then add capabilities beyond them (the abstraction-from-continuous
  + GPU-parallel niche is IMPaCT's differentiator).
- **Logics beyond LTL**: **STL** (signal temporal logic) and **CTL** (and CTL*/PCTL)
  in addition to LTL/LTLf — selectable specification front-ends.

## Big-machine verification queue (needs the SYCL/Armadillo/GLPK toolchain — Windows/Linux)
These changes are made and reasoned but NOT compile-verified in the current
environment; verify on a machine with AdaptiveCpp + Armadillo + GLPK + HDF5:
1. **ISSUE-0012** — v1 inner-solve unification (public controllers → O-maximization
   `*Sorted`). Compile-verify the examples still run and match the old LP outputs;
   then DELETE the dead GLPK LP implementations and drop the GLPK dependency.
2. **ISSUE-0013** — finite-horizon safety `Sorted` UPPER-bound kernel fixes
   (over-count; uninitialized `temp1`; missing base term; missing avoid-residual
   block). Verify `finiteHorizonSafeController` upper bounds against the LP path,
   the v2 `solve::maxSafety*`, and the cross-tool `.imdp` safety references, and that
   they bracket a Monte-Carlo estimate. If verification fails, hold back ONLY the
   finite-safety redirect (keep it on the LP path) until fixed.
3. **Live peer comparisons** — install IntervalMDP.jl / PRISM / Storm and run the
   `benchmarks/crosstool` models side-by-side, dropping measured values into the
   `.ref.json` `checks` (they must match the analytic references already there).

## Guardrails (apply to every addition)
- Verified references only; differential/brute-force oracles for every algorithm.
- No machine-dependent correctness crutches (e.g., `maxStates` is a safety valve, not a
  dedup mechanism — see ISSUE-0005).
- Track soundness limitations as issues with minimal counterexamples (ISSUE-0003).
