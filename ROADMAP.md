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
  sound robust reachability with selectable methods — OVI (default) + MEC-collapse
  interval iteration (`solve`); **safety** = 1 - min-reach.
- **Specifications**: LTLf parser + finite-trace evaluator and LTLf→DFA via semantic
  canonicalization (`ltl`); IMDP×DFA product for co-safe LTL (`product`); **Büchi
  ω-regular** via accepting-MEC reachability (`omega`).
- **Sparse abstraction** (`abstraction`): affine n-D and nonlinear (interval
  arithmetic) diagonal-Gaussian systems; O(nnz) memory.
- **Verification**: a doctest suite (56 cases, ~600k assertions) of differential /
  brute-force oracles, plus end-to-end Monte-Carlo validators for reach, safety,
  co-safe LTL, and nonlinear (VP) — see `benchmarks/`.
- **Issues**: resolved 0001(superseded),0002,0003,0005,0007,0008; in-progress 0006
  (sparse landed for the v2 path); open 0004 (Spot back-end optional), 0009 (robust
  accepting ECs), 0010 (OVI perf on recurrent IMDPs).
- **Not yet**: robust accepting ECs (ISSUE-0009, the CAV-core robust EC), wiring the
  sparse path through the full ARCH-COMP suite, ω-automata for general LTL (Büchi/
  Streett/parity back-ends), and Phase 4 (web app).

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

## Guardrails (apply to every addition)
- Verified references only; differential/brute-force oracles for every algorithm.
- No machine-dependent correctness crutches (e.g., `maxStates` is a safety valve, not a
  dedup mechanism — see ISSUE-0005).
- Track soundness limitations as issues with minimal counterexamples (ISSUE-0003).
