# IMPaCT v2.0 — Living Research Document

> **Dual purpose.** This file is (A) an append-only development & decision log
> updated every working session, and (B) the growing draft of the CAV paper.
> Section B distils what section A records. Every citation must resolve to a
> **verified, non-hallucinated** entry in [`References.bib`](References.bib)
> (verification is a hard project rule). Algorithms are tied to immutable tests
> in [`../tests/TEST_PLAN.md`](../tests/TEST_PLAN.md).

Working title: *IMPaCT 2.0: Sound and Scalable ω-Regular Controller Synthesis
for Continuous-State Stochastic Systems via Interval MDPs.* (provisional)

---

## A. Development & decision log (append-only)

Format per entry: `date — feature | decision + rationale | algorithm | refs | files | test/benchmark status`.

### 2026-06-25
- **Repo & branch.** Cloned `Kiguli/IMPaCT`, created branch `IMPaCT-v2.0`.
  Rationale: isolate v2.0 work; `main` stays the v1 artifact.
- **TDD-first.** Designed the full immutable test suite before any
  implementation (`tests/`), framework = doctest (C++ unit) + pytest
  (integration/oracle/golden). Decision: internal-algorithm unit tests are pure
  std C++ (no SYCL/HDF5/GLPK) so the new components (`omax`, `graph`, `solve`)
  are testable in isolation, then wired into `IMDP.cpp`. Oracles are derived
  independently (hand + brute-force vertex enumeration + numpy MC solves) so
  expected values are not reverse-engineered from the code.
  Status: 23 C++ contract cases RED ("not implemented"); Python oracle
  consistency GREEN.
- **Interface contracts fixed** in `tests/contracts/contracts.h`:
  `impact::omax::optimize` (O-maximization), `impact::graph::{sccs,mecs}`,
  `impact::solve::maxReach{Pessimistic,Optimistic}` (sound interval iteration),
  `impact::ltl::{compileFinite,acceptsFinite}` (LTLf front-end).
- **References policy.** Only papers confirmed real (DOI/arXiv/DBLP resolved)
  may enter `References.bib`. Verification pass DONE: all 42 sources confirmed
  real (44 BibTeX entries), none unconfirmed. Key corrections caught:
  *Asadi et al.* is **AAAI 2026** (not ICML 2025), arXiv:2505.04539;
  arXiv:2207.13660 is **Weininger-Meggendorfer-Křetínský, BMDP ω-regular, CDC
  2019** (distinct from the Dutreix-Coogan interval-MC line); *IntervalMDP.jl* =
  Mathiesen-Lahijanian-Laurenti; *Owl* = Křetínský-Meggendorfer-Sickert;
  *Seminator 2* = CAV 2020; *Cauchi et al. HSCC 2019* 5th author = Kwiatkowska.
  Two notes: IMPaCT's proceedings DOI unconfirmed (cite arXiv:2401.03555);
  Asadi AAAI-26 volume is forward-dated (preprint is safe to cite).

- **Phase 1a — O-maximization VERIFIED.** Confirmed IMPaCT's existing "sorted"
  synthesis already implements the O-maximization sort-and-assign (GPU_synthesis.cpp
  inner loop: start at lower bounds, allocate residual in value order — avoid set
  (value 0) → states ascending → target (value 1) — capping at interval widths).
  Extracted the core into `src/omaximization.{h,cpp}` and validated against the
  contract: 7 cases / 8465 assertions green, incl. a 500-instance randomized
  differential vs independent brute-force vertex enumeration. Algorithm: Givan-
  Leach-Dean (AIJ 2000); Lahijanian et al. (IEEE TAC 2015).
  Also noted: the sorted loop stops on a tolerance and only *warns* about absorbing
  states (GPU_synthesis.cpp ~L186) — motivates the Phase 1c sound interval iteration.
  Next: refactor IMDP/GPU_synthesis to call the shared verified routine (after goldens).
- **Phase 1b — SCC + MEC implemented.** `src/graph_utils.{h,cpp}`: iterative
  Tarjan SCC + maximal-end-component decomposition. MEC uses the correct
  recompute-SCC-under-candidate's-own-staying-actions fixpoint (a naive "every
  state has a staying action" acceptance is unsound under leaking actions —
  verified by constructing a counterexample before coding). 6 cases green.
  Refs: Tarjan (1972); de Alfaro (1997); Chatterjee-Henzinger (JACM 2014).
  Status after 1a+1b: 13/23 contract cases green; remaining red = Phase 1c
  interval iteration (5) + Phase 2 LTLf (5).

<!-- add new dated entries above this line -->

---

## B. Paper draft (grows into the CAV submission)

### Abstract
*TODO.* One paragraph: IMPaCT v2.0 lifts abstraction-based stochastic controller
synthesis from reach-avoid to full ω-regular (LTL) specifications, soundly and at
GPU scale; first tool to solve ω-regular benchmarks in the ARCH-COMP Stochastic
Models category.

### 1. Introduction & contributions
Contributions (kept in sync with shipped features):
1. A sound, GPU-parallel robust interval-iteration engine for IMDPs replacing the
   per-state LP with closed-form O-maximization. *(Phase 1)*
2. Co-safe-LTL/LTLf synthesis via automaton-product reachability — solving
   Package Delivery at full spec (no spec reduction). *(Phase 2)*
3. **Full ω-regular synthesis via robust accepting end components** — a
   first-class robust-MEC primitive + a quantitative ω-regular IMDP algorithm.
   *(Phase 3, core)*
4. **An abstraction-soundness theorem** for ω-regular specs of continuous-state
   stochastic systems. *(Phase 3, theory)*
5. A hostable web interface (spec builder, algorithm/parallelism selection).
   *(Phase 4)*
6. Evaluation across the ARCH-COMP Stochastic Models suite, incl. new ω-regular
   benchmark variants no current stochastic tool can solve.

### 2. Preliminaries
*TODO:* IMDPs / abstraction of continuous-state stochastic systems; ω-automata
(LDBA); end components; robust (2.5-player) semantics.

### 3. Sound robust interval iteration + O-maximization  *(Phase 1)*
*TODO.* Tests: `tests/unit/test_omaximization.cpp`, `test_interval_iteration.cpp`,
`test_graph.cpp`. Refs: Givan-Leach-Dean; Haddad-Monmege; Baier et al.

### 4. Co-safe LTL / LTLf via reachability  *(Phase 2)*
*TODO.* Tests: `test_ltl_dfa.cpp` + product/PD integration. Refs: Kupferman-Vardi;
De Giacomo-Vardi; Lacerda-Parker-Hawes.

### 5. Full LTL: robust accepting end components & ω-regular synthesis  *(Phase 3, core)*
*TODO.* Refs: Sickert et al. (LDBA); Asadi et al. (robust ω-regular); Dutreix-Coogan.

### 6. Abstraction-soundness theorem for ω-regular specs  *(Phase 3, theory)*
*TODO* — the genuinely novel theorem; differentiate from Dutreix-Coogan
(class-restricted, qualitative).

### 7. Tool architecture, parallelism, and web interface  *(Phase 4)*
*TODO.* AdaptiveCpp/SYCL backend; Vue+Flask hostable app modelled on TRUST.

### 8. Experimental evaluation
*TODO* — tables assembled from `benchmarks/` as each capability lands (see
`benchmarks/RESULTS.md`). ARCH-COMP Stochastic Models coverage + ω-regular
variants.

### 9. Related work · ### 10. Conclusion
*TODO.*
