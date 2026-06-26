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

- **Issue tracking** set up locally in `issues/` (file-per-issue + INDEX +
  strict literature-counterexample protocol — candidate→confirmed only after
  exact citation + minimal counterexample + independent verification +
  adversarial review). Filed ISSUE-0001 (v1 sorted synthesis tolerance-stopping
  soundness gap → fixed by Phase 1c) and ISSUE-0002 (MEC naive-rule pitfall,
  classified `naive-strawman` pending the audit — NOT a literature counterexample).
- **MEC confidence**: added a definition-based brute-force MEC oracle and a
  randomized differential test (1500 random MDPs) + the explicit leaking-action
  case; `mecs()` matches the oracle everywhere.
- **MEC audit CONFIRMED (`mec-correctness-audit` workflow; primary sources read
  directly).** The implemented algorithm is the standard MEC decomposition:
  **Baier-Katoen, *Principles of Model Checking*, Algorithm 47 (pp. 878–879)**;
  de Alfaro 1997 Sec. 3.3; Chatterjee-Henzinger JACM 2014 Sec. 3 — recompute SCCs
  under the candidate's own staying actions and accept only single-SCC fixpoints.
  Honest correction: the earlier "counterexample to the literature" wording was
  OVERSTATED — it refutes only a naive one-pass strawman. Baier-Katoen p. 877
  explicitly notes action removal can break strong connectivity ("We therefore have
  to repeat the whole procedure"), i.e. the literature anticipates exactly this trap;
  the refinement loop exists to avoid it. ISSUE-0002 resolved as `naive-strawman`.
  Confidence basis: matches a proven published algorithm AND passes a definition-based
  brute-force differential (1500 random MDPs). (One adversarial-verdict sub-agent hit a
  StructuredOutput retry-cap tooling failure — not substantive; the direct source reads
  are conclusive.)

- **Phase 1c — sound robust interval iteration.** `src/solve.{h,cpp}`:
  `maxReach{Pessimistic,Optimistic}` = robust Bellman (omax, Sense Min/Max) +
  **MEC collapse** (graph::mecs) for a unique fixpoint + two-sided interval
  iteration (lower↑ from 0, upper↓ from 1, stop at gap ≤ 2·eps). Refs:
  Haddad-Monmege (TCS 2018); Baier et al. (CAV 2017); Givan-Leach-Dean (omax).
  All 5 interval-iteration contracts green + Model 6 (lossy-exit EC, *forces* MEC
  collapse). Confidence: a `VI-from-below` differential oracle (independent of
  collapse) over ~900 random MDPs — point (both senses) + optimistic intervals.
- **ISSUE-0003 found by the differential test (genuine, important).** Pessimistic
  (robust) interval iteration does NOT converge when nature can confine the play in
  a state via a lo=0 leaving edge (minimal counterexample: `0:a→{0:[.5,1],1:[0,.5]}`,
  `1→target` ⇒ V*(0)=0 but upper sticks at 1; non-unique fixpoint from a
  nature-confinable EC the support-graph MEC misses). The upper bound stays *sound*
  (1 ≥ 0) but the gap doesn't close. Classified `our-bug`/limitation — NOT a
  literature counterexample (Haddad-Monmege handle it). Scope documented in
  `src/solve.h`; counterexample pinned as a skipped contract. Proper robust-EC fix
  is deferred to / merged with Phase 3 (robust accepting ECs use the same machinery).
  Suite: 22/27 cases green, 31618 assertions, 1 skipped (ISSUE-0003), 5 red (Phase 2 LTLf).

- **Phase 2 (part 1) — LTLf front-end (parse + finite-trace membership).**
  `src/ltl.{h,cpp}`: self-contained tokenizer + recursive-descent parser
  (`! & | -> <-> X F G U R`, true/false, parens) + LTLf finite-trace semantics
  evaluator (De Giacomo-Vardi IJCAI 2013/2015). No external dependency. All 5
  LTLf membership contracts green. Decision: deliberately membership-first so the
  semantics are pinned and testable before building a DFA; this evaluator will
  also serve as the differential ORACLE for the DFA construction.
  **Remaining Phase 2 work (tracked, ISSUE-0004):** LTLf→DFA construction + the
  IMDP×DFA product + co-safe reachability (to solve Package Delivery at full spec).
  Plan: self-contained derivative/progression-based DFA (no Spot dependency),
  validated against this evaluator; Spot remains a future performance option.
  Suite after Phase 2 (part 1): 27/27 cases green (1 skipped = ISSUE-0003), 31642 assertions.

- **Phase 2 (part 2) — LTLf→DFA.** `ltl::toDFA` builds the DFA by formula
  progression (Brzozowski-style derivatives; strong-X via a Tail marker). Found &
  fixed a real bug (ISSUE-0005): syntactic normalization lacks Boolean absorption,
  so `(G a) U (G a)`-style formulas exploded (caught by the randomized differential
  hanging; isolated via a generator-replay driver). Fix: **semantic
  canonicalization** — states keyed by their truth table over the formula's anchor
  subformulas (der is a Boolean homomorphism over anchors ⇒ finite minimal DFA).
  Now 300 random formulas build 3–6-state DFAs instantly; `maxStates` is purely a
  safety valve (configurable; never the dedup mechanism — per user feedback).
  Validated by `dfaAccepts == acceptsFinite` differential on random formulas×traces.
  Suite: 29/29 cases green, 37684 assertions, 1 skipped (ISSUE-0003).

- **Phase 2 (part 3) — IMDP×DFA product + co-safe reachability.** `src/product.{h,cpp}`:
  builds the product (state = s·nQ+q), DFA reads the label of each entered state,
  accepting DFA states become reachability targets (good-prefix co-safe semantics),
  then reuses `solve::maxReach*`. Correctness gate: for `F goal`, product reachability
  == plain `solve::maxReach` to goal-labelled states — verified differentially on 200
  random MDPs; plus a sequential `F(pickup & F deliver)` example = 0.5. The full
  co-safe pipeline (LTLf→DFA→product→robust reachability) now works end-to-end on
  abstract IMDPs. Remaining for the real PD benchmark: drive it from the actual
  continuous→IMDP abstraction (tool-integration step). Suite: 31/31 cases, 37886 assertions.

- **Docker baseline / ISSUE-0006.** v1 image rebuilt; ran the smallest sorted case
  (2D robot, CPU, `--memory=4g`). It **OOM-killed**: the dense (state×input)×state
  transition matrix is 698103×1583 ≈ **8.84 GB**. Confirms the dense-matrix
  scalability limit — goldens need a tiny model or sparse storage (ISSUE-0006;
  feeds the planned sparse-representation work). No controller golden produced.
- **ISSUE-0003 RESOLVED + toolbox principle.** Added **Optimistic Value Iteration**
  (Hartmanns-Kaminski, CAV 2020) as `solve::Method::OptimisticVI` (now the default):
  VI-from-below lower bound + a verified inductive upper bound (F(U)≤U ⇒ V*≤U), no
  EC handling, so it converges on nature-confinable ECs. The nature-trap test now
  converges; the randomized differential covers BOTH senses on point AND interval
  MDPs. `MECCollapse` kept as a selectable faster option (tested on point/controller
  ECs). Suite: 33/33 cases, 45326 assertions, **0 skipped**.
- **Roadmap captured (`ROADMAP.md`)** per user direction: toolbox of literature-backed
  selectable approaches (solver variants, automata back-ends, then automata TYPES —
  Büchi/Streett/parity — in a later phase), and stretch targets: sparse storage, other
  robust-MDP models incl. POMDPs, PRISM/Storm parity and beyond, and STL/CTL in
  addition to LTL. Phase 3 pipelines chosen from verified literature (Dutreix-Coogan
  Rabin+permanent-WC primary; Weininger game-reduction; Asadi qualitative).

- **Sparse abstraction (priority: memory / larger benchmarks, ISSUE-0006).**
  `src/abstraction.{h,cpp}`: builds the sparse `solve::IMDPModel` directly for affine,
  axis-decoupled, diagonal-Gaussian systems (storing only kernel-window successors) —
  same sparse model as IntervalMDP.jl, so memory is O(nnz). Verified: the interval
  kernel `transitionInterval1D` vs brute-force mean sampling; box == product of 1-D;
  lossless pruning (sparse synthesis == dense-window synthesis); target-cell = 1.
  Scaling (`benchmarks/sparse_scaling.cpp`, domain grown at fixed step/noise):
  **125,000 states ≈ 255 MB + 0.42 s** vs a dense (min+max) matrix ≈ **250 GB** — the
  large runs that v1's dense path OOMs on (8.84 GB on the *smallest* 2D-robot case).
  Honest caveat recorded: O(N) holds for fixed kernel-in-cells (grow domain); refining
  eta at fixed sigma is only a constant-factor saving. Suite: 38/38, 54375 assertions.
  Next: n-D/coupled abstraction (mixed-monotone/NLopt bounds), sparse products, and
  driving ARCH benchmarks through the sparse pipeline. Branch pushed to origin.

- **Sparse abstraction extended to n-D (affine, diagonal Gaussian).**
  `abstraction::buildSparseReachND`: per-dimension mean RANGE via interval arithmetic
  on the affine map (A,B,c) — exact for axis-decoupled, and for COUPLED systems the
  product-of-1-D box bound is a SOUND over-approximation (true probability always lies
  within [∏lo_i, ∏hi_i]); tight when decoupled. Sparse via Cartesian kernel-window
  enumeration. Verified: dim=1 reproduces the verified 1-D builder; 2-D decoupled
  pruning lossless (sparse==dense); 2-D coupled affine sound (value in [lower,upper],
  target cells = 1); 2-D O(N) sparsity (nnz/cell converges to a constant as the domain
  grows — boundary effect, not unbounded). Suite: 42/42, 54971 assertions.
  Next: nonlinear dynamics (mixed-monotone/NLopt mean bounds) + drive an ARCH-COMP
  case through the sparse pipeline and verify the value end-to-end.

- **End-to-end verification vs the real continuous system (Monte-Carlo).**
  `benchmarks/validate_montecarlo.cpp`: build sparse IMDP → synthesize robust
  controller → extract policy (argmax of `omax` over `val.lower`) → simulate the
  continuous closed loop → check empirical reach vs synthesized bounds. This caught
  **ISSUE-0007** (sink interval too loose: `1 - sumLo` ≈ 1 let nature drain all mass
  to the value-0 sink → reach ≈ 0 vs true ≈ 1). Fixed: bound outside-grid prob
  tightly as the grid-box complement `[1-gridHi, 1-gridLo]` (both builders). Notable:
  internal differential tests missed it (dense & sparse shared the error) — the
  closed-loop check found it. After the fix: empirical reach ≥ robust lower bound for
  all start states, in a saturated case (≈1) AND a discriminating high-noise case
  (value≈0.32, empirical≈0.36). Interpretation recorded: a robust controller's
  guarantee is empirical ≥ lower; benign noise may exceed the (worst-case) upper.
  Also found a Monte-Carlo gotcha: validating an infinite-horizon reach needs a long
  sim horizon (truncation gave 0.73 vs true 1.0). Suite still 42/42, 54971 assertions.

- **Nonlinear abstraction + a real ARCH nonlinear benchmark (Van der Pol).**
  Generalized the abstraction (`buildSparseReachGeneral` + a `MeanBoundFn`): the
  per-dimension mean enclosure is pluggable, computed for nonlinear systems by
  **interval arithmetic** (`abstraction::Ival`, `isquare`). The affine n-D builder now
  delegates to it. VP (`x0'=x0+0.1x1`, `x1'=x1+0.1(-x0+(1-x0)²x1)`, no input) runs
  end-to-end through the sparse pipeline. Verified: the VP interval mean bound is
  SOUND (vs brute-force, 600k+ assertions); the VP interval-MC build is sound
  (pess ≤ opt, target = 1); and **6/6 start states have empirical reach (real
  nonlinear stochastic sim) within [lower, upper]** (`benchmarks/validate_vp.cpp`).
- **ISSUE-0008 (found by VP MC, fixed).** Grid-unaligned targets were double-counted
  (separate target-region aggregate + partial-overlap cells) → lower bound too high
  (0.354 vs true 0.236). Fix: drop the aggregate; route window target cells to the
  absorbing TARGET (disjoint cells, no double count) in both builders. After the fix
  VP is 6/6; suite still 44/44, 603273 assertions. (Second soundness bug the
  closed-loop validation caught that internal differential tests missed.)

- **Phase 3 (step 2a) — Büchi ω-regular via accepting-MEC reachability.**
  `src/omega.{h,cpp}`: `maxBuchi{Optimistic,Pessimistic}` reduce "visit `accepting`
  infinitely often" to reachability of accepting MECs (de Alfaro 1997; Baier-Katoen;
  the LDBA/GFM accepting-frontier route, Sickert CAV 2016 / Hahn TACAS 2020). Reuses
  graph::mecs + solve. Verified (test_omega.cpp): reduction to reachability,
  GF(true)=1, transient→0, recurrence-in-EC=0.5, pess==opt on point MDPs. ISSUE-0009
  filed: robust (pessimistic) accepting ECs use the optimistic support-MEC structure
  for now (sound for point/optimistic; robust path = Dutreix-Coogan/Weininger/Asadi).
- **HEADLINE INTEGRATION — co-safe LTL over a continuous system (Package Delivery).**
  `benchmarks/validate_cosafe.cpp` ties everything together: sparse abstraction (full
  IMDP) → LTLf→DFA → IMDP×DFA product → robust solver → policy → Monte-Carlo. Spec
  `F(pickup & F deliver)` on a 2-D robot; 4/4 start states have empirical co-safe
  satisfaction (continuous closed-loop, trace evaluated by the LTLf semantics) ≥ the
  robust lower bound. Locked into CI by an integration unit test ("F r" over the
  abstraction == plain reachability to r-cells). Suite: 51/51, 603306 assertions.
- ISSUE-0010 (perf, low): OVI's VI-from-below is slow on large strongly-recurrent
  IMDPs; MECCollapse (optimistic) / robust-EC interval iteration is the fast path.

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
