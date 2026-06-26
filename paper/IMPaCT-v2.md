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

- **Safety synthesis** (broadens to the ARCH safety category: BAS, rooms).
  `solve::maxSafety{Pessimistic,Optimistic}` = max P(never reach `avoid`) =
  1 - min-reach-to-avoid. Added a `controllerMax` flag to the solver so the
  controller can MINIMIZE reach (for safety) as well as maximize (reachability),
  reusing OVI. Verified (test_safety.cpp): controller-can-avoid → 1; unavoidable →
  0; coin → 0.5; point pess==opt; and a differential vs reachability on
  single-action interval MDPs (safety == 1 - reach). Suite: 56/56, 606499 assertions.
- **Consolidated CAV paper draft** at `paper/draft.md` (algorithms + verified
  citations + verification methodology incl. the MC-found bugs + experimental results).

### 2026-06-26
- **Cross-tool benchmarking (reference-values mode)** — user request: run the
  benchmarks we should be able to solve side-by-side against IntervalMDP.jl /
  PRISM / Storm and check we get the *same outputs* (times may differ). Built on
  a tool-neutral exchange format so no peer install is needed yet (user chose
  "reference values only, defer installs").
  - **`src/imdp_io.{h,cpp}`** — explicit `.imdp` format (`states` / `init` /
    `label` / `tran s a to:lo:hi ...`) + parser/writer. Point MDPs use lo==hi.
    Round-trip + parse-then-solve contracts (`test_imdp_io.cpp`).
  - **`tools/imdp_solve.cpp`** — CLI runner (pure std C++, no SYCL): reads a
    model + property, prints machine-parseable `result\tlower=..\tupper=..` lines.
  - **`benchmarks/crosstool/`** — shared models + sibling `.ref.json` carrying
    INDEPENDENTLY-COMPUTED reference values with provenance (point MDPs: exact
    linear solve; interval MDPs: closed-form O-maximization / action×extremal-
    nature enumeration — exactly the robust semantics IntervalMDP.jl computes).
    `compare.py` checks IMPaCT's sound `[lower,upper]` brackets each reference AND
    the midpoint matches within tol, both senses. **12/12 checks pass.** Design:
    references are data, so a peer's measured numbers later slot into the same
    `.ref.json` with zero harness change; discrepancies become `issues/` entries.
- **PRISM input-style support** (user extension). **`src/prism.{h,cpp}`** parses a
  documented PRISM-language *subset* (single bounded int var: `mdp`/`const int`/
  `module..endmodule`, `x:[lo..hi] init i`, guarded commands `[a] x=K -> p:(x'=E)
  + ...` with point OR interval `[lo,hi]` probabilities and updates in {int, x,
  x±int}, `label "n" = x=K | ...`) into the *same* `io::Problem`, so PRISM models
  feed the identical solver/CLI/harness. `test_prism.cpp` (immutable contracts):
  point chain → exact 0.25, interval fork → [0.4,0.6], const + controller-choice
  + `x'=x+1` + disjunctive label, and **a PRISM model and the equivalent `.imdp`
  model solve identically**. The CLI detects `.prism`/`.pm`/`.nm` by extension;
  `choice.prism` added to the crosstool suite confirms front-end equivalence
  end-to-end. Design decision: subset, single-variable, no synchronization/
  formulas/rewards yet (documented future work) — keeps state index = var value,
  TDD-clean. Refs: PRISM (Kwiatkowska-Norman-Parker, CAV 2011).
  Suite after both: **64 cases / 606,532 assertions green.**

- **Phase 3 core — robust accepting end components / quantitative robust Büchi
  (ISSUE-0009 RESOLVED).** Replaced the unsound support-MEC pessimistic reduction
  with an exact robust almost-sure-Büchi computation (`omega::robustBuchiWinningStates`).
  Algorithm (2.5-player a.s.-Büchi nested fixpoint): repeat { `robustClosure` (keep
  states with an action whose hi>0-support ⊆ X — controller stays for all nature);
  remove `natureAttractor(sureAvoid(X, accepting∩X))` } where `sureAvoid` = greatest
  accepting-free set nature can contain in for sure (in-set hi-sum ≥ 1 AND out-of-set
  lo-sum = 0) and `natureAttractor` = states from which nature forces positive-prob
  entry (every controller action has a may-successor into it). `maxBuchiPessimistic`
  = robust reachability of the winning region.
  - *The correction:* even inside a support EC (which nature cannot LEAVE), nature
    can (a) route around the accepting state via lo=0 edges or (b) partially leak to
    an accepting-free trap. Headline counterexample to the NAIVE reduction (recorded,
    not a literature counterexample): 2-state support-MEC {0,1} containing the
    accepting state but with a lo=0 edge into it ⇒ old method 1, true robust 0.
  - *Confidence:* an independent brute-force **strategy-enumeration oracle**
    (enumerate every memoryless controller strategy; per strategy find nature's
    reachable accepting-free traps) matches the production algorithm EXACTLY on
    ~12,000 checks over 4,000 random IMDPs (immutable differential contract in
    `test_omega.cpp`) — evidence of exactness, not merely soundness. Plus 4 hand
    cases (route-around→0, forced-edge→1, sink-leak→0, quantitative reach→0.5).
  - *Design:* optimistic path keeps the support-MEC method (exact when nature
    cooperates); pessimistic path uses the robust region. Refs: Chatterjee-Henzinger
    (graph games); Dutreix-Coogan permanent components; Asadi et al. (qualitative).
  Suite: **69 cases / 618,546 assertions green.**
- **Generalized Büchi / patrol** (`omega::maxGenBuchi{Optimistic,Pessimistic}`):
  "visit each of several regions infinitely often" (conjunction of GF), the headline
  ω-regular spec no stochastic-category tool solves. Standard round-robin
  degeneralization (counter product, state = s*k+c) into the validated robust Büchi
  solver; reduces exactly to maxBuchi* for one set, vacuously true for none.
  Tests: deterministic 3-cycle patrol of two regions → 1; a robust-vs-optimistic
  contrast where nature can deny one region via a lo=0 edge (pessimistic 0,
  optimistic 1). Suite: **73 cases / 618,555 assertions green.**
- **Persistence / co-Büchi — F G p (reach-then-stay)** (`omega::maxPersistence
  {Optimistic,Pessimistic}`): the reach-then-stay objective (an ARCH benchmark
  family). Value = reach the largest sub-region of p the controller can remain in
  forever — robustClosure (may-support ⊆ region, robust) or optimisticClosure
  (nature can contain all mass, cooperative) — then it stays in p forever. Reuses
  the validated robust-closure + reachability machinery. Tests: reach-then-stay → 1;
  a robust-vs-optimistic contrast where nature leaks out of p via a lo=0 edge
  (pessimistic 0.5, optimistic 1); F G false → 0 / F G true → 1.
  With this, IMPaCT now covers the three core ω-regular acceptance types robustly:
  Büchi (recurrence GF), generalized Büchi (patrol ⋀GF), co-Büchi (persistence FG).
  Suite: **76 cases / 618,562 assertions green.**

- **ω-regular benchmarking + a negative continuous-abstraction finding (ISSUE-0011).**
  Extended `tools/imdp_solve` with `buchi` / `persist` / `patrol` properties and added
  ω-regular models to `benchmarks/crosstool` with independently hand-computed reference
  values: Büchi recurrence reaching an accepting EC (0.5), the route-around support-EC
  (robust 0 / optimistic 1), persistence `F G safe` with a leak (robust 0.5 /
  optimistic 1), patrol `⋀ G F` on a 3-cycle (1). **20/20 cross-tool checks pass.**
  Bonus robustness: the `.imdp` parser now rejects out-of-range state indices.
  - **Finding (validate_omega.cpp):** abstracting an UNBOUNDED-noise system with an
    absorbing off-grid SINK gives a transition system with NO non-trivial end
    components (the SINK is the only recurrent class; the loose support window erodes
    any candidate region), so EVERY infinite-horizon ω-regular value is 0 — robustly
    AND optimistically. This is mathematically correct, not a solver bug: the robust
    ω-regular solvers are exact/non-trivial on explicit IMDPs (12k-check oracle
    differential). Non-trivial continuous ω-regular needs a bounded-disturbance /
    reflecting-boundary abstraction — recorded for the abstraction roadmap. This is
    an honest scoping result: the ω-regular CONTRIBUTION is demonstrated on explicit
    IMDPs; the continuous side currently covers reach-avoid / co-safe / finite-horizon.

- **Bounded-disturbance abstraction → non-trivial robust continuous ω-regular
  (closes the ISSUE-0011 gap).** Added `abstraction::transitionInterval1DUniform`:
  the 1-D transition mass interval for a BOUNDED uniform disturbance `w ~ U[-W,W]`
  (overlap of the target box with the reachable window `[mu-W, mu+W]`, min/max over
  the source-cell mean range; concave ⇒ max at the box centre, min at an endpoint).
  TDD'd vs a fine mu-scan oracle (18k assertions) + forced-mass/disjoint edge cases.
  With bounded support a cell well inside the domain has NO sink edge, so genuine
  robust end components exist. `benchmarks/validate_omega_bounded.cpp`:
  `x' = 0.9x + 0.5u + U[-0.3,0.3]`, recurrence `G F region` → robust value 1 at all
  12 cells, and the continuous closed loop is Monte-Carlo validated (empirical
  recurrence ≥ robust lower bound at every start). So robust ω-regular synthesis on
  continuous systems is non-trivial once the disturbance is bounded — the
  unbounded-Gaussian triviality (ISSUE-0011) is specific to unbounded noise.
  Suite: **80 cases / 627,572 assertions green.**

- **v1 inner-solve unification + audit (ISSUE-0012, user request).** The original
  IMPaCT computed the robust Bellman update two ways — a per-state GLPK LP and the
  closed-form O-maximization sort-and-assign — returning the same value, but
  O-maximization is O(n log n) vs an LP solve and is verified against brute-force.
  Made O-maximization the single default: the four public controller entry points
  (`{infinite,finite}Horizon{Reach,Safe}Controller`) now delegate to the `*Sorted`
  (O-maximization) implementations as backwards-compatible aliases; the GLPK LP
  implementations are dead-coded (DEPRECATED in place) pending deletion + GLPK
  removal once compile-verified on the SYCL toolchain. Audit recorded in ISSUE-0012:
  a REAL finite-vs-infinite safety inconsistency in the (now-dead) LP path; a
  FALSE-POSITIVE "GLPK indexing bug" (it is the standard 1-based idiom — flagged to
  avoid re-claiming, per the ISSUE-0002 lesson); minor issues (ignored `glp_simplex`
  status, dead `ia`, two tolerances) all in the dead LP path; and a live follow-up
  (Sorted-path `//TODO` error stubs). Edits are mechanical signature-preserving
  redirects, NOT compile-verified here (no SYCL/GLPK toolchain in this environment).

- **More noise types (user request a).** Added a pluggable noise interface:
  `abstraction::transitionInterval1DGeneric(noiseCdf, ...)` computes the 1-D
  transition mass interval for ANY symmetric, unimodal additive disturbance from its
  CDF (mass = F(b-mu) - F(a-mu), unimodal in mu ⇒ max at the box centre, min at an
  endpoint — the same structure as the Gaussian/uniform specializations, which stay
  as fast paths). Added `triangularCdf` (bounded support, so it also yields the
  sink-free transitions needed for non-trivial robust ω-regular) and `laplaceCdf`
  (heavy-tailed). TDD: generic-with-Gaussian-CDF matches the Gaussian specialization
  to 1e-9, and triangular/Laplace match brute-force mu-scans (≈29k assertions); CDF
  validity (monotone/normalized/symmetric) checked. New noise models now drop in by
  supplying a CDF. Suite: **83 cases / 638,804 assertions green.**

- **Phase 4 — web app: small-model IMDP + satisfaction-probability visualization
  (user request).** `tools/imdp_solve` gained a `--json` mode emitting the full model
  structure (states/init/labels/edges) + per-state `[lower,upper]` values for the
  chosen property/sense. `webapp/` is a dependency-free app: a stdlib-Python backend
  (`server.py`) that serves the front-end and shells to the verified CLI on
  `POST /api/solve`, and a self-contained SVG front-end (`static/`) that draws the
  IMDP (circular layout, no CDN) with each state **heat-mapped by its robust/
  optimistic satisfaction probability** (initial state highlighted, labelled states
  starred, edges showing `[lo,hi]` on hover). Works for every property (reach/safety/
  buchi/persist/patrol) in `.imdp` or PRISM input, with a configurable state cap so
  the view stays legible for the small models it targets. The C++ core is untouched
  (thin shell over the CLI, so the picture always reflects the exact synthesis).
  Tested end-to-end (HTTP 200 + correct values/errors); the richer Vue/Flask/Docker
  build from the plan layers on the same `/api/solve` contract later.

- **Finite-safety audit follow-up (user pushback → ISSUE-0012 retraction + ISSUE-0013).**
  The user challenged the ISSUE-0012 "finite-vs-infinite safety inconsistency = bug"
  finding (correctly). An adversarially-checked multi-agent verification against the
  IMPaCT theory paper (arXiv:2401.03555v2, Sec 5.1 / Algorithm 3 / Remark 5.1)
  confirmed it is NOT a bug: finite safety uses direct stay-safe VI (avoid = value 0)
  and infinite uses reach-avoid VI (avoid = value 1) + 1-complement — provably equal
  (W_k = 1 - V_k ∀ finite k); the infinite affine avoid term is for convergence, not
  a different quantity. Over-claim RETRACTED (ISSUE-0002 lesson reinforced).
  - But the same review surfaced GENUINE, separate bugs in the now-live finite-safe
    `Sorted` UPPER-bound kernels (`GPU_synthesis.cpp`): (1) avoid mass wrongly added
    to the value (over-count); (2) an uninitialized `temp1` (UB); (3) a missing base
    transition·value term; (4) a missing avoid-residual block — all in the upper
    loops, fixed by mirroring the correct lower-bound siblings. Tracked in ISSUE-0013.
    NOT compile-verified here (no SYCL toolchain) — flagged for the Windows toolchain
    run, with the LP path / v2 `maxSafety` / cross-tool refs as oracles. Method note:
    a Workflow (4 readers → reconcile → 3 adversarial refuters, 368k tokens) caught
    both the retraction and the real bugs that a single pass missed.

- **Stretch: PCTL / CTL model checking on IMDPs** (`src/pctl.{h,cpp}`). State
  subformulas are state sets (compose bottom-up); each path operator returns a sound
  robust probability interval via the verified solver. Added `next` (X), `until`
  (φ U ψ = constrained reach: ¬φ∧¬ψ states made absorbing-dead), `boundedUntil`
  (φ U^{≤k} ψ = exact finite-horizon DP), pessimistic (robust) + optimistic, with
  Finally=reach / Globally=safety reused. Threshold verdicts `check`/`satStates`
  return Sat/Unsat/Unknown soundly over `[lower,upper]`. CTL sugar: `EF` (∃ positive
  reach), `AG` (robust almost-sure stay). TDD (`test_pctl.cpp`): X/until/bounded hand
  cases, until==reach when φ=all, bounded→unbounded convergence, and a **600-model
  differential** of bounded-until vs an explicit finite-horizon DP on random point
  MDPs. Scope: controller-Pmax flavour (Pmin/adversarial-controller future). Refs:
  Hansson-Jonsson (FAC 1994); Baier-Katoen Ch.10; Puggelli et al. (CAV 2013, interval
  PCTL) — all verified in References.bib. Suite: **91 cases / 640,900 assertions green.**

- **Stretch: bounded STL on IMDP abstractions** (`src/stl.{h,cpp}`). Probabilistic
  (robust) satisfaction of bounded Signal Temporal Logic over the discretized signal.
  STL's defining pieces: (i) `predicateCells` maps a real atomic predicate μ(x)≥0 to
  the grid cells whose centre satisfies it (the signal→abstraction bridge); (ii)
  time-bounded operators `F_[a,b]`, `G_[a,b]`, `φ U_[0,b] ψ` by EXACT finite-horizon
  DP — F/U reuse the verified `pctl::boundedUntil`, `G_[0,b]` is a finite-horizon
  bounded-safety DP, and the windowed `[a,b]` forms compose `a` steps of free robust
  evolution with the `[0,b-a]` operator. Pessimistic (robust) + optimistic.
  TDD (`test_stl.cpp`): 1-D/2-D predicate→cell mapping, F_[0,b]=bounded reach,
  G_[0,b] hand cases + robust-vs-optimistic invariance, windowed F_[a,b], and a
  500-model differential of G_[0,b] vs explicit finite-horizon DP. Algorithm→paper
  links (all verified in References.bib): STL syntax/semantics = Maler-Nickovic
  (FORMATS 2004); probabilistic STL for control = Sadigh-Kapoor (RSS 2016);
  finite-horizon bounded operators = Baier-Katoen Ch.10. Spatial-robustness STL
  (single-trace) noted as future work. Suite: **98 cases / 642,650 assertions green.**

- **Stretch: timed-automata foundation — DBM zone abstraction** (`src/dbm.{h,cpp}`).
  The canonical data structure for zone-based timed-automaton reachability: a zone =
  convex set of clock valuations via difference constraints x_i - x_j ≤/< c, stored
  as an (n+1)² Difference Bound Matrix (clock 0 = zero reference). Implemented the
  standard operations (Bengtsson-Yi 2004): `canonicalize` (Floyd-Warshall),
  `isEmpty` (negative cycle), `constrain`/`intersect` (guards), `up` (delay/time
  elapse), `reset` (clock:=0), `includes` (fixpoint inclusion), `contains`
  (membership). TDD (`test_dbm.cpp`): emptiness/guard/inclusion hand cases plus a
  GRID DIFFERENTIAL oracle that checks intersect/up/reset against an explicit
  finite-grid set model of the continuous semantics (385 assertions). This is the
  foundation a probabilistic-timed-automaton front-end builds on — the zone graph
  yields a finite (I)MDP the verified reach/PCTL solvers then handle (next step).
  Algorithm→paper links (all verified in References.bib): Alur-Dill (TCS 1994, timed
  automata + regions); Bengtsson-Yi (2004, DBM/zone algorithms); the PTA target is
  Kwiatkowska-Norman-Segala-Sproston (TCS 2002). Suite: **105 cases / 643,035 green.**

- **Stretch: timed-automaton zone-graph reachability** (`src/ta.{h,cpp}` + DBM
  `extrapolate`). A timed automaton (locations with clock invariants; edges with
  clock guards and resets) is model-checked for location reachability by forward
  exploration of symbolic states (location, zone): delay within the invariant, take
  an edge (intersect guard, reset clocks), delay within the target invariant;
  termination via maximal-constant extrapolation (`dbm::extrapolate`, Behrmann et al.
  2006) + zone inclusion. Convenience clock-constraint builders (`clkLe/Lt/Ge/Gt`).
  TDD (`test_ta.cpp`): timing-feasible vs invariant-blocked reachability, sequential
  resets, two-clock coupling under delay (x,y rise together so x≥2 forces y≥2), and
  termination on a resetting self-loop (finite zone graph, no cap hit). The
  probabilistic-TA step adds edge distributions so the zone graph becomes a finite
  (I)MDP for the verified reach/PCTL solvers — the timed↔stochastic integration
  point. Refs (verified): Alur-Dill (TCS 1994), Bengtsson-Yi (2004),
  Behrmann-Bouyer-Larsen-Pelanek (2006), KNSS (TCS 2002, PTA).
  Suite: **110 cases / 643,044 assertions green.**

- **Stretch: probabilistic timed automata (PTA) → MDP → reach** (`src/pta.{h,cpp}`),
  the timed↔stochastic integration point. A PTA edge carries a distribution over
  (reset,target) branches; the forward zone graph induces a finite MDP whose states
  are symbolic states (location, canonical zone): each enabled edge is an action
  sending probability p_k to the branch-k successor symbolic state. `maxReachLocation`
  = `solve::maxReach` on this MDP = EXACT max probability of reaching a target
  location (point distributions ⇒ exact). Identity by canonical-zone EQUALITY (exact
  probabilities) + extrapolation (finiteness); empty/blocked branches and no-edge
  locations route to an absorbing deadlock sink. TDD (`test_pta.cpp`): timing-gated
  probabilistic edge (Pmax=branch prob), controller picks the better edge,
  sequential resets compose, invariant-blocked unreachability, deadlock. Scope:
  Pmax only — Pmin needs the backward/game or digital-clocks construction (ISSUE-0014).
  Refs (verified): Kwiatkowska-Norman-Segala-Sproston (TCS 2002),
  Kwiatkowska-Norman-Sproston-Wang (Inf.&Comp. 2007). Suite: **115 cases / 643,053 green.**

- **Stretch: POMDPs — exact finite-horizon reachability** (`src/pomdp.{h,cpp}`).
  Belief-state value iteration: `V_0(b)=b(target)`, `V_t(b)=max_a Σ_o P(o|b,a)
  V_{t-1}(belief-update(b,a,o))` with target made absorbing (so "in target at H" =
  "reached within H"). Exact for finite horizon (no α-vector pruning needed); beliefs
  continuous but the reachable belief tree is finite per horizon. TDD (`test_pomdp.cpp`):
  fully-observable POMDP == MDP finite-horizon DP (200 random models), a hand case
  (per-step 0.5 → 0.75 at H=2), and a **250-model differential vs brute-force
  observation-history-policy enumeration** (the exact finite-horizon optimum). Robust
  interval-POMDP (adversarial nature + robust belief update) noted as future work.
  Refs (verified): Smallwood-Sondik (Oper. Res. 1973), Kaelbling-Littman-Cassandra
  (AIJ 1998). Suite: **118 cases / 643,506 assertions green.**

  Citations added & Crossref-verified this round: PRISM (CAV 2011), Hansson-Jonsson
  (FAC 1994), Maler-Nickovic (FORMATS 2004), Sadigh-Kapoor (RSS 2016), Alur-Dill
  (TCS 1994), Bengtsson-Yi (2004), Behrmann et al. (2006), KNSS (TCS 2002), KNSW
  (Inf.&Comp. 2007), Smallwood-Sondik (1973), Kaelbling-Littman-Cassandra (1998) —
  every new algorithm linked to a verified paper, per the standing directive.

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
