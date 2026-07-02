# Issue Index

See [`README.md`](README.md) for conventions and the literature-counterexample protocol.

| ID | Title | Status | Severity | Labels |
|----|-------|--------|----------|--------|
| [0001](0001-sorted-synthesis-tolerance-stopping.md) | v1 sorted synthesis uses tolerance stopping, not sound interval iteration | in-progress | medium | soundness, tool-v1 |
| [0002](0002-mec-naive-acceptance-rule.md) | MEC naive "staying-action-exists" acceptance rule is unsound | resolved | low | methodology, naive-strawman |
| [0003](0003-pessimistic-interval-nature-trap.md) | Pessimistic interval iteration doesn't converge on nature-confinable ECs | resolved | medium | soundness, robust-ec |
| [0006](0006-dense-transition-matrix-oom.md) | Dense transition matrix causes memory overflow even on "small" cases | in-progress | high | scalability, memory |
| [0007](0007-abstraction-sink-too-loose.md) | Abstraction sink bound too loose -> reach under-estimation (found by Monte-Carlo) | resolved | high | correctness, abstraction |
| [0008](0008-target-double-count.md) | Grid-unaligned target double-counted -> too-high reach lower bound (found by VP MC) | resolved | high | correctness, abstraction |
| [0004](0004-ltlf-dfa-and-product.md) | Phase 2 — LTLf→DFA (done) + IMDP×DFA product + co-safe reachability | open | medium | enhancement, phase-2 |
| [0005](0005-ltlf-dfa-state-explosion.md) | LTLf→DFA state explosion from syntactic normalization | resolved | high | correctness, bug |


| [0009](0009-robust-accepting-ec.md) | Robust accepting end components use optimistic support-MEC structure | resolved | medium | robust-ec, omega-regular |
| [0010](0010-ovi-slow-on-recurrent.md) | OVI slow on large strongly-recurrent IMDPs (perf) | open | low | performance, solver |
| [0011](0011-robust-infinite-horizon-unbounded-noise.md) | Infinite-horizon ω-regular value is 0 on unbounded-noise abstractions (no non-trivial ECs) | resolved | medium | design-decision, abstraction, omega-regular |
| [0012](0012-v1-inner-solve-lp-omax-unify-and-audit.md) | Unify v1 inner-solve on O-maximization (LP names -> aliases) + v1 code audit | in-progress | medium | refactor, cleanup, performance, tool-v1 |
| [0013](0013-finite-safe-sorted-upper-bound-bugs.md) | Bugs in the finite-horizon safety Sorted upper-bound kernels (now-live path) | in-progress | high | correctness, tool-v1, safety, needs-toolchain-verification |
| [0014](0014-pta-zone-mdp-pmax-only.md) | PTA forward zone-MDP is exact for Pmax reachability; Pmin needs backward/game construction | resolved | low | design-decision, timed-automata, scope |
| [0015](0015-robust-interval-pomdp.md) | Robust (interval) POMDP solving — research item, transfer to the big machine | open | medium | research, pomdp, robust, needs-big-machine |
| [0016](0016-full-ltl-ldba-external.md) | Full arbitrary-LTL -> LDBA front-end (Spot/Owl) — needs external toolchain | open | medium | enhancement, omega-regular, ltl, needs-big-machine |
| [0017](0017-sycl-infinite-vi-slow-vs-peers.md) | SYCL infinite-horizon robust VI slow on recurrent IMDPs vs Storm/IntervalMDP.jl | resolved | low | performance, solver, cross-tool |
| [0018](0018-optimistic-bound-is-robust-policy-bracket.md) | Optimistic value is a bracket of the robust controller, not the cooperative optimum | open | low | semantics, documentation, cross-tool |
| [0019](0019-sparse-default-abstraction.md) | Make the sparse abstraction the default everywhere (design decision) | in-progress | medium | design-decision, scalability, abstraction |
| [0020](0020-sparse-vs-dense-abstraction-discrepancy.md) | Sparse (closed-form) and dense (nlopt) abstractions disagree on AS/BA | open | medium | correctness, abstraction, soundness |
| [0021](0021-nonlinear-mean-enclosure-not-certified.md) | Nonlinear mean enclosure (point-sampling / nlopt) is heuristic, not certified-sound | open | medium | soundness, abstraction, nonlinear, design-decision |
| [0022](0022-odimdp-reduction-order-semantics.md) | odIMDP factored ambiguity is reduction-order sensitive (matched IntervalMDP.jl convention) | resolved | medium | design-decision, semantics, odimdp, cross-tool |
| [0023](0023-exact-mode-surfaces-infeasible-rows.md) | Exact rational mode surfaces interval rows with sum(hi) < 1 (silent in floating point) | open | low | correctness, data-quality, exact, cross-tool |
