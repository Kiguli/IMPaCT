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
