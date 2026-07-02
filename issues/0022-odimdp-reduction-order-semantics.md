---
id: ISSUE-0022
title: odIMDP factored ambiguity is reduction-order sensitive — IMPaCT matches IntervalMDP.jl's dim-1-innermost convention
status: resolved
severity: medium
labels: design-decision, semantics, odimdp, cross-tool
created: 2026-07-01
updated: 2026-07-01
related:
  - src/odimdp.cpp
  - benchmarks/crosstool/models/od_demo.ref.json
  - benchmarks/crosstool/peers/odimdp_oracle.jl
---

## Summary
For orthogonally-decoupled IMDPs the robust Bellman value depends on the ORDER in which
the per-dimension marginals are optimised, because nature's later (inner) marginal choices
may condition on the destination coordinates already realised by the outer dimensions —
the per-prefix rectangular semantics. Reducing dimension n innermost (nature's dim-n
choice sees dims 1..n-1) and reducing dimension 1 innermost (nature's dim-1 choice sees
dims 2..n) define DIFFERENT nature strategy classes and can give different pessimistic
values.

## How it was found (empirically, cross-tool)
First implementation reduced the HIGHEST dimension innermost. On the discriminating
`od_demo` model the optimistic values matched IntervalMDP.jl exactly (0.8536585360) but
the pessimistic values differed: IMPaCT 0.3478/0.7246 vs IntervalMDP.jl 0.3714/0.7286 —
IMPaCT's nature was strictly more powerful under that order for this model. Reading
IntervalMDP.jl's `state_action_bellman` (src/bellman.jl: dimension 1 is reduced first via
`dense_sorted_state_action_bellman(@view(V[:, I]), prob[1], ...)` per combination `I` of
the remaining coordinates, then dims 2..n outward) identified its convention: **dimension
1 innermost**.

## Decision
`odimdp::factoredOpt` reduces the FASTEST (first) dimension per pass over contiguous
slices — i.e. dimension 1 innermost, then 2, ..., n outermost — matching IntervalMDP.jl.
After the fix, both senses agree with IntervalMDP.jl to 10 digits on `od_demo`
(pess 0.3714285709 / 0.7285714278; opt 0.8536585360 / 0.9999999995).

## Note for the literature-facing docs
Mathiesen-Haesaert-Laurenti (arXiv:2411.11803) define the factored ambiguity set; when
comparing tools or citing values, the reduction order (equivalently, which conditional
nature class is intended) must be stated. Cross-tool agreement is only meaningful once
the convention is pinned — another instance of the semantics-alignment lesson from the
optimistic-safety and threshold observations (TOOL_COMPARISONS §5.2).
