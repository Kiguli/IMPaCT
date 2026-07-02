---
id: ISSUE-0023
title: Exact rational mode surfaces interval rows with sum(hi) < 1 that floating point silently tolerates
status: open
severity: low
labels: correctness, data-quality, exact, cross-tool
created: 2026-07-02
related:
  - src/exact.cpp
  - benchmarks/crosstool/models/multiObj_robotIMDP.imdp
---

## Summary
`imdp_solve reach --exact` on `multiObj_robotIMDP.imdp` aborts with
"infeasible interval row (sum hi < 1)": some rows of the IntervalMDP.jl-shipped robot
model have upper bounds summing to slightly below 1 (truncated decimals such as
0.999997), so NO probability distribution fits the intervals — the row is infeasible as
an interval MDP in the strict sense. The double-precision solvers (IMPaCT's omax, and
evidently PRISM/Storm/IntervalMDP.jl too) tolerate this silently, effectively assigning
the missing ~3e-6 mass to the highest-value successor anyway.

## Why it matters
Exact arithmetic makes the model-validity check unavoidable: this is a feature, not a
bug — the same strictness that certifies exact fixpoints also certifies input sanity.
Cross-tool comparisons on such models are comparing tools' silent-repair conventions.

## Decision
Keep the strict behaviour in exact mode (abort with a clear message). Possible future
option: `--exact-repair` that scales upper bounds up to feasibility and reports the
perturbation. Floating-point solvers unchanged.
