---
id: ISSUE-0006
title: Dense transition matrix causes memory overflow even on "small" cases
status: open
severity: high
labels: scalability, performance, tool-v1, memory
created: 2026-06-25
updated: 2026-06-25
related:
  - src/IMDP.cpp
  - src/MDP.cpp
  - benchmarks/RESULTS.md
---

## Summary
v1 stores transition probabilities as DENSE Armadillo matrices of size
(state×input) × state. For the 2D-robot reach-avoid example (state 1583 after
target/avoid removal, input 441) this is 698103 × 1583 ≈ 1.1e9 doubles ≈ **8.84 GB**,
which OOM-kills the run under a 4 GB container cap. So even the smallest shipped
example is not "small" in memory; truly tiny cases or sparse storage are required.

## Evidence
Docker run (`impact:baseline`, `--memory=4g`, sorted/CPU, 2D-robot-RA):
abstraction vectors computed (is/ss/ts/min*/max* h5 produced), then:
`maximum transition Matrix dimensions: 698103 x 1583 ... 8840.78Mb` → `Killed`
(real 13m6s before OOM). controller.h5 NOT produced.

## Impact
- Regression goldens can't be generated from this example under tight memory.
- Confirms the plan's dense→sparse concern; products (IMDP×automaton, Phase 2/3)
  multiply state count and will hit this much sooner.

## Resolution / plan
- Short term: generate goldens from a *genuinely tiny* model (few states/inputs)
  or raise the memory cap only where the host allows.
- Core: **sparse transition representation** (CSR-style) for transition matrices
  and products — scheduled as the Phase 2/3 cross-cutting "sparse" task in the plan.
  This is the right place to also reduce the (state×input)×state blow-up.
- Keep "very small case studies only" for local/CI runs until sparse lands.
