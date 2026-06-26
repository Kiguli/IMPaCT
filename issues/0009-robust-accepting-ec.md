---
id: ISSUE-0009
title: Robust (pessimistic) accepting end components use the optimistic support-MEC structure
status: resolved
severity: medium
labels: soundness, robust-ec, omega-regular, phase-3
created: 2026-06-25
updated: 2026-06-26
related:
  - src/omega.cpp
  - Dutreix2018satbounds
  - Dutreix2022abstraction
  - WeiningerMeggendorferKretinsky2019BMDP
  - Asadi2026qualitative
---

## Summary
`omega::maxBuchi*` identifies "accepting" end components as maximal end components
(of the SUPPORT graph, hi>0) that contain an accepting state, then reduces to
reachability of those MECs. The support-graph MEC is the OPTIMISTIC end-component
notion: it assumes the play can be kept inside the component. For the robust
(pessimistic, adversarial-nature) value this can be unsound: nature may be able to
EJECT the play from a candidate EC (an interval edge leaving the set with lower
bound 0 that nature can drive positive), so the controller cannot actually visit
the accepting set infinitely often there.

This is the omega-regular analogue of ISSUE-0003 (which was the reachability-level
non-convergence, resolved with OVI). Here it is at the EC-identification level.

## Current status / scope
- `maxBuchiOptimistic` is sound (nature cooperative; support-MEC = optimistic EC).
- `maxBuchiPessimistic` is sound for POINT MDPs (lo==hi, nature has no choice) but
  may OVER-estimate for interval MDPs with nature-ejectable candidate ECs.
- Verified accordingly in tests/unit/test_omega.cpp (point-MDP pess==opt).

## Resolution / plan (literature)
Replace the support-MEC accepting-EC test with a ROBUST accepting-EC computation:
- Dutreix-Coogan "permanent winning components" (leaky-state SCC pruning on the
  upper>0 graph) — CDC 2018 / NAHS 2022; the most graph-native fit for graph::mecs.
- Weininger-Meggendorfer-Kretinsky game reduction (intervals -> a player) — CDC 2019.
- Asadi et al. force_A/force_E attractors (qualitative) — AAAI 2026.
These are the Phase-3 selectable pipelines already recorded in ROADMAP.md; offer
the robust accepting-EC as the pessimistic path while keeping the current
support-MEC method as the optimistic one.

## Resolution (2026-06-26)
Replaced the support-MEC pessimistic reduction with a correct robust almost-sure
Büchi computation in `src/omega.cpp`:
`robustBuchiWinningStates` = nested fixpoint of
1. `robustClosure(X)` — keep states with an action whose may-support (hi>0) ⊆ X
   (controller can stay in X for all nature); then
2. remove `natureAttractor(X, sureAvoid(X, accepting∩X))`, where
   - `sureAvoid` = greatest accepting-free set Z in which, for every within-X
     action, nature can contain ALL mass (sum of in-Z hi ≥ 1 AND sum of out-of-Z
     lo = 0 — i.e. nature is never forced out and can absorb mass 1 inside Z), and
   - `natureAttractor` = states from which, for EVERY controller action, nature has
     a may-successor toward the avoid region (nature forces positive-prob entry);
repeat until stable. `maxBuchiPessimistic` = robust reachability of this region.

The key correction over the support-MEC view: even inside a support EC (which
nature cannot LEAVE), nature can (a) ROUTE AROUND the accepting state via lo=0
edges and (b) PARTIALLY LEAK to an accepting-free trap; both are captured by
`sureAvoid` (feasibility of full containment) rather than mere strong connectivity.

### Counterexample that motivated it (recorded; NOT a literature counterexample)
2-state support EC {0,1}, edges `s -> {0:[0.5,1], 1:[0,0.5]}` for s∈{0,1},
accepting={1}. {0,1} is a genuine support MEC containing the accepting state, so
the old reduction returned value 1. But the edge into 1 has lo=0, so nature sets
P(→1)=0 forever and 1 is never visited ⇒ robust Büchi = 0. Captured in
`tests/unit/test_omega.cpp` ("nature routes around accepting inside a support-EC").
This refutes the NAIVE support-MEC reduction, which the omega-regular literature
already replaces with robust/permanent end-component machinery — it is not a
counterexample to a published algorithm (cf. the ISSUE-0002 wording lesson).

### Confidence basis
- Matches a literature-grounded characterization (2.5-player a.s.-Büchi: robust
  closure + sure-avoid + nature attractor; Chatterjee-Henzinger graph games, with
  the IMDP/robust reading of Dutreix-Coogan permanent components / Asadi et al.).
- **Independent brute-force differential**: a strategy-enumeration oracle
  (`oracleBuchiWinning`) computes the winning region by enumerating every
  memoryless controller strategy and finding nature's reachable accepting-free
  traps. The production algorithm matches it on ~12,000 checks over 4,000 random
  IMDPs (exact equality, not just soundness). Filed as an immutable contract test.

## Classification
Resolved. The optimistic path keeps the support-MEC method (exact for that sense);
the pessimistic path now uses the exact robust a.s.-Büchi region. Both selectable
via `maxBuchiOptimistic` / `maxBuchiPessimistic`.
