---
id: ISSUE-0009
title: Robust (pessimistic) accepting end components use the optimistic support-MEC structure
status: open
severity: medium
labels: soundness, robust-ec, omega-regular, phase-3
created: 2026-06-25
updated: 2026-06-25
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

## Classification
Known limitation (documented), not a silent bug: the pessimistic path is exposed
with this caveat; soundness holds for the optimistic sense and for point MDPs.
