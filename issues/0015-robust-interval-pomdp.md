---
id: ISSUE-0015
title: Robust (interval) POMDP solving — research item, transfer to the big machine
status: open
severity: medium
labels: research, pomdp, robust, needs-big-machine, future-work
created: 2026-06-26
updated: 2026-06-26
related:
  - src/pomdp.cpp
---

## Summary
The implemented POMDP solver (`src/pomdp.cpp`) is for POINT models: exact
finite-horizon reachability by belief-state value iteration. The natural IMPaCT
extension is the ROBUST / interval POMDP: transition T and observation O
probabilities lie in intervals and an adversarial nature resolves them. This is NOT
implemented and should be transferred to the toolchain/research machine — it is a
genuine research problem, not a mechanical extension.

## Why it is hard (and not implemented here)
The robust value couples the belief update with nature's choice:
  V_t(b) = max_a  min_{T,O ∈ intervals}  sum_o P_{T,O}(o | b,a) * V_{t-1}( b'_{T,O,a,o} )
Nature's choice affects BOTH the observation probabilities P(o|b,a) AND the posterior
belief b' (which then indexes V_{t-1}). The inner min is over a continuous polytope of
distributions and is not, in general, optimized at an easily-enumerated vertex because
of the belief-dependence — so the clean O-maximization trick used for fully-observable
IMDPs does not directly apply. Implementing an EXACT or provably-sound solver needs
care; an unverifiable heuristic would violate this project's verification discipline
(every algorithm differential-/oracle-checked), so it is deferred rather than guessed.

## Plan / approaches (literature)
- **Itoh-Nakamura (2007)**, "Partially observable Markov decision processes with
  imprecise parameters" — interval/imprecise POMDPs, belief-based robust DP.
- **Osogami (NeurIPS 2015)**, "Robust partially observable Markov decision process" —
  robust value iteration with (s,a)-rectangular ambiguity; conditions for tractable
  robust Bellman backups.
- **Suilen-Simão-Jansen-Parker** and related work on robust/uncertain POMDPs
  (finite-state-controller and convex-optimization formulations).
Verify each citation (Crossref/DBLP) before adding to References.bib, per project policy.

## Verification target (when implemented)
A brute-force oracle analogous to `test_pomdp.cpp`: enumerate controller
history-policies AND nature history-policies (over interval vertices) for tiny models
and small horizon; the solver's robust value must equal the max-min of that game.

## UPDATE (2026-07-02) — literature grounded (verified), design constraint recorded
Canonical verified references for robust/uncertain POMDPs: Osogami, "Robust Partially
Observable Markov Decision Process", ICML 2015 (PMLR 37:106-115); Cubuktepe, Jansen,
Junges, Marandi, Suilen, Topcu, "Robust Finite-State Controllers for Uncertain POMDPs",
AAAI 2021 (DOI 10.1609/aaai.v35i13.17401); Suilen, Jansen, Cubuktepe, Topcu, IJCAI 2020
(convex optimization for uncertain POMDPs). Key design constraint learned from this
literature: under interval transitions the BELIEF is not a point (interval belief sets /
FSC synthesis is required), so a naive per-step O-max on point beliefs is UNSOUND — no
prototype will be shipped until the belief-set (or FSC) formulation is implemented.
Remains open as the research item it is, now with the correct starting papers.

## RESEARCH PROBLEM STATEMENT (2026-07-02) — see session summary
Formal statement recorded with the verified literature above: robust value computation
for interval POMDPs requires set-valued beliefs (nature's per-step interval choice makes
the Bayes update non-unique), so the sound objects are belief SETS / weight polytopes or
finite-state controllers optimized against nature (Osogami ICML 2015 = convex-concave
point-based updates under rectangularity; Cubuktepe et al. AAAI 2021 = FSC synthesis via
sequential convex programming). The IMPaCT-specific open problem: produce interval
POMDPs by ABSTRACTION of continuous partially-observed stochastic systems (observation
kernel + dynamics discretized together) and solve them with certified robust bounds that
refine to the continuous closed loop — no existing tool does either half.
