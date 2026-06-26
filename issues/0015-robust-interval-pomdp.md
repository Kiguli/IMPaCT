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
