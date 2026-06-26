---
id: ISSUE-0005
title: LTLf→DFA state explosion from syntactic normalization (e.g. (G a) U (G a))
status: resolved
severity: high
labels: correctness, bug, phase-2
created: 2026-06-25
updated: 2026-06-25
related:
  - src/ltl.cpp
  - tests/unit/test_ltl_dfa.cpp
---

## Summary
The first LTLf→DFA implementation keyed DFA states by a SYNTACTIC canonical form
(flatten + dedup + identity/annihilator on And/Or). That normalization lacks
Boolean **absorption** (x ∨ (x∧y) = x), so progression of formulas like
`(G a) U (G a)` produced ever-larger residuals that never deduplicated → unbounded
state growth → `toDFA` hung (spinning toward the state cap). Found by the
randomized differential test against the verified evaluator (it hung), then
isolated to the minimal culprit `(G a) U (G a)` with a generator-replay driver.

## Root cause
`der(φ U φ)` unfolds to `der(φ) ∨ (der(φ) ∧ φUφ ∧ Tail)`, i.e. `A ∨ (A ∧ y)`.
Syntactic flatten/dedup does not collapse `A ∨ (A ∧ y)` to `A`, so each
derivative nests a new distinct term forever.

## Fix
Canonicalize DFA states SEMANTICALLY: represent each residual as a Boolean
function over the formula's "anchors" (atoms, temporal subformulas, and the Tail
marker) and key states by their **truth table** over those anchors. `der` is a
Boolean homomorphism over anchors, so equivalent residuals share a truth table
and collapse to a single state ⇒ a finite, minimal DFA. Accepting = truth-table
value at the empty-trace assignment (G,R true; atoms,X,F,U,Tail false).
(`src/ltl.cpp`: Builder::collectAnchors/evalStruct/truthKey/nuAssignment; toDFA
keys by truthKey.)

## Verification
- The minimal culprit and 300 generated formulas now build 3–6-state DFAs instantly.
- Randomized differential (`tests/unit/test_ltl_dfa.cpp`): `dfaAccepts` ==
  `acceptsFinite` on random formulas × random traces. Full suite green.

## Note on the cap (per user feedback)
`maxStates` is now ONLY a safety valve (configurable parameter, generous default)
— never the dedup mechanism, and not a machine-dependent correctness crutch. The
algorithmic bound is the number of anchors K (truth table is 2^K); co-safe specs
have small K. Very large K would warrant a BDD-based canonicalization (future).
