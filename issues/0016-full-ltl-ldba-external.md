---
id: ISSUE-0016
title: Full arbitrary-LTL -> LDBA front-end (Spot/Owl) — needs external toolchain, transfer to big machine
status: open
severity: medium
labels: enhancement, omega-regular, ltl, needs-big-machine, external-dependency
created: 2026-06-26
updated: 2026-06-26
related:
  - src/ltlspec.cpp
  - src/omega.cpp
  - SickertEsparzaJaaxKretinsky2016
  - Hahn2020GoodForMDP
---

## Summary
IMPaCT supports a large, practically-important LTL fragment NATIVELY (no external
deps) — see the dispatcher `src/ltlspec.*`: boolean atoms, X, F, G, U (and bounded
forms), G F (Büchi/recurrence), F G (co-Büchi/persistence), and conjunctions of G F
(generalized-Büchi/patrol), each reducing to a VERIFIED solver (solve / pctl / omega).
ARBITRARY LTL (nested temporal operators outside this fragment) requires translating
the formula to a Limit-Deterministic Büchi Automaton (LDBA) / good-for-MDP automaton
and taking the IMDP × LDBA product, then robust accepting-EC analysis (already built:
`omega::robustBuchiWinningStates`). The LDBA translation itself is best delegated to a
mature library; doing it from scratch, soundly, is large and error-prone.

## Why deferred here
A correct LTL→LDBA translator is a substantial component; the standard, trustworthy
route is **Spot** (spot.lre.epita.fr, C++) or **Owl** (owl.model.in.tum.de, Java)
plus a degeneralization (Seminator 2). These are external toolchains not buildable in
the current environment. Transfer to the machine that will host the AdaptiveCpp build
(it can also carry Spot/Owl).

## Plan (big machine)
1. Build-system spike: link Spot's C++ API (`spot::translator` → deterministic/
   limit-deterministic Büchi) into the toolchain (alongside AdaptiveCpp).
2. `ltlspec`: when a formula is OUT of the native fragment, translate φ → LDBA via
   Spot, build IMDP × LDBA (extend `product::build` to an ω-acceptance product), set
   the Büchi accepting set = product states with an accepting automaton component,
   and solve with `omega::maxBuchiPessimistic` (robust accepting ECs already
   verified, ISSUE-0009). Provide Owl/Rabinizer as alternative back-ends (toolbox).
3. Differential check: on fragment formulas, the LDBA route must equal the native
   dispatcher; on co-safe formulas, the ω route must agree with the LTLf→DFA product.

## References (verified, already in References.bib)
Sickert-Esparza-Jaax-Křetínský (CAV 2016, LDBA); Hahn et al. (TACAS 2020,
good-for-MDP); Křetínský et al. (Owl). Spot/Owl tool URLs are not BibTeX refs.
