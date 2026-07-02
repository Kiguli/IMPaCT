---
id: ISSUE-0016
title: Full arbitrary-LTL -> LDBA front-end (Spot/Owl) — needs external toolchain, transfer to big machine
status: partially-resolved
severity: medium
labels: enhancement, omega-regular, ltl, external-dependency
created: 2026-06-26
updated: 2026-07-01
related:
  - src/ltlspec.cpp
  - src/ltl_spot.cpp
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
## PARTIALLY RESOLVED (2026-07-01): `imdp_solve ltlx` via Spot `ltl2tgba -D`
The DETERMINISTIC-Büchi class of LTL is now handled end-to-end (`src/ltl_spot.cpp`):
φ → deterministic (generalized) Büchi via Spot `ltl2tgba -D` → HOA parse → IMDP×automaton
product → robust ω-regular via `omega::maxGenBuchi{Pessimistic,Optimistic}` (or product
safety for `t`/all acceptance). Validated: point-MDP `ltlx` == Storm `Pmax=?[φ]` for
`F a`/`G F a`/`X X a`/`G(a→X a)`; interval-MDP `G F a` robust 0.3 / opt 0.7 (no peer does
LTL on intervals). Spot is invoked via `IMPACT_LTL2TGBA` (the native binary is in the
source-built Storm tree). **Remaining gap:** NONdeterministic automata (co-Büchi `F G`,
Rabin/parity) are soundly REJECTED — they need a true LDBA (Owl `ltl2ldba`) or a robust
parity solver. The built-in `ltl` fragment already covers `FG`/persistence, so the union of
`ltl` + `ltlx` is a large usable class. Full LDBA remains the open part of this issue.

Refs: Sickert-Esparza-Jaax-Křetínský (CAV 2016, LDBA); Hahn et al. (TACAS 2020,
good-for-MDP); Křetínský et al. (Owl). Spot/Owl tool URLs are not BibTeX refs.

## UPDATE (2026-07-02) — Owl toolchain installed; jump-product is the remaining work
Owl 21.0 (Kretinsky-Meggendorfer-Sickert, ATVA 2018, DOI 10.1007/978-3-030-01090-4_34)
is now installed in the container as a NATIVE binary (no JRE):
/opt/owl/owl-linux-musl-amd64-21.0/bin/owl (`owl ltl2ldba` emits HOA LDBAs). Remaining
implementation: extend the ltl_spot product to CONTROLLER-RESOLVED automaton
nondeterminism (the epsilon-jumps of the LDBA become extra product actions; sound by the
good-for-MDPs property, Hahn et al. TACAS 2020), then the existing robust Buchi engine
applies unchanged. Until then `ltlx` covers the deterministic class and `ltl` covers the
F/G/U/X/GF/FG/patrol fragment (incl. the FG/persistence cases an LDBA would add).
