# IMPaCT v2.0 — Full Test Plan (immutable contracts, all phases)

This is the authoritative enumeration of every test the implementation must
satisfy. Phase 1–2 contracts are already coded (`tests/unit/*`, `tests/integration/*`).
Phase 3–4 contracts are specified here now (inputs + expected + oracle method)
and become concrete test files at the start of each phase, following the same
oracle discipline. Expected values are derived independently of the code.

Legend: ✅ coded · ⏳ specified-here, coded when phase begins.

---

## §1a — O-maximization (`impact::omax::optimize`) ✅
Inner robust Bellman solve over a box-interval ambiguity set. Ref: Givan-Leach-Dean (AIJ 2000); Lahijanian et al. (IEEE TAC 2015).

| Test | Input (lower; upper; V) | Sense | Expected | Oracle |
|---|---|---|---|---|
| hand A | [.1,.2,.1]; [.6,.7,.8]; [0,.5,1] | min | p=[.6,.3,.1], v=0.25 | hand + brute vertices |
| hand A | same | max | p=[.1,.2,.7], v=0.80 | hand + brute |
| no slack | [.3,.3,.4]; [1,1,1]; [0,.5,1] | min=max | v=0.55 | forced distribution |
| single | [0];[1];[.7] | both | p=[1], v=0.7 | trivial |
| tied | [0,0];[1,1];[.5,.5] | both | v=0.5 | any split |
| pivot-exact | [.1,.3];[.7,.3];[0,1] | min | p=[.7,.3], v=0.3 | residual exactly fills gap |
| infeasible×4 | sum(lo)>1 / sum(up)<1 / lo>up / size-mismatch | — | throws `invalid_argument` | feasibility |
| differential | 500 random feasible boxes, n∈[1,5], seed 12345 | both | == brute-force vertex optimum (1e-9) | independent enumeration |

Invariants on every result: `lo ≤ p ≤ up`, `sum(p)=1`, `value=dot(p,V)`.

## §1b — SCC + MEC (`impact::graph`) ✅
Refs: Tarjan (1972); de Alfaro (1997); Chatterjee-Henzinger (JACM 2014).

- **SCC** two linked cycles `0→1→2→0, 3→4→3, 2→3, 4→5`: ⇒ `{0,1,2},{3,4},{5}`.
- **SCC** DAG `0→1→2`: ⇒ `{0},{1},{2}`. **SCC** self-loop `0→0`: ⇒ `{0}`.
- **MEC** `0:→1 | 1:→2 | 2:→1,→3 | 3:→4 | 4:→4`: ⇒ `{1,2}` (choose staying action) and `{4}` (self-loop). 0,3 transient. *Derivation*: {1,2} closed under 1:a0→2, 2:a0→1 and strongly connected; {4} self-sustaining.
- **MEC** deterministic 2-cycle `0↔1`: ⇒ `{0,1}`.
- **MEC** `0:→1 | 1:→1`: ⇒ `{1}` only (0 transient — its only action leaves).

## §1c — Sound robust interval iteration (`impact::solve`) ✅
Refs: Haddad-Monmege (TCS 2018, IMDP soundness); Baier et al. (CAV 2017).
Invariants per state: `lower ≤ V* ≤ upper` (soundness), `upper-lower ≤ 2ε`, midpoint ≈ V*.

| Model | Structure | Target | Expected V*(s0) | Oracle |
|---|---|---|---|---|
| 1 point MC | 0→1@.5,→sink@.5; 1→tgt | {3} | 0.5 | MC linear solve |
| 2 interval | 0→1∈[.4,.6],→sink; 1→tgt | {3} | pess 0.4 / opt 0.6 | adversary vertex |
| 3 self-loop reaches | 0→0@.5,→tgt@.5 | {1} | 1.0 | x=.5+.5x |
| **4 EC, no exit** | EC {0,1} (0↔1), tgt unreachable | {2} | **0.0** | prob0 / MEC-collapse |
| **5 EC, with exit** | 0:stay\|exit→tgt; 1→0 | {2} | **1.0** | controller exits |

Models 4 & 5 are the load-bearing soundness tests — naive VI from [0,1] is stuck above V* on the end component of Model 4; only correct Prob0/Prob1 + EC handling gives 0.

## §2 — LTLf / co-safe front-end (`impact::ltl`) ✅ (front-end) + ⏳ (product)
Semantics: LTLf finite-trace (De Giacomo-Vardi 2013/2015). Membership tables in `test_ltl_dfa.cpp`: `F a`, `a U b`, `G a`, `F(a & X b)`, `F(pickup & F deliver)`.

⏳ **Product IMDP × DFA** (`impact::product`, coded at Phase 2 start):
- size = |reachable (s,q)| ≤ |X|·|Q|; deterministic DFA ⇒ exactly one q-successor per letter.
- a hand 2-state IMDP × 2-state DFA: enumerate and check the product transition relation + which product states are accepting-absorbing.
- **reduction correctness**: for `φ = F target`, product-reachability value == the existing reach solver's value on the bare IMDP (cross-check, must match to 1e-9).
- **PD end-to-end**: Package-Delivery scLTL (`F(pickup & F deliver)` while `G ¬obstacle`) on the ARCH PD model — satisfaction-probability bounds cross-checked vs `oracles.py` on a coarsely-gridded small instance and (when available) vs SySCoRe / IntervalMDP.jl. The current "reduced-spec" single reach-avoid must be strictly improved (no spec collapse).

## §3 — Full LTL / ω-regular ⏳ (coded at Phase 3 start)
Refs: Sickert et al. (CAV 2016) LDBA; Asadi et al. (AAAI 2026; arXiv:2505.04539) robust ω-regular; de Alfaro/Chatterjee-Henzinger ECs; Dutreix-Coogan (CDC 2018 / IEEE TAC 2021) interval-MC precedent. (All keys in `../paper/References.bib`.)

**(a) LDBA front-end (`impact::omega::ldba`)** — back-end-agnostic via ω-word lassos `u·vᵒᵐᵉᵍᵃ`:
- structural: produced automaton passes a limit-determinism check.
- membership: `GF a` accepts `(a)ᵒᵐᵉᵍᵃ` and `(¬a · a)ᵒᵐᵉᵍᵃ`, rejects `a·(¬a)ᵒᵐᵉᵍᵃ`. `FG a` accepts `¬a·(a)ᵒᵐᵉᵍᵃ`, rejects `(¬a·a)ᵒᵐᵉᵍᵃ`.

**(b) Robust accepting end components (`impact::omega::robustAcceptingECs`)** — the novel core:
- tiny explicit IMDP with a hand-derived controllable accepting EC under *all* interval resolutions ⇒ returns exactly that component; a near-identical IMDP where an interval lets nature break the EC ⇒ it is NOT returned. (Derivation tables to be fixed when coded, mirroring §1c style, cross-checked against a brute-force 2.5-player game solver on ≤6-state instances.)

**(c) Qualitative ω-regular** (almost-sure / positive):
- 2-state IMDP where controller forces staying in accepting region ⇒ `GF acc` almost-sure = TRUE (prob 1).
- IMDP where accepting region is a transient set ⇒ almost-sure = FALSE.

**(d) Quantitative ω-regular**:
- small IMDP with hand-computed optimal recurrence probability (reduction to robust reachability of the robust accepting ECs); pessimistic/optimistic sandwich, gap ≤ 2ε. Oracle: brute-force game value on the tiny instance.

**(e) ω-regular ⊇ co-safe regression**: for a co-safe φ, the Phase-3 path must return the same value as the Phase-2 path (no regression).

## §4 — Web app (usability layer) ⏳ (coded at Phase 4 start)
- **codegen**: a JSON spec for the 2D-robot reach-avoid case generates an `example.cpp` that compiles and runs and reproduces the hand-written `ex_2Drobot-RA-U` controller.h5 within ABS_TOL (golden equivalence).
- **spec builder**: Dwyer pattern × scope selections map to the expected LTL string (table, e.g. Response/Globally `"G(p -> F q)"`); generated LTL parses in the front-end.
- **parallelism control**: device selection emits the correct `--acpp-targets` string (`"omp"`, `"cuda:sm_70"`, `"cuda:sm_70;omp"`).
- **API**: POST malformed spec ⇒ 4xx with validation message; POST valid spec ⇒ job id ⇒ results payload; no server-side path traversal in generated filenames.

## §X — Cross-cutting ⏳
- **sparse == dense**: product/solve results identical (≤1e-12) between dense and sparse transition representations on the same model.
- **determinism**: same input ⇒ byte-identical HDF5 across two runs (single-thread) / identical within 1e-9 (parallel).
- **HDF5 round-trip**: save→load of matrices/vectors/controller is identity.
- **External oracles** (CI-gated, optional): PD/PDx and safety cases cross-checked vs PRISM/Storm/IntervalMDP.jl where a comparable model exists.

---

### Oracle provenance
All §1 constants are reproduced independently in `tests/oracles/oracles.py`
(numpy linear-algebra + brute-force vertex enumeration) and guarded by
`tests/integration/test_oracle_consistency.py`. §3 game-value oracles will use a
brute-force 2.5-player solver restricted to ≤6-state instances.
