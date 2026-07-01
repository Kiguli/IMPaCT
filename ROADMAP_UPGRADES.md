# IMPaCT upgrade roadmap (from the §10 feature gaps)

Literature-grounded plan for closing the capability gaps surfaced by the cross-tool
feature matrix ([`TOOL_COMPARISONS.md`](TOOL_COMPARISONS.md) §3). Each item names the
canonical paper(s) whose algorithm is applied, how it reuses IMPaCT's existing machinery
(the O-maximization inner solve `omax::optimize`, the OVI/interval-iteration in
`solve.cpp`, and the MEC decomposition in `graph_utils.cpp`), the correctness argument,
and the cross-tool / analytic **oracle** used to check it. Every upgrade is validated the
same way the reward feature was: analytic hand-value + the peer tool on its native query.

The common substrate: for an interval MDP the robust Bellman inner problem
`opt_{p in [lo,hi]} Σ p(s')V(s')` is solved in closed form by O-maximization
([`omax::optimize`](src/omaximization.h), Givan-Leach-Dean AIJ 2000 / Lahijanian-
Andersson-Belta TAC 2015). Pessimistic (robust) = `Sense::Min`, optimistic = `Sense::Max`.
Every objective below is "the same VI with a different value/reward recursion around the
same inner solve".

---

## 1. Expected reward — ✅ DONE (2026-07-01)

**Papers.** Iyengar, *Robust Dynamic Programming*, Math. OR 2005; Nilim & El Ghaoui,
*Robust control of MDPs with uncertain transition matrices*, Oper. Res. 2005; Givan-Leach-
Dean, *Bounded-parameter MDPs*, AIJ 2000; Baier et al. (interval iteration, CAV 2017);
Hartmanns-Kaminski (OVI, CAV 2020); Puterman 1994 (SSP finiteness).

**What shipped.** `solve::expReachReward` / `expDiscountedReward` (+ `maxReachReward*`),
the `reward <s> <v>` `.imdp` directive, `imdp_solve reward LABEL [--discount g]`, DRN +
IntervalMDP.jl reward converters. `V(s)=r(s)+γ·max_a opt_p Σ p V`, target absorbing for
γ=1 (SSP; +∞ when the target is not reached a.s.), contraction for γ<1.

**Validated** (`reward_chain`, [`benchmarks/crosstool/models/reward_chain.ref.json`](benchmarks/crosstool/models/reward_chain.ref.json)):
reachability reward pess/opt **3 / 5** = PRISM `Rmaxmin/Rmaxmax` = analytic; discounted
(γ=0.9) **2.8 / 4.2727** = IntervalMDP.jl = analytic. (Storm's interval reward returns only
the nature-max value — documented.)

---

## 2. Long-run average / mean-payoff reward — NEXT (flagship)

**Papers.** Puterman 1994 Ch. 8–9 (average-reward optimality / gain-bias equation
`g + h = max_a{ r_a + P_a h }`, relative VI); **Ashok, Chatterjee, Daca, Křetínský,
Meggendorfer, *Value Iteration for Long-Run Average Reward in MDPs*, CAV 2017**
(arXiv:1705.02326) — the two-phase VI (mean-payoff *inside* each maximal end component,
then weighted-reachability VI on the MEC-quotient), which is the exact template IMPaCT's
MEC code plugs into. Robust/interval mean-payoff extends the inner solve to `omax`.

**Reuse.** Very high: `graph::mecs` + `collapseMECs` give phase-2's quotient verbatim;
`omax::optimize` is the robust inner solve for the gain within a MEC; the reward field +
`.imdp` parser (from item 1) are reused.

**Correctness / oracle.** Two-phase VI is sound on the MEC-quotient (Ashok et al.). No peer
does LRA on *intervals*, so validate the phases separately: (a) mean-payoff on an ordinary
MDP vs **Storm `LRAmax=?`**; (b) robust value brackets the endpoint LRAs (compute LRA on
the lower- and upper-probability MDPs; the robust value must lie between). Plus a small
analytic 2-state cycle.

**Effort:** medium-high. **Value:** high (unique — no tool does robust interval LRA).

---

## 3. Multi-objective / Pareto (robust reachability, then reward) — planned

**Papers.** **Etessami, Kwiatkowska, Vardi, Yannakakis, *Multi-Objective Model Checking of
MDPs*, TACAS 2007** (achievable set convex; memoryless-randomised strategies suffice;
LP + ε-approximable Pareto curve); Forejt, Kwiatkowska, Norman, Parker, Qu, *Quantitative
Multi-Objective Verification*, TACAS 2011 (achievability / numerical / Pareto trichotomy;
implemented in PRISM).

**Reuse.** The robust weighted-sum VI is `omax::optimize` with the value promoted to a
vector and scalarised `vv[i] = Σ_j w_j V_j[to]` — one extra dot-product per successor list;
O-maximization itself unchanged. Sweep the weight simplex to trace the Pareto curve.

**Correctness / oracle.** Degenerate intervals (`lo=hi`) reduce to an ordinary MDP →
compare the Pareto points to **PRISM/Storm `multi(Pmax=?[F t1], Pmax=?[F t2])`**. Validate
the k=1 case against the existing `maxReachPessimistic`.

**Effort:** medium-high (~3–4 wk for 2-objective robust reachability; +reward later).
**Value:** medium-high (PRISM/Storm have it only for non-interval models).

---

## 4. Full arbitrary LTL via LDBA — planned (needs Spot/Owl; ISSUE-0016)

**Papers.** **Sickert, Esparza, Jaax, Křetínský, *Limit-Deterministic Büchi Automata for
LTL*, CAV 2016** (LTL→LDBA correct for probabilistic/MDP model checking; already in
`References.bib`); Hahn, Perez, Schewe, Somenzi, Trivedi, Wojtczak, *Good-for-MDPs Automata*,
TACAS 2020 (GFM = the soundness condition for the automaton×MDP product).

**Reuse.** Essentially everything: the LTL→LDBA is delegated to **Spot** (shell-out +
HOA parse) behind an `IMPACT_HAVE_SPOT` flag; the product with the IMDP is just another
IMDP; and the robust accepting-EC / Büchi solve is IMPaCT's existing
`omega::robustBuchiWinningStates` / `maxBuchiPessimistic` (the hard, oracle-validated part,
ISSUE-0009). Only the LDBA×IMDP product construction is new.

**Correctness / oracle.** Robust ω-regular on intervals is IMPaCT-only, so validate the
automaton+product on *ordinary* MDPs against **Storm/PRISM full LTL** (degenerate
intervals), then rely on the already-validated robust-Büchi core for the interval case.

**Effort:** medium-high (mostly integration). **Value:** high (extends the existing
IMPaCT-only ω-regular capability from the F/G/U/X/GF/FG fragment to arbitrary LTL).

---

## Sequencing

1. ✅ Expected reward (done — lands the reward plumbing).
2. Long-run average (reuses reward plumbing + MEC code) — flagship, fully IMPaCT-only.
3. Full LTL via Spot (reuses the ω-regular core; mostly integration).
4. Multi-objective (new Pareto layer over the scalarised O-max VI).

Each is a self-contained VI variant over the same O-maximization inner solve, validated
against an analytic value + the peer tool on its native query (or endpoint/ordinary-MDP
sanity where no peer does the robust-interval version).
