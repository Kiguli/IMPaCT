# IMPaCT 2.0: Sound, Sparse, ω-Regular Controller Synthesis for Continuous-State Stochastic Systems

*Evolving CAV draft. Every cited result is a verified entry in [`References.bib`](References.bib);
every algorithm is backed by an immutable test in [`../tests/`](../tests). The development
log and rationale live in [`IMPaCT-v2.md`](IMPaCT-v2.md); open/closed issues in
[`../issues/`](../issues).*

## Abstract (draft)
We present IMPaCT 2.0, a tool for sound controller synthesis and verification of
continuous-state, discrete-time stochastic systems against temporal specifications,
via abstraction to Interval Markov Decision Processes (IMDPs). IMPaCT 2.0 extends the
reach-avoid capability of IMPaCT with (i) a **sparse** abstraction and solver whose
memory is linear in the number of stored transitions (as in IntervalMDP.jl) rather
than quadratic in the state count, enabling large state spaces that previously
exhausted memory; (ii) **co-safe LTL / LTLf** synthesis via an automaton product; and
(iii) **ω-regular (Büchi)** objectives via accepting end-component reachability.
The solver offers selectable, literature-backed methods (optimistic value iteration;
interval iteration with end-component collapse). Every algorithm is validated by
independent differential oracles, and the abstraction soundness is checked end-to-end
by Monte-Carlo simulation of the closed-loop continuous system — a process that
uncovered and fixed two abstraction soundness bugs.

## 1. Introduction & contributions
IMPaCT [wooding2024impact] abstracts a continuous-state stochastic system into an IMDP
and synthesizes robust controllers for reach-avoid / safety. Two limitations motivate
this work: dense transition matrices make even small case studies exhaust memory
(8.84 GB on the smallest shipped 2-D example; ISSUE-0006), and the specification class
is limited to □S / ◇T / S U T.

Contributions:
1. **Sparse abstraction + O(nnz) synthesis** for affine and (via interval arithmetic)
   nonlinear, diagonal-Gaussian systems (§5). 125k states synthesize in 0.4 s / 255 MB
   where a dense matrix would need ~250 GB.
2. **Sound robust solving with selectable methods** (§3): optimistic value iteration
   [Hartmanns2020optimisticVI] and end-component-collapse interval iteration
   [HaddadMonmege2018IntervalIteration, Baier2017IntervalIteration], with the inner
   robust Bellman solve in closed form [Givan2000BMDP, Lahijanian2015FormalVerification].
3. **Co-safe LTL / LTLf synthesis** (§4) via an LTLf→DFA construction and an IMDP×DFA
   product reducing to robust reachability [KupfermanVardi2001SafetyMC,
   DeGiacomoVardi2013LTLf, LacerdaParkerHawes2015].
4. **ω-regular (Büchi) synthesis** (§6) via accepting-end-component reachability
   [deAlfaro1997formal, ChatterjeeHenzinger2011], the LDBA / good-for-MDP route
   [SickertEsparzaJaaxKretinsky2016, HahnPerezSchewe2020GoodForMDPs].
5. **A verification methodology** (§7): definition-based differential oracles per
   algorithm + end-to-end Monte-Carlo validation of abstraction soundness.

## 2. Preliminaries
Interval MDP, abstraction of a discrete-time stochastic system x[k+1]=f(x[k],u[k])+w[k]
into an IMDP over a finite partition; robust (controller-vs-nature) semantics; end
components [deAlfaro1997formal, CourcoubetisYannakakis1995]; ω-automata. (To expand.)

## 3. Sound robust reachability
Robust max-reachability V*(s) = max over the controller, min (pessimistic) / max
(optimistic) over nature within the per-(s,a) interval ambiguity set, of the reach
probability. The inner robust Bellman solve is the closed-form **O-maximization**
sort-and-assign over a box-interval ambiguity set [Givan2000BMDP,
Lahijanian2015FormalVerification] — verified to be exactly IMPaCT's existing "sorted"
synthesis. Two sound solvers are offered:
- **Optimistic value iteration** (default) [Hartmanns2020optimisticVI]: VI-from-below
  lower bound + a guessed upper bound verified inductive (F(U)≤U ⇒ V*≤U by
  Knaster-Tarski). Needs no end-component handling, so it converges on
  nature-confinable end components (the failure mode of plain interval iteration;
  ISSUE-0003).
- **Interval iteration with MEC collapse** [HaddadMonmege2018IntervalIteration,
  Baier2017IntervalIteration]: faster on controller end components.
MEC decomposition follows the standard recompute-SCC-under-staying-actions fixpoint
[ChatterjeeHenzinger2011] (Baier-Katoen Alg. 47); a naive one-pass acceptance is
unsound, as the literature itself notes (ISSUE-0002 records the analysis).

## 4. Co-safe LTL / LTLf synthesis
LTLf formulas are compiled to a DFA by formula progression (Brzozowski-style
derivatives), with states canonicalized **semantically** (truth tables over the
formula's anchor subformulas) so the automaton is finite and minimal — a syntactic
canonicalization explodes on e.g. (G a) U (G a) (ISSUE-0005). The IMDP×DFA product
makes accepting product states absorbing targets, reducing co-safe synthesis to robust
reachability. Refs: [KupfermanVardi2001SafetyMC, DeGiacomoVardi2013LTLf,
LacerdaParkerHawes2015]; Spot/Owl [duretlutz2016spot, kretinsky2018owl] remain optional
back-ends (ROADMAP).

## 5. Sparse abstraction
The transition probability from a source cell to a target box under diagonal Gaussian
noise is, per dimension, the mass of a normal in an interval; its bound over the
source cell's mean range is closed form (unimodal in the mean). The box bound is the
product of per-dimension bounds — exact for axis-decoupled dynamics, sound for coupled
ones. Only successors in the per-dimension kernel window are stored, giving O(nnz)
memory (IntervalMDP.jl-style [MathiesenLahijanianLaurenti2024]). Nonlinear dynamics use
interval arithmetic for the per-dimension mean enclosure. The asymptotic O(N) win holds
at fixed grid step and noise as the domain grows (constant kernel-in-cells). Refs for
the abstraction soundness: [Lahijanian2015FormalVerification, SoudjaniGevaertsAbate2015,
LavaeiKhaledSoudjaniZamani2020].

## 6. ω-regular synthesis
A Büchi objective "visit `accepting` infinitely often" reduces to reachability of
accepting end components: a MEC containing an accepting state is good; the value is the
max reachability of the union of good MECs [deAlfaro1997formal] (Baier-Katoen Ch. 10).
This is the accepting-frontier of the LDBA / good-for-MDP product route
[SickertEsparzaJaaxKretinsky2016, HahnPerezSchewe2020GoodForMDPs]. For the robust
(pessimistic) value, accepting ECs must be robust; the planned methods are
Dutreix-Coogan permanent winning components [Dutreix2018satbounds, Dutreix2022abstraction],
the Weininger et al. game reduction [WeiningerMeggendorferKretinsky2019BMDP], and
Asadi et al. qualitative force-oracle attractors [Asadi2026qualitative] (ISSUE-0009).

## 7. Verification methodology
Two independent layers. **Differential oracles**: each algorithm is checked against a
method-independent ground truth — O-maximization vs brute-force polytope-vertex
enumeration; MEC vs definition-based brute-force end-component enumeration; interval
iteration vs VI-from-below; LTLf→DFA vs a direct finite-trace evaluator; ω-regular vs
reachability reductions. **End-to-end Monte-Carlo**: synthesize a controller, simulate
the closed-loop *continuous* system, and check the empirical satisfaction against the
synthesized bounds (for a robust controller, empirical ≥ the worst-case lower bound).
This closed-loop check found two soundness bugs the differential tests missed — a loose
outside-grid sink (ISSUE-0007) and a double-counted grid-unaligned target (ISSUE-0008) —
both fixed.

## 8. Experimental evaluation
- **Sparse scaling** (`benchmarks/sparse_scaling.cpp`): 1-D reach, domain grown at
  fixed step/noise — 125,000 states, 255 MB, 0.42 s synthesis, vs ~250 GB dense.
- **Nonlinear ARCH Van der Pol** (`benchmarks/validate_vp.cpp`): sparse interval-MC of
  the standard VP verification system; 6/6 start states have empirical reach within the
  synthesized [lower, upper].
- **Co-safe over a continuous robot — Package Delivery** (`benchmarks/validate_cosafe.cpp`):
  F(pickup ∧ F deliver); 4/4 start states have empirical satisfaction ≥ the robust
  lower bound.
(To add: full ARCH-COMP Stochastic-Models suite once nonlinear coverage + the v1 path
or sparse-v1 wiring lands; cross-checks vs IntervalMDP.jl / PRISM / Storm.)

## 9. Related work
IMPaCT [wooding2024impact]; FAUST² [SoudjaniGevaertsAbate2015]; StocHy [CauchiAbate2019];
AMYTISS [LavaeiKhaledSoudjaniZamani2020]; SySCoRe [vanHuijgevoortSchonSoudjaniHaesaert2023];
IntervalMDP.jl [MathiesenLahijanianLaurenti2024]; Dutreix-Coogan ω-regular IMC/BMDP
[Dutreix2018satbounds, Dutreix2021specguided, Dutreix2022abstraction]. The differentiator:
sparse + GPU-amenable abstraction with co-safe/ω-regular synthesis and a Monte-Carlo
soundness-validation methodology.

## 10. Conclusion
(To write.)
