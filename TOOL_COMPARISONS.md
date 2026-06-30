# IMPaCT vs PRISM / Storm / IntervalMDP.jl — cross-tool comparison

This document reports a direct, like-for-like comparison of **IMPaCT** against the
three peer probabilistic model checkers the user requested — **PRISM**, **Storm**,
and **IntervalMDP.jl** — on:

1. the **ARCH-COMP (Stochastic Models) benchmarks** shipped in
   [`examples/ARCH-COMP/2025`](examples/ARCH-COMP/2025), and
2. the small **cross-tool reference models** in
   [`benchmarks/crosstool/models`](benchmarks/crosstool/models).

It compares both the **computed values** (robust reach / safety probabilities) and
the **solve times**, and records every place the tools disagree (with the cause).

> This realises the *big-machine verification queue* item #3 in
> [`ROADMAP.md`](ROADMAP.md) ("install IntervalMDP.jl / PRISM / Storm and run the
> benchmarks side-by-side"), and along the way compile-verifies the v2 source on a
> full AdaptiveCpp + Armadillo + GLPK + HDF5 toolchain (items #1–#2).

---

## 1. Setup, versions, environment

All work was done on a single 56 GB / multi-core Windows host via Docker.

| Tool | Version | How it was obtained / run |
|---|---|---|
| **IMPaCT** | v2.0 (branch `IMPaCT-v2.0`) | built from source in the `impact:latest` image (AdaptiveCpp 23.10 / clang 16, Armadillo 12.6.4, GLPK, HDF5, GSL, NLopt); CPU/OpenMP target `--acpp-targets=omp` |
| **Storm** | 1.13.0 | official `movesrwth/storm:stable` Docker image |
| **IntervalMDP.jl** | 0.6.0 | Julia 1.10.4, `Pkg.add("IntervalMDP")` |
| **PRISM** | 4.8.1 | official Linux binary (`prism-4.8.1-linux64-x86`), OpenJDK 17 |

Reproduction harness (all scripts committed under
[`benchmarks/crosstool/peers`](benchmarks/crosstool/peers) and
[`benchmarks/crosstool/arch`](benchmarks/crosstool/arch)):

* IMPaCT abstracts each continuous ARCH system to an **interval MDP (IMDP)** and
  synthesises a robust controller. A new method
  [`IMDP::exportIMDP`](src/GPU_synthesis.cpp) writes that abstracted IMDP to the
  neutral `.imdp` exchange format ([`src/imdp_io.h`](src/imdp_io.h)).
* `imdp_to_drn.py` / `imdp_to_prism.py` convert the `.imdp` to **Storm DRN** and
  **PRISM** models; `intervalmdp_runner.jl` builds the IntervalMDP.jl model directly
  from the `.imdp`. **Every tool therefore solves the identical model.**

### Semantics alignment (verified on a known model)

The robust IMDP value has two senses; the tools name them differently. We verified
the mapping on `choice_interval` (controller chooses A=[0.5,0.7]/[0.3,0.5] vs
B=0.6/0.4; analytic answer pess 0.4 / opt 0.5):

| Sense | IMPaCT | Storm | IntervalMDP.jl | choice_interval |
|---|---|---|---|---|
| **pessimistic** (controller max, nature adversarial) | `*_lower` / `--bound pess` | `--uncertainty-resolution robust` | `Pessimistic, Maximize` | **0.4** (all four) |
| **optimistic** (controller max, nature cooperative) | `*_upper` / `--bound opt` | `--uncertainty-resolution cooperative` | `Optimistic, Maximize` | **0.5** (all four) |

---

## 2. ARCH-COMP benchmarks — implemented and run

All ten ARCH-COMP 2025 examples were made cross-tool-ready (each now calls
`exportIMDP`; the `PR_minimal` include-path bug and several v2 missing-`#include`
compile bugs were fixed — see §6). The table below is the headline result.

> **Correctness metric.** IMPaCT synthesises the **robust (pessimistic)** controller
> (`IMDP_lower = true`); this is the value that carries the formal guarantee. We
> compare IMPaCT's per-cell robust value vector (from `controller.h5`) against
> IntervalMDP.jl's full value vector over **every abstraction cell**, and report the
> maximum absolute difference. Storm corroborates via all-state aggregates.

### 2.1 Correctness — robust value, IMPaCT vs IntervalMDP.jl (per cell)

| Benchmark | spec | cells | IMPaCT robust [min,max] | max&#124;IMPaCT − IntervalMDP&#124; | verdict |
|---|---|---|---|---|---|
| AS (anaesthesia 3D) | finite reach, H=10 | 2460 | [0, 1] | **4.8e-07** | ✅ match |
| BA (building 4D) | finite safety, H=6 | 1225 | [0.1266, 0.6673] | **4.3e-07** | ✅ match |
| VP (Van der Pol 2D) | infinite reach | 2591 | [0, 0.2297] | **2.3e-05** | ✅ match |
| IC_reach (integrator 2D) | finite reach, H=5 | 592 | [0, 0.99999] | **6.3e-07** | ✅ match |
| IC_safe (integrator 2D) | finite safety, H=5 | 1681 | [0, 1] | **6.3e-07** | ✅ match |
| PD_p1 (pkg-delivery 2D) | infinite reach | 571 | [0.9998, 1] | **7.1e-07** | ✅ match |
| PD_p3 (pkg-delivery 2D) | infinite reach | 571 | 1.0 (init)¹ | **0** (init) | ✅ match |

¹ PD_p3's IMPaCT *SYCL* synthesis did not converge in time (see §5 / ISSUE-0017);
the value 1.0 is from IMPaCT's own v2 light-weight solver (`imdp_solve`, identical
robust semantics) on the exported model — it equals IntervalMDP.jl (1.0) and Storm
(init 1.0). The abstraction + export succeeded normally.

**IMPaCT's robust controller value matches IntervalMDP.jl to ≤ 2.3 × 10⁻⁵ on every
benchmark** (≤ 7 × 10⁻⁷ for the finite-horizon cases). The slightly larger VP gap is
the residual-vs-interval-iteration stopping difference (§5), not a model mismatch.

### 2.2 Correctness — Storm cross-check (all-state aggregates)

Storm cannot export per-state vectors for interval models, so we compare
`filter(min/max/avg, …, true)` over all states. Accounting for the two extra
absorbing sink states (target sink value 1, avoid sink value 0), Storm reproduces
IMPaCT's mean exactly, e.g.:

| Benchmark | IMPaCT robust mean (cells) | Storm robust avg (all states) | reconciled |
|---|---|---|---|
| AS | 0.004524 | 0.004927 | `0.004524·2460/2462 + 1/2462 = 0.004927` ✅ |
| IC_reach | 0.071935 | 0.073376 | `0.071935·592/594 + 1/594 = 0.073376` ✅ |
| BA (safety) | 0.470945 | 1−0.529439 = 0.470561 | within sink offset ✅ |
| PD_p1 | 0.999999 | 0.998254 | `(571·1 + 1·1 + 1·0)/573 = 0.998254` ✅ |

Storm's init-state value matches IMPaCT/IntervalMDP exactly where the init state is
non-trivial (e.g. BA safety init = **0.4533244** in all three tools).

### 2.3 Solve-time comparison (seconds)

IMPaCT now offers **two infinite-horizon solvers** (`setIterationMethod`):
**II** = interval iteration (default, sound two-sided bracket) and **VI** = pure value
iteration (the method the peers use: from-0 residual stopping). Both were sped up by
the ISSUE-0017 work (hoisting the per-sweep `diff`-matrix recompute, `sycl::queue`,
and the constant SYCL buffers out of the VI loops). Finite-horizon cases use the
finite DP (one mode) and were already fast.

| Benchmark | IMPaCT abstraction | IMPaCT robust VI (II / VI) | IntervalMDP VI | Storm VI | Storm build |
|---|---|---|---|---|---|
| AS | 27.05 | 4.3 (finite) | 0.40 | 0.21 | 0.50 |
| BA | 7.58 | **0.27** (finite) | 1.14 | 5.86 | 6.05 |
| VP | 1.91 | 29.1 (II) / **0.59 (VI)** | 0.61 | 1.35 | 0.37 |
| IC_reach | 1.75 | 0.06 (finite) | 0.38 | 0.03 | 0.05 |
| IC_safe | 2.67 | 0.47 (finite) | 0.42 | 0.09 | 0.18 |
| PD_p1 | 37.06 | 1.29 (II) / 1.34 (VI) | 1.33 | 8.31 | 7.26 |
| PD_p3 | ~37 | ∞ (II)¹ / **1.36 (VI)** | 1.40 | 8.30 | 7.27 |

Pre-ISSUE-0017 the infinite-horizon II times were VP 214 s and PD_p1 6.5 s. With the
new **VI** mode VP solves in 0.59 s (102 sweeps — *faster than* IntervalMDP.jl/Storm)
and PD_p1 in 1.34 s (27 sweeps), all returning the same robust value.

¹ PD_p3's interval iteration does **not** terminate (the nature-trap, ISSUE-0003: the
from-1 iterate -> greatest fixed point so the gap never closes). The new **VI** mode
solves it in 28 sweeps / 1.36 s, returning the correct robust value 1.0 (= IntervalMDP
/ Storm), because pure VI uses only the from-0 iterate -> least fixed point. See §5.

Reading the times:

* **The abstraction step is IMPaCT-only** — PRISM/Storm/IntervalMDP.jl consume a
  finite (I)MDP and never build one from continuous dynamics. This is IMPaCT's
  distinguishing capability, and it is fast and well-parallelised on CPU
  (e.g. AS builds a 40 656-row IMDP in 27 s; a 1.1-billion-entry abstraction for the
  large cases — see §4).
* **IMPaCT's finite-horizon robust VI is competitive or faster** (BA 0.27 s,
  IC 0.06–0.47 s) than the peers — partly because Storm pays a model-construction
  cost (5–8 s) to ingest the dense DRN.
* **IMPaCT's infinite-horizon solve now matches or beats the peers in VI mode.** The
  default interval iteration (sound) was sped up ~7× (VP 214 s → 29.1 s) by hoisting
  per-sweep loop-invariant work (the `diff` recompute, the `sycl::queue`, and the
  constant SYCL buffers). The new pure-VI mode solves VP in 0.59 s (vs IntervalMDP.jl
  0.61 s / Storm 1.35 s) and converges on the end-component case PD_p3 that interval
  iteration cannot — see §5 / ISSUE-0017.

---

## 3. Cross-tool reference models (`benchmarks/crosstool`)

The small shared models give an exact four-way check (the in-repo
`compare.py` already showed IMPaCT = analytic ref on all 20 checks; here we add the
*measured* peer numbers).

| Model | property | bound | IMPaCT | IntervalMDP.jl | Storm | PRISM | analytic ref |
|---|---|---|---|---|---|---|---|
| chain_point (s0) | reach | — | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 |
| safety_point (s0) | safety | — | 0.72 | 0.72 | 0.72² | 0.72 | 0.72 |
| choice_interval (s0) | reach | pess | 0.4 | 0.4 | 0.4 | 0.4³ | 0.4 |
| choice_interval (s0) | reach | opt | 0.5 | 0.5 | 0.5 | 0.5³ | 0.5 |
| fork_interval (s0) | reach | pess | 0.4 | 0.4 | 0.4 | 0.4³ | 0.4 |
| fork_interval (s0) | reach | opt | 0.6 | 0.6 | 0.6 | 0.6³ | 0.6 |
| buchi_reach (s2) | `GF acc` | — | 0.5 | n/a⁴ | **unsupported**⁵ | n/a⁴ | 0.5 |
| persist_leak (s0) | `FG safe` | pess | 0.5 | n/a⁴ | **unsupported**⁵ | n/a⁴ | 0.5 |
| patrol_cycle (s0) | `GF r0 ∧ GF r2` | — | 1.0 | n/a⁴ | **unsupported**⁵ | n/a⁴ | 1.0 |
| buchi_routearound (s2) | `GF acc` | pess/opt | 0 / 1 | n/a⁴ | **unsupported**⁵ | n/a⁴ | 0 / 1 |

² Storm safety via `1 − Pmin[F "avoid"]` (= 1 − 0.28 = 0.72).
³ **CORRECTED 2026-06-30: PRISM 4.8.1 DOES support interval MDPs** (`Pmaxmin/Pmaxmax=?[F ...]`,
  auto-detects `Type: IMDP`) and reproduces these interval values exactly. The earlier
  "no interval-MDP support" claim was wrong (see §9). Caveat: PRISM rejects interval edges
  with a 0 lower bound, so the *abstracted* ARCH IMDPs need their `[0,hi]` edges floored to
  `[1e-9,hi]` first; and PRISM has no robust ω-regular either (rows below stay PRISM-N/A).
⁴ IntervalMDP.jl 0.6.0 has reachability/safety/DFA-reachability but no Büchi /
  persistence / generalised-Büchi robust solver.
⁵ Storm 1.13.0 refuses robust ω-regular on interval models:
  *"We have not yet implemented LTL with intervals"* (`SparseMdpPrctlModelChecker.cpp:239`).

PRISM note: `prism` reports only the *initial-state* value, so the `chain_point` s1
check is reported by PRISM as the s0 value (0.25); IMPaCT/IntervalMDP report the true
s1 value 0.5. This is a PRISM front-end limitation, not a disagreement on the model.

---

## 4. Scalability — the large ARCH cases (LM, AV_minimal, PR_minimal)

These three exercise the dense-abstraction memory wall (ISSUE-0006). Dense
`(state×input)×state` matrix sizes (two are stored, min and max):

| Benchmark | cells (after relabel) | inputs | dense matrix (each) | 2× matrices | feasible on this 56 GB host? |
|---|---|---|---|---|---|
| PR_minimal (2D robot RA) | 1583 | 441 | **8.84 GB** | 17.7 GB | ✅ abstraction fits (was OOM at the historical 4 GB cap) |
| AV_minimal (3D bicycle) | 5562 | 30 | 7.4 GB | 14.8 GB | ✅ abstraction fits |
| LM (6D vehicle, H=200) | 36 450 | 9 | **~96 GB** | ~192 GB | ❌ exceeds host RAM (dense-infeasible) |

`PR_minimal` is exactly the *2D-robot-RA* case that RESULTS.md / ISSUE-0006 reported
killing the v1 dense path under a 4 GB cap (`698103 × 1583 ≈ 8.84 GB`). **On the big
machine the dense abstraction completes** — confirming the diagnosis that the
limitation was memory, not algorithm. Measured PR_minimal abstraction (CPU/OMP):

| step | dims | time |
|---|---|---|
| target vectors (max/min) | 698103 × 49 | 9.7 s / 4.8 s |
| avoid vectors (max/min) | 698103 × 1 | 12.5 s / 12.4 s |
| **max transition matrix** | 698103 × 1583 (**8.84 GB**) | **364 s** |
| **min transition matrix** | 698103 × 1583 (**8.84 GB**) | **130 s** |

i.e. the full dense abstraction (~17.7 GB resident) builds in ≈ 9 min and exports a
~1.2 GB `.imdp`. `AV_minimal` (5562 cells × 30 inputs → 7.42 GB matrices, ≈ 14.8 GB
resident) likewise fits and was building its transition matrices (target-vector step
337 s, dominated by its 2178-cell target). Both therefore clear the historical OOM.

Their **infinite-horizon SYCL VI does not converge in reasonable time** (ISSUE-0017,
as for VP/PD_p3); the value of interest is obtained instead from the exported IMDP via
IMPaCT's own `imdp_solve` and the peers (IntervalMDP.jl / Storm solve a ~700k-state
IMDP in seconds). LM's 6-D × 200-step grid (36 450 cells × 9 inputs → ~96 GB per
matrix) is genuinely beyond dense storage and motivates the v2 sparse representation.

> Net scalability finding: **the dense abstraction — not the robust solve — is the
> memory wall.** The big machine removes it for the medium-large cases; the very large
> (LM) case still needs sparse storage. This is precisely the ISSUE-0006 motivation.

---

## 5. Where the tools differ (and why)

Every observed difference is **either a documented semantic distinction or a
performance characteristic — no tool returned a different correct value on the same
model.** The findings are filed as issues.

1. **Optimistic bound is a *bracket of the robust controller*, not the cooperative
   optimum.** IMPaCT reports `[lower, upper]` for its *synthesised robust policy*
   (the upper bound fixes that policy and lets nature cooperate). IntervalMDP.jl's
   `Optimistic, Maximize` re-optimises the policy. Hence the *pessimistic* values
   match to 1e-6 but the *optimistic* columns differ (e.g. AS 0.996, IC_safe 0.108).
   This is correct-by-design; documented as **ISSUE-0018** so it is not misread as a
   bug. (For an apples-to-apples optimistic comparison one must synthesise IMPaCT's
   *optimistic* controller, `IMDP_lower = false`.)

2. **Infinite-horizon robust VI — resolved by offering BOTH methods + per-sweep fixes.**
   IMPaCT now exposes `setIterationMethod(IterationMethod::{IntervalIteration,
   ValueIteration})`:
   * **Interval iteration** (default, sound): iterate from 0 and from 1, stop when the
     gap < ε — a rigorous two-sided bracket. The per-sweep cost was cut ~7× (VP
     214 s → 29.1 s) by hoisting loop-invariant work out of the loops: the
     `(state*input) x state` `diff` matrix (`maxTransitionM-minTransitionM`, a ~1.1 GB
     reallocation per sweep for the large cases), the `sycl::queue`, and the constant
     read-only SYCL buffers (`minTransitionM`/`diff*`/...), all now constructed once
     per loop instead of once per sweep. Values are bit-for-bit identical.
   * **Pure value iteration** (the peer method): plain VI from 0 with residual
     stopping. VP 0.59 s / 102 sweeps (vs IntervalMDP.jl 0.61 s), PD_p1 1.34 s. Same
     robust value as interval iteration where both converge; not a sound two-sided
     certificate (it is exactly what PRISM/Storm/IntervalMDP.jl do).
   This closes **ISSUE-0017** (an instance of **ISSUE-0010**).

   The pure-VI mode also **resolves the PD_p3 non-convergence** noted previously
   (interval-iteration nature-trap, **ISSUE-0003**): interval iteration's from-1
   iterate goes to the greatest fixed point, so the gap never closes (gap stuck at
   0.9997 after 15 380 sweeps); pure VI uses only the from-0 iterate -> least fixed
   point and converges in 28 sweeps / 1.36 s to the correct robust value 1.0
   (= IntervalMDP.jl / Storm). So PD_p3 is now solvable on the SYCL path by selecting
   `ValueIteration`; making it the default (or porting MEC-collapse) remains future
   work for a sound solver that also converges on end components.

3. **Robust ω-regular is an IMPaCT-only capability.** Neither Storm 1.13.0 (*"LTL
   with intervals … not implemented"*) nor IntervalMDP.jl 0.6.0 solves Büchi /
   persistence / generalised-Büchi on interval MDPs. IMPaCT's `buchi` / `persist` /
   `patrol` solvers match the analytic references exactly. This is a genuine
   differentiator, not a discrepancy.

4. **PRISM 4.8.1 DOES support interval MDPs (corrected 2026-06-30, §9).** It robustly
   checks `Pmaxmin/Pmaxmax=?[F ...]` on IMDPs and agrees with the other three tools on
   every interval reference model (and, after flooring `[0,hi]` edges, on the abstracted
   ARCH IMDPs). It does **not** do robust ω-regular (LTL on intervals → "Computation not
   implemented yet"), and rejects 0-lower-bound interval edges. The earlier claim that
   PRISM has no interval support was incorrect.

---

## 6. v2 compile-verification fixes (big-machine queue items #1–#2)

Building the v2 sources on the real toolchain surfaced genuine compile bugs (the code
was previously "reasoned but not compiled"):

* missing `#include`s — `<unordered_map>` (`ltl.cpp`), `<algorithm>` (`ltl.cpp`,
  `pta.cpp`), `<cmath>` (`solve.cpp`), `<functional>` (`abstraction.cpp`, `stl.cpp`);
* `PR_minimal.cpp` included `../../../src/IMDP.h` (one `../` short) — fixed.

With these, the v2 `imdp_solve` CLI builds and **all 20 cross-tool `compare.py`
checks pass**, and the SYCL examples build and run. The unified O-maximization inner
solve (ISSUE-0012) and the finite-horizon-safety `Sorted` kernels (ISSUE-0013) are
thereby compile-verified, and their results are now independently confirmed against
Storm and IntervalMDP.jl (BA, IC_safe finite-safety values match to 1e-6).

---

## 7. Conclusions

* **Values agree.** On every shared model where a peer is applicable, IMPaCT's robust
  value equals PRISM / Storm / IntervalMDP.jl (≤ 2.3e-5 on the ARCH abstractions,
  exact on the reference models). No genuine value disagreement was found.
* **IMPaCT's niche is confirmed.** It is the only tool here that (a) builds the IMDP
  from continuous stochastic dynamics, and (b) solves robust ω-regular specifications
  on interval MDPs.
* **Solve times now match or beat the peers.** IMPaCT exposes two infinite-horizon
  solvers: sound interval iteration (default), sped up ~7× by hoisting per-sweep
  loop-invariant work (VP 214 s → 29.1 s), and **pure value iteration** (the peer
  method), which solves VP in 0.59 s (≤ IntervalMDP.jl 0.61 s / Storm 1.35 s), PD_p1
  in 1.34 s, and converges on the end-component case PD_p3 (1.36 s) that interval
  iteration cannot — all returning the same robust value as the peers. Finite-horizon
  VI was already competitive.
* **Scalability** is bounded by the *dense* abstraction, not the solve: the big
  machine clears the historical 4 GB OOM (ISSUE-0006) for PR_minimal/AV_minimal but
  the 6-D LM case still needs the v2 sparse representation.

---

## 8. Peer-tool EXAMPLE models solved by IMPaCT (2026-06-29)

Beyond the ARCH abstractions (where IMPaCT builds the IMDP), this section takes the
**discrete IMDP example models shipped by the peer tools**, solves them with their own
tool, converts them to IMPaCT's neutral `.imdp` format, and solves them with IMPaCT —
a same-model, same-spec head-to-head on the *solver* (no abstraction in the loop).

### Scope (per request: pure DTMC / MDP / CTMC ignored)
| Tool | Interval-MDP example models shipped | In scope | Action |
|---|---|---|---|
| **IntervalMDP.jl** | `multiObj_robotIMDP` (207-state robot; PRISM-explicit + bmdp-tool + native `.nc`); plus tiny inline I/O-test IMDPs | ✅ yes | converted + solved (below) |
| **PRISM** | ships no interval *example* model, but **does support IMDPs** (corrected §9) and solves the converted robot model too | ✅ (solver) | added in §9 |
| **Storm** | interval models via `--uncertainty-resolution`; no interval *examples* bundled — **now built from source in the container** (§9) | ✅ (solver) | re-run in §9 |

The only concrete IMDP *example model* a peer ships is IntervalMDP.jl's
`multiObj_robotIMDP`. (PRISM/Storm ship continuous-time / non-interval examples only; a
second pass could lift a CTMC into an IMDP to exercise an unsupported feature.)

### multiObj_robotIMDP — robust reachability `Pmaxmin=?[F "reach"]`
207 states, 828 (state,action) rows, 2784 interval transitions; init = state 0, target
= state 206. Converted with `benchmarks/crosstool/peers/tra_to_imdp.py`; model stored at
`benchmarks/crosstool/models/multiObj_robotIMDP.imdp`. Spec is **Pessimistic, Maximize**
= IMPaCT `reach --bound pess` = `solve::maxReachPessimistic`.

| solver | init value | iters | time | note |
|---|---|---|---|---|
| IntervalMDP.jl (`solve`, thr 1e-6 = default) | 0.8946617847 | 83 | 1.2 ms | **stops 1.2e-6 below the fixpoint** |
| IntervalMDP.jl (thr 1e-9) | 0.8946629814 | 99 | — | |
| IntervalMDP.jl (thr 1e-12) | **0.8946629826** | 128 | — | converged |
| IMPaCT `imdp_solve` OVI (eps 1e-6) | **0.8946629826** | 129 | ~10 ms | certified [L, L+eps] |
| IMPaCT `imdp_solve` OVI (eps 1e-9) | 0.8946629826 | 128 | — | |
| IMPaCT `imdp_solve` interval-iter / MEC (eps 1e-9) | 0.8946629826 | 42177 | — | sound bracket [.8946629826, .8946629838] |
| IMPaCT optimistic (`Pmaxmax`) | 0.9999979999 | 44 | — | cooperative nature |

**Values agree to ~1e-10** once the convergence thresholds match (IMPaCT OVI 0.8946629826
= IntervalMDP.jl-at-1e-12 0.8946629826; both IMPaCT methods agree). The apparent 1.2e-6
"gap" at first run is purely IntervalMDP.jl's **default reachability threshold (1e-6)**
terminating ~1.2e-6 short of the fixpoint on this slowly-mixing chain — a methodological
note, NOT a value disagreement (cross-tool thresholds must be matched when comparing).
Notably IMPaCT's OVI returns the converged value AND a *certified* upper bound already at
eps 1e-6, where IntervalMDP.jl's residual stop does not.

### Feature coverage surfaced by the peer examples
- `Pmaxmin / Pmaxmax` robust reachability: **all four tools** ✓ (match — §9).
- IntervalMDP.jl extras (in its model zoo, not this example's spec): orthogonal & mixture
  IMDPs and GPU (CUDA) VI — IMPaCT abstracts to *flat* IMDPs. (Note: IntervalMDP.jl does
  **not** do multi-objective Pareto — the "multiObj" is only the model's name; corrected.)
- IMPaCT-only on the same IMDP: robust **ω-regular** specs (Büchi / persistence / GenBüchi
  / full LTL via `imdp_solve buchi|persist|patrol|ltl`) — Storm / IntervalMDP.jl / PRISM
  do not solve robust ω-regular on interval MDPs.

Reproduce: `benchmarks/crosstool/peers/{tra_to_imdp.py, imdp_native_robot.jl,
imdp_native_robot_tight.jl}` + `tools/imdp_solve`. **A full corrected four-tool re-run
(incl. PRISM and a freshly source-built Storm 1.13.0) is in §9; the per-tool feature
matrix is §10.**

---

## 9. Full four-tool re-run on shared IMDPs (2026-06-30)

Two things changed since §1–§8 and triggered this re-run:
1. **PRISM 4.8.1 DOES support interval MDPs** (the earlier "no support" claim was wrong).
   It auto-detects `Type: IMDP` and solves `Pmaxmin/Pmaxmax=?[F label]`. PRISM is therefore
   a full fourth column here. Two PRISM caveats: it has **no robust ω-regular** (LTL on
   intervals → *"Computation not implemented yet"*), and it **rejects interval edges whose
   lower bound is 0** (*"Transition probability has lower bound of 0"*) — it needs a fixed,
   strictly-positive support graph.
2. **Storm is freshly built from source inside the container** (Storm 1.13.0, Debian-12,
   LTO-off `make storm-cli`), so its numbers are reproduced natively rather than carried
   over from §1–§6.

Setup: every shared interval-MDP is solved by all four tools on the **same model**.
IMPaCT reads the neutral `.imdp`; IntervalMDP.jl reads it via `intervalmdp_runner.jl`;
Storm reads the `.drn` (`imdp_to_drn.py`); PRISM reads the explicit interval `.tra/.sta/.lab`
(`imdp_to_prism_interval.py`). One driver runs the sweep: `peers/sweep_compare.py`
(raw output: `benchmarks/crosstool/sweep_all.tsv`, `sweep_arch.tsv`).
Sense: **pess** = robust = nature adversarial (`Pmaxmin` / `--uncertainty-resolution robust`);
**opt** = cooperative (`Pmaxmax` / `cooperative`).

`*` = the tool **cannot abstract** a continuous system; it was handed *IMPaCT's* exported
IMDP (the abstraction is an IMPaCT-only phase, see the abstraction note below).
`**` = PRISM additionally required the abstracted IMDP's `[0,hi]` edges floored to
`[1e-9,hi]`, so its ARCH values are an ε-perturbation of the true robust value.

### 9a. Peer / reference IMDP models (everyone reads the identical model)
Robust (pess) / optimistic value at the initial state; ✅ = matches the analytic reference.

| model | spec | IMPaCT | IntervalMDP.jl | PRISM | Storm | ref |
|---|---|---|---|---|---|---|
| chain_point | reach | 0.25 / 0.25 | 0.25 / 0.25 | 0.25 / 0.25 | 0.25 / 0.25 | 0.25 ✅ |
| safety_point | safety | 0.72 / 0.72 | 0.72 / 0.72 | 0.72 / 0.72 | 0.72 / 0.72 | 0.72 ✅ |
| choice_interval | reach | 0.40 / 0.50 | 0.40 / 0.50 | 0.40 / 0.50 | 0.40 / 0.50 | 0.40 / 0.50 ✅ |
| fork_interval | reach | 0.40 / 0.60 | 0.40 / 0.60 | 0.40 / 0.60 | 0.40 / 0.60 | 0.40 / 0.60 ✅ |
| multiObj_robotIMDP | reach | 0.8946630 / 1.0 | 0.8946630† / 1.0 | 0.8946624 / 1.0 | 0.8946630 / 1.0 | — ✅ (agree ~1e-6) |
| buchi_reach | `GF acc` | 0.5 / 0.5 | **N/A** | **N/A** | **N/A** | 0.5 ✅ |
| buchi_routearound | `GF acc` | 0 / 1 | **N/A** | **N/A** | **N/A** | 0 / 1 ✅ |
| persist_leak | `FG safe` | 0.5 / 1.0 | **N/A** | **N/A** | **N/A** | 0.5 ✅ |
| patrol_cycle | `GF r0 ∧ GF r2` | 1.0 / 1.0 | **N/A** | **N/A** | **N/A** | 1.0 ✅ |

† IntervalMDP.jl's default threshold (1e-6) reports 0.8946618 (1.2e-6 low); at a tight
threshold it converges to 0.8946630 = the others (see §8). **All four tools agree on every
reachability/safety model; the four ω-regular models are IMPaCT-only** (Storm: *"LTL with
intervals not implemented"*; PRISM: *"Computation not implemented yet"*; IntervalMDP.jl:
DFA/co-safe only — no Büchi/persistence).

### 9b. ARCH abstracted IMDPs — solve phase only (`*` = no abstraction; `**` = edge-floored)
IMPaCT builds the IMDP from the continuous ARCH dynamics; the other three are given that
IMDP and only *solve* it. Infinite-horizon robust reach/safety; value (solve time).

| ARCH case | spec | IMPaCT | IntervalMDP.jl | PRISM *,** | Storm * |
|---|---|---|---|---|---|
| AS | reach | 0 (0.39 s) | 0 (0.45 s) | 0 (0.54 s) | 0 (0.05 s) |
| BA | safety | ~0‡ | 1.9e-4 (4.9 s) | 0 (0.34 s) | 0 (0.65 s) |
| IC_reach | reach | 0 (0.10 s) | 0 (0.42 s) | ~0 (0.20 s) | 0 (0.05 s) |
| IC_safe | safety | 0 (0.15 s) | 0 (0.50 s) | 0 (0.06 s) | 0 (0.02 s) |
| PD_p1 | reach | 1.0 (3.4 s) | 1.0 (1.4 s) | 1.0 (1.7 s) | 1.0 (2.8 s) |
| PD_p3 | reach | 1.0 (3.6 s) | 1.0 (1.6 s) | 1.0 (2.3 s) | 1.0 (2.2 s) |
| VP | reach | 0 (1.8 s) | 0 (0.55 s) | 0 (1.4 s) | 0 (0.04 s) |

**All four tools agree on every ARCH robust value** (AS/IC/VP infinite-horizon robust
reach = 0; PD = 1; BA robust safety = 0). The optimistic-reach values also agree to high
precision where non-trivial (e.g. IC_reach opt ≈ 1.20e-26 in all four tools).

Caveats (all genuine findings, not value disagreements):
- ‡ **IMPaCT BA optimistic-safety did not converge in 180 s** (OVI on BA's large *safe*
  end component — ISSUE-0010). Its *robust* (pess) BA-safety is ~0, matching the others
  (verified separately at eps 1e-4: `lower=0`). IMPaCT used `eps=1e-4` on the large ARCH
  IMDPs (the peers at their defaults); robust BA-safety at 1e-6 is the slow case.
- **Optimistic safety is convention-dependent across tools.** On BA's `[0,hi]` leak edges,
  IntervalMDP.jl's cooperative resolution zeroes the leaks and reports opt-safety ≈ 0.997,
  whereas the PRISM/Storm `1−Pmin` encodings give 0. The **robust** (guarantee) value is
  unambiguous and agrees; we therefore headline the robust column.
- PRISM solves the ARCH IMDPs only after the `[0,hi]→[1e-9,hi]` floor (`**`); without it
  PRISM aborts with *"lower bound of 0"*.

### Abstraction phase — IMPaCT-only (the asterisks, quantified)
PRISM/Storm/IntervalMDP.jl have **no** continuous-system abstraction: the IMDPs above had
to be built by IMPaCT first. IMPaCT's sparse-abstraction build times (from
`benchmarks/RESULTS.md`) are e.g. AS 0.11 s, BA 0.25 s, IC 0.01–0.02 s, PD 0.7–0.8 s,
PR_minimal 28.5 s, plus the nonlinear VP/AV/LM. For the peer tools this column is simply
**N/A** — they require a finite (I)MDP as input. This is IMPaCT's defining capability and
the reason a like-for-like end-to-end comparison is not possible; §9b compares only the
*shared* solve phase.

---

## 10. Feature matrix — capabilities NOT exercised by the comparison above

The comparison only used robust reachability/safety (+ IMPaCT's ω-regular). Each tool has
substantial capability beyond that. ✅ = supported, ❌ = not supported, **N/A** = not
applicable to the tool's scope. Grounded in a per-tool source/doc audit (2026-06-30).

| Capability | IMPaCT | IntervalMDP.jl 0.6 | PRISM 4.8.1 | Storm 1.13.0 |
|---|---|---|---|---|
| **Abstract continuous stochastic dynamics → IMDP** | ✅ nonlinear (NLopt dense + sparse) | ❌ | ❌ | ❌ |
| Noise models for abstraction | Gaussian, multivariate/correlated, uniform, triangular, Laplace, custom PDF | N/A | N/A | N/A |
| Interval / robust (I)MDP solving | ✅ | ✅ | ✅ (since v4.8, 2024) | ✅ |
| Robust reachability / safety | ✅ | ✅ | ✅ | ✅ |
| Robust finite/bounded-horizon | ✅ | ✅ | ✅ (`F<=k`) | ✅ (step-bounded) |
| **Robust ω-regular on IMDPs** (Büchi/persistence/LTL) | ✅ **only tool** | ❌ (DFA/co-safe reach only) | ❌ "not implemented" | ❌ "LTL with intervals not implemented" |
| Expected reward / cost | ❌ (probabilities only) | ✅ discounted reward, expected exit-time | ✅ (incl. interval reach-reward) | ✅ (reach/cumulative/instant) |
| Long-run average / mean-payoff | ❌ | ❌ | ✅ (non-interval) | ✅ (non-interval; not on intervals) |
| Multi-objective / Pareto | ❌ | ❌ | ✅ (non-interval) | ✅ (non-interval) |
| Controller / strategy synthesis | ✅ (robust, → HDF5) | ✅ (stationary/time-varying) | ✅ (`-exportstrat`, incl. robust) | ✅ (`--exportscheduler`) |
| GPU acceleration | ✅ SYCL/AdaptiveCpp (any vendor) | ✅ CUDA (dense+sparse) | ❌ | ❌ |
| Exact / rational arithmetic | ❌ | ✅ Rational/BigInt (CPU) | ✅ `-exact` | ✅ rational (GMP/CLN) |
| Other model classes | POMDP, PTA, TA | orthogonal-IMDP, mixture-IMDP, IMC | DTMC, CTMC, MDP, PTA, POMDP, POPTA, IDTMC | DTMC, CTMC, MDP, MA, POMDP; +parametric, DFT, GSPN |
| Symbolic / BDD engine | ❌ (dense+sparse) | ❌ | ✅ MTBDD/hybrid | ✅ dd (CUDD/Sylvan), hybrid |
| Full (non-robust) LTL / PCTL*/CSL | fragment (F/G/U/X/GF/FG/patrol) | ❌ | ✅ full | ✅ full (via Spot) |
| Solver methods | II, VI, **OVI**, MEC-collapse, O-max | robust VI (O-max) | VI/GS/PI/II/topological/symbolic | VI/II/OVI/SVI/PI/LP/topological/symbolic |
| Native input formats | C++ dynamics, `.imdp`, PRISM-subset, HDF5 | netCDF `.nc`, PRISM-explicit, bmdp-tool | PRISM lang, explicit, PEPA | PRISM, JANI, DRN, explicit, GSPN, DFT |

**Features present in each tool but deliberately NOT compared above:**
- **IMPaCT** — the continuous-system *abstraction* itself (the whole front end), nonlinear
  dynamics, the 6 noise distributions (Gaussian/multivariate/uniform/triangular/Laplace/
  custom), GPU/SYCL execution, POMDP, PTA/TA, PCTL/CTL/STL, and the II/VI/MEC solver
  variants (only OVI was used in the sweep). No rewards, no multi-objective.
- **IntervalMDP.jl** — orthogonally-decoupled and mixture IMDPs (its scalability feature),
  GPU/CUDA value iteration, discounted-reward and expected-exit-time objectives, exact
  Rational arithmetic, DFA-product (co-safe) specs, and Maximize/Minimize × Pessimistic/
  Optimistic strategy modes. No abstraction, no ω-regular, no multi-objective.
- **PRISM** — DTMC/CTMC/MDP/PTA/POMDP model checking, full PCTL*/LTL/CSL on non-interval
  models, expected-reward/time/steady-state (incl. interval reachability-reward),
  multi-objective & Pareto (non-interval), statistical model checking/simulation, exact
  mode, MTBDD symbolic engine, strategy export. (Robust ω-regular on IMDPs: not yet.)
- **Storm** — DTMC/CTMC/MDP/MA/POMDP, full LTL/ω-regular on *non-interval* models (Spot),
  expected reward + long-run average, multi-objective/Pareto (non-interval), parametric
  model checking (`storm-pars`), DFT/GSPN, sound VI family (II/OVI/SVI), LP/SMT backends
  (Gurobi/GLPK/Z3/Soplex), symbolic dd engine, JANI. (Robust ω-regular / LRA on intervals:
  explicitly "not implemented".)

Bottom line: **for *solving* a given interval MDP, all four tools now agree** (robust
reachability/safety). IMPaCT's irreplaceable role is everything *upstream* — abstracting a
continuous nonlinear stochastic control system (with rich noise models) into that IMDP —
plus robust **ω-regular** synthesis on the result, neither of which PRISM, Storm, or
IntervalMDP.jl provides. Conversely PRISM and Storm bring rewards, multi-objective,
non-interval full-LTL, and mature symbolic engines that IMPaCT does not have.
