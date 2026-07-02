# Solver Engines

IMPaCT ships a toolbox of infinite-horizon convergence methods. This page explains
what each computes, when each is sound, and which to choose.

## The three iteration methods (`IterationMethod`)

The abstraction-side synthesis API (`IMDP::setIterationMethod`, `src/IMDP.h`) selects
among three methods for infinite-horizon robust value iteration; the same solvers back
the CLI (the dense SYCL path dispatches infinite-horizon OVI through the shared CPU
solver `src/solve.cpp`):

- **`IntervalIteration`** (dense-path default) — sound interval iteration: iterate the
  robust Bellman operator from 0 (lower) and from 1 (upper), stop when
  `|upper − lower| < eps`. A rigorous two-sided bracket (Baier et al., CAV 2017;
  Haddad & Monmege, TCS 2018), but on weakly-contracting / end-component models the
  gap can close slowly or never (ISSUE-0003).
- **`ValueIteration`** — what the peers (PRISM / Storm / IntervalMDP.jl) run by
  default: plain VI with *residual* stopping. Far fewer sweeps and converges even with
  end components (the from-0 iterate tends to the least fixed point = the robust
  value), but the residual stop is **not** a sound two-sided certificate.
- **`OptimisticVI`** (CLI default) — optimistic value iteration (Hartmanns & Kaminski,
  CAV 2020): VI from below for the lower bound plus a *verified inductive* upper bound
  — a candidate `U` with `F(U) <= U` implies `V* <= U` by Knaster-Tarski. A sound
  two-sided certificate that needs **no end-component handling**. The from-below phase
  is **topological Gauss–Seidel** (ISSUE-0010, resolved 2026-07-02): the support graph
  is decomposed into strongly connected components, iterated to convergence in reverse
  topological order with successor components already fixed (Dai, Mausam, Weld,
  Goldsmith, *Topological Value Iteration Algorithms*, JAIR 42:181–209, 2011, DOI
  10.1613/jair.3390; acyclic states take a single pass), with in-place Gauss–Seidel
  sweeps (Puterman, 1994, §6.3.3). Both preserve monotone convergence from below to the
  same least fixpoint, and the inductive certificate is verified globally afterwards,
  so soundness is independent of the sweep schedule. Measured effect: the ARCH `BA`
  robust safety solve at `--eps 1e-6` went from **>180 s (timeout) to 7.8 s** (>23×),
  with all reference values bit-identical.

In `imdp_solve`, `--method ovi` (default) selects `solve::Method::OptimisticVI` and
`--method mec` selects `solve::Method::MECCollapse` (interval iteration after
support-graph maximal-end-component collapse).

## The nature trap (ISSUE-0003)

Pessimistic interval iteration with *support-graph* MEC collapse does not converge on
**nature-confinable end components**: nature can confine the play in a set by sending
a leaving edge to its lower bound 0, creating a non-unique fixpoint that the upper
iterate never leaves. Minimal example (`benchmarks/crosstool/models/naturetrap.imdp`):
state 0 stays with `[0.5, 1.0]` or leaves with `[0.0, 0.5]` — nature zeroes the
leaving edge, so the true robust value at 0 is **0**, but the support graph sees no
MEC and the classic upper bound sticks at 1.

```bash
# OVI (default): converges to the true robust value 0 with a certified bracket
tools/imdp_solve benchmarks/crosstool/models/naturetrap.imdp reach target --method ovi
# result  prop=reach  bound=pess  state=0  lower=0  upper=1e-06  iters=2

# MEC collapse in the OPTIMISTIC sense: valid and fast (support EC = optimistic EC)
tools/imdp_solve benchmarks/crosstool/models/naturetrap.imdp reach target --bound opt --method mec
# result  prop=reach  bound=opt  state=0  lower=0.9999980927  upper=1  iters=20
```

OVI resolving the nature trap is why it is the default; `MECCollapse` remains valid
for point MDPs and for the optimistic sense (`src/solve.h`).

## When to use which

| Situation | Method |
|---|---|
| Default; interval models, pessimistic sense | `OptimisticVI` (sound, converges on nature traps) |
| Large *recurrent* models (topological Gauss–Seidel handles most; ISSUE-0010 resolved) | `OptimisticVI`, or `MECCollapse` for the optimistic sense |
| Point MDPs, either sense | either; `MECCollapse` is fast on controller ECs |
| Matching a peer tool's convergence behaviour | `ValueIteration` (dense API only; residual stop, no certificate) |
| Exact certified values on small models | {doc}`exact` (rational policy iteration) |

## Performance notes

- **ISSUE-0010** (resolved 2026-07-02): OVI's from-below iterate mixed slowly on large,
  strongly-recurrent IMDPs (the ARCH `BA` case needed >180 s at `eps 1e-6`). Resolved
  by the topological Gauss–Seidel from-below phase described above: BA robust safety at
  `eps 1e-6` now solves in **7.8 s** with identical values. For stubborn cases the
  older remedies remain available: `--method mec` or a looser `--eps`.
- **ISSUE-0017** (resolved): the SYCL dense VI was slow versus Storm/IntervalMDP.jl on
  recurrent IMDPs; the infinite-horizon dense path now dispatches through the shared
  CPU OVI solver (`IMDP::infiniteHorizonOVIDispatch`, `src/IMDP.h`).
- All methods share the same O-maximization inner solve, so per-sweep cost is
  identical; they differ only in sweep counts and certification.

## Soundness summary

`OptimisticVI` and `IntervalIteration` return certified `[lower, upper]` with
`gap <= 2*eps`; `ValueIteration` returns an uncertified estimate. The pessimistic
`MECCollapse` upper bound is *valid* but may not converge on interval nature traps —
the CLI defaults protect you from this; only override `--method` in the situations
listed above.
