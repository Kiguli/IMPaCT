# Statistical Model Checking

Monte-Carlo simulation of point Markov chains: probability estimation with confidence
intervals, and sequential hypothesis testing (SPRT) for threshold queries.

## What is computed

`smc LABEL` samples paths of a **point** chain (one action per state, `lo == hi`) from
the initial state (or `--state S`) and estimates the step-bounded reachability
`P(F<=horizon LABEL)`; paths stop early in absorbing states. Default horizon is 1000
(override with `--horizon k`), default sample count 100000 (`--samples N`). The
estimate line reports:

- `estimate` — fraction of sampled paths that hit the target;
- `wilson95=[lo,hi]` — the Wilson 95% confidence interval;
- `chernoff_eps` — the APMC / Chernoff-Hoeffding half-width ε such that
  `P(|estimate − p| >= ε) <= δ` with δ = 0.05.

With `--threshold p`, a second line gives the **Wald SPRT** verdict for
`H0: P >= θ+δ` vs `H1: P <= θ−δ` with error bounds α = β = 0.01 and indifference
region 2δ (δ = 0.01): `accept(P>=theta)`, `reject(P<theta)`, or `undecided` (sample
budget exhausted, capped at 10× `--samples`), plus the number of samples the
sequential test actually used.

Seeds are fixed (estimation seed 20260702, SPRT seed 20260703), so runs are
reproducible.

## Algorithm and literature

Plain Monte-Carlo path sampling with rigorous statistics (`src/smc.{h,cpp}`):

- Younes, Simmons, *Probabilistic Verification of Discrete Event Systems Using
  Acceptance Sampling*, CAV 2002 (SPRT-based statistical model checking).
- Hérault, Lassaigne, Magniette, Peyronnet, *Approximate Probabilistic Model
  Checking*, VMCAI 2004 (APMC: Chernoff-Hoeffding sample bounds).
- Wald, *Sequential Tests of Statistical Hypotheses*, Annals of Mathematical
  Statistics 16(2), 1945 (the SPRT itself).

## CLI examples (validated)

`bounded_demo` reaches the goal with probability `1 − 0.5^3 = 0.875` within 3 steps:

```bash
# Estimation with Wilson 95% CI and Chernoff half-width
tools/imdp_solve benchmarks/crosstool/models/bounded_demo.imdp smc goal --horizon 3
# result  prop=smc  state=0  estimate=0.87471  wilson95=[0.8726437795,0.876747433]
#         chernoff_eps=0.004294694083  samples=100000

# SPRT: is P >= 0.8?  (true: 0.875 >= 0.8) — accepted after 570 samples
tools/imdp_solve benchmarks/crosstool/models/bounded_demo.imdp smc goal --horizon 3 --threshold 0.8
# result  prop=smc-sprt  threshold=0.8   verdict=accept(P>=theta)  samples=570

# SPRT: is P >= 0.95? (false) — rejected after 199 samples
tools/imdp_solve benchmarks/crosstool/models/bounded_demo.imdp smc goal --horizon 3 --threshold 0.95
# result  prop=smc-sprt  threshold=0.95  verdict=reject(P<theta)   samples=199
```

The Wilson interval `[0.87264, 0.87675]` contains the analytic 0.875, and the SPRT
decides both directions correctly with two to three orders of magnitude fewer samples
than the fixed-size estimate.

## Peer tools and validation

- **PRISM**: `-sim` (its discrete-event simulator, also DTMC-only for probability
  estimation). On the same query PRISM `-sim` returned **0.87479** vs IMPaCT's
  **0.87471** — both within their confidence intervals of the analytic 0.875
  (`FEATURE_PARITY.md` §B).
- **Storm**: has a simulation-based engine for point models.
- **IntervalMDP.jl**: no statistical model checking.

## Limitations

- **Point chains only**: the model must have one action per state with degenerate
  intervals; `smc` throws otherwise. MDP scheduler sampling is future work — the same
  scope as PRISM's simulator (`src/smc.h`). There is no meaningful SMC semantics for
  *interval* uncertainty without fixing a nature policy; use the robust numeric
  solvers for intervals.
- The property is step-bounded reachability only (choose `--horizon` large enough to
  approximate unbounded reachability on absorbing chains; paths that reach an
  absorbing non-target state stop early, so a large horizon is cheap there).
- Statistical guarantees are confidence statements, not the sound `[lower, upper]`
  brackets of the numeric engines — use `smc` for quick estimation or very large
  chains, and the numeric solvers for certified values.
