# CTMCs and CSL Time-Bounded Reachability

Continuous-time Markov chains (CTMCs), CSL time-bounded reachability `P(F<=t goal)`,
and the interval (uncertain-rate) CTMC extension unique to IMPaCT.

## Model class

A CTMC is written in the plain `.imdp` format where each state has a **single action**
whose entries `to:r:r` are transition **rates** (not probabilities; no self rate). An
**interval CTMC** uses rate intervals `to:rlo:rhi` — uncertain rates. The entries are
interpreted as rates only by the `csl` property; see {doc}`formats`.

```text
# ctmc_demo.imdp: s0 --rate 1--> s1 (absorbing goal); P(F<=t goal) = 1 - e^{-t}
states 2
init 0
label goal 1
tran 0 0 1:1:1
```

## What is computed

`csl LABEL --time t` returns the per-state probability of reaching `LABEL` within
continuous time `t` (goal states made absorbing). For an interval CTMC, `--bound pess`
resolves the uncertain rates adversarially per uniformisation step (sound lower CSL
bound) and `--bound opt` cooperatively (sound upper bound) — the three-valued CSL
reading of Katoen et al.

## Algorithm and literature

**Uniformisation + Fox-Glynn** (`src/ctmc.{h,cpp}`): Jensen's uniformisation builds the
embedded DTMC `P = I + Q/Λ` with `Λ` = max exit rate; the time-bounded value is the
Poisson-weighted transient of `P`, with the Poisson weights computed by the numerically
stable Fox-Glynn truncation. For interval rates, the uniformised chain is an interval
DTMC whose step distributions are resolved by O-maximization.

- C. Baier, B. Haverkort, H. Hermanns, J.-P. Katoen, *Model-Checking Algorithms for
  Continuous-Time Markov Chains*, IEEE Trans. Software Eng. 29(6):524-541, 2003
  (DOI 10.1109/TSE.2003.1205180).
- B. L. Fox, P. W. Glynn, *Computing Poisson probabilities*, CACM 31(4):440-445, 1988
  (DOI 10.1145/42404.42409).
- Interval CTMC: Katoen, Klink, Leucker, Wolf, *Three-Valued Abstraction for CTMCs*,
  CAV 2007 (DOI 10.1007/978-3-540-73368-3_37); Badings, Jansen, Junges, Stoelinga,
  Volk, *Sampling-Based Verification of CTMCs with Uncertain Rates*, CAV 2022
  (arXiv:2205.08300).

Steady-state analysis of a CTMC reuses the LRA machinery: the uniformised DTMC has the
same stationary distribution as the CTMC (see {doc}`lra-steadystate`).

## CLI examples (validated)

```bash
# Point CTMC: P(F<=1 goal) = 1 - e^{-1} = 0.6321206
tools/imdp_solve benchmarks/crosstool/models/ctmc_demo.imdp csl goal --time 1.0
# result  prop=csl  bound=pess  state=0  lower=0.6321205588  upper=0.6321205588  iters=0

tools/imdp_solve benchmarks/crosstool/models/ctmc_demo.imdp csl goal --time 2.0
# result  prop=csl  bound=pess  state=0  lower=0.8646647168  upper=0.8646647168  iters=0

# Interval CTMC (rate in [0.5, 1.5]) — IMPaCT-only:
# robust 1-e^{-0.5} = 0.3934693, best-case 1-e^{-1.5} = 0.7768698
tools/imdp_solve benchmarks/crosstool/models/ctmc_interval.imdp csl goal --time 1.0 --bound both
# result  prop=csl  bound=pess  state=0  lower=0.3934693403  upper=0.3934693403  iters=0
# result  prop=csl  bound=opt   state=0  lower=0.7768698399  upper=0.7768698399  iters=0
```

## Peer tools and validation

- **Point CTMC**: IMPaCT == Storm CSL (`storm --prism ctmc_demo.sm -pc --prop
  'P=?[F<=t "goal"]'`) == the analytic value `1 - e^{-t}` **to 10 digits** for
  t = 0.5, 1, 2 (`ctmc_demo.ref.json`; the PRISM-language twin `ctmc_demo.sm` ships
  alongside the `.imdp`). PRISM supports point-rate CSL equivalently.
- **Interval CTMC**: Storm and PRISM support point rates only, so the robust values
  were validated against the exact analytic bounds above (`ctmc_interval.ref.json`) —
  **an IMPaCT-only capability** (`FEATURE_PARITY.md`).

## Limitations

- The `csl` property covers **time-bounded reachability** `P(F<=t goal)`; nested CSL
  (time-bounded until with a lower time bound, nested `P`/`S` operators) is not
  exposed. Unbounded CSL reachability equals the embedded-DTMC reachability (`reach`),
  and steady-state uses `ss`.
- The model must be a chain at the CTMC level: one action per state (CTMDPs are not
  supported; nondeterminism plus continuous time is the Markov-automata gap tracked in
  `FEATURE_PARITY.md` §B).
- The interval-rate resolution is per uniformisation step (step-wise adversary), the
  standard sound bound of the three-valued CSL construction; a time-invariant adversary
  could be tighter.
