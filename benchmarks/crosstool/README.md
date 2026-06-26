# Cross-tool benchmarking (IMPaCT vs IntervalMDP.jl / PRISM / Storm)

Goal (user request): run the benchmarks we should be able to solve on the same
models in IMPaCT and in peer tools, and check we get the **same outputs** (solve
times may differ). This directory does that on a shared, tool-neutral model
format.

## Mode: reference values (peers deferred)

We are running in **reference-values mode**: peer tools are *not* installed here.
Instead each model carries a sibling `.ref.json` with **independently-computed**
reference values plus the provenance of how each was obtained:

- **Point (degenerate-interval) MDPs** — exact reachability/safety by linear
  solve. PRISM, Storm and IntervalMDP.jl all return these exactly (no slack).
- **Interval MDPs** — exact robust value by closed-form **O-maximization**
  (nature pushes mass to the worst/best successor) and/or enumeration of the
  controller's actions × extremal nature. This is exactly the robust semantics
  IntervalMDP.jl computes.

So the comparison is meaningful *today* without the peers, and stays valid when
they are added (their measured numbers must equal the same references).

## Files

- `models/*.imdp` — shared models in the neutral explicit format (`src/imdp_io.h`):
  `states N` / `init s` / `label NAME s...` / `tran s a to:lo:hi ...`.
- `models/*.prism` — models in the PRISM-language subset (`src/prism.h`); the CLI
  detects `.prism` by extension and parses them into the same `io::Problem`, so a
  PRISM model and the equivalent `.imdp` model solve identically (see
  `choice.prism` vs `choice_interval.imdp`).
- `models/*.ref.json` — reference values + provenance for each model.
- `compare.py` — runs `tools/imdp_solve` on every model and checks IMPaCT's sound
  interval `[lower, upper]` brackets the reference **and** the midpoint matches it
  within `tol`, for both the pessimistic and optimistic senses.

## Run

```sh
# build the CLI solver (pure std C++ — no SYCL/Armadillo needed)
c++ -std=c++17 -O2 tools/imdp_solve.cpp \
    src/imdp_io.cpp src/solve.cpp src/omaximization.cpp src/graph_utils.cpp \
    -o tools/imdp_solve

python3 benchmarks/crosstool/compare.py -v   # exit 0 iff all checks pass
```

Current status: **20/20 checks pass** across 9 models:
- reach/safety: point chain reach, point safety, interval fork, controller-choice
  (and the same choice model in PRISM-language input — front-end equivalence);
- ω-regular: Büchi recurrence reaching an accepting EC (0.5), the route-around
  support-EC (robust 0 / optimistic 1, ISSUE-0009), persistence `F G safe` with a
  leak (robust 0.5 / optimistic 1), and patrol `⋀ G F` on a 3-cycle (1).

ω-regular properties are exercised through `tools/imdp_solve` (`buchi` / `persist`
/ `patrol`); each reference value is computed by hand and matches what a peer with
interval-MDP ω-regular support (e.g. Storm) returns under the same robust semantics.

## Adding live peers later

The harness is structured so installing a peer is additive — no harness changes:

1. **IntervalMDP.jl** — translate `.imdp` → its sparse IMDP arrays (the format is a
   near-direct match: per (s,a) lower/upper rows), run robust value iteration,
   and write the measured `[lower, upper]` into the model's `.ref.json` `checks`
   alongside the analytic value (they must agree).
2. **PRISM / Storm** — point models map directly to a PRISM MDP (`Pmax=?[F target]`
   / `Pmax=?[!(F avoid)]`); interval models need a tool with IMDP support (Storm's
   `--engine` + interval models, or PRISM via a uMDP encoding). Drop measured
   values into the same `checks`.

Because checks are data (`.ref.json`), a peer's numbers slot in without touching
`compare.py`. Discrepancies (semantic mismatches, rounding, "reduced spec"
collapses) get logged as issues under `issues/` per the project protocol.
