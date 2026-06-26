# IMPaCT v2.0 — Test Suite (Test-Driven Development)

These tests are **immutable contracts**. We write them *first*, derive their
expected values independently of any implementation, and then build the code to
satisfy them. **Do not weaken or edit a test to make failing code pass** — fix
the code, or, if an interface genuinely must change, change it as a reviewed
design decision recorded in [`../paper/IMPaCT-v2.md`](../paper/IMPaCT-v2.md) and
the [`TEST_PLAN.md`](TEST_PLAN.md), never as a quiet patch to a red test.

## Layout

```
tests/
├── doctest.h                # vendored single-header C++ test framework (v2.4.11)
├── contracts/
│   ├── contracts.h          # the TARGET API for all v2.0 components (TDD targets)
│   └── stubs.cpp            # temporary throwing stubs -> suite runs "red" pre-impl
├── unit/                    # C++ unit tests (pure std C++, no SYCL/HDF5/GLPK)
│   ├── test_main.cpp        # doctest entry point
│   ├── test_omaximization.cpp     # Phase 1a  (oracle: hand + brute-force vertices)
│   ├── test_graph.cpp             # Phase 1b  (SCC + MEC, hand-derived)
│   ├── test_interval_iteration.cpp# Phase 1c  (sound robust reachability)
│   └── test_ltl_dfa.cpp           # Phase 2   (LTLf membership)
├── oracles/oracles.py       # independent numpy/brute-force oracle generators
├── integration/             # pytest: oracle consistency + HDF5 golden regression
│   ├── conftest.py
│   ├── test_oracle_consistency.py # runs now, no toolchain needed
│   ├── test_regression_golden.py  # numerical regression vs golden HDF5
│   ├── pytest.ini, requirements.txt
├── golden/                  # golden HDF5 outputs (generated on the baseline)
└── TEST_PLAN.md             # full enumeration of every contract, all phases
```

## How the oracles are trustworthy (not reverse-engineered)

Every expected value is obtained by a method **independent** of the code under test:

- **O-maximization**: each optimum is hand-derived *and* cross-checked against
  brute-force enumeration of the probability-polytope vertices (an independent
  algorithm from the sort-and-assign we implement), in both C++ and
  `oracles.py`.
- **Interval iteration**: oracle reachabilities come from exact linear-algebra
  Markov-chain solves and direct adversary-vertex evaluation (`oracles.py`),
  not from running our solver.
- **SCC/MEC**: components are derived by hand from the graph definition.
- **LTLf**: accept/reject is read off the LTLf semantics, independent of the
  automaton back-end (Spot/Owl/Lydia).

`oracles.py` re-derives every constant baked into the C++ tests; CI runs it as a
guard (`test_oracle_consistency.py`).

## Running

```bash
# C++ unit contracts (currently RED: components not yet implemented)
make -C tests test

# Python oracle consistency (GREEN now — no toolchain needed)
python3 -m pytest tests/integration/test_oracle_consistency.py

# Numerical golden regression (needs acpp+HDF5+h5py; skips otherwise)
pip install -r tests/integration/requirements.txt
IMPACT_GEN_GOLDEN=1 git stash ... # generate goldens on the baseline first
python3 -m pytest tests/integration/test_regression_golden.py
```

## TDD status / wiring

`tests/Makefile` builds the unit tests against `IMPL_SRCS`. Today that is
`contracts/stubs.cpp`, so the new-feature tests fail with `not implemented`. As
each Phase-1 component is built in `src/`, swap its stub for the real source in
`IMPL_SRCS` and the corresponding tests should go green — with **no edits to the
test files**.
