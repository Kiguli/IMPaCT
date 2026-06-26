# IMPaCT web app (Phase 4)

A lightweight, **dependency-free** web app that visualizes a small Interval-MDP and
overlays the **robust satisfaction probabilities** synthesized by IMPaCT, per the
plan's usability phase and the user request: *"for small models with some capped
number of states, visualize the IMDP and provide some visual of the resulting
satisfaction probabilities."*

The C++ core is untouched: the backend shells to the verified `tools/imdp_solve`
CLI (`--json`), which returns the model structure (states/edges/labels) plus
per-state `[lower, upper]` values for the chosen property and sense.

## Visualizer modes (tabs)
- **IMDP graph** — Interval-MDP node-link graph, states heat-mapped by satisfaction
  probability, for every property (reach/safety/until/next/buchi/persist/patrol/LTL).
- **Grid heatmap** — a 2-D continuous system (`.sys`) abstracted to a sparse IMDP;
  per quantized grid cell, the **robust (min)** and **optimistic (max)** probability
  of the spec (reach/safety) shown as two heatmaps side by side.
- **Zone graph** — the symbolic zone graph of a (probabilistic) timed automaton
  (`.pta`): nodes are (location, zone) heat-mapped by reach probability.
- **Belief tree** — the POMDP (`.pomdp`) optimal-policy belief tree: nodes show the
  belief (stacked bar) and value; edges are (observation, P(obs)).
- **About** — overview of the engines behind each tab.

## What it shows
- The IMDP as a node-link graph (circular layout, self-contained SVG — no CDN).
- Each state **heat-mapped** by its satisfaction probability (red → amber → green).
- Initial state (thick border) and labelled states (★); edges carry the interval
  `[lo, hi]` on hover; per-state robust/optimistic values on hover.
- **Category tabs** (Reachability · ω-regular · LTL · Input · Other models), each
  with one **clickable example per feature** so the app doubles as a tutorial.
- Every IMDP property the toolbox exposes, all robust (pessimistic) + optimistic:
  `reach` (F), `safety` (G), `until` (a U b), `next` (X), `buchi` (G F recurrence),
  `persist` (F G), `patrol` (⋀ G F), and a free-form **LTL formula** mode
  (`ltlspec` dispatcher: F/G/U/X/GF/FG/patrol over boolean atoms), in `.imdp` or
  PRISM input. Out-of-fragment LTL is reported (needs the LDBA route, ISSUE-0016).
- The **Other models** tab documents the non-IMDP features (bounded STL, timed
  automata, PTA, POMDPs, continuous-system abstraction) that are used via the CLI /
  library rather than this graph view.
- A **state cap** (default 60): larger models report their size and are not
  rendered (the backend also has a 200-state hard cap), so the visualization stays
  legible for the small models it is meant for.

## Run
```sh
# 1. build the CLIs the web app shells to (pure std C++ — no SYCL/Armadillo needed)
c++ -std=c++17 -O2 tools/imdp_solve.cpp \
    src/imdp_io.cpp src/prism.cpp src/solve.cpp \
    src/omaximization.cpp src/graph_utils.cpp src/omega.cpp \
    src/pctl.cpp src/ltlspec.cpp \
    -o tools/imdp_solve
# grid-cell heatmap (continuous abstraction):
c++ -std=c++17 -O2 tools/grid_heatmap.cpp \
    src/system_io.cpp src/abstraction.cpp src/solve.cpp \
    src/omaximization.cpp src/graph_utils.cpp -o tools/grid_heatmap
# timed-automaton zone graph:
c++ -std=c++17 -O2 tools/ta_zonegraph.cpp \
    src/pta_io.cpp src/pta.cpp src/ta.cpp src/dbm.cpp src/solve.cpp \
    src/omaximization.cpp src/graph_utils.cpp -o tools/ta_zonegraph
# POMDP belief tree:
c++ -std=c++17 -O2 tools/pomdp_belieftree.cpp \
    src/pomdp_io.cpp src/pomdp.cpp -o tools/pomdp_belieftree

# 2. run the app (Python 3, standard library only)
python3 webapp/server.py            # open http://localhost:8000
```

### Windows
Build `tools\imdp_solve.exe` (any C++17 compiler — MSVC `cl /std:c++17 /EHsc ...`
or g++), then `python webapp\server.py`. The server auto-detects `imdp_solve.exe`;
override with `--solver path\to\imdp_solve.exe` or the `IMPACT_SOLVE` env var.

## API
`POST /api/solve` with JSON `{model, format: "imdp"|"prism", prop, label, bound,
eps}` → `{nStates, init, labels, edges, prop, label, values: {pess|opt: [{lower,
upper}, …]}, stateCap}`. Errors return `{error}` with a 4xx/5xx status.

## Relationship to the full Phase-4 plan
The plan describes a richer hosted app (Vue 3 + Vite + Tailwind + Inertia.js +
Flask, Dockerized like the user's TRUST, with Jinja2 codegen → `acpp` compile for
the continuous-system synthesis pipeline, and a Dwyer-pattern spec builder with a
live LTL→automaton preview). This stdlib version delivers the **small-model
IMDP + satisfaction-probability visualization** today with zero install friction;
the codegen/compile pipeline and the spec builder layer on top of the same
`/api/solve`-style contract later. Keeping the backend a thin shell over the
verified CLI means the visualization always reflects the exact synthesis result.

## Notes / limitations
- Designed for small models (the visualization is the point); large models are
  capped, not rendered.
- The hosted/Docker target is CPU-only (no GPU) and best for demos; large/GPU runs
  stay local via the generated C++ project, as in the plan.
- Tracked design decisions and issues: see `paper/IMPaCT-v2.md` and `issues/`.
