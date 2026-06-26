#ifndef IMPACT_TEST_CONTRACTS_H
#define IMPACT_TEST_CONTRACTS_H

// ============================================================================
// IMPaCT v2.0 — TEST CONTRACTS (interfaces the implementation must satisfy)
// ----------------------------------------------------------------------------
// These declarations are the *target API* for the new IMPaCT v2.0 components.
// The TDD test suite is written against THESE signatures. Implementation work
// fills in the corresponding src/*.cpp (replacing tests/contracts/stubs.cpp).
//
// RULE: tests are immutable contracts. We change the implementation to satisfy
// the tests, never the other way around. If an interface here must change, it
// is a deliberate design change reviewed against the test plan — not a hack to
// make a failing test pass.
// ============================================================================

#include <vector>
#include <set>
#include <string>
#include <cstddef>

// Phase 1a — O-maximization. Now implemented (verified extraction of IMPaCT's
// sorted approach); the production interface lives in src/. The behavioural
// contract (feasibility, optimality, throwing) is documented there and in
// tests/unit/test_omaximization.cpp.
#include "../../src/omaximization.h"

// Phase 1b — SCC + MEC decomposition. Implemented; production interface in src/.
#include "../../src/graph_utils.h"

// Phase 1c — sound robust interval iteration. Implemented; interface in src/.
#include "../../src/solve.h"

// Phase 2 — LTLf / co-safe-LTL front-end (parse + finite-trace membership).
// Tested via language membership so the contract is back-end-independent.
// Refs: Kupferman-Vardi (FMSD 2001); De Giacomo-Vardi (IJCAI 2013/2015).
#include "../../src/ltl.h"

// Phase 2 part 3 — IMDP x DFA product + co-safe reachability.
#include "../../src/product.h"

// Sparse interval-MDP abstraction of continuous systems (scalability / ISSUE-0006).
#include "../../src/abstraction.h"

// Phase 3 — omega-regular (Büchi) synthesis via accepting-MEC reachability.
#include "../../src/omega.h"

// Explicit (interval) MDP exchange format (cross-tool benchmarking).
#include "../../src/imdp_io.h"

// PRISM-language front-end (subset) -> io::Problem (user-requested input style).
#include "../../src/prism.h"

#endif // IMPACT_TEST_CONTRACTS_H
