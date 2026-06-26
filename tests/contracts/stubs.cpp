// ============================================================================
// IMPaCT v2.0 — TEMPORARY throwing stubs for the v2.0 contracts.
// ----------------------------------------------------------------------------
// Purpose: make the TDD suite BUILD and RUN in a "red" state before any feature
// is implemented. Every new-feature unit test should FAIL with "not implemented"
// until the real src/*.cpp is written.
//
// As each feature lands, DELETE the matching stub here and compile the real
// implementation instead. This file is scaffolding, NOT the implementation and
// NOT a test — never weaken a test to make it pass; implement the contract.
// ============================================================================

#include "contracts.h"
#include <stdexcept>

namespace impact {

namespace omax {
    Result optimize(const std::vector<double>&, const std::vector<double>&,
                    const std::vector<double>&, Sense) {
        throw std::logic_error("not implemented: impact::omax::optimize (Phase 1a)");
    }
}

namespace graph {
    std::vector<std::vector<int>> sccs(const AdjList&) {
        throw std::logic_error("not implemented: impact::graph::sccs (Phase 1b)");
    }
    std::vector<std::vector<int>> mecs(const MDPGraph&) {
        throw std::logic_error("not implemented: impact::graph::mecs (Phase 1b)");
    }
}

namespace solve {
    IntervalResult maxReachPessimistic(const IMDPModel&, const std::set<int>&, double) {
        throw std::logic_error("not implemented: impact::solve::maxReachPessimistic (Phase 1c)");
    }
    IntervalResult maxReachOptimistic(const IMDPModel&, const std::set<int>&, double) {
        throw std::logic_error("not implemented: impact::solve::maxReachOptimistic (Phase 1c)");
    }
}

namespace ltl {
    struct Automaton {};  // concrete definition so the stubs link
    Automaton* compileFinite(const std::string&, const std::vector<std::string>&) {
        throw std::logic_error("not implemented: impact::ltl::compileFinite (Phase 2)");
    }
    bool acceptsFinite(const Automaton*, const FiniteTrace&) {
        throw std::logic_error("not implemented: impact::ltl::acceptsFinite (Phase 2)");
    }
    void destroy(Automaton*) {}
}

} // namespace impact
