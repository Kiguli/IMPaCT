// ============================================================================
// CONTRACT TESTS — Phase 2 part 3: IMDP x DFA product + co-safe reachability.
// Key correctness gate: for phi = "F goal", product-reachability MUST equal plain
// solve::maxReach to the goal-labelled states (validates the whole DFA->product->
// solve pipeline against the already-verified solver). Plus a sequential PD-style
// example with a hand-computed probability.
// ============================================================================
#include "../doctest.h"
#include "../contracts/contracts.h"

#include <vector>
#include <set>
#include <random>
#include <cmath>

using namespace impact::solve;
using impact::product::Label;
using impact::product::build;

static const double EPS = 1e-7;

TEST_CASE("product: sequential F(pickup & F deliver) on a small chain == 0.5") {
    // 0 -> 1 (w.p.1); 1 -> 2 @0.5, -> 3(sink) @0.5; 2(deliver) sink; 3 sink.
    // labels: 1=pickup, 2=deliver. Satisfying paths reach 2 after 1: prob 0.5.
    IMDPModel m = {
        /*0*/ {{ {1,1,1} }},
        /*1*/ {{ {2,0.5,0.5}, {3,0.5,0.5} }},
        /*2*/ {{ {2,1,1} }},
        /*3*/ {{ {3,1,1} }},
    };
    std::vector<Label> labels = { {}, {"pickup"}, {"deliver"}, {} };
    auto* aut = impact::ltl::compileFinite("F(pickup & F deliver)", {"pickup", "deliver"});
    auto dfa = impact::ltl::toDFA(aut);
    auto P = build(m, labels, dfa, /*s0=*/0);
    auto r = maxReachOptimistic(P.model, P.targets, EPS);
    double v = 0.5 * (r.lower[P.start] + r.upper[P.start]);
    CHECK(v == doctest::Approx(0.5).epsilon(1e-3));
    impact::ltl::destroy(aut);
}

TEST_CASE("product: 'F goal' reachability == plain reach to goal states (randomized)") {
    std::mt19937 rng(4242);
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    auto* aut = impact::ltl::compileFinite("F goal", {"goal"});
    auto dfa = impact::ltl::toDFA(aut);

    int checked = 0;
    for (int trial = 0; trial < 400 && checked < 200; ++trial) {
        int n = 2 + (int)(rng() % 5);
        // random point-probability MDP
        IMDPModel m(n);
        std::set<int> goal;
        std::vector<Label> labels(n);
        for (int s = 0; s < n; ++s) {
            bool isGoal = (rng() % 3u) == 0u;
            if (isGoal) { goal.insert(s); labels[s] = {"goal"}; }
            int na = 1 + (int)(rng() % 3u);
            for (int a = 0; a < na; ++a) {
                std::vector<int> succ;
                for (int t = 0; t < n; ++t) if (rng() & 1u) succ.push_back(t);
                if (succ.empty()) succ.push_back((int)(rng() % n));
                std::vector<double> w(succ.size());
                double sum = 0; for (double& x : w) { x = u01(rng) + 1e-3; sum += x; }
                ActionDist act;
                for (size_t k = 0; k < succ.size(); ++k) { double p = w[k] / sum; act.push_back({succ[k], p, p}); }
                m[s].push_back(act);
            }
        }
        if (goal.empty()) continue;
        ++checked;

        auto P = build(m, labels, dfa, /*s0=*/0);
        double v_prod = 0.5 * ([&]{ auto r = maxReachOptimistic(P.model, P.targets, EPS);
                                    return r.lower[P.start] + r.upper[P.start]; }());
        double v_plain = 0.5 * ([&]{ auto r = maxReachOptimistic(m, goal, EPS);
                                     return r.lower[0] + r.upper[0]; }());
        CHECK(std::fabs(v_prod - v_plain) < 2e-3);
    }
    CHECK(checked > 80);
    impact::ltl::destroy(aut);
}
