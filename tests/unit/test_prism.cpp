// ============================================================================
// CONTRACT TESTS — PRISM-language front-end (subset) -> io::Problem.
//
// Supports the recognizable PRISM modelling style for a SINGLE bounded integer
// state variable: model type `mdp`, `const int`, one `module..endmodule` with a
// `x : [lo..hi] init i;` decl, guarded commands
//     [act] x=K -> p:(x'=E) + p:(x'=E) ... ;   (p a literal OR an interval [lo,hi])
// (a single update may omit the probability => 1), updates E in {int, x, x+int,
// x-int}, and `label "n" = x=K [| x=K2 ...];`. Interval probabilities [lo,hi]
// give an IMDP; point probabilities give an ordinary MDP. // comments allowed.
//
// These are immutable contracts: solve values are independently known, and a
// PRISM model and the equivalent explicit .imdp model must solve identically.
// ============================================================================
#include "../doctest.h"
#include "../contracts/contracts.h"

#include <string>

using impact::prism::parse;
using impact::io::Problem;

// ---- point Markov chain: P(reach x=3) = 0.25 (exact) -----------------------
static const char* CHAIN =
    "mdp\n"
    "module chain\n"
    "  x : [0..3] init 0;\n"
    "  [] x=0 -> 0.5:(x'=1) + 0.5:(x'=2);   // fork\n"
    "  [] x=1 -> 0.5:(x'=3) + 0.5:(x'=2);\n"
    "  [] x=2 -> 1:(x'=2);\n"
    "  [] x=3 -> 1:(x'=3);\n"
    "endmodule\n"
    "label \"target\" = x=3;\n";

// ---- interval fork: robust [0.4,0.6]; uses prob-less update (x'=3) ----------
static const char* FORK =
    "mdp\n"
    "module fork\n"
    "  x : [0..3] init 0;\n"
    "  [] x=0 -> [0.4,0.6]:(x'=1) + [0.4,0.6]:(x'=2);\n"
    "  [a] x=1 -> (x'=3);\n"
    "  [a] x=2 -> (x'=2);\n"
    "  [a] x=3 -> (x'=3);\n"
    "endmodule\n"
    "label \"target\" = x=3;\n";

// ---- controller choice: const N, two commands per guard, x'=x+1, disjunct label
static const char* CHOICE =
    "mdp\n"
    "const int N = 2;\n"
    "module choice\n"
    "  x : [0..N] init 0;\n"
    "  [] x=0 -> [0.5,0.7]:(x'=1) + [0.3,0.5]:(x'=2);\n"
    "  [] x=0 -> 0.6:(x'=x+1) + 0.4:(x'=2);\n"
    "  [] x=1 -> 1:(x'=1);\n"
    "  [] x=2 -> 1:(x'=2);\n"
    "endmodule\n"
    "label \"target\" = x=2;\n"
    "label \"high\" = x=1 | x=2;\n";

TEST_CASE("prism: parse structure (chain)") {
    Problem p = parse(CHAIN);
    CHECK(p.nStates == 4);
    CHECK(p.init == 0);
    CHECK(p.labels.at("target") == std::set<int>{3});
    CHECK(p.model[0].size() == 1);
    CHECK(p.model[0][0].size() == 2);
}

TEST_CASE("prism: point chain solves to exact 0.25") {
    Problem p = parse(CHAIN);
    auto r = impact::solve::maxReachPessimistic(p.model, p.labels.at("target"), 1e-7);
    CHECK(0.5*(r.lower[p.init]+r.upper[p.init]) == doctest::Approx(0.25).epsilon(1e-4));
}

TEST_CASE("prism: interval fork robust value [0.4,0.6]") {
    Problem p = parse(FORK);
    auto pess = impact::solve::maxReachPessimistic(p.model, p.labels.at("target"), 1e-7);
    auto opt  = impact::solve::maxReachOptimistic (p.model, p.labels.at("target"), 1e-7);
    CHECK(0.5*(pess.lower[p.init]+pess.upper[p.init]) == doctest::Approx(0.4).epsilon(1e-3));
    CHECK(0.5*(opt.lower[p.init]+opt.upper[p.init])  == doctest::Approx(0.6).epsilon(1e-3));
}

TEST_CASE("prism: const + controller choice + x'=x+1 + disjunctive label") {
    Problem p = parse(CHOICE);
    CHECK(p.nStates == 3);                          // [0..N], N=2
    CHECK(p.model[0].size() == 2);                  // two actions at x=0
    CHECK(p.labels.at("high") == std::set<int>{1,2});
    auto pess = impact::solve::maxReachPessimistic(p.model, p.labels.at("target"), 1e-7);
    auto opt  = impact::solve::maxReachOptimistic (p.model, p.labels.at("target"), 1e-7);
    CHECK(0.5*(pess.lower[p.init]+pess.upper[p.init]) == doctest::Approx(0.4).epsilon(1e-3));
    CHECK(0.5*(opt.lower[p.init]+opt.upper[p.init])  == doctest::Approx(0.5).epsilon(1e-3));
}

TEST_CASE("prism: same model via PRISM and explicit .imdp solve identically") {
    static const char* IMDP =
        "states 4\ninit 0\nlabel target 3\n"
        "tran 0 0 1:0.5:0.5 2:0.5:0.5\n"
        "tran 1 0 3:0.5:0.5 2:0.5:0.5\n"
        "tran 2 0 2:1:1\ntran 3 0 3:1:1\n";
    Problem a = parse(CHAIN);
    Problem b = impact::io::parse(IMDP);
    auto ra = impact::solve::maxReachPessimistic(a.model, a.labels.at("target"), 1e-7);
    auto rb = impact::solve::maxReachPessimistic(b.model, b.labels.at("target"), 1e-7);
    for (int s = 0; s < 4; ++s)
        CHECK(0.5*(ra.lower[s]+ra.upper[s]) ==
              doctest::Approx(0.5*(rb.lower[s]+rb.upper[s])).epsilon(1e-6));
}
