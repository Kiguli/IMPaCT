// ============================================================================
// CONTRACT TESTS — Phase 3: omega-regular (Büchi) via accepting-MEC reachability.
// Verified by reduction to the already-verified reachability solver + hand cases.
// ============================================================================
#include "../doctest.h"
#include "../contracts/contracts.h"

#include <set>
#include <vector>
#include <algorithm>
#include <cstdint>

using namespace impact::solve;
using impact::omega::maxBuchiOptimistic;
using impact::omega::maxBuchiPessimistic;
using impact::omega::maxGenBuchiOptimistic;
using impact::omega::maxGenBuchiPessimistic;
using impact::omega::maxPersistenceOptimistic;
using impact::omega::maxPersistencePessimistic;
using impact::omega::acceptingMECStates;
using impact::omega::robustBuchiWinningStates;

static const double EPS = 1e-7;

TEST_CASE("buchi: accepting absorbing state -> equals reachability of it") {
    // 0 ->1 @0.5, ->2(sink) @0.5 ; 1 self-loop (accepting) ; 2 sink.
    IMDPModel m = {
        /*0*/ {{ {1,0.5,0.5}, {2,0.5,0.5} }},
        /*1*/ {{ {1,1,1} }},        // accepting absorbing
        /*2*/ {{ {2,1,1} }},
    };
    auto b = maxBuchiOptimistic(m, {1}, EPS);
    auto r = maxReachOptimistic(m, {1}, EPS);   // {1} is a self-loop MEC
    for (int s = 0; s < 3; ++s)
        CHECK(0.5*(b.lower[s]+b.upper[s]) == doctest::Approx(0.5*(r.lower[s]+r.upper[s])).epsilon(1e-6));
    CHECK(0.5*(b.lower[0]+b.upper[0]) == doctest::Approx(0.5));
}

TEST_CASE("buchi: GF(true) = 1 (every path ends in a MEC)") {
    IMDPModel m = { /*0*/ {{ {1,1,1} }}, /*1*/ {{ {1,1,1} }} };
    auto b = maxBuchiOptimistic(m, {0, 1}, EPS);
    CHECK(0.5*(b.lower[0]+b.upper[0]) == doctest::Approx(1.0).epsilon(1e-6));
    CHECK(0.5*(b.lower[1]+b.upper[1]) == doctest::Approx(1.0).epsilon(1e-6));
}

TEST_CASE("buchi: transient accepting state (not in any MEC) -> value 0") {
    // 0 (accepting, transient) -> 1 ; 1 self-loop (not accepting).
    IMDPModel m = { /*0*/ {{ {1,1,1} }}, /*1*/ {{ {1,1,1} }} };
    auto b = maxBuchiOptimistic(m, {0}, EPS);
    CHECK(0.5*(b.lower[0]+b.upper[0]) == doctest::Approx(0.0).epsilon(1e-6));
}

TEST_CASE("buchi: recurrence in an end component, reached with prob 0.5") {
    // EC {0,1} (0<->1), 0 accepting ; start 2 ->0 @0.5, ->3(sink) @0.5 ; 3 sink.
    IMDPModel m = {
        /*0*/ {{ {1,1,1} }},
        /*1*/ {{ {0,1,1} }},
        /*2*/ {{ {0,0.5,0.5}, {3,0.5,0.5} }},
        /*3*/ {{ {3,1,1} }},
    };
    auto b = maxBuchiOptimistic(m, {0}, EPS);
    CHECK(0.5*(b.lower[0]+b.upper[0]) == doctest::Approx(1.0).epsilon(1e-6));   // inside the EC
    CHECK(0.5*(b.lower[1]+b.upper[1]) == doctest::Approx(1.0).epsilon(1e-6));
    CHECK(0.5*(b.lower[2]+b.upper[2]) == doctest::Approx(0.5).epsilon(1e-3));   // start
    CHECK(0.5*(b.lower[3]+b.upper[3]) == doctest::Approx(0.0).epsilon(1e-6));   // sink
}

TEST_CASE("buchi: acceptingMECStates picks the right components") {
    // MEC {0,1} (0<->1), MEC {3} self-loop; 2 transient; 4 transient.
    // accepting = {1} -> only {0,1} is a good MEC.
    IMDPModel m = {
        /*0*/ {{ {1,1,1} }},
        /*1*/ {{ {0,1,1} }},
        /*2*/ {{ {0,1,1} }},
        /*3*/ {{ {3,1,1} }},
        /*4*/ {{ {3,1,1} }},
    };
    auto good = acceptingMECStates(m, {1});
    std::sort(good.begin(), good.end());
    CHECK(good == std::vector<int>{0, 1});
    auto good2 = acceptingMECStates(m, {3});
    CHECK(good2 == std::vector<int>{3});
    auto good3 = acceptingMECStates(m, {2});   // 2 is transient -> no good MEC
    CHECK(good3.empty());
}

// ---- Robust accepting end components (ISSUE-0009) --------------------------

TEST_CASE("buchi robust: nature routes around accepting inside a support-EC -> 0") {
    // {0,1} is a genuine support end component (every leaving edge has hi=0, so
    // nature CANNOT leave). accepting = {1}. But the edge into 1 has lo=0, so nature
    // can set P(->1)=0 forever and route the play to stay at 0 -> 1 is never visited.
    // Optimistic: support-MEC {0,1} contains 1 -> value 1.  Robust: value 0.
    IMDPModel m = {
        /*0*/ {{ {0,0.5,1.0}, {1,0.0,0.5} }},
        /*1*/ {{ {0,0.5,1.0}, {1,0.0,0.5} }},
    };
    auto bo = maxBuchiOptimistic(m, {1}, EPS);
    auto bp = maxBuchiPessimistic(m, {1}, EPS);
    CHECK(0.5*(bo.lower[0]+bo.upper[0]) == doctest::Approx(1.0).epsilon(1e-6));  // optimistic
    CHECK(0.5*(bp.lower[0]+bp.upper[0]) == doctest::Approx(0.0).epsilon(1e-6));  // robust
    CHECK(0.5*(bp.lower[1]+bp.upper[1]) == doctest::Approx(0.0).epsilon(1e-6));
    CHECK(robustBuchiWinningStates(m, {1}).empty());
}

TEST_CASE("buchi robust: accepting edge is FORCED (lo>0) -> robust value 1") {
    // Deterministic cycle 0->1->0, accepting {1}. Nature has no slack on the cycle,
    // so accepting is visited i.o. for ALL nature -> robust winning {0,1}, value 1.
    IMDPModel m = {
        /*0*/ {{ {1,1,1} }},
        /*1*/ {{ {0,1,1} }},
    };
    auto bp = maxBuchiPessimistic(m, {1}, EPS);
    CHECK(0.5*(bp.lower[0]+bp.upper[0]) == doctest::Approx(1.0).epsilon(1e-6));
    CHECK(0.5*(bp.lower[1]+bp.upper[1]) == doctest::Approx(1.0).epsilon(1e-6));
    auto win = robustBuchiWinningStates(m, {1});
    std::sort(win.begin(), win.end());
    CHECK(win == std::vector<int>{0, 1});
}

TEST_CASE("buchi robust: nature can leak the cycle to a sink -> robust value 0") {
    // 0->1 (forced); at 1 nature may divert to sink 2 (lo=0,hi=0.5). Each visit to 1
    // nature can leak, so the controller cannot force i.o. acceptance -> robust 0.
    IMDPModel m = {
        /*0*/ {{ {1,1,1} }},
        /*1*/ {{ {0,0.5,1.0}, {2,0.0,0.5} }},
        /*2*/ {{ {2,1,1} }},   // sink
    };
    auto bp = maxBuchiPessimistic(m, {0}, EPS);
    CHECK(0.5*(bp.lower[0]+bp.upper[0]) == doctest::Approx(0.0).epsilon(1e-6));
    CHECK(robustBuchiWinningStates(m, {0}).empty());
}

TEST_CASE("buchi robust: quantitative reach to a robust accepting EC = 0.5") {
    // Robust EC {0,1} (forced cycle, accepting 1). Start 2 -> EC @0.5, sink 3 @0.5.
    IMDPModel m = {
        /*0*/ {{ {1,1,1} }},
        /*1*/ {{ {0,1,1} }},
        /*2*/ {{ {0,0.5,0.5}, {3,0.5,0.5} }},
        /*3*/ {{ {3,1,1} }},
    };
    auto bp = maxBuchiPessimistic(m, {1}, EPS);
    CHECK(0.5*(bp.lower[2]+bp.upper[2]) == doctest::Approx(0.5).epsilon(1e-3));
    CHECK(0.5*(bp.lower[0]+bp.upper[0]) == doctest::Approx(1.0).epsilon(1e-6));
    CHECK(0.5*(bp.lower[3]+bp.upper[3]) == doctest::Approx(0.0).epsilon(1e-6));
}

TEST_CASE("buchi: pessimistic == optimistic on point MDPs (no nature choice)") {
    IMDPModel m = {
        /*0*/ {{ {1,0.5,0.5}, {2,0.5,0.5} }},
        /*1*/ {{ {1,1,1} }},
        /*2*/ {{ {2,1,1} }},
    };
    auto bo = maxBuchiOptimistic(m, {1}, EPS);
    auto bp = maxBuchiPessimistic(m, {1}, EPS);
    for (int s = 0; s < 3; ++s)
        CHECK(0.5*(bo.lower[s]+bo.upper[s]) == doctest::Approx(0.5*(bp.lower[s]+bp.upper[s])).epsilon(1e-4));
}

// ---- Generalized Büchi / patrol (visit several regions infinitely often) ----

TEST_CASE("genbuchi: one set reduces to ordinary Büchi") {
    IMDPModel m = {
        /*0*/ {{ {1,0.5,0.5}, {2,0.5,0.5} }},
        /*1*/ {{ {1,1,1} }},
        /*2*/ {{ {2,1,1} }},
    };
    auto g = maxGenBuchiPessimistic(m, {{1}}, EPS);
    auto b = maxBuchiPessimistic(m, {1}, EPS);
    for (int s = 0; s < 3; ++s)
        CHECK(0.5*(g.lower[s]+g.upper[s]) == doctest::Approx(0.5*(b.lower[s]+b.upper[s])).epsilon(1e-6));
}

TEST_CASE("genbuchi patrol: deterministic 3-cycle visits both regions i.o. -> 1") {
    // 0->1->2->0 forced cycle. Patrol {0} and {2}: both visited every lap -> value 1.
    IMDPModel m = {
        /*0*/ {{ {1,1,1} }},
        /*1*/ {{ {2,1,1} }},
        /*2*/ {{ {0,1,1} }},
    };
    auto g = maxGenBuchiPessimistic(m, {{0}, {2}}, EPS);
    for (int s = 0; s < 3; ++s)
        CHECK(0.5*(g.lower[s]+g.upper[s]) == doctest::Approx(1.0).epsilon(1e-6));
}

TEST_CASE("genbuchi patrol: nature can deny one region -> robust 0 but optimistic 1") {
    // 0->1 (forced); at 1 nature may divert to 2 (lo=0) or back to 0 (lo>0). 2->0.
    // Patrol {0} and {2}: robustly nature sets P(1->2)=0 forever -> never visits 2 -> 0.
    // Optimistically nature routes through 2 -> both i.o. -> 1.
    IMDPModel m = {
        /*0*/ {{ {1,1,1} }},
        /*1*/ {{ {0,0.5,1.0}, {2,0.0,0.5} }},
        /*2*/ {{ {0,1,1} }},
    };
    auto gp = maxGenBuchiPessimistic(m, {{0}, {2}}, EPS);
    auto go = maxGenBuchiOptimistic (m, {{0}, {2}}, EPS);
    CHECK(0.5*(gp.lower[0]+gp.upper[0]) == doctest::Approx(0.0).epsilon(1e-6));
    CHECK(0.5*(go.lower[0]+go.upper[0]) == doctest::Approx(1.0).epsilon(1e-6));
}

TEST_CASE("genbuchi: empty conjunction is vacuously true -> 1") {
    IMDPModel m = { /*0*/ {{ {0,1,1} }} };
    auto g = maxGenBuchiPessimistic(m, {}, EPS);
    CHECK(0.5*(g.lower[0]+g.upper[0]) == doctest::Approx(1.0).epsilon(1e-6));
}

// ---- Persistence / co-Büchi : F G p (reach-then-stay) -----------------------

TEST_CASE("persistence: reach-then-stay achievable -> 1") {
    // 0(¬p) -> 1(p) -> 2(p, safe sink). p = {1,2}. From 0 reach p-invariant region.
    IMDPModel m = {
        /*0*/ {{ {1,1,1} }},
        /*1*/ {{ {2,1,1} }},
        /*2*/ {{ {2,1,1} }},
    };
    auto r = maxPersistencePessimistic(m, {1, 2}, EPS);
    for (int s = 0; s < 3; ++s)
        CHECK(0.5*(r.lower[s]+r.upper[s]) == doctest::Approx(1.0).epsilon(1e-6));
}

TEST_CASE("persistence: nature can leak out of p each step -> robust 0.5, optimistic 1") {
    // p = {0,1} (indices). 1 is a safe p-sink. At 0 nature sends >=0.5 to 1 (stay) and
    // <=0.5 to 2 (¬p sink). Robust: 0 cannot stay in p (2 has hi>0) -> invariant region
    // {1}; reach {1} from 0 = 0.5. Optimistic: nature keeps mass in p -> stay from 0 -> 1.
    IMDPModel m = {
        /*0 (p)     */ {{ {1,0.5,1.0}, {2,0.0,0.5} }},
        /*1 (p sink)*/ {{ {1,1,1} }},
        /*2 (¬p sink)*/ {{ {2,1,1} }},
    };
    auto rp = maxPersistencePessimistic(m, {0, 1}, EPS);
    auto ro = maxPersistenceOptimistic (m, {0, 1}, EPS);
    CHECK(0.5*(rp.lower[0]+rp.upper[0]) == doctest::Approx(0.5).epsilon(1e-3));
    CHECK(0.5*(ro.lower[0]+ro.upper[0]) == doctest::Approx(1.0).epsilon(1e-6));
}

TEST_CASE("persistence: F G false = 0 ; F G true = 1") {
    IMDPModel m = { /*0*/ {{ {1,0.5,0.5}, {0,0.5,0.5} }}, /*1*/ {{ {1,1,1} }} };
    auto rf = maxPersistencePessimistic(m, {}, EPS);          // F G false
    CHECK(0.5*(rf.lower[0]+rf.upper[0]) == doctest::Approx(0.0).epsilon(1e-6));
    auto rt = maxPersistencePessimistic(m, {0, 1}, EPS);      // F G true
    CHECK(0.5*(rt.lower[0]+rt.upper[0]) == doctest::Approx(1.0).epsilon(1e-6));
}

// ---- Independent brute-force oracle for the robust a.s.-Büchi region --------
// Enumerate every memoryless controller strategy sigma. Under sigma the model is an
// interval Markov chain; nature defeats Büchi from s iff it can reach (positive
// prob, i.e. along hi>0 edges) an accepting-free "trap" D it can stay in for sure
// (every d in D: nature can keep all mass in D). The controller a.s.-wins from s
// iff SOME sigma leaves s unable to reach any such trap. W = union over sigma.
// This is structurally independent of the production nested-fixpoint algorithm.
namespace {

std::set<int> oracleBuchiWinning(const IMDPModel& m, const std::set<int>& acc) {
    const int n = (int)m.size();
    std::vector<int> nact(n);
    long total = 1;
    for (int s = 0; s < n; ++s) { nact[s] = std::max<int>(1, (int)m[s].size()); total *= nact[s]; }
    std::set<int> W;
    std::vector<int> choice(n, 0);
    for (long code = 0; code < total; ++code) {
        long c = code;
        for (int s = 0; s < n; ++s) { choice[s] = (int)(c % nact[s]); c /= nact[s]; }
        // D = greatest accepting-free set nature can stay in for sure under sigma.
        std::set<int> D;
        for (int s = 0; s < n; ++s) if (!acc.count(s)) D.insert(s);
        for (bool ch = true; ch; ) {
            ch = false;
            for (auto it = D.begin(); it != D.end(); ) {
                const ActionDist& act = m[*it][choice[*it]];
                double inHi = 0, outLo = 0;
                for (const Interval& iv : act) { if (D.count(iv.to)) inHi += iv.hi; else outLo += iv.lo; }
                if (!(inHi >= 1 - 1e-9 && outLo <= 1e-9)) { it = D.erase(it); ch = true; }
                else ++it;
            }
        }
        // Lose = states that may-reach D under sigma (hi>0 edges).
        std::set<int> lose = D;
        for (bool ch = true; ch; ) {
            ch = false;
            for (int s = 0; s < n; ++s) {
                if (lose.count(s)) continue;
                for (const Interval& iv : m[s][choice[s]])
                    if (iv.hi > 0 && lose.count(iv.to)) { lose.insert(s); ch = true; break; }
            }
        }
        for (int s = 0; s < n; ++s) if (!lose.count(s)) W.insert(s);
    }
    return W;
}

struct Lcg {
    uint64_t s;
    uint32_t next() { s = s * 6364136223846793005ULL + 1442695040888963407ULL; return (uint32_t)(s >> 33); }
    int in(int a, int b) { return a + (int)(next() % (uint32_t)(b - a + 1)); }
};

} // namespace

TEST_CASE("buchi robust: differential vs brute-force strategy-enumeration oracle") {
    Lcg rng{0x9E3779B97F4A7C15ULL};
    int mism = 0, ran = 0;
    for (int t = 0; t < 4000; ++t) {
        int n = rng.in(2, 4);
        IMDPModel m(n);
        for (int s = 0; s < n; ++s) {
            int na = rng.in(1, 2);
            for (int a = 0; a < na; ++a) {
                int k = rng.in(1, 3);
                ActionDist act;
                double sumHi = 0;
                for (int e = 0; e < k; ++e) {
                    int to = rng.in(0, n - 1);
                    double lo = (rng.in(0, 2) == 0) ? (rng.in(1, 4) / 10.0) : 0.0;  // often lo=0
                    double hi = lo + rng.in(0, 10) / 10.0;
                    if (hi > 1) hi = 1;
                    if (hi < lo) hi = lo;
                    act.push_back({to, lo, hi});
                    sumHi += hi;
                }
                if (sumHi < 1.0 && !act.empty()) act[0].hi = 1.0;   // ensure a feasible mass-1 resolution
                m[s].push_back(std::move(act));
            }
        }
        // try a few accepting sets per model
        for (int q = 0; q < 3; ++q) {
            std::set<int> acc;
            for (int s = 0; s < n; ++s) if (rng.in(0, 2) == 0) acc.insert(s);
            if (acc.empty()) acc.insert(rng.in(0, n - 1));
            auto win = robustBuchiWinningStates(m, acc);
            std::set<int> got(win.begin(), win.end());
            std::set<int> want = oracleBuchiWinning(m, acc);
            ran++;
            if (got != want) { ++mism; }
            CHECK(got == want);
        }
    }
    CHECK(mism == 0);
    CHECK(ran > 10000);
}
