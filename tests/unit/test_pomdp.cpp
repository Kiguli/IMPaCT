// ============================================================================
// CONTRACT TESTS — POMDP finite-horizon reachability (belief-MDP value iteration).
// Verified against: fully-observable == MDP DP; no-information == best open-loop
// action sequence; a hand case; and a brute-force observation-history-policy
// enumeration oracle (the exact finite-horizon optimum) on random tiny POMDPs.
// ============================================================================
#include "../doctest.h"
#include "../contracts/contracts.h"

#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <cstdint>

using namespace impact::pomdp;

namespace {
struct Lcg { uint64_t s; uint32_t nx(){ s=s*6364136223846793005ULL+1442695040888963407ULL; return (uint32_t)(s>>33);} int in(int a,int b){return a+(int)(nx()%(uint32_t)(b-a+1));} };

// random row-stochastic vector of length n
std::vector<double> randDist(Lcg& r, int n) {
    std::vector<double> v(n); double s = 0;
    for (int i = 0; i < n; ++i) { v[i] = r.in(1, 9); s += v[i]; }
    for (int i = 0; i < n; ++i) v[i] /= s;
    return v;
}
POMDP randPomdp(Lcg& r, int nS, int nA, int nO) {
    POMDP p; p.nStates = nS; p.nActions = nA; p.nObs = nO;
    p.T.assign(nA, {}); p.O.assign(nA, {});
    for (int a = 0; a < nA; ++a) {
        for (int s = 0; s < nS; ++s) p.T[a].push_back(randDist(r, nS));
        for (int sp = 0; sp < nS; ++sp) p.O[a].push_back(randDist(r, nO));
    }
    p.b0 = randDist(r, nS);
    return p;
}

// belief-update mirror for the oracle (target absorbing), returns (post, P(o)).
std::pair<std::vector<double>,double> upd(const POMDP& p, const std::vector<double>& b, int a, int o, const std::set<int>& tgt) {
    double po = 0; auto post = beliefUpdate(p, b, a, o, tgt, true, &po); return {post, po};
}
double beliefTgt(const std::vector<double>& b, const std::set<int>& t){ double m=0; for(int s:t) m+=b[s]; return m; }

// all observation histories of length 0..H-1 (the policy's decision points)
void allHistories(int nObs, int H, std::vector<std::vector<int>>& out) {
    for (int len = 0; len < H; ++len) {
        std::vector<int> idx(len, 0);
        for (long long c = 0; ; ++c) {
            out.push_back(idx);
            int i = len - 1; while (i >= 0) { if (++idx[i] < nObs) break; idx[i] = 0; --i; }
            if (i < 0) break;
        }
        if (len == 0) { /* single empty history already pushed */ }
    }
}

double valueUnderPolicy(const POMDP& p, const std::vector<double>& b, std::vector<int> hist,
                        int t, const std::set<int>& tgt, const std::map<std::vector<int>,int>& pi) {
    if (t == 0) return beliefTgt(b, tgt);
    int a = pi.at(hist);
    double val = 0;
    for (int o = 0; o < p.nObs; ++o) {
        auto [bn, po] = upd(p, b, a, o, tgt);
        if (po <= 1e-15) continue;
        auto h2 = hist; h2.push_back(o);
        val += po * valueUnderPolicy(p, bn, h2, t - 1, tgt, pi);
    }
    return val;
}

double oracleOptimal(const POMDP& p, const std::set<int>& tgt, int H) {
    std::vector<std::vector<int>> hs; allHistories(p.nObs, H, hs);
    const int D = (int)hs.size();
    long long total = 1; for (int i = 0; i < D; ++i) total *= p.nActions;
    double best = 0;
    for (long long code = 0; code < total; ++code) {
        long long c = code; std::map<std::vector<int>,int> pi;
        for (int i = 0; i < D; ++i) { pi[hs[i]] = (int)(c % p.nActions); c /= p.nActions; }
        best = std::max(best, valueUnderPolicy(p, p.b0, {}, H, tgt, pi));
    }
    return best;
}
} // namespace

TEST_CASE("pomdp: fully observable == MDP finite-horizon reach") {
    Lcg rng{0xF00DULL};
    for (int t = 0; t < 200; ++t) {
        int nS = rng.in(2, 4), nA = rng.in(1, 2);
        POMDP p; p.nStates = nS; p.nActions = nA; p.nObs = nS;   // obs reveals the (next) state
        p.T.assign(nA, {}); p.O.assign(nA, {});
        for (int a = 0; a < nA; ++a) {
            for (int s = 0; s < nS; ++s) p.T[a].push_back(randDist(rng, nS));
            for (int sp = 0; sp < nS; ++sp) { std::vector<double> o(nS, 0.0); o[sp] = 1.0; p.O[a].push_back(o); }
        }
        int init = rng.in(0, nS - 1);
        p.b0.assign(nS, 0.0); p.b0[init] = 1.0;
        std::set<int> tgt; for (int s = 0; s < nS; ++s) if (rng.in(0, 2) == 0) tgt.insert(s);
        if (tgt.empty()) tgt.insert(rng.in(0, nS - 1));
        int H = rng.in(0, 4);
        // explicit MDP finite-horizon reach (target absorbing)
        std::vector<double> V(nS, 0.0); for (int s : tgt) V[s] = 1.0;
        for (int it = 0; it < H; ++it) {
            std::vector<double> Vn(nS, 0.0);
            for (int s = 0; s < nS; ++s) {
                if (tgt.count(s)) { Vn[s] = 1.0; continue; }
                double best = 0; for (int a = 0; a < nA; ++a) { double q = 0; for (int sp = 0; sp < nS; ++sp) q += p.T[a][s][sp]*V[sp]; best = std::max(best, q); }
                Vn[s] = best;
            }
            V = std::move(Vn);
        }
        CHECK(maxReachFiniteHorizon(p, tgt, H) == doctest::Approx(V[init]).epsilon(1e-9));
    }
}

TEST_CASE("pomdp: hand case — per-step 0.5 to target, H=2 -> 0.75") {
    POMDP p; p.nStates = 2; p.nActions = 1; p.nObs = 1;
    p.T = { { {0.5, 0.5}, {0.0, 1.0} } };       // s0->{0.5,0.5}; s1 absorbing
    p.O = { { {1.0}, {1.0} } };                 // no information
    p.b0 = { 1.0, 0.0 };
    CHECK(maxReachFiniteHorizon(p, {1}, 1) == doctest::Approx(0.5));
    CHECK(maxReachFiniteHorizon(p, {1}, 2) == doctest::Approx(0.75));
    CHECK(maxReachFiniteHorizon(p, {1}, 3) == doctest::Approx(0.875));
}

TEST_CASE("pomdp: differential vs brute-force history-policy enumeration") {
    Lcg rng{0xBEEF1ULL};
    for (int t = 0; t < 250; ++t) {
        int nS = rng.in(2, 3), nA = rng.in(1, 2), nO = rng.in(1, 2);
        POMDP p = randPomdp(rng, nS, nA, nO);
        std::set<int> tgt; for (int s = 0; s < nS; ++s) if (rng.in(0, 1)) tgt.insert(s);
        if (tgt.empty()) tgt.insert(rng.in(0, nS - 1));
        int H = rng.in(0, 3);
        double got = maxReachFiniteHorizon(p, tgt, H);
        double want = oracleOptimal(p, tgt, H);
        CHECK(got == doctest::Approx(want).epsilon(1e-9));
    }
}
