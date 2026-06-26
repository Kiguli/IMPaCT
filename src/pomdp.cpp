#include "pomdp.h"

#include <algorithm>
#include <cmath>

namespace impact {
namespace pomdp {

namespace {
// transition prob with target made absorbing (target self-loops) when `absorb`.
double trans(const POMDP& p, int a, int s, int sp, const std::set<int>& target, bool absorb) {
    if (absorb && target.count(s)) return (s == sp) ? 1.0 : 0.0;
    return p.T[a][s][sp];
}
} // namespace

std::vector<double> beliefUpdate(const POMDP& p, const std::vector<double>& b,
                                 int a, int o, const std::set<int>& target,
                                 bool absorb, double* probO) {
    const int n = p.nStates;
    std::vector<double> bp(n, 0.0);                     // predicted state dist
    for (int s = 0; s < n; ++s) {
        if (b[s] == 0.0) continue;
        for (int sp = 0; sp < n; ++sp) bp[sp] += b[s] * trans(p, a, s, sp, target, absorb);
    }
    double po = 0.0;
    std::vector<double> post(n, 0.0);
    for (int sp = 0; sp < n; ++sp) { double w = bp[sp] * p.O[a][sp][o]; post[sp] = w; po += w; }
    if (probO) *probO = po;
    if (po <= 1e-15) return {};
    for (int sp = 0; sp < n; ++sp) post[sp] /= po;
    return post;
}

namespace {
double beliefTarget(const std::vector<double>& b, const std::set<int>& target) {
    double m = 0.0; for (int s : target) if (s >= 0 && s < (int)b.size()) m += b[s]; return m;
}

double solve(const POMDP& p, const std::vector<double>& b, const std::set<int>& target, int t) {
    if (t == 0) return beliefTarget(b, target);
    double best = 0.0;
    for (int a = 0; a < p.nActions; ++a) {
        double val = 0.0;
        for (int o = 0; o < p.nObs; ++o) {
            double po = 0.0;
            std::vector<double> bn = beliefUpdate(p, b, a, o, target, /*absorb=*/true, &po);
            if (po <= 1e-15) continue;
            val += po * solve(p, bn, target, t - 1);
        }
        best = std::max(best, val);
    }
    return best;
}
} // namespace

double maxReachFromBelief(const POMDP& p, const std::vector<double>& belief,
                          const std::set<int>& target, int horizon) {
    return solve(p, belief, target, horizon);
}

double maxReachFiniteHorizon(const POMDP& p, const std::set<int>& target, int horizon) {
    return solve(p, p.b0, target, horizon);
}

} // namespace pomdp
} // namespace impact
