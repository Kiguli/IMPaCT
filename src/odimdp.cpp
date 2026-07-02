#include "odimdp.h"
#include "omaximization.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

namespace impact {
namespace odimdp {

Model parseFile(const std::string& path, const std::string& targetLabel) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("odimdp: cannot open " + path);
    Model m;
    std::string line;
    long long n = 0;
    while (std::getline(f, line)) {
        auto h = line.find('#');
        if (h != std::string::npos) line = line.substr(0, h);
        std::istringstream is(line);
        std::string kw;
        if (!(is >> kw)) continue;
        if (kw == "odimdp") continue;
        if (kw == "dims") { int d; while (is >> d) m.dims.push_back(d);
            n = m.nStates(); m.actions.assign(n, {}); }
        else if (kw == "init") { is >> m.init; }
        else if (kw == "label") { std::string name; is >> name; int s;
            while (is >> s) if (name == targetLabel) m.targets.insert(s); }
        else if (kw == "otran") {
            int s, a, d; is >> s >> a >> d;
            if (s < 0 || s >= n || d < 0 || d >= (int)m.dims.size())
                throw std::runtime_error("odimdp: otran index out of range");
            if ((int)m.actions[s].size() <= a) m.actions[s].resize(a + 1, Marginals(m.dims.size()));
            solve::ActionDist dist;
            std::string t;
            while (is >> t) {
                auto c1 = t.find(':'), c2 = t.rfind(':');
                dist.push_back({ std::stoi(t.substr(0, c1)),
                                 std::stod(t.substr(c1 + 1, c2 - c1 - 1)),
                                 std::stod(t.substr(c2 + 1)) });
            }
            m.actions[s][a][d] = std::move(dist);
        }
        else throw std::runtime_error("odimdp: unknown directive '" + kw + "'");
    }
    if (m.dims.empty()) throw std::runtime_error("odimdp: missing dims");
    return m;
}

// Per-dimension reduction, LOWEST dimension first (matching IntervalMDP.jl's
// state_action_bellman: dim 1 is optimised innermost, per combination of the remaining
// coordinates; then dim 2, ...). W starts as V over the full destination product
// (linearised, dim 0 fastest); each pass optimises the current fastest dimension over
// every contiguous slice, shrinking the array by that dimension.
static double factoredOpt(const Marginals& mg, const std::vector<int>& dims,
                          const std::vector<double>& V, omax::Sense sense) {
    std::vector<double> W = V;
    std::vector<double> lo, hi, vv, Wnext;
    for (size_t d = 0; d < dims.size(); ++d) {
        const solve::ActionDist& dist = mg[d];
        const int sz = dims[d];
        const long long m = (long long)W.size() / sz;
        Wnext.assign(m, 0.0);
        for (long long q = 0; q < m; ++q) {
            lo.clear(); hi.clear(); vv.clear();
            for (const solve::Interval& iv : dist) {
                lo.push_back(iv.lo); hi.push_back(iv.hi);
                vv.push_back(W[q * sz + iv.to]);
            }
            Wnext[q] = lo.empty() ? 0.0 : omax::optimize(lo, hi, vv, sense).value;
        }
        W.swap(Wnext);
    }
    return W[0];
}

double backup(const Model& m, int s, const std::vector<double>& V, bool pessimistic) {
    const omax::Sense sense = pessimistic ? omax::Sense::Min : omax::Sense::Max;
    double best = 0.0; bool any = false;
    for (const Marginals& mg : m.actions[s]) {
        const double q = factoredOpt(mg, m.dims, V, sense);
        if (!any || q > best) { best = q; any = true; }
    }
    return any ? best : 0.0;
}

std::vector<double> reach(const Model& m, double eps, bool pessimistic, int* iters) {
    const long long n = m.nStates();
    std::vector<double> V(n, 0.0), Vn(n);
    for (int t : m.targets) V[t] = 1.0;
    const int MAXIT = 2000000;
    int it = 0;
    for (; it < MAXIT; ++it) {
        double change = 0.0;
        for (long long s = 0; s < n; ++s) {
            if (m.targets.count((int)s)) { Vn[s] = 1.0; continue; }
            Vn[s] = m.actions[s].empty() ? 0.0 : backup(m, (int)s, V, pessimistic);
            change = std::max(change, std::fabs(Vn[s] - V[s]));
        }
        V.swap(Vn);
        if (change < eps) break;
    }
    if (iters) *iters = it;
    return V;
}

} // namespace odimdp
} // namespace impact
