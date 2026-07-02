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
        else if (kw == "otran" || kw == "mtran" || kw == "mweight") {
            int s, a; is >> s >> a;
            if (s < 0 || s >= n) throw std::runtime_error("odimdp: state out of range");
            if ((int)m.actions[s].size() <= a) m.actions[s].resize(a + 1);
            Action& act = m.actions[s][a];
            auto parseEdges = [&](solve::ActionDist& dist) {
                std::string t;
                while (is >> t) {
                    auto c1 = t.find(':'), c2 = t.rfind(':');
                    dist.push_back({ std::stoi(t.substr(0, c1)),
                                     std::stod(t.substr(c1 + 1, c2 - c1 - 1)),
                                     std::stod(t.substr(c2 + 1)) });
                }
            };
            if (kw == "mweight") { act.weights.clear(); parseEdges(act.weights); }
            else {
                int k = 0, d;
                if (kw == "mtran") is >> k;
                is >> d;
                if (d < 0 || d >= (int)m.dims.size()) throw std::runtime_error("odimdp: dim out of range");
                if ((int)act.comps.size() <= k) act.comps.resize(k + 1, Marginals(m.dims.size()));
                solve::ActionDist dist; parseEdges(dist);
                act.comps[k][d] = std::move(dist);
            }
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
    std::vector<double> lo, hi, vk;
    for (const Action& act : m.actions[s]) {
        double q;
        if (act.comps.size() <= 1 || act.weights.empty()) {
            q = act.comps.empty() ? 0.0 : factoredOpt(act.comps[0], m.dims, V, sense);
        } else {
            // mixture: per-component factored O-max, then O-max over the weight intervals
            // (weights sorted by component value like any interval distribution).
            vk.assign(act.comps.size(), 0.0);
            for (size_t k = 0; k < act.comps.size(); ++k)
                vk[k] = factoredOpt(act.comps[k], m.dims, V, sense);
            lo.clear(); hi.clear();
            std::vector<double> vv;
            for (const solve::Interval& w : act.weights) {
                lo.push_back(w.lo); hi.push_back(w.hi); vv.push_back(vk[w.to]);
            }
            q = omax::optimize(lo, hi, vv, sense).value;
        }
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
