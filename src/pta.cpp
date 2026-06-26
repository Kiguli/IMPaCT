#include "pta.h"

#include <deque>
#include <map>
#include <string>
#include <cmath>

namespace impact {
namespace pta {

namespace {

void applyAll(dbm::Zone& z, const std::vector<Constraint>& cs) {
    for (const Constraint& c : cs) dbm::constrain(z, c.i, c.j, c.b);
}
void delayWithin(dbm::Zone& z, const std::vector<Constraint>& inv) {
    dbm::up(z);
    applyAll(z, inv);
}

// Canonical key for a symbolic state: location + the (canonical) DBM bounds.
std::string key(int loc, const dbm::Zone& z) {
    std::string s = std::to_string(loc) + "|";
    for (int i = 0; i <= z.n; ++i)
        for (int j = 0; j <= z.n; ++j) {
            const dbm::Bound& b = z.m[i][j];
            s += (b.isInf() ? "I" : std::to_string(b.c)) + (b.strict ? "<" : "=") + ",";
        }
    return s;
}

} // namespace

SymbolicMDP build(const PTA& p, int targetLoc, int maxStates) {
    SymbolicMDP out;
    const int n = p.nClocks;

    std::map<std::string, int> index;
    std::vector<dbm::Zone> zones;
    std::vector<int> locs;

    auto intern = [&](int loc, dbm::Zone z) -> int {
        dbm::canonicalize(z);
        std::string k = key(loc, z);
        auto it = index.find(k);
        if (it != index.end()) return it->second;
        int id = (int)zones.size();
        index[k] = id; zones.push_back(std::move(z)); locs.push_back(loc);
        return id;
    };

    // initial symbolic state
    dbm::Zone z0(n);
    for (int i = 1; i <= n; ++i) dbm::constrain(z0, i, 0, dbm::Bound::leq(0));
    delayWithin(z0, p.invariant[p.init]);
    dbm::extrapolate(z0, p.kmax);
    const int initId = intern(p.init, z0);
    out.init = initId;

    // a single absorbing deadlock sink (for empty branches / no-edge states)
    // created lazily; index -1 location.
    int sinkId = -1;
    auto getSink = [&]() -> int {
        if (sinkId >= 0) return sinkId;
        sinkId = (int)zones.size();
        zones.push_back(dbm::Zone(n)); locs.push_back(-1);
        index["__sink__"] = sinkId;
        return sinkId;
    };

    // model grows lazily; we size it at the end. Track per-state action lists.
    std::vector<solve::StateActions> acts;
    auto ensure = [&](int id) { if ((int)acts.size() <= id) acts.resize(id + 1); };

    std::deque<int> frontier;
    frontier.push_back(initId);
    std::vector<char> expanded;
    bool cap = false;

    while (!frontier.empty()) {
        int id = frontier.front(); frontier.pop_front();
        if ((int)expanded.size() <= id) expanded.resize(id + 1, 0);
        if (expanded[id]) continue;
        expanded[id] = 1;
        ensure(id);
        const int loc = locs[id];
        if (loc < 0) { acts[id].push_back({ {id, 1.0, 1.0} }); continue; }   // sink self-loop

        bool anyEdge = false;
        for (const Edge& e : p.edges) {
            if (e.from != loc) continue;
            dbm::Zone zg = zones[id];
            applyAll(zg, e.guard);
            if (dbm::isEmpty(zg)) continue;                 // edge not enabled in this zone
            anyEdge = true;
            solve::ActionDist dist;
            std::map<int, double> agg;                      // successor symbolic state -> probability
            for (const Branch& br : e.dist) {
                dbm::Zone zk = zg;
                for (int r : br.reset) dbm::reset(zk, r);
                delayWithin(zk, p.invariant[br.to]);
                dbm::extrapolate(zk, p.kmax);
                int succ = dbm::isEmpty(zk) ? getSink() : intern(br.to, zk);
                agg[succ] += br.prob;
                if ((int)expanded.size() <= succ || !expanded[succ]) frontier.push_back(succ);
            }
            for (const auto& kv : agg) dist.push_back({ kv.first, kv.second, kv.second });
            ensure(id);
            acts[id].push_back(std::move(dist));
            if ((int)zones.size() > maxStates) { cap = true; break; }
        }
        if (!anyEdge) { ensure(id); acts[id].push_back({ {id, 1.0, 1.0} }); }   // deadlock: self-loop
        if (cap) break;
    }

    const int N = (int)zones.size();
    out.nSym = N;
    out.locOf = locs;
    out.model.assign(N, {});
    for (int s = 0; s < N; ++s) {
        if (s < (int)acts.size() && !acts[s].empty()) out.model[s] = acts[s];
        else out.model[s] = { { {s, 1.0, 1.0} } };          // unexpanded / sink -> absorbing
    }
    for (int s = 0; s < N; ++s) if (locs[s] == targetLoc) out.targets.insert(s);
    out.hitCap = cap;
    return out;
}

namespace {
// evaluate constraints at an integer clock valuation (v[0]=0, clocks 1..n).
bool satInt(const std::vector<Constraint>& cs, const std::vector<int>& v) {
    for (const Constraint& c : cs) {
        if (c.b.isInf()) continue;
        long long xi = (c.i == 0) ? 0 : v[c.i];
        long long xj = (c.j == 0) ? 0 : v[c.j];
        long long d = xi - xj;
        if (c.b.strict) { if (!(d <  c.b.c)) return false; }
        else            { if (!(d <= c.b.c)) return false; }
    }
    return true;
}
} // namespace

SymbolicMDP buildDigital(const PTA& p, int targetLoc, int maxStates) {
    SymbolicMDP out;
    const int n = p.nClocks;
    std::vector<int> capv(n + 1, 0);
    for (int i = 1; i <= n; ++i) capv[i] = (int)p.kmax[i] + 1;   // saturate above the max constant

    std::map<std::string, int> index;
    std::vector<int> locs;
    std::vector<std::vector<int>> vals;
    auto intern = [&](int loc, const std::vector<int>& v) -> int {
        std::string k = std::to_string(loc);
        for (int i = 1; i <= n; ++i) { k += ","; k += std::to_string(v[i]); }
        auto it = index.find(k);
        if (it != index.end()) return it->second;
        int id = (int)locs.size(); index[k] = id; locs.push_back(loc); vals.push_back(v);
        return id;
    };
    std::vector<int> v0(n + 1, 0);
    out.init = intern(p.init, v0);

    int sinkId = -1;
    auto getSink = [&]() -> int {
        if (sinkId >= 0) return sinkId;
        sinkId = (int)locs.size(); locs.push_back(-1); vals.push_back(std::vector<int>(n + 1, 0));
        index["__sink__"] = sinkId; return sinkId;
    };

    std::vector<solve::StateActions> acts;
    auto ensure = [&](int id) { if ((int)acts.size() <= id) acts.resize(id + 1); };
    std::deque<int> fr; fr.push_back(out.init);
    std::vector<char> expanded; bool cap = false;

    while (!fr.empty()) {
        int id = fr.front(); fr.pop_front();
        if ((int)expanded.size() <= id) expanded.resize(id + 1, 0);
        if (expanded[id]) continue;
        expanded[id] = 1; ensure(id);
        const int loc = locs[id];
        if (loc < 0) { acts[id].push_back({ {id, 1.0, 1.0} }); continue; }
        const std::vector<int> v = vals[id];

        bool anyAction = false;
        // tick (let time elapse, staying within the invariant)
        std::vector<int> v2 = v;
        for (int i = 1; i <= n; ++i) v2[i] = std::min(v[i] + 1, capv[i]);
        if (satInt(p.invariant[loc], v2)) {
            int succ = (v2 == v) ? id : intern(loc, v2);   // saturated => self-loop (time diverges)
            acts[id].push_back({ {succ, 1.0, 1.0} });
            anyAction = true;
            if (succ != id && ((int)expanded.size() <= succ || !expanded[succ])) fr.push_back(succ);
        }
        // edges
        for (const Edge& e : p.edges) {
            if (e.from != loc) continue;
            if (!satInt(e.guard, v)) continue;
            anyAction = true;
            std::map<int, double> agg;
            for (const Branch& br : e.dist) {
                std::vector<int> vk = v;
                for (int r : br.reset) vk[r] = 0;
                int succ = satInt(p.invariant[br.to], vk) ? intern(br.to, vk) : getSink();
                agg[succ] += br.prob;
                if ((int)expanded.size() <= succ || !expanded[succ]) fr.push_back(succ);
            }
            solve::ActionDist dist;
            for (const auto& kv : agg) dist.push_back({ kv.first, kv.second, kv.second });
            ensure(id); acts[id].push_back(std::move(dist));
            if ((int)locs.size() > maxStates) { cap = true; break; }
        }
        if (!anyAction) { ensure(id); acts[id].push_back({ {id, 1.0, 1.0} }); }
        if (cap) break;
    }

    const int N = (int)locs.size();
    out.nSym = N; out.locOf = locs; out.model.assign(N, {});
    for (int s = 0; s < N; ++s) {
        if (s < (int)acts.size() && !acts[s].empty()) out.model[s] = acts[s];
        else out.model[s] = { { {s, 1.0, 1.0} } };
    }
    for (int s = 0; s < N; ++s) if (locs[s] == targetLoc) out.targets.insert(s);
    out.hitCap = cap;
    return out;
}

double maxReachLocationDigital(const PTA& p, int targetLoc, double eps, int maxStates) {
    SymbolicMDP smdp = buildDigital(p, targetLoc, maxStates);
    if (smdp.targets.count(smdp.init)) return 1.0;
    if (smdp.targets.empty()) return 0.0;
    auto r = solve::maxReachPessimistic(smdp.model, smdp.targets, eps);
    return 0.5 * (r.lower[smdp.init] + r.upper[smdp.init]);
}

double minReachLocationDigital(const PTA& p, int targetLoc, double eps, int maxStates) {
    SymbolicMDP smdp = buildDigital(p, targetLoc, maxStates);
    if (smdp.targets.count(smdp.init)) return 1.0;
    if (smdp.targets.empty()) return 0.0;
    // min P(reach target) = 1 - max P(never reach target) = 1 - maxSafety(avoid=target).
    auto r = solve::maxSafetyPessimistic(smdp.model, smdp.targets, eps);
    return 1.0 - 0.5 * (r.lower[smdp.init] + r.upper[smdp.init]);
}

double maxReachLocation(const PTA& p, int targetLoc, double eps, int maxStates) {
    SymbolicMDP smdp = build(p, targetLoc, maxStates);
    if (smdp.targets.count(smdp.init)) return 1.0;
    if (smdp.targets.empty()) return 0.0;
    auto r = solve::maxReachPessimistic(smdp.model, smdp.targets, eps);  // point dists => exact Pmax
    return 0.5 * (r.lower[smdp.init] + r.upper[smdp.init]);
}

} // namespace pta
} // namespace impact
