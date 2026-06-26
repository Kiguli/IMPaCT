#include "ta.h"

#include <deque>
#include <vector>

namespace impact {
namespace ta {

namespace {

void applyAll(dbm::Zone& z, const std::vector<Constraint>& cs) {
    for (const Constraint& c : cs) dbm::constrain(z, c.i, c.j, c.b);
}

// Delay within an invariant: let time elapse, then re-impose the invariant.
void delayWithin(dbm::Zone& z, const std::vector<Constraint>& inv) {
    dbm::up(z);
    applyAll(z, inv);
}

} // namespace

bool reachable(const TA& ta, int target, int maxStates, bool* hitCap) {
    if (hitCap) *hitCap = false;
    const int n = ta.nClocks;

    // initial zone: all clocks 0, then delay within the initial invariant.
    dbm::Zone z0(n);
    for (int i = 1; i <= n; ++i) dbm::constrain(z0, i, 0, dbm::Bound::leq(0));  // x_i <= 0 (and >=0) => 0
    delayWithin(z0, ta.invariant[ta.init]);
    dbm::extrapolate(z0, ta.kmax);
    if (dbm::isEmpty(z0)) return false;
    if (ta.init == target) return true;

    std::vector<std::vector<dbm::Zone>> seen(ta.nLoc);   // per-location visited zones
    seen[ta.init].push_back(z0);
    std::deque<std::pair<int, dbm::Zone>> frontier;
    frontier.push_back({ta.init, z0});
    int count = 1;

    while (!frontier.empty()) {
        auto [loc, z] = frontier.front();
        frontier.pop_front();
        for (const Edge& e : ta.edges) {
            if (e.from != loc) continue;
            dbm::Zone z1 = z;
            applyAll(z1, e.guard);                 // intersect guard
            if (dbm::isEmpty(z1)) continue;
            for (int r : e.reset) dbm::reset(z1, r);
            delayWithin(z1, ta.invariant[e.to]);   // delay within target invariant
            dbm::extrapolate(z1, ta.kmax);
            if (dbm::isEmpty(z1)) continue;
            if (e.to == target) return true;

            bool subsumed = false;
            for (const dbm::Zone& prev : seen[e.to])
                if (dbm::includes(prev, z1)) { subsumed = true; break; }
            if (subsumed) continue;

            seen[e.to].push_back(z1);
            frontier.push_back({e.to, z1});
            if (++count > maxStates) { if (hitCap) *hitCap = true; return false; }
        }
    }
    return false;
}

} // namespace ta
} // namespace impact
