#include "exact.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <map>
#include <set>
#include <vector>

namespace impact {
namespace exact {

namespace {

// ---- exact rationals (long long, __int128 intermediates, overflow-checked) ----------
struct Rat {
    long long n = 0, d = 1;
};
Rat norm(__int128 n, __int128 d) {
    if (d == 0) throw std::runtime_error("exact: division by zero");
    if (d < 0) { n = -n; d = -d; }
    __int128 a = n < 0 ? -n : n, b = d;
    while (b) { __int128 t = a % b; a = b; b = t; }
    if (a) { n /= a; d /= a; }
    const __int128 LIM = (__int128)1 << 62;
    if (n > LIM || n < -LIM || d > LIM) throw std::runtime_error("exact: rational overflow (model too large for long long rationals)");
    return { (long long)n, (long long)d };
}
Rat add(Rat a, Rat b) { return norm((__int128)a.n * b.d + (__int128)b.n * a.d, (__int128)a.d * b.d); }
Rat sub(Rat a, Rat b) { return norm((__int128)a.n * b.d - (__int128)b.n * a.d, (__int128)a.d * b.d); }
Rat mul(Rat a, Rat b) { return norm((__int128)a.n * b.n, (__int128)a.d * b.d); }
Rat divr(Rat a, Rat b) { return norm((__int128)a.n * b.d, (__int128)a.d * b.n); }
int cmp(Rat a, Rat b) { __int128 l = (__int128)a.n * b.d, r = (__int128)b.n * a.d; return l < r ? -1 : l > r ? 1 : 0; }
bool isZero(Rat a) { return a.n == 0; }
Rat one() { return {1, 1}; }
std::string str(Rat a) { return a.d == 1 ? std::to_string(a.n) : std::to_string(a.n) + "/" + std::to_string(a.d); }
double dec(Rat a) { return (double)a.n / (double)a.d; }

// parse a decimal or fraction token exactly: "0.25" -> 1/4, "1", "2/5"
Rat parseRat(const std::string& s) {
    auto sl = s.find('/');
    if (sl != std::string::npos) return norm(std::stoll(s.substr(0, sl)), std::stoll(s.substr(sl + 1)));
    auto dot = s.find('.');
    if (dot == std::string::npos) return norm(std::stoll(s), 1);
    std::string ip = s.substr(0, dot), fp = s.substr(dot + 1);
    __int128 den = 1; for (size_t i = 0; i < fp.size(); ++i) den *= 10;
    __int128 num = (__int128)std::stoll(ip.empty() ? "0" : ip) * den + std::stoll(fp.empty() ? "0" : fp);
    return norm(num, den);
}

// ---- exact .imdp model --------------------------------------------------------------
struct Edge { int to; Rat lo, hi; };
struct XModel {
    int nStates = 0, init = 0;
    std::vector<std::vector<std::vector<Edge>>> act;   // act[s][a] = interval distribution
    std::set<int> targets;
};

XModel parse(const std::string& path, const std::string& targetLabel) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("exact: cannot open " + path);
    XModel m; std::string line;
    while (std::getline(f, line)) {
        auto h = line.find('#');
        if (h != std::string::npos) line = line.substr(0, h);
        std::istringstream is(line); std::string kw;
        if (!(is >> kw)) continue;
        if (kw == "states") { is >> m.nStates; m.act.assign(m.nStates, {}); }
        else if (kw == "init") is >> m.init;
        else if (kw == "label") { std::string nm; is >> nm; int s;
            while (is >> s) if (nm == targetLabel) m.targets.insert(s); }
        else if (kw == "reward") { int s; double v; is >> s >> v; }  // ignored here
        else if (kw == "tran") {
            int s, a; is >> s >> a;
            if ((int)m.act[s].size() <= a) m.act[s].resize(a + 1);
            std::string t;
            while (is >> t) {
                auto c1 = t.find(':'), c2 = t.rfind(':');
                m.act[s][a].push_back({ std::stoi(t.substr(0, c1)),
                                        parseRat(t.substr(c1 + 1, c2 - c1 - 1)),
                                        parseRat(t.substr(c2 + 1)) });
            }
        }
    }
    if (m.nStates == 0) throw std::runtime_error("exact: missing states");
    return m;
}

// ---- exact O-maximization: nature's extremal vertex given exact values V ------------
// returns the chosen distribution p over the edges (Givan-Leach-Dean sort-and-assign).
std::vector<Rat> omaxVertex(const std::vector<Edge>& e, const std::vector<Rat>& V, bool natureMin) {
    const int k = (int)e.size();
    std::vector<int> ord(k); std::iota(ord.begin(), ord.end(), 0);
    std::sort(ord.begin(), ord.end(), [&](int i, int j) {
        int c = cmp(V[e[i].to], V[e[j].to]);
        return natureMin ? c < 0 : c > 0;                 // min: ascending; max: descending
    });
    std::vector<Rat> p(k);
    Rat residual = one();
    for (int i = 0; i < k; ++i) { p[i] = e[i].lo; residual = sub(residual, e[i].lo); }
    for (int idx : ord) {
        if (cmp(residual, {0,1}) <= 0) break;
        Rat room = sub(e[idx].hi, e[idx].lo);
        Rat take = cmp(room, residual) < 0 ? room : residual;
        p[idx] = add(p[idx], take);
        residual = sub(residual, take);
    }
    if (!isZero(residual))
        throw std::runtime_error(cmp(residual, {0,1}) > 0
            ? "exact: infeasible interval row (sum hi < 1)"
            : "exact: infeasible interval row (sum lo > 1)");
    return p;
}
Rat dot(const std::vector<Edge>& e, const std::vector<Rat>& p, const std::vector<Rat>& V) {
    Rat s{0,1};
    for (size_t i = 0; i < e.size(); ++i) s = add(s, mul(p[i], V[e[i].to]));
    return s;
}

// ---- exact robust prob-0 set (pessimistic): largest target-free C s.t. every action
// from C can be confined to C by nature (sum_{C} hi >= 1 and lo = 0 outside C). For the
// optimistic sense: C = states with no support path to target. --------------------------
std::vector<char> probZero(const XModel& m, bool pessimistic) {
    const int n = m.nStates;
    std::vector<char> inC(n, 1);
    for (int t : m.targets) inC[t] = 0;
    bool changed = true;
    while (changed) {
        changed = false;
        for (int s = 0; s < n; ++s) {
            if (!inC[s]) continue;
            bool allConfinable = true;                      // pess: for ALL actions nature confines
            bool anySupportEscape = false;                  // opt: value 0 iff no support edge out of C
            for (const auto& a : m.act[s]) {
                Rat sumHiIn{0,1}; bool loOutZero = true;
                for (const Edge& ed : a) {
                    if (inC[ed.to]) sumHiIn = add(sumHiIn, ed.hi);
                    else { if (!isZero(ed.lo)) loOutZero = false;
                           if (!isZero(ed.hi)) anySupportEscape = true; }
                }
                if (!(loOutZero && cmp(sumHiIn, one()) >= 0)) allConfinable = false;
            }
            if (m.act[s].empty()) continue;                 // actionless: stays value 0
            const bool stays = pessimistic ? allConfinable : !anySupportEscape;
            if (!stays) { inC[s] = 0; changed = true; }
        }
    }
    return inC;                                             // 1 = provably value 0
}

// ---- exact linear solve of the induced chain: V(s) = sum p(s,s') V(s'), targets 1,
// zero-set 0, chain-unreachable-to-target 0 (Gaussian elimination over rationals) ------
std::vector<Rat> evalChain(int n, const std::set<int>& targets,
                           const std::vector<std::vector<std::pair<int,Rat>>>& P,
                           const std::vector<char>& pinnedZero) {
    // states that can reach a target in the support of P
    std::vector<std::vector<int>> pred(n);
    for (int s = 0; s < n; ++s) for (auto& [t, pr] : P[s]) if (!isZero(pr)) pred[t].push_back(s);
    std::vector<char> reach(n, 0); std::vector<int> stack(targets.begin(), targets.end());
    for (int t : targets) reach[t] = 1;
    while (!stack.empty()) { int u = stack.back(); stack.pop_back();
        for (int v : pred[u]) if (!reach[v] && !pinnedZero[v]) { reach[v] = 1; stack.push_back(v); } }

    std::vector<int> idx(n, -1); std::vector<int> vars;
    for (int s = 0; s < n; ++s)
        if (reach[s] && !targets.count(s) && !pinnedZero[s]) { idx[s] = (int)vars.size(); vars.push_back(s); }
    const int k = (int)vars.size();
    // A x = b with A = I - P (restricted), b = P(s, targets)
    std::vector<std::vector<Rat>> A(k, std::vector<Rat>(k, Rat{0,1}));
    std::vector<Rat> b(k, Rat{0,1});
    for (int i = 0; i < k; ++i) {
        A[i][i] = one();
        for (auto& [t, pr] : P[vars[i]]) {
            if (targets.count(t)) b[i] = add(b[i], pr);
            else if (idx[t] >= 0) A[i][idx[t]] = sub(A[i][idx[t]], pr);
        }
    }
    for (int c = 0; c < k; ++c) {                           // exact Gaussian elimination
        int piv = -1;
        for (int r = c; r < k; ++r) if (!isZero(A[r][c])) { piv = r; break; }
        if (piv < 0) throw std::runtime_error("exact: singular system");
        std::swap(A[c], A[piv]); std::swap(b[c], b[piv]);
        for (int r = 0; r < k; ++r) {
            if (r == c || isZero(A[r][c])) continue;
            Rat f = divr(A[r][c], A[c][c]);
            for (int cc = c; cc < k; ++cc) A[r][cc] = sub(A[r][cc], mul(f, A[c][cc]));
            b[r] = sub(b[r], mul(f, b[c]));
        }
    }
    std::vector<Rat> V(n, Rat{0,1});
    for (int t : targets) V[t] = one();
    for (int i = 0; i < k; ++i) V[vars[i]] = divr(b[i], A[i][i]);
    return V;
}

} // namespace

Result maxReach(const std::string& path, const std::string& targetLabel,
                int state, bool pessimistic, bool repair) {
    XModel m = parse(path, targetLabel);
    const int n = m.nStates;
    const int q = state >= 0 ? state : m.init;

    // ISSUE-0023 repair: rows with sum(hi) < 1 are strictly infeasible (no distribution
    // fits). In repair mode scale every hi in the row by 1/sum(hi) (exact rational),
    // recording the largest relative scaling; lo bounds are untouched (still <= hi).
    double maxRepair = 0.0;
    if (repair)
        for (auto& acts : m.act)
            for (auto& a : acts) {
                Rat sumHi{0,1}, sumLo{0,1};
                for (const Edge& e : a) { sumHi = add(sumHi, e.hi); sumLo = add(sumLo, e.lo); }
                if (!isZero(sumHi) && cmp(sumHi, one()) < 0) {       // sum(hi) < 1: scale hi up
                    Rat f = divr(one(), sumHi);
                    for (Edge& e : a) e.hi = mul(e.hi, f);
                    maxRepair = std::max(maxRepair, dec(f) - 1.0);
                }
                if (cmp(sumLo, one()) > 0) {                          // sum(lo) > 1: scale lo down
                    Rat f = divr(one(), sumLo);
                    for (Edge& e : a) e.lo = mul(e.lo, f);
                    maxRepair = std::max(maxRepair, 1.0 - dec(f));
                }
            }
    std::vector<char> p0 = probZero(m, pessimistic);
    for (int t : m.targets) p0[t] = 0;

    std::vector<Rat> V(n, Rat{0,1});
    for (int t : m.targets) V[t] = one();

    int rounds = 0;
    const int MAXROUNDS = 10000;
    for (; rounds < MAXROUNDS; ++rounds) {
        // improvement: exact O-max vertex + argmax action under current V
        std::vector<std::vector<std::pair<int,Rat>>> P(n);
        for (int s = 0; s < n; ++s) {
            if (m.targets.count(s) || p0[s] || m.act[s].empty()) continue;
            Rat best{0,1}; int bestA = -1; std::vector<Rat> bestP;
            for (size_t a = 0; a < m.act[s].size(); ++a) {
                std::vector<Rat> p = omaxVertex(m.act[s][a], V, pessimistic);
                Rat val = dot(m.act[s][a], p, V);
                if (bestA < 0 || cmp(val, best) > 0) { best = val; bestA = (int)a; bestP = std::move(p); }
            }
            for (size_t i = 0; i < m.act[s][bestA].size(); ++i)
                if (!isZero(bestP[i])) P[s].push_back({ m.act[s][bestA][i].to, bestP[i] });
        }
        // evaluation: exact value of the induced chain
        std::vector<Rat> Vn = evalChain(n, m.targets, P, p0);
        bool same = true;
        for (int s = 0; s < n && same; ++s) same = (cmp(V[s], Vn[s]) == 0);
        V.swap(Vn);
        if (same) break;
    }

    // certification: V is an exact fixpoint of the robust Bellman operator
    bool ok = true;
    for (int s = 0; s < n && ok; ++s) {
        if (m.targets.count(s)) { ok = cmp(V[s], one()) == 0; continue; }
        if (m.act[s].empty() || p0[s]) { ok = isZero(V[s]); continue; }
        Rat best{0,1}; bool any = false;
        for (const auto& a : m.act[s]) {
            std::vector<Rat> p = omaxVertex(a, V, pessimistic);
            Rat val = dot(a, p, V);
            if (!any || cmp(val, best) > 0) { best = val; any = true; }
        }
        ok = (cmp(best, V[s]) == 0);
    }
    return { str(V[q]), dec(V[q]), rounds + 1, ok, maxRepair };
}

} // namespace exact
} // namespace impact
