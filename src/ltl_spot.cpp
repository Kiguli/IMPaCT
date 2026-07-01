#include "ltl_spot.h"
#include "omega.h"

#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <string>
#include <algorithm>

namespace impact {
namespace ltlspot {

namespace {

// ---- run Spot's ltl2tgba, capture the deterministic HOA automaton -------------------
std::string runLtl2tgba(const std::string& cmd, const std::string& formula) {
    // -D deterministic, -H HOA output; single-quote the formula (escape embedded quotes).
    std::string f; for (char c : formula) { if (c == '\'') f += "'\\''"; else f += c; }
    std::string full = cmd + " -D -H '" + f + "' 2>/dev/null";
    FILE* pipe = popen(full.c_str(), "r");
    if (!pipe) throw std::runtime_error("ltl_spot: cannot run '" + cmd + "'");
    std::string out; char buf[4096]; size_t n;
    while ((n = fread(buf, 1, sizeof buf, pipe)) > 0) out.append(buf, n);
    int rc = pclose(pipe);
    if (rc != 0 || out.find("--BODY--") == std::string::npos)
        throw std::runtime_error("ltl_spot: ltl2tgba failed for formula '" + formula +
                                 "' (is Spot installed / IMPACT_LTL2TGBA set?)");
    return out;
}

// ---- a parsed transition-based deterministic (generalized) Büchi automaton ----------
struct Edge { std::string guard; int dest; std::vector<int> marks; };
struct Automaton {
    int nStates = 0, start = 0, nAcc = 0;
    bool allAccept = false;                     // acceptance "t"/all => every non-stuck run accepts (safety)
    bool deterministic = false;                 // Spot marks this in `properties:` when det
    std::vector<std::string> ap;               // atomic-proposition names, HOA index order
    std::vector<std::vector<Edge>> edges;       // edges[q] (transition-based marks)
    std::vector<std::vector<int>> stateMarks;   // stateMarks[q] (state-based marks)
};

std::vector<std::string> tokenize(const std::string& s) {
    std::vector<std::string> t; std::istringstream is(s); std::string w; while (is >> w) t.push_back(w); return t;
}

// Evaluate an HOA edge guard (boolean over AP indices: & | ! t f ( ), higher precedence
// ! then & then |) against a valuation `val` (val[i] = does AP i hold).
struct GuardEval {
    const std::string& s; size_t i = 0; const std::vector<char>& val;
    GuardEval(const std::string& s_, const std::vector<char>& v) : s(s_), val(v) {}
    void ws() { while (i < s.size() && s[i] == ' ') ++i; }
    bool orE()  { bool a = andE(); ws(); while (i < s.size() && s[i] == '|') { ++i; bool b = andE(); a = a || b; ws(); } return a; }
    bool andE() { bool a = notE(); ws(); while (i < s.size() && s[i] == '&') { ++i; bool b = notE(); a = a && b; ws(); } return a; }
    bool notE() { ws(); if (i < s.size() && s[i] == '!') { ++i; return !notE(); } return atom(); }
    bool atom() {
        ws();
        if (i < s.size() && s[i] == '(') { ++i; bool r = orE(); ws(); if (i < s.size() && s[i] == ')') ++i; return r; }
        if (i < s.size() && s[i] == 't') { ++i; return true; }
        if (i < s.size() && s[i] == 'f') { ++i; return false; }
        size_t j = i; while (j < s.size() && s[j] >= '0' && s[j] <= '9') ++j;
        int idx = std::stoi(s.substr(i, j - i)); i = j;
        return idx < (int)val.size() && val[idx];
    }
};
bool guardHolds(const std::string& g, const std::vector<char>& val) {
    if (g == "t") return true; if (g == "f") return false;
    GuardEval e(g, val); return e.orE();
}

Automaton parseHOA(const std::string& hoa) {
    Automaton A; std::istringstream in(hoa); std::string line; bool body = false; int cur = -1;
    while (std::getline(in, line)) {
        if (!body) {
            if (line.rfind("AP:", 0) == 0) {
                auto t = tokenize(line);                 // AP: <n> "a" "b" ...
                for (size_t k = 2; k < t.size(); ++k) { std::string a = t[k];
                    if (a.size() >= 2 && a.front() == '"') a = a.substr(1, a.size() - 2); A.ap.push_back(a); }
            } else if (line.rfind("Start:", 0) == 0) { A.start = std::stoi(tokenize(line).at(1)); }
            else if (line.rfind("States:", 0) == 0) { A.nStates = std::stoi(tokenize(line).at(1)); }
            else if (line.rfind("Acceptance:", 0) == 0) {
                auto t = tokenize(line); A.nAcc = std::stoi(t.at(1));
                std::string cond = line.substr(line.find(t.at(1)) + t.at(1).size());
                if (cond.find("Fin(") != std::string::npos)
                    throw std::runtime_error("ltl_spot: automaton acceptance is not (generalized) Buchi "
                                             "(contains Fin -> needs LDBA/parity, ISSUE-0016)");
            } else if (line.rfind("acc-name:", 0) == 0) {
                std::string n = line.substr(9);
                if (n.find("Buchi") == std::string::npos && n.find("generalized-Buchi") == std::string::npos
                    && n.find("all") == std::string::npos && n.find("none") == std::string::npos)
                    throw std::runtime_error("ltl_spot: acceptance '" + n + "' not (generalized) Buchi (ISSUE-0016)");
            } else if (line.rfind("properties:", 0) == 0) {
                if (line.find(" deterministic") != std::string::npos) A.deterministic = true;
            } else if (line.rfind("--BODY--", 0) == 0) {
                body = true; A.edges.assign(A.nStates, {}); A.stateMarks.assign(A.nStates, {});
            }
        } else {
            if (line.rfind("State:", 0) == 0) {
                cur = std::stoi(tokenize(line).at(1));
                size_t lb = line.find('{');       // state-based acceptance marks: "State: q {marks}"
                if (lb != std::string::npos) { std::string mk = line.substr(lb + 1, line.find('}') - lb - 1);
                    std::istringstream ms(mk); int x; while (ms >> x) A.stateMarks.at(cur).push_back(x); }
            }
            else if (line.rfind("--END--", 0) == 0) break;
            else if (!line.empty() && line[0] == '[') {
                Edge e; size_t rb = line.find(']');
                e.guard = line.substr(1, rb - 1);
                std::string rest = line.substr(rb + 1);
                std::istringstream rs(rest); rs >> e.dest;
                size_t lb = rest.find('{');
                if (lb != std::string::npos) { std::string m = rest.substr(lb + 1, rest.find('}') - lb - 1);
                    std::istringstream ms(m); int x; while (ms >> x) e.marks.push_back(x); }
                A.edges.at(cur).push_back(std::move(e));
            }
        }
    }
    if (A.nAcc == 0) A.allAccept = true;   // "all"/"t": every non-stuck run accepts (=> safety of the product)
    return A;
}

} // namespace

solve::IntervalResult synthesizeLTL(const solve::IMDPModel& m,
                                    const std::map<std::string, std::set<int>>& labels,
                                    int init, const std::string& formula,
                                    bool pessimistic, double eps, const std::string& ltl2tgba) {
    const Automaton A = parseHOA(runLtl2tgba(ltl2tgba, formula));
    if (!A.deterministic && A.nStates > 1)
        throw std::runtime_error("ltl_spot: '" + formula + "' compiles to a NON-deterministic "
            "automaton (e.g. co-Buchi / F G-type) — sound only via an LDBA (ISSUE-0016). Use the "
            "`ltl` fragment solver (it handles F/G/U/X/GF/FG/persistence) for such formulas.");
    const int n = (int)m.size(), nQ = A.nStates;

    // AP valuation of each IMDP state: which of the automaton's APs hold there.
    std::vector<std::vector<char>> val(n, std::vector<char>(A.ap.size(), 0));
    for (size_t a = 0; a < A.ap.size(); ++a) {
        auto it = labels.find(A.ap[a]);
        if (it != labels.end()) for (int s : it->second) if (s >= 0 && s < n) val[s][a] = 1;
    }
    // Deterministic automaton step from q on the valuation of IMDP state s.
    auto step = [&](int q, int s, std::vector<int>& marks) -> int {
        for (const Edge& e : A.edges[q]) if (guardHolds(e.guard, val[s])) { marks = e.marks; return e.dest; }
        marks.clear(); return -1;   // no matching edge => reject sink
    };

    // Synchronous product: product state (s,q) -> index s*nQ + q, plus one reject sink.
    const int SINK = n * nQ;
    solve::IMDPModel P(n * nQ + 1);
    std::vector<std::set<int>> accSets(std::max(1, A.nAcc));
    for (int s = 0; s < n; ++s)
        for (int q = 0; q < nQ; ++q) {
            const int ps = s * nQ + q;
            std::vector<int> marks;
            const int q2 = step(q, s, marks);
            // acceptance mark of (s,q): state-based marks of q PLUS the taken edge's marks
            for (int j : A.stateMarks[q]) if (j < (int)accSets.size()) accSets[j].insert(ps);
            for (int j : marks) if (j < (int)accSets.size()) accSets[j].insert(ps);
            if (q2 < 0) { P[ps].push_back({ solve::Interval{SINK, 1.0, 1.0} }); continue; }
            for (const solve::ActionDist& act : m[s]) {
                solve::ActionDist pa; pa.reserve(act.size());
                for (const solve::Interval& iv : act) pa.push_back({ iv.to * nQ + q2, iv.lo, iv.hi });
                P[ps].push_back(std::move(pa));
            }
            if (m[s].empty()) P[ps].push_back({ solve::Interval{ps, 1.0, 1.0} });  // absorbing
        }
    P[SINK].push_back({ solve::Interval{SINK, 1.0, 1.0} });

    // "all"/"t" acceptance: every run that never gets STUCK (reaches the reject sink) accepts,
    // so the LTL value is robust SAFETY of the product w.r.t. the sink. Otherwise (generalized)
    // Büchi on the accepting mark sets.
    solve::IntervalResult r;
    if (A.allAccept)
        r = pessimistic ? solve::maxSafetyPessimistic(P, {SINK}, eps)
                        : solve::maxSafetyOptimistic(P, {SINK}, eps);
    else
        r = pessimistic ? omega::maxGenBuchiPessimistic(P, accSets, eps)
                        : omega::maxGenBuchiOptimistic(P, accSets, eps);
    // Report at the initial product state (init, start); remap to IMDP-state indexing so the
    // caller's `state` index still selects the initial value.
    const int ip = init * nQ + A.start;
    solve::IntervalResult out; out.iterations = r.iterations;
    out.lower.assign(n, 0.0); out.upper.assign(n, 0.0);
    out.lower[init] = r.lower[ip]; out.upper[init] = r.upper[ip];
    return out;
}

} // namespace ltlspot
} // namespace impact
