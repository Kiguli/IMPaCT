#include "prism.h"

#include <sstream>
#include <fstream>
#include <stdexcept>
#include <map>
#include <vector>
#include <cctype>

namespace impact {
namespace prism {

namespace {

[[noreturn]] void fail(const std::string& msg) { throw std::runtime_error("prism: " + msg); }

std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

// Strip "// ..." line comments from the whole text.
std::string stripComments(const std::string& text) {
    std::ostringstream out;
    std::istringstream in(text);
    std::string line;
    while (std::getline(in, line)) {
        size_t c = line.find("//");
        if (c != std::string::npos) line = line.substr(0, c);
        out << line << "\n";
    }
    return out.str();
}

struct Consts { std::map<std::string, int> v; };

// Evaluate an integer atom: a literal or a declared const name.
int evalAtom(const std::string& tokRaw, const Consts& c) {
    std::string tok = trim(tokRaw);
    if (tok.empty()) fail("empty integer expression");
    bool numeric = (tok[0] == '-' || std::isdigit((unsigned char)tok[0]));
    if (numeric) { try { return std::stoi(tok); } catch (...) { fail("bad integer '" + tok + "'"); } }
    auto it = c.v.find(tok);
    if (it == c.v.end()) fail("unknown constant '" + tok + "'");
    return it->second;
}

// Evaluate an update RHS expression E in {int, x, x+int, x-int} at x=cur.
int evalUpdate(const std::string& exprRaw, int cur, const Consts& c) {
    std::string e = trim(exprRaw);
    if (e == "x") return cur;
    if (e.size() > 1 && e[0] == 'x' && (e[1] == '+' || e[1] == '-')) {
        int delta = evalAtom(e.substr(2), c);
        return (e[1] == '+') ? cur + delta : cur - delta;
    }
    return evalAtom(e, c);   // plain literal / const
}

// Split on `sep` at top level (ignoring chars inside () or []).
std::vector<std::string> splitTop(const std::string& s, char sep) {
    std::vector<std::string> out;
    int depth = 0; std::string cur;
    for (char ch : s) {
        if (ch == '(' || ch == '[') depth++;
        else if (ch == ')' || ch == ']') depth--;
        if (ch == sep && depth == 0) { out.push_back(cur); cur.clear(); }
        else cur += ch;
    }
    out.push_back(cur);
    return out;
}

} // namespace

io::Problem parse(const std::string& textIn) {
    const std::string text = stripComments(textIn);

    Consts consts;
    int lo = 0, hi = -1, initVal = 0;
    bool haveVar = false;
    std::string varName;
    bool sawModelType = false;

    // Collected commands: (stateIdx, action edges) — grouped per state below.
    struct Edge { int to; double plo; double phi; };
    // model[stateIdx] -> list of actions; each action a list of Edges.
    std::map<int, std::vector<std::vector<Edge>>> commands;
    std::map<std::string, std::set<int>> labels;

    auto idxOf = [&](int value) -> int {
        if (value < lo || value > hi) fail("value " + std::to_string(value) + " out of variable range");
        return value - lo;
    };

    // Iterate statements: keywords are single-line; the rest end with ';'.
    std::istringstream in(text);
    std::string raw;
    std::string buf;
    auto handleStatement = [&](std::string stmt) {
        stmt = trim(stmt);
        if (stmt.empty()) return;

        if (stmt.rfind("const", 0) == 0) {
            // const int NAME = VALUE
            std::istringstream ss(stmt);
            std::string kw, ty, name, eq;
            ss >> kw >> ty >> name >> eq;
            std::string rest; std::getline(ss, rest);
            if (eq != "=") fail("malformed const: " + stmt);
            consts.v[name] = evalAtom(rest, consts);
            return;
        }
        if (stmt.rfind("label", 0) == 0) {
            // label "NAME" = x=K | x=K2 ...
            size_t q1 = stmt.find('"'), q2 = stmt.find('"', q1 + 1);
            if (q1 == std::string::npos || q2 == std::string::npos) fail("malformed label: " + stmt);
            std::string name = stmt.substr(q1 + 1, q2 - q1 - 1);
            size_t eq = stmt.find('=', q2);
            if (eq == std::string::npos) fail("label missing '=': " + stmt);
            std::string expr = stmt.substr(eq + 1);
            for (std::string disj : splitTop(expr, '|')) {
                size_t e = disj.find('=');
                if (e == std::string::npos) fail("label clause needs x=K: " + disj);
                labels[name].insert(idxOf(evalAtom(disj.substr(e + 1), consts)));
            }
            return;
        }
        // variable decl: NAME : [LO..HI] init I
        if (!haveVar && stmt.find('[') != std::string::npos && stmt.find("..") != std::string::npos
            && stmt.find("->") == std::string::npos) {
            size_t colon = stmt.find(':');
            varName = trim(stmt.substr(0, colon));
            size_t lb = stmt.find('['), dots = stmt.find("..", lb), rb = stmt.find(']', dots);
            lo = evalAtom(stmt.substr(lb + 1, dots - lb - 1), consts);
            hi = evalAtom(stmt.substr(dots + 2, rb - dots - 2), consts);
            size_t ip = stmt.find("init", rb);
            initVal = (ip == std::string::npos) ? lo : evalAtom(stmt.substr(ip + 4), consts);
            haveVar = true;
            return;
        }
        // command: [ACT] x=K -> RHS
        if (stmt[0] == '[') {
            if (!haveVar) fail("command before variable declaration");
            size_t rb = stmt.find(']');
            if (rb == std::string::npos) fail("command missing ']': " + stmt);
            size_t arrow = stmt.find("->", rb);
            if (arrow == std::string::npos) fail("command missing '->': " + stmt);
            std::string guard = stmt.substr(rb + 1, arrow - rb - 1);
            std::string rhs = stmt.substr(arrow + 2);
            size_t e = guard.find('=');
            if (e == std::string::npos) fail("only x=K guards supported: " + guard);
            int K = evalAtom(guard.substr(e + 1), consts);
            int sIdx = idxOf(K);

            std::vector<Edge> edges;
            for (std::string term : splitTop(rhs, '+')) {
                term = trim(term);
                if (term.empty()) continue;
                double plo, phi;
                std::string upd;
                size_t lp = term.find('(');
                if (lp == std::string::npos) fail("term missing update: " + term);
                std::string probPart = trim(term.substr(0, lp));
                // strip trailing ':' if present
                if (!probPart.empty() && probPart.back() == ':') probPart.pop_back();
                probPart = trim(probPart);
                if (probPart.empty()) { plo = phi = 1.0; }            // (x'=E) => prob 1
                else if (probPart[0] == '[') {                         // [lo,hi]
                    size_t comma = probPart.find(',');
                    plo = std::stod(probPart.substr(1, comma - 1));
                    phi = std::stod(probPart.substr(comma + 1, probPart.find(']') - comma - 1));
                } else { plo = phi = std::stod(probPart); }            // point literal
                // update (x'=E)
                size_t rp = term.find(')', lp);
                std::string body = term.substr(lp + 1, rp - lp - 1);   // x'=E
                size_t eq = body.find('=');
                if (eq == std::string::npos) fail("update needs x'=E: " + body);
                std::string lhs = trim(body.substr(0, eq));
                if (lhs != varName + "'") fail("update assigns unknown var: " + lhs);
                int target = evalUpdate(body.substr(eq + 1), K, consts);
                edges.push_back({idxOf(target), plo, phi});
            }
            // merge duplicate successors within one action
            std::map<int, Edge> merged;
            for (const Edge& ed : edges) {
                auto it = merged.find(ed.to);
                if (it == merged.end()) merged[ed.to] = ed;
                else { it->second.plo += ed.plo; it->second.phi += ed.phi; }
            }
            std::vector<Edge> act;
            for (auto& kv : merged) act.push_back(kv.second);
            commands[sIdx].push_back(std::move(act));
            return;
        }
        fail("unrecognized statement: " + stmt);
    };

    while (std::getline(in, raw)) {
        std::string line = trim(raw);
        if (line.empty()) continue;
        if (line == "mdp" || line == "imdp") { sawModelType = true; continue; }
        if (line == "dtmc" || line == "ctmc" || line == "pomdp")
            fail("model type '" + line + "' not supported (use mdp)");
        if (line.rfind("module", 0) == 0) continue;   // single module; name ignored
        if (line == "endmodule") continue;
        // accumulate until ';'
        buf += " " + line;
        size_t semi;
        while ((semi = buf.find(';')) != std::string::npos) {
            handleStatement(buf.substr(0, semi));
            buf = buf.substr(semi + 1);
        }
    }
    if (!trim(buf).empty()) fail("trailing unterminated statement: " + trim(buf));
    if (!haveVar) fail("no state variable declared");
    (void)sawModelType;  // tolerated-but-not-required; subset is MDP/IMDP only

    io::Problem p;
    p.nStates = hi - lo + 1;
    p.init = idxOf(initVal);
    p.labels = std::move(labels);
    p.model.assign(p.nStates, {});
    for (auto& kv : commands) {
        for (auto& act : kv.second) {
            solve::ActionDist dist;
            for (const Edge& ed : act) dist.push_back({ed.to, ed.plo, ed.phi});
            p.model[kv.first].push_back(std::move(dist));
        }
    }
    return p;
}

io::Problem parseFile(const std::string& path) {
    std::ifstream f(path);
    if (!f) fail("cannot open " + path);
    std::stringstream ss; ss << f.rdbuf();
    return parse(ss.str());
}

} // namespace prism
} // namespace impact
