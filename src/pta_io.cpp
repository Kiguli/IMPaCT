#include "pta_io.h"

#include <sstream>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <algorithm>

namespace impact {
namespace pta_io {

namespace {
[[noreturn]] void fail(const std::string& m) { throw std::runtime_error("pta_io: " + m); }

std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> o; std::string cur;
    for (char c : s) { if (c == sep) { o.push_back(cur); cur.clear(); } else cur += c; }
    o.push_back(cur);
    return o;
}
std::vector<std::string> toks(const std::string& s) {
    std::vector<std::string> o; std::istringstream is(s); std::string w; while (is >> w) o.push_back(w); return o;
}
std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n"); if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n"); return s.substr(a, b - a + 1);
}

// parse "<clk><op><int>" -> ta::Constraint (clk 1-based)
pta::Constraint parseConstraint(const std::string& t) {
    static const char* ops[] = {"<=", ">=", "<", ">"};
    for (const std::string op : ops) {
        size_t pos = t.find(op);
        if (pos == std::string::npos) continue;
        int clk = std::stoi(t.substr(0, pos));
        long long val = std::stoll(t.substr(pos + op.size()));
        if (op == "<=") return ta::clkLe(clk, val);
        if (op == ">=") return ta::clkGe(clk, val);
        if (op == "<")  return ta::clkLt(clk, val);
        return ta::clkGt(clk, val);
    }
    fail("bad constraint '" + t + "' (need <clk><op><int>)");
}
} // namespace

Parsed parse(const std::string& text) {
    Parsed out;
    pta::PTA& p = out.pta;
    int nClocks = -1, maxLoc = 0;
    std::vector<std::pair<int,long long>> kmaxPairs;
    std::vector<std::vector<pta::Constraint>> invByLoc;   // grown as needed
    auto noteLoc = [&](int l) { maxLoc = std::max(maxLoc, l); };
    auto ensureInv = [&](int l) { if ((int)invByLoc.size() <= l) invByLoc.resize(l + 1); };

    std::istringstream in(text);
    std::string line;
    std::vector<pta::Edge> edges;
    while (std::getline(in, line)) {
        auto h = line.find('#'); if (h != std::string::npos) line = line.substr(0, h);
        std::string t = trim(line); if (t.empty()) continue;
        std::istringstream is(t); std::string key; is >> key;
        if (key == "clocks") { is >> nClocks; }
        else if (key == "init") { is >> p.init; noteLoc(p.init); }
        else if (key == "target") { is >> out.target; noteLoc(out.target); }
        else if (key == "kmax") { long long a, b; while (is >> a >> b) kmaxPairs.push_back({(int)a, b}); }
        else if (key == "inv") {
            int loc; is >> loc; noteLoc(loc); ensureInv(loc);
            std::string c; while (is >> c) invByLoc[loc].push_back(parseConstraint(c));
        }
        else if (key == "edge") {
            // edge <from> | guard | branch ; branch ; ...
            std::string rest = trim(t.substr(4));
            auto fields = split(rest, '|');
            if (fields.size() < 3) fail("edge needs: from | guard | branches");
            pta::Edge e;
            e.from = std::stoi(trim(fields[0])); noteLoc(e.from);
            for (const std::string& g : toks(fields[1])) e.guard.push_back(parseConstraint(g.back()==','?g.substr(0,g.size()-1):g));
            for (const std::string& br : split(fields[2], ';')) {
                std::string b = trim(br); if (b.empty()) continue;
                auto tk = toks(b);
                if (tk.size() < 2) fail("branch needs: <prob> <toLoc> [r<clk>...]");
                pta::Branch branch;
                branch.prob = std::stod(tk[0]);
                branch.to = std::stoi(tk[1]); noteLoc(branch.to);
                for (size_t i = 2; i < tk.size(); ++i) if (tk[i].size() > 1 && tk[i][0] == 'r') branch.reset.push_back(std::stoi(tk[i].substr(1)));
                e.dist.push_back(std::move(branch));
            }
            edges.push_back(std::move(e));
        }
        else fail("unknown key '" + key + "'");
    }
    if (nClocks < 0) fail("missing 'clocks'");
    p.nClocks = nClocks;
    p.nLoc = maxLoc + 1;
    p.edges = std::move(edges);
    p.invariant.assign(p.nLoc, {});
    for (int l = 0; l < (int)invByLoc.size() && l < p.nLoc; ++l) p.invariant[l] = invByLoc[l];
    p.kmax.assign(p.nClocks + 1, 0);
    for (auto& kv : kmaxPairs) if (kv.first >= 1 && kv.first <= p.nClocks) p.kmax[kv.first] = kv.second;
    return out;
}

Parsed parseFile(const std::string& path) {
    std::ifstream f(path); if (!f) fail("cannot open " + path);
    std::stringstream ss; ss << f.rdbuf(); return parse(ss.str());
}

} // namespace pta_io
} // namespace impact
