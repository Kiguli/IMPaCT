#include "pomdp_io.h"

#include <sstream>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace impact {
namespace pomdp_io {

namespace {
[[noreturn]] void fail(const std::string& m) { throw std::runtime_error("pomdp_io: " + m); }
// parse "<i>:<p>" pairs from a stream into (index,prob).
void readPairs(std::istringstream& is, std::vector<std::pair<int,double>>& out) {
    std::string tok;
    while (is >> tok) {
        auto c = tok.find(':'); if (c == std::string::npos) fail("expected i:p, got '" + tok + "'");
        out.push_back({ std::stoi(tok.substr(0, c)), std::stod(tok.substr(c + 1)) });
    }
}
} // namespace

Parsed parse(const std::string& text) {
    Parsed out;
    pomdp::POMDP& p = out.pomdp;
    int N = -1, A = -1, O = -1;
    std::vector<std::pair<int,double>> initPairs;
    // staged rows (filled after dims known)
    struct Row { int a, s; std::vector<std::pair<int,double>> ps; };
    std::vector<Row> trows, orows;

    std::istringstream in(text); std::string line;
    while (std::getline(in, line)) {
        auto h = line.find('#'); if (h != std::string::npos) line = line.substr(0, h);
        std::istringstream is(line); std::string key; if (!(is >> key)) continue;
        if (key == "states")  is >> N;
        else if (key == "actions") is >> A;
        else if (key == "obs") is >> O;
        else if (key == "horizon") is >> out.horizon;
        else if (key == "init") readPairs(is, initPairs);
        else if (key == "target") { int s; while (is >> s) out.target.insert(s); }
        else if (key == "T") { Row r; std::string colon; is >> r.a >> r.s >> colon; readPairs(is, r.ps); trows.push_back(r); }
        else if (key == "O") { Row r; std::string colon; is >> r.a >> r.s >> colon; readPairs(is, r.ps); orows.push_back(r); }
        else fail("unknown key '" + key + "'");
    }
    if (N <= 0 || A <= 0 || O <= 0) fail("need states/actions/obs > 0");
    p.nStates = N; p.nActions = A; p.nObs = O;

    // defaults: T[a][s] = self-loop (delta_s); O[a][s'] = obs 0.
    p.T.assign(A, std::vector<std::vector<double>>(N, std::vector<double>(N, 0.0)));
    p.O.assign(A, std::vector<std::vector<double>>(N, std::vector<double>(O, 0.0)));
    for (int a = 0; a < A; ++a) for (int s = 0; s < N; ++s) { p.T[a][s][s] = 1.0; p.O[a][s][0] = 1.0; }
    for (const Row& r : trows) {
        if (r.a < 0 || r.a >= A || r.s < 0 || r.s >= N) fail("T row out of range");
        for (int s = 0; s < N; ++s) p.T[r.a][r.s][s] = 0.0;
        for (auto& kv : r.ps) { if (kv.first < 0 || kv.first >= N) fail("T target out of range"); p.T[r.a][r.s][kv.first] = kv.second; }
    }
    for (const Row& r : orows) {
        if (r.a < 0 || r.a >= A || r.s < 0 || r.s >= N) fail("O row out of range");
        for (int o = 0; o < O; ++o) p.O[r.a][r.s][o] = 0.0;
        for (auto& kv : r.ps) { if (kv.first < 0 || kv.first >= O) fail("O obs out of range"); p.O[r.a][r.s][kv.first] = kv.second; }
    }
    p.b0.assign(N, 0.0);
    if (initPairs.empty()) p.b0[0] = 1.0;
    else { for (auto& kv : initPairs) { if (kv.first < 0 || kv.first >= N) fail("init state out of range"); p.b0[kv.first] = kv.second; } }
    return out;
}

Parsed parseFile(const std::string& path) {
    std::ifstream f(path); if (!f) fail("cannot open " + path);
    std::stringstream ss; ss << f.rdbuf(); return parse(ss.str());
}

} // namespace pomdp_io
} // namespace impact
