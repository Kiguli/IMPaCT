// ta_zonegraph — emit the symbolic zone graph of a (probabilistic) timed automaton
// as JSON for the web visualizer: nodes are symbolic states (location, zone),
// heat-mapped by the probability of reaching the target location; edges are the
// (probabilistic) transitions of the induced MDP.
//
// Build: c++ -std=c++17 -O2 tools/ta_zonegraph.cpp \
//   src/pta_io.cpp src/pta.cpp src/ta.cpp src/dbm.cpp src/solve.cpp \
//   src/omaximization.cpp src/graph_utils.cpp -o tools/ta_zonegraph
#include "../src/pta_io.h"
#include "../src/pta.h"
#include "../src/solve.h"
#include <cstdio>
#include <map>
#include <string>

using namespace impact;

static void jstr(const std::string& s) {   // minimal JSON string escaping
    putchar('"');
    for (char c : s) {
        if (c == '"' || c == '\\') { putchar('\\'); putchar(c); }
        else if ((unsigned char)c < 0x20) printf("\\u%04x", c);
        else putchar(c);
    }
    putchar('"');
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: ta_zonegraph MODEL.pta [targetLoc] [engine zone|digital] [bound pess|opt]\n"); return 2; }
    pta_io::Parsed parsed;
    try { parsed = pta_io::parseFile(argv[1]); }
    catch (const std::exception& e) { fprintf(stderr, "parse error: %s\n", e.what()); return 1; }

    int target = (argc >= 3 && argv[2][0] != '\0' && std::string(argv[2]) != "-") ? atoi(argv[2]) : parsed.target;
    std::string engine = (argc >= 4) ? argv[3] : "zone";
    std::string bound  = (argc >= 5) ? argv[4] : "pess";

    pta::SymbolicMDP smdp = (engine == "digital") ? pta::buildDigital(parsed.pta, target)
                                                  : pta::build(parsed.pta, target);
    solve::IntervalResult val;
    if (!smdp.targets.empty())
        val = (bound == "opt") ? solve::maxReachOptimistic(smdp.model, smdp.targets, 1e-7)
                               : solve::maxReachPessimistic(smdp.model, smdp.targets, 1e-7);

    auto value = [&](int s) -> double {
        if (smdp.targets.empty()) return 0.0;
        if (smdp.targets.count(s)) return 1.0;
        return 0.5 * (val.lower[s] + val.upper[s]);
    };

    printf("{\n  \"init\": %d, \"target\": %d, \"nNodes\": %d,\n", smdp.init, target, smdp.nSym);
    printf("  \"nodes\": [");
    for (int s = 0; s < smdp.nSym; ++s) {
        printf("%s{\"id\":%d,\"loc\":%d,\"value\":%.6f,\"target\":%s,\"descr\":",
               s ? "," : "", s, smdp.locOf[s], value(s), smdp.targets.count(s) ? "true" : "false");
        jstr(s < (int)smdp.descr.size() ? smdp.descr[s] : "");
        printf("}");
    }
    printf("],\n  \"edges\": [");
    bool first = true;
    for (int s = 0; s < smdp.nSym; ++s) {
        std::map<int, double> best;   // successor -> max prob across actions (for display)
        for (const auto& act : smdp.model[s])
            for (const auto& iv : act) { double p = 0.5 * (iv.lo + iv.hi); if (iv.to != s && p > best[iv.to]) best[iv.to] = p; }
        for (const auto& kv : best) {
            printf("%s{\"from\":%d,\"to\":%d,\"prob\":%.4f}", first ? "" : ",", s, kv.first, kv.second);
            first = false;
        }
    }
    printf("]\n}\n");
    return 0;
}
