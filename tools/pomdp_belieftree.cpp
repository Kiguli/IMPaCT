// pomdp_belieftree — emit the POMDP optimal-policy belief tree as JSON for the web
// visualizer. Each node is a reachable belief: its value (max P(reach target within
// the remaining horizon)), the optimal action, and the (obs, P(obs)) edge from its
// parent. Only the optimal action is expanded (the synthesized-policy tree).
//
// Build: c++ -std=c++17 -O2 tools/pomdp_belieftree.cpp \
//   src/pomdp_io.cpp src/pomdp.cpp -o tools/pomdp_belieftree
#include "../src/pomdp_io.h"
#include "../src/pomdp.h"
#include <cstdio>
#include <vector>
#include <set>

using namespace impact;

struct Node { int id, depth, parent, obs, action; double prob, value; std::vector<double> belief; };

static int build(const pomdp::POMDP& p, const std::set<int>& target, const std::vector<double>& b,
                 int t, int parent, int obs, double prob, std::vector<Node>& nodes, int cap) {
    int id = (int)nodes.size();
    if (id > cap) return -1;
    nodes.push_back({ id, t, parent, obs, -1, prob, pomdp::maxReachFromBelief(p, b, target, t), b });
    if (t == 0) return id;
    // optimal action at this belief
    int bestA = 0; double bestV = -1;
    for (int a = 0; a < p.nActions; ++a) {
        double v = 0;
        for (int o = 0; o < p.nObs; ++o) {
            double po; auto bn = pomdp::beliefUpdate(p, b, a, o, target, true, &po);
            if (po > 1e-12) v += po * pomdp::maxReachFromBelief(p, bn, target, t - 1);
        }
        if (v > bestV) { bestV = v; bestA = a; }
    }
    nodes[id].action = bestA;
    for (int o = 0; o < p.nObs; ++o) {
        double po; auto bn = pomdp::beliefUpdate(p, b, bestA, o, target, true, &po);
        if (po > 1e-9) build(p, target, bn, t - 1, id, o, po, nodes, cap);
    }
    return id;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: pomdp_belieftree MODEL.pomdp [horizon] [bound]\n"); return 2; }
    pomdp_io::Parsed parsed;
    try { parsed = pomdp_io::parseFile(argv[1]); }
    catch (const std::exception& e) { fprintf(stderr, "parse error: %s\n", e.what()); return 1; }
    int H = (argc >= 3 && atoi(argv[2]) > 0) ? atoi(argv[2]) : parsed.horizon;
    if (H > 6) H = 6;   // keep the tree small for the demo

    std::vector<Node> nodes;
    build(parsed.pomdp, parsed.target, parsed.pomdp.b0, H, -1, -1, 1.0, nodes, 2000);

    printf("{\n  \"horizon\": %d, \"nStates\": %d, \"nodes\": [\n", H, parsed.pomdp.nStates);
    for (size_t i = 0; i < nodes.size(); ++i) {
        const Node& n = nodes[i];
        printf("    {\"id\":%d,\"depth\":%d,\"parent\":%d,\"obs\":%d,\"action\":%d,\"prob\":%.6f,\"value\":%.6f,\"belief\":[",
               n.id, n.depth, n.parent, n.obs, n.action, n.prob, n.value);
        for (size_t s = 0; s < n.belief.size(); ++s) printf("%s%.4f", s ? "," : "", n.belief[s]);
        printf("]}%s\n", i + 1 < nodes.size() ? "," : "");
    }
    printf("  ]\n}\n");
    return 0;
}
