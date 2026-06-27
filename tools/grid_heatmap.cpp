// grid_heatmap — per-cell satisfaction-probability heatmap for a 2-D continuous
// stochastic system, via the sparse IMDP abstraction. Emits, for every quantized
// grid cell, the ROBUST (pessimistic, "min") and OPTIMISTIC ("max") probability of
// the spec (reach or safety), as two 2-D arrays for the web app to render.
//
// Build: c++ -std=c++17 -O2 tools/grid_heatmap.cpp \
//   src/system_io.cpp src/abstraction.cpp src/solve.cpp src/omaximization.cpp \
//   src/graph_utils.cpp -o tools/grid_heatmap
#include "../src/system_io.h"
#include "../src/abstraction.h"
#include "../src/solve.h"
#include "../src/imdp_io.h"
#include "../src/expr.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <iostream>

using namespace impact;

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: grid_heatmap MODEL.sys [--eps E] [--emit-imdp] [--emit-graph]\n"); return 2; }
    double eps = 1e-6; bool emitImdp = false, emitGraph = false;
    for (int i = 2; i < argc; ++i) { std::string a = argv[i];
        if (a == "--eps" && i + 1 < argc) eps = atof(argv[++i]);
        else if (a == "--emit-imdp") emitImdp = true;
        else if (a == "--emit-graph") emitGraph = true; }

    system_io::SystemSpec spec;
    try { spec = system_io::parseFile(argv[1]); }
    catch (const std::exception& e) { fprintf(stderr, "parse error: %s\n", e.what()); return 1; }

    abstraction::SystemND& s = spec.sys;
    abstraction::SparseReach ab;
    try {
        if (spec.nonlinear) {
            // Build a sound mean-enclosure from the f0/f1 expressions (interval arithmetic).
            abstraction::GridSpec g;
            g.dim_x=s.dim_x; g.dim_u=s.dim_u; g.xlb=s.xlb; g.xub=s.xub; g.eta=s.eta;
            g.ulb=s.ulb; g.uub=s.uub; g.ueta=s.ueta; g.sigma=s.sigma; g.tlo=s.tlo; g.thi=s.thi;
            std::vector<std::string> vars;
            for (int i=0;i<g.dim_x;i++) vars.push_back("x"+std::to_string(i));
            for (int i=0;i<g.dim_u;i++) vars.push_back("u"+std::to_string(i));
            std::vector<expr::Expr> fs;
            for (int i=0;i<g.dim_x;i++) fs.push_back(expr::parse(spec.fexpr[i], vars));
            abstraction::MeanBoundFn mean = [fs,g](const std::vector<double>& cl, const std::vector<double>& ch,
                                                   const std::vector<double>& u, std::vector<double>& muLo, std::vector<double>& muHi){
                std::vector<abstraction::Ival> vals;
                for (int i=0;i<g.dim_x;i++) vals.push_back(abstraction::Ival(cl[i], ch[i]));
                for (int i=0;i<g.dim_u;i++) vals.push_back(abstraction::Ival(u[i], u[i]));
                muLo.resize(g.dim_x); muHi.resize(g.dim_x);
                for (int i=0;i<g.dim_x;i++){ auto r=expr::evalInterval(fs[i], vals); muLo[i]=r.lo; muHi[i]=r.hi; }
            };
            ab = abstraction::buildSparseReachGeneral(g, mean, spec.prune);
        } else {
            ab = abstraction::buildSparseReachND(s, spec.prune);
        }
    } catch (const std::exception& e) { fprintf(stderr, "abstraction error: %s\n", e.what()); return 1; }

    const int Nx = (int)llround((s.xub[0] - s.xlb[0]) / s.eta[0]);
    const int Ny = (int)llround((s.xub[1] - s.xlb[1]) / s.eta[1]);
    if ((long long)Nx * Ny > 250000) { fprintf(stderr, "grid too large for the heatmap demo (%dx%d)\n", Nx, Ny); return 1; }

    // Export the abstracted Interval-MDP (.imdp) so it can be analysed/visualised further.
    if (emitImdp) {
        io::Problem prob;
        prob.model = ab.model;
        prob.nStates = (int)ab.model.size();
        prob.init = 0;
        prob.labels[spec.prop == "safety" ? "avoid" : "target"] = ab.targets;
        std::cout << "# IMDP abstracted from a discrete-time stochastic system ("
                  << Nx << "x" << Ny << " cells; cell c=j0+j1*" << Nx << ", +TARGET +SINK)\n";
        std::cout << io::write(prob);
        return 0;
    }

    solve::IntervalResult pess, opt;
    try {
        if (spec.prop == "safety") {
            pess = solve::maxSafetyPessimistic(ab.model, ab.targets, eps);
            opt  = solve::maxSafetyOptimistic (ab.model, ab.targets, eps);
        } else {
            pess = solve::maxReachPessimistic(ab.model, ab.targets, eps);
            opt  = solve::maxReachOptimistic (ab.model, ab.targets, eps);
        }
    } catch (const std::exception& e) { fprintf(stderr, "solve error: %s\n", e.what()); return 1; }

    // Emit the abstracted IMDP as a node-link graph (same JSON the graph renderer
    // consumes): states/init/labels/edges + per-state pessimistic & optimistic values.
    if (emitGraph) {
        const int N = (int)ab.model.size();
        const std::string lbl = (spec.prop == "safety") ? "avoid" : "target";
        printf("{\n  \"nStates\": %d, \"init\": 0, \"prop\": \"%s\",\n", N, spec.prop.c_str());
        printf("  \"labels\": {\"%s\": [", lbl.c_str());
        { bool f = true; for (int t : ab.targets) { printf("%s%d", f ? "" : ",", t); f = false; } }
        printf("]},\n  \"edges\": [");
        bool firstE = true;
        for (int from = 0; from < N; ++from)
            for (size_t a = 0; a < ab.model[from].size(); ++a)
                for (const auto& iv : ab.model[from][a]) {
                    printf("%s{\"from\":%d,\"action\":%zu,\"to\":%d,\"lo\":%.6f,\"hi\":%.6f}",
                           firstE ? "" : ",", from, a, iv.to, iv.lo, iv.hi);
                    firstE = false;
                }
        printf("],\n  \"values\": {\"pess\": [");
        for (int st = 0; st < N; ++st) printf("%s{\"lower\":%.6f,\"upper\":%.6f}", st ? "," : "", pess.lower[st], pess.upper[st]);
        printf("], \"opt\": [");
        for (int st = 0; st < N; ++st) printf("%s{\"lower\":%.6f,\"upper\":%.6f}", st ? "," : "", opt.lower[st], opt.upper[st]);
        printf("]},\n");
        // per-state descriptor: which quantized cell (region box) of the state space.
        const char* absorbName = (spec.prop == "safety") ? "AVOID" : "TARGET";
        printf("  \"descr\": [");
        for (int st = 0; st < N; ++st) {
            if (st < ab.nCells) {
                int j0 = st % Nx, j1 = st / Nx;
                double x0 = s.xlb[0] + j0 * s.eta[0], y0 = s.xlb[1] + j1 * s.eta[1];
                printf("%s\"x[%.3g,%.3g] y[%.3g,%.3g]\"", st ? "," : "", x0, x0 + s.eta[0], y0, y0 + s.eta[1]);
            } else if (st == ab.nCells) printf("%s\"%s\"", st ? "," : "", absorbName);
            else printf("%s\"SINK (off-grid)\"", st ? "," : "");
        }
        printf("]\n}\n");
        return 0;
    }

    // cell linear index c = j0 + j1*Nx (stride_0 = 1, stride_1 = Nx).
    auto dump = [&](const char* name, const solve::IntervalResult& r, bool useLower) {
        printf("  \"%s\": [", name);
        for (int j1 = 0; j1 < Ny; ++j1) {
            printf("%s[", j1 ? "," : "");
            for (int j0 = 0; j0 < Nx; ++j0) {
                int c = j0 + j1 * Nx;
                double v = useLower ? r.lower[c] : r.upper[c];
                printf("%s%.6f", j0 ? "," : "", v);
            }
            printf("]");
        }
        printf("]");
    };

    printf("{\n");
    printf("  \"nx\": %d, \"ny\": %d,\n", Nx, Ny);
    printf("  \"xlb\": %g, \"ylb\": %g, \"etax\": %g, \"etay\": %g,\n", s.xlb[0], s.xlb[1], s.eta[0], s.eta[1]);
    printf("  \"prop\": \"%s\", \"nnz\": %lld,\n", spec.prop.c_str(), ab.nnz);
    dump("min", pess, /*useLower=*/true);  printf(",\n");   // robust (pessimistic) lower bound
    dump("max", opt,  /*useLower=*/false); printf("\n");    // optimistic upper bound
    printf("}\n");
    return 0;
}
