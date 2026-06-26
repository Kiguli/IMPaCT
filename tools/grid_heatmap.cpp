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
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>

using namespace impact;

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: grid_heatmap MODEL.sys [--eps E]\n"); return 2; }
    double eps = 1e-6;
    for (int i = 2; i < argc; ++i) { std::string a = argv[i]; if (a == "--eps" && i + 1 < argc) eps = atof(argv[++i]); }

    system_io::SystemSpec spec;
    try { spec = system_io::parseFile(argv[1]); }
    catch (const std::exception& e) { fprintf(stderr, "parse error: %s\n", e.what()); return 1; }

    abstraction::SystemND& s = spec.sys;
    abstraction::SparseReach ab;
    try { ab = abstraction::buildSparseReachND(s, spec.prune); }
    catch (const std::exception& e) { fprintf(stderr, "abstraction error: %s\n", e.what()); return 1; }

    const int Nx = (int)llround((s.xub[0] - s.xlb[0]) / s.eta[0]);
    const int Ny = (int)llround((s.xub[1] - s.xlb[1]) / s.eta[1]);
    if ((long long)Nx * Ny > 250000) { fprintf(stderr, "grid too large for the heatmap demo (%dx%d)\n", Nx, Ny); return 1; }

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
