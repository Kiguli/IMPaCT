// ============================================================================
// imdp_solve — CLI runner for the explicit .imdp exchange format.
//
// Reads a model in the neutral .imdp format (src/imdp_io.h), solves a robust
// reachability/safety property with IMPaCT's sound interval solver, and prints
// the result in a machine-parseable form. Used by the cross-tool benchmarking
// harness (benchmarks/crosstool/) to compare IMPaCT against reference values
// computed by IntervalMDP.jl / PRISM / Storm on shared models.
//
// Build (no SYCL/Armadillo needed — pure std C++):
//   c++ -std=c++17 -O2 tools/imdp_solve.cpp \
//       src/imdp_io.cpp src/solve.cpp src/omaximization.cpp src/graph_utils.cpp \
//       -o tools/imdp_solve
//
// Usage:
//   imdp_solve MODEL.imdp PROP LABEL [--bound pess|opt|both] [--eps E]
//                                    [--method ovi|mec] [--state S]
//   PROP   : reach | safety
//   LABEL  : name of a label set declared in the model (e.g. target / avoid)
//   --bound: pess (nature adversarial, default), opt (cooperative), or both
//   --state: report value for this state index (default: the model's init)
//
// Output (one "result" line per requested bound), KEY=VALUE, tab-separated:
//   result  prop=reach  bound=pess  state=0  lower=0.399999  upper=0.400001  iters=37
// ============================================================================
#include "../src/imdp_io.h"
#include "../src/prism.h"
#include "../src/solve.h"
#include "../src/omega.h"

#include <iostream>
#include <string>
#include <set>
#include <cstdlib>

using namespace impact;

static void usage() {
    std::cerr <<
      "usage: imdp_solve MODEL(.imdp|.prism) PROP LABEL "
      "[--bound pess|opt|both] [--eps E] [--method ovi|mec] [--state S]\n"
      "  PROP  : reach | safety | buchi | persist | patrol\n"
      "  LABEL : a label name; for patrol, comma-separated labels (visit each i.o.)\n"
      "  buchi   = G F label (recurrence)   persist = F G label (reach-then-stay)\n";
}

// split "a,b,c" -> {"a","b","c"}
static std::vector<std::string> splitComma(const std::string& s) {
    std::vector<std::string> out; std::string cur;
    for (char c : s) { if (c == ',') { if (!cur.empty()) out.push_back(cur); cur.clear(); } else cur += c; }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

int main(int argc, char** argv) {
    if (argc < 4) { usage(); return 2; }
    const std::string path = argv[1];
    const std::string prop = argv[2];
    const std::string label = argv[3];

    std::string bound = "pess";
    double eps = 1e-6;
    solve::Method method = solve::Method::OptimisticVI;
    bool methodSet = false;
    int stateArg = -1;  // -1 => use model init

    for (int i = 4; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char* what) -> std::string {
            if (i + 1 >= argc) { std::cerr << "missing value for " << what << "\n"; std::exit(2); }
            return argv[++i];
        };
        if      (a == "--bound")  bound = need("--bound");
        else if (a == "--eps")    eps = std::stod(need("--eps"));
        else if (a == "--state")  stateArg = std::stoi(need("--state"));
        else if (a == "--method") {
            std::string m = need("--method");
            method = (m == "mec") ? solve::Method::MECCollapse : solve::Method::OptimisticVI;
            methodSet = true;
        } else { std::cerr << "unknown option: " << a << "\n"; usage(); return 2; }
    }

    auto endsWith = [](const std::string& s, const std::string& suf) {
        return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
    };
    const bool isPrism = endsWith(path, ".prism") || endsWith(path, ".pm") || endsWith(path, ".nm");

    io::Problem p;
    try {
        p = isPrism ? prism::parseFile(path) : io::parseFile(path);
    } catch (const std::exception& e) {
        std::cerr << "parse error: " << e.what() << "\n";
        return 1;
    }

    // Resolve label(s) -> state set(s). patrol takes several (comma-separated) labels.
    std::vector<std::set<int>> accSets;
    for (const std::string& nm : splitComma(label)) {
        auto it = p.labels.find(nm);
        if (it == p.labels.end()) { std::cerr << "no such label '" << nm << "' in model\n"; return 1; }
        accSets.push_back(it->second);
    }
    if (accSets.empty()) { std::cerr << "no label given\n"; return 1; }
    const std::set<int>& states = accSets.front();
    const int s = (stateArg >= 0) ? stateArg : p.init;
    if (s < 0 || s >= p.nStates) { std::cerr << "state out of range\n"; return 1; }

    auto solveOne = [&](const std::string& which) -> int {
        const bool pess = (which == "pess");
        solve::IntervalResult r;
        try {
            if (prop == "reach") {
                r = pess
                  ? (methodSet ? solve::maxReachPessimistic(p.model, states, eps, method)
                               : solve::maxReachPessimistic(p.model, states, eps))
                  : (methodSet ? solve::maxReachOptimistic(p.model, states, eps, method)
                               : solve::maxReachOptimistic(p.model, states, eps));
            } else if (prop == "safety") {
                r = pess ? solve::maxSafetyPessimistic(p.model, states, eps)
                         : solve::maxSafetyOptimistic(p.model, states, eps);
            } else if (prop == "buchi") {
                r = pess ? omega::maxBuchiPessimistic(p.model, states, eps)
                         : omega::maxBuchiOptimistic(p.model, states, eps);
            } else if (prop == "persist") {
                r = pess ? omega::maxPersistencePessimistic(p.model, states, eps)
                         : omega::maxPersistenceOptimistic(p.model, states, eps);
            } else if (prop == "patrol") {
                r = pess ? omega::maxGenBuchiPessimistic(p.model, accSets, eps)
                         : omega::maxGenBuchiOptimistic(p.model, accSets, eps);
            } else {
                std::cerr << "unknown property '" << prop << "'\n"; return 1;
            }
        } catch (const std::exception& e) {
            std::cerr << "solve error: " << e.what() << "\n"; return 1;
        }
        std::cout.precision(10);
        std::cout << "result\tprop=" << prop << "\tbound=" << which
                  << "\tstate=" << s
                  << "\tlower=" << r.lower[s]
                  << "\tupper=" << r.upper[s]
                  << "\titers=" << r.iterations << "\n";
        return 0;
    };

    if (bound == "both") {
        int rc1 = solveOne("pess");
        int rc2 = solveOne("opt");
        return (rc1 || rc2) ? 1 : 0;
    }
    if (bound != "pess" && bound != "opt") { std::cerr << "bad --bound\n"; return 2; }
    return solveOne(bound);
}
