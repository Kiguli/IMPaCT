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
//       src/imdp_io.cpp src/prism.cpp src/solve.cpp src/omaximization.cpp src/graph_utils.cpp \
//       src/omega.cpp src/pctl.cpp src/ltlspec.cpp src/ltl_spot.cpp \
//       -o tools/imdp_solve
// `ltlx` (arbitrary LTL) shells out to Spot's ltl2tgba; set IMPACT_LTL2TGBA to its path.
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
#include "../src/pctl.h"
#include "../src/ltlspec.h"
#include "../src/ltl_spot.h"
#include "../src/ctmc.h"
#include "../src/odimdp.h"
#include "../src/smc.h"
#include "../src/exact.h"
#include "../src/omaximization.h"

#include <iostream>
#include <string>
#include <set>
#include <vector>
#include <cmath>
#include <cstdlib>

// Value-iteration trace (the actual robust Bellman / O-maximization backup, applied
// from below) for the tutorial animation: returns the per-state value vector at each
// iteration. controllerMax + natureMin select the sense (reach: max/min).
static std::vector<std::vector<double>> reachTrace(const impact::solve::IMDPModel& m,
        const std::set<int>& tgt, double eps, int maxIt, bool natureMin, bool controllerMax) {
    using namespace impact;
    const int n = (int)m.size();
    std::vector<double> V(n, 0.0); for (int s : tgt) if (s>=0 && s<n) V[s] = 1.0;
    std::vector<std::vector<double>> frames; frames.push_back(V);
    std::vector<double> lo, hi, v;
    for (int it = 0; it < maxIt; ++it) {
        std::vector<double> Vn(n, 0.0); double diff = 0;
        for (int s = 0; s < n; ++s) {
            if (tgt.count(s)) { Vn[s] = 1.0; continue; }
            double best = controllerMax ? -1e18 : 1e18;
            for (const solve::ActionDist& act : m[s]) {
                lo.clear(); hi.clear(); v.clear();
                for (const solve::Interval& iv : act) { lo.push_back(iv.lo); hi.push_back(iv.hi); v.push_back(V[iv.to]); }
                double q = lo.empty() ? 0.0 : omax::optimize(lo, hi, v, natureMin ? omax::Sense::Min : omax::Sense::Max).value;
                best = controllerMax ? std::max(best, q) : std::min(best, q);
            }
            if (best < -1e17 || best > 1e17) best = 0.0;
            Vn[s] = best; diff = std::max(diff, std::fabs(Vn[s] - V[s]));
        }
        frames.push_back(Vn); V.swap(Vn);
        if (diff < eps) break;
    }
    return frames;
}

using namespace impact;

static void usage() {
    std::cerr <<
      "usage: imdp_solve MODEL(.imdp|.prism) PROP LABEL "
      "[--bound pess|opt|both] [--eps E] [--method ovi|mec] [--state S] [--discount g]\n"
      "  PROP  : reach | safety | buchi | persist | patrol | next | until | ltl | ltlx\n"
      "          | reward | lra   (reward: LABEL=target, or --discount g; lra/ltlx: LABEL=formula/-)\n"
      "  ltlx  : arbitrary LTL via Spot (deterministic (gen.)Buchi); ltl = built-in fragment\n"
      "  LABEL : a label name; for patrol/until, comma-separated labels;\n"
      "          for ltl, an LTL formula (e.g. \"F goal\", \"G F r\", \"a U b\")\n"
      "  buchi = G F label   persist = F G label   next = X label   until = a,b (a U b)\n";
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
    double discount = 1.0;   // reward prop: <1 => discounted (no target); ==1 => reachability reward
    std::string weightsArg;  // multi prop: comma-separated weights, or empty => sweep (2 objectives)
    double cslTime = 1.0;    // csl prop: the time bound t in P(F<=t goal)
    int horizon = 0;         // reach/until: step bound k (P[F<=k] / a U<=k b); 0 => unbounded
    long long samples = 100000;  // smc: number of simulated paths
    double threshold = -1.0;     // smc: if >=0, run SPRT for P >= threshold
    bool exactMode = false;      // reach: exact rational robust policy iteration
    solve::Method method = solve::Method::OptimisticVI;
    bool methodSet = false;
    int stateArg = -1;  // -1 => use model init
    bool jsonOut = false;  // emit full model structure + per-state values (for the web app)
    bool traceOut = false; // also emit the per-iteration value-iteration trace (tutorial animation)

    for (int i = 4; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char* what) -> std::string {
            if (i + 1 >= argc) { std::cerr << "missing value for " << what << "\n"; std::exit(2); }
            return argv[++i];
        };
        if      (a == "--bound")  bound = need("--bound");
        else if (a == "--eps")    eps = std::stod(need("--eps"));
        else if (a == "--discount") discount = std::stod(need("--discount"));
        else if (a == "--weights") weightsArg = need("--weights");
        else if (a == "--time")   cslTime = std::stod(need("--time"));
        else if (a == "--horizon") horizon = std::stoi(need("--horizon"));
        else if (a == "--samples") samples = std::stoll(need("--samples"));
        else if (a == "--threshold") threshold = std::stod(need("--threshold"));
        else if (a == "--exact") exactMode = true;
        else if (a == "--state")  stateArg = std::stoi(need("--state"));
        else if (a == "--json")   jsonOut = true;
        else if (a == "--trace")  { jsonOut = true; traceOut = true; }
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

    // Orthogonally-decoupled IMDP (.odimdp): factored per-dimension marginals, robust
    // reachability by recursive per-dimension O-maximization (src/odimdp.h).
    if (endsWith(path, ".odimdp")) {
        if (prop != "reach") { std::cerr << "odimdp: only `reach` is supported\n"; return 2; }
        try {
            odimdp::Model om = odimdp::parseFile(path, label);
            const int s = (stateArg >= 0) ? stateArg : om.init;
            std::cout.precision(10);
            for (const std::string& which : (bound == "both")
                     ? std::vector<std::string>{"pess", "opt"} : std::vector<std::string>{bound}) {
                int iters = 0;
                std::vector<double> V = odimdp::reach(om, eps, which == "pess", &iters);
                std::cout << "result\tprop=reach\tbound=" << which << "\tstate=" << s
                          << "\tlower=" << V[s] << "\tupper=" << V[s] << "\titers=" << iters << "\n";
            }
            return 0;
        } catch (const std::exception& e) { std::cerr << "odimdp error: " << e.what() << "\n"; return 1; }
    }

    io::Problem p;
    try {
        p = isPrism ? prism::parseFile(path) : io::parseFile(path);
    } catch (const std::exception& e) {
        std::cerr << "parse error: " << e.what() << "\n";
        return 1;
    }

    // For `ltl` the LABEL arg is an LTL formula (not a label name); `lra` (and discounted
    // reward) take no target label; every other property resolves label name(s) to states.
    const bool isLtl = (prop == "ltl" || prop == "ltlx");   // LABEL is an LTL formula
    const bool noLabel = isLtl || prop == "lra";
    std::vector<std::set<int>> accSets;
    if (!noLabel) {
        for (const std::string& nm : splitComma(label)) {
            auto it = p.labels.find(nm);
            if (it == p.labels.end()) { std::cerr << "no such label '" << nm << "' in model\n"; return 1; }
            accSets.push_back(it->second);
        }
        if (accSets.empty()) { std::cerr << "no label given\n"; return 1; }
    }
    const std::set<int> empty;
    const std::set<int>& states = noLabel ? empty : accSets.front();
    const int s = (stateArg >= 0) ? stateArg : p.init;
    if (s < 0 || s >= p.nStates) { std::cerr << "state out of range\n"; return 1; }

    // Exact rational robust reachability (policy iteration over rationals; unique among
    // tools for INTERVAL models — PRISM -exact / Storm --exact are point-only).
    if (exactMode) {
        if (prop != "reach") { std::cerr << "--exact supports `reach` only\n"; return 2; }
        try {
            std::cout.precision(10);
            for (const std::string& which : (bound == "both")
                     ? std::vector<std::string>{"pess", "opt"} : std::vector<std::string>{bound}) {
                exact::Result r = exact::maxReach(path, label, s, which == "pess");
                std::cout << "result\tprop=reach-exact\tbound=" << which << "\tstate=" << s
                          << "\texact=" << r.fraction << "\tapprox=" << r.approx
                          << "\tcertified=" << (r.certified ? "yes" : "NO") << "\titers=" << r.iterations << "\n";
            }
            return 0;
        } catch (const std::exception& e) { std::cerr << "exact error: " << e.what() << "\n"; return 1; }
    }

    // Statistical model checking (simulation) on point chains: estimate P(F<=horizon LABEL)
    // with Wilson CI + APMC/Chernoff half-width; --threshold p adds a Wald SPRT verdict.
    if (prop == "smc") {
        const int H = horizon > 0 ? horizon : 1000;
        try {
            std::cout.precision(10);
            smc::Estimate e = smc::estimateReach(p.model, states, s, H, samples, /*seed=*/20260702u);
            std::cout << "result\tprop=smc\tstate=" << s << "\testimate=" << e.mean
                      << "\twilson95=[" << e.ciLo << "," << e.ciHi << "]"
                      << "\tchernoff_eps=" << e.chernoffEps
                      << "\tsamples=" << e.samples << "\n";
            if (threshold >= 0.0) {
                long long used = 0;
                int v = smc::sprt(p.model, states, s, H, threshold, /*delta=*/0.01,
                                  samples * 10, /*seed=*/20260703u, &used);
                std::cout << "result\tprop=smc-sprt\tthreshold=" << threshold
                          << "\tverdict=" << (v > 0 ? "accept(P>=theta)" : v < 0 ? "reject(P<theta)" : "undecided")
                          << "\tsamples=" << used << "\n";
            }
            return 0;
        } catch (const std::exception& e) { std::cerr << "smc error: " << e.what() << "\n"; return 1; }
    }

    // Multi-objective robust reachability: LABEL = comma-separated target labels; --weights
    // gives one point, otherwise sweep the weight simplex (2 objectives) to trace the Pareto
    // frontier. Prints one `pareto` line per weight with the achievable objective vector.
    if (prop == "multi") {
        const bool pess = (bound != "opt");
        std::vector<std::vector<double>> weightList;
        if (!weightsArg.empty()) {
            std::vector<double> w; for (const std::string& x : splitComma(weightsArg)) w.push_back(std::stod(x));
            weightList.push_back(w);
        } else if (accSets.size() == 2) {
            for (double lam : {0.0, 0.25, 0.5, 0.75, 1.0}) weightList.push_back({lam, 1.0 - lam});
        } else { std::cerr << "multi: give --weights w1,..,wk (sweep only for 2 objectives)\n"; return 1; }
        std::cout.precision(10);
        for (const auto& w : weightList) {
            solve::MultiObjResult r = solve::multiReach(p.model, accSets, w, s, eps, pess);
            std::cout << "pareto\tbound=" << (pess ? "pess" : "opt") << "\tstate=" << s << "\tweights=";
            for (size_t i = 0; i < w.size(); ++i) std::cout << (i ? "," : "") << w[i];
            std::cout << "\tobjectives=";
            for (size_t i = 0; i < r.objective.size(); ++i) std::cout << (i ? "," : "") << r.objective[i];
            std::cout << "\tweighted=" << r.weighted << "\n";
        }
        return 0;
    }

    // Compute the property value vector for one sense ("pess"/"opt"). Throws on a
    // bad property name; lets both the text and JSON outputs share one dispatch.
    auto compute = [&](const std::string& which) -> solve::IntervalResult {
        const bool pess = (which == "pess");
        if (prop == "reach") {
            if (horizon > 0) {   // step-bounded P[F<=k label] = true U<=k label (finite-horizon, exact)
                std::set<int> all; for (int i = 0; i < p.nStates; ++i) all.insert(i);
                return pess ? pctl::boundedUntilPessimistic(p.model, all, states, horizon, eps)
                            : pctl::boundedUntilOptimistic(p.model, all, states, horizon, eps);
            }
            return pess ? (methodSet ? solve::maxReachPessimistic(p.model, states, eps, method)
                                     : solve::maxReachPessimistic(p.model, states, eps))
                        : (methodSet ? solve::maxReachOptimistic(p.model, states, eps, method)
                                     : solve::maxReachOptimistic(p.model, states, eps));
        }
        if (prop == "safety")
            return pess ? solve::maxSafetyPessimistic(p.model, states, eps)
                        : solve::maxSafetyOptimistic(p.model, states, eps);
        if (prop == "reward") { // expected reward: discounted (--discount<1) or reach-reward to LABEL
            if (discount < 1.0)
                return solve::expDiscountedReward(p.model, p.reward, discount, eps,
                                                  /*natureAdversarial=*/pess, /*controllerMax=*/true);
            return pess ? solve::maxReachRewardPessimistic(p.model, states, p.reward, eps)
                        : solve::maxReachRewardOptimistic(p.model, states, p.reward, eps);
        }
        if (prop == "lra")      // robust long-run average (mean-payoff) reward
            return pess ? solve::maxLRAPessimistic(p.model, p.reward, eps)
                        : solve::maxLRAOptimistic(p.model, p.reward, eps);
        if (prop == "ss") {     // steady-state prob of LABEL = long-run average of its indicator
            std::vector<double> ind(p.nStates, 0.0);
            for (int st : states) if (st >= 0 && st < p.nStates) ind[st] = 1.0;
            return pess ? solve::maxLRAPessimistic(p.model, ind, eps)
                        : solve::maxLRAOptimistic(p.model, ind, eps);
        }
        if (prop == "csl") {    // CTMC time-bounded reachability P(F<=--time LABEL); model = rates
            ctmc::Uniformized u = ctmc::uniformize(p.model);
            std::vector<double> v = ctmc::timeBoundedReach(u, states, cslTime, eps, /*robust=*/pess);
            solve::IntervalResult r; r.iterations = 0; r.lower = v; r.upper = v; return r;
        }
        if (prop == "buchi")
            return pess ? omega::maxBuchiPessimistic(p.model, states, eps)
                        : omega::maxBuchiOptimistic(p.model, states, eps);
        if (prop == "persist")
            return pess ? omega::maxPersistencePessimistic(p.model, states, eps)
                        : omega::maxPersistenceOptimistic(p.model, states, eps);
        if (prop == "patrol")
            return pess ? omega::maxGenBuchiPessimistic(p.model, accSets, eps)
                        : omega::maxGenBuchiOptimistic(p.model, accSets, eps);
        if (prop == "next")
            return pess ? pctl::nextPessimistic(p.model, states, eps)
                        : pctl::nextOptimistic(p.model, states, eps);
        if (prop == "until") {
            if (accSets.size() < 2) throw std::runtime_error("until needs two labels: a,b (a U b)");
            if (horizon > 0)   // a U<=k b (step-bounded)
                return pess ? pctl::boundedUntilPessimistic(p.model, accSets[0], accSets[1], horizon, eps)
                            : pctl::boundedUntilOptimistic(p.model, accSets[0], accSets[1], horizon, eps);
            return pess ? pctl::untilPessimistic(p.model, accSets[0], accSets[1], eps)
                        : pctl::untilOptimistic(p.model, accSets[0], accSets[1], eps);
        }
        if (prop == "ltl")
            return ltlspec::synthesize(p.model, p.labels, label, pess, eps);
        if (prop == "ltlx") {   // arbitrary LTL via Spot ltl2tgba (deterministic (gen.)Buchi)
            const char* cmd = std::getenv("IMPACT_LTL2TGBA");
            return ltlspot::synthesizeLTL(p.model, p.labels, p.init, label, pess, eps,
                                          cmd ? cmd : "ltl2tgba");
        }
        throw std::runtime_error("unknown property '" + prop + "'");
    };

    auto solveOne = [&](const std::string& which) -> int {
        solve::IntervalResult r;
        try { r = compute(which); }
        catch (const std::exception& e) { std::cerr << "solve error: " << e.what() << "\n"; return 1; }
        std::cout.precision(10);
        std::cout << "result\tprop=" << prop << "\tbound=" << which
                  << "\tstate=" << s
                  << "\tlower=" << r.lower[s]
                  << "\tupper=" << r.upper[s]
                  << "\titers=" << r.iterations << "\n";
        return 0;
    };

    // JSON mode: emit the full model structure (states/init/labels/edges) + per-state
    // [lower,upper] values for the requested bound(s). Consumed by the web app to
    // draw the IMDP and overlay satisfaction probabilities (small/state-capped models).
    if (jsonOut) {
        std::vector<std::string> senses = (bound == "both")
            ? std::vector<std::string>{"pess", "opt"} : std::vector<std::string>{bound};
        std::map<std::string, solve::IntervalResult> res;
        try { for (const auto& w : senses) res[w] = compute(w); }
        catch (const std::exception& e) { std::cerr << "solve error: " << e.what() << "\n"; return 1; }

        std::cout.precision(10);
        std::cout << "{\n";
        std::cout << "  \"nStates\": " << p.nStates << ",\n";
        std::cout << "  \"init\": " << p.init << ",\n";
        std::cout << "  \"prop\": \"" << prop << "\", \"label\": \"" << label << "\",\n";
        // labels
        std::cout << "  \"labels\": {";
        bool firstL = true;
        for (const auto& kv : p.labels) {
            std::cout << (firstL ? "" : ", ") << "\"" << kv.first << "\": [";
            bool f2 = true; for (int st : kv.second) { std::cout << (f2 ? "" : ",") << st; f2 = false; }
            std::cout << "]"; firstL = false;
        }
        std::cout << "},\n";
        // edges
        std::cout << "  \"edges\": [";
        bool firstE = true;
        for (int from = 0; from < p.nStates; ++from)
            for (size_t a = 0; a < p.model[from].size(); ++a)
                for (const auto& iv : p.model[from][a]) {
                    std::cout << (firstE ? "\n    " : ",\n    ")
                              << "{\"from\":" << from << ",\"action\":" << a
                              << ",\"to\":" << iv.to << ",\"lo\":" << iv.lo << ",\"hi\":" << iv.hi << "}";
                    firstE = false;
                }
        std::cout << "\n  ],\n";
        // values per sense
        std::cout << "  \"values\": {";
        bool firstS = true;
        for (const auto& w : senses) {
            const auto& r = res[w];
            std::cout << (firstS ? "\n    " : ",\n    ") << "\"" << w << "\": [";
            for (int st = 0; st < p.nStates; ++st)
                std::cout << (st ? "," : "") << "{\"lower\":" << r.lower[st] << ",\"upper\":" << r.upper[st] << "}";
            std::cout << "]"; firstS = false;
        }
        std::cout << "\n  }";

        // Value-iteration trace (tutorial animation) for reach / safety: the actual
        // robust Bellman / O-maximization backup applied iteration by iteration.
        if (traceOut && (prop == "reach" || prop == "safety")) {
            std::vector<std::vector<double>> frames;
            std::string tprop;
            if (prop == "reach") {                       // VI from below: controller max, nature min
                frames = reachTrace(p.model, states, eps, 80, /*natureMin=*/true, /*controllerMax=*/true);
                tprop = "reach (P max, robust)";
            } else {                                     // safety = 1 - reach-to-avoid (controller min, nature max)
                auto W = reachTrace(p.model, states, eps, 80, /*natureMin=*/false, /*controllerMax=*/false);
                for (auto& f : W) { for (double& x : f) x = 1.0 - x; frames.push_back(f); }
                tprop = "safety (1 - reach avoid, robust)";
            }
            std::cout << ",\n  \"trace\": {\"prop\": \"" << tprop << "\", \"frames\": [";
            for (size_t k = 0; k < frames.size(); ++k) {
                std::cout << (k ? ",\n    " : "\n    ") << "[";
                for (int st = 0; st < p.nStates; ++st) std::cout << (st ? "," : "") << frames[k][st];
                std::cout << "]";
            }
            std::cout << "]}";
        }
        std::cout << "\n}\n";
        return 0;
    }

    if (bound == "both") {
        int rc1 = solveOne("pess");
        int rc2 = solveOne("opt");
        return (rc1 || rc2) ? 1 : 0;
    }
    if (bound != "pess" && bound != "opt") { std::cerr << "bad --bound\n"; return 2; }
    return solveOne(bound);
}
