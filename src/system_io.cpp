#include "system_io.h"

#include <sstream>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace impact {
namespace system_io {

namespace {
[[noreturn]] void fail(const std::string& m) { throw std::runtime_error("system_io: " + m); }
std::vector<double> nums(std::istringstream& is) {
    std::vector<double> v; double x; while (is >> x) v.push_back(x); return v;
}
} // namespace

SystemSpec parse(const std::string& text) {
    SystemSpec spec;
    abstraction::SystemND& s = spec.sys;
    s.dim_x = 2; s.dim_u = 0;
    s.A = {{0,0},{0,0}}; s.c = {0.0, 0.0};
    std::vector<double> region;
    bool haveA = false, haveSigma = false, haveGrid = false, haveRegion = false;
    spec.fexpr.assign(2, "");

    std::istringstream in(text);
    std::string line;
    while (std::getline(in, line)) {
        auto h = line.find('#'); if (h != std::string::npos) line = line.substr(0, h);
        std::istringstream is(line);
        std::string key; if (!(is >> key)) continue;
        if (key == "f0" || key == "f1") {                 // nonlinear dynamics expression
            int idx = key[1] - '0';
            std::string rest; std::getline(is, rest);
            size_t eq = rest.find('='); if (eq != std::string::npos) rest = rest.substr(eq + 1);
            // trim
            size_t a = rest.find_first_not_of(" \t"); size_t b = rest.find_last_not_of(" \t");
            if (a == std::string::npos) fail("empty expression for " + key);
            spec.fexpr[idx] = rest.substr(a, b - a + 1);
            spec.nonlinear = true;
            continue;
        }
        if (key == "xlb")        { s.xlb = nums(is); }
        else if (key == "xub")   { s.xub = nums(is); haveGrid = true; }
        else if (key == "eta")   { s.eta = nums(is); }
        else if (key == "ulb")   { s.ulb = nums(is); }
        else if (key == "uub")   { s.uub = nums(is); }
        else if (key == "ueta")  { s.ueta = nums(is); }
        else if (key == "c")     { s.c = nums(is); }
        else if (key == "sigma") { s.sigma = nums(is); haveSigma = true; }
        else if (key == "prune") { is >> spec.prune; }
        else if (key == "prop")  { is >> spec.prop; }
        else if (key == "A") { auto a = nums(is); if (a.size()!=4) fail("A needs 4 numbers (row-major 2x2)"); s.A = {{a[0],a[1]},{a[2],a[3]}}; haveA = true; }
        else if (key == "B") {
            auto b = nums(is);
            if (b.size()==4) { s.B = {{b[0],b[1]},{b[2],b[3]}}; s.dim_u = 2; }
            else if (b.size()==2) { s.B = {{b[0]},{b[1]}}; s.dim_u = 1; }
            else fail("B needs 2 or 4 numbers");
        }
        else if (key == "region") { region = nums(is); if (region.size()!=4) fail("region needs 4 numbers: lo0 hi0 lo1 hi1"); haveRegion = true; }
        else if (!key.empty()) fail("unknown key '" + key + "'");
    }
    if (!haveGrid)   fail("missing xlb/xub/eta");
    if (!haveSigma)  fail("missing sigma");
    if (spec.nonlinear) { if (spec.fexpr[0].empty() || spec.fexpr[1].empty()) fail("nonlinear systems need both f0 and f1"); }
    else if (!haveA) fail("missing A (affine) or f0/f1 (nonlinear)");
    if (s.xlb.size()!=2 || s.xub.size()!=2 || s.eta.size()!=2 || s.sigma.size()!=2)
        fail("xlb/xub/eta/sigma must each have 2 entries (2-D heatmap)");
    if (s.dim_u == 0) { s.B = {{0},{0}}; s.dim_u = 1; s.ulb = {0}; s.uub = {0}; s.ueta = {1}; }  // a no-op input
    // target/avoid box: drive the absorbing region of the abstraction.
    if (haveRegion) { s.tlo = {region[0], region[2]}; s.thi = {region[1], region[3]}; }
    else            { s.tlo = {1e9, 1e9}; s.thi = {1e9+1, 1e9+1}; }   // off-grid (no absorbing region)
    return spec;
}

SystemSpec parseFile(const std::string& path) {
    std::ifstream f(path);
    if (!f) fail("cannot open " + path);
    std::stringstream ss; ss << f.rdbuf();
    return parse(ss.str());
}

} // namespace system_io
} // namespace impact
