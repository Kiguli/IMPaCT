#include "imdp_io.h"

#include <sstream>
#include <fstream>
#include <stdexcept>

namespace impact {
namespace io {

namespace {
std::vector<std::string> tokens(const std::string& line) {
    std::vector<std::string> t;
    std::istringstream is(line);
    std::string w;
    while (is >> w) t.push_back(w);
    return t;
}
// parse "to:lo:hi" -> Interval
solve::Interval parseEdge(const std::string& s) {
    auto a = s.find(':'); auto b = s.rfind(':');
    if (a == std::string::npos || b == a) throw std::runtime_error("imdp_io: bad edge '" + s + "'");
    solve::Interval iv;
    iv.to = std::stoi(s.substr(0, a));
    iv.lo = std::stod(s.substr(a + 1, b - a - 1));
    iv.hi = std::stod(s.substr(b + 1));
    return iv;
}
} // namespace

Problem parse(const std::string& text) {
    Problem p;
    std::istringstream in(text);
    std::string line;
    bool haveStates = false;
    while (std::getline(in, line)) {
        auto h = line.find('#');
        if (h != std::string::npos) line = line.substr(0, h);
        std::vector<std::string> t = tokens(line);
        if (t.empty()) continue;
        if (t[0] == "states") {
            p.nStates = std::stoi(t.at(1));
            p.model.assign(p.nStates, {});
            p.reward.assign(p.nStates, 0.0);
            haveStates = true;
        } else if (t[0] == "init") {
            p.init = std::stoi(t.at(1));
        } else if (t[0] == "label") {
            if (t.size() < 2) throw std::runtime_error("imdp_io: label needs a name");
            for (size_t i = 2; i < t.size(); ++i) p.labels[t[1]].insert(std::stoi(t[i]));
        } else if (t[0] == "reward") {
            if (!haveStates) throw std::runtime_error("imdp_io: 'reward' before 'states'");
            int s = std::stoi(t.at(1));
            if (s < 0 || s >= p.nStates) throw std::runtime_error("imdp_io: reward state out of range");
            p.reward[s] = std::stod(t.at(2));
        } else if (t[0] == "tran") {
            if (!haveStates) throw std::runtime_error("imdp_io: 'tran' before 'states'");
            int s = std::stoi(t.at(1));
            if (s < 0 || s >= p.nStates) throw std::runtime_error("imdp_io: tran state out of range");
            solve::ActionDist act;
            for (size_t i = 3; i < t.size(); ++i) act.push_back(parseEdge(t[i]));
            p.model[s].push_back(std::move(act));
        } else {
            throw std::runtime_error("imdp_io: unknown directive '" + t[0] + "'");
        }
    }
    if (!haveStates) throw std::runtime_error("imdp_io: missing 'states'");

    // Validate every referenced state index is in range (catches successor labels
    // mistakenly used as indices, which would otherwise be a silent out-of-bounds).
    auto chk = [&](int s, const std::string& what) {
        if (s < 0 || s >= p.nStates)
            throw std::runtime_error("imdp_io: " + what + " state " + std::to_string(s) +
                                     " out of range [0," + std::to_string(p.nStates) + ")");
    };
    chk(p.init, "init");
    for (const auto& kv : p.labels)
        for (int s : kv.second) chk(s, "label '" + kv.first + "'");
    for (int s = 0; s < p.nStates; ++s)
        for (const solve::ActionDist& act : p.model[s])
            for (const solve::Interval& iv : act) chk(iv.to, "transition target");
    return p;
}

Problem parseFile(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("imdp_io: cannot open " + path);
    std::stringstream ss; ss << f.rdbuf();
    return parse(ss.str());
}

std::string write(const Problem& p) {
    std::ostringstream o;
    o << "states " << p.nStates << "\n";
    o << "init " << p.init << "\n";
    for (const auto& kv : p.labels) {
        o << "label " << kv.first;
        for (int s : kv.second) o << " " << s;
        o << "\n";
    }
    for (int s = 0; s < (int)p.reward.size(); ++s)
        if (p.reward[s] != 0.0) o << "reward " << s << " " << p.reward[s] << "\n";
    for (int s = 0; s < p.nStates; ++s)
        for (const solve::ActionDist& act : p.model[s]) {
            o << "tran " << s << " 0";
            for (const solve::Interval& iv : act) o << " " << iv.to << ":" << iv.lo << ":" << iv.hi;
            o << "\n";
        }
    return o.str();
}

} // namespace io
} // namespace impact
