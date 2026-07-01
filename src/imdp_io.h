#ifndef IMPACT_IMDP_IO_H
#define IMPACT_IMDP_IO_H

// ============================================================================
// Explicit (interval) MDP exchange format + parser/writer, for cross-tool
// benchmarking (compare IMPaCT against IntervalMDP.jl / PRISM / Storm on shared
// models). Tool-agnostic text format; the PRISM front-end (src/prism.h) targets
// the same Problem struct so any front-end feeds one solver path.
//
// Format (.imdp), whitespace-separated, '#' comments, blank lines ignored:
//   states <N>
//   init <s>                      # optional, default 0
//   label <NAME> <s> <s> ...      # repeatable; accumulates
//   reward <s> <value>            # optional; state reward (for expected-reward props)
//   tran <s> <a> <to>:<lo>:<hi> <to>:<lo>:<hi> ...   # one action's interval dist
// Point MDPs use lo==hi. One `tran` line == one action of state <s>.
// ============================================================================

#include <string>
#include <map>
#include <set>
#include <vector>
#include "solve.h"

namespace impact {
namespace io {

    struct Problem {
        solve::IMDPModel model;
        std::map<std::string, std::set<int>> labels;   // label name -> state set
        std::vector<double> reward;                     // per-state reward (empty => all 0)
        int init = 0;
        int nStates = 0;
    };

    // Parse the explicit .imdp format. THROWS std::runtime_error on malformed input.
    Problem parse(const std::string& text);
    Problem parseFile(const std::string& path);

    // Serialize back to the .imdp format (round-trips parse()).
    std::string write(const Problem& p);

} // namespace io
} // namespace impact

#endif // IMPACT_IMDP_IO_H
