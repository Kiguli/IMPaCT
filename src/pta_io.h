#ifndef IMPACT_PTA_IO_H
#define IMPACT_PTA_IO_H

// ============================================================================
// Text format for (probabilistic) timed automata, for the zone-graph visualizer.
// Clocks are 1-based; constraints are "<clk><op><int>" with op in {<=,>=,<,>}.
//
// Format (.pta), line-based, '#' comments, keys in any order:
//   clocks <N>
//   init <loc>
//   kmax <clk> <v> [<clk> <v> ...]      # max constant per clock (for extrapolation/digital)
//   target <loc>                         # the location whose reachability is queried
//   inv <loc> <constraint> [<constraint> ...]            # location invariant
//   edge <from> | <guard constraints, space/comma-sep> | <branch> ; <branch> ; ...
//        branch = <prob> <toLoc> [r<clk> ...]            # r2 resets clock 2
//
// A non-probabilistic timed automaton is just edges with a single prob-1 branch.
// nLoc is inferred from the largest location index used. THROWS std::runtime_error.
// ============================================================================

#include <string>
#include "pta.h"

namespace impact {
namespace pta_io {

    struct Parsed { pta::PTA pta; int target = 0; };

    Parsed parse(const std::string& text);
    Parsed parseFile(const std::string& path);

} // namespace pta_io
} // namespace impact

#endif // IMPACT_PTA_IO_H
