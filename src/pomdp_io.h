#ifndef IMPACT_POMDP_IO_H
#define IMPACT_POMDP_IO_H

// ============================================================================
// Text format for POMDPs, for the belief-tree visualizer.
//
// Format (.pomdp), line-based, '#' comments:
//   states <N>
//   actions <A>
//   obs <O>
//   init <s>:<p> [<s>:<p> ...]      # initial belief (default: state 0 w.p. 1)
//   target <s> [<s> ...]            # target states (reachability)
//   horizon <H>                     # finite horizon (default 4)
//   T <a> <s> : <s'>:<p> ...        # transition row (missing rows default to self-loop)
//   O <a> <s'> : <o>:<p> ...        # observation row (missing rows default to obs 0)
//
// THROWS std::runtime_error on malformed input.
// ============================================================================

#include <string>
#include <set>
#include "pomdp.h"

namespace impact {
namespace pomdp_io {

    struct Parsed { pomdp::POMDP pomdp; std::set<int> target; int horizon = 4; };

    Parsed parse(const std::string& text);
    Parsed parseFile(const std::string& path);

} // namespace pomdp_io
} // namespace impact

#endif // IMPACT_POMDP_IO_H
