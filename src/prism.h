#ifndef IMPACT_PRISM_H
#define IMPACT_PRISM_H

// ============================================================================
// PRISM modelling-language front-end (SUBSET) -> io::Problem -> one solver path.
//
// Lets users write models in the familiar PRISM style instead of the explicit
// .imdp format. Supported subset (single bounded integer state variable):
//
//   mdp                                  // model type (mdp / imdp); required-ish
//   const int N = <int>;                 // integer constants (usable in [..]/guards/updates)
//   module <name>
//     x : [<lo>..<hi>] init <i>;         // ONE state variable
//     [<act>] x=<K> -> <p>:(x'=<E>) + <p>:(x'=<E>) ... ;
//     [<act>] x=<K> -> (x'=<E>);         // single update => probability 1
//   endmodule
//   label "<name>" = x=<K> [| x=<K> ...] ;
//
//   <p>  : a decimal literal (point)  OR  [<lo>,<hi>] (interval -> IMDP edge)
//   <E>  : <int> | x | x+<int> | x-<int>   (evaluated at the guard value K)
//   // line comments allowed; whitespace/newlines free.
//
// State index = variable value - lo. Interval probabilities yield an IMDP;
// point probabilities yield an ordinary MDP. Same Problem/IMDPModel as
// src/imdp_io.h, so PRISM models feed the identical solver, CLI and benchmarks.
//
// NOT yet supported (documented future work): multiple variables / modules,
// synchronization, formulas, rate/real expressions, non-equality guards,
// rewards. THROWS std::runtime_error on unsupported or malformed input.
// ============================================================================

#include <string>
#include "imdp_io.h"   // io::Problem

namespace impact {
namespace prism {

    io::Problem parse(const std::string& text);
    io::Problem parseFile(const std::string& path);

} // namespace prism
} // namespace impact

#endif // IMPACT_PRISM_H
