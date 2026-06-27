#ifndef IMPACT_SYSTEM_IO_H
#define IMPACT_SYSTEM_IO_H

// ============================================================================
// Text format for a 2-D affine continuous stochastic system, for the grid-cell
// satisfaction-probability HEATMAP visualizer. Parses into an abstraction::SystemND
// (built by buildSparseReachND) plus the property (reach/safety) and noise type.
//
// Format (.sys), line-based, '#' comments, keys in any order:
//   xlb <x0> <x1>          # per-dim lower grid bounds
//   xub <x0> <x1>          # per-dim upper grid bounds
//   eta <e0> <e1>          # per-dim grid step
//   ulb <u0> <u1>          # per-dim input lower (optional; default no input)
//   uub <u0> <u1>
//   ueta <ue0> <ue1>
//   A  a00 a01 a10 a11     # row-major 2x2 dynamics (AFFINE systems)
//   B  b00 b01 b10 b11     # row-major 2x(du); optional
//   c  c0 c1               # affine offset (optional, default 0)
//   f0 = <expr>            # NONLINEAR dynamics: x0' mean = expr over x0,x1,u0,u1
//   f1 = <expr>            #                     x1' mean = expr  (use either A/B/c OR f0/f1)
//   sigma <s0> <s1>        # per-dim Gaussian std-dev
//   region <lo0> <hi0> <lo1> <hi1>   # target box (reach) or avoid box (safety)
//   prop reach|safety
//   prune <p>              # successor pruning threshold (default 1e-6)
//
// Only 2-D, Gaussian-noise, diagonal/affine systems (what the heatmap renders).
// THROWS std::runtime_error on malformed input.
// ============================================================================

#include <string>
#include "abstraction.h"

namespace impact {
namespace system_io {

    struct SystemSpec {
        abstraction::SystemND sys;        // grid fields always filled; A/B/c for affine
        std::string prop = "reach";       // reach | safety
        double prune = 1e-6;
        bool nonlinear = false;           // true if f0/f1 expressions were given
        std::vector<std::string> fexpr;   // per state dimension (nonlinear dynamics text)
    };

    SystemSpec parse(const std::string& text);
    SystemSpec parseFile(const std::string& path);

} // namespace system_io
} // namespace impact

#endif // IMPACT_SYSTEM_IO_H
