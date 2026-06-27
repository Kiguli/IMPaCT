#ifndef IMPACT_EXPR_H
#define IMPACT_EXPR_H

// ============================================================================
// Arithmetic-expression parser + SOUND interval evaluator, for specifying
// NONLINEAR system dynamics as text (so the web app / CLI can abstract nonlinear
// stochastic systems, e.g. ARCH-COMP benchmarks, not just affine ones).
//
// Evaluating an expression with interval arithmetic over a grid cell's state box
// (and fixed input) yields a SOUND enclosure [muLo, muHi] of the dynamics' mean over
// that cell — exactly the MeanBoundFn that abstraction::buildSparseReachGeneral needs
// (the resulting IMDP is sound because the enclosure contains the true mean).
//
// Grammar: + - * / , unary -, ^ (integer constant exponent), parentheses, numbers,
// variables (caller-supplied names), and functions sin cos exp sqrt abs.
// Refs: Moore, "Interval Analysis" (1966); interval-MDP abstraction of nonlinear
// stochastic systems (Lavaei et al.; IMPaCT, arXiv:2401.03555). Contracts:
// tests/unit/test_expr.cpp (interval enclosure contains brute-force point samples).
// ============================================================================

#include <string>
#include <vector>
#include <memory>
#include "abstraction.h"   // abstraction::Ival

namespace impact {
namespace expr {

    struct Node;
    using Expr = std::shared_ptr<Node>;

    // Parse `s` over the given ordered variable names. THROWS std::runtime_error on a
    // parse error / unknown identifier / non-integer exponent.
    Expr parse(const std::string& s, const std::vector<std::string>& vars);

    // Sound interval evaluation: vals[i] is the interval for variable i (in `vars` order).
    abstraction::Ival evalInterval(const Expr& e, const std::vector<abstraction::Ival>& vals);

    // Exact point evaluation (for testing / sampling oracles).
    double evalPoint(const Expr& e, const std::vector<double>& vals);

} // namespace expr
} // namespace impact

#endif // IMPACT_EXPR_H
