// ============================================================================
// CONTRACT TESTS — expression parser + interval evaluator (nonlinear dynamics).
// Soundness oracle: the interval enclosure must contain EVERY sampled point value
// (this is exactly what makes the nonlinear abstraction sound).
// ============================================================================
#include "../doctest.h"
#include "../contracts/contracts.h"

#include <vector>
#include <string>
#include <cstdint>
#include <cmath>

using impact::abstraction::Ival;
namespace expr = impact::expr;

namespace { struct Lcg{ uint64_t s; double u(){ s=s*6364136223846793005ULL+1442695040888963407ULL; return ((s>>11)&((1ULL<<53)-1))/9007199254740992.0; } double in(double a,double b){return a+(b-a)*u();} }; }

TEST_CASE("expr: point evaluation matches the formula") {
    std::vector<std::string> vars={"x0","x1"};
    CHECK(expr::evalPoint(expr::parse("x0^2 + 2*x0 + 1", vars), {3,0}) == doctest::Approx(16.0));
    CHECK(expr::evalPoint(expr::parse("x1 + 0.1*(-x0 + (1 - x0^2)*x1)", vars), {0.5,1.0}) == doctest::Approx(1.0+0.1*(-0.5+(1-0.25)*1.0)));
    CHECK(expr::evalPoint(expr::parse("sin(x0) + exp(0)", vars), {0,0}) == doctest::Approx(1.0));
}

TEST_CASE("expr: interval enclosure contains all point samples (soundness)") {
    std::vector<std::string> vars={"x0","x1","u0"};
    std::vector<std::string> forms = {
        "0.9*x0 + 0.5*u0",
        "x0 + 0.1*x1",
        "x1 + 0.1*(-x0 + (1 - x0^2)*x1)",     // Van der Pol-style nonlinearity
        "sin(x0) + cos(x1)",
        "exp(0.3*x0) * x1 + u0",
        "abs(x0) + sqrt(x1 + 5)",
        "1.0/(2.0 + x0) + x1^3",
    };
    Lcg rng{0x9E3779B9ULL};
    for (const auto& f : forms) {
        auto E = expr::parse(f, vars);
        for (int t = 0; t < 400; ++t) {
            // random sub-box within safe ranges (sqrt arg = x1+5 > 0; div denom = 2+x0 != 0 for x0 in [-1,3])
            double x0lo=rng.in(-1.0,2.5), x1lo=rng.in(-3.0,3.0), u0lo=rng.in(-1.0,1.0);
            std::vector<Ival> box = { Ival(x0lo, x0lo+rng.in(0,1.0)), Ival(x1lo, x1lo+rng.in(0,1.5)), Ival(u0lo, u0lo+rng.in(0,0.5)) };
            Ival enc = expr::evalInterval(E, box);
            for (int s = 0; s <= 6; ++s) for (int r = 0; r <= 6; ++r) for (int w = 0; w <= 2; ++w) {
                std::vector<double> pt = { box[0].lo + (box[0].hi-box[0].lo)*s/6.0,
                                           box[1].lo + (box[1].hi-box[1].lo)*r/6.0,
                                           box[2].lo + (box[2].hi-box[2].lo)*w/2.0 };
                double val = expr::evalPoint(E, pt);
                CHECK(val >= enc.lo - 1e-9);
                CHECK(val <= enc.hi + 1e-9);
            }
        }
    }
}

TEST_CASE("expr: parse errors throw") {
    std::vector<std::string> vars={"x0"};
    CHECK_THROWS(expr::parse("x0 +", vars));          // trailing op
    CHECK_THROWS(expr::parse("nope(x0)", vars));      // unknown function
    CHECK_THROWS(expr::parse("y0", vars));            // unknown variable
    CHECK_THROWS(expr::parse("x0 ^ 1.5", vars));      // non-integer exponent
    CHECK_NOTHROW(expr::parse("x0^3 - 2*x0", vars));
}
