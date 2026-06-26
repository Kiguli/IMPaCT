// ============================================================================
// CONTRACT TESTS — 2-D continuous-system spec parser (grid-heatmap visualizer).
// ============================================================================
#include "../doctest.h"
#include "../contracts/contracts.h"

using impact::system_io::parse;

static const char* SPEC =
    "# 2D affine system\n"
    "xlb -3 -3\n"
    "xub 3 3\n"
    "eta 0.5 0.5\n"
    "ulb -1 -1\n"
    "uub 1 1\n"
    "ueta 1 1\n"
    "A 0.9 0 0 0.8\n"
    "B 0.5 0 0 0.5\n"
    "sigma 0.3 0.4\n"
    "region 1.5 3 2 3\n"
    "prop reach\n"
    "prune 1e-5\n";

TEST_CASE("system_io: parse a 2-D affine spec") {
    auto s = parse(SPEC);
    CHECK(s.prop == "reach");
    CHECK(s.prune == doctest::Approx(1e-5));
    CHECK(s.sys.dim_x == 2);
    CHECK(s.sys.dim_u == 2);
    CHECK(s.sys.A[0][0] == doctest::Approx(0.9));
    CHECK(s.sys.A[1][1] == doctest::Approx(0.8));
    CHECK(s.sys.sigma[1] == doctest::Approx(0.4));
    CHECK(s.sys.tlo[0] == doctest::Approx(1.5));
    CHECK(s.sys.thi[0] == doctest::Approx(3.0));
    CHECK(s.sys.tlo[1] == doctest::Approx(2.0));
}

TEST_CASE("system_io: missing required keys throw") {
    CHECK_THROWS(parse("xlb -3 -3\nxub 3 3\neta 0.5 0.5\n"));   // no sigma / A
}
