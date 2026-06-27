# ============================================================================
# IMPaCT v2.0 unit-test build (doctest).
#
# Internal-algorithm unit tests are PURE std C++ (no SYCL/Armadillo/HDF5/GLPK),
# so they build with any C++17 compiler and run anywhere. They link against the
# v2.0 component implementations.
#
# TDD workflow:
#   * Now (red): IMPL_SRCS = contracts/stubs.cpp  -> tests fail "not implemented".
#   * As each Phase-1 component lands, swap its stub for the real source, e.g.
#       IMPL_SRCS = ../src/omaximization.cpp ../src/GraphUtils.cpp ../src/solve.cpp
#     (and drop the corresponding stub). NEVER edit the tests to pass.
# ============================================================================

CXX ?= c++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra

# Implementation under test. Real src/*.cpp replace stubs as components land.
# Real implementations under test. (contracts/stubs.cpp holds no symbols now —
# all contracts are implemented; re-add it here if a future contract needs a stub.)
IMPL_SRCS := ../src/omaximization.cpp \
             ../src/graph_utils.cpp \
             ../src/solve.cpp \
             ../src/ltl.cpp \
             ../src/product.cpp \
             ../src/abstraction.cpp \
             ../src/system_io.cpp \
             ../src/expr.cpp \
             ../src/omega.cpp \
             ../src/imdp_io.cpp \
             ../src/prism.cpp \
             ../src/pctl.cpp \
             ../src/stl.cpp \
             ../src/dbm.cpp \
             ../src/ta.cpp \
             ../src/pta.cpp \
             ../src/pomdp.cpp \
             ../src/pta_io.cpp \
             ../src/pomdp_io.cpp \
             ../src/ltlspec.cpp

TEST_SRCS := unit/test_main.cpp \
             unit/test_omaximization.cpp \
             unit/test_graph.cpp \
             unit/test_interval_iteration.cpp \
             unit/test_ltl_dfa.cpp \
             unit/test_product.cpp \
             unit/test_abstraction.cpp \
             unit/test_system_io.cpp \
             unit/test_expr.cpp \
             unit/test_omega.cpp \
             unit/test_safety.cpp \
             unit/test_imdp_io.cpp \
             unit/test_prism.cpp \
             unit/test_pctl.cpp \
             unit/test_stl.cpp \
             unit/test_dbm.cpp \
             unit/test_ta.cpp \
             unit/test_pta.cpp \
             unit/test_pomdp.cpp \
             unit/test_model_io.cpp \
             unit/test_ltlspec.cpp

BIN := run_tests

.PHONY: test clean
test: $(BIN)
	./$(BIN)

$(BIN): $(TEST_SRCS) $(IMPL_SRCS) doctest.h contracts/contracts.h
	$(CXX) $(CXXFLAGS) $(TEST_SRCS) $(IMPL_SRCS) -o $(BIN)

clean:
	rm -f $(BIN)
