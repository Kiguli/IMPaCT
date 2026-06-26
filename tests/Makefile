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
IMPL_SRCS := ../src/omaximization.cpp \
             ../src/graph_utils.cpp \
             contracts/stubs.cpp

TEST_SRCS := unit/test_main.cpp \
             unit/test_omaximization.cpp \
             unit/test_graph.cpp \
             unit/test_interval_iteration.cpp \
             unit/test_ltl_dfa.cpp

BIN := run_tests

.PHONY: test clean
test: $(BIN)
	./$(BIN)

$(BIN): $(TEST_SRCS) $(IMPL_SRCS) doctest.h contracts/contracts.h
	$(CXX) $(CXXFLAGS) $(TEST_SRCS) $(IMPL_SRCS) -o $(BIN)

clean:
	rm -f $(BIN)
