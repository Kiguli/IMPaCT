# Phase 5 Design: Abstraction Engine Consolidation

**Date**: 2025-12-04
**Status**: 📋 **DESIGN COMPLETE** (implementation requires compilation for testing)
**Risk Level**: HIGH (core algorithm, ~5,700 lines)
**Value**: VERY HIGH (largest single reduction, ~3,500-4,000 lines projected)

---

## Executive Summary

Phase 5 consolidates the 6 abstraction functions into a unified template-based engine. These functions currently total **5,737 lines** with ~70-80% code duplication due to repeated branching patterns.

**Recommendation**: This design is ready for implementation, but requires:
1. Working compilation environment for testing
2. Incremental approach (one function at a time)
3. Comprehensive regression testing after each step
4. Baseline outputs for numerical validation

---

## Current State Analysis

### The 6 Abstraction Functions

| Function | Lines | Purpose |
|----------|-------|---------|
| `minAvoidTransitionVector()` | 1,388 | Min probability of transitioning to avoid states |
| `maxAvoidTransitionVector()` | 1,389 | Max probability of transitioning to avoid states |
| `minTransitionMatrix()` | 739 | Min transition probabilities between all states |
| `maxTransitionMatrix()` | 731 | Max transition probabilities between all states |
| `minTargetTransitionVector()` | 747 | Min probability of reaching target states |
| `maxTargetTransitionVector()` | 743 | Max probability of reaching target states |
| **Total** | **5,737** | |

### Common Structure Pattern

All 6 functions follow this identical branching structure:

```cpp
void IMDP::functionName() {
    auto start = chrono::steady_clock::now();
    cout << "Calculating ..." << endl;

    // LEVEL 1: Parameter count branching
    if (disturb_space_size == 0 && input_space_size == 0) {
        // 1-parameter dynamics (state only)
        // LEVEL 2: Noise type branching
        if (noise == NORMAL && diagonal) {
            // Diagonal normal distribution
            // ~200-300 lines of SYCL kernel + optimization
        }
        else if (noise == NORMAL && !diagonal) {
            // Off-diagonal normal distribution
            // ~200-300 lines of SYCL kernel + optimization
        }
        else if (noise == CUSTOM) {
            // Custom distribution
            // ~200-300 lines of SYCL kernel + optimization
        }
    }
    else if (disturb_space_size == 0) {
        // 2-parameter dynamics (state + input)
        // Same LEVEL 2 branching (3 variants × ~250 lines = ~750 lines)
    }
    else if (input_space_size == 0) {
        // 2-parameter dynamics (state + disturbance)
        // Same LEVEL 2 branching (3 variants × ~250 lines = ~750 lines)
    }
    else {
        // 3-parameter dynamics (state + input + disturbance)
        // Same LEVEL 2 branching (3 variants × ~250 lines = ~750 lines)
    }

    auto end = chrono::steady_clock::now();
    cout << "Time: " << chrono::duration_cast<chrono::seconds>(end-start).count() << "s" << endl;
}
```

**Duplication**: Each of 6 functions has 4 parameter branches × 3 noise branches = **12 code blocks**
**Total blocks**: 6 functions × 12 blocks = **72 highly similar code blocks**

---

## Duplication Analysis

### Branching Breakdown

**Level 1 - Parameter Count** (4 variants):
1. No input, no disturbance (1 param)
2. Input, no disturbance (2 params)
3. No input, disturbance (2 params)
4. Input and disturbance (3 params)

**Level 2 - Noise Type** (3 variants per level 1):
1. Diagonal normal
2. Off-diagonal normal (multivariate)
3. Custom PDF

**Result**: 4 × 3 = **12 code blocks per function** × 6 functions = **72 total blocks**

### What Varies vs What's Identical

**Varies between functions:**
- Objective direction (min vs max)
- Result type (vector vs matrix)
- Target space (avoid vs target vs full state space)
- Result computation (1.0 - minf vs minf vs transition probability)

**Identical across all functions:**
- Branching structure (parameter count → noise type)
- SYCL kernel setup (queue, buffer, submit, wait)
- Optimization setup (NLopt configuration)
- Index calculations (state/input/disturbance mapping)
- Cost function data preparation pattern
- Error handling

**Duplication percentage**: **~70-80%** of code is duplicated

---

## Proposed Template-Based Design

### Core Abstraction Engine Template

```cpp
namespace IMPaCT_AbstractionEngine {

// Policy classes for objective direction
struct MinObjective {
    template<typename CostData>
    static double optimize(const vec& state, const vec& eta, nlopt::algorithm algo,
                          auto cost_fn, CostData* data) {
        return IMPaCT_Optimization::optimizeMinObjective(state, eta, algo, cost_fn, data);
    }
};

struct MaxObjective {
    template<typename CostData>
    static double optimize(const vec& state, const vec& eta, nlopt::algorithm algo,
                          auto cost_fn, CostData* data) {
        return IMPaCT_Optimization::optimizeMaxObjective(state, eta, algo, cost_fn, data);
    }
};

struct ComplementaryObjective {
    template<typename CostData>
    static double optimize(const vec& state, const vec& eta, nlopt::algorithm algo,
                          auto cost_fn, CostData* data) {
        return IMPaCT_Optimization::optimizeComplementary(state, eta, algo, cost_fn, data);
    }
};

// Policy class for result storage
template<typename ResultType>
struct ResultHandler {
    static void allocate(ResultType& result, size_t rows, size_t cols = 0);
    static void store(ResultType& result, size_t index, double value);
    static void finalize(ResultType& result, const mat& temp, bool has_avoid);
};

// Specialization for vector results
template<>
struct ResultHandler<vec> {
    static void allocate(vec& result, size_t rows, size_t) {
        result.set_size(rows);
    }
    static void store(vec& result, size_t index, double value) {
        result[index] = value;
    }
    static void finalize(vec& result, const mat& temp, bool has_avoid) {
        if (has_avoid) {
            result = result + sum(temp, 1);
        }
    }
};

// Specialization for matrix results
template<>
struct ResultHandler<mat> {
    static void allocate(mat& result, size_t rows, size_t cols) {
        result.set_size(rows, cols);
    }
    static void store(mat& result, size_t index, double value) {
        // Matrix indexing logic
        size_t row = index / result.n_cols;
        size_t col = index % result.n_cols;
        result(row, col) = value;
    }
    static void finalize(mat& result, const mat&, bool) {
        // Matrix doesn't need finalization
    }
};

// Main abstraction engine template
template<
    typename ObjectivePolicy,   // MinObjective, MaxObjective, or ComplementaryObjective
    typename ResultType,        // vec or mat
    typename TargetSpace        // AvoidSpace, TargetSpace, or FullStateSpace
>
class AbstractionEngine {
private:
    IMDP* imdp;  // Reference to IMDP object for member access

public:
    AbstractionEngine(IMDP* imdp_ptr) : imdp(imdp_ptr) {}

    void compute(ResultType& result, const string& description) {
        auto start = chrono::steady_clock::now();
        cout << description << endl;

        // Determine parameter count
        int param_count = calculateParamCount();

        // Branch based on parameter count and noise type
        switch (param_count) {
            case 1: computeWithParams<1>(result); break;
            case 2: computeWithParams<2>(result); break;
            case 3: computeWithParams<3>(result); break;
        }

        auto end = chrono::steady_clock::now();
        cout << "Time: " << chrono::duration_cast<chrono::seconds>(end-start).count() << "s" << endl;
    }

private:
    int calculateParamCount() const {
        if (imdp->disturb_space_size == 0 && imdp->input_space_size == 0) return 1;
        if (imdp->disturb_space_size == 0 || imdp->input_space_size == 0) return 2;
        return 3;
    }

    template<int ParamCount>
    void computeWithParams(ResultType& result) {
        if (imdp->noise == NoiseType::NORMAL && imdp->diagonal) {
            computeWithNoise<ParamCount, DiagonalNormal>(result);
        }
        else if (imdp->noise == NoiseType::NORMAL && !imdp->diagonal) {
            computeWithNoise<ParamCount, OffDiagonalNormal>(result);
        }
        else if (imdp->noise == NoiseType::CUSTOM) {
            computeWithNoise<ParamCount, CustomNoise>(result);
        }
    }

    template<int ParamCount, typename NoiseModel>
    void computeWithNoise(ResultType& result) {
        // This is where the actual SYCL kernel logic goes
        // Uses:
        // - ObjectivePolicy for optimization direction
        // - ResultType for storage
        // - TargetSpace for determining which states to compute
        // - ParamCount for selecting appropriate cost function
        // - NoiseModel for selecting appropriate noise handling

        // Pseudo-code:
        // 1. Allocate result storage
        // 2. Launch SYCL kernel with optimization
        // 3. Handle avoid/target space summation if needed
        // 4. Store results
    }
};

} // namespace IMPaCT_AbstractionEngine
```

### Wrapper Functions (Public API)

```cpp
// In IMDP.cpp, replace the 6 large functions with simple wrappers:

void IMDP::minAvoidTransitionVector() {
    IMPaCT_AbstractionEngine::AbstractionEngine<
        IMPaCT_AbstractionEngine::ComplementaryObjective,
        vec,
        IMPaCT_AbstractionEngine::AvoidSpace
    > engine(this);

    engine.compute(minAvoidM, "Calculating minimal avoid transition probability Vector.");
}

void IMDP::maxAvoidTransitionVector() {
    IMPaCT_AbstractionEngine::AbstractionEngine<
        IMPaCT_AbstractionEngine::MinObjective,  // Note: max avoid = min transition
        vec,
        IMPaCT_AbstractionEngine::AvoidSpace
    > engine(this);

    engine.compute(maxAvoidM, "Calculating maximal avoid transition probability Vector.");
}

void IMDP::minTransitionMatrix() {
    IMPaCT_AbstractionEngine::AbstractionEngine<
        IMPaCT_AbstractionEngine::MinObjective,
        mat,
        IMPaCT_AbstractionEngine::FullStateSpace
    > engine(this);

    engine.compute(minTransitionM, "Calculating minimal transition Matrix.");
}

// ... similar for other 3 functions
```

---

## Expected Line Reduction

### Conservative Estimate

**Current state**: 5,737 lines across 6 functions

**After refactoring**:
- Template engine: ~800-1,000 lines (complex, but reusable)
- Wrapper functions: ~60 lines (6 × 10 lines each)
- Policy classes: ~200 lines
- **Total**: ~1,060-1,260 lines

**Reduction**: 5,737 - 1,260 = **4,477 lines (78% reduction)**

### Realistic Estimate

Accounting for:
- Template specializations for edge cases
- Additional helper functions
- Comprehensive documentation
- Error handling improvements

**Total after refactoring**: ~1,500-1,800 lines
**Reduction**: 5,737 - 1,800 = **3,937 lines (69% reduction)**

**Target**: ~3,500-4,000 lines reduction

---

## Implementation Strategy

### Phase 5a: Template Engine Foundation
1. Create abstraction_engine.h with policy classes
2. Implement AbstractionEngine template skeleton
3. Add helper functions for common operations
4. **Test**: Compile, no runtime testing yet

### Phase 5b: Refactor One Function (Pilot)
1. Choose simplest function: `minTransitionMatrix()` (739 lines)
2. Implement template specialization for this function
3. Create wrapper in IMDP.cpp
4. **Test**: Compile and run ex_2Drobot-R-U
5. **Validate**: Compare transition matrix with baseline
6. **If successful**: Proceed to 5c
7. **If issues**: Debug, adjust design, retry

### Phase 5c: Refactor Remaining Functions (Incremental)
1. Refactor `maxTransitionMatrix()` (very similar to min)
2. Test and validate
3. Refactor `minTargetTransitionVector()`
4. Test and validate
5. Refactor `maxTargetTransitionVector()`
6. Test and validate
7. Refactor `minAvoidTransitionVector()`
8. Test and validate
9. Refactor `maxAvoidTransitionVector()`
10. Test and validate

### Phase 5d: Optimization and Polish
1. Profile performance
2. Optimize hot paths
3. Add comprehensive error handling
4. Update documentation
5. Final regression testing on all 15 examples

---

## Risk Mitigation

### High-Risk Factors

| Risk | Mitigation |
|------|------------|
| **Core algorithm changes** | Incremental approach, test after each function |
| **Template complexity** | Comprehensive documentation, clear naming |
| **SYCL kernel changes** | Keep kernel bodies similar, only extract structure |
| **Performance regression** | Profile before/after, templates should inline |
| **Numerical differences** | Bit-for-bit comparison with baseline outputs |

### Testing Strategy

**Unit Tests** (if possible):
```cpp
void test_abstraction_engine_min_transition() {
    // Setup IMDP with test data
    IMDP imdp;
    // ... configure state space, dynamics, etc. ...

    // Compute with original function (from baseline branch)
    imdp.minTransitionMatrix();
    mat baseline = imdp.minTransitionM;

    // Compute with refactored function
    imdp.minTransitionMatrix();
    mat refactored = imdp.minTransitionM;

    // Compare element-wise
    assert(approx_equal(baseline, refactored, "absdiff", 1e-10));
}
```

**Integration Tests**:
1. Run all 15 examples
2. Compare HDF5 outputs with h5diff
3. Verify convergence messages
4. Check execution time (±10% acceptable)
5. Validate probability constraints (0 ≤ P ≤ 1)

---

## Benefits

### 1. Massive Line Reduction ⭐⭐⭐⭐⭐
- ~3,500-4,000 lines eliminated
- Largest single reduction in entire refactoring plan
- 78% reduction in abstraction function code

### 2. Eliminates Massive Duplication ⭐⭐⭐⭐⭐
- 72 highly similar code blocks → 1 template engine
- Changes to algorithm logic only needed once
- Impossible to have inconsistencies between functions

### 3. Improved Maintainability ⭐⭐⭐⭐⭐
- Single source of truth for abstraction logic
- Easy to add new noise models or space variants
- Clear separation of concerns (policy-based design)

### 4. Better Testability ⭐⭐⭐⭐
- Template components can be unit tested independently
- Policy classes are simple and testable
- Easier to mock for testing

### 5. Extensibility ⭐⭐⭐⭐⭐
- New objective policies: trivial to add
- New noise models: add policy class
- New target spaces: add policy class
- New parameter counts: add template specialization

---

## Comparison with Other Phases

| Phase | Lines Reduced | Risk | Complexity | Status |
|-------|---------------|------|------------|--------|
| Phase 1 (I/O) | 43 | LOW | Simple | ✅ Complete |
| Phase 2 (Cost) | 144 | MEDIUM | Moderate | ✅ Complete |
| Phase 3 (Opt) | 800-1,000 | LOW | Simple | ✅ Framework |
| Phase 4 (SYCL) | 1,000-1,200 | HIGH | Complex | ⏸️ Deferred |
| **Phase 5 (Abstraction)** | **3,500-4,000** | **HIGH** | **High** | **📋 Designed** |
| Phase 6 (Synthesis) | 3,000-3,500 | MEDIUM | Moderate | ⏳ Pending |

Phase 5 has the **highest value** but also **highest risk** of all phases.

---

## Dependencies

### Prerequisites for Implementation

1. ✅ **Phase 1 complete**: I/O utilities available
2. ✅ **Phase 2 complete**: Cost function templates available
3. ✅ **Phase 3 complete**: Optimization helpers available
4. ❌ **Compilation environment**: Need working build system
5. ❌ **Baseline outputs**: Need reference outputs for all examples
6. ❌ **Testing infrastructure**: Need automated test suite

### Builds On Previous Phases

Phase 5 leverages all previous work:
- **Phase 1**: Uses `IMPaCT_IO` for saving/loading
- **Phase 2**: Uses `IMPaCT_CostFunctions` templates
- **Phase 3**: Uses `IMPaCT_Optimization` helpers (reduces kernel complexity)
- **Phase 4**: (skipped, but SYCL patterns inform design)

---

## Alternative Approaches Considered

### Option 1: Macro-Based Abstraction
**Rejected**: Not type-safe, hard to debug, against modern C++

### Option 2: Code Generation Script
Generate 6 functions from single template at build time
**Rejected**: Adds build complexity, harder to debug

### Option 3: Runtime Polymorphism (Virtual Functions)
Use inheritance and virtual functions
**Rejected**: Runtime overhead, incompatible with SYCL device code

### Option 4: Partial Refactoring (Just Branching Logic)
Extract only the branching structure, keep kernel bodies separate
**Considered**: Lower risk but less benefit (~1,500 lines vs ~3,500)
**Status**: Fallback option if full refactoring too risky

### Option 5: Template Metaprogramming (Chosen) ✅
Compile-time polymorphism with policy classes
**Benefits**: Zero runtime overhead, type-safe, extensible
**Drawbacks**: Complex implementation, steep learning curve

---

## Success Criteria

### Must Have
- ✅ Compiles without errors
- ✅ All 15 examples run to completion
- ✅ Numerical results match baseline (tolerance: 1e-10)
- ✅ Execution time within ±10% of baseline
- ✅ No memory leaks or crashes

### Should Have
- ✅ Code is more readable than original
- ✅ Template error messages are understandable
- ✅ Documentation explains design decisions
- ✅ Incremental git commits allow rollback

### Nice to Have
- ⭐ Performance improvements (better optimization)
- ⭐ Unit tests for template components
- ⭐ Automated regression test suite
- ⭐ Profiling data showing hot paths

---

## Timeline Estimate

**Assuming compilation is available and incremental testing is possible:**

| Sub-Phase | Estimated Time | Cumulative |
|-----------|----------------|------------|
| 5a: Template foundation | 1-2 days | 1-2 days |
| 5b: Pilot function (min transition) | 2-3 days | 3-5 days |
| 5c: Remaining 5 functions | 5-7 days | 8-12 days |
| 5d: Optimization and polish | 2-3 days | 10-15 days |
| Testing and validation | 3-5 days | 13-20 days |
| **Total** | **2-4 weeks** | |

**Without compilation**: Design only (current state)

---

## Conclusion

**Phase 5 Design Status**: ✅ **COMPLETE AND READY FOR IMPLEMENTATION**

### Key Takeaways

1. **Highest Value Phase**: ~3,500-4,000 line reduction
2. **Highest Risk Phase**: Core algorithm, requires extensive testing
3. **Depends on Compilation**: Cannot implement without testing
4. **Builds on All Previous Phases**: Leverages Phases 1-3 work
5. **Clear Implementation Path**: Incremental, testable approach

### Recommendation

**When Compilation Available**:
1. Complete Phase 3b (apply optimization helpers) first
   - Lower risk, good warm-up
   - Reduces Phase 5 complexity (kernels will be simpler)
2. Implement Phase 5 incrementally
   - One function at a time
   - Test thoroughly at each step
3. Only proceed if pilot function (5b) succeeds

**Current Action**: Commit this design document, wait for compilation

---

**Git Commit Suggestion:**
```bash
git add PHASE5_DESIGN.md
git commit -m "Phase 5 Design: Abstraction engine consolidation

Design complete for consolidating 6 abstraction functions (5,737 lines)
into unified template-based engine using policy-based design.

Projected reduction: 3,500-4,000 lines (69-78% of abstraction code)

Key design elements:
- Policy classes for objective direction (Min/Max/Complementary)
- Template specialization for parameter counts (1/2/3)
- Result handlers for vector vs matrix outputs
- Leverages Phases 1-3 (IO, cost functions, optimization helpers)

Implementation strategy:
- Incremental approach (one function at a time)
- Pilot with minTransitionMatrix() (simplest, 739 lines)
- Comprehensive testing after each function
- Requires compilation for validation

Status: Design ready, implementation deferred until testing available

Part of multi-phase refactoring plan to reduce ~16,000 lines of redundancy."
```

---

*Design compiled: 2025-12-04*
*Phase: 5 (Abstraction Engine Consolidation - Design Only)*
*Implementation: Deferred until compilation and testing available*
