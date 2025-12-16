# Possible Errors in IMPaCT Codebase

This document lists potential bugs and issues found during code review.

**Last updated**: December 2024 (after bug fixes)

---

## ~~1. Bug in setDynamics() - Redundant Assignment~~ (FIXED)

**Status**: FIXED

Removed the unconditional assignment before validation. The dynamics is now only assigned inside the if-block after validation passes.

---

## ~~2. Typo in Error Message - setAvoidSpace()~~ (FIXED)

**Status**: FIXED

Changed "can't create target" to "can't create avoid region" in `setAvoidSpace()`.

---

## ~~3. Typo in Warning Message - setNoise()~~ (FIXED)

**Status**: FIXED

Changed "choise" to "choice".

---

## ~~4. Inconsistent filter Initialization~~ (FIXED)

**Status**: FIXED

Added the `is_vec()` check to `setTargetAvoidSpace()` to match the behavior of `setTargetSpace()` and `setAvoidSpace()`.

---

## ~~5. get_spaceC() Never Used~~ (RESOLVED)

**Status**: RESOLVED in Phase 3 refactoring

The unused `get_spaceC()` function was removed from the codebase.

---

## ~~6. TODO Comment Indicates Design Uncertainty~~ (RESOLVED)

**Status**: RESOLVED in Phase 3 refactoring

The TODO comment was removed along with `get_spaceC()`. The codebase now consistently uses only `get_spaceU()` (uncentered discretization).

---

## ~~7. Unused state_idx Parameters~~ (RESOLVED)

**Status**: RESOLVED in Phase 4 refactoring

The unused `ivec& state_idx` parameter was removed from `get_spaceU()`, and the corresponding member variables (`ss_idx`, `is_idx`, `ws_idx`) were removed from MDP.h.

---

## Summary of Resolved Issues

| Issue | Severity | Type | Status |
|-------|----------|------|--------|
| ~~1. setDynamics redundant assignment~~ | ~~HIGH~~ | ~~Logic bug~~ | FIXED |
| ~~2. "target" vs "avoid" message~~ | ~~LOW~~ | ~~Typo~~ | FIXED |
| ~~3. "choise" typo~~ | ~~LOW~~ | ~~Typo~~ | FIXED |
| ~~4. Inconsistent filter check~~ | ~~MEDIUM~~ | ~~Inconsistency~~ | FIXED |
| ~~5. Unused get_spaceC()~~ | ~~LOW~~ | ~~Dead code~~ | RESOLVED |
| ~~6. TODO re: centering~~ | ~~MEDIUM~~ | ~~Design uncertainty~~ | RESOLVED |
| ~~7. Unused state_idx~~ | ~~LOW~~ | ~~Dead code~~ | RESOLVED |

---

## Outstanding Issues (December 2024 Code Review)

### CRITICAL

#### 8. Missing Error Handling in GPU_synthesis.cpp

**Location**: `src/GPU_synthesis.cpp` lines 3223, 8640
**Issue**: TODO comments indicate unimplemented error handling
```cpp
//TODO: throw an error here.
temp0 += accdTT[i];
```
**Impact**: Silent failures - program continues with incorrect computations when error conditions are encountered
**Status**: OPEN

---

### WARNING

#### 9. GLPK API Usage with nullptr

**Location**: `src/IMDP.cpp` lines 5266, 5374, 5517, 5628, 5783, 5890, 6029 (40+ occurrences)
**Issue**: `glp_simplex(lp, nullptr)` passes nullptr instead of explicit control parameters
```cpp
glp_simplex(lp, nullptr);  // Uses default parameters
```
**Recommendation**: Initialize `glp_smcp` structure for explicit solver control
**Impact**: Suboptimal solver performance, non-deterministic behavior
**Status**: OPEN

#### 10. Inefficient Matrix Row Removal Pattern

**Location**: `src/MDP.cpp` lines 81-102, 105-118
**Issue**: `shed_row()` called inside loop is O(n) per removal
```cpp
for (int i = 0; i < base_space.n_rows; ++i) {
    if (condition(base_space.row(i).t())) {
        base_space.shed_row(i);  // O(n) operation inside loop
        --i;
    }
}
```
**Recommendation**: Collect indices first, then batch remove
**Impact**: O(n*m) complexity instead of O(n) for large state spaces
**Status**: OPEN

---

### INFO

#### 11. Including .cpp Files Directly

**Location**: `src/IMDP.cpp` lines 17, 19
**Issue**: Non-standard practice of including implementation files
```cpp
#include "custom.cpp"
#include "GPU_synthesis.cpp"
```
**Impact**: Potential linker issues, violates C++ conventions
**Status**: OPEN (by design for single compilation unit)

#### 12. Mixed 0-based and 1-based Indexing

**Location**: `src/IMDP.cpp` lines 5271-5273 and 70+ other locations
**Issue**: GLPK uses 1-based indexing while Armadillo uses 0-based
```cpp
for (size_t i = 1; i <= n; ++i) {
    cdfAccessor0[index] += glp_get_col_prim(lp, i) * first0(i-1);  // i-1 for Armadillo
}
```
**Impact**: Code clarity issue, potential for off-by-one errors
**Status**: OPEN (by design for GLPK compatibility)

---

## Summary Table

| Issue | Severity | Type | Status |
|-------|----------|------|--------|
| 8. GPU TODO error handling | CRITICAL | Incomplete code | OPEN |
| 9. GLPK nullptr usage | WARNING | API misuse | OPEN |
| 10. shed_row inefficiency | WARNING | Performance | OPEN |
| 11. .cpp includes | INFO | Code style | OPEN (by design) |
| 12. Mixed indexing | INFO | Code style | OPEN (by design) |

---

## Redundant Code Patterns (for future refactoring)

| Pattern | Severity | Lines Affected |
|---------|----------|----------------|
| SYCL kernel duplication (92 kernels) | HIGH | ~3000+ lines |
| LP synthesis loop duplication | MEDIUM | ~400 lines |
| Noise type conditional branches | HIGH | Throughout |
| Optimization bounds setup | MEDIUM | 90+ occurrences |

---

*Updated: December 2024 - Comprehensive code review*
