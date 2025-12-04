# IMPaCT Testing Infrastructure

Automated testing infrastructure for validating IMPaCT examples across branches.

## Overview

This testing infrastructure provides:
- **Automated compilation and execution** of all examples
- **Output validation** (HDF5 file comparison)
- **Performance benchmarking** (compile and runtime)
- **Branch comparison** for merge validation

## Quick Start

### Running Tests Locally

```bash
# From IMPaCT root directory

# Run smoke tests (3 examples, ~5 minutes)
python test/test_runner.py --suite smoke

# Run all tests
python test/test_runner.py --all

# Run specific example
python test/test_runner.py --example ex_2Drobot-R-U
```

### Comparing Branch Outputs

```bash
# Step 1: Run tests on main branch
git checkout main
python test/test_runner.py --suite critical --output main_results.json

# Step 2: Run tests on refactor branch
git checkout refactor/phase1-io-utils
python test/test_runner.py --suite critical --output refactor_results.json

# Step 3: Compare outputs
python test/compare_outputs.py main_outputs/ refactor_outputs/ --report comparison.md

# Step 4: Compare performance
python test/benchmark.py main_results.json refactor_results.json --report benchmark.md
```

## Tools

### 1. test_runner.py

Automated test runner that compiles and executes examples.

**Features:**
- Discovers examples from configuration
- Compiles using Makefiles
- Executes with timeout protection
- Validates output files exist
- Generates JSON results

**Usage:**
```bash
python test/test_runner.py --suite smoke          # Run smoke test suite
python test/test_runner.py --suite comprehensive  # Run all examples
python test/test_runner.py --suite critical       # Run critical tests
python test/test_runner.py --example ex_2Drobot-R-U  # Run single example
python test/test_runner.py --all                  # Run everything
```

**Options:**
- `--config PATH`: Custom config file (default: test/test_config.yaml)
- `--output FILE`: Output results file (default: test_results.json)
- `--root DIR`: IMPaCT root directory (default: auto-detect)

**Exit Codes:**
- `0`: All tests passed
- `1`: One or more tests failed

### 2. compare_outputs.py

Compares HDF5 outputs between two test runs.

**Features:**
- Byte-for-byte comparison
- Element-wise numerical comparison with tolerance
- Generates detailed Markdown report
- Flags differences and missing files

**Usage:**
```bash
python test/compare_outputs.py main_outputs/ refactor_outputs/
python test/compare_outputs.py main_outputs/ refactor_outputs/ --report comparison.md
python test/compare_outputs.py main_outputs/ refactor_outputs/ --tolerance 1e-10
```

**Options:**
- `--tolerance FLOAT`: Numerical tolerance (default: 1e-12)
- `--report FILE`: Output report file (default: comparison-report.md)
- `--main-results FILE`: Main branch test_results.json
- `--refactor-results FILE`: Refactor branch test_results.json

**Exit Codes:**
- `0`: All files match (identical or within tolerance)
- `1`: Differences or errors detected

### 3. benchmark.py

Compares performance between branches.

**Features:**
- Compilation time comparison
- Runtime comparison
- Flags performance regressions
- Generates Markdown report

**Usage:**
```bash
python test/benchmark.py main_results.json refactor_results.json
python test/benchmark.py main_results.json refactor_results.json --threshold 0.15  # 15% threshold
```

**Options:**
- `--threshold FLOAT`: Regression threshold (default: 0.10 = 10%)
- `--report FILE`: Output report file (default: benchmark-report.md)

**Exit Codes:**
- `0`: No performance regressions
- `1`: Regressions detected

## Configuration

### test_config.yaml

Configuration file defining:
- Example metadata (paths, executables, expected outputs)
- Timeout settings (compile and run)
- Test suites (smoke, comprehensive, critical)
- Validation rules

**Example Structure:**
```yaml
examples:
  small:
    - name: ex_2Drobot-R-U
      path: examples/ex_2Drobot-R-U
      executable: robot2D
      compile_timeout: 300
      run_timeout: 120
      expected_outputs:
        - ss.h5
        - is.h5
        - controller.h5

test_suites:
  smoke:
    description: "Quick smoke tests"
    examples:
      - ex_2Drobot-R-U
      - ex_2Drobot-R-D
```

## GitHub Actions Workflows

### ci-fast.yml

**Trigger**: On every push and pull request
**Duration**: ~10 minutes
**Purpose**: Quick validation

Runs smoke test suite (3 small examples) to catch obvious issues.

### ci-comprehensive.yml

**Trigger**: Nightly or manual
**Duration**: ~60 minutes
**Purpose**: Full integration testing

Runs all 15 examples to ensure complete functionality.

### branch-comparison.yml (CRITICAL)

**Trigger**: Manual workflow dispatch
**Duration**: ~60 minutes
**Purpose**: Validate branch merge

This is the key workflow for deciding whether to merge a refactor branch:

1. Runs tests on both main and refactor branches in parallel
2. Compares all outputs (numerical comparison)
3. Compares performance (compile and runtime)
4. Generates comprehensive report
5. Provides clear SAFE TO MERGE / REVIEW REQUIRED verdict

**How to Run:**
```
1. Go to GitHub Actions tab
2. Select "Branch Comparison - Validate Merge"
3. Click "Run workflow"
4. Enter:
   - Base branch: main
   - Compare branch: refactor/phase1-io-utils
5. Wait for completion (~60 minutes)
6. Download artifacts:
   - comparison-report.md
   - benchmark-report.md
   - test results (JSON)
7. Review reports and make merge decision
```

## Test Suites

### Smoke (Fast - 5 minutes)

Minimal validation for quick feedback:
- `ex_2Drobot-R-U` - Basic reachability
- `ex_2Drobot-R-D` - With noise
- `ex_4DBAS-S` - Safety example

### Critical (15 minutes)

Essential tests for Phase 1 I/O validation:
- `ex_2Drobot-R-U` - Basic test
- `ex_2Drobot-R-D` - Noise test
- `ex_load_reach` - Load function test (critical for Phase 1)
- `ex_load_safe` - Load function test (critical for Phase 1)
- `ex_4DBAS-S` - Safety test

### Comprehensive (60+ minutes)

All 15 examples:
- All small examples (2D robots, 4D BAS, custom PDFs)
- All medium examples (reach-avoid variants, 3D vehicle)
- All large examples (3D/5D/7D/14D systems)
- Special cases (load tests)

## Output Files

### test_results.json

Structured test results including:
```json
{
  "name": "ex_2Drobot-R-U",
  "status": "PASS",
  "compile": {
    "success": true,
    "time_seconds": 12.3
  },
  "run": {
    "success": true,
    "time_seconds": 45.6
  },
  "validate": {
    "success": true,
    "found_files": ["ss.h5", "is.h5", "controller.h5"]
  }
}
```

### comparison-report.md

Detailed comparison report including:
- Executive summary
- Compilation results table
- Output file comparison table
- Numerical accuracy comparison
- Issues detected
- Final verdict and recommendation

### benchmark-report.md

Performance comparison report including:
- Compilation time comparison
- Runtime comparison
- Regression flags
- Verdict

## Troubleshooting

### Compilation Failures

**Issue**: Example fails to compile
**Check**:
- Are all dependencies installed? (see [installation.md](../installation.md))
- Is the Makefile correct for your platform?
- Try compiling manually: `cd examples/ex_2Drobot-R-U && make`

**Issue**: Timeout during compilation
**Solution**: Increase `compile_timeout` in test_config.yaml

### Runtime Failures

**Issue**: Example fails to execute
**Check**:
- Does the executable exist after compilation?
- Check for runtime errors in logs
- Try running manually: `cd examples/ex_2Drobot-R-U && ./robot2D`

**Issue**: Timeout during execution
**Solution**: Increase `run_timeout` in test_config.yaml

### Output Comparison Issues

**Issue**: Files marked as DIFFERENT
**Check**:
- Is the difference within acceptable tolerance?
- Are the differences due to floating-point precision?
- Try increasing tolerance: `--tolerance 1e-10`

**Issue**: Missing output files
**Check**:
- Did the example run successfully?
- Are the expected_outputs correct in test_config.yaml?

### Docker Issues

**Issue**: Docker build fails
**Solution**: Check Dockerfile and dependencies

**Issue**: Tests fail in Docker but work locally
**Check**:
- Are all dependencies in the Docker image?
- Check Docker logs for specific errors

## Best Practices

1. **Run smoke tests frequently** - Quick feedback on changes
2. **Run comprehensive tests before merge** - Ensure no regressions
3. **Compare outputs numerically** - Don't rely on file sizes alone
4. **Review performance** - Catch unexpected slowdowns
5. **Keep test_config.yaml updated** - Add new examples as they're created
6. **Document test failures** - Help others reproduce issues

## Integration with Development Workflow

### Before Committing
```bash
python test/test_runner.py --suite smoke
```

### Before Creating PR
```bash
python test/test_runner.py --suite critical
```

### Before Merging PR
```bash
# Use GitHub Actions branch-comparison workflow
# Review generated reports
# Ensure all tests pass
```

## Future Enhancements

Potential improvements:
- [ ] Unit tests for core algorithms
- [ ] GPU-specific test runners
- [ ] Cross-platform testing (Linux, macOS, Windows)
- [ ] Performance regression tracking over time
- [ ] Automatic PR comments with test results
- [ ] Coverage analysis

## Support

For issues or questions:
- Check [installation.md](../installation.md) for setup help
- Check [setup.md](../setup.md) for configuration guide
- Review [YouTube tutorials](https://www.youtube.com/playlist?list=PL50OJg3FHS4fBxhua92ZS3e6bxEnFaetL)
- Open an issue on GitHub

---

*Testing infrastructure created for IMPaCT v1.1*
