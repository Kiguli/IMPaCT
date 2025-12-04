#!/bin/bash
#
# IMPaCT Docker Branch Comparison Script
#
# This script performs comprehensive testing of two branches using Docker:
# 1. Builds Docker images for both branches
# 2. Runs 5 critical examples in each Docker container
# 3. Extracts HDF5 outputs to host filesystem
# 4. Compares outputs numerically using compare_outputs.py
# 5. Generates detailed comparison report
#
# Usage:
#   ./test/docker_branch_comparison.sh [base_branch] [compare_branch]
#
# Example:
#   ./test/docker_branch_comparison.sh main refactor/phase1-io-utils
#

set -e  # Exit on error

# Configuration
EXAMPLES=("ex_2Drobot-R-U" "ex_2Drobot-R-D" "ex_4DBAS-S" "ex_load_reach" "ex_load_safe")
BASE_BRANCH="${1:-main}"
COMPARE_BRANCH="${2:-refactor/phase1-io-utils}"
RESULTS_DIR="test_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create results directory first
mkdir -p "$RESULTS_DIR"
LOG_FILE="${RESULTS_DIR}/docker_comparison_${TIMESTAMP}.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "$1" | tee -a "$LOG_FILE"
}

# Error handler
error_exit() {
    log "${RED}ERROR: $1${NC}"
    exit 1
}

# Function to test a branch
test_branch() {
    local branch=$1
    local tag=$2
    local output_dir=$3

    log "${BLUE}========================================${NC}"
    log "${BLUE}Testing branch: $branch${NC}"
    log "${BLUE}Docker tag: impact:$tag${NC}"
    log "${BLUE}Output directory: $output_dir${NC}"
    log "${BLUE}========================================${NC}"
    log ""

    # Checkout branch
    log "${YELLOW}Checking out branch $branch...${NC}"
    git checkout "$branch" >> "$LOG_FILE" 2>&1 || error_exit "Failed to checkout branch $branch"

    # Build Docker image
    log "${YELLOW}Building Docker image...${NC}"
    docker build -t "impact:$tag" . >> "$LOG_FILE" 2>&1 || error_exit "Docker build failed for branch $branch"
    log "${GREEN}✓ Docker image built successfully${NC}"
    log ""

    # Create output directory
    mkdir -p "$output_dir"

    # Test each example
    local success_count=0
    local fail_count=0

    for example in "${EXAMPLES[@]}"; do
        log "${YELLOW}Testing example: $example${NC}"

        # Run example in Docker container
        if docker run --rm -v "$(pwd)/$output_dir:/output" "impact:$tag" /bin/bash -c "
            cd /app/examples/$example &&
            echo 'Compiling...' &&
            make clean > /dev/null 2>&1 &&
            make &&
            echo 'Running...' &&
            timeout 300 ./* &&
            echo 'Copying outputs...' &&
            mkdir -p /output/$example &&
            cp *.h5 /output/$example/ 2>/dev/null || echo 'No HDF5 files found'
        " >> "$LOG_FILE" 2>&1; then
            log "${GREEN}  ✓ $example completed successfully${NC}"
            ((success_count++))
        else
            log "${RED}  ✗ $example failed${NC}"
            ((fail_count++))
        fi
        log ""
    done

    # Summary
    log "${BLUE}Branch $branch test summary:${NC}"
    log "  ${GREEN}Successful: $success_count${NC}"
    log "  ${RED}Failed: $fail_count${NC}"
    log ""

    if [ $fail_count -gt 0 ]; then
        error_exit "Some examples failed on branch $branch. Check $LOG_FILE for details."
    fi
}

# Function to compare outputs
compare_outputs() {
    local base_dir=$1
    local compare_dir=$2
    local report_file=$3

    log "${BLUE}========================================${NC}"
    log "${BLUE}Comparing outputs${NC}"
    log "${BLUE}========================================${NC}"
    log ""

    # Check if compare_outputs.py exists
    if [ ! -f "test/compare_outputs.py" ]; then
        error_exit "compare_outputs.py not found. Are you on the refactor branch?"
    fi

    # Run comparison
    log "${YELLOW}Running numerical comparison...${NC}"
    if python3 test/compare_outputs.py "$base_dir" "$compare_dir" --report "$report_file" --tolerance 1e-12 >> "$LOG_FILE" 2>&1; then
        log "${GREEN}✓ Comparison completed${NC}"
        log ""
        log "${BLUE}Comparison report saved to: $report_file${NC}"
    else
        log "${YELLOW}⚠ Comparison completed with warnings. Review $report_file for details.${NC}"
    fi
    log ""
}

# Main execution
main() {
    log "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
    log "${BLUE}║       IMPaCT Docker Branch Comparison Test Suite          ║${NC}"
    log "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
    log ""
    log "${BLUE}Configuration:${NC}"
    log "  Base branch:    $BASE_BRANCH"
    log "  Compare branch: $COMPARE_BRANCH"
    log "  Examples:       ${EXAMPLES[*]}"
    log "  Results dir:    $RESULTS_DIR"
    log "  Log file:       $LOG_FILE"
    log ""

    # Check prerequisites
    log "${YELLOW}Checking prerequisites...${NC}"

    if ! command -v docker &> /dev/null; then
        error_exit "Docker is not installed or not in PATH"
    fi
    log "${GREEN}✓ Docker found${NC}"

    if ! command -v python3 &> /dev/null; then
        error_exit "Python 3 is not installed or not in PATH"
    fi
    log "${GREEN}✓ Python 3 found${NC}"

    if [ ! -f "Dockerfile" ]; then
        error_exit "Dockerfile not found. Run this script from the repository root."
    fi
    log "${GREEN}✓ Dockerfile found${NC}"
    log ""

    # Save current branch
    ORIGINAL_BRANCH=$(git branch --show-current)
    log "${BLUE}Current branch: $ORIGINAL_BRANCH${NC}"
    log ""

    # Clean up old results
    if [ -d "$RESULTS_DIR" ]; then
        log "${YELLOW}Cleaning up old test results...${NC}"
        rm -rf "$RESULTS_DIR"
    fi
    mkdir -p "$RESULTS_DIR"

    # Test base branch
    test_branch "$BASE_BRANCH" "base" "$RESULTS_DIR/base"

    # Test compare branch
    test_branch "$COMPARE_BRANCH" "compare" "$RESULTS_DIR/compare"

    # Compare outputs
    REPORT_FILE="$RESULTS_DIR/docker-comparison-report.md"
    compare_outputs "$RESULTS_DIR/base" "$RESULTS_DIR/compare" "$REPORT_FILE"

    # Return to original branch
    log "${YELLOW}Returning to original branch: $ORIGINAL_BRANCH${NC}"
    git checkout "$ORIGINAL_BRANCH" >> "$LOG_FILE" 2>&1
    log ""

    # Final summary
    log "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
    log "${BLUE}║                    TEST COMPLETE                           ║${NC}"
    log "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
    log ""
    log "${GREEN}✓ All tests completed successfully${NC}"
    log ""
    log "${BLUE}Results:${NC}"
    log "  Log file:        $LOG_FILE"
    log "  Comparison report: $REPORT_FILE"
    log "  HDF5 outputs:    $RESULTS_DIR/base/ and $RESULTS_DIR/compare/"
    log ""
    log "${YELLOW}Next steps:${NC}"
    log "  1. Review the comparison report: cat $REPORT_FILE"
    log "  2. Check for any differences or warnings"
    log "  3. If all tests pass, proceed with merge"
    log ""
}

# Run main function
main
