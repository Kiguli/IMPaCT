#!/usr/bin/env python3
"""
IMPaCT Benchmark Tool

Compares performance (compilation time, execution time) between branches.

Usage:
    python benchmark.py main_results.json refactor_results.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

class BenchmarkComparator:
    """Compare performance between two test runs"""

    def __init__(self, regression_threshold: float = 0.10):
        """
        Initialize benchmark comparator

        Args:
            regression_threshold: Threshold for flagging regressions (default: 10%)
        """
        self.regression_threshold = regression_threshold

    def load_results(self, results_file: Path) -> List[Dict]:
        """Load test results from JSON file"""
        try:
            with open(results_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Error loading {results_file}: {e}")
            sys.exit(1)

    def compare_performance(
        self,
        main_results: List[Dict],
        refactor_results: List[Dict]
    ) -> List[Dict]:
        """
        Compare performance between two test runs

        Args:
            main_results: Results from main branch
            refactor_results: Results from refactor branch

        Returns:
            List of performance comparisons
        """
        comparisons = []

        # Create lookup for refactor results
        refactor_lookup = {r['name']: r for r in refactor_results}

        for main_result in main_results:
            example_name = main_result['name']

            if example_name not in refactor_lookup:
                print(f"⚠️ {example_name} not found in refactor results")
                continue

            refactor_result = refactor_lookup[example_name]

            comparison = {
                'example': example_name,
                'main_compile_time': main_result.get('compile', {}).get('time_seconds', 0),
                'refactor_compile_time': refactor_result.get('compile', {}).get('time_seconds', 0),
                'main_run_time': main_result.get('run', {}).get('time_seconds', 0),
                'refactor_run_time': refactor_result.get('run', {}).get('time_seconds', 0),
            }

            # Calculate differences
            if comparison['main_compile_time'] > 0:
                compile_diff = comparison['refactor_compile_time'] - comparison['main_compile_time']
                compile_pct = (compile_diff / comparison['main_compile_time']) * 100
                comparison['compile_diff_seconds'] = compile_diff
                comparison['compile_diff_pct'] = compile_pct
            else:
                comparison['compile_diff_seconds'] = 0
                comparison['compile_diff_pct'] = 0

            if comparison['main_run_time'] > 0:
                run_diff = comparison['refactor_run_time'] - comparison['main_run_time']
                run_pct = (run_diff / comparison['main_run_time']) * 100
                comparison['run_diff_seconds'] = run_diff
                comparison['run_diff_pct'] = run_pct
            else:
                comparison['run_diff_seconds'] = 0
                comparison['run_diff_pct'] = 0

            # Flag regressions
            comparison['compile_regression'] = comparison['compile_diff_pct'] > (self.regression_threshold * 100)
            comparison['run_regression'] = comparison['run_diff_pct'] > (self.regression_threshold * 100)

            comparisons.append(comparison)

        return comparisons

    def generate_report(self, comparisons: List[Dict], output_file: str = "benchmark-report.md") -> str:
        """Generate benchmark report"""
        report = []
        report.append("# IMPaCT Performance Benchmark Report")
        report.append("")

        # Summary statistics
        total = len(comparisons)
        compile_regressions = sum(1 for c in comparisons if c['compile_regression'])
        run_regressions = sum(1 for c in comparisons if c['run_regression'])

        report.append("## Summary")
        report.append("")
        report.append(f"- **Examples tested**: {total}")
        report.append(f"- **Compilation regressions**: {compile_regressions}")
        report.append(f"- **Runtime regressions**: {run_regressions}")
        report.append(f"- **Regression threshold**: {self.regression_threshold*100:.0f}%")
        report.append("")

        # Compilation performance
        report.append("## Compilation Performance")
        report.append("")
        report.append("| Example | Main Time | Refactor Time | Diff | % Change | Status |")
        report.append("|---------|-----------|---------------|------|----------|--------|")

        for comp in comparisons:
            main_time = comp['main_compile_time']
            refactor_time = comp['refactor_compile_time']
            diff = comp['compile_diff_seconds']
            pct = comp['compile_diff_pct']

            if comp['compile_regression']:
                status = f"⚠️ REGRESSION"
            elif pct < -5:
                status = "✅ IMPROVED"
            else:
                status = "✅ OK"

            report.append(
                f"| {comp['example']} | {main_time:.1f}s | {refactor_time:.1f}s | "
                f"{diff:+.1f}s | {pct:+.1f}% | {status} |"
            )
        report.append("")

        # Runtime performance
        report.append("## Runtime Performance")
        report.append("")
        report.append("| Example | Main Time | Refactor Time | Diff | % Change | Status |")
        report.append("|---------|-----------|---------------|------|----------|--------|")

        for comp in comparisons:
            main_time = comp['main_run_time']
            refactor_time = comp['refactor_run_time']
            diff = comp['run_diff_seconds']
            pct = comp['run_diff_pct']

            if comp['run_regression']:
                status = f"⚠️ REGRESSION"
            elif pct < -5:
                status = "✅ IMPROVED"
            else:
                status = "✅ OK"

            report.append(
                f"| {comp['example']} | {main_time:.1f}s | {refactor_time:.1f}s | "
                f"{diff:+.1f}s | {pct:+.1f}% | {status} |"
            )
        report.append("")

        # Verdict
        report.append("## Verdict")
        report.append("")

        if compile_regressions == 0 and run_regressions == 0:
            report.append("✅ **NO PERFORMANCE REGRESSIONS DETECTED**")
        else:
            report.append(f"⚠️ **{compile_regressions + run_regressions} REGRESSIONS DETECTED**")
            report.append("")
            report.append("Review the flagged examples above.")

        report.append("")
        report.append("---")
        report.append(f"*Generated by IMPaCT Benchmark Tool (threshold: {self.regression_threshold*100:.0f}%)*")

        # Save report
        report_text = "\n".join(report)
        with open(output_file, 'w') as f:
            f.write(report_text)

        print(f"\n📝 Benchmark report saved to: {output_file}")
        return report_text


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="IMPaCT Benchmark Tool - Compare performance between branches"
    )
    parser.add_argument(
        'main_results',
        type=Path,
        help='Main branch test_results.json file'
    )
    parser.add_argument(
        'refactor_results',
        type=Path,
        help='Refactor branch test_results.json file'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.10,
        help='Regression threshold (default: 0.10 = 10%%)'
    )
    parser.add_argument(
        '--report',
        default='benchmark-report.md',
        help='Output report file (default: benchmark-report.md)'
    )

    args = parser.parse_args()

    # Create comparator
    comparator = BenchmarkComparator(regression_threshold=args.threshold)

    # Load results
    print(f"Loading main branch results from {args.main_results}...")
    main_results = comparator.load_results(args.main_results)
    print(f"✅ Loaded {len(main_results)} results")

    print(f"Loading refactor branch results from {args.refactor_results}...")
    refactor_results = comparator.load_results(args.refactor_results)
    print(f"✅ Loaded {len(refactor_results)} results")

    # Compare performance
    print("\nComparing performance...")
    comparisons = comparator.compare_performance(main_results, refactor_results)

    # Generate report
    report = comparator.generate_report(comparisons, args.report)

    # Print summary
    print("\n" + "="*70)
    print("PERFORMANCE SUMMARY")
    print("="*70)

    compile_regressions = sum(1 for c in comparisons if c['compile_regression'])
    run_regressions = sum(1 for c in comparisons if c['run_regression'])

    print(f"Compilation regressions: {compile_regressions}/{len(comparisons)}")
    print(f"Runtime regressions: {run_regressions}/{len(comparisons)}")

    if compile_regressions == 0 and run_regressions == 0:
        print("\n✅ NO PERFORMANCE REGRESSIONS")
        sys.exit(0)
    else:
        print(f"\n⚠️ {compile_regressions + run_regressions} REGRESSIONS DETECTED")
        sys.exit(1)


if __name__ == "__main__":
    main()
