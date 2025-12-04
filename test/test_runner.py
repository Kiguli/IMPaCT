#!/usr/bin/env python3
"""
IMPaCT Test Runner

Automated test runner for IMPaCT examples that:
- Discovers and compiles all examples
- Executes with timeout protection
- Collects and validates HDF5 outputs
- Returns structured test results

Usage:
    python test_runner.py --suite smoke
    python test_runner.py --suite comprehensive
    python test_runner.py --example ex_2Drobot-R-U
    python test_runner.py --all
"""

import argparse
import subprocess
import time
import os
import sys
from pathlib import Path
import yaml
import json
from typing import Dict, List, Tuple, Optional

class TestRunner:
    """Main test runner class"""

    def __init__(self, config_path: str = "test/test_config.yaml", root_dir: Optional[str] = None):
        """Initialize test runner with configuration"""
        if root_dir:
            self.root_dir = Path(root_dir)
        else:
            # Assume we're running from root or test directory
            if Path.cwd().name == "test":
                self.root_dir = Path.cwd().parent
            else:
                self.root_dir = Path.cwd()

        self.config_path = self.root_dir / config_path
        self.load_config()
        self.results = []

    def load_config(self):
        """Load test configuration from YAML"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            print(f"✅ Loaded configuration from {self.config_path}")
        except FileNotFoundError:
            print(f"❌ Configuration file not found: {self.config_path}")
            sys.exit(1)
        except yaml.YAMLError as e:
            print(f"❌ Error parsing configuration: {e}")
            sys.exit(1)

    def get_examples(self, suite: Optional[str] = None, example: Optional[str] = None, all_examples: bool = False) -> List[str]:
        """Get list of examples to run based on arguments"""
        if example:
            return [example]
        elif suite:
            if suite in self.config['test_suites']:
                return self.config['test_suites'][suite]['examples']
            else:
                print(f"❌ Unknown test suite: {suite}")
                print(f"Available suites: {list(self.config['test_suites'].keys())}")
                sys.exit(1)
        elif all_examples:
            # Run all examples from all categories
            examples = []
            for category in ['small', 'medium', 'large', 'special']:
                if category in self.config['examples']:
                    examples.extend([ex['name'] for ex in self.config['examples'][category]])
            return examples
        else:
            print("❌ Must specify --suite, --example, or --all")
            sys.exit(1)

    def find_example_config(self, example_name: str) -> Optional[Dict]:
        """Find configuration for a specific example"""
        for category in ['small', 'medium', 'large', 'special']:
            if category in self.config['examples']:
                for ex in self.config['examples'][category]:
                    if ex['name'] == example_name:
                        return ex
        return None

    def compile_example(self, example_config: Dict) -> Tuple[bool, float, str]:
        """
        Compile an example using its Makefile

        Returns:
            (success, compile_time, error_message)
        """
        example_path = self.root_dir / example_config['path']
        timeout = example_config.get('compile_timeout', 300)

        print(f"📦 Compiling {example_config['name']}...")
        print(f"   Path: {example_path}")

        start_time = time.time()

        try:
            # Run make clean first
            subprocess.run(
                ['make', 'clean'],
                cwd=example_path,
                capture_output=True,
                timeout=30,
                check=False  # Don't fail if no clean target
            )

            # Run make
            result = subprocess.run(
                ['make'],
                cwd=example_path,
                capture_output=True,
                timeout=timeout,
                text=True
            )

            compile_time = time.time() - start_time

            if result.returncode == 0:
                print(f"✅ Compilation successful ({compile_time:.1f}s)")
                return True, compile_time, ""
            else:
                error_msg = f"Make failed with return code {result.returncode}\n"
                error_msg += f"STDOUT:\n{result.stdout}\n"
                error_msg += f"STDERR:\n{result.stderr}"
                print(f"❌ Compilation failed ({compile_time:.1f}s)")
                return False, compile_time, error_msg

        except subprocess.TimeoutExpired:
            compile_time = time.time() - start_time
            error_msg = f"Compilation timeout after {timeout}s"
            print(f"❌ {error_msg}")
            return False, compile_time, error_msg
        except Exception as e:
            compile_time = time.time() - start_time
            error_msg = f"Compilation error: {str(e)}"
            print(f"❌ {error_msg}")
            return False, compile_time, error_msg

    def run_example(self, example_config: Dict) -> Tuple[bool, float, str]:
        """
        Execute a compiled example

        Returns:
            (success, run_time, error_message)
        """
        example_path = self.root_dir / example_config['path']
        executable = example_config['executable']
        timeout = example_config.get('run_timeout', 300)

        print(f"🚀 Running {example_config['name']}...")

        # Check executable exists
        executable_path = example_path / executable
        if not executable_path.exists():
            error_msg = f"Executable not found: {executable_path}"
            print(f"❌ {error_msg}")
            return False, 0.0, error_msg

        start_time = time.time()

        try:
            # Run the executable
            result = subprocess.run(
                [f'./{executable}'],
                cwd=example_path,
                capture_output=True,
                timeout=timeout,
                text=True
            )

            run_time = time.time() - start_time

            if result.returncode == 0:
                print(f"✅ Execution successful ({run_time:.1f}s)")
                return True, run_time, ""
            else:
                error_msg = f"Execution failed with return code {result.returncode}\n"
                error_msg += f"STDOUT:\n{result.stdout[:500]}\n"  # First 500 chars
                error_msg += f"STDERR:\n{result.stderr[:500]}"
                print(f"❌ Execution failed ({run_time:.1f}s)")
                return False, run_time, error_msg

        except subprocess.TimeoutExpired:
            run_time = time.time() - start_time
            error_msg = f"Execution timeout after {timeout}s"
            print(f"❌ {error_msg}")
            return False, run_time, error_msg
        except Exception as e:
            run_time = time.time() - start_time
            error_msg = f"Execution error: {str(e)}"
            print(f"❌ {error_msg}")
            return False, run_time, error_msg

    def validate_outputs(self, example_config: Dict) -> Tuple[bool, List[str], List[str]]:
        """
        Validate that expected output files exist

        Returns:
            (all_present, found_files, missing_files)
        """
        example_path = self.root_dir / example_config['path']
        expected = example_config.get('expected_outputs', [])

        found = []
        missing = []

        print(f"🔍 Validating outputs for {example_config['name']}...")

        for output_file in expected:
            output_path = example_path / output_file
            if output_path.exists():
                size = output_path.stat().st_size
                print(f"   ✅ {output_file} ({size / 1024:.1f} KB)")
                found.append(output_file)
            else:
                print(f"   ❌ {output_file} (missing)")
                missing.append(output_file)

        all_present = len(missing) == 0
        if all_present:
            print(f"✅ All {len(found)} output files present")
        else:
            print(f"⚠️  {len(missing)} output files missing")

        return all_present, found, missing

    def test_example(self, example_name: str) -> Dict:
        """
        Test a single example (compile, run, validate)

        Returns:
            Dictionary with test results
        """
        print("\n" + "="*70)
        print(f"Testing: {example_name}")
        print("="*70)

        # Find configuration
        example_config = self.find_example_config(example_name)
        if not example_config:
            error_msg = f"Example not found in configuration: {example_name}"
            print(f"❌ {error_msg}")
            return {
                'name': example_name,
                'status': 'ERROR',
                'error': error_msg
            }

        result = {
            'name': example_name,
            'description': example_config.get('description', ''),
            'path': example_config['path'],
            'status': 'PENDING',
            'compile': {},
            'run': {},
            'validate': {}
        }

        # Step 1: Compile
        compile_success, compile_time, compile_error = self.compile_example(example_config)
        result['compile'] = {
            'success': compile_success,
            'time_seconds': compile_time,
            'error': compile_error if not compile_success else None
        }

        if not compile_success:
            result['status'] = 'COMPILE_FAILED'
            return result

        # Step 2: Run
        run_success, run_time, run_error = self.run_example(example_config)
        result['run'] = {
            'success': run_success,
            'time_seconds': run_time,
            'error': run_error if not run_success else None
        }

        if not run_success:
            result['status'] = 'RUN_FAILED'
            return result

        # Step 3: Validate outputs
        outputs_valid, found_files, missing_files = self.validate_outputs(example_config)
        result['validate'] = {
            'success': outputs_valid,
            'found_files': found_files,
            'missing_files': missing_files
        }

        if not outputs_valid:
            result['status'] = 'VALIDATION_FAILED'
        else:
            result['status'] = 'PASS'

        print(f"\n✅ {example_name}: {result['status']}")
        return result

    def run_tests(self, examples: List[str]) -> List[Dict]:
        """Run tests on multiple examples"""
        print("\n" + "="*70)
        print(f"IMPaCT Test Runner - Testing {len(examples)} examples")
        print("="*70)

        results = []
        start_time = time.time()

        for example_name in examples:
            result = self.test_example(example_name)
            results.append(result)

        total_time = time.time() - start_time

        # Print summary
        self.print_summary(results, total_time)
        return results

    def print_summary(self, results: List[Dict], total_time: float):
        """Print test summary"""
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)

        # Count statuses
        passed = sum(1 for r in results if r['status'] == 'PASS')
        compile_failed = sum(1 for r in results if r['status'] == 'COMPILE_FAILED')
        run_failed = sum(1 for r in results if r['status'] == 'RUN_FAILED')
        validation_failed = sum(1 for r in results if r['status'] == 'VALIDATION_FAILED')
        errors = sum(1 for r in results if r['status'] == 'ERROR')

        print(f"\nTotal examples: {len(results)}")
        print(f"✅ Passed:             {passed}")
        print(f"❌ Compile failed:     {compile_failed}")
        print(f"❌ Run failed:         {run_failed}")
        print(f"❌ Validation failed:  {validation_failed}")
        print(f"❌ Errors:             {errors}")
        print(f"\nTotal time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")

        # Detailed results
        if compile_failed + run_failed + validation_failed + errors > 0:
            print("\n" + "="*70)
            print("FAILED TESTS")
            print("="*70)
            for result in results:
                if result['status'] != 'PASS':
                    print(f"\n{result['name']}: {result['status']}")
                    if result['status'] == 'COMPILE_FAILED':
                        print(f"  Compile error: {result['compile'].get('error', 'Unknown')[:200]}")
                    elif result['status'] == 'RUN_FAILED':
                        print(f"  Run error: {result['run'].get('error', 'Unknown')[:200]}")
                    elif result['status'] == 'VALIDATION_FAILED':
                        missing = result['validate'].get('missing_files', [])
                        print(f"  Missing files: {', '.join(missing)}")

        # Overall status
        print("\n" + "="*70)
        if passed == len(results):
            print("🎉 ALL TESTS PASSED")
        else:
            print(f"⚠️  {len(results) - passed} OF {len(results)} TESTS FAILED")
        print("="*70)

    def save_results(self, results: List[Dict], output_file: str = "test_results.json"):
        """Save results to JSON file"""
        output_path = self.root_dir / output_file
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n📝 Results saved to: {output_path}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="IMPaCT Test Runner - Automated testing for IMPaCT examples"
    )
    parser.add_argument(
        '--suite',
        choices=['smoke', 'comprehensive', 'critical'],
        help='Run a predefined test suite'
    )
    parser.add_argument(
        '--example',
        help='Run a specific example (e.g., ex_2Drobot-R-U)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all examples'
    )
    parser.add_argument(
        '--config',
        default='test/test_config.yaml',
        help='Path to configuration file (default: test/test_config.yaml)'
    )
    parser.add_argument(
        '--output',
        default='test_results.json',
        help='Output file for results (default: test_results.json)'
    )
    parser.add_argument(
        '--root',
        help='Root directory of IMPaCT project (default: auto-detect)'
    )

    args = parser.parse_args()

    # Validate arguments
    if not (args.suite or args.example or args.all):
        parser.error("Must specify --suite, --example, or --all")

    # Create test runner
    runner = TestRunner(config_path=args.config, root_dir=args.root)

    # Get examples to test
    examples = runner.get_examples(
        suite=args.suite,
        example=args.example,
        all_examples=args.all
    )

    # Run tests
    results = runner.run_tests(examples)

    # Save results
    runner.save_results(results, args.output)

    # Exit with appropriate code
    passed = sum(1 for r in results if r['status'] == 'PASS')
    if passed == len(results):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
