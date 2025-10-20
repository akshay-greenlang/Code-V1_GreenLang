"""
Comprehensive Test Execution Framework for CSRD Reporting Platform
===================================================================

Orchestrates all test suites with detailed reporting and metrics.

Author: GreenLang QA Team
Date: 2025-10-20
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import re


class TestRunner:
    """Comprehensive test execution and reporting framework."""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.tests_dir = self.project_root / "tests"
        self.reports_dir = self.project_root / "test-reports"
        self.reports_dir.mkdir(exist_ok=True)

        self.results = {
            "timestamp": datetime.now().isoformat(),
            "project": "CSRD-Reporting-Platform",
            "test_suites": {},
            "summary": {}
        }

    def discover_tests(self) -> Dict[str, List[str]]:
        """Discover all test files and categorize them."""
        print("\n" + "="*80)
        print("DISCOVERING TEST SUITE")
        print("="*80)

        test_categories = {
            "unit": [],
            "integration": [],
            "security": [],
            "performance": [],
            "e2e": []
        }

        # Find all test files
        test_files = list(self.tests_dir.glob("test_*.py"))

        for test_file in test_files:
            filename = test_file.name

            # Categorize tests
            if "security" in filename.lower() or "encryption" in filename.lower() or "validation" in filename.lower():
                test_categories["security"].append(str(test_file))
            elif "integration" in filename.lower() or "pipeline" in filename.lower():
                test_categories["integration"].append(str(test_file))
            elif "performance" in filename.lower() or "benchmark" in filename.lower():
                test_categories["performance"].append(str(test_file))
            elif "e2e" in filename.lower() or "end_to_end" in filename.lower():
                test_categories["e2e"].append(str(test_file))
            else:
                test_categories["unit"].append(str(test_file))

        # Print discovery results
        print(f"\n📋 Test Suite Discovery:")
        for category, files in test_categories.items():
            print(f"  {category.upper()}: {len(files)} test files")
            for test_file in files:
                print(f"    - {Path(test_file).name}")

        print(f"\n  TOTAL: {sum(len(files) for files in test_categories.values())} test files")

        return test_categories

    def run_test_suite(self, name: str, test_files: List[str], markers: str = None) -> Dict:
        """Run a specific test suite with pytest."""
        print(f"\n{'='*80}")
        print(f"RUNNING {name.upper()} TEST SUITE")
        print(f"{'='*80}")

        if not test_files:
            print(f"  No {name} tests found - skipping")
            return {
                "status": "skipped",
                "reason": "no test files"
            }

        start_time = time.time()

        # Build pytest command
        cmd = [
            "pytest",
            *test_files,
            "-v",                                    # Verbose
            "--tb=short",                            # Short traceback
            "--strict-markers",                      # Strict marker checking
            f"--junit-xml={self.reports_dir}/{name}_junit.xml",
            f"--html={self.reports_dir}/{name}_report.html",
            "--self-contained-html",                 # Standalone HTML report
            "--cov=agents",                          # Coverage for agents
            "--cov=utils",                           # Coverage for utils
            f"--cov-report=html:{self.reports_dir}/{name}_coverage",
            f"--cov-report=json:{self.reports_dir}/{name}_coverage.json",
            "--cov-report=term-missing",             # Show missing lines
        ]

        # Add markers if specified
        if markers:
            cmd.extend(["-m", markers])

        print(f"\nCommand: {' '.join(cmd)}\n")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,  # 30 minute timeout
                cwd=self.project_root
            )

            duration = time.time() - start_time

            # Parse output
            output = result.stdout + result.stderr

            # Extract test counts
            passed = len(re.findall(r'PASSED', output))
            failed = len(re.findall(r'FAILED', output))
            skipped = len(re.findall(r'SKIPPED', output))
            errors = len(re.findall(r'ERROR', output))
            total = passed + failed + skipped + errors

            # Extract coverage
            coverage_match = re.search(r'TOTAL.*?(\d+)%', output)
            coverage = int(coverage_match.group(1)) if coverage_match else 0

            suite_result = {
                "status": "completed",
                "duration": round(duration, 2),
                "total_tests": total,
                "passed": passed,
                "failed": failed,
                "skipped": skipped,
                "errors": errors,
                "pass_rate": round((passed / total * 100) if total > 0 else 0, 2),
                "coverage": coverage,
                "exit_code": result.returncode,
                "reports": {
                    "junit": f"{name}_junit.xml",
                    "html": f"{name}_report.html",
                    "coverage_html": f"{name}_coverage/index.html",
                    "coverage_json": f"{name}_coverage.json"
                }
            }

            # Print summary
            print(f"\n📊 {name.upper()} Test Results:")
            print(f"  Duration: {suite_result['duration']}s")
            print(f"  Total Tests: {suite_result['total_tests']}")
            print(f"  ✅ Passed: {suite_result['passed']}")
            print(f"  ❌ Failed: {suite_result['failed']}")
            print(f"  ⏭️  Skipped: {suite_result['skipped']}")
            print(f"  ⚠️  Errors: {suite_result['errors']}")
            print(f"  Pass Rate: {suite_result['pass_rate']}%")
            print(f"  Coverage: {suite_result['coverage']}%")

            if suite_result['pass_rate'] < 95:
                print(f"\n⚠️  WARNING: Pass rate below 95% target")
            if suite_result['coverage'] < 80:
                print(f"\n⚠️  WARNING: Coverage below 80% target")

            return suite_result

        except subprocess.TimeoutExpired:
            print(f"\n❌ {name} tests TIMEOUT after 30 minutes")
            return {
                "status": "timeout",
                "duration": 1800,
                "error": "Test execution exceeded 30 minute timeout"
            }
        except FileNotFoundError:
            print(f"\n❌ pytest not found - install with: pip install pytest")
            return {
                "status": "error",
                "error": "pytest not installed"
            }
        except Exception as e:
            print(f"\n❌ {name} tests FAILED with error: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    def run_all_tests(self) -> Dict:
        """Run all test suites in sequence."""
        print("\n" + "="*80)
        print("CSRD REPORTING PLATFORM - COMPREHENSIVE TEST EXECUTION")
        print("="*80)
        print(f"Timestamp: {self.results['timestamp']}")
        print(f"Project: {self.project_root}")
        print("="*80)

        # Discover tests
        test_categories = self.discover_tests()

        # Run test suites in order
        print("\n" + "="*80)
        print("EXECUTING TEST SUITES")
        print("="*80)

        # 1. Unit Tests (fastest, run first)
        if test_categories["unit"]:
            print("\n🔧 Running Unit Tests...")
            self.results["test_suites"]["unit"] = self.run_test_suite(
                "unit",
                test_categories["unit"]
            )

        # 2. Security Tests (critical)
        if test_categories["security"]:
            print("\n🔒 Running Security Tests...")
            self.results["test_suites"]["security"] = self.run_test_suite(
                "security",
                test_categories["security"]
            )

        # 3. Integration Tests
        if test_categories["integration"]:
            print("\n🔗 Running Integration Tests...")
            self.results["test_suites"]["integration"] = self.run_test_suite(
                "integration",
                test_categories["integration"]
            )

        # 4. Performance Tests
        if test_categories["performance"]:
            print("\n⚡ Running Performance Tests...")
            self.results["test_suites"]["performance"] = self.run_test_suite(
                "performance",
                test_categories["performance"]
            )

        # 5. End-to-End Tests (slowest, run last)
        if test_categories["e2e"]:
            print("\n🎯 Running End-to-End Tests...")
            self.results["test_suites"]["e2e"] = self.run_test_suite(
                "e2e",
                test_categories["e2e"]
            )

        # Generate summary
        self.generate_summary()

        return self.results

    def generate_summary(self) -> None:
        """Generate overall test execution summary."""
        print("\n" + "="*80)
        print("TEST EXECUTION SUMMARY")
        print("="*80)

        total_tests = 0
        total_passed = 0
        total_failed = 0
        total_skipped = 0
        total_errors = 0
        total_duration = 0
        coverage_values = []

        for suite_name, suite_results in self.results["test_suites"].items():
            if suite_results.get("status") == "completed":
                total_tests += suite_results.get("total_tests", 0)
                total_passed += suite_results.get("passed", 0)
                total_failed += suite_results.get("failed", 0)
                total_skipped += suite_results.get("skipped", 0)
                total_errors += suite_results.get("errors", 0)
                total_duration += suite_results.get("duration", 0)

                coverage = suite_results.get("coverage", 0)
                if coverage > 0:
                    coverage_values.append(coverage)

        # Calculate averages
        pass_rate = round((total_passed / total_tests * 100) if total_tests > 0 else 0, 2)
        avg_coverage = round(sum(coverage_values) / len(coverage_values) if coverage_values else 0, 2)

        self.results["summary"] = {
            "total_tests": total_tests,
            "passed": total_passed,
            "failed": total_failed,
            "skipped": total_skipped,
            "errors": total_errors,
            "pass_rate": pass_rate,
            "average_coverage": avg_coverage,
            "total_duration": round(total_duration, 2),
            "status": "PASS" if (pass_rate >= 95 and avg_coverage >= 80 and total_failed == 0) else "FAIL"
        }

        summary = self.results["summary"]

        # Print summary
        print(f"\n📊 Overall Results:")
        print(f"  Total Tests: {summary['total_tests']}")
        print(f"  ✅ Passed: {summary['passed']}")
        print(f"  ❌ Failed: {summary['failed']}")
        print(f"  ⏭️  Skipped: {summary['skipped']}")
        print(f"  ⚠️  Errors: {summary['errors']}")
        print(f"  Pass Rate: {summary['pass_rate']}%")
        print(f"  Average Coverage: {summary['average_coverage']}%")
        print(f"  Total Duration: {summary['total_duration']}s ({round(summary['total_duration']/60, 1)} minutes)")
        print(f"\n  Overall Status: {summary['status']}")

        # Quality gates
        print(f"\n🚦 Quality Gates:")
        print(f"  Pass Rate ≥95%:        {'✅ PASS' if summary['pass_rate'] >= 95 else '❌ FAIL'} ({summary['pass_rate']}%)")
        print(f"  Coverage ≥80%:         {'✅ PASS' if summary['average_coverage'] >= 80 else '❌ FAIL'} ({summary['average_coverage']}%)")
        print(f"  Zero Critical Failures: {'✅ PASS' if summary['failed'] == 0 else '❌ FAIL'} ({summary['failed']} failures)")

        # Save results
        summary_path = self.reports_dir / "test_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\n📄 Full report saved to: {summary_path}")
        print(f"📁 All reports in: {self.reports_dir}")
        print("="*80)

        if summary['status'] == "PASS":
            print("\n✅ ALL TESTS PASSED - READY FOR PRODUCTION")
        else:
            print("\n❌ TESTS FAILED - REVIEW FAILURES BEFORE DEPLOYMENT")

    def run_quick_smoke_tests(self) -> Dict:
        """Run quick smoke tests for rapid feedback."""
        print("\n" + "="*80)
        print("RUNNING QUICK SMOKE TESTS")
        print("="*80)

        cmd = [
            "pytest",
            str(self.tests_dir),
            "-v",
            "-x",                     # Stop on first failure
            "--maxfail=3",            # Stop after 3 failures
            "-m", "smoke",            # Only smoke tests
            "--tb=line",              # Line-only traceback
            "--duration=0",           # Show test durations
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                cwd=self.project_root
            )

            if result.returncode == 0:
                print("✅ Smoke tests PASSED")
                return {"status": "passed"}
            else:
                print("❌ Smoke tests FAILED")
                print(result.stdout)
                return {"status": "failed", "output": result.stdout}

        except Exception as e:
            print(f"❌ Smoke tests ERROR: {e}")
            return {"status": "error", "error": str(e)}


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="CSRD Test Runner")
    parser.add_argument("--suite", choices=["all", "unit", "integration", "security", "performance", "e2e", "smoke"],
                       default="all", help="Test suite to run")
    parser.add_argument("--quick", action="store_true", help="Run quick smoke tests only")
    args = parser.parse_args()

    runner = TestRunner()

    if args.quick:
        result = runner.run_quick_smoke_tests()
        sys.exit(0 if result["status"] == "passed" else 1)

    if args.suite == "all":
        result = runner.run_all_tests()
    else:
        test_categories = runner.discover_tests()
        result = {
            "test_suites": {
                args.suite: runner.run_test_suite(args.suite, test_categories.get(args.suite, []))
            }
        }
        runner.results = result
        runner.generate_summary()

    # Exit with appropriate code
    summary = runner.results.get("summary", {})
    sys.exit(0 if summary.get("status") == "PASS" else 1)


if __name__ == "__main__":
    main()
