#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GL-007 FurnacePerformanceMonitor Test Runner

Convenient script to run tests with various configurations.

Usage:
    python run_tests.py                    # Run all tests
    python run_tests.py --unit             # Run unit tests only
    python run_tests.py --integration      # Run integration tests only
    python run_tests.py --e2e              # Run e2e tests only
    python run_tests.py --coverage         # Run with coverage report
    python run_tests.py --fast             # Run fast tests only (skip slow)
    python run_tests.py --benchmark        # Run performance benchmarks
    python run_tests.py --compliance       # Run compliance tests only

Version: 1.0.0
Date: 2025-11-21
"""

import sys
import subprocess
from pathlib import Path


def run_command(cmd: list) -> int:
    """Run command and return exit code."""
    print(f"\n{'='*80}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*80}\n")

    result = subprocess.run(cmd)
    return result.returncode


def main():
    """Main test runner."""
    args = sys.argv[1:]

    # Default pytest command
    pytest_cmd = ["pytest", "-v"]

    # Parse arguments
    if "--unit" in args:
        pytest_cmd.extend(["tests/unit/"])
        print("Running UNIT tests only")

    elif "--integration" in args:
        pytest_cmd.extend(["tests/integration/"])
        print("Running INTEGRATION tests only")

    elif "--e2e" in args:
        pytest_cmd.extend(["tests/e2e/"])
        print("Running END-TO-END tests only")

    elif "--fast" in args:
        pytest_cmd.extend(["-m", "not slow"])
        print("Running FAST tests only (excluding slow tests)")

    elif "--benchmark" in args:
        pytest_cmd.extend(["-m", "performance", "--benchmark-only"])
        print("Running PERFORMANCE BENCHMARKS only")

    elif "--compliance" in args:
        pytest_cmd.extend(["-m", "compliance"])
        print("Running COMPLIANCE tests only")

    elif "--asme" in args:
        pytest_cmd.extend(["-m", "asme_ptc"])
        print("Running ASME PTC 4.1 compliance tests only")

    elif "--iso" in args:
        pytest_cmd.extend(["-m", "iso_50001"])
        print("Running ISO 50001 compliance tests only")

    elif "--epa" in args:
        pytest_cmd.extend(["-m", "epa_cems"])
        print("Running EPA CEMS compliance tests only")

    else:
        pytest_cmd.extend(["tests/"])
        print("Running ALL tests")

    # Add coverage if requested
    if "--coverage" in args or "--cov" in args:
        pytest_cmd.extend([
            "--cov=src",
            "--cov-report=html:tests/coverage_html",
            "--cov-report=term-missing",
            "--cov-report=xml:tests/coverage.xml",
            "--cov-fail-under=85",
        ])
        print("Coverage reporting ENABLED (minimum 85%)")

    # Add parallel execution if requested
    if "--parallel" in args or "-n" in args:
        pytest_cmd.extend(["-n", "auto"])
        print("Parallel execution ENABLED")

    # Add HTML report if requested
    if "--html" in args:
        pytest_cmd.extend([
            "--html=tests/report.html",
            "--self-contained-html"
        ])
        print("HTML report will be generated")

    # Add JUnit XML report if requested
    if "--xml" in args or "--junit" in args:
        pytest_cmd.extend(["--junitxml=tests/report.xml"])
        print("JUnit XML report will be generated")

    # Run tests
    exit_code = run_command(pytest_cmd)

    # Print summary
    print(f"\n{'='*80}")
    if exit_code == 0:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print(f"{'='*80}\n")

    # Open coverage report if generated and tests passed
    if ("--coverage" in args or "--cov" in args) and exit_code == 0:
        coverage_html = Path("tests/coverage_html/index.html")
        if coverage_html.exists():
            print(f"\n📊 Coverage report generated: {coverage_html.absolute()}")
            print("Open in browser to view detailed coverage")

    # Open HTML report if generated
    if "--html" in args:
        html_report = Path("tests/report.html")
        if html_report.exists():
            print(f"\n📄 HTML report generated: {html_report.absolute()}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
