#!/usr/bin/env python
"""
Property-Based Test Runner for GL-003 UnifiedSteam

This script provides convenient execution of property-based tests with
different Hypothesis profiles.

Usage:
    python run_property_tests.py [profile] [test_file]

Profiles:
    ci        - CI/CD pipeline (200 examples)
    dev       - Local development (50 examples)
    full      - Comprehensive testing (1000 examples)
    debug     - Debug failures (10 examples, verbose)
    quick     - Smoke tests (10 examples)
    exhaustive - Release testing (5000 examples)

Examples:
    python run_property_tests.py
    python run_property_tests.py ci
    python run_property_tests.py full test_thermodynamics_properties.py
    python run_property_tests.py debug test_input_validation.py::TestTemperatureFuzzing

Author: GL-TestEngineer
Version: 1.0.0
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime
from pathlib import Path


# Test directory
TEST_DIR = Path(__file__).parent


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run property-based tests for GL-003 UnifiedSteam",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "profile",
        nargs="?",
        default="dev",
        choices=["ci", "dev", "full", "debug", "quick", "exhaustive"],
        help="Hypothesis profile to use (default: dev)"
    )

    parser.add_argument(
        "test_path",
        nargs="?",
        default="",
        help="Specific test file or test to run (optional)"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )

    parser.add_argument(
        "-x", "--exitfirst",
        action="store_true",
        help="Exit on first failure"
    )

    parser.add_argument(
        "-k", "--keyword",
        help="Run tests matching keyword expression"
    )

    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Run with coverage reporting"
    )

    parser.add_argument(
        "--html-report",
        action="store_true",
        help="Generate HTML test report"
    )

    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run tests in parallel (requires pytest-xdist)"
    )

    return parser.parse_args()


def run_tests(args):
    """Run the property-based tests."""
    # Set environment variable for Hypothesis profile
    os.environ["HYPOTHESIS_PROFILE"] = args.profile

    # Build pytest command
    cmd = ["python", "-m", "pytest"]

    # Test path
    if args.test_path:
        if not args.test_path.startswith("/") and not args.test_path.startswith("\\"):
            test_path = TEST_DIR / args.test_path
        else:
            test_path = Path(args.test_path)
        cmd.append(str(test_path))
    else:
        cmd.append(str(TEST_DIR))

    # Add options
    cmd.extend(["-v", "--tb=short"])

    if args.verbose:
        cmd.append("-vv")

    if args.exitfirst:
        cmd.append("-x")

    if args.keyword:
        cmd.extend(["-k", args.keyword])

    if args.coverage:
        cmd.extend([
            "--cov=../../",
            "--cov-report=term-missing",
            "--cov-report=html:../../coverage_html"
        ])

    if args.html_report:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cmd.extend([
            f"--html=../../test_reports/property_tests_{timestamp}.html",
            "--self-contained-html"
        ])

    if args.parallel:
        cmd.extend(["-n", "auto"])

    # Add markers
    cmd.extend(["-m", "hypothesis"])

    # Print configuration
    print("=" * 70)
    print("GL-003 UnifiedSteam Property-Based Test Runner")
    print("=" * 70)
    print(f"Profile:      {args.profile}")
    print(f"Test path:    {args.test_path or 'all property tests'}")
    print(f"Command:      {' '.join(cmd)}")
    print("=" * 70)
    print()

    # Run tests
    result = subprocess.run(cmd, cwd=str(TEST_DIR.parent.parent))

    return result.returncode


def main():
    """Main entry point."""
    args = parse_args()
    sys.exit(run_tests(args))


if __name__ == "__main__":
    main()
