#!/usr/bin/env python
"""
Test runner script to verify test infrastructure is working.

This script attempts to run a subset of tests to verify the test suite
is properly configured and can execute.
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set test environment
os.environ['GL_ENV'] = 'test'
os.environ['GL_SIGNING_MODE'] = 'ephemeral'
os.environ['GL_TEST_MODE'] = 'true'

# Load test environment if exists
env_test_file = PROJECT_ROOT / '.env.test'
if env_test_file.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(env_test_file)
    except ImportError:
        # dotenv not installed, skip loading .env.test
        pass


def run_basic_tests():
    """Run basic tests to verify infrastructure."""
    import pytest

    print("=" * 60)
    print("GreenLang Test Infrastructure Verification")
    print("=" * 60)
    print()

    # Define test categories to run
    test_categories = [
        ("Unit Tests - Simple", ["tests/test_utils.py::test_"], 5),
        ("Unit Tests - Version", ["tests/test_version.py::test_"], 5),
        ("Unit Tests - Init", ["tests/test_init.py::test_"], 10),
        ("Config Tests", ["tests/conftest.py::test_"], 5),
    ]

    results = []

    for category, patterns, max_tests in test_categories:
        print(f"\n{category}:")
        print("-" * 40)

        for pattern in patterns:
            # Try to run tests matching pattern
            test_path = PROJECT_ROOT / pattern.replace("::", "").replace("test_", "")

            # Check if test file exists
            test_file = pattern.split("::")[0]
            test_file_path = PROJECT_ROOT / test_file

            if not test_file_path.exists():
                print(f"  WARNING {test_file} - NOT FOUND")
                continue

            # Run pytest on the file with minimal output
            try:
                # Run with minimal verbosity and collect only
                result = pytest.main([
                    str(test_file_path),
                    "--co",  # Collect only
                    "-q",    # Quiet
                    "--maxfail=1"
                ])

                if result == 0:
                    print(f"  OK {test_file} - LOADABLE")
                    results.append((category, test_file, "PASS"))
                else:
                    print(f"  FAIL {test_file} - LOAD ERROR")
                    results.append((category, test_file, "FAIL"))

            except Exception as e:
                print(f"  ERROR {test_file} - ERROR: {e}")
                results.append((category, test_file, "ERROR"))

    # Summary
    print("\n" + "=" * 60)
    print("Test Infrastructure Summary")
    print("=" * 60)

    pass_count = sum(1 for _, _, status in results if status == "PASS")
    fail_count = sum(1 for _, _, status in results if status == "FAIL")
    error_count = sum(1 for _, _, status in results if status == "ERROR")

    print(f"\nTotal Files Checked: {len(results)}")
    print(f"  [OK] Loadable: {pass_count}")
    print(f"  [FAIL] Load Errors: {fail_count}")
    print(f"  [ERROR] Other Errors: {error_count}")

    return pass_count > 0 and fail_count == 0 and error_count == 0


def check_dependencies():
    """Check if required test dependencies are installed."""
    print("\nChecking Test Dependencies:")
    print("-" * 40)

    required_packages = [
        "pytest",
        "pytest-cov",
        "pytest-asyncio",
        "pytest-timeout",
    ]

    missing = []

    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"  [OK] {package} - INSTALLED")
        except ImportError:
            print(f"  [MISSING] {package} - MISSING")
            missing.append(package)

    if missing:
        print(f"\n[WARNING] Missing packages: {', '.join(missing)}")
        print(f"   Install with: pip install -r requirements-test.txt")
        return False

    return True


def main():
    """Main entry point."""
    print("\nGreenLang Test Infrastructure Checker\n")

    # Check dependencies first
    if not check_dependencies():
        print("\n[ERROR] Missing dependencies. Please install test requirements.")
        sys.exit(1)

    # Run basic tests
    success = run_basic_tests()

    if success:
        print("\n[SUCCESS] Test infrastructure is working!")
        print("\nNext steps:")
        print("  1. Run full test suite: pytest tests/")
        print("  2. Run with coverage: pytest --cov=greenlang tests/")
        print("  3. Run specific test: pytest tests/test_utils.py")
        sys.exit(0)
    else:
        print("\n[WARNING] Some tests could not be loaded.")
        print("\nDebugging steps:")
        print("  1. Check error messages above")
        print("  2. Verify module imports in failing files")
        print("  3. Check conftest.py fixtures")
        print("  4. Run pytest with -v for verbose output")
        sys.exit(1)


if __name__ == "__main__":
    main()