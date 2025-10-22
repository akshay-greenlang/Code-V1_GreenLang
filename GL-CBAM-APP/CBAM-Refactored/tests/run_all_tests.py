"""
Test Runner for CBAM Refactored Agents

Runs all test suites and generates comprehensive test report:
- Base agent tests
- Provenance framework tests
- Validation framework tests
- I/O utilities tests

Author: GreenLang CBAM Team
Date: 2025-10-16
"""

import subprocess
import sys
import time
from pathlib import Path


def run_test_suite(test_file, description):
    """Run a single test suite and report results."""
    print(f"\n{'=' * 80}")
    print(f"🧪 Running: {description}")
    print(f"{'=' * 80}")

    start_time = time.time()

    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_file), "-v", "--tb=short", "--color=yes"],
        capture_output=True,
        text=True
    )

    elapsed = time.time() - start_time

    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    status = "✅ PASSED" if result.returncode == 0 else "❌ FAILED"
    print(f"\n{status} - {description} ({elapsed:.2f}s)")

    return {
        "name": description,
        "file": test_file.name,
        "passed": result.returncode == 0,
        "time": elapsed,
        "output": result.stdout
    }


def main():
    """Run all test suites and generate report."""
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "CBAM REFACTORED AGENTS - TEST SUITE" + " " * 27 + "║")
    print("╚" + "=" * 78 + "╝")

    test_dir = Path(__file__).parent

    test_suites = [
        (test_dir / "test_cbam_agents.py", "Base Agent Tests (CBAM Agents)"),
        (test_dir / "test_provenance_framework.py", "Provenance Framework Tests"),
        (test_dir / "test_validation_framework.py", "Validation Framework Tests"),
        (test_dir / "test_io_utilities.py", "I/O Utilities Tests")
    ]

    results = []

    for test_file, description in test_suites:
        if test_file.exists():
            result = run_test_suite(test_file, description)
            results.append(result)
        else:
            print(f"\n⚠️  Test file not found: {test_file}")
            results.append({
                "name": description,
                "file": test_file.name,
                "passed": False,
                "time": 0,
                "output": "File not found"
            })

    # Generate summary
    print("\n\n╔" + "=" * 78 + "╗")
    print("║" + " " * 30 + "TEST SUMMARY" + " " * 36 + "║")
    print("╚" + "=" * 78 + "╝\n")

    total_passed = sum(1 for r in results if r["passed"])
    total_failed = len(results) - total_passed
    total_time = sum(r["time"] for r in results)

    for result in results:
        status = "✅ PASS" if result["passed"] else "❌ FAIL"
        print(f"{status}  {result['name']:<50} ({result['time']:.2f}s)")

    print(f"\n{'─' * 80}")
    print(f"Total Test Suites: {len(results)}")
    print(f"Passed: {total_passed} ✅")
    print(f"Failed: {total_failed} {'❌' if total_failed > 0 else ''}")
    print(f"Total Time: {total_time:.2f}s")
    print(f"{'─' * 80}\n")

    if total_failed == 0:
        print("🎉 ALL TESTS PASSED! Framework validation complete.\n")
        return 0
    else:
        print("⚠️  SOME TESTS FAILED. Review output above for details.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
