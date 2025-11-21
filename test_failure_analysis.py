#!/usr/bin/env python3
"""
Deep analysis of test failures - tries to identify actual failing tests
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
import json

def analyze_test_failures():
    """Analyze actual test failures by running sample tests."""

    print("=" * 80)
    print("GREENLANG TEST FAILURE ANALYSIS")
    print("=" * 80)
    print()

    # Try to run tests without dependencies
    test_samples = [
        # Simple unit tests that should work
        ("greenlang/tests/test_infrastructure.py", "Infrastructure tests"),
        ("tests/test_version.py", "Version module tests"),
        ("tests/test_utils.py", "Utility tests"),
        ("tests/simple_test.py", "Simple smoke test"),

        # Agent tests
        ("tests/test_agents.py", "Agent base tests"),
        ("greenlang/tests/agents/test_base_agent.py", "Base agent tests"),

        # Core functionality
        ("tests/test_calculation.py", "Calculation tests"),
        ("tests/test_orchestrator.py", "Orchestrator tests"),
    ]

    results = []

    for test_file, description in test_samples:
        test_path = Path(f"C:/Users/aksha/Code-V1_GreenLang/{test_file}")

        if not test_path.exists():
            results.append((test_file, description, "FILE_NOT_FOUND", None))
            continue

        # Try to run the test
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", str(test_path), "--tb=no", "-q"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd="C:/Users/aksha/Code-V1_GreenLang"
            )

            if "ModuleNotFoundError" in result.stderr or "ImportError" in result.stderr:
                # Extract the missing module
                lines = result.stderr.split('\n')
                for line in lines:
                    if "ModuleNotFoundError" in line or "ImportError" in line:
                        results.append((test_file, description, "IMPORT_ERROR", line.strip()))
                        break
                else:
                    results.append((test_file, description, "IMPORT_ERROR", "Unknown import error"))
            elif result.returncode == 0:
                # Parse output for pass/fail
                if "passed" in result.stdout:
                    results.append((test_file, description, "PASSED", result.stdout.split('\n')[0]))
                else:
                    results.append((test_file, description, "NO_TESTS", "No tests found or collected"))
            else:
                # Tests failed
                if "failed" in result.stdout:
                    results.append((test_file, description, "FAILED", result.stdout.split('\n')[0]))
                else:
                    results.append((test_file, description, "ERROR", result.stderr.split('\n')[0] if result.stderr else "Unknown error"))

        except subprocess.TimeoutExpired:
            results.append((test_file, description, "TIMEOUT", "Test took too long"))
        except Exception as e:
            results.append((test_file, description, "EXCEPTION", str(e)))

    # Print results
    print("TEST EXECUTION RESULTS:")
    print("-" * 40)

    passed = []
    failed = []
    import_errors = []
    missing = []
    other = []

    for test_file, desc, status, detail in results:
        if status == "PASSED":
            passed.append((test_file, desc, detail))
        elif status == "FAILED":
            failed.append((test_file, desc, detail))
        elif status == "IMPORT_ERROR":
            import_errors.append((test_file, desc, detail))
        elif status == "FILE_NOT_FOUND":
            missing.append((test_file, desc))
        else:
            other.append((test_file, desc, status, detail))

    print(f"\n[PASSED] {len(passed)} tests passed:")
    for test, desc, detail in passed[:3]:
        print(f"  - {test}: {desc}")
        if detail:
            print(f"    {detail}")

    print(f"\n[FAILED] {len(failed)} tests failed:")
    for test, desc, detail in failed[:5]:
        print(f"  - {test}: {desc}")
        if detail:
            print(f"    {detail}")

    print(f"\n[IMPORT_ERROR] {len(import_errors)} tests have import errors:")
    for test, desc, detail in import_errors[:5]:
        print(f"  - {test}: {desc}")
        if detail:
            print(f"    {detail}")

    print(f"\n[MISSING] {len(missing)} test files not found:")
    for test, desc in missing[:5]:
        print(f"  - {test}: {desc}")

    print(f"\n[OTHER] {len(other)} tests had other issues:")
    for test, desc, status, detail in other[:5]:
        print(f"  - {test} ({status}): {desc}")
        if detail:
            print(f"    {detail}")

    print()
    print("=" * 80)
    print("MISSING TEST COVERAGE ANALYSIS")
    print("-" * 40)

    # Check for missing test coverage
    critical_modules = [
        ("greenlang/agents/", "Agent implementations"),
        ("greenlang/calculation/", "Calculation engine"),
        ("greenlang/connectors/", "ERP/Database connectors"),
        ("greenlang/auth/", "Authentication/Authorization"),
        ("greenlang/api/", "API endpoints"),
        ("greenlang/provenance/", "Provenance tracking"),
        ("greenlang/cache/", "Caching system"),
        ("greenlang/core/", "Core functionality"),
        ("GL-CBAM-APP/CBAM-Importer-Copilot/agents/", "CBAM agents"),
        ("GL-CSRD-APP/CSRD-Reporting-Platform/agents/", "CSRD agents"),
        ("GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/agents/", "VCCI agents"),
    ]

    coverage_gaps = []

    for module_path, description in critical_modules:
        full_path = Path(f"C:/Users/aksha/Code-V1_GreenLang/{module_path}")
        if full_path.exists():
            # Count source files
            source_files = list(full_path.rglob("*.py"))
            source_count = len([f for f in source_files if not f.name.startswith("test_")])

            # Count test files
            test_count = len([f for f in source_files if f.name.startswith("test_")])

            # Check for tests directory
            test_dir = full_path.parent / "tests"
            if test_dir.exists():
                test_count += len(list(test_dir.rglob("test_*.py")))

            if source_count > 0:
                coverage_ratio = (test_count / source_count) * 100
                if coverage_ratio < 50:
                    coverage_gaps.append((module_path, description, source_count, test_count, coverage_ratio))

    print("MODULES WITH INSUFFICIENT TEST COVERAGE (<50%):")
    for module, desc, src_count, test_count, ratio in sorted(coverage_gaps, key=lambda x: x[4]):
        print(f"  [{ratio:5.1f}%] {module}")
        print(f"          {desc}: {test_count}/{src_count} files tested")

    print()
    print("=" * 80)
    print("CRITICAL MISSING TESTS")
    print("-" * 40)

    # Identify critical missing tests
    critical_missing = [
        "greenlang/infrastructure/ - Module doesn't exist, breaking all infrastructure tests",
        "Emission factor tests - Need pandas/numpy dependencies",
        "Agent pipeline integration tests - Missing dependencies",
        "Provenance tracking tests - Core functionality untested",
        "Calculation engine tests - Mathematical accuracy validation",
        "Security tests - Authentication/authorization untested",
        "Performance benchmarks - No load testing available",
        "E2E workflow tests - Complete user scenarios untested",
    ]

    print("CRITICAL TESTING GAPS:")
    for gap in critical_missing:
        print(f"  - {gap}")

    print()
    print("=" * 80)
    print("RECOMMENDATIONS FOR IMMEDIATE ACTION")
    print("-" * 40)

    print("""
1. INSTALL DEPENDENCIES (Critical):
   pip install pytest-cov pytest-asyncio pytest-timeout hypothesis pandas numpy fastapi httpx

2. CREATE MISSING INFRASTRUCTURE:
   - Create greenlang/infrastructure/__init__.py
   - Add base classes: ValidationFramework, CacheManager, TelemetryCollector, ProvenanceTracker

3. FIX BROKEN IMPORTS:
   - Update test files to use existing module paths
   - Remove references to non-existent modules

4. PRIORITY TEST FIXES:
   a) Fix simple unit tests first (test_utils.py, test_version.py)
   b) Fix agent base tests
   c) Fix calculation tests
   d) Fix pipeline integration tests

5. ADD MISSING CRITICAL TESTS:
   - Agent initialization and configuration
   - Provenance hash generation
   - Calculation accuracy validation
   - Error handling and edge cases
   - Performance benchmarks
""")

if __name__ == "__main__":
    analyze_test_failures()