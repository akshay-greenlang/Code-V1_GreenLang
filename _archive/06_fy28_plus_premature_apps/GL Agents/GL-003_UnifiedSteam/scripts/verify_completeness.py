#!/usr/bin/env python3
"""
GL-003 UNIFIEDSTEAM Completeness Verification Script

Verifies that all required components are present and properly structured.

Author: GL-003 DevOps Team
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Base directory
BASE_DIR = Path(__file__).parent.parent


def check_required_modules() -> Tuple[List[str], List[str]]:
    """Check that all required modules exist."""
    required_modules = [
        # Core
        "thermodynamics/__init__.py",
        "thermodynamics/iapws_if97.py",

        # Calculators
        "calculators/__init__.py",

        # Climate
        "climate/__init__.py",
        "climate/emission_factors.py",
        "climate/m_and_v.py",
        "climate/co2e_calculator.py",
        "climate/climate_reporter.py",
        "climate/savings_attribution.py",

        # MLOps
        "mlops/__init__.py",
        "mlops/model_cards.py",
        "mlops/model_registry.py",
        "mlops/model_monitoring.py",
        "mlops/feature_store.py",
        "mlops/deployment_governance.py",

        # Schemas
        "schemas/__init__.py",
        "schemas/kafka_schemas.py",
        "schemas/avro_schemas.py",
        "schemas/schema_registry.py",

        # API
        "api/__init__.py",
        "api/protos/steam_service.proto",
        "api/protos/generate_stubs.py",

        # Control
        "control/__init__.py",

        # Causal
        "causal/__init__.py",

        # Safety
        "safety/__init__.py",

        # Monitoring
        "monitoring/__init__.py",

        # Explainability
        "explainability/__init__.py",

        # Audit
        "audit/__init__.py",
    ]

    found = []
    missing = []

    for module in required_modules:
        path = BASE_DIR / module
        if path.exists():
            found.append(module)
        else:
            missing.append(module)

    return found, missing


def check_required_tests() -> Tuple[List[str], List[str]]:
    """Check that all required test files exist."""
    required_tests = [
        # Performance tests
        "tests/test_performance/__init__.py",
        "tests/test_performance/test_benchmarks.py",

        # Validation tests
        "tests/test_validation/__init__.py",
        "tests/test_validation/test_if97_reference.py",
        "tests/test_validation/test_golden_period.py",

        # Property tests
        "tests/test_property/__init__.py",
        "tests/test_property/test_thermodynamic_properties.py",
        "tests/test_property/test_schema_properties.py",

        # Unit tests (check existing)
        "tests/unit/__init__.py",

        # Integration tests (check existing)
        "tests/integration/__init__.py",
    ]

    found = []
    missing = []

    for test in required_tests:
        path = BASE_DIR / test
        if path.exists():
            found.append(test)
        else:
            missing.append(test)

    return found, missing


def check_config_files() -> Tuple[List[str], List[str]]:
    """Check that configuration files exist."""
    required_configs = [
        "pyproject.toml",
        "setup.py",
        "Dockerfile",
        "docker-compose.yml",
    ]

    found = []
    missing = []

    for config in required_configs:
        path = BASE_DIR / config
        if path.exists():
            found.append(config)
        else:
            missing.append(config)

    return found, missing


def check_deployment_files() -> Tuple[List[str], List[str]]:
    """Check deployment configuration files."""
    required_deployment = [
        "deployment/kubernetes",
        "deployment/helm",
    ]

    found = []
    missing = []

    for deploy in required_deployment:
        path = BASE_DIR / deploy
        if path.exists():
            found.append(deploy)
        else:
            missing.append(deploy)

    return found, missing


def count_lines_of_code() -> Dict[str, int]:
    """Count lines of code by module."""
    counts = {}

    for py_file in BASE_DIR.rglob("*.py"):
        if "test" in str(py_file).lower() or "__pycache__" in str(py_file):
            continue

        # Get module name
        rel_path = py_file.relative_to(BASE_DIR)
        parts = rel_path.parts
        if len(parts) > 1:
            module = parts[0]
        else:
            module = "root"

        try:
            with open(py_file, "r", encoding="utf-8") as f:
                lines = len(f.readlines())
                counts[module] = counts.get(module, 0) + lines
        except Exception:
            pass

    return counts


def verify_imports() -> List[str]:
    """Verify that key modules can be imported."""
    import_errors = []

    # Add base dir to path
    sys.path.insert(0, str(BASE_DIR))

    modules_to_check = [
        "climate",
        "climate.emission_factors",
        "climate.co2e_calculator",
        "mlops",
        "mlops.model_cards",
        "schemas",
        "schemas.kafka_schemas",
    ]

    for module in modules_to_check:
        try:
            __import__(module)
        except ImportError as e:
            import_errors.append(f"{module}: {e}")
        except Exception as e:
            import_errors.append(f"{module}: {type(e).__name__}: {e}")

    return import_errors


def main():
    """Run all verification checks."""
    print("=" * 70)
    print("GL-003 UNIFIEDSTEAM Completeness Verification")
    print("=" * 70)

    all_passed = True

    # Check modules
    print("\n[1] Checking Required Modules...")
    found, missing = check_required_modules()
    print(f"    Found: {len(found)}")
    print(f"    Missing: {len(missing)}")
    if missing:
        all_passed = False
        for m in missing:
            print(f"      - MISSING: {m}")

    # Check tests
    print("\n[2] Checking Required Tests...")
    found, missing = check_required_tests()
    print(f"    Found: {len(found)}")
    print(f"    Missing: {len(missing)}")
    if missing:
        # Only warn, don't fail for missing tests
        for m in missing:
            print(f"      - MISSING: {m}")

    # Check configs
    print("\n[3] Checking Configuration Files...")
    found, missing = check_config_files()
    print(f"    Found: {len(found)}")
    print(f"    Missing: {len(missing)}")
    if missing:
        all_passed = False
        for m in missing:
            print(f"      - MISSING: {m}")

    # Check deployment
    print("\n[4] Checking Deployment Files...")
    found, missing = check_deployment_files()
    print(f"    Found: {len(found)}")
    print(f"    Missing: {len(missing)}")

    # Count lines
    print("\n[5] Lines of Code by Module...")
    counts = count_lines_of_code()
    total = 0
    for module, lines in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"    {module}: {lines:,} lines")
        total += lines
    print(f"    TOTAL: {total:,} lines")

    # Verify imports
    print("\n[6] Verifying Module Imports...")
    errors = verify_imports()
    if errors:
        print(f"    Import errors: {len(errors)}")
        for err in errors:
            print(f"      - {err}")
    else:
        print("    All imports successful")

    # Summary
    print("\n" + "=" * 70)
    if all_passed:
        print("VERIFICATION PASSED: All required components present")
    else:
        print("VERIFICATION FAILED: Some components missing")
    print("=" * 70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
