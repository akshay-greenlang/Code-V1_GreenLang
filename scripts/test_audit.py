#!/usr/bin/env python3
"""
GreenLang Test Suite Audit Script
Identifies failing tests, missing coverage, and broken fixtures
"""

import os
import sys
import json
from pathlib import Path
import subprocess
from typing import Dict, List, Tuple

def find_test_files(root_dir: Path) -> Dict[str, List[Path]]:
    """Find all test files organized by application."""
    test_files = {
        'root': [],
        'greenlang': [],
        'GL-CBAM-APP': [],
        'GL-CSRD-APP': [],
        'GL-VCCI-Carbon-APP': [],
        'other': []
    }

    for path in root_dir.rglob('test_*.py'):
        rel_path = path.relative_to(root_dir)
        path_str = str(rel_path)

        if path_str.startswith('GL-CBAM-APP'):
            test_files['GL-CBAM-APP'].append(path)
        elif path_str.startswith('GL-CSRD-APP'):
            test_files['GL-CSRD-APP'].append(path)
        elif path_str.startswith('GL-VCCI-Carbon-APP'):
            test_files['GL-VCCI-Carbon-APP'].append(path)
        elif path_str.startswith('greenlang'):
            test_files['greenlang'].append(path)
        elif path_str.startswith('tests'):
            test_files['root'].append(path)
        else:
            test_files['other'].append(path)

    return test_files

def check_imports(file_path: Path) -> List[str]:
    """Check what modules a test file imports."""
    imports = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip().startswith('import ') or line.strip().startswith('from '):
                    imports.append(line.strip())
    except Exception as e:
        pass
    return imports

def analyze_test_structure(root_dir: Path):
    """Analyze the overall test structure."""

    print("=" * 80)
    print("GREENLANG TEST SUITE AUDIT REPORT")
    print("=" * 80)
    print()

    # Find all test files
    test_files = find_test_files(root_dir)

    # Report on test file counts
    print("TEST FILE DISTRIBUTION:")
    print("-" * 40)
    for category, files in test_files.items():
        print(f"{category:20s}: {len(files):4d} test files")
    print(f"{'TOTAL':20s}: {sum(len(f) for f in test_files.values()):4d} test files")
    print()

    # Check for conftest files
    print("CONFTEST FILES:")
    print("-" * 40)
    conftest_files = list(root_dir.rglob('conftest.py'))
    for conf in conftest_files[:10]:  # Limit to first 10
        rel_path = conf.relative_to(root_dir)
        print(f"  - {rel_path}")
    if len(conftest_files) > 10:
        print(f"  ... and {len(conftest_files) - 10} more")
    print()

    # Check for missing dependencies
    print("DEPENDENCY ANALYSIS:")
    print("-" * 40)

    # Required testing packages
    test_deps = [
        'pytest', 'pytest-cov', 'pytest-asyncio', 'pytest-timeout',
        'hypothesis', 'pandas', 'numpy', 'fastapi', 'httpx'
    ]

    missing_deps = []
    for dep in test_deps:
        try:
            __import__(dep.replace('-', '_'))
            print(f"  [OK] {dep:20s} INSTALLED")
        except ImportError:
            missing_deps.append(dep)
            print(f"  [MISSING] {dep:20s} MISSING")
    print()

    # Identify critical test categories
    print("CRITICAL TEST CATEGORIES:")
    print("-" * 40)

    test_categories = {
        'agents': [],
        'integration': [],
        'pipeline': [],
        'provenance': [],
        'calculation': [],
        'emission': [],
        'security': [],
        'performance': [],
        'e2e': [],
        'cli': []
    }

    for category, files in test_files.items():
        for file in files:
            file_name = file.name.lower()
            for test_cat, cat_files in test_categories.items():
                if test_cat in file_name:
                    cat_files.append((category, file))

    for test_cat, files in test_categories.items():
        print(f"{test_cat:15s}: {len(files):3d} test files")
    print()

    # Check for test infrastructure problems
    print("TEST INFRASTRUCTURE ISSUES:")
    print("-" * 40)

    # Check if greenlang.infrastructure exists
    infra_path = root_dir / 'greenlang' / 'infrastructure'
    if not infra_path.exists():
        print("  [X] greenlang.infrastructure module MISSING")
        print("    This breaks: test_infrastructure.py tests")
    else:
        print("  [OK] greenlang.infrastructure module exists")

    # Check for agent base classes
    agent_base_files = [
        root_dir / 'greenlang' / 'agents' / 'base.py',
        root_dir / 'greenlang' / 'core' / 'agent.py',
        root_dir / 'greenlang_core' / 'agent.py'
    ]

    agent_base_found = False
    for agent_file in agent_base_files:
        if agent_file.exists():
            print(f"  [OK] Agent base class found: {agent_file.relative_to(root_dir)}")
            agent_base_found = True
            break

    if not agent_base_found:
        print("  [X] No agent base class found")
        print("    Expected locations:")
        for f in agent_base_files:
            print(f"      - {f.relative_to(root_dir)}")
    print()

    # Identify broken imports
    print("IMPORT ANALYSIS (Sample):")
    print("-" * 40)

    sample_tests = []
    for cat, files in test_files.items():
        if files and cat != 'other':
            sample_tests.append((cat, files[0]))

    for category, test_file in sample_tests:
        imports = check_imports(test_file)
        rel_path = test_file.relative_to(root_dir)
        print(f"\n  {category}/{test_file.name}:")

        # Check for problematic imports
        for imp in imports[:5]:  # First 5 imports
            if 'greenlang.infrastructure' in imp:
                print(f"    [X] {imp} (MISSING MODULE)")
            elif 'greenlang_core' in imp:
                print(f"    ? {imp} (CHECK IF EXISTS)")
            else:
                print(f"    - {imp}")
    print()

    # Summary of issues
    print("=" * 80)
    print("CRITICAL ISSUES SUMMARY:")
    print("-" * 40)

    issues = []

    if missing_deps:
        issues.append(f"Missing {len(missing_deps)} critical dependencies: {', '.join(missing_deps)}")

    if not infra_path.exists():
        issues.append("greenlang.infrastructure module is missing")

    if not agent_base_found:
        issues.append("No agent base class found in expected locations")

    # Check for pytest.ini
    pytest_ini = root_dir / 'pytest.ini'
    if not pytest_ini.exists():
        issues.append("pytest.ini configuration file missing")

    if issues:
        print("CRITICAL ISSUES FOUND:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("  No critical infrastructure issues found")
    print()

    # Recommendations
    print("RECOMMENDATIONS:")
    print("-" * 40)
    print("1. Install missing dependencies:")
    if missing_deps:
        print(f"   pip install {' '.join(missing_deps)}")
    else:
        print("   All critical dependencies installed")
    print()
    print("2. Fix module structure:")
    print("   - Create greenlang/infrastructure/ directory with __init__.py")
    print("   - Implement base classes: validation.py, cache.py, telemetry.py, provenance.py")
    print()
    print("3. Update imports in test files to match actual module structure")
    print()
    print("4. Create or fix pytest.ini configuration")
    print()

    # Test coverage estimate
    print("ESTIMATED TEST COVERAGE:")
    print("-" * 40)

    # Count source files
    source_files = {
        'greenlang': len(list((root_dir / 'greenlang').rglob('*.py'))) if (root_dir / 'greenlang').exists() else 0,
        'GL-CBAM-APP': len(list((root_dir / 'GL-CBAM-APP').rglob('*.py'))) if (root_dir / 'GL-CBAM-APP').exists() else 0,
        'GL-CSRD-APP': len(list((root_dir / 'GL-CSRD-APP').rglob('*.py'))) if (root_dir / 'GL-CSRD-APP').exists() else 0,
        'GL-VCCI-Carbon-APP': len(list((root_dir / 'GL-VCCI-Carbon-APP').rglob('*.py'))) if (root_dir / 'GL-VCCI-Carbon-APP').exists() else 0,
    }

    for module, src_count in source_files.items():
        test_count = len(test_files.get(module, []))
        if src_count > 0:
            ratio = (test_count / src_count) * 100
            status = "[OK]" if ratio > 30 else "[WARN]" if ratio > 10 else "[X]"
            print(f"  {status} {module:20s}: {test_count:4d} tests / {src_count:4d} source files ({ratio:.1f}%)")

    print()
    print("=" * 80)
    print("END OF AUDIT REPORT")
    print("=" * 80)

if __name__ == "__main__":
    root = Path("C:/Users/aksha/Code-V1_GreenLang")
    analyze_test_structure(root)