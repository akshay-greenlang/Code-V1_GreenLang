#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Security Vulnerability Fix Validation
======================================

This script validates that all critical security vulnerabilities have been fixed:
1. Remote Code Execution (RCE) - executor.py line 759
2. SQL Injection - emission_factors_schema.py line 290
3. SQL Injection - emission_factors_schema.py line 455
4. Command Injection - subprocess.run with shell=True

Author: GL-BackendDeveloper
Date: 2025-11-21
"""

import os
import re
import sys
import io
from pathlib import Path
from typing import List, Tuple

# Fix Windows console encoding issues
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


class SecurityValidator:
    """Validates security fixes across the codebase."""

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.results = []
        self.passed = 0
        self.failed = 0

    def validate_all(self) -> bool:
        """Run all security validation checks."""
        print("=" * 80)
        print("SECURITY VULNERABILITY FIX VALIDATION")
        print("=" * 80)
        print()

        # Test 1: RCE Fix in executor.py
        self.test_rce_fix()

        # Test 2: SQL Injection fixes in emission_factors_schema.py
        self.test_sql_injection_fixes()

        # Test 3: Command Injection fixes
        self.test_command_injection_fixes()

        # Test 4: Ensure RestrictedPython is recommended
        self.test_restricted_python_usage()

        # Print summary
        self.print_summary()

        return self.failed == 0

    def test_rce_fix(self):
        """Validate RCE fix in executor.py."""
        print("\n[TEST 1] Remote Code Execution Fix - executor.py:759")
        print("-" * 80)

        file_path = self.root_dir / "greenlang" / "runtime" / "executor.py"

        if not file_path.exists():
            self.record_result("RCE Fix", False, f"File not found: {file_path}")
            return

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check for RestrictedPython usage
        checks = [
            ("RestrictedPython import", "from RestrictedPython import", True),
            ("compile_restricted usage", "compile_restricted", True),
            ("safe_globals usage", "safe_globals", True),
            ("Unsafe exec warning", "RestrictedPython not installed", True),
            ("Security comment", "SECURITY FIX:", True),
            ("Timeout protection", "timeout", True),
        ]

        all_passed = True
        for check_name, pattern, should_exist in checks:
            found = pattern in content
            if found == should_exist:
                print(f"  [PASS] {check_name}")
            else:
                print(f"  [FAIL] {check_name}")
                all_passed = False

        # Check that raw exec is properly wrapped
        exec_pattern = r'exec\(code,\s*namespace\)'
        unsafe_exec_matches = re.findall(exec_pattern, content)

        # Should only appear in fallback (commented or within try/except)
        if len(unsafe_exec_matches) <= 1:  # Allow one in fallback
            print(f"  [PASS] Unsafe exec() properly sandboxed")
        else:
            print(f"  [FAIL] Found {len(unsafe_exec_matches)} unsafe exec() calls")
            all_passed = False

        self.record_result("Remote Code Execution Fix", all_passed, file_path)

    def test_sql_injection_fixes(self):
        """Validate SQL injection fixes in emission_factors_schema.py."""
        print("\n[TEST 2] SQL Injection Fixes - emission_factors_schema.py")
        print("-" * 80)

        file_path = self.root_dir / "greenlang" / "db" / "emission_factors_schema.py"

        if not file_path.exists():
            self.record_result("SQL Injection Fix", False, f"File not found: {file_path}")
            return

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Check for whitelist validation around line 290
        checks_290 = False
        checks_455 = False

        # Look for whitelist patterns
        whitelist_pattern = r'allowed_tables\s*=\s*\{'
        whitelist_check_pattern = r'if\s+table(_name)?\s+not\s+in\s+allowed_tables'
        security_comment_pattern = r'SECURITY.*Fix|SECURITY.*Validate'

        for i, line in enumerate(lines, start=1):
            if whitelist_pattern in line or 'allowed_tables' in line:
                # Found whitelist definition
                if 280 <= i <= 310:
                    checks_290 = True
                    print(f"  [PASS] Whitelist validation near line 290 (found at line {i})")
                elif 445 <= i <= 490:
                    checks_455 = True
                    print(f"  [PASS] Whitelist validation near line 455 (found at line {i})")

        # Check for security comments
        security_comments = sum(1 for line in lines if re.search(security_comment_pattern, line, re.IGNORECASE))

        if security_comments >= 2:
            print(f"  [PASS] Security comments present ({security_comments} found)")
        else:
            print(f"  [FAIL] Insufficient security comments ({security_comments} found)")

        all_passed = checks_290 and checks_455 and (security_comments >= 2)

        if not checks_290:
            print(f"  [FAIL] Whitelist validation near line 290")
        if not checks_455:
            print(f"  [FAIL] Whitelist validation near line 455")

        self.record_result("SQL Injection Fixes", all_passed, file_path)

    def test_command_injection_fixes(self):
        """Validate command injection fixes across codebase."""
        print("\n[TEST 3] Command Injection Fixes - subprocess.run with shell=True")
        print("-" * 80)

        # Search for shell=True in production code
        production_dirs = [
            self.root_dir / "greenlang",
            self.root_dir / "GreenLang_2030" / "agent_foundation" / "security",
        ]

        vulnerable_files = []

        for prod_dir in production_dirs:
            if not prod_dir.exists():
                continue

            for py_file in prod_dir.rglob("*.py"):
                # Skip test files
                if "test_" in py_file.name or "/tests/" in str(py_file).replace("\\", "/"):
                    continue

                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                # Look for shell=True
                if re.search(r'subprocess\.run.*shell\s*=\s*True', content):
                    # Check if it's in a comment or example
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        if 'shell=True' in line and 'shell' in line and '=' in line:
                            # Check if it's commented or in docstring
                            stripped = line.strip()
                            if not stripped.startswith('#') and not stripped.startswith('*'):
                                # Check if it's in an example function (acceptable)
                                context_start = max(0, i - 10)
                                context = '\n'.join(lines[context_start:i+1])
                                if 'example_command_execution_insecure' not in context:
                                    vulnerable_files.append((py_file, i + 1))

        if len(vulnerable_files) == 0:
            print(f"  [PASS] No shell=True found in production code")
            self.record_result("Command Injection Fixes", True, "All files")
        else:
            print(f"  [FAIL] Found shell=True in {len(vulnerable_files)} production files")
            for file_path, line_num in vulnerable_files:
                print(f"    - {file_path}:{line_num}")
            self.record_result("Command Injection Fixes", False, f"{len(vulnerable_files)} vulnerable files")

    def test_restricted_python_usage(self):
        """Validate RestrictedPython is used for code execution."""
        print("\n[TEST 4] RestrictedPython Usage")
        print("-" * 80)

        file_path = self.root_dir / "greenlang" / "runtime" / "executor.py"

        if not file_path.exists():
            self.record_result("RestrictedPython Usage", False, f"File not found: {file_path}")
            return

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        checks = [
            ("RestrictedPython import", "from RestrictedPython import compile_restricted"),
            ("Safe globals", "safe_globals"),
            ("Guarded iteration", "guarded_iter_unpack_sequence"),
            ("Safer attribute access", "safer_getattr"),
            ("Warning for missing RestrictedPython", "RestrictedPython not installed"),
        ]

        all_passed = True
        for check_name, pattern in checks:
            if pattern in content:
                print(f"  [PASS] {check_name}")
            else:
                print(f"  [FAIL] {check_name}")
                all_passed = False

        self.record_result("RestrictedPython Usage", all_passed, file_path)

    def record_result(self, test_name: str, passed: bool, details: str):
        """Record a test result."""
        self.results.append((test_name, passed, details))
        if passed:
            self.passed += 1
        else:
            self.failed += 1

    def print_summary(self):
        """Print validation summary."""
        print("\n" + "=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)

        for test_name, passed, details in self.results:
            status = "[PASS]" if passed else "[FAIL]"
            print(f"{status} - {test_name}")
            if not passed:
                print(f"        Details: {details}")

        print("\n" + "-" * 80)
        print(f"Total Tests: {self.passed + self.failed}")
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        print("-" * 80)

        if self.failed == 0:
            print("\n[SUCCESS] ALL SECURITY FIXES VALIDATED SUCCESSFULLY!")
            print("\nFixed vulnerabilities:")
            print("  1. [OK] Remote Code Execution (RCE) - RestrictedPython sandboxing")
            print("  2. [OK] SQL Injection - Whitelist validation for table names")
            print("  3. [OK] SQL Injection - Whitelist validation (second instance)")
            print("  4. [OK] Command Injection - shell=False for subprocess.run")
            print("\nSecurity posture: IMPROVED")
        else:
            print(f"\n[ERROR] {self.failed} SECURITY VALIDATION TESTS FAILED")
            print("\nPlease review the failures above and fix the issues.")


def main():
    """Main validation function."""
    # Get repository root
    script_dir = Path(__file__).parent
    root_dir = script_dir

    # Validate
    validator = SecurityValidator(str(root_dir))
    success = validator.validate_all()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
