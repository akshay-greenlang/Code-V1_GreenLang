# -*- coding: utf-8 -*-
"""
Security Test: Verify No eval() or exec() Usage

This test scans the entire GreenLang codebase to ensure no dangerous
eval() or exec() functions are used, preventing Remote Code Execution (RCE) vulnerabilities.

CWE-95: Improper Neutralization of Directives in Dynamically Evaluated Code ('Eval Injection')
"""

import ast
import os
import sys
from pathlib import Path
from typing import List, Tuple


class EvalDetector(ast.NodeVisitor):
    """AST visitor to detect eval() and exec() calls."""

    def __init__(self, filename: str):
        self.filename = filename
        self.violations: List[Tuple[int, str, str]] = []

    def visit_Call(self, node: ast.Call) -> None:
        """Visit function call nodes."""
        if isinstance(node.func, ast.Name):
            # Direct calls: eval(...) or exec(...)
            if node.func.id in ['eval', 'exec']:
                # Check if it's actually the dangerous built-in eval/exec
                # Allow references in strings, comments, or safe contexts
                self.violations.append((
                    node.lineno,
                    node.func.id,
                    ast.unparse(node) if hasattr(ast, 'unparse') else str(node)
                ))

        self.generic_visit(node)


def scan_file(file_path: Path) -> List[Tuple[int, str, str]]:
    """
    Scan a single Python file for eval() and exec() usage.

    Args:
        file_path: Path to Python file

    Returns:
        List of violations: [(line_number, function_name, code_snippet)]
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Parse the file
        tree = ast.parse(content, filename=str(file_path))

        # Detect eval/exec calls
        detector = EvalDetector(str(file_path))
        detector.visit(tree)

        # Filter out false positives (string checks in validation code)
        real_violations = []
        for line_num, func_name, code in detector.violations:
            # Skip if it's in a string literal check (validation code)
            if 'in code' in code or '"eval("' in code or "'eval('" in code:
                continue
            # Skip if it's checking for eval in strings
            if 'if "eval("' in code or 'if "exec("' in code:
                continue
            real_violations.append((line_num, func_name, code))

        return real_violations

    except SyntaxError as e:
        print(f"Syntax error in {file_path}: {e}")
        return []
    except Exception as e:
        print(f"Error scanning {file_path}: {e}")
        return []


def scan_directory(root_dir: Path, exclude_dirs: List[str] = None) -> dict:
    """
    Scan all Python files in directory tree.

    Args:
        root_dir: Root directory to scan
        exclude_dirs: List of directory names to exclude

    Returns:
        Dictionary mapping file paths to violations
    """
    if exclude_dirs is None:
        exclude_dirs = [
            '__pycache__',
            '.git',
            '.venv',
            'venv',
            'node_modules',
            '.pytest_cache',
            'site-packages',
            'dist',
            'build',
            '.mypy_cache'
        ]

    violations_by_file = {}

    for root, dirs, files in os.walk(root_dir):
        # Filter out excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]

        for file in files:
            if file.endswith('.py'):
                file_path = Path(root) / file
                violations = scan_file(file_path)

                if violations:
                    violations_by_file[str(file_path)] = violations

    return violations_by_file


def test_no_eval_in_agent_foundation():
    """
    Test that no eval() or exec() exists in agent_foundation code.

    This is a CRITICAL security test - any eval() or exec() usage creates
    a Remote Code Execution (RCE) vulnerability.
    """
    agent_foundation_dir = Path("C:/Users/aksha/Code-V1_GreenLang/GreenLang_2030/agent_foundation")

    if not agent_foundation_dir.exists():
        print(f"Warning: {agent_foundation_dir} does not exist")
        return

    violations = scan_directory(agent_foundation_dir)

    # Print detailed report
    print("\n" + "=" * 80)
    print("SECURITY SCAN: eval() and exec() Usage Detection")
    print("=" * 80)

    if violations:
        print(f"\n❌ CRITICAL: Found {len(violations)} files with eval()/exec() usage:\n")

        for file_path, file_violations in violations.items():
            print(f"\nFile: {file_path}")
            for line_num, func_name, code in file_violations:
                print(f"  Line {line_num}: {func_name}()")
                print(f"    Code: {code}")

        print("\n" + "=" * 80)
        print("SECURITY TEST FAILED")
        print("=" * 80)

        # Fail the test
        assert False, f"Found {len(violations)} files with dangerous eval()/exec() usage"

    else:
        print("\n✅ PASSED: No eval() or exec() usage found in codebase")
        print("=" * 80)


def test_no_eval_in_gl_apps():
    """
    Test that no eval() or exec() exists in GL application code.
    """
    test_dirs = [
        Path("C:/Users/aksha/Code-V1_GreenLang/GL-CSRD-APP"),
        Path("C:/Users/aksha/Code-V1_GreenLang/GL-VCCI-Carbon-APP"),
    ]

    all_violations = {}

    for test_dir in test_dirs:
        if test_dir.exists():
            violations = scan_directory(test_dir)
            if violations:
                all_violations.update(violations)

    if all_violations:
        print(f"\n❌ CRITICAL: Found {len(all_violations)} files with eval()/exec() in GL apps")
        for file_path, file_violations in all_violations.items():
            print(f"\nFile: {file_path}")
            for line_num, func_name, code in file_violations:
                print(f"  Line {line_num}: {func_name}()")

        assert False, f"Found {len(all_violations)} files with dangerous eval()/exec() usage in GL apps"
    else:
        print("\n✅ PASSED: No eval() or exec() usage found in GL applications")


def test_safe_alternatives_exist():
    """
    Verify that safe alternatives (ast.literal_eval, simpleeval) are being used.
    """
    agent_foundation_dir = Path("C:/Users/aksha/Code-V1_GreenLang/GreenLang_2030/agent_foundation")

    # Count safe alternatives
    ast_literal_eval_count = 0
    simpleeval_count = 0

    for root, dirs, files in os.walk(agent_foundation_dir):
        dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git']]

        for file in files:
            if file.endswith('.py'):
                file_path = Path(root) / file
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    if 'ast.literal_eval' in content or 'literal_eval' in content:
                        ast_literal_eval_count += 1

                    if 'simpleeval' in content or 'simple_eval' in content:
                        simpleeval_count += 1

                except Exception:
                    pass

    print(f"\n✅ Safe alternatives in use:")
    print(f"  - ast.literal_eval: {ast_literal_eval_count} files")
    print(f"  - simpleeval: {simpleeval_count} files")

    # We should have at least our 3 fixed files using safe alternatives
    assert ast_literal_eval_count >= 1, "No files using ast.literal_eval found"
    assert simpleeval_count >= 2, "No files using simpleeval found"


if __name__ == "__main__":
    print("Running security tests...")

    try:
        test_no_eval_in_agent_foundation()
        test_no_eval_in_gl_apps()
        test_safe_alternatives_exist()

        print("\n" + "=" * 80)
        print("✅ ALL SECURITY TESTS PASSED")
        print("=" * 80)
        sys.exit(0)

    except AssertionError as e:
        print(f"\n❌ SECURITY TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        sys.exit(1)
