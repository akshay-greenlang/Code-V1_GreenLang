#!/usr/bin/env python3
"""
Security Check Script
====================

Runs security checks locally to identify potential security issues
before committing code.
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple


def check_insecure_patterns() -> List[str]:
    """Check for insecure SSL/TLS patterns"""
    issues = []

    patterns = [
        (r'verify\s*=\s*False', 'SSL verification disabled'),
        (r'InsecureRequestWarning', 'Insecure request warning suppressed'),
        (r'ssl\._create_unverified_context', 'Unverified SSL context'),
        (r'_create_default_https_context.*_create_unverified_context', 'Default HTTPS context made insecure'),
        (r'ALLOW_INSECURE', 'Insecure mode flag'),
        (r'disable_warnings.*InsecureRequestWarning', 'Security warnings disabled'),
        (r'check_hostname\s*=\s*False', 'Hostname checking disabled'),
        (r'cert_reqs.*CERT_NONE', 'Certificate requirements disabled'),
    ]

    # Directories to check
    dirs_to_check = ['core/greenlang', 'greenlang']

    for dir_path in dirs_to_check:
        if not Path(dir_path).exists():
            continue

        for root, _, files in os.walk(dir_path):
            for file in files:
                if not file.endswith('.py'):
                    continue

                file_path = Path(root) / file

                # Skip test files
                if 'test_' in file or '__pycache__' in str(file_path):
                    continue

                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')

                    for line_no, line in enumerate(lines, 1):
                        # Skip comments
                        if line.strip().startswith('#'):
                            continue

                        for pattern, desc in patterns:
                            if re.search(pattern, line):
                                issues.append(
                                    f"{file_path}:{line_no} - {desc}: {line.strip()}"
                                )

    return issues


def check_http_urls() -> List[str]:
    """Check for insecure HTTP URLs"""
    issues = []

    # Directories to check
    dirs_to_check = ['core/greenlang', 'greenlang']

    for dir_path in dirs_to_check:
        if not Path(dir_path).exists():
            continue

        for root, _, files in os.walk(dir_path):
            for file in files:
                if not file.endswith('.py'):
                    continue

                file_path = Path(root) / file

                # Skip test files
                if 'test_' in file or '__pycache__' in str(file_path):
                    continue

                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                    for line_no, line in enumerate(lines, 1):
                        # Skip comments and docstrings
                        if line.strip().startswith('#') or '"""' in line or "'''" in line:
                            continue

                        # Check for http:// URLs (excluding localhost)
                        if 'http://' in line:
                            if not any(x in line for x in ['localhost', '127.0.0.1', 'example']):
                                issues.append(
                                    f"{file_path}:{line_no} - Insecure HTTP URL: {line.strip()}"
                                )

    return issues


def check_hardcoded_secrets() -> List[str]:
    """Check for potential hardcoded secrets"""
    issues = []

    patterns = [
        (r'api_key\s*=\s*["\'][a-zA-Z0-9]{20,}', 'Potential API key'),
        (r'secret\s*=\s*["\'][a-zA-Z0-9]{20,}', 'Potential secret'),
        (r'token\s*=\s*["\'][a-zA-Z0-9]{20,}', 'Potential token'),
        (r'password\s*=\s*["\'][^"\']+["\']', 'Potential password'),
        (r'AWS_SECRET', 'AWS secret reference'),
        (r'PRIVATE_KEY.*=', 'Private key assignment'),
    ]

    # Directories to check
    dirs_to_check = ['core/greenlang', 'greenlang']

    for dir_path in dirs_to_check:
        if not Path(dir_path).exists():
            continue

        for root, _, files in os.walk(dir_path):
            for file in files:
                if not file.endswith('.py'):
                    continue

                file_path = Path(root) / file

                # Skip test files
                if 'test_' in file or '__pycache__' in str(file_path):
                    continue

                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')

                    for line_no, line in enumerate(lines, 1):
                        # Skip comments
                        if line.strip().startswith('#'):
                            continue

                        # Skip lines with placeholder/example
                        if any(x in line.lower() for x in ['placeholder', 'example', 'dummy']):
                            continue

                        for pattern, desc in patterns:
                            if re.search(pattern, line, re.IGNORECASE):
                                issues.append(
                                    f"{file_path}:{line_no} - {desc}: {line.strip()[:80]}..."
                                )

    return issues


def check_path_traversal() -> List[str]:
    """Check for potential path traversal vulnerabilities"""
    issues = []

    patterns = [
        (r'extractall\s*\([^)]*\)\s*$', 'Unvalidated extractall'),
        (r'extract\s*\([^)]*\)\s*$', 'Unvalidated extract'),
        (r'os\.path\.join.*\.\.', 'Path join with ..'),
        (r'open\s*\([^)]*\.\.', 'Open with .. in path'),
    ]

    # Directories to check
    dirs_to_check = ['core/greenlang', 'greenlang']

    for dir_path in dirs_to_check:
        if not Path(dir_path).exists():
            continue

        for root, _, files in os.walk(dir_path):
            for file in files:
                if not file.endswith('.py'):
                    continue

                file_path = Path(root) / file

                # Skip test files and security module
                if 'test_' in file or 'security' in str(file_path) or '__pycache__' in str(file_path):
                    continue

                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')

                    for line_no, line in enumerate(lines, 1):
                        # Skip comments
                        if line.strip().startswith('#'):
                            continue

                        for pattern, desc in patterns:
                            if re.search(pattern, line):
                                # Skip if it's using safe_extract
                                if 'safe_extract' not in line:
                                    issues.append(
                                        f"{file_path}:{line_no} - {desc}: {line.strip()}"
                                    )

    return issues


def main():
    """Run all security checks"""
    print("=" * 60)
    print("GreenLang Security Check")
    print("=" * 60)
    print()

    all_issues = []

    # Check insecure patterns
    print("Checking for insecure SSL/TLS patterns...")
    issues = check_insecure_patterns()
    if issues:
        print(f"[X] Found {len(issues)} insecure pattern(s):")
        for issue in issues:
            print(f"  - {issue}")
        all_issues.extend(issues)
    else:
        print("[OK] No insecure patterns found")
    print()

    # Check HTTP URLs
    print("Checking for insecure HTTP URLs...")
    issues = check_http_urls()
    if issues:
        print(f"[X] Found {len(issues)} insecure URL(s):")
        for issue in issues:
            print(f"  - {issue}")
        all_issues.extend(issues)
    else:
        print("[OK] No insecure HTTP URLs found")
    print()

    # Check hardcoded secrets
    print("Checking for hardcoded secrets...")
    issues = check_hardcoded_secrets()
    if issues:
        print(f"[!] Found {len(issues)} potential secret(s):")
        for issue in issues:
            print(f"  - {issue}")
        print("  Please review these and ensure no real secrets are hardcoded")
        # Don't fail for potential secrets, just warn
    else:
        print("[OK] No hardcoded secrets found")
    print()

    # Check path traversal
    print("Checking for path traversal vulnerabilities...")
    issues = check_path_traversal()
    if issues:
        print(f"[!] Found {len(issues)} potential path traversal issue(s):")
        for issue in issues:
            print(f"  - {issue}")
        print("  Please ensure all path operations use safe functions")
        # Don't fail for potential issues, just warn
    else:
        print("[OK] No path traversal vulnerabilities found")
    print()

    # Summary
    print("=" * 60)
    if all_issues:
        print(f"[X] Security check failed with {len(all_issues)} issue(s)")
        print("Please fix the issues before committing")
        sys.exit(1)
    else:
        print("[OK] All security checks passed")
        sys.exit(0)


if __name__ == "__main__":
    main()