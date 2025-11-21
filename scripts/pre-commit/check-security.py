#!/usr/bin/env python3
"""
Custom Security Checker for GreenLang

Detects hardcoded secrets, API keys, passwords, and security vulnerabilities
that may not be caught by standard tools.
"""

import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple


class SecurityChecker:
    """Custom security checker for Python files."""

    # Patterns for common security issues
    PATTERNS = {
        "hardcoded_password": [
            r'password\s*=\s*["\'](?!<|{|\$|%|\*|test|example|password|changeme)[^"\']{8,}["\']',
            r'passwd\s*=\s*["\'](?!<|{|\$|%|\*|test|example)[^"\']{8,}["\']',
            r'pwd\s*=\s*["\'](?!<|{|\$|%|\*|test|example)[^"\']{8,}["\']',
        ],
        "api_key": [
            r'api[_-]?key\s*=\s*["\'][a-zA-Z0-9]{20,}["\']',
            r'apikey\s*=\s*["\'][a-zA-Z0-9]{20,}["\']',
            r'api[_-]?secret\s*=\s*["\'][a-zA-Z0-9]{20,}["\']',
        ],
        "access_token": [
            r'access[_-]?token\s*=\s*["\'][a-zA-Z0-9]{20,}["\']',
            r'auth[_-]?token\s*=\s*["\'][a-zA-Z0-9]{20,}["\']',
            r'bearer\s*=\s*["\'][a-zA-Z0-9]{20,}["\']',
        ],
        "aws_credentials": [
            r'aws[_-]?access[_-]?key[_-]?id\s*=\s*["\']AKIA[A-Z0-9]{16}["\']',
            r'aws[_-]?secret[_-]?access[_-]?key\s*=\s*["\'][a-zA-Z0-9/+]{40}["\']',
        ],
        "private_key": [
            r"-----BEGIN (?:RSA |EC |DSA )?PRIVATE KEY-----",
            r"-----BEGIN OPENSSH PRIVATE KEY-----",
        ],
        "database_url": [
            r'postgresql://[^:]+:[^@]+@[^/]+/\w+',
            r'mysql://[^:]+:[^@]+@[^/]+/\w+',
            r'mongodb://[^:]+:[^@]+@[^/]+/\w+',
        ],
        "hardcoded_ip": [
            r'(?<![0-9])(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(?![0-9])',
        ],
        "jwt_token": [
            r'["\']eyJ[a-zA-Z0-9_-]*\.eyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*["\']',
        ],
        "generic_secret": [
            r'secret\s*=\s*["\'](?!<|{|\$|%|\*|test|example|secret)[^"\']{16,}["\']',
            r'client[_-]?secret\s*=\s*["\'][a-zA-Z0-9]{20,}["\']',
        ],
    }

    # Whitelist patterns that should be ignored
    WHITELIST = [
        r"test_.*\.py$",  # Test files
        r"example",  # Example values
        r"<.*>",  # Template placeholders
        r"\${.*}",  # Environment variable references
        r"%\(.*\)s",  # Python format strings
        r"YOUR_.*",  # Placeholder values
        r"REPLACE_.*",  # Placeholder values
    ]

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.issues: List[Tuple[int, str, str]] = []

    def check(self) -> bool:
        """Check the file for security issues."""
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for line_num, line in enumerate(lines, start=1):
                # Skip comments
                if line.strip().startswith("#"):
                    continue

                # Check against whitelist
                if self._is_whitelisted(line):
                    continue

                # Check all patterns
                for issue_type, patterns in self.PATTERNS.items():
                    for pattern in patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            self.issues.append(
                                (line_num, issue_type, line.strip())
                            )

            # Additional checks
            self._check_insecure_functions(lines)
            self._check_hardcoded_urls(lines)

            return len(self.issues) == 0

        except FileNotFoundError:
            print(f"ERROR: File not found: {self.file_path}")
            return False
        except Exception as e:
            print(f"ERROR: Unexpected error checking {self.file_path}: {e}")
            return False

    def _is_whitelisted(self, line: str) -> bool:
        """Check if line matches any whitelist pattern."""
        for pattern in self.WHITELIST:
            if re.search(pattern, line):
                return True
        return False

    def _check_insecure_functions(self, lines: List[str]) -> None:
        """Check for usage of insecure functions."""
        insecure_patterns = {
            "eval": r"\beval\s*\(",
            "exec": r"\bexec\s*\(",
            "pickle.loads": r"pickle\.loads\s*\(",
            "yaml.load": r"yaml\.load\s*\(",  # Should use yaml.safe_load
            "subprocess.shell": r"subprocess\.\w+\([^)]*shell\s*=\s*True",
        }

        for line_num, line in enumerate(lines, start=1):
            for func_name, pattern in insecure_patterns.items():
                if re.search(pattern, line):
                    self.issues.append(
                        (
                            line_num,
                            f"insecure_function_{func_name}",
                            line.strip(),
                        )
                    )

    def _check_hardcoded_urls(self, lines: List[str]) -> None:
        """Check for hardcoded production URLs."""
        production_patterns = [
            r'https?://(?:www\.)?greenlang\.io',
            r'https?://api\.greenlang\.io',
            r'https?://.*\.amazonaws\.com',
            r'https?://.*\.cloudfront\.net',
        ]

        for line_num, line in enumerate(lines, start=1):
            if line.strip().startswith("#"):
                continue

            for pattern in production_patterns:
                if re.search(pattern, line):
                    self.issues.append(
                        (
                            line_num,
                            "hardcoded_production_url",
                            line.strip(),
                        )
                    )

    def print_results(self) -> None:
        """Print security check results."""
        if self.issues:
            print(f"\n{'='*80}")
            print(f"SECURITY ISSUES FOUND: {self.file_path}")
            print(f"{'='*80}\n")

            for line_num, issue_type, line_content in self.issues:
                print(f"Line {line_num}: {issue_type}")
                print(f"  {line_content}")
                print(f"  Recommendation: {self._get_recommendation(issue_type)}\n")

            print(f"{'='*80}")
        else:
            print(f"✓ {self.file_path} - No security issues found")

    def _get_recommendation(self, issue_type: str) -> str:
        """Get recommendation for fixing the issue."""
        recommendations = {
            "hardcoded_password": "Use environment variables or secrets management",
            "api_key": "Store API keys in environment variables or secure vault",
            "access_token": "Use OAuth flow or environment variables",
            "aws_credentials": "Use IAM roles or AWS Secrets Manager",
            "private_key": "Store private keys in secure vault, never in code",
            "database_url": "Use environment variables for database connection strings",
            "hardcoded_ip": "Use configuration files or service discovery",
            "jwt_token": "Generate tokens dynamically, never hardcode",
            "generic_secret": "Use environment variables or secrets management",
            "insecure_function_eval": "Avoid eval(), use ast.literal_eval() or safer alternatives",
            "insecure_function_exec": "Avoid exec(), refactor to use functions",
            "insecure_function_pickle.loads": "Use json or safer serialization",
            "insecure_function_yaml.load": "Use yaml.safe_load() instead",
            "insecure_function_subprocess.shell": "Avoid shell=True, use list arguments",
            "hardcoded_production_url": "Use environment-based configuration",
        }

        return recommendations.get(
            issue_type, "Review security best practices"
        )


def main() -> int:
    """Main entry point for the hook."""
    if len(sys.argv) < 2:
        print("Usage: check-security.py <file.py> [file.py ...]")
        return 1

    all_secure = True
    for file_path in sys.argv[1:]:
        checker = SecurityChecker(file_path)
        is_secure = checker.check()
        checker.print_results()

        if not is_secure:
            all_secure = False

    return 0 if all_secure else 1


if __name__ == "__main__":
    sys.exit(main())
