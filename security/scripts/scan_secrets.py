#!/usr/bin/env python3
"""
GreenLang Secret Scanner

Scans codebase for:
- Hardcoded API keys (OpenAI, Anthropic, AWS, etc.)
- Database credentials
- JWT secrets
- Private keys
- Passwords in code/comments
- .env files in git

Uses multiple detection methods:
- Pattern matching (regex)
- Entropy analysis
- Context-aware detection

Usage:
    python scan_secrets.py
    python scan_secrets.py --scan-all
    python scan_secrets.py --fix

Author: Security & Compliance Audit Team
Date: 2025-11-09
"""

import hashlib
import json
import math
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Secret:
    """Detected secret"""
    type: str  # e.g., "aws_access_key", "anthropic_api_key"
    file: str
    line_number: int
    line: str
    matched_text: str
    entropy: float
    severity: str  # critical, high, medium, low
    confidence: str  # confirmed, likely, possible


class SecretPatterns:
    """Secret detection patterns"""

    PATTERNS = {
        # API Keys
        "openai_api_key": {
            "pattern": r'sk-[A-Za-z0-9]{48}',
            "severity": "critical",
            "description": "OpenAI API Key"
        },
        "anthropic_api_key": {
            "pattern": r'sk-ant-api03-[A-Za-z0-9\-_]{95}',
            "severity": "critical",
            "description": "Anthropic API Key"
        },
        "aws_access_key": {
            "pattern": r'AKIA[0-9A-Z]{16}',
            "severity": "critical",
            "description": "AWS Access Key ID"
        },
        "aws_secret_key": {
            "pattern": r'aws_secret_access_key\s*=\s*[\'"]([A-Za-z0-9/+=]{40})[\'"]',
            "severity": "critical",
            "description": "AWS Secret Access Key"
        },
        "github_token": {
            "pattern": r'ghp_[A-Za-z0-9]{36}',
            "severity": "critical",
            "description": "GitHub Personal Access Token"
        },
        "github_oauth": {
            "pattern": r'gho_[A-Za-z0-9]{36}',
            "severity": "critical",
            "description": "GitHub OAuth Token"
        },

        # Database Credentials
        "postgres_url": {
            "pattern": r'postgresql://[^:]+:[^@]+@[^/]+/\w+',
            "severity": "critical",
            "description": "PostgreSQL Connection String with Password"
        },
        "mysql_url": {
            "pattern": r'mysql://[^:]+:[^@]+@[^/]+/\w+',
            "severity": "critical",
            "description": "MySQL Connection String with Password"
        },
        "mongodb_url": {
            "pattern": r'mongodb://[^:]+:[^@]+@[^/]+',
            "severity": "critical",
            "description": "MongoDB Connection String with Password"
        },

        # JWT/Auth
        "jwt_secret": {
            "pattern": r'(?:jwt_secret|jwt_key|secret_key|secret_token)\s*=\s*[\'"]([A-Za-z0-9+/=]{32,})[\'"]',
            "severity": "critical",
            "description": "JWT Secret Key"
        },

        # Private Keys
        "rsa_private_key": {
            "pattern": r'-----BEGIN RSA PRIVATE KEY-----',
            "severity": "critical",
            "description": "RSA Private Key"
        },
        "private_key": {
            "pattern": r'-----BEGIN PRIVATE KEY-----',
            "severity": "critical",
            "description": "Private Key"
        },
        "openssh_private_key": {
            "pattern": r'-----BEGIN OPENSSH PRIVATE KEY-----',
            "severity": "critical",
            "description": "OpenSSH Private Key"
        },

        # Passwords (various patterns)
        "password_var": {
            "pattern": r'(?:password|passwd|pwd)\s*=\s*[\'"](?!.*[\{\}])[A-Za-z0-9!@#$%^&*]{8,}[\'"]',
            "severity": "high",
            "description": "Hardcoded Password"
        },

        # Twilio
        "twilio_api_key": {
            "pattern": r'SK[a-z0-9]{32}',
            "severity": "critical",
            "description": "Twilio API Key"
        },

        # Slack
        "slack_token": {
            "pattern": r'xox[baprs]-[0-9]{10,12}-[0-9]{10,12}-[A-Za-z0-9]{24}',
            "severity": "high",
            "description": "Slack Token"
        },

        # Generic High Entropy Strings
        "generic_secret": {
            "pattern": r'(?:api_key|apikey|secret|token|auth)\s*=\s*[\'"]([A-Za-z0-9+/=]{32,})[\'"]',
            "severity": "high",
            "description": "Generic Secret/API Key"
        }
    }


class SecretScanner:
    """Scan for secrets in code"""

    EXCLUDED_DIRS = {
        '.git',
        '__pycache__',
        'node_modules',
        '.venv',
        'venv',
        '.pytest_cache',
        '.mypy_cache',
        'dist',
        'build',
        '.egg-info'
    }

    EXCLUDED_FILES = {
        '.pyc',
        '.pyo',
        '.so',
        '.dylib',
        '.dll'
    }

    # Entropy threshold for flagging potential secrets
    ENTROPY_THRESHOLD = 4.5

    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.secrets: List[Secret] = []
        self.scanned_files = 0
        self.false_positives: Set[str] = set()

        # Load false positives list
        self._load_false_positives()

    def _load_false_positives(self):
        """Load known false positives"""
        # These are examples/test data, not real secrets
        self.false_positives = {
            "sk-test-",  # Test keys
            "example",
            "dummy",
            "fake",
            "sample",
            "YOUR_API_KEY",
            "INSERT_KEY_HERE"
        }

    def scan_file(self, file_path: Path) -> List[Secret]:
        """Scan a single file for secrets"""
        secrets = []

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()

            for line_num, line in enumerate(lines, start=1):
                # Skip comments (simple heuristic)
                if line.strip().startswith('#'):
                    # But still check for accidentally committed secrets in comments
                    pass

                # Check each pattern
                for secret_type, config in SecretPatterns.PATTERNS.items():
                    pattern = config["pattern"]
                    matches = re.finditer(pattern, line, re.IGNORECASE)

                    for match in matches:
                        matched_text = match.group(0)

                        # Check false positives
                        if self._is_false_positive(matched_text):
                            continue

                        # Calculate entropy
                        entropy = self._calculate_entropy(matched_text)

                        # Determine confidence
                        confidence = self._determine_confidence(
                            secret_type,
                            matched_text,
                            entropy,
                            line
                        )

                        secret = Secret(
                            type=secret_type,
                            file=str(file_path.relative_to(self.root_dir)),
                            line_number=line_num,
                            line=line.strip(),
                            matched_text=self._redact_secret(matched_text),
                            entropy=entropy,
                            severity=config["severity"],
                            confidence=confidence
                        )

                        secrets.append(secret)

        except Exception as e:
            logger.warning(f"Failed to scan {file_path}: {e}")

        return secrets

    def scan_directory(self) -> List[Secret]:
        """Scan entire directory"""
        logger.info(f"Scanning directory: {self.root_dir}")

        for root, dirs, files in os.walk(self.root_dir):
            # Exclude certain directories
            dirs[:] = [d for d in dirs if d not in self.EXCLUDED_DIRS]

            for file in files:
                # Skip excluded extensions
                if any(file.endswith(ext) for ext in self.EXCLUDED_FILES):
                    continue

                file_path = Path(root) / file

                # Scan file
                file_secrets = self.scan_file(file_path)
                self.secrets.extend(file_secrets)
                self.scanned_files += 1

        logger.info(f"Scanned {self.scanned_files} files")
        logger.info(f"Found {len(self.secrets)} potential secrets")

        return self.secrets

    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of text"""
        if not text:
            return 0.0

        # Count character frequencies
        freq = Counter(text)
        length = len(text)

        # Calculate entropy
        entropy = 0.0
        for count in freq.values():
            p = count / length
            entropy -= p * math.log2(p)

        return entropy

    def _determine_confidence(
        self,
        secret_type: str,
        matched_text: str,
        entropy: float,
        line: str
    ) -> str:
        """Determine confidence level of detection"""

        # High confidence: known format + high entropy
        if secret_type in ["openai_api_key", "anthropic_api_key", "aws_access_key"]:
            if entropy > 4.0:
                return "confirmed"
            return "likely"

        # Medium confidence: generic pattern + high entropy
        if entropy > self.ENTROPY_THRESHOLD:
            return "likely"

        # Low confidence: pattern match but low entropy
        if entropy > 3.0:
            return "possible"

        return "unlikely"

    def _is_false_positive(self, text: str) -> bool:
        """Check if text is a known false positive"""
        text_lower = text.lower()

        for fp in self.false_positives:
            if fp.lower() in text_lower:
                return True

        return False

    def _redact_secret(self, text: str) -> str:
        """Redact secret for logging (show first/last 4 chars)"""
        if len(text) <= 8:
            return "***REDACTED***"

        return f"{text[:4]}...{text[-4:]}"

    def check_env_in_git(self) -> List[str]:
        """Check if .env files are tracked in git"""
        logger.info("Checking for .env files in git...")
        env_files = []

        try:
            import subprocess

            result = subprocess.run(
                ["git", "ls-files"],
                capture_output=True,
                text=True,
                cwd=self.root_dir
            )

            if result.returncode == 0:
                tracked_files = result.stdout.splitlines()

                for file in tracked_files:
                    if '.env' in file and not file.endswith('.example'):
                        env_files.append(file)
                        logger.warning(f"Found .env file in git: {file}")

        except Exception as e:
            logger.error(f"Failed to check git files: {e}")

        return env_files

    def generate_report(self) -> str:
        """Generate JSON report"""
        report_path = self.root_dir / "security" / "reports" / "SECRET_SCAN_RESULTS.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)

        # Group by severity
        by_severity = {
            "critical": [],
            "high": [],
            "medium": [],
            "low": []
        }

        for secret in self.secrets:
            by_severity[secret.severity].append(asdict(secret))

        # Check .env files
        env_in_git = self.check_env_in_git()

        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "scanned_files": self.scanned_files,
            "total_secrets_found": len(self.secrets),
            "summary": {
                "critical": len(by_severity["critical"]),
                "high": len(by_severity["high"]),
                "medium": len(by_severity["medium"]),
                "low": len(by_severity["low"])
            },
            "secrets": by_severity,
            "env_files_in_git": env_in_git,
            "recommendations": self._generate_recommendations()
        }

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Report saved to: {report_path}")

        return str(report_path)

    def _generate_recommendations(self) -> List[str]:
        """Generate remediation recommendations"""
        recommendations = []

        if len(self.secrets) > 0:
            recommendations.append("Rotate all exposed secrets immediately")
            recommendations.append("Use environment variables for secrets")
            recommendations.append("Implement secret management (Vault, AWS Secrets Manager)")
            recommendations.append("Add pre-commit hooks to prevent secret commits")
            recommendations.append("Review git history for previously committed secrets")

        if self.check_env_in_git():
            recommendations.append("Remove .env files from git tracking")
            recommendations.append("Add .env to .gitignore")
            recommendations.append("Use git-filter-repo to clean history")

        return recommendations

    def create_pre_commit_hook(self) -> bool:
        """Create pre-commit hook to prevent secret commits"""
        hook_path = self.root_dir / ".git" / "hooks" / "pre-commit"

        hook_content = """#!/bin/bash
# GreenLang Secret Scanner Pre-Commit Hook

echo "Running secret scan..."

python security/scripts/scan_secrets.py --pre-commit

if [ $? -ne 0 ]; then
    echo "ERROR: Secrets detected! Commit blocked."
    echo "Remove secrets before committing or use .gitignore"
    exit 1
fi

echo "Secret scan passed."
exit 0
"""

        try:
            hook_path.parent.mkdir(parents=True, exist_ok=True)

            with open(hook_path, 'w') as f:
                f.write(hook_content)

            # Make executable
            os.chmod(hook_path, 0o755)

            logger.info(f"Created pre-commit hook: {hook_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to create pre-commit hook: {e}")
            return False


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Scan for secrets in codebase")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Root directory")
    parser.add_argument("--pre-commit", action="store_true", help="Run as pre-commit hook")
    parser.add_argument("--install-hook", action="store_true", help="Install pre-commit hook")

    args = parser.parse_args()

    scanner = SecretScanner(args.root)

    if args.install_hook:
        scanner.create_pre_commit_hook()
        return

    # Scan for secrets
    secrets = scanner.scan_directory()

    # Generate report
    report_path = scanner.generate_report()

    # Print summary
    print("\n" + "="*80)
    print("SECRET SCAN RESULTS")
    print("="*80)
    print(f"Files Scanned: {scanner.scanned_files}")
    print(f"Secrets Found: {len(secrets)}")
    print(f"\nBy Severity:")

    summary = {
        "critical": sum(1 for s in secrets if s.severity == "critical"),
        "high": sum(1 for s in secrets if s.severity == "high"),
        "medium": sum(1 for s in secrets if s.severity == "medium"),
        "low": sum(1 for s in secrets if s.severity == "low")
    }

    print(f"  Critical: {summary['critical']}")
    print(f"  High: {summary['high']}")
    print(f"  Medium: {summary['medium']}")
    print(f"  Low: {summary['low']}")

    print(f"\nReport: {report_path}")
    print("="*80 + "\n")

    # Fail if secrets found (for CI/pre-commit)
    if args.pre_commit and len(secrets) > 0:
        print("ERROR: Secrets detected in code!")
        print("Please remove secrets before committing.")
        sys.exit(1)

    # Fail CI if critical/high secrets found
    if summary['critical'] > 0 or summary['high'] > 0:
        print("ERROR: Critical or High severity secrets found!")
        print("Immediate action required:")
        print("1. Rotate all exposed secrets")
        print("2. Remove secrets from code")
        print("3. Use environment variables or secret management")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
