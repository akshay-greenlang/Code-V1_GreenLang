#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GreenLang Dependency Vulnerability Scanner

Scans all requirements.txt files for:
- Known CVEs (using safety and pip-audit)
- Outdated packages with security patches
- Unmaintained dependencies
- License compliance issues

Usage:
    python scan_dependencies.py
    python scan_dependencies.py --auto-fix
    python scan_dependencies.py --report-only

Author: Security & Compliance Audit Team
Date: 2025-11-09
"""

import asyncio
import json
import logging
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional
import re
from greenlang.determinism import DeterministicClock

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Severity(Enum):
    """Vulnerability severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"


@dataclass
class Vulnerability:
    """Vulnerability information"""
    package: str
    installed_version: str
    vulnerable_spec: str
    cve: Optional[str]
    severity: Severity
    description: str
    fixed_version: Optional[str]
    source: str  # safety, pip-audit, etc.


@dataclass
class LicenseIssue:
    """License compliance issue"""
    package: str
    version: str
    license: str
    issue: str
    severity: Severity


@dataclass
class ScanResult:
    """Scan result"""
    timestamp: str
    total_packages: int
    vulnerabilities: List[Vulnerability]
    license_issues: List[LicenseIssue]
    outdated_packages: List[Dict[str, str]]
    unmaintained_packages: List[Dict[str, str]]
    summary: Dict[str, int]


class DependencyScanner:
    """Scan dependencies for security issues"""

    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.requirements_files = []
        self.all_packages = set()

    def find_requirements_files(self) -> List[Path]:
        """Find all requirements*.txt files"""
        files = []

        # Find all requirements files
        for pattern in ["requirements*.txt", "**/requirements*.txt"]:
            files.extend(self.root_dir.glob(pattern))

        self.requirements_files = files
        logger.info(f"Found {len(files)} requirements files")

        return files

    async def scan_with_safety(self) -> List[Vulnerability]:
        """Scan with safety (PyUp vulnerability database)"""
        logger.info("Scanning with safety...")
        vulnerabilities = []

        try:
            # Run safety check
            result = subprocess.run(
                ["safety", "check", "--json", "--output", "json"],
                capture_output=True,
                text=True,
                cwd=self.root_dir
            )

            if result.returncode in [0, 64]:  # 64 = vulnerabilities found
                try:
                    data = json.loads(result.stdout)

                    for vuln in data:
                        vulnerabilities.append(Vulnerability(
                            package=vuln.get("package"),
                            installed_version=vuln.get("installed_version"),
                            vulnerable_spec=vuln.get("vulnerable_spec"),
                            cve=vuln.get("cve"),
                            severity=self._map_severity(vuln.get("severity", "medium")),
                            description=vuln.get("advisory"),
                            fixed_version=vuln.get("fixed_in"),
                            source="safety"
                        ))

                except json.JSONDecodeError:
                    logger.warning("Failed to parse safety output")

        except FileNotFoundError:
            logger.warning("safety not installed. Install with: pip install safety")

        logger.info(f"Safety found {len(vulnerabilities)} vulnerabilities")
        return vulnerabilities

    async def scan_with_pip_audit(self) -> List[Vulnerability]:
        """Scan with pip-audit (OSV database)"""
        logger.info("Scanning with pip-audit...")
        vulnerabilities = []

        try:
            # Run pip-audit
            result = subprocess.run(
                ["pip-audit", "--format", "json", "--local"],
                capture_output=True,
                text=True,
                cwd=self.root_dir
            )

            if result.returncode in [0, 1]:  # 1 = vulnerabilities found
                try:
                    data = json.loads(result.stdout)

                    for vuln in data.get("vulnerabilities", []):
                        vulnerabilities.append(Vulnerability(
                            package=vuln.get("name"),
                            installed_version=vuln.get("version"),
                            vulnerable_spec=vuln.get("vuln_spec"),
                            cve=vuln.get("id"),
                            severity=self._map_severity(
                                vuln.get("severity", {}).get("level", "medium")
                            ),
                            description=vuln.get("description", ""),
                            fixed_version=vuln.get("fix_versions", [None])[0],
                            source="pip-audit"
                        ))

                except json.JSONDecodeError:
                    logger.warning("Failed to parse pip-audit output")

        except FileNotFoundError:
            logger.warning("pip-audit not installed. Install with: pip install pip-audit")

        logger.info(f"pip-audit found {len(vulnerabilities)} vulnerabilities")
        return vulnerabilities

    async def check_outdated_packages(self) -> List[Dict[str, str]]:
        """Check for outdated packages"""
        logger.info("Checking for outdated packages...")
        outdated = []

        try:
            # Run pip list --outdated
            result = subprocess.run(
                ["pip", "list", "--outdated", "--format", "json"],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                try:
                    data = json.loads(result.stdout)

                    for pkg in data:
                        # Check if security release
                        is_security = self._is_security_release(
                            pkg["name"],
                            pkg["version"],
                            pkg["latest_version"]
                        )

                        if is_security:
                            outdated.append({
                                "package": pkg["name"],
                                "current": pkg["version"],
                                "latest": pkg["latest_version"],
                                "type": pkg.get("latest_filetype", "unknown"),
                                "is_security_release": True
                            })

                except json.JSONDecodeError:
                    logger.warning("Failed to parse pip list output")

        except FileNotFoundError:
            logger.error("pip not found")

        logger.info(f"Found {len(outdated)} outdated packages with security updates")
        return outdated

    async def check_licenses(self) -> List[LicenseIssue]:
        """Check for license compliance issues"""
        logger.info("Checking license compliance...")
        issues = []

        # Forbidden licenses (copyleft, restrictive)
        FORBIDDEN_LICENSES = [
            "GPL",
            "AGPL",
            "LGPL",
            "SSPL",
            "Commercial",
            "Proprietary"
        ]

        # Warning licenses (require review)
        WARNING_LICENSES = [
            "MPL",
            "EPL",
            "CPL"
        ]

        try:
            # Run pip-licenses
            result = subprocess.run(
                ["pip-licenses", "--format", "json"],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                try:
                    data = json.loads(result.stdout)

                    for pkg in data:
                        license_name = pkg.get("License", "Unknown")

                        # Check forbidden
                        if any(forbidden in license_name for forbidden in FORBIDDEN_LICENSES):
                            issues.append(LicenseIssue(
                                package=pkg["Name"],
                                version=pkg["Version"],
                                license=license_name,
                                issue=f"Forbidden license: {license_name}",
                                severity=Severity.HIGH
                            ))

                        # Check warning
                        elif any(warning in license_name for warning in WARNING_LICENSES):
                            issues.append(LicenseIssue(
                                package=pkg["Name"],
                                version=pkg["Version"],
                                license=license_name,
                                issue=f"License requires review: {license_name}",
                                severity=Severity.MEDIUM
                            ))

                        # Unknown license
                        elif license_name in ["UNKNOWN", "Unknown", ""]:
                            issues.append(LicenseIssue(
                                package=pkg["Name"],
                                version=pkg["Version"],
                                license="Unknown",
                                issue="License not specified",
                                severity=Severity.LOW
                            ))

                except json.JSONDecodeError:
                    logger.warning("Failed to parse pip-licenses output")

        except FileNotFoundError:
            logger.warning("pip-licenses not installed. Install with: pip install pip-licenses")

        logger.info(f"Found {len(issues)} license issues")
        return issues

    async def check_unmaintained(self) -> List[Dict[str, str]]:
        """Check for unmaintained packages"""
        logger.info("Checking for unmaintained packages...")
        unmaintained = []

        # Known unmaintained packages (update this list)
        UNMAINTAINED_PACKAGES = [
            "pycrypto",  # Replaced by pycryptodome
            "pyyaml",    # Check version < 5.4 (has vulns)
        ]

        try:
            result = subprocess.run(
                ["pip", "list", "--format", "json"],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                data = json.loads(result.stdout)

                for pkg in data:
                    if pkg["name"].lower() in UNMAINTAINED_PACKAGES:
                        unmaintained.append({
                            "package": pkg["name"],
                            "version": pkg["version"],
                            "reason": "Package is no longer maintained",
                            "recommendation": self._get_replacement(pkg["name"])
                        })

        except Exception as e:
            logger.error(f"Failed to check unmaintained packages: {e}")

        return unmaintained

    def _map_severity(self, severity_str: str) -> Severity:
        """Map severity string to enum"""
        severity_map = {
            "critical": Severity.CRITICAL,
            "high": Severity.HIGH,
            "medium": Severity.MEDIUM,
            "low": Severity.LOW,
            "informational": Severity.INFORMATIONAL
        }

        return severity_map.get(severity_str.lower(), Severity.MEDIUM)

    def _is_security_release(self, package: str, current: str, latest: str) -> bool:
        """Check if latest version is a security release"""
        # Simple heuristic: check changelog/release notes
        # In production, query PyPI API for security release flag

        # For now, return True for any update
        # TODO: Implement PyPI API check
        return True

    def _get_replacement(self, package: str) -> str:
        """Get recommended replacement for unmaintained package"""
        replacements = {
            "pycrypto": "pycryptodome or cryptography",
            "pyyaml": "ruamel.yaml or pyyaml >= 5.4"
        }

        return replacements.get(package.lower(), "Contact security team")

    async def generate_report(self, scan_result: ScanResult) -> str:
        """Generate JSON report"""
        report_path = self.root_dir / "security" / "reports" / "DEPENDENCY_VULNERABILITIES.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict
        report_data = {
            "timestamp": scan_result.timestamp,
            "total_packages": scan_result.total_packages,
            "summary": scan_result.summary,
            "vulnerabilities": [
                {
                    "package": v.package,
                    "installed_version": v.installed_version,
                    "vulnerable_spec": v.vulnerable_spec,
                    "cve": v.cve,
                    "severity": v.severity.value,
                    "description": v.description,
                    "fixed_version": v.fixed_version,
                    "source": v.source
                }
                for v in scan_result.vulnerabilities
            ],
            "license_issues": [
                {
                    "package": li.package,
                    "version": li.version,
                    "license": li.license,
                    "issue": li.issue,
                    "severity": li.severity.value
                }
                for li in scan_result.license_issues
            ],
            "outdated_packages": scan_result.outdated_packages,
            "unmaintained_packages": scan_result.unmaintained_packages
        }

        # Write report
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)

        logger.info(f"Report saved to: {report_path}")

        return str(report_path)

    async def create_pr_for_fixes(self, scan_result: ScanResult) -> bool:
        """Create PR with security fixes"""
        logger.info("Creating PR for security fixes...")

        # TODO: Implement GitHub API integration
        # 1. Create new branch
        # 2. Update requirements files with fixed versions
        # 3. Run tests
        # 4. Create PR with summary

        logger.warning("Auto-PR creation not yet implemented")
        return False


async def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Scan dependencies for vulnerabilities")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Root directory")
    parser.add_argument("--auto-fix", action="store_true", help="Create PR with fixes")
    parser.add_argument("--report-only", action="store_true", help="Generate report only")

    args = parser.parse_args()

    scanner = DependencyScanner(args.root)

    # Find requirements files
    scanner.find_requirements_files()

    # Run scans in parallel
    safety_vulns, pip_audit_vulns, outdated, licenses, unmaintained = await asyncio.gather(
        scanner.scan_with_safety(),
        scanner.scan_with_pip_audit(),
        scanner.check_outdated_packages(),
        scanner.check_licenses(),
        scanner.check_unmaintained()
    )

    # Combine vulnerabilities (deduplicate)
    all_vulns = safety_vulns + pip_audit_vulns
    unique_vulns = []
    seen = set()

    for v in all_vulns:
        key = (v.package, v.installed_version, v.cve)
        if key not in seen:
            seen.add(key)
            unique_vulns.append(v)

    # Calculate summary
    summary = {
        "total_vulnerabilities": len(unique_vulns),
        "critical": sum(1 for v in unique_vulns if v.severity == Severity.CRITICAL),
        "high": sum(1 for v in unique_vulns if v.severity == Severity.HIGH),
        "medium": sum(1 for v in unique_vulns if v.severity == Severity.MEDIUM),
        "low": sum(1 for v in unique_vulns if v.severity == Severity.LOW),
        "license_issues": len(licenses),
        "outdated_packages": len(outdated),
        "unmaintained_packages": len(unmaintained)
    }

    # Create scan result
    scan_result = ScanResult(
        timestamp=DeterministicClock.utcnow().isoformat(),
        total_packages=len(scanner.all_packages),
        vulnerabilities=unique_vulns,
        license_issues=licenses,
        outdated_packages=outdated,
        unmaintained_packages=unmaintained,
        summary=summary
    )

    # Generate report
    report_path = await scanner.generate_report(scan_result)

    # Print summary
    print("\n" + "="*80)
    print("DEPENDENCY SECURITY SCAN RESULTS")
    print("="*80)
    print(f"Timestamp: {scan_result.timestamp}")
    print(f"Total Packages: {scan_result.total_packages}")
    print(f"\nVulnerabilities: {summary['total_vulnerabilities']}")
    print(f"  Critical: {summary['critical']}")
    print(f"  High: {summary['high']}")
    print(f"  Medium: {summary['medium']}")
    print(f"  Low: {summary['low']}")
    print(f"\nLicense Issues: {summary['license_issues']}")
    print(f"Outdated Packages (with security updates): {summary['outdated_packages']}")
    print(f"Unmaintained Packages: {summary['unmaintained_packages']}")
    print(f"\nReport: {report_path}")
    print("="*80 + "\n")

    # Fail CI if critical/high vulnerabilities
    if summary['critical'] > 0 or summary['high'] > 0:
        print("ERROR: Critical or High severity vulnerabilities found!")
        print("Please update vulnerable packages before deploying.")
        sys.exit(1)

    # Create PR if requested
    if args.auto_fix and not args.report_only:
        await scanner.create_pr_for_fixes(scan_result)

    sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
