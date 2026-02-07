# -*- coding: utf-8 -*-
"""
SCA Scanner Implementations - SEC-007

Software Composition Analysis (SCA) scanners for dependency vulnerability
detection. Implements Trivy, Snyk, pip-audit, and Safety scanner integrations.

Scanners:
    - TrivyScanner: Comprehensive vulnerability scanner for dependencies
    - SnykScanner: Commercial-grade SCA with fix recommendations
    - PipAuditScanner: Python-specific pip vulnerability scanner
    - SafetyScanner: Python dependency vulnerability scanner

Example:
    >>> from greenlang.infrastructure.security_scanning.scanners.sca import (
    ...     TrivyScanner,
    ...     PipAuditScanner,
    ... )
    >>> config = ScannerConfig(name="trivy", scanner_type=ScannerType.SCA)
    >>> scanner = TrivyScanner(config)
    >>> result = await scanner.scan("/path/to/project")

Author: GreenLang Security Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from greenlang.infrastructure.security_scanning.config import (
    Severity,
    ScannerConfig,
    ScannerType,
)
from greenlang.infrastructure.security_scanning.models import (
    FileLocation,
    ScanFinding,
    ScanResult,
    ScanStatus,
    VulnerabilityInfo,
    RemediationInfo,
)
from greenlang.infrastructure.security_scanning.scanners.base import (
    BaseScanner,
    ScannerParseError,
    normalize_path,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trivy Scanner (Filesystem/Dependency)
# ---------------------------------------------------------------------------


class TrivyScanner(BaseScanner):
    """Trivy scanner for dependency vulnerability detection.

    Trivy scans filesystem for vulnerabilities in dependencies including
    Python (pip), Node.js (npm), Go, Ruby, and more.

    Example:
        >>> config = ScannerConfig(name="trivy", scanner_type=ScannerType.SCA)
        >>> scanner = TrivyScanner(config)
        >>> result = await scanner.scan("/path/to/project")
    """

    # Trivy severity mapping
    SEVERITY_MAP: Dict[str, Severity] = {
        "CRITICAL": Severity.CRITICAL,
        "HIGH": Severity.HIGH,
        "MEDIUM": Severity.MEDIUM,
        "LOW": Severity.LOW,
        "UNKNOWN": Severity.INFO,
    }

    async def scan(self, target_path: str) -> ScanResult:
        """Execute Trivy filesystem scan.

        Args:
            target_path: Path to scan.

        Returns:
            ScanResult with findings.
        """
        started_at = datetime.now(timezone.utc)

        if not self.is_available():
            return self._create_result(
                findings=[],
                status=ScanStatus.FAILED,
                started_at=started_at,
                error_message=f"Trivy not found: {self.config.executable}",
            )

        command = self._build_command(target_path)

        try:
            stdout, stderr, exit_code = await self._run_command(command)

            # Trivy exit codes:
            # 0 = success, no vulnerabilities
            # 1 = success, vulnerabilities found
            # other = error

            if exit_code > 1:
                return self._create_result(
                    findings=[],
                    status=ScanStatus.FAILED,
                    started_at=started_at,
                    error_message=stderr or f"Trivy failed with exit code {exit_code}",
                    exit_code=exit_code,
                    command=" ".join(command),
                    scan_path=target_path,
                )

            findings = self.parse_results(stdout)
            filtered_findings = self._apply_filters(findings)

            return self._create_result(
                findings=filtered_findings,
                status=ScanStatus.COMPLETED,
                started_at=started_at,
                exit_code=exit_code,
                raw_output=stdout,
                command=" ".join(command),
                scan_path=target_path,
            )

        except Exception as e:
            logger.error("Trivy scan failed: %s", e, exc_info=True)
            return self._create_result(
                findings=[],
                status=ScanStatus.FAILED,
                started_at=started_at,
                error_message=str(e),
                scan_path=target_path,
            )

    def _build_command(self, target_path: str) -> List[str]:
        """Build Trivy command.

        Args:
            target_path: Path to scan.

        Returns:
            Command list.
        """
        cmd = [self.config.executable or "trivy"]
        cmd.extend(["fs", target_path])
        cmd.extend(["--format", "json"])
        cmd.extend(["--scanners", "vuln"])

        # Add severity filter
        severity_filter = []
        if self.config.severity_threshold == Severity.CRITICAL:
            severity_filter = ["CRITICAL"]
        elif self.config.severity_threshold == Severity.HIGH:
            severity_filter = ["CRITICAL", "HIGH"]
        elif self.config.severity_threshold == Severity.MEDIUM:
            severity_filter = ["CRITICAL", "HIGH", "MEDIUM"]
        else:
            severity_filter = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]

        cmd.extend(["--severity", ",".join(severity_filter)])

        # Add skip dirs
        for path in self.config.exclude_paths:
            cmd.extend(["--skip-dirs", path])

        cmd.extend(self.config.extra_args)
        return cmd

    def parse_results(self, raw_output: str) -> List[ScanFinding]:
        """Parse Trivy JSON output.

        Args:
            raw_output: JSON output from Trivy.

        Returns:
            List of findings.
        """
        if not raw_output.strip():
            return []

        try:
            data = json.loads(raw_output)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse Trivy output: %s", e)
            return []

        findings = []
        results = data.get("Results", [])

        for result in results:
            target = result.get("Target", "")
            vulns = result.get("Vulnerabilities", []) or []

            for vuln in vulns:
                finding = self._parse_vulnerability(vuln, target)
                if finding:
                    findings.append(finding)

        logger.info("Trivy found %d vulnerabilities", len(findings))
        return findings

    def _parse_vulnerability(
        self, vuln: Dict[str, Any], target: str
    ) -> Optional[ScanFinding]:
        """Parse a single Trivy vulnerability.

        Args:
            vuln: Vulnerability dictionary.
            target: Target file/package.

        Returns:
            ScanFinding or None.
        """
        try:
            severity_str = vuln.get("Severity", "MEDIUM")
            severity = self.SEVERITY_MAP.get(severity_str, Severity.MEDIUM)

            vuln_id = vuln.get("VulnerabilityID", "")
            pkg_name = vuln.get("PkgName", "")
            installed_version = vuln.get("InstalledVersion", "")
            fixed_version = vuln.get("FixedVersion", "")
            title = vuln.get("Title", vuln_id)
            description = vuln.get("Description", "")
            primary_url = vuln.get("PrimaryURL", "")

            # CVSS score
            cvss_score = None
            cvss_vector = None
            cvss = vuln.get("CVSS", {})
            if cvss:
                # Try to get NVD or any available CVSS
                for source in ["nvd", "redhat", "ghsa"]:
                    if source in cvss:
                        cvss_score = cvss[source].get("V3Score")
                        cvss_vector = cvss[source].get("V3Vector")
                        break

            location = FileLocation(
                file_path=normalize_path(target),
                start_line=1,
            )

            vuln_info = VulnerabilityInfo(
                cve_id=vuln_id if vuln_id.startswith("CVE-") else None,
                cvss_score=cvss_score,
                cvss_vector=cvss_vector,
                description=description,
                references=vuln.get("References", []),
                published_date=self._parse_date(vuln.get("PublishedDate")),
            )

            remediation = RemediationInfo(
                fixed_version=fixed_version,
                description=f"Upgrade {pkg_name} to {fixed_version}"
                if fixed_version
                else "No fix available",
                patch_available=bool(fixed_version),
                auto_fixable=bool(fixed_version),
            )

            return ScanFinding(
                title=f"{vuln_id}: {title}" if title != vuln_id else vuln_id,
                description=f"{pkg_name} {installed_version}: {description}",
                severity=severity,
                scanner_name=self.name,
                scanner_type=ScannerType.SCA,
                rule_id=vuln_id,
                location=location,
                vulnerability_info=vuln_info,
                remediation_info=remediation,
                tags={"sca", "dependency", pkg_name},
                raw_data=vuln,
            )

        except Exception as e:
            logger.warning("Failed to parse Trivy vulnerability: %s", e)
            return None

    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse ISO date string.

        Args:
            date_str: ISO format date string.

        Returns:
            datetime or None.
        """
        if not date_str:
            return None
        try:
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return None

    def to_sarif(self, findings: List[ScanFinding]) -> Dict[str, Any]:
        """Convert findings to SARIF format."""
        rules = {}
        results = []

        for finding in findings:
            rule_id = finding.rule_id or finding.finding_id

            if rule_id not in rules:
                rules[rule_id] = {
                    "id": rule_id,
                    "name": finding.title,
                    "shortDescription": {"text": finding.title},
                    "fullDescription": {"text": finding.description},
                    "helpUri": f"https://nvd.nist.gov/vuln/detail/{rule_id}",
                    "properties": {
                        "security-severity": str(finding.get_risk_score()),
                    },
                }

            results.append(finding.to_sarif_result())

        return {
            "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
            "version": "2.1.0",
            "runs": [
                {
                    "tool": {
                        "driver": {
                            "name": "Trivy",
                            "version": self.version or "unknown",
                            "informationUri": "https://trivy.dev/",
                            "rules": list(rules.values()),
                        }
                    },
                    "results": results,
                }
            ],
        }

    def _get_scanner_url(self) -> str:
        return "https://trivy.dev/"


# ---------------------------------------------------------------------------
# Snyk Scanner
# ---------------------------------------------------------------------------


class SnykScanner(BaseScanner):
    """Snyk scanner for dependency vulnerability detection.

    Snyk provides detailed vulnerability information and fix recommendations.
    Requires SNYK_TOKEN environment variable for authentication.

    Example:
        >>> config = ScannerConfig(
        ...     name="snyk",
        ...     scanner_type=ScannerType.SCA,
        ...     environment={"SNYK_TOKEN": "your-token"}
        ... )
        >>> scanner = SnykScanner(config)
        >>> result = await scanner.scan("/path/to/project")
    """

    # Snyk severity mapping
    SEVERITY_MAP: Dict[str, Severity] = {
        "critical": Severity.CRITICAL,
        "high": Severity.HIGH,
        "medium": Severity.MEDIUM,
        "low": Severity.LOW,
    }

    async def scan(self, target_path: str) -> ScanResult:
        """Execute Snyk scan.

        Args:
            target_path: Path to scan.

        Returns:
            ScanResult with findings.
        """
        started_at = datetime.now(timezone.utc)

        if not self.is_available():
            return self._create_result(
                findings=[],
                status=ScanStatus.FAILED,
                started_at=started_at,
                error_message=f"Snyk not found: {self.config.executable}",
            )

        # Check for token
        import os

        if not os.environ.get("SNYK_TOKEN") and not self.config.environment.get(
            "SNYK_TOKEN"
        ):
            return self._create_result(
                findings=[],
                status=ScanStatus.FAILED,
                started_at=started_at,
                error_message="SNYK_TOKEN environment variable not set",
            )

        command = self._build_command(target_path)

        try:
            stdout, stderr, exit_code = await self._run_command(
                command,
                cwd=target_path,
            )

            # Snyk exit codes:
            # 0 = success, no vulnerabilities
            # 1 = vulnerabilities found
            # 2 = action needed
            # 3 = failure

            if exit_code == 3:
                return self._create_result(
                    findings=[],
                    status=ScanStatus.FAILED,
                    started_at=started_at,
                    error_message=stderr or "Snyk scan failed",
                    exit_code=exit_code,
                    command=" ".join(command),
                    scan_path=target_path,
                )

            findings = self.parse_results(stdout)
            filtered_findings = self._apply_filters(findings)

            return self._create_result(
                findings=filtered_findings,
                status=ScanStatus.COMPLETED,
                started_at=started_at,
                exit_code=exit_code,
                raw_output=stdout,
                command=" ".join(command),
                scan_path=target_path,
            )

        except Exception as e:
            logger.error("Snyk scan failed: %s", e, exc_info=True)
            return self._create_result(
                findings=[],
                status=ScanStatus.FAILED,
                started_at=started_at,
                error_message=str(e),
                scan_path=target_path,
            )

    def _build_command(self, target_path: str) -> List[str]:
        """Build Snyk command.

        Args:
            target_path: Path to scan.

        Returns:
            Command list.
        """
        cmd = [self.config.executable or "snyk"]
        cmd.extend(["test", "--json"])

        # Severity threshold
        if self.config.severity_threshold == Severity.CRITICAL:
            cmd.extend(["--severity-threshold=critical"])
        elif self.config.severity_threshold == Severity.HIGH:
            cmd.extend(["--severity-threshold=high"])
        elif self.config.severity_threshold == Severity.MEDIUM:
            cmd.extend(["--severity-threshold=medium"])

        cmd.extend(self.config.extra_args)
        return cmd

    def parse_results(self, raw_output: str) -> List[ScanFinding]:
        """Parse Snyk JSON output.

        Args:
            raw_output: JSON output from Snyk.

        Returns:
            List of findings.
        """
        if not raw_output.strip():
            return []

        try:
            data = json.loads(raw_output)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse Snyk output: %s", e)
            return []

        findings = []
        vulns = data.get("vulnerabilities", [])

        for vuln in vulns:
            finding = self._parse_vulnerability(vuln)
            if finding:
                findings.append(finding)

        logger.info("Snyk found %d vulnerabilities", len(findings))
        return findings

    def _parse_vulnerability(self, vuln: Dict[str, Any]) -> Optional[ScanFinding]:
        """Parse a single Snyk vulnerability.

        Args:
            vuln: Vulnerability dictionary.

        Returns:
            ScanFinding or None.
        """
        try:
            severity_str = vuln.get("severity", "medium")
            severity = self.SEVERITY_MAP.get(severity_str, Severity.MEDIUM)

            vuln_id = vuln.get("id", "")
            title = vuln.get("title", vuln_id)
            description = vuln.get("description", "")
            pkg_name = vuln.get("packageName", "")
            version = vuln.get("version", "")

            # CVE identifiers
            identifiers = vuln.get("identifiers", {})
            cve_list = identifiers.get("CVE", [])
            cve_id = cve_list[0] if cve_list else None
            cwe_list = identifiers.get("CWE", [])
            cwe_id = cwe_list[0] if cwe_list else None

            # CVSS
            cvss_score = vuln.get("cvssScore")

            # Fix info
            is_upgradable = vuln.get("isUpgradable", False)
            upgrade_path = vuln.get("upgradePath", [])
            fixed_in = vuln.get("fixedIn", [])

            vuln_info = VulnerabilityInfo(
                cve_id=cve_id,
                cwe_id=cwe_id,
                cvss_score=cvss_score,
                description=description,
                references=vuln.get("references", []),
            )

            remediation = RemediationInfo(
                fixed_version=fixed_in[0] if fixed_in else None,
                upgrade_path=upgrade_path,
                patch_available=is_upgradable,
                auto_fixable=is_upgradable,
                description=f"Upgrade {pkg_name} to {fixed_in[0]}"
                if fixed_in
                else "No fix available",
            )

            return ScanFinding(
                title=f"{vuln_id}: {title}",
                description=f"{pkg_name}@{version}: {description}",
                severity=severity,
                scanner_name=self.name,
                scanner_type=ScannerType.SCA,
                rule_id=vuln_id,
                vulnerability_info=vuln_info,
                remediation_info=remediation,
                tags={"sca", "snyk", pkg_name},
                raw_data=vuln,
            )

        except Exception as e:
            logger.warning("Failed to parse Snyk vulnerability: %s", e)
            return None

    def to_sarif(self, findings: List[ScanFinding]) -> Dict[str, Any]:
        """Convert findings to SARIF format."""
        rules = {}
        results = []

        for finding in findings:
            rule_id = finding.rule_id or finding.finding_id

            if rule_id not in rules:
                rules[rule_id] = {
                    "id": rule_id,
                    "name": finding.title,
                    "shortDescription": {"text": finding.title},
                    "fullDescription": {"text": finding.description},
                    "helpUri": f"https://security.snyk.io/vuln/{rule_id}",
                }

            results.append(finding.to_sarif_result())

        return {
            "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
            "version": "2.1.0",
            "runs": [
                {
                    "tool": {
                        "driver": {
                            "name": "Snyk",
                            "version": self.version or "unknown",
                            "informationUri": "https://snyk.io/",
                            "rules": list(rules.values()),
                        }
                    },
                    "results": results,
                }
            ],
        }

    def _get_scanner_url(self) -> str:
        return "https://snyk.io/"


# ---------------------------------------------------------------------------
# pip-audit Scanner
# ---------------------------------------------------------------------------


class PipAuditScanner(BaseScanner):
    """pip-audit scanner for Python dependency vulnerabilities.

    Uses the Python Packaging Advisory Database to check for known
    vulnerabilities in installed packages.

    Example:
        >>> config = ScannerConfig(name="pip-audit", scanner_type=ScannerType.SCA)
        >>> scanner = PipAuditScanner(config)
        >>> result = await scanner.scan("/path/to/project")
    """

    async def scan(self, target_path: str) -> ScanResult:
        """Execute pip-audit scan.

        Args:
            target_path: Path to scan (directory with requirements.txt or venv).

        Returns:
            ScanResult with findings.
        """
        started_at = datetime.now(timezone.utc)

        if not self.is_available():
            return self._create_result(
                findings=[],
                status=ScanStatus.FAILED,
                started_at=started_at,
                error_message=f"pip-audit not found: {self.config.executable}",
            )

        command = self._build_command(target_path)

        try:
            stdout, stderr, exit_code = await self._run_command(
                command,
                cwd=target_path,
            )

            # pip-audit exit codes:
            # 0 = no vulnerabilities
            # 1 = vulnerabilities found
            # 2+ = error

            if exit_code > 1:
                return self._create_result(
                    findings=[],
                    status=ScanStatus.FAILED,
                    started_at=started_at,
                    error_message=stderr or f"pip-audit failed with exit code {exit_code}",
                    exit_code=exit_code,
                    command=" ".join(command),
                    scan_path=target_path,
                )

            findings = self.parse_results(stdout)
            filtered_findings = self._apply_filters(findings)

            return self._create_result(
                findings=filtered_findings,
                status=ScanStatus.COMPLETED,
                started_at=started_at,
                exit_code=exit_code,
                raw_output=stdout,
                command=" ".join(command),
                scan_path=target_path,
            )

        except Exception as e:
            logger.error("pip-audit scan failed: %s", e, exc_info=True)
            return self._create_result(
                findings=[],
                status=ScanStatus.FAILED,
                started_at=started_at,
                error_message=str(e),
                scan_path=target_path,
            )

    def _build_command(self, target_path: str) -> List[str]:
        """Build pip-audit command.

        Args:
            target_path: Path to scan.

        Returns:
            Command list.
        """
        import os

        cmd = [self.config.executable or "pip-audit"]
        cmd.extend(["--format", "json"])

        # Check for requirements file
        req_file = os.path.join(target_path, "requirements.txt")
        if os.path.exists(req_file):
            cmd.extend(["-r", req_file])

        cmd.extend(self.config.extra_args)
        return cmd

    def parse_results(self, raw_output: str) -> List[ScanFinding]:
        """Parse pip-audit JSON output.

        Args:
            raw_output: JSON output from pip-audit.

        Returns:
            List of findings.
        """
        if not raw_output.strip():
            return []

        try:
            data = json.loads(raw_output)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse pip-audit output: %s", e)
            return []

        findings = []
        dependencies = data.get("dependencies", [])

        for dep in dependencies:
            vulns = dep.get("vulns", [])
            for vuln in vulns:
                finding = self._parse_vulnerability(vuln, dep)
                if finding:
                    findings.append(finding)

        logger.info("pip-audit found %d vulnerabilities", len(findings))
        return findings

    def _parse_vulnerability(
        self, vuln: Dict[str, Any], dep: Dict[str, Any]
    ) -> Optional[ScanFinding]:
        """Parse a single pip-audit vulnerability.

        Args:
            vuln: Vulnerability dictionary.
            dep: Dependency dictionary.

        Returns:
            ScanFinding or None.
        """
        try:
            vuln_id = vuln.get("id", "")
            description = vuln.get("description", "")
            fix_versions = vuln.get("fix_versions", [])

            pkg_name = dep.get("name", "")
            version = dep.get("version", "")

            # Determine severity from description or default to HIGH
            severity = Severity.HIGH
            desc_lower = description.lower()
            if "critical" in desc_lower:
                severity = Severity.CRITICAL
            elif "low" in desc_lower:
                severity = Severity.LOW
            elif "medium" in desc_lower or "moderate" in desc_lower:
                severity = Severity.MEDIUM

            vuln_info = VulnerabilityInfo(
                cve_id=vuln_id if vuln_id.startswith("CVE-") else None,
                description=description,
            )

            remediation = RemediationInfo(
                fixed_version=fix_versions[0] if fix_versions else None,
                patch_available=bool(fix_versions),
                auto_fixable=bool(fix_versions),
                description=f"Upgrade {pkg_name} to {fix_versions[0]}"
                if fix_versions
                else "No fix available",
            )

            return ScanFinding(
                title=f"{vuln_id}: {pkg_name} vulnerability",
                description=f"{pkg_name}=={version}: {description}",
                severity=severity,
                scanner_name=self.name,
                scanner_type=ScannerType.SCA,
                rule_id=vuln_id,
                vulnerability_info=vuln_info,
                remediation_info=remediation,
                tags={"sca", "pip-audit", "python", pkg_name},
                raw_data={"vulnerability": vuln, "dependency": dep},
            )

        except Exception as e:
            logger.warning("Failed to parse pip-audit vulnerability: %s", e)
            return None

    def to_sarif(self, findings: List[ScanFinding]) -> Dict[str, Any]:
        """Convert findings to SARIF format."""
        rules = {}
        results = []

        for finding in findings:
            rule_id = finding.rule_id or finding.finding_id

            if rule_id not in rules:
                rules[rule_id] = {
                    "id": rule_id,
                    "name": finding.title,
                    "shortDescription": {"text": finding.title},
                    "fullDescription": {"text": finding.description},
                }

            results.append(finding.to_sarif_result())

        return {
            "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
            "version": "2.1.0",
            "runs": [
                {
                    "tool": {
                        "driver": {
                            "name": "pip-audit",
                            "version": self.version or "unknown",
                            "informationUri": "https://pypi.org/project/pip-audit/",
                            "rules": list(rules.values()),
                        }
                    },
                    "results": results,
                }
            ],
        }

    def _get_scanner_url(self) -> str:
        return "https://pypi.org/project/pip-audit/"


# ---------------------------------------------------------------------------
# Safety Scanner
# ---------------------------------------------------------------------------


class SafetyScanner(BaseScanner):
    """Safety scanner for Python dependency vulnerabilities.

    Uses Safety DB to check for known vulnerabilities in Python packages.

    Example:
        >>> config = ScannerConfig(name="safety", scanner_type=ScannerType.SCA)
        >>> scanner = SafetyScanner(config)
        >>> result = await scanner.scan("/path/to/project")
    """

    async def scan(self, target_path: str) -> ScanResult:
        """Execute Safety scan.

        Args:
            target_path: Path to scan.

        Returns:
            ScanResult with findings.
        """
        started_at = datetime.now(timezone.utc)

        if not self.is_available():
            return self._create_result(
                findings=[],
                status=ScanStatus.FAILED,
                started_at=started_at,
                error_message=f"Safety not found: {self.config.executable}",
            )

        command = self._build_command(target_path)

        try:
            stdout, stderr, exit_code = await self._run_command(
                command,
                cwd=target_path,
            )

            # Safety exit codes vary, parse output regardless
            findings = self.parse_results(stdout)
            filtered_findings = self._apply_filters(findings)

            return self._create_result(
                findings=filtered_findings,
                status=ScanStatus.COMPLETED,
                started_at=started_at,
                exit_code=exit_code,
                raw_output=stdout,
                command=" ".join(command),
                scan_path=target_path,
            )

        except Exception as e:
            logger.error("Safety scan failed: %s", e, exc_info=True)
            return self._create_result(
                findings=[],
                status=ScanStatus.FAILED,
                started_at=started_at,
                error_message=str(e),
                scan_path=target_path,
            )

    def _build_command(self, target_path: str) -> List[str]:
        """Build Safety command.

        Args:
            target_path: Path to scan.

        Returns:
            Command list.
        """
        import os

        cmd = [self.config.executable or "safety"]
        cmd.extend(["check", "--json"])

        # Check for requirements file
        req_file = os.path.join(target_path, "requirements.txt")
        if os.path.exists(req_file):
            cmd.extend(["-r", req_file])

        cmd.extend(self.config.extra_args)
        return cmd

    def parse_results(self, raw_output: str) -> List[ScanFinding]:
        """Parse Safety JSON output.

        Args:
            raw_output: JSON output from Safety.

        Returns:
            List of findings.
        """
        if not raw_output.strip():
            return []

        try:
            data = json.loads(raw_output)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse Safety output: %s", e)
            return []

        findings = []

        # Safety v2+ format
        vulns = data.get("vulnerabilities", [])
        for vuln in vulns:
            finding = self._parse_vulnerability_v2(vuln)
            if finding:
                findings.append(finding)

        # Safety v1 format (fallback)
        if not vulns and isinstance(data, list):
            for vuln in data:
                finding = self._parse_vulnerability_v1(vuln)
                if finding:
                    findings.append(finding)

        logger.info("Safety found %d vulnerabilities", len(findings))
        return findings

    def _parse_vulnerability_v2(self, vuln: Dict[str, Any]) -> Optional[ScanFinding]:
        """Parse Safety v2 vulnerability format.

        Args:
            vuln: Vulnerability dictionary.

        Returns:
            ScanFinding or None.
        """
        try:
            vuln_id = vuln.get("vulnerability_id", "")
            pkg_name = vuln.get("package_name", "")
            vulnerable_versions = vuln.get("vulnerable_versions", "")
            analyzed_version = vuln.get("analyzed_version", "")
            advisory = vuln.get("advisory", "")

            severity = Severity.HIGH  # Safety doesn't provide severity

            vuln_info = VulnerabilityInfo(
                cve_id=vuln.get("CVE"),
                description=advisory,
            )

            return ScanFinding(
                title=f"Safety {vuln_id}: {pkg_name} vulnerability",
                description=f"{pkg_name}=={analyzed_version}: {advisory}",
                severity=severity,
                scanner_name=self.name,
                scanner_type=ScannerType.SCA,
                rule_id=vuln_id,
                vulnerability_info=vuln_info,
                tags={"sca", "safety", "python", pkg_name},
                raw_data=vuln,
            )

        except Exception as e:
            logger.warning("Failed to parse Safety vulnerability: %s", e)
            return None

    def _parse_vulnerability_v1(self, vuln: List[Any]) -> Optional[ScanFinding]:
        """Parse Safety v1 vulnerability format (list format).

        Args:
            vuln: Vulnerability as list [pkg, version, spec, vuln_id, advisory].

        Returns:
            ScanFinding or None.
        """
        try:
            if len(vuln) < 5:
                return None

            pkg_name = vuln[0]
            version = vuln[1]
            vuln_id = vuln[3]
            advisory = vuln[4]

            vuln_info = VulnerabilityInfo(description=advisory)

            return ScanFinding(
                title=f"Safety {vuln_id}: {pkg_name} vulnerability",
                description=f"{pkg_name}=={version}: {advisory}",
                severity=Severity.HIGH,
                scanner_name=self.name,
                scanner_type=ScannerType.SCA,
                rule_id=str(vuln_id),
                vulnerability_info=vuln_info,
                tags={"sca", "safety", "python", pkg_name},
                raw_data={"vuln_list": vuln},
            )

        except Exception as e:
            logger.warning("Failed to parse Safety v1 vulnerability: %s", e)
            return None

    def to_sarif(self, findings: List[ScanFinding]) -> Dict[str, Any]:
        """Convert findings to SARIF format."""
        rules = {}
        results = []

        for finding in findings:
            rule_id = finding.rule_id or finding.finding_id

            if rule_id not in rules:
                rules[rule_id] = {
                    "id": rule_id,
                    "name": finding.title,
                    "shortDescription": {"text": finding.title},
                    "fullDescription": {"text": finding.description},
                }

            results.append(finding.to_sarif_result())

        return {
            "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
            "version": "2.1.0",
            "runs": [
                {
                    "tool": {
                        "driver": {
                            "name": "Safety",
                            "version": self.version or "unknown",
                            "informationUri": "https://safetycli.com/",
                            "rules": list(rules.values()),
                        }
                    },
                    "results": results,
                }
            ],
        }

    def _get_scanner_url(self) -> str:
        return "https://safetycli.com/"
