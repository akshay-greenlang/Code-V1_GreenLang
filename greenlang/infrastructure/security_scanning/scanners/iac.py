# -*- coding: utf-8 -*-
"""
IaC Scanner Implementations - SEC-007

Infrastructure as Code (IaC) security scanners for detecting
misconfigurations in Terraform, Kubernetes, and CloudFormation.

Scanners:
    - TfsecScanner: Terraform security scanning
    - CheckovScanner: Multi-IaC security and compliance
    - KubeconformScanner: Kubernetes manifest validation

Example:
    >>> from greenlang.infrastructure.security_scanning.scanners.iac import (
    ...     TfsecScanner,
    ...     CheckovScanner,
    ... )
    >>> config = ScannerConfig(name="tfsec", scanner_type=ScannerType.IAC)
    >>> scanner = TfsecScanner(config)
    >>> result = await scanner.scan("/path/to/terraform")

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
    normalize_path,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TFSec Scanner
# ---------------------------------------------------------------------------


class TfsecScanner(BaseScanner):
    """TFSec scanner for Terraform security.

    TFSec is a static analysis security scanner for Terraform code.
    Detects potential security issues and misconfigurations.

    Example:
        >>> config = ScannerConfig(name="tfsec", scanner_type=ScannerType.IAC)
        >>> scanner = TfsecScanner(config)
        >>> result = await scanner.scan("/path/to/terraform")
    """

    # TFSec severity mapping
    SEVERITY_MAP: Dict[str, Severity] = {
        "CRITICAL": Severity.CRITICAL,
        "HIGH": Severity.HIGH,
        "MEDIUM": Severity.MEDIUM,
        "LOW": Severity.LOW,
    }

    async def scan(self, target_path: str) -> ScanResult:
        """Execute TFSec scan on Terraform files.

        Args:
            target_path: Path to Terraform directory.

        Returns:
            ScanResult with findings.
        """
        started_at = datetime.now(timezone.utc)

        if not self.is_available():
            return self._create_result(
                findings=[],
                status=ScanStatus.FAILED,
                started_at=started_at,
                error_message=f"tfsec not found: {self.config.executable}",
            )

        command = self._build_command(target_path)

        try:
            stdout, stderr, exit_code = await self._run_command(command)

            # tfsec exit codes:
            # 0 = no issues
            # 1 = issues found
            # other = error

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
            logger.error("TFSec scan failed: %s", e, exc_info=True)
            return self._create_result(
                findings=[],
                status=ScanStatus.FAILED,
                started_at=started_at,
                error_message=str(e),
                scan_path=target_path,
            )

    def _build_command(self, target_path: str) -> List[str]:
        """Build tfsec command.

        Args:
            target_path: Path to scan.

        Returns:
            Command list.
        """
        cmd = [self.config.executable or "tfsec"]
        cmd.append(target_path)
        cmd.extend(["--format", "json"])

        # Add minimum severity
        severity_map = {
            Severity.CRITICAL: "CRITICAL",
            Severity.HIGH: "HIGH",
            Severity.MEDIUM: "MEDIUM",
            Severity.LOW: "LOW",
        }
        min_severity = severity_map.get(
            self.config.severity_threshold, "LOW"
        )
        cmd.extend(["--minimum-severity", min_severity])

        # Exclude rules
        for rule in self.config.exclude_rules:
            cmd.extend(["--exclude", rule])

        cmd.extend(self.config.extra_args)
        return cmd

    def parse_results(self, raw_output: str) -> List[ScanFinding]:
        """Parse tfsec JSON output.

        Args:
            raw_output: JSON output from tfsec.

        Returns:
            List of findings.
        """
        if not raw_output.strip():
            return []

        try:
            data = json.loads(raw_output)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse tfsec output: %s", e)
            return []

        findings = []
        results = data.get("results", [])

        for result in results:
            finding = self._parse_result(result)
            if finding:
                findings.append(finding)

        logger.info("TFSec found %d issues", len(findings))
        return findings

    def _parse_result(self, result: Dict[str, Any]) -> Optional[ScanFinding]:
        """Parse a single tfsec result.

        Args:
            result: Result dictionary.

        Returns:
            ScanFinding or None.
        """
        try:
            rule_id = result.get("rule_id", "")
            severity_str = result.get("severity", "MEDIUM")
            severity = self.SEVERITY_MAP.get(severity_str.upper(), Severity.MEDIUM)

            description = result.get("description", "")
            impact = result.get("impact", "")
            resolution = result.get("resolution", "")
            resource = result.get("resource", "")

            location_data = result.get("location", {})
            filename = location_data.get("filename", "")
            start_line = location_data.get("start_line", 1)
            end_line = location_data.get("end_line", start_line)

            location = FileLocation(
                file_path=normalize_path(filename),
                start_line=start_line,
                end_line=end_line,
            )

            # Links for more info
            links = result.get("links", [])

            vuln_info = VulnerabilityInfo(
                cwe_id=result.get("cwe_id"),
                description=description,
                references=links,
            )

            remediation = RemediationInfo(
                description=resolution,
                effort_estimate="low",
            )

            return ScanFinding(
                title=f"[{rule_id}] {description[:80]}",
                description=(
                    f"Resource: {resource}\n"
                    f"Impact: {impact}\n"
                    f"Resolution: {resolution}"
                ),
                severity=severity,
                scanner_name=self.name,
                scanner_type=ScannerType.IAC,
                rule_id=rule_id,
                location=location,
                vulnerability_info=vuln_info,
                remediation_info=remediation,
                tags={"iac", "terraform", rule_id.split("-")[0].lower()},
                raw_data=result,
            )

        except Exception as e:
            logger.warning("Failed to parse tfsec result: %s", e)
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
                    "fullDescription": {"text": finding.description[:500]},
                    "helpUri": f"https://aquasecurity.github.io/tfsec/latest/checks/{rule_id.lower()}/",
                }

            results.append(finding.to_sarif_result())

        return {
            "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
            "version": "2.1.0",
            "runs": [
                {
                    "tool": {
                        "driver": {
                            "name": "tfsec",
                            "version": self.version or "unknown",
                            "informationUri": "https://tfsec.dev/",
                            "rules": list(rules.values()),
                        }
                    },
                    "results": results,
                }
            ],
        }

    def _get_scanner_url(self) -> str:
        return "https://tfsec.dev/"


# ---------------------------------------------------------------------------
# Checkov Scanner
# ---------------------------------------------------------------------------


class CheckovScanner(BaseScanner):
    """Checkov scanner for multi-IaC security.

    Checkov scans cloud infrastructure configurations for security
    and compliance issues. Supports Terraform, CloudFormation,
    Kubernetes, ARM, and more.

    Example:
        >>> config = ScannerConfig(name="checkov", scanner_type=ScannerType.IAC)
        >>> scanner = CheckovScanner(config)
        >>> result = await scanner.scan("/path/to/infra")
    """

    SEVERITY_MAP: Dict[str, Severity] = {
        "CRITICAL": Severity.CRITICAL,
        "HIGH": Severity.HIGH,
        "MEDIUM": Severity.MEDIUM,
        "LOW": Severity.LOW,
    }

    async def scan(self, target_path: str) -> ScanResult:
        """Execute Checkov scan.

        Args:
            target_path: Path to IaC files.

        Returns:
            ScanResult with findings.
        """
        started_at = datetime.now(timezone.utc)

        if not self.is_available():
            return self._create_result(
                findings=[],
                status=ScanStatus.FAILED,
                started_at=started_at,
                error_message=f"Checkov not found: {self.config.executable}",
            )

        command = self._build_command(target_path)

        try:
            stdout, stderr, exit_code = await self._run_command(command)

            # Checkov exit codes:
            # 0 = passed
            # 1 = failed checks
            # 2 = error

            if exit_code == 2:
                return self._create_result(
                    findings=[],
                    status=ScanStatus.FAILED,
                    started_at=started_at,
                    error_message=stderr or "Checkov execution error",
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
            logger.error("Checkov scan failed: %s", e, exc_info=True)
            return self._create_result(
                findings=[],
                status=ScanStatus.FAILED,
                started_at=started_at,
                error_message=str(e),
                scan_path=target_path,
            )

    def _build_command(self, target_path: str) -> List[str]:
        """Build Checkov command.

        Args:
            target_path: Path to scan.

        Returns:
            Command list.
        """
        cmd = [self.config.executable or "checkov"]
        cmd.extend(["-d", target_path])
        cmd.extend(["--output", "json"])

        # Skip checks
        for rule in self.config.exclude_rules:
            cmd.extend(["--skip-check", rule])

        # Framework-specific (auto-detect if not specified)
        if "--framework" not in " ".join(self.config.extra_args):
            cmd.extend(["--framework", "all"])

        cmd.extend(self.config.extra_args)
        return cmd

    def parse_results(self, raw_output: str) -> List[ScanFinding]:
        """Parse Checkov JSON output.

        Args:
            raw_output: JSON output from Checkov.

        Returns:
            List of findings.
        """
        if not raw_output.strip():
            return []

        try:
            data = json.loads(raw_output)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse Checkov output: %s", e)
            return []

        findings = []

        # Checkov outputs results per check type
        if isinstance(data, list):
            for check_type_result in data:
                failed_checks = check_type_result.get("results", {}).get(
                    "failed_checks", []
                )
                for check in failed_checks:
                    finding = self._parse_check(check)
                    if finding:
                        findings.append(finding)
        else:
            # Single framework output
            failed_checks = data.get("results", {}).get("failed_checks", [])
            for check in failed_checks:
                finding = self._parse_check(check)
                if finding:
                    findings.append(finding)

        logger.info("Checkov found %d issues", len(findings))
        return findings

    def _parse_check(self, check: Dict[str, Any]) -> Optional[ScanFinding]:
        """Parse a single Checkov failed check.

        Args:
            check: Check dictionary.

        Returns:
            ScanFinding or None.
        """
        try:
            check_id = check.get("check_id", "")
            check_name = check.get("check_name", "")
            severity_str = check.get("severity", "MEDIUM")
            severity = self.SEVERITY_MAP.get(
                severity_str.upper() if severity_str else "MEDIUM",
                Severity.MEDIUM,
            )

            resource = check.get("resource", "")
            file_path = check.get("file_path", "")
            file_line_range = check.get("file_line_range", [1, 1])
            guideline = check.get("guideline", "")

            location = FileLocation(
                file_path=normalize_path(file_path.lstrip("/")),
                start_line=file_line_range[0] if file_line_range else 1,
                end_line=file_line_range[1] if len(file_line_range) > 1 else None,
            )

            vuln_info = VulnerabilityInfo(
                description=check_name,
                references=[guideline] if guideline else [],
            )

            remediation = RemediationInfo(
                description=f"See guideline: {guideline}" if guideline else "",
            )

            return ScanFinding(
                title=f"[{check_id}] {check_name}",
                description=(
                    f"Resource: {resource}\n"
                    f"Check: {check_name}\n"
                    f"File: {file_path}"
                ),
                severity=severity,
                scanner_name=self.name,
                scanner_type=ScannerType.IAC,
                rule_id=check_id,
                location=location,
                vulnerability_info=vuln_info,
                remediation_info=remediation,
                tags={"iac", "checkov", check_id.split("_")[0].lower()},
                raw_data=check,
            )

        except Exception as e:
            logger.warning("Failed to parse Checkov check: %s", e)
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
                    "fullDescription": {"text": finding.description[:500]},
                    "helpUri": "https://www.checkov.io/5.Policy%20Index/all.html",
                }

            results.append(finding.to_sarif_result())

        return {
            "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
            "version": "2.1.0",
            "runs": [
                {
                    "tool": {
                        "driver": {
                            "name": "Checkov",
                            "version": self.version or "unknown",
                            "informationUri": "https://www.checkov.io/",
                            "rules": list(rules.values()),
                        }
                    },
                    "results": results,
                }
            ],
        }

    def _get_scanner_url(self) -> str:
        return "https://www.checkov.io/"


# ---------------------------------------------------------------------------
# Kubeconform Scanner
# ---------------------------------------------------------------------------


class KubeconformScanner(BaseScanner):
    """Kubeconform scanner for Kubernetes manifest validation.

    Kubeconform validates Kubernetes manifests against schemas.
    Fast validation with support for custom resource definitions.

    Example:
        >>> config = ScannerConfig(name="kubeconform", scanner_type=ScannerType.IAC)
        >>> scanner = KubeconformScanner(config)
        >>> result = await scanner.scan("/path/to/k8s/manifests")
    """

    async def scan(self, target_path: str) -> ScanResult:
        """Execute Kubeconform validation.

        Args:
            target_path: Path to Kubernetes manifests.

        Returns:
            ScanResult with findings.
        """
        started_at = datetime.now(timezone.utc)

        if not self.is_available():
            return self._create_result(
                findings=[],
                status=ScanStatus.FAILED,
                started_at=started_at,
                error_message=f"Kubeconform not found: {self.config.executable}",
            )

        command = self._build_command(target_path)

        try:
            stdout, stderr, exit_code = await self._run_command(command)

            # Kubeconform exit codes:
            # 0 = valid
            # 1 = invalid
            # 2+ = error

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
            logger.error("Kubeconform scan failed: %s", e, exc_info=True)
            return self._create_result(
                findings=[],
                status=ScanStatus.FAILED,
                started_at=started_at,
                error_message=str(e),
                scan_path=target_path,
            )

    def _build_command(self, target_path: str) -> List[str]:
        """Build Kubeconform command.

        Args:
            target_path: Path to scan.

        Returns:
            Command list.
        """
        cmd = [self.config.executable or "kubeconform"]
        cmd.extend(["-output", "json"])
        cmd.extend(["-summary"])

        # Skip unknown resources by default
        if "-skip" not in " ".join(self.config.extra_args):
            cmd.append("-skip=CustomResourceDefinition")

        cmd.extend(self.config.extra_args)
        cmd.append(target_path)
        return cmd

    def parse_results(self, raw_output: str) -> List[ScanFinding]:
        """Parse Kubeconform JSON output.

        Args:
            raw_output: JSON output from Kubeconform.

        Returns:
            List of findings.
        """
        if not raw_output.strip():
            return []

        findings = []

        # Kubeconform outputs JSON lines
        for line in raw_output.strip().split("\n"):
            if not line.strip():
                continue

            try:
                result = json.loads(line)
                # Only process invalid resources
                if result.get("status") == "statusInvalid":
                    finding = self._parse_result(result)
                    if finding:
                        findings.append(finding)
            except json.JSONDecodeError:
                continue

        logger.info("Kubeconform found %d validation errors", len(findings))
        return findings

    def _parse_result(self, result: Dict[str, Any]) -> Optional[ScanFinding]:
        """Parse a single Kubeconform result.

        Args:
            result: Result dictionary.

        Returns:
            ScanFinding or None.
        """
        try:
            filename = result.get("filename", "")
            kind = result.get("kind", "")
            name = result.get("name", "")
            err = result.get("err", "")

            location = FileLocation(
                file_path=normalize_path(filename),
                start_line=1,
            )

            return ScanFinding(
                title=f"Invalid {kind}: {name}",
                description=(
                    f"Kubernetes manifest validation failed.\n"
                    f"Resource: {kind}/{name}\n"
                    f"Error: {err}"
                ),
                severity=Severity.MEDIUM,
                scanner_name=self.name,
                scanner_type=ScannerType.IAC,
                rule_id=f"kubeconform-{kind.lower()}",
                location=location,
                remediation_info=RemediationInfo(
                    description="Fix the manifest according to Kubernetes schema.",
                ),
                tags={"iac", "kubernetes", "kubeconform", kind.lower()},
                raw_data=result,
            )

        except Exception as e:
            logger.warning("Failed to parse Kubeconform result: %s", e)
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
                    "helpUri": "https://github.com/yannh/kubeconform",
                }

            results.append(finding.to_sarif_result())

        return {
            "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
            "version": "2.1.0",
            "runs": [
                {
                    "tool": {
                        "driver": {
                            "name": "Kubeconform",
                            "version": self.version or "unknown",
                            "informationUri": "https://github.com/yannh/kubeconform",
                            "rules": list(rules.values()),
                        }
                    },
                    "results": results,
                }
            ],
        }

    def _get_scanner_url(self) -> str:
        return "https://github.com/yannh/kubeconform"
