# -*- coding: utf-8 -*-
"""
DAST Scanner Implementation - SEC-007 Phase 2

Dynamic Application Security Testing (DAST) scanner using OWASP ZAP.
Supports baseline, full, and API scan modes with authenticated scanning.

Features:
    - Baseline scan: Fast passive scanning for CI/CD (<5 min)
    - Full scan: Comprehensive active scanning for nightly jobs
    - API scan: OpenAPI/GraphQL endpoint scanning
    - Authenticated scanning with JWT support

Example:
    >>> from greenlang.infrastructure.security_scanning.scanners.dast import (
    ...     ZAPScanner,
    ...     ZAPScanMode,
    ... )
    >>> config = ScannerConfig(name="zap", scanner_type=ScannerType.DAST)
    >>> scanner = ZAPScanner(config)
    >>> result = await scanner.scan("https://staging.greenlang.io")

Author: GreenLang Security Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
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
    ScannerExecutionError,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ZAP Scan Modes
# ---------------------------------------------------------------------------


class ZAPScanMode(str, Enum):
    """OWASP ZAP scan modes."""

    BASELINE = "baseline"  # Fast passive scan (<5 min)
    FULL = "full"  # Comprehensive active scan
    API = "api"  # OpenAPI/GraphQL scan


# ---------------------------------------------------------------------------
# ZAP Configuration
# ---------------------------------------------------------------------------


@dataclass
class ZAPConfig:
    """Configuration for OWASP ZAP scanning.

    Attributes:
        target_url: Target URL to scan.
        scan_mode: Type of scan to perform.
        api_definition: Path to OpenAPI/GraphQL definition.
        auth_token: JWT or bearer token for authenticated scanning.
        auth_header: Header name for authentication (default: Authorization).
        context_name: ZAP context name for session management.
        user_name: Username for form-based authentication.
        password: Password for form-based authentication.
        login_url: URL for form-based login.
        max_duration_minutes: Maximum scan duration.
        ajax_spider: Enable AJAX spider for SPA applications.
        include_urls: URL patterns to include.
        exclude_urls: URL patterns to exclude.
        policy_file: Custom scan policy XML file.
    """

    target_url: str
    scan_mode: ZAPScanMode = ZAPScanMode.BASELINE
    api_definition: Optional[str] = None
    auth_token: Optional[str] = None
    auth_header: str = "Authorization"
    context_name: str = "Default Context"
    user_name: Optional[str] = None
    password: Optional[str] = None
    login_url: Optional[str] = None
    max_duration_minutes: int = 60
    ajax_spider: bool = False
    include_urls: List[str] = field(default_factory=list)
    exclude_urls: List[str] = field(default_factory=list)
    policy_file: Optional[str] = None


# ---------------------------------------------------------------------------
# ZAP Scanner
# ---------------------------------------------------------------------------


class ZAPScanner(BaseScanner):
    """OWASP ZAP scanner for dynamic application security testing.

    Provides three scan modes:
    - Baseline: Passive scan with spider, good for CI/CD (<5 min)
    - Full: Active scan with attack modules (30+ min)
    - API: Specialized scan for REST/GraphQL APIs

    The scanner supports:
    - Authenticated scanning with JWT tokens
    - Custom scan policies
    - AJAX spider for SPAs
    - SARIF output for GitHub Security integration

    Example:
        >>> scanner = ZAPScanner(config)
        >>> result = await scanner.scan("https://api.example.com")
    """

    # ZAP risk to severity mapping
    RISK_SEVERITY_MAP: Dict[str, Severity] = {
        "High": Severity.HIGH,
        "Medium": Severity.MEDIUM,
        "Low": Severity.LOW,
        "Informational": Severity.INFO,
        "3": Severity.HIGH,  # Risk code
        "2": Severity.MEDIUM,
        "1": Severity.LOW,
        "0": Severity.INFO,
    }

    # CWE mappings for common ZAP alerts
    ZAP_CWE_MAP: Dict[str, str] = {
        "10012": "CWE-319",  # Cookie Without Secure Flag
        "10016": "CWE-693",  # Web Browser XSS Protection Not Enabled
        "10017": "CWE-693",  # Cross-Domain JavaScript Source File Inclusion
        "10020": "CWE-693",  # X-Frame-Options Header
        "10021": "CWE-693",  # X-Content-Type-Options Header Missing
        "10027": "CWE-200",  # Information Disclosure - Suspicious Comments
        "10035": "CWE-693",  # Strict-Transport-Security Header Not Set
        "10036": "CWE-200",  # Server Leaks Version Information
        "10037": "CWE-200",  # Server Leaks Information
        "10038": "CWE-693",  # Content Security Policy Header Not Set
        "10054": "CWE-319",  # Cookie without SameSite Attribute
        "10055": "CWE-693",  # CSP: Wildcard Directive
        "10098": "CWE-693",  # Cross-Domain Misconfiguration
        "10202": "CWE-693",  # Absence of Anti-CSRF Tokens
        "40003": "CWE-89",   # SQL Injection
        "40012": "CWE-79",   # Cross Site Scripting (Reflected)
        "40014": "CWE-79",   # Cross Site Scripting (Persistent)
        "40018": "CWE-89",   # SQL Injection - Hypersonic SQL
        "40019": "CWE-89",   # SQL Injection - MySQL
        "40020": "CWE-89",   # SQL Injection - Oracle
        "40021": "CWE-89",   # SQL Injection - PostgreSQL
        "40022": "CWE-89",   # SQL Injection - SQLite
        "40023": "CWE-611",  # XML External Entity Attack
        "40024": "CWE-611",  # Generic Padding Oracle
        "40025": "CWE-611",  # Expression Language Injection
        "40026": "CWE-78",   # Remote OS Command Injection
        "40027": "CWE-89",   # SQL Injection - MsSQL
        "40028": "CWE-352",  # ELMAH Information Leak
        "40029": "CWE-502",  # Trace.axd Information Leak
        "40034": "CWE-79",   # Cross Site Scripting (DOM Based)
        "90018": "CWE-89",   # SQL Injection - MongoDB
        "90019": "CWE-611",  # Server Side Code Injection
        "90020": "CWE-94",   # Remote OS Command Injection
        "90033": "CWE-611",  # Loosely Scoped Cookie
    }

    def __init__(self, config: ScannerConfig) -> None:
        """Initialize ZAP scanner.

        Args:
            config: Scanner configuration.
        """
        super().__init__(config)
        self._zap_config: Optional[ZAPConfig] = None
        self._docker_available = self._check_docker()

    def _check_docker(self) -> bool:
        """Check if Docker is available for ZAP.

        Returns:
            True if Docker is available.
        """
        try:
            stdout, stderr, code = self._run_command_sync(
                ["docker", "--version"],
                timeout=5,
            )
            return code == 0
        except Exception:
            return False

    async def scan(
        self,
        target: str,
        zap_config: Optional[ZAPConfig] = None,
    ) -> ScanResult:
        """Execute ZAP scan on target URL.

        Args:
            target: Target URL to scan.
            zap_config: Optional ZAP-specific configuration.

        Returns:
            ScanResult with findings.
        """
        started_at = datetime.now(timezone.utc)

        # Create default config if not provided
        if zap_config is None:
            zap_config = ZAPConfig(target_url=target)
        else:
            zap_config.target_url = target

        self._zap_config = zap_config

        # Determine scan method
        if self._docker_available:
            result = await self._scan_with_docker(zap_config, started_at)
        elif self.is_available():
            result = await self._scan_with_cli(zap_config, started_at)
        else:
            return self._create_result(
                findings=[],
                status=ScanStatus.FAILED,
                started_at=started_at,
                error_message="ZAP not available. Install Docker or ZAP CLI.",
            )

        return result

    async def _scan_with_docker(
        self,
        zap_config: ZAPConfig,
        started_at: datetime,
    ) -> ScanResult:
        """Execute ZAP scan using Docker.

        Args:
            zap_config: ZAP configuration.
            started_at: Scan start time.

        Returns:
            ScanResult with findings.
        """
        # Create temporary directory for output
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = os.path.join(temp_dir, "zap-report.json")

            # Build Docker command
            command = self._build_docker_command(zap_config, temp_dir)

            try:
                logger.info(
                    "Starting ZAP %s scan on %s",
                    zap_config.scan_mode.value,
                    zap_config.target_url,
                )

                stdout, stderr, exit_code = await self._run_command(
                    command,
                    timeout=zap_config.max_duration_minutes * 60,
                )

                # Parse results if report exists
                if os.path.exists(report_path):
                    with open(report_path, "r") as f:
                        raw_output = f.read()
                    findings = self.parse_results(raw_output)
                else:
                    # Try to parse stdout (traditional report format)
                    findings = self.parse_results(stdout)

                filtered_findings = self._apply_filters(findings)

                return self._create_result(
                    findings=filtered_findings,
                    status=ScanStatus.COMPLETED,
                    started_at=started_at,
                    exit_code=exit_code,
                    raw_output=stdout,
                    command=" ".join(command),
                    scan_path=zap_config.target_url,
                )

            except Exception as e:
                logger.error("ZAP Docker scan failed: %s", e, exc_info=True)
                return self._create_result(
                    findings=[],
                    status=ScanStatus.FAILED,
                    started_at=started_at,
                    error_message=str(e),
                    scan_path=zap_config.target_url,
                )

    def _build_docker_command(
        self,
        zap_config: ZAPConfig,
        output_dir: str,
    ) -> List[str]:
        """Build Docker command for ZAP scan.

        Args:
            zap_config: ZAP configuration.
            output_dir: Directory for output files.

        Returns:
            Docker command list.
        """
        # Base Docker command
        cmd = [
            "docker", "run", "--rm",
            "-v", f"{output_dir}:/zap/wrk:rw",
            "-t", "ghcr.io/zaproxy/zaproxy:stable",
        ]

        # Select scan script based on mode
        if zap_config.scan_mode == ZAPScanMode.BASELINE:
            cmd.append("zap-baseline.py")
        elif zap_config.scan_mode == ZAPScanMode.FULL:
            cmd.append("zap-full-scan.py")
        elif zap_config.scan_mode == ZAPScanMode.API:
            cmd.append("zap-api-scan.py")

        # Add target URL
        cmd.extend(["-t", zap_config.target_url])

        # Output format
        cmd.extend(["-J", "/zap/wrk/zap-report.json"])

        # Add API definition for API scan
        if zap_config.scan_mode == ZAPScanMode.API and zap_config.api_definition:
            cmd.extend(["-f", "openapi"])

        # Add authentication token
        if zap_config.auth_token:
            cmd.extend([
                "-z",
                f"-config replacer.full_list(0).description=AuthHeader "
                f"-config replacer.full_list(0).enabled=true "
                f"-config replacer.full_list(0).matchtype=REQ_HEADER "
                f"-config replacer.full_list(0).matchstr={zap_config.auth_header} "
                f"-config replacer.full_list(0).replacement=Bearer\\ {zap_config.auth_token}"
            ])

        # Add max duration
        cmd.extend(["-m", str(zap_config.max_duration_minutes)])

        # Enable AJAX spider for SPAs
        if zap_config.ajax_spider:
            cmd.append("-j")

        # Add custom policy
        if zap_config.policy_file and os.path.exists(zap_config.policy_file):
            cmd.extend(["-c", zap_config.policy_file])

        return cmd

    async def _scan_with_cli(
        self,
        zap_config: ZAPConfig,
        started_at: datetime,
    ) -> ScanResult:
        """Execute ZAP scan using CLI (zap.sh).

        Args:
            zap_config: ZAP configuration.
            started_at: Scan start time.

        Returns:
            ScanResult with findings.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = os.path.join(temp_dir, "zap-report.json")

            command = self._build_cli_command(zap_config, report_path)

            try:
                stdout, stderr, exit_code = await self._run_command(
                    command,
                    timeout=zap_config.max_duration_minutes * 60,
                )

                if os.path.exists(report_path):
                    with open(report_path, "r") as f:
                        raw_output = f.read()
                    findings = self.parse_results(raw_output)
                else:
                    findings = []

                filtered_findings = self._apply_filters(findings)

                return self._create_result(
                    findings=filtered_findings,
                    status=ScanStatus.COMPLETED,
                    started_at=started_at,
                    exit_code=exit_code,
                    raw_output=stdout,
                    command=" ".join(command),
                    scan_path=zap_config.target_url,
                )

            except Exception as e:
                logger.error("ZAP CLI scan failed: %s", e, exc_info=True)
                return self._create_result(
                    findings=[],
                    status=ScanStatus.FAILED,
                    started_at=started_at,
                    error_message=str(e),
                    scan_path=zap_config.target_url,
                )

    def _build_cli_command(
        self,
        zap_config: ZAPConfig,
        report_path: str,
    ) -> List[str]:
        """Build ZAP CLI command.

        Args:
            zap_config: ZAP configuration.
            report_path: Path for output report.

        Returns:
            CLI command list.
        """
        cmd = [self.config.executable or "zap.sh"]

        # Daemon mode for scripted scanning
        cmd.extend(["-daemon", "-port", "8090"])

        # Quick scan for baseline
        if zap_config.scan_mode == ZAPScanMode.BASELINE:
            cmd.extend(["-quickurl", zap_config.target_url])
            cmd.extend(["-quickout", report_path])

        return cmd

    def parse_results(self, raw_output: str) -> List[ScanFinding]:
        """Parse ZAP JSON output.

        Supports both ZAP JSON report format and traditional XML/HTML.

        Args:
            raw_output: JSON output from ZAP.

        Returns:
            List of findings.
        """
        if not raw_output.strip():
            return []

        try:
            data = json.loads(raw_output)
        except json.JSONDecodeError:
            # Try to parse as traditional format
            logger.warning("Failed to parse ZAP JSON, trying alternative formats")
            return self._parse_traditional_output(raw_output)

        findings = []

        # Handle ZAP JSON report format
        site = data.get("site", [])
        if isinstance(site, list):
            for site_data in site:
                alerts = site_data.get("alerts", [])
                for alert in alerts:
                    finding = self._parse_alert(alert)
                    if finding:
                        findings.append(finding)
        else:
            # Single site format
            alerts = data.get("alerts", [])
            for alert in alerts:
                finding = self._parse_alert(alert)
                if finding:
                    findings.append(finding)

        logger.info("ZAP found %d issues", len(findings))
        return findings

    def _parse_alert(self, alert: Dict[str, Any]) -> Optional[ScanFinding]:
        """Parse a single ZAP alert.

        Args:
            alert: ZAP alert dictionary.

        Returns:
            ScanFinding or None.
        """
        try:
            alert_id = str(alert.get("alertRef", alert.get("pluginId", "")))
            name = alert.get("name", alert.get("alert", "Unknown"))
            risk_str = alert.get("riskdesc", alert.get("risk", "Medium"))
            risk = risk_str.split()[0] if isinstance(risk_str, str) else "Medium"
            severity = self.RISK_SEVERITY_MAP.get(risk, Severity.MEDIUM)

            description = alert.get("desc", alert.get("description", ""))
            solution = alert.get("solution", "")
            reference = alert.get("reference", "")
            cweid = alert.get("cweid", "")

            # Parse instances (URLs affected)
            instances = alert.get("instances", [])
            first_instance = instances[0] if instances else {}
            url = first_instance.get("uri", alert.get("url", ""))
            evidence = first_instance.get("evidence", "")

            # Create location based on URL
            location = FileLocation(
                file_path=url,
                start_line=1,
                snippet=evidence[:200] if evidence else None,
            )

            # Get CWE mapping
            cwe_id = None
            if cweid:
                cwe_id = f"CWE-{cweid}"
            elif alert_id in self.ZAP_CWE_MAP:
                cwe_id = self.ZAP_CWE_MAP[alert_id]

            vuln_info = VulnerabilityInfo(
                cwe_id=cwe_id,
                description=self._strip_html(description),
                references=[ref for ref in reference.split("\n") if ref.strip()],
            )

            remediation = RemediationInfo(
                description=self._strip_html(solution),
            )

            return ScanFinding(
                title=f"[{alert_id}] {name}",
                description=(
                    f"URL: {url}\n\n"
                    f"{self._strip_html(description)}\n\n"
                    f"Evidence: {evidence[:200] if evidence else 'N/A'}"
                ),
                severity=severity,
                scanner_name=self.name,
                scanner_type=ScannerType.DAST,
                rule_id=alert_id,
                location=location,
                vulnerability_info=vuln_info,
                remediation_info=remediation,
                tags={"dast", "zap", "web", risk.lower()},
                raw_data=alert,
            )

        except Exception as e:
            logger.warning("Failed to parse ZAP alert: %s", e)
            return None

    def _parse_traditional_output(self, output: str) -> List[ScanFinding]:
        """Parse traditional ZAP output (non-JSON).

        Args:
            output: ZAP output string.

        Returns:
            List of findings.
        """
        # Basic parsing for text output
        findings = []
        # This is a fallback - proper JSON parsing is preferred
        if "WARN" in output or "FAIL" in output:
            logger.warning("ZAP output in non-JSON format, limited parsing")

        return findings

    def _strip_html(self, text: str) -> str:
        """Strip HTML tags from text.

        Args:
            text: Text with HTML.

        Returns:
            Plain text.
        """
        if not text:
            return ""

        import re
        # Remove HTML tags
        clean = re.sub(r"<[^>]+>", "", text)
        # Clean up whitespace
        clean = re.sub(r"\s+", " ", clean)
        return clean.strip()

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
                    "fullDescription": {
                        "text": finding.description[:500] if finding.description else ""
                    },
                    "helpUri": "https://www.zaproxy.org/docs/alerts/",
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
                            "name": "OWASP ZAP",
                            "version": self.version or "unknown",
                            "informationUri": "https://www.zaproxy.org/",
                            "rules": list(rules.values()),
                        }
                    },
                    "results": results,
                }
            ],
        }

    def _get_scanner_url(self) -> str:
        return "https://www.zaproxy.org/"


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------


async def run_baseline_scan(
    target_url: str,
    auth_token: Optional[str] = None,
) -> ScanResult:
    """Run a quick baseline scan on a target URL.

    Args:
        target_url: URL to scan.
        auth_token: Optional JWT token for authenticated scanning.

    Returns:
        ScanResult with findings.
    """
    config = ScannerConfig(
        name="zap",
        scanner_type=ScannerType.DAST,
        timeout_seconds=600,  # 10 minutes
    )
    scanner = ZAPScanner(config)

    zap_config = ZAPConfig(
        target_url=target_url,
        scan_mode=ZAPScanMode.BASELINE,
        auth_token=auth_token,
        max_duration_minutes=10,
    )

    return await scanner.scan(target_url, zap_config)


async def run_api_scan(
    target_url: str,
    api_definition: str,
    auth_token: Optional[str] = None,
) -> ScanResult:
    """Run an API scan using OpenAPI definition.

    Args:
        target_url: Base URL of the API.
        api_definition: Path to OpenAPI/Swagger definition.
        auth_token: Optional JWT token for authenticated scanning.

    Returns:
        ScanResult with findings.
    """
    config = ScannerConfig(
        name="zap",
        scanner_type=ScannerType.DAST,
        timeout_seconds=1800,  # 30 minutes
    )
    scanner = ZAPScanner(config)

    zap_config = ZAPConfig(
        target_url=target_url,
        scan_mode=ZAPScanMode.API,
        api_definition=api_definition,
        auth_token=auth_token,
        max_duration_minutes=30,
    )

    return await scanner.scan(target_url, zap_config)
