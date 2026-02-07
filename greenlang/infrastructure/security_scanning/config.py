# -*- coding: utf-8 -*-
"""
Security Scanning Configuration - SEC-007

Configuration dataclasses for the security scanning orchestration module.
Provides tool-specific settings, environment variable mapping, severity
thresholds, and timeout configurations.

Follows the GreenLang pattern of dataclass-based configuration with
sensible defaults and environment variable overrides.

Example:
    >>> from greenlang.infrastructure.security_scanning.config import (
    ...     ScannerConfig,
    ...     ScanOrchestratorConfig,
    ... )
    >>> config = ScanOrchestratorConfig.from_environment()
    >>> bandit_config = config.get_scanner_config("bandit")

Author: GreenLang Security Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ScannerType(str, Enum):
    """Types of security scanners supported by the orchestrator."""

    SAST = "sast"
    SCA = "sca"
    SECRETS = "secrets"
    CONTAINER = "container"
    IAC = "iac"
    DAST = "dast"
    LICENSE = "license"
    PII = "pii"


class Severity(str, Enum):
    """Normalized severity levels aligned with CVSS 3.1."""

    CRITICAL = "CRITICAL"  # CVSS 9.0-10.0
    HIGH = "HIGH"  # CVSS 7.0-8.9
    MEDIUM = "MEDIUM"  # CVSS 4.0-6.9
    LOW = "LOW"  # CVSS 0.1-3.9
    INFO = "INFO"  # Informational, no CVSS

    @classmethod
    def from_cvss(cls, score: float) -> Severity:
        """Convert CVSS score to severity level.

        Args:
            score: CVSS 3.1 score (0.0-10.0).

        Returns:
            Corresponding severity level.
        """
        if score >= 9.0:
            return cls.CRITICAL
        elif score >= 7.0:
            return cls.HIGH
        elif score >= 4.0:
            return cls.MEDIUM
        elif score > 0.0:
            return cls.LOW
        else:
            return cls.INFO


class SLAPriority(str, Enum):
    """SLA priority levels for vulnerability remediation."""

    P0 = "P0"  # 24 hours
    P1 = "P1"  # 7 days
    P2 = "P2"  # 30 days
    P3 = "P3"  # 90 days
    P4 = "P4"  # Best effort


# Severity to SLA mapping
SEVERITY_SLA_MAP: Dict[Severity, SLAPriority] = {
    Severity.CRITICAL: SLAPriority.P0,
    Severity.HIGH: SLAPriority.P1,
    Severity.MEDIUM: SLAPriority.P2,
    Severity.LOW: SLAPriority.P3,
    Severity.INFO: SLAPriority.P4,
}

# SLA days mapping
SLA_DAYS_MAP: Dict[SLAPriority, int] = {
    SLAPriority.P0: 1,
    SLAPriority.P1: 7,
    SLAPriority.P2: 30,
    SLAPriority.P3: 90,
    SLAPriority.P4: 365,
}


# ---------------------------------------------------------------------------
# Scanner Configuration
# ---------------------------------------------------------------------------


@dataclass
class ScannerConfig:
    """Configuration for an individual security scanner.

    Attributes:
        name: Scanner identifier (e.g., "bandit", "trivy", "gitleaks").
        scanner_type: Type category of the scanner.
        enabled: Whether this scanner is enabled.
        executable: Path to the scanner executable or command name.
        timeout_seconds: Maximum execution time for the scanner.
        severity_threshold: Minimum severity to report (filters lower).
        exclude_paths: Paths to exclude from scanning.
        exclude_rules: Rule IDs to exclude (false positive management).
        extra_args: Additional command-line arguments.
        environment: Environment variables to set during execution.
        config_file: Path to scanner-specific configuration file.
        output_format: Preferred output format (json, sarif, etc.).
    """

    name: str
    scanner_type: ScannerType
    enabled: bool = True
    executable: Optional[str] = None
    timeout_seconds: int = 300  # 5 minutes default
    severity_threshold: Severity = Severity.LOW
    exclude_paths: List[str] = field(default_factory=list)
    exclude_rules: List[str] = field(default_factory=list)
    extra_args: List[str] = field(default_factory=list)
    environment: Dict[str, str] = field(default_factory=dict)
    config_file: Optional[str] = None
    output_format: str = "json"

    def __post_init__(self) -> None:
        """Set default executable if not provided."""
        if self.executable is None:
            self.executable = self.name

    def get_command(self) -> List[str]:
        """Get the base command for running the scanner.

        Returns:
            List of command components.
        """
        if self.executable is None:
            return [self.name]
        return [self.executable]


# ---------------------------------------------------------------------------
# Default Scanner Configurations
# ---------------------------------------------------------------------------

DEFAULT_EXCLUDE_PATHS: List[str] = [
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "build",
    "dist",
    ".hypothesis",
    "htmlcov",
    ".coverage_html",
]


def _get_default_scanner_configs() -> Dict[str, ScannerConfig]:
    """Get default configurations for all supported scanners.

    Returns:
        Dictionary mapping scanner name to configuration.
    """
    return {
        # SAST Scanners
        "bandit": ScannerConfig(
            name="bandit",
            scanner_type=ScannerType.SAST,
            executable="bandit",
            timeout_seconds=300,
            exclude_paths=DEFAULT_EXCLUDE_PATHS + ["tests"],
            config_file=".bandit",
            output_format="json",
        ),
        "semgrep": ScannerConfig(
            name="semgrep",
            scanner_type=ScannerType.SAST,
            executable="semgrep",
            timeout_seconds=600,
            exclude_paths=DEFAULT_EXCLUDE_PATHS,
            extra_args=["--config", "auto", "--sarif"],
            output_format="sarif",
        ),
        "codeql": ScannerConfig(
            name="codeql",
            scanner_type=ScannerType.SAST,
            executable="codeql",
            timeout_seconds=1800,  # 30 minutes
            enabled=False,  # Requires database creation
            output_format="sarif",
        ),
        # SCA Scanners
        "trivy": ScannerConfig(
            name="trivy",
            scanner_type=ScannerType.SCA,
            executable="trivy",
            timeout_seconds=300,
            extra_args=["fs", "--scanners", "vuln,misconfig"],
            output_format="json",
        ),
        "snyk": ScannerConfig(
            name="snyk",
            scanner_type=ScannerType.SCA,
            executable="snyk",
            timeout_seconds=300,
            environment={"SNYK_TOKEN": os.environ.get("SNYK_TOKEN", "")},
            output_format="json",
        ),
        "pip-audit": ScannerConfig(
            name="pip-audit",
            scanner_type=ScannerType.SCA,
            executable="pip-audit",
            timeout_seconds=300,
            output_format="json",
        ),
        "safety": ScannerConfig(
            name="safety",
            scanner_type=ScannerType.SCA,
            executable="safety",
            timeout_seconds=300,
            output_format="json",
        ),
        # Secret Scanners
        "gitleaks": ScannerConfig(
            name="gitleaks",
            scanner_type=ScannerType.SECRETS,
            executable="gitleaks",
            timeout_seconds=300,
            config_file=".gitleaks.toml",
            extra_args=["detect", "--no-git"],
            output_format="json",
        ),
        "trufflehog": ScannerConfig(
            name="trufflehog",
            scanner_type=ScannerType.SECRETS,
            executable="trufflehog",
            timeout_seconds=300,
            extra_args=["filesystem", "--json"],
            output_format="json",
        ),
        "detect-secrets": ScannerConfig(
            name="detect-secrets",
            scanner_type=ScannerType.SECRETS,
            executable="detect-secrets",
            timeout_seconds=300,
            extra_args=["scan"],
            output_format="json",
        ),
        # Container Scanners
        "trivy-container": ScannerConfig(
            name="trivy-container",
            scanner_type=ScannerType.CONTAINER,
            executable="trivy",
            timeout_seconds=600,
            extra_args=["image", "--scanners", "vuln"],
            output_format="json",
        ),
        "grype": ScannerConfig(
            name="grype",
            scanner_type=ScannerType.CONTAINER,
            executable="grype",
            timeout_seconds=300,
            output_format="json",
        ),
        "cosign": ScannerConfig(
            name="cosign",
            scanner_type=ScannerType.CONTAINER,
            executable="cosign",
            timeout_seconds=60,
            extra_args=["verify"],
            output_format="json",
        ),
        # IaC Scanners
        "tfsec": ScannerConfig(
            name="tfsec",
            scanner_type=ScannerType.IAC,
            executable="tfsec",
            timeout_seconds=300,
            extra_args=["--format", "json"],
            output_format="json",
        ),
        "checkov": ScannerConfig(
            name="checkov",
            scanner_type=ScannerType.IAC,
            executable="checkov",
            timeout_seconds=600,
            extra_args=["--output", "json"],
            output_format="json",
        ),
        "kubeconform": ScannerConfig(
            name="kubeconform",
            scanner_type=ScannerType.IAC,
            executable="kubeconform",
            timeout_seconds=120,
            extra_args=["-output", "json", "-summary"],
            output_format="json",
        ),
        # DAST Scanner
        "zap": ScannerConfig(
            name="zap",
            scanner_type=ScannerType.DAST,
            executable="zap.sh",
            timeout_seconds=1800,  # 30 minutes for full scan
            enabled=False,  # Requires target URL
            output_format="json",
        ),
    }


# ---------------------------------------------------------------------------
# Orchestrator Configuration
# ---------------------------------------------------------------------------


@dataclass
class ScanOrchestratorConfig:
    """Configuration for the security scan orchestrator.

    Attributes:
        scan_path: Root path to scan.
        scanners: Dictionary of scanner configurations.
        enabled_scanner_types: Set of scanner types to run.
        parallel_scans: Maximum number of parallel scanner executions.
        global_timeout_seconds: Maximum total scan time.
        fail_on_severity: Fail the scan if findings at or above this severity.
        deduplication_enabled: Enable CVE-based deduplication.
        sarif_output_path: Path for SARIF output file.
        json_output_path: Path for JSON output file.
        html_output_path: Path for HTML report output.
        upload_to_github: Upload SARIF to GitHub Security tab.
        enable_metrics: Emit Prometheus metrics.
        enable_audit: Log scan operations for audit trail.
        cache_results: Cache scan results for incremental scanning.
        cache_ttl_seconds: TTL for cached results.
    """

    scan_path: str = "."
    scanners: Dict[str, ScannerConfig] = field(default_factory=dict)
    enabled_scanner_types: Set[ScannerType] = field(
        default_factory=lambda: {
            ScannerType.SAST,
            ScannerType.SCA,
            ScannerType.SECRETS,
        }
    )
    parallel_scans: int = 4
    global_timeout_seconds: int = 1800  # 30 minutes
    fail_on_severity: Severity = Severity.HIGH
    deduplication_enabled: bool = True
    sarif_output_path: Optional[str] = None
    json_output_path: Optional[str] = None
    html_output_path: Optional[str] = None
    upload_to_github: bool = False
    enable_metrics: bool = True
    enable_audit: bool = True
    cache_results: bool = False
    cache_ttl_seconds: int = 3600  # 1 hour

    def __post_init__(self) -> None:
        """Initialize default scanner configurations if not provided."""
        if not self.scanners:
            self.scanners = _get_default_scanner_configs()

    @classmethod
    def from_environment(cls) -> ScanOrchestratorConfig:
        """Create configuration from environment variables.

        Environment variables:
            GL_SECURITY_SCAN_PATH: Root path to scan
            GL_SECURITY_PARALLEL_SCANS: Max parallel scans
            GL_SECURITY_TIMEOUT: Global timeout in seconds
            GL_SECURITY_FAIL_SEVERITY: Severity to fail on
            GL_SECURITY_SARIF_PATH: SARIF output path
            GL_SECURITY_JSON_PATH: JSON output path
            GL_SECURITY_HTML_PATH: HTML output path
            GL_SECURITY_UPLOAD_GITHUB: Upload to GitHub (true/false)
            GL_SECURITY_DEDUP_ENABLED: Enable deduplication (true/false)
            GL_SECURITY_METRICS: Enable metrics (true/false)

        Returns:
            Configuration populated from environment.
        """
        def _bool_env(key: str, default: bool) -> bool:
            val = os.environ.get(key, "").lower()
            if val in ("true", "1", "yes"):
                return True
            elif val in ("false", "0", "no"):
                return False
            return default

        fail_severity_str = os.environ.get(
            "GL_SECURITY_FAIL_SEVERITY", "HIGH"
        ).upper()
        try:
            fail_severity = Severity(fail_severity_str)
        except ValueError:
            fail_severity = Severity.HIGH
            logger.warning(
                "Invalid GL_SECURITY_FAIL_SEVERITY '%s', using HIGH",
                fail_severity_str,
            )

        return cls(
            scan_path=os.environ.get("GL_SECURITY_SCAN_PATH", "."),
            parallel_scans=int(os.environ.get("GL_SECURITY_PARALLEL_SCANS", "4")),
            global_timeout_seconds=int(
                os.environ.get("GL_SECURITY_TIMEOUT", "1800")
            ),
            fail_on_severity=fail_severity,
            deduplication_enabled=_bool_env("GL_SECURITY_DEDUP_ENABLED", True),
            sarif_output_path=os.environ.get("GL_SECURITY_SARIF_PATH"),
            json_output_path=os.environ.get("GL_SECURITY_JSON_PATH"),
            html_output_path=os.environ.get("GL_SECURITY_HTML_PATH"),
            upload_to_github=_bool_env("GL_SECURITY_UPLOAD_GITHUB", False),
            enable_metrics=_bool_env("GL_SECURITY_METRICS", True),
            enable_audit=_bool_env("GL_SECURITY_AUDIT", True),
        )

    def get_scanner_config(self, scanner_name: str) -> Optional[ScannerConfig]:
        """Get configuration for a specific scanner.

        Args:
            scanner_name: Name of the scanner.

        Returns:
            Scanner configuration or None if not found.
        """
        return self.scanners.get(scanner_name)

    def get_enabled_scanners(self) -> List[ScannerConfig]:
        """Get list of enabled scanners filtered by type.

        Returns:
            List of enabled scanner configurations.
        """
        return [
            config
            for config in self.scanners.values()
            if config.enabled and config.scanner_type in self.enabled_scanner_types
        ]

    def get_scanners_by_type(
        self, scanner_type: ScannerType
    ) -> List[ScannerConfig]:
        """Get all scanners of a specific type.

        Args:
            scanner_type: Type of scanners to retrieve.

        Returns:
            List of scanner configurations of the specified type.
        """
        return [
            config
            for config in self.scanners.values()
            if config.scanner_type == scanner_type and config.enabled
        ]

    def enable_scanner(self, scanner_name: str) -> None:
        """Enable a specific scanner.

        Args:
            scanner_name: Name of the scanner to enable.
        """
        if scanner_name in self.scanners:
            self.scanners[scanner_name].enabled = True
            logger.info("Scanner '%s' enabled", scanner_name)

    def disable_scanner(self, scanner_name: str) -> None:
        """Disable a specific scanner.

        Args:
            scanner_name: Name of the scanner to disable.
        """
        if scanner_name in self.scanners:
            self.scanners[scanner_name].enabled = False
            logger.info("Scanner '%s' disabled", scanner_name)

    def set_scanner_timeout(
        self, scanner_name: str, timeout_seconds: int
    ) -> None:
        """Set timeout for a specific scanner.

        Args:
            scanner_name: Name of the scanner.
            timeout_seconds: Timeout in seconds.
        """
        if scanner_name in self.scanners:
            self.scanners[scanner_name].timeout_seconds = timeout_seconds

    def add_exclude_path(self, path: str, scanner_name: Optional[str] = None) -> None:
        """Add a path to exclude from scanning.

        Args:
            path: Path to exclude.
            scanner_name: Specific scanner, or None for all scanners.
        """
        if scanner_name:
            if scanner_name in self.scanners:
                if path not in self.scanners[scanner_name].exclude_paths:
                    self.scanners[scanner_name].exclude_paths.append(path)
        else:
            for config in self.scanners.values():
                if path not in config.exclude_paths:
                    config.exclude_paths.append(path)

    def add_exclude_rule(
        self, rule_id: str, scanner_name: Optional[str] = None
    ) -> None:
        """Add a rule ID to exclude (false positive management).

        Args:
            rule_id: Rule ID to exclude.
            scanner_name: Specific scanner, or None for all scanners.
        """
        if scanner_name:
            if scanner_name in self.scanners:
                if rule_id not in self.scanners[scanner_name].exclude_rules:
                    self.scanners[scanner_name].exclude_rules.append(rule_id)
        else:
            for config in self.scanners.values():
                if rule_id not in config.exclude_rules:
                    config.exclude_rules.append(rule_id)
