# -*- coding: utf-8 -*-
"""
Security Scanning Models - SEC-007

Data models for security scan results, findings, vulnerabilities, and reports.
All models are immutable dataclasses for thread-safety and audit compliance.

Follows SARIF 2.1.0 schema for interoperability with GitHub Security tab
and other security tools.

Example:
    >>> from greenlang.infrastructure.security_scanning.models import (
    ...     ScanFinding,
    ...     ScanResult,
    ...     ScanReport,
    ... )
    >>> finding = ScanFinding(
    ...     finding_id="CVE-2024-1234",
    ...     severity=Severity.HIGH,
    ...     title="SQL Injection Vulnerability",
    ...     ...
    ... )

Author: GreenLang Security Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

from greenlang.infrastructure.security_scanning.config import (
    Severity,
    ScannerType,
    SLAPriority,
    SEVERITY_SLA_MAP,
    SLA_DAYS_MAP,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Status Enums
# ---------------------------------------------------------------------------


class FindingStatus(str, Enum):
    """Status of a security finding."""

    OPEN = "open"
    IN_PROGRESS = "in_progress"
    FIXED = "fixed"
    ACCEPTED = "accepted"  # Risk accepted
    FALSE_POSITIVE = "false_positive"
    DUPLICATE = "duplicate"
    WONT_FIX = "wont_fix"


class ScanStatus(str, Enum):
    """Status of a security scan."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


# ---------------------------------------------------------------------------
# Location Models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FileLocation:
    """Location of a finding within a file.

    Attributes:
        file_path: Relative path to the file.
        start_line: Starting line number (1-indexed).
        end_line: Ending line number (1-indexed).
        start_column: Starting column (1-indexed).
        end_column: Ending column (1-indexed).
        snippet: Code snippet showing the vulnerability.
    """

    file_path: str
    start_line: int = 1
    end_line: Optional[int] = None
    start_column: Optional[int] = None
    end_column: Optional[int] = None
    snippet: Optional[str] = None

    def __post_init__(self) -> None:
        """Set end_line to start_line if not provided."""
        if self.end_line is None:
            object.__setattr__(self, "end_line", self.start_line)

    def to_sarif_location(self) -> Dict[str, Any]:
        """Convert to SARIF location format.

        Returns:
            SARIF-compatible location dictionary.
        """
        region: Dict[str, Any] = {"startLine": self.start_line}

        if self.end_line and self.end_line != self.start_line:
            region["endLine"] = self.end_line
        if self.start_column:
            region["startColumn"] = self.start_column
        if self.end_column:
            region["endColumn"] = self.end_column
        if self.snippet:
            region["snippet"] = {"text": self.snippet}

        return {
            "physicalLocation": {
                "artifactLocation": {
                    "uri": self.file_path,
                    "uriBaseId": "%SRCROOT%",
                },
                "region": region,
            }
        }


@dataclass(frozen=True)
class ContainerLocation:
    """Location of a finding within a container image.

    Attributes:
        image_ref: Container image reference (e.g., "repo/image:tag").
        layer_digest: Digest of the specific layer.
        layer_index: Index of the layer in the image.
        package_path: Path to the affected package within the layer.
    """

    image_ref: str
    layer_digest: Optional[str] = None
    layer_index: Optional[int] = None
    package_path: Optional[str] = None


# ---------------------------------------------------------------------------
# Vulnerability Information
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VulnerabilityInfo:
    """Detailed information about a vulnerability.

    Attributes:
        cve_id: CVE identifier (e.g., "CVE-2024-1234").
        cwe_id: CWE identifier (e.g., "CWE-89").
        cvss_score: CVSS 3.1 base score (0.0-10.0).
        cvss_vector: CVSS 3.1 vector string.
        epss_score: EPSS probability score (0.0-1.0).
        epss_percentile: EPSS percentile (0.0-100.0).
        is_kev: Whether in CISA KEV (Known Exploited Vulnerabilities).
        description: Human-readable description.
        references: List of reference URLs.
        published_date: When the CVE was published.
        modified_date: Last modification date.
    """

    cve_id: Optional[str] = None
    cwe_id: Optional[str] = None
    cvss_score: Optional[float] = None
    cvss_vector: Optional[str] = None
    epss_score: Optional[float] = None
    epss_percentile: Optional[float] = None
    is_kev: bool = False
    description: Optional[str] = None
    references: List[str] = field(default_factory=list)
    published_date: Optional[datetime] = None
    modified_date: Optional[datetime] = None

    def get_risk_score(self) -> float:
        """Calculate composite risk score considering multiple factors.

        Combines CVSS, EPSS, and KEV status for prioritization.

        Returns:
            Risk score (0.0-10.0).
        """
        base_score = self.cvss_score or 0.0

        # Boost for high EPSS (exploitability)
        if self.epss_score:
            if self.epss_score > 0.5:
                base_score = min(10.0, base_score + 2.0)
            elif self.epss_score > 0.2:
                base_score = min(10.0, base_score + 1.0)

        # Significant boost for KEV status
        if self.is_kev:
            base_score = min(10.0, base_score + 2.5)

        return round(base_score, 2)


# ---------------------------------------------------------------------------
# Remediation Information
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RemediationInfo:
    """Information for remediating a vulnerability.

    Attributes:
        description: Human-readable remediation guidance.
        fixed_version: Version that fixes the vulnerability.
        upgrade_path: Sequence of upgrades needed.
        patch_available: Whether a patch is available.
        workaround: Temporary workaround if no fix available.
        effort_estimate: Estimated remediation effort (low/medium/high).
        breaking_changes: Whether fix involves breaking changes.
        auto_fixable: Whether automated fix is possible.
    """

    description: Optional[str] = None
    fixed_version: Optional[str] = None
    upgrade_path: List[str] = field(default_factory=list)
    patch_available: bool = False
    workaround: Optional[str] = None
    effort_estimate: str = "medium"
    breaking_changes: bool = False
    auto_fixable: bool = False


# ---------------------------------------------------------------------------
# Scan Finding
# ---------------------------------------------------------------------------


@dataclass
class ScanFinding:
    """A single security finding from a scanner.

    Represents a potential vulnerability, misconfiguration, or security
    issue discovered during scanning.

    Attributes:
        finding_id: Unique identifier for this finding.
        title: Short title describing the finding.
        description: Detailed description.
        severity: Normalized severity level.
        scanner_name: Name of the scanner that found this.
        scanner_type: Category of the scanner.
        rule_id: Scanner-specific rule identifier.
        location: Where the finding was discovered.
        container_location: Container-specific location (if applicable).
        vulnerability_info: Detailed CVE/CVSS information.
        remediation_info: How to fix the finding.
        status: Current status of the finding.
        fingerprint: Unique hash for deduplication.
        tags: Metadata tags.
        raw_data: Original scanner output.
        discovered_at: When the finding was discovered.
        sla_deadline: Remediation deadline based on SLA.
    """

    title: str
    description: str
    severity: Severity
    scanner_name: str
    scanner_type: ScannerType
    finding_id: str = field(default_factory=lambda: str(uuid4()))
    rule_id: Optional[str] = None
    location: Optional[FileLocation] = None
    container_location: Optional[ContainerLocation] = None
    vulnerability_info: Optional[VulnerabilityInfo] = None
    remediation_info: Optional[RemediationInfo] = None
    status: FindingStatus = FindingStatus.OPEN
    fingerprint: Optional[str] = None
    tags: Set[str] = field(default_factory=set)
    raw_data: Dict[str, Any] = field(default_factory=dict)
    discovered_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    sla_deadline: Optional[datetime] = None

    def __post_init__(self) -> None:
        """Generate fingerprint and calculate SLA deadline."""
        if self.fingerprint is None:
            self.fingerprint = self._generate_fingerprint()

        if self.sla_deadline is None:
            self.sla_deadline = self._calculate_sla_deadline()

    def _generate_fingerprint(self) -> str:
        """Generate a unique fingerprint for deduplication.

        Uses CVE, rule, location, and title to create a stable hash.

        Returns:
            SHA-256 hash fingerprint.
        """
        components = [
            self.vulnerability_info.cve_id
            if self.vulnerability_info and self.vulnerability_info.cve_id
            else "",
            self.rule_id or "",
            self.location.file_path if self.location else "",
            str(self.location.start_line) if self.location else "",
            self.title,
        ]
        fingerprint_str = "|".join(components)
        return hashlib.sha256(fingerprint_str.encode()).hexdigest()[:32]

    def _calculate_sla_deadline(self) -> datetime:
        """Calculate SLA deadline based on severity.

        Returns:
            Deadline datetime.
        """
        sla_priority = SEVERITY_SLA_MAP.get(self.severity, SLAPriority.P3)
        days = SLA_DAYS_MAP.get(sla_priority, 30)
        from datetime import timedelta

        return self.discovered_at + timedelta(days=days)

    def is_sla_breached(self) -> bool:
        """Check if SLA deadline has passed.

        Returns:
            True if SLA is breached.
        """
        if self.status in (
            FindingStatus.FIXED,
            FindingStatus.ACCEPTED,
            FindingStatus.FALSE_POSITIVE,
        ):
            return False
        if self.sla_deadline is None:
            return False
        return datetime.now(timezone.utc) > self.sla_deadline

    def to_sarif_result(self) -> Dict[str, Any]:
        """Convert to SARIF result format.

        Returns:
            SARIF-compatible result dictionary.
        """
        level_map = {
            Severity.CRITICAL: "error",
            Severity.HIGH: "error",
            Severity.MEDIUM: "warning",
            Severity.LOW: "note",
            Severity.INFO: "none",
        }

        result: Dict[str, Any] = {
            "ruleId": self.rule_id or self.finding_id,
            "level": level_map.get(self.severity, "warning"),
            "message": {"text": self.description},
        }

        if self.location:
            result["locations"] = [self.location.to_sarif_location()]

        if self.fingerprint:
            result["fingerprints"] = {"gl-fingerprint-v1": self.fingerprint}

        if self.vulnerability_info and self.vulnerability_info.cve_id:
            result["taxa"] = [
                {
                    "id": self.vulnerability_info.cve_id,
                    "toolComponent": {"name": "CVE"},
                }
            ]

        return result

    def get_risk_score(self) -> float:
        """Get the composite risk score for this finding.

        Returns:
            Risk score (0.0-10.0).
        """
        if self.vulnerability_info:
            return self.vulnerability_info.get_risk_score()

        # Default score based on severity
        severity_scores = {
            Severity.CRITICAL: 9.5,
            Severity.HIGH: 7.5,
            Severity.MEDIUM: 5.0,
            Severity.LOW: 2.5,
            Severity.INFO: 0.5,
        }
        return severity_scores.get(self.severity, 5.0)


# ---------------------------------------------------------------------------
# Scan Result
# ---------------------------------------------------------------------------


@dataclass
class ScanResult:
    """Result from a single scanner execution.

    Attributes:
        scanner_name: Name of the scanner.
        scanner_type: Category of the scanner.
        status: Execution status.
        findings: List of security findings.
        started_at: When the scan started.
        completed_at: When the scan completed.
        duration_seconds: Scan duration in seconds.
        error_message: Error message if scan failed.
        exit_code: Scanner process exit code.
        raw_output: Raw scanner output.
        command: Command that was executed.
        scan_path: Path that was scanned.
    """

    scanner_name: str
    scanner_type: ScannerType
    status: ScanStatus = ScanStatus.PENDING
    findings: List[ScanFinding] = field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: float = 0.0
    error_message: Optional[str] = None
    exit_code: Optional[int] = None
    raw_output: Optional[str] = None
    command: Optional[str] = None
    scan_path: Optional[str] = None

    def get_finding_counts(self) -> Dict[Severity, int]:
        """Get count of findings by severity.

        Returns:
            Dictionary mapping severity to count.
        """
        counts = {sev: 0 for sev in Severity}
        for finding in self.findings:
            counts[finding.severity] += 1
        return counts

    def get_critical_count(self) -> int:
        """Get count of critical findings.

        Returns:
            Number of critical findings.
        """
        return sum(1 for f in self.findings if f.severity == Severity.CRITICAL)

    def get_high_count(self) -> int:
        """Get count of high severity findings.

        Returns:
            Number of high findings.
        """
        return sum(1 for f in self.findings if f.severity == Severity.HIGH)

    def has_blocking_findings(self, threshold: Severity) -> bool:
        """Check if there are findings at or above the threshold.

        Args:
            threshold: Minimum severity to consider blocking.

        Returns:
            True if blocking findings exist.
        """
        severity_order = [
            Severity.INFO,
            Severity.LOW,
            Severity.MEDIUM,
            Severity.HIGH,
            Severity.CRITICAL,
        ]
        threshold_idx = severity_order.index(threshold)

        for finding in self.findings:
            finding_idx = severity_order.index(finding.severity)
            if finding_idx >= threshold_idx:
                return True
        return False


# ---------------------------------------------------------------------------
# Scan Report
# ---------------------------------------------------------------------------


@dataclass
class ScanReport:
    """Aggregated report from all scanners.

    Attributes:
        report_id: Unique report identifier.
        scan_results: Results from individual scanners.
        all_findings: Deduplicated list of all findings.
        started_at: When the overall scan started.
        completed_at: When the overall scan completed.
        total_duration_seconds: Total scan duration.
        scan_path: Path that was scanned.
        git_commit: Git commit SHA if available.
        git_branch: Git branch name if available.
        scanners_run: List of scanner names executed.
        scanners_failed: List of scanners that failed.
        deduplication_enabled: Whether deduplication was applied.
        metadata: Additional metadata.
    """

    scan_results: List[ScanResult] = field(default_factory=list)
    all_findings: List[ScanFinding] = field(default_factory=list)
    report_id: str = field(default_factory=lambda: str(uuid4()))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_duration_seconds: float = 0.0
    scan_path: Optional[str] = None
    git_commit: Optional[str] = None
    git_branch: Optional[str] = None
    scanners_run: List[str] = field(default_factory=list)
    scanners_failed: List[str] = field(default_factory=list)
    deduplication_enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_total_finding_count(self) -> int:
        """Get total number of findings.

        Returns:
            Total finding count.
        """
        return len(self.all_findings)

    def get_finding_counts_by_severity(self) -> Dict[Severity, int]:
        """Get finding counts grouped by severity.

        Returns:
            Dictionary mapping severity to count.
        """
        counts = {sev: 0 for sev in Severity}
        for finding in self.all_findings:
            counts[finding.severity] += 1
        return counts

    def get_finding_counts_by_scanner(self) -> Dict[str, int]:
        """Get finding counts grouped by scanner.

        Returns:
            Dictionary mapping scanner name to count.
        """
        counts: Dict[str, int] = {}
        for finding in self.all_findings:
            counts[finding.scanner_name] = counts.get(finding.scanner_name, 0) + 1
        return counts

    def get_finding_counts_by_type(self) -> Dict[ScannerType, int]:
        """Get finding counts grouped by scanner type.

        Returns:
            Dictionary mapping scanner type to count.
        """
        counts = {st: 0 for st in ScannerType}
        for finding in self.all_findings:
            counts[finding.scanner_type] += 1
        return counts

    def get_cve_list(self) -> List[str]:
        """Get list of unique CVE IDs.

        Returns:
            List of CVE identifiers.
        """
        cves = set()
        for finding in self.all_findings:
            if finding.vulnerability_info and finding.vulnerability_info.cve_id:
                cves.add(finding.vulnerability_info.cve_id)
        return sorted(cves)

    def get_critical_findings(self) -> List[ScanFinding]:
        """Get all critical severity findings.

        Returns:
            List of critical findings.
        """
        return [f for f in self.all_findings if f.severity == Severity.CRITICAL]

    def get_high_findings(self) -> List[ScanFinding]:
        """Get all high severity findings.

        Returns:
            List of high findings.
        """
        return [f for f in self.all_findings if f.severity == Severity.HIGH]

    def get_kev_findings(self) -> List[ScanFinding]:
        """Get findings that are in CISA KEV list.

        Returns:
            List of KEV findings.
        """
        return [
            f
            for f in self.all_findings
            if f.vulnerability_info and f.vulnerability_info.is_kev
        ]

    def has_blocking_findings(self, threshold: Severity) -> bool:
        """Check if report contains blocking findings.

        Args:
            threshold: Minimum severity to consider blocking.

        Returns:
            True if blocking findings exist.
        """
        severity_order = [
            Severity.INFO,
            Severity.LOW,
            Severity.MEDIUM,
            Severity.HIGH,
            Severity.CRITICAL,
        ]
        threshold_idx = severity_order.index(threshold)

        for finding in self.all_findings:
            finding_idx = severity_order.index(finding.severity)
            if finding_idx >= threshold_idx:
                return True
        return False

    def get_scan_success_rate(self) -> float:
        """Get percentage of successful scanner executions.

        Returns:
            Success rate (0.0-100.0).
        """
        if not self.scanners_run:
            return 100.0
        success_count = len(self.scanners_run) - len(self.scanners_failed)
        return (success_count / len(self.scanners_run)) * 100.0

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the report.

        Returns:
            Summary dictionary with key metrics.
        """
        counts = self.get_finding_counts_by_severity()
        return {
            "report_id": self.report_id,
            "scan_path": self.scan_path,
            "git_commit": self.git_commit,
            "git_branch": self.git_branch,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
            "duration_seconds": self.total_duration_seconds,
            "total_findings": self.get_total_finding_count(),
            "critical_count": counts[Severity.CRITICAL],
            "high_count": counts[Severity.HIGH],
            "medium_count": counts[Severity.MEDIUM],
            "low_count": counts[Severity.LOW],
            "info_count": counts[Severity.INFO],
            "unique_cves": len(self.get_cve_list()),
            "kev_count": len(self.get_kev_findings()),
            "scanners_run": len(self.scanners_run),
            "scanners_failed": len(self.scanners_failed),
            "success_rate": self.get_scan_success_rate(),
        }
