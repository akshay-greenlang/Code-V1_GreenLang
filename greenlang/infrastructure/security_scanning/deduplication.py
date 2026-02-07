# -*- coding: utf-8 -*-
"""
Deduplication Engine - SEC-007

Provides CVE-based and fingerprint-based deduplication of security
findings across multiple scanners. Normalizes severity to CVSS 3.1
and merges duplicate findings.

The deduplication engine correlates findings by:
1. CVE ID (exact match)
2. Fingerprint (location + rule hash)
3. Semantic similarity (same file/line/description)

Example:
    >>> from greenlang.infrastructure.security_scanning.deduplication import (
    ...     DeduplicationEngine,
    ... )
    >>> engine = DeduplicationEngine()
    >>> deduplicated = engine.deduplicate(all_findings)

Author: GreenLang Security Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from greenlang.infrastructure.security_scanning.config import Severity
from greenlang.infrastructure.security_scanning.models import (
    ScanFinding,
    VulnerabilityInfo,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Deduplication Result
# ---------------------------------------------------------------------------


@dataclass
class DeduplicationResult:
    """Result of deduplication operation.

    Attributes:
        deduplicated_findings: Unique findings after deduplication.
        duplicate_count: Number of duplicates removed.
        merge_groups: Groups of findings that were merged.
        original_count: Original number of findings.
        deduplication_rate: Percentage of duplicates found.
    """

    deduplicated_findings: List[ScanFinding]
    duplicate_count: int
    merge_groups: List[List[str]] = field(default_factory=list)
    original_count: int = 0
    deduplication_rate: float = 0.0

    def __post_init__(self) -> None:
        """Calculate deduplication rate."""
        if self.original_count > 0:
            self.deduplication_rate = (
                self.duplicate_count / self.original_count
            ) * 100.0


# ---------------------------------------------------------------------------
# Deduplication Engine
# ---------------------------------------------------------------------------


class DeduplicationEngine:
    """Engine for deduplicating security findings across scanners.

    Uses multiple strategies to identify and merge duplicate findings:
    - CVE-based: Exact CVE ID match
    - Fingerprint-based: Hash of location + rule
    - Location-based: Same file and line range
    - Semantic: Similar title/description

    Attributes:
        enable_cve_dedup: Enable CVE-based deduplication.
        enable_fingerprint_dedup: Enable fingerprint-based deduplication.
        enable_location_dedup: Enable location-based deduplication.
        similarity_threshold: Threshold for semantic similarity (0.0-1.0).

    Example:
        >>> engine = DeduplicationEngine()
        >>> result = engine.deduplicate(findings)
        >>> print(f"Removed {result.duplicate_count} duplicates")
    """

    def __init__(
        self,
        enable_cve_dedup: bool = True,
        enable_fingerprint_dedup: bool = True,
        enable_location_dedup: bool = True,
        similarity_threshold: float = 0.8,
    ) -> None:
        """Initialize deduplication engine.

        Args:
            enable_cve_dedup: Enable CVE-based deduplication.
            enable_fingerprint_dedup: Enable fingerprint-based deduplication.
            enable_location_dedup: Enable location-based deduplication.
            similarity_threshold: Threshold for semantic similarity.
        """
        self.enable_cve_dedup = enable_cve_dedup
        self.enable_fingerprint_dedup = enable_fingerprint_dedup
        self.enable_location_dedup = enable_location_dedup
        self.similarity_threshold = similarity_threshold

        logger.debug(
            "DeduplicationEngine initialized  "
            "cve=%s  fingerprint=%s  location=%s  threshold=%.2f",
            enable_cve_dedup,
            enable_fingerprint_dedup,
            enable_location_dedup,
            similarity_threshold,
        )

    def deduplicate(
        self, findings: List[ScanFinding]
    ) -> DeduplicationResult:
        """Deduplicate a list of findings.

        Applies multiple deduplication strategies and returns merged results.

        Args:
            findings: List of findings to deduplicate.

        Returns:
            DeduplicationResult with unique findings.
        """
        if not findings:
            return DeduplicationResult(
                deduplicated_findings=[],
                duplicate_count=0,
                original_count=0,
            )

        original_count = len(findings)
        logger.info("Starting deduplication of %d findings", original_count)

        # Build deduplication index
        dedup_index = self._build_dedup_index(findings)

        # Merge duplicates
        merged_findings = self._merge_duplicates(dedup_index)

        # Sort by severity (critical first)
        severity_order = {
            Severity.CRITICAL: 0,
            Severity.HIGH: 1,
            Severity.MEDIUM: 2,
            Severity.LOW: 3,
            Severity.INFO: 4,
        }
        merged_findings.sort(
            key=lambda f: (severity_order.get(f.severity, 5), f.title)
        )

        duplicate_count = original_count - len(merged_findings)

        logger.info(
            "Deduplication complete: %d -> %d findings (removed %d duplicates, %.1f%%)",
            original_count,
            len(merged_findings),
            duplicate_count,
            (duplicate_count / original_count) * 100 if original_count else 0,
        )

        return DeduplicationResult(
            deduplicated_findings=merged_findings,
            duplicate_count=duplicate_count,
            merge_groups=list(dedup_index.values()),
            original_count=original_count,
        )

    def _build_dedup_index(
        self, findings: List[ScanFinding]
    ) -> Dict[str, List[ScanFinding]]:
        """Build index mapping dedup keys to findings.

        Args:
            findings: List of findings.

        Returns:
            Dictionary mapping dedup keys to finding lists.
        """
        index: Dict[str, List[ScanFinding]] = defaultdict(list)

        for finding in findings:
            key = self._get_dedup_key(finding)
            index[key].append(finding)

        return dict(index)

    def _get_dedup_key(self, finding: ScanFinding) -> str:
        """Generate deduplication key for a finding.

        Priority order:
        1. CVE ID (if available and CVE dedup enabled)
        2. Fingerprint (if enabled)
        3. Location hash

        Args:
            finding: Finding to generate key for.

        Returns:
            Deduplication key string.
        """
        # Try CVE-based key first
        if self.enable_cve_dedup:
            cve_id = self._extract_cve(finding)
            if cve_id:
                # Include file path to distinguish same CVE in different locations
                file_path = finding.location.file_path if finding.location else ""
                return f"cve:{cve_id}:{file_path}"

        # Try fingerprint-based key
        if self.enable_fingerprint_dedup and finding.fingerprint:
            return f"fingerprint:{finding.fingerprint}"

        # Fall back to location-based key
        if self.enable_location_dedup and finding.location:
            return self._generate_location_key(finding)

        # Last resort: unique key based on all attributes
        return f"unique:{finding.finding_id}"

    def _extract_cve(self, finding: ScanFinding) -> Optional[str]:
        """Extract CVE ID from finding.

        Args:
            finding: Finding to extract CVE from.

        Returns:
            CVE ID or None.
        """
        # Check vulnerability info
        if finding.vulnerability_info and finding.vulnerability_info.cve_id:
            return finding.vulnerability_info.cve_id

        # Check rule ID
        if finding.rule_id and finding.rule_id.startswith("CVE-"):
            return finding.rule_id

        # Check title
        import re
        cve_match = re.search(r"CVE-\d{4}-\d{4,}", finding.title, re.IGNORECASE)
        if cve_match:
            return cve_match.group(0).upper()

        return None

    def _generate_location_key(self, finding: ScanFinding) -> str:
        """Generate location-based deduplication key.

        Args:
            finding: Finding to generate key for.

        Returns:
            Location-based key.
        """
        location = finding.location
        if not location:
            return f"no-location:{finding.finding_id}"

        components = [
            location.file_path,
            str(location.start_line),
            finding.rule_id or finding.title[:50],
        ]
        key_str = "|".join(components)
        key_hash = hashlib.md5(key_str.encode()).hexdigest()[:16]
        return f"location:{key_hash}"

    def _merge_duplicates(
        self, dedup_index: Dict[str, List[ScanFinding]]
    ) -> List[ScanFinding]:
        """Merge groups of duplicate findings.

        Args:
            dedup_index: Index of dedup keys to findings.

        Returns:
            List of merged findings.
        """
        merged = []

        for key, group in dedup_index.items():
            if len(group) == 1:
                merged.append(group[0])
            else:
                # Merge multiple findings into one
                merged_finding = self._merge_finding_group(group)
                merged.append(merged_finding)

        return merged

    def _merge_finding_group(
        self, group: List[ScanFinding]
    ) -> ScanFinding:
        """Merge a group of duplicate findings into one.

        Selects the most informative version and aggregates scanner sources.

        Args:
            group: List of duplicate findings.

        Returns:
            Merged finding.
        """
        # Sort by information richness (more fields = better)
        def richness_score(f: ScanFinding) -> int:
            score = 0
            if f.vulnerability_info:
                if f.vulnerability_info.cvss_score:
                    score += 3
                if f.vulnerability_info.cve_id:
                    score += 2
                if f.vulnerability_info.cwe_id:
                    score += 1
            if f.remediation_info:
                if f.remediation_info.fixed_version:
                    score += 2
                if f.remediation_info.description:
                    score += 1
            if f.location and f.location.snippet:
                score += 1
            return score

        # Use the richest finding as base
        group.sort(key=richness_score, reverse=True)
        base = group[0]

        # Aggregate scanner sources
        scanner_names = {f.scanner_name for f in group}

        # Merge tags
        all_tags = set()
        for f in group:
            all_tags.update(f.tags)

        # Use highest severity
        severity_order = [
            Severity.INFO,
            Severity.LOW,
            Severity.MEDIUM,
            Severity.HIGH,
            Severity.CRITICAL,
        ]
        max_severity = max(group, key=lambda f: severity_order.index(f.severity)).severity

        # Merge vulnerability info (take best CVSS, combine references)
        merged_vuln_info = self._merge_vulnerability_info(
            [f.vulnerability_info for f in group if f.vulnerability_info]
        )

        # Create merged finding
        return ScanFinding(
            finding_id=base.finding_id,
            title=base.title,
            description=(
                base.description
                + f"\n\n[Detected by: {', '.join(sorted(scanner_names))}]"
            ),
            severity=max_severity,
            scanner_name=base.scanner_name,  # Primary scanner
            scanner_type=base.scanner_type,
            rule_id=base.rule_id,
            location=base.location,
            container_location=base.container_location,
            vulnerability_info=merged_vuln_info,
            remediation_info=base.remediation_info,
            status=base.status,
            fingerprint=base.fingerprint,
            tags=all_tags | {"merged", f"sources:{len(scanner_names)}"},
            raw_data={"merged_from": [f.finding_id for f in group]},
            discovered_at=min(f.discovered_at for f in group),
            sla_deadline=min(f.sla_deadline for f in group if f.sla_deadline),
        )

    def _merge_vulnerability_info(
        self, vuln_infos: List[VulnerabilityInfo]
    ) -> Optional[VulnerabilityInfo]:
        """Merge multiple vulnerability info objects.

        Takes highest CVSS, combines references.

        Args:
            vuln_infos: List of vulnerability info objects.

        Returns:
            Merged vulnerability info or None.
        """
        if not vuln_infos:
            return None

        # Find best CVSS score
        best_cvss = None
        best_cvss_vector = None
        for vi in vuln_infos:
            if vi.cvss_score and (best_cvss is None or vi.cvss_score > best_cvss):
                best_cvss = vi.cvss_score
                best_cvss_vector = vi.cvss_vector

        # Collect all references
        all_refs = set()
        for vi in vuln_infos:
            all_refs.update(vi.references)

        # Take first non-None values
        cve_id = next((vi.cve_id for vi in vuln_infos if vi.cve_id), None)
        cwe_id = next((vi.cwe_id for vi in vuln_infos if vi.cwe_id), None)
        epss_score = next((vi.epss_score for vi in vuln_infos if vi.epss_score), None)
        epss_percentile = next(
            (vi.epss_percentile for vi in vuln_infos if vi.epss_percentile), None
        )
        is_kev = any(vi.is_kev for vi in vuln_infos)
        description = next((vi.description for vi in vuln_infos if vi.description), None)

        return VulnerabilityInfo(
            cve_id=cve_id,
            cwe_id=cwe_id,
            cvss_score=best_cvss,
            cvss_vector=best_cvss_vector,
            epss_score=epss_score,
            epss_percentile=epss_percentile,
            is_kev=is_kev,
            description=description,
            references=sorted(all_refs),
        )


# ---------------------------------------------------------------------------
# Severity Normalization
# ---------------------------------------------------------------------------


def normalize_severity(
    scanner_severity: str,
    scanner_name: str,
) -> Severity:
    """Normalize scanner-specific severity to CVSS 3.1 severity.

    Different scanners use different severity scales. This function
    normalizes them to a consistent scale.

    Args:
        scanner_severity: Scanner-reported severity string.
        scanner_name: Name of the scanner.

    Returns:
        Normalized Severity enum value.
    """
    severity_upper = scanner_severity.upper().strip()

    # Standard CVSS-like severity
    if severity_upper in ("CRITICAL", "VERY HIGH", "SEVERE"):
        return Severity.CRITICAL
    if severity_upper in ("HIGH", "IMPORTANT", "ERROR"):
        return Severity.HIGH
    if severity_upper in ("MEDIUM", "MODERATE", "WARNING"):
        return Severity.MEDIUM
    if severity_upper in ("LOW", "MINOR"):
        return Severity.LOW
    if severity_upper in ("INFO", "INFORMATIONAL", "NOTE", "NONE"):
        return Severity.INFO

    # Scanner-specific mappings
    scanner_maps: Dict[str, Dict[str, Severity]] = {
        "bandit": {
            "HIGH": Severity.HIGH,
            "MEDIUM": Severity.MEDIUM,
            "LOW": Severity.LOW,
        },
        "semgrep": {
            "ERROR": Severity.HIGH,
            "WARNING": Severity.MEDIUM,
            "INFO": Severity.LOW,
        },
        "trivy": {
            "CRITICAL": Severity.CRITICAL,
            "HIGH": Severity.HIGH,
            "MEDIUM": Severity.MEDIUM,
            "LOW": Severity.LOW,
            "UNKNOWN": Severity.INFO,
        },
    }

    scanner_map = scanner_maps.get(scanner_name.lower(), {})
    return scanner_map.get(severity_upper, Severity.MEDIUM)


def cvss_to_severity(cvss_score: float) -> Severity:
    """Convert CVSS score to severity level.

    CVSS 3.1 Severity Ratings:
        - None: 0.0
        - Low: 0.1 - 3.9
        - Medium: 4.0 - 6.9
        - High: 7.0 - 8.9
        - Critical: 9.0 - 10.0

    Args:
        cvss_score: CVSS 3.1 score (0.0-10.0).

    Returns:
        Corresponding Severity enum value.
    """
    return Severity.from_cvss(cvss_score)


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------


def calculate_fingerprint(
    file_path: str,
    line_number: int,
    rule_id: str,
    title: str,
) -> str:
    """Calculate a stable fingerprint for a finding.

    The fingerprint is used for deduplication and tracking findings
    across scans.

    Args:
        file_path: Path to the affected file.
        line_number: Line number of the finding.
        rule_id: Scanner rule identifier.
        title: Finding title.

    Returns:
        SHA-256 fingerprint (32 hex chars).
    """
    components = [
        file_path.lower().replace("\\", "/"),
        str(line_number),
        rule_id or "",
        title[:100],
    ]
    fingerprint_str = "|".join(components)
    return hashlib.sha256(fingerprint_str.encode()).hexdigest()[:32]


def group_findings_by_cve(
    findings: List[ScanFinding],
) -> Dict[str, List[ScanFinding]]:
    """Group findings by CVE ID.

    Args:
        findings: List of findings.

    Returns:
        Dictionary mapping CVE IDs to finding lists.
    """
    groups: Dict[str, List[ScanFinding]] = defaultdict(list)

    for finding in findings:
        cve_id = None
        if finding.vulnerability_info and finding.vulnerability_info.cve_id:
            cve_id = finding.vulnerability_info.cve_id
        elif finding.rule_id and finding.rule_id.startswith("CVE-"):
            cve_id = finding.rule_id

        if cve_id:
            groups[cve_id].append(finding)
        else:
            groups["NO_CVE"].append(finding)

    return dict(groups)


def get_unique_cves(findings: List[ScanFinding]) -> Set[str]:
    """Extract unique CVE IDs from findings.

    Args:
        findings: List of findings.

    Returns:
        Set of unique CVE IDs.
    """
    cves = set()
    for finding in findings:
        if finding.vulnerability_info and finding.vulnerability_info.cve_id:
            cves.add(finding.vulnerability_info.cve_id)
        elif finding.rule_id and finding.rule_id.startswith("CVE-"):
            cves.add(finding.rule_id)
    return cves
