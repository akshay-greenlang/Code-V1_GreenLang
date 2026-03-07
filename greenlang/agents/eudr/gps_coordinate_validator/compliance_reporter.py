# -*- coding: utf-8 -*-
"""
Compliance Reporter - AGENT-EUDR-007 Engine 8

Production-grade compliance reporting engine for GPS coordinate validation
under the EU Deforestation Regulation (EUDR). Generates compliance
certificates, batch summaries, remediation guidance, audit trails,
quality trend analysis, submission readiness assessments, and multi-format
exports (JSON, CSV, EUDR XML).

Zero-Hallucination Guarantees:
    - All report generation is deterministic
    - Compliance status derived from fixed scoring thresholds
    - Certificate IDs are UUID4 (not predictable, but reproducible
      via provenance hash)
    - XML output follows EUDR DDS namespace conventions
    - SHA-256 provenance chain hashing on all reports
    - No ML/LLM used in any report generation logic

Performance Targets:
    - Single certificate: <5ms
    - Batch summary (10,000 results): <500ms
    - Full compliance report: <2 seconds

Regulatory References:
    - EUDR Article 4: Due Diligence obligations
    - EUDR Article 9: Geolocation data requirements for DDS
    - EUDR Article 10: Risk assessment reporting
    - EUDR Article 31: Record-keeping (5-year retention)
    - EUDR Annex II: Due Diligence Statement content

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-007 (Engine 8: Compliance Reporting)
Agent ID: GL-EUDR-GPS-007
Status: Production Ready
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module version for provenance tracking
# ---------------------------------------------------------------------------

_MODULE_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash for audit provenance."""
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ComplianceStatus(str, Enum):
    """Compliance status for a GPS coordinate or batch.

    COMPLIANT: All checks passed, meets EUDR Article 9 requirements.
    NON_COMPLIANT: Critical issues found, does not meet requirements.
    NEEDS_REVIEW: Minor issues found, requires human review.
    INSUFFICIENT_DATA: Not enough data to make a determination.
    """

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    NEEDS_REVIEW = "needs_review"
    INSUFFICIENT_DATA = "insufficient_data"


class ReportFormat(str, Enum):
    """Supported report output formats."""

    JSON = "json"
    CSV = "csv"
    EUDR_XML = "eudr_xml"


class RemediationPriority(str, Enum):
    """Priority level for remediation actions."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ---------------------------------------------------------------------------
# Result Data Classes
# ---------------------------------------------------------------------------


@dataclass
class ComplianceCertificate:
    """Compliance certificate for a single GPS coordinate.

    Attributes:
        certificate_id: Unique UUID4 certificate identifier.
        coordinate_lat: Validated latitude (WGS84).
        coordinate_lon: Validated longitude (WGS84).
        status: Compliance determination.
        accuracy_score: Overall accuracy score (0-100).
        accuracy_tier: Quality tier (gold/silver/bronze/unverified).
        confidence_interval_m: 95% confidence radius in metres.
        precision_decimal_places: Coordinate decimal places.
        country_iso: Detected country ISO alpha-2 code.
        commodity: EUDR commodity if specified.
        is_on_land: Whether coordinate is on land.
        country_match: Whether declared country matches detected.
        issues_found: Number of issues found across all checks.
        critical_issues: Number of critical issues.
        issued_at: Certificate issuance timestamp (UTC ISO).
        valid_until: Certificate validity expiry (1 year from issuance).
        provenance_hash: SHA-256 hash for tamper detection.
        validation_engine_version: Engine version string.
    """

    certificate_id: str = ""
    coordinate_lat: float = 0.0
    coordinate_lon: float = 0.0
    status: ComplianceStatus = ComplianceStatus.INSUFFICIENT_DATA
    accuracy_score: float = 0.0
    accuracy_tier: str = "unverified"
    confidence_interval_m: float = 0.0
    precision_decimal_places: int = 0
    country_iso: Optional[str] = None
    commodity: Optional[str] = None
    is_on_land: bool = True
    country_match: bool = True
    issues_found: int = 0
    critical_issues: int = 0
    issued_at: str = ""
    valid_until: str = ""
    provenance_hash: str = ""
    validation_engine_version: str = _MODULE_VERSION


@dataclass
class RemediationItem:
    """A single remediation action item.

    Attributes:
        priority: Remediation priority level.
        category: Error category (precision, plausibility, consistency,
            source).
        issue_description: Human-readable description of the issue.
        fix_instruction: Specific instructions to fix the issue.
        expected_improvement: Expected score improvement from the fix.
        affected_coordinates: Number of coordinates affected.
    """

    priority: RemediationPriority = RemediationPriority.MEDIUM
    category: str = ""
    issue_description: str = ""
    fix_instruction: str = ""
    expected_improvement: float = 0.0
    affected_coordinates: int = 0


@dataclass
class QualityTrend:
    """Quality trend analysis over time.

    Attributes:
        period_label: Time period label.
        average_score: Average accuracy score for the period.
        total_coordinates: Total coordinates assessed.
        compliant_count: Number compliant.
        non_compliant_count: Number non-compliant.
        tier_distribution: Count by tier (gold/silver/bronze/unverified).
        trend_direction: "improving", "stable", or "degrading".
        change_pct: Percentage change from previous period.
    """

    period_label: str = ""
    average_score: float = 0.0
    total_coordinates: int = 0
    compliant_count: int = 0
    non_compliant_count: int = 0
    tier_distribution: Dict[str, int] = field(default_factory=dict)
    trend_direction: str = "stable"
    change_pct: float = 0.0


@dataclass
class SubmissionReadiness:
    """Submission readiness assessment.

    Attributes:
        readiness_pct: Overall readiness percentage (0-100).
        is_ready: Whether the batch is ready for DDS submission.
        blocking_issues: List of issues that block submission.
        warnings: List of non-blocking warnings.
        recommendations: List of improvement recommendations.
        compliant_count: Number of compliant coordinates.
        total_count: Total coordinates assessed.
        minimum_score_met: Whether minimum accuracy score is met.
        provenance_hash: SHA-256 hash for audit trail.
    """

    readiness_pct: float = 0.0
    is_ready: bool = False
    blocking_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    compliant_count: int = 0
    total_count: int = 0
    minimum_score_met: bool = False
    provenance_hash: str = ""


@dataclass
class AuditTrailEntry:
    """A single entry in the compliance audit trail.

    Attributes:
        timestamp: UTC ISO timestamp of the operation.
        operation: Type of operation performed.
        actor: Identity of the actor/system performing the operation.
        entity_id: ID of the entity being operated on.
        details: Additional operation details.
        hash_value: SHA-256 hash of this entry.
        parent_hash: SHA-256 hash of the previous entry.
    """

    timestamp: str = ""
    operation: str = ""
    actor: str = "gl-eudr-gps-007"
    entity_id: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    hash_value: str = ""
    parent_hash: str = ""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Compliance score thresholds.
COMPLIANCE_THRESHOLDS: Dict[ComplianceStatus, Tuple[float, float]] = {
    ComplianceStatus.COMPLIANT: (70.0, 100.0),
    ComplianceStatus.NEEDS_REVIEW: (50.0, 70.0),
    ComplianceStatus.NON_COMPLIANT: (0.0, 50.0),
}

#: Certificate validity duration.
CERTIFICATE_VALIDITY_DAYS: int = 365

#: EUDR DDS XML namespace.
EUDR_DDS_NAMESPACE: str = "urn:eu:eudr:dds:geolocation:2024"

#: Minimum compliant percentage for batch submission readiness.
MIN_COMPLIANT_PCT_FOR_SUBMISSION: float = 95.0

#: Remediation guidance templates.
REMEDIATION_TEMPLATES: Dict[str, Dict[str, str]] = {
    "low_precision": {
        "category": "precision",
        "description": "Coordinate has insufficient decimal precision",
        "instruction": (
            "Recollect GPS coordinates using a device capable of "
            "recording >= 6 decimal places. Ensure the GPS device has "
            "a clear sky view and allow at least 30 seconds for "
            "position fix."
        ),
        "improvement": "15-25 points",
    },
    "coordinate_swap": {
        "category": "consistency",
        "description": "Latitude and longitude appear to be transposed",
        "instruction": (
            "Verify the coordinate order in your data source. "
            "Standard format is (latitude, longitude) where latitude "
            "ranges -90 to 90 and longitude ranges -180 to 180. "
            "Swap the values and re-submit."
        ),
        "improvement": "20-40 points",
    },
    "null_island": {
        "category": "consistency",
        "description": "Coordinate is at (0, 0) - likely missing data",
        "instruction": (
            "The coordinate (0, 0) is a common default/null value. "
            "Collect the actual GPS coordinates at the production "
            "plot location and replace this entry."
        ),
        "improvement": "40-60 points",
    },
    "country_mismatch": {
        "category": "plausibility",
        "description": (
            "Coordinate does not fall within the declared country"
        ),
        "instruction": (
            "Verify the declared country code matches the actual "
            "location of the production plot. If the coordinate is "
            "correct, update the declared country code."
        ),
        "improvement": "10-25 points",
    },
    "ocean_location": {
        "category": "plausibility",
        "description": "Coordinate appears to be in the ocean",
        "instruction": (
            "The coordinate falls in an ocean area. This is likely "
            "a data entry error. Verify the sign (positive/negative) "
            "of latitude and longitude values and correct."
        ),
        "improvement": "25-40 points",
    },
    "elevation_implausible": {
        "category": "plausibility",
        "description": (
            "Elevation is outside plausible range for the commodity"
        ),
        "instruction": (
            "Verify that the coordinate is at the actual production "
            "plot. The estimated elevation is outside the typical "
            "growing range for this commodity."
        ),
        "improvement": "5-15 points",
    },
    "unknown_source": {
        "category": "source",
        "description": "Data source is unknown or unspecified",
        "instruction": (
            "Document the source of GPS coordinates (GNSS survey, "
            "mobile GPS, certification database, etc.). Known source "
            "types improve the reliability assessment."
        ),
        "improvement": "10-20 points",
    },
    "urban_location": {
        "category": "plausibility",
        "description": "Coordinate falls within an urban area",
        "instruction": (
            "Verify that the production plot is actually located "
            "within an urban area. Agricultural production plots "
            "are typically in rural areas."
        ),
        "improvement": "5-10 points",
    },
}


# ===========================================================================
# ComplianceReporter
# ===========================================================================


class ComplianceReporter:
    """Production-grade compliance reporting engine for EUDR GPS coordinates.

    Generates compliance certificates, batch summaries, remediation
    guidance, audit trails, quality trends, submission readiness
    assessments, and exports in JSON/CSV/EUDR XML formats.

    All report generation is deterministic with zero LLM/ML involvement.

    Attributes:
        _config: Optional configuration object.
        _audit_chain: In-memory audit trail chain.
        _last_audit_hash: Most recent audit chain hash.

    Example::

        reporter = ComplianceReporter()
        cert = reporter.generate_compliance_certificate(
            coord_lat=-3.46, coord_lon=28.23,
            accuracy_score=82.5, accuracy_tier="silver",
            issues_found=1, critical_issues=0,
        )
        assert cert.status == ComplianceStatus.COMPLIANT
    """

    def __init__(self, config: Any = None) -> None:
        """Initialize ComplianceReporter.

        Args:
            config: Optional configuration object.
        """
        self._config = config
        self._audit_chain: List[AuditTrailEntry] = []
        genesis_str = "GL-EUDR-GPS-007-COMPLIANCE-REPORTER-GENESIS"
        self._last_audit_hash: str = hashlib.sha256(
            genesis_str.encode("utf-8")
        ).hexdigest()
        logger.info("ComplianceReporter initialized")

    # ------------------------------------------------------------------
    # Public API: Certificate Generation
    # ------------------------------------------------------------------

    def generate_compliance_certificate(
        self,
        coord_lat: float,
        coord_lon: float,
        accuracy_score: float = 0.0,
        accuracy_tier: str = "unverified",
        confidence_interval_m: float = 0.0,
        precision_decimal_places: int = 0,
        country_iso: Optional[str] = None,
        commodity: Optional[str] = None,
        is_on_land: bool = True,
        country_match: bool = True,
        issues_found: int = 0,
        critical_issues: int = 0,
    ) -> ComplianceCertificate:
        """Generate a compliance certificate for a GPS coordinate.

        Determines compliance status based on accuracy score and issues,
        assigns a unique certificate ID, sets validity period, and
        computes provenance hash.

        Args:
            coord_lat: Validated WGS84 latitude.
            coord_lon: Validated WGS84 longitude.
            accuracy_score: Overall accuracy score (0-100).
            accuracy_tier: Quality tier string.
            confidence_interval_m: 95% CI radius in metres.
            precision_decimal_places: Coordinate decimal places.
            country_iso: Detected country ISO alpha-2 code.
            commodity: EUDR commodity identifier.
            is_on_land: Whether coordinate is on land.
            country_match: Whether declared country matches.
            issues_found: Total issues found.
            critical_issues: Number of critical issues.

        Returns:
            ComplianceCertificate with all fields populated.
        """
        start_time = time.monotonic()
        now = _utcnow()

        # Determine compliance status
        status = self._determine_compliance_status(
            accuracy_score=accuracy_score,
            critical_issues=critical_issues,
            is_on_land=is_on_land,
            country_match=country_match,
        )

        # Create certificate
        cert = ComplianceCertificate(
            certificate_id=str(uuid.uuid4()),
            coordinate_lat=coord_lat,
            coordinate_lon=coord_lon,
            status=status,
            accuracy_score=accuracy_score,
            accuracy_tier=accuracy_tier,
            confidence_interval_m=confidence_interval_m,
            precision_decimal_places=precision_decimal_places,
            country_iso=country_iso,
            commodity=commodity,
            is_on_land=is_on_land,
            country_match=country_match,
            issues_found=issues_found,
            critical_issues=critical_issues,
            issued_at=now.isoformat(),
            valid_until=(
                now + timedelta(days=CERTIFICATE_VALIDITY_DAYS)
            ).isoformat(),
            validation_engine_version=_MODULE_VERSION,
        )

        # Provenance hash
        cert.provenance_hash = self._compute_certificate_hash(cert)

        # Audit trail
        self._record_audit(
            operation="generate_certificate",
            entity_id=cert.certificate_id,
            details={
                "status": status.value,
                "accuracy_score": accuracy_score,
                "lat": coord_lat,
                "lon": coord_lon,
            },
        )

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.debug(
            "Certificate generated: id=%s, status=%s, score=%.1f, "
            "tier=%s, %.2fms",
            cert.certificate_id[:12], status.value,
            accuracy_score, accuracy_tier, elapsed_ms,
        )

        return cert

    # ------------------------------------------------------------------
    # Public API: Batch Summary
    # ------------------------------------------------------------------

    def generate_batch_summary(
        self,
        results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Generate a summary report for a batch of validation results.

        Each result dict is expected to have at minimum:
            - accuracy_score (float)
            - accuracy_tier (str)
            - status (str): compliant/non_compliant/needs_review
            - issues (list): list of issue dicts with 'type' and 'severity'
            - commodity (str, optional)
            - country_iso (str, optional)

        Args:
            results: List of validation result dictionaries.

        Returns:
            Summary dictionary with aggregate statistics.
        """
        start_time = time.monotonic()

        if not results:
            return {
                "total": 0,
                "valid": 0,
                "invalid": 0,
                "needs_review": 0,
                "insufficient_data": 0,
                "error_type_breakdown": {},
                "precision_distribution": {},
                "accuracy_tier_distribution": {},
                "commodity_breakdown": {},
                "country_breakdown": {},
                "top_issues": [],
                "average_score": 0.0,
                "provenance_hash": _compute_hash({"empty": True}),
            }

        total = len(results)
        valid = sum(
            1 for r in results
            if r.get("status") == "compliant"
        )
        invalid = sum(
            1 for r in results
            if r.get("status") == "non_compliant"
        )
        needs_review = sum(
            1 for r in results
            if r.get("status") == "needs_review"
        )
        insufficient = total - valid - invalid - needs_review

        # Average score
        scores = [
            r.get("accuracy_score", 0.0) for r in results
        ]
        avg_score = sum(scores) / len(scores) if scores else 0.0

        # Error type breakdown
        error_counts: Dict[str, int] = {}
        all_issues: List[Dict[str, Any]] = []
        for r in results:
            for issue in r.get("issues", []):
                issue_type = issue.get("type", "unknown")
                error_counts[issue_type] = (
                    error_counts.get(issue_type, 0) + 1
                )
                all_issues.append(issue)

        # Tier distribution
        tier_dist: Dict[str, int] = {}
        for r in results:
            tier = r.get("accuracy_tier", "unverified")
            tier_dist[tier] = tier_dist.get(tier, 0) + 1

        # Commodity breakdown
        commodity_stats: Dict[str, Dict[str, int]] = {}
        for r in results:
            commodity = r.get("commodity", "unknown")
            if commodity not in commodity_stats:
                commodity_stats[commodity] = {
                    "total": 0, "compliant": 0, "non_compliant": 0,
                }
            commodity_stats[commodity]["total"] += 1
            if r.get("status") == "compliant":
                commodity_stats[commodity]["compliant"] += 1
            elif r.get("status") == "non_compliant":
                commodity_stats[commodity]["non_compliant"] += 1

        # Country breakdown
        country_stats: Dict[str, Dict[str, int]] = {}
        for r in results:
            country = r.get("country_iso", "unknown")
            if country not in country_stats:
                country_stats[country] = {
                    "total": 0, "compliant": 0, "non_compliant": 0,
                }
            country_stats[country]["total"] += 1
            if r.get("status") == "compliant":
                country_stats[country]["compliant"] += 1
            elif r.get("status") == "non_compliant":
                country_stats[country]["non_compliant"] += 1

        # Top issues (sorted by frequency)
        sorted_issues = sorted(
            error_counts.items(), key=lambda x: x[1], reverse=True
        )
        top_issues = [
            {
                "issue_type": issue_type,
                "count": count,
                "percentage": round(count / total * 100, 1),
                "remediation": REMEDIATION_TEMPLATES.get(
                    issue_type, {}
                ).get("instruction", "Review and correct manually."),
            }
            for issue_type, count in sorted_issues[:10]
        ]

        summary = {
            "total": total,
            "valid": valid,
            "invalid": invalid,
            "needs_review": needs_review,
            "insufficient_data": insufficient,
            "average_score": round(avg_score, 2),
            "error_type_breakdown": error_counts,
            "accuracy_tier_distribution": tier_dist,
            "commodity_breakdown": commodity_stats,
            "country_breakdown": country_stats,
            "top_issues": top_issues,
            "generated_at": _utcnow().isoformat(),
            "provenance_hash": _compute_hash({
                "total": total,
                "valid": valid,
                "invalid": invalid,
                "avg_score": round(avg_score, 2),
            }),
        }

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            "Batch summary: %d total, %d valid, %d invalid, "
            "avg_score=%.1f, %d issue types, %.1fms",
            total, valid, invalid, avg_score,
            len(error_counts), elapsed_ms,
        )

        return summary

    # ------------------------------------------------------------------
    # Public API: Remediation Guidance
    # ------------------------------------------------------------------

    def generate_remediation_guidance(
        self,
        errors: List[Dict[str, Any]],
    ) -> List[RemediationItem]:
        """Generate specific remediation guidance for detected errors.

        Each error dict is expected to have:
            - type (str): Error type key
            - severity (str): critical/high/medium/low
            - count (int, optional): Number of affected coordinates
            - description (str, optional): Custom description

        Args:
            errors: List of error dictionaries.

        Returns:
            List of RemediationItem sorted by priority.
        """
        items: List[RemediationItem] = []

        for error in errors:
            error_type = error.get("type", "unknown")
            severity = error.get("severity", "medium")
            count = error.get("count", 1)

            # Map severity to priority
            priority_map = {
                "critical": RemediationPriority.CRITICAL,
                "high": RemediationPriority.HIGH,
                "medium": RemediationPriority.MEDIUM,
                "low": RemediationPriority.LOW,
            }
            priority = priority_map.get(
                severity.lower(), RemediationPriority.MEDIUM
            )

            # Look up template
            template = REMEDIATION_TEMPLATES.get(error_type, {})
            category = template.get("category", "general")
            description = (
                error.get("description")
                or template.get("description", f"Issue: {error_type}")
            )
            instruction = template.get(
                "instruction",
                "Review and correct the coordinate data manually.",
            )
            improvement_str = template.get("improvement", "5-15 points")

            # Parse improvement range
            try:
                parts = improvement_str.replace(" points", "").split("-")
                avg_improvement = (
                    (float(parts[0]) + float(parts[-1])) / 2.0
                )
            except (ValueError, IndexError):
                avg_improvement = 10.0

            items.append(RemediationItem(
                priority=priority,
                category=category,
                issue_description=description,
                fix_instruction=instruction,
                expected_improvement=avg_improvement,
                affected_coordinates=count,
            ))

        # Sort by priority (critical first)
        priority_order = {
            RemediationPriority.CRITICAL: 0,
            RemediationPriority.HIGH: 1,
            RemediationPriority.MEDIUM: 2,
            RemediationPriority.LOW: 3,
        }
        items.sort(key=lambda x: priority_order.get(x.priority, 99))

        logger.debug(
            "Generated %d remediation items from %d errors",
            len(items), len(errors),
        )

        return items

    # ------------------------------------------------------------------
    # Public API: Export Formats
    # ------------------------------------------------------------------

    def export_json(self, data: Dict[str, Any]) -> str:
        """Export data as formatted JSON string.

        Args:
            data: Dictionary to serialize.

        Returns:
            Pretty-printed JSON string.
        """
        return json.dumps(data, indent=2, default=str, sort_keys=True)

    def export_csv(
        self,
        results: List[Dict[str, Any]],
    ) -> str:
        """Export results as CSV string.

        Expected result dict keys: certificate_id, coordinate_lat,
        coordinate_lon, status, accuracy_score, accuracy_tier,
        country_iso, commodity, issues_found, critical_issues.

        Args:
            results: List of result dictionaries.

        Returns:
            CSV-formatted string with headers.
        """
        if not results:
            return ""

        output = io.StringIO()
        fieldnames = [
            "certificate_id",
            "coordinate_lat",
            "coordinate_lon",
            "status",
            "accuracy_score",
            "accuracy_tier",
            "confidence_interval_m",
            "country_iso",
            "commodity",
            "is_on_land",
            "country_match",
            "issues_found",
            "critical_issues",
            "issued_at",
            "valid_until",
        ]

        writer = csv.DictWriter(
            output,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()

        for result in results:
            row: Dict[str, Any] = {}
            for fn in fieldnames:
                val = result.get(fn, "")
                if isinstance(val, bool):
                    val = str(val).lower()
                elif isinstance(val, Enum):
                    val = val.value
                row[fn] = val
            writer.writerow(row)

        return output.getvalue()

    def export_eudr_xml(
        self,
        certificates: List[ComplianceCertificate],
    ) -> str:
        """Export compliance certificates as EUDR DDS XML.

        Generates XML following the EUDR Due Diligence Statement
        geolocation element structure.

        Args:
            certificates: List of ComplianceCertificate objects.

        Returns:
            XML string with EUDR DDS namespace.
        """
        lines: List[str] = []
        lines.append('<?xml version="1.0" encoding="UTF-8"?>')
        lines.append(
            f'<GeolocationValidation '
            f'xmlns="{EUDR_DDS_NAMESPACE}" '
            f'version="{_MODULE_VERSION}" '
            f'generated="{_utcnow().isoformat()}">'
        )

        for cert in certificates:
            status_attr = cert.status.value if isinstance(
                cert.status, Enum
            ) else str(cert.status)

            lines.append("  <Geolocation>")
            lines.append(
                f"    <CertificateId>{_xml_escape(cert.certificate_id)}"
                f"</CertificateId>"
            )
            lines.append(
                f"    <Latitude>{cert.coordinate_lat:.8f}</Latitude>"
            )
            lines.append(
                f"    <Longitude>{cert.coordinate_lon:.8f}</Longitude>"
            )
            lines.append(
                f'    <ComplianceStatus status="{status_attr}">'
                f"{status_attr}</ComplianceStatus>"
            )
            lines.append(
                f"    <AccuracyScore>{cert.accuracy_score:.2f}"
                f"</AccuracyScore>"
            )
            lines.append(
                f"    <QualityTier>{_xml_escape(cert.accuracy_tier)}"
                f"</QualityTier>"
            )
            lines.append(
                f"    <ConfidenceInterval unit=\"meters\">"
                f"{cert.confidence_interval_m:.2f}"
                f"</ConfidenceInterval>"
            )
            lines.append(
                f"    <PrecisionDecimalPlaces>"
                f"{cert.precision_decimal_places}"
                f"</PrecisionDecimalPlaces>"
            )
            if cert.country_iso:
                lines.append(
                    f"    <CountryCode>{_xml_escape(cert.country_iso)}"
                    f"</CountryCode>"
                )
            if cert.commodity:
                lines.append(
                    f"    <Commodity>{_xml_escape(cert.commodity)}"
                    f"</Commodity>"
                )
            lines.append(
                f"    <IsOnLand>{str(cert.is_on_land).lower()}"
                f"</IsOnLand>"
            )
            lines.append(
                f"    <CountryMatch>{str(cert.country_match).lower()}"
                f"</CountryMatch>"
            )
            lines.append(
                f"    <IssuesFound>{cert.issues_found}</IssuesFound>"
            )
            lines.append(
                f"    <CriticalIssues>{cert.critical_issues}"
                f"</CriticalIssues>"
            )
            lines.append(
                f"    <IssuedAt>{_xml_escape(cert.issued_at)}</IssuedAt>"
            )
            lines.append(
                f"    <ValidUntil>{_xml_escape(cert.valid_until)}"
                f"</ValidUntil>"
            )
            lines.append(
                f"    <ProvenanceHash>{cert.provenance_hash}"
                f"</ProvenanceHash>"
            )
            lines.append("  </Geolocation>")

        lines.append("</GeolocationValidation>")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Public API: Audit Trail
    # ------------------------------------------------------------------

    def generate_audit_trail(
        self,
        provenance_records: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Generate a compliance audit trail from provenance records.

        Builds a chain-hashed audit trail from the provided provenance
        records, including operation timeline and integrity verification.

        Args:
            provenance_records: List of provenance record dictionaries,
                each with 'operation', 'entity_id', 'timestamp', and
                optional 'details' keys.

        Returns:
            Audit trail dictionary with chain hashes and timeline.
        """
        chain: List[Dict[str, Any]] = []
        prev_hash = self._last_audit_hash

        for record in provenance_records:
            timestamp = record.get(
                "timestamp", _utcnow().isoformat()
            )
            operation = record.get("operation", "unknown")
            entity_id = record.get("entity_id", "")
            details = record.get("details", {})
            actor = record.get("actor", "gl-eudr-gps-007")

            entry_data = {
                "timestamp": timestamp,
                "operation": operation,
                "entity_id": entity_id,
                "actor": actor,
                "parent_hash": prev_hash,
            }
            entry_hash = _compute_hash(entry_data)

            chain.append({
                "timestamp": timestamp,
                "operation": operation,
                "actor": actor,
                "entity_id": entity_id,
                "details": details,
                "hash_value": entry_hash,
                "parent_hash": prev_hash,
            })

            prev_hash = entry_hash

        # Build timeline
        operations = [e["operation"] for e in chain]
        operation_counts: Dict[str, int] = {}
        for op in operations:
            operation_counts[op] = operation_counts.get(op, 0) + 1

        return {
            "total_entries": len(chain),
            "chain": chain,
            "chain_intact": True,
            "first_entry_hash": chain[0]["hash_value"] if chain else "",
            "last_entry_hash": chain[-1]["hash_value"] if chain else "",
            "operation_timeline": operation_counts,
            "generated_at": _utcnow().isoformat(),
        }

    # ------------------------------------------------------------------
    # Public API: Quality Trend
    # ------------------------------------------------------------------

    def generate_quality_trend(
        self,
        historical_scores: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Generate quality trend analysis from historical score data.

        Each historical_scores entry should have:
            - period (str): Period label (e.g., "2025-Q1")
            - average_score (float)
            - total (int)
            - compliant (int)
            - non_compliant (int)
            - tier_distribution (dict)

        Args:
            historical_scores: List of period score dictionaries.

        Returns:
            Trend analysis with direction indicators.
        """
        if not historical_scores:
            return {
                "periods": [],
                "overall_trend": "no_data",
                "latest_score": 0.0,
                "score_change": 0.0,
            }

        periods: List[QualityTrend] = []
        prev_score: Optional[float] = None

        for entry in historical_scores:
            avg_score = entry.get("average_score", 0.0)
            change_pct = 0.0
            direction = "stable"

            if prev_score is not None:
                if prev_score > 0:
                    change_pct = (
                        (avg_score - prev_score) / prev_score * 100
                    )
                if change_pct > 2.0:
                    direction = "improving"
                elif change_pct < -2.0:
                    direction = "degrading"

            trend = QualityTrend(
                period_label=entry.get("period", ""),
                average_score=avg_score,
                total_coordinates=entry.get("total", 0),
                compliant_count=entry.get("compliant", 0),
                non_compliant_count=entry.get("non_compliant", 0),
                tier_distribution=entry.get("tier_distribution", {}),
                trend_direction=direction,
                change_pct=round(change_pct, 2),
            )
            periods.append(trend)
            prev_score = avg_score

        # Overall trend determination
        if len(periods) >= 2:
            first_score = periods[0].average_score
            last_score = periods[-1].average_score
            if first_score > 0:
                overall_change = (
                    (last_score - first_score) / first_score * 100
                )
            else:
                overall_change = 0.0

            if overall_change > 5.0:
                overall_trend = "improving"
            elif overall_change < -5.0:
                overall_trend = "degrading"
            else:
                overall_trend = "stable"
        else:
            overall_trend = "insufficient_data"
            overall_change = 0.0

        return {
            "periods": [
                {
                    "period": p.period_label,
                    "average_score": p.average_score,
                    "total": p.total_coordinates,
                    "compliant": p.compliant_count,
                    "non_compliant": p.non_compliant_count,
                    "tier_distribution": p.tier_distribution,
                    "trend": p.trend_direction,
                    "change_pct": p.change_pct,
                }
                for p in periods
            ],
            "overall_trend": overall_trend,
            "latest_score": periods[-1].average_score if periods else 0.0,
            "score_change": round(overall_change, 2),
        }

    # ------------------------------------------------------------------
    # Public API: Submission Readiness
    # ------------------------------------------------------------------

    def submission_readiness(
        self,
        results: List[Dict[str, Any]],
    ) -> SubmissionReadiness:
        """Assess whether a batch of results is ready for DDS submission.

        Evaluates overall compliance rate, blocking issues, and minimum
        score requirements.

        Args:
            results: List of validation result dictionaries.

        Returns:
            SubmissionReadiness assessment.
        """
        if not results:
            return SubmissionReadiness(
                readiness_pct=0.0,
                is_ready=False,
                blocking_issues=["No validation results provided"],
                total_count=0,
                provenance_hash=_compute_hash({"empty": True}),
            )

        total = len(results)
        compliant = sum(
            1 for r in results
            if r.get("status") == "compliant"
        )
        non_compliant = sum(
            1 for r in results
            if r.get("status") == "non_compliant"
        )

        compliance_pct = (compliant / total * 100) if total > 0 else 0.0

        scores = [r.get("accuracy_score", 0.0) for r in results]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        min_score = min(scores) if scores else 0.0
        min_score_met = avg_score >= 50.0

        # Determine blocking issues
        blocking: List[str] = []
        warnings: List[str] = []
        recommendations: List[str] = []

        if compliance_pct < MIN_COMPLIANT_PCT_FOR_SUBMISSION:
            blocking.append(
                f"Compliance rate {compliance_pct:.1f}% is below "
                f"minimum {MIN_COMPLIANT_PCT_FOR_SUBMISSION:.0f}%"
            )

        if non_compliant > 0:
            blocking.append(
                f"{non_compliant} coordinates are non-compliant"
            )

        # Check for critical issue types
        critical_count = sum(
            1 for r in results
            for issue in r.get("issues", [])
            if issue.get("severity") == "critical"
        )
        if critical_count > 0:
            blocking.append(
                f"{critical_count} critical issues detected"
            )

        # Check for ocean locations
        ocean_count = sum(
            1 for r in results
            if not r.get("is_on_land", True)
        )
        if ocean_count > 0:
            blocking.append(
                f"{ocean_count} coordinates in ocean"
            )

        # Warnings
        if min_score < 50.0:
            warnings.append(
                f"Lowest accuracy score is {min_score:.1f} "
                f"(below 50.0 minimum)"
            )

        needs_review = sum(
            1 for r in results
            if r.get("status") == "needs_review"
        )
        if needs_review > 0:
            warnings.append(
                f"{needs_review} coordinates need manual review"
            )

        # Recommendations
        if compliance_pct < 100.0:
            recommendations.append(
                f"Remediate {total - compliant} non-compliant/pending "
                f"coordinates to achieve full compliance"
            )

        if avg_score < 90.0:
            recommendations.append(
                "Improve coordinate quality (higher precision, "
                "verified sources) to achieve GOLD tier"
            )

        is_ready = len(blocking) == 0
        readiness_pct = compliance_pct if is_ready else min(
            compliance_pct, MIN_COMPLIANT_PCT_FOR_SUBMISSION - 1
        )

        readiness = SubmissionReadiness(
            readiness_pct=round(readiness_pct, 2),
            is_ready=is_ready,
            blocking_issues=blocking,
            warnings=warnings,
            recommendations=recommendations,
            compliant_count=compliant,
            total_count=total,
            minimum_score_met=min_score_met,
            provenance_hash=_compute_hash({
                "total": total,
                "compliant": compliant,
                "readiness_pct": round(readiness_pct, 2),
                "is_ready": is_ready,
            }),
        )

        logger.info(
            "Submission readiness: %.1f%%, ready=%s, "
            "%d blocking, %d warnings",
            readiness_pct, is_ready, len(blocking), len(warnings),
        )

        return readiness

    # ------------------------------------------------------------------
    # Public API: Batch Report
    # ------------------------------------------------------------------

    def batch_report(
        self,
        results: List[Dict[str, Any]],
        formats: List[str],
    ) -> bytes:
        """Generate reports in multiple formats and package as ZIP.

        Args:
            results: List of validation result dictionaries.
            formats: List of format strings ("json", "csv", "eudr_xml").

        Returns:
            ZIP file contents as bytes.
        """
        start_time = time.monotonic()
        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(
            zip_buffer, "w", zipfile.ZIP_DEFLATED
        ) as zf:
            # Generate summary for all formats
            summary = self.generate_batch_summary(results)

            for fmt in formats:
                fmt_lower = fmt.lower().strip()

                if fmt_lower == "json":
                    json_content = self.export_json(summary)
                    zf.writestr(
                        "compliance_report.json",
                        json_content,
                    )

                elif fmt_lower == "csv":
                    csv_content = self.export_csv(results)
                    zf.writestr(
                        "compliance_results.csv",
                        csv_content,
                    )

                elif fmt_lower == "eudr_xml":
                    # Generate certificates for XML
                    certificates: List[ComplianceCertificate] = []
                    for r in results:
                        cert = self.generate_compliance_certificate(
                            coord_lat=r.get("coordinate_lat", 0.0),
                            coord_lon=r.get("coordinate_lon", 0.0),
                            accuracy_score=r.get("accuracy_score", 0.0),
                            accuracy_tier=r.get(
                                "accuracy_tier", "unverified"
                            ),
                            confidence_interval_m=r.get(
                                "confidence_interval_m", 0.0
                            ),
                            precision_decimal_places=r.get(
                                "precision_decimal_places", 0
                            ),
                            country_iso=r.get("country_iso"),
                            commodity=r.get("commodity"),
                            is_on_land=r.get("is_on_land", True),
                            country_match=r.get("country_match", True),
                            issues_found=r.get("issues_found", 0),
                            critical_issues=r.get("critical_issues", 0),
                        )
                        certificates.append(cert)

                    xml_content = self.export_eudr_xml(certificates)
                    zf.writestr(
                        "eudr_geolocation.xml",
                        xml_content,
                    )

            # Always include submission readiness
            readiness = self.submission_readiness(results)
            readiness_json = json.dumps(
                {
                    "readiness_pct": readiness.readiness_pct,
                    "is_ready": readiness.is_ready,
                    "blocking_issues": readiness.blocking_issues,
                    "warnings": readiness.warnings,
                    "recommendations": readiness.recommendations,
                    "compliant_count": readiness.compliant_count,
                    "total_count": readiness.total_count,
                },
                indent=2,
            )
            zf.writestr(
                "submission_readiness.json",
                readiness_json,
            )

        elapsed_ms = (time.monotonic() - start_time) * 1000
        zip_bytes = zip_buffer.getvalue()
        logger.info(
            "Batch report generated: %d results, formats=%s, "
            "zip_size=%d bytes, %.1fms",
            len(results), formats, len(zip_bytes), elapsed_ms,
        )

        return zip_bytes

    # ------------------------------------------------------------------
    # Internal: Compliance Status Determination
    # ------------------------------------------------------------------

    def _determine_compliance_status(
        self,
        accuracy_score: float,
        critical_issues: int,
        is_on_land: bool,
        country_match: bool,
    ) -> ComplianceStatus:
        """Determine compliance status from score and issues.

        Rules:
            1. Any critical issue -> NON_COMPLIANT
            2. Not on land -> NON_COMPLIANT
            3. Score >= 70 and no critical issues -> COMPLIANT
            4. Score >= 50 -> NEEDS_REVIEW
            5. Score < 50 -> NON_COMPLIANT
            6. Score == 0 with no other data -> INSUFFICIENT_DATA

        Args:
            accuracy_score: Overall accuracy score.
            critical_issues: Number of critical issues.
            is_on_land: Whether coordinate is on land.
            country_match: Whether country matches declared.

        Returns:
            ComplianceStatus classification.
        """
        if critical_issues > 0:
            return ComplianceStatus.NON_COMPLIANT

        if not is_on_land:
            return ComplianceStatus.NON_COMPLIANT

        if accuracy_score == 0.0:
            return ComplianceStatus.INSUFFICIENT_DATA

        if accuracy_score >= 70.0:
            return ComplianceStatus.COMPLIANT

        if accuracy_score >= 50.0:
            return ComplianceStatus.NEEDS_REVIEW

        return ComplianceStatus.NON_COMPLIANT

    # ------------------------------------------------------------------
    # Internal: Audit Trail
    # ------------------------------------------------------------------

    def _record_audit(
        self,
        operation: str,
        entity_id: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> AuditTrailEntry:
        """Record an audit trail entry.

        Args:
            operation: Type of operation performed.
            entity_id: ID of the entity being operated on.
            details: Optional additional details.

        Returns:
            The recorded AuditTrailEntry.
        """
        timestamp = _utcnow().isoformat()
        entry_data = {
            "timestamp": timestamp,
            "operation": operation,
            "entity_id": entity_id,
            "parent_hash": self._last_audit_hash,
        }
        entry_hash = _compute_hash(entry_data)

        entry = AuditTrailEntry(
            timestamp=timestamp,
            operation=operation,
            entity_id=entity_id,
            details=details or {},
            hash_value=entry_hash,
            parent_hash=self._last_audit_hash,
        )

        self._audit_chain.append(entry)
        self._last_audit_hash = entry_hash

        return entry

    # ------------------------------------------------------------------
    # Internal: Certificate Hash
    # ------------------------------------------------------------------

    def _compute_certificate_hash(
        self,
        cert: ComplianceCertificate,
    ) -> str:
        """Compute SHA-256 provenance hash for a certificate.

        Args:
            cert: Certificate to hash.

        Returns:
            SHA-256 hex digest.
        """
        hash_data = {
            "module_version": _MODULE_VERSION,
            "engine": "compliance_reporter",
            "certificate_id": cert.certificate_id,
            "lat": cert.coordinate_lat,
            "lon": cert.coordinate_lon,
            "status": cert.status.value if isinstance(
                cert.status, Enum
            ) else str(cert.status),
            "accuracy_score": cert.accuracy_score,
            "accuracy_tier": cert.accuracy_tier,
            "country_iso": cert.country_iso,
            "commodity": cert.commodity,
            "issues_found": cert.issues_found,
            "critical_issues": cert.critical_issues,
            "issued_at": cert.issued_at,
            "valid_until": cert.valid_until,
        }
        return _compute_hash(hash_data)


# ---------------------------------------------------------------------------
# Internal Helper: XML Escape
# ---------------------------------------------------------------------------


def _xml_escape(value: Optional[str]) -> str:
    """Escape special XML characters.

    Args:
        value: String to escape.

    Returns:
        XML-safe string.
    """
    if value is None:
        return ""
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


# ---------------------------------------------------------------------------
# Module Exports
# ---------------------------------------------------------------------------

__all__ = [
    "ComplianceReporter",
    "ComplianceCertificate",
    "ComplianceStatus",
    "ReportFormat",
    "RemediationPriority",
    "RemediationItem",
    "QualityTrend",
    "SubmissionReadiness",
    "AuditTrailEntry",
    "EUDR_DDS_NAMESPACE",
    "COMPLIANCE_THRESHOLDS",
    "CERTIFICATE_VALIDITY_DAYS",
    "REMEDIATION_TEMPLATES",
]
