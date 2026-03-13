# -*- coding: utf-8 -*-
"""
Compliance Monitor - AGENT-EUDR-008 Engine 6

Production-grade compliance status monitoring engine for multi-tier
supplier tracking under the EU Deforestation Regulation (EUDR).
Continuously monitors supplier compliance across four dimensions
(DDS validity, certification status, geolocation coverage,
deforestation-free verification), calculates composite compliance
scores, classifies status, generates alerts on status changes, and
tracks compliance history.

Zero-Hallucination Guarantees:
    - All compliance scores are deterministic arithmetic
    - Compliance status thresholds are fixed per PRD Appendix C
    - Expiry calculations use standard datetime arithmetic
    - No ML/LLM used in any compliance determination
    - SHA-256 provenance chain hashing on all results

Performance Targets:
    - Single compliance check: <5ms
    - Batch compliance check (10,000 suppliers): <5s
    - Compliance history retrieval: <2ms

Regulatory References:
    - EUDR Article 4: Due diligence system maintenance
    - EUDR Article 9: Traceability information requirements
    - EUDR Article 10: Trader obligations (DDS references)
    - EUDR Article 14: Competent authority data requests (5-year retention)
    - PRD Appendix C: Compliance Status Definitions

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-008 (Engine 6: Compliance Monitoring)
Agent ID: GL-EUDR-MST-008
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
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


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    """Clamp a numeric value to [lo, hi]."""
    return max(lo, min(hi, value))


def _parse_datetime(dt_str: str) -> Optional[datetime]:
    """Parse an ISO datetime string to a timezone-aware datetime.

    Args:
        dt_str: ISO formatted datetime string.

    Returns:
        Timezone-aware datetime or None if parsing fails.
    """
    if not dt_str or not dt_str.strip():
        return None
    try:
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ComplianceStatus(str, Enum):
    """Compliance status per PRD Appendix C.

    COMPLIANT: All checks pass; valid DDS, certifications, GPS.
        Can include in DDS submission.
    CONDITIONALLY_COMPLIANT: Minor gaps; remediation in progress.
        Can include with disclosure.
    NON_COMPLIANT: Critical gaps; failed checks.
        Cannot include in DDS submission.
    UNVERIFIED: Not yet assessed; insufficient data.
        Cannot include in DDS submission.
    EXPIRED: Previously compliant; certifications/DDS expired.
        Cannot include until renewed.
    """

    COMPLIANT = "compliant"
    CONDITIONALLY_COMPLIANT = "conditionally_compliant"
    NON_COMPLIANT = "non_compliant"
    UNVERIFIED = "unverified"
    EXPIRED = "expired"


class ComplianceDimension(str, Enum):
    """Four compliance dimensions per PRD F6.1."""

    DDS_VALIDITY = "dds_validity"
    CERTIFICATION_STATUS = "certification_status"
    GEOLOCATION_COVERAGE = "geolocation_coverage"
    DEFORESTATION_FREE = "deforestation_free"


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class TrendDirection(str, Enum):
    """Compliance trend direction."""

    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"


# ---------------------------------------------------------------------------
# Compliance Score Thresholds (PRD Appendix C)
# ---------------------------------------------------------------------------

#: Compliance status classification thresholds.
COMPLIANCE_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    ComplianceStatus.COMPLIANT.value: (80.0, 100.0),
    ComplianceStatus.CONDITIONALLY_COMPLIANT.value: (60.0, 80.0),
    ComplianceStatus.NON_COMPLIANT.value: (0.0, 60.0),
}

#: DDS and certification expiry warning thresholds (days before expiry).
EXPIRY_WARNING_DAYS: List[int] = [90, 60, 30, 14, 7]

#: Compliance dimension weights for composite score.
COMPLIANCE_DIMENSION_WEIGHTS: Dict[str, float] = {
    ComplianceDimension.DDS_VALIDITY.value: 0.30,
    ComplianceDimension.CERTIFICATION_STATUS.value: 0.25,
    ComplianceDimension.GEOLOCATION_COVERAGE.value: 0.25,
    ComplianceDimension.DEFORESTATION_FREE.value: 0.20,
}

#: Recognized certification types for EUDR compliance.
RECOGNIZED_CERTIFICATIONS: Dict[str, Dict[str, Any]] = {
    "fsc": {"name": "Forest Stewardship Council", "commodities": ["wood", "timber", "rubber"]},
    "pefc": {"name": "PEFC", "commodities": ["wood", "timber"]},
    "rspo": {"name": "Roundtable on Sustainable Palm Oil", "commodities": ["palm_oil"]},
    "rspo_ip": {"name": "RSPO Identity Preserved", "commodities": ["palm_oil"]},
    "rspo_mb": {"name": "RSPO Mass Balance", "commodities": ["palm_oil"]},
    "rspo_sg": {"name": "RSPO Segregated", "commodities": ["palm_oil"]},
    "utz": {"name": "UTZ Certified", "commodities": ["cocoa", "coffee"]},
    "rainforest_alliance": {"name": "Rainforest Alliance", "commodities": ["cocoa", "coffee", "palm_oil"]},
    "fairtrade": {"name": "Fairtrade International", "commodities": ["cocoa", "coffee"]},
    "iscc": {"name": "International Sustainability and Carbon Certification", "commodities": ["soya", "palm_oil"]},
    "rtrs": {"name": "Round Table on Responsible Soy", "commodities": ["soya"]},
    "4c": {"name": "4C Association", "commodities": ["coffee"]},
    "bonsucro": {"name": "Bonsucro", "commodities": ["cattle"]},
    "globalg.a.p.": {"name": "GLOBALG.A.P.", "commodities": ["cattle", "soya"]},
}


# ---------------------------------------------------------------------------
# Result Data Classes
# ---------------------------------------------------------------------------


@dataclass
class SupplierComplianceProfile:
    """Minimal supplier profile data required for compliance monitoring.

    This is a local model used within the compliance monitor to decouple
    from external model dependencies.

    Attributes:
        supplier_id: Unique supplier identifier.
        legal_name: Legal entity name.
        country_iso: ISO 3166-1 alpha-2 country code.
        tier: Tier level in supply chain (1 = direct).
        commodity_types: EUDR commodity types handled.
        certifications: List of certification records.
        gps_latitude: Production plot latitude (WGS84).
        gps_longitude: Production plot longitude (WGS84).
        dds_references: List of DDS references with validity info.
        deforestation_free_status: Verification status string.
        deforestation_verified_date: Date of last deforestation-free check.
        geolocation_plots: Number of production plots with GPS.
        total_plots: Total number of production plots.
        last_compliance_check: ISO datetime of last check.
        registration_id: Legal registration ID.
    """

    supplier_id: str = ""
    legal_name: str = ""
    country_iso: str = ""
    tier: int = 1
    commodity_types: List[str] = field(default_factory=list)
    certifications: List[Dict[str, Any]] = field(default_factory=list)
    gps_latitude: Optional[float] = None
    gps_longitude: Optional[float] = None
    dds_references: List[Dict[str, Any]] = field(default_factory=list)
    deforestation_free_status: str = ""
    deforestation_verified_date: str = ""
    geolocation_plots: int = 0
    total_plots: int = 0
    last_compliance_check: str = ""
    registration_id: str = ""


@dataclass
class DimensionCheckResult:
    """Result of a single compliance dimension check.

    Attributes:
        dimension: Compliance dimension checked.
        score: Dimension score (0-100).
        weight: Dimension weight for composite.
        weighted_score: score * weight.
        status: Dimension-level status.
        details: Supporting evidence and findings.
        issues: List of issues found.
    """

    dimension: str = ""
    score: float = 0.0
    weight: float = 0.0
    weighted_score: float = 0.0
    status: str = ComplianceStatus.UNVERIFIED.value
    details: Dict[str, Any] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)


@dataclass
class ComplianceCheckResult:
    """Full compliance assessment result for a supplier.

    Attributes:
        check_id: Unique UUID4 check identifier.
        supplier_id: Supplier assessed.
        composite_score: Weighted composite compliance score (0-100).
        status: Overall compliance status classification.
        dimension_results: Per-dimension check results.
        dds_can_include: Whether supplier can be included in DDS.
        issues: Combined list of all issues found.
        checked_at: UTC ISO timestamp of check.
        processing_time_ms: Check duration in milliseconds.
        provenance_hash: SHA-256 hash for audit trail.
        engine_version: Engine version string.
    """

    check_id: str = ""
    supplier_id: str = ""
    composite_score: float = 0.0
    status: str = ComplianceStatus.UNVERIFIED.value
    dimension_results: List[DimensionCheckResult] = field(
        default_factory=list
    )
    dds_can_include: bool = False
    issues: List[str] = field(default_factory=list)
    checked_at: str = ""
    processing_time_ms: float = 0.0
    provenance_hash: str = ""
    engine_version: str = _MODULE_VERSION


@dataclass
class ComplianceAlert:
    """Alert generated on compliance status change or expiry warning.

    Attributes:
        alert_id: Unique UUID4 alert identifier.
        supplier_id: Supplier triggering the alert.
        alert_type: Type of alert (status_change, expiry_warning, etc.).
        severity: Alert severity level.
        previous_status: Previous compliance status (for transitions).
        current_status: Current compliance status.
        message: Human-readable alert message.
        dimension: Compliance dimension that triggered the alert.
        days_until_expiry: Days until expiry (for expiry warnings).
        triggered_at: UTC ISO timestamp.
        provenance_hash: SHA-256 hash for audit trail.
    """

    alert_id: str = ""
    supplier_id: str = ""
    alert_type: str = ""
    severity: str = AlertSeverity.MEDIUM.value
    previous_status: str = ""
    current_status: str = ""
    message: str = ""
    dimension: str = ""
    days_until_expiry: Optional[int] = None
    triggered_at: str = ""
    provenance_hash: str = ""


@dataclass
class ExpiryWarning:
    """Warning for upcoming DDS or certification expiry.

    Attributes:
        warning_id: Unique UUID4 warning identifier.
        supplier_id: Supplier with upcoming expiry.
        item_type: Type of item expiring (dds, certification).
        item_id: ID of the expiring item.
        item_name: Description of the expiring item.
        expires_at: Expiry date ISO string.
        days_until_expiry: Days remaining until expiry.
        severity: Warning severity based on days remaining.
        action_required: Recommended action.
    """

    warning_id: str = ""
    supplier_id: str = ""
    item_type: str = ""
    item_id: str = ""
    item_name: str = ""
    expires_at: str = ""
    days_until_expiry: int = 0
    severity: str = AlertSeverity.LOW.value
    action_required: str = ""


@dataclass
class ComplianceHistoryEntry:
    """A single entry in a supplier's compliance history timeline.

    Attributes:
        timestamp: UTC ISO timestamp of the check.
        composite_score: Score at this point in time.
        status: Compliance status at this point.
        dimension_scores: Per-dimension score snapshot.
        change_reason: What changed from previous entry.
    """

    timestamp: str = ""
    composite_score: float = 0.0
    status: str = ComplianceStatus.UNVERIFIED.value
    dimension_scores: Dict[str, float] = field(default_factory=dict)
    change_reason: str = ""


@dataclass
class BatchComplianceResult:
    """Batch compliance checking result.

    Attributes:
        batch_id: Unique UUID4 batch identifier.
        total_suppliers: Total suppliers checked.
        successful: Number successfully checked.
        failed: Number that failed checking.
        results: Individual check results.
        summary: Aggregate statistics.
        processing_time_ms: Total batch duration.
        provenance_hash: SHA-256 hash of entire batch.
    """

    batch_id: str = ""
    total_suppliers: int = 0
    successful: int = 0
    failed: int = 0
    results: List[ComplianceCheckResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    processing_time_ms: float = 0.0
    provenance_hash: str = ""


# ===========================================================================
# ComplianceMonitor
# ===========================================================================


class ComplianceMonitor:
    """Production-grade compliance monitoring engine for EUDR multi-tier suppliers.

    Monitors supplier compliance across four dimensions (DDS validity,
    certification status, geolocation coverage, deforestation-free
    verification), calculates composite compliance scores, classifies
    status per PRD Appendix C, generates alerts on transitions, and
    tracks compliance history.

    All determinations are deterministic with zero LLM/ML involvement.

    Attributes:
        _dimension_weights: Compliance dimension weights.
        _compliance_history: In-memory compliance history per supplier.
        _check_count: Running count of compliance checks performed.
        _alert_count: Running count of alerts generated.

    Example::

        monitor = ComplianceMonitor()
        profile = SupplierComplianceProfile(
            supplier_id="SUP-001",
            dds_references=[{
                "dds_id": "DDS-2025-001",
                "valid_until": "2026-12-31T00:00:00+00:00",
            }],
        )
        result = monitor.check_compliance("SUP-001", profile)
        assert result.status in ["compliant", "conditionally_compliant",
                                  "non_compliant", "unverified", "expired"]
    """

    def __init__(
        self,
        dimension_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        """Initialize ComplianceMonitor.

        Args:
            dimension_weights: Optional custom dimension weights. Must
                sum to 1.0 (+/- 0.001). Defaults to COMPLIANCE_DIMENSION_WEIGHTS.

        Raises:
            ValueError: If custom weights do not sum to approximately 1.0.
        """
        if dimension_weights is not None:
            weight_sum = sum(dimension_weights.values())
            if abs(weight_sum - 1.0) > 0.001:
                raise ValueError(
                    f"Dimension weights must sum to 1.0, got {weight_sum:.4f}"
                )
            self._dimension_weights: Dict[str, float] = dict(
                dimension_weights
            )
        else:
            self._dimension_weights = dict(COMPLIANCE_DIMENSION_WEIGHTS)

        self._compliance_history: Dict[
            str, List[ComplianceCheckResult]
        ] = {}
        self._check_count: int = 0
        self._alert_count: int = 0

        logger.info(
            "ComplianceMonitor initialized with dimension weights: %s",
            {k: f"{v:.2f}" for k, v in self._dimension_weights.items()},
        )

    # ------------------------------------------------------------------
    # Public API: Full Compliance Check
    # ------------------------------------------------------------------

    def check_compliance(
        self,
        supplier_id: str,
        profile: SupplierComplianceProfile,
    ) -> ComplianceCheckResult:
        """Perform full compliance assessment for a supplier.

        Evaluates four compliance dimensions, calculates composite score,
        classifies status, and determines DDS inclusion eligibility.

        Args:
            supplier_id: Unique supplier identifier.
            profile: Supplier compliance profile data.

        Returns:
            ComplianceCheckResult with composite score and status.

        Raises:
            ValueError: If supplier_id is empty.
        """
        if not supplier_id:
            raise ValueError("supplier_id must not be empty")

        t_start = time.monotonic()
        check_id = str(uuid.uuid4())
        dimension_results: List[DimensionCheckResult] = []
        all_issues: List[str] = []

        # Check each compliance dimension
        dds_result = self.check_dds_validity(profile.dds_references)
        dimension_results.append(dds_result)
        all_issues.extend(dds_result.issues)

        cert_result = self.check_certification_status(
            profile.certifications
        )
        dimension_results.append(cert_result)
        all_issues.extend(cert_result.issues)

        geo_result = self.check_geolocation_coverage(profile)
        dimension_results.append(geo_result)
        all_issues.extend(geo_result.issues)

        deforest_result = self.check_deforestation_status(profile)
        dimension_results.append(deforest_result)
        all_issues.extend(deforest_result.issues)

        # Calculate composite score
        composite_score = self.calculate_compliance_score(dimension_results)

        # Check for expired status
        has_expired = self._has_expired_items(profile)

        # Check for insufficient data
        has_sufficient_data = self._has_sufficient_data(profile)

        # Classify status
        if not has_sufficient_data:
            status = ComplianceStatus.UNVERIFIED
        elif has_expired:
            status = ComplianceStatus.EXPIRED
        else:
            status = self.classify_compliance_status(composite_score)

        # Determine DDS inclusion eligibility
        dds_can_include = status in (
            ComplianceStatus.COMPLIANT,
            ComplianceStatus.CONDITIONALLY_COMPLIANT,
        )

        elapsed_ms = (time.monotonic() - t_start) * 1000.0

        # Build provenance hash
        provenance_data = {
            "check_id": check_id,
            "supplier_id": supplier_id,
            "composite_score": round(composite_score, 4),
            "status": status.value,
            "dimension_scores": {
                dr.dimension: round(dr.score, 4) for dr in dimension_results
            },
            "engine_version": _MODULE_VERSION,
        }
        provenance_hash = _compute_hash(provenance_data)

        result = ComplianceCheckResult(
            check_id=check_id,
            supplier_id=supplier_id,
            composite_score=round(composite_score, 2),
            status=status.value,
            dimension_results=dimension_results,
            dds_can_include=dds_can_include,
            issues=all_issues,
            checked_at=_utcnow().isoformat(),
            processing_time_ms=round(elapsed_ms, 3),
            provenance_hash=provenance_hash,
            engine_version=_MODULE_VERSION,
        )

        # Track history
        if supplier_id not in self._compliance_history:
            self._compliance_history[supplier_id] = []
        self._compliance_history[supplier_id].append(result)
        self._check_count += 1

        logger.info(
            "Compliance check completed: supplier=%s score=%.1f status=%s "
            "dds_eligible=%s issues=%d time=%.2fms hash_prefix=%s",
            supplier_id,
            composite_score,
            status.value,
            dds_can_include,
            len(all_issues),
            elapsed_ms,
            provenance_hash[:16],
        )

        return result

    # ------------------------------------------------------------------
    # Public API: Individual Dimension Checks
    # ------------------------------------------------------------------

    def check_dds_validity(
        self,
        dds_refs: List[Dict[str, Any]],
    ) -> DimensionCheckResult:
        """Check DDS (Due Diligence Statement) validity and expiry.

        Evaluates all DDS references for validity: present, not expired,
        properly formatted.

        Scoring Logic:
            - No DDS references: 0 (unverified)
            - All DDS expired: 10 (expired)
            - Some DDS expired, some valid: 50
            - All DDS valid, expiring within 30 days: 70
            - All DDS valid, expiring within 90 days: 85
            - All DDS valid, >90 days remaining: 100

        Args:
            dds_refs: List of DDS reference dicts with "dds_id" and
                "valid_until" keys. Optional: "issued_at", "authority".

        Returns:
            DimensionCheckResult for DDS validity.
        """
        weight = self._dimension_weights.get(
            ComplianceDimension.DDS_VALIDITY.value, 0.30
        )
        now = _utcnow()
        issues: List[str] = []
        details: Dict[str, Any] = {
            "total_dds": len(dds_refs),
            "valid_dds": 0,
            "expired_dds": 0,
            "min_days_until_expiry": None,
        }

        if not dds_refs:
            score = 0.0
            status = ComplianceStatus.UNVERIFIED.value
            issues.append("No DDS references found for this supplier")
            details["reason"] = "no_dds_references"
        else:
            valid_count = 0
            expired_count = 0
            min_days_remaining: Optional[int] = None

            for dds in dds_refs:
                dds_id = dds.get("dds_id", "unknown")
                valid_until_str = str(dds.get("valid_until", ""))
                valid_until = _parse_datetime(valid_until_str)

                if valid_until is None:
                    issues.append(
                        f"DDS {dds_id}: no valid_until date specified"
                    )
                    continue

                days_remaining = (valid_until - now).days

                if days_remaining < 0:
                    expired_count += 1
                    issues.append(
                        f"DDS {dds_id}: expired "
                        f"{abs(days_remaining)} days ago"
                    )
                else:
                    valid_count += 1
                    if (
                        min_days_remaining is None
                        or days_remaining < min_days_remaining
                    ):
                        min_days_remaining = days_remaining

            details["valid_dds"] = valid_count
            details["expired_dds"] = expired_count
            details["min_days_until_expiry"] = min_days_remaining

            if valid_count == 0:
                score = 10.0
                status = ComplianceStatus.EXPIRED.value
                details["reason"] = "all_dds_expired"
            elif expired_count > 0:
                score = 50.0
                status = ComplianceStatus.CONDITIONALLY_COMPLIANT.value
                details["reason"] = "some_dds_expired"
                issues.append(
                    f"{expired_count} of {len(dds_refs)} DDS references "
                    "have expired"
                )
            elif min_days_remaining is not None and min_days_remaining <= 30:
                score = 70.0
                status = ComplianceStatus.CONDITIONALLY_COMPLIANT.value
                details["reason"] = "dds_expiring_within_30_days"
                issues.append(
                    f"DDS expires in {min_days_remaining} days"
                )
            elif min_days_remaining is not None and min_days_remaining <= 90:
                score = 85.0
                status = ComplianceStatus.COMPLIANT.value
                details["reason"] = "dds_expiring_within_90_days"
            else:
                score = 100.0
                status = ComplianceStatus.COMPLIANT.value
                details["reason"] = "all_dds_valid"

        weighted_score = score * weight

        logger.debug(
            "DDS validity check: total=%d valid=%d expired=%d score=%.1f",
            len(dds_refs),
            details.get("valid_dds", 0),
            details.get("expired_dds", 0),
            score,
        )

        return DimensionCheckResult(
            dimension=ComplianceDimension.DDS_VALIDITY.value,
            score=round(score, 2),
            weight=weight,
            weighted_score=round(weighted_score, 2),
            status=status,
            details=details,
            issues=issues,
        )

    def check_certification_status(
        self,
        certifications: List[Dict[str, Any]],
    ) -> DimensionCheckResult:
        """Check certification validity for EUDR compliance.

        Evaluates certifications for: presence, recognized type, validity
        period, and commodity relevance.

        Scoring Logic:
            - No certifications: 20 (certification not mandatory but
              strongly recommended for EUDR)
            - All expired: 15
            - Valid but unrecognized: 50
            - Valid recognized, expiring <30 days: 70
            - Valid recognized, expiring <90 days: 85
            - Valid recognized, >90 days remaining: 100

        Args:
            certifications: List of certification dicts with "type",
                "valid_until" keys. Optional: "certificate_id", "valid_from",
                "commodities".

        Returns:
            DimensionCheckResult for certification status.
        """
        weight = self._dimension_weights.get(
            ComplianceDimension.CERTIFICATION_STATUS.value, 0.25
        )
        now = _utcnow()
        issues: List[str] = []
        details: Dict[str, Any] = {
            "total_certifications": len(certifications),
            "valid_certifications": 0,
            "expired_certifications": 0,
            "recognized_certifications": 0,
        }

        if not certifications:
            score = 20.0
            status = ComplianceStatus.CONDITIONALLY_COMPLIANT.value
            issues.append(
                "No certifications found; certification is strongly "
                "recommended for EUDR compliance"
            )
            details["reason"] = "no_certifications"
        else:
            valid_recognized = 0
            valid_unrecognized = 0
            expired_count = 0
            min_days: Optional[int] = None

            for cert in certifications:
                cert_type = str(cert.get("type", "")).lower().strip()
                valid_until_str = str(cert.get("valid_until", ""))
                valid_until = _parse_datetime(valid_until_str)

                is_valid = True
                if valid_until is not None:
                    days_remaining = (valid_until - now).days
                    if days_remaining < 0:
                        is_valid = False
                        expired_count += 1
                        issues.append(
                            f"Certification {cert_type}: expired "
                            f"{abs(days_remaining)} days ago"
                        )
                    else:
                        if min_days is None or days_remaining < min_days:
                            min_days = days_remaining
                else:
                    # No expiry date: treat as valid but flag
                    issues.append(
                        f"Certification {cert_type}: no valid_until date"
                    )

                if is_valid:
                    if cert_type in RECOGNIZED_CERTIFICATIONS:
                        valid_recognized += 1
                    else:
                        valid_unrecognized += 1

            details["valid_certifications"] = (
                valid_recognized + valid_unrecognized
            )
            details["expired_certifications"] = expired_count
            details["recognized_certifications"] = valid_recognized
            details["min_days_until_expiry"] = min_days

            if valid_recognized == 0 and valid_unrecognized == 0:
                score = 15.0
                status = ComplianceStatus.EXPIRED.value
                details["reason"] = "all_certifications_expired"
            elif valid_recognized == 0 and valid_unrecognized > 0:
                score = 50.0
                status = ComplianceStatus.CONDITIONALLY_COMPLIANT.value
                details["reason"] = "valid_but_unrecognized_certifications"
                issues.append(
                    "No recognized EUDR-relevant certifications found"
                )
            elif min_days is not None and min_days <= 30:
                score = 70.0
                status = ComplianceStatus.CONDITIONALLY_COMPLIANT.value
                details["reason"] = "certification_expiring_within_30_days"
                issues.append(
                    f"Certification expires in {min_days} days"
                )
            elif min_days is not None and min_days <= 90:
                score = 85.0
                status = ComplianceStatus.COMPLIANT.value
                details["reason"] = "certification_expiring_within_90_days"
            else:
                score = 100.0
                status = ComplianceStatus.COMPLIANT.value
                details["reason"] = "valid_recognized_certifications"

        weighted_score = score * weight

        logger.debug(
            "Certification status check: total=%d valid=%d expired=%d "
            "score=%.1f",
            len(certifications),
            details.get("valid_certifications", 0),
            details.get("expired_certifications", 0),
            score,
        )

        return DimensionCheckResult(
            dimension=ComplianceDimension.CERTIFICATION_STATUS.value,
            score=round(score, 2),
            weight=weight,
            weighted_score=round(weighted_score, 2),
            status=status,
            details=details,
            issues=issues,
        )

    def check_geolocation_coverage(
        self,
        supplier: SupplierComplianceProfile,
    ) -> DimensionCheckResult:
        """Check GPS geolocation coverage percentage.

        EUDR Article 9(1)(d) requires geolocation of all production
        plots. This checks the percentage of plots with valid GPS
        coordinates.

        Scoring Logic:
            - No GPS data at all: 0
            - Coverage < 25%: 20
            - Coverage 25-50%: 40
            - Coverage 50-75%: 60
            - Coverage 75-90%: 80
            - Coverage 90-99%: 90
            - Coverage 100%: 100

        Args:
            supplier: Supplier profile with geolocation data.

        Returns:
            DimensionCheckResult for geolocation coverage.
        """
        weight = self._dimension_weights.get(
            ComplianceDimension.GEOLOCATION_COVERAGE.value, 0.25
        )
        issues: List[str] = []
        details: Dict[str, Any] = {
            "geolocation_plots": supplier.geolocation_plots,
            "total_plots": supplier.total_plots,
        }

        # Check if supplier has GPS coordinates at profile level
        has_profile_gps = (
            supplier.gps_latitude is not None
            and supplier.gps_longitude is not None
        )

        if supplier.total_plots == 0 and not has_profile_gps:
            score = 0.0
            status = ComplianceStatus.UNVERIFIED.value
            details["reason"] = "no_geolocation_data"
            details["coverage_pct"] = 0.0
            issues.append(
                "No GPS coordinates or production plot data available"
            )
        elif supplier.total_plots == 0 and has_profile_gps:
            # Has profile-level GPS but no plot breakdown
            score = 60.0
            status = ComplianceStatus.CONDITIONALLY_COMPLIANT.value
            details["reason"] = "profile_gps_only_no_plot_breakdown"
            details["coverage_pct"] = None
            issues.append(
                "GPS coordinates at profile level only; individual "
                "production plot coordinates not provided"
            )
        else:
            coverage_pct = (
                supplier.geolocation_plots / supplier.total_plots * 100.0
            )
            details["coverage_pct"] = round(coverage_pct, 2)

            if coverage_pct >= 100.0:
                score = 100.0
                status = ComplianceStatus.COMPLIANT.value
                details["reason"] = "full_geolocation_coverage"
            elif coverage_pct >= 90.0:
                score = 90.0
                status = ComplianceStatus.COMPLIANT.value
                details["reason"] = "near_full_coverage"
                issues.append(
                    f"{supplier.total_plots - supplier.geolocation_plots} "
                    "plots missing GPS coordinates"
                )
            elif coverage_pct >= 75.0:
                score = 80.0
                status = ComplianceStatus.CONDITIONALLY_COMPLIANT.value
                details["reason"] = "good_coverage"
                issues.append(
                    f"GPS coverage at {coverage_pct:.1f}%; "
                    "target 100% for full EUDR compliance"
                )
            elif coverage_pct >= 50.0:
                score = 60.0
                status = ComplianceStatus.CONDITIONALLY_COMPLIANT.value
                details["reason"] = "moderate_coverage"
                issues.append(
                    f"GPS coverage at {coverage_pct:.1f}%; significant "
                    "gaps in production plot geolocation"
                )
            elif coverage_pct >= 25.0:
                score = 40.0
                status = ComplianceStatus.NON_COMPLIANT.value
                details["reason"] = "low_coverage"
                issues.append(
                    f"GPS coverage at {coverage_pct:.1f}%; majority of "
                    "production plots lack geolocation"
                )
            else:
                score = 20.0
                status = ComplianceStatus.NON_COMPLIANT.value
                details["reason"] = "very_low_coverage"
                issues.append(
                    f"GPS coverage at {coverage_pct:.1f}%; critical "
                    "deficiency in geolocation data"
                )

        weighted_score = score * weight

        logger.debug(
            "Geolocation coverage check: supplier=%s plots=%d/%d "
            "score=%.1f",
            supplier.supplier_id,
            supplier.geolocation_plots,
            supplier.total_plots,
            score,
        )

        return DimensionCheckResult(
            dimension=ComplianceDimension.GEOLOCATION_COVERAGE.value,
            score=round(score, 2),
            weight=weight,
            weighted_score=round(weighted_score, 2),
            status=status,
            details=details,
            issues=issues,
        )

    def check_deforestation_status(
        self,
        supplier: SupplierComplianceProfile,
    ) -> DimensionCheckResult:
        """Check deforestation-free verification status.

        EUDR requires products to be deforestation-free (produced on land
        not deforested after December 31, 2020). This checks the
        verification status of the supplier.

        Scoring Logic:
            - Not assessed: 0
            - Self-declared only: 40
            - Third-party verified but outdated (>1 year): 60
            - Third-party verified within 1 year: 85
            - Third-party verified with satellite confirmation: 100

        Args:
            supplier: Supplier profile with deforestation verification data.

        Returns:
            DimensionCheckResult for deforestation-free status.
        """
        weight = self._dimension_weights.get(
            ComplianceDimension.DEFORESTATION_FREE.value, 0.20
        )
        now = _utcnow()
        issues: List[str] = []
        status_str = (
            supplier.deforestation_free_status.lower().strip()
            if supplier.deforestation_free_status
            else ""
        )
        details: Dict[str, Any] = {
            "verification_status": status_str or "not_assessed",
            "verification_date": supplier.deforestation_verified_date or None,
        }

        if not status_str or status_str == "not_assessed":
            score = 0.0
            status = ComplianceStatus.UNVERIFIED.value
            details["reason"] = "not_assessed"
            issues.append(
                "Deforestation-free status has not been assessed"
            )
        elif status_str == "self_declared":
            score = 40.0
            status = ComplianceStatus.CONDITIONALLY_COMPLIANT.value
            details["reason"] = "self_declared_only"
            issues.append(
                "Deforestation-free status is self-declared only; "
                "third-party verification recommended"
            )
        elif status_str in ("verified", "third_party_verified"):
            verified_date = _parse_datetime(
                supplier.deforestation_verified_date
            )
            if verified_date is None:
                score = 60.0
                status = ComplianceStatus.CONDITIONALLY_COMPLIANT.value
                details["reason"] = "verified_no_date"
                issues.append(
                    "Verification date not provided for deforestation-free "
                    "assessment"
                )
            else:
                days_since = (now - verified_date).days
                details["days_since_verification"] = days_since

                if days_since > 365:
                    score = 60.0
                    status = ComplianceStatus.CONDITIONALLY_COMPLIANT.value
                    details["reason"] = "verified_outdated"
                    issues.append(
                        f"Deforestation-free verification is {days_since} "
                        "days old; re-verification recommended within "
                        "12 months"
                    )
                else:
                    score = 85.0
                    status = ComplianceStatus.COMPLIANT.value
                    details["reason"] = "verified_current"
        elif status_str in (
            "satellite_confirmed",
            "verified_satellite",
        ):
            score = 100.0
            status = ComplianceStatus.COMPLIANT.value
            details["reason"] = "satellite_confirmed"
        elif status_str == "non_compliant":
            score = 0.0
            status = ComplianceStatus.NON_COMPLIANT.value
            details["reason"] = "deforestation_detected"
            issues.append(
                "Deforestation detected in supplier's sourcing area; "
                "EUDR non-compliant"
            )
        else:
            score = 20.0
            status = ComplianceStatus.UNVERIFIED.value
            details["reason"] = "unknown_status"
            issues.append(
                f"Unknown deforestation-free status: {status_str}"
            )

        weighted_score = score * weight

        logger.debug(
            "Deforestation status check: supplier=%s status=%s score=%.1f",
            supplier.supplier_id,
            status_str or "not_assessed",
            score,
        )

        return DimensionCheckResult(
            dimension=ComplianceDimension.DEFORESTATION_FREE.value,
            score=round(score, 2),
            weight=weight,
            weighted_score=round(weighted_score, 2),
            status=status,
            details=details,
            issues=issues,
        )

    # ------------------------------------------------------------------
    # Public API: Composite Score and Classification
    # ------------------------------------------------------------------

    def calculate_compliance_score(
        self,
        dimension_results: List[DimensionCheckResult],
    ) -> float:
        """Calculate composite compliance score from dimension results.

        Composite score = sum of (dimension_score * dimension_weight).

        Args:
            dimension_results: List of dimension check results.

        Returns:
            Composite compliance score (0-100).
        """
        if not dimension_results:
            return 0.0

        total = sum(dr.weighted_score for dr in dimension_results)
        return _clamp(total)

    def classify_compliance_status(
        self,
        score: float,
    ) -> ComplianceStatus:
        """Classify a compliance score into a compliance status.

        Uses fixed thresholds from PRD Appendix C:
            - >= 80: COMPLIANT
            - >= 60: CONDITIONALLY_COMPLIANT
            - < 60: NON_COMPLIANT

        Args:
            score: Composite compliance score (0-100).

        Returns:
            ComplianceStatus enum value.
        """
        if score >= 80.0:
            return ComplianceStatus.COMPLIANT
        elif score >= 60.0:
            return ComplianceStatus.CONDITIONALLY_COMPLIANT
        else:
            return ComplianceStatus.NON_COMPLIANT

    # ------------------------------------------------------------------
    # Public API: Alert Generation
    # ------------------------------------------------------------------

    def generate_alerts(
        self,
        supplier_id: str,
        old_status: str,
        new_status: str,
    ) -> List[ComplianceAlert]:
        """Generate alerts for compliance status changes.

        Creates alerts when a supplier's compliance status transitions,
        with severity based on the direction and magnitude of the change.

        Args:
            supplier_id: Supplier whose status changed.
            old_status: Previous compliance status value.
            new_status: New compliance status value.

        Returns:
            List of ComplianceAlert objects.
        """
        alerts: List[ComplianceAlert] = []

        if old_status == new_status:
            return alerts

        # Determine severity based on transition direction
        severity = self._determine_alert_severity(old_status, new_status)
        direction = self._determine_transition_direction(
            old_status, new_status
        )

        message = (
            f"Supplier {supplier_id} compliance status changed from "
            f"{old_status} to {new_status} ({direction})"
        )

        provenance_data = {
            "supplier_id": supplier_id,
            "old_status": old_status,
            "new_status": new_status,
            "severity": severity.value,
        }
        provenance_hash = _compute_hash(provenance_data)

        alert = ComplianceAlert(
            alert_id=str(uuid.uuid4()),
            supplier_id=supplier_id,
            alert_type="status_change",
            severity=severity.value,
            previous_status=old_status,
            current_status=new_status,
            message=message,
            dimension="composite",
            triggered_at=_utcnow().isoformat(),
            provenance_hash=provenance_hash,
        )
        alerts.append(alert)
        self._alert_count += 1

        logger.info(
            "Compliance alert generated: supplier=%s type=status_change "
            "severity=%s transition=%s->%s",
            supplier_id,
            severity.value,
            old_status,
            new_status,
        )

        return alerts

    def get_expiry_warnings(
        self,
        supplier: SupplierComplianceProfile,
    ) -> List[ExpiryWarning]:
        """Generate expiry warnings for DDS and certifications.

        Checks all DDS references and certifications against warning
        thresholds (90, 60, 30, 14, 7 days before expiry).

        Args:
            supplier: Supplier profile with DDS and certification data.

        Returns:
            List of ExpiryWarning objects sorted by urgency.
        """
        warnings: List[ExpiryWarning] = []
        now = _utcnow()

        # Check DDS expiry
        for dds in supplier.dds_references:
            dds_id = dds.get("dds_id", "unknown")
            valid_until_str = str(dds.get("valid_until", ""))
            valid_until = _parse_datetime(valid_until_str)

            if valid_until is None:
                continue

            days_remaining = (valid_until - now).days
            if days_remaining < 0:
                severity = AlertSeverity.CRITICAL.value
                action = (
                    f"DDS {dds_id} has expired. Submit a new DDS "
                    "immediately to maintain EUDR compliance."
                )
            elif days_remaining <= 7:
                severity = AlertSeverity.CRITICAL.value
                action = (
                    f"DDS {dds_id} expires in {days_remaining} days. "
                    "Urgent: initiate DDS renewal process immediately."
                )
            elif days_remaining <= 14:
                severity = AlertSeverity.HIGH.value
                action = (
                    f"DDS {dds_id} expires in {days_remaining} days. "
                    "Begin DDS renewal process."
                )
            elif days_remaining <= 30:
                severity = AlertSeverity.HIGH.value
                action = (
                    f"DDS {dds_id} expires in {days_remaining} days. "
                    "Schedule DDS renewal."
                )
            elif days_remaining <= 60:
                severity = AlertSeverity.MEDIUM.value
                action = (
                    f"DDS {dds_id} expires in {days_remaining} days. "
                    "Plan for DDS renewal."
                )
            elif days_remaining <= 90:
                severity = AlertSeverity.LOW.value
                action = (
                    f"DDS {dds_id} expires in {days_remaining} days. "
                    "Monitor and prepare for renewal."
                )
            else:
                continue

            warnings.append(
                ExpiryWarning(
                    warning_id=str(uuid.uuid4()),
                    supplier_id=supplier.supplier_id,
                    item_type="dds",
                    item_id=dds_id,
                    item_name=f"DDS Reference {dds_id}",
                    expires_at=valid_until.isoformat(),
                    days_until_expiry=days_remaining,
                    severity=severity,
                    action_required=action,
                )
            )

        # Check certification expiry
        for cert in supplier.certifications:
            cert_type = str(cert.get("type", "unknown"))
            cert_id = str(cert.get("certificate_id", "unknown"))
            valid_until_str = str(cert.get("valid_until", ""))
            valid_until = _parse_datetime(valid_until_str)

            if valid_until is None:
                continue

            days_remaining = (valid_until - now).days
            if days_remaining < 0:
                severity = AlertSeverity.HIGH.value
                action = (
                    f"Certification {cert_type} ({cert_id}) has expired. "
                    "Renew certification to maintain compliance."
                )
            elif days_remaining <= 14:
                severity = AlertSeverity.HIGH.value
                action = (
                    f"Certification {cert_type} ({cert_id}) expires in "
                    f"{days_remaining} days. Urgent renewal required."
                )
            elif days_remaining <= 30:
                severity = AlertSeverity.MEDIUM.value
                action = (
                    f"Certification {cert_type} ({cert_id}) expires in "
                    f"{days_remaining} days. Initiate renewal."
                )
            elif days_remaining <= 60:
                severity = AlertSeverity.MEDIUM.value
                action = (
                    f"Certification {cert_type} ({cert_id}) expires in "
                    f"{days_remaining} days. Plan for renewal."
                )
            elif days_remaining <= 90:
                severity = AlertSeverity.LOW.value
                action = (
                    f"Certification {cert_type} ({cert_id}) expires in "
                    f"{days_remaining} days. Monitor and prepare."
                )
            else:
                continue

            warnings.append(
                ExpiryWarning(
                    warning_id=str(uuid.uuid4()),
                    supplier_id=supplier.supplier_id,
                    item_type="certification",
                    item_id=cert_id,
                    item_name=f"{cert_type} Certificate {cert_id}",
                    expires_at=valid_until.isoformat(),
                    days_until_expiry=days_remaining,
                    severity=severity,
                    action_required=action,
                )
            )

        # Sort by days_until_expiry ascending (most urgent first)
        warnings.sort(key=lambda w: w.days_until_expiry)

        logger.debug(
            "Expiry warnings for supplier=%s: %d warnings generated",
            supplier.supplier_id,
            len(warnings),
        )

        return warnings

    # ------------------------------------------------------------------
    # Public API: Compliance History
    # ------------------------------------------------------------------

    def get_compliance_history(
        self,
        supplier_id: str,
    ) -> List[ComplianceHistoryEntry]:
        """Get compliance check history timeline for a supplier.

        Returns chronological entries showing how compliance has changed.

        Args:
            supplier_id: Supplier to retrieve history for.

        Returns:
            List of ComplianceHistoryEntry in chronological order.
        """
        history = self._compliance_history.get(supplier_id, [])
        entries: List[ComplianceHistoryEntry] = []

        for i, check in enumerate(history):
            dimension_scores = {
                dr.dimension: dr.score for dr in check.dimension_results
            }

            change_reason = ""
            if i > 0:
                prev = history[i - 1]
                if check.status != prev.status:
                    change_reason = (
                        f"Status changed from {prev.status} to "
                        f"{check.status}"
                    )
                else:
                    score_delta = check.composite_score - prev.composite_score
                    if abs(score_delta) >= 1.0:
                        direction = (
                            "improved" if score_delta > 0 else "degraded"
                        )
                        change_reason = (
                            f"Score {direction} by "
                            f"{abs(score_delta):.1f} points"
                        )
                    else:
                        change_reason = "No significant change"

            entries.append(
                ComplianceHistoryEntry(
                    timestamp=check.checked_at,
                    composite_score=check.composite_score,
                    status=check.status,
                    dimension_scores=dimension_scores,
                    change_reason=change_reason,
                )
            )

        return entries

    def get_compliance_trend(
        self,
        supplier_id: str,
    ) -> TrendDirection:
        """Determine compliance trend direction for a supplier.

        Args:
            supplier_id: Supplier to analyze.

        Returns:
            TrendDirection enum value.
        """
        history = self._compliance_history.get(supplier_id, [])
        if len(history) < 2:
            return TrendDirection.STABLE

        recent = history[-3:] if len(history) >= 3 else history
        scores = [r.composite_score for r in recent]

        delta = scores[-1] - scores[0]
        if delta > 5.0:
            return TrendDirection.IMPROVING
        elif delta < -5.0:
            return TrendDirection.DEGRADING
        return TrendDirection.STABLE

    # ------------------------------------------------------------------
    # Public API: Batch Compliance Checking
    # ------------------------------------------------------------------

    def batch_check(
        self,
        suppliers: List[SupplierComplianceProfile],
        batch_size: int = 1000,
    ) -> BatchComplianceResult:
        """Perform batch compliance checking for multiple suppliers.

        Args:
            suppliers: List of supplier profiles to check.
            batch_size: Chunk size for memory-efficient processing.

        Returns:
            BatchComplianceResult with all results and summary.
        """
        t_start = time.monotonic()
        batch_id = str(uuid.uuid4())
        results: List[ComplianceCheckResult] = []
        failed_count = 0
        status_counts: Dict[str, int] = {}
        score_sum = 0.0

        logger.info(
            "Starting batch compliance check: batch=%s suppliers=%d",
            batch_id,
            len(suppliers),
        )

        for i in range(0, len(suppliers), batch_size):
            chunk = suppliers[i : i + batch_size]
            for profile in chunk:
                try:
                    result = self.check_compliance(
                        profile.supplier_id, profile
                    )
                    results.append(result)
                    score_sum += result.composite_score
                    status_counts[result.status] = (
                        status_counts.get(result.status, 0) + 1
                    )
                except Exception as exc:
                    failed_count += 1
                    logger.warning(
                        "Batch compliance check failed for supplier=%s: %s",
                        profile.supplier_id,
                        str(exc),
                    )

        elapsed_ms = (time.monotonic() - t_start) * 1000.0
        successful = len(results)
        avg_score = (
            round(score_sum / successful, 2) if successful > 0 else 0.0
        )

        # Count DDS-eligible suppliers
        dds_eligible = sum(1 for r in results if r.dds_can_include)

        summary: Dict[str, Any] = {
            "average_compliance_score": avg_score,
            "status_distribution": status_counts,
            "dds_eligible_count": dds_eligible,
            "dds_eligible_pct": (
                round(dds_eligible / successful * 100.0, 2)
                if successful > 0
                else 0.0
            ),
            "total_issues": sum(len(r.issues) for r in results),
        }

        provenance_data = {
            "batch_id": batch_id,
            "total_suppliers": len(suppliers),
            "successful": successful,
            "failed": failed_count,
            "avg_score": avg_score,
        }
        provenance_hash = _compute_hash(provenance_data)

        logger.info(
            "Batch compliance check completed: batch=%s total=%d "
            "success=%d failed=%d avg_score=%.1f dds_eligible=%d "
            "time=%.2fms",
            batch_id,
            len(suppliers),
            successful,
            failed_count,
            avg_score,
            dds_eligible,
            elapsed_ms,
        )

        return BatchComplianceResult(
            batch_id=batch_id,
            total_suppliers=len(suppliers),
            successful=successful,
            failed=failed_count,
            results=results,
            summary=summary,
            processing_time_ms=round(elapsed_ms, 3),
            provenance_hash=provenance_hash,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def check_count(self) -> int:
        """Return total number of compliance checks performed."""
        return self._check_count

    @property
    def alert_count(self) -> int:
        """Return total number of alerts generated."""
        return self._alert_count

    @property
    def tracked_supplier_count(self) -> int:
        """Return number of unique suppliers with compliance history."""
        return len(self._compliance_history)

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _has_expired_items(
        self,
        profile: SupplierComplianceProfile,
    ) -> bool:
        """Check if a supplier has any expired DDS or certifications.

        Args:
            profile: Supplier profile to check.

        Returns:
            True if any item is expired and no valid items remain.
        """
        now = _utcnow()
        has_valid_dds = False
        has_expired_dds = False

        for dds in profile.dds_references:
            valid_until = _parse_datetime(str(dds.get("valid_until", "")))
            if valid_until is not None:
                if valid_until < now:
                    has_expired_dds = True
                else:
                    has_valid_dds = True

        # Only mark as expired if there are expired items and no valid ones
        if has_expired_dds and not has_valid_dds and profile.dds_references:
            return True

        has_valid_cert = False
        has_expired_cert = False
        for cert in profile.certifications:
            valid_until = _parse_datetime(str(cert.get("valid_until", "")))
            if valid_until is not None:
                if valid_until < now:
                    has_expired_cert = True
                else:
                    has_valid_cert = True

        if (
            has_expired_cert
            and not has_valid_cert
            and profile.certifications
            and not profile.dds_references
        ):
            return True

        return False

    def _has_sufficient_data(
        self,
        profile: SupplierComplianceProfile,
    ) -> bool:
        """Check if a supplier profile has minimum required data.

        A supplier is considered to have sufficient data if it has at
        least a supplier_id and at least one of: DDS references,
        certifications, GPS coordinates, or deforestation status.

        Args:
            profile: Supplier profile to check.

        Returns:
            True if sufficient data exists for compliance assessment.
        """
        if not profile.supplier_id:
            return False

        has_dds = len(profile.dds_references) > 0
        has_certs = len(profile.certifications) > 0
        has_gps = (
            profile.gps_latitude is not None
            and profile.gps_longitude is not None
        )
        has_deforestation = bool(
            profile.deforestation_free_status
            and profile.deforestation_free_status.strip()
        )
        has_plots = profile.total_plots > 0

        return has_dds or has_certs or has_gps or has_deforestation or has_plots

    def _determine_alert_severity(
        self,
        old_status: str,
        new_status: str,
    ) -> AlertSeverity:
        """Determine alert severity based on status transition.

        Transitions to non_compliant or expired are critical.
        Transitions from compliant to conditionally_compliant are medium.
        Improvements are low/info severity.

        Args:
            old_status: Previous status value.
            new_status: New status value.

        Returns:
            AlertSeverity for the transition.
        """
        status_order = {
            ComplianceStatus.COMPLIANT.value: 4,
            ComplianceStatus.CONDITIONALLY_COMPLIANT.value: 3,
            ComplianceStatus.UNVERIFIED.value: 2,
            ComplianceStatus.EXPIRED.value: 1,
            ComplianceStatus.NON_COMPLIANT.value: 0,
        }

        old_rank = status_order.get(old_status, 2)
        new_rank = status_order.get(new_status, 2)

        if new_status == ComplianceStatus.NON_COMPLIANT.value:
            return AlertSeverity.CRITICAL
        elif new_status == ComplianceStatus.EXPIRED.value:
            return AlertSeverity.HIGH
        elif new_rank < old_rank:
            return AlertSeverity.MEDIUM
        elif new_rank > old_rank:
            return AlertSeverity.LOW
        return AlertSeverity.INFO

    def _determine_transition_direction(
        self,
        old_status: str,
        new_status: str,
    ) -> str:
        """Determine whether a status transition is an improvement or degradation.

        Args:
            old_status: Previous status value.
            new_status: New status value.

        Returns:
            "improvement", "degradation", or "lateral".
        """
        status_order = {
            ComplianceStatus.COMPLIANT.value: 4,
            ComplianceStatus.CONDITIONALLY_COMPLIANT.value: 3,
            ComplianceStatus.UNVERIFIED.value: 2,
            ComplianceStatus.EXPIRED.value: 1,
            ComplianceStatus.NON_COMPLIANT.value: 0,
        }

        old_rank = status_order.get(old_status, 2)
        new_rank = status_order.get(new_status, 2)

        if new_rank > old_rank:
            return "improvement"
        elif new_rank < old_rank:
            return "degradation"
        return "lateral"

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a developer-friendly string representation."""
        return (
            f"ComplianceMonitor("
            f"checks={self._check_count}, "
            f"alerts={self._alert_count}, "
            f"tracked_suppliers={len(self._compliance_history)}, "
            f"version={_MODULE_VERSION!r})"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Enumerations
    "ComplianceStatus",
    "ComplianceDimension",
    "AlertSeverity",
    "TrendDirection",
    # Constants
    "COMPLIANCE_THRESHOLDS",
    "EXPIRY_WARNING_DAYS",
    "COMPLIANCE_DIMENSION_WEIGHTS",
    "RECOGNIZED_CERTIFICATIONS",
    # Data classes
    "SupplierComplianceProfile",
    "DimensionCheckResult",
    "ComplianceCheckResult",
    "ComplianceAlert",
    "ExpiryWarning",
    "ComplianceHistoryEntry",
    "BatchComplianceResult",
    # Engine
    "ComplianceMonitor",
]
