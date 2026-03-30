# -*- coding: utf-8 -*-
"""
Labeling Verification Engine - AGENT-EUDR-010: Segregation Verifier (Feature 6)

Verifies physical labeling and marking compliance for EUDR segregation
control points. Manages label registration, content verification, placement
validation, condition assessment, color-code compliance, expiry tracking,
and comprehensive labeling audits across 8 label types.

Zero-Hallucination Guarantees:
    - All label verification is deterministic field-by-field comparison
    - Color-code validation uses static mapping dictionary (no ML/LLM)
    - Score calculation uses weighted arithmetic formulas
    - Content field validation uses required-field lists per label type
    - Expiry detection uses pure datetime comparison
    - SHA-256 provenance hashes on all verification and audit results
    - No ML/LLM used for any labeling verification logic

Performance Targets:
    - Single label verification: <5ms
    - Facility labeling audit (500 labels): <500ms
    - Missing label detection: <100ms

Regulatory References:
    - EUDR Article 10(2)(f): Physical segregation marking requirements
    - EUDR Article 14: Competent authority inspection readiness
    - ISO 22095:2020: Chain of Custody - Visual identification requirements
    - EUDR Article 31: Five-year record retention for label audit trails

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-010 (Feature 6: Labeling and Marking Verification)
Agent ID: GL-EUDR-SGV-010
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module version for provenance tracking
# ---------------------------------------------------------------------------

_MODULE_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash for audit provenance."""
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _generate_id(prefix: str = "lbl") -> str:
    """Generate a unique identifier with the given prefix."""
    return f"{prefix}-{uuid.uuid4().hex[:16]}"

# ---------------------------------------------------------------------------
# Constants: Label Types and Requirements
# ---------------------------------------------------------------------------

#: All recognized label types.
LABEL_TYPES: List[str] = [
    "compliance_tag",
    "zone_sign",
    "vehicle_placard",
    "container_seal_label",
    "batch_sticker",
    "pallet_marker",
    "silo_sign",
    "processing_line_marker",
]

#: Valid label statuses.
LABEL_STATUSES: List[str] = [
    "applied",
    "readable",
    "damaged",
    "missing",
    "expired",
]

#: Valid label event types.
LABEL_EVENT_TYPES: List[str] = [
    "applied",
    "verified",
    "damaged",
    "replaced",
    "removed",
]

#: Required content fields per label type (deterministic templates).
REQUIRED_LABEL_CONTENT: Dict[str, List[str]] = {
    "compliance_tag": [
        "compliance_status",
        "batch_id",
        "commodity",
        "date_applied",
        "operator_id",
        "origin_country",
    ],
    "zone_sign": [
        "zone_id",
        "compliance_status",
        "zone_color_code",
        "commodities_allowed",
        "date_posted",
        "responsible_person",
    ],
    "vehicle_placard": [
        "vehicle_id",
        "compliance_status",
        "cargo_type",
        "cleaning_status",
        "date_applied",
        "driver_id",
    ],
    "container_seal_label": [
        "container_id",
        "seal_number",
        "compliance_status",
        "batch_id",
        "date_sealed",
        "origin_facility",
    ],
    "batch_sticker": [
        "batch_id",
        "compliance_status",
        "commodity",
        "quantity_kg",
        "date_applied",
        "operator_id",
    ],
    "pallet_marker": [
        "pallet_id",
        "compliance_status",
        "batch_ids",
        "color_code",
        "date_applied",
    ],
    "silo_sign": [
        "silo_id",
        "compliance_status",
        "commodity",
        "capacity_mt",
        "date_posted",
        "responsible_person",
    ],
    "processing_line_marker": [
        "line_id",
        "compliance_status",
        "current_run_type",
        "changeover_status",
        "date_updated",
        "operator_id",
    ],
}

#: Color code mapping for segregation zone compliance identification.
COLOR_CODE_MAP: Dict[str, str] = {
    "green": "compliant",
    "red": "non_compliant",
    "yellow": "pending",
    "blue": "buffer",
    "white": "neutral",
}

#: Reverse mapping: compliance status to expected color code.
STATUS_TO_COLOR: Dict[str, str] = {
    v: k for k, v in COLOR_CODE_MAP.items()
}

#: Score weights for composite labeling assessment.
LABEL_SCORE_WEIGHTS: Dict[str, float] = {
    "coverage": 0.30,
    "readability": 0.25,
    "accuracy": 0.25,
    "timeliness": 0.20,
}

#: Default label validity period in days per label type.
DEFAULT_LABEL_VALIDITY_DAYS: Dict[str, int] = {
    "compliance_tag": 90,
    "zone_sign": 365,
    "vehicle_placard": 30,
    "container_seal_label": 180,
    "batch_sticker": 90,
    "pallet_marker": 60,
    "silo_sign": 365,
    "processing_line_marker": 30,
}

# ---------------------------------------------------------------------------
# Internal Dataclass Result Types
# ---------------------------------------------------------------------------

@dataclass
class LabelRecord:
    """A registered label record with full lifecycle tracking.

    Attributes:
        label_id: Unique identifier for this label.
        scp_id: Segregation control point this label is associated with.
        label_type: One of 8 label types.
        status: Current label status (applied/readable/damaged/missing/expired).
        content_fields: Dictionary of label content field values.
        placement_verified: Whether label placement has been verified.
        applied_date: ISO timestamp when label was applied.
        verified_date: ISO timestamp of most recent verification.
        expiry_date: ISO timestamp when label expires.
        metadata: Additional contextual data.
        provenance_hash: SHA-256 hash for audit trail.
    """

    label_id: str
    scp_id: str
    label_type: str
    status: str
    content_fields: Dict[str, Any]
    placement_verified: bool
    applied_date: str
    verified_date: Optional[str]
    expiry_date: Optional[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    provenance_hash: str = ""

@dataclass
class LabelVerificationResult:
    """Result of verifying a single label or SCP's labels.

    Attributes:
        scp_id: Segregation control point identifier.
        labels_checked: Total number of labels verified.
        labels_compliant: Number of labels passing verification.
        labels_non_compliant: Number of labels failing verification.
        coverage_score: Percentage of required labels present (0-100).
        readability_score: Percentage of labels readable (0-100).
        accuracy_score: Percentage of labels with correct content (0-100).
        timeliness_score: Percentage of labels within validity (0-100).
        overall_score: Weighted composite score (0-100).
        findings: List of finding dictionaries.
        provenance_hash: SHA-256 hash for audit trail.
    """

    scp_id: str
    labels_checked: int
    labels_compliant: int
    labels_non_compliant: int
    coverage_score: float
    readability_score: float
    accuracy_score: float
    timeliness_score: float
    overall_score: float
    findings: List[Dict[str, Any]]
    provenance_hash: str = ""

@dataclass
class LabelAuditResult:
    """Result of a comprehensive facility labeling audit.

    Attributes:
        facility_id: Facility identifier.
        total_scps: Total SCPs in the facility.
        scps_with_labels: Number of SCPs with valid labels.
        scps_without_labels: Number of SCPs missing labels.
        overall_compliance_score: Weighted composite score (0-100).
        label_type_breakdown: Score breakdown by label type.
        color_code_compliance: Color code verification results.
        recommendations: Prioritized improvement recommendations.
        provenance_hash: SHA-256 hash for audit trail.
    """

    facility_id: str
    total_scps: int
    scps_with_labels: int
    scps_without_labels: int
    overall_compliance_score: float
    label_type_breakdown: Dict[str, float]
    color_code_compliance: Dict[str, Any]
    recommendations: List[str]
    provenance_hash: str = ""

@dataclass
class LabelEvent:
    """A label lifecycle event record.

    Attributes:
        event_id: Unique event identifier.
        label_id: Label this event relates to.
        event_type: Event type (applied/verified/damaged/replaced/removed).
        timestamp: ISO timestamp of the event.
        operator_id: Identifier of the operator who performed the action.
        notes: Free-text notes about the event.
        provenance_hash: SHA-256 hash for audit trail.
    """

    event_id: str
    label_id: str
    event_type: str
    timestamp: str
    operator_id: str
    notes: str
    provenance_hash: str = ""

# ---------------------------------------------------------------------------
# LabelingVerificationEngine
# ---------------------------------------------------------------------------

class LabelingVerificationEngine:
    """Verifies labeling and marking compliance for EUDR segregation.

    Manages the full label lifecycle: registration, content verification,
    placement validation, condition assessment, expiry tracking, color-code
    compliance, missing label detection, and comprehensive facility audits
    across 8 label types.

    All verification logic is deterministic field-by-field comparison.
    Scores are computed via weighted arithmetic formulas. No ML/LLM is
    used for any verification or scoring logic.

    Attributes:
        _labels: In-memory store of labels keyed by label_id.
        _scp_labels: Mapping of scp_id to list of label_ids.
        _label_events: Mapping of label_id to list of LabelEvent objects.

    Example:
        >>> engine = LabelingVerificationEngine()
        >>> label = engine.register_label(
        ...     scp_id="scp-001",
        ...     label_type="compliance_tag",
        ...     content_fields={"compliance_status": "compliant", ...},
        ...     applied_date="2026-01-15T10:00:00+00:00",
        ... )
        >>> result = engine.verify_label(label.label_id)
        >>> assert result.overall_score >= 0.0
    """

    def __init__(self) -> None:
        """Initialize LabelingVerificationEngine."""
        self._labels: Dict[str, LabelRecord] = {}
        self._scp_labels: Dict[str, List[str]] = {}
        self._label_events: Dict[str, List[LabelEvent]] = {}
        logger.info(
            "LabelingVerificationEngine initialized: "
            "label_types=%d, module_version=%s",
            len(LABEL_TYPES),
            _MODULE_VERSION,
        )

    # ------------------------------------------------------------------
    # Public API: Label Registration
    # ------------------------------------------------------------------

    def register_label(
        self,
        scp_id: str,
        label_type: str,
        content_fields: Dict[str, Any],
        applied_date: Optional[str] = None,
        expiry_date: Optional[str] = None,
    ) -> LabelRecord:
        """Register a new label for a segregation control point.

        Creates a label record with initial status 'applied' and
        generates a provenance hash. If no expiry_date is provided,
        a default validity period is assigned based on label type.

        Args:
            scp_id: Segregation control point identifier.
            label_type: One of 8 label types.
            content_fields: Dictionary of label content field values.
            applied_date: Optional ISO timestamp when label was applied.
                Defaults to current UTC time.
            expiry_date: Optional ISO timestamp when label expires.
                Defaults based on label type validity period.

        Returns:
            LabelRecord with unique label_id and provenance hash.

        Raises:
            ValueError: If scp_id or label_type is invalid.
        """
        if not scp_id:
            raise ValueError("scp_id must not be empty")
        if label_type not in LABEL_TYPES:
            raise ValueError(
                f"label_type must be one of {LABEL_TYPES}, "
                f"got '{label_type}'"
            )

        label_id = _generate_id("lbl")
        now = utcnow()

        if applied_date is None:
            applied_date = now.isoformat()

        if expiry_date is None:
            validity_days = DEFAULT_LABEL_VALIDITY_DAYS.get(label_type, 90)
            try:
                applied_dt = datetime.fromisoformat(applied_date)
                if applied_dt.tzinfo is None:
                    applied_dt = applied_dt.replace(tzinfo=timezone.utc)
                expiry_dt = applied_dt + timedelta(days=validity_days)
                expiry_date = expiry_dt.isoformat()
            except (ValueError, TypeError):
                expiry_date = (
                    now + timedelta(days=validity_days)
                ).isoformat()

        label = LabelRecord(
            label_id=label_id,
            scp_id=scp_id,
            label_type=label_type,
            status="applied",
            content_fields=dict(content_fields),
            placement_verified=False,
            applied_date=applied_date,
            verified_date=None,
            expiry_date=expiry_date,
            metadata={
                "module_version": _MODULE_VERSION,
                "registered_at": now.isoformat(),
            },
        )
        label.provenance_hash = _compute_hash({
            "label_id": label_id,
            "scp_id": scp_id,
            "label_type": label_type,
            "content_fields": content_fields,
            "applied_date": applied_date,
            "expiry_date": expiry_date,
        })

        # Store label
        self._labels[label_id] = label
        if scp_id not in self._scp_labels:
            self._scp_labels[scp_id] = []
        self._scp_labels[scp_id].append(label_id)
        self._label_events[label_id] = []

        # Record initial event
        self._record_event_internal(
            label_id, "applied", "system", "Label registered and applied",
        )

        logger.info(
            "Label registered: label_id=%s, scp_id=%s, type=%s, "
            "expiry=%s",
            label_id,
            scp_id,
            label_type,
            expiry_date,
        )
        return label

    def get_label(self, label_id: str) -> Optional[LabelRecord]:
        """Retrieve a label record by label_id.

        Args:
            label_id: Unique label identifier.

        Returns:
            LabelRecord if found, None otherwise.
        """
        return self._labels.get(label_id)

    def get_scp_labels(self, scp_id: str) -> List[LabelRecord]:
        """Retrieve all labels for a segregation control point.

        Args:
            scp_id: Segregation control point identifier.

        Returns:
            List of LabelRecord objects for the given SCP.
        """
        label_ids = self._scp_labels.get(scp_id, [])
        return [
            self._labels[lid] for lid in label_ids
            if lid in self._labels
        ]

    # ------------------------------------------------------------------
    # Public API: Label Verification
    # ------------------------------------------------------------------

    def verify_label(
        self,
        label_id: str,
    ) -> LabelVerificationResult:
        """Verify a single label against all compliance requirements.

        Checks content completeness, placement verification, label
        condition, and validity period. Produces a composite score.

        Args:
            label_id: Unique label identifier.

        Returns:
            LabelVerificationResult with component and overall scores.

        Raises:
            ValueError: If label_id is not found.
        """
        if label_id not in self._labels:
            raise ValueError(f"Label not found: {label_id}")

        label = self._labels[label_id]
        findings: List[Dict[str, Any]] = []
        now = utcnow()

        # Coverage: is the label present?
        coverage_score = 100.0 if label.status != "missing" else 0.0
        if label.status == "missing":
            findings.append({
                "type": "coverage",
                "severity": "critical",
                "message": f"Label {label_id} is missing from SCP {label.scp_id}",
            })

        # Readability: is the label readable?
        readability_score = self._assess_readability(label, findings)

        # Accuracy: are content fields correct?
        accuracy_score = self._assess_content_accuracy(label, findings)

        # Timeliness: is the label within validity?
        timeliness_score = self._assess_timeliness(label, now, findings)

        # Compute weighted overall score
        overall_score = (
            coverage_score * LABEL_SCORE_WEIGHTS["coverage"]
            + readability_score * LABEL_SCORE_WEIGHTS["readability"]
            + accuracy_score * LABEL_SCORE_WEIGHTS["accuracy"]
            + timeliness_score * LABEL_SCORE_WEIGHTS["timeliness"]
        )

        # Update label verification date
        label.verified_date = now.isoformat()

        # Determine compliance
        is_compliant = overall_score >= 70.0 and label.status not in (
            "missing", "expired",
        )

        result = LabelVerificationResult(
            scp_id=label.scp_id,
            labels_checked=1,
            labels_compliant=1 if is_compliant else 0,
            labels_non_compliant=0 if is_compliant else 1,
            coverage_score=round(coverage_score, 2),
            readability_score=round(readability_score, 2),
            accuracy_score=round(accuracy_score, 2),
            timeliness_score=round(timeliness_score, 2),
            overall_score=round(overall_score, 2),
            findings=findings,
        )
        result.provenance_hash = _compute_hash({
            "label_id": label_id,
            "scp_id": label.scp_id,
            "overall_score": result.overall_score,
            "findings_count": len(findings),
            "verified_at": label.verified_date,
            "module_version": _MODULE_VERSION,
        })

        # Record verification event
        self._record_event_internal(
            label_id, "verified", "system",
            f"Label verified: score={result.overall_score:.1f}",
        )

        logger.info(
            "Label verified: label_id=%s, score=%.1f, compliant=%s",
            label_id,
            result.overall_score,
            is_compliant,
        )
        return result

    def verify_label_content(
        self,
        label_id: str,
        required_fields: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Check all required content fields are present on a label.

        Args:
            label_id: Unique label identifier.
            required_fields: Optional override list of required fields.
                If None, uses the default required fields for the label type.

        Returns:
            Dictionary with keys: label_id, label_type, fields_present,
            fields_missing, completeness_score, details.

        Raises:
            ValueError: If label_id is not found.
        """
        if label_id not in self._labels:
            raise ValueError(f"Label not found: {label_id}")

        label = self._labels[label_id]

        if required_fields is None:
            required_fields = REQUIRED_LABEL_CONTENT.get(
                label.label_type, [],
            )

        fields_present: List[str] = []
        fields_missing: List[str] = []
        details: List[Dict[str, Any]] = []

        for req_field in required_fields:
            value = label.content_fields.get(req_field)
            if value is not None and str(value).strip():
                fields_present.append(req_field)
                details.append({
                    "field": req_field,
                    "present": True,
                    "value_provided": True,
                })
            else:
                fields_missing.append(req_field)
                details.append({
                    "field": req_field,
                    "present": False,
                    "value_provided": False,
                })

        total = len(required_fields)
        completeness = (
            (len(fields_present) / total * 100.0) if total > 0 else 100.0
        )

        return {
            "label_id": label_id,
            "label_type": label.label_type,
            "fields_present": fields_present,
            "fields_missing": fields_missing,
            "completeness_score": round(completeness, 2),
            "total_required": total,
            "details": details,
        }

    def verify_label_placement(
        self,
        label_id: str,
        scp_id: str,
    ) -> Dict[str, Any]:
        """Verify correct label is on correct segregation control point.

        Args:
            label_id: Unique label identifier.
            scp_id: Expected SCP identifier.

        Returns:
            Dictionary with keys: label_id, scp_id, placement_correct,
            label_scp_match, findings.

        Raises:
            ValueError: If label_id is not found.
        """
        if label_id not in self._labels:
            raise ValueError(f"Label not found: {label_id}")

        label = self._labels[label_id]
        findings: List[Dict[str, Any]] = []

        placement_correct = label.scp_id == scp_id
        if not placement_correct:
            findings.append({
                "type": "placement",
                "severity": "major",
                "message": (
                    f"Label {label_id} is registered for SCP "
                    f"{label.scp_id} but found on SCP {scp_id}"
                ),
            })

        # Update placement verification status
        if placement_correct:
            label.placement_verified = True

        return {
            "label_id": label_id,
            "scp_id": scp_id,
            "placement_correct": placement_correct,
            "label_scp_match": label.scp_id == scp_id,
            "label_type": label.label_type,
            "findings": findings,
        }

    def check_label_condition(
        self,
        label_id: str,
    ) -> Dict[str, Any]:
        """Assess the physical condition of a label.

        Args:
            label_id: Unique label identifier.

        Returns:
            Dictionary with keys: label_id, status, is_readable,
            is_damaged, is_missing, is_expired, condition_score.

        Raises:
            ValueError: If label_id is not found.
        """
        if label_id not in self._labels:
            raise ValueError(f"Label not found: {label_id}")

        label = self._labels[label_id]
        now = utcnow()

        # Check expiry
        is_expired = False
        if label.expiry_date:
            try:
                expiry_dt = datetime.fromisoformat(label.expiry_date)
                if expiry_dt.tzinfo is None:
                    expiry_dt = expiry_dt.replace(tzinfo=timezone.utc)
                is_expired = now > expiry_dt
            except (ValueError, TypeError):
                is_expired = False

        if is_expired and label.status != "expired":
            label.status = "expired"

        is_readable = label.status in ("applied", "readable")
        is_damaged = label.status == "damaged"
        is_missing = label.status == "missing"

        # Condition score
        condition_scores = {
            "applied": 100.0,
            "readable": 80.0,
            "damaged": 30.0,
            "expired": 20.0,
            "missing": 0.0,
        }
        condition_score = condition_scores.get(label.status, 0.0)

        return {
            "label_id": label_id,
            "status": label.status,
            "is_readable": is_readable,
            "is_damaged": is_damaged,
            "is_missing": is_missing,
            "is_expired": is_expired,
            "condition_score": condition_score,
        }

    # ------------------------------------------------------------------
    # Public API: Missing Labels and Color Codes
    # ------------------------------------------------------------------

    def detect_missing_labels(
        self,
        facility_id: str,
        scp_list: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Detect SCPs without valid labels in a facility.

        Args:
            facility_id: Facility identifier.
            scp_list: List of SCP dicts with keys: scp_id,
                scp_type, required_label_types.

        Returns:
            List of missing label finding dicts with keys:
            scp_id, scp_type, missing_label_types, finding_type.
        """
        missing_findings: List[Dict[str, Any]] = []

        for scp in scp_list:
            scp_id = scp.get("scp_id", "")
            scp_type = scp.get("scp_type", "unknown")
            required_types = scp.get("required_label_types", [])

            if not required_types:
                # Determine required label types from SCP type
                required_types = self._get_required_label_types(scp_type)

            # Get existing labels for this SCP
            existing_labels = self.get_scp_labels(scp_id)
            existing_types = set(
                lbl.label_type for lbl in existing_labels
                if lbl.status not in ("missing", "expired")
            )

            missing_types = [
                lt for lt in required_types
                if lt not in existing_types
            ]

            if missing_types:
                missing_findings.append({
                    "scp_id": scp_id,
                    "scp_type": scp_type,
                    "facility_id": facility_id,
                    "missing_label_types": missing_types,
                    "existing_label_types": list(existing_types),
                    "finding_type": "missing_labels",
                    "severity": (
                        "critical" if len(missing_types) >= 3
                        else "major" if len(missing_types) >= 2
                        else "minor"
                    ),
                })

        logger.info(
            "Missing label detection: facility=%s, scps=%d, "
            "scps_with_gaps=%d",
            facility_id,
            len(scp_list),
            len(missing_findings),
        )
        return missing_findings

    def validate_color_code(
        self,
        facility_id: str,
        zones: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Validate color coding consistency across facility zones.

        Checks that zone color codes match the expected compliance
        status mapping defined in COLOR_CODE_MAP.

        Args:
            facility_id: Facility identifier.
            zones: List of zone dicts with keys: zone_id,
                color_code, compliance_status.

        Returns:
            Dictionary with keys: facility_id, zones_checked,
            zones_compliant, zones_non_compliant, findings,
            compliance_score.
        """
        findings: List[Dict[str, Any]] = []
        compliant_count = 0
        total_checked = 0

        for zone in zones:
            zone_id = zone.get("zone_id", "unknown")
            color = str(zone.get("color_code", "")).lower().strip()
            status = str(zone.get("compliance_status", "")).lower().strip()
            total_checked += 1

            if not color:
                findings.append({
                    "zone_id": zone_id,
                    "type": "missing_color",
                    "severity": "major",
                    "message": f"Zone {zone_id} has no color code assigned",
                })
                continue

            expected_status = COLOR_CODE_MAP.get(color)
            if expected_status is None:
                findings.append({
                    "zone_id": zone_id,
                    "type": "invalid_color",
                    "severity": "minor",
                    "message": (
                        f"Zone {zone_id} uses unrecognized color code "
                        f"'{color}' (valid: {list(COLOR_CODE_MAP.keys())})"
                    ),
                })
                continue

            if expected_status == status:
                compliant_count += 1
            else:
                expected_color = STATUS_TO_COLOR.get(status, "unknown")
                findings.append({
                    "zone_id": zone_id,
                    "type": "color_mismatch",
                    "severity": "major",
                    "message": (
                        f"Zone {zone_id} color '{color}' indicates "
                        f"'{expected_status}' but actual status is "
                        f"'{status}' (expected color: '{expected_color}')"
                    ),
                })

        score = (
            (compliant_count / total_checked * 100.0)
            if total_checked > 0 else 100.0
        )

        return {
            "facility_id": facility_id,
            "zones_checked": total_checked,
            "zones_compliant": compliant_count,
            "zones_non_compliant": total_checked - compliant_count,
            "findings": findings,
            "compliance_score": round(score, 2),
        }

    # ------------------------------------------------------------------
    # Public API: Label Events
    # ------------------------------------------------------------------

    def record_label_event(
        self,
        label_id: str,
        event_type: str,
        operator_id: str,
        notes: str = "",
    ) -> LabelEvent:
        """Record a lifecycle event for a label.

        Args:
            label_id: Label identifier.
            event_type: Event type (applied/verified/damaged/replaced/removed).
            operator_id: Identifier of the operator performing the action.
            notes: Optional notes about the event.

        Returns:
            LabelEvent with provenance hash.

        Raises:
            ValueError: If label_id is not found or event_type is invalid.
        """
        if label_id not in self._labels:
            raise ValueError(f"Label not found: {label_id}")
        if event_type not in LABEL_EVENT_TYPES:
            raise ValueError(
                f"event_type must be one of {LABEL_EVENT_TYPES}, "
                f"got '{event_type}'"
            )

        event = self._record_event_internal(
            label_id, event_type, operator_id, notes,
        )

        # Update label status based on event
        label = self._labels[label_id]
        status_map = {
            "damaged": "damaged",
            "removed": "missing",
            "replaced": "applied",
        }
        if event_type in status_map:
            label.status = status_map[event_type]

        if event_type == "verified":
            label.verified_date = event.timestamp

        logger.info(
            "Label event recorded: label_id=%s, event=%s, operator=%s",
            label_id,
            event_type,
            operator_id,
        )
        return event

    # ------------------------------------------------------------------
    # Public API: Facility Audit
    # ------------------------------------------------------------------

    def audit_labeling_compliance(
        self,
        facility_id: str,
    ) -> LabelAuditResult:
        """Conduct a comprehensive labeling audit for a facility.

        Evaluates all labels across all SCPs in the facility,
        computing coverage, readability, accuracy, and timeliness
        scores with label type breakdown.

        Args:
            facility_id: Facility identifier.

        Returns:
            LabelAuditResult with overall score and recommendations.
        """
        all_labels = list(self._labels.values())
        now = utcnow()

        # Group labels by SCP
        scp_ids = set(lbl.scp_id for lbl in all_labels)
        total_scps = len(scp_ids)

        scps_with_valid_labels = 0
        scps_without_labels = 0
        type_scores: Dict[str, List[float]] = {lt: [] for lt in LABEL_TYPES}
        all_scores: List[float] = []

        # Aggregate color code data
        color_compliance_data: Dict[str, int] = {
            "correct": 0,
            "incorrect": 0,
            "missing": 0,
        }

        for scp_id in scp_ids:
            scp_labels = self.get_scp_labels(scp_id)
            valid_labels = [
                lbl for lbl in scp_labels
                if lbl.status not in ("missing", "expired")
            ]

            if valid_labels:
                scps_with_valid_labels += 1
            else:
                scps_without_labels += 1

            for lbl in scp_labels:
                score = self._compute_single_label_score(lbl, now)
                all_scores.append(score)
                if lbl.label_type in type_scores:
                    type_scores[lbl.label_type].append(score)

                # Check color code if applicable
                color = lbl.content_fields.get("zone_color_code", "")
                status = str(
                    lbl.content_fields.get("compliance_status", "")
                ).lower()
                if color:
                    expected = COLOR_CODE_MAP.get(str(color).lower())
                    if expected == status:
                        color_compliance_data["correct"] += 1
                    else:
                        color_compliance_data["incorrect"] += 1
                else:
                    color_compliance_data["missing"] += 1

        # Overall compliance score
        overall_score = (
            sum(all_scores) / len(all_scores)
            if all_scores else 0.0
        )

        # Type breakdown
        type_breakdown: Dict[str, float] = {}
        for lt, scores in type_scores.items():
            type_breakdown[lt] = (
                round(sum(scores) / len(scores), 2)
                if scores else 0.0
            )

        # Color code compliance
        total_color_checks = sum(color_compliance_data.values())
        color_compliance = {
            "total_checked": total_color_checks,
            "correct": color_compliance_data["correct"],
            "incorrect": color_compliance_data["incorrect"],
            "missing": color_compliance_data["missing"],
            "score": round(
                (color_compliance_data["correct"] / max(total_color_checks, 1))
                * 100.0,
                2,
            ),
        }

        # Generate recommendations
        recommendations = self._generate_audit_recommendations(
            overall_score,
            type_breakdown,
            scps_without_labels,
            total_scps,
            color_compliance,
        )

        result = LabelAuditResult(
            facility_id=facility_id,
            total_scps=total_scps,
            scps_with_labels=scps_with_valid_labels,
            scps_without_labels=scps_without_labels,
            overall_compliance_score=round(overall_score, 2),
            label_type_breakdown=type_breakdown,
            color_code_compliance=color_compliance,
            recommendations=recommendations,
        )
        result.provenance_hash = _compute_hash({
            "facility_id": facility_id,
            "total_scps": total_scps,
            "overall_score": result.overall_compliance_score,
            "scps_with_labels": scps_with_valid_labels,
            "module_version": _MODULE_VERSION,
        })

        logger.info(
            "Labeling audit completed: facility=%s, scps=%d, "
            "overall_score=%.1f, with_labels=%d, without=%d",
            facility_id,
            total_scps,
            result.overall_compliance_score,
            scps_with_valid_labels,
            scps_without_labels,
        )
        return result

    # ------------------------------------------------------------------
    # Public API: Label History and Expiry
    # ------------------------------------------------------------------

    def get_label_history(
        self,
        label_id: str,
    ) -> List[LabelEvent]:
        """Retrieve the full event history for a label.

        Args:
            label_id: Unique label identifier.

        Returns:
            List of LabelEvent objects in chronological order.
        """
        return list(self._label_events.get(label_id, []))

    def get_expiring_labels(
        self,
        days_ahead: int = 30,
    ) -> List[LabelRecord]:
        """Retrieve labels expiring within the specified period.

        Args:
            days_ahead: Number of days to look ahead for expiring labels.

        Returns:
            List of LabelRecord objects expiring within the period.
        """
        now = utcnow()
        cutoff = now + timedelta(days=days_ahead)
        expiring: List[LabelRecord] = []

        for label in self._labels.values():
            if label.status == "expired":
                continue
            if not label.expiry_date:
                continue

            try:
                expiry_dt = datetime.fromisoformat(label.expiry_date)
                if expiry_dt.tzinfo is None:
                    expiry_dt = expiry_dt.replace(tzinfo=timezone.utc)
                if now <= expiry_dt <= cutoff:
                    expiring.append(label)
            except (ValueError, TypeError):
                continue

        expiring.sort(key=lambda l: l.expiry_date or "")

        logger.debug(
            "Expiring labels: %d labels expiring within %d days",
            len(expiring),
            days_ahead,
        )
        return expiring

    def calculate_labeling_score(
        self,
        facility_id: str,
    ) -> float:
        """Calculate composite labeling score for a facility.

        Computes a weighted score across coverage, readability,
        accuracy, and timeliness dimensions for all labels in the
        facility.

        Args:
            facility_id: Facility identifier.

        Returns:
            Composite labeling score (0.0 to 100.0).
        """
        audit_result = self.audit_labeling_compliance(facility_id)
        return audit_result.overall_compliance_score

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _record_event_internal(
        self,
        label_id: str,
        event_type: str,
        operator_id: str,
        notes: str,
    ) -> LabelEvent:
        """Internal helper to create and store a label event.

        Args:
            label_id: Label identifier.
            event_type: Event type string.
            operator_id: Operator identifier.
            notes: Event notes.

        Returns:
            Created LabelEvent.
        """
        event_id = _generate_id("le")
        now = utcnow()

        event = LabelEvent(
            event_id=event_id,
            label_id=label_id,
            event_type=event_type,
            timestamp=now.isoformat(),
            operator_id=operator_id,
            notes=notes,
        )
        event.provenance_hash = _compute_hash({
            "event_id": event_id,
            "label_id": label_id,
            "event_type": event_type,
            "timestamp": event.timestamp,
            "operator_id": operator_id,
        })

        if label_id not in self._label_events:
            self._label_events[label_id] = []
        self._label_events[label_id].append(event)

        return event

    def _assess_readability(
        self,
        label: LabelRecord,
        findings: List[Dict[str, Any]],
    ) -> float:
        """Assess label readability and return score.

        Args:
            label: The label record to assess.
            findings: List to append findings to.

        Returns:
            Readability score (0-100).
        """
        status_readability = {
            "applied": 100.0,
            "readable": 80.0,
            "damaged": 20.0,
            "missing": 0.0,
            "expired": 40.0,
        }
        score = status_readability.get(label.status, 0.0)

        if score < 50.0:
            findings.append({
                "type": "readability",
                "severity": "major" if score < 30.0 else "minor",
                "message": (
                    f"Label {label.label_id} readability issue: "
                    f"status={label.status}"
                ),
            })
        return score

    def _assess_content_accuracy(
        self,
        label: LabelRecord,
        findings: List[Dict[str, Any]],
    ) -> float:
        """Assess label content accuracy and return score.

        Args:
            label: The label record to assess.
            findings: List to append findings to.

        Returns:
            Content accuracy score (0-100).
        """
        required_fields = REQUIRED_LABEL_CONTENT.get(
            label.label_type, [],
        )

        if not required_fields:
            return 100.0

        present_count = 0
        for req_field in required_fields:
            value = label.content_fields.get(req_field)
            if value is not None and str(value).strip():
                present_count += 1
            else:
                findings.append({
                    "type": "accuracy",
                    "severity": "minor",
                    "message": (
                        f"Label {label.label_id} missing required "
                        f"field: {req_field}"
                    ),
                })

        score = (present_count / len(required_fields)) * 100.0
        return score

    def _assess_timeliness(
        self,
        label: LabelRecord,
        now: datetime,
        findings: List[Dict[str, Any]],
    ) -> float:
        """Assess label timeliness (validity period) and return score.

        Args:
            label: The label record to assess.
            now: Current UTC datetime.
            findings: List to append findings to.

        Returns:
            Timeliness score (0-100).
        """
        if not label.expiry_date:
            return 100.0

        try:
            expiry_dt = datetime.fromisoformat(label.expiry_date)
            if expiry_dt.tzinfo is None:
                expiry_dt = expiry_dt.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            return 50.0

        if now > expiry_dt:
            label.status = "expired"
            findings.append({
                "type": "timeliness",
                "severity": "critical",
                "message": (
                    f"Label {label.label_id} expired on "
                    f"{label.expiry_date}"
                ),
            })
            return 0.0

        # Score based on remaining validity
        try:
            applied_dt = datetime.fromisoformat(label.applied_date)
            if applied_dt.tzinfo is None:
                applied_dt = applied_dt.replace(tzinfo=timezone.utc)
            total_validity = (expiry_dt - applied_dt).total_seconds()
            remaining = (expiry_dt - now).total_seconds()
            if total_validity > 0:
                pct_remaining = remaining / total_validity
                score = max(0.0, min(100.0, pct_remaining * 100.0))
            else:
                score = 100.0
        except (ValueError, TypeError):
            score = 50.0

        if score < 20.0:
            findings.append({
                "type": "timeliness",
                "severity": "minor",
                "message": (
                    f"Label {label.label_id} approaching expiry: "
                    f"{label.expiry_date}"
                ),
            })

        return score

    def _compute_single_label_score(
        self,
        label: LabelRecord,
        now: datetime,
    ) -> float:
        """Compute a composite score for a single label.

        Args:
            label: The label record.
            now: Current UTC datetime.

        Returns:
            Composite score (0-100).
        """
        findings_placeholder: List[Dict[str, Any]] = []

        coverage = 100.0 if label.status != "missing" else 0.0
        readability = self._assess_readability(label, findings_placeholder)
        accuracy = self._assess_content_accuracy(label, findings_placeholder)
        timeliness = self._assess_timeliness(label, now, findings_placeholder)

        score = (
            coverage * LABEL_SCORE_WEIGHTS["coverage"]
            + readability * LABEL_SCORE_WEIGHTS["readability"]
            + accuracy * LABEL_SCORE_WEIGHTS["accuracy"]
            + timeliness * LABEL_SCORE_WEIGHTS["timeliness"]
        )
        return round(score, 2)

    def _get_required_label_types(
        self,
        scp_type: str,
    ) -> List[str]:
        """Determine required label types based on SCP type.

        Args:
            scp_type: Type of segregation control point.

        Returns:
            List of required label type strings.
        """
        scp_label_requirements: Dict[str, List[str]] = {
            "storage": ["zone_sign", "batch_sticker", "pallet_marker"],
            "transport": ["vehicle_placard", "container_seal_label"],
            "processing": ["processing_line_marker", "batch_sticker"],
            "handling": ["compliance_tag", "batch_sticker"],
            "loading_unloading": [
                "compliance_tag",
                "container_seal_label",
                "batch_sticker",
            ],
        }
        return scp_label_requirements.get(scp_type, ["compliance_tag"])

    def _generate_audit_recommendations(
        self,
        overall_score: float,
        type_breakdown: Dict[str, float],
        scps_without_labels: int,
        total_scps: int,
        color_compliance: Dict[str, Any],
    ) -> List[str]:
        """Generate prioritized audit recommendations.

        Args:
            overall_score: Composite labeling score.
            type_breakdown: Scores by label type.
            scps_without_labels: Count of SCPs missing labels.
            total_scps: Total SCP count.
            color_compliance: Color code compliance results.

        Returns:
            List of recommendation strings.
        """
        recommendations: List[str] = []

        if scps_without_labels > 0:
            pct = (scps_without_labels / max(total_scps, 1)) * 100.0
            recommendations.append(
                f"Apply labels to {scps_without_labels} SCPs "
                f"({pct:.0f}% of facility) currently without valid labels"
            )

        # Identify weakest label types
        weak_types = [
            (lt, score) for lt, score in type_breakdown.items()
            if 0 < score < 70.0
        ]
        weak_types.sort(key=lambda x: x[1])
        for lt, score in weak_types[:3]:
            recommendations.append(
                f"Improve {lt} labels (current score: {score:.0f}/100)"
            )

        # Color code recommendations
        color_score = color_compliance.get("score", 100.0)
        if color_score < 80.0:
            recommendations.append(
                f"Correct color coding discrepancies "
                f"(current compliance: {color_score:.0f}%)"
            )

        if overall_score < 60.0:
            recommendations.append(
                "Conduct comprehensive relabeling program across facility"
            )
        elif overall_score < 80.0:
            recommendations.append(
                "Schedule targeted label maintenance for low-scoring areas"
            )

        if not recommendations:
            recommendations.append(
                "Maintain current labeling program with regular verification"
            )

        return recommendations

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Constants
    "LABEL_TYPES",
    "LABEL_STATUSES",
    "LABEL_EVENT_TYPES",
    "REQUIRED_LABEL_CONTENT",
    "COLOR_CODE_MAP",
    "STATUS_TO_COLOR",
    "LABEL_SCORE_WEIGHTS",
    "DEFAULT_LABEL_VALIDITY_DAYS",
    # Result types
    "LabelRecord",
    "LabelVerificationResult",
    "LabelAuditResult",
    "LabelEvent",
    # Engine
    "LabelingVerificationEngine",
]
