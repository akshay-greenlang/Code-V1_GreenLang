# -*- coding: utf-8 -*-
"""
Segregation Point Validator Engine - AGENT-EUDR-010: Segregation Verifier (Feature 1)

Registers, validates, and manages Segregation Control Points (SCPs) within EUDR
supply chains. Each SCP represents a location or stage where physical segregation
of EUDR-compliant vs non-compliant material must be verified. Supports SCP
registration with capacity and method tracking, deterministic risk classification,
composite compliance scoring, reverification scheduling, bulk import, facility-
scoped queries, and full amendment trail.

Zero-Hallucination Guarantees:
    - Risk classification is a pure dictionary lookup on segregation_method and
      scp_type (no ML/LLM).
    - Compliance scoring uses deterministic weighted arithmetic with configurable
      weights (evidence=0.35, documentation=0.25, historical=0.20,
      method_adequacy=0.20).
    - Reverification scheduling is datetime arithmetic (no ML/LLM).
    - Expiry detection is datetime comparison (no ML/LLM).
    - Auto-discovery from custody events is deterministic pattern matching on
      event facility_id and event_type (no ML/LLM).
    - SHA-256 provenance hashes on all SCP records and validation results.
    - No ML/LLM used for any classification, scoring, or scheduling logic.

Performance Targets:
    - Single SCP registration: <5ms
    - SCP validation: <10ms
    - Risk classification: <1ms
    - Compliance score calculation: <5ms
    - Bulk import (10,000 SCPs): <5 seconds
    - Facility SCP lookup: <2ms

Regulatory References:
    - EUDR Article 4: Due diligence obligations requiring traceability.
    - EUDR Article 10(2)(f): Physical segregation verification requirements.
    - EUDR Article 14: Record-keeping for segregation evidence.
    - EUDR Article 31: Five-year data retention for audit trails.
    - ISO 22095:2020: Chain of custody -- Physical segregation model.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-010 (Feature 1: Segregation Point Validation)
Agent ID: GL-EUDR-SGV-010
Status: Production Ready
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import time
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

def _generate_id() -> str:
    """Generate a unique identifier using UUID4."""
    return str(uuid.uuid4())

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default reverification interval in days.
REVERIFICATION_DAYS_DEFAULT: int = 90

#: Maximum number of SCPs per facility before warning.
MAX_SCPS_PER_FACILITY: int = 10_000

#: Maximum records in a single bulk import.
MAX_BULK_IMPORT_SIZE: int = 100_000

#: SCP compliance score weights for composite calculation.
SCP_SCORE_WEIGHTS: Dict[str, float] = {
    "evidence": 0.35,
    "documentation": 0.25,
    "historical": 0.20,
    "method_adequacy": 0.20,
}

#: Risk classification map: segregation_method -> base risk level.
#: Lower-assurance methods receive higher risk. Deterministic lookup.
RISK_CLASSIFICATION_MAP: Dict[str, str] = {
    "dedicated_facility": "low",
    "separate_building": "low",
    "physical_barrier": "medium",
    "sealed_container": "medium",
    "locked_area": "medium",
    "dedicated_line": "medium",
    "color_coded_zone": "high",
    "temporal_separation": "high",
}

#: SCP type risk modifiers: applied on top of method-based classification.
#: Processing and transport carry higher inherent risk than storage.
SCP_TYPE_RISK_MODIFIERS: Dict[str, int] = {
    "storage": 0,
    "transport": 1,
    "processing": 1,
    "handling": 1,
    "loading_unloading": 1,
}

#: Risk hierarchy for deterministic escalation.
RISK_HIERARCHY: List[str] = ["low", "medium", "high", "critical"]

#: Method adequacy scores (0-100) for compliance scoring.
METHOD_ADEQUACY_SCORES: Dict[str, float] = {
    "dedicated_facility": 100.0,
    "separate_building": 95.0,
    "physical_barrier": 80.0,
    "sealed_container": 75.0,
    "locked_area": 70.0,
    "dedicated_line": 70.0,
    "color_coded_zone": 50.0,
    "temporal_separation": 40.0,
}

#: Valid SCP statuses.
VALID_SCP_STATUSES: Tuple[str, ...] = (
    "verified",
    "unverified",
    "failed",
    "expired",
    "pending_inspection",
)

#: Valid SCP types.
VALID_SCP_TYPES: Tuple[str, ...] = (
    "storage",
    "transport",
    "processing",
    "handling",
    "loading_unloading",
)

#: Valid segregation methods.
VALID_SEGREGATION_METHODS: Tuple[str, ...] = (
    "dedicated_facility",
    "physical_barrier",
    "sealed_container",
    "temporal_separation",
    "dedicated_line",
    "color_coded_zone",
    "locked_area",
    "separate_building",
)

#: Primary EUDR-regulated commodities.
VALID_COMMODITIES: Tuple[str, ...] = (
    "cattle", "cocoa", "coffee", "oil_palm",
    "rubber", "soya", "wood",
)

#: Event types from custody chain that indicate SCP discovery points.
SCP_DISCOVERY_EVENT_TYPES: Tuple[str, ...] = (
    "storage_in",
    "storage_out",
    "processing_in",
    "processing_out",
    "transfer",
    "receipt",
)

#: Mapping from custody event type to default SCP type.
EVENT_TYPE_TO_SCP_TYPE: Dict[str, str] = {
    "storage_in": "storage",
    "storage_out": "storage",
    "processing_in": "processing",
    "processing_out": "processing",
    "transfer": "transport",
    "receipt": "transport",
    "export": "loading_unloading",
    "import": "loading_unloading",
}

# ---------------------------------------------------------------------------
# Internal Data Classes
# ---------------------------------------------------------------------------

@dataclass
class SCPRecord:
    """Internal record for a Segregation Control Point.

    Attributes:
        scp_id: Unique identifier for this segregation control point.
        facility_id: Identifier of the facility hosting this SCP.
        location_lat: Latitude of the SCP location (WGS 84).
        location_lon: Longitude of the SCP location (WGS 84).
        scp_type: Type of segregation control point.
        commodity: EUDR commodity handled at this SCP.
        capacity_kg: Maximum capacity in kilograms.
        segregation_method: Physical method used for segregation.
        status: Current verification status.
        risk_classification: Deterministic risk classification.
        compliance_score: Composite compliance score (0.0-100.0).
        verification_date: Date of most recent verification.
        next_verification_date: Date when reverification is due.
        evidence_score: Physical evidence sub-score (0-100).
        documentation_score: Documentation sub-score (0-100).
        historical_score: Historical performance sub-score (0-100).
        amendment_history: List of amendment records.
        metadata: Arbitrary key-value metadata.
        provenance_hash: SHA-256 provenance hash.
        created_at: Record creation timestamp (UTC).
        updated_at: Record last update timestamp (UTC).
    """

    scp_id: str = ""
    facility_id: str = ""
    location_lat: Optional[float] = None
    location_lon: Optional[float] = None
    scp_type: str = ""
    commodity: str = ""
    capacity_kg: float = 0.0
    segregation_method: str = ""
    status: str = "unverified"
    risk_classification: str = "high"
    compliance_score: float = 0.0
    verification_date: Optional[datetime] = None
    next_verification_date: Optional[datetime] = None
    evidence_score: float = 0.0
    documentation_score: float = 0.0
    historical_score: float = 0.0
    amendment_history: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    provenance_hash: str = ""
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert SCP record to a dictionary for hashing and serialization."""
        return {
            "scp_id": self.scp_id,
            "facility_id": self.facility_id,
            "location_lat": self.location_lat,
            "location_lon": self.location_lon,
            "scp_type": self.scp_type,
            "commodity": self.commodity,
            "capacity_kg": self.capacity_kg,
            "segregation_method": self.segregation_method,
            "status": self.status,
            "risk_classification": self.risk_classification,
            "compliance_score": self.compliance_score,
            "verification_date": (
                str(self.verification_date) if self.verification_date else ""
            ),
            "next_verification_date": (
                str(self.next_verification_date)
                if self.next_verification_date else ""
            ),
            "evidence_score": self.evidence_score,
            "documentation_score": self.documentation_score,
            "historical_score": self.historical_score,
            "metadata": dict(self.metadata),
            "created_at": (
                str(self.created_at) if self.created_at else ""
            ),
            "updated_at": (
                str(self.updated_at) if self.updated_at else ""
            ),
        }

@dataclass
class SCPValidationResult:
    """Result of validating a segregation control point.

    Attributes:
        scp_id: Identifier of the validated SCP.
        valid: Whether the SCP passes all validation checks.
        score: Overall compliance score (0.0-100.0).
        risk_classification: Deterministic risk level.
        findings: List of validation issues or observations.
        evidence: Supporting evidence for the validation.
        processing_time_ms: Validation processing time in milliseconds.
        validated_at: When the validation was performed (UTC).
        provenance_hash: SHA-256 provenance hash.
    """

    scp_id: str = ""
    valid: bool = True
    score: float = 0.0
    risk_classification: str = "high"
    findings: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    processing_time_ms: float = 0.0
    validated_at: Optional[datetime] = None
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert validation result to dictionary."""
        return {
            "scp_id": self.scp_id,
            "valid": self.valid,
            "score": self.score,
            "risk_classification": self.risk_classification,
            "findings": list(self.findings),
            "evidence": dict(self.evidence),
            "processing_time_ms": self.processing_time_ms,
            "validated_at": (
                str(self.validated_at) if self.validated_at else ""
            ),
        }

@dataclass
class SCPSearchResult:
    """Paginated search result for SCP queries.

    Attributes:
        results: List of matching SCP records.
        total_count: Total number of records matching the filter.
        page: Current page number (1-based).
        page_size: Number of records per page.
    """

    results: List[SCPRecord] = field(default_factory=list)
    total_count: int = 0
    page: int = 1
    page_size: int = 50

    def to_dict(self) -> Dict[str, Any]:
        """Convert search result to dictionary."""
        return {
            "total_count": self.total_count,
            "page": self.page,
            "page_size": self.page_size,
            "results": [r.to_dict() for r in self.results],
        }

@dataclass
class SCPAmendment:
    """A single amendment record for an SCP change.

    Attributes:
        amendment_id: Unique identifier for this amendment.
        scp_id: Identifier of the amended SCP.
        field_name: Name of the field that was changed.
        old_value: Previous value (serialized as string).
        new_value: Updated value (serialized as string).
        reason: Reason for the amendment.
        amended_by: Identifier of the user making the change.
        amended_at: Timestamp of the amendment (UTC).
        provenance_hash: SHA-256 hash of this amendment.
    """

    amendment_id: str = ""
    scp_id: str = ""
    field_name: str = ""
    old_value: str = ""
    new_value: str = ""
    reason: str = ""
    amended_by: str = ""
    amended_at: Optional[datetime] = None
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert amendment to dictionary."""
        return {
            "amendment_id": self.amendment_id,
            "scp_id": self.scp_id,
            "field_name": self.field_name,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "reason": self.reason,
            "amended_by": self.amended_by,
            "amended_at": (
                str(self.amended_at) if self.amended_at else ""
            ),
        }

# ---------------------------------------------------------------------------
# SegregationPointValidator
# ---------------------------------------------------------------------------

class SegregationPointValidator:
    """Production-grade segregation control point validation engine.

    Registers, validates, and manages Segregation Control Points (SCPs)
    within EUDR supply chains. Each SCP represents a location or stage
    where physical segregation of EUDR-compliant vs non-compliant material
    must be verified and maintained.

    All operations are deterministic with zero LLM/ML involvement. Every
    result object carries a SHA-256 provenance hash for complete audit
    trail per EUDR Article 31 (5-year retention).

    Example::

        validator = SegregationPointValidator()
        scp = validator.register_scp(
            scp_id="SCP-001",
            facility_id="FAC-COCOA-01",
            scp_type="storage",
            commodity="cocoa",
            segregation_method="physical_barrier",
        )
        assert scp.provenance_hash != ""
        result = validator.validate_scp("SCP-001")
        assert result.risk_classification in ("low", "medium", "high", "critical")

    Attributes:
        _scps: In-memory SCP store keyed by scp_id.
        _facility_index: Facility -> list of scp_ids index.
        _reverification_days: Default reverification interval.
    """

    def __init__(self, config: Any = None) -> None:
        """Initialize the SegregationPointValidator.

        Args:
            config: Optional configuration object. Supports attributes:
                - reverification_interval_days (int): Default days between
                  reverification inspections.
                - risk_threshold_low (float): Score above which risk is low.
                - risk_threshold_medium (float): Score above which risk is medium.
                - risk_threshold_high (float): Score above which risk is high.
        """
        self._reverification_days: int = REVERIFICATION_DAYS_DEFAULT
        self._risk_threshold_low: float = 80.0
        self._risk_threshold_medium: float = 60.0
        self._risk_threshold_high: float = 40.0

        if config is not None:
            self._reverification_days = int(
                getattr(
                    config, "reverification_interval_days",
                    REVERIFICATION_DAYS_DEFAULT,
                )
            )
            self._risk_threshold_low = float(
                getattr(config, "risk_threshold_low", 80.0)
            )
            self._risk_threshold_medium = float(
                getattr(config, "risk_threshold_medium", 60.0)
            )
            self._risk_threshold_high = float(
                getattr(config, "risk_threshold_high", 40.0)
            )

        # In-memory SCP store: scp_id -> SCPRecord
        self._scps: Dict[str, SCPRecord] = {}

        # Facility index: facility_id -> [scp_id, ...]
        self._facility_index: Dict[str, List[str]] = {}

        logger.info(
            "SegregationPointValidator initialized: reverification=%dd, "
            "risk_thresholds=[low=%.1f, medium=%.1f, high=%.1f]",
            self._reverification_days,
            self._risk_threshold_low,
            self._risk_threshold_medium,
            self._risk_threshold_high,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def scp_count(self) -> int:
        """Return total number of registered SCPs."""
        return len(self._scps)

    @property
    def facility_count(self) -> int:
        """Return total number of distinct facilities with SCPs."""
        return len(self._facility_index)

    # ------------------------------------------------------------------
    # Public API: register_scp
    # ------------------------------------------------------------------

    def register_scp(
        self,
        scp_id: str,
        facility_id: str,
        scp_type: str,
        commodity: str,
        segregation_method: str,
        capacity_kg: float = 0.0,
        location_lat: Optional[float] = None,
        location_lon: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SCPRecord:
        """Register a new Segregation Control Point.

        Creates an SCP record with deterministic risk classification and
        initial compliance scoring. The SCP starts in 'unverified' status
        and must be validated via ``validate_scp()`` to achieve 'verified'.

        Args:
            scp_id: Unique identifier for the SCP. Must not already exist.
            facility_id: Identifier of the hosting facility.
            scp_type: Type of SCP (storage, transport, processing,
                handling, loading_unloading).
            commodity: EUDR commodity (cattle, cocoa, coffee, oil_palm,
                rubber, soya, wood).
            segregation_method: Physical segregation method (dedicated_facility,
                physical_barrier, sealed_container, temporal_separation,
                dedicated_line, color_coded_zone, locked_area, separate_building).
            capacity_kg: Maximum capacity in kilograms (>= 0).
            location_lat: Latitude of SCP location (WGS 84, -90 to 90).
            location_lon: Longitude of SCP location (WGS 84, -180 to 180).
            metadata: Optional additional key-value metadata.

        Returns:
            The newly registered SCPRecord with provenance hash.

        Raises:
            ValueError: If scp_id already exists or required fields are
                missing or invalid.
        """
        start_time = time.monotonic()

        # Validate inputs
        self._validate_registration_inputs(
            scp_id, facility_id, scp_type, commodity,
            segregation_method, capacity_kg, location_lat, location_lon,
        )

        # Check for duplicate
        if scp_id in self._scps:
            raise ValueError(
                f"SCP '{scp_id}' already exists. Use update_scp() to modify."
            )

        # Deterministic risk classification
        risk = self.classify_risk(segregation_method, scp_type)

        # Method adequacy for initial compliance score
        method_score = METHOD_ADEQUACY_SCORES.get(segregation_method, 50.0)

        now = utcnow()
        scp = SCPRecord(
            scp_id=scp_id,
            facility_id=facility_id,
            location_lat=location_lat,
            location_lon=location_lon,
            scp_type=scp_type,
            commodity=commodity,
            capacity_kg=capacity_kg,
            segregation_method=segregation_method,
            status="unverified",
            risk_classification=risk,
            compliance_score=method_score * SCP_SCORE_WEIGHTS["method_adequacy"],
            evidence_score=0.0,
            documentation_score=0.0,
            historical_score=0.0,
            metadata=dict(metadata) if metadata else {},
            created_at=now,
            updated_at=now,
        )

        # Compute provenance hash
        scp.provenance_hash = _compute_hash(scp.to_dict())

        # Store
        self._scps[scp_id] = scp

        # Update facility index
        if facility_id not in self._facility_index:
            self._facility_index[facility_id] = []
        self._facility_index[facility_id].append(scp_id)

        elapsed_ms = (time.monotonic() - start_time) * 1000.0
        logger.info(
            "Registered SCP %s at facility %s [type=%s, method=%s, "
            "risk=%s, commodity=%s] in %.2fms",
            scp_id, facility_id, scp_type, segregation_method,
            risk, commodity, elapsed_ms,
        )

        return scp

    # ------------------------------------------------------------------
    # Public API: get_scp
    # ------------------------------------------------------------------

    def get_scp(self, scp_id: str) -> Optional[SCPRecord]:
        """Retrieve an SCP record by identifier.

        Args:
            scp_id: The SCP identifier to look up.

        Returns:
            The SCPRecord if found, None otherwise.
        """
        return self._scps.get(scp_id)

    # ------------------------------------------------------------------
    # Public API: update_scp
    # ------------------------------------------------------------------

    def update_scp(
        self,
        scp_id: str,
        updates: Dict[str, Any],
        reason: str = "",
        amended_by: str = "",
    ) -> SCPRecord:
        """Update an existing SCP record with amendment tracking.

        Records the old and new values of each changed field as an
        amendment entry in the SCP's amendment history. Immutable fields
        (scp_id, created_at) cannot be updated.

        Args:
            scp_id: Identifier of the SCP to update.
            updates: Dictionary of field_name -> new_value pairs.
            reason: Reason for the update (for audit trail).
            amended_by: Identifier of the user making the change.

        Returns:
            The updated SCPRecord with new provenance hash.

        Raises:
            ValueError: If SCP not found, no valid updates, or
                attempting to modify immutable fields.
        """
        start_time = time.monotonic()

        scp = self._scps.get(scp_id)
        if scp is None:
            raise ValueError(f"SCP '{scp_id}' not found")

        # Immutable fields
        immutable = {"scp_id", "created_at", "provenance_hash", "amendment_history"}
        invalid_fields = set(updates.keys()) & immutable
        if invalid_fields:
            raise ValueError(
                f"Cannot modify immutable fields: {sorted(invalid_fields)}"
            )

        # Filter to valid SCP fields only
        valid_fields = {
            "facility_id", "location_lat", "location_lon", "scp_type",
            "commodity", "capacity_kg", "segregation_method", "status",
            "risk_classification", "compliance_score", "verification_date",
            "next_verification_date", "evidence_score", "documentation_score",
            "historical_score", "metadata",
        }
        applicable = {k: v for k, v in updates.items() if k in valid_fields}
        if not applicable:
            raise ValueError("No valid fields to update")

        now = utcnow()
        amendments: List[SCPAmendment] = []

        for field_name, new_value in applicable.items():
            old_value = getattr(scp, field_name, None)
            if str(old_value) == str(new_value):
                continue

            amendment = SCPAmendment(
                amendment_id=_generate_id(),
                scp_id=scp_id,
                field_name=field_name,
                old_value=str(old_value),
                new_value=str(new_value),
                reason=reason,
                amended_by=amended_by,
                amended_at=now,
            )
            amendment.provenance_hash = _compute_hash(amendment.to_dict())
            amendments.append(amendment)

            setattr(scp, field_name, new_value)

        if not amendments:
            logger.debug("No actual changes for SCP %s", scp_id)
            return scp

        # Record amendments
        for a in amendments:
            scp.amendment_history.append(a.to_dict())

        # If segregation method changed, reclassify risk
        if "segregation_method" in applicable:
            scp.risk_classification = self.classify_risk(
                scp.segregation_method, scp.scp_type,
            )

        scp.updated_at = now
        scp.provenance_hash = _compute_hash(scp.to_dict())

        elapsed_ms = (time.monotonic() - start_time) * 1000.0
        logger.info(
            "Updated SCP %s: %d amendments in %.2fms",
            scp_id, len(amendments), elapsed_ms,
        )

        return scp

    # ------------------------------------------------------------------
    # Public API: validate_scp
    # ------------------------------------------------------------------

    def validate_scp(
        self,
        scp_id: str,
        evidence_score: Optional[float] = None,
        documentation_score: Optional[float] = None,
        historical_score: Optional[float] = None,
    ) -> SCPValidationResult:
        """Validate an SCP for compliance with segregation requirements.

        Performs a comprehensive deterministic validation covering physical
        evidence adequacy, documentation completeness, historical
        performance, and method adequacy. Calculates a composite
        compliance score using the SCP_SCORE_WEIGHTS and determines
        pass/fail status.

        Args:
            scp_id: Identifier of the SCP to validate.
            evidence_score: Physical evidence score (0-100). If None,
                uses the SCP's stored evidence_score.
            documentation_score: Documentation score (0-100). If None,
                uses the SCP's stored documentation_score.
            historical_score: Historical performance score (0-100). If None,
                uses the SCP's stored historical_score.

        Returns:
            SCPValidationResult with score, risk, and findings.

        Raises:
            ValueError: If SCP not found.
        """
        start_time = time.monotonic()

        scp = self._scps.get(scp_id)
        if scp is None:
            raise ValueError(f"SCP '{scp_id}' not found")

        findings: List[str] = []
        evidence: Dict[str, Any] = {
            "scp_type": scp.scp_type,
            "commodity": scp.commodity,
            "segregation_method": scp.segregation_method,
            "module_version": _MODULE_VERSION,
        }

        # Update sub-scores if provided
        if evidence_score is not None:
            evidence_score = max(0.0, min(100.0, evidence_score))
            scp.evidence_score = evidence_score
        else:
            evidence_score = scp.evidence_score

        if documentation_score is not None:
            documentation_score = max(0.0, min(100.0, documentation_score))
            scp.documentation_score = documentation_score
        else:
            documentation_score = scp.documentation_score

        if historical_score is not None:
            historical_score = max(0.0, min(100.0, historical_score))
            scp.historical_score = historical_score
        else:
            historical_score = scp.historical_score

        # Validate physical evidence
        ev_findings = self._check_evidence(scp, evidence_score)
        findings.extend(ev_findings)

        # Validate documentation
        doc_findings = self._check_documentation(scp, documentation_score)
        findings.extend(doc_findings)

        # Validate method adequacy
        method_findings = self._check_method_adequacy(scp)
        findings.extend(method_findings)

        # Validate capacity
        cap_findings = self._check_capacity(scp)
        findings.extend(cap_findings)

        # Check verification expiry
        expiry_findings = self._check_verification_expiry(scp)
        findings.extend(expiry_findings)

        # Calculate composite compliance score
        composite_score = self.calculate_compliance_score(scp_id)
        scp.compliance_score = composite_score

        # Determine risk classification
        risk = self.classify_risk(scp.segregation_method, scp.scp_type)
        scp.risk_classification = risk

        # Determine validity (no critical findings, score above threshold)
        has_critical = any("CRITICAL" in f for f in findings)
        is_valid = (
            not has_critical
            and composite_score >= self._risk_threshold_medium
        )

        # Update SCP status
        now = utcnow()
        if is_valid:
            scp.status = "verified"
            scp.verification_date = now
            scp.next_verification_date = now + timedelta(
                days=self._reverification_days
            )
        else:
            scp.status = "failed"

        scp.updated_at = now
        scp.provenance_hash = _compute_hash(scp.to_dict())

        evidence["composite_score"] = composite_score
        evidence["evidence_score"] = evidence_score
        evidence["documentation_score"] = documentation_score
        evidence["historical_score"] = historical_score
        evidence["method_adequacy_score"] = METHOD_ADEQUACY_SCORES.get(
            scp.segregation_method, 50.0,
        )

        elapsed_ms = (time.monotonic() - start_time) * 1000.0

        result = SCPValidationResult(
            scp_id=scp_id,
            valid=is_valid,
            score=composite_score,
            risk_classification=risk,
            findings=findings,
            evidence=evidence,
            processing_time_ms=elapsed_ms,
            validated_at=now,
        )
        result.provenance_hash = _compute_hash(result.to_dict())

        logger.info(
            "Validated SCP %s: valid=%s, score=%.1f, risk=%s, "
            "findings=%d in %.2fms",
            scp_id, is_valid, composite_score, risk,
            len(findings), elapsed_ms,
        )

        return result

    # ------------------------------------------------------------------
    # Public API: classify_risk
    # ------------------------------------------------------------------

    def classify_risk(
        self,
        segregation_method: str,
        scp_type: str,
    ) -> str:
        """Classify risk for a segregation method and SCP type.

        Uses a deterministic two-step lookup:
        1. Base risk from RISK_CLASSIFICATION_MAP keyed on segregation_method.
        2. Risk escalation from SCP_TYPE_RISK_MODIFIERS keyed on scp_type.

        No ML/LLM involved. Pure dictionary lookup and index arithmetic.

        Args:
            segregation_method: Physical segregation method string.
            scp_type: SCP type string.

        Returns:
            Risk classification string: "low", "medium", "high", or "critical".
        """
        # Step 1: base risk from method
        base_risk = RISK_CLASSIFICATION_MAP.get(segregation_method, "high")

        # Step 2: modifier from SCP type
        modifier = SCP_TYPE_RISK_MODIFIERS.get(scp_type, 0)

        # Escalate by modifier steps in the hierarchy
        base_index = RISK_HIERARCHY.index(base_risk)
        escalated_index = min(
            base_index + modifier, len(RISK_HIERARCHY) - 1,
        )

        return RISK_HIERARCHY[escalated_index]

    # ------------------------------------------------------------------
    # Public API: calculate_compliance_score
    # ------------------------------------------------------------------

    def calculate_compliance_score(self, scp_id: str) -> float:
        """Calculate the composite compliance score for an SCP.

        Uses deterministic weighted arithmetic:
            composite = (evidence * 0.35) + (documentation * 0.25)
                      + (historical * 0.20) + (method_adequacy * 0.20)

        Each sub-score is clamped to [0, 100]. The composite is rounded
        to 2 decimal places.

        Args:
            scp_id: Identifier of the SCP.

        Returns:
            Composite compliance score (0.0-100.0).

        Raises:
            ValueError: If SCP not found.
        """
        scp = self._scps.get(scp_id)
        if scp is None:
            raise ValueError(f"SCP '{scp_id}' not found")

        evidence = max(0.0, min(100.0, scp.evidence_score))
        documentation = max(0.0, min(100.0, scp.documentation_score))
        historical = max(0.0, min(100.0, scp.historical_score))
        method_adequacy = METHOD_ADEQUACY_SCORES.get(
            scp.segregation_method, 50.0,
        )

        composite = (
            evidence * SCP_SCORE_WEIGHTS["evidence"]
            + documentation * SCP_SCORE_WEIGHTS["documentation"]
            + historical * SCP_SCORE_WEIGHTS["historical"]
            + method_adequacy * SCP_SCORE_WEIGHTS["method_adequacy"]
        )

        return round(composite, 2)

    # ------------------------------------------------------------------
    # Public API: schedule_reverification
    # ------------------------------------------------------------------

    def schedule_reverification(
        self,
        scp_id: str,
        interval_days: int = 90,
    ) -> SCPRecord:
        """Schedule the next reverification for an SCP.

        Sets the next_verification_date based on the current UTC time
        plus the specified interval. Updates the SCP status to
        'pending_inspection' if not currently verified.

        Args:
            scp_id: Identifier of the SCP.
            interval_days: Number of days until reverification (1-365).

        Returns:
            Updated SCPRecord with new next_verification_date.

        Raises:
            ValueError: If SCP not found or interval_days out of range.
        """
        scp = self._scps.get(scp_id)
        if scp is None:
            raise ValueError(f"SCP '{scp_id}' not found")

        if not (1 <= interval_days <= 365):
            raise ValueError(
                f"interval_days must be in [1, 365], got {interval_days}"
            )

        now = utcnow()
        scp.next_verification_date = now + timedelta(days=interval_days)

        if scp.status != "verified":
            scp.status = "pending_inspection"

        scp.updated_at = now
        scp.provenance_hash = _compute_hash(scp.to_dict())

        logger.info(
            "Scheduled reverification for SCP %s: next=%s (in %d days)",
            scp_id,
            scp.next_verification_date.isoformat(),
            interval_days,
        )

        return scp

    # ------------------------------------------------------------------
    # Public API: get_expired_scps
    # ------------------------------------------------------------------

    def get_expired_scps(self) -> List[SCPRecord]:
        """Return all SCPs whose verification has expired.

        An SCP is considered expired when its next_verification_date is
        in the past (i.e., before the current UTC time). This check is
        pure datetime comparison with no ML/LLM involvement.

        Returns:
            List of SCPRecord objects with expired verification. Also
            updates their status to 'expired' if currently 'verified'.
        """
        now = utcnow()
        expired: List[SCPRecord] = []

        for scp in self._scps.values():
            if scp.next_verification_date is None:
                continue

            if scp.next_verification_date <= now:
                if scp.status == "verified":
                    scp.status = "expired"
                    scp.updated_at = now
                    scp.provenance_hash = _compute_hash(scp.to_dict())
                expired.append(scp)

        logger.info(
            "Found %d expired SCPs out of %d total",
            len(expired), len(self._scps),
        )

        return expired

    # ------------------------------------------------------------------
    # Public API: discover_scps_from_events
    # ------------------------------------------------------------------

    def discover_scps_from_events(
        self,
        custody_events: List[Dict[str, Any]],
    ) -> List[SCPRecord]:
        """Auto-discover potential SCPs from custody chain events.

        Scans custody events for facility_id + event_type patterns that
        indicate a segregation control point should exist. Uses
        deterministic pattern matching (string equality, set membership)
        with zero ML/LLM involvement.

        Discovery logic:
        1. For each event with event_type in SCP_DISCOVERY_EVENT_TYPES,
           extract (facility_id, scp_type from EVENT_TYPE_TO_SCP_TYPE).
        2. If no SCP exists for that (facility_id, scp_type) pair,
           register a new SCP with default settings.
        3. Skip events without facility_id or commodity.

        Args:
            custody_events: List of custody event dictionaries. Expected
                keys: facility_id, event_type, commodity.

        Returns:
            List of newly discovered SCPRecords.
        """
        start_time = time.monotonic()
        discovered: List[SCPRecord] = []
        seen_pairs: set = set()

        # Build existing pair index for fast lookup
        existing_pairs: set = set()
        for scp in self._scps.values():
            existing_pairs.add((scp.facility_id, scp.scp_type))

        for event in custody_events:
            event_type = event.get("event_type", "")
            facility_id = event.get("facility_id", "")
            commodity = event.get("commodity", "")

            if not facility_id or not commodity:
                continue

            if event_type not in SCP_DISCOVERY_EVENT_TYPES:
                continue

            scp_type = EVENT_TYPE_TO_SCP_TYPE.get(event_type, "")
            if not scp_type:
                continue

            pair = (facility_id, scp_type)
            if pair in existing_pairs or pair in seen_pairs:
                continue

            seen_pairs.add(pair)

            # Register discovered SCP with default method
            scp_id = f"scp-auto-{_generate_id()[:8]}"
            default_method = self._default_method_for_type(scp_type)

            try:
                scp = self.register_scp(
                    scp_id=scp_id,
                    facility_id=facility_id,
                    scp_type=scp_type,
                    commodity=commodity,
                    segregation_method=default_method,
                    metadata={"auto_discovered": True, "source_event_type": event_type},
                )
                discovered.append(scp)
            except ValueError as exc:
                logger.warning(
                    "Failed to auto-register SCP for facility=%s type=%s: %s",
                    facility_id, scp_type, str(exc),
                )

        elapsed_ms = (time.monotonic() - start_time) * 1000.0
        logger.info(
            "Discovered %d new SCPs from %d custody events in %.2fms",
            len(discovered), len(custody_events), elapsed_ms,
        )

        return discovered

    # ------------------------------------------------------------------
    # Public API: bulk_import_scps
    # ------------------------------------------------------------------

    def bulk_import_scps(
        self,
        records: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Bulk import SCP records with validation.

        Imports multiple SCP records in a single operation. Each record
        is individually validated and registered. Failed records are
        collected with their error messages.

        Args:
            records: List of SCP data dictionaries. Each must contain:
                scp_id, facility_id, scp_type, commodity, segregation_method.
                Optional: capacity_kg, location_lat, location_lon, metadata.

        Returns:
            Dictionary with import statistics:
                imported_count: Number of successfully imported SCPs.
                failed_count: Number of failed records.
                total_records: Total records in the input.
                errors: List of {index, scp_id, error} for failures.
                imported_ids: List of successfully imported SCP IDs.
                processing_time_ms: Total processing time.

        Raises:
            ValueError: If records exceed MAX_BULK_IMPORT_SIZE.
        """
        start_time = time.monotonic()

        if len(records) > MAX_BULK_IMPORT_SIZE:
            raise ValueError(
                f"Bulk import size {len(records)} exceeds maximum "
                f"{MAX_BULK_IMPORT_SIZE}"
            )

        imported_ids: List[str] = []
        errors: List[Dict[str, Any]] = []

        for idx, record in enumerate(records):
            scp_id = record.get("scp_id", "")
            try:
                scp = self.register_scp(
                    scp_id=scp_id,
                    facility_id=record.get("facility_id", ""),
                    scp_type=record.get("scp_type", ""),
                    commodity=record.get("commodity", ""),
                    segregation_method=record.get("segregation_method", ""),
                    capacity_kg=float(record.get("capacity_kg", 0.0)),
                    location_lat=record.get("location_lat"),
                    location_lon=record.get("location_lon"),
                    metadata=record.get("metadata"),
                )
                imported_ids.append(scp.scp_id)
            except (ValueError, TypeError) as exc:
                errors.append({
                    "index": idx,
                    "scp_id": scp_id,
                    "error": str(exc),
                })

        elapsed_ms = (time.monotonic() - start_time) * 1000.0

        result = {
            "imported_count": len(imported_ids),
            "failed_count": len(errors),
            "total_records": len(records),
            "errors": errors,
            "imported_ids": imported_ids,
            "processing_time_ms": round(elapsed_ms, 2),
            "provenance_hash": _compute_hash({
                "imported_count": len(imported_ids),
                "failed_count": len(errors),
                "total_records": len(records),
            }),
        }

        logger.info(
            "Bulk import: %d imported, %d failed out of %d total in %.2fms",
            len(imported_ids), len(errors), len(records), elapsed_ms,
        )

        return result

    # ------------------------------------------------------------------
    # Public API: search_scps
    # ------------------------------------------------------------------

    def search_scps(
        self,
        filters: Optional[Dict[str, Any]] = None,
        page: int = 1,
        page_size: int = 50,
    ) -> SCPSearchResult:
        """Search SCPs with filtering and pagination.

        Supports filtering by any combination of: facility_id, scp_type,
        commodity, segregation_method, status, risk_classification.
        All filtering is exact string match (deterministic).

        Args:
            filters: Optional dictionary of filter criteria. Supported keys:
                facility_id, scp_type, commodity, segregation_method,
                status, risk_classification, min_score, max_score.
            page: Page number (1-based, default 1).
            page_size: Records per page (default 50, max 1000).

        Returns:
            SCPSearchResult with paginated results and total count.
        """
        if page < 1:
            page = 1
        if page_size < 1:
            page_size = 1
        if page_size > 1000:
            page_size = 1000

        filters = filters or {}

        # Apply filters
        matched = self._apply_filters(list(self._scps.values()), filters)

        # Sort by created_at descending (newest first)
        matched.sort(
            key=lambda s: s.created_at or utcnow(),
            reverse=True,
        )

        total_count = len(matched)

        # Paginate
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        page_results = matched[start_idx:end_idx]

        return SCPSearchResult(
            results=page_results,
            total_count=total_count,
            page=page,
            page_size=page_size,
        )

    # ------------------------------------------------------------------
    # Public API: get_facility_scps
    # ------------------------------------------------------------------

    def get_facility_scps(self, facility_id: str) -> List[SCPRecord]:
        """Return all SCPs for a specific facility.

        Args:
            facility_id: Identifier of the facility.

        Returns:
            List of SCPRecords at the facility, ordered by creation time.
        """
        scp_ids = self._facility_index.get(facility_id, [])
        scps = [
            self._scps[sid] for sid in scp_ids
            if sid in self._scps
        ]

        scps.sort(key=lambda s: s.created_at or utcnow())
        return scps

    # ------------------------------------------------------------------
    # Public API: get_scp_history
    # ------------------------------------------------------------------

    def get_scp_history(self, scp_id: str) -> List[Dict[str, Any]]:
        """Return the full amendment trail for an SCP.

        Args:
            scp_id: Identifier of the SCP.

        Returns:
            List of amendment dictionaries in chronological order.

        Raises:
            ValueError: If SCP not found.
        """
        scp = self._scps.get(scp_id)
        if scp is None:
            raise ValueError(f"SCP '{scp_id}' not found")

        return list(scp.amendment_history)

    # ------------------------------------------------------------------
    # Public API: get_scps_by_status
    # ------------------------------------------------------------------

    def get_scps_by_status(self, status: str) -> List[SCPRecord]:
        """Return all SCPs with a specific status.

        Args:
            status: SCP status string (verified, unverified, failed,
                expired, pending_inspection).

        Returns:
            List of matching SCPRecords.

        Raises:
            ValueError: If status is not a valid SCP status.
        """
        if status not in VALID_SCP_STATUSES:
            raise ValueError(
                f"Invalid status '{status}'. "
                f"Must be one of {VALID_SCP_STATUSES}"
            )

        return [
            scp for scp in self._scps.values()
            if scp.status == status
        ]

    # ------------------------------------------------------------------
    # Public API: get_scps_by_risk
    # ------------------------------------------------------------------

    def get_scps_by_risk(self, risk: str) -> List[SCPRecord]:
        """Return all SCPs with a specific risk classification.

        Args:
            risk: Risk classification (low, medium, high, critical).

        Returns:
            List of matching SCPRecords.

        Raises:
            ValueError: If risk is not valid.
        """
        if risk not in RISK_HIERARCHY:
            raise ValueError(
                f"Invalid risk '{risk}'. "
                f"Must be one of {RISK_HIERARCHY}"
            )

        return [
            scp for scp in self._scps.values()
            if scp.risk_classification == risk
        ]

    # ------------------------------------------------------------------
    # Public API: get_summary_statistics
    # ------------------------------------------------------------------

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Return aggregate statistics for all registered SCPs.

        Returns:
            Dictionary with counts by status, risk, type, and commodity,
            plus average compliance score.
        """
        by_status: Dict[str, int] = {}
        by_risk: Dict[str, int] = {}
        by_type: Dict[str, int] = {}
        by_commodity: Dict[str, int] = {}
        scores: List[float] = []

        for scp in self._scps.values():
            by_status[scp.status] = by_status.get(scp.status, 0) + 1
            by_risk[scp.risk_classification] = (
                by_risk.get(scp.risk_classification, 0) + 1
            )
            by_type[scp.scp_type] = by_type.get(scp.scp_type, 0) + 1
            by_commodity[scp.commodity] = (
                by_commodity.get(scp.commodity, 0) + 1
            )
            scores.append(scp.compliance_score)

        avg_score = sum(scores) / len(scores) if scores else 0.0

        return {
            "total_scps": len(self._scps),
            "total_facilities": len(self._facility_index),
            "by_status": by_status,
            "by_risk": by_risk,
            "by_type": by_type,
            "by_commodity": by_commodity,
            "average_compliance_score": round(avg_score, 2),
        }

    # ------------------------------------------------------------------
    # Public API: delete_scp
    # ------------------------------------------------------------------

    def delete_scp(self, scp_id: str) -> bool:
        """Remove an SCP from the registry.

        Args:
            scp_id: Identifier of the SCP to remove.

        Returns:
            True if the SCP was deleted, False if not found.
        """
        scp = self._scps.pop(scp_id, None)
        if scp is None:
            return False

        # Remove from facility index
        if scp.facility_id in self._facility_index:
            idx_list = self._facility_index[scp.facility_id]
            if scp_id in idx_list:
                idx_list.remove(scp_id)
            if not idx_list:
                del self._facility_index[scp.facility_id]

        logger.info("Deleted SCP %s from facility %s", scp_id, scp.facility_id)
        return True

    # ------------------------------------------------------------------
    # Public API: clear
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Remove all SCPs and reset internal state.

        Intended for testing and teardown.
        """
        self._scps.clear()
        self._facility_index.clear()
        logger.info("SegregationPointValidator cleared all SCPs")

    # ------------------------------------------------------------------
    # Internal: _validate_registration_inputs
    # ------------------------------------------------------------------

    def _validate_registration_inputs(
        self,
        scp_id: str,
        facility_id: str,
        scp_type: str,
        commodity: str,
        segregation_method: str,
        capacity_kg: float,
        location_lat: Optional[float],
        location_lon: Optional[float],
    ) -> None:
        """Validate all registration inputs with deterministic checks.

        Args:
            scp_id: SCP identifier (must be non-empty).
            facility_id: Facility identifier (must be non-empty).
            scp_type: SCP type (must be in VALID_SCP_TYPES).
            commodity: Commodity (must be in VALID_COMMODITIES).
            segregation_method: Method (must be in VALID_SEGREGATION_METHODS).
            capacity_kg: Capacity (must be >= 0).
            location_lat: Latitude (-90 to 90, or None).
            location_lon: Longitude (-180 to 180, or None).

        Raises:
            ValueError: If any input fails validation.
        """
        errors: List[str] = []

        if not scp_id or not scp_id.strip():
            errors.append("scp_id must not be empty")

        if not facility_id or not facility_id.strip():
            errors.append("facility_id must not be empty")

        if scp_type not in VALID_SCP_TYPES:
            errors.append(
                f"scp_type must be one of {VALID_SCP_TYPES}, "
                f"got '{scp_type}'"
            )

        if commodity not in VALID_COMMODITIES:
            errors.append(
                f"commodity must be one of {VALID_COMMODITIES}, "
                f"got '{commodity}'"
            )

        if segregation_method not in VALID_SEGREGATION_METHODS:
            errors.append(
                f"segregation_method must be one of "
                f"{VALID_SEGREGATION_METHODS}, got '{segregation_method}'"
            )

        if capacity_kg < 0:
            errors.append(
                f"capacity_kg must be >= 0, got {capacity_kg}"
            )

        if location_lat is not None:
            if not (-90.0 <= location_lat <= 90.0):
                errors.append(
                    f"location_lat must be in [-90, 90], got {location_lat}"
                )

        if location_lon is not None:
            if not (-180.0 <= location_lon <= 180.0):
                errors.append(
                    f"location_lon must be in [-180, 180], got {location_lon}"
                )

        if errors:
            raise ValueError(
                "SCP registration validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

    # ------------------------------------------------------------------
    # Internal: _check_evidence
    # ------------------------------------------------------------------

    def _check_evidence(
        self,
        scp: SCPRecord,
        evidence_score: float,
    ) -> List[str]:
        """Check physical evidence adequacy for an SCP.

        Args:
            scp: The SCP record to check.
            evidence_score: Current evidence score (0-100).

        Returns:
            List of finding strings.
        """
        findings: List[str] = []

        if evidence_score <= 0.0:
            findings.append(
                "CRITICAL: No physical evidence of segregation provided"
            )
        elif evidence_score < 30.0:
            findings.append(
                f"WARNING: Physical evidence score is very low ({evidence_score:.1f}/100)"
            )
        elif evidence_score < 60.0:
            findings.append(
                f"INFO: Physical evidence score is below target ({evidence_score:.1f}/100)"
            )

        # Method-specific evidence requirements
        method = scp.segregation_method
        if method == "physical_barrier" and evidence_score < 50.0:
            findings.append(
                "WARNING: Physical barrier method requires barrier inspection evidence"
            )
        elif method == "dedicated_facility" and evidence_score < 40.0:
            findings.append(
                "WARNING: Dedicated facility requires facility exclusivity evidence"
            )
        elif method == "sealed_container" and evidence_score < 50.0:
            findings.append(
                "WARNING: Sealed container method requires seal integrity evidence"
            )

        return findings

    # ------------------------------------------------------------------
    # Internal: _check_documentation
    # ------------------------------------------------------------------

    def _check_documentation(
        self,
        scp: SCPRecord,
        documentation_score: float,
    ) -> List[str]:
        """Check documentation completeness for an SCP.

        Args:
            scp: The SCP record to check.
            documentation_score: Current documentation score (0-100).

        Returns:
            List of finding strings.
        """
        findings: List[str] = []

        if documentation_score <= 0.0:
            findings.append(
                "CRITICAL: No documentation provided for segregation procedures"
            )
        elif documentation_score < 30.0:
            findings.append(
                f"WARNING: Documentation score is very low ({documentation_score:.1f}/100)"
            )
        elif documentation_score < 60.0:
            findings.append(
                f"INFO: Documentation score is below target ({documentation_score:.1f}/100)"
            )

        # EUDR Article 14 requires documented procedures
        if documentation_score < 50.0:
            findings.append(
                "WARNING: EUDR Article 14 requires documented segregation procedures"
            )

        return findings

    # ------------------------------------------------------------------
    # Internal: _check_method_adequacy
    # ------------------------------------------------------------------

    def _check_method_adequacy(self, scp: SCPRecord) -> List[str]:
        """Check if the segregation method is adequate for the SCP type.

        Args:
            scp: The SCP record to check.

        Returns:
            List of finding strings.
        """
        findings: List[str] = []

        method = scp.segregation_method
        scp_type = scp.scp_type
        score = METHOD_ADEQUACY_SCORES.get(method, 50.0)

        if score < 50.0:
            findings.append(
                f"WARNING: Segregation method '{method}' has low adequacy "
                f"score ({score:.0f}/100) for {scp_type} SCP"
            )

        # Type-specific method checks
        if scp_type == "processing" and method == "color_coded_zone":
            findings.append(
                "WARNING: Color-coded zones alone are insufficient for "
                "processing line segregation"
            )

        if scp_type == "transport" and method == "temporal_separation":
            findings.append(
                "WARNING: Temporal separation in transport requires "
                "cleaning verification between loads"
            )

        if scp_type == "storage" and method == "temporal_separation":
            findings.append(
                "WARNING: Temporal separation is not recommended for "
                "storage segregation"
            )

        return findings

    # ------------------------------------------------------------------
    # Internal: _check_capacity
    # ------------------------------------------------------------------

    def _check_capacity(self, scp: SCPRecord) -> List[str]:
        """Check capacity configuration for an SCP.

        Args:
            scp: The SCP record to check.

        Returns:
            List of finding strings.
        """
        findings: List[str] = []

        if scp.capacity_kg <= 0:
            findings.append(
                "INFO: SCP has no capacity specified; "
                "capacity tracking is disabled"
            )

        return findings

    # ------------------------------------------------------------------
    # Internal: _check_verification_expiry
    # ------------------------------------------------------------------

    def _check_verification_expiry(self, scp: SCPRecord) -> List[str]:
        """Check if the SCP verification has expired or is near expiry.

        Args:
            scp: The SCP record to check.

        Returns:
            List of finding strings.
        """
        findings: List[str] = []
        now = utcnow()

        if scp.verification_date is None:
            findings.append(
                "INFO: SCP has never been verified"
            )
            return findings

        if scp.next_verification_date is not None:
            if scp.next_verification_date <= now:
                findings.append(
                    "CRITICAL: SCP verification has expired "
                    f"(expired on {scp.next_verification_date.isoformat()})"
                )
            elif scp.next_verification_date <= now + timedelta(days=14):
                days_left = (scp.next_verification_date - now).days
                findings.append(
                    f"WARNING: SCP verification expires in {days_left} days"
                )
            elif scp.next_verification_date <= now + timedelta(days=30):
                days_left = (scp.next_verification_date - now).days
                findings.append(
                    f"INFO: SCP verification due for renewal in {days_left} days"
                )

        return findings

    # ------------------------------------------------------------------
    # Internal: _apply_filters
    # ------------------------------------------------------------------

    def _apply_filters(
        self,
        scps: List[SCPRecord],
        filters: Dict[str, Any],
    ) -> List[SCPRecord]:
        """Apply search filters to a list of SCPs.

        All filtering is exact string match (deterministic). Score
        filters use numeric comparison.

        Args:
            scps: List of SCPs to filter.
            filters: Filter criteria dictionary.

        Returns:
            Filtered list of SCPRecords.
        """
        result = scps

        if "facility_id" in filters:
            result = [
                s for s in result
                if s.facility_id == filters["facility_id"]
            ]

        if "scp_type" in filters:
            result = [
                s for s in result
                if s.scp_type == filters["scp_type"]
            ]

        if "commodity" in filters:
            result = [
                s for s in result
                if s.commodity == filters["commodity"]
            ]

        if "segregation_method" in filters:
            result = [
                s for s in result
                if s.segregation_method == filters["segregation_method"]
            ]

        if "status" in filters:
            result = [
                s for s in result
                if s.status == filters["status"]
            ]

        if "risk_classification" in filters:
            result = [
                s for s in result
                if s.risk_classification == filters["risk_classification"]
            ]

        if "min_score" in filters:
            min_score = float(filters["min_score"])
            result = [
                s for s in result
                if s.compliance_score >= min_score
            ]

        if "max_score" in filters:
            max_score = float(filters["max_score"])
            result = [
                s for s in result
                if s.compliance_score <= max_score
            ]

        return result

    # ------------------------------------------------------------------
    # Internal: _default_method_for_type
    # ------------------------------------------------------------------

    def _default_method_for_type(self, scp_type: str) -> str:
        """Return the default segregation method for an SCP type.

        Used during auto-discovery when no method is specified.

        Args:
            scp_type: SCP type string.

        Returns:
            Default segregation method string.
        """
        defaults: Dict[str, str] = {
            "storage": "physical_barrier",
            "transport": "dedicated_line",
            "processing": "dedicated_line",
            "handling": "color_coded_zone",
            "loading_unloading": "temporal_separation",
        }
        return defaults.get(scp_type, "physical_barrier")

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return total number of registered SCPs."""
        return self.scp_count

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        return (
            f"SegregationPointValidator("
            f"scps={self.scp_count}, "
            f"facilities={self.facility_count}, "
            f"reverification={self._reverification_days}d)"
        )

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Module version
    "_MODULE_VERSION",
    # Constants
    "REVERIFICATION_DAYS_DEFAULT",
    "MAX_SCPS_PER_FACILITY",
    "MAX_BULK_IMPORT_SIZE",
    "SCP_SCORE_WEIGHTS",
    "RISK_CLASSIFICATION_MAP",
    "SCP_TYPE_RISK_MODIFIERS",
    "RISK_HIERARCHY",
    "METHOD_ADEQUACY_SCORES",
    "VALID_SCP_STATUSES",
    "VALID_SCP_TYPES",
    "VALID_SEGREGATION_METHODS",
    "VALID_COMMODITIES",
    "SCP_DISCOVERY_EVENT_TYPES",
    "EVENT_TYPE_TO_SCP_TYPE",
    # Data classes
    "SCPRecord",
    "SCPValidationResult",
    "SCPSearchResult",
    "SCPAmendment",
    # Engine class
    "SegregationPointValidator",
]
