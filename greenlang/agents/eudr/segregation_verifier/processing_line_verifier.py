# -*- coding: utf-8 -*-
"""
Processing Line Verifier Engine - AGENT-EUDR-010: Segregation Verifier (Feature 4)

Verifies processing line segregation for EUDR-compliant vs non-compliant
commodity material. Registers processing lines with dedication status and
shared equipment tracking, records and validates changeover events (flush
volumes, duration, cleaning, purge methods), checks equipment sharing risks,
verifies temporal separation between compliant and non-compliant runs, flags
first production runs after changeover for enhanced monitoring, and calculates
composite processing segregation scores.

Zero-Hallucination Guarantees:
    - Changeover compliance checks flush volume and duration against per-line-
      type minimum requirements from static lookup tables (no ML/LLM).
    - Temporal separation is datetime arithmetic: gap_hours = (start - end)
      in hours (no ML/LLM).
    - Equipment sharing risk is deterministic weighted sum on shared equipment
      types from EQUIPMENT_SHARING_RISK_WEIGHTS (no ML/LLM).
    - First-run flagging is boolean logic on changeover recency (no ML/LLM).
    - Processing score uses deterministic weighted arithmetic across four
      sub-scores: line_dedication, changeover_compliance, equipment_sharing,
      temporal_separation (no ML/LLM).
    - SHA-256 provenance hashes on all records and verification results.
    - No ML/LLM used for any scoring, verification, or risk logic.

Performance Targets:
    - Line registration: <5ms
    - Changeover recording: <5ms
    - Changeover compliance check: <2ms
    - Equipment sharing assessment: <3ms
    - Temporal separation verification: <2ms
    - Facility-wide processing score: <20ms

Regulatory References:
    - EUDR Article 4: Due diligence obligations requiring segregation.
    - EUDR Article 10(2)(f): Verification of processing segregation measures.
    - EUDR Article 14: Record-keeping for processing evidence.
    - EUDR Article 31: Five-year data retention for audit trails.
    - ISO 22095:2020: Chain of custody -- Processing segregation model.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-010 (Feature 4: Processing Line Verification)
Agent ID: GL-EUDR-SGV-010
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


def _generate_id() -> str:
    """Generate a unique identifier using UUID4."""
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Processing line types with their changeover complexity level (0-1).
#: Higher complexity = stricter changeover requirements.
PROCESSING_LINE_TYPES: Dict[str, float] = {
    "extraction": 0.90,
    "pressing": 0.85,
    "milling": 0.75,
    "refining": 0.95,
    "roasting": 0.70,
    "fermenting": 0.80,
    "drying": 0.60,
    "cutting": 0.50,
    "tanning": 0.85,
    "spinning": 0.65,
    "smelting": 0.90,
    "fractionation": 0.95,
    "blending_line": 0.90,
    "packaging": 0.40,
    "grading": 0.30,
}

#: Minimum changeover times per line type (minutes).
#: Based on industry standards for complete line purge.
MIN_CHANGEOVER_TIMES: Dict[str, int] = {
    "extraction": 120,
    "pressing": 90,
    "milling": 60,
    "refining": 180,
    "roasting": 45,
    "fermenting": 60,
    "drying": 30,
    "cutting": 20,
    "tanning": 90,
    "spinning": 45,
    "smelting": 120,
    "fractionation": 180,
    "blending_line": 120,
    "packaging": 15,
    "grading": 10,
}

#: Minimum flush volumes per line type (liters).
#: Applicable to liquid and semi-liquid processing lines.
MIN_FLUSH_VOLUMES: Dict[str, float] = {
    "extraction": 200.0,
    "pressing": 150.0,
    "milling": 50.0,
    "refining": 500.0,
    "roasting": 0.0,
    "fermenting": 100.0,
    "drying": 0.0,
    "cutting": 0.0,
    "tanning": 300.0,
    "spinning": 0.0,
    "smelting": 0.0,
    "fractionation": 400.0,
    "blending_line": 300.0,
    "packaging": 0.0,
    "grading": 0.0,
}

#: Equipment sharing risk weights: equipment_type -> risk weight (0-1).
#: Higher weight = higher contamination risk from sharing.
EQUIPMENT_SHARING_RISK_WEIGHTS: Dict[str, float] = {
    "scales": 0.30,
    "conveyors": 0.50,
    "hoppers": 0.40,
    "tanks": 0.60,
    "pumps": 0.55,
    "pipes": 0.65,
    "valves": 0.50,
    "filters": 0.45,
    "heat_exchangers": 0.40,
    "forklifts": 0.25,
    "mixers": 0.70,
    "centrifuges": 0.60,
    "dryers": 0.35,
    "grinders": 0.55,
    "packaging_machines": 0.20,
}

#: Minimum temporal separation between non-compliant and compliant runs (hours).
TEMPORAL_SEPARATION_MINIMUM_HOURS: float = 2.0

#: Processing score weights for composite calculation.
PROCESSING_SCORE_WEIGHTS: Dict[str, float] = {
    "line_dedication": 0.30,
    "changeover_compliance": 0.30,
    "equipment_sharing": 0.20,
    "temporal_separation": 0.20,
}

#: Valid cleaning methods for changeover.
VALID_CLEANING_METHODS: Tuple[str, ...] = (
    "power_wash", "steam_clean", "flush", "compressed_air",
    "sweep_wash", "tank_wash", "manual_clean", "cip",
)

#: Valid purge methods for changeover.
VALID_PURGE_METHODS: Tuple[str, ...] = (
    "air_purge", "water_purge", "product_purge", "solvent_purge",
    "steam_purge", "vacuum_purge", "none",
)

#: First-run window: changeovers within this many hours trigger first-run flag.
FIRST_RUN_WINDOW_HOURS: float = 24.0

#: Changeover compliance status values.
CHANGEOVER_COMPLIANT = "compliant"
CHANGEOVER_NON_COMPLIANT = "non_compliant"
CHANGEOVER_PARTIAL = "partial"


# ---------------------------------------------------------------------------
# Internal Data Classes
# ---------------------------------------------------------------------------


@dataclass
class LineRecord:
    """Internal record for a processing line.

    Attributes:
        line_id: Unique identifier for this processing line.
        facility_id: Identifier of the hosting facility.
        line_type: Type of processing line.
        commodity: EUDR commodity processed on this line.
        capacity_kg_per_hour: Maximum throughput in kg/hour.
        dedicated_status: Whether this line is dedicated to compliant material.
        last_changeover_date: Date of most recent changeover.
        shared_equipment: List of equipment types shared with other lines.
        changeover_count: Number of changeovers performed.
        last_batch_type: Type of the last batch processed.
        first_run_flagged: Whether the next run is flagged as first-run.
        metadata: Arbitrary key-value metadata.
        provenance_hash: SHA-256 provenance hash.
        created_at: Record creation timestamp (UTC).
    """

    line_id: str = ""
    facility_id: str = ""
    line_type: str = ""
    commodity: str = ""
    capacity_kg_per_hour: float = 0.0
    dedicated_status: bool = False
    last_changeover_date: Optional[datetime] = None
    shared_equipment: List[str] = field(default_factory=list)
    changeover_count: int = 0
    last_batch_type: str = ""
    first_run_flagged: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    provenance_hash: str = ""
    created_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert line record to dictionary for hashing."""
        return {
            "line_id": self.line_id,
            "facility_id": self.facility_id,
            "line_type": self.line_type,
            "commodity": self.commodity,
            "capacity_kg_per_hour": self.capacity_kg_per_hour,
            "dedicated_status": self.dedicated_status,
            "last_changeover_date": (
                str(self.last_changeover_date)
                if self.last_changeover_date else ""
            ),
            "shared_equipment": list(self.shared_equipment),
            "changeover_count": self.changeover_count,
            "last_batch_type": self.last_batch_type,
            "first_run_flagged": self.first_run_flagged,
            "metadata": dict(self.metadata),
            "created_at": str(self.created_at) if self.created_at else "",
        }


@dataclass
class ChangeoverRecord:
    """Record of a processing line changeover event.

    Attributes:
        changeover_id: Unique identifier for this changeover.
        line_id: Identifier of the processing line.
        facility_id: Identifier of the hosting facility.
        previous_batch_type: Compliance type of the previous batch.
        next_batch_type: Compliance type of the next batch.
        flush_volume_liters: Volume of flush material used (liters).
        flush_duration_minutes: Duration of flushing (minutes).
        cleaning_method: Method used for cleaning during changeover.
        purge_method: Method used for line purging.
        verified_by: Identifier of the verification party.
        verification_notes: Free-text notes from the verifier.
        timestamp: When the changeover was performed (UTC).
        compliant: Whether the changeover meets minimum requirements.
        compliance_status: Detailed compliance status string.
        findings: List of changeover findings.
        provenance_hash: SHA-256 provenance hash.
    """

    changeover_id: str = ""
    line_id: str = ""
    facility_id: str = ""
    previous_batch_type: str = ""
    next_batch_type: str = ""
    flush_volume_liters: float = 0.0
    flush_duration_minutes: int = 0
    cleaning_method: str = ""
    purge_method: str = "none"
    verified_by: str = ""
    verification_notes: str = ""
    timestamp: Optional[datetime] = None
    compliant: bool = False
    compliance_status: str = CHANGEOVER_NON_COMPLIANT
    findings: List[str] = field(default_factory=list)
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert changeover record to dictionary."""
        return {
            "changeover_id": self.changeover_id,
            "line_id": self.line_id,
            "facility_id": self.facility_id,
            "previous_batch_type": self.previous_batch_type,
            "next_batch_type": self.next_batch_type,
            "flush_volume_liters": self.flush_volume_liters,
            "flush_duration_minutes": self.flush_duration_minutes,
            "cleaning_method": self.cleaning_method,
            "purge_method": self.purge_method,
            "verified_by": self.verified_by,
            "verification_notes": self.verification_notes,
            "timestamp": str(self.timestamp) if self.timestamp else "",
            "compliant": self.compliant,
            "compliance_status": self.compliance_status,
            "findings": list(self.findings),
        }


@dataclass
class ProcessingVerificationResult:
    """Result of a comprehensive processing line verification.

    Attributes:
        line_id: Identifier of the verified processing line.
        facility_id: Identifier of the hosting facility.
        overall_score: Composite verification score (0.0-100.0).
        line_dedication_score: Line dedication sub-score (0-100).
        changeover_compliance_score: Changeover compliance sub-score (0-100).
        equipment_sharing_score: Equipment sharing risk sub-score (0-100).
        temporal_separation_score: Temporal separation sub-score (0-100).
        findings: List of verification findings.
        recommendations: List of improvement recommendations.
        verified_at: When the verification was performed (UTC).
        processing_time_ms: Verification processing time in milliseconds.
        provenance_hash: SHA-256 provenance hash.
    """

    line_id: str = ""
    facility_id: str = ""
    overall_score: float = 0.0
    line_dedication_score: float = 0.0
    changeover_compliance_score: float = 0.0
    equipment_sharing_score: float = 0.0
    temporal_separation_score: float = 0.0
    findings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    verified_at: Optional[datetime] = None
    processing_time_ms: float = 0.0
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert verification result to dictionary."""
        return {
            "line_id": self.line_id,
            "facility_id": self.facility_id,
            "overall_score": self.overall_score,
            "line_dedication_score": self.line_dedication_score,
            "changeover_compliance_score": self.changeover_compliance_score,
            "equipment_sharing_score": self.equipment_sharing_score,
            "temporal_separation_score": self.temporal_separation_score,
            "findings": list(self.findings),
            "recommendations": list(self.recommendations),
            "verified_at": (
                str(self.verified_at) if self.verified_at else ""
            ),
            "processing_time_ms": self.processing_time_ms,
        }


# ---------------------------------------------------------------------------
# ProcessingLineVerifier
# ---------------------------------------------------------------------------


class ProcessingLineVerifier:
    """Production-grade processing line verification engine for EUDR compliance.

    Registers processing lines, records changeover events, validates
    changeover compliance against minimum requirements, checks equipment
    sharing risks, verifies temporal separation between runs, flags
    first runs after changeover, and calculates composite processing
    segregation scores.

    All operations are deterministic with zero LLM/ML involvement. Every
    result object carries a SHA-256 provenance hash for complete audit
    trail per EUDR Article 31 (5-year retention).

    Example::

        verifier = ProcessingLineVerifier()
        line = verifier.register_line(
            line_id="LINE-001",
            facility_id="FAC-01",
            line_type="pressing",
            commodity="cocoa",
            capacity_kg_per_hour=1000.0,
            dedicated_status=False,
            shared_equipment=["conveyors", "hoppers"],
        )
        changeover = verifier.record_changeover(
            line_id="LINE-001",
            previous_batch_type="non_compliant",
            next_batch_type="compliant",
            flush_volume=200.0,
            flush_duration=120,
            cleaning_method="steam_clean",
            purge_method="water_purge",
            verified_by="inspector-01",
        )
        result = verifier.verify_processing_segregation("LINE-001")
        assert result.overall_score >= 0.0

    Attributes:
        _lines: In-memory line store keyed by line_id.
        _facility_lines: Facility -> [line_id, ...] index.
        _changeovers: Line -> [ChangeoverRecord, ...] changeover log.
    """

    def __init__(self, config: Any = None) -> None:
        """Initialize the ProcessingLineVerifier.

        Args:
            config: Optional configuration object. Supports attributes:
                - min_changeover_time_minutes (int): Default minimum
                  changeover time override.
                - flush_volume_threshold (float): Default minimum flush
                  volume override.
                - first_run_after_changeover_flag (bool): Whether to flag
                  first runs after changeover.
        """
        self._min_changeover_override: Optional[int] = None
        self._flush_threshold_override: Optional[float] = None
        self._first_run_flag_enabled: bool = True

        if config is not None:
            override_time = getattr(
                config, "min_changeover_time_minutes", None,
            )
            if override_time is not None:
                self._min_changeover_override = int(override_time)

            override_flush = getattr(
                config, "flush_volume_threshold", None,
            )
            if override_flush is not None:
                self._flush_threshold_override = float(override_flush)

            self._first_run_flag_enabled = bool(
                getattr(
                    config, "first_run_after_changeover_flag", True,
                )
            )

        # In-memory stores
        self._lines: Dict[str, LineRecord] = {}
        self._facility_lines: Dict[str, List[str]] = {}
        self._changeovers: Dict[str, List[ChangeoverRecord]] = {}

        logger.info(
            "ProcessingLineVerifier initialized: "
            "changeover_override=%s, flush_override=%s, "
            "first_run_flag=%s",
            self._min_changeover_override,
            self._flush_threshold_override,
            self._first_run_flag_enabled,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def line_count(self) -> int:
        """Return total number of registered processing lines."""
        return len(self._lines)

    @property
    def facility_count(self) -> int:
        """Return total number of distinct facilities."""
        return len(self._facility_lines)

    # ------------------------------------------------------------------
    # Public API: register_line
    # ------------------------------------------------------------------

    def register_line(
        self,
        line_id: str,
        facility_id: str,
        line_type: str,
        commodity: str,
        capacity_kg_per_hour: float = 0.0,
        dedicated_status: bool = False,
        shared_equipment: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LineRecord:
        """Register a new processing line.

        Creates a line record with type, commodity, capacity, dedication
        status, and shared equipment configuration. The line is
        immediately available for changeover recording and verification.

        Args:
            line_id: Unique identifier for the processing line.
            facility_id: Identifier of the hosting facility.
            line_type: Type of processing line (see PROCESSING_LINE_TYPES keys).
            commodity: EUDR commodity processed on this line.
            capacity_kg_per_hour: Maximum throughput in kg/hour (>= 0).
            dedicated_status: Whether line is dedicated to compliant material.
            shared_equipment: List of shared equipment types.
            metadata: Optional additional metadata.

        Returns:
            The newly registered LineRecord with provenance hash.

        Raises:
            ValueError: If line_id already exists or inputs are invalid.
        """
        start_time = time.monotonic()

        self._validate_line_inputs(
            line_id, facility_id, line_type, commodity,
            capacity_kg_per_hour,
        )

        if line_id in self._lines:
            raise ValueError(
                f"Line '{line_id}' already exists"
            )

        # Validate shared equipment types
        equipment = list(shared_equipment) if shared_equipment else []
        unknown_equipment = [
            eq for eq in equipment
            if eq not in EQUIPMENT_SHARING_RISK_WEIGHTS
        ]
        if unknown_equipment:
            logger.warning(
                "Unknown shared equipment types for line %s: %s",
                line_id, unknown_equipment,
            )

        now = _utcnow()
        line = LineRecord(
            line_id=line_id,
            facility_id=facility_id,
            line_type=line_type,
            commodity=commodity,
            capacity_kg_per_hour=capacity_kg_per_hour,
            dedicated_status=dedicated_status,
            shared_equipment=equipment,
            metadata=dict(metadata) if metadata else {},
            created_at=now,
        )
        line.provenance_hash = _compute_hash(line.to_dict())

        # Store
        self._lines[line_id] = line
        if facility_id not in self._facility_lines:
            self._facility_lines[facility_id] = []
        self._facility_lines[facility_id].append(line_id)
        self._changeovers[line_id] = []

        elapsed_ms = (time.monotonic() - start_time) * 1000.0
        logger.info(
            "Registered line %s at facility %s [type=%s, commodity=%s, "
            "dedicated=%s, shared_equipment=%d] in %.2fms",
            line_id, facility_id, line_type, commodity,
            dedicated_status, len(equipment), elapsed_ms,
        )

        return line

    # ------------------------------------------------------------------
    # Public API: get_line
    # ------------------------------------------------------------------

    def get_line(self, line_id: str) -> Optional[LineRecord]:
        """Retrieve a processing line record by identifier.

        Args:
            line_id: The line identifier to look up.

        Returns:
            The LineRecord if found, None otherwise.
        """
        return self._lines.get(line_id)

    # ------------------------------------------------------------------
    # Public API: get_facility_lines
    # ------------------------------------------------------------------

    def get_facility_lines(self, facility_id: str) -> List[LineRecord]:
        """Return all processing lines at a specific facility.

        Args:
            facility_id: Identifier of the facility.

        Returns:
            List of LineRecords at the facility.
        """
        line_ids = self._facility_lines.get(facility_id, [])
        return [
            self._lines[lid] for lid in line_ids
            if lid in self._lines
        ]

    # ------------------------------------------------------------------
    # Public API: record_changeover
    # ------------------------------------------------------------------

    def record_changeover(
        self,
        line_id: str,
        previous_batch_type: str,
        next_batch_type: str,
        flush_volume: float,
        flush_duration: int,
        cleaning_method: str,
        purge_method: str = "none",
        verified_by: str = "",
        verification_notes: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ChangeoverRecord:
        """Record a changeover event for a processing line.

        Validates the changeover against minimum requirements for the
        line type and records the result. If the next batch is compliant,
        flags the first run after changeover for enhanced monitoring.

        Args:
            line_id: Identifier of the processing line.
            previous_batch_type: Compliance type of the previous batch
                (compliant, non_compliant, mixed, unknown).
            next_batch_type: Compliance type of the next batch.
            flush_volume: Volume of flush material used (liters, >= 0).
            flush_duration: Duration of flushing (minutes, >= 0).
            cleaning_method: Method used for cleaning.
            purge_method: Method used for purging (default "none").
            verified_by: Identifier of the verification party.
            verification_notes: Free-text notes from the verifier.
            metadata: Optional additional metadata.

        Returns:
            The recorded ChangeoverRecord with compliance status.

        Raises:
            ValueError: If line not found or inputs are invalid.
        """
        start_time = time.monotonic()

        line = self._lines.get(line_id)
        if line is None:
            raise ValueError(f"Line '{line_id}' not found")

        # Validate inputs
        self._validate_changeover_inputs(
            previous_batch_type, next_batch_type,
            flush_volume, flush_duration,
            cleaning_method, purge_method,
        )

        # Check compliance against minimum requirements
        compliance_result = self._check_changeover_compliance(
            line.line_type, flush_volume, flush_duration,
            cleaning_method, purge_method,
        )
        compliant = compliance_result["compliant"]
        findings = compliance_result.get("findings", [])
        compliance_status = compliance_result.get("status", CHANGEOVER_NON_COMPLIANT)

        now = _utcnow()
        changeover = ChangeoverRecord(
            changeover_id=_generate_id(),
            line_id=line_id,
            facility_id=line.facility_id,
            previous_batch_type=previous_batch_type,
            next_batch_type=next_batch_type,
            flush_volume_liters=flush_volume,
            flush_duration_minutes=flush_duration,
            cleaning_method=cleaning_method,
            purge_method=purge_method,
            verified_by=verified_by,
            verification_notes=verification_notes,
            timestamp=now,
            compliant=compliant,
            compliance_status=compliance_status,
            findings=findings,
        )
        changeover.provenance_hash = _compute_hash(changeover.to_dict())

        # Update line state
        line.last_changeover_date = now
        line.changeover_count += 1
        line.last_batch_type = next_batch_type

        # Flag first run if transitioning to compliant
        if (
            self._first_run_flag_enabled
            and next_batch_type == "compliant"
            and previous_batch_type != "compliant"
        ):
            line.first_run_flagged = True
        else:
            line.first_run_flagged = False

        line.provenance_hash = _compute_hash(line.to_dict())

        # Store changeover
        self._changeovers[line_id].append(changeover)

        elapsed_ms = (time.monotonic() - start_time) * 1000.0
        logger.info(
            "Recorded changeover %s for line %s [%s -> %s, "
            "flush=%.1fL/%dmin, compliant=%s] in %.2fms",
            changeover.changeover_id, line_id,
            previous_batch_type, next_batch_type,
            flush_volume, flush_duration, compliant, elapsed_ms,
        )

        return changeover

    # ------------------------------------------------------------------
    # Public API: verify_changeover_compliance
    # ------------------------------------------------------------------

    def verify_changeover_compliance(
        self,
        changeover_id: str,
    ) -> Dict[str, Any]:
        """Validate a specific changeover against minimum requirements.

        Looks up the changeover by ID and re-validates against the
        processing line type requirements.

        Args:
            changeover_id: Identifier of the changeover to validate.

        Returns:
            Dictionary with keys:
                changeover_id: The changeover identifier.
                compliant: Whether the changeover passes.
                flush_volume_ok: Whether flush volume meets minimum.
                flush_duration_ok: Whether flush duration meets minimum.
                cleaning_method_ok: Whether cleaning method is valid.
                findings: List of detailed findings.

        Raises:
            ValueError: If changeover not found.
        """
        # Search all changeover lists for the ID
        changeover = self._find_changeover(changeover_id)
        if changeover is None:
            raise ValueError(f"Changeover '{changeover_id}' not found")

        line = self._lines.get(changeover.line_id)
        if line is None:
            raise ValueError(
                f"Line '{changeover.line_id}' not found for "
                f"changeover '{changeover_id}'"
            )

        # Re-validate
        result = self._check_changeover_compliance(
            line.line_type,
            changeover.flush_volume_liters,
            changeover.flush_duration_minutes,
            changeover.cleaning_method,
            changeover.purge_method,
        )

        min_time = self._get_min_changeover_time(line.line_type)
        min_flush = self._get_min_flush_volume(line.line_type)

        return {
            "changeover_id": changeover_id,
            "line_id": changeover.line_id,
            "line_type": line.line_type,
            "compliant": result["compliant"],
            "flush_volume_ok": changeover.flush_volume_liters >= min_flush,
            "flush_duration_ok": changeover.flush_duration_minutes >= min_time,
            "cleaning_method_ok": (
                changeover.cleaning_method in VALID_CLEANING_METHODS
            ),
            "min_changeover_time": min_time,
            "actual_duration": changeover.flush_duration_minutes,
            "min_flush_volume": min_flush,
            "actual_flush_volume": changeover.flush_volume_liters,
            "findings": result.get("findings", []),
        }

    # ------------------------------------------------------------------
    # Public API: verify_processing_segregation
    # ------------------------------------------------------------------

    def verify_processing_segregation(
        self,
        line_id: str,
    ) -> ProcessingVerificationResult:
        """Perform a comprehensive processing segregation verification.

        Evaluates the line across four dimensions: dedication status,
        changeover compliance, equipment sharing risk, and temporal
        separation. Produces a composite score using deterministic
        weighted arithmetic.

        Args:
            line_id: Identifier of the processing line.

        Returns:
            ProcessingVerificationResult with scores and findings.

        Raises:
            ValueError: If line not found.
        """
        start_time = time.monotonic()

        line = self._lines.get(line_id)
        if line is None:
            raise ValueError(f"Line '{line_id}' not found")

        findings: List[str] = []
        recommendations: List[str] = []

        # 1. Line dedication score
        dedication_score = self._assess_dedication(line)
        if dedication_score < 50.0:
            findings.append(
                f"WARNING: Line '{line_id}' is not dedicated "
                f"(score={dedication_score:.1f})"
            )
            recommendations.append(
                f"Consider dedicating line '{line_id}' to compliant "
                f"material to reduce changeover risk"
            )

        # 2. Changeover compliance score
        changeover_score = self._assess_changeover_history(line_id)
        if changeover_score < 50.0:
            findings.append(
                f"WARNING: Changeover compliance for line '{line_id}' "
                f"is below threshold ({changeover_score:.1f})"
            )
            recommendations.append(
                f"Review changeover procedures for line '{line_id}' - "
                f"ensure minimum flush volumes and durations are met"
            )

        # 3. Equipment sharing score
        equipment_result = self.check_equipment_sharing(line_id)
        equipment_score = equipment_result.get("sharing_score", 100.0)
        findings.extend(equipment_result.get("findings", []))
        if equipment_score < 50.0:
            recommendations.append(
                f"Reduce shared equipment for line '{line_id}' - "
                f"prioritize dedicated scales, conveyors, and hoppers"
            )

        # 4. Temporal separation score
        temporal_score = self._assess_temporal_separation(line_id)
        if temporal_score < 50.0:
            findings.append(
                f"WARNING: Temporal separation for line '{line_id}' "
                f"is inadequate ({temporal_score:.1f})"
            )
            recommendations.append(
                f"Increase time gap between non-compliant and compliant "
                f"runs on line '{line_id}' to at least "
                f"{TEMPORAL_SEPARATION_MINIMUM_HOURS} hours"
            )

        # Composite score
        overall = (
            dedication_score
            * PROCESSING_SCORE_WEIGHTS["line_dedication"]
            + changeover_score
            * PROCESSING_SCORE_WEIGHTS["changeover_compliance"]
            + equipment_score
            * PROCESSING_SCORE_WEIGHTS["equipment_sharing"]
            + temporal_score
            * PROCESSING_SCORE_WEIGHTS["temporal_separation"]
        )

        now = _utcnow()
        elapsed_ms = (time.monotonic() - start_time) * 1000.0

        result = ProcessingVerificationResult(
            line_id=line_id,
            facility_id=line.facility_id,
            overall_score=round(overall, 2),
            line_dedication_score=round(dedication_score, 2),
            changeover_compliance_score=round(changeover_score, 2),
            equipment_sharing_score=round(equipment_score, 2),
            temporal_separation_score=round(temporal_score, 2),
            findings=findings,
            recommendations=recommendations,
            verified_at=now,
            processing_time_ms=round(elapsed_ms, 2),
        )
        result.provenance_hash = _compute_hash(result.to_dict())

        logger.info(
            "Verified line %s: overall=%.1f [ded=%.1f, chg=%.1f, "
            "eq=%.1f, temp=%.1f], findings=%d in %.2fms",
            line_id, overall, dedication_score, changeover_score,
            equipment_score, temporal_score, len(findings), elapsed_ms,
        )

        return result

    # ------------------------------------------------------------------
    # Public API: check_equipment_sharing
    # ------------------------------------------------------------------

    def check_equipment_sharing(
        self,
        line_id: str,
    ) -> Dict[str, Any]:
        """Identify shared equipment risks for a processing line.

        Uses deterministic weighted sum from EQUIPMENT_SHARING_RISK_WEIGHTS
        to calculate a risk-adjusted sharing score. More shared equipment
        types and higher-risk types result in a lower score.

        Args:
            line_id: Identifier of the processing line.

        Returns:
            Dictionary with keys:
                sharing_score: Score from 0 (all shared, high-risk) to
                    100 (nothing shared).
                shared_equipment_count: Number of shared equipment types.
                risk_weighted_total: Sum of risk weights.
                equipment_details: Per-equipment risk assessment.
                findings: List of findings.

        Raises:
            ValueError: If line not found.
        """
        line = self._lines.get(line_id)
        if line is None:
            raise ValueError(f"Line '{line_id}' not found")

        findings: List[str] = []

        if not line.shared_equipment:
            return {
                "sharing_score": 100.0,
                "shared_equipment_count": 0,
                "risk_weighted_total": 0.0,
                "equipment_details": [],
                "findings": [
                    "INFO: No shared equipment - full equipment dedication"
                ],
            }

        equipment_details: List[Dict[str, Any]] = []
        total_risk = 0.0

        for eq in line.shared_equipment:
            weight = EQUIPMENT_SHARING_RISK_WEIGHTS.get(eq, 0.3)
            total_risk += weight
            equipment_details.append({
                "equipment_type": eq,
                "risk_weight": weight,
            })

            if weight >= 0.6:
                findings.append(
                    f"WARNING: Shared '{eq}' has high contamination "
                    f"risk weight ({weight:.2f})"
                )
            elif weight >= 0.4:
                findings.append(
                    f"INFO: Shared '{eq}' has moderate contamination "
                    f"risk weight ({weight:.2f})"
                )

        # Normalize: max possible risk is sum of all weights (if all shared)
        max_possible_risk = sum(EQUIPMENT_SHARING_RISK_WEIGHTS.values())
        risk_ratio = total_risk / max_possible_risk if max_possible_risk > 0 else 0.0

        # Score: 100 when no risk, 0 when maximum risk
        sharing_score = max(0.0, (1.0 - risk_ratio) * 100.0)

        return {
            "sharing_score": round(sharing_score, 2),
            "shared_equipment_count": len(line.shared_equipment),
            "risk_weighted_total": round(total_risk, 4),
            "equipment_details": equipment_details,
            "findings": findings,
        }

    # ------------------------------------------------------------------
    # Public API: verify_temporal_separation
    # ------------------------------------------------------------------

    def verify_temporal_separation(
        self,
        line_id: str,
        compliant_run_end: datetime,
        non_compliant_run_start: datetime,
    ) -> Dict[str, Any]:
        """Verify minimum temporal separation between runs.

        Calculates the gap in hours between a compliant run's end and
        a non-compliant run's start. The gap must be at least
        TEMPORAL_SEPARATION_MINIMUM_HOURS (2.0 hours).

        Pure datetime arithmetic with no ML/LLM.

        Args:
            line_id: Identifier of the processing line.
            compliant_run_end: End timestamp of the compliant run.
            non_compliant_run_start: Start timestamp of the non-compliant run.

        Returns:
            Dictionary with keys:
                compliant: Whether the temporal gap is sufficient.
                gap_hours: Actual gap in hours.
                minimum_hours: Required minimum gap.
                score: Temporal separation score (0-100).
                findings: List of findings.

        Raises:
            ValueError: If line not found.
        """
        line = self._lines.get(line_id)
        if line is None:
            raise ValueError(f"Line '{line_id}' not found")

        # Calculate gap in hours
        delta = non_compliant_run_start - compliant_run_end
        gap_hours = delta.total_seconds() / 3600.0

        minimum = TEMPORAL_SEPARATION_MINIMUM_HOURS
        compliant = gap_hours >= minimum
        findings: List[str] = []

        if gap_hours < 0:
            findings.append(
                "CRITICAL: Temporal overlap detected - runs are concurrent"
            )
            score = 0.0
        elif gap_hours < minimum:
            findings.append(
                f"WARNING: Temporal gap ({gap_hours:.1f}h) is below "
                f"minimum ({minimum:.1f}h)"
            )
            score = max(0.0, (gap_hours / minimum) * 100.0)
        else:
            score = 100.0
            if gap_hours < minimum * 2:
                findings.append(
                    f"INFO: Temporal gap ({gap_hours:.1f}h) meets minimum "
                    f"but is less than recommended ({minimum * 2:.1f}h)"
                )

        return {
            "compliant": compliant,
            "gap_hours": round(gap_hours, 2),
            "minimum_hours": minimum,
            "score": round(score, 2),
            "findings": findings,
        }

    # ------------------------------------------------------------------
    # Public API: flag_first_run_after_changeover
    # ------------------------------------------------------------------

    def flag_first_run_after_changeover(
        self,
        line_id: str,
    ) -> Dict[str, Any]:
        """Check if the next run on this line should be flagged as first-run.

        A first-run flag indicates that this is the first production run
        after a changeover from non-compliant to compliant material.
        The first run should receive enhanced quality monitoring.

        Args:
            line_id: Identifier of the processing line.

        Returns:
            Dictionary with keys:
                flagged: Whether the first-run flag is active.
                changeover_date: ISO date of the most recent changeover.
                hours_since_changeover: Hours since last changeover.
                changeover_compliant: Whether the last changeover passed.
                recommendation: Action recommendation.

        Raises:
            ValueError: If line not found.
        """
        line = self._lines.get(line_id)
        if line is None:
            raise ValueError(f"Line '{line_id}' not found")

        if not self._first_run_flag_enabled:
            return {
                "flagged": False,
                "changeover_date": "",
                "hours_since_changeover": -1.0,
                "changeover_compliant": False,
                "recommendation": "First-run flagging is disabled",
            }

        flagged = line.first_run_flagged
        changeover_date = ""
        hours_since = -1.0
        last_changeover_compliant = False

        changeovers = self._changeovers.get(line_id, [])
        if changeovers:
            last = changeovers[-1]
            if last.timestamp:
                changeover_date = last.timestamp.isoformat()
                now = _utcnow()
                hours_since = (now - last.timestamp).total_seconds() / 3600.0
            last_changeover_compliant = last.compliant

        recommendation = ""
        if flagged:
            recommendation = (
                "Enhanced quality monitoring required for this batch. "
                "Retain first-run samples for 30 days. "
                "Verify no residual contamination."
            )
            # Check if first-run window has expired
            if hours_since > FIRST_RUN_WINDOW_HOURS:
                recommendation = (
                    "First-run window has expired. "
                    "Flag can be cleared after sample verification."
                )
        else:
            recommendation = "No first-run flag active. Normal operations."

        return {
            "flagged": flagged,
            "changeover_date": changeover_date,
            "hours_since_changeover": round(hours_since, 2),
            "changeover_compliant": last_changeover_compliant,
            "recommendation": recommendation,
        }

    # ------------------------------------------------------------------
    # Public API: calculate_processing_score
    # ------------------------------------------------------------------

    def calculate_processing_score(
        self,
        facility_id: str,
    ) -> float:
        """Calculate the facility-wide processing segregation score.

        Averages the verification scores for all processing lines at
        the facility. Uses verify_processing_segregation for each line.

        Args:
            facility_id: Identifier of the facility.

        Returns:
            Average processing segregation score (0.0-100.0).

        Raises:
            ValueError: If facility has no processing lines.
        """
        lines = self.get_facility_lines(facility_id)
        if not lines:
            raise ValueError(
                f"No processing lines at facility '{facility_id}'"
            )

        scores: List[float] = []
        for line in lines:
            result = self.verify_processing_segregation(line.line_id)
            scores.append(result.overall_score)

        avg = sum(scores) / len(scores) if scores else 0.0
        return round(avg, 2)

    # ------------------------------------------------------------------
    # Public API: get_changeover_history
    # ------------------------------------------------------------------

    def get_changeover_history(
        self,
        line_id: str,
    ) -> List[Dict[str, Any]]:
        """Return the full changeover history for a processing line.

        Args:
            line_id: Identifier of the processing line.

        Returns:
            List of changeover dictionaries, most recent first.

        Raises:
            ValueError: If line not found.
        """
        if line_id not in self._lines:
            raise ValueError(f"Line '{line_id}' not found")

        changeovers = self._changeovers.get(line_id, [])
        return [c.to_dict() for c in reversed(changeovers)]

    # ------------------------------------------------------------------
    # Public API: get_line_contamination_history
    # ------------------------------------------------------------------

    def get_line_contamination_history(
        self,
        line_id: str,
    ) -> List[Dict[str, Any]]:
        """Return the history of non-compliant changeovers for a line.

        Filters changeover history to only include changeovers that
        failed compliance checks, which represent contamination risk
        events.

        Args:
            line_id: Identifier of the processing line.

        Returns:
            List of non-compliant changeover dictionaries.

        Raises:
            ValueError: If line not found.
        """
        if line_id not in self._lines:
            raise ValueError(f"Line '{line_id}' not found")

        changeovers = self._changeovers.get(line_id, [])
        non_compliant = [
            c.to_dict() for c in changeovers
            if not c.compliant
        ]
        non_compliant.reverse()
        return non_compliant

    # ------------------------------------------------------------------
    # Public API: get_summary_statistics
    # ------------------------------------------------------------------

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Return aggregate statistics for all processing lines.

        Returns:
            Dictionary with line counts, changeover counts, and averages.
        """
        by_type: Dict[str, int] = {}
        dedicated_count = 0
        total_changeovers = 0
        compliant_changeovers = 0

        for line in self._lines.values():
            by_type[line.line_type] = (
                by_type.get(line.line_type, 0) + 1
            )
            if line.dedicated_status:
                dedicated_count += 1

        for c_list in self._changeovers.values():
            total_changeovers += len(c_list)
            compliant_changeovers += sum(
                1 for c in c_list if c.compliant
            )

        return {
            "total_lines": len(self._lines),
            "total_facilities": len(self._facility_lines),
            "dedicated_lines": dedicated_count,
            "by_line_type": by_type,
            "total_changeovers": total_changeovers,
            "compliant_changeovers": compliant_changeovers,
            "changeover_compliance_rate": (
                round(compliant_changeovers / total_changeovers * 100, 2)
                if total_changeovers > 0 else 0.0
            ),
        }

    # ------------------------------------------------------------------
    # Public API: clear
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Remove all lines and changeovers. For testing."""
        self._lines.clear()
        self._facility_lines.clear()
        self._changeovers.clear()
        logger.info("ProcessingLineVerifier cleared all data")

    # ------------------------------------------------------------------
    # Internal: _validate_line_inputs
    # ------------------------------------------------------------------

    def _validate_line_inputs(
        self,
        line_id: str,
        facility_id: str,
        line_type: str,
        commodity: str,
        capacity_kg_per_hour: float,
    ) -> None:
        """Validate line registration inputs.

        Raises:
            ValueError: If any input fails validation.
        """
        errors: List[str] = []

        if not line_id or not line_id.strip():
            errors.append("line_id must not be empty")

        if not facility_id or not facility_id.strip():
            errors.append("facility_id must not be empty")

        if line_type not in PROCESSING_LINE_TYPES:
            errors.append(
                f"line_type must be one of "
                f"{sorted(PROCESSING_LINE_TYPES.keys())}, "
                f"got '{line_type}'"
            )

        if not commodity or not commodity.strip():
            errors.append("commodity must not be empty")

        if capacity_kg_per_hour < 0:
            errors.append(
                f"capacity_kg_per_hour must be >= 0, "
                f"got {capacity_kg_per_hour}"
            )

        if errors:
            raise ValueError(
                "Line registration validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

    # ------------------------------------------------------------------
    # Internal: _validate_changeover_inputs
    # ------------------------------------------------------------------

    def _validate_changeover_inputs(
        self,
        previous_batch_type: str,
        next_batch_type: str,
        flush_volume: float,
        flush_duration: int,
        cleaning_method: str,
        purge_method: str,
    ) -> None:
        """Validate changeover inputs.

        Raises:
            ValueError: If any input fails validation.
        """
        errors: List[str] = []

        valid_batch_types = (
            "compliant", "non_compliant", "mixed", "unknown",
        )
        if previous_batch_type not in valid_batch_types:
            errors.append(
                f"previous_batch_type must be one of {valid_batch_types}, "
                f"got '{previous_batch_type}'"
            )
        if next_batch_type not in valid_batch_types:
            errors.append(
                f"next_batch_type must be one of {valid_batch_types}, "
                f"got '{next_batch_type}'"
            )

        if flush_volume < 0:
            errors.append(f"flush_volume must be >= 0, got {flush_volume}")

        if flush_duration < 0:
            errors.append(f"flush_duration must be >= 0, got {flush_duration}")

        if cleaning_method not in VALID_CLEANING_METHODS:
            errors.append(
                f"cleaning_method must be one of "
                f"{VALID_CLEANING_METHODS}, got '{cleaning_method}'"
            )

        if purge_method not in VALID_PURGE_METHODS:
            errors.append(
                f"purge_method must be one of "
                f"{VALID_PURGE_METHODS}, got '{purge_method}'"
            )

        if errors:
            raise ValueError(
                "Changeover validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

    # ------------------------------------------------------------------
    # Internal: _check_changeover_compliance
    # ------------------------------------------------------------------

    def _check_changeover_compliance(
        self,
        line_type: str,
        flush_volume: float,
        flush_duration: int,
        cleaning_method: str,
        purge_method: str,
    ) -> Dict[str, Any]:
        """Check changeover compliance against minimum requirements.

        Args:
            line_type: Processing line type.
            flush_volume: Flush volume in liters.
            flush_duration: Flush duration in minutes.
            cleaning_method: Cleaning method used.
            purge_method: Purge method used.

        Returns:
            Dictionary with compliant flag, status, and findings.
        """
        findings: List[str] = []
        issues = 0

        min_time = self._get_min_changeover_time(line_type)
        min_flush = self._get_min_flush_volume(line_type)

        # Check duration
        if flush_duration < min_time:
            findings.append(
                f"WARNING: Changeover duration {flush_duration}min is below "
                f"minimum {min_time}min for {line_type}"
            )
            issues += 1

        # Check flush volume (only for line types that require flushing)
        if min_flush > 0 and flush_volume < min_flush:
            findings.append(
                f"WARNING: Flush volume {flush_volume:.1f}L is below "
                f"minimum {min_flush:.1f}L for {line_type}"
            )
            issues += 1

        # Check cleaning method quality vs line complexity
        line_complexity = PROCESSING_LINE_TYPES.get(line_type, 0.5)
        # Cleaning methods not in CLEANING_METHODS constant need a
        # separate quality lookup for processing-specific methods
        processing_cleaning_quality: Dict[str, float] = {
            "power_wash": 0.80,
            "steam_clean": 0.90,
            "flush": 0.75,
            "compressed_air": 0.50,
            "sweep_wash": 0.60,
            "tank_wash": 0.95,
            "manual_clean": 0.40,
            "cip": 0.95,
        }
        method_quality = processing_cleaning_quality.get(cleaning_method, 0.50)

        if method_quality < line_complexity * 0.8:
            findings.append(
                f"INFO: Cleaning method '{cleaning_method}' quality "
                f"({method_quality:.2f}) is below recommended level for "
                f"{line_type} (complexity={line_complexity:.2f})"
            )

        # Check purge method for high-complexity lines
        if line_complexity >= 0.8 and purge_method == "none":
            findings.append(
                f"WARNING: High-complexity line type '{line_type}' "
                f"should use a purge method"
            )
            issues += 1

        if issues == 0:
            compliant = True
            status = CHANGEOVER_COMPLIANT
        elif issues == 1:
            compliant = False
            status = CHANGEOVER_PARTIAL
        else:
            compliant = False
            status = CHANGEOVER_NON_COMPLIANT

        return {
            "compliant": compliant,
            "status": status,
            "issues": issues,
            "min_changeover_time": min_time,
            "min_flush_volume": min_flush,
            "findings": findings,
        }

    # ------------------------------------------------------------------
    # Internal: _get_min_changeover_time
    # ------------------------------------------------------------------

    def _get_min_changeover_time(self, line_type: str) -> int:
        """Get the minimum changeover time for a line type.

        Respects the config override if set.

        Args:
            line_type: Processing line type.

        Returns:
            Minimum changeover time in minutes.
        """
        if self._min_changeover_override is not None:
            return self._min_changeover_override

        return MIN_CHANGEOVER_TIMES.get(line_type, 60)

    # ------------------------------------------------------------------
    # Internal: _get_min_flush_volume
    # ------------------------------------------------------------------

    def _get_min_flush_volume(self, line_type: str) -> float:
        """Get the minimum flush volume for a line type.

        Respects the config override if set.

        Args:
            line_type: Processing line type.

        Returns:
            Minimum flush volume in liters.
        """
        if self._flush_threshold_override is not None:
            return self._flush_threshold_override

        return MIN_FLUSH_VOLUMES.get(line_type, 0.0)

    # ------------------------------------------------------------------
    # Internal: _assess_dedication
    # ------------------------------------------------------------------

    def _assess_dedication(self, line: LineRecord) -> float:
        """Assess the dedication status of a processing line.

        Args:
            line: The line record to assess.

        Returns:
            Dedication score (0-100).
        """
        if line.dedicated_status:
            # Dedicated line with no shared equipment = perfect score
            if not line.shared_equipment:
                return 100.0
            # Dedicated but with some shared equipment
            return 80.0

        # Non-dedicated line
        if not line.shared_equipment:
            return 50.0

        # Non-dedicated with shared equipment = lowest score
        return 30.0

    # ------------------------------------------------------------------
    # Internal: _assess_changeover_history
    # ------------------------------------------------------------------

    def _assess_changeover_history(self, line_id: str) -> float:
        """Assess changeover compliance history for a line.

        Args:
            line_id: Identifier of the processing line.

        Returns:
            Changeover compliance score (0-100).
        """
        changeovers = self._changeovers.get(line_id, [])

        if not changeovers:
            # No changeovers = unknown compliance
            return 50.0

        # Calculate compliance rate from recent changeovers (last 10)
        recent = changeovers[-10:]
        compliant_count = sum(1 for c in recent if c.compliant)
        compliance_rate = compliant_count / len(recent)

        return round(compliance_rate * 100.0, 2)

    # ------------------------------------------------------------------
    # Internal: _assess_temporal_separation
    # ------------------------------------------------------------------

    def _assess_temporal_separation(self, line_id: str) -> float:
        """Assess temporal separation quality from changeover history.

        Examines the time gaps between changeovers to determine if
        adequate temporal separation is being maintained.

        Args:
            line_id: Identifier of the processing line.

        Returns:
            Temporal separation score (0-100).
        """
        changeovers = self._changeovers.get(line_id, [])

        if len(changeovers) < 2:
            return 100.0

        # Check gaps between consecutive changeovers
        gaps: List[float] = []
        for i in range(1, len(changeovers)):
            prev = changeovers[i - 1]
            curr = changeovers[i]

            if prev.timestamp and curr.timestamp:
                delta = curr.timestamp - prev.timestamp
                gap_hours = delta.total_seconds() / 3600.0
                gaps.append(gap_hours)

        if not gaps:
            return 100.0

        minimum = TEMPORAL_SEPARATION_MINIMUM_HOURS
        violations = sum(1 for g in gaps if g < minimum)

        if violations == 0:
            return 100.0

        violation_rate = violations / len(gaps)
        return round(max(0.0, (1.0 - violation_rate) * 100.0), 2)

    # ------------------------------------------------------------------
    # Internal: _find_changeover
    # ------------------------------------------------------------------

    def _find_changeover(
        self,
        changeover_id: str,
    ) -> Optional[ChangeoverRecord]:
        """Find a changeover record by ID across all lines.

        Args:
            changeover_id: The changeover identifier to find.

        Returns:
            The ChangeoverRecord if found, None otherwise.
        """
        for line_changeovers in self._changeovers.values():
            for changeover in line_changeovers:
                if changeover.changeover_id == changeover_id:
                    return changeover
        return None

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return total number of registered processing lines."""
        return self.line_count

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        return (
            f"ProcessingLineVerifier("
            f"lines={self.line_count}, "
            f"facilities={self.facility_count})"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Module version
    "_MODULE_VERSION",
    # Constants
    "PROCESSING_LINE_TYPES",
    "MIN_CHANGEOVER_TIMES",
    "MIN_FLUSH_VOLUMES",
    "EQUIPMENT_SHARING_RISK_WEIGHTS",
    "TEMPORAL_SEPARATION_MINIMUM_HOURS",
    "PROCESSING_SCORE_WEIGHTS",
    "VALID_CLEANING_METHODS",
    "VALID_PURGE_METHODS",
    "FIRST_RUN_WINDOW_HOURS",
    "CHANGEOVER_COMPLIANT",
    "CHANGEOVER_NON_COMPLIANT",
    "CHANGEOVER_PARTIAL",
    # Data classes
    "LineRecord",
    "ChangeoverRecord",
    "ProcessingVerificationResult",
    # Engine class
    "ProcessingLineVerifier",
]
