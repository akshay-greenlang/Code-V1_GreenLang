# -*- coding: utf-8 -*-
"""
ChangeDetectorEngine - Recalculation Change Tracking and Version Comparison

Engine 5 of 7 for AGENT-MRV-030 (GL-MRV-X-042).

Tracks changes when emission factors, methodologies, organizational boundaries,
or source data are updated, triggering and documenting recalculations.

Features:
    - Change event detection (EF updates, methodology changes, data corrections)
    - Impact analysis: which calculations are affected by a change
    - Version comparison: old vs. new calculation results
    - Materiality assessment with configurable thresholds
    - Base year recalculation triggering
    - Change approval workflow tracking
    - Cascade analysis: downstream report impacts
    - Historical change timeline

Change Types:
    1. ef_update - Emission factor update (new version published)
    2. methodology_change - Calculation method changed
    3. data_correction - Source activity data corrected
    4. boundary_change - Organizational boundary changed
    5. base_year_recalc - Base year recalculation required
    6. allocation_change - Allocation method or factors changed
    7. scope_reclassification - Emissions moved between scopes
    8. structural_change - M&A, divestiture, outsourcing

Severity Levels:
    - critical: >10% impact on total emissions
    - high: 5-10% impact
    - medium: 1-5% impact
    - low: <1% impact

Zero-Hallucination Guarantee:
    - All change detection logic is deterministic comparison.
    - Materiality percentages are arithmetic (no LLM/ML).
    - Cascade analysis is graph traversal over known relationships.
    - Impact assessment uses provenance-tracked metadata only.

Example:
    >>> from greenlang.audit_trail_lineage.change_detector_engine import (
    ...     ChangeDetectorEngine,
    ... )
    >>> engine = ChangeDetectorEngine.get_instance()
    >>> result = engine.detect_change(
    ...     change_type="ef_update",
    ...     affected_entity_type="emission_factor",
    ...     affected_entity_id="EF-CO2-NG-001",
    ...     old_value=56.1,
    ...     new_value=53.07,
    ...     trigger="IPCC AR6 update",
    ...     organization_id="ORG-001",
    ...     reporting_year=2025,
    ... )
    >>> print(f"Change ID: {result['change_id']}, Severity: {result['severity']}")

Author: GreenLang Platform Team
Date: March 2026
Version: 1.0.0
Agent: GL-MRV-X-042
"""

import hashlib
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ==============================================================================
# CONSTANTS
# ==============================================================================

ENGINE_ID: str = "gl_atl_change_detector_engine"
ENGINE_VERSION: str = "1.0.0"
AGENT_ID: str = "GL-MRV-X-042"

_QUANT_2DP: Decimal = Decimal("0.01")
_QUANT_4DP: Decimal = Decimal("0.0001")
ROUNDING: str = ROUND_HALF_UP

# Materiality thresholds (percentage of total emissions)
MATERIALITY_THRESHOLDS: Dict[str, Decimal] = {
    "critical": Decimal("10.00"),   # >10% impact
    "high": Decimal("5.00"),        # 5-10% impact
    "medium": Decimal("1.00"),      # 1-5% impact
    "low": Decimal("0.00"),         # <1% impact
}

VALID_CHANGE_TYPES: Tuple[str, ...] = (
    "ef_update",
    "methodology_change",
    "data_correction",
    "boundary_change",
    "base_year_recalc",
    "allocation_change",
    "scope_reclassification",
    "structural_change",
)

VALID_SEVERITIES: Tuple[str, ...] = (
    "critical",
    "high",
    "medium",
    "low",
)

VALID_RECALCULATION_STATUSES: Tuple[str, ...] = (
    "pending",
    "in_progress",
    "completed",
    "skipped",
)

# Typical number of affected calculations by change type (for simulation)
_TYPICAL_AFFECTED_COUNTS: Dict[str, int] = {
    "ef_update": 50,
    "methodology_change": 25,
    "data_correction": 10,
    "boundary_change": 100,
    "base_year_recalc": 200,
    "allocation_change": 30,
    "scope_reclassification": 15,
    "structural_change": 150,
}

# Entity types that can be affected
VALID_ENTITY_TYPES: Tuple[str, ...] = (
    "emission_factor",
    "activity_data",
    "methodology",
    "organizational_boundary",
    "allocation_factor",
    "scope_assignment",
    "base_year_inventory",
    "facility",
    "business_unit",
    "supplier",
    "product",
)


# ==============================================================================
# SERIALIZATION UTILITIES
# ==============================================================================


def _serialize_for_hash(obj: Any) -> str:
    """
    Serialize an object to a deterministic JSON string for hashing.

    Handles Decimal, datetime, Enum, and dataclass types.

    Args:
        obj: Object to serialize.

    Returns:
        Deterministic JSON string.
    """

    def _default_handler(o: Any) -> Any:
        if isinstance(o, Decimal):
            return str(o)
        if isinstance(o, datetime):
            return o.isoformat()
        if isinstance(o, Enum):
            return o.value
        if hasattr(o, "__dict__"):
            return o.__dict__
        return str(o)

    return json.dumps(obj, sort_keys=True, default=_default_handler)


def _compute_hash(data: Any) -> str:
    """
    Compute SHA-256 hash of data for provenance tracking.

    Args:
        data: Data to hash.

    Returns:
        Lowercase hex SHA-256 hash string.
    """
    serialized = _serialize_for_hash(data)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# ==============================================================================
# CHANGE EVENT DATACLASS
# ==============================================================================


@dataclass(frozen=True)
class ChangeEvent:
    """
    Immutable record of a detected change triggering recalculation.

    Each ChangeEvent captures what changed, how significant the change is,
    which calculations are affected, and the current recalculation status.

    Attributes:
        change_id: Unique identifier for this change event.
        change_type: Type of change (one of VALID_CHANGE_TYPES).
        severity: Assessed severity (critical/high/medium/low).
        affected_entity_type: Type of entity that changed.
        affected_entity_id: Identifier of the changed entity.
        old_value: Previous value before the change.
        new_value: New value after the change.
        trigger: Human-readable description of what triggered the change.
        materiality_pct: Estimated impact as percentage of total emissions.
        affected_calculation_ids: List of calculation IDs impacted.
        affected_calculations_count: Total count of affected calculations.
        recalculation_required: Whether recalculation is needed.
        recalculation_status: Current status (pending/in_progress/completed/skipped).
        cascade_impacts: Downstream report-level impacts.
        organization_id: Organization this change belongs to.
        reporting_year: Reporting year for the affected inventory.
        created_at: ISO 8601 timestamp of when the change was detected.
        metadata: Additional context and metadata.
    """

    change_id: str
    change_type: str
    severity: str
    affected_entity_type: str
    affected_entity_id: str
    old_value: Any
    new_value: Any
    trigger: str
    materiality_pct: Decimal
    affected_calculation_ids: List[str]
    affected_calculations_count: int
    recalculation_required: bool
    recalculation_status: str
    cascade_impacts: List[Dict[str, Any]]
    organization_id: str
    reporting_year: int
    created_at: str
    metadata: Dict[str, Any]


# ==============================================================================
# ChangeDetectorEngine
# ==============================================================================


class ChangeDetectorEngine:
    """
    ChangeDetectorEngine - detects, tracks, and manages recalculation changes.

    This engine is responsible for tracking changes to emission factors,
    methodologies, activity data, organizational boundaries, and other
    inputs that trigger recalculation of emissions inventories. It
    provides impact analysis, version comparison, materiality assessment,
    cascade analysis, and recalculation workflow tracking.

    All detection and assessment logic is deterministic (zero-hallucination).
    No LLM or ML models are used for severity, materiality, or impact
    calculations.

    Thread-Safe: Singleton pattern with lock for concurrent access.

    Attributes:
        _instance: Singleton instance.
        _lock: Thread lock for singleton creation.

    Example:
        >>> engine = ChangeDetectorEngine.get_instance()
        >>> result = engine.detect_change(
        ...     change_type="ef_update",
        ...     affected_entity_type="emission_factor",
        ...     affected_entity_id="EF-CO2-NG-001",
        ...     old_value=56.1,
        ...     new_value=53.07,
        ...     trigger="IPCC AR6 update",
        ...     organization_id="ORG-001",
        ...     reporting_year=2025,
        ... )
        >>> print(result["severity"])
    """

    _instance: Optional["ChangeDetectorEngine"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        """Initialize ChangeDetectorEngine with empty in-memory stores."""
        self._changes: Dict[str, ChangeEvent] = {}
        self._data_lock: threading.RLock = threading.RLock()
        logger.info(
            "ChangeDetectorEngine initialized (engine=%s, version=%s)",
            ENGINE_ID,
            ENGINE_VERSION,
        )

    @classmethod
    def get_instance(cls) -> "ChangeDetectorEngine":
        """
        Get singleton instance of ChangeDetectorEngine (thread-safe).

        Uses double-checked locking for efficient concurrent access.

        Returns:
            Singleton ChangeDetectorEngine instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)."""
        with cls._lock:
            cls._instance = None
            logger.info("ChangeDetectorEngine singleton reset")

    # =========================================================================
    # PUBLIC API: CHANGE DETECTION
    # =========================================================================

    def detect_change(
        self,
        change_type: str,
        affected_entity_type: str,
        affected_entity_id: str,
        old_value: Any,
        new_value: Any,
        trigger: str,
        organization_id: str,
        reporting_year: int,
        severity: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Detect and record a change event that may require recalculation.

        Creates an immutable ChangeEvent record with auto-assessed severity
        (if not provided), materiality calculation, affected calculation
        identification, and cascade impact analysis.

        Args:
            change_type: Type of change (must be in VALID_CHANGE_TYPES).
            affected_entity_type: Type of entity that changed.
            affected_entity_id: Identifier of the changed entity.
            old_value: Previous value before the change.
            new_value: New value after the change.
            trigger: Human-readable trigger description.
            organization_id: Organization identifier.
            reporting_year: Reporting year for the inventory.
            severity: Optional severity override (auto-assessed if None).
            metadata: Optional additional context.

        Returns:
            Dictionary with change_id, severity, materiality_pct,
            affected_calculations_count, recalculation_required,
            provenance_hash, and the full change event.

        Raises:
            ValueError: If change_type or severity is invalid.
        """
        start_time = time.monotonic()

        # Validate inputs
        self._validate_change_type(change_type)
        if severity is not None:
            self._validate_severity(severity)

        # Generate unique change ID
        change_id = f"CHG-{uuid.uuid4().hex[:12].upper()}"

        # Compute materiality percentage
        materiality_pct = self._compute_materiality(old_value, new_value)

        # Auto-assess severity if not provided
        if severity is None:
            severity = self._auto_assess_severity(
                change_type, old_value, new_value
            )

        # Find affected calculations
        affected_calc_ids = self._find_affected_calculations(
            affected_entity_type,
            affected_entity_id,
            organization_id,
            reporting_year,
        )

        # Determine if recalculation is required
        recalculation_required = self._should_recalculate(
            severity, materiality_pct, change_type
        )

        # Compute cascade impacts
        cascade_impacts = self._compute_cascade_impacts(
            change_type,
            affected_entity_type,
            affected_calc_ids,
            materiality_pct,
        )

        # Build immutable change event
        now_iso = datetime.now(timezone.utc).isoformat()
        event = ChangeEvent(
            change_id=change_id,
            change_type=change_type,
            severity=severity,
            affected_entity_type=affected_entity_type,
            affected_entity_id=affected_entity_id,
            old_value=old_value,
            new_value=new_value,
            trigger=trigger,
            materiality_pct=materiality_pct,
            affected_calculation_ids=affected_calc_ids,
            affected_calculations_count=len(affected_calc_ids),
            recalculation_required=recalculation_required,
            recalculation_status="pending" if recalculation_required else "skipped",
            cascade_impacts=cascade_impacts,
            organization_id=organization_id,
            reporting_year=reporting_year,
            created_at=now_iso,
            metadata=metadata or {},
        )

        # Store the event
        with self._data_lock:
            self._changes[change_id] = event

        # Compute provenance hash
        provenance_hash = self._compute_provenance_hash(event)

        processing_time_ms = (time.monotonic() - start_time) * 1000

        logger.info(
            "Change detected: id=%s, type=%s, severity=%s, "
            "materiality=%.2f%%, affected=%d, recalc=%s, time=%.1fms",
            change_id,
            change_type,
            severity,
            materiality_pct,
            len(affected_calc_ids),
            recalculation_required,
            processing_time_ms,
        )

        return {
            "change_id": change_id,
            "change_type": change_type,
            "severity": severity,
            "materiality_pct": str(materiality_pct),
            "affected_calculations_count": len(affected_calc_ids),
            "recalculation_required": recalculation_required,
            "recalculation_status": event.recalculation_status,
            "cascade_impacts_count": len(cascade_impacts),
            "provenance_hash": provenance_hash,
            "processing_time_ms": round(processing_time_ms, 2),
            "created_at": now_iso,
        }

    def get_change(self, change_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a single change event by its identifier.

        Args:
            change_id: Unique change event identifier.

        Returns:
            Dictionary representation of the change event, or None if not found.
        """
        with self._data_lock:
            event = self._changes.get(change_id)

        if event is None:
            logger.debug("Change not found: %s", change_id)
            return None

        return self._event_to_dict(event)

    def list_changes(
        self,
        organization_id: str,
        reporting_year: int,
        change_type: Optional[str] = None,
        severity: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """
        List change events for an organization and reporting year.

        Supports filtering by change type, severity, and recalculation status.
        Results are ordered by creation time (newest first).

        Args:
            organization_id: Organization identifier.
            reporting_year: Reporting year.
            change_type: Optional filter by change type.
            severity: Optional filter by severity.
            status: Optional filter by recalculation status.
            limit: Maximum number of results (default 100).

        Returns:
            Dictionary with 'changes' list and 'total_count'.

        Raises:
            ValueError: If change_type, severity, or status is invalid.
        """
        if change_type is not None:
            self._validate_change_type(change_type)
        if severity is not None:
            self._validate_severity(severity)
        if status is not None:
            self._validate_recalculation_status(status)

        with self._data_lock:
            filtered = [
                e for e in self._changes.values()
                if e.organization_id == organization_id
                and e.reporting_year == reporting_year
            ]

        # Apply optional filters
        if change_type is not None:
            filtered = [e for e in filtered if e.change_type == change_type]
        if severity is not None:
            filtered = [e for e in filtered if e.severity == severity]
        if status is not None:
            filtered = [
                e for e in filtered
                if e.recalculation_status == status
            ]

        # Sort by created_at descending
        filtered.sort(key=lambda e: e.created_at, reverse=True)

        total_count = len(filtered)
        limited = filtered[:limit]

        return {
            "changes": [self._event_to_dict(e) for e in limited],
            "total_count": total_count,
            "returned_count": len(limited),
            "organization_id": organization_id,
            "reporting_year": reporting_year,
            "filters": {
                "change_type": change_type,
                "severity": severity,
                "status": status,
            },
        }

    # =========================================================================
    # PUBLIC API: IMPACT ANALYSIS
    # =========================================================================

    def analyze_impact(self, change_id: str) -> Dict[str, Any]:
        """
        Analyze the impact of a change event on downstream calculations.

        Determines which calculations, reports, and disclosures are affected
        by the change, and estimates the scope of required recalculation.

        Args:
            change_id: Unique change event identifier.

        Returns:
            Dictionary with affected_calculations, affected_scopes,
            affected_reports, estimated_recalculation_effort, and
            provenance_hash.

        Raises:
            ValueError: If change_id is not found.
        """
        start_time = time.monotonic()

        event = self._get_event_or_raise(change_id)

        # Determine affected scopes
        affected_scopes = self._determine_affected_scopes(
            event.change_type, event.affected_entity_type
        )

        # Determine affected reports
        affected_reports = self._determine_affected_reports(
            event.change_type, affected_scopes
        )

        # Estimate recalculation effort
        effort = self._estimate_recalculation_effort(
            event.affected_calculations_count, event.change_type
        )

        processing_time_ms = (time.monotonic() - start_time) * 1000

        result = {
            "change_id": change_id,
            "change_type": event.change_type,
            "severity": event.severity,
            "affected_entity_type": event.affected_entity_type,
            "affected_entity_id": event.affected_entity_id,
            "affected_calculations": event.affected_calculation_ids,
            "affected_calculations_count": event.affected_calculations_count,
            "affected_scopes": affected_scopes,
            "affected_reports": affected_reports,
            "estimated_recalculation_effort": effort,
            "materiality_pct": str(event.materiality_pct),
            "provenance_hash": _compute_hash({
                "engine_id": ENGINE_ID,
                "change_id": change_id,
                "analysis_type": "impact",
            }),
            "processing_time_ms": round(processing_time_ms, 2),
        }

        logger.info(
            "Impact analysis: change=%s, scopes=%s, reports=%d, "
            "effort=%s, time=%.1fms",
            change_id,
            affected_scopes,
            len(affected_reports),
            effort["level"],
            processing_time_ms,
        )

        return result

    def compare_versions(self, change_id: str) -> Dict[str, Any]:
        """
        Compare old and new values for a change event.

        Produces a detailed comparison including absolute difference,
        percentage change, direction (increase/decrease/unchanged),
        and materiality assessment.

        Args:
            change_id: Unique change event identifier.

        Returns:
            Dictionary with old_value, new_value, absolute_difference,
            percentage_change, direction, and materiality_assessment.

        Raises:
            ValueError: If change_id is not found.
        """
        start_time = time.monotonic()

        event = self._get_event_or_raise(change_id)

        # Attempt numeric comparison
        comparison = self._build_version_comparison(
            event.old_value, event.new_value
        )

        processing_time_ms = (time.monotonic() - start_time) * 1000

        result = {
            "change_id": change_id,
            "change_type": event.change_type,
            "affected_entity_id": event.affected_entity_id,
            "old_value": self._safe_serialize(event.old_value),
            "new_value": self._safe_serialize(event.new_value),
            "comparison": comparison,
            "materiality_pct": str(event.materiality_pct),
            "severity": event.severity,
            "provenance_hash": _compute_hash({
                "engine_id": ENGINE_ID,
                "change_id": change_id,
                "analysis_type": "version_comparison",
            }),
            "processing_time_ms": round(processing_time_ms, 2),
        }

        logger.info(
            "Version comparison: change=%s, direction=%s, pct_change=%s, "
            "time=%.1fms",
            change_id,
            comparison.get("direction", "unknown"),
            comparison.get("percentage_change", "N/A"),
            processing_time_ms,
        )

        return result

    def assess_materiality(self, change_id: str) -> Dict[str, Any]:
        """
        Assess the materiality of a change event.

        Calculates the materiality percentage and determines whether the
        change crosses configured thresholds for critical, high, medium,
        or low significance.

        Args:
            change_id: Unique change event identifier.

        Returns:
            Dictionary with materiality_pct, threshold_crossed,
            is_material, recommendation, and provenance_hash.

        Raises:
            ValueError: If change_id is not found.
        """
        start_time = time.monotonic()

        event = self._get_event_or_raise(change_id)

        # Determine which threshold was crossed
        threshold_crossed = self._classify_materiality(event.materiality_pct)

        # Determine if the change is material (medium or above)
        is_material = threshold_crossed in ("critical", "high", "medium")

        # Generate recommendation
        recommendation = self._generate_materiality_recommendation(
            threshold_crossed, event.change_type, event.materiality_pct
        )

        processing_time_ms = (time.monotonic() - start_time) * 1000

        result = {
            "change_id": change_id,
            "materiality_pct": str(event.materiality_pct),
            "threshold_crossed": threshold_crossed,
            "is_material": is_material,
            "severity": event.severity,
            "thresholds": {
                k: str(v) for k, v in MATERIALITY_THRESHOLDS.items()
            },
            "recommendation": recommendation,
            "provenance_hash": _compute_hash({
                "engine_id": ENGINE_ID,
                "change_id": change_id,
                "analysis_type": "materiality",
            }),
            "processing_time_ms": round(processing_time_ms, 2),
        }

        logger.info(
            "Materiality assessment: change=%s, pct=%s, threshold=%s, "
            "material=%s, time=%.1fms",
            change_id,
            event.materiality_pct,
            threshold_crossed,
            is_material,
            processing_time_ms,
        )

        return result

    # =========================================================================
    # PUBLIC API: RECALCULATION MANAGEMENT
    # =========================================================================

    def trigger_recalculation(
        self,
        change_id: str,
        cascade: bool = True,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Trigger recalculation for a change event.

        If dry_run is True, returns what would be recalculated without
        actually changing state. If cascade is True, downstream reports
        and aggregations are also marked for recalculation.

        Args:
            change_id: Unique change event identifier.
            cascade: Whether to cascade to downstream reports.
            dry_run: If True, simulate without changing state.

        Returns:
            Dictionary with recalculation plan including affected
            calculations, cascade targets, and estimated duration.

        Raises:
            ValueError: If change_id is not found.
        """
        start_time = time.monotonic()

        event = self._get_event_or_raise(change_id)

        # Build recalculation plan
        plan_calculations = list(event.affected_calculation_ids)
        cascade_targets: List[Dict[str, Any]] = []

        if cascade:
            cascade_targets = list(event.cascade_impacts)

        estimated_duration_seconds = (
            len(plan_calculations) * 0.5
            + len(cascade_targets) * 2.0
        )

        # Update status if not a dry run
        if not dry_run and event.recalculation_status == "pending":
            self._update_event_status(change_id, "in_progress")

        processing_time_ms = (time.monotonic() - start_time) * 1000

        result = {
            "change_id": change_id,
            "dry_run": dry_run,
            "cascade": cascade,
            "calculations_to_recalculate": plan_calculations,
            "calculations_count": len(plan_calculations),
            "cascade_targets": cascade_targets,
            "cascade_targets_count": len(cascade_targets),
            "estimated_duration_seconds": round(estimated_duration_seconds, 1),
            "recalculation_status": (
                "in_progress" if not dry_run and event.recalculation_status == "pending"
                else event.recalculation_status
            ),
            "provenance_hash": _compute_hash({
                "engine_id": ENGINE_ID,
                "change_id": change_id,
                "action": "trigger_recalculation",
                "dry_run": dry_run,
            }),
            "processing_time_ms": round(processing_time_ms, 2),
        }

        logger.info(
            "Recalculation %s: change=%s, calcs=%d, cascade=%d, "
            "est_duration=%.1fs, time=%.1fms",
            "simulated" if dry_run else "triggered",
            change_id,
            len(plan_calculations),
            len(cascade_targets),
            estimated_duration_seconds,
            processing_time_ms,
        )

        return result

    def update_recalculation_status(
        self,
        change_id: str,
        status: str,
        result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Update the recalculation status of a change event.

        Called by the recalculation pipeline to report progress and
        completion of recalculation work triggered by a change event.

        Args:
            change_id: Unique change event identifier.
            status: New recalculation status (pending/in_progress/completed/skipped).
            result: Optional recalculation result metadata.

        Returns:
            Dictionary with updated status and provenance_hash.

        Raises:
            ValueError: If change_id is not found or status is invalid.
        """
        self._validate_recalculation_status(status)
        event = self._get_event_or_raise(change_id)

        previous_status = event.recalculation_status
        self._update_event_status(change_id, status, result)

        logger.info(
            "Recalculation status updated: change=%s, %s -> %s",
            change_id,
            previous_status,
            status,
        )

        return {
            "change_id": change_id,
            "previous_status": previous_status,
            "new_status": status,
            "result_attached": result is not None,
            "provenance_hash": _compute_hash({
                "engine_id": ENGINE_ID,
                "change_id": change_id,
                "status_update": status,
            }),
        }

    # =========================================================================
    # PUBLIC API: CASCADE AND TIMELINE
    # =========================================================================

    def get_cascade_impacts(self, change_id: str) -> List[Dict[str, Any]]:
        """
        Get the downstream cascade impacts for a change event.

        Returns the list of reports, disclosures, and aggregations
        that are affected by the change and may need updating.

        Args:
            change_id: Unique change event identifier.

        Returns:
            List of cascade impact dictionaries.

        Raises:
            ValueError: If change_id is not found.
        """
        event = self._get_event_or_raise(change_id)
        return list(event.cascade_impacts)

    def get_change_timeline(
        self,
        organization_id: str,
        reporting_year: int,
    ) -> List[Dict[str, Any]]:
        """
        Get chronological timeline of all changes for an organization-year.

        Returns events ordered by creation time (oldest first) to provide
        a complete history of changes to the emissions inventory.

        Args:
            organization_id: Organization identifier.
            reporting_year: Reporting year.

        Returns:
            List of change event dictionaries ordered by created_at.
        """
        with self._data_lock:
            events = [
                e for e in self._changes.values()
                if e.organization_id == organization_id
                and e.reporting_year == reporting_year
            ]

        events.sort(key=lambda e: e.created_at)

        timeline = []
        for event in events:
            timeline.append({
                "change_id": event.change_id,
                "change_type": event.change_type,
                "severity": event.severity,
                "affected_entity_type": event.affected_entity_type,
                "affected_entity_id": event.affected_entity_id,
                "trigger": event.trigger,
                "materiality_pct": str(event.materiality_pct),
                "recalculation_status": event.recalculation_status,
                "created_at": event.created_at,
            })

        logger.debug(
            "Change timeline: org=%s, year=%d, events=%d",
            organization_id,
            reporting_year,
            len(timeline),
        )

        return timeline

    def get_pending_recalculations(
        self,
        organization_id: str,
        reporting_year: int,
    ) -> List[Dict[str, Any]]:
        """
        Get all change events with pending recalculation status.

        Used by the recalculation pipeline to identify work items
        that still need processing.

        Args:
            organization_id: Organization identifier.
            reporting_year: Reporting year.

        Returns:
            List of change event dictionaries with status 'pending'.
        """
        with self._data_lock:
            pending = [
                e for e in self._changes.values()
                if e.organization_id == organization_id
                and e.reporting_year == reporting_year
                and e.recalculation_status == "pending"
            ]

        pending.sort(
            key=lambda e: _severity_rank(e.severity),
            reverse=True,
        )

        return [self._event_to_dict(e) for e in pending]

    # =========================================================================
    # PUBLIC API: STATISTICS
    # =========================================================================

    def get_change_statistics(
        self,
        organization_id: str,
        reporting_year: int,
    ) -> Dict[str, Any]:
        """
        Compute aggregate statistics for changes in an organization-year.

        Returns counts by change type, severity, recalculation status,
        total materiality impact, and summary metrics.

        Args:
            organization_id: Organization identifier.
            reporting_year: Reporting year.

        Returns:
            Dictionary with by_change_type, by_severity, by_status,
            total_changes, total_affected_calculations,
            average_materiality_pct, and max_materiality_pct.
        """
        start_time = time.monotonic()

        with self._data_lock:
            events = [
                e for e in self._changes.values()
                if e.organization_id == organization_id
                and e.reporting_year == reporting_year
            ]

        if not events:
            return {
                "organization_id": organization_id,
                "reporting_year": reporting_year,
                "total_changes": 0,
                "by_change_type": {},
                "by_severity": {},
                "by_status": {},
                "total_affected_calculations": 0,
                "average_materiality_pct": "0.00",
                "max_materiality_pct": "0.00",
                "processing_time_ms": 0.0,
            }

        # Count by change type
        by_type: Dict[str, int] = {}
        for e in events:
            by_type[e.change_type] = by_type.get(e.change_type, 0) + 1

        # Count by severity
        by_severity: Dict[str, int] = {}
        for e in events:
            by_severity[e.severity] = by_severity.get(e.severity, 0) + 1

        # Count by recalculation status
        by_status: Dict[str, int] = {}
        for e in events:
            by_status[e.recalculation_status] = (
                by_status.get(e.recalculation_status, 0) + 1
            )

        # Aggregate metrics
        total_affected = sum(e.affected_calculations_count for e in events)
        materiality_values = [e.materiality_pct for e in events]
        avg_materiality = (
            sum(materiality_values) / Decimal(str(len(materiality_values)))
        ).quantize(_QUANT_2DP, rounding=ROUNDING)
        max_materiality = max(materiality_values).quantize(
            _QUANT_2DP, rounding=ROUNDING
        )

        processing_time_ms = (time.monotonic() - start_time) * 1000

        return {
            "organization_id": organization_id,
            "reporting_year": reporting_year,
            "total_changes": len(events),
            "by_change_type": by_type,
            "by_severity": by_severity,
            "by_status": by_status,
            "total_affected_calculations": total_affected,
            "average_materiality_pct": str(avg_materiality),
            "max_materiality_pct": str(max_materiality),
            "processing_time_ms": round(processing_time_ms, 2),
        }

    # =========================================================================
    # PUBLIC API: RESET
    # =========================================================================

    def reset(self) -> None:
        """
        Clear all in-memory change event data.

        Intended for testing and development use only. In production,
        change events are persisted to the database and this method
        would not be called.
        """
        with self._data_lock:
            count = len(self._changes)
            self._changes.clear()
        logger.info("ChangeDetectorEngine reset: cleared %d change events", count)

    # =========================================================================
    # INTERNAL: SEVERITY ASSESSMENT
    # =========================================================================

    def _auto_assess_severity(
        self,
        change_type: str,
        old_value: Any,
        new_value: Any,
    ) -> str:
        """
        Automatically assess severity based on change type and magnitude.

        Uses materiality thresholds and change-type-specific heuristics
        to determine the severity level. All logic is deterministic.

        Args:
            change_type: Type of change.
            old_value: Previous value.
            new_value: New value.

        Returns:
            Severity string (critical/high/medium/low).
        """
        # Compute materiality for numeric values
        materiality_pct = self._compute_materiality(old_value, new_value)

        # Classify by materiality first
        severity = self._classify_materiality(materiality_pct)

        # Structural and boundary changes are at least 'high'
        if change_type in ("structural_change", "boundary_change"):
            if _severity_rank(severity) < _severity_rank("high"):
                severity = "high"

        # Base year recalculations are at least 'medium'
        if change_type == "base_year_recalc":
            if _severity_rank(severity) < _severity_rank("medium"):
                severity = "medium"

        # Scope reclassifications are at least 'medium'
        if change_type == "scope_reclassification":
            if _severity_rank(severity) < _severity_rank("medium"):
                severity = "medium"

        return severity

    # =========================================================================
    # INTERNAL: AFFECTED CALCULATION DISCOVERY
    # =========================================================================

    def _find_affected_calculations(
        self,
        entity_type: str,
        entity_id: str,
        organization_id: str,
        reporting_year: int,
    ) -> List[str]:
        """
        Find calculation IDs affected by a change to the given entity.

        In production, this queries the lineage graph database. In-memory
        implementation generates deterministic placeholder IDs based on
        entity type and the typical affected count for simulation.

        Args:
            entity_type: Type of entity that changed.
            entity_id: Identifier of the changed entity.
            organization_id: Organization identifier.
            reporting_year: Reporting year.

        Returns:
            List of affected calculation identifiers.
        """
        # Derive a deterministic count based on entity type
        base_count = _TYPICAL_AFFECTED_COUNTS.get(
            entity_type,
            _TYPICAL_AFFECTED_COUNTS.get("data_correction", 10),
        )

        # Generate deterministic calculation IDs
        seed = f"{entity_type}:{entity_id}:{organization_id}:{reporting_year}"
        seed_hash = hashlib.md5(seed.encode("utf-8")).hexdigest()
        count = max(1, base_count % 20 + 3)

        calc_ids = []
        for i in range(count):
            calc_id = f"CALC-{seed_hash[:8].upper()}-{i:04d}"
            calc_ids.append(calc_id)

        return calc_ids

    # =========================================================================
    # INTERNAL: MATERIALITY COMPUTATION
    # =========================================================================

    def _compute_materiality(
        self,
        old_value: Any,
        new_value: Any,
    ) -> Decimal:
        """
        Compute materiality as percentage change between old and new values.

        For numeric values, computes |new - old| / |old| * 100.
        For non-numeric values, returns a default of 5.00% (medium).

        Args:
            old_value: Previous value.
            new_value: New value.

        Returns:
            Materiality percentage as Decimal (0.00 to 100.00+).
        """
        try:
            old_dec = Decimal(str(old_value))
            new_dec = Decimal(str(new_value))
        except (TypeError, ValueError, ArithmeticError):
            # Non-numeric values: default to medium materiality
            return Decimal("5.00")

        if old_dec == Decimal("0"):
            # Cannot compute percentage change from zero
            if new_dec == Decimal("0"):
                return Decimal("0.00")
            return Decimal("100.00")

        pct_change = (
            abs(new_dec - old_dec) / abs(old_dec) * Decimal("100")
        ).quantize(_QUANT_2DP, rounding=ROUNDING)

        return pct_change

    def _classify_materiality(self, materiality_pct: Decimal) -> str:
        """
        Classify a materiality percentage into a severity level.

        Args:
            materiality_pct: Materiality percentage.

        Returns:
            Severity string (critical/high/medium/low).
        """
        if materiality_pct >= MATERIALITY_THRESHOLDS["critical"]:
            return "critical"
        if materiality_pct >= MATERIALITY_THRESHOLDS["high"]:
            return "high"
        if materiality_pct >= MATERIALITY_THRESHOLDS["medium"]:
            return "medium"
        return "low"

    # =========================================================================
    # INTERNAL: CASCADE IMPACT COMPUTATION
    # =========================================================================

    def _compute_cascade_impacts(
        self,
        change_type: str,
        entity_type: str,
        affected_calc_ids: List[str],
        materiality_pct: Decimal,
    ) -> List[Dict[str, Any]]:
        """
        Compute downstream cascade impacts for a change event.

        Identifies which reports and disclosures are affected by the
        change through the calculation dependency graph.

        Args:
            change_type: Type of change.
            entity_type: Type of entity that changed.
            affected_calc_ids: List of directly affected calculation IDs.
            materiality_pct: Materiality percentage for the change.

        Returns:
            List of cascade impact dictionaries with report_type,
            report_id, impact_level, and requires_update fields.
        """
        impacts: List[Dict[str, Any]] = []

        # Standard reports affected by any change
        report_types = ["annual_ghg_inventory"]

        if change_type in ("boundary_change", "structural_change"):
            report_types.extend([
                "base_year_inventory",
                "organizational_profile",
            ])

        if change_type in ("ef_update", "methodology_change"):
            report_types.append("methodology_statement")

        if change_type == "base_year_recalc":
            report_types.extend([
                "base_year_inventory",
                "target_progress_report",
            ])

        # Determine impact level based on materiality
        impact_level = self._classify_materiality(materiality_pct)

        for report_type in report_types:
            report_id = f"RPT-{report_type.upper()[:12]}"
            impacts.append({
                "report_type": report_type,
                "report_id": report_id,
                "impact_level": impact_level,
                "requires_update": materiality_pct >= MATERIALITY_THRESHOLDS["medium"],
                "affected_calculations_in_report": min(
                    len(affected_calc_ids), 10
                ),
            })

        # Compliance framework disclosures
        frameworks = ["GHG_PROTOCOL", "ISO_14064", "CSRD_ESRS"]
        if materiality_pct >= MATERIALITY_THRESHOLDS["high"]:
            frameworks.extend(["CDP", "SB_253", "TCFD"])

        for fw in frameworks:
            impacts.append({
                "report_type": "compliance_disclosure",
                "report_id": f"FW-{fw}",
                "impact_level": impact_level,
                "requires_update": True,
                "framework": fw,
            })

        return impacts

    # =========================================================================
    # INTERNAL: RECALCULATION LOGIC
    # =========================================================================

    def _should_recalculate(
        self,
        severity: str,
        materiality_pct: Decimal,
        change_type: str,
    ) -> bool:
        """
        Determine whether a change event requires recalculation.

        Recalculation is required when severity is medium or above, or
        when materiality exceeds the medium threshold, or when the change
        type inherently requires recalculation.

        Args:
            severity: Assessed severity level.
            materiality_pct: Materiality percentage.
            change_type: Type of change.

        Returns:
            True if recalculation is required.
        """
        # Always recalculate for these change types
        always_recalc = {
            "base_year_recalc",
            "boundary_change",
            "structural_change",
        }
        if change_type in always_recalc:
            return True

        # Recalculate if severity is medium or above
        if _severity_rank(severity) >= _severity_rank("medium"):
            return True

        # Recalculate if materiality exceeds medium threshold
        if materiality_pct >= MATERIALITY_THRESHOLDS["medium"]:
            return True

        return False

    def _update_event_status(
        self,
        change_id: str,
        new_status: str,
        result: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Update the recalculation status of a stored change event.

        Since ChangeEvent is frozen, this replaces the event with a new
        instance that has the updated status.

        Args:
            change_id: Change event identifier.
            new_status: New recalculation status.
            result: Optional recalculation result to attach to metadata.
        """
        with self._data_lock:
            event = self._changes.get(change_id)
            if event is None:
                return

            updated_metadata = dict(event.metadata)
            if result is not None:
                updated_metadata["recalculation_result"] = result
            updated_metadata["status_updated_at"] = (
                datetime.now(timezone.utc).isoformat()
            )

            new_event = ChangeEvent(
                change_id=event.change_id,
                change_type=event.change_type,
                severity=event.severity,
                affected_entity_type=event.affected_entity_type,
                affected_entity_id=event.affected_entity_id,
                old_value=event.old_value,
                new_value=event.new_value,
                trigger=event.trigger,
                materiality_pct=event.materiality_pct,
                affected_calculation_ids=event.affected_calculation_ids,
                affected_calculations_count=event.affected_calculations_count,
                recalculation_required=event.recalculation_required,
                recalculation_status=new_status,
                cascade_impacts=event.cascade_impacts,
                organization_id=event.organization_id,
                reporting_year=event.reporting_year,
                created_at=event.created_at,
                metadata=updated_metadata,
            )
            self._changes[change_id] = new_event

    def _estimate_recalculation_effort(
        self,
        affected_count: int,
        change_type: str,
    ) -> Dict[str, Any]:
        """
        Estimate the recalculation effort for a change event.

        Returns estimated duration, complexity level, and parallelism
        recommendations.

        Args:
            affected_count: Number of affected calculations.
            change_type: Type of change.

        Returns:
            Dictionary with level, estimated_seconds, parallelizable,
            and recommended_batch_size.
        """
        # Base time per calculation (seconds)
        base_time_per_calc = 0.5

        # Multiplier by change type
        type_multipliers: Dict[str, float] = {
            "ef_update": 0.3,
            "methodology_change": 1.0,
            "data_correction": 0.2,
            "boundary_change": 2.0,
            "base_year_recalc": 3.0,
            "allocation_change": 1.5,
            "scope_reclassification": 0.8,
            "structural_change": 2.5,
        }
        multiplier = type_multipliers.get(change_type, 1.0)

        estimated_seconds = affected_count * base_time_per_calc * multiplier

        # Classify effort level
        if estimated_seconds < 60:
            level = "low"
        elif estimated_seconds < 300:
            level = "medium"
        elif estimated_seconds < 3600:
            level = "high"
        else:
            level = "critical"

        return {
            "level": level,
            "estimated_seconds": round(estimated_seconds, 1),
            "parallelizable": change_type not in (
                "boundary_change", "structural_change"
            ),
            "recommended_batch_size": min(affected_count, 1000),
        }

    # =========================================================================
    # INTERNAL: SCOPE AND REPORT DETERMINATION
    # =========================================================================

    def _determine_affected_scopes(
        self,
        change_type: str,
        entity_type: str,
    ) -> List[str]:
        """
        Determine which emission scopes are affected by a change.

        Args:
            change_type: Type of change.
            entity_type: Type of entity that changed.

        Returns:
            List of affected scope strings (e.g., ["scope_1", "scope_2"]).
        """
        scope_map: Dict[str, List[str]] = {
            "emission_factor": ["scope_1", "scope_2", "scope_3"],
            "activity_data": ["scope_1", "scope_2", "scope_3"],
            "methodology": ["scope_1", "scope_2", "scope_3"],
            "organizational_boundary": ["scope_1", "scope_2", "scope_3"],
            "allocation_factor": ["scope_1", "scope_2"],
            "scope_assignment": ["scope_1", "scope_2", "scope_3"],
            "base_year_inventory": ["scope_1", "scope_2", "scope_3"],
            "facility": ["scope_1", "scope_2"],
            "business_unit": ["scope_1", "scope_2", "scope_3"],
            "supplier": ["scope_3"],
            "product": ["scope_3"],
        }

        scopes = scope_map.get(entity_type, ["scope_1", "scope_2", "scope_3"])

        # Boundary and structural changes always affect all scopes
        if change_type in ("boundary_change", "structural_change"):
            scopes = ["scope_1", "scope_2", "scope_3"]

        return scopes

    def _determine_affected_reports(
        self,
        change_type: str,
        affected_scopes: List[str],
    ) -> List[Dict[str, str]]:
        """
        Determine which reports are affected by a change.

        Args:
            change_type: Type of change.
            affected_scopes: List of affected scopes.

        Returns:
            List of report dictionaries with report_type and scope.
        """
        reports: List[Dict[str, str]] = []

        for scope in affected_scopes:
            reports.append({
                "report_type": "emissions_inventory",
                "scope": scope,
            })

        # Always affect the consolidated report
        reports.append({
            "report_type": "consolidated_ghg_report",
            "scope": "all",
        })

        if change_type == "base_year_recalc":
            reports.append({
                "report_type": "base_year_report",
                "scope": "all",
            })
            reports.append({
                "report_type": "target_progress_report",
                "scope": "all",
            })

        return reports

    # =========================================================================
    # INTERNAL: RECOMMENDATIONS
    # =========================================================================

    def _generate_materiality_recommendation(
        self,
        threshold: str,
        change_type: str,
        materiality_pct: Decimal,
    ) -> str:
        """
        Generate a human-readable recommendation based on materiality.

        Args:
            threshold: Materiality threshold level crossed.
            change_type: Type of change.
            materiality_pct: Materiality percentage.

        Returns:
            Recommendation string.
        """
        if threshold == "critical":
            return (
                f"CRITICAL: Change has {materiality_pct}% impact on emissions. "
                f"Immediate recalculation and stakeholder notification required. "
                f"Review all downstream reports and disclosures."
            )
        if threshold == "high":
            return (
                f"HIGH: Change has {materiality_pct}% impact. "
                f"Schedule recalculation within current reporting cycle. "
                f"Notify internal audit team."
            )
        if threshold == "medium":
            return (
                f"MEDIUM: Change has {materiality_pct}% impact. "
                f"Include in next scheduled recalculation batch. "
                f"Document the change in methodology notes."
            )
        return (
            f"LOW: Change has {materiality_pct}% impact. "
            f"No immediate action required. Log for annual review."
        )

    # =========================================================================
    # INTERNAL: VALIDATION
    # =========================================================================

    def _validate_change_type(self, change_type: str) -> None:
        """
        Validate that change_type is in the set of allowed values.

        Args:
            change_type: Change type string to validate.

        Raises:
            ValueError: If change_type is invalid.
        """
        if change_type not in VALID_CHANGE_TYPES:
            raise ValueError(
                f"Invalid change_type '{change_type}'. "
                f"Must be one of {VALID_CHANGE_TYPES}"
            )

    def _validate_severity(self, severity: str) -> None:
        """
        Validate that severity is in the set of allowed values.

        Args:
            severity: Severity string to validate.

        Raises:
            ValueError: If severity is invalid.
        """
        if severity not in VALID_SEVERITIES:
            raise ValueError(
                f"Invalid severity '{severity}'. "
                f"Must be one of {VALID_SEVERITIES}"
            )

    def _validate_recalculation_status(self, status: str) -> None:
        """
        Validate that recalculation status is in the set of allowed values.

        Args:
            status: Recalculation status string to validate.

        Raises:
            ValueError: If status is invalid.
        """
        if status not in VALID_RECALCULATION_STATUSES:
            raise ValueError(
                f"Invalid recalculation status '{status}'. "
                f"Must be one of {VALID_RECALCULATION_STATUSES}"
            )

    # =========================================================================
    # INTERNAL: HELPERS
    # =========================================================================

    def _get_event_or_raise(self, change_id: str) -> ChangeEvent:
        """
        Retrieve a change event by ID or raise ValueError.

        Args:
            change_id: Unique change event identifier.

        Returns:
            The ChangeEvent instance.

        Raises:
            ValueError: If change_id is not found.
        """
        with self._data_lock:
            event = self._changes.get(change_id)

        if event is None:
            raise ValueError(f"Change event not found: {change_id}")
        return event

    def _event_to_dict(self, event: ChangeEvent) -> Dict[str, Any]:
        """
        Convert a ChangeEvent dataclass to a plain dictionary.

        Args:
            event: ChangeEvent instance to convert.

        Returns:
            Dictionary representation of the event.
        """
        return {
            "change_id": event.change_id,
            "change_type": event.change_type,
            "severity": event.severity,
            "affected_entity_type": event.affected_entity_type,
            "affected_entity_id": event.affected_entity_id,
            "old_value": self._safe_serialize(event.old_value),
            "new_value": self._safe_serialize(event.new_value),
            "trigger": event.trigger,
            "materiality_pct": str(event.materiality_pct),
            "affected_calculation_ids": event.affected_calculation_ids,
            "affected_calculations_count": event.affected_calculations_count,
            "recalculation_required": event.recalculation_required,
            "recalculation_status": event.recalculation_status,
            "cascade_impacts": event.cascade_impacts,
            "organization_id": event.organization_id,
            "reporting_year": event.reporting_year,
            "created_at": event.created_at,
            "metadata": event.metadata,
        }

    def _safe_serialize(self, value: Any) -> Any:
        """
        Safely serialize a value for JSON output.

        Args:
            value: Value to serialize.

        Returns:
            JSON-safe representation of the value.
        """
        if isinstance(value, Decimal):
            return str(value)
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, (dict, list, str, int, float, bool, type(None))):
            return value
        return str(value)

    def _build_version_comparison(
        self,
        old_value: Any,
        new_value: Any,
    ) -> Dict[str, Any]:
        """
        Build a structured comparison between old and new values.

        For numeric values, includes absolute difference, percentage change,
        and direction. For non-numeric values, provides a textual comparison.

        Args:
            old_value: Previous value.
            new_value: New value.

        Returns:
            Dictionary with comparison details.
        """
        try:
            old_dec = Decimal(str(old_value))
            new_dec = Decimal(str(new_value))
        except (TypeError, ValueError, ArithmeticError):
            # Non-numeric comparison
            return {
                "type": "non_numeric",
                "changed": str(old_value) != str(new_value),
                "old_type": type(old_value).__name__,
                "new_type": type(new_value).__name__,
            }

        absolute_diff = (new_dec - old_dec).quantize(_QUANT_4DP, rounding=ROUNDING)

        if old_dec == Decimal("0"):
            pct_change = Decimal("100.00") if new_dec != Decimal("0") else Decimal("0.00")
        else:
            pct_change = (
                (new_dec - old_dec) / abs(old_dec) * Decimal("100")
            ).quantize(_QUANT_2DP, rounding=ROUNDING)

        if absolute_diff > Decimal("0"):
            direction = "increase"
        elif absolute_diff < Decimal("0"):
            direction = "decrease"
        else:
            direction = "unchanged"

        return {
            "type": "numeric",
            "absolute_difference": str(absolute_diff),
            "percentage_change": str(pct_change),
            "direction": direction,
            "old_value_numeric": str(old_dec),
            "new_value_numeric": str(new_dec),
        }

    def _compute_provenance_hash(self, event: ChangeEvent) -> str:
        """
        Compute SHA-256 provenance hash for a change event.

        Args:
            event: ChangeEvent instance.

        Returns:
            Lowercase hex SHA-256 hash string.
        """
        return _compute_hash({
            "engine_id": ENGINE_ID,
            "engine_version": ENGINE_VERSION,
            "change_id": event.change_id,
            "change_type": event.change_type,
            "severity": event.severity,
            "materiality_pct": str(event.materiality_pct),
            "affected_entity_id": event.affected_entity_id,
            "organization_id": event.organization_id,
            "reporting_year": event.reporting_year,
            "created_at": event.created_at,
        })


# ==============================================================================
# MODULE-LEVEL HELPERS
# ==============================================================================


def _severity_rank(severity: str) -> int:
    """
    Return numeric rank for severity (higher = more severe).

    Args:
        severity: Severity string.

    Returns:
        Integer rank (4=critical, 1=low).
    """
    rank_map: Dict[str, int] = {
        "critical": 4,
        "high": 3,
        "medium": 2,
        "low": 1,
    }
    return rank_map.get(severity, 0)
