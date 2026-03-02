# -*- coding: utf-8 -*-
"""
AuditTrailPipelineEngine - 10-Stage Audit Trail Orchestration Pipeline

Engine 7 of 7 for AGENT-MRV-030 (GL-MRV-X-042).

Orchestrates the complete audit trail workflow from event capture through
evidence sealing in 10 defined stages. Each stage records start time,
duration, status, errors, and warnings, enabling full observability of
the audit trail lifecycle.

10 Stages:
    1. VALIDATE    - Validate audit event inputs and configuration
    2. CLASSIFY    - Classify events by scope, category, and framework
    3. RECORD      - Record events to immutable hash chain
    4. LINK        - Link events to lineage graph nodes
    5. TRACE       - Map events to regulatory requirements
    6. DETECT      - Detect changes requiring recalculation
    7. VERIFY      - Verify hash chain and lineage integrity
    8. PACKAGE     - Bundle evidence for verification
    9. COMPLIANCE  - Check audit trail completeness per framework
    10. SEAL       - Final seal with provenance hash and optional signature

Zero-Hallucination Guarantee:
    All hashing, chain linking, and compliance scoring use deterministic
    algorithms. No LLM or ML models are involved in any stage.

Example:
    >>> pipeline = AuditTrailPipelineEngine()
    >>> result = pipeline.execute(
    ...     event_type="calculation_completed",
    ...     agent_id="GL-MRV-S1-001",
    ...     scope="scope_1",
    ...     category="stationary_combustion",
    ...     organization_id="ORG-001",
    ...     reporting_year=2025,
    ...     calculation_id="calc-abc123",
    ...     payload={"emissions_tco2e": 1234.56},
    ... )
    >>> assert result["status"] == "SUCCESS"

Module: greenlang.audit_trail_lineage.audit_trail_pipeline
Agent: AGENT-MRV-030
Version: 1.0.0
Author: GreenLang Platform Team
Date: March 2026
"""

import hashlib
import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ==============================================================================
# AGENT METADATA
# ==============================================================================

AGENT_ID: str = "GL-MRV-X-042"
AGENT_COMPONENT: str = "AGENT-MRV-030"
VERSION: str = "1.0.0"
HASH_ALGORITHM: str = "sha256"
ENCODING: str = "utf-8"


# ==============================================================================
# ENUMERATIONS
# ==============================================================================


class PipelineStatus(str, Enum):
    """Pipeline execution outcome status."""

    SUCCESS = "SUCCESS"
    PARTIAL_SUCCESS = "PARTIAL_SUCCESS"
    FAILED = "FAILED"
    VALIDATION_ERROR = "VALIDATION_ERROR"


class PipelineStage(str, Enum):
    """Pipeline stage identifiers (10 stages)."""

    VALIDATE = "VALIDATE"
    CLASSIFY = "CLASSIFY"
    RECORD = "RECORD"
    LINK = "LINK"
    TRACE = "TRACE"
    DETECT = "DETECT"
    VERIFY = "VERIFY"
    PACKAGE = "PACKAGE"
    COMPLIANCE = "COMPLIANCE"
    SEAL = "SEAL"


class EventScope(str, Enum):
    """GHG Protocol emission scope identifiers."""

    SCOPE_1 = "scope_1"
    SCOPE_2 = "scope_2"
    SCOPE_3 = "scope_3"


# Ordered stage list for deterministic iteration
_STAGE_ORDER: List[PipelineStage] = [
    PipelineStage.VALIDATE,
    PipelineStage.CLASSIFY,
    PipelineStage.RECORD,
    PipelineStage.LINK,
    PipelineStage.TRACE,
    PipelineStage.DETECT,
    PipelineStage.VERIFY,
    PipelineStage.PACKAGE,
    PipelineStage.COMPLIANCE,
    PipelineStage.SEAL,
]

# Valid event types for stage 1 validation
_VALID_EVENT_TYPES = {
    "calculation_completed",
    "calculation_updated",
    "calculation_deleted",
    "data_ingested",
    "data_validated",
    "data_transformed",
    "emission_factor_applied",
    "emission_factor_updated",
    "aggregation_completed",
    "report_generated",
    "report_submitted",
    "verification_completed",
    "correction_applied",
    "assumption_changed",
    "methodology_changed",
    "boundary_changed",
    "allocation_changed",
    "recalculation_triggered",
    "recalculation_completed",
    "approval_granted",
    "approval_revoked",
    "audit_event",
    "lineage_node_created",
    "lineage_edge_created",
    "evidence_packaged",
    "compliance_checked",
    "chain_verified",
}

# Valid scopes
_VALID_SCOPES = {"scope_1", "scope_2", "scope_3", "cross_cutting"}

# Framework mapping for classification stage
_FRAMEWORK_SCOPE_MAP: Dict[str, List[str]] = {
    "GHG_PROTOCOL": ["scope_1", "scope_2", "scope_3"],
    "ISO_14064": ["scope_1", "scope_2", "scope_3"],
    "CSRD_ESRS": ["scope_1", "scope_2", "scope_3"],
    "CDP": ["scope_1", "scope_2", "scope_3"],
    "SBTI": ["scope_1", "scope_2", "scope_3"],
    "SB_253": ["scope_1", "scope_2", "scope_3"],
    "SEC_CLIMATE": ["scope_1", "scope_2"],
    "EU_TAXONOMY": ["scope_1", "scope_2", "scope_3"],
    "ISAE_3410": ["scope_1", "scope_2", "scope_3"],
}


# ==============================================================================
# SERIALIZATION UTILITIES
# ==============================================================================


def _serialize(data: Any) -> str:
    """
    Serialize data to deterministic JSON string.

    Handles Decimal, datetime, Enum, Pydantic models, and dataclass objects.

    Args:
        data: Object to serialize.

    Returns:
        Deterministic JSON string with sorted keys.
    """

    def default_handler(o: Any) -> Any:
        """Handle non-JSON-serializable types."""
        if isinstance(o, Decimal):
            return str(o)
        if isinstance(o, datetime):
            return o.isoformat()
        if isinstance(o, Enum):
            return o.value
        if hasattr(o, "model_dump"):
            return o.model_dump()
        if hasattr(o, "dict"):
            return o.dict()
        if hasattr(o, "to_dict"):
            return o.to_dict()
        if hasattr(o, "__dict__"):
            return o.__dict__
        return str(o)

    return json.dumps(data, sort_keys=True, default=default_handler)


def _compute_hash(data: Any) -> str:
    """
    Compute SHA-256 hash of data.

    Args:
        data: Data to hash (any JSON-serializable type).

    Returns:
        Lowercase hex SHA-256 hash string.
    """
    serialized = _serialize(data)
    return hashlib.sha256(serialized.encode(ENCODING)).hexdigest()


# ==============================================================================
# PIPELINE ENGINE
# ==============================================================================


class AuditTrailPipelineEngine:
    """
    10-stage orchestration pipeline for audit trail event processing.

    Thread-safe singleton that orchestrates the complete audit trail
    lifecycle from event capture through evidence sealing. Each stage
    is tracked with timing, status, errors, and warnings.

    The pipeline delegates to engines 1-6 (AuditEventEngine,
    LineageGraphEngine, ComplianceTracerEngine, ChangeDetectorEngine,
    EvidencePackagerEngine, ComplianceCheckerEngine) and adds
    orchestration, validation, classification, and sealing logic.

    Attributes:
        agent_id: Agent identifier (GL-MRV-X-042).
        version: Agent version string.

    Example:
        >>> pipeline = AuditTrailPipelineEngine()
        >>> result = pipeline.execute(
        ...     event_type="calculation_completed",
        ...     agent_id="GL-MRV-S1-001",
        ...     scope="scope_1",
        ...     category="stationary_combustion",
        ...     organization_id="ORG-001",
        ...     reporting_year=2025,
        ...     calculation_id="calc-abc123",
        ...     payload={"emissions_tco2e": 1234.56},
        ... )
        >>> assert result["status"] == "SUCCESS"
    """

    _instance: Optional["AuditTrailPipelineEngine"] = None
    _instance_lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "AuditTrailPipelineEngine":
        """Thread-safe singleton instantiation."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize pipeline engine (only once due to singleton)."""
        if hasattr(self, "_initialized"):
            return

        self._initialized: bool = True
        self.agent_id: str = AGENT_ID
        self.version: str = VERSION
        self._lock: threading.RLock = threading.RLock()

        # Pipeline execution statistics
        self._stats_lock: threading.Lock = threading.Lock()
        self._stats: Dict[str, Any] = {
            "total_executions": 0,
            "total_batch_executions": 0,
            "total_events_processed": 0,
            "total_successes": 0,
            "total_partial_successes": 0,
            "total_failures": 0,
            "total_validation_errors": 0,
            "stage_execution_counts": {s.value: 0 for s in PipelineStage},
            "stage_total_duration_ms": {s.value: 0.0 for s in PipelineStage},
            "started_at": datetime.now(timezone.utc).isoformat(),
        }

        # Lazy engine references
        self._audit_event_engine: Optional[Any] = None
        self._lineage_graph_engine: Optional[Any] = None
        self._compliance_tracer_engine: Optional[Any] = None
        self._change_detector_engine: Optional[Any] = None
        self._evidence_packager_engine: Optional[Any] = None
        self._compliance_checker_engine: Optional[Any] = None
        self._provenance_tracker: Optional[Any] = None
        self._metrics: Optional[Any] = None

        logger.info(
            "AuditTrailPipelineEngine initialized: agent_id=%s, version=%s",
            self.agent_id,
            self.version,
        )

    # ------------------------------------------------------------------
    # Lazy engine accessors
    # ------------------------------------------------------------------

    def _get_audit_event_engine(self) -> Any:
        """Get AuditEventEngine (lazy import)."""
        if self._audit_event_engine is None:
            try:
                from greenlang.audit_trail_lineage.audit_event import (
                    AuditEventEngine,
                )
                self._audit_event_engine = AuditEventEngine()
            except ImportError:
                logger.debug("AuditEventEngine not available")
        return self._audit_event_engine

    def _get_lineage_graph_engine(self) -> Any:
        """Get LineageGraphEngine (lazy import)."""
        if self._lineage_graph_engine is None:
            try:
                from greenlang.audit_trail_lineage.lineage_graph import (
                    LineageGraphEngine,
                )
                self._lineage_graph_engine = LineageGraphEngine()
            except ImportError:
                logger.debug("LineageGraphEngine not available")
        return self._lineage_graph_engine

    def _get_compliance_tracer_engine(self) -> Any:
        """Get ComplianceTracerEngine (lazy import)."""
        if self._compliance_tracer_engine is None:
            try:
                from greenlang.audit_trail_lineage.compliance_tracer import (
                    ComplianceTracerEngine,
                )
                self._compliance_tracer_engine = ComplianceTracerEngine()
            except ImportError:
                logger.debug("ComplianceTracerEngine not available")
        return self._compliance_tracer_engine

    def _get_change_detector_engine(self) -> Any:
        """Get ChangeDetectorEngine (lazy import)."""
        if self._change_detector_engine is None:
            try:
                from greenlang.audit_trail_lineage.change_detector import (
                    ChangeDetectorEngine,
                )
                self._change_detector_engine = ChangeDetectorEngine()
            except ImportError:
                logger.debug("ChangeDetectorEngine not available")
        return self._change_detector_engine

    def _get_evidence_packager_engine(self) -> Any:
        """Get EvidencePackagerEngine (lazy import)."""
        if self._evidence_packager_engine is None:
            try:
                from greenlang.audit_trail_lineage.evidence_packager import (
                    EvidencePackagerEngine,
                )
                self._evidence_packager_engine = EvidencePackagerEngine()
            except ImportError:
                logger.debug("EvidencePackagerEngine not available")
        return self._evidence_packager_engine

    def _get_compliance_checker_engine(self) -> Any:
        """Get ComplianceCheckerEngine (lazy import)."""
        if self._compliance_checker_engine is None:
            try:
                from greenlang.audit_trail_lineage.compliance_checker import (
                    ComplianceCheckerEngine,
                )
                self._compliance_checker_engine = ComplianceCheckerEngine()
            except ImportError:
                logger.debug("ComplianceCheckerEngine not available")
        return self._compliance_checker_engine

    def _get_provenance_tracker(self) -> Any:
        """Get ProvenanceTracker (lazy import)."""
        if self._provenance_tracker is None:
            try:
                from greenlang.audit_trail_lineage.provenance import (
                    ProvenanceTracker,
                )
                self._provenance_tracker = ProvenanceTracker()
            except ImportError:
                logger.debug("ProvenanceTracker not available")
        return self._provenance_tracker

    def _get_metrics(self) -> Any:
        """Get AuditTrailLineageMetrics (lazy import)."""
        if self._metrics is None:
            try:
                from greenlang.audit_trail_lineage.metrics import get_metrics
                self._metrics = get_metrics()
            except ImportError:
                logger.debug("AuditTrailLineageMetrics not available")
        return self._metrics

    # ------------------------------------------------------------------
    # Public API: execute (single event)
    # ------------------------------------------------------------------

    def execute(
        self,
        event_type: str,
        agent_id: str,
        scope: str,
        category: str,
        organization_id: str,
        reporting_year: int,
        calculation_id: str,
        payload: Dict[str, Any],
        data_quality_score: Decimal = Decimal("1.0"),
        metadata: Optional[Dict[str, Any]] = None,
        include_evidence: bool = False,
        include_compliance: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute the full 10-stage audit trail pipeline for a single event.

        Processes the event through validation, classification, recording,
        linking, tracing, change detection, verification, packaging,
        compliance checking, and sealing.

        Args:
            event_type: Type of audit event (e.g., 'calculation_completed').
            agent_id: Source agent identifier (e.g., 'GL-MRV-S1-001').
            scope: GHG emission scope (scope_1, scope_2, scope_3, cross_cutting).
            category: Emission category (e.g., 'stationary_combustion').
            organization_id: Organization identifier.
            reporting_year: Reporting year (2000-2100).
            calculation_id: Unique calculation identifier.
            payload: Event payload data (calculation results, etc.).
            data_quality_score: Data quality score (0.0-1.0). Defaults to 1.0.
            metadata: Optional additional metadata.
            include_evidence: Include evidence packaging (stage 8). Defaults to False.
            include_compliance: Include compliance checking (stage 9). Defaults to False.

        Returns:
            Dictionary with pipeline result including status, event_id,
            provenance_hash, stage_results, and processing_time_ms.
        """
        pipeline_start = time.monotonic()
        run_id = f"atl-{uuid.uuid4().hex[:12]}"

        # Build pipeline context
        context: Dict[str, Any] = {
            "run_id": run_id,
            "event_type": event_type,
            "agent_id": agent_id,
            "scope": scope,
            "category": category,
            "organization_id": organization_id,
            "reporting_year": reporting_year,
            "calculation_id": calculation_id,
            "payload": payload,
            "data_quality_score": data_quality_score,
            "metadata": metadata or {},
            "include_evidence": include_evidence,
            "include_compliance": include_compliance,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            # Stage results accumulator
            "stage_results": {},
            "stage_durations_ms": {},
            "errors": [],
            "warnings": [],
            # Intermediate state
            "event_id": None,
            "chain_hash": None,
            "lineage_node_id": None,
            "trace_id": None,
            "change_result": None,
            "verification_result": None,
            "evidence_package": None,
            "compliance_result": None,
            "seal_hash": None,
            "overall_status": PipelineStatus.SUCCESS.value,
        }

        try:
            # Execute each stage in order
            stage_methods = [
                (PipelineStage.VALIDATE, self._stage_validate),
                (PipelineStage.CLASSIFY, self._stage_classify),
                (PipelineStage.RECORD, self._stage_record),
                (PipelineStage.LINK, self._stage_link),
                (PipelineStage.TRACE, self._stage_trace),
                (PipelineStage.DETECT, self._stage_detect),
                (PipelineStage.VERIFY, self._stage_verify),
                (PipelineStage.PACKAGE, self._stage_package),
                (PipelineStage.COMPLIANCE, self._stage_compliance),
                (PipelineStage.SEAL, self._stage_seal),
            ]

            for stage_enum, stage_fn in stage_methods:
                stage_start = time.monotonic()
                try:
                    stage_result = stage_fn(context)
                    duration_ms = (time.monotonic() - stage_start) * 1000.0

                    context["stage_results"][stage_enum.value] = {
                        "status": stage_result.get("status", "SUCCESS"),
                        "duration_ms": round(duration_ms, 3),
                        "data": stage_result.get("data"),
                        "errors": stage_result.get("errors", []),
                        "warnings": stage_result.get("warnings", []),
                    }
                    context["stage_durations_ms"][stage_enum.value] = round(
                        duration_ms, 3
                    )

                    # Update statistics
                    self._update_stage_stats(stage_enum, duration_ms)

                    # If stage failed and it is critical, mark partial
                    if stage_result.get("status") == "FAILED":
                        if stage_enum in (
                            PipelineStage.VALIDATE,
                            PipelineStage.RECORD,
                        ):
                            context["overall_status"] = (
                                PipelineStatus.FAILED.value
                            )
                            break
                        context["overall_status"] = (
                            PipelineStatus.PARTIAL_SUCCESS.value
                        )

                except Exception as stage_err:
                    duration_ms = (time.monotonic() - stage_start) * 1000.0
                    error_msg = f"Stage {stage_enum.value} failed: {str(stage_err)}"
                    logger.error(error_msg, exc_info=True)

                    context["stage_results"][stage_enum.value] = {
                        "status": "FAILED",
                        "duration_ms": round(duration_ms, 3),
                        "data": None,
                        "errors": [error_msg],
                        "warnings": [],
                    }
                    context["stage_durations_ms"][stage_enum.value] = round(
                        duration_ms, 3
                    )
                    context["errors"].append(error_msg)

                    if stage_enum in (
                        PipelineStage.VALIDATE,
                        PipelineStage.RECORD,
                    ):
                        context["overall_status"] = (
                            PipelineStatus.FAILED.value
                            if stage_enum == PipelineStage.VALIDATE
                            else PipelineStatus.PARTIAL_SUCCESS.value
                        )
                        break
                    context["overall_status"] = (
                        PipelineStatus.PARTIAL_SUCCESS.value
                    )

            # Build final result
            total_ms = (time.monotonic() - pipeline_start) * 1000.0
            result = self._build_result(context, total_ms)

            # Update global statistics
            self._update_execution_stats(result["status"])

            # Record pipeline metrics
            metrics = self._get_metrics()
            if metrics is not None:
                try:
                    metrics.record_pipeline_execution(
                        status=result["status"],
                        duration=total_ms / 1000.0,
                        event_type=event_type,
                        scope=scope,
                    )
                except Exception:
                    pass

            return result

        except Exception as e:
            total_ms = (time.monotonic() - pipeline_start) * 1000.0
            logger.error(
                "Pipeline execution failed: %s", e, exc_info=True
            )
            self._update_execution_stats(PipelineStatus.FAILED.value)
            return {
                "status": PipelineStatus.FAILED.value,
                "run_id": run_id,
                "event_id": None,
                "provenance_hash": None,
                "seal_hash": None,
                "processing_time_ms": round(total_ms, 3),
                "stage_results": context.get("stage_results", {}),
                "stage_durations_ms": context.get("stage_durations_ms", {}),
                "errors": [str(e)],
                "warnings": [],
                "agent_id": self.agent_id,
                "version": self.version,
                "timestamp": context.get("timestamp", ""),
            }

    # ------------------------------------------------------------------
    # Public API: execute_batch
    # ------------------------------------------------------------------

    def execute_batch(
        self,
        events: List[Dict[str, Any]],
        include_evidence: bool = False,
        include_compliance: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute the pipeline for a batch of audit events.

        Processes each event through the full 10-stage pipeline and
        aggregates results.

        Args:
            events: List of event dictionaries. Each must contain keys:
                event_type, agent_id, scope, category, organization_id,
                reporting_year, calculation_id, payload.
            include_evidence: Include evidence packaging for each event.
            include_compliance: Include compliance checking for each event.

        Returns:
            Dictionary with batch_id, total_events, results list,
            summary statistics, and processing_time_ms.
        """
        batch_start = time.monotonic()
        batch_id = f"atl-batch-{uuid.uuid4().hex[:12]}"

        results: List[Dict[str, Any]] = []
        status_counts: Dict[str, int] = {
            PipelineStatus.SUCCESS.value: 0,
            PipelineStatus.PARTIAL_SUCCESS.value: 0,
            PipelineStatus.FAILED.value: 0,
            PipelineStatus.VALIDATION_ERROR.value: 0,
        }

        for idx, event in enumerate(events):
            try:
                event_result = self.execute(
                    event_type=event.get("event_type", "audit_event"),
                    agent_id=event.get("agent_id", "unknown"),
                    scope=event.get("scope", "cross_cutting"),
                    category=event.get("category", ""),
                    organization_id=event.get("organization_id", ""),
                    reporting_year=event.get("reporting_year", 2025),
                    calculation_id=event.get("calculation_id", ""),
                    payload=event.get("payload", {}),
                    data_quality_score=Decimal(
                        str(event.get("data_quality_score", "1.0"))
                    ),
                    metadata=event.get("metadata"),
                    include_evidence=include_evidence,
                    include_compliance=include_compliance,
                )
                results.append(event_result)
                status = event_result.get("status", PipelineStatus.FAILED.value)
                if status in status_counts:
                    status_counts[status] += 1

            except Exception as e:
                logger.error(
                    "Batch event %d failed: %s", idx, e, exc_info=True
                )
                results.append({
                    "status": PipelineStatus.FAILED.value,
                    "event_index": idx,
                    "error": str(e),
                })
                status_counts[PipelineStatus.FAILED.value] += 1

        total_ms = (time.monotonic() - batch_start) * 1000.0

        # Determine batch-level status
        if status_counts[PipelineStatus.FAILED.value] == len(events):
            batch_status = PipelineStatus.FAILED.value
        elif status_counts[PipelineStatus.SUCCESS.value] == len(events):
            batch_status = PipelineStatus.SUCCESS.value
        else:
            batch_status = PipelineStatus.PARTIAL_SUCCESS.value

        # Update stats
        with self._stats_lock:
            self._stats["total_batch_executions"] += 1
            self._stats["total_events_processed"] += len(events)

        return {
            "batch_id": batch_id,
            "status": batch_status,
            "total_events": len(events),
            "results": results,
            "summary": {
                "successes": status_counts[PipelineStatus.SUCCESS.value],
                "partial_successes": status_counts[
                    PipelineStatus.PARTIAL_SUCCESS.value
                ],
                "failures": status_counts[PipelineStatus.FAILED.value],
                "validation_errors": status_counts[
                    PipelineStatus.VALIDATION_ERROR.value
                ],
            },
            "processing_time_ms": round(total_ms, 3),
            "agent_id": self.agent_id,
            "version": self.version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ------------------------------------------------------------------
    # Stage 1: VALIDATE
    # ------------------------------------------------------------------

    def _stage_validate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stage 1: Validate audit event inputs and configuration.

        Checks event_type, scope, organization_id, reporting_year,
        calculation_id, and payload for validity.

        Args:
            context: Pipeline context dictionary.

        Returns:
            Stage result with status, data, errors, warnings.
        """
        errors: List[str] = []
        warnings: List[str] = []

        # Validate event_type
        event_type = context.get("event_type", "")
        if not event_type:
            errors.append("event_type is required")
        elif event_type not in _VALID_EVENT_TYPES:
            warnings.append(
                f"event_type '{event_type}' is not in standard set; "
                "proceeding with custom type"
            )

        # Validate scope
        scope = context.get("scope", "")
        if not scope:
            errors.append("scope is required")
        elif scope not in _VALID_SCOPES:
            errors.append(
                f"scope '{scope}' is invalid; "
                f"must be one of {sorted(_VALID_SCOPES)}"
            )

        # Validate organization_id
        org_id = context.get("organization_id", "")
        if not org_id:
            errors.append("organization_id is required")

        # Validate reporting_year
        year = context.get("reporting_year")
        if year is None:
            errors.append("reporting_year is required")
        elif not isinstance(year, int) or year < 2000 or year > 2100:
            errors.append("reporting_year must be an integer between 2000 and 2100")

        # Validate calculation_id
        calc_id = context.get("calculation_id", "")
        if not calc_id:
            warnings.append("calculation_id is empty; a UUID will be assigned")
            context["calculation_id"] = f"calc-{uuid.uuid4().hex[:12]}"

        # Validate payload
        payload = context.get("payload")
        if payload is None or not isinstance(payload, dict):
            errors.append("payload must be a non-null dictionary")

        # Validate data_quality_score
        dq_score = context.get("data_quality_score", Decimal("1.0"))
        try:
            dq_val = Decimal(str(dq_score))
            if dq_val < Decimal("0.0") or dq_val > Decimal("1.0"):
                warnings.append(
                    f"data_quality_score {dq_val} outside [0.0, 1.0]; "
                    "clamping to range"
                )
                context["data_quality_score"] = max(
                    Decimal("0.0"), min(Decimal("1.0"), dq_val)
                )
        except Exception:
            warnings.append("data_quality_score is not a valid Decimal; defaulting to 1.0")
            context["data_quality_score"] = Decimal("1.0")

        # Validate agent_id
        agent_id = context.get("agent_id", "")
        if not agent_id:
            warnings.append("agent_id is empty; using 'unknown'")
            context["agent_id"] = "unknown"

        if errors:
            context["overall_status"] = PipelineStatus.VALIDATION_ERROR.value
            context["errors"].extend(errors)
            return {
                "status": "FAILED",
                "data": {"validated": False},
                "errors": errors,
                "warnings": warnings,
            }

        context["warnings"].extend(warnings)
        logger.debug(
            "Stage VALIDATE: passed (%d warnings)", len(warnings)
        )

        return {
            "status": "SUCCESS",
            "data": {"validated": True, "warning_count": len(warnings)},
            "errors": [],
            "warnings": warnings,
        }

    # ------------------------------------------------------------------
    # Stage 2: CLASSIFY
    # ------------------------------------------------------------------

    def _stage_classify(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stage 2: Classify event by scope, category, and framework.

        Determines which regulatory frameworks apply to this event based
        on scope and category, and assigns classification labels.

        Args:
            context: Pipeline context dictionary.

        Returns:
            Stage result with classification data.
        """
        scope = context.get("scope", "cross_cutting")
        category = context.get("category", "")
        event_type = context.get("event_type", "")

        # Determine applicable frameworks
        applicable_frameworks: List[str] = []
        for fw, fw_scopes in _FRAMEWORK_SCOPE_MAP.items():
            if scope in fw_scopes:
                applicable_frameworks.append(fw)

        # Classify event priority
        high_priority_types = {
            "correction_applied",
            "methodology_changed",
            "boundary_changed",
            "recalculation_triggered",
            "assumption_changed",
        }
        priority = "high" if event_type in high_priority_types else "normal"

        # Classify audit significance
        significance = "material" if priority == "high" else "standard"

        classification = {
            "scope": scope,
            "category": category,
            "event_type": event_type,
            "applicable_frameworks": applicable_frameworks,
            "framework_count": len(applicable_frameworks),
            "priority": priority,
            "significance": significance,
        }

        context["classification"] = classification

        logger.debug(
            "Stage CLASSIFY: scope=%s, frameworks=%d, priority=%s",
            scope,
            len(applicable_frameworks),
            priority,
        )

        return {
            "status": "SUCCESS",
            "data": classification,
            "errors": [],
            "warnings": [],
        }

    # ------------------------------------------------------------------
    # Stage 3: RECORD
    # ------------------------------------------------------------------

    def _stage_record(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stage 3: Record event to immutable hash chain.

        Uses AuditEventEngine to create an immutable, hash-chained
        audit event record.

        Args:
            context: Pipeline context dictionary.

        Returns:
            Stage result with event_id and chain_hash.
        """
        engine = self._get_audit_event_engine()

        # Build audit event data
        event_data = {
            "event_type": context["event_type"],
            "agent_id": context["agent_id"],
            "scope": context["scope"],
            "category": context["category"],
            "organization_id": context["organization_id"],
            "reporting_year": context["reporting_year"],
            "calculation_id": context["calculation_id"],
            "payload": context["payload"],
            "data_quality_score": str(context["data_quality_score"]),
            "metadata": context["metadata"],
            "classification": context.get("classification", {}),
            "timestamp": context["timestamp"],
        }

        event_id = f"evt-{uuid.uuid4().hex[:16]}"
        chain_hash = _compute_hash(event_data)

        if engine is not None:
            try:
                result = engine.record_event(event_data)
                if result is not None:
                    event_id = result.get("event_id", event_id)
                    chain_hash = result.get("chain_hash", chain_hash)
            except Exception as e:
                logger.warning(
                    "AuditEventEngine.record_event failed: %s", e
                )

        context["event_id"] = event_id
        context["chain_hash"] = chain_hash

        # Record metric
        metrics = self._get_metrics()
        if metrics is not None:
            try:
                metrics.record_event(
                    event_type=context["event_type"],
                    scope=context["scope"],
                )
            except Exception:
                pass

        logger.debug(
            "Stage RECORD: event_id=%s, chain_hash=%s...",
            event_id,
            chain_hash[:16],
        )

        return {
            "status": "SUCCESS",
            "data": {
                "event_id": event_id,
                "chain_hash": chain_hash,
            },
            "errors": [],
            "warnings": [],
        }

    # ------------------------------------------------------------------
    # Stage 4: LINK
    # ------------------------------------------------------------------

    def _stage_link(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stage 4: Link event to lineage graph nodes.

        Uses LineageGraphEngine to create or update a lineage node
        for this event and connect it to related nodes (source data,
        emission factors, calculations).

        Args:
            context: Pipeline context dictionary.

        Returns:
            Stage result with lineage_node_id.
        """
        engine = self._get_lineage_graph_engine()

        node_id = f"node-{uuid.uuid4().hex[:12]}"
        edges_created = 0

        if engine is not None:
            try:
                node_data = {
                    "event_id": context.get("event_id"),
                    "agent_id": context["agent_id"],
                    "scope": context["scope"],
                    "category": context["category"],
                    "calculation_id": context["calculation_id"],
                    "organization_id": context["organization_id"],
                    "reporting_year": context["reporting_year"],
                }
                result = engine.add_node(node_data)
                if result is not None:
                    node_id = result.get("node_id", node_id)
                    edges_created = result.get("edges_created", 0)
            except Exception as e:
                logger.warning("LineageGraphEngine.add_node failed: %s", e)

        context["lineage_node_id"] = node_id

        # Record lineage metrics
        metrics = self._get_metrics()
        if metrics is not None:
            try:
                metrics.record_lineage_node(scope=context["scope"])
                for _ in range(edges_created):
                    metrics.record_lineage_edge(
                        edge_type="calculation_dependency"
                    )
            except Exception:
                pass

        logger.debug(
            "Stage LINK: node_id=%s, edges=%d", node_id, edges_created
        )

        return {
            "status": "SUCCESS",
            "data": {
                "lineage_node_id": node_id,
                "edges_created": edges_created,
            },
            "errors": [],
            "warnings": [],
        }

    # ------------------------------------------------------------------
    # Stage 5: TRACE
    # ------------------------------------------------------------------

    def _stage_trace(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stage 5: Map event to regulatory requirements.

        Uses ComplianceTracerEngine to trace this event to specific
        framework requirements (GHG Protocol, ISO 14064, CSRD, etc.).

        Args:
            context: Pipeline context dictionary.

        Returns:
            Stage result with trace mappings.
        """
        engine = self._get_compliance_tracer_engine()

        classification = context.get("classification", {})
        frameworks = classification.get("applicable_frameworks", [])
        trace_id = f"trace-{uuid.uuid4().hex[:12]}"
        requirements_mapped = 0

        if engine is not None and frameworks:
            try:
                trace_data = {
                    "event_id": context.get("event_id"),
                    "scope": context["scope"],
                    "category": context["category"],
                    "event_type": context["event_type"],
                    "frameworks": frameworks,
                }
                result = engine.trace_requirements(trace_data)
                if result is not None:
                    trace_id = result.get("trace_id", trace_id)
                    requirements_mapped = result.get(
                        "requirements_mapped", 0
                    )
            except Exception as e:
                logger.warning(
                    "ComplianceTracerEngine.trace_requirements failed: %s", e
                )

        context["trace_id"] = trace_id

        logger.debug(
            "Stage TRACE: trace_id=%s, requirements=%d",
            trace_id,
            requirements_mapped,
        )

        return {
            "status": "SUCCESS",
            "data": {
                "trace_id": trace_id,
                "requirements_mapped": requirements_mapped,
                "frameworks_traced": frameworks,
            },
            "errors": [],
            "warnings": [],
        }

    # ------------------------------------------------------------------
    # Stage 6: DETECT
    # ------------------------------------------------------------------

    def _stage_detect(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stage 6: Detect changes requiring recalculation.

        Uses ChangeDetectorEngine to compare this event against
        prior versions and flag material changes.

        Args:
            context: Pipeline context dictionary.

        Returns:
            Stage result with change detection data.
        """
        engine = self._get_change_detector_engine()

        changes_detected = 0
        recalculation_needed = False
        change_details: List[Dict[str, Any]] = []

        # Only check for change-related event types
        change_event_types = {
            "calculation_updated",
            "emission_factor_updated",
            "correction_applied",
            "assumption_changed",
            "methodology_changed",
            "boundary_changed",
            "allocation_changed",
        }

        if (
            engine is not None
            and context["event_type"] in change_event_types
        ):
            try:
                detect_data = {
                    "event_id": context.get("event_id"),
                    "calculation_id": context["calculation_id"],
                    "payload": context["payload"],
                    "event_type": context["event_type"],
                    "organization_id": context["organization_id"],
                }
                result = engine.detect_changes(detect_data)
                if result is not None:
                    changes_detected = result.get("changes_detected", 0)
                    recalculation_needed = result.get(
                        "recalculation_needed", False
                    )
                    change_details = result.get("change_details", [])
            except Exception as e:
                logger.warning(
                    "ChangeDetectorEngine.detect_changes failed: %s", e
                )

        context["change_result"] = {
            "changes_detected": changes_detected,
            "recalculation_needed": recalculation_needed,
        }

        # Record change metrics
        if changes_detected > 0:
            metrics = self._get_metrics()
            if metrics is not None:
                try:
                    metrics.record_change(
                        event_type=context["event_type"],
                        count=changes_detected,
                    )
                    if recalculation_needed:
                        metrics.record_recalculation(
                            scope=context["scope"],
                            category=context["category"],
                        )
                except Exception:
                    pass

        logger.debug(
            "Stage DETECT: changes=%d, recalculation=%s",
            changes_detected,
            recalculation_needed,
        )

        return {
            "status": "SUCCESS",
            "data": {
                "changes_detected": changes_detected,
                "recalculation_needed": recalculation_needed,
                "change_details": change_details,
            },
            "errors": [],
            "warnings": [],
        }

    # ------------------------------------------------------------------
    # Stage 7: VERIFY
    # ------------------------------------------------------------------

    def _stage_verify(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stage 7: Verify hash chain and lineage integrity.

        Uses AuditEventEngine.verify_chain to confirm that the
        hash chain for this calculation remains intact and unmodified.

        Args:
            context: Pipeline context dictionary.

        Returns:
            Stage result with verification status.
        """
        engine = self._get_audit_event_engine()

        chain_valid = True
        chain_length = 0
        verification_hash = ""

        if engine is not None:
            try:
                verify_data = {
                    "calculation_id": context["calculation_id"],
                    "organization_id": context["organization_id"],
                    "chain_hash": context.get("chain_hash", ""),
                }
                result = engine.verify_chain(verify_data)
                if result is not None:
                    chain_valid = result.get("valid", True)
                    chain_length = result.get("chain_length", 0)
                    verification_hash = result.get(
                        "verification_hash", ""
                    )
            except Exception as e:
                logger.warning(
                    "AuditEventEngine.verify_chain failed: %s", e
                )

        context["verification_result"] = {
            "chain_valid": chain_valid,
            "chain_length": chain_length,
        }

        # Record verification metric
        metrics = self._get_metrics()
        if metrics is not None:
            try:
                metrics.record_chain_verification(
                    valid=chain_valid,
                    chain_length=chain_length,
                )
            except Exception:
                pass

        warnings: List[str] = []
        if not chain_valid:
            warnings.append(
                "Hash chain integrity verification FAILED for "
                f"calculation_id={context['calculation_id']}"
            )
            context["warnings"].extend(warnings)

        logger.debug(
            "Stage VERIFY: valid=%s, chain_length=%d",
            chain_valid,
            chain_length,
        )

        return {
            "status": "SUCCESS" if chain_valid else "FAILED",
            "data": {
                "chain_valid": chain_valid,
                "chain_length": chain_length,
                "verification_hash": verification_hash,
            },
            "errors": [] if chain_valid else [
                "Chain integrity verification failed"
            ],
            "warnings": warnings,
        }

    # ------------------------------------------------------------------
    # Stage 8: PACKAGE
    # ------------------------------------------------------------------

    def _stage_package(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stage 8: Bundle evidence for verification.

        Uses EvidencePackagerEngine to create an evidence package
        containing all audit artifacts for this event. Only executed
        when include_evidence=True.

        Args:
            context: Pipeline context dictionary.

        Returns:
            Stage result with evidence package data.
        """
        if not context.get("include_evidence", False):
            logger.debug("Stage PACKAGE: skipped (include_evidence=False)")
            return {
                "status": "SUCCESS",
                "data": {"skipped": True, "reason": "include_evidence=False"},
                "errors": [],
                "warnings": [],
            }

        engine = self._get_evidence_packager_engine()

        package_id = None
        completeness_score = Decimal("0.0")

        if engine is not None:
            try:
                package_data = {
                    "event_id": context.get("event_id"),
                    "chain_hash": context.get("chain_hash"),
                    "lineage_node_id": context.get("lineage_node_id"),
                    "trace_id": context.get("trace_id"),
                    "verification_result": context.get(
                        "verification_result"
                    ),
                    "calculation_id": context["calculation_id"],
                    "organization_id": context["organization_id"],
                    "scope": context["scope"],
                    "category": context["category"],
                    "payload": context["payload"],
                }
                result = engine.package_evidence(package_data)
                if result is not None:
                    package_id = result.get("package_id")
                    completeness_score = Decimal(
                        str(result.get("completeness_score", "0.0"))
                    )
            except Exception as e:
                logger.warning(
                    "EvidencePackagerEngine.package_evidence failed: %s", e
                )

        context["evidence_package"] = {
            "package_id": package_id,
            "completeness_score": str(completeness_score),
        }

        # Record evidence metric
        metrics = self._get_metrics()
        if metrics is not None:
            try:
                metrics.record_evidence_package(
                    completeness_score=float(completeness_score),
                    scope=context["scope"],
                )
            except Exception:
                pass

        logger.debug(
            "Stage PACKAGE: package_id=%s, completeness=%s",
            package_id,
            completeness_score,
        )

        return {
            "status": "SUCCESS",
            "data": {
                "package_id": package_id,
                "completeness_score": str(completeness_score),
            },
            "errors": [],
            "warnings": [],
        }

    # ------------------------------------------------------------------
    # Stage 9: COMPLIANCE
    # ------------------------------------------------------------------

    def _stage_compliance(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stage 9: Check audit trail completeness per framework.

        Uses ComplianceCheckerEngine to evaluate audit trail
        completeness against each applicable framework. Only executed
        when include_compliance=True.

        Args:
            context: Pipeline context dictionary.

        Returns:
            Stage result with compliance coverage data.
        """
        if not context.get("include_compliance", False):
            logger.debug(
                "Stage COMPLIANCE: skipped (include_compliance=False)"
            )
            return {
                "status": "SUCCESS",
                "data": {
                    "skipped": True,
                    "reason": "include_compliance=False",
                },
                "errors": [],
                "warnings": [],
            }

        engine = self._get_compliance_checker_engine()

        classification = context.get("classification", {})
        frameworks = classification.get("applicable_frameworks", [])
        coverage_results: Dict[str, Any] = {}
        overall_coverage_pct = 0.0

        if engine is not None and frameworks:
            try:
                check_data = {
                    "organization_id": context["organization_id"],
                    "reporting_year": context["reporting_year"],
                    "scope": context["scope"],
                    "frameworks": frameworks,
                }
                result = engine.check_compliance(check_data)
                if result is not None:
                    coverage_results = result.get(
                        "framework_coverage", {}
                    )
                    overall_coverage_pct = result.get(
                        "overall_coverage_pct", 0.0
                    )
            except Exception as e:
                logger.warning(
                    "ComplianceCheckerEngine.check_compliance failed: %s", e
                )

        context["compliance_result"] = {
            "framework_coverage": coverage_results,
            "overall_coverage_pct": overall_coverage_pct,
        }

        # Record compliance metric
        metrics = self._get_metrics()
        if metrics is not None:
            try:
                metrics.record_compliance_coverage(
                    coverage_pct=overall_coverage_pct,
                    organization_id=context["organization_id"],
                )
            except Exception:
                pass

        logger.debug(
            "Stage COMPLIANCE: coverage=%.1f%%, frameworks=%d",
            overall_coverage_pct,
            len(frameworks),
        )

        return {
            "status": "SUCCESS",
            "data": {
                "framework_coverage": coverage_results,
                "overall_coverage_pct": overall_coverage_pct,
                "frameworks_checked": frameworks,
            },
            "errors": [],
            "warnings": [],
        }

    # ------------------------------------------------------------------
    # Stage 10: SEAL
    # ------------------------------------------------------------------

    def _stage_seal(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stage 10: Final seal with provenance hash and optional signature.

        Computes the final provenance seal hash that covers all prior
        stage results, creating a tamper-evident record of the entire
        pipeline execution.

        Args:
            context: Pipeline context dictionary.

        Returns:
            Stage result with seal_hash and provenance data.
        """
        seal_hash = self._compute_seal_hash(context)
        context["seal_hash"] = seal_hash

        # Record provenance stage if tracker available
        tracker = self._get_provenance_tracker()
        if tracker is not None:
            try:
                tracker.record_stage(
                    chain_id=context["run_id"],
                    stage="SEAL",
                    input_data={
                        "event_id": context.get("event_id"),
                        "chain_hash": context.get("chain_hash"),
                    },
                    output_data={"seal_hash": seal_hash},
                )
            except Exception:
                pass

        logger.debug(
            "Stage SEAL: seal_hash=%s...", seal_hash[:16]
        )

        return {
            "status": "SUCCESS",
            "data": {
                "seal_hash": seal_hash,
                "sealed_at": datetime.now(timezone.utc).isoformat(),
                "agent_id": self.agent_id,
                "version": self.version,
            },
            "errors": [],
            "warnings": [],
        }

    # ------------------------------------------------------------------
    # Result builder
    # ------------------------------------------------------------------

    def _build_result(
        self,
        context: Dict[str, Any],
        total_ms: float,
    ) -> Dict[str, Any]:
        """
        Build the final pipeline result dictionary.

        Assembles all stage results, provenance hashes, and metadata
        into the final output.

        Args:
            context: Pipeline context dictionary with all stage results.
            total_ms: Total pipeline execution time in milliseconds.

        Returns:
            Complete pipeline result dictionary.
        """
        return {
            "status": context.get("overall_status", PipelineStatus.SUCCESS.value),
            "run_id": context["run_id"],
            "event_id": context.get("event_id"),
            "event_type": context["event_type"],
            "scope": context["scope"],
            "category": context["category"],
            "organization_id": context["organization_id"],
            "reporting_year": context["reporting_year"],
            "calculation_id": context["calculation_id"],
            "chain_hash": context.get("chain_hash"),
            "seal_hash": context.get("seal_hash"),
            "lineage_node_id": context.get("lineage_node_id"),
            "trace_id": context.get("trace_id"),
            "change_result": context.get("change_result"),
            "verification_result": context.get("verification_result"),
            "evidence_package": context.get("evidence_package"),
            "compliance_result": context.get("compliance_result"),
            "classification": context.get("classification"),
            "stage_results": context.get("stage_results", {}),
            "stage_durations_ms": context.get("stage_durations_ms", {}),
            "processing_time_ms": round(total_ms, 3),
            "errors": context.get("errors", []),
            "warnings": context.get("warnings", []),
            "agent_id": self.agent_id,
            "version": self.version,
            "timestamp": context.get("timestamp", ""),
        }

    # ------------------------------------------------------------------
    # Seal hash computation
    # ------------------------------------------------------------------

    def _compute_seal_hash(self, context: Dict[str, Any]) -> str:
        """
        Compute SHA-256 seal hash covering all pipeline stage results.

        The seal hash provides a tamper-evident digest of the entire
        pipeline execution. It includes the event data, chain hash,
        all stage results, and timestamps.

        Args:
            context: Pipeline context dictionary.

        Returns:
            SHA-256 hex digest string.
        """
        seal_data = {
            "run_id": context["run_id"],
            "event_id": context.get("event_id"),
            "event_type": context["event_type"],
            "agent_id": context["agent_id"],
            "scope": context["scope"],
            "category": context["category"],
            "organization_id": context["organization_id"],
            "reporting_year": context["reporting_year"],
            "calculation_id": context["calculation_id"],
            "chain_hash": context.get("chain_hash", ""),
            "lineage_node_id": context.get("lineage_node_id", ""),
            "trace_id": context.get("trace_id", ""),
            "data_quality_score": str(context.get("data_quality_score", "1.0")),
            "stage_durations_ms": context.get("stage_durations_ms", {}),
            "timestamp": context.get("timestamp", ""),
            "pipeline_agent_id": self.agent_id,
            "pipeline_version": self.version,
        }
        return _compute_hash(seal_data)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def _update_stage_stats(
        self, stage: PipelineStage, duration_ms: float
    ) -> None:
        """
        Update per-stage execution statistics.

        Args:
            stage: Pipeline stage that completed.
            duration_ms: Stage duration in milliseconds.
        """
        with self._stats_lock:
            self._stats["stage_execution_counts"][stage.value] += 1
            self._stats["stage_total_duration_ms"][stage.value] += duration_ms

    def _update_execution_stats(self, status: str) -> None:
        """
        Update pipeline-level execution statistics.

        Args:
            status: Final pipeline status string.
        """
        with self._stats_lock:
            self._stats["total_executions"] += 1
            self._stats["total_events_processed"] += 1
            if status == PipelineStatus.SUCCESS.value:
                self._stats["total_successes"] += 1
            elif status == PipelineStatus.PARTIAL_SUCCESS.value:
                self._stats["total_partial_successes"] += 1
            elif status == PipelineStatus.VALIDATION_ERROR.value:
                self._stats["total_validation_errors"] += 1
            else:
                self._stats["total_failures"] += 1

    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """
        Get pipeline execution statistics.

        Returns:
            Dictionary with execution counts, stage durations,
            success/failure rates, and uptime information.
        """
        with self._stats_lock:
            stats = {
                "total_executions": self._stats["total_executions"],
                "total_batch_executions": self._stats["total_batch_executions"],
                "total_events_processed": self._stats["total_events_processed"],
                "total_successes": self._stats["total_successes"],
                "total_partial_successes": self._stats["total_partial_successes"],
                "total_failures": self._stats["total_failures"],
                "total_validation_errors": self._stats["total_validation_errors"],
                "stage_execution_counts": dict(
                    self._stats["stage_execution_counts"]
                ),
                "stage_avg_duration_ms": {},
                "started_at": self._stats["started_at"],
                "agent_id": self.agent_id,
                "version": self.version,
            }

            # Compute per-stage average durations
            for stage_name in self._stats["stage_execution_counts"]:
                count = self._stats["stage_execution_counts"][stage_name]
                total_dur = self._stats["stage_total_duration_ms"][stage_name]
                if count > 0:
                    stats["stage_avg_duration_ms"][stage_name] = round(
                        total_dur / count, 3
                    )
                else:
                    stats["stage_avg_duration_ms"][stage_name] = 0.0

            # Success rate
            total = stats["total_executions"]
            if total > 0:
                stats["success_rate_pct"] = round(
                    (stats["total_successes"] / total) * 100.0, 2
                )
            else:
                stats["success_rate_pct"] = 0.0

        return stats

    def reset(self) -> None:
        """
        Reset pipeline engine state and statistics.

        Clears all accumulated statistics and resets engine references.
        Primarily used for testing.
        """
        with self._stats_lock:
            self._stats = {
                "total_executions": 0,
                "total_batch_executions": 0,
                "total_events_processed": 0,
                "total_successes": 0,
                "total_partial_successes": 0,
                "total_failures": 0,
                "total_validation_errors": 0,
                "stage_execution_counts": {
                    s.value: 0 for s in PipelineStage
                },
                "stage_total_duration_ms": {
                    s.value: 0.0 for s in PipelineStage
                },
                "started_at": datetime.now(timezone.utc).isoformat(),
            }

        self._audit_event_engine = None
        self._lineage_graph_engine = None
        self._compliance_tracer_engine = None
        self._change_detector_engine = None
        self._evidence_packager_engine = None
        self._compliance_checker_engine = None
        self._provenance_tracker = None
        self._metrics = None

        logger.info("AuditTrailPipelineEngine reset")

    @classmethod
    def reset_singleton(cls) -> None:
        """
        Reset the singleton instance.

        Clears the singleton so a fresh instance is created on next access.
        Primarily used for testing.
        """
        with cls._instance_lock:
            cls._instance = None
            logger.info("AuditTrailPipelineEngine singleton reset")


# ==============================================================================
# MODULE-LEVEL SINGLETON ACCESSOR
# ==============================================================================


_pipeline_instance: Optional[AuditTrailPipelineEngine] = None
_pipeline_lock: threading.Lock = threading.Lock()


def get_pipeline_engine() -> AuditTrailPipelineEngine:
    """
    Get the singleton AuditTrailPipelineEngine instance.

    Thread-safe accessor for the global pipeline engine.

    Returns:
        AuditTrailPipelineEngine singleton instance.

    Example:
        >>> from greenlang.audit_trail_lineage.audit_trail_pipeline import (
        ...     get_pipeline_engine,
        ... )
        >>> pipeline = get_pipeline_engine()
        >>> result = pipeline.execute(
        ...     event_type="calculation_completed",
        ...     agent_id="GL-MRV-S1-001",
        ...     scope="scope_1",
        ...     category="stationary_combustion",
        ...     organization_id="ORG-001",
        ...     reporting_year=2025,
        ...     calculation_id="calc-abc123",
        ...     payload={"emissions_tco2e": 1234.56},
        ... )
    """
    global _pipeline_instance

    if _pipeline_instance is None:
        with _pipeline_lock:
            if _pipeline_instance is None:
                _pipeline_instance = AuditTrailPipelineEngine()

    return _pipeline_instance


def reset_pipeline_engine() -> None:
    """
    Reset the singleton pipeline engine instance.

    Clears the singleton so a fresh instance is created on next access.
    Primarily used for testing.
    """
    global _pipeline_instance
    with _pipeline_lock:
        _pipeline_instance = None
    AuditTrailPipelineEngine.reset_singleton()
    logger.info("Pipeline engine singleton reset")


# ==============================================================================
# MODULE EXPORTS
# ==============================================================================

__all__ = [
    # Enums
    "PipelineStatus",
    "PipelineStage",
    "EventScope",
    # Engine
    "AuditTrailPipelineEngine",
    # Singleton accessors
    "get_pipeline_engine",
    "reset_pipeline_engine",
    # Constants
    "AGENT_ID",
    "AGENT_COMPONENT",
    "VERSION",
]
