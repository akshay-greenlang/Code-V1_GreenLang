# -*- coding: utf-8 -*-
"""
Audit Trail & Lineage Service Setup - AGENT-MRV-030

This module provides the service facade that wires together all 7 engines
for the Audit Trail & Lineage Agent (GL-MRV-X-042), a cross-cutting MRV
component providing immutable audit trails and end-to-end calculation
lineage for all Scope 1, 2, and 3 emissions calculations.

The AuditTrailLineageService class provides a high-level API for:
- Immutable audit event recording with SHA-256 hash chains
- Batch event recording for historical data
- Event querying by type, scope, agent, time range
- Chain integrity verification (forward and backward)
- MRV calculation lineage DAG construction and traversal
- Forward-impact and backward-traceability analysis
- Evidence packaging for third-party verification
- Digital signature and package verification
- Regulatory framework requirement traceability
- Compliance coverage assessment across 9 frameworks
- Recalculation change detection and version comparison
- Change materiality impact analysis
- 10-stage orchestration pipeline execution
- Audit trail summary aggregation across all engines

Engines:
    1. AuditEventEngine - Immutable event recording with SHA-256 hash chains
    2. LineageGraphEngine - MRV calculation lineage DAG construction and traversal
    3. EvidencePackagerEngine - Audit evidence bundling for third-party verification
    4. ComplianceTracerEngine - Regulatory framework requirement traceability
    5. ChangeDetectorEngine - Recalculation change tracking and version comparison
    6. ComplianceCheckerEngine - Multi-framework audit trail compliance validation
    7. AuditTrailPipelineEngine - 10-stage orchestration pipeline

Architecture:
    - Thread-safe singleton pattern for service instance
    - Graceful imports with try/except for optional dependencies
    - Comprehensive metrics tracking via OBS-001 integration
    - Provenance tracking for all mutations via AGENT-FOUND-008
    - Type-safe request/response models using Pydantic
    - Structured logging with contextual information

Example:
    >>> from greenlang.agents.mrv.audit_trail_lineage.setup import get_service
    >>> service = get_service()
    >>> response = service.record_event(RecordEventRequest(
    ...     event_type="CALCULATION_COMPLETED",
    ...     agent_id="GL-MRV-S1-001",
    ...     scope="scope_1",
    ...     organization_id="org-001",
    ...     reporting_year=2025,
    ...     payload={"total_co2e": "1234.56"},
    ... ))
    >>> assert response.success

Integration:
    >>> from greenlang.agents.mrv.audit_trail_lineage.setup import get_router
    >>> app.include_router(get_router(), prefix="/api/v1/audit-trail-lineage")

Module: greenlang.agents.mrv.audit_trail_lineage.setup
Agent: AGENT-MRV-030
Agent ID: GL-MRV-X-042
Version: 1.0.0
"""

import hashlib
import importlib
import logging
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import Field, validator
from greenlang.schemas import GreenLangBase

# Thread-safe singleton lock
_service_lock = threading.RLock()
_service_instance: Optional["AuditTrailLineageService"] = None

logger = logging.getLogger(__name__)


# ============================================================================
# Request Models
# ============================================================================


class RecordEventRequest(GreenLangBase):
    """Request model for recording a single audit event."""

    event_type: str = Field(
        ...,
        description=(
            "Audit event type: DATA_INGESTED, DATA_VALIDATED, "
            "DATA_TRANSFORMED, EMISSION_FACTOR_RESOLVED, "
            "CALCULATION_STARTED, CALCULATION_COMPLETED, "
            "CALCULATION_FAILED, COMPLIANCE_CHECKED, "
            "REPORT_GENERATED, PROVENANCE_SEALED, "
            "MANUAL_OVERRIDE, CHAIN_VERIFIED"
        ),
    )
    agent_id: str = Field(..., description="Identifier of the originating agent")
    scope: Optional[str] = Field(
        None, description="GHG scope: scope_1, scope_2, scope_3"
    )
    category: Optional[int] = Field(
        None, ge=1, le=15, description="Scope 3 category number (1-15)"
    )
    organization_id: str = Field(..., description="Organization identifier")
    reporting_year: int = Field(
        ..., ge=1990, le=2100, description="Reporting year"
    )
    calculation_id: Optional[str] = Field(
        None, description="Optional calculation identifier"
    )
    payload: Dict[str, Any] = Field(
        default_factory=dict, description="Event-specific payload data"
    )
    data_quality_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Data quality score (0.00-1.00)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional context metadata"
    )
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")

    @validator("event_type")
    def validate_event_type(cls, v: str) -> str:
        """Validate event type against known types."""
        allowed = {
            "DATA_INGESTED", "DATA_VALIDATED", "DATA_TRANSFORMED",
            "EMISSION_FACTOR_RESOLVED", "CALCULATION_STARTED",
            "CALCULATION_COMPLETED", "CALCULATION_FAILED",
            "COMPLIANCE_CHECKED", "REPORT_GENERATED",
            "PROVENANCE_SEALED", "MANUAL_OVERRIDE", "CHAIN_VERIFIED",
        }
        if v.upper() not in allowed:
            raise ValueError(f"event_type must be one of {sorted(allowed)}")
        return v.upper()

    @validator("scope")
    def validate_scope(cls, v: Optional[str]) -> Optional[str]:
        """Validate GHG scope value."""
        if v is not None:
            allowed = {"scope_1", "scope_2", "scope_3"}
            if v not in allowed:
                raise ValueError(f"scope must be one of {sorted(allowed)}")
        return v


class RecordBatchRequest(GreenLangBase):
    """Request model for recording a batch of audit events."""

    events: List[Dict[str, Any]] = Field(
        ..., min_length=1, max_length=5000,
        description="List of event dictionaries (up to 5,000)"
    )
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")


class GetEventsRequest(GreenLangBase):
    """Request model for querying audit events."""

    organization_id: str = Field(..., description="Organization identifier")
    reporting_year: int = Field(..., ge=1990, le=2100)
    event_type: Optional[str] = Field(None, description="Filter by event type")
    agent_id: Optional[str] = Field(None, description="Filter by agent")
    scope: Optional[str] = Field(None, description="Filter by GHG scope")
    category: Optional[int] = Field(None, ge=1, le=15)
    start_time: Optional[str] = Field(None, description="ISO 8601 lower bound")
    end_time: Optional[str] = Field(None, description="ISO 8601 upper bound")
    limit: int = Field(default=1000, ge=1, le=10000)
    offset: int = Field(default=0, ge=0)


class VerifyChainRequest(GreenLangBase):
    """Request model for chain integrity verification."""

    organization_id: str = Field(..., description="Organization identifier")
    reporting_year: int = Field(..., ge=1990, le=2100)
    start_position: Optional[int] = Field(None, ge=0)
    end_position: Optional[int] = Field(None, ge=0)


class AddLineageNodeRequest(GreenLangBase):
    """Request model for adding a lineage graph node."""

    node_id: str = Field(..., description="Unique node identifier")
    node_type: str = Field(
        ..., description="Node type: source, emission_factor, calculation, aggregation, disclosure"
    )
    agent_id: str = Field(..., description="Agent that produced this node")
    scope: Optional[str] = Field(None, description="GHG scope")
    category: Optional[int] = Field(None, ge=1, le=15)
    organization_id: str = Field(..., description="Organization identifier")
    reporting_year: int = Field(..., ge=1990, le=2100)
    data: Dict[str, Any] = Field(
        default_factory=dict, description="Node-specific data"
    )
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")


class AddLineageEdgeRequest(GreenLangBase):
    """Request model for adding a lineage graph edge."""

    source_node_id: str = Field(..., description="Source node identifier")
    target_node_id: str = Field(..., description="Target node identifier")
    edge_type: str = Field(
        ..., description="Edge type: derived_from, contributes_to, aggregated_into"
    )
    transformation: Optional[str] = Field(
        None, description="Transformation applied along this edge"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")


class GetLineageGraphRequest(GreenLangBase):
    """Request model for retrieving a lineage graph."""

    organization_id: str = Field(..., description="Organization identifier")
    reporting_year: int = Field(..., ge=1990, le=2100)
    scope: Optional[str] = Field(None, description="Filter by scope")
    max_depth: Optional[int] = Field(None, ge=1, le=500)


class TraceLineageRequest(GreenLangBase):
    """Request model for tracing lineage from a node."""

    node_id: str = Field(..., description="Starting node identifier")
    direction: str = Field(
        "backward",
        description="Traversal direction: forward or backward"
    )
    max_depth: Optional[int] = Field(None, ge=1, le=500)

    @validator("direction")
    def validate_direction(cls, v: str) -> str:
        """Validate traversal direction."""
        if v not in ("forward", "backward"):
            raise ValueError("direction must be 'forward' or 'backward'")
        return v


class CreateEvidencePackageRequest(GreenLangBase):
    """Request model for creating an evidence package."""

    organization_id: str = Field(..., description="Organization identifier")
    reporting_year: int = Field(..., ge=1990, le=2100)
    scope: Optional[str] = Field(None, description="Scope filter for evidence")
    assurance_level: str = Field(
        default="limited",
        description="Assurance level: limited, reasonable, none"
    )
    include_lineage: bool = Field(True, description="Include lineage graph")
    include_chain: bool = Field(True, description="Include full audit chain")
    description: Optional[str] = Field(None, description="Package description")
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")

    @validator("assurance_level")
    def validate_assurance_level(cls, v: str) -> str:
        """Validate assurance level."""
        allowed = {"limited", "reasonable", "none"}
        if v not in allowed:
            raise ValueError(f"assurance_level must be one of {sorted(allowed)}")
        return v


class TraceComplianceRequest(GreenLangBase):
    """Request model for compliance traceability."""

    organization_id: str = Field(..., description="Organization identifier")
    reporting_year: int = Field(..., ge=1990, le=2100)
    framework: str = Field(
        ..., description="Framework: GHG_PROTOCOL, ISO_14064, CSRD_ESRS, etc."
    )
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")


class DetectChangeRequest(GreenLangBase):
    """Request model for change detection."""

    calculation_id: str = Field(
        ..., description="Calculation identifier to check for changes"
    )
    organization_id: str = Field(..., description="Organization identifier")
    reporting_year: int = Field(..., ge=1990, le=2100)
    previous_version: Optional[Dict[str, Any]] = Field(
        None, description="Previous calculation result for comparison"
    )
    current_version: Optional[Dict[str, Any]] = Field(
        None, description="Current calculation result for comparison"
    )
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")


class ExecutePipelineRequest(GreenLangBase):
    """Request model for pipeline execution."""

    organization_id: str = Field(..., description="Organization identifier")
    reporting_year: int = Field(..., ge=1990, le=2100)
    scope: Optional[str] = Field(None, description="Scope filter")
    frameworks: List[str] = Field(
        default_factory=lambda: ["GHG_PROTOCOL"],
        description="Compliance frameworks to check"
    )
    include_evidence: bool = Field(True, description="Generate evidence package")
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")


class ExecutePipelineBatchRequest(GreenLangBase):
    """Request model for batch pipeline execution."""

    requests: List[ExecutePipelineRequest] = Field(
        ..., min_length=1, max_length=100,
        description="List of pipeline execution requests"
    )
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")


class ComplianceCheckRequest(GreenLangBase):
    """Request model for multi-framework compliance checking."""

    organization_id: str = Field(..., description="Organization identifier")
    reporting_year: int = Field(..., ge=1990, le=2100)
    frameworks: List[str] = Field(
        default_factory=lambda: [
            "GHG_PROTOCOL", "ISO_14064", "CSRD_ESRS",
        ],
        description="Frameworks to check"
    )
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")


# ============================================================================
# Response Models
# ============================================================================


class RecordEventResponse(GreenLangBase):
    """Response model for recording a single audit event."""

    success: bool = Field(..., description="Success flag")
    event_id: str = Field(..., description="Unique event identifier")
    event_type: str = Field(..., description="Audit event type")
    event_hash: str = Field(..., description="SHA-256 hash of this event")
    chain_position: int = Field(..., description="Position in the hash chain")
    chain_key: str = Field(..., description="Chain key (org:year)")
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp")
    processing_time_ms: float = Field(..., description="Processing time in ms")
    error: Optional[str] = Field(None, description="Error message if failed")


class RecordBatchResponse(GreenLangBase):
    """Response model for batch event recording."""

    success: bool = Field(..., description="Success flag")
    total_recorded: int = Field(..., description="Total events recorded")
    event_ids: List[str] = Field(default_factory=list, description="Recorded event IDs")
    errors: List[dict] = Field(default_factory=list, description="Error details")
    processing_time_ms: float = Field(..., description="Processing time in ms")


class GetEventsResponse(GreenLangBase):
    """Response model for event queries."""

    success: bool = Field(..., description="Success flag")
    events: List[Dict[str, Any]] = Field(default_factory=list)
    total_matching: int = Field(default=0, description="Total matching events")
    returned_count: int = Field(default=0, description="Events in this page")
    has_more: bool = Field(default=False, description="More events available")
    processing_time_ms: float = Field(..., description="Processing time in ms")
    error: Optional[str] = Field(None, description="Error message if failed")


class VerifyChainResponse(GreenLangBase):
    """Response model for chain verification."""

    success: bool = Field(..., description="Success flag")
    valid: bool = Field(..., description="True if chain integrity is intact")
    chain_key: str = Field(..., description="Chain key (org:year)")
    verified_count: int = Field(..., description="Number of events verified")
    first_invalid_position: Optional[int] = Field(
        None, description="Position of first invalid event"
    )
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    verification_time_ms: float = Field(..., description="Verification time in ms")


class LineageResponse(GreenLangBase):
    """Response model for lineage graph operations."""

    success: bool = Field(..., description="Success flag")
    result: Optional[Dict[str, Any]] = Field(
        None, description="Lineage operation result"
    )
    processing_time_ms: float = Field(..., description="Processing time in ms")
    error: Optional[str] = Field(None, description="Error message if failed")


class EvidencePackageResponse(GreenLangBase):
    """Response model for evidence package operations."""

    success: bool = Field(..., description="Success flag")
    package_id: Optional[str] = Field(None, description="Evidence package ID")
    organization_id: str = Field(..., description="Organization identifier")
    reporting_year: int = Field(..., description="Reporting year")
    assurance_level: Optional[str] = Field(None, description="Assurance level")
    package_hash: Optional[str] = Field(None, description="SHA-256 hash of package")
    result: Optional[Dict[str, Any]] = Field(None, description="Package details")
    processing_time_ms: float = Field(..., description="Processing time in ms")
    error: Optional[str] = Field(None, description="Error message if failed")


class ComplianceTraceResponse(GreenLangBase):
    """Response model for compliance traceability."""

    success: bool = Field(..., description="Success flag")
    framework: str = Field(..., description="Framework assessed")
    coverage: Optional[Dict[str, Any]] = Field(None, description="Coverage details")
    result: Optional[Dict[str, Any]] = Field(None, description="Trace result")
    processing_time_ms: float = Field(..., description="Processing time in ms")
    error: Optional[str] = Field(None, description="Error message if failed")


class ChangeDetectionResponse(GreenLangBase):
    """Response model for change detection."""

    success: bool = Field(..., description="Success flag")
    calculation_id: str = Field(..., description="Calculation checked")
    changes_detected: bool = Field(False, description="Whether changes found")
    change_count: int = Field(0, description="Number of changes detected")
    materiality_status: Optional[str] = Field(
        None, description="MATERIAL, IMMATERIAL, or UNKNOWN"
    )
    result: Optional[Dict[str, Any]] = Field(None, description="Change details")
    processing_time_ms: float = Field(..., description="Processing time in ms")
    error: Optional[str] = Field(None, description="Error message if failed")


class PipelineResponse(GreenLangBase):
    """Response model for pipeline execution."""

    success: bool = Field(..., description="Success flag")
    organization_id: str = Field(..., description="Organization identifier")
    reporting_year: int = Field(..., description="Reporting year")
    status: str = Field(default="unknown", description="Pipeline status")
    stages_completed: int = Field(default=0, description="Stages completed")
    total_stages: int = Field(default=10, description="Total pipeline stages")
    result: Optional[Dict[str, Any]] = Field(None, description="Pipeline result")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")
    processing_time_ms: float = Field(..., description="Processing time in ms")
    error: Optional[str] = Field(None, description="Error message if failed")


class PipelineBatchResponse(GreenLangBase):
    """Response model for batch pipeline execution."""

    success: bool = Field(..., description="Overall success flag")
    total_requests: int = Field(..., description="Total requests")
    successful: int = Field(..., description="Successful executions")
    failed: int = Field(..., description="Failed executions")
    results: List[PipelineResponse] = Field(default_factory=list)
    errors: List[dict] = Field(default_factory=list)
    processing_time_ms: float = Field(..., description="Total processing time")


class ComplianceCheckResponse(GreenLangBase):
    """Response model for multi-framework compliance checking."""

    success: bool = Field(..., description="Success flag")
    overall_status: str = Field(..., description="PASS, WARNING, or FAIL")
    overall_score: float = Field(..., description="Overall compliance score (0-100)")
    framework_results: List[dict] = Field(
        default_factory=list, description="Per-framework results"
    )
    recommendations: List[str] = Field(
        default_factory=list, description="Improvement recommendations"
    )
    checked_at: datetime = Field(..., description="Check timestamp")
    processing_time_ms: float = Field(..., description="Processing time")


class AuditTrailSummaryResponse(GreenLangBase):
    """Response model for audit trail summary."""

    success: bool = Field(..., description="Success flag")
    organization_id: str = Field(..., description="Organization identifier")
    reporting_year: int = Field(..., description="Reporting year")
    total_events: int = Field(default=0)
    chain_length: int = Field(default=0)
    chain_valid: Optional[bool] = Field(None)
    lineage_nodes: int = Field(default=0)
    lineage_edges: int = Field(default=0)
    evidence_packages: int = Field(default=0)
    compliance_coverage: Optional[Dict[str, Any]] = Field(None)
    changes_detected: int = Field(default=0)
    provenance_hash: str = Field(default="")
    processing_time_ms: float = Field(..., description="Processing time in ms")
    error: Optional[str] = Field(None)


class HealthResponse(GreenLangBase):
    """Response model for health check."""

    status: str = Field(..., description="Service status: healthy, degraded, unhealthy")
    version: str = Field(..., description="Service version")
    agent_id: str = Field(..., description="Agent identifier")
    engines_status: Dict[str, bool] = Field(
        default_factory=dict, description="Per-engine status"
    )
    uptime_seconds: float = Field(..., description="Service uptime")


# ============================================================================
# AuditTrailLineageService Class
# ============================================================================


class AuditTrailLineageService:
    """
    Audit Trail & Lineage Service Facade.

    This service wires together all 7 engines to provide a complete API
    for immutable audit trails and MRV calculation lineage tracking across
    all Scope 1, 2, and 3 emissions calculations.

    The service supports:
        - Immutable SHA-256 hash-chained audit events
        - Batch event recording (up to 5,000 events)
        - Event querying with rich filtering
        - Chain integrity verification
        - Calculation lineage DAG construction and traversal
        - Forward-impact and backward-traceability analysis
        - Evidence packaging for third-party verification
        - Digital signature and package verification
        - Regulatory framework requirement traceability
        - Compliance coverage assessment (9 frameworks)
        - Recalculation change detection
        - Change materiality impact analysis
        - 10-stage orchestration pipeline
        - Provenance tracking with SHA-256 audit trail

    Engines:
        1. AuditEventEngine - Immutable event recording
        2. LineageGraphEngine - MRV lineage DAG
        3. EvidencePackagerEngine - Audit evidence bundling
        4. ComplianceTracerEngine - Regulatory traceability
        5. ChangeDetectorEngine - Change tracking
        6. ComplianceCheckerEngine - Multi-framework compliance
        7. AuditTrailPipelineEngine - 10-stage orchestration

    Thread Safety:
        This service is thread-safe. Use get_service() to obtain a singleton.

    Example:
        >>> service = get_service()
        >>> response = service.record_event(RecordEventRequest(...))
        >>> assert response.success
    """

    _instance: Optional["AuditTrailLineageService"] = None
    _lock: threading.RLock = threading.RLock()

    def __new__(cls) -> "AuditTrailLineageService":
        """Thread-safe singleton instantiation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        """Initialize AuditTrailLineageService with all 7 engines."""
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return

            logger.info("Initializing AuditTrailLineageService")
            self._start_time = datetime.now(timezone.utc)

            # Load configuration
            self._config = self._load_config()

            # Initialize all 7 engines with graceful fallback
            self._audit_event_engine = self._init_engine(
                "greenlang.agents.mrv.audit_trail_lineage.audit_event_engine",
                "AuditEventEngine",
            )
            self._lineage_graph_engine = self._init_engine(
                "greenlang.agents.mrv.audit_trail_lineage.lineage_graph",
                "LineageGraphEngine",
            )
            self._evidence_packager_engine = self._init_engine(
                "greenlang.agents.mrv.audit_trail_lineage.evidence_packager",
                "EvidencePackagerEngine",
            )
            self._compliance_tracer_engine = self._init_engine(
                "greenlang.agents.mrv.audit_trail_lineage.compliance_tracer",
                "ComplianceTracerEngine",
            )
            self._change_detector_engine = self._init_engine(
                "greenlang.agents.mrv.audit_trail_lineage.change_detector",
                "ChangeDetectorEngine",
            )
            self._compliance_checker_engine = self._init_engine(
                "greenlang.agents.mrv.audit_trail_lineage.compliance_checker",
                "ComplianceCheckerEngine",
            )
            self._pipeline_engine = self._init_engine(
                "greenlang.agents.mrv.audit_trail_lineage.audit_trail_pipeline",
                "AuditTrailPipelineEngine",
            )

            self._initialized = True
            logger.info("AuditTrailLineageService initialized successfully")

    @staticmethod
    def _load_config() -> Optional[Any]:
        """Load agent configuration with graceful fallback.

        Returns:
            AuditTrailLineageConfig instance or None.
        """
        try:
            from greenlang.agents.mrv.audit_trail_lineage.config import get_config
            return get_config()
        except ImportError:
            logger.warning("Config module not available, using defaults")
            return None
        except Exception as e:
            logger.warning("Config loading failed: %s", e)
            return None

    @staticmethod
    def _init_engine(module_path: str, class_name: str) -> Optional[Any]:
        """
        Initialize an engine with graceful ImportError handling.

        Args:
            module_path: Fully qualified module path.
            class_name: Class name within the module.

        Returns:
            Engine instance or None if import fails.
        """
        try:
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            # Use get_instance() for singletons that support it
            if hasattr(cls, "get_instance"):
                instance = cls.get_instance()
            else:
                instance = cls()
            logger.info("%s initialized", class_name)
            return instance
        except ImportError:
            logger.warning("%s not available (ImportError)", class_name)
            return None
        except Exception as e:
            logger.warning("%s initialization failed: %s", class_name, e)
            return None

    # ========================================================================
    # Event Methods (Engine 1: AuditEventEngine)
    # ========================================================================

    def record_event(
        self, request: RecordEventRequest,
    ) -> RecordEventResponse:
        """
        Record a single immutable audit event.

        Delegates to AuditEventEngine.record_event for SHA-256 hash chain
        recording. Each event is cryptographically linked to the previous
        event in the same organization/year chain.

        Args:
            request: Audit event recording request.

        Returns:
            RecordEventResponse with event hash and chain position.
        """
        start_time = time.monotonic()

        try:
            if self._audit_event_engine is None:
                raise RuntimeError("AuditEventEngine not available")

            dq_score = None
            if request.data_quality_score is not None:
                dq_score = Decimal(str(request.data_quality_score))

            result = self._audit_event_engine.record_event(
                event_type=request.event_type,
                agent_id=request.agent_id,
                scope=request.scope,
                category=request.category,
                organization_id=request.organization_id,
                reporting_year=request.reporting_year,
                calculation_id=request.calculation_id,
                payload=request.payload,
                data_quality_score=dq_score,
                metadata=request.metadata,
            )

            elapsed = (time.monotonic() - start_time) * 1000.0

            return RecordEventResponse(
                success=True,
                event_id=result["event_id"],
                event_type=result["event_type"],
                event_hash=result["event_hash"],
                chain_position=result["chain_position"],
                chain_key=result["chain_key"],
                timestamp=result["timestamp"],
                processing_time_ms=elapsed,
            )

        except Exception as e:
            elapsed = (time.monotonic() - start_time) * 1000.0
            logger.error("record_event failed: %s", e, exc_info=True)
            return RecordEventResponse(
                success=False,
                event_id="",
                event_type=request.event_type,
                event_hash="",
                chain_position=-1,
                chain_key="",
                timestamp=datetime.now(timezone.utc).isoformat(),
                processing_time_ms=elapsed,
                error=str(e),
            )

    def record_batch(
        self, request: RecordBatchRequest,
    ) -> RecordBatchResponse:
        """
        Record a batch of audit events atomically.

        Args:
            request: Batch recording request with up to 5,000 events.

        Returns:
            RecordBatchResponse with recorded event IDs.
        """
        start_time = time.monotonic()

        try:
            if self._audit_event_engine is None:
                raise RuntimeError("AuditEventEngine not available")

            result = self._audit_event_engine.record_batch(request.events)
            elapsed = (time.monotonic() - start_time) * 1000.0

            return RecordBatchResponse(
                success=True,
                total_recorded=result["total_recorded"],
                event_ids=result["event_ids"],
                processing_time_ms=elapsed,
            )

        except Exception as e:
            elapsed = (time.monotonic() - start_time) * 1000.0
            logger.error("record_batch failed: %s", e, exc_info=True)
            return RecordBatchResponse(
                success=False,
                total_recorded=0,
                errors=[{"error": str(e)}],
                processing_time_ms=elapsed,
            )

    def get_event(self, event_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a single audit event by its identifier.

        Args:
            event_id: UUID-based event identifier.

        Returns:
            Event dictionary if found, None otherwise.
        """
        if self._audit_event_engine is None:
            return None
        return self._audit_event_engine.get_event(event_id)

    def get_events(
        self, request: GetEventsRequest,
    ) -> GetEventsResponse:
        """
        Query audit events with optional filters.

        Args:
            request: Event query request with filters and pagination.

        Returns:
            GetEventsResponse with matching events.
        """
        start_time = time.monotonic()

        try:
            if self._audit_event_engine is None:
                raise RuntimeError("AuditEventEngine not available")

            result = self._audit_event_engine.get_events(
                organization_id=request.organization_id,
                reporting_year=request.reporting_year,
                event_type=request.event_type,
                agent_id=request.agent_id,
                scope=request.scope,
                category=request.category,
                start_time=request.start_time,
                end_time=request.end_time,
                limit=request.limit,
                offset=request.offset,
            )

            elapsed = (time.monotonic() - start_time) * 1000.0

            return GetEventsResponse(
                success=True,
                events=result.get("events", []),
                total_matching=result.get("total_matching", 0),
                returned_count=result.get("returned_count", 0),
                has_more=result.get("has_more", False),
                processing_time_ms=elapsed,
            )

        except Exception as e:
            elapsed = (time.monotonic() - start_time) * 1000.0
            logger.error("get_events failed: %s", e, exc_info=True)
            return GetEventsResponse(
                success=False,
                processing_time_ms=elapsed,
                error=str(e),
            )

    def verify_chain(
        self, request: VerifyChainRequest,
    ) -> VerifyChainResponse:
        """
        Verify the integrity of an audit hash chain.

        Recomputes every hash in the chain and validates linkage.

        Args:
            request: Chain verification request.

        Returns:
            VerifyChainResponse with verification results.
        """
        start_time = time.monotonic()

        try:
            if self._audit_event_engine is None:
                raise RuntimeError("AuditEventEngine not available")

            result = self._audit_event_engine.verify_chain(
                organization_id=request.organization_id,
                reporting_year=request.reporting_year,
                start_position=request.start_position,
                end_position=request.end_position,
            )

            elapsed = (time.monotonic() - start_time) * 1000.0

            return VerifyChainResponse(
                success=True,
                valid=result["valid"],
                chain_key=result["chain_key"],
                verified_count=result["verified_count"],
                first_invalid_position=result.get("first_invalid_position"),
                errors=result.get("errors", []),
                verification_time_ms=elapsed,
            )

        except Exception as e:
            elapsed = (time.monotonic() - start_time) * 1000.0
            logger.error("verify_chain failed: %s", e, exc_info=True)
            return VerifyChainResponse(
                success=False,
                valid=False,
                chain_key=f"{request.organization_id}:{request.reporting_year}",
                verified_count=0,
                errors=[{"error": str(e)}],
                verification_time_ms=elapsed,
            )

    def get_chain(
        self,
        organization_id: str,
        reporting_year: int,
    ) -> Dict[str, Any]:
        """
        Get the full audit event chain for an organization/year.

        Args:
            organization_id: Organization identifier.
            reporting_year: Reporting year.

        Returns:
            Chain data dictionary with events and metadata.
        """
        if self._audit_event_engine is None:
            return {"success": False, "error": "AuditEventEngine not available"}
        return self._audit_event_engine.get_chain(
            organization_id=organization_id,
            reporting_year=reporting_year,
        )

    # ========================================================================
    # Lineage Methods (Engine 2: LineageGraphEngine)
    # ========================================================================

    def add_lineage_node(
        self, request: AddLineageNodeRequest,
    ) -> LineageResponse:
        """
        Add a node to the MRV calculation lineage graph.

        Args:
            request: Lineage node request with node data.

        Returns:
            LineageResponse with operation result.
        """
        start_time = time.monotonic()

        try:
            if self._lineage_graph_engine is None:
                raise RuntimeError("LineageGraphEngine not available")

            result = self._lineage_graph_engine.add_node(
                node_id=request.node_id,
                node_type=request.node_type,
                agent_id=request.agent_id,
                scope=request.scope,
                category=request.category,
                organization_id=request.organization_id,
                reporting_year=request.reporting_year,
                data=request.data,
            )

            elapsed = (time.monotonic() - start_time) * 1000.0

            return LineageResponse(
                success=True,
                result=result if isinstance(result, dict) else {"node_id": request.node_id},
                processing_time_ms=elapsed,
            )

        except Exception as e:
            elapsed = (time.monotonic() - start_time) * 1000.0
            logger.error("add_lineage_node failed: %s", e, exc_info=True)
            return LineageResponse(
                success=False,
                processing_time_ms=elapsed,
                error=str(e),
            )

    def add_lineage_edge(
        self, request: AddLineageEdgeRequest,
    ) -> LineageResponse:
        """
        Add an edge to the MRV calculation lineage graph.

        Args:
            request: Lineage edge request with source/target nodes.

        Returns:
            LineageResponse with operation result.
        """
        start_time = time.monotonic()

        try:
            if self._lineage_graph_engine is None:
                raise RuntimeError("LineageGraphEngine not available")

            result = self._lineage_graph_engine.add_edge(
                source_node_id=request.source_node_id,
                target_node_id=request.target_node_id,
                edge_type=request.edge_type,
                transformation=request.transformation,
                metadata=request.metadata,
            )

            elapsed = (time.monotonic() - start_time) * 1000.0

            return LineageResponse(
                success=True,
                result=result if isinstance(result, dict) else {
                    "source": request.source_node_id,
                    "target": request.target_node_id,
                },
                processing_time_ms=elapsed,
            )

        except Exception as e:
            elapsed = (time.monotonic() - start_time) * 1000.0
            logger.error("add_lineage_edge failed: %s", e, exc_info=True)
            return LineageResponse(
                success=False,
                processing_time_ms=elapsed,
                error=str(e),
            )

    def get_lineage_graph(
        self, request: GetLineageGraphRequest,
    ) -> LineageResponse:
        """
        Retrieve the lineage graph for an organization/year.

        Args:
            request: Lineage graph query request.

        Returns:
            LineageResponse with graph data (nodes, edges, stats).
        """
        start_time = time.monotonic()

        try:
            if self._lineage_graph_engine is None:
                raise RuntimeError("LineageGraphEngine not available")

            result = self._lineage_graph_engine.get_graph(
                organization_id=request.organization_id,
                reporting_year=request.reporting_year,
                scope=request.scope,
                max_depth=request.max_depth,
            )

            elapsed = (time.monotonic() - start_time) * 1000.0

            return LineageResponse(
                success=True,
                result=result if isinstance(result, dict) else {},
                processing_time_ms=elapsed,
            )

        except Exception as e:
            elapsed = (time.monotonic() - start_time) * 1000.0
            logger.error("get_lineage_graph failed: %s", e, exc_info=True)
            return LineageResponse(
                success=False,
                processing_time_ms=elapsed,
                error=str(e),
            )

    def trace_lineage(
        self, request: TraceLineageRequest,
    ) -> LineageResponse:
        """
        Trace lineage from a node in the specified direction.

        Forward traversal answers "what downstream values are impacted?"
        Backward traversal answers "what source records contributed?"

        Args:
            request: Lineage trace request with node and direction.

        Returns:
            LineageResponse with traversal path.
        """
        start_time = time.monotonic()

        try:
            if self._lineage_graph_engine is None:
                raise RuntimeError("LineageGraphEngine not available")

            if request.direction == "forward":
                result = self._lineage_graph_engine.traverse_forward(
                    node_id=request.node_id,
                    max_depth=request.max_depth,
                )
            else:
                result = self._lineage_graph_engine.traverse_backward(
                    node_id=request.node_id,
                    max_depth=request.max_depth,
                )

            elapsed = (time.monotonic() - start_time) * 1000.0

            return LineageResponse(
                success=True,
                result=result if isinstance(result, dict) else {
                    "node_id": request.node_id,
                    "direction": request.direction,
                    "path": result,
                },
                processing_time_ms=elapsed,
            )

        except Exception as e:
            elapsed = (time.monotonic() - start_time) * 1000.0
            logger.error("trace_lineage failed: %s", e, exc_info=True)
            return LineageResponse(
                success=False,
                processing_time_ms=elapsed,
                error=str(e),
            )

    def visualize_lineage(
        self,
        organization_id: str,
        reporting_year: int,
        output_format: str = "json",
    ) -> LineageResponse:
        """
        Generate a visualization of the lineage graph.

        Args:
            organization_id: Organization identifier.
            reporting_year: Reporting year.
            output_format: Visualization format (json, dot, mermaid).

        Returns:
            LineageResponse with visualization data.
        """
        start_time = time.monotonic()

        try:
            if self._lineage_graph_engine is None:
                raise RuntimeError("LineageGraphEngine not available")

            result = self._lineage_graph_engine.visualize(
                organization_id=organization_id,
                reporting_year=reporting_year,
                output_format=output_format,
            )

            elapsed = (time.monotonic() - start_time) * 1000.0

            return LineageResponse(
                success=True,
                result=result if isinstance(result, dict) else {
                    "format": output_format,
                    "data": result,
                },
                processing_time_ms=elapsed,
            )

        except Exception as e:
            elapsed = (time.monotonic() - start_time) * 1000.0
            logger.error("visualize_lineage failed: %s", e, exc_info=True)
            return LineageResponse(
                success=False,
                processing_time_ms=elapsed,
                error=str(e),
            )

    # ========================================================================
    # Evidence Methods (Engine 3: EvidencePackagerEngine)
    # ========================================================================

    def create_evidence_package(
        self, request: CreateEvidencePackageRequest,
    ) -> EvidencePackageResponse:
        """
        Create an audit evidence package for third-party verification.

        Bundles audit events, lineage data, and compliance information
        into a signed evidence package.

        Args:
            request: Evidence package creation request.

        Returns:
            EvidencePackageResponse with package details.
        """
        start_time = time.monotonic()
        package_id = f"ep-{uuid4().hex[:12]}"

        try:
            if self._evidence_packager_engine is None:
                raise RuntimeError("EvidencePackagerEngine not available")

            result = self._evidence_packager_engine.create_package(
                organization_id=request.organization_id,
                reporting_year=request.reporting_year,
                scope=request.scope,
                assurance_level=request.assurance_level,
                include_lineage=request.include_lineage,
                include_chain=request.include_chain,
                description=request.description,
            )

            elapsed = (time.monotonic() - start_time) * 1000.0

            pkg_id = result.get("package_id", package_id) if isinstance(result, dict) else package_id
            pkg_hash = result.get("package_hash", "") if isinstance(result, dict) else ""

            return EvidencePackageResponse(
                success=True,
                package_id=pkg_id,
                organization_id=request.organization_id,
                reporting_year=request.reporting_year,
                assurance_level=request.assurance_level,
                package_hash=pkg_hash,
                result=result if isinstance(result, dict) else {},
                processing_time_ms=elapsed,
            )

        except Exception as e:
            elapsed = (time.monotonic() - start_time) * 1000.0
            logger.error("create_evidence_package failed: %s", e, exc_info=True)
            return EvidencePackageResponse(
                success=False,
                package_id=None,
                organization_id=request.organization_id,
                reporting_year=request.reporting_year,
                processing_time_ms=elapsed,
                error=str(e),
            )

    def get_evidence_package(
        self, package_id: str,
    ) -> EvidencePackageResponse:
        """
        Retrieve an existing evidence package.

        Args:
            package_id: Evidence package identifier.

        Returns:
            EvidencePackageResponse with package data.
        """
        start_time = time.monotonic()

        try:
            if self._evidence_packager_engine is None:
                raise RuntimeError("EvidencePackagerEngine not available")

            result = self._evidence_packager_engine.get_package(
                package_id=package_id,
            )

            elapsed = (time.monotonic() - start_time) * 1000.0

            org_id = result.get("organization_id", "") if isinstance(result, dict) else ""
            year = result.get("reporting_year", 0) if isinstance(result, dict) else 0

            return EvidencePackageResponse(
                success=True,
                package_id=package_id,
                organization_id=org_id,
                reporting_year=year,
                result=result if isinstance(result, dict) else {},
                processing_time_ms=elapsed,
            )

        except Exception as e:
            elapsed = (time.monotonic() - start_time) * 1000.0
            logger.error("get_evidence_package failed: %s", e, exc_info=True)
            return EvidencePackageResponse(
                success=False,
                package_id=package_id,
                organization_id="",
                reporting_year=0,
                processing_time_ms=elapsed,
                error=str(e),
            )

    def sign_evidence_package(
        self, package_id: str,
    ) -> EvidencePackageResponse:
        """
        Apply a digital signature to an evidence package.

        Args:
            package_id: Evidence package identifier.

        Returns:
            EvidencePackageResponse with signature details.
        """
        start_time = time.monotonic()

        try:
            if self._evidence_packager_engine is None:
                raise RuntimeError("EvidencePackagerEngine not available")

            result = self._evidence_packager_engine.sign_package(
                package_id=package_id,
            )

            elapsed = (time.monotonic() - start_time) * 1000.0

            return EvidencePackageResponse(
                success=True,
                package_id=package_id,
                organization_id=result.get("organization_id", "") if isinstance(result, dict) else "",
                reporting_year=result.get("reporting_year", 0) if isinstance(result, dict) else 0,
                package_hash=result.get("signature_hash", "") if isinstance(result, dict) else "",
                result=result if isinstance(result, dict) else {},
                processing_time_ms=elapsed,
            )

        except Exception as e:
            elapsed = (time.monotonic() - start_time) * 1000.0
            logger.error("sign_evidence_package failed: %s", e, exc_info=True)
            return EvidencePackageResponse(
                success=False,
                package_id=package_id,
                organization_id="",
                reporting_year=0,
                processing_time_ms=elapsed,
                error=str(e),
            )

    def verify_evidence_package(
        self, package_id: str,
    ) -> EvidencePackageResponse:
        """
        Verify the integrity and signature of an evidence package.

        Args:
            package_id: Evidence package identifier.

        Returns:
            EvidencePackageResponse with verification result.
        """
        start_time = time.monotonic()

        try:
            if self._evidence_packager_engine is None:
                raise RuntimeError("EvidencePackagerEngine not available")

            result = self._evidence_packager_engine.verify_package(
                package_id=package_id,
            )

            elapsed = (time.monotonic() - start_time) * 1000.0

            return EvidencePackageResponse(
                success=True,
                package_id=package_id,
                organization_id=result.get("organization_id", "") if isinstance(result, dict) else "",
                reporting_year=result.get("reporting_year", 0) if isinstance(result, dict) else 0,
                result=result if isinstance(result, dict) else {},
                processing_time_ms=elapsed,
            )

        except Exception as e:
            elapsed = (time.monotonic() - start_time) * 1000.0
            logger.error("verify_evidence_package failed: %s", e, exc_info=True)
            return EvidencePackageResponse(
                success=False,
                package_id=package_id,
                organization_id="",
                reporting_year=0,
                processing_time_ms=elapsed,
                error=str(e),
            )

    # ========================================================================
    # Compliance Tracing Methods (Engine 4: ComplianceTracerEngine)
    # ========================================================================

    def trace_compliance(
        self, request: TraceComplianceRequest,
    ) -> ComplianceTraceResponse:
        """
        Trace regulatory compliance for a specific framework.

        Maps audit events and lineage to framework-specific requirements
        and identifies coverage gaps.

        Args:
            request: Compliance traceability request.

        Returns:
            ComplianceTraceResponse with coverage and gaps.
        """
        start_time = time.monotonic()

        try:
            if self._compliance_tracer_engine is None:
                raise RuntimeError("ComplianceTracerEngine not available")

            result = self._compliance_tracer_engine.trace_compliance(
                organization_id=request.organization_id,
                reporting_year=request.reporting_year,
                framework=request.framework,
            )

            elapsed = (time.monotonic() - start_time) * 1000.0

            return ComplianceTraceResponse(
                success=True,
                framework=request.framework,
                result=result if isinstance(result, dict) else {},
                processing_time_ms=elapsed,
            )

        except Exception as e:
            elapsed = (time.monotonic() - start_time) * 1000.0
            logger.error("trace_compliance failed: %s", e, exc_info=True)
            return ComplianceTraceResponse(
                success=False,
                framework=request.framework,
                processing_time_ms=elapsed,
                error=str(e),
            )

    def get_compliance_coverage(
        self,
        organization_id: str,
        reporting_year: int,
    ) -> ComplianceTraceResponse:
        """
        Get compliance coverage across all supported frameworks.

        Args:
            organization_id: Organization identifier.
            reporting_year: Reporting year.

        Returns:
            ComplianceTraceResponse with coverage summary.
        """
        start_time = time.monotonic()

        try:
            if self._compliance_tracer_engine is None:
                raise RuntimeError("ComplianceTracerEngine not available")

            result = self._compliance_tracer_engine.get_coverage(
                organization_id=organization_id,
                reporting_year=reporting_year,
            )

            elapsed = (time.monotonic() - start_time) * 1000.0

            return ComplianceTraceResponse(
                success=True,
                framework="ALL",
                coverage=result if isinstance(result, dict) else {},
                processing_time_ms=elapsed,
            )

        except Exception as e:
            elapsed = (time.monotonic() - start_time) * 1000.0
            logger.error("get_compliance_coverage failed: %s", e, exc_info=True)
            return ComplianceTraceResponse(
                success=False,
                framework="ALL",
                processing_time_ms=elapsed,
                error=str(e),
            )

    # ========================================================================
    # Change Detection Methods (Engine 5: ChangeDetectorEngine)
    # ========================================================================

    def detect_change(
        self, request: DetectChangeRequest,
    ) -> ChangeDetectionResponse:
        """
        Detect changes between calculation versions.

        Compares previous and current versions of a calculation to
        identify material vs. immaterial changes.

        Args:
            request: Change detection request.

        Returns:
            ChangeDetectionResponse with change details.
        """
        start_time = time.monotonic()

        try:
            if self._change_detector_engine is None:
                raise RuntimeError("ChangeDetectorEngine not available")

            result = self._change_detector_engine.detect_change(
                calculation_id=request.calculation_id,
                organization_id=request.organization_id,
                reporting_year=request.reporting_year,
                previous_version=request.previous_version,
                current_version=request.current_version,
            )

            elapsed = (time.monotonic() - start_time) * 1000.0
            result_dict = result if isinstance(result, dict) else {}

            return ChangeDetectionResponse(
                success=True,
                calculation_id=request.calculation_id,
                changes_detected=result_dict.get("changes_detected", False),
                change_count=result_dict.get("change_count", 0),
                materiality_status=result_dict.get("materiality_status"),
                result=result_dict,
                processing_time_ms=elapsed,
            )

        except Exception as e:
            elapsed = (time.monotonic() - start_time) * 1000.0
            logger.error("detect_change failed: %s", e, exc_info=True)
            return ChangeDetectionResponse(
                success=False,
                calculation_id=request.calculation_id,
                processing_time_ms=elapsed,
                error=str(e),
            )

    def get_change(self, change_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific change record.

        Args:
            change_id: Change record identifier.

        Returns:
            Change record dictionary or None.
        """
        if self._change_detector_engine is None:
            return None
        try:
            return self._change_detector_engine.get_change(
                change_id=change_id,
            )
        except Exception as e:
            logger.error("get_change failed: %s", e, exc_info=True)
            return None

    def get_change_impact(
        self, calculation_id: str,
    ) -> ChangeDetectionResponse:
        """
        Analyze the downstream impact of changes to a calculation.

        Args:
            calculation_id: Calculation identifier.

        Returns:
            ChangeDetectionResponse with impact analysis.
        """
        start_time = time.monotonic()

        try:
            if self._change_detector_engine is None:
                raise RuntimeError("ChangeDetectorEngine not available")

            result = self._change_detector_engine.analyze_impact(
                calculation_id=calculation_id,
            )

            elapsed = (time.monotonic() - start_time) * 1000.0
            result_dict = result if isinstance(result, dict) else {}

            return ChangeDetectionResponse(
                success=True,
                calculation_id=calculation_id,
                changes_detected=result_dict.get("has_impact", False),
                change_count=result_dict.get("impacted_count", 0),
                materiality_status=result_dict.get("materiality_status"),
                result=result_dict,
                processing_time_ms=elapsed,
            )

        except Exception as e:
            elapsed = (time.monotonic() - start_time) * 1000.0
            logger.error("get_change_impact failed: %s", e, exc_info=True)
            return ChangeDetectionResponse(
                success=False,
                calculation_id=calculation_id,
                processing_time_ms=elapsed,
                error=str(e),
            )

    # ========================================================================
    # Compliance Checking Methods (Engine 6: ComplianceCheckerEngine)
    # ========================================================================

    def check_compliance(
        self, request: ComplianceCheckRequest,
    ) -> ComplianceCheckResponse:
        """
        Run compliance checks against specified frameworks.

        Validates that the audit trail and lineage meet framework-specific
        requirements for audit trail completeness, data quality, and
        provenance tracking.

        Args:
            request: Compliance check request.

        Returns:
            ComplianceCheckResponse with per-framework results.
        """
        start_time = time.monotonic()

        try:
            if self._compliance_checker_engine is not None:
                result = self._compliance_checker_engine.check_compliance(
                    organization_id=request.organization_id,
                    reporting_year=request.reporting_year,
                    frameworks=request.frameworks,
                )

                elapsed = (time.monotonic() - start_time) * 1000.0

                if isinstance(result, dict):
                    return ComplianceCheckResponse(
                        success=True,
                        overall_status=result.get("overall_status", "UNKNOWN"),
                        overall_score=float(result.get("overall_score", 0.0)),
                        framework_results=result.get("framework_results", []),
                        recommendations=result.get("recommendations", []),
                        checked_at=datetime.now(timezone.utc),
                        processing_time_ms=elapsed,
                    )

                # If result is a list of per-framework results
                framework_results = []
                if isinstance(result, list):
                    for cr in result:
                        if hasattr(cr, "framework"):
                            framework_results.append({
                                "framework": cr.framework.value if hasattr(cr.framework, "value") else str(cr.framework),
                                "status": cr.status.value if hasattr(cr.status, "value") else str(cr.status),
                                "score": float(cr.score) if hasattr(cr, "score") else 0.0,
                                "findings": cr.findings if hasattr(cr, "findings") else [],
                                "recommendations": cr.recommendations if hasattr(cr, "recommendations") else [],
                            })
                        elif isinstance(cr, dict):
                            framework_results.append(cr)

                overall_status = "PASS"
                if any(fr.get("status") == "FAIL" for fr in framework_results):
                    overall_status = "FAIL"
                elif any(fr.get("status") == "WARNING" for fr in framework_results):
                    overall_status = "WARNING"

                scores = [fr.get("score", 0.0) for fr in framework_results if fr.get("score") is not None]
                overall_score = sum(scores) / len(scores) if scores else 0.0

                return ComplianceCheckResponse(
                    success=True,
                    overall_status=overall_status,
                    overall_score=overall_score,
                    framework_results=framework_results,
                    checked_at=datetime.now(timezone.utc),
                    processing_time_ms=elapsed,
                )

            # Inline fallback when engine is not available
            elapsed = (time.monotonic() - start_time) * 1000.0
            return ComplianceCheckResponse(
                success=True,
                overall_status="WARNING",
                overall_score=50.0,
                framework_results=[{
                    "framework": "inline",
                    "status": "WARNING",
                    "findings": ["ComplianceCheckerEngine not available"],
                }],
                checked_at=datetime.now(timezone.utc),
                processing_time_ms=elapsed,
            )

        except Exception as e:
            elapsed = (time.monotonic() - start_time) * 1000.0
            logger.error("check_compliance failed: %s", e, exc_info=True)
            return ComplianceCheckResponse(
                success=False,
                overall_status="FAIL",
                overall_score=0.0,
                framework_results=[],
                checked_at=datetime.now(timezone.utc),
                processing_time_ms=elapsed,
            )

    # ========================================================================
    # Pipeline Methods (Engine 7: AuditTrailPipelineEngine)
    # ========================================================================

    def execute_pipeline(
        self, request: ExecutePipelineRequest,
    ) -> PipelineResponse:
        """
        Execute the full 10-stage audit trail pipeline.

        The pipeline orchestrates all engines in sequence:
        validation, event collection, lineage construction, evidence
        packaging, compliance tracing, change detection, compliance
        checking, summary generation, provenance sealing, and output.

        Args:
            request: Pipeline execution request.

        Returns:
            PipelineResponse with execution results.
        """
        start_time = time.monotonic()

        try:
            if self._pipeline_engine is None:
                raise RuntimeError("AuditTrailPipelineEngine not available")

            result = self._pipeline_engine.execute(
                organization_id=request.organization_id,
                reporting_year=request.reporting_year,
                scope=request.scope,
                frameworks=request.frameworks,
                include_evidence=request.include_evidence,
            )

            elapsed = (time.monotonic() - start_time) * 1000.0
            result_dict = result if isinstance(result, dict) else {}

            return PipelineResponse(
                success=True,
                organization_id=request.organization_id,
                reporting_year=request.reporting_year,
                status=result_dict.get("status", "COMPLETED"),
                stages_completed=result_dict.get("stages_completed", 10),
                total_stages=10,
                result=result_dict,
                provenance_hash=result_dict.get("provenance_hash", ""),
                processing_time_ms=elapsed,
            )

        except Exception as e:
            elapsed = (time.monotonic() - start_time) * 1000.0
            logger.error("execute_pipeline failed: %s", e, exc_info=True)
            return PipelineResponse(
                success=False,
                organization_id=request.organization_id,
                reporting_year=request.reporting_year,
                status="FAILED",
                processing_time_ms=elapsed,
                error=str(e),
            )

    def execute_pipeline_batch(
        self, request: ExecutePipelineBatchRequest,
    ) -> PipelineBatchResponse:
        """
        Execute multiple pipeline runs in batch.

        Args:
            request: Batch pipeline execution request.

        Returns:
            PipelineBatchResponse with individual results.
        """
        start_time = time.monotonic()
        results: List[PipelineResponse] = []
        errors: List[dict] = []

        for idx, pipeline_req in enumerate(request.requests):
            resp = self.execute_pipeline(pipeline_req)
            results.append(resp)
            if not resp.success:
                errors.append({
                    "index": idx,
                    "organization_id": pipeline_req.organization_id,
                    "reporting_year": pipeline_req.reporting_year,
                    "error": resp.error,
                })

        successful = sum(1 for r in results if r.success)
        elapsed = (time.monotonic() - start_time) * 1000.0

        return PipelineBatchResponse(
            success=len(errors) == 0,
            total_requests=len(request.requests),
            successful=successful,
            failed=len(errors),
            results=results,
            errors=errors,
            processing_time_ms=elapsed,
        )

    # ========================================================================
    # Summary
    # ========================================================================

    def get_audit_trail_summary(
        self,
        organization_id: str,
        reporting_year: int,
    ) -> AuditTrailSummaryResponse:
        """
        Get a comprehensive audit trail summary across all engines.

        Aggregates event statistics, chain status, lineage graph metrics,
        evidence package counts, compliance coverage, and change records
        into a single summary view.

        Args:
            organization_id: Organization identifier.
            reporting_year: Reporting year.

        Returns:
            AuditTrailSummaryResponse with aggregated summary.
        """
        start_time = time.monotonic()

        total_events = 0
        chain_length = 0
        chain_valid: Optional[bool] = None
        lineage_nodes = 0
        lineage_edges = 0
        evidence_packages = 0
        compliance_coverage: Optional[Dict[str, Any]] = None
        changes_detected = 0

        # Gather event statistics from AuditEventEngine
        if self._audit_event_engine is not None:
            try:
                stats = self._audit_event_engine.get_event_statistics(
                    organization_id=organization_id,
                    reporting_year=reporting_year,
                )
                total_events = stats.get("total_events", 0)
                chain_length = total_events

                chain_result = self._audit_event_engine.verify_chain(
                    organization_id=organization_id,
                    reporting_year=reporting_year,
                )
                chain_valid = chain_result.get("valid")
            except Exception as e:
                logger.warning("Failed to get event statistics: %s", e)

        # Gather lineage graph metrics from LineageGraphEngine
        if self._lineage_graph_engine is not None:
            try:
                graph = self._lineage_graph_engine.get_graph(
                    organization_id=organization_id,
                    reporting_year=reporting_year,
                )
                if isinstance(graph, dict):
                    lineage_nodes = graph.get("node_count", 0)
                    lineage_edges = graph.get("edge_count", 0)
            except Exception as e:
                logger.warning("Failed to get lineage metrics: %s", e)

        # Gather compliance coverage from ComplianceTracerEngine
        if self._compliance_tracer_engine is not None:
            try:
                coverage = self._compliance_tracer_engine.get_coverage(
                    organization_id=organization_id,
                    reporting_year=reporting_year,
                )
                if isinstance(coverage, dict):
                    compliance_coverage = coverage
            except Exception as e:
                logger.warning("Failed to get compliance coverage: %s", e)

        # Compute provenance hash for the summary
        provenance_input = (
            f"{organization_id}:{reporting_year}:"
            f"{total_events}:{chain_valid}:"
            f"{lineage_nodes}:{lineage_edges}"
        )
        provenance_hash = hashlib.sha256(
            provenance_input.encode("utf-8")
        ).hexdigest()

        elapsed = (time.monotonic() - start_time) * 1000.0

        return AuditTrailSummaryResponse(
            success=True,
            organization_id=organization_id,
            reporting_year=reporting_year,
            total_events=total_events,
            chain_length=chain_length,
            chain_valid=chain_valid,
            lineage_nodes=lineage_nodes,
            lineage_edges=lineage_edges,
            evidence_packages=evidence_packages,
            compliance_coverage=compliance_coverage,
            changes_detected=changes_detected,
            provenance_hash=provenance_hash,
            processing_time_ms=elapsed,
        )

    # ========================================================================
    # Health and Status
    # ========================================================================

    def health_check(self) -> HealthResponse:
        """
        Perform service health check.

        Checks the availability of all 7 engines and reports the
        overall service status as healthy, degraded, or unhealthy.

        Returns:
            HealthResponse with engine statuses and uptime.
        """
        uptime = (
            datetime.now(timezone.utc) - self._start_time
        ).total_seconds()

        engines_status = {
            "audit_event": self._audit_event_engine is not None,
            "lineage_graph": self._lineage_graph_engine is not None,
            "evidence_packager": self._evidence_packager_engine is not None,
            "compliance_tracer": self._compliance_tracer_engine is not None,
            "change_detector": self._change_detector_engine is not None,
            "compliance_checker": self._compliance_checker_engine is not None,
            "pipeline": self._pipeline_engine is not None,
        }

        all_healthy = all(engines_status.values())
        any_healthy = any(engines_status.values())

        if all_healthy:
            status = "healthy"
        elif any_healthy:
            status = "degraded"
        else:
            status = "unhealthy"

        return HealthResponse(
            status=status,
            version="1.0.0",
            agent_id="GL-MRV-X-042",
            engines_status=engines_status,
            uptime_seconds=uptime,
        )


# ============================================================================
# Module-Level Helpers
# ============================================================================


def get_service() -> AuditTrailLineageService:
    """
    Get singleton AuditTrailLineageService instance.

    Thread-safe via double-checked locking with RLock.

    Returns:
        AuditTrailLineageService singleton instance.

    Example:
        >>> service = get_service()
        >>> service.health_check().status
        'healthy'
    """
    global _service_instance
    if _service_instance is None:
        with _service_lock:
            if _service_instance is None:
                _service_instance = AuditTrailLineageService()
    return _service_instance


def get_router():
    """
    Get the FastAPI router for audit trail lineage endpoints.

    Returns:
        FastAPI APIRouter instance, or None if router is unavailable.

    Example:
        >>> router = get_router()
        >>> app.include_router(router, prefix="/api/v1/audit-trail-lineage")
    """
    try:
        from greenlang.agents.mrv.audit_trail_lineage.api.router import router
        return router
    except ImportError:
        logger.warning("Audit trail lineage router not available (ImportError)")
        return None
    except Exception as e:
        logger.warning("Failed to load router: %s", e)
        return None


def create_app():
    """
    Create a standalone FastAPI application for testing.

    Creates a FastAPI application with audit trail lineage routes
    registered. Used for development and integration testing.

    Returns:
        FastAPI application instance, or None if FastAPI is unavailable.

    Example:
        >>> app = create_app()
        >>> # use with TestClient(app) for testing
    """
    try:
        from fastapi import FastAPI

        app = FastAPI(
            title="GreenLang Audit Trail & Lineage Service",
            description=(
                "Cross-Cutting MRV -- Immutable Audit Trails & "
                "Calculation Lineage (AGENT-MRV-030)"
            ),
            version="1.0.0",
        )

        try:
            router = get_router()
            if router is not None:
                app.include_router(
                    router,
                    prefix="/api/v1/audit-trail-lineage",
                    tags=["audit-trail-lineage"],
                )
        except ImportError:
            logger.warning("Audit trail lineage router not available for test app")

        @app.get("/health")
        def health():
            """Health check endpoint."""
            service = get_service()
            return service.health_check().dict()

        return app

    except ImportError:
        logger.warning("FastAPI not available; create_app() returns None")
        return None


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    # Service
    "AuditTrailLineageService",
    "get_service",
    "get_router",
    "create_app",
    # Request models
    "RecordEventRequest",
    "RecordBatchRequest",
    "GetEventsRequest",
    "VerifyChainRequest",
    "AddLineageNodeRequest",
    "AddLineageEdgeRequest",
    "GetLineageGraphRequest",
    "TraceLineageRequest",
    "CreateEvidencePackageRequest",
    "TraceComplianceRequest",
    "DetectChangeRequest",
    "ExecutePipelineRequest",
    "ExecutePipelineBatchRequest",
    "ComplianceCheckRequest",
    # Response models
    "RecordEventResponse",
    "RecordBatchResponse",
    "GetEventsResponse",
    "VerifyChainResponse",
    "LineageResponse",
    "EvidencePackageResponse",
    "ComplianceTraceResponse",
    "ChangeDetectionResponse",
    "PipelineResponse",
    "PipelineBatchResponse",
    "ComplianceCheckResponse",
    "AuditTrailSummaryResponse",
    "HealthResponse",
]
