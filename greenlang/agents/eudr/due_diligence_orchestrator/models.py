# -*- coding: utf-8 -*-
"""
Due Diligence Orchestrator Data Models - AGENT-EUDR-026

Pydantic v2 data models for the Due Diligence Orchestrator Agent covering
DAG-based workflow definition, three-phase due diligence orchestration
(information gathering, risk assessment, risk mitigation), quality gate
enforcement, workflow state management with checkpointing, parallel
execution scheduling, error recovery with circuit breaker, and audit-ready
due diligence package generation for EUDR DDS submission.

Every model is designed for deterministic serialization and SHA-256
provenance hashing to ensure zero-hallucination, bit-perfect
reproducibility across all workflow orchestration operations per
EU 2023/1115 Articles 4, 8, 9, 10, 11, 12, and 13.

Enumerations (10):
    - DueDiligencePhase, WorkflowType, AgentExecutionStatus,
      WorkflowStatus, QualityGateResultEnum, CircuitBreakerState,
      ErrorClassification, FallbackStrategy, EUDRCommodity,
      QualityGateId

Core Models (18):
    - AgentNode, WorkflowEdge, WorkflowDefinition, AgentExecutionRecord,
      WorkflowCheckpoint, QualityGateCheck, QualityGateEvaluation,
      WorkflowStateTransition, WorkflowState, RiskScoreContribution,
      CompositeRiskProfile, MitigationDecision, CircuitBreakerRecord,
      RetryRecord, DeadLetterEntry, DDSField, DDSSection,
      DueDiligencePackage

Request Models (6):
    - CreateWorkflowRequest, StartWorkflowRequest, ResumeWorkflowRequest,
      EvaluateQualityGateRequest, GeneratePackageRequest,
      BatchWorkflowRequest

Response Models (6):
    - WorkflowStatusResponse, WorkflowProgressResponse,
      QualityGateResponse, PackageGenerationResponse,
      WorkflowAuditTrailResponse, BatchWorkflowResponse

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-026 Due Diligence Orchestrator (GL-EUDR-DDO-026)
Status: Production Ready
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional
from greenlang.schemas import GreenLangBase, utcnow

from pydantic import (
    Field,
    field_validator,
    model_validator,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Return a new UUID4 string."""
    return str(uuid.uuid4())

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Service version string.
VERSION: str = "1.0.0"

#: Total number of EUDR agents orchestrated.
TOTAL_EUDR_AGENTS: int = 25

#: Phase 1 agent IDs (Supply Chain Traceability).
PHASE_1_AGENTS: List[str] = [
    f"EUDR-{i:03d}" for i in range(1, 16)
]

#: Phase 2 agent IDs (Risk Assessment).
PHASE_2_AGENTS: List[str] = [
    f"EUDR-{i:03d}" for i in range(16, 26)
]

#: Phase 3 agent ID (Risk Mitigation via EUDR-025).
PHASE_3_AGENTS: List[str] = ["EUDR-025"]

#: All EUDR agent IDs.
ALL_EUDR_AGENTS: List[str] = [
    f"EUDR-{i:03d}" for i in range(1, 26)
]

#: EUDR Article 31 data retention in years.
EUDR_RETENTION_YEARS: int = 5

#: Maximum agents in a single workflow.
MAX_WORKFLOW_AGENTS: int = 50

#: EUDR-regulated commodities.
SUPPORTED_COMMODITIES: List[str] = [
    "cattle", "cocoa", "coffee", "palm_oil", "rubber", "soya", "wood",
]

#: Agent human-readable name mapping.
AGENT_NAMES: Dict[str, str] = {
    "EUDR-001": "Supply Chain Mapping Master",
    "EUDR-002": "Geolocation Verification",
    "EUDR-003": "Satellite Monitoring",
    "EUDR-004": "Forest Cover Analysis",
    "EUDR-005": "Land Use Change Detector",
    "EUDR-006": "Plot Boundary Manager",
    "EUDR-007": "GPS Coordinate Validator",
    "EUDR-008": "Multi-Tier Supplier Tracker",
    "EUDR-009": "Chain of Custody",
    "EUDR-010": "Segregation Verifier",
    "EUDR-011": "Mass Balance Calculator",
    "EUDR-012": "Document Authentication",
    "EUDR-013": "Blockchain Integration",
    "EUDR-014": "QR Code Generator",
    "EUDR-015": "Mobile Data Collector",
    "EUDR-016": "Country Risk Evaluator",
    "EUDR-017": "Supplier Risk Scorer",
    "EUDR-018": "Commodity Risk Analyzer",
    "EUDR-019": "Corruption Index Monitor",
    "EUDR-020": "Deforestation Alert System",
    "EUDR-021": "Indigenous Rights Checker",
    "EUDR-022": "Protected Area Validator",
    "EUDR-023": "Legal Compliance Verifier",
    "EUDR-024": "Third-Party Audit Manager",
    "EUDR-025": "Risk Mitigation Advisor",
}

# ---------------------------------------------------------------------------
# Enumerations (10)
# ---------------------------------------------------------------------------

class DueDiligencePhase(str, Enum):
    """EUDR due diligence phases per Article 8.

    Maps to the three mandatory sequential phases plus package generation.
    """

    INFORMATION_GATHERING = "information_gathering"
    """Phase 1: Article 9 -- collect supply chain data."""

    RISK_ASSESSMENT = "risk_assessment"
    """Phase 2: Article 10 -- assess and identify risk."""

    RISK_MITIGATION = "risk_mitigation"
    """Phase 3: Article 11 -- adopt risk mitigation measures."""

    PACKAGE_GENERATION = "package_generation"
    """Phase 4: Article 12 -- compile DDS evidence package."""

class WorkflowType(str, Enum):
    """Type of due diligence workflow.

    Determines agent topology and quality gate thresholds.
    """

    STANDARD = "standard"
    """Full 25-agent due diligence per Article 8."""

    SIMPLIFIED = "simplified"
    """Reduced agent set per Article 13 for low-risk origins."""

    CUSTOM = "custom"
    """Operator-customized workflow with selected agents."""

class AgentExecutionStatus(str, Enum):
    """Execution status of an individual agent within a workflow."""

    PENDING = "pending"
    """Not yet started, waiting for dependencies."""

    QUEUED = "queued"
    """In execution queue, ready to start."""

    RUNNING = "running"
    """Currently executing."""

    COMPLETED = "completed"
    """Finished successfully with output."""

    FAILED = "failed"
    """Failed, may be retried."""

    RETRYING = "retrying"
    """Retrying after a transient failure."""

    SKIPPED = "skipped"
    """Skipped -- not applicable or degraded mode."""

    CIRCUIT_BROKEN = "circuit_broken"
    """Circuit breaker open, rejecting calls."""

    TIMED_OUT = "timed_out"
    """Exceeded timeout limit."""

class WorkflowStatus(str, Enum):
    """Overall status of a due diligence workflow."""

    CREATED = "created"
    """Workflow defined but not started."""

    VALIDATING = "validating"
    """Validating workflow definition before start."""

    RUNNING = "running"
    """Actively executing agents."""

    PAUSED = "paused"
    """Execution paused by user or system."""

    QUALITY_GATE = "quality_gate"
    """Evaluating a quality gate between phases."""

    GATE_FAILED = "gate_failed"
    """Quality gate failed, awaiting remediation."""

    RESUMING = "resuming"
    """Resuming from checkpoint."""

    COMPLETING = "completing"
    """Finalizing workflow (package generation)."""

    COMPLETED = "completed"
    """All phases complete, package generated."""

    CANCELLED = "cancelled"
    """Cancelled by user."""

    TERMINATED = "terminated"
    """Unrecoverable failure after max retries."""

class QualityGateResultEnum(str, Enum):
    """Result of a quality gate evaluation."""

    PASSED = "passed"
    """All checks passed the threshold."""

    FAILED = "failed"
    """One or more checks below threshold."""

    OVERRIDDEN = "overridden"
    """Failed but manually overridden with justification."""

    PENDING = "pending"
    """Not yet evaluated."""

class QualityGateId(str, Enum):
    """Identifier for the three quality gates."""

    QG1 = "QG-1"
    """Information Gathering completeness gate (Art. 9 -> Art. 10)."""

    QG2 = "QG-2"
    """Risk Assessment coverage gate (Art. 10 -> Art. 11)."""

    QG3 = "QG-3"
    """Mitigation Adequacy gate (Art. 11 -> Art. 12)."""

class CircuitBreakerState(str, Enum):
    """State of a circuit breaker for an agent."""

    CLOSED = "closed"
    """Normal operation, allowing calls."""

    OPEN = "open"
    """Failing, rejecting all calls."""

    HALF_OPEN = "half_open"
    """Testing recovery with a single probe call."""

class ErrorClassification(str, Enum):
    """Classification of agent execution errors."""

    TRANSIENT = "transient"
    """Temporary failure, safe to retry (HTTP 429, 503, timeout)."""

    PERMANENT = "permanent"
    """Permanent failure, do not retry (HTTP 400, 401, 403)."""

    DEGRADED = "degraded"
    """Partial result available, use fallback."""

    UNKNOWN = "unknown"
    """Unclassified error."""

class FallbackStrategy(str, Enum):
    """Fallback strategy when an agent fails permanently."""

    CACHED_RESULT = "cached_result"
    """Use last known good cached output."""

    DEGRADED_MODE = "degraded_mode"
    """Skip non-critical agent, proceed with warning."""

    MANUAL_OVERRIDE = "manual_override"
    """Pause workflow for human decision."""

    FAIL = "fail"
    """No fallback, fail the workflow step."""

class EUDRCommodity(str, Enum):
    """EUDR-regulated forest-risk commodities per Article 1."""

    CATTLE = "cattle"
    COCOA = "cocoa"
    COFFEE = "coffee"
    PALM_OIL = "palm_oil"
    RUBBER = "rubber"
    SOYA = "soya"
    WOOD = "wood"

# ---------------------------------------------------------------------------
# Core Models (18)
# ---------------------------------------------------------------------------

class AgentNode(GreenLangBase):
    """A single agent node in the workflow DAG.

    Represents one of the 25 EUDR agents as a vertex in the directed
    acyclic graph with execution metadata.

    Attributes:
        agent_id: EUDR agent identifier (e.g. "EUDR-001").
        name: Human-readable agent name.
        phase: Due diligence phase this agent belongs to.
        layer: Execution layer for parallelization (0 = entry point).
        is_critical: Whether this agent is on the critical path.
        is_required: Whether this agent is required (vs optional for commodity).
        timeout_s: Per-agent timeout override in seconds.
        retry_max: Per-agent max retry override.
        fallback: Fallback strategy on permanent failure.
        metadata: Additional agent-specific metadata.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
    )

    agent_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="EUDR agent identifier (e.g. EUDR-001)",
    )
    name: str = Field(
        "",
        max_length=255,
        description="Human-readable agent name",
    )
    phase: DueDiligencePhase = Field(
        ...,
        description="Due diligence phase this agent belongs to",
    )
    layer: int = Field(
        0,
        ge=0,
        le=20,
        description="Execution layer for parallelization",
    )
    is_critical: bool = Field(
        True,
        description="Whether this agent is on the critical path",
    )
    is_required: bool = Field(
        True,
        description="Whether this agent is required for this workflow",
    )
    timeout_s: Optional[int] = Field(
        None,
        ge=10,
        le=3600,
        description="Per-agent timeout override in seconds",
    )
    retry_max: Optional[int] = Field(
        None,
        ge=0,
        le=10,
        description="Per-agent max retry override",
    )
    fallback: FallbackStrategy = Field(
        FallbackStrategy.FAIL,
        description="Fallback strategy on permanent failure",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional agent-specific metadata",
    )

class WorkflowEdge(GreenLangBase):
    """A dependency edge in the workflow DAG.

    Defines a directed edge from source agent to target agent,
    meaning target cannot start until source completes.

    Attributes:
        source: Source agent ID (must complete first).
        target: Target agent ID (depends on source).
        data_flow: Description of data passed along this edge.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    source: str = Field(
        ...,
        min_length=1,
        description="Source agent ID (must complete first)",
    )
    target: str = Field(
        ...,
        min_length=1,
        description="Target agent ID (depends on source)",
    )
    data_flow: Optional[str] = Field(
        None,
        max_length=500,
        description="Description of data passed along this edge",
    )

class WorkflowDefinition(GreenLangBase):
    """Complete workflow DAG definition.

    Defines the agent topology, dependency edges, quality gates,
    and execution parameters for a due diligence workflow.

    Attributes:
        definition_id: Unique workflow definition identifier.
        name: Human-readable workflow name.
        description: Detailed workflow description.
        workflow_type: Standard, simplified, or custom.
        commodity: EUDR commodity this workflow is designed for.
        version: Workflow definition version.
        nodes: Agent nodes in the DAG.
        edges: Dependency edges between agents.
        quality_gates: Quality gate definitions between phases.
        created_at: Definition creation timestamp.
        created_by: User who created the definition.
        provenance_hash: SHA-256 hash of the definition.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
    )

    definition_id: str = Field(
        default_factory=_new_uuid,
        description="Unique workflow definition identifier",
    )
    name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Human-readable workflow name",
    )
    description: Optional[str] = Field(
        None,
        max_length=5000,
        description="Detailed workflow description",
    )
    workflow_type: WorkflowType = Field(
        WorkflowType.STANDARD,
        description="Standard, simplified, or custom",
    )
    commodity: Optional[EUDRCommodity] = Field(
        None,
        description="EUDR commodity this workflow is designed for",
    )
    version: str = Field(
        "1.0.0",
        description="Workflow definition version",
    )
    nodes: List[AgentNode] = Field(
        default_factory=list,
        description="Agent nodes in the DAG",
    )
    edges: List[WorkflowEdge] = Field(
        default_factory=list,
        description="Dependency edges between agents",
    )
    quality_gates: List[str] = Field(
        default_factory=lambda: ["QG-1", "QG-2", "QG-3"],
        description="Quality gate IDs between phases",
    )
    created_at: datetime = Field(
        default_factory=utcnow,
        description="Definition creation timestamp",
    )
    created_by: str = Field(
        "system",
        description="User who created the definition",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash of the definition",
    )

class AgentExecutionRecord(GreenLangBase):
    """Execution record for a single agent within a workflow.

    Tracks the lifecycle of an agent invocation including timing,
    status, output reference, retry count, and error details.

    Attributes:
        record_id: Unique execution record identifier.
        workflow_id: Parent workflow identifier.
        agent_id: EUDR agent identifier.
        status: Current execution status.
        started_at: Execution start timestamp.
        completed_at: Execution completion timestamp.
        duration_ms: Execution duration in milliseconds.
        retry_count: Number of retries attempted.
        output_ref: Reference to agent output (S3 key or inline).
        output_summary: Brief summary of agent output.
        error_message: Error message if failed.
        error_classification: Error type classification.
        provenance_hash: SHA-256 hash of the execution record.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
    )

    record_id: str = Field(
        default_factory=_new_uuid,
        description="Unique execution record identifier",
    )
    workflow_id: str = Field(
        ...,
        description="Parent workflow identifier",
    )
    agent_id: str = Field(
        ...,
        description="EUDR agent identifier",
    )
    status: AgentExecutionStatus = Field(
        AgentExecutionStatus.PENDING,
        description="Current execution status",
    )
    started_at: Optional[datetime] = Field(
        None,
        description="Execution start timestamp",
    )
    completed_at: Optional[datetime] = Field(
        None,
        description="Execution completion timestamp",
    )
    duration_ms: Optional[Decimal] = Field(
        None,
        ge=Decimal("0"),
        description="Execution duration in milliseconds",
    )
    retry_count: int = Field(
        0,
        ge=0,
        description="Number of retries attempted",
    )
    output_ref: Optional[str] = Field(
        None,
        description="Reference to agent output (S3 key or inline)",
    )
    output_summary: Optional[Dict[str, Any]] = Field(
        None,
        description="Brief summary of agent output",
    )
    error_message: Optional[str] = Field(
        None,
        max_length=5000,
        description="Error message if failed",
    )
    error_classification: Optional[ErrorClassification] = Field(
        None,
        description="Error type classification",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash of the execution record",
    )

class QualityGateCheck(GreenLangBase):
    """A single quality gate check within a gate evaluation.

    Represents one validation check (e.g. plot geolocation coverage)
    with its weight, measured score, threshold, and pass/fail result.

    Attributes:
        check_id: Unique check identifier.
        name: Human-readable check name.
        description: Detailed check description.
        weight: Relative weight of this check in the gate score (0-1).
        measured_value: Measured value for this check (0-1 or 0-100).
        threshold: Required threshold to pass.
        passed: Whether this check passed.
        source_agents: Agent IDs that contributed data for this check.
        remediation: Suggested remediation if check failed.
        evidence: Evidence supporting the check result.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    check_id: str = Field(
        default_factory=_new_uuid,
        description="Unique check identifier",
    )
    name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Human-readable check name",
    )
    description: Optional[str] = Field(
        None,
        max_length=2000,
        description="Detailed check description",
    )
    weight: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Relative weight of this check (0-1)",
    )
    measured_value: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Measured value for this check",
    )
    threshold: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Required threshold to pass",
    )
    passed: bool = Field(
        ...,
        description="Whether this check passed",
    )
    source_agents: List[str] = Field(
        default_factory=list,
        description="Agent IDs that contributed data",
    )
    remediation: Optional[str] = Field(
        None,
        max_length=2000,
        description="Suggested remediation if check failed",
    )
    evidence: Dict[str, Any] = Field(
        default_factory=dict,
        description="Evidence supporting the check result",
    )

class QualityGateEvaluation(GreenLangBase):
    """Complete quality gate evaluation result.

    Contains all individual checks, weighted score, pass/fail
    determination, and override information for a single gate.

    Attributes:
        evaluation_id: Unique evaluation identifier.
        workflow_id: Parent workflow identifier.
        gate_id: Quality gate identifier (QG-1, QG-2, QG-3).
        phase_from: Phase transitioning from.
        phase_to: Phase transitioning to.
        result: Overall gate result.
        weighted_score: Weighted aggregate score across all checks.
        threshold: Required threshold for passing.
        checks: Individual check results.
        override_justification: Justification if result is OVERRIDDEN.
        override_by: User who overrode the gate.
        evaluated_at: Evaluation timestamp.
        provenance_hash: SHA-256 hash of the evaluation.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
    )

    evaluation_id: str = Field(
        default_factory=_new_uuid,
        description="Unique evaluation identifier",
    )
    workflow_id: str = Field(
        ...,
        description="Parent workflow identifier",
    )
    gate_id: QualityGateId = Field(
        ...,
        description="Quality gate identifier",
    )
    phase_from: DueDiligencePhase = Field(
        ...,
        description="Phase transitioning from",
    )
    phase_to: DueDiligencePhase = Field(
        ...,
        description="Phase transitioning to",
    )
    result: QualityGateResultEnum = Field(
        QualityGateResultEnum.PENDING,
        description="Overall gate result",
    )
    weighted_score: Decimal = Field(
        Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Weighted aggregate score (0-1)",
    )
    threshold: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Required threshold for passing",
    )
    checks: List[QualityGateCheck] = Field(
        default_factory=list,
        description="Individual check results",
    )
    override_justification: Optional[str] = Field(
        None,
        max_length=5000,
        description="Justification if overridden",
    )
    override_by: Optional[str] = Field(
        None,
        description="User who overrode the gate",
    )
    evaluated_at: datetime = Field(
        default_factory=utcnow,
        description="Evaluation timestamp",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash of the evaluation",
    )

class WorkflowStateTransition(GreenLangBase):
    """A single state transition in the workflow lifecycle.

    Records movement between workflow states for the audit trail.

    Attributes:
        from_status: Previous workflow status.
        to_status: New workflow status.
        reason: Reason for the transition.
        actor: User or system that triggered the transition.
        timestamp: Transition timestamp.
        agent_id: Agent that triggered the transition (if applicable).
        metadata: Additional transition context.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    from_status: WorkflowStatus = Field(
        ...,
        description="Previous workflow status",
    )
    to_status: WorkflowStatus = Field(
        ...,
        description="New workflow status",
    )
    reason: Optional[str] = Field(
        None,
        max_length=2000,
        description="Reason for the transition",
    )
    actor: str = Field(
        "system",
        description="User or system that triggered the transition",
    )
    timestamp: datetime = Field(
        default_factory=utcnow,
        description="Transition timestamp",
    )
    agent_id: Optional[str] = Field(
        None,
        description="Agent that triggered the transition",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional transition context",
    )

class WorkflowCheckpoint(GreenLangBase):
    """Persistent checkpoint of workflow execution state.

    Captures the complete state at a point in time for resume and
    audit trail purposes. Written after every agent completion and
    quality gate evaluation.

    Attributes:
        checkpoint_id: Unique checkpoint identifier.
        workflow_id: Parent workflow identifier.
        sequence_number: Monotonic sequence within workflow.
        phase: Current due diligence phase.
        agent_id: Agent that just completed (if agent checkpoint).
        gate_id: Quality gate evaluated (if gate checkpoint).
        agent_statuses: Status of all agents at this point.
        agent_outputs: Accumulated output references from completed agents.
        quality_gate_results: Gate evaluation results so far.
        cumulative_provenance_hash: SHA-256 covering all state.
        created_at: Checkpoint creation timestamp.
        created_by: User or system that triggered checkpoint.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
    )

    checkpoint_id: str = Field(
        default_factory=_new_uuid,
        description="Unique checkpoint identifier",
    )
    workflow_id: str = Field(
        ...,
        description="Parent workflow identifier",
    )
    sequence_number: int = Field(
        ...,
        ge=0,
        description="Monotonic sequence within workflow",
    )
    phase: DueDiligencePhase = Field(
        ...,
        description="Current due diligence phase",
    )
    agent_id: Optional[str] = Field(
        None,
        description="Agent that just completed",
    )
    gate_id: Optional[str] = Field(
        None,
        description="Quality gate evaluated",
    )
    agent_statuses: Dict[str, str] = Field(
        default_factory=dict,
        description="Status of all agents at this point",
    )
    agent_outputs: Dict[str, str] = Field(
        default_factory=dict,
        description="Accumulated output references from completed agents",
    )
    quality_gate_results: Dict[str, str] = Field(
        default_factory=dict,
        description="Gate evaluation results so far",
    )
    cumulative_provenance_hash: str = Field(
        "",
        description="SHA-256 covering all accumulated state",
    )
    created_at: datetime = Field(
        default_factory=utcnow,
        description="Checkpoint creation timestamp",
    )
    created_by: str = Field(
        "system",
        description="User or system that triggered checkpoint",
    )

class WorkflowState(GreenLangBase):
    """Complete runtime state of a due diligence workflow.

    Tracks the end-to-end execution of a workflow including agent
    executions, quality gate evaluations, checkpoints, state
    transitions, timing, and provenance.

    Attributes:
        workflow_id: Unique workflow identifier.
        definition_id: Reference to the workflow definition.
        status: Current workflow status.
        workflow_type: Type of workflow (standard/simplified/custom).
        commodity: EUDR commodity being assessed.
        current_phase: Current due diligence phase.
        operator_id: Operator or organization identifier.
        operator_name: Operator name for DDS.
        product_ids: Product identifiers being assessed.
        shipment_ids: Shipment identifiers included.
        country_codes: Countries of production.
        agent_executions: Execution records for all agents.
        quality_gates: Quality gate evaluation results.
        checkpoints: Checkpoint history.
        transitions: State transition history.
        composite_risk_score: Computed composite risk score.
        mitigation_decision: Mitigation decision and outcome.
        package_id: Generated due diligence package ID.
        started_at: Workflow start timestamp.
        completed_at: Workflow completion timestamp.
        total_duration_ms: Total execution duration.
        eta_seconds: Estimated time to completion.
        progress_pct: Overall progress percentage (0-100).
        created_at: Workflow creation timestamp.
        created_by: User who created the workflow.
        provenance_hash: SHA-256 hash of the workflow state.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
    )

    workflow_id: str = Field(
        default_factory=_new_uuid,
        description="Unique workflow identifier",
    )
    definition_id: str = Field(
        ...,
        description="Reference to the workflow definition",
    )
    status: WorkflowStatus = Field(
        WorkflowStatus.CREATED,
        description="Current workflow status",
    )
    workflow_type: WorkflowType = Field(
        WorkflowType.STANDARD,
        description="Type of workflow",
    )
    commodity: Optional[EUDRCommodity] = Field(
        None,
        description="EUDR commodity being assessed",
    )
    current_phase: DueDiligencePhase = Field(
        DueDiligencePhase.INFORMATION_GATHERING,
        description="Current due diligence phase",
    )
    operator_id: Optional[str] = Field(
        None,
        description="Operator or organization identifier",
    )
    operator_name: Optional[str] = Field(
        None,
        max_length=500,
        description="Operator name for DDS",
    )
    product_ids: List[str] = Field(
        default_factory=list,
        description="Product identifiers being assessed",
    )
    shipment_ids: List[str] = Field(
        default_factory=list,
        description="Shipment identifiers included",
    )
    country_codes: List[str] = Field(
        default_factory=list,
        description="Countries of production (ISO 3166-1 alpha-2)",
    )
    agent_executions: Dict[str, AgentExecutionRecord] = Field(
        default_factory=dict,
        description="Execution records keyed by agent_id",
    )
    quality_gates: Dict[str, QualityGateEvaluation] = Field(
        default_factory=dict,
        description="Quality gate results keyed by gate_id",
    )
    checkpoints: List[WorkflowCheckpoint] = Field(
        default_factory=list,
        description="Checkpoint history",
    )
    transitions: List[WorkflowStateTransition] = Field(
        default_factory=list,
        description="State transition history",
    )
    composite_risk_score: Optional[Decimal] = Field(
        None,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Computed composite risk score",
    )
    mitigation_decision: Optional[Dict[str, Any]] = Field(
        None,
        description="Mitigation decision and outcome",
    )
    package_id: Optional[str] = Field(
        None,
        description="Generated due diligence package ID",
    )
    started_at: Optional[datetime] = Field(
        None,
        description="Workflow start timestamp",
    )
    completed_at: Optional[datetime] = Field(
        None,
        description="Workflow completion timestamp",
    )
    total_duration_ms: Optional[Decimal] = Field(
        None,
        ge=Decimal("0"),
        description="Total execution duration in milliseconds",
    )
    eta_seconds: Optional[int] = Field(
        None,
        ge=0,
        description="Estimated time to completion in seconds",
    )
    progress_pct: Decimal = Field(
        Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Overall progress percentage (0-100)",
    )
    created_at: datetime = Field(
        default_factory=utcnow,
        description="Workflow creation timestamp",
    )
    created_by: str = Field(
        "system",
        description="User who created the workflow",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash of the workflow state",
    )

class RiskScoreContribution(GreenLangBase):
    """Individual risk score contribution from a single risk agent.

    Captures the risk score from one of the 10 risk assessment agents
    along with its weight and weighted contribution to the composite.

    Attributes:
        agent_id: Risk assessment agent identifier.
        agent_name: Human-readable agent name.
        raw_score: Raw risk score from the agent (0-100).
        weight: Weight of this risk dimension.
        weighted_score: raw_score * weight.
        risk_factors: Key risk factors identified.
        article_10_mapping: Art. 10(2) factor mapping.
        confidence: Confidence in this score (0-1).
        provenance_hash: SHA-256 hash of the score data.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    agent_id: str = Field(
        ...,
        description="Risk assessment agent identifier",
    )
    agent_name: str = Field(
        "",
        description="Human-readable agent name",
    )
    raw_score: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Raw risk score from the agent (0-100)",
    )
    weight: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Weight of this risk dimension",
    )
    weighted_score: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="raw_score * weight",
    )
    risk_factors: List[str] = Field(
        default_factory=list,
        description="Key risk factors identified",
    )
    article_10_mapping: List[str] = Field(
        default_factory=list,
        description="Art. 10(2) factor mapping",
    )
    confidence: Decimal = Field(
        Decimal("1.0"),
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Confidence in this score (0-1)",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash of the score data",
    )

class CompositeRiskProfile(GreenLangBase):
    """Aggregated risk profile from all 10 risk assessment agents.

    Combines individual risk scores into a weighted composite using
    the deterministic formula from the PRD.

    Attributes:
        profile_id: Unique profile identifier.
        workflow_id: Parent workflow identifier.
        contributions: Individual risk score contributions.
        composite_score: Weighted composite risk score (0-100).
        risk_level: Textual risk level classification.
        highest_risk_dimensions: Top risk dimensions.
        all_dimensions_scored: Whether all 10 dimensions are scored.
        coverage_pct: Percentage of risk dimensions scored.
        assessed_at: Assessment timestamp.
        provenance_hash: SHA-256 hash of the profile.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
    )

    profile_id: str = Field(
        default_factory=_new_uuid,
        description="Unique profile identifier",
    )
    workflow_id: str = Field(
        ...,
        description="Parent workflow identifier",
    )
    contributions: List[RiskScoreContribution] = Field(
        default_factory=list,
        description="Individual risk score contributions",
    )
    composite_score: Decimal = Field(
        Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Weighted composite risk score (0-100)",
    )
    risk_level: str = Field(
        "unknown",
        description="Textual risk level (negligible/low/medium/high/critical)",
    )
    highest_risk_dimensions: List[str] = Field(
        default_factory=list,
        description="Top risk dimensions by score",
    )
    all_dimensions_scored: bool = Field(
        False,
        description="Whether all 10 dimensions are scored",
    )
    coverage_pct: Decimal = Field(
        Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Percentage of risk dimensions scored",
    )
    assessed_at: datetime = Field(
        default_factory=utcnow,
        description="Assessment timestamp",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash of the profile",
    )

class MitigationDecision(GreenLangBase):
    """Risk mitigation decision and outcome.

    Records whether mitigation was required, what measures were
    applied, and whether residual risk was reduced to acceptable levels.

    Attributes:
        decision_id: Unique decision identifier.
        workflow_id: Parent workflow identifier.
        mitigation_required: Whether mitigation was required.
        mitigation_level: Level of mitigation (none/standard/enhanced).
        pre_mitigation_score: Composite risk score before mitigation.
        post_mitigation_score: Residual risk score after mitigation.
        mitigation_strategies: List of mitigation strategies applied.
        adequacy_verified: Whether mitigation adequacy was verified.
        proportionality_verified: Whether proportionality was verified.
        evidence: Mitigation evidence documentation.
        bypass_justification: Justification if mitigation was bypassed.
        decided_at: Decision timestamp.
        provenance_hash: SHA-256 hash of the decision.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
    )

    decision_id: str = Field(
        default_factory=_new_uuid,
        description="Unique decision identifier",
    )
    workflow_id: str = Field(
        ...,
        description="Parent workflow identifier",
    )
    mitigation_required: bool = Field(
        ...,
        description="Whether mitigation was required",
    )
    mitigation_level: str = Field(
        "none",
        description="Level: none/standard/enhanced",
    )
    pre_mitigation_score: Decimal = Field(
        Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Composite risk score before mitigation",
    )
    post_mitigation_score: Optional[Decimal] = Field(
        None,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Residual risk score after mitigation",
    )
    mitigation_strategies: List[str] = Field(
        default_factory=list,
        description="List of mitigation strategies applied",
    )
    adequacy_verified: bool = Field(
        False,
        description="Whether mitigation adequacy was verified",
    )
    proportionality_verified: bool = Field(
        False,
        description="Whether proportionality was verified",
    )
    evidence: Dict[str, Any] = Field(
        default_factory=dict,
        description="Mitigation evidence documentation",
    )
    bypass_justification: Optional[str] = Field(
        None,
        max_length=5000,
        description="Justification if mitigation was bypassed",
    )
    decided_at: datetime = Field(
        default_factory=utcnow,
        description="Decision timestamp",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash of the decision",
    )

class CircuitBreakerRecord(GreenLangBase):
    """Circuit breaker state record for an agent.

    Tracks the circuit breaker state machine for a specific agent
    including failure counts, state transitions, and probe results.

    Attributes:
        agent_id: EUDR agent identifier.
        state: Current circuit breaker state.
        failure_count: Consecutive failure count.
        success_count: Consecutive success count (half-open).
        last_failure_at: Timestamp of last failure.
        last_success_at: Timestamp of last success.
        opened_at: Timestamp when circuit was opened.
        half_open_at: Timestamp when circuit entered half-open.
        failure_threshold: Configured failure threshold.
        reset_timeout_s: Configured reset timeout.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    agent_id: str = Field(
        ...,
        description="EUDR agent identifier",
    )
    state: CircuitBreakerState = Field(
        CircuitBreakerState.CLOSED,
        description="Current circuit breaker state",
    )
    failure_count: int = Field(
        0,
        ge=0,
        description="Consecutive failure count",
    )
    success_count: int = Field(
        0,
        ge=0,
        description="Consecutive success count (half-open)",
    )
    last_failure_at: Optional[datetime] = Field(
        None,
        description="Timestamp of last failure",
    )
    last_success_at: Optional[datetime] = Field(
        None,
        description="Timestamp of last success",
    )
    opened_at: Optional[datetime] = Field(
        None,
        description="Timestamp when circuit was opened",
    )
    half_open_at: Optional[datetime] = Field(
        None,
        description="Timestamp when circuit entered half-open",
    )
    failure_threshold: int = Field(
        5,
        ge=1,
        description="Configured failure threshold",
    )
    reset_timeout_s: int = Field(
        60,
        ge=1,
        description="Configured reset timeout",
    )

class RetryRecord(GreenLangBase):
    """Record of a retry attempt for an agent invocation.

    Captures the details of each retry including delay, attempt number,
    error that triggered the retry, and outcome.

    Attributes:
        retry_id: Unique retry record identifier.
        workflow_id: Parent workflow identifier.
        agent_id: Agent being retried.
        attempt_number: Current attempt number (1-based).
        delay_s: Computed backoff delay in seconds.
        error_message: Error that triggered the retry.
        error_classification: Classification of the error.
        outcome: Result of the retry (success/failure).
        attempted_at: Retry attempt timestamp.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    retry_id: str = Field(
        default_factory=_new_uuid,
        description="Unique retry record identifier",
    )
    workflow_id: str = Field(
        ...,
        description="Parent workflow identifier",
    )
    agent_id: str = Field(
        ...,
        description="Agent being retried",
    )
    attempt_number: int = Field(
        ...,
        ge=1,
        description="Current attempt number (1-based)",
    )
    delay_s: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Computed backoff delay in seconds",
    )
    error_message: str = Field(
        "",
        description="Error that triggered the retry",
    )
    error_classification: ErrorClassification = Field(
        ErrorClassification.TRANSIENT,
        description="Classification of the error",
    )
    outcome: str = Field(
        "pending",
        description="Result: success/failure/pending",
    )
    attempted_at: datetime = Field(
        default_factory=utcnow,
        description="Retry attempt timestamp",
    )

class DeadLetterEntry(GreenLangBase):
    """Dead letter entry for a permanently failed agent invocation.

    Captures the complete context of a failed agent call that has
    exhausted all retry attempts for investigation and manual resolution.

    Attributes:
        entry_id: Unique dead letter entry identifier.
        workflow_id: Parent workflow identifier.
        agent_id: Failed agent identifier.
        error_message: Final error message.
        error_classification: Error classification.
        retry_history: Complete retry attempt history.
        input_data: Input data that was sent to the agent.
        circuit_breaker_state: Circuit breaker state at time of failure.
        created_at: Dead letter entry creation timestamp.
        resolved: Whether this entry has been manually resolved.
        resolved_by: User who resolved the entry.
        resolved_at: Resolution timestamp.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
    )

    entry_id: str = Field(
        default_factory=_new_uuid,
        description="Unique dead letter entry identifier",
    )
    workflow_id: str = Field(
        ...,
        description="Parent workflow identifier",
    )
    agent_id: str = Field(
        ...,
        description="Failed agent identifier",
    )
    error_message: str = Field(
        "",
        description="Final error message",
    )
    error_classification: ErrorClassification = Field(
        ErrorClassification.PERMANENT,
        description="Error classification",
    )
    retry_history: List[RetryRecord] = Field(
        default_factory=list,
        description="Complete retry attempt history",
    )
    input_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Input data that was sent to the agent",
    )
    circuit_breaker_state: Optional[CircuitBreakerState] = Field(
        None,
        description="Circuit breaker state at time of failure",
    )
    created_at: datetime = Field(
        default_factory=utcnow,
        description="Dead letter creation timestamp",
    )
    resolved: bool = Field(
        False,
        description="Whether entry has been manually resolved",
    )
    resolved_by: Optional[str] = Field(
        None,
        description="User who resolved the entry",
    )
    resolved_at: Optional[datetime] = Field(
        None,
        description="Resolution timestamp",
    )

class DDSField(GreenLangBase):
    """A single DDS (Due Diligence Statement) field per Article 12(2).

    Maps a required DDS content field to its source agents, value,
    and validation status.

    Attributes:
        field_id: Unique field identifier.
        article_ref: EUDR article reference (e.g. "12(2)(a)").
        field_name: Field name per DDS specification.
        description: Field description.
        value: Field value (serialized).
        source_agents: Agents that contributed data for this field.
        validated: Whether the field value has been validated.
        validation_notes: Validation notes or errors.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    field_id: str = Field(
        default_factory=_new_uuid,
        description="Unique field identifier",
    )
    article_ref: str = Field(
        ...,
        description="EUDR article reference",
    )
    field_name: str = Field(
        ...,
        description="Field name per DDS specification",
    )
    description: Optional[str] = Field(
        None,
        description="Field description",
    )
    value: Optional[Any] = Field(
        None,
        description="Field value (serialized)",
    )
    source_agents: List[str] = Field(
        default_factory=list,
        description="Agents that contributed data",
    )
    validated: bool = Field(
        False,
        description="Whether the field value has been validated",
    )
    validation_notes: Optional[str] = Field(
        None,
        description="Validation notes or errors",
    )

class DDSSection(GreenLangBase):
    """A section in the due diligence package report.

    Organizes DDS fields and evidence into thematic sections
    for the human-readable report and structured JSON output.

    Attributes:
        section_id: Unique section identifier.
        section_number: Section number in the report (1-9).
        title: Section title.
        description: Section description.
        fields: DDS fields in this section.
        evidence_refs: References to evidence artifacts.
        agent_outputs: Summarized outputs from contributing agents.
        completeness_pct: Section completeness percentage.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    section_id: str = Field(
        default_factory=_new_uuid,
        description="Unique section identifier",
    )
    section_number: int = Field(
        ...,
        ge=1,
        le=20,
        description="Section number in the report",
    )
    title: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Section title",
    )
    description: Optional[str] = Field(
        None,
        max_length=2000,
        description="Section description",
    )
    fields: List[DDSField] = Field(
        default_factory=list,
        description="DDS fields in this section",
    )
    evidence_refs: List[str] = Field(
        default_factory=list,
        description="References to evidence artifacts",
    )
    agent_outputs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Summarized outputs from contributing agents",
    )
    completeness_pct: Decimal = Field(
        Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Section completeness percentage",
    )

class DueDiligencePackage(GreenLangBase):
    """Complete audit-ready due diligence evidence package.

    Compiles all 25 agent outputs into a structured, DDS-compatible
    evidence bundle with provenance chain, ready for EU Information
    System submission per EUDR Article 12.

    Attributes:
        package_id: Unique package identifier.
        workflow_id: Source workflow identifier.
        dds_schema_version: DDS schema version.
        operator_id: Operator identifier.
        operator_name: Operator name and contact.
        commodity: EUDR commodity assessed.
        workflow_type: Type of due diligence performed.
        sections: Report sections with DDS fields and evidence.
        executive_summary: Executive summary text.
        risk_profile: Composite risk assessment profile.
        mitigation_summary: Mitigation decision summary.
        quality_gate_results: All quality gate evaluations.
        workflow_metadata: Workflow execution metadata.
        total_agents_executed: Number of agents that executed.
        total_duration_ms: Total workflow duration.
        language: Report language.
        integrity_hash: Package-level SHA-256 integrity hash.
        artifact_hashes: Per-artifact SHA-256 hashes.
        generated_at: Package generation timestamp.
        generated_by: User or system that generated the package.
        download_urls: URLs for downloading package formats.
        provenance_hash: SHA-256 provenance chain hash.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
    )

    package_id: str = Field(
        default_factory=_new_uuid,
        description="Unique package identifier",
    )
    workflow_id: str = Field(
        ...,
        description="Source workflow identifier",
    )
    dds_schema_version: str = Field(
        "1.0.0",
        description="DDS schema version",
    )
    operator_id: Optional[str] = Field(
        None,
        description="Operator identifier",
    )
    operator_name: Optional[str] = Field(
        None,
        max_length=500,
        description="Operator name and contact",
    )
    commodity: Optional[EUDRCommodity] = Field(
        None,
        description="EUDR commodity assessed",
    )
    workflow_type: WorkflowType = Field(
        WorkflowType.STANDARD,
        description="Type of due diligence performed",
    )
    sections: List[DDSSection] = Field(
        default_factory=list,
        description="Report sections with DDS fields and evidence",
    )
    executive_summary: Optional[str] = Field(
        None,
        max_length=10000,
        description="Executive summary text",
    )
    risk_profile: Optional[CompositeRiskProfile] = Field(
        None,
        description="Composite risk assessment profile",
    )
    mitigation_summary: Optional[MitigationDecision] = Field(
        None,
        description="Mitigation decision summary",
    )
    quality_gate_results: Dict[str, QualityGateEvaluation] = Field(
        default_factory=dict,
        description="All quality gate evaluations",
    )
    workflow_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Workflow execution metadata",
    )
    total_agents_executed: int = Field(
        0,
        ge=0,
        description="Number of agents that executed",
    )
    total_duration_ms: Optional[Decimal] = Field(
        None,
        ge=Decimal("0"),
        description="Total workflow duration in milliseconds",
    )
    language: str = Field(
        "en",
        description="Report language",
    )
    integrity_hash: Optional[str] = Field(
        None,
        description="Package-level SHA-256 integrity hash",
    )
    artifact_hashes: Dict[str, str] = Field(
        default_factory=dict,
        description="Per-artifact SHA-256 hashes",
    )
    generated_at: datetime = Field(
        default_factory=utcnow,
        description="Package generation timestamp",
    )
    generated_by: str = Field(
        "system",
        description="User or system that generated the package",
    )
    download_urls: Dict[str, str] = Field(
        default_factory=dict,
        description="URLs for downloading package formats",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 provenance chain hash",
    )

# ---------------------------------------------------------------------------
# Request Models (6)
# ---------------------------------------------------------------------------

class CreateWorkflowRequest(GreenLangBase):
    """Request to create a new due diligence workflow.

    Attributes:
        workflow_type: Standard, simplified, or custom.
        commodity: EUDR commodity to assess.
        operator_id: Operator identifier.
        operator_name: Operator name.
        product_ids: Product identifiers.
        shipment_ids: Shipment identifiers.
        country_codes: Countries of production.
        custom_definition: Custom workflow definition (for CUSTOM type).
        language: Preferred report language.
        request_id: Client-provided request identifier.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    workflow_type: WorkflowType = Field(
        WorkflowType.STANDARD,
        description="Standard, simplified, or custom",
    )
    commodity: Optional[EUDRCommodity] = Field(
        None,
        description="EUDR commodity to assess",
    )
    operator_id: Optional[str] = Field(
        None,
        description="Operator identifier",
    )
    operator_name: Optional[str] = Field(
        None,
        max_length=500,
        description="Operator name",
    )
    product_ids: List[str] = Field(
        default_factory=list,
        description="Product identifiers",
    )
    shipment_ids: List[str] = Field(
        default_factory=list,
        description="Shipment identifiers",
    )
    country_codes: List[str] = Field(
        default_factory=list,
        description="Countries of production (ISO 3166-1 alpha-2)",
    )
    custom_definition: Optional[WorkflowDefinition] = Field(
        None,
        description="Custom workflow definition (for CUSTOM type)",
    )
    language: str = Field(
        "en",
        description="Preferred report language",
    )
    request_id: Optional[str] = Field(
        None,
        description="Client-provided request identifier",
    )

class StartWorkflowRequest(GreenLangBase):
    """Request to start executing a created workflow.

    Attributes:
        workflow_id: Workflow identifier to start.
        input_data: Initial input data for Phase 1 agents.
        priority: Execution priority (1=highest, 5=lowest).
        request_id: Client-provided request identifier.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    workflow_id: str = Field(
        ...,
        description="Workflow identifier to start",
    )
    input_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Initial input data for Phase 1 agents",
    )
    priority: int = Field(
        3,
        ge=1,
        le=5,
        description="Execution priority (1=highest, 5=lowest)",
    )
    request_id: Optional[str] = Field(
        None,
        description="Client-provided request identifier",
    )

class ResumeWorkflowRequest(GreenLangBase):
    """Request to resume a paused or failed workflow from checkpoint.

    Attributes:
        workflow_id: Workflow identifier to resume.
        checkpoint_id: Specific checkpoint to resume from (None = latest).
        retry_failed: Whether to retry previously failed agents.
        request_id: Client-provided request identifier.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    workflow_id: str = Field(
        ...,
        description="Workflow identifier to resume",
    )
    checkpoint_id: Optional[str] = Field(
        None,
        description="Specific checkpoint to resume from",
    )
    retry_failed: bool = Field(
        True,
        description="Whether to retry previously failed agents",
    )
    request_id: Optional[str] = Field(
        None,
        description="Client-provided request identifier",
    )

class EvaluateQualityGateRequest(GreenLangBase):
    """Request to evaluate a quality gate for a workflow.

    Attributes:
        workflow_id: Workflow identifier.
        gate_id: Quality gate to evaluate (QG-1, QG-2, QG-3).
        override: Whether to override a failed gate.
        override_justification: Justification for override.
        override_by: User requesting override.
        request_id: Client-provided request identifier.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    workflow_id: str = Field(
        ...,
        description="Workflow identifier",
    )
    gate_id: QualityGateId = Field(
        ...,
        description="Quality gate to evaluate",
    )
    override: bool = Field(
        False,
        description="Whether to override a failed gate",
    )
    override_justification: Optional[str] = Field(
        None,
        max_length=5000,
        description="Justification for override",
    )
    override_by: Optional[str] = Field(
        None,
        description="User requesting override",
    )
    request_id: Optional[str] = Field(
        None,
        description="Client-provided request identifier",
    )

class GeneratePackageRequest(GreenLangBase):
    """Request to generate a due diligence package.

    Attributes:
        workflow_id: Workflow identifier.
        formats: Output formats to generate.
        language: Report language.
        include_executive_summary: Whether to include executive summary.
        include_evidence_annexes: Whether to include evidence annexes.
        request_id: Client-provided request identifier.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    workflow_id: str = Field(
        ...,
        description="Workflow identifier",
    )
    formats: List[str] = Field(
        default_factory=lambda: ["json", "pdf"],
        description="Output formats to generate",
    )
    language: str = Field(
        "en",
        description="Report language",
    )
    include_executive_summary: bool = Field(
        True,
        description="Whether to include executive summary",
    )
    include_evidence_annexes: bool = Field(
        True,
        description="Whether to include evidence annexes",
    )
    request_id: Optional[str] = Field(
        None,
        description="Client-provided request identifier",
    )

class BatchWorkflowRequest(GreenLangBase):
    """Request to launch multiple workflows in a batch.

    Attributes:
        workflows: List of individual workflow creation requests.
        batch_priority: Priority for the entire batch.
        sequential: Whether to run workflows sequentially (vs parallel).
        request_id: Client-provided request identifier.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    workflows: List[CreateWorkflowRequest] = Field(
        ...,
        min_length=1,
        description="Individual workflow creation requests",
    )
    batch_priority: int = Field(
        3,
        ge=1,
        le=5,
        description="Priority for the entire batch",
    )
    sequential: bool = Field(
        False,
        description="Whether to run workflows sequentially",
    )
    request_id: Optional[str] = Field(
        None,
        description="Client-provided request identifier",
    )

# ---------------------------------------------------------------------------
# Response Models (6)
# ---------------------------------------------------------------------------

class WorkflowStatusResponse(GreenLangBase):
    """Response containing current workflow status.

    Attributes:
        workflow_id: Workflow identifier.
        status: Current workflow status.
        current_phase: Current due diligence phase.
        progress_pct: Overall progress percentage.
        agents_completed: Number of agents completed.
        agents_total: Total number of agents in workflow.
        agents_running: Number of agents currently running.
        agents_failed: Number of agents that failed.
        eta_seconds: Estimated time to completion.
        processing_time_ms: Response generation time.
        provenance_hash: SHA-256 hash.
        request_id: Echoed request identifier.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    workflow_id: str = Field(
        ...,
        description="Workflow identifier",
    )
    status: WorkflowStatus = Field(
        ...,
        description="Current workflow status",
    )
    current_phase: DueDiligencePhase = Field(
        ...,
        description="Current due diligence phase",
    )
    progress_pct: Decimal = Field(
        Decimal("0"),
        description="Overall progress percentage",
    )
    agents_completed: int = Field(0, description="Agents completed")
    agents_total: int = Field(0, description="Total agents")
    agents_running: int = Field(0, description="Agents running")
    agents_failed: int = Field(0, description="Agents failed")
    eta_seconds: Optional[int] = Field(None, description="ETA seconds")
    processing_time_ms: Decimal = Field(Decimal("0"))
    provenance_hash: Optional[str] = Field(None)
    request_id: Optional[str] = Field(None)

class WorkflowProgressResponse(GreenLangBase):
    """Detailed workflow progress response with per-agent status.

    Attributes:
        workflow_id: Workflow identifier.
        status: Current workflow status.
        current_phase: Current phase.
        progress_pct: Overall progress.
        agent_statuses: Per-agent execution status.
        quality_gate_results: Quality gate results.
        composite_risk_score: Current composite risk score.
        timeline: Execution timeline data for visualization.
        processing_time_ms: Response generation time.
        provenance_hash: SHA-256 hash.
        request_id: Echoed request identifier.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    workflow_id: str = Field(..., description="Workflow identifier")
    status: WorkflowStatus = Field(...)
    current_phase: DueDiligencePhase = Field(...)
    progress_pct: Decimal = Field(Decimal("0"))
    agent_statuses: Dict[str, str] = Field(default_factory=dict)
    quality_gate_results: Dict[str, str] = Field(default_factory=dict)
    composite_risk_score: Optional[Decimal] = Field(None)
    timeline: List[Dict[str, Any]] = Field(default_factory=list)
    processing_time_ms: Decimal = Field(Decimal("0"))
    provenance_hash: Optional[str] = Field(None)
    request_id: Optional[str] = Field(None)

class QualityGateResponse(GreenLangBase):
    """Response from quality gate evaluation.

    Attributes:
        evaluation: Quality gate evaluation result.
        processing_time_ms: Evaluation duration.
        provenance_hash: SHA-256 hash.
        request_id: Echoed request identifier.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    evaluation: QualityGateEvaluation = Field(...)
    processing_time_ms: Decimal = Field(Decimal("0"))
    provenance_hash: Optional[str] = Field(None)
    request_id: Optional[str] = Field(None)

class PackageGenerationResponse(GreenLangBase):
    """Response from due diligence package generation.

    Attributes:
        package: Generated due diligence package.
        processing_time_ms: Generation duration.
        provenance_hash: SHA-256 hash.
        request_id: Echoed request identifier.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    package: DueDiligencePackage = Field(...)
    processing_time_ms: Decimal = Field(Decimal("0"))
    provenance_hash: Optional[str] = Field(None)
    request_id: Optional[str] = Field(None)

class WorkflowAuditTrailResponse(GreenLangBase):
    """Response containing complete workflow audit trail.

    Attributes:
        workflow_id: Workflow identifier.
        transitions: State transition history.
        checkpoints: Checkpoint history.
        quality_gates: Quality gate evaluations.
        retry_history: Retry attempt records.
        dead_letters: Dead letter entries.
        provenance_chain: Provenance hash chain.
        processing_time_ms: Response generation time.
        request_id: Echoed request identifier.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    workflow_id: str = Field(..., description="Workflow identifier")
    transitions: List[WorkflowStateTransition] = Field(default_factory=list)
    checkpoints: List[WorkflowCheckpoint] = Field(default_factory=list)
    quality_gates: Dict[str, QualityGateEvaluation] = Field(default_factory=dict)
    retry_history: List[RetryRecord] = Field(default_factory=list)
    dead_letters: List[DeadLetterEntry] = Field(default_factory=list)
    provenance_chain: List[Dict[str, Any]] = Field(default_factory=list)
    processing_time_ms: Decimal = Field(Decimal("0"))
    request_id: Optional[str] = Field(None)

class BatchWorkflowResponse(GreenLangBase):
    """Response from batch workflow creation.

    Attributes:
        batch_id: Unique batch identifier.
        workflow_ids: Created workflow identifiers.
        total_workflows: Total workflows in batch.
        status: Overall batch status.
        processing_time_ms: Batch creation duration.
        request_id: Echoed request identifier.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    batch_id: str = Field(default_factory=_new_uuid)
    workflow_ids: List[str] = Field(default_factory=list)
    total_workflows: int = Field(0, ge=0)
    status: str = Field("created")
    processing_time_ms: Decimal = Field(Decimal("0"))
    request_id: Optional[str] = Field(None)
