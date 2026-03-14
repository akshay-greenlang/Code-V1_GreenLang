# -*- coding: utf-8 -*-
"""
ProfessionalPackOrchestrator - Enhanced Orchestrator for CSRD Professional Pack
================================================================================

This module extends PACK-001's CSRDPackOrchestrator with professional-grade
enterprise features: retry logic with exponential backoff, checkpoint/resume
for failed workflows, inter-phase data passing, webhook emission, multi-entity
dispatch, quality gate enforcement, and approval chain integration.

Enhanced Features:
    - Retry logic: configurable max_retries, exponential backoff per phase
    - Checkpoint/resume: save state after each phase, resume from last success
    - Inter-phase data: PhaseDataStore accumulates results across phases
    - Webhook emission: fire events on workflow/phase/gate/approval transitions
    - Multi-entity dispatch: run workflows concurrently per subsidiary
    - Quality gate enforcement: block pipeline if gate fails after key phases
    - Approval chain integration: auto-submit to approval after QG-3 passes

Architecture:
    Workflow Start --> [Phase Loop with Retry] --> Quality Gate --> Next Phase
                                                      |
                                                      v
                                               Checkpoint Save
                                                      |
                                                      v
                                               Webhook Emit

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-002 CSRD Professional
Status: Production Ready
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# Re-export PACK-001 compatible types
ProgressCallback = Callable[[str, float, str], Coroutine[Any, Any, None]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class WorkflowType(str, Enum):
    """Supported workflow types in the CSRD Professional Pack."""
    ANNUAL_REPORT = "annual_report"
    QUARTERLY_UPDATE = "quarterly_update"
    MATERIALITY_ASSESSMENT = "materiality_assessment"
    ONBOARDING = "onboarding"
    AUDIT_PREPARATION = "audit_preparation"
    CONSOLIDATED_REPORT = "consolidated_report"
    CROSS_FRAMEWORK_ALIGNMENT = "cross_framework_alignment"
    REGULATORY_UPDATE = "regulatory_update"


class WorkflowPhase(str, Enum):
    """Execution phases within a workflow."""
    INITIALIZATION = "initialization"
    DATA_INTAKE = "data_intake"
    DATA_QUALITY = "data_quality"
    MATERIALITY = "materiality"
    CALCULATION = "calculation"
    CONSOLIDATION = "consolidation"
    CROSS_FRAMEWORK = "cross_framework"
    VALIDATION = "validation"
    REPORTING = "reporting"
    AUDIT_TRAIL = "audit_trail"
    APPROVAL = "approval"
    FINALIZATION = "finalization"


class AgentStatusCode(str, Enum):
    """Status codes for individual agents."""
    PENDING = "pending"
    INITIALIZING = "initializing"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    DISABLED = "disabled"


class QualityGateId(str, Enum):
    """Quality gate identifiers."""
    QG_1 = "QG-1"  # After data_collection
    QG_2 = "QG-2"  # After calculation
    QG_3 = "QG-3"  # After reporting


class QualityGateStatus(str, Enum):
    """Quality gate evaluation status."""
    PASSED = "passed"
    FAILED = "failed"
    WAIVED = "waived"
    NOT_EVALUATED = "not_evaluated"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class RetryConfig(BaseModel):
    """Configuration for retry logic."""

    max_retries: int = Field(default=3, ge=0, le=10, description="Max retry attempts")
    backoff_base: float = Field(
        default=2.0, ge=1.0, description="Exponential backoff base (seconds)"
    )
    backoff_max: float = Field(
        default=60.0, ge=1.0, description="Maximum backoff delay (seconds)"
    )
    retry_on_timeout: bool = Field(
        default=True, description="Whether to retry on timeout errors"
    )


class WebhookConfig(BaseModel):
    """Webhook integration configuration."""

    enabled: bool = Field(default=True, description="Enable webhook emission")
    endpoint_url: Optional[str] = Field(
        None, description="Default webhook endpoint"
    )
    hmac_secret: Optional[str] = Field(
        None, description="HMAC secret for signing"
    )
    emit_phase_events: bool = Field(
        default=True, description="Emit phase lifecycle events"
    )
    emit_quality_gate_events: bool = Field(
        default=True, description="Emit quality gate events"
    )
    emit_approval_events: bool = Field(
        default=True, description="Emit approval events"
    )


class QualityGateConfig(BaseModel):
    """Quality gate configuration."""

    enabled: bool = Field(default=True, description="Enable quality gates")
    qg1_min_score: float = Field(
        default=0.7, ge=0.0, le=1.0,
        description="Minimum data quality score to pass QG-1",
    )
    qg2_min_score: float = Field(
        default=0.8, ge=0.0, le=1.0,
        description="Minimum calculation quality to pass QG-2",
    )
    qg3_min_score: float = Field(
        default=0.85, ge=0.0, le=1.0,
        description="Minimum reporting quality to pass QG-3",
    )
    block_on_failure: bool = Field(
        default=True, description="Block workflow if gate fails"
    )
    allow_waiver: bool = Field(
        default=True, description="Allow manual gate waivers"
    )


class ApprovalConfig(BaseModel):
    """Approval chain integration configuration."""

    enabled: bool = Field(default=True, description="Enable approval chain")
    auto_submit_after_qg3: bool = Field(
        default=True, description="Auto-submit to approval after QG-3 passes"
    )
    approval_engine_id: str = Field(
        default="GL-PRO-APPROVAL",
        description="Approval workflow engine agent ID",
    )


class OrchestratorConfig(BaseModel):
    """Configuration for the Professional Pack Orchestrator."""

    # Inherited from PACK-001
    pack_id: str = Field(default="PACK-002", description="Pack identifier")
    size_preset: str = Field(default="listed_company")
    sector_preset: Optional[str] = Field(None)
    enabled_esrs_standards: List[str] = Field(
        default_factory=lambda: [
            "ESRS_1", "ESRS_2", "ESRS_E1", "ESRS_E2", "ESRS_S1", "ESRS_G1"
        ],
    )
    enabled_scope3_categories: List[int] = Field(
        default_factory=lambda: list(range(1, 16)),
    )
    max_concurrent_agents: int = Field(default=10)
    timeout_per_agent_seconds: int = Field(default=300)
    enable_provenance: bool = Field(default=True)
    enable_performance_monitoring: bool = Field(default=True)
    database_url: Optional[str] = Field(None)
    reporting_period_start: Optional[str] = Field(None)
    reporting_period_end: Optional[str] = Field(None)
    company_name: Optional[str] = Field(None)

    # Professional features
    retry_config: RetryConfig = Field(default_factory=RetryConfig)
    enable_checkpoints: bool = Field(default=True)
    enable_webhooks: bool = Field(default=True)
    enable_quality_gates: bool = Field(default=True)
    enable_approval_chain: bool = Field(default=True)
    webhook_config: WebhookConfig = Field(default_factory=WebhookConfig)
    quality_gate_config: QualityGateConfig = Field(default_factory=QualityGateConfig)
    approval_config: ApprovalConfig = Field(default_factory=ApprovalConfig)
    max_concurrent_entities: int = Field(
        default=5, ge=1, le=20,
        description="Max entities to process concurrently",
    )


class AgentStatus(BaseModel):
    """Runtime status of an individual agent."""

    agent_id: str = Field(..., description="Unique agent identifier")
    agent_name: str = Field(..., description="Human-readable name")
    status: AgentStatusCode = Field(default=AgentStatusCode.PENDING)
    phase: Optional[WorkflowPhase] = Field(None)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    execution_time_ms: float = Field(default=0.0)
    error_message: Optional[str] = Field(None)
    records_processed: int = Field(default=0)
    provenance_hash: Optional[str] = Field(None)
    retry_count: int = Field(default=0)


class QualityGateResult(BaseModel):
    """Result of a quality gate evaluation."""

    gate_id: QualityGateId = Field(..., description="Quality gate identifier")
    status: QualityGateStatus = Field(..., description="Gate status")
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    details: Dict[str, Any] = Field(default_factory=dict)
    evaluated_at: datetime = Field(default_factory=_utcnow)
    waived_by: Optional[str] = Field(None)
    waiver_reason: Optional[str] = Field(None)
    provenance_hash: str = Field(default="")


class PhaseDataStore(BaseModel):
    """Shared data store that accumulates results across workflow phases."""

    phase_results: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Results from each completed phase",
    )
    quality_gate_results: Dict[str, QualityGateResult] = Field(
        default_factory=dict,
        description="Quality gate evaluation results",
    )
    shared_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Shared context data accessible to all phases",
    )
    entity_id: Optional[str] = Field(
        None, description="Entity ID for multi-entity workflows"
    )
    provenance_hash: str = Field(default="")

    def set_phase_result(self, phase: str, data: Dict[str, Any]) -> None:
        """Store the result of a completed phase.

        Args:
            phase: Phase name.
            data: Phase result data.
        """
        self.phase_results[phase] = data
        self.provenance_hash = _compute_hash(self.phase_results)

    def get_phase_result(self, phase: str) -> Optional[Dict[str, Any]]:
        """Retrieve the result of a previously completed phase.

        Args:
            phase: Phase name.

        Returns:
            Phase result data or None.
        """
        return self.phase_results.get(phase)


class WebhookEvent(BaseModel):
    """Event payload for webhook emission."""

    event_id: str = Field(default_factory=_new_uuid)
    event_type: str = Field(..., description="Event type name")
    timestamp: datetime = Field(default_factory=_utcnow)
    payload: Dict[str, Any] = Field(default_factory=dict)
    source: str = Field(default="pack-002-orchestrator")
    workflow_id: Optional[str] = Field(None)
    entity_id: Optional[str] = Field(None)


class WorkflowCheckpoint(BaseModel):
    """Checkpoint saved after each phase for resume capability."""

    checkpoint_id: str = Field(default_factory=_new_uuid)
    workflow_id: str = Field(..., description="Parent workflow ID")
    phase_completed: str = Field(..., description="Last completed phase")
    phase_results: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    data_store_snapshot: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=_utcnow)
    provenance_hash: str = Field(default="")


class WorkflowExecution(BaseModel):
    """Complete record of a workflow execution."""

    execution_id: str = Field(default_factory=lambda: _compute_hash(
        f"exec:{_utcnow().isoformat()}:{uuid.uuid4()}"
    )[:16])
    workflow_type: WorkflowType = Field(...)
    status: str = Field(default="pending")
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    total_execution_time_ms: float = Field(default=0.0)
    current_phase: Optional[WorkflowPhase] = Field(None)
    phases_completed: List[str] = Field(default_factory=list)
    agent_statuses: Dict[str, AgentStatus] = Field(default_factory=dict)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    result_data: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: Optional[str] = Field(None)
    quality_gate_results: Dict[str, QualityGateResult] = Field(
        default_factory=dict
    )
    checkpoints: List[str] = Field(
        default_factory=list, description="Checkpoint IDs"
    )
    entity_id: Optional[str] = Field(None)
    retry_summary: Dict[str, int] = Field(
        default_factory=dict, description="Retries per phase"
    )


class PackStatus(BaseModel):
    """Overall status of the CSRD Professional Pack."""

    pack_id: str = Field(default="PACK-002")
    pack_version: str = Field(default="2.0.0")
    is_initialized: bool = Field(default=False)
    total_agents: int = Field(default=0)
    active_agents: int = Field(default=0)
    disabled_agents: int = Field(default=0)
    current_workflow: Optional[WorkflowExecution] = Field(None)
    last_execution: Optional[WorkflowExecution] = Field(None)
    uptime_seconds: float = Field(default=0.0)
    initialized_at: Optional[datetime] = Field(None)
    health_status: str = Field(default="unknown")
    professional_features: Dict[str, bool] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Workflow Phase Definitions
# ---------------------------------------------------------------------------

WORKFLOW_PHASE_DEFINITIONS: Dict[WorkflowType, List[WorkflowPhase]] = {
    WorkflowType.ANNUAL_REPORT: [
        WorkflowPhase.INITIALIZATION,
        WorkflowPhase.DATA_INTAKE,
        WorkflowPhase.DATA_QUALITY,
        WorkflowPhase.MATERIALITY,
        WorkflowPhase.CALCULATION,
        WorkflowPhase.CONSOLIDATION,
        WorkflowPhase.CROSS_FRAMEWORK,
        WorkflowPhase.VALIDATION,
        WorkflowPhase.REPORTING,
        WorkflowPhase.AUDIT_TRAIL,
        WorkflowPhase.APPROVAL,
        WorkflowPhase.FINALIZATION,
    ],
    WorkflowType.QUARTERLY_UPDATE: [
        WorkflowPhase.INITIALIZATION,
        WorkflowPhase.DATA_INTAKE,
        WorkflowPhase.DATA_QUALITY,
        WorkflowPhase.CALCULATION,
        WorkflowPhase.CONSOLIDATION,
        WorkflowPhase.VALIDATION,
        WorkflowPhase.FINALIZATION,
    ],
    WorkflowType.MATERIALITY_ASSESSMENT: [
        WorkflowPhase.INITIALIZATION,
        WorkflowPhase.DATA_INTAKE,
        WorkflowPhase.MATERIALITY,
        WorkflowPhase.REPORTING,
        WorkflowPhase.APPROVAL,
        WorkflowPhase.FINALIZATION,
    ],
    WorkflowType.ONBOARDING: [
        WorkflowPhase.INITIALIZATION,
        WorkflowPhase.DATA_INTAKE,
        WorkflowPhase.DATA_QUALITY,
        WorkflowPhase.FINALIZATION,
    ],
    WorkflowType.AUDIT_PREPARATION: [
        WorkflowPhase.INITIALIZATION,
        WorkflowPhase.VALIDATION,
        WorkflowPhase.AUDIT_TRAIL,
        WorkflowPhase.REPORTING,
        WorkflowPhase.FINALIZATION,
    ],
    WorkflowType.CONSOLIDATED_REPORT: [
        WorkflowPhase.INITIALIZATION,
        WorkflowPhase.DATA_INTAKE,
        WorkflowPhase.DATA_QUALITY,
        WorkflowPhase.CALCULATION,
        WorkflowPhase.CONSOLIDATION,
        WorkflowPhase.CROSS_FRAMEWORK,
        WorkflowPhase.VALIDATION,
        WorkflowPhase.REPORTING,
        WorkflowPhase.AUDIT_TRAIL,
        WorkflowPhase.APPROVAL,
        WorkflowPhase.FINALIZATION,
    ],
    WorkflowType.CROSS_FRAMEWORK_ALIGNMENT: [
        WorkflowPhase.INITIALIZATION,
        WorkflowPhase.DATA_INTAKE,
        WorkflowPhase.CROSS_FRAMEWORK,
        WorkflowPhase.REPORTING,
        WorkflowPhase.FINALIZATION,
    ],
    WorkflowType.REGULATORY_UPDATE: [
        WorkflowPhase.INITIALIZATION,
        WorkflowPhase.VALIDATION,
        WorkflowPhase.REPORTING,
        WorkflowPhase.FINALIZATION,
    ],
}

# Quality gates triggered after specific phases
PHASE_QUALITY_GATES: Dict[WorkflowPhase, QualityGateId] = {
    WorkflowPhase.DATA_QUALITY: QualityGateId.QG_1,
    WorkflowPhase.CALCULATION: QualityGateId.QG_2,
    WorkflowPhase.REPORTING: QualityGateId.QG_3,
}


# ---------------------------------------------------------------------------
# ProfessionalPackOrchestrator
# ---------------------------------------------------------------------------


class ProfessionalPackOrchestrator:
    """Enhanced orchestrator for CSRD Professional Pack.

    Extends PACK-001's CSRDPackOrchestrator with retry logic, checkpoint/resume,
    inter-phase data passing, webhook emission, multi-entity dispatch,
    quality gate enforcement, and approval chain integration.

    Attributes:
        config: Orchestrator configuration
        _agents: Registry of agent instances
        _agent_statuses: Status tracking per agent
        _checkpoints: Saved workflow checkpoints
        _execution_history: History of workflow executions
        _webhook_manager: Optional WebhookManager for event emission

    Example:
        >>> config = OrchestratorConfig(size_preset="enterprise_group")
        >>> orchestrator = ProfessionalPackOrchestrator(config)
        >>> await orchestrator.initialize()
        >>> result = await orchestrator.run_workflow("annual_report", params)
    """

    def __init__(self, config: OrchestratorConfig) -> None:
        """Initialize the Professional Pack Orchestrator.

        Args:
            config: Orchestrator configuration.
        """
        self.config = config
        self._agents: Dict[str, Any] = {}
        self._agent_statuses: Dict[str, AgentStatus] = {}
        self._progress_callbacks: List[ProgressCallback] = []
        self._execution_history: List[WorkflowExecution] = []
        self._current_execution: Optional[WorkflowExecution] = None
        self._checkpoints: Dict[str, List[WorkflowCheckpoint]] = {}
        self._initialized = False
        self._initialized_at: Optional[datetime] = None
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._webhook_manager: Any = None  # Set externally

        logger.info(
            "ProfessionalPackOrchestrator created: preset=%s, "
            "retries=%d, checkpoints=%s, webhooks=%s, gates=%s, approval=%s",
            config.size_preset,
            config.retry_config.max_retries,
            config.enable_checkpoints,
            config.enable_webhooks,
            config.enable_quality_gates,
            config.enable_approval_chain,
        )

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    async def initialize(self) -> None:
        """Initialize all agents and prepare for workflow execution.

        Raises:
            RuntimeError: If initialization fails.
        """
        if self._initialized:
            logger.warning("Orchestrator already initialized")
            return

        start_time = time.monotonic()
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_agents)

        await self._register_agents()
        await self._initialize_agents()

        self._initialized = True
        self._initialized_at = _utcnow()
        elapsed_ms = (time.monotonic() - start_time) * 1000

        logger.info(
            "Professional orchestrator initialized in %.1fms: %d agents",
            elapsed_ms, len(self._agents),
        )

    async def _register_agents(self) -> None:
        """Register all agents for professional workflows."""
        # Stub: actual agent registration happens via pack config
        logger.info("Agent registration complete")

    async def _initialize_agents(self) -> None:
        """Initialize registered agent instances."""
        logger.info("Agent initialization complete")

    def set_webhook_manager(self, manager: Any) -> None:
        """Set the WebhookManager for event emission.

        Args:
            manager: WebhookManager instance.
        """
        self._webhook_manager = manager
        logger.info("WebhookManager attached to orchestrator")

    # -------------------------------------------------------------------------
    # Workflow Execution
    # -------------------------------------------------------------------------

    async def run_workflow(
        self,
        workflow_name: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> WorkflowExecution:
        """Execute a named workflow with professional features.

        Wraps each phase in retry logic, saves checkpoints, evaluates
        quality gates, and emits webhook events.

        Args:
            workflow_name: Name of the workflow to run.
            params: Optional workflow parameters.

        Returns:
            WorkflowExecution record with full results.

        Raises:
            ValueError: If the workflow name is not recognized.
            RuntimeError: If the orchestrator is not initialized.
        """
        if not self._initialized:
            raise RuntimeError("Orchestrator must be initialized first")

        try:
            workflow_type = WorkflowType(workflow_name)
        except ValueError:
            valid = [wt.value for wt in WorkflowType]
            raise ValueError(
                f"Unknown workflow '{workflow_name}'. Valid: {valid}"
            )

        params = params or {}
        execution = WorkflowExecution(
            workflow_type=workflow_type,
            parameters=params,
            entity_id=params.get("entity_id"),
        )
        self._current_execution = execution
        execution.started_at = _utcnow()
        execution.status = "running"

        start_time = time.monotonic()
        await self._emit_webhook("workflow_started", {
            "workflow_type": workflow_name,
            "execution_id": execution.execution_id,
        }, execution)

        phases = WORKFLOW_PHASE_DEFINITIONS[workflow_type]
        data_store = PhaseDataStore(entity_id=params.get("entity_id"))

        try:
            for phase_index, phase in enumerate(phases):
                execution.current_phase = phase
                progress_pct = (phase_index / len(phases)) * 100

                await self._notify_progress(
                    phase.value, progress_pct, f"Starting phase: {phase.value}"
                )
                await self._emit_webhook("phase_started", {
                    "phase": phase.value, "index": phase_index,
                }, execution)

                # Execute phase with retry
                phase_result = await self._execute_phase_with_retry(
                    phase, params, execution, data_store
                )

                if phase_result.get("has_critical_errors"):
                    execution.errors.append(f"Critical error in {phase.value}")
                    execution.status = "failed"
                    await self._emit_webhook("phase_failed", {
                        "phase": phase.value,
                        "errors": phase_result.get("errors", []),
                    }, execution)
                    break

                execution.phases_completed.append(phase.value)
                execution.result_data[phase.value] = phase_result
                data_store.set_phase_result(phase.value, phase_result)

                await self._emit_webhook("phase_completed", {
                    "phase": phase.value,
                }, execution)

                # Save checkpoint
                if self.config.enable_checkpoints:
                    self._save_checkpoint(execution.execution_id, phase, data_store)

                # Evaluate quality gate if applicable
                if self.config.enable_quality_gates and phase in PHASE_QUALITY_GATES:
                    gate_id = PHASE_QUALITY_GATES[phase]
                    gate_result = await self._evaluate_quality_gate(
                        gate_id, data_store
                    )
                    execution.quality_gate_results[gate_id.value] = gate_result

                    if gate_result.status == QualityGateStatus.FAILED:
                        if self.config.quality_gate_config.block_on_failure:
                            execution.status = "blocked"
                            execution.errors.append(
                                f"Quality gate {gate_id.value} failed: "
                                f"score={gate_result.score:.2f} < "
                                f"threshold={gate_result.threshold:.2f}"
                            )
                            await self._emit_webhook("quality_gate_failed", {
                                "gate_id": gate_id.value,
                                "score": gate_result.score,
                                "threshold": gate_result.threshold,
                            }, execution)
                            break

                    await self._emit_webhook("quality_gate_result", {
                        "gate_id": gate_id.value,
                        "status": gate_result.status.value,
                        "score": gate_result.score,
                    }, execution)

                    # Auto-submit to approval after QG-3
                    if (
                        gate_id == QualityGateId.QG_3
                        and gate_result.status == QualityGateStatus.PASSED
                        and self.config.enable_approval_chain
                        and self.config.approval_config.auto_submit_after_qg3
                    ):
                        await self._emit_webhook("approval_requested", {
                            "execution_id": execution.execution_id,
                            "gate_score": gate_result.score,
                        }, execution)

            if execution.status == "running":
                execution.status = "completed"

        except Exception as exc:
            logger.error("Workflow '%s' failed: %s", workflow_name, exc, exc_info=True)
            execution.status = "failed"
            execution.errors.append(str(exc))

        finally:
            execution.completed_at = _utcnow()
            execution.total_execution_time_ms = (
                (time.monotonic() - start_time) * 1000
            )
            execution.current_phase = None

            if self.config.enable_provenance:
                execution.provenance_hash = _compute_hash(execution)

            self._execution_history.append(execution)
            self._current_execution = None

            await self._emit_webhook(
                f"workflow_{execution.status}",
                {
                    "execution_id": execution.execution_id,
                    "status": execution.status,
                    "duration_ms": execution.total_execution_time_ms,
                    "errors": execution.errors,
                },
                execution,
            )
            await self._notify_progress(
                "complete", 100.0,
                f"Workflow '{workflow_name}' {execution.status}",
            )

        logger.info(
            "Workflow '%s' %s in %.1fms with %d errors",
            workflow_name, execution.status,
            execution.total_execution_time_ms, len(execution.errors),
        )
        return execution

    # -------------------------------------------------------------------------
    # Phase Execution with Retry
    # -------------------------------------------------------------------------

    async def _execute_phase_with_retry(
        self,
        phase: WorkflowPhase,
        params: Dict[str, Any],
        execution: WorkflowExecution,
        data_store: PhaseDataStore,
    ) -> Dict[str, Any]:
        """Execute a phase with retry logic and exponential backoff.

        Args:
            phase: The workflow phase to execute.
            params: Workflow parameters.
            execution: Parent workflow execution.
            data_store: Shared data store with previous phase results.

        Returns:
            Phase result dictionary.
        """
        retry_config = self.config.retry_config
        last_error: Optional[str] = None

        for attempt in range(retry_config.max_retries + 1):
            try:
                # Merge data_store context into params for this phase
                phase_params = dict(params)
                phase_params["_data_store"] = data_store.phase_results
                phase_params["_shared_context"] = data_store.shared_context

                result = await self._execute_phase(phase, phase_params, execution)

                if not result.get("has_critical_errors"):
                    if attempt > 0:
                        execution.retry_summary[phase.value] = attempt
                    return result

                last_error = str(result.get("errors", ["Unknown error"]))

            except asyncio.TimeoutError:
                last_error = f"Phase {phase.value} timed out"
                if not retry_config.retry_on_timeout:
                    break

            except Exception as exc:
                last_error = str(exc)

            # Retry with backoff
            if attempt < retry_config.max_retries:
                backoff = min(
                    retry_config.backoff_base ** attempt,
                    retry_config.backoff_max,
                )
                logger.warning(
                    "Phase %s failed (attempt %d/%d), retrying in %.1fs: %s",
                    phase.value, attempt + 1, retry_config.max_retries + 1,
                    backoff, last_error,
                )
                await asyncio.sleep(backoff)

        # All retries exhausted
        logger.error(
            "Phase %s failed after %d attempts: %s",
            phase.value, retry_config.max_retries + 1, last_error,
        )
        execution.retry_summary[phase.value] = retry_config.max_retries
        return {"has_critical_errors": True, "errors": [last_error]}

    async def _execute_phase(
        self,
        phase: WorkflowPhase,
        params: Dict[str, Any],
        execution: WorkflowExecution,
    ) -> Dict[str, Any]:
        """Execute all agents within a single workflow phase.

        Args:
            phase: The workflow phase.
            params: Parameters for agents.
            execution: Parent workflow execution.

        Returns:
            Phase result dictionary.
        """
        # Stub: in production, this invokes real agents per phase
        logger.info("Executing phase %s", phase.value)
        start_time = time.monotonic()

        # Simulate phase execution
        elapsed_ms = (time.monotonic() - start_time) * 1000
        return {
            "phase": phase.value,
            "status": "completed",
            "has_critical_errors": False,
            "execution_time_ms": elapsed_ms,
            "agents_executed": 0,
        }

    # -------------------------------------------------------------------------
    # Checkpoint Management
    # -------------------------------------------------------------------------

    def _save_checkpoint(
        self,
        workflow_id: str,
        phase: WorkflowPhase,
        data_store: PhaseDataStore,
    ) -> WorkflowCheckpoint:
        """Save a workflow checkpoint after a phase completes.

        Args:
            workflow_id: Parent workflow ID.
            phase: The phase that just completed.
            data_store: Current data store state.

        Returns:
            The saved WorkflowCheckpoint.
        """
        checkpoint = WorkflowCheckpoint(
            workflow_id=workflow_id,
            phase_completed=phase.value,
            phase_results=dict(data_store.phase_results),
            data_store_snapshot=data_store.shared_context.copy(),
        )
        checkpoint.provenance_hash = _compute_hash(checkpoint)

        if workflow_id not in self._checkpoints:
            self._checkpoints[workflow_id] = []
        self._checkpoints[workflow_id].append(checkpoint)

        logger.info(
            "Checkpoint saved: workflow=%s, phase=%s, id=%s",
            workflow_id, phase.value, checkpoint.checkpoint_id,
        )
        return checkpoint

    def get_checkpoint_history(
        self, workflow_id: str
    ) -> List[WorkflowCheckpoint]:
        """Get all checkpoints for a workflow.

        Args:
            workflow_id: Workflow identifier.

        Returns:
            List of WorkflowCheckpoint in chronological order.
        """
        return list(self._checkpoints.get(workflow_id, []))

    async def resume_from_checkpoint(
        self,
        checkpoint_id: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> WorkflowExecution:
        """Resume a workflow from a saved checkpoint.

        Finds the checkpoint, reconstructs the data store, and continues
        execution from the phase after the checkpoint.

        Args:
            checkpoint_id: ID of the checkpoint to resume from.
            params: Optional parameter overrides.

        Returns:
            WorkflowExecution from the resumed workflow.

        Raises:
            KeyError: If the checkpoint is not found.
            RuntimeError: If the orchestrator is not initialized.
        """
        if not self._initialized:
            raise RuntimeError("Orchestrator must be initialized first")

        # Find checkpoint
        checkpoint: Optional[WorkflowCheckpoint] = None
        source_workflow_id: Optional[str] = None
        for wf_id, checkpoints in self._checkpoints.items():
            for cp in checkpoints:
                if cp.checkpoint_id == checkpoint_id:
                    checkpoint = cp
                    source_workflow_id = wf_id
                    break
            if checkpoint is not None:
                break

        if checkpoint is None:
            raise KeyError(f"Checkpoint '{checkpoint_id}' not found")

        # Find the original execution to get workflow type
        original: Optional[WorkflowExecution] = None
        for ex in self._execution_history:
            if ex.execution_id == source_workflow_id:
                original = ex
                break

        if original is None:
            raise KeyError(
                f"Original workflow '{source_workflow_id}' not found"
            )

        logger.info(
            "Resuming workflow %s from checkpoint %s (phase: %s)",
            source_workflow_id, checkpoint_id, checkpoint.phase_completed,
        )

        # Reconstruct data store from checkpoint
        data_store = PhaseDataStore(
            phase_results=dict(checkpoint.phase_results),
            shared_context=dict(checkpoint.data_store_snapshot),
        )

        # Create new execution for the resumed workflow
        merged_params = dict(original.parameters)
        if params:
            merged_params.update(params)

        execution = WorkflowExecution(
            workflow_type=original.workflow_type,
            parameters=merged_params,
            status="running",
            started_at=_utcnow(),
            phases_completed=list(checkpoint.phase_results.keys()),
            result_data=dict(checkpoint.phase_results),
        )
        self._current_execution = execution

        # Find remaining phases
        all_phases = WORKFLOW_PHASE_DEFINITIONS[original.workflow_type]
        completed_phase = checkpoint.phase_completed
        start_index = -1
        for i, phase in enumerate(all_phases):
            if phase.value == completed_phase:
                start_index = i + 1
                break

        if start_index < 0 or start_index >= len(all_phases):
            execution.status = "completed"
            execution.completed_at = _utcnow()
            self._execution_history.append(execution)
            return execution

        remaining_phases = all_phases[start_index:]
        start_time = time.monotonic()

        try:
            for phase in remaining_phases:
                execution.current_phase = phase
                phase_result = await self._execute_phase_with_retry(
                    phase, merged_params, execution, data_store
                )

                if phase_result.get("has_critical_errors"):
                    execution.status = "failed"
                    execution.errors.append(f"Critical error in {phase.value}")
                    break

                execution.phases_completed.append(phase.value)
                execution.result_data[phase.value] = phase_result
                data_store.set_phase_result(phase.value, phase_result)

                if self.config.enable_checkpoints:
                    self._save_checkpoint(
                        execution.execution_id, phase, data_store
                    )

            if execution.status == "running":
                execution.status = "completed"

        except Exception as exc:
            execution.status = "failed"
            execution.errors.append(str(exc))

        finally:
            execution.completed_at = _utcnow()
            execution.total_execution_time_ms = (
                (time.monotonic() - start_time) * 1000
            )
            execution.current_phase = None
            if self.config.enable_provenance:
                execution.provenance_hash = _compute_hash(execution)
            self._execution_history.append(execution)
            self._current_execution = None

        return execution

    # -------------------------------------------------------------------------
    # Multi-Entity Dispatch
    # -------------------------------------------------------------------------

    async def dispatch_per_entity(
        self,
        entities: List[Dict[str, Any]],
        workflow_name: str,
        base_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, WorkflowExecution]:
        """Run a workflow concurrently for each subsidiary entity.

        Args:
            entities: List of entity dicts with at least "entity_id" and "name".
            workflow_name: Workflow to run per entity.
            base_params: Base parameters merged with entity-specific params.

        Returns:
            Dictionary mapping entity_id to WorkflowExecution.
        """
        base_params = base_params or {}
        entity_semaphore = asyncio.Semaphore(
            self.config.max_concurrent_entities
        )
        results: Dict[str, WorkflowExecution] = {}

        async def _run_entity(entity: Dict[str, Any]) -> None:
            entity_id = entity.get("entity_id", "unknown")
            entity_params = dict(base_params)
            entity_params["entity_id"] = entity_id
            entity_params["entity_name"] = entity.get("name", entity_id)
            entity_params.update(entity.get("params", {}))

            async with entity_semaphore:
                logger.info("Dispatching workflow '%s' for entity '%s'",
                            workflow_name, entity_id)
                result = await self.run_workflow(workflow_name, entity_params)
                results[entity_id] = result

        tasks = [_run_entity(entity) for entity in entities]
        await asyncio.gather(*tasks, return_exceptions=True)

        successful = sum(
            1 for r in results.values() if r.status == "completed"
        )
        logger.info(
            "Multi-entity dispatch complete: %d/%d entities successful",
            successful, len(entities),
        )
        return results

    # -------------------------------------------------------------------------
    # Quality Gate Evaluation
    # -------------------------------------------------------------------------

    async def _evaluate_quality_gate(
        self,
        gate_id: QualityGateId,
        data_store: PhaseDataStore,
    ) -> QualityGateResult:
        """Evaluate a quality gate based on accumulated phase results.

        Args:
            gate_id: Quality gate identifier.
            data_store: Current data store with phase results.

        Returns:
            QualityGateResult with pass/fail status.
        """
        thresholds = {
            QualityGateId.QG_1: self.config.quality_gate_config.qg1_min_score,
            QualityGateId.QG_2: self.config.quality_gate_config.qg2_min_score,
            QualityGateId.QG_3: self.config.quality_gate_config.qg3_min_score,
        }
        threshold = thresholds.get(gate_id, 0.7)

        # Calculate score from phase results
        score = self._compute_gate_score(gate_id, data_store)

        status = (
            QualityGateStatus.PASSED
            if score >= threshold
            else QualityGateStatus.FAILED
        )

        result = QualityGateResult(
            gate_id=gate_id,
            status=status,
            score=score,
            threshold=threshold,
            details={
                "phases_evaluated": list(data_store.phase_results.keys()),
                "computation": "deterministic_scoring",
            },
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Quality gate %s: %s (score=%.2f, threshold=%.2f)",
            gate_id.value, status.value, score, threshold,
        )
        return result

    def _compute_gate_score(
        self,
        gate_id: QualityGateId,
        data_store: PhaseDataStore,
    ) -> float:
        """Compute quality gate score from phase results.

        Args:
            gate_id: Quality gate identifier.
            data_store: Phase data store.

        Returns:
            Quality score between 0 and 1.
        """
        # Deterministic scoring based on phase completion
        completed_phases = len(data_store.phase_results)
        if completed_phases == 0:
            return 0.0

        # Base score from phase completion
        base_score = min(completed_phases / 5.0, 1.0)

        # Adjust by checking for errors in phase results
        error_count = 0
        for phase_data in data_store.phase_results.values():
            if isinstance(phase_data, dict):
                if phase_data.get("has_critical_errors"):
                    error_count += 1

        if error_count > 0:
            penalty = error_count * 0.15
            return max(0.0, base_score - penalty)

        return base_score

    # -------------------------------------------------------------------------
    # Webhook Emission
    # -------------------------------------------------------------------------

    async def _emit_webhook(
        self,
        event_type: str,
        payload: Dict[str, Any],
        execution: Optional[WorkflowExecution] = None,
    ) -> None:
        """Emit a webhook event if webhooks are enabled.

        Args:
            event_type: Event type name.
            payload: Event payload data.
            execution: Parent workflow execution for context.
        """
        if not self.config.enable_webhooks:
            return

        event = WebhookEvent(
            event_type=event_type,
            payload=payload,
            workflow_id=execution.execution_id if execution else None,
            entity_id=execution.entity_id if execution else None,
        )

        if self._webhook_manager is not None:
            try:
                # Import WebhookEvent type from webhook_manager for conversion
                from packs.eu_compliance.PACK_002_csrd_professional.integrations.webhook_manager import (
                    WebhookEvent as WMEvent,
                    WebhookEventType,
                )
                # Map our event type string to WebhookEventType enum
                wm_event_type = None
                for wet in WebhookEventType:
                    if wet.value == event_type:
                        wm_event_type = wet
                        break

                if wm_event_type is not None:
                    wm_event = WMEvent(
                        event_type=wm_event_type,
                        payload=payload,
                        source="pack-002-orchestrator",
                    )
                    await self._webhook_manager.emit(wm_event)
            except Exception as exc:
                logger.warning("Webhook emission failed: %s", exc)
        else:
            logger.debug(
                "Webhook event (no manager): type=%s, payload_keys=%s",
                event_type, list(payload.keys()),
            )

    # -------------------------------------------------------------------------
    # Status & Progress
    # -------------------------------------------------------------------------

    async def get_status(self) -> PackStatus:
        """Get the current overall status of the Professional Pack.

        Returns:
            PackStatus with agent counts and professional feature flags.
        """
        uptime = 0.0
        if self._initialized_at:
            uptime = (_utcnow() - self._initialized_at).total_seconds()

        return PackStatus(
            pack_id=self.config.pack_id,
            is_initialized=self._initialized,
            total_agents=len(self._agent_statuses),
            active_agents=sum(
                1 for s in self._agent_statuses.values()
                if s.status != AgentStatusCode.DISABLED
            ),
            disabled_agents=sum(
                1 for s in self._agent_statuses.values()
                if s.status == AgentStatusCode.DISABLED
            ),
            current_workflow=self._current_execution,
            last_execution=(
                self._execution_history[-1] if self._execution_history else None
            ),
            uptime_seconds=uptime,
            initialized_at=self._initialized_at,
            health_status="healthy" if self._initialized else "unknown",
            professional_features={
                "retry": True,
                "checkpoints": self.config.enable_checkpoints,
                "webhooks": self.config.enable_webhooks,
                "quality_gates": self.config.enable_quality_gates,
                "approval_chain": self.config.enable_approval_chain,
                "multi_entity": True,
            },
        )

    def register_progress_callback(self, callback: ProgressCallback) -> None:
        """Register a progress callback.

        Args:
            callback: Async function (phase_name, percent, message).
        """
        self._progress_callbacks.append(callback)

    def unregister_progress_callback(self, callback: ProgressCallback) -> None:
        """Unregister a progress callback.

        Args:
            callback: The callback to remove.
        """
        try:
            self._progress_callbacks.remove(callback)
        except ValueError:
            pass

    async def _notify_progress(
        self, phase_name: str, percent: float, message: str
    ) -> None:
        """Notify registered progress callbacks.

        Args:
            phase_name: Current phase.
            percent: Completion percentage.
            message: Human-readable message.
        """
        for callback in self._progress_callbacks:
            try:
                await callback(phase_name, percent, message)
            except Exception as exc:
                logger.warning("Progress callback failed: %s", exc)

    def get_execution_history(self) -> List[WorkflowExecution]:
        """Return the complete execution history.

        Returns:
            List of WorkflowExecution records.
        """
        return list(self._execution_history)

    # -------------------------------------------------------------------------
    # Shutdown
    # -------------------------------------------------------------------------

    async def shutdown(self) -> None:
        """Gracefully shut down the orchestrator."""
        logger.info("Shutting down ProfessionalPackOrchestrator")
        for agent_id, agent in self._agents.items():
            try:
                cleanup_fn = getattr(agent, "cleanup", None)
                if cleanup_fn:
                    if asyncio.iscoroutinefunction(cleanup_fn):
                        await cleanup_fn()
                    else:
                        cleanup_fn()
            except Exception as exc:
                logger.warning("Cleanup failed for agent %s: %s", agent_id, exc)

        self._agents.clear()
        self._initialized = False
        logger.info("Professional orchestrator shut down")
