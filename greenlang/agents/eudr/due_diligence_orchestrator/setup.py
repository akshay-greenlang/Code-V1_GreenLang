# -*- coding: utf-8 -*-
"""
Due Diligence Orchestrator Service Facade - AGENT-EUDR-026

High-level service facade that wires together all orchestrator
components (engines, coordinators, clients, event bus) into a
single cohesive entry point. Provides the primary API used by
the FastAPI router layer to create, start, evaluate, and manage
due diligence workflows.

This facade implements the Facade Pattern to hide the complexity
of the 8 internal engines, 4 reference data modules, and 4
integration client modules behind a clean, use-case-oriented
interface.

Service Methods:
    Workflow Lifecycle:
        - create_workflow()  -> Create a new due diligence workflow
        - start_workflow()   -> Begin executing a workflow
        - pause_workflow()   -> Pause a running workflow
        - resume_workflow()  -> Resume a paused or gate-failed workflow
        - cancel_workflow()  -> Cancel a workflow
        - get_workflow_status() -> Get current workflow status

    Quality Gates:
        - evaluate_quality_gate() -> Evaluate a quality gate
        - override_quality_gate() -> Override a failed gate

    Package Generation:
        - generate_package() -> Generate due diligence package

    Health & Monitoring:
        - health_check()     -> Check all agent endpoints
        - get_metrics()      -> Get orchestrator metrics

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-026 Due Diligence Orchestrator (GL-EUDR-DDO-026)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional
from greenlang.schemas import utcnow

from greenlang.agents.eudr.due_diligence_orchestrator.config import (
    DueDiligenceOrchestratorConfig,
    get_config,
)
from greenlang.agents.eudr.due_diligence_orchestrator.due_diligence_package_generator import (
    DueDiligencePackageGenerator,
)
from greenlang.agents.eudr.due_diligence_orchestrator.error_recovery_manager import (
    ErrorRecoveryManager,
)
from greenlang.agents.eudr.due_diligence_orchestrator.information_gathering_coordinator import (
    InformationGatheringCoordinator,
)
from greenlang.agents.eudr.due_diligence_orchestrator.integration.agent_client import (
    AgentCallResult,
    AgentClient,
)
from greenlang.agents.eudr.due_diligence_orchestrator.integration.event_bus import (
    AGENT_COMPLETED,
    AGENT_FAILED,
    AGENT_STARTED,
    EventBus,
    PACKAGE_COMPLETED,
    PACKAGE_FAILED,
    PACKAGE_GENERATING,
    PHASE_COMPLETED,
    PHASE_FAILED,
    PHASE_STARTED,
    QUALITY_GATE_EVALUATING,
    QUALITY_GATE_FAILED,
    QUALITY_GATE_OVERRIDDEN,
    QUALITY_GATE_PASSED,
    WORKFLOW_CANCELLED,
    WORKFLOW_COMPLETED,
    WORKFLOW_CREATED,
    WORKFLOW_FAILED,
    WORKFLOW_PAUSED,
    WORKFLOW_RESUMED,
    WORKFLOW_STARTED,
    WorkflowEvent,
    get_event_bus,
)
from greenlang.agents.eudr.due_diligence_orchestrator.integration.risk_assessment_clients import (
    get_all_phase2_clients,
)
from greenlang.agents.eudr.due_diligence_orchestrator.integration.supply_chain_clients import (
    get_all_phase1_clients,
)
from greenlang.agents.eudr.due_diligence_orchestrator.models import (
    ALL_EUDR_AGENTS,
    AGENT_NAMES,
    AgentExecutionStatus,
    BatchWorkflowRequest,
    BatchWorkflowResponse,
    CompositeRiskProfile,
    CreateWorkflowRequest,
    DueDiligencePackage,
    DueDiligencePhase,
    EUDRCommodity,
    EvaluateQualityGateRequest,
    GeneratePackageRequest,
    MitigationDecision,
    PackageGenerationResponse,
    QualityGateEvaluation,
    QualityGateId,
    QualityGateResponse,
    QualityGateResultEnum,
    ResumeWorkflowRequest,
    StartWorkflowRequest,
    VERSION,
    WorkflowProgressResponse,
    WorkflowState,
    WorkflowStatus,
    WorkflowStatusResponse,
    WorkflowType,
    _new_uuid,
    _utcnow,
)
from greenlang.agents.eudr.due_diligence_orchestrator.parallel_execution_engine import (
    ParallelExecutionEngine,
)
from greenlang.agents.eudr.due_diligence_orchestrator.provenance import (
    ProvenanceTracker,
    get_tracker,
)
from greenlang.agents.eudr.due_diligence_orchestrator.quality_gate_engine import (
    QualityGateEngine,
)
from greenlang.agents.eudr.due_diligence_orchestrator.risk_assessment_coordinator import (
    RiskAssessmentCoordinator,
)
from greenlang.agents.eudr.due_diligence_orchestrator.risk_mitigation_coordinator import (
    RiskMitigationCoordinator,
)
from greenlang.agents.eudr.due_diligence_orchestrator.workflow_definition_engine import (
    WorkflowDefinitionEngine,
)
from greenlang.agents.eudr.due_diligence_orchestrator.workflow_state_manager import (
    WorkflowStateManager,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Service Facade
# ---------------------------------------------------------------------------


class DueDiligenceOrchestratorService:
    """High-level service facade for the Due Diligence Orchestrator.

    Wires together all internal components and provides a unified API
    for workflow lifecycle management, quality gate enforcement, and
    DDS package generation.

    Attributes:
        _config: Orchestrator configuration.
        _workflow_engine: DAG definition and topological sort engine.
        _state_manager: Workflow state machine manager.
        _quality_gate_engine: Quality gate evaluator.
        _info_coordinator: Phase 1 information gathering coordinator.
        _risk_coordinator: Phase 2 risk assessment coordinator.
        _mitigation_coordinator: Phase 3 risk mitigation coordinator.
        _parallel_engine: Parallel execution scheduler.
        _error_manager: Error recovery and circuit breaker manager.
        _package_generator: DDS package compiler.
        _agent_client: Shared HTTP client for agent calls.
        _event_bus: In-process event bus.
        _provenance: Provenance tracker.

    Example:
        >>> service = DueDiligenceOrchestratorService()
        >>> workflow = service.create_workflow(
        ...     CreateWorkflowRequest(
        ...         operator_id="OP-001",
        ...         commodity="cocoa",
        ...         countries=["GH", "CI"],
        ...         workflow_type="standard"
        ...     )
        ... )
        >>> service.start_workflow(
        ...     StartWorkflowRequest(workflow_id=workflow.workflow_id)
        ... )
    """

    def __init__(
        self,
        config: Optional[DueDiligenceOrchestratorConfig] = None,
    ) -> None:
        """Initialize the orchestrator service with all components.

        Args:
            config: Optional configuration override.
        """
        self._config = config or get_config()

        # Core engines
        self._workflow_engine = WorkflowDefinitionEngine(self._config)
        self._state_manager = WorkflowStateManager(self._config)
        self._quality_gate_engine = QualityGateEngine(self._config)
        self._info_coordinator = InformationGatheringCoordinator(self._config)
        self._risk_coordinator = RiskAssessmentCoordinator(self._config)
        self._mitigation_coordinator = RiskMitigationCoordinator(self._config)
        self._parallel_engine = ParallelExecutionEngine(self._config)
        self._error_manager = ErrorRecoveryManager(self._config)
        self._package_generator = DueDiligencePackageGenerator(self._config)

        # Integration components
        self._agent_client = AgentClient(self._config)
        self._event_bus = get_event_bus(self._config)
        self._provenance = get_tracker()

        logger.info(
            f"DueDiligenceOrchestratorService v{VERSION} initialized "
            f"(log_level={self._config.log_level})"
        )

    # ------------------------------------------------------------------
    # Workflow lifecycle
    # ------------------------------------------------------------------

    def create_workflow(
        self, request: CreateWorkflowRequest
    ) -> WorkflowStatusResponse:
        """Create a new due diligence workflow.

        Creates the workflow definition (DAG), initializes the state
        machine, and publishes a workflow.created event.

        Args:
            request: Workflow creation request with operator, commodity,
                     and country parameters.

        Returns:
            WorkflowStatusResponse with the created workflow details.

        Raises:
            ValueError: If request parameters are invalid.
        """
        start_time = utcnow()
        workflow_id = _new_uuid()

        logger.info(
            f"Creating workflow {workflow_id} for operator "
            f"{request.operator_id}, commodity={request.commodity}"
        )

        # Resolve workflow type and commodity from request (already enums)
        workflow_type = request.workflow_type
        commodity = request.commodity

        # Build workflow definition (DAG)
        if workflow_type == WorkflowType.SIMPLIFIED:
            definition = self._workflow_engine.create_simplified_workflow(
                commodity=commodity,
            )
        else:
            definition = self._workflow_engine.create_standard_workflow(
                commodity=commodity,
            )

        # Validate the definition
        is_valid, validation_errors = self._workflow_engine.validate_definition(
            definition
        )
        if not is_valid:
            raise ValueError(
                f"Workflow definition validation failed: "
                f"{'; '.join(validation_errors)}"
            )

        # Initialize state machine
        commodity_str = commodity.value if commodity else None
        state = self._state_manager.create_workflow(
            definition_id=definition.definition_id,
            workflow_type=workflow_type,
            commodity=commodity_str,
            operator_id=request.operator_id,
            operator_name=request.operator_name,
            country_codes=request.country_codes,
        )
        workflow_id = state.workflow_id

        # Track provenance
        self._provenance.record(
            entity_type="workflow",
            action="create",
            entity_id=workflow_id,
            metadata={
                "operator_id": request.operator_id,
                "commodity": commodity_str,
                "workflow_type": workflow_type.value,
                "agent_count": len(definition.nodes),
            },
        )

        # Publish event
        self._event_bus.publish_event(
            WORKFLOW_CREATED,
            workflow_id,
            {
                "operator_id": request.operator_id or "",
                "commodity": commodity_str or "",
                "workflow_type": workflow_type.value,
            },
        )

        elapsed_ms = Decimal(str(
            (utcnow() - start_time).total_seconds() * 1000
        )).quantize(Decimal("0.01"))

        agent_count = len(definition.nodes)
        logger.info(
            f"Workflow {workflow_id} created in {elapsed_ms}ms "
            f"with {agent_count} agents"
        )

        return WorkflowStatusResponse(
            workflow_id=workflow_id,
            status=WorkflowStatus.CREATED,
            current_phase=DueDiligencePhase.INFORMATION_GATHERING,
            agents_total=agent_count,
            agents_completed=0,
            progress_pct=Decimal("0.00"),
            processing_time_ms=elapsed_ms,
        )

    def start_workflow(
        self, request: StartWorkflowRequest
    ) -> WorkflowStatusResponse:
        """Start executing a created workflow.

        Transitions the workflow from CREATED to VALIDATING to RUNNING
        and begins executing the first layer of agents.

        Args:
            request: Workflow start request with workflow ID.

        Returns:
            WorkflowStatusResponse with running status.

        Raises:
            ValueError: If workflow cannot be started.
        """
        workflow_id = request.workflow_id
        start_time = utcnow()

        logger.info(f"Starting workflow {workflow_id}")

        # Transition: CREATED -> VALIDATING
        self._state_manager.transition(
            workflow_id,
            WorkflowStatus.VALIDATING,
            reason="Workflow start requested",
        )

        # Transition: VALIDATING -> RUNNING
        self._state_manager.transition(
            workflow_id,
            WorkflowStatus.RUNNING,
            reason="Validation passed, starting execution",
        )

        # Publish event
        self._event_bus.publish_event(
            WORKFLOW_STARTED,
            workflow_id,
            {"started_at": start_time.isoformat()},
        )

        # Track provenance
        self._provenance.record(
            entity_type="workflow",
            action="start",
            entity_id=workflow_id,
            metadata={"started_at": start_time.isoformat()},
        )

        elapsed_ms = Decimal(str(
            (utcnow() - start_time).total_seconds() * 1000
        )).quantize(Decimal("0.01"))

        logger.info(
            f"Workflow {workflow_id} started in {elapsed_ms}ms"
        )

        return WorkflowStatusResponse(
            workflow_id=workflow_id,
            status=WorkflowStatus.RUNNING,
            current_phase=DueDiligencePhase.INFORMATION_GATHERING,
            agents_total=len(ALL_EUDR_AGENTS),
            agents_completed=0,
            progress_pct=Decimal("0.00"),
            processing_time_ms=elapsed_ms,
        )

    def pause_workflow(self, workflow_id: str) -> WorkflowStatusResponse:
        """Pause a running workflow.

        Saves a checkpoint and pauses all agent execution.

        Args:
            workflow_id: Workflow identifier.

        Returns:
            WorkflowStatusResponse with paused status.
        """
        logger.info(f"Pausing workflow {workflow_id}")

        # Transition: RUNNING -> PAUSED
        self._state_manager.transition(
            workflow_id,
            WorkflowStatus.PAUSED,
            reason="User-initiated pause",
        )

        # Create checkpoint
        self._state_manager.create_checkpoint(workflow_id)

        # Publish event
        self._event_bus.publish_event(
            WORKFLOW_PAUSED,
            workflow_id,
            {"paused_at": utcnow().isoformat()},
        )

        return WorkflowStatusResponse(
            workflow_id=workflow_id,
            status=WorkflowStatus.PAUSED,
            current_phase=DueDiligencePhase.INFORMATION_GATHERING,
            agents_total=len(ALL_EUDR_AGENTS),
        )

    def resume_workflow(
        self, request: ResumeWorkflowRequest
    ) -> WorkflowStatusResponse:
        """Resume a paused or gate-failed workflow.

        Restores from the latest checkpoint and continues execution.

        Args:
            request: Resume request with workflow ID.

        Returns:
            WorkflowStatusResponse with running status.
        """
        workflow_id = request.workflow_id

        logger.info(f"Resuming workflow {workflow_id}")

        # Transition: PAUSED/GATE_FAILED -> RESUMING -> RUNNING
        self._state_manager.transition(
            workflow_id,
            WorkflowStatus.RESUMING,
            reason="Resume requested",
        )
        self._state_manager.transition(
            workflow_id,
            WorkflowStatus.RUNNING,
            reason="Resumed from checkpoint",
        )

        # Publish event
        self._event_bus.publish_event(
            WORKFLOW_RESUMED,
            workflow_id,
            {"resumed_at": utcnow().isoformat()},
        )

        return WorkflowStatusResponse(
            workflow_id=workflow_id,
            status=WorkflowStatus.RUNNING,
            current_phase=DueDiligencePhase.INFORMATION_GATHERING,
            agents_total=len(ALL_EUDR_AGENTS),
        )

    def cancel_workflow(
        self, workflow_id: str, reason: str = ""
    ) -> WorkflowStatusResponse:
        """Cancel a workflow.

        Terminates all running agents and marks the workflow cancelled.

        Args:
            workflow_id: Workflow identifier.
            reason: Optional cancellation reason.

        Returns:
            WorkflowStatusResponse with cancelled status.
        """
        logger.info(f"Cancelling workflow {workflow_id}: {reason}")

        # Transition: any -> CANCELLED
        self._state_manager.transition(
            workflow_id,
            WorkflowStatus.CANCELLED,
            reason=reason or "User-initiated cancellation",
        )

        # Cleanup parallel execution slots
        self._parallel_engine.cleanup_workflow(workflow_id)

        # Remove workflow-scoped event subscriptions
        self._event_bus.unsubscribe_workflow(workflow_id)

        # Publish event
        self._event_bus.publish_event(
            WORKFLOW_CANCELLED,
            workflow_id,
            {
                "cancelled_at": utcnow().isoformat(),
                "reason": reason,
            },
        )

        # Track provenance
        self._provenance.record(
            entity_type="workflow",
            action="fail",
            entity_id=workflow_id,
            metadata={"reason": reason, "sub_action": "cancelled"},
        )

        return WorkflowStatusResponse(
            workflow_id=workflow_id,
            status=WorkflowStatus.CANCELLED,
            current_phase=DueDiligencePhase.INFORMATION_GATHERING,
            agents_total=len(ALL_EUDR_AGENTS),
        )

    def get_workflow_status(
        self, workflow_id: str
    ) -> WorkflowStatusResponse:
        """Get the current status of a workflow.

        Args:
            workflow_id: Workflow identifier.

        Returns:
            WorkflowStatusResponse with current status details.
        """
        state = self._state_manager.get_state(workflow_id)
        if state is None:
            raise ValueError(f"Workflow {workflow_id} not found")

        # Compute agent counts from execution records
        completed = sum(
            1 for r in state.agent_executions.values()
            if r.status == AgentExecutionStatus.COMPLETED
        )
        running = sum(
            1 for r in state.agent_executions.values()
            if r.status == AgentExecutionStatus.RUNNING
        )
        failed = sum(
            1 for r in state.agent_executions.values()
            if r.status == AgentExecutionStatus.FAILED
        )
        total = len(state.agent_executions) or len(ALL_EUDR_AGENTS)

        return WorkflowStatusResponse(
            workflow_id=workflow_id,
            status=state.status,
            current_phase=state.current_phase,
            agents_total=total,
            agents_completed=completed,
            agents_running=running,
            agents_failed=failed,
            progress_pct=state.progress_pct,
        )

    def get_workflow_progress(
        self, workflow_id: str
    ) -> WorkflowProgressResponse:
        """Get detailed progress information for a workflow.

        Args:
            workflow_id: Workflow identifier.

        Returns:
            WorkflowProgressResponse with per-agent progress.
        """
        state = self._state_manager.get_state(workflow_id)
        if state is None:
            raise ValueError(f"Workflow {workflow_id} not found")

        # Get per-agent status
        agent_statuses: Dict[str, str] = {}
        for agent_id in ALL_EUDR_AGENTS:
            record = state.agent_executions.get(agent_id)
            if record:
                agent_statuses[agent_id] = record.status.value
            else:
                agent_statuses[agent_id] = AgentExecutionStatus.PENDING.value

        return WorkflowProgressResponse(
            workflow_id=workflow_id,
            status=state.status,
            current_phase=state.current_phase,
            progress_pct=state.progress_pct,
            agent_statuses=agent_statuses,
        )

    # ------------------------------------------------------------------
    # Quality gates
    # ------------------------------------------------------------------

    def evaluate_quality_gate(
        self, request: EvaluateQualityGateRequest
    ) -> QualityGateResponse:
        """Evaluate a quality gate for a workflow.

        Evaluates the specified quality gate (QG-1, QG-2, or QG-3)
        against the current workflow state and agent outputs.

        Args:
            request: Quality gate evaluation request.

        Returns:
            QualityGateResponse with evaluation results.
        """
        workflow_id = request.workflow_id
        gate_id = request.gate_id  # Already a QualityGateId enum
        start_time = utcnow()

        logger.info(
            f"Evaluating {gate_id.value} for workflow {workflow_id}"
        )

        # Transition to QUALITY_GATE state
        self._state_manager.transition(
            workflow_id,
            WorkflowStatus.QUALITY_GATE,
            reason=f"Evaluating {gate_id.value}",
        )

        # Publish evaluating event
        self._event_bus.publish_event(
            QUALITY_GATE_EVALUATING,
            workflow_id,
            {"gate_id": gate_id.value},
        )

        # Get workflow state for agent outputs
        state = self._state_manager.get_state(workflow_id)
        if state is None:
            raise ValueError(f"Workflow {workflow_id} not found")

        # Build check scores from agent execution outputs
        check_scores: Dict[str, Decimal] = {}
        for agent_id, record in state.agent_executions.items():
            if record.status == AgentExecutionStatus.COMPLETED:
                # Use 1.0 for completed, 0.0 for others
                check_scores[agent_id] = Decimal("1.00")
            else:
                check_scores[agent_id] = Decimal("0.00")

        # Evaluate the gate
        evaluation = self._quality_gate_engine.evaluate_gate(
            workflow_id=workflow_id,
            gate_id=gate_id,
            check_scores=check_scores,
            workflow_type=state.workflow_type,
        )

        # Record provenance
        self._provenance.record(
            entity_type="quality_gate",
            action="evaluate",
            entity_id=f"{workflow_id}:{gate_id.value}",
            metadata={
                "result": evaluation.result.value,
                "score": str(evaluation.score),
                "threshold": str(evaluation.threshold),
            },
        )

        # Handle gate result
        if evaluation.result == QualityGateResultEnum.PASSED:
            # Gate passed: transition back to RUNNING
            self._state_manager.transition(
                workflow_id,
                WorkflowStatus.RUNNING,
                reason=f"{gate_id.value} passed",
            )
            self._event_bus.publish_event(
                QUALITY_GATE_PASSED,
                workflow_id,
                {
                    "gate_id": gate_id.value,
                    "score": str(evaluation.score),
                },
            )
        else:
            # Gate failed: transition to GATE_FAILED
            self._state_manager.transition(
                workflow_id,
                WorkflowStatus.GATE_FAILED,
                reason=f"{gate_id.value} failed: {evaluation.score} < {evaluation.threshold}",
            )
            self._event_bus.publish_event(
                QUALITY_GATE_FAILED,
                workflow_id,
                {
                    "gate_id": gate_id.value,
                    "score": str(evaluation.score),
                    "threshold": str(evaluation.threshold),
                },
            )

        # Create checkpoint after gate evaluation
        self._state_manager.create_checkpoint(
            workflow_id,
            gate_id=gate_id.value,
        )

        elapsed_ms = Decimal(str(
            (utcnow() - start_time).total_seconds() * 1000
        )).quantize(Decimal("0.01"))

        logger.info(
            f"Quality gate {gate_id.value} for workflow {workflow_id}: "
            f"{evaluation.result.value} (score={evaluation.score}, "
            f"threshold={evaluation.threshold}, "
            f"elapsed={elapsed_ms}ms)"
        )

        return QualityGateResponse(
            evaluation=evaluation,
        )

    def override_quality_gate(
        self,
        workflow_id: str,
        gate_id: str,
        justification: str,
        override_by: str,
    ) -> QualityGateResponse:
        """Override a failed quality gate with justification.

        Allows an authorized user to override a failed quality gate
        with a documented justification for audit purposes.

        Args:
            workflow_id: Workflow identifier.
            gate_id: Quality gate identifier.
            justification: Override justification text.
            override_by: User or role performing the override.

        Returns:
            QualityGateResponse with override status.
        """
        gate = QualityGateId(gate_id)

        logger.warning(
            f"Quality gate override: {gate.value} for workflow "
            f"{workflow_id} by {override_by}: {justification}"
        )

        # Create a placeholder failed evaluation for the override
        # In production, this would retrieve the stored failed evaluation
        state = self._state_manager.get_state(workflow_id)
        if state is None:
            raise ValueError(f"Workflow {workflow_id} not found")

        # Build check scores for re-evaluation
        check_scores: Dict[str, Decimal] = {}
        for agent_id, record in state.agent_executions.items():
            if record.status == AgentExecutionStatus.COMPLETED:
                check_scores[agent_id] = Decimal("1.00")
            else:
                check_scores[agent_id] = Decimal("0.00")

        # Evaluate with override flag
        evaluation = self._quality_gate_engine.evaluate_gate(
            workflow_id=workflow_id,
            gate_id=gate,
            check_scores=check_scores,
            workflow_type=state.workflow_type,
            override=True,
            override_justification=justification,
            override_by=override_by,
        )

        # Record provenance
        self._provenance.record(
            entity_type="quality_gate",
            action="override",
            entity_id=f"{workflow_id}:{gate.value}",
            metadata={
                "justification": justification,
                "override_by": override_by,
            },
        )

        # Transition: GATE_FAILED -> RESUMING -> RUNNING
        self._state_manager.transition(
            workflow_id,
            WorkflowStatus.RESUMING,
            reason=f"{gate.value} overridden by {override_by}",
        )
        self._state_manager.transition(
            workflow_id,
            WorkflowStatus.RUNNING,
            reason="Resumed after quality gate override",
        )

        # Publish event
        self._event_bus.publish_event(
            QUALITY_GATE_OVERRIDDEN,
            workflow_id,
            {
                "gate_id": gate.value,
                "override_by": override_by,
                "justification": justification,
            },
        )

        return QualityGateResponse(
            evaluation=evaluation,
        )

    # ------------------------------------------------------------------
    # Package generation
    # ------------------------------------------------------------------

    def generate_package(
        self, request: GeneratePackageRequest
    ) -> PackageGenerationResponse:
        """Generate a due diligence package (DDS).

        Compiles all agent outputs into a DDS-compatible evidence
        bundle per EUDR Article 12(2).

        Args:
            request: Package generation request.

        Returns:
            PackageGenerationResponse with the generated package.
        """
        workflow_id = request.workflow_id
        start_time = utcnow()

        logger.info(
            f"Generating due diligence package for workflow {workflow_id}"
        )

        # Publish generating event
        self._event_bus.publish_event(
            PACKAGE_GENERATING,
            workflow_id,
            {"started_at": start_time.isoformat()},
        )

        # Get workflow state
        state = self._state_manager.get_state(workflow_id)
        if state is None:
            raise ValueError(f"Workflow {workflow_id} not found")

        # Collect agent outputs for the package generator
        agent_outputs: Dict[str, Dict[str, Any]] = {}
        for agent_id, record in state.agent_executions.items():
            if record.status == AgentExecutionStatus.COMPLETED:
                agent_outputs[agent_id] = record.output_summary or {}

        try:
            # Generate the package
            package = self._package_generator.generate_package(
                workflow_state=state,
                agent_outputs=agent_outputs,
                language=request.language,
            )

            # Transition: RUNNING -> COMPLETING -> COMPLETED
            self._state_manager.transition(
                workflow_id,
                WorkflowStatus.COMPLETING,
                reason="Package generation started",
            )
            self._state_manager.transition(
                workflow_id,
                WorkflowStatus.COMPLETED,
                reason="Due diligence package generated successfully",
            )

            # Record provenance
            self._provenance.record(
                entity_type="package",
                action="generate",
                entity_id=f"{workflow_id}:dds",
                metadata={
                    "package_id": package.package_id,
                    "integrity_hash": package.integrity_hash,
                    "completeness_pct": str(package.completeness_pct),
                },
            )

            # Publish completion events
            self._event_bus.publish_event(
                PACKAGE_COMPLETED,
                workflow_id,
                {
                    "package_id": package.package_id,
                    "integrity_hash": package.integrity_hash,
                },
            )
            self._event_bus.publish_event(
                WORKFLOW_COMPLETED,
                workflow_id,
                {"completed_at": utcnow().isoformat()},
            )

            elapsed_ms = Decimal(str(
                (utcnow() - start_time).total_seconds() * 1000
            )).quantize(Decimal("0.01"))

            logger.info(
                f"Due diligence package generated for workflow "
                f"{workflow_id} in {elapsed_ms}ms "
                f"(completeness={package.completeness_pct}%)"
            )

            return PackageGenerationResponse(
                package=package,
                processing_time_ms=elapsed_ms,
            )

        except Exception as e:
            logger.error(
                f"Package generation failed for workflow {workflow_id}: "
                f"{type(e).__name__}: {str(e)[:500]}"
            )

            self._event_bus.publish_event(
                PACKAGE_FAILED,
                workflow_id,
                {
                    "error": str(e)[:500],
                    "error_type": type(e).__name__,
                },
            )

            raise

    # ------------------------------------------------------------------
    # Agent execution (called by parallel engine)
    # ------------------------------------------------------------------

    def execute_agent(
        self,
        workflow_id: str,
        agent_id: str,
        input_data: Dict[str, Any],
    ) -> AgentCallResult:
        """Execute a single agent within a workflow.

        Handles the full agent lifecycle: pre-checks, HTTP call,
        retry on failure, circuit breaker, and provenance tracking.

        Args:
            workflow_id: Workflow identifier.
            agent_id: EUDR agent identifier.
            input_data: Agent input payload.

        Returns:
            AgentCallResult with response data or error.
        """
        logger.info(
            f"Executing agent {agent_id} for workflow {workflow_id}"
        )

        # Publish agent started event
        self._event_bus.publish_event(
            AGENT_STARTED,
            workflow_id,
            {"agent_id": agent_id},
        )

        # Record agent start in state manager
        self._state_manager.record_agent_start(workflow_id, agent_id)

        # Check circuit breaker
        from greenlang.agents.eudr.due_diligence_orchestrator.models import (
            CircuitBreakerState,
        )
        cb_state = self._error_manager.check_circuit_breaker(agent_id)
        if cb_state == CircuitBreakerState.OPEN:
            logger.warning(
                f"Circuit breaker OPEN for {agent_id}, skipping"
            )
            result = AgentCallResult(
                agent_id=agent_id,
                success=False,
                error_message=f"Circuit breaker OPEN for {agent_id}",
            )
            self._event_bus.publish_event(
                AGENT_FAILED,
                workflow_id,
                {
                    "agent_id": agent_id,
                    "error": "Circuit breaker OPEN",
                },
            )
            return result

        # Execute the agent call
        result = self._agent_client.call_agent(agent_id, input_data)

        if result.success:
            # Record success
            self._state_manager.record_agent_completion(
                workflow_id, agent_id,
                output_summary=result.output_data,
            )
            self._error_manager.record_success(agent_id)

            self._event_bus.publish_event(
                AGENT_COMPLETED,
                workflow_id,
                {
                    "agent_id": agent_id,
                    "duration_ms": str(result.duration_ms),
                    "provenance_hash": result.provenance_hash,
                },
            )

            # Track provenance
            self._provenance.record(
                entity_type="agent_execution",
                action="complete",
                entity_id=f"{workflow_id}:{agent_id}",
                metadata={
                    "duration_ms": str(result.duration_ms),
                    "provenance_hash": result.provenance_hash,
                },
            )
        else:
            # Record failure
            self._state_manager.record_agent_failure(
                workflow_id, agent_id, result.error_message or "Unknown error"
            )
            self._error_manager.record_failure(agent_id)

            self._event_bus.publish_event(
                AGENT_FAILED,
                workflow_id,
                {
                    "agent_id": agent_id,
                    "error": result.error_message,
                    "classification": (
                        result.error_classification.value
                        if result.error_classification
                        else "unknown"
                    ),
                },
            )

        return result

    # ------------------------------------------------------------------
    # Health and monitoring
    # ------------------------------------------------------------------

    def health_check(self) -> Dict[str, Any]:
        """Check the health of all orchestrator components.

        Returns:
            Dictionary with component health statuses.
        """
        agent_health = self._agent_client.check_all_agents_health()

        healthy_count = sum(1 for v in agent_health.values() if v)
        total_count = len(agent_health)

        return {
            "service": "due_diligence_orchestrator",
            "version": VERSION,
            "status": "healthy" if healthy_count == total_count else "degraded",
            "agents": {
                "total": total_count,
                "healthy": healthy_count,
                "unhealthy": total_count - healthy_count,
                "details": agent_health,
            },
            "event_bus": self._event_bus.get_stats(),
            "timestamp": utcnow().isoformat(),
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get orchestrator operational metrics.

        Returns:
            Dictionary with current metric values.
        """
        return {
            "service": "due_diligence_orchestrator",
            "version": VERSION,
            "event_bus_stats": self._event_bus.get_stats(),
            "provenance_stats": {
                "chain_valid": self._provenance.verify_chain(),
            },
            "timestamp": utcnow().isoformat(),
        }

    def get_audit_trail(
        self, workflow_id: str
    ) -> Dict[str, Any]:
        """Get the complete audit trail for a workflow.

        Args:
            workflow_id: Workflow identifier.

        Returns:
            Dictionary with provenance records and event timeline.
        """
        # Get provenance records
        provenance_records = self._provenance.get_records_by_workflow(
            workflow_id=workflow_id
        )

        # Get event timeline
        timeline = self._event_bus.get_workflow_timeline(workflow_id)

        # Get state transitions
        state = self._state_manager.get_state(workflow_id)
        transitions = []
        if state and hasattr(state, "transitions"):
            transitions = [
                {
                    "from_status": t.from_status.value if t.from_status else "",
                    "to_status": t.to_status.value,
                    "reason": t.reason,
                    "timestamp": t.timestamp.isoformat(),
                }
                for t in state.transitions
            ]

        return {
            "workflow_id": workflow_id,
            "provenance_records": [
                r.to_dict() if hasattr(r, "to_dict") else str(r)
                for r in provenance_records
            ],
            "event_timeline": timeline,
            "state_transitions": transitions,
            "generated_at": utcnow().isoformat(),
        }

    # ------------------------------------------------------------------
    # Batch operations
    # ------------------------------------------------------------------

    def create_batch_workflows(
        self, request: BatchWorkflowRequest
    ) -> BatchWorkflowResponse:
        """Create multiple workflows in a batch.

        Args:
            request: Batch workflow creation request.

        Returns:
            BatchWorkflowResponse with results for each workflow.
        """
        workflow_ids: List[str] = []

        for item in request.workflows:
            try:
                response = self.create_workflow(item)
                workflow_ids.append(response.workflow_id)
            except Exception as e:
                logger.error(
                    f"Batch workflow creation failed: "
                    f"{type(e).__name__}: {str(e)[:200]}"
                )

        return BatchWorkflowResponse(
            workflow_ids=workflow_ids,
            total_workflows=len(request.workflows),
            status="created" if workflow_ids else "failed",
        )

    # ------------------------------------------------------------------
    # Component accessors (for advanced usage)
    # ------------------------------------------------------------------

    @property
    def workflow_engine(self) -> WorkflowDefinitionEngine:
        """Access the workflow definition engine."""
        return self._workflow_engine

    @property
    def state_manager(self) -> WorkflowStateManager:
        """Access the workflow state manager."""
        return self._state_manager

    @property
    def quality_gate_engine(self) -> QualityGateEngine:
        """Access the quality gate engine."""
        return self._quality_gate_engine

    @property
    def info_coordinator(self) -> InformationGatheringCoordinator:
        """Access the information gathering coordinator."""
        return self._info_coordinator

    @property
    def risk_coordinator(self) -> RiskAssessmentCoordinator:
        """Access the risk assessment coordinator."""
        return self._risk_coordinator

    @property
    def mitigation_coordinator(self) -> RiskMitigationCoordinator:
        """Access the risk mitigation coordinator."""
        return self._mitigation_coordinator

    @property
    def parallel_engine(self) -> ParallelExecutionEngine:
        """Access the parallel execution engine."""
        return self._parallel_engine

    @property
    def error_manager(self) -> ErrorRecoveryManager:
        """Access the error recovery manager."""
        return self._error_manager

    @property
    def package_generator(self) -> DueDiligencePackageGenerator:
        """Access the DDS package generator."""
        return self._package_generator

    @property
    def agent_client(self) -> AgentClient:
        """Access the shared agent HTTP client."""
        return self._agent_client

    @property
    def event_bus(self) -> EventBus:
        """Access the event bus."""
        return self._event_bus

    @property
    def provenance(self) -> ProvenanceTracker:
        """Access the provenance tracker."""
        return self._provenance
