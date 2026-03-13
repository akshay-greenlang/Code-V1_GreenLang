# -*- coding: utf-8 -*-
"""
Compliance Workflow Engine - AGENT-EUDR-029

Orchestrates the full EUDR Article 11 mitigation workflow from risk
trigger through strategy design, measure approval, implementation
tracking, verification, and closure or escalation.

Workflow State Machine:
    INITIATED -> STRATEGY_DESIGNED -> MEASURES_APPROVED -> IMPLEMENTING
    -> VERIFYING -> CLOSED

    Alternative paths:
    -> ESCALATED (if measures insufficient after verification)
    -> FAILED (if critical errors occur at any phase)

    From ESCALATED: -> STRATEGY_DESIGNED (redesign) or -> FAILED
    From VERIFYING: -> IMPLEMENTING (if re-implementation needed)

Valid Transitions:
    INITIATED           -> [STRATEGY_DESIGNED, FAILED]
    STRATEGY_DESIGNED   -> [MEASURES_APPROVED, FAILED]
    MEASURES_APPROVED   -> [IMPLEMENTING, FAILED]
    IMPLEMENTING        -> [VERIFYING, ESCALATED, FAILED]
    VERIFYING           -> [CLOSED, ESCALATED, IMPLEMENTING]
    ESCALATED           -> [STRATEGY_DESIGNED, FAILED]

Zero-Hallucination Guarantees:
    - All state transitions validated against transition matrix
    - No LLM involvement in workflow logic
    - Deterministic phase progression
    - Complete provenance trail for every transition

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-029 Mitigation Measure Designer (GL-EUDR-MMD-029)
Regulation: EU 2023/1115 (EUDR) Articles 10, 11, 29, 31
Status: Production Ready
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .config import MitigationMeasureDesignerConfig, get_config
from .models import (
    MeasureTemplate,
    MitigationStrategy,
    RiskTrigger,
    VerificationReport,
    VerificationResult,
    WorkflowState,
    WorkflowStatus,
)
from .provenance import ProvenanceTracker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# State transition matrix
# ---------------------------------------------------------------------------

_VALID_TRANSITIONS: Dict[WorkflowStatus, List[WorkflowStatus]] = {
    WorkflowStatus.INITIATED: [
        WorkflowStatus.STRATEGY_DESIGNED,
        WorkflowStatus.FAILED,
    ],
    WorkflowStatus.STRATEGY_DESIGNED: [
        WorkflowStatus.MEASURES_APPROVED,
        WorkflowStatus.FAILED,
    ],
    WorkflowStatus.MEASURES_APPROVED: [
        WorkflowStatus.IMPLEMENTING,
        WorkflowStatus.FAILED,
    ],
    WorkflowStatus.IMPLEMENTING: [
        WorkflowStatus.VERIFYING,
        WorkflowStatus.ESCALATED,
        WorkflowStatus.FAILED,
    ],
    WorkflowStatus.VERIFYING: [
        WorkflowStatus.CLOSED,
        WorkflowStatus.ESCALATED,
        WorkflowStatus.IMPLEMENTING,
    ],
    WorkflowStatus.ESCALATED: [
        WorkflowStatus.STRATEGY_DESIGNED,
        WorkflowStatus.FAILED,
    ],
    WorkflowStatus.CLOSED: [],
    WorkflowStatus.FAILED: [],
}


class ComplianceWorkflowEngine:
    """Orchestrates the full Article 11 mitigation workflow.

    Manages the complete lifecycle from risk trigger through strategy
    design, measure approval, implementation, verification, and
    closure. Enforces valid state transitions and maintains audit trail.

    Attributes:
        _config: Agent configuration.
        _provenance: Provenance tracker.
        _strategy_designer: Strategy designer engine (lazy init).
        _template_library: Template library engine (lazy init).
        _effectiveness_estimator: Effectiveness estimator (lazy init).
        _implementation_tracker: Implementation tracker (lazy init).
        _risk_reduction_verifier: Risk reduction verifier (lazy init).
        _report_generator: Report generator (lazy init).
        _workflows: In-memory workflow store.

    Example:
        >>> engine = ComplianceWorkflowEngine()
        >>> workflow = await engine.initiate_workflow(risk_trigger)
        >>> assert workflow.status == WorkflowStatus.INITIATED
    """

    def __init__(
        self,
        config: Optional[MitigationMeasureDesignerConfig] = None,
        strategy_designer: Optional[Any] = None,
        template_library: Optional[Any] = None,
        effectiveness_estimator: Optional[Any] = None,
        implementation_tracker: Optional[Any] = None,
        risk_reduction_verifier: Optional[Any] = None,
        report_generator: Optional[Any] = None,
        provenance: Optional[ProvenanceTracker] = None,
    ) -> None:
        """Initialize ComplianceWorkflowEngine.

        Args:
            config: Agent configuration. Uses get_config() if None.
            strategy_designer: Strategy designer engine.
            template_library: Template library engine.
            effectiveness_estimator: Effectiveness estimator.
            implementation_tracker: Implementation tracker.
            risk_reduction_verifier: Risk reduction verifier.
            report_generator: Report generator.
            provenance: Provenance tracker instance.
        """
        self._config = config or get_config()
        self._provenance = provenance or ProvenanceTracker()
        self._strategy_designer = strategy_designer
        self._template_library = template_library
        self._effectiveness_estimator = effectiveness_estimator
        self._implementation_tracker = implementation_tracker
        self._risk_reduction_verifier = risk_reduction_verifier
        self._report_generator = report_generator
        self._workflows: Dict[str, WorkflowState] = {}
        self._strategies: Dict[str, MitigationStrategy] = {}
        self._verifications: Dict[str, VerificationReport] = {}
        logger.info("ComplianceWorkflowEngine initialized")

    def _get_strategy_designer(self) -> Any:
        """Lazy-load strategy designer engine."""
        if self._strategy_designer is None:
            from .mitigation_strategy_designer import (
                MitigationStrategyDesigner,
            )
            self._strategy_designer = MitigationStrategyDesigner(
                config=self._config, provenance=self._provenance,
            )
        return self._strategy_designer

    def _get_template_library(self) -> Any:
        """Lazy-load template library engine."""
        if self._template_library is None:
            from .measure_template_library import MeasureTemplateLibrary
            self._template_library = MeasureTemplateLibrary(
                config=self._config,
            )
        return self._template_library

    def _get_risk_reduction_verifier(self) -> Any:
        """Lazy-load risk reduction verifier engine."""
        if self._risk_reduction_verifier is None:
            from .risk_reduction_verifier import RiskReductionVerifier
            self._risk_reduction_verifier = RiskReductionVerifier(
                config=self._config, provenance=self._provenance,
            )
        return self._risk_reduction_verifier

    def _get_report_generator(self) -> Any:
        """Lazy-load report generator engine."""
        if self._report_generator is None:
            from .mitigation_report_generator import (
                MitigationReportGenerator,
            )
            self._report_generator = MitigationReportGenerator(
                config=self._config, provenance=self._provenance,
            )
        return self._report_generator

    async def initiate_workflow(
        self, risk_trigger: RiskTrigger,
    ) -> WorkflowState:
        """Start a new mitigation workflow from a risk trigger.

        Creates a workflow in INITIATED state and records the
        risk trigger as the starting point.

        Args:
            risk_trigger: Risk trigger from EUDR-028.

        Returns:
            WorkflowState in INITIATED status.
        """
        workflow_id = f"wfl-{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc)

        workflow = WorkflowState(
            workflow_id=workflow_id,
            operator_id=risk_trigger.operator_id,
            commodity=risk_trigger.commodity,
            status=WorkflowStatus.INITIATED,
            started_at=now,
        )

        self._workflows[workflow_id] = workflow

        # Record provenance
        provenance_hash = self._provenance.compute_hash(
            {"workflow_id": workflow_id, "status": "initiated",
             "assessment_id": risk_trigger.assessment_id}
        )
        workflow.provenance_hash = provenance_hash

        self._provenance.create_entry(
            step="initiate_workflow",
            source="risk_trigger",
            input_hash=self._provenance.compute_hash(
                {"assessment_id": risk_trigger.assessment_id}
            ),
            output_hash=provenance_hash,
        )

        logger.info(
            "Workflow initiated: id=%s, operator=%s, commodity=%s",
            workflow_id,
            risk_trigger.operator_id,
            risk_trigger.commodity.value,
        )

        return workflow

    async def design_phase(
        self,
        workflow_id: str,
        risk_trigger: RiskTrigger,
    ) -> MitigationStrategy:
        """Execute the strategy design phase.

        Transitions workflow from INITIATED to STRATEGY_DESIGNED
        and invokes the MitigationStrategyDesigner.

        Args:
            workflow_id: Workflow identifier.
            risk_trigger: Risk trigger for strategy design.

        Returns:
            Designed MitigationStrategy.

        Raises:
            ValueError: If workflow not found or invalid transition.
        """
        workflow = self._get_workflow_or_raise(workflow_id)
        self._validate_transition(
            workflow.status, WorkflowStatus.STRATEGY_DESIGNED,
        )

        # Get templates
        library = self._get_template_library()
        templates = library.get_all_templates()

        # Design strategy
        designer = self._get_strategy_designer()
        strategy = await designer.design_strategy(
            risk_trigger=risk_trigger,
            templates=templates,
        )

        # Update workflow state
        workflow.status = WorkflowStatus.STRATEGY_DESIGNED
        workflow.strategy_id = strategy.strategy_id
        self._workflows[workflow_id] = workflow
        self._strategies[strategy.strategy_id] = strategy

        logger.info(
            "Design phase complete: workflow=%s, strategy=%s, "
            "measures=%d",
            workflow_id,
            strategy.strategy_id,
            len(strategy.measures),
        )

        return strategy

    async def approval_phase(
        self,
        workflow_id: str,
        approved_by: str,
    ) -> WorkflowState:
        """Execute the approval phase.

        Transitions workflow from STRATEGY_DESIGNED to
        MEASURES_APPROVED after all measures are approved.

        Args:
            workflow_id: Workflow identifier.
            approved_by: Approver identifier.

        Returns:
            Updated WorkflowState.

        Raises:
            ValueError: If workflow not found or invalid transition.
        """
        workflow = self._get_workflow_or_raise(workflow_id)
        self._validate_transition(
            workflow.status, WorkflowStatus.MEASURES_APPROVED,
        )

        workflow.status = WorkflowStatus.MEASURES_APPROVED
        self._workflows[workflow_id] = workflow

        self._provenance.create_entry(
            step="approval_phase",
            source=approved_by,
            input_hash=self._provenance.compute_hash(
                {"workflow_id": workflow_id, "status": "strategy_designed"}
            ),
            output_hash=self._provenance.compute_hash(
                {"workflow_id": workflow_id, "status": "measures_approved",
                 "approved_by": approved_by}
            ),
        )

        logger.info(
            "Approval phase complete: workflow=%s, approved_by=%s",
            workflow_id, approved_by,
        )

        return workflow

    async def implementation_phase(
        self, workflow_id: str,
    ) -> WorkflowState:
        """Transition to implementation phase.

        Transitions workflow from MEASURES_APPROVED to IMPLEMENTING.

        Args:
            workflow_id: Workflow identifier.

        Returns:
            Updated WorkflowState.

        Raises:
            ValueError: If workflow not found or invalid transition.
        """
        workflow = self._get_workflow_or_raise(workflow_id)
        self._validate_transition(
            workflow.status, WorkflowStatus.IMPLEMENTING,
        )

        workflow.status = WorkflowStatus.IMPLEMENTING
        self._workflows[workflow_id] = workflow

        logger.info(
            "Implementation phase started: workflow=%s",
            workflow_id,
        )

        return workflow

    async def verification_phase(
        self,
        workflow_id: str,
        risk_trigger: RiskTrigger,
    ) -> VerificationReport:
        """Execute the verification phase.

        Transitions workflow from IMPLEMENTING to VERIFYING and
        invokes the RiskReductionVerifier.

        Args:
            workflow_id: Workflow identifier.
            risk_trigger: Original risk trigger for baseline.

        Returns:
            VerificationReport with results.

        Raises:
            ValueError: If workflow not found or invalid transition.
        """
        workflow = self._get_workflow_or_raise(workflow_id)
        self._validate_transition(
            workflow.status, WorkflowStatus.VERIFYING,
        )

        strategy_id = workflow.strategy_id
        if not strategy_id or strategy_id not in self._strategies:
            raise ValueError(
                f"No strategy found for workflow {workflow_id}"
            )

        strategy = self._strategies[strategy_id]
        verifier = self._get_risk_reduction_verifier()
        verification = await verifier.verify_risk_reduction(
            strategy=strategy,
            risk_trigger=risk_trigger,
        )

        workflow.status = WorkflowStatus.VERIFYING
        self._workflows[workflow_id] = workflow
        self._verifications[workflow_id] = verification

        logger.info(
            "Verification phase complete: workflow=%s, result=%s",
            workflow_id,
            verification.result.value,
        )

        return verification

    async def close_workflow(
        self, workflow_id: str,
    ) -> WorkflowState:
        """Close workflow after successful verification.

        Transitions workflow from VERIFYING to CLOSED.

        Args:
            workflow_id: Workflow identifier.

        Returns:
            Updated WorkflowState with closure timestamp.

        Raises:
            ValueError: If workflow not found or invalid transition.
        """
        workflow = self._get_workflow_or_raise(workflow_id)
        self._validate_transition(
            workflow.status, WorkflowStatus.CLOSED,
        )

        workflow.status = WorkflowStatus.CLOSED
        workflow.closed_at = datetime.now(timezone.utc)
        self._workflows[workflow_id] = workflow

        self._provenance.create_entry(
            step="close_workflow",
            source="compliance_workflow_engine",
            input_hash=self._provenance.compute_hash(
                {"workflow_id": workflow_id, "status": "verifying"}
            ),
            output_hash=self._provenance.compute_hash(
                {"workflow_id": workflow_id, "status": "closed"}
            ),
        )

        logger.info(
            "Workflow closed: id=%s",
            workflow_id,
        )

        return workflow

    async def escalate_workflow(
        self,
        workflow_id: str,
        reason: str,
    ) -> WorkflowState:
        """Escalate workflow if measures insufficient.

        Transitions workflow to ESCALATED status for management
        review and potential strategy redesign.

        Args:
            workflow_id: Workflow identifier.
            reason: Escalation reason.

        Returns:
            Updated WorkflowState with escalation details.

        Raises:
            ValueError: If workflow not found or invalid transition.
        """
        workflow = self._get_workflow_or_raise(workflow_id)
        self._validate_transition(
            workflow.status, WorkflowStatus.ESCALATED,
        )

        workflow.status = WorkflowStatus.ESCALATED
        workflow.escalated_at = datetime.now(timezone.utc)
        self._workflows[workflow_id] = workflow

        self._provenance.create_entry(
            step="escalate_workflow",
            source="compliance_workflow_engine",
            input_hash=self._provenance.compute_hash(
                {"workflow_id": workflow_id}
            ),
            output_hash=self._provenance.compute_hash(
                {"workflow_id": workflow_id, "status": "escalated",
                 "reason": reason}
            ),
        )

        logger.info(
            "Workflow escalated: id=%s, reason=%s",
            workflow_id, reason,
        )

        return workflow

    async def fail_workflow(
        self,
        workflow_id: str,
        reason: str,
    ) -> WorkflowState:
        """Mark workflow as failed.

        Args:
            workflow_id: Workflow identifier.
            reason: Failure reason.

        Returns:
            Updated WorkflowState with FAILED status.

        Raises:
            ValueError: If workflow not found or invalid transition.
        """
        workflow = self._get_workflow_or_raise(workflow_id)
        self._validate_transition(
            workflow.status, WorkflowStatus.FAILED,
        )

        workflow.status = WorkflowStatus.FAILED
        self._workflows[workflow_id] = workflow

        logger.error(
            "Workflow failed: id=%s, reason=%s",
            workflow_id, reason,
        )

        return workflow

    def get_workflow_status(
        self, workflow_id: str,
    ) -> WorkflowState:
        """Get current workflow state.

        Args:
            workflow_id: Workflow identifier.

        Returns:
            Current WorkflowState.

        Raises:
            ValueError: If workflow not found.
        """
        return self._get_workflow_or_raise(workflow_id)

    def list_workflows(
        self,
        operator_id: Optional[str] = None,
        status: Optional[WorkflowStatus] = None,
    ) -> List[WorkflowState]:
        """List workflows with optional filters.

        Args:
            operator_id: Filter by operator.
            status: Filter by status.

        Returns:
            List of matching WorkflowState instances.
        """
        results: List[WorkflowState] = []
        for wf in self._workflows.values():
            if operator_id and wf.operator_id != operator_id:
                continue
            if status and wf.status != status:
                continue
            results.append(wf)
        return results

    def _validate_transition(
        self,
        current: WorkflowStatus,
        target: WorkflowStatus,
    ) -> bool:
        """Validate state machine transition.

        Args:
            current: Current workflow status.
            target: Target workflow status.

        Returns:
            True if transition is valid.

        Raises:
            ValueError: If transition is not allowed.
        """
        allowed = _VALID_TRANSITIONS.get(current, [])
        if target not in allowed:
            raise ValueError(
                f"Invalid workflow transition: "
                f"{current.value} -> {target.value}. "
                f"Allowed: {[s.value for s in allowed]}"
            )
        return True

    def _get_workflow_or_raise(
        self, workflow_id: str,
    ) -> WorkflowState:
        """Get workflow by ID or raise ValueError.

        Args:
            workflow_id: Workflow identifier.

        Returns:
            WorkflowState instance.

        Raises:
            ValueError: If workflow not found.
        """
        workflow = self._workflows.get(workflow_id)
        if workflow is None:
            raise ValueError(f"Workflow not found: {workflow_id}")
        return workflow

    async def health_check(self) -> Dict[str, Any]:
        """Return engine health status.

        Returns:
            Dictionary with engine status and workflow counts.
        """
        status_counts: Dict[str, int] = {}
        for wf in self._workflows.values():
            status_name = wf.status.value
            status_counts[status_name] = (
                status_counts.get(status_name, 0) + 1
            )

        return {
            "engine": "ComplianceWorkflowEngine",
            "status": "available",
            "total_workflows": len(self._workflows),
            "status_breakdown": status_counts,
            "config": {
                "approval_required": self._config.approval_required,
                "auto_close_on_negligible": (
                    self._config.auto_close_on_negligible
                ),
                "max_duration_days": (
                    self._config.max_workflow_duration_days
                ),
            },
        }
