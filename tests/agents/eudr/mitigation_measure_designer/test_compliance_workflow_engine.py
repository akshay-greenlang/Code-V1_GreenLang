# -*- coding: utf-8 -*-
"""
Unit tests for ComplianceWorkflowEngine - AGENT-EUDR-029

Tests the full compliance workflow lifecycle: initiate, design phase,
approval, implementation, verification, close, escalate, and fail.
Also tests state transition validation and listing/retrieval.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from greenlang.agents.eudr.mitigation_measure_designer.compliance_workflow_engine import (
    ComplianceWorkflowEngine,
    _VALID_TRANSITIONS,
)
from greenlang.agents.eudr.mitigation_measure_designer.config import (
    MitigationMeasureDesignerConfig,
)
from greenlang.agents.eudr.mitigation_measure_designer.models import (
    EUDRCommodity,
    MitigationStrategy,
    RiskLevel,
    RiskTrigger,
    VerificationReport,
    VerificationResult,
    WorkflowState,
    WorkflowStatus,
)
from greenlang.agents.eudr.mitigation_measure_designer.provenance import (
    ProvenanceTracker,
)


@pytest.fixture
def config():
    return MitigationMeasureDesignerConfig()


@pytest.fixture
def engine(config):
    return ComplianceWorkflowEngine(
        config=config, provenance=ProvenanceTracker(),
    )


class TestInitiateWorkflow:
    """Test initiate_workflow."""

    @pytest.mark.asyncio
    async def test_initiate_returns_workflow_state(
        self, engine, sample_risk_trigger,
    ):
        wf = await engine.initiate_workflow(sample_risk_trigger)
        assert isinstance(wf, WorkflowState)
        assert wf.workflow_id.startswith("wfl-")
        assert wf.status == WorkflowStatus.INITIATED

    @pytest.mark.asyncio
    async def test_initiate_sets_operator_and_commodity(
        self, engine, sample_risk_trigger,
    ):
        wf = await engine.initiate_workflow(sample_risk_trigger)
        assert wf.operator_id == sample_risk_trigger.operator_id
        assert wf.commodity == sample_risk_trigger.commodity

    @pytest.mark.asyncio
    async def test_initiate_sets_started_at(
        self, engine, sample_risk_trigger,
    ):
        wf = await engine.initiate_workflow(sample_risk_trigger)
        assert wf.started_at is not None

    @pytest.mark.asyncio
    async def test_initiate_stores_provenance_hash(
        self, engine, sample_risk_trigger,
    ):
        wf = await engine.initiate_workflow(sample_risk_trigger)
        assert wf.provenance_hash is not None
        assert len(wf.provenance_hash) == 64


class TestWorkflowTransitions:
    """Test valid and invalid state transitions."""

    def test_valid_transitions_from_initiated(self):
        allowed = _VALID_TRANSITIONS[WorkflowStatus.INITIATED]
        assert WorkflowStatus.STRATEGY_DESIGNED in allowed
        assert WorkflowStatus.FAILED in allowed

    def test_valid_transitions_from_verifying(self):
        allowed = _VALID_TRANSITIONS[WorkflowStatus.VERIFYING]
        assert WorkflowStatus.CLOSED in allowed
        assert WorkflowStatus.ESCALATED in allowed
        assert WorkflowStatus.IMPLEMENTING in allowed

    def test_no_transitions_from_closed(self):
        allowed = _VALID_TRANSITIONS[WorkflowStatus.CLOSED]
        assert len(allowed) == 0

    def test_no_transitions_from_failed(self):
        allowed = _VALID_TRANSITIONS[WorkflowStatus.FAILED]
        assert len(allowed) == 0

    def test_validate_transition_raises_on_invalid(self, engine):
        with pytest.raises(ValueError, match="Invalid workflow transition"):
            engine._validate_transition(
                WorkflowStatus.INITIATED,
                WorkflowStatus.CLOSED,
            )

    def test_validate_transition_returns_true_on_valid(self, engine):
        assert engine._validate_transition(
            WorkflowStatus.INITIATED,
            WorkflowStatus.STRATEGY_DESIGNED,
        ) is True


class TestApprovalPhase:
    """Test approval_phase transition."""

    @pytest.mark.asyncio
    async def test_approval_transitions_to_measures_approved(
        self, engine, sample_risk_trigger,
    ):
        wf = await engine.initiate_workflow(sample_risk_trigger)
        # Must manually set to STRATEGY_DESIGNED to test approval
        engine._workflows[wf.workflow_id].status = (
            WorkflowStatus.STRATEGY_DESIGNED
        )
        approved_wf = await engine.approval_phase(
            wf.workflow_id, "admin@test.com",
        )
        assert approved_wf.status == WorkflowStatus.MEASURES_APPROVED


class TestImplementationPhase:
    """Test implementation_phase transition."""

    @pytest.mark.asyncio
    async def test_implementation_transitions_to_implementing(
        self, engine, sample_risk_trigger,
    ):
        wf = await engine.initiate_workflow(sample_risk_trigger)
        engine._workflows[wf.workflow_id].status = (
            WorkflowStatus.MEASURES_APPROVED
        )
        impl_wf = await engine.implementation_phase(wf.workflow_id)
        assert impl_wf.status == WorkflowStatus.IMPLEMENTING

    @pytest.mark.asyncio
    async def test_implementation_from_wrong_state_raises(
        self, engine, sample_risk_trigger,
    ):
        wf = await engine.initiate_workflow(sample_risk_trigger)
        # Still INITIATED, not MEASURES_APPROVED
        with pytest.raises(ValueError, match="Invalid workflow transition"):
            await engine.implementation_phase(wf.workflow_id)


class TestCloseWorkflow:
    """Test close_workflow."""

    @pytest.mark.asyncio
    async def test_close_from_verifying(
        self, engine, sample_risk_trigger,
    ):
        wf = await engine.initiate_workflow(sample_risk_trigger)
        engine._workflows[wf.workflow_id].status = (
            WorkflowStatus.VERIFYING
        )
        closed_wf = await engine.close_workflow(wf.workflow_id)
        assert closed_wf.status == WorkflowStatus.CLOSED
        assert closed_wf.closed_at is not None

    @pytest.mark.asyncio
    async def test_close_from_implementing_raises(
        self, engine, sample_risk_trigger,
    ):
        wf = await engine.initiate_workflow(sample_risk_trigger)
        engine._workflows[wf.workflow_id].status = (
            WorkflowStatus.IMPLEMENTING
        )
        with pytest.raises(ValueError, match="Invalid workflow transition"):
            await engine.close_workflow(wf.workflow_id)


class TestEscalateWorkflow:
    """Test escalate_workflow."""

    @pytest.mark.asyncio
    async def test_escalate_from_implementing(
        self, engine, sample_risk_trigger,
    ):
        wf = await engine.initiate_workflow(sample_risk_trigger)
        engine._workflows[wf.workflow_id].status = (
            WorkflowStatus.IMPLEMENTING
        )
        esc_wf = await engine.escalate_workflow(
            wf.workflow_id, "Measures insufficient",
        )
        assert esc_wf.status == WorkflowStatus.ESCALATED
        assert esc_wf.escalated_at is not None

    @pytest.mark.asyncio
    async def test_escalate_from_verifying(
        self, engine, sample_risk_trigger,
    ):
        wf = await engine.initiate_workflow(sample_risk_trigger)
        engine._workflows[wf.workflow_id].status = (
            WorkflowStatus.VERIFYING
        )
        esc_wf = await engine.escalate_workflow(
            wf.workflow_id, "Post-verification escalation",
        )
        assert esc_wf.status == WorkflowStatus.ESCALATED


class TestFailWorkflow:
    """Test fail_workflow."""

    @pytest.mark.asyncio
    async def test_fail_from_initiated(
        self, engine, sample_risk_trigger,
    ):
        wf = await engine.initiate_workflow(sample_risk_trigger)
        failed_wf = await engine.fail_workflow(
            wf.workflow_id, "Configuration error",
        )
        assert failed_wf.status == WorkflowStatus.FAILED

    @pytest.mark.asyncio
    async def test_fail_from_closed_raises(
        self, engine, sample_risk_trigger,
    ):
        wf = await engine.initiate_workflow(sample_risk_trigger)
        engine._workflows[wf.workflow_id].status = (
            WorkflowStatus.CLOSED
        )
        with pytest.raises(ValueError, match="Invalid workflow transition"):
            await engine.fail_workflow(wf.workflow_id, "Too late")


class TestGetAndListWorkflows:
    """Test get_workflow_status and list_workflows."""

    @pytest.mark.asyncio
    async def test_get_workflow_status(
        self, engine, sample_risk_trigger,
    ):
        wf = await engine.initiate_workflow(sample_risk_trigger)
        retrieved = engine.get_workflow_status(wf.workflow_id)
        assert retrieved.workflow_id == wf.workflow_id
        assert retrieved.status == WorkflowStatus.INITIATED

    def test_get_workflow_nonexistent_raises(self, engine):
        with pytest.raises(ValueError, match="Workflow not found"):
            engine.get_workflow_status("nonexistent-wfl")

    @pytest.mark.asyncio
    async def test_list_workflows_no_filter(
        self, engine, sample_risk_trigger,
    ):
        await engine.initiate_workflow(sample_risk_trigger)
        await engine.initiate_workflow(sample_risk_trigger)
        workflows = engine.list_workflows()
        assert len(workflows) == 2

    @pytest.mark.asyncio
    async def test_list_workflows_filter_by_operator(
        self, engine, sample_risk_trigger,
    ):
        await engine.initiate_workflow(sample_risk_trigger)
        workflows = engine.list_workflows(
            operator_id=sample_risk_trigger.operator_id,
        )
        assert len(workflows) == 1

    @pytest.mark.asyncio
    async def test_list_workflows_filter_by_status(
        self, engine, sample_risk_trigger,
    ):
        wf = await engine.initiate_workflow(sample_risk_trigger)
        engine._workflows[wf.workflow_id].status = (
            WorkflowStatus.CLOSED
        )
        # Should return 0 INITIATED workflows, 1 CLOSED
        initiated = engine.list_workflows(status=WorkflowStatus.INITIATED)
        closed = engine.list_workflows(status=WorkflowStatus.CLOSED)
        assert len(initiated) == 0
        assert len(closed) == 1


class TestHealthCheck:
    """Test health_check method."""

    @pytest.mark.asyncio
    async def test_health_check_available(self, engine):
        health = await engine.health_check()
        assert health["engine"] == "ComplianceWorkflowEngine"
        assert health["status"] == "available"
        assert health["total_workflows"] == 0
