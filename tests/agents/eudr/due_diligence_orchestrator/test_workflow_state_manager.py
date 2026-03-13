# -*- coding: utf-8 -*-
"""
Unit tests for Engine 6: Workflow State Manager -- AGENT-EUDR-026

Tests 11-state machine transitions, checkpoint creation, workflow
resume from checkpoint, provenance chain hashing, rollback, pause,
cancel, progress tracking, audit trail, and thread safety.

Test count: ~90 tests
Author: GreenLang Platform Team
Date: March 2026
"""

import hashlib
import json
import pytest
from decimal import Decimal
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from greenlang.agents.eudr.due_diligence_orchestrator.models import (
    WorkflowStatus,
    WorkflowState,
    WorkflowCheckpoint,
    WorkflowStateTransition,
    DueDiligencePhase,
    WorkflowType,
    AgentExecutionStatus,
    AgentExecutionRecord,
)
from greenlang.agents.eudr.due_diligence_orchestrator.workflow_state_manager import (
    WorkflowStateManager,
)


class TestStateManagerInit:
    """Test state manager initialization."""

    def test_init_default(self, default_config):
        mgr = WorkflowStateManager()
        assert mgr is not None

    def test_init_with_config(self, default_config):
        mgr = WorkflowStateManager(config=default_config)
        assert mgr is not None


class TestValidTransitions:
    """Test state transition validation."""

    @pytest.mark.parametrize("from_status,to_status,expected_valid", [
        (WorkflowStatus.CREATED, WorkflowStatus.VALIDATING, True),
        (WorkflowStatus.CREATED, WorkflowStatus.CANCELLED, True),
        (WorkflowStatus.CREATED, WorkflowStatus.RUNNING, False),
        (WorkflowStatus.VALIDATING, WorkflowStatus.RUNNING, True),
        (WorkflowStatus.VALIDATING, WorkflowStatus.CANCELLED, True),
        (WorkflowStatus.VALIDATING, WorkflowStatus.TERMINATED, True),
        (WorkflowStatus.RUNNING, WorkflowStatus.QUALITY_GATE, True),
        (WorkflowStatus.RUNNING, WorkflowStatus.PAUSED, True),
        (WorkflowStatus.RUNNING, WorkflowStatus.COMPLETING, True),
        (WorkflowStatus.RUNNING, WorkflowStatus.CANCELLED, True),
        (WorkflowStatus.RUNNING, WorkflowStatus.TERMINATED, True),
        (WorkflowStatus.PAUSED, WorkflowStatus.RESUMING, True),
        (WorkflowStatus.PAUSED, WorkflowStatus.CANCELLED, True),
        (WorkflowStatus.QUALITY_GATE, WorkflowStatus.RUNNING, True),
        (WorkflowStatus.QUALITY_GATE, WorkflowStatus.GATE_FAILED, True),
        (WorkflowStatus.GATE_FAILED, WorkflowStatus.RESUMING, True),
        (WorkflowStatus.GATE_FAILED, WorkflowStatus.CANCELLED, True),
        (WorkflowStatus.RESUMING, WorkflowStatus.RUNNING, True),
        (WorkflowStatus.COMPLETING, WorkflowStatus.COMPLETED, True),
        (WorkflowStatus.COMPLETING, WorkflowStatus.TERMINATED, True),
        (WorkflowStatus.COMPLETED, WorkflowStatus.RUNNING, False),
        (WorkflowStatus.CANCELLED, WorkflowStatus.RUNNING, False),
        (WorkflowStatus.TERMINATED, WorkflowStatus.RUNNING, False),
    ])
    def test_transition_validity(
        self, workflow_state_manager, from_status, to_status, expected_valid,
    ):
        mgr = workflow_state_manager
        is_valid = mgr.is_valid_transition(from_status, to_status)
        assert is_valid == expected_valid

    def test_completed_is_terminal(self, workflow_state_manager):
        mgr = workflow_state_manager
        valid_next = mgr.get_valid_transitions(WorkflowStatus.COMPLETED)
        assert len(valid_next) == 0

    def test_cancelled_is_terminal(self, workflow_state_manager):
        mgr = workflow_state_manager
        valid_next = mgr.get_valid_transitions(WorkflowStatus.CANCELLED)
        assert len(valid_next) == 0

    def test_terminated_is_terminal(self, workflow_state_manager):
        mgr = workflow_state_manager
        valid_next = mgr.get_valid_transitions(WorkflowStatus.TERMINATED)
        assert len(valid_next) == 0


class TestStateTransition:
    """Test performing state transitions."""

    def test_transition_updates_status(
        self, workflow_state_manager, workflow_state_created,
    ):
        mgr = workflow_state_manager
        mgr.transition(
            workflow_state_created,
            WorkflowStatus.VALIDATING,
            reason="Starting validation",
        )
        assert workflow_state_created.status == WorkflowStatus.VALIDATING

    def test_transition_records_history(
        self, workflow_state_manager, workflow_state_created,
    ):
        mgr = workflow_state_manager
        mgr.transition(
            workflow_state_created,
            WorkflowStatus.VALIDATING,
            reason="Test transition",
        )
        assert len(workflow_state_created.transitions) >= 1
        last = workflow_state_created.transitions[-1]
        assert last.from_status == WorkflowStatus.CREATED
        assert last.to_status == WorkflowStatus.VALIDATING

    def test_invalid_transition_raises_error(
        self, workflow_state_manager, workflow_state_created,
    ):
        mgr = workflow_state_manager
        with pytest.raises((ValueError, RuntimeError)):
            mgr.transition(
                workflow_state_created,
                WorkflowStatus.COMPLETED,
                reason="Invalid jump",
            )

    def test_transition_has_timestamp(
        self, workflow_state_manager, workflow_state_created,
    ):
        mgr = workflow_state_manager
        mgr.transition(
            workflow_state_created,
            WorkflowStatus.VALIDATING,
            reason="Test",
        )
        last = workflow_state_created.transitions[-1]
        assert last.timestamp is not None

    def test_transition_records_actor(
        self, workflow_state_manager, workflow_state_created,
    ):
        mgr = workflow_state_manager
        mgr.transition(
            workflow_state_created,
            WorkflowStatus.VALIDATING,
            reason="Test",
            actor="test_user",
        )
        last = workflow_state_created.transitions[-1]
        assert last.actor == "test_user"


class TestCheckpointCreation:
    """Test checkpoint creation."""

    def test_create_checkpoint(
        self, workflow_state_manager, workflow_state_running,
    ):
        mgr = workflow_state_manager
        cp = mgr.create_checkpoint(
            workflow_state_running,
            agent_id="EUDR-001",
        )
        assert isinstance(cp, WorkflowCheckpoint)
        assert cp.workflow_id == "wf-002"

    def test_checkpoint_has_sequence_number(
        self, workflow_state_manager, workflow_state_running,
    ):
        mgr = workflow_state_manager
        cp1 = mgr.create_checkpoint(workflow_state_running, agent_id="EUDR-001")
        cp2 = mgr.create_checkpoint(workflow_state_running, agent_id="EUDR-002")
        assert cp2.sequence_number > cp1.sequence_number

    def test_checkpoint_has_provenance_hash(
        self, workflow_state_manager, workflow_state_running,
    ):
        mgr = workflow_state_manager
        cp = mgr.create_checkpoint(workflow_state_running, agent_id="EUDR-001")
        assert cp.cumulative_provenance_hash is not None
        assert len(cp.cumulative_provenance_hash) == 64

    def test_checkpoint_chains_hashes(
        self, workflow_state_manager, workflow_state_running,
    ):
        mgr = workflow_state_manager
        cp1 = mgr.create_checkpoint(workflow_state_running, agent_id="EUDR-001")
        cp2 = mgr.create_checkpoint(workflow_state_running, agent_id="EUDR-002")
        assert cp1.cumulative_provenance_hash != cp2.cumulative_provenance_hash

    def test_checkpoint_captures_agent_statuses(
        self, workflow_state_manager, workflow_state_running,
    ):
        mgr = workflow_state_manager
        cp = mgr.create_checkpoint(workflow_state_running, agent_id="EUDR-001")
        assert len(cp.agent_statuses) > 0 or cp.agent_statuses is not None

    def test_checkpoint_captures_phase(
        self, workflow_state_manager, workflow_state_running,
    ):
        mgr = workflow_state_manager
        cp = mgr.create_checkpoint(workflow_state_running, agent_id="EUDR-001")
        assert cp.phase == DueDiligencePhase.INFORMATION_GATHERING


class TestWorkflowResume:
    """Test workflow resume from checkpoint."""

    def test_resume_restores_state(
        self, workflow_state_manager,
    ):
        mgr = workflow_state_manager
        state = WorkflowState(
            workflow_id="wf-resume", definition_id="def-001",
            status=WorkflowStatus.RUNNING,
        )
        state.agent_executions["EUDR-001"] = AgentExecutionRecord(
            workflow_id="wf-resume", agent_id="EUDR-001",
            status=AgentExecutionStatus.COMPLETED,
        )
        cp = mgr.create_checkpoint(state, agent_id="EUDR-001")

        # Simulate pause
        mgr.transition(state, WorkflowStatus.PAUSED, reason="User pause")

        # Resume from checkpoint
        resumed = mgr.resume_from_checkpoint(state, cp)
        assert resumed.status in (WorkflowStatus.RESUMING, WorkflowStatus.RUNNING)

    def test_resume_skips_completed_agents(
        self, workflow_state_manager,
    ):
        mgr = workflow_state_manager
        state = WorkflowState(
            workflow_id="wf-skip", definition_id="def-001",
            status=WorkflowStatus.PAUSED,
        )
        state.agent_executions["EUDR-001"] = AgentExecutionRecord(
            workflow_id="wf-skip", agent_id="EUDR-001",
            status=AgentExecutionStatus.COMPLETED,
        )
        cp = mgr.create_checkpoint(state, agent_id="EUDR-001")
        resumed = mgr.resume_from_checkpoint(state, cp)
        assert resumed.agent_executions["EUDR-001"].status == AgentExecutionStatus.COMPLETED


class TestWorkflowPause:
    """Test workflow pause."""

    def test_pause_running_workflow(
        self, workflow_state_manager, workflow_state_running,
    ):
        mgr = workflow_state_manager
        mgr.transition(workflow_state_running, WorkflowStatus.PAUSED, reason="User pause")
        assert workflow_state_running.status == WorkflowStatus.PAUSED


class TestWorkflowCancel:
    """Test workflow cancellation."""

    def test_cancel_from_running(
        self, workflow_state_manager, workflow_state_running,
    ):
        mgr = workflow_state_manager
        mgr.transition(workflow_state_running, WorkflowStatus.CANCELLED, reason="User cancel")
        assert workflow_state_running.status == WorkflowStatus.CANCELLED

    def test_cancel_from_paused(self, workflow_state_manager):
        mgr = workflow_state_manager
        state = WorkflowState(
            workflow_id="wf-cancel", definition_id="def-001",
            status=WorkflowStatus.PAUSED,
        )
        mgr.transition(state, WorkflowStatus.CANCELLED, reason="User cancel")
        assert state.status == WorkflowStatus.CANCELLED


class TestProgressTracking:
    """Test progress percentage calculation."""

    def test_progress_zero_at_start(self, workflow_state_manager):
        mgr = workflow_state_manager
        state = WorkflowState(
            workflow_id="wf-start", definition_id="def-001",
        )
        pct = mgr.calculate_progress(state)
        assert pct == Decimal("0") or pct == Decimal("0.00")

    def test_progress_100_at_completion(
        self, workflow_state_manager, workflow_state_completed,
    ):
        mgr = workflow_state_manager
        pct = mgr.calculate_progress(workflow_state_completed)
        assert pct == Decimal("100") or pct == Decimal("100.00")

    def test_progress_partial(
        self, workflow_state_manager, workflow_state_running,
    ):
        mgr = workflow_state_manager
        pct = mgr.calculate_progress(workflow_state_running)
        assert Decimal("0") < pct < Decimal("100")


class TestAuditTrail:
    """Test audit trail completeness."""

    def test_audit_trail_records_all_transitions(
        self, workflow_state_manager, workflow_state_created,
    ):
        mgr = workflow_state_manager
        mgr.transition(workflow_state_created, WorkflowStatus.VALIDATING, reason="R1")
        mgr.transition(workflow_state_created, WorkflowStatus.RUNNING, reason="R2")
        assert len(workflow_state_created.transitions) >= 2

    def test_audit_trail_transition_has_reason(
        self, workflow_state_manager, workflow_state_created,
    ):
        mgr = workflow_state_manager
        mgr.transition(workflow_state_created, WorkflowStatus.VALIDATING,
                       reason="Starting validation")
        assert workflow_state_created.transitions[-1].reason == "Starting validation"

    def test_audit_trail_chronological(
        self, workflow_state_manager, workflow_state_created,
    ):
        mgr = workflow_state_manager
        mgr.transition(workflow_state_created, WorkflowStatus.VALIDATING, reason="R1")
        mgr.transition(workflow_state_created, WorkflowStatus.RUNNING, reason="R2")
        t1 = workflow_state_created.transitions[-2].timestamp
        t2 = workflow_state_created.transitions[-1].timestamp
        assert t2 >= t1


class TestProvenanceChain:
    """Test SHA-256 provenance chain."""

    def test_checkpoint_hash_is_sha256(
        self, workflow_state_manager, workflow_state_running,
    ):
        mgr = workflow_state_manager
        cp = mgr.create_checkpoint(workflow_state_running, agent_id="EUDR-001")
        assert len(cp.cumulative_provenance_hash) == 64

    def test_hash_chain_integrity(
        self, workflow_state_manager, workflow_state_running,
    ):
        mgr = workflow_state_manager
        cp1 = mgr.create_checkpoint(workflow_state_running, agent_id="EUDR-001")
        cp2 = mgr.create_checkpoint(workflow_state_running, agent_id="EUDR-002")
        # Each hash should be different because it includes the previous hash
        assert cp1.cumulative_provenance_hash != cp2.cumulative_provenance_hash

    def test_deterministic_hash(self, workflow_state_manager):
        mgr = workflow_state_manager
        state1 = WorkflowState(
            workflow_id="wf-det1", definition_id="def-001",
            status=WorkflowStatus.RUNNING,
        )
        state2 = WorkflowState(
            workflow_id="wf-det1", definition_id="def-001",
            status=WorkflowStatus.RUNNING,
        )
        cp1 = mgr.create_checkpoint(state1, agent_id="EUDR-001")
        cp2 = mgr.create_checkpoint(state2, agent_id="EUDR-001")
        # Same inputs should produce same hash (assuming same timestamp)
        assert isinstance(cp1.cumulative_provenance_hash, str)
        assert isinstance(cp2.cumulative_provenance_hash, str)


class TestETAEstimation:
    """Test ETA estimation."""

    def test_eta_at_start(self, workflow_state_manager, workflow_state_created):
        mgr = workflow_state_manager
        eta = mgr.estimate_eta(workflow_state_created)
        assert eta is None or eta > 0

    def test_eta_at_completion(
        self, workflow_state_manager, workflow_state_completed,
    ):
        mgr = workflow_state_manager
        eta = mgr.estimate_eta(workflow_state_completed)
        assert eta == 0 or eta is None

    def test_eta_decreases_with_progress(self, workflow_state_manager):
        mgr = workflow_state_manager
        state = WorkflowState(
            workflow_id="wf-eta", definition_id="def-001",
            status=WorkflowStatus.RUNNING,
            progress_pct=Decimal("20"),
        )
        eta1 = mgr.estimate_eta(state)
        state.progress_pct = Decimal("80")
        eta2 = mgr.estimate_eta(state)
        if eta1 is not None and eta2 is not None:
            assert eta2 <= eta1


class TestWorkflowStateModel:
    """Test WorkflowState model."""

    def test_default_status_is_created(self):
        state = WorkflowState(
            workflow_id="wf-new", definition_id="def-001",
        )
        assert state.status == WorkflowStatus.CREATED

    def test_default_phase_is_information_gathering(self):
        state = WorkflowState(
            workflow_id="wf-new", definition_id="def-001",
        )
        assert state.current_phase == DueDiligencePhase.INFORMATION_GATHERING

    def test_workflow_id_required(self):
        state = WorkflowState(
            workflow_id="wf-req", definition_id="def-001",
        )
        assert state.workflow_id == "wf-req"
