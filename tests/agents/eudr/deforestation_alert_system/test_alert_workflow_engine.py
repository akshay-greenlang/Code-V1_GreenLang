# -*- coding: utf-8 -*-
"""
Unit tests for AlertWorkflowEngine - AGENT-EUDR-020 Engine 7

Tests alert lifecycle workflow management including triage, assignment,
investigation, resolution, escalation, false positive marking, SLA tracking,
state transition validation, audit trail recording, and provenance tracking.

Workflow States:
    PENDING -> TRIAGED -> INVESTIGATING -> RESOLVED / ESCALATED / FALSE_POSITIVE
    ESCALATED -> INVESTIGATING / RESOLVED
    RESOLVED -> CLOSED / REOPEN (-> INVESTIGATING)
    FALSE_POSITIVE -> REOPEN (-> INVESTIGATING)

SLA Defaults: Triage 4h, Investigation 48h, Resolution 168h (7 days).

Coverage targets: 85%+ across all AlertWorkflowEngine methods.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-020
Agent ID: GL-EUDR-DAS-020
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from greenlang.agents.eudr.deforestation_alert_system.engines.alert_workflow_engine import (
    AlertPriority,
    AlertWorkflowEngine,
    DEFAULT_MAX_ESCALATION_LEVELS,
    DEFAULT_SLA_INVESTIGATION_HOURS,
    DEFAULT_SLA_RESOLUTION_HOURS,
    DEFAULT_SLA_TRIAGE_HOURS,
    ESCALATION_SLA_FACTOR,
    PRIORITY_SLA_MULTIPLIERS,
    SLAConfig,
    SLAReport,
    SLAStatus,
    TERMINAL_STATES,
    VALID_TRANSITIONS,
    WorkflowAction,
    WorkflowNote,
    WorkflowState,
    WorkflowStatus,
    WorkflowTransition,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine() -> AlertWorkflowEngine:
    """Create a default AlertWorkflowEngine instance."""
    return AlertWorkflowEngine()


@pytest.fixture
def engine_no_auto_escalate() -> AlertWorkflowEngine:
    """Create engine with auto-escalation disabled."""
    return AlertWorkflowEngine(auto_escalate=False)


@pytest.fixture
def engine_custom_sla() -> AlertWorkflowEngine:
    """Create engine with custom SLA configuration."""
    config = SLAConfig(triage_hours=2, investigation_hours=24, resolution_hours=72)
    return AlertWorkflowEngine(sla_config=config, max_escalation_levels=5)


@pytest.fixture
def pending_alert(engine: AlertWorkflowEngine) -> Dict[str, Any]:
    """Create a PENDING alert workflow."""
    return engine.create_workflow("ALERT-001", priority=AlertPriority.HIGH.value)


@pytest.fixture
def triaged_alert(
    engine: AlertWorkflowEngine, pending_alert: Dict[str, Any]
) -> Dict[str, Any]:
    """Create a TRIAGED alert workflow."""
    return engine.triage(
        "ALERT-001", AlertPriority.HIGH.value, "analyst-1", "Initial triage"
    )


@pytest.fixture
def investigating_alert(
    engine: AlertWorkflowEngine, triaged_alert: Dict[str, Any]
) -> Dict[str, Any]:
    """Create an INVESTIGATING alert workflow."""
    return engine.investigate("ALERT-001", "analyst-1", "Starting investigation")


@pytest.fixture
def resolved_alert(
    engine: AlertWorkflowEngine, investigating_alert: Dict[str, Any]
) -> Dict[str, Any]:
    """Create a RESOLVED alert workflow."""
    return engine.resolve(
        "ALERT-001",
        "Deforestation confirmed and supplier notified",
        "analyst-1",
        "Resolution complete",
    )


# ---------------------------------------------------------------------------
# TestTriage
# ---------------------------------------------------------------------------


class TestTriage:
    """Tests for triage operation: PENDING -> TRIAGED."""

    def test_triage_pending_to_triaged(
        self, engine: AlertWorkflowEngine, pending_alert: Dict[str, Any]
    ) -> None:
        """Triage moves alert from PENDING to TRIAGED."""
        result = engine.triage(
            "ALERT-001", AlertPriority.HIGH.value, "analyst-1"
        )
        assert result["current_status"] == WorkflowStatus.TRIAGED.value
        assert result["previous_status"] == WorkflowStatus.PENDING.value
        assert result["priority"] == AlertPriority.HIGH.value

    def test_triage_with_notes(
        self, engine: AlertWorkflowEngine, pending_alert: Dict[str, Any]
    ) -> None:
        """Triage with notes records them."""
        result = engine.triage(
            "ALERT-001",
            AlertPriority.CRITICAL.value,
            "analyst-1",
            "Urgent: large deforestation event detected",
        )
        assert any(
            "Urgent" in n.get("content", "") for n in result["notes"]
        )

    def test_triage_invalid_state_raises(
        self, engine: AlertWorkflowEngine, triaged_alert: Dict[str, Any]
    ) -> None:
        """Triaging an already-triaged alert raises ValueError."""
        with pytest.raises(ValueError, match="Invalid transition"):
            engine.triage("ALERT-001", AlertPriority.HIGH.value, "analyst-2")

    def test_triage_nonexistent_alert_raises(
        self, engine: AlertWorkflowEngine
    ) -> None:
        """Triaging nonexistent alert raises ValueError."""
        with pytest.raises(ValueError):
            engine.triage("NONEXISTENT", AlertPriority.HIGH.value, "analyst-1")

    def test_triage_records_transition_history(
        self, engine: AlertWorkflowEngine, pending_alert: Dict[str, Any]
    ) -> None:
        """Triage adds an entry to transitions_history."""
        result = engine.triage(
            "ALERT-001", AlertPriority.HIGH.value, "analyst-1"
        )
        assert len(result["transitions_history"]) >= 1
        last = result["transitions_history"][-1]
        assert last["action"] == WorkflowAction.TRIAGE.value
        assert last["from_status"] == WorkflowStatus.PENDING.value
        assert last["to_status"] == WorkflowStatus.TRIAGED.value


# ---------------------------------------------------------------------------
# TestAssign
# ---------------------------------------------------------------------------


class TestAssign:
    """Tests for assign operation."""

    def test_assign_to_investigator(
        self, engine: AlertWorkflowEngine, triaged_alert: Dict[str, Any]
    ) -> None:
        """Assign alert to an investigator."""
        result = engine.assign("ALERT-001", "investigator-1", "manager-1")
        assert result["assigned_to"] == "investigator-1"

    def test_reassign(
        self, engine: AlertWorkflowEngine, triaged_alert: Dict[str, Any]
    ) -> None:
        """Reassign alert to different investigator."""
        engine.assign("ALERT-001", "investigator-1", "manager-1")
        result = engine.assign("ALERT-001", "investigator-2", "manager-1")
        assert result["assigned_to"] == "investigator-2"

    def test_assign_empty_assignee_raises(
        self, engine: AlertWorkflowEngine, triaged_alert: Dict[str, Any]
    ) -> None:
        """Empty assignee raises ValueError."""
        with pytest.raises(ValueError, match="assignee"):
            engine.assign("ALERT-001", "", "manager-1")


# ---------------------------------------------------------------------------
# TestInvestigate
# ---------------------------------------------------------------------------


class TestInvestigate:
    """Tests for investigate operation: TRIAGED -> INVESTIGATING."""

    def test_investigate_triaged_to_investigating(
        self, engine: AlertWorkflowEngine, triaged_alert: Dict[str, Any]
    ) -> None:
        """Investigate moves alert to INVESTIGATING."""
        result = engine.investigate("ALERT-001", "analyst-1")
        assert result["current_status"] == WorkflowStatus.INVESTIGATING.value

    def test_investigate_from_wrong_state_raises(
        self, engine: AlertWorkflowEngine, pending_alert: Dict[str, Any]
    ) -> None:
        """Cannot investigate directly from PENDING."""
        with pytest.raises(ValueError, match="Invalid transition"):
            engine.investigate("ALERT-001", "analyst-1")


# ---------------------------------------------------------------------------
# TestResolve
# ---------------------------------------------------------------------------


class TestResolve:
    """Tests for resolve operation: INVESTIGATING -> RESOLVED."""

    def test_resolve_investigating_to_resolved(
        self, engine: AlertWorkflowEngine, investigating_alert: Dict[str, Any]
    ) -> None:
        """Resolve moves alert to RESOLVED."""
        result = engine.resolve(
            "ALERT-001", "Issue confirmed and mitigated", "analyst-1"
        )
        assert result["current_status"] == WorkflowStatus.RESOLVED.value
        assert result["resolution"] == "Issue confirmed and mitigated"
        assert result["resolved_at"] != ""

    def test_resolve_empty_resolution_raises(
        self, engine: AlertWorkflowEngine, investigating_alert: Dict[str, Any]
    ) -> None:
        """Empty resolution text raises ValueError."""
        with pytest.raises(ValueError, match="resolution"):
            engine.resolve("ALERT-001", "", "analyst-1")

    def test_resolve_with_notes(
        self, engine: AlertWorkflowEngine, investigating_alert: Dict[str, Any]
    ) -> None:
        """Resolution with notes records them."""
        result = engine.resolve(
            "ALERT-001",
            "Confirmed false alarm",
            "analyst-1",
            "Satellite cloud cover caused misdetection",
        )
        note_contents = [n.get("content", "") for n in result["notes"]]
        assert any("cloud cover" in c for c in note_contents)


# ---------------------------------------------------------------------------
# TestEscalate
# ---------------------------------------------------------------------------


class TestEscalate:
    """Tests for escalation with level increment (0 -> 1 -> 2 -> 3)."""

    def test_escalate_pending(
        self, engine: AlertWorkflowEngine, pending_alert: Dict[str, Any]
    ) -> None:
        """Escalate from PENDING."""
        result = engine.escalate(
            "ALERT-001", "SLA at risk", "system"
        )
        assert result["current_status"] == WorkflowStatus.ESCALATED.value
        assert result["escalation_level"] == 1

    def test_escalate_triaged(
        self, engine: AlertWorkflowEngine, triaged_alert: Dict[str, Any]
    ) -> None:
        """Escalate from TRIAGED."""
        result = engine.escalate(
            "ALERT-001", "Needs senior review", "analyst-1"
        )
        assert result["current_status"] == WorkflowStatus.ESCALATED.value
        assert result["escalation_level"] == 1

    def test_escalate_investigating(
        self, engine: AlertWorkflowEngine, investigating_alert: Dict[str, Any]
    ) -> None:
        """Escalate from INVESTIGATING."""
        result = engine.escalate(
            "ALERT-001", "Complex case requiring management", "analyst-1"
        )
        assert result["current_status"] == WorkflowStatus.ESCALATED.value
        assert result["escalation_level"] == 1

    def test_double_escalation(
        self, engine: AlertWorkflowEngine, pending_alert: Dict[str, Any]
    ) -> None:
        """Multiple escalations increment level."""
        engine.escalate("ALERT-001", "First escalation", "system")
        # Return to INVESTIGATING from ESCALATED
        engine.investigate("ALERT-001", "senior-analyst")
        result = engine.escalate(
            "ALERT-001", "Second escalation", "senior-analyst"
        )
        assert result["escalation_level"] == 2

    def test_escalation_caps_at_max(
        self, engine: AlertWorkflowEngine, pending_alert: Dict[str, Any]
    ) -> None:
        """Escalation level caps at max_escalation_levels."""
        for i in range(5):
            engine.escalate("ALERT-001", f"Escalation {i}", "system")
            if i < 4:
                engine.investigate("ALERT-001", "senior")
        state = engine.get_workflow_state("ALERT-001")
        assert state["escalation_level"] <= DEFAULT_MAX_ESCALATION_LEVELS

    def test_escalate_empty_reason_raises(
        self, engine: AlertWorkflowEngine, pending_alert: Dict[str, Any]
    ) -> None:
        """Empty escalation reason raises ValueError."""
        with pytest.raises(ValueError, match="reason"):
            engine.escalate("ALERT-001", "", "system")


# ---------------------------------------------------------------------------
# TestFalsePositive
# ---------------------------------------------------------------------------


class TestFalsePositive:
    """Tests for mark_false_positive operation."""

    def test_mark_false_positive(
        self, engine: AlertWorkflowEngine, investigating_alert: Dict[str, Any]
    ) -> None:
        """Mark investigating alert as false positive."""
        result = engine.mark_false_positive(
            "ALERT-001",
            "Cloud shadow misidentified as clearing",
            "analyst-1",
        )
        assert result["current_status"] == WorkflowStatus.FALSE_POSITIVE.value
        assert result["false_positive_reason"] == "Cloud shadow misidentified as clearing"

    def test_false_positive_empty_reason_raises(
        self, engine: AlertWorkflowEngine, investigating_alert: Dict[str, Any]
    ) -> None:
        """Empty false positive reason raises ValueError."""
        with pytest.raises(ValueError, match="reason"):
            engine.mark_false_positive("ALERT-001", "", "analyst-1")


# ---------------------------------------------------------------------------
# TestSLAStatus
# ---------------------------------------------------------------------------


class TestSLAStatus:
    """Tests for get_sla_status."""

    def test_sla_within_sla(
        self, engine: AlertWorkflowEngine, pending_alert: Dict[str, Any]
    ) -> None:
        """Freshly created alert is WITHIN_SLA."""
        result = engine.get_sla_status("ALERT-001")
        assert result["sla_status"] in (
            SLAStatus.WITHIN_SLA.value,
            SLAStatus.NOT_APPLICABLE.value,
        )

    def test_sla_not_applicable_for_closed(
        self, engine: AlertWorkflowEngine, resolved_alert: Dict[str, Any]
    ) -> None:
        """Closed alerts have NOT_APPLICABLE SLA."""
        engine.close("ALERT-001", "admin")
        result = engine.get_sla_status("ALERT-001")
        assert result["sla_status"] == SLAStatus.NOT_APPLICABLE.value

    def test_sla_all_alerts_report(
        self, engine: AlertWorkflowEngine
    ) -> None:
        """Get SLA status for all alerts returns a report."""
        engine.create_workflow("ALERT-SLA-1")
        engine.create_workflow("ALERT-SLA-2")
        result = engine.get_sla_status()
        assert result["total_alerts"] == 2
        assert "provenance_hash" in result

    def test_sla_report_provenance(
        self, engine: AlertWorkflowEngine
    ) -> None:
        """SLA report includes provenance hash."""
        engine.create_workflow("ALERT-SLA-P")
        result = engine.get_sla_status()
        assert len(result["provenance_hash"]) == 64


# ---------------------------------------------------------------------------
# TestValidTransitions
# ---------------------------------------------------------------------------


class TestValidTransitions:
    """Tests for _validate_transition for all valid/invalid combos."""

    @pytest.mark.parametrize(
        "from_status,action,expected",
        [
            (WorkflowStatus.PENDING.value, WorkflowAction.TRIAGE.value, True),
            (WorkflowStatus.PENDING.value, WorkflowAction.ESCALATE.value, True),
            (WorkflowStatus.PENDING.value, WorkflowAction.INVESTIGATE.value, False),
            (WorkflowStatus.TRIAGED.value, WorkflowAction.INVESTIGATE.value, True),
            (WorkflowStatus.TRIAGED.value, WorkflowAction.ESCALATE.value, True),
            (WorkflowStatus.TRIAGED.value, WorkflowAction.RESOLVE.value, False),
            (WorkflowStatus.INVESTIGATING.value, WorkflowAction.RESOLVE.value, True),
            (WorkflowStatus.INVESTIGATING.value, WorkflowAction.ESCALATE.value, True),
            (WorkflowStatus.INVESTIGATING.value, WorkflowAction.MARK_FALSE_POSITIVE.value, True),
            (WorkflowStatus.INVESTIGATING.value, WorkflowAction.TRIAGE.value, False),
            (WorkflowStatus.ESCALATED.value, WorkflowAction.INVESTIGATE.value, True),
            (WorkflowStatus.ESCALATED.value, WorkflowAction.RESOLVE.value, True),
            (WorkflowStatus.ESCALATED.value, WorkflowAction.TRIAGE.value, False),
            (WorkflowStatus.RESOLVED.value, WorkflowAction.CLOSE.value, True),
            (WorkflowStatus.RESOLVED.value, WorkflowAction.REOPEN.value, True),
            (WorkflowStatus.RESOLVED.value, WorkflowAction.ESCALATE.value, False),
            (WorkflowStatus.FALSE_POSITIVE.value, WorkflowAction.REOPEN.value, True),
            (WorkflowStatus.FALSE_POSITIVE.value, WorkflowAction.CLOSE.value, True),
            (WorkflowStatus.CLOSED.value, WorkflowAction.REOPEN.value, False),
        ],
    )
    def test_transition_validity(
        self,
        engine: AlertWorkflowEngine,
        from_status: str,
        action: str,
        expected: bool,
    ) -> None:
        """Parametrized test for all valid/invalid state transitions."""
        assert engine._validate_transition(from_status, action) is expected


# ---------------------------------------------------------------------------
# TestSLACalculation
# ---------------------------------------------------------------------------


class TestSLACalculation:
    """Tests for _calculate_sla_deadline."""

    def test_triage_sla_4h(self, engine: AlertWorkflowEngine) -> None:
        """Triage SLA for MEDIUM priority is 4 hours."""
        deadline = engine._calculate_sla_deadline(
            WorkflowStatus.PENDING.value,
            AlertPriority.MEDIUM.value,
        )
        now = datetime.now(timezone.utc).replace(microsecond=0)
        diff = (deadline - now).total_seconds() / 3600
        assert 3.5 <= diff <= 4.5

    def test_investigation_sla_48h(self, engine: AlertWorkflowEngine) -> None:
        """Investigation SLA for MEDIUM priority is 48 hours."""
        deadline = engine._calculate_sla_deadline(
            WorkflowStatus.TRIAGED.value,
            AlertPriority.MEDIUM.value,
        )
        now = datetime.now(timezone.utc).replace(microsecond=0)
        diff = (deadline - now).total_seconds() / 3600
        assert 47.0 <= diff <= 49.0

    def test_resolution_sla_168h(self, engine: AlertWorkflowEngine) -> None:
        """Resolution SLA for MEDIUM priority is 168 hours (7 days)."""
        deadline = engine._calculate_sla_deadline(
            WorkflowStatus.INVESTIGATING.value,
            AlertPriority.MEDIUM.value,
        )
        now = datetime.now(timezone.utc).replace(microsecond=0)
        diff = (deadline - now).total_seconds() / 3600
        assert 167.0 <= diff <= 169.0

    def test_critical_priority_halves_sla(self, engine: AlertWorkflowEngine) -> None:
        """CRITICAL priority applies 0.5 multiplier."""
        deadline = engine._calculate_sla_deadline(
            WorkflowStatus.PENDING.value,
            AlertPriority.CRITICAL.value,
        )
        now = datetime.now(timezone.utc).replace(microsecond=0)
        diff = (deadline - now).total_seconds() / 3600
        # 4h * 0.5 = 2h
        assert 1.5 <= diff <= 2.5

    def test_escalation_reduces_sla(self, engine: AlertWorkflowEngine) -> None:
        """Each escalation level halves SLA window."""
        d0 = engine._calculate_sla_deadline(
            WorkflowStatus.INVESTIGATING.value,
            AlertPriority.MEDIUM.value,
            escalation_level=0,
        )
        d1 = engine._calculate_sla_deadline(
            WorkflowStatus.INVESTIGATING.value,
            AlertPriority.MEDIUM.value,
            escalation_level=1,
        )
        now = datetime.now(timezone.utc).replace(microsecond=0)
        hours_0 = (d0 - now).total_seconds() / 3600
        hours_1 = (d1 - now).total_seconds() / 3600
        assert hours_1 < hours_0


# ---------------------------------------------------------------------------
# TestTransitionRecording
# ---------------------------------------------------------------------------


class TestTransitionRecording:
    """Tests for _record_transition creating proper audit trail."""

    def test_transition_records_actor(
        self, engine: AlertWorkflowEngine, pending_alert: Dict[str, Any]
    ) -> None:
        """Transition records the actor."""
        result = engine.triage(
            "ALERT-001", AlertPriority.HIGH.value, "analyst-jane"
        )
        last = result["transitions_history"][-1]
        assert last["actor"] == "analyst-jane"

    def test_transition_records_timestamp(
        self, engine: AlertWorkflowEngine, pending_alert: Dict[str, Any]
    ) -> None:
        """Transition records a timestamp."""
        result = engine.triage(
            "ALERT-001", AlertPriority.HIGH.value, "analyst-1"
        )
        last = result["transitions_history"][-1]
        assert last["timestamp"] != ""


# ---------------------------------------------------------------------------
# TestWorkflowLifecycle
# ---------------------------------------------------------------------------


class TestWorkflowLifecycle:
    """Full lifecycle: PENDING -> TRIAGED -> INVESTIGATING -> RESOLVED -> CLOSED."""

    def test_full_lifecycle(self, engine: AlertWorkflowEngine) -> None:
        """Complete workflow lifecycle from creation to closure."""
        # Create
        state = engine.create_workflow("ALERT-LIFE")
        assert state["current_status"] == WorkflowStatus.PENDING.value

        # Triage
        state = engine.triage("ALERT-LIFE", AlertPriority.HIGH.value, "analyst-1")
        assert state["current_status"] == WorkflowStatus.TRIAGED.value

        # Assign
        state = engine.assign("ALERT-LIFE", "investigator-1", "manager-1")
        assert state["assigned_to"] == "investigator-1"

        # Investigate
        state = engine.investigate("ALERT-LIFE", "investigator-1")
        assert state["current_status"] == WorkflowStatus.INVESTIGATING.value

        # Resolve
        state = engine.resolve(
            "ALERT-LIFE", "Issue fully resolved", "investigator-1"
        )
        assert state["current_status"] == WorkflowStatus.RESOLVED.value
        assert state["resolved_at"] != ""

        # Close
        state = engine.close("ALERT-LIFE", "manager-1")
        assert state["current_status"] == WorkflowStatus.CLOSED.value
        assert state["closed_at"] != ""

        # Verify complete audit trail
        assert len(state["transitions_history"]) >= 4

    def test_full_lifecycle_provenance(self, engine: AlertWorkflowEngine) -> None:
        """Every step in lifecycle produces provenance hash."""
        engine.create_workflow("ALERT-PROV-LIFE")
        state = engine.triage(
            "ALERT-PROV-LIFE", AlertPriority.MEDIUM.value, "a1"
        )
        assert len(state["provenance_hash"]) == 64


# ---------------------------------------------------------------------------
# TestReopenFlow
# ---------------------------------------------------------------------------


class TestReopenFlow:
    """Tests for reopening alerts."""

    def test_reopen_resolved(
        self, engine: AlertWorkflowEngine, resolved_alert: Dict[str, Any]
    ) -> None:
        """Reopen RESOLVED -> INVESTIGATING."""
        result = engine.reopen(
            "ALERT-001", "New evidence found", "analyst-1"
        )
        assert result["current_status"] == WorkflowStatus.INVESTIGATING.value

    def test_reopen_false_positive(
        self, engine: AlertWorkflowEngine, investigating_alert: Dict[str, Any]
    ) -> None:
        """Reopen FALSE_POSITIVE -> INVESTIGATING."""
        engine.mark_false_positive(
            "ALERT-001", "Initially thought false positive", "analyst-1"
        )
        result = engine.reopen(
            "ALERT-001", "Actually confirmed deforestation", "analyst-2"
        )
        assert result["current_status"] == WorkflowStatus.INVESTIGATING.value

    def test_reopen_empty_reason_raises(
        self, engine: AlertWorkflowEngine, resolved_alert: Dict[str, Any]
    ) -> None:
        """Empty reopen reason raises ValueError."""
        with pytest.raises(ValueError, match="reason"):
            engine.reopen("ALERT-001", "", "analyst-1")


# ---------------------------------------------------------------------------
# TestProvenance
# ---------------------------------------------------------------------------


class TestProvenance:
    """Tests for provenance hash on all transitions."""

    def test_create_workflow_provenance(
        self, pending_alert: Dict[str, Any]
    ) -> None:
        """Workflow creation has provenance hash."""
        assert len(pending_alert["provenance_hash"]) == 64

    def test_triage_provenance(
        self, engine: AlertWorkflowEngine, pending_alert: Dict[str, Any]
    ) -> None:
        """Triage result has provenance hash."""
        result = engine.triage(
            "ALERT-001", AlertPriority.HIGH.value, "analyst-1"
        )
        assert len(result["provenance_hash"]) == 64

    def test_resolve_provenance(
        self, engine: AlertWorkflowEngine, investigating_alert: Dict[str, Any]
    ) -> None:
        """Resolve result has provenance hash."""
        result = engine.resolve("ALERT-001", "Resolved", "analyst-1")
        assert len(result["provenance_hash"]) == 64


# ---------------------------------------------------------------------------
# TestAddNote
# ---------------------------------------------------------------------------


class TestAddNote:
    """Tests for add_note without state change."""

    def test_add_note_to_alert(
        self, engine: AlertWorkflowEngine, pending_alert: Dict[str, Any]
    ) -> None:
        """Add a note without changing state."""
        result = engine.add_note(
            "ALERT-001", "Follow-up observation scheduled", "analyst-1"
        )
        assert result["current_status"] == WorkflowStatus.PENDING.value
        assert any(
            "Follow-up" in n.get("content", "") for n in result["notes"]
        )

    def test_add_note_empty_content_raises(
        self, engine: AlertWorkflowEngine, pending_alert: Dict[str, Any]
    ) -> None:
        """Empty note content raises ValueError."""
        with pytest.raises(ValueError, match="content"):
            engine.add_note("ALERT-001", "", "analyst-1")

    def test_add_note_nonexistent_alert_raises(
        self, engine: AlertWorkflowEngine
    ) -> None:
        """Adding note to nonexistent alert raises ValueError."""
        with pytest.raises(ValueError):
            engine.add_note("NONEXISTENT", "Some note", "analyst-1")


# ---------------------------------------------------------------------------
# TestCreateWorkflow
# ---------------------------------------------------------------------------


class TestCreateWorkflow:
    """Tests for create_workflow."""

    def test_create_workflow_default_priority(
        self, engine: AlertWorkflowEngine
    ) -> None:
        """Default priority is MEDIUM."""
        result = engine.create_workflow("ALERT-NEW")
        assert result["priority"] == AlertPriority.MEDIUM.value

    def test_create_workflow_duplicate_raises(
        self, engine: AlertWorkflowEngine
    ) -> None:
        """Creating duplicate workflow raises ValueError."""
        engine.create_workflow("ALERT-DUP")
        with pytest.raises(ValueError, match="already exists"):
            engine.create_workflow("ALERT-DUP")

    def test_create_workflow_empty_id_raises(
        self, engine: AlertWorkflowEngine
    ) -> None:
        """Empty alert_id raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            engine.create_workflow("")

    def test_create_workflow_initial_state(
        self, engine: AlertWorkflowEngine
    ) -> None:
        """Initial workflow state is PENDING."""
        result = engine.create_workflow("ALERT-INIT")
        assert result["current_status"] == WorkflowStatus.PENDING.value
        assert result["escalation_level"] == 0
        assert result["created_at"] != ""


# ---------------------------------------------------------------------------
# TestDataClasses
# ---------------------------------------------------------------------------


class TestDataClasses:
    """Tests for data class serialization."""

    def test_workflow_transition_to_dict(self) -> None:
        """WorkflowTransition serialization works."""
        wt = WorkflowTransition(
            transition_id="tr-1",
            from_status="PENDING",
            to_status="TRIAGED",
            action="TRIAGE",
            actor="analyst-1",
        )
        d = wt.to_dict()
        assert d["from_status"] == "PENDING"
        assert d["actor"] == "analyst-1"

    def test_workflow_note_to_dict(self) -> None:
        """WorkflowNote serialization works."""
        wn = WorkflowNote(
            note_id="nt-1",
            alert_id="ALERT-1",
            author="analyst-1",
            content="Investigation notes",
        )
        d = wn.to_dict()
        assert d["content"] == "Investigation notes"

    def test_sla_config_to_dict(self) -> None:
        """SLAConfig serialization works."""
        sc = SLAConfig(triage_hours=2, investigation_hours=24, resolution_hours=72)
        d = sc.to_dict()
        assert d["triage_hours"] == 2

    def test_sla_report_to_dict(self) -> None:
        """SLAReport serialization works."""
        sr = SLAReport(
            report_id="sla-1",
            total_alerts=10,
            within_sla_count=8,
            breached_count=2,
        )
        d = sr.to_dict()
        assert d["total_alerts"] == 10

    def test_workflow_state_to_dict(self) -> None:
        """WorkflowState serialization includes all fields."""
        ws = WorkflowState(
            state_id="ws-1",
            alert_id="ALERT-1",
            current_status=WorkflowStatus.INVESTIGATING.value,
            escalation_level=1,
        )
        d = ws.to_dict()
        assert d["escalation_level"] == 1

    def test_constants(self) -> None:
        """Verify key constants."""
        assert DEFAULT_SLA_TRIAGE_HOURS == 4
        assert DEFAULT_SLA_INVESTIGATION_HOURS == 48
        assert DEFAULT_SLA_RESOLUTION_HOURS == 168
        assert DEFAULT_MAX_ESCALATION_LEVELS == 3
        assert WorkflowStatus.CLOSED.value in TERMINAL_STATES


# ---------------------------------------------------------------------------
# TestGetStatistics
# ---------------------------------------------------------------------------


class TestGetStatistics:
    """Tests for get_statistics."""

    def test_statistics_empty(self, engine: AlertWorkflowEngine) -> None:
        """Statistics with no workflows."""
        stats = engine.get_statistics()
        assert stats["total_workflows"] == 0

    def test_statistics_with_workflows(
        self, engine: AlertWorkflowEngine, pending_alert: Dict[str, Any]
    ) -> None:
        """Statistics reflect current state."""
        stats = engine.get_statistics()
        assert stats["total_workflows"] >= 1
        assert "status_counts" in stats
