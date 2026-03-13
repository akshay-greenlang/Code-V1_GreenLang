# -*- coding: utf-8 -*-
"""
Unit tests for GrievanceMechanism Engine - AGENT-EUDR-031

Tests grievance submission, triage, investigation, resolution,
satisfaction assessment, and appeal workflows.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from greenlang.agents.eudr.stakeholder_engagement.config import (
    StakeholderEngagementConfig,
)
from greenlang.agents.eudr.stakeholder_engagement.grievance_mechanism import (
    GrievanceMechanism,
)
from greenlang.agents.eudr.stakeholder_engagement.models import (
    GrievanceRecord,
    GrievanceSeverity,
    GrievanceStatus,
)
from greenlang.agents.eudr.stakeholder_engagement.provenance import (
    ProvenanceTracker,
)


@pytest.fixture
def config():
    return StakeholderEngagementConfig()


@pytest.fixture
def mechanism(config):
    return GrievanceMechanism(config=config)


# ---------------------------------------------------------------------------
# Test: SubmitGrievance
# ---------------------------------------------------------------------------

class TestSubmitGrievance:
    """Test grievance submission across all channels and severities."""

    @pytest.mark.asyncio
    async def test_submit_grievance_success(self, mechanism):
        """Test successful grievance submission."""
        grv = await mechanism.submit_grievance(
            stakeholder_id="STK-IND-001",
            operator_id="OP-001",
            title="Land access violation",
            description="Unauthorized access to community land.",
            severity=GrievanceSeverity.CRITICAL,
            channel="field_visit",
            category="land_rights_violation",
        )
        assert grv.grievance_id.startswith("GRV-")
        assert grv.status == GrievanceStatus.SUBMITTED
        assert grv.severity == GrievanceSeverity.CRITICAL

    @pytest.mark.asyncio
    async def test_submit_grievance_sets_sla_deadline(self, mechanism):
        """Test submission sets SLA deadline based on severity."""
        grv = await mechanism.submit_grievance(
            stakeholder_id="STK-001",
            operator_id="OP-001",
            title="Critical Issue",
            description="Critical test.",
            severity=GrievanceSeverity.CRITICAL,
            channel="email",
        )
        assert grv.sla_deadline is not None
        assert grv.sla_deadline > grv.submitted_at

    @pytest.mark.asyncio
    async def test_submit_grievance_all_severities(self, mechanism):
        """Test submission with all severity levels."""
        for severity in GrievanceSeverity:
            grv = await mechanism.submit_grievance(
                stakeholder_id="STK-001",
                operator_id="OP-001",
                title=f"Test {severity.value}",
                description=f"Severity: {severity.value}",
                severity=severity,
                channel="email",
            )
            assert grv.severity == severity

    @pytest.mark.asyncio
    async def test_submit_grievance_all_channels(self, mechanism):
        """Test submission via all supported channels."""
        for channel in ["email", "phone", "field_visit", "letter", "sms", "web_form"]:
            grv = await mechanism.submit_grievance(
                stakeholder_id="STK-001",
                operator_id="OP-001",
                title=f"Via {channel}",
                description=f"Submitted via {channel}.",
                severity=GrievanceSeverity.STANDARD,
                channel=channel,
            )
            assert grv.channel == channel

    @pytest.mark.asyncio
    async def test_submit_grievance_missing_title_raises(self, mechanism):
        """Test submission with empty title raises error."""
        with pytest.raises(ValueError, match="title is required"):
            await mechanism.submit_grievance(
                stakeholder_id="STK-001",
                operator_id="OP-001",
                title="",
                description="Description only.",
                severity=GrievanceSeverity.STANDARD,
                channel="email",
            )

    @pytest.mark.asyncio
    async def test_submit_grievance_missing_stakeholder_raises(self, mechanism):
        """Test submission with empty stakeholder_id raises error."""
        with pytest.raises(ValueError, match="stakeholder_id is required"):
            await mechanism.submit_grievance(
                stakeholder_id="",
                operator_id="OP-001",
                title="Test",
                description="Test",
                severity=GrievanceSeverity.STANDARD,
                channel="email",
            )

    @pytest.mark.asyncio
    async def test_submit_grievance_unique_ids(self, mechanism):
        """Test each submission generates unique IDs."""
        ids = set()
        for i in range(5):
            grv = await mechanism.submit_grievance(
                stakeholder_id="STK-001",
                operator_id="OP-001",
                title=f"Grievance {i}",
                description=f"Test {i}",
                severity=GrievanceSeverity.MINOR,
                channel="email",
            )
            ids.add(grv.grievance_id)
        assert len(ids) == 5

    @pytest.mark.asyncio
    async def test_submit_grievance_timestamps(self, mechanism):
        """Test submission sets proper timestamps."""
        grv = await mechanism.submit_grievance(
            stakeholder_id="STK-001",
            operator_id="OP-001",
            title="Timestamp Test",
            description="Test",
            severity=GrievanceSeverity.STANDARD,
            channel="email",
        )
        assert isinstance(grv.submitted_at, datetime)
        assert grv.submitted_at <= datetime.now(tz=timezone.utc)


# ---------------------------------------------------------------------------
# Test: TriageGrievance
# ---------------------------------------------------------------------------

class TestTriageGrievance:
    """Test grievance triage operations."""

    @pytest.mark.asyncio
    async def test_triage_grievance_success(self, mechanism):
        """Test successful grievance triage."""
        grv = await mechanism.submit_grievance(
            "STK-001", "OP-001", "Test", "Test desc",
            GrievanceSeverity.STANDARD, "email",
        )
        triaged = await mechanism.triage_grievance(
            grievance_id=grv.grievance_id,
            assigned_to="investigator-001",
            priority_notes="Requires immediate field visit.",
        )
        assert triaged.status == GrievanceStatus.TRIAGED

    @pytest.mark.asyncio
    async def test_triage_assigns_investigator(self, mechanism):
        """Test triage assigns investigator."""
        grv = await mechanism.submit_grievance(
            "STK-001", "OP-001", "Test", "Test desc",
            GrievanceSeverity.HIGH, "phone",
        )
        triaged = await mechanism.triage_grievance(grv.grievance_id, "investigator-002")
        assert triaged.assigned_to == "investigator-002"

    @pytest.mark.asyncio
    async def test_triage_nonexistent_grievance_raises(self, mechanism):
        """Test triage of nonexistent grievance raises error."""
        with pytest.raises(ValueError, match="grievance not found"):
            await mechanism.triage_grievance("GRV-NONEXISTENT", "inv-001")

    @pytest.mark.asyncio
    async def test_triage_escalates_severity(self, mechanism):
        """Test triage can escalate severity."""
        grv = await mechanism.submit_grievance(
            "STK-001", "OP-001", "Test", "Desc",
            GrievanceSeverity.STANDARD, "email",
        )
        triaged = await mechanism.triage_grievance(
            grv.grievance_id, "inv-001",
            escalate_to=GrievanceSeverity.HIGH,
        )
        assert triaged.severity == GrievanceSeverity.HIGH

    @pytest.mark.asyncio
    async def test_triage_already_triaged(self, mechanism):
        """Test re-triage of already triaged grievance."""
        grv = await mechanism.submit_grievance(
            "STK-001", "OP-001", "Test", "Desc",
            GrievanceSeverity.STANDARD, "email",
        )
        await mechanism.triage_grievance(grv.grievance_id, "inv-001")
        # Re-triage should succeed (reassignment)
        retriaged = await mechanism.triage_grievance(grv.grievance_id, "inv-002")
        assert retriaged.assigned_to == "inv-002"

    @pytest.mark.asyncio
    async def test_triage_updates_provenance(self, mechanism):
        """Test triage updates provenance chain."""
        grv = await mechanism.submit_grievance(
            "STK-001", "OP-001", "Test", "Desc",
            GrievanceSeverity.STANDARD, "email",
        )
        chain_before = len(mechanism._provenance.get_chain())
        await mechanism.triage_grievance(grv.grievance_id, "inv-001")
        chain_after = len(mechanism._provenance.get_chain())
        assert chain_after > chain_before

    @pytest.mark.asyncio
    async def test_triage_empty_grievance_id_raises(self, mechanism):
        """Test triage with empty grievance_id raises error."""
        with pytest.raises(ValueError, match="grievance_id is required"):
            await mechanism.triage_grievance("", "inv-001")

    @pytest.mark.asyncio
    async def test_triage_empty_assignee_raises(self, mechanism):
        """Test triage with empty assigned_to raises error."""
        grv = await mechanism.submit_grievance(
            "STK-001", "OP-001", "Test", "Desc",
            GrievanceSeverity.STANDARD, "email",
        )
        with pytest.raises(ValueError, match="assigned_to is required"):
            await mechanism.triage_grievance(grv.grievance_id, "")


# ---------------------------------------------------------------------------
# Test: Investigate
# ---------------------------------------------------------------------------

class TestInvestigate:
    """Test grievance investigation operations."""

    @pytest.mark.asyncio
    async def test_investigate_adds_notes(self, mechanism, sample_investigation_notes):
        """Test investigation adds notes."""
        grv = await mechanism.submit_grievance(
            "STK-001", "OP-001", "Test", "Desc",
            GrievanceSeverity.HIGH, "field_visit",
        )
        await mechanism.triage_grievance(grv.grievance_id, "inv-001")
        updated = await mechanism.investigate(
            grievance_id=grv.grievance_id,
            notes=sample_investigation_notes,
        )
        assert updated.status == GrievanceStatus.INVESTIGATING
        assert len(updated.investigation_notes) >= 2

    @pytest.mark.asyncio
    async def test_investigate_transitions_status(self, mechanism):
        """Test investigation transitions status to INVESTIGATING."""
        grv = await mechanism.submit_grievance(
            "STK-001", "OP-001", "Test", "Desc",
            GrievanceSeverity.STANDARD, "email",
        )
        await mechanism.triage_grievance(grv.grievance_id, "inv-001")
        updated = await mechanism.investigate(
            grv.grievance_id,
            [{"finding": "Initial assessment complete"}],
        )
        assert updated.status == GrievanceStatus.INVESTIGATING

    @pytest.mark.asyncio
    async def test_investigate_nonexistent_raises(self, mechanism):
        """Test investigation of nonexistent grievance raises error."""
        with pytest.raises(ValueError, match="grievance not found"):
            await mechanism.investigate("GRV-NONEXISTENT", [{"finding": "test"}])

    @pytest.mark.asyncio
    async def test_investigate_appends_multiple_notes(self, mechanism):
        """Test multiple investigation rounds append notes."""
        grv = await mechanism.submit_grievance(
            "STK-001", "OP-001", "Test", "Desc",
            GrievanceSeverity.STANDARD, "email",
        )
        await mechanism.triage_grievance(grv.grievance_id, "inv-001")
        await mechanism.investigate(grv.grievance_id, [{"finding": "Round 1"}])
        updated = await mechanism.investigate(grv.grievance_id, [{"finding": "Round 2"}])
        assert len(updated.investigation_notes) >= 2

    @pytest.mark.asyncio
    async def test_investigate_with_evidence(self, mechanism):
        """Test investigation with evidence references."""
        grv = await mechanism.submit_grievance(
            "STK-001", "OP-001", "Test", "Desc",
            GrievanceSeverity.HIGH, "field_visit",
        )
        await mechanism.triage_grievance(grv.grievance_id, "inv-001")
        updated = await mechanism.investigate(
            grv.grievance_id,
            [{"finding": "Evidence collected", "evidence_ref": "PHOTO-001"}],
        )
        assert len(updated.investigation_notes) >= 1

    @pytest.mark.asyncio
    async def test_investigate_empty_notes_raises(self, mechanism):
        """Test investigation with empty notes raises error."""
        grv = await mechanism.submit_grievance(
            "STK-001", "OP-001", "Test", "Desc",
            GrievanceSeverity.STANDARD, "email",
        )
        await mechanism.triage_grievance(grv.grievance_id, "inv-001")
        with pytest.raises(ValueError, match="notes are required"):
            await mechanism.investigate(grv.grievance_id, [])

    @pytest.mark.asyncio
    async def test_investigate_preserves_existing_notes(self, mechanism):
        """Test investigation preserves previously added notes."""
        grv = await mechanism.submit_grievance(
            "STK-001", "OP-001", "Test", "Desc",
            GrievanceSeverity.STANDARD, "email",
        )
        await mechanism.triage_grievance(grv.grievance_id, "inv-001")
        await mechanism.investigate(grv.grievance_id, [{"finding": "First note"}])
        updated = await mechanism.investigate(grv.grievance_id, [{"finding": "Second note"}])
        notes = [n.get("finding", "") for n in updated.investigation_notes]
        assert "First note" in notes
        assert "Second note" in notes

    @pytest.mark.asyncio
    async def test_investigate_updates_provenance(self, mechanism):
        """Test investigation updates provenance chain."""
        grv = await mechanism.submit_grievance(
            "STK-001", "OP-001", "Test", "Desc",
            GrievanceSeverity.STANDARD, "email",
        )
        await mechanism.triage_grievance(grv.grievance_id, "inv-001")
        chain_before = len(mechanism._provenance.get_chain())
        await mechanism.investigate(grv.grievance_id, [{"finding": "Test"}])
        assert len(mechanism._provenance.get_chain()) > chain_before


# ---------------------------------------------------------------------------
# Test: Resolve
# ---------------------------------------------------------------------------

class TestResolve:
    """Test grievance resolution operations."""

    @pytest.mark.asyncio
    async def test_resolve_success(self, mechanism, sample_resolution_actions):
        """Test successful grievance resolution."""
        grv = await mechanism.submit_grievance(
            "STK-001", "OP-001", "Test", "Desc",
            GrievanceSeverity.STANDARD, "email",
        )
        await mechanism.triage_grievance(grv.grievance_id, "inv-001")
        await mechanism.investigate(grv.grievance_id, [{"finding": "Complete"}])
        resolved = await mechanism.resolve(
            grievance_id=grv.grievance_id,
            resolution_actions=sample_resolution_actions,
            resolution_summary="Issue resolved through corrective actions.",
        )
        assert resolved.status == GrievanceStatus.RESOLVED

    @pytest.mark.asyncio
    async def test_resolve_sets_resolution_actions(self, mechanism, sample_resolution_actions):
        """Test resolution sets action items."""
        grv = await mechanism.submit_grievance(
            "STK-001", "OP-001", "Test", "Desc",
            GrievanceSeverity.STANDARD, "email",
        )
        await mechanism.triage_grievance(grv.grievance_id, "inv-001")
        await mechanism.investigate(grv.grievance_id, [{"finding": "Done"}])
        resolved = await mechanism.resolve(
            grv.grievance_id, sample_resolution_actions, "Resolved.",
        )
        assert len(resolved.resolution_actions) >= 2

    @pytest.mark.asyncio
    async def test_resolve_nonexistent_raises(self, mechanism):
        """Test resolving nonexistent grievance raises error."""
        with pytest.raises(ValueError, match="grievance not found"):
            await mechanism.resolve("GRV-NONEXISTENT", [], "Summary")

    @pytest.mark.asyncio
    async def test_resolve_without_actions_raises(self, mechanism):
        """Test resolution without actions raises error."""
        grv = await mechanism.submit_grievance(
            "STK-001", "OP-001", "Test", "Desc",
            GrievanceSeverity.STANDARD, "email",
        )
        await mechanism.triage_grievance(grv.grievance_id, "inv-001")
        await mechanism.investigate(grv.grievance_id, [{"finding": "Done"}])
        with pytest.raises(ValueError, match="resolution_actions are required"):
            await mechanism.resolve(grv.grievance_id, [], "Summary")

    @pytest.mark.asyncio
    async def test_resolve_sets_resolved_at(self, mechanism):
        """Test resolution sets resolved_at timestamp."""
        grv = await mechanism.submit_grievance(
            "STK-001", "OP-001", "Test", "Desc",
            GrievanceSeverity.MINOR, "email",
        )
        await mechanism.triage_grievance(grv.grievance_id, "inv-001")
        await mechanism.investigate(grv.grievance_id, [{"finding": "Done"}])
        resolved = await mechanism.resolve(
            grv.grievance_id,
            [{"action": "corrective", "description": "Fixed"}],
            "Resolved.",
        )
        assert resolved.resolved_at is not None

    @pytest.mark.asyncio
    async def test_resolve_preserves_investigation(self, mechanism):
        """Test resolution preserves investigation notes."""
        grv = await mechanism.submit_grievance(
            "STK-001", "OP-001", "Test", "Desc",
            GrievanceSeverity.STANDARD, "email",
        )
        await mechanism.triage_grievance(grv.grievance_id, "inv-001")
        await mechanism.investigate(grv.grievance_id, [{"finding": "Important note"}])
        resolved = await mechanism.resolve(
            grv.grievance_id,
            [{"action": "corrective"}],
            "Done.",
        )
        assert len(resolved.investigation_notes) >= 1

    @pytest.mark.asyncio
    async def test_resolve_critical_grievance(self, mechanism):
        """Test resolving a critical severity grievance."""
        grv = await mechanism.submit_grievance(
            "STK-001", "OP-001", "Critical", "Critical issue.",
            GrievanceSeverity.CRITICAL, "field_visit",
        )
        await mechanism.triage_grievance(grv.grievance_id, "inv-001")
        await mechanism.investigate(grv.grievance_id, [{"finding": "Urgent action taken"}])
        resolved = await mechanism.resolve(
            grv.grievance_id,
            [{"action": "immediate_halt"}, {"action": "corrective_plan"}],
            "Emergency resolution.",
        )
        assert resolved.status == GrievanceStatus.RESOLVED
        assert resolved.severity == GrievanceSeverity.CRITICAL

    @pytest.mark.asyncio
    async def test_resolve_empty_summary_raises(self, mechanism):
        """Test resolution with empty summary raises error."""
        grv = await mechanism.submit_grievance(
            "STK-001", "OP-001", "Test", "Desc",
            GrievanceSeverity.STANDARD, "email",
        )
        await mechanism.triage_grievance(grv.grievance_id, "inv-001")
        await mechanism.investigate(grv.grievance_id, [{"finding": "Done"}])
        with pytest.raises(ValueError, match="resolution_summary is required"):
            await mechanism.resolve(grv.grievance_id, [{"action": "fix"}], "")


# ---------------------------------------------------------------------------
# Test: SatisfactionAssessment
# ---------------------------------------------------------------------------

class TestSatisfactionAssessment:
    """Test post-resolution satisfaction assessment."""

    @pytest.mark.asyncio
    async def test_assess_satisfaction_success(self, mechanism):
        """Test successful satisfaction assessment."""
        grv = await mechanism.submit_grievance(
            "STK-001", "OP-001", "Test", "Desc",
            GrievanceSeverity.STANDARD, "email",
        )
        await mechanism.triage_grievance(grv.grievance_id, "inv-001")
        await mechanism.investigate(grv.grievance_id, [{"finding": "Done"}])
        await mechanism.resolve(grv.grievance_id, [{"action": "fix"}], "Resolved.")
        assessment = await mechanism.assess_satisfaction(
            grievance_id=grv.grievance_id,
            satisfaction_score=Decimal("85"),
            feedback="Satisfied with the resolution process.",
        )
        assert assessment["satisfaction_score"] == Decimal("85")

    @pytest.mark.asyncio
    async def test_assess_satisfaction_low_score(self, mechanism):
        """Test satisfaction assessment with low score."""
        grv = await mechanism.submit_grievance(
            "STK-001", "OP-001", "Test", "Desc",
            GrievanceSeverity.STANDARD, "email",
        )
        await mechanism.triage_grievance(grv.grievance_id, "inv-001")
        await mechanism.investigate(grv.grievance_id, [{"finding": "Done"}])
        await mechanism.resolve(grv.grievance_id, [{"action": "partial_fix"}], "Partial.")
        assessment = await mechanism.assess_satisfaction(
            grv.grievance_id, Decimal("25"), "Not fully satisfied.",
        )
        assert assessment["satisfaction_score"] == Decimal("25")

    @pytest.mark.asyncio
    async def test_assess_satisfaction_nonexistent_raises(self, mechanism):
        """Test satisfaction assessment for nonexistent grievance."""
        with pytest.raises(ValueError, match="grievance not found"):
            await mechanism.assess_satisfaction("GRV-NONEXISTENT", Decimal("50"), "Test")

    @pytest.mark.asyncio
    async def test_assess_satisfaction_score_bounds(self, mechanism):
        """Test satisfaction score must be between 0 and 100."""
        grv = await mechanism.submit_grievance(
            "STK-001", "OP-001", "Test", "Desc",
            GrievanceSeverity.MINOR, "email",
        )
        await mechanism.triage_grievance(grv.grievance_id, "inv-001")
        await mechanism.investigate(grv.grievance_id, [{"finding": "Done"}])
        await mechanism.resolve(grv.grievance_id, [{"action": "fix"}], "Done.")
        with pytest.raises(ValueError, match="score must be between"):
            await mechanism.assess_satisfaction(grv.grievance_id, Decimal("150"), "Bad")

    @pytest.mark.asyncio
    async def test_assess_satisfaction_includes_feedback(self, mechanism):
        """Test assessment includes feedback text."""
        grv = await mechanism.submit_grievance(
            "STK-001", "OP-001", "Test", "Desc",
            GrievanceSeverity.MINOR, "email",
        )
        await mechanism.triage_grievance(grv.grievance_id, "inv-001")
        await mechanism.investigate(grv.grievance_id, [{"finding": "Done"}])
        await mechanism.resolve(grv.grievance_id, [{"action": "fix"}], "Done.")
        assessment = await mechanism.assess_satisfaction(
            grv.grievance_id, Decimal("70"), "Good process overall.",
        )
        assert assessment["feedback"] == "Good process overall."


# ---------------------------------------------------------------------------
# Test: Appeal
# ---------------------------------------------------------------------------

class TestAppeal:
    """Test grievance appeal operations."""

    @pytest.mark.asyncio
    async def test_appeal_success(self, mechanism):
        """Test successful grievance appeal."""
        grv = await mechanism.submit_grievance(
            "STK-001", "OP-001", "Test", "Desc",
            GrievanceSeverity.STANDARD, "email",
        )
        await mechanism.triage_grievance(grv.grievance_id, "inv-001")
        await mechanism.investigate(grv.grievance_id, [{"finding": "Done"}])
        await mechanism.resolve(grv.grievance_id, [{"action": "fix"}], "Resolved.")
        appealed = await mechanism.appeal(
            grievance_id=grv.grievance_id,
            reason="Resolution did not address root cause.",
        )
        assert appealed.status == GrievanceStatus.APPEALED

    @pytest.mark.asyncio
    async def test_appeal_nonexistent_raises(self, mechanism):
        """Test appeal of nonexistent grievance raises error."""
        with pytest.raises(ValueError, match="grievance not found"):
            await mechanism.appeal("GRV-NONEXISTENT", "Reason")

    @pytest.mark.asyncio
    async def test_appeal_unresolved_raises(self, mechanism):
        """Test appeal of unresolved grievance raises error."""
        grv = await mechanism.submit_grievance(
            "STK-001", "OP-001", "Test", "Desc",
            GrievanceSeverity.STANDARD, "email",
        )
        with pytest.raises(ValueError, match="must be resolved before appeal"):
            await mechanism.appeal(grv.grievance_id, "Reason")

    @pytest.mark.asyncio
    async def test_appeal_missing_reason_raises(self, mechanism):
        """Test appeal with empty reason raises error."""
        grv = await mechanism.submit_grievance(
            "STK-001", "OP-001", "Test", "Desc",
            GrievanceSeverity.STANDARD, "email",
        )
        await mechanism.triage_grievance(grv.grievance_id, "inv-001")
        await mechanism.investigate(grv.grievance_id, [{"finding": "Done"}])
        await mechanism.resolve(grv.grievance_id, [{"action": "fix"}], "Done.")
        with pytest.raises(ValueError, match="reason is required"):
            await mechanism.appeal(grv.grievance_id, "")

    @pytest.mark.asyncio
    async def test_appeal_within_window(self, mechanism):
        """Test appeal within appeal window succeeds."""
        grv = await mechanism.submit_grievance(
            "STK-001", "OP-001", "Test", "Desc",
            GrievanceSeverity.MINOR, "email",
        )
        await mechanism.triage_grievance(grv.grievance_id, "inv-001")
        await mechanism.investigate(grv.grievance_id, [{"finding": "Done"}])
        await mechanism.resolve(grv.grievance_id, [{"action": "fix"}], "Done.")
        appealed = await mechanism.appeal(grv.grievance_id, "Inadequate resolution")
        assert appealed.status == GrievanceStatus.APPEALED

    @pytest.mark.asyncio
    async def test_appeal_reopens_investigation(self, mechanism):
        """Test appeal can lead to reopened investigation."""
        grv = await mechanism.submit_grievance(
            "STK-001", "OP-001", "Test", "Desc",
            GrievanceSeverity.STANDARD, "email",
        )
        await mechanism.triage_grievance(grv.grievance_id, "inv-001")
        await mechanism.investigate(grv.grievance_id, [{"finding": "Done"}])
        await mechanism.resolve(grv.grievance_id, [{"action": "fix"}], "Done.")
        appealed = await mechanism.appeal(grv.grievance_id, "Need re-investigation")
        assert appealed.status in [GrievanceStatus.APPEALED, GrievanceStatus.REOPENED]

    @pytest.mark.asyncio
    async def test_appeal_preserves_history(self, mechanism):
        """Test appeal preserves full grievance history."""
        grv = await mechanism.submit_grievance(
            "STK-001", "OP-001", "Test", "Desc",
            GrievanceSeverity.STANDARD, "email",
        )
        await mechanism.triage_grievance(grv.grievance_id, "inv-001")
        await mechanism.investigate(grv.grievance_id, [{"finding": "Note 1"}])
        await mechanism.resolve(grv.grievance_id, [{"action": "fix"}], "Done.")
        appealed = await mechanism.appeal(grv.grievance_id, "Unsatisfied")
        assert len(appealed.investigation_notes) >= 1
        assert len(appealed.resolution_actions) >= 1

    @pytest.mark.asyncio
    async def test_double_appeal_handled(self, mechanism):
        """Test double appeal is handled gracefully."""
        grv = await mechanism.submit_grievance(
            "STK-001", "OP-001", "Test", "Desc",
            GrievanceSeverity.STANDARD, "email",
        )
        await mechanism.triage_grievance(grv.grievance_id, "inv-001")
        await mechanism.investigate(grv.grievance_id, [{"finding": "Done"}])
        await mechanism.resolve(grv.grievance_id, [{"action": "fix"}], "Done.")
        await mechanism.appeal(grv.grievance_id, "First appeal")
        # Second appeal should be handled
        result = await mechanism.appeal(grv.grievance_id, "Second appeal")
        assert result.status in [GrievanceStatus.APPEALED, GrievanceStatus.REOPENED]
