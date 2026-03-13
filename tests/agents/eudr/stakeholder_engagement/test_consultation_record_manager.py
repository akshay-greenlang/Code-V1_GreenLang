# -*- coding: utf-8 -*-
"""
Unit tests for ConsultationRecordManager Engine - AGENT-EUDR-031

Tests consultation creation, participant management, outcome recording,
evidence attachment, finalization, and register generation.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from greenlang.agents.eudr.stakeholder_engagement.config import (
    StakeholderEngagementConfig,
)
from greenlang.agents.eudr.stakeholder_engagement.consultation_record_manager import (
    ConsultationRecordManager,
)
from greenlang.agents.eudr.stakeholder_engagement.models import (
    ConsultationRecord,
    ConsultationType,
)
from greenlang.agents.eudr.stakeholder_engagement.provenance import (
    ProvenanceTracker,
)


@pytest.fixture
def config():
    return StakeholderEngagementConfig()


@pytest.fixture
def manager(config):
    return ConsultationRecordManager(config=config)


# ---------------------------------------------------------------------------
# Test: CreateConsultation
# ---------------------------------------------------------------------------

class TestCreateConsultation:
    """Test consultation record creation for all types."""

    @pytest.mark.asyncio
    async def test_create_community_meeting(self, manager):
        """Test creating a community meeting consultation."""
        record = await manager.create_consultation(
            operator_id="OP-001",
            consultation_type=ConsultationType.COMMUNITY_MEETING,
            title="Q1 Community Meeting",
            scheduled_at=datetime.now(tz=timezone.utc) + timedelta(days=14),
            location="Community Center, Antioquia",
            stakeholder_ids=["STK-IND-001", "STK-COM-001"],
        )
        assert record.consultation_id.startswith("CON-")
        assert record.consultation_type == ConsultationType.COMMUNITY_MEETING
        assert record.status == "scheduled"

    @pytest.mark.asyncio
    async def test_create_bilateral_meeting(self, manager):
        """Test creating a bilateral meeting."""
        record = await manager.create_consultation(
            operator_id="OP-001",
            consultation_type=ConsultationType.BILATERAL,
            title="Partnership Review",
            scheduled_at=datetime.now(tz=timezone.utc) + timedelta(days=7),
            location="Cooperative Office",
            stakeholder_ids=["STK-COOP-001"],
        )
        assert record.consultation_type == ConsultationType.BILATERAL

    @pytest.mark.asyncio
    async def test_create_focus_group(self, manager):
        """Test creating a focus group consultation."""
        record = await manager.create_consultation(
            operator_id="OP-001",
            consultation_type=ConsultationType.FOCUS_GROUP,
            title="Water Quality Impact Assessment",
            scheduled_at=datetime.now(tz=timezone.utc) + timedelta(days=10),
            location="Field Office",
            stakeholder_ids=["STK-COM-001"],
        )
        assert record.consultation_type == ConsultationType.FOCUS_GROUP

    @pytest.mark.asyncio
    async def test_create_all_consultation_types(self, manager):
        """Test creating consultations of all types."""
        for ctype in ConsultationType:
            record = await manager.create_consultation(
                operator_id="OP-001",
                consultation_type=ctype,
                title=f"Test {ctype.value}",
                scheduled_at=datetime.now(tz=timezone.utc) + timedelta(days=7),
                stakeholder_ids=["STK-001"],
            )
            assert record.consultation_type == ctype

    @pytest.mark.asyncio
    async def test_create_consultation_missing_title_raises(self, manager):
        """Test creating consultation without title raises error."""
        with pytest.raises(ValueError, match="title is required"):
            await manager.create_consultation(
                operator_id="OP-001",
                consultation_type=ConsultationType.COMMUNITY_MEETING,
                title="",
                scheduled_at=datetime.now(tz=timezone.utc) + timedelta(days=7),
                stakeholder_ids=["STK-001"],
            )

    @pytest.mark.asyncio
    async def test_create_consultation_missing_operator_raises(self, manager):
        """Test creating consultation without operator raises error."""
        with pytest.raises(ValueError, match="operator_id is required"):
            await manager.create_consultation(
                operator_id="",
                consultation_type=ConsultationType.BILATERAL,
                title="Test",
                scheduled_at=datetime.now(tz=timezone.utc),
                stakeholder_ids=["STK-001"],
            )

    @pytest.mark.asyncio
    async def test_create_consultation_unique_ids(self, manager):
        """Test each consultation gets a unique ID."""
        ids = set()
        for i in range(5):
            record = await manager.create_consultation(
                operator_id="OP-001",
                consultation_type=ConsultationType.WORKSHOP,
                title=f"Workshop {i}",
                scheduled_at=datetime.now(tz=timezone.utc) + timedelta(days=7),
                stakeholder_ids=["STK-001"],
            )
            ids.add(record.consultation_id)
        assert len(ids) == 5

    @pytest.mark.asyncio
    async def test_create_consultation_with_language(self, manager):
        """Test creating consultation with specific language."""
        record = await manager.create_consultation(
            operator_id="OP-001",
            consultation_type=ConsultationType.PUBLIC_HEARING,
            title="Audiencia Publica",
            scheduled_at=datetime.now(tz=timezone.utc) + timedelta(days=14),
            stakeholder_ids=["STK-IND-001"],
            language="es",
        )
        assert record.language == "es"


# ---------------------------------------------------------------------------
# Test: AddParticipants
# ---------------------------------------------------------------------------

class TestAddParticipants:
    """Test participant management."""

    @pytest.mark.asyncio
    async def test_add_participants_success(self, manager, sample_participants):
        """Test adding participants to consultation."""
        record = await manager.create_consultation(
            "OP-001", ConsultationType.COMMUNITY_MEETING, "Test",
            datetime.now(tz=timezone.utc) + timedelta(days=7), ["STK-001"],
        )
        updated = await manager.add_participants(record.consultation_id, sample_participants)
        assert len(updated.participants) >= 4

    @pytest.mark.asyncio
    async def test_add_participants_nonexistent_raises(self, manager):
        """Test adding participants to nonexistent consultation raises error."""
        with pytest.raises(ValueError, match="consultation not found"):
            await manager.add_participants("CON-NONEXISTENT", [{"name": "Test"}])

    @pytest.mark.asyncio
    async def test_add_participants_empty_list_raises(self, manager):
        """Test adding empty participant list raises error."""
        record = await manager.create_consultation(
            "OP-001", ConsultationType.BILATERAL, "Test",
            datetime.now(tz=timezone.utc) + timedelta(days=7), ["STK-001"],
        )
        with pytest.raises(ValueError, match="participants are required"):
            await manager.add_participants(record.consultation_id, [])

    @pytest.mark.asyncio
    async def test_add_participants_preserves_existing(self, manager):
        """Test adding participants preserves existing participants."""
        record = await manager.create_consultation(
            "OP-001", ConsultationType.WORKSHOP, "Test",
            datetime.now(tz=timezone.utc) + timedelta(days=7), ["STK-001"],
        )
        await manager.add_participants(record.consultation_id, [{"name": "Person A"}])
        updated = await manager.add_participants(record.consultation_id, [{"name": "Person B"}])
        names = [p.get("name") for p in updated.participants]
        assert "Person A" in names
        assert "Person B" in names

    @pytest.mark.asyncio
    async def test_add_participants_with_roles(self, manager):
        """Test adding participants with specific roles."""
        record = await manager.create_consultation(
            "OP-001", ConsultationType.COMMUNITY_MEETING, "Test",
            datetime.now(tz=timezone.utc) + timedelta(days=7), ["STK-001"],
        )
        participants = [
            {"name": "Leader", "role": "facilitator"},
            {"name": "Observer", "role": "observer"},
        ]
        updated = await manager.add_participants(record.consultation_id, participants)
        roles = [p.get("role") for p in updated.participants]
        assert "facilitator" in roles

    @pytest.mark.asyncio
    async def test_add_participants_returns_updated_record(self, manager):
        """Test add_participants returns ConsultationRecord."""
        record = await manager.create_consultation(
            "OP-001", ConsultationType.BILATERAL, "Test",
            datetime.now(tz=timezone.utc) + timedelta(days=7), ["STK-001"],
        )
        updated = await manager.add_participants(record.consultation_id, [{"name": "Test"}])
        assert isinstance(updated, ConsultationRecord)


# ---------------------------------------------------------------------------
# Test: RecordOutcomes
# ---------------------------------------------------------------------------

class TestRecordOutcomes:
    """Test outcome recording."""

    @pytest.mark.asyncio
    async def test_record_outcomes_success(self, manager, sample_outcomes_commitments):
        """Test recording outcomes successfully."""
        record = await manager.create_consultation(
            "OP-001", ConsultationType.COMMUNITY_MEETING, "Test",
            datetime.now(tz=timezone.utc), ["STK-001"],
        )
        updated = await manager.record_outcomes(record.consultation_id, sample_outcomes_commitments)
        assert len(updated.outcomes) >= 3

    @pytest.mark.asyncio
    async def test_record_outcomes_nonexistent_raises(self, manager):
        """Test recording outcomes for nonexistent consultation raises error."""
        with pytest.raises(ValueError, match="consultation not found"):
            await manager.record_outcomes("CON-NONEXISTENT", [{"type": "agreement"}])

    @pytest.mark.asyncio
    async def test_record_outcomes_empty_list_raises(self, manager):
        """Test recording empty outcomes raises error."""
        record = await manager.create_consultation(
            "OP-001", ConsultationType.BILATERAL, "Test",
            datetime.now(tz=timezone.utc), ["STK-001"],
        )
        with pytest.raises(ValueError, match="outcomes are required"):
            await manager.record_outcomes(record.consultation_id, [])

    @pytest.mark.asyncio
    async def test_record_outcomes_agreement_type(self, manager):
        """Test recording agreement type outcomes."""
        record = await manager.create_consultation(
            "OP-001", ConsultationType.BILATERAL, "Test",
            datetime.now(tz=timezone.utc), ["STK-001"],
        )
        outcomes = [
            {"type": "agreement", "description": "Partnership extended"},
        ]
        updated = await manager.record_outcomes(record.consultation_id, outcomes)
        assert updated.outcomes[0]["type"] == "agreement"

    @pytest.mark.asyncio
    async def test_record_outcomes_commitment_type(self, manager):
        """Test recording commitment type outcomes."""
        record = await manager.create_consultation(
            "OP-001", ConsultationType.COMMUNITY_MEETING, "Test",
            datetime.now(tz=timezone.utc), ["STK-001"],
        )
        outcomes = [
            {"type": "commitment", "description": "Monthly reports to community"},
        ]
        updated = await manager.record_outcomes(record.consultation_id, outcomes)
        assert any(o["type"] == "commitment" for o in updated.outcomes)

    @pytest.mark.asyncio
    async def test_record_outcomes_action_item(self, manager):
        """Test recording action item outcomes."""
        record = await manager.create_consultation(
            "OP-001", ConsultationType.WORKSHOP, "Test",
            datetime.now(tz=timezone.utc), ["STK-001"],
        )
        outcomes = [
            {"type": "action_item", "description": "Follow-up meeting in 30 days"},
        ]
        updated = await manager.record_outcomes(record.consultation_id, outcomes)
        assert any(o["type"] == "action_item" for o in updated.outcomes)

    @pytest.mark.asyncio
    async def test_record_outcomes_multiple_types(self, manager):
        """Test recording outcomes of multiple types."""
        record = await manager.create_consultation(
            "OP-001", ConsultationType.COMMUNITY_MEETING, "Test",
            datetime.now(tz=timezone.utc), ["STK-001"],
        )
        outcomes = [
            {"type": "agreement", "description": "A1"},
            {"type": "commitment", "description": "C1"},
            {"type": "action_item", "description": "AI1"},
        ]
        updated = await manager.record_outcomes(record.consultation_id, outcomes)
        types = {o["type"] for o in updated.outcomes}
        assert types == {"agreement", "commitment", "action_item"}

    @pytest.mark.asyncio
    async def test_record_outcomes_preserves_existing(self, manager):
        """Test recording outcomes preserves existing outcomes."""
        record = await manager.create_consultation(
            "OP-001", ConsultationType.BILATERAL, "Test",
            datetime.now(tz=timezone.utc), ["STK-001"],
        )
        await manager.record_outcomes(record.consultation_id, [{"type": "agreement", "description": "First"}])
        updated = await manager.record_outcomes(record.consultation_id, [{"type": "commitment", "description": "Second"}])
        assert len(updated.outcomes) >= 2


# ---------------------------------------------------------------------------
# Test: AttachEvidence
# ---------------------------------------------------------------------------

class TestAttachEvidence:
    """Test evidence attachment."""

    @pytest.mark.asyncio
    async def test_attach_evidence_success(self, manager):
        """Test attaching evidence to consultation."""
        record = await manager.create_consultation(
            "OP-001", ConsultationType.COMMUNITY_MEETING, "Test",
            datetime.now(tz=timezone.utc), ["STK-001"],
        )
        updated = await manager.attach_evidence(
            record.consultation_id, ["PHOTO-001", "MINUTES-001"],
        )
        assert "PHOTO-001" in updated.evidence_refs
        assert "MINUTES-001" in updated.evidence_refs

    @pytest.mark.asyncio
    async def test_attach_evidence_nonexistent_raises(self, manager):
        """Test attaching evidence to nonexistent consultation raises error."""
        with pytest.raises(ValueError, match="consultation not found"):
            await manager.attach_evidence("CON-NONEXISTENT", ["PHOTO-001"])

    @pytest.mark.asyncio
    async def test_attach_evidence_empty_list_raises(self, manager):
        """Test attaching empty evidence list raises error."""
        record = await manager.create_consultation(
            "OP-001", ConsultationType.BILATERAL, "Test",
            datetime.now(tz=timezone.utc), ["STK-001"],
        )
        with pytest.raises(ValueError, match="evidence_refs are required"):
            await manager.attach_evidence(record.consultation_id, [])

    @pytest.mark.asyncio
    async def test_attach_evidence_multiple_types(self, manager):
        """Test attaching various evidence types."""
        record = await manager.create_consultation(
            "OP-001", ConsultationType.COMMUNITY_MEETING, "Test",
            datetime.now(tz=timezone.utc), ["STK-001"],
        )
        evidence = ["PHOTO-001", "VIDEO-001", "MINUTES-001", "ATTENDANCE-001", "MAP-001"]
        updated = await manager.attach_evidence(record.consultation_id, evidence)
        assert len(updated.evidence_refs) >= 5

    @pytest.mark.asyncio
    async def test_attach_evidence_preserves_existing(self, manager):
        """Test attaching evidence preserves existing references."""
        record = await manager.create_consultation(
            "OP-001", ConsultationType.WORKSHOP, "Test",
            datetime.now(tz=timezone.utc), ["STK-001"],
        )
        await manager.attach_evidence(record.consultation_id, ["PHOTO-001"])
        updated = await manager.attach_evidence(record.consultation_id, ["VIDEO-001"])
        assert "PHOTO-001" in updated.evidence_refs
        assert "VIDEO-001" in updated.evidence_refs

    @pytest.mark.asyncio
    async def test_attach_evidence_returns_record(self, manager):
        """Test attach_evidence returns ConsultationRecord."""
        record = await manager.create_consultation(
            "OP-001", ConsultationType.FIELD_VISIT, "Test",
            datetime.now(tz=timezone.utc), ["STK-001"],
        )
        updated = await manager.attach_evidence(record.consultation_id, ["DOC-001"])
        assert isinstance(updated, ConsultationRecord)


# ---------------------------------------------------------------------------
# Test: FinalizeConsultation
# ---------------------------------------------------------------------------

class TestFinalizeConsultation:
    """Test consultation finalization."""

    @pytest.mark.asyncio
    async def test_finalize_success(self, manager, sample_participants, sample_outcomes_commitments):
        """Test successful consultation finalization."""
        record = await manager.create_consultation(
            "OP-001", ConsultationType.COMMUNITY_MEETING, "Test",
            datetime.now(tz=timezone.utc), ["STK-001"],
        )
        await manager.add_participants(record.consultation_id, sample_participants)
        await manager.record_outcomes(record.consultation_id, sample_outcomes_commitments)
        finalized = await manager.finalize_consultation(record.consultation_id)
        assert finalized.status == "completed"

    @pytest.mark.asyncio
    async def test_finalize_sets_conducted_at(self, manager, sample_participants, sample_outcomes_commitments):
        """Test finalization sets conducted_at timestamp."""
        record = await manager.create_consultation(
            "OP-001", ConsultationType.BILATERAL, "Test",
            datetime.now(tz=timezone.utc), ["STK-001"],
        )
        await manager.add_participants(record.consultation_id, sample_participants)
        await manager.record_outcomes(record.consultation_id, sample_outcomes_commitments)
        finalized = await manager.finalize_consultation(record.consultation_id)
        assert finalized.conducted_at is not None

    @pytest.mark.asyncio
    async def test_finalize_nonexistent_raises(self, manager):
        """Test finalizing nonexistent consultation raises error."""
        with pytest.raises(ValueError, match="consultation not found"):
            await manager.finalize_consultation("CON-NONEXISTENT")

    @pytest.mark.asyncio
    async def test_finalize_without_participants_raises(self, manager):
        """Test finalizing without participants raises error."""
        record = await manager.create_consultation(
            "OP-001", ConsultationType.COMMUNITY_MEETING, "Test",
            datetime.now(tz=timezone.utc), ["STK-001"],
        )
        with pytest.raises(ValueError, match="participants are required"):
            await manager.finalize_consultation(record.consultation_id)

    @pytest.mark.asyncio
    async def test_finalize_without_outcomes_raises(self, manager, sample_participants):
        """Test finalizing without outcomes raises error."""
        record = await manager.create_consultation(
            "OP-001", ConsultationType.COMMUNITY_MEETING, "Test",
            datetime.now(tz=timezone.utc), ["STK-001"],
        )
        await manager.add_participants(record.consultation_id, sample_participants)
        with pytest.raises(ValueError, match="outcomes are required"):
            await manager.finalize_consultation(record.consultation_id)

    @pytest.mark.asyncio
    async def test_finalize_already_completed_raises(self, manager, sample_participants, sample_outcomes_commitments):
        """Test finalizing already completed consultation raises error."""
        record = await manager.create_consultation(
            "OP-001", ConsultationType.BILATERAL, "Test",
            datetime.now(tz=timezone.utc), ["STK-001"],
        )
        await manager.add_participants(record.consultation_id, sample_participants)
        await manager.record_outcomes(record.consultation_id, sample_outcomes_commitments)
        await manager.finalize_consultation(record.consultation_id)
        with pytest.raises(ValueError, match="already completed"):
            await manager.finalize_consultation(record.consultation_id)

    @pytest.mark.asyncio
    async def test_finalize_generates_provenance(self, manager, sample_participants, sample_outcomes_commitments):
        """Test finalization generates provenance hash."""
        record = await manager.create_consultation(
            "OP-001", ConsultationType.COMMUNITY_MEETING, "Test",
            datetime.now(tz=timezone.utc), ["STK-001"],
        )
        await manager.add_participants(record.consultation_id, sample_participants)
        await manager.record_outcomes(record.consultation_id, sample_outcomes_commitments)
        chain_before = len(manager._provenance.get_chain())
        await manager.finalize_consultation(record.consultation_id)
        assert len(manager._provenance.get_chain()) > chain_before

    @pytest.mark.asyncio
    async def test_finalize_returns_record(self, manager, sample_participants, sample_outcomes_commitments):
        """Test finalize returns ConsultationRecord."""
        record = await manager.create_consultation(
            "OP-001", ConsultationType.WORKSHOP, "Test",
            datetime.now(tz=timezone.utc), ["STK-001"],
        )
        await manager.add_participants(record.consultation_id, sample_participants)
        await manager.record_outcomes(record.consultation_id, sample_outcomes_commitments)
        finalized = await manager.finalize_consultation(record.consultation_id)
        assert isinstance(finalized, ConsultationRecord)


# ---------------------------------------------------------------------------
# Test: GenerateRegister
# ---------------------------------------------------------------------------

class TestGenerateRegister:
    """Test consultation register generation."""

    @pytest.mark.asyncio
    async def test_generate_register_returns_dict(self, manager):
        """Test register generation returns a dictionary."""
        register = await manager.generate_register(operator_id="OP-001")
        assert isinstance(register, dict)

    @pytest.mark.asyncio
    async def test_generate_register_includes_summary(self, manager):
        """Test register includes summary information."""
        register = await manager.generate_register(operator_id="OP-001")
        assert "total_consultations" in register or "summary" in register

    @pytest.mark.asyncio
    async def test_generate_register_empty_operator(self, manager):
        """Test register for operator with no consultations."""
        register = await manager.generate_register(operator_id="OP-EMPTY")
        assert isinstance(register, dict)

    @pytest.mark.asyncio
    async def test_generate_register_missing_operator_raises(self, manager):
        """Test register generation with empty operator_id raises error."""
        with pytest.raises(ValueError, match="operator_id is required"):
            await manager.generate_register(operator_id="")
