# -*- coding: utf-8 -*-
"""
Unit tests for MediationWorkflowManager - AGENT-EUDR-032

Tests 7-stage state machine, session recording, agreement recording,
settlement handling, stage advancement, retrieval, listing, and health checks.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

import pytest

from greenlang.agents.eudr.grievance_mechanism_manager.config import (
    GrievanceMechanismManagerConfig,
)
from greenlang.agents.eudr.grievance_mechanism_manager.mediation_workflow_manager import (
    MediationWorkflowManager,
)
from greenlang.agents.eudr.grievance_mechanism_manager.models import (
    MediationRecord,
    MediationSession,
    MediationStage,
    MediatorType,
    SettlementStatus,
)


@pytest.fixture
def config():
    return GrievanceMechanismManagerConfig()


@pytest.fixture
def manager(config):
    return MediationWorkflowManager(config=config)


@pytest.fixture
def parties():
    return [
        {"role": "complainant", "name": "Community A", "id": "stk-001"},
        {"role": "respondent", "name": "Operator Corp", "id": "OP-001"},
    ]


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInit:
    def test_manager_created(self, manager):
        assert manager is not None

    def test_default_config(self):
        m = MediationWorkflowManager()
        assert m.config is not None

    def test_empty_mediations(self, manager):
        assert len(manager._mediations) == 0


# ---------------------------------------------------------------------------
# Initiate Mediation
# ---------------------------------------------------------------------------


class TestInitiateMediation:
    @pytest.mark.asyncio
    async def test_returns_record(self, manager, parties):
        record = await manager.initiate_mediation(
            "g-001", "OP-001", parties,
        )
        assert isinstance(record, MediationRecord)

    @pytest.mark.asyncio
    async def test_initial_stage_is_initiated(self, manager, parties):
        record = await manager.initiate_mediation(
            "g-001", "OP-001", parties,
        )
        assert record.mediation_stage == MediationStage.INITIATED

    @pytest.mark.asyncio
    async def test_default_mediator_type_internal(self, manager, parties):
        record = await manager.initiate_mediation(
            "g-001", "OP-001", parties,
        )
        assert record.mediator_type == MediatorType.INTERNAL

    @pytest.mark.asyncio
    async def test_custom_mediator_type(self, manager, parties):
        record = await manager.initiate_mediation(
            "g-001", "OP-001", parties, mediator_type="external",
        )
        assert record.mediator_type == MediatorType.EXTERNAL

    @pytest.mark.asyncio
    async def test_invalid_mediator_type_defaults(self, manager, parties):
        record = await manager.initiate_mediation(
            "g-001", "OP-001", parties, mediator_type="invalid",
        )
        assert record.mediator_type == MediatorType.INTERNAL

    @pytest.mark.asyncio
    async def test_settlement_status_pending(self, manager, parties):
        record = await manager.initiate_mediation(
            "g-001", "OP-001", parties,
        )
        assert record.settlement_status == SettlementStatus.PENDING

    @pytest.mark.asyncio
    async def test_parties_stored(self, manager, parties):
        record = await manager.initiate_mediation(
            "g-001", "OP-001", parties,
        )
        assert len(record.parties) == 2

    @pytest.mark.asyncio
    async def test_session_count_zero(self, manager, parties):
        record = await manager.initiate_mediation(
            "g-001", "OP-001", parties,
        )
        assert record.session_count == 0

    @pytest.mark.asyncio
    async def test_provenance_hash_set(self, manager, parties):
        record = await manager.initiate_mediation(
            "g-001", "OP-001", parties,
        )
        assert len(record.provenance_hash) == 64

    @pytest.mark.asyncio
    async def test_record_stored(self, manager, parties):
        record = await manager.initiate_mediation(
            "g-001", "OP-001", parties,
        )
        assert record.mediation_id in manager._mediations

    @pytest.mark.asyncio
    async def test_mediator_id_set(self, manager, parties):
        record = await manager.initiate_mediation(
            "g-001", "OP-001", parties, mediator_id="med-ext-001",
        )
        assert record.mediator_id == "med-ext-001"


# ---------------------------------------------------------------------------
# Advance Stage
# ---------------------------------------------------------------------------


class TestAdvanceStage:
    @pytest.mark.asyncio
    async def test_advance_to_next(self, manager, parties):
        record = await manager.initiate_mediation("g-001", "OP-001", parties)
        record = await manager.advance_stage(record.mediation_id)
        assert record.mediation_stage == MediationStage.PREPARATION

    @pytest.mark.asyncio
    async def test_advance_multiple_steps(self, manager, parties):
        record = await manager.initiate_mediation("g-001", "OP-001", parties)
        record = await manager.advance_stage(record.mediation_id)
        record = await manager.advance_stage(record.mediation_id)
        assert record.mediation_stage == MediationStage.DIALOGUE

    @pytest.mark.asyncio
    async def test_advance_to_specific_stage(self, manager, parties):
        record = await manager.initiate_mediation("g-001", "OP-001", parties)
        record = await manager.advance_stage(record.mediation_id, "negotiation")
        assert record.mediation_stage == MediationStage.NEGOTIATION

    @pytest.mark.asyncio
    async def test_advance_to_closed_sets_completed(self, manager, parties):
        record = await manager.initiate_mediation("g-001", "OP-001", parties)
        record = await manager.advance_stage(record.mediation_id, "closed")
        assert record.mediation_stage == MediationStage.CLOSED
        assert record.completed_at is not None

    @pytest.mark.asyncio
    async def test_cannot_advance_from_closed(self, manager, parties):
        record = await manager.initiate_mediation("g-001", "OP-001", parties)
        await manager.advance_stage(record.mediation_id, "closed")
        with pytest.raises(ValueError, match="already in final stage"):
            await manager.advance_stage(record.mediation_id)

    @pytest.mark.asyncio
    async def test_cannot_move_backward(self, manager, parties):
        record = await manager.initiate_mediation("g-001", "OP-001", parties)
        await manager.advance_stage(record.mediation_id, "dialogue")
        with pytest.raises(ValueError, match="must advance forward"):
            await manager.advance_stage(record.mediation_id, "preparation")

    @pytest.mark.asyncio
    async def test_cannot_advance_same_stage(self, manager, parties):
        record = await manager.initiate_mediation("g-001", "OP-001", parties)
        await manager.advance_stage(record.mediation_id, "dialogue")
        with pytest.raises(ValueError, match="must advance forward"):
            await manager.advance_stage(record.mediation_id, "dialogue")

    @pytest.mark.asyncio
    async def test_invalid_target_stage(self, manager, parties):
        record = await manager.initiate_mediation("g-001", "OP-001", parties)
        with pytest.raises(ValueError, match="Invalid mediation stage"):
            await manager.advance_stage(record.mediation_id, "nonexistent")

    @pytest.mark.asyncio
    async def test_nonexistent_mediation(self, manager):
        with pytest.raises(ValueError, match="not found"):
            await manager.advance_stage("nonexistent")

    @pytest.mark.asyncio
    async def test_full_lifecycle(self, manager, parties):
        """Walk through all 7 stages."""
        record = await manager.initiate_mediation("g-001", "OP-001", parties)
        stages = [
            MediationStage.PREPARATION, MediationStage.DIALOGUE,
            MediationStage.NEGOTIATION, MediationStage.SETTLEMENT,
            MediationStage.IMPLEMENTATION, MediationStage.CLOSED,
        ]
        for expected in stages:
            record = await manager.advance_stage(record.mediation_id)
            assert record.mediation_stage == expected


# ---------------------------------------------------------------------------
# Record Session
# ---------------------------------------------------------------------------


class TestRecordSession:
    @pytest.mark.asyncio
    async def test_record_session(self, manager, parties):
        record = await manager.initiate_mediation("g-001", "OP-001", parties)
        record = await manager.record_session(
            record.mediation_id,
            {"summary": "Initial dialogue", "attendees": ["A", "B"], "duration_minutes": 90},
        )
        assert record.session_count == 1
        assert record.total_duration_minutes == 90

    @pytest.mark.asyncio
    async def test_multiple_sessions(self, manager, parties):
        record = await manager.initiate_mediation("g-001", "OP-001", parties)
        await manager.record_session(record.mediation_id, {"duration_minutes": 60})
        record = await manager.record_session(record.mediation_id, {"duration_minutes": 90})
        assert record.session_count == 2
        assert record.total_duration_minutes == 150

    @pytest.mark.asyncio
    async def test_session_default_duration(self, manager, parties):
        record = await manager.initiate_mediation("g-001", "OP-001", parties)
        record = await manager.record_session(record.mediation_id, {})
        assert record.total_duration_minutes == manager.config.mediation_default_session_minutes

    @pytest.mark.asyncio
    async def test_max_sessions_exceeded(self, manager, parties):
        record = await manager.initiate_mediation("g-001", "OP-001", parties)
        for i in range(manager.config.mediation_max_sessions):
            await manager.record_session(record.mediation_id, {"duration_minutes": 10})
        with pytest.raises(ValueError, match="Maximum sessions"):
            await manager.record_session(record.mediation_id, {"duration_minutes": 10})

    @pytest.mark.asyncio
    async def test_session_stored_in_list(self, manager, parties):
        record = await manager.initiate_mediation("g-001", "OP-001", parties)
        record = await manager.record_session(
            record.mediation_id, {"summary": "Test", "attendees": ["X"]},
        )
        assert len(record.sessions) == 1
        assert record.sessions[0].summary == "Test"

    @pytest.mark.asyncio
    async def test_nonexistent_mediation(self, manager):
        with pytest.raises(ValueError, match="not found"):
            await manager.record_session("nonexistent", {})

    @pytest.mark.asyncio
    async def test_session_numbering(self, manager, parties):
        record = await manager.initiate_mediation("g-001", "OP-001", parties)
        await manager.record_session(record.mediation_id, {})
        record = await manager.record_session(record.mediation_id, {})
        assert record.sessions[0].session_number == 1
        assert record.sessions[1].session_number == 2


# ---------------------------------------------------------------------------
# Record Agreement
# ---------------------------------------------------------------------------


class TestRecordAgreement:
    @pytest.mark.asyncio
    async def test_record_agreement(self, manager, parties):
        record = await manager.initiate_mediation("g-001", "OP-001", parties)
        record = await manager.record_agreement(
            record.mediation_id,
            {"terms": "Community water access restored"},
        )
        assert len(record.agreements) == 1
        assert "recorded_at" in record.agreements[0]

    @pytest.mark.asyncio
    async def test_multiple_agreements(self, manager, parties):
        record = await manager.initiate_mediation("g-001", "OP-001", parties)
        await manager.record_agreement(record.mediation_id, {"terms": "Term 1"})
        record = await manager.record_agreement(record.mediation_id, {"terms": "Term 2"})
        assert len(record.agreements) == 2

    @pytest.mark.asyncio
    async def test_nonexistent_mediation(self, manager):
        with pytest.raises(ValueError, match="not found"):
            await manager.record_agreement("nonexistent", {})


# ---------------------------------------------------------------------------
# Set Settlement
# ---------------------------------------------------------------------------


class TestSetSettlement:
    @pytest.mark.asyncio
    async def test_set_settlement_accepted(self, manager, parties):
        record = await manager.initiate_mediation("g-001", "OP-001", parties)
        record = await manager.set_settlement(
            record.mediation_id,
            {"compensation": 50000, "timeline": "30 days"},
            status="accepted",
        )
        assert record.settlement_status == SettlementStatus.ACCEPTED
        assert record.settlement_terms["compensation"] == 50000

    @pytest.mark.asyncio
    async def test_set_settlement_rejected(self, manager, parties):
        record = await manager.initiate_mediation("g-001", "OP-001", parties)
        record = await manager.set_settlement(
            record.mediation_id, {}, status="rejected",
        )
        assert record.settlement_status == SettlementStatus.REJECTED

    @pytest.mark.asyncio
    async def test_set_settlement_invalid_status_defaults(self, manager, parties):
        record = await manager.initiate_mediation("g-001", "OP-001", parties)
        record = await manager.set_settlement(
            record.mediation_id, {}, status="invalid",
        )
        assert record.settlement_status == SettlementStatus.PENDING

    @pytest.mark.asyncio
    async def test_nonexistent_mediation(self, manager):
        with pytest.raises(ValueError, match="not found"):
            await manager.set_settlement("nonexistent", {})


# ---------------------------------------------------------------------------
# Retrieval and Listing
# ---------------------------------------------------------------------------


class TestRetrievalAndListing:
    @pytest.mark.asyncio
    async def test_get_mediation(self, manager, parties):
        record = await manager.initiate_mediation("g-001", "OP-001", parties)
        retrieved = await manager.get_mediation(record.mediation_id)
        assert retrieved is not None
        assert retrieved.mediation_id == record.mediation_id

    @pytest.mark.asyncio
    async def test_get_mediation_not_found(self, manager):
        result = await manager.get_mediation("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_all(self, manager, parties):
        await manager.initiate_mediation("g-001", "OP-001", parties)
        await manager.initiate_mediation("g-002", "OP-002", parties)
        results = await manager.list_mediations()
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_list_filter_operator(self, manager, parties):
        await manager.initiate_mediation("g-001", "OP-001", parties)
        await manager.initiate_mediation("g-002", "OP-002", parties)
        results = await manager.list_mediations(operator_id="OP-001")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_list_filter_grievance(self, manager, parties):
        await manager.initiate_mediation("g-001", "OP-001", parties)
        await manager.initiate_mediation("g-002", "OP-001", parties)
        results = await manager.list_mediations(grievance_id="g-001")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_list_filter_stage(self, manager, parties):
        med1 = await manager.initiate_mediation("g-001", "OP-001", parties)
        await manager.initiate_mediation("g-002", "OP-001", parties)
        await manager.advance_stage(med1.mediation_id, "dialogue")
        results = await manager.list_mediations(stage="dialogue")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_list_empty(self, manager):
        results = await manager.list_mediations()
        assert results == []


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_health_check(self, manager):
        health = await manager.health_check()
        assert health["status"] == "healthy"
        assert health["engine"] == "MediationWorkflowManager"
        assert health["mediation_count"] == 0
        assert health["active_mediations"] == 0

    @pytest.mark.asyncio
    async def test_health_check_active_count(self, manager, parties):
        med1 = await manager.initiate_mediation("g-001", "OP-001", parties)
        await manager.initiate_mediation("g-002", "OP-001", parties)
        await manager.advance_stage(med1.mediation_id, "closed")
        health = await manager.health_check()
        assert health["mediation_count"] == 2
        assert health["active_mediations"] == 1
