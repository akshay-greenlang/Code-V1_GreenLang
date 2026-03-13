# -*- coding: utf-8 -*-
"""
Unit tests for RemediationTracker - AGENT-EUDR-032

Tests remediation creation, progress updates, satisfaction recording,
cost tracking, verification, lessons learned, listing, and health checks.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from greenlang.agents.eudr.grievance_mechanism_manager.config import (
    GrievanceMechanismManagerConfig,
)
from greenlang.agents.eudr.grievance_mechanism_manager.remediation_tracker import (
    RemediationTracker,
)
from greenlang.agents.eudr.grievance_mechanism_manager.models import (
    ImplementationStatus,
    RemediationRecord,
    RemediationType,
)


@pytest.fixture
def config():
    return GrievanceMechanismManagerConfig()


@pytest.fixture
def tracker(config):
    return RemediationTracker(config=config)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInit:
    def test_tracker_created(self, tracker):
        assert tracker is not None

    def test_default_config(self):
        t = RemediationTracker()
        assert t.config is not None

    def test_empty_remediations(self, tracker):
        assert len(tracker._remediations) == 0


# ---------------------------------------------------------------------------
# Create Remediation
# ---------------------------------------------------------------------------


class TestCreateRemediation:
    @pytest.mark.asyncio
    async def test_returns_record(self, tracker):
        record = await tracker.create_remediation(
            "g-001", "OP-001", "compensation",
        )
        assert isinstance(record, RemediationRecord)

    @pytest.mark.asyncio
    async def test_type_compensation(self, tracker):
        record = await tracker.create_remediation(
            "g-001", "OP-001", "compensation",
        )
        assert record.remediation_type == RemediationType.COMPENSATION

    @pytest.mark.asyncio
    async def test_type_process_change(self, tracker):
        record = await tracker.create_remediation(
            "g-001", "OP-001", "process_change",
        )
        assert record.remediation_type == RemediationType.PROCESS_CHANGE

    @pytest.mark.asyncio
    async def test_type_infrastructure(self, tracker):
        record = await tracker.create_remediation(
            "g-001", "OP-001", "infrastructure",
        )
        assert record.remediation_type == RemediationType.INFRASTRUCTURE

    @pytest.mark.asyncio
    async def test_invalid_type_defaults(self, tracker):
        record = await tracker.create_remediation(
            "g-001", "OP-001", "invalid_type",
        )
        assert record.remediation_type == RemediationType.PROCESS_CHANGE

    @pytest.mark.asyncio
    async def test_initial_status_planned(self, tracker):
        record = await tracker.create_remediation(
            "g-001", "OP-001", "compensation",
        )
        assert record.implementation_status == ImplementationStatus.PLANNED

    @pytest.mark.asyncio
    async def test_initial_completion_zero(self, tracker):
        record = await tracker.create_remediation(
            "g-001", "OP-001", "compensation",
        )
        assert record.completion_percentage == Decimal("0")

    @pytest.mark.asyncio
    async def test_initial_cost_zero(self, tracker):
        record = await tracker.create_remediation(
            "g-001", "OP-001", "compensation",
        )
        assert record.cost_incurred == Decimal("0")

    @pytest.mark.asyncio
    async def test_actions_parsed(self, tracker, sample_remediation_actions):
        record = await tracker.create_remediation(
            "g-001", "OP-001", "process_change",
            actions=sample_remediation_actions,
        )
        assert len(record.remediation_actions) == 2
        assert record.remediation_actions[0].action == "Install water treatment system"

    @pytest.mark.asyncio
    async def test_no_actions(self, tracker):
        record = await tracker.create_remediation(
            "g-001", "OP-001", "compensation",
        )
        assert len(record.remediation_actions) == 0

    @pytest.mark.asyncio
    async def test_provenance_hash_set(self, tracker):
        record = await tracker.create_remediation(
            "g-001", "OP-001", "compensation",
        )
        assert len(record.provenance_hash) == 64

    @pytest.mark.asyncio
    async def test_record_stored(self, tracker):
        record = await tracker.create_remediation(
            "g-001", "OP-001", "compensation",
        )
        assert record.remediation_id in tracker._remediations

    @pytest.mark.asyncio
    async def test_grievance_id_set(self, tracker):
        record = await tracker.create_remediation(
            "g-001", "OP-001", "compensation",
        )
        assert record.grievance_id == "g-001"

    @pytest.mark.asyncio
    async def test_operator_id_set(self, tracker):
        record = await tracker.create_remediation(
            "g-001", "OP-001", "compensation",
        )
        assert record.operator_id == "OP-001"


# ---------------------------------------------------------------------------
# Update Progress
# ---------------------------------------------------------------------------


class TestUpdateProgress:
    @pytest.mark.asyncio
    async def test_update_progress(self, tracker):
        record = await tracker.create_remediation("g-001", "OP-001", "compensation")
        record = await tracker.update_progress(record.remediation_id, 50.0)
        assert record.completion_percentage == Decimal("50")

    @pytest.mark.asyncio
    async def test_update_with_status(self, tracker):
        record = await tracker.create_remediation("g-001", "OP-001", "compensation")
        record = await tracker.update_progress(
            record.remediation_id, 50.0, status="in_progress",
        )
        assert record.implementation_status == ImplementationStatus.IN_PROGRESS

    @pytest.mark.asyncio
    async def test_auto_complete_at_100(self, tracker):
        record = await tracker.create_remediation("g-001", "OP-001", "compensation")
        record = await tracker.update_progress(record.remediation_id, 100.0)
        assert record.implementation_status == ImplementationStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_clamp_above_100(self, tracker):
        record = await tracker.create_remediation("g-001", "OP-001", "compensation")
        record = await tracker.update_progress(record.remediation_id, 150.0)
        assert record.completion_percentage == Decimal("100")

    @pytest.mark.asyncio
    async def test_clamp_below_zero(self, tracker):
        record = await tracker.create_remediation("g-001", "OP-001", "compensation")
        record = await tracker.update_progress(record.remediation_id, -10.0)
        assert record.completion_percentage == Decimal("0")

    @pytest.mark.asyncio
    async def test_invalid_status_ignored(self, tracker):
        record = await tracker.create_remediation("g-001", "OP-001", "compensation")
        record = await tracker.update_progress(
            record.remediation_id, 50.0, status="invalid",
        )
        assert record.implementation_status == ImplementationStatus.PLANNED

    @pytest.mark.asyncio
    async def test_nonexistent_remediation(self, tracker):
        with pytest.raises(ValueError, match="not found"):
            await tracker.update_progress("nonexistent", 50.0)


# ---------------------------------------------------------------------------
# Record Satisfaction
# ---------------------------------------------------------------------------


class TestRecordSatisfaction:
    @pytest.mark.asyncio
    async def test_record_satisfaction(self, tracker):
        record = await tracker.create_remediation("g-001", "OP-001", "compensation")
        record = await tracker.record_satisfaction(record.remediation_id, 4.5)
        assert record.stakeholder_satisfaction == Decimal("4.5")

    @pytest.mark.asyncio
    async def test_clamp_above_five(self, tracker):
        record = await tracker.create_remediation("g-001", "OP-001", "compensation")
        record = await tracker.record_satisfaction(record.remediation_id, 6.0)
        assert record.stakeholder_satisfaction == Decimal("5.0")

    @pytest.mark.asyncio
    async def test_clamp_below_one(self, tracker):
        record = await tracker.create_remediation("g-001", "OP-001", "compensation")
        record = await tracker.record_satisfaction(record.remediation_id, 0.5)
        assert record.stakeholder_satisfaction == Decimal("1.0")

    @pytest.mark.asyncio
    async def test_nonexistent_remediation(self, tracker):
        with pytest.raises(ValueError, match="not found"):
            await tracker.record_satisfaction("nonexistent", 4.0)


# ---------------------------------------------------------------------------
# Record Cost
# ---------------------------------------------------------------------------


class TestRecordCost:
    @pytest.mark.asyncio
    async def test_record_cost(self, tracker):
        record = await tracker.create_remediation("g-001", "OP-001", "compensation")
        record = await tracker.record_cost(record.remediation_id, 15000.0)
        assert record.cost_incurred == Decimal("15000")

    @pytest.mark.asyncio
    async def test_negative_cost_clamped(self, tracker):
        record = await tracker.create_remediation("g-001", "OP-001", "compensation")
        record = await tracker.record_cost(record.remediation_id, -500.0)
        assert record.cost_incurred == Decimal("0")

    @pytest.mark.asyncio
    async def test_nonexistent_remediation(self, tracker):
        with pytest.raises(ValueError, match="not found"):
            await tracker.record_cost("nonexistent", 100.0)


# ---------------------------------------------------------------------------
# Verify Remediation
# ---------------------------------------------------------------------------


class TestVerifyRemediation:
    @pytest.mark.asyncio
    async def test_verify(self, tracker):
        record = await tracker.create_remediation("g-001", "OP-001", "compensation")
        evidence = [{"type": "photo", "url": "https://evidence.example.com/1.jpg"}]
        record = await tracker.verify_remediation(
            record.remediation_id, evidence,
        )
        assert record.implementation_status == ImplementationStatus.VERIFIED
        assert record.verified_at is not None

    @pytest.mark.asyncio
    async def test_evidence_appended(self, tracker):
        record = await tracker.create_remediation("g-001", "OP-001", "compensation")
        await tracker.verify_remediation(
            record.remediation_id, [{"type": "photo"}],
        )
        record = await tracker.verify_remediation(
            record.remediation_id, [{"type": "document"}],
        )
        assert len(record.verification_evidence) == 2

    @pytest.mark.asyncio
    async def test_effectiveness_indicators_set(self, tracker):
        record = await tracker.create_remediation("g-001", "OP-001", "compensation")
        indicators = {"recurrence_rate": 0.0, "stakeholder_approval": 95}
        record = await tracker.verify_remediation(
            record.remediation_id, [{"type": "report"}],
            effectiveness_indicators=indicators,
        )
        assert record.effectiveness_indicators["stakeholder_approval"] == 95

    @pytest.mark.asyncio
    async def test_provenance_hash_updated(self, tracker):
        record = await tracker.create_remediation("g-001", "OP-001", "compensation")
        old_hash = record.provenance_hash
        record = await tracker.verify_remediation(
            record.remediation_id, [{"type": "inspection"}],
        )
        assert record.provenance_hash != old_hash

    @pytest.mark.asyncio
    async def test_nonexistent_remediation(self, tracker):
        with pytest.raises(ValueError, match="not found"):
            await tracker.verify_remediation("nonexistent", [])


# ---------------------------------------------------------------------------
# Lessons Learned
# ---------------------------------------------------------------------------


class TestLessonsLearned:
    @pytest.mark.asyncio
    async def test_add_lessons(self, tracker):
        record = await tracker.create_remediation("g-001", "OP-001", "compensation")
        record = await tracker.add_lessons_learned(
            record.remediation_id, "Regular monitoring prevents recurrence",
        )
        assert "monitoring" in record.lessons_learned

    @pytest.mark.asyncio
    async def test_nonexistent_remediation(self, tracker):
        with pytest.raises(ValueError, match="not found"):
            await tracker.add_lessons_learned("nonexistent", "Test")


# ---------------------------------------------------------------------------
# Retrieval and Listing
# ---------------------------------------------------------------------------


class TestRetrievalAndListing:
    @pytest.mark.asyncio
    async def test_get_remediation(self, tracker):
        record = await tracker.create_remediation("g-001", "OP-001", "compensation")
        retrieved = await tracker.get_remediation(record.remediation_id)
        assert retrieved is not None

    @pytest.mark.asyncio
    async def test_get_not_found(self, tracker):
        result = await tracker.get_remediation("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_all(self, tracker):
        await tracker.create_remediation("g-001", "OP-001", "compensation")
        await tracker.create_remediation("g-002", "OP-002", "process_change")
        results = await tracker.list_remediations()
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_list_filter_grievance(self, tracker):
        await tracker.create_remediation("g-001", "OP-001", "compensation")
        await tracker.create_remediation("g-002", "OP-001", "compensation")
        results = await tracker.list_remediations(grievance_id="g-001")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_list_filter_operator(self, tracker):
        await tracker.create_remediation("g-001", "OP-001", "compensation")
        await tracker.create_remediation("g-002", "OP-002", "compensation")
        results = await tracker.list_remediations(operator_id="OP-001")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_list_filter_status(self, tracker):
        rem = await tracker.create_remediation("g-001", "OP-001", "compensation")
        await tracker.create_remediation("g-002", "OP-001", "compensation")
        await tracker.update_progress(rem.remediation_id, 50.0, status="in_progress")
        results = await tracker.list_remediations(status="in_progress")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_list_filter_type(self, tracker):
        await tracker.create_remediation("g-001", "OP-001", "compensation")
        await tracker.create_remediation("g-002", "OP-001", "process_change")
        results = await tracker.list_remediations(remediation_type="compensation")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_list_empty(self, tracker):
        results = await tracker.list_remediations()
        assert results == []


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_health_check(self, tracker):
        health = await tracker.health_check()
        assert health["status"] == "healthy"
        assert health["engine"] == "RemediationTracker"

    @pytest.mark.asyncio
    async def test_health_check_open_count(self, tracker):
        rem1 = await tracker.create_remediation("g-001", "OP-001", "compensation")
        rem2 = await tracker.create_remediation("g-002", "OP-001", "compensation")
        await tracker.update_progress(rem1.remediation_id, 100.0)
        health = await tracker.health_check()
        assert health["remediation_count"] == 2
        assert health["open_remediations"] == 1
