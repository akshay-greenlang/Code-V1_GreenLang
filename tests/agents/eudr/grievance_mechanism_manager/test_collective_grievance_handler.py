# -*- coding: utf-8 -*-
"""
Unit tests for CollectiveGrievanceHandler - AGENT-EUDR-032

Tests collective creation, individual addition, spokesperson management,
demand tracking, status updates, negotiation status, listing, and
health checks.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

import pytest

from greenlang.agents.eudr.grievance_mechanism_manager.config import (
    GrievanceMechanismManagerConfig,
)
from greenlang.agents.eudr.grievance_mechanism_manager.collective_grievance_handler import (
    CollectiveGrievanceHandler,
)
from greenlang.agents.eudr.grievance_mechanism_manager.models import (
    CollectiveDemand,
    CollectiveGrievanceRecord,
    CollectiveStatus,
    NegotiationStatus,
)


@pytest.fixture
def config():
    return GrievanceMechanismManagerConfig()


@pytest.fixture
def handler(config):
    return CollectiveGrievanceHandler(config=config)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInit:
    def test_handler_created(self, handler):
        assert handler is not None

    def test_default_config(self):
        h = CollectiveGrievanceHandler()
        assert h.config is not None

    def test_empty_collectives(self, handler):
        assert len(handler._collectives) == 0


# ---------------------------------------------------------------------------
# Create Collective
# ---------------------------------------------------------------------------


class TestCreateCollective:
    @pytest.mark.asyncio
    async def test_returns_record(self, handler):
        record = await handler.create_collective(
            "OP-001", "Community Water Rights", ["g-001", "g-002", "g-003"],
        )
        assert isinstance(record, CollectiveGrievanceRecord)

    @pytest.mark.asyncio
    async def test_initial_status_forming(self, handler):
        record = await handler.create_collective(
            "OP-001", "Title", ["g-001"],
        )
        assert record.collective_status == CollectiveStatus.FORMING

    @pytest.mark.asyncio
    async def test_initial_negotiation_not_started(self, handler):
        record = await handler.create_collective(
            "OP-001", "Title", ["g-001"],
        )
        assert record.negotiation_status == NegotiationStatus.NOT_STARTED

    @pytest.mark.asyncio
    async def test_title_set(self, handler):
        record = await handler.create_collective(
            "OP-001", "Community Water Rights",
        )
        assert record.title == "Community Water Rights"

    @pytest.mark.asyncio
    async def test_individual_ids_stored(self, handler):
        ids = ["g-001", "g-002", "g-003"]
        record = await handler.create_collective("OP-001", "Title", ids)
        assert record.individual_grievance_ids == ids

    @pytest.mark.asyncio
    async def test_affected_count_from_ids(self, handler):
        record = await handler.create_collective(
            "OP-001", "Title", ["g-001", "g-002"],
        )
        assert record.affected_stakeholder_count >= 2

    @pytest.mark.asyncio
    async def test_affected_count_explicit(self, handler):
        record = await handler.create_collective(
            "OP-001", "Title", affected_count=50,
        )
        assert record.affected_stakeholder_count == 50

    @pytest.mark.asyncio
    async def test_affected_count_max_of_ids_and_explicit(self, handler):
        record = await handler.create_collective(
            "OP-001", "Title", ["g-001", "g-002"], affected_count=1,
        )
        assert record.affected_stakeholder_count == 2

    @pytest.mark.asyncio
    async def test_description_set(self, handler):
        record = await handler.create_collective(
            "OP-001", "Title", description="Detailed description",
        )
        assert record.description == "Detailed description"

    @pytest.mark.asyncio
    async def test_category_set(self, handler):
        record = await handler.create_collective(
            "OP-001", "Title", category="environmental",
        )
        assert record.grievance_category == "environmental"

    @pytest.mark.asyncio
    async def test_lead_complainant(self, handler):
        record = await handler.create_collective(
            "OP-001", "Title", lead_complainant_id="stk-001",
        )
        assert record.lead_complainant_id == "stk-001"

    @pytest.mark.asyncio
    async def test_provenance_hash_set(self, handler):
        record = await handler.create_collective("OP-001", "Title")
        assert len(record.provenance_hash) == 64

    @pytest.mark.asyncio
    async def test_record_stored(self, handler):
        record = await handler.create_collective("OP-001", "Title")
        assert record.collective_id in handler._collectives

    @pytest.mark.asyncio
    async def test_no_individual_ids(self, handler):
        record = await handler.create_collective("OP-001", "Title")
        assert record.individual_grievance_ids == []
        assert record.affected_stakeholder_count == 1


# ---------------------------------------------------------------------------
# Add Individual Grievances
# ---------------------------------------------------------------------------


class TestAddIndividualGrievances:
    @pytest.mark.asyncio
    async def test_add_grievances(self, handler):
        record = await handler.create_collective("OP-001", "Title", ["g-001"])
        record = await handler.add_individual_grievances(
            record.collective_id, ["g-002", "g-003"],
        )
        assert len(record.individual_grievance_ids) == 3

    @pytest.mark.asyncio
    async def test_no_duplicates(self, handler):
        record = await handler.create_collective("OP-001", "Title", ["g-001"])
        record = await handler.add_individual_grievances(
            record.collective_id, ["g-001", "g-002"],
        )
        assert len(record.individual_grievance_ids) == 2

    @pytest.mark.asyncio
    async def test_affected_count_updated(self, handler):
        record = await handler.create_collective("OP-001", "Title", ["g-001"])
        record = await handler.add_individual_grievances(
            record.collective_id, ["g-002", "g-003"],
        )
        assert record.affected_stakeholder_count >= 3

    @pytest.mark.asyncio
    async def test_nonexistent_collective(self, handler):
        with pytest.raises(ValueError, match="not found"):
            await handler.add_individual_grievances("nonexistent", ["g-001"])


# ---------------------------------------------------------------------------
# Set Spokesperson
# ---------------------------------------------------------------------------


class TestSetSpokesperson:
    @pytest.mark.asyncio
    async def test_set_spokesperson(self, handler):
        record = await handler.create_collective("OP-001", "Title")
        record = await handler.set_spokesperson(
            record.collective_id, "Chief Elder",
        )
        assert record.spokesperson == "Chief Elder"

    @pytest.mark.asyncio
    async def test_set_representative_body(self, handler):
        record = await handler.create_collective("OP-001", "Title")
        record = await handler.set_spokesperson(
            record.collective_id, "Elder",
            representative_body="Community Council",
        )
        assert record.representative_body == "Community Council"

    @pytest.mark.asyncio
    async def test_no_representative_body(self, handler):
        record = await handler.create_collective("OP-001", "Title")
        record = await handler.set_spokesperson(
            record.collective_id, "Elder",
        )
        assert record.representative_body is None

    @pytest.mark.asyncio
    async def test_nonexistent_collective(self, handler):
        with pytest.raises(ValueError, match="not found"):
            await handler.set_spokesperson("nonexistent", "Elder")


# ---------------------------------------------------------------------------
# Add Demands
# ---------------------------------------------------------------------------


class TestAddDemands:
    @pytest.mark.asyncio
    async def test_add_demands(self, handler, sample_collective_demands):
        record = await handler.create_collective("OP-001", "Title")
        record = await handler.add_demands(
            record.collective_id, sample_collective_demands,
        )
        assert len(record.collective_demands) == 3

    @pytest.mark.asyncio
    async def test_demand_properties(self, handler, sample_collective_demands):
        record = await handler.create_collective("OP-001", "Title")
        record = await handler.add_demands(
            record.collective_id, sample_collective_demands,
        )
        first = record.collective_demands[0]
        assert first.demand == "Clean water supply restoration"
        assert first.priority == "critical"
        assert first.negotiable is False

    @pytest.mark.asyncio
    async def test_append_more_demands(self, handler):
        record = await handler.create_collective("OP-001", "Title")
        await handler.add_demands(
            record.collective_id, [{"demand": "D1"}],
        )
        record = await handler.add_demands(
            record.collective_id, [{"demand": "D2"}],
        )
        assert len(record.collective_demands) == 2

    @pytest.mark.asyncio
    async def test_default_demand_properties(self, handler):
        record = await handler.create_collective("OP-001", "Title")
        record = await handler.add_demands(
            record.collective_id, [{"demand": "Test"}],
        )
        d = record.collective_demands[0]
        assert d.priority == "medium"
        assert d.negotiable is True

    @pytest.mark.asyncio
    async def test_nonexistent_collective(self, handler):
        with pytest.raises(ValueError, match="not found"):
            await handler.add_demands("nonexistent", [{"demand": "Test"}])


# ---------------------------------------------------------------------------
# Update Status
# ---------------------------------------------------------------------------


class TestUpdateStatus:
    @pytest.mark.asyncio
    async def test_update_to_submitted(self, handler):
        record = await handler.create_collective("OP-001", "Title")
        record = await handler.update_status(record.collective_id, "submitted")
        assert record.collective_status == CollectiveStatus.SUBMITTED

    @pytest.mark.asyncio
    async def test_update_to_investigating(self, handler):
        record = await handler.create_collective("OP-001", "Title")
        record = await handler.update_status(record.collective_id, "investigating")
        assert record.collective_status == CollectiveStatus.INVESTIGATING

    @pytest.mark.asyncio
    async def test_update_to_mediating(self, handler):
        record = await handler.create_collective("OP-001", "Title")
        record = await handler.update_status(record.collective_id, "mediating")
        assert record.collective_status == CollectiveStatus.MEDIATING

    @pytest.mark.asyncio
    async def test_resolved_sets_resolved_at(self, handler):
        record = await handler.create_collective("OP-001", "Title")
        record = await handler.update_status(record.collective_id, "resolved")
        assert record.collective_status == CollectiveStatus.RESOLVED
        assert record.resolved_at is not None

    @pytest.mark.asyncio
    async def test_closed_sets_resolved_at(self, handler):
        record = await handler.create_collective("OP-001", "Title")
        record = await handler.update_status(record.collective_id, "closed")
        assert record.collective_status == CollectiveStatus.CLOSED
        assert record.resolved_at is not None

    @pytest.mark.asyncio
    async def test_invalid_status(self, handler):
        record = await handler.create_collective("OP-001", "Title")
        with pytest.raises(ValueError, match="Invalid status"):
            await handler.update_status(record.collective_id, "invalid")

    @pytest.mark.asyncio
    async def test_nonexistent_collective(self, handler):
        with pytest.raises(ValueError, match="not found"):
            await handler.update_status("nonexistent", "submitted")


# ---------------------------------------------------------------------------
# Update Negotiation Status
# ---------------------------------------------------------------------------


class TestUpdateNegotiationStatus:
    @pytest.mark.asyncio
    async def test_in_progress(self, handler):
        record = await handler.create_collective("OP-001", "Title")
        record = await handler.update_negotiation_status(
            record.collective_id, "in_progress",
        )
        assert record.negotiation_status == NegotiationStatus.IN_PROGRESS

    @pytest.mark.asyncio
    async def test_agreement_reached(self, handler):
        record = await handler.create_collective("OP-001", "Title")
        record = await handler.update_negotiation_status(
            record.collective_id, "agreement_reached",
        )
        assert record.negotiation_status == NegotiationStatus.AGREEMENT_REACHED

    @pytest.mark.asyncio
    async def test_stalled(self, handler):
        record = await handler.create_collective("OP-001", "Title")
        record = await handler.update_negotiation_status(
            record.collective_id, "stalled",
        )
        assert record.negotiation_status == NegotiationStatus.STALLED

    @pytest.mark.asyncio
    async def test_failed(self, handler):
        record = await handler.create_collective("OP-001", "Title")
        record = await handler.update_negotiation_status(
            record.collective_id, "failed",
        )
        assert record.negotiation_status == NegotiationStatus.FAILED

    @pytest.mark.asyncio
    async def test_invalid_status(self, handler):
        record = await handler.create_collective("OP-001", "Title")
        with pytest.raises(ValueError, match="Invalid negotiation status"):
            await handler.update_negotiation_status(
                record.collective_id, "invalid",
            )

    @pytest.mark.asyncio
    async def test_nonexistent_collective(self, handler):
        with pytest.raises(ValueError, match="not found"):
            await handler.update_negotiation_status("nonexistent", "in_progress")


# ---------------------------------------------------------------------------
# Retrieval and Listing
# ---------------------------------------------------------------------------


class TestRetrievalAndListing:
    @pytest.mark.asyncio
    async def test_get_collective(self, handler):
        record = await handler.create_collective("OP-001", "Title")
        retrieved = await handler.get_collective(record.collective_id)
        assert retrieved is not None

    @pytest.mark.asyncio
    async def test_get_not_found(self, handler):
        result = await handler.get_collective("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_all(self, handler):
        await handler.create_collective("OP-001", "Title A")
        await handler.create_collective("OP-002", "Title B")
        results = await handler.list_collectives()
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_list_filter_operator(self, handler):
        await handler.create_collective("OP-001", "Title A")
        await handler.create_collective("OP-002", "Title B")
        results = await handler.list_collectives(operator_id="OP-001")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_list_filter_status(self, handler):
        c = await handler.create_collective("OP-001", "Title A")
        await handler.create_collective("OP-001", "Title B")
        await handler.update_status(c.collective_id, "submitted")
        results = await handler.list_collectives(status="submitted")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_list_filter_category(self, handler):
        await handler.create_collective("OP-001", "Title A", category="environmental")
        await handler.create_collective("OP-001", "Title B", category="labor")
        results = await handler.list_collectives(category="environmental")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_list_empty(self, handler):
        results = await handler.list_collectives()
        assert results == []


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_health_check(self, handler):
        health = await handler.health_check()
        assert health["status"] == "healthy"
        assert health["engine"] == "CollectiveGrievanceHandler"

    @pytest.mark.asyncio
    async def test_health_check_active_count(self, handler):
        c1 = await handler.create_collective("OP-001", "Title A")
        await handler.create_collective("OP-001", "Title B")
        await handler.update_status(c1.collective_id, "resolved")
        health = await handler.health_check()
        assert health["collective_count"] == 2
        assert health["active_collectives"] == 1
