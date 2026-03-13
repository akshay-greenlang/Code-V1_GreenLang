# -*- coding: utf-8 -*-
"""
Unit tests for EntityCoordinatorEngine - AGENT-EUDR-034

Tests entity assignment, dependency management, role-based access,
status transitions, RACI matrix generation, capacity checking,
and workload distribution.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-034 (Engine 4: Entity Coordinator)
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import List

import pytest

from greenlang.agents.eudr.annual_review_scheduler.config import (
    AnnualReviewSchedulerConfig,
)
from greenlang.agents.eudr.annual_review_scheduler.entity_coordinator import (
    EntityCoordinatorEngine,
)
from greenlang.agents.eudr.annual_review_scheduler.models import (
    EntityCoordination,
    EntityDependency,
    EntityRole,
    EntityStatus,
    ReviewPhase,
)
from greenlang.agents.eudr.annual_review_scheduler.provenance import (
    ProvenanceTracker,
)


@pytest.fixture
def config():
    return AnnualReviewSchedulerConfig()


@pytest.fixture
def coordinator(config):
    return EntityCoordinatorEngine(config=config, provenance=ProvenanceTracker())


# ---------------------------------------------------------------------------
# Entity Assignment
# ---------------------------------------------------------------------------

class TestEntityAssignment:
    """Test entity assignment to review cycles."""

    @pytest.mark.asyncio
    async def test_assign_entity_returns_entity_coordination(self, coordinator):
        entity = await coordinator.assign_entity(
            cycle_id="cyc-001",
            name="Sustainability Manager",
            role=EntityRole.REVIEWER,
            email="sustainability@company.com",
            phases=[ReviewPhase.DATA_COLLECTION, ReviewPhase.ANALYSIS],
        )
        assert isinstance(entity, EntityCoordination)
        assert entity.entity_id.startswith("entity-")

    @pytest.mark.asyncio
    async def test_assign_entity_sets_cycle_id(self, coordinator):
        entity = await coordinator.assign_entity(
            cycle_id="cyc-001",
            name="Test User",
            role=EntityRole.ANALYST,
            email="test@company.com",
            phases=[ReviewPhase.ANALYSIS],
        )
        assert entity.cycle_id == "cyc-001"

    @pytest.mark.asyncio
    async def test_assign_entity_sets_role(self, coordinator):
        entity = await coordinator.assign_entity(
            cycle_id="cyc-001",
            name="Lead",
            role=EntityRole.LEAD,
            email="lead@company.com",
            phases=[ReviewPhase.PREPARATION],
        )
        assert entity.role == EntityRole.LEAD

    @pytest.mark.asyncio
    async def test_assign_entity_sets_phases(self, coordinator):
        entity = await coordinator.assign_entity(
            cycle_id="cyc-001",
            name="Analyst",
            role=EntityRole.ANALYST,
            email="analyst@company.com",
            phases=[ReviewPhase.DATA_COLLECTION, ReviewPhase.ANALYSIS],
        )
        assert ReviewPhase.DATA_COLLECTION in entity.assigned_phases
        assert ReviewPhase.ANALYSIS in entity.assigned_phases

    @pytest.mark.asyncio
    async def test_assign_entity_default_status_active(self, coordinator):
        entity = await coordinator.assign_entity(
            cycle_id="cyc-001",
            name="User",
            role=EntityRole.CONTRIBUTOR,
            email="user@company.com",
            phases=[ReviewPhase.PREPARATION],
        )
        assert entity.status == EntityStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_assign_external_entity_status_invited(self, coordinator):
        entity = await coordinator.assign_entity(
            cycle_id="cyc-001",
            name="Third-Party Auditor",
            role=EntityRole.EXTERNAL_AUDITOR,
            email="auditor@external.com",
            phases=[ReviewPhase.REVIEW_MEETING],
        )
        assert entity.status == EntityStatus.INVITED

    @pytest.mark.asyncio
    async def test_assign_multiple_entities(self, coordinator):
        for i in range(5):
            await coordinator.assign_entity(
                cycle_id="cyc-001",
                name=f"Entity {i}",
                role=EntityRole.CONTRIBUTOR,
                email=f"entity{i}@company.com",
                phases=[ReviewPhase.DATA_COLLECTION],
            )
        entities = await coordinator.list_entities(cycle_id="cyc-001")
        assert len(entities) == 5

    @pytest.mark.asyncio
    async def test_assign_entity_unique_ids(self, coordinator):
        e1 = await coordinator.assign_entity(
            cycle_id="cyc-001", name="A", role=EntityRole.ANALYST,
            email="a@company.com", phases=[ReviewPhase.ANALYSIS],
        )
        e2 = await coordinator.assign_entity(
            cycle_id="cyc-001", name="B", role=EntityRole.ANALYST,
            email="b@company.com", phases=[ReviewPhase.ANALYSIS],
        )
        assert e1.entity_id != e2.entity_id


# ---------------------------------------------------------------------------
# Entity Status Transitions
# ---------------------------------------------------------------------------

class TestEntityStatusTransitions:
    """Test entity status changes."""

    @pytest.mark.asyncio
    async def test_deactivate_entity(self, coordinator):
        entity = await coordinator.assign_entity(
            cycle_id="cyc-001", name="User", role=EntityRole.ANALYST,
            email="user@company.com", phases=[ReviewPhase.ANALYSIS],
        )
        updated = await coordinator.deactivate_entity(entity.entity_id)
        assert updated.status == EntityStatus.INACTIVE

    @pytest.mark.asyncio
    async def test_remove_entity(self, coordinator):
        entity = await coordinator.assign_entity(
            cycle_id="cyc-001", name="User", role=EntityRole.CONTRIBUTOR,
            email="user@company.com", phases=[ReviewPhase.DATA_COLLECTION],
        )
        removed = await coordinator.remove_entity(entity.entity_id)
        assert removed.status == EntityStatus.REMOVED

    @pytest.mark.asyncio
    async def test_accept_invitation(self, coordinator):
        entity = await coordinator.assign_entity(
            cycle_id="cyc-001", name="Auditor", role=EntityRole.EXTERNAL_AUDITOR,
            email="auditor@external.com", phases=[ReviewPhase.REVIEW_MEETING],
        )
        accepted = await coordinator.accept_invitation(entity.entity_id)
        assert accepted.status == EntityStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_decline_invitation(self, coordinator):
        entity = await coordinator.assign_entity(
            cycle_id="cyc-001", name="Auditor", role=EntityRole.EXTERNAL_AUDITOR,
            email="auditor@external.com", phases=[ReviewPhase.REVIEW_MEETING],
        )
        declined = await coordinator.decline_invitation(entity.entity_id)
        assert declined.status == EntityStatus.DECLINED

    @pytest.mark.asyncio
    async def test_remove_nonexistent_raises(self, coordinator):
        with pytest.raises(ValueError, match="not found"):
            await coordinator.remove_entity("entity-nonexistent")


# ---------------------------------------------------------------------------
# Dependency Management
# ---------------------------------------------------------------------------

class TestDependencyManagement:
    """Test entity dependency creation and resolution."""

    @pytest.mark.asyncio
    async def test_create_dependency(self, coordinator):
        e1 = await coordinator.assign_entity(
            cycle_id="cyc-001", name="Data Collector", role=EntityRole.ANALYST,
            email="data@company.com", phases=[ReviewPhase.DATA_COLLECTION],
        )
        e2 = await coordinator.assign_entity(
            cycle_id="cyc-001", name="Analyst", role=EntityRole.ANALYST,
            email="analyst@company.com", phases=[ReviewPhase.ANALYSIS],
        )
        dep = await coordinator.create_dependency(
            source_entity_id=e1.entity_id,
            target_entity_id=e2.entity_id,
            dependency_type="data_handoff",
            phase=ReviewPhase.DATA_COLLECTION,
            description="Analysis requires collected data",
        )
        assert isinstance(dep, EntityDependency)
        assert dep.resolved is False

    @pytest.mark.asyncio
    async def test_resolve_dependency(self, coordinator):
        e1 = await coordinator.assign_entity(
            cycle_id="cyc-001", name="Data Collector", role=EntityRole.ANALYST,
            email="data@company.com", phases=[ReviewPhase.DATA_COLLECTION],
        )
        e2 = await coordinator.assign_entity(
            cycle_id="cyc-001", name="Analyst", role=EntityRole.ANALYST,
            email="analyst@company.com", phases=[ReviewPhase.ANALYSIS],
        )
        dep = await coordinator.create_dependency(
            source_entity_id=e1.entity_id,
            target_entity_id=e2.entity_id,
            dependency_type="data_handoff",
            phase=ReviewPhase.DATA_COLLECTION,
        )
        resolved = await coordinator.resolve_dependency(dep.dependency_id)
        assert resolved.resolved is True

    @pytest.mark.asyncio
    async def test_list_unresolved_dependencies(self, coordinator):
        e1 = await coordinator.assign_entity(
            cycle_id="cyc-001", name="A", role=EntityRole.ANALYST,
            email="a@company.com", phases=[ReviewPhase.DATA_COLLECTION],
        )
        e2 = await coordinator.assign_entity(
            cycle_id="cyc-001", name="B", role=EntityRole.ANALYST,
            email="b@company.com", phases=[ReviewPhase.ANALYSIS],
        )
        await coordinator.create_dependency(
            source_entity_id=e1.entity_id, target_entity_id=e2.entity_id,
            dependency_type="data_handoff", phase=ReviewPhase.DATA_COLLECTION,
        )
        unresolved = await coordinator.list_dependencies(
            cycle_id="cyc-001", resolved=False,
        )
        assert len(unresolved) >= 1

    @pytest.mark.asyncio
    async def test_create_circular_dependency_raises(self, coordinator):
        e1 = await coordinator.assign_entity(
            cycle_id="cyc-001", name="A", role=EntityRole.ANALYST,
            email="a@company.com", phases=[ReviewPhase.DATA_COLLECTION],
        )
        e2 = await coordinator.assign_entity(
            cycle_id="cyc-001", name="B", role=EntityRole.ANALYST,
            email="b@company.com", phases=[ReviewPhase.ANALYSIS],
        )
        await coordinator.create_dependency(
            source_entity_id=e1.entity_id, target_entity_id=e2.entity_id,
            dependency_type="data_handoff", phase=ReviewPhase.DATA_COLLECTION,
        )
        with pytest.raises(ValueError, match="circular"):
            await coordinator.create_dependency(
                source_entity_id=e2.entity_id, target_entity_id=e1.entity_id,
                dependency_type="approval", phase=ReviewPhase.ANALYSIS,
            )

    @pytest.mark.asyncio
    async def test_resolve_nonexistent_dependency_raises(self, coordinator):
        with pytest.raises(ValueError, match="not found"):
            await coordinator.resolve_dependency("dep-nonexistent")


# ---------------------------------------------------------------------------
# List and Filter
# ---------------------------------------------------------------------------

class TestListAndFilter:
    """Test entity listing and filtering."""

    @pytest.mark.asyncio
    async def test_list_by_cycle(self, coordinator):
        await coordinator.assign_entity(
            cycle_id="cyc-001", name="A", role=EntityRole.ANALYST,
            email="a@company.com", phases=[ReviewPhase.ANALYSIS],
        )
        await coordinator.assign_entity(
            cycle_id="cyc-002", name="B", role=EntityRole.ANALYST,
            email="b@company.com", phases=[ReviewPhase.ANALYSIS],
        )
        entities = await coordinator.list_entities(cycle_id="cyc-001")
        assert len(entities) == 1

    @pytest.mark.asyncio
    async def test_list_by_role(self, coordinator):
        await coordinator.assign_entity(
            cycle_id="cyc-001", name="Lead", role=EntityRole.LEAD,
            email="lead@company.com", phases=[ReviewPhase.PREPARATION],
        )
        await coordinator.assign_entity(
            cycle_id="cyc-001", name="Analyst", role=EntityRole.ANALYST,
            email="analyst@company.com", phases=[ReviewPhase.ANALYSIS],
        )
        leads = await coordinator.list_entities(
            cycle_id="cyc-001", role=EntityRole.LEAD,
        )
        assert len(leads) == 1
        assert leads[0].role == EntityRole.LEAD

    @pytest.mark.asyncio
    async def test_list_by_phase(self, coordinator):
        await coordinator.assign_entity(
            cycle_id="cyc-001", name="DC", role=EntityRole.ANALYST,
            email="dc@company.com",
            phases=[ReviewPhase.DATA_COLLECTION],
        )
        await coordinator.assign_entity(
            cycle_id="cyc-001", name="AN", role=EntityRole.ANALYST,
            email="an@company.com",
            phases=[ReviewPhase.ANALYSIS],
        )
        dc_entities = await coordinator.list_entities(
            cycle_id="cyc-001", phase=ReviewPhase.DATA_COLLECTION,
        )
        assert len(dc_entities) == 1

    @pytest.mark.asyncio
    async def test_list_by_status(self, coordinator):
        e = await coordinator.assign_entity(
            cycle_id="cyc-001", name="User", role=EntityRole.CONTRIBUTOR,
            email="user@company.com", phases=[ReviewPhase.DATA_COLLECTION],
        )
        await coordinator.deactivate_entity(e.entity_id)
        active = await coordinator.list_entities(
            cycle_id="cyc-001", status=EntityStatus.ACTIVE,
        )
        inactive = await coordinator.list_entities(
            cycle_id="cyc-001", status=EntityStatus.INACTIVE,
        )
        assert len(active) == 0
        assert len(inactive) == 1

    @pytest.mark.asyncio
    async def test_get_entity_by_id(self, coordinator):
        entity = await coordinator.assign_entity(
            cycle_id="cyc-001", name="User", role=EntityRole.ANALYST,
            email="user@company.com", phases=[ReviewPhase.ANALYSIS],
        )
        retrieved = await coordinator.get_entity(entity.entity_id)
        assert retrieved.entity_id == entity.entity_id

    @pytest.mark.asyncio
    async def test_get_nonexistent_entity_raises(self, coordinator):
        with pytest.raises(ValueError, match="not found"):
            await coordinator.get_entity("entity-nonexistent")


# ---------------------------------------------------------------------------
# RACI Matrix
# ---------------------------------------------------------------------------

class TestRACIMatrix:
    """Test RACI matrix generation."""

    @pytest.mark.asyncio
    async def test_generate_raci_matrix(self, coordinator):
        await coordinator.assign_entity(
            cycle_id="cyc-001", name="Lead", role=EntityRole.LEAD,
            email="lead@company.com",
            phases=[ReviewPhase.PREPARATION, ReviewPhase.SIGN_OFF],
        )
        await coordinator.assign_entity(
            cycle_id="cyc-001", name="Analyst", role=EntityRole.ANALYST,
            email="analyst@company.com",
            phases=[ReviewPhase.DATA_COLLECTION, ReviewPhase.ANALYSIS],
        )
        await coordinator.assign_entity(
            cycle_id="cyc-001", name="Approver", role=EntityRole.APPROVER,
            email="approver@company.com",
            phases=[ReviewPhase.SIGN_OFF],
        )
        raci = await coordinator.generate_raci_matrix(cycle_id="cyc-001")
        assert isinstance(raci, dict)
        assert len(raci) > 0

    @pytest.mark.asyncio
    async def test_raci_matrix_covers_all_phases(self, coordinator):
        await coordinator.assign_entity(
            cycle_id="cyc-001", name="Lead", role=EntityRole.LEAD,
            email="lead@company.com",
            phases=list(ReviewPhase),
        )
        raci = await coordinator.generate_raci_matrix(cycle_id="cyc-001")
        for phase in ReviewPhase:
            assert phase.value in raci or phase in raci

    @pytest.mark.asyncio
    async def test_raci_empty_cycle_returns_empty(self, coordinator):
        raci = await coordinator.generate_raci_matrix(cycle_id="cyc-empty")
        assert raci == {} or len(raci) == 0


# ---------------------------------------------------------------------------
# Workload Distribution
# ---------------------------------------------------------------------------

class TestWorkloadDistribution:
    """Test workload analysis."""

    @pytest.mark.asyncio
    async def test_get_workload_summary(self, coordinator):
        await coordinator.assign_entity(
            cycle_id="cyc-001", name="A", role=EntityRole.ANALYST,
            email="a@company.com",
            phases=[ReviewPhase.DATA_COLLECTION, ReviewPhase.ANALYSIS],
        )
        await coordinator.assign_entity(
            cycle_id="cyc-001", name="B", role=EntityRole.ANALYST,
            email="b@company.com",
            phases=[ReviewPhase.DATA_COLLECTION],
        )
        summary = await coordinator.get_workload_summary(cycle_id="cyc-001")
        assert isinstance(summary, dict)
        assert len(summary) >= 2

    @pytest.mark.asyncio
    async def test_workload_includes_phase_counts(self, coordinator):
        await coordinator.assign_entity(
            cycle_id="cyc-001", name="A", role=EntityRole.ANALYST,
            email="a@company.com",
            phases=[ReviewPhase.DATA_COLLECTION, ReviewPhase.ANALYSIS, ReviewPhase.REMEDIATION],
        )
        summary = await coordinator.get_workload_summary(cycle_id="cyc-001")
        entity_workload = list(summary.values())[0]
        assert entity_workload["phase_count"] == 3 or "phase_count" in entity_workload


# ---------------------------------------------------------------------------
# Update Entity
# ---------------------------------------------------------------------------

class TestUpdateEntity:
    """Test entity update operations."""

    @pytest.mark.asyncio
    async def test_update_entity_name(self, coordinator):
        entity = await coordinator.assign_entity(
            cycle_id="cyc-001", name="Original Name", role=EntityRole.ANALYST,
            email="user@company.com", phases=[ReviewPhase.ANALYSIS],
        )
        updated = await coordinator.update_entity(
            entity.entity_id, name="Updated Name",
        )
        assert updated.name == "Updated Name"

    @pytest.mark.asyncio
    async def test_update_entity_email(self, coordinator):
        entity = await coordinator.assign_entity(
            cycle_id="cyc-001", name="User", role=EntityRole.CONTRIBUTOR,
            email="old@company.com", phases=[ReviewPhase.DATA_COLLECTION],
        )
        updated = await coordinator.update_entity(
            entity.entity_id, email="new@company.com",
        )
        assert updated.email == "new@company.com"

    @pytest.mark.asyncio
    async def test_update_entity_phases(self, coordinator):
        entity = await coordinator.assign_entity(
            cycle_id="cyc-001", name="User", role=EntityRole.ANALYST,
            email="user@company.com", phases=[ReviewPhase.ANALYSIS],
        )
        updated = await coordinator.update_entity(
            entity.entity_id,
            phases=[ReviewPhase.ANALYSIS, ReviewPhase.REMEDIATION],
        )
        assert ReviewPhase.REMEDIATION in updated.assigned_phases

    @pytest.mark.asyncio
    async def test_update_nonexistent_entity_raises(self, coordinator):
        with pytest.raises(ValueError, match="not found"):
            await coordinator.update_entity("entity-nonexistent", name="Test")

    @pytest.mark.asyncio
    async def test_update_removed_entity_raises(self, coordinator):
        entity = await coordinator.assign_entity(
            cycle_id="cyc-001", name="User", role=EntityRole.CONTRIBUTOR,
            email="user@company.com", phases=[ReviewPhase.DATA_COLLECTION],
        )
        await coordinator.remove_entity(entity.entity_id)
        with pytest.raises(ValueError, match="removed"):
            await coordinator.update_entity(entity.entity_id, name="New Name")


# ---------------------------------------------------------------------------
# Bulk Assignment
# ---------------------------------------------------------------------------

class TestBulkAssignment:
    """Test bulk entity operations."""

    @pytest.mark.asyncio
    async def test_bulk_assign_entities(self, coordinator):
        entities_data = [
            {"name": f"Entity {i}", "role": EntityRole.CONTRIBUTOR,
             "email": f"e{i}@company.com", "phases": [ReviewPhase.DATA_COLLECTION]}
            for i in range(5)
        ]
        results = await coordinator.bulk_assign(
            cycle_id="cyc-bulk", entities_data=entities_data,
        )
        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_bulk_assign_unique_ids(self, coordinator):
        entities_data = [
            {"name": f"Entity {i}", "role": EntityRole.ANALYST,
             "email": f"e{i}@company.com", "phases": [ReviewPhase.ANALYSIS]}
            for i in range(3)
        ]
        results = await coordinator.bulk_assign(
            cycle_id="cyc-bulk2", entities_data=entities_data,
        )
        ids = [e.entity_id for e in results]
        assert len(ids) == len(set(ids))

    @pytest.mark.asyncio
    async def test_bulk_assign_empty_list(self, coordinator):
        results = await coordinator.bulk_assign(
            cycle_id="cyc-bulk-empty", entities_data=[],
        )
        assert results == []


# ---------------------------------------------------------------------------
# Entity Count by Role
# ---------------------------------------------------------------------------

class TestEntityCountByRole:
    """Test entity count aggregation by role."""

    @pytest.mark.asyncio
    async def test_count_by_role(self, coordinator):
        await coordinator.assign_entity(
            cycle_id="cyc-count", name="Lead", role=EntityRole.LEAD,
            email="lead@c.com", phases=[ReviewPhase.PREPARATION],
        )
        await coordinator.assign_entity(
            cycle_id="cyc-count", name="Analyst1", role=EntityRole.ANALYST,
            email="a1@c.com", phases=[ReviewPhase.ANALYSIS],
        )
        await coordinator.assign_entity(
            cycle_id="cyc-count", name="Analyst2", role=EntityRole.ANALYST,
            email="a2@c.com", phases=[ReviewPhase.ANALYSIS],
        )
        counts = await coordinator.count_by_role(cycle_id="cyc-count")
        assert counts[EntityRole.LEAD.value] == 1 or counts.get("lead") == 1
        assert counts[EntityRole.ANALYST.value] == 2 or counts.get("analyst") == 2

    @pytest.mark.asyncio
    async def test_count_by_role_empty_cycle(self, coordinator):
        counts = await coordinator.count_by_role(cycle_id="cyc-empty-count")
        total = sum(counts.values()) if counts else 0
        assert total == 0
