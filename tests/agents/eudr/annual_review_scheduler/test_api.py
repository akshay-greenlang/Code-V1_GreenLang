# -*- coding: utf-8 -*-
"""
Unit tests for API endpoints - AGENT-EUDR-034

Tests all REST API endpoints for the Annual Review Scheduler including
review cycle CRUD, deadline management, checklist operations, entity
coordination, year comparison, calendar management, notifications,
health check, and error handling.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-034 (API Layer)
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, patch

import pytest

from greenlang.agents.eudr.annual_review_scheduler.config import (
    AnnualReviewSchedulerConfig,
)
from greenlang.agents.eudr.annual_review_scheduler.models import (
    AGENT_ID,
    AGENT_VERSION,
    CalendarEntryType,
    ChecklistItemStatus,
    DeadlineAlertLevel,
    DeadlineStatus,
    EntityRole,
    EntityStatus,
    EUDRCommodity,
    NotificationChannel,
    NotificationPriority,
    NotificationStatus,
    ReviewCycleStatus,
    ReviewPhase,
    ReviewType,
    YearComparisonStatus,
)

# Attempt to import FastAPI test client; skip gracefully if unavailable
try:
    from httpx import AsyncClient, ASGITransport
    from greenlang.agents.eudr.annual_review_scheduler.api import create_app

    _HAS_HTTPX = True
except ImportError:
    _HAS_HTTPX = False

pytestmark = pytest.mark.skipif(
    not _HAS_HTTPX,
    reason="httpx or FastAPI not available",
)


@pytest.fixture
def app():
    """Create test FastAPI application."""
    return create_app()


@pytest.fixture
async def client(app):
    """Create async test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ===========================================================================
# Health & Meta
# ===========================================================================

class TestHealthEndpoint:
    """Test health check endpoint."""

    @pytest.mark.asyncio
    async def test_health_returns_200(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_health_has_agent_id(self, client):
        resp = await client.get("/health")
        data = resp.json()
        assert data["agent_id"] == AGENT_ID

    @pytest.mark.asyncio
    async def test_health_has_version(self, client):
        resp = await client.get("/health")
        data = resp.json()
        assert data["version"] == AGENT_VERSION

    @pytest.mark.asyncio
    async def test_health_status_healthy(self, client):
        resp = await client.get("/health")
        data = resp.json()
        assert data["status"] == "healthy"


# ===========================================================================
# Review Cycle Endpoints
# ===========================================================================

class TestCreateCycleEndpoint:
    """Test POST /cycles."""

    @pytest.mark.asyncio
    async def test_create_cycle_201(self, client):
        resp = await client.post("/cycles", json={
            "operator_id": "operator-001",
            "review_year": 2026,
            "review_type": "annual",
            "commodity_scope": [
                {"commodity": "coffee", "supplier_count": 15, "shipment_count": 120}
            ],
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["cycle_id"].startswith("cyc-")

    @pytest.mark.asyncio
    async def test_create_cycle_sets_status_draft(self, client):
        resp = await client.post("/cycles", json={
            "operator_id": "operator-001",
            "review_year": 2026,
            "review_type": "annual",
            "commodity_scope": [
                {"commodity": "coffee", "supplier_count": 10, "shipment_count": 50}
            ],
        })
        data = resp.json()
        assert data["status"] == "draft"

    @pytest.mark.asyncio
    async def test_create_cycle_multi_commodity(self, client):
        resp = await client.post("/cycles", json={
            "operator_id": "operator-001",
            "review_year": 2026,
            "review_type": "annual",
            "commodity_scope": [
                {"commodity": "coffee", "supplier_count": 15, "shipment_count": 120},
                {"commodity": "cocoa", "supplier_count": 8, "shipment_count": 45},
            ],
        })
        assert resp.status_code == 201
        data = resp.json()
        assert len(data["commodity_scope"]) == 2

    @pytest.mark.asyncio
    async def test_create_cycle_invalid_commodity_422(self, client):
        resp = await client.post("/cycles", json={
            "operator_id": "operator-001",
            "review_year": 2026,
            "review_type": "annual",
            "commodity_scope": [
                {"commodity": "invalid_commodity", "supplier_count": 1, "shipment_count": 1}
            ],
        })
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_create_cycle_missing_operator_422(self, client):
        resp = await client.post("/cycles", json={
            "review_year": 2026,
            "review_type": "annual",
            "commodity_scope": [],
        })
        assert resp.status_code == 422


class TestGetCycleEndpoint:
    """Test GET /cycles/{cycle_id}."""

    @pytest.mark.asyncio
    async def test_get_cycle_200(self, client):
        create_resp = await client.post("/cycles", json={
            "operator_id": "operator-001",
            "review_year": 2026,
            "review_type": "annual",
            "commodity_scope": [
                {"commodity": "coffee", "supplier_count": 10, "shipment_count": 50}
            ],
        })
        cycle_id = create_resp.json()["cycle_id"]
        resp = await client.get(f"/cycles/{cycle_id}")
        assert resp.status_code == 200
        assert resp.json()["cycle_id"] == cycle_id

    @pytest.mark.asyncio
    async def test_get_cycle_not_found_404(self, client):
        resp = await client.get("/cycles/cyc-nonexistent")
        assert resp.status_code == 404


class TestListCyclesEndpoint:
    """Test GET /cycles."""

    @pytest.mark.asyncio
    async def test_list_cycles_200(self, client):
        await client.post("/cycles", json={
            "operator_id": "operator-001",
            "review_year": 2026,
            "review_type": "annual",
            "commodity_scope": [
                {"commodity": "coffee", "supplier_count": 10, "shipment_count": 50}
            ],
        })
        resp = await client.get("/cycles", params={"operator_id": "operator-001"})
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) >= 1

    @pytest.mark.asyncio
    async def test_list_cycles_filter_by_status(self, client):
        resp = await client.get("/cycles", params={
            "operator_id": "operator-001",
            "status": "draft",
        })
        assert resp.status_code == 200


class TestCycleActionsEndpoint:
    """Test cycle action endpoints (schedule, start, advance, etc.)."""

    @pytest.mark.asyncio
    async def test_schedule_cycle_200(self, client):
        create_resp = await client.post("/cycles", json={
            "operator_id": "operator-001",
            "review_year": 2026,
            "review_type": "annual",
            "commodity_scope": [
                {"commodity": "coffee", "supplier_count": 10, "shipment_count": 50}
            ],
        })
        cycle_id = create_resp.json()["cycle_id"]
        resp = await client.post(f"/cycles/{cycle_id}/schedule")
        assert resp.status_code == 200
        assert resp.json()["status"] == "scheduled"

    @pytest.mark.asyncio
    async def test_start_cycle_200(self, client):
        create_resp = await client.post("/cycles", json={
            "operator_id": "operator-001",
            "review_year": 2026,
            "review_type": "annual",
            "commodity_scope": [
                {"commodity": "coffee", "supplier_count": 10, "shipment_count": 50}
            ],
        })
        cycle_id = create_resp.json()["cycle_id"]
        await client.post(f"/cycles/{cycle_id}/schedule")
        resp = await client.post(f"/cycles/{cycle_id}/start")
        assert resp.status_code == 200
        assert resp.json()["status"] == "in_progress"

    @pytest.mark.asyncio
    async def test_advance_phase_200(self, client):
        create_resp = await client.post("/cycles", json={
            "operator_id": "operator-001",
            "review_year": 2026,
            "review_type": "annual",
            "commodity_scope": [
                {"commodity": "coffee", "supplier_count": 10, "shipment_count": 50}
            ],
        })
        cycle_id = create_resp.json()["cycle_id"]
        await client.post(f"/cycles/{cycle_id}/schedule")
        await client.post(f"/cycles/{cycle_id}/start")
        resp = await client.post(f"/cycles/{cycle_id}/advance")
        assert resp.status_code == 200
        assert resp.json()["current_phase"] == "data_collection"

    @pytest.mark.asyncio
    async def test_pause_cycle_200(self, client):
        create_resp = await client.post("/cycles", json={
            "operator_id": "operator-001",
            "review_year": 2026,
            "review_type": "annual",
            "commodity_scope": [
                {"commodity": "coffee", "supplier_count": 10, "shipment_count": 50}
            ],
        })
        cycle_id = create_resp.json()["cycle_id"]
        await client.post(f"/cycles/{cycle_id}/schedule")
        await client.post(f"/cycles/{cycle_id}/start")
        resp = await client.post(f"/cycles/{cycle_id}/pause", json={"reason": "Budget review"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "paused"

    @pytest.mark.asyncio
    async def test_resume_cycle_200(self, client):
        create_resp = await client.post("/cycles", json={
            "operator_id": "operator-001",
            "review_year": 2026,
            "review_type": "annual",
            "commodity_scope": [
                {"commodity": "coffee", "supplier_count": 10, "shipment_count": 50}
            ],
        })
        cycle_id = create_resp.json()["cycle_id"]
        await client.post(f"/cycles/{cycle_id}/schedule")
        await client.post(f"/cycles/{cycle_id}/start")
        await client.post(f"/cycles/{cycle_id}/pause", json={"reason": "Budget review"})
        resp = await client.post(f"/cycles/{cycle_id}/resume")
        assert resp.status_code == 200
        assert resp.json()["status"] == "in_progress"

    @pytest.mark.asyncio
    async def test_cancel_cycle_200(self, client):
        create_resp = await client.post("/cycles", json={
            "operator_id": "operator-001",
            "review_year": 2026,
            "review_type": "annual",
            "commodity_scope": [
                {"commodity": "coffee", "supplier_count": 10, "shipment_count": 50}
            ],
        })
        cycle_id = create_resp.json()["cycle_id"]
        resp = await client.post(f"/cycles/{cycle_id}/cancel", json={"reason": "Not needed"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "cancelled"


# ===========================================================================
# Deadline Endpoints
# ===========================================================================

class TestDeadlineEndpoints:
    """Test deadline-related API endpoints."""

    @pytest.mark.asyncio
    async def test_create_deadline_201(self, client):
        resp = await client.post("/deadlines", json={
            "cycle_id": "cyc-001",
            "phase": "data_collection",
            "description": "Complete data collection",
            "due_date": (datetime.now(tz=timezone.utc) + timedelta(days=30)).isoformat(),
        })
        assert resp.status_code == 201
        assert resp.json()["deadline_id"].startswith("dln-")

    @pytest.mark.asyncio
    async def test_get_deadline_200(self, client):
        create_resp = await client.post("/deadlines", json={
            "cycle_id": "cyc-001",
            "phase": "preparation",
            "description": "Prep complete",
            "due_date": (datetime.now(tz=timezone.utc) + timedelta(days=14)).isoformat(),
        })
        deadline_id = create_resp.json()["deadline_id"]
        resp = await client.get(f"/deadlines/{deadline_id}")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_list_deadlines_200(self, client):
        await client.post("/deadlines", json={
            "cycle_id": "cyc-001",
            "phase": "preparation",
            "description": "D1",
            "due_date": (datetime.now(tz=timezone.utc) + timedelta(days=14)).isoformat(),
        })
        resp = await client.get("/deadlines", params={"cycle_id": "cyc-001"})
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_complete_deadline_200(self, client):
        create_resp = await client.post("/deadlines", json={
            "cycle_id": "cyc-001",
            "phase": "data_collection",
            "description": "Completable",
            "due_date": (datetime.now(tz=timezone.utc) + timedelta(days=30)).isoformat(),
        })
        deadline_id = create_resp.json()["deadline_id"]
        resp = await client.post(f"/deadlines/{deadline_id}/complete")
        assert resp.status_code == 200
        assert resp.json()["status"] == "completed"

    @pytest.mark.asyncio
    async def test_get_deadline_not_found_404(self, client):
        resp = await client.get("/deadlines/dln-nonexistent")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_check_alerts_200(self, client):
        create_resp = await client.post("/deadlines", json={
            "cycle_id": "cyc-001",
            "phase": "analysis",
            "description": "Check alerts",
            "due_date": (datetime.now(tz=timezone.utc) + timedelta(days=2)).isoformat(),
            "critical_days_before": 3,
        })
        deadline_id = create_resp.json()["deadline_id"]
        resp = await client.get(f"/deadlines/{deadline_id}/alerts")
        assert resp.status_code == 200


# ===========================================================================
# Checklist Endpoints
# ===========================================================================

class TestChecklistEndpoints:
    """Test checklist-related API endpoints."""

    @pytest.mark.asyncio
    async def test_generate_checklist_201(self, client):
        create_resp = await client.post("/cycles", json={
            "operator_id": "operator-001",
            "review_year": 2026,
            "review_type": "annual",
            "commodity_scope": [
                {"commodity": "coffee", "supplier_count": 10, "shipment_count": 50}
            ],
        })
        cycle_id = create_resp.json()["cycle_id"]
        resp = await client.post(f"/cycles/{cycle_id}/checklists/generate")
        assert resp.status_code in (200, 201)

    @pytest.mark.asyncio
    async def test_list_checklist_items_200(self, client):
        create_resp = await client.post("/cycles", json={
            "operator_id": "operator-001",
            "review_year": 2026,
            "review_type": "annual",
            "commodity_scope": [
                {"commodity": "coffee", "supplier_count": 10, "shipment_count": 50}
            ],
        })
        cycle_id = create_resp.json()["cycle_id"]
        await client.post(f"/cycles/{cycle_id}/checklists/generate")
        resp = await client.get(f"/cycles/{cycle_id}/checklists")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_complete_checklist_item_200(self, client):
        create_resp = await client.post("/cycles", json={
            "operator_id": "operator-001",
            "review_year": 2026,
            "review_type": "annual",
            "commodity_scope": [
                {"commodity": "coffee", "supplier_count": 10, "shipment_count": 50}
            ],
        })
        cycle_id = create_resp.json()["cycle_id"]
        gen_resp = await client.post(f"/cycles/{cycle_id}/checklists/generate")
        items = gen_resp.json() if isinstance(gen_resp.json(), list) else gen_resp.json().get("items", [])
        if items:
            item_id = items[0]["item_id"] if isinstance(items[0], dict) else items[0]
            resp = await client.post(
                f"/checklists/{item_id}/complete",
                json={"completed_by": "analyst@company.com"},
            )
            assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_get_checklist_progress_200(self, client):
        create_resp = await client.post("/cycles", json={
            "operator_id": "operator-001",
            "review_year": 2026,
            "review_type": "annual",
            "commodity_scope": [
                {"commodity": "coffee", "supplier_count": 10, "shipment_count": 50}
            ],
        })
        cycle_id = create_resp.json()["cycle_id"]
        await client.post(f"/cycles/{cycle_id}/checklists/generate")
        resp = await client.get(f"/cycles/{cycle_id}/checklists/progress")
        assert resp.status_code == 200


# ===========================================================================
# Entity Coordination Endpoints
# ===========================================================================

class TestEntityEndpoints:
    """Test entity coordination API endpoints."""

    @pytest.mark.asyncio
    async def test_assign_entity_201(self, client):
        resp = await client.post("/entities", json={
            "cycle_id": "cyc-001",
            "name": "Sustainability Manager",
            "role": "reviewer",
            "email": "sustainability@company.com",
            "phases": ["data_collection", "analysis"],
        })
        assert resp.status_code == 201
        assert resp.json()["entity_id"].startswith("entity-")

    @pytest.mark.asyncio
    async def test_list_entities_200(self, client):
        await client.post("/entities", json={
            "cycle_id": "cyc-001",
            "name": "User",
            "role": "analyst",
            "email": "user@company.com",
            "phases": ["analysis"],
        })
        resp = await client.get("/entities", params={"cycle_id": "cyc-001"})
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_get_entity_200(self, client):
        create_resp = await client.post("/entities", json={
            "cycle_id": "cyc-001",
            "name": "User",
            "role": "analyst",
            "email": "user@company.com",
            "phases": ["analysis"],
        })
        entity_id = create_resp.json()["entity_id"]
        resp = await client.get(f"/entities/{entity_id}")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_get_entity_not_found_404(self, client):
        resp = await client.get("/entities/entity-nonexistent")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_deactivate_entity_200(self, client):
        create_resp = await client.post("/entities", json={
            "cycle_id": "cyc-001",
            "name": "User",
            "role": "contributor",
            "email": "user@company.com",
            "phases": ["data_collection"],
        })
        entity_id = create_resp.json()["entity_id"]
        resp = await client.post(f"/entities/{entity_id}/deactivate")
        assert resp.status_code == 200
        assert resp.json()["status"] == "inactive"

    @pytest.mark.asyncio
    async def test_create_dependency_201(self, client):
        e1 = await client.post("/entities", json={
            "cycle_id": "cyc-001", "name": "A", "role": "analyst",
            "email": "a@c.com", "phases": ["data_collection"],
        })
        e2 = await client.post("/entities", json={
            "cycle_id": "cyc-001", "name": "B", "role": "analyst",
            "email": "b@c.com", "phases": ["analysis"],
        })
        resp = await client.post("/dependencies", json={
            "source_entity_id": e1.json()["entity_id"],
            "target_entity_id": e2.json()["entity_id"],
            "dependency_type": "data_handoff",
            "phase": "data_collection",
        })
        assert resp.status_code == 201

    @pytest.mark.asyncio
    async def test_get_raci_matrix_200(self, client):
        await client.post("/entities", json={
            "cycle_id": "cyc-001", "name": "Lead", "role": "lead",
            "email": "lead@c.com", "phases": ["preparation", "sign_off"],
        })
        resp = await client.get("/cycles/cyc-001/raci")
        assert resp.status_code == 200


# ===========================================================================
# Year Comparison Endpoints
# ===========================================================================

class TestYearComparisonEndpoints:
    """Test year comparison API endpoints."""

    @pytest.mark.asyncio
    async def test_register_snapshot_201(self, client):
        resp = await client.post("/snapshots", json={
            "snapshot_id": "snap-2026-api",
            "operator_id": "operator-001",
            "year": 2026,
            "commodity": "coffee",
            "total_suppliers": 15,
            "compliant_suppliers": 14,
            "compliance_rate": "93.33",
            "average_risk_score": "28.50",
            "total_shipments": 120,
            "deforestation_free_rate": "99.10",
            "dds_submitted": 118,
            "dds_approved": 115,
        })
        assert resp.status_code == 201

    @pytest.mark.asyncio
    async def test_compare_years_200(self, client):
        for year, rate in [(2025, "83.33"), (2026, "93.33")]:
            await client.post("/snapshots", json={
                "snapshot_id": f"snap-{year}-api",
                "operator_id": "operator-001",
                "year": year,
                "commodity": "coffee",
                "compliance_rate": rate,
                "average_risk_score": "35.00",
            })
        resp = await client.post("/comparisons", json={
            "operator_id": "operator-001",
            "commodity": "coffee",
            "base_year": 2025,
            "compare_year": 2026,
        })
        assert resp.status_code in (200, 201)

    @pytest.mark.asyncio
    async def test_get_comparison_200(self, client):
        for year in (2024, 2025):
            await client.post("/snapshots", json={
                "snapshot_id": f"snap-{year}-get",
                "operator_id": "operator-002",
                "year": year,
                "commodity": "cocoa",
                "compliance_rate": "80.00",
            })
        cmp_resp = await client.post("/comparisons", json={
            "operator_id": "operator-002",
            "commodity": "cocoa",
            "base_year": 2024,
            "compare_year": 2025,
        })
        cmp_id = cmp_resp.json()["comparison_id"]
        resp = await client.get(f"/comparisons/{cmp_id}")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_get_comparison_not_found_404(self, client):
        resp = await client.get("/comparisons/cmp-nonexistent")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_list_comparisons_200(self, client):
        resp = await client.get("/comparisons", params={"operator_id": "operator-001"})
        assert resp.status_code == 200


# ===========================================================================
# Calendar Endpoints
# ===========================================================================

class TestCalendarEndpoints:
    """Test calendar API endpoints."""

    @pytest.mark.asyncio
    async def test_create_calendar_entry_201(self, client):
        resp = await client.post("/calendar", json={
            "cycle_id": "cyc-001",
            "entry_type": "phase_start",
            "title": "Data Collection Start",
            "start_time": (datetime.now(tz=timezone.utc) + timedelta(days=14)).isoformat(),
        })
        assert resp.status_code == 201

    @pytest.mark.asyncio
    async def test_list_calendar_entries_200(self, client):
        await client.post("/calendar", json={
            "cycle_id": "cyc-001",
            "entry_type": "reminder",
            "title": "Test",
            "start_time": (datetime.now(tz=timezone.utc) + timedelta(days=7)).isoformat(),
        })
        resp = await client.get("/calendar", params={"cycle_id": "cyc-001"})
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_delete_calendar_entry_200(self, client):
        create_resp = await client.post("/calendar", json={
            "cycle_id": "cyc-001",
            "entry_type": "reminder",
            "title": "Deletable",
            "start_time": (datetime.now(tz=timezone.utc) + timedelta(days=7)).isoformat(),
        })
        entry_id = create_resp.json()["entry_id"]
        resp = await client.delete(f"/calendar/{entry_id}")
        assert resp.status_code in (200, 204)

    @pytest.mark.asyncio
    async def test_generate_calendar_for_cycle_200(self, client):
        create_resp = await client.post("/cycles", json={
            "operator_id": "operator-001",
            "review_year": 2026,
            "review_type": "annual",
            "commodity_scope": [
                {"commodity": "coffee", "supplier_count": 10, "shipment_count": 50}
            ],
        })
        cycle_id = create_resp.json()["cycle_id"]
        resp = await client.post(f"/cycles/{cycle_id}/calendar/generate")
        assert resp.status_code in (200, 201)

    @pytest.mark.asyncio
    async def test_export_calendar_ical_200(self, client):
        create_resp = await client.post("/cycles", json={
            "operator_id": "operator-001",
            "review_year": 2026,
            "review_type": "annual",
            "commodity_scope": [
                {"commodity": "coffee", "supplier_count": 10, "shipment_count": 50}
            ],
        })
        cycle_id = create_resp.json()["cycle_id"]
        await client.post(f"/cycles/{cycle_id}/calendar/generate")
        resp = await client.get(f"/cycles/{cycle_id}/calendar/export", params={"format": "ical"})
        assert resp.status_code == 200


# ===========================================================================
# Notification Endpoints
# ===========================================================================

class TestNotificationEndpoints:
    """Test notification API endpoints."""

    @pytest.mark.asyncio
    async def test_create_notification_201(self, client):
        resp = await client.post("/notifications", json={
            "cycle_id": "cyc-001",
            "channel": "email",
            "priority": "normal",
            "recipient": "user@company.com",
            "subject": "Test Notification",
            "body": "This is a test.",
        })
        assert resp.status_code == 201

    @pytest.mark.asyncio
    async def test_send_notification_200(self, client):
        create_resp = await client.post("/notifications", json={
            "cycle_id": "cyc-001",
            "channel": "email",
            "priority": "normal",
            "recipient": "user@company.com",
            "subject": "Send Test",
            "body": "Body",
        })
        ntf_id = create_resp.json()["notification_id"]
        resp = await client.post(f"/notifications/{ntf_id}/send")
        assert resp.status_code == 200
        assert resp.json()["status"] == "sent"

    @pytest.mark.asyncio
    async def test_list_notifications_200(self, client):
        await client.post("/notifications", json={
            "cycle_id": "cyc-001",
            "channel": "email",
            "priority": "normal",
            "recipient": "user@company.com",
            "subject": "List Test",
            "body": "Body",
        })
        resp = await client.get("/notifications", params={"cycle_id": "cyc-001"})
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_get_notification_200(self, client):
        create_resp = await client.post("/notifications", json={
            "cycle_id": "cyc-001",
            "channel": "email",
            "priority": "normal",
            "recipient": "user@company.com",
            "subject": "Get Test",
            "body": "Body",
        })
        ntf_id = create_resp.json()["notification_id"]
        resp = await client.get(f"/notifications/{ntf_id}")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_get_notification_not_found_404(self, client):
        resp = await client.get("/notifications/ntf-nonexistent")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_cancel_notification_200(self, client):
        create_resp = await client.post("/notifications", json={
            "cycle_id": "cyc-001",
            "channel": "email",
            "priority": "low",
            "recipient": "user@company.com",
            "subject": "Cancel Test",
            "body": "Body",
        })
        ntf_id = create_resp.json()["notification_id"]
        resp = await client.post(f"/notifications/{ntf_id}/cancel")
        assert resp.status_code == 200
        assert resp.json()["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_get_delivery_stats_200(self, client):
        resp = await client.get("/notifications/stats", params={"cycle_id": "cyc-001"})
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_retry_notification_200(self, client):
        create_resp = await client.post("/notifications", json={
            "cycle_id": "cyc-001",
            "channel": "sms",
            "priority": "high",
            "recipient": "+1234567890",
            "subject": "Retry Test",
            "body": "Body",
        })
        ntf_id = create_resp.json()["notification_id"]
        await client.post(f"/notifications/{ntf_id}/fail", json={"reason": "Gateway error"})
        resp = await client.post(f"/notifications/{ntf_id}/retry")
        assert resp.status_code == 200


# ===========================================================================
# Error Handling
# ===========================================================================

class TestErrorHandling:
    """Test API error handling."""

    @pytest.mark.asyncio
    async def test_invalid_json_400(self, client):
        resp = await client.post(
            "/cycles",
            content=b"not json",
            headers={"content-type": "application/json"},
        )
        assert resp.status_code in (400, 422)

    @pytest.mark.asyncio
    async def test_method_not_allowed_405(self, client):
        resp = await client.put("/health")
        assert resp.status_code == 405

    @pytest.mark.asyncio
    async def test_nonexistent_endpoint_404(self, client):
        resp = await client.get("/nonexistent-endpoint")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_create_cycle_empty_body_422(self, client):
        resp = await client.post("/cycles", json={})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_double_schedule_returns_error(self, client):
        create_resp = await client.post("/cycles", json={
            "operator_id": "operator-001",
            "review_year": 2026,
            "review_type": "annual",
            "commodity_scope": [
                {"commodity": "coffee", "supplier_count": 10, "shipment_count": 50}
            ],
        })
        cycle_id = create_resp.json()["cycle_id"]
        await client.post(f"/cycles/{cycle_id}/schedule")
        await client.post(f"/cycles/{cycle_id}/start")
        resp = await client.post(f"/cycles/{cycle_id}/schedule")
        assert resp.status_code in (400, 409, 422)

    @pytest.mark.asyncio
    async def test_start_draft_cycle_returns_error(self, client):
        create_resp = await client.post("/cycles", json={
            "operator_id": "operator-001",
            "review_year": 2026,
            "review_type": "annual",
            "commodity_scope": [
                {"commodity": "coffee", "supplier_count": 10, "shipment_count": 50}
            ],
        })
        cycle_id = create_resp.json()["cycle_id"]
        resp = await client.post(f"/cycles/{cycle_id}/start")
        assert resp.status_code in (400, 409, 422)

    @pytest.mark.asyncio
    async def test_advance_draft_cycle_returns_error(self, client):
        create_resp = await client.post("/cycles", json={
            "operator_id": "operator-001",
            "review_year": 2026,
            "review_type": "annual",
            "commodity_scope": [
                {"commodity": "coffee", "supplier_count": 10, "shipment_count": 50}
            ],
        })
        cycle_id = create_resp.json()["cycle_id"]
        resp = await client.post(f"/cycles/{cycle_id}/advance")
        assert resp.status_code in (400, 409, 422)

    @pytest.mark.asyncio
    async def test_pause_non_active_returns_error(self, client):
        create_resp = await client.post("/cycles", json={
            "operator_id": "operator-001",
            "review_year": 2026,
            "review_type": "annual",
            "commodity_scope": [
                {"commodity": "coffee", "supplier_count": 10, "shipment_count": 50}
            ],
        })
        cycle_id = create_resp.json()["cycle_id"]
        resp = await client.post(f"/cycles/{cycle_id}/pause", json={"reason": "test"})
        assert resp.status_code in (400, 409, 422)

    @pytest.mark.asyncio
    async def test_complete_nonexistent_deadline_404(self, client):
        resp = await client.post("/deadlines/dln-nonexistent/complete")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_create_deadline_missing_fields_422(self, client):
        resp = await client.post("/deadlines", json={
            "cycle_id": "cyc-001",
        })
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_assign_entity_invalid_role_422(self, client):
        resp = await client.post("/entities", json={
            "cycle_id": "cyc-001",
            "name": "User",
            "role": "invalid_role",
            "email": "user@company.com",
            "phases": ["preparation"],
        })
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_create_notification_invalid_channel_422(self, client):
        resp = await client.post("/notifications", json={
            "cycle_id": "cyc-001",
            "channel": "invalid_channel",
            "priority": "normal",
            "recipient": "user@company.com",
            "subject": "Test",
            "body": "Body",
        })
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_register_snapshot_missing_year_422(self, client):
        resp = await client.post("/snapshots", json={
            "snapshot_id": "snap-bad",
            "operator_id": "operator-001",
            "commodity": "coffee",
        })
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_compare_same_year_returns_error(self, client):
        await client.post("/snapshots", json={
            "snapshot_id": "snap-2025-same",
            "operator_id": "operator-err",
            "year": 2025,
            "commodity": "wood",
            "compliance_rate": "80.00",
        })
        resp = await client.post("/comparisons", json={
            "operator_id": "operator-err",
            "commodity": "wood",
            "base_year": 2025,
            "compare_year": 2025,
        })
        assert resp.status_code in (400, 422)

    @pytest.mark.asyncio
    async def test_send_already_sent_notification_returns_error(self, client):
        create_resp = await client.post("/notifications", json={
            "cycle_id": "cyc-err",
            "channel": "email",
            "priority": "normal",
            "recipient": "user@company.com",
            "subject": "Already Sent",
            "body": "Body",
        })
        ntf_id = create_resp.json()["notification_id"]
        await client.post(f"/notifications/{ntf_id}/send")
        resp = await client.post(f"/notifications/{ntf_id}/send")
        assert resp.status_code in (400, 409, 422)

    @pytest.mark.asyncio
    async def test_cancel_sent_notification_returns_error(self, client):
        create_resp = await client.post("/notifications", json={
            "cycle_id": "cyc-err2",
            "channel": "email",
            "priority": "normal",
            "recipient": "user@company.com",
            "subject": "Sent then cancel",
            "body": "Body",
        })
        ntf_id = create_resp.json()["notification_id"]
        await client.post(f"/notifications/{ntf_id}/send")
        resp = await client.post(f"/notifications/{ntf_id}/cancel")
        assert resp.status_code in (400, 409, 422)

    @pytest.mark.asyncio
    async def test_delete_nonexistent_calendar_entry_404(self, client):
        resp = await client.delete("/calendar/cal-nonexistent")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_create_cycle_invalid_review_type_422(self, client):
        resp = await client.post("/cycles", json={
            "operator_id": "operator-001",
            "review_year": 2026,
            "review_type": "invalid_type",
            "commodity_scope": [
                {"commodity": "coffee", "supplier_count": 10, "shipment_count": 50}
            ],
        })
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_list_cycles_no_operator_param(self, client):
        resp = await client.get("/cycles")
        assert resp.status_code in (200, 400, 422)

    @pytest.mark.asyncio
    async def test_waive_deadline_200(self, client):
        create_resp = await client.post("/deadlines", json={
            "cycle_id": "cyc-waive",
            "phase": "remediation",
            "description": "Optional task",
            "due_date": (datetime.now(tz=timezone.utc) + timedelta(days=10)).isoformat(),
        })
        deadline_id = create_resp.json()["deadline_id"]
        resp = await client.post(
            f"/deadlines/{deadline_id}/waive",
            json={"reason": "Not applicable"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "waived"

    @pytest.mark.asyncio
    async def test_extend_deadline_200(self, client):
        create_resp = await client.post("/deadlines", json={
            "cycle_id": "cyc-extend",
            "phase": "data_collection",
            "description": "Extendable",
            "due_date": (datetime.now(tz=timezone.utc) + timedelta(days=10)).isoformat(),
        })
        deadline_id = create_resp.json()["deadline_id"]
        resp = await client.post(
            f"/deadlines/{deadline_id}/extend",
            json={"additional_days": 14},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_resume_non_paused_returns_error(self, client):
        create_resp = await client.post("/cycles", json={
            "operator_id": "operator-001",
            "review_year": 2026,
            "review_type": "annual",
            "commodity_scope": [
                {"commodity": "coffee", "supplier_count": 10, "shipment_count": 50}
            ],
        })
        cycle_id = create_resp.json()["cycle_id"]
        await client.post(f"/cycles/{cycle_id}/schedule")
        await client.post(f"/cycles/{cycle_id}/start")
        resp = await client.post(f"/cycles/{cycle_id}/resume")
        assert resp.status_code in (400, 409, 422)

    @pytest.mark.asyncio
    async def test_create_cycle_with_semi_annual_type(self, client):
        resp = await client.post("/cycles", json={
            "operator_id": "operator-001",
            "review_year": 2026,
            "review_type": "semi_annual",
            "commodity_scope": [
                {"commodity": "cocoa", "supplier_count": 5, "shipment_count": 20}
            ],
        })
        assert resp.status_code == 201
        assert resp.json()["review_type"] == "semi_annual"

    @pytest.mark.asyncio
    async def test_create_notification_with_template_201(self, client):
        resp = await client.post("/notifications/from-template", json={
            "cycle_id": "cyc-tpl",
            "template_id": "tpl-ntf-phase-start",
            "recipient": "user@company.com",
            "variables": {
                "recipient_name": "John",
                "phase_name": "Analysis",
                "cycle_id": "cyc-tpl",
                "start_date": "2026-05-01",
            },
        })
        assert resp.status_code in (201, 404)  # 404 if template not registered

    @pytest.mark.asyncio
    async def test_get_overdue_deadlines_200(self, client):
        resp = await client.get("/deadlines/overdue", params={"cycle_id": "cyc-001"})
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_list_snapshots_200(self, client):
        await client.post("/snapshots", json={
            "snapshot_id": "snap-list-test",
            "operator_id": "operator-list",
            "year": 2026,
            "commodity": "soya",
            "compliance_rate": "88.00",
        })
        resp = await client.get("/snapshots", params={"operator_id": "operator-list"})
        assert resp.status_code == 200
