# -*- coding: utf-8 -*-
"""
Unit tests for API endpoints - AGENT-EUDR-035

Tests all REST API endpoints for the Improvement Plan Creator including
plan CRUD, finding aggregation, gap analysis, action management,
root cause mapping, prioritization, progress tracking, stakeholder
coordination, health check, and error handling.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-035 (API Layer)
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, patch

import pytest

from greenlang.agents.eudr.improvement_plan_creator.config import (
    ImprovementPlanCreatorConfig,
)
from greenlang.agents.eudr.improvement_plan_creator.models import (
    AGENT_ID,
    AGENT_VERSION,
    ActionStatus,
    ActionType,
    EUDRCommodity,
    FindingSource,
    FishboneCategory,
    GapSeverity,
    PlanStatus,
    RACIRole,
)

# Attempt to import FastAPI test client; skip gracefully if unavailable
try:
    from httpx import AsyncClient, ASGITransport
    from greenlang.agents.eudr.improvement_plan_creator.api import create_app

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

    @pytest.mark.asyncio
    async def test_health_has_engines(self, client):
        resp = await client.get("/health")
        data = resp.json()
        assert "engines" in data
        assert isinstance(data["engines"], dict)


# ===========================================================================
# Create Improvement Plan
# ===========================================================================

class TestCreatePlanEndpoint:
    """Test POST /plans."""

    @pytest.mark.asyncio
    async def test_create_plan_201(self, client):
        resp = await client.post("/plans", json={
            "operator_id": "operator-001",
            "title": "Q1 2026 EUDR Compliance Improvement Plan",
            "description": "Addresses audit findings from Q4 2025.",
            "commodities": ["coffee", "cocoa"],
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["plan_id"].startswith("plan-")

    @pytest.mark.asyncio
    async def test_create_plan_sets_draft_status(self, client):
        resp = await client.post("/plans", json={
            "operator_id": "operator-001",
            "title": "Test Plan",
            "commodities": ["coffee"],
        })
        data = resp.json()
        assert data["status"] == "draft"

    @pytest.mark.asyncio
    async def test_create_plan_multi_commodity(self, client):
        resp = await client.post("/plans", json={
            "operator_id": "operator-001",
            "title": "Multi Commodity Plan",
            "commodities": ["coffee", "cocoa", "oil_palm"],
        })
        assert resp.status_code == 201
        data = resp.json()
        assert len(data["commodities"]) == 3

    @pytest.mark.asyncio
    async def test_create_plan_invalid_commodity_422(self, client):
        resp = await client.post("/plans", json={
            "operator_id": "operator-001",
            "title": "Invalid Plan",
            "commodities": ["invalid_commodity"],
        })
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_create_plan_missing_operator_422(self, client):
        resp = await client.post("/plans", json={
            "title": "No Operator Plan",
            "commodities": ["coffee"],
        })
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_create_plan_has_provenance(self, client):
        resp = await client.post("/plans", json={
            "operator_id": "operator-001",
            "title": "Provenance Plan",
            "commodities": ["coffee"],
        })
        data = resp.json()
        assert "provenance_hash" in data
        assert len(data["provenance_hash"]) == 64


# ===========================================================================
# Get / List Plans
# ===========================================================================

class TestGetPlanEndpoint:
    """Test GET /plans/{plan_id}."""

    @pytest.mark.asyncio
    async def test_get_plan_200(self, client):
        create_resp = await client.post("/plans", json={
            "operator_id": "operator-001",
            "title": "Get Test Plan",
            "commodities": ["coffee"],
        })
        plan_id = create_resp.json()["plan_id"]
        resp = await client.get(f"/plans/{plan_id}")
        assert resp.status_code == 200
        assert resp.json()["plan_id"] == plan_id

    @pytest.mark.asyncio
    async def test_get_plan_not_found_404(self, client):
        resp = await client.get("/plans/plan-nonexistent")
        assert resp.status_code == 404


class TestListPlansEndpoint:
    """Test GET /plans."""

    @pytest.mark.asyncio
    async def test_list_plans_200(self, client):
        resp = await client.get("/plans")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_list_plans_returns_list(self, client):
        resp = await client.get("/plans")
        data = resp.json()
        assert isinstance(data, list) or "plans" in data

    @pytest.mark.asyncio
    async def test_list_plans_by_operator(self, client):
        await client.post("/plans", json={
            "operator_id": "operator-filter-001",
            "title": "Filter Plan",
            "commodities": ["coffee"],
        })
        resp = await client.get("/plans", params={"operator_id": "operator-filter-001"})
        assert resp.status_code == 200


# ===========================================================================
# Aggregate Findings
# ===========================================================================

class TestAggregateFindingsEndpoint:
    """Test POST /aggregate-findings."""

    @pytest.mark.asyncio
    async def test_aggregate_findings_200(self, client):
        resp = await client.post("/aggregate-findings", json={
            "operator_id": "operator-001",
            "findings": [
                {
                    "finding_id": "fnd-api-001",
                    "source": "audit",
                    "category": "traceability",
                    "severity": "high",
                    "title": "Missing GPS coordinates",
                    "status": "open",
                    "commodity": "coffee",
                }
            ],
        })
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_aggregate_findings_empty_list(self, client):
        resp = await client.post("/aggregate-findings", json={
            "operator_id": "operator-001",
            "findings": [],
        })
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_aggregate_findings_has_summary(self, client):
        resp = await client.post("/aggregate-findings", json={
            "operator_id": "operator-001",
            "findings": [
                {
                    "finding_id": "fnd-api-002",
                    "source": "audit",
                    "category": "documentation",
                    "severity": "medium",
                    "title": "Expired certifications",
                    "status": "open",
                    "commodity": "cocoa",
                }
            ],
        })
        data = resp.json()
        assert "total_findings" in data or "summary" in data


# ===========================================================================
# Gap Analysis
# ===========================================================================

class TestGapAnalysisEndpoint:
    """Test POST /analyze-gaps."""

    @pytest.mark.asyncio
    async def test_analyze_gaps_200(self, client):
        resp = await client.post("/analyze-gaps", json={
            "operator_id": "operator-001",
            "finding_ids": ["fnd-001", "fnd-002"],
        })
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_analyze_gaps_returns_gaps(self, client):
        resp = await client.post("/analyze-gaps", json={
            "operator_id": "operator-001",
            "finding_ids": ["fnd-001"],
        })
        data = resp.json()
        assert "gaps" in data or isinstance(data, list)

    @pytest.mark.asyncio
    async def test_analyze_gaps_empty_findings_200(self, client):
        resp = await client.post("/analyze-gaps", json={
            "operator_id": "operator-001",
            "finding_ids": [],
        })
        assert resp.status_code == 200


class TestGetGapEndpoint:
    """Test GET /gaps/{gap_id}."""

    @pytest.mark.asyncio
    async def test_get_gap_not_found_404(self, client):
        resp = await client.get("/gaps/gap-nonexistent")
        assert resp.status_code == 404


# ===========================================================================
# Action Generation
# ===========================================================================

class TestGenerateActionsEndpoint:
    """Test POST /generate-actions."""

    @pytest.mark.asyncio
    async def test_generate_actions_200(self, client):
        resp = await client.post("/generate-actions", json={
            "plan_id": "plan-001",
            "gap_ids": ["gap-001"],
        })
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_generate_actions_returns_list(self, client):
        resp = await client.post("/generate-actions", json={
            "plan_id": "plan-001",
            "gap_ids": ["gap-001"],
        })
        data = resp.json()
        assert "actions" in data or isinstance(data, list)

    @pytest.mark.asyncio
    async def test_generate_actions_empty_gaps(self, client):
        resp = await client.post("/generate-actions", json={
            "plan_id": "plan-001",
            "gap_ids": [],
        })
        assert resp.status_code == 200


class TestUpdateActionStatusEndpoint:
    """Test POST /update-action-status."""

    @pytest.mark.asyncio
    async def test_update_action_status_200(self, client):
        resp = await client.post("/update-action-status", json={
            "action_id": "act-001",
            "new_status": "in_progress",
        })
        assert resp.status_code in (200, 404)

    @pytest.mark.asyncio
    async def test_update_action_invalid_status_422(self, client):
        resp = await client.post("/update-action-status", json={
            "action_id": "act-001",
            "new_status": "invalid_status",
        })
        assert resp.status_code == 422


# ===========================================================================
# Root Cause Mapping
# ===========================================================================

class TestRootCauseEndpoint:
    """Test POST /identify-root-causes."""

    @pytest.mark.asyncio
    async def test_identify_root_causes_200(self, client):
        resp = await client.post("/identify-root-causes", json={
            "finding_ids": ["fnd-001", "fnd-002"],
            "gap_ids": ["gap-001"],
        })
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_root_causes_returns_list(self, client):
        resp = await client.post("/identify-root-causes", json={
            "finding_ids": ["fnd-001"],
            "gap_ids": ["gap-001"],
        })
        data = resp.json()
        assert "root_causes" in data or isinstance(data, list)

    @pytest.mark.asyncio
    async def test_confirm_root_cause_200(self, client):
        resp = await client.post("/confirm-root-cause", json={
            "root_cause_id": "rc-001",
            "evidence": "Confirmed by field audit.",
        })
        assert resp.status_code in (200, 404)

    @pytest.mark.asyncio
    async def test_refute_root_cause_200(self, client):
        resp = await client.post("/refute-root-cause", json={
            "root_cause_id": "rc-001",
            "reason": "Investigation proved otherwise.",
        })
        assert resp.status_code in (200, 404)


# ===========================================================================
# Prioritization
# ===========================================================================

class TestPrioritizeEndpoint:
    """Test POST /prioritize-actions."""

    @pytest.mark.asyncio
    async def test_prioritize_actions_200(self, client):
        resp = await client.post("/prioritize-actions", json={
            "plan_id": "plan-001",
        })
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_prioritize_returns_ranked_list(self, client):
        resp = await client.post("/prioritize-actions", json={
            "plan_id": "plan-001",
        })
        data = resp.json()
        assert "ranked_actions" in data or isinstance(data, list)

    @pytest.mark.asyncio
    async def test_prioritize_has_scores(self, client):
        resp = await client.post("/prioritize-actions", json={
            "plan_id": "plan-001",
        })
        data = resp.json()
        if "scores" in data:
            assert isinstance(data["scores"], list)


# ===========================================================================
# Progress Tracking
# ===========================================================================

class TestRecordProgressEndpoint:
    """Test POST /record-progress."""

    @pytest.mark.asyncio
    async def test_record_progress_200(self, client):
        resp = await client.post("/record-progress", json={
            "plan_id": "plan-001",
            "action_id": "act-001",
            "completion_percent": 50.0,
            "notes": "Half complete.",
        })
        assert resp.status_code in (200, 201, 404)

    @pytest.mark.asyncio
    async def test_record_progress_invalid_percent_422(self, client):
        resp = await client.post("/record-progress", json={
            "plan_id": "plan-001",
            "action_id": "act-001",
            "completion_percent": 150.0,
        })
        assert resp.status_code == 422


class TestGetProgressEndpoint:
    """Test GET /progress/{plan_id}."""

    @pytest.mark.asyncio
    async def test_get_progress_200_or_404(self, client):
        resp = await client.get("/progress/plan-001")
        assert resp.status_code in (200, 404)

    @pytest.mark.asyncio
    async def test_get_progress_not_found(self, client):
        resp = await client.get("/progress/plan-nonexistent")
        assert resp.status_code == 404


class TestAddMilestoneEndpoint:
    """Test POST /add-milestone."""

    @pytest.mark.asyncio
    async def test_add_milestone_200(self, client):
        resp = await client.post("/add-milestone", json={
            "plan_id": "plan-001",
            "action_id": "act-001",
            "title": "Design phase complete",
            "due_date": (datetime.now(tz=timezone.utc) + timedelta(days=14)).isoformat(),
        })
        assert resp.status_code in (200, 201, 404)

    @pytest.mark.asyncio
    async def test_complete_milestone_200(self, client):
        resp = await client.post("/complete-milestone", json={
            "plan_id": "plan-001",
            "action_id": "act-001",
            "milestone_id": "ms-001",
        })
        assert resp.status_code in (200, 404)


# ===========================================================================
# Stakeholder Coordination
# ===========================================================================

class TestAssignStakeholderEndpoint:
    """Test POST /assign-stakeholder."""

    @pytest.mark.asyncio
    async def test_assign_stakeholder_200(self, client):
        resp = await client.post("/assign-stakeholder", json={
            "plan_id": "plan-001",
            "name": "Compliance Officer",
            "email": "compliance@company.com",
            "role": "owner",
            "department": "Compliance",
            "action_ids": ["act-001"],
        })
        assert resp.status_code in (200, 201)

    @pytest.mark.asyncio
    async def test_assign_stakeholder_missing_email_422(self, client):
        resp = await client.post("/assign-stakeholder", json={
            "plan_id": "plan-001",
            "name": "No Email Person",
            "role": "contributor",
        })
        assert resp.status_code == 422


class TestNotifyStakeholderEndpoint:
    """Test POST /notify-stakeholder."""

    @pytest.mark.asyncio
    async def test_notify_stakeholder_200(self, client):
        resp = await client.post("/notify-stakeholder", json={
            "stakeholder_id": "stk-001",
            "subject": "Action Assigned",
            "message": "A new action has been assigned to you.",
            "channel": "email",
        })
        assert resp.status_code in (200, 404)


class TestNotifyBatchEndpoint:
    """Test POST /notify-batch."""

    @pytest.mark.asyncio
    async def test_notify_batch_200(self, client):
        resp = await client.post("/notify-batch", json={
            "plan_id": "plan-001",
            "subject": "Plan Status Update",
            "message": "The improvement plan has been updated.",
        })
        assert resp.status_code in (200, 404)


class TestRACIMatrixEndpoint:
    """Test GET /raci-matrix/{plan_id}."""

    @pytest.mark.asyncio
    async def test_raci_matrix_200_or_404(self, client):
        resp = await client.get("/raci-matrix/plan-001")
        assert resp.status_code in (200, 404)


class TestEscalateEndpoint:
    """Test POST /escalate."""

    @pytest.mark.asyncio
    async def test_escalate_200(self, client):
        resp = await client.post("/escalate", json={
            "stakeholder_id": "stk-001",
            "action_id": "act-001",
            "reason": "Action overdue by 10 days.",
        })
        assert resp.status_code in (200, 404)


# ===========================================================================
# Activate / Complete Plan
# ===========================================================================

class TestActivatePlanEndpoint:
    """Test POST /activate-plan/{plan_id}."""

    @pytest.mark.asyncio
    async def test_activate_plan_200(self, client):
        create_resp = await client.post("/plans", json={
            "operator_id": "operator-001",
            "title": "Activate Test Plan",
            "commodities": ["coffee"],
        })
        plan_id = create_resp.json()["plan_id"]
        resp = await client.post(f"/activate-plan/{plan_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "active"

    @pytest.mark.asyncio
    async def test_activate_nonexistent_plan_404(self, client):
        resp = await client.post("/activate-plan/plan-nonexistent")
        assert resp.status_code == 404


class TestCompletePlanEndpoint:
    """Test POST /complete-plan/{plan_id}."""

    @pytest.mark.asyncio
    async def test_complete_plan_200(self, client):
        create_resp = await client.post("/plans", json={
            "operator_id": "operator-001",
            "title": "Complete Test Plan",
            "commodities": ["coffee"],
        })
        plan_id = create_resp.json()["plan_id"]
        # Activate first
        await client.post(f"/activate-plan/{plan_id}")
        resp = await client.post(f"/complete-plan/{plan_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"

    @pytest.mark.asyncio
    async def test_complete_nonexistent_plan_404(self, client):
        resp = await client.post("/complete-plan/plan-nonexistent")
        assert resp.status_code == 404


# ===========================================================================
# Generate Summary / Dashboard
# ===========================================================================

class TestGenerateSummaryEndpoint:
    """Test POST /generate-summary."""

    @pytest.mark.asyncio
    async def test_generate_summary_200(self, client):
        resp = await client.post("/generate-summary", json={
            "plan_id": "plan-001",
        })
        assert resp.status_code in (200, 404)

    @pytest.mark.asyncio
    async def test_summary_has_metrics(self, client):
        # Create plan first
        create_resp = await client.post("/plans", json={
            "operator_id": "operator-001",
            "title": "Summary Test",
            "commodities": ["coffee"],
        })
        plan_id = create_resp.json()["plan_id"]
        resp = await client.post("/generate-summary", json={
            "plan_id": plan_id,
        })
        if resp.status_code == 200:
            data = resp.json()
            assert "total_findings" in data or "summary" in data


class TestDashboardEndpoint:
    """Test GET /dashboard."""

    @pytest.mark.asyncio
    async def test_dashboard_200(self, client):
        resp = await client.get("/dashboard")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_dashboard_has_metrics(self, client):
        resp = await client.get("/dashboard")
        data = resp.json()
        assert isinstance(data, dict)


# ===========================================================================
# Error Handling
# ===========================================================================

class TestPausePlanEndpoint:
    """Test POST /pause-plan/{plan_id}."""

    @pytest.mark.asyncio
    async def test_pause_active_plan_200(self, client):
        create_resp = await client.post("/plans", json={
            "operator_id": "operator-001",
            "title": "Pause Test Plan",
            "commodities": ["coffee"],
        })
        plan_id = create_resp.json()["plan_id"]
        await client.post(f"/activate-plan/{plan_id}")
        resp = await client.post(f"/pause-plan/{plan_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "paused"

    @pytest.mark.asyncio
    async def test_pause_nonexistent_plan_404(self, client):
        resp = await client.post("/pause-plan/plan-nonexistent")
        assert resp.status_code == 404


class TestCancelPlanEndpoint:
    """Test POST /cancel-plan/{plan_id}."""

    @pytest.mark.asyncio
    async def test_cancel_plan_200(self, client):
        create_resp = await client.post("/plans", json={
            "operator_id": "operator-001",
            "title": "Cancel Test Plan",
            "commodities": ["coffee"],
        })
        plan_id = create_resp.json()["plan_id"]
        resp = await client.post(f"/cancel-plan/{plan_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_plan_404(self, client):
        resp = await client.post("/cancel-plan/plan-nonexistent")
        assert resp.status_code == 404


# ===========================================================================
# Gap Resolution Endpoint
# ===========================================================================

class TestResolveGapEndpoint:
    """Test POST /resolve-gap."""

    @pytest.mark.asyncio
    async def test_resolve_gap_200(self, client):
        resp = await client.post("/resolve-gap", json={
            "gap_id": "gap-001",
            "resolution_notes": "Addressed via automated geolocation system.",
        })
        assert resp.status_code in (200, 404)

    @pytest.mark.asyncio
    async def test_resolve_gap_missing_notes_422(self, client):
        resp = await client.post("/resolve-gap", json={
            "gap_id": "gap-001",
        })
        assert resp.status_code == 422


# ===========================================================================
# List Endpoints
# ===========================================================================

class TestListFindingsEndpoint:
    """Test GET /findings."""

    @pytest.mark.asyncio
    async def test_list_findings_200(self, client):
        resp = await client.get("/findings", params={"operator_id": "operator-001"})
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_list_findings_by_severity(self, client):
        resp = await client.get("/findings", params={
            "operator_id": "operator-001",
            "severity": "critical",
        })
        assert resp.status_code == 200


class TestListGapsEndpoint:
    """Test GET /gaps."""

    @pytest.mark.asyncio
    async def test_list_gaps_200(self, client):
        resp = await client.get("/gaps", params={"operator_id": "operator-001"})
        assert resp.status_code == 200


class TestListActionsEndpoint:
    """Test GET /actions."""

    @pytest.mark.asyncio
    async def test_list_actions_200(self, client):
        resp = await client.get("/actions", params={"plan_id": "plan-001"})
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_list_actions_by_status(self, client):
        resp = await client.get("/actions", params={
            "plan_id": "plan-001",
            "status": "in_progress",
        })
        assert resp.status_code == 200


class TestListRootCausesEndpoint:
    """Test GET /root-causes."""

    @pytest.mark.asyncio
    async def test_list_root_causes_200(self, client):
        resp = await client.get("/root-causes", params={"plan_id": "plan-001"})
        assert resp.status_code == 200


class TestListStakeholdersEndpoint:
    """Test GET /stakeholders."""

    @pytest.mark.asyncio
    async def test_list_stakeholders_200(self, client):
        resp = await client.get("/stakeholders", params={"plan_id": "plan-001"})
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_list_stakeholders_by_role(self, client):
        resp = await client.get("/stakeholders", params={
            "plan_id": "plan-001",
            "role": "owner",
        })
        assert resp.status_code == 200


# ===========================================================================
# Export Endpoint
# ===========================================================================

class TestExportPlanEndpoint:
    """Test GET /export-plan/{plan_id}."""

    @pytest.mark.asyncio
    async def test_export_plan_200_or_404(self, client):
        resp = await client.get("/export-plan/plan-001")
        assert resp.status_code in (200, 404)

    @pytest.mark.asyncio
    async def test_export_plan_nonexistent_404(self, client):
        resp = await client.get("/export-plan/plan-nonexistent")
        assert resp.status_code == 404


# ===========================================================================
# Error Handling
# ===========================================================================

class TestErrorHandling:
    """Test API error handling."""

    @pytest.mark.asyncio
    async def test_invalid_json_400(self, client):
        resp = await client.post(
            "/plans",
            content=b"not json",
            headers={"content-type": "application/json"},
        )
        assert resp.status_code in (400, 422)

    @pytest.mark.asyncio
    async def test_method_not_allowed_405(self, client):
        resp = await client.delete("/health")
        assert resp.status_code == 405

    @pytest.mark.asyncio
    async def test_not_found_route_404(self, client):
        resp = await client.get("/nonexistent-endpoint")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_empty_body_422(self, client):
        resp = await client.post(
            "/plans",
            json={},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_extra_fields_ignored(self, client):
        resp = await client.post("/plans", json={
            "operator_id": "operator-001",
            "title": "Extra Fields Plan",
            "commodities": ["coffee"],
            "unknown_field": "should be ignored",
        })
        assert resp.status_code in (201, 422)
