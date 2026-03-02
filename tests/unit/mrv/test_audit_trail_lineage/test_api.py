# -*- coding: utf-8 -*-
"""
Test suite for audit_trail_lineage.api.router - AGENT-MRV-030.

Tests all 25+ REST API endpoints using FastAPI TestClient with mock
service dependency override for the Audit Trail & Lineage Agent
(GL-MRV-X-042).

Coverage:
- POST /events (record event)
- POST /events/batch (batch record)
- GET /events/{id} (get event)
- GET /events (list with filters)
- POST /chain/verify (chain verification)
- GET /chain/{org_id}/{year} (get chain)
- GET /chain/{org_id}/{year}/export (export chain)
- GET /chain/{org_id}/{year}/head (get chain head)
- GET /chain/{org_id}/{year}/length (get chain length)
- GET /chain/{org_id}/{year}/statistics (get statistics)
- POST /lineage/nodes (create node)
- POST /lineage/edges (create edge)
- POST /lineage/trace/forward (forward traversal)
- POST /lineage/trace/backward (backward traversal)
- GET /lineage/statistics (graph statistics)
- POST /evidence/package (create package)
- POST /evidence/{id}/sign (sign package)
- POST /evidence/{id}/verify (verify package)
- GET /evidence/packages (list packages)
- POST /compliance/trace (trace compliance)
- POST /compliance/trace-all (trace all frameworks)
- POST /changes/detect (detect change)
- GET /changes/timeline (get timeline)
- POST /pipeline/execute (execute pipeline)
- POST /pipeline/execute-batch (batch pipeline)
- GET /health (health check)
- 404 for missing resources
- 400/422 for invalid inputs

Target: ~100 tests, 85%+ coverage.

Author: GL-TestEngineer
Date: March 2026
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Graceful imports
# ---------------------------------------------------------------------------

try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = None  # type: ignore[assignment, misc]
    TestClient = None  # type: ignore[assignment, misc]

try:
    from greenlang.audit_trail_lineage.api.router import router, get_service
    ROUTER_AVAILABLE = True
except ImportError:
    ROUTER_AVAILABLE = False

_SKIP = pytest.mark.skipif(
    not (FASTAPI_AVAILABLE and ROUTER_AVAILABLE),
    reason="FastAPI or audit trail lineage router not available",
)

PREFIX = "/api/v1/audit-trail-lineage"


# ===========================================================================
# Mock Service
# ===========================================================================


class MockAuditTrailLineageService:
    """
    Mock service returning deterministic responses for all API endpoints.
    Every async/sync method returns a dict matching the router expectations.
    """

    # -- Audit Events -------------------------------------------------------

    async def record_event(self, data: dict) -> dict:
        return {
            "success": True,
            "event_id": "atl-mock001",
            "event_type": data.get("event_type", "DATA_INGESTED"),
            "event_hash": "a" * 64,
            "prev_event_hash": "greenlang-atl-genesis-v1",
            "chain_position": 0,
            "chain_key": "org-001:2025",
            "timestamp": "2025-01-01T00:00:00+00:00",
            "processing_time_ms": 1.5,
        }

    async def record_batch(self, data: dict) -> dict:
        events = data.get("events", [])
        return {
            "success": True,
            "total_recorded": len(events),
            "event_ids": [f"atl-batch-{i}" for i in range(len(events))],
            "processing_time_ms": 5.0,
        }

    async def get_event(self, event_id: str) -> dict:
        if event_id == "atl-missing":
            return None
        return {
            "event_id": event_id,
            "event_type": "DATA_INGESTED",
            "agent_id": "GL-MRV-S1-001",
            "scope": "scope_1",
            "category": None,
            "organization_id": "org-001",
            "reporting_year": 2025,
            "calculation_id": None,
            "data_quality_score": "0.85",
            "payload": {"rows": 500},
            "prev_event_hash": "greenlang-atl-genesis-v1",
            "event_hash": "a" * 64,
            "chain_position": 0,
            "timestamp": "2025-01-01T00:00:00+00:00",
            "metadata": {},
        }

    async def get_events(self, data: dict) -> dict:
        return {
            "success": True,
            "events": [],
            "total_matching": 0,
            "returned_count": 0,
            "has_more": False,
            "limit": 1000,
            "offset": 0,
        }

    # -- Chain Operations ---------------------------------------------------

    async def verify_chain(self, data: dict) -> dict:
        return {
            "success": True,
            "valid": True,
            "chain_key": "org-001:2025",
            "verified_count": 10,
            "first_invalid_position": None,
            "errors": [],
            "verification_time_ms": 2.0,
        }

    async def get_chain(self, org_id: str, year: int) -> dict:
        return {
            "success": True,
            "chain_key": f"{org_id}:{year}",
            "genesis_hash": "greenlang-atl-genesis-v1",
            "head_hash": "b" * 64,
            "length": 5,
            "events": [],
        }

    async def export_chain(self, org_id: str, year: int) -> dict:
        return {
            "success": True,
            "export_id": "export-mock001",
            "chain_key": f"{org_id}:{year}",
            "chain_length": 5,
            "verification": {"valid": True},
            "exported_at": "2025-01-01T00:00:00+00:00",
        }

    async def get_chain_head(self, org_id: str, year: int) -> dict:
        return {"chain_key": f"{org_id}:{year}", "head_hash": "c" * 64}

    async def get_chain_length(self, org_id: str, year: int) -> dict:
        return {"chain_key": f"{org_id}:{year}", "length": 10}

    async def get_event_statistics(self, org_id: str, year: int) -> dict:
        return {
            "success": True,
            "chain_key": f"{org_id}:{year}",
            "total_events": 10,
            "by_event_type": {"DATA_INGESTED": 5, "CALCULATION_COMPLETED": 5},
            "by_scope": {"scope_1": 10},
            "by_agent": {"GL-MRV-S1-001": 10},
        }

    # -- Lineage Operations -------------------------------------------------

    async def add_lineage_node(self, data: dict) -> dict:
        return {"success": True, "node_id": data.get("node_id", "node-001")}

    async def add_lineage_edge(self, data: dict) -> dict:
        return {"success": True, "edge_id": "edge-001"}

    async def traverse_forward(self, data: dict) -> dict:
        return {"success": True, "nodes": [], "edges": []}

    async def traverse_backward(self, data: dict) -> dict:
        return {"success": True, "nodes": [], "edges": []}

    async def get_lineage_statistics(self) -> dict:
        return {"total_nodes": 0, "total_edges": 0, "by_node_type": {}}

    # -- Evidence Operations ------------------------------------------------

    async def create_evidence_package(self, data: dict) -> dict:
        return {
            "success": True,
            "package_id": "pkg-mock-001",
            "created_at": "2025-01-01T00:00:00+00:00",
        }

    async def sign_evidence_package(self, package_id: str, data: dict) -> dict:
        return {"success": True, "signature": "d" * 64}

    async def verify_evidence_package(self, package_id: str) -> dict:
        return {"success": True, "valid": True}

    async def list_evidence_packages(self, data: dict) -> dict:
        return {"success": True, "packages": []}

    # -- Compliance Operations ----------------------------------------------

    async def trace_compliance(self, data: dict) -> dict:
        return {
            "success": True,
            "framework": data.get("framework", "GHG_PROTOCOL"),
            "coverage_pct": 85.0,
            "requirements": [],
            "gaps": [],
        }

    async def trace_all_frameworks(self, data: dict) -> dict:
        return {
            "success": True,
            "frameworks": [],
            "aggregate_coverage_pct": 80.0,
            "total_gaps": 0,
        }

    # -- Change Detection ---------------------------------------------------

    async def detect_change(self, data: dict) -> dict:
        return {
            "success": True,
            "change_id": "chg-mock-001",
            "severity": "LOW",
            "is_material": False,
        }

    async def get_change_timeline(self, data: dict) -> dict:
        return {"success": True, "changes": []}

    # -- Pipeline Operations ------------------------------------------------

    async def execute_pipeline(self, data: dict) -> dict:
        return {
            "success": True,
            "status": "SUCCESS",
            "event_id": "atl-pipe-001",
            "event_hash": "e" * 64,
            "provenance_hash": "f" * 64,
            "processing_time_ms": 3.0,
        }

    async def execute_pipeline_batch(self, data: dict) -> dict:
        return {
            "success": True,
            "total_processed": len(data.get("inputs", [])),
            "results": [],
        }

    # -- Health -------------------------------------------------------------

    async def health_check(self) -> dict:
        return {
            "status": "healthy",
            "agent_id": "GL-MRV-X-042",
            "version": "1.0.0",
            "engines": {
                "audit_event": "ok",
                "lineage_graph": "ok",
                "evidence_packager": "ok",
                "compliance_tracer": "ok",
                "change_detector": "ok",
                "compliance_checker": "ok",
                "pipeline": "ok",
            },
        }


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def mock_service():
    """Create mock service instance."""
    return MockAuditTrailLineageService()


@pytest.fixture
def client(mock_service):
    """Create FastAPI TestClient with mock service override."""
    if not FASTAPI_AVAILABLE or not ROUTER_AVAILABLE:
        pytest.skip("FastAPI or router not available")

    app = FastAPI()
    app.include_router(router, prefix=PREFIX)

    def _override_get_service():
        return mock_service

    app.dependency_overrides[get_service] = _override_get_service
    return TestClient(app)


# ===========================================================================
# HEALTH CHECK TESTS
# ===========================================================================


@_SKIP
class TestHealthEndpoint:
    """Test GET /health endpoint."""

    def test_health_check(self, client):
        """Test health check returns 200."""
        resp = client.get(f"{PREFIX}/health")
        assert resp.status_code == 200

    def test_health_check_body(self, client):
        """Test health check returns expected body."""
        resp = client.get(f"{PREFIX}/health")
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["agent_id"] == "GL-MRV-X-042"

    def test_health_check_has_version(self, client):
        """Test health check includes version."""
        resp = client.get(f"{PREFIX}/health")
        data = resp.json()
        assert data["version"] == "1.0.0"

    def test_health_check_has_engines(self, client):
        """Test health check includes engine statuses."""
        resp = client.get(f"{PREFIX}/health")
        data = resp.json()
        assert "engines" in data
        assert len(data["engines"]) == 7


# ===========================================================================
# RECORD EVENT TESTS
# ===========================================================================


@_SKIP
class TestRecordEventEndpoint:
    """Test POST /events endpoint."""

    def test_record_event_success(self, client):
        """Test recording an event returns 200/201."""
        resp = client.post(
            f"{PREFIX}/events",
            json={
                "event_type": "DATA_INGESTED",
                "agent_id": "GL-MRV-S1-001",
                "scope": "scope_1",
                "organization_id": "org-001",
                "reporting_year": 2025,
                "payload": {"rows": 500},
            },
        )
        assert resp.status_code in [200, 201]
        data = resp.json()
        assert data["success"] is True

    def test_record_event_returns_event_id(self, client):
        """Test response includes event_id."""
        resp = client.post(
            f"{PREFIX}/events",
            json={
                "event_type": "DATA_INGESTED",
                "agent_id": "GL-MRV-S1-001",
                "scope": "scope_1",
                "organization_id": "org-001",
                "reporting_year": 2025,
            },
        )
        data = resp.json()
        assert "event_id" in data

    def test_record_event_returns_hash(self, client):
        """Test response includes event_hash."""
        resp = client.post(
            f"{PREFIX}/events",
            json={
                "event_type": "CALCULATION_COMPLETED",
                "agent_id": "GL-MRV-S1-001",
                "scope": "scope_1",
                "organization_id": "org-001",
                "reporting_year": 2025,
            },
        )
        data = resp.json()
        assert "event_hash" in data
        assert len(data["event_hash"]) == 64

    def test_record_event_missing_required_field(self, client):
        """Test missing required field returns 422."""
        resp = client.post(
            f"{PREFIX}/events",
            json={"event_type": "DATA_INGESTED"},
        )
        assert resp.status_code == 422


# ===========================================================================
# BATCH RECORD TESTS
# ===========================================================================


@_SKIP
class TestBatchRecordEndpoint:
    """Test POST /events/batch endpoint."""

    def test_batch_record_success(self, client):
        """Test batch recording returns 200."""
        resp = client.post(
            f"{PREFIX}/events/batch",
            json={
                "events": [
                    {
                        "event_type": "DATA_INGESTED",
                        "agent_id": "GL-MRV-S1-001",
                        "organization_id": "org-001",
                        "reporting_year": 2025,
                    },
                    {
                        "event_type": "CALCULATION_COMPLETED",
                        "agent_id": "GL-MRV-S1-001",
                        "organization_id": "org-001",
                        "reporting_year": 2025,
                    },
                ],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["total_recorded"] == 2

    def test_batch_record_empty(self, client):
        """Test empty batch returns 400/422."""
        resp = client.post(
            f"{PREFIX}/events/batch",
            json={"events": []},
        )
        assert resp.status_code in [400, 422]


# ===========================================================================
# GET EVENT TESTS
# ===========================================================================


@_SKIP
class TestGetEventEndpoint:
    """Test GET /events/{id} endpoint."""

    def test_get_event_success(self, client):
        """Test getting an existing event returns 200."""
        resp = client.get(f"{PREFIX}/events/atl-mock001")
        assert resp.status_code == 200
        data = resp.json()
        assert data["event_id"] == "atl-mock001"

    def test_get_event_not_found(self, client):
        """Test getting missing event returns 404."""
        resp = client.get(f"{PREFIX}/events/atl-missing")
        assert resp.status_code == 404


# ===========================================================================
# LIST EVENTS TESTS
# ===========================================================================


@_SKIP
class TestListEventsEndpoint:
    """Test GET /events endpoint with query parameters."""

    def test_list_events_success(self, client):
        """Test listing events returns 200."""
        resp = client.get(
            f"{PREFIX}/events",
            params={"organization_id": "org-001", "reporting_year": 2025},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True

    def test_list_events_with_filters(self, client):
        """Test listing events with filter parameters."""
        resp = client.get(
            f"{PREFIX}/events",
            params={
                "organization_id": "org-001",
                "reporting_year": 2025,
                "event_type": "DATA_INGESTED",
                "scope": "scope_1",
                "limit": 10,
            },
        )
        assert resp.status_code == 200


# ===========================================================================
# CHAIN VERIFICATION TESTS
# ===========================================================================


@_SKIP
class TestChainVerificationEndpoint:
    """Test POST /chain/verify endpoint."""

    def test_verify_chain_success(self, client):
        """Test chain verification returns 200."""
        resp = client.post(
            f"{PREFIX}/chain/verify",
            json={"organization_id": "org-001", "reporting_year": 2025},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is True

    def test_verify_chain_with_range(self, client):
        """Test chain verification with start/end positions."""
        resp = client.post(
            f"{PREFIX}/chain/verify",
            json={
                "organization_id": "org-001",
                "reporting_year": 2025,
                "start_position": 0,
                "end_position": 5,
            },
        )
        assert resp.status_code == 200


# ===========================================================================
# CHAIN RETRIEVAL TESTS
# ===========================================================================


@_SKIP
class TestChainRetrievalEndpoints:
    """Test chain retrieval endpoints."""

    def test_get_chain(self, client):
        """Test GET /chain/{org_id}/{year} returns 200."""
        resp = client.get(f"{PREFIX}/chain/org-001/2025")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True

    def test_export_chain(self, client):
        """Test GET /chain/{org_id}/{year}/export returns 200."""
        resp = client.get(f"{PREFIX}/chain/org-001/2025/export")
        assert resp.status_code == 200
        data = resp.json()
        assert "export_id" in data

    def test_get_chain_head(self, client):
        """Test GET /chain/{org_id}/{year}/head returns 200."""
        resp = client.get(f"{PREFIX}/chain/org-001/2025/head")
        assert resp.status_code == 200

    def test_get_chain_length(self, client):
        """Test GET /chain/{org_id}/{year}/length returns 200."""
        resp = client.get(f"{PREFIX}/chain/org-001/2025/length")
        assert resp.status_code == 200

    def test_get_statistics(self, client):
        """Test GET /chain/{org_id}/{year}/statistics returns 200."""
        resp = client.get(f"{PREFIX}/chain/org-001/2025/statistics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_events"] == 10


# ===========================================================================
# LINEAGE NODE TESTS
# ===========================================================================


@_SKIP
class TestLineageNodeEndpoint:
    """Test POST /lineage/nodes endpoint."""

    def test_create_node(self, client):
        """Test creating a lineage node returns 200/201."""
        resp = client.post(
            f"{PREFIX}/lineage/nodes",
            json={
                "node_id": "node-001",
                "node_type": "activity_data",
                "label": "Diesel consumption",
                "scope": "scope_1",
                "agent_id": "GL-MRV-S1-001",
                "organization_id": "org-001",
                "reporting_year": 2025,
            },
        )
        assert resp.status_code in [200, 201]
        data = resp.json()
        assert data["success"] is True


# ===========================================================================
# LINEAGE EDGE TESTS
# ===========================================================================


@_SKIP
class TestLineageEdgeEndpoint:
    """Test POST /lineage/edges endpoint."""

    def test_create_edge(self, client):
        """Test creating a lineage edge returns 200/201."""
        resp = client.post(
            f"{PREFIX}/lineage/edges",
            json={
                "source_node_id": "node-001",
                "target_node_id": "node-002",
                "edge_type": "feeds_into",
            },
        )
        assert resp.status_code in [200, 201]
        data = resp.json()
        assert data["success"] is True


# ===========================================================================
# LINEAGE TRAVERSAL TESTS
# ===========================================================================


@_SKIP
class TestLineageTraversalEndpoints:
    """Test lineage traversal endpoints."""

    def test_forward_traversal(self, client):
        """Test POST /lineage/trace/forward returns 200."""
        resp = client.post(
            f"{PREFIX}/lineage/trace/forward",
            json={"node_id": "node-001"},
        )
        assert resp.status_code == 200

    def test_backward_traversal(self, client):
        """Test POST /lineage/trace/backward returns 200."""
        resp = client.post(
            f"{PREFIX}/lineage/trace/backward",
            json={"node_id": "node-001"},
        )
        assert resp.status_code == 200

    def test_lineage_statistics(self, client):
        """Test GET /lineage/statistics returns 200."""
        resp = client.get(f"{PREFIX}/lineage/statistics")
        assert resp.status_code == 200


# ===========================================================================
# EVIDENCE PACKAGE TESTS
# ===========================================================================


@_SKIP
class TestEvidenceEndpoints:
    """Test evidence package endpoints."""

    def test_create_package(self, client):
        """Test POST /evidence/package returns 200/201."""
        resp = client.post(
            f"{PREFIX}/evidence/package",
            json={
                "organization_id": "org-001",
                "reporting_year": 2025,
                "scope": "scope_1",
                "frameworks": ["GHG_PROTOCOL"],
            },
        )
        assert resp.status_code in [200, 201]
        data = resp.json()
        assert data["success"] is True
        assert "package_id" in data

    def test_sign_package(self, client):
        """Test POST /evidence/{id}/sign returns 200."""
        resp = client.post(
            f"{PREFIX}/evidence/pkg-001/sign",
            json={"signer_id": "auditor-001"},
        )
        assert resp.status_code == 200

    def test_verify_package(self, client):
        """Test POST /evidence/{id}/verify returns 200."""
        resp = client.post(f"{PREFIX}/evidence/pkg-001/verify")
        assert resp.status_code == 200

    def test_list_packages(self, client):
        """Test GET /evidence/packages returns 200."""
        resp = client.get(
            f"{PREFIX}/evidence/packages",
            params={"organization_id": "org-001", "reporting_year": 2025},
        )
        assert resp.status_code == 200


# ===========================================================================
# COMPLIANCE ENDPOINTS TESTS
# ===========================================================================


@_SKIP
class TestComplianceEndpoints:
    """Test compliance tracing endpoints."""

    def test_trace_compliance(self, client):
        """Test POST /compliance/trace returns 200."""
        resp = client.post(
            f"{PREFIX}/compliance/trace",
            json={
                "organization_id": "org-001",
                "reporting_year": 2025,
                "framework": "GHG_PROTOCOL",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True

    def test_trace_all_frameworks(self, client):
        """Test POST /compliance/trace-all returns 200."""
        resp = client.post(
            f"{PREFIX}/compliance/trace-all",
            json={"organization_id": "org-001", "reporting_year": 2025},
        )
        assert resp.status_code == 200


# ===========================================================================
# CHANGE DETECTION ENDPOINTS TESTS
# ===========================================================================


@_SKIP
class TestChangeDetectionEndpoints:
    """Test change detection endpoints."""

    def test_detect_change(self, client):
        """Test POST /changes/detect returns 200."""
        resp = client.post(
            f"{PREFIX}/changes/detect",
            json={
                "change_type": "EMISSION_FACTOR_UPDATE",
                "organization_id": "org-001",
                "reporting_year": 2025,
                "scope": "scope_1",
                "agent_id": "GL-MRV-S1-001",
                "previous_value": {"ef": "2.68"},
                "new_value": {"ef": "2.71"},
                "reason": "DEFRA update",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "change_id" in data

    def test_get_change_timeline(self, client):
        """Test GET /changes/timeline returns 200."""
        resp = client.get(
            f"{PREFIX}/changes/timeline",
            params={"organization_id": "org-001", "reporting_year": 2025},
        )
        assert resp.status_code == 200


# ===========================================================================
# PIPELINE ENDPOINTS TESTS
# ===========================================================================


@_SKIP
class TestPipelineEndpoints:
    """Test pipeline execution endpoints."""

    def test_execute_pipeline(self, client):
        """Test POST /pipeline/execute returns 200."""
        resp = client.post(
            f"{PREFIX}/pipeline/execute",
            json={
                "event_type": "CALCULATION_COMPLETED",
                "agent_id": "GL-MRV-S1-001",
                "scope": "scope_1",
                "organization_id": "org-001",
                "reporting_year": 2025,
                "payload": {"total_co2e": "1234.56"},
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["status"] == "SUCCESS"

    def test_execute_pipeline_returns_hashes(self, client):
        """Test pipeline response includes hashes."""
        resp = client.post(
            f"{PREFIX}/pipeline/execute",
            json={
                "event_type": "DATA_INGESTED",
                "agent_id": "GL-MRV-S1-001",
                "scope": "scope_1",
                "organization_id": "org-001",
                "reporting_year": 2025,
            },
        )
        data = resp.json()
        assert "event_hash" in data
        assert "provenance_hash" in data

    def test_execute_pipeline_batch(self, client):
        """Test POST /pipeline/execute-batch returns 200."""
        resp = client.post(
            f"{PREFIX}/pipeline/execute-batch",
            json={
                "inputs": [
                    {
                        "event_type": "DATA_INGESTED",
                        "agent_id": "GL-MRV-S1-001",
                        "scope": "scope_1",
                        "organization_id": "org-001",
                        "reporting_year": 2025,
                    },
                ],
            },
        )
        assert resp.status_code == 200


# ===========================================================================
# ERROR HANDLING TESTS
# ===========================================================================


@_SKIP
class TestErrorHandling:
    """Test API error handling."""

    def test_nonexistent_endpoint_404(self, client):
        """Test requesting nonexistent endpoint returns 404."""
        resp = client.get(f"{PREFIX}/nonexistent")
        assert resp.status_code in [404, 405]

    def test_invalid_json_returns_422(self, client):
        """Test sending invalid JSON returns 422."""
        resp = client.post(
            f"{PREFIX}/events",
            content="not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422

    def test_wrong_method_405(self, client):
        """Test using wrong HTTP method returns 405."""
        resp = client.delete(f"{PREFIX}/health")
        assert resp.status_code in [404, 405]

    def test_missing_content_type(self, client):
        """Test POST without content-type is handled."""
        resp = client.post(f"{PREFIX}/events")
        assert resp.status_code == 422

    def test_empty_body_post(self, client):
        """Test POST with empty body returns 422."""
        resp = client.post(
            f"{PREFIX}/events",
            json={},
        )
        assert resp.status_code == 422


# ===========================================================================
# QUERY PARAMETER TESTS
# ===========================================================================


@_SKIP
class TestQueryParameters:
    """Test API query parameter handling."""

    def test_list_events_limit_parameter(self, client):
        """Test events list respects limit query param."""
        resp = client.get(
            f"{PREFIX}/events",
            params={
                "organization_id": "org-001",
                "reporting_year": 2025,
                "limit": 5,
            },
        )
        assert resp.status_code == 200

    def test_list_events_offset_parameter(self, client):
        """Test events list respects offset query param."""
        resp = client.get(
            f"{PREFIX}/events",
            params={
                "organization_id": "org-001",
                "reporting_year": 2025,
                "offset": 10,
            },
        )
        assert resp.status_code == 200

    def test_list_events_event_type_filter(self, client):
        """Test events list filters by event_type."""
        resp = client.get(
            f"{PREFIX}/events",
            params={
                "organization_id": "org-001",
                "reporting_year": 2025,
                "event_type": "DATA_INGESTED",
            },
        )
        assert resp.status_code == 200

    def test_list_events_scope_filter(self, client):
        """Test events list filters by scope."""
        resp = client.get(
            f"{PREFIX}/events",
            params={
                "organization_id": "org-001",
                "reporting_year": 2025,
                "scope": "scope_1",
            },
        )
        assert resp.status_code == 200

    def test_list_events_agent_filter(self, client):
        """Test events list filters by agent_id."""
        resp = client.get(
            f"{PREFIX}/events",
            params={
                "organization_id": "org-001",
                "reporting_year": 2025,
                "agent_id": "GL-MRV-S1-001",
            },
        )
        assert resp.status_code == 200

    def test_list_events_category_filter(self, client):
        """Test events list filters by category."""
        resp = client.get(
            f"{PREFIX}/events",
            params={
                "organization_id": "org-001",
                "reporting_year": 2025,
                "category": 6,
            },
        )
        assert resp.status_code == 200
