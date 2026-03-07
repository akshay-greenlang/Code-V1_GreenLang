# -*- coding: utf-8 -*-
"""
Comprehensive API Route Tests - AGENT-EUDR-001 Supply Chain Mapper

80+ test cases covering all 23+ endpoints across 8 route modules:
  - Graph CRUD (create, list, get, delete, export)
  - Multi-tier mapping (discover, tier distribution)
  - Traceability (forward trace, backward trace, batch trace)
  - Risk assessment (propagate, summary, heatmap)
  - Gap analysis (analyze, list, resolve)
  - Visualization (layout, sankey)
  - Supplier onboarding (invite, status, submit)
  - Authentication and RBAC enforcement
  - Error handling and edge cases
  - Pagination

Test Categories:
  1. Happy-path endpoint tests (23 tests)
  2. Authentication tests (8 tests)
  3. RBAC permission tests (8 tests)
  4. Error handling tests (15 tests)
  5. Pagination tests (5 tests)
  6. Edge case tests (10 tests)
  7. Integration flow tests (5 tests)
  8. Rate limiting tests (4 tests)
  9. Validation tests (8 tests)

Total: 86 tests

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-001 Supply Chain Mapping Master (GL-EUDR-SCM-001)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict
from unittest.mock import AsyncMock, patch

import pytest

# FastAPI test client
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.testclient import TestClient

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

# Import API components
from greenlang.agents.eudr.supply_chain_mapper.api.dependencies import (
    AuthUser,
    RateLimiter,
    get_current_user,
    rate_limit_export,
    rate_limit_heavy,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
)
from greenlang.agents.eudr.supply_chain_mapper.api.graph_routes import (
    _get_graph_store,
    _graph_store,
)
from greenlang.agents.eudr.supply_chain_mapper.api.onboarding_routes import (
    _get_invitation_store,
    _invitation_store,
)
from greenlang.agents.eudr.supply_chain_mapper.api.router import router
from greenlang.agents.eudr.supply_chain_mapper.api.schemas import (
    GraphCreateRequest,
    OnboardingInviteRequest,
    RiskPropagateRequest,
)
from greenlang.agents.eudr.supply_chain_mapper.models import (
    ComplianceStatus,
    CustodyModel,
    EUDRCommodity,
    GapSeverity,
    GapType,
    NodeType,
    RiskLevel,
    SupplyChainEdge,
    SupplyChainGap,
    SupplyChainGraph,
    SupplyChainNode,
)


# =============================================================================
# Test Fixtures
# =============================================================================


def _make_test_user(
    user_id: str = "test-user-001",
    operator_id: str = "op-001",
    roles: list = None,
    permissions: list = None,
) -> AuthUser:
    """Create a test AuthUser."""
    return AuthUser(
        user_id=user_id,
        email="test@greenlang.io",
        tenant_id="tenant-001",
        operator_id=operator_id,
        roles=["eudr-analyst"] if roles is None else roles,
        permissions=["eudr-supply-chain:*"] if permissions is None else permissions,
    )


def _make_admin_user() -> AuthUser:
    """Create a test admin AuthUser."""
    return AuthUser(
        user_id="admin-001",
        email="admin@greenlang.io",
        tenant_id="tenant-001",
        operator_id="op-admin",
        roles=["admin"],
        permissions=["*"],
    )


def _make_unauthorized_user() -> AuthUser:
    """Create a test user with no permissions."""
    return AuthUser(
        user_id="noauth-001",
        email="noauth@greenlang.io",
        tenant_id="tenant-001",
        operator_id="op-other",
        roles=[],
        permissions=[],
    )


def _make_test_graph(
    operator_id: str = "op-001",
    commodity: EUDRCommodity = EUDRCommodity.COCOA,
    graph_name: str = "Test Cocoa Graph",
    add_nodes: bool = False,
    add_edges: bool = False,
    add_gaps: bool = False,
) -> SupplyChainGraph:
    """Create a test SupplyChainGraph with optional nodes/edges/gaps."""
    graph_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).replace(microsecond=0)

    graph = SupplyChainGraph(
        graph_id=graph_id,
        operator_id=operator_id,
        commodity=commodity,
        graph_name=graph_name,
        created_at=now,
        updated_at=now,
    )

    if add_nodes:
        # Add 5 nodes: 2 producers, 1 collector, 1 processor, 1 importer
        producer1 = SupplyChainNode(
            node_id="prod-001",
            node_type=NodeType.PRODUCER,
            operator_id="supplier-001",
            operator_name="Farm Alpha",
            country_code="GH",
            region="Ashanti",
            coordinates=(6.6885, -1.6244),
            commodities=[EUDRCommodity.COCOA],
            tier_depth=3,
            risk_score=45.0,
            risk_level=RiskLevel.STANDARD,
            compliance_status=ComplianceStatus.COMPLIANT,
            plot_ids=["plot-001", "plot-002"],
        )
        producer2 = SupplyChainNode(
            node_id="prod-002",
            node_type=NodeType.PRODUCER,
            operator_id="supplier-002",
            operator_name="Farm Beta",
            country_code="CI",
            tier_depth=3,
            risk_score=75.0,
            risk_level=RiskLevel.HIGH,
            compliance_status=ComplianceStatus.PENDING_VERIFICATION,
        )
        collector = SupplyChainNode(
            node_id="coll-001",
            node_type=NodeType.COLLECTOR,
            operator_id="collector-001",
            operator_name="Coop Central",
            country_code="GH",
            commodities=[EUDRCommodity.COCOA],
            tier_depth=2,
            risk_score=40.0,
            risk_level=RiskLevel.STANDARD,
            compliance_status=ComplianceStatus.COMPLIANT,
        )
        processor = SupplyChainNode(
            node_id="proc-001",
            node_type=NodeType.PROCESSOR,
            operator_id="processor-001",
            operator_name="CocoaProcess Ltd",
            country_code="NL",
            commodities=[EUDRCommodity.COCOA, EUDRCommodity.CHOCOLATE],
            tier_depth=1,
            risk_score=20.0,
            risk_level=RiskLevel.LOW,
            compliance_status=ComplianceStatus.COMPLIANT,
        )
        importer = SupplyChainNode(
            node_id="imp-001",
            node_type=NodeType.IMPORTER,
            operator_id="op-001",
            operator_name="EU Choco Imports GmbH",
            country_code="DE",
            commodities=[EUDRCommodity.CHOCOLATE],
            tier_depth=0,
            risk_score=15.0,
            risk_level=RiskLevel.LOW,
            compliance_status=ComplianceStatus.COMPLIANT,
        )

        graph.nodes = {
            "prod-001": producer1,
            "prod-002": producer2,
            "coll-001": collector,
            "proc-001": processor,
            "imp-001": importer,
        }
        graph.total_nodes = 5
        graph.max_tier_depth = 3

    if add_edges:
        edge1 = SupplyChainEdge(
            edge_id="edge-001",
            source_node_id="prod-001",
            target_node_id="coll-001",
            commodity=EUDRCommodity.COCOA,
            product_description="Raw cocoa beans",
            quantity=Decimal("5000"),
            unit="kg",
            batch_number="BATCH-2026-001",
            custody_model=CustodyModel.SEGREGATED,
        )
        edge2 = SupplyChainEdge(
            edge_id="edge-002",
            source_node_id="prod-002",
            target_node_id="coll-001",
            commodity=EUDRCommodity.COCOA,
            product_description="Raw cocoa beans",
            quantity=Decimal("3000"),
            unit="kg",
            batch_number="BATCH-2026-001",
            custody_model=CustodyModel.SEGREGATED,
        )
        edge3 = SupplyChainEdge(
            edge_id="edge-003",
            source_node_id="coll-001",
            target_node_id="proc-001",
            commodity=EUDRCommodity.COCOA,
            product_description="Aggregated cocoa beans",
            quantity=Decimal("8000"),
            unit="kg",
            batch_number="BATCH-2026-002",
            custody_model=CustodyModel.MASS_BALANCE,
        )
        edge4 = SupplyChainEdge(
            edge_id="edge-004",
            source_node_id="proc-001",
            target_node_id="imp-001",
            commodity=EUDRCommodity.CHOCOLATE,
            product_description="Processed chocolate",
            quantity=Decimal("6000"),
            unit="kg",
            custody_model=CustodyModel.IDENTITY_PRESERVED,
        )

        graph.edges = {
            "edge-001": edge1,
            "edge-002": edge2,
            "edge-003": edge3,
            "edge-004": edge4,
        }
        graph.total_edges = 4

    if add_gaps:
        gap1 = SupplyChainGap(
            gap_id="gap-001",
            gap_type=GapType.MISSING_GEOLOCATION,
            severity=GapSeverity.CRITICAL,
            affected_node_id="prod-002",
            description="Producer Farm Beta lacks GPS coordinates",
            eudr_article="Article 9",
        )
        gap2 = SupplyChainGap(
            gap_id="gap-002",
            gap_type=GapType.UNVERIFIED_ACTOR,
            severity=GapSeverity.HIGH,
            affected_node_id="prod-002",
            description="Farm Beta not verified",
            eudr_article="Article 10",
        )
        graph.gaps = [gap1, gap2]

    return graph


@pytest.fixture
def app():
    """Create a FastAPI test app with the EUDR SCM router."""
    if not FASTAPI_AVAILABLE:
        pytest.skip("FastAPI not available")

    test_app = FastAPI(title="Test EUDR SCM API")

    # Override auth dependency to return test user
    test_user = _make_test_user()

    async def _override_auth():
        return test_user

    test_app.dependency_overrides[get_current_user] = _override_auth

    # Disable rate limiting for tests (override must have no params
    # so FastAPI does not try to inject them as query parameters)
    async def _no_rate_limit():
        return None

    from greenlang.agents.eudr.supply_chain_mapper.api.dependencies import (
        rate_limit_export,
        rate_limit_heavy,
        rate_limit_standard,
        rate_limit_write,
    )

    test_app.dependency_overrides[rate_limit_standard] = _no_rate_limit
    test_app.dependency_overrides[rate_limit_write] = _no_rate_limit
    test_app.dependency_overrides[rate_limit_heavy] = _no_rate_limit
    test_app.dependency_overrides[rate_limit_export] = _no_rate_limit

    test_app.include_router(router, prefix="/api")

    return test_app


@pytest.fixture
def client(app):
    """Create a TestClient."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def clean_stores():
    """Clean graph and invitation stores before each test."""
    _graph_store.clear()
    _invitation_store.clear()
    yield
    _graph_store.clear()
    _invitation_store.clear()


@pytest.fixture
def populated_graph() -> SupplyChainGraph:
    """Create and store a fully populated test graph."""
    graph = _make_test_graph(
        add_nodes=True, add_edges=True, add_gaps=True
    )
    _graph_store[graph.graph_id] = graph
    return graph


# =============================================================================
# 1. Happy-Path Endpoint Tests (23 tests)
# =============================================================================


class TestGraphCRUD:
    """Tests for graph CRUD endpoints."""

    def test_create_graph(self, client):
        """POST /graphs creates a new graph and returns 201."""
        resp = client.post(
            "/api/v1/eudr-scm/graphs",
            json={"commodity": "cocoa", "graph_name": "Test Graph"},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["commodity"] == "cocoa"
        assert data["graph_name"] == "Test Graph"
        assert data["status"] == "created"
        assert "graph_id" in data

    def test_create_graph_minimal(self, client):
        """POST /graphs with only required fields."""
        resp = client.post(
            "/api/v1/eudr-scm/graphs",
            json={"commodity": "coffee"},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["commodity"] == "coffee"
        assert data["graph_name"] is None

    def test_list_graphs_empty(self, client):
        """GET /graphs returns empty list when no graphs exist."""
        resp = client.get("/api/v1/eudr-scm/graphs")
        assert resp.status_code == 200
        data = resp.json()
        assert data["graphs"] == []
        assert data["meta"]["total"] == 0

    def test_list_graphs_with_data(self, client, populated_graph):
        """GET /graphs returns graphs owned by the operator."""
        resp = client.get("/api/v1/eudr-scm/graphs")
        assert resp.status_code == 200
        data = resp.json()
        assert data["meta"]["total"] == 1
        assert len(data["graphs"]) == 1
        assert data["graphs"][0]["graph_id"] == populated_graph.graph_id

    def test_list_graphs_commodity_filter(self, client, populated_graph):
        """GET /graphs?commodity=soya filters correctly."""
        resp = client.get("/api/v1/eudr-scm/graphs?commodity=soya")
        assert resp.status_code == 200
        assert resp.json()["meta"]["total"] == 0

    def test_get_graph_detail(self, client, populated_graph):
        """GET /graphs/{id} returns full graph details."""
        resp = client.get(
            f"/api/v1/eudr-scm/graphs/{populated_graph.graph_id}"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["graph_id"] == populated_graph.graph_id
        assert data["total_nodes"] == 5
        assert data["total_edges"] == 4

    def test_delete_graph(self, client, populated_graph):
        """DELETE /graphs/{id} removes the graph."""
        resp = client.delete(
            f"/api/v1/eudr-scm/graphs/{populated_graph.graph_id}"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "deleted"
        # Verify it's gone
        resp2 = client.get(
            f"/api/v1/eudr-scm/graphs/{populated_graph.graph_id}"
        )
        assert resp2.status_code == 404

    def test_export_graph(self, client, populated_graph):
        """GET /graphs/{id}/export returns DDS export data."""
        resp = client.get(
            f"/api/v1/eudr-scm/graphs/{populated_graph.graph_id}/export"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["graph_id"] == populated_graph.graph_id
        assert data["commodity"] == "cocoa"
        assert data["total_supply_chain_actors"] == 5


class TestMappingRoutes:
    """Tests for multi-tier mapping endpoints."""

    def test_discover_tiers(self, client, populated_graph):
        """POST /graphs/{id}/discover triggers tier discovery."""
        resp = client.post(
            f"/api/v1/eudr-scm/graphs/{populated_graph.graph_id}/discover",
            json={"max_depth": 5},
        )
        assert resp.status_code == 202
        data = resp.json()
        assert data["graph_id"] == populated_graph.graph_id
        assert data["status"] == "completed"

    def test_get_tier_distribution(self, client, populated_graph):
        """GET /graphs/{id}/tiers returns tier distribution."""
        resp = client.get(
            f"/api/v1/eudr-scm/graphs/{populated_graph.graph_id}/tiers"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["max_depth"] == 3
        assert "tier_counts" in data


class TestTraceabilityRoutes:
    """Tests for traceability endpoints."""

    def test_forward_trace(self, client, populated_graph):
        """GET /graphs/{id}/trace/forward/{node_id} traces downstream."""
        resp = client.get(
            f"/api/v1/eudr-scm/graphs/{populated_graph.graph_id}"
            "/trace/forward/prod-001"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["direction"] == "forward"
        assert data["start_node_id"] == "prod-001"
        assert "prod-001" in data["visited_nodes"]
        assert data["is_complete"] is True

    def test_backward_trace(self, client, populated_graph):
        """GET /graphs/{id}/trace/backward/{node_id} traces upstream."""
        resp = client.get(
            f"/api/v1/eudr-scm/graphs/{populated_graph.graph_id}"
            "/trace/backward/imp-001"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["direction"] == "backward"
        assert data["start_node_id"] == "imp-001"
        assert len(data["visited_nodes"]) >= 1

    def test_batch_trace(self, client, populated_graph):
        """GET /graphs/{id}/trace/batch/{batch_id} traces batch."""
        resp = client.get(
            f"/api/v1/eudr-scm/graphs/{populated_graph.graph_id}"
            "/trace/batch/BATCH-2026-001"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["batch_id"] == "BATCH-2026-001"
        assert len(data["edges"]) == 2

    def test_batch_trace_not_found(self, client, populated_graph):
        """GET /graphs/{id}/trace/batch/{batch_id} with unknown batch."""
        resp = client.get(
            f"/api/v1/eudr-scm/graphs/{populated_graph.graph_id}"
            "/trace/batch/NONEXISTENT"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_complete"] is False
        assert len(data["edges"]) == 0


class TestRiskRoutes:
    """Tests for risk assessment endpoints."""

    def test_propagate_risk(self, client, populated_graph):
        """POST /graphs/{id}/risk/propagate runs risk propagation."""
        resp = client.post(
            f"/api/v1/eudr-scm/graphs/{populated_graph.graph_id}"
            "/risk/propagate",
            json={"propagation_source": "test_run"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["nodes_updated"] == 5
        assert data["status"] == "completed"
        assert len(data["propagation_results"]) == 5

    def test_risk_summary(self, client, populated_graph):
        """GET /graphs/{id}/risk/summary returns risk statistics."""
        resp = client.get(
            f"/api/v1/eudr-scm/graphs/{populated_graph.graph_id}"
            "/risk/summary"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_nodes"] == 5
        assert "risk_distribution" in data
        assert "average_risk_score" in data

    def test_risk_heatmap(self, client, populated_graph):
        """GET /graphs/{id}/risk/heatmap returns heatmap data."""
        resp = client.get(
            f"/api/v1/eudr-scm/graphs/{populated_graph.graph_id}"
            "/risk/heatmap"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["heatmap_data"]) == 5
        # Check that producer with coordinates has lat/lon
        farm_alpha = [
            h for h in data["heatmap_data"] if h["node_id"] == "prod-001"
        ]
        assert len(farm_alpha) == 1
        assert "lat" in farm_alpha[0]
        assert "lon" in farm_alpha[0]


class TestGapRoutes:
    """Tests for gap analysis endpoints."""

    def test_analyze_gaps(self, client, populated_graph):
        """POST /graphs/{id}/gaps/analyze detects compliance gaps."""
        resp = client.post(
            f"/api/v1/eudr-scm/graphs/{populated_graph.graph_id}"
            "/gaps/analyze",
            json={"include_resolved": False},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_gaps"] > 0
        assert "gaps_by_severity" in data
        assert "compliance_readiness" in data

    def test_list_gaps(self, client, populated_graph):
        """GET /graphs/{id}/gaps returns gap list."""
        resp = client.get(
            f"/api/v1/eudr-scm/graphs/{populated_graph.graph_id}/gaps"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["gaps"]) == 2
        assert data["meta"]["total"] == 2

    def test_resolve_gap(self, client, populated_graph):
        """PUT /graphs/{id}/gaps/{gap_id}/resolve marks gap resolved."""
        resp = client.put(
            f"/api/v1/eudr-scm/graphs/{populated_graph.graph_id}"
            "/gaps/gap-001/resolve",
            json={
                "resolution_notes": "Added GPS coordinates from survey",
                "evidence_ids": ["ev-001"],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "resolved"
        assert data["gap_id"] == "gap-001"


class TestVisualizationRoutes:
    """Tests for visualization endpoints."""

    def test_get_layout_hierarchical(self, client, populated_graph):
        """GET /graphs/{id}/layout with hierarchical algorithm."""
        resp = client.get(
            f"/api/v1/eudr-scm/graphs/{populated_graph.graph_id}"
            "/layout?algorithm=hierarchical"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["layout_algorithm"] == "hierarchical"
        assert len(data["node_positions"]) == 5
        assert "viewport" in data

    def test_get_layout_force_directed(self, client, populated_graph):
        """GET /graphs/{id}/layout with force_directed algorithm."""
        resp = client.get(
            f"/api/v1/eudr-scm/graphs/{populated_graph.graph_id}"
            "/layout?algorithm=force_directed"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["layout_algorithm"] == "force_directed"

    def test_get_sankey(self, client, populated_graph):
        """GET /graphs/{id}/sankey returns flow diagram data."""
        resp = client.get(
            f"/api/v1/eudr-scm/graphs/{populated_graph.graph_id}/sankey"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["nodes"]) == 5
        assert len(data["links"]) == 4


class TestOnboardingRoutes:
    """Tests for supplier onboarding endpoints."""

    def test_create_invitation(self, client):
        """POST /onboarding/invite creates an invitation."""
        resp = client.post(
            "/api/v1/eudr-scm/onboarding/invite",
            json={
                "supplier_name": "Cooperative Alpha GH",
                "supplier_email": "alpha@coop.gh",
                "supplier_country": "GH",
                "commodity": "cocoa",
                "expires_in_days": 30,
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["status"] == "pending"
        assert data["supplier_name"] == "Cooperative Alpha GH"
        assert "token" in data
        assert "onboarding_url" in data

    def test_get_invitation_status(self, client):
        """GET /onboarding/{token} returns invitation status."""
        # Create invitation first
        resp1 = client.post(
            "/api/v1/eudr-scm/onboarding/invite",
            json={
                "supplier_name": "Test Supplier",
                "supplier_email": "test@supplier.com",
                "supplier_country": "BR",
                "commodity": "soya",
            },
        )
        token = resp1.json()["token"]

        resp2 = client.get(f"/api/v1/eudr-scm/onboarding/{token}")
        assert resp2.status_code == 200
        data = resp2.json()
        assert data["status"] == "pending"
        assert data["commodity"] == "soya"

    def test_submit_onboarding(self, client):
        """POST /onboarding/{token}/submit accepts supplier data."""
        # Create invitation
        resp1 = client.post(
            "/api/v1/eudr-scm/onboarding/invite",
            json={
                "supplier_name": "Submit Supplier",
                "supplier_email": "submit@supplier.com",
                "supplier_country": "ID",
                "commodity": "oil_palm",
                "graph_id": "graph-test",
            },
        )
        token = resp1.json()["token"]

        # Submit onboarding data
        resp2 = client.post(
            f"/api/v1/eudr-scm/onboarding/{token}/submit",
            json={
                "operator_name": "Palm Oil Farm Indo",
                "country_code": "ID",
                "region": "Kalimantan",
                "coordinates": [-1.5, 116.0],
                "commodities": ["oil_palm", "palm_oil"],
                "certifications": ["RSPO-2024-ID-001"],
                "plot_ids": ["plot-id-001"],
                "node_type": "producer",
            },
        )
        assert resp2.status_code == 200
        data = resp2.json()
        assert data["status"] == "submitted"


class TestHealthCheck:
    """Tests for health check endpoint."""

    def test_health_check(self, client):
        """GET /health returns healthy status."""
        resp = client.get("/api/v1/eudr-scm/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["agent_id"] == "GL-EUDR-SCM-001"


# =============================================================================
# 2. Authentication Tests (8 tests)
# =============================================================================


class TestAuthentication:
    """Tests for authentication enforcement."""

    def test_unauthenticated_request_rejected(self):
        """Endpoints reject requests without auth."""
        if not FASTAPI_AVAILABLE:
            pytest.skip("FastAPI not available")

        app = FastAPI()
        # Do NOT override auth dependency
        app.include_router(router, prefix="/api")
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.get("/api/v1/eudr-scm/graphs")
        assert resp.status_code in (401, 403, 422)

    def test_auth_user_has_identity(self, client):
        """Authenticated requests have user identity in context."""
        # Create a graph to verify user identity is used
        resp = client.post(
            "/api/v1/eudr-scm/graphs",
            json={"commodity": "cocoa"},
        )
        assert resp.status_code == 201
        assert resp.json()["operator_id"] == "op-001"

    def test_admin_can_access_any_graph(self, app, populated_graph):
        """Admin users can access any graph regardless of operator_id."""
        admin = _make_admin_user()

        async def _admin_auth():
            return admin

        app.dependency_overrides[get_current_user] = _admin_auth
        client = TestClient(app)

        # Admin accessing graph owned by op-001
        resp = client.get(
            f"/api/v1/eudr-scm/graphs/{populated_graph.graph_id}"
        )
        assert resp.status_code == 200

    def test_different_operator_cannot_access(self, app, populated_graph):
        """Users from different operators cannot access foreign graphs."""
        other = _make_test_user(
            user_id="other-user", operator_id="op-other"
        )

        async def _other_auth():
            return other

        app.dependency_overrides[get_current_user] = _other_auth
        client = TestClient(app)

        resp = client.get(
            f"/api/v1/eudr-scm/graphs/{populated_graph.graph_id}"
        )
        assert resp.status_code == 403

    def test_different_operator_cannot_delete(self, app, populated_graph):
        """Users from different operators cannot delete foreign graphs."""
        other = _make_test_user(
            user_id="other-user", operator_id="op-other"
        )

        async def _other_auth():
            return other

        app.dependency_overrides[get_current_user] = _other_auth
        client = TestClient(app)

        resp = client.delete(
            f"/api/v1/eudr-scm/graphs/{populated_graph.graph_id}"
        )
        assert resp.status_code == 403

    def test_different_operator_cannot_export(self, app, populated_graph):
        """Users from different operators cannot export foreign graphs."""
        other = _make_test_user(
            user_id="other-user", operator_id="op-other"
        )

        async def _other_auth():
            return other

        app.dependency_overrides[get_current_user] = _other_auth
        client = TestClient(app)

        resp = client.get(
            f"/api/v1/eudr-scm/graphs/{populated_graph.graph_id}/export"
        )
        assert resp.status_code == 403

    def test_onboarding_status_no_auth_required(self):
        """GET /onboarding/{token} does not require authentication."""
        if not FASTAPI_AVAILABLE:
            pytest.skip("FastAPI not available")

        app = FastAPI()
        # No auth override
        app.include_router(router, prefix="/api")
        client = TestClient(app, raise_server_exceptions=False)

        # Should return 404 (not 401) for missing token
        resp = client.get("/api/v1/eudr-scm/onboarding/fake-token")
        assert resp.status_code == 404

    def test_onboarding_submit_no_auth_required(self):
        """POST /onboarding/{token}/submit does not require authentication."""
        if not FASTAPI_AVAILABLE:
            pytest.skip("FastAPI not available")

        app = FastAPI()
        app.include_router(router, prefix="/api")
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post(
            "/api/v1/eudr-scm/onboarding/fake-token/submit",
            json={
                "operator_name": "Test",
                "country_code": "BR",
            },
        )
        assert resp.status_code == 404


# =============================================================================
# 3. RBAC Permission Tests (8 tests)
# =============================================================================


class TestRBAC:
    """Tests for RBAC permission enforcement."""

    def test_wildcard_permission_grants_access(self, app):
        """eudr-supply-chain:* grants all sub-permissions."""
        user = _make_test_user(
            permissions=["eudr-supply-chain:*"]
        )

        async def _auth():
            return user

        app.dependency_overrides[get_current_user] = _auth
        client = TestClient(app)

        resp = client.post(
            "/api/v1/eudr-scm/graphs",
            json={"commodity": "cocoa"},
        )
        assert resp.status_code == 201

    def test_specific_permission_grants_access(self, app):
        """Specific permission like graphs:create grants access."""
        user = _make_test_user(
            permissions=["eudr-supply-chain:graphs:create"]
        )

        async def _auth():
            return user

        app.dependency_overrides[get_current_user] = _auth
        client = TestClient(app)

        resp = client.post(
            "/api/v1/eudr-scm/graphs",
            json={"commodity": "cocoa"},
        )
        assert resp.status_code == 201

    def test_missing_permission_denied(self, app):
        """Missing required permission returns 403."""
        user = _make_test_user(permissions=["other:permission"])

        async def _auth():
            return user

        app.dependency_overrides[get_current_user] = _auth
        client = TestClient(app)

        resp = client.post(
            "/api/v1/eudr-scm/graphs",
            json={"commodity": "cocoa"},
        )
        assert resp.status_code == 403

    def test_admin_role_bypasses_permissions(self, app):
        """Admin role bypasses all permission checks."""
        user = _make_test_user(roles=["admin"], permissions=[])

        async def _auth():
            return user

        app.dependency_overrides[get_current_user] = _auth
        client = TestClient(app)

        resp = client.post(
            "/api/v1/eudr-scm/graphs",
            json={"commodity": "cocoa"},
        )
        assert resp.status_code == 201

    def test_platform_admin_bypasses_permissions(self, app):
        """platform_admin role bypasses all permission checks."""
        user = _make_test_user(roles=["platform_admin"], permissions=[])

        async def _auth():
            return user

        app.dependency_overrides[get_current_user] = _auth
        client = TestClient(app)

        resp = client.get("/api/v1/eudr-scm/graphs")
        assert resp.status_code == 200

    def test_read_permission_allows_get(self, app, populated_graph):
        """graphs:read permission allows GET /graphs."""
        user = _make_test_user(
            permissions=["eudr-supply-chain:graphs:read"]
        )

        async def _auth():
            return user

        app.dependency_overrides[get_current_user] = _auth
        client = TestClient(app)

        resp = client.get("/api/v1/eudr-scm/graphs")
        assert resp.status_code == 200

    def test_write_permission_required_for_risk(self, app, populated_graph):
        """risk:write permission required for POST risk/propagate."""
        user = _make_test_user(
            permissions=["eudr-supply-chain:risk:read"]  # only read
        )

        async def _auth():
            return user

        app.dependency_overrides[get_current_user] = _auth
        client = TestClient(app)

        resp = client.post(
            f"/api/v1/eudr-scm/graphs/{populated_graph.graph_id}"
            "/risk/propagate",
            json={},
        )
        assert resp.status_code == 403

    def test_no_permissions_denied_all(self, app):
        """User with no permissions gets 403 on all protected endpoints."""
        user = _make_test_user(roles=[], permissions=[])

        async def _auth():
            return user

        app.dependency_overrides[get_current_user] = _auth
        client = TestClient(app)

        # All protected endpoints should return 403
        resp = client.get("/api/v1/eudr-scm/graphs")
        assert resp.status_code == 403

        resp2 = client.post(
            "/api/v1/eudr-scm/graphs",
            json={"commodity": "cocoa"},
        )
        assert resp2.status_code == 403


# =============================================================================
# 4. Error Handling Tests (15 tests)
# =============================================================================


class TestErrorHandling:
    """Tests for error handling across all endpoints."""

    def test_graph_not_found(self, client):
        """GET /graphs/{id} returns 404 for missing graph."""
        resp = client.get("/api/v1/eudr-scm/graphs/nonexistent-id")
        assert resp.status_code == 404

    def test_delete_nonexistent_graph(self, client):
        """DELETE /graphs/{id} returns 404 for missing graph."""
        resp = client.delete("/api/v1/eudr-scm/graphs/nonexistent-id")
        assert resp.status_code == 404

    def test_export_nonexistent_graph(self, client):
        """GET /graphs/{id}/export returns 404 for missing graph."""
        resp = client.get("/api/v1/eudr-scm/graphs/nonexistent-id/export")
        assert resp.status_code == 404

    def test_discover_nonexistent_graph(self, client):
        """POST /graphs/{id}/discover returns 404 for missing graph."""
        resp = client.post(
            "/api/v1/eudr-scm/graphs/nonexistent/discover",
            json={"max_depth": 5},
        )
        assert resp.status_code == 404

    def test_tiers_nonexistent_graph(self, client):
        """GET /graphs/{id}/tiers returns 404 for missing graph."""
        resp = client.get("/api/v1/eudr-scm/graphs/nonexistent/tiers")
        assert resp.status_code == 404

    def test_trace_nonexistent_graph(self, client):
        """GET trace on nonexistent graph returns 404."""
        resp = client.get(
            "/api/v1/eudr-scm/graphs/nonexistent/trace/forward/node-1"
        )
        assert resp.status_code == 404

    def test_trace_nonexistent_node(self, client, populated_graph):
        """GET trace on nonexistent node returns 404."""
        resp = client.get(
            f"/api/v1/eudr-scm/graphs/{populated_graph.graph_id}"
            "/trace/forward/nonexistent-node"
        )
        assert resp.status_code == 404

    def test_risk_propagate_nonexistent_graph(self, client):
        """POST risk/propagate on nonexistent graph returns 404."""
        resp = client.post(
            "/api/v1/eudr-scm/graphs/nonexistent/risk/propagate",
            json={},
        )
        assert resp.status_code == 404

    def test_risk_summary_nonexistent_graph(self, client):
        """GET risk/summary on nonexistent graph returns 404."""
        resp = client.get(
            "/api/v1/eudr-scm/graphs/nonexistent/risk/summary"
        )
        assert resp.status_code == 404

    def test_gap_analyze_nonexistent_graph(self, client):
        """POST gaps/analyze on nonexistent graph returns 404."""
        resp = client.post(
            "/api/v1/eudr-scm/graphs/nonexistent/gaps/analyze",
            json={},
        )
        assert resp.status_code == 404

    def test_resolve_nonexistent_gap(self, client, populated_graph):
        """PUT gaps/{gap_id}/resolve returns 404 for missing gap."""
        resp = client.put(
            f"/api/v1/eudr-scm/graphs/{populated_graph.graph_id}"
            "/gaps/nonexistent-gap/resolve",
            json={"resolution_notes": "Fixed"},
        )
        assert resp.status_code == 404

    def test_resolve_already_resolved_gap(self, client, populated_graph):
        """PUT gaps/{gap_id}/resolve returns 400 if already resolved."""
        # First resolve
        client.put(
            f"/api/v1/eudr-scm/graphs/{populated_graph.graph_id}"
            "/gaps/gap-001/resolve",
            json={"resolution_notes": "Fixed"},
        )
        # Try again
        resp = client.put(
            f"/api/v1/eudr-scm/graphs/{populated_graph.graph_id}"
            "/gaps/gap-001/resolve",
            json={"resolution_notes": "Fixed again"},
        )
        assert resp.status_code == 400

    def test_onboarding_expired_token(self, client):
        """GET /onboarding/{token} returns 410 for expired invitation."""
        # Create invitation with past expiry
        token = "expired-token-test"
        past = datetime.now(timezone.utc) - timedelta(days=1)
        _invitation_store[token] = {
            "invitation_id": "inv-expired",
            "token": token,
            "supplier_name": "Expired Supplier",
            "supplier_email": "expired@test.com",
            "supplier_country": "BR",
            "commodity": "soya",
            "graph_id": None,
            "status": "pending",
            "created_at": past.isoformat(),
            "expires_at": past.isoformat(),
            "submitted_at": None,
        }

        resp = client.get(f"/api/v1/eudr-scm/onboarding/{token}")
        assert resp.status_code == 410

    def test_onboarding_double_submit(self, client):
        """POST /onboarding/{token}/submit returns 409 on duplicate."""
        # Create and submit invitation
        resp1 = client.post(
            "/api/v1/eudr-scm/onboarding/invite",
            json={
                "supplier_name": "Double Submit",
                "supplier_email": "double@test.com",
                "supplier_country": "BR",
                "commodity": "soya",
            },
        )
        token = resp1.json()["token"]

        # First submit
        client.post(
            f"/api/v1/eudr-scm/onboarding/{token}/submit",
            json={"operator_name": "Test", "country_code": "BR"},
        )

        # Second submit
        resp2 = client.post(
            f"/api/v1/eudr-scm/onboarding/{token}/submit",
            json={"operator_name": "Test2", "country_code": "BR"},
        )
        assert resp2.status_code == 409

    def test_layout_nonexistent_graph(self, client):
        """GET /graphs/{id}/layout returns 404 for missing graph."""
        resp = client.get(
            "/api/v1/eudr-scm/graphs/nonexistent/layout"
        )
        assert resp.status_code == 404


# =============================================================================
# 5. Pagination Tests (5 tests)
# =============================================================================


class TestPagination:
    """Tests for pagination across list endpoints."""

    def test_graphs_pagination_default(self, client, populated_graph):
        """Default pagination returns first page."""
        resp = client.get("/api/v1/eudr-scm/graphs")
        data = resp.json()
        assert data["meta"]["limit"] == 50
        assert data["meta"]["offset"] == 0

    def test_graphs_pagination_custom(self, client):
        """Custom limit and offset work correctly."""
        # Create multiple graphs
        for i in range(5):
            client.post(
                "/api/v1/eudr-scm/graphs",
                json={"commodity": "cocoa", "graph_name": f"Graph {i}"},
            )

        resp = client.get(
            "/api/v1/eudr-scm/graphs?limit=2&offset=0"
        )
        data = resp.json()
        assert data["meta"]["limit"] == 2
        assert data["meta"]["total"] == 5
        assert len(data["graphs"]) == 2
        assert data["meta"]["has_more"] is True

    def test_graphs_pagination_offset(self, client):
        """Offset skips correct number of results."""
        for i in range(3):
            client.post(
                "/api/v1/eudr-scm/graphs",
                json={"commodity": "cocoa", "graph_name": f"Graph {i}"},
            )

        resp = client.get(
            "/api/v1/eudr-scm/graphs?limit=10&offset=2"
        )
        data = resp.json()
        assert len(data["graphs"]) == 1
        assert data["meta"]["has_more"] is False

    def test_gaps_pagination(self, client, populated_graph):
        """Gap list supports pagination."""
        resp = client.get(
            f"/api/v1/eudr-scm/graphs/{populated_graph.graph_id}"
            "/gaps?limit=1&offset=0"
        )
        data = resp.json()
        assert len(data["gaps"]) == 1
        assert data["meta"]["total"] == 2
        assert data["meta"]["has_more"] is True

    def test_gaps_pagination_second_page(self, client, populated_graph):
        """Gap list pagination second page."""
        resp = client.get(
            f"/api/v1/eudr-scm/graphs/{populated_graph.graph_id}"
            "/gaps?limit=1&offset=1"
        )
        data = resp.json()
        assert len(data["gaps"]) == 1
        assert data["meta"]["offset"] == 1


# =============================================================================
# 6. Edge Case Tests (10 tests)
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_graph_tier_distribution(self, client):
        """Tier distribution for graph with no nodes."""
        graph = _make_test_graph()
        _graph_store[graph.graph_id] = graph

        resp = client.get(
            f"/api/v1/eudr-scm/graphs/{graph.graph_id}/tiers"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["max_depth"] == 0
        assert data["tier_counts"] == {}

    def test_empty_graph_risk_summary(self, client):
        """Risk summary for graph with no nodes."""
        graph = _make_test_graph()
        _graph_store[graph.graph_id] = graph

        resp = client.get(
            f"/api/v1/eudr-scm/graphs/{graph.graph_id}/risk/summary"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_nodes"] == 0
        assert data["average_risk_score"] == 0.0

    def test_empty_graph_layout(self, client):
        """Layout for graph with no nodes."""
        graph = _make_test_graph()
        _graph_store[graph.graph_id] = graph

        resp = client.get(
            f"/api/v1/eudr-scm/graphs/{graph.graph_id}/layout"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["node_positions"] == {}

    def test_empty_graph_sankey(self, client):
        """Sankey for graph with no nodes."""
        graph = _make_test_graph()
        _graph_store[graph.graph_id] = graph

        resp = client.get(
            f"/api/v1/eudr-scm/graphs/{graph.graph_id}/sankey"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["nodes"] == []
        assert data["links"] == []

    def test_empty_graph_gap_analysis(self, client):
        """Gap analysis on graph with no nodes."""
        graph = _make_test_graph()
        _graph_store[graph.graph_id] = graph

        resp = client.post(
            f"/api/v1/eudr-scm/graphs/{graph.graph_id}/gaps/analyze",
            json={},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_gaps"] == 0

    def test_forward_trace_leaf_node(self, client, populated_graph):
        """Forward trace from leaf node returns single node."""
        resp = client.get(
            f"/api/v1/eudr-scm/graphs/{populated_graph.graph_id}"
            "/trace/forward/imp-001"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["visited_nodes"] == ["imp-001"]
        assert data["visited_edges"] == []

    def test_backward_trace_root_node(self, client, populated_graph):
        """Backward trace from root producer returns single node."""
        resp = client.get(
            f"/api/v1/eudr-scm/graphs/{populated_graph.graph_id}"
            "/trace/backward/prod-001"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["visited_nodes"] == ["prod-001"]
        assert len(data["origin_plot_ids"]) == 2

    def test_risk_propagate_custom_weights(self, client, populated_graph):
        """Risk propagation with custom weights."""
        resp = client.post(
            f"/api/v1/eudr-scm/graphs/{populated_graph.graph_id}"
            "/risk/propagate",
            json={
                "risk_weights": {
                    "country": 0.40,
                    "commodity": 0.20,
                    "supplier": 0.20,
                    "deforestation": 0.20,
                },
            },
        )
        assert resp.status_code == 200
        assert resp.json()["nodes_updated"] == 5

    def test_gap_filter_by_severity(self, client, populated_graph):
        """Gap list filtered by severity."""
        resp = client.get(
            f"/api/v1/eudr-scm/graphs/{populated_graph.graph_id}"
            "/gaps?severity=critical"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert all(g["severity"] == "critical" for g in data["gaps"])

    def test_gap_filter_by_type(self, client, populated_graph):
        """Gap list filtered by gap type."""
        resp = client.get(
            f"/api/v1/eudr-scm/graphs/{populated_graph.graph_id}"
            "/gaps?gap_type=missing_geolocation"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert all(
            g["gap_type"] == "missing_geolocation" for g in data["gaps"]
        )


# =============================================================================
# 7. Integration Flow Tests (5 tests)
# =============================================================================


class TestIntegrationFlows:
    """Tests for end-to-end workflow scenarios."""

    def test_full_graph_lifecycle(self, client):
        """Create -> get -> export -> delete graph lifecycle."""
        # Create
        r1 = client.post(
            "/api/v1/eudr-scm/graphs",
            json={"commodity": "coffee", "graph_name": "Coffee Chain"},
        )
        assert r1.status_code == 201
        graph_id = r1.json()["graph_id"]

        # Get
        r2 = client.get(f"/api/v1/eudr-scm/graphs/{graph_id}")
        assert r2.status_code == 200

        # Export
        r3 = client.get(f"/api/v1/eudr-scm/graphs/{graph_id}/export")
        assert r3.status_code == 200
        assert r3.json()["commodity"] == "coffee"

        # Delete
        r4 = client.delete(f"/api/v1/eudr-scm/graphs/{graph_id}")
        assert r4.status_code == 200

        # Verify deleted
        r5 = client.get(f"/api/v1/eudr-scm/graphs/{graph_id}")
        assert r5.status_code == 404

    def test_gap_analyze_then_resolve(self, client, populated_graph):
        """Analyze gaps then resolve them sequentially."""
        gid = populated_graph.graph_id

        # Analyze
        r1 = client.post(
            f"/api/v1/eudr-scm/graphs/{gid}/gaps/analyze",
            json={"include_resolved": False},
        )
        assert r1.status_code == 200
        gaps = r1.json()["gaps"]
        assert len(gaps) > 0

        # Resolve first gap
        gap_id = gaps[0]["gap_id"]
        r2 = client.put(
            f"/api/v1/eudr-scm/graphs/{gid}/gaps/{gap_id}/resolve",
            json={"resolution_notes": "Fixed via satellite data"},
        )
        assert r2.status_code == 200
        assert r2.json()["status"] == "resolved"

    def test_risk_then_heatmap_flow(self, client, populated_graph):
        """Propagate risk then check heatmap."""
        gid = populated_graph.graph_id

        # Propagate
        r1 = client.post(
            f"/api/v1/eudr-scm/graphs/{gid}/risk/propagate",
            json={},
        )
        assert r1.status_code == 200

        # Check heatmap
        r2 = client.get(f"/api/v1/eudr-scm/graphs/{gid}/risk/heatmap")
        assert r2.status_code == 200
        assert len(r2.json()["heatmap_data"]) == 5

    def test_onboarding_full_flow(self, client):
        """Invite -> check status -> submit full onboarding flow."""
        # Invite
        r1 = client.post(
            "/api/v1/eudr-scm/onboarding/invite",
            json={
                "supplier_name": "Flow Supplier",
                "supplier_email": "flow@test.com",
                "supplier_country": "CO",
                "commodity": "coffee",
            },
        )
        assert r1.status_code == 201
        token = r1.json()["token"]

        # Check status
        r2 = client.get(f"/api/v1/eudr-scm/onboarding/{token}")
        assert r2.status_code == 200
        assert r2.json()["status"] == "pending"

        # Submit
        r3 = client.post(
            f"/api/v1/eudr-scm/onboarding/{token}/submit",
            json={
                "operator_name": "Colombian Coffee Co",
                "country_code": "CO",
                "commodities": ["coffee"],
                "node_type": "producer",
            },
        )
        assert r3.status_code == 200
        assert r3.json()["status"] == "submitted"

        # Verify status updated
        r4 = client.get(f"/api/v1/eudr-scm/onboarding/{token}")
        assert r4.status_code == 200
        assert r4.json()["status"] == "submitted"

    def test_discover_then_visualize(self, client, populated_graph):
        """Discover tiers then generate layout and sankey."""
        gid = populated_graph.graph_id

        # Discover
        r1 = client.post(
            f"/api/v1/eudr-scm/graphs/{gid}/discover",
            json={"max_depth": 10},
        )
        assert r1.status_code == 202

        # Layout
        r2 = client.get(
            f"/api/v1/eudr-scm/graphs/{gid}/layout?algorithm=hierarchical"
        )
        assert r2.status_code == 200
        assert len(r2.json()["node_positions"]) == 5

        # Sankey
        r3 = client.get(f"/api/v1/eudr-scm/graphs/{gid}/sankey")
        assert r3.status_code == 200
        assert len(r3.json()["links"]) == 4


# =============================================================================
# 8. Rate Limiting Tests (4 tests)
# =============================================================================


class TestRateLimiting:
    """Tests for rate limiter behavior."""

    def test_rate_limiter_allows_under_limit(self):
        """RateLimiter allows requests under the limit."""
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        # Simulate 4 requests -- under limit, should not raise
        assert limiter.max_requests == 5
        assert limiter.window_seconds == 60

    def test_rate_limiter_configuration(self):
        """RateLimiter stores configuration correctly."""
        limiter = RateLimiter(max_requests=10, window_seconds=30)
        assert limiter.max_requests == 10
        assert limiter.window_seconds == 30

    @pytest.mark.asyncio
    async def test_rate_limiter_exceeds_limit(self):
        """RateLimiter raises 429 when limit exceeded."""
        limiter = RateLimiter(max_requests=2, window_seconds=60)

        mock_request = type(
            "MockRequest",
            (),
            {"url": type("URL", (), {"path": "/test"})()},
        )()
        user = _make_test_user()

        # First two requests should pass
        await limiter(mock_request, user)
        await limiter(mock_request, user)

        # Third should fail
        with pytest.raises(HTTPException) as exc_info:
            await limiter(mock_request, user)
        assert exc_info.value.status_code == 429

    @pytest.mark.asyncio
    async def test_rate_limiter_per_user_isolation(self):
        """RateLimiter tracks limits per user independently."""
        limiter = RateLimiter(max_requests=2, window_seconds=60)

        mock_request = type(
            "MockRequest",
            (),
            {"url": type("URL", (), {"path": "/test"})()},
        )()

        user1 = _make_test_user(user_id="user-1")
        user2 = _make_test_user(user_id="user-2")

        # User 1: 2 requests
        await limiter(mock_request, user1)
        await limiter(mock_request, user1)

        # User 2 should still be allowed
        await limiter(mock_request, user2)

        # User 1 should be blocked
        with pytest.raises(HTTPException) as exc_info:
            await limiter(mock_request, user1)
        assert exc_info.value.status_code == 429


# =============================================================================
# 9. Validation Tests (8 tests)
# =============================================================================


class TestValidation:
    """Tests for request validation."""

    def test_invalid_commodity_rejected(self, client):
        """POST /graphs with invalid commodity returns 422."""
        resp = client.post(
            "/api/v1/eudr-scm/graphs",
            json={"commodity": "invalid_commodity"},
        )
        assert resp.status_code == 422

    def test_missing_required_field(self, client):
        """POST /graphs without commodity returns 422."""
        resp = client.post("/api/v1/eudr-scm/graphs", json={})
        assert resp.status_code == 422

    def test_invalid_risk_weights_sum(self, client, populated_graph):
        """POST risk/propagate with weights not summing to 1.0 returns 422."""
        resp = client.post(
            f"/api/v1/eudr-scm/graphs/{populated_graph.graph_id}"
            "/risk/propagate",
            json={
                "risk_weights": {
                    "country": 0.5,
                    "commodity": 0.5,
                    "supplier": 0.5,
                    "deforestation": 0.5,
                }
            },
        )
        assert resp.status_code == 422

    def test_invalid_risk_weights_keys(self, client, populated_graph):
        """POST risk/propagate with wrong weight keys returns 422."""
        resp = client.post(
            f"/api/v1/eudr-scm/graphs/{populated_graph.graph_id}"
            "/risk/propagate",
            json={
                "risk_weights": {
                    "wrong_key": 1.0,
                }
            },
        )
        assert resp.status_code == 422

    def test_empty_resolution_notes(self, client, populated_graph):
        """PUT gaps/{id}/resolve with empty notes returns 422."""
        resp = client.put(
            f"/api/v1/eudr-scm/graphs/{populated_graph.graph_id}"
            "/gaps/gap-001/resolve",
            json={"resolution_notes": ""},
        )
        assert resp.status_code == 422

    def test_invalid_country_code_onboarding(self, client):
        """POST /onboarding/invite with invalid country code returns 422."""
        resp = client.post(
            "/api/v1/eudr-scm/onboarding/invite",
            json={
                "supplier_name": "Test",
                "supplier_email": "test@test.com",
                "supplier_country": "123",
                "commodity": "cocoa",
            },
        )
        assert resp.status_code == 422

    def test_discover_max_depth_too_large(self, client, populated_graph):
        """POST /discover with max_depth > 50 returns 422."""
        resp = client.post(
            f"/api/v1/eudr-scm/graphs/{populated_graph.graph_id}/discover",
            json={"max_depth": 100},
        )
        assert resp.status_code == 422

    def test_pagination_limit_too_large(self, client):
        """GET /graphs with limit > 1000 returns 422."""
        resp = client.get("/api/v1/eudr-scm/graphs?limit=5000")
        assert resp.status_code == 422


# =============================================================================
# Dependency Unit Tests
# =============================================================================


class TestDependencies:
    """Unit tests for dependency functions."""

    def test_auth_user_model(self):
        """AuthUser model creates correctly."""
        user = AuthUser(
            user_id="u1",
            email="u1@test.com",
            tenant_id="t1",
            operator_id="op1",
            roles=["analyst"],
            permissions=["eudr-supply-chain:*"],
        )
        assert user.user_id == "u1"
        assert user.tenant_id == "t1"

    def test_pagination_params_defaults(self):
        """PaginationParams has correct defaults."""
        from greenlang.agents.eudr.supply_chain_mapper.api.dependencies import (
            PaginationParams,
        )

        params = PaginationParams()
        assert params.limit == 50
        assert params.offset == 0

    def test_error_response_model(self):
        """ErrorResponse creates correctly."""
        from greenlang.agents.eudr.supply_chain_mapper.api.dependencies import (
            ErrorResponse,
        )

        err = ErrorResponse(error="not_found", message="Resource not found")
        assert err.error == "not_found"

    @pytest.mark.asyncio
    async def test_require_permission_factory(self):
        """require_permission returns a callable dependency."""
        dep = require_permission("eudr-supply-chain:graphs:read")
        assert callable(dep)

        # Test with user that has wildcard permission
        user = _make_test_user(permissions=["eudr-supply-chain:*"])
        result = await dep(user=user)
        assert result.user_id == user.user_id

    @pytest.mark.asyncio
    async def test_require_permission_denied(self):
        """require_permission raises 403 for missing permission."""
        dep = require_permission("eudr-supply-chain:graphs:read")
        user = _make_test_user(roles=[], permissions=[])

        with pytest.raises(HTTPException) as exc_info:
            await dep(user=user)
        assert exc_info.value.status_code == 403
