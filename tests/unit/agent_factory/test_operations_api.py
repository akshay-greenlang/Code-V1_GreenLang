# -*- coding: utf-8 -*-
"""
Unit tests for the Agent Factory Operations API (INFRA-010 iteration).

Tests the async operations endpoints (202 Accepted pattern) including the
OperationManager business logic for deploy, rollback, progress tracking,
cancellation, idempotency, and timeout handling. Also tests the CRUD
endpoints in factory_routes.py.

Coverage target: 85%+ of operations_routes.py, lifecycle_routes.py, factory_routes.py
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from greenlang.infrastructure.agent_factory.api.lifecycle_routes import (
    DeployRequest,
    DrainRequest,
    DrainResponse,
    HealthResponse,
    HistoryResponse,
    LifecycleEvent,
    RestartRequest,
    RestartResponse,
    RetireRequest,
    RetireResponse,
    RollbackRequest,
    router as lifecycle_router,
    _lifecycle_store,
)
from greenlang.infrastructure.agent_factory.api.factory_routes import (
    AgentCreateRequest,
    AgentListResponse,
    AgentMetricsResponse,
    AgentResponse,
    AgentUpdateRequest,
    BatchExecuteRequest,
    BatchExecuteResponse,
    ExecuteRequest,
    ExecuteResponse,
    router as factory_router,
    _agent_store,
)
from greenlang.infrastructure.agent_factory.api.operations_routes import (
    OperationCreateRequest,
    OperationManager,
    OperationResponse,
    operation_manager,
    router as operations_router,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def clear_stores():
    """Clear in-memory stores before and after each test."""
    _agent_store.clear()
    _lifecycle_store.clear()
    # Also clear the operation_manager state
    operation_manager._operations.clear()
    operation_manager._idempotency_cache.clear()
    operation_manager._cancel_flags.clear()
    yield
    _agent_store.clear()
    _lifecycle_store.clear()
    operation_manager._operations.clear()
    operation_manager._idempotency_cache.clear()
    operation_manager._cancel_flags.clear()


@pytest.fixture
def app() -> FastAPI:
    """Create a FastAPI test application with all routers."""
    app = FastAPI()
    app.include_router(factory_router)
    app.include_router(lifecycle_router)
    app.include_router(operations_router)
    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Create a synchronous test client for the FastAPI application."""
    return TestClient(app)


@pytest.fixture
def registered_agent(client: TestClient) -> Dict[str, Any]:
    """Register a test agent and return the response data."""
    body = {
        "agent_key": "test-agent",
        "version": "1.0.0",
        "agent_type": "deterministic",
        "description": "Test agent for unit tests",
    }
    response = client.post("/api/v1/factory/agents", json=body)
    return response.json()


@pytest.fixture
def deploy_request_body() -> Dict[str, Any]:
    """Standard deploy request body."""
    return {
        "version": "1.2.0",
        "environment": "staging",
        "strategy": "canary",
        "canary_percent": 10,
        "config_overrides": {"timeout_seconds": 90},
    }


# ============================================================================
# TestOperationsAPI
# ============================================================================


class TestOperationsAPI:
    """Tests for async operations endpoints (202 Accepted pattern)."""

    def test_create_operation_returns_202(
        self, client: TestClient, deploy_request_body: Dict[str, Any]
    ) -> None:
        """Deploy endpoint returns 202 Accepted with operation tracking info."""
        response = client.post(
            "/api/v1/factory/lifecycle/agents/test-agent/deploy",
            json=deploy_request_body,
        )
        assert response.status_code == 202
        data = response.json()
        assert data["status"] in ("pending", "running")
        assert data["agent_key"] == "test-agent"
        assert data["operation_type"] == "deploy"
        assert data["operation_id"] is not None
        assert data["poll_url"] is not None
        assert "/api/v1/factory/operations/" in data["poll_url"]

    def test_create_operation_idempotency(
        self, client: TestClient, deploy_request_body: Dict[str, Any]
    ) -> None:
        """Deploy calls with same params return same operation (idempotency)."""
        response1 = client.post(
            "/api/v1/factory/lifecycle/agents/test-agent/deploy",
            json=deploy_request_body,
        )
        response2 = client.post(
            "/api/v1/factory/lifecycle/agents/test-agent/deploy",
            json=deploy_request_body,
        )
        assert response1.status_code == 202
        assert response2.status_code == 202
        # Same params -> same idempotency key -> same operation_id
        assert response1.json()["operation_id"] == response2.json()["operation_id"]

    def test_create_operation_different_params(
        self, client: TestClient
    ) -> None:
        """Deploy calls with different params create different operations."""
        resp1 = client.post(
            "/api/v1/factory/lifecycle/agents/test-agent/deploy",
            json={"version": "1.0.0", "environment": "dev", "strategy": "rolling"},
        )
        resp2 = client.post(
            "/api/v1/factory/lifecycle/agents/test-agent/deploy",
            json={"version": "2.0.0", "environment": "prod", "strategy": "canary"},
        )
        assert resp1.json()["operation_id"] != resp2.json()["operation_id"]

    def test_get_operation_status(self, client: TestClient) -> None:
        """Health endpoint returns current agent health status."""
        response = client.get(
            "/api/v1/factory/lifecycle/agents/test-agent/health"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["agent_key"] == "test-agent"
        assert data["status"] == "healthy"
        assert data["liveness"] == "pass"
        assert data["readiness"] == "pass"
        assert "database" in data["checks"]
        assert "redis" in data["checks"]

    def test_cancel_operation_via_drain(self, client: TestClient) -> None:
        """Drain endpoint transitions agent to draining state."""
        response = client.post(
            "/api/v1/factory/lifecycle/agents/test-agent/drain",
            json={"timeout_seconds": 30, "reason": "maintenance window"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "draining"
        assert data["agent_key"] == "test-agent"
        assert data["reason"] == "maintenance window"
        assert data["timeout_seconds"] == 30

    def test_list_operations_via_history(self, client: TestClient) -> None:
        """History endpoint returns lifecycle events for an agent."""
        # Create some events
        client.post(
            "/api/v1/factory/lifecycle/agents/test-agent/deploy",
            json={"version": "1.0.0", "environment": "dev", "strategy": "rolling"},
        )
        client.post(
            "/api/v1/factory/lifecycle/agents/test-agent/drain",
            json={"timeout_seconds": 60},
        )

        response = client.get(
            "/api/v1/factory/lifecycle/agents/test-agent/history"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["agent_key"] == "test-agent"
        assert data["total"] == 2
        assert len(data["events"]) == 2

    def test_list_operations_filter_by_limit(self, client: TestClient) -> None:
        """History endpoint respects the limit query parameter."""
        for i in range(3):
            client.post(
                "/api/v1/factory/lifecycle/agents/test-agent/drain",
                json={"timeout_seconds": 60, "reason": f"drain-{i}"},
            )

        response = client.get(
            "/api/v1/factory/lifecycle/agents/test-agent/history?limit=2"
        )
        data = response.json()
        assert data["total"] == 3
        assert len(data["events"]) == 2

    def test_list_operations_empty_history(self, client: TestClient) -> None:
        """History endpoint returns empty events for unknown agent."""
        response = client.get(
            "/api/v1/factory/lifecycle/agents/unknown-agent/history"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert data["events"] == []


# ============================================================================
# TestOperationManager
# ============================================================================


class TestOperationManager:
    """Tests for OperationManager business logic."""

    def test_create_deploy_operation(self) -> None:
        """OperationManager creates a deploy operation with correct fields."""
        record = operation_manager.create_operation(
            operation_type="deploy",
            agent_key="carbon-agent",
            params={"version": "1.2.0", "environment": "staging"},
            idempotency_key="test-key-1",
            created_by="admin",
        )
        assert record["operation_type"] == "deploy"
        assert record["agent_key"] == "carbon-agent"
        assert record["status"] == "pending"
        assert record["progress_pct"] == 0
        assert record["operation_id"] is not None
        assert record["created_by"] == "admin"

    def test_create_rollback_operation(self) -> None:
        """OperationManager creates a rollback operation."""
        record = operation_manager.create_operation(
            operation_type="rollback",
            agent_key="carbon-agent",
            params={"target_version": "1.0.0", "reason": "regression"},
            idempotency_key="test-rollback-1",
            created_by="deployer",
        )
        assert record["operation_type"] == "rollback"
        assert record["status"] == "pending"

    def test_update_progress(self) -> None:
        """update_progress sets the completion percentage."""
        record = operation_manager.create_operation(
            operation_type="deploy",
            agent_key="test-agent",
            params={},
            idempotency_key="progress-test-1",
            created_by="test",
        )
        op_id = record["operation_id"]
        operation_manager.update_progress(op_id, 50)

        updated = operation_manager.get_operation(op_id)
        assert updated["progress_pct"] == 50

    def test_update_progress_clamps_values(self) -> None:
        """update_progress clamps to 0-100 range."""
        record = operation_manager.create_operation(
            operation_type="deploy",
            agent_key="test-agent",
            params={},
            idempotency_key="clamp-test",
            created_by="test",
        )
        op_id = record["operation_id"]

        operation_manager.update_progress(op_id, 150)
        assert operation_manager.get_operation(op_id)["progress_pct"] == 100

        operation_manager.update_progress(op_id, -10)
        assert operation_manager.get_operation(op_id)["progress_pct"] == 0

    def test_complete_operation(self) -> None:
        """Operations have a 'completed' terminal status after execution."""
        record = operation_manager.create_operation(
            operation_type="deploy",
            agent_key="test-agent",
            params={},
            idempotency_key="complete-test",
            created_by="test",
        )
        # Simulate completion
        record["status"] = "completed"
        record["progress_pct"] = 100
        record["result"] = {"deployment_id": "abc123"}

        updated = operation_manager.get_operation(record["operation_id"])
        assert updated["status"] == "completed"
        assert updated["result"]["deployment_id"] == "abc123"

    def test_fail_operation(self) -> None:
        """Failed operations record error messages."""
        record = operation_manager.create_operation(
            operation_type="deploy",
            agent_key="test-agent",
            params={},
            idempotency_key="fail-test",
            created_by="test",
        )
        record["status"] = "failed"
        record["error_message"] = "Container image not found"

        updated = operation_manager.get_operation(record["operation_id"])
        assert updated["status"] == "failed"
        assert "Container image" in updated["error_message"]

    def test_cancel_running_operation(self) -> None:
        """request_cancellation marks a pending operation as cancelled."""
        record = operation_manager.create_operation(
            operation_type="deploy",
            agent_key="test-agent",
            params={},
            idempotency_key="cancel-test",
            created_by="test",
        )
        cancelled = operation_manager.request_cancellation(record["operation_id"])
        assert cancelled["status"] == "cancelled"
        assert cancelled["cancelled_at"] is not None

    def test_cancel_terminal_operation_raises(self) -> None:
        """Cancelling an already-completed operation raises ValueError."""
        record = operation_manager.create_operation(
            operation_type="deploy",
            agent_key="test-agent",
            params={},
            idempotency_key="cancel-terminal-test",
            created_by="test",
        )
        record["status"] = "completed"

        with pytest.raises(ValueError, match="terminal state"):
            operation_manager.request_cancellation(record["operation_id"])

    def test_idempotency_key_dedup(self) -> None:
        """Same idempotency key returns the same operation record."""
        record1 = operation_manager.create_operation(
            operation_type="deploy",
            agent_key="test-agent",
            params={"version": "1.0.0"},
            idempotency_key="idem-key-123",
            created_by="test",
        )
        record2 = operation_manager.create_operation(
            operation_type="deploy",
            agent_key="test-agent",
            params={"version": "1.0.0"},
            idempotency_key="idem-key-123",
            created_by="test",
        )
        assert record1["operation_id"] == record2["operation_id"]

    def test_operation_not_found(self) -> None:
        """get_operation returns None for unknown IDs."""
        assert operation_manager.get_operation("nonexistent-id") is None

    def test_update_progress_not_found_raises(self) -> None:
        """update_progress raises KeyError for unknown operation."""
        with pytest.raises(KeyError):
            operation_manager.update_progress("bad-id", 50)

    def test_list_operations_filter_by_status(self) -> None:
        """list_operations filters by status."""
        op1 = operation_manager.create_operation(
            operation_type="deploy", agent_key="a", params={},
            idempotency_key="list-s1", created_by="test",
        )
        op2 = operation_manager.create_operation(
            operation_type="rollback", agent_key="b", params={},
            idempotency_key="list-s2", created_by="test",
        )
        op2["status"] = "completed"

        pending_ops, pending_total = operation_manager.list_operations(status="pending")
        assert pending_total == 1
        assert pending_ops[0]["operation_id"] == op1["operation_id"]

    def test_list_operations_filter_by_agent(self) -> None:
        """list_operations filters by agent_key."""
        operation_manager.create_operation(
            operation_type="deploy", agent_key="agent-a", params={},
            idempotency_key="list-a1", created_by="test",
        )
        operation_manager.create_operation(
            operation_type="deploy", agent_key="agent-b", params={},
            idempotency_key="list-a2", created_by="test",
        )

        ops, total = operation_manager.list_operations(agent_key="agent-a")
        assert total == 1
        assert ops[0]["agent_key"] == "agent-a"

    def test_invalid_operation_type_raises(self) -> None:
        """Invalid operation type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid operation_type"):
            operation_manager.create_operation(
                operation_type="invalid",
                agent_key="test",
                params={},
                idempotency_key="bad-type",
                created_by="test",
            )

    def test_is_cancelled_flag(self) -> None:
        """is_cancelled returns True after cancellation request."""
        record = operation_manager.create_operation(
            operation_type="deploy", agent_key="a", params={},
            idempotency_key="cancel-flag-test", created_by="test",
        )
        assert operation_manager.is_cancelled(record["operation_id"]) is False
        operation_manager.request_cancellation(record["operation_id"])
        assert operation_manager.is_cancelled(record["operation_id"]) is True


# ============================================================================
# TestOperationsAPIEndpoints (via operations_routes)
# ============================================================================


class TestOperationsAPIEndpoints:
    """Tests for the /api/v1/factory/operations endpoints."""

    def test_create_operation_endpoint(self, client: TestClient) -> None:
        """POST /operations returns 202 with operation details."""
        response = client.post(
            "/api/v1/factory/operations/",
            json={
                "operation_type": "deploy",
                "agent_key": "test-agent",
                "params": {"version": "1.0.0"},
            },
        )
        assert response.status_code == 202
        data = response.json()
        assert data["operation_type"] == "deploy"
        assert data["status"] in ("pending", "running")
        assert data["poll_url"] is not None

    def test_get_operation_endpoint(self, client: TestClient) -> None:
        """GET /operations/{id} returns operation status."""
        create_resp = client.post(
            "/api/v1/factory/operations/",
            json={
                "operation_type": "pack",
                "agent_key": "test-agent",
                "params": {},
            },
        )
        op_id = create_resp.json()["operation_id"]

        get_resp = client.get(f"/api/v1/factory/operations/{op_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["operation_id"] == op_id

    def test_get_operation_not_found(self, client: TestClient) -> None:
        """GET /operations/{bad_id} returns 404."""
        response = client.get("/api/v1/factory/operations/nonexistent")
        assert response.status_code == 404

    def test_cancel_operation_endpoint(self, client: TestClient) -> None:
        """DELETE /operations/{id} requests cancellation."""
        create_resp = client.post(
            "/api/v1/factory/operations/",
            json={
                "operation_type": "deploy",
                "agent_key": "test-agent",
                "params": {"version": "1.0.0"},
                "idempotency_key": "cancel-endpoint-test",
            },
        )
        op_id = create_resp.json()["operation_id"]

        cancel_resp = client.delete(f"/api/v1/factory/operations/{op_id}")
        assert cancel_resp.status_code == 200
        data = cancel_resp.json()
        assert data["status"] == "cancelled"

    def test_list_operations_endpoint(self, client: TestClient) -> None:
        """GET /operations/ returns a paginated list."""
        for i in range(3):
            client.post(
                "/api/v1/factory/operations/",
                json={
                    "operation_type": "pack",
                    "agent_key": f"agent-{i}",
                    "params": {},
                    "idempotency_key": f"list-endpoint-{i}",
                },
            )

        response = client.get("/api/v1/factory/operations/?limit=2")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3
        assert len(data["operations"]) == 2


# ============================================================================
# TestFactoryAgentCRUD
# ============================================================================


class TestFactoryAgentCRUD:
    """Tests for core agent CRUD operations in factory_routes."""

    def test_create_agent_returns_201(self, client: TestClient) -> None:
        """Creating an agent returns 201 with agent details."""
        body = {
            "agent_key": "emissions-calc",
            "version": "1.0.0",
            "agent_type": "deterministic",
            "description": "Emissions calculator agent",
        }
        response = client.post("/api/v1/factory/agents", json=body)
        assert response.status_code == 201
        data = response.json()
        assert data["agent_key"] == "emissions-calc"
        assert data["version"] == "1.0.0"
        assert data["status"] == "created"

    def test_create_duplicate_agent_returns_409(
        self, client: TestClient, registered_agent: Dict[str, Any]
    ) -> None:
        """Creating an agent with an existing key returns 409 Conflict."""
        body = {"agent_key": "test-agent", "version": "2.0.0"}
        response = client.post("/api/v1/factory/agents", json=body)
        assert response.status_code == 409

    def test_get_agent(
        self, client: TestClient, registered_agent: Dict[str, Any]
    ) -> None:
        """Get returns agent details for a registered agent."""
        response = client.get("/api/v1/factory/agents/test-agent")
        assert response.status_code == 200
        assert response.json()["agent_key"] == "test-agent"

    def test_get_agent_not_found(self, client: TestClient) -> None:
        """Get returns 404 for unknown agent."""
        response = client.get("/api/v1/factory/agents/nonexistent")
        assert response.status_code == 404

    def test_update_agent(
        self, client: TestClient, registered_agent: Dict[str, Any]
    ) -> None:
        """Update modifies agent configuration."""
        response = client.put(
            "/api/v1/factory/agents/test-agent",
            json={"version": "2.0.0", "description": "Updated description"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["version"] == "2.0.0"
        assert data["description"] == "Updated description"

    def test_delete_agent(
        self, client: TestClient, registered_agent: Dict[str, Any]
    ) -> None:
        """Delete removes an agent and returns 204."""
        response = client.delete("/api/v1/factory/agents/test-agent")
        assert response.status_code == 204
        response = client.get("/api/v1/factory/agents/test-agent")
        assert response.status_code == 404

    def test_list_agents_pagination(self, client: TestClient) -> None:
        """List agents supports pagination."""
        for i in range(5):
            client.post(
                "/api/v1/factory/agents",
                json={"agent_key": f"agent-{i:03d}", "version": "1.0.0"},
            )
        response = client.get("/api/v1/factory/agents?page=1&page_size=2")
        data = response.json()
        assert data["total"] == 5
        assert len(data["agents"]) == 2

    def test_execute_agent(
        self, client: TestClient, registered_agent: Dict[str, Any]
    ) -> None:
        """Execute triggers agent execution and returns task details."""
        response = client.post(
            "/api/v1/factory/agents/test-agent/execute",
            json={"input_data": {"scope": 1}},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["agent_key"] == "test-agent"
        assert data["status"] == "queued"

    def test_execute_agent_not_found(self, client: TestClient) -> None:
        """Execute returns 404 for unknown agent."""
        response = client.post(
            "/api/v1/factory/agents/nonexistent/execute",
            json={"input_data": {}},
        )
        assert response.status_code == 404

    def test_get_agent_metrics(
        self, client: TestClient, registered_agent: Dict[str, Any]
    ) -> None:
        """Metrics endpoint returns agent execution metrics."""
        response = client.get("/api/v1/factory/agents/test-agent/metrics")
        assert response.status_code == 200
        data = response.json()
        assert data["agent_key"] == "test-agent"
        assert "execution_count" in data

    def test_batch_execute(self, client: TestClient) -> None:
        """Batch execute processes multiple agents."""
        client.post("/api/v1/factory/agents", json={"agent_key": "agent-a", "version": "1.0.0"})
        client.post("/api/v1/factory/agents", json={"agent_key": "agent-b", "version": "1.0.0"})
        response = client.post(
            "/api/v1/factory/agents/batch-execute",
            json={
                "agent_keys": ["agent-a", "agent-b"],
                "tasks": [{"input_data": {"x": 1}}, {"input_data": {"x": 2}}],
            },
        )
        data = response.json()
        assert data["total"] == 2
        assert data["succeeded"] == 2
        assert data["failed"] == 0

    def test_batch_execute_partial_failure(self, client: TestClient) -> None:
        """Batch execute handles missing agents gracefully."""
        client.post("/api/v1/factory/agents", json={"agent_key": "agent-a", "version": "1.0.0"})
        response = client.post(
            "/api/v1/factory/agents/batch-execute",
            json={
                "agent_keys": ["agent-a", "nonexistent-agent"],
                "tasks": [{"input_data": {"x": 1}}, {"input_data": {"x": 2}}],
            },
        )
        data = response.json()
        assert data["succeeded"] == 1
        assert data["failed"] == 1
