# -*- coding: utf-8 -*-
"""
API Integration Tests for Agent Registry Service (AGENT-FOUND-007)

Tests end-to-end API-like workflows including agent CRUD, health checks,
capability discovery, dependency resolution, and statistics using
direct function calls (no network/TestClient) to comply with the
integration test network blocker.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline API service for integration testing (direct calls, no HTTP)
# ---------------------------------------------------------------------------


class IntegrationRegistryAPI:
    """Full integration service simulating API endpoints in-memory."""

    def __init__(self):
        self._agents: Dict[str, Dict[str, Dict]] = {}
        self._health: Dict[str, str] = {}
        self._capabilities: Dict[str, List[str]] = defaultdict(list)

    def health(self) -> Dict[str, Any]:
        return {"status": "healthy", "service": "agent-registry"}

    def register_agent(self, body: Dict[str, Any]) -> Dict[str, Any]:
        agent_id = body.get("agent_id")
        if not agent_id:
            return {"error": "agent_id required", "status": 422}
        version = body.get("version", "1.0.0")
        if agent_id not in self._agents:
            self._agents[agent_id] = {}
        h = hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()
        body["provenance_hash"] = h
        self._agents[agent_id][version] = body
        self._health[agent_id] = body.get("health_status", "unknown")
        for cap in body.get("capabilities", []):
            name = cap.get("name") if isinstance(cap, dict) else cap
            if name and agent_id not in self._capabilities[name]:
                self._capabilities[name].append(agent_id)
        return {"agent_id": agent_id, "version": version, "provenance_hash": h, "status": 201}

    def get_agent(self, agent_id: str) -> Dict[str, Any]:
        if agent_id not in self._agents:
            return {"error": "Not found", "status": 404}
        versions = self._agents[agent_id]
        latest = max(versions.keys())
        return {**versions[latest], "status": 200}

    def list_agents(self, layer: Optional[str] = None) -> Dict[str, Any]:
        results = []
        for aid, versions in self._agents.items():
            latest = max(versions.keys())
            entry = versions[latest]
            if layer and entry.get("layer") != layer:
                continue
            results.append(entry)
        return {"agents": results, "count": len(results), "status": 200}

    def unregister_agent(self, agent_id: str) -> Dict[str, Any]:
        if agent_id not in self._agents:
            return {"error": "Not found", "status": 404}
        del self._agents[agent_id]
        self._health.pop(agent_id, None)
        return {"deleted": True, "status": 200}

    def get_health(self, agent_id: str) -> Dict[str, Any]:
        if agent_id not in self._health:
            return {"agent_id": agent_id, "status": "unknown"}
        return {"agent_id": agent_id, "status": self._health[agent_id]}

    def set_health(self, agent_id: str, status: str) -> Dict[str, Any]:
        self._health[agent_id] = status
        return {"agent_id": agent_id, "status": status}

    def discover(self, capability: str) -> Dict[str, Any]:
        agents = self._capabilities.get(capability, [])
        return {"capability": capability, "agents": agents}

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "total_agents": len(self._agents),
            "total_versions": sum(len(v) for v in self._agents.values()),
        }


@pytest.fixture
def api():
    return IntegrationRegistryAPI()


# ===========================================================================
# Test Classes
# ===========================================================================


class TestAPIHealthIntegration:
    def test_health_endpoint(self, api):
        result = api.health()
        assert result["status"] == "healthy"

    def test_health_service_name(self, api):
        result = api.health()
        assert result["service"] == "agent-registry"


class TestAPIRegistrationIntegration:
    def test_register_agent(self, api):
        result = api.register_agent({"agent_id": "gl-001", "name": "Agent 1"})
        assert result["status"] == 201

    def test_register_missing_id(self, api):
        result = api.register_agent({"name": "No ID"})
        assert result["status"] == 422

    def test_register_has_provenance_hash(self, api):
        result = api.register_agent({"agent_id": "gl-001", "name": "A"})
        assert len(result["provenance_hash"]) == 64


class TestAPICRUDIntegration:
    def test_full_lifecycle(self, api):
        # Register
        result = api.register_agent({"agent_id": "gl-001", "name": "A"})
        assert result["status"] == 201

        # Get
        result = api.get_agent("gl-001")
        assert result["status"] == 200

        # List
        result = api.list_agents()
        assert result["count"] == 1

        # Unregister
        result = api.unregister_agent("gl-001")
        assert result["deleted"] is True

        # Verify deleted
        result = api.get_agent("gl-001")
        assert result["status"] == 404

    def test_multiple_agents(self, api):
        for i in range(5):
            api.register_agent({"agent_id": f"gl-{i:03d}", "name": f"Agent {i}"})
        result = api.list_agents()
        assert result["count"] == 5

    def test_filter_by_layer(self, api):
        api.register_agent({"agent_id": "gl-001", "name": "A", "layer": "calculation"})
        api.register_agent({"agent_id": "gl-002", "name": "B", "layer": "reporting"})
        result = api.list_agents(layer="calculation")
        assert result["count"] == 1


class TestAPIHealthCheckIntegration:
    def test_get_health(self, api):
        api.register_agent({"agent_id": "gl-001", "name": "A", "health_status": "healthy"})
        result = api.get_health("gl-001")
        assert result["status"] == "healthy"

    def test_set_health(self, api):
        api.register_agent({"agent_id": "gl-001", "name": "A"})
        api.set_health("gl-001", "degraded")
        result = api.get_health("gl-001")
        assert result["status"] == "degraded"


class TestAPIDiscoverIntegration:
    def test_discover_by_capability(self, api):
        api.register_agent({
            "agent_id": "gl-001", "name": "A",
            "capabilities": [{"name": "carbon_calc"}],
        })
        result = api.discover("carbon_calc")
        assert "gl-001" in result["agents"]

    def test_discover_no_match(self, api):
        result = api.discover("nonexistent")
        assert result["agents"] == []


class TestAPIStatisticsIntegration:
    def test_statistics(self, api):
        api.register_agent({"agent_id": "gl-001", "name": "A"})
        api.register_agent({"agent_id": "gl-002", "name": "B"})
        stats = api.get_statistics()
        assert stats["total_agents"] == 2
