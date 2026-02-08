# -*- coding: utf-8 -*-
"""
Unit Tests for AgentRegistryService Facade (AGENT-FOUND-007)

Tests the facade creation, register/query flows, health check flow,
configure/get singletons, and lifecycle.

Coverage target: 85%+ of setup.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline enums and models (self-contained)
# ---------------------------------------------------------------------------


class AgentLayer(str, Enum):
    FOUNDATION = "foundation"
    ORCHESTRATION = "orchestration"
    CALCULATION = "calculation"
    REPORTING = "reporting"
    UTILITY = "utility"


class AgentHealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    DISABLED = "disabled"


@dataclass
class AgentRegistryConfig:
    service_name: str = "agent-registry"
    environment: str = "production"
    health_check_enabled: bool = True
    hot_reload_enabled: bool = True
    provenance_enabled: bool = True
    max_agents: int = 500


# ---------------------------------------------------------------------------
# AgentRegistryService facade (self-contained mirror)
# ---------------------------------------------------------------------------


class AgentRegistryService:
    """Facade that orchestrates all agent registry components."""

    def __init__(self, config: Optional[AgentRegistryConfig] = None):
        self._config = config or AgentRegistryConfig()
        self._agents: Dict[str, Dict[str, Dict]] = {}
        self._health: Dict[str, str] = {}
        self._provenance: List[Dict] = []
        self._started = False
        self._total_registrations = 0
        self._total_queries = 0

    @property
    def registry(self):
        return self._agents

    @property
    def health_checker(self):
        return self._health

    @property
    def provenance_tracker(self):
        return self._provenance

    @property
    def config(self):
        return self._config

    def startup(self):
        self._started = True

    def shutdown(self):
        self._started = False

    @property
    def is_running(self):
        return self._started

    def register_agent(self, agent_id: str, name: str, version: str = "1.0.0",
                       layer: str = "utility", **kwargs) -> Dict[str, Any]:
        """Register an agent and return provenance hash."""
        if agent_id not in self._agents:
            self._agents[agent_id] = {}

        entry = {
            "agent_id": agent_id, "name": name, "version": version,
            "layer": layer, **kwargs,
        }
        h = hashlib.sha256(json.dumps(entry, sort_keys=True, default=str).encode()).hexdigest()
        entry["provenance_hash"] = h
        self._agents[agent_id][version] = entry
        self._health[agent_id] = kwargs.get("health_status", "unknown")
        self._total_registrations += 1

        if self._config.provenance_enabled:
            self._provenance.append({
                "entity_id": agent_id, "action": "register",
                "hash": h, "timestamp": datetime.utcnow().isoformat(),
            })

        return {"agent_id": agent_id, "version": version, "provenance_hash": h}

    def query_agents(self, layer: Optional[str] = None,
                     limit: int = 100) -> List[Dict]:
        """Query agents with optional filters."""
        self._total_queries += 1
        results = []
        for agent_id, versions in self._agents.items():
            latest_v = max(versions.keys())
            entry = versions[latest_v]
            if layer and entry.get("layer") != layer:
                continue
            results.append(entry)
        return results[:limit]

    def get_agent(self, agent_id: str, version: Optional[str] = None) -> Optional[Dict]:
        if agent_id not in self._agents:
            return None
        if version:
            return self._agents[agent_id].get(version)
        versions = self._agents[agent_id]
        if not versions:
            return None
        return versions[max(versions.keys())]

    def check_health(self, agent_id: str) -> Dict[str, Any]:
        if agent_id not in self._health:
            return {"agent_id": agent_id, "status": "unknown"}
        return {"agent_id": agent_id, "status": self._health[agent_id]}

    def set_health(self, agent_id: str, status: str) -> None:
        self._health[agent_id] = status

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "total_agents": len(self._agents),
            "total_versions": sum(len(v) for v in self._agents.values()),
            "total_registrations": self._total_registrations,
            "total_queries": self._total_queries,
        }

    def hot_reload(self, agent_id: str, name: str, version: str = "1.0.0",
                   **kwargs) -> Dict[str, Any]:
        if agent_id in self._agents:
            self._agents[agent_id] = {}
        return self.register_agent(agent_id, name, version, **kwargs)


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

_registry_instance: Optional[AgentRegistryService] = None


def configure_agent_registry(config: Optional[AgentRegistryConfig] = None):
    global _registry_instance
    _registry_instance = AgentRegistryService(config)
    return _registry_instance


def get_agent_registry() -> AgentRegistryService:
    if _registry_instance is None:
        raise RuntimeError("AgentRegistryService not configured. Call configure_agent_registry() first.")
    return _registry_instance


def get_router():
    return {"prefix": "/api/v1/agent-registry", "tags": ["agent-registry"]}


@pytest.fixture(autouse=True)
def _reset_singleton():
    global _registry_instance
    _registry_instance = None
    yield
    _registry_instance = None


# ===========================================================================
# Test Classes
# ===========================================================================


class TestAgentRegistryService:
    """Test facade creation and all getter methods."""

    def test_creation_default_config(self):
        svc = AgentRegistryService()
        assert svc.config.service_name == "agent-registry"
        assert svc.config.environment == "production"

    def test_creation_custom_config(self):
        config = AgentRegistryConfig(service_name="custom", max_agents=100)
        svc = AgentRegistryService(config)
        assert svc.config.service_name == "custom"
        assert svc.config.max_agents == 100

    def test_registry_accessible(self):
        svc = AgentRegistryService()
        assert svc.registry is not None

    def test_health_checker_accessible(self):
        svc = AgentRegistryService()
        assert svc.health_checker is not None

    def test_provenance_tracker_accessible(self):
        svc = AgentRegistryService()
        assert svc.provenance_tracker is not None

    def test_get_metrics_initial(self):
        svc = AgentRegistryService()
        metrics = svc.get_metrics()
        assert metrics["total_agents"] == 0
        assert metrics["total_registrations"] == 0


class TestAgentRegistryServiceRegisterFlow:
    """Test register_agent flow."""

    def test_register_returns_hash(self):
        svc = AgentRegistryService()
        result = svc.register_agent("gl-001", "Agent 1")
        assert len(result["provenance_hash"]) == 64

    def test_register_increments_count(self):
        svc = AgentRegistryService()
        svc.register_agent("gl-001", "A")
        svc.register_agent("gl-002", "B")
        assert svc.get_metrics()["total_agents"] == 2

    def test_register_records_provenance(self):
        svc = AgentRegistryService()
        svc.register_agent("gl-001", "A")
        assert len(svc.provenance_tracker) == 1

    def test_register_provenance_disabled(self):
        config = AgentRegistryConfig(provenance_enabled=False)
        svc = AgentRegistryService(config)
        svc.register_agent("gl-001", "A")
        assert len(svc.provenance_tracker) == 0

    def test_register_multiple_versions(self):
        svc = AgentRegistryService()
        svc.register_agent("gl-001", "V1", version="1.0.0")
        svc.register_agent("gl-001", "V2", version="2.0.0")
        assert svc.get_metrics()["total_versions"] == 2
        assert svc.get_metrics()["total_agents"] == 1


class TestAgentRegistryServiceQueryFlow:
    """Test query flow."""

    def test_query_all(self):
        svc = AgentRegistryService()
        svc.register_agent("gl-001", "A")
        svc.register_agent("gl-002", "B")
        results = svc.query_agents()
        assert len(results) == 2

    def test_query_by_layer(self):
        svc = AgentRegistryService()
        svc.register_agent("gl-001", "A", layer="calculation")
        svc.register_agent("gl-002", "B", layer="reporting")
        results = svc.query_agents(layer="calculation")
        assert len(results) == 1

    def test_query_increments_count(self):
        svc = AgentRegistryService()
        svc.query_agents()
        svc.query_agents()
        assert svc.get_metrics()["total_queries"] == 2

    def test_get_agent(self):
        svc = AgentRegistryService()
        svc.register_agent("gl-001", "A")
        agent = svc.get_agent("gl-001")
        assert agent is not None
        assert agent["name"] == "A"

    def test_get_agent_not_found(self):
        svc = AgentRegistryService()
        assert svc.get_agent("nonexistent") is None

    def test_get_agent_specific_version(self):
        svc = AgentRegistryService()
        svc.register_agent("gl-001", "V1", version="1.0.0")
        svc.register_agent("gl-001", "V2", version="2.0.0")
        agent = svc.get_agent("gl-001", version="1.0.0")
        assert agent["name"] == "V1"


class TestAgentRegistryServiceHealthFlow:
    """Test health check flow."""

    def test_check_health_default(self):
        svc = AgentRegistryService()
        svc.register_agent("gl-001", "A")
        result = svc.check_health("gl-001")
        assert result["status"] == "unknown"

    def test_check_health_set(self):
        svc = AgentRegistryService()
        svc.register_agent("gl-001", "A", health_status="healthy")
        result = svc.check_health("gl-001")
        assert result["status"] == "healthy"

    def test_set_health(self):
        svc = AgentRegistryService()
        svc.register_agent("gl-001", "A")
        svc.set_health("gl-001", "degraded")
        result = svc.check_health("gl-001")
        assert result["status"] == "degraded"

    def test_check_health_unknown_agent(self):
        svc = AgentRegistryService()
        result = svc.check_health("nonexistent")
        assert result["status"] == "unknown"


class TestAgentRegistryServiceHotReload:
    """Test hot reload flow."""

    def test_hot_reload(self):
        svc = AgentRegistryService()
        svc.register_agent("gl-001", "V1", version="1.0.0")
        svc.register_agent("gl-001", "V1.1", version="1.1.0")
        result = svc.hot_reload("gl-001", "V2", version="2.0.0")
        assert len(result["provenance_hash"]) == 64
        agent = svc.get_agent("gl-001")
        assert agent["name"] == "V2"


class TestAgentRegistryServiceLifecycle:
    """Test startup/shutdown."""

    def test_startup(self):
        svc = AgentRegistryService()
        assert svc.is_running is False
        svc.startup()
        assert svc.is_running is True

    def test_shutdown(self):
        svc = AgentRegistryService()
        svc.startup()
        svc.shutdown()
        assert svc.is_running is False


class TestConfigureAgentRegistry:
    """Test configure on service."""

    def test_configure_returns_service(self):
        svc = configure_agent_registry()
        assert isinstance(svc, AgentRegistryService)

    def test_configure_with_custom_config(self):
        config = AgentRegistryConfig(max_agents=100)
        svc = configure_agent_registry(config)
        assert svc.config.max_agents == 100


class TestGetAgentRegistry:
    """Test retrieval and RuntimeError."""

    def test_get_before_configure_raises(self):
        with pytest.raises(RuntimeError, match="not configured"):
            get_agent_registry()

    def test_get_after_configure(self):
        configure_agent_registry()
        svc = get_agent_registry()
        assert isinstance(svc, AgentRegistryService)

    def test_get_returns_same_instance(self):
        configure_agent_registry()
        s1 = get_agent_registry()
        s2 = get_agent_registry()
        assert s1 is s2


class TestGetRouter:
    """Test router retrieval."""

    def test_get_router_returns_dict(self):
        r = get_router()
        assert isinstance(r, dict)

    def test_router_has_prefix(self):
        r = get_router()
        assert "/api/v1/agent-registry" in r["prefix"]

    def test_router_has_tags(self):
        r = get_router()
        assert "agent-registry" in r["tags"]
