# -*- coding: utf-8 -*-
"""
Registry Workflow Integration Tests for Agent Registry Service (AGENT-FOUND-007)

Tests CRUD lifecycle, version management, GLIP migration, capability discovery,
and dependency resolution workflows.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import pytest


# ---------------------------------------------------------------------------
# Self-contained workflow service
# ---------------------------------------------------------------------------


class WorkflowRegistryService:
    """Agent registry service for workflow testing."""

    def __init__(self):
        self._agents: Dict[str, Dict[str, Dict]] = {}
        self._health: Dict[str, str] = {}
        self._capabilities: Dict[str, List[str]] = defaultdict(list)
        self._dependencies: Dict[str, List[str]] = {}

    def register(self, agent_id, name, version="1.0.0", layer="utility",
                 execution_mode="glip_v1", sectors=None, capabilities=None,
                 dependencies=None, health_status="unknown",
                 tags=None, idempotency="none") -> str:
        if agent_id not in self._agents:
            self._agents[agent_id] = {}
        entry = {
            "agent_id": agent_id, "name": name, "version": version,
            "layer": layer, "execution_mode": execution_mode,
            "sectors": sectors or [], "capabilities": capabilities or [],
            "dependencies": dependencies or [],
            "health_status": health_status, "tags": tags or [],
            "idempotency_support": idempotency,
        }
        h = hashlib.sha256(json.dumps(entry, sort_keys=True, default=str).encode()).hexdigest()
        entry["provenance_hash"] = h
        self._agents[agent_id][version] = entry
        self._health[agent_id] = health_status
        for cap in (capabilities or []):
            if agent_id not in self._capabilities[cap]:
                self._capabilities[cap].append(agent_id)
        self._dependencies[agent_id] = dependencies or []
        return h

    def unregister(self, agent_id):
        if agent_id in self._agents:
            del self._agents[agent_id]
            return True
        return False

    def get(self, agent_id, version=None):
        if agent_id not in self._agents:
            return None
        if version:
            return self._agents[agent_id].get(version)
        v = self._agents[agent_id]
        return v[max(v.keys())] if v else None

    def query(self, **kwargs):
        results = []
        for aid, versions in self._agents.items():
            entry = versions[max(versions.keys())]
            if kwargs.get("layer") and entry["layer"] != kwargs["layer"]:
                continue
            if kwargs.get("execution_mode") and entry["execution_mode"] != kwargs["execution_mode"]:
                continue
            if kwargs.get("sector") and kwargs["sector"] not in entry.get("sectors", []):
                continue
            if kwargs.get("capability") and kwargs["capability"] not in entry.get("capabilities", []):
                continue
            if kwargs.get("health") and self._health.get(aid) != kwargs["health"]:
                continue
            results.append(entry)
        return results

    def list_versions(self, agent_id):
        if agent_id not in self._agents:
            return []
        return sorted(self._agents[agent_id].keys())

    def update(self, agent_id, version, **kwargs):
        if agent_id not in self._agents or version not in self._agents[agent_id]:
            return None
        entry = self._agents[agent_id][version]
        entry.update(kwargs)
        h = hashlib.sha256(json.dumps(entry, sort_keys=True, default=str).encode()).hexdigest()
        entry["provenance_hash"] = h
        return h

    def hot_reload(self, agent_id, name, version="1.0.0", **kwargs):
        if agent_id in self._agents:
            self._agents[agent_id] = {}
        return self.register(agent_id, name, version, **kwargs)

    def set_health(self, agent_id, status):
        self._health[agent_id] = status

    def discover(self, capability):
        return self._capabilities.get(capability, [])

    def resolve_deps(self, agent_id, visited=None):
        if visited is None:
            visited = set()
        if agent_id in visited:
            return []
        visited.add(agent_id)
        order = []
        for dep_id in self._dependencies.get(agent_id, []):
            if dep_id in self._agents:
                order.extend(self.resolve_deps(dep_id, visited))
        order.append(agent_id)
        return order

    def get_statistics(self):
        by_mode = defaultdict(int)
        for aid, versions in self._agents.items():
            entry = versions[max(versions.keys())]
            by_mode[entry["execution_mode"]] += 1
        return {"total": len(self._agents), "by_mode": dict(by_mode)}


# ===========================================================================
# Test Classes
# ===========================================================================


class TestCRUDLifecycle:
    """Test full CRUD lifecycle."""

    def test_create_read_update_delete(self):
        svc = WorkflowRegistryService()
        h = svc.register("gl-001", "Agent 1")
        assert len(h) == 64

        agent = svc.get("gl-001")
        assert agent["name"] == "Agent 1"

        h2 = svc.update("gl-001", "1.0.0", name="Agent 1 Updated")
        assert h2 != h

        assert svc.unregister("gl-001") is True
        assert svc.get("gl-001") is None

    def test_register_10_agents(self):
        svc = WorkflowRegistryService()
        for i in range(10):
            svc.register(f"gl-{i:03d}", f"Agent {i}")
        assert len(svc.query()) == 10

    def test_unregister_nonexistent(self):
        svc = WorkflowRegistryService()
        assert svc.unregister("nonexistent") is False

    def test_update_nonexistent(self):
        svc = WorkflowRegistryService()
        assert svc.update("nonexistent", "1.0.0", name="X") is None


class TestVersionManagementWorkflow:
    """Test version management."""

    def test_multi_version_register(self):
        svc = WorkflowRegistryService()
        svc.register("gl-001", "V1", version="1.0.0")
        svc.register("gl-001", "V2", version="2.0.0")
        svc.register("gl-001", "V3", version="3.0.0")
        assert svc.list_versions("gl-001") == ["1.0.0", "2.0.0", "3.0.0"]

    def test_get_latest_version(self):
        svc = WorkflowRegistryService()
        svc.register("gl-001", "V1", version="1.0.0")
        svc.register("gl-001", "V2", version="2.0.0")
        assert svc.get("gl-001")["name"] == "V2"

    def test_get_specific_version(self):
        svc = WorkflowRegistryService()
        svc.register("gl-001", "V1", version="1.0.0")
        svc.register("gl-001", "V2", version="2.0.0")
        assert svc.get("gl-001", "1.0.0")["name"] == "V1"

    def test_hot_reload_clears_versions(self):
        svc = WorkflowRegistryService()
        svc.register("gl-001", "V1", version="1.0.0")
        svc.register("gl-001", "V2", version="2.0.0")
        svc.hot_reload("gl-001", "V3", version="3.0.0")
        assert svc.list_versions("gl-001") == ["3.0.0"]

    def test_version_ordering(self):
        svc = WorkflowRegistryService()
        for v in ["2.0.0", "1.0.0", "3.0.0", "1.5.0"]:
            svc.register("gl-001", f"V{v}", version=v)
        versions = svc.list_versions("gl-001")
        assert versions == ["1.0.0", "1.5.0", "2.0.0", "3.0.0"]


class TestGLIPMigrationWorkflow:
    """Test GLIP migration workflow (legacy -> hybrid -> glip_v1)."""

    def test_legacy_to_hybrid(self):
        svc = WorkflowRegistryService()
        svc.register("gl-legacy-001", "Legacy Agent", execution_mode="legacy_http")
        agent = svc.get("gl-legacy-001")
        assert agent["execution_mode"] == "legacy_http"

        svc.update("gl-legacy-001", "1.0.0", execution_mode="hybrid")
        agent = svc.get("gl-legacy-001")
        assert agent["execution_mode"] == "hybrid"

    def test_hybrid_to_glip(self):
        svc = WorkflowRegistryService()
        svc.register("gl-001", "Hybrid Agent", execution_mode="hybrid")
        svc.update("gl-001", "1.0.0", execution_mode="glip_v1")
        agent = svc.get("gl-001")
        assert agent["execution_mode"] == "glip_v1"

    def test_query_by_execution_mode(self):
        svc = WorkflowRegistryService()
        svc.register("gl-001", "A", execution_mode="glip_v1")
        svc.register("gl-002", "B", execution_mode="legacy_http")
        svc.register("gl-003", "C", execution_mode="hybrid")
        glip = svc.query(execution_mode="glip_v1")
        assert len(glip) == 1
        legacy = svc.query(execution_mode="legacy_http")
        assert len(legacy) == 1

    def test_migration_statistics(self):
        svc = WorkflowRegistryService()
        svc.register("gl-001", "A", execution_mode="glip_v1")
        svc.register("gl-002", "B", execution_mode="glip_v1")
        svc.register("gl-003", "C", execution_mode="legacy_http")
        stats = svc.get_statistics()
        assert stats["by_mode"]["glip_v1"] == 2
        assert stats["by_mode"]["legacy_http"] == 1


class TestCapabilityDiscoveryWorkflow:
    """Test capability discovery."""

    def test_discover_single(self):
        svc = WorkflowRegistryService()
        svc.register("gl-001", "A", capabilities=["carbon_calc"])
        assert "gl-001" in svc.discover("carbon_calc")

    def test_discover_multiple(self):
        svc = WorkflowRegistryService()
        svc.register("gl-001", "A", capabilities=["carbon_calc"])
        svc.register("gl-002", "B", capabilities=["carbon_calc", "cbam_report"])
        assert set(svc.discover("carbon_calc")) == {"gl-001", "gl-002"}

    def test_discover_no_match(self):
        svc = WorkflowRegistryService()
        assert svc.discover("nonexistent") == []

    def test_query_by_capability(self):
        svc = WorkflowRegistryService()
        svc.register("gl-001", "A", capabilities=["calc"])
        svc.register("gl-002", "B", capabilities=["report"])
        results = svc.query(capability="calc")
        assert len(results) == 1

    def test_query_by_sector(self):
        svc = WorkflowRegistryService()
        svc.register("gl-001", "A", sectors=["energy"])
        svc.register("gl-002", "B", sectors=["manufacturing"])
        results = svc.query(sector="energy")
        assert len(results) == 1


class TestDependencyResolutionWorkflow:
    """Test dependency resolution."""

    def test_simple_chain(self):
        svc = WorkflowRegistryService()
        svc.register("gl-003", "C")
        svc.register("gl-002", "B", dependencies=["gl-003"])
        svc.register("gl-001", "A", dependencies=["gl-002"])
        order = svc.resolve_deps("gl-001")
        assert order == ["gl-003", "gl-002", "gl-001"]

    def test_diamond_resolution(self):
        svc = WorkflowRegistryService()
        svc.register("gl-004", "D")
        svc.register("gl-003", "C", dependencies=["gl-004"])
        svc.register("gl-002", "B", dependencies=["gl-004"])
        svc.register("gl-001", "A", dependencies=["gl-002", "gl-003"])
        order = svc.resolve_deps("gl-001")
        assert order[-1] == "gl-001"
        assert order.index("gl-004") < order.index("gl-002")

    def test_missing_dependency_skipped(self):
        svc = WorkflowRegistryService()
        svc.register("gl-001", "A", dependencies=["gl-missing"])
        order = svc.resolve_deps("gl-001")
        assert order == ["gl-001"]

    def test_no_dependencies(self):
        svc = WorkflowRegistryService()
        svc.register("gl-001", "A")
        order = svc.resolve_deps("gl-001")
        assert order == ["gl-001"]


class TestHealthWorkflow:
    """Test health management workflows."""

    def test_health_lifecycle(self):
        svc = WorkflowRegistryService()
        svc.register("gl-001", "A", health_status="healthy")
        assert svc.query(health="healthy")[0]["agent_id"] == "gl-001"

        svc.set_health("gl-001", "degraded")
        assert len(svc.query(health="healthy")) == 0

    def test_multiple_health_states(self):
        svc = WorkflowRegistryService()
        svc.register("gl-001", "A", health_status="healthy")
        svc.register("gl-002", "B", health_status="degraded")
        svc.register("gl-003", "C", health_status="unhealthy")
        healthy = svc.query(health="healthy")
        assert len(healthy) == 1

    def test_provenance_hash_changes_on_update(self):
        svc = WorkflowRegistryService()
        h1 = svc.register("gl-001", "V1")
        h2 = svc.update("gl-001", "1.0.0", name="V2")
        assert h1 != h2
        assert len(h2) == 64
