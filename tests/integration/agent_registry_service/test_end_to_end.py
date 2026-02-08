# -*- coding: utf-8 -*-
"""
End-to-End Integration Tests for Agent Registry Service (AGENT-FOUND-007)

Tests full workflows: register -> query -> health check -> dependency resolve
-> hot reload -> export/import -> provenance chain integrity.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import pytest


# ---------------------------------------------------------------------------
# Self-contained implementations for end-to-end testing
# ---------------------------------------------------------------------------


class E2EAgentRegistryService:
    """Full-stack agent registry for E2E testing."""

    def __init__(self):
        self._agents: Dict[str, Dict[str, Dict]] = {}
        self._health: Dict[str, str] = {}
        self._capabilities: Dict[str, List[str]] = defaultdict(list)
        self._provenance: Dict[str, List[Dict]] = {}
        self._dependencies: Dict[str, List[Dict]] = {}

    def register(self, agent_id: str, name: str, version: str = "1.0.0",
                 layer: str = "utility", sectors: Optional[List[str]] = None,
                 capabilities: Optional[List[str]] = None,
                 dependencies: Optional[List[Dict]] = None,
                 health_status: str = "unknown",
                 tags: Optional[List[str]] = None) -> Dict[str, Any]:
        if agent_id not in self._agents:
            self._agents[agent_id] = {}
        entry = {
            "agent_id": agent_id, "name": name, "version": version,
            "layer": layer, "sectors": sectors or [],
            "capabilities": capabilities or [],
            "dependencies": dependencies or [],
            "health_status": health_status,
            "tags": tags or [],
            "registered_at": datetime.utcnow().isoformat(),
        }
        h = hashlib.sha256(json.dumps(entry, sort_keys=True, default=str).encode()).hexdigest()
        entry["provenance_hash"] = h
        self._agents[agent_id][version] = entry
        self._health[agent_id] = health_status
        for cap in (capabilities or []):
            if agent_id not in self._capabilities[cap]:
                self._capabilities[cap].append(agent_id)
        self._dependencies[agent_id] = dependencies or []
        self._record_provenance(agent_id, "register", h)
        return {"agent_id": agent_id, "version": version, "provenance_hash": h}

    def unregister(self, agent_id: str) -> bool:
        if agent_id not in self._agents:
            return False
        del self._agents[agent_id]
        self._health.pop(agent_id, None)
        self._record_provenance(agent_id, "unregister", "")
        return True

    def get(self, agent_id: str, version: Optional[str] = None) -> Optional[Dict]:
        if agent_id not in self._agents:
            return None
        if version:
            return self._agents[agent_id].get(version)
        versions = self._agents[agent_id]
        return versions[max(versions.keys())] if versions else None

    def query(self, layer: Optional[str] = None, sector: Optional[str] = None,
              capability: Optional[str] = None, tag: Optional[str] = None,
              search: Optional[str] = None) -> List[Dict]:
        results = []
        for aid, versions in self._agents.items():
            entry = versions[max(versions.keys())]
            if layer and entry.get("layer") != layer:
                continue
            if sector and sector not in entry.get("sectors", []):
                continue
            if capability and capability not in entry.get("capabilities", []):
                continue
            if tag and tag not in entry.get("tags", []):
                continue
            if search and search.lower() not in entry.get("name", "").lower():
                continue
            results.append(entry)
        return results

    def list_versions(self, agent_id: str) -> List[str]:
        if agent_id not in self._agents:
            return []
        return sorted(self._agents[agent_id].keys())

    def check_health(self, agent_id: str) -> str:
        return self._health.get(agent_id, "unknown")

    def set_health(self, agent_id: str, status: str) -> None:
        self._health[agent_id] = status

    def hot_reload(self, agent_id: str, name: str, version: str = "1.0.0",
                   **kwargs) -> Dict[str, Any]:
        if agent_id in self._agents:
            self._agents[agent_id] = {}
        result = self.register(agent_id, name, version, **kwargs)
        self._record_provenance(agent_id, "hot_reload", result["provenance_hash"])
        return result

    def resolve_dependencies(self, agent_id: str) -> List[str]:
        """Simple DFS-based resolution."""
        visited: Set[str] = set()
        order: List[str] = []

        def _visit(aid: str):
            if aid in visited:
                return
            visited.add(aid)
            for dep in self._dependencies.get(aid, []):
                dep_id = dep.get("agent_id") if isinstance(dep, dict) else dep
                if dep_id in self._agents:
                    _visit(dep_id)
            order.append(aid)

        _visit(agent_id)
        return order

    def discover_by_capability(self, capability: str) -> List[str]:
        return self._capabilities.get(capability, [])

    def export_registry(self) -> Dict[str, Any]:
        return {aid: dict(v) for aid, v in self._agents.items()}

    def import_registry(self, data: Dict[str, Any]) -> int:
        imported = 0
        for aid, versions in data.items():
            for v, entry in versions.items():
                self.register(
                    agent_id=entry["agent_id"], name=entry["name"],
                    version=entry["version"], layer=entry.get("layer", "utility"),
                    sectors=entry.get("sectors", []),
                    capabilities=entry.get("capabilities", []),
                    tags=entry.get("tags", []),
                )
                imported += 1
        return imported

    def get_statistics(self) -> Dict[str, Any]:
        by_layer = defaultdict(int)
        by_health = defaultdict(int)
        for aid, versions in self._agents.items():
            entry = versions[max(versions.keys())]
            by_layer[entry.get("layer", "utility")] += 1
            by_health[self._health.get(aid, "unknown")] += 1
        return {
            "total_agents": len(self._agents),
            "total_versions": sum(len(v) for v in self._agents.values()),
            "by_layer": dict(by_layer),
            "by_health": dict(by_health),
        }

    def verify_provenance(self, agent_id: str) -> Dict[str, Any]:
        chain = self._provenance.get(agent_id, [])
        if not chain:
            return {"valid": False, "error": "Not found"}
        for i in range(1, len(chain)):
            if chain[i]["previous_hash"] != chain[i - 1]["chain_hash"]:
                return {"valid": False, "error": f"Break at entry {i}"}
        return {"valid": True, "entries": len(chain)}

    def _record_provenance(self, entity_id, action, data_hash):
        chain = self._provenance.get(entity_id, [])
        prev = chain[-1]["chain_hash"] if chain else ""
        entry = {
            "entity_id": entity_id, "action": action, "data_hash": data_hash,
            "previous_hash": prev,
            "chain_hash": hashlib.sha256(
                f"{entity_id}:{action}:{data_hash}:{prev}".encode()
            ).hexdigest(),
        }
        chain.append(entry)
        self._provenance[entity_id] = chain


# ===========================================================================
# Test Classes
# ===========================================================================


class TestRegistrationWorkflow:
    """Test register -> get -> query -> unregister."""

    def test_full_lifecycle(self):
        svc = E2EAgentRegistryService()
        result = svc.register("gl-001", "Carbon Calc", version="2.1.0",
                               layer="calculation", sectors=["energy"],
                               capabilities=["carbon_calc"], tags=["carbon"])
        assert len(result["provenance_hash"]) == 64

        agent = svc.get("gl-001")
        assert agent["name"] == "Carbon Calc"

        results = svc.query(layer="calculation")
        assert len(results) == 1

        assert svc.unregister("gl-001") is True
        assert svc.get("gl-001") is None

    def test_multiple_agents_lifecycle(self):
        svc = E2EAgentRegistryService()
        for i in range(10):
            svc.register(f"gl-{i:03d}", f"Agent {i}", layer="calculation")
        assert len(svc.query()) == 10
        svc.unregister("gl-000")
        assert len(svc.query()) == 9


class TestVersionManagement:
    """Test multi-version agent management."""

    def test_register_multiple_versions(self):
        svc = E2EAgentRegistryService()
        svc.register("gl-001", "V1", version="1.0.0")
        svc.register("gl-001", "V2", version="2.0.0")
        svc.register("gl-001", "V1.5", version="1.5.0")
        versions = svc.list_versions("gl-001")
        assert versions == ["1.0.0", "1.5.0", "2.0.0"]

    def test_get_specific_version(self):
        svc = E2EAgentRegistryService()
        svc.register("gl-001", "V1", version="1.0.0")
        svc.register("gl-001", "V2", version="2.0.0")
        agent = svc.get("gl-001", version="1.0.0")
        assert agent["name"] == "V1"

    def test_get_latest_version(self):
        svc = E2EAgentRegistryService()
        svc.register("gl-001", "V1", version="1.0.0")
        svc.register("gl-001", "V2", version="2.0.0")
        agent = svc.get("gl-001")
        assert agent["name"] == "V2"


class TestHealthCheckWorkflow:
    """Test health check lifecycle."""

    def test_initial_health_unknown(self):
        svc = E2EAgentRegistryService()
        svc.register("gl-001", "A")
        assert svc.check_health("gl-001") == "unknown"

    def test_set_and_check_health(self):
        svc = E2EAgentRegistryService()
        svc.register("gl-001", "A")
        svc.set_health("gl-001", "healthy")
        assert svc.check_health("gl-001") == "healthy"

    def test_health_transitions(self):
        svc = E2EAgentRegistryService()
        svc.register("gl-001", "A")
        for status in ["healthy", "degraded", "unhealthy", "healthy"]:
            svc.set_health("gl-001", status)
            assert svc.check_health("gl-001") == status


class TestDependencyWorkflow:
    """Test dependency resolution workflow."""

    def test_resolve_simple_chain(self):
        svc = E2EAgentRegistryService()
        svc.register("gl-003", "C")
        svc.register("gl-002", "B", dependencies=[{"agent_id": "gl-003"}])
        svc.register("gl-001", "A", dependencies=[{"agent_id": "gl-002"}])
        order = svc.resolve_dependencies("gl-001")
        assert order.index("gl-003") < order.index("gl-002")
        assert order.index("gl-002") < order.index("gl-001")

    def test_resolve_diamond(self):
        svc = E2EAgentRegistryService()
        svc.register("gl-004", "D")
        svc.register("gl-003", "C", dependencies=[{"agent_id": "gl-004"}])
        svc.register("gl-002", "B", dependencies=[{"agent_id": "gl-004"}])
        svc.register("gl-001", "A", dependencies=[
            {"agent_id": "gl-002"}, {"agent_id": "gl-003"},
        ])
        order = svc.resolve_dependencies("gl-001")
        assert order[-1] == "gl-001"
        assert order.index("gl-004") < order.index("gl-002")


class TestHotReloadWorkflow:
    """Test hot reload workflow."""

    def test_hot_reload_replaces(self):
        svc = E2EAgentRegistryService()
        svc.register("gl-001", "V1", version="1.0.0")
        svc.register("gl-001", "V1.1", version="1.1.0")
        assert len(svc.list_versions("gl-001")) == 2

        svc.hot_reload("gl-001", "V2", version="2.0.0")
        assert len(svc.list_versions("gl-001")) == 1
        assert svc.get("gl-001")["name"] == "V2"


class TestCapabilityDiscovery:
    """Test capability discovery workflow."""

    def test_discover_agents(self):
        svc = E2EAgentRegistryService()
        svc.register("gl-001", "A", capabilities=["carbon_calc"])
        svc.register("gl-002", "B", capabilities=["carbon_calc", "cbam_report"])
        svc.register("gl-003", "C", capabilities=["data_validate"])
        agents = svc.discover_by_capability("carbon_calc")
        assert set(agents) == {"gl-001", "gl-002"}


class TestExportImportWorkflow:
    """Test export/import round-trip."""

    def test_round_trip(self):
        svc1 = E2EAgentRegistryService()
        svc1.register("gl-001", "A", layer="calculation", sectors=["energy"])
        svc1.register("gl-002", "B", layer="reporting")
        data = svc1.export_registry()

        svc2 = E2EAgentRegistryService()
        imported = svc2.import_registry(data)
        assert imported == 2
        assert svc2.get("gl-001")["name"] == "A"


class TestStatisticsWorkflow:
    """Test statistics aggregation."""

    def test_statistics(self):
        svc = E2EAgentRegistryService()
        svc.register("gl-001", "A", layer="calculation", health_status="healthy")
        svc.register("gl-002", "B", layer="reporting", health_status="degraded")
        svc.register("gl-003", "C", layer="calculation", health_status="healthy")
        stats = svc.get_statistics()
        assert stats["total_agents"] == 3
        assert stats["by_layer"]["calculation"] == 2
        assert stats["by_health"]["healthy"] == 2


class TestProvenanceWorkflow:
    """Test provenance chain integrity."""

    def test_provenance_on_register(self):
        svc = E2EAgentRegistryService()
        svc.register("gl-001", "A")
        result = svc.verify_provenance("gl-001")
        assert result["valid"] is True
        assert result["entries"] == 1

    def test_provenance_hash_is_sha256(self):
        svc = E2EAgentRegistryService()
        h = svc.register("gl-001", "A")["provenance_hash"]
        assert len(h) == 64
        assert re.match(r"^[0-9a-f]{64}$", h)

    def test_provenance_chain_after_updates(self):
        svc = E2EAgentRegistryService()
        svc.register("gl-001", "V1", version="1.0.0")
        svc.register("gl-001", "V2", version="2.0.0")
        result = svc.verify_provenance("gl-001")
        assert result["valid"] is True
        assert result["entries"] == 2

    def test_nonexistent_provenance(self):
        svc = E2EAgentRegistryService()
        result = svc.verify_provenance("nope")
        assert result["valid"] is False

    def test_hot_reload_extends_provenance(self):
        svc = E2EAgentRegistryService()
        svc.register("gl-001", "V1")
        svc.hot_reload("gl-001", "V2")
        result = svc.verify_provenance("gl-001")
        assert result["valid"] is True
        # register + hot_reload (which does register + hot_reload provenance entries)
        assert result["entries"] >= 2


class TestQueryFilters:
    """Test query with various filters."""

    def test_query_by_sector(self):
        svc = E2EAgentRegistryService()
        svc.register("gl-001", "A", sectors=["energy"])
        svc.register("gl-002", "B", sectors=["manufacturing"])
        results = svc.query(sector="energy")
        assert len(results) == 1

    def test_query_by_capability(self):
        svc = E2EAgentRegistryService()
        svc.register("gl-001", "A", capabilities=["calc"])
        svc.register("gl-002", "B", capabilities=["report"])
        results = svc.query(capability="calc")
        assert len(results) == 1

    def test_query_by_tag(self):
        svc = E2EAgentRegistryService()
        svc.register("gl-001", "A", tags=["carbon"])
        results = svc.query(tag="carbon")
        assert len(results) == 1

    def test_query_by_search(self):
        svc = E2EAgentRegistryService()
        svc.register("gl-001", "Carbon Calculator")
        results = svc.query(search="Carbon")
        assert len(results) == 1
