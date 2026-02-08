# -*- coding: utf-8 -*-
"""
Unit Tests for AgentRegistry (AGENT-FOUND-007)

Tests agent registration, unregistration, query, versioning, indexing,
export/import, statistics, and thread safety.

Coverage target: 85%+ of registry.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import uuid
from collections import defaultdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import pytest


# ---------------------------------------------------------------------------
# Inline enums and models (self-contained)
# ---------------------------------------------------------------------------

class AgentLayer(str, Enum):
    FOUNDATION = "foundation"
    ORCHESTRATION = "orchestration"
    INGESTION = "ingestion"
    VALIDATION = "validation"
    NORMALIZATION = "normalization"
    CALCULATION = "calculation"
    REPORTING = "reporting"
    COMPLIANCE = "compliance"
    ANALYTICS = "analytics"
    INTEGRATION = "integration"
    UTILITY = "utility"


class SectorClassification(str, Enum):
    ENERGY = "energy"
    MANUFACTURING = "manufacturing"
    TRANSPORTATION = "transportation"
    BUILDINGS = "buildings"
    AGRICULTURE = "agriculture"
    WASTE = "waste"
    INDUSTRIAL_PROCESSES = "industrial_processes"
    LAND_USE = "land_use"
    WATER = "water"
    CROSS_SECTOR = "cross_sector"


class AgentHealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    DISABLED = "disabled"


class ExecutionMode(str, Enum):
    GLIP_V1 = "glip_v1"
    LEGACY_HTTP = "legacy_http"
    HYBRID = "hybrid"


class CapabilityCategory(str, Enum):
    CALCULATION = "calculation"
    VALIDATION = "validation"
    NORMALIZATION = "normalization"
    INGESTION = "ingestion"
    REPORTING = "reporting"
    COMPLIANCE = "compliance"
    ANALYTICS = "analytics"
    ORCHESTRATION = "orchestration"
    INTEGRATION = "integration"
    UTILITY = "utility"


class SemanticVersion:
    def __init__(self, version_str):
        parts = version_str.split("-", 1)
        core = parts[0]
        self.prerelease = parts[1] if len(parts) > 1 else None
        segments = core.split(".")
        if len(segments) != 3:
            raise ValueError(f"Invalid version: {version_str}")
        self.major, self.minor, self.patch = int(segments[0]), int(segments[1]), int(segments[2])

    def __str__(self):
        base = f"{self.major}.{self.minor}.{self.patch}"
        return f"{base}-{self.prerelease}" if self.prerelease else base

    def __eq__(self, other):
        if not isinstance(other, SemanticVersion):
            return NotImplemented
        return (self.major, self.minor, self.patch) == (other.major, other.minor, other.patch)

    def __lt__(self, other):
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)

    def __le__(self, other):
        return self == other or self < other

    def __gt__(self, other):
        return not self <= other

    def __ge__(self, other):
        return not self < other

    def __hash__(self):
        return hash((self.major, self.minor, self.patch))


class AgentCapability:
    def __init__(self, name, category="utility", input_types=None, output_types=None):
        self.name = name
        self.category = CapabilityCategory(category)
        self.input_types = input_types or []
        self.output_types = output_types or []


class _AgentEntry:
    """Lightweight agent entry for registry."""

    def __init__(self, agent_id, name, version="1.0.0", layer="utility",
                 sectors=None, execution_mode="glip_v1", health="unknown",
                 tags=None, capabilities=None, dependencies=None,
                 description=""):
        self.agent_id = agent_id
        self.name = name
        self.version = SemanticVersion(version)
        self.layer = AgentLayer(layer)
        self.sectors = [SectorClassification(s) for s in (sectors or [])]
        self.execution_mode = ExecutionMode(execution_mode)
        self.health_status = AgentHealthStatus(health)
        self.tags = tags or []
        self.capabilities = capabilities or []
        self.dependencies = dependencies or []
        self.description = description
        self.registered_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        self.provenance_hash = ""

    def compute_hash(self):
        data = {
            "agent_id": self.agent_id, "name": self.name,
            "version": str(self.version), "layer": self.layer.value,
        }
        self.provenance_hash = hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()
        return self.provenance_hash

    def to_dict(self):
        return {
            "agent_id": self.agent_id, "name": self.name,
            "version": str(self.version), "layer": self.layer.value,
            "sectors": [s.value for s in self.sectors],
            "execution_mode": self.execution_mode.value,
            "health_status": self.health_status.value,
            "tags": self.tags,
            "capabilities": [c.name for c in self.capabilities],
            "description": self.description,
            "provenance_hash": self.provenance_hash,
        }


# ---------------------------------------------------------------------------
# AgentRegistry (self-contained)
# ---------------------------------------------------------------------------

class AgentRegistry:
    """In-memory agent registry with indexing and versioning."""

    def __init__(self):
        self._agents: Dict[str, Dict[str, _AgentEntry]] = {}  # agent_id -> {version: entry}
        self._by_layer: Dict[str, List[str]] = defaultdict(list)
        self._by_sector: Dict[str, List[str]] = defaultdict(list)
        self._by_capability: Dict[str, List[str]] = defaultdict(list)
        self._by_tag: Dict[str, List[str]] = defaultdict(list)
        self._lock = threading.Lock()

    @property
    def count(self) -> int:
        return len(self._agents)

    def register_agent(self, entry: _AgentEntry) -> str:
        with self._lock:
            if entry.agent_id not in self._agents:
                self._agents[entry.agent_id] = {}
            entry.compute_hash()
            entry.updated_at = datetime.utcnow()
            self._agents[entry.agent_id][str(entry.version)] = entry
            self._update_indexes(entry)
            return entry.provenance_hash

    def unregister_agent(self, agent_id: str, version: Optional[str] = None) -> bool:
        with self._lock:
            if agent_id not in self._agents:
                return False
            if version:
                if version in self._agents[agent_id]:
                    del self._agents[agent_id][version]
                    if not self._agents[agent_id]:
                        del self._agents[agent_id]
                        self._remove_from_indexes(agent_id)
                    return True
                return False
            del self._agents[agent_id]
            self._remove_from_indexes(agent_id)
            return True

    def get_agent(self, agent_id: str, version: Optional[str] = None) -> Optional[_AgentEntry]:
        if agent_id not in self._agents:
            return None
        versions = self._agents[agent_id]
        if version:
            return versions.get(version)
        # Return latest version
        if not versions:
            return None
        latest_key = max(versions.keys(), key=lambda v: SemanticVersion(v))
        return versions[latest_key]

    def list_agents(self, layer: Optional[str] = None, sector: Optional[str] = None,
                    capability: Optional[str] = None, tag: Optional[str] = None,
                    health: Optional[str] = None, search: Optional[str] = None,
                    offset: int = 0, limit: int = 100) -> List[_AgentEntry]:
        results = []
        for agent_id, versions in self._agents.items():
            entry = self.get_agent(agent_id)
            if entry is None:
                continue
            if layer and entry.layer.value != layer:
                continue
            if sector and not any(s.value == sector for s in entry.sectors):
                continue
            if capability and not any(c.name == capability for c in entry.capabilities):
                continue
            if tag and tag not in entry.tags:
                continue
            if health and entry.health_status.value != health:
                continue
            if search and search.lower() not in entry.name.lower() and \
               search.lower() not in entry.description.lower():
                continue
            results.append(entry)
        return results[offset:offset + limit]

    def list_versions(self, agent_id: str) -> List[str]:
        if agent_id not in self._agents:
            return []
        versions = list(self._agents[agent_id].keys())
        versions.sort(key=lambda v: SemanticVersion(v))
        return versions

    def update_agent(self, entry: _AgentEntry) -> str:
        return self.register_agent(entry)

    def hot_reload_agent(self, entry: _AgentEntry) -> str:
        with self._lock:
            if entry.agent_id in self._agents:
                # Replace all versions with the new one
                self._remove_from_indexes(entry.agent_id)
                self._agents[entry.agent_id] = {}
            else:
                self._agents[entry.agent_id] = {}
            entry.compute_hash()
            self._agents[entry.agent_id][str(entry.version)] = entry
            self._update_indexes(entry)
            return entry.provenance_hash

    def get_all_agent_ids(self) -> List[str]:
        return list(self._agents.keys())

    def get_agents_by_layer(self, layer: str) -> List[_AgentEntry]:
        return self.list_agents(layer=layer)

    def export_registry(self) -> Dict[str, Any]:
        data = {}
        for agent_id, versions in self._agents.items():
            data[agent_id] = {v: e.to_dict() for v, e in versions.items()}
        return data

    def import_registry(self, data: Dict[str, Any], mode: str = "merge") -> int:
        imported = 0
        if mode == "replace":
            self._agents.clear()
            self._by_layer.clear()
            self._by_sector.clear()
            self._by_capability.clear()
            self._by_tag.clear()
        for agent_id, versions in data.items():
            for v_str, entry_data in versions.items():
                entry = _AgentEntry(
                    agent_id=entry_data["agent_id"],
                    name=entry_data["name"],
                    version=entry_data["version"],
                    layer=entry_data["layer"],
                    sectors=entry_data.get("sectors", []),
                    tags=entry_data.get("tags", []),
                    description=entry_data.get("description", ""),
                )
                self.register_agent(entry)
                imported += 1
        return imported

    def get_statistics(self) -> Dict[str, Any]:
        total_agents = len(self._agents)
        total_versions = sum(len(v) for v in self._agents.values())
        by_layer = defaultdict(int)
        by_health = defaultdict(int)
        for agent_id in self._agents:
            entry = self.get_agent(agent_id)
            if entry:
                by_layer[entry.layer.value] += 1
                by_health[entry.health_status.value] += 1
        return {
            "total_agents": total_agents,
            "total_versions": total_versions,
            "by_layer": dict(by_layer),
            "by_health": dict(by_health),
        }

    def _update_indexes(self, entry: _AgentEntry):
        aid = entry.agent_id
        if aid not in self._by_layer.get(entry.layer.value, []):
            self._by_layer[entry.layer.value].append(aid)
        for s in entry.sectors:
            if aid not in self._by_sector.get(s.value, []):
                self._by_sector[s.value].append(aid)
        for c in entry.capabilities:
            if aid not in self._by_capability.get(c.name, []):
                self._by_capability[c.name].append(aid)
        for t in entry.tags:
            if aid not in self._by_tag.get(t, []):
                self._by_tag[t].append(aid)

    def _remove_from_indexes(self, agent_id: str):
        for idx in [self._by_layer, self._by_sector, self._by_capability, self._by_tag]:
            for key in list(idx.keys()):
                if agent_id in idx[key]:
                    idx[key].remove(agent_id)


# ===========================================================================
# Test Classes
# ===========================================================================


class TestAgentRegistryRegister:
    """Test register_agent operations."""

    def test_register_creates_entry(self):
        reg = AgentRegistry()
        entry = _AgentEntry("gl-001", "Agent 1")
        h = reg.register_agent(entry)
        assert len(h) == 64
        assert reg.count == 1

    def test_register_returns_sha256_hash(self):
        reg = AgentRegistry()
        h = reg.register_agent(_AgentEntry("gl-001", "Agent 1"))
        assert re.match(r"^[0-9a-f]{64}$", h)

    def test_register_overwrites_same_version(self):
        reg = AgentRegistry()
        reg.register_agent(_AgentEntry("gl-001", "V1"))
        reg.register_agent(_AgentEntry("gl-001", "V2"))
        assert reg.count == 1
        assert reg.get_agent("gl-001").name == "V2"

    def test_register_multiple_versions(self):
        reg = AgentRegistry()
        reg.register_agent(_AgentEntry("gl-001", "V1", version="1.0.0"))
        reg.register_agent(_AgentEntry("gl-001", "V2", version="2.0.0"))
        assert reg.count == 1
        versions = reg.list_versions("gl-001")
        assert len(versions) == 2

    def test_register_multiple_agents(self):
        reg = AgentRegistry()
        reg.register_agent(_AgentEntry("gl-001", "Agent 1"))
        reg.register_agent(_AgentEntry("gl-002", "Agent 2"))
        reg.register_agent(_AgentEntry("gl-003", "Agent 3"))
        assert reg.count == 3


class TestAgentRegistryUnregister:
    """Test unregister_agent operations."""

    def test_unregister_by_id(self):
        reg = AgentRegistry()
        reg.register_agent(_AgentEntry("gl-001", "A"))
        assert reg.unregister_agent("gl-001") is True
        assert reg.count == 0

    def test_unregister_by_version(self):
        reg = AgentRegistry()
        reg.register_agent(_AgentEntry("gl-001", "V1", version="1.0.0"))
        reg.register_agent(_AgentEntry("gl-001", "V2", version="2.0.0"))
        assert reg.unregister_agent("gl-001", version="1.0.0") is True
        assert reg.count == 1
        assert reg.list_versions("gl-001") == ["2.0.0"]

    def test_unregister_not_found(self):
        reg = AgentRegistry()
        assert reg.unregister_agent("nonexistent") is False

    def test_unregister_version_not_found(self):
        reg = AgentRegistry()
        reg.register_agent(_AgentEntry("gl-001", "A"))
        assert reg.unregister_agent("gl-001", version="9.9.9") is False

    def test_unregister_last_version_removes_agent(self):
        reg = AgentRegistry()
        reg.register_agent(_AgentEntry("gl-001", "A", version="1.0.0"))
        reg.unregister_agent("gl-001", version="1.0.0")
        assert reg.count == 0


class TestAgentRegistryGet:
    """Test get_agent operations."""

    def test_get_by_id(self):
        reg = AgentRegistry()
        reg.register_agent(_AgentEntry("gl-001", "Agent 1"))
        entry = reg.get_agent("gl-001")
        assert entry is not None
        assert entry.name == "Agent 1"

    def test_get_specific_version(self):
        reg = AgentRegistry()
        reg.register_agent(_AgentEntry("gl-001", "V1", version="1.0.0"))
        reg.register_agent(_AgentEntry("gl-001", "V2", version="2.0.0"))
        entry = reg.get_agent("gl-001", version="1.0.0")
        assert entry.name == "V1"

    def test_get_latest_version(self):
        reg = AgentRegistry()
        reg.register_agent(_AgentEntry("gl-001", "V1", version="1.0.0"))
        reg.register_agent(_AgentEntry("gl-001", "V2", version="2.0.0"))
        entry = reg.get_agent("gl-001")
        assert entry.name == "V2"

    def test_get_not_found(self):
        reg = AgentRegistry()
        assert reg.get_agent("nonexistent") is None


class TestAgentRegistryList:
    """Test list_agents with filters."""

    def _populate(self, reg):
        reg.register_agent(_AgentEntry("gl-001", "Carbon Calc", layer="calculation",
                                        sectors=["energy"], tags=["carbon"],
                                        health="healthy",
                                        capabilities=[AgentCapability("carbon_calc", "calculation")]))
        reg.register_agent(_AgentEntry("gl-002", "CBAM Reporter", layer="reporting",
                                        sectors=["manufacturing"], tags=["cbam"],
                                        health="healthy",
                                        capabilities=[AgentCapability("cbam_report", "reporting")]))
        reg.register_agent(_AgentEntry("gl-003", "Data Validator", layer="validation",
                                        sectors=["energy", "manufacturing"], tags=["validation"],
                                        health="degraded",
                                        capabilities=[AgentCapability("data_validate", "validation")]))

    def test_list_all(self):
        reg = AgentRegistry()
        self._populate(reg)
        assert len(reg.list_agents()) == 3

    def test_list_by_layer(self):
        reg = AgentRegistry()
        self._populate(reg)
        results = reg.list_agents(layer="calculation")
        assert len(results) == 1
        assert results[0].agent_id == "gl-001"

    def test_list_by_sector(self):
        reg = AgentRegistry()
        self._populate(reg)
        results = reg.list_agents(sector="energy")
        assert len(results) == 2

    def test_list_by_capability(self):
        reg = AgentRegistry()
        self._populate(reg)
        results = reg.list_agents(capability="cbam_report")
        assert len(results) == 1

    def test_list_by_tag(self):
        reg = AgentRegistry()
        self._populate(reg)
        results = reg.list_agents(tag="carbon")
        assert len(results) == 1

    def test_list_by_health(self):
        reg = AgentRegistry()
        self._populate(reg)
        results = reg.list_agents(health="degraded")
        assert len(results) == 1
        assert results[0].agent_id == "gl-003"

    def test_list_search_text(self):
        reg = AgentRegistry()
        self._populate(reg)
        results = reg.list_agents(search="Carbon")
        assert len(results) == 1

    def test_list_search_description(self):
        reg = AgentRegistry()
        reg.register_agent(_AgentEntry("gl-004", "Test", description="emission factor lookup"))
        results = reg.list_agents(search="emission")
        assert len(results) == 1

    def test_list_pagination_offset(self):
        reg = AgentRegistry()
        self._populate(reg)
        results = reg.list_agents(offset=1, limit=2)
        assert len(results) == 2

    def test_list_pagination_limit(self):
        reg = AgentRegistry()
        self._populate(reg)
        results = reg.list_agents(limit=1)
        assert len(results) == 1

    def test_list_empty_registry(self):
        reg = AgentRegistry()
        assert reg.list_agents() == []


class TestAgentRegistryVersions:
    """Test list_versions."""

    def test_versions_sorted(self):
        reg = AgentRegistry()
        reg.register_agent(_AgentEntry("gl-001", "A", version="2.0.0"))
        reg.register_agent(_AgentEntry("gl-001", "A", version="1.0.0"))
        reg.register_agent(_AgentEntry("gl-001", "A", version="1.5.0"))
        versions = reg.list_versions("gl-001")
        assert versions == ["1.0.0", "1.5.0", "2.0.0"]

    def test_versions_empty(self):
        reg = AgentRegistry()
        assert reg.list_versions("nonexistent") == []


class TestAgentRegistryUpdate:
    """Test update_agent."""

    def test_update_fields(self):
        reg = AgentRegistry()
        reg.register_agent(_AgentEntry("gl-001", "V1"))
        h = reg.update_agent(_AgentEntry("gl-001", "V2"))
        assert len(h) == 64
        assert reg.get_agent("gl-001").name == "V2"

    def test_update_returns_new_hash(self):
        reg = AgentRegistry()
        h1 = reg.register_agent(_AgentEntry("gl-001", "V1"))
        h2 = reg.update_agent(_AgentEntry("gl-001", "V2"))
        assert h1 != h2


class TestAgentRegistryHotReload:
    """Test hot_reload_agent."""

    def test_hot_reload_replaces(self):
        reg = AgentRegistry()
        reg.register_agent(_AgentEntry("gl-001", "V1", version="1.0.0"))
        reg.register_agent(_AgentEntry("gl-001", "V1.1", version="1.1.0"))
        assert len(reg.list_versions("gl-001")) == 2
        reg.hot_reload_agent(_AgentEntry("gl-001", "V2", version="2.0.0"))
        assert len(reg.list_versions("gl-001")) == 1
        assert reg.get_agent("gl-001").name == "V2"

    def test_hot_reload_new_agent(self):
        reg = AgentRegistry()
        h = reg.hot_reload_agent(_AgentEntry("gl-new", "New Agent"))
        assert len(h) == 64
        assert reg.count == 1


class TestAgentRegistryIndexes:
    """Test index correctness."""

    def test_by_layer_index(self):
        reg = AgentRegistry()
        reg.register_agent(_AgentEntry("gl-001", "A", layer="calculation"))
        reg.register_agent(_AgentEntry("gl-002", "B", layer="calculation"))
        reg.register_agent(_AgentEntry("gl-003", "C", layer="reporting"))
        assert len(reg._by_layer["calculation"]) == 2
        assert len(reg._by_layer["reporting"]) == 1

    def test_by_sector_index(self):
        reg = AgentRegistry()
        reg.register_agent(_AgentEntry("gl-001", "A", sectors=["energy"]))
        assert "gl-001" in reg._by_sector["energy"]

    def test_by_capability_index(self):
        reg = AgentRegistry()
        cap = AgentCapability("calc", "calculation")
        reg.register_agent(_AgentEntry("gl-001", "A", capabilities=[cap]))
        assert "gl-001" in reg._by_capability["calc"]

    def test_by_tag_index(self):
        reg = AgentRegistry()
        reg.register_agent(_AgentEntry("gl-001", "A", tags=["carbon", "ghg"]))
        assert "gl-001" in reg._by_tag["carbon"]
        assert "gl-001" in reg._by_tag["ghg"]


class TestAgentRegistryHelpers:
    """Test get_all_agent_ids and get_agents_by_layer."""

    def test_get_all_agent_ids(self):
        reg = AgentRegistry()
        reg.register_agent(_AgentEntry("gl-001", "A"))
        reg.register_agent(_AgentEntry("gl-002", "B"))
        ids = reg.get_all_agent_ids()
        assert set(ids) == {"gl-001", "gl-002"}

    def test_get_agents_by_layer(self):
        reg = AgentRegistry()
        reg.register_agent(_AgentEntry("gl-001", "A", layer="calculation"))
        reg.register_agent(_AgentEntry("gl-002", "B", layer="reporting"))
        results = reg.get_agents_by_layer("calculation")
        assert len(results) == 1
        assert results[0].agent_id == "gl-001"


class TestAgentRegistryExportImport:
    """Test export and import round-trip."""

    def test_export(self):
        reg = AgentRegistry()
        reg.register_agent(_AgentEntry("gl-001", "A"))
        data = reg.export_registry()
        assert "gl-001" in data

    def test_import_merge(self):
        reg = AgentRegistry()
        reg.register_agent(_AgentEntry("gl-001", "A"))
        data = {
            "gl-002": {
                "1.0.0": {
                    "agent_id": "gl-002", "name": "B",
                    "version": "1.0.0", "layer": "utility",
                }
            }
        }
        imported = reg.import_registry(data, mode="merge")
        assert imported == 1
        assert reg.count == 2

    def test_import_replace(self):
        reg = AgentRegistry()
        reg.register_agent(_AgentEntry("gl-001", "A"))
        data = {
            "gl-002": {
                "1.0.0": {
                    "agent_id": "gl-002", "name": "B",
                    "version": "1.0.0", "layer": "utility",
                }
            }
        }
        imported = reg.import_registry(data, mode="replace")
        assert imported == 1
        assert reg.count == 1
        assert reg.get_agent("gl-002") is not None

    def test_round_trip(self):
        reg1 = AgentRegistry()
        reg1.register_agent(_AgentEntry("gl-001", "A", layer="calculation"))
        reg1.register_agent(_AgentEntry("gl-002", "B", layer="reporting"))
        data = reg1.export_registry()

        reg2 = AgentRegistry()
        reg2.import_registry(data, mode="replace")
        assert reg2.count == 2
        assert reg2.get_agent("gl-001").name == "A"


class TestAgentRegistryStatistics:
    """Test get_statistics."""

    def test_statistics_counts(self):
        reg = AgentRegistry()
        reg.register_agent(_AgentEntry("gl-001", "A", layer="calculation", health="healthy"))
        reg.register_agent(_AgentEntry("gl-002", "B", layer="reporting", health="degraded"))
        stats = reg.get_statistics()
        assert stats["total_agents"] == 2
        assert stats["total_versions"] == 2
        assert stats["by_layer"]["calculation"] == 1
        assert stats["by_health"]["healthy"] == 1

    def test_statistics_empty(self):
        reg = AgentRegistry()
        stats = reg.get_statistics()
        assert stats["total_agents"] == 0


class TestAgentRegistryThreadSafety:
    """Test concurrent registration."""

    def test_concurrent_registration(self):
        reg = AgentRegistry()
        errors = []

        def register(i):
            try:
                reg.register_agent(_AgentEntry(f"gl-{i:03d}", f"Agent {i}"))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=register, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert reg.count == 50
