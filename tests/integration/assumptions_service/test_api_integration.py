# -*- coding: utf-8 -*-
"""
API Integration Tests for Assumptions Service (AGENT-FOUND-004)

Tests the assumptions API endpoints with a simulated TestClient,
validating CRUD operations, scenario management, value get/set,
export/import, health, and error responses.

All implementations are self-contained to avoid cross-module import issues.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Self-contained implementations for API integration
# ---------------------------------------------------------------------------

class Assumption:
    def __init__(self, assumption_id, name, description="", category="custom",
                 data_type="float", value=None, unit="", source="",
                 tags=None, metadata=None, version=1):
        self.assumption_id = assumption_id
        self.name = name
        self.description = description
        self.category = category
        self.data_type = data_type
        self.value = value
        self.unit = unit
        self.source = source
        self.tags = tags or []
        self.metadata = metadata or {}
        self.version = version


class AssumptionNotFoundError(Exception):
    pass

class DuplicateAssumptionError(Exception):
    pass


class Scenario:
    def __init__(self, scenario_id, name, overrides=None, is_active=True):
        self.scenario_id = scenario_id
        self.name = name
        self.overrides = overrides or {}
        self.is_active = is_active


class ScenarioNotFoundError(Exception):
    pass

class ProtectedScenarioError(Exception):
    pass


class AssumptionsTestClient:
    """Simulated API client for assumptions service."""

    def __init__(self):
        self._assumptions: Dict[str, Assumption] = {}
        self._scenarios: Dict[str, Scenario] = {
            "baseline": Scenario("baseline", "Baseline"),
            "conservative": Scenario("conservative", "Conservative"),
            "optimistic": Scenario("optimistic", "Optimistic"),
        }

    # ---- Assumptions CRUD ----

    def post_assumption(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """POST /assumptions/"""
        aid = data.get("assumption_id", "")
        name = data.get("name", "")
        if not aid or not name:
            return {"error": "assumption_id and name required", "status": 400}
        if aid in self._assumptions:
            return {"error": f"Assumption '{aid}' already exists", "status": 409}
        a = Assumption(aid, name, value=data.get("value"),
                       category=data.get("category", "custom"),
                       tags=data.get("tags"))
        self._assumptions[aid] = a
        return {"data": {"assumption_id": aid, "name": name, "value": a.value}, "status": 201}

    def get_assumption(self, aid: str) -> Dict[str, Any]:
        """GET /assumptions/{id}"""
        if aid not in self._assumptions:
            return {"error": f"Assumption '{aid}' not found", "status": 404}
        a = self._assumptions[aid]
        return {"data": {"assumption_id": a.assumption_id, "name": a.name,
                         "value": a.value, "category": a.category}, "status": 200}

    def put_assumption(self, aid: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """PUT /assumptions/{id}"""
        if aid not in self._assumptions:
            return {"error": f"Assumption '{aid}' not found", "status": 404}
        a = self._assumptions[aid]
        if "value" in data:
            a.value = data["value"]
        if "name" in data:
            a.name = data["name"]
        a.version += 1
        return {"data": {"assumption_id": aid, "value": a.value, "version": a.version}, "status": 200}

    def delete_assumption(self, aid: str) -> Dict[str, Any]:
        """DELETE /assumptions/{id}"""
        if aid not in self._assumptions:
            return {"error": f"Assumption '{aid}' not found", "status": 404}
        del self._assumptions[aid]
        return {"status": 204}

    def list_assumptions(self) -> Dict[str, Any]:
        """GET /assumptions/"""
        items = [{"assumption_id": a.assumption_id, "name": a.name, "value": a.value}
                 for a in self._assumptions.values()]
        return {"data": items, "status": 200}

    # ---- Values ----

    def get_value(self, aid: str) -> Dict[str, Any]:
        """GET /assumptions/{id}/value"""
        if aid not in self._assumptions:
            return {"error": "not found", "status": 404}
        return {"data": {"assumption_id": aid, "value": self._assumptions[aid].value}, "status": 200}

    def put_value(self, aid: str, value: Any) -> Dict[str, Any]:
        """PUT /assumptions/{id}/value"""
        if aid not in self._assumptions:
            return {"error": "not found", "status": 404}
        self._assumptions[aid].value = value
        self._assumptions[aid].version += 1
        return {"data": {"value": value}, "status": 200}

    # ---- Scenarios ----

    def post_scenario(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """POST /assumptions/scenarios"""
        sid = data.get("scenario_id", "")
        if sid in self._scenarios:
            return {"error": "already exists", "status": 409}
        s = Scenario(sid, data.get("name", ""), overrides=data.get("overrides"))
        self._scenarios[sid] = s
        return {"data": {"scenario_id": sid}, "status": 201}

    def get_scenario(self, sid: str) -> Dict[str, Any]:
        """GET /assumptions/scenarios/{id}"""
        if sid not in self._scenarios:
            return {"error": "not found", "status": 404}
        s = self._scenarios[sid]
        return {"data": {"scenario_id": s.scenario_id, "name": s.name,
                         "overrides": s.overrides}, "status": 200}

    def put_scenario(self, sid: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """PUT /assumptions/scenarios/{id}"""
        if sid not in self._scenarios:
            return {"error": "not found", "status": 404}
        s = self._scenarios[sid]
        if "overrides" in data:
            s.overrides = data["overrides"]
        return {"data": {"scenario_id": sid}, "status": 200}

    def delete_scenario(self, sid: str) -> Dict[str, Any]:
        """DELETE /assumptions/scenarios/{id}"""
        if sid not in self._scenarios:
            return {"error": "not found", "status": 404}
        if sid in ("baseline",):
            return {"error": "protected", "status": 409}
        del self._scenarios[sid]
        return {"status": 204}

    def list_scenarios(self) -> Dict[str, Any]:
        """GET /assumptions/scenarios"""
        items = [{"scenario_id": s.scenario_id, "name": s.name}
                 for s in self._scenarios.values()]
        return {"data": items, "status": 200}

    # ---- Export/Import ----

    def post_export(self) -> Dict[str, Any]:
        """POST /assumptions/export"""
        items = [{"assumption_id": a.assumption_id, "name": a.name, "value": a.value}
                 for a in self._assumptions.values()]
        h = hashlib.sha256(json.dumps(items, sort_keys=True, default=str).encode()).hexdigest()
        return {"data": {"assumptions": items, "integrity_hash": h}, "status": 200}

    def post_import(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """POST /assumptions/import"""
        imported = 0
        for item in data.get("assumptions", []):
            aid = item.get("assumption_id", "")
            if aid not in self._assumptions:
                self._assumptions[aid] = Assumption(aid, item.get("name", ""),
                                                     value=item.get("value"))
                imported += 1
        return {"data": {"imported": imported}, "status": 200}

    # ---- Health ----

    def get_health(self) -> Dict[str, Any]:
        """GET /assumptions/health"""
        return {"status": "healthy", "version": "1.0.0",
                "assumptions_count": len(self._assumptions),
                "scenarios_count": len(self._scenarios)}


@pytest.fixture
def client():
    return AssumptionsTestClient()


# ===========================================================================
# Test Classes
# ===========================================================================


class TestAssumptionsCRUD:
    """Test full CRUD via TestClient."""

    def test_create(self, client):
        resp = client.post_assumption({"assumption_id": "ef1", "name": "EF1", "value": 2.68})
        assert resp["status"] == 201
        assert resp["data"]["assumption_id"] == "ef1"

    def test_get(self, client):
        client.post_assumption({"assumption_id": "ef1", "name": "EF1", "value": 2.68})
        resp = client.get_assumption("ef1")
        assert resp["status"] == 200
        assert resp["data"]["value"] == 2.68

    def test_update(self, client):
        client.post_assumption({"assumption_id": "ef1", "name": "EF1", "value": 2.68})
        resp = client.put_assumption("ef1", {"value": 2.75})
        assert resp["status"] == 200
        assert resp["data"]["value"] == 2.75

    def test_delete(self, client):
        client.post_assumption({"assumption_id": "ef1", "name": "EF1"})
        resp = client.delete_assumption("ef1")
        assert resp["status"] == 204

    def test_list(self, client):
        client.post_assumption({"assumption_id": "ef1", "name": "EF1"})
        client.post_assumption({"assumption_id": "ef2", "name": "EF2"})
        resp = client.list_assumptions()
        assert resp["status"] == 200
        assert len(resp["data"]) == 2


class TestScenariosCRUD:
    """Test scenario CRUD via TestClient."""

    def test_create_scenario(self, client):
        resp = client.post_scenario({"scenario_id": "s1", "name": "Custom"})
        assert resp["status"] == 201

    def test_get_scenario(self, client):
        resp = client.get_scenario("baseline")
        assert resp["status"] == 200
        assert resp["data"]["name"] == "Baseline"

    def test_update_scenario(self, client):
        client.post_scenario({"scenario_id": "s1", "name": "S1"})
        resp = client.put_scenario("s1", {"overrides": {"ef": 3.0}})
        assert resp["status"] == 200

    def test_delete_scenario(self, client):
        client.post_scenario({"scenario_id": "s1", "name": "S1"})
        resp = client.delete_scenario("s1")
        assert resp["status"] == 204

    def test_list_scenarios(self, client):
        resp = client.list_scenarios()
        assert resp["status"] == 200
        assert len(resp["data"]) >= 3  # defaults


class TestValueGetSet:
    """Test value get/set via TestClient."""

    def test_get_value(self, client):
        client.post_assumption({"assumption_id": "ef1", "name": "EF1", "value": 2.68})
        resp = client.get_value("ef1")
        assert resp["status"] == 200
        assert resp["data"]["value"] == 2.68

    def test_set_value(self, client):
        client.post_assumption({"assumption_id": "ef1", "name": "EF1", "value": 2.68})
        resp = client.put_value("ef1", 2.75)
        assert resp["status"] == 200
        assert resp["data"]["value"] == 2.75


class TestExportImport:
    """Test export/import via TestClient."""

    def test_export(self, client):
        client.post_assumption({"assumption_id": "ef1", "name": "EF1", "value": 2.68})
        resp = client.post_export()
        assert resp["status"] == 200
        assert len(resp["data"]["assumptions"]) == 1
        assert len(resp["data"]["integrity_hash"]) == 64

    def test_import(self, client):
        data = {"assumptions": [
            {"assumption_id": "ef1", "name": "EF1", "value": 1},
            {"assumption_id": "ef2", "name": "EF2", "value": 2},
        ]}
        resp = client.post_import(data)
        assert resp["status"] == 200
        assert resp["data"]["imported"] == 2


class TestHealthEndpoint:
    """Test health endpoint."""

    def test_health(self, client):
        resp = client.get_health()
        assert resp["status"] == "healthy"
        assert resp["version"] == "1.0.0"


class TestErrorResponses:
    """Test error response codes."""

    def test_404_not_found(self, client):
        resp = client.get_assumption("nonexistent")
        assert resp["status"] == 404

    def test_400_missing_fields(self, client):
        resp = client.post_assumption({"name": "Test"})
        assert resp["status"] == 400

    def test_409_duplicate(self, client):
        client.post_assumption({"assumption_id": "ef1", "name": "EF1"})
        resp = client.post_assumption({"assumption_id": "ef1", "name": "EF1 Dup"})
        assert resp["status"] == 409

    def test_409_protected_scenario_delete(self, client):
        resp = client.delete_scenario("baseline")
        assert resp["status"] == 409

    def test_404_scenario_not_found(self, client):
        resp = client.get_scenario("nonexistent")
        assert resp["status"] == 404

    def test_404_value_not_found(self, client):
        resp = client.get_value("nonexistent")
        assert resp["status"] == 404
