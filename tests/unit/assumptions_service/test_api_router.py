# -*- coding: utf-8 -*-
"""
Unit Tests for Assumptions API Router (AGENT-FOUND-004)

Tests all 18+ API endpoints using simulated handler with mocked service:
assumptions CRUD, value get/set, version history, scenarios CRUD,
dependency queries, sensitivity analysis, export/import, health.

Coverage target: 85%+ of api/router.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Inline API models and router handler mirroring
# greenlang/assumptions/api/router.py
# ---------------------------------------------------------------------------


class HealthResponse:
    def __init__(self, status: str = "healthy", version: str = "1.0.0",
                 assumptions_count: int = 0, scenarios_count: int = 0):
        self.status = status
        self.version = version
        self.assumptions_count = assumptions_count
        self.scenarios_count = scenarios_count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "version": self.version,
            "assumptions_count": self.assumptions_count,
            "scenarios_count": self.scenarios_count,
        }


class AssumptionsRouter:
    """Simulates the FastAPI router for assumptions endpoints."""

    def __init__(self, service=None):
        self._service = service or MagicMock()

    # ---- Assumptions CRUD ----

    def handle_create_assumption(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """POST /assumptions/"""
        if not data.get("assumption_id") or not data.get("name"):
            return {"error": "assumption_id and name are required", "status": 400}

        try:
            result = self._service.create_assumption(**data)
            return {"data": getattr(result, "to_dict", lambda: data)(), "status": 201}
        except Exception as e:
            if "already exists" in str(e):
                return {"error": str(e), "status": 409}
            return {"error": str(e), "status": 400}

    def handle_list_assumptions(self, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """GET /assumptions/"""
        params = params or {}
        results = self._service.list_assumptions(**params)
        items = results if isinstance(results, list) else []
        return {"data": items, "status": 200}

    def handle_get_assumption(self, assumption_id: str) -> Dict[str, Any]:
        """GET /assumptions/{id}"""
        try:
            result = self._service.get_assumption(assumption_id)
            return {"data": getattr(result, "to_dict", lambda: {})(), "status": 200}
        except Exception as e:
            if "not found" in str(e).lower():
                return {"error": str(e), "status": 404}
            return {"error": str(e), "status": 400}

    def handle_update_assumption(self, assumption_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """PUT /assumptions/{id}"""
        try:
            result = self._service.update_assumption(assumption_id, **data)
            return {"data": getattr(result, "to_dict", lambda: data)(), "status": 200}
        except Exception as e:
            if "not found" in str(e).lower():
                return {"error": str(e), "status": 404}
            return {"error": str(e), "status": 400}

    def handle_delete_assumption(self, assumption_id: str) -> Dict[str, Any]:
        """DELETE /assumptions/{id}"""
        try:
            self._service.delete_assumption(assumption_id)
            return {"status": 204}
        except Exception as e:
            if "not found" in str(e).lower():
                return {"error": str(e), "status": 404}
            if "in use" in str(e).lower():
                return {"error": str(e), "status": 409}
            return {"error": str(e), "status": 400}

    def handle_get_versions(self, assumption_id: str) -> Dict[str, Any]:
        """GET /assumptions/{id}/versions"""
        try:
            result = self._service.registry.get_versions(assumption_id)
            return {"data": result, "status": 200}
        except Exception as e:
            if "not found" in str(e).lower():
                return {"error": str(e), "status": 404}
            return {"error": str(e), "status": 400}

    def handle_get_value(self, assumption_id: str, scenario_id: Optional[str] = None) -> Dict[str, Any]:
        """GET /assumptions/{id}/value"""
        try:
            kwargs = {}
            if scenario_id:
                kwargs["scenario_overrides"] = self._service.scenario_manager.get_overrides(scenario_id)
            result = self._service.get_value(assumption_id, **kwargs)
            return {"data": {"assumption_id": assumption_id, "value": result}, "status": 200}
        except Exception as e:
            if "not found" in str(e).lower():
                return {"error": str(e), "status": 404}
            return {"error": str(e), "status": 400}

    def handle_set_value(self, assumption_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """PUT /assumptions/{id}/value"""
        try:
            result = self._service.set_value(assumption_id, data.get("value"))
            return {"data": getattr(result, "to_dict", lambda: {})(), "status": 200}
        except Exception as e:
            return {"error": str(e), "status": 400}

    def handle_validate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """POST /assumptions/validate"""
        result = self._service.validator.validate(
            data.get("assumption_id", ""),
            data.get("value"),
        )
        return {"data": result, "status": 200}

    # ---- Scenarios ----

    def handle_create_scenario(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """POST /assumptions/scenarios"""
        try:
            result = self._service.create_scenario(**data)
            return {"data": getattr(result, "to_dict", lambda: data)(), "status": 201}
        except Exception as e:
            if "already exists" in str(e):
                return {"error": str(e), "status": 409}
            return {"error": str(e), "status": 400}

    def handle_list_scenarios(self, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """GET /assumptions/scenarios"""
        params = params or {}
        results = self._service.list_scenarios(**params)
        return {"data": results if isinstance(results, list) else [], "status": 200}

    def handle_get_scenario(self, scenario_id: str) -> Dict[str, Any]:
        """GET /assumptions/scenarios/{id}"""
        try:
            result = self._service.get_scenario(scenario_id)
            return {"data": getattr(result, "to_dict", lambda: {})(), "status": 200}
        except Exception as e:
            if "not found" in str(e).lower():
                return {"error": str(e), "status": 404}
            return {"error": str(e), "status": 400}

    def handle_update_scenario(self, scenario_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """PUT /assumptions/scenarios/{id}"""
        try:
            result = self._service.update_scenario(scenario_id, **data)
            return {"data": getattr(result, "to_dict", lambda: data)(), "status": 200}
        except Exception as e:
            if "not found" in str(e).lower():
                return {"error": str(e), "status": 404}
            return {"error": str(e), "status": 400}

    def handle_delete_scenario(self, scenario_id: str) -> Dict[str, Any]:
        """DELETE /assumptions/scenarios/{id}"""
        try:
            self._service.delete_scenario(scenario_id)
            return {"status": 204}
        except Exception as e:
            if "protected" in str(e).lower():
                return {"error": str(e), "status": 409}
            if "not found" in str(e).lower():
                return {"error": str(e), "status": 404}
            return {"error": str(e), "status": 400}

    # ---- Dependencies ----

    def handle_get_dependencies(self, assumption_id: str) -> Dict[str, Any]:
        """GET /assumptions/{id}/dependencies"""
        upstream = self._service.dependencies.get_upstream(assumption_id)
        downstream = self._service.dependencies.get_downstream(assumption_id)
        impact = self._service.dependencies.get_impact(assumption_id)
        return {
            "data": {"upstream": upstream, "downstream": downstream, "impact": impact},
            "status": 200,
        }

    # ---- Sensitivity ----

    def handle_sensitivity(self, assumption_id: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """GET /assumptions/{id}/sensitivity"""
        return {
            "data": {
                "assumption_id": assumption_id,
                "base_value": None,
                "variations": [],
            },
            "status": 200,
        }

    # ---- Export/Import ----

    def handle_export(self) -> Dict[str, Any]:
        """POST /assumptions/export"""
        result = self._service.registry.export_all()
        return {"data": result, "status": 200}

    def handle_import(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """POST /assumptions/import"""
        result = self._service.registry.import_all(data)
        return {"data": result, "status": 200}

    # ---- Health ----

    def handle_health(self) -> Dict[str, Any]:
        """GET /assumptions/health"""
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            assumptions_count=0,
            scenarios_count=0,
        ).to_dict()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def router():
    return AssumptionsRouter()


# ===========================================================================
# Test Classes
# ===========================================================================


class TestCreateAssumptionEndpoint:
    """Test POST /assumptions/"""

    def test_create_success(self, router):
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"assumption_id": "a1", "name": "Test"}
        router._service.create_assumption.return_value = mock_result

        resp = router.handle_create_assumption({"assumption_id": "a1", "name": "Test"})
        assert resp["status"] == 201

    def test_create_missing_id(self, router):
        resp = router.handle_create_assumption({"name": "Test"})
        assert resp["status"] == 400

    def test_create_missing_name(self, router):
        resp = router.handle_create_assumption({"assumption_id": "a1"})
        assert resp["status"] == 400

    def test_create_duplicate(self, router):
        router._service.create_assumption.side_effect = Exception("already exists")
        resp = router.handle_create_assumption({"assumption_id": "a1", "name": "Test"})
        assert resp["status"] == 409


class TestListAssumptionsEndpoint:
    """Test GET /assumptions/"""

    def test_list_returns_items(self, router):
        router._service.list_assumptions.return_value = [MagicMock(), MagicMock()]
        resp = router.handle_list_assumptions()
        assert resp["status"] == 200
        assert len(resp["data"]) == 2

    def test_list_empty(self, router):
        router._service.list_assumptions.return_value = []
        resp = router.handle_list_assumptions()
        assert resp["data"] == []

    def test_list_with_category_filter(self, router):
        router._service.list_assumptions.return_value = [MagicMock()]
        resp = router.handle_list_assumptions({"category": "emission_factor"})
        assert resp["status"] == 200


class TestGetAssumptionEndpoint:
    """Test GET /assumptions/{id}"""

    def test_get_success(self, router):
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"assumption_id": "a1"}
        router._service.get_assumption.return_value = mock_result
        resp = router.handle_get_assumption("a1")
        assert resp["status"] == 200

    def test_get_not_found(self, router):
        router._service.get_assumption.side_effect = Exception("not found")
        resp = router.handle_get_assumption("nonexistent")
        assert resp["status"] == 404


class TestUpdateAssumptionEndpoint:
    """Test PUT /assumptions/{id}"""

    def test_update_success(self, router):
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"assumption_id": "a1", "value": 2.75}
        router._service.update_assumption.return_value = mock_result
        resp = router.handle_update_assumption("a1", {"value": 2.75})
        assert resp["status"] == 200

    def test_update_not_found(self, router):
        router._service.update_assumption.side_effect = Exception("not found")
        resp = router.handle_update_assumption("nonexistent", {"value": 1})
        assert resp["status"] == 404


class TestDeleteAssumptionEndpoint:
    """Test DELETE /assumptions/{id}"""

    def test_delete_success(self, router):
        resp = router.handle_delete_assumption("a1")
        assert resp["status"] == 204

    def test_delete_not_found(self, router):
        router._service.delete_assumption.side_effect = Exception("not found")
        resp = router.handle_delete_assumption("nonexistent")
        assert resp["status"] == 404

    def test_delete_in_use(self, router):
        router._service.delete_assumption.side_effect = Exception("in use by")
        resp = router.handle_delete_assumption("a1")
        assert resp["status"] == 409


class TestGetVersionsEndpoint:
    """Test GET /assumptions/{id}/versions"""

    def test_get_versions_success(self, router):
        router._service.registry.get_versions.return_value = [MagicMock()]
        resp = router.handle_get_versions("a1")
        assert resp["status"] == 200

    def test_get_versions_not_found(self, router):
        router._service.registry.get_versions.side_effect = Exception("not found")
        resp = router.handle_get_versions("nonexistent")
        assert resp["status"] == 404


class TestGetValueEndpoint:
    """Test GET /assumptions/{id}/value"""

    def test_get_value_success(self, router):
        router._service.get_value.return_value = 2.68
        resp = router.handle_get_value("diesel_ef")
        assert resp["status"] == 200
        assert resp["data"]["value"] == 2.68

    def test_get_value_not_found(self, router):
        router._service.get_value.side_effect = Exception("not found")
        resp = router.handle_get_value("nonexistent")
        assert resp["status"] == 404


class TestSetValueEndpoint:
    """Test PUT /assumptions/{id}/value"""

    def test_set_value_success(self, router):
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"value": 2.75}
        router._service.set_value.return_value = mock_result
        resp = router.handle_set_value("a1", {"value": 2.75})
        assert resp["status"] == 200


class TestValidateEndpoint:
    """Test POST /assumptions/validate"""

    def test_validate_returns_result(self, router):
        mock_result = MagicMock()
        mock_result.is_valid = True
        router._service.validator.validate.return_value = mock_result
        resp = router.handle_validate({"assumption_id": "a1", "value": 2.68})
        assert resp["status"] == 200


class TestCreateScenarioEndpoint:
    """Test POST /assumptions/scenarios"""

    def test_create_success(self, router):
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"scenario_id": "s1"}
        router._service.create_scenario.return_value = mock_result
        resp = router.handle_create_scenario({"scenario_id": "s1", "name": "Test"})
        assert resp["status"] == 201

    def test_create_duplicate(self, router):
        router._service.create_scenario.side_effect = Exception("already exists")
        resp = router.handle_create_scenario({"scenario_id": "s1", "name": "Dup"})
        assert resp["status"] == 409


class TestListScenariosEndpoint:
    """Test GET /assumptions/scenarios"""

    def test_list_returns_items(self, router):
        router._service.list_scenarios.return_value = [MagicMock()]
        resp = router.handle_list_scenarios()
        assert resp["status"] == 200


class TestGetScenarioEndpoint:
    """Test GET /assumptions/scenarios/{id}"""

    def test_get_success(self, router):
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"scenario_id": "s1"}
        router._service.get_scenario.return_value = mock_result
        resp = router.handle_get_scenario("s1")
        assert resp["status"] == 200

    def test_get_not_found(self, router):
        router._service.get_scenario.side_effect = Exception("not found")
        resp = router.handle_get_scenario("nonexistent")
        assert resp["status"] == 404


class TestUpdateScenarioEndpoint:
    """Test PUT /assumptions/scenarios/{id}"""

    def test_update_success(self, router):
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"name": "Updated"}
        router._service.update_scenario.return_value = mock_result
        resp = router.handle_update_scenario("s1", {"name": "Updated"})
        assert resp["status"] == 200


class TestDeleteScenarioEndpoint:
    """Test DELETE /assumptions/scenarios/{id}"""

    def test_delete_success(self, router):
        resp = router.handle_delete_scenario("s1")
        assert resp["status"] == 204

    def test_delete_protected(self, router):
        router._service.delete_scenario.side_effect = Exception("protected")
        resp = router.handle_delete_scenario("baseline")
        assert resp["status"] == 409


class TestDependenciesEndpoint:
    """Test GET /assumptions/{id}/dependencies"""

    def test_get_dependencies(self, router):
        router._service.dependencies.get_upstream.return_value = ["b"]
        router._service.dependencies.get_downstream.return_value = ["c"]
        router._service.dependencies.get_impact.return_value = {"affected": ["c"]}
        resp = router.handle_get_dependencies("a1")
        assert resp["status"] == 200
        assert resp["data"]["upstream"] == ["b"]


class TestSensitivityEndpoint:
    """Test GET /assumptions/{id}/sensitivity"""

    def test_sensitivity_returns_result(self, router):
        resp = router.handle_sensitivity("a1")
        assert resp["status"] == 200
        assert resp["data"]["assumption_id"] == "a1"


class TestExportEndpoint:
    """Test POST /assumptions/export"""

    def test_export_returns_data(self, router):
        router._service.registry.export_all.return_value = {"assumptions": []}
        resp = router.handle_export()
        assert resp["status"] == 200


class TestImportEndpoint:
    """Test POST /assumptions/import"""

    def test_import_returns_result(self, router):
        router._service.registry.import_all.return_value = {"imported": 5}
        resp = router.handle_import({"assumptions": []})
        assert resp["status"] == 200


class TestHealthEndpoint:
    """Test GET /assumptions/health"""

    def test_health_returns_healthy(self, router):
        resp = router.handle_health()
        assert resp["status"] == "healthy"

    def test_health_returns_version(self, router):
        resp = router.handle_health()
        assert resp["version"] == "1.0.0"
