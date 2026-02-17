# -*- coding: utf-8 -*-
"""
Unit Tests for Validation Rule Engine API Router - AGENT-DATA-019
=================================================================

Comprehensive tests for the FastAPI router module providing 20 endpoints
under ``/api/v1/validation-rules``. Tests cover:

- Router availability and importability
- Rule CRUD endpoints (POST, GET, GET detail, PUT, DELETE /rules)
- Rule set CRUD endpoints (POST, GET, GET detail, PUT, DELETE /rule-sets)
- Evaluation endpoints (POST /evaluate, POST /evaluate/batch, GET /evaluations/{id})
- Conflict detection endpoints (POST /conflicts/detect, GET /conflicts)
- Rule pack endpoints (POST /packs/{name}/apply, GET /packs)
- Report generation endpoint (POST /reports)
- Pipeline orchestration endpoint (POST /pipeline)
- Health check endpoint (GET /health)
- Error handling (404, 400, 422, 503)

Target: 60+ test functions, 85%+ coverage of router.py.

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-019 Validation Rule Engine (GL-DATA-X-022)
"""

from __future__ import annotations

import importlib
import uuid
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Conditional imports -- skip entire module if FastAPI is unavailable
# ---------------------------------------------------------------------------

try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not FASTAPI_AVAILABLE, reason="FastAPI not installed"
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = "/api/v1/validation-rules"


# ---------------------------------------------------------------------------
# Helper: build mock return values
# ---------------------------------------------------------------------------


def _rule_dict(**overrides: Any) -> Dict[str, Any]:
    """Return a minimal rule dict with optional overrides."""
    data: Dict[str, Any] = {
        "rule_id": overrides.pop("rule_id", str(uuid.uuid4())),
        "name": "co2e_range_check",
        "rule_type": "range_check",
        "severity": "error",
        "status": "active",
        "field": "co2e",
        "description": "CO2e must be in range",
        "tags": ["ghg"],
        "created_at": "2026-02-01T00:00:00Z",
        "provenance_hash": "a" * 64,
    }
    data.update(overrides)
    return data


def _rule_set_dict(**overrides: Any) -> Dict[str, Any]:
    """Return a minimal rule set dict with optional overrides."""
    data: Dict[str, Any] = {
        "set_id": overrides.pop("set_id", str(uuid.uuid4())),
        "name": "GHG Scope 1 Rules",
        "pack_type": "ghg_protocol",
        "rule_ids": [str(uuid.uuid4()), str(uuid.uuid4())],
        "rule_count": 2,
        "status": "active",
        "description": "Scope 1 validation rules",
        "created_at": "2026-02-01T00:00:00Z",
        "provenance_hash": "b" * 64,
    }
    data.update(overrides)
    return data


def _evaluation_dict(**overrides: Any) -> Dict[str, Any]:
    """Return a minimal evaluation dict with optional overrides."""
    data: Dict[str, Any] = {
        "evaluation_id": overrides.pop("evaluation_id", str(uuid.uuid4())),
        "rule_set_id": str(uuid.uuid4()),
        "dataset_id": "ds-001",
        "status": "completed",
        "total_rules": 10,
        "rules_passed": 8,
        "rules_failed": 2,
        "rules_warned": 0,
        "pass_rate": 0.8,
        "result": "fail",
        "elapsed_seconds": 1.5,
        "created_at": "2026-02-01T00:00:00Z",
        "provenance_hash": "c" * 64,
    }
    data.update(overrides)
    return data


def _batch_evaluation_dict(**overrides: Any) -> Dict[str, Any]:
    """Return a minimal batch evaluation dict with optional overrides."""
    data: Dict[str, Any] = {
        "batch_id": overrides.pop("batch_id", str(uuid.uuid4())),
        "evaluations": [_evaluation_dict(), _evaluation_dict()],
        "total_datasets": 2,
        "datasets_passed": 1,
        "datasets_failed": 1,
        "overall_pass_rate": 0.5,
        "elapsed_seconds": 3.0,
        "provenance_hash": "d" * 64,
    }
    data.update(overrides)
    return data


def _conflict_dict(**overrides: Any) -> Dict[str, Any]:
    """Return a minimal conflict detection dict with optional overrides."""
    data: Dict[str, Any] = {
        "detection_id": overrides.pop("detection_id", str(uuid.uuid4())),
        "rule_set_id": str(uuid.uuid4()),
        "conflicts": [
            {"conflict_type": "overlap", "rule_a": "r1", "rule_b": "r2"},
        ],
        "conflict_count": 1,
        "conflict_types": ["overlap"],
        "provenance_hash": "e" * 64,
    }
    data.update(overrides)
    return data


def _pack_apply_dict(**overrides: Any) -> Dict[str, Any]:
    """Return a minimal pack apply dict with optional overrides."""
    data: Dict[str, Any] = {
        "pack_name": "ghg_protocol",
        "version": "2.0",
        "rules_imported": 25,
        "rule_sets_created": 3,
        "status": "applied",
        "provenance_hash": "f" * 64,
    }
    data.update(overrides)
    return data


def _report_dict(**overrides: Any) -> Dict[str, Any]:
    """Return a minimal report dict with optional overrides."""
    data: Dict[str, Any] = {
        "report_id": overrides.pop("report_id", str(uuid.uuid4())),
        "report_type": "compliance_report",
        "format": "json",
        "evaluation_id": str(uuid.uuid4()),
        "content": {"summary": "all passed"},
        "provenance_hash": "0" * 64,
    }
    data.update(overrides)
    return data


def _pipeline_dict(**overrides: Any) -> Dict[str, Any]:
    """Return a minimal pipeline result dict with optional overrides."""
    data: Dict[str, Any] = {
        "pipeline_id": str(uuid.uuid4()),
        "stages_completed": ["register", "compose", "evaluate", "detect", "report"],
        "final_status": "completed",
        "elapsed_seconds": 5.0,
        "provenance_hash": "1" * 64,
    }
    data.update(overrides)
    return data


def _pack_list_item(**overrides: Any) -> Dict[str, Any]:
    """Return a minimal pack list item dict."""
    data: Dict[str, Any] = {
        "pack_name": "ghg_protocol",
        "version": "2.0",
        "framework": "ghg_protocol",
        "rule_count": 50,
        "description": "GHG Protocol validation rules",
    }
    data.update(overrides)
    return data


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_service() -> MagicMock:
    """Create a MagicMock service with sensible default return values."""
    svc = MagicMock()

    # Rule CRUD
    svc.register_rule.return_value = _rule_dict()
    svc.search_rules.return_value = [_rule_dict(), _rule_dict()]
    svc.get_rule.return_value = _rule_dict()
    svc.update_rule.return_value = _rule_dict()
    svc.delete_rule.return_value = True

    # Rule set CRUD
    svc.create_rule_set.return_value = _rule_set_dict()
    svc.list_rule_sets.return_value = [_rule_set_dict()]
    svc.get_rule_set.return_value = _rule_set_dict()
    svc.update_rule_set.return_value = _rule_set_dict()
    svc.delete_rule_set.return_value = True

    # Evaluation
    svc.evaluate.return_value = _evaluation_dict()
    svc.batch_evaluate.return_value = _batch_evaluation_dict()
    svc.get_evaluation.return_value = _evaluation_dict()

    # Conflicts
    svc.detect_conflicts.return_value = _conflict_dict()
    svc.list_conflicts.return_value = [_conflict_dict()]

    # Packs
    svc.apply_pack.return_value = _pack_apply_dict()
    svc.list_packs.return_value = [_pack_list_item()]

    # Reports
    svc.generate_report.return_value = _report_dict()

    # Pipeline
    svc.run_pipeline.return_value = _pipeline_dict()

    # Health
    svc.get_health.return_value = {
        "status": "healthy",
        "service": "validation_rule_engine",
        "engines": {"rule_registry": True},
        "started": True,
    }

    return svc


@pytest.fixture
def client(mock_service: MagicMock) -> "TestClient":
    """Create a FastAPI TestClient wired to the mock service."""
    from greenlang.validation_rule_engine.api.router import router

    app = FastAPI()

    # Patch the _get_service function so it returns our mock
    with patch(
        "greenlang.validation_rule_engine.api.router._get_service",
        return_value=mock_service,
    ):
        app.include_router(router)
        yield TestClient(app)


@pytest.fixture
def error_client(mock_service: MagicMock) -> "TestClient":
    """Create a TestClient that does NOT raise server exceptions.

    Used by error-handling tests so that 500 responses are returned
    as HTTP responses instead of propagating as Python exceptions.
    """
    from greenlang.validation_rule_engine.api.router import router

    app = FastAPI()

    with patch(
        "greenlang.validation_rule_engine.api.router._get_service",
        return_value=mock_service,
    ):
        app.include_router(router)
        yield TestClient(app, raise_server_exceptions=False)


# ===========================================================================
# TestRouterAvailability
# ===========================================================================


class TestRouterAvailability:
    """Verify the router module is importable and exposes expected symbols."""

    def test_module_importable(self):
        mod = importlib.import_module("greenlang.validation_rule_engine.api.router")
        assert mod is not None

    def test_fastapi_available_flag(self):
        from greenlang.validation_rule_engine.api.router import FASTAPI_AVAILABLE as flag
        assert flag is True

    def test_router_module_level_instance(self):
        from greenlang.validation_rule_engine.api.router import router
        assert router is not None

    def test_all_exports(self):
        from greenlang.validation_rule_engine.api.router import __all__
        assert "router" in __all__


# ===========================================================================
# TestRuleEndpoints
# ===========================================================================


class TestRuleEndpoints:
    """Tests for the five rule CRUD endpoints."""

    def test_register_rule_200(self, client, mock_service):
        body = {"name": "test_rule", "rule_type": "range_check", "severity": "error", "field": "co2e"}
        resp = client.post(f"{BASE_URL}/rules", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert "rule_id" in data

    def test_register_rule_calls_service(self, client, mock_service):
        body = {"name": "test_rule", "rule_type": "format_validation", "severity": "warning", "field": "email"}
        client.post(f"{BASE_URL}/rules", json=body)
        mock_service.register_rule.assert_called_once()

    def test_list_rules_200(self, client, mock_service):
        resp = client.get(f"{BASE_URL}/rules")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 2

    def test_list_rules_passes_filter_params(self, client, mock_service):
        client.get(
            f"{BASE_URL}/rules",
            params={"rule_type": "range_check", "severity": "error", "limit": 10, "offset": 5},
        )
        mock_service.search_rules.assert_called_once_with(
            rule_type="range_check", severity="error",
            status=None, tag=None,
            limit=10, offset=5,
        )

    def test_get_rule_200(self, client, mock_service):
        resp = client.get(f"{BASE_URL}/rules/rule-123")
        assert resp.status_code == 200
        mock_service.get_rule.assert_called_once_with("rule-123")

    def test_get_rule_404(self, client, mock_service):
        mock_service.get_rule.return_value = None
        resp = client.get(f"{BASE_URL}/rules/nonexistent")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_update_rule_200(self, client, mock_service):
        body = {"severity": "warning", "description": "Updated"}
        resp = client.put(f"{BASE_URL}/rules/rule-123", json=body)
        assert resp.status_code == 200

    def test_update_rule_404(self, client, mock_service):
        mock_service.update_rule.return_value = None
        body = {"severity": "warning"}
        resp = client.put(f"{BASE_URL}/rules/rule-123", json=body)
        assert resp.status_code == 404

    def test_delete_rule_200(self, client, mock_service):
        resp = client.delete(f"{BASE_URL}/rules/rule-123")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "deleted"
        assert data["rule_id"] == "rule-123"

    def test_delete_rule_404(self, client, mock_service):
        mock_service.delete_rule.return_value = False
        resp = client.delete(f"{BASE_URL}/rules/nonexistent")
        assert resp.status_code == 404


# ===========================================================================
# TestRuleSetEndpoints
# ===========================================================================


class TestRuleSetEndpoints:
    """Tests for the five rule set CRUD endpoints."""

    def test_create_rule_set_200(self, client, mock_service):
        body = {"name": "GHG Scope 1", "pack_type": "ghg_protocol"}
        resp = client.post(f"{BASE_URL}/rule-sets", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert "set_id" in data

    def test_create_rule_set_calls_service(self, client, mock_service):
        body = {"name": "CSRD Rules", "pack_type": "csrd_esrs", "description": "CSRD validation"}
        client.post(f"{BASE_URL}/rule-sets", json=body)
        mock_service.create_rule_set.assert_called_once()

    def test_list_rule_sets_200(self, client, mock_service):
        resp = client.get(f"{BASE_URL}/rule-sets")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    def test_list_rule_sets_passes_filter_params(self, client, mock_service):
        client.get(
            f"{BASE_URL}/rule-sets",
            params={"pack_type": "ghg_protocol", "status": "active", "limit": 25, "offset": 10},
        )
        mock_service.list_rule_sets.assert_called_once_with(
            pack_type="ghg_protocol", status="active",
            limit=25, offset=10,
        )

    def test_get_rule_set_200(self, client, mock_service):
        resp = client.get(f"{BASE_URL}/rule-sets/rs-123")
        assert resp.status_code == 200
        mock_service.get_rule_set.assert_called_once_with("rs-123")

    def test_get_rule_set_404(self, client, mock_service):
        mock_service.get_rule_set.return_value = None
        resp = client.get(f"{BASE_URL}/rule-sets/nonexistent")
        assert resp.status_code == 404

    def test_update_rule_set_200(self, client, mock_service):
        body = {"name": "Updated Set"}
        resp = client.put(f"{BASE_URL}/rule-sets/rs-123", json=body)
        assert resp.status_code == 200

    def test_update_rule_set_404(self, client, mock_service):
        mock_service.update_rule_set.return_value = None
        body = {"name": "New Name"}
        resp = client.put(f"{BASE_URL}/rule-sets/rs-123", json=body)
        assert resp.status_code == 404

    def test_delete_rule_set_200(self, client, mock_service):
        resp = client.delete(f"{BASE_URL}/rule-sets/rs-123")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "deleted"
        assert data["set_id"] == "rs-123"

    def test_delete_rule_set_404(self, client, mock_service):
        mock_service.delete_rule_set.return_value = False
        resp = client.delete(f"{BASE_URL}/rule-sets/nonexistent")
        assert resp.status_code == 404


# ===========================================================================
# TestEvaluationEndpoints
# ===========================================================================


class TestEvaluationEndpoints:
    """Tests for evaluation endpoints."""

    def test_evaluate_200(self, client, mock_service):
        body = {"rule_set_id": str(uuid.uuid4()), "dataset": [{"co2e": 50.0}]}
        resp = client.post(f"{BASE_URL}/evaluate", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert "evaluation_id" in data

    def test_evaluate_calls_service(self, client, mock_service):
        rs_id = str(uuid.uuid4())
        body = {"rule_set_id": rs_id, "dataset": [{"val": 10}]}
        client.post(f"{BASE_URL}/evaluate", json=body)
        mock_service.evaluate.assert_called_once()

    def test_batch_evaluate_200(self, client, mock_service):
        body = {
            "rule_set_id": str(uuid.uuid4()),
            "datasets": [[{"val": 10}], [{"val": 20}]],
        }
        resp = client.post(f"{BASE_URL}/evaluate/batch", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert "batch_id" in data

    def test_batch_evaluate_calls_service(self, client, mock_service):
        body = {
            "rule_set_id": str(uuid.uuid4()),
            "datasets": [[{"val": 10}]],
        }
        client.post(f"{BASE_URL}/evaluate/batch", json=body)
        mock_service.batch_evaluate.assert_called_once()

    def test_get_evaluation_200(self, client, mock_service):
        resp = client.get(f"{BASE_URL}/evaluations/eval-123")
        assert resp.status_code == 200
        mock_service.get_evaluation.assert_called_once_with("eval-123")

    def test_get_evaluation_404(self, client, mock_service):
        mock_service.get_evaluation.return_value = None
        resp = client.get(f"{BASE_URL}/evaluations/nonexistent")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()


# ===========================================================================
# TestConflictEndpoints
# ===========================================================================


class TestConflictEndpoints:
    """Tests for conflict detection and listing endpoints."""

    def test_detect_conflicts_200(self, client, mock_service):
        body = {"rule_set_id": str(uuid.uuid4())}
        resp = client.post(f"{BASE_URL}/conflicts/detect", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert "detection_id" in data

    def test_detect_conflicts_calls_service(self, client, mock_service):
        rs_id = str(uuid.uuid4())
        body = {"rule_set_id": rs_id}
        client.post(f"{BASE_URL}/conflicts/detect", json=body)
        mock_service.detect_conflicts.assert_called_once()

    def test_list_conflicts_200(self, client, mock_service):
        resp = client.get(f"{BASE_URL}/conflicts")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    def test_list_conflicts_passes_filter_params(self, client, mock_service):
        client.get(
            f"{BASE_URL}/conflicts",
            params={
                "set_id": "rs-001",
                "conflict_type": "contradiction",
                "severity": "high",
                "limit": 50,
                "offset": 0,
            },
        )
        mock_service.list_conflicts.assert_called_once_with(
            set_id="rs-001",
            conflict_type="contradiction",
            severity="high",
            limit=50,
            offset=0,
        )


# ===========================================================================
# TestPackEndpoints
# ===========================================================================


class TestPackEndpoints:
    """Tests for rule pack endpoints."""

    def test_apply_pack_200(self, client, mock_service):
        body = {"version": "2.0"}
        resp = client.post(f"{BASE_URL}/packs/ghg_protocol/apply", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert data["pack_name"] == "ghg_protocol"

    def test_apply_pack_calls_service(self, client, mock_service):
        body = {"version": "1.0"}
        client.post(f"{BASE_URL}/packs/csrd_esrs/apply", json=body)
        mock_service.apply_pack.assert_called_once()
        call_args = mock_service.apply_pack.call_args
        assert call_args[0][0] == "csrd_esrs"

    def test_list_packs_200(self, client, mock_service):
        resp = client.get(f"{BASE_URL}/packs")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    def test_list_packs_passes_framework_filter(self, client, mock_service):
        client.get(
            f"{BASE_URL}/packs",
            params={"framework": "ghg_protocol", "limit": 10, "offset": 0},
        )
        mock_service.list_packs.assert_called_once_with(
            framework="ghg_protocol", limit=10, offset=0,
        )


# ===========================================================================
# TestReportEndpoints
# ===========================================================================


class TestReportEndpoints:
    """Tests for report generation endpoint."""

    def test_generate_report_200(self, client, mock_service):
        body = {
            "evaluation_id": str(uuid.uuid4()),
            "report_type": "compliance_report",
            "format": "json",
        }
        resp = client.post(f"{BASE_URL}/reports", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert "report_id" in data

    def test_generate_report_calls_service(self, client, mock_service):
        body = {
            "evaluation_id": "eval-001",
            "report_type": "audit_trail",
            "format": "html",
        }
        client.post(f"{BASE_URL}/reports", json=body)
        mock_service.generate_report.assert_called_once()


# ===========================================================================
# TestPipelineEndpoints
# ===========================================================================


class TestPipelineEndpoints:
    """Tests for the pipeline orchestration endpoint."""

    def test_run_pipeline_200(self, client, mock_service):
        body = {
            "dataset": [{"co2e": 50.0}],
            "pack_name": "ghg_protocol",
        }
        resp = client.post(f"{BASE_URL}/pipeline", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert "pipeline_id" in data

    def test_run_pipeline_calls_service(self, client, mock_service):
        body = {
            "dataset": [{"val": 10}],
            "pack_name": "csrd_esrs",
        }
        client.post(f"{BASE_URL}/pipeline", json=body)
        mock_service.run_pipeline.assert_called_once()


# ===========================================================================
# TestHealthEndpoint
# ===========================================================================


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    def test_health_200(self, client, mock_service):
        resp = client.get(f"{BASE_URL}/health")
        assert resp.status_code == 200

    def test_health_contains_status(self, client, mock_service):
        resp = client.get(f"{BASE_URL}/health")
        data = resp.json()
        assert "status" in data
        assert data["status"] == "healthy"

    def test_health_contains_engines(self, client, mock_service):
        resp = client.get(f"{BASE_URL}/health")
        data = resp.json()
        assert "engines" in data

    def test_health_calls_service(self, client, mock_service):
        client.get(f"{BASE_URL}/health")
        mock_service.get_health.assert_called_once()


# ===========================================================================
# TestErrorHandling
# ===========================================================================


class TestErrorHandling:
    """Tests for error handling across all endpoints.

    Uses ``error_client`` fixture which sets ``raise_server_exceptions=False``
    so that unhandled exceptions surface as 500 HTTP responses instead of
    being re-raised in the test process.
    """

    def test_register_rule_valueerror_raises(self, error_client, mock_service):
        mock_service.register_rule.side_effect = ValueError("Invalid rule type")
        body = {"name": "test", "rule_type": "bad_type", "severity": "error", "field": "x"}
        resp = error_client.post(f"{BASE_URL}/rules", json=body)
        # The router may return 400 or 500 depending on exception handling
        assert resp.status_code in (400, 422, 500)

    def test_evaluate_valueerror_raises(self, error_client, mock_service):
        mock_service.evaluate.side_effect = ValueError("Rule set not found")
        body = {"rule_set_id": "nonexistent", "dataset": [{"val": 1}]}
        resp = error_client.post(f"{BASE_URL}/evaluate", json=body)
        assert resp.status_code in (400, 422, 500)

    def test_detect_conflicts_valueerror_raises(self, error_client, mock_service):
        mock_service.detect_conflicts.side_effect = ValueError("Invalid set")
        body = {"rule_set_id": "bad"}
        resp = error_client.post(f"{BASE_URL}/conflicts/detect", json=body)
        assert resp.status_code in (400, 422, 500)

    def test_apply_pack_valueerror_raises(self, error_client, mock_service):
        mock_service.apply_pack.side_effect = ValueError("Unknown pack")
        body = {"version": "1.0"}
        resp = error_client.post(f"{BASE_URL}/packs/unknown_pack/apply", json=body)
        assert resp.status_code in (400, 422, 500)

    def test_generate_report_valueerror_raises(self, error_client, mock_service):
        mock_service.generate_report.side_effect = ValueError("Evaluation not found")
        body = {"evaluation_id": "bad", "report_type": "compliance_report", "format": "json"}
        resp = error_client.post(f"{BASE_URL}/reports", json=body)
        assert resp.status_code in (400, 422, 500)

    def test_run_pipeline_valueerror_raises(self, error_client, mock_service):
        mock_service.run_pipeline.side_effect = ValueError("Invalid config")
        body = {"dataset": [{"val": 1}], "pack_name": "bad"}
        resp = error_client.post(f"{BASE_URL}/pipeline", json=body)
        assert resp.status_code in (400, 422, 500)

    def test_register_rule_runtime_error_500(self, error_client, mock_service):
        mock_service.register_rule.side_effect = RuntimeError("DB crash")
        body = {"name": "test", "rule_type": "range_check", "severity": "error", "field": "x"}
        resp = error_client.post(f"{BASE_URL}/rules", json=body)
        assert resp.status_code == 500

    def test_list_rules_runtime_error_500(self, error_client, mock_service):
        mock_service.search_rules.side_effect = RuntimeError("DB down")
        resp = error_client.get(f"{BASE_URL}/rules")
        assert resp.status_code == 500

    def test_get_rule_runtime_error_500(self, error_client, mock_service):
        mock_service.get_rule.side_effect = RuntimeError("timeout")
        resp = error_client.get(f"{BASE_URL}/rules/abc")
        assert resp.status_code == 500

    def test_delete_rule_runtime_error_500(self, error_client, mock_service):
        mock_service.delete_rule.side_effect = RuntimeError("disk full")
        resp = error_client.delete(f"{BASE_URL}/rules/abc")
        assert resp.status_code == 500


# ===========================================================================
# TestServiceNotConfigured
# ===========================================================================


class TestServiceNotConfigured:
    """Tests for the case when service is not initialized."""

    def test_no_service_503(self):
        from fastapi import HTTPException as _HTTPException
        from greenlang.validation_rule_engine.api.router import router

        app = FastAPI()

        with patch(
            "greenlang.validation_rule_engine.api.router._get_service",
            side_effect=_HTTPException(
                status_code=503,
                detail="Validation Rule Engine service not initialized",
            ),
        ):
            app.include_router(router)
            no_svc_client = TestClient(app, raise_server_exceptions=False)
            resp = no_svc_client.get(f"{BASE_URL}/health")
            # The _get_service function raises HTTPException(503)
            assert resp.status_code == 503


# ===========================================================================
# TestPaginationQueryParams
# ===========================================================================


class TestPaginationQueryParams:
    """Tests for pagination defaults and bounds on list endpoints."""

    def test_rules_default_pagination(self, client, mock_service):
        client.get(f"{BASE_URL}/rules")
        kw = mock_service.search_rules.call_args.kwargs
        assert kw["limit"] == 100
        assert kw["offset"] == 0

    def test_rules_custom_pagination(self, client, mock_service):
        client.get(f"{BASE_URL}/rules", params={"limit": 25, "offset": 10})
        kw = mock_service.search_rules.call_args.kwargs
        assert kw["limit"] == 25
        assert kw["offset"] == 10

    def test_rule_sets_default_pagination(self, client, mock_service):
        client.get(f"{BASE_URL}/rule-sets")
        kw = mock_service.list_rule_sets.call_args.kwargs
        assert kw["limit"] == 100
        assert kw["offset"] == 0

    def test_conflicts_default_pagination(self, client, mock_service):
        client.get(f"{BASE_URL}/conflicts")
        kw = mock_service.list_conflicts.call_args.kwargs
        assert kw["limit"] == 100
        assert kw["offset"] == 0

    def test_packs_default_pagination(self, client, mock_service):
        client.get(f"{BASE_URL}/packs")
        kw = mock_service.list_packs.call_args.kwargs
        assert kw["limit"] == 100
        assert kw["offset"] == 0

    def test_limit_lower_bound_rejected(self, client, mock_service):
        resp = client.get(f"{BASE_URL}/rules", params={"limit": 0})
        assert resp.status_code == 422

    def test_limit_upper_bound_rejected(self, client, mock_service):
        resp = client.get(f"{BASE_URL}/rules", params={"limit": 1001})
        assert resp.status_code == 422

    def test_offset_negative_rejected(self, client, mock_service):
        resp = client.get(f"{BASE_URL}/rules", params={"offset": -1})
        assert resp.status_code == 422

    def test_limit_at_maximum(self, client, mock_service):
        resp = client.get(f"{BASE_URL}/rules", params={"limit": 1000})
        assert resp.status_code == 200
        kw = mock_service.search_rules.call_args.kwargs
        assert kw["limit"] == 1000

    def test_limit_at_minimum(self, client, mock_service):
        resp = client.get(f"{BASE_URL}/rules", params={"limit": 1})
        assert resp.status_code == 200
        kw = mock_service.search_rules.call_args.kwargs
        assert kw["limit"] == 1
