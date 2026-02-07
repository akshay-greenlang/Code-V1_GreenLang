# -*- coding: utf-8 -*-
"""
API Integration Tests for Schema Service (AGENT-FOUND-002)

Tests the FastAPI endpoints using a self-contained test application
with mocked backend services. Validates HTTP responses, status codes,
error formats, and content types.

Endpoints tested:
    POST /v1/schema/validate        - Single payload validation
    POST /v1/schema/validate/batch  - Batch payload validation
    POST /v1/schema/compile         - Schema compilation
    GET  /health                    - Health check
    GET  /metrics                   - Prometheus metrics (text)
    Error responses                 - GLSCHEMA error code format
    Large payloads                  - Rejection of oversized requests

Author: GreenLang Platform Team
Date: February 2026
"""

import json
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.testclient import TestClient
    from fastapi.responses import PlainTextResponse, JSONResponse

    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False


# ---------------------------------------------------------------------------
# Inline API app for self-contained testing
# ---------------------------------------------------------------------------


def _create_test_app():
    """Create a minimal FastAPI app mirroring the schema service API."""
    if not _FASTAPI_AVAILABLE:
        return None

    app = FastAPI(title="GreenLang Schema Service Test")

    # In-memory schema registry for testing
    _schemas: Dict[str, Dict[str, Any]] = {}

    @app.post("/v1/schema/validate")
    async def validate_payload(request: Request):
        body = await request.json()
        payload = body.get("payload")
        schema = body.get("schema")

        if payload is None:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "code": "GLSCHEMA-API-400",
                        "message": "Missing 'payload' field",
                    }
                },
            )

        if schema is None:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "code": "GLSCHEMA-API-400",
                        "message": "Missing 'schema' field",
                    }
                },
            )

        # Check payload size
        payload_bytes = len(json.dumps(payload).encode("utf-8"))
        if payload_bytes > 1_048_576:  # 1MB
            return JSONResponse(
                status_code=413,
                content={
                    "error": {
                        "code": "GLSCHEMA-API-413",
                        "message": "Payload exceeds maximum size (1MB)",
                    }
                },
            )

        # Simulate validation
        findings = []
        valid = True

        required = schema.get("required", [])
        for field in required:
            if field not in payload:
                findings.append({
                    "code": "GLSCHEMA-E100",
                    "severity": "error",
                    "path": f"/{field}",
                    "message": f"Missing required field: {field}",
                })
                valid = False

        return {
            "valid": valid,
            "schema_hash": "a" * 64,
            "summary": {
                "error_count": len([f for f in findings if f["severity"] == "error"]),
                "warning_count": len([f for f in findings if f["severity"] == "warning"]),
            },
            "findings": findings,
        }

    @app.post("/v1/schema/validate/batch")
    async def validate_batch(request: Request):
        body = await request.json()
        payloads = body.get("payloads", [])
        schema = body.get("schema")

        if not payloads:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "code": "GLSCHEMA-API-400",
                        "message": "Missing or empty 'payloads' field",
                    }
                },
            )

        results = []
        valid_count = 0
        total_errors = 0

        for i, payload in enumerate(payloads):
            findings = []
            item_valid = True

            required = schema.get("required", []) if schema else []
            for field in required:
                if field not in payload:
                    findings.append({
                        "code": "GLSCHEMA-E100",
                        "severity": "error",
                        "path": f"/{field}",
                        "message": f"Missing required field: {field}",
                    })
                    item_valid = False

            if item_valid:
                valid_count += 1
            else:
                total_errors += len(findings)

            results.append({
                "index": i,
                "valid": item_valid,
                "findings": findings,
            })

        return {
            "summary": {
                "total_items": len(payloads),
                "valid_count": valid_count,
                "error_count": total_errors,
            },
            "results": results,
        }

    @app.post("/v1/schema/compile")
    async def compile_schema(request: Request):
        body = await request.json()
        schema = body.get("schema")

        if schema is None:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "code": "GLSCHEMA-API-400",
                        "message": "Missing 'schema' field",
                    }
                },
            )

        return {
            "schema_id": "inline/schema",
            "version": "1.0.0",
            "schema_hash": "b" * 64,
            "properties": len(schema.get("properties", {})),
            "rules": len(schema.get("$rules", [])),
            "compile_time_ms": 1.23,
        }

    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "version": "1.0.0",
            "components": {
                "validator": "ok",
                "compiler": "ok",
                "registry": "ok",
            },
        }

    @app.get("/metrics")
    async def metrics():
        metrics_text = (
            "# HELP gl_schema_validations_total Total validations\n"
            "# TYPE gl_schema_validations_total counter\n"
            "gl_schema_validations_total{status=\"valid\"} 42\n"
            "gl_schema_validations_total{status=\"invalid\"} 8\n"
        )
        return PlainTextResponse(content=metrics_text, media_type="text/plain")

    return app


# ===========================================================================
# Test Classes
# ===========================================================================


@pytest.mark.skipif(not _FASTAPI_AVAILABLE, reason="fastapi not installed")
class TestValidateEndpoint:
    """Test POST /v1/schema/validate."""

    @pytest.fixture(autouse=True)
    def setup_client(self, restore_sockets):
        app = _create_test_app()
        self.client = TestClient(app)

    def test_valid_payload_returns_200(self, emissions_schema, valid_emissions_payload):
        response = self.client.post(
            "/v1/schema/validate",
            json={"payload": valid_emissions_payload, "schema": emissions_schema},
        )
        assert response.status_code == 200

    def test_valid_payload_result(self, emissions_schema, valid_emissions_payload):
        response = self.client.post(
            "/v1/schema/validate",
            json={"payload": valid_emissions_payload, "schema": emissions_schema},
        )
        data = response.json()
        assert data["valid"] is True
        assert "schema_hash" in data
        assert data["summary"]["error_count"] == 0

    def test_invalid_payload_returns_200(self, emissions_schema, invalid_emissions_payload):
        """Invalid data should still return 200 (validation errors in body)."""
        response = self.client.post(
            "/v1/schema/validate",
            json={"payload": invalid_emissions_payload, "schema": emissions_schema},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False
        assert data["summary"]["error_count"] > 0

    def test_invalid_payload_has_findings(self, emissions_schema, invalid_emissions_payload):
        response = self.client.post(
            "/v1/schema/validate",
            json={"payload": invalid_emissions_payload, "schema": emissions_schema},
        )
        data = response.json()
        assert len(data["findings"]) > 0

    def test_missing_payload_returns_400(self, emissions_schema):
        response = self.client.post(
            "/v1/schema/validate",
            json={"schema": emissions_schema},
        )
        assert response.status_code == 400
        data = response.json()
        assert "GLSCHEMA-API-400" in data["error"]["code"]

    def test_missing_schema_returns_400(self, valid_emissions_payload):
        response = self.client.post(
            "/v1/schema/validate",
            json={"payload": valid_emissions_payload},
        )
        assert response.status_code == 400


@pytest.mark.skipif(not _FASTAPI_AVAILABLE, reason="fastapi not installed")
class TestValidateBatchEndpoint:
    """Test POST /v1/schema/validate/batch."""

    @pytest.fixture(autouse=True)
    def setup_client(self, restore_sockets):
        app = _create_test_app()
        self.client = TestClient(app)

    def test_batch_returns_200(self, emissions_schema, batch_payloads):
        response = self.client.post(
            "/v1/schema/validate/batch",
            json={"payloads": batch_payloads, "schema": emissions_schema},
        )
        assert response.status_code == 200

    def test_batch_summary(self, emissions_schema, batch_payloads):
        response = self.client.post(
            "/v1/schema/validate/batch",
            json={"payloads": batch_payloads, "schema": emissions_schema},
        )
        data = response.json()
        assert data["summary"]["total_items"] == len(batch_payloads)

    def test_batch_results_indexed(self, emissions_schema, batch_payloads):
        response = self.client.post(
            "/v1/schema/validate/batch",
            json={"payloads": batch_payloads, "schema": emissions_schema},
        )
        data = response.json()
        for i, result in enumerate(data["results"]):
            assert result["index"] == i

    def test_empty_batch_returns_400(self, emissions_schema):
        response = self.client.post(
            "/v1/schema/validate/batch",
            json={"payloads": [], "schema": emissions_schema},
        )
        assert response.status_code == 400


@pytest.mark.skipif(not _FASTAPI_AVAILABLE, reason="fastapi not installed")
class TestCompileEndpoint:
    """Test POST /v1/schema/compile."""

    @pytest.fixture(autouse=True)
    def setup_client(self, restore_sockets):
        app = _create_test_app()
        self.client = TestClient(app)

    def test_compile_returns_200(self, emissions_schema):
        response = self.client.post(
            "/v1/schema/compile",
            json={"schema": emissions_schema},
        )
        assert response.status_code == 200

    def test_compile_returns_hash(self, emissions_schema):
        response = self.client.post(
            "/v1/schema/compile",
            json={"schema": emissions_schema},
        )
        data = response.json()
        assert len(data["schema_hash"]) == 64

    def test_compile_returns_property_count(self, emissions_schema):
        response = self.client.post(
            "/v1/schema/compile",
            json={"schema": emissions_schema},
        )
        data = response.json()
        assert data["properties"] == len(emissions_schema.get("properties", {}))

    def test_compile_missing_schema_returns_400(self):
        response = self.client.post(
            "/v1/schema/compile",
            json={},
        )
        assert response.status_code == 400


@pytest.mark.skipif(not _FASTAPI_AVAILABLE, reason="fastapi not installed")
class TestHealthEndpoint:
    """Test GET /health."""

    @pytest.fixture(autouse=True)
    def setup_client(self, restore_sockets):
        app = _create_test_app()
        self.client = TestClient(app)

    def test_health_returns_200(self):
        response = self.client.get("/health")
        assert response.status_code == 200

    def test_health_returns_healthy(self):
        response = self.client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_has_components(self):
        response = self.client.get("/health")
        data = response.json()
        assert "components" in data
        assert data["components"]["validator"] == "ok"


@pytest.mark.skipif(not _FASTAPI_AVAILABLE, reason="fastapi not installed")
class TestMetricsEndpoint:
    """Test GET /metrics returns Prometheus format."""

    @pytest.fixture(autouse=True)
    def setup_client(self, restore_sockets):
        app = _create_test_app()
        self.client = TestClient(app)

    def test_metrics_returns_200(self):
        response = self.client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_content_type(self):
        response = self.client.get("/metrics")
        assert "text/plain" in response.headers["content-type"]

    def test_metrics_contains_counter(self):
        response = self.client.get("/metrics")
        assert "gl_schema_validations_total" in response.text


@pytest.mark.skipif(not _FASTAPI_AVAILABLE, reason="fastapi not installed")
class TestErrorResponseFormat:
    """Test error responses have correct GLSCHEMA error codes."""

    @pytest.fixture(autouse=True)
    def setup_client(self, restore_sockets):
        app = _create_test_app()
        self.client = TestClient(app)

    def test_error_has_code(self, emissions_schema):
        response = self.client.post(
            "/v1/schema/validate",
            json={"schema": emissions_schema},  # Missing payload
        )
        data = response.json()
        assert "code" in data["error"]
        assert data["error"]["code"].startswith("GLSCHEMA-")

    def test_error_has_message(self, emissions_schema):
        response = self.client.post(
            "/v1/schema/validate",
            json={"schema": emissions_schema},
        )
        data = response.json()
        assert "message" in data["error"]
        assert len(data["error"]["message"]) > 0


@pytest.mark.skipif(not _FASTAPI_AVAILABLE, reason="fastapi not installed")
class TestLargePayloadRejection:
    """Test large payload rejection."""

    @pytest.fixture(autouse=True)
    def setup_client(self, restore_sockets):
        app = _create_test_app()
        self.client = TestClient(app)

    def test_oversized_payload_returns_413(self, emissions_schema):
        """Payloads over 1MB should be rejected."""
        large_payload = {"data": "x" * (1_048_577)}  # Just over 1MB
        response = self.client.post(
            "/v1/schema/validate",
            json={"payload": large_payload, "schema": emissions_schema},
        )
        assert response.status_code == 413
        data = response.json()
        assert "GLSCHEMA-API-413" in data["error"]["code"]
