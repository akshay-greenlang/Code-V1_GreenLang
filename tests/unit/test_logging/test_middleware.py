# -*- coding: utf-8 -*-
"""
Unit Tests for FastAPI Structured Logging Middleware - INFRA-009

Tests StructuredLoggingMiddleware: request-ID generation and propagation,
trace/tenant header extraction, request/response lifecycle logging, health
endpoint exclusion, error handling, duration measurement, and context cleanup.

Module under test: greenlang.infrastructure.logging.middleware
"""

import json
import logging
import time
import uuid

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from greenlang.infrastructure.logging.config import LoggingConfig
from greenlang.infrastructure.logging.middleware import StructuredLoggingMiddleware


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def app() -> FastAPI:
    """Create a FastAPI app with StructuredLoggingMiddleware installed."""
    application = FastAPI()
    application.add_middleware(StructuredLoggingMiddleware)

    @application.get("/api/data")
    async def get_data():
        return {"status": "ok"}

    @application.get("/api/error")
    async def get_error():
        raise ValueError("intentional test error")

    @application.get("/api/slow")
    async def get_slow():
        time.sleep(0.05)  # 50 ms
        return {"status": "slow_ok"}

    @application.get("/health")
    async def health():
        return {"status": "healthy"}

    @application.get("/ready")
    async def ready():
        return {"status": "ready"}

    @application.get("/metrics")
    async def metrics():
        return {"metrics": []}

    @application.get("/api/server-error")
    async def server_error():
        return JSONResponse(status_code=500, content={"error": "boom"})

    return application


@pytest.fixture
def client(app) -> TestClient:
    """TestClient wrapping the FastAPI app."""
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Request-ID Tests
# ---------------------------------------------------------------------------


class TestRequestId:
    """Tests for X-Request-ID generation and propagation."""

    def test_middleware_generates_request_id(self, client):
        """When no X-Request-ID header is sent, the middleware generates one."""
        resp = client.get("/api/data")
        assert resp.status_code == 200
        request_id = resp.headers.get("X-Request-ID")
        assert request_id is not None
        assert len(request_id) > 0
        # Should be a valid UUID4
        uuid.UUID(request_id, version=4)

    def test_middleware_preserves_existing_request_id(self, client):
        """When the caller provides X-Request-ID, the same value is echoed back."""
        custom_id = "custom-req-" + uuid.uuid4().hex[:8]
        resp = client.get("/api/data", headers={"X-Request-ID": custom_id})
        assert resp.status_code == 200
        returned_id = resp.headers.get("X-Request-ID")
        assert returned_id == custom_id


# ---------------------------------------------------------------------------
# Header Extraction Tests
# ---------------------------------------------------------------------------


class TestHeaderExtraction:
    """Tests for trace-ID and tenant-ID extraction from request headers."""

    def test_middleware_extracts_trace_id(self, client, caplog):
        """X-Trace-ID header value is logged and propagated on response."""
        trace_id = "trace-" + uuid.uuid4().hex[:12]
        with caplog.at_level(logging.DEBUG):
            resp = client.get(
                "/api/data", headers={"X-Trace-ID": trace_id}
            )
        assert resp.status_code == 200
        # Trace ID should be echoed in response header
        assert resp.headers.get("X-Trace-ID") == trace_id

    def test_middleware_extracts_tenant_id(self, client, caplog):
        """X-Tenant-ID header value is captured into the log context."""
        tenant_id = "tenant-acme-001"
        with caplog.at_level(logging.DEBUG):
            resp = client.get(
                "/api/data", headers={"X-Tenant-ID": tenant_id}
            )
        assert resp.status_code == 200
        # Tenant ID should appear in at least one log record extra
        found = any(
            tenant_id in r.getMessage()
            or tenant_id in str(getattr(r, "__dict__", {}))
            for r in caplog.records
        )
        assert found, f"tenant_id '{tenant_id}' not found in log records"


# ---------------------------------------------------------------------------
# Lifecycle Logging Tests
# ---------------------------------------------------------------------------


class TestLifecycleLogging:
    """Tests for request_started and request_completed log events."""

    def test_middleware_logs_request_started(self, client, caplog):
        """A 'request_started' log event is emitted for non-skip paths."""
        with caplog.at_level(logging.DEBUG):
            client.get("/api/data")

        messages = " ".join(r.getMessage() for r in caplog.records)
        assert "request_started" in messages, (
            f"Expected 'request_started' log. Got: {messages[:500]}"
        )

    def test_middleware_logs_request_completed(self, client, caplog):
        """A 'request_completed' log event is emitted with status_code."""
        with caplog.at_level(logging.DEBUG):
            client.get("/api/data")

        messages = " ".join(r.getMessage() for r in caplog.records)
        assert "request_completed" in messages, (
            f"Expected 'request_completed' log. Got: {messages[:500]}"
        )

    def test_middleware_measures_duration(self, client, caplog):
        """The completed log event includes a positive duration_ms value."""
        with caplog.at_level(logging.DEBUG):
            client.get("/api/slow")

        # Look for duration_ms in the extra dict of log records
        for record in caplog.records:
            duration = getattr(record, "duration_ms", None)
            if duration is not None:
                assert float(duration) > 0
                return  # Found it

        # Fallback: check inside the message string
        messages = " ".join(r.getMessage() for r in caplog.records)
        assert "duration" in messages.lower() or "ms" in messages.lower(), (
            "Expected duration_ms in log records"
        )


# ---------------------------------------------------------------------------
# Endpoint Exclusion Tests
# ---------------------------------------------------------------------------


class TestEndpointExclusion:
    """Tests for skipping log noise on health/ready/metrics endpoints."""

    def test_middleware_skips_health_endpoint(self, client, caplog):
        """/health endpoint does not produce request_started logs."""
        with caplog.at_level(logging.DEBUG):
            resp = client.get("/health")
        assert resp.status_code == 200

        messages = " ".join(r.getMessage() for r in caplog.records)
        assert "request_started" not in messages

    def test_middleware_skips_ready_endpoint(self, client, caplog):
        """/ready endpoint does not produce request_started logs."""
        with caplog.at_level(logging.DEBUG):
            resp = client.get("/ready")
        assert resp.status_code == 200

        messages = " ".join(r.getMessage() for r in caplog.records)
        assert "request_started" not in messages

    def test_middleware_skips_metrics_endpoint(self, client, caplog):
        """/metrics endpoint does not produce request_started logs."""
        with caplog.at_level(logging.DEBUG):
            resp = client.get("/metrics")
        assert resp.status_code == 200

        messages = " ".join(r.getMessage() for r in caplog.records)
        assert "request_started" not in messages


# ---------------------------------------------------------------------------
# Error Handling Tests
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for error and exception logging in middleware."""

    def test_middleware_logs_errors(self, client, caplog):
        """HTTP 500 responses are logged at WARNING level (status >= 400)."""
        with caplog.at_level(logging.DEBUG):
            resp = client.get("/api/server-error")
        assert resp.status_code == 500

        # The middleware logs 4xx+ responses at WARNING level
        warning_plus = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warning_plus) >= 1, "Expected WARNING+ log for 500 response"

    def test_middleware_handles_exceptions(self, client, caplog):
        """Unhandled exceptions are caught, logged at ERROR, and return 500."""
        with caplog.at_level(logging.DEBUG):
            resp = client.get("/api/error")

        assert resp.status_code == 500

        # Should see an ERROR-level record with request_failed
        error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert len(error_records) >= 1, (
            "Expected at least one ERROR log for unhandled exception"
        )

        # At least one record should reference the ValueError or request_failed
        all_messages = " ".join(r.getMessage() for r in error_records)
        assert (
            "ValueError" in all_messages
            or "intentional test error" in all_messages
            or "request_failed" in all_messages
            or "error" in all_messages.lower()
        )


# ---------------------------------------------------------------------------
# Context Cleanup Tests
# ---------------------------------------------------------------------------


class TestContextCleanup:
    """Tests verifying request context does not leak between requests."""

    def test_middleware_clears_context(self, client, caplog):
        """Context variables are cleared after the request completes.

        We send two sequential requests with different tenant IDs and verify
        the second request does not contain context from the first.
        """
        # First request with a specific tenant
        with caplog.at_level(logging.DEBUG):
            client.get("/api/data", headers={"X-Tenant-ID": "tenant-first"})

        caplog.clear()

        # Second request with a different tenant
        with caplog.at_level(logging.DEBUG):
            client.get("/api/data", headers={"X-Tenant-ID": "tenant-second"})

        # The second batch of records should NOT contain tenant-first
        second_messages = " ".join(r.getMessage() for r in caplog.records)
        second_extras = " ".join(str(getattr(r, "__dict__", "")) for r in caplog.records)
        combined = second_messages + second_extras
        assert "tenant-first" not in combined


# ---------------------------------------------------------------------------
# Default Configuration Test
# ---------------------------------------------------------------------------


class TestDefaultConfig:
    """Tests for middleware operation without explicit configuration."""

    def test_middleware_with_default_config(self):
        """Middleware works when added without explicit LoggingConfig."""
        app = FastAPI()
        # Should not raise
        app.add_middleware(StructuredLoggingMiddleware)

        @app.get("/ping")
        async def ping():
            return {"pong": True}

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/ping")
        assert resp.status_code == 200
        assert resp.json() == {"pong": True}
        # Should have a request ID header
        assert resp.headers.get("X-Request-ID") is not None
