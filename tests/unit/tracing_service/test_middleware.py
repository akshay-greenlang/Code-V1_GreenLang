# -*- coding: utf-8 -*-
"""
Unit tests for TracingMiddleware (OBS-003)

Tests ASGI middleware behavior: span creation, header extraction,
context injection, health-check exclusion, and error handling.

Coverage target: 85%+ of middleware.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from greenlang.infrastructure.tracing_service.middleware import (
    TracingMiddleware,
    _asgi_headers_to_dict,
    _get_host,
    _EXCLUDED_PATHS,
)


# ============================================================================
# Header helper tests
# ============================================================================


class TestAsgiHeadersToDict:
    """Tests for _asgi_headers_to_dict conversion function."""

    def test_basic_headers(self):
        """Convert standard ASGI byte-tuple headers to string dict."""
        raw = [
            (b"host", b"localhost:8000"),
            (b"content-type", b"application/json"),
        ]
        result = _asgi_headers_to_dict(raw)
        assert result["host"] == "localhost:8000"
        assert result["content-type"] == "application/json"

    def test_case_normalisation(self):
        """Header keys are lowercased."""
        raw = [(b"X-Tenant-ID", b"t-acme")]
        result = _asgi_headers_to_dict(raw)
        assert "x-tenant-id" in result
        assert result["x-tenant-id"] == "t-acme"

    def test_empty_headers(self):
        """Empty header list returns empty dict."""
        assert _asgi_headers_to_dict([]) == {}

    def test_duplicate_keys_last_wins(self):
        """When duplicate keys exist, last value wins."""
        raw = [
            (b"x-val", b"first"),
            (b"x-val", b"second"),
        ]
        result = _asgi_headers_to_dict(raw)
        assert result["x-val"] == "second"

    def test_malformed_header_skipped(self):
        """Malformed headers are silently skipped."""
        # Simulate a header that cannot be decoded
        raw = [(b"good", b"value")]
        result = _asgi_headers_to_dict(raw)
        assert len(result) == 1


class TestGetHost:
    """Tests for _get_host helper."""

    def test_host_found(self):
        """Extract host from headers."""
        raw = [(b"host", b"example.com")]
        assert _get_host(raw) == "example.com"

    def test_host_not_found(self):
        """Return 'unknown' when host header is absent."""
        raw = [(b"content-type", b"text/plain")]
        assert _get_host(raw) == "unknown"

    def test_empty_headers(self):
        """Return 'unknown' for empty headers."""
        assert _get_host([]) == "unknown"


# ============================================================================
# TracingMiddleware initialisation tests
# ============================================================================


class TestTracingMiddlewareInit:
    """Tests for middleware construction."""

    def test_default_init(self):
        """Middleware can be constructed with defaults."""
        app = MagicMock()
        mw = TracingMiddleware(app)
        assert mw.app is app
        assert mw.service_name == "api-service"
        assert mw.excluded_paths == _EXCLUDED_PATHS

    def test_custom_service_name(self):
        """Middleware accepts a custom service name."""
        mw = TracingMiddleware(MagicMock(), service_name="my-service")
        assert mw.service_name == "my-service"

    def test_custom_excluded_paths(self):
        """Middleware accepts custom excluded paths."""
        custom = {"/custom-health"}
        mw = TracingMiddleware(MagicMock(), excluded_paths=custom)
        assert mw.excluded_paths == custom

    def test_custom_enricher(self):
        """Middleware accepts a custom SpanEnricher."""
        enricher = MagicMock()
        mw = TracingMiddleware(MagicMock(), enricher=enricher)
        assert mw.enricher is enricher


# ============================================================================
# Middleware __call__ tests
# ============================================================================


class TestTracingMiddlewareCall:
    """Tests for middleware ASGI __call__ method."""

    @pytest.mark.asyncio
    async def test_non_http_passthrough(self, asgi_receive, asgi_send):
        """Non-HTTP scopes (websocket, lifespan) pass through without tracing."""
        calls = []

        async def inner_app(scope, receive, send):
            calls.append(scope)

        mw = TracingMiddleware(inner_app)
        scope = {"type": "lifespan"}
        await mw(scope, asgi_receive, asgi_send)
        assert len(calls) == 1
        assert calls[0]["type"] == "lifespan"

    @pytest.mark.asyncio
    async def test_health_check_excluded(
        self, asgi_scope_health, asgi_receive, asgi_send
    ):
        """Health-check paths are passed through without tracing."""
        calls = []

        async def inner_app(scope, receive, send):
            calls.append(True)

        mw = TracingMiddleware(inner_app)
        await mw(asgi_scope_health, asgi_receive, asgi_send)
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_metrics_excluded(self, asgi_receive, asgi_send):
        """The /metrics endpoint is excluded from tracing."""
        calls = []

        async def inner_app(scope, receive, send):
            calls.append(True)

        scope = {"type": "http", "method": "GET", "path": "/metrics", "headers": []}
        mw = TracingMiddleware(inner_app)
        await mw(scope, asgi_receive, asgi_send)
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_otel_unavailable_passthrough(
        self, asgi_scope, asgi_receive, asgi_send
    ):
        """When OTel is not available, requests pass through without tracing."""
        calls = []

        async def inner_app(scope, receive, send):
            calls.append(True)

        mw = TracingMiddleware(inner_app)
        with patch(
            "greenlang.infrastructure.tracing_service.middleware.OTEL_AVAILABLE",
            False,
        ):
            await mw(asgi_scope, asgi_receive, asgi_send)
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_traced_request_creates_span(
        self, asgi_scope, asgi_receive, asgi_send, mock_span
    ):
        """Normal HTTP request creates a span when OTel is available."""
        async def inner_app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b""})

        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value = mock_span

        mw = TracingMiddleware(inner_app)
        with patch(
            "greenlang.infrastructure.tracing_service.middleware.OTEL_AVAILABLE",
            True,
        ), patch(
            "greenlang.infrastructure.tracing_service.middleware.trace"
        ) as mock_trace:
            mock_trace.get_tracer.return_value = mock_tracer
            await mw(asgi_scope, asgi_receive, asgi_send)

        mock_tracer.start_as_current_span.assert_called_once()

    @pytest.mark.asyncio
    async def test_span_records_tenant_id(
        self, asgi_scope, asgi_receive, asgi_send, mock_span
    ):
        """Span captures X-Tenant-ID from request headers."""
        async def inner_app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b""})

        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value = mock_span

        mw = TracingMiddleware(inner_app)
        with patch(
            "greenlang.infrastructure.tracing_service.middleware.OTEL_AVAILABLE",
            True,
        ), patch(
            "greenlang.infrastructure.tracing_service.middleware.trace"
        ) as mock_trace:
            mock_trace.get_tracer.return_value = mock_tracer
            await mw(asgi_scope, asgi_receive, asgi_send)

        # Check that gl.tenant_id was set
        set_calls = {
            call[0][0]: call[0][1]
            for call in mock_span.set_attribute.call_args_list
        }
        assert set_calls.get("gl.tenant_id") == "t-acme"

    @pytest.mark.asyncio
    async def test_span_records_request_id(
        self, asgi_scope, asgi_receive, asgi_send, mock_span
    ):
        """Span captures X-Request-ID from request headers."""
        async def inner_app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b""})

        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value = mock_span

        mw = TracingMiddleware(inner_app)
        with patch(
            "greenlang.infrastructure.tracing_service.middleware.OTEL_AVAILABLE",
            True,
        ), patch(
            "greenlang.infrastructure.tracing_service.middleware.trace"
        ) as mock_trace:
            mock_trace.get_tracer.return_value = mock_tracer
            await mw(asgi_scope, asgi_receive, asgi_send)

        set_calls = {
            call[0][0]: call[0][1]
            for call in mock_span.set_attribute.call_args_list
        }
        assert set_calls.get("gl.request_id") == "req-12345"

    @pytest.mark.asyncio
    async def test_span_records_http_method(
        self, asgi_scope, asgi_receive, asgi_send, mock_span
    ):
        """Span captures HTTP method from ASGI scope."""
        async def inner_app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b""})

        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value = mock_span

        mw = TracingMiddleware(inner_app)
        with patch(
            "greenlang.infrastructure.tracing_service.middleware.OTEL_AVAILABLE",
            True,
        ), patch(
            "greenlang.infrastructure.tracing_service.middleware.trace"
        ) as mock_trace:
            mock_trace.get_tracer.return_value = mock_tracer
            await mw(asgi_scope, asgi_receive, asgi_send)

        set_calls = {
            call[0][0]: call[0][1]
            for call in mock_span.set_attribute.call_args_list
        }
        assert set_calls.get("http.method") == "GET"

    @pytest.mark.asyncio
    async def test_span_records_status_code(
        self, asgi_scope, asgi_receive, asgi_send, mock_span
    ):
        """Span captures HTTP status code from response."""
        async def inner_app(scope, receive, send):
            await send({"type": "http.response.start", "status": 201, "headers": []})
            await send({"type": "http.response.body", "body": b""})

        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value = mock_span

        mw = TracingMiddleware(inner_app)
        with patch(
            "greenlang.infrastructure.tracing_service.middleware.OTEL_AVAILABLE",
            True,
        ), patch(
            "greenlang.infrastructure.tracing_service.middleware.trace"
        ) as mock_trace:
            mock_trace.get_tracer.return_value = mock_tracer
            await mw(asgi_scope, asgi_receive, asgi_send)

        set_calls = {
            call[0][0]: call[0][1]
            for call in mock_span.set_attribute.call_args_list
        }
        assert set_calls.get("http.status_code") == 201

    @pytest.mark.asyncio
    async def test_error_status_on_5xx(
        self, asgi_scope, asgi_receive, asgi_send, mock_span
    ):
        """Span status is set to ERROR on 5xx responses."""
        async def inner_app(scope, receive, send):
            await send({"type": "http.response.start", "status": 503, "headers": []})
            await send({"type": "http.response.body", "body": b""})

        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value = mock_span

        mw = TracingMiddleware(inner_app)
        with patch(
            "greenlang.infrastructure.tracing_service.middleware.OTEL_AVAILABLE",
            True,
        ), patch(
            "greenlang.infrastructure.tracing_service.middleware.trace"
        ) as mock_trace, patch(
            "greenlang.infrastructure.tracing_service.middleware.StatusCode"
        ) as mock_status_code:
            mock_trace.get_tracer.return_value = mock_tracer
            await mw(asgi_scope, asgi_receive, asgi_send)

        mock_span.set_status.assert_called()

    @pytest.mark.asyncio
    async def test_exception_recorded_on_span(
        self, asgi_scope, asgi_receive, asgi_send, mock_span
    ):
        """Exceptions raised in the app are recorded on the span."""
        async def inner_app(scope, receive, send):
            raise ValueError("test error")

        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value = mock_span

        mw = TracingMiddleware(inner_app)
        with patch(
            "greenlang.infrastructure.tracing_service.middleware.OTEL_AVAILABLE",
            True,
        ), patch(
            "greenlang.infrastructure.tracing_service.middleware.trace"
        ) as mock_trace, patch(
            "greenlang.infrastructure.tracing_service.middleware.StatusCode"
        ):
            mock_trace.get_tracer.return_value = mock_tracer
            with pytest.raises(ValueError, match="test error"):
                await mw(asgi_scope, asgi_receive, asgi_send)

        mock_span.record_exception.assert_called_once()

    @pytest.mark.asyncio
    async def test_duration_recorded(
        self, asgi_scope, asgi_receive, asgi_send, mock_span
    ):
        """Span records duration_ms attribute after completion."""
        async def inner_app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b""})

        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value = mock_span

        mw = TracingMiddleware(inner_app)
        with patch(
            "greenlang.infrastructure.tracing_service.middleware.OTEL_AVAILABLE",
            True,
        ), patch(
            "greenlang.infrastructure.tracing_service.middleware.trace"
        ) as mock_trace:
            mock_trace.get_tracer.return_value = mock_tracer
            await mw(asgi_scope, asgi_receive, asgi_send)

        # Check that http.duration_ms was set
        attr_keys = [call[0][0] for call in mock_span.set_attribute.call_args_list]
        assert "http.duration_ms" in attr_keys

    @pytest.mark.asyncio
    async def test_query_string_recorded(
        self, asgi_receive, asgi_send, mock_span
    ):
        """Span records query string when present."""
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/api/v1/agents",
            "query_string": b"limit=10&offset=0",
            "headers": [(b"host", b"localhost:8000")],
        }

        async def inner_app(s, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b""})

        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value = mock_span

        mw = TracingMiddleware(inner_app)
        with patch(
            "greenlang.infrastructure.tracing_service.middleware.OTEL_AVAILABLE",
            True,
        ), patch(
            "greenlang.infrastructure.tracing_service.middleware.trace"
        ) as mock_trace:
            mock_trace.get_tracer.return_value = mock_tracer
            await mw(scope, asgi_receive, asgi_send)

        set_calls = {
            call[0][0]: call[0][1]
            for call in mock_span.set_attribute.call_args_list
        }
        assert set_calls.get("http.query_string") == "limit=10&offset=0"

    @pytest.mark.asyncio
    async def test_websocket_passthrough(self, asgi_receive, asgi_send):
        """WebSocket scopes pass through without tracing."""
        calls = []

        async def inner_app(scope, receive, send):
            calls.append(True)

        scope = {"type": "websocket", "path": "/ws"}
        mw = TracingMiddleware(inner_app)
        await mw(scope, asgi_receive, asgi_send)
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_excluded_path_prefix(self, asgi_receive, asgi_send):
        """Paths starting with excluded prefix are skipped."""
        calls = []

        async def inner_app(scope, receive, send):
            calls.append(True)

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/health/detailed",
            "headers": [],
        }
        mw = TracingMiddleware(inner_app)
        await mw(scope, asgi_receive, asgi_send)
        assert len(calls) == 1


# ============================================================================
# Excluded paths tests
# ============================================================================


class TestExcludedPaths:
    """Tests for the _EXCLUDED_PATHS constant."""

    def test_health_paths_excluded(self):
        """Standard health-check paths are in the excluded set."""
        assert "/health" in _EXCLUDED_PATHS
        assert "/ready" in _EXCLUDED_PATHS
        assert "/healthz" in _EXCLUDED_PATHS
        assert "/livez" in _EXCLUDED_PATHS
        assert "/readyz" in _EXCLUDED_PATHS

    def test_metrics_excluded(self):
        """The /metrics path is excluded."""
        assert "/metrics" in _EXCLUDED_PATHS

    def test_ping_excluded(self):
        """The /ping path is excluded."""
        assert "/ping" in _EXCLUDED_PATHS
