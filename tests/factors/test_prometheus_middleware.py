# -*- coding: utf-8 -*-
"""
Tests for Factors Prometheus ASGI middleware and convenience helpers.

Covers:
  - FactorsMetricsMiddleware path filtering and status capture
  - Path normalization for label cardinality control
  - Convenience helper functions (record_search_results, record_match_score, etc.)
  - /metrics endpoint response format
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from greenlang.factors.observability.prometheus import (
    FactorsMetricsMiddleware,
    _normalize_path,
    record_match_score,
    record_qa_failure,
    record_search_results,
    update_edition_gauge,
)


# ---------------------------------------------------------------------------
# Path normalization
# ---------------------------------------------------------------------------


class TestNormalizePath:
    """Test _normalize_path cardinality control."""

    def test_root_factors_path(self):
        assert _normalize_path("/api/v1/factors") == "/api/v1/factors"

    def test_search_path_unchanged(self):
        assert _normalize_path("/api/v1/factors/search") == "/api/v1/factors/search"

    def test_search_v2_path_unchanged(self):
        assert _normalize_path("/api/v1/factors/search/v2") == "/api/v1/factors/search/v2"

    def test_search_facets_path_unchanged(self):
        assert _normalize_path("/api/v1/factors/search/facets") == "/api/v1/factors/search/facets"

    def test_match_path_unchanged(self):
        assert _normalize_path("/api/v1/factors/match") == "/api/v1/factors/match"

    def test_export_path_unchanged(self):
        assert _normalize_path("/api/v1/factors/export") == "/api/v1/factors/export"

    def test_coverage_path_unchanged(self):
        assert _normalize_path("/api/v1/factors/coverage") == "/api/v1/factors/coverage"

    def test_factor_id_replaced(self):
        result = _normalize_path("/api/v1/factors/abc-123-def")
        assert result == "/api/v1/factors/{factor_id}"

    def test_factor_id_audit_bundle(self):
        result = _normalize_path("/api/v1/factors/abc-123/audit-bundle")
        assert result == "/api/v1/factors/{factor_id}/audit-bundle"

    def test_factor_id_diff(self):
        result = _normalize_path("/api/v1/factors/abc-123/diff")
        assert result == "/api/v1/factors/{factor_id}/diff"

    def test_trailing_slash_stripped(self):
        result = _normalize_path("/api/v1/factors/search/")
        assert result == "/api/v1/factors/search"

    def test_short_path(self):
        assert _normalize_path("/api/v1") == "/api/v1"

    def test_health_path_unchanged(self):
        assert _normalize_path("/api/v1/factors/health") == "/api/v1/factors/health"

    def test_metrics_path_unchanged(self):
        assert _normalize_path("/api/v1/factors/metrics") == "/api/v1/factors/metrics"


# ---------------------------------------------------------------------------
# ASGI Middleware
# ---------------------------------------------------------------------------


def _make_scope(method: str = "GET", path: str = "/api/v1/factors/search") -> dict:
    """Create a minimal ASGI HTTP scope."""
    return {
        "type": "http",
        "method": method,
        "path": path,
    }


def _make_ws_scope(path: str = "/ws") -> dict:
    """Create a minimal ASGI WebSocket scope."""
    return {"type": "websocket", "path": path}


class TestFactorsMetricsMiddleware:
    """Tests for the ASGI middleware."""

    @pytest.fixture()
    def metrics_mock(self):
        """Patch the singleton to avoid touching real Prometheus collectors."""
        mock = MagicMock()
        with patch(
            "greenlang.factors.observability.prometheus.get_factors_metrics",
            return_value=mock,
        ):
            yield mock

    def _run(self, coro):
        """Run an async coroutine in a new event loop."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    def test_non_factors_path_passes_through(self, metrics_mock):
        """Requests not starting with /api/v1/factors should not be instrumented."""
        inner_app = AsyncMock()
        mw = FactorsMetricsMiddleware(inner_app)

        scope = _make_scope(path="/api/v1/users")
        receive = AsyncMock()
        send = AsyncMock()

        self._run(mw(scope, receive, send))

        inner_app.assert_awaited_once_with(scope, receive, send)
        metrics_mock.record_api_request.assert_not_called()

    def test_websocket_scope_passes_through(self, metrics_mock):
        """Non-HTTP scopes should pass through without instrumentation."""
        inner_app = AsyncMock()
        mw = FactorsMetricsMiddleware(inner_app)

        scope = _make_ws_scope()
        receive = AsyncMock()
        send = AsyncMock()

        self._run(mw(scope, receive, send))

        inner_app.assert_awaited_once()
        metrics_mock.record_api_request.assert_not_called()

    def test_factors_path_records_metrics(self, metrics_mock):
        """Requests to /api/v1/factors/* should record request metrics."""

        async def fake_app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200})
            await send({"type": "http.response.body", "body": b""})

        mw = FactorsMetricsMiddleware(fake_app)
        scope = _make_scope(method="GET", path="/api/v1/factors/search")
        receive = AsyncMock()
        send = AsyncMock()

        self._run(mw(scope, receive, send))

        metrics_mock.record_api_request.assert_called_once()
        call_args = metrics_mock.record_api_request.call_args
        assert call_args[0][0] == "GET"  # method
        assert call_args[0][1] == "/api/v1/factors/search"  # normalized path
        assert call_args[0][2] == 200  # status code
        assert isinstance(call_args[0][3], float)  # latency

    def test_captures_status_code_from_response(self, metrics_mock):
        """Status code should be captured from the response start message."""

        async def fake_app(scope, receive, send):
            await send({"type": "http.response.start", "status": 404})
            await send({"type": "http.response.body", "body": b""})

        mw = FactorsMetricsMiddleware(fake_app)
        scope = _make_scope(path="/api/v1/factors/xyz-123")
        receive = AsyncMock()
        send = AsyncMock()

        self._run(mw(scope, receive, send))

        call_args = metrics_mock.record_api_request.call_args
        assert call_args[0][2] == 404

    def test_records_500_on_exception(self, metrics_mock):
        """If the inner app raises, status should default to 500."""

        async def failing_app(scope, receive, send):
            raise RuntimeError("boom")

        mw = FactorsMetricsMiddleware(failing_app)
        scope = _make_scope(path="/api/v1/factors/search")
        receive = AsyncMock()
        send = AsyncMock()

        with pytest.raises(RuntimeError, match="boom"):
            self._run(mw(scope, receive, send))

        call_args = metrics_mock.record_api_request.call_args
        assert call_args[0][2] == 500

    def test_factor_id_path_normalized(self, metrics_mock):
        """Factor ID paths should be normalized to {factor_id} template."""

        async def fake_app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200})
            await send({"type": "http.response.body", "body": b""})

        mw = FactorsMetricsMiddleware(fake_app)
        scope = _make_scope(path="/api/v1/factors/some-uuid-factor-id")
        receive = AsyncMock()
        send = AsyncMock()

        self._run(mw(scope, receive, send))

        call_args = metrics_mock.record_api_request.call_args
        assert call_args[0][1] == "/api/v1/factors/{factor_id}"


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


class TestConvenienceHelpers:
    """Test module-level convenience helper functions."""

    def test_record_search_results(self):
        with patch(
            "greenlang.factors.observability.prometheus.get_factors_metrics"
        ) as mock_get:
            mock_metrics = MagicMock()
            mock_get.return_value = mock_metrics
            record_search_results(42)
            mock_metrics.record_search_results.assert_called_once_with(42)

    def test_record_match_score(self):
        with patch(
            "greenlang.factors.observability.prometheus.get_factors_metrics"
        ) as mock_get:
            mock_metrics = MagicMock()
            mock_get.return_value = mock_metrics
            record_match_score(0.87)
            mock_metrics.record_match_score.assert_called_once_with(0.87)

    def test_record_qa_failure(self):
        with patch(
            "greenlang.factors.observability.prometheus.get_factors_metrics"
        ) as mock_get:
            mock_metrics = MagicMock()
            mock_get.return_value = mock_metrics
            record_qa_failure("unit_check")
            mock_metrics.record_qa_failure.assert_called_once_with("unit_check")

    def test_record_qa_failure_default_gate(self):
        with patch(
            "greenlang.factors.observability.prometheus.get_factors_metrics"
        ) as mock_get:
            mock_metrics = MagicMock()
            mock_get.return_value = mock_metrics
            record_qa_failure()
            mock_metrics.record_qa_failure.assert_called_once_with("default")

    def test_update_edition_gauge(self):
        with patch(
            "greenlang.factors.observability.prometheus.get_factors_metrics"
        ) as mock_get:
            mock_metrics = MagicMock()
            mock_get.return_value = mock_metrics
            update_edition_gauge("ed-2026-04", 102000)
            mock_metrics.set_edition_factor_count.assert_called_once_with(
                "ed-2026-04", "certified", 102000
            )

    def test_update_edition_gauge_with_status(self):
        with patch(
            "greenlang.factors.observability.prometheus.get_factors_metrics"
        ) as mock_get:
            mock_metrics = MagicMock()
            mock_get.return_value = mock_metrics
            update_edition_gauge("ed-2026-04", 500, status="preview")
            mock_metrics.set_edition_factor_count.assert_called_once_with(
                "ed-2026-04", "preview", 500
            )
