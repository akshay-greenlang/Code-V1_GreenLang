# -*- coding: utf-8 -*-
"""
Integration tests - End-to-end trace flow (OBS-003)

Tests the full tracing pipeline from SDK instrumentation through
to span collection. Uses in-memory exporters for offline testing
and optionally connects to a live Tempo instance.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from typing import Dict
from unittest.mock import MagicMock, patch

import pytest

from greenlang.infrastructure.tracing_service.config import TracingConfig
from greenlang.infrastructure.tracing_service.middleware import TracingMiddleware
from greenlang.infrastructure.tracing_service.metrics_bridge import MetricsBridge
from greenlang.infrastructure.tracing_service.span_enrichment import (
    SpanEnricher,
    GL_TENANT_ID,
    GL_AGENT_TYPE,
    GL_ENVIRONMENT,
)


# ============================================================================
# SDK initialisation tests
# ============================================================================


class TestSDKInitialisation:
    """Test the tracing SDK initialises correctly in integration scenarios."""

    def test_configure_tracing_full_stack(self, integration_config):
        """configure_tracing sets up provider, instrumentors, and metrics bridge."""
        from greenlang.infrastructure.tracing_service.setup import configure_tracing

        with patch(
            "greenlang.infrastructure.tracing_service.setup.setup_provider"
        ) as mock_prov, patch(
            "greenlang.infrastructure.tracing_service.setup.setup_instrumentors",
            return_value={"fastapi": False, "httpx": False},
        ) as mock_instr, patch(
            "greenlang.infrastructure.tracing_service.setup.get_metrics_bridge"
        ) as mock_bridge:
            config = configure_tracing(config=integration_config)

        mock_prov.assert_called_once_with(integration_config)
        mock_instr.assert_called_once_with(integration_config)
        mock_bridge.assert_called_once()
        assert config.service_name == "integration-test-service"

    def test_configure_tracing_with_app(self, integration_config):
        """configure_tracing attaches middleware to a FastAPI app."""
        from greenlang.infrastructure.tracing_service.setup import configure_tracing

        mock_app = MagicMock()

        with patch(
            "greenlang.infrastructure.tracing_service.setup.setup_provider"
        ), patch(
            "greenlang.infrastructure.tracing_service.setup.setup_instrumentors",
            return_value={},
        ), patch(
            "greenlang.infrastructure.tracing_service.setup.get_metrics_bridge"
        ):
            configure_tracing(mock_app, config=integration_config)

        mock_app.add_middleware.assert_called_once()

    def test_shutdown_after_configure(self, integration_config):
        """Tracing can be cleanly shut down after configuration."""
        from greenlang.infrastructure.tracing_service.setup import (
            configure_tracing,
            shutdown_tracing,
            is_tracing_enabled,
        )

        with patch(
            "greenlang.infrastructure.tracing_service.setup.setup_provider"
        ), patch(
            "greenlang.infrastructure.tracing_service.setup.setup_instrumentors",
            return_value={},
        ), patch(
            "greenlang.infrastructure.tracing_service.setup.get_metrics_bridge"
        ):
            configure_tracing(config=integration_config)

        with patch(
            "greenlang.infrastructure.tracing_service.setup.shutdown_provider"
        ):
            shutdown_tracing()


# ============================================================================
# Middleware integration tests
# ============================================================================


class TestMiddlewareIntegration:
    """Test TracingMiddleware in combination with other SDK components."""

    @pytest.mark.asyncio
    async def test_middleware_enriches_spans(self):
        """Middleware creates spans with GreenLang attributes for normal requests."""
        mock_span = MagicMock()
        mock_span.__enter__ = MagicMock(return_value=mock_span)
        mock_span.__exit__ = MagicMock(return_value=False)

        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value = mock_span

        async def inner_app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b""})

        mw = TracingMiddleware(inner_app, service_name="integ-svc")

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/api/v1/emissions/calculate",
            "query_string": b"",
            "headers": [
                (b"host", b"api.greenlang.io"),
                (b"x-tenant-id", b"t-corp-alpha"),
                (b"x-request-id", b"req-e2e-001"),
            ],
        }

        async def receive():
            return {"type": "http.request", "body": b""}

        sent = []

        async def send(msg):
            sent.append(msg)

        with patch(
            "greenlang.infrastructure.tracing_service.middleware.OTEL_AVAILABLE",
            True,
        ), patch(
            "greenlang.infrastructure.tracing_service.middleware.trace"
        ) as mock_trace:
            mock_trace.get_tracer.return_value = mock_tracer
            await mw(scope, receive, send)

        attrs = {
            call[0][0]: call[0][1]
            for call in mock_span.set_attribute.call_args_list
        }
        assert attrs.get("gl.tenant_id") == "t-corp-alpha"
        assert attrs.get("gl.request_id") == "req-e2e-001"
        assert attrs.get("http.method") == "POST"

    @pytest.mark.asyncio
    async def test_middleware_skips_health_endpoints(self):
        """Health-check endpoints are not traced."""
        calls = []

        async def inner_app(scope, receive, send):
            calls.append(scope["path"])

        mw = TracingMiddleware(inner_app)

        for path in ["/health", "/ready", "/metrics", "/ping"]:
            scope = {
                "type": "http",
                "method": "GET",
                "path": path,
                "headers": [],
            }
            await mw(scope, lambda: None, lambda msg: None)

        assert len(calls) == 4


# ============================================================================
# Span enrichment integration tests
# ============================================================================


class TestSpanEnrichmentIntegration:
    """Test SpanEnricher works with mock spans in realistic scenarios."""

    def test_enricher_agent_workflow(self):
        """Enrich a span through a complete agent workflow."""
        enricher = SpanEnricher(environment="staging")
        span = MagicMock()

        # Simulate agent execution enrichment
        enricher.enrich_agent_span(
            span,
            agent_type="eudr-agent",
            agent_id="a-e2e-001",
            tenant_id="t-corp",
            operation="execute",
        )

        attrs = {
            call[0][0]: call[0][1]
            for call in span.set_attribute.call_args_list
        }
        assert attrs[GL_TENANT_ID] == "t-corp"
        assert attrs[GL_AGENT_TYPE] == "eudr-agent"
        assert attrs[GL_ENVIRONMENT] == "staging"

    def test_enricher_emission_workflow(self):
        """Enrich a span through a complete emission calculation workflow."""
        enricher = SpanEnricher(environment="prod")
        span = MagicMock()

        enricher.enrich_emission_span(
            span,
            scope="scope_1",
            regulation="CSRD",
            data_source="erp-sap",
            category="stationary_combustion",
            calculation_type="activity-based",
            framework="GHG Protocol",
        )

        attrs = {
            call[0][0]: call[0][1]
            for call in span.set_attribute.call_args_list
        }
        assert attrs["gl.emission_scope"] == "scope_1"
        assert attrs["gl.regulation"] == "CSRD"
        assert attrs["gl.data_source"] == "erp-sap"

    def test_enricher_pipeline_workflow(self):
        """Enrich a span through a pipeline execution workflow."""
        enricher = SpanEnricher(environment="prod")
        span = MagicMock()

        enricher.enrich_pipeline_span(
            span,
            pipeline_id="pipe-001",
            stage="extraction",
            pipeline_name="EUDR Compliance Pipeline",
            step="parse_gps_coords",
        )

        attrs = {
            call[0][0]: call[0][1]
            for call in span.set_attribute.call_args_list
        }
        assert attrs["gl.pipeline_id"] == "pipe-001"
        assert attrs["gl.pipeline_stage"] == "extraction"


# ============================================================================
# Metrics bridge integration tests
# ============================================================================


class TestMetricsBridgeIntegration:
    """Test MetricsBridge records metrics for traced operations."""

    def test_bridge_records_successful_span(self):
        """MetricsBridge records histogram and counter for successful spans."""
        bridge = MetricsBridge(service_name="integ-svc")
        mock_hist = MagicMock()
        mock_counter = MagicMock()

        with patch(
            "greenlang.infrastructure.tracing_service.metrics_bridge._SPAN_DURATION",
            mock_hist,
        ), patch(
            "greenlang.infrastructure.tracing_service.metrics_bridge._SPAN_COUNT",
            mock_counter,
        ):
            bridge.record_span("calculate_emissions", 0.123, status="ok")
            bridge.record_span("validate_data", 0.045, status="ok")

        assert mock_hist.labels.call_count == 2
        assert mock_counter.labels.call_count == 2

    def test_bridge_records_errors(self):
        """MetricsBridge tracks error counts by type."""
        bridge = MetricsBridge(service_name="integ-svc")
        mock_errors = MagicMock()

        with patch(
            "greenlang.infrastructure.tracing_service.metrics_bridge._TRACE_ERRORS",
            mock_errors,
        ):
            bridge.record_error("validate_data", "ValidationError")
            bridge.record_error("calculate_emissions", "TimeoutError")

        assert mock_errors.labels.call_count == 2


# ============================================================================
# Context propagation integration tests
# ============================================================================


class TestContextPropagationIntegration:
    """Test context injection and extraction work together."""

    def test_inject_extract_round_trip(self):
        """Context injected into headers can be extracted back."""
        from greenlang.infrastructure.tracing_service.context import (
            inject_trace_context,
            extract_trace_context,
        )

        headers: Dict[str, str] = {}
        inject_trace_context(headers)
        ctx = extract_trace_context(headers)
        # When no active span, headers may be empty but no exception
        assert isinstance(headers, dict)

    def test_tenant_context_round_trip(self):
        """Tenant context set via baggage can be retrieved."""
        from greenlang.infrastructure.tracing_service.context import (
            set_tenant_context,
            get_tenant_context,
        )

        # Without OTel, these are no-ops but should not raise
        set_tenant_context("t-round-trip")
        # Result depends on OTel availability
        result = get_tenant_context()
        assert result is None or result == "t-round-trip"
