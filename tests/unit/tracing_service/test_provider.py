# -*- coding: utf-8 -*-
"""
Unit tests for TracerProvider setup and lifecycle (OBS-003)

Tests provider initialisation, resource creation, exporter attachment,
no-op fallback, idempotency, shutdown, and force-flush.

Coverage target: 85%+ of provider.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from unittest.mock import MagicMock, patch, call

import pytest

from greenlang.infrastructure.tracing_service.config import TracingConfig
from greenlang.infrastructure.tracing_service.provider import (
    _NoOpSpan,
    _NoOpSpanContext,
    _NoOpTracer,
    get_tracer,
    get_tracer_provider,
    get_current_span,
    setup_provider,
    shutdown,
    force_flush,
    OTEL_AVAILABLE,
)
import greenlang.infrastructure.tracing_service.provider as provider_mod


# ============================================================================
# No-op class tests
# ============================================================================


class TestNoOpSpanContext:
    """Tests for the _NoOpSpanContext fallback."""

    def test_trace_id_is_zero(self):
        """Verify no-op span context reports trace_id = 0."""
        ctx = _NoOpSpanContext()
        assert ctx.trace_id == 0

    def test_span_id_is_zero(self):
        """Verify no-op span context reports span_id = 0."""
        ctx = _NoOpSpanContext()
        assert ctx.span_id == 0

    def test_is_valid_false(self):
        """Verify no-op span context is not valid."""
        ctx = _NoOpSpanContext()
        assert ctx.is_valid is False

    def test_is_remote_false(self):
        """Verify no-op span context is not remote."""
        ctx = _NoOpSpanContext()
        assert ctx.is_remote is False


class TestNoOpSpan:
    """Tests for the _NoOpSpan fallback."""

    def test_set_attribute_noop(self):
        """Verify set_attribute does not raise."""
        span = _NoOpSpan()
        span.set_attribute("key", "value")  # should not raise

    def test_set_status_noop(self):
        """Verify set_status does not raise."""
        span = _NoOpSpan()
        span.set_status("OK")  # should not raise

    def test_add_event_noop(self):
        """Verify add_event does not raise."""
        span = _NoOpSpan()
        span.add_event("event_name", {"attr": 1})

    def test_record_exception_noop(self):
        """Verify record_exception does not raise."""
        span = _NoOpSpan()
        span.record_exception(ValueError("test"))

    def test_end_noop(self):
        """Verify end() does not raise."""
        span = _NoOpSpan()
        span.end()

    def test_is_recording_false(self):
        """Verify no-op span reports is_recording = False."""
        span = _NoOpSpan()
        assert span.is_recording() is False

    def test_get_span_context_returns_noop(self):
        """Verify get_span_context returns _NoOpSpanContext."""
        span = _NoOpSpan()
        ctx = span.get_span_context()
        assert isinstance(ctx, _NoOpSpanContext)

    def test_context_manager_enter_exit(self):
        """Verify _NoOpSpan works as a context manager."""
        span = _NoOpSpan()
        with span as s:
            assert s is span


class TestNoOpTracer:
    """Tests for the _NoOpTracer fallback."""

    def test_start_span_returns_noop_span(self):
        """Verify start_span returns a _NoOpSpan."""
        tracer = _NoOpTracer()
        span = tracer.start_span("test")
        assert isinstance(span, _NoOpSpan)

    def test_start_as_current_span_context_manager(self):
        """Verify start_as_current_span yields a _NoOpSpan."""
        tracer = _NoOpTracer()
        with tracer.start_as_current_span("test") as span:
            assert isinstance(span, _NoOpSpan)

    def test_start_span_with_all_args(self):
        """Verify start_span accepts all optional arguments."""
        tracer = _NoOpTracer()
        span = tracer.start_span(
            "op",
            context=None,
            kind=None,
            attributes={"k": "v"},
            links=None,
            start_time=None,
            record_exception=False,
            set_status_on_exception=False,
        )
        assert isinstance(span, _NoOpSpan)

    def test_start_as_current_span_with_all_args(self):
        """Verify start_as_current_span accepts all optional arguments."""
        tracer = _NoOpTracer()
        with tracer.start_as_current_span(
            "op",
            context=None,
            kind=None,
            attributes={"k": "v"},
            links=None,
            start_time=None,
            record_exception=False,
            set_status_on_exception=False,
            end_on_exit=False,
        ) as span:
            assert isinstance(span, _NoOpSpan)


# ============================================================================
# setup_provider tests
# ============================================================================


class TestSetupProvider:
    """Tests for setup_provider() function."""

    def test_setup_provider_skips_when_otel_unavailable(self, tracing_config):
        """Verify setup is a no-op when OTel SDK is not installed."""
        with patch.object(provider_mod, "OTEL_AVAILABLE", False):
            # Reset state so it can run
            with provider_mod._lock:
                provider_mod._initialized = False
                provider_mod._provider = None

            setup_provider(tracing_config)

            assert provider_mod._initialized is True
            assert provider_mod._provider is None

    def test_setup_provider_skips_when_disabled(self, disabled_config):
        """Verify setup is a no-op when config.enabled is False."""
        if not OTEL_AVAILABLE:
            pytest.skip("OTel SDK not installed")

        setup_provider(disabled_config)

        assert provider_mod._initialized is True
        assert provider_mod._provider is None

    def test_setup_provider_idempotent(self, tracing_config):
        """Verify subsequent calls to setup_provider are silently ignored."""
        with patch.object(provider_mod, "OTEL_AVAILABLE", False):
            with provider_mod._lock:
                provider_mod._initialized = False
                provider_mod._provider = None

            setup_provider(tracing_config)
            first_config = provider_mod._config

            # Second call should not change config
            other_config = TracingConfig(service_name="other")
            setup_provider(other_config)
            assert provider_mod._config is first_config

    @pytest.mark.skipif(not OTEL_AVAILABLE, reason="OTel SDK not installed")
    def test_setup_provider_creates_tracer_provider(self, tracing_config):
        """Verify setup creates a real TracerProvider when OTel is available."""
        setup_provider(tracing_config)
        assert provider_mod._provider is not None
        assert provider_mod._initialized is True

    @pytest.mark.skipif(not OTEL_AVAILABLE, reason="OTel SDK not installed")
    def test_setup_provider_sets_resource_attributes(self, tracing_config):
        """Verify the provider resource includes required attributes."""
        setup_provider(tracing_config)
        resource = provider_mod._provider.resource
        attrs = dict(resource.attributes)

        assert attrs.get("service.name") == "test-service"
        assert attrs.get("service.version") == "1.0.0"
        assert attrs.get("deployment.environment") == "test"
        assert attrs.get("service.namespace") == "greenlang"
        assert attrs.get("service.platform") == "greenlang-climate-os"

    @pytest.mark.skipif(not OTEL_AVAILABLE, reason="OTel SDK not installed")
    def test_setup_provider_adds_console_exporter_when_enabled(self):
        """Verify console exporter is added when console_exporter=True."""
        config = TracingConfig(
            service_name="console-test",
            console_exporter=True,
            enabled=True,
        )
        setup_provider(config)

        # The provider should have at least 2 span processors (OTLP + console)
        processors = provider_mod._provider._active_span_processor._span_processors
        assert len(processors) >= 2

    @pytest.mark.skipif(not OTEL_AVAILABLE, reason="OTel SDK not installed")
    def test_resource_includes_k8s_attributes(self, tracing_config, monkeypatch):
        """Verify K8s attributes are included when env vars are present."""
        monkeypatch.setenv("HOSTNAME", "pod-abc-123")
        monkeypatch.setenv("K8S_NAMESPACE", "greenlang-prod")
        monkeypatch.setenv("K8S_NODE_NAME", "node-01")

        setup_provider(tracing_config)
        resource = provider_mod._provider.resource
        attrs = dict(resource.attributes)

        assert attrs.get("k8s.pod.name") == "pod-abc-123"
        assert attrs.get("k8s.namespace.name") == "greenlang-prod"
        assert attrs.get("k8s.node.name") == "node-01"


# ============================================================================
# get_tracer / get_tracer_provider / get_current_span tests
# ============================================================================


class TestGetTracer:
    """Tests for get_tracer() function."""

    def test_get_tracer_returns_noop_when_not_initialized(self):
        """Verify get_tracer returns _NoOpTracer before setup."""
        tracer = get_tracer("test_module")
        assert isinstance(tracer, _NoOpTracer)

    def test_get_tracer_returns_noop_when_otel_unavailable(self, tracing_config):
        """Verify get_tracer returns _NoOpTracer when OTel is absent."""
        with patch.object(provider_mod, "OTEL_AVAILABLE", False):
            with provider_mod._lock:
                provider_mod._initialized = True
                provider_mod._provider = MagicMock()

            tracer = get_tracer("test_module")
            assert isinstance(tracer, _NoOpTracer)

    @pytest.mark.skipif(not OTEL_AVAILABLE, reason="OTel SDK not installed")
    def test_get_tracer_returns_real_tracer_after_setup(self, tracing_config):
        """Verify get_tracer returns a real tracer after setup."""
        setup_provider(tracing_config)
        tracer = get_tracer("test_module")
        assert not isinstance(tracer, _NoOpTracer)


class TestGetTracerProvider:
    """Tests for get_tracer_provider() function."""

    def test_returns_none_before_setup(self):
        """Verify get_tracer_provider returns None before initialisation."""
        assert get_tracer_provider() is None

    @pytest.mark.skipif(not OTEL_AVAILABLE, reason="OTel SDK not installed")
    def test_returns_provider_after_setup(self, tracing_config):
        """Verify get_tracer_provider returns the provider after setup."""
        setup_provider(tracing_config)
        assert get_tracer_provider() is not None


class TestGetCurrentSpan:
    """Tests for get_current_span() function."""

    def test_returns_noop_span_when_otel_unavailable(self):
        """Verify get_current_span returns _NoOpSpan when OTel absent."""
        with patch.object(provider_mod, "OTEL_AVAILABLE", False):
            span = get_current_span()
            assert isinstance(span, _NoOpSpan)


# ============================================================================
# Shutdown / force_flush tests
# ============================================================================


class TestShutdown:
    """Tests for shutdown() function."""

    def test_shutdown_noop_when_not_initialized(self):
        """Verify shutdown does not raise when never initialised."""
        shutdown()  # should not raise

    def test_shutdown_clears_state(self):
        """Verify shutdown resets module state."""
        # Simulate an initialised state
        with provider_mod._lock:
            provider_mod._provider = MagicMock()
            provider_mod._initialized = True

        shutdown()

        assert provider_mod._provider is None
        assert provider_mod._initialized is False

    def test_shutdown_calls_provider_shutdown(self):
        """Verify shutdown calls the provider's shutdown method."""
        mock_provider = MagicMock()
        with provider_mod._lock:
            provider_mod._provider = mock_provider
            provider_mod._initialized = True

        shutdown()

        mock_provider.shutdown.assert_called_once()

    def test_shutdown_handles_exception_gracefully(self):
        """Verify shutdown handles provider.shutdown() exceptions."""
        mock_provider = MagicMock()
        mock_provider.shutdown.side_effect = RuntimeError("shutdown error")
        with provider_mod._lock:
            provider_mod._provider = mock_provider
            provider_mod._initialized = True

        shutdown()  # should not raise

        assert provider_mod._provider is None
        assert provider_mod._initialized is False


class TestForceFlush:
    """Tests for force_flush() function."""

    def test_force_flush_returns_false_when_not_initialized(self):
        """Verify force_flush returns False when no provider exists."""
        assert force_flush() is False

    def test_force_flush_delegates_to_provider(self):
        """Verify force_flush calls provider.force_flush()."""
        mock_provider = MagicMock()
        mock_provider.force_flush.return_value = True
        with provider_mod._lock:
            provider_mod._provider = mock_provider
            provider_mod._initialized = True

        result = force_flush(timeout_millis=5000)

        mock_provider.force_flush.assert_called_once_with(5000)
        assert result is True

    def test_force_flush_handles_exception(self):
        """Verify force_flush returns False on provider error."""
        mock_provider = MagicMock()
        mock_provider.force_flush.side_effect = RuntimeError("flush error")
        with provider_mod._lock:
            provider_mod._provider = mock_provider
            provider_mod._initialized = True

        result = force_flush()
        assert result is False
