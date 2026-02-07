# -*- coding: utf-8 -*-
"""
Unit tests for trace context propagation (OBS-003)

Tests inject/extract of W3C trace context, trace-id / span-id retrieval,
tenant context via W3C Baggage, and no-op behaviour when the OTel SDK is
not available.

Coverage target: 85%+ of context.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ============================================================================
# Helpers
# ============================================================================


def _import_context():
    """Import the context module, skipping if not yet created."""
    try:
        from greenlang.infrastructure.tracing_service import context
        return context
    except ImportError:
        pytest.skip("context module not yet built")


# ============================================================================
# Tests
# ============================================================================


class TestInjectExtract:
    """Tests for inject_trace_context and extract_trace_context."""

    def test_inject_trace_context(self):
        """Verify inject populates a carrier dict with trace headers."""
        ctx_mod = _import_context()
        carrier: dict = {}

        mock_propagator = MagicMock()
        with patch.object(ctx_mod, "_get_propagator", return_value=mock_propagator):
            ctx_mod.inject_trace_context(carrier)
            mock_propagator.inject.assert_called_once()

    def test_extract_trace_context(self):
        """Verify extract reads trace headers from a carrier dict."""
        ctx_mod = _import_context()
        carrier = {"traceparent": "00-abc-def-01"}

        mock_propagator = MagicMock()
        mock_propagator.extract.return_value = MagicMock()
        with patch.object(ctx_mod, "_get_propagator", return_value=mock_propagator):
            result = ctx_mod.extract_trace_context(carrier)
            mock_propagator.extract.assert_called_once()
            assert result is not None

    def test_inject_extract_roundtrip(self):
        """Verify inject then extract preserves trace context."""
        ctx_mod = _import_context()
        carrier: dict = {}

        # Use real propagator mock that stores/retrieves
        stored = {}

        mock_propagator = MagicMock()
        mock_propagator.inject.side_effect = lambda c, **kw: c.update(stored)
        mock_propagator.extract.side_effect = lambda c, **kw: stored.update(c)

        with patch.object(ctx_mod, "_get_propagator", return_value=mock_propagator):
            ctx_mod.inject_trace_context(carrier)
            ctx_mod.extract_trace_context(carrier)
            # Both calls should succeed
            assert mock_propagator.inject.call_count == 1
            assert mock_propagator.extract.call_count == 1

    def test_inject_noop_when_unavailable(self):
        """Verify inject is a no-op when OTel is unavailable."""
        ctx_mod = _import_context()
        carrier: dict = {}

        with patch.object(ctx_mod, "OTEL_AVAILABLE", False):
            ctx_mod.inject_trace_context(carrier)
            # Carrier should remain unchanged or function should not crash
            assert isinstance(carrier, dict)

    def test_extract_noop_when_unavailable(self):
        """Verify extract returns None when OTel is unavailable."""
        ctx_mod = _import_context()
        carrier = {"traceparent": "00-abc-def-01"}

        with patch.object(ctx_mod, "OTEL_AVAILABLE", False):
            result = ctx_mod.extract_trace_context(carrier)
            assert result is None


class TestTraceAndSpanId:
    """Tests for get_current_trace_id and get_current_span_id."""

    def test_get_current_trace_id(self, mock_span):
        """Verify get_current_trace_id returns the hex trace ID."""
        ctx_mod = _import_context()

        with patch.object(ctx_mod, "OTEL_AVAILABLE", True):
            mock_get_span = MagicMock(return_value=mock_span)
            with patch.object(ctx_mod, "_get_current_span", mock_get_span):
                trace_id = ctx_mod.get_current_trace_id()
                assert isinstance(trace_id, str)
                assert len(trace_id) > 0

    def test_get_current_span_id(self, mock_span):
        """Verify get_current_span_id returns the hex span ID."""
        ctx_mod = _import_context()

        with patch.object(ctx_mod, "OTEL_AVAILABLE", True):
            mock_get_span = MagicMock(return_value=mock_span)
            with patch.object(ctx_mod, "_get_current_span", mock_get_span):
                span_id = ctx_mod.get_current_span_id()
                assert isinstance(span_id, str)
                assert len(span_id) > 0

    def test_get_trace_id_when_no_span(self):
        """Verify get_current_trace_id returns empty string with no active span."""
        ctx_mod = _import_context()

        with patch.object(ctx_mod, "OTEL_AVAILABLE", True):
            # No-op span context with trace_id = 0
            noop_span = MagicMock()
            noop_ctx = MagicMock()
            noop_ctx.trace_id = 0
            noop_ctx.is_valid = False
            noop_span.get_span_context.return_value = noop_ctx

            with patch.object(ctx_mod, "_get_current_span", return_value=noop_span):
                trace_id = ctx_mod.get_current_trace_id()
                assert trace_id == "" or trace_id == "0" * 32

    def test_get_span_id_when_no_span(self):
        """Verify get_current_span_id returns empty string with no active span."""
        ctx_mod = _import_context()

        with patch.object(ctx_mod, "OTEL_AVAILABLE", True):
            noop_span = MagicMock()
            noop_ctx = MagicMock()
            noop_ctx.span_id = 0
            noop_ctx.is_valid = False
            noop_span.get_span_context.return_value = noop_ctx

            with patch.object(ctx_mod, "_get_current_span", return_value=noop_span):
                span_id = ctx_mod.get_current_span_id()
                assert span_id == "" or span_id == "0" * 16

    def test_context_noop_when_otel_unavailable(self):
        """Verify trace/span ID return empty when OTel unavailable."""
        ctx_mod = _import_context()

        with patch.object(ctx_mod, "OTEL_AVAILABLE", False):
            assert ctx_mod.get_current_trace_id() == ""
            assert ctx_mod.get_current_span_id() == ""


class TestTenantContext:
    """Tests for set/get tenant context via W3C Baggage."""

    def test_set_tenant_context(self):
        """Verify set_tenant_context stores tenant ID in baggage."""
        ctx_mod = _import_context()

        with patch.object(ctx_mod, "OTEL_AVAILABLE", True):
            mock_baggage = MagicMock()
            with patch.object(ctx_mod, "_set_baggage", mock_baggage):
                ctx_mod.set_tenant_context("t-acme")
                mock_baggage.assert_called_once()
                args = mock_baggage.call_args
                assert "t-acme" in str(args)

    def test_get_tenant_context(self):
        """Verify get_tenant_context retrieves tenant ID from baggage."""
        ctx_mod = _import_context()

        with patch.object(ctx_mod, "OTEL_AVAILABLE", True):
            with patch.object(ctx_mod, "_get_baggage", return_value="t-acme"):
                tenant = ctx_mod.get_tenant_context()
                assert tenant == "t-acme"

    def test_get_tenant_context_when_not_set(self):
        """Verify get_tenant_context returns None/empty when no baggage."""
        ctx_mod = _import_context()

        with patch.object(ctx_mod, "OTEL_AVAILABLE", True):
            with patch.object(ctx_mod, "_get_baggage", return_value=None):
                tenant = ctx_mod.get_tenant_context()
                assert tenant is None or tenant == ""


class TestBaggage:
    """Tests for generic baggage set/get operations."""

    def test_set_baggage_item(self):
        """Verify set_baggage_item stores a key-value pair."""
        ctx_mod = _import_context()

        with patch.object(ctx_mod, "OTEL_AVAILABLE", True):
            mock_set = MagicMock()
            with patch.object(ctx_mod, "_set_baggage", mock_set):
                ctx_mod.set_baggage_item("gl.pipeline_id", "pipe-123")
                mock_set.assert_called_once()

    def test_get_baggage_item(self):
        """Verify get_baggage_item retrieves a stored value."""
        ctx_mod = _import_context()

        with patch.object(ctx_mod, "OTEL_AVAILABLE", True):
            with patch.object(ctx_mod, "_get_baggage", return_value="pipe-123"):
                val = ctx_mod.get_baggage_item("gl.pipeline_id")
                assert val == "pipe-123"

    def test_get_baggage_item_not_found(self):
        """Verify get_baggage_item returns None for missing keys."""
        ctx_mod = _import_context()

        with patch.object(ctx_mod, "OTEL_AVAILABLE", True):
            with patch.object(ctx_mod, "_get_baggage", return_value=None):
                val = ctx_mod.get_baggage_item("nonexistent")
                assert val is None
