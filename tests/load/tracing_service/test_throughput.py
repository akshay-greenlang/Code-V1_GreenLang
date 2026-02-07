# -*- coding: utf-8 -*-
"""
Load tests - Tracing SDK throughput (OBS-003)

Tests that the tracing SDK can sustain high span creation rates
(target: 10,000 spans/sec) with acceptable overhead (< 5ms P99).

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import statistics
import time
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from greenlang.infrastructure.tracing_service.config import TracingConfig
from greenlang.infrastructure.tracing_service.provider import (
    _NoOpTracer,
    _NoOpSpan,
)
from greenlang.infrastructure.tracing_service.span_enrichment import SpanEnricher
from greenlang.infrastructure.tracing_service.metrics_bridge import MetricsBridge


# ============================================================================
# Constants
# ============================================================================

TARGET_SPANS_PER_SEC = 10_000
TARGET_P99_OVERHEAD_MS = 5.0
BENCHMARK_DURATION_SEC = 2.0


# ============================================================================
# NoOp tracer throughput tests
# ============================================================================


class TestNoOpTracerThroughput:
    """Test that the no-op fallback tracer adds negligible overhead."""

    def test_noop_span_creation_rate(self):
        """NoOpTracer can create > 10K spans/sec."""
        tracer = _NoOpTracer()
        count = 0
        start = time.monotonic()

        while (time.monotonic() - start) < BENCHMARK_DURATION_SEC:
            with tracer.start_as_current_span("benchmark-op") as span:
                span.set_attribute("key", "value")
                count += 1

        elapsed = time.monotonic() - start
        rate = count / elapsed
        assert rate > TARGET_SPANS_PER_SEC, (
            f"NoOp rate {rate:.0f} spans/sec below target {TARGET_SPANS_PER_SEC}"
        )

    def test_noop_span_overhead(self):
        """NoOpTracer overhead is < 1ms per span."""
        tracer = _NoOpTracer()
        durations: List[float] = []

        for _ in range(10_000):
            start = time.perf_counter()
            with tracer.start_as_current_span("bench") as span:
                span.set_attribute("k", "v")
            durations.append((time.perf_counter() - start) * 1000)

        p99 = sorted(durations)[int(len(durations) * 0.99)]
        assert p99 < 1.0, f"NoOp P99 overhead {p99:.3f}ms exceeds 1ms"


# ============================================================================
# Span enrichment throughput tests
# ============================================================================


class TestSpanEnrichmentThroughput:
    """Test SpanEnricher can sustain high enrichment rates."""

    def test_agent_enrichment_rate(self):
        """SpanEnricher.enrich_agent_span sustains > 10K/sec."""
        enricher = SpanEnricher(environment="load-test")
        span = MagicMock()
        count = 0
        start = time.monotonic()

        while (time.monotonic() - start) < BENCHMARK_DURATION_SEC:
            enricher.enrich_agent_span(
                span,
                agent_type="carbon-calc",
                agent_id=f"a-{count}",
                tenant_id="t-corp",
                operation="execute",
            )
            count += 1

        elapsed = time.monotonic() - start
        rate = count / elapsed
        assert rate > TARGET_SPANS_PER_SEC, (
            f"Enrichment rate {rate:.0f}/sec below target {TARGET_SPANS_PER_SEC}"
        )

    def test_emission_enrichment_rate(self):
        """SpanEnricher.enrich_emission_span sustains > 10K/sec."""
        enricher = SpanEnricher(environment="load-test")
        span = MagicMock()
        count = 0
        start = time.monotonic()

        while (time.monotonic() - start) < BENCHMARK_DURATION_SEC:
            enricher.enrich_emission_span(
                span,
                scope="scope_1",
                regulation="CSRD",
                data_source="erp",
                category="combustion",
            )
            count += 1

        elapsed = time.monotonic() - start
        rate = count / elapsed
        assert rate > TARGET_SPANS_PER_SEC


# ============================================================================
# MetricsBridge throughput tests
# ============================================================================


class TestMetricsBridgeThroughput:
    """Test MetricsBridge can sustain high recording rates."""

    def test_record_span_rate(self):
        """MetricsBridge.record_span sustains > 10K/sec with Prometheus mocked."""
        bridge = MetricsBridge(service_name="load-test")
        mock_hist = MagicMock()
        mock_counter = MagicMock()
        count = 0
        start = time.monotonic()

        with patch(
            "greenlang.infrastructure.tracing_service.metrics_bridge._SPAN_DURATION",
            mock_hist,
        ), patch(
            "greenlang.infrastructure.tracing_service.metrics_bridge._SPAN_COUNT",
            mock_counter,
        ):
            while (time.monotonic() - start) < BENCHMARK_DURATION_SEC:
                bridge.record_span("op", 0.001 * count, status="ok")
                count += 1

        elapsed = time.monotonic() - start
        rate = count / elapsed
        assert rate > TARGET_SPANS_PER_SEC, (
            f"MetricsBridge rate {rate:.0f}/sec below target"
        )

    def test_record_span_overhead_p99(self):
        """MetricsBridge.record_span P99 overhead < 5ms."""
        bridge = MetricsBridge(service_name="load-test")
        durations: List[float] = []

        with patch(
            "greenlang.infrastructure.tracing_service.metrics_bridge._SPAN_DURATION",
            MagicMock(),
        ), patch(
            "greenlang.infrastructure.tracing_service.metrics_bridge._SPAN_COUNT",
            MagicMock(),
        ):
            for i in range(10_000):
                start = time.perf_counter()
                bridge.record_span("op", 0.001 * i, status="ok")
                durations.append((time.perf_counter() - start) * 1000)

        p99 = sorted(durations)[int(len(durations) * 0.99)]
        assert p99 < TARGET_P99_OVERHEAD_MS, (
            f"MetricsBridge P99 overhead {p99:.3f}ms exceeds {TARGET_P99_OVERHEAD_MS}ms"
        )
