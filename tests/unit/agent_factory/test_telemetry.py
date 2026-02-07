# -*- coding: utf-8 -*-
"""
Unit tests for Agent Factory Telemetry: tracing, span creation,
correlation ID management, propagation, and metrics collection.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest


# ============================================================================
# Inline Implementations (contract definitions)
# ============================================================================


@dataclass
class Span:
    trace_id: str
    span_id: str = field(default_factory=lambda: str(uuid.uuid4())[:16])
    parent_span_id: Optional[str] = None
    operation: str = ""
    service: str = "agent-factory"
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    status: str = "ok"
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000

    def finish(self, status: str = "ok") -> None:
        self.end_time = time.time()
        self.status = status

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        self.events.append({
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {},
        })


class Tracer:
    def __init__(self, service_name: str = "agent-factory") -> None:
        self.service_name = service_name
        self._spans: List[Span] = []
        self._active_span: Optional[Span] = None

    def create_span(
        self,
        operation: str,
        parent: Optional[Span] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Span:
        trace_id = parent.trace_id if parent else str(uuid.uuid4())[:32]
        span = Span(
            trace_id=trace_id,
            operation=operation,
            service=self.service_name,
            parent_span_id=parent.span_id if parent else None,
            attributes=attributes or {},
        )
        self._spans.append(span)
        self._active_span = span
        return span

    @property
    def active_span(self) -> Optional[Span]:
        return self._active_span

    @property
    def spans(self) -> List[Span]:
        return list(self._spans)


class SpanFactory:
    def __init__(self, tracer: Tracer) -> None:
        self._tracer = tracer

    def execution_span(
        self,
        agent_key: str,
        task_id: str,
        parent: Optional[Span] = None,
    ) -> Span:
        return self._tracer.create_span(
            operation=f"execute:{agent_key}",
            parent=parent,
            attributes={"agent_key": agent_key, "task_id": task_id},
        )

    def lifecycle_span(
        self,
        agent_key: str,
        transition: str,
        parent: Optional[Span] = None,
    ) -> Span:
        return self._tracer.create_span(
            operation=f"lifecycle:{transition}",
            parent=parent,
            attributes={"agent_key": agent_key, "transition": transition},
        )


class CorrelationManager:
    """Manages correlation IDs for distributed tracing."""

    HEADER_NAME = "X-Correlation-ID"
    TRACE_HEADER = "X-Trace-ID"

    @staticmethod
    def generate_id() -> str:
        return str(uuid.uuid4())

    @staticmethod
    def inject_headers(
        headers: Dict[str, str],
        correlation_id: str,
        trace_id: Optional[str] = None,
    ) -> Dict[str, str]:
        headers[CorrelationManager.HEADER_NAME] = correlation_id
        if trace_id:
            headers[CorrelationManager.TRACE_HEADER] = trace_id
        return headers

    @staticmethod
    def extract_headers(
        headers: Dict[str, str],
    ) -> tuple:
        correlation_id = headers.get(CorrelationManager.HEADER_NAME)
        trace_id = headers.get(CorrelationManager.TRACE_HEADER)
        return correlation_id, trace_id


@dataclass
class MetricSample:
    name: str
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class MetricsCollector:
    def __init__(self) -> None:
        self._counters: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = {}
        self._samples: List[MetricSample] = []

    def record(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        key = name
        self._counters[key] = self._counters.get(key, 0.0) + value
        self._samples.append(MetricSample(name=name, value=value, labels=labels or {}))

    def histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        self._histograms.setdefault(name, []).append(value)
        self._samples.append(MetricSample(name=name, value=value, labels=labels or {}))

    def get_counter(self, name: str) -> float:
        return self._counters.get(name, 0.0)

    def get_histogram(self, name: str) -> List[float]:
        return list(self._histograms.get(name, []))

    @property
    def samples(self) -> List[MetricSample]:
        return list(self._samples)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def tracer() -> Tracer:
    return Tracer(service_name="test-factory")


@pytest.fixture
def span_factory(tracer: Tracer) -> SpanFactory:
    return SpanFactory(tracer)


@pytest.fixture
def metrics() -> MetricsCollector:
    return MetricsCollector()


# ============================================================================
# Tests
# ============================================================================


class TestTracer:
    """Tests for the distributed tracer."""

    def test_tracer_initialization(self, tracer: Tracer) -> None:
        """Tracer initializes with service name and empty spans."""
        assert tracer.service_name == "test-factory"
        assert len(tracer.spans) == 0

    def test_tracer_create_span(self, tracer: Tracer) -> None:
        """Creating a span sets operation and trace ID."""
        span = tracer.create_span("test-operation")
        assert span.operation == "test-operation"
        assert span.trace_id is not None
        assert span.service == "test-factory"
        assert len(tracer.spans) == 1

    def test_tracer_create_child_span(self, tracer: Tracer) -> None:
        """Child spans share the parent's trace ID."""
        parent = tracer.create_span("parent-op")
        child = tracer.create_span("child-op", parent=parent)
        assert child.trace_id == parent.trace_id
        assert child.parent_span_id == parent.span_id

    def test_span_finish(self, tracer: Tracer) -> None:
        """Finishing a span records end_time and computes duration."""
        span = tracer.create_span("timed-op")
        time.sleep(0.01)
        span.finish()
        assert span.end_time is not None
        assert span.duration_ms > 0

    def test_span_add_event(self, tracer: Tracer) -> None:
        """Events can be attached to spans."""
        span = tracer.create_span("with-events")
        span.add_event("cache_miss", {"key": "emission_factor"})
        assert len(span.events) == 1
        assert span.events[0]["name"] == "cache_miss"


class TestSpanFactory:
    """Tests for the span factory creating typed spans."""

    def test_span_factory_execution_span(
        self, span_factory: SpanFactory
    ) -> None:
        """Execution spans have agent_key and task_id attributes."""
        span = span_factory.execution_span("carbon-agent", "task-001")
        assert span.operation == "execute:carbon-agent"
        assert span.attributes["agent_key"] == "carbon-agent"
        assert span.attributes["task_id"] == "task-001"

    def test_span_factory_lifecycle_span(
        self, span_factory: SpanFactory
    ) -> None:
        """Lifecycle spans have agent_key and transition attributes."""
        span = span_factory.lifecycle_span("carbon-agent", "RUNNING")
        assert span.operation == "lifecycle:RUNNING"
        assert span.attributes["transition"] == "RUNNING"


class TestCorrelationManager:
    """Tests for correlation ID management and propagation."""

    def test_correlation_manager_generate_id(self) -> None:
        """Generated IDs are valid UUIDs."""
        cid = CorrelationManager.generate_id()
        assert len(cid) == 36  # UUID format

    def test_correlation_manager_propagation(self) -> None:
        """Same correlation ID can be propagated across calls."""
        cid = CorrelationManager.generate_id()
        headers: Dict[str, str] = {}
        CorrelationManager.inject_headers(headers, cid)
        extracted_cid, _ = CorrelationManager.extract_headers(headers)
        assert extracted_cid == cid

    def test_correlation_inject_headers(self) -> None:
        """Correlation and trace IDs are injected into headers."""
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        result = CorrelationManager.inject_headers(
            headers, "corr-123", "trace-456"
        )
        assert result["X-Correlation-ID"] == "corr-123"
        assert result["X-Trace-ID"] == "trace-456"

    def test_correlation_extract_headers(self) -> None:
        """Correlation and trace IDs are extracted from headers."""
        headers = {
            "X-Correlation-ID": "corr-abc",
            "X-Trace-ID": "trace-def",
        }
        cid, tid = CorrelationManager.extract_headers(headers)
        assert cid == "corr-abc"
        assert tid == "trace-def"

    def test_correlation_extract_missing(self) -> None:
        """Missing headers return None."""
        cid, tid = CorrelationManager.extract_headers({})
        assert cid is None
        assert tid is None


class TestMetricsCollector:
    """Tests for the metrics collection system."""

    def test_metrics_collector_record(self, metrics: MetricsCollector) -> None:
        """Recording increments the counter."""
        metrics.record("agent.executions", 1)
        metrics.record("agent.executions", 1)
        assert metrics.get_counter("agent.executions") == 2.0

    def test_metrics_collector_histogram(
        self, metrics: MetricsCollector
    ) -> None:
        """Histogram records individual observations."""
        metrics.histogram("agent.latency_ms", 10.5)
        metrics.histogram("agent.latency_ms", 15.2)
        metrics.histogram("agent.latency_ms", 8.7)
        values = metrics.get_histogram("agent.latency_ms")
        assert len(values) == 3
        assert min(values) == pytest.approx(8.7)
        assert max(values) == pytest.approx(15.2)

    def test_metrics_collector_samples(
        self, metrics: MetricsCollector
    ) -> None:
        """All samples are stored with timestamps."""
        metrics.record("a", 1, labels={"env": "test"})
        assert len(metrics.samples) == 1
        assert metrics.samples[0].labels["env"] == "test"
