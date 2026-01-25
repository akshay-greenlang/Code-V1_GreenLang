"""
GreenLang Observability - Tracing Module Tests
===============================================

Comprehensive unit tests for OpenTelemetry tracing functionality.
"""

import pytest
import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from observability.tracing import (
    TracingManager,
    TracingConfig,
    TraceContext,
    Span,
    SpanKind,
    SpanAttributes,
    ExporterType,
    SamplingStrategy,
    AlwaysOnSampler,
    AlwaysOffSampler,
    TraceIdRatioSampler,
    ParentBasedSampler,
    ConsoleSpanExporter,
    traced,
    traced_async,
    get_current_span,
    get_current_trace_id,
    inject_trace_context,
    extract_trace_context,
)


class TestTracingConfig:
    """Tests for TracingConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = TracingConfig(service_name="test-service")

        assert config.service_name == "test-service"
        assert config.service_version == "1.0.0"
        assert config.environment == "development"
        assert config.exporter_type == ExporterType.CONSOLE
        assert config.sampling_strategy == SamplingStrategy.ALWAYS_ON
        assert config.sampling_ratio == 1.0
        assert config.enabled is True

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = TracingConfig(
            service_name="gl-006-heatreclaim",
            service_version="2.0.0",
            environment="production",
            exporter_type=ExporterType.JAEGER,
            jaeger_endpoint="http://jaeger:14268/api/traces",
            sampling_strategy=SamplingStrategy.TRACE_ID_RATIO,
            sampling_ratio=0.5,
        )

        assert config.service_name == "gl-006-heatreclaim"
        assert config.service_version == "2.0.0"
        assert config.environment == "production"
        assert config.exporter_type == ExporterType.JAEGER
        assert config.jaeger_endpoint == "http://jaeger:14268/api/traces"
        assert config.sampling_ratio == 0.5

    def test_invalid_sampling_ratio(self) -> None:
        """Test validation of sampling ratio."""
        with pytest.raises(ValueError):
            TracingConfig(service_name="test", sampling_ratio=1.5)

        with pytest.raises(ValueError):
            TracingConfig(service_name="test", sampling_ratio=-0.1)

    def test_from_env(self) -> None:
        """Test configuration from environment variables."""
        with patch.dict(os.environ, {
            "SERVICE_VERSION": "3.0.0",
            "ENVIRONMENT": "staging",
            "OTEL_EXPORTER_TYPE": "zipkin",
        }):
            config = TracingConfig.from_env("test-service")
            assert config.service_name == "test-service"
            assert config.service_version == "3.0.0"
            assert config.environment == "staging"


class TestTraceContext:
    """Tests for TraceContext."""

    def test_to_traceparent(self) -> None:
        """Test W3C traceparent format generation."""
        context = TraceContext(
            trace_id="0af7651916cd43dd8448eb211c80319c",
            span_id="b7ad6b7169203331",
            trace_flags=1,
        )

        traceparent = context.to_traceparent()
        assert traceparent == "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"

    def test_from_traceparent(self) -> None:
        """Test parsing W3C traceparent header."""
        traceparent = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
        context = TraceContext.from_traceparent(traceparent)

        assert context is not None
        assert context.trace_id == "0af7651916cd43dd8448eb211c80319c"
        assert context.span_id == "b7ad6b7169203331"
        assert context.trace_flags == 1

    def test_from_traceparent_invalid(self) -> None:
        """Test parsing invalid traceparent header."""
        assert TraceContext.from_traceparent("invalid") is None
        assert TraceContext.from_traceparent("01-abc-def-00") is None
        assert TraceContext.from_traceparent("") is None

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        context = TraceContext(
            trace_id="abc123",
            span_id="def456",
            trace_state="vendor=value",
        )

        d = context.to_dict()
        assert "traceparent" in d
        assert d.get("tracestate") == "vendor=value"


class TestSpan:
    """Tests for Span class."""

    def test_span_creation(self) -> None:
        """Test span creation with attributes."""
        span = Span(
            name="test-span",
            span_id="1234567890abcdef",
            trace_id="0af7651916cd43dd8448eb211c80319c",
            kind=SpanKind.SERVER,
        )

        assert span.name == "test-span"
        assert span.span_id == "1234567890abcdef"
        assert span.kind == SpanKind.SERVER
        assert span.status == "OK"

    def test_set_attribute(self) -> None:
        """Test setting span attributes."""
        span = Span(name="test", span_id="abc", trace_id="def")
        span.set_attribute("key", "value")
        span.set_attribute("count", 42)

        assert span.attributes["key"] == "value"
        assert span.attributes["count"] == 42

    def test_add_event(self) -> None:
        """Test adding events to span."""
        span = Span(name="test", span_id="abc", trace_id="def")
        span.add_event("checkpoint", attributes={"step": 1})

        assert len(span.events) == 1
        assert span.events[0]["name"] == "checkpoint"
        assert span.events[0]["attributes"]["step"] == 1

    def test_set_status(self) -> None:
        """Test setting span status."""
        span = Span(name="test", span_id="abc", trace_id="def")
        span.set_status("ERROR", "Something went wrong")

        assert span.status == "ERROR"
        assert span.status_message == "Something went wrong"

    def test_record_exception(self) -> None:
        """Test recording exception."""
        span = Span(name="test", span_id="abc", trace_id="def")
        try:
            raise ValueError("Test error")
        except ValueError as e:
            span.record_exception(e)

        assert span.status == "ERROR"
        assert len(span.events) == 1
        assert span.events[0]["name"] == "exception"
        assert span.events[0]["attributes"]["exception.type"] == "ValueError"

    def test_to_dict(self) -> None:
        """Test span serialization."""
        span = Span(
            name="test",
            span_id="abc",
            trace_id="def",
            start_time=datetime.now(timezone.utc),
        )
        span.end_time = datetime.now(timezone.utc)

        d = span.to_dict()
        assert d["name"] == "test"
        assert d["span_id"] == "abc"
        assert d["trace_id"] == "def"
        assert "duration_ms" in d


class TestSamplers:
    """Tests for sampling strategies."""

    def test_always_on_sampler(self) -> None:
        """Test AlwaysOnSampler."""
        sampler = AlwaysOnSampler()
        assert sampler.should_sample("any-trace-id") is True

    def test_always_off_sampler(self) -> None:
        """Test AlwaysOffSampler."""
        sampler = AlwaysOffSampler()
        assert sampler.should_sample("any-trace-id") is False

    def test_trace_id_ratio_sampler(self) -> None:
        """Test TraceIdRatioSampler."""
        # With ratio 1.0, should always sample
        sampler = TraceIdRatioSampler(1.0)
        assert sampler.should_sample("ffffffff") is True

        # With ratio 0.0, should never sample
        sampler = TraceIdRatioSampler(0.0)
        assert sampler.should_sample("00000000") is False

    def test_parent_based_sampler(self) -> None:
        """Test ParentBasedSampler."""
        root_sampler = AlwaysOnSampler()
        sampler = ParentBasedSampler(root_sampler)

        # Without parent, use root sampler
        assert sampler.should_sample("trace-id") is True

        # With sampled parent
        parent = TraceContext(trace_id="abc", span_id="def", trace_flags=1)
        assert sampler.should_sample("trace-id", parent) is True

        # With non-sampled parent
        parent = TraceContext(trace_id="abc", span_id="def", trace_flags=0)
        assert sampler.should_sample("trace-id", parent) is False


class TestTracingManager:
    """Tests for TracingManager."""

    def test_manager_initialization(self) -> None:
        """Test manager initialization."""
        manager = TracingManager(
            service_name="test-service",
            exporter_type=ExporterType.CONSOLE,
        )

        assert manager.config.service_name == "test-service"
        assert manager.config.exporter_type == ExporterType.CONSOLE

    def test_start_span_context_manager(self) -> None:
        """Test span creation with context manager."""
        manager = TracingManager(service_name="test")

        with manager.start_span("test-operation") as span:
            assert span.name == "test-operation"
            assert span.start_time is not None
            span.set_attribute("test", "value")

        assert span.end_time is not None

    def test_nested_spans(self) -> None:
        """Test nested span creation."""
        manager = TracingManager(service_name="test")

        with manager.start_span("parent") as parent_span:
            parent_trace_id = parent_span.trace_id

            with manager.start_span("child") as child_span:
                # Child should have same trace_id
                assert child_span.trace_id == parent_trace_id
                # Child should have parent as parent_span_id
                assert child_span.parent_span_id == parent_span.span_id

    def test_span_with_attributes(self) -> None:
        """Test span creation with initial attributes."""
        manager = TracingManager(service_name="test")

        attrs = {"agent_id": "GL-006", "calculation_type": "pinch"}
        with manager.start_span("calc", attributes=attrs) as span:
            assert span.attributes["agent_id"] == "GL-006"
            assert span.attributes["calculation_type"] == "pinch"

    def test_span_exception_handling(self) -> None:
        """Test exception recording in spans."""
        manager = TracingManager(service_name="test")

        with pytest.raises(ValueError):
            with manager.start_span("failing") as span:
                raise ValueError("Test error")

        assert span.status == "ERROR"

    def test_disabled_tracing(self) -> None:
        """Test behavior when tracing is disabled."""
        manager = TracingManager(service_name="test", enabled=False)

        with manager.start_span("test") as span:
            # Should return no-op span
            assert span.span_id == ""
            assert span.trace_id == ""

    def test_get_current_context(self) -> None:
        """Test trace context retrieval."""
        manager = TracingManager(service_name="test")

        # No context when no span active
        assert manager.get_current_context() is None

        with manager.start_span("test") as span:
            context = manager.get_current_context()
            assert context is not None
            assert context.trace_id == span.trace_id
            assert context.span_id == span.span_id

    def test_flush(self) -> None:
        """Test span flushing."""
        manager = TracingManager(service_name="test")

        with manager.start_span("test"):
            pass

        # Should not raise
        manager.flush()


class TestTracedDecorator:
    """Tests for @traced decorator."""

    def test_traced_decorator(self) -> None:
        """Test basic traced decorator."""
        manager = TracingManager(service_name="test")

        @traced(name="my-function")
        def my_function(x: int) -> int:
            return x * 2

        result = my_function(5)
        assert result == 10

    def test_traced_with_attributes(self) -> None:
        """Test traced decorator with attributes."""
        manager = TracingManager(service_name="test")

        @traced(
            name="calculate",
            attributes={"type": "thermal", "version": "1.0"},
        )
        def calculate(x: int) -> int:
            return x + 1

        result = calculate(10)
        assert result == 11

    def test_traced_exception_handling(self) -> None:
        """Test traced decorator with exception."""
        manager = TracingManager(service_name="test")

        @traced(name="failing")
        def failing_function() -> None:
            raise RuntimeError("Test error")

        with pytest.raises(RuntimeError):
            failing_function()

    def test_traced_record_args(self) -> None:
        """Test traced decorator with argument recording."""
        manager = TracingManager(service_name="test")

        @traced(name="with-args", record_args=True)
        def with_args(a: int, b: str, c: float = 1.0) -> str:
            return f"{a}-{b}-{c}"

        result = with_args(1, "test", c=2.5)
        assert result == "1-test-2.5"


class TestContextPropagation:
    """Tests for trace context propagation."""

    def test_inject_trace_context(self) -> None:
        """Test injecting trace context into headers."""
        manager = TracingManager(service_name="test")

        headers: dict = {}
        with manager.start_span("test"):
            headers = inject_trace_context(headers)

        assert "traceparent" in headers

    def test_extract_trace_context(self) -> None:
        """Test extracting trace context from headers."""
        headers = {
            "traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
            "tracestate": "vendor=value",
        }

        context = extract_trace_context(headers)
        assert context is not None
        assert context.trace_id == "0af7651916cd43dd8448eb211c80319c"
        assert context.trace_state == "vendor=value"

    def test_context_propagation_roundtrip(self) -> None:
        """Test full context propagation roundtrip."""
        manager = TracingManager(service_name="test")

        # Create span and get context
        with manager.start_span("origin") as origin_span:
            context = manager.get_current_context()
            headers = context.to_dict() if context else {}

        # Extract context in "downstream" service
        extracted = extract_trace_context(headers)
        assert extracted is not None
        assert extracted.trace_id == origin_span.trace_id


class TestSpanAttributes:
    """Tests for SpanAttributes helper class."""

    def test_span_attributes_to_dict(self) -> None:
        """Test SpanAttributes conversion to dictionary."""
        attrs = SpanAttributes(
            agent_id="GL-006",
            agent_version="1.0.0",
            calculation_type="pinch_analysis",
            provenance_hash="abc123",
        )

        d = attrs.to_dict()
        assert d["greenlang.agent.id"] == "GL-006"
        assert d["greenlang.agent.version"] == "1.0.0"
        assert d["greenlang.calculation.type"] == "pinch_analysis"
        assert d["greenlang.provenance.hash"] == "abc123"
        assert d["greenlang.framework.name"] == "greenlang"

    def test_span_attributes_custom(self) -> None:
        """Test SpanAttributes with custom attributes."""
        attrs = SpanAttributes(
            agent_id="GL-006",
            custom={"stream_count": 5, "min_approach_temp": 10.0},
        )

        d = attrs.to_dict()
        assert d["greenlang.custom.stream_count"] == 5
        assert d["greenlang.custom.min_approach_temp"] == 10.0


class TestConsoleExporter:
    """Tests for ConsoleSpanExporter."""

    def test_export(self) -> None:
        """Test exporting spans to console."""
        exporter = ConsoleSpanExporter()

        span = Span(
            name="test",
            span_id="abc",
            trace_id="def",
            start_time=datetime.now(timezone.utc),
        )
        span.end_time = datetime.now(timezone.utc)

        result = exporter.export([span])
        assert result is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
