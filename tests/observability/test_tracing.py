# -*- coding: utf-8 -*-
"""
Tests for distributed tracing
"""

import pytest
from greenlang.observability import (
    SpanKind,
    SpanContext,
    TraceConfig,
    TracingManager,
    trace_operation,
    add_span_attributes,
    get_tracing_manager,
    get_tracer,
)


class TestSpanContext:
    """Test SpanContext functionality"""

    def test_span_context_initialization(self):
        """Test span context initialization"""
        context = SpanContext()
        assert context.trace_id is not None
        assert context.span_id is not None
        assert len(context.trace_id) > 0
        assert len(context.span_id) > 0

    def test_span_context_with_parent(self):
        """Test span context with parent"""
        parent_id = "parent123"
        context = SpanContext(parent_span_id=parent_id)
        assert context.parent_span_id == parent_id

    def test_span_context_with_baggage(self):
        """Test span context with baggage"""
        baggage = {"key1": "value1", "key2": "value2"}
        context = SpanContext(baggage=baggage)
        assert context.baggage == baggage

    def test_span_context_with_attributes(self):
        """Test span context with attributes"""
        attrs = {"service": "api", "version": "1.0"}
        context = SpanContext(attributes=attrs)
        assert context.attributes == attrs


class TestTraceConfig:
    """Test TraceConfig functionality"""

    def test_trace_config_defaults(self):
        """Test trace config default values"""
        config = TraceConfig()
        assert config.service_name == "greenlang"
        assert config.environment == "production"
        assert config.sampling_rate == 1.0

    def test_trace_config_custom(self):
        """Test trace config with custom values"""
        config = TraceConfig(
            service_name="test-service",
            environment="staging",
            sampling_rate=0.5,
            console_export=True,
        )
        assert config.service_name == "test-service"
        assert config.environment == "staging"
        assert config.sampling_rate == 0.5
        assert config.console_export is True


class TestTracingManager:
    """Test TracingManager functionality"""

    def test_tracing_manager_initialization(self):
        """Test tracing manager initialization"""
        config = TraceConfig(service_name="test")
        manager = TracingManager(config)
        assert manager.config.service_name == "test"

    def test_tracing_manager_get_tracer(self):
        """Test getting tracer from manager"""
        manager = TracingManager()
        tracer = manager.get_tracer()
        assert tracer is not None

    def test_create_span(self):
        """Test creating a span"""
        manager = TracingManager()
        with manager.create_span("test_operation") as span:
            assert span is not None

    def test_create_span_with_attributes(self):
        """Test creating span with attributes"""
        manager = TracingManager()
        attrs = {"key": "value", "count": 42}
        with manager.create_span("test_operation", attributes=attrs) as span:
            assert span is not None

    def test_create_span_with_kind(self):
        """Test creating span with specific kind"""
        manager = TracingManager()
        with manager.create_span(
            "test_operation", kind=SpanKind.CLIENT
        ) as span:
            assert span is not None


class TestTraceDecorators:
    """Test trace decorator functions"""

    def test_trace_operation_decorator(self):
        """Test trace_operation decorator"""

        @trace_operation("test_function")
        def my_function():
            return "result"

        result = my_function()
        assert result == "result"

    def test_trace_operation_with_exception(self):
        """Test trace_operation decorator with exception"""

        @trace_operation("failing_function")
        def failing_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            failing_function()

    def test_trace_operation_with_tenant_id(self):
        """Test trace_operation decorator with tenant_id"""

        @trace_operation("tenant_function")
        def tenant_function(tenant_id=None):
            return tenant_id

        result = tenant_function(tenant_id="test_tenant")
        assert result == "test_tenant"

    def test_add_span_attributes_in_function(self):
        """Test adding attributes to current span"""

        @trace_operation("attributed_function")
        def attributed_function():
            add_span_attributes(custom_attr="custom_value", count=123)
            return "done"

        result = attributed_function()
        assert result == "done"


class TestGlobalTracingInstances:
    """Test global tracing instances"""

    def test_get_tracing_manager_singleton(self):
        """Test getting global tracing manager"""
        manager1 = get_tracing_manager()
        manager2 = get_tracing_manager()
        assert manager1 is manager2

    def test_get_tracer(self):
        """Test getting global tracer"""
        tracer = get_tracer()
        assert tracer is not None
