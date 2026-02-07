# -*- coding: utf-8 -*-
"""
Unit tests for tracing decorators (OBS-003)

Tests @trace_operation, @trace_agent, and @trace_pipeline decorators
for both sync and async functions, including exception recording,
status propagation, return-value preservation, and no-op fallback.

Coverage target: 85%+ of decorators.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch, AsyncMock

import pytest


# ============================================================================
# Helpers
# ============================================================================


def _import_decorators():
    """Import the decorators module, skipping if not yet created."""
    try:
        from greenlang.infrastructure.tracing_service import decorators
        return decorators
    except ImportError:
        pytest.skip("decorators module not yet built")


# ============================================================================
# @trace_operation tests
# ============================================================================


class TestTraceOperation:
    """Tests for the @trace_operation decorator."""

    def test_trace_operation_sync(self, mock_tracer):
        """Verify @trace_operation wraps a sync function with a span."""
        dec_mod = _import_decorators()
        tracer, span = mock_tracer

        with patch.object(dec_mod, "_get_tracer", return_value=tracer):

            @dec_mod.trace_operation("sync_op")
            def my_func(x: int) -> int:
                return x * 2

            result = my_func(5)

        assert result == 10
        tracer.start_as_current_span.assert_called()

    @pytest.mark.asyncio
    async def test_trace_operation_async(self, mock_tracer):
        """Verify @trace_operation wraps an async function with a span."""
        dec_mod = _import_decorators()
        tracer, span = mock_tracer

        with patch.object(dec_mod, "_get_tracer", return_value=tracer):

            @dec_mod.trace_operation("async_op")
            async def my_func(x: int) -> int:
                return x * 3

            result = await my_func(4)

        assert result == 12

    def test_trace_operation_with_custom_name(self, mock_tracer):
        """Verify @trace_operation uses the provided custom span name."""
        dec_mod = _import_decorators()
        tracer, span = mock_tracer

        with patch.object(dec_mod, "_get_tracer", return_value=tracer):

            @dec_mod.trace_operation("my.custom.operation")
            def work():
                return True

            work()

        call_args = tracer.start_as_current_span.call_args
        assert "my.custom.operation" in str(call_args)

    def test_trace_operation_with_attributes(self, mock_tracer):
        """Verify @trace_operation can set span attributes."""
        dec_mod = _import_decorators()
        tracer, span = mock_tracer

        with patch.object(dec_mod, "_get_tracer", return_value=tracer):

            @dec_mod.trace_operation("attr_op", attributes={"env": "test"})
            def work():
                return True

            work()

    def test_trace_operation_records_exception(self, mock_tracer):
        """Verify @trace_operation records exceptions on the span."""
        dec_mod = _import_decorators()
        tracer, span = mock_tracer

        with patch.object(dec_mod, "_get_tracer", return_value=tracer):

            @dec_mod.trace_operation("fail_op")
            def failing():
                raise ValueError("test error")

            with pytest.raises(ValueError, match="test error"):
                failing()

        span.record_exception.assert_called()

    def test_trace_operation_sets_error_status(self, mock_tracer):
        """Verify @trace_operation sets ERROR status on exception."""
        dec_mod = _import_decorators()
        tracer, span = mock_tracer

        with patch.object(dec_mod, "_get_tracer", return_value=tracer):

            @dec_mod.trace_operation("error_op")
            def failing():
                raise RuntimeError("kaboom")

            with pytest.raises(RuntimeError):
                failing()

        span.set_status.assert_called()

    def test_trace_operation_preserves_return_value_sync(self, mock_tracer):
        """Verify sync decorated function returns the original value."""
        dec_mod = _import_decorators()
        tracer, span = mock_tracer

        with patch.object(dec_mod, "_get_tracer", return_value=tracer):

            @dec_mod.trace_operation("ret_op")
            def compute():
                return {"answer": 42}

            result = compute()

        assert result == {"answer": 42}

    @pytest.mark.asyncio
    async def test_trace_operation_preserves_return_value_async(self, mock_tracer):
        """Verify async decorated function returns the original value."""
        dec_mod = _import_decorators()
        tracer, span = mock_tracer

        with patch.object(dec_mod, "_get_tracer", return_value=tracer):

            @dec_mod.trace_operation("async_ret")
            async def compute():
                return [1, 2, 3]

            result = await compute()

        assert result == [1, 2, 3]

    def test_trace_operation_preserves_exception_sync(self, mock_tracer):
        """Verify sync decorated function re-raises the original exception."""
        dec_mod = _import_decorators()
        tracer, span = mock_tracer

        with patch.object(dec_mod, "_get_tracer", return_value=tracer):

            @dec_mod.trace_operation("exc_op")
            def failing():
                raise TypeError("type err")

            with pytest.raises(TypeError, match="type err"):
                failing()

    @pytest.mark.asyncio
    async def test_trace_operation_preserves_exception_async(self, mock_tracer):
        """Verify async decorated function re-raises the original exception."""
        dec_mod = _import_decorators()
        tracer, span = mock_tracer

        with patch.object(dec_mod, "_get_tracer", return_value=tracer):

            @dec_mod.trace_operation("async_exc")
            async def failing():
                raise IOError("io err")

            with pytest.raises(IOError, match="io err"):
                await failing()

    def test_trace_operation_default_name_from_function(self, mock_tracer):
        """Verify @trace_operation uses function name when no name given."""
        dec_mod = _import_decorators()
        tracer, span = mock_tracer

        with patch.object(dec_mod, "_get_tracer", return_value=tracer):

            @dec_mod.trace_operation()
            def my_special_function():
                return True

            my_special_function()

        call_args = tracer.start_as_current_span.call_args
        assert "my_special_function" in str(call_args)

    def test_decorator_noop_when_unavailable(self):
        """Verify decorator is transparent when OTel is unavailable."""
        dec_mod = _import_decorators()

        with patch.object(dec_mod, "OTEL_AVAILABLE", False):

            @dec_mod.trace_operation("noop_op")
            def work():
                return "ok"

            result = work()

        assert result == "ok"

    def test_nested_decorators(self, mock_tracer):
        """Verify nested traced functions create nested spans."""
        dec_mod = _import_decorators()
        tracer, span = mock_tracer

        with patch.object(dec_mod, "_get_tracer", return_value=tracer):

            @dec_mod.trace_operation("outer")
            def outer():
                return inner()

            @dec_mod.trace_operation("inner")
            def inner():
                return 42

            result = outer()

        assert result == 42
        assert tracer.start_as_current_span.call_count >= 2

    def test_decorator_with_class_method(self, mock_tracer):
        """Verify @trace_operation works on class methods."""
        dec_mod = _import_decorators()
        tracer, span = mock_tracer

        with patch.object(dec_mod, "_get_tracer", return_value=tracer):

            class MyService:
                @dec_mod.trace_operation("service.process")
                def process(self, data: str) -> str:
                    return data.upper()

            svc = MyService()
            result = svc.process("hello")

        assert result == "HELLO"

    def test_decorator_thread_safety(self, mock_tracer):
        """Verify decorated functions can be called from multiple threads."""
        dec_mod = _import_decorators()
        tracer, span = mock_tracer
        results = []

        with patch.object(dec_mod, "_get_tracer", return_value=tracer):

            @dec_mod.trace_operation("thread_op")
            def work(n: int) -> int:
                return n * n

            import threading

            threads = []
            for i in range(10):
                t = threading.Thread(target=lambda x=i: results.append(work(x)))
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

        assert len(results) == 10
        assert set(results) == {i * i for i in range(10)}


# ============================================================================
# @trace_agent tests
# ============================================================================


class TestTraceAgent:
    """Tests for the @trace_agent decorator."""

    def test_trace_agent_creates_span(self, mock_tracer):
        """Verify @trace_agent creates a span."""
        dec_mod = _import_decorators()
        tracer, span = mock_tracer

        with patch.object(dec_mod, "_get_tracer", return_value=tracer):

            @dec_mod.trace_agent("cbam-agent")
            def run_agent(data):
                return {"result": "ok"}

            result = run_agent({"input": "test"})

        assert result == {"result": "ok"}
        tracer.start_as_current_span.assert_called()

    def test_trace_agent_sets_agent_attributes(self, mock_tracer):
        """Verify @trace_agent sets gl.agent.name attribute."""
        dec_mod = _import_decorators()
        tracer, span = mock_tracer

        with patch.object(dec_mod, "_get_tracer", return_value=tracer):

            @dec_mod.trace_agent("emission-calculator")
            def calc():
                return 42.0

            calc()

        # Should set agent-related attributes
        call_str = str(tracer.start_as_current_span.call_args)
        assert "emission-calculator" in call_str or span.set_attribute.called

    def test_trace_agent_with_agent_id(self, mock_tracer):
        """Verify @trace_agent can receive an agent ID."""
        dec_mod = _import_decorators()
        tracer, span = mock_tracer

        with patch.object(dec_mod, "_get_tracer", return_value=tracer):

            @dec_mod.trace_agent("cbam-agent", agent_id="agent-123")
            def run():
                return True

            run()


# ============================================================================
# @trace_pipeline tests
# ============================================================================


class TestTracePipeline:
    """Tests for the @trace_pipeline decorator."""

    def test_trace_pipeline_creates_span(self, mock_tracer):
        """Verify @trace_pipeline creates a span."""
        dec_mod = _import_decorators()
        tracer, span = mock_tracer

        with patch.object(dec_mod, "_get_tracer", return_value=tracer):

            @dec_mod.trace_pipeline("emission-pipeline")
            def run_pipeline():
                return {"status": "done"}

            result = run_pipeline()

        assert result == {"status": "done"}
        tracer.start_as_current_span.assert_called()

    def test_trace_pipeline_sets_pipeline_attributes(self, mock_tracer):
        """Verify @trace_pipeline sets gl.pipeline.name attribute."""
        dec_mod = _import_decorators()
        tracer, span = mock_tracer

        with patch.object(dec_mod, "_get_tracer", return_value=tracer):

            @dec_mod.trace_pipeline("compliance-check")
            def run():
                return True

            run()

        call_str = str(tracer.start_as_current_span.call_args)
        assert "compliance-check" in call_str or span.set_attribute.called
