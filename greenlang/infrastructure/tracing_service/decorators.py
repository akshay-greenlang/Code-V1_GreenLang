# -*- coding: utf-8 -*-
"""
Tracing Decorators - @trace_operation, @trace_agent, @trace_pipeline (OBS-003)

Provides composable decorators that wrap sync and async functions in
OpenTelemetry spans with automatic exception recording, status tracking,
and GreenLang-specific attribute injection.

All decorators degrade to passthrough when the OTel SDK is not installed
or tracing is disabled.

Example:
    >>> from greenlang.infrastructure.tracing_service.decorators import (
    ...     trace_operation, trace_agent, trace_pipeline,
    ... )
    >>>
    >>> @trace_operation(name="calculate_emissions")
    ... async def calculate(data):
    ...     return data * 2.5
    >>>
    >>> @trace_agent(agent_type="carbon-calc", agent_id="a-001")
    ... async def run_agent(input_data):
    ...     return process(input_data)
    >>>
    >>> @trace_pipeline(pipeline_name="ghg-pipeline")
    ... async def run_stage(stage_data):
    ...     return transform(stage_data)

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import functools
import inspect
import logging
import time
from typing import Any, Callable, Dict, Optional, TypeVar, Union

from greenlang.infrastructure.tracing_service.provider import get_tracer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional OTel imports
# ---------------------------------------------------------------------------

try:
    from opentelemetry.trace import StatusCode, Status, SpanKind

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

    class StatusCode:  # type: ignore[no-redef]
        OK = "OK"
        ERROR = "ERROR"
        UNSET = "UNSET"

    class SpanKind:  # type: ignore[no-redef]
        INTERNAL = 0
        SERVER = 1
        CLIENT = 2
        PRODUCER = 3
        CONSUMER = 4

F = TypeVar("F", bound=Callable[..., Any])


# ---------------------------------------------------------------------------
# @trace_operation
# ---------------------------------------------------------------------------

def trace_operation(
    name: Optional[str] = None,
    *,
    kind: Any = None,
    attributes: Optional[Dict[str, Any]] = None,
    record_exception: bool = True,
    set_status_on_exception: bool = True,
) -> Callable[[F], F]:
    """Decorate a function to be traced as an OpenTelemetry span.

    Works with both sync and async functions.  The span name defaults to
    ``module.qualified_name`` if *name* is not provided.

    Args:
        name: Explicit span name (defaults to ``module.qualname``).
        kind: OTel SpanKind (defaults to INTERNAL).
        attributes: Static attributes to set on every invocation.
        record_exception: Whether to call ``span.record_exception()`` on error.
        set_status_on_exception: Whether to set ERROR status on error.

    Returns:
        The decorated function.
    """

    def decorator(func: F) -> F:
        span_name = name or f"{func.__module__}.{func.__qualname__}"
        static_attrs = dict(attributes or {})
        static_attrs.setdefault("code.function", func.__qualname__)
        static_attrs.setdefault("code.module", func.__module__,)
        span_kind = kind if kind is not None else (
            SpanKind.INTERNAL if OTEL_AVAILABLE else None
        )

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                tracer = get_tracer(func.__module__)
                start = time.monotonic()
                with tracer.start_as_current_span(
                    span_name, kind=span_kind
                ) as span:
                    _set_attributes(span, static_attrs)
                    _set_kwarg_attributes(span, kwargs)
                    try:
                        result = await func(*args, **kwargs)
                        _set_ok(span)
                        return result
                    except Exception as exc:
                        _handle_exception(
                            span, exc, record_exception, set_status_on_exception,
                        )
                        raise
                    finally:
                        elapsed_ms = (time.monotonic() - start) * 1000
                        _safe_set(span, "gl.duration_ms", round(elapsed_ms, 2))

            return async_wrapper  # type: ignore[return-value]

        else:

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                tracer = get_tracer(func.__module__)
                start = time.monotonic()
                with tracer.start_as_current_span(
                    span_name, kind=span_kind
                ) as span:
                    _set_attributes(span, static_attrs)
                    _set_kwarg_attributes(span, kwargs)
                    try:
                        result = func(*args, **kwargs)
                        _set_ok(span)
                        return result
                    except Exception as exc:
                        _handle_exception(
                            span, exc, record_exception, set_status_on_exception,
                        )
                        raise
                    finally:
                        elapsed_ms = (time.monotonic() - start) * 1000
                        _safe_set(span, "gl.duration_ms", round(elapsed_ms, 2))

            return sync_wrapper  # type: ignore[return-value]

    return decorator


# ---------------------------------------------------------------------------
# @trace_agent
# ---------------------------------------------------------------------------

def trace_agent(
    agent_type: str,
    agent_id: Optional[str] = None,
    *,
    record_exception: bool = True,
) -> Callable[[F], F]:
    """Decorate a function as an agent execution span.

    Adds ``gl.agent_type``, ``gl.agent_id``, and ``gl.agent_operation``
    attributes to the span.  The span name follows the convention
    ``agent.<agent_type>.execute``.

    Args:
        agent_type: Logical agent type name (e.g. "carbon-calc").
        agent_id: Unique agent instance identifier.
        record_exception: Whether to record exceptions on the span.

    Returns:
        The decorated function.
    """

    def decorator(func: F) -> F:
        span_name = f"agent.{agent_type}.execute"
        base_attrs: Dict[str, Any] = {
            "gl.agent_type": agent_type,
            "gl.agent_operation": "execute",
            "code.function": func.__qualname__,
            "code.module": func.__module__,
        }
        if agent_id:
            base_attrs["gl.agent_id"] = agent_id

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                tracer = get_tracer(func.__module__)
                start = time.monotonic()
                with tracer.start_as_current_span(span_name) as span:
                    _set_attributes(span, base_attrs)
                    _set_kwarg_attributes(span, kwargs)
                    try:
                        result = await func(*args, **kwargs)
                        _set_ok(span)
                        return result
                    except Exception as exc:
                        _handle_exception(span, exc, record_exception, True)
                        raise
                    finally:
                        elapsed_ms = (time.monotonic() - start) * 1000
                        _safe_set(span, "gl.duration_ms", round(elapsed_ms, 2))

            return async_wrapper  # type: ignore[return-value]

        else:

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                tracer = get_tracer(func.__module__)
                start = time.monotonic()
                with tracer.start_as_current_span(span_name) as span:
                    _set_attributes(span, base_attrs)
                    _set_kwarg_attributes(span, kwargs)
                    try:
                        result = func(*args, **kwargs)
                        _set_ok(span)
                        return result
                    except Exception as exc:
                        _handle_exception(span, exc, record_exception, True)
                        raise
                    finally:
                        elapsed_ms = (time.monotonic() - start) * 1000
                        _safe_set(span, "gl.duration_ms", round(elapsed_ms, 2))

            return sync_wrapper  # type: ignore[return-value]

    return decorator


# ---------------------------------------------------------------------------
# @trace_pipeline
# ---------------------------------------------------------------------------

def trace_pipeline(
    pipeline_name: str,
    *,
    stage: str = "",
    record_exception: bool = True,
) -> Callable[[F], F]:
    """Decorate a function as a pipeline stage span.

    Adds ``gl.pipeline_name``, ``gl.pipeline_stage``, and related attributes.
    The span name follows ``pipeline.<name>.<stage or func_name>``.

    Args:
        pipeline_name: Pipeline logical name.
        stage: Pipeline stage name (defaults to function name).
        record_exception: Whether to record exceptions on the span.

    Returns:
        The decorated function.
    """

    def decorator(func: F) -> F:
        stage_name = stage or func.__name__
        span_name = f"pipeline.{pipeline_name}.{stage_name}"
        base_attrs: Dict[str, Any] = {
            "gl.pipeline_name": pipeline_name,
            "gl.pipeline_stage": stage_name,
            "code.function": func.__qualname__,
            "code.module": func.__module__,
        }

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                tracer = get_tracer(func.__module__)
                start = time.monotonic()
                with tracer.start_as_current_span(span_name) as span:
                    _set_attributes(span, base_attrs)
                    # Inject pipeline_id from kwargs if provided
                    pipeline_id = kwargs.get("pipeline_id", "")
                    if pipeline_id:
                        _safe_set(span, "gl.pipeline_id", str(pipeline_id))
                    _set_kwarg_attributes(span, kwargs)
                    try:
                        result = await func(*args, **kwargs)
                        _set_ok(span)
                        return result
                    except Exception as exc:
                        _handle_exception(span, exc, record_exception, True)
                        raise
                    finally:
                        elapsed_ms = (time.monotonic() - start) * 1000
                        _safe_set(span, "gl.duration_ms", round(elapsed_ms, 2))

            return async_wrapper  # type: ignore[return-value]

        else:

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                tracer = get_tracer(func.__module__)
                start = time.monotonic()
                with tracer.start_as_current_span(span_name) as span:
                    _set_attributes(span, base_attrs)
                    pipeline_id = kwargs.get("pipeline_id", "")
                    if pipeline_id:
                        _safe_set(span, "gl.pipeline_id", str(pipeline_id))
                    _set_kwarg_attributes(span, kwargs)
                    try:
                        result = func(*args, **kwargs)
                        _set_ok(span)
                        return result
                    except Exception as exc:
                        _handle_exception(span, exc, record_exception, True)
                        raise
                    finally:
                        elapsed_ms = (time.monotonic() - start) * 1000
                        _safe_set(span, "gl.duration_ms", round(elapsed_ms, 2))

            return sync_wrapper  # type: ignore[return-value]

    return decorator


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _safe_set(span: Any, key: str, value: Any) -> None:
    """Set an attribute on *span*, swallowing errors."""
    try:
        span.set_attribute(key, value)
    except Exception:
        pass


def _set_attributes(span: Any, attrs: Dict[str, Any]) -> None:
    """Bulk-set attributes on a span."""
    for key, value in attrs.items():
        if value is not None:
            _safe_set(span, key, value)


def _set_kwarg_attributes(span: Any, kwargs: Dict[str, Any]) -> None:
    """Extract well-known kwargs and set them as span attributes."""
    tenant_id = kwargs.get("tenant_id")
    if tenant_id:
        _safe_set(span, "gl.tenant_id", str(tenant_id))
    request_id = kwargs.get("request_id")
    if request_id:
        _safe_set(span, "gl.request_id", str(request_id))


def _set_ok(span: Any) -> None:
    """Mark a span as OK."""
    if OTEL_AVAILABLE:
        try:
            span.set_status(Status(StatusCode.OK))
        except Exception:
            pass


def _handle_exception(
    span: Any,
    exc: Exception,
    record_exception: bool,
    set_status: bool,
) -> None:
    """Record an exception on the span and optionally set error status."""
    if record_exception:
        try:
            span.record_exception(exc)
        except Exception:
            pass
    if set_status and OTEL_AVAILABLE:
        try:
            span.set_status(Status(StatusCode.ERROR, str(exc)))
        except Exception:
            pass


__all__ = [
    "trace_operation",
    "trace_agent",
    "trace_pipeline",
]
