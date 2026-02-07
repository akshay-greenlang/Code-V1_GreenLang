# -*- coding: utf-8 -*-
"""
Trace Context Propagation - W3C TraceContext + Baggage for GreenLang (OBS-003)

Manages distributed trace context propagation across service boundaries using
W3C TraceContext and W3C Baggage standards.  Also provides GreenLang-specific
helpers for injecting tenant context into baggage so it automatically
propagates to all downstream services.

All functions degrade to no-ops when the OTel SDK is not installed.

Example:
    >>> from greenlang.infrastructure.tracing_service.context import (
    ...     inject_trace_context, extract_trace_context,
    ...     get_current_trace_id, set_tenant_context,
    ... )
    >>> headers = {}
    >>> inject_trace_context(headers)
    >>> ctx = extract_trace_context(incoming_headers)
    >>> set_tenant_context("t-acme")
    >>> trace_id = get_current_trace_id()

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional OTel imports
# ---------------------------------------------------------------------------

try:
    from opentelemetry import trace, context as otel_context, baggage
    from opentelemetry.trace.propagation.tracecontext import (
        TraceContextTextMapPropagator,
    )
    from opentelemetry.baggage.propagation import W3CBaggagePropagator
    from opentelemetry.propagators.composite import CompositePropagator
    from opentelemetry.context import attach, detach

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Composite propagator (TraceContext + Baggage)
# ---------------------------------------------------------------------------

_propagator: Optional[Any] = None

if OTEL_AVAILABLE:
    _propagator = CompositePropagator([
        TraceContextTextMapPropagator(),
        W3CBaggagePropagator(),
    ])


# ---------------------------------------------------------------------------
# Simple carrier adapter for dict[str, str]
# ---------------------------------------------------------------------------

class _DictCarrier:
    """Adapter that lets OTel propagators read/write a plain ``dict``."""

    @staticmethod
    def get(carrier: Dict[str, str], key: str) -> Optional[str]:
        """Get a value from the carrier."""
        return carrier.get(key)

    @staticmethod
    def set(carrier: Dict[str, str], key: str, value: str) -> None:
        """Set a value in the carrier."""
        carrier[key] = value

    @staticmethod
    def keys(carrier: Dict[str, str]):
        """Return all keys in the carrier."""
        return carrier.keys()


# ---------------------------------------------------------------------------
# Context injection / extraction
# ---------------------------------------------------------------------------

def inject_trace_context(carrier: Dict[str, str]) -> Dict[str, str]:
    """Inject the current trace context into *carrier*.

    Writes ``traceparent``, ``tracestate``, and ``baggage`` headers into
    the mutable dictionary.

    Args:
        carrier: Mutable dict (e.g. HTTP headers) to inject into.

    Returns:
        The same carrier dict, now containing trace context headers.
    """
    if OTEL_AVAILABLE and _propagator is not None:
        try:
            _propagator.inject(carrier)
        except Exception as exc:
            logger.debug("Failed to inject trace context: %s", exc)
    return carrier


def extract_trace_context(carrier: Dict[str, str]) -> Any:
    """Extract trace context from *carrier*.

    Returns an OTel ``Context`` token that can be used as the ``context``
    argument when starting a new span.

    Args:
        carrier: Immutable dict (e.g. incoming HTTP headers).

    Returns:
        An OTel Context, or ``None`` if extraction fails or OTel is absent.
    """
    if OTEL_AVAILABLE and _propagator is not None:
        try:
            return _propagator.extract(carrier)
        except Exception as exc:
            logger.debug("Failed to extract trace context: %s", exc)
    return None


# ---------------------------------------------------------------------------
# Current span / trace identifiers
# ---------------------------------------------------------------------------

def get_current_trace_id() -> Optional[str]:
    """Return the current trace ID as a 32-char lowercase hex string.

    Returns:
        The trace ID hex string, or ``None`` if no active span exists.
    """
    if not OTEL_AVAILABLE:
        return None
    try:
        span = trace.get_current_span()
        ctx = span.get_span_context()
        if ctx is not None and ctx.trace_id != 0:
            return format(ctx.trace_id, "032x")
    except Exception:
        pass
    return None


def get_current_span_id() -> Optional[str]:
    """Return the current span ID as a 16-char lowercase hex string.

    Returns:
        The span ID hex string, or ``None`` if no active span exists.
    """
    if not OTEL_AVAILABLE:
        return None
    try:
        span = trace.get_current_span()
        ctx = span.get_span_context()
        if ctx is not None and ctx.span_id != 0:
            return format(ctx.span_id, "016x")
    except Exception:
        pass
    return None


def get_current_trace_context() -> Dict[str, Optional[str]]:
    """Return a dict with the current ``trace_id`` and ``span_id``.

    Useful for injecting into structured log records.

    Returns:
        Dict with ``trace_id`` and ``span_id`` keys.
    """
    return {
        "trace_id": get_current_trace_id(),
        "span_id": get_current_span_id(),
    }


# ---------------------------------------------------------------------------
# Baggage management
# ---------------------------------------------------------------------------

def set_baggage_item(key: str, value: str) -> Optional[Any]:
    """Set a key/value pair in the current context's W3C Baggage.

    The baggage item will be propagated to all downstream services that
    honour the W3C Baggage header.

    Args:
        key: Baggage key.
        value: Baggage value (must be a string).

    Returns:
        An attached context token, or ``None``.
    """
    if not OTEL_AVAILABLE:
        return None
    try:
        ctx = baggage.set_baggage(key, value)
        token = attach(ctx)
        return token
    except Exception as exc:
        logger.debug("Failed to set baggage item %s: %s", key, exc)
    return None


def get_baggage_item(key: str) -> Optional[str]:
    """Retrieve a baggage item from the current context.

    Args:
        key: Baggage key to retrieve.

    Returns:
        The baggage value, or ``None``.
    """
    if not OTEL_AVAILABLE:
        return None
    try:
        return baggage.get_baggage(key)
    except Exception:
        return None


def get_all_baggage() -> Dict[str, str]:
    """Retrieve all baggage items from the current context.

    Returns:
        A dict of all baggage key/value pairs.
    """
    if not OTEL_AVAILABLE:
        return {}
    try:
        return dict(baggage.get_all())
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# GreenLang-specific tenant context
# ---------------------------------------------------------------------------

_TENANT_BAGGAGE_KEY = "gl.tenant_id"
_USER_BAGGAGE_KEY = "gl.user_id"
_CORRELATION_BAGGAGE_KEY = "gl.correlation_id"


def set_tenant_context(tenant_id: str) -> Optional[Any]:
    """Set the tenant ID in baggage for propagation to downstream services.

    Args:
        tenant_id: The tenant identifier.

    Returns:
        An attached context token.
    """
    return set_baggage_item(_TENANT_BAGGAGE_KEY, tenant_id)


def get_tenant_context() -> Optional[str]:
    """Get the tenant ID from baggage.

    Returns:
        The tenant ID string, or ``None``.
    """
    return get_baggage_item(_TENANT_BAGGAGE_KEY)


def set_user_context(user_id: str) -> Optional[Any]:
    """Set the user ID in baggage for propagation.

    Args:
        user_id: The user identifier.

    Returns:
        An attached context token.
    """
    return set_baggage_item(_USER_BAGGAGE_KEY, user_id)


def get_user_context() -> Optional[str]:
    """Get the user ID from baggage.

    Returns:
        The user ID string, or ``None``.
    """
    return get_baggage_item(_USER_BAGGAGE_KEY)


def set_correlation_id(correlation_id: str) -> Optional[Any]:
    """Set a correlation ID in baggage for cross-service tracking.

    Args:
        correlation_id: The correlation identifier.

    Returns:
        An attached context token.
    """
    return set_baggage_item(_CORRELATION_BAGGAGE_KEY, correlation_id)


def get_correlation_id() -> Optional[str]:
    """Get the correlation ID from baggage.

    Returns:
        The correlation ID string, or ``None``.
    """
    return get_baggage_item(_CORRELATION_BAGGAGE_KEY)


__all__ = [
    # Context propagation
    "inject_trace_context",
    "extract_trace_context",
    # Identifiers
    "get_current_trace_id",
    "get_current_span_id",
    "get_current_trace_context",
    # Baggage
    "set_baggage_item",
    "get_baggage_item",
    "get_all_baggage",
    # GreenLang tenant/user context
    "set_tenant_context",
    "get_tenant_context",
    "set_user_context",
    "get_user_context",
    "set_correlation_id",
    "get_correlation_id",
]
