# -*- coding: utf-8 -*-
"""
CorrelationManager - Generate and propagate correlation IDs through agent
execution chains.

Uses Python contextvars for async-safe propagation.  Supports injection
into HTTP headers (W3C traceparent / tracestate) and Redis messages, and
extraction from incoming requests.

Example:
    >>> mgr = CorrelationManager()
    >>> ctx = mgr.new_context(tenant_id="acme-corp")
    >>> headers = mgr.inject_http_headers()
    >>> # ... on the receiving side ...
    >>> mgr.extract_from_http(headers)

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import contextvars
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Context variable for async propagation
# ---------------------------------------------------------------------------

_correlation_ctx: contextvars.ContextVar[Optional["CorrelationContext"]] = (
    contextvars.ContextVar("_correlation_ctx", default=None)
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CorrelationContext:
    """Immutable correlation context propagated through an execution chain.

    Attributes:
        correlation_id: Unique ID for the entire request chain.
        parent_span_id: Span ID of the parent operation.
        trace_id: W3C trace-id (32 hex chars).
        tenant_id: Tenant identifier.
        baggage: Arbitrary key-value baggage items.
        created_at: UTC ISO-8601 creation timestamp.
    """

    correlation_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    parent_span_id: str = ""
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    tenant_id: str = ""
    baggage: Dict[str, str] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "correlation_id": self.correlation_id,
            "parent_span_id": self.parent_span_id,
            "trace_id": self.trace_id,
            "tenant_id": self.tenant_id,
            "baggage": dict(self.baggage),
            "created_at": self.created_at,
        }


# ---------------------------------------------------------------------------
# CorrelationManager
# ---------------------------------------------------------------------------

class CorrelationManager:
    """Manage correlation-ID lifecycle and propagation.

    Thread-safe and async-safe via contextvars.

    Example:
        >>> mgr = CorrelationManager()
        >>> ctx = mgr.new_context(tenant_id="acme")
        >>> assert mgr.current_context().tenant_id == "acme"
    """

    # W3C trace-context header names
    HEADER_TRACEPARENT = "traceparent"
    HEADER_TRACESTATE = "tracestate"
    HEADER_CORRELATION_ID = "x-correlation-id"
    HEADER_TENANT_ID = "x-tenant-id"
    HEADER_BAGGAGE = "baggage"

    # Redis message field names
    REDIS_CORRELATION_ID = "gl_correlation_id"
    REDIS_TRACE_ID = "gl_trace_id"
    REDIS_TENANT_ID = "gl_tenant_id"

    # ---- Context lifecycle -----------------------------------------------

    def new_context(
        self,
        *,
        correlation_id: Optional[str] = None,
        parent_span_id: str = "",
        trace_id: Optional[str] = None,
        tenant_id: str = "",
        baggage: Optional[Dict[str, str]] = None,
    ) -> CorrelationContext:
        """Create a new correlation context and set it as current.

        Args:
            correlation_id: Override correlation ID (auto-generated if None).
            parent_span_id: Parent span ID for nesting.
            trace_id: Override trace ID (auto-generated if None).
            tenant_id: Tenant identifier.
            baggage: Extra key-value pairs.

        Returns:
            The newly created CorrelationContext.
        """
        ctx = CorrelationContext(
            correlation_id=correlation_id or uuid.uuid4().hex,
            parent_span_id=parent_span_id,
            trace_id=trace_id or uuid.uuid4().hex,
            tenant_id=tenant_id,
            baggage=baggage or {},
        )
        _correlation_ctx.set(ctx)
        logger.debug("New correlation context: %s", ctx.correlation_id)
        return ctx

    def current_context(self) -> Optional[CorrelationContext]:
        """Return the current correlation context (or None)."""
        return _correlation_ctx.get()

    def set_context(self, ctx: CorrelationContext) -> None:
        """Explicitly set the current correlation context.

        Args:
            ctx: Context to make current.
        """
        _correlation_ctx.set(ctx)

    def clear_context(self) -> None:
        """Remove the current correlation context."""
        _correlation_ctx.set(None)

    # ---- HTTP injection / extraction -------------------------------------

    def inject_http_headers(
        self,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """Inject correlation context into HTTP headers.

        Adds traceparent, tracestate, x-correlation-id, x-tenant-id,
        and baggage headers.

        Args:
            headers: Existing header dict to mutate (created if None).

        Returns:
            The header dict with context injected.
        """
        headers = dict(headers or {})
        ctx = self.current_context()
        if ctx is None:
            return headers

        # W3C traceparent: version-trace_id-parent_id-flags
        trace_id_32 = ctx.trace_id.ljust(32, "0")[:32]
        parent_id_16 = ctx.parent_span_id.ljust(16, "0")[:16] if ctx.parent_span_id else "0" * 16
        headers[self.HEADER_TRACEPARENT] = f"00-{trace_id_32}-{parent_id_16}-01"

        # tracestate (vendor-specific)
        headers[self.HEADER_TRACESTATE] = f"gl={ctx.correlation_id}"

        # Custom headers
        headers[self.HEADER_CORRELATION_ID] = ctx.correlation_id
        if ctx.tenant_id:
            headers[self.HEADER_TENANT_ID] = ctx.tenant_id

        # Baggage (RFC 8941 member format)
        if ctx.baggage:
            baggage_items = ",".join(f"{k}={v}" for k, v in ctx.baggage.items())
            headers[self.HEADER_BAGGAGE] = baggage_items

        return headers

    def extract_from_http(self, headers: Dict[str, str]) -> CorrelationContext:
        """Extract correlation context from incoming HTTP headers.

        If a correlation ID header is present it is reused; otherwise a new
        context is generated.

        Args:
            headers: Incoming request headers.

        Returns:
            The restored or newly-created CorrelationContext.
        """
        correlation_id = headers.get(self.HEADER_CORRELATION_ID)
        tenant_id = headers.get(self.HEADER_TENANT_ID, "")
        trace_id: Optional[str] = None
        parent_span_id = ""

        # Parse traceparent
        traceparent = headers.get(self.HEADER_TRACEPARENT, "")
        if traceparent:
            parts = traceparent.split("-")
            if len(parts) >= 3:
                trace_id = parts[1]
                parent_span_id = parts[2]

        # Parse baggage
        baggage: Dict[str, str] = {}
        raw_baggage = headers.get(self.HEADER_BAGGAGE, "")
        if raw_baggage:
            for item in raw_baggage.split(","):
                if "=" in item:
                    k, v = item.split("=", 1)
                    baggage[k.strip()] = v.strip()

        ctx = self.new_context(
            correlation_id=correlation_id,
            parent_span_id=parent_span_id,
            trace_id=trace_id,
            tenant_id=tenant_id,
            baggage=baggage,
        )
        return ctx

    # ---- Redis injection / extraction ------------------------------------

    def inject_redis_message(
        self,
        message: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Inject correlation context into a Redis message dict.

        Args:
            message: Message payload to augment.

        Returns:
            The message with correlation fields added.
        """
        ctx = self.current_context()
        if ctx is None:
            return message

        message[self.REDIS_CORRELATION_ID] = ctx.correlation_id
        message[self.REDIS_TRACE_ID] = ctx.trace_id
        if ctx.tenant_id:
            message[self.REDIS_TENANT_ID] = ctx.tenant_id
        return message

    def extract_from_redis(self, message: Dict[str, Any]) -> CorrelationContext:
        """Extract correlation context from a Redis message.

        Args:
            message: Incoming Redis message payload.

        Returns:
            The restored CorrelationContext.
        """
        return self.new_context(
            correlation_id=message.get(self.REDIS_CORRELATION_ID),
            trace_id=message.get(self.REDIS_TRACE_ID),
            tenant_id=message.get(self.REDIS_TENANT_ID, ""),
        )


__all__ = ["CorrelationContext", "CorrelationManager"]
