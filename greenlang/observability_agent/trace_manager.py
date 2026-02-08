# -*- coding: utf-8 -*-
"""
Distributed Tracing Engine - AGENT-FOUND-010: Observability & Telemetry Agent

Provides OTel-compatible distributed tracing with span creation, lifecycle
management, event annotation, and trace context propagation. All trace
records include SHA-256 provenance hashes for complete audit trails.

Zero-Hallucination Guarantees:
    - All span timing is deterministic UTC timestamps
    - Trace IDs and span IDs use UUID-4 generation
    - Duration calculations use pure arithmetic
    - No probabilistic sampling in the engine (sampling is caller's concern)

Example:
    >>> from greenlang.observability_agent.trace_manager import TraceManager
    >>> from greenlang.observability_agent.config import ObservabilityConfig
    >>> manager = TraceManager(ObservabilityConfig())
    >>> ctx = manager.create_trace_context()
    >>> span = manager.start_span("process_data", trace_id=ctx.trace_id)
    >>> manager.end_span(ctx.trace_id, span.span_id, status="OK")

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-010 Observability & Telemetry Agent
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class SpanEvent:
    """An event annotation on a span.

    Attributes:
        event_id: Unique identifier for this event.
        name: Event name.
        timestamp: UTC timestamp of the event.
        attributes: Arbitrary key-value attributes.
    """

    event_id: str = ""
    name: str = ""
    timestamp: datetime = field(default_factory=_utcnow)
    attributes: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Generate event_id if not provided."""
        if not self.event_id:
            self.event_id = str(uuid.uuid4())


@dataclass
class TraceRecord:
    """Record of a single span within a distributed trace.

    Attributes:
        trace_id: Trace identifier (shared by all spans in a trace).
        span_id: Unique span identifier.
        parent_span_id: Parent span ID or empty for root spans.
        name: Span operation name.
        kind: Span kind (SERVER, CLIENT, INTERNAL, PRODUCER, CONSUMER).
        status: Span status (UNSET, OK, ERROR).
        attributes: Arbitrary key-value attributes.
        events: List of span events.
        start_time: UTC start timestamp.
        end_time: UTC end timestamp (None if still active).
        duration_ms: Computed duration in milliseconds.
        is_active: Whether this span is still in progress.
        provenance_hash: SHA-256 hash for audit trail.
    """

    trace_id: str = ""
    span_id: str = ""
    parent_span_id: str = ""
    name: str = ""
    kind: str = "INTERNAL"
    status: str = "UNSET"
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[SpanEvent] = field(default_factory=list)
    start_time: datetime = field(default_factory=_utcnow)
    end_time: Optional[datetime] = None
    duration_ms: float = 0.0
    is_active: bool = True
    provenance_hash: str = ""

    def __post_init__(self) -> None:
        """Generate span_id if not provided."""
        if not self.span_id:
            self.span_id = str(uuid.uuid4())


@dataclass
class TraceContext:
    """Propagated trace context for distributed tracing.

    Attributes:
        trace_id: Trace identifier.
        span_id: Current span identifier.
        trace_flags: W3C trace flags (01 = sampled).
        trace_state: W3C trace state entries.
        created_at: Creation timestamp.
    """

    trace_id: str = ""
    span_id: str = ""
    trace_flags: str = "01"
    trace_state: Dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_utcnow)


# Valid span kinds matching OTel specification
VALID_SPAN_KINDS: Tuple[str, ...] = (
    "SERVER", "CLIENT", "INTERNAL", "PRODUCER", "CONSUMER",
)

VALID_STATUSES: Tuple[str, ...] = ("UNSET", "OK", "ERROR")


# =============================================================================
# TraceManager
# =============================================================================


class TraceManager:
    """OTel-compatible distributed tracing engine.

    Manages span lifecycle (start, annotate, end), trace context
    propagation, and stale span cleanup. All operations are thread-safe.

    Attributes:
        _config: Observability configuration.
        _spans: Active and completed spans keyed by (trace_id, span_id).
        _traces: Index mapping trace_id to list of span_ids.
        _total_spans_created: Running count of all spans created.
        _total_spans_completed: Running count of completed spans.
        _lock: Thread lock for concurrent access.

    Example:
        >>> manager = TraceManager(config)
        >>> ctx = manager.create_trace_context()
        >>> root = manager.start_span("handle_request", trace_id=ctx.trace_id)
        >>> child = manager.start_span("db_query", trace_id=ctx.trace_id,
        ...                            parent_span_id=root.span_id)
        >>> manager.end_span(ctx.trace_id, child.span_id, status="OK")
        >>> manager.end_span(ctx.trace_id, root.span_id, status="OK")
    """

    def __init__(self, config: Any) -> None:
        """Initialize TraceManager.

        Args:
            config: Observability configuration instance. May expose
                    ``max_spans``, ``span_ttl_seconds`` attributes.
        """
        self._config = config
        self._spans: Dict[Tuple[str, str], TraceRecord] = {}
        self._traces: Dict[str, List[str]] = {}
        self._total_spans_created: int = 0
        self._total_spans_completed: int = 0
        self._lock = threading.RLock()

        self._max_spans: int = getattr(config, "max_spans", 100000)
        self._span_ttl_seconds: int = getattr(config, "span_ttl_seconds", 3600)

        logger.info(
            "TraceManager initialized: max_spans=%d, span_ttl=%ds",
            self._max_spans,
            self._span_ttl_seconds,
        )

    # ------------------------------------------------------------------
    # Trace context
    # ------------------------------------------------------------------

    def create_trace_context(
        self,
        trace_id: Optional[str] = None,
    ) -> TraceContext:
        """Generate a new trace context for distributed propagation.

        If ``trace_id`` is not provided, a new UUID-4 is generated.

        Args:
            trace_id: Optional existing trace ID to reuse.

        Returns:
            TraceContext with trace_id and initial span_id.
        """
        ctx = TraceContext(
            trace_id=trace_id or str(uuid.uuid4()),
            span_id=str(uuid.uuid4()),
            trace_flags="01",
        )
        logger.debug(
            "Created trace context: trace_id=%s, span_id=%s",
            ctx.trace_id[:8], ctx.span_id[:8],
        )
        return ctx

    # ------------------------------------------------------------------
    # Span lifecycle
    # ------------------------------------------------------------------

    def start_span(
        self,
        name: str,
        trace_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
        kind: str = "INTERNAL",
    ) -> TraceRecord:
        """Start a new span.

        Args:
            name: Operation name for the span.
            trace_id: Trace identifier (generated if not provided).
            parent_span_id: Parent span ID for child spans.
            attributes: Initial span attributes.
            kind: Span kind (SERVER, CLIENT, INTERNAL, PRODUCER, CONSUMER).

        Returns:
            TraceRecord representing the started span.

        Raises:
            ValueError: If name is empty or kind is invalid.
        """
        if not name or not name.strip():
            raise ValueError("Span name must be non-empty")

        if kind not in VALID_SPAN_KINDS:
            raise ValueError(
                f"Invalid span kind '{kind}'; must be one of {VALID_SPAN_KINDS}"
            )

        resolved_trace_id = trace_id or str(uuid.uuid4())
        now = _utcnow()

        provenance_hash = self._compute_span_hash(
            resolved_trace_id, name, now, attributes or {},
        )

        span = TraceRecord(
            trace_id=resolved_trace_id,
            parent_span_id=parent_span_id or "",
            name=name,
            kind=kind,
            status="UNSET",
            attributes=dict(attributes or {}),
            start_time=now,
            is_active=True,
            provenance_hash=provenance_hash,
        )

        with self._lock:
            if len(self._spans) >= self._max_spans:
                self._cleanup_old_spans()

                if len(self._spans) >= self._max_spans:
                    raise ValueError(
                        f"Maximum span limit ({self._max_spans}) reached after cleanup"
                    )

            self._spans[(resolved_trace_id, span.span_id)] = span

            if resolved_trace_id not in self._traces:
                self._traces[resolved_trace_id] = []
            self._traces[resolved_trace_id].append(span.span_id)

            self._total_spans_created += 1

        logger.debug(
            "Started span: trace=%s, span=%s, name=%s, kind=%s",
            resolved_trace_id[:8], span.span_id[:8], name, kind,
        )
        return span

    def end_span(
        self,
        trace_id: str,
        span_id: str,
        status: str = "OK",
        attributes: Optional[Dict[str, Any]] = None,
    ) -> TraceRecord:
        """End an active span.

        Args:
            trace_id: Trace identifier.
            span_id: Span identifier to end.
            status: Final span status (UNSET, OK, ERROR).
            attributes: Additional attributes to merge.

        Returns:
            Updated TraceRecord with end_time and duration.

        Raises:
            ValueError: If span not found, already ended, or invalid status.
        """
        if status not in VALID_STATUSES:
            raise ValueError(
                f"Invalid span status '{status}'; must be one of {VALID_STATUSES}"
            )

        key = (trace_id, span_id)

        with self._lock:
            span = self._spans.get(key)
            if span is None:
                raise ValueError(
                    f"Span not found: trace_id={trace_id[:8]}, span_id={span_id[:8]}"
                )

            if not span.is_active:
                raise ValueError(
                    f"Span already ended: trace_id={trace_id[:8]}, span_id={span_id[:8]}"
                )

            now = _utcnow()
            delta = now - span.start_time
            duration_ms = delta.total_seconds() * 1000.0

            span.end_time = now
            span.duration_ms = duration_ms
            span.status = status
            span.is_active = False

            if attributes:
                span.attributes.update(attributes)

            # Recompute provenance with final state
            span.provenance_hash = self._compute_span_hash(
                trace_id, span.name, span.start_time,
                span.attributes, span.end_time, status,
            )

            self._total_spans_completed += 1

        logger.debug(
            "Ended span: trace=%s, span=%s, status=%s, duration=%.1fms",
            trace_id[:8], span_id[:8], status, duration_ms,
        )
        return span

    def add_span_event(
        self,
        trace_id: str,
        span_id: str,
        event_name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> SpanEvent:
        """Add an event annotation to an active span.

        Args:
            trace_id: Trace identifier.
            span_id: Span identifier.
            event_name: Name of the event.
            attributes: Event attributes.

        Returns:
            SpanEvent that was added.

        Raises:
            ValueError: If span is not found or not active.
        """
        if not event_name or not event_name.strip():
            raise ValueError("Event name must be non-empty")

        key = (trace_id, span_id)

        with self._lock:
            span = self._spans.get(key)
            if span is None:
                raise ValueError(
                    f"Span not found: trace_id={trace_id[:8]}, span_id={span_id[:8]}"
                )

            if not span.is_active:
                raise ValueError(
                    f"Cannot add event to ended span: span_id={span_id[:8]}"
                )

            event = SpanEvent(
                name=event_name,
                timestamp=_utcnow(),
                attributes=dict(attributes or {}),
            )
            span.events.append(event)

        logger.debug(
            "Added event '%s' to span %s in trace %s",
            event_name, span_id[:8], trace_id[:8],
        )
        return event

    def set_span_attribute(
        self,
        trace_id: str,
        span_id: str,
        key: str,
        value: Any,
    ) -> None:
        """Set a single attribute on an active span.

        Args:
            trace_id: Trace identifier.
            span_id: Span identifier.
            key: Attribute key.
            value: Attribute value.

        Raises:
            ValueError: If span is not found or not active.
        """
        span_key = (trace_id, span_id)

        with self._lock:
            span = self._spans.get(span_key)
            if span is None:
                raise ValueError(f"Span not found: span_id={span_id[:8]}")

            if not span.is_active:
                raise ValueError(f"Cannot set attribute on ended span: span_id={span_id[:8]}")

            span.attributes[key] = value

    # ------------------------------------------------------------------
    # Query operations
    # ------------------------------------------------------------------

    def get_span(self, trace_id: str, span_id: str) -> Optional[TraceRecord]:
        """Get a specific span by trace and span ID.

        Args:
            trace_id: Trace identifier.
            span_id: Span identifier.

        Returns:
            TraceRecord or None if not found.
        """
        with self._lock:
            return self._spans.get((trace_id, span_id))

    def get_trace(self, trace_id: str) -> List[TraceRecord]:
        """Get all spans belonging to a trace, ordered by start time.

        Args:
            trace_id: Trace identifier.

        Returns:
            List of TraceRecord objects sorted by start_time.
        """
        with self._lock:
            span_ids = self._traces.get(trace_id, [])
            spans = []
            for sid in span_ids:
                span = self._spans.get((trace_id, sid))
                if span is not None:
                    spans.append(span)

        spans.sort(key=lambda s: s.start_time)
        return spans

    def get_active_spans(self) -> List[TraceRecord]:
        """Get all currently active (in-progress) spans.

        Returns:
            List of active TraceRecord objects, newest first.
        """
        with self._lock:
            active = [s for s in self._spans.values() if s.is_active]

        active.sort(key=lambda s: s.start_time, reverse=True)
        return active

    def get_child_spans(self, trace_id: str, parent_span_id: str) -> List[TraceRecord]:
        """Get all direct child spans of a parent span.

        Args:
            trace_id: Trace identifier.
            parent_span_id: Parent span identifier.

        Returns:
            List of child TraceRecord objects.
        """
        with self._lock:
            span_ids = self._traces.get(trace_id, [])
            children = []
            for sid in span_ids:
                span = self._spans.get((trace_id, sid))
                if span is not None and span.parent_span_id == parent_span_id:
                    children.append(span)

        children.sort(key=lambda s: s.start_time)
        return children

    def list_traces(self, limit: int = 100) -> List[Dict[str, Any]]:
        """List recent traces with summary information.

        Args:
            limit: Maximum number of traces to return.

        Returns:
            List of trace summary dicts with trace_id, span_count,
            root_span_name, and status.
        """
        with self._lock:
            trace_summaries: List[Dict[str, Any]] = []

            for trace_id, span_ids in self._traces.items():
                root_span: Optional[TraceRecord] = None
                total_spans = len(span_ids)
                active_count = 0

                for sid in span_ids:
                    span = self._spans.get((trace_id, sid))
                    if span is None:
                        continue
                    if not span.parent_span_id:
                        root_span = span
                    if span.is_active:
                        active_count += 1

                trace_summaries.append({
                    "trace_id": trace_id,
                    "span_count": total_spans,
                    "active_spans": active_count,
                    "root_span_name": root_span.name if root_span else "unknown",
                    "status": root_span.status if root_span else "UNSET",
                    "start_time": (
                        root_span.start_time.isoformat() if root_span else None
                    ),
                })

        trace_summaries.sort(
            key=lambda t: t.get("start_time") or "",
            reverse=True,
        )
        return trace_summaries[:limit]

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _cleanup_old_spans(self) -> int:
        """Remove expired spans that exceed the TTL.

        Must be called within the lock context.

        Returns:
            Number of spans removed.
        """
        cutoff = _utcnow().timestamp() - self._span_ttl_seconds
        removed = 0

        stale_keys: List[Tuple[str, str]] = []
        for key, span in self._spans.items():
            ref_time = span.end_time or span.start_time
            if ref_time.timestamp() < cutoff:
                stale_keys.append(key)

        for key in stale_keys:
            trace_id, span_id = key
            del self._spans[key]
            if trace_id in self._traces:
                try:
                    self._traces[trace_id].remove(span_id)
                except ValueError:
                    pass
                if not self._traces[trace_id]:
                    del self._traces[trace_id]
            removed += 1

        if removed > 0:
            logger.info("Cleaned up %d expired spans", removed)
        return removed

    def cleanup_completed_traces(self, max_age_seconds: Optional[int] = None) -> int:
        """Remove completed traces older than the specified age.

        Args:
            max_age_seconds: Maximum age in seconds (defaults to span_ttl_seconds).

        Returns:
            Number of traces removed.
        """
        age_limit = max_age_seconds or self._span_ttl_seconds
        cutoff = _utcnow().timestamp() - age_limit
        removed = 0

        with self._lock:
            traces_to_remove: List[str] = []

            for trace_id, span_ids in self._traces.items():
                all_completed = True
                oldest_end: Optional[datetime] = None

                for sid in span_ids:
                    span = self._spans.get((trace_id, sid))
                    if span is None:
                        continue
                    if span.is_active:
                        all_completed = False
                        break
                    if span.end_time is not None:
                        if oldest_end is None or span.end_time < oldest_end:
                            oldest_end = span.end_time

                if all_completed and oldest_end and oldest_end.timestamp() < cutoff:
                    traces_to_remove.append(trace_id)

            for trace_id in traces_to_remove:
                span_ids = self._traces.pop(trace_id, [])
                for sid in span_ids:
                    self._spans.pop((trace_id, sid), None)
                removed += 1

        if removed > 0:
            logger.info("Cleaned up %d completed traces", removed)
        return removed

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Get tracing statistics.

        Returns:
            Dictionary with total_spans_created, total_spans_completed,
            active_spans, total_traces, and span_store_size.
        """
        with self._lock:
            active_count = sum(1 for s in self._spans.values() if s.is_active)

            return {
                "total_spans_created": self._total_spans_created,
                "total_spans_completed": self._total_spans_completed,
                "active_spans": active_count,
                "total_traces": len(self._traces),
                "span_store_size": len(self._spans),
                "max_spans": self._max_spans,
                "span_ttl_seconds": self._span_ttl_seconds,
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_span_hash(
        self,
        trace_id: str,
        name: str,
        start_time: datetime,
        attributes: Dict[str, Any],
        end_time: Optional[datetime] = None,
        status: str = "UNSET",
    ) -> str:
        """Compute SHA-256 provenance hash for a span.

        Args:
            trace_id: Trace identifier.
            name: Span name.
            start_time: Span start timestamp.
            attributes: Span attributes.
            end_time: Span end timestamp (None for active spans).
            status: Span status.

        Returns:
            Hex-encoded SHA-256 hash.
        """
        payload = json.dumps(
            {
                "trace_id": trace_id,
                "name": name,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat() if end_time else None,
                "status": status,
                "attributes": attributes,
            },
            sort_keys=True,
            ensure_ascii=True,
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "TraceManager",
    "TraceRecord",
    "TraceContext",
    "SpanEvent",
    "VALID_SPAN_KINDS",
    "VALID_STATUSES",
]
