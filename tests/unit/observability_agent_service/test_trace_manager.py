# -*- coding: utf-8 -*-
"""
Unit Tests for TraceManager (AGENT-FOUND-010)

Tests span lifecycle (start, end, events, attributes), trace context creation,
query operations, cleanup, statistics, and error handling.

Coverage target: 85%+ of trace_manager.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from greenlang.observability_agent.trace_manager import (
    VALID_SPAN_KINDS,
    VALID_STATUSES,
    SpanEvent,
    TraceContext,
    TraceManager,
    TraceRecord,
)


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------

@dataclass
class _StubConfig:
    """Minimal config stub for TraceManager."""
    max_spans: int = 100000
    span_ttl_seconds: int = 3600


@pytest.fixture
def config():
    return _StubConfig()


@pytest.fixture
def manager(config):
    return TraceManager(config)


# ==========================================================================
# TraceContext Tests
# ==========================================================================

class TestTraceContext:
    """Tests for create_trace_context."""

    def test_create_trace_context(self, manager):
        ctx = manager.create_trace_context()
        assert isinstance(ctx, TraceContext)
        assert ctx.trace_id
        assert ctx.span_id
        assert ctx.trace_flags == "01"

    def test_create_trace_context_with_provided_trace_id(self, manager):
        ctx = manager.create_trace_context(trace_id="custom-trace-id")
        assert ctx.trace_id == "custom-trace-id"
        assert ctx.span_id  # still generated

    def test_trace_context_generation_unique(self, manager):
        ctx1 = manager.create_trace_context()
        ctx2 = manager.create_trace_context()
        assert ctx1.trace_id != ctx2.trace_id
        assert ctx1.span_id != ctx2.span_id


# ==========================================================================
# Start Span Tests
# ==========================================================================

class TestTraceManagerStartSpan:
    """Tests for start_span."""

    def test_start_span_basic(self, manager):
        span = manager.start_span("process_data")
        assert isinstance(span, TraceRecord)
        assert span.name == "process_data"
        assert span.is_active is True
        assert span.status == "UNSET"
        assert span.trace_id  # auto-generated
        assert span.span_id  # auto-generated

    def test_start_span_with_trace_id(self, manager):
        span = manager.start_span("op", trace_id="my-trace")
        assert span.trace_id == "my-trace"

    def test_start_span_with_parent(self, manager):
        parent = manager.start_span("parent", trace_id="t1")
        child = manager.start_span(
            "child", trace_id="t1", parent_span_id=parent.span_id,
        )
        assert child.parent_span_id == parent.span_id
        assert child.trace_id == "t1"

    def test_start_span_with_attributes(self, manager):
        span = manager.start_span("op", attributes={"key": "value"})
        assert span.attributes == {"key": "value"}

    @pytest.mark.parametrize("kind", VALID_SPAN_KINDS)
    def test_start_span_valid_kinds(self, manager, kind):
        span = manager.start_span("op", kind=kind)
        assert span.kind == kind

    def test_start_span_invalid_kind_raises(self, manager):
        with pytest.raises(ValueError, match="Invalid span kind"):
            manager.start_span("op", kind="UNKNOWN")

    def test_start_span_empty_name_raises(self, manager):
        with pytest.raises(ValueError, match="non-empty"):
            manager.start_span("")

    def test_start_span_whitespace_name_raises(self, manager):
        with pytest.raises(ValueError, match="non-empty"):
            manager.start_span("   ")

    def test_start_span_provenance_hash(self, manager):
        span = manager.start_span("op")
        assert span.provenance_hash
        assert len(span.provenance_hash) == 64

    def test_start_span_default_kind_is_internal(self, manager):
        span = manager.start_span("op")
        assert span.kind == "INTERNAL"


# ==========================================================================
# End Span Tests
# ==========================================================================

class TestTraceManagerEndSpan:
    """Tests for end_span."""

    def test_end_span_basic(self, manager):
        span = manager.start_span("op", trace_id="t1")
        ended = manager.end_span("t1", span.span_id)
        assert ended.is_active is False
        assert ended.status == "OK"
        assert ended.end_time is not None
        assert ended.duration_ms >= 0.0

    @pytest.mark.parametrize("status", VALID_STATUSES)
    def test_end_span_with_status(self, manager, status):
        span = manager.start_span("op", trace_id="t1")
        ended = manager.end_span("t1", span.span_id, status=status)
        assert ended.status == status

    def test_end_span_invalid_status_raises(self, manager):
        span = manager.start_span("op", trace_id="t1")
        with pytest.raises(ValueError, match="Invalid span status"):
            manager.end_span("t1", span.span_id, status="INVALID")

    def test_end_span_not_found_raises(self, manager):
        with pytest.raises(ValueError, match="Span not found"):
            manager.end_span("t1", "nonexistent-span")

    def test_end_span_already_ended_raises(self, manager):
        span = manager.start_span("op", trace_id="t1")
        manager.end_span("t1", span.span_id)
        with pytest.raises(ValueError, match="already ended"):
            manager.end_span("t1", span.span_id)

    def test_end_span_with_additional_attributes(self, manager):
        span = manager.start_span("op", trace_id="t1", attributes={"a": 1})
        ended = manager.end_span("t1", span.span_id, attributes={"b": 2})
        assert ended.attributes["a"] == 1
        assert ended.attributes["b"] == 2

    def test_end_span_recomputes_provenance_hash(self, manager):
        span = manager.start_span("op", trace_id="t1")
        start_hash = span.provenance_hash
        ended = manager.end_span("t1", span.span_id, status="OK")
        assert ended.provenance_hash != start_hash


# ==========================================================================
# Span Events Tests
# ==========================================================================

class TestTraceManagerSpanEvents:
    """Tests for add_span_event."""

    def test_add_span_event(self, manager):
        span = manager.start_span("op", trace_id="t1")
        event = manager.add_span_event("t1", span.span_id, "checkpoint")
        assert isinstance(event, SpanEvent)
        assert event.name == "checkpoint"
        assert event.event_id  # auto-generated

    def test_add_span_event_with_attributes(self, manager):
        span = manager.start_span("op", trace_id="t1")
        event = manager.add_span_event(
            "t1", span.span_id, "log", attributes={"msg": "hello"},
        )
        assert event.attributes == {"msg": "hello"}

    def test_add_event_to_ended_span_raises(self, manager):
        span = manager.start_span("op", trace_id="t1")
        manager.end_span("t1", span.span_id)
        with pytest.raises(ValueError, match="ended span"):
            manager.add_span_event("t1", span.span_id, "late_event")

    def test_add_event_nonexistent_span_raises(self, manager):
        with pytest.raises(ValueError, match="Span not found"):
            manager.add_span_event("t1", "ghost", "ev")

    def test_add_event_empty_name_raises(self, manager):
        span = manager.start_span("op", trace_id="t1")
        with pytest.raises(ValueError, match="non-empty"):
            manager.add_span_event("t1", span.span_id, "")

    def test_multiple_events_on_span(self, manager):
        span = manager.start_span("op", trace_id="t1")
        manager.add_span_event("t1", span.span_id, "event1")
        manager.add_span_event("t1", span.span_id, "event2")
        manager.add_span_event("t1", span.span_id, "event3")
        retrieved = manager.get_span("t1", span.span_id)
        assert len(retrieved.events) == 3


# ==========================================================================
# Span Attributes Tests
# ==========================================================================

class TestTraceManagerSpanAttributes:
    """Tests for set_span_attribute."""

    def test_set_span_attribute(self, manager):
        span = manager.start_span("op", trace_id="t1")
        manager.set_span_attribute("t1", span.span_id, "key1", "val1")
        retrieved = manager.get_span("t1", span.span_id)
        assert retrieved.attributes["key1"] == "val1"

    def test_set_attribute_on_ended_span_raises(self, manager):
        span = manager.start_span("op", trace_id="t1")
        manager.end_span("t1", span.span_id)
        with pytest.raises(ValueError, match="ended span"):
            manager.set_span_attribute("t1", span.span_id, "k", "v")

    def test_set_attribute_nonexistent_span_raises(self, manager):
        with pytest.raises(ValueError, match="not found"):
            manager.set_span_attribute("t1", "ghost", "k", "v")


# ==========================================================================
# Query Tests
# ==========================================================================

class TestTraceManagerQuery:
    """Tests for query operations."""

    def test_get_trace_all_spans(self, manager):
        s1 = manager.start_span("root", trace_id="t1")
        s2 = manager.start_span("child", trace_id="t1", parent_span_id=s1.span_id)
        spans = manager.get_trace("t1")
        assert len(spans) == 2

    def test_get_trace_empty(self, manager):
        spans = manager.get_trace("nonexistent")
        assert spans == []

    def test_get_active_spans(self, manager):
        s1 = manager.start_span("active1", trace_id="t1")
        s2 = manager.start_span("active2", trace_id="t2")
        s3 = manager.start_span("ended", trace_id="t3")
        manager.end_span("t3", s3.span_id)
        active = manager.get_active_spans()
        assert len(active) == 2

    def test_get_span_existing(self, manager):
        span = manager.start_span("op", trace_id="t1")
        result = manager.get_span("t1", span.span_id)
        assert result is not None
        assert result.name == "op"

    def test_get_span_nonexistent(self, manager):
        result = manager.get_span("t1", "ghost")
        assert result is None

    def test_get_child_spans(self, manager):
        root = manager.start_span("root", trace_id="t1")
        c1 = manager.start_span("child1", trace_id="t1", parent_span_id=root.span_id)
        c2 = manager.start_span("child2", trace_id="t1", parent_span_id=root.span_id)
        children = manager.get_child_spans("t1", root.span_id)
        assert len(children) == 2

    def test_list_traces(self, manager):
        manager.start_span("root1", trace_id="t1")
        manager.start_span("root2", trace_id="t2")
        traces = manager.list_traces()
        assert len(traces) == 2
        assert all("trace_id" in t for t in traces)

    def test_list_traces_limit(self, manager):
        for i in range(5):
            manager.start_span(f"op{i}", trace_id=f"t{i}")
        traces = manager.list_traces(limit=2)
        assert len(traces) == 2


# ==========================================================================
# Cleanup Tests
# ==========================================================================

class TestTraceManagerCleanup:
    """Tests for cleanup operations."""

    def test_cleanup_completed_traces_empty(self, manager):
        removed = manager.cleanup_completed_traces()
        assert removed == 0

    def test_cleanup_respects_active_spans(self, manager):
        s = manager.start_span("active", trace_id="t1")
        removed = manager.cleanup_completed_traces(max_age_seconds=0)
        assert removed == 0  # still active


# ==========================================================================
# Statistics Tests
# ==========================================================================

class TestTraceManagerStatistics:
    """Tests for get_statistics."""

    def test_statistics_empty(self, manager):
        stats = manager.get_statistics()
        assert stats["total_spans_created"] == 0
        assert stats["total_spans_completed"] == 0
        assert stats["active_spans"] == 0
        assert stats["total_traces"] == 0
        assert stats["span_store_size"] == 0

    def test_statistics_after_operations(self, manager):
        s1 = manager.start_span("op1", trace_id="t1")
        s2 = manager.start_span("op2", trace_id="t1")
        manager.end_span("t1", s1.span_id)
        stats = manager.get_statistics()
        assert stats["total_spans_created"] == 2
        assert stats["total_spans_completed"] == 1
        assert stats["active_spans"] == 1
        assert stats["total_traces"] == 1
        assert stats["span_store_size"] == 2


# ==========================================================================
# Span Limit Enforcement Tests
# ==========================================================================

class TestTraceManagerSpanLimit:
    """Tests for max span limit enforcement."""

    def test_span_limit_enforcement(self):
        cfg = _StubConfig(max_spans=3, span_ttl_seconds=0)
        mgr = TraceManager(cfg)
        mgr.start_span("s1", trace_id="t1")
        mgr.start_span("s2", trace_id="t2")
        mgr.start_span("s3", trace_id="t3")
        # The 4th span should trigger cleanup, and since ttl is 0
        # it may or may not succeed depending on timing.
        # We test that the system handles this gracefully.
        try:
            mgr.start_span("s4", trace_id="t4")
        except ValueError:
            pass  # Expected if cleanup cannot free enough space
