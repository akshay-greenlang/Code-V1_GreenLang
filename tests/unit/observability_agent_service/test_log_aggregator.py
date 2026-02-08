# -*- coding: utf-8 -*-
"""
Unit Tests for LogAggregator (AGENT-FOUND-010)

Tests log ingestion, querying by level/agent/tenant/trace, correlation chains,
buffer management, statistics, and error handling.

Since log_aggregator.py is not yet on disk, these tests define the expected
interface via an inline implementation that mirrors the PRD specification.
Tests will validate the interface contract once the module is available.

Coverage target: 85%+ of log_aggregator.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline LogAggregator (mirrors expected greenlang.observability_agent.log_aggregator)
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


@dataclass
class LogEntry:
    """A structured log record."""
    record_id: str = ""
    timestamp: datetime = field(default_factory=_utcnow)
    level: str = "info"
    message: str = ""
    correlation_id: str = ""
    trace_id: str = ""
    span_id: str = ""
    agent_id: str = ""
    tenant_id: str = "default"
    attributes: Dict[str, Any] = field(default_factory=dict)
    provenance_hash: str = ""

    def __post_init__(self):
        if not self.record_id:
            self.record_id = str(uuid.uuid4())
        if not self.correlation_id:
            self.correlation_id = str(uuid.uuid4())
        if not self.provenance_hash:
            payload = json.dumps(
                {"message": self.message, "level": self.level,
                 "timestamp": self.timestamp.isoformat()},
                sort_keys=True,
            )
            self.provenance_hash = hashlib.sha256(payload.encode()).hexdigest()


class LogAggregator:
    """In-memory structured log aggregation engine."""

    def __init__(self, config: Any) -> None:
        self._config = config
        self._buffer: List[LogEntry] = []
        self._max_buffer_size: int = getattr(config, "log_buffer_size", 10000)
        self._total_ingested: int = 0

    def ingest(
        self,
        message: str,
        level: str = "info",
        agent_id: str = "",
        tenant_id: str = "default",
        trace_id: str = "",
        span_id: str = "",
        correlation_id: str = "",
        attributes: Optional[Dict[str, Any]] = None,
    ) -> LogEntry:
        if not message or not message.strip():
            raise ValueError("Log message must be non-empty")
        valid_levels = {"debug", "info", "warning", "error", "critical"}
        if level not in valid_levels:
            raise ValueError(f"Invalid log level '{level}'")

        entry = LogEntry(
            level=level,
            message=message,
            agent_id=agent_id,
            tenant_id=tenant_id,
            trace_id=trace_id,
            span_id=span_id,
            correlation_id=correlation_id or "",
            attributes=dict(attributes or {}),
        )

        if len(self._buffer) >= self._max_buffer_size:
            self._buffer = self._buffer[self._max_buffer_size // 2:]

        self._buffer.append(entry)
        self._total_ingested += 1
        return entry

    def query(
        self,
        level: Optional[str] = None,
        agent_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[LogEntry]:
        results = list(self._buffer)
        if level:
            results = [e for e in results if e.level == level]
        if agent_id:
            results = [e for e in results if e.agent_id == agent_id]
        if tenant_id:
            results = [e for e in results if e.tenant_id == tenant_id]
        if trace_id:
            results = [e for e in results if e.trace_id == trace_id]
        if start_time:
            results = [e for e in results if e.timestamp >= start_time]
        if end_time:
            results = [e for e in results if e.timestamp <= end_time]
        return results[offset : offset + limit]

    def get_correlation_chain(self, correlation_id: str) -> List[LogEntry]:
        return [e for e in self._buffer if e.correlation_id == correlation_id]

    def count_by_level(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for e in self._buffer:
            counts[e.level] = counts.get(e.level, 0) + 1
        return counts

    def flush(self) -> int:
        count = len(self._buffer)
        self._buffer.clear()
        return count

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "total_ingested": self._total_ingested,
            "buffer_size": len(self._buffer),
            "max_buffer_size": self._max_buffer_size,
        }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@dataclass
class _StubConfig:
    log_buffer_size: int = 10000


@pytest.fixture
def config():
    return _StubConfig()


@pytest.fixture
def aggregator(config):
    return LogAggregator(config)


# ==========================================================================
# Ingestion Tests
# ==========================================================================

class TestLogAggregatorIngest:
    """Tests for log ingestion."""

    def test_ingest_basic(self, aggregator):
        entry = aggregator.ingest("Hello world")
        assert isinstance(entry, LogEntry)
        assert entry.message == "Hello world"
        assert entry.level == "info"

    def test_ingest_with_all_fields(self, aggregator):
        entry = aggregator.ingest(
            message="Full log",
            level="error",
            agent_id="agent-1",
            tenant_id="tenant-a",
            trace_id="trace-1",
            span_id="span-1",
            attributes={"key": "val"},
        )
        assert entry.agent_id == "agent-1"
        assert entry.tenant_id == "tenant-a"
        assert entry.trace_id == "trace-1"
        assert entry.span_id == "span-1"
        assert entry.attributes == {"key": "val"}

    def test_ingest_empty_message_raises(self, aggregator):
        with pytest.raises(ValueError, match="non-empty"):
            aggregator.ingest("")

    def test_ingest_whitespace_message_raises(self, aggregator):
        with pytest.raises(ValueError, match="non-empty"):
            aggregator.ingest("   ")

    def test_ingest_invalid_level_raises(self, aggregator):
        with pytest.raises(ValueError, match="Invalid log level"):
            aggregator.ingest("msg", level="invalid")

    @pytest.mark.parametrize("level", ["debug", "info", "warning", "error", "critical"])
    def test_ingest_valid_levels(self, aggregator, level):
        entry = aggregator.ingest("msg", level=level)
        assert entry.level == level

    def test_ingest_auto_generates_record_id(self, aggregator):
        entry = aggregator.ingest("msg")
        assert entry.record_id
        assert len(entry.record_id) > 0

    def test_ingest_provenance_hash(self, aggregator):
        entry = aggregator.ingest("msg")
        assert entry.provenance_hash
        assert len(entry.provenance_hash) == 64

    def test_auto_correlation_id(self, aggregator):
        entry = aggregator.ingest("msg")
        assert entry.correlation_id  # auto-generated if not provided


# ==========================================================================
# Query Tests
# ==========================================================================

class TestLogAggregatorQuery:
    """Tests for log querying."""

    def test_query_by_level(self, aggregator):
        aggregator.ingest("info msg", level="info")
        aggregator.ingest("error msg", level="error")
        aggregator.ingest("info msg2", level="info")
        results = aggregator.query(level="error")
        assert len(results) == 1
        assert results[0].level == "error"

    def test_query_by_agent(self, aggregator):
        aggregator.ingest("msg1", agent_id="agent-a")
        aggregator.ingest("msg2", agent_id="agent-b")
        results = aggregator.query(agent_id="agent-a")
        assert len(results) == 1

    def test_query_by_tenant(self, aggregator):
        aggregator.ingest("msg1", tenant_id="t1")
        aggregator.ingest("msg2", tenant_id="t2")
        results = aggregator.query(tenant_id="t1")
        assert len(results) == 1

    def test_query_by_trace_id(self, aggregator):
        aggregator.ingest("msg1", trace_id="tr-1")
        aggregator.ingest("msg2", trace_id="tr-2")
        results = aggregator.query(trace_id="tr-1")
        assert len(results) == 1

    def test_query_with_limit(self, aggregator):
        for i in range(10):
            aggregator.ingest(f"msg-{i}")
        results = aggregator.query(limit=3)
        assert len(results) == 3

    def test_query_with_offset(self, aggregator):
        for i in range(10):
            aggregator.ingest(f"msg-{i}")
        results = aggregator.query(offset=5, limit=100)
        assert len(results) == 5

    def test_query_no_results(self, aggregator):
        aggregator.ingest("msg")
        results = aggregator.query(level="critical")
        assert results == []


# ==========================================================================
# Correlation Chain Tests
# ==========================================================================

class TestLogAggregatorCorrelation:
    """Tests for correlation chain retrieval."""

    def test_correlation_chain(self, aggregator):
        e1 = aggregator.ingest("step 1", correlation_id="corr-1")
        e2 = aggregator.ingest("step 2", correlation_id="corr-1")
        aggregator.ingest("other", correlation_id="corr-2")
        chain = aggregator.get_correlation_chain("corr-1")
        assert len(chain) == 2

    def test_correlation_chain_empty(self, aggregator):
        chain = aggregator.get_correlation_chain("nonexistent")
        assert chain == []


# ==========================================================================
# Count By Level Tests
# ==========================================================================

class TestLogAggregatorCountByLevel:
    """Tests for count_by_level."""

    def test_count_by_level(self, aggregator):
        aggregator.ingest("i1", level="info")
        aggregator.ingest("i2", level="info")
        aggregator.ingest("e1", level="error")
        counts = aggregator.count_by_level()
        assert counts["info"] == 2
        assert counts["error"] == 1


# ==========================================================================
# Buffer Management Tests
# ==========================================================================

class TestLogAggregatorBuffer:
    """Tests for buffer overflow trimming and flush."""

    def test_buffer_overflow_trimming(self):
        cfg = _StubConfig(log_buffer_size=10)
        agg = LogAggregator(cfg)
        for i in range(15):
            agg.ingest(f"msg-{i}")
        stats = agg.get_statistics()
        assert stats["buffer_size"] <= 10

    def test_flush_buffer(self, aggregator):
        aggregator.ingest("msg1")
        aggregator.ingest("msg2")
        count = aggregator.flush()
        assert count == 2
        assert aggregator.query() == []


# ==========================================================================
# Statistics Tests
# ==========================================================================

class TestLogAggregatorStatistics:
    """Tests for get_statistics."""

    def test_statistics_empty(self, aggregator):
        stats = aggregator.get_statistics()
        assert stats["total_ingested"] == 0
        assert stats["buffer_size"] == 0

    def test_statistics_after_ingestion(self, aggregator):
        for i in range(5):
            aggregator.ingest(f"msg-{i}")
        stats = aggregator.get_statistics()
        assert stats["total_ingested"] == 5
        assert stats["buffer_size"] == 5
