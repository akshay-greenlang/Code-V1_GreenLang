# -*- coding: utf-8 -*-
"""
Structured Log Aggregation Engine - AGENT-FOUND-010: Observability & Telemetry Agent

Provides structured log collection, in-memory buffering, correlation chain
tracking, and multi-dimensional query capabilities. All log records include
SHA-256 provenance hashes for tamper-evident audit trails.

Zero-Hallucination Guarantees:
    - All log records are timestamped with deterministic UTC
    - Provenance hashes use SHA-256 for tamper-evidence
    - Query filtering uses exact match and range comparisons
    - No probabilistic log classification or scoring

Example:
    >>> from greenlang.observability_agent.log_aggregator import LogAggregator
    >>> from greenlang.observability_agent.config import ObservabilityConfig
    >>> aggregator = LogAggregator(ObservabilityConfig())
    >>> record = aggregator.ingest("INFO", "Processing started", agent_id="emissions-calc")
    >>> results = aggregator.query(level_filter="INFO", limit=10)

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
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_LOG_LEVELS: Tuple[str, ...] = (
    "TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL",
)

LOG_LEVEL_SEVERITY: Dict[str, int] = {
    "TRACE": 0,
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50,
}


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class LogRecord:
    """Structured log record with correlation and provenance.

    Attributes:
        record_id: Unique identifier for this log record.
        level: Log level (TRACE, DEBUG, INFO, WARNING, ERROR, CRITICAL).
        message: Log message body.
        agent_id: ID of the agent that produced this log.
        tenant_id: Tenant identifier for multi-tenant isolation.
        trace_id: Distributed trace ID for correlation.
        span_id: Span ID within the trace.
        correlation_id: Application-level correlation identifier.
        attributes: Arbitrary structured attributes.
        timestamp: UTC timestamp of the log record.
        provenance_hash: SHA-256 hash for tamper evidence.
    """

    record_id: str = ""
    level: str = "INFO"
    message: str = ""
    agent_id: str = ""
    tenant_id: str = ""
    trace_id: str = ""
    span_id: str = ""
    correlation_id: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=_utcnow)
    provenance_hash: str = ""

    def __post_init__(self) -> None:
        """Generate record_id if not provided."""
        if not self.record_id:
            self.record_id = str(uuid.uuid4())


# =============================================================================
# LogAggregator
# =============================================================================


class LogAggregator:
    """Structured log aggregation engine.

    Collects, buffers, and queries structured log records with support for
    multi-dimensional filtering, correlation chain tracking, and provenance
    hashing for audit compliance.

    Thread-safe via a reentrant lock on all mutating operations.

    Attributes:
        _config: Observability configuration.
        _buffer: Bounded in-memory deque of log records.
        _buffer_size: Maximum buffer capacity.
        _total_ingested: Running count of all ingested records.
        _level_counts: Per-level ingestion counters.
        _correlation_index: Index mapping correlation_id to record_ids.
        _trace_index: Index mapping trace_id to record_ids.
        _lock: Thread lock for concurrent access.

    Example:
        >>> agg = LogAggregator(config)
        >>> agg.ingest("ERROR", "Calculation failed", agent_id="calc-agent")
        >>> errors = agg.query(level_filter="ERROR")
        >>> print(len(errors))
    """

    def __init__(self, config: Any) -> None:
        """Initialize LogAggregator.

        Args:
            config: Observability configuration. May expose ``log_buffer_size``,
                    ``log_retention_seconds`` attributes.
        """
        self._config = config
        self._buffer_size: int = getattr(config, "log_buffer_size", 100000)
        self._retention_seconds: int = getattr(config, "log_retention_seconds", 86400)

        self._buffer: Deque[LogRecord] = deque(maxlen=self._buffer_size)
        self._records_by_id: Dict[str, LogRecord] = {}
        self._total_ingested: int = 0
        self._level_counts: Dict[str, int] = {level: 0 for level in VALID_LOG_LEVELS}
        self._correlation_index: Dict[str, List[str]] = {}
        self._trace_index: Dict[str, List[str]] = {}
        self._lock = threading.RLock()

        logger.info(
            "LogAggregator initialized: buffer_size=%d, retention=%ds",
            self._buffer_size,
            self._retention_seconds,
        )

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(
        self,
        level: str,
        message: str,
        agent_id: str = "",
        tenant_id: str = "",
        trace_id: str = "",
        span_id: str = "",
        correlation_id: str = "",
        **attributes: Any,
    ) -> LogRecord:
        """Ingest a structured log record.

        Args:
            level: Log level (TRACE, DEBUG, INFO, WARNING, ERROR, CRITICAL).
            message: Log message body.
            agent_id: Producing agent identifier.
            tenant_id: Tenant identifier.
            trace_id: Distributed trace ID.
            span_id: Span ID within the trace.
            correlation_id: Application-level correlation ID.
            **attributes: Arbitrary structured attributes.

        Returns:
            LogRecord with provenance hash.

        Raises:
            ValueError: If level is invalid or message is empty.
        """
        normalized_level = level.upper()
        if normalized_level not in VALID_LOG_LEVELS:
            raise ValueError(
                f"Invalid log level '{level}'; must be one of {VALID_LOG_LEVELS}"
            )

        if not message or not message.strip():
            raise ValueError("Log message must be non-empty")

        now = _utcnow()
        provenance_hash = self._compute_log_hash(
            normalized_level, message, agent_id, tenant_id,
            trace_id, correlation_id, now,
        )

        record = LogRecord(
            level=normalized_level,
            message=message,
            agent_id=agent_id,
            tenant_id=tenant_id,
            trace_id=trace_id,
            span_id=span_id,
            correlation_id=correlation_id,
            attributes=dict(attributes),
            timestamp=now,
            provenance_hash=provenance_hash,
        )

        with self._lock:
            # If buffer is full, the oldest record will be evicted by deque
            if len(self._buffer) >= self._buffer_size:
                evicted = self._buffer[0]
                self._evict_record(evicted)

            self._buffer.append(record)
            self._records_by_id[record.record_id] = record
            self._total_ingested += 1
            self._level_counts[normalized_level] = (
                self._level_counts.get(normalized_level, 0) + 1
            )

            # Update indexes
            if correlation_id:
                if correlation_id not in self._correlation_index:
                    self._correlation_index[correlation_id] = []
                self._correlation_index[correlation_id].append(record.record_id)

            if trace_id:
                if trace_id not in self._trace_index:
                    self._trace_index[trace_id] = []
                self._trace_index[trace_id].append(record.record_id)

        logger.debug(
            "Ingested log: level=%s, agent=%s, hash=%s",
            normalized_level, agent_id, provenance_hash[:12],
        )
        return record

    # ------------------------------------------------------------------
    # Query operations
    # ------------------------------------------------------------------

    def query(
        self,
        level_filter: Optional[str] = None,
        agent_filter: Optional[str] = None,
        tenant_filter: Optional[str] = None,
        trace_id: Optional[str] = None,
        time_from: Optional[datetime] = None,
        time_to: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[LogRecord]:
        """Query log records with multi-dimensional filtering.

        All filters are optional and combined with AND logic.

        Args:
            level_filter: Filter by log level (exact match or minimum severity).
            agent_filter: Filter by agent_id (exact match).
            tenant_filter: Filter by tenant_id (exact match).
            trace_id: Filter by trace_id (exact match).
            time_from: Include records at or after this timestamp.
            time_to: Include records at or before this timestamp.
            limit: Maximum records to return.
            offset: Number of matching records to skip.

        Returns:
            List of matching LogRecord objects, newest first.
        """
        with self._lock:
            candidates = list(self._buffer)

        # Apply filters
        results = self._apply_filters(
            candidates, level_filter, agent_filter, tenant_filter,
            trace_id, time_from, time_to,
        )

        # Sort newest first
        results.sort(key=lambda r: r.timestamp, reverse=True)

        # Apply pagination
        return results[offset: offset + limit]

    def get_record(self, record_id: str) -> Optional[LogRecord]:
        """Get a specific log record by ID.

        Args:
            record_id: Unique record identifier.

        Returns:
            LogRecord or None if not found.
        """
        with self._lock:
            return self._records_by_id.get(record_id)

    def get_log_count(self, level_filter: Optional[str] = None) -> int:
        """Get the count of ingested log records.

        Args:
            level_filter: If provided, count only records at this level.

        Returns:
            Number of matching records.
        """
        with self._lock:
            if level_filter:
                normalized = level_filter.upper()
                return self._level_counts.get(normalized, 0)
            return len(self._buffer)

    def get_correlation_chain(self, correlation_id: str) -> List[LogRecord]:
        """Get all log records sharing the same correlation ID, ordered by time.

        Args:
            correlation_id: Application-level correlation identifier.

        Returns:
            List of LogRecord objects sorted by timestamp ascending.
        """
        with self._lock:
            record_ids = self._correlation_index.get(correlation_id, [])
            records = []
            for rid in record_ids:
                rec = self._records_by_id.get(rid)
                if rec is not None:
                    records.append(rec)

        records.sort(key=lambda r: r.timestamp)
        return records

    def get_trace_logs(self, trace_id: str) -> List[LogRecord]:
        """Get all log records belonging to a distributed trace.

        Args:
            trace_id: Distributed trace identifier.

        Returns:
            List of LogRecord objects sorted by timestamp ascending.
        """
        with self._lock:
            record_ids = self._trace_index.get(trace_id, [])
            records = []
            for rid in record_ids:
                rec = self._records_by_id.get(rid)
                if rec is not None:
                    records.append(rec)

        records.sort(key=lambda r: r.timestamp)
        return records

    def get_level_distribution(self) -> Dict[str, int]:
        """Get the distribution of log records by level.

        Returns:
            Dictionary mapping log level to count.
        """
        with self._lock:
            return dict(self._level_counts)

    def search_message(
        self,
        pattern: str,
        limit: int = 100,
    ) -> List[LogRecord]:
        """Search log records by message substring match.

        Args:
            pattern: Substring to search for (case-insensitive).
            limit: Maximum records to return.

        Returns:
            List of matching LogRecord objects, newest first.
        """
        pattern_lower = pattern.lower()

        with self._lock:
            matches = [
                rec for rec in self._buffer
                if pattern_lower in rec.message.lower()
            ]

        matches.sort(key=lambda r: r.timestamp, reverse=True)
        return matches[:limit]

    # ------------------------------------------------------------------
    # Buffer management
    # ------------------------------------------------------------------

    def flush_buffer(self) -> int:
        """Flush the in-memory log buffer, removing all records.

        Returns:
            Number of records flushed.
        """
        with self._lock:
            count = len(self._buffer)
            self._buffer.clear()
            self._records_by_id.clear()
            self._correlation_index.clear()
            self._trace_index.clear()

        logger.info("Flushed log buffer: %d records removed", count)
        return count

    def trim_to_retention(self) -> int:
        """Remove records older than the retention window.

        Returns:
            Number of records trimmed.
        """
        cutoff = _utcnow().timestamp() - self._retention_seconds
        removed = 0

        with self._lock:
            while self._buffer and self._buffer[0].timestamp.timestamp() < cutoff:
                evicted = self._buffer.popleft()
                self._evict_record(evicted)
                removed += 1

        if removed > 0:
            logger.info("Trimmed %d expired log records", removed)
        return removed

    def get_buffer_utilization(self) -> Dict[str, Any]:
        """Get buffer utilization information.

        Returns:
            Dictionary with current_size, max_size, utilization_pct.
        """
        with self._lock:
            current = len(self._buffer)
            return {
                "current_size": current,
                "max_size": self._buffer_size,
                "utilization_pct": (current / self._buffer_size * 100) if self._buffer_size else 0.0,
            }

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregator statistics.

        Returns:
            Dictionary with total_ingested, buffer_size, current_buffer_count,
            level_counts, correlation_chains, and trace_count.
        """
        with self._lock:
            return {
                "total_ingested": self._total_ingested,
                "buffer_max_size": self._buffer_size,
                "current_buffer_count": len(self._buffer),
                "retention_seconds": self._retention_seconds,
                "level_counts": dict(self._level_counts),
                "correlation_chains": len(self._correlation_index),
                "trace_count": len(self._trace_index),
                "records_indexed": len(self._records_by_id),
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_filters(
        self,
        records: List[LogRecord],
        level_filter: Optional[str],
        agent_filter: Optional[str],
        tenant_filter: Optional[str],
        trace_id: Optional[str],
        time_from: Optional[datetime],
        time_to: Optional[datetime],
    ) -> List[LogRecord]:
        """Apply multi-dimensional filters to a list of log records.

        Args:
            records: Input records to filter.
            level_filter: Log level filter.
            agent_filter: Agent ID filter.
            tenant_filter: Tenant ID filter.
            trace_id: Trace ID filter.
            time_from: Start time filter.
            time_to: End time filter.

        Returns:
            Filtered list of LogRecord objects.
        """
        result = records

        if level_filter:
            normalized = level_filter.upper()
            min_severity = LOG_LEVEL_SEVERITY.get(normalized, 0)
            result = [
                r for r in result
                if LOG_LEVEL_SEVERITY.get(r.level, 0) >= min_severity
            ]

        if agent_filter:
            result = [r for r in result if r.agent_id == agent_filter]

        if tenant_filter:
            result = [r for r in result if r.tenant_id == tenant_filter]

        if trace_id:
            result = [r for r in result if r.trace_id == trace_id]

        if time_from:
            result = [r for r in result if r.timestamp >= time_from]

        if time_to:
            result = [r for r in result if r.timestamp <= time_to]

        return result

    def _evict_record(self, record: LogRecord) -> None:
        """Remove a record from all indexes.

        Must be called within the lock context.

        Args:
            record: LogRecord to evict.
        """
        self._records_by_id.pop(record.record_id, None)

        if record.correlation_id and record.correlation_id in self._correlation_index:
            try:
                self._correlation_index[record.correlation_id].remove(record.record_id)
            except ValueError:
                pass
            if not self._correlation_index[record.correlation_id]:
                del self._correlation_index[record.correlation_id]

        if record.trace_id and record.trace_id in self._trace_index:
            try:
                self._trace_index[record.trace_id].remove(record.record_id)
            except ValueError:
                pass
            if not self._trace_index[record.trace_id]:
                del self._trace_index[record.trace_id]

    def _compute_log_hash(
        self,
        level: str,
        message: str,
        agent_id: str,
        tenant_id: str,
        trace_id: str,
        correlation_id: str,
        timestamp: datetime,
    ) -> str:
        """Compute SHA-256 provenance hash for a log record.

        Args:
            level: Log level.
            message: Log message.
            agent_id: Agent identifier.
            tenant_id: Tenant identifier.
            trace_id: Trace identifier.
            correlation_id: Correlation identifier.
            timestamp: Record timestamp.

        Returns:
            Hex-encoded SHA-256 hash.
        """
        payload = json.dumps(
            {
                "level": level,
                "message": message,
                "agent_id": agent_id,
                "tenant_id": tenant_id,
                "trace_id": trace_id,
                "correlation_id": correlation_id,
                "timestamp": timestamp.isoformat(),
            },
            sort_keys=True,
            ensure_ascii=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "LogAggregator",
    "LogRecord",
    "VALID_LOG_LEVELS",
    "LOG_LEVEL_SEVERITY",
]
