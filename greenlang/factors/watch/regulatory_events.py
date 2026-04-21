# -*- coding: utf-8 -*-
"""Regulatory change event stream.

A ``RegulatoryChangeEvent`` is the first-class record emitted when the
watch pipeline detects that an upstream regulatory source has changed in
a way that matters for downstream consumers (customers pinning factors,
approval gate reviewers, methodology leads).

This module is deliberately storage-light: the event stream lives in a
single SQLite table (or PostgreSQL via the JSONB-friendly schema below)
and is the durable source of truth that the customer webhook dispatcher,
status API, and release orchestrator all read from.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event schema
# ---------------------------------------------------------------------------


class RegulatoryEventKind:
    """Well-known regulatory-change event kinds.

    The values double as webhook event names so downstream routing stays
    stable even when the producer changes.
    """

    SOURCE_ARTIFACT_CHANGED = "source.artifact_changed"
    SOURCE_UNAVAILABLE = "source.unavailable"
    FACTOR_ADDED = "factor.added"
    FACTOR_UPDATED = "factor.updated"
    FACTOR_REMOVED = "factor.removed"
    FACTOR_DEPRECATED = "factor.deprecated"
    LICENSE_CHANGED = "license.changed"
    METHODOLOGY_CHANGED = "methodology.changed"
    BREAKING_CHANGE = "source.breaking_change"

    ALL = (
        SOURCE_ARTIFACT_CHANGED,
        SOURCE_UNAVAILABLE,
        FACTOR_ADDED,
        FACTOR_UPDATED,
        FACTOR_REMOVED,
        FACTOR_DEPRECATED,
        LICENSE_CHANGED,
        METHODOLOGY_CHANGED,
        BREAKING_CHANGE,
    )


@dataclass
class RegulatoryChangeEvent:
    """A single regulatory change detected by the watch pipeline."""

    event_id: str
    source_id: str
    event_kind: str
    detected_at: str
    severity: str = "info"                     # info | warning | breaking
    factor_id: Optional[str] = None
    old_value: Optional[float] = None
    new_value: Optional[float] = None
    unit: Optional[str] = None
    artifact_hash_old: Optional[str] = None
    artifact_hash_new: Optional[str] = None
    url: Optional[str] = None
    requires_human_review: bool = False
    review_reason: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def new_id() -> str:
        return f"rce_{uuid.uuid4().hex[:20]}"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_webhook_payload(self) -> Dict[str, Any]:
        """Shape exposed to customer webhooks.

        Intentionally a subset of the raw row — the SQLite ``payload`` blob
        may carry implementation detail (artifact URLs, parser metadata)
        that customers should not see.
        """
        return {
            "event_id": self.event_id,
            "event_type": self.event_kind,
            "source_id": self.source_id,
            "detected_at": self.detected_at,
            "severity": self.severity,
            "factor_id": self.factor_id,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "unit": self.unit,
            "requires_human_review": self.requires_human_review,
            "review_reason": self.review_reason,
        }


# ---------------------------------------------------------------------------
# SQLite store
# ---------------------------------------------------------------------------


_SCHEMA = """
CREATE TABLE IF NOT EXISTS factor_regulatory_events (
    event_id                TEXT PRIMARY KEY,
    source_id               TEXT NOT NULL,
    event_kind              TEXT NOT NULL,
    detected_at             TEXT NOT NULL,
    severity                TEXT NOT NULL DEFAULT 'info',
    factor_id               TEXT,
    old_value               REAL,
    new_value               REAL,
    unit                    TEXT,
    artifact_hash_old       TEXT,
    artifact_hash_new       TEXT,
    url                     TEXT,
    requires_human_review   INTEGER NOT NULL DEFAULT 0,
    review_reason           TEXT,
    payload_json            TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_rce_source
    ON factor_regulatory_events (source_id, detected_at DESC);
CREATE INDEX IF NOT EXISTS idx_rce_kind
    ON factor_regulatory_events (event_kind, detected_at DESC);
CREATE INDEX IF NOT EXISTS idx_rce_factor
    ON factor_regulatory_events (factor_id, detected_at DESC);
"""


class RegulatoryEventStore:
    """Thread-safe SQLite-backed event log.

    Callers that want PostgreSQL persistence can subclass and override
    ``append`` / ``list_events`` — the pipeline only depends on this
    surface.
    """

    def __init__(self, sqlite_path: Union[str, Path]) -> None:
        self.sqlite_path = Path(sqlite_path)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(self.sqlite_path),
            isolation_level=None,
            check_same_thread=False,
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)

    # -- writes ------------------------------------------------------------

    def append(self, event: RegulatoryChangeEvent) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO factor_regulatory_events (
                    event_id, source_id, event_kind, detected_at, severity,
                    factor_id, old_value, new_value, unit,
                    artifact_hash_old, artifact_hash_new, url,
                    requires_human_review, review_reason, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.source_id,
                    event.event_kind,
                    event.detected_at,
                    event.severity,
                    event.factor_id,
                    event.old_value,
                    event.new_value,
                    event.unit,
                    event.artifact_hash_old,
                    event.artifact_hash_new,
                    event.url,
                    int(event.requires_human_review),
                    event.review_reason,
                    json.dumps(event.payload, default=str),
                ),
            )

    def append_many(self, events: Iterable[RegulatoryChangeEvent]) -> int:
        count = 0
        for ev in events:
            self.append(ev)
            count += 1
        return count

    # -- reads -------------------------------------------------------------

    def list_events(
        self,
        *,
        source_id: Optional[str] = None,
        event_kind: Optional[str] = None,
        factor_id: Optional[str] = None,
        since: Optional[str] = None,
        limit: int = 100,
    ) -> List[RegulatoryChangeEvent]:
        clauses: List[str] = []
        params: List[Any] = []
        if source_id:
            clauses.append("source_id = ?")
            params.append(source_id)
        if event_kind:
            clauses.append("event_kind = ?")
            params.append(event_kind)
        if factor_id:
            clauses.append("factor_id = ?")
            params.append(factor_id)
        if since:
            clauses.append("detected_at >= ?")
            params.append(since)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(int(limit))
        sql = f"""
            SELECT event_id, source_id, event_kind, detected_at, severity,
                   factor_id, old_value, new_value, unit,
                   artifact_hash_old, artifact_hash_new, url,
                   requires_human_review, review_reason, payload_json
            FROM factor_regulatory_events
            {where}
            ORDER BY detected_at DESC
            LIMIT ?
        """
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        out: List[RegulatoryChangeEvent] = []
        for r in rows:
            try:
                payload = json.loads(r[14]) if r[14] else {}
            except (TypeError, ValueError):
                payload = {}
            out.append(
                RegulatoryChangeEvent(
                    event_id=r[0],
                    source_id=r[1],
                    event_kind=r[2],
                    detected_at=r[3],
                    severity=r[4],
                    factor_id=r[5],
                    old_value=r[6],
                    new_value=r[7],
                    unit=r[8],
                    artifact_hash_old=r[9],
                    artifact_hash_new=r[10],
                    url=r[11],
                    requires_human_review=bool(r[12]),
                    review_reason=r[13],
                    payload=payload,
                )
            )
        return out

    def count(self, *, source_id: Optional[str] = None) -> int:
        sql = "SELECT COUNT(*) FROM factor_regulatory_events"
        params: List[Any] = []
        if source_id:
            sql += " WHERE source_id = ?"
            params.append(source_id)
        with self._lock:
            row = self._conn.execute(sql, params).fetchone()
        return int(row[0]) if row else 0

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# ---------------------------------------------------------------------------
# Convenience builders
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_artifact_change_event(
    *,
    source_id: str,
    artifact_hash_old: Optional[str],
    artifact_hash_new: Optional[str],
    url: Optional[str],
    requires_human_review: bool = False,
    review_reason: Optional[str] = None,
) -> RegulatoryChangeEvent:
    severity = "warning" if requires_human_review else "info"
    return RegulatoryChangeEvent(
        event_id=RegulatoryChangeEvent.new_id(),
        source_id=source_id,
        event_kind=RegulatoryEventKind.SOURCE_ARTIFACT_CHANGED,
        detected_at=_now_iso(),
        severity=severity,
        artifact_hash_old=artifact_hash_old,
        artifact_hash_new=artifact_hash_new,
        url=url,
        requires_human_review=requires_human_review,
        review_reason=review_reason,
    )


def build_source_unavailable_event(
    *,
    source_id: str,
    url: Optional[str],
    error: str,
) -> RegulatoryChangeEvent:
    return RegulatoryChangeEvent(
        event_id=RegulatoryChangeEvent.new_id(),
        source_id=source_id,
        event_kind=RegulatoryEventKind.SOURCE_UNAVAILABLE,
        detected_at=_now_iso(),
        severity="warning",
        url=url,
        requires_human_review=True,
        review_reason=f"source_unreachable:{error[:200]}",
    )


def build_factor_event(
    *,
    source_id: str,
    kind: str,
    factor_id: str,
    old_value: Optional[float] = None,
    new_value: Optional[float] = None,
    unit: Optional[str] = None,
    severity: str = "info",
    payload: Optional[Dict[str, Any]] = None,
) -> RegulatoryChangeEvent:
    if kind not in RegulatoryEventKind.ALL:
        raise ValueError(f"unknown event_kind: {kind}")
    return RegulatoryChangeEvent(
        event_id=RegulatoryChangeEvent.new_id(),
        source_id=source_id,
        event_kind=kind,
        detected_at=_now_iso(),
        severity=severity,
        factor_id=factor_id,
        old_value=old_value,
        new_value=new_value,
        unit=unit,
        payload=payload or {},
    )


__all__ = [
    "RegulatoryEventKind",
    "RegulatoryChangeEvent",
    "RegulatoryEventStore",
    "build_artifact_change_event",
    "build_source_unavailable_event",
    "build_factor_event",
]
