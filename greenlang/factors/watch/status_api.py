# -*- coding: utf-8 -*-
"""
Watch-pipeline status aggregation (Phase 5.4).

Small helper that reads the ``watch_results`` table and returns the last
N rows per source plus simple health signals for the public status
endpoint.

Storage lookup order:

1. ``GL_FACTORS_WATCH_SQLITE`` (dev / CI)
2. ``GL_FACTORS_DATABASE_URL`` (production — not implemented here yet)
3. Empty response (graceful degrade, so the public endpoint never 500s)
"""
from __future__ import annotations

import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from greenlang.factors.source_registry import load_source_registry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SQLite path resolution
# ---------------------------------------------------------------------------


def _watch_sqlite_path() -> Optional[Path]:
    """Return the SQLite path holding watch results, or ``None``."""
    raw = os.getenv("GL_FACTORS_WATCH_SQLITE", "").strip()
    if not raw:
        # Fall back to the main factors SQLite db if it exists.
        raw = os.getenv("GL_FACTORS_SQLITE_PATH", "").strip()
    if not raw:
        return None
    p = Path(raw).expanduser()
    return p if p.exists() else None


# ---------------------------------------------------------------------------
# Schema bootstrap + read path
# ---------------------------------------------------------------------------


_SCHEMA = """
CREATE TABLE IF NOT EXISTS watch_results (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id        TEXT NOT NULL,
    check_timestamp  TEXT NOT NULL,
    watch_mechanism  TEXT NOT NULL DEFAULT 'http_head',
    url              TEXT,
    http_status      INTEGER,
    file_hash        TEXT,
    previous_hash    TEXT,
    change_detected  INTEGER NOT NULL DEFAULT 0,
    change_type      TEXT,
    response_ms      INTEGER,
    error_message    TEXT,
    metadata_json    TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_watch_source
    ON watch_results (source_id, check_timestamp DESC);
"""


def _read_sqlite_rows(
    sqlite_path: Path,
    source_id: str,
    limit: int,
) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(str(sqlite_path))
    try:
        conn.executescript(_SCHEMA)  # no-op if table exists
        rows = list(
            conn.execute(
                """
                SELECT id, source_id, check_timestamp, watch_mechanism, url,
                       http_status, file_hash, previous_hash, change_detected,
                       change_type, response_ms, error_message
                FROM watch_results
                WHERE source_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (source_id, limit),
            )
        )
    finally:
        conn.close()
    return [
        {
            "id": r[0],
            "source_id": r[1],
            "check_timestamp": r[2],
            "watch_mechanism": r[3],
            "url": r[4],
            "http_status": r[5],
            "file_hash": r[6],
            "previous_hash": r[7],
            "change_detected": bool(r[8]),
            "change_type": r[9],
            "response_ms": r[10],
            "error_message": r[11],
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Health classification
# ---------------------------------------------------------------------------


def _classify_source(rows: List[Dict[str, Any]]) -> str:
    """Return ``"healthy" | "stale" | "error" | "unknown"`` for a source.

    - ``"error"`` if the most recent check recorded an error or non-2xx.
    - ``"stale"`` if the most recent check is older than 7 days.
    - ``"healthy"`` if the most recent check is 2xx within 7 days.
    - ``"unknown"`` if no rows.
    """
    if not rows:
        return "unknown"
    latest = rows[0]
    if latest.get("error_message"):
        return "error"
    status = latest.get("http_status")
    if status is not None and (status < 200 or status >= 400):
        return "error"
    try:
        ts = datetime.fromisoformat(str(latest["check_timestamp"]).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return "unknown"
    age_days = (datetime.now(timezone.utc) - ts).total_seconds() / 86400
    if age_days > 7:
        return "stale"
    return "healthy"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def collect_watch_status(*, limit_per_source: int = 10) -> Dict[str, Any]:
    """Return a per-source snapshot for the public ``/watch/status`` endpoint."""
    try:
        registry = load_source_registry()
    except Exception as exc:  # noqa: BLE001
        logger.warning("watch/status: could not load source registry: %s", exc)
        registry = []

    sqlite_path = _watch_sqlite_path()
    sources: List[Dict[str, Any]] = []

    for entry in registry:
        # ``entry`` is a ``SourceRegistryEntry`` dataclass.
        source_id = getattr(entry, "source_id", None)
        if not source_id:
            continue

        rows: List[Dict[str, Any]] = []
        if sqlite_path is not None:
            try:
                rows = _read_sqlite_rows(sqlite_path, source_id, limit_per_source)
            except Exception as exc:  # noqa: BLE001
                logger.debug("watch/status: sqlite read failed for %s: %s", source_id, exc)

        sources.append(
            {
                "source_id": source_id,
                "display_name": getattr(entry, "display_name", source_id),
                "cadence": getattr(entry, "cadence", None),
                "connector_only": getattr(entry, "connector_only", False),
                "health": _classify_source(rows),
                "recent_checks": rows,
                "latest_timestamp": rows[0]["check_timestamp"] if rows else None,
                "checks_in_window": len(rows),
            }
        )

    # Deterministic alphabetical order.
    sources.sort(key=lambda s: s["source_id"])

    health_counts: Dict[str, int] = {
        "healthy": 0, "stale": 0, "error": 0, "unknown": 0,
    }
    for src in sources:
        health_counts[src["health"]] = health_counts.get(src["health"], 0) + 1

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_count": len(sources),
        "health_counts": health_counts,
        "sources": sources,
        "limit_per_source": limit_per_source,
        "sqlite_path": str(sqlite_path) if sqlite_path else None,
    }
