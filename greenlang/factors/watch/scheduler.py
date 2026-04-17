# -*- coding: utf-8 -*-
"""
Automated source watch scheduler (F050).

Daily HTTP HEAD/GET checks for all sources in the registry. Stores results
in watch_results table (V430), detects changes, and sends notifications.

CLI: ``python -m greenlang.factors.cli watch-run``
Cron: Daily at 06:00 UTC via Docker/K8s CronJob.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from greenlang.factors.ingestion.fetchers import head_exists
from greenlang.factors.source_registry import load_source_registry
from greenlang.factors.watch.change_classification import classify_change

logger = logging.getLogger(__name__)


@dataclass
class WatchResult:
    """Result of a single source check."""

    source_id: str
    check_timestamp: str
    watch_mechanism: str
    url: Optional[str] = None
    http_status: Optional[int] = None
    file_hash: Optional[str] = None
    previous_hash: Optional[str] = None
    change_detected: bool = False
    change_type: Optional[str] = None
    response_ms: Optional[int] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fetch_and_hash(url: str, mechanism: str) -> tuple[Optional[int], Optional[str], Optional[int], Optional[str]]:
    """
    Fetch a URL and return (http_status, content_hash, response_ms, error).

    Supports http_head (status-only), http_get (full content hash),
    and html_diff (full content hash).
    """
    import urllib.error
    import urllib.request

    start = time.monotonic()
    try:
        if mechanism == "http_head":
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=30) as resp:  # nosec B310
                elapsed = int((time.monotonic() - start) * 1000)
                return resp.status, None, elapsed, None
        else:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=60) as resp:  # nosec B310
                body = resp.read()
                elapsed = int((time.monotonic() - start) * 1000)
                h = hashlib.sha256(body).hexdigest()
                return resp.status, h, elapsed, None
    except urllib.error.HTTPError as e:
        elapsed = int((time.monotonic() - start) * 1000)
        return e.code, None, elapsed, str(e.reason)
    except Exception as e:
        elapsed = int((time.monotonic() - start) * 1000)
        return None, None, elapsed, str(e)


def _get_previous_hash(db_path: Optional[Path], source_id: str) -> Optional[str]:
    """Retrieve the most recent file_hash for a source from SQLite watch_results."""
    if not db_path or not db_path.is_file():
        return None
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT file_hash FROM watch_results "
            "WHERE source_id = ? AND file_hash IS NOT NULL "
            "ORDER BY check_timestamp DESC LIMIT 1",
            (source_id,),
        ).fetchone()
        conn.close()
        return str(row["file_hash"]) if row else None
    except Exception:
        return None


def _store_result_sqlite(db_path: Path, result: WatchResult) -> None:
    """Store a watch result in the local SQLite watch_results table."""
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS watch_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id TEXT NOT NULL,
            check_timestamp TEXT NOT NULL,
            watch_mechanism TEXT NOT NULL DEFAULT 'http_head',
            url TEXT,
            http_status INTEGER,
            file_hash TEXT,
            previous_hash TEXT,
            change_detected BOOLEAN NOT NULL DEFAULT 0,
            change_type TEXT,
            response_ms INTEGER,
            error_message TEXT,
            metadata_json TEXT NOT NULL DEFAULT '{}'
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_wr_source "
        "ON watch_results (source_id, check_timestamp DESC)",
    )
    conn.execute(
        """
        INSERT INTO watch_results (
            source_id, check_timestamp, watch_mechanism, url,
            http_status, file_hash, previous_hash, change_detected,
            change_type, response_ms, error_message, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            result.source_id,
            result.check_timestamp,
            result.watch_mechanism,
            result.url,
            result.http_status,
            result.file_hash,
            result.previous_hash,
            int(result.change_detected),
            result.change_type,
            result.response_ms,
            result.error_message,
            json.dumps(result.metadata, default=str),
        ),
    )
    conn.commit()
    conn.close()


NotifyCallback = Callable[[str, WatchResult], None]


def run_watch(
    *,
    registry_path: Optional[Path] = None,
    db_path: Optional[Path] = None,
    notify: Optional[NotifyCallback] = None,
    store: bool = True,
) -> List[WatchResult]:
    """
    Execute a full watch run across all registry sources.

    Args:
        registry_path: Path to source_registry.yaml (None = default).
        db_path: SQLite path for storing results and retrieving previous hashes.
        notify: Callback(message, result) for change/error notifications.
        store: If True, persist results to SQLite.

    Returns:
        List of WatchResult for each source checked.
    """
    sources = load_source_registry(registry_path)
    results: List[WatchResult] = []
    changes_found = 0
    errors_found = 0

    for entry in sources:
        url = entry.watch_url
        mechanism = entry.watch_mechanism or "http_head"

        result = WatchResult(
            source_id=entry.source_id,
            check_timestamp=_now_iso(),
            watch_mechanism=mechanism,
            url=url,
        )

        if not url or mechanism in ("manual", "none"):
            result.metadata["skipped"] = True
            results.append(result)
            logger.debug("Skipping source %s (mechanism=%s)", entry.source_id, mechanism)
            continue

        # Fetch and check
        http_status, file_hash, response_ms, error = _fetch_and_hash(url, mechanism)
        result.http_status = http_status
        result.file_hash = file_hash
        result.response_ms = response_ms
        result.error_message = error

        # Compare to previous
        if file_hash:
            prev = _get_previous_hash(db_path, entry.source_id)
            result.previous_hash = prev
            if prev and prev != file_hash:
                result.change_detected = True
                result.change_type = "content_changed"
                changes_found += 1
                logger.info(
                    "Change detected: source=%s hash=%s->%s",
                    entry.source_id, prev[:12], file_hash[:12],
                )
            elif not prev:
                result.change_type = "first_check"

        # Detect errors
        if error:
            errors_found += 1
            logger.warning(
                "Watch error: source=%s url=%s error=%s",
                entry.source_id, url, error,
            )

        # Detect 404/gone
        if http_status and http_status >= 400:
            result.change_detected = True
            result.change_type = "source_unavailable"
            errors_found += 1

        # Store
        if store and db_path:
            _store_result_sqlite(db_path, result)

        # Notify on changes or errors
        if notify and (result.change_detected or result.error_message):
            severity = "WARNING" if result.error_message else "INFO"
            msg = (
                f"[{severity}] Source watch: {entry.source_id} — "
                f"{'CHANGE: ' + (result.change_type or '') if result.change_detected else ''}"
                f"{'ERROR: ' + (result.error_message or '') if result.error_message else ''}"
            )
            try:
                notify(msg, result)
            except Exception:
                logger.exception("Notify callback failed for source=%s", entry.source_id)

        results.append(result)

    logger.info(
        "Watch run complete: sources=%d changes=%d errors=%d",
        len(results), changes_found, errors_found,
    )
    return results


def watch_summary(results: List[WatchResult]) -> Dict[str, Any]:
    """Generate a summary dict from watch results."""
    return {
        "total_sources": len(results),
        "changes_detected": sum(1 for r in results if r.change_detected),
        "errors": sum(1 for r in results if r.error_message),
        "skipped": sum(1 for r in results if r.metadata.get("skipped")),
        "sources_checked": sum(
            1 for r in results
            if not r.metadata.get("skipped") and r.http_status is not None
        ),
        "changed_sources": [
            r.source_id for r in results if r.change_detected
        ],
        "error_sources": [
            r.source_id for r in results if r.error_message
        ],
        "timestamp": _now_iso(),
    }
