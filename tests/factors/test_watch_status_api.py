# -*- coding: utf-8 -*-
"""Phase 5.4 watch-status API tests."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from greenlang.factors.watch.status_api import (
    _classify_source,
    _read_sqlite_rows,
    collect_watch_status,
)


def _seed_sqlite(db: Path, rows: list[dict]) -> None:
    conn = sqlite3.connect(str(db))
    try:
        conn.executescript(
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
                change_detected INTEGER NOT NULL DEFAULT 0,
                change_type TEXT,
                response_ms INTEGER,
                error_message TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );
            """
        )
        for r in rows:
            conn.execute(
                """
                INSERT INTO watch_results
                    (source_id, check_timestamp, watch_mechanism, url,
                     http_status, change_detected, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    r["source_id"],
                    r["check_timestamp"],
                    r.get("watch_mechanism", "http_head"),
                    r.get("url"),
                    r.get("http_status"),
                    1 if r.get("change_detected") else 0,
                    r.get("error_message"),
                ),
            )
        conn.commit()
    finally:
        conn.close()


# --------------------------------------------------------------------------
# Health classifier
# --------------------------------------------------------------------------


class TestClassify:
    def test_empty_is_unknown(self):
        assert _classify_source([]) == "unknown"

    def test_error_message_makes_error(self):
        rows = [{"check_timestamp": "2026-04-20T00:00:00+00:00", "error_message": "boom"}]
        assert _classify_source(rows) == "error"

    def test_non_2xx_is_error(self):
        rows = [{"check_timestamp": "2026-04-20T00:00:00+00:00", "http_status": 500}]
        assert _classify_source(rows) == "error"

    def test_recent_2xx_is_healthy(self):
        now = datetime.now(timezone.utc).isoformat()
        rows = [{"check_timestamp": now, "http_status": 200}]
        assert _classify_source(rows) == "healthy"

    def test_old_check_is_stale(self):
        old = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        rows = [{"check_timestamp": old, "http_status": 200}]
        assert _classify_source(rows) == "stale"


# --------------------------------------------------------------------------
# SQLite read
# --------------------------------------------------------------------------


class TestReadRows:
    def test_returns_rows_in_descending_id_order(self, tmp_path: Path):
        db = tmp_path / "watch.sqlite"
        _seed_sqlite(
            db,
            [
                {"source_id": "epa", "check_timestamp": "2026-04-01T00:00:00+00:00", "http_status": 200},
                {"source_id": "epa", "check_timestamp": "2026-04-02T00:00:00+00:00", "http_status": 200},
                {"source_id": "desnz", "check_timestamp": "2026-04-01T00:00:00+00:00", "http_status": 200},
            ],
        )
        rows = _read_sqlite_rows(db, "epa", limit=10)
        assert len(rows) == 2
        # Most-recent first.
        assert rows[0]["check_timestamp"] == "2026-04-02T00:00:00+00:00"

    def test_honours_limit(self, tmp_path: Path):
        db = tmp_path / "watch.sqlite"
        _seed_sqlite(
            db,
            [
                {"source_id": "epa", "check_timestamp": f"2026-04-{i:02d}T00:00:00+00:00", "http_status": 200}
                for i in range(1, 6)
            ],
        )
        rows = _read_sqlite_rows(db, "epa", limit=2)
        assert len(rows) == 2


# --------------------------------------------------------------------------
# collect_watch_status (end-to-end)
# --------------------------------------------------------------------------


class TestCollectWatchStatus:
    def test_returns_registry_sources_even_without_sqlite(self, monkeypatch):
        monkeypatch.delenv("GL_FACTORS_WATCH_SQLITE", raising=False)
        monkeypatch.delenv("GL_FACTORS_SQLITE_PATH", raising=False)
        summary = collect_watch_status()
        assert summary["source_count"] > 0
        assert summary["sqlite_path"] is None
        assert all(s["health"] == "unknown" for s in summary["sources"])
        assert summary["health_counts"]["unknown"] == summary["source_count"]

    def test_healthy_when_recent_check(self, tmp_path: Path, monkeypatch):
        db = tmp_path / "watch.sqlite"
        now = datetime.now(timezone.utc).isoformat()
        _seed_sqlite(
            db,
            [
                {"source_id": "epa_hub", "check_timestamp": now, "http_status": 200},
            ],
        )
        monkeypatch.setenv("GL_FACTORS_WATCH_SQLITE", str(db))
        summary = collect_watch_status(limit_per_source=5)
        epa = next(s for s in summary["sources"] if s["source_id"] == "epa_hub")
        assert epa["health"] == "healthy"
        assert epa["checks_in_window"] == 1

    def test_error_when_latest_check_failed(self, tmp_path: Path, monkeypatch):
        db = tmp_path / "watch.sqlite"
        now = datetime.now(timezone.utc).isoformat()
        _seed_sqlite(
            db,
            [
                {
                    "source_id": "desnz_ghg_conversion",
                    "check_timestamp": now,
                    "http_status": 502,
                    "error_message": "upstream timeout",
                }
            ],
        )
        monkeypatch.setenv("GL_FACTORS_WATCH_SQLITE", str(db))
        summary = collect_watch_status()
        desnz = next(s for s in summary["sources"] if s["source_id"] == "desnz_ghg_conversion")
        assert desnz["health"] == "error"
        assert desnz["recent_checks"][0]["http_status"] == 502

    def test_deterministic_source_order(self, monkeypatch):
        monkeypatch.delenv("GL_FACTORS_WATCH_SQLITE", raising=False)
        monkeypatch.delenv("GL_FACTORS_SQLITE_PATH", raising=False)
        summary = collect_watch_status()
        ids = [s["source_id"] for s in summary["sources"]]
        assert ids == sorted(ids)
