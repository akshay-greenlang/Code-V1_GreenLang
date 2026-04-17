# -*- coding: utf-8 -*-
"""Tests for greenlang.factors.watch.scheduler (F050)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from greenlang.factors.watch.scheduler import (
    WatchResult,
    _store_result_sqlite,
    run_watch,
    watch_summary,
)


def test_watch_result_defaults():
    r = WatchResult(source_id="test", check_timestamp="2026-04-17T00:00:00Z", watch_mechanism="http_head")
    assert r.change_detected is False
    assert r.error_message is None
    assert r.metadata == {}


def test_store_result_sqlite(tmp_path):
    db_path = tmp_path / "watch.db"
    r = WatchResult(
        source_id="epa_ghg",
        check_timestamp="2026-04-17T06:00:00Z",
        watch_mechanism="http_get",
        url="https://example.com/epa",
        http_status=200,
        file_hash="abc123",
        response_ms=150,
    )
    _store_result_sqlite(db_path, r)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM watch_results").fetchall()
    conn.close()

    assert len(rows) == 1
    assert rows[0]["source_id"] == "epa_ghg"
    assert rows[0]["http_status"] == 200
    assert rows[0]["file_hash"] == "abc123"


def test_store_result_sqlite_change_detected(tmp_path):
    db_path = tmp_path / "watch.db"
    r = WatchResult(
        source_id="defra",
        check_timestamp="2026-04-17T06:00:00Z",
        watch_mechanism="http_get",
        change_detected=True,
        change_type="content_changed",
    )
    _store_result_sqlite(db_path, r)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM watch_results WHERE change_detected = 1").fetchall()
    conn.close()

    assert len(rows) == 1
    assert rows[0]["change_type"] == "content_changed"


def test_watch_summary():
    results = [
        WatchResult(source_id="a", check_timestamp="t", watch_mechanism="http_head",
                     http_status=200),
        WatchResult(source_id="b", check_timestamp="t", watch_mechanism="http_get",
                     http_status=200, change_detected=True, change_type="content_changed"),
        WatchResult(source_id="c", check_timestamp="t", watch_mechanism="manual",
                     metadata={"skipped": True}),
        WatchResult(source_id="d", check_timestamp="t", watch_mechanism="http_head",
                     error_message="timeout"),
    ]
    summary = watch_summary(results)
    assert summary["total_sources"] == 4
    assert summary["changes_detected"] == 1
    assert summary["errors"] == 1
    assert summary["skipped"] == 1
    assert summary["sources_checked"] == 3
    assert summary["changed_sources"] == ["b"]
    assert summary["error_sources"] == ["d"]


@patch("greenlang.factors.watch.scheduler.load_source_registry")
@patch("greenlang.factors.watch.scheduler._fetch_and_hash")
def test_run_watch_basic(mock_fetch, mock_registry):
    """Test run_watch with mocked registry and fetch."""
    from types import SimpleNamespace

    mock_registry.return_value = [
        SimpleNamespace(source_id="epa", watch_url="https://epa.gov/data", watch_mechanism="http_get"),
        SimpleNamespace(source_id="manual_src", watch_url=None, watch_mechanism="manual"),
    ]
    mock_fetch.return_value = (200, "hashABC", 100, None)

    results = run_watch(store=False)
    assert len(results) == 2
    assert results[0].source_id == "epa"
    assert results[0].http_status == 200
    assert results[0].file_hash == "hashABC"
    assert results[1].metadata.get("skipped") is True


@patch("greenlang.factors.watch.scheduler.load_source_registry")
@patch("greenlang.factors.watch.scheduler._fetch_and_hash")
def test_run_watch_detects_change(mock_fetch, mock_registry, tmp_path):
    """Test that run_watch detects content changes."""
    from types import SimpleNamespace

    db_path = tmp_path / "watch.db"

    # Seed a previous result
    _store_result_sqlite(db_path, WatchResult(
        source_id="epa",
        check_timestamp="2026-04-16T06:00:00Z",
        watch_mechanism="http_get",
        file_hash="old_hash",
    ))

    mock_registry.return_value = [
        SimpleNamespace(source_id="epa", watch_url="https://epa.gov/data", watch_mechanism="http_get"),
    ]
    mock_fetch.return_value = (200, "new_hash", 120, None)

    results = run_watch(db_path=db_path, store=True)
    assert len(results) == 1
    assert results[0].change_detected is True
    assert results[0].change_type == "content_changed"
    assert results[0].previous_hash == "old_hash"


@patch("greenlang.factors.watch.scheduler.load_source_registry")
@patch("greenlang.factors.watch.scheduler._fetch_and_hash")
def test_run_watch_error_notification(mock_fetch, mock_registry):
    """Test notification callback is called on errors."""
    from types import SimpleNamespace

    mock_registry.return_value = [
        SimpleNamespace(source_id="broken", watch_url="https://broken.com", watch_mechanism="http_get"),
    ]
    mock_fetch.return_value = (None, None, 5000, "Connection timeout")

    notifications = []

    def on_notify(msg, result):
        notifications.append((msg, result))

    results = run_watch(store=False, notify=on_notify)
    assert len(results) == 1
    assert results[0].error_message == "Connection timeout"
    assert len(notifications) == 1
    assert "WARNING" in notifications[0][0]
