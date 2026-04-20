# -*- coding: utf-8 -*-
"""Tests for Phase 4.2 Factors billing / credit metering."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from greenlang.factors.billing import metering
from greenlang.factors.billing.metering import (
    ENDPOINT_CREDITS,
    credits_for,
    record_usage_event,
    totals_by_tenant,
)


# --------------------------------------------------------------------------
# Credit cost calculation
# --------------------------------------------------------------------------


class TestCreditsFor:
    def test_search_costs_1(self):
        assert credits_for("/api/v1/factors/search") == 1

    def test_search_v2_costs_1(self):
        assert credits_for("/api/v1/factors/search/v2") == 1

    def test_match_costs_2(self):
        assert credits_for("/api/v1/factors/match") == 2

    def test_diff_costs_2(self):
        assert credits_for("/api/v1/factors/abc/diff") == 2

    def test_audit_bundle_costs_5(self):
        assert credits_for("/api/v1/factors/abc/audit-bundle") == 5

    def test_export_scales_per_100_rows(self):
        assert credits_for("/api/v1/factors/export", row_count=1) == 1
        assert credits_for("/api/v1/factors/export", row_count=50) == 1
        assert credits_for("/api/v1/factors/export", row_count=100) == 1
        assert credits_for("/api/v1/factors/export", row_count=101) == 2
        assert credits_for("/api/v1/factors/export", row_count=1000) == 10
        assert credits_for("/api/v1/factors/export", row_count=250) == 3

    def test_unknown_endpoint_costs_1(self):
        assert credits_for("/api/v1/factors/custom/action") == 1

    def test_table_has_expected_keys(self):
        for key in ("/search", "/match", "/export", "/audit-bundle"):
            assert key in ENDPOINT_CREDITS


# --------------------------------------------------------------------------
# Event recording end-to-end
# --------------------------------------------------------------------------


class TestRecordUsageEvent:
    def test_writes_row_to_sqlite(self, tmp_path: Path, monkeypatch):
        db = tmp_path / "usage.sqlite"
        monkeypatch.setenv("GL_FACTORS_USAGE_SQLITE", str(db))

        event = record_usage_event(
            tier="pro",
            endpoint="/api/v1/factors/match",
            method="POST",
            user={
                "user_id": "u1",
                "tenant_id": "t1",
                "api_key_id": "k-001",
            },
        )
        assert event.credits == 2
        assert event.tier == "pro"

        conn = sqlite3.connect(str(db))
        try:
            row = conn.execute(
                "SELECT endpoint, credits, row_count FROM api_usage_credits"
            ).fetchone()
        finally:
            conn.close()
        assert row == ("/api/v1/factors/match", 2, 1)

    def test_export_scales_credits(self, tmp_path: Path, monkeypatch):
        db = tmp_path / "usage.sqlite"
        monkeypatch.setenv("GL_FACTORS_USAGE_SQLITE", str(db))
        event = record_usage_event(
            tier="pro",
            endpoint="/api/v1/factors/export",
            row_count=500,
        )
        assert event.credits == 5
        assert event.row_count == 500

    def test_no_sqlite_is_noop(self, tmp_path: Path, monkeypatch):
        monkeypatch.delenv("GL_FACTORS_USAGE_SQLITE", raising=False)
        # Still returns an event, just doesn't persist.
        event = record_usage_event(
            tier="community",
            endpoint="/api/v1/factors/search",
        )
        assert event.credits == 1

    def test_totals_by_tenant(self, tmp_path: Path, monkeypatch):
        db = tmp_path / "usage.sqlite"
        monkeypatch.setenv("GL_FACTORS_USAGE_SQLITE", str(db))
        record_usage_event(
            tier="pro", endpoint="/api/v1/factors/search",
            user={"tenant_id": "acme"},
        )
        record_usage_event(
            tier="pro", endpoint="/api/v1/factors/match",
            user={"tenant_id": "acme"},
        )
        record_usage_event(
            tier="enterprise", endpoint="/api/v1/factors/abc/audit-bundle",
            user={"tenant_id": "big-co"},
        )
        totals = totals_by_tenant()
        assert totals["acme"] == 3  # search (1) + match (2)
        assert totals["big-co"] == 5  # audit-bundle

    def test_api_key_hash_recorded(self, tmp_path: Path, monkeypatch):
        db = tmp_path / "usage.sqlite"
        monkeypatch.setenv("GL_FACTORS_USAGE_SQLITE", str(db))
        record_usage_event(
            tier="pro",
            endpoint="/api/v1/factors/search",
            user={"api_key_id": "k-001", "tenant_id": "t1"},
            api_key="sk_live_test",
        )
        conn = sqlite3.connect(str(db))
        try:
            row = conn.execute(
                "SELECT api_key_id, tenant_id FROM api_usage_credits"
            ).fetchone()
        finally:
            conn.close()
        assert row == ("k-001", "t1")
