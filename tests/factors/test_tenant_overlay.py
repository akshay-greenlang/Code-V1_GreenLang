# -*- coding: utf-8 -*-
"""Tests for greenlang.factors.tenant_overlay (F064)."""

from __future__ import annotations

from datetime import date

import pytest

from greenlang.factors.tenant_overlay import (
    OverlayAuditEntry,
    TenantOverlay,
    TenantOverlayRegistry,
)


class TestTenantOverlay:
    def test_is_valid_today(self):
        o = TenantOverlay(
            overlay_id="uuid1",
            tenant_id="t1",
            factor_id="EF:1",
            override_value=1.5,
            valid_from="2020-01-01",
        )
        assert o.is_valid_on(date.today().isoformat()) is True

    def test_not_valid_before_start(self):
        o = TenantOverlay(
            overlay_id="uuid1",
            tenant_id="t1",
            factor_id="EF:1",
            override_value=1.5,
            valid_from="2030-01-01",
        )
        assert o.is_valid_on("2026-04-17") is False

    def test_not_valid_after_end(self):
        o = TenantOverlay(
            overlay_id="uuid1",
            tenant_id="t1",
            factor_id="EF:1",
            override_value=1.5,
            valid_from="2020-01-01",
            valid_to="2025-12-31",
        )
        assert o.is_valid_on("2026-04-17") is False

    def test_inactive_not_valid(self):
        o = TenantOverlay(
            overlay_id="uuid1",
            tenant_id="t1",
            factor_id="EF:1",
            override_value=1.5,
            valid_from="2020-01-01",
            active=False,
        )
        assert o.is_valid_on("2026-04-17") is False

    def test_to_dict(self):
        o = TenantOverlay(
            overlay_id="uuid1",
            tenant_id="t1",
            factor_id="EF:1",
            override_value=1.5,
        )
        d = o.to_dict()
        assert d["overlay_id"] == "uuid1"
        assert d["override_value"] == 1.5


class TestTenantOverlayRegistry:
    def test_create_overlay(self):
        reg = TenantOverlayRegistry()
        o = reg.create_overlay("tenant_a", "EF:1", 2.5, created_by="alice")
        assert o.tenant_id == "tenant_a"
        assert o.factor_id == "EF:1"
        assert o.override_value == 2.5
        assert o.active is True

    def test_list_overlays(self):
        reg = TenantOverlayRegistry()
        reg.create_overlay("t1", "EF:1", 1.0)
        reg.create_overlay("t1", "EF:2", 2.0)
        reg.create_overlay("t2", "EF:3", 3.0)
        assert len(reg.list_overlays("t1")) == 2
        assert len(reg.list_overlays("t2")) == 1
        assert len(reg.list_overlays("t3")) == 0

    def test_get_overlay(self):
        reg = TenantOverlayRegistry()
        o = reg.create_overlay("t1", "EF:1", 1.5)
        fetched = reg.get_overlay("t1", o.overlay_id)
        assert fetched is not None
        assert fetched.override_value == 1.5

    def test_update_overlay(self):
        reg = TenantOverlayRegistry()
        o = reg.create_overlay("t1", "EF:1", 1.0)
        updated = reg.update_overlay("t1", o.overlay_id, override_value=2.0, updated_by="bob")
        assert updated is not None
        assert updated.override_value == 2.0

    def test_delete_overlay(self):
        reg = TenantOverlayRegistry()
        o = reg.create_overlay("t1", "EF:1", 1.0)
        assert reg.delete_overlay("t1", o.overlay_id) is True
        assert len(reg.list_overlays("t1", active_only=True)) == 0
        assert len(reg.list_overlays("t1", active_only=False)) == 1

    # ------------------------------------------------------------------
    # Tenant isolation tests
    # ------------------------------------------------------------------

    def test_tenant_isolation_get(self):
        """Tenant A cannot access Tenant B's overlay by ID."""
        reg = TenantOverlayRegistry()
        o_a = reg.create_overlay("tenant_a", "EF:1", 1.0)
        # Tenant B tries to get Tenant A's overlay
        assert reg.get_overlay("tenant_b", o_a.overlay_id) is None

    def test_tenant_isolation_list(self):
        """Tenant A cannot see Tenant B's overlays."""
        reg = TenantOverlayRegistry()
        reg.create_overlay("tenant_a", "EF:1", 1.0)
        reg.create_overlay("tenant_a", "EF:2", 2.0)
        reg.create_overlay("tenant_b", "EF:3", 3.0)
        a_overlays = reg.list_overlays("tenant_a")
        assert len(a_overlays) == 2
        assert all(o.tenant_id == "tenant_a" for o in a_overlays)

    def test_tenant_isolation_update(self):
        """Tenant B cannot update Tenant A's overlay."""
        reg = TenantOverlayRegistry()
        o_a = reg.create_overlay("tenant_a", "EF:1", 1.0)
        result = reg.update_overlay("tenant_b", o_a.overlay_id, override_value=999.0)
        assert result is None
        # Original should be unchanged
        assert reg.get_overlay("tenant_a", o_a.overlay_id).override_value == 1.0

    def test_tenant_isolation_delete(self):
        """Tenant B cannot delete Tenant A's overlay."""
        reg = TenantOverlayRegistry()
        o_a = reg.create_overlay("tenant_a", "EF:1", 1.0)
        assert reg.delete_overlay("tenant_b", o_a.overlay_id) is False
        assert reg.get_overlay("tenant_a", o_a.overlay_id).active is True

    # ------------------------------------------------------------------
    # Resolution and merge
    # ------------------------------------------------------------------

    def test_resolve_factor(self):
        reg = TenantOverlayRegistry()
        reg.create_overlay("t1", "EF:1", 2.0, valid_from="2020-01-01")
        resolved = reg.resolve_factor("t1", "EF:1", check_date="2026-04-17")
        assert resolved is not None
        assert resolved.override_value == 2.0

    def test_resolve_factor_no_overlay(self):
        reg = TenantOverlayRegistry()
        assert reg.resolve_factor("t1", "EF:nonexistent") is None

    def test_resolve_most_recent(self):
        reg = TenantOverlayRegistry()
        reg.create_overlay("t1", "EF:1", 1.0, valid_from="2024-01-01")
        reg.create_overlay("t1", "EF:1", 2.0, valid_from="2025-01-01")
        resolved = reg.resolve_factor("t1", "EF:1", check_date="2026-04-17")
        assert resolved.override_value == 2.0

    def test_merge_search_results(self):
        reg = TenantOverlayRegistry()
        reg.create_overlay("t1", "EF:1", 99.0, valid_from="2020-01-01")

        catalog_results = [
            {"factor_id": "EF:1", "co2e_total": 1.0, "unit": "kg_co2e"},
            {"factor_id": "EF:2", "co2e_total": 2.0, "unit": "kg_co2e"},
        ]
        merged = reg.merge_search_results("t1", catalog_results)
        assert merged[0]["co2e_total"] == 99.0  # Overridden
        assert merged[0]["_overlay_id"] is not None
        assert merged[1]["co2e_total"] == 2.0  # Unchanged

    def test_merge_no_overlays(self):
        reg = TenantOverlayRegistry()
        catalog_results = [{"factor_id": "EF:1", "co2e_total": 1.0}]
        merged = reg.merge_search_results("t1", catalog_results)
        assert merged == catalog_results

    # ------------------------------------------------------------------
    # Audit trail
    # ------------------------------------------------------------------

    def test_audit_log(self):
        reg = TenantOverlayRegistry()
        o = reg.create_overlay("t1", "EF:1", 1.0, created_by="alice")
        reg.update_overlay("t1", o.overlay_id, override_value=2.0, updated_by="bob")
        reg.delete_overlay("t1", o.overlay_id, deleted_by="alice")

        audit = reg.audit_log
        assert len(audit) == 3
        assert audit[0].action == "create"
        assert audit[1].action == "update"
        assert audit[1].old_value == 1.0
        assert audit[1].new_value == 2.0
        assert audit[2].action == "delete"

    def test_audit_for_tenant(self):
        reg = TenantOverlayRegistry()
        reg.create_overlay("t1", "EF:1", 1.0)
        reg.create_overlay("t2", "EF:2", 2.0)
        t1_audit = reg.audit_for_tenant("t1")
        assert len(t1_audit) == 1
        assert t1_audit[0].tenant_id == "t1"

    # ------------------------------------------------------------------
    # SQLite persistence
    # ------------------------------------------------------------------

    def test_sqlite_persistence(self, tmp_path):
        db = tmp_path / "overlays.db"
        reg = TenantOverlayRegistry(db_path=db)
        o = reg.create_overlay("t1", "EF:1", 3.14, created_by="alice")
        assert db.exists()

        # Verify data was written
        import sqlite3
        conn = sqlite3.connect(str(db))
        rows = conn.execute("SELECT * FROM tenant_overlays").fetchall()
        conn.close()
        assert len(rows) == 1

    def test_sqlite_audit(self, tmp_path):
        db = tmp_path / "overlays.db"
        reg = TenantOverlayRegistry(db_path=db)
        reg.create_overlay("t1", "EF:1", 1.0)

        import sqlite3
        conn = sqlite3.connect(str(db))
        rows = conn.execute("SELECT * FROM tenant_overlay_audit").fetchall()
        conn.close()
        assert len(rows) == 1
