# -*- coding: utf-8 -*-
"""Tests for the customer-private (encrypted) tenant overlay path.

Extends the existing ``test_tenant_overlay.py`` coverage with:

* Plaintext MUST NOT be persisted to SQLite for customer_private rows.
* Reads via ``decrypt_private_value`` return the original float.
* Cross-tenant reads raise ``TenantKeyAccessError`` (the analogue of 403).
* Audit log never carries the plaintext float for private overlays.
* ``merge_search_results`` transparently surfaces the plaintext for
  the overlay's owner.
"""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path

import pytest

from greenlang.factors.security.tenant_vault_transit import (
    TenantKeyAccessError,
    TenantVaultTransit,
    reset_default_transit,
)
from greenlang.factors.tenant_overlay import (
    REDISTRIBUTION_CUSTOMER_PRIVATE,
    TenantOverlayRegistry,
)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    """Ensure dev-mode transit on every test, no env leak."""
    for var in ("VAULT_ADDR", "VAULT_TOKEN", "GL_ENV", "ENVIRONMENT"):
        monkeypatch.delenv(var, raising=False)
    reset_default_transit()
    yield
    reset_default_transit()


@pytest.fixture
def transit():
    return TenantVaultTransit()


@pytest.fixture
def reg(transit):
    return TenantOverlayRegistry(transit=transit)


@pytest.fixture
def reg_with_db(tmp_path, transit) -> TenantOverlayRegistry:
    db = tmp_path / "overlays.sqlite"
    return TenantOverlayRegistry(db_path=db, transit=transit)


class TestPlaintextNeverPersisted:
    def test_private_value_not_in_db(self, tmp_path, transit):
        db = tmp_path / "overlays.sqlite"
        reg = TenantOverlayRegistry(db_path=db, transit=transit)
        reg.create_overlay(
            tenant_id="tenant-a",
            factor_id="EF:custom:widget",
            override_value=12345.6789,
            redistribution_class=REDISTRIBUTION_CUSTOMER_PRIVATE,
            created_by="alice",
        )

        conn = sqlite3.connect(str(db))
        rows = conn.execute(
            "SELECT override_value, override_value_ciphertext, "
            "override_value_digest, redistribution_class "
            "FROM tenant_overlays"
        ).fetchall()
        conn.close()

        assert len(rows) == 1
        stored_value, ciphertext, digest, klass = rows[0]
        # Column is NaN sentinel, not the real number.
        assert stored_value is None or (
            isinstance(stored_value, float) and math.isnan(stored_value)
        )
        assert ciphertext and ciphertext.startswith("vault:v")
        # Plaintext literal must not be present anywhere in the row.
        plaintext_repr = "12345.6789"
        assert plaintext_repr not in (ciphertext or "")
        assert plaintext_repr not in (digest or "")
        assert klass == REDISTRIBUTION_CUSTOMER_PRIVATE

    def test_raw_db_blob_never_contains_plaintext(self, tmp_path, transit):
        db = tmp_path / "overlays.sqlite"
        reg = TenantOverlayRegistry(db_path=db, transit=transit)
        reg.create_overlay(
            tenant_id="tenant-a",
            factor_id="EF:x",
            override_value=1234.5,
            redistribution_class=REDISTRIBUTION_CUSTOMER_PRIVATE,
        )
        raw = Path(db).read_bytes()
        # SQLite pages should not contain the raw float literal.
        assert b"1234.5" not in raw


class TestReadBack:
    def test_owner_can_decrypt(self, reg):
        overlay = reg.create_overlay(
            tenant_id="tenant-a",
            factor_id="EF:x",
            override_value=99.125,
            redistribution_class=REDISTRIBUTION_CUSTOMER_PRIVATE,
        )
        value = reg.decrypt_private_value("tenant-a", overlay)
        assert value == pytest.approx(99.125)

    def test_owner_reads_via_in_memory_plaintext(self, reg):
        # After write, the in-memory object still carries plaintext so
        # hot-path callers don't pay for a Vault round-trip.
        overlay = reg.create_overlay(
            tenant_id="tenant-a",
            factor_id="EF:y",
            override_value=42.0,
            redistribution_class=REDISTRIBUTION_CUSTOMER_PRIVATE,
        )
        assert overlay.override_value == 42.0
        assert overlay.is_private()
        assert overlay.override_value_ciphertext
        assert overlay.override_value_digest

    def test_non_private_overlay_reads_directly(self, reg):
        overlay = reg.create_overlay(
            tenant_id="tenant-a",
            factor_id="EF:z",
            override_value=7.5,
        )
        assert not overlay.is_private()
        assert overlay.override_value == 7.5


class TestCrossTenantDenied:
    def test_cross_tenant_decrypt_raises(self, reg):
        overlay_a = reg.create_overlay(
            tenant_id="tenant-a",
            factor_id="EF:priv",
            override_value=3.14,
            redistribution_class=REDISTRIBUTION_CUSTOMER_PRIVATE,
        )
        with pytest.raises(TenantKeyAccessError):
            reg.decrypt_private_value("tenant-b", overlay_a)

    def test_ciphertext_handed_to_other_tenant_rejected(self, reg, transit):
        overlay_a = reg.create_overlay(
            tenant_id="tenant-a",
            factor_id="EF:priv",
            override_value=3.14,
            redistribution_class=REDISTRIBUTION_CUSTOMER_PRIVATE,
        )
        ct = overlay_a.override_value_ciphertext
        assert ct is not None
        with pytest.raises(TenantKeyAccessError):
            transit.decrypt("tenant-b", ct)


class TestAuditLogHygiene:
    def test_audit_entry_omits_plaintext_for_private(self, reg):
        reg.create_overlay(
            tenant_id="tenant-a",
            factor_id="EF:priv",
            override_value=424242.424242,
            redistribution_class=REDISTRIBUTION_CUSTOMER_PRIVATE,
        )
        audits = reg.audit_log
        assert len(audits) == 1
        entry = audits[0]
        # Neither old nor new numeric value is captured.
        assert entry.old_value is None
        assert entry.new_value is None
        # But the digest + pointer ARE captured.
        assert "value_digest" in entry.details
        assert entry.details["value_digest"]
        serialised = repr(entry).encode("utf-8")
        assert b"424242.424242" not in serialised

    def test_audit_entry_includes_plaintext_for_public(self, reg):
        reg.create_overlay(
            tenant_id="tenant-a",
            factor_id="EF:pub",
            override_value=3.5,  # non-private: logged as usual
        )
        entry = reg.audit_log[-1]
        assert entry.new_value == pytest.approx(3.5)


class TestMergeSearchResults:
    def test_merge_surfaces_plaintext_for_owner(self, reg):
        reg.create_overlay(
            tenant_id="tenant-a",
            factor_id="EF:priv",
            override_value=10.0,
            redistribution_class=REDISTRIBUTION_CUSTOMER_PRIVATE,
        )
        catalog = [
            {"factor_id": "EF:priv", "co2e_total": 99.0, "unit": "kg_co2e"},
            {"factor_id": "EF:other", "co2e_total": 5.0, "unit": "kg_co2e"},
        ]
        merged = reg.merge_search_results("tenant-a", catalog)

        priv = next(r for r in merged if r["factor_id"] == "EF:priv")
        assert priv["co2e_total"] == pytest.approx(10.0)
        assert (
            priv["_overlay_redistribution_class"]
            == REDISTRIBUTION_CUSTOMER_PRIVATE
        )

        other = next(r for r in merged if r["factor_id"] == "EF:other")
        assert other["co2e_total"] == 5.0  # untouched


class TestUpdate:
    def test_update_reencrypts_private_value(self, reg, transit):
        overlay = reg.create_overlay(
            tenant_id="tenant-a",
            factor_id="EF:priv",
            override_value=1.0,
            redistribution_class=REDISTRIBUTION_CUSTOMER_PRIVATE,
        )
        old_ct = overlay.override_value_ciphertext

        updated = reg.update_overlay(
            "tenant-a", overlay.overlay_id, override_value=2.0, updated_by="bob",
        )
        assert updated is not None
        assert updated.override_value == 2.0
        # New ciphertext should be different from the old one.
        assert updated.override_value_ciphertext
        assert updated.override_value_ciphertext != old_ct
        # Owner can decrypt and get the new value.
        assert reg.decrypt_private_value("tenant-a", updated) == pytest.approx(2.0)


class TestErrorPaths:
    def test_decrypt_on_non_private_raises(self, reg):
        overlay = reg.create_overlay(
            tenant_id="tenant-a",
            factor_id="EF:pub",
            override_value=1.0,
        )
        with pytest.raises(ValueError):
            reg.decrypt_private_value("tenant-a", overlay)
