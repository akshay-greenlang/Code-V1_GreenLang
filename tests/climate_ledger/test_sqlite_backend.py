# -*- coding: utf-8 -*-
"""
Tests for the Climate Ledger SQLite backend (Phase 2.1).

Covers:
- append-only write path
- chain-hash consistency between memory and SQLite
- verify() cross-checks both stores
- export() returns persisted data
- append-only triggers reject UPDATE and DELETE
- concurrent writes do not corrupt the chain
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path

import pytest

from greenlang.climate_ledger import ClimateLedger
from greenlang.climate_ledger.ledger import _SQLiteLedgerBackend


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _h(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# Backend configuration
# --------------------------------------------------------------------------


class TestBackendConfiguration:
    def test_memory_default(self):
        ledger = ClimateLedger(agent_name="t1")
        try:
            assert ledger.storage_backend == "memory"
            assert ledger.sqlite_backend is None
        finally:
            ledger.close()

    def test_sqlite_requires_path(self):
        with pytest.raises(ValueError):
            ClimateLedger(agent_name="t1", storage_backend="sqlite")

    def test_unknown_backend_rejected(self):
        with pytest.raises(ValueError):
            ClimateLedger(agent_name="t1", storage_backend="mysql")

    def test_sqlite_creates_file(self, tmp_path: Path):
        db_path = tmp_path / "ledger.sqlite"
        ledger = ClimateLedger(
            agent_name="t1", storage_backend="sqlite", sqlite_path=db_path
        )
        try:
            assert db_path.exists()
            assert ledger.sqlite_backend is not None
        finally:
            ledger.close()


# --------------------------------------------------------------------------
# Record + verify + export
# --------------------------------------------------------------------------


class TestRecordVerifyExport:
    def test_record_persists_to_sqlite(self, tmp_path: Path):
        db_path = tmp_path / "ledger.sqlite"
        ledger = ClimateLedger(
            agent_name="t1", storage_backend="sqlite", sqlite_path=db_path
        )
        try:
            h1 = ledger.record_entry("emission", "e-1", "calculate", _h("p1"))
            h2 = ledger.record_entry("emission", "e-1", "validate", _h("p2"))
            assert h1 != h2
            assert ledger.entry_count == 2

            rows = ledger.sqlite_backend.read_entity("e-1")
            assert len(rows) == 2
            assert rows[0]["chain_hash"] == h1
            assert rows[1]["chain_hash"] == h2
            assert rows[0]["operation"] == "calculate"
            assert rows[1]["operation"] == "validate"
        finally:
            ledger.close()

    def test_verify_returns_true_for_intact_chain(self, tmp_path: Path):
        db_path = tmp_path / "ledger.sqlite"
        ledger = ClimateLedger(
            agent_name="t1", storage_backend="sqlite", sqlite_path=db_path
        )
        try:
            ledger.record_entry("facility", "f-1", "ingest", _h("p1"))
            ledger.record_entry("facility", "f-1", "validate", _h("p2"))
            valid, chain = ledger.verify("f-1")
            assert valid is True
            assert len(chain) == 2
        finally:
            ledger.close()

    def test_export_single_entity(self, tmp_path: Path):
        db_path = tmp_path / "ledger.sqlite"
        ledger = ClimateLedger(
            agent_name="t1", storage_backend="sqlite", sqlite_path=db_path
        )
        try:
            ledger.record_entry("emission", "e-1", "calculate", _h("p1"))
            ledger.record_entry("emission", "e-1", "validate", _h("p2"))
            data = ledger.export(entity_id="e-1")
            assert isinstance(data, list)
            assert len(data) == 2
        finally:
            ledger.close()

    def test_export_global(self, tmp_path: Path):
        db_path = tmp_path / "ledger.sqlite"
        ledger = ClimateLedger(
            agent_name="t1", storage_backend="sqlite", sqlite_path=db_path
        )
        try:
            ledger.record_entry("emission", "e-1", "calc", _h("p1"))
            ledger.record_entry("emission", "e-2", "calc", _h("p2"))
            ledger.record_entry("facility", "f-1", "ingest", _h("p3"))
            data = ledger.export()
            assert data["entry_count"] == 3
            assert data["entity_count"] == 3
            assert data["storage_backend"] == "sqlite"
            assert len(data["entries"]) == 3
        finally:
            ledger.close()

    def test_metadata_round_trip(self, tmp_path: Path):
        db_path = tmp_path / "ledger.sqlite"
        ledger = ClimateLedger(
            agent_name="t1", storage_backend="sqlite", sqlite_path=db_path
        )
        try:
            ledger.record_entry(
                "emission", "e-1", "calculate", _h("p1"),
                metadata={"framework": "GHG Protocol", "scope": 1},
            )
            rows = ledger.sqlite_backend.read_entity("e-1")
            assert rows[0]["metadata"] == {
                "framework": "GHG Protocol",
                "scope": 1,
            }
        finally:
            ledger.close()

    def test_unknown_format_rejected(self, tmp_path: Path):
        ledger = ClimateLedger(agent_name="t1")
        try:
            with pytest.raises(ValueError):
                ledger.export(format="csv")
        finally:
            ledger.close()


# --------------------------------------------------------------------------
# Append-only enforcement
# --------------------------------------------------------------------------


class TestAppendOnlyTriggers:
    def test_update_raises(self, tmp_path: Path):
        db_path = tmp_path / "ledger.sqlite"
        backend = _SQLiteLedgerBackend(db_path)
        backend.append(
            agent_name="t1",
            entity_type="emission",
            entity_id="e-1",
            operation="calc",
            content_hash=_h("p1"),
            chain_hash="a" * 64,
        )
        with pytest.raises(sqlite3.IntegrityError):
            backend._conn.execute(
                "UPDATE climate_ledger_entries SET operation='TAMPER' WHERE id=?",
                (1,),
            )
        backend.close()

    def test_delete_raises(self, tmp_path: Path):
        db_path = tmp_path / "ledger.sqlite"
        backend = _SQLiteLedgerBackend(db_path)
        backend.append(
            agent_name="t1",
            entity_type="emission",
            entity_id="e-1",
            operation="calc",
            content_hash=_h("p1"),
            chain_hash="a" * 64,
        )
        with pytest.raises(sqlite3.IntegrityError):
            backend._conn.execute(
                "DELETE FROM climate_ledger_entries WHERE id=?", (1,)
            )
        backend.close()


# --------------------------------------------------------------------------
# Tamper detection in verify()
# --------------------------------------------------------------------------


class TestTamperDetection:
    def test_length_mismatch_returns_false(self, tmp_path: Path, monkeypatch):
        """If SQLite has fewer rows than the in-memory tracker, verify fails."""
        db_path = tmp_path / "ledger.sqlite"
        ledger = ClimateLedger(
            agent_name="t1", storage_backend="sqlite", sqlite_path=db_path
        )
        try:
            ledger.record_entry("emission", "e-1", "calc", _h("p1"))
            ledger.record_entry("emission", "e-1", "validate", _h("p2"))

            # Simulate tamper: replace the SQLite-side read so it returns
            # only one entry even though memory has two.
            monkeypatch.setattr(
                ledger.sqlite_backend,
                "read_entity",
                lambda entity_id: [
                    {
                        "id": 1,
                        "agent_name": "t1",
                        "entity_type": "emission",
                        "entity_id": "e-1",
                        "operation": "calc",
                        "content_hash": _h("p1"),
                        "chain_hash": "00" * 32,
                        "metadata": {},
                        "recorded_at": "now",
                    }
                ],
            )

            valid, _chain = ledger.verify("e-1")
            assert valid is False
        finally:
            ledger.close()


# --------------------------------------------------------------------------
# Concurrency
# --------------------------------------------------------------------------


class TestConcurrency:
    def test_concurrent_writes_do_not_crash(self, tmp_path: Path):
        """Basic smoke test: many threads appending in parallel."""
        db_path = tmp_path / "ledger.sqlite"
        ledger = ClimateLedger(
            agent_name="t1", storage_backend="sqlite", sqlite_path=db_path
        )
        try:
            def worker(idx: int):
                ledger.record_entry(
                    "emission", f"e-{idx}", "calc", _h(f"payload-{idx}")
                )

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # Each thread writes one record, so we expect exactly 20.
            assert ledger.entry_count == 20
            assert ledger.entity_count == 20
        finally:
            ledger.close()


# --------------------------------------------------------------------------
# Migration SQL sanity
# --------------------------------------------------------------------------


class TestMigrationFile:
    def test_v439_migration_present(self):
        migration = Path(
            "deployment/database/migrations/sql/V439__climate_ledger.sql"
        )
        assert migration.exists(), (
            "Climate Ledger Postgres migration missing: %s" % migration
        )
        sql = migration.read_text(encoding="utf-8")
        # Key guarantees we encode in the schema.
        assert "CREATE TABLE IF NOT EXISTS climate_ledger_entries" in sql
        assert "chain_hash" in sql
        assert "append-only" in sql
        assert "TRIGGER" in sql
