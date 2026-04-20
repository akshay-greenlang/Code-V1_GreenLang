# -*- coding: utf-8 -*-
"""Tests for Phase 2.2 Evidence Vault hardening."""
from __future__ import annotations

import io
import json
import sqlite3
import zipfile
from pathlib import Path

import pytest

from greenlang.evidence_vault import EvidenceVault


# --------------------------------------------------------------------------
# Collection + verify
# --------------------------------------------------------------------------


class TestMemoryBackend:
    def test_collect_and_verify(self):
        vault = EvidenceVault("unit-test")
        eid = vault.collect(
            "emission_factor", "scope1_agent", {"co2e": 42.0},
            case_id="CBAM-2025-Q1",
        )
        ok, details = vault.verify(eid)
        assert ok is True
        assert details["stored_hash"] == details["computed_hash"]

    def test_case_filter(self):
        vault = EvidenceVault("unit-test")
        a = vault.collect("ef", "x", {"v": 1}, case_id="A")
        b = vault.collect("ef", "x", {"v": 2}, case_id="B")
        c = vault.collect("ef", "x", {"v": 3}, case_id="A")
        case_a = vault.list_evidence(case_id="A")
        assert {r["evidence_id"] for r in case_a} == {a, c}
        assert vault.list_cases() == ["A", "B"]

    def test_unknown_backend_rejected(self):
        with pytest.raises(ValueError):
            EvidenceVault("v", storage="s3")


# --------------------------------------------------------------------------
# Attachments
# --------------------------------------------------------------------------


class TestAttachments:
    def test_attach_stores_and_hashes(self):
        vault = EvidenceVault("unit-test")
        eid = vault.collect("invoice", "erp", {"amount": 100}, case_id="CASE1")
        content = b"RAW PARSER LOG\nline 2\n"
        content_hash = vault.attach(eid, "parser.log", content)
        assert len(content_hash) == 64  # sha256 hex

        atts = vault._attachments_for(eid)
        assert len(atts) == 1
        assert atts[0]["filename"] == "parser.log"
        assert atts[0]["content_hash"] == content_hash

    def test_attach_unknown_evidence_raises(self):
        vault = EvidenceVault("unit-test")
        with pytest.raises(KeyError):
            vault.attach("nonexistent", "f.txt", b"x")


# --------------------------------------------------------------------------
# Bundle (signed ZIP)
# --------------------------------------------------------------------------


class TestBundle:
    def test_bundle_contains_manifest_records_signature(self, tmp_path: Path):
        vault = EvidenceVault("csrd-fy25")
        eid = vault.collect(
            "emission_factor", "scope1_agent",
            {"factor_id": "ef.001", "co2e_kg_per_unit": 2.31},
            case_id="CBAM-2025-Q1",
        )
        vault.attach(eid, "source.xml", b"<cbam_xml/>")

        bundle_path = vault.bundle(
            output_path=tmp_path / "bundle.zip",
            case_id="CBAM-2025-Q1",
        )
        assert bundle_path.exists()

        with zipfile.ZipFile(bundle_path) as zf:
            names = set(zf.namelist())
            assert "manifest.json" in names
            assert "signature.json" in names
            assert any(n.startswith("records/") and n.endswith(".json") for n in names)
            assert any(n.startswith("attachments/") for n in names)

            manifest = json.loads(zf.read("manifest.json"))
            sig = json.loads(zf.read("signature.json"))

        assert manifest["case_id"] == "CBAM-2025-Q1"
        assert manifest["record_count"] == 1
        assert manifest["attachment_count"] == 1
        assert sig["algorithm"] == "sha256"
        assert len(sig["manifest_hash"]) == 64

    def test_bundle_requires_case_or_ids(self, tmp_path: Path):
        vault = EvidenceVault("v1")
        with pytest.raises(ValueError):
            vault.bundle(tmp_path / "b.zip")

    def test_bundle_empty_case_raises(self, tmp_path: Path):
        vault = EvidenceVault("v1")
        with pytest.raises(ValueError):
            vault.bundle(tmp_path / "b.zip", case_id="NOPE")

    def test_bundle_deterministic_manifest(self, tmp_path: Path):
        """Same evidence + same case_id produce manifests with identical hashes."""
        vault = EvidenceVault("v1")
        eid = vault.collect("ef", "x", {"v": 1}, case_id="CASE")

        b1 = vault.bundle(tmp_path / "b1.zip", case_id="CASE")
        b2 = vault.bundle(tmp_path / "b2.zip", case_id="CASE")

        with zipfile.ZipFile(b1) as z1, zipfile.ZipFile(b2) as z2:
            m1 = json.loads(z1.read("manifest.json"))
            m2 = json.loads(z2.read("manifest.json"))
        # record metadata (evidence_id, content_hash, source, case_id) is deterministic
        m1_record = {k: v for k, v in m1["records"][0].items()}
        m2_record = {k: v for k, v in m2["records"][0].items()}
        assert m1_record == m2_record


# --------------------------------------------------------------------------
# SQLite backend
# --------------------------------------------------------------------------


class TestSQLiteBackend:
    def test_sqlite_requires_path(self):
        with pytest.raises(ValueError):
            EvidenceVault("v1", storage="sqlite")

    def test_sqlite_persists(self, tmp_path: Path):
        db_path = tmp_path / "vault.sqlite"
        vault = EvidenceVault("v1", storage="sqlite", sqlite_path=db_path)
        try:
            eid = vault.collect("invoice", "erp", {"amt": 100}, case_id="CASE")
            assert db_path.exists()
        finally:
            vault.close()

        # New vault instance can read back
        vault2 = EvidenceVault("v1", storage="sqlite", sqlite_path=db_path)
        try:
            records = vault2.list_evidence()
            assert len(records) == 1
            assert records[0]["evidence_id"] == eid
            ok, _ = vault2.verify(eid)
            assert ok is True
        finally:
            vault2.close()

    def test_sqlite_attachments_persist(self, tmp_path: Path):
        db_path = tmp_path / "vault.sqlite"
        vault = EvidenceVault("v1", storage="sqlite", sqlite_path=db_path)
        try:
            eid = vault.collect("invoice", "erp", {"amt": 100}, case_id="CASE")
            vault.attach(eid, "raw.pdf", b"PDFDATA")
        finally:
            vault.close()

        vault2 = EvidenceVault("v1", storage="sqlite", sqlite_path=db_path)
        try:
            atts = vault2._attachments_for(eid)
            assert len(atts) == 1
            assert atts[0]["filename"] == "raw.pdf"
            assert atts[0]["content_bytes"] == b"PDFDATA"
        finally:
            vault2.close()

    def test_sqlite_bundle_after_reload(self, tmp_path: Path):
        db_path = tmp_path / "vault.sqlite"
        vault = EvidenceVault("v1", storage="sqlite", sqlite_path=db_path)
        try:
            eid = vault.collect("invoice", "erp", {"amt": 100}, case_id="CASE")
            vault.attach(eid, "raw.pdf", b"PDFDATA")
        finally:
            vault.close()

        vault2 = EvidenceVault("v1", storage="sqlite", sqlite_path=db_path)
        try:
            out = vault2.bundle(tmp_path / "bundle.zip", case_id="CASE")
        finally:
            vault2.close()

        with zipfile.ZipFile(out) as zf:
            manifest = json.loads(zf.read("manifest.json"))
        assert manifest["record_count"] == 1
        assert manifest["attachment_count"] == 1

    def test_sqlite_append_only_update_fails(self, tmp_path: Path):
        db_path = tmp_path / "vault.sqlite"
        vault = EvidenceVault("v1", storage="sqlite", sqlite_path=db_path)
        try:
            vault.collect("invoice", "erp", {"amt": 100})
            with pytest.raises(sqlite3.IntegrityError):
                vault.sqlite_backend._conn.execute(
                    "UPDATE evidence_records SET source='TAMPER'"
                )
        finally:
            vault.close()


# --------------------------------------------------------------------------
# Migration SQL sanity
# --------------------------------------------------------------------------


class TestMigrationFile:
    def test_v440_migration_present(self):
        mig = Path("deployment/database/migrations/sql/V440__evidence_vault.sql")
        assert mig.exists()
        sql = mig.read_text(encoding="utf-8")
        assert "CREATE TABLE IF NOT EXISTS evidence_records" in sql
        assert "CREATE TABLE IF NOT EXISTS evidence_attachments" in sql
        assert "append-only" in sql
        assert "TRIGGER" in sql
