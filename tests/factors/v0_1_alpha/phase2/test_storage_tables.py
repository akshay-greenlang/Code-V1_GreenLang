# -*- coding: utf-8 -*-
"""Phase 2 / WS7-T1 — canonical storage table DDL tests.

These tests do NOT require a running Postgres. They load each Phase 2
forward migration SQL file and assert that every table, column,
constraint, and index expected by the WS7 brief is textually present.

Coverage:
  * V505 — factor_aliases, source_artifacts (V501-aliases-artifacts moved
    here after the V501_additive collision; geography enum extension is
    owned by the peer V501_additive migration).
  * V503 — provenance_edges, changelog_events.
  * V504 — api_keys, entitlements, release_manifests.

A separate Postgres-driven apply test lives in
``test_alembic_up_down.py`` (gated by ``requires_postgres``).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Tuple

import pytest


REPO_ROOT = Path(__file__).resolve().parents[4]
SQL_DIR = REPO_ROOT / "deployment" / "database" / "migrations" / "sql"

V505_UP = SQL_DIR / "V505__factors_v0_1_phase2_aliases_artifacts.sql"
V505_DOWN = SQL_DIR / "V505__factors_v0_1_phase2_aliases_artifacts_DOWN.sql"
V503_UP = SQL_DIR / "V503__factors_v0_1_phase2_provenance_changelog.sql"
V503_DOWN = SQL_DIR / "V503__factors_v0_1_phase2_provenance_changelog_DOWN.sql"
V504_UP = SQL_DIR / "V504__factors_v0_1_phase2_apikeys_entitlements_releases.sql"
V504_DOWN = SQL_DIR / "V504__factors_v0_1_phase2_apikeys_entitlements_releases_DOWN.sql"


# ---------------------------------------------------------------------------
# File-existence smoke checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [V505_UP, V505_DOWN, V503_UP, V503_DOWN, V504_UP, V504_DOWN],
    ids=lambda p: p.name,
)
def test_migration_file_exists(path: Path) -> None:
    assert path.exists(), f"Missing Phase 2 migration: {path}"
    assert path.stat().st_size > 0, f"Empty migration file: {path}"


# ---------------------------------------------------------------------------
# V505 — factor_aliases + source_artifacts
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def v505_up_sql() -> str:
    return V505_UP.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def v505_down_sql() -> str:
    return V505_DOWN.read_text(encoding="utf-8")


def test_v505_creates_factor_aliases_table(v505_up_sql: str) -> None:
    assert "CREATE TABLE factors_v0_1.factor_aliases" in v505_up_sql
    # Required columns
    for col in ("pk_id", "urn", "legacy_id", "kind", "created_at", "retired_at"):
        assert re.search(rf"\b{col}\b", v505_up_sql), f"missing column {col}"
    # FK on urn -> factor(urn)
    assert "REFERENCES factors_v0_1.factor(urn)" in v505_up_sql
    # legacy_id UNIQUE
    assert re.search(r"legacy_id\s+TEXT\s+NOT NULL\s+UNIQUE", v505_up_sql)
    # kind CHECK
    assert "CHECK (kind IN ('EF', 'custom'))" in v505_up_sql
    # Index on urn
    assert "factor_aliases_urn_idx" in v505_up_sql


def test_v505_creates_source_artifacts_table(v505_up_sql: str) -> None:
    assert "CREATE TABLE factors_v0_1.source_artifacts" in v505_up_sql
    for col in (
        "pk_id", "sha256", "source_urn", "source_version", "uri",
        "content_type", "size_bytes", "parser_id", "parser_version",
        "parser_commit", "ingested_at", "metadata",
    ):
        assert re.search(rf"\b{col}\b", v505_up_sql), f"missing column {col}"
    # sha256 64-char lowercase hex CHECK
    assert "CHECK (sha256 ~ '^[a-f0-9]{64}$')" in v505_up_sql
    # FK to source(urn)
    assert "REFERENCES factors_v0_1.source(urn)" in v505_up_sql
    # size_bytes positive CHECK
    assert "size_bytes IS NULL OR size_bytes > 0" in v505_up_sql
    # Both expected indexes
    assert "source_artifacts_source_idx" in v505_up_sql
    assert "source_artifacts_version_idx" in v505_up_sql


def test_v505_does_not_extend_geography_enum(v505_up_sql: str) -> None:
    """V505 must not duplicate the geography enum extension (owned by V501_additive)."""
    # Header comments may name the constraints for documentation; check only
    # for *executable SQL* that touches them.
    code_lines = "\n".join(
        line for line in v505_up_sql.splitlines() if not line.lstrip().startswith("--")
    )
    assert "geography_type_check" not in code_lines, (
        "V505 must not redefine geography_type_check (V501_additive owns it)"
    )
    assert "geography_urn_pattern" not in code_lines, (
        "V505 must not redefine geography_urn_pattern (V501_additive owns it)"
    )
    assert "ALTER TABLE factors_v0_1.geography" not in code_lines, (
        "V505 must not ALTER the geography table"
    )


def test_v505_down_drops_phase2_tables(v505_down_sql: str) -> None:
    assert "DROP TABLE IF EXISTS factors_v0_1.source_artifacts" in v505_down_sql
    assert "DROP TABLE IF EXISTS factors_v0_1.factor_aliases" in v505_down_sql
    # DOWN must NOT touch the geography constraints (V501_additive owns those).
    code_lines = "\n".join(
        line for line in v505_down_sql.splitlines() if not line.lstrip().startswith("--")
    )
    assert "geography_type_check" not in code_lines
    assert "geography_urn_pattern" not in code_lines
    assert "ALTER TABLE factors_v0_1.geography" not in code_lines


# ---------------------------------------------------------------------------
# V503 — provenance_edges + changelog_events
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def v503_up_sql() -> str:
    return V503_UP.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def v503_down_sql() -> str:
    return V503_DOWN.read_text(encoding="utf-8")


def test_v503_creates_provenance_edges_table(v503_up_sql: str) -> None:
    assert "CREATE TABLE factors_v0_1.provenance_edges" in v503_up_sql
    for col in (
        "pk_id", "factor_urn", "source_artifact_pk", "row_ref",
        "edge_type", "created_at",
    ):
        assert re.search(rf"\b{col}\b", v503_up_sql), f"missing column {col}"
    # FKs
    assert "REFERENCES factors_v0_1.factor(urn)" in v503_up_sql
    assert "REFERENCES factors_v0_1.source_artifacts(pk_id)" in v503_up_sql
    # edge_type CHECK
    for et in ("extraction", "derivation", "correction", "supersedes"):
        assert f"'{et}'" in v503_up_sql
    # UNIQUE composite
    assert (
        "UNIQUE (factor_urn, source_artifact_pk, row_ref, edge_type)"
        in v503_up_sql
    )
    # Indexes
    assert "provenance_edges_factor_idx" in v503_up_sql
    assert "provenance_edges_artifact_idx" in v503_up_sql


def test_v503_creates_changelog_events_table(v503_up_sql: str) -> None:
    assert "CREATE TABLE factors_v0_1.changelog_events" in v503_up_sql
    for col in (
        "pk_id", "event_type", "schema_version", "subject_urn",
        "change_class", "migration_note_uri", "actor", "occurred_at",
        "metadata",
    ):
        assert re.search(rf"\b{col}\b", v503_up_sql), f"missing column {col}"
    # event_type CHECK with all 8 values
    for et in (
        "schema_change", "factor_publish", "factor_supersede",
        "factor_deprecate", "source_add", "source_deprecate",
        "pack_release", "migration_apply",
    ):
        assert f"'{et}'" in v503_up_sql
    # change_class CHECK
    for cc in ("additive", "breaking", "deprecated", "removed"):
        assert f"'{cc}'" in v503_up_sql
    # Indexes
    assert "changelog_events_subject_idx" in v503_up_sql
    assert "changelog_events_type_idx" in v503_up_sql


def test_v503_down_drops_in_reverse_order(v503_down_sql: str) -> None:
    # changelog_events drops first (no inbound FK), then provenance_edges.
    chg_pos = v503_down_sql.find("DROP TABLE IF EXISTS factors_v0_1.changelog_events")
    prov_pos = v503_down_sql.find("DROP TABLE IF EXISTS factors_v0_1.provenance_edges")
    assert chg_pos > 0 and prov_pos > 0
    assert chg_pos < prov_pos, "DOWN should drop changelog_events before provenance_edges"


# ---------------------------------------------------------------------------
# V504 — api_keys + entitlements + release_manifests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def v504_up_sql() -> str:
    return V504_UP.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def v504_down_sql() -> str:
    return V504_DOWN.read_text(encoding="utf-8")


def test_v504_creates_api_keys_table(v504_up_sql: str) -> None:
    assert "CREATE TABLE factors_v0_1.api_keys" in v504_up_sql
    for col in (
        "pk_id", "key_hash", "tenant", "scopes", "created_at",
        "last_used_at", "revoked_at", "metadata",
    ):
        assert re.search(rf"\b{col}\b", v504_up_sql), f"missing column {col}"
    # key_hash UNIQUE
    assert re.search(r"key_hash\s+TEXT\s+NOT NULL\s+UNIQUE", v504_up_sql)
    # scopes is a TEXT[] array
    assert "TEXT[]" in v504_up_sql
    # Partial index on active keys
    assert "api_keys_tenant_idx" in v504_up_sql
    assert "WHERE revoked_at IS NULL" in v504_up_sql


def test_v504_creates_entitlements_table(v504_up_sql: str) -> None:
    assert "CREATE TABLE factors_v0_1.entitlements" in v504_up_sql
    for col in (
        "pk_id", "tenant", "source_urn", "granted_at", "expires_at",
        "terms_uri", "metadata",
    ):
        assert re.search(rf"\b{col}\b", v504_up_sql), f"missing column {col}"
    # FK to source(urn)
    assert "REFERENCES factors_v0_1.source(urn)" in v504_up_sql
    # UNIQUE (tenant, source_urn)
    assert "UNIQUE (tenant, source_urn)" in v504_up_sql
    # Indexes
    assert "entitlements_tenant_idx" in v504_up_sql
    assert "entitlements_source_idx" in v504_up_sql


def test_v504_creates_release_manifests_table(v504_up_sql: str) -> None:
    assert "CREATE TABLE factors_v0_1.release_manifests" in v504_up_sql
    for col in (
        "pk_id", "release_id", "factor_urns", "schema_version",
        "signature", "released_at", "released_by", "metadata",
    ):
        assert re.search(rf"\b{col}\b", v504_up_sql), f"missing column {col}"
    # release_id UNIQUE
    assert re.search(r"release_id\s+TEXT\s+NOT NULL\s+UNIQUE", v504_up_sql)
    # factor_urns array
    assert re.search(r"factor_urns\s+TEXT\[\]\s+NOT NULL", v504_up_sql)
    # Index on released_at DESC
    assert "release_manifests_released_at_idx" in v504_up_sql


def test_v504_documents_sec001_linkage(v504_up_sql: str) -> None:
    """Header must document why we don't reuse public.api_keys (SEC-001)."""
    # Header should mention SEC-001 / public.api_keys explicitly so future
    # readers understand the deliberate non-reuse.
    assert "SEC-001" in v504_up_sql or "public.api_keys" in v504_up_sql


def test_v504_down_drops_all_three_tables(v504_down_sql: str) -> None:
    for tbl in ("release_manifests", "entitlements", "api_keys"):
        assert (
            f"DROP TABLE IF EXISTS factors_v0_1.{tbl}" in v504_down_sql
        ), f"V504 DOWN missing drop for {tbl}"


# ---------------------------------------------------------------------------
# Cross-migration invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [V505_UP, V503_UP, V504_UP],
    ids=lambda p: p.name,
)
def test_no_v500_table_redefined(path: Path) -> None:
    """Phase 2 forward migrations must never redefine V500 tables."""
    sql = path.read_text(encoding="utf-8")
    for v500_tbl in (
        "source", "methodology", "geography", "unit", "factor_pack",
        "factor", "factor_publish_log",
    ):
        # Allow REFERENCES factors_v0_1.<table> (FKs are fine), but never
        # CREATE TABLE on a V500 table.
        bad = re.search(
            rf"CREATE\s+TABLE\s+(IF\s+NOT\s+EXISTS\s+)?factors_v0_1\.{v500_tbl}\b",
            sql,
            re.IGNORECASE,
        )
        assert bad is None, (
            f"{path.name} redefines V500 table {v500_tbl!r}; that's forbidden"
        )


@pytest.mark.parametrize(
    "up_path,down_path",
    [
        (V505_UP, V505_DOWN),
        (V503_UP, V503_DOWN),
        (V504_UP, V504_DOWN),
    ],
    ids=["V505", "V503", "V504"],
)
def test_down_drops_every_table_up_creates(
    up_path: Path, down_path: Path
) -> None:
    """Every CREATE TABLE in UP must have a matching DROP TABLE in DOWN."""
    up_sql = up_path.read_text(encoding="utf-8")
    down_sql = down_path.read_text(encoding="utf-8")
    created = re.findall(
        r"CREATE\s+TABLE\s+factors_v0_1\.(\w+)",
        up_sql,
        re.IGNORECASE,
    )
    for tbl in created:
        assert (
            re.search(
                rf"DROP\s+TABLE\s+IF\s+EXISTS\s+factors_v0_1\.{tbl}\b",
                down_sql,
                re.IGNORECASE,
            )
            is not None
        ), f"DOWN {down_path.name} missing DROP TABLE for {tbl}"
