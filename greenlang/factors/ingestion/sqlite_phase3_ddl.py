# -*- coding: utf-8 -*-
"""SQLite mirror DDL for Phase 3 ingestion tables (V507 + V508).

This module is the SQLite parity counterpart of the Postgres migrations:

- ``deployment/database/migrations/sql/V507__factors_v0_1_phase3_ingestion_runs.sql``
- ``deployment/database/migrations/sql/V508__factors_v0_1_phase3_ingestion_run_diffs.sql``

It exposes inline ``CREATE TABLE`` / ``CREATE INDEX`` strings + a small
helper that applies them on a ``sqlite3.Connection``. Tests that run the
ingestion pipeline against an in-memory SQLite database use this module
to materialise the same logical schema the Postgres alembic chain
produces, with the following sqlite-vs-postgres translations applied:

================================  ===========================================
Postgres                          SQLite mirror
================================  ===========================================
``UUID``                          ``TEXT`` (caller stringifies UUIDs)
``ENUM ingestion_run_status``     ``TEXT`` + ``CHECK (status IN (...))``
``ENUM ingestion_diff_kind``      ``TEXT`` + ``CHECK (kind IN (...))``
``BIGSERIAL``                     ``INTEGER PRIMARY KEY AUTOINCREMENT``
``TIMESTAMPTZ DEFAULT now()``     ``TEXT`` ISO-8601 default via strftime
``JSONB``                         ``TEXT`` (caller serialises with json.dumps)
Trigger ``set_last_updated_at``   sqlite trigger ``BEFORE UPDATE``
================================  ===========================================

The ``CHECK`` constraints reproduce the operator/approver-policy guards
exactly: bots cannot approve, terminal-publish requires an approver, and
``current_stage`` is bounded to the seven canonical stage names.

The follow-on ingestion-package modules (run-state machine, CLI,
pipeline orchestrator) call :func:`apply_phase3_ddl` once per test fixture
to bootstrap a fresh SQLite database in lockstep with what the Postgres
chain produces after ``alembic upgrade head``.

Wave: Phase 3 / Wave 1.0 / TaskCreate #27
Author: GL-BackendDeveloper
Created: 2026-04-28
"""

from __future__ import annotations

import sqlite3
from typing import Tuple


__all__ = [
    "PHASE3_DDL_STATEMENTS",
    "PHASE3_INDEX_STATEMENTS",
    "PHASE3_TRIGGER_STATEMENTS",
    "INGESTION_RUN_STATUS_VALUES",
    "INGESTION_DIFF_KIND_VALUES",
    "INGESTION_CURRENT_STAGE_VALUES",
    "apply_phase3_ddl",
    "apply_v509_reviewer_notes_column",
]


# ---------------------------------------------------------------------------
# ENUM mirrors — kept as Python tuples so callers can validate values
# in-process without round-tripping through SQLite.
# ---------------------------------------------------------------------------

#: Mirror of ``factors_v0_1.ingestion_run_status`` (V507).
INGESTION_RUN_STATUS_VALUES: Tuple[str, ...] = (
    "created",
    "fetched",
    "parsed",
    "normalized",
    "validated",
    "deduped",
    "staged",
    "review_required",
    "published",
    "rejected",
    "failed",
    "rolled_back",
)

#: Mirror of the ``current_stage`` CHECK whitelist (V507).
INGESTION_CURRENT_STAGE_VALUES: Tuple[str, ...] = (
    "fetch",
    "parse",
    "normalize",
    "validate",
    "dedupe",
    "stage",
    "publish",
)

#: Mirror of ``factors_v0_1.ingestion_diff_kind`` (V508).
INGESTION_DIFF_KIND_VALUES: Tuple[str, ...] = (
    "added",
    "removed",
    "changed",
    "supersedes",
    "unchanged",
    "parser_version_change",
    "licence_change",
    "methodology_change",
    "removal_candidate",
)


def _quote_csv(values: Tuple[str, ...]) -> str:
    """Render an enum tuple as a SQL ``IN (...)`` literal list."""
    return ",".join(f"'{v}'" for v in values)


# ---------------------------------------------------------------------------
# Table DDL
#
# Naming convention mirrors the rest of the SQLite repository surface:
# ``alpha_<table>_v0_1`` (see alpha_v0_1_repository._SQLITE_DDL). The
# Postgres-side names are unprefixed (factors_v0_1.ingestion_runs); the
# repository / pipeline layer maps between the two when reading.
# ---------------------------------------------------------------------------

# V507 mirror — ingestion_runs (UUID -> TEXT, ENUM -> TEXT + CHECK).
_INGESTION_RUNS_DDL = (
    "CREATE TABLE IF NOT EXISTS alpha_ingestion_runs_v0_1 ("
    " run_id TEXT PRIMARY KEY,"
    " source_urn TEXT NOT NULL,"
    " source_version TEXT NOT NULL,"
    " started_at TIMESTAMPTZ NOT NULL"
    "  DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),"
    " status TEXT NOT NULL DEFAULT 'created'"
    f"  CHECK (status IN ({_quote_csv(INGESTION_RUN_STATUS_VALUES)})),"
    " current_stage TEXT"
    "  CHECK (current_stage IS NULL OR current_stage IN"
    f"   ({_quote_csv(INGESTION_CURRENT_STAGE_VALUES)})),"
    " artifact_pk_id INTEGER,"
    " parser_module TEXT,"
    " parser_function TEXT,"
    " parser_version TEXT,"
    " parser_commit TEXT,"
    " operator TEXT NOT NULL"
    # SQLite supports glob-like CHECK patterns via LIKE; we keep the same
    # bot:<id> | human:<email> shape but use a permissive LIKE that the
    # Python operator-validation helper hardens with a strict regex.
    "  CHECK ("
    "   operator LIKE 'bot:%'"
    "   OR (operator LIKE 'human:%@%.%')"
    "  ),"
    " batch_id TEXT,"
    " approved_by TEXT"
    "  CHECK (approved_by IS NULL OR approved_by LIKE 'human:%@%.%'),"
    " approved_at TIMESTAMPTZ,"
    " diff_json_uri TEXT,"
    " diff_md_uri TEXT,"
    " accepted_count INTEGER,"
    " rejected_count INTEGER,"
    " supersedes_count INTEGER,"
    " unchanged_count INTEGER,"
    " removal_candidate_count INTEGER,"
    " error_json TEXT,"
    " last_updated_at TIMESTAMPTZ NOT NULL"
    "  DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),"
    # Terminal-publish requires an approver. Mirrors V507 SQL CHECK.
    " CHECK ("
    "  status NOT IN ('published','rolled_back')"
    "  OR approved_by IS NOT NULL"
    " )"
    ")"
)

# V508 mirror — ingestion_run_diffs (one row per diff entry).
# Note: factor_urn intentionally has no FK (added/removed/supersedes-preview
# entries can reference a not-yet-published or removed factor — same as
# the Postgres side).
_INGESTION_RUN_DIFFS_DDL = (
    "CREATE TABLE IF NOT EXISTS alpha_ingestion_run_diffs_v0_1 ("
    " pk_id INTEGER PRIMARY KEY AUTOINCREMENT,"
    " run_id TEXT NOT NULL"
    "  REFERENCES alpha_ingestion_runs_v0_1(run_id) ON DELETE CASCADE,"
    " kind TEXT NOT NULL"
    f"  CHECK (kind IN ({_quote_csv(INGESTION_DIFF_KIND_VALUES)})),"
    " factor_urn TEXT NOT NULL,"
    " prior_factor_urn TEXT,"
    " attribute_changed TEXT,"
    " prior_value TEXT,"
    " new_value TEXT,"
    " generated_at TIMESTAMPTZ NOT NULL"
    "  DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))"
    ")"
)

#: Tuple of ``CREATE TABLE`` statements applied in order.
PHASE3_DDL_STATEMENTS: Tuple[str, ...] = (
    _INGESTION_RUNS_DDL,
    _INGESTION_RUN_DIFFS_DDL,
)


# ---------------------------------------------------------------------------
# Indexes — operator-query workload (mirrors the Postgres-side indexes).
# ---------------------------------------------------------------------------

PHASE3_INDEX_STATEMENTS: Tuple[str, ...] = (
    "CREATE INDEX IF NOT EXISTS ix_alpha_ingestion_runs_v0_1_source"
    " ON alpha_ingestion_runs_v0_1(source_urn)",
    "CREATE INDEX IF NOT EXISTS ix_alpha_ingestion_runs_v0_1_status"
    " ON alpha_ingestion_runs_v0_1(status)",
    "CREATE INDEX IF NOT EXISTS ix_alpha_ingestion_runs_v0_1_batch"
    " ON alpha_ingestion_runs_v0_1(batch_id)"
    " WHERE batch_id IS NOT NULL",
    "CREATE INDEX IF NOT EXISTS ix_alpha_ingestion_runs_v0_1_started_at"
    " ON alpha_ingestion_runs_v0_1(started_at DESC)",
    "CREATE INDEX IF NOT EXISTS ix_alpha_ingestion_run_diffs_v0_1_run_kind"
    " ON alpha_ingestion_run_diffs_v0_1(run_id, kind)",
    "CREATE INDEX IF NOT EXISTS ix_alpha_ingestion_run_diffs_v0_1_factor"
    " ON alpha_ingestion_run_diffs_v0_1(factor_urn)",
)


# ---------------------------------------------------------------------------
# Trigger — last_updated_at maintenance on UPDATE (parity with V507's
# Postgres BEFORE-UPDATE trigger).
# ---------------------------------------------------------------------------

PHASE3_TRIGGER_STATEMENTS: Tuple[str, ...] = (
    "CREATE TRIGGER IF NOT EXISTS"
    " alpha_ingestion_runs_v0_1_last_updated_at_trg"
    " AFTER UPDATE ON alpha_ingestion_runs_v0_1"
    " FOR EACH ROW"
    " BEGIN"
    "  UPDATE alpha_ingestion_runs_v0_1"
    "   SET last_updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')"
    "   WHERE run_id = NEW.run_id;"
    " END",
)


# ---------------------------------------------------------------------------
# Helper — apply the full Phase 3 SQLite mirror in one call.
# ---------------------------------------------------------------------------


def apply_phase3_ddl(conn: sqlite3.Connection) -> None:
    """Create the Phase 3 ingestion tables, indexes, and trigger on *conn*.

    This is idempotent: every statement uses ``IF NOT EXISTS``. Foreign
    keys are turned on for the connection so the CASCADE delete from
    ``alpha_ingestion_runs_v0_1`` -> ``alpha_ingestion_run_diffs_v0_1``
    behaves the same way as it does on Postgres.

    Args:
        conn: An open :class:`sqlite3.Connection`. The caller owns the
            connection lifecycle; this helper neither commits nor closes.
    """
    # SQLite needs PRAGMA foreign_keys = ON for FK enforcement. This is a
    # connection-level setting; calling it here keeps test fixtures terse.
    conn.execute("PRAGMA foreign_keys = ON")

    for ddl in PHASE3_DDL_STATEMENTS:
        conn.execute(ddl)
    for idx in PHASE3_INDEX_STATEMENTS:
        conn.execute(idx)
    for trg in PHASE3_TRIGGER_STATEMENTS:
        conn.execute(trg)

    # Phase 3 / Wave 2.5 — V509 mirror: add ``reviewer_notes`` to the
    # SQLite ``alpha_source_artifacts_v0_1`` table if it exists. The table
    # is created by :class:`AlphaFactorRepository._SQLITE_DDL` (V501
    # mirror); if a test fixture has already materialised it, we ALTER
    # the column in. Idempotent — duplicate-column errors are tolerated.
    apply_v509_reviewer_notes_column(conn)


# ---------------------------------------------------------------------------
# V509 mirror helper — additive ``reviewer_notes TEXT`` (JSON-as-text on
# SQLite) on the existing ``alpha_source_artifacts_v0_1`` table.
# ---------------------------------------------------------------------------


def apply_v509_reviewer_notes_column(conn: sqlite3.Connection) -> None:
    """Add ``reviewer_notes TEXT`` to ``alpha_source_artifacts_v0_1`` (idempotent).

    This is the SQLite mirror of the V509 Postgres migration
    (``ALTER TABLE factors_v0_1.source_artifacts ADD COLUMN reviewer_notes
    JSONB``). On SQLite, JSONB is stored as ``TEXT`` (callers JSON-encode
    on write, JSON-decode on read).

    Behaviour:
      * If the underlying ``alpha_source_artifacts_v0_1`` table does not
        exist yet (e.g. fixture order varies), this is a no-op. The table
        is created later by :class:`AlphaFactorRepository`'s DDL bundle;
        the caller is responsible for re-applying this helper after that
        DDL runs if it needs the column for the current test run.
      * If the column already exists, the duplicate-column ``OperationalError``
        is swallowed.

    Args:
        conn: An open :class:`sqlite3.Connection`. The caller owns the
            connection lifecycle; this helper neither commits nor closes.
    """
    try:
        # Fast path — only ALTER if the table is present. Otherwise the
        # ALTER TABLE fails with "no such table" which is noisy in test
        # logs even though we'd swallow it.
        cur = conn.execute(
            "SELECT 1 FROM sqlite_master "
            "WHERE type = 'table' AND name = 'alpha_source_artifacts_v0_1'"
        )
        if cur.fetchone() is None:
            return
    except Exception:  # noqa: BLE001 — sqlite_master access is best-effort
        return

    try:
        conn.execute(
            "ALTER TABLE alpha_source_artifacts_v0_1 "
            "ADD COLUMN reviewer_notes TEXT"
        )
    except sqlite3.OperationalError as exc:
        # SQLite raises OperationalError("duplicate column name: ...") if
        # the column already exists. Tolerate that one; re-raise anything
        # else so a real schema error surfaces.
        if "duplicate column" not in str(exc).lower():
            raise
