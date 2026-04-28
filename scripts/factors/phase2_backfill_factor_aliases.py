#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 2 / WS2 — Backfill factor_aliases from existing factor records.

Purpose
-------
Walk every row in the v0.1 alpha factor table (Postgres
``factors_v0_1.factor`` or SQLite ``alpha_factors_v0_1``), inspect the
``record_jsonb`` blob for a ``factor_id_alias`` value, and INSERT a row
into the ``factor_aliases`` mirror with::

    (urn, legacy_id, kind='EF')

Conflict policy
---------------
Idempotent. The ``factor_aliases`` table declares ``legacy_id UNIQUE``
(both Postgres and SQLite mirrors). Re-running this script after a
successful run is a no-op: every alias is already present and we
silently skip via ``ON CONFLICT (legacy_id) DO NOTHING`` (Postgres) or
``INSERT OR IGNORE`` (SQLite). The dry-run mode prints the proposed
INSERTs without executing.

DSN resolution
--------------
1. ``--dsn`` CLI flag (highest precedence).
2. ``GL_FACTORS_DSN`` environment variable.
3. Default: ``sqlite:///./factors_phase2.db`` (matches the WS7 alpha
   default — the same DB the publisher writes to).

Logged metrics
--------------

* ``scanned``: rows inspected in the factor table.
* ``inserted``: alias rows created on this run.
* ``skipped``: rows whose ``factor_id_alias`` was missing / blank
  (nothing to alias).
* ``conflicts``: rows whose ``factor_id_alias`` already mapped to a
  URN. The script is idempotent — conflicts are reported, not raised.

The script never updates an existing alias. Per the WS7 contract
(:meth:`AlphaFactorRepository.register_alias`), aliases are append-only;
corrections are issued as new factor records with ``supersedes_urn``,
not by mutating the alias table.

Usage
-----
.. code-block:: bash

    # Dry-run against a sqlite DB
    python scripts/factors/phase2_backfill_factor_aliases.py \
        --dsn sqlite:///./factors_phase2.db --dry-run

    # Apply against Postgres. Keep credentials outside shell history,
    # for example by injecting GL_FACTORS_DSN from a secret manager.
    GL_FACTORS_DSN=postgresql://db.example.invalid/greenlang_factors \
        python scripts/factors/phase2_backfill_factor_aliases.py
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import unquote

logger = logging.getLogger("phase2_backfill_factor_aliases")


_DEFAULT_DSN = "sqlite:///./factors_phase2.db"


# ---------------------------------------------------------------------------
# Result accumulator
# ---------------------------------------------------------------------------


@dataclass
class BackfillResult:
    """Counters returned by :func:`backfill`. Useful for tests."""

    scanned: int = 0
    inserted: int = 0
    skipped: int = 0
    conflicts: int = 0
    proposed: List[Tuple[str, str]] = field(default_factory=list)
    """List of ``(urn, legacy_id)`` tuples we would have INSERTed in
    dry-run mode. Empty when ``dry_run=False``."""


# ---------------------------------------------------------------------------
# DSN helpers (mirror the alpha repo's logic without importing it directly so
# the script also runs under a leaner test harness — e.g. when the alpha
# repository's heavy import chain is unavailable).
# ---------------------------------------------------------------------------


def _is_postgres_dsn(dsn: str) -> bool:
    return dsn.startswith("postgres://") or dsn.startswith("postgresql://")


def _resolve_sqlite_path(dsn: str) -> str:
    """Extract the SQLite path from a ``sqlite:///...`` DSN.

    Mirrors :func:`greenlang.factors.repositories.alpha_v0_1_repository.
    _resolve_sqlite_path` so the script can run in environments where
    that module is unavailable (e.g. inside a release sandbox).
    """
    raw = (dsn or "").strip()
    if not raw:
        return ":memory:"
    if raw.startswith("sqlite:"):
        body = raw[len("sqlite:"):].lstrip("/")
        if body in ("", ":memory:"):
            return ":memory:"
        if not body.startswith(":") and not (
            len(body) >= 2 and body[1] == ":"
        ):
            body = "/" + body
        return unquote(body)
    return raw


# ---------------------------------------------------------------------------
# SQLite path
# ---------------------------------------------------------------------------


def _ensure_sqlite_alias_table(conn: sqlite3.Connection) -> None:
    """Create the SQLite alias mirror table if missing (idempotent).

    The DDL matches the canonical alpha repository schema. The script
    creates the table itself so it can be invoked against a partially
    initialized DB (e.g. one that pre-dates the WS7 PR).
    """
    conn.execute(
        "CREATE TABLE IF NOT EXISTS alpha_factor_aliases_v0_1 ("
        " pk_id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " urn TEXT NOT NULL,"
        " legacy_id TEXT NOT NULL UNIQUE,"
        " kind TEXT NOT NULL DEFAULT 'EF' CHECK (kind IN ('EF','custom')),"
        " created_at TIMESTAMPTZ DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),"
        " retired_at TIMESTAMPTZ"
        ")"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_alpha_factor_aliases_v0_1_urn "
        "ON alpha_factor_aliases_v0_1(urn)"
    )


def _iter_sqlite_aliases(
    conn: sqlite3.Connection,
) -> Iterable[Tuple[str, Optional[str]]]:
    """Yield ``(urn, factor_id_alias_or_None)`` for every factor row."""
    cur = conn.execute(
        "SELECT urn, record_jsonb FROM alpha_factors_v0_1 ORDER BY urn ASC"
    )
    for row in cur:
        urn = row[0]
        try:
            blob = json.loads(row[1]) if isinstance(row[1], str) else row[1]
        except (TypeError, ValueError):
            blob = None
        if not isinstance(blob, dict):
            yield urn, None
            continue
        alias = blob.get("factor_id_alias")
        if not isinstance(alias, str) or not alias.strip():
            yield urn, None
            continue
        yield urn, alias.strip()


def _backfill_sqlite(
    dsn: str, dry_run: bool, result: BackfillResult
) -> BackfillResult:
    path = _resolve_sqlite_path(dsn)
    conn = sqlite3.connect(path, isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        _ensure_sqlite_alias_table(conn)
        # Pre-load the existing alias set so we can attribute every miss
        # to either "no alias to backfill" (skipped) or "alias already
        # backfilled" (conflicts) without a second round-trip per row.
        existing: Dict[str, str] = {}
        for r in conn.execute(
            "SELECT legacy_id, urn FROM alpha_factor_aliases_v0_1"
        ):
            existing[r["legacy_id"]] = r["urn"]

        for urn, alias in _iter_sqlite_aliases(conn):
            result.scanned += 1
            if alias is None:
                result.skipped += 1
                continue
            if alias in existing:
                # Already backfilled — idempotent no-op.
                result.conflicts += 1
                continue
            if dry_run:
                result.proposed.append((urn, alias))
                logger.info(
                    "[dry-run] would INSERT alias legacy_id=%r urn=%r kind='EF'",
                    alias, urn,
                )
                continue
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO alpha_factor_aliases_v0_1 "
                    " (urn, legacy_id, kind) VALUES (?, ?, 'EF')",
                    (urn, alias),
                )
                # SQLite's ``INSERT OR IGNORE`` succeeds silently on
                # UNIQUE conflict — count via ``changes()`` so we know
                # whether the row was actually written.
                changes = conn.execute(
                    "SELECT changes()"
                ).fetchone()[0]
                if changes >= 1:
                    result.inserted += 1
                else:
                    result.conflicts += 1
            except sqlite3.IntegrityError:
                # Race against another writer: the row appeared
                # between our existing-set load and the INSERT.
                result.conflicts += 1
    finally:
        conn.close()
    return result


# ---------------------------------------------------------------------------
# Postgres path
# ---------------------------------------------------------------------------


def _ensure_pg_alias_table_exists(conn: Any) -> None:
    """Verify the ``factors_v0_1.factor_aliases`` table exists.

    The script does NOT create this table — it is owned by Alembic
    revision V501 (WS7). If the table is missing, the migration has
    not been applied and we abort with a clear error so the operator
    knows to run Alembic first.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT to_regclass('factors_v0_1.factor_aliases')"
        )
        row = cur.fetchone()
    if row is None or row[0] is None:
        raise RuntimeError(
            "factors_v0_1.factor_aliases table is missing — apply "
            "Alembic revision V501 (WS7) before running this backfill."
        )


def _backfill_postgres(
    dsn: str, dry_run: bool, result: BackfillResult
) -> BackfillResult:
    try:
        import psycopg  # type: ignore  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - hard fail
        raise RuntimeError(
            "psycopg is required for Postgres DSNs; install via "
            "`pip install greenlang[server]`."
        ) from exc

    with psycopg.connect(dsn) as conn:  # type: ignore[arg-type]
        _ensure_pg_alias_table_exists(conn)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT urn, record_jsonb FROM factors_v0_1.factor "
                "ORDER BY urn ASC"
            )
            rows = cur.fetchall()

        # Pre-load the existing alias set for accurate counters.
        with conn.cursor() as cur:
            cur.execute(
                "SELECT legacy_id, urn FROM factors_v0_1.factor_aliases"
            )
            existing = {r[0]: r[1] for r in cur.fetchall()}

        for row in rows:
            urn = row[0]
            blob = row[1]
            if isinstance(blob, str):
                try:
                    blob = json.loads(blob)
                except ValueError:
                    blob = None
            if not isinstance(blob, dict):
                result.scanned += 1
                result.skipped += 1
                continue
            alias = blob.get("factor_id_alias")
            result.scanned += 1
            if not isinstance(alias, str) or not alias.strip():
                result.skipped += 1
                continue
            alias = alias.strip()
            if alias in existing:
                result.conflicts += 1
                continue
            if dry_run:
                result.proposed.append((urn, alias))
                logger.info(
                    "[dry-run] would INSERT alias legacy_id=%r urn=%r kind='EF'",
                    alias, urn,
                )
                continue
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO factors_v0_1.factor_aliases "
                    " (urn, legacy_id, kind) VALUES (%s, %s, 'EF') "
                    "ON CONFLICT (legacy_id) DO NOTHING",
                    (urn, alias),
                )
                # ``cur.rowcount`` is 1 on insert, 0 on conflict skip.
                if cur.rowcount == 1:
                    result.inserted += 1
                else:
                    result.conflicts += 1
        if not dry_run:
            conn.commit()
    return result


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def backfill(dsn: str, *, dry_run: bool = False) -> BackfillResult:
    """Backfill factor_aliases from the factor table.

    Args:
        dsn: ``sqlite:///path.db`` or ``postgresql://...``. A bare path
            is treated as SQLite.
        dry_run: When ``True`` log proposed INSERTs without executing.

    Returns:
        :class:`BackfillResult` with ``scanned`` / ``inserted`` /
        ``skipped`` / ``conflicts`` counters.
    """
    result = BackfillResult()
    if _is_postgres_dsn(dsn):
        return _backfill_postgres(dsn, dry_run, result)
    return _backfill_sqlite(dsn, dry_run, result)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="phase2_backfill_factor_aliases",
        description=(
            "Backfill factor_aliases from existing factor records "
            "(Phase 2 / WS2). Idempotent."
        ),
    )
    p.add_argument(
        "--dsn",
        type=str,
        default=None,
        help=(
            "Database DSN. Falls back to GL_FACTORS_DSN env var, then "
            f"to {_DEFAULT_DSN!r}."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Log proposed INSERTs without executing them.",
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return p


def _resolve_dsn(cli_dsn: Optional[str]) -> str:
    return cli_dsn or os.getenv("GL_FACTORS_DSN") or _DEFAULT_DSN


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    dsn = _resolve_dsn(args.dsn)
    logger.info(
        "phase2_backfill_factor_aliases starting dsn=%r dry_run=%s",
        dsn, args.dry_run,
    )
    result = backfill(dsn, dry_run=args.dry_run)
    logger.info(
        "phase2_backfill_factor_aliases finished | "
        "scanned=%d inserted=%d skipped=%d conflicts=%d dry_run=%s",
        result.scanned, result.inserted, result.skipped,
        result.conflicts, args.dry_run,
    )
    if args.dry_run and result.proposed:
        logger.info("dry-run proposed_count=%d", len(result.proposed))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
