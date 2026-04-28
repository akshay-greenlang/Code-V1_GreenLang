# -*- coding: utf-8 -*-
"""Methodology ontology seed loader (Phase 2 / WS6).

Reads :file:`greenlang/factors/data/ontology/methodology_seed_v0_1.yaml`
and idempotently inserts each row into the ``factors_v0_1.methodology``
table created by V500. Uses the V501 ``seed_source`` column to mark
Phase 2 seed rows so the downgrade path can roll them back without
disturbing production-ingested rows.

Every row's ``urn`` is validated through
:func:`greenlang.factors.ontology.urn.parse` before SQL is issued.

Public surface:
    * :func:`load_methodologies` -- idempotent insert against any DB-API
      2.0 connection (psycopg, psycopg2, sqlite3).
    * :func:`load_seed_yaml` -- parse + validate the YAML, return rows.
    * :func:`create_sqlite_methodology_table` -- mirror of the V500
      Postgres DDL for sqlite-backed unit tests.
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence

import yaml

from greenlang.factors.data.ontology.loaders._common import (
    LoadReport,
    PHASE2_SEED_SOURCE,
    is_sqlite_connection,
)
from greenlang.factors.ontology.urn import InvalidUrnError, parse

logger = logging.getLogger(__name__)

__all__ = [
    "MethodologyRow",
    "MethodologyLoaderError",
    "METHODOLOGY_SEED_PATH",
    "ALLOWED_APPROACHES",
    "create_sqlite_methodology_table",
    "load_seed_yaml",
    "load_methodologies",
]

_THIS_DIR = Path(__file__).resolve().parent
METHODOLOGY_SEED_PATH: Path = (
    _THIS_DIR.parent / "methodology_seed_v0_1.yaml"
)

# Mirror of the V500 CHECK on methodology.approach. NULL is also a
# legal value at the column level (CHECK approach IS NULL OR
# approach IN (...)) — we encode that separately.
ALLOWED_APPROACHES = frozenset(
    [
        "activity-based",
        "spend-based",
        "supplier-specific",
        "hybrid",
        "measurement-based",
    ]
)


class MethodologyLoaderError(ValueError):
    """Raised when the seed YAML cannot be parsed/validated."""


@dataclass(frozen=True)
class MethodologyRow:
    """One canonical methodology row.

    Mirrors the columns of ``factors_v0_1.methodology`` 1:1 (after V501).
    """

    urn: str
    name: str
    framework: str
    tier: Optional[str]
    approach: Optional[str]
    boundary_template: Optional[str]
    allocation_rules: Optional[str]
    notes: Optional[str]
    seed_source: str = PHASE2_SEED_SOURCE


# ---------------------------------------------------------------------------
# Seed parsing + URN validation
# ---------------------------------------------------------------------------


def _validate_row(raw: Any, idx: int) -> MethodologyRow:
    if not isinstance(raw, dict):
        raise MethodologyLoaderError(
            f"methodology row #{idx} is not a mapping: {raw!r}"
        )
    required = ("urn", "name", "framework")
    missing = [k for k in required if k not in raw or raw[k] in (None, "")]
    if missing:
        raise MethodologyLoaderError(
            f"methodology row #{idx} ({raw.get('urn')!r}) missing "
            f"required fields: {missing}"
        )

    urn = str(raw["urn"]).strip()
    name = str(raw["name"]).strip()
    framework = str(raw["framework"]).strip()
    tier = raw.get("tier")
    approach = raw.get("approach")
    boundary_template = raw.get("boundary_template")
    allocation_rules = raw.get("allocation_rules")
    notes = raw.get("notes")

    try:
        parsed = parse(urn)
    except InvalidUrnError as exc:
        raise MethodologyLoaderError(
            f"methodology row #{idx} ({urn!r}) failed URN parse: {exc}"
        ) from exc
    if parsed.kind != "methodology":
        raise MethodologyLoaderError(
            f"methodology row #{idx} ({urn!r}) parsed as "
            f"kind={parsed.kind!r}, expected 'methodology'"
        )

    if approach is not None and approach not in ALLOWED_APPROACHES:
        raise MethodologyLoaderError(
            f"methodology row #{idx} ({urn!r}) has unknown approach "
            f"{approach!r}; allowed: {sorted(ALLOWED_APPROACHES)} or null"
        )

    if tier is not None:
        tier = str(tier).strip()

    return MethodologyRow(
        urn=urn,
        name=name,
        framework=framework,
        tier=tier if tier not in (None, "") else None,
        approach=str(approach) if approach is not None else None,
        boundary_template=str(boundary_template)
        if boundary_template is not None
        else None,
        allocation_rules=str(allocation_rules)
        if allocation_rules is not None
        else None,
        notes=str(notes) if notes is not None else None,
    )


def load_seed_yaml(path: Optional[Path] = None) -> List[MethodologyRow]:
    """Parse and validate every row in the methodology seed YAML."""
    seed_path = path or METHODOLOGY_SEED_PATH
    if not seed_path.exists():
        raise FileNotFoundError(
            f"methodology seed file not found: {seed_path}"
        )
    raw_text = seed_path.read_text(encoding="utf-8")
    doc = yaml.safe_load(raw_text)
    if doc is None:
        raise MethodologyLoaderError(
            f"methodology seed {seed_path} is empty or contains only "
            "comments"
        )
    if not isinstance(doc, dict) or "methodologies" not in doc:
        raise MethodologyLoaderError(
            f"methodology seed {seed_path} root must be a mapping with "
            f"a 'methodologies' list, got: {type(doc).__name__}"
        )
    rows_raw = doc["methodologies"]
    if not isinstance(rows_raw, list) or not rows_raw:
        raise MethodologyLoaderError(
            f"methodology seed {seed_path} 'methodologies' must be a "
            "non-empty list"
        )
    rows: List[MethodologyRow] = []
    seen_urns: set[str] = set()
    for idx, raw in enumerate(rows_raw):
        row = _validate_row(raw, idx)
        if row.urn in seen_urns:
            raise MethodologyLoaderError(
                f"methodology row #{idx} ({row.urn!r}) is a duplicate URN"
            )
        seen_urns.add(row.urn)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Insert SQL — driver-aware
# ---------------------------------------------------------------------------

_PG_INSERT_SQL = (
    "INSERT INTO factors_v0_1.methodology "
    "(urn, name, framework, tier, approach, boundary_template, "
    "allocation_rules, notes, seed_source) "
    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) "
    "ON CONFLICT (urn) DO NOTHING"
)

_SQLITE_INSERT_SQL = (
    "INSERT OR IGNORE INTO methodology "
    "(urn, name, framework, tier, approach, boundary_template, "
    "allocation_rules, notes, seed_source) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
)


def load_methodologies(
    conn: Any,
    rows: Optional[Sequence[MethodologyRow]] = None,
    *,
    path: Optional[Path] = None,
) -> LoadReport:
    """Idempotently insert the methodology seed.

    Args:
        conn: A PostgreSQL DB-API connection (``psycopg`` v3 or
            ``psycopg2``) OR a stdlib ``sqlite3.Connection``.
        rows: Pre-loaded rows. If None, the default seed YAML is parsed.
        path: Optional override for the seed YAML path.

    Returns:
        A :class:`LoadReport`.

    Raises:
        MethodologyLoaderError: when the seed is malformed.
    """
    if rows is None:
        rows = load_seed_yaml(path)
    sqlite = is_sqlite_connection(conn)
    insert_sql = _SQLITE_INSERT_SQL if sqlite else _PG_INSERT_SQL

    inserted = 0
    skipped = 0
    cur = conn.cursor()
    try:
        for row in rows:
            cur.execute(
                insert_sql,
                (
                    row.urn,
                    row.name,
                    row.framework,
                    row.tier,
                    row.approach,
                    row.boundary_template,
                    row.allocation_rules,
                    row.notes,
                    row.seed_source,
                ),
            )
            rc = getattr(cur, "rowcount", 0) or 0
            if rc > 0:
                inserted += 1
            else:
                skipped += 1
    finally:
        if hasattr(cur, "close"):
            try:
                cur.close()
            except Exception:  # pragma: no cover - defensive
                pass

    logger.info(
        "methodology loader inserted=%d skipped=%d total=%d (driver=%s)",
        inserted,
        skipped,
        len(rows),
        "sqlite" if sqlite else "postgres",
    )
    return LoadReport(
        count_inserted=inserted,
        count_skipped=skipped,
        total_seen=len(rows),
    )


# ---------------------------------------------------------------------------
# SQLite mirror DDL for tests
# ---------------------------------------------------------------------------

_SQLITE_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS methodology (
    pk_id             INTEGER PRIMARY KEY AUTOINCREMENT,
    urn               TEXT NOT NULL UNIQUE
        CHECK (urn GLOB 'urn:gl:methodology:*'),
    name              TEXT NOT NULL,
    framework         TEXT NOT NULL,
    tier              TEXT,
    approach          TEXT CHECK (approach IS NULL OR approach IN (
        'activity-based','spend-based','supplier-specific',
        'hybrid','measurement-based'
    )),
    boundary_template TEXT,
    allocation_rules  TEXT,
    notes             TEXT,
    seed_source       TEXT,
    created_at        TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""


def create_sqlite_methodology_table(conn: sqlite3.Connection) -> None:
    """Create an SQLite mirror of the V500+V501 methodology table."""
    cur = conn.cursor()
    cur.execute(_SQLITE_CREATE_SQL)
    conn.commit()
