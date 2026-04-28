# -*- coding: utf-8 -*-
"""Unit ontology seed loader (Phase 2 / WS4).

Reads :file:`greenlang/factors/data/ontology/unit_seed_v0_1.yaml` and
idempotently inserts each row into the ``factors_v0_1.unit`` table
created by V500. Uses the V501 ``seed_source`` column to mark Phase 2
seed rows so the downgrade path can roll them back without disturbing
production-ingested rows.

Every row's ``urn`` is validated through
:func:`greenlang.factors.ontology.urn.parse` before SQL is issued; the
unit-symbol grammar in the canonical parser is stricter than the V500
CHECK pattern, and we comply with the stricter rule (lower-case only,
plus ``_ ^ %``).

Public surface:
    * :func:`load_units` -- idempotent insert against any DB-API 2.0
      connection (psycopg, psycopg2, sqlite3).
    * :func:`load_seed_yaml` -- parse + validate the YAML, return rows.
    * :func:`create_sqlite_unit_table` -- mirror of the V500 Postgres
      DDL for sqlite-backed unit tests.
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

from greenlang.factors.data.ontology.loaders._common import (
    LoadReport,
    PHASE2_SEED_SOURCE,
    encode_jsonb_for_driver,
    is_sqlite_connection,
)
from greenlang.factors.ontology.urn import InvalidUrnError, parse

logger = logging.getLogger(__name__)

__all__ = [
    "UnitRow",
    "UnitLoaderError",
    "UNIT_SEED_PATH",
    "ALLOWED_DIMENSIONS",
    "create_sqlite_unit_table",
    "load_seed_yaml",
    "load_units",
]

_THIS_DIR = Path(__file__).resolve().parent
UNIT_SEED_PATH: Path = (_THIS_DIR.parent / "unit_seed_v0_1.yaml")

# Per CTO Phase 2 §2.3 (Units / WS4) — pre-defined dimensions for v0.1
# alpha. The V500 unit.dimension column is unconstrained TEXT; the loader
# enforces this enum at validation time so v0.1 seed rows cannot drift.
ALLOWED_DIMENSIONS = frozenset(
    [
        "mass",
        "energy",
        "volume",
        "distance",
        "freight_activity",
        "currency",
        "composite_climate",
    ]
)


class UnitLoaderError(ValueError):
    """Raised when the seed YAML cannot be parsed/validated."""


@dataclass(frozen=True)
class UnitRow:
    """One canonical unit row.

    Mirrors the columns of ``factors_v0_1.unit`` 1:1 (after V501).
    """

    urn: str
    symbol: str
    dimension: str
    conversions: Dict[str, float] = field(default_factory=dict)
    iso_reference: Optional[str] = None
    seed_source: str = PHASE2_SEED_SOURCE


# ---------------------------------------------------------------------------
# Seed parsing + URN validation
# ---------------------------------------------------------------------------


def _validate_row(raw: Any, idx: int) -> UnitRow:
    if not isinstance(raw, dict):
        raise UnitLoaderError(
            f"unit row #{idx} is not a mapping: {raw!r}"
        )
    required = ("urn", "symbol", "dimension")
    missing = [k for k in required if k not in raw or raw[k] in (None, "")]
    if missing:
        raise UnitLoaderError(
            f"unit row #{idx} ({raw.get('urn')!r}) missing required "
            f"fields: {missing}"
        )

    urn = str(raw["urn"]).strip()
    symbol = str(raw["symbol"]).strip()
    dimension = str(raw["dimension"]).strip()
    conversions_raw = raw.get("conversions") or {}
    iso_reference = raw.get("iso_reference")

    if dimension not in ALLOWED_DIMENSIONS:
        raise UnitLoaderError(
            f"unit row #{idx} ({urn!r}) has unknown dimension "
            f"{dimension!r}; allowed: {sorted(ALLOWED_DIMENSIONS)}"
        )

    try:
        parsed = parse(urn)
    except InvalidUrnError as exc:
        raise UnitLoaderError(
            f"unit row #{idx} ({urn!r}) failed URN parse: {exc}"
        ) from exc
    if parsed.kind != "unit":
        raise UnitLoaderError(
            f"unit row #{idx} ({urn!r}) parsed as kind={parsed.kind!r}, "
            f"expected 'unit'"
        )

    if not isinstance(conversions_raw, dict):
        raise UnitLoaderError(
            f"unit row #{idx} ({urn!r}) conversions must be a mapping, "
            f"got {type(conversions_raw).__name__}"
        )
    conversions: Dict[str, float] = {}
    for k, v in conversions_raw.items():
        if not isinstance(k, str) or not k:
            raise UnitLoaderError(
                f"unit row #{idx} ({urn!r}) conversions key must be a "
                f"non-empty string, got {k!r}"
            )
        try:
            conversions[k] = float(v)
        except (TypeError, ValueError) as exc:
            raise UnitLoaderError(
                f"unit row #{idx} ({urn!r}) conversions[{k!r}]={v!r} "
                f"is not a finite number"
            ) from exc
        if conversions[k] <= 0:
            raise UnitLoaderError(
                f"unit row #{idx} ({urn!r}) conversions[{k!r}]={v!r} "
                f"must be > 0"
            )

    return UnitRow(
        urn=urn,
        symbol=symbol,
        dimension=dimension,
        conversions=conversions,
        iso_reference=str(iso_reference)
        if iso_reference is not None
        else None,
    )


def load_seed_yaml(path: Optional[Path] = None) -> List[UnitRow]:
    """Parse and validate every row in the unit seed YAML."""
    seed_path = path or UNIT_SEED_PATH
    if not seed_path.exists():
        raise FileNotFoundError(f"unit seed file not found: {seed_path}")
    raw_text = seed_path.read_text(encoding="utf-8")
    doc = yaml.safe_load(raw_text)
    if doc is None:
        raise UnitLoaderError(
            f"unit seed {seed_path} is empty or contains only comments"
        )
    if not isinstance(doc, dict) or "units" not in doc:
        raise UnitLoaderError(
            f"unit seed {seed_path} root must be a mapping with a "
            f"'units' list, got: {type(doc).__name__}"
        )
    rows_raw = doc["units"]
    if not isinstance(rows_raw, list) or not rows_raw:
        raise UnitLoaderError(
            f"unit seed {seed_path} 'units' must be a non-empty list"
        )
    rows: List[UnitRow] = []
    seen_urns: set[str] = set()
    for idx, raw in enumerate(rows_raw):
        row = _validate_row(raw, idx)
        if row.urn in seen_urns:
            raise UnitLoaderError(
                f"unit row #{idx} ({row.urn!r}) is a duplicate URN"
            )
        seen_urns.add(row.urn)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Insert SQL — driver-aware
# ---------------------------------------------------------------------------

_PG_INSERT_SQL = (
    "INSERT INTO factors_v0_1.unit "
    "(urn, symbol, dimension, conversions, iso_reference, seed_source) "
    "VALUES (%s, %s, %s, %s::jsonb, %s, %s) "
    "ON CONFLICT (urn) DO NOTHING"
)

_SQLITE_INSERT_SQL = (
    "INSERT OR IGNORE INTO unit "
    "(urn, symbol, dimension, conversions, iso_reference, seed_source) "
    "VALUES (?, ?, ?, ?, ?, ?)"
)


def load_units(
    conn: Any,
    rows: Optional[Sequence[UnitRow]] = None,
    *,
    path: Optional[Path] = None,
) -> LoadReport:
    """Idempotently insert the unit seed.

    Args:
        conn: A PostgreSQL DB-API connection (``psycopg`` v3 or
            ``psycopg2``) OR a stdlib ``sqlite3.Connection``.
        rows: Pre-loaded rows. If None, the default seed YAML is parsed.
        path: Optional override for the seed YAML path.

    Returns:
        A :class:`LoadReport`.

    Raises:
        UnitLoaderError: when the seed is malformed.
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
            conversions_payload = encode_jsonb_for_driver(
                row.conversions, sqlite=sqlite
            )
            cur.execute(
                insert_sql,
                (
                    row.urn,
                    row.symbol,
                    row.dimension,
                    conversions_payload,
                    row.iso_reference,
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
        "unit loader inserted=%d skipped=%d total=%d (driver=%s)",
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
CREATE TABLE IF NOT EXISTS unit (
    pk_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    urn           TEXT NOT NULL UNIQUE
        CHECK (urn GLOB 'urn:gl:unit:*'),
    symbol        TEXT NOT NULL,
    dimension     TEXT NOT NULL CHECK (dimension IN (
        'mass','energy','volume','distance',
        'freight_activity','currency','composite_climate'
    )),
    conversions   TEXT NOT NULL DEFAULT '{}',
    iso_reference TEXT,
    seed_source   TEXT,
    created_at    TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""


def create_sqlite_unit_table(conn: sqlite3.Connection) -> None:
    """Create an SQLite mirror of the V500+V501 unit table."""
    cur = conn.cursor()
    cur.execute(_SQLITE_CREATE_SQL)
    conn.commit()
