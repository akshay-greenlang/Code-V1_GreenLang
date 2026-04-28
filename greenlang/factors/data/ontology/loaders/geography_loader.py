# -*- coding: utf-8 -*-
"""Geography ontology seed loader (Phase 2 / WS3).

Reads :file:`greenlang/factors/data/ontology/geography_seed_v0_1.yaml`
and idempotently inserts each row into the ``factors_v0_1.geography``
table created by V500 (and extended by V501 to admit ``basin`` and
``tenant`` types).

Every row's ``urn`` is validated through
:func:`greenlang.factors.ontology.urn.parse` before SQL is issued so a
malformed seed entry fails the loader, not the database.

Public surface:
    * :func:`load_geography`  -- idempotent insert against any DB-API 2.0
      connection (psycopg, psycopg2, sqlite3).
    * :func:`load_seed_yaml`  -- parse + validate the YAML, return rows.
    * :func:`create_sqlite_geography_table` -- mirror of the V500/V501
      Postgres DDL for sqlite-backed unit tests.

Returns a :class:`greenlang.factors.data.ontology.loaders.LoadReport`.
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
    "GeographyRow",
    "GeographyLoaderError",
    "GEOGRAPHY_SEED_PATH",
    "create_sqlite_geography_table",
    "load_seed_yaml",
    "load_geography",
]

# ---------------------------------------------------------------------------
# Resolved relative to this file so the loader works regardless of cwd.
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
GEOGRAPHY_SEED_PATH: Path = (_THIS_DIR.parent / "geography_seed_v0_1.yaml")

# V501-widened set of geography types accepted by the canonical schema +
# Postgres CHECK. Single source of truth in the loader (the YAML's row
# 'type' field is cross-checked against this set).
ALLOWED_GEOGRAPHY_TYPES = frozenset(
    [
        "global",
        "country",
        "subregion",
        "state_or_province",
        "grid_zone",
        "bidding_zone",
        "balancing_authority",
        "basin",
        "tenant",
    ]
)


class GeographyLoaderError(ValueError):
    """Raised when the seed YAML cannot be parsed/validated."""


@dataclass(frozen=True)
class GeographyRow:
    """One canonical geography row.

    Mirrors the columns of ``factors_v0_1.geography`` 1:1 (after V501).
    """

    urn: str
    type: str
    iso_code: Optional[str]
    name: str
    parent_urn: Optional[str]
    centroid_lat: Optional[float]
    centroid_lon: Optional[float]
    tags: Optional[List[str]]
    seed_source: str = PHASE2_SEED_SOURCE


# ---------------------------------------------------------------------------
# Seed parsing + URN validation
# ---------------------------------------------------------------------------


def _validate_row(raw: Any, idx: int) -> GeographyRow:
    if not isinstance(raw, dict):
        raise GeographyLoaderError(
            f"geography row #{idx} is not a mapping: {raw!r}"
        )
    required = ("urn", "type", "name")
    missing = [k for k in required if k not in raw or raw[k] in (None, "")]
    if missing:
        raise GeographyLoaderError(
            f"geography row #{idx} ({raw.get('urn')!r}) missing required "
            f"fields: {missing}"
        )

    urn = str(raw["urn"]).strip()
    geo_type = str(raw["type"]).strip()
    iso_code = raw.get("iso_code")
    name = str(raw["name"]).strip()
    parent_urn = raw.get("parent_urn")
    centroid_lat = raw.get("centroid_lat")
    centroid_lon = raw.get("centroid_lon")
    tags = raw.get("tags")

    if geo_type not in ALLOWED_GEOGRAPHY_TYPES:
        raise GeographyLoaderError(
            f"geography row #{idx} ({urn!r}) has unknown type "
            f"{geo_type!r}; allowed: {sorted(ALLOWED_GEOGRAPHY_TYPES)}"
        )

    # Mandatory: every URN parses canonically (lower-case, valid kind).
    try:
        parsed = parse(urn)
    except InvalidUrnError as exc:
        raise GeographyLoaderError(
            f"geography row #{idx} ({urn!r}) failed URN parse: {exc}"
        ) from exc
    if parsed.kind != "geo":
        raise GeographyLoaderError(
            f"geography row #{idx} ({urn!r}) parsed as kind={parsed.kind!r}, "
            f"expected 'geo'"
        )
    if parsed.geo_type != geo_type:
        raise GeographyLoaderError(
            f"geography row #{idx} ({urn!r}) URN type {parsed.geo_type!r} "
            f"!= row type {geo_type!r}"
        )

    if parent_urn is not None:
        try:
            parent_parsed = parse(str(parent_urn))
        except InvalidUrnError as exc:
            raise GeographyLoaderError(
                f"geography row #{idx} ({urn!r}) parent_urn "
                f"{parent_urn!r} failed parse: {exc}"
            ) from exc
        if parent_parsed.kind != "geo":
            raise GeographyLoaderError(
                f"geography row #{idx} ({urn!r}) parent_urn "
                f"{parent_urn!r} is not a geo URN"
            )

    if iso_code is not None:
        iso_code = str(iso_code).strip().upper()
        if len(iso_code) != 2:
            raise GeographyLoaderError(
                f"geography row #{idx} ({urn!r}) iso_code must be a 2-char "
                f"ISO-3166-1 alpha-2 code, got {iso_code!r}"
            )

    if centroid_lat is not None:
        centroid_lat = float(centroid_lat)
        if not -90.0 <= centroid_lat <= 90.0:
            raise GeographyLoaderError(
                f"geography row #{idx} ({urn!r}) centroid_lat out of range "
                f"[-90, 90]: {centroid_lat}"
            )
    if centroid_lon is not None:
        centroid_lon = float(centroid_lon)
        if not -180.0 <= centroid_lon <= 180.0:
            raise GeographyLoaderError(
                f"geography row #{idx} ({urn!r}) centroid_lon out of range "
                f"[-180, 180]: {centroid_lon}"
            )

    if tags is not None:
        if not isinstance(tags, list) or not all(
            isinstance(t, str) for t in tags
        ):
            raise GeographyLoaderError(
                f"geography row #{idx} ({urn!r}) tags must be a list of "
                f"strings, got {tags!r}"
            )

    return GeographyRow(
        urn=urn,
        type=geo_type,
        iso_code=iso_code,
        name=name,
        parent_urn=str(parent_urn) if parent_urn is not None else None,
        centroid_lat=centroid_lat,
        centroid_lon=centroid_lon,
        tags=list(tags) if tags is not None else None,
    )


def load_seed_yaml(path: Optional[Path] = None) -> List[GeographyRow]:
    """Parse and validate every row in the geography seed YAML.

    Returns rows in YAML order — which is topologically sorted so that
    parent_urn FKs always resolve at insert time (global -> countries
    -> subregions -> states -> grid_zones -> bidding_zones ->
    balancing_authorities -> basins).
    """
    seed_path = path or GEOGRAPHY_SEED_PATH
    if not seed_path.exists():
        raise FileNotFoundError(f"geography seed file not found: {seed_path}")
    raw_text = seed_path.read_text(encoding="utf-8")
    doc = yaml.safe_load(raw_text)
    if doc is None:
        raise GeographyLoaderError(
            f"geography seed {seed_path} is empty or contains only comments"
        )
    if not isinstance(doc, dict) or "geographies" not in doc:
        raise GeographyLoaderError(
            f"geography seed {seed_path} root must be a mapping with a "
            f"'geographies' list, got: {type(doc).__name__}"
        )
    rows_raw = doc["geographies"]
    if not isinstance(rows_raw, list) or not rows_raw:
        raise GeographyLoaderError(
            f"geography seed {seed_path} 'geographies' must be a non-empty "
            "list"
        )
    rows: List[GeographyRow] = []
    seen_urns: set[str] = set()
    for idx, raw in enumerate(rows_raw):
        row = _validate_row(raw, idx)
        if row.urn in seen_urns:
            raise GeographyLoaderError(
                f"geography row #{idx} ({row.urn!r}) is a duplicate URN"
            )
        seen_urns.add(row.urn)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Insert SQL — driver-aware
# ---------------------------------------------------------------------------

# Postgres uses %s placeholders, ARRAY[...] for tags, and ON CONFLICT.
_PG_INSERT_SQL = (
    "INSERT INTO factors_v0_1.geography "
    "(urn, type, iso_code, name, parent_urn, centroid_lat, centroid_lon, "
    "tags, seed_source) "
    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) "
    "ON CONFLICT (urn) DO NOTHING"
)

# SQLite uses ? placeholders, no ARRAY type (we serialise tags as JSON
# text), and INSERT OR IGNORE for idempotency.
_SQLITE_INSERT_SQL = (
    "INSERT OR IGNORE INTO geography "
    "(urn, type, iso_code, name, parent_urn, centroid_lat, centroid_lon, "
    "tags, seed_source) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
)


def load_geography(
    conn: Any,
    rows: Optional[Sequence[GeographyRow]] = None,
    *,
    path: Optional[Path] = None,
) -> LoadReport:
    """Idempotently insert the geography seed.

    Args:
        conn: A PostgreSQL DB-API connection (``psycopg`` v3 or
            ``psycopg2``) OR a stdlib ``sqlite3.Connection``. The loader
            auto-detects the driver and switches placeholder style + ON
            CONFLICT syntax accordingly.
        rows: Pre-loaded rows. If None, the default seed YAML is parsed.
        path: Optional override for the seed YAML path.

    Returns:
        A :class:`LoadReport` with ``count_inserted`` /
        ``count_skipped`` / ``total_seen``.

    Raises:
        GeographyLoaderError: when the seed is malformed.

    Notes:
        The loader does NOT call ``conn.commit()``. The caller owns the
        transaction. (For Alembic data migrations, the bind manages
        commit at the migration boundary.)
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
            # tags: array on PG; JSON-encoded TEXT on sqlite.
            if sqlite:
                import json as _json

                tags_payload: Any = (
                    _json.dumps(row.tags, ensure_ascii=False)
                    if row.tags is not None
                    else None
                )
            else:
                # psycopg / psycopg2: lists are auto-adapted to ARRAY.
                tags_payload = row.tags

            cur.execute(
                insert_sql,
                (
                    row.urn,
                    row.type,
                    row.iso_code,
                    row.name,
                    row.parent_urn,
                    row.centroid_lat,
                    row.centroid_lon,
                    tags_payload,
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
        "geography loader inserted=%d skipped=%d total=%d (driver=%s)",
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
CREATE TABLE IF NOT EXISTS geography (
    pk_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    urn          TEXT NOT NULL UNIQUE
        CHECK (urn GLOB 'urn:gl:geo:*'),
    type         TEXT NOT NULL CHECK (type IN (
        'global','country','subregion','state_or_province',
        'grid_zone','bidding_zone','balancing_authority',
        'basin','tenant'
    )),
    iso_code     TEXT,
    name         TEXT NOT NULL,
    parent_urn   TEXT REFERENCES geography(urn),
    centroid_lat REAL,
    centroid_lon REAL,
    tags         TEXT,
    seed_source  TEXT,
    created_at   TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""


def create_sqlite_geography_table(conn: sqlite3.Connection) -> None:
    """Create an SQLite mirror of the V500+V501 geography table."""
    cur = conn.cursor()
    cur.execute(_SQLITE_CREATE_SQL)
    conn.commit()
