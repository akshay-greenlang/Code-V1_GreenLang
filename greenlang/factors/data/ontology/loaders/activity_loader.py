# -*- coding: utf-8 -*-
"""Activity-taxonomy seed loader (Phase 2 / WS5).

Reads :file:`greenlang/factors/data/ontology/activity_seed_v0_1.yaml`
and idempotently inserts each row into the ``factors_v0_1.activity``
table created by the V502 migration.

Every row's ``urn`` is validated through
:func:`greenlang.factors.ontology.urn.parse` before the SQL is issued so
a malformed seed entry fails the loader, not the database.

The loader supports two execution targets:

* **Postgres** (production): pass a live ``psycopg`` connection. Uses
  ``INSERT ... ON CONFLICT (urn) DO NOTHING`` so re-running the loader
  is a no-op.
* **SQLite** (tests): pass a ``sqlite3`` connection. Uses
  ``INSERT OR IGNORE`` for idempotency. The CHECK constraint on
  ``taxonomy`` is mirrored in the SQLite-create helper so tests exercise
  the same validation envelope.

Returns a 2-tuple ``(inserted, skipped)``.
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import yaml

from greenlang.factors.ontology.urn import (
    ALLOWED_ACTIVITY_TAXONOMIES,
    InvalidUrnError,
    parse,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ActivityRow",
    "ActivityLoaderError",
    "DEFAULT_SEED_PATH",
    "create_sqlite_activity_table",
    "load_seed_yaml",
    "load_into_postgres",
    "load_into_sqlite",
]

# ---------------------------------------------------------------------------
# Resolved relative to this file so the loader works regardless of cwd.
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
DEFAULT_SEED_PATH: Path = (_THIS_DIR.parent / "activity_seed_v0_1.yaml")


class ActivityLoaderError(ValueError):
    """Raised when the seed YAML cannot be parsed/validated."""


@dataclass(frozen=True)
class ActivityRow:
    """One canonical activity-taxonomy row.

    Mirrors the columns of ``factors_v0_1.activity`` 1:1.
    """

    urn: str
    taxonomy: str
    code: str
    name: str
    description: Optional[str] = None
    parent_urn: Optional[str] = None


# ---------------------------------------------------------------------------
# Seed parsing + URN validation
# ---------------------------------------------------------------------------


def _validate_row(raw: Any, idx: int) -> ActivityRow:
    """Validate one raw mapping from YAML; raise on any defect."""
    if not isinstance(raw, dict):
        raise ActivityLoaderError(
            f"activity row #{idx} is not a mapping: {raw!r}"
        )
    required = ("urn", "taxonomy", "code", "name")
    missing = [k for k in required if k not in raw or raw[k] in (None, "")]
    if missing:
        raise ActivityLoaderError(
            f"activity row #{idx} ({raw.get('urn')!r}) missing required "
            f"fields: {missing}"
        )

    urn = str(raw["urn"]).strip()
    taxonomy = str(raw["taxonomy"]).strip()
    code = str(raw["code"]).strip()
    name = str(raw["name"]).strip()
    description = raw.get("description")
    parent_urn = raw.get("parent_urn")

    if taxonomy not in ALLOWED_ACTIVITY_TAXONOMIES:
        raise ActivityLoaderError(
            f"activity row #{idx} ({urn!r}) has unknown taxonomy "
            f"{taxonomy!r}; allowed: {ALLOWED_ACTIVITY_TAXONOMIES}"
        )

    # URN must parse and the parsed taxonomy/code must match the row's
    # explicit columns. (Catches typos like
    # ``urn:gl:activity:naics:11`` paired with ``taxonomy: ipcc``.)
    try:
        parsed = parse(urn)
    except InvalidUrnError as exc:
        raise ActivityLoaderError(
            f"activity row #{idx} ({urn!r}) failed URN parse: {exc}"
        ) from exc
    if parsed.kind != "activity":
        raise ActivityLoaderError(
            f"activity row #{idx} ({urn!r}) parsed as kind={parsed.kind!r}"
        )
    if parsed.taxonomy != taxonomy:
        raise ActivityLoaderError(
            f"activity row #{idx} ({urn!r}) URN taxonomy "
            f"{parsed.taxonomy!r} != row taxonomy {taxonomy!r}"
        )
    # The URN's code segment is the URN-encoded form of the row's `code`
    # column: lowercase, with dots replaced by hyphens. We verify the
    # URN-encoded form rather than strict equality so that source codes
    # carrying uppercase (NACE 'A'..'U') or dots (IPCC '1.A.1.a') survive
    # into the `code` column verbatim while the URN stays canonical.
    expected_code_slug = code.lower().replace(".", "-")
    if parsed.code != expected_code_slug:
        raise ActivityLoaderError(
            f"activity row #{idx} ({urn!r}) URN code {parsed.code!r} != "
            f"expected URN-encoded code {expected_code_slug!r} (derived "
            f"from row code {code!r})"
        )

    if parent_urn is not None:
        try:
            parent_parsed = parse(str(parent_urn))
        except InvalidUrnError as exc:
            raise ActivityLoaderError(
                f"activity row #{idx} ({urn!r}) parent_urn "
                f"{parent_urn!r} failed parse: {exc}"
            ) from exc
        if parent_parsed.kind != "activity":
            raise ActivityLoaderError(
                f"activity row #{idx} ({urn!r}) parent_urn "
                f"{parent_urn!r} is not an activity URN"
            )

    return ActivityRow(
        urn=urn,
        taxonomy=taxonomy,
        code=code,
        name=name,
        description=str(description) if description is not None else None,
        parent_urn=str(parent_urn) if parent_urn is not None else None,
    )


def load_seed_yaml(path: Optional[Path] = None) -> List[ActivityRow]:
    """Parse and validate every row in the activity seed YAML.

    Args:
        path: Override the default seed path.

    Returns:
        List of fully validated :class:`ActivityRow` instances, in YAML
        order (which matches insertion order so parent_urn FK is always
        satisfied at insert time).

    Raises:
        ActivityLoaderError: on any structural or URN defect.
        FileNotFoundError: if the seed file is missing.
    """
    seed_path = path or DEFAULT_SEED_PATH
    if not seed_path.exists():
        raise FileNotFoundError(f"activity seed file not found: {seed_path}")
    raw_text = seed_path.read_text(encoding="utf-8")
    doc = yaml.safe_load(raw_text)
    if doc is None:
        raise ActivityLoaderError(
            f"activity seed {seed_path} is empty or contains only comments"
        )
    if not isinstance(doc, dict) or "activities" not in doc:
        raise ActivityLoaderError(
            f"activity seed {seed_path} root must be a mapping with an "
            f"'activities' list, got: {type(doc).__name__}"
        )
    rows_raw = doc["activities"]
    if not isinstance(rows_raw, list) or not rows_raw:
        raise ActivityLoaderError(
            f"activity seed {seed_path} 'activities' must be a non-empty list"
        )
    rows: List[ActivityRow] = []
    seen_urns: set[str] = set()
    seen_pairs: set[Tuple[str, str]] = set()
    for idx, raw in enumerate(rows_raw):
        row = _validate_row(raw, idx)
        if row.urn in seen_urns:
            raise ActivityLoaderError(
                f"activity row #{idx} ({row.urn!r}) is a duplicate URN"
            )
        pair = (row.taxonomy, row.code)
        if pair in seen_pairs:
            raise ActivityLoaderError(
                f"activity row #{idx} ({row.urn!r}) has duplicate "
                f"(taxonomy, code) pair: {pair}"
            )
        seen_urns.add(row.urn)
        seen_pairs.add(pair)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Postgres target
# ---------------------------------------------------------------------------

_PG_INSERT_SQL = (
    "INSERT INTO factors_v0_1.activity "
    "(urn, taxonomy, code, name, description, parent_urn) "
    "VALUES (%s, %s, %s, %s, %s, %s) "
    "ON CONFLICT (urn) DO NOTHING"
)


def load_into_postgres(
    conn: Any,
    rows: Optional[Sequence[ActivityRow]] = None,
    *,
    path: Optional[Path] = None,
) -> Tuple[int, int]:
    """Idempotently insert the activity seed into a live Postgres conn.

    Args:
        conn: A ``psycopg`` (v3) or ``psycopg2`` connection. The loader
            uses positional ``%s`` placeholders, which both drivers accept.
        rows: Pre-loaded rows. If None, the default seed YAML is parsed.
        path: Optional override for the seed YAML path (used only when
            ``rows`` is None).

    Returns:
        ``(inserted, skipped)`` — the number of rows newly inserted vs.
        the number ignored because their URN already existed.

    Raises:
        ActivityLoaderError: when the seed itself is malformed.
    """
    if rows is None:
        rows = load_seed_yaml(path)
    inserted = 0
    skipped = 0
    with conn.cursor() as cur:
        for row in rows:
            cur.execute(
                _PG_INSERT_SQL,
                (
                    row.urn,
                    row.taxonomy,
                    row.code,
                    row.name,
                    row.description,
                    row.parent_urn,
                ),
            )
            # psycopg sets cursor.rowcount = 1 on insert, 0 on conflict-skip.
            rc = getattr(cur, "rowcount", 0) or 0
            if rc > 0:
                inserted += 1
            else:
                skipped += 1
    logger.info(
        "activity loader (postgres) inserted=%d skipped=%d total=%d",
        inserted,
        skipped,
        len(rows),
    )
    return inserted, skipped


# ---------------------------------------------------------------------------
# SQLite target (tests)
# ---------------------------------------------------------------------------

_SQLITE_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS activity (
    pk_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    urn          TEXT NOT NULL UNIQUE
        CHECK (urn GLOB 'urn:gl:activity:[a-z0-9]*'),
    taxonomy     TEXT NOT NULL CHECK (taxonomy IN (
        'ipcc','ghgp','hs-cn','cpc','nace','naics','sic','pact',
        'freight','cbam','pcf','refrigerants','agriculture','waste',
        'land-use'
    )),
    code         TEXT NOT NULL,
    name         TEXT NOT NULL,
    description  TEXT,
    parent_urn   TEXT REFERENCES activity(urn),
    created_at   TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (taxonomy, code)
)
"""

_SQLITE_INSERT_SQL = (
    "INSERT OR IGNORE INTO activity "
    "(urn, taxonomy, code, name, description, parent_urn) "
    "VALUES (?, ?, ?, ?, ?, ?)"
)


def create_sqlite_activity_table(conn: sqlite3.Connection) -> None:
    """Create an SQLite mirror of the V502 activity table for tests."""
    cur = conn.cursor()
    cur.execute(_SQLITE_CREATE_SQL)
    conn.commit()


def load_into_sqlite(
    conn: sqlite3.Connection,
    rows: Optional[Sequence[ActivityRow]] = None,
    *,
    path: Optional[Path] = None,
) -> Tuple[int, int]:
    """Idempotently insert the activity seed into an SQLite connection.

    Mirrors :func:`load_into_postgres` for use in unit tests.
    """
    if rows is None:
        rows = load_seed_yaml(path)
    inserted = 0
    skipped = 0
    cur = conn.cursor()
    for row in rows:
        cur.execute(
            _SQLITE_INSERT_SQL,
            (
                row.urn,
                row.taxonomy,
                row.code,
                row.name,
                row.description,
                row.parent_urn,
            ),
        )
        # In SQLite ``INSERT OR IGNORE`` returns rowcount=1 for inserts
        # and 0 for ignored rows.
        rc = cur.rowcount or 0
        if rc > 0:
            inserted += 1
        else:
            skipped += 1
    conn.commit()
    logger.info(
        "activity loader (sqlite) inserted=%d skipped=%d total=%d",
        inserted,
        skipped,
        len(rows),
    )
    return inserted, skipped
