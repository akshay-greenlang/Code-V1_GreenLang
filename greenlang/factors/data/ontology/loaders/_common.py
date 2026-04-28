# -*- coding: utf-8 -*-
"""Common helpers for the Phase 2 ontology seed loaders.

Single source-of-truth for:
    * the canonical Phase 2 seed-source marker string (``phase2_v0_1``)
      used by the V501 ALTER's ``seed_source`` column;
    * a tiny driver-detection helper that distinguishes ``sqlite3``
      connections from PostgreSQL DBAPI connections (``psycopg``,
      ``psycopg2``);
    * a generic :class:`LoadReport` named tuple returned by every
      loader.

The split exists because the geography / unit / methodology loaders all
share these primitives but otherwise have wildly different row shapes,
JSONB columns, and CHECK contracts.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from typing import Any, NamedTuple

logger = logging.getLogger(__name__)

# Marker string written to the ``seed_source`` column added by V501. The
# Alembic ``downgrade()`` for revision 0002 deletes ONLY rows carrying
# this marker so production-ingested rows are not collateral damage.
PHASE2_SEED_SOURCE: str = "phase2_v0_1"


class LoadReport(NamedTuple):
    """Outcome of an idempotent ontology seed load.

    Attributes:
        count_inserted: rows inserted on this call.
        count_skipped: rows already present (ON CONFLICT DO NOTHING /
            INSERT OR IGNORE).
        total_seen: total rows in the seed file (== inserted + skipped).
    """

    count_inserted: int
    count_skipped: int
    total_seen: int


def is_sqlite_connection(conn: Any) -> bool:
    """Return True if *conn* is a stdlib ``sqlite3.Connection``.

    Used by the loaders to switch placeholder style (``?`` vs ``%s``)
    and to translate ``ON CONFLICT (urn) DO NOTHING`` (PG) to
    ``OR IGNORE`` (sqlite).
    """
    return isinstance(conn, sqlite3.Connection)


def encode_jsonb_for_driver(value: Any, *, sqlite: bool) -> Any:
    """Serialise a Python dict for either Postgres JSONB or sqlite TEXT.

    psycopg/psycopg2 both auto-encode dicts/lists into JSONB on the
    server side via the standard adapter, but only when the column is
    typed JSONB. Encoding to a JSON string up-front works for both
    drivers and is portable to sqlite (which has no native JSON
    column).
    """
    if value is None:
        return None
    if sqlite:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    # psycopg adapters accept Python dicts directly; we still json-dump
    # to keep the wire format identical between drivers (and to avoid
    # silent re-ordering of keys in regression diffs).
    return json.dumps(value, ensure_ascii=False, sort_keys=True)
