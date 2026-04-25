# -*- coding: utf-8 -*-
"""Real v0.1 alpha factor repository (Wave D / TaskCreate #31 / WS9-T5).

Replaces the in-test shim path that the SDK E2E demo (Wave C #19) relied on.
Records are stored verbatim — the JSON blob written by :meth:`publish` is
the EXACT same dict returned by :meth:`get_by_urn`. No coercion, no field
loss; the ``factor_record_v0_1.schema.json`` contract round-trips bit-for-bit.

Storage backends
----------------
- SQLite (``sqlite:///path/to.db`` or ``sqlite:///:memory:``)
- Postgres (``postgresql://...`` or ``postgres://...``) — uses the
  ``factors_v0_1.factor`` table created by Alembic revision 0001.

The SQLite schema mirrors the Postgres DDL columns 1:1 so a record written
on SQLite can be moved to Postgres with a column-aligned dump.

Immutability
------------
Once a URN is published it is immutable at the repository surface. There
is no ``update`` / ``upsert`` API; the SQLite schema declares
``urn TEXT PRIMARY KEY`` so a duplicate INSERT is rejected by the engine,
and on top of that :meth:`publish` raises :class:`FactorURNAlreadyExistsError`
on conflict. SQLite still permits low-level ``UPDATE`` SQL — the
JSONB-stored ``record_jsonb`` blob is therefore a *trust boundary*: callers
must use this repository class (never raw DB cursors) for every write.
Corrections happen via the v0.1 ``supersedes_urn`` field on a NEW record,
not by mutating an existing row.

CTO doc references: §6.1, §19.1 (FY27 Q1 alpha publish pipeline).
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote, urlparse

from greenlang.factors.quality.alpha_provenance_gate import (
    AlphaProvenanceGate,
    AlphaProvenanceGateError,
)

logger = logging.getLogger(__name__)


__all__ = [
    "AlphaFactorRepository",
    "FactorURNAlreadyExistsError",
]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class FactorURNAlreadyExistsError(Exception):
    """Raised when :meth:`AlphaFactorRepository.publish` is called with a URN
    that already exists in the catalogue.

    The v0.1 contract is strictly immutable — a correction MUST be issued
    as a NEW record carrying ``supersedes_urn`` pointing at the original.
    """

    def __init__(self, urn: str) -> None:
        self.urn = urn
        super().__init__(
            f"Factor URN {urn!r} already exists; v0.1 alpha records are "
            "immutable. Issue a correction with supersedes_urn pointing at "
            "the original."
        )


# ---------------------------------------------------------------------------
# DSN parsing
# ---------------------------------------------------------------------------


def _resolve_sqlite_path(dsn: str) -> str:
    """Extract a sqlite filesystem path (or ``:memory:``) from ``dsn``.

    Accepts: ``sqlite:///abs/path.db``, ``sqlite:///:memory:``,
    ``sqlite://:memory:`` (SQLAlchemy-style), or a bare path.
    """
    raw = (dsn or "").strip()
    if not raw:
        return ":memory:"
    if raw.startswith("sqlite:"):
        # sqlite:///abs -> "/abs"; sqlite:///:memory: -> "/:memory:"
        body = raw[len("sqlite:"):]
        body = body.lstrip("/")
        if body in ("", ":memory:"):
            return ":memory:"
        # Re-add leading slash on POSIX absolute paths (sqlite:////root/x ->
        # /root/x) but keep Windows drive letters intact (sqlite:///C:/x).
        if not body.startswith(":") and not (
            len(body) >= 2 and body[1] == ":"
        ):
            body = "/" + body
        return unquote(body)
    return raw


def _is_postgres_dsn(dsn: str) -> bool:
    return dsn.startswith("postgres://") or dsn.startswith("postgresql://")


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------


class AlphaFactorRepository:
    """Stores and serves v0.1-alpha-shape factor records.

    Backed by SQLite (alpha) or Postgres (production via Alembic 0001).
    Records are stored as JSONB blobs that exactly match the
    ``factor_record_v0_1.schema.json`` contract — no lossy coercion.

    Thread-safe: each call opens a fresh connection in SQLite mode (so
    multi-thread access never crosses the SQLite ``check_same_thread``
    boundary). Postgres mode uses ``psycopg`` connections per call.
    """

    # SQLite DDL — column types match the Postgres factors_v0_1.factor table
    # (TEXT for ISO-8601 timestamps; no engine-specific casting).
    _SQLITE_DDL = (
        "CREATE TABLE IF NOT EXISTS alpha_factors_v0_1 ("
        " urn TEXT PRIMARY KEY,"
        " source_urn TEXT,"
        " factor_pack_urn TEXT,"
        " category TEXT,"
        " geography_urn TEXT,"
        " vintage_start DATE,"
        " vintage_end DATE,"
        " published_at TIMESTAMPTZ,"
        " record_jsonb TEXT NOT NULL,"
        " created_at TIMESTAMPTZ DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))"
        ")"
    )
    _SQLITE_INDEXES = (
        "CREATE INDEX IF NOT EXISTS ix_alpha_factors_v0_1_source ON alpha_factors_v0_1(source_urn)",
        "CREATE INDEX IF NOT EXISTS ix_alpha_factors_v0_1_pack ON alpha_factors_v0_1(factor_pack_urn)",
        "CREATE INDEX IF NOT EXISTS ix_alpha_factors_v0_1_category ON alpha_factors_v0_1(category)",
        "CREATE INDEX IF NOT EXISTS ix_alpha_factors_v0_1_geo ON alpha_factors_v0_1(geography_urn)",
        "CREATE INDEX IF NOT EXISTS ix_alpha_factors_v0_1_published ON alpha_factors_v0_1(published_at DESC)",
    )

    # Whitelist of filter columns — guards SQL injection by NEVER
    # interpolating user-controlled column names into SQL. Values are
    # always passed as bind parameters.
    _FILTER_COLUMNS = (
        "source_urn",
        "factor_pack_urn",
        "category",
        "geography_urn",
    )

    def __init__(self, dsn: str, *, gate: Optional[AlphaProvenanceGate] = None) -> None:
        """Open the repository.

        Args:
            dsn: ``sqlite:///path.db`` / ``sqlite:///:memory:`` /
                 ``postgresql://...``. A bare path is treated as SQLite.
            gate: An optional :class:`AlphaProvenanceGate` instance. The
                  default constructs a fresh gate (lazy-loads the schema).
        """
        self._dsn = dsn or "sqlite:///:memory:"
        self._gate = gate or AlphaProvenanceGate()
        self._lock = threading.Lock()
        self._is_postgres = _is_postgres_dsn(self._dsn)
        # In-memory SQLite: hold a single shared connection for the lifetime
        # of the repo so the schema and rows persist across method calls.
        self._memory_conn: Optional[sqlite3.Connection] = None
        if not self._is_postgres:
            sqlite_path = _resolve_sqlite_path(self._dsn)
            self._sqlite_path = sqlite_path
            if sqlite_path == ":memory:":
                self._memory_conn = sqlite3.connect(
                    ":memory:", check_same_thread=False, isolation_level=None
                )
                self._memory_conn.row_factory = sqlite3.Row
            else:
                # Ensure parent dir exists.
                p = Path(sqlite_path)
                if p.parent and not p.parent.exists():
                    p.parent.mkdir(parents=True, exist_ok=True)
        else:
            self._sqlite_path = None
        self._ensure_schema()

    # -- connection management --------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """Return a SQLite connection (memory-shared or per-call file)."""
        if self._is_postgres:
            raise RuntimeError(
                "Postgres mode connections are managed via psycopg pool — "
                "use _connect_pg() instead."
            )
        if self._memory_conn is not None:
            return self._memory_conn
        conn = sqlite3.connect(
            self._sqlite_path,  # type: ignore[arg-type]
            check_same_thread=False,
            isolation_level=None,
        )
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        """Create table + indexes if they don't exist (SQLite only).

        Postgres mode relies on Alembic revision 0001 having already created
        the ``factors_v0_1.factor`` table.
        """
        if self._is_postgres:
            return
        conn = self._connect()
        try:
            conn.execute(self._SQLITE_DDL)
            for ddl in self._SQLITE_INDEXES:
                conn.execute(ddl)
        finally:
            if self._memory_conn is None:
                conn.close()

    def close(self) -> None:
        """Release the in-memory connection (if any)."""
        if self._memory_conn is not None:
            try:
                self._memory_conn.close()
            except Exception:  # noqa: BLE001
                pass
            self._memory_conn = None

    # -- public API --------------------------------------------------------

    def publish(self, record: Dict[str, Any]) -> str:
        """Validate the record, then persist atomically.

        Args:
            record: A v0.1-shape factor dict that satisfies the
                ``factor_record_v0_1.schema.json`` contract AND the alpha
                provenance gate.

        Returns:
            The canonical URN of the published record.

        Raises:
            AlphaProvenanceGateError: validation failed.
            FactorURNAlreadyExistsError: a record with the same URN exists.
            ValueError: the record is missing the ``urn`` key entirely.
        """
        # Run the gate FIRST — if validation fails we never touch the DB.
        self._gate.assert_valid(record)

        urn = record.get("urn")
        if not isinstance(urn, str) or not urn:
            # Defensive — schema gate already enforces this.
            raise ValueError("record['urn'] must be a non-empty string")

        # Snapshot the columns we mirror so SQL filters can hit indexes.
        source_urn = record.get("source_urn")
        pack_urn = record.get("factor_pack_urn")
        category = record.get("category")
        geo_urn = record.get("geography_urn")
        vintage_start = record.get("vintage_start")
        vintage_end = record.get("vintage_end")
        published_at = record.get("published_at")
        record_blob = json.dumps(record, sort_keys=True, default=str, ensure_ascii=False)

        if self._is_postgres:
            return self._publish_pg(
                urn,
                source_urn,
                pack_urn,
                category,
                geo_urn,
                vintage_start,
                vintage_end,
                published_at,
                record_blob,
            )

        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "SELECT 1 FROM alpha_factors_v0_1 WHERE urn = ?",
                    (urn,),
                )
                if cur.fetchone() is not None:
                    raise FactorURNAlreadyExistsError(urn)
                try:
                    conn.execute(
                        "INSERT INTO alpha_factors_v0_1 ("
                        " urn, source_urn, factor_pack_urn, category,"
                        " geography_urn, vintage_start, vintage_end,"
                        " published_at, record_jsonb"
                        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            urn,
                            source_urn,
                            pack_urn,
                            category,
                            geo_urn,
                            vintage_start,
                            vintage_end,
                            published_at,
                            record_blob,
                        ),
                    )
                except sqlite3.IntegrityError as exc:
                    # Race: another writer beat us between SELECT and INSERT.
                    raise FactorURNAlreadyExistsError(urn) from exc
            finally:
                if self._memory_conn is None:
                    conn.close()

        logger.info("alpha_factor_repo: published urn=%s source=%s", urn, source_urn)
        return urn

    def _publish_pg(
        self,
        urn: str,
        source_urn: Optional[str],
        pack_urn: Optional[str],
        category: Optional[str],
        geo_urn: Optional[str],
        vintage_start: Optional[str],
        vintage_end: Optional[str],
        published_at: Optional[str],
        record_blob: str,
    ) -> str:
        """Postgres publish path. Lazy import so SQLite users don't pay
        the ``psycopg`` dependency cost.
        """
        try:
            import psycopg  # type: ignore  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "Postgres DSN requires the 'psycopg' driver; install with "
                "`pip install greenlang[server]`."
            ) from exc

        with psycopg.connect(self._dsn) as conn:  # type: ignore[arg-type]
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM factors_v0_1.factor WHERE urn = %s",
                    (urn,),
                )
                if cur.fetchone() is not None:
                    raise FactorURNAlreadyExistsError(urn)
                try:
                    cur.execute(
                        "INSERT INTO factors_v0_1.factor ("
                        " urn, source_urn, factor_pack_urn, category,"
                        " geography_urn, vintage_start, vintage_end,"
                        " published_at, record_jsonb"
                        ") VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)",
                        (
                            urn,
                            source_urn,
                            pack_urn,
                            category,
                            geo_urn,
                            vintage_start,
                            vintage_end,
                            published_at,
                            record_blob,
                        ),
                    )
                except psycopg.errors.UniqueViolation as exc:  # type: ignore[attr-defined]
                    raise FactorURNAlreadyExistsError(urn) from exc
            conn.commit()
        return urn

    def get_by_urn(self, urn: str) -> Optional[Dict[str, Any]]:
        """Return the JSON-decoded ``record_jsonb`` directly (no coercion).

        Returns ``None`` if no record matches the URN.
        """
        if not isinstance(urn, str) or not urn:
            return None
        if self._is_postgres:
            return self._get_by_urn_pg(urn)
        conn = self._connect()
        try:
            cur = conn.execute(
                "SELECT record_jsonb FROM alpha_factors_v0_1 WHERE urn = ?",
                (urn,),
            )
            row = cur.fetchone()
        finally:
            if self._memory_conn is None:
                conn.close()
        if row is None:
            return None
        return json.loads(row["record_jsonb"])

    def _get_by_urn_pg(self, urn: str) -> Optional[Dict[str, Any]]:
        try:
            import psycopg  # type: ignore  # noqa: PLC0415
        except ImportError:
            return None
        with psycopg.connect(self._dsn) as conn:  # type: ignore[arg-type]
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT record_jsonb FROM factors_v0_1.factor WHERE urn = %s",
                    (urn,),
                )
                row = cur.fetchone()
        if row is None:
            return None
        # psycopg returns dict for JSONB columns directly; tolerate str too.
        blob = row[0]
        if isinstance(blob, dict):
            return blob
        if isinstance(blob, str):
            return json.loads(blob)
        return None

    def list_factors(
        self,
        *,
        geography_urn: Optional[str] = None,
        source_urn: Optional[str] = None,
        pack_urn: Optional[str] = None,
        category: Optional[str] = None,
        vintage_start_after: Optional[str] = None,
        vintage_end_before: Optional[str] = None,
        cursor: Optional[str] = None,
        limit: int = 50,
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """Return ``(records, next_cursor)`` filtered & cursor-paginated.

        Filters are AND-combined and bound as parameters (SQL-injection safe).
        Sort order is ``published_at DESC, urn ASC`` so cursors are stable
        even when two rows share the same publish timestamp.
        """
        # Clamp limit to reasonable bounds.
        if limit < 1:
            limit = 1
        if limit > 500:
            limit = 500

        # Build the WHERE clause from the column whitelist only — never
        # interpolate user-controlled identifiers.
        where: List[str] = []
        params: List[Any] = []
        for col, val in (
            ("geography_urn", geography_urn),
            ("source_urn", source_urn),
            ("factor_pack_urn", pack_urn),
            ("category", category),
        ):
            if val is not None:
                where.append(f"{col} = ?")
                params.append(val)
        if vintage_start_after is not None:
            where.append("vintage_start > ?")
            params.append(vintage_start_after)
        if vintage_end_before is not None:
            where.append("vintage_end < ?")
            params.append(vintage_end_before)

        # Cursor encodes the last seen (published_at, urn) tuple so pages
        # never overlap or skip. Decoded form: "v1:<published_at>|<urn>".
        cursor_state = _decode_cursor(cursor)
        if cursor_state is not None:
            last_pub, last_urn = cursor_state
            where.append("(published_at < ? OR (published_at = ? AND urn > ?))")
            params.extend([last_pub, last_pub, last_urn])

        where_sql = (" WHERE " + " AND ".join(where)) if where else ""
        sql = (
            "SELECT urn, published_at, record_jsonb FROM alpha_factors_v0_1"
            + where_sql
            + " ORDER BY published_at DESC, urn ASC LIMIT ?"
        )
        params.append(limit + 1)  # Fetch one extra to detect more-pages.

        if self._is_postgres:
            return self._list_factors_pg(where, params, limit)

        conn = self._connect()
        try:
            cur = conn.execute(sql, tuple(params))
            rows = cur.fetchall()
        finally:
            if self._memory_conn is None:
                conn.close()

        records = [json.loads(r["record_jsonb"]) for r in rows[:limit]]
        next_cursor: Optional[str] = None
        if len(rows) > limit:
            last_keep = rows[limit - 1]
            next_cursor = _encode_cursor(last_keep["published_at"], last_keep["urn"])
        return records, next_cursor

    def _list_factors_pg(
        self, where: List[str], params: List[Any], limit: int
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        try:
            import psycopg  # type: ignore  # noqa: PLC0415
        except ImportError:
            return [], None
        # psycopg uses %s placeholders; rewrite the ? from above.
        where_pg = [w.replace("?", "%s") for w in where]
        where_sql = (" WHERE " + " AND ".join(where_pg)) if where_pg else ""
        sql = (
            "SELECT urn, published_at, record_jsonb FROM factors_v0_1.factor"
            + where_sql
            + " ORDER BY published_at DESC, urn ASC LIMIT %s"
        )
        with psycopg.connect(self._dsn) as conn:  # type: ignore[arg-type]
            with conn.cursor() as cur:
                cur.execute(sql, tuple(params))
                rows = cur.fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows[:limit]:
            blob = r[2]
            out.append(blob if isinstance(blob, dict) else json.loads(blob))
        next_cursor: Optional[str] = None
        if len(rows) > limit:
            keep = rows[limit - 1]
            pub_str = (
                keep[1].isoformat()
                if hasattr(keep[1], "isoformat")
                else str(keep[1])
            )
            next_cursor = _encode_cursor(pub_str, keep[0])
        return out, next_cursor

    def list_sources(self) -> List[Dict[str, Any]]:
        """Return the alpha-flagged source registry rows."""
        try:
            from greenlang.factors.source_registry import alpha_v0_1_sources
            rows = alpha_v0_1_sources()
        except Exception as exc:  # noqa: BLE001
            logger.warning("list_sources: registry load failed: %s", exc)
            return []
        out: List[Dict[str, Any]] = []
        for source_id, item in sorted((rows or {}).items()):
            row = dict(item) if isinstance(item, dict) else {}
            row.setdefault("source_id", source_id)
            out.append(row)
        return out

    def list_packs(
        self, source_urn: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Return synthetic packs from the source registry (one per source).

        When the v0.1 alpha catalogue grows real pack metadata, this
        method will switch to a SELECT against an ``alpha_packs_v0_1``
        table; for now we mirror the synthetic shape used by the alpha
        router so the SDK contract is unchanged.
        """
        sources = self.list_sources()
        out: List[Dict[str, Any]] = []
        for item in sources:
            sid = str(item.get("source_id") or "unknown")
            s_urn = str(item.get("urn") or f"urn:gl:source:{sid}")
            if source_urn and s_urn != source_urn:
                continue
            version_str = str(item.get("source_version") or "0.1")
            out.append(
                {
                    "urn": f"urn:gl:pack:{sid}:default:v1",
                    "source_urn": s_urn,
                    "pack_id": "default",
                    "version": version_str,
                    "display_name": item.get("display_name"),
                    "factor_count": None,
                }
            )
        return out

    # -- diagnostics -------------------------------------------------------

    def count(self) -> int:
        """Total number of stored records (mostly used by tests)."""
        if self._is_postgres:
            try:
                import psycopg  # type: ignore  # noqa: PLC0415
            except ImportError:
                return 0
            with psycopg.connect(self._dsn) as conn:  # type: ignore[arg-type]
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM factors_v0_1.factor")
                    row = cur.fetchone()
            return int(row[0]) if row else 0
        conn = self._connect()
        try:
            cur = conn.execute("SELECT COUNT(*) FROM alpha_factors_v0_1")
            row = cur.fetchone()
        finally:
            if self._memory_conn is None:
                conn.close()
        return int(row[0]) if row else 0


# ---------------------------------------------------------------------------
# Cursor encoding — opaque to clients, stable round-trip
# ---------------------------------------------------------------------------


_CURSOR_PREFIX = "v1:"
_CURSOR_SEP = "|"


def _encode_cursor(published_at: Any, urn: str) -> str:
    """Encode the last-seen (published_at, urn) tuple as an opaque cursor."""
    pub = published_at if isinstance(published_at, str) else (
        published_at.isoformat()
        if hasattr(published_at, "isoformat")
        else _now_iso()
    )
    return f"{_CURSOR_PREFIX}{pub}{_CURSOR_SEP}{urn}"


def _decode_cursor(cursor: Optional[str]) -> Optional[Tuple[str, str]]:
    """Decode the opaque cursor; ``None`` means start-of-list."""
    if not cursor or not isinstance(cursor, str):
        return None
    if not cursor.startswith(_CURSOR_PREFIX):
        return None
    body = cursor[len(_CURSOR_PREFIX):]
    if _CURSOR_SEP not in body:
        return None
    pub, urn = body.split(_CURSOR_SEP, 1)
    if not pub or not urn:
        return None
    return pub, urn


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
