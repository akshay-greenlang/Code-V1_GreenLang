# -*- coding: utf-8 -*-
"""v0.1 alpha publisher — staging vs production namespace flip.

Wave E / TaskCreate #23 / WS9-T1. Implements CTO doc §19.1 acceptance:

    "Runbook for a manual 'publish to production namespace' step:
     climate-methodology lead reviews staging diffs and flips visibility."

Two namespaces share the same physical store (the
``alpha_factors_v0_1`` SQLite table or ``factors_v0_1.factor`` Postgres
table). A ``namespace`` column distinguishes ``'staging'`` (open — anything
that passes the :class:`AlphaProvenanceGate` lands here) from
``'production'`` (closed — only methodology-lead-approved flips).

Design notes
------------

* The :class:`AlphaProvenanceGate` runs *exactly once*, at staging-entry.
  The flip-to-production step is purely a visibility change: it does NOT
  re-run the gate.  Records that fail the gate never reach staging.

* Promotions are recorded in an append-only ``factor_publish_log`` table
  so the methodology-lead audit trail is immutable. Every flip records
  the URN, ``approved_by``, ``approved_at``, ``batch_id`` and a
  ``"flip"`` / ``"rollback"`` action.

* Rollbacks demote a promoted record back to ``'staging'``. They do NOT
  delete the record — the v0.1 alpha contract is strictly immutable at
  the URN level. Demotions are also append-only entries in the publish
  log so a record's full visibility history is recoverable.

The publisher is deliberately thin: it owns *only* the namespace column
+ publish log. It piggy-backs on :class:`AlphaFactorRepository` for
schema management, gate enforcement, and the ``record_jsonb`` blob.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from greenlang.factors.repositories.alpha_v0_1_repository import (
    AlphaFactorRepository,
    FactorURNAlreadyExistsError,
)

logger = logging.getLogger(__name__)


__all__ = [
    "AlphaPublisher",
    "AlphaPublisherError",
    "StagingDiff",
    "NAMESPACE_STAGING",
    "NAMESPACE_PRODUCTION",
]


NAMESPACE_STAGING = "staging"
NAMESPACE_PRODUCTION = "production"
_VALID_NAMESPACES = (NAMESPACE_STAGING, NAMESPACE_PRODUCTION)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class AlphaPublisherError(Exception):
    """Raised when the publisher refuses an operation.

    Distinct from :class:`AlphaProvenanceGateError` (validation) and
    :class:`FactorURNAlreadyExistsError` (duplicate write) so callers can
    pinpoint which layer rejected.
    """


# ---------------------------------------------------------------------------
# StagingDiff dataclass
# ---------------------------------------------------------------------------


@dataclass
class StagingDiff:
    """Result of :meth:`AlphaPublisher.diff_staging_vs_production`.

    Attributes
    ----------
    additions
        Records present in staging that have no production counterpart at
        the same URN (and are not the *new* side of a supersede pair).
        Each element is the full v0.1 record dict.
    removals
        URNs present in production but *not* in staging. The methodology
        lead should ALWAYS scrutinise these — accidental removals are the
        single biggest risk during a flip.
    changes
        Pairs of ``(old_urn, new_urn)`` where ``new_urn`` is a staging
        record whose ``supersedes_urn`` field points at ``old_urn`` AND
        ``old_urn`` is currently in production.
    unchanged
        Count of URNs present in both namespaces with identical
        ``record_jsonb``. Sanity check: a flip that reports zero
        unchanged after a steady-state period is suspicious.
    """

    additions: List[Dict[str, Any]] = field(default_factory=list)
    removals: List[str] = field(default_factory=list)
    changes: List[Tuple[str, str]] = field(default_factory=list)
    unchanged: int = 0

    def is_empty(self) -> bool:
        return (
            not self.additions
            and not self.removals
            and not self.changes
        )

    def summary(self) -> str:
        """Single-line summary used in CLI output."""
        return (
            f"+{len(self.additions)} additions, "
            f"-{len(self.removals)} removals, "
            f"~{len(self.changes)} supersedes, "
            f"={self.unchanged} unchanged"
        )


# ---------------------------------------------------------------------------
# AlphaPublisher
# ---------------------------------------------------------------------------


class AlphaPublisher:
    """v0.1 alpha publish flow — staging entry + production flip.

    Construct with an existing :class:`AlphaFactorRepository`. The
    publisher will idempotently extend the repo's schema with a
    ``namespace`` column (default ``'staging'``) and a
    ``factor_publish_log`` table on first call.
    """

    # SQL DDL for the publish log. Append-only by convention.
    _SQLITE_LOG_DDL = (
        "CREATE TABLE IF NOT EXISTS factor_publish_log ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " batch_id TEXT NOT NULL,"
        " urn TEXT NOT NULL,"
        " action TEXT NOT NULL,"
        " from_namespace TEXT,"
        " to_namespace TEXT NOT NULL,"
        " approved_by TEXT NOT NULL,"
        " approved_at TIMESTAMPTZ NOT NULL"
        ")"
    )
    _SQLITE_LOG_INDEXES = (
        "CREATE INDEX IF NOT EXISTS ix_publish_log_urn ON factor_publish_log(urn)",
        "CREATE INDEX IF NOT EXISTS ix_publish_log_batch ON factor_publish_log(batch_id)",
    )

    # ALTER TABLE statement that adds the namespace column. Wrapped in
    # try/except for idempotency — SQLite raises OperationalError if the
    # column already exists.
    _SQLITE_ALTER_NS = (
        "ALTER TABLE alpha_factors_v0_1 "
        "ADD COLUMN namespace TEXT NOT NULL DEFAULT 'staging'"
    )
    _SQLITE_NS_INDEX = (
        "CREATE INDEX IF NOT EXISTS ix_alpha_factors_v0_1_ns "
        "ON alpha_factors_v0_1(namespace)"
    )

    def __init__(self, repo: AlphaFactorRepository) -> None:
        if repo is None:
            raise AlphaPublisherError("AlphaPublisher requires a repository")
        self._repo = repo
        self._lock = threading.Lock()
        if getattr(repo, "_is_postgres", False):
            # Postgres mode — defer to the schema extension method.
            self._is_postgres = True
            self._extend_schema_pg()
        else:
            self._is_postgres = False
            self._extend_schema_sqlite()

    # -- schema extension --------------------------------------------------

    def _extend_schema_sqlite(self) -> None:
        """Add ``namespace`` column + ``factor_publish_log`` table.

        Both operations are idempotent: the ALTER TABLE is wrapped in
        try/except (SQLite returns OperationalError "duplicate column
        name") and the CREATE TABLE uses ``IF NOT EXISTS``.
        """
        conn = self._connect()
        try:
            try:
                conn.execute(self._SQLITE_ALTER_NS)
            except sqlite3.OperationalError as exc:
                msg = str(exc).lower()
                if "duplicate column" not in msg and "already exists" not in msg:
                    raise
            try:
                conn.execute(self._SQLITE_NS_INDEX)
            except sqlite3.OperationalError:
                # Index already exists — non-fatal.
                pass
            conn.execute(self._SQLITE_LOG_DDL)
            for ddl in self._SQLITE_LOG_INDEXES:
                conn.execute(ddl)
        finally:
            self._maybe_close(conn)

    def _extend_schema_pg(self) -> None:
        """Postgres equivalent — wraps the ALTER in IF NOT EXISTS."""
        try:
            import psycopg  # type: ignore  # noqa: PLC0415
        except ImportError:
            return
        with psycopg.connect(self._repo._dsn) as conn:  # type: ignore[arg-type]
            with conn.cursor() as cur:
                cur.execute(
                    "ALTER TABLE factors_v0_1.factor "
                    "ADD COLUMN IF NOT EXISTS namespace TEXT NOT NULL "
                    "DEFAULT 'staging'"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS ix_factors_v0_1_namespace "
                    "ON factors_v0_1.factor(namespace)"
                )
                cur.execute(
                    "CREATE TABLE IF NOT EXISTS factors_v0_1.factor_publish_log ("
                    " id BIGSERIAL PRIMARY KEY,"
                    " batch_id TEXT NOT NULL,"
                    " urn TEXT NOT NULL,"
                    " action TEXT NOT NULL,"
                    " from_namespace TEXT,"
                    " to_namespace TEXT NOT NULL,"
                    " approved_by TEXT NOT NULL,"
                    " approved_at TIMESTAMPTZ NOT NULL DEFAULT NOW()"
                    ")"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS ix_publish_log_urn "
                    "ON factors_v0_1.factor_publish_log(urn)"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS ix_publish_log_batch "
                    "ON factors_v0_1.factor_publish_log(batch_id)"
                )
            conn.commit()

    # -- connection helpers ------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        return self._repo._connect()

    def _maybe_close(self, conn: sqlite3.Connection) -> None:
        if self._repo._memory_conn is None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass

    # -- staging entry -----------------------------------------------------

    def publish_to_staging(self, record: Dict[str, Any]) -> str:
        """Validate and persist a v0.1 record into the staging namespace.

        The :class:`AlphaProvenanceGate` runs FIRST. A record that fails
        validation never touches the DB. On success the record is
        written with ``namespace='staging'`` — even though the underlying
        column has that as the default, we set it explicitly so a future
        schema change can't accidentally promote staging writes.

        Returns:
            The canonical URN of the published record.
        """
        # Run gate via the repo's publish() helper for shared validation.
        # The repo's publish() also writes the row; our job is to ensure
        # the row's namespace ends up explicitly 'staging' afterwards.
        urn = self._repo.publish(record)

        if self._is_postgres:
            self._set_namespace_pg(urn, NAMESPACE_STAGING)
        else:
            with self._lock:
                conn = self._connect()
                try:
                    conn.execute(
                        "UPDATE alpha_factors_v0_1 SET namespace = ? WHERE urn = ?",
                        (NAMESPACE_STAGING, urn),
                    )
                finally:
                    self._maybe_close(conn)

        logger.info("alpha_publisher: staged urn=%s", urn)
        return urn

    def _set_namespace_pg(self, urn: str, namespace: str) -> None:
        try:
            import psycopg  # type: ignore  # noqa: PLC0415
        except ImportError:
            return
        with psycopg.connect(self._repo._dsn) as conn:  # type: ignore[arg-type]
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE factors_v0_1.factor SET namespace = %s WHERE urn = %s",
                    (namespace, urn),
                )
            conn.commit()

    # -- diff --------------------------------------------------------------

    def diff_staging_vs_production(self) -> StagingDiff:
        """Compute a snapshot diff between the two namespaces.

        Pure read; no DB writes. The result is the methodology lead's
        primary review artefact.
        """
        staging_rows = self._scan(NAMESPACE_STAGING)
        prod_rows = self._scan(NAMESPACE_PRODUCTION)

        staging_by_urn = {urn: rec for urn, rec in staging_rows}
        prod_by_urn = {urn: rec for urn, rec in prod_rows}

        # Supersede pairs: any staging record whose `supersedes_urn`
        # points at a URN currently in production.
        changes: List[Tuple[str, str]] = []
        superseded_old: set = set()
        for new_urn, new_rec in staging_by_urn.items():
            old = new_rec.get("supersedes_urn")
            if isinstance(old, str) and old and old in prod_by_urn:
                changes.append((old, new_urn))
                superseded_old.add(old)

        # Additions: in staging, not in production. Excludes records that
        # would already be visible via a supersede pair.
        additions: List[Dict[str, Any]] = [
            rec
            for urn, rec in staging_by_urn.items()
            if urn not in prod_by_urn
        ]

        # Removals: in production, not in staging, AND not superseded.
        # If a URN is superseded by a staging entry it shows up under
        # `changes`, never `removals`.
        removals: List[str] = [
            urn
            for urn in prod_by_urn
            if urn not in staging_by_urn and urn not in superseded_old
        ]

        # Unchanged: same URN in both namespaces with identical blobs.
        unchanged = 0
        for urn, prod_rec in prod_by_urn.items():
            stg_rec = staging_by_urn.get(urn)
            if stg_rec is None:
                continue
            if json.dumps(prod_rec, sort_keys=True) == json.dumps(stg_rec, sort_keys=True):
                unchanged += 1

        return StagingDiff(
            additions=additions,
            removals=removals,
            changes=changes,
            unchanged=unchanged,
        )

    def _scan(self, namespace: str) -> List[Tuple[str, Dict[str, Any]]]:
        """Return ``[(urn, record), ...]`` for every row in ``namespace``."""
        if self._is_postgres:
            return self._scan_pg(namespace)
        conn = self._connect()
        try:
            cur = conn.execute(
                "SELECT urn, record_jsonb FROM alpha_factors_v0_1 "
                "WHERE namespace = ?",
                (namespace,),
            )
            rows = cur.fetchall()
        finally:
            self._maybe_close(conn)
        return [(r["urn"], json.loads(r["record_jsonb"])) for r in rows]

    def _scan_pg(self, namespace: str) -> List[Tuple[str, Dict[str, Any]]]:
        try:
            import psycopg  # type: ignore  # noqa: PLC0415
        except ImportError:
            return []
        with psycopg.connect(self._repo._dsn) as conn:  # type: ignore[arg-type]
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT urn, record_jsonb FROM factors_v0_1.factor "
                    "WHERE namespace = %s",
                    (namespace,),
                )
                rows = cur.fetchall()
        out: List[Tuple[str, Dict[str, Any]]] = []
        for r in rows:
            blob = r[1]
            rec = blob if isinstance(blob, dict) else json.loads(blob)
            out.append((r[0], rec))
        return out

    # -- flip --------------------------------------------------------------

    def flip_to_production(
        self,
        *,
        urns: Sequence[str],
        approved_by: str,
        batch_id: Optional[str] = None,
    ) -> int:
        """Promote staging URNs to production.

        Idempotent — promoting an already-production URN is a no-op (no
        publish-log entry is added on the second call).

        Args:
            urns: URNs to promote. Must all currently be in staging.
            approved_by: ``human:<email>`` identifier of the methodology
                lead approving the flip. The ``human:`` prefix is required.
            batch_id: Optional batch id. Defaults to a fresh UUID4.

        Returns:
            Count of URNs ACTUALLY promoted (excluding no-ops).

        Raises:
            AlphaPublisherError: ``approved_by`` is empty / wrong shape, or
                a URN is not present in staging or production.
        """
        approver = (approved_by or "").strip()
        if not approver:
            raise AlphaPublisherError(
                "approved_by is required and must be non-empty"
            )
        if not approver.startswith("human:"):
            raise AlphaPublisherError(
                f"approved_by must start with 'human:' prefix; got {approver!r}"
            )
        if not urns:
            return 0

        bid = batch_id or f"flip-{uuid.uuid4().hex}"
        now = _now_iso()

        # Resolve current namespace for every URN. Reject if missing.
        current = self._namespace_of(list(urns))
        missing = [u for u in urns if u not in current]
        if missing:
            raise AlphaPublisherError(
                f"flip: {len(missing)} URN(s) not in repository: "
                f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
            )

        promoted = 0
        with self._lock:
            for urn in urns:
                ns = current[urn]
                if ns == NAMESPACE_PRODUCTION:
                    # Idempotent no-op.
                    continue
                if ns != NAMESPACE_STAGING:
                    raise AlphaPublisherError(
                        f"flip: urn {urn!r} is in unexpected namespace {ns!r}"
                    )
                self._update_namespace(urn, NAMESPACE_PRODUCTION)
                self._append_log(
                    batch_id=bid,
                    urn=urn,
                    action="flip",
                    from_ns=NAMESPACE_STAGING,
                    to_ns=NAMESPACE_PRODUCTION,
                    approved_by=approver,
                    approved_at=now,
                )
                promoted += 1

        logger.info(
            "alpha_publisher: flip batch=%s promoted=%d approved_by=%s",
            bid,
            promoted,
            approver,
        )
        return promoted

    def rollback(
        self,
        *,
        batch_id: str,
        approved_by: str,
    ) -> int:
        """Demote every URN in ``batch_id`` back to staging.

        Records are never deleted — the v0.1 catalogue is URN-immutable.
        A rollback is just another append-only log entry plus a namespace
        flip back to ``'staging'``. Re-running rollback for an already-
        rolled-back batch is a no-op.

        Returns:
            Count of URNs ACTUALLY demoted.
        """
        approver = (approved_by or "").strip()
        if not approver:
            raise AlphaPublisherError(
                "approved_by is required and must be non-empty"
            )
        if not approver.startswith("human:"):
            raise AlphaPublisherError(
                f"approved_by must start with 'human:' prefix; got {approver!r}"
            )
        if not batch_id:
            raise AlphaPublisherError("batch_id is required for rollback")

        urns = self._urns_in_batch(batch_id, action="flip")
        if not urns:
            raise AlphaPublisherError(
                f"rollback: no flip entries found for batch_id={batch_id!r}"
            )

        now = _now_iso()
        current = self._namespace_of(urns)
        demoted = 0
        with self._lock:
            for urn in urns:
                if current.get(urn) != NAMESPACE_PRODUCTION:
                    # Already in staging — idempotent skip.
                    continue
                self._update_namespace(urn, NAMESPACE_STAGING)
                self._append_log(
                    batch_id=batch_id,
                    urn=urn,
                    action="rollback",
                    from_ns=NAMESPACE_PRODUCTION,
                    to_ns=NAMESPACE_STAGING,
                    approved_by=approver,
                    approved_at=now,
                )
                demoted += 1

        logger.info(
            "alpha_publisher: rollback batch=%s demoted=%d approved_by=%s",
            batch_id,
            demoted,
            approver,
        )
        return demoted

    # -- listing -----------------------------------------------------------

    def list_staging(self) -> List[Dict[str, Any]]:
        return [rec for _, rec in self._scan(NAMESPACE_STAGING)]

    def list_production(self) -> List[Dict[str, Any]]:
        return [rec for _, rec in self._scan(NAMESPACE_PRODUCTION)]

    def list_log(
        self,
        *,
        batch_id: Optional[str] = None,
        urn: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return publish-log entries — used by tests + the rollback CLI."""
        if self._is_postgres:
            return self._list_log_pg(batch_id=batch_id, urn=urn)
        clauses: List[str] = []
        params: List[Any] = []
        if batch_id is not None:
            clauses.append("batch_id = ?")
            params.append(batch_id)
        if urn is not None:
            clauses.append("urn = ?")
            params.append(urn)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = (
            "SELECT id, batch_id, urn, action, from_namespace, to_namespace,"
            " approved_by, approved_at FROM factor_publish_log"
            + where
            + " ORDER BY id ASC"
        )
        conn = self._connect()
        try:
            cur = conn.execute(sql, tuple(params))
            rows = cur.fetchall()
        finally:
            self._maybe_close(conn)
        return [dict(r) for r in rows]

    def _list_log_pg(
        self,
        *,
        batch_id: Optional[str] = None,
        urn: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        try:
            import psycopg  # type: ignore  # noqa: PLC0415
        except ImportError:
            return []
        clauses: List[str] = []
        params: List[Any] = []
        if batch_id is not None:
            clauses.append("batch_id = %s")
            params.append(batch_id)
        if urn is not None:
            clauses.append("urn = %s")
            params.append(urn)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = (
            "SELECT id, batch_id, urn, action, from_namespace, to_namespace,"
            " approved_by, approved_at FROM factors_v0_1.factor_publish_log"
            + where
            + " ORDER BY id ASC"
        )
        out: List[Dict[str, Any]] = []
        with psycopg.connect(self._repo._dsn) as conn:  # type: ignore[arg-type]
            with conn.cursor() as cur:
                cur.execute(sql, tuple(params))
                for r in cur.fetchall():
                    out.append({
                        "id": r[0],
                        "batch_id": r[1],
                        "urn": r[2],
                        "action": r[3],
                        "from_namespace": r[4],
                        "to_namespace": r[5],
                        "approved_by": r[6],
                        "approved_at": r[7],
                    })
        return out

    # -- internal helpers --------------------------------------------------

    def _namespace_of(self, urns: List[str]) -> Dict[str, str]:
        if not urns:
            return {}
        if self._is_postgres:
            return self._namespace_of_pg(urns)
        placeholders = ",".join(["?"] * len(urns))
        sql = (
            f"SELECT urn, namespace FROM alpha_factors_v0_1 "
            f"WHERE urn IN ({placeholders})"
        )
        conn = self._connect()
        try:
            cur = conn.execute(sql, tuple(urns))
            rows = cur.fetchall()
        finally:
            self._maybe_close(conn)
        return {r["urn"]: r["namespace"] for r in rows}

    def _namespace_of_pg(self, urns: List[str]) -> Dict[str, str]:
        try:
            import psycopg  # type: ignore  # noqa: PLC0415
        except ImportError:
            return {}
        placeholders = ",".join(["%s"] * len(urns))
        sql = (
            f"SELECT urn, namespace FROM factors_v0_1.factor "
            f"WHERE urn IN ({placeholders})"
        )
        with psycopg.connect(self._repo._dsn) as conn:  # type: ignore[arg-type]
            with conn.cursor() as cur:
                cur.execute(sql, tuple(urns))
                rows = cur.fetchall()
        return {r[0]: r[1] for r in rows}

    def _update_namespace(self, urn: str, namespace: str) -> None:
        if namespace not in _VALID_NAMESPACES:
            raise AlphaPublisherError(
                f"invalid namespace {namespace!r}; expected one of {_VALID_NAMESPACES}"
            )
        if self._is_postgres:
            self._set_namespace_pg(urn, namespace)
            return
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE alpha_factors_v0_1 SET namespace = ? WHERE urn = ?",
                (namespace, urn),
            )
        finally:
            self._maybe_close(conn)

    def _append_log(
        self,
        *,
        batch_id: str,
        urn: str,
        action: str,
        from_ns: str,
        to_ns: str,
        approved_by: str,
        approved_at: str,
    ) -> None:
        if self._is_postgres:
            self._append_log_pg(
                batch_id=batch_id,
                urn=urn,
                action=action,
                from_ns=from_ns,
                to_ns=to_ns,
                approved_by=approved_by,
                approved_at=approved_at,
            )
            return
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO factor_publish_log ("
                " batch_id, urn, action, from_namespace, to_namespace,"
                " approved_by, approved_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?)",
                (batch_id, urn, action, from_ns, to_ns, approved_by, approved_at),
            )
        finally:
            self._maybe_close(conn)

    def _append_log_pg(
        self,
        *,
        batch_id: str,
        urn: str,
        action: str,
        from_ns: str,
        to_ns: str,
        approved_by: str,
        approved_at: str,
    ) -> None:
        try:
            import psycopg  # type: ignore  # noqa: PLC0415
        except ImportError:
            return
        with psycopg.connect(self._repo._dsn) as conn:  # type: ignore[arg-type]
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO factors_v0_1.factor_publish_log ("
                    " batch_id, urn, action, from_namespace, to_namespace,"
                    " approved_by, approved_at"
                    ") VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (
                        batch_id,
                        urn,
                        action,
                        from_ns,
                        to_ns,
                        approved_by,
                        approved_at,
                    ),
                )
            conn.commit()

    def _urns_in_batch(self, batch_id: str, *, action: str) -> List[str]:
        rows = self.list_log(batch_id=batch_id)
        return [r["urn"] for r in rows if r["action"] == action]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
