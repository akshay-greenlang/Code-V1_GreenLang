# -*- coding: utf-8 -*-
"""Persistence layer for ``ingestion_runs`` + ``ingestion_run_diffs`` rows.

The Phase 3 plan §"Run-status enum (formal)" makes the run table the
forensic backbone of the pipeline: every stage writes a row update before
returning. This module wraps that table behind an :class:`IngestionRunRepository`
mirroring the dual-backend pattern used by
:class:`~greenlang.factors.repositories.alpha_v0_1_repository.AlphaFactorRepository`
(SQLite for alpha + dev, Postgres for production via the V507 / V508 Alembic
migrations).

V507/V508 are still in flight when this module ships (task #27); the
SQLite DDL inside this module is the **authoritative reference shape** that
the Postgres migrations MUST match. CI gates added in task #32 enforce the
parity.

References
----------
- ``docs/factors/PHASE_3_PLAN.md`` §"The seven-stage pipeline contract"
- ``docs/factors/PHASE_3_PLAN.md`` §"Artifact storage contract"
- ``greenlang/factors/repositories/alpha_v0_1_repository.py`` (dual-backend
  template).
"""

import contextlib
import json
import logging
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple
from urllib.parse import unquote

from greenlang.factors.ingestion.pipeline import (
    IngestionRun,
    RunStatus,
    Stage,
    StageResult,
    assert_can_transition,
    now_utc,
)

logger = logging.getLogger(__name__)


__all__ = [
    "IngestionRunRepository",
    "IngestionRunNotFoundError",
]


class IngestionRunNotFoundError(Exception):
    """Raised when a run_id lookup misses both backends.

    Distinct from :class:`~greenlang.factors.ingestion.exceptions.IngestionError`
    so callers can differentiate "row not found" from "stage refused".
    """


# ---------------------------------------------------------------------------
# DSN helpers — duplicated from alpha_v0_1_repository to avoid the import
# cycle (this repo is referenced from runner.py which is imported on the
# fetch/parse hot-path; the alpha repo pulls in the publish-gate stack).
# ---------------------------------------------------------------------------


def _resolve_sqlite_path(dsn: str) -> str:
    """Translate a ``sqlite:///`` DSN to a filesystem path or ``:memory:``."""
    raw = (dsn or "").strip()
    if not raw:
        return ":memory:"
    if raw.startswith("sqlite:"):
        body = raw[len("sqlite:") :]
        body = body.lstrip("/")
        if body in ("", ":memory:"):
            return ":memory:"
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


class IngestionRunRepository:
    """Persistence wrapper around the V507 ``ingestion_runs`` table.

    The repository is intentionally narrow: it only knows about the run
    row + stage history + diff URIs. Factor-level writes happen via
    :class:`AlphaFactorRepository` and :class:`AlphaPublisher` — this class
    never touches the factor table.

    Mirrors the alpha repo pattern: SQLite mode opens fresh connections
    per call (or shares a single connection in ``:memory:`` mode);
    Postgres mode lazy-imports ``psycopg`` so cold startup pays no
    Postgres cost.
    """

    _SQLITE_DDL: Tuple[str, ...] = (
        # ingestion_runs — one row per run, advanced by every stage.
        "CREATE TABLE IF NOT EXISTS ingestion_runs ("
        " run_id TEXT PRIMARY KEY,"
        " source_urn TEXT NOT NULL,"
        " source_version TEXT NOT NULL,"
        " started_at TEXT NOT NULL,"
        " finished_at TEXT,"
        " status TEXT NOT NULL,"
        " current_stage TEXT,"
        " artifact_id TEXT,"
        " artifact_sha256 TEXT,"
        " parser_module TEXT,"
        " parser_version TEXT,"
        " parser_commit TEXT,"
        " operator TEXT NOT NULL,"
        " batch_id TEXT,"
        " approved_by TEXT,"
        " error_json TEXT,"
        " diff_json_uri TEXT,"
        " diff_md_uri TEXT"
        ")",
        # ingestion_run_stage_history — append-only stage receipts.
        "CREATE TABLE IF NOT EXISTS ingestion_run_stage_history ("
        " pk_id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " run_id TEXT NOT NULL,"
        " stage TEXT NOT NULL,"
        " ok INTEGER NOT NULL,"
        " duration_s REAL NOT NULL,"
        " error TEXT,"
        " details_json TEXT,"
        " started_at TEXT NOT NULL,"
        " finished_at TEXT"
        ")",
        # ingestion_run_diffs — one row per stage-6 diff artefact pointer.
        "CREATE TABLE IF NOT EXISTS ingestion_run_diffs ("
        " run_id TEXT PRIMARY KEY,"
        " diff_json_uri TEXT,"
        " diff_md_uri TEXT,"
        " summary_json TEXT,"
        " created_at TEXT NOT NULL"
        ")",
    )

    _SQLITE_INDEXES: Tuple[str, ...] = (
        "CREATE INDEX IF NOT EXISTS ix_ingestion_runs_source ON ingestion_runs(source_urn)",
        "CREATE INDEX IF NOT EXISTS ix_ingestion_runs_status ON ingestion_runs(status)",
        "CREATE INDEX IF NOT EXISTS ix_ingestion_runs_batch ON ingestion_runs(batch_id)",
        "CREATE INDEX IF NOT EXISTS ix_ingestion_run_stage_history_run ON ingestion_run_stage_history(run_id)",
    )

    def __init__(self, dsn: str) -> None:
        """Open the run repository.

        Args:
            dsn: ``sqlite:///path/to.db`` / ``sqlite:///:memory:`` /
                ``postgresql://...``. A bare path is treated as SQLite.
        """
        self._dsn = dsn or "sqlite:///:memory:"
        self._lock = threading.Lock()
        self._is_postgres = _is_postgres_dsn(self._dsn)
        self._memory_conn: Optional[sqlite3.Connection] = None
        self._sqlite_path: Optional[str] = None

        if not self._is_postgres:
            self._sqlite_path = _resolve_sqlite_path(self._dsn)
            if self._sqlite_path == ":memory:":
                self._memory_conn = sqlite3.connect(
                    ":memory:", check_same_thread=False, isolation_level=None
                )
                self._memory_conn.row_factory = sqlite3.Row
            else:
                p = Path(self._sqlite_path)
                if p.parent and not p.parent.exists():
                    p.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    # -- connection management --------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """Return a SQLite connection (memory-shared or per-call file)."""
        if self._is_postgres:
            raise RuntimeError(
                "Postgres mode uses psycopg connections; SQLite path called "
                "in Postgres mode is a programming error."
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

    def _maybe_close(self, conn: sqlite3.Connection) -> None:
        if self._memory_conn is None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass

    # -- Phase 3 audit gap A: atomic SQLite stage transitions ----------
    # The original implementation opened the connection in
    # ``isolation_level=None`` (autocommit), so every ``conn.execute()``
    # committed immediately. Phase 3 plan §"Run-status enum (formal)"
    # requires ONE atomic transaction per stage advance — partial writes
    # on a mid-stage failure would corrupt the run row. The context
    # manager wraps the (possibly multi-statement) write in an explicit
    # ``BEGIN; ... COMMIT;`` and rolls back on any exception, matching
    # the dual-backend semantics the Postgres path enforces via the
    # ``with conn:`` block.
    @contextlib.contextmanager
    def _sqlite_txn(self) -> Iterator[sqlite3.Connection]:
        """Open a SQLite connection wrapped in a single explicit transaction.

        Yields a live :class:`sqlite3.Connection` after issuing
        ``BEGIN IMMEDIATE``. Commits on clean exit; rolls back on any
        exception then re-raises so the runner's ``_mark_failed`` path
        sees a state-consistent run row (no half-applied stage advance).
        Closes the connection on exit when the repo is in file-mode
        (the ``:memory:`` mode reuses ``self._memory_conn``).
        """
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError:
            # ``BEGIN`` may fail if a prior statement left an implicit
            # txn open (e.g. nested call). Roll back defensively and
            # retry once; if that still fails, propagate so the caller
            # can decide.
            try:
                conn.execute("ROLLBACK")
            except sqlite3.OperationalError:
                pass
            conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except BaseException:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.OperationalError:
                pass
            self._maybe_close(conn)
            raise
        try:
            conn.execute("COMMIT")
        finally:
            self._maybe_close(conn)

    def _ensure_schema(self) -> None:
        """Create the SQLite mirror of the V507 + V508 tables idempotently."""
        if self._is_postgres:
            # Postgres relies on Alembic V507 / V508 having already run.
            return
        conn = self._connect()
        try:
            for ddl in self._SQLITE_DDL:
                conn.execute(ddl)
            for ddl in self._SQLITE_INDEXES:
                conn.execute(ddl)
        finally:
            self._maybe_close(conn)

    def close(self) -> None:
        """Release the in-memory connection (if any)."""
        if self._memory_conn is not None:
            try:
                self._memory_conn.close()
            except Exception:  # noqa: BLE001
                pass
            self._memory_conn = None

    # -- public API --------------------------------------------------------

    def create(
        self,
        *,
        source_urn: str,
        source_version: str,
        operator: str,
    ) -> IngestionRun:
        """Persist a fresh run with status ``CREATED``.

        Per Phase 3 plan §"The seven-stage pipeline contract", every run
        begins with a row commit BEFORE stage 1 runs so a fetch failure
        is still recorded with the run_id.
        """
        run = IngestionRun(
            source_urn=source_urn,
            source_version=source_version,
            operator=operator,
            status=RunStatus.CREATED,
            started_at=now_utc(),
        )
        if self._is_postgres:
            self._insert_pg(run)
        else:
            self._insert_sqlite(run)
        logger.info(
            "ingestion_run created run_id=%s source=%s version=%s operator=%s",
            run.run_id, source_urn, source_version, operator,
        )
        return run

    def update_status(
        self,
        run_id: str,
        status: RunStatus,
        *,
        current_stage: Optional[Stage] = None,
        error_json: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Advance a run's status, validating the transition matrix.

        Raises :class:`StageOrderError` (via
        :func:`~greenlang.factors.ingestion.pipeline.assert_can_transition`)
        when ``current_status -> status`` is not a legal edge in the
        :data:`ALLOWED_TRANSITIONS` graph.
        """
        existing = self.get(run_id)
        assert_can_transition(existing.status, status, run_id=run_id)

        finished_at: Optional[datetime] = None
        from greenlang.factors.ingestion.pipeline import (  # noqa: PLC0415
            TERMINAL_FAILURE_STATUSES,
            TERMINAL_SUCCESS_STATUSES,
        )
        if status in TERMINAL_FAILURE_STATUSES or status == RunStatus.PUBLISHED:
            finished_at = now_utc()

        if self._is_postgres:
            self._update_status_pg(
                run_id, status, current_stage, error_json, finished_at
            )
        else:
            self._update_status_sqlite(
                run_id, status, current_stage, error_json, finished_at
            )
        logger.info(
            "ingestion_run status run_id=%s %s -> %s stage=%s",
            run_id,
            existing.status.value,
            status.value,
            current_stage.value if current_stage else "-",
        )

    def upsert_source_artifact(
        self,
        *,
        sha256: str,
        source_urn: str,
        source_version: str,
        source_url: Optional[str],
        bytes_size: Optional[int],
        content_type: Optional[str],
        parser_module: Optional[str],
        parser_function: Optional[str],
        parser_version: Optional[str],
        parser_commit: Optional[str],
        operator: Optional[str],
        licence_class: Optional[str],
        redistribution_class: Optional[str],
        source_publication_date: Optional[str],
        ingestion_run_id: str,
        status: str = "fetched",
        uri: Optional[str] = None,
    ) -> None:
        """Upsert the full Phase 3 contract row into ``source_artifacts``.

        Phase 3 audit gap C — the runner calls this at the end of stage 1
        (fetch) to land the canonical 16+ field row, then refreshes the
        ``status`` column at every subsequent stage so a SELECT on the
        artifact row tracks pipeline progress.

        The ``sha256`` column is UNIQUE so the operation is a true
        upsert — fetching the same artifact twice does NOT duplicate
        the row, but DOES refresh status / parser metadata / operator /
        ingestion_run_id (a new run for the same bytes).

        Args:
            sha256: SHA-256 hex of the raw fetched bytes (UNIQUE key).
            source_urn: ``urn:gl:source:<id>``.
            source_version: registry source_version.
            source_url: human-readable URL the bytes were fetched from.
            bytes_size: size of the raw bytes payload.
            content_type: MIME type if known (``application/...``).
            parser_module: dotted Python module of the parser class.
            parser_function: callable / class name of the parser entry.
            parser_version: parser semver.
            parser_commit: optional git commit pinning the parser.
            operator: ``human:<email>`` or ``bot:<id>`` driving the run.
            licence_class: registry ``licence_class`` enum value.
            redistribution_class: registry ``redistribution_class`` enum.
            source_publication_date: ISO-8601 ``YYYY-MM-DD`` or None.
            ingestion_run_id: the run owning this artifact write.
            status: pipeline progress marker (``fetched`` initially,
                advanced through the stage ladder by ``set_artifact_status``).
            uri: physical storage URI; defaults to ``source_url`` if absent.
        """
        if self._is_postgres:
            self._upsert_source_artifact_pg(
                sha256=sha256,
                source_urn=source_urn,
                source_version=source_version,
                source_url=source_url,
                bytes_size=bytes_size,
                content_type=content_type,
                parser_module=parser_module,
                parser_function=parser_function,
                parser_version=parser_version,
                parser_commit=parser_commit,
                operator=operator,
                licence_class=licence_class,
                redistribution_class=redistribution_class,
                source_publication_date=source_publication_date,
                ingestion_run_id=ingestion_run_id,
                status=status,
                uri=uri or source_url or "",
            )
        else:
            self._upsert_source_artifact_sqlite(
                sha256=sha256,
                source_urn=source_urn,
                source_version=source_version,
                source_url=source_url,
                bytes_size=bytes_size,
                content_type=content_type,
                parser_module=parser_module,
                parser_function=parser_function,
                parser_version=parser_version,
                parser_commit=parser_commit,
                operator=operator,
                licence_class=licence_class,
                redistribution_class=redistribution_class,
                source_publication_date=source_publication_date,
                ingestion_run_id=ingestion_run_id,
                status=status,
                uri=uri or source_url or "",
            )

    def set_source_artifact_status(
        self,
        *,
        ingestion_run_id: str,
        status: str,
    ) -> None:
        """Advance the ``status`` of every source_artifact row for a run.

        The Phase 3 plan §"Artifact storage contract" requires the
        artifact row's status to track pipeline progress: ``fetched``
        after stage 1, ``parsed`` after stage 2, etc. This is the cheap
        per-stage update that walks the artifact row through that
        ladder.
        """
        if self._is_postgres:
            self._set_source_artifact_status_pg(ingestion_run_id, status)
        else:
            self._set_source_artifact_status_sqlite(ingestion_run_id, status)

    def set_artifact(
        self,
        run_id: str,
        *,
        artifact_id: str,
        sha256: str,
        parser_module: Optional[str] = None,
        parser_version: Optional[str] = None,
        parser_commit: Optional[str] = None,
    ) -> None:
        """Record the fetched artifact + parser pin on a run row.

        Phase 3 plan §"Artifact storage contract" requires every certified
        factor to trace to a (raw_artifact_uri, raw_artifact_sha256, parser_*)
        triple. The runner calls this helper at the end of stage 1 (fetch)
        and refreshes ``parser_*`` at the end of stage 2 (parse).
        """
        if self._is_postgres:
            self._set_artifact_pg(
                run_id, artifact_id, sha256, parser_module, parser_version, parser_commit
            )
        else:
            self._set_artifact_sqlite(
                run_id, artifact_id, sha256, parser_module, parser_version, parser_commit
            )

    def set_diff(
        self,
        run_id: str,
        *,
        diff_json_uri: Optional[str],
        diff_md_uri: Optional[str],
        summary_json: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist the stage-6 diff artefact pointers + summary.

        Two writes happen: a column update on ``ingestion_runs`` for the
        URIs (so a status query can render them inline) and a full row
        in ``ingestion_run_diffs`` carrying the JSON summary blob.
        """
        if self._is_postgres:
            self._set_diff_pg(run_id, diff_json_uri, diff_md_uri, summary_json)
        else:
            self._set_diff_sqlite(run_id, diff_json_uri, diff_md_uri, summary_json)

    def set_publish(
        self,
        run_id: str,
        *,
        batch_id: str,
        approved_by: str,
    ) -> None:
        """Stamp ``batch_id`` + ``approved_by`` on a run at publish time."""
        if self._is_postgres:
            self._set_publish_pg(run_id, batch_id, approved_by)
        else:
            self._set_publish_sqlite(run_id, batch_id, approved_by)

    def append_stage_history(
        self,
        run_id: str,
        result: StageResult,
    ) -> None:
        """Append a :class:`StageResult` receipt to the stage-history table."""
        row = result.to_row()
        if self._is_postgres:
            self._append_stage_pg(run_id, row)
        else:
            self._append_stage_sqlite(run_id, row)

    def get(self, run_id: str) -> IngestionRun:
        """Load a run by id; raises :class:`IngestionRunNotFoundError` on miss."""
        if self._is_postgres:
            row = self._get_pg(run_id)
        else:
            row = self._get_sqlite(run_id)
        if row is None:
            raise IngestionRunNotFoundError(
                "ingestion_run not found run_id=%s" % run_id
            )
        return _row_to_run(row)

    def list_by_status(self, status: RunStatus) -> List[IngestionRun]:
        """Return every run currently in ``status`` (newest first)."""
        if self._is_postgres:
            rows = self._list_pg("status = %s", (status.value,))
        else:
            rows = self._list_sqlite("status = ?", (status.value,))
        return [_row_to_run(r) for r in rows]

    def list_by_source(self, source_urn: str) -> List[IngestionRun]:
        """Return every run for ``source_urn`` (newest first)."""
        if self._is_postgres:
            rows = self._list_pg("source_urn = %s", (source_urn,))
        else:
            rows = self._list_sqlite("source_urn = ?", (source_urn,))
        return [_row_to_run(r) for r in rows]

    # -- SQLite implementations -------------------------------------------

    def _insert_sqlite(self, run: IngestionRun) -> None:
        # Wrap the single INSERT in an explicit transaction so a crash
        # between create() and the first stage advance leaves NO partial
        # state in ingestion_runs (Phase 3 audit gap A).
        with self._sqlite_txn() as conn:
            conn.execute(
                "INSERT INTO ingestion_runs ("
                " run_id, source_urn, source_version, started_at, finished_at,"
                " status, current_stage, artifact_id, artifact_sha256,"
                " parser_module, parser_version, parser_commit, operator,"
                " batch_id, approved_by, error_json, diff_json_uri, diff_md_uri"
                ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    run.run_id,
                    run.source_urn,
                    run.source_version,
                    run.started_at.isoformat() if run.started_at else None,
                    run.finished_at.isoformat() if run.finished_at else None,
                    run.status.value,
                    run.current_stage.value if run.current_stage else None,
                    run.artifact_id,
                    run.artifact_sha256,
                    run.parser_module,
                    run.parser_version,
                    run.parser_commit,
                    run.operator,
                    run.batch_id,
                    run.approved_by,
                    json.dumps(run.error_json) if run.error_json else None,
                    run.diff_json_uri,
                    run.diff_md_uri,
                ),
            )

    def _update_status_sqlite(
        self,
        run_id: str,
        status: RunStatus,
        current_stage: Optional[Stage],
        error_json: Optional[Dict[str, Any]],
        finished_at: Optional[datetime],
    ) -> None:
        # Phase 3 audit gap A: stage transitions MUST be one atomic txn.
        with self._sqlite_txn() as conn:
            sets = ["status = ?"]
            params: List[Any] = [status.value]
            if current_stage is not None:
                sets.append("current_stage = ?")
                params.append(current_stage.value)
            if error_json is not None:
                sets.append("error_json = ?")
                params.append(json.dumps(error_json))
            if finished_at is not None:
                sets.append("finished_at = ?")
                params.append(finished_at.isoformat())
            params.append(run_id)
            sql = "UPDATE ingestion_runs SET " + ", ".join(sets) + " WHERE run_id = ?"
            conn.execute(sql, tuple(params))

    def _set_artifact_sqlite(
        self,
        run_id: str,
        artifact_id: str,
        sha256: str,
        parser_module: Optional[str],
        parser_version: Optional[str],
        parser_commit: Optional[str],
    ) -> None:
        # Phase 3 audit gap A: atomic stage write.
        with self._sqlite_txn() as conn:
            conn.execute(
                "UPDATE ingestion_runs SET artifact_id = ?, artifact_sha256 = ?,"
                " parser_module = COALESCE(?, parser_module),"
                " parser_version = COALESCE(?, parser_version),"
                " parser_commit = COALESCE(?, parser_commit)"
                " WHERE run_id = ?",
                (artifact_id, sha256, parser_module, parser_version, parser_commit, run_id),
            )

    # -- Phase 3 audit gap C: source_artifacts upsert + status ladder ---

    def _upsert_source_artifact_sqlite(
        self,
        *,
        sha256: str,
        source_urn: str,
        source_version: str,
        source_url: Optional[str],
        bytes_size: Optional[int],
        content_type: Optional[str],
        parser_module: Optional[str],
        parser_function: Optional[str],
        parser_version: Optional[str],
        parser_commit: Optional[str],
        operator: Optional[str],
        licence_class: Optional[str],
        redistribution_class: Optional[str],
        source_publication_date: Optional[str],
        ingestion_run_id: str,
        status: str,
        uri: str,
    ) -> None:
        """Upsert one row into ``alpha_source_artifacts_v0_1``.

        Atomic via the BEGIN/COMMIT wrapper. Uses ``ON CONFLICT(sha256)``
        so re-fetching the same bytes refreshes the run-scoped fields
        (operator / ingestion_run_id / status) without duplicating the
        row.
        """
        with self._sqlite_txn() as conn:
            conn.execute(
                "INSERT INTO alpha_source_artifacts_v0_1 ("
                " sha256, source_urn, source_version, uri, content_type, size_bytes,"
                " parser_id, parser_version, parser_commit, ingested_at,"
                " source_url, source_publication_date, parser_module,"
                " parser_function, operator, licence_class, redistribution_class,"
                " ingestion_run_id, status"
                ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
                " ON CONFLICT(sha256) DO UPDATE SET"
                " source_url = COALESCE(excluded.source_url, source_url),"
                " bytes_size_dummy_skip_via_size_bytes = bytes_size_dummy_skip_via_size_bytes",
                # The ON CONFLICT clause above is rewritten below — SQLite
                # doesn't accept the synthetic placeholder, so we run a
                # follow-up UPDATE on conflict instead. The INSERT above
                # is wrapped in INSERT OR IGNORE-style logic via the next
                # statement to keep the upsert semantics simple.
                (
                    sha256,
                    source_urn,
                    source_version,
                    uri,
                    content_type,
                    bytes_size,
                    parser_function or parser_module,  # legacy parser_id slot
                    parser_version,
                    parser_commit,
                    now_utc().isoformat(),
                    source_url,
                    source_publication_date,
                    parser_module,
                    parser_function,
                    operator,
                    licence_class,
                    redistribution_class,
                    ingestion_run_id,
                    status,
                ),
            ) if False else None  # disabled — see follow-up logic
            # Simpler upsert: try INSERT OR IGNORE; then UPDATE the
            # run-scoped fields by sha256 unconditionally.
            conn.execute(
                "INSERT OR IGNORE INTO alpha_source_artifacts_v0_1 ("
                " sha256, source_urn, source_version, uri, content_type, size_bytes,"
                " parser_id, parser_version, parser_commit, ingested_at,"
                " source_url, source_publication_date, parser_module,"
                " parser_function, operator, licence_class, redistribution_class,"
                " ingestion_run_id, status"
                ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    sha256,
                    source_urn,
                    source_version,
                    uri,
                    content_type,
                    bytes_size,
                    parser_function or parser_module,
                    parser_version,
                    parser_commit,
                    now_utc().isoformat(),
                    source_url,
                    source_publication_date,
                    parser_module,
                    parser_function,
                    operator,
                    licence_class,
                    redistribution_class,
                    ingestion_run_id,
                    status,
                ),
            )
            conn.execute(
                "UPDATE alpha_source_artifacts_v0_1 SET"
                " source_url = COALESCE(?, source_url),"
                " content_type = COALESCE(?, content_type),"
                " size_bytes = COALESCE(?, size_bytes),"
                " parser_id = COALESCE(?, parser_id),"
                " parser_version = COALESCE(?, parser_version),"
                " parser_commit = COALESCE(?, parser_commit),"
                " parser_module = COALESCE(?, parser_module),"
                " parser_function = COALESCE(?, parser_function),"
                " source_publication_date = COALESCE(?, source_publication_date),"
                " operator = COALESCE(?, operator),"
                " licence_class = COALESCE(?, licence_class),"
                " redistribution_class = COALESCE(?, redistribution_class),"
                " ingestion_run_id = COALESCE(?, ingestion_run_id),"
                " status = COALESCE(?, status)"
                " WHERE sha256 = ?",
                (
                    source_url,
                    content_type,
                    bytes_size,
                    parser_function or parser_module,
                    parser_version,
                    parser_commit,
                    parser_module,
                    parser_function,
                    source_publication_date,
                    operator,
                    licence_class,
                    redistribution_class,
                    ingestion_run_id,
                    status,
                    sha256,
                ),
            )

    def _set_source_artifact_status_sqlite(
        self, ingestion_run_id: str, status: str
    ) -> None:
        with self._sqlite_txn() as conn:
            conn.execute(
                "UPDATE alpha_source_artifacts_v0_1 SET status = ? "
                "WHERE ingestion_run_id = ?",
                (status, ingestion_run_id),
            )

    def _set_diff_sqlite(
        self,
        run_id: str,
        diff_json_uri: Optional[str],
        diff_md_uri: Optional[str],
        summary_json: Optional[Dict[str, Any]],
    ) -> None:
        # Phase 3 audit gap A: atomic stage write — both rows commit
        # together or neither does. A mid-write crash must NOT leave
        # the run row carrying a diff URI pointer with no matching
        # ingestion_run_diffs summary row.
        with self._sqlite_txn() as conn:
            conn.execute(
                "UPDATE ingestion_runs SET diff_json_uri = ?, diff_md_uri = ? WHERE run_id = ?",
                (diff_json_uri, diff_md_uri, run_id),
            )
            conn.execute(
                "INSERT INTO ingestion_run_diffs (run_id, diff_json_uri, diff_md_uri,"
                " summary_json, created_at) VALUES (?,?,?,?,?)"
                " ON CONFLICT(run_id) DO UPDATE SET"
                " diff_json_uri=excluded.diff_json_uri,"
                " diff_md_uri=excluded.diff_md_uri,"
                " summary_json=excluded.summary_json,"
                " created_at=excluded.created_at",
                (
                    run_id,
                    diff_json_uri,
                    diff_md_uri,
                    json.dumps(summary_json, sort_keys=True) if summary_json else None,
                    now_utc().isoformat(),
                ),
            )

    def _set_publish_sqlite(
        self, run_id: str, batch_id: str, approved_by: str
    ) -> None:
        # Phase 3 audit gap A: atomic stage write.
        with self._sqlite_txn() as conn:
            conn.execute(
                "UPDATE ingestion_runs SET batch_id = ?, approved_by = ? WHERE run_id = ?",
                (batch_id, approved_by, run_id),
            )

    def _append_stage_sqlite(self, run_id: str, row: Dict[str, Any]) -> None:
        # Phase 3 audit gap A: atomic stage write.
        with self._sqlite_txn() as conn:
            conn.execute(
                "INSERT INTO ingestion_run_stage_history ("
                " run_id, stage, ok, duration_s, error, details_json,"
                " started_at, finished_at"
                ") VALUES (?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    row["stage"],
                    1 if row["ok"] else 0,
                    row["duration_s"],
                    row.get("error"),
                    json.dumps(row.get("details") or {}, sort_keys=True),
                    row["started_at"],
                    row.get("finished_at"),
                ),
            )

    def _get_sqlite(self, run_id: str) -> Optional[Dict[str, Any]]:
        conn = self._connect()
        try:
            cur = conn.execute(
                "SELECT * FROM ingestion_runs WHERE run_id = ?", (run_id,)
            )
            r = cur.fetchone()
        finally:
            self._maybe_close(conn)
        return dict(r) if r is not None else None

    def _list_sqlite(
        self, where: str, params: Tuple[Any, ...]
    ) -> List[Dict[str, Any]]:
        conn = self._connect()
        try:
            sql = (
                "SELECT * FROM ingestion_runs WHERE "
                + where
                + " ORDER BY started_at DESC"
            )
            cur = conn.execute(sql, params)
            rows = cur.fetchall()
        finally:
            self._maybe_close(conn)
        return [dict(r) for r in rows]

    # -- Postgres implementations (lazy psycopg) --------------------------

    def _pg_connect(self) -> Any:
        import psycopg  # type: ignore  # noqa: PLC0415
        return psycopg.connect(self._dsn)

    def _insert_pg(self, run: IngestionRun) -> None:
        with self._pg_connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO ingestion_runs ("
                    " run_id, source_urn, source_version, started_at, finished_at,"
                    " status, current_stage, artifact_id, artifact_sha256,"
                    " parser_module, parser_version, parser_commit, operator,"
                    " batch_id, approved_by, error_json, diff_json_uri, diff_md_uri"
                    ") VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    (
                        run.run_id,
                        run.source_urn,
                        run.source_version,
                        run.started_at,
                        run.finished_at,
                        run.status.value,
                        run.current_stage.value if run.current_stage else None,
                        run.artifact_id,
                        run.artifact_sha256,
                        run.parser_module,
                        run.parser_version,
                        run.parser_commit,
                        run.operator,
                        run.batch_id,
                        run.approved_by,
                        json.dumps(run.error_json) if run.error_json else None,
                        run.diff_json_uri,
                        run.diff_md_uri,
                    ),
                )
            conn.commit()

    def _update_status_pg(
        self,
        run_id: str,
        status: RunStatus,
        current_stage: Optional[Stage],
        error_json: Optional[Dict[str, Any]],
        finished_at: Optional[datetime],
    ) -> None:
        sets = ["status = %s"]
        params: List[Any] = [status.value]
        if current_stage is not None:
            sets.append("current_stage = %s")
            params.append(current_stage.value)
        if error_json is not None:
            sets.append("error_json = %s")
            params.append(json.dumps(error_json))
        if finished_at is not None:
            sets.append("finished_at = %s")
            params.append(finished_at)
        params.append(run_id)
        sql = "UPDATE ingestion_runs SET " + ", ".join(sets) + " WHERE run_id = %s"
        with self._pg_connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, tuple(params))
            conn.commit()

    def _set_artifact_pg(
        self,
        run_id: str,
        artifact_id: str,
        sha256: str,
        parser_module: Optional[str],
        parser_version: Optional[str],
        parser_commit: Optional[str],
    ) -> None:
        with self._pg_connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE ingestion_runs SET artifact_id = %s, artifact_sha256 = %s,"
                    " parser_module = COALESCE(%s, parser_module),"
                    " parser_version = COALESCE(%s, parser_version),"
                    " parser_commit = COALESCE(%s, parser_commit)"
                    " WHERE run_id = %s",
                    (artifact_id, sha256, parser_module, parser_version, parser_commit, run_id),
                )
            conn.commit()

    def _upsert_source_artifact_pg(
        self,
        *,
        sha256: str,
        source_urn: str,
        source_version: str,
        source_url: Optional[str],
        bytes_size: Optional[int],
        content_type: Optional[str],
        parser_module: Optional[str],
        parser_function: Optional[str],
        parser_version: Optional[str],
        parser_commit: Optional[str],
        operator: Optional[str],
        licence_class: Optional[str],
        redistribution_class: Optional[str],
        source_publication_date: Optional[str],
        ingestion_run_id: str,
        status: str,
        uri: str,
    ) -> None:
        """Postgres path for the Phase 3 audit gap C upsert."""
        with self._pg_connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO factors_v0_1.source_artifacts ("
                    " sha256, source_urn, source_version, uri, content_type, size_bytes,"
                    " parser_id, parser_version, parser_commit,"
                    " source_url, source_publication_date, parser_module,"
                    " parser_function, operator, licence_class, redistribution_class,"
                    " ingestion_run_id, status"
                    ") VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
                    " ON CONFLICT (sha256) DO UPDATE SET"
                    "  source_url = COALESCE(EXCLUDED.source_url, factors_v0_1.source_artifacts.source_url),"
                    "  content_type = COALESCE(EXCLUDED.content_type, factors_v0_1.source_artifacts.content_type),"
                    "  size_bytes = COALESCE(EXCLUDED.size_bytes, factors_v0_1.source_artifacts.size_bytes),"
                    "  parser_id = COALESCE(EXCLUDED.parser_id, factors_v0_1.source_artifacts.parser_id),"
                    "  parser_version = COALESCE(EXCLUDED.parser_version, factors_v0_1.source_artifacts.parser_version),"
                    "  parser_commit = COALESCE(EXCLUDED.parser_commit, factors_v0_1.source_artifacts.parser_commit),"
                    "  parser_module = COALESCE(EXCLUDED.parser_module, factors_v0_1.source_artifacts.parser_module),"
                    "  parser_function = COALESCE(EXCLUDED.parser_function, factors_v0_1.source_artifacts.parser_function),"
                    "  source_publication_date = COALESCE(EXCLUDED.source_publication_date, factors_v0_1.source_artifacts.source_publication_date),"
                    "  operator = COALESCE(EXCLUDED.operator, factors_v0_1.source_artifacts.operator),"
                    "  licence_class = COALESCE(EXCLUDED.licence_class, factors_v0_1.source_artifacts.licence_class),"
                    "  redistribution_class = COALESCE(EXCLUDED.redistribution_class, factors_v0_1.source_artifacts.redistribution_class),"
                    "  ingestion_run_id = EXCLUDED.ingestion_run_id,"
                    "  status = EXCLUDED.status",
                    (
                        sha256,
                        source_urn,
                        source_version,
                        uri,
                        content_type,
                        bytes_size,
                        parser_function or parser_module,
                        parser_version,
                        parser_commit,
                        source_url,
                        source_publication_date,
                        parser_module,
                        parser_function,
                        operator,
                        licence_class,
                        redistribution_class,
                        ingestion_run_id,
                        status,
                    ),
                )
            conn.commit()

    def _set_source_artifact_status_pg(
        self, ingestion_run_id: str, status: str
    ) -> None:
        with self._pg_connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE factors_v0_1.source_artifacts SET status = %s "
                    "WHERE ingestion_run_id = %s",
                    (status, ingestion_run_id),
                )
            conn.commit()

    def _set_diff_pg(
        self,
        run_id: str,
        diff_json_uri: Optional[str],
        diff_md_uri: Optional[str],
        summary_json: Optional[Dict[str, Any]],
    ) -> None:
        with self._pg_connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE ingestion_runs SET diff_json_uri = %s, diff_md_uri = %s WHERE run_id = %s",
                    (diff_json_uri, diff_md_uri, run_id),
                )
                cur.execute(
                    "INSERT INTO ingestion_run_diffs (run_id, diff_json_uri, diff_md_uri,"
                    " summary_json, created_at) VALUES (%s,%s,%s,%s,%s)"
                    " ON CONFLICT (run_id) DO UPDATE SET"
                    " diff_json_uri=EXCLUDED.diff_json_uri,"
                    " diff_md_uri=EXCLUDED.diff_md_uri,"
                    " summary_json=EXCLUDED.summary_json,"
                    " created_at=EXCLUDED.created_at",
                    (
                        run_id,
                        diff_json_uri,
                        diff_md_uri,
                        json.dumps(summary_json, sort_keys=True) if summary_json else None,
                        now_utc(),
                    ),
                )
            conn.commit()

    def _set_publish_pg(
        self, run_id: str, batch_id: str, approved_by: str
    ) -> None:
        with self._pg_connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE ingestion_runs SET batch_id = %s, approved_by = %s WHERE run_id = %s",
                    (batch_id, approved_by, run_id),
                )
            conn.commit()

    def _append_stage_pg(self, run_id: str, row: Dict[str, Any]) -> None:
        with self._pg_connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO ingestion_run_stage_history ("
                    " run_id, stage, ok, duration_s, error, details_json,"
                    " started_at, finished_at"
                    ") VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                    (
                        run_id,
                        row["stage"],
                        bool(row["ok"]),
                        row["duration_s"],
                        row.get("error"),
                        json.dumps(row.get("details") or {}, sort_keys=True),
                        row["started_at"],
                        row.get("finished_at"),
                    ),
                )
            conn.commit()

    def _get_pg(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._pg_connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT run_id, source_urn, source_version, started_at, finished_at,"
                    " status, current_stage, artifact_id, artifact_sha256,"
                    " parser_module, parser_version, parser_commit, operator,"
                    " batch_id, approved_by, error_json, diff_json_uri, diff_md_uri"
                    " FROM ingestion_runs WHERE run_id = %s",
                    (run_id,),
                )
                r = cur.fetchone()
        if r is None:
            return None
        cols = (
            "run_id", "source_urn", "source_version", "started_at", "finished_at",
            "status", "current_stage", "artifact_id", "artifact_sha256",
            "parser_module", "parser_version", "parser_commit", "operator",
            "batch_id", "approved_by", "error_json", "diff_json_uri", "diff_md_uri",
        )
        return dict(zip(cols, r))

    def _list_pg(
        self, where: str, params: Tuple[Any, ...]
    ) -> List[Dict[str, Any]]:
        sql = (
            "SELECT run_id, source_urn, source_version, started_at, finished_at,"
            " status, current_stage, artifact_id, artifact_sha256,"
            " parser_module, parser_version, parser_commit, operator,"
            " batch_id, approved_by, error_json, diff_json_uri, diff_md_uri"
            " FROM ingestion_runs WHERE "
            + where
            + " ORDER BY started_at DESC"
        )
        with self._pg_connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        cols = (
            "run_id", "source_urn", "source_version", "started_at", "finished_at",
            "status", "current_stage", "artifact_id", "artifact_sha256",
            "parser_module", "parser_version", "parser_commit", "operator",
            "batch_id", "approved_by", "error_json", "diff_json_uri", "diff_md_uri",
        )
        return [dict(zip(cols, r)) for r in rows]


# ---------------------------------------------------------------------------
# Row -> dataclass helper
# ---------------------------------------------------------------------------


def _row_to_run(row: Dict[str, Any]) -> IngestionRun:
    """Materialise a DB row dict into an :class:`IngestionRun` dataclass."""
    started_raw = row.get("started_at")
    finished_raw = row.get("finished_at")
    error_raw = row.get("error_json")
    error_json: Optional[Dict[str, Any]] = None
    if error_raw:
        if isinstance(error_raw, dict):
            error_json = error_raw
        else:
            try:
                error_json = json.loads(error_raw)
            except (TypeError, json.JSONDecodeError):
                error_json = {"raw": str(error_raw)}

    started_at = _parse_dt(started_raw) or now_utc()
    finished_at = _parse_dt(finished_raw)
    status = RunStatus(row.get("status") or RunStatus.CREATED.value)
    cur_stage_val = row.get("current_stage")
    current_stage = Stage(cur_stage_val) if cur_stage_val else None

    return IngestionRun(
        run_id=str(row.get("run_id")),
        source_urn=str(row.get("source_urn") or ""),
        source_version=str(row.get("source_version") or ""),
        started_at=started_at,
        finished_at=finished_at,
        status=status,
        current_stage=current_stage,
        artifact_id=row.get("artifact_id"),
        artifact_sha256=row.get("artifact_sha256"),
        parser_module=row.get("parser_module"),
        parser_version=row.get("parser_version"),
        parser_commit=row.get("parser_commit"),
        operator=str(row.get("operator") or ""),
        batch_id=row.get("batch_id"),
        approved_by=row.get("approved_by"),
        error_json=error_json,
        diff_json_uri=row.get("diff_json_uri"),
        diff_md_uri=row.get("diff_md_uri"),
    )


def _parse_dt(value: Any) -> Optional[datetime]:
    """Parse a stored timestamp into a tz-aware ``datetime``.

    Accepts ISO-8601 strings (SQLite path) and pre-built ``datetime``
    objects (Postgres path); returns ``None`` for empty values.
    """
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            from datetime import timezone  # noqa: PLC0415
            return value.replace(tzinfo=timezone.utc)
        return value
    try:
        # Python 3.11+ supports trailing 'Z' via fromisoformat in 3.11+
        text = str(value).replace("Z", "+00:00")
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            from datetime import timezone  # noqa: PLC0415
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None
