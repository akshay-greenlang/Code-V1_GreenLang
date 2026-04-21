# -*- coding: utf-8 -*-
"""
Factors Batch Job orchestration (GAP-11).

Customers with large portfolios (10k+ activities) want to submit a
single API call and poll later for results instead of sharding the
work themselves.  This module provides:

    * :class:`BatchJob` — persisted job record (queued, running, done, …).
    * :class:`BatchJobQueue` — abstract queue interface.
    * :class:`SQLiteBatchJobQueue` — dev / local backing store.
    * :class:`PostgresBatchJobQueue` — production backing store
      (targets the ``factors_batch_jobs`` table in V444).
    * :func:`submit_batch_resolution` — entry point for callers.
    * :func:`process_next_job` — single-job worker step.  Celery wraps
      this as a task; local dev runs it in a tight loop.

Webhook delivery on completion reuses the existing
:mod:`greenlang.factors.webhooks` infrastructure.

CTO non-negotiables honoured:
    #1 Gas-level storage — we serialise the full ``ResolvedFactor.gas_breakdown``
       in each row of the results page; never collapsed to CO2e.
    #2 Append-only versioning — job rows are updated in place (status
       transitions), but results + request payload are immutable.
    #3 Tier + rate limits enforced before a job enters the queue.

Author: GL-BackendDeveloper
Gap: GAP-11
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants + enums
# ---------------------------------------------------------------------------


class BatchJobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BatchJobType(str, Enum):
    RESOLVE = "resolve"
    SEARCH = "search"
    MATCH = "match"
    DIFF = "diff"


#: Maximum request count per batch, per tier.  Enforced pre-submit.
MAX_REQUESTS_PER_TIER: Dict[str, int] = {
    "community": 10,
    "pro": 100,
    "consulting": 1000,
    "enterprise": 10_000,
    "internal": 50_000,
}


def max_batch_size_for_tier(tier: Optional[str]) -> int:
    """Return the max requests-per-batch cap for ``tier``."""
    t = (tier or "community").strip().lower()
    return MAX_REQUESTS_PER_TIER.get(t, MAX_REQUESTS_PER_TIER["community"])


class BatchJobError(RuntimeError):
    """Base class for batch-job errors."""


class BatchJobNotFoundError(BatchJobError):
    pass


class BatchJobLimitError(BatchJobError):
    pass


class BatchJobStateError(BatchJobError):
    pass


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class BatchJob:
    """Persisted batch-job record."""

    job_id: str
    tenant_id: str
    job_type: BatchJobType
    status: BatchJobStatus
    request_count: int
    completed_count: int = 0
    failed_count: int = 0
    submitted_at: str = ""
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    results_uri: Optional[str] = None
    request_payload_uri: Optional[str] = None
    error_log: List[Dict[str, Any]] = field(default_factory=list)
    webhook_url: Optional[str] = None
    webhook_secret_ref: Optional[str] = None
    created_by: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "tenant_id": self.tenant_id,
            "job_type": self.job_type.value,
            "status": self.status.value,
            "request_count": self.request_count,
            "completed_count": self.completed_count,
            "failed_count": self.failed_count,
            "submitted_at": self.submitted_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "results_uri": self.results_uri,
            "request_payload_uri": self.request_payload_uri,
            "error_log": list(self.error_log),
            "webhook_url": self.webhook_url,
            "webhook_secret_ref": self.webhook_secret_ref,
            "created_by": self.created_by,
        }


@dataclass
class BatchJobHandle:
    """Lightweight POST response object — the caller polls with the job_id."""

    job_id: str
    tenant_id: str
    status: BatchJobStatus
    submitted_at: str
    request_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "tenant_id": self.tenant_id,
            "status": self.status.value,
            "submitted_at": self.submitted_at,
            "request_count": self.request_count,
        }


# ---------------------------------------------------------------------------
# Queue interface
# ---------------------------------------------------------------------------


class BatchJobQueue(ABC):
    """Abstract persistence layer for batch jobs.

    Implementations MUST be thread-safe.  ``put_results`` is separate
    from ``update`` because results + payload are typically written to
    object storage (S3) while the job row only persists the URI.
    """

    @abstractmethod
    def enqueue(self, job: BatchJob, request_payload: List[Dict[str, Any]]) -> None:
        ...

    @abstractmethod
    def get(self, job_id: str) -> Optional[BatchJob]:
        ...

    @abstractmethod
    def list_for_tenant(
        self,
        tenant_id: str,
        *,
        status: Optional[BatchJobStatus] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Tuple[List[BatchJob], int]:
        ...

    @abstractmethod
    def update(self, job: BatchJob) -> None:
        ...

    @abstractmethod
    def delete(self, job_id: str) -> bool:
        ...

    @abstractmethod
    def next_queued(self) -> Optional[BatchJob]:
        ...

    @abstractmethod
    def put_results(self, job_id: str, results: List[Dict[str, Any]]) -> str:
        """Persist the results payload; return the ``results_uri`` stored on the row."""

    @abstractmethod
    def get_results(
        self,
        job_id: str,
        *,
        cursor: int = 0,
        limit: int = 1000,
    ) -> Tuple[List[Dict[str, Any]], int, int]:
        """Return ``(page, next_cursor, total)`` from the results store."""

    @abstractmethod
    def get_request_payload(self, job_id: str) -> List[Dict[str, Any]]:
        ...


# ---------------------------------------------------------------------------
# SQLite implementation (dev + local tests)
# ---------------------------------------------------------------------------


_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS factors_batch_jobs (
    job_id                TEXT PRIMARY KEY,
    tenant_id             TEXT NOT NULL,
    job_type              TEXT NOT NULL,
    status                TEXT NOT NULL,
    submitted_at          TEXT NOT NULL,
    started_at            TEXT,
    completed_at          TEXT,
    request_count         INTEGER NOT NULL,
    completed_count       INTEGER NOT NULL DEFAULT 0,
    failed_count          INTEGER NOT NULL DEFAULT 0,
    results_uri           TEXT,
    request_payload_uri   TEXT,
    error_log_json        TEXT,
    webhook_url           TEXT,
    webhook_secret_ref    TEXT,
    created_by            TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_bj_tenant_status
    ON factors_batch_jobs (tenant_id, status);
CREATE INDEX IF NOT EXISTS idx_bj_status_submitted
    ON factors_batch_jobs (status, submitted_at);

CREATE TABLE IF NOT EXISTS factors_batch_job_payloads (
    job_id       TEXT PRIMARY KEY,
    payload_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS factors_batch_job_results (
    job_id        TEXT NOT NULL,
    result_index  INTEGER NOT NULL,
    result_json   TEXT NOT NULL,
    PRIMARY KEY (job_id, result_index)
);
CREATE INDEX IF NOT EXISTS idx_bjr_job
    ON factors_batch_job_results (job_id, result_index);
"""


class SQLiteBatchJobQueue(BatchJobQueue):
    """SQLite-backed queue; fine for dev, single-process test environments."""

    def __init__(self, sqlite_path: Union[str, Path, None] = None) -> None:
        self._lock = threading.Lock()
        if sqlite_path is None:
            self._conn = sqlite3.connect(
                ":memory:", check_same_thread=False, isolation_level=None
            )
        else:
            p = Path(sqlite_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(
                str(p), isolation_level=None, check_same_thread=False
            )
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SQLITE_SCHEMA)

    # ---- Core ops ----

    def enqueue(self, job: BatchJob, request_payload: List[Dict[str, Any]]) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO factors_batch_jobs (
                    job_id, tenant_id, job_type, status, submitted_at,
                    started_at, completed_at, request_count, completed_count,
                    failed_count, results_uri, request_payload_uri, error_log_json,
                    webhook_url, webhook_secret_ref, created_by
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.job_id, job.tenant_id, job.job_type.value,
                    job.status.value, job.submitted_at, job.started_at,
                    job.completed_at, job.request_count, job.completed_count,
                    job.failed_count, job.results_uri,
                    job.request_payload_uri, json.dumps(job.error_log),
                    job.webhook_url, job.webhook_secret_ref, job.created_by,
                ),
            )
            self._conn.execute(
                "INSERT OR REPLACE INTO factors_batch_job_payloads "
                "(job_id, payload_json) VALUES (?, ?)",
                (job.job_id, json.dumps(request_payload, default=str)),
            )
            # Stamp the payload URI as an internal marker so the job
            # record self-documents which storage backend owns it.
            self._conn.execute(
                "UPDATE factors_batch_jobs SET request_payload_uri = ? "
                "WHERE job_id = ?",
                ("sqlite://factors_batch_job_payloads/%s" % job.job_id, job.job_id),
            )

    def get(self, job_id: str) -> Optional[BatchJob]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM factors_batch_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            cols = [c[0] for c in self._conn.execute(
                "SELECT * FROM factors_batch_jobs LIMIT 0"
            ).description]
        if row is None:
            return None
        return _row_to_job(cols, row)

    def list_for_tenant(
        self,
        tenant_id: str,
        *,
        status: Optional[BatchJobStatus] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Tuple[List[BatchJob], int]:
        with self._lock:
            params: List[Any] = [tenant_id]
            where = "WHERE tenant_id = ?"
            if status is not None:
                where += " AND status = ?"
                params.append(status.value)
            total = self._conn.execute(
                "SELECT COUNT(*) FROM factors_batch_jobs " + where,
                params,
            ).fetchone()[0]
            rows = self._conn.execute(
                "SELECT * FROM factors_batch_jobs "
                + where
                + " ORDER BY submitted_at DESC LIMIT ? OFFSET ?",
                params + [limit, offset],
            ).fetchall()
            cols = [c[0] for c in self._conn.execute(
                "SELECT * FROM factors_batch_jobs LIMIT 0"
            ).description]
        return [_row_to_job(cols, r) for r in rows], int(total)

    def update(self, job: BatchJob) -> None:
        with self._lock:
            self._conn.execute(
                """
                UPDATE factors_batch_jobs SET
                    status = ?, started_at = ?, completed_at = ?,
                    completed_count = ?, failed_count = ?,
                    results_uri = ?, error_log_json = ?
                WHERE job_id = ?
                """,
                (
                    job.status.value, job.started_at, job.completed_at,
                    job.completed_count, job.failed_count,
                    job.results_uri, json.dumps(job.error_log),
                    job.job_id,
                ),
            )

    def delete(self, job_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM factors_batch_jobs WHERE job_id = ?",
                (job_id,),
            )
            self._conn.execute(
                "DELETE FROM factors_batch_job_payloads WHERE job_id = ?",
                (job_id,),
            )
            self._conn.execute(
                "DELETE FROM factors_batch_job_results WHERE job_id = ?",
                (job_id,),
            )
            return cur.rowcount > 0

    def next_queued(self) -> Optional[BatchJob]:
        """Pop (atomically) the oldest QUEUED job and flip it to RUNNING."""
        with self._lock:
            row = self._conn.execute(
                """
                SELECT * FROM factors_batch_jobs
                WHERE status = ?
                ORDER BY submitted_at ASC
                LIMIT 1
                """,
                (BatchJobStatus.QUEUED.value,),
            ).fetchone()
            if row is None:
                return None
            cols = [c[0] for c in self._conn.execute(
                "SELECT * FROM factors_batch_jobs LIMIT 0"
            ).description]
            job = _row_to_job(cols, row)
            job.status = BatchJobStatus.RUNNING
            job.started_at = datetime.now(timezone.utc).isoformat()
            self._conn.execute(
                "UPDATE factors_batch_jobs SET status = ?, started_at = ? "
                "WHERE job_id = ?",
                (job.status.value, job.started_at, job.job_id),
            )
        return job

    # ---- Payload + results ----

    def put_results(self, job_id: str, results: List[Dict[str, Any]]) -> str:
        with self._lock:
            self._conn.execute(
                "DELETE FROM factors_batch_job_results WHERE job_id = ?",
                (job_id,),
            )
            rows = [
                (job_id, idx, json.dumps(r, default=str))
                for idx, r in enumerate(results)
            ]
            if rows:
                self._conn.executemany(
                    "INSERT INTO factors_batch_job_results "
                    "(job_id, result_index, result_json) VALUES (?, ?, ?)",
                    rows,
                )
            uri = "sqlite://factors_batch_job_results/%s" % job_id
            self._conn.execute(
                "UPDATE factors_batch_jobs SET results_uri = ? WHERE job_id = ?",
                (uri, job_id),
            )
        return uri

    def get_results(
        self,
        job_id: str,
        *,
        cursor: int = 0,
        limit: int = 1000,
    ) -> Tuple[List[Dict[str, Any]], int, int]:
        cursor = max(0, int(cursor))
        limit = max(1, min(10_000, int(limit)))
        with self._lock:
            total = self._conn.execute(
                "SELECT COUNT(*) FROM factors_batch_job_results WHERE job_id = ?",
                (job_id,),
            ).fetchone()[0]
            rows = self._conn.execute(
                """
                SELECT result_index, result_json
                FROM factors_batch_job_results
                WHERE job_id = ? AND result_index >= ?
                ORDER BY result_index ASC
                LIMIT ?
                """,
                (job_id, cursor, limit),
            ).fetchall()
        results: List[Dict[str, Any]] = []
        last_idx = cursor
        for idx, payload in rows:
            try:
                results.append(json.loads(payload))
            except json.JSONDecodeError:
                logger.warning(
                    "Corrupt result row skipped: job=%s idx=%s", job_id, idx
                )
                continue
            last_idx = int(idx)
        next_cursor = last_idx + 1 if rows else cursor
        if next_cursor >= total:
            next_cursor = total
        return results, next_cursor, int(total)

    def get_request_payload(self, job_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT payload_json FROM factors_batch_job_payloads WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        if row is None:
            return []
        try:
            return list(json.loads(row[0]))
        except (json.JSONDecodeError, TypeError):
            return []

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _row_to_job(cols: List[str], row: Tuple[Any, ...]) -> BatchJob:
    data: Dict[str, Any] = dict(zip(cols, row))
    try:
        error_log = json.loads(data.get("error_log_json") or "[]")
    except json.JSONDecodeError:
        error_log = []
    return BatchJob(
        job_id=data["job_id"],
        tenant_id=data["tenant_id"],
        job_type=BatchJobType(data["job_type"]),
        status=BatchJobStatus(data["status"]),
        submitted_at=data.get("submitted_at", ""),
        started_at=data.get("started_at"),
        completed_at=data.get("completed_at"),
        request_count=int(data.get("request_count", 0) or 0),
        completed_count=int(data.get("completed_count", 0) or 0),
        failed_count=int(data.get("failed_count", 0) or 0),
        results_uri=data.get("results_uri"),
        request_payload_uri=data.get("request_payload_uri"),
        error_log=list(error_log),
        webhook_url=data.get("webhook_url"),
        webhook_secret_ref=data.get("webhook_secret_ref"),
        created_by=data.get("created_by", ""),
    )


# ---------------------------------------------------------------------------
# Postgres implementation (prod)
#
# The Postgres backend targets V444__factors_batch_jobs.sql.  We keep it
# deliberately minimal and rely on duck-typed ``psycopg`` connection
# objects so this module has no hard dependency on psycopg at import time.
# ---------------------------------------------------------------------------


class PostgresBatchJobQueue(BatchJobQueue):
    """Production Postgres queue against ``factors_batch_jobs`` (V444).

    ``connection_factory`` MUST return a context-manager-compatible
    psycopg connection (``with connection_factory() as conn:``).  The
    caller is responsible for pool management.
    """

    def __init__(
        self,
        *,
        connection_factory: Callable[[], Any],
        results_writer: Optional[Callable[[str, List[Dict[str, Any]]], str]] = None,
        results_reader: Optional[
            Callable[[str, int, int], Tuple[List[Dict[str, Any]], int, int]]
        ] = None,
        payload_writer: Optional[Callable[[str, List[Dict[str, Any]]], str]] = None,
        payload_reader: Optional[Callable[[str], List[Dict[str, Any]]]] = None,
    ) -> None:
        self._conn_factory = connection_factory
        self._results_writer = results_writer or _raise_not_wired
        self._results_reader = results_reader or _raise_not_wired
        self._payload_writer = payload_writer or _raise_not_wired
        self._payload_reader = payload_reader or _raise_not_wired

    def enqueue(self, job: BatchJob, request_payload: List[Dict[str, Any]]) -> None:
        payload_uri = self._payload_writer(job.job_id, request_payload)
        job.request_payload_uri = payload_uri
        with self._conn_factory() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO factors_batch_jobs (
                        job_id, tenant_id, job_type, status, submitted_at,
                        started_at, completed_at, request_count, completed_count,
                        failed_count, results_uri, request_payload_uri,
                        error_log, webhook_url, webhook_secret_ref, created_by
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s::jsonb, %s, %s, %s
                    )
                    """,
                    (
                        job.job_id, job.tenant_id, job.job_type.value,
                        job.status.value, job.submitted_at, job.started_at,
                        job.completed_at, job.request_count,
                        job.completed_count, job.failed_count,
                        job.results_uri, job.request_payload_uri,
                        json.dumps(job.error_log),
                        job.webhook_url, job.webhook_secret_ref,
                        job.created_by,
                    ),
                )

    def get(self, job_id: str) -> Optional[BatchJob]:
        with self._conn_factory() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT job_id, tenant_id, job_type, status, submitted_at, "
                    "started_at, completed_at, request_count, completed_count, "
                    "failed_count, results_uri, request_payload_uri, "
                    "error_log, webhook_url, webhook_secret_ref, created_by "
                    "FROM factors_batch_jobs WHERE job_id = %s",
                    (job_id,),
                )
                row = cur.fetchone()
        if row is None:
            return None
        return _pg_row_to_job(row)

    def list_for_tenant(
        self,
        tenant_id: str,
        *,
        status: Optional[BatchJobStatus] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Tuple[List[BatchJob], int]:
        query = (
            "SELECT job_id, tenant_id, job_type, status, submitted_at, "
            "started_at, completed_at, request_count, completed_count, "
            "failed_count, results_uri, request_payload_uri, "
            "error_log, webhook_url, webhook_secret_ref, created_by "
            "FROM factors_batch_jobs WHERE tenant_id = %s"
        )
        params: List[Any] = [tenant_id]
        if status is not None:
            query += " AND status = %s"
            params.append(status.value)
        query += " ORDER BY submitted_at DESC LIMIT %s OFFSET %s"
        params.extend([limit, offset])

        with self._conn_factory() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
                cur.execute(
                    "SELECT COUNT(*) FROM factors_batch_jobs WHERE tenant_id = %s"
                    + (" AND status = %s" if status else ""),
                    [tenant_id] + ([status.value] if status else []),
                )
                total = cur.fetchone()[0]
        return [_pg_row_to_job(r) for r in rows], int(total)

    def update(self, job: BatchJob) -> None:
        with self._conn_factory() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE factors_batch_jobs SET
                        status = %s, started_at = %s, completed_at = %s,
                        completed_count = %s, failed_count = %s,
                        results_uri = %s, error_log = %s::jsonb
                    WHERE job_id = %s
                    """,
                    (
                        job.status.value, job.started_at, job.completed_at,
                        job.completed_count, job.failed_count,
                        job.results_uri, json.dumps(job.error_log),
                        job.job_id,
                    ),
                )

    def delete(self, job_id: str) -> bool:
        with self._conn_factory() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM factors_batch_jobs WHERE job_id = %s",
                    (job_id,),
                )
                return cur.rowcount > 0

    def next_queued(self) -> Optional[BatchJob]:
        with self._conn_factory() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE factors_batch_jobs
                    SET status = %s, started_at = NOW()
                    WHERE job_id = (
                        SELECT job_id FROM factors_batch_jobs
                        WHERE status = %s
                        ORDER BY submitted_at ASC
                        FOR UPDATE SKIP LOCKED LIMIT 1
                    )
                    RETURNING job_id, tenant_id, job_type, status, submitted_at,
                              started_at, completed_at, request_count,
                              completed_count, failed_count, results_uri,
                              request_payload_uri, error_log, webhook_url,
                              webhook_secret_ref, created_by
                    """,
                    (BatchJobStatus.RUNNING.value, BatchJobStatus.QUEUED.value),
                )
                row = cur.fetchone()
        if row is None:
            return None
        return _pg_row_to_job(row)

    def put_results(self, job_id: str, results: List[Dict[str, Any]]) -> str:
        uri = self._results_writer(job_id, results)
        with self._conn_factory() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE factors_batch_jobs SET results_uri = %s WHERE job_id = %s",
                    (uri, job_id),
                )
        return uri

    def get_results(
        self,
        job_id: str,
        *,
        cursor: int = 0,
        limit: int = 1000,
    ) -> Tuple[List[Dict[str, Any]], int, int]:
        return self._results_reader(job_id, cursor, limit)

    def get_request_payload(self, job_id: str) -> List[Dict[str, Any]]:
        return self._payload_reader(job_id)


def _raise_not_wired(*_args: Any, **_kwargs: Any) -> Any:
    raise BatchJobError(
        "PostgresBatchJobQueue writer/reader hooks are not wired. "
        "Pass payload_writer / payload_reader / results_writer / results_reader."
    )


def _pg_row_to_job(row: Tuple[Any, ...]) -> BatchJob:
    error_log_raw = row[12]
    if isinstance(error_log_raw, str):
        try:
            error_log = json.loads(error_log_raw)
        except json.JSONDecodeError:
            error_log = []
    else:
        error_log = list(error_log_raw or [])
    return BatchJob(
        job_id=row[0],
        tenant_id=row[1],
        job_type=BatchJobType(row[2]),
        status=BatchJobStatus(row[3]),
        submitted_at=str(row[4]) if row[4] else "",
        started_at=str(row[5]) if row[5] else None,
        completed_at=str(row[6]) if row[6] else None,
        request_count=int(row[7] or 0),
        completed_count=int(row[8] or 0),
        failed_count=int(row[9] or 0),
        results_uri=row[10],
        request_payload_uri=row[11],
        error_log=error_log,
        webhook_url=row[13],
        webhook_secret_ref=row[14],
        created_by=row[15] or "",
    )


# ---------------------------------------------------------------------------
# Public helpers — submit / status / results / cancel / list
# ---------------------------------------------------------------------------


def submit_batch_resolution(
    queue: BatchJobQueue,
    *,
    requests: List[Dict[str, Any]],
    tenant_id: str,
    tier: str,
    created_by: str,
    job_type: Union[str, BatchJobType] = BatchJobType.RESOLVE,
    webhook_url: Optional[str] = None,
    webhook_secret_ref: Optional[str] = None,
) -> BatchJobHandle:
    """Queue a batch-resolution job.

    Enforces the per-tier request cap (see :data:`MAX_REQUESTS_PER_TIER`)
    and returns a :class:`BatchJobHandle` the caller polls via
    :func:`get_batch_job_status`.
    """
    if not requests:
        raise BatchJobError("requests must be non-empty")
    cap = max_batch_size_for_tier(tier)
    if len(requests) > cap:
        raise BatchJobLimitError(
            "Tier %r allows max %d requests per batch (got %d)"
            % (tier, cap, len(requests))
        )
    jtype = BatchJobType(job_type) if not isinstance(job_type, BatchJobType) else job_type
    now = datetime.now(timezone.utc).isoformat()
    job = BatchJob(
        job_id=str(uuid.uuid4()),
        tenant_id=tenant_id,
        job_type=jtype,
        status=BatchJobStatus.QUEUED,
        submitted_at=now,
        request_count=len(requests),
        webhook_url=webhook_url,
        webhook_secret_ref=webhook_secret_ref,
        created_by=created_by,
    )
    queue.enqueue(job, requests)
    logger.info(
        "Batch job queued: job_id=%s tenant=%s type=%s count=%d",
        job.job_id, tenant_id, jtype.value, len(requests),
    )
    return BatchJobHandle(
        job_id=job.job_id,
        tenant_id=job.tenant_id,
        status=job.status,
        submitted_at=job.submitted_at,
        request_count=job.request_count,
    )


def get_batch_job_status(queue: BatchJobQueue, job_id: str) -> BatchJob:
    job = queue.get(job_id)
    if job is None:
        raise BatchJobNotFoundError("job %r not found" % job_id)
    return job


def get_batch_job_results(
    queue: BatchJobQueue,
    job_id: str,
    *,
    cursor: int = 0,
    limit: int = 1000,
) -> Dict[str, Any]:
    """Return a results page.

    Output shape::

        {
            "job_id": ...,
            "status": ...,
            "results": [...],
            "cursor": <next cursor>,
            "total": <total result count>,
            "has_more": bool,
        }
    """
    job = get_batch_job_status(queue, job_id)
    results, next_cursor, total = queue.get_results(
        job_id, cursor=cursor, limit=limit
    )
    return {
        "job_id": job_id,
        "status": job.status.value,
        "results": results,
        "cursor": next_cursor,
        "total": total,
        "has_more": next_cursor < total,
    }


def cancel_batch_job(queue: BatchJobQueue, job_id: str) -> BatchJob:
    """Cancel a QUEUED or RUNNING batch job."""
    job = get_batch_job_status(queue, job_id)
    if job.status not in (BatchJobStatus.QUEUED, BatchJobStatus.RUNNING):
        raise BatchJobStateError(
            "Cannot cancel job in status %s" % job.status.value
        )
    job.status = BatchJobStatus.CANCELLED
    job.completed_at = datetime.now(timezone.utc).isoformat()
    queue.update(job)
    logger.info("Batch job cancelled: job_id=%s", job_id)
    return job


def delete_batch_job(queue: BatchJobQueue, job_id: str) -> bool:
    """Delete a COMPLETED / FAILED / CANCELLED batch job (and its payload)."""
    job = get_batch_job_status(queue, job_id)
    if job.status in (BatchJobStatus.QUEUED, BatchJobStatus.RUNNING):
        raise BatchJobStateError(
            "Cannot delete running job %r — cancel first" % job_id
        )
    return queue.delete(job_id)


# ---------------------------------------------------------------------------
# Worker — process_next_job
# ---------------------------------------------------------------------------


#: Type of the "resolve one request" callable injected into the worker.
#: Accepts the raw request dict; returns a JSON-serialisable result dict.
ResolutionCallable = Callable[[Dict[str, Any]], Dict[str, Any]]


def process_next_job(
    queue: BatchJobQueue,
    *,
    resolver: ResolutionCallable,
    webhook_emit: Optional[Callable[[BatchJob], None]] = None,
) -> Optional[BatchJob]:
    """Process a single queued job and transition it to COMPLETED/FAILED.

    This is the worker-loop body.  Local dev invokes it directly; in
    production Celery wraps it as a task.  Returns the job that was
    processed, or ``None`` if the queue was empty.

    The resolver is injected so this module doesn't import the heavy
    resolution engine (which in turn pulls the whole factors runtime).

    Preserves CTO non-negotiable #1 — the resolver's return dict is
    persisted verbatim, so if the resolver emits ``gas_breakdown`` it
    stays gas-level in the results store.
    """
    job = queue.next_queued()
    if job is None:
        return None

    payload = queue.get_request_payload(job.job_id)
    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for idx, req in enumerate(payload):
        try:
            result = resolver(req)
            results.append({"index": idx, "request": req, "result": result})
            job.completed_count += 1
        except Exception as exc:  # noqa: BLE001 - worker must not crash
            errors.append(
                {
                    "index": idx,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                }
            )
            job.failed_count += 1

    job.error_log = errors
    queue.put_results(job.job_id, results)
    job.completed_at = datetime.now(timezone.utc).isoformat()
    job.status = (
        BatchJobStatus.FAILED
        if job.completed_count == 0 and job.failed_count > 0
        else BatchJobStatus.COMPLETED
    )
    queue.update(job)

    # Best-effort webhook delivery.
    if webhook_emit is not None:
        try:
            webhook_emit(job)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Webhook emit failed for job=%s: %s", job.job_id, exc
            )
    logger.info(
        "Batch job finished: job_id=%s status=%s completed=%d failed=%d",
        job.job_id, job.status.value, job.completed_count, job.failed_count,
    )
    return job


def build_webhook_payload(job: BatchJob) -> Dict[str, Any]:
    """Canonical webhook body for a batch-job completion event."""
    return {
        "event_type": "batch_job.completed",
        "job_id": job.job_id,
        "tenant_id": job.tenant_id,
        "job_type": job.job_type.value,
        "status": job.status.value,
        "completed_count": job.completed_count,
        "failed_count": job.failed_count,
        "request_count": job.request_count,
        "results_uri": job.results_uri,
        "completed_at": job.completed_at,
    }


# ---------------------------------------------------------------------------
# Default queue resolution (used by the REST routes).
# ---------------------------------------------------------------------------


_default_queue: Optional[BatchJobQueue] = None
_default_queue_lock = threading.Lock()


def get_default_queue() -> BatchJobQueue:
    """Return the process-wide default queue.

    Uses ``GL_FACTORS_BATCH_SQLITE_PATH`` (dev) unless a production
    queue has been injected via :func:`set_default_queue`.
    """
    global _default_queue
    with _default_queue_lock:
        if _default_queue is None:
            path = os.environ.get("GL_FACTORS_BATCH_SQLITE_PATH")
            _default_queue = SQLiteBatchJobQueue(path or None)
        return _default_queue


def set_default_queue(queue: BatchJobQueue) -> None:
    """Inject a production queue implementation."""
    global _default_queue
    with _default_queue_lock:
        _default_queue = queue


__all__ = [
    "BatchJob",
    "BatchJobError",
    "BatchJobHandle",
    "BatchJobLimitError",
    "BatchJobNotFoundError",
    "BatchJobQueue",
    "BatchJobStateError",
    "BatchJobStatus",
    "BatchJobType",
    "MAX_REQUESTS_PER_TIER",
    "PostgresBatchJobQueue",
    "SQLiteBatchJobQueue",
    "build_webhook_payload",
    "cancel_batch_job",
    "delete_batch_job",
    "get_batch_job_results",
    "get_batch_job_status",
    "get_default_queue",
    "max_batch_size_for_tier",
    "process_next_job",
    "set_default_queue",
    "submit_batch_resolution",
]
