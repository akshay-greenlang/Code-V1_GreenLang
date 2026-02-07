"""
Sandbox Audit - Agent Factory Sandbox (INFRA-010)

Provides audit logging for all sandbox executions.  v1 scope records
only the essential execution facts: agent_key, exit_code, duration_ms,
peak_memory_mb, stdout_size, stderr_size, and error_category.

Syscall logging is deferred to v2.

Classes:
    - SandboxAuditRecord: Lightweight v1 audit record (dataclass).
    - AuditEntry: Full immutable record for PostgreSQL persistence.
    - SandboxAudit: Audit logging service with PostgreSQL persistence.

Example:
    >>> audit = SandboxAudit(db_pool)
    >>> await audit.record(entry)
    >>> entries = await audit.query_by_agent("intake-agent", limit=50)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SandboxAuditRecord (v1 lightweight)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SandboxAuditRecord:
    """Lightweight v1 audit record for a single sandbox execution.

    Contains only the essential fields required for operational
    monitoring and compliance.  Syscall-level detail is deferred to v2.

    Attributes:
        agent_key: The agent that was executed.
        exit_code: Process exit code (0 = success).
        duration_ms: Wall-clock execution duration in milliseconds.
        peak_memory_mb: Peak memory usage in megabytes (best-effort).
        stdout_size: Size of captured stdout in bytes.
        stderr_size: Size of captured stderr in bytes.
        error_category: Human-readable error classification, empty on
            success.
        timestamp: When the execution occurred (UTC).
    """

    agent_key: str
    exit_code: int
    duration_ms: float
    peak_memory_mb: float = 0.0
    stdout_size: int = 0
    stderr_size: int = 0
    error_category: str = ""
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the record to a plain dictionary."""
        return {
            "agent_key": self.agent_key,
            "exit_code": self.exit_code,
            "duration_ms": round(self.duration_ms, 2),
            "peak_memory_mb": round(self.peak_memory_mb, 2),
            "stdout_size": self.stdout_size,
            "stderr_size": self.stderr_size,
            "error_category": self.error_category,
            "timestamp": self.timestamp.isoformat(),
        }


# ---------------------------------------------------------------------------
# Audit Entry (full persistence model)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AuditEntry:
    """Immutable record of a single sandbox execution.

    Attributes:
        entry_id: Unique identifier for this audit entry.
        agent_key: The agent that was executed.
        agent_module: Fully qualified Python module path.
        tenant_id: Tenant identifier for multi-tenant isolation.
        input_hash: SHA-256 hash of the serialised input data.
        output_hash: SHA-256 hash of the captured stdout.
        exit_code: Process exit code (0 = success).
        duration_ms: Wall-clock execution duration in milliseconds.
        peak_memory_mb: Peak memory usage in megabytes.
        stdout_size: Size of captured stdout in bytes.
        stderr_size: Size of captured stderr in bytes.
        error_category: Error category if exit_code != 0.
        status: Execution status (success, failure, timeout, error).
        metadata: Additional metadata for context.
        created_at: When the execution occurred (UTC).
    """

    entry_id: str = field(default_factory=lambda: str(uuid4()))
    agent_key: str = ""
    agent_module: str = ""
    tenant_id: str = ""
    input_hash: str = ""
    output_hash: str = ""
    exit_code: int = 0
    duration_ms: float = 0.0
    peak_memory_mb: float = 0.0
    stdout_size: int = 0
    stderr_size: int = 0
    error_category: str = ""
    status: str = "success"
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# ---------------------------------------------------------------------------
# SQL Statements
# ---------------------------------------------------------------------------

_CREATE_TABLE_SQL = """\
CREATE TABLE IF NOT EXISTS sandbox_audit (
    entry_id        TEXT PRIMARY KEY,
    agent_key       TEXT NOT NULL,
    agent_module    TEXT NOT NULL DEFAULT '',
    tenant_id       TEXT NOT NULL DEFAULT '',
    input_hash      TEXT NOT NULL DEFAULT '',
    output_hash     TEXT NOT NULL DEFAULT '',
    exit_code       INTEGER NOT NULL DEFAULT 0,
    duration_ms     DOUBLE PRECISION NOT NULL DEFAULT 0,
    peak_memory_mb  DOUBLE PRECISION NOT NULL DEFAULT 0,
    stdout_size     INTEGER NOT NULL DEFAULT 0,
    stderr_size     INTEGER NOT NULL DEFAULT 0,
    error_category  TEXT NOT NULL DEFAULT '',
    status          TEXT NOT NULL DEFAULT 'success',
    metadata        JSONB NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sandbox_audit_agent
    ON sandbox_audit (agent_key, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_sandbox_audit_tenant
    ON sandbox_audit (tenant_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_sandbox_audit_status
    ON sandbox_audit (status, created_at DESC);
"""

_INSERT_SQL = """\
INSERT INTO sandbox_audit (
    entry_id, agent_key, agent_module, tenant_id,
    input_hash, output_hash, exit_code, duration_ms,
    peak_memory_mb, stdout_size, stderr_size, error_category,
    status, metadata, created_at
) VALUES (
    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14::jsonb, $15
)
"""

_SELECT_BY_AGENT_SQL = """\
SELECT entry_id, agent_key, agent_module, tenant_id,
       input_hash, output_hash, exit_code, duration_ms,
       peak_memory_mb, stdout_size, stderr_size, error_category,
       status, metadata, created_at
FROM sandbox_audit
WHERE agent_key = $1
ORDER BY created_at DESC
LIMIT $2 OFFSET $3
"""

_SELECT_BY_TENANT_SQL = """\
SELECT entry_id, agent_key, agent_module, tenant_id,
       input_hash, output_hash, exit_code, duration_ms,
       peak_memory_mb, stdout_size, stderr_size, error_category,
       status, metadata, created_at
FROM sandbox_audit
WHERE tenant_id = $1
ORDER BY created_at DESC
LIMIT $2 OFFSET $3
"""

_SELECT_BY_STATUS_SQL = """\
SELECT entry_id, agent_key, agent_module, tenant_id,
       input_hash, output_hash, exit_code, duration_ms,
       peak_memory_mb, stdout_size, stderr_size, error_category,
       status, metadata, created_at
FROM sandbox_audit
WHERE status = $1 AND created_at >= $2 AND created_at <= $3
ORDER BY created_at DESC
LIMIT $4 OFFSET $5
"""

_SELECT_BY_TIME_RANGE_SQL = """\
SELECT entry_id, agent_key, agent_module, tenant_id,
       input_hash, output_hash, exit_code, duration_ms,
       peak_memory_mb, stdout_size, stderr_size, error_category,
       status, metadata, created_at
FROM sandbox_audit
WHERE created_at >= $1 AND created_at <= $2
ORDER BY created_at DESC
LIMIT $3 OFFSET $4
"""


# ---------------------------------------------------------------------------
# Sandbox Audit Service
# ---------------------------------------------------------------------------


class SandboxAudit:
    """Audit logging service for sandbox executions.

    Writes immutable AuditEntry records to PostgreSQL. Provides query
    methods for retrieving entries by agent, tenant, status, and time range.

    Attributes:
        _pool: Async PostgreSQL connection pool (psycopg or asyncpg).
        _initialised: Whether the audit table has been created.
    """

    def __init__(self, db_pool: Any) -> None:
        """Initialize the audit service.

        Args:
            db_pool: An async database connection pool. Expected to support
                ``async with pool.connection() as conn`` (psycopg) or
                ``async with pool.acquire() as conn`` (asyncpg).
        """
        self._pool = db_pool
        self._initialised = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Create the audit table if it does not exist.

        Idempotent - safe to call multiple times.
        """
        if self._initialised:
            return
        try:
            conn = await self._acquire_connection()
            try:
                await conn.execute(_CREATE_TABLE_SQL)
            finally:
                await self._release_connection(conn)
            self._initialised = True
            logger.info("SandboxAudit: audit table initialised")
        except Exception as exc:
            logger.error("SandboxAudit: failed to initialise table: %s", exc)
            raise

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    async def record(self, entry: AuditEntry) -> None:
        """Record a single audit entry to the database.

        Args:
            entry: The audit entry to persist.
        """
        import json

        try:
            conn = await self._acquire_connection()
            try:
                await conn.execute(
                    _INSERT_SQL,
                    entry.entry_id,
                    entry.agent_key,
                    entry.agent_module,
                    entry.tenant_id,
                    entry.input_hash,
                    entry.output_hash,
                    entry.exit_code,
                    entry.duration_ms,
                    entry.peak_memory_mb,
                    entry.stdout_size,
                    entry.stderr_size,
                    entry.error_category,
                    entry.status,
                    json.dumps(entry.metadata, default=str),
                    entry.created_at,
                )
            finally:
                await self._release_connection(conn)

            logger.debug(
                "SandboxAudit: recorded entry %s (agent=%s, status=%s)",
                entry.entry_id, entry.agent_key, entry.status,
            )
        except Exception as exc:
            logger.error(
                "SandboxAudit: failed to record entry %s: %s",
                entry.entry_id, exc,
            )
            raise

    async def record_batch(self, entries: List[AuditEntry]) -> int:
        """Record multiple audit entries in a single transaction.

        Args:
            entries: List of audit entries to persist.

        Returns:
            Number of entries successfully written.
        """
        import json

        if not entries:
            return 0

        written = 0
        try:
            conn = await self._acquire_connection()
            try:
                for entry in entries:
                    await conn.execute(
                        _INSERT_SQL,
                        entry.entry_id,
                        entry.agent_key,
                        entry.agent_module,
                        entry.tenant_id,
                        entry.input_hash,
                        entry.output_hash,
                        entry.exit_code,
                        entry.duration_ms,
                        entry.peak_memory_mb,
                        entry.stdout_size,
                        entry.stderr_size,
                        entry.error_category,
                        entry.status,
                        json.dumps(entry.metadata, default=str),
                        entry.created_at,
                    )
                    written += 1
            finally:
                await self._release_connection(conn)

            logger.info("SandboxAudit: recorded %d entries in batch", written)
        except Exception as exc:
            logger.error(
                "SandboxAudit: batch write failed after %d/%d entries: %s",
                written, len(entries), exc,
            )
            raise

        return written

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    async def query_by_agent(
        self,
        agent_key: str,
        limit: int = 50,
        offset: int = 0,
    ) -> List[AuditEntry]:
        """Query audit entries by agent key.

        Args:
            agent_key: The agent to filter by.
            limit: Maximum entries to return.
            offset: Pagination offset.

        Returns:
            List of AuditEntry objects, newest first.
        """
        return await self._query(
            _SELECT_BY_AGENT_SQL, agent_key, limit, offset,
        )

    async def query_by_tenant(
        self,
        tenant_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> List[AuditEntry]:
        """Query audit entries by tenant ID.

        Args:
            tenant_id: The tenant to filter by.
            limit: Maximum entries to return.
            offset: Pagination offset.

        Returns:
            List of AuditEntry objects, newest first.
        """
        return await self._query(
            _SELECT_BY_TENANT_SQL, tenant_id, limit, offset,
        )

    async def query_by_time_range(
        self,
        start: datetime,
        end: datetime,
        limit: int = 100,
        offset: int = 0,
    ) -> List[AuditEntry]:
        """Query audit entries by time range.

        Args:
            start: Start of the time range (inclusive).
            end: End of the time range (inclusive).
            limit: Maximum entries to return.
            offset: Pagination offset.

        Returns:
            List of AuditEntry objects, newest first.
        """
        return await self._query(
            _SELECT_BY_TIME_RANGE_SQL, start, end, limit, offset,
        )

    async def query_by_status(
        self,
        status: str,
        start: datetime,
        end: datetime,
        limit: int = 100,
        offset: int = 0,
    ) -> List[AuditEntry]:
        """Query audit entries by status within a time range.

        Args:
            status: Execution status to filter by.
            start: Start of the time range (inclusive).
            end: End of the time range (inclusive).
            limit: Maximum entries to return.
            offset: Pagination offset.

        Returns:
            List of AuditEntry objects, newest first.
        """
        return await self._query(
            _SELECT_BY_STATUS_SQL, status, start, end, limit, offset,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _query(self, sql: str, *params: Any) -> List[AuditEntry]:
        """Execute a query and return AuditEntry objects."""
        import json

        entries: List[AuditEntry] = []
        try:
            conn = await self._acquire_connection()
            try:
                rows = await conn.fetch(sql, *params)
                for row in rows:
                    metadata = row["metadata"]
                    if isinstance(metadata, str):
                        metadata = json.loads(metadata)
                    entries.append(
                        AuditEntry(
                            entry_id=row["entry_id"],
                            agent_key=row["agent_key"],
                            agent_module=row["agent_module"],
                            tenant_id=row["tenant_id"],
                            input_hash=row["input_hash"],
                            output_hash=row["output_hash"],
                            exit_code=row["exit_code"],
                            duration_ms=row["duration_ms"],
                            peak_memory_mb=row["peak_memory_mb"],
                            stdout_size=row["stdout_size"],
                            stderr_size=row["stderr_size"],
                            error_category=row["error_category"],
                            status=row["status"],
                            metadata=metadata,
                            created_at=row["created_at"],
                        )
                    )
            finally:
                await self._release_connection(conn)
        except Exception as exc:
            logger.error("SandboxAudit: query failed: %s", exc)
            raise
        return entries

    async def _acquire_connection(self) -> Any:
        """Acquire a connection from the pool.

        Supports both psycopg (connection()) and asyncpg (acquire()) patterns.
        """
        if hasattr(self._pool, "connection"):
            ctx = self._pool.connection()
            conn = await ctx.__aenter__()
            conn._ctx = ctx  # type: ignore[attr-defined]
            return conn
        elif hasattr(self._pool, "acquire"):
            return await self._pool.acquire()
        else:
            raise TypeError(
                f"Unsupported pool type: {type(self._pool).__name__}"
            )

    async def _release_connection(self, conn: Any) -> None:
        """Release a connection back to the pool."""
        if hasattr(conn, "_ctx"):
            ctx = conn._ctx
            await ctx.__aexit__(None, None, None)
        elif hasattr(self._pool, "release"):
            await self._pool.release(conn)


__all__ = [
    "AuditEntry",
    "SandboxAudit",
    "SandboxAuditRecord",
]
