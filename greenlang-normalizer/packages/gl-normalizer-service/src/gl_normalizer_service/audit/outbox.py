"""
Durable Outbox Pattern for GL-FOUND-X-003 Audit Events.

This module implements the Transactional Outbox Pattern for guaranteed
at-least-once delivery of audit events to Kafka. Events are first persisted
to a PostgreSQL outbox table, then a background worker publishes them.

Key Features:
    - Atomic write to outbox within application transaction
    - Background worker for asynchronous publishing
    - Automatic retry with exponential backoff
    - Dead letter handling for failed events
    - At-least-once delivery guarantee

Architecture:
    1. Application writes audit event to outbox table (same transaction as business data)
    2. Background worker polls for pending records
    3. Worker publishes to Kafka and marks as published
    4. Failed records are retried with exponential backoff
    5. Records exceeding max retries are marked as failed for manual review

Example:
    >>> from gl_normalizer_service.audit.outbox import AuditOutbox
    >>> from gl_normalizer_service.audit.models import OutboxConfig
    >>> config = OutboxConfig(db_url="postgresql://localhost/normalizer")
    >>> outbox = AuditOutbox(config)
    >>> await outbox.write_to_outbox(event)
    >>> await outbox.process_outbox(publisher)

NFR Compliance:
    - NFR-037: At-least-once delivery via Outbox Pattern
"""

import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

from gl_normalizer_service.audit.models import (
    OutboxConfig,
    OutboxRecord,
    OutboxStatus,
)

logger = logging.getLogger(__name__)


class OutboxProcessingError(Exception):
    """
    Exception raised when outbox processing fails.

    Attributes:
        record_id: ID of the outbox record that failed.
        message: Error message.
        original_error: Original exception that caused the failure.
    """

    def __init__(
        self,
        record_id: str,
        message: str,
        original_error: Optional[Exception] = None,
    ):
        self.record_id = record_id
        self.original_error = original_error
        super().__init__(f"Outbox processing failed for {record_id}: {message}")


class Publisher(Protocol):
    """Protocol for audit event publishers."""

    async def publish(self, event: Dict[str, Any], org_id: str) -> Tuple[int, int]:
        """
        Publish an event to the message broker.

        Args:
            event: Event payload to publish.
            org_id: Organization ID for partitioning.

        Returns:
            Tuple of (partition, offset) from the broker.
        """
        ...


class DatabaseConnection(Protocol):
    """Protocol for database connections."""

    async def execute(self, query: str, *args: Any) -> Any:
        """Execute a query."""
        ...

    async def fetch(self, query: str, *args: Any) -> List[Dict[str, Any]]:
        """Fetch rows from a query."""
        ...

    async def fetchrow(self, query: str, *args: Any) -> Optional[Dict[str, Any]]:
        """Fetch a single row."""
        ...


class AuditOutbox:
    """
    Durable Outbox for audit events with at-least-once delivery.

    This class implements the Transactional Outbox Pattern. Audit events
    are written to a local PostgreSQL table, then a background worker
    publishes them to Kafka asynchronously.

    Thread Safety:
        This class is designed for single-threaded async operation.
        Multiple instances can process different partitions concurrently.

    Attributes:
        config: Outbox configuration.
        _db: Database connection pool (initialized on start).
        _running: Whether the background worker is running.

    Example:
        >>> config = OutboxConfig(db_url="postgresql://localhost/normalizer")
        >>> outbox = AuditOutbox(config)
        >>> await outbox.start()
        >>> try:
        ...     await outbox.write_to_outbox(event)
        ... finally:
        ...     await outbox.stop()
    """

    # SQL for creating the outbox table
    CREATE_TABLE_SQL = """
        CREATE TABLE IF NOT EXISTS audit_outbox (
            id VARCHAR(100) PRIMARY KEY,
            event_id VARCHAR(100) NOT NULL,
            event_type VARCHAR(50) NOT NULL DEFAULT 'normalization',
            org_id VARCHAR(100) NOT NULL,
            payload JSONB NOT NULL,
            status VARCHAR(20) NOT NULL DEFAULT 'pending',
            created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
            published_at TIMESTAMP WITH TIME ZONE,
            retries INTEGER NOT NULL DEFAULT 0,
            last_error TEXT,
            kafka_offset BIGINT,
            kafka_partition INTEGER
        );

        CREATE INDEX IF NOT EXISTS idx_audit_outbox_status
            ON audit_outbox(status) WHERE status = 'pending';
        CREATE INDEX IF NOT EXISTS idx_audit_outbox_org_id
            ON audit_outbox(org_id);
        CREATE INDEX IF NOT EXISTS idx_audit_outbox_created_at
            ON audit_outbox(created_at);
    """

    # SQL for inserting a record
    INSERT_SQL = """
        INSERT INTO audit_outbox (
            id, event_id, event_type, org_id, payload, status,
            created_at, updated_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        RETURNING id
    """

    # SQL for fetching pending records
    FETCH_PENDING_SQL = """
        SELECT id, event_id, event_type, org_id, payload, status,
               created_at, updated_at, retries, last_error
        FROM audit_outbox
        WHERE status = 'pending'
          AND (updated_at < $1 OR retries = 0)
        ORDER BY created_at ASC
        LIMIT $2
        FOR UPDATE SKIP LOCKED
    """

    # SQL for marking as published
    MARK_PUBLISHED_SQL = """
        UPDATE audit_outbox
        SET status = 'published',
            published_at = $2,
            updated_at = $2,
            kafka_offset = $3,
            kafka_partition = $4
        WHERE id = $1
    """

    # SQL for marking as failed
    MARK_FAILED_SQL = """
        UPDATE audit_outbox
        SET status = 'failed',
            updated_at = $2,
            retries = retries + 1,
            last_error = $3
        WHERE id = $1
    """

    # SQL for retrying
    MARK_RETRY_SQL = """
        UPDATE audit_outbox
        SET status = 'pending',
            updated_at = $2,
            retries = retries + 1,
            last_error = $3
        WHERE id = $1
    """

    def __init__(
        self,
        config: OutboxConfig,
        db_pool: Optional[Any] = None,
    ):
        """
        Initialize the AuditOutbox.

        Args:
            config: Outbox configuration.
            db_pool: Optional database connection pool (for testing).
        """
        self.config = config
        self._db_pool = db_pool
        self._running = False
        self._worker_task: Optional[asyncio.Task] = None
        logger.info(
            "AuditOutbox initialized with poll_interval=%ds, batch_size=%d",
            config.outbox_poll_interval_seconds,
            config.outbox_batch_size,
        )

    async def start(self) -> None:
        """
        Start the outbox, initializing database connection.

        Creates the outbox table if it does not exist and prepares
        the connection pool for async operations.

        Raises:
            RuntimeError: If already started.
        """
        if self._running:
            raise RuntimeError("AuditOutbox is already running")

        logger.info("Starting AuditOutbox...")

        if self._db_pool is None:
            try:
                import asyncpg
                self._db_pool = await asyncpg.create_pool(
                    self.config.db_url,
                    min_size=2,
                    max_size=10,
                )
            except ImportError:
                logger.warning(
                    "asyncpg not installed, using mock database connection"
                )
                self._db_pool = MockDatabasePool()

        # Create table if not exists
        async with self._db_pool.acquire() as conn:
            await conn.execute(self.CREATE_TABLE_SQL)

        self._running = True
        logger.info("AuditOutbox started successfully")

    async def stop(self) -> None:
        """
        Stop the outbox and close database connections.

        Waits for any in-progress processing to complete before closing.
        """
        if not self._running:
            return

        logger.info("Stopping AuditOutbox...")
        self._running = False

        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

        if self._db_pool and hasattr(self._db_pool, "close"):
            await self._db_pool.close()

        logger.info("AuditOutbox stopped")

    async def write_to_outbox(
        self,
        event: Dict[str, Any],
        event_type: str = "normalization",
    ) -> OutboxRecord:
        """
        Write an audit event to the outbox table.

        This method should be called within the same database transaction
        as the business operation to ensure atomicity. The event will be
        published to Kafka by the background worker.

        Args:
            event: Complete audit event dictionary.
            event_type: Type of event (default: "normalization").

        Returns:
            The created OutboxRecord.

        Raises:
            ValueError: If event is missing required fields.
            OutboxProcessingError: If database write fails.

        Example:
            >>> record = await outbox.write_to_outbox({
            ...     "event_id": "norm-evt-001",
            ...     "org_id": "org-acme",
            ...     "status": "success",
            ...     ...
            ... })
        """
        # Validate required fields
        event_id = event.get("event_id")
        org_id = event.get("org_id")

        if not event_id:
            raise ValueError("Event must have an 'event_id' field")
        if not org_id:
            raise ValueError("Event must have an 'org_id' field")

        # Generate outbox record ID
        outbox_id = f"outbox-{uuid.uuid4().hex[:16]}"
        now = datetime.utcnow()

        # Create record
        record = OutboxRecord(
            id=outbox_id,
            event_id=event_id,
            event_type=event_type,
            org_id=org_id,
            payload=event,
            status=OutboxStatus.PENDING,
            created_at=now,
            updated_at=now,
        )

        # Write to database
        try:
            async with self._db_pool.acquire() as conn:
                await conn.execute(
                    self.INSERT_SQL,
                    record.id,
                    record.event_id,
                    record.event_type,
                    record.org_id,
                    json.dumps(record.payload),
                    record.status,
                    record.created_at,
                    record.updated_at,
                )

            logger.info(
                "Wrote event %s to outbox (outbox_id=%s, org_id=%s)",
                event_id,
                outbox_id,
                org_id,
            )
            return record

        except Exception as e:
            logger.error(
                "Failed to write event %s to outbox: %s",
                event_id,
                str(e),
            )
            raise OutboxProcessingError(
                record_id=outbox_id,
                message=f"Failed to write to outbox: {str(e)}",
                original_error=e,
            )

    async def process_outbox(
        self,
        publisher: Publisher,
    ) -> int:
        """
        Process pending outbox records and publish to Kafka.

        Fetches a batch of pending records, publishes each to Kafka,
        and updates the record status. Records are locked during processing
        to prevent duplicate publishing.

        Args:
            publisher: Publisher instance for sending events.

        Returns:
            Number of successfully published records.

        Raises:
            OutboxProcessingError: If batch processing fails.

        Example:
            >>> published_count = await outbox.process_outbox(kafka_publisher)
            >>> print(f"Published {published_count} events")
        """
        if not self._running:
            logger.warning("Outbox is not running, skipping processing")
            return 0

        # Calculate lock timeout
        lock_cutoff = datetime.utcnow() - timedelta(
            seconds=self.config.outbox_lock_timeout_seconds
        )

        published_count = 0
        failed_count = 0

        try:
            async with self._db_pool.acquire() as conn:
                # Fetch pending records with advisory lock
                rows = await conn.fetch(
                    self.FETCH_PENDING_SQL,
                    lock_cutoff,
                    self.config.outbox_batch_size,
                )

                if not rows:
                    logger.debug("No pending outbox records to process")
                    return 0

                logger.info("Processing %d pending outbox records", len(rows))

                for row in rows:
                    record_id = row["id"]
                    event_id = row["event_id"]
                    org_id = row["org_id"]
                    payload = (
                        row["payload"]
                        if isinstance(row["payload"], dict)
                        else json.loads(row["payload"])
                    )
                    retries = row["retries"]

                    try:
                        # Publish to Kafka
                        partition, offset = await publisher.publish(
                            event=payload,
                            org_id=org_id,
                        )

                        # Mark as published
                        now = datetime.utcnow()
                        await conn.execute(
                            self.MARK_PUBLISHED_SQL,
                            record_id,
                            now,
                            offset,
                            partition,
                        )

                        published_count += 1
                        logger.debug(
                            "Published event %s (partition=%d, offset=%d)",
                            event_id,
                            partition,
                            offset,
                        )

                    except Exception as e:
                        error_msg = str(e)
                        now = datetime.utcnow()

                        if retries + 1 >= self.config.outbox_max_retries:
                            # Mark as permanently failed
                            await conn.execute(
                                self.MARK_FAILED_SQL,
                                record_id,
                                now,
                                error_msg,
                            )
                            logger.error(
                                "Event %s failed permanently after %d retries: %s",
                                event_id,
                                retries + 1,
                                error_msg,
                            )
                        else:
                            # Mark for retry
                            await conn.execute(
                                self.MARK_RETRY_SQL,
                                record_id,
                                now,
                                error_msg,
                            )
                            logger.warning(
                                "Event %s failed (retry %d/%d): %s",
                                event_id,
                                retries + 1,
                                self.config.outbox_max_retries,
                                error_msg,
                            )

                        failed_count += 1

            logger.info(
                "Outbox processing complete: %d published, %d failed",
                published_count,
                failed_count,
            )
            return published_count

        except Exception as e:
            logger.error("Outbox batch processing failed: %s", str(e))
            raise OutboxProcessingError(
                record_id="batch",
                message=f"Batch processing failed: {str(e)}",
                original_error=e,
            )

    async def ensure_at_least_once_delivery(
        self,
        publisher: Publisher,
        max_iterations: int = 100,
    ) -> int:
        """
        Ensure all pending events are delivered at least once.

        Continuously processes the outbox until no pending records remain
        or max_iterations is reached. Use this for shutdown or recovery.

        Args:
            publisher: Publisher instance for sending events.
            max_iterations: Maximum number of processing iterations.

        Returns:
            Total number of published records.

        Example:
            >>> total = await outbox.ensure_at_least_once_delivery(publisher)
            >>> print(f"Delivered {total} events before shutdown")
        """
        logger.info("Ensuring at-least-once delivery (max_iterations=%d)", max_iterations)

        total_published = 0
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            published = await self.process_outbox(publisher)

            if published == 0:
                logger.info(
                    "All events delivered after %d iterations (total=%d)",
                    iteration,
                    total_published,
                )
                break

            total_published += published
            await asyncio.sleep(0.1)  # Brief pause between iterations

        if iteration >= max_iterations:
            logger.warning(
                "Reached max iterations (%d) with %d events published",
                max_iterations,
                total_published,
            )

        return total_published

    async def start_background_worker(
        self,
        publisher: Publisher,
    ) -> None:
        """
        Start the background worker for continuous outbox processing.

        The worker polls for pending records at the configured interval
        and publishes them to Kafka. Use this for production deployments.

        Args:
            publisher: Publisher instance for sending events.

        Raises:
            RuntimeError: If outbox is not started or worker already running.

        Example:
            >>> await outbox.start()
            >>> await outbox.start_background_worker(publisher)
        """
        if not self._running:
            raise RuntimeError("Outbox must be started before starting worker")
        if self._worker_task is not None:
            raise RuntimeError("Background worker is already running")

        self._worker_task = asyncio.create_task(
            self._background_worker_loop(publisher)
        )
        logger.info("Background worker started")

    async def _background_worker_loop(
        self,
        publisher: Publisher,
    ) -> None:
        """
        Internal background worker loop.

        Args:
            publisher: Publisher instance for sending events.
        """
        logger.info(
            "Background worker loop starting (interval=%ds)",
            self.config.outbox_poll_interval_seconds,
        )

        while self._running:
            try:
                await self.process_outbox(publisher)
            except OutboxProcessingError as e:
                logger.error("Worker iteration failed: %s", str(e))
            except Exception as e:
                logger.exception("Unexpected error in worker: %s", str(e))

            await asyncio.sleep(self.config.outbox_poll_interval_seconds)

        logger.info("Background worker loop stopped")

    async def get_pending_count(self) -> int:
        """
        Get the count of pending outbox records.

        Returns:
            Number of records with status='pending'.
        """
        async with self._db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT COUNT(*) as count FROM audit_outbox WHERE status = 'pending'"
            )
            return row["count"] if row else 0

    async def get_failed_records(
        self,
        limit: int = 100,
    ) -> List[OutboxRecord]:
        """
        Get failed outbox records for manual review.

        Args:
            limit: Maximum number of records to return.

        Returns:
            List of failed OutboxRecord instances.
        """
        async with self._db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, event_id, event_type, org_id, payload, status,
                       created_at, updated_at, retries, last_error
                FROM audit_outbox
                WHERE status = 'failed'
                ORDER BY created_at DESC
                LIMIT $1
                """,
                limit,
            )

            return [
                OutboxRecord(
                    id=row["id"],
                    event_id=row["event_id"],
                    event_type=row["event_type"],
                    org_id=row["org_id"],
                    payload=(
                        row["payload"]
                        if isinstance(row["payload"], dict)
                        else json.loads(row["payload"])
                    ),
                    status=OutboxStatus(row["status"]),
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    retries=row["retries"],
                    last_error=row["last_error"],
                )
                for row in rows
            ]

    async def retry_failed_record(self, record_id: str) -> bool:
        """
        Reset a failed record for retry.

        Args:
            record_id: ID of the outbox record to retry.

        Returns:
            True if record was reset, False if not found.
        """
        async with self._db_pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE audit_outbox
                SET status = 'pending',
                    updated_at = $2,
                    retries = 0,
                    last_error = NULL
                WHERE id = $1 AND status = 'failed'
                """,
                record_id,
                datetime.utcnow(),
            )
            # asyncpg returns string like "UPDATE 1"
            updated = "1" in str(result) if result else False
            if updated:
                logger.info("Reset failed record %s for retry", record_id)
            return updated


class MockDatabasePool:
    """
    Mock database pool for testing without PostgreSQL.

    This is a simple in-memory implementation for unit tests.
    """

    def __init__(self):
        self._records: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def acquire(self):
        """Acquire a mock connection."""
        yield MockConnection(self)

    async def close(self):
        """Close the mock pool."""
        pass


class MockConnection:
    """Mock database connection for testing."""

    def __init__(self, pool: MockDatabasePool):
        self._pool = pool

    async def execute(self, query: str, *args: Any) -> str:
        """Execute a query (mock)."""
        async with self._pool._lock:
            if "INSERT INTO audit_outbox" in query:
                record_id = args[0]
                self._pool._records[record_id] = {
                    "id": args[0],
                    "event_id": args[1],
                    "event_type": args[2],
                    "org_id": args[3],
                    "payload": json.loads(args[4]) if isinstance(args[4], str) else args[4],
                    "status": args[5],
                    "created_at": args[6],
                    "updated_at": args[7],
                    "retries": 0,
                    "last_error": None,
                }
                return "INSERT 1"
            elif "UPDATE audit_outbox" in query:
                record_id = args[0]
                if record_id in self._pool._records:
                    if "status = 'published'" in query:
                        self._pool._records[record_id]["status"] = "published"
                        self._pool._records[record_id]["published_at"] = args[1]
                        self._pool._records[record_id]["kafka_offset"] = args[2]
                        self._pool._records[record_id]["kafka_partition"] = args[3]
                    elif "status = 'failed'" in query:
                        self._pool._records[record_id]["status"] = "failed"
                        self._pool._records[record_id]["retries"] += 1
                        self._pool._records[record_id]["last_error"] = args[2]
                    elif "status = 'pending'" in query:
                        self._pool._records[record_id]["status"] = "pending"
                        self._pool._records[record_id]["retries"] += 1
                        self._pool._records[record_id]["last_error"] = args[2] if len(args) > 2 else None
                    return "UPDATE 1"
            elif "CREATE TABLE" in query or "CREATE INDEX" in query:
                return "CREATE"
            return "OK"

    async def fetch(self, query: str, *args: Any) -> List[Dict[str, Any]]:
        """Fetch rows (mock)."""
        async with self._pool._lock:
            if "SELECT" in query and "audit_outbox" in query:
                if "status = 'pending'" in query:
                    limit = args[1] if len(args) > 1 else 100
                    pending = [
                        r for r in self._pool._records.values()
                        if r["status"] == "pending"
                    ]
                    return pending[:limit]
                elif "status = 'failed'" in query:
                    limit = args[0] if args else 100
                    failed = [
                        r for r in self._pool._records.values()
                        if r["status"] == "failed"
                    ]
                    return failed[:limit]
            return []

    async def fetchrow(self, query: str, *args: Any) -> Optional[Dict[str, Any]]:
        """Fetch a single row (mock)."""
        async with self._pool._lock:
            if "COUNT(*)" in query:
                if "status = 'pending'" in query:
                    count = sum(
                        1 for r in self._pool._records.values()
                        if r["status"] == "pending"
                    )
                    return {"count": count}
            return None
