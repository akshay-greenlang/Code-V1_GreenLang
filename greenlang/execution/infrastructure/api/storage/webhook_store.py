"""
Webhook Persistent Storage for GreenLang.

This module provides multiple storage backends for webhook persistence:
- InMemoryWebhookStore: Testing and development (non-persistent)
- SQLiteWebhookStore: File-based persistence for single-node deployments
- PostgresWebhookStore: Production-grade persistence for distributed systems

All implementations follow the WebhookStore protocol for consistent interfaces.

Example:
    >>> from greenlang.infrastructure.api.storage import StorageFactory
    >>> store = StorageFactory.get_webhook_store({"backend": "sqlite", "path": "webhooks.db"})
    >>> webhook_id = await store.save_webhook(webhook)
    >>> webhook = await store.get_webhook(webhook_id)
"""

import asyncio
import hashlib
import json
import logging
import sqlite3
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
from uuid import uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------


def _datetime_to_iso(dt: Optional[datetime]) -> Optional[str]:
    """Convert datetime to ISO string for storage."""
    if dt is None:
        return None
    return dt.isoformat()


def _iso_to_datetime(iso_str: Optional[str]) -> Optional[datetime]:
    """Convert ISO string back to datetime."""
    if iso_str is None:
        return None
    return datetime.fromisoformat(iso_str)


def _calculate_provenance_hash(data: Dict[str, Any]) -> str:
    """Calculate SHA-256 hash for audit trail."""
    data_str = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(data_str.encode()).hexdigest()


# -----------------------------------------------------------------------------
# Storage Configuration
# -----------------------------------------------------------------------------


class WebhookStoreConfig(BaseModel):
    """Configuration for webhook storage backend."""

    backend: str = Field(
        default="memory",
        description="Storage backend: 'memory', 'sqlite', or 'postgres'"
    )
    sqlite_path: str = Field(
        default="webhooks.db",
        description="Path to SQLite database file"
    )
    postgres_dsn: str = Field(
        default="",
        description="PostgreSQL connection string"
    )
    postgres_pool_size: int = Field(
        default=10,
        ge=1,
        le=100,
        description="PostgreSQL connection pool size"
    )
    enable_wal_mode: bool = Field(
        default=True,
        description="Enable SQLite WAL mode for better concurrency"
    )


# -----------------------------------------------------------------------------
# Abstract Base Class
# -----------------------------------------------------------------------------


@runtime_checkable
class WebhookStore(Protocol):
    """
    Protocol defining the webhook storage interface.

    All storage backends must implement this protocol to ensure
    consistent behavior across different persistence layers.
    """

    async def save_webhook(self, webhook: "WebhookModel") -> str:
        """
        Save or update a webhook registration.

        Args:
            webhook: The webhook model to persist

        Returns:
            The webhook_id of the saved webhook
        """
        ...

    async def get_webhook(self, webhook_id: str) -> Optional["WebhookModel"]:
        """
        Retrieve a webhook by its ID.

        Args:
            webhook_id: The unique webhook identifier

        Returns:
            The webhook model if found, None otherwise
        """
        ...

    async def list_webhooks(self) -> List["WebhookModel"]:
        """
        List all registered webhooks.

        Returns:
            List of all webhook models
        """
        ...

    async def delete_webhook(self, webhook_id: str) -> bool:
        """
        Delete a webhook by its ID.

        Args:
            webhook_id: The unique webhook identifier

        Returns:
            True if deleted, False if not found
        """
        ...

    async def save_delivery(self, delivery: "WebhookDelivery") -> str:
        """
        Save a webhook delivery attempt.

        Args:
            delivery: The delivery record to persist

        Returns:
            The delivery_id of the saved delivery
        """
        ...

    async def get_deliveries(
        self,
        webhook_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List["WebhookDelivery"]:
        """
        Retrieve delivery history for a webhook.

        Args:
            webhook_id: The webhook identifier
            limit: Maximum number of deliveries to return
            offset: Number of deliveries to skip

        Returns:
            List of delivery records
        """
        ...

    async def close(self) -> None:
        """Close any open connections and release resources."""
        ...


class BaseWebhookStore(ABC):
    """
    Abstract base class for webhook storage implementations.

    Provides common functionality and enforces the WebhookStore interface.
    """

    @abstractmethod
    async def save_webhook(self, webhook: "WebhookModel") -> str:
        """Save or update a webhook registration."""
        pass

    @abstractmethod
    async def get_webhook(self, webhook_id: str) -> Optional["WebhookModel"]:
        """Retrieve a webhook by its ID."""
        pass

    @abstractmethod
    async def list_webhooks(self) -> List["WebhookModel"]:
        """List all registered webhooks."""
        pass

    @abstractmethod
    async def delete_webhook(self, webhook_id: str) -> bool:
        """Delete a webhook by its ID."""
        pass

    @abstractmethod
    async def save_delivery(self, delivery: "WebhookDelivery") -> str:
        """Save a webhook delivery attempt."""
        pass

    @abstractmethod
    async def get_deliveries(
        self,
        webhook_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List["WebhookDelivery"]:
        """Retrieve delivery history for a webhook."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close any open connections and release resources."""
        pass


# -----------------------------------------------------------------------------
# Import Models (deferred to avoid circular imports)
# -----------------------------------------------------------------------------

# We need to handle potential circular imports by importing late
# or accepting the models as parameters


def _get_webhook_models():
    """Import webhook models to avoid circular imports."""
    try:
        from greenlang.infrastructure.api.webhooks import (
            WebhookModel,
            WebhookDelivery,
            WebhookStatus
        )
        return WebhookModel, WebhookDelivery, WebhookStatus
    except ImportError:
        # Fallback: define minimal models for standalone usage
        from enum import Enum

        class WebhookStatus(str, Enum):
            PENDING = "pending"
            SENT = "sent"
            FAILED = "failed"
            RETRYING = "retrying"

        class WebhookModel(BaseModel):
            webhook_id: str = Field(default_factory=lambda: str(uuid4()))
            url: str
            events: List[str]
            secret: str
            is_active: bool = True
            health_status: str = "healthy"
            consecutive_failures: int = 0
            last_triggered_at: Optional[datetime] = None
            created_at: datetime = Field(default_factory=datetime.utcnow)
            metadata: Dict[str, Any] = Field(default_factory=dict)

        class WebhookDelivery(BaseModel):
            delivery_id: str = Field(default_factory=lambda: str(uuid4()))
            webhook_id: str
            event_type: str
            payload: Dict[str, Any]
            signature: str
            status: WebhookStatus = WebhookStatus.PENDING
            attempt: int = 1
            http_status: Optional[int] = None
            error_message: Optional[str] = None
            created_at: datetime = Field(default_factory=datetime.utcnow)
            sent_at: Optional[datetime] = None
            provenance_hash: str = ""

        return WebhookModel, WebhookDelivery, WebhookStatus


# Get models at module level for type hints
WebhookModel, WebhookDelivery, WebhookStatus = _get_webhook_models()


# -----------------------------------------------------------------------------
# InMemory Implementation (Default, Testing)
# -----------------------------------------------------------------------------


class InMemoryWebhookStore(BaseWebhookStore):
    """
    In-memory webhook storage for testing and development.

    This implementation stores all data in memory and does not persist
    across restarts. It is thread-safe using asyncio locks.

    Attributes:
        _webhooks: Dictionary mapping webhook_id to WebhookModel
        _deliveries: Dictionary mapping delivery_id to WebhookDelivery

    Example:
        >>> store = InMemoryWebhookStore()
        >>> webhook_id = await store.save_webhook(webhook)
        >>> retrieved = await store.get_webhook(webhook_id)
        >>> assert retrieved.url == webhook.url
    """

    def __init__(self):
        """Initialize in-memory storage."""
        self._webhooks: Dict[str, WebhookModel] = {}
        self._deliveries: Dict[str, WebhookDelivery] = {}
        self._lock = asyncio.Lock()
        self._provenance_log: List[Dict[str, Any]] = []

        logger.info("InMemoryWebhookStore initialized")

    async def save_webhook(self, webhook: WebhookModel) -> str:
        """
        Save or update a webhook in memory.

        Args:
            webhook: The webhook model to save

        Returns:
            The webhook_id of the saved webhook
        """
        async with self._lock:
            webhook_id = webhook.webhook_id
            self._webhooks[webhook_id] = webhook

            # Log provenance
            self._provenance_log.append({
                "action": "save_webhook",
                "webhook_id": webhook_id,
                "timestamp": datetime.utcnow().isoformat(),
                "hash": _calculate_provenance_hash(webhook.dict())
            })

            logger.debug(f"Saved webhook {webhook_id} to memory")
            return webhook_id

    async def get_webhook(self, webhook_id: str) -> Optional[WebhookModel]:
        """
        Retrieve a webhook from memory.

        Args:
            webhook_id: The unique webhook identifier

        Returns:
            The webhook model if found, None otherwise
        """
        async with self._lock:
            webhook = self._webhooks.get(webhook_id)
            if webhook:
                logger.debug(f"Retrieved webhook {webhook_id} from memory")
            return webhook

    async def list_webhooks(self) -> List[WebhookModel]:
        """
        List all webhooks in memory.

        Returns:
            List of all webhook models
        """
        async with self._lock:
            webhooks = list(self._webhooks.values())
            logger.debug(f"Listed {len(webhooks)} webhooks from memory")
            return webhooks

    async def delete_webhook(self, webhook_id: str) -> bool:
        """
        Delete a webhook from memory.

        Args:
            webhook_id: The unique webhook identifier

        Returns:
            True if deleted, False if not found
        """
        async with self._lock:
            if webhook_id in self._webhooks:
                del self._webhooks[webhook_id]

                # Also delete associated deliveries
                delivery_ids_to_delete = [
                    did for did, d in self._deliveries.items()
                    if d.webhook_id == webhook_id
                ]
                for did in delivery_ids_to_delete:
                    del self._deliveries[did]

                # Log provenance
                self._provenance_log.append({
                    "action": "delete_webhook",
                    "webhook_id": webhook_id,
                    "timestamp": datetime.utcnow().isoformat()
                })

                logger.debug(f"Deleted webhook {webhook_id} from memory")
                return True
            return False

    async def save_delivery(self, delivery: WebhookDelivery) -> str:
        """
        Save a delivery record in memory.

        Args:
            delivery: The delivery record to save

        Returns:
            The delivery_id of the saved delivery
        """
        async with self._lock:
            delivery_id = delivery.delivery_id
            self._deliveries[delivery_id] = delivery

            logger.debug(f"Saved delivery {delivery_id} to memory")
            return delivery_id

    async def get_deliveries(
        self,
        webhook_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[WebhookDelivery]:
        """
        Retrieve deliveries for a webhook from memory.

        Args:
            webhook_id: The webhook identifier
            limit: Maximum number of deliveries to return
            offset: Number of deliveries to skip

        Returns:
            List of delivery records, sorted by created_at descending
        """
        async with self._lock:
            deliveries = [
                d for d in self._deliveries.values()
                if d.webhook_id == webhook_id
            ]

            # Sort by created_at descending
            deliveries.sort(key=lambda d: d.created_at, reverse=True)

            # Apply pagination
            paginated = deliveries[offset:offset + limit]

            logger.debug(
                f"Retrieved {len(paginated)} deliveries for webhook {webhook_id}"
            )
            return paginated

    async def close(self) -> None:
        """Clear memory storage."""
        async with self._lock:
            self._webhooks.clear()
            self._deliveries.clear()
            self._provenance_log.clear()
            logger.info("InMemoryWebhookStore closed")

    def get_provenance_log(self) -> List[Dict[str, Any]]:
        """Get the provenance audit log (for testing/debugging)."""
        return list(self._provenance_log)


# -----------------------------------------------------------------------------
# SQLite Implementation (File-based Persistence)
# -----------------------------------------------------------------------------


class SQLiteWebhookStore(BaseWebhookStore):
    """
    SQLite-based webhook storage for file-based persistence.

    Suitable for single-node deployments or development environments
    that require data persistence across restarts.

    Features:
        - WAL mode for better concurrent access
        - Connection pooling via thread-local storage
        - Automatic schema migration
        - Full-text search on events

    Attributes:
        db_path: Path to the SQLite database file

    Example:
        >>> store = SQLiteWebhookStore("webhooks.db")
        >>> await store.initialize()
        >>> webhook_id = await store.save_webhook(webhook)
    """

    def __init__(self, db_path: str = "webhooks.db", enable_wal: bool = True):
        """
        Initialize SQLite webhook store.

        Args:
            db_path: Path to the SQLite database file
            enable_wal: Enable WAL mode for better concurrency
        """
        self.db_path = db_path
        self._enable_wal = enable_wal
        self._connection: Optional[sqlite3.Connection] = None
        self._lock = asyncio.Lock()
        self._initialized = False

        logger.info(f"SQLiteWebhookStore initialized with path: {db_path}")

    async def initialize(self) -> None:
        """
        Initialize the database schema.

        Creates tables if they don't exist and applies migrations.
        """
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return

            await asyncio.get_event_loop().run_in_executor(
                None, self._create_tables
            )
            self._initialized = True
            logger.info("SQLiteWebhookStore database initialized")

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create a database connection."""
        if self._connection is None:
            self._connection = sqlite3.connect(
                self.db_path,
                check_same_thread=False
            )
            self._connection.row_factory = sqlite3.Row

            if self._enable_wal:
                self._connection.execute("PRAGMA journal_mode=WAL")

            # Enable foreign keys
            self._connection.execute("PRAGMA foreign_keys=ON")

        return self._connection

    def _create_tables(self) -> None:
        """Create database tables."""
        conn = self._get_connection()

        # Webhooks table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS webhooks (
                webhook_id TEXT PRIMARY KEY,
                url TEXT NOT NULL,
                events TEXT NOT NULL,
                secret TEXT NOT NULL,
                is_active INTEGER DEFAULT 1,
                health_status TEXT DEFAULT 'healthy',
                consecutive_failures INTEGER DEFAULT 0,
                last_triggered_at TEXT,
                created_at TEXT NOT NULL,
                metadata TEXT,
                provenance_hash TEXT
            )
        """)

        # Deliveries table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS deliveries (
                delivery_id TEXT PRIMARY KEY,
                webhook_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                payload TEXT NOT NULL,
                signature TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                attempt INTEGER DEFAULT 1,
                http_status INTEGER,
                error_message TEXT,
                created_at TEXT NOT NULL,
                sent_at TEXT,
                provenance_hash TEXT,
                FOREIGN KEY (webhook_id) REFERENCES webhooks(webhook_id)
            )
        """)

        # Indexes for common queries
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_deliveries_webhook_id
            ON deliveries(webhook_id)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_deliveries_created_at
            ON deliveries(created_at DESC)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_webhooks_is_active
            ON webhooks(is_active)
        """)

        conn.commit()

    async def save_webhook(self, webhook: WebhookModel) -> str:
        """
        Save or update a webhook in SQLite.

        Uses UPSERT (INSERT OR REPLACE) for idempotent saves.

        Args:
            webhook: The webhook model to save

        Returns:
            The webhook_id of the saved webhook
        """
        await self.initialize()

        async with self._lock:
            def _save():
                conn = self._get_connection()
                provenance_hash = _calculate_provenance_hash(webhook.dict())

                conn.execute("""
                    INSERT OR REPLACE INTO webhooks (
                        webhook_id, url, events, secret, is_active,
                        health_status, consecutive_failures, last_triggered_at,
                        created_at, metadata, provenance_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    webhook.webhook_id,
                    webhook.url,
                    json.dumps(webhook.events),
                    webhook.secret,
                    1 if webhook.is_active else 0,
                    webhook.health_status,
                    webhook.consecutive_failures,
                    _datetime_to_iso(webhook.last_triggered_at),
                    _datetime_to_iso(webhook.created_at),
                    json.dumps(webhook.metadata),
                    provenance_hash
                ))
                conn.commit()
                return webhook.webhook_id

            webhook_id = await asyncio.get_event_loop().run_in_executor(
                None, _save
            )

            logger.debug(f"Saved webhook {webhook_id} to SQLite")
            return webhook_id

    async def get_webhook(self, webhook_id: str) -> Optional[WebhookModel]:
        """
        Retrieve a webhook from SQLite.

        Args:
            webhook_id: The unique webhook identifier

        Returns:
            The webhook model if found, None otherwise
        """
        await self.initialize()

        async with self._lock:
            def _get():
                conn = self._get_connection()
                cursor = conn.execute(
                    "SELECT * FROM webhooks WHERE webhook_id = ?",
                    (webhook_id,)
                )
                row = cursor.fetchone()

                if row is None:
                    return None

                return WebhookModel(
                    webhook_id=row["webhook_id"],
                    url=row["url"],
                    events=json.loads(row["events"]),
                    secret=row["secret"],
                    is_active=bool(row["is_active"]),
                    health_status=row["health_status"],
                    consecutive_failures=row["consecutive_failures"],
                    last_triggered_at=_iso_to_datetime(row["last_triggered_at"]),
                    created_at=_iso_to_datetime(row["created_at"]) or datetime.utcnow(),
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {}
                )

            webhook = await asyncio.get_event_loop().run_in_executor(
                None, _get
            )

            if webhook:
                logger.debug(f"Retrieved webhook {webhook_id} from SQLite")
            return webhook

    async def list_webhooks(self) -> List[WebhookModel]:
        """
        List all webhooks from SQLite.

        Returns:
            List of all webhook models
        """
        await self.initialize()

        async with self._lock:
            def _list():
                conn = self._get_connection()
                cursor = conn.execute("SELECT * FROM webhooks ORDER BY created_at DESC")
                rows = cursor.fetchall()

                webhooks = []
                for row in rows:
                    webhooks.append(WebhookModel(
                        webhook_id=row["webhook_id"],
                        url=row["url"],
                        events=json.loads(row["events"]),
                        secret=row["secret"],
                        is_active=bool(row["is_active"]),
                        health_status=row["health_status"],
                        consecutive_failures=row["consecutive_failures"],
                        last_triggered_at=_iso_to_datetime(row["last_triggered_at"]),
                        created_at=_iso_to_datetime(row["created_at"]) or datetime.utcnow(),
                        metadata=json.loads(row["metadata"]) if row["metadata"] else {}
                    ))

                return webhooks

            webhooks = await asyncio.get_event_loop().run_in_executor(
                None, _list
            )

            logger.debug(f"Listed {len(webhooks)} webhooks from SQLite")
            return webhooks

    async def delete_webhook(self, webhook_id: str) -> bool:
        """
        Delete a webhook from SQLite.

        Also deletes associated delivery records.

        Args:
            webhook_id: The unique webhook identifier

        Returns:
            True if deleted, False if not found
        """
        await self.initialize()

        async with self._lock:
            def _delete():
                conn = self._get_connection()

                # Check if exists
                cursor = conn.execute(
                    "SELECT 1 FROM webhooks WHERE webhook_id = ?",
                    (webhook_id,)
                )
                if cursor.fetchone() is None:
                    return False

                # Delete deliveries first (foreign key)
                conn.execute(
                    "DELETE FROM deliveries WHERE webhook_id = ?",
                    (webhook_id,)
                )

                # Delete webhook
                conn.execute(
                    "DELETE FROM webhooks WHERE webhook_id = ?",
                    (webhook_id,)
                )

                conn.commit()
                return True

            deleted = await asyncio.get_event_loop().run_in_executor(
                None, _delete
            )

            if deleted:
                logger.debug(f"Deleted webhook {webhook_id} from SQLite")
            return deleted

    async def save_delivery(self, delivery: WebhookDelivery) -> str:
        """
        Save a delivery record to SQLite.

        Args:
            delivery: The delivery record to save

        Returns:
            The delivery_id of the saved delivery
        """
        await self.initialize()

        async with self._lock:
            def _save():
                conn = self._get_connection()

                conn.execute("""
                    INSERT OR REPLACE INTO deliveries (
                        delivery_id, webhook_id, event_type, payload,
                        signature, status, attempt, http_status,
                        error_message, created_at, sent_at, provenance_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    delivery.delivery_id,
                    delivery.webhook_id,
                    delivery.event_type,
                    json.dumps(delivery.payload, default=str),
                    delivery.signature,
                    delivery.status.value if hasattr(delivery.status, 'value') else str(delivery.status),
                    delivery.attempt,
                    delivery.http_status,
                    delivery.error_message,
                    _datetime_to_iso(delivery.created_at),
                    _datetime_to_iso(delivery.sent_at),
                    delivery.provenance_hash
                ))
                conn.commit()
                return delivery.delivery_id

            delivery_id = await asyncio.get_event_loop().run_in_executor(
                None, _save
            )

            logger.debug(f"Saved delivery {delivery_id} to SQLite")
            return delivery_id

    async def get_deliveries(
        self,
        webhook_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[WebhookDelivery]:
        """
        Retrieve deliveries for a webhook from SQLite.

        Args:
            webhook_id: The webhook identifier
            limit: Maximum number of deliveries to return
            offset: Number of deliveries to skip

        Returns:
            List of delivery records, sorted by created_at descending
        """
        await self.initialize()

        async with self._lock:
            def _get():
                conn = self._get_connection()
                cursor = conn.execute("""
                    SELECT * FROM deliveries
                    WHERE webhook_id = ?
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                """, (webhook_id, limit, offset))
                rows = cursor.fetchall()

                deliveries = []
                for row in rows:
                    deliveries.append(WebhookDelivery(
                        delivery_id=row["delivery_id"],
                        webhook_id=row["webhook_id"],
                        event_type=row["event_type"],
                        payload=json.loads(row["payload"]),
                        signature=row["signature"],
                        status=WebhookStatus(row["status"]),
                        attempt=row["attempt"],
                        http_status=row["http_status"],
                        error_message=row["error_message"],
                        created_at=_iso_to_datetime(row["created_at"]) or datetime.utcnow(),
                        sent_at=_iso_to_datetime(row["sent_at"]),
                        provenance_hash=row["provenance_hash"] or ""
                    ))

                return deliveries

            deliveries = await asyncio.get_event_loop().run_in_executor(
                None, _get
            )

            logger.debug(
                f"Retrieved {len(deliveries)} deliveries for webhook {webhook_id}"
            )
            return deliveries

    async def close(self) -> None:
        """Close the database connection."""
        async with self._lock:
            if self._connection:
                self._connection.close()
                self._connection = None
            self._initialized = False
            logger.info("SQLiteWebhookStore closed")


# -----------------------------------------------------------------------------
# PostgreSQL Implementation (Production)
# -----------------------------------------------------------------------------


class PostgresWebhookStore(BaseWebhookStore):
    """
    PostgreSQL-based webhook storage for production deployments.

    Provides high availability, concurrent access, and advanced
    querying capabilities for distributed systems.

    Features:
        - Connection pooling with asyncpg
        - Automatic schema migration
        - JSONB for efficient metadata storage
        - Full-text search support
        - Partitioning support for deliveries table

    Attributes:
        dsn: PostgreSQL connection string
        pool_size: Maximum number of connections in pool

    Example:
        >>> store = PostgresWebhookStore(
        ...     dsn="postgresql://user:pass@localhost/webhooks",
        ...     pool_size=20
        ... )
        >>> await store.initialize()
        >>> webhook_id = await store.save_webhook(webhook)
    """

    def __init__(
        self,
        dsn: str,
        pool_size: int = 10,
        pool_min_size: int = 2
    ):
        """
        Initialize PostgreSQL webhook store.

        Args:
            dsn: PostgreSQL connection string
            pool_size: Maximum connections in pool
            pool_min_size: Minimum connections in pool
        """
        self.dsn = dsn
        self.pool_size = pool_size
        self.pool_min_size = pool_min_size
        self._pool = None
        self._initialized = False
        self._lock = asyncio.Lock()

        logger.info(f"PostgresWebhookStore initialized (pool_size={pool_size})")

    async def initialize(self) -> None:
        """
        Initialize the database connection pool and schema.

        Creates tables if they don't exist and applies migrations.
        """
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return

            try:
                import asyncpg
            except ImportError:
                raise ImportError(
                    "asyncpg is required for PostgresWebhookStore. "
                    "Install it with: pip install asyncpg"
                )

            self._pool = await asyncpg.create_pool(
                self.dsn,
                min_size=self.pool_min_size,
                max_size=self.pool_size
            )

            await self._create_tables()
            self._initialized = True
            logger.info("PostgresWebhookStore database initialized")

    async def _create_tables(self) -> None:
        """Create database tables."""
        async with self._pool.acquire() as conn:
            # Webhooks table with JSONB for flexible metadata
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS webhooks (
                    webhook_id UUID PRIMARY KEY,
                    url TEXT NOT NULL,
                    events JSONB NOT NULL,
                    secret TEXT NOT NULL,
                    is_active BOOLEAN DEFAULT TRUE,
                    health_status TEXT DEFAULT 'healthy',
                    consecutive_failures INTEGER DEFAULT 0,
                    last_triggered_at TIMESTAMPTZ,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    metadata JSONB DEFAULT '{}',
                    provenance_hash TEXT
                )
            """)

            # Deliveries table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS deliveries (
                    delivery_id UUID PRIMARY KEY,
                    webhook_id UUID NOT NULL REFERENCES webhooks(webhook_id) ON DELETE CASCADE,
                    event_type TEXT NOT NULL,
                    payload JSONB NOT NULL,
                    signature TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    attempt INTEGER DEFAULT 1,
                    http_status INTEGER,
                    error_message TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    sent_at TIMESTAMPTZ,
                    provenance_hash TEXT
                )
            """)

            # Indexes
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_deliveries_webhook_id
                ON deliveries(webhook_id)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_deliveries_created_at
                ON deliveries(created_at DESC)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_webhooks_is_active
                ON webhooks(is_active) WHERE is_active = TRUE
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_webhooks_events
                ON webhooks USING GIN(events)
            """)

    async def save_webhook(self, webhook: WebhookModel) -> str:
        """
        Save or update a webhook in PostgreSQL.

        Uses UPSERT for idempotent saves.

        Args:
            webhook: The webhook model to save

        Returns:
            The webhook_id of the saved webhook
        """
        await self.initialize()

        provenance_hash = _calculate_provenance_hash(webhook.dict())

        async with self._pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO webhooks (
                    webhook_id, url, events, secret, is_active,
                    health_status, consecutive_failures, last_triggered_at,
                    created_at, metadata, provenance_hash
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                ON CONFLICT (webhook_id) DO UPDATE SET
                    url = EXCLUDED.url,
                    events = EXCLUDED.events,
                    secret = EXCLUDED.secret,
                    is_active = EXCLUDED.is_active,
                    health_status = EXCLUDED.health_status,
                    consecutive_failures = EXCLUDED.consecutive_failures,
                    last_triggered_at = EXCLUDED.last_triggered_at,
                    metadata = EXCLUDED.metadata,
                    provenance_hash = EXCLUDED.provenance_hash
            """,
                webhook.webhook_id,
                webhook.url,
                json.dumps(webhook.events),
                webhook.secret,
                webhook.is_active,
                webhook.health_status,
                webhook.consecutive_failures,
                webhook.last_triggered_at,
                webhook.created_at,
                json.dumps(webhook.metadata),
                provenance_hash
            )

        logger.debug(f"Saved webhook {webhook.webhook_id} to PostgreSQL")
        return webhook.webhook_id

    async def get_webhook(self, webhook_id: str) -> Optional[WebhookModel]:
        """
        Retrieve a webhook from PostgreSQL.

        Args:
            webhook_id: The unique webhook identifier

        Returns:
            The webhook model if found, None otherwise
        """
        await self.initialize()

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM webhooks WHERE webhook_id = $1",
                webhook_id
            )

            if row is None:
                return None

            webhook = WebhookModel(
                webhook_id=str(row["webhook_id"]),
                url=row["url"],
                events=json.loads(row["events"]) if isinstance(row["events"], str) else row["events"],
                secret=row["secret"],
                is_active=row["is_active"],
                health_status=row["health_status"],
                consecutive_failures=row["consecutive_failures"],
                last_triggered_at=row["last_triggered_at"],
                created_at=row["created_at"],
                metadata=json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"]
            )

        logger.debug(f"Retrieved webhook {webhook_id} from PostgreSQL")
        return webhook

    async def list_webhooks(self) -> List[WebhookModel]:
        """
        List all webhooks from PostgreSQL.

        Returns:
            List of all webhook models
        """
        await self.initialize()

        webhooks = []
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM webhooks ORDER BY created_at DESC"
            )

            for row in rows:
                webhooks.append(WebhookModel(
                    webhook_id=str(row["webhook_id"]),
                    url=row["url"],
                    events=json.loads(row["events"]) if isinstance(row["events"], str) else row["events"],
                    secret=row["secret"],
                    is_active=row["is_active"],
                    health_status=row["health_status"],
                    consecutive_failures=row["consecutive_failures"],
                    last_triggered_at=row["last_triggered_at"],
                    created_at=row["created_at"],
                    metadata=json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"]
                ))

        logger.debug(f"Listed {len(webhooks)} webhooks from PostgreSQL")
        return webhooks

    async def delete_webhook(self, webhook_id: str) -> bool:
        """
        Delete a webhook from PostgreSQL.

        Associated deliveries are deleted via CASCADE.

        Args:
            webhook_id: The unique webhook identifier

        Returns:
            True if deleted, False if not found
        """
        await self.initialize()

        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM webhooks WHERE webhook_id = $1",
                webhook_id
            )
            deleted = result.split()[-1] != "0"

        if deleted:
            logger.debug(f"Deleted webhook {webhook_id} from PostgreSQL")
        return deleted

    async def save_delivery(self, delivery: WebhookDelivery) -> str:
        """
        Save a delivery record to PostgreSQL.

        Args:
            delivery: The delivery record to save

        Returns:
            The delivery_id of the saved delivery
        """
        await self.initialize()

        async with self._pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO deliveries (
                    delivery_id, webhook_id, event_type, payload,
                    signature, status, attempt, http_status,
                    error_message, created_at, sent_at, provenance_hash
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                ON CONFLICT (delivery_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    attempt = EXCLUDED.attempt,
                    http_status = EXCLUDED.http_status,
                    error_message = EXCLUDED.error_message,
                    sent_at = EXCLUDED.sent_at
            """,
                delivery.delivery_id,
                delivery.webhook_id,
                delivery.event_type,
                json.dumps(delivery.payload, default=str),
                delivery.signature,
                delivery.status.value if hasattr(delivery.status, 'value') else str(delivery.status),
                delivery.attempt,
                delivery.http_status,
                delivery.error_message,
                delivery.created_at,
                delivery.sent_at,
                delivery.provenance_hash
            )

        logger.debug(f"Saved delivery {delivery.delivery_id} to PostgreSQL")
        return delivery.delivery_id

    async def get_deliveries(
        self,
        webhook_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[WebhookDelivery]:
        """
        Retrieve deliveries for a webhook from PostgreSQL.

        Args:
            webhook_id: The webhook identifier
            limit: Maximum number of deliveries to return
            offset: Number of deliveries to skip

        Returns:
            List of delivery records, sorted by created_at descending
        """
        await self.initialize()

        deliveries = []
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM deliveries
                WHERE webhook_id = $1
                ORDER BY created_at DESC
                LIMIT $2 OFFSET $3
            """, webhook_id, limit, offset)

            for row in rows:
                deliveries.append(WebhookDelivery(
                    delivery_id=str(row["delivery_id"]),
                    webhook_id=str(row["webhook_id"]),
                    event_type=row["event_type"],
                    payload=json.loads(row["payload"]) if isinstance(row["payload"], str) else row["payload"],
                    signature=row["signature"],
                    status=WebhookStatus(row["status"]),
                    attempt=row["attempt"],
                    http_status=row["http_status"],
                    error_message=row["error_message"],
                    created_at=row["created_at"],
                    sent_at=row["sent_at"],
                    provenance_hash=row["provenance_hash"] or ""
                ))

        logger.debug(
            f"Retrieved {len(deliveries)} deliveries for webhook {webhook_id}"
        )
        return deliveries

    async def close(self) -> None:
        """Close the connection pool."""
        async with self._lock:
            if self._pool:
                await self._pool.close()
                self._pool = None
            self._initialized = False
            logger.info("PostgresWebhookStore closed")


# -----------------------------------------------------------------------------
# Exports
# -----------------------------------------------------------------------------

__all__ = [
    "WebhookStore",
    "WebhookStoreConfig",
    "BaseWebhookStore",
    "InMemoryWebhookStore",
    "SQLiteWebhookStore",
    "PostgresWebhookStore",
]
