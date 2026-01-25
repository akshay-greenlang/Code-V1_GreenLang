"""
Async Database Client for GreenLang Agent Registry

This module provides the async database client using SQLAlchemy 2.0 with asyncpg.
Features:
- Connection pooling with configurable pool size
- Async context managers for session management
- Health check functionality
- Transaction support

Example:
    >>> from greenlang_registry.db.client import DatabaseClient
    >>> client = DatabaseClient(database_url)
    >>> async with client.session() as session:
    ...     result = await session.execute(select(Agent))
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncEngine,
    create_async_engine,
    async_sessionmaker,
)
from sqlalchemy import text
from sqlalchemy.pool import NullPool, AsyncAdaptedQueuePool

from greenlang_registry.db.models import Base

logger = logging.getLogger(__name__)


class DatabaseClient:
    """
    Async database client with connection pooling for PostgreSQL.

    This client manages database connections using SQLAlchemy 2.0 async
    patterns with the asyncpg driver. It supports connection pooling,
    health checks, and automatic session management.

    Attributes:
        engine: SQLAlchemy async engine
        session_factory: Factory for creating async sessions

    Example:
        >>> client = DatabaseClient("postgresql+asyncpg://user:pass@host/db")
        >>> await client.connect()
        >>> async with client.session() as session:
        ...     agents = await session.execute(select(Agent))
        >>> await client.disconnect()
    """

    def __init__(
        self,
        database_url: str,
        pool_size: int = 20,
        max_overflow: int = 10,
        pool_timeout: int = 30,
        pool_recycle: int = 1800,
        echo: bool = False,
        use_null_pool: bool = False,
    ):
        """
        Initialize database client.

        Args:
            database_url: PostgreSQL connection string (asyncpg format)
            pool_size: Number of connections to keep in the pool
            max_overflow: Additional connections allowed beyond pool_size
            pool_timeout: Seconds to wait for a connection from pool
            pool_recycle: Seconds before recycling connections
            echo: Enable SQL query logging
            use_null_pool: Disable pooling (useful for testing)
        """
        self._database_url = database_url
        self._pool_size = pool_size
        self._max_overflow = max_overflow
        self._pool_timeout = pool_timeout
        self._pool_recycle = pool_recycle
        self._echo = echo
        self._use_null_pool = use_null_pool

        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker[AsyncSession]] = None

    @property
    def engine(self) -> AsyncEngine:
        """Get the SQLAlchemy async engine."""
        if self._engine is None:
            raise RuntimeError("Database client not connected. Call connect() first.")
        return self._engine

    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        """Get the session factory."""
        if self._session_factory is None:
            raise RuntimeError("Database client not connected. Call connect() first.")
        return self._session_factory

    async def connect(self) -> None:
        """
        Initialize the database connection pool.

        Creates the async engine and session factory with configured
        pool settings. Should be called on application startup.

        Raises:
            RuntimeError: If connection fails
        """
        logger.info("Connecting to database...")

        pool_class = NullPool if self._use_null_pool else AsyncAdaptedQueuePool
        pool_kwargs = {}

        if not self._use_null_pool:
            pool_kwargs = {
                "pool_size": self._pool_size,
                "max_overflow": self._max_overflow,
                "pool_timeout": self._pool_timeout,
                "pool_recycle": self._pool_recycle,
            }

        try:
            self._engine = create_async_engine(
                self._database_url,
                echo=self._echo,
                poolclass=pool_class,
                **pool_kwargs,
            )

            self._session_factory = async_sessionmaker(
                bind=self._engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autocommit=False,
                autoflush=False,
            )

            # Verify connection
            async with self._engine.begin() as conn:
                await conn.execute(text("SELECT 1"))

            logger.info(
                "Database connected successfully. Pool size: %d, max overflow: %d",
                self._pool_size,
                self._max_overflow,
            )
        except Exception as e:
            logger.error("Failed to connect to database: %s", str(e))
            raise RuntimeError(f"Database connection failed: {str(e)}") from e

    async def disconnect(self) -> None:
        """
        Close all database connections.

        Disposes of the engine and closes all pooled connections.
        Should be called on application shutdown.
        """
        if self._engine is not None:
            logger.info("Disconnecting from database...")
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None
            logger.info("Database disconnected successfully")

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get an async database session.

        Provides a context manager that automatically handles
        session lifecycle, commit, and rollback on errors.

        Yields:
            AsyncSession: Database session for executing queries

        Example:
            >>> async with client.session() as session:
            ...     agent = Agent(agent_id="test", name="Test Agent")
            ...     session.add(agent)
            ...     # Commits automatically on exit
        """
        session = self.session_factory()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get a session with explicit transaction management.

        Use this for operations requiring explicit transaction control
        or when you need savepoints for partial rollbacks.

        Yields:
            AsyncSession: Database session within a transaction

        Example:
            >>> async with client.transaction() as session:
            ...     # Explicit transaction block
            ...     await session.execute(...)
        """
        async with self.session() as session:
            async with session.begin():
                yield session

    async def health_check(self) -> dict:
        """
        Check database connectivity and pool health.

        Returns:
            dict: Health status including pool metrics

        Example:
            >>> status = await client.health_check()
            >>> assert status["status"] == "healthy"
        """
        try:
            async with self._engine.begin() as conn:
                result = await conn.execute(text("SELECT 1"))
                result.fetchone()

            pool_status = {}
            if hasattr(self._engine.pool, "status"):
                pool_status = {
                    "pool_size": self._engine.pool.size(),
                    "checked_in": self._engine.pool.checkedin(),
                    "checked_out": self._engine.pool.checkedout(),
                    "overflow": self._engine.pool.overflow(),
                }

            return {
                "status": "healthy",
                "database": "connected",
                "pool": pool_status,
            }
        except Exception as e:
            logger.error("Database health check failed: %s", str(e))
            return {
                "status": "unhealthy",
                "database": "disconnected",
                "error": str(e),
            }

    async def create_tables(self) -> None:
        """
        Create all database tables.

        Uses SQLAlchemy metadata to create all defined tables.
        Safe to call multiple times (uses IF NOT EXISTS).
        """
        logger.info("Creating database tables...")
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created successfully")

    async def drop_tables(self) -> None:
        """
        Drop all database tables.

        WARNING: This will delete all data. Use with caution.
        """
        logger.warning("Dropping all database tables...")
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        logger.warning("All database tables dropped")


# Global client instance
_db_client: Optional[DatabaseClient] = None


def get_database_client() -> DatabaseClient:
    """
    Get the global database client instance.

    Returns:
        DatabaseClient: The configured database client

    Raises:
        RuntimeError: If client hasn't been initialized
    """
    if _db_client is None:
        raise RuntimeError("Database client not initialized. Call init_database() first.")
    return _db_client


async def init_database(
    database_url: str,
    pool_size: int = 20,
    max_overflow: int = 10,
    echo: bool = False,
) -> DatabaseClient:
    """
    Initialize the global database client.

    Args:
        database_url: PostgreSQL connection string
        pool_size: Connection pool size
        max_overflow: Max additional connections
        echo: Enable SQL logging

    Returns:
        DatabaseClient: Initialized database client
    """
    global _db_client

    _db_client = DatabaseClient(
        database_url=database_url,
        pool_size=pool_size,
        max_overflow=max_overflow,
        echo=echo,
    )
    await _db_client.connect()
    return _db_client


async def close_database() -> None:
    """Close the global database client connection."""
    global _db_client

    if _db_client is not None:
        await _db_client.disconnect()
        _db_client = None
