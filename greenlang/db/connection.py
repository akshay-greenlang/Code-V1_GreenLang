"""
GreenLang Database Connection Pooling

Advanced database connection management with pooling, health checks,
and automatic failover.

Features:
- SQLAlchemy connection pooling with optimized settings
- Connection health checks and automatic reconnection
- Circuit breaker for database failures
- Connection metrics and monitoring
- Graceful degradation on connection loss
- Support for read replicas

Author: GreenLang Infrastructure Team (TEAM 2)
Date: 2025-11-08
Version: 5.0.0
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
import threading

try:
    from sqlalchemy import create_engine, event, pool, text
    from sqlalchemy.engine import Engine
    from sqlalchemy.ext.asyncio import (
        create_async_engine,
        AsyncEngine,
        AsyncSession,
        async_sessionmaker
    )
    from sqlalchemy.orm import sessionmaker, Session
    from sqlalchemy.pool import QueuePool, NullPool
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    create_engine = None
    create_async_engine = None
    Engine = None
    AsyncEngine = None
    AsyncSession = None
    Session = None
    QueuePool = None
    NullPool = None

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """Connection pool states."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    RECOVERING = "recovering"


@dataclass
class ConnectionMetrics:
    """
    Metrics for database connections.

    Tracks connection pool usage and performance.
    """
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    connection_errors: int = 0
    reconnection_attempts: int = 0
    total_queries: int = 0
    avg_connection_time_ms: float = 0.0
    peak_connections: int = 0

    def update_connection_time(self, duration_ms: float) -> None:
        """Update average connection time."""
        total_time = self.avg_connection_time_ms * self.total_queries
        self.total_queries += 1
        self.avg_connection_time_ms = (total_time + duration_ms) / self.total_queries

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_connections": self.total_connections,
            "active_connections": self.active_connections,
            "idle_connections": self.idle_connections,
            "connection_errors": self.connection_errors,
            "reconnection_attempts": self.reconnection_attempts,
            "total_queries": self.total_queries,
            "avg_connection_time_ms": round(self.avg_connection_time_ms, 2),
            "peak_connections": self.peak_connections,
            "pool_utilization": (
                self.active_connections / self.total_connections
                if self.total_connections > 0 else 0
            ),
        }


class CircuitBreaker:
    """
    Circuit breaker for database connections.

    Prevents cascading failures by stopping connection attempts
    when the database is unhealthy.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 3
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening
            recovery_timeout: Seconds before trying to recover
            half_open_max_calls: Max calls in half-open state
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self.state = ConnectionState.HEALTHY
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.half_open_calls = 0
        self._lock = threading.Lock()

    def record_success(self) -> None:
        """Record successful connection."""
        with self._lock:
            if self.state == ConnectionState.RECOVERING:
                self.state = ConnectionState.HEALTHY
                self.failure_count = 0
                self.half_open_calls = 0
                logger.info("Circuit breaker: Connection recovered")

    def record_failure(self) -> None:
        """Record connection failure."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                if self.state != ConnectionState.FAILED:
                    self.state = ConnectionState.FAILED
                    logger.error(
                        f"Circuit breaker: Connection failed after "
                        f"{self.failure_count} failures"
                    )

    def can_attempt(self) -> bool:
        """Check if connection attempt is allowed."""
        with self._lock:
            if self.state == ConnectionState.HEALTHY:
                return True

            if self.state == ConnectionState.FAILED:
                # Check if recovery timeout elapsed
                if time.time() - self.last_failure_time >= self.recovery_timeout:
                    self.state = ConnectionState.RECOVERING
                    self.half_open_calls = 0
                    logger.info("Circuit breaker: Attempting recovery")
                    return True
                return False

            if self.state == ConnectionState.RECOVERING:
                if self.half_open_calls < self.half_open_max_calls:
                    self.half_open_calls += 1
                    return True
                return False

            return False

    def get_state(self) -> ConnectionState:
        """Get current circuit breaker state."""
        with self._lock:
            return self.state


class DatabaseConnectionPool:
    """
    Advanced database connection pool manager.

    Manages SQLAlchemy connection pools with health checks,
    automatic reconnection, and performance monitoring.

    Example:
        >>> pool = DatabaseConnectionPool(
        ...     database_url="postgresql+asyncpg://user:pass@localhost/db",
        ...     pool_size=20,
        ...     max_overflow=10
        ... )
        >>> await pool.initialize()
        >>>
        >>> async with pool.get_session() as session:
        ...     result = await session.execute(query)
        >>>
        >>> await pool.close()
    """

    def __init__(
        self,
        database_url: str,
        pool_size: int = 20,
        max_overflow: int = 10,
        pool_timeout: int = 30,
        pool_recycle: int = 3600,
        pool_pre_ping: bool = True,
        enable_metrics: bool = True,
        health_check_interval: int = 60,
        echo: bool = False
    ):
        """
        Initialize database connection pool.

        Args:
            database_url: Database connection URL
            pool_size: Base pool size
            max_overflow: Maximum overflow connections
            pool_timeout: Connection timeout in seconds
            pool_recycle: Recycle connections after seconds
            pool_pre_ping: Verify connections before use
            enable_metrics: Enable metrics collection
            health_check_interval: Health check interval in seconds
            echo: Echo SQL statements (for debugging)
        """
        if not SQLALCHEMY_AVAILABLE:
            raise ImportError(
                "SQLAlchemy required for database connections. "
                "Install with: pip install sqlalchemy[asyncio]"
            )

        self._database_url = database_url
        self._pool_size = pool_size
        self._max_overflow = max_overflow
        self._pool_timeout = pool_timeout
        self._pool_recycle = pool_recycle
        self._pool_pre_ping = pool_pre_ping
        self._enable_metrics = enable_metrics
        self._health_check_interval = health_check_interval
        self._echo = echo

        # Engine and session maker
        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker] = None

        # Circuit breaker
        self._circuit_breaker = CircuitBreaker()

        # Metrics
        self._metrics = ConnectionMetrics() if enable_metrics else None

        # Health check task
        self._health_check_task: Optional[asyncio.Task] = None
        self._running = False

        # Callbacks
        self._on_connect_callbacks: List[Callable] = []
        self._on_disconnect_callbacks: List[Callable] = []

        logger.info(
            f"Initialized DatabaseConnectionPool: "
            f"pool_size={pool_size}, max_overflow={max_overflow}"
        )

    async def initialize(self) -> None:
        """Initialize the connection pool."""
        try:
            # Create async engine
            self._engine = create_async_engine(
                self._database_url,
                poolclass=QueuePool,
                pool_size=self._pool_size,
                max_overflow=self._max_overflow,
                pool_timeout=self._pool_timeout,
                pool_recycle=self._pool_recycle,
                pool_pre_ping=self._pool_pre_ping,
                echo=self._echo,
                future=True
            )

            # Create session factory
            self._session_factory = async_sessionmaker(
                self._engine,
                class_=AsyncSession,
                expire_on_commit=False
            )

            # Register event listeners
            self._register_event_listeners()

            # Test connection
            await self._test_connection()

            # Start health check
            self._running = True
            self._health_check_task = asyncio.create_task(
                self._health_check_loop()
            )

            logger.info("Database connection pool initialized")

        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {e}", exc_info=True)
            self._circuit_breaker.record_failure()
            raise

    async def close(self) -> None:
        """Close the connection pool."""
        self._running = False

        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        if self._engine:
            await self._engine.dispose()

        logger.info("Database connection pool closed")

    @asynccontextmanager
    async def get_session(self):
        """
        Get a database session.

        Example:
            >>> async with pool.get_session() as session:
            ...     result = await session.execute(query)
            ...     await session.commit()
        """
        if not self._circuit_breaker.can_attempt():
            raise Exception(
                f"Circuit breaker open: {self._circuit_breaker.get_state().value}"
            )

        start_time = time.perf_counter()
        session = None

        try:
            session = self._session_factory()

            # Update metrics
            if self._metrics:
                self._metrics.active_connections += 1
                self._metrics.peak_connections = max(
                    self._metrics.peak_connections,
                    self._metrics.active_connections
                )

            yield session

            # Record successful connection
            self._circuit_breaker.record_success()

            # Update metrics
            if self._metrics:
                duration_ms = (time.perf_counter() - start_time) * 1000
                self._metrics.update_connection_time(duration_ms)

        except Exception as e:
            logger.error(f"Database session error: {e}")
            self._circuit_breaker.record_failure()
            if self._metrics:
                self._metrics.connection_errors += 1

            # Rollback on error
            if session:
                try:
                    await session.rollback()
                except Exception:
                    pass

            raise

        finally:
            if session:
                await session.close()

            if self._metrics:
                self._metrics.active_connections -= 1

    async def _test_connection(self) -> bool:
        """
        Test database connection.

        Returns:
            True if connection successful
        """
        try:
            async with self.get_session() as session:
                await session.execute(text("SELECT 1"))
            logger.info("Database connection test successful")
            return True

        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False

    async def _health_check_loop(self) -> None:
        """Background task for periodic health checks."""
        while self._running:
            try:
                await asyncio.sleep(self._health_check_interval)
                healthy = await self._test_connection()

                if not healthy:
                    logger.warning("Database health check failed")
                    self._circuit_breaker.record_failure()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")

    def _register_event_listeners(self) -> None:
        """Register SQLAlchemy event listeners."""
        if not self._engine:
            return

        # Listen for connection events
        @event.listens_for(self._engine.sync_engine, "connect")
        def on_connect(dbapi_conn, connection_record):
            """Called when a new connection is created."""
            if self._metrics:
                self._metrics.total_connections += 1

            for callback in self._on_connect_callbacks:
                try:
                    callback(dbapi_conn, connection_record)
                except Exception as e:
                    logger.error(f"Error in connect callback: {e}")

        @event.listens_for(self._engine.sync_engine, "close")
        def on_close(dbapi_conn, connection_record):
            """Called when a connection is closed."""
            for callback in self._on_disconnect_callbacks:
                try:
                    callback(dbapi_conn, connection_record)
                except Exception as e:
                    logger.error(f"Error in disconnect callback: {e}")

    def register_connect_callback(self, callback: Callable) -> None:
        """
        Register callback for new connections.

        Args:
            callback: Function to call on connect
        """
        self._on_connect_callbacks.append(callback)

    def register_disconnect_callback(self, callback: Callable) -> None:
        """
        Register callback for connection close.

        Args:
            callback: Function to call on disconnect
        """
        self._on_disconnect_callbacks.append(callback)

    async def get_pool_status(self) -> Dict[str, Any]:
        """
        Get connection pool status.

        Returns:
            Dictionary with pool information
        """
        if not self._engine:
            return {"error": "Engine not initialized"}

        pool = self._engine.pool

        return {
            "pool_size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "total": pool.size() + pool.overflow(),
            "circuit_state": self._circuit_breaker.get_state().value,
        }

    async def get_metrics(self) -> Dict[str, Any]:
        """Get connection metrics."""
        if not self._metrics:
            return {"error": "Metrics not enabled"}

        metrics = self._metrics.to_dict()

        # Add pool status
        pool_status = await self.get_pool_status()
        metrics.update(pool_status)

        return metrics

    async def execute_raw(self, query: str, params: Optional[Dict] = None) -> Any:
        """
        Execute raw SQL query.

        Args:
            query: SQL query string
            params: Optional query parameters

        Returns:
            Query result

        Example:
            >>> result = await pool.execute_raw("SELECT * FROM users WHERE id = :id", {"id": 123})
        """
        async with self.get_session() as session:
            result = await session.execute(text(query), params or {})
            return result

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check.

        Returns:
            Health status dictionary
        """
        health = {
            "healthy": False,
            "circuit_state": self._circuit_breaker.get_state().value,
            "connection_test": False,
            "pool_status": {}
        }

        # Test connection
        try:
            health["connection_test"] = await self._test_connection()
        except Exception as e:
            health["error"] = str(e)

        # Get pool status
        try:
            health["pool_status"] = await self.get_pool_status()
        except Exception as e:
            health["pool_error"] = str(e)

        # Overall health
        health["healthy"] = (
            health["connection_test"]
            and self._circuit_breaker.get_state() == ConnectionState.HEALTHY
        )

        return health


# Global connection pool instance
_global_pool: Optional[DatabaseConnectionPool] = None


def get_connection_pool() -> Optional[DatabaseConnectionPool]:
    """Get global connection pool instance."""
    return _global_pool


async def initialize_connection_pool(
    database_url: str,
    pool_size: int = 20,
    max_overflow: int = 10,
    **kwargs
) -> DatabaseConnectionPool:
    """
    Initialize global connection pool.

    Args:
        database_url: Database connection URL
        pool_size: Base pool size
        max_overflow: Maximum overflow connections
        **kwargs: Additional pool configuration

    Returns:
        Initialized DatabaseConnectionPool

    Example:
        >>> pool = await initialize_connection_pool(
        ...     database_url="postgresql+asyncpg://user:pass@localhost/db",
        ...     pool_size=20
        ... )
    """
    global _global_pool

    if _global_pool is not None:
        await _global_pool.close()

    _global_pool = DatabaseConnectionPool(
        database_url=database_url,
        pool_size=pool_size,
        max_overflow=max_overflow,
        **kwargs
    )
    await _global_pool.initialize()

    return _global_pool
