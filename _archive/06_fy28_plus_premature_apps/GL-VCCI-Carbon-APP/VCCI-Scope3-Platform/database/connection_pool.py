# -*- coding: utf-8 -*-
"""
Database Connection Pooling Configuration
GL-VCCI Scope 3 Platform - Performance Optimization

This module provides optimized database connection pooling for maximum performance:
- SQLAlchemy connection pool tuning
- PgBouncer integration
- Connection health monitoring
- Pool statistics and metrics
- Automatic connection recycling

Performance Targets:
- Pool efficiency: >90%
- Connection acquisition: <10ms
- Zero connection timeouts
- Optimal resource utilization

Version: 1.0.0
Team: Performance Optimization Team
Date: 2025-11-09
"""

import logging
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from contextlib import asynccontextmanager

from sqlalchemy import create_engine, pool, event, text
from greenlang.determinism import DeterministicClock
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncEngine,
    AsyncSession,
    async_sessionmaker
)
from sqlalchemy.pool import QueuePool, NullPool

logger = logging.getLogger(__name__)


# ============================================================================
# CONNECTION POOL CONFIGURATION
# ============================================================================

@dataclass
class PoolConfig:
    """Database connection pool configuration"""
    # Pool size settings
    pool_size: int = 20  # Base connection pool size
    max_overflow: int = 10  # Additional connections beyond pool_size
    pool_timeout: int = 30  # Timeout for getting connection (seconds)
    pool_recycle: int = 3600  # Recycle connections after 1 hour
    pool_pre_ping: bool = True  # Test connections before use

    # Performance settings
    echo_pool: bool = False  # Log pool events (dev only)
    pool_use_lifo: bool = True  # Use LIFO for better connection reuse

    # Connection settings
    connect_args: Dict[str, Any] = None

    def __post_init__(self):
        if self.connect_args is None:
            self.connect_args = {
                "connect_timeout": 10,
                "server_settings": {
                    "application_name": "vcci-scope3-platform",
                    "jit": "off",  # Disable JIT compilation for faster connection
                }
            }


# Production-optimized configuration
PRODUCTION_POOL_CONFIG = PoolConfig(
    pool_size=20,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=3600,
    pool_pre_ping=True,
    echo_pool=False,
    pool_use_lifo=True
)

# Development configuration
DEVELOPMENT_POOL_CONFIG = PoolConfig(
    pool_size=5,
    max_overflow=5,
    pool_timeout=30,
    pool_recycle=3600,
    pool_pre_ping=True,
    echo_pool=True,
    pool_use_lifo=True
)

# High-load configuration
HIGH_LOAD_POOL_CONFIG = PoolConfig(
    pool_size=50,
    max_overflow=20,
    pool_timeout=30,
    pool_recycle=1800,  # More frequent recycling under load
    pool_pre_ping=True,
    echo_pool=False,
    pool_use_lifo=True
)


# ============================================================================
# ASYNC DATABASE ENGINE FACTORY
# ============================================================================

class DatabaseEngineFactory:
    """
    Factory for creating optimized database engines.

    Features:
    - Async engine creation
    - Connection pool optimization
    - Event listener setup
    - Health monitoring
    """

    @staticmethod
    def create_async_engine(
        database_url: str,
        pool_config: Optional[PoolConfig] = None,
        echo: bool = False
    ) -> AsyncEngine:
        """
        Create optimized async SQLAlchemy engine.

        Args:
            database_url: Database connection URL
            pool_config: Connection pool configuration
            echo: Whether to echo SQL statements

        Returns:
            Configured async engine
        """
        config = pool_config or PRODUCTION_POOL_CONFIG

        # Create async engine with optimal settings
        engine = create_async_engine(
            database_url,
            # Pool settings
            poolclass=QueuePool,
            pool_size=config.pool_size,
            max_overflow=config.max_overflow,
            pool_timeout=config.pool_timeout,
            pool_recycle=config.pool_recycle,
            pool_pre_ping=config.pool_pre_ping,
            pool_use_lifo=config.pool_use_lifo,
            # Connection settings
            connect_args=config.connect_args,
            # Logging
            echo=echo,
            echo_pool=config.echo_pool,
            # Performance
            future=True,  # Use SQLAlchemy 2.0 style
        )

        # Setup event listeners
        DatabaseEngineFactory._setup_event_listeners(engine.sync_engine)

        logger.info(
            f"Created async database engine with pool_size={config.pool_size}, "
            f"max_overflow={config.max_overflow}"
        )

        return engine

    @staticmethod
    def _setup_event_listeners(engine):
        """Setup SQLAlchemy event listeners for monitoring"""

        @event.listens_for(engine, "connect")
        def receive_connect(dbapi_conn, connection_record):
            """Called when a new connection is created"""
            logger.debug("Database connection created")

            # Set connection-level settings for performance
            cursor = dbapi_conn.cursor()
            try:
                # Set statement timeout (30 seconds)
                cursor.execute("SET statement_timeout = '30s'")

                # Set work_mem for complex queries
                cursor.execute("SET work_mem = '256MB'")

                # Disable synchronous commit for better write performance
                # (only in specific cases, not for critical transactions)
                # cursor.execute("SET synchronous_commit = 'off'")

            finally:
                cursor.close()

        @event.listens_for(engine, "checkout")
        def receive_checkout(dbapi_conn, connection_record, connection_proxy):
            """Called when connection is retrieved from pool"""
            logger.debug("Database connection checked out from pool")

        @event.listens_for(engine, "checkin")
        def receive_checkin(dbapi_conn, connection_record):
            """Called when connection is returned to pool"""
            logger.debug("Database connection checked in to pool")


# ============================================================================
# CONNECTION POOL MONITOR
# ============================================================================

class ConnectionPoolMonitor:
    """
    Monitor database connection pool health and performance.

    Tracks:
    - Pool size and utilization
    - Connection acquisition times
    - Connection errors
    - Pool overflow events
    """

    def __init__(self, engine: AsyncEngine):
        """
        Initialize pool monitor.

        Args:
            engine: SQLAlchemy async engine to monitor
        """
        self.engine = engine
        self.sync_engine = engine.sync_engine

        # Statistics
        self.total_checkouts = 0
        self.total_checkins = 0
        self.checkout_times: list = []  # Keep last 1000
        self.errors = 0

    def get_pool_stats(self) -> Dict[str, Any]:
        """
        Get current connection pool statistics.

        Returns:
            Dictionary with pool metrics
        """
        pool = self.sync_engine.pool

        stats = {
            "pool_size": pool.size(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "queue_size": pool.size() - pool.checkedout(),
            "total_checkouts": self.total_checkouts,
            "total_checkins": self.total_checkins,
            "errors": self.errors,
            "timestamp": DeterministicClock.utcnow().isoformat()
        }

        # Calculate utilization
        if pool.size() > 0:
            stats["utilization_pct"] = round(
                (pool.checkedout() / pool.size()) * 100, 2
            )
        else:
            stats["utilization_pct"] = 0.0

        # Calculate average checkout time
        if self.checkout_times:
            stats["avg_checkout_time_ms"] = round(
                sum(self.checkout_times) / len(self.checkout_times), 2
            )
        else:
            stats["avg_checkout_time_ms"] = 0.0

        return stats

    def log_pool_stats(self):
        """Log current pool statistics"""
        stats = self.get_pool_stats()

        logger.info(
            f"Connection Pool Stats: "
            f"size={stats['pool_size']}, "
            f"checked_out={stats['checked_out']}, "
            f"overflow={stats['overflow']}, "
            f"utilization={stats['utilization_pct']}%"
        )

        # Warning if pool is heavily utilized
        if stats['utilization_pct'] > 80:
            logger.warning(
                f"Connection pool utilization high: {stats['utilization_pct']}%. "
                "Consider increasing pool_size."
            )

        # Warning if overflow is being used
        if stats['overflow'] > 0:
            logger.warning(
                f"Connection pool overflow in use: {stats['overflow']} connections. "
                "Pool may be undersized."
            )

    @asynccontextmanager
    async def track_checkout(self):
        """Context manager to track connection checkout time"""
        start_time = time.time()

        try:
            yield
        finally:
            checkout_time_ms = (time.time() - start_time) * 1000

            # Store checkout time (keep last 1000)
            self.checkout_times.append(checkout_time_ms)
            if len(self.checkout_times) > 1000:
                self.checkout_times.pop(0)

            # Log slow checkouts
            if checkout_time_ms > 100:
                logger.warning(
                    f"Slow connection checkout: {checkout_time_ms:.2f}ms"
                )


# ============================================================================
# SESSION FACTORY
# ============================================================================

class OptimizedSessionFactory:
    """
    Factory for creating optimized database sessions.

    Features:
    - Session pooling
    - Automatic transaction management
    - Connection health checks
    - Performance monitoring
    """

    def __init__(
        self,
        engine: AsyncEngine,
        expire_on_commit: bool = False
    ):
        """
        Initialize session factory.

        Args:
            engine: SQLAlchemy async engine
            expire_on_commit: Whether to expire objects after commit
        """
        self.engine = engine
        self.monitor = ConnectionPoolMonitor(engine)

        # Create session factory
        self.session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=expire_on_commit,
            autoflush=False,  # Manual flush for better control
            autocommit=False  # Explicit commits
        )

    @asynccontextmanager
    async def session(self) -> AsyncSession:
        """
        Get database session with automatic transaction management.

        Usage:
            async with session_factory.session() as session:
                result = await session.execute(query)
                await session.commit()
        """
        async with self.monitor.track_checkout():
            async with self.session_factory() as session:
                try:
                    self.monitor.total_checkouts += 1
                    yield session

                except Exception as e:
                    self.monitor.errors += 1
                    await session.rollback()
                    logger.error(f"Session error: {e}")
                    raise

                finally:
                    self.monitor.total_checkins += 1

    async def health_check(self) -> bool:
        """
        Perform database health check.

        Returns:
            True if healthy, False otherwise
        """
        try:
            async with self.session() as session:
                await session.execute(text("SELECT 1"))
                return True

        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False


# ============================================================================
# PGBOUNCER CONFIGURATION
# ============================================================================

PGBOUNCER_CONFIG_TEMPLATE = """
# ============================================================================
# PgBouncer Configuration for GL-VCCI Scope 3 Platform
# ============================================================================
# PgBouncer provides connection pooling at the database level,
# reducing connection overhead and improving scalability.
#
# Installation:
#   apt-get install pgbouncer
#
# Configuration file location:
#   /etc/pgbouncer/pgbouncer.ini
#
# Start PgBouncer:
#   systemctl start pgbouncer
# ============================================================================

[databases]
vcci_scope3 = host={db_host} port={db_port} dbname={db_name}

[pgbouncer]
# Connection Pooling Mode
# - session: One server connection per client connection (default)
# - transaction: Server connection released after transaction
# - statement: Server connection released after each statement
pool_mode = transaction

# Pool Size Configuration
max_client_conn = 1000           # Maximum client connections
default_pool_size = 25           # Connections per user+database
min_pool_size = 5                # Minimum pool size
reserve_pool_size = 5            # Additional connections for emergencies
reserve_pool_timeout = 3         # Timeout for reserve pool (seconds)

# Server Connection Settings
server_lifetime = 3600           # Max connection lifetime (seconds)
server_idle_timeout = 600        # Close idle connections after 10 minutes
server_connect_timeout = 15      # Connection timeout (seconds)
server_login_retry = 15          # Retry failed login (seconds)

# DNS and Network
dns_max_ttl = 15                 # DNS cache TTL
dns_zone_check_period = 0        # Disable DNS zone checking

# Timeouts
query_timeout = 30               # Query timeout (seconds)
query_wait_timeout = 120         # Client wait timeout (seconds)
client_idle_timeout = 0          # No client idle timeout
idle_transaction_timeout = 0     # No idle transaction timeout

# Performance
max_db_connections = 100         # Global connection limit
max_user_connections = 100       # Per-user connection limit
pkt_buf = 4096                   # Packet buffer size
listen_backlog = 128             # TCP listen backlog

# Logging
log_connections = 1              # Log connections
log_disconnections = 1           # Log disconnections
log_pooler_errors = 1            # Log pooler errors
stats_period = 60                # Stats logging period (seconds)

# Listen Address
listen_addr = 0.0.0.0
listen_port = 6432

# Authentication
auth_type = md5
auth_file = /etc/pgbouncer/userlist.txt

# Admin Interface
admin_users = pgbouncer_admin
stats_users = pgbouncer_stats

[users]
# User-specific settings (optional)
# {db_user} = pool_size=50
"""


def generate_pgbouncer_config(
    db_host: str,
    db_port: int,
    db_name: str,
    db_user: str
) -> str:
    """
    Generate PgBouncer configuration file.

    Args:
        db_host: Database host
        db_port: Database port
        db_name: Database name
        db_user: Database user

    Returns:
        PgBouncer configuration as string
    """
    return PGBOUNCER_CONFIG_TEMPLATE.format(
        db_host=db_host,
        db_port=db_port,
        db_name=db_name,
        db_user=db_user
    )


# ============================================================================
# CONNECTION POOL UTILITIES
# ============================================================================

async def warm_connection_pool(engine: AsyncEngine, target_size: int = 10):
    """
    Pre-warm connection pool by creating connections.

    Args:
        engine: SQLAlchemy async engine
        target_size: Number of connections to create
    """
    logger.info(f"Warming connection pool (target: {target_size} connections)")

    sessions = []

    try:
        # Create sessions to warm pool
        session_factory = async_sessionmaker(engine, class_=AsyncSession)

        for i in range(target_size):
            session = session_factory()
            await session.execute(text("SELECT 1"))
            sessions.append(session)

        logger.info(f"Connection pool warmed: {target_size} connections created")

    finally:
        # Close all sessions
        for session in sessions:
            await session.close()


async def monitor_pool_health(
    monitor: ConnectionPoolMonitor,
    interval_seconds: int = 60
):
    """
    Continuously monitor pool health.

    Args:
        monitor: Connection pool monitor
        interval_seconds: Monitoring interval
    """
    import asyncio

    while True:
        monitor.log_pool_stats()
        await asyncio.sleep(interval_seconds)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

EXAMPLE_USAGE = """
# ============================================================================
# Connection Pool Usage Examples
# ============================================================================

# Example 1: Create Optimized Engine
# ----------------------------------------------------------------------------
from database.connection_pool import (
    DatabaseEngineFactory,
    PRODUCTION_POOL_CONFIG,
    OptimizedSessionFactory
)

# Create engine
engine = DatabaseEngineFactory.create_async_engine(
    database_url="postgresql+asyncpg://user:pass@localhost/vcci_scope3",
    pool_config=PRODUCTION_POOL_CONFIG
)

# Create session factory
session_factory = OptimizedSessionFactory(engine)


# Example 2: Use Database Session
# ----------------------------------------------------------------------------
async def get_emissions(supplier_id: str):
    async with session_factory.session() as session:
        result = await session.execute(
            select(Emission).where(Emission.supplier_id == supplier_id)
        )
        return result.scalars().all()


# Example 3: Monitor Pool Health
# ----------------------------------------------------------------------------
from database.connection_pool import ConnectionPoolMonitor

monitor = ConnectionPoolMonitor(engine)

# Get pool statistics
stats = monitor.get_pool_stats()
print(f"Pool utilization: {stats['utilization_pct']}%")

# Log stats periodically
import asyncio
asyncio.create_task(monitor_pool_health(monitor, interval_seconds=60))


# Example 4: PgBouncer Integration
# ----------------------------------------------------------------------------
# 1. Generate PgBouncer config
from database.connection_pool import generate_pgbouncer_config

config = generate_pgbouncer_config(
    db_host="localhost",
    db_port=5432,
    db_name="vcci_scope3",
    db_user="vcci_user"
)

# Save to /etc/pgbouncer/pgbouncer.ini
with open("/etc/pgbouncer/pgbouncer.ini", "w") as f:
    f.write(config)

# 2. Update database URL to use PgBouncer
DATABASE_URL = "postgresql+asyncpg://user:pass@localhost:6432/vcci_scope3"

# 3. Reduce application pool size (PgBouncer handles pooling)
PGBOUNCER_POOL_CONFIG = PoolConfig(
    pool_size=5,  # Smaller pool when using PgBouncer
    max_overflow=5,
    pool_recycle=-1,  # Disable recycling (PgBouncer handles this)
    pool_pre_ping=False  # Disable pre-ping (PgBouncer handles this)
)


# Example 5: Warm Connection Pool on Startup
# ----------------------------------------------------------------------------
from database.connection_pool import warm_connection_pool

# Warm pool during application startup
await warm_connection_pool(engine, target_size=10)
"""


__all__ = [
    "PoolConfig",
    "DatabaseEngineFactory",
    "ConnectionPoolMonitor",
    "OptimizedSessionFactory",
    "PRODUCTION_POOL_CONFIG",
    "DEVELOPMENT_POOL_CONFIG",
    "HIGH_LOAD_POOL_CONFIG",
    "generate_pgbouncer_config",
    "warm_connection_pool",
    "monitor_pool_health",
]
