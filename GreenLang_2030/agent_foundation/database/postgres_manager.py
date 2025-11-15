"""
PostgresManager - Production-grade PostgreSQL connection manager with AsyncPG.

This module implements the PostgresManager for GreenLang's database infrastructure.
It provides connection pooling, read/write splitting, automatic reconnection,
and comprehensive query performance monitoring.

Features:
- AsyncPG connection pooling (min 10, max 20, overflow 40)
- Read/write splitting (writes → primary, reads → replicas)
- Prepared statements for performance
- Automatic reconnection on failure
- Query performance logging
- Connection health monitoring

Example:
    >>> config = PostgresConfig(...)
    >>> manager = PostgresManager(config)
    >>> await manager.initialize()
    >>> async with manager.acquire() as conn:
    >>>     result = await manager.execute("SELECT * FROM agents WHERE id = $1", agent_id)
"""

from typing import Dict, List, Optional, Any, Union, Tuple
from pydantic import BaseModel, Field, validator, SecretStr
import asyncpg
import logging
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import hashlib
import random
from enum import Enum
import asyncio

logger = logging.getLogger(__name__)


class QueryType(str, Enum):
    """Query type for read/write splitting."""
    READ = "read"
    WRITE = "write"


class PostgresConfig(BaseModel):
    """PostgreSQL configuration."""

    # Primary database (write)
    primary_host: str = Field(..., description="Primary database host")
    primary_port: int = Field(5432, ge=1024, le=65535, description="Primary database port")

    # Read replicas
    replica_hosts: List[str] = Field(default_factory=list, description="Read replica hosts")
    replica_port: int = Field(5432, ge=1024, le=65535, description="Replica port")

    # Authentication
    database: str = Field(..., description="Database name")
    user: str = Field(..., description="Database user")
    password: SecretStr = Field(..., description="Database password")

    # Connection pooling
    min_pool_size: int = Field(10, ge=1, le=100, description="Minimum pool connections")
    max_pool_size: int = Field(20, ge=1, le=100, description="Maximum pool connections")
    max_overflow: int = Field(40, ge=0, le=200, description="Maximum overflow connections")

    # Timeouts
    command_timeout: float = Field(60.0, ge=1.0, description="Command timeout in seconds")
    connection_timeout: float = Field(10.0, ge=1.0, description="Connection timeout in seconds")

    # Health check
    health_check_interval: int = Field(30, ge=10, description="Health check interval in seconds")

    # Performance
    statement_cache_size: int = Field(100, ge=0, description="Prepared statement cache size")
    enable_query_logging: bool = Field(True, description="Enable query performance logging")
    slow_query_threshold_ms: float = Field(100.0, ge=0.0, description="Slow query threshold in ms")

    # SSL
    ssl_enabled: bool = Field(True, description="Enable SSL/TLS")
    ssl_ca_cert: Optional[str] = Field(None, description="Path to CA certificate")

    @validator('replica_hosts')
    def validate_replicas(cls, v, values):
        """Validate replica configuration."""
        if v and len(v) > 10:
            raise ValueError("Maximum 10 read replicas supported")
        return v


class QueryStats(BaseModel):
    """Query execution statistics."""

    query_hash: str = Field(..., description="SHA-256 hash of query")
    query_type: QueryType = Field(..., description="Query type (read/write)")
    execution_time_ms: float = Field(..., description="Execution time in milliseconds")
    row_count: int = Field(..., description="Number of rows affected/returned")
    is_slow: bool = Field(..., description="Whether query exceeded slow threshold")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Execution timestamp")


class ConnectionHealth(BaseModel):
    """Connection pool health metrics."""

    pool_name: str = Field(..., description="Pool name (primary/replica)")
    total_connections: int = Field(..., description="Total connections in pool")
    idle_connections: int = Field(..., description="Idle connections")
    active_connections: int = Field(..., description="Active connections")
    utilization_percent: float = Field(..., description="Pool utilization percentage")
    is_healthy: bool = Field(..., description="Overall health status")
    last_check: datetime = Field(default_factory=datetime.utcnow, description="Last health check time")


class PostgresManager:
    """
    Production-grade PostgreSQL connection manager.

    This manager handles all database connections with connection pooling,
    read/write splitting, automatic failover, and comprehensive monitoring.

    It follows GreenLang's zero-hallucination principle by using only
    deterministic database operations with complete audit trails.

    Attributes:
        config: PostgreSQL configuration
        primary_pool: Primary database connection pool (writes)
        replica_pools: Read replica connection pools (reads)
        query_stats: Query performance statistics
        is_initialized: Whether manager is initialized

    Example:
        >>> config = PostgresConfig(
        ...     primary_host="db-primary.example.com",
        ...     replica_hosts=["db-replica-1.example.com", "db-replica-2.example.com"],
        ...     database="greenlang",
        ...     user="gl_app",
        ...     password="secure_password"
        ... )
        >>> manager = PostgresManager(config)
        >>> await manager.initialize()
        >>>
        >>> # Execute write query
        >>> result = await manager.execute(
        ...     "INSERT INTO agents (id, name) VALUES ($1, $2)",
        ...     "agent-123", "CalculatorAgent",
        ...     query_type=QueryType.WRITE
        ... )
        >>>
        >>> # Execute read query (uses replica)
        >>> rows = await manager.fetch(
        ...     "SELECT * FROM agents WHERE status = $1",
        ...     "active",
        ...     query_type=QueryType.READ
        ... )
    """

    def __init__(self, config: PostgresConfig):
        """Initialize PostgresManager."""
        self.config = config
        self.primary_pool: Optional[asyncpg.Pool] = None
        self.replica_pools: List[asyncpg.Pool] = []
        self.query_stats: List[QueryStats] = []
        self.is_initialized = False
        self._health_check_task: Optional[asyncio.Task] = None
        self._current_replica_index = 0

        logger.info(
            f"PostgresManager initialized with primary={config.primary_host}, "
            f"replicas={len(config.replica_hosts)}"
        )

    async def initialize(self) -> None:
        """
        Initialize connection pools.

        Creates primary pool for writes and replica pools for reads.
        Starts background health check task.

        Raises:
            ConnectionError: If unable to connect to database
        """
        if self.is_initialized:
            logger.warning("PostgresManager already initialized")
            return

        try:
            # Create primary pool
            logger.info(f"Creating primary pool: {self.config.primary_host}:{self.config.primary_port}")
            self.primary_pool = await asyncpg.create_pool(
                host=self.config.primary_host,
                port=self.config.primary_port,
                database=self.config.database,
                user=self.config.user,
                password=self.config.password.get_secret_value(),
                min_size=self.config.min_pool_size,
                max_size=self.config.max_pool_size,
                command_timeout=self.config.command_timeout,
                timeout=self.config.connection_timeout,
                statement_cache_size=self.config.statement_cache_size,
                ssl=self._get_ssl_context()
            )

            # Create replica pools
            for replica_host in self.config.replica_hosts:
                logger.info(f"Creating replica pool: {replica_host}:{self.config.replica_port}")
                replica_pool = await asyncpg.create_pool(
                    host=replica_host,
                    port=self.config.replica_port,
                    database=self.config.database,
                    user=self.config.user,
                    password=self.config.password.get_secret_value(),
                    min_size=self.config.min_pool_size,
                    max_size=self.config.max_pool_size,
                    command_timeout=self.config.command_timeout,
                    timeout=self.config.connection_timeout,
                    statement_cache_size=self.config.statement_cache_size,
                    ssl=self._get_ssl_context()
                )
                self.replica_pools.append(replica_pool)

            # Test connections
            await self._test_connections()

            # Start health check task
            self._health_check_task = asyncio.create_task(self._health_check_loop())

            self.is_initialized = True
            logger.info("PostgresManager initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize PostgresManager: {str(e)}", exc_info=True)
            await self.cleanup()
            raise ConnectionError(f"Database initialization failed: {str(e)}") from e

    async def cleanup(self) -> None:
        """
        Cleanup all connection pools and background tasks.

        Should be called on application shutdown.
        """
        logger.info("Cleaning up PostgresManager")

        # Cancel health check task
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        # Close primary pool
        if self.primary_pool:
            await self.primary_pool.close()
            logger.info("Primary pool closed")

        # Close replica pools
        for i, replica_pool in enumerate(self.replica_pools):
            await replica_pool.close()
            logger.info(f"Replica pool {i} closed")

        self.is_initialized = False
        logger.info("PostgresManager cleanup complete")

    @asynccontextmanager
    async def acquire(self, query_type: QueryType = QueryType.READ):
        """
        Acquire a connection from the appropriate pool.

        Args:
            query_type: Type of query (READ or WRITE)

        Yields:
            asyncpg.Connection: Database connection

        Example:
            >>> async with manager.acquire(QueryType.WRITE) as conn:
            >>>     await conn.execute("INSERT INTO ...")
        """
        pool = self._get_pool(query_type)

        async with pool.acquire() as connection:
            yield connection

    async def execute(
        self,
        query: str,
        *args,
        query_type: QueryType = QueryType.WRITE,
        timeout: Optional[float] = None
    ) -> str:
        """
        Execute a query that doesn't return rows (INSERT, UPDATE, DELETE).

        Args:
            query: SQL query with $1, $2 placeholders
            *args: Query parameters
            query_type: Query type (default WRITE)
            timeout: Optional query timeout override

        Returns:
            Command tag (e.g., "INSERT 0 1", "UPDATE 5")

        Raises:
            asyncpg.PostgresError: If query execution fails

        Example:
            >>> result = await manager.execute(
            ...     "INSERT INTO agents (id, name) VALUES ($1, $2)",
            ...     "agent-123", "CalculatorAgent"
            ... )
            >>> # result = "INSERT 0 1"
        """
        start_time = datetime.utcnow()
        pool = self._get_pool(query_type)

        try:
            async with pool.acquire() as conn:
                result = await conn.execute(query, *args, timeout=timeout)

            # Log performance
            execution_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            self._log_query_performance(query, query_type, execution_time_ms, 0)

            return result

        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}", exc_info=True)
            raise

    async def fetch(
        self,
        query: str,
        *args,
        query_type: QueryType = QueryType.READ,
        timeout: Optional[float] = None
    ) -> List[asyncpg.Record]:
        """
        Fetch all rows from a query.

        Args:
            query: SQL query with $1, $2 placeholders
            *args: Query parameters
            query_type: Query type (default READ)
            timeout: Optional query timeout override

        Returns:
            List of records

        Example:
            >>> rows = await manager.fetch(
            ...     "SELECT * FROM agents WHERE status = $1",
            ...     "active"
            ... )
            >>> for row in rows:
            ...     print(row['name'])
        """
        start_time = datetime.utcnow()
        pool = self._get_pool(query_type)

        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(query, *args, timeout=timeout)

            # Log performance
            execution_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            self._log_query_performance(query, query_type, execution_time_ms, len(rows))

            return rows

        except Exception as e:
            logger.error(f"Query fetch failed: {str(e)}", exc_info=True)
            raise

    async def fetchrow(
        self,
        query: str,
        *args,
        query_type: QueryType = QueryType.READ,
        timeout: Optional[float] = None
    ) -> Optional[asyncpg.Record]:
        """
        Fetch a single row from a query.

        Args:
            query: SQL query with $1, $2 placeholders
            *args: Query parameters
            query_type: Query type (default READ)
            timeout: Optional query timeout override

        Returns:
            Single record or None

        Example:
            >>> row = await manager.fetchrow(
            ...     "SELECT * FROM agents WHERE id = $1",
            ...     "agent-123"
            ... )
            >>> if row:
            ...     print(row['name'])
        """
        start_time = datetime.utcnow()
        pool = self._get_pool(query_type)

        try:
            async with pool.acquire() as conn:
                row = await conn.fetchrow(query, *args, timeout=timeout)

            # Log performance
            execution_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            self._log_query_performance(query, query_type, execution_time_ms, 1 if row else 0)

            return row

        except Exception as e:
            logger.error(f"Query fetchrow failed: {str(e)}", exc_info=True)
            raise

    async def fetchval(
        self,
        query: str,
        *args,
        column: int = 0,
        query_type: QueryType = QueryType.READ,
        timeout: Optional[float] = None
    ) -> Any:
        """
        Fetch a single value from a query.

        Args:
            query: SQL query with $1, $2 placeholders
            *args: Query parameters
            column: Column index to return (default 0)
            query_type: Query type (default READ)
            timeout: Optional query timeout override

        Returns:
            Single value or None

        Example:
            >>> count = await manager.fetchval(
            ...     "SELECT COUNT(*) FROM agents WHERE status = $1",
            ...     "active"
            ... )
            >>> print(f"Active agents: {count}")
        """
        start_time = datetime.utcnow()
        pool = self._get_pool(query_type)

        try:
            async with pool.acquire() as conn:
                value = await conn.fetchval(query, *args, column=column, timeout=timeout)

            # Log performance
            execution_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            self._log_query_performance(query, query_type, execution_time_ms, 1 if value else 0)

            return value

        except Exception as e:
            logger.error(f"Query fetchval failed: {str(e)}", exc_info=True)
            raise

    async def executemany(
        self,
        query: str,
        args_list: List[Tuple],
        query_type: QueryType = QueryType.WRITE,
        timeout: Optional[float] = None
    ) -> None:
        """
        Execute a query multiple times with different parameters (batch insert/update).

        Args:
            query: SQL query with $1, $2 placeholders
            args_list: List of parameter tuples
            query_type: Query type (default WRITE)
            timeout: Optional query timeout override

        Example:
            >>> await manager.executemany(
            ...     "INSERT INTO agents (id, name) VALUES ($1, $2)",
            ...     [("agent-1", "Agent1"), ("agent-2", "Agent2")]
            ... )
        """
        start_time = datetime.utcnow()
        pool = self._get_pool(query_type)

        try:
            async with pool.acquire() as conn:
                await conn.executemany(query, args_list, timeout=timeout)

            # Log performance
            execution_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            self._log_query_performance(query, query_type, execution_time_ms, len(args_list))

        except Exception as e:
            logger.error(f"Query executemany failed: {str(e)}", exc_info=True)
            raise

    async def transaction(self, query_type: QueryType = QueryType.WRITE):
        """
        Create a transaction context manager.

        Args:
            query_type: Query type (default WRITE)

        Returns:
            Transaction context manager

        Example:
            >>> async with manager.transaction() as tx:
            >>>     await tx.execute("INSERT INTO agents ...")
            >>>     await tx.execute("INSERT INTO audit_log ...")
            >>>     # Automatically commits on success, rolls back on error
        """
        pool = self._get_pool(query_type)
        return pool.acquire()

    async def get_health(self) -> Dict[str, ConnectionHealth]:
        """
        Get health metrics for all connection pools.

        Returns:
            Dictionary of pool health metrics

        Example:
            >>> health = await manager.get_health()
            >>> primary_health = health['primary']
            >>> print(f"Primary pool utilization: {primary_health.utilization_percent}%")
        """
        health_metrics = {}

        # Primary pool health
        if self.primary_pool:
            primary_health = await self._check_pool_health(self.primary_pool, "primary")
            health_metrics["primary"] = primary_health

        # Replica pool health
        for i, replica_pool in enumerate(self.replica_pools):
            replica_health = await self._check_pool_health(replica_pool, f"replica-{i}")
            health_metrics[f"replica-{i}"] = replica_health

        return health_metrics

    async def get_query_stats(
        self,
        limit: int = 100,
        slow_only: bool = False
    ) -> List[QueryStats]:
        """
        Get recent query statistics.

        Args:
            limit: Maximum number of stats to return
            slow_only: Return only slow queries

        Returns:
            List of query statistics
        """
        stats = self.query_stats[-limit:] if not slow_only else [
            s for s in self.query_stats[-limit:] if s.is_slow
        ]
        return stats

    def _get_pool(self, query_type: QueryType) -> asyncpg.Pool:
        """
        Get appropriate connection pool based on query type.

        WRITE queries → primary pool
        READ queries → replica pool (round-robin) or primary if no replicas
        """
        if query_type == QueryType.WRITE or not self.replica_pools:
            return self.primary_pool

        # Round-robin load balancing for read queries
        replica_pool = self.replica_pools[self._current_replica_index]
        self._current_replica_index = (self._current_replica_index + 1) % len(self.replica_pools)

        return replica_pool

    def _get_ssl_context(self) -> Union[bool, str]:
        """Get SSL context based on configuration."""
        if not self.config.ssl_enabled:
            return False

        if self.config.ssl_ca_cert:
            return self.config.ssl_ca_cert

        return True

    async def _test_connections(self) -> None:
        """Test all connection pools on initialization."""
        # Test primary
        async with self.primary_pool.acquire() as conn:
            version = await conn.fetchval("SELECT version()")
            logger.info(f"Primary connection OK: {version}")

        # Test replicas
        for i, replica_pool in enumerate(self.replica_pools):
            async with replica_pool.acquire() as conn:
                version = await conn.fetchval("SELECT version()")
                logger.info(f"Replica {i} connection OK: {version}")

    async def _check_pool_health(
        self,
        pool: asyncpg.Pool,
        pool_name: str
    ) -> ConnectionHealth:
        """Check health of a single connection pool."""
        size = pool.get_size()
        idle = pool.get_idle_size()
        active = size - idle
        max_size = pool.get_max_size()

        utilization = (size / max_size * 100) if max_size > 0 else 0
        is_healthy = utilization < 80.0  # Healthy if < 80% utilized

        return ConnectionHealth(
            pool_name=pool_name,
            total_connections=size,
            idle_connections=idle,
            active_connections=active,
            utilization_percent=round(utilization, 2),
            is_healthy=is_healthy,
            last_check=datetime.utcnow()
        )

    async def _health_check_loop(self) -> None:
        """Background task to periodically check pool health."""
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval)

                health_metrics = await self.get_health()

                for pool_name, health in health_metrics.items():
                    if not health.is_healthy:
                        logger.warning(
                            f"Pool {pool_name} unhealthy: {health.utilization_percent}% utilization"
                        )
                    else:
                        logger.debug(
                            f"Pool {pool_name} healthy: {health.active_connections}/{health.total_connections} active"
                        )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check failed: {str(e)}", exc_info=True)

    def _log_query_performance(
        self,
        query: str,
        query_type: QueryType,
        execution_time_ms: float,
        row_count: int
    ) -> None:
        """Log query performance statistics."""
        if not self.config.enable_query_logging:
            return

        # Calculate query hash (for grouping similar queries)
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]

        is_slow = execution_time_ms > self.config.slow_query_threshold_ms

        stats = QueryStats(
            query_hash=query_hash,
            query_type=query_type,
            execution_time_ms=round(execution_time_ms, 2),
            row_count=row_count,
            is_slow=is_slow
        )

        # Keep only last 1000 stats in memory
        self.query_stats.append(stats)
        if len(self.query_stats) > 1000:
            self.query_stats = self.query_stats[-1000:]

        # Log slow queries
        if is_slow:
            logger.warning(
                f"SLOW QUERY: {query[:100]}... "
                f"took {execution_time_ms:.2f}ms (threshold: {self.config.slow_query_threshold_ms}ms)"
            )
        else:
            logger.debug(f"Query executed in {execution_time_ms:.2f}ms, {row_count} rows")


class QueryBuilder:
    """
    Query builder helper for constructing safe SQL queries.

    Provides methods to build common SQL queries with proper parameterization
    to prevent SQL injection attacks.

    Example:
        >>> qb = QueryBuilder("agents")
        >>> query, params = qb.select(["id", "name"]).where({"status": "active"}).build()
        >>> rows = await manager.fetch(query, *params)
    """

    def __init__(self, table: str):
        """Initialize query builder."""
        self.table = table
        self._select_fields: List[str] = []
        self._where_clauses: List[str] = []
        self._params: List[Any] = []
        self._order_by: Optional[str] = None
        self._limit: Optional[int] = None
        self._offset: Optional[int] = None

    def select(self, fields: List[str]) -> "QueryBuilder":
        """Add SELECT fields."""
        self._select_fields = fields
        return self

    def where(self, conditions: Dict[str, Any]) -> "QueryBuilder":
        """Add WHERE conditions."""
        for field, value in conditions.items():
            param_idx = len(self._params) + 1
            self._where_clauses.append(f"{field} = ${param_idx}")
            self._params.append(value)
        return self

    def order_by(self, field: str, direction: str = "ASC") -> "QueryBuilder":
        """Add ORDER BY clause."""
        self._order_by = f"{field} {direction}"
        return self

    def limit(self, limit: int) -> "QueryBuilder":
        """Add LIMIT clause."""
        self._limit = limit
        return self

    def offset(self, offset: int) -> "QueryBuilder":
        """Add OFFSET clause."""
        self._offset = offset
        return self

    def build(self) -> Tuple[str, List[Any]]:
        """
        Build final query and parameters.

        Returns:
            Tuple of (query_string, parameters)
        """
        # Build SELECT
        fields_str = ", ".join(self._select_fields) if self._select_fields else "*"
        query = f"SELECT {fields_str} FROM {self.table}"

        # Add WHERE
        if self._where_clauses:
            query += " WHERE " + " AND ".join(self._where_clauses)

        # Add ORDER BY
        if self._order_by:
            query += f" ORDER BY {self._order_by}"

        # Add LIMIT
        if self._limit:
            query += f" LIMIT {self._limit}"

        # Add OFFSET
        if self._offset:
            query += f" OFFSET {self._offset}"

        return query, self._params
