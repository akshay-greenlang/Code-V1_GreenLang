"""
Async PostgreSQL connection pool with pgvector support.

Manages connection pools for both writer (primary) and reader (replica)
endpoints, with automatic pgvector type registration.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

import psycopg
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from greenlang.data.vector.config import VectorDBConfig

logger = logging.getLogger(__name__)


async def _register_pgvector(conn: AsyncConnection) -> None:
    """Register pgvector types on a connection."""
    try:
        from pgvector.psycopg import register_vector_async
        await register_vector_async(conn)
    except ImportError:
        logger.warning(
            "pgvector Python package not installed. "
            "Install with: pip install pgvector"
        )
    except Exception as e:
        logger.error("Failed to register pgvector types: %s", e)
        raise


class VectorDBConnection:
    """
    Async PostgreSQL connection pool manager with pgvector support.

    Provides separate pools for writer (primary) and reader (replica)
    connections, with automatic pgvector type registration.

    Usage:
        db = VectorDBConnection(config)
        await db.initialize()

        async with db.acquire_writer() as conn:
            await conn.execute("INSERT INTO ...")

        async with db.acquire_reader() as conn:
            rows = await conn.execute("SELECT ...")

        await db.close()
    """

    def __init__(self, config: VectorDBConfig):
        self.config = config
        self._writer_pool: Optional[AsyncConnectionPool] = None
        self._reader_pool: Optional[AsyncConnectionPool] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize connection pools."""
        if self._initialized:
            return

        conninfo = (
            f"host={self.config.host} "
            f"port={self.config.port} "
            f"dbname={self.config.database} "
            f"user={self.config.user} "
            f"password={self.config.password} "
            f"sslmode={self.config.ssl_mode}"
        )

        self._writer_pool = AsyncConnectionPool(
            conninfo=conninfo,
            min_size=self.config.min_pool_size,
            max_size=self.config.max_pool_size,
            open=False,
            kwargs={"row_factory": dict_row, "autocommit": False},
            configure=_register_pgvector,
        )
        await self._writer_pool.open()

        if self.config.reader_host:
            reader_conninfo = (
                f"host={self.config.reader_host} "
                f"port={self.config.reader_port} "
                f"dbname={self.config.database} "
                f"user={self.config.user} "
                f"password={self.config.password} "
                f"sslmode={self.config.ssl_mode}"
            )
            self._reader_pool = AsyncConnectionPool(
                conninfo=reader_conninfo,
                min_size=self.config.min_pool_size,
                max_size=self.config.max_pool_size,
                open=False,
                kwargs={"row_factory": dict_row, "autocommit": False},
                configure=_register_pgvector,
            )
            await self._reader_pool.open()

        self._initialized = True
        logger.info(
            "VectorDB connection pools initialized (writer=%s, reader=%s)",
            self.config.host,
            self.config.reader_host or "none",
        )

    async def close(self) -> None:
        """Close all connection pools."""
        if self._writer_pool:
            await self._writer_pool.close()
        if self._reader_pool:
            await self._reader_pool.close()
        self._initialized = False
        logger.info("VectorDB connection pools closed")

    @asynccontextmanager
    async def acquire_writer(self) -> AsyncGenerator[AsyncConnection, None]:
        """Acquire a writer connection from the pool."""
        if not self._initialized:
            await self.initialize()
        async with self._writer_pool.connection() as conn:
            yield conn

    @asynccontextmanager
    async def acquire_reader(self) -> AsyncGenerator[AsyncConnection, None]:
        """
        Acquire a reader connection from the pool.
        Falls back to writer pool if no reader is configured.
        """
        if not self._initialized:
            await self.initialize()
        pool = self._reader_pool if self._reader_pool else self._writer_pool
        async with pool.connection() as conn:
            yield conn

    async def execute(
        self, query: str, params: tuple = (), use_reader: bool = False
    ) -> list:
        """Execute a query and return results."""
        acquire = self.acquire_reader if use_reader else self.acquire_writer
        async with acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, params)
                if cur.description:
                    return await cur.fetchall()
                return []

    async def execute_one(
        self, query: str, params: tuple = (), use_reader: bool = False
    ) -> Optional[dict]:
        """Execute a query and return a single result."""
        results = await self.execute(query, params, use_reader=use_reader)
        return results[0] if results else None

    async def health_check(self) -> dict:
        """Check connection health."""
        result = {"writer": False, "reader": False, "pgvector": False}
        try:
            async with self.acquire_writer() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("SELECT 1")
                    result["writer"] = True

                    await cur.execute(
                        "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
                    )
                    row = await cur.fetchone()
                    if row:
                        result["pgvector"] = True
                        result["pgvector_version"] = row["extversion"]
        except Exception as e:
            logger.error("Writer health check failed: %s", e)

        if self._reader_pool:
            try:
                async with self.acquire_reader() as conn:
                    async with conn.cursor() as cur:
                        await cur.execute("SELECT 1")
                        result["reader"] = True
            except Exception as e:
                logger.error("Reader health check failed: %s", e)

        return result

    async def set_search_params(
        self, conn: AsyncConnection, ef_search: int = 100, work_mem: str = "256MB"
    ) -> None:
        """Set pgvector search parameters for a session."""
        async with conn.cursor() as cur:
            await cur.execute(f"SET hnsw.ef_search = {ef_search}")
            await cur.execute(f"SET work_mem = '{work_mem}'")
