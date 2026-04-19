"""
Database Connection Management

This module provides database connection pooling and session management.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

logger = logging.getLogger(__name__)

# Global engine and session factory
_engine = None
_async_session_factory = None


async def init_db_pool(
    database_url: str,
    pool_size: int = 20,
    max_overflow: int = 10,
    pool_timeout: int = 30,
    pool_recycle: int = 3600,
    echo: bool = False,
) -> None:
    """
    Initialize the database connection pool.

    Args:
        database_url: PostgreSQL connection URL
        pool_size: Base pool size
        max_overflow: Max connections beyond pool_size
        pool_timeout: Timeout for getting connection
        pool_recycle: Recycle connections after N seconds
        echo: Echo SQL statements
    """
    global _engine, _async_session_factory

    # Convert to async URL if needed
    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+asyncpg://")

    _engine = create_async_engine(
        database_url,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_timeout=pool_timeout,
        pool_recycle=pool_recycle,
        echo=echo,
    )

    _async_session_factory = async_sessionmaker(
        bind=_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    logger.info(f"Database pool initialized: pool_size={pool_size}, max_overflow={max_overflow}")


async def close_db_pool() -> None:
    """Close the database connection pool."""
    global _engine

    if _engine:
        await _engine.dispose()
        logger.info("Database pool closed")


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Get a database session.

    Usage:
        async with get_db_session() as session:
            result = await session.execute(query)

    Yields:
        AsyncSession: Database session
    """
    if not _async_session_factory:
        raise RuntimeError("Database pool not initialized. Call init_db_pool first.")

    session = _async_session_factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()
