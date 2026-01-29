"""
Database session management for Review Console Backend.

This module provides async SQLAlchemy session management with connection
pooling optimized for FastAPI applications.

Features:
    - Async connection pooling with asyncpg
    - Automatic session cleanup via context managers
    - FastAPI dependency injection support
    - Health check queries

Example:
    >>> from review_console.db.session import get_db
    >>> async def get_items(db: AsyncSession = Depends(get_db)):
    ...     result = await db.execute(select(ReviewQueueItem))
    ...     return result.scalars().all()
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from review_console.config import get_settings

settings = get_settings()

# Create async engine with connection pooling
# Use NullPool in development to avoid connection issues during reloads
if settings.env == "development":
    async_engine = create_async_engine(
        settings.database_url,
        echo=settings.debug,
        poolclass=NullPool,
    )
else:
    async_engine = create_async_engine(
        settings.database_url,
        echo=False,
        pool_size=settings.database_pool_size,
        max_overflow=settings.database_max_overflow,
        pool_pre_ping=True,  # Enable connection health checks
        pool_recycle=3600,  # Recycle connections after 1 hour
    )

# Session factory
AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


@asynccontextmanager
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Context manager for async database sessions.

    Provides an async database session that is automatically closed
    and rolled back on exceptions.

    Yields:
        AsyncSession: SQLAlchemy async session

    Example:
        >>> async with get_async_session() as session:
        ...     result = await session.execute(query)
        ...     await session.commit()
    """
    session = AsyncSessionLocal()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency for database sessions.

    This dependency creates a new database session for each request
    and automatically handles cleanup.

    Yields:
        AsyncSession: SQLAlchemy async session

    Example:
        >>> @app.get("/items")
        ... async def get_items(db: AsyncSession = Depends(get_db)):
        ...     return await db.execute(select(ReviewQueueItem))
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def check_database_health() -> bool:
    """
    Check database connectivity.

    Executes a simple query to verify the database is reachable
    and responding.

    Returns:
        bool: True if database is healthy, False otherwise

    Example:
        >>> healthy = await check_database_health()
        >>> if not healthy:
        ...     raise HTTPException(503, "Database unavailable")
    """
    try:
        async with AsyncSessionLocal() as session:
            await session.execute("SELECT 1")
            return True
    except Exception:
        return False


async def init_db() -> None:
    """
    Initialize database tables.

    Creates all tables defined in the SQLAlchemy models if they
    don't exist. Should be called during application startup.

    Note:
        In production, use Alembic migrations instead of this function.

    Example:
        >>> @app.on_event("startup")
        ... async def startup():
        ...     await init_db()
    """
    from review_console.db.models import Base

    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    """
    Close database connections.

    Disposes of the engine and closes all connections in the pool.
    Should be called during application shutdown.

    Example:
        >>> @app.on_event("shutdown")
        ... async def shutdown():
        ...     await close_db()
    """
    await async_engine.dispose()
