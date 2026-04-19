"""
GL-EUDR-001: Database Integration Module

Provides PostgreSQL database connectivity with:
- SQLAlchemy async session management
- Connection pooling
- Transaction support
- Multi-tenancy via organization_id

Configuration:
    Set DATABASE_URL environment variable or configure programmatically.
    Example: postgresql+asyncpg://user:pass@localhost:5432/eudr_db
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from sqlalchemy import create_engine, event, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

class DatabaseConfig:
    """Database configuration."""

    # Sync URL (for migrations, testing)
    SYNC_DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/eudr_supply_chain"
    )

    # Async URL (for API operations)
    ASYNC_DATABASE_URL: str = os.getenv(
        "ASYNC_DATABASE_URL",
        SYNC_DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")
        if SYNC_DATABASE_URL else None
    )

    # Pool settings
    POOL_SIZE: int = int(os.getenv("DB_POOL_SIZE", "10"))
    MAX_OVERFLOW: int = int(os.getenv("DB_MAX_OVERFLOW", "20"))
    POOL_TIMEOUT: int = int(os.getenv("DB_POOL_TIMEOUT", "30"))
    POOL_RECYCLE: int = int(os.getenv("DB_POOL_RECYCLE", "1800"))  # 30 minutes

    # Query settings
    QUERY_TIMEOUT: int = int(os.getenv("DB_QUERY_TIMEOUT", "30"))  # seconds

    # Echo SQL (for debugging)
    ECHO_SQL: bool = os.getenv("DB_ECHO", "false").lower() == "true"


# =============================================================================
# ENGINE CREATION
# =============================================================================

def create_sync_engine():
    """Create synchronous SQLAlchemy engine."""
    engine = create_engine(
        DatabaseConfig.SYNC_DATABASE_URL,
        poolclass=QueuePool,
        pool_size=DatabaseConfig.POOL_SIZE,
        max_overflow=DatabaseConfig.MAX_OVERFLOW,
        pool_timeout=DatabaseConfig.POOL_TIMEOUT,
        pool_recycle=DatabaseConfig.POOL_RECYCLE,
        echo=DatabaseConfig.ECHO_SQL,
    )

    # Add event listeners for connection setup
    @event.listens_for(engine, "connect")
    def set_search_path(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("SET search_path TO public")
        cursor.close()

    return engine


def create_async_engine_instance():
    """Create asynchronous SQLAlchemy engine."""
    if not DatabaseConfig.ASYNC_DATABASE_URL:
        raise ValueError("ASYNC_DATABASE_URL not configured")

    engine = create_async_engine(
        DatabaseConfig.ASYNC_DATABASE_URL,
        pool_size=DatabaseConfig.POOL_SIZE,
        max_overflow=DatabaseConfig.MAX_OVERFLOW,
        pool_timeout=DatabaseConfig.POOL_TIMEOUT,
        pool_recycle=DatabaseConfig.POOL_RECYCLE,
        echo=DatabaseConfig.ECHO_SQL,
    )

    return engine


# =============================================================================
# SESSION FACTORIES
# =============================================================================

# Sync engine and session factory
_sync_engine = None
_sync_session_factory = None


def get_sync_engine():
    """Get or create sync engine."""
    global _sync_engine
    if _sync_engine is None:
        _sync_engine = create_sync_engine()
    return _sync_engine


def get_sync_session_factory():
    """Get or create sync session factory."""
    global _sync_session_factory
    if _sync_session_factory is None:
        _sync_session_factory = sessionmaker(
            bind=get_sync_engine(),
            autocommit=False,
            autoflush=False,
            expire_on_commit=False
        )
    return _sync_session_factory


# Async engine and session factory
_async_engine = None
_async_session_factory = None


def get_async_engine():
    """Get or create async engine."""
    global _async_engine
    if _async_engine is None:
        _async_engine = create_async_engine_instance()
    return _async_engine


def get_async_session_factory():
    """Get or create async session factory."""
    global _async_session_factory
    if _async_session_factory is None:
        _async_session_factory = async_sessionmaker(
            bind=get_async_engine(),
            class_=AsyncSession,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False
        )
    return _async_session_factory


# =============================================================================
# SESSION DEPENDENCY FOR FASTAPI
# =============================================================================

async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency for database sessions.

    Usage:
        @router.get("/endpoint")
        async def endpoint(db: AsyncSession = Depends(get_db_session)):
            ...
    """
    session_factory = get_async_session_factory()
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


def get_sync_db_session() -> Session:
    """
    Get synchronous database session.
    For use in non-async contexts (migrations, scripts).
    """
    session_factory = get_sync_session_factory()
    session = session_factory()
    try:
        return session
    except Exception:
        session.rollback()
        raise


@asynccontextmanager
async def get_db_context() -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager for database sessions.

    Usage:
        async with get_db_context() as db:
            result = await db.execute(query)
    """
    session_factory = get_async_session_factory()
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# =============================================================================
# MULTI-TENANCY SUPPORT
# =============================================================================

class TenantContext:
    """
    Context for multi-tenant operations.
    Ensures queries are scoped to the correct organization.
    """

    def __init__(self, organization_id: str):
        self.organization_id = organization_id

    def apply_filter(self, query, model):
        """
        Apply tenant filter to a query.

        Args:
            query: SQLAlchemy query
            model: Model class with organization_id column

        Returns:
            Filtered query
        """
        if hasattr(model, 'metadata') and 'organization_id' in model.__table__.columns:
            return query.filter(model.metadata['organization_id'] == self.organization_id)
        return query


# =============================================================================
# DATABASE UTILITIES
# =============================================================================

async def check_database_connection() -> bool:
    """Check if database is accessible."""
    try:
        async with get_db_context() as db:
            await db.execute(text("SELECT 1"))
            return True
    except Exception as e:
        logger.error(f"Database connection check failed: {e}")
        return False


async def init_database():
    """Initialize database schema."""
    from .models import Base

    engine = get_async_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    logger.info("Database schema initialized")


async def cleanup_database():
    """Cleanup database connections."""
    global _async_engine, _sync_engine

    if _async_engine:
        await _async_engine.dispose()
        _async_engine = None

    if _sync_engine:
        _sync_engine.dispose()
        _sync_engine = None

    logger.info("Database connections cleaned up")


# =============================================================================
# REPOSITORY BASE CLASS
# =============================================================================

class BaseRepository:
    """
    Base repository providing common database operations.
    Inherit from this class for specific entity repositories.
    """

    def __init__(self, session: AsyncSession, model_class):
        self.session = session
        self.model_class = model_class

    async def get_by_id(self, id_value):
        """Get entity by ID."""
        from sqlalchemy import select

        result = await self.session.execute(
            select(self.model_class).where(self.model_class.node_id == id_value)
        )
        return result.scalar_one_or_none()

    async def get_all(self, limit: int = 100, offset: int = 0):
        """Get all entities with pagination."""
        from sqlalchemy import select

        result = await self.session.execute(
            select(self.model_class).limit(limit).offset(offset)
        )
        return result.scalars().all()

    async def create(self, entity):
        """Create new entity."""
        self.session.add(entity)
        await self.session.flush()
        return entity

    async def update(self, entity):
        """Update existing entity."""
        await self.session.merge(entity)
        await self.session.flush()
        return entity

    async def delete(self, entity):
        """Delete entity."""
        await self.session.delete(entity)
        await self.session.flush()

    async def count(self) -> int:
        """Count all entities."""
        from sqlalchemy import func, select

        result = await self.session.execute(
            select(func.count()).select_from(self.model_class)
        )
        return result.scalar()


# =============================================================================
# SUPPLY CHAIN REPOSITORIES
# =============================================================================

class NodeRepository(BaseRepository):
    """Repository for supply chain nodes."""

    def __init__(self, session: AsyncSession):
        from .models import SupplyChainNodeModel
        super().__init__(session, SupplyChainNodeModel)

    async def get_by_id(self, node_id):
        """Get node by ID."""
        from sqlalchemy import select
        from .models import SupplyChainNodeModel

        result = await self.session.execute(
            select(SupplyChainNodeModel).where(
                SupplyChainNodeModel.node_id == node_id
            )
        )
        return result.scalar_one_or_none()

    async def get_by_commodity(self, commodity: str, limit: int = 100):
        """Get nodes by commodity."""
        from sqlalchemy import select
        from .models import SupplyChainNodeModel

        result = await self.session.execute(
            select(SupplyChainNodeModel).where(
                SupplyChainNodeModel.commodities.contains([commodity])
            ).limit(limit)
        )
        return result.scalars().all()

    async def get_by_organization(self, org_id: str, limit: int = 100):
        """Get nodes by organization."""
        from sqlalchemy import select
        from .models import SupplyChainNodeModel

        result = await self.session.execute(
            select(SupplyChainNodeModel).where(
                SupplyChainNodeModel.metadata['organization_id'].astext == org_id
            ).limit(limit)
        )
        return result.scalars().all()


class EdgeRepository(BaseRepository):
    """Repository for supply chain edges."""

    def __init__(self, session: AsyncSession):
        from .models import SupplyChainEdgeModel
        super().__init__(session, SupplyChainEdgeModel)

    async def get_incoming_edges(self, node_id, commodity: str = None):
        """Get edges where node is the target."""
        from sqlalchemy import select
        from .models import SupplyChainEdgeModel

        query = select(SupplyChainEdgeModel).where(
            SupplyChainEdgeModel.target_node_id == node_id
        )
        if commodity:
            query = query.where(SupplyChainEdgeModel.commodity == commodity)

        result = await self.session.execute(query)
        return result.scalars().all()

    async def get_outgoing_edges(self, node_id, commodity: str = None):
        """Get edges where node is the source."""
        from sqlalchemy import select
        from .models import SupplyChainEdgeModel

        query = select(SupplyChainEdgeModel).where(
            SupplyChainEdgeModel.source_node_id == node_id
        )
        if commodity:
            query = query.where(SupplyChainEdgeModel.commodity == commodity)

        result = await self.session.execute(query)
        return result.scalars().all()


class SnapshotRepository(BaseRepository):
    """Repository for supply chain snapshots."""

    def __init__(self, session: AsyncSession):
        from .models import SupplyChainSnapshotModel
        super().__init__(session, SupplyChainSnapshotModel)

    async def get_latest(self, importer_id, commodity: str):
        """Get latest snapshot for importer/commodity."""
        from sqlalchemy import select
        from .models import SupplyChainSnapshotModel

        result = await self.session.execute(
            select(SupplyChainSnapshotModel).where(
                SupplyChainSnapshotModel.importer_node_id == importer_id,
                SupplyChainSnapshotModel.commodity == commodity
            ).order_by(
                SupplyChainSnapshotModel.snapshot_date.desc()
            ).limit(1)
        )
        return result.scalar_one_or_none()

    async def get_as_of(self, importer_id, commodity: str, as_of):
        """Get snapshot as of a specific date."""
        from sqlalchemy import select
        from .models import SupplyChainSnapshotModel

        result = await self.session.execute(
            select(SupplyChainSnapshotModel).where(
                SupplyChainSnapshotModel.importer_node_id == importer_id,
                SupplyChainSnapshotModel.commodity == commodity,
                SupplyChainSnapshotModel.snapshot_date <= as_of
            ).order_by(
                SupplyChainSnapshotModel.snapshot_date.desc()
            ).limit(1)
        )
        return result.scalar_one_or_none()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Configuration
    "DatabaseConfig",

    # Engines
    "create_sync_engine",
    "create_async_engine_instance",
    "get_sync_engine",
    "get_async_engine",

    # Session factories
    "get_sync_session_factory",
    "get_async_session_factory",
    "get_db_session",
    "get_sync_db_session",
    "get_db_context",

    # Multi-tenancy
    "TenantContext",

    # Utilities
    "check_database_connection",
    "init_database",
    "cleanup_database",

    # Repositories
    "BaseRepository",
    "NodeRepository",
    "EdgeRepository",
    "SnapshotRepository",
]
