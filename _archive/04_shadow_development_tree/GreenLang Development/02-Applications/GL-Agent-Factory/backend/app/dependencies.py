"""
FastAPI Dependency Injection

This module provides dependency injection for FastAPI endpoints:
- Database session management
- Service layer instantiation
- Authentication dependencies

Example:
    >>> from app.dependencies import get_registry_service
    >>>
    >>> @router.get("/agents")
    >>> async def list_agents(
    ...     service: AgentRegistryService = Depends(get_registry_service)
    ... ):
    ...     return await service.list_agents()
"""

import logging
from typing import AsyncGenerator, Optional

from fastapi import Depends, HTTPException, Request, status

# Import database connection utilities
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db.connection import get_db_session, _async_session_factory
from registry.service import AgentRegistryService

logger = logging.getLogger(__name__)


async def get_db() -> AsyncGenerator:
    """
    Dependency for database session.

    Provides an async database session with automatic
    commit/rollback and cleanup.

    Yields:
        AsyncSession: Database session

    Example:
        >>> @router.get("/items")
        >>> async def get_items(db: AsyncSession = Depends(get_db)):
        ...     result = await db.execute(select(Item))
        ...     return result.scalars().all()
    """
    if _async_session_factory is None:
        # Database not initialized - use in-memory mode
        logger.warning("Database not initialized, using in-memory storage")
        yield None
        return

    async with get_db_session() as session:
        yield session


async def get_registry_service(
    request: Request,
    db=Depends(get_db),
) -> AgentRegistryService:
    """
    Dependency for Agent Registry Service.

    Provides an initialized AgentRegistryService with database
    connection. If database is not available, falls back to
    in-memory storage.

    Args:
        request: FastAPI request object
        db: Database session from get_db dependency

    Returns:
        AgentRegistryService instance

    Example:
        >>> @router.post("/agents")
        >>> async def create_agent(
        ...     data: AgentCreate,
        ...     service: AgentRegistryService = Depends(get_registry_service)
        ... ):
        ...     return await service.create_agent(**data.dict())
    """
    # Check if service is cached on app state
    if hasattr(request.app.state, 'registry_service') and db is None:
        return request.app.state.registry_service

    # Create new service instance with database session
    return AgentRegistryService(session=db)


def get_tenant_id(request: Request) -> Optional[str]:
    """
    Extract tenant ID from request.

    Checks headers and request state for tenant context.

    Args:
        request: FastAPI request object

    Returns:
        Tenant ID string or None
    """
    # Check header first
    tenant_id = request.headers.get("X-Tenant-ID")
    if tenant_id:
        return tenant_id

    # Check request state (set by middleware)
    if hasattr(request.state, "tenant_id"):
        return request.state.tenant_id

    return None


def get_user_id(request: Request) -> Optional[str]:
    """
    Extract user ID from request.

    Checks request state for authenticated user.

    Args:
        request: FastAPI request object

    Returns:
        User ID string or None
    """
    if hasattr(request.state, "user_id"):
        return request.state.user_id
    return None


def require_auth(request: Request) -> str:
    """
    Require authentication for endpoint.

    Raises HTTPException if user is not authenticated.

    Args:
        request: FastAPI request object

    Returns:
        Authenticated user ID

    Raises:
        HTTPException: If not authenticated
    """
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user_id


def require_tenant(request: Request) -> str:
    """
    Require tenant context for endpoint.

    Raises HTTPException if tenant is not set.

    Args:
        request: FastAPI request object

    Returns:
        Tenant ID

    Raises:
        HTTPException: If tenant not set
    """
    tenant_id = get_tenant_id(request)
    if not tenant_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tenant context required. Set X-Tenant-ID header.",
        )
    return tenant_id
