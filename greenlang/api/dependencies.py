# -*- coding: utf-8 -*-
"""
GreenLang API Dependencies
===========================

Common dependencies for FastAPI endpoints.
Provides dependency injection for authentication, database sessions,
and other shared resources.

Author: GreenLang Framework Team
"""

from typing import Optional, AsyncGenerator, Dict, Any
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, AsyncEngine
from sqlalchemy.orm import sessionmaker
import jwt
import logging
import os
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Security scheme
security = HTTPBearer()

# Database configuration (TODO: Move to config)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./greenlang.db")
JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

# Database engine and session
engine: Optional[AsyncEngine] = None
AsyncSessionLocal: Optional[sessionmaker] = None


def init_database(database_url: Optional[str] = None) -> None:
    """
    Initialize database engine and session factory.

    Args:
        database_url: Optional database URL override
    """
    global engine, AsyncSessionLocal

    url = database_url or DATABASE_URL
    engine = create_async_engine(url, echo=False)
    AsyncSessionLocal = sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False
    )
    logger.info(f"Initialized database connection: {url.split('@')[-1] if '@' in url else url}")


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Get database session dependency.

    Yields:
        AsyncSession: Database session

    Raises:
        RuntimeError: If database not initialized
    """
    if AsyncSessionLocal is None:
        init_database()

    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception as e:
            await session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            await session.close()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict[str, Any]:
    """
    Get current authenticated user from JWT token.

    Args:
        credentials: HTTP authorization credentials

    Returns:
        User information dictionary

    Raises:
        HTTPException: If authentication fails
    """
    token = credentials.credentials

    try:
        # Decode JWT token
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])

        # Check expiration
        exp = payload.get("exp")
        if exp and datetime.fromtimestamp(exp) < datetime.now():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"}
            )

        # Extract user information
        user = {
            "user_id": payload.get("sub"),
            "email": payload.get("email"),
            "tenant_id": payload.get("tenant_id"),
            "roles": payload.get("roles", []),
            "permissions": payload.get("permissions", [])
        }

        if not user["user_id"]:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload",
                headers={"WWW-Authenticate": "Bearer"}
            )

        logger.debug(f"Authenticated user: {user['user_id']}")
        return user

    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"}
        )
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid token: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"}
        )


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[Dict[str, Any]]:
    """
    Get current user if authenticated, None otherwise.

    Args:
        credentials: Optional HTTP authorization credentials

    Returns:
        User information or None
    """
    if not credentials:
        return None

    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None


def require_roles(*required_roles: str):
    """
    Dependency to require specific user roles.

    Args:
        *required_roles: Required role names

    Returns:
        Dependency function
    """
    async def check_roles(user: Dict = Depends(get_current_user)) -> Dict:
        user_roles = set(user.get("roles", []))
        if not any(role in user_roles for role in required_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Required roles: {', '.join(required_roles)}"
            )
        return user

    return check_roles


def require_permissions(*required_permissions: str):
    """
    Dependency to require specific permissions.

    Args:
        *required_permissions: Required permission names

    Returns:
        Dependency function
    """
    async def check_permissions(user: Dict = Depends(get_current_user)) -> Dict:
        user_permissions = set(user.get("permissions", []))
        missing = [p for p in required_permissions if p not in user_permissions]
        if missing:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing permissions: {', '.join(missing)}"
            )
        return user

    return check_permissions


class RateLimiter:
    """Simple rate limiting dependency."""

    def __init__(self, calls: int = 10, period: int = 60):
        """
        Initialize rate limiter.

        Args:
            calls: Number of calls allowed
            period: Time period in seconds
        """
        self.calls = calls
        self.period = period
        self.storage: Dict[str, list] = {}

    async def __call__(self, user: Dict = Depends(get_current_user)) -> None:
        """Check rate limit for user."""
        user_id = user["user_id"]
        now = datetime.now()

        # Clean old entries
        if user_id in self.storage:
            cutoff = now - timedelta(seconds=self.period)
            self.storage[user_id] = [
                t for t in self.storage[user_id]
                if t > cutoff
            ]

        # Check rate limit
        if user_id not in self.storage:
            self.storage[user_id] = []

        if len(self.storage[user_id]) >= self.calls:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded: {self.calls} calls per {self.period} seconds"
            )

        self.storage[user_id].append(now)


# Utility functions for creating JWT tokens (for testing/development)
def create_access_token(
    user_id: str,
    email: Optional[str] = None,
    tenant_id: Optional[str] = None,
    roles: Optional[list] = None,
    permissions: Optional[list] = None,
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create a JWT access token.

    Args:
        user_id: User identifier
        email: User email
        tenant_id: Tenant identifier
        roles: User roles
        permissions: User permissions
        expires_delta: Token expiration time

    Returns:
        Encoded JWT token
    """
    expire = datetime.now() + (expires_delta or timedelta(hours=JWT_EXPIRATION_HOURS))

    payload = {
        "sub": user_id,
        "email": email,
        "tenant_id": tenant_id,
        "roles": roles or [],
        "permissions": permissions or [],
        "exp": expire.timestamp(),
        "iat": datetime.now().timestamp()
    }

    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)