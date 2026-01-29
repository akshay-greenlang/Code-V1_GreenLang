"""
Authentication and authorization for Review Console API.

This module provides JWT-based authentication with role-based access control
for the Review Console API.

Features:
    - JWT token validation and generation
    - User model with roles
    - FastAPI dependency for current user
    - Role-based access control decorators

Example:
    >>> from review_console.api.auth import get_current_user, User
    >>> @app.get("/protected")
    ... async def protected_route(user: User = Depends(get_current_user)):
    ...     return {"user": user.email}
"""

from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from pydantic import BaseModel, Field
import structlog

from review_console.config import get_settings

logger = structlog.get_logger()
settings = get_settings()

# HTTP Bearer security scheme
security = HTTPBearer(
    scheme_name="JWT",
    description="JWT Bearer token authentication",
    auto_error=True,
)


class Role(str, Enum):
    """
    User roles for access control.

    Attributes:
        ADMIN: Full access to all resources
        REVIEWER: Can review and resolve items
        VIEWER: Read-only access
        API: Service account for API access
    """
    ADMIN = "admin"
    REVIEWER = "reviewer"
    VIEWER = "viewer"
    API = "api"


class User(BaseModel):
    """
    Authenticated user model.

    Represents the current authenticated user extracted from the JWT token.

    Attributes:
        id: Unique user identifier
        email: User email address
        name: User display name
        roles: List of user roles
        org_id: Organization ID (for tenant isolation)
        metadata: Additional user metadata
    """
    id: str = Field(..., description="Unique user identifier")
    email: str = Field(..., description="User email address")
    name: Optional[str] = Field(None, description="User display name")
    roles: List[Role] = Field(default_factory=list, description="User roles")
    org_id: Optional[str] = Field(None, description="Organization ID")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def has_role(self, role: Role) -> bool:
        """Check if user has a specific role."""
        return role in self.roles or Role.ADMIN in self.roles

    def can_review(self) -> bool:
        """Check if user can review items."""
        return self.has_role(Role.REVIEWER) or self.has_role(Role.ADMIN)

    def can_admin(self) -> bool:
        """Check if user has admin access."""
        return self.has_role(Role.ADMIN)


class TokenData(BaseModel):
    """
    JWT token payload data.

    Attributes:
        sub: Subject (user ID)
        email: User email
        name: User name
        roles: User roles
        org_id: Organization ID
        exp: Expiration timestamp
        iat: Issued at timestamp
    """
    sub: str
    email: str
    name: Optional[str] = None
    roles: List[str] = Field(default_factory=list)
    org_id: Optional[str] = None
    exp: Optional[datetime] = None
    iat: Optional[datetime] = None


def create_access_token(
    user_id: str,
    email: str,
    name: Optional[str] = None,
    roles: Optional[List[str]] = None,
    org_id: Optional[str] = None,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Create a JWT access token.

    Args:
        user_id: Unique user identifier
        email: User email address
        name: User display name
        roles: List of user roles
        org_id: Organization ID
        expires_delta: Token expiration time delta

    Returns:
        Encoded JWT token string

    Example:
        >>> token = create_access_token(
        ...     user_id="user-123",
        ...     email="user@example.com",
        ...     roles=["reviewer"]
        ... )
    """
    now = datetime.now(timezone.utc)

    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=settings.access_token_expire_minutes)

    payload = {
        "sub": user_id,
        "email": email,
        "name": name,
        "roles": roles or [],
        "org_id": org_id,
        "exp": expire,
        "iat": now,
    }

    encoded_jwt = jwt.encode(
        payload,
        settings.secret_key,
        algorithm=settings.algorithm,
    )

    return encoded_jwt


def decode_token(token: str) -> TokenData:
    """
    Decode and validate a JWT token.

    Args:
        token: JWT token string

    Returns:
        Decoded token data

    Raises:
        HTTPException: If token is invalid or expired
    """
    try:
        payload = jwt.decode(
            token,
            settings.secret_key,
            algorithms=[settings.algorithm],
        )

        return TokenData(
            sub=payload.get("sub", ""),
            email=payload.get("email", ""),
            name=payload.get("name"),
            roles=payload.get("roles", []),
            org_id=payload.get("org_id"),
            exp=datetime.fromtimestamp(payload.get("exp", 0), tz=timezone.utc),
            iat=datetime.fromtimestamp(payload.get("iat", 0), tz=timezone.utc),
        )

    except JWTError as e:
        logger.warning("JWT decode error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "INVALID_TOKEN", "message": "Could not validate credentials"},
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> User:
    """
    FastAPI dependency to get the current authenticated user.

    Extracts and validates the JWT token from the Authorization header,
    then returns the corresponding User object.

    Args:
        credentials: HTTP Bearer credentials from header

    Returns:
        Authenticated User object

    Raises:
        HTTPException: 401 if token is invalid or expired

    Example:
        >>> @app.get("/me")
        ... async def get_me(user: User = Depends(get_current_user)):
        ...     return {"email": user.email}
    """
    token_data = decode_token(credentials.credentials)

    # Validate required fields
    if not token_data.sub or not token_data.email:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "INVALID_TOKEN", "message": "Token missing required fields"},
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Parse roles
    roles = []
    for role_str in token_data.roles:
        try:
            roles.append(Role(role_str))
        except ValueError:
            # Skip invalid roles
            logger.warning("Unknown role in token", role=role_str)

    return User(
        id=token_data.sub,
        email=token_data.email,
        name=token_data.name,
        roles=roles,
        org_id=token_data.org_id,
    )


async def get_current_reviewer(
    user: User = Depends(get_current_user),
) -> User:
    """
    FastAPI dependency to get the current user if they have reviewer access.

    Args:
        user: Current authenticated user

    Returns:
        User with reviewer access

    Raises:
        HTTPException: 403 if user is not a reviewer

    Example:
        >>> @app.post("/resolve")
        ... async def resolve(user: User = Depends(get_current_reviewer)):
        ...     return {"reviewer": user.email}
    """
    if not user.can_review():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"error": "FORBIDDEN", "message": "Reviewer access required"},
        )
    return user


async def get_current_admin(
    user: User = Depends(get_current_user),
) -> User:
    """
    FastAPI dependency to get the current user if they have admin access.

    Args:
        user: Current authenticated user

    Returns:
        User with admin access

    Raises:
        HTTPException: 403 if user is not an admin

    Example:
        >>> @app.delete("/item/{id}")
        ... async def delete_item(user: User = Depends(get_current_admin)):
        ...     return {"admin": user.email}
    """
    if not user.can_admin():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"error": "FORBIDDEN", "message": "Admin access required"},
        )
    return user


def require_roles(*required_roles: Role):
    """
    Dependency factory for role-based access control.

    Creates a FastAPI dependency that requires the user to have at least
    one of the specified roles.

    Args:
        required_roles: Roles that grant access (user needs at least one)

    Returns:
        FastAPI dependency function

    Example:
        >>> @app.post("/special")
        ... async def special(user: User = Depends(require_roles(Role.ADMIN, Role.REVIEWER))):
        ...     return {"user": user.email}
    """
    async def role_checker(user: User = Depends(get_current_user)) -> User:
        for role in required_roles:
            if user.has_role(role):
                return user

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "FORBIDDEN",
                "message": f"Required roles: {[r.value for r in required_roles]}",
            },
        )

    return role_checker
