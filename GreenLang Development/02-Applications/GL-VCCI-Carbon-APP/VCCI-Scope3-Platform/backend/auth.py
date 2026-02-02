# -*- coding: utf-8 -*-
"""
Authentication and Authorization Middleware
GL-VCCI Scope 3 Platform

Provides JWT-based authentication for API endpoints.

SECURITY FIX (CRIT-003): Implements authentication middleware to protect all API endpoints.
MIGRATION (2025-11-09): Migrated from jose to greenlang.auth.AuthManager

Version: 3.0.0
Security Update: 2025-11-09
"""

import os
import logging
from typing import Optional
from datetime import datetime, timedelta

from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from greenlang.auth import AuthManager, AuthToken

logger = logging.getLogger(__name__)

# Security scheme
security = HTTPBearer()

# Initialize GreenLang Auth Manager
_auth_manager = None

def get_auth_manager() -> AuthManager:
    """Get or create global auth manager instance."""
    global _auth_manager
    if _auth_manager is None:
        config = {
            "secret_key": os.getenv("JWT_SECRET"),
            "token_expiry": int(os.getenv("JWT_EXPIRATION_SECONDS", "3600")),
        }
        _auth_manager = AuthManager(config=config)
        logger.info("GreenLang AuthManager initialized")
    return _auth_manager


class AuthenticationError(HTTPException):
    """Custom authentication error."""

    def __init__(self, detail: str = "Could not validate credentials"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"},
        )


def validate_jwt_config():
    """
    Validate JWT configuration on startup.

    Raises:
        ValueError: If JWT configuration is invalid
    """
    jwt_secret = os.getenv("JWT_SECRET")
    if not jwt_secret:
        raise ValueError(
            "JWT_SECRET environment variable is required. "
            "Generate a strong secret: python -c 'import secrets; print(secrets.token_urlsafe(32))'"
        )

    if len(jwt_secret) < 32:
        raise ValueError(
            f"JWT_SECRET must be at least 32 characters long for security. "
            f"Current length: {len(jwt_secret)}"
        )

    # Initialize auth manager to validate configuration
    auth_mgr = get_auth_manager()
    logger.info(f"GreenLang auth configured (token_expiry: {auth_mgr.token_expiry}s)")


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token using GreenLang AuthManager.

    Args:
        data: Payload data to encode in the token
        expires_delta: Optional custom expiration time

    Returns:
        Token value string

    Example:
        >>> token = create_access_token({"sub": "user@example.com"})
    """
    auth_mgr = get_auth_manager()

    # Extract user info from data
    user_id = data.get("sub", "")
    tenant_id = data.get("tenant_id", "default")

    # Calculate expiry
    expires_in = None
    if expires_delta:
        expires_in = int(expires_delta.total_seconds())

    # Create token using AuthManager
    auth_token = auth_mgr.create_token(
        tenant_id=tenant_id,
        user_id=user_id,
        name=f"Access token for {user_id}",
        token_type="bearer",
        expires_in=expires_in,
        # Store additional claims as metadata
        scopes=data.get("scopes", []),
        roles=data.get("roles", []),
    )

    return auth_token.token_value


def decode_access_token(token: str) -> dict:
    """
    Decode and validate a JWT access token using GreenLang AuthManager.

    Args:
        token: Token value to decode

    Returns:
        Decoded payload as dict

    Raises:
        AuthenticationError: If token is invalid or expired
    """
    auth_mgr = get_auth_manager()

    # Validate token
    auth_token = auth_mgr.validate_token(token)

    if not auth_token:
        logger.warning(f"Token validation failed: token not found or invalid")
        raise AuthenticationError(detail="Invalid or expired token")

    # Convert AuthToken to dict payload (compatible with old format)
    payload = {
        "sub": auth_token.user_id,
        "tenant_id": auth_token.tenant_id,
        "token_id": auth_token.token_id,
        "token_type": auth_token.token_type,
        "scopes": auth_token.scopes,
        "roles": auth_token.roles,
        "exp": auth_token.expires_at.timestamp() if auth_token.expires_at else None,
        "iat": auth_token.created_at.timestamp() if auth_token.created_at else None,
    }

    return payload


async def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    """
    Verify JWT token from Authorization header.

    This dependency can be used to protect individual endpoints.

    Args:
        credentials: HTTP Bearer credentials from request

    Returns:
        Decoded token payload

    Raises:
        AuthenticationError: If authentication fails

    Example:
        ```python
        @app.get("/protected")
        async def protected_endpoint(token: dict = Depends(verify_token)):
            user_id = token.get("sub")
            return {"message": f"Hello {user_id}"}
        ```
    """
    token = credentials.credentials

    try:
        payload = decode_access_token(token)

        # Extract user identifier
        user_id: str = payload.get("sub")
        if user_id is None:
            raise AuthenticationError(detail="Token missing 'sub' claim")

        # Log successful authentication
        logger.debug(f"Authenticated request from user: {user_id}")

        return payload

    except AuthenticationError:
        raise

    except Exception as e:
        logger.error(f"Unexpected authentication error: {str(e)}")
        raise AuthenticationError()


async def verify_token_optional(
    request: Request,
) -> Optional[dict]:
    """
    Optional token verification.

    Returns None if no token is provided, validates if present.

    Args:
        request: FastAPI request object

    Returns:
        Decoded token payload or None
    """
    auth_header = request.headers.get("Authorization")

    if not auth_header:
        return None

    if not auth_header.startswith("Bearer "):
        return None

    token = auth_header.replace("Bearer ", "")

    try:
        return decode_access_token(token)
    except AuthenticationError:
        return None


def get_current_user(token: dict = Depends(verify_token)) -> str:
    """
    Extract current user ID from verified token.

    Args:
        token: Verified JWT token payload

    Returns:
        User identifier

    Example:
        ```python
        @app.get("/me")
        async def get_user_info(user_id: str = Depends(get_current_user)):
            return {"user_id": user_id}
        ```
    """
    return token.get("sub")


# For backward compatibility and clarity
async def require_auth(token: dict = Depends(verify_token)) -> dict:
    """
    Require authentication (alias for verify_token).

    Use this as a dependency to protect endpoints.

    Args:
        token: Verified JWT token payload

    Returns:
        Decoded token payload
    """
    return token


__all__ = [
    "security",
    "validate_jwt_config",
    "create_access_token",
    "decode_access_token",
    "verify_token",
    "verify_token_optional",
    "get_current_user",
    "require_auth",
    "AuthenticationError",
]
