"""
Authentication and Authorization Middleware
GL-VCCI Scope 3 Platform

Provides JWT-based authentication for API endpoints.

SECURITY FIX (CRIT-003): Implements authentication middleware to protect all API endpoints.

Version: 2.0.0
Security Update: 2025-11-08
"""

import os
import logging
from typing import Optional
from datetime import datetime, timedelta

from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt

logger = logging.getLogger(__name__)

# Security scheme
security = HTTPBearer()

# JWT configuration from environment
JWT_SECRET = os.getenv("JWT_SECRET")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRATION_SECONDS = int(os.getenv("JWT_EXPIRATION_SECONDS", "3600"))


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
    if not JWT_SECRET:
        raise ValueError(
            "JWT_SECRET environment variable is required. "
            "Generate a strong secret: python -c 'import secrets; print(secrets.token_urlsafe(32))'"
        )

    if len(JWT_SECRET) < 32:
        raise ValueError(
            "JWT_SECRET must be at least 32 characters long for security. "
            "Current length: {len(JWT_SECRET)}"
        )

    logger.info(f"JWT authentication configured (algorithm: {JWT_ALGORITHM})")


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token.

    Args:
        data: Payload data to encode in the token
        expires_delta: Optional custom expiration time

    Returns:
        Encoded JWT token

    Example:
        >>> token = create_access_token({"sub": "user@example.com"})
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(seconds=JWT_EXPIRATION_SECONDS)

    to_encode.update({"exp": expire})

    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str) -> dict:
    """
    Decode and validate a JWT access token.

    Args:
        token: JWT token to decode

    Returns:
        Decoded payload

    Raises:
        AuthenticationError: If token is invalid or expired
    """
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload

    except JWTError as e:
        logger.warning(f"JWT validation failed: {str(e)}")
        raise AuthenticationError(detail="Invalid or expired token")


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
