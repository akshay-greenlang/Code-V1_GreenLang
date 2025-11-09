"""
JWT Refresh Token Management System
GL-VCCI Scope 3 Platform

Implements secure refresh token mechanism with:
- Access tokens: 1 hour expiration
- Refresh tokens: 7 days expiration
- Token rotation on refresh
- Redis-based token storage
- Token blacklisting support

MIGRATION (2025-11-09): Migrated to use greenlang.auth.AuthManager

Version: 2.0.0
Security Enhancement: 2025-11-09
"""

import os
import secrets
import logging
from typing import Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from fastapi import HTTPException, status
from greenlang.auth import AuthManager, AuthToken
import redis.asyncio as redis

# Import auth manager from main auth module
from backend.auth import get_auth_manager

logger = logging.getLogger(__name__)

# Configuration
ACCESS_TOKEN_EXPIRE_SECONDS = int(os.getenv("ACCESS_TOKEN_EXPIRE_SECONDS", "3600"))  # 1 hour
REFRESH_TOKEN_EXPIRE_SECONDS = int(os.getenv("REFRESH_TOKEN_EXPIRE_SECONDS", "604800"))  # 7 days
REFRESH_TOKEN_ROTATION = os.getenv("REFRESH_TOKEN_ROTATION", "true").lower() == "true"

# Redis configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")

# Redis client (lazy initialization)
_redis_client: Optional[redis.Redis] = None


@dataclass
class TokenPair:
    """Pair of access and refresh tokens."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = ACCESS_TOKEN_EXPIRE_SECONDS

    def to_dict(self):
        """Convert to dictionary for API response."""
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "token_type": self.token_type,
            "expires_in": self.expires_in,
        }


@dataclass
class RefreshTokenData:
    """Refresh token metadata stored in Redis."""

    user_id: str
    jti: str  # JWT ID (unique token identifier)
    issued_at: datetime
    expires_at: datetime
    device_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

    def to_dict(self):
        """Convert to dictionary for Redis storage."""
        return {
            "user_id": self.user_id,
            "jti": self.jti,
            "issued_at": self.issued_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "device_id": self.device_id,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
        }

    @classmethod
    def from_dict(cls, data: dict):
        """Create from dictionary retrieved from Redis."""
        return cls(
            user_id=data["user_id"],
            jti=data["jti"],
            issued_at=datetime.fromisoformat(data["issued_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]),
            device_id=data.get("device_id"),
            ip_address=data.get("ip_address"),
            user_agent=data.get("user_agent"),
        )


class RefreshTokenError(HTTPException):
    """Refresh token validation error."""

    def __init__(self, detail: str = "Invalid or expired refresh token"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_redis_client() -> redis.Redis:
    """
    Get or create Redis client for token storage.

    Returns:
        Redis client instance
    """
    global _redis_client

    if _redis_client is None:
        _redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            password=REDIS_PASSWORD,
            decode_responses=True,
        )

        # Test connection
        try:
            await _redis_client.ping()
            logger.info("Connected to Redis for token storage")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            raise

    return _redis_client


async def close_redis_client():
    """Close Redis connection."""
    global _redis_client

    if _redis_client:
        await _redis_client.close()
        _redis_client = None
        logger.info("Closed Redis connection")


def create_access_token(
    user_id: str,
    additional_claims: Optional[dict] = None,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Create an access token using GreenLang AuthManager.

    Args:
        user_id: User identifier (subject)
        additional_claims: Optional additional claims to include
        expires_delta: Optional custom expiration time

    Returns:
        Access token value

    Example:
        >>> token = create_access_token("user@example.com")
        >>> token = create_access_token("user@example.com", {"role": "admin"})
    """
    auth_mgr = get_auth_manager()

    # Extract tenant and other data
    tenant_id = additional_claims.get("tenant_id", "default") if additional_claims else "default"

    # Calculate expiry
    expires_in = None
    if expires_delta:
        expires_in = int(expires_delta.total_seconds())
    else:
        expires_in = ACCESS_TOKEN_EXPIRE_SECONDS

    # Create token
    auth_token = auth_mgr.create_token(
        tenant_id=tenant_id,
        user_id=user_id,
        name=f"Access token for {user_id}",
        token_type="access",
        expires_in=expires_in,
        scopes=additional_claims.get("scopes", []) if additional_claims else [],
        roles=additional_claims.get("roles", []) if additional_claims else [],
    )

    return auth_token.token_value


def create_refresh_token(
    user_id: str,
    device_id: Optional[str] = None,
    expires_delta: Optional[timedelta] = None,
) -> Tuple[str, str]:
    """
    Create a refresh token using GreenLang AuthManager.

    Args:
        user_id: User identifier (subject)
        device_id: Optional device identifier for multi-device support
        expires_delta: Optional custom expiration time

    Returns:
        Tuple of (refresh token value, token ID)

    Example:
        >>> token, jti = create_refresh_token("user@example.com")
    """
    auth_mgr = get_auth_manager()

    # Calculate expiry
    expires_in = None
    if expires_delta:
        expires_in = int(expires_delta.total_seconds())
    else:
        expires_in = REFRESH_TOKEN_EXPIRE_SECONDS

    # Create refresh token
    auth_token = auth_mgr.create_token(
        tenant_id="default",
        user_id=user_id,
        name=f"Refresh token for {user_id}",
        token_type="refresh",
        expires_in=expires_in,
    )

    return auth_token.token_value, auth_token.token_id


async def issue_token_pair(
    user_id: str,
    device_id: Optional[str] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    additional_access_claims: Optional[dict] = None,
) -> TokenPair:
    """
    Issue a new access token and refresh token pair.

    This is used during login to provide both tokens to the client.
    The refresh token is stored in Redis for validation and rotation.

    Args:
        user_id: User identifier
        device_id: Optional device identifier
        ip_address: Optional client IP address (for audit)
        user_agent: Optional client user agent (for audit)
        additional_access_claims: Optional additional claims for access token

    Returns:
        TokenPair containing access and refresh tokens

    Example:
        >>> tokens = await issue_token_pair(
        ...     "user@example.com",
        ...     device_id="mobile-app",
        ...     ip_address="192.168.1.1",
        ... )
    """
    # Create access token
    access_token = create_access_token(user_id, additional_access_claims)

    # Create refresh token
    refresh_token, jti = create_refresh_token(user_id, device_id)

    # Store refresh token metadata in Redis
    redis_client = await get_redis_client()

    token_data = RefreshTokenData(
        user_id=user_id,
        jti=jti,
        issued_at=datetime.utcnow(),
        expires_at=datetime.utcnow() + timedelta(seconds=REFRESH_TOKEN_EXPIRE_SECONDS),
        device_id=device_id,
        ip_address=ip_address,
        user_agent=user_agent,
    )

    # Store in Redis with TTL
    key = f"refresh_token:{jti}"
    await redis_client.hset(key, mapping=token_data.to_dict())
    await redis_client.expire(key, REFRESH_TOKEN_EXPIRE_SECONDS)

    logger.info(
        f"Issued token pair for user: {user_id}, "
        f"device: {device_id}, jti: {jti}"
    )

    return TokenPair(
        access_token=access_token,
        refresh_token=refresh_token,
    )


async def refresh_access_token(
    refresh_token: str,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
) -> TokenPair:
    """
    Exchange a refresh token for a new access token.

    Optionally rotates the refresh token (invalidates old, issues new).

    Args:
        refresh_token: Valid refresh token
        ip_address: Optional client IP address (for audit)
        user_agent: Optional client user agent (for audit)

    Returns:
        TokenPair with new access token (and new refresh token if rotation enabled)

    Raises:
        RefreshTokenError: If refresh token is invalid or expired

    Example:
        >>> new_tokens = await refresh_access_token(old_refresh_token)
    """
    try:
        # Decode refresh token
        payload = jwt.decode(refresh_token, REFRESH_SECRET, algorithms=[JWT_ALGORITHM])

        # Validate token type
        if payload.get("type") != "refresh":
            raise RefreshTokenError("Invalid token type")

        user_id = payload.get("sub")
        jti = payload.get("jti")
        device_id = payload.get("device_id")

        if not user_id or not jti:
            raise RefreshTokenError("Invalid token claims")

        # Verify token exists in Redis (not revoked)
        redis_client = await get_redis_client()
        key = f"refresh_token:{jti}"

        token_data_dict = await redis_client.hgetall(key)
        if not token_data_dict:
            raise RefreshTokenError("Refresh token not found or revoked")

        # Parse token data
        token_data = RefreshTokenData.from_dict(token_data_dict)

        # Verify user_id matches
        if token_data.user_id != user_id:
            logger.warning(f"User ID mismatch in refresh token: {jti}")
            raise RefreshTokenError("Invalid token")

        # Create new access token
        access_token = create_access_token(user_id)

        # Token rotation: issue new refresh token and invalidate old one
        if REFRESH_TOKEN_ROTATION:
            # Delete old refresh token
            await redis_client.delete(key)

            # Issue new token pair
            new_tokens = await issue_token_pair(
                user_id=user_id,
                device_id=device_id,
                ip_address=ip_address,
                user_agent=user_agent,
            )

            logger.info(
                f"Rotated refresh token for user: {user_id}, "
                f"old_jti: {jti}, new_jti: {new_tokens.refresh_token[-8:]}"
            )

            return new_tokens

        else:
            # Just return new access token with existing refresh token
            logger.info(f"Refreshed access token for user: {user_id}, jti: {jti}")

            return TokenPair(
                access_token=access_token,
                refresh_token=refresh_token,
            )

    except JWTError as e:
        logger.warning(f"JWT validation failed during refresh: {str(e)}")
        raise RefreshTokenError("Invalid or expired refresh token")

    except RefreshTokenError:
        raise

    except Exception as e:
        logger.error(f"Unexpected error during token refresh: {str(e)}")
        raise RefreshTokenError("Token refresh failed")


async def revoke_refresh_token(refresh_token: str) -> bool:
    """
    Revoke a refresh token (used during logout).

    Args:
        refresh_token: Refresh token to revoke

    Returns:
        True if token was revoked, False if not found

    Example:
        >>> await revoke_refresh_token(user_refresh_token)
    """
    try:
        # Decode token to get JTI
        payload = jwt.decode(refresh_token, REFRESH_SECRET, algorithms=[JWT_ALGORITHM])
        jti = payload.get("jti")

        if not jti:
            return False

        # Delete from Redis
        redis_client = await get_redis_client()
        key = f"refresh_token:{jti}"
        deleted = await redis_client.delete(key)

        if deleted:
            logger.info(f"Revoked refresh token: {jti}")
            return True

        return False

    except JWTError:
        return False

    except Exception as e:
        logger.error(f"Error revoking refresh token: {str(e)}")
        return False


async def revoke_all_user_tokens(user_id: str) -> int:
    """
    Revoke all refresh tokens for a user.

    Used when:
    - User changes password
    - User reports account compromise
    - Admin force logout

    Args:
        user_id: User identifier

    Returns:
        Number of tokens revoked

    Example:
        >>> count = await revoke_all_user_tokens("user@example.com")
    """
    redis_client = await get_redis_client()

    # Scan for all refresh tokens
    pattern = "refresh_token:*"
    revoked_count = 0

    async for key in redis_client.scan_iter(match=pattern):
        token_data_dict = await redis_client.hgetall(key)

        if token_data_dict and token_data_dict.get("user_id") == user_id:
            await redis_client.delete(key)
            revoked_count += 1

    logger.info(f"Revoked {revoked_count} refresh tokens for user: {user_id}")
    return revoked_count


async def get_user_active_tokens(user_id: str) -> list:
    """
    Get all active refresh tokens for a user.

    Useful for showing user their active sessions/devices.

    Args:
        user_id: User identifier

    Returns:
        List of RefreshTokenData objects

    Example:
        >>> active_tokens = await get_user_active_tokens("user@example.com")
        >>> for token in active_tokens:
        ...     print(f"Device: {token.device_id}, Issued: {token.issued_at}")
    """
    redis_client = await get_redis_client()

    pattern = "refresh_token:*"
    active_tokens = []

    async for key in redis_client.scan_iter(match=pattern):
        token_data_dict = await redis_client.hgetall(key)

        if token_data_dict and token_data_dict.get("user_id") == user_id:
            token_data = RefreshTokenData.from_dict(token_data_dict)
            active_tokens.append(token_data)

    # Sort by issued_at (newest first)
    active_tokens.sort(key=lambda x: x.issued_at, reverse=True)

    return active_tokens


# Validate configuration on module import
def validate_refresh_config():
    """Validate refresh token configuration."""
    if not JWT_SECRET:
        raise ValueError("JWT_SECRET is required for refresh token system")

    if not REFRESH_SECRET:
        logger.warning(
            "REFRESH_SECRET not set, using JWT_SECRET. "
            "Consider using separate secrets for access and refresh tokens."
        )

    logger.info(
        f"Refresh token system configured: "
        f"access_ttl={ACCESS_TOKEN_EXPIRE_SECONDS}s, "
        f"refresh_ttl={REFRESH_TOKEN_EXPIRE_SECONDS}s, "
        f"rotation={REFRESH_TOKEN_ROTATION}"
    )


# Call validation on import
validate_refresh_config()


__all__ = [
    "TokenPair",
    "RefreshTokenData",
    "RefreshTokenError",
    "create_access_token",
    "create_refresh_token",
    "issue_token_pair",
    "refresh_access_token",
    "revoke_refresh_token",
    "revoke_all_user_tokens",
    "get_user_active_tokens",
    "get_redis_client",
    "close_redis_client",
]
