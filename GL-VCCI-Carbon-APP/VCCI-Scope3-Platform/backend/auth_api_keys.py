"""
API Key Authentication System
GL-VCCI Scope 3 Platform

Service-to-service authentication using API keys with:
- Secure key generation (vcci_<env>_<random_32_chars>)
- Hashed key storage (bcrypt)
- Scoped permissions (read, write, admin)
- Rate limiting per key
- Key rotation support
- Audit logging

Version: 1.0.0
Security Enhancement: 2025-11-09
"""

import os
import secrets
import logging
from typing import Optional, List
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

from fastapi import HTTPException, status, Security, Request
from fastapi.security import APIKeyHeader
from passlib.hash import bcrypt
import redis.asyncio as redis

logger = logging.getLogger(__name__)

# Configuration
ENVIRONMENT = os.getenv("ENVIRONMENT", "dev")
API_KEY_PREFIX = f"vcci_{ENVIRONMENT}_"

# Redis configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")

# Rate limiting (requests per hour per API key)
DEFAULT_RATE_LIMIT = int(os.getenv("API_KEY_RATE_LIMIT", "1000"))

# Security scheme
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Redis client (lazy initialization)
_redis_client: Optional[redis.Redis] = None


class APIKeyScope(str, Enum):
    """API key permission scopes."""

    READ = "read"  # Read-only access
    WRITE = "write"  # Read and write access
    ADMIN = "admin"  # Full administrative access
    CALCULATE = "calculate"  # Calculation endpoints only
    REPORT = "report"  # Reporting endpoints only


@dataclass
class APIKeyData:
    """API key metadata."""

    key_id: str  # Unique identifier
    service_name: str  # Service using the key
    scopes: List[APIKeyScope]  # Permissions
    rate_limit: int = DEFAULT_RATE_LIMIT  # Requests per hour
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_used_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    is_active: bool = True
    description: Optional[str] = None
    created_by: Optional[str] = None
    ip_whitelist: Optional[List[str]] = None  # Optional IP restrictions

    def to_dict(self):
        """Convert to dictionary for storage."""
        return {
            "key_id": self.key_id,
            "service_name": self.service_name,
            "scopes": ",".join([s.value for s in self.scopes]),
            "rate_limit": str(self.rate_limit),
            "created_at": self.created_at.isoformat(),
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else "",
            "expires_at": self.expires_at.isoformat() if self.expires_at else "",
            "is_active": "1" if self.is_active else "0",
            "description": self.description or "",
            "created_by": self.created_by or "",
            "ip_whitelist": ",".join(self.ip_whitelist) if self.ip_whitelist else "",
        }

    @classmethod
    def from_dict(cls, data: dict):
        """Create from dictionary retrieved from storage."""
        return cls(
            key_id=data["key_id"],
            service_name=data["service_name"],
            scopes=[APIKeyScope(s) for s in data["scopes"].split(",") if s],
            rate_limit=int(data["rate_limit"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_used_at=datetime.fromisoformat(data["last_used_at"]) if data.get("last_used_at") else None,
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            is_active=data.get("is_active") == "1",
            description=data.get("description") or None,
            created_by=data.get("created_by") or None,
            ip_whitelist=data.get("ip_whitelist").split(",") if data.get("ip_whitelist") else None,
        )


class APIKeyError(HTTPException):
    """API key authentication error."""

    def __init__(self, detail: str = "Invalid API key"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "API-Key"},
        )


async def get_redis_client() -> redis.Redis:
    """
    Get or create Redis client for API key storage.

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
            logger.info("Connected to Redis for API key storage")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            raise

    return _redis_client


def generate_api_key() -> tuple[str, str]:
    """
    Generate a new API key.

    Returns:
        Tuple of (api_key, key_id)
        - api_key: Full key to give to service (vcci_<env>_<random>)
        - key_id: Unique identifier for the key

    Example:
        >>> api_key, key_id = generate_api_key()
        >>> print(api_key)  # vcci_prod_Xj2kL9m...
    """
    key_id = secrets.token_urlsafe(16)  # Short ID
    random_part = secrets.token_urlsafe(32)  # 43 chars
    api_key = f"{API_KEY_PREFIX}{random_part}"

    return api_key, key_id


def hash_api_key(api_key: str) -> str:
    """
    Hash an API key for secure storage.

    Uses bcrypt for secure one-way hashing.

    Args:
        api_key: Plain API key to hash

    Returns:
        Hashed API key

    Example:
        >>> hashed = hash_api_key(api_key)
    """
    return bcrypt.hash(api_key)


def verify_api_key_hash(api_key: str, hashed: str) -> bool:
    """
    Verify an API key against its hash.

    Args:
        api_key: Plain API key
        hashed: Hashed API key from storage

    Returns:
        True if matches, False otherwise

    Example:
        >>> if verify_api_key_hash(provided_key, stored_hash):
        ...     print("Valid key")
    """
    return bcrypt.verify(api_key, hashed)


async def create_api_key(
    service_name: str,
    scopes: List[APIKeyScope],
    rate_limit: int = DEFAULT_RATE_LIMIT,
    description: Optional[str] = None,
    created_by: Optional[str] = None,
    expires_at: Optional[datetime] = None,
    ip_whitelist: Optional[List[str]] = None,
) -> tuple[str, APIKeyData]:
    """
    Create a new API key.

    Args:
        service_name: Name of service using the key
        scopes: List of permission scopes
        rate_limit: Requests per hour limit
        description: Optional description
        created_by: Optional creator identifier
        expires_at: Optional expiration datetime
        ip_whitelist: Optional list of allowed IP addresses

    Returns:
        Tuple of (plain_api_key, api_key_data)
        NOTE: plain_api_key is only returned here - store it securely!

    Example:
        >>> api_key, data = await create_api_key(
        ...     "reporting-service",
        ...     [APIKeyScope.READ, APIKeyScope.REPORT],
        ...     description="Automated reporting system"
        ... )
        >>> print(f"API Key (save this!): {api_key}")
    """
    # Generate key
    api_key, key_id = generate_api_key()

    # Hash for storage
    hashed_key = hash_api_key(api_key)

    # Create metadata
    key_data = APIKeyData(
        key_id=key_id,
        service_name=service_name,
        scopes=scopes,
        rate_limit=rate_limit,
        description=description,
        created_by=created_by,
        expires_at=expires_at,
        ip_whitelist=ip_whitelist,
    )

    # Store in Redis
    redis_client = await get_redis_client()

    # Store hash
    hash_key = f"api_key:hash:{key_id}"
    await redis_client.set(hash_key, hashed_key)

    # Store metadata
    metadata_key = f"api_key:metadata:{key_id}"
    await redis_client.hset(metadata_key, mapping=key_data.to_dict())

    # Create index by service name for lookups
    index_key = f"api_key:index:service:{service_name}"
    await redis_client.sadd(index_key, key_id)

    logger.info(
        f"Created API key for service: {service_name}, "
        f"scopes: {[s.value for s in scopes]}, key_id: {key_id}"
    )

    return api_key, key_data


async def verify_api_key(api_key: str) -> Optional[APIKeyData]:
    """
    Verify an API key and return its metadata.

    Args:
        api_key: API key to verify

    Returns:
        APIKeyData if valid, None if invalid

    Example:
        >>> key_data = await verify_api_key(provided_api_key)
        >>> if key_data:
        ...     print(f"Valid key for: {key_data.service_name}")
    """
    try:
        redis_client = await get_redis_client()

        # Scan all keys to find matching hash
        # Note: In production, consider maintaining a key->key_id index
        pattern = "api_key:hash:*"

        async for hash_key in redis_client.scan_iter(match=pattern):
            key_id = hash_key.split(":")[-1]
            stored_hash = await redis_client.get(hash_key)

            if stored_hash and verify_api_key_hash(api_key, stored_hash):
                # Found matching key, get metadata
                metadata_key = f"api_key:metadata:{key_id}"
                metadata = await redis_client.hgetall(metadata_key)

                if not metadata:
                    logger.warning(f"Metadata not found for key_id: {key_id}")
                    return None

                key_data = APIKeyData.from_dict(metadata)

                # Check if active
                if not key_data.is_active:
                    logger.warning(f"Inactive API key used: {key_id}")
                    return None

                # Check expiration
                if key_data.expires_at and key_data.expires_at < datetime.utcnow():
                    logger.warning(f"Expired API key used: {key_id}")
                    return None

                # Update last used timestamp
                key_data.last_used_at = datetime.utcnow()
                await redis_client.hset(
                    metadata_key,
                    "last_used_at",
                    key_data.last_used_at.isoformat(),
                )

                logger.debug(f"API key verified: {key_id}")
                return key_data

        logger.warning("API key verification failed: no matching key found")
        return None

    except Exception as e:
        logger.error(f"Error verifying API key: {str(e)}")
        return None


async def check_rate_limit(key_data: APIKeyData) -> bool:
    """
    Check if API key has exceeded rate limit.

    Uses a sliding window counter in Redis.

    Args:
        key_data: API key metadata

    Returns:
        True if within limit, False if exceeded

    Example:
        >>> if not await check_rate_limit(key_data):
        ...     raise HTTPException(429, "Rate limit exceeded")
    """
    try:
        redis_client = await get_redis_client()

        # Rate limit key (per hour window)
        now = datetime.utcnow()
        hour_key = now.strftime("%Y-%m-%d-%H")
        rate_key = f"api_key:rate:{key_data.key_id}:{hour_key}"

        # Increment counter
        count = await redis_client.incr(rate_key)

        # Set expiration on first increment
        if count == 1:
            await redis_client.expire(rate_key, 3600)  # 1 hour

        # Check limit
        if count > key_data.rate_limit:
            logger.warning(
                f"Rate limit exceeded for API key {key_data.key_id}: "
                f"{count}/{key_data.rate_limit}"
            )
            return False

        return True

    except Exception as e:
        logger.error(f"Error checking rate limit: {str(e)}")
        # Fail open for rate limiting (don't block on error)
        return True


async def revoke_api_key(key_id: str) -> bool:
    """
    Revoke an API key (mark as inactive).

    Args:
        key_id: API key identifier

    Returns:
        True if revoked, False if not found

    Example:
        >>> await revoke_api_key("abc123")
    """
    try:
        redis_client = await get_redis_client()

        metadata_key = f"api_key:metadata:{key_id}"
        metadata = await redis_client.hgetall(metadata_key)

        if not metadata:
            return False

        # Mark as inactive
        await redis_client.hset(metadata_key, "is_active", "0")

        logger.info(f"Revoked API key: {key_id}")
        return True

    except Exception as e:
        logger.error(f"Error revoking API key: {str(e)}")
        return False


async def delete_api_key(key_id: str) -> bool:
    """
    Permanently delete an API key.

    Args:
        key_id: API key identifier

    Returns:
        True if deleted, False if not found

    Example:
        >>> await delete_api_key("abc123")
    """
    try:
        redis_client = await get_redis_client()

        # Get metadata to remove from index
        metadata_key = f"api_key:metadata:{key_id}"
        metadata = await redis_client.hgetall(metadata_key)

        if metadata:
            service_name = metadata.get("service_name")
            if service_name:
                index_key = f"api_key:index:service:{service_name}"
                await redis_client.srem(index_key, key_id)

        # Delete hash and metadata
        hash_key = f"api_key:hash:{key_id}"
        deleted_hash = await redis_client.delete(hash_key)
        deleted_metadata = await redis_client.delete(metadata_key)

        if deleted_hash or deleted_metadata:
            logger.info(f"Deleted API key: {key_id}")
            return True

        return False

    except Exception as e:
        logger.error(f"Error deleting API key: {str(e)}")
        return False


async def list_service_keys(service_name: str) -> List[APIKeyData]:
    """
    List all API keys for a service.

    Args:
        service_name: Service name

    Returns:
        List of APIKeyData objects

    Example:
        >>> keys = await list_service_keys("reporting-service")
        >>> for key in keys:
        ...     print(f"{key.key_id}: {key.scopes}")
    """
    try:
        redis_client = await get_redis_client()

        index_key = f"api_key:index:service:{service_name}"
        key_ids = await redis_client.smembers(index_key)

        keys = []
        for key_id in key_ids:
            metadata_key = f"api_key:metadata:{key_id}"
            metadata = await redis_client.hgetall(metadata_key)

            if metadata:
                key_data = APIKeyData.from_dict(metadata)
                keys.append(key_data)

        return keys

    except Exception as e:
        logger.error(f"Error listing service keys: {str(e)}")
        return []


# FastAPI dependency for API key authentication
async def require_api_key(
    request: Request,
    api_key: Optional[str] = Security(api_key_header),
    required_scopes: Optional[List[APIKeyScope]] = None,
) -> APIKeyData:
    """
    FastAPI dependency to require API key authentication.

    Args:
        request: FastAPI request object
        api_key: API key from header
        required_scopes: Optional list of required scopes

    Returns:
        APIKeyData if authenticated

    Raises:
        APIKeyError: If authentication fails

    Example:
        ```python
        @app.get("/api/data")
        async def get_data(
            key_data: APIKeyData = Depends(require_api_key)
        ):
            return {"service": key_data.service_name}
        ```
    """
    if not api_key:
        raise APIKeyError("API key required")

    # Verify key
    key_data = await verify_api_key(api_key)

    if not key_data:
        raise APIKeyError("Invalid API key")

    # Check IP whitelist if configured
    if key_data.ip_whitelist:
        client_ip = request.client.host if request.client else None

        if client_ip not in key_data.ip_whitelist:
            logger.warning(
                f"API key {key_data.key_id} used from unauthorized IP: {client_ip}"
            )
            raise APIKeyError("API key not authorized from this IP")

    # Check scopes
    if required_scopes:
        if not any(scope in key_data.scopes for scope in required_scopes):
            raise APIKeyError(
                f"API key missing required scopes: {[s.value for s in required_scopes]}"
            )

    # Check rate limit
    if not await check_rate_limit(key_data):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: {key_data.rate_limit} requests per hour",
        )

    return key_data


# Helper function to create scope-specific dependencies
def require_scopes(*scopes: APIKeyScope):
    """
    Create a dependency that requires specific scopes.

    Args:
        scopes: Required scopes

    Returns:
        FastAPI dependency

    Example:
        ```python
        @app.post("/api/write")
        async def write_data(
            key_data: APIKeyData = Depends(require_scopes(APIKeyScope.WRITE))
        ):
            return {"status": "ok"}
        ```
    """

    async def _require_scopes(
        request: Request,
        api_key: Optional[str] = Security(api_key_header),
    ) -> APIKeyData:
        return await require_api_key(request, api_key, list(scopes))

    return _require_scopes


__all__ = [
    "APIKeyScope",
    "APIKeyData",
    "APIKeyError",
    "generate_api_key",
    "create_api_key",
    "verify_api_key",
    "check_rate_limit",
    "revoke_api_key",
    "delete_api_key",
    "list_service_keys",
    "require_api_key",
    "require_scopes",
    "get_redis_client",
]
