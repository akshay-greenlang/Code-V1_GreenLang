"""
Token Blacklist / Revocation System
GL-VCCI Scope 3 Platform

Implements Redis-based token blacklisting for:
- Logout (blacklist tokens)
- Password change (blacklist all user tokens)
- Security incidents (force logout)
- Token compromise detection

MIGRATION (2025-11-09): Migrated to use greenlang.auth.AuthManager for token operations

Version: 2.0.0
Security Enhancement: 2025-11-09
"""

import os
import logging
from typing import Optional
from datetime import datetime, timedelta

from greenlang.auth import AuthManager
import redis.asyncio as redis

# Import auth manager from main auth module
from backend.auth import get_auth_manager

logger = logging.getLogger(__name__)

# Redis configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")

# Redis client (lazy initialization)
_redis_client: Optional[redis.Redis] = None


async def get_redis_client() -> redis.Redis:
    """
    Get or create Redis client for blacklist storage.

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
            logger.info("Connected to Redis for token blacklist")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            raise

    return _redis_client


async def blacklist_token(
    token: str,
    reason: str = "logout",
    metadata: Optional[dict] = None,
) -> bool:
    """
    Add a token to the blacklist using GreenLang AuthManager.

    The token is revoked in AuthManager and stored in Redis for distributed tracking.

    Args:
        token: Token value to blacklist
        reason: Reason for blacklisting (logout, password_change, security, etc.)
        metadata: Optional metadata (IP, user_agent, etc.)

    Returns:
        True if successfully blacklisted, False otherwise

    Example:
        >>> await blacklist_token(
        ...     access_token,
        ...     reason="logout",
        ...     metadata={"ip": "192.168.1.1"}
        ... )
    """
    try:
        auth_mgr = get_auth_manager()

        # Get token to extract metadata
        auth_token = auth_mgr.validate_token(token)

        if not auth_token:
            logger.warning("Token not found for blacklisting")
            return False

        user_id = auth_token.user_id
        jti = auth_token.token_id

        # Revoke token in AuthManager
        auth_mgr.revoke_token(token, by="system", reason=reason)

        # Calculate TTL (time until token expires)
        now = datetime.utcnow()
        exp_datetime = auth_token.expires_at

        if not exp_datetime:
            # No expiration, use default TTL of 24 hours
            ttl_seconds = 86400
        elif exp_datetime <= now:
            # Token already expired
            logger.debug(f"Token already expired, skipping blacklist: {jti}")
            return True
        else:
            ttl_seconds = int((exp_datetime - now).total_seconds())

        # Store in Redis for distributed blacklist
        redis_client = await get_redis_client()
        key = f"blacklist:token:{jti}"

        blacklist_data = {
            "user_id": user_id,
            "reason": reason,
            "blacklisted_at": now.isoformat(),
            "expires_at": exp_datetime.isoformat() if exp_datetime else "",
        }

        if metadata:
            blacklist_data.update(metadata)

        await redis_client.hset(key, mapping=blacklist_data)
        await redis_client.expire(key, ttl_seconds)

        logger.info(
            f"Blacklisted token for user {user_id}: jti={jti}, "
            f"reason={reason}, ttl={ttl_seconds}s"
        )

        return True

    except Exception as e:
        logger.error(f"Error blacklisting token: {str(e)}")
        return False


async def is_blacklisted(token: str) -> bool:
    """
    Check if a token is blacklisted using GreenLang AuthManager.

    This should be called on every authenticated request to ensure
    revoked tokens are rejected.

    Args:
        token: Token value to check

    Returns:
        True if blacklisted, False otherwise

    Example:
        >>> if await is_blacklisted(access_token):
        ...     raise HTTPException(401, "Token has been revoked")
    """
    try:
        auth_mgr = get_auth_manager()

        # Check if token is valid in AuthManager
        auth_token = auth_mgr.tokens.get(token)

        if not auth_token:
            # Token not found - consider blacklisted
            return True

        if auth_token.revoked:
            # Token explicitly revoked
            return True

        # Also check Redis for distributed blacklist
        redis_client = await get_redis_client()
        key = f"blacklist:token:{auth_token.token_id}"

        exists = await redis_client.exists(key)
        return bool(exists)

    except Exception as e:
        logger.error(f"Error checking blacklist: {str(e)}")
        # Fail secure: treat as blacklisted if we can't check
        return True


async def blacklist_all_user_tokens(
    user_id: str,
    reason: str = "password_change",
) -> int:
    """
    Blacklist all tokens for a specific user.

    Used when:
    - User changes password
    - Account compromise detected
    - Admin force logout

    This scans all active tokens and blacklists those belonging to the user.

    Args:
        user_id: User identifier
        reason: Reason for blacklisting all tokens

    Returns:
        Number of tokens blacklisted

    Example:
        >>> count = await blacklist_all_user_tokens(
        ...     "user@example.com",
        ...     reason="password_change"
        ... )
        >>> print(f"Blacklisted {count} tokens")
    """
    redis_client = await get_redis_client()
    blacklisted_count = 0

    # Create a user-specific blacklist entry
    # This is more efficient than scanning all tokens
    key = f"blacklist:user:{user_id}"
    blacklist_data = {
        "reason": reason,
        "blacklisted_at": datetime.utcnow().isoformat(),
    }

    # Set with a long TTL (7 days - max refresh token lifetime)
    await redis_client.hset(key, mapping=blacklist_data)
    await redis_client.expire(key, 604800)  # 7 days

    logger.info(
        f"Created user-level blacklist for {user_id}: reason={reason}"
    )

    return 1  # Return 1 to indicate user was blacklisted


async def is_user_blacklisted(user_id: str) -> bool:
    """
    Check if all tokens for a user are blacklisted.

    This is a fast check that doesn't require decoding the token.

    Args:
        user_id: User identifier

    Returns:
        True if user is blacklisted, False otherwise

    Example:
        >>> if await is_user_blacklisted(token_data["sub"]):
        ...     raise HTTPException(401, "All user tokens revoked")
    """
    try:
        redis_client = await get_redis_client()
        key = f"blacklist:user:{user_id}"

        exists = await redis_client.exists(key)
        return bool(exists)

    except Exception as e:
        logger.error(f"Error checking user blacklist: {str(e)}")
        # Fail secure: treat as blacklisted if we can't check
        return True


async def remove_from_blacklist(token: str) -> bool:
    """
    Remove a token from the blacklist.

    This is rarely needed but can be used to restore a token
    that was mistakenly blacklisted.

    Args:
        token: JWT token to remove from blacklist

    Returns:
        True if removed, False otherwise

    Example:
        >>> await remove_from_blacklist(access_token)
    """
    try:
        # Decode token to get JTI
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        jti = payload.get("jti", token[-16:])

        # Remove from Redis
        redis_client = await get_redis_client()
        key = f"blacklist:token:{jti}"

        deleted = await redis_client.delete(key)

        if deleted:
            logger.info(f"Removed token from blacklist: {jti}")
            return True

        return False

    except JWTError as e:
        logger.warning(f"Failed to decode token for removal: {str(e)}")
        return False

    except Exception as e:
        logger.error(f"Error removing from blacklist: {str(e)}")
        return False


async def clear_user_blacklist(user_id: str) -> bool:
    """
    Remove user-level blacklist.

    Used to restore access after password reset or false alarm.

    Args:
        user_id: User identifier

    Returns:
        True if removed, False otherwise

    Example:
        >>> await clear_user_blacklist("user@example.com")
    """
    try:
        redis_client = await get_redis_client()
        key = f"blacklist:user:{user_id}"

        deleted = await redis_client.delete(key)

        if deleted:
            logger.info(f"Removed user blacklist: {user_id}")
            return True

        return False

    except Exception as e:
        logger.error(f"Error clearing user blacklist: {str(e)}")
        return False


async def get_blacklist_info(token: str) -> Optional[dict]:
    """
    Get blacklist information for a token.

    Returns the reason and metadata if the token is blacklisted.

    Args:
        token: JWT token to check

    Returns:
        Blacklist data dictionary or None if not blacklisted

    Example:
        >>> info = await get_blacklist_info(access_token)
        >>> if info:
        ...     print(f"Token blacklisted: {info['reason']}")
    """
    try:
        # Decode token to get JTI
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        jti = payload.get("jti", token[-16:])

        # Get from Redis
        redis_client = await get_redis_client()
        key = f"blacklist:token:{jti}"

        data = await redis_client.hgetall(key)

        if data:
            return dict(data)

        return None

    except JWTError:
        return None

    except Exception as e:
        logger.error(f"Error getting blacklist info: {str(e)}")
        return None


async def cleanup_expired_blacklist_entries() -> int:
    """
    Manual cleanup of expired blacklist entries.

    Note: This is usually not needed because Redis TTL handles cleanup
    automatically. This is provided for manual maintenance if needed.

    Returns:
        Number of entries cleaned up

    Example:
        >>> count = await cleanup_expired_blacklist_entries()
        >>> print(f"Cleaned up {count} expired entries")
    """
    redis_client = await get_redis_client()
    cleaned_count = 0

    # Scan for all blacklist entries
    pattern = "blacklist:token:*"

    async for key in redis_client.scan_iter(match=pattern):
        data = await redis_client.hgetall(key)

        if data and "expires_at" in data:
            expires_at = datetime.fromisoformat(data["expires_at"])

            if expires_at <= datetime.utcnow():
                await redis_client.delete(key)
                cleaned_count += 1

    if cleaned_count > 0:
        logger.info(f"Cleaned up {cleaned_count} expired blacklist entries")

    return cleaned_count


async def get_blacklist_stats() -> dict:
    """
    Get statistics about the blacklist.

    Returns:
        Dictionary with blacklist statistics

    Example:
        >>> stats = await get_blacklist_stats()
        >>> print(f"Active blacklisted tokens: {stats['total_tokens']}")
    """
    redis_client = await get_redis_client()

    token_count = 0
    user_count = 0

    # Count token blacklist entries
    pattern_tokens = "blacklist:token:*"
    async for _ in redis_client.scan_iter(match=pattern_tokens):
        token_count += 1

    # Count user blacklist entries
    pattern_users = "blacklist:user:*"
    async for _ in redis_client.scan_iter(match=pattern_users):
        user_count += 1

    return {
        "total_tokens": token_count,
        "total_users": user_count,
        "total_entries": token_count + user_count,
    }


# Middleware-friendly verification function
async def verify_token_not_blacklisted(token: str, user_id: str) -> bool:
    """
    Comprehensive blacklist check for use in auth middleware.

    Checks both token-level and user-level blacklists.

    Args:
        token: JWT token to verify
        user_id: User identifier from token

    Returns:
        True if token is valid (not blacklisted), False if blacklisted

    Raises:
        None - returns boolean for easy middleware integration

    Example:
        >>> if not await verify_token_not_blacklisted(token, user_id):
        ...     raise HTTPException(401, "Token has been revoked")
    """
    try:
        # Check user-level blacklist first (faster)
        if await is_user_blacklisted(user_id):
            logger.debug(f"User-level blacklist hit: {user_id}")
            return False

        # Check token-level blacklist
        if await is_blacklisted(token):
            logger.debug(f"Token-level blacklist hit for user: {user_id}")
            return False

        return True

    except Exception as e:
        logger.error(f"Error in blacklist verification: {str(e)}")
        # Fail secure: treat as blacklisted
        return False


__all__ = [
    "blacklist_token",
    "is_blacklisted",
    "blacklist_all_user_tokens",
    "is_user_blacklisted",
    "remove_from_blacklist",
    "clear_user_blacklist",
    "get_blacklist_info",
    "cleanup_expired_blacklist_entries",
    "get_blacklist_stats",
    "verify_token_not_blacklisted",
    "get_redis_client",
]
