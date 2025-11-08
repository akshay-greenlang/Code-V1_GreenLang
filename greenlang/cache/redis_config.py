"""
Redis configuration for Phase 4 session storage and caching
"""

import json
import logging
import os
from datetime import timedelta
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)

try:
    import redis
    from redis.connection import ConnectionPool
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available. Install with: pip install redis")


class RedisConfig:
    """Redis configuration manager"""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        db: Optional[int] = None,
        password: Optional[str] = None,
        ssl: bool = False,
        **kwargs
    ):
        """
        Initialize Redis configuration

        Args:
            host: Redis host (defaults to REDIS_HOST env or 'localhost')
            port: Redis port (defaults to REDIS_PORT env or 6379)
            db: Redis database number (defaults to REDIS_DB env or 0)
            password: Redis password (defaults to REDIS_PASSWORD env)
            ssl: Use SSL connection
            **kwargs: Additional redis connection parameters
        """
        self.host = host or os.getenv("REDIS_HOST", "localhost")
        self.port = port or int(os.getenv("REDIS_PORT", "6379"))
        self.db = db or int(os.getenv("REDIS_DB", "0"))
        self.password = password or os.getenv("REDIS_PASSWORD")
        self.ssl = ssl or os.getenv("REDIS_SSL", "false").lower() == "true"

        self.kwargs = kwargs
        self.kwargs.update({
            "decode_responses": True,
            "socket_keepalive": True,
            "socket_timeout": kwargs.get("socket_timeout", 5),
            "retry_on_timeout": True,
        })

        if self.ssl:
            self.kwargs["ssl"] = True
            self.kwargs["ssl_cert_reqs"] = "required"

        self._pool: Optional[ConnectionPool] = None
        self._client: Optional[redis.Redis] = None

    def get_pool(self) -> Optional[ConnectionPool]:
        """
        Get or create Redis connection pool

        Returns:
            Redis connection pool or None if Redis not available
        """
        if not REDIS_AVAILABLE:
            return None

        if self._pool is None:
            self._pool = ConnectionPool(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                **self.kwargs
            )

        return self._pool

    def get_client(self) -> Optional[redis.Redis]:
        """
        Get or create Redis client

        Returns:
            Redis client or None if Redis not available
        """
        if not REDIS_AVAILABLE:
            logger.warning("Redis client requested but redis not available")
            return None

        if self._client is None:
            pool = self.get_pool()
            if pool:
                self._client = redis.Redis(connection_pool=pool)

        return self._client

    def health_check(self) -> bool:
        """
        Check if Redis connection is healthy

        Returns:
            True if connection is healthy
        """
        if not REDIS_AVAILABLE:
            return False

        try:
            client = self.get_client()
            if client:
                return client.ping()
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")

        return False

    def close(self):
        """Close Redis connections"""
        if self._client:
            self._client.close()
            self._client = None

        if self._pool:
            self._pool.disconnect()
            self._pool = None


class RedisSessionStore:
    """Redis-backed session store for Phase 4"""

    def __init__(self, redis_config: Optional[RedisConfig] = None, prefix: str = "session:"):
        """
        Initialize Redis session store

        Args:
            redis_config: Redis configuration (creates default if None)
            prefix: Key prefix for sessions
        """
        self.config = redis_config or RedisConfig()
        self.prefix = prefix
        self.client = self.config.get_client()

    def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session data

        Args:
            session_id: Session ID

        Returns:
            Session data dictionary or None if not found
        """
        if not self.client:
            return None

        try:
            key = f"{self.prefix}{session_id}"
            data = self.client.get(key)

            if data:
                return json.loads(data)

        except Exception as e:
            logger.error(f"Error getting session {session_id}: {e}")

        return None

    def set(
        self,
        session_id: str,
        data: Dict[str, Any],
        ttl: Optional[Union[int, timedelta]] = None
    ) -> bool:
        """
        Set session data

        Args:
            session_id: Session ID
            data: Session data dictionary
            ttl: Time to live in seconds or timedelta

        Returns:
            True if successful
        """
        if not self.client:
            return False

        try:
            key = f"{self.prefix}{session_id}"
            value = json.dumps(data)

            if ttl:
                if isinstance(ttl, timedelta):
                    ttl = int(ttl.total_seconds())
                self.client.setex(key, ttl, value)
            else:
                self.client.set(key, value)

            return True

        except Exception as e:
            logger.error(f"Error setting session {session_id}: {e}")
            return False

    def delete(self, session_id: str) -> bool:
        """
        Delete session

        Args:
            session_id: Session ID

        Returns:
            True if successful
        """
        if not self.client:
            return False

        try:
            key = f"{self.prefix}{session_id}"
            self.client.delete(key)
            return True

        except Exception as e:
            logger.error(f"Error deleting session {session_id}: {e}")
            return False

    def exists(self, session_id: str) -> bool:
        """
        Check if session exists

        Args:
            session_id: Session ID

        Returns:
            True if session exists
        """
        if not self.client:
            return False

        try:
            key = f"{self.prefix}{session_id}"
            return bool(self.client.exists(key))

        except Exception as e:
            logger.error(f"Error checking session {session_id}: {e}")
            return False

    def extend_ttl(self, session_id: str, ttl: Union[int, timedelta]) -> bool:
        """
        Extend session TTL

        Args:
            session_id: Session ID
            ttl: Time to live in seconds or timedelta

        Returns:
            True if successful
        """
        if not self.client:
            return False

        try:
            key = f"{self.prefix}{session_id}"

            if isinstance(ttl, timedelta):
                ttl = int(ttl.total_seconds())

            return bool(self.client.expire(key, ttl))

        except Exception as e:
            logger.error(f"Error extending TTL for session {session_id}: {e}")
            return False

    def get_ttl(self, session_id: str) -> Optional[int]:
        """
        Get remaining TTL for session

        Args:
            session_id: Session ID

        Returns:
            TTL in seconds or None
        """
        if not self.client:
            return None

        try:
            key = f"{self.prefix}{session_id}"
            ttl = self.client.ttl(key)

            if ttl >= 0:
                return ttl

        except Exception as e:
            logger.error(f"Error getting TTL for session {session_id}: {e}")

        return None

    def list_sessions(self, pattern: str = "*") -> list:
        """
        List session IDs matching pattern

        Args:
            pattern: Pattern to match (default: all sessions)

        Returns:
            List of session IDs
        """
        if not self.client:
            return []

        try:
            key_pattern = f"{self.prefix}{pattern}"
            keys = self.client.keys(key_pattern)

            # Remove prefix from keys
            prefix_len = len(self.prefix)
            return [key[prefix_len:] for key in keys]

        except Exception as e:
            logger.error(f"Error listing sessions: {e}")
            return []

    def clear_all(self) -> int:
        """
        Clear all sessions

        Returns:
            Number of sessions cleared
        """
        if not self.client:
            return 0

        try:
            keys = self.client.keys(f"{self.prefix}*")
            if keys:
                return self.client.delete(*keys)

        except Exception as e:
            logger.error(f"Error clearing sessions: {e}")

        return 0


# Global Redis configuration
_global_redis_config: Optional[RedisConfig] = None
_global_session_store: Optional[RedisSessionStore] = None


def get_redis_config() -> RedisConfig:
    """Get or create global Redis configuration"""
    global _global_redis_config

    if _global_redis_config is None:
        _global_redis_config = RedisConfig()

    return _global_redis_config


def get_session_store() -> RedisSessionStore:
    """Get or create global session store"""
    global _global_session_store

    if _global_session_store is None:
        _global_session_store = RedisSessionStore(get_redis_config())

    return _global_session_store


def reset_redis() -> None:
    """Reset global Redis instances (for testing)"""
    global _global_redis_config, _global_session_store

    if _global_redis_config:
        _global_redis_config.close()
        _global_redis_config = None

    _global_session_store = None
