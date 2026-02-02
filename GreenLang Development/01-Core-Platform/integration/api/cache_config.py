# -*- coding: utf-8 -*-
"""
greenlang/api/cache_config.py

Redis caching configuration for FastAPI.
"""

from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.decorator import cache
from redis import asyncio as aioredis
import os
import logging

logger = logging.getLogger(__name__)


async def init_cache():
    """
    Initialize Redis cache for FastAPI.

    Environment variables:
    - REDIS_URL: Redis connection URL (default: redis://localhost:6379)
    - CACHE_ENABLED: Enable/disable caching (default: true)
    """
    cache_enabled = os.getenv("CACHE_ENABLED", "true").lower() == "true"

    if not cache_enabled:
        logger.info("Caching is disabled")
        return

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")

    try:
        redis = aioredis.from_url(
            redis_url,
            encoding="utf8",
            decode_responses=True
        )

        FastAPICache.init(
            RedisBackend(redis),
            prefix="greenlang-api:"
        )

        logger.info(f"Redis cache initialized: {redis_url}")

    except Exception as e:
        logger.error(f"Failed to initialize Redis cache: {e}")
        logger.warning("Continuing without caching")


async def close_cache():
    """Close Redis cache connection"""
    try:
        await FastAPICache.clear()
        logger.info("Redis cache closed")
    except Exception as e:
        logger.error(f"Error closing cache: {e}")


# Cache decorator with default settings
def cache_response(expire: int = 3600):
    """
    Decorator for caching endpoint responses.

    Args:
        expire: Cache TTL in seconds (default: 1 hour)
    """
    return cache(expire=expire)
