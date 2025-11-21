# -*- coding: utf-8 -*-
"""
Pytest configuration for messaging tests.

Provides fixtures and configuration for all test modules.
"""

import pytest
import asyncio
import os
from typing import AsyncGenerator

# Configure Redis connection for tests
REDIS_URL = os.getenv("GREENLANG_TEST_REDIS_URL", "redis://localhost:6379")


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def redis_url():
    """Redis connection URL for tests."""
    return REDIS_URL


@pytest.fixture
async def redis_broker():
    """Create Redis broker for tests."""
    from ..redis_streams_broker import RedisStreamsBroker

    broker = RedisStreamsBroker(redis_url=REDIS_URL)
    await broker.connect()

    yield broker

    await broker.disconnect()


@pytest.fixture
async def consumer_manager(redis_broker):
    """Create consumer group manager for tests."""
    from ..consumer_group import ConsumerGroupManager

    manager = ConsumerGroupManager(redis_broker)

    yield manager

    await manager.shutdown()


def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "integration: integration tests requiring Redis"
    )
    config.addinivalue_line(
        "markers", "slow: slow tests"
    )
    config.addinivalue_line(
        "markers", "performance: performance tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection."""
    # Add integration marker to all async tests
    for item in items:
        if asyncio.iscoroutinefunction(item.function):
            item.add_marker(pytest.mark.asyncio)
