"""
Configuration for DLQ handler tests.

This module configures pytest-asyncio for proper async fixture handling.
"""

import pytest
import pytest_asyncio
from typing import AsyncGenerator

from greenlang.infrastructure.events.dlq_handler import (
    DeadLetterQueueHandler,
    DLQHandlerConfig,
)


@pytest_asyncio.fixture
async def dlq_handler_fixture() -> AsyncGenerator[DeadLetterQueueHandler, None]:
    """Create DLQ handler for testing."""
    config = DLQHandlerConfig(
        kafka_enabled=False,  # Disable Kafka for unit tests
        redis_enabled=False,  # Disable Redis for unit tests
        max_retries=3,
        initial_backoff_seconds=10,
        dlq_depth_threshold=5
    )
    handler = DeadLetterQueueHandler(config)
    await handler.start()
    yield handler
    await handler.stop()
