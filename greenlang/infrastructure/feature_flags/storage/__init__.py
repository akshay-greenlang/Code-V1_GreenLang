# -*- coding: utf-8 -*-
"""
Feature Flag Storage Backends - INFRA-008

Provides a pluggable storage architecture for the feature flag system:

- ``IFlagStorage`` -- Abstract interface all backends implement.
- ``InMemoryFlagStorage`` -- L1 in-memory cache with LRU eviction and TTL.
- ``RedisFlagStorage`` -- L2 Redis cache with circuit breaker and pub/sub.
- ``PostgresFlagStorage`` -- L3 persistent PostgreSQL store.
- ``MultiLayerFlagStorage`` -- Orchestrator that cascades L1 -> L2 -> L3.

Usage:
    >>> from greenlang.infrastructure.feature_flags.storage import (
    ...     MultiLayerFlagStorage,
    ...     InMemoryFlagStorage,
    ... )
    >>> storage = MultiLayerFlagStorage(l1=InMemoryFlagStorage())
"""

from greenlang.infrastructure.feature_flags.storage.base import IFlagStorage
from greenlang.infrastructure.feature_flags.storage.memory import InMemoryFlagStorage
from greenlang.infrastructure.feature_flags.storage.multi_layer import (
    MultiLayerFlagStorage,
)

# Redis and Postgres are optional (may require extra dependencies).
# Import them lazily so the package works even without redis / psycopg.
try:
    from greenlang.infrastructure.feature_flags.storage.redis_store import (
        RedisFlagStorage,
    )
except ImportError:
    RedisFlagStorage = None  # type: ignore[assignment,misc]

try:
    from greenlang.infrastructure.feature_flags.storage.postgres_store import (
        PostgresFlagStorage,
    )
except ImportError:
    PostgresFlagStorage = None  # type: ignore[assignment,misc]

__all__ = [
    "IFlagStorage",
    "InMemoryFlagStorage",
    "RedisFlagStorage",
    "PostgresFlagStorage",
    "MultiLayerFlagStorage",
]
