"""
Registry Module for GL-FOUND-X-002.

This module provides schema registry capabilities including:
    - Git-backed schema storage
    - HTTP registry client
    - IR cache service
    - Schema resolver interface

Components:
    - resolver: Schema resolver interface (SchemaRegistry, SchemaSource)
    - git_backend: Git-backed registry (GitSchemaRegistry)
    - cache: IR cache service (IRCacheService, CacheWarmupScheduler)
    - client: HTTP registry client

Example:
    >>> from greenlang.schema.registry import GitSchemaRegistry
    >>> registry = GitSchemaRegistry("./schemas")
    >>> schema = registry.resolve("emissions/activity", "1.3.0")
    >>> versions = registry.list_versions("emissions/activity")
    >>> latest = registry.get_latest("emissions/activity", "^1.0.0")

    >>> from greenlang.schema.registry import IRCacheService, CacheWarmupScheduler
    >>> cache = IRCacheService(max_size=1000, ttl_seconds=3600)
    >>> scheduler = CacheWarmupScheduler(cache, interval_seconds=300)
"""

from greenlang.schema.registry.cache import (
    CacheEntry,
    CacheMetrics,
    CacheWarmupScheduler,
    IRCacheService,
)
from greenlang.schema.registry.git_backend import (
    CachedSchema,
    GitOperationError,
    GitSchemaRegistry,
    InvalidSchemaIdError,
    SchemaCache,
    SchemaNotFoundError,
    SchemaParseError,
    SchemaSourceModel,
    SemVer,
    VersionConstraint,
    VersionConstraintError,
    compare_versions,
    create_git_registry,
    filter_versions,
    sort_versions,
)
from greenlang.schema.registry.resolver import (
    SchemaRegistry,
    SchemaSource,
)

__all__ = [
    # Resolver interface
    "SchemaRegistry",
    "SchemaSource",
    # Git backend
    "GitSchemaRegistry",
    "create_git_registry",
    "SchemaSourceModel",
    "CachedSchema",
    "SchemaCache",
    "SchemaNotFoundError",
    "InvalidSchemaIdError",
    "VersionConstraintError",
    "GitOperationError",
    "SchemaParseError",
    # Semver utilities
    "SemVer",
    "VersionConstraint",
    "compare_versions",
    "sort_versions",
    "filter_versions",
    # Cache models
    "CacheEntry",
    "CacheMetrics",
    # Cache services
    "IRCacheService",
    "CacheWarmupScheduler",
]
