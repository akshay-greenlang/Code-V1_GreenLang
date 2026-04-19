"""
GreenLang Storage Layer - Persistence Infrastructure.

This module provides a unified storage abstraction layer for GreenLang,
supporting multiple backends for webhook and agent state persistence.

Features:
    - Multiple storage backends (InMemory, SQLite, PostgreSQL)
    - Automatic backend selection based on configuration
    - Connection pooling and async support
    - Provenance tracking with SHA-256 hashes
    - TTL-based cleanup

Backend Selection:
    - memory: In-memory storage (default, for testing)
    - sqlite: SQLite file-based storage (single-node)
    - postgres: PostgreSQL (production, distributed)

Example:
    >>> from greenlang.infrastructure.api.storage import StorageFactory
    >>>
    >>> # Auto-select based on config
    >>> config = {"backend": "sqlite", "sqlite_path": "data.db"}
    >>> webhook_store = StorageFactory.get_webhook_store(config)
    >>> agent_store = StorageFactory.get_agent_store(config)
    >>>
    >>> # Or use convenience functions
    >>> webhook_store = get_default_webhook_store()
    >>> agent_store = get_default_agent_store()
"""

import logging
import os
from typing import Any, Dict, Optional, Union

from greenlang.infrastructure.api.storage.webhook_store import (
    WebhookStore,
    WebhookStoreConfig,
    BaseWebhookStore,
    InMemoryWebhookStore,
    SQLiteWebhookStore,
    PostgresWebhookStore,
)
from greenlang.infrastructure.api.storage.agent_state_store import (
    AgentState,
    CalculationResult,
    AgentStateStore,
    AgentStateStoreConfig,
    BaseAgentStateStore,
    InMemoryAgentStateStore,
    SQLiteAgentStateStore,
    PostgresAgentStateStore,
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Storage Factory
# -----------------------------------------------------------------------------


class StorageFactory:
    """
    Factory for creating storage backend instances.

    Automatically selects the appropriate backend based on configuration
    or environment variables.

    Class Methods:
        get_webhook_store: Create webhook storage instance
        get_agent_store: Create agent state storage instance
        get_backend_from_env: Detect backend from environment

    Configuration Options:
        backend: "memory", "sqlite", or "postgres"
        sqlite_path: Path to SQLite database file
        postgres_dsn: PostgreSQL connection string
        postgres_pool_size: Connection pool size (default: 10)
        enable_wal_mode: Enable SQLite WAL mode (default: True)

    Environment Variables:
        GREENLANG_STORAGE_BACKEND: Override backend selection
        GREENLANG_SQLITE_PATH: SQLite database path
        GREENLANG_POSTGRES_DSN: PostgreSQL connection string
        DATABASE_URL: Alternative PostgreSQL connection string

    Example:
        >>> # From configuration dict
        >>> store = StorageFactory.get_webhook_store({
        ...     "backend": "sqlite",
        ...     "sqlite_path": "webhooks.db"
        ... })
        >>>
        >>> # From environment
        >>> os.environ["GREENLANG_STORAGE_BACKEND"] = "postgres"
        >>> os.environ["DATABASE_URL"] = "postgresql://localhost/db"
        >>> store = StorageFactory.get_webhook_store({})
    """

    # Singleton instances for reuse
    _webhook_store_instance: Optional[BaseWebhookStore] = None
    _agent_store_instance: Optional[BaseAgentStateStore] = None
    _current_config: Optional[Dict[str, Any]] = None

    @classmethod
    def get_backend_from_env(cls) -> str:
        """
        Detect storage backend from environment variables.

        Returns:
            Backend name: "memory", "sqlite", or "postgres"
        """
        # Explicit backend override
        if os.getenv("GREENLANG_STORAGE_BACKEND"):
            return os.getenv("GREENLANG_STORAGE_BACKEND", "memory")

        # Auto-detect from connection strings
        if os.getenv("DATABASE_URL") or os.getenv("GREENLANG_POSTGRES_DSN"):
            return "postgres"

        if os.getenv("GREENLANG_SQLITE_PATH"):
            return "sqlite"

        return "memory"

    @classmethod
    def _normalize_config(
        cls,
        config: Optional[Union[Dict[str, Any], WebhookStoreConfig, AgentStateStoreConfig]]
    ) -> Dict[str, Any]:
        """
        Normalize configuration to dictionary format.

        Args:
            config: Configuration dict or Pydantic model

        Returns:
            Normalized configuration dictionary
        """
        if config is None:
            config = {}

        if isinstance(config, (WebhookStoreConfig, AgentStateStoreConfig)):
            config = config.dict()

        # Apply environment variable overrides
        env_backend = cls.get_backend_from_env()
        if "backend" not in config:
            config["backend"] = env_backend

        # SQLite path
        if config.get("backend") == "sqlite":
            if "sqlite_path" not in config:
                config["sqlite_path"] = os.getenv(
                    "GREENLANG_SQLITE_PATH",
                    "greenlang_storage.db"
                )

        # PostgreSQL DSN
        if config.get("backend") == "postgres":
            if "postgres_dsn" not in config:
                config["postgres_dsn"] = os.getenv(
                    "GREENLANG_POSTGRES_DSN",
                    os.getenv("DATABASE_URL", "")
                )

        return config

    @classmethod
    def get_webhook_store(
        cls,
        config: Optional[Union[Dict[str, Any], WebhookStoreConfig]] = None,
        reuse_existing: bool = True
    ) -> BaseWebhookStore:
        """
        Create or retrieve a webhook storage instance.

        Args:
            config: Storage configuration (dict or WebhookStoreConfig)
            reuse_existing: If True, return existing instance if config matches

        Returns:
            Webhook storage instance

        Raises:
            ValueError: If invalid backend specified

        Example:
            >>> # In-memory (default)
            >>> store = StorageFactory.get_webhook_store()
            >>>
            >>> # SQLite
            >>> store = StorageFactory.get_webhook_store({
            ...     "backend": "sqlite",
            ...     "sqlite_path": "webhooks.db"
            ... })
            >>>
            >>> # PostgreSQL
            >>> store = StorageFactory.get_webhook_store({
            ...     "backend": "postgres",
            ...     "postgres_dsn": "postgresql://user:pass@localhost/db"
            ... })
        """
        normalized = cls._normalize_config(config)
        backend = normalized.get("backend", "memory")

        # Check for existing instance
        if reuse_existing and cls._webhook_store_instance is not None:
            if cls._current_config == normalized:
                logger.debug("Reusing existing webhook store instance")
                return cls._webhook_store_instance

        logger.info(f"Creating webhook store with backend: {backend}")

        if backend == "memory":
            store = InMemoryWebhookStore()

        elif backend == "sqlite":
            sqlite_path = normalized.get("sqlite_path", "webhooks.db")
            enable_wal = normalized.get("enable_wal_mode", True)
            store = SQLiteWebhookStore(
                db_path=sqlite_path,
                enable_wal=enable_wal
            )

        elif backend == "postgres":
            postgres_dsn = normalized.get("postgres_dsn", "")
            if not postgres_dsn:
                raise ValueError(
                    "PostgreSQL DSN required. Set postgres_dsn in config or "
                    "DATABASE_URL/GREENLANG_POSTGRES_DSN environment variable."
                )
            pool_size = normalized.get("postgres_pool_size", 10)
            store = PostgresWebhookStore(
                dsn=postgres_dsn,
                pool_size=pool_size
            )

        else:
            raise ValueError(
                f"Invalid storage backend: {backend}. "
                "Supported: memory, sqlite, postgres"
            )

        # Cache instance
        cls._webhook_store_instance = store
        cls._current_config = normalized

        return store

    @classmethod
    def get_agent_store(
        cls,
        config: Optional[Union[Dict[str, Any], AgentStateStoreConfig]] = None,
        reuse_existing: bool = True
    ) -> BaseAgentStateStore:
        """
        Create or retrieve an agent state storage instance.

        Args:
            config: Storage configuration (dict or AgentStateStoreConfig)
            reuse_existing: If True, return existing instance if config matches

        Returns:
            Agent state storage instance

        Raises:
            ValueError: If invalid backend specified

        Example:
            >>> # In-memory (default)
            >>> store = StorageFactory.get_agent_store()
            >>>
            >>> # SQLite
            >>> store = StorageFactory.get_agent_store({
            ...     "backend": "sqlite",
            ...     "sqlite_path": "agent_state.db"
            ... })
            >>>
            >>> # PostgreSQL
            >>> store = StorageFactory.get_agent_store({
            ...     "backend": "postgres",
            ...     "postgres_dsn": "postgresql://user:pass@localhost/db"
            ... })
        """
        normalized = cls._normalize_config(config)
        backend = normalized.get("backend", "memory")

        # Check for existing instance
        if reuse_existing and cls._agent_store_instance is not None:
            if cls._current_config == normalized:
                logger.debug("Reusing existing agent store instance")
                return cls._agent_store_instance

        logger.info(f"Creating agent store with backend: {backend}")

        if backend == "memory":
            store = InMemoryAgentStateStore()

        elif backend == "sqlite":
            sqlite_path = normalized.get("sqlite_path", "agent_state.db")
            enable_wal = normalized.get("enable_wal_mode", True)
            store = SQLiteAgentStateStore(
                db_path=sqlite_path,
                enable_wal=enable_wal
            )

        elif backend == "postgres":
            postgres_dsn = normalized.get("postgres_dsn", "")
            if not postgres_dsn:
                raise ValueError(
                    "PostgreSQL DSN required. Set postgres_dsn in config or "
                    "DATABASE_URL/GREENLANG_POSTGRES_DSN environment variable."
                )
            pool_size = normalized.get("postgres_pool_size", 10)
            store = PostgresAgentStateStore(
                dsn=postgres_dsn,
                pool_size=pool_size
            )

        else:
            raise ValueError(
                f"Invalid storage backend: {backend}. "
                "Supported: memory, sqlite, postgres"
            )

        # Cache instance
        cls._agent_store_instance = store
        cls._current_config = normalized

        return store

    @classmethod
    def reset(cls) -> None:
        """
        Reset factory state and close any cached instances.

        Call this when reconfiguring storage or during testing.
        """
        cls._webhook_store_instance = None
        cls._agent_store_instance = None
        cls._current_config = None
        logger.info("StorageFactory reset")


# -----------------------------------------------------------------------------
# Convenience Functions
# -----------------------------------------------------------------------------


def get_default_webhook_store() -> BaseWebhookStore:
    """
    Get the default webhook store based on environment configuration.

    Returns:
        Webhook storage instance

    Example:
        >>> store = get_default_webhook_store()
        >>> await store.save_webhook(webhook)
    """
    return StorageFactory.get_webhook_store()


def get_default_agent_store() -> BaseAgentStateStore:
    """
    Get the default agent store based on environment configuration.

    Returns:
        Agent state storage instance

    Example:
        >>> store = get_default_agent_store()
        >>> await store.save_agent_state("agent-1", state)
    """
    return StorageFactory.get_agent_store()


async def initialize_stores(
    config: Optional[Dict[str, Any]] = None
) -> tuple:
    """
    Initialize both webhook and agent stores.

    Useful for application startup to ensure database schemas exist.

    Args:
        config: Storage configuration

    Returns:
        Tuple of (webhook_store, agent_store)

    Example:
        >>> webhook_store, agent_store = await initialize_stores({
        ...     "backend": "sqlite"
        ... })
    """
    webhook_store = StorageFactory.get_webhook_store(config)
    agent_store = StorageFactory.get_agent_store(config)

    # Initialize if needed (SQLite/Postgres)
    if hasattr(webhook_store, 'initialize'):
        await webhook_store.initialize()
    if hasattr(agent_store, 'initialize'):
        await agent_store.initialize()

    return webhook_store, agent_store


async def close_stores() -> None:
    """
    Close all storage connections and reset factory.

    Call this during application shutdown.

    Example:
        >>> await close_stores()
    """
    if StorageFactory._webhook_store_instance:
        await StorageFactory._webhook_store_instance.close()
    if StorageFactory._agent_store_instance:
        await StorageFactory._agent_store_instance.close()
    StorageFactory.reset()


# -----------------------------------------------------------------------------
# Exports
# -----------------------------------------------------------------------------

__all__ = [
    # Factory
    "StorageFactory",

    # Webhook Store
    "WebhookStore",
    "WebhookStoreConfig",
    "BaseWebhookStore",
    "InMemoryWebhookStore",
    "SQLiteWebhookStore",
    "PostgresWebhookStore",

    # Agent State Store
    "AgentState",
    "CalculationResult",
    "AgentStateStore",
    "AgentStateStoreConfig",
    "BaseAgentStateStore",
    "InMemoryAgentStateStore",
    "SQLiteAgentStateStore",
    "PostgresAgentStateStore",

    # Convenience functions
    "get_default_webhook_store",
    "get_default_agent_store",
    "initialize_stores",
    "close_stores",
]
