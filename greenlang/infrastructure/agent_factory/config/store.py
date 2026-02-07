"""
Config Store - Agent Factory Config (INFRA-010)

Two-layer configuration store for agent settings. Uses Redis (L1) as a
fast cache with TTL and PostgreSQL (L2) for durable persistence.
Supports optimistic locking via config versioning and publishes change
notifications for hot reload.

Classes:
    - ConfigStore: Two-layer configuration persistence service.

Example:
    >>> store = ConfigStore(redis_client, db_pool, hot_reload)
    >>> config = AgentConfigSchema(agent_key="intake-agent", version=1)
    >>> await store.set_config("intake-agent", config, changed_by="admin")
    >>> loaded = await store.get_config("intake-agent")
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from greenlang.infrastructure.agent_factory.config.schema import AgentConfigSchema

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REDIS_KEY_PREFIX = "gl:agent_config:"
_REDIS_TTL_SECONDS = 300  # 5 minutes


# ---------------------------------------------------------------------------
# SQL Statements
# ---------------------------------------------------------------------------

_CREATE_TABLE_SQL = """\
CREATE TABLE IF NOT EXISTS agent_configs (
    agent_key       TEXT PRIMARY KEY,
    config_data     JSONB NOT NULL,
    version         INTEGER NOT NULL DEFAULT 1,
    schema_version  INTEGER NOT NULL DEFAULT 1,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_by      TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_agent_configs_updated
    ON agent_configs (updated_at DESC);
"""

_UPSERT_SQL = """\
INSERT INTO agent_configs (agent_key, config_data, version, schema_version, updated_at, updated_by)
VALUES ($1, $2::jsonb, $3, $4, $5, $6)
ON CONFLICT (agent_key) DO UPDATE
SET config_data = EXCLUDED.config_data,
    version = EXCLUDED.version,
    schema_version = EXCLUDED.schema_version,
    updated_at = EXCLUDED.updated_at,
    updated_by = EXCLUDED.updated_by
WHERE agent_configs.version < EXCLUDED.version
"""

_SELECT_SQL = """\
SELECT agent_key, config_data, version, schema_version, updated_at, updated_by
FROM agent_configs
WHERE agent_key = $1
"""

_SELECT_ALL_SQL = """\
SELECT agent_key, config_data, version, schema_version, updated_at, updated_by
FROM agent_configs
ORDER BY agent_key
"""

_DELETE_SQL = """\
DELETE FROM agent_configs WHERE agent_key = $1
"""


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ConfigVersionConflictError(Exception):
    """Raised when an optimistic locking conflict occurs.

    Attributes:
        agent_key: The agent whose config had a conflict.
        expected_version: The version that was expected.
        actual_version: The version found in the store.
    """

    def __init__(
        self,
        agent_key: str,
        expected_version: int,
        actual_version: int,
    ) -> None:
        self.agent_key = agent_key
        self.expected_version = expected_version
        self.actual_version = actual_version
        super().__init__(
            f"Config version conflict for '{agent_key}': "
            f"expected v{expected_version}, found v{actual_version}. "
            f"Reload and retry."
        )


# ---------------------------------------------------------------------------
# Config Store
# ---------------------------------------------------------------------------


class ConfigStore:
    """Two-layer configuration store for agent settings.

    Layer 1 (L1): Redis cache with 5-minute TTL for fast reads.
    Layer 2 (L2): PostgreSQL for durable persistence.

    Writes go to both layers. Reads check L1 first, then L2.
    Config versioning provides optimistic concurrency control.

    Attributes:
        redis_ttl_s: Redis cache TTL in seconds.
    """

    def __init__(
        self,
        redis_client: Any,
        db_pool: Any,
        hot_reload: Optional[Any] = None,
        redis_ttl_s: int = _REDIS_TTL_SECONDS,
    ) -> None:
        """Initialize the config store.

        Args:
            redis_client: Async Redis client (redis.asyncio).
            db_pool: Async PostgreSQL connection pool.
            hot_reload: Optional ConfigHotReload for change notifications.
            redis_ttl_s: Redis cache TTL in seconds.
        """
        self._redis = redis_client
        self._pool = db_pool
        self._hot_reload = hot_reload
        self.redis_ttl_s = redis_ttl_s
        logger.info(
            "ConfigStore initialised (redis_ttl=%ds)", redis_ttl_s,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Create the configs table if it does not exist."""
        conn = await self._acquire_connection()
        try:
            await conn.execute(_CREATE_TABLE_SQL)
        finally:
            await self._release_connection(conn)
        logger.info("ConfigStore: table initialised")

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def get_config(self, agent_key: str) -> Optional[AgentConfigSchema]:
        """Get agent configuration, checking L1 then L2.

        Args:
            agent_key: The agent to look up.

        Returns:
            AgentConfigSchema if found, None otherwise.
        """
        # L1: Redis cache
        cached = await self._get_from_redis(agent_key)
        if cached is not None:
            logger.debug("ConfigStore: L1 hit for '%s'", agent_key)
            return cached

        # L2: PostgreSQL
        config = await self._get_from_postgres(agent_key)
        if config is not None:
            logger.debug("ConfigStore: L2 hit for '%s', populating L1", agent_key)
            await self._set_in_redis(agent_key, config)
            return config

        logger.debug("ConfigStore: no config found for '%s'", agent_key)
        return None

    async def list_configs(self) -> List[AgentConfigSchema]:
        """List all agent configurations from PostgreSQL.

        Returns:
            List of all agent configs.
        """
        configs: List[AgentConfigSchema] = []
        conn = await self._acquire_connection()
        try:
            rows = await conn.fetch(_SELECT_ALL_SQL)
            for row in rows:
                data = row["config_data"]
                if isinstance(data, str):
                    data = json.loads(data)
                data = AgentConfigSchema.migrate(data)
                try:
                    configs.append(AgentConfigSchema(**data))
                except Exception as exc:
                    logger.error(
                        "ConfigStore: failed to parse config for '%s': %s",
                        row["agent_key"], exc,
                    )
        finally:
            await self._release_connection(conn)
        return configs

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    async def set_config(
        self,
        agent_key: str,
        config: AgentConfigSchema,
        changed_by: str = "",
    ) -> AgentConfigSchema:
        """Set agent configuration with optimistic locking.

        Writes to both L1 (Redis) and L2 (PostgreSQL). Publishes a
        change notification for hot reload if configured.

        Args:
            agent_key: The agent key.
            config: The configuration to save.
            changed_by: Identity of who made the change.

        Returns:
            The saved configuration.

        Raises:
            ConfigVersionConflictError: If the version is stale.
        """
        # Get existing config for version check and diff
        existing = await self._get_from_postgres(agent_key)
        old_config_dict: Dict[str, Any] = {}

        if existing is not None:
            if config.version <= existing.version:
                raise ConfigVersionConflictError(
                    agent_key=agent_key,
                    expected_version=config.version,
                    actual_version=existing.version,
                )
            old_config_dict = existing.model_dump(mode="json")

        # Update metadata
        now = datetime.now(timezone.utc)
        config_dict = config.model_dump(mode="json")
        config_dict["updated_at"] = now.isoformat()
        config_dict["updated_by"] = changed_by

        # L2: PostgreSQL (persistent)
        await self._set_in_postgres(agent_key, config_dict, config.version)

        # L1: Redis (cache)
        updated_config = AgentConfigSchema(**config_dict)
        await self._set_in_redis(agent_key, updated_config)

        # Publish change notification
        if self._hot_reload is not None:
            from greenlang.infrastructure.agent_factory.config.hot_reload import (
                ConfigChangeEvent,
            )
            new_config_dict = updated_config.model_dump(mode="json")
            changed_keys = frozenset(
                k for k in new_config_dict
                if old_config_dict.get(k) != new_config_dict.get(k)
            )
            event = ConfigChangeEvent(
                agent_key=agent_key,
                old_config=old_config_dict,
                new_config=new_config_dict,
                changed_keys=changed_keys,
                config_version=config.version,
                changed_by=changed_by,
            )
            try:
                await self._hot_reload.publish_change(event)
            except Exception as exc:
                logger.error(
                    "ConfigStore: failed to publish change notification: %s", exc,
                )

        logger.info(
            "ConfigStore: saved config for '%s' v%d (by %s)",
            agent_key, config.version, changed_by,
        )
        return updated_config

    async def delete_config(self, agent_key: str) -> bool:
        """Delete an agent configuration from both layers.

        Args:
            agent_key: The agent key to delete.

        Returns:
            True if a config was found and deleted.
        """
        # L1: Redis
        redis_key = f"{_REDIS_KEY_PREFIX}{agent_key}"
        await self._redis.delete(redis_key)

        # L2: PostgreSQL
        conn = await self._acquire_connection()
        try:
            result = await conn.execute(_DELETE_SQL, agent_key)
        finally:
            await self._release_connection(conn)

        deleted = result and "DELETE 1" in str(result)
        if deleted:
            logger.info("ConfigStore: deleted config for '%s'", agent_key)
        return deleted

    # ------------------------------------------------------------------
    # Internal: Redis (L1)
    # ------------------------------------------------------------------

    async def _get_from_redis(self, agent_key: str) -> Optional[AgentConfigSchema]:
        """Get config from Redis cache."""
        redis_key = f"{_REDIS_KEY_PREFIX}{agent_key}"
        try:
            raw = await self._redis.get(redis_key)
            if raw is None:
                return None
            data = json.loads(raw)
            data = AgentConfigSchema.migrate(data)
            return AgentConfigSchema(**data)
        except Exception as exc:
            logger.warning(
                "ConfigStore: Redis read failed for '%s': %s", agent_key, exc,
            )
            return None

    async def _set_in_redis(
        self, agent_key: str, config: AgentConfigSchema,
    ) -> None:
        """Write config to Redis cache with TTL."""
        redis_key = f"{_REDIS_KEY_PREFIX}{agent_key}"
        try:
            data = config.model_dump(mode="json")
            await self._redis.setex(
                redis_key,
                self.redis_ttl_s,
                json.dumps(data, default=str),
            )
        except Exception as exc:
            logger.warning(
                "ConfigStore: Redis write failed for '%s': %s", agent_key, exc,
            )

    # ------------------------------------------------------------------
    # Internal: PostgreSQL (L2)
    # ------------------------------------------------------------------

    async def _get_from_postgres(
        self, agent_key: str,
    ) -> Optional[AgentConfigSchema]:
        """Get config from PostgreSQL."""
        conn = await self._acquire_connection()
        try:
            rows = await conn.fetch(_SELECT_SQL, agent_key)
            if not rows:
                return None
            row = rows[0]
            data = row["config_data"]
            if isinstance(data, str):
                data = json.loads(data)
            data = AgentConfigSchema.migrate(data)
            return AgentConfigSchema(**data)
        except Exception as exc:
            logger.error(
                "ConfigStore: Postgres read failed for '%s': %s",
                agent_key, exc,
            )
            return None
        finally:
            await self._release_connection(conn)

    async def _set_in_postgres(
        self,
        agent_key: str,
        config_data: Dict[str, Any],
        version: int,
    ) -> None:
        """Write config to PostgreSQL with optimistic locking."""
        from greenlang.infrastructure.agent_factory.config.schema import (
            CURRENT_SCHEMA_VERSION,
        )

        conn = await self._acquire_connection()
        try:
            await conn.execute(
                _UPSERT_SQL,
                agent_key,
                json.dumps(config_data, default=str),
                version,
                CURRENT_SCHEMA_VERSION,
                datetime.now(timezone.utc),
                config_data.get("updated_by", ""),
            )
        finally:
            await self._release_connection(conn)

    # ------------------------------------------------------------------
    # Connection Helpers
    # ------------------------------------------------------------------

    async def _acquire_connection(self) -> Any:
        """Acquire a connection from the pool."""
        if hasattr(self._pool, "connection"):
            ctx = self._pool.connection()
            conn = await ctx.__aenter__()
            conn._ctx = ctx  # type: ignore[attr-defined]
            return conn
        elif hasattr(self._pool, "acquire"):
            return await self._pool.acquire()
        raise TypeError(f"Unsupported pool type: {type(self._pool).__name__}")

    async def _release_connection(self, conn: Any) -> None:
        """Release a connection back to the pool."""
        if hasattr(conn, "_ctx"):
            await conn._ctx.__aexit__(None, None, None)
        elif hasattr(self._pool, "release"):
            await self._pool.release(conn)


__all__ = [
    "ConfigStore",
    "ConfigVersionConflictError",
]
