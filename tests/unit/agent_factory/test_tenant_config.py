# -*- coding: utf-8 -*-
"""
Unit tests for Multi-Tenant Configuration (INFRA-010 iteration).

Tests the AgentConfigSchema (version pinning, execution mode overrides,
quota/budget overrides, per-tenant isolation), the ConfigStore (two-layer
get/set/delete with optimistic locking), and the ConfigHotReload (change
notification publication and stale-version guard).

Coverage target: 85%+ of:
  - greenlang.infrastructure.agent_factory.config.schema
  - greenlang.infrastructure.agent_factory.config.store
  - greenlang.infrastructure.agent_factory.config.hot_reload
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from greenlang.infrastructure.agent_factory.config.schema import (
    CURRENT_SCHEMA_VERSION,
    AgentConfigSchema,
    CircuitBreakerConfigSchema,
    ResourceLimitsSchema,
    RetryConfigSchema,
)
from greenlang.infrastructure.agent_factory.config.store import (
    ConfigStore,
    ConfigVersionConflictError,
)
from greenlang.infrastructure.agent_factory.config.hot_reload import (
    ConfigChangeEvent,
    ConfigHotReload,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def base_config() -> AgentConfigSchema:
    """Create a base agent configuration for testing."""
    return AgentConfigSchema(
        agent_key="intake-agent",
        version=1,
        environment="dev",
        timeout_seconds=60.0,
        enabled=True,
    )


@pytest.fixture
def prod_config() -> AgentConfigSchema:
    """Create a production agent configuration."""
    return AgentConfigSchema(
        agent_key="intake-agent",
        version=2,
        environment="prod",
        timeout_seconds=30.0,
        log_level="WARNING",
        resource_limits=ResourceLimitsSchema(
            cpu_limit_cores=4.0,
            memory_limit_mb=4096,
            max_concurrent=100,
        ),
    )


@pytest.fixture
def mock_redis() -> AsyncMock:
    """Create a mock async Redis client."""
    redis = AsyncMock()
    redis.get = AsyncMock(return_value=None)
    redis.setex = AsyncMock()
    redis.delete = AsyncMock()
    redis.publish = AsyncMock(return_value=0)
    redis.pubsub = MagicMock()
    return redis


class _AsyncpgPoolSpec:
    """Minimal spec for an asyncpg-style pool.

    Used as ``spec=`` so that ``hasattr(pool, "connection")`` returns False
    while ``hasattr(pool, "acquire")`` returns True.  This steers
    ConfigStore._acquire_connection down the correct code path.
    """

    async def acquire(self): ...  # noqa: D102
    async def release(self, conn): ...  # noqa: D102


@pytest.fixture
def mock_db_pool() -> AsyncMock:
    """Create a mock async database pool with connection context.

    The mock uses ``spec=_AsyncpgPoolSpec`` so that only ``acquire`` and
    ``release`` attributes exist (no ``connection``), matching an asyncpg pool.
    """
    pool = AsyncMock(spec=_AsyncpgPoolSpec)

    # Shared mock connection returned by every pool.acquire() call
    conn = AsyncMock()
    conn.fetch = AsyncMock(return_value=[])
    conn.execute = AsyncMock(return_value="INSERT 1")

    pool.acquire = AsyncMock(return_value=conn)
    pool.release = AsyncMock()

    # Expose the connection mock so tests can configure it via pool._conn
    pool._conn = conn
    return pool


@pytest.fixture
def config_store(mock_redis: AsyncMock, mock_db_pool: AsyncMock) -> ConfigStore:
    """Create a ConfigStore with mocked Redis and PostgreSQL."""
    return ConfigStore(
        redis_client=mock_redis,
        db_pool=mock_db_pool,
        redis_ttl_s=300,
    )


@pytest.fixture
def hot_reload(mock_redis: AsyncMock) -> ConfigHotReload:
    """Create a ConfigHotReload with mocked Redis."""
    return ConfigHotReload(redis_client=mock_redis)


# ============================================================================
# TestTenantConfig (AgentConfigSchema)
# ============================================================================


class TestTenantConfig:
    """Tests for multi-tenant configuration via AgentConfigSchema."""

    def test_version_pinning(self, base_config: AgentConfigSchema) -> None:
        """Agent config tracks a version number for optimistic concurrency."""
        assert base_config.version == 1
        assert isinstance(base_config.version, int)
        assert base_config.version >= 1

    def test_execution_mode_override(self) -> None:
        """Custom settings can override execution mode per tenant."""
        config = AgentConfigSchema(
            agent_key="intake-agent",
            custom_settings={"execution_mode": "batch", "batch_size": 100},
        )
        assert config.custom_settings["execution_mode"] == "batch"
        assert config.custom_settings["batch_size"] == 100

    def test_quota_override(self) -> None:
        """Resource limits can be customized per tenant."""
        config = AgentConfigSchema(
            agent_key="intake-agent",
            resource_limits=ResourceLimitsSchema(
                cpu_limit_cores=2.0,
                memory_limit_mb=1024,
                max_concurrent=20,
                max_execution_seconds=120.0,
            ),
        )
        assert config.resource_limits.cpu_limit_cores == 2.0
        assert config.resource_limits.memory_limit_mb == 1024
        assert config.resource_limits.max_concurrent == 20

    def test_budget_override(self) -> None:
        """Custom settings can carry budget constraints per tenant."""
        config = AgentConfigSchema(
            agent_key="reasoning-agent",
            custom_settings={
                "monthly_budget_usd": 500.0,
                "cost_per_execution_usd": 0.05,
            },
        )
        assert config.custom_settings["monthly_budget_usd"] == 500.0
        assert config.custom_settings["cost_per_execution_usd"] == 0.05

    def test_config_overrides_merge(self) -> None:
        """Environment defaults are merged with explicit overrides."""
        config = AgentConfigSchema.with_environment_defaults(
            "intake-agent",
            environment="dev",
            timeout_seconds=90.0,  # Override the dev default of 120.0
        )
        assert config.timeout_seconds == 90.0
        assert config.environment == "dev"
        assert config.log_level == "DEBUG"  # From dev defaults

    def test_enable_disable_per_tenant(self) -> None:
        """Agents can be enabled or disabled per configuration."""
        enabled = AgentConfigSchema(agent_key="intake-agent", enabled=True)
        disabled = AgentConfigSchema(agent_key="intake-agent", enabled=False)
        assert enabled.enabled is True
        assert disabled.enabled is False

    def test_tenant_isolation(self) -> None:
        """Two configs for different agents are independent instances."""
        config_a = AgentConfigSchema(
            agent_key="agent-a",
            custom_settings={"tenant_id": "t-001"},
        )
        config_b = AgentConfigSchema(
            agent_key="agent-b",
            custom_settings={"tenant_id": "t-002"},
        )
        assert config_a.custom_settings["tenant_id"] != config_b.custom_settings["tenant_id"]
        assert config_a.agent_key != config_b.agent_key

    def test_default_when_no_override(self) -> None:
        """Config uses global defaults when no overrides are specified."""
        config = AgentConfigSchema(agent_key="intake-agent")
        assert config.timeout_seconds == 60.0
        assert config.log_level == "INFO"
        assert config.resource_limits.cpu_limit_cores == 4.0
        assert config.retry_config.max_attempts == 3

    def test_multiple_tenants_different_versions(self) -> None:
        """Different tenants can pin different config versions."""
        config_v1 = AgentConfigSchema(agent_key="intake-agent", version=1)
        config_v5 = AgentConfigSchema(agent_key="intake-agent", version=5)
        assert config_v1.version == 1
        assert config_v5.version == 5

    def test_create_tenant_config_with_environment_defaults(self) -> None:
        """with_environment_defaults factory applies environment-specific settings."""
        config = AgentConfigSchema.with_environment_defaults(
            "intake-agent", environment="prod"
        )
        assert config.environment == "prod"
        assert config.timeout_seconds == 60.0  # Prod default
        assert config.log_level == "WARNING"  # Prod default

    def test_update_tenant_config(self, base_config: AgentConfigSchema) -> None:
        """Config fields can be updated by creating a new instance."""
        updated_data = base_config.model_dump()
        updated_data["version"] = 2
        updated_data["timeout_seconds"] = 90.0
        updated_data["updated_by"] = "admin"

        updated = AgentConfigSchema(**updated_data)
        assert updated.version == 2
        assert updated.timeout_seconds == 90.0
        assert updated.updated_by == "admin"

    def test_delete_tenant_config_concept(self) -> None:
        """Deletion is handled by the ConfigStore, not the schema itself."""
        config = AgentConfigSchema(agent_key="to-delete-agent")
        assert config.agent_key == "to-delete-agent"
        # Schema itself has no delete method; deletion is a store operation

    def test_feature_flags_per_tenant(self) -> None:
        """Feature flags can be overridden per agent config."""
        config = AgentConfigSchema(
            agent_key="intake-agent",
            feature_flags={
                "use_v2_engine": True,
                "enable_caching": False,
            },
        )
        assert config.feature_flags["use_v2_engine"] is True
        assert config.feature_flags["enable_caching"] is False

    def test_tags_deduplication(self) -> None:
        """Tags are deduplicated and lowercased."""
        config = AgentConfigSchema(
            agent_key="intake-agent",
            tags=["Emissions", "CBAM", "emissions", "cbam"],
        )
        assert config.tags == ["emissions", "cbam"]

    def test_environment_normalization(self) -> None:
        """Environment names are normalized (production -> prod, etc.)."""
        config = AgentConfigSchema(agent_key="intake-agent", environment="production")
        assert config.environment == "prod"

        config2 = AgentConfigSchema(agent_key="intake-agent", environment="development")
        assert config2.environment == "dev"

    def test_invalid_agent_key_raises_error(self) -> None:
        """Invalid agent_key format raises a validation error."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            AgentConfigSchema(agent_key="INVALID_KEY")

    def test_invalid_environment_raises_error(self) -> None:
        """Invalid environment raises a validation error."""
        with pytest.raises(Exception):
            AgentConfigSchema(agent_key="intake-agent", environment="jupiter")

    def test_invalid_log_level_raises_error(self) -> None:
        """Invalid log level raises a validation error."""
        with pytest.raises(Exception):
            AgentConfigSchema(agent_key="intake-agent", log_level="VERBOSE")


# ============================================================================
# TestConfigStore
# ============================================================================


class TestConfigStore:
    """Tests for the two-layer ConfigStore (Redis L1 + PostgreSQL L2)."""

    @pytest.mark.asyncio
    async def test_get_config_l1_hit(
        self, config_store: ConfigStore, mock_redis: AsyncMock, base_config: AgentConfigSchema
    ) -> None:
        """Config is returned from Redis (L1) when cached."""
        cached_data = base_config.model_dump(mode="json")
        mock_redis.get = AsyncMock(return_value=json.dumps(cached_data, default=str))

        result = await config_store.get_config("intake-agent")
        assert result is not None
        assert result.agent_key == "intake-agent"
        mock_redis.get.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_config_l2_hit(
        self, config_store: ConfigStore, mock_redis: AsyncMock, mock_db_pool: AsyncMock,
        base_config: AgentConfigSchema
    ) -> None:
        """Config is fetched from PostgreSQL (L2) when Redis cache misses."""
        mock_redis.get = AsyncMock(return_value=None)  # L1 miss

        conn = mock_db_pool._conn
        config_data = base_config.model_dump(mode="json")
        conn.fetch = AsyncMock(return_value=[{
            "agent_key": "intake-agent",
            "config_data": config_data,
            "version": 1,
            "schema_version": 1,
            "updated_at": datetime.now(timezone.utc),
            "updated_by": "admin",
        }])

        result = await config_store.get_config("intake-agent")
        assert result is not None
        assert result.agent_key == "intake-agent"
        # L1 should be populated after L2 hit
        mock_redis.setex.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_config_not_found(
        self, config_store: ConfigStore, mock_redis: AsyncMock, mock_db_pool: AsyncMock
    ) -> None:
        """get_config returns None when config is not in either layer."""
        mock_redis.get = AsyncMock(return_value=None)
        conn = mock_db_pool._conn
        conn.fetch = AsyncMock(return_value=[])

        result = await config_store.get_config("nonexistent-agent")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_config_writes_to_both_layers(
        self, config_store: ConfigStore, mock_redis: AsyncMock, mock_db_pool: AsyncMock,
        base_config: AgentConfigSchema
    ) -> None:
        """set_config writes to both Redis and PostgreSQL."""
        conn = mock_db_pool._conn
        conn.fetch = AsyncMock(return_value=[])  # No existing config

        result = await config_store.set_config(
            "intake-agent", base_config, changed_by="admin"
        )
        assert result is not None
        assert result.agent_key == "intake-agent"

        # Verify L2 write
        conn.execute.assert_awaited()
        # Verify L1 write
        mock_redis.setex.assert_awaited()

    @pytest.mark.asyncio
    async def test_set_config_version_conflict(
        self, config_store: ConfigStore, mock_db_pool: AsyncMock
    ) -> None:
        """set_config raises ConfigVersionConflictError on stale version."""
        existing_config = AgentConfigSchema(
            agent_key="intake-agent",
            version=5,
        )
        conn = mock_db_pool._conn
        conn.fetch = AsyncMock(return_value=[{
            "agent_key": "intake-agent",
            "config_data": existing_config.model_dump(mode="json"),
            "version": 5,
            "schema_version": 1,
            "updated_at": datetime.now(timezone.utc),
            "updated_by": "admin",
        }])

        stale_config = AgentConfigSchema(
            agent_key="intake-agent",
            version=3,  # Stale -- less than existing version 5
        )

        with pytest.raises(ConfigVersionConflictError) as exc_info:
            await config_store.set_config("intake-agent", stale_config)

        assert exc_info.value.agent_key == "intake-agent"
        assert exc_info.value.expected_version == 3
        assert exc_info.value.actual_version == 5

    @pytest.mark.asyncio
    async def test_delete_config(
        self, config_store: ConfigStore, mock_redis: AsyncMock, mock_db_pool: AsyncMock
    ) -> None:
        """delete_config removes from both Redis and PostgreSQL."""
        conn = mock_db_pool._conn
        conn.execute = AsyncMock(return_value="DELETE 1")

        deleted = await config_store.delete_config("intake-agent")
        assert deleted is True

        mock_redis.delete.assert_awaited_once()
        conn.execute.assert_awaited()

    @pytest.mark.asyncio
    async def test_delete_config_not_found(
        self, config_store: ConfigStore, mock_redis: AsyncMock, mock_db_pool: AsyncMock
    ) -> None:
        """delete_config returns False when config does not exist."""
        conn = mock_db_pool._conn
        conn.execute = AsyncMock(return_value="DELETE 0")

        deleted = await config_store.delete_config("nonexistent-agent")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_redis_read_failure_falls_through_to_postgres(
        self, config_store: ConfigStore, mock_redis: AsyncMock, mock_db_pool: AsyncMock,
        base_config: AgentConfigSchema
    ) -> None:
        """Redis failure falls through to PostgreSQL gracefully."""
        mock_redis.get = AsyncMock(side_effect=ConnectionError("Redis down"))

        conn = mock_db_pool._conn
        config_data = base_config.model_dump(mode="json")
        conn.fetch = AsyncMock(return_value=[{
            "agent_key": "intake-agent",
            "config_data": config_data,
            "version": 1,
            "schema_version": 1,
            "updated_at": datetime.now(timezone.utc),
            "updated_by": "system",
        }])

        result = await config_store.get_config("intake-agent")
        assert result is not None
        assert result.agent_key == "intake-agent"


# ============================================================================
# TestConfigHotReload
# ============================================================================


class TestConfigHotReload:
    """Tests for ConfigHotReload change notification system."""

    def test_register_callback(self, hot_reload: ConfigHotReload) -> None:
        """Registering a callback stores it for the agent."""
        async def on_change(event: ConfigChangeEvent) -> None:
            pass

        hot_reload.register("intake-agent", on_change)
        snapshot = hot_reload.snapshot()
        assert "intake-agent" in snapshot["registered_agents"]

    def test_register_global_callback(self, hot_reload: ConfigHotReload) -> None:
        """Global callbacks are registered separately."""
        async def on_any_change(event: ConfigChangeEvent) -> None:
            pass

        hot_reload.register_global(on_any_change)
        snapshot = hot_reload.snapshot()
        assert snapshot["global_callbacks"] == 1

    def test_unregister_callback(self, hot_reload: ConfigHotReload) -> None:
        """Unregistering removes all callbacks for an agent."""
        async def on_change(event: ConfigChangeEvent) -> None:
            pass

        hot_reload.register("intake-agent", on_change)
        removed = hot_reload.unregister("intake-agent")
        assert removed == 1

        snapshot = hot_reload.snapshot()
        assert "intake-agent" not in snapshot["registered_agents"]

    def test_unregister_unknown_agent(self, hot_reload: ConfigHotReload) -> None:
        """Unregistering unknown agent returns 0."""
        removed = hot_reload.unregister("nonexistent")
        assert removed == 0

    @pytest.mark.asyncio
    async def test_publish_change_event(
        self, hot_reload: ConfigHotReload, mock_redis: AsyncMock
    ) -> None:
        """Publishing a change event publishes JSON to Redis channel."""
        mock_redis.publish = AsyncMock(return_value=2)

        event = ConfigChangeEvent(
            agent_key="intake-agent",
            old_config={"timeout_seconds": 60},
            new_config={"timeout_seconds": 90},
            changed_keys=frozenset(["timeout_seconds"]),
            config_version=2,
            changed_by="admin",
        )

        receivers = await hot_reload.publish_change(event)
        assert receivers == 2
        mock_redis.publish.assert_awaited_once()

        # Verify the published payload
        call_args = mock_redis.publish.call_args
        payload = json.loads(call_args[0][1])
        assert payload["agent_key"] == "intake-agent"
        assert payload["config_version"] == 2

    @pytest.mark.asyncio
    async def test_publish_skips_stale_version(
        self, hot_reload: ConfigHotReload, mock_redis: AsyncMock
    ) -> None:
        """Publishing a stale version is skipped."""
        # First publish (version 3)
        event_v3 = ConfigChangeEvent(
            agent_key="intake-agent", config_version=3,
        )
        await hot_reload.publish_change(event_v3)

        # Attempt stale publish (version 2)
        event_v2 = ConfigChangeEvent(
            agent_key="intake-agent", config_version=2,
        )
        receivers = await hot_reload.publish_change(event_v2)
        assert receivers == 0  # Skipped

    def test_snapshot_diagnostic(self, hot_reload: ConfigHotReload) -> None:
        """Snapshot returns diagnostic info about the hot reload state."""
        snapshot = hot_reload.snapshot()
        assert "channel" in snapshot
        assert "running" in snapshot
        assert "registered_agents" in snapshot
        assert "global_callbacks" in snapshot
        assert "known_versions" in snapshot
        assert snapshot["running"] is False  # Not started yet

    def test_default_channel(self, hot_reload: ConfigHotReload) -> None:
        """Default channel is gl:config:changes."""
        assert hot_reload.channel == "gl:config:changes"

    def test_custom_channel(self, mock_redis: AsyncMock) -> None:
        """Custom channel name is respected."""
        reload = ConfigHotReload(
            redis_client=mock_redis,
            channel="custom:config:channel",
        )
        assert reload.channel == "custom:config:channel"

    def test_is_running_initially_false(self, hot_reload: ConfigHotReload) -> None:
        """Hot reload is not running until start() is called."""
        assert hot_reload.is_running is False


# ============================================================================
# TestConfigChangeEvent
# ============================================================================


class TestConfigChangeEvent:
    """Tests for the ConfigChangeEvent dataclass."""

    def test_event_creation(self) -> None:
        """ConfigChangeEvent captures all fields."""
        event = ConfigChangeEvent(
            agent_key="intake-agent",
            old_config={"timeout_seconds": 60},
            new_config={"timeout_seconds": 90},
            changed_keys=frozenset(["timeout_seconds"]),
            config_version=2,
            changed_by="admin",
        )
        assert event.agent_key == "intake-agent"
        assert event.config_version == 2
        assert "timeout_seconds" in event.changed_keys
        assert event.changed_by == "admin"

    def test_event_is_frozen(self) -> None:
        """ConfigChangeEvent is immutable (frozen dataclass)."""
        event = ConfigChangeEvent(agent_key="intake-agent")
        with pytest.raises(AttributeError):
            event.agent_key = "other-agent"  # type: ignore[misc]

    def test_event_timestamp_is_utc(self) -> None:
        """ConfigChangeEvent timestamp is UTC."""
        event = ConfigChangeEvent(agent_key="intake-agent")
        assert event.timestamp.tzinfo is not None

    def test_event_defaults(self) -> None:
        """ConfigChangeEvent has sensible defaults."""
        event = ConfigChangeEvent(agent_key="intake-agent")
        assert event.old_config == {}
        assert event.new_config == {}
        assert event.changed_keys == frozenset()
        assert event.config_version == 0
        assert event.changed_by == ""


# ============================================================================
# TestSchemaValidation
# ============================================================================


class TestSchemaValidation:
    """Tests for AgentConfigSchema validation rules."""

    def test_agent_key_valid_patterns(self) -> None:
        """Valid agent key patterns are accepted."""
        valid_keys = [
            "intake-agent",
            "emissions.calc",
            "cbam_scope1_v2",
            "a123",
        ]
        for key in valid_keys:
            config = AgentConfigSchema(agent_key=key)
            assert config.agent_key == key

    def test_retry_config_defaults(self) -> None:
        """Retry config has sensible defaults."""
        config = AgentConfigSchema(agent_key="intake-agent")
        assert config.retry_config.max_attempts == 3
        assert config.retry_config.base_delay_s == 1.0
        assert config.retry_config.backoff_multiplier == 2.0

    def test_circuit_breaker_config_defaults(self) -> None:
        """Circuit breaker config has sensible defaults."""
        config = AgentConfigSchema(agent_key="intake-agent")
        assert config.circuit_breaker_config.failure_rate_threshold == 0.5
        assert config.circuit_breaker_config.wait_in_open_s == 60.0

    def test_schema_version_is_current(self) -> None:
        """New configs use the current schema version."""
        config = AgentConfigSchema(agent_key="intake-agent")
        assert config.schema_version == CURRENT_SCHEMA_VERSION

    def test_schema_migration_from_v0(self) -> None:
        """Migrating from v0 adds required fields."""
        raw_data = {
            "agent_key": "legacy-agent",
            "schema_version": 0,
        }
        migrated = AgentConfigSchema.migrate(raw_data)
        assert migrated["schema_version"] == CURRENT_SCHEMA_VERSION
        assert "resource_limits" in migrated
        assert "retry_config" in migrated
        assert "circuit_breaker_config" in migrated

    def test_schema_migration_noop_for_current(self) -> None:
        """Migrating from current version is a no-op."""
        raw_data = {
            "agent_key": "modern-agent",
            "schema_version": CURRENT_SCHEMA_VERSION,
        }
        migrated = AgentConfigSchema.migrate(raw_data)
        assert migrated == raw_data

    def test_model_dump_json_serializable(self, base_config: AgentConfigSchema) -> None:
        """model_dump(mode='json') produces JSON-serializable output."""
        data = base_config.model_dump(mode="json")
        json_str = json.dumps(data, default=str)
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["agent_key"] == "intake-agent"
