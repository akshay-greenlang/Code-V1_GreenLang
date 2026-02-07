# -*- coding: utf-8 -*-
"""
Unit tests for Agent Factory Hot Reload: configuration schema validation,
config store get/set, Redis caching, hot reload callbacks, config diff
detection, and rollback.
"""

from __future__ import annotations

import asyncio
import copy
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set
from unittest.mock import AsyncMock, MagicMock

import pytest


# ============================================================================
# Inline Implementations (contract definitions)
# ============================================================================


class ConfigValidationError(Exception):
    def __init__(self, errors: List[str]) -> None:
        self.errors = errors
        super().__init__(f"Config validation failed: {errors}")


@dataclass
class ConfigSchema:
    fields: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    """Mapping of field_name -> {type, required, default, min, max}"""

    def validate(self, config: Dict[str, Any]) -> List[str]:
        errors: List[str] = []
        for name, spec in self.fields.items():
            if spec.get("required", False) and name not in config:
                errors.append(f"Missing required field: {name}")
                continue
            if name in config:
                expected_type = spec.get("type")
                value = config[name]
                if expected_type == "int" and not isinstance(value, int):
                    errors.append(f"Field '{name}' must be int, got {type(value).__name__}")
                elif expected_type == "float" and not isinstance(value, (int, float)):
                    errors.append(f"Field '{name}' must be float, got {type(value).__name__}")
                elif expected_type == "str" and not isinstance(value, str):
                    errors.append(f"Field '{name}' must be str, got {type(value).__name__}")
                if "min" in spec and isinstance(value, (int, float)):
                    if value < spec["min"]:
                        errors.append(f"Field '{name}' is below minimum {spec['min']}")
                if "max" in spec and isinstance(value, (int, float)):
                    if value > spec["max"]:
                        errors.append(f"Field '{name}' exceeds maximum {spec['max']}")
        return errors

    def apply_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(config)
        for name, spec in self.fields.items():
            if name not in result and "default" in spec:
                result[name] = spec["default"]
        return result


@dataclass
class ConfigDiff:
    added: Dict[str, Any] = field(default_factory=dict)
    removed: Dict[str, Any] = field(default_factory=dict)
    changed: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @property
    def has_changes(self) -> bool:
        return bool(self.added or self.removed or self.changed)


def compute_diff(old: Dict[str, Any], new: Dict[str, Any]) -> ConfigDiff:
    added = {k: v for k, v in new.items() if k not in old}
    removed = {k: v for k, v in old.items() if k not in new}
    changed = {}
    for k in old:
        if k in new and old[k] != new[k]:
            changed[k] = {"old": old[k], "new": new[k]}
    return ConfigDiff(added=added, removed=removed, changed=changed)


ReloadCallback = Callable[[str, Dict[str, Any], ConfigDiff], Coroutine[Any, Any, None]]


class ConfigStore:
    def __init__(
        self,
        schema: Optional[ConfigSchema] = None,
        redis_client: Optional[Any] = None,
    ) -> None:
        self._schema = schema
        self._redis = redis_client
        self._store: Dict[str, Dict[str, Any]] = {}
        self._history: Dict[str, List[Dict[str, Any]]] = {}
        self._callbacks: List[ReloadCallback] = []

    def on_reload(self, callback: ReloadCallback) -> None:
        self._callbacks.append(callback)

    async def get(self, agent_key: str) -> Optional[Dict[str, Any]]:
        # Try redis cache first
        if self._redis:
            cached = await self._redis.get(f"config:{agent_key}")
            if cached is not None:
                return cached
        return self._store.get(agent_key)

    async def set(
        self,
        agent_key: str,
        config: Dict[str, Any],
        validate: bool = True,
    ) -> None:
        if validate and self._schema:
            errors = self._schema.validate(config)
            if errors:
                raise ConfigValidationError(errors)

        old_config = self._store.get(agent_key, {})
        if self._schema:
            config = self._schema.apply_defaults(config)

        # Track history
        if agent_key in self._store:
            self._history.setdefault(agent_key, []).append(
                copy.deepcopy(self._store[agent_key])
            )

        self._store[agent_key] = config

        # Update redis cache
        if self._redis:
            await self._redis.set(f"config:{agent_key}", config)

        # Compute diff and invoke callbacks
        diff = compute_diff(old_config, config)
        if diff.has_changes:
            for cb in self._callbacks:
                try:
                    await cb(agent_key, config, diff)
                except Exception:
                    pass

    async def rollback(self, agent_key: str) -> bool:
        history = self._history.get(agent_key, [])
        if not history:
            return False
        previous = history.pop()
        self._store[agent_key] = previous
        if self._redis:
            await self._redis.set(f"config:{agent_key}", previous)
        return True


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def schema() -> ConfigSchema:
    return ConfigSchema(fields={
        "timeout_seconds": {"type": "int", "required": True, "min": 1, "max": 3600},
        "max_retries": {"type": "int", "required": False, "default": 3, "min": 0},
        "log_level": {"type": "str", "required": False, "default": "INFO"},
    })


@pytest.fixture
def mock_redis() -> MagicMock:
    redis = MagicMock()
    redis.get = AsyncMock(return_value=None)
    redis.set = AsyncMock()
    return redis


@pytest.fixture
def config_store(schema: ConfigSchema, mock_redis: MagicMock) -> ConfigStore:
    return ConfigStore(schema=schema, redis_client=mock_redis)


# ============================================================================
# Tests
# ============================================================================


class TestConfigSchema:
    """Tests for configuration schema validation."""

    def test_config_schema_validation_valid(self, schema: ConfigSchema) -> None:
        """Valid config passes schema validation."""
        config = {"timeout_seconds": 30, "max_retries": 5}
        errors = schema.validate(config)
        assert errors == []

    def test_config_schema_validation_missing_required(
        self, schema: ConfigSchema
    ) -> None:
        """Missing required field is reported."""
        config = {"max_retries": 3}
        errors = schema.validate(config)
        assert len(errors) == 1
        assert "timeout_seconds" in errors[0]

    def test_config_schema_validation_wrong_type(
        self, schema: ConfigSchema
    ) -> None:
        """Wrong type is reported."""
        config = {"timeout_seconds": "not_an_int"}
        errors = schema.validate(config)
        assert any("must be int" in e for e in errors)

    def test_config_schema_validation_min_max(
        self, schema: ConfigSchema
    ) -> None:
        """Values outside min/max range are reported."""
        config = {"timeout_seconds": 0}
        errors = schema.validate(config)
        assert any("below minimum" in e for e in errors)

        config2 = {"timeout_seconds": 9999}
        errors2 = schema.validate(config2)
        assert any("exceeds maximum" in e for e in errors2)

    def test_config_schema_defaults(self, schema: ConfigSchema) -> None:
        """Defaults are applied for missing optional fields."""
        config = {"timeout_seconds": 30}
        result = schema.apply_defaults(config)
        assert result["max_retries"] == 3
        assert result["log_level"] == "INFO"
        assert result["timeout_seconds"] == 30


class TestConfigStore:
    """Tests for the config store get/set operations."""

    @pytest.mark.asyncio
    async def test_config_store_get_set(
        self, config_store: ConfigStore
    ) -> None:
        """Setting and getting a config works."""
        await config_store.set("agent-a", {"timeout_seconds": 60})
        result = await config_store.get("agent-a")
        assert result is not None
        assert result["timeout_seconds"] == 60
        assert result["max_retries"] == 3  # default applied

    @pytest.mark.asyncio
    async def test_config_store_redis_cache(
        self, config_store: ConfigStore, mock_redis: MagicMock
    ) -> None:
        """Config is written to Redis on set."""
        await config_store.set("agent-a", {"timeout_seconds": 30})
        mock_redis.set.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_config_store_redis_cache_read(
        self, config_store: ConfigStore, mock_redis: MagicMock
    ) -> None:
        """Config is read from Redis if available."""
        mock_redis.get = AsyncMock(return_value={"timeout_seconds": 99})
        result = await config_store.get("agent-cached")
        assert result == {"timeout_seconds": 99}

    @pytest.mark.asyncio
    async def test_hot_reload_callback_invoked(
        self, config_store: ConfigStore
    ) -> None:
        """Callbacks are invoked on config change."""
        callback = AsyncMock()
        config_store.on_reload(callback)

        await config_store.set("agent-a", {"timeout_seconds": 30})
        await config_store.set("agent-a", {"timeout_seconds": 60})

        assert callback.await_count >= 1
        _, args, _ = callback.mock_calls[-1]
        agent_key, new_config, diff = args
        assert agent_key == "agent-a"
        assert diff.has_changes is True

    @pytest.mark.asyncio
    async def test_hot_reload_validation_rejects_invalid(
        self, config_store: ConfigStore
    ) -> None:
        """Invalid config is rejected during hot reload."""
        with pytest.raises(ConfigValidationError):
            await config_store.set("agent-a", {"timeout_seconds": -1})


class TestConfigDiff:
    """Tests for configuration diff computation."""

    def test_config_diff_detect_changes(self) -> None:
        """Changed values are detected."""
        diff = compute_diff(
            {"timeout": 30, "retries": 3},
            {"timeout": 60, "retries": 3},
        )
        assert diff.has_changes is True
        assert "timeout" in diff.changed
        assert diff.changed["timeout"]["old"] == 30
        assert diff.changed["timeout"]["new"] == 60

    def test_config_diff_added_field(self) -> None:
        """Added fields are detected."""
        diff = compute_diff(
            {"timeout": 30},
            {"timeout": 30, "new_field": "value"},
        )
        assert "new_field" in diff.added
        assert diff.has_changes is True

    def test_config_diff_removed_field(self) -> None:
        """Removed fields are detected."""
        diff = compute_diff(
            {"timeout": 30, "old_field": "value"},
            {"timeout": 30},
        )
        assert "old_field" in diff.removed
        assert diff.has_changes is True

    def test_config_diff_no_changes(self) -> None:
        """Identical configs produce no diff."""
        diff = compute_diff(
            {"timeout": 30, "retries": 3},
            {"timeout": 30, "retries": 3},
        )
        assert diff.has_changes is False


class TestConfigRollback:
    """Tests for config rollback functionality."""

    @pytest.mark.asyncio
    async def test_config_rollback(
        self, config_store: ConfigStore
    ) -> None:
        """Rollback restores the previous configuration."""
        await config_store.set("agent-a", {"timeout_seconds": 30})
        await config_store.set("agent-a", {"timeout_seconds": 60})

        success = await config_store.rollback("agent-a")
        assert success is True

        result = await config_store.get("agent-a")
        assert result is not None
        assert result["timeout_seconds"] == 30

    @pytest.mark.asyncio
    async def test_config_rollback_no_history(
        self, config_store: ConfigStore
    ) -> None:
        """Rollback with no history returns False."""
        success = await config_store.rollback("unknown-agent")
        assert success is False
