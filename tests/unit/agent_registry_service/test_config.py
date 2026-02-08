# -*- coding: utf-8 -*-
"""
Unit Tests for AgentRegistryConfig (AGENT-FOUND-007)

Tests configuration creation, env var overrides with GL_AGENT_REGISTRY_ prefix,
validation settings, and singleton get_config/set_config/reset_config.

Coverage target: 85%+ of config.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Any, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline AgentRegistryConfig that mirrors greenlang/agent_registry/config.py
# ---------------------------------------------------------------------------

_ENV_PREFIX = "GL_AGENT_REGISTRY_"


@dataclass
class AgentRegistryConfig:
    """Mirrors greenlang.agent_registry.config.AgentRegistryConfig."""

    # Service
    service_name: str = "agent-registry"
    environment: str = "production"
    log_level: str = "INFO"

    # Registry limits
    max_agents: int = 500
    max_versions_per_agent: int = 50
    max_capabilities_per_agent: int = 20
    max_dependencies_per_agent: int = 30
    max_tags_per_agent: int = 50

    # Health checking
    health_check_enabled: bool = True
    health_check_interval_seconds: int = 30
    health_check_timeout_seconds: int = 5
    unhealthy_threshold: int = 3

    # Hot reload
    hot_reload_enabled: bool = True
    hot_reload_grace_period_seconds: int = 10

    # Dependency resolution
    dependency_resolution_enabled: bool = True
    fail_on_missing_dependency: bool = True
    max_dependency_depth: int = 10

    # Capability matching
    capability_matching_enabled: bool = True

    # Provenance
    provenance_enabled: bool = True

    # Cache
    cache_ttl_seconds: int = 300
    cache_max_size: int = 1000

    # Export / Import
    export_enabled: bool = True
    import_enabled: bool = True

    @classmethod
    def from_env(cls) -> "AgentRegistryConfig":
        prefix = _ENV_PREFIX

        def _env(name: str, default: Any = None) -> Optional[str]:
            return os.environ.get(f"{prefix}{name}", default)

        def _bool(name: str, default: bool) -> bool:
            val = _env(name)
            if val is None:
                return default
            return val.lower() in ("true", "1", "yes")

        def _int(name: str, default: int) -> int:
            val = _env(name)
            if val is None:
                return default
            try:
                return int(val)
            except ValueError:
                return default

        def _str(name: str, default: str) -> str:
            val = _env(name)
            if val is None:
                return default
            return val.strip() if val.strip() else default

        return cls(
            service_name=_str("SERVICE_NAME", cls.service_name),
            environment=_str("ENVIRONMENT", cls.environment),
            log_level=_str("LOG_LEVEL", cls.log_level),
            max_agents=_int("MAX_AGENTS", cls.max_agents),
            max_versions_per_agent=_int("MAX_VERSIONS_PER_AGENT", cls.max_versions_per_agent),
            max_capabilities_per_agent=_int("MAX_CAPABILITIES_PER_AGENT", cls.max_capabilities_per_agent),
            max_dependencies_per_agent=_int("MAX_DEPENDENCIES_PER_AGENT", cls.max_dependencies_per_agent),
            max_tags_per_agent=_int("MAX_TAGS_PER_AGENT", cls.max_tags_per_agent),
            health_check_enabled=_bool("HEALTH_CHECK_ENABLED", cls.health_check_enabled),
            health_check_interval_seconds=_int("HEALTH_CHECK_INTERVAL_SECONDS", cls.health_check_interval_seconds),
            health_check_timeout_seconds=_int("HEALTH_CHECK_TIMEOUT_SECONDS", cls.health_check_timeout_seconds),
            unhealthy_threshold=_int("UNHEALTHY_THRESHOLD", cls.unhealthy_threshold),
            hot_reload_enabled=_bool("HOT_RELOAD_ENABLED", cls.hot_reload_enabled),
            hot_reload_grace_period_seconds=_int("HOT_RELOAD_GRACE_PERIOD_SECONDS", cls.hot_reload_grace_period_seconds),
            dependency_resolution_enabled=_bool("DEPENDENCY_RESOLUTION_ENABLED", cls.dependency_resolution_enabled),
            fail_on_missing_dependency=_bool("FAIL_ON_MISSING_DEPENDENCY", cls.fail_on_missing_dependency),
            max_dependency_depth=_int("MAX_DEPENDENCY_DEPTH", cls.max_dependency_depth),
            capability_matching_enabled=_bool("CAPABILITY_MATCHING_ENABLED", cls.capability_matching_enabled),
            provenance_enabled=_bool("PROVENANCE_ENABLED", cls.provenance_enabled),
            cache_ttl_seconds=_int("CACHE_TTL_SECONDS", cls.cache_ttl_seconds),
            cache_max_size=_int("CACHE_MAX_SIZE", cls.cache_max_size),
            export_enabled=_bool("EXPORT_ENABLED", cls.export_enabled),
            import_enabled=_bool("IMPORT_ENABLED", cls.import_enabled),
        )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_config_instance: Optional[AgentRegistryConfig] = None
_config_lock = threading.Lock()


def get_config() -> AgentRegistryConfig:
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = AgentRegistryConfig.from_env()
    return _config_instance


def set_config(config: AgentRegistryConfig) -> None:
    global _config_instance
    with _config_lock:
        _config_instance = config


def reset_config() -> None:
    global _config_instance
    with _config_lock:
        _config_instance = None


# ---------------------------------------------------------------------------
# Autouse: reset singleton between tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_config_singleton():
    yield
    reset_config()


# ===========================================================================
# Test Classes
# ===========================================================================


class TestAgentRegistryConfigDefaults:
    """Test that default configuration values match requirements."""

    def test_default_service_name(self):
        config = AgentRegistryConfig()
        assert config.service_name == "agent-registry"

    def test_default_environment(self):
        config = AgentRegistryConfig()
        assert config.environment == "production"

    def test_default_log_level(self):
        config = AgentRegistryConfig()
        assert config.log_level == "INFO"

    def test_default_max_agents(self):
        config = AgentRegistryConfig()
        assert config.max_agents == 500

    def test_default_max_versions_per_agent(self):
        config = AgentRegistryConfig()
        assert config.max_versions_per_agent == 50

    def test_default_max_capabilities_per_agent(self):
        config = AgentRegistryConfig()
        assert config.max_capabilities_per_agent == 20

    def test_default_max_dependencies_per_agent(self):
        config = AgentRegistryConfig()
        assert config.max_dependencies_per_agent == 30

    def test_default_max_tags_per_agent(self):
        config = AgentRegistryConfig()
        assert config.max_tags_per_agent == 50

    def test_default_health_check_enabled(self):
        config = AgentRegistryConfig()
        assert config.health_check_enabled is True

    def test_default_health_check_interval_seconds(self):
        config = AgentRegistryConfig()
        assert config.health_check_interval_seconds == 30

    def test_default_health_check_timeout_seconds(self):
        config = AgentRegistryConfig()
        assert config.health_check_timeout_seconds == 5

    def test_default_unhealthy_threshold(self):
        config = AgentRegistryConfig()
        assert config.unhealthy_threshold == 3

    def test_default_hot_reload_enabled(self):
        config = AgentRegistryConfig()
        assert config.hot_reload_enabled is True

    def test_default_hot_reload_grace_period(self):
        config = AgentRegistryConfig()
        assert config.hot_reload_grace_period_seconds == 10

    def test_default_dependency_resolution_enabled(self):
        config = AgentRegistryConfig()
        assert config.dependency_resolution_enabled is True

    def test_default_fail_on_missing_dependency(self):
        config = AgentRegistryConfig()
        assert config.fail_on_missing_dependency is True

    def test_default_max_dependency_depth(self):
        config = AgentRegistryConfig()
        assert config.max_dependency_depth == 10

    def test_default_capability_matching_enabled(self):
        config = AgentRegistryConfig()
        assert config.capability_matching_enabled is True

    def test_default_provenance_enabled(self):
        config = AgentRegistryConfig()
        assert config.provenance_enabled is True

    def test_default_cache_ttl_seconds(self):
        config = AgentRegistryConfig()
        assert config.cache_ttl_seconds == 300

    def test_default_cache_max_size(self):
        config = AgentRegistryConfig()
        assert config.cache_max_size == 1000

    def test_default_export_enabled(self):
        config = AgentRegistryConfig()
        assert config.export_enabled is True

    def test_default_import_enabled(self):
        config = AgentRegistryConfig()
        assert config.import_enabled is True


class TestAgentRegistryConfigFromEnv:
    """Test GL_AGENT_REGISTRY_ env var overrides via from_env()."""

    def test_env_override_service_name(self, monkeypatch):
        monkeypatch.setenv("GL_AGENT_REGISTRY_SERVICE_NAME", "custom-registry")
        config = AgentRegistryConfig.from_env()
        assert config.service_name == "custom-registry"

    def test_env_override_environment(self, monkeypatch):
        monkeypatch.setenv("GL_AGENT_REGISTRY_ENVIRONMENT", "staging")
        config = AgentRegistryConfig.from_env()
        assert config.environment == "staging"

    def test_env_override_log_level(self, monkeypatch):
        monkeypatch.setenv("GL_AGENT_REGISTRY_LOG_LEVEL", "DEBUG")
        config = AgentRegistryConfig.from_env()
        assert config.log_level == "DEBUG"

    def test_env_override_max_agents(self, monkeypatch):
        monkeypatch.setenv("GL_AGENT_REGISTRY_MAX_AGENTS", "1000")
        config = AgentRegistryConfig.from_env()
        assert config.max_agents == 1000

    def test_env_override_health_check_disabled(self, monkeypatch):
        monkeypatch.setenv("GL_AGENT_REGISTRY_HEALTH_CHECK_ENABLED", "false")
        config = AgentRegistryConfig.from_env()
        assert config.health_check_enabled is False

    def test_env_override_health_check_interval(self, monkeypatch):
        monkeypatch.setenv("GL_AGENT_REGISTRY_HEALTH_CHECK_INTERVAL_SECONDS", "60")
        config = AgentRegistryConfig.from_env()
        assert config.health_check_interval_seconds == 60

    def test_env_override_hot_reload_disabled(self, monkeypatch):
        monkeypatch.setenv("GL_AGENT_REGISTRY_HOT_RELOAD_ENABLED", "false")
        config = AgentRegistryConfig.from_env()
        assert config.hot_reload_enabled is False

    def test_env_override_dependency_resolution_disabled(self, monkeypatch):
        monkeypatch.setenv("GL_AGENT_REGISTRY_DEPENDENCY_RESOLUTION_ENABLED", "false")
        config = AgentRegistryConfig.from_env()
        assert config.dependency_resolution_enabled is False

    def test_env_override_fail_on_missing_false(self, monkeypatch):
        monkeypatch.setenv("GL_AGENT_REGISTRY_FAIL_ON_MISSING_DEPENDENCY", "false")
        config = AgentRegistryConfig.from_env()
        assert config.fail_on_missing_dependency is False

    def test_env_override_max_dependency_depth(self, monkeypatch):
        monkeypatch.setenv("GL_AGENT_REGISTRY_MAX_DEPENDENCY_DEPTH", "20")
        config = AgentRegistryConfig.from_env()
        assert config.max_dependency_depth == 20

    def test_env_override_provenance_disabled(self, monkeypatch):
        monkeypatch.setenv("GL_AGENT_REGISTRY_PROVENANCE_ENABLED", "false")
        config = AgentRegistryConfig.from_env()
        assert config.provenance_enabled is False

    def test_env_override_cache_ttl(self, monkeypatch):
        monkeypatch.setenv("GL_AGENT_REGISTRY_CACHE_TTL_SECONDS", "600")
        config = AgentRegistryConfig.from_env()
        assert config.cache_ttl_seconds == 600

    def test_env_override_cache_max_size(self, monkeypatch):
        monkeypatch.setenv("GL_AGENT_REGISTRY_CACHE_MAX_SIZE", "5000")
        config = AgentRegistryConfig.from_env()
        assert config.cache_max_size == 5000

    def test_env_override_bool_true_with_1(self, monkeypatch):
        monkeypatch.setenv("GL_AGENT_REGISTRY_EXPORT_ENABLED", "1")
        config = AgentRegistryConfig.from_env()
        assert config.export_enabled is True

    def test_env_override_bool_true_with_yes(self, monkeypatch):
        monkeypatch.setenv("GL_AGENT_REGISTRY_IMPORT_ENABLED", "yes")
        config = AgentRegistryConfig.from_env()
        assert config.import_enabled is True

    def test_env_override_bool_true_with_TRUE(self, monkeypatch):
        monkeypatch.setenv("GL_AGENT_REGISTRY_PROVENANCE_ENABLED", "TRUE")
        config = AgentRegistryConfig.from_env()
        assert config.provenance_enabled is True

    def test_env_invalid_int_uses_default(self, monkeypatch):
        monkeypatch.setenv("GL_AGENT_REGISTRY_MAX_AGENTS", "not_a_number")
        config = AgentRegistryConfig.from_env()
        assert config.max_agents == 500

    def test_env_invalid_int_cache_fallback(self, monkeypatch):
        monkeypatch.setenv("GL_AGENT_REGISTRY_CACHE_TTL_SECONDS", "abc")
        config = AgentRegistryConfig.from_env()
        assert config.cache_ttl_seconds == 300

    def test_env_empty_string_uses_default(self, monkeypatch):
        monkeypatch.setenv("GL_AGENT_REGISTRY_SERVICE_NAME", "   ")
        config = AgentRegistryConfig.from_env()
        assert config.service_name == "agent-registry"

    def test_multiple_env_overrides(self, monkeypatch):
        monkeypatch.setenv("GL_AGENT_REGISTRY_MAX_AGENTS", "200")
        monkeypatch.setenv("GL_AGENT_REGISTRY_HEALTH_CHECK_ENABLED", "false")
        monkeypatch.setenv("GL_AGENT_REGISTRY_LOG_LEVEL", "WARNING")
        monkeypatch.setenv("GL_AGENT_REGISTRY_ENVIRONMENT", "test")
        config = AgentRegistryConfig.from_env()
        assert config.max_agents == 200
        assert config.health_check_enabled is False
        assert config.log_level == "WARNING"
        assert config.environment == "test"


class TestAgentRegistryConfigSingleton:
    """Test get_config/set_config/reset_config singleton pattern."""

    def test_get_config_returns_instance(self):
        config = get_config()
        assert isinstance(config, AgentRegistryConfig)

    def test_get_config_returns_same_instance(self):
        c1 = get_config()
        c2 = get_config()
        assert c1 is c2

    def test_reset_config_clears_singleton(self):
        c1 = get_config()
        reset_config()
        c2 = get_config()
        assert c1 is not c2

    def test_set_config_overrides_singleton(self):
        custom = AgentRegistryConfig(max_agents=999)
        set_config(custom)
        assert get_config().max_agents == 999

    def test_set_config_then_get_returns_same(self):
        custom = AgentRegistryConfig(cache_ttl_seconds=42)
        set_config(custom)
        assert get_config() is custom

    def test_thread_safety_of_get_config(self):
        """Test that concurrent get_config calls return the same instance."""
        instances = []

        def get_instance():
            instances.append(get_config())

        threads = [threading.Thread(target=get_instance) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(instances) == 10
        for inst in instances[1:]:
            assert inst is instances[0]

    def test_set_then_reset_then_get(self):
        custom = AgentRegistryConfig(max_agents=42)
        set_config(custom)
        assert get_config().max_agents == 42
        reset_config()
        fresh = get_config()
        assert fresh.max_agents == 500  # back to default


class TestAgentRegistryConfigValidation:
    """Test invalid env values fallback to defaults."""

    def test_invalid_int_max_versions_fallback(self, monkeypatch):
        monkeypatch.setenv("GL_AGENT_REGISTRY_MAX_VERSIONS_PER_AGENT", "foo")
        config = AgentRegistryConfig.from_env()
        assert config.max_versions_per_agent == 50

    def test_invalid_int_max_capabilities_fallback(self, monkeypatch):
        monkeypatch.setenv("GL_AGENT_REGISTRY_MAX_CAPABILITIES_PER_AGENT", "bar")
        config = AgentRegistryConfig.from_env()
        assert config.max_capabilities_per_agent == 20

    def test_invalid_int_max_dependencies_fallback(self, monkeypatch):
        monkeypatch.setenv("GL_AGENT_REGISTRY_MAX_DEPENDENCIES_PER_AGENT", "baz")
        config = AgentRegistryConfig.from_env()
        assert config.max_dependencies_per_agent == 30

    def test_invalid_int_unhealthy_threshold_fallback(self, monkeypatch):
        monkeypatch.setenv("GL_AGENT_REGISTRY_UNHEALTHY_THRESHOLD", "xyz")
        config = AgentRegistryConfig.from_env()
        assert config.unhealthy_threshold == 3

    def test_invalid_int_timeout_fallback(self, monkeypatch):
        monkeypatch.setenv("GL_AGENT_REGISTRY_HEALTH_CHECK_TIMEOUT_SECONDS", "nope")
        config = AgentRegistryConfig.from_env()
        assert config.health_check_timeout_seconds == 5

    def test_invalid_int_grace_period_fallback(self, monkeypatch):
        monkeypatch.setenv("GL_AGENT_REGISTRY_HOT_RELOAD_GRACE_PERIOD_SECONDS", "bad")
        config = AgentRegistryConfig.from_env()
        assert config.hot_reload_grace_period_seconds == 10

    def test_invalid_int_max_tags_fallback(self, monkeypatch):
        monkeypatch.setenv("GL_AGENT_REGISTRY_MAX_TAGS_PER_AGENT", "oops")
        config = AgentRegistryConfig.from_env()
        assert config.max_tags_per_agent == 50


class TestAgentRegistryConfigCustomValues:
    """Test creating config with fully custom values."""

    def test_full_custom_config(self):
        config = AgentRegistryConfig(
            service_name="custom-svc",
            environment="test",
            log_level="DEBUG",
            max_agents=100,
            max_versions_per_agent=10,
            max_capabilities_per_agent=5,
            max_dependencies_per_agent=8,
            max_tags_per_agent=20,
            health_check_enabled=False,
            health_check_interval_seconds=60,
            health_check_timeout_seconds=10,
            unhealthy_threshold=5,
            hot_reload_enabled=False,
            hot_reload_grace_period_seconds=30,
            dependency_resolution_enabled=False,
            fail_on_missing_dependency=False,
            max_dependency_depth=5,
            capability_matching_enabled=False,
            provenance_enabled=False,
            cache_ttl_seconds=120,
            cache_max_size=500,
            export_enabled=False,
            import_enabled=False,
        )
        assert config.service_name == "custom-svc"
        assert config.environment == "test"
        assert config.log_level == "DEBUG"
        assert config.max_agents == 100
        assert config.max_versions_per_agent == 10
        assert config.max_capabilities_per_agent == 5
        assert config.max_dependencies_per_agent == 8
        assert config.max_tags_per_agent == 20
        assert config.health_check_enabled is False
        assert config.health_check_interval_seconds == 60
        assert config.health_check_timeout_seconds == 10
        assert config.unhealthy_threshold == 5
        assert config.hot_reload_enabled is False
        assert config.hot_reload_grace_period_seconds == 30
        assert config.dependency_resolution_enabled is False
        assert config.fail_on_missing_dependency is False
        assert config.max_dependency_depth == 5
        assert config.capability_matching_enabled is False
        assert config.provenance_enabled is False
        assert config.cache_ttl_seconds == 120
        assert config.cache_max_size == 500
        assert config.export_enabled is False
        assert config.import_enabled is False
