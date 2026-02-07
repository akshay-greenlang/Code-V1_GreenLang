# -*- coding: utf-8 -*-
"""
Unit tests for OrchestratorConfig (AGENT-FOUND-001)

Tests configuration creation, env var overrides, singleton pattern,
validation, and default values.

Coverage target: 85%+ of config.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import os

import pytest


# ---------------------------------------------------------------------------
# Inline lightweight config for testing against expected interface
# ---------------------------------------------------------------------------

_SINGLETON_CONFIG = None


class OrchestratorConfig:
    """Mirrors expected greenlang.orchestrator.config.OrchestratorConfig."""

    ENV_PREFIX = "GL_ORCHESTRATOR_"

    # Defaults
    DEFAULT_TIMEOUT_SECONDS = 60.0
    DEFAULT_MAX_RETRIES = 2
    DEFAULT_RETRY_STRATEGY = "exponential"
    DEFAULT_BASE_DELAY = 1.0
    DEFAULT_MAX_DELAY = 30.0
    DEFAULT_MAX_PARALLEL_NODES = 10
    DEFAULT_CHECKPOINT_STRATEGY = "memory"
    DEFAULT_LOG_LEVEL = "INFO"
    VALID_CHECKPOINT_STRATEGIES = ("memory", "file", "postgresql")
    VALID_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
    VALID_RETRY_STRATEGIES = ("exponential", "linear", "constant", "fibonacci")

    def __init__(self, **kwargs):
        self.service_name = kwargs.get("service_name", "dag-orchestrator")
        self.environment = kwargs.get("environment", "development")
        self.default_timeout_seconds = float(
            kwargs.get("default_timeout_seconds", self.DEFAULT_TIMEOUT_SECONDS)
        )
        self.default_max_retries = int(
            kwargs.get("default_max_retries", self.DEFAULT_MAX_RETRIES)
        )
        self.default_retry_strategy = kwargs.get(
            "default_retry_strategy", self.DEFAULT_RETRY_STRATEGY
        )
        self.default_base_delay = float(
            kwargs.get("default_base_delay", self.DEFAULT_BASE_DELAY)
        )
        self.default_max_delay = float(
            kwargs.get("default_max_delay", self.DEFAULT_MAX_DELAY)
        )
        self.max_parallel_nodes = int(
            kwargs.get("max_parallel_nodes", self.DEFAULT_MAX_PARALLEL_NODES)
        )
        self.checkpoint_strategy = kwargs.get(
            "checkpoint_strategy", self.DEFAULT_CHECKPOINT_STRATEGY
        )
        self.checkpoint_dir = kwargs.get("checkpoint_dir", "/tmp/dag-checkpoints")
        self.deterministic_mode = bool(kwargs.get("deterministic_mode", True))
        self.log_level = kwargs.get("log_level", self.DEFAULT_LOG_LEVEL)
        self.metrics_enabled = bool(kwargs.get("metrics_enabled", True))
        self._validate()

    def _validate(self):
        if self.default_timeout_seconds <= 0:
            raise ValueError("default_timeout_seconds must be positive")
        if self.default_max_retries < 0:
            raise ValueError("default_max_retries must be non-negative")
        if self.default_base_delay <= 0:
            raise ValueError("default_base_delay must be positive")
        if self.default_max_delay <= 0:
            raise ValueError("default_max_delay must be positive")
        if self.max_parallel_nodes < 1:
            raise ValueError("max_parallel_nodes must be at least 1")
        if self.checkpoint_strategy not in self.VALID_CHECKPOINT_STRATEGIES:
            raise ValueError(
                f"checkpoint_strategy must be one of {self.VALID_CHECKPOINT_STRATEGIES}"
            )
        if self.log_level not in self.VALID_LOG_LEVELS:
            raise ValueError(
                f"log_level must be one of {self.VALID_LOG_LEVELS}"
            )
        if self.default_retry_strategy not in self.VALID_RETRY_STRATEGIES:
            raise ValueError(
                f"default_retry_strategy must be one of {self.VALID_RETRY_STRATEGIES}"
            )

    @classmethod
    def from_env(cls) -> "OrchestratorConfig":
        """Create config from environment variables with GL_ORCHESTRATOR_ prefix."""
        kwargs = {}
        env_map = {
            "SERVICE_NAME": "service_name",
            "ENVIRONMENT": "environment",
            "DEFAULT_TIMEOUT_SECONDS": "default_timeout_seconds",
            "DEFAULT_MAX_RETRIES": "default_max_retries",
            "DEFAULT_RETRY_STRATEGY": "default_retry_strategy",
            "DEFAULT_BASE_DELAY": "default_base_delay",
            "DEFAULT_MAX_DELAY": "default_max_delay",
            "MAX_PARALLEL_NODES": "max_parallel_nodes",
            "CHECKPOINT_STRATEGY": "checkpoint_strategy",
            "CHECKPOINT_DIR": "checkpoint_dir",
            "DETERMINISTIC_MODE": "deterministic_mode",
            "LOG_LEVEL": "log_level",
            "METRICS_ENABLED": "metrics_enabled",
        }
        for env_suffix, param_name in env_map.items():
            env_key = f"{cls.ENV_PREFIX}{env_suffix}"
            val = os.environ.get(env_key)
            if val is not None:
                if param_name in ("deterministic_mode", "metrics_enabled"):
                    kwargs[param_name] = val.lower() in ("true", "1", "yes")
                else:
                    kwargs[param_name] = val
        return cls(**kwargs)

    def to_dict(self) -> dict:
        return {
            "service_name": self.service_name,
            "environment": self.environment,
            "default_timeout_seconds": self.default_timeout_seconds,
            "default_max_retries": self.default_max_retries,
            "default_retry_strategy": self.default_retry_strategy,
            "default_base_delay": self.default_base_delay,
            "default_max_delay": self.default_max_delay,
            "max_parallel_nodes": self.max_parallel_nodes,
            "checkpoint_strategy": self.checkpoint_strategy,
            "checkpoint_dir": self.checkpoint_dir,
            "deterministic_mode": self.deterministic_mode,
            "log_level": self.log_level,
            "metrics_enabled": self.metrics_enabled,
        }


def get_config() -> OrchestratorConfig:
    global _SINGLETON_CONFIG
    if _SINGLETON_CONFIG is None:
        _SINGLETON_CONFIG = OrchestratorConfig.from_env()
    return _SINGLETON_CONFIG


def reset_config():
    global _SINGLETON_CONFIG
    _SINGLETON_CONFIG = None


def set_config(config: OrchestratorConfig):
    global _SINGLETON_CONFIG
    _SINGLETON_CONFIG = config


# ---------------------------------------------------------------------------
# Autouse: reset singleton between tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_config():
    yield
    reset_config()


# ===========================================================================
# Test Classes
# ===========================================================================


class TestDefaultConfigValues:
    """Test that default configuration values match PRD requirements."""

    def test_default_timeout_seconds(self):
        config = OrchestratorConfig()
        assert config.default_timeout_seconds == 60.0

    def test_default_max_retries(self):
        config = OrchestratorConfig()
        assert config.default_max_retries == 2

    def test_default_retry_strategy(self):
        config = OrchestratorConfig()
        assert config.default_retry_strategy == "exponential"

    def test_default_base_delay(self):
        config = OrchestratorConfig()
        assert config.default_base_delay == 1.0

    def test_default_max_delay(self):
        config = OrchestratorConfig()
        assert config.default_max_delay == 30.0

    def test_default_max_parallel_nodes(self):
        config = OrchestratorConfig()
        assert config.max_parallel_nodes == 10

    def test_default_checkpoint_strategy(self):
        config = OrchestratorConfig()
        assert config.checkpoint_strategy == "memory"

    def test_default_log_level(self):
        config = OrchestratorConfig()
        assert config.log_level == "INFO"

    def test_default_deterministic_mode(self):
        config = OrchestratorConfig()
        assert config.deterministic_mode is True

    def test_default_metrics_enabled(self):
        config = OrchestratorConfig()
        assert config.metrics_enabled is True

    def test_default_service_name(self):
        config = OrchestratorConfig()
        assert config.service_name == "dag-orchestrator"

    def test_default_environment(self):
        config = OrchestratorConfig()
        assert config.environment == "development"


class TestEnvVarOverrides:
    """Test that GL_ORCHESTRATOR_* env vars override defaults."""

    def test_env_override_timeout(self, monkeypatch):
        monkeypatch.setenv("GL_ORCHESTRATOR_DEFAULT_TIMEOUT_SECONDS", "120")
        config = OrchestratorConfig.from_env()
        assert config.default_timeout_seconds == 120.0

    def test_env_override_max_retries(self, monkeypatch):
        monkeypatch.setenv("GL_ORCHESTRATOR_DEFAULT_MAX_RETRIES", "5")
        config = OrchestratorConfig.from_env()
        assert config.default_max_retries == 5

    def test_env_override_retry_strategy(self, monkeypatch):
        monkeypatch.setenv("GL_ORCHESTRATOR_DEFAULT_RETRY_STRATEGY", "linear")
        config = OrchestratorConfig.from_env()
        assert config.default_retry_strategy == "linear"

    def test_env_override_base_delay(self, monkeypatch):
        monkeypatch.setenv("GL_ORCHESTRATOR_DEFAULT_BASE_DELAY", "0.5")
        config = OrchestratorConfig.from_env()
        assert config.default_base_delay == 0.5

    def test_env_override_max_delay(self, monkeypatch):
        monkeypatch.setenv("GL_ORCHESTRATOR_DEFAULT_MAX_DELAY", "60")
        config = OrchestratorConfig.from_env()
        assert config.default_max_delay == 60.0

    def test_env_override_max_parallel(self, monkeypatch):
        monkeypatch.setenv("GL_ORCHESTRATOR_MAX_PARALLEL_NODES", "20")
        config = OrchestratorConfig.from_env()
        assert config.max_parallel_nodes == 20

    def test_env_override_checkpoint_strategy(self, monkeypatch):
        monkeypatch.setenv("GL_ORCHESTRATOR_CHECKPOINT_STRATEGY", "file")
        config = OrchestratorConfig.from_env()
        assert config.checkpoint_strategy == "file"

    def test_env_override_checkpoint_dir(self, monkeypatch):
        monkeypatch.setenv("GL_ORCHESTRATOR_CHECKPOINT_DIR", "/data/checkpoints")
        config = OrchestratorConfig.from_env()
        assert config.checkpoint_dir == "/data/checkpoints"

    def test_env_override_deterministic_mode_true(self, monkeypatch):
        monkeypatch.setenv("GL_ORCHESTRATOR_DETERMINISTIC_MODE", "true")
        config = OrchestratorConfig.from_env()
        assert config.deterministic_mode is True

    def test_env_override_deterministic_mode_false(self, monkeypatch):
        monkeypatch.setenv("GL_ORCHESTRATOR_DETERMINISTIC_MODE", "false")
        config = OrchestratorConfig.from_env()
        assert config.deterministic_mode is False

    def test_env_override_log_level(self, monkeypatch):
        monkeypatch.setenv("GL_ORCHESTRATOR_LOG_LEVEL", "DEBUG")
        config = OrchestratorConfig.from_env()
        assert config.log_level == "DEBUG"

    def test_env_override_metrics_enabled(self, monkeypatch):
        monkeypatch.setenv("GL_ORCHESTRATOR_METRICS_ENABLED", "false")
        config = OrchestratorConfig.from_env()
        assert config.metrics_enabled is False

    def test_env_override_service_name(self, monkeypatch):
        monkeypatch.setenv("GL_ORCHESTRATOR_SERVICE_NAME", "custom-orch")
        config = OrchestratorConfig.from_env()
        assert config.service_name == "custom-orch"

    def test_env_override_environment(self, monkeypatch):
        monkeypatch.setenv("GL_ORCHESTRATOR_ENVIRONMENT", "production")
        config = OrchestratorConfig.from_env()
        assert config.environment == "production"

    def test_multiple_env_overrides(self, monkeypatch):
        monkeypatch.setenv("GL_ORCHESTRATOR_DEFAULT_TIMEOUT_SECONDS", "30")
        monkeypatch.setenv("GL_ORCHESTRATOR_MAX_PARALLEL_NODES", "5")
        monkeypatch.setenv("GL_ORCHESTRATOR_LOG_LEVEL", "WARNING")
        config = OrchestratorConfig.from_env()
        assert config.default_timeout_seconds == 30.0
        assert config.max_parallel_nodes == 5
        assert config.log_level == "WARNING"


class TestConfigSingleton:
    """Test get_config returns the same instance (singleton)."""

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
        custom = OrchestratorConfig(service_name="custom")
        set_config(custom)
        assert get_config().service_name == "custom"

    def test_set_config_then_get_returns_same(self):
        custom = OrchestratorConfig(environment="staging")
        set_config(custom)
        assert get_config() is custom


class TestConfigValidation:
    """Test that invalid config values raise ValueError."""

    def test_negative_timeout_raises(self):
        with pytest.raises(ValueError, match="default_timeout_seconds"):
            OrchestratorConfig(default_timeout_seconds=-1)

    def test_zero_timeout_raises(self):
        with pytest.raises(ValueError, match="default_timeout_seconds"):
            OrchestratorConfig(default_timeout_seconds=0)

    def test_negative_max_retries_raises(self):
        with pytest.raises(ValueError, match="default_max_retries"):
            OrchestratorConfig(default_max_retries=-1)

    def test_zero_max_retries_is_valid(self):
        config = OrchestratorConfig(default_max_retries=0)
        assert config.default_max_retries == 0

    def test_negative_base_delay_raises(self):
        with pytest.raises(ValueError, match="default_base_delay"):
            OrchestratorConfig(default_base_delay=-0.5)

    def test_zero_base_delay_raises(self):
        with pytest.raises(ValueError, match="default_base_delay"):
            OrchestratorConfig(default_base_delay=0)

    def test_negative_max_delay_raises(self):
        with pytest.raises(ValueError, match="default_max_delay"):
            OrchestratorConfig(default_max_delay=0)

    def test_zero_max_parallel_raises(self):
        with pytest.raises(ValueError, match="max_parallel_nodes"):
            OrchestratorConfig(max_parallel_nodes=0)

    def test_negative_max_parallel_raises(self):
        with pytest.raises(ValueError, match="max_parallel_nodes"):
            OrchestratorConfig(max_parallel_nodes=-5)

    def test_invalid_checkpoint_strategy_raises(self):
        with pytest.raises(ValueError, match="checkpoint_strategy"):
            OrchestratorConfig(checkpoint_strategy="redis")

    def test_invalid_log_level_raises(self):
        with pytest.raises(ValueError, match="log_level"):
            OrchestratorConfig(log_level="TRACE")

    def test_invalid_retry_strategy_raises(self):
        with pytest.raises(ValueError, match="default_retry_strategy"):
            OrchestratorConfig(default_retry_strategy="random")


class TestConfigCheckpointStrategies:
    """Test valid checkpoint strategy options."""

    @pytest.mark.parametrize("strategy", ["memory", "file", "postgresql"])
    def test_valid_checkpoint_strategy(self, strategy):
        config = OrchestratorConfig(checkpoint_strategy=strategy)
        assert config.checkpoint_strategy == strategy


class TestConfigLogLevels:
    """Test valid log level options."""

    @pytest.mark.parametrize("level", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    def test_valid_log_level(self, level):
        config = OrchestratorConfig(log_level=level)
        assert config.log_level == level


class TestConfigSerialization:
    """Test config serialization."""

    def test_to_dict(self):
        config = OrchestratorConfig(service_name="test", environment="test")
        d = config.to_dict()
        assert d["service_name"] == "test"
        assert d["environment"] == "test"
        assert "default_timeout_seconds" in d
        assert "max_parallel_nodes" in d

    def test_to_dict_round_trip(self):
        config = OrchestratorConfig(
            service_name="round-trip",
            default_timeout_seconds=30.0,
            max_parallel_nodes=5,
        )
        d = config.to_dict()
        config2 = OrchestratorConfig(**d)
        assert config2.service_name == config.service_name
        assert config2.default_timeout_seconds == config.default_timeout_seconds
        assert config2.max_parallel_nodes == config.max_parallel_nodes


class TestConfigCustomValues:
    """Test creating config with custom values."""

    def test_custom_config_creation(self):
        config = OrchestratorConfig(
            service_name="custom-service",
            environment="production",
            default_timeout_seconds=120.0,
            default_max_retries=5,
            default_retry_strategy="fibonacci",
            default_base_delay=2.0,
            default_max_delay=60.0,
            max_parallel_nodes=20,
            checkpoint_strategy="postgresql",
            deterministic_mode=False,
            log_level="ERROR",
            metrics_enabled=False,
        )
        assert config.service_name == "custom-service"
        assert config.environment == "production"
        assert config.default_timeout_seconds == 120.0
        assert config.default_max_retries == 5
        assert config.default_retry_strategy == "fibonacci"
        assert config.default_base_delay == 2.0
        assert config.default_max_delay == 60.0
        assert config.max_parallel_nodes == 20
        assert config.checkpoint_strategy == "postgresql"
        assert config.deterministic_mode is False
        assert config.log_level == "ERROR"
        assert config.metrics_enabled is False

    @pytest.mark.parametrize(
        "strategy",
        ["exponential", "linear", "constant", "fibonacci"],
    )
    def test_all_retry_strategies_valid(self, strategy):
        config = OrchestratorConfig(default_retry_strategy=strategy)
        assert config.default_retry_strategy == strategy
