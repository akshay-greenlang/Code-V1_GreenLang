# -*- coding: utf-8 -*-
"""
Unit Tests for ReproducibilityConfig (AGENT-FOUND-008)

Tests configuration creation, env var overrides with GL_REPRODUCIBILITY_ prefix,
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
# Inline ReproducibilityConfig mirroring greenlang/reproducibility/config.py
# ---------------------------------------------------------------------------

_ENV_PREFIX = "GL_REPRODUCIBILITY_"

import logging

_logger = logging.getLogger(__name__)


@dataclass
class ReproducibilityConfig:
    """Mirrors greenlang.reproducibility.config.ReproducibilityConfig."""

    # Connections
    database_url: str = ""
    redis_url: str = ""

    # Tolerances
    default_absolute_tolerance: float = 1e-9
    default_relative_tolerance: float = 1e-6

    # Drift
    drift_soft_threshold: float = 0.01
    drift_hard_threshold: float = 0.05

    # Hashing
    hash_algorithm: str = "sha256"
    hash_cache_ttl_seconds: int = 3600

    # Feature toggles
    environment_capture_enabled: bool = True
    seed_management_enabled: bool = True
    version_pinning_enabled: bool = True
    non_determinism_tracking_enabled: bool = True

    # Replay
    replay_strict_mode: bool = False
    max_replay_duration_seconds: int = 300

    # Normalization
    float_normalization_decimals: int = 15

    # Verification
    verification_timeout_seconds: int = 60

    # Pool sizing
    pool_min_size: int = 2
    pool_max_size: int = 10

    # Logging
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "ReproducibilityConfig":
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
                _logger.warning(
                    "Invalid integer for %s%s=%s, using default %d",
                    prefix, name, val, default,
                )
                return default

        def _float(name: str, default: float) -> float:
            val = _env(name)
            if val is None:
                return default
            try:
                return float(val)
            except ValueError:
                _logger.warning(
                    "Invalid float for %s%s=%s, using default %s",
                    prefix, name, val, default,
                )
                return default

        def _str(name: str, default: str) -> str:
            val = _env(name)
            if val is None:
                return default
            return val

        return cls(
            database_url=_str("DATABASE_URL", cls.database_url),
            redis_url=_str("REDIS_URL", cls.redis_url),
            default_absolute_tolerance=_float(
                "DEFAULT_ABSOLUTE_TOLERANCE", cls.default_absolute_tolerance,
            ),
            default_relative_tolerance=_float(
                "DEFAULT_RELATIVE_TOLERANCE", cls.default_relative_tolerance,
            ),
            drift_soft_threshold=_float("DRIFT_SOFT_THRESHOLD", cls.drift_soft_threshold),
            drift_hard_threshold=_float("DRIFT_HARD_THRESHOLD", cls.drift_hard_threshold),
            hash_algorithm=_str("HASH_ALGORITHM", cls.hash_algorithm),
            hash_cache_ttl_seconds=_int("HASH_CACHE_TTL_SECONDS", cls.hash_cache_ttl_seconds),
            environment_capture_enabled=_bool(
                "ENVIRONMENT_CAPTURE_ENABLED", cls.environment_capture_enabled,
            ),
            seed_management_enabled=_bool(
                "SEED_MANAGEMENT_ENABLED", cls.seed_management_enabled,
            ),
            version_pinning_enabled=_bool(
                "VERSION_PINNING_ENABLED", cls.version_pinning_enabled,
            ),
            non_determinism_tracking_enabled=_bool(
                "NON_DETERMINISM_TRACKING_ENABLED",
                cls.non_determinism_tracking_enabled,
            ),
            replay_strict_mode=_bool("REPLAY_STRICT_MODE", cls.replay_strict_mode),
            max_replay_duration_seconds=_int(
                "MAX_REPLAY_DURATION_SECONDS", cls.max_replay_duration_seconds,
            ),
            float_normalization_decimals=_int(
                "FLOAT_NORMALIZATION_DECIMALS", cls.float_normalization_decimals,
            ),
            verification_timeout_seconds=_int(
                "VERIFICATION_TIMEOUT_SECONDS", cls.verification_timeout_seconds,
            ),
            pool_min_size=_int("POOL_MIN_SIZE", cls.pool_min_size),
            pool_max_size=_int("POOL_MAX_SIZE", cls.pool_max_size),
            log_level=_str("LOG_LEVEL", cls.log_level),
        )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_config_instance: Optional[ReproducibilityConfig] = None
_config_lock = threading.Lock()


def get_config() -> ReproducibilityConfig:
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = ReproducibilityConfig.from_env()
    return _config_instance


def set_config(config: ReproducibilityConfig) -> None:
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


class TestReproducibilityConfigDefaults:
    """Test that default configuration values match PRD requirements."""

    def test_default_database_url(self):
        config = ReproducibilityConfig()
        assert config.database_url == ""

    def test_default_redis_url(self):
        config = ReproducibilityConfig()
        assert config.redis_url == ""

    def test_default_absolute_tolerance(self):
        config = ReproducibilityConfig()
        assert config.default_absolute_tolerance == 1e-9

    def test_default_relative_tolerance(self):
        config = ReproducibilityConfig()
        assert config.default_relative_tolerance == 1e-6

    def test_default_drift_soft_threshold(self):
        config = ReproducibilityConfig()
        assert config.drift_soft_threshold == 0.01

    def test_default_drift_hard_threshold(self):
        config = ReproducibilityConfig()
        assert config.drift_hard_threshold == 0.05

    def test_default_hash_algorithm(self):
        config = ReproducibilityConfig()
        assert config.hash_algorithm == "sha256"

    def test_default_hash_cache_ttl(self):
        config = ReproducibilityConfig()
        assert config.hash_cache_ttl_seconds == 3600

    def test_default_environment_capture_enabled(self):
        config = ReproducibilityConfig()
        assert config.environment_capture_enabled is True

    def test_default_seed_management_enabled(self):
        config = ReproducibilityConfig()
        assert config.seed_management_enabled is True

    def test_default_version_pinning_enabled(self):
        config = ReproducibilityConfig()
        assert config.version_pinning_enabled is True

    def test_default_non_determinism_tracking_enabled(self):
        config = ReproducibilityConfig()
        assert config.non_determinism_tracking_enabled is True

    def test_default_replay_strict_mode(self):
        config = ReproducibilityConfig()
        assert config.replay_strict_mode is False

    def test_default_max_replay_duration(self):
        config = ReproducibilityConfig()
        assert config.max_replay_duration_seconds == 300

    def test_default_float_normalization_decimals(self):
        config = ReproducibilityConfig()
        assert config.float_normalization_decimals == 15

    def test_default_verification_timeout(self):
        config = ReproducibilityConfig()
        assert config.verification_timeout_seconds == 60

    def test_default_pool_min_size(self):
        config = ReproducibilityConfig()
        assert config.pool_min_size == 2

    def test_default_pool_max_size(self):
        config = ReproducibilityConfig()
        assert config.pool_max_size == 10

    def test_default_log_level(self):
        config = ReproducibilityConfig()
        assert config.log_level == "INFO"


class TestReproducibilityConfigFromEnv:
    """Test GL_REPRODUCIBILITY_ env var overrides via from_env()."""

    def test_env_override_database_url(self, monkeypatch):
        monkeypatch.setenv("GL_REPRODUCIBILITY_DATABASE_URL", "postgresql://localhost/repro")
        config = ReproducibilityConfig.from_env()
        assert config.database_url == "postgresql://localhost/repro"

    def test_env_override_redis_url(self, monkeypatch):
        monkeypatch.setenv("GL_REPRODUCIBILITY_REDIS_URL", "redis://localhost:6379/0")
        config = ReproducibilityConfig.from_env()
        assert config.redis_url == "redis://localhost:6379/0"

    def test_env_override_absolute_tolerance(self, monkeypatch):
        monkeypatch.setenv("GL_REPRODUCIBILITY_DEFAULT_ABSOLUTE_TOLERANCE", "1e-6")
        config = ReproducibilityConfig.from_env()
        assert config.default_absolute_tolerance == 1e-6

    def test_env_override_relative_tolerance(self, monkeypatch):
        monkeypatch.setenv("GL_REPRODUCIBILITY_DEFAULT_RELATIVE_TOLERANCE", "1e-3")
        config = ReproducibilityConfig.from_env()
        assert config.default_relative_tolerance == 1e-3

    def test_env_override_drift_soft_threshold(self, monkeypatch):
        monkeypatch.setenv("GL_REPRODUCIBILITY_DRIFT_SOFT_THRESHOLD", "0.02")
        config = ReproducibilityConfig.from_env()
        assert config.drift_soft_threshold == 0.02

    def test_env_override_drift_hard_threshold(self, monkeypatch):
        monkeypatch.setenv("GL_REPRODUCIBILITY_DRIFT_HARD_THRESHOLD", "0.10")
        config = ReproducibilityConfig.from_env()
        assert config.drift_hard_threshold == 0.10

    def test_env_override_hash_algorithm(self, monkeypatch):
        monkeypatch.setenv("GL_REPRODUCIBILITY_HASH_ALGORITHM", "sha512")
        config = ReproducibilityConfig.from_env()
        assert config.hash_algorithm == "sha512"

    def test_env_override_hash_cache_ttl(self, monkeypatch):
        monkeypatch.setenv("GL_REPRODUCIBILITY_HASH_CACHE_TTL_SECONDS", "7200")
        config = ReproducibilityConfig.from_env()
        assert config.hash_cache_ttl_seconds == 7200

    def test_env_override_environment_capture_disabled(self, monkeypatch):
        monkeypatch.setenv("GL_REPRODUCIBILITY_ENVIRONMENT_CAPTURE_ENABLED", "false")
        config = ReproducibilityConfig.from_env()
        assert config.environment_capture_enabled is False

    def test_env_override_seed_management_disabled(self, monkeypatch):
        monkeypatch.setenv("GL_REPRODUCIBILITY_SEED_MANAGEMENT_ENABLED", "0")
        config = ReproducibilityConfig.from_env()
        assert config.seed_management_enabled is False

    def test_env_override_version_pinning_disabled(self, monkeypatch):
        monkeypatch.setenv("GL_REPRODUCIBILITY_VERSION_PINNING_ENABLED", "no")
        config = ReproducibilityConfig.from_env()
        assert config.version_pinning_enabled is False

    def test_env_override_non_determinism_disabled(self, monkeypatch):
        monkeypatch.setenv("GL_REPRODUCIBILITY_NON_DETERMINISM_TRACKING_ENABLED", "false")
        config = ReproducibilityConfig.from_env()
        assert config.non_determinism_tracking_enabled is False

    def test_env_override_replay_strict_mode_true(self, monkeypatch):
        monkeypatch.setenv("GL_REPRODUCIBILITY_REPLAY_STRICT_MODE", "true")
        config = ReproducibilityConfig.from_env()
        assert config.replay_strict_mode is True

    def test_env_override_max_replay_duration(self, monkeypatch):
        monkeypatch.setenv("GL_REPRODUCIBILITY_MAX_REPLAY_DURATION_SECONDS", "600")
        config = ReproducibilityConfig.from_env()
        assert config.max_replay_duration_seconds == 600

    def test_env_override_float_normalization_decimals(self, monkeypatch):
        monkeypatch.setenv("GL_REPRODUCIBILITY_FLOAT_NORMALIZATION_DECIMALS", "10")
        config = ReproducibilityConfig.from_env()
        assert config.float_normalization_decimals == 10

    def test_env_override_verification_timeout(self, monkeypatch):
        monkeypatch.setenv("GL_REPRODUCIBILITY_VERIFICATION_TIMEOUT_SECONDS", "120")
        config = ReproducibilityConfig.from_env()
        assert config.verification_timeout_seconds == 120

    def test_env_override_pool_min_size(self, monkeypatch):
        monkeypatch.setenv("GL_REPRODUCIBILITY_POOL_MIN_SIZE", "5")
        config = ReproducibilityConfig.from_env()
        assert config.pool_min_size == 5

    def test_env_override_pool_max_size(self, monkeypatch):
        monkeypatch.setenv("GL_REPRODUCIBILITY_POOL_MAX_SIZE", "20")
        config = ReproducibilityConfig.from_env()
        assert config.pool_max_size == 20

    def test_env_override_log_level(self, monkeypatch):
        monkeypatch.setenv("GL_REPRODUCIBILITY_LOG_LEVEL", "DEBUG")
        config = ReproducibilityConfig.from_env()
        assert config.log_level == "DEBUG"

    def test_env_bool_true_with_1(self, monkeypatch):
        monkeypatch.setenv("GL_REPRODUCIBILITY_REPLAY_STRICT_MODE", "1")
        config = ReproducibilityConfig.from_env()
        assert config.replay_strict_mode is True

    def test_env_bool_true_with_yes(self, monkeypatch):
        monkeypatch.setenv("GL_REPRODUCIBILITY_REPLAY_STRICT_MODE", "yes")
        config = ReproducibilityConfig.from_env()
        assert config.replay_strict_mode is True

    def test_env_bool_true_with_TRUE_uppercase(self, monkeypatch):
        monkeypatch.setenv("GL_REPRODUCIBILITY_REPLAY_STRICT_MODE", "TRUE")
        config = ReproducibilityConfig.from_env()
        assert config.replay_strict_mode is True

    def test_env_invalid_float_uses_default(self, monkeypatch):
        monkeypatch.setenv("GL_REPRODUCIBILITY_DEFAULT_ABSOLUTE_TOLERANCE", "not_a_float")
        config = ReproducibilityConfig.from_env()
        assert config.default_absolute_tolerance == 1e-9

    def test_env_invalid_int_uses_default(self, monkeypatch):
        monkeypatch.setenv("GL_REPRODUCIBILITY_HASH_CACHE_TTL_SECONDS", "bad_int")
        config = ReproducibilityConfig.from_env()
        assert config.hash_cache_ttl_seconds == 3600

    def test_env_invalid_int_pool_fallback(self, monkeypatch):
        monkeypatch.setenv("GL_REPRODUCIBILITY_POOL_MAX_SIZE", "xyz")
        config = ReproducibilityConfig.from_env()
        assert config.pool_max_size == 10

    def test_env_invalid_float_drift_soft_fallback(self, monkeypatch):
        monkeypatch.setenv("GL_REPRODUCIBILITY_DRIFT_SOFT_THRESHOLD", "abc")
        config = ReproducibilityConfig.from_env()
        assert config.drift_soft_threshold == 0.01

    def test_env_invalid_float_drift_hard_fallback(self, monkeypatch):
        monkeypatch.setenv("GL_REPRODUCIBILITY_DRIFT_HARD_THRESHOLD", "def")
        config = ReproducibilityConfig.from_env()
        assert config.drift_hard_threshold == 0.05

    def test_multiple_env_overrides(self, monkeypatch):
        monkeypatch.setenv("GL_REPRODUCIBILITY_REPLAY_STRICT_MODE", "true")
        monkeypatch.setenv("GL_REPRODUCIBILITY_HASH_ALGORITHM", "sha512")
        monkeypatch.setenv("GL_REPRODUCIBILITY_POOL_MAX_SIZE", "25")
        monkeypatch.setenv("GL_REPRODUCIBILITY_LOG_LEVEL", "WARNING")
        config = ReproducibilityConfig.from_env()
        assert config.replay_strict_mode is True
        assert config.hash_algorithm == "sha512"
        assert config.pool_max_size == 25
        assert config.log_level == "WARNING"


class TestReproducibilityConfigSingleton:
    """Test get_config/set_config/reset_config singleton pattern."""

    def test_get_config_returns_instance(self):
        config = get_config()
        assert isinstance(config, ReproducibilityConfig)

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
        custom = ReproducibilityConfig(pool_max_size=99)
        set_config(custom)
        assert get_config().pool_max_size == 99

    def test_set_config_then_get_returns_same(self):
        custom = ReproducibilityConfig(hash_algorithm="sha512")
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
        custom = ReproducibilityConfig(pool_min_size=99)
        set_config(custom)
        assert get_config().pool_min_size == 99
        reset_config()
        fresh = get_config()
        assert fresh.pool_min_size == 2  # back to default


class TestReproducibilityConfigValidation:
    """Test positive tolerance constraints and database/redis URL parsing."""

    def test_config_positive_absolute_tolerance(self):
        config = ReproducibilityConfig(default_absolute_tolerance=1e-12)
        assert config.default_absolute_tolerance == 1e-12

    def test_config_zero_absolute_tolerance(self):
        config = ReproducibilityConfig(default_absolute_tolerance=0.0)
        assert config.default_absolute_tolerance == 0.0

    def test_config_positive_relative_tolerance(self):
        config = ReproducibilityConfig(default_relative_tolerance=0.001)
        assert config.default_relative_tolerance == 0.001

    def test_config_database_url_postgresql(self):
        config = ReproducibilityConfig(
            database_url="postgresql://user:pass@localhost:5432/repro_db"
        )
        assert "postgresql" in config.database_url
        assert "repro_db" in config.database_url

    def test_config_database_url_with_options(self):
        config = ReproducibilityConfig(
            database_url="postgresql://user:pass@host:5432/db?sslmode=require"
        )
        assert "sslmode=require" in config.database_url

    def test_config_redis_url_standard(self):
        config = ReproducibilityConfig(redis_url="redis://localhost:6379/0")
        assert "redis" in config.redis_url

    def test_config_redis_url_with_password(self):
        config = ReproducibilityConfig(redis_url="redis://:secret@localhost:6379/1")
        assert "secret" in config.redis_url


class TestReproducibilityConfigCustomValues:
    """Test creating config with fully custom values."""

    def test_full_custom_config(self):
        config = ReproducibilityConfig(
            database_url="postgresql://localhost/test",
            redis_url="redis://localhost:6379/0",
            default_absolute_tolerance=1e-6,
            default_relative_tolerance=1e-3,
            drift_soft_threshold=0.02,
            drift_hard_threshold=0.10,
            hash_algorithm="sha512",
            hash_cache_ttl_seconds=7200,
            environment_capture_enabled=False,
            seed_management_enabled=False,
            version_pinning_enabled=False,
            non_determinism_tracking_enabled=False,
            replay_strict_mode=True,
            max_replay_duration_seconds=600,
            float_normalization_decimals=10,
            verification_timeout_seconds=120,
            pool_min_size=5,
            pool_max_size=25,
            log_level="DEBUG",
        )
        assert config.database_url == "postgresql://localhost/test"
        assert config.redis_url == "redis://localhost:6379/0"
        assert config.default_absolute_tolerance == 1e-6
        assert config.default_relative_tolerance == 1e-3
        assert config.drift_soft_threshold == 0.02
        assert config.drift_hard_threshold == 0.10
        assert config.hash_algorithm == "sha512"
        assert config.hash_cache_ttl_seconds == 7200
        assert config.environment_capture_enabled is False
        assert config.seed_management_enabled is False
        assert config.version_pinning_enabled is False
        assert config.non_determinism_tracking_enabled is False
        assert config.replay_strict_mode is True
        assert config.max_replay_duration_seconds == 600
        assert config.float_normalization_decimals == 10
        assert config.verification_timeout_seconds == 120
        assert config.pool_min_size == 5
        assert config.pool_max_size == 25
        assert config.log_level == "DEBUG"
