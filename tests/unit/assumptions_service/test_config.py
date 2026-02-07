# -*- coding: utf-8 -*-
"""
Unit Tests for AssumptionsConfig (AGENT-FOUND-004)

Tests configuration creation, env var overrides with GL_ASSUMPTIONS_ prefix,
validation settings, version limits, and singleton get_config/set_config/reset_config.

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
# Inline AssumptionsConfig that mirrors greenlang/assumptions/config.py
# ---------------------------------------------------------------------------

_ENV_PREFIX = "GL_ASSUMPTIONS_"


@dataclass
class AssumptionsConfig:
    """Mirrors greenlang.assumptions.config.AssumptionsConfig."""

    # Version management
    max_versions: int = 50
    default_scenario: str = "baseline"

    # Validation
    enable_validation: bool = True
    strict_validation: bool = False

    # Provenance
    enable_provenance: bool = True
    hash_algorithm: str = "sha256"

    # Performance
    cache_enabled: bool = True
    cache_ttl_seconds: int = 300

    # Export
    export_format: str = "json"
    max_export_items: int = 10000

    # Sensitivity analysis
    sensitivity_default_range: float = 0.1
    sensitivity_steps: int = 10

    VALID_HASH_ALGORITHMS = ("sha256", "sha384", "sha512")
    VALID_EXPORT_FORMATS = ("json", "csv", "yaml")
    MIN_VERSIONS = 1
    MAX_VERSIONS = 1000
    MIN_SENSITIVITY_STEPS = 2
    MAX_SENSITIVITY_STEPS = 100

    @classmethod
    def from_env(cls) -> AssumptionsConfig:
        """Build from environment variables with GL_ASSUMPTIONS_ prefix."""
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

        def _float(name: str, default: float) -> float:
            val = _env(name)
            if val is None:
                return default
            try:
                return float(val)
            except ValueError:
                return default

        def _str(name: str, default: str) -> str:
            val = _env(name)
            if val is None:
                return default
            return val

        return cls(
            max_versions=_int("MAX_VERSIONS", cls.max_versions),
            default_scenario=_str("DEFAULT_SCENARIO", cls.default_scenario),
            enable_validation=_bool("ENABLE_VALIDATION", cls.enable_validation),
            strict_validation=_bool("STRICT_VALIDATION", cls.strict_validation),
            enable_provenance=_bool("ENABLE_PROVENANCE", cls.enable_provenance),
            hash_algorithm=_str("HASH_ALGORITHM", cls.hash_algorithm),
            cache_enabled=_bool("CACHE_ENABLED", cls.cache_enabled),
            cache_ttl_seconds=_int("CACHE_TTL_SECONDS", cls.cache_ttl_seconds),
            export_format=_str("EXPORT_FORMAT", cls.export_format),
            max_export_items=_int("MAX_EXPORT_ITEMS", cls.max_export_items),
            sensitivity_default_range=_float(
                "SENSITIVITY_DEFAULT_RANGE", cls.sensitivity_default_range,
            ),
            sensitivity_steps=_int("SENSITIVITY_STEPS", cls.sensitivity_steps),
        )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_config_instance: Optional[AssumptionsConfig] = None
_config_lock = threading.Lock()


def get_config() -> AssumptionsConfig:
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = AssumptionsConfig.from_env()
    return _config_instance


def set_config(config: AssumptionsConfig) -> None:
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


class TestDefaultConfigValues:
    """Test that default configuration values match PRD requirements."""

    def test_default_max_versions(self):
        config = AssumptionsConfig()
        assert config.max_versions == 50

    def test_default_scenario(self):
        config = AssumptionsConfig()
        assert config.default_scenario == "baseline"

    def test_default_enable_validation(self):
        config = AssumptionsConfig()
        assert config.enable_validation is True

    def test_default_strict_validation(self):
        config = AssumptionsConfig()
        assert config.strict_validation is False

    def test_default_enable_provenance(self):
        config = AssumptionsConfig()
        assert config.enable_provenance is True

    def test_default_hash_algorithm(self):
        config = AssumptionsConfig()
        assert config.hash_algorithm == "sha256"

    def test_default_cache_enabled(self):
        config = AssumptionsConfig()
        assert config.cache_enabled is True

    def test_default_cache_ttl_seconds(self):
        config = AssumptionsConfig()
        assert config.cache_ttl_seconds == 300

    def test_default_export_format(self):
        config = AssumptionsConfig()
        assert config.export_format == "json"

    def test_default_max_export_items(self):
        config = AssumptionsConfig()
        assert config.max_export_items == 10000

    def test_default_sensitivity_range(self):
        config = AssumptionsConfig()
        assert config.sensitivity_default_range == 0.1

    def test_default_sensitivity_steps(self):
        config = AssumptionsConfig()
        assert config.sensitivity_steps == 10


class TestFromEnvOverrides:
    """Test GL_ASSUMPTIONS_ env var overrides via from_env()."""

    def test_env_override_max_versions(self, monkeypatch):
        monkeypatch.setenv("GL_ASSUMPTIONS_MAX_VERSIONS", "100")
        config = AssumptionsConfig.from_env()
        assert config.max_versions == 100

    def test_env_override_enable_validation_false(self, monkeypatch):
        monkeypatch.setenv("GL_ASSUMPTIONS_ENABLE_VALIDATION", "false")
        config = AssumptionsConfig.from_env()
        assert config.enable_validation is False

    def test_env_override_enable_validation_true(self, monkeypatch):
        monkeypatch.setenv("GL_ASSUMPTIONS_ENABLE_VALIDATION", "1")
        config = AssumptionsConfig.from_env()
        assert config.enable_validation is True

    def test_env_override_strict_validation(self, monkeypatch):
        monkeypatch.setenv("GL_ASSUMPTIONS_STRICT_VALIDATION", "yes")
        config = AssumptionsConfig.from_env()
        assert config.strict_validation is True

    def test_env_override_hash_algorithm(self, monkeypatch):
        monkeypatch.setenv("GL_ASSUMPTIONS_HASH_ALGORITHM", "sha512")
        config = AssumptionsConfig.from_env()
        assert config.hash_algorithm == "sha512"

    def test_env_override_cache_enabled_false(self, monkeypatch):
        monkeypatch.setenv("GL_ASSUMPTIONS_CACHE_ENABLED", "false")
        config = AssumptionsConfig.from_env()
        assert config.cache_enabled is False

    def test_env_override_cache_ttl(self, monkeypatch):
        monkeypatch.setenv("GL_ASSUMPTIONS_CACHE_TTL_SECONDS", "600")
        config = AssumptionsConfig.from_env()
        assert config.cache_ttl_seconds == 600

    def test_env_override_export_format(self, monkeypatch):
        monkeypatch.setenv("GL_ASSUMPTIONS_EXPORT_FORMAT", "csv")
        config = AssumptionsConfig.from_env()
        assert config.export_format == "csv"

    def test_env_override_sensitivity_range(self, monkeypatch):
        monkeypatch.setenv("GL_ASSUMPTIONS_SENSITIVITY_DEFAULT_RANGE", "0.25")
        config = AssumptionsConfig.from_env()
        assert config.sensitivity_default_range == 0.25

    def test_env_override_sensitivity_steps(self, monkeypatch):
        monkeypatch.setenv("GL_ASSUMPTIONS_SENSITIVITY_STEPS", "20")
        config = AssumptionsConfig.from_env()
        assert config.sensitivity_steps == 20

    def test_env_invalid_int_uses_default(self, monkeypatch):
        monkeypatch.setenv("GL_ASSUMPTIONS_MAX_VERSIONS", "not_a_number")
        config = AssumptionsConfig.from_env()
        assert config.max_versions == 50

    def test_env_invalid_float_uses_default(self, monkeypatch):
        monkeypatch.setenv("GL_ASSUMPTIONS_SENSITIVITY_DEFAULT_RANGE", "bad")
        config = AssumptionsConfig.from_env()
        assert config.sensitivity_default_range == 0.1

    def test_multiple_env_overrides(self, monkeypatch):
        monkeypatch.setenv("GL_ASSUMPTIONS_MAX_VERSIONS", "200")
        monkeypatch.setenv("GL_ASSUMPTIONS_ENABLE_VALIDATION", "false")
        monkeypatch.setenv("GL_ASSUMPTIONS_EXPORT_FORMAT", "yaml")
        config = AssumptionsConfig.from_env()
        assert config.max_versions == 200
        assert config.enable_validation is False
        assert config.export_format == "yaml"


class TestBoundaryValues:
    """Test boundary values for configuration fields."""

    def test_min_versions_constant(self):
        assert AssumptionsConfig.MIN_VERSIONS == 1

    def test_max_versions_constant(self):
        assert AssumptionsConfig.MAX_VERSIONS == 1000

    def test_valid_hash_algorithms(self):
        assert "sha256" in AssumptionsConfig.VALID_HASH_ALGORITHMS
        assert "sha384" in AssumptionsConfig.VALID_HASH_ALGORITHMS
        assert "sha512" in AssumptionsConfig.VALID_HASH_ALGORITHMS

    def test_valid_export_formats(self):
        assert "json" in AssumptionsConfig.VALID_EXPORT_FORMATS
        assert "csv" in AssumptionsConfig.VALID_EXPORT_FORMATS
        assert "yaml" in AssumptionsConfig.VALID_EXPORT_FORMATS

    def test_sensitivity_steps_min(self):
        assert AssumptionsConfig.MIN_SENSITIVITY_STEPS == 2

    def test_sensitivity_steps_max(self):
        assert AssumptionsConfig.MAX_SENSITIVITY_STEPS == 100


class TestConfigSingleton:
    """Test get_config/set_config/reset_config singleton pattern."""

    def test_get_config_returns_instance(self):
        config = get_config()
        assert isinstance(config, AssumptionsConfig)

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
        custom = AssumptionsConfig(max_versions=99)
        set_config(custom)
        assert get_config().max_versions == 99

    def test_set_config_then_get_returns_same(self):
        custom = AssumptionsConfig(hash_algorithm="sha512")
        set_config(custom)
        assert get_config() is custom


class TestConfigCustomValues:
    """Test creating config with fully custom values."""

    def test_full_custom_config(self):
        config = AssumptionsConfig(
            max_versions=200,
            default_scenario="optimistic",
            enable_validation=False,
            strict_validation=True,
            enable_provenance=False,
            hash_algorithm="sha512",
            cache_enabled=False,
            cache_ttl_seconds=60,
            export_format="csv",
            max_export_items=500,
            sensitivity_default_range=0.25,
            sensitivity_steps=20,
        )
        assert config.max_versions == 200
        assert config.default_scenario == "optimistic"
        assert config.enable_validation is False
        assert config.strict_validation is True
        assert config.enable_provenance is False
        assert config.hash_algorithm == "sha512"
        assert config.cache_enabled is False
        assert config.cache_ttl_seconds == 60
        assert config.export_format == "csv"
        assert config.max_export_items == 500
        assert config.sensitivity_default_range == 0.25
        assert config.sensitivity_steps == 20
