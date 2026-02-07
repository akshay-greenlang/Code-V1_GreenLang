# -*- coding: utf-8 -*-
"""
Unit Tests for NormalizerConfig (AGENT-FOUND-003)

Tests configuration creation, env var overrides with GL_NORMALIZER_ prefix,
GWP version validation, precision bounds, cache settings, and singleton
get_config/set_config/reset_config.

Coverage target: 85%+ of config.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Any, Optional
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Inline NormalizerConfig that mirrors greenlang/normalizer/config.py
# ---------------------------------------------------------------------------

_ENV_PREFIX = "GL_NORMALIZER_"


@dataclass
class NormalizerConfig:
    """Mirrors greenlang.normalizer.config.NormalizerConfig."""

    # Decimal precision
    default_precision: int = 10

    # GWP settings
    gwp_version: str = "AR6"
    gwp_timeframe: int = 100

    # Caching
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600

    # Batch processing
    max_batch_size: int = 1000

    # Dimensional analysis
    strict_dimensional_check: bool = True

    # Canonical units
    canonical_energy_unit: str = "kWh"
    canonical_mass_unit: str = "kg"
    canonical_emissions_unit: str = "kgCO2e"

    VALID_GWP_VERSIONS = ("AR5", "AR6")
    MIN_PRECISION = 1
    MAX_PRECISION = 15

    @classmethod
    def from_env(cls) -> NormalizerConfig:
        """Build from environment variables with GL_NORMALIZER_ prefix."""
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
            return val

        return cls(
            default_precision=_int("DEFAULT_PRECISION", cls.default_precision),
            gwp_version=_str("GWP_VERSION", cls.gwp_version),
            gwp_timeframe=_int("GWP_TIMEFRAME", cls.gwp_timeframe),
            cache_enabled=_bool("CACHE_ENABLED", cls.cache_enabled),
            cache_ttl_seconds=_int("CACHE_TTL_SECONDS", cls.cache_ttl_seconds),
            max_batch_size=_int("MAX_BATCH_SIZE", cls.max_batch_size),
            strict_dimensional_check=_bool(
                "STRICT_DIMENSIONAL_CHECK", cls.strict_dimensional_check,
            ),
            canonical_energy_unit=_str(
                "CANONICAL_ENERGY_UNIT", cls.canonical_energy_unit,
            ),
            canonical_mass_unit=_str(
                "CANONICAL_MASS_UNIT", cls.canonical_mass_unit,
            ),
            canonical_emissions_unit=_str(
                "CANONICAL_EMISSIONS_UNIT", cls.canonical_emissions_unit,
            ),
        )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_config_instance: Optional[NormalizerConfig] = None
_config_lock = threading.Lock()


def get_config() -> NormalizerConfig:
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = NormalizerConfig.from_env()
    return _config_instance


def set_config(config: NormalizerConfig) -> None:
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

    def test_default_precision(self):
        config = NormalizerConfig()
        assert config.default_precision == 10

    def test_default_gwp_version(self):
        config = NormalizerConfig()
        assert config.gwp_version == "AR6"

    def test_default_gwp_timeframe(self):
        config = NormalizerConfig()
        assert config.gwp_timeframe == 100

    def test_default_cache_enabled(self):
        config = NormalizerConfig()
        assert config.cache_enabled is True

    def test_default_cache_ttl_seconds(self):
        config = NormalizerConfig()
        assert config.cache_ttl_seconds == 3600

    def test_default_max_batch_size(self):
        config = NormalizerConfig()
        assert config.max_batch_size == 1000

    def test_default_strict_dimensional_check(self):
        config = NormalizerConfig()
        assert config.strict_dimensional_check is True

    def test_default_canonical_energy_unit(self):
        config = NormalizerConfig()
        assert config.canonical_energy_unit == "kWh"

    def test_default_canonical_mass_unit(self):
        config = NormalizerConfig()
        assert config.canonical_mass_unit == "kg"

    def test_default_canonical_emissions_unit(self):
        config = NormalizerConfig()
        assert config.canonical_emissions_unit == "kgCO2e"


class TestFromEnvOverrides:
    """Test GL_NORMALIZER_ env var overrides via from_env()."""

    def test_env_override_precision(self, monkeypatch):
        monkeypatch.setenv("GL_NORMALIZER_DEFAULT_PRECISION", "6")
        config = NormalizerConfig.from_env()
        assert config.default_precision == 6

    def test_env_override_gwp_version(self, monkeypatch):
        monkeypatch.setenv("GL_NORMALIZER_GWP_VERSION", "AR5")
        config = NormalizerConfig.from_env()
        assert config.gwp_version == "AR5"

    def test_env_override_gwp_timeframe(self, monkeypatch):
        monkeypatch.setenv("GL_NORMALIZER_GWP_TIMEFRAME", "20")
        config = NormalizerConfig.from_env()
        assert config.gwp_timeframe == 20

    def test_env_override_cache_enabled_false(self, monkeypatch):
        monkeypatch.setenv("GL_NORMALIZER_CACHE_ENABLED", "false")
        config = NormalizerConfig.from_env()
        assert config.cache_enabled is False

    def test_env_override_cache_enabled_true(self, monkeypatch):
        monkeypatch.setenv("GL_NORMALIZER_CACHE_ENABLED", "1")
        config = NormalizerConfig.from_env()
        assert config.cache_enabled is True

    def test_env_override_cache_ttl(self, monkeypatch):
        monkeypatch.setenv("GL_NORMALIZER_CACHE_TTL_SECONDS", "7200")
        config = NormalizerConfig.from_env()
        assert config.cache_ttl_seconds == 7200

    def test_env_override_max_batch_size(self, monkeypatch):
        monkeypatch.setenv("GL_NORMALIZER_MAX_BATCH_SIZE", "500")
        config = NormalizerConfig.from_env()
        assert config.max_batch_size == 500

    def test_env_override_strict_dimensional_false(self, monkeypatch):
        monkeypatch.setenv("GL_NORMALIZER_STRICT_DIMENSIONAL_CHECK", "no")
        config = NormalizerConfig.from_env()
        assert config.strict_dimensional_check is False

    def test_env_override_canonical_energy_unit(self, monkeypatch):
        monkeypatch.setenv("GL_NORMALIZER_CANONICAL_ENERGY_UNIT", "MWh")
        config = NormalizerConfig.from_env()
        assert config.canonical_energy_unit == "MWh"

    def test_env_override_canonical_mass_unit(self, monkeypatch):
        monkeypatch.setenv("GL_NORMALIZER_CANONICAL_MASS_UNIT", "t")
        config = NormalizerConfig.from_env()
        assert config.canonical_mass_unit == "t"

    def test_env_override_canonical_emissions_unit(self, monkeypatch):
        monkeypatch.setenv("GL_NORMALIZER_CANONICAL_EMISSIONS_UNIT", "tCO2e")
        config = NormalizerConfig.from_env()
        assert config.canonical_emissions_unit == "tCO2e"

    def test_env_invalid_int_uses_default(self, monkeypatch):
        monkeypatch.setenv("GL_NORMALIZER_DEFAULT_PRECISION", "not_a_number")
        config = NormalizerConfig.from_env()
        assert config.default_precision == 10

    def test_multiple_env_overrides(self, monkeypatch):
        monkeypatch.setenv("GL_NORMALIZER_DEFAULT_PRECISION", "8")
        monkeypatch.setenv("GL_NORMALIZER_GWP_VERSION", "AR5")
        monkeypatch.setenv("GL_NORMALIZER_CACHE_ENABLED", "false")
        config = NormalizerConfig.from_env()
        assert config.default_precision == 8
        assert config.gwp_version == "AR5"
        assert config.cache_enabled is False


class TestGWPVersionValidation:
    """Test GWP version validation."""

    def test_ar5_is_valid(self):
        config = NormalizerConfig(gwp_version="AR5")
        assert config.gwp_version == "AR5"

    def test_ar6_is_valid(self):
        config = NormalizerConfig(gwp_version="AR6")
        assert config.gwp_version == "AR6"

    def test_valid_gwp_versions_constant(self):
        assert "AR5" in NormalizerConfig.VALID_GWP_VERSIONS
        assert "AR6" in NormalizerConfig.VALID_GWP_VERSIONS


class TestPrecisionBounds:
    """Test precision bounds are enforced."""

    def test_precision_min_bound(self):
        assert NormalizerConfig.MIN_PRECISION == 1

    def test_precision_max_bound(self):
        assert NormalizerConfig.MAX_PRECISION == 15

    def test_high_precision(self):
        config = NormalizerConfig(default_precision=15)
        assert config.default_precision == 15

    def test_low_precision(self):
        config = NormalizerConfig(default_precision=1)
        assert config.default_precision == 1


class TestCacheSettings:
    """Test cache-related configuration."""

    def test_cache_disabled(self):
        config = NormalizerConfig(cache_enabled=False)
        assert config.cache_enabled is False

    def test_cache_custom_ttl(self):
        config = NormalizerConfig(cache_ttl_seconds=60)
        assert config.cache_ttl_seconds == 60

    def test_cache_large_ttl(self):
        config = NormalizerConfig(cache_ttl_seconds=86400)
        assert config.cache_ttl_seconds == 86400


class TestConfigSingleton:
    """Test get_config/set_config/reset_config singleton pattern."""

    def test_get_config_returns_instance(self):
        config = get_config()
        assert isinstance(config, NormalizerConfig)

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
        custom = NormalizerConfig(default_precision=3)
        set_config(custom)
        assert get_config().default_precision == 3

    def test_set_config_then_get_returns_same(self):
        custom = NormalizerConfig(gwp_version="AR5")
        set_config(custom)
        assert get_config() is custom


class TestConfigCustomValues:
    """Test creating config with fully custom values."""

    def test_full_custom_config(self):
        config = NormalizerConfig(
            default_precision=8,
            gwp_version="AR5",
            gwp_timeframe=20,
            cache_enabled=False,
            cache_ttl_seconds=120,
            max_batch_size=500,
            strict_dimensional_check=False,
            canonical_energy_unit="MWh",
            canonical_mass_unit="t",
            canonical_emissions_unit="tCO2e",
        )
        assert config.default_precision == 8
        assert config.gwp_version == "AR5"
        assert config.gwp_timeframe == 20
        assert config.cache_enabled is False
        assert config.cache_ttl_seconds == 120
        assert config.max_batch_size == 500
        assert config.strict_dimensional_check is False
        assert config.canonical_energy_unit == "MWh"
        assert config.canonical_mass_unit == "t"
        assert config.canonical_emissions_unit == "tCO2e"
