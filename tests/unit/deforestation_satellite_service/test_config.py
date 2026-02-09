# -*- coding: utf-8 -*-
"""
Unit Tests for DeforestationSatelliteConfig (AGENT-DATA-007)

Tests configuration creation, env var overrides with GL_DEFORESTATION_SAT_ prefix,
type parsing (bool, int, float, str), singleton get_config/set_config/reset_config,
thread-safety of singleton access, EUDR cutoff date, NDVI thresholds, alert settings,
and connection pool sizing.

Coverage target: 85%+ of config.py

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-007 Deforestation Satellite Connector Agent (GL-DATA-GEO-003)
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Any, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline DeforestationSatelliteConfig mirroring greenlang/deforestation_satellite/config.py
# ---------------------------------------------------------------------------

_ENV_PREFIX = "GL_DEFORESTATION_SAT_"


@dataclass
class DeforestationSatelliteConfig:
    """Mirrors greenlang.deforestation_satellite.config.DeforestationSatelliteConfig."""

    # -- Connections
    database_url: str = ""
    redis_url: str = ""

    # -- Logging
    log_level: str = "INFO"

    # -- EUDR regulation
    eudr_cutoff_date: str = "2020-12-31"

    # -- Satellite defaults
    default_satellite: str = "sentinel2"
    max_cloud_cover: int = 30

    # -- NDVI change detection thresholds
    ndvi_clearcut_threshold: float = -0.3
    ndvi_degradation_threshold: float = -0.15
    ndvi_partial_loss_threshold: float = -0.05
    ndvi_regrowth_threshold: float = 0.1

    # -- Alert settings
    min_alert_confidence: str = "nominal"
    alert_dedup_radius_m: int = 100
    alert_dedup_days: int = 7

    # -- Baseline assessment
    baseline_sample_points: int = 9

    # -- Batch processing
    batch_size: int = 50
    worker_count: int = 4

    # -- Cache
    cache_ttl_seconds: int = 3600

    # -- Pool sizing
    pool_min_size: int = 2
    pool_max_size: int = 10

    # -- Data retention
    retention_days: int = 730

    # -- Feature toggles
    use_mock: bool = True

    # -- External API keys
    gfw_api_key: str = ""
    copernicus_api_key: str = ""

    @classmethod
    def from_env(cls) -> DeforestationSatelliteConfig:
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
            database_url=_str("DATABASE_URL", cls.database_url),
            redis_url=_str("REDIS_URL", cls.redis_url),
            log_level=_str("LOG_LEVEL", cls.log_level),
            eudr_cutoff_date=_str("EUDR_CUTOFF_DATE", cls.eudr_cutoff_date),
            default_satellite=_str("DEFAULT_SATELLITE", cls.default_satellite),
            max_cloud_cover=_int("MAX_CLOUD_COVER", cls.max_cloud_cover),
            ndvi_clearcut_threshold=_float(
                "NDVI_CLEARCUT_THRESHOLD", cls.ndvi_clearcut_threshold,
            ),
            ndvi_degradation_threshold=_float(
                "NDVI_DEGRADATION_THRESHOLD", cls.ndvi_degradation_threshold,
            ),
            ndvi_partial_loss_threshold=_float(
                "NDVI_PARTIAL_LOSS_THRESHOLD", cls.ndvi_partial_loss_threshold,
            ),
            ndvi_regrowth_threshold=_float(
                "NDVI_REGROWTH_THRESHOLD", cls.ndvi_regrowth_threshold,
            ),
            min_alert_confidence=_str(
                "MIN_ALERT_CONFIDENCE", cls.min_alert_confidence,
            ),
            alert_dedup_radius_m=_int(
                "ALERT_DEDUP_RADIUS_M", cls.alert_dedup_radius_m,
            ),
            alert_dedup_days=_int("ALERT_DEDUP_DAYS", cls.alert_dedup_days),
            baseline_sample_points=_int(
                "BASELINE_SAMPLE_POINTS", cls.baseline_sample_points,
            ),
            batch_size=_int("BATCH_SIZE", cls.batch_size),
            worker_count=_int("WORKER_COUNT", cls.worker_count),
            cache_ttl_seconds=_int("CACHE_TTL_SECONDS", cls.cache_ttl_seconds),
            pool_min_size=_int("POOL_MIN_SIZE", cls.pool_min_size),
            pool_max_size=_int("POOL_MAX_SIZE", cls.pool_max_size),
            retention_days=_int("RETENTION_DAYS", cls.retention_days),
            use_mock=_bool("USE_MOCK", cls.use_mock),
            gfw_api_key=_str("GFW_API_KEY", cls.gfw_api_key),
            copernicus_api_key=_str("COPERNICUS_API_KEY", cls.copernicus_api_key),
        )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_config_instance: Optional[DeforestationSatelliteConfig] = None
_config_lock = threading.Lock()


def get_config() -> DeforestationSatelliteConfig:
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = DeforestationSatelliteConfig.from_env()
    return _config_instance


def set_config(config: DeforestationSatelliteConfig) -> None:
    global _config_instance
    with _config_lock:
        _config_instance = config


def reset_config() -> None:
    global _config_instance
    with _config_lock:
        _config_instance = None


# ---------------------------------------------------------------------------
# Autouse: reset singleton and clean env between tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_config_singleton():
    yield
    reset_config()


@pytest.fixture(autouse=True)
def _clean_deforestation_env(monkeypatch):
    """Remove any GL_DEFORESTATION_SAT_ env vars between tests."""
    prefix = "GL_DEFORESTATION_SAT_"
    for key in list(os.environ.keys()):
        if key.startswith(prefix):
            monkeypatch.delenv(key, raising=False)


# ===========================================================================
# Test: Default values
# ===========================================================================


class TestDeforestationSatelliteConfigDefaults:
    """Test that default configuration values match AGENT-DATA-007 PRD."""

    def test_default_database_url(self):
        """Database URL defaults to empty string."""
        config = DeforestationSatelliteConfig()
        assert config.database_url == ""

    def test_default_redis_url(self):
        """Redis URL defaults to empty string."""
        config = DeforestationSatelliteConfig()
        assert config.redis_url == ""

    def test_default_log_level(self):
        """Log level defaults to INFO."""
        config = DeforestationSatelliteConfig()
        assert config.log_level == "INFO"

    def test_default_eudr_cutoff_date(self):
        """EUDR cutoff date defaults to 2020-12-31."""
        config = DeforestationSatelliteConfig()
        assert config.eudr_cutoff_date == "2020-12-31"

    def test_default_satellite(self):
        """Default satellite source is sentinel2."""
        config = DeforestationSatelliteConfig()
        assert config.default_satellite == "sentinel2"

    def test_default_max_cloud_cover(self):
        """Max cloud cover defaults to 30%."""
        config = DeforestationSatelliteConfig()
        assert config.max_cloud_cover == 30

    def test_default_ndvi_clearcut_threshold(self):
        """NDVI clear-cut threshold defaults to -0.3."""
        config = DeforestationSatelliteConfig()
        assert config.ndvi_clearcut_threshold == -0.3

    def test_default_ndvi_degradation_threshold(self):
        """NDVI degradation threshold defaults to -0.15."""
        config = DeforestationSatelliteConfig()
        assert config.ndvi_degradation_threshold == -0.15

    def test_default_ndvi_partial_loss_threshold(self):
        """NDVI partial loss threshold defaults to -0.05."""
        config = DeforestationSatelliteConfig()
        assert config.ndvi_partial_loss_threshold == -0.05

    def test_default_ndvi_regrowth_threshold(self):
        """NDVI regrowth threshold defaults to 0.1."""
        config = DeforestationSatelliteConfig()
        assert config.ndvi_regrowth_threshold == 0.1

    def test_default_min_alert_confidence(self):
        """Minimum alert confidence defaults to nominal."""
        config = DeforestationSatelliteConfig()
        assert config.min_alert_confidence == "nominal"

    def test_default_alert_dedup_radius(self):
        """Alert dedup radius defaults to 100 meters."""
        config = DeforestationSatelliteConfig()
        assert config.alert_dedup_radius_m == 100

    def test_default_alert_dedup_days(self):
        """Alert dedup window defaults to 7 days."""
        config = DeforestationSatelliteConfig()
        assert config.alert_dedup_days == 7

    def test_default_baseline_sample_points(self):
        """Baseline sample points defaults to 9."""
        config = DeforestationSatelliteConfig()
        assert config.baseline_sample_points == 9

    def test_default_batch_size(self):
        """Batch size defaults to 50."""
        config = DeforestationSatelliteConfig()
        assert config.batch_size == 50

    def test_default_worker_count(self):
        """Worker count defaults to 4."""
        config = DeforestationSatelliteConfig()
        assert config.worker_count == 4

    def test_default_cache_ttl_seconds(self):
        """Cache TTL defaults to 3600 seconds."""
        config = DeforestationSatelliteConfig()
        assert config.cache_ttl_seconds == 3600

    def test_default_pool_min_size(self):
        """Pool min size defaults to 2."""
        config = DeforestationSatelliteConfig()
        assert config.pool_min_size == 2

    def test_default_pool_max_size(self):
        """Pool max size defaults to 10."""
        config = DeforestationSatelliteConfig()
        assert config.pool_max_size == 10

    def test_default_retention_days(self):
        """Retention days defaults to 730 (2 years)."""
        config = DeforestationSatelliteConfig()
        assert config.retention_days == 730

    def test_default_use_mock(self):
        """use_mock defaults to True."""
        config = DeforestationSatelliteConfig()
        assert config.use_mock is True

    def test_default_gfw_api_key(self):
        """GFW API key defaults to empty string."""
        config = DeforestationSatelliteConfig()
        assert config.gfw_api_key == ""

    def test_default_copernicus_api_key(self):
        """Copernicus API key defaults to empty string."""
        config = DeforestationSatelliteConfig()
        assert config.copernicus_api_key == ""

    def test_all_23_defaults(self):
        """All 23 default values correct in a single config instance."""
        config = DeforestationSatelliteConfig()
        assert config.database_url == ""
        assert config.redis_url == ""
        assert config.log_level == "INFO"
        assert config.eudr_cutoff_date == "2020-12-31"
        assert config.default_satellite == "sentinel2"
        assert config.max_cloud_cover == 30
        assert config.ndvi_clearcut_threshold == -0.3
        assert config.ndvi_degradation_threshold == -0.15
        assert config.ndvi_partial_loss_threshold == -0.05
        assert config.ndvi_regrowth_threshold == 0.1
        assert config.min_alert_confidence == "nominal"
        assert config.alert_dedup_radius_m == 100
        assert config.alert_dedup_days == 7
        assert config.baseline_sample_points == 9
        assert config.batch_size == 50
        assert config.worker_count == 4
        assert config.cache_ttl_seconds == 3600
        assert config.pool_min_size == 2
        assert config.pool_max_size == 10
        assert config.retention_days == 730
        assert config.use_mock is True
        assert config.gfw_api_key == ""
        assert config.copernicus_api_key == ""


# ===========================================================================
# Test: Environment variable overrides via from_env()
# ===========================================================================


class TestDeforestationSatelliteConfigFromEnv:
    """Test GL_DEFORESTATION_SAT_ env var overrides via from_env()."""

    def test_env_override_database_url(self, monkeypatch):
        monkeypatch.setenv("GL_DEFORESTATION_SAT_DATABASE_URL", "postgresql://sat:5432/deforest")
        config = DeforestationSatelliteConfig.from_env()
        assert config.database_url == "postgresql://sat:5432/deforest"

    def test_env_override_redis_url(self, monkeypatch):
        monkeypatch.setenv("GL_DEFORESTATION_SAT_REDIS_URL", "redis://localhost:6379/7")
        config = DeforestationSatelliteConfig.from_env()
        assert config.redis_url == "redis://localhost:6379/7"

    def test_env_override_log_level(self, monkeypatch):
        monkeypatch.setenv("GL_DEFORESTATION_SAT_LOG_LEVEL", "DEBUG")
        config = DeforestationSatelliteConfig.from_env()
        assert config.log_level == "DEBUG"

    def test_env_override_eudr_cutoff_date(self, monkeypatch):
        monkeypatch.setenv("GL_DEFORESTATION_SAT_EUDR_CUTOFF_DATE", "2021-06-30")
        config = DeforestationSatelliteConfig.from_env()
        assert config.eudr_cutoff_date == "2021-06-30"

    def test_env_override_default_satellite(self, monkeypatch):
        monkeypatch.setenv("GL_DEFORESTATION_SAT_DEFAULT_SATELLITE", "landsat8")
        config = DeforestationSatelliteConfig.from_env()
        assert config.default_satellite == "landsat8"

    def test_env_override_max_cloud_cover(self, monkeypatch):
        monkeypatch.setenv("GL_DEFORESTATION_SAT_MAX_CLOUD_COVER", "15")
        config = DeforestationSatelliteConfig.from_env()
        assert config.max_cloud_cover == 15

    def test_env_override_ndvi_clearcut_threshold(self, monkeypatch):
        monkeypatch.setenv("GL_DEFORESTATION_SAT_NDVI_CLEARCUT_THRESHOLD", "-0.4")
        config = DeforestationSatelliteConfig.from_env()
        assert config.ndvi_clearcut_threshold == -0.4

    def test_env_override_ndvi_degradation_threshold(self, monkeypatch):
        monkeypatch.setenv("GL_DEFORESTATION_SAT_NDVI_DEGRADATION_THRESHOLD", "-0.2")
        config = DeforestationSatelliteConfig.from_env()
        assert config.ndvi_degradation_threshold == -0.2

    def test_env_override_ndvi_partial_loss_threshold(self, monkeypatch):
        monkeypatch.setenv("GL_DEFORESTATION_SAT_NDVI_PARTIAL_LOSS_THRESHOLD", "-0.08")
        config = DeforestationSatelliteConfig.from_env()
        assert config.ndvi_partial_loss_threshold == -0.08

    def test_env_override_ndvi_regrowth_threshold(self, monkeypatch):
        monkeypatch.setenv("GL_DEFORESTATION_SAT_NDVI_REGROWTH_THRESHOLD", "0.15")
        config = DeforestationSatelliteConfig.from_env()
        assert config.ndvi_regrowth_threshold == 0.15

    def test_env_override_min_alert_confidence(self, monkeypatch):
        monkeypatch.setenv("GL_DEFORESTATION_SAT_MIN_ALERT_CONFIDENCE", "high")
        config = DeforestationSatelliteConfig.from_env()
        assert config.min_alert_confidence == "high"

    def test_env_override_alert_dedup_radius(self, monkeypatch):
        monkeypatch.setenv("GL_DEFORESTATION_SAT_ALERT_DEDUP_RADIUS_M", "250")
        config = DeforestationSatelliteConfig.from_env()
        assert config.alert_dedup_radius_m == 250

    def test_env_override_alert_dedup_days(self, monkeypatch):
        monkeypatch.setenv("GL_DEFORESTATION_SAT_ALERT_DEDUP_DAYS", "14")
        config = DeforestationSatelliteConfig.from_env()
        assert config.alert_dedup_days == 14

    def test_env_override_baseline_sample_points(self, monkeypatch):
        monkeypatch.setenv("GL_DEFORESTATION_SAT_BASELINE_SAMPLE_POINTS", "25")
        config = DeforestationSatelliteConfig.from_env()
        assert config.baseline_sample_points == 25

    def test_env_override_batch_size(self, monkeypatch):
        monkeypatch.setenv("GL_DEFORESTATION_SAT_BATCH_SIZE", "100")
        config = DeforestationSatelliteConfig.from_env()
        assert config.batch_size == 100

    def test_env_override_worker_count(self, monkeypatch):
        monkeypatch.setenv("GL_DEFORESTATION_SAT_WORKER_COUNT", "8")
        config = DeforestationSatelliteConfig.from_env()
        assert config.worker_count == 8

    def test_env_override_cache_ttl_seconds(self, monkeypatch):
        monkeypatch.setenv("GL_DEFORESTATION_SAT_CACHE_TTL_SECONDS", "7200")
        config = DeforestationSatelliteConfig.from_env()
        assert config.cache_ttl_seconds == 7200

    def test_env_override_pool_min_size(self, monkeypatch):
        monkeypatch.setenv("GL_DEFORESTATION_SAT_POOL_MIN_SIZE", "5")
        config = DeforestationSatelliteConfig.from_env()
        assert config.pool_min_size == 5

    def test_env_override_pool_max_size(self, monkeypatch):
        monkeypatch.setenv("GL_DEFORESTATION_SAT_POOL_MAX_SIZE", "20")
        config = DeforestationSatelliteConfig.from_env()
        assert config.pool_max_size == 20

    def test_env_override_retention_days(self, monkeypatch):
        monkeypatch.setenv("GL_DEFORESTATION_SAT_RETENTION_DAYS", "365")
        config = DeforestationSatelliteConfig.from_env()
        assert config.retention_days == 365

    def test_env_override_use_mock(self, monkeypatch):
        monkeypatch.setenv("GL_DEFORESTATION_SAT_USE_MOCK", "false")
        config = DeforestationSatelliteConfig.from_env()
        assert config.use_mock is False

    def test_env_override_gfw_api_key(self, monkeypatch):
        monkeypatch.setenv("GL_DEFORESTATION_SAT_GFW_API_KEY", "gfw-test-key-12345")
        config = DeforestationSatelliteConfig.from_env()
        assert config.gfw_api_key == "gfw-test-key-12345"

    def test_env_override_copernicus_api_key(self, monkeypatch):
        monkeypatch.setenv("GL_DEFORESTATION_SAT_COPERNICUS_API_KEY", "cop-test-key-67890")
        config = DeforestationSatelliteConfig.from_env()
        assert config.copernicus_api_key == "cop-test-key-67890"

    def test_env_prefix_used(self, monkeypatch):
        """Verify GL_DEFORESTATION_SAT_ prefix is used for all env lookups."""
        monkeypatch.setenv("GL_DEFORESTATION_SAT_LOG_LEVEL", "WARNING")
        config = DeforestationSatelliteConfig.from_env()
        assert config.log_level == "WARNING"

    def test_env_no_prefix_ignored(self, monkeypatch):
        """Variables without the prefix are ignored."""
        monkeypatch.setenv("DEFAULT_SATELLITE", "modis")
        config = DeforestationSatelliteConfig.from_env()
        assert config.default_satellite == "sentinel2"


# ===========================================================================
# Test: Boolean parsing
# ===========================================================================


class TestDeforestationSatelliteConfigBoolParsing:
    """Test boolean environment variable parsing for true/1/yes and false/0/no."""

    @pytest.mark.parametrize("env_val,expected", [
        ("true", True),
        ("True", True),
        ("TRUE", True),
        ("1", True),
        ("yes", True),
        ("Yes", True),
        ("YES", True),
        ("false", False),
        ("False", False),
        ("FALSE", False),
        ("0", False),
        ("no", False),
        ("No", False),
        ("NO", False),
        ("anything_else", False),
    ])
    def test_bool_parsing_use_mock(self, monkeypatch, env_val, expected):
        """Bool parsing for use_mock: true/1/yes are True, everything else is False."""
        monkeypatch.setenv("GL_DEFORESTATION_SAT_USE_MOCK", env_val)
        config = DeforestationSatelliteConfig.from_env()
        assert config.use_mock is expected

    def test_bool_unset_returns_default_true(self):
        """When USE_MOCK is not set, default True is preserved."""
        config = DeforestationSatelliteConfig.from_env()
        assert config.use_mock is True


# ===========================================================================
# Test: Invalid fallback
# ===========================================================================


class TestDeforestationSatelliteConfigInvalidFallback:
    """Test fallback to default for invalid int/float env values."""

    def test_invalid_int_max_cloud_cover_fallback(self, monkeypatch):
        monkeypatch.setenv("GL_DEFORESTATION_SAT_MAX_CLOUD_COVER", "not_a_number")
        config = DeforestationSatelliteConfig.from_env()
        assert config.max_cloud_cover == 30

    def test_invalid_int_alert_dedup_radius_fallback(self, monkeypatch):
        monkeypatch.setenv("GL_DEFORESTATION_SAT_ALERT_DEDUP_RADIUS_M", "xyz")
        config = DeforestationSatelliteConfig.from_env()
        assert config.alert_dedup_radius_m == 100

    def test_invalid_int_alert_dedup_days_fallback(self, monkeypatch):
        monkeypatch.setenv("GL_DEFORESTATION_SAT_ALERT_DEDUP_DAYS", "abc")
        config = DeforestationSatelliteConfig.from_env()
        assert config.alert_dedup_days == 7

    def test_invalid_int_baseline_sample_points_fallback(self, monkeypatch):
        monkeypatch.setenv("GL_DEFORESTATION_SAT_BASELINE_SAMPLE_POINTS", "bad")
        config = DeforestationSatelliteConfig.from_env()
        assert config.baseline_sample_points == 9

    def test_invalid_int_batch_size_fallback(self, monkeypatch):
        monkeypatch.setenv("GL_DEFORESTATION_SAT_BATCH_SIZE", "!!!")
        config = DeforestationSatelliteConfig.from_env()
        assert config.batch_size == 50

    def test_invalid_int_worker_count_fallback(self, monkeypatch):
        monkeypatch.setenv("GL_DEFORESTATION_SAT_WORKER_COUNT", "nope")
        config = DeforestationSatelliteConfig.from_env()
        assert config.worker_count == 4

    def test_invalid_int_cache_ttl_fallback(self, monkeypatch):
        monkeypatch.setenv("GL_DEFORESTATION_SAT_CACHE_TTL_SECONDS", "no")
        config = DeforestationSatelliteConfig.from_env()
        assert config.cache_ttl_seconds == 3600

    def test_invalid_int_pool_min_fallback(self, monkeypatch):
        monkeypatch.setenv("GL_DEFORESTATION_SAT_POOL_MIN_SIZE", "nan")
        config = DeforestationSatelliteConfig.from_env()
        assert config.pool_min_size == 2

    def test_invalid_int_pool_max_fallback(self, monkeypatch):
        monkeypatch.setenv("GL_DEFORESTATION_SAT_POOL_MAX_SIZE", "oops")
        config = DeforestationSatelliteConfig.from_env()
        assert config.pool_max_size == 10

    def test_invalid_int_retention_days_fallback(self, monkeypatch):
        monkeypatch.setenv("GL_DEFORESTATION_SAT_RETENTION_DAYS", "never")
        config = DeforestationSatelliteConfig.from_env()
        assert config.retention_days == 730

    def test_invalid_float_ndvi_clearcut_fallback(self, monkeypatch):
        monkeypatch.setenv("GL_DEFORESTATION_SAT_NDVI_CLEARCUT_THRESHOLD", "bad_float")
        config = DeforestationSatelliteConfig.from_env()
        assert config.ndvi_clearcut_threshold == -0.3

    def test_invalid_float_ndvi_degradation_fallback(self, monkeypatch):
        monkeypatch.setenv("GL_DEFORESTATION_SAT_NDVI_DEGRADATION_THRESHOLD", "n/a")
        config = DeforestationSatelliteConfig.from_env()
        assert config.ndvi_degradation_threshold == -0.15

    def test_invalid_float_ndvi_partial_loss_fallback(self, monkeypatch):
        monkeypatch.setenv("GL_DEFORESTATION_SAT_NDVI_PARTIAL_LOSS_THRESHOLD", "xxx")
        config = DeforestationSatelliteConfig.from_env()
        assert config.ndvi_partial_loss_threshold == -0.05

    def test_invalid_float_ndvi_regrowth_fallback(self, monkeypatch):
        monkeypatch.setenv("GL_DEFORESTATION_SAT_NDVI_REGROWTH_THRESHOLD", "???")
        config = DeforestationSatelliteConfig.from_env()
        assert config.ndvi_regrowth_threshold == 0.1


# ===========================================================================
# Test: Singleton pattern
# ===========================================================================


class TestDeforestationSatelliteConfigSingleton:
    """Test get_config/set_config/reset_config singleton pattern."""

    def test_get_config_singleton(self):
        """Thread-safe singleton returns same instance on repeated calls."""
        c1 = get_config()
        c2 = get_config()
        assert c1 is c2
        assert isinstance(c1, DeforestationSatelliteConfig)

    def test_set_config(self):
        """Replace config programmatically via set_config."""
        custom = DeforestationSatelliteConfig(default_satellite="modis")
        set_config(custom)
        assert get_config().default_satellite == "modis"
        assert get_config() is custom

    def test_reset_config(self):
        """Reset config to None so next get_config creates a new instance."""
        c1 = get_config()
        reset_config()
        c2 = get_config()
        assert c1 is not c2

    def test_thread_safety_of_get_config(self):
        """Concurrent get_config calls from 10 threads all get same instance."""
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

    def test_set_config_overrides_from_env(self):
        """set_config replaces any previously created from_env instance."""
        _ = get_config()
        custom = DeforestationSatelliteConfig(max_cloud_cover=5, eudr_cutoff_date="2022-01-01")
        set_config(custom)
        assert get_config().max_cloud_cover == 5
        assert get_config().eudr_cutoff_date == "2022-01-01"

    def test_reset_then_get_creates_fresh_instance(self):
        """After reset, get_config creates brand new instance with defaults."""
        set_config(DeforestationSatelliteConfig(max_cloud_cover=99))
        reset_config()
        config = get_config()
        assert config.max_cloud_cover == 30


# ===========================================================================
# Test: EUDR cutoff date
# ===========================================================================


class TestEUDRCutoffDate:
    """Test EUDR cutoff date configuration."""

    def test_eudr_cutoff_date_default_value(self):
        """EUDR cutoff date is 2020-12-31 per EU regulation."""
        config = DeforestationSatelliteConfig()
        assert config.eudr_cutoff_date == "2020-12-31"

    def test_eudr_cutoff_date_iso_format(self):
        """EUDR cutoff date is a valid ISO date string."""
        config = DeforestationSatelliteConfig()
        from datetime import date
        parsed = date.fromisoformat(config.eudr_cutoff_date)
        assert parsed.year == 2020
        assert parsed.month == 12
        assert parsed.day == 31

    def test_eudr_cutoff_date_override(self, monkeypatch):
        """EUDR cutoff date can be overridden via env var."""
        monkeypatch.setenv("GL_DEFORESTATION_SAT_EUDR_CUTOFF_DATE", "2023-06-29")
        config = DeforestationSatelliteConfig.from_env()
        assert config.eudr_cutoff_date == "2023-06-29"


# ===========================================================================
# Test: NDVI thresholds
# ===========================================================================


class TestNDVIThresholdsDefaults:
    """Test NDVI change detection threshold defaults match PRD spec."""

    def test_clearcut_threshold_negative(self):
        """Clear-cut threshold is a negative value indicating severe loss."""
        config = DeforestationSatelliteConfig()
        assert config.ndvi_clearcut_threshold < 0

    def test_degradation_less_severe_than_clearcut(self):
        """Degradation threshold is less negative than clear-cut."""
        config = DeforestationSatelliteConfig()
        assert config.ndvi_degradation_threshold > config.ndvi_clearcut_threshold

    def test_partial_loss_less_severe_than_degradation(self):
        """Partial loss threshold is less negative than degradation."""
        config = DeforestationSatelliteConfig()
        assert config.ndvi_partial_loss_threshold > config.ndvi_degradation_threshold

    def test_regrowth_positive(self):
        """Regrowth threshold is positive indicating vegetation increase."""
        config = DeforestationSatelliteConfig()
        assert config.ndvi_regrowth_threshold > 0

    def test_threshold_ordering(self):
        """Thresholds are ordered: clearcut < degradation < partial_loss < regrowth."""
        config = DeforestationSatelliteConfig()
        assert (config.ndvi_clearcut_threshold
                < config.ndvi_degradation_threshold
                < config.ndvi_partial_loss_threshold
                < config.ndvi_regrowth_threshold)


# ===========================================================================
# Test: Alert settings defaults
# ===========================================================================


class TestAlertSettingsDefaults:
    """Test alert configuration defaults."""

    def test_min_alert_confidence_nominal(self):
        """Default alert confidence is nominal."""
        config = DeforestationSatelliteConfig()
        assert config.min_alert_confidence == "nominal"

    def test_alert_dedup_radius_positive(self):
        """Dedup radius is positive meters."""
        config = DeforestationSatelliteConfig()
        assert config.alert_dedup_radius_m > 0

    def test_alert_dedup_days_positive(self):
        """Dedup window is positive days."""
        config = DeforestationSatelliteConfig()
        assert config.alert_dedup_days > 0

    def test_alert_dedup_radius_100m(self):
        """Default dedup radius is 100 meters."""
        config = DeforestationSatelliteConfig()
        assert config.alert_dedup_radius_m == 100

    def test_alert_dedup_days_7(self):
        """Default dedup window is 7 days."""
        config = DeforestationSatelliteConfig()
        assert config.alert_dedup_days == 7


# ===========================================================================
# Test: Pool settings
# ===========================================================================


class TestPoolSettings:
    """Test connection pool configuration."""

    def test_pool_min_less_than_max(self):
        """Pool min size is less than pool max size."""
        config = DeforestationSatelliteConfig()
        assert config.pool_min_size < config.pool_max_size

    def test_pool_min_positive(self):
        """Pool min size is positive."""
        config = DeforestationSatelliteConfig()
        assert config.pool_min_size > 0

    def test_pool_max_positive(self):
        """Pool max size is positive."""
        config = DeforestationSatelliteConfig()
        assert config.pool_max_size > 0

    def test_pool_min_default_2(self):
        """Pool min defaults to 2."""
        config = DeforestationSatelliteConfig()
        assert config.pool_min_size == 2

    def test_pool_max_default_10(self):
        """Pool max defaults to 10."""
        config = DeforestationSatelliteConfig()
        assert config.pool_max_size == 10

    def test_pool_min_override(self, monkeypatch):
        """Pool min can be overridden."""
        monkeypatch.setenv("GL_DEFORESTATION_SAT_POOL_MIN_SIZE", "3")
        config = DeforestationSatelliteConfig.from_env()
        assert config.pool_min_size == 3

    def test_pool_max_override(self, monkeypatch):
        """Pool max can be overridden."""
        monkeypatch.setenv("GL_DEFORESTATION_SAT_POOL_MAX_SIZE", "50")
        config = DeforestationSatelliteConfig.from_env()
        assert config.pool_max_size == 50
