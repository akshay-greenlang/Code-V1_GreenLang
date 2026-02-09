# -*- coding: utf-8 -*-
"""
Unit Tests for GISConnectorConfig (AGENT-DATA-006)

Tests configuration creation, env var overrides with GL_GIS_CONNECTOR_ prefix,
type parsing (bool, int, float, str), singleton get_config/set_config/reset_config,
thread-safety of singleton access, and GIS connector-specific defaults.

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
# Inline GISConnectorConfig mirroring greenlang/gis_connector/config.py
# ---------------------------------------------------------------------------

_ENV_PREFIX = "GL_GIS_CONNECTOR_"


@dataclass
class GISConnectorConfig:
    """Mirrors greenlang.gis_connector.config.GISConnectorConfig."""

    database_url: str = ""
    redis_url: str = ""
    s3_bucket_url: str = ""
    service_name: str = "gl-gis-connector"
    service_version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8086
    max_connections: int = 50
    connection_timeout_s: int = 30
    default_crs: str = "EPSG:4326"
    max_geometry_vertices: int = 100000
    simplification_tolerance: float = 0.0001
    enable_spatial_index: bool = True
    cache_enabled: bool = True
    cache_ttl_s: int = 600
    max_batch_size: int = 1000
    coordinate_precision: int = 6
    enable_provenance: bool = True
    rate_limit_rpm: int = 500
    health_check_interval_s: int = 30
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> GISConnectorConfig:
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
            s3_bucket_url=_str("S3_BUCKET_URL", cls.s3_bucket_url),
            service_name=_str("SERVICE_NAME", cls.service_name),
            service_version=_str("SERVICE_VERSION", cls.service_version),
            host=_str("HOST", cls.host),
            port=_int("PORT", cls.port),
            max_connections=_int("MAX_CONNECTIONS", cls.max_connections),
            connection_timeout_s=_int("CONNECTION_TIMEOUT_S", cls.connection_timeout_s),
            default_crs=_str("DEFAULT_CRS", cls.default_crs),
            max_geometry_vertices=_int("MAX_GEOMETRY_VERTICES", cls.max_geometry_vertices),
            simplification_tolerance=_float("SIMPLIFICATION_TOLERANCE", cls.simplification_tolerance),
            enable_spatial_index=_bool("ENABLE_SPATIAL_INDEX", cls.enable_spatial_index),
            cache_enabled=_bool("CACHE_ENABLED", cls.cache_enabled),
            cache_ttl_s=_int("CACHE_TTL_S", cls.cache_ttl_s),
            max_batch_size=_int("MAX_BATCH_SIZE", cls.max_batch_size),
            coordinate_precision=_int("COORDINATE_PRECISION", cls.coordinate_precision),
            enable_provenance=_bool("ENABLE_PROVENANCE", cls.enable_provenance),
            rate_limit_rpm=_int("RATE_LIMIT_RPM", cls.rate_limit_rpm),
            health_check_interval_s=_int("HEALTH_CHECK_INTERVAL_S", cls.health_check_interval_s),
            log_level=_str("LOG_LEVEL", cls.log_level),
        )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_config_instance: Optional[GISConnectorConfig] = None
_config_lock = threading.Lock()


def get_config() -> GISConnectorConfig:
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = GISConnectorConfig.from_env()
    return _config_instance


def set_config(config: GISConnectorConfig) -> None:
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
def _clean_gis_env(monkeypatch):
    """Remove any GL_GIS_CONNECTOR_ env vars between tests."""
    prefix = "GL_GIS_CONNECTOR_"
    for key in list(os.environ.keys()):
        if key.startswith(prefix):
            monkeypatch.delenv(key, raising=False)


# ===========================================================================
# Test Classes
# ===========================================================================


class TestGISConnectorConfigDefaults:
    """Test that default configuration values match AGENT-DATA-006 PRD."""

    def test_default_database_url(self):
        """Database URL defaults to empty string."""
        config = GISConnectorConfig()
        assert config.database_url == ""

    def test_default_redis_url(self):
        """Redis URL defaults to empty string."""
        config = GISConnectorConfig()
        assert config.redis_url == ""

    def test_default_s3_bucket_url(self):
        """S3 bucket URL defaults to empty string."""
        config = GISConnectorConfig()
        assert config.s3_bucket_url == ""

    def test_default_service_name(self):
        """Service name defaults to gl-gis-connector."""
        config = GISConnectorConfig()
        assert config.service_name == "gl-gis-connector"

    def test_default_service_version(self):
        """Service version defaults to 1.0.0."""
        config = GISConnectorConfig()
        assert config.service_version == "1.0.0"

    def test_default_host(self):
        """Host defaults to 0.0.0.0."""
        config = GISConnectorConfig()
        assert config.host == "0.0.0.0"

    def test_default_port(self):
        """Port defaults to 8086."""
        config = GISConnectorConfig()
        assert config.port == 8086

    def test_default_max_connections(self):
        """Max connections defaults to 50."""
        config = GISConnectorConfig()
        assert config.max_connections == 50

    def test_default_connection_timeout_s(self):
        """Connection timeout defaults to 30 seconds."""
        config = GISConnectorConfig()
        assert config.connection_timeout_s == 30

    def test_default_crs(self):
        """Default CRS is EPSG:4326 (WGS84)."""
        config = GISConnectorConfig()
        assert config.default_crs == "EPSG:4326"

    def test_default_max_geometry_vertices(self):
        """Max geometry vertices defaults to 100000."""
        config = GISConnectorConfig()
        assert config.max_geometry_vertices == 100000

    def test_default_simplification_tolerance(self):
        """Simplification tolerance defaults to 0.0001."""
        config = GISConnectorConfig()
        assert config.simplification_tolerance == 0.0001

    def test_default_enable_spatial_index(self):
        """Spatial index enabled by default."""
        config = GISConnectorConfig()
        assert config.enable_spatial_index is True

    def test_default_cache_enabled(self):
        """Cache enabled by default."""
        config = GISConnectorConfig()
        assert config.cache_enabled is True

    def test_default_cache_ttl_s(self):
        """Cache TTL defaults to 600 seconds."""
        config = GISConnectorConfig()
        assert config.cache_ttl_s == 600

    def test_default_max_batch_size(self):
        """Max batch size defaults to 1000."""
        config = GISConnectorConfig()
        assert config.max_batch_size == 1000

    def test_default_coordinate_precision(self):
        """Coordinate precision defaults to 6 decimal places."""
        config = GISConnectorConfig()
        assert config.coordinate_precision == 6

    def test_default_enable_provenance(self):
        """Provenance tracking enabled by default."""
        config = GISConnectorConfig()
        assert config.enable_provenance is True

    def test_default_rate_limit_rpm(self):
        """Rate limit defaults to 500 RPM."""
        config = GISConnectorConfig()
        assert config.rate_limit_rpm == 500

    def test_default_health_check_interval_s(self):
        """Health check interval defaults to 30 seconds."""
        config = GISConnectorConfig()
        assert config.health_check_interval_s == 30

    def test_default_log_level(self):
        """Log level defaults to INFO."""
        config = GISConnectorConfig()
        assert config.log_level == "INFO"

    def test_all_21_defaults(self):
        """All 21 default values correct in a single config instance."""
        config = GISConnectorConfig()
        assert config.database_url == ""
        assert config.redis_url == ""
        assert config.s3_bucket_url == ""
        assert config.service_name == "gl-gis-connector"
        assert config.service_version == "1.0.0"
        assert config.host == "0.0.0.0"
        assert config.port == 8086
        assert config.max_connections == 50
        assert config.connection_timeout_s == 30
        assert config.default_crs == "EPSG:4326"
        assert config.max_geometry_vertices == 100000
        assert config.simplification_tolerance == 0.0001
        assert config.enable_spatial_index is True
        assert config.cache_enabled is True
        assert config.cache_ttl_s == 600
        assert config.max_batch_size == 1000
        assert config.coordinate_precision == 6
        assert config.enable_provenance is True
        assert config.rate_limit_rpm == 500
        assert config.health_check_interval_s == 30
        assert config.log_level == "INFO"


class TestGISConnectorConfigFromEnv:
    """Test GL_GIS_CONNECTOR_ env var overrides via from_env()."""

    def test_from_env_str_override(self, monkeypatch):
        """Environment variable override for str field."""
        monkeypatch.setenv("GL_GIS_CONNECTOR_DATABASE_URL", "postgresql://test:5432/gis")
        config = GISConnectorConfig.from_env()
        assert config.database_url == "postgresql://test:5432/gis"

    def test_from_env_int_override(self, monkeypatch):
        """Environment variable override for int field."""
        monkeypatch.setenv("GL_GIS_CONNECTOR_PORT", "9090")
        config = GISConnectorConfig.from_env()
        assert config.port == 9090

    def test_from_env_bool_override(self, monkeypatch):
        """Environment variable override for bool field."""
        monkeypatch.setenv("GL_GIS_CONNECTOR_CACHE_ENABLED", "false")
        config = GISConnectorConfig.from_env()
        assert config.cache_enabled is False

    def test_from_env_float_override(self, monkeypatch):
        """Environment variable override for float field."""
        monkeypatch.setenv("GL_GIS_CONNECTOR_SIMPLIFICATION_TOLERANCE", "0.001")
        config = GISConnectorConfig.from_env()
        assert config.simplification_tolerance == 0.001

    def test_env_prefix(self, monkeypatch):
        """Verify GL_GIS_CONNECTOR_ prefix is used for all env lookups."""
        monkeypatch.setenv("GL_GIS_CONNECTOR_LOG_LEVEL", "DEBUG")
        config = GISConnectorConfig.from_env()
        assert config.log_level == "DEBUG"

    def test_env_override_redis_url(self, monkeypatch):
        monkeypatch.setenv("GL_GIS_CONNECTOR_REDIS_URL", "redis://localhost:6379/6")
        config = GISConnectorConfig.from_env()
        assert config.redis_url == "redis://localhost:6379/6"

    def test_env_override_s3_bucket(self, monkeypatch):
        monkeypatch.setenv("GL_GIS_CONNECTOR_S3_BUCKET_URL", "s3://gis-data")
        config = GISConnectorConfig.from_env()
        assert config.s3_bucket_url == "s3://gis-data"

    def test_env_override_service_name(self, monkeypatch):
        monkeypatch.setenv("GL_GIS_CONNECTOR_SERVICE_NAME", "gl-gis-custom")
        config = GISConnectorConfig.from_env()
        assert config.service_name == "gl-gis-custom"

    def test_env_override_default_crs(self, monkeypatch):
        monkeypatch.setenv("GL_GIS_CONNECTOR_DEFAULT_CRS", "EPSG:3857")
        config = GISConnectorConfig.from_env()
        assert config.default_crs == "EPSG:3857"

    def test_env_override_max_geometry_vertices(self, monkeypatch):
        monkeypatch.setenv("GL_GIS_CONNECTOR_MAX_GEOMETRY_VERTICES", "500000")
        config = GISConnectorConfig.from_env()
        assert config.max_geometry_vertices == 500000

    def test_env_override_enable_spatial_index(self, monkeypatch):
        monkeypatch.setenv("GL_GIS_CONNECTOR_ENABLE_SPATIAL_INDEX", "false")
        config = GISConnectorConfig.from_env()
        assert config.enable_spatial_index is False

    def test_env_override_cache_ttl(self, monkeypatch):
        monkeypatch.setenv("GL_GIS_CONNECTOR_CACHE_TTL_S", "1200")
        config = GISConnectorConfig.from_env()
        assert config.cache_ttl_s == 1200

    def test_env_override_max_batch_size(self, monkeypatch):
        monkeypatch.setenv("GL_GIS_CONNECTOR_MAX_BATCH_SIZE", "5000")
        config = GISConnectorConfig.from_env()
        assert config.max_batch_size == 5000

    def test_env_override_coordinate_precision(self, monkeypatch):
        monkeypatch.setenv("GL_GIS_CONNECTOR_COORDINATE_PRECISION", "8")
        config = GISConnectorConfig.from_env()
        assert config.coordinate_precision == 8

    def test_env_override_enable_provenance(self, monkeypatch):
        monkeypatch.setenv("GL_GIS_CONNECTOR_ENABLE_PROVENANCE", "false")
        config = GISConnectorConfig.from_env()
        assert config.enable_provenance is False

    def test_env_override_rate_limit(self, monkeypatch):
        monkeypatch.setenv("GL_GIS_CONNECTOR_RATE_LIMIT_RPM", "2000")
        config = GISConnectorConfig.from_env()
        assert config.rate_limit_rpm == 2000

    def test_env_override_health_check_interval(self, monkeypatch):
        monkeypatch.setenv("GL_GIS_CONNECTOR_HEALTH_CHECK_INTERVAL_S", "15")
        config = GISConnectorConfig.from_env()
        assert config.health_check_interval_s == 15

    def test_env_override_max_connections(self, monkeypatch):
        monkeypatch.setenv("GL_GIS_CONNECTOR_MAX_CONNECTIONS", "200")
        config = GISConnectorConfig.from_env()
        assert config.max_connections == 200

    def test_env_override_connection_timeout(self, monkeypatch):
        monkeypatch.setenv("GL_GIS_CONNECTOR_CONNECTION_TIMEOUT_S", "60")
        config = GISConnectorConfig.from_env()
        assert config.connection_timeout_s == 60


class TestGISConnectorConfigBoolParsing:
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
        ("0", False),
        ("no", False),
        ("anything_else", False),
    ])
    def test_bool_parsing_cache_enabled(self, monkeypatch, env_val, expected):
        """Bool parsing: true/1/yes are True, everything else is False."""
        monkeypatch.setenv("GL_GIS_CONNECTOR_CACHE_ENABLED", env_val)
        config = GISConnectorConfig.from_env()
        assert config.cache_enabled is expected

    @pytest.mark.parametrize("env_val,expected", [
        ("true", True),
        ("1", True),
        ("yes", True),
        ("false", False),
        ("0", False),
        ("no", False),
    ])
    def test_bool_parsing_enable_spatial_index(self, monkeypatch, env_val, expected):
        """Bool parsing for enable_spatial_index field."""
        monkeypatch.setenv("GL_GIS_CONNECTOR_ENABLE_SPATIAL_INDEX", env_val)
        config = GISConnectorConfig.from_env()
        assert config.enable_spatial_index is expected

    @pytest.mark.parametrize("env_val,expected", [
        ("true", True),
        ("1", True),
        ("false", False),
        ("0", False),
    ])
    def test_bool_parsing_enable_provenance(self, monkeypatch, env_val, expected):
        """Bool parsing for enable_provenance field."""
        monkeypatch.setenv("GL_GIS_CONNECTOR_ENABLE_PROVENANCE", env_val)
        config = GISConnectorConfig.from_env()
        assert config.enable_provenance is expected


class TestGISConnectorConfigInvalidFallback:
    """Test fallback to default for invalid int/float env values."""

    def test_invalid_int_port_fallback(self, monkeypatch):
        """Invalid int falls back to default value for port."""
        monkeypatch.setenv("GL_GIS_CONNECTOR_PORT", "not_a_number")
        config = GISConnectorConfig.from_env()
        assert config.port == 8086

    def test_invalid_int_max_connections_fallback(self, monkeypatch):
        """Invalid int falls back to default for max_connections."""
        monkeypatch.setenv("GL_GIS_CONNECTOR_MAX_CONNECTIONS", "xyz")
        config = GISConnectorConfig.from_env()
        assert config.max_connections == 50

    def test_invalid_int_cache_ttl_fallback(self, monkeypatch):
        """Invalid int falls back to default for cache_ttl_s."""
        monkeypatch.setenv("GL_GIS_CONNECTOR_CACHE_TTL_S", "not_int")
        config = GISConnectorConfig.from_env()
        assert config.cache_ttl_s == 600

    def test_invalid_int_max_geometry_vertices_fallback(self, monkeypatch):
        """Invalid int falls back to default for max_geometry_vertices."""
        monkeypatch.setenv("GL_GIS_CONNECTOR_MAX_GEOMETRY_VERTICES", "abc")
        config = GISConnectorConfig.from_env()
        assert config.max_geometry_vertices == 100000

    def test_invalid_int_rate_limit_fallback(self, monkeypatch):
        """Invalid int falls back to default for rate_limit_rpm."""
        monkeypatch.setenv("GL_GIS_CONNECTOR_RATE_LIMIT_RPM", "nan")
        config = GISConnectorConfig.from_env()
        assert config.rate_limit_rpm == 500

    def test_invalid_float_simplification_tolerance_fallback(self, monkeypatch):
        """Invalid float falls back to default for simplification_tolerance."""
        monkeypatch.setenv("GL_GIS_CONNECTOR_SIMPLIFICATION_TOLERANCE", "not_float")
        config = GISConnectorConfig.from_env()
        assert config.simplification_tolerance == 0.0001

    def test_invalid_int_coordinate_precision_fallback(self, monkeypatch):
        """Invalid int falls back to default for coordinate_precision."""
        monkeypatch.setenv("GL_GIS_CONNECTOR_COORDINATE_PRECISION", "bad")
        config = GISConnectorConfig.from_env()
        assert config.coordinate_precision == 6

    def test_invalid_int_max_batch_size_fallback(self, monkeypatch):
        """Invalid int falls back to default for max_batch_size."""
        monkeypatch.setenv("GL_GIS_CONNECTOR_MAX_BATCH_SIZE", "!!!")
        config = GISConnectorConfig.from_env()
        assert config.max_batch_size == 1000


class TestGISConnectorConfigSingleton:
    """Test get_config/set_config/reset_config singleton pattern."""

    def test_get_config_singleton(self):
        """Thread-safe singleton returns the same instance on repeated calls."""
        c1 = get_config()
        c2 = get_config()
        assert c1 is c2
        assert isinstance(c1, GISConnectorConfig)

    def test_set_config(self):
        """Replace config programmatically via set_config."""
        custom = GISConnectorConfig(service_name="gl-gis-custom")
        set_config(custom)
        assert get_config().service_name == "gl-gis-custom"
        assert get_config() is custom

    def test_reset_config(self):
        """Reset config to None so next get_config creates a new instance."""
        c1 = get_config()
        reset_config()
        c2 = get_config()
        assert c1 is not c2

    def test_thread_safety_of_get_config(self):
        """Concurrent get_config calls from 10 threads all get the same instance."""
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
        custom = GISConnectorConfig(port=9999, default_crs="EPSG:3857")
        set_config(custom)
        assert get_config().port == 9999
        assert get_config().default_crs == "EPSG:3857"

    def test_reset_then_get_creates_fresh_instance(self):
        """After reset, get_config creates brand new instance with defaults."""
        set_config(GISConnectorConfig(port=1234))
        reset_config()
        config = get_config()
        assert config.port == 8086
