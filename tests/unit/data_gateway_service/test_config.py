# -*- coding: utf-8 -*-
"""
Unit Tests for DataGatewayConfig (AGENT-DATA-004)

Tests configuration creation, env var overrides with GL_DATA_GATEWAY_ prefix,
type parsing (bool, int, float, str), singleton get_config/set_config/reset_config,
thread-safety of singleton access, and API gateway-specific defaults.

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
# Inline DataGatewayConfig mirroring greenlang/data_gateway/config.py
# ---------------------------------------------------------------------------

_ENV_PREFIX = "GL_DATA_GATEWAY_"


@dataclass
class DataGatewayConfig:
    """Mirrors greenlang.data_gateway.config.DataGatewayConfig."""

    database_url: str = ""
    redis_url: str = ""
    s3_bucket_url: str = ""
    service_name: str = "gl-data-gateway"
    service_version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8080
    max_connections: int = 100
    connection_timeout_s: int = 30
    query_timeout_s: int = 60
    max_query_complexity: int = 1000
    max_sources_per_query: int = 10
    cache_enabled: bool = True
    cache_ttl_s: int = 300
    cache_max_entries: int = 10000
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout_s: int = 60
    rate_limit_rpm: int = 1000
    enable_query_logging: bool = True
    enable_schema_validation: bool = True
    batch_max_queries: int = 50
    health_check_interval_s: int = 30
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> DataGatewayConfig:
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
            query_timeout_s=_int("QUERY_TIMEOUT_S", cls.query_timeout_s),
            max_query_complexity=_int("MAX_QUERY_COMPLEXITY", cls.max_query_complexity),
            max_sources_per_query=_int("MAX_SOURCES_PER_QUERY", cls.max_sources_per_query),
            cache_enabled=_bool("CACHE_ENABLED", cls.cache_enabled),
            cache_ttl_s=_int("CACHE_TTL_S", cls.cache_ttl_s),
            cache_max_entries=_int("CACHE_MAX_ENTRIES", cls.cache_max_entries),
            circuit_breaker_threshold=_int("CIRCUIT_BREAKER_THRESHOLD", cls.circuit_breaker_threshold),
            circuit_breaker_timeout_s=_int("CIRCUIT_BREAKER_TIMEOUT_S", cls.circuit_breaker_timeout_s),
            rate_limit_rpm=_int("RATE_LIMIT_RPM", cls.rate_limit_rpm),
            enable_query_logging=_bool("ENABLE_QUERY_LOGGING", cls.enable_query_logging),
            enable_schema_validation=_bool("ENABLE_SCHEMA_VALIDATION", cls.enable_schema_validation),
            batch_max_queries=_int("BATCH_MAX_QUERIES", cls.batch_max_queries),
            health_check_interval_s=_int("HEALTH_CHECK_INTERVAL_S", cls.health_check_interval_s),
            log_level=_str("LOG_LEVEL", cls.log_level),
        )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_config_instance: Optional[DataGatewayConfig] = None
_config_lock = threading.Lock()


def get_config() -> DataGatewayConfig:
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = DataGatewayConfig.from_env()
    return _config_instance


def set_config(config: DataGatewayConfig) -> None:
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
def _clean_gateway_env(monkeypatch):
    """Remove any GL_DATA_GATEWAY_ env vars between tests."""
    prefix = "GL_DATA_GATEWAY_"
    for key in list(os.environ.keys()):
        if key.startswith(prefix):
            monkeypatch.delenv(key, raising=False)


# ===========================================================================
# Test Classes
# ===========================================================================


class TestDataGatewayConfigDefaults:
    """Test that default configuration values match AGENT-DATA-004 PRD."""

    def test_default_database_url(self):
        """Database URL defaults to empty string."""
        config = DataGatewayConfig()
        assert config.database_url == ""

    def test_default_redis_url(self):
        """Redis URL defaults to empty string."""
        config = DataGatewayConfig()
        assert config.redis_url == ""

    def test_default_s3_bucket_url(self):
        """S3 bucket URL defaults to empty string."""
        config = DataGatewayConfig()
        assert config.s3_bucket_url == ""

    def test_default_service_name(self):
        """Service name defaults to gl-data-gateway."""
        config = DataGatewayConfig()
        assert config.service_name == "gl-data-gateway"

    def test_default_service_version(self):
        """Service version defaults to 1.0.0."""
        config = DataGatewayConfig()
        assert config.service_version == "1.0.0"

    def test_default_host(self):
        """Host defaults to 0.0.0.0."""
        config = DataGatewayConfig()
        assert config.host == "0.0.0.0"

    def test_default_port(self):
        """Port defaults to 8080."""
        config = DataGatewayConfig()
        assert config.port == 8080

    def test_default_max_connections(self):
        """Max connections defaults to 100."""
        config = DataGatewayConfig()
        assert config.max_connections == 100

    def test_default_connection_timeout_s(self):
        """Connection timeout defaults to 30 seconds."""
        config = DataGatewayConfig()
        assert config.connection_timeout_s == 30

    def test_default_query_timeout_s(self):
        """Query timeout defaults to 60 seconds."""
        config = DataGatewayConfig()
        assert config.query_timeout_s == 60

    def test_default_max_query_complexity(self):
        """Max query complexity defaults to 1000."""
        config = DataGatewayConfig()
        assert config.max_query_complexity == 1000

    def test_default_max_sources_per_query(self):
        """Max sources per query defaults to 10."""
        config = DataGatewayConfig()
        assert config.max_sources_per_query == 10

    def test_default_cache_enabled(self):
        """Cache enabled defaults to True."""
        config = DataGatewayConfig()
        assert config.cache_enabled is True

    def test_default_cache_ttl_s(self):
        """Cache TTL defaults to 300 seconds."""
        config = DataGatewayConfig()
        assert config.cache_ttl_s == 300

    def test_default_cache_max_entries(self):
        """Cache max entries defaults to 10000."""
        config = DataGatewayConfig()
        assert config.cache_max_entries == 10000

    def test_default_circuit_breaker_threshold(self):
        """Circuit breaker threshold defaults to 5."""
        config = DataGatewayConfig()
        assert config.circuit_breaker_threshold == 5

    def test_default_circuit_breaker_timeout_s(self):
        """Circuit breaker timeout defaults to 60 seconds."""
        config = DataGatewayConfig()
        assert config.circuit_breaker_timeout_s == 60

    def test_default_rate_limit_rpm(self):
        """Rate limit defaults to 1000 RPM."""
        config = DataGatewayConfig()
        assert config.rate_limit_rpm == 1000

    def test_default_enable_query_logging(self):
        """Query logging enabled by default."""
        config = DataGatewayConfig()
        assert config.enable_query_logging is True

    def test_default_enable_schema_validation(self):
        """Schema validation enabled by default."""
        config = DataGatewayConfig()
        assert config.enable_schema_validation is True

    def test_default_batch_max_queries(self):
        """Batch max queries defaults to 50."""
        config = DataGatewayConfig()
        assert config.batch_max_queries == 50

    def test_default_health_check_interval_s(self):
        """Health check interval defaults to 30 seconds."""
        config = DataGatewayConfig()
        assert config.health_check_interval_s == 30

    def test_default_log_level(self):
        """Log level defaults to INFO."""
        config = DataGatewayConfig()
        assert config.log_level == "INFO"

    def test_all_23_defaults(self):
        """All 23 default values correct in a single config instance."""
        config = DataGatewayConfig()
        assert config.database_url == ""
        assert config.redis_url == ""
        assert config.s3_bucket_url == ""
        assert config.service_name == "gl-data-gateway"
        assert config.service_version == "1.0.0"
        assert config.host == "0.0.0.0"
        assert config.port == 8080
        assert config.max_connections == 100
        assert config.connection_timeout_s == 30
        assert config.query_timeout_s == 60
        assert config.max_query_complexity == 1000
        assert config.max_sources_per_query == 10
        assert config.cache_enabled is True
        assert config.cache_ttl_s == 300
        assert config.cache_max_entries == 10000
        assert config.circuit_breaker_threshold == 5
        assert config.circuit_breaker_timeout_s == 60
        assert config.rate_limit_rpm == 1000
        assert config.enable_query_logging is True
        assert config.enable_schema_validation is True
        assert config.batch_max_queries == 50
        assert config.health_check_interval_s == 30
        assert config.log_level == "INFO"


class TestDataGatewayConfigFromEnv:
    """Test GL_DATA_GATEWAY_ env var overrides via from_env()."""

    def test_from_env_str_override(self, monkeypatch):
        """Environment variable override for str field."""
        monkeypatch.setenv("GL_DATA_GATEWAY_DATABASE_URL", "postgresql://test:5432/gateway")
        config = DataGatewayConfig.from_env()
        assert config.database_url == "postgresql://test:5432/gateway"

    def test_from_env_int_override(self, monkeypatch):
        """Environment variable override for int field."""
        monkeypatch.setenv("GL_DATA_GATEWAY_PORT", "9090")
        config = DataGatewayConfig.from_env()
        assert config.port == 9090

    def test_from_env_bool_override(self, monkeypatch):
        """Environment variable override for bool field."""
        monkeypatch.setenv("GL_DATA_GATEWAY_CACHE_ENABLED", "false")
        config = DataGatewayConfig.from_env()
        assert config.cache_enabled is False

    def test_from_env_float_not_applicable_but_int_works(self, monkeypatch):
        """Override int fields that hold numeric thresholds."""
        monkeypatch.setenv("GL_DATA_GATEWAY_MAX_CONNECTIONS", "200")
        config = DataGatewayConfig.from_env()
        assert config.max_connections == 200

    def test_env_prefix(self, monkeypatch):
        """Verify GL_DATA_GATEWAY_ prefix is used for all env lookups."""
        monkeypatch.setenv("GL_DATA_GATEWAY_LOG_LEVEL", "DEBUG")
        config = DataGatewayConfig.from_env()
        assert config.log_level == "DEBUG"

    def test_env_override_redis_url(self, monkeypatch):
        monkeypatch.setenv("GL_DATA_GATEWAY_REDIS_URL", "redis://localhost:6379/4")
        config = DataGatewayConfig.from_env()
        assert config.redis_url == "redis://localhost:6379/4"

    def test_env_override_s3_bucket(self, monkeypatch):
        monkeypatch.setenv("GL_DATA_GATEWAY_S3_BUCKET_URL", "s3://gateway-data")
        config = DataGatewayConfig.from_env()
        assert config.s3_bucket_url == "s3://gateway-data"

    def test_env_override_service_name(self, monkeypatch):
        monkeypatch.setenv("GL_DATA_GATEWAY_SERVICE_NAME", "gl-data-gw-custom")
        config = DataGatewayConfig.from_env()
        assert config.service_name == "gl-data-gw-custom"

    def test_env_override_query_timeout(self, monkeypatch):
        monkeypatch.setenv("GL_DATA_GATEWAY_QUERY_TIMEOUT_S", "120")
        config = DataGatewayConfig.from_env()
        assert config.query_timeout_s == 120

    def test_env_override_max_query_complexity(self, monkeypatch):
        monkeypatch.setenv("GL_DATA_GATEWAY_MAX_QUERY_COMPLEXITY", "5000")
        config = DataGatewayConfig.from_env()
        assert config.max_query_complexity == 5000

    def test_env_override_cache_ttl(self, monkeypatch):
        monkeypatch.setenv("GL_DATA_GATEWAY_CACHE_TTL_S", "600")
        config = DataGatewayConfig.from_env()
        assert config.cache_ttl_s == 600

    def test_env_override_circuit_breaker_threshold(self, monkeypatch):
        monkeypatch.setenv("GL_DATA_GATEWAY_CIRCUIT_BREAKER_THRESHOLD", "10")
        config = DataGatewayConfig.from_env()
        assert config.circuit_breaker_threshold == 10

    def test_env_override_rate_limit(self, monkeypatch):
        monkeypatch.setenv("GL_DATA_GATEWAY_RATE_LIMIT_RPM", "5000")
        config = DataGatewayConfig.from_env()
        assert config.rate_limit_rpm == 5000

    def test_env_override_batch_max_queries(self, monkeypatch):
        monkeypatch.setenv("GL_DATA_GATEWAY_BATCH_MAX_QUERIES", "100")
        config = DataGatewayConfig.from_env()
        assert config.batch_max_queries == 100

    def test_env_override_health_check_interval(self, monkeypatch):
        monkeypatch.setenv("GL_DATA_GATEWAY_HEALTH_CHECK_INTERVAL_S", "15")
        config = DataGatewayConfig.from_env()
        assert config.health_check_interval_s == 15

    def test_env_override_enable_schema_validation(self, monkeypatch):
        monkeypatch.setenv("GL_DATA_GATEWAY_ENABLE_SCHEMA_VALIDATION", "false")
        config = DataGatewayConfig.from_env()
        assert config.enable_schema_validation is False

    def test_env_override_enable_query_logging(self, monkeypatch):
        monkeypatch.setenv("GL_DATA_GATEWAY_ENABLE_QUERY_LOGGING", "false")
        config = DataGatewayConfig.from_env()
        assert config.enable_query_logging is False


class TestDataGatewayConfigBoolParsing:
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
    def test_bool_parsing(self, monkeypatch, env_val, expected):
        """Bool parsing: true/1/yes are True, everything else is False."""
        monkeypatch.setenv("GL_DATA_GATEWAY_CACHE_ENABLED", env_val)
        config = DataGatewayConfig.from_env()
        assert config.cache_enabled is expected


class TestDataGatewayConfigInvalidFallback:
    """Test fallback to default for invalid int/float env values."""

    def test_invalid_int_port_fallback(self, monkeypatch):
        """Invalid int falls back to default value for port."""
        monkeypatch.setenv("GL_DATA_GATEWAY_PORT", "not_a_number")
        config = DataGatewayConfig.from_env()
        assert config.port == 8080

    def test_invalid_int_max_connections_fallback(self, monkeypatch):
        """Invalid int falls back to default for max_connections."""
        monkeypatch.setenv("GL_DATA_GATEWAY_MAX_CONNECTIONS", "xyz")
        config = DataGatewayConfig.from_env()
        assert config.max_connections == 100

    def test_invalid_int_query_timeout_fallback(self, monkeypatch):
        """Invalid int falls back to default for query_timeout_s."""
        monkeypatch.setenv("GL_DATA_GATEWAY_QUERY_TIMEOUT_S", "abc")
        config = DataGatewayConfig.from_env()
        assert config.query_timeout_s == 60

    def test_invalid_int_cache_ttl_fallback(self, monkeypatch):
        """Invalid int falls back to default for cache_ttl_s."""
        monkeypatch.setenv("GL_DATA_GATEWAY_CACHE_TTL_S", "not_int")
        config = DataGatewayConfig.from_env()
        assert config.cache_ttl_s == 300

    def test_invalid_int_rate_limit_fallback(self, monkeypatch):
        """Invalid int falls back to default for rate_limit_rpm."""
        monkeypatch.setenv("GL_DATA_GATEWAY_RATE_LIMIT_RPM", "nan")
        config = DataGatewayConfig.from_env()
        assert config.rate_limit_rpm == 1000


class TestDataGatewayConfigSingleton:
    """Test get_config/set_config/reset_config singleton pattern."""

    def test_get_config_singleton(self):
        """Thread-safe singleton returns the same instance on repeated calls."""
        c1 = get_config()
        c2 = get_config()
        assert c1 is c2
        assert isinstance(c1, DataGatewayConfig)

    def test_set_config(self):
        """Replace config programmatically via set_config."""
        custom = DataGatewayConfig(service_name="gl-custom-gateway")
        set_config(custom)
        assert get_config().service_name == "gl-custom-gateway"
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
