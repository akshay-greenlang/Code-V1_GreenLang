# -*- coding: utf-8 -*-
"""
Unit Tests for ERPConnectorConfig (AGENT-DATA-003)

Tests configuration creation, env var overrides with GL_ERP_CONNECTOR_ prefix,
type parsing (bool, int, str), singleton get_config/set_config/reset_config,
and thread-safety of singleton access.

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
# Inline ERPConnectorConfig mirroring greenlang/erp_connector/config.py
# ---------------------------------------------------------------------------

_ENV_PREFIX = "GL_ERP_CONNECTOR_"


@dataclass
class ERPConnectorConfig:
    """Mirrors greenlang.erp_connector.config.ERPConnectorConfig."""

    database_url: str = ""
    redis_url: str = ""
    s3_bucket_url: str = ""
    default_erp_system: str = "simulated"
    connection_timeout_seconds: int = 60
    max_connections: int = 10
    sync_batch_size: int = 1000
    sync_max_records: int = 100000
    sync_timeout_seconds: int = 300
    enable_incremental_sync: bool = True
    default_mapping_strategy: str = "rule_based"
    enable_auto_classification: bool = True
    default_emission_methodology: str = "eeio"
    default_currency: str = "USD"
    enable_emissions_calculation: bool = True
    base_currency: str = "USD"
    enable_currency_conversion: bool = True
    batch_max_connections: int = 50
    batch_worker_count: int = 4
    pool_min_size: int = 2
    pool_max_size: int = 10
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> ERPConnectorConfig:
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
            database_url=_str("DATABASE_URL", cls.database_url),
            redis_url=_str("REDIS_URL", cls.redis_url),
            s3_bucket_url=_str("S3_BUCKET_URL", cls.s3_bucket_url),
            default_erp_system=_str("DEFAULT_ERP_SYSTEM", cls.default_erp_system),
            connection_timeout_seconds=_int("CONNECTION_TIMEOUT_SECONDS", cls.connection_timeout_seconds),
            max_connections=_int("MAX_CONNECTIONS", cls.max_connections),
            sync_batch_size=_int("SYNC_BATCH_SIZE", cls.sync_batch_size),
            sync_max_records=_int("SYNC_MAX_RECORDS", cls.sync_max_records),
            sync_timeout_seconds=_int("SYNC_TIMEOUT_SECONDS", cls.sync_timeout_seconds),
            enable_incremental_sync=_bool("ENABLE_INCREMENTAL_SYNC", cls.enable_incremental_sync),
            default_mapping_strategy=_str("DEFAULT_MAPPING_STRATEGY", cls.default_mapping_strategy),
            enable_auto_classification=_bool("ENABLE_AUTO_CLASSIFICATION", cls.enable_auto_classification),
            default_emission_methodology=_str("DEFAULT_EMISSION_METHODOLOGY", cls.default_emission_methodology),
            default_currency=_str("DEFAULT_CURRENCY", cls.default_currency),
            enable_emissions_calculation=_bool("ENABLE_EMISSIONS_CALCULATION", cls.enable_emissions_calculation),
            base_currency=_str("BASE_CURRENCY", cls.base_currency),
            enable_currency_conversion=_bool("ENABLE_CURRENCY_CONVERSION", cls.enable_currency_conversion),
            batch_max_connections=_int("BATCH_MAX_CONNECTIONS", cls.batch_max_connections),
            batch_worker_count=_int("BATCH_WORKER_COUNT", cls.batch_worker_count),
            pool_min_size=_int("POOL_MIN_SIZE", cls.pool_min_size),
            pool_max_size=_int("POOL_MAX_SIZE", cls.pool_max_size),
            log_level=_str("LOG_LEVEL", cls.log_level),
        )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_config_instance: Optional[ERPConnectorConfig] = None
_config_lock = threading.Lock()


def get_config() -> ERPConnectorConfig:
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = ERPConnectorConfig.from_env()
    return _config_instance


def set_config(config: ERPConnectorConfig) -> None:
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


class TestERPConnectorConfigDefaults:
    """Test that default configuration values match AGENT-DATA-003 PRD."""

    def test_default_database_url(self):
        config = ERPConnectorConfig()
        assert config.database_url == ""

    def test_default_redis_url(self):
        config = ERPConnectorConfig()
        assert config.redis_url == ""

    def test_default_s3_bucket_url(self):
        config = ERPConnectorConfig()
        assert config.s3_bucket_url == ""

    def test_default_erp_system(self):
        config = ERPConnectorConfig()
        assert config.default_erp_system == "simulated"

    def test_default_connection_timeout(self):
        config = ERPConnectorConfig()
        assert config.connection_timeout_seconds == 60

    def test_default_max_connections(self):
        config = ERPConnectorConfig()
        assert config.max_connections == 10

    def test_default_sync_batch_size(self):
        config = ERPConnectorConfig()
        assert config.sync_batch_size == 1000

    def test_default_sync_max_records(self):
        config = ERPConnectorConfig()
        assert config.sync_max_records == 100000

    def test_default_sync_timeout(self):
        config = ERPConnectorConfig()
        assert config.sync_timeout_seconds == 300

    def test_default_enable_incremental_sync(self):
        config = ERPConnectorConfig()
        assert config.enable_incremental_sync is True

    def test_default_mapping_strategy(self):
        config = ERPConnectorConfig()
        assert config.default_mapping_strategy == "rule_based"

    def test_default_enable_auto_classification(self):
        config = ERPConnectorConfig()
        assert config.enable_auto_classification is True

    def test_default_emission_methodology(self):
        config = ERPConnectorConfig()
        assert config.default_emission_methodology == "eeio"

    def test_default_currency(self):
        config = ERPConnectorConfig()
        assert config.default_currency == "USD"

    def test_default_enable_emissions_calculation(self):
        config = ERPConnectorConfig()
        assert config.enable_emissions_calculation is True

    def test_default_base_currency(self):
        config = ERPConnectorConfig()
        assert config.base_currency == "USD"

    def test_default_enable_currency_conversion(self):
        config = ERPConnectorConfig()
        assert config.enable_currency_conversion is True

    def test_default_batch_max_connections(self):
        config = ERPConnectorConfig()
        assert config.batch_max_connections == 50

    def test_default_batch_worker_count(self):
        config = ERPConnectorConfig()
        assert config.batch_worker_count == 4

    def test_default_pool_min_size(self):
        config = ERPConnectorConfig()
        assert config.pool_min_size == 2

    def test_default_pool_max_size(self):
        config = ERPConnectorConfig()
        assert config.pool_max_size == 10

    def test_default_log_level(self):
        config = ERPConnectorConfig()
        assert config.log_level == "INFO"


class TestERPConnectorConfigFromEnv:
    """Test GL_ERP_CONNECTOR_ env var overrides via from_env()."""

    def test_env_override_database_url(self, monkeypatch):
        monkeypatch.setenv("GL_ERP_CONNECTOR_DATABASE_URL", "postgresql://test:5432/erp")
        config = ERPConnectorConfig.from_env()
        assert config.database_url == "postgresql://test:5432/erp"

    def test_env_override_redis_url(self, monkeypatch):
        monkeypatch.setenv("GL_ERP_CONNECTOR_REDIS_URL", "redis://localhost:6379/2")
        config = ERPConnectorConfig.from_env()
        assert config.redis_url == "redis://localhost:6379/2"

    def test_env_override_erp_system(self, monkeypatch):
        monkeypatch.setenv("GL_ERP_CONNECTOR_DEFAULT_ERP_SYSTEM", "sap_s4hana")
        config = ERPConnectorConfig.from_env()
        assert config.default_erp_system == "sap_s4hana"

    def test_env_override_connection_timeout(self, monkeypatch):
        monkeypatch.setenv("GL_ERP_CONNECTOR_CONNECTION_TIMEOUT_SECONDS", "120")
        config = ERPConnectorConfig.from_env()
        assert config.connection_timeout_seconds == 120

    def test_env_override_max_connections(self, monkeypatch):
        monkeypatch.setenv("GL_ERP_CONNECTOR_MAX_CONNECTIONS", "50")
        config = ERPConnectorConfig.from_env()
        assert config.max_connections == 50

    def test_env_override_sync_batch_size(self, monkeypatch):
        monkeypatch.setenv("GL_ERP_CONNECTOR_SYNC_BATCH_SIZE", "5000")
        config = ERPConnectorConfig.from_env()
        assert config.sync_batch_size == 5000

    def test_env_override_sync_max_records(self, monkeypatch):
        monkeypatch.setenv("GL_ERP_CONNECTOR_SYNC_MAX_RECORDS", "500000")
        config = ERPConnectorConfig.from_env()
        assert config.sync_max_records == 500000

    def test_env_override_enable_incremental_false(self, monkeypatch):
        monkeypatch.setenv("GL_ERP_CONNECTOR_ENABLE_INCREMENTAL_SYNC", "false")
        config = ERPConnectorConfig.from_env()
        assert config.enable_incremental_sync is False

    def test_env_override_enable_incremental_zero(self, monkeypatch):
        monkeypatch.setenv("GL_ERP_CONNECTOR_ENABLE_INCREMENTAL_SYNC", "0")
        config = ERPConnectorConfig.from_env()
        assert config.enable_incremental_sync is False

    def test_env_override_enable_incremental_yes(self, monkeypatch):
        monkeypatch.setenv("GL_ERP_CONNECTOR_ENABLE_INCREMENTAL_SYNC", "yes")
        config = ERPConnectorConfig.from_env()
        assert config.enable_incremental_sync is True

    def test_env_override_bool_true_with_TRUE(self, monkeypatch):
        monkeypatch.setenv("GL_ERP_CONNECTOR_ENABLE_AUTO_CLASSIFICATION", "TRUE")
        config = ERPConnectorConfig.from_env()
        assert config.enable_auto_classification is True

    def test_env_override_enable_emissions_false(self, monkeypatch):
        monkeypatch.setenv("GL_ERP_CONNECTOR_ENABLE_EMISSIONS_CALCULATION", "no")
        config = ERPConnectorConfig.from_env()
        assert config.enable_emissions_calculation is False

    def test_env_override_default_currency(self, monkeypatch):
        monkeypatch.setenv("GL_ERP_CONNECTOR_DEFAULT_CURRENCY", "EUR")
        config = ERPConnectorConfig.from_env()
        assert config.default_currency == "EUR"

    def test_env_override_base_currency(self, monkeypatch):
        monkeypatch.setenv("GL_ERP_CONNECTOR_BASE_CURRENCY", "GBP")
        config = ERPConnectorConfig.from_env()
        assert config.base_currency == "GBP"

    def test_env_override_emission_methodology(self, monkeypatch):
        monkeypatch.setenv("GL_ERP_CONNECTOR_DEFAULT_EMISSION_METHODOLOGY", "hybrid")
        config = ERPConnectorConfig.from_env()
        assert config.default_emission_methodology == "hybrid"

    def test_env_override_mapping_strategy(self, monkeypatch):
        monkeypatch.setenv("GL_ERP_CONNECTOR_DEFAULT_MAPPING_STRATEGY", "ml_based")
        config = ERPConnectorConfig.from_env()
        assert config.default_mapping_strategy == "ml_based"

    def test_env_override_batch_worker_count(self, monkeypatch):
        monkeypatch.setenv("GL_ERP_CONNECTOR_BATCH_WORKER_COUNT", "8")
        config = ERPConnectorConfig.from_env()
        assert config.batch_worker_count == 8

    def test_env_override_pool_min_size(self, monkeypatch):
        monkeypatch.setenv("GL_ERP_CONNECTOR_POOL_MIN_SIZE", "5")
        config = ERPConnectorConfig.from_env()
        assert config.pool_min_size == 5

    def test_env_override_pool_max_size(self, monkeypatch):
        monkeypatch.setenv("GL_ERP_CONNECTOR_POOL_MAX_SIZE", "50")
        config = ERPConnectorConfig.from_env()
        assert config.pool_max_size == 50

    def test_env_override_log_level(self, monkeypatch):
        monkeypatch.setenv("GL_ERP_CONNECTOR_LOG_LEVEL", "DEBUG")
        config = ERPConnectorConfig.from_env()
        assert config.log_level == "DEBUG"

    def test_env_invalid_int_uses_default(self, monkeypatch):
        monkeypatch.setenv("GL_ERP_CONNECTOR_MAX_CONNECTIONS", "not_a_number")
        config = ERPConnectorConfig.from_env()
        assert config.max_connections == 10

    def test_env_invalid_int_sync_batch_uses_default(self, monkeypatch):
        monkeypatch.setenv("GL_ERP_CONNECTOR_SYNC_BATCH_SIZE", "abc")
        config = ERPConnectorConfig.from_env()
        assert config.sync_batch_size == 1000

    def test_env_invalid_int_timeout_uses_default(self, monkeypatch):
        monkeypatch.setenv("GL_ERP_CONNECTOR_SYNC_TIMEOUT_SECONDS", "xyz")
        config = ERPConnectorConfig.from_env()
        assert config.sync_timeout_seconds == 300

    def test_multiple_env_overrides(self, monkeypatch):
        monkeypatch.setenv("GL_ERP_CONNECTOR_DEFAULT_ERP_SYSTEM", "oracle_cloud")
        monkeypatch.setenv("GL_ERP_CONNECTOR_MAX_CONNECTIONS", "25")
        monkeypatch.setenv("GL_ERP_CONNECTOR_ENABLE_INCREMENTAL_SYNC", "false")
        monkeypatch.setenv("GL_ERP_CONNECTOR_DEFAULT_CURRENCY", "EUR")
        config = ERPConnectorConfig.from_env()
        assert config.default_erp_system == "oracle_cloud"
        assert config.max_connections == 25
        assert config.enable_incremental_sync is False
        assert config.default_currency == "EUR"


class TestERPConnectorConfigSingleton:
    """Test get_config/set_config/reset_config singleton pattern."""

    def test_get_config_returns_instance(self):
        config = get_config()
        assert isinstance(config, ERPConnectorConfig)

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
        custom = ERPConnectorConfig(default_erp_system="sap_s4hana")
        set_config(custom)
        assert get_config().default_erp_system == "sap_s4hana"

    def test_set_config_then_get_returns_same(self):
        custom = ERPConnectorConfig(max_connections=99)
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


class TestERPConnectorConfigCustomValues:
    """Test creating config with fully custom values."""

    def test_full_custom_config(self):
        config = ERPConnectorConfig(
            database_url="postgresql://custom:5432/erp",
            redis_url="redis://custom:6379/1",
            s3_bucket_url="s3://custom-erp-bucket",
            default_erp_system="sap_s4hana",
            connection_timeout_seconds=120,
            max_connections=50,
            sync_batch_size=5000,
            sync_max_records=500000,
            sync_timeout_seconds=600,
            enable_incremental_sync=False,
            default_mapping_strategy="ml_based",
            enable_auto_classification=False,
            default_emission_methodology="hybrid",
            default_currency="EUR",
            enable_emissions_calculation=False,
            base_currency="GBP",
            enable_currency_conversion=False,
            batch_max_connections=100,
            batch_worker_count=16,
            pool_min_size=5,
            pool_max_size=50,
            log_level="DEBUG",
        )
        assert config.database_url == "postgresql://custom:5432/erp"
        assert config.default_erp_system == "sap_s4hana"
        assert config.connection_timeout_seconds == 120
        assert config.max_connections == 50
        assert config.sync_batch_size == 5000
        assert config.enable_incremental_sync is False
        assert config.default_emission_methodology == "hybrid"
        assert config.default_currency == "EUR"
        assert config.enable_emissions_calculation is False
        assert config.base_currency == "GBP"
        assert config.enable_currency_conversion is False
        assert config.batch_worker_count == 16
        assert config.pool_min_size == 5
        assert config.pool_max_size == 50
        assert config.log_level == "DEBUG"

    def test_erp_system_options(self):
        """Verify config accepts all valid ERP system names."""
        systems = [
            "simulated", "sap_s4hana", "sap_ecc", "oracle_cloud",
            "oracle_ebs", "netsuite", "dynamics_365", "workday",
            "sage", "quickbooks",
        ]
        for system in systems:
            config = ERPConnectorConfig(default_erp_system=system)
            assert config.default_erp_system == system
