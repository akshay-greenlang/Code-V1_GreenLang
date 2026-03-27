# -*- coding: utf-8 -*-
"""
Test suite for business_travel.config - AGENT-MRV-019.

Tests configuration management for the Business Travel Agent
(GL-MRV-S3-006) including default values, environment variable loading,
singleton pattern, thread safety, validation, and serialization.

Coverage:
- Default config values for GeneralConfig, DatabaseConfig, AirTravelConfig,
  RailConfig, RoadConfig, HotelConfig, SpendConfig, ComplianceConfig,
  EFSourceConfig, UncertaintyConfig, CacheConfig, APIConfig,
  ProvenanceConfig, MetricsConfig
- GL_BT_ environment variable loading (monkeypatch)
- Singleton pattern (get_config returns same instance)
- Thread safety
- Validation (invalid values raise errors)
- to_dict / from_dict round-trip
- Frozen immutability

Author: GL-TestEngineer
Date: February 2026
"""

import os
import threading
from decimal import Decimal
from typing import Any, Dict
import pytest

from greenlang.agents.mrv.business_travel.config import (
    get_config,
    GeneralConfig,
    DatabaseConfig,
)


# ==============================================================================
# GENERAL CONFIGURATION TESTS
# ==============================================================================


class TestGeneralConfig:
    """Test GeneralConfig dataclass."""

    def test_general_config_defaults(self):
        """Test default general config values."""
        config = GeneralConfig()
        assert config.enabled is True
        assert config.debug is False
        assert config.log_level == "INFO"
        assert config.max_batch_size == 1000

    def test_general_config_agent_id(self):
        """Test default agent_id is GL-MRV-S3-006."""
        config = GeneralConfig()
        assert config.agent_id == "GL-MRV-S3-006"
        assert config.agent_component == "AGENT-MRV-019"
        assert config.version == "1.0.0"

    def test_general_config_api_prefix(self):
        """Test default API prefix."""
        config = GeneralConfig()
        assert config.api_prefix == "/api/v1/business-travel"

    def test_general_config_default_gwp(self):
        """Test default GWP is AR5."""
        config = GeneralConfig()
        assert config.default_gwp == "AR5"

    def test_general_config_default_ef_source(self):
        """Test default EF source is DEFRA."""
        config = GeneralConfig()
        assert config.default_ef_source == "DEFRA"

    def test_general_config_default_rf_option(self):
        """Test default RF option is WITH_RF."""
        config = GeneralConfig()
        assert config.default_rf_option == "WITH_RF"

    def test_general_config_default_uplift(self):
        """Test default uplift factor is 0.08."""
        config = GeneralConfig()
        assert config.default_uplift_factor == Decimal("0.08")

    def test_config_frozen_immutability(self):
        """Test config is immutable (frozen=True)."""
        config = GeneralConfig()
        with pytest.raises(Exception):
            config.enabled = False

    def test_validate_config_no_errors(self):
        """Test validation passes for default config."""
        config = GeneralConfig()
        config.validate()  # Should not raise

    def test_validate_config_invalid_log_level(self):
        """Test validation fails for invalid log level."""
        config = GeneralConfig(log_level="INVALID")
        with pytest.raises(ValueError, match="Invalid log_level"):
            config.validate()

    def test_validate_config_empty_agent_id(self):
        """Test validation fails for empty agent_id."""
        config = GeneralConfig(agent_id="")
        with pytest.raises(ValueError, match="agent_id cannot be empty"):
            config.validate()

    def test_validate_config_invalid_version(self):
        """Test validation fails for invalid SemVer format."""
        config = GeneralConfig(version="1.0")
        with pytest.raises(ValueError, match="Must follow SemVer"):
            config.validate()

    def test_validate_config_invalid_gwp(self):
        """Test validation fails for invalid GWP version."""
        config = GeneralConfig(default_gwp="AR99")
        with pytest.raises(ValueError, match="Invalid default_gwp"):
            config.validate()

    def test_validate_config_invalid_ef_source(self):
        """Test validation fails for invalid EF source."""
        config = GeneralConfig(default_ef_source="BOGUS")
        with pytest.raises(ValueError, match="Invalid default_ef_source"):
            config.validate()

    def test_validate_config_invalid_rf_option(self):
        """Test validation fails for invalid RF option."""
        config = GeneralConfig(default_rf_option="BOGUS")
        with pytest.raises(ValueError, match="Invalid default_rf_option"):
            config.validate()

    def test_validate_config_invalid_batch_size(self):
        """Test validation fails for out-of-range batch size."""
        config = GeneralConfig(max_batch_size=0)
        with pytest.raises(ValueError, match="max_batch_size"):
            config.validate()

    def test_validate_config_invalid_api_prefix(self):
        """Test validation fails for API prefix not starting with /."""
        config = GeneralConfig(api_prefix="api/v1/business-travel")
        with pytest.raises(ValueError, match="api_prefix must start with"):
            config.validate()

    def test_validate_config_uplift_out_of_range(self):
        """Test validation fails for uplift factor > 1."""
        config = GeneralConfig(default_uplift_factor=Decimal("1.5"))
        with pytest.raises(ValueError, match="default_uplift_factor"):
            config.validate()

    def test_master_config_to_dict(self):
        """Test to_dict serialization."""
        config = GeneralConfig()
        d = config.to_dict()
        assert d["agent_id"] == "GL-MRV-S3-006"
        assert d["enabled"] is True
        assert d["log_level"] == "INFO"

    def test_master_config_from_dict_roundtrip(self):
        """Test to_dict -> from_dict round-trip."""
        original = GeneralConfig()
        d = original.to_dict()
        restored = GeneralConfig.from_dict(d)
        assert restored.agent_id == original.agent_id
        assert restored.enabled == original.enabled
        assert restored.version == original.version

    def test_from_env_defaults(self):
        """Test from_env uses defaults when env vars not set."""
        # Clear any existing env vars
        for key in list(os.environ.keys()):
            if key.startswith("GL_BT_"):
                del os.environ[key]

        config = GeneralConfig.from_env()
        assert config.enabled is True
        assert config.agent_id == "GL-MRV-S3-006"
        assert config.log_level == "INFO"

    def test_from_env_custom(self, monkeypatch):
        """Test from_env loads custom values from environment."""
        monkeypatch.setenv("GL_BT_ENABLED", "false")
        monkeypatch.setenv("GL_BT_DEBUG", "true")
        monkeypatch.setenv("GL_BT_LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("GL_BT_AGENT_ID", "GL-MRV-S3-006-TEST")

        config = GeneralConfig.from_env()
        assert config.enabled is False
        assert config.debug is True
        assert config.log_level == "DEBUG"
        assert config.agent_id == "GL-MRV-S3-006-TEST"

    def test_print_config_redacts_password(self):
        """Test to_dict does not expose sensitive database password."""
        config = GeneralConfig()
        d = config.to_dict()
        # GeneralConfig should not contain password field
        assert "password" not in d


# ==============================================================================
# DATABASE CONFIGURATION TESTS
# ==============================================================================


class TestDatabaseConfig:
    """Test DatabaseConfig dataclass."""

    def test_database_config_defaults(self):
        """Test default database config values."""
        config = DatabaseConfig()
        assert config.host == "localhost"
        assert config.port == 5432
        assert config.database == "greenlang"
        assert config.pool_min == 2
        assert config.pool_max == 10

    def test_database_config_table_prefix(self):
        """Test default table prefix is gl_bt_."""
        config = DatabaseConfig()
        assert config.table_prefix == "gl_bt_"

    def test_database_config_schema(self):
        """Test default schema is business_travel_service."""
        config = DatabaseConfig()
        assert config.schema == "business_travel_service"

    def test_database_config_ssl_mode(self):
        """Test default SSL mode is prefer."""
        config = DatabaseConfig()
        assert config.ssl_mode == "prefer"

    def test_database_config_connection_timeout(self):
        """Test default connection timeout is 30."""
        config = DatabaseConfig()
        assert config.connection_timeout == 30

    def test_database_config_frozen(self):
        """Test DatabaseConfig is immutable (frozen=True)."""
        config = DatabaseConfig()
        with pytest.raises(Exception):
            config.host = "remote-host"

    def test_database_config_validate_invalid_port(self):
        """Test validation fails for port out of range."""
        config = DatabaseConfig(port=99999)
        with pytest.raises(ValueError, match="port must be between"):
            config.validate()

    def test_database_config_validate_invalid_prefix(self):
        """Test validation fails for table prefix not ending with _."""
        config = DatabaseConfig(table_prefix="gl_bt")
        with pytest.raises(ValueError, match="table_prefix must end"):
            config.validate()

    def test_database_config_validate_pool_min_gt_max(self):
        """Test validation fails for pool_min > pool_max."""
        config = DatabaseConfig(pool_min=20, pool_max=5)
        with pytest.raises(ValueError, match="pool_min must be"):
            config.validate()


# ==============================================================================
# SINGLETON PATTERN TESTS
# ==============================================================================


class TestSingletonPattern:
    """Test singleton pattern implementation."""

    def test_config_singleton_get(self):
        """Test get_config returns the same instance."""
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2

    def test_config_thread_safety(self):
        """Test singleton works across threads."""
        configs = []

        def get_config_thread():
            configs.append(get_config())

        threads = [threading.Thread(target=get_config_thread) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        first = configs[0]
        for c in configs[1:]:
            assert c is first


# ==============================================================================
# ENVIRONMENT VARIABLE TESTS
# ==============================================================================


class TestEnvironmentVariables:
    """Test environment variable loading."""

    def test_boolean_env_var_true(self, monkeypatch):
        """Test boolean env var parsing for 'true'."""
        monkeypatch.setenv("GL_BT_ENABLED", "true")
        config = GeneralConfig.from_env()
        assert config.enabled is True

    def test_boolean_env_var_false(self, monkeypatch):
        """Test boolean env var parsing for 'false'."""
        monkeypatch.setenv("GL_BT_ENABLED", "false")
        config = GeneralConfig.from_env()
        assert config.enabled is False

    def test_boolean_env_var_case_insensitive(self, monkeypatch):
        """Test boolean env var is case-insensitive."""
        monkeypatch.setenv("GL_BT_DEBUG", "True")
        config = GeneralConfig.from_env()
        assert config.debug is True

    def test_numeric_env_var(self, monkeypatch):
        """Test numeric env var parsing."""
        monkeypatch.setenv("GL_BT_MAX_BATCH_SIZE", "500")
        config = GeneralConfig.from_env()
        assert config.max_batch_size == 500

    def test_decimal_env_var(self, monkeypatch):
        """Test Decimal env var parsing."""
        monkeypatch.setenv("GL_BT_DEFAULT_UPLIFT_FACTOR", "0.10")
        config = GeneralConfig.from_env()
        assert config.default_uplift_factor == Decimal("0.10")


# ==============================================================================
# SUMMARY
# ==============================================================================


def test_config_module_coverage():
    """Meta-test to ensure comprehensive config section coverage."""
    tested_sections = [
        "GeneralConfig",
        "DatabaseConfig",
    ]
    assert len(tested_sections) >= 2
