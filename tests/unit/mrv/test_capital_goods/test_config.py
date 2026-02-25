# -*- coding: utf-8 -*-
"""Unit tests for Capital Goods Agent configuration."""

from __future__ import annotations

import os
from decimal import Decimal
from unittest.mock import patch

import pytest

from greenlang.capital_goods.config import CapitalGoodsConfig


# ============================================================================
# SINGLETON PATTERN TESTS
# ============================================================================


class TestConfigSingleton:
    """Test configuration singleton pattern."""

    def test_singleton_same_instance(self):
        """Test that multiple calls return the same instance."""
        config1 = CapitalGoodsConfig()
        config2 = CapitalGoodsConfig()
        assert config1 is config2

    def test_reset_creates_new_instance(self):
        """Test that reset() creates a new instance."""
        config1 = CapitalGoodsConfig()
        old_id = id(config1)
        CapitalGoodsConfig.reset()
        config2 = CapitalGoodsConfig()
        assert id(config2) != old_id

    def test_singleton_thread_safe(self):
        """Test singleton is thread-safe."""
        import threading

        instances = []

        def get_instance():
            instances.append(CapitalGoodsConfig())

        threads = [threading.Thread(target=get_instance) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All instances should be the same object
        assert all(inst is instances[0] for inst in instances)


# ============================================================================
# DEFAULT VALUES TESTS
# ============================================================================


class TestConfigDefaults:
    """Test configuration default values."""

    @pytest.fixture(autouse=True)
    def reset_config(self):
        """Reset config before each test."""
        CapitalGoodsConfig.reset()
        yield
        CapitalGoodsConfig.reset()

    def test_database_defaults(self):
        """Test database configuration defaults."""
        config = CapitalGoodsConfig()
        assert config.database["host"] == "localhost"
        assert config.database["port"] == 5432
        assert config.database["database"] == "greenlang"
        assert config.database["user"] == "greenlang_user"
        assert config.database["pool_size"] == 20
        assert config.database["max_overflow"] == 10
        assert config.database["pool_timeout"] == 30
        assert config.database["pool_recycle"] == 3600

    def test_calculation_defaults(self):
        """Test calculation configuration defaults."""
        config = CapitalGoodsConfig()
        assert config.calculation["default_gwp_version"] == "AR5"
        assert config.calculation["capitalization_threshold_usd"] == Decimal("5000.00")
        assert config.calculation["apply_uncertainty_by_default"] is True
        assert config.calculation["rolling_average_years"] == 3
        assert config.calculation["volatility_ratio_threshold"] == Decimal("0.25")
        assert config.calculation["include_biogenic_by_default"] is False

    def test_data_quality_defaults(self):
        """Test data quality configuration defaults."""
        config = CapitalGoodsConfig()
        assert config.data_quality["min_acceptable_score"] == Decimal("2.0")
        assert config.data_quality["high_quality_threshold"] == Decimal("4.0")
        assert config.data_quality["require_verification_for_supplier_data"] is False
        assert config.data_quality["prefer_supplier_specific_over_average"] is True
        assert config.data_quality["apply_uncertainty_discount_factor"] is True

    def test_performance_defaults(self):
        """Test performance configuration defaults."""
        config = CapitalGoodsConfig()
        assert config.performance["batch_size"] == 100
        assert config.performance["max_workers"] == 4
        assert config.performance["cache_ttl_seconds"] == 3600
        assert config.performance["query_timeout_seconds"] == 30
        assert config.performance["enable_query_cache"] is True
        assert config.performance["enable_result_cache"] is True

    def test_logging_defaults(self):
        """Test logging configuration defaults."""
        config = CapitalGoodsConfig()
        assert config.logging["level"] == "INFO"
        assert config.logging["format"] == "json"
        assert config.logging["enable_metrics"] is True
        assert config.logging["enable_provenance"] is True

    def test_security_defaults(self):
        """Test security configuration defaults."""
        config = CapitalGoodsConfig()
        assert config.security["require_authentication"] is True
        assert config.security["enable_rate_limiting"] is True
        assert config.security["rate_limit_requests_per_minute"] == 60
        assert config.security["enable_audit_logging"] is True

    def test_integration_defaults(self):
        """Test integration configuration defaults."""
        config = CapitalGoodsConfig()
        assert config.integration["enable_erp_connector"] is False
        assert config.integration["enable_supplier_portal"] is False
        assert config.integration["enable_external_factor_db"] is True

    def test_regulatory_defaults(self):
        """Test regulatory configuration defaults."""
        config = CapitalGoodsConfig()
        assert config.regulatory["default_framework"] == "ghg_protocol_scope3"
        assert config.regulatory["require_compliance_check"] is True
        assert config.regulatory["enable_multi_framework_reporting"] is True

    def test_agent_defaults(self):
        """Test agent configuration defaults."""
        config = CapitalGoodsConfig()
        assert config.agent["agent_id"] == "AGENT-MRV-015"
        assert config.agent["agent_name"] == "Capital Goods Agent"
        assert config.agent["version"] == "1.0.0"
        assert config.agent["scope3_category"] == "3.2"


# ============================================================================
# ENVIRONMENT VARIABLE TESTS
# ============================================================================


class TestConfigFromEnvironment:
    """Test configuration from environment variables."""

    @pytest.fixture(autouse=True)
    def reset_config(self):
        """Reset config before each test."""
        CapitalGoodsConfig.reset()
        yield
        CapitalGoodsConfig.reset()

    def test_from_env_database_host(self):
        """Test loading database host from environment."""
        with patch.dict(os.environ, {"CG_DB_HOST": "db.example.com"}):
            config = CapitalGoodsConfig.from_env()
            assert config.database["host"] == "db.example.com"

    def test_from_env_database_port(self):
        """Test loading database port from environment."""
        with patch.dict(os.environ, {"CG_DB_PORT": "5433"}):
            config = CapitalGoodsConfig.from_env()
            assert config.database["port"] == 5433

    def test_from_env_database_name(self):
        """Test loading database name from environment."""
        with patch.dict(os.environ, {"CG_DB_NAME": "greenlang_prod"}):
            config = CapitalGoodsConfig.from_env()
            assert config.database["database"] == "greenlang_prod"

    def test_from_env_database_user(self):
        """Test loading database user from environment."""
        with patch.dict(os.environ, {"CG_DB_USER": "prod_user"}):
            config = CapitalGoodsConfig.from_env()
            assert config.database["user"] == "prod_user"

    def test_from_env_database_password(self):
        """Test loading database password from environment."""
        with patch.dict(os.environ, {"CG_DB_PASSWORD": "secret123"}):
            config = CapitalGoodsConfig.from_env()
            assert config.database["password"] == "secret123"

    def test_from_env_capitalization_threshold(self):
        """Test loading capitalization threshold from environment."""
        with patch.dict(os.environ, {"CG_CAPITALIZATION_THRESHOLD_USD": "10000.00"}):
            config = CapitalGoodsConfig.from_env()
            assert config.calculation["capitalization_threshold_usd"] == Decimal("10000.00")

    def test_from_env_gwp_version(self):
        """Test loading GWP version from environment."""
        with patch.dict(os.environ, {"CG_DEFAULT_GWP_VERSION": "AR6"}):
            config = CapitalGoodsConfig.from_env()
            assert config.calculation["default_gwp_version"] == "AR6"

    def test_from_env_rolling_average_years(self):
        """Test loading rolling average years from environment."""
        with patch.dict(os.environ, {"CG_ROLLING_AVERAGE_YEARS": "5"}):
            config = CapitalGoodsConfig.from_env()
            assert config.calculation["rolling_average_years"] == 5

    def test_from_env_volatility_ratio_threshold(self):
        """Test loading volatility ratio threshold from environment."""
        with patch.dict(os.environ, {"CG_VOLATILITY_RATIO_THRESHOLD": "0.30"}):
            config = CapitalGoodsConfig.from_env()
            assert config.calculation["volatility_ratio_threshold"] == Decimal("0.30")

    def test_from_env_batch_size(self):
        """Test loading batch size from environment."""
        with patch.dict(os.environ, {"CG_BATCH_SIZE": "200"}):
            config = CapitalGoodsConfig.from_env()
            assert config.performance["batch_size"] == 200

    def test_from_env_max_workers(self):
        """Test loading max workers from environment."""
        with patch.dict(os.environ, {"CG_MAX_WORKERS": "8"}):
            config = CapitalGoodsConfig.from_env()
            assert config.performance["max_workers"] == 8

    def test_from_env_cache_ttl(self):
        """Test loading cache TTL from environment."""
        with patch.dict(os.environ, {"CG_CACHE_TTL_SECONDS": "7200"}):
            config = CapitalGoodsConfig.from_env()
            assert config.performance["cache_ttl_seconds"] == 7200

    def test_from_env_log_level(self):
        """Test loading log level from environment."""
        with patch.dict(os.environ, {"CG_LOG_LEVEL": "DEBUG"}):
            config = CapitalGoodsConfig.from_env()
            assert config.logging["level"] == "DEBUG"


# ============================================================================
# VALIDATION TESTS
# ============================================================================


class TestConfigValidation:
    """Test configuration validation."""

    @pytest.fixture(autouse=True)
    def reset_config(self):
        """Reset config before each test."""
        CapitalGoodsConfig.reset()
        yield
        CapitalGoodsConfig.reset()

    def test_validate_defaults_passes(self):
        """Test that default configuration passes validation."""
        config = CapitalGoodsConfig()
        config.validate()  # Should not raise

    def test_validate_invalid_port_fails(self):
        """Test that invalid port fails validation."""
        config = CapitalGoodsConfig()
        config.database["port"] = -1
        with pytest.raises(ValueError, match="port must be between"):
            config.validate()

    def test_validate_invalid_pool_size_fails(self):
        """Test that invalid pool size fails validation."""
        config = CapitalGoodsConfig()
        config.database["pool_size"] = 0
        with pytest.raises(ValueError, match="pool_size must be positive"):
            config.validate()

    def test_validate_invalid_gwp_version_fails(self):
        """Test that invalid GWP version fails validation."""
        config = CapitalGoodsConfig()
        config.calculation["default_gwp_version"] = "INVALID"
        with pytest.raises(ValueError, match="gwp_version must be"):
            config.validate()

    def test_validate_negative_capitalization_threshold_fails(self):
        """Test that negative capitalization threshold fails validation."""
        config = CapitalGoodsConfig()
        config.calculation["capitalization_threshold_usd"] = Decimal("-1000.00")
        with pytest.raises(ValueError, match="capitalization_threshold_usd must be non-negative"):
            config.validate()

    def test_validate_zero_rolling_average_years_fails(self):
        """Test that zero rolling average years fails validation."""
        config = CapitalGoodsConfig()
        config.calculation["rolling_average_years"] = 0
        with pytest.raises(ValueError, match="rolling_average_years must be positive"):
            config.validate()

    def test_validate_invalid_volatility_ratio_fails(self):
        """Test that invalid volatility ratio fails validation."""
        config = CapitalGoodsConfig()
        config.calculation["volatility_ratio_threshold"] = Decimal("1.5")
        with pytest.raises(ValueError, match="volatility_ratio_threshold must be between 0 and 1"):
            config.validate()

    def test_validate_invalid_data_quality_score_fails(self):
        """Test that invalid data quality score fails validation."""
        config = CapitalGoodsConfig()
        config.data_quality["min_acceptable_score"] = Decimal("6.0")
        with pytest.raises(ValueError, match="min_acceptable_score must be between 1 and 5"):
            config.validate()

    def test_validate_invalid_batch_size_fails(self):
        """Test that invalid batch size fails validation."""
        config = CapitalGoodsConfig()
        config.performance["batch_size"] = -10
        with pytest.raises(ValueError, match="batch_size must be positive"):
            config.validate()

    def test_validate_invalid_max_workers_fails(self):
        """Test that invalid max workers fails validation."""
        config = CapitalGoodsConfig()
        config.performance["max_workers"] = 0
        with pytest.raises(ValueError, match="max_workers must be positive"):
            config.validate()

    def test_validate_invalid_log_level_fails(self):
        """Test that invalid log level fails validation."""
        config = CapitalGoodsConfig()
        config.logging["level"] = "INVALID"
        with pytest.raises(ValueError, match="log level must be"):
            config.validate()


# ============================================================================
# SERIALIZATION TESTS
# ============================================================================


class TestConfigSerialization:
    """Test configuration serialization."""

    @pytest.fixture(autouse=True)
    def reset_config(self):
        """Reset config before each test."""
        CapitalGoodsConfig.reset()
        yield
        CapitalGoodsConfig.reset()

    def test_to_dict(self):
        """Test converting configuration to dictionary."""
        config = CapitalGoodsConfig()
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert "database" in config_dict
        assert "calculation" in config_dict
        assert "data_quality" in config_dict
        assert "performance" in config_dict

    def test_to_dict_contains_all_sections(self):
        """Test that to_dict contains all configuration sections."""
        config = CapitalGoodsConfig()
        config_dict = config.to_dict()
        expected_sections = [
            "database",
            "calculation",
            "data_quality",
            "performance",
            "logging",
            "security",
            "integration",
            "regulatory",
            "agent",
        ]
        for section in expected_sections:
            assert section in config_dict

    def test_to_dict_database_section(self):
        """Test database section in to_dict output."""
        config = CapitalGoodsConfig()
        config_dict = config.to_dict()
        assert config_dict["database"]["host"] == "localhost"
        assert config_dict["database"]["port"] == 5432

    def test_to_dict_calculation_section(self):
        """Test calculation section in to_dict output."""
        config = CapitalGoodsConfig()
        config_dict = config.to_dict()
        assert config_dict["calculation"]["default_gwp_version"] == "AR5"
        assert config_dict["calculation"]["capitalization_threshold_usd"] == Decimal("5000.00")

    def test_to_dict_preserves_types(self):
        """Test that to_dict preserves data types."""
        config = CapitalGoodsConfig()
        config_dict = config.to_dict()
        assert isinstance(config_dict["database"]["port"], int)
        assert isinstance(config_dict["calculation"]["capitalization_threshold_usd"], Decimal)
        assert isinstance(config_dict["calculation"]["apply_uncertainty_by_default"], bool)


# ============================================================================
# CONFIGURATION UPDATE TESTS
# ============================================================================


class TestConfigUpdate:
    """Test configuration updates."""

    @pytest.fixture(autouse=True)
    def reset_config(self):
        """Reset config before each test."""
        CapitalGoodsConfig.reset()
        yield
        CapitalGoodsConfig.reset()

    def test_update_database_host(self):
        """Test updating database host."""
        config = CapitalGoodsConfig()
        config.database["host"] = "newhost.example.com"
        assert config.database["host"] == "newhost.example.com"

    def test_update_capitalization_threshold(self):
        """Test updating capitalization threshold."""
        config = CapitalGoodsConfig()
        config.calculation["capitalization_threshold_usd"] = Decimal("10000.00")
        assert config.calculation["capitalization_threshold_usd"] == Decimal("10000.00")

    def test_update_rolling_average_years(self):
        """Test updating rolling average years."""
        config = CapitalGoodsConfig()
        config.calculation["rolling_average_years"] = 5
        assert config.calculation["rolling_average_years"] == 5

    def test_update_volatility_ratio_threshold(self):
        """Test updating volatility ratio threshold."""
        config = CapitalGoodsConfig()
        config.calculation["volatility_ratio_threshold"] = Decimal("0.30")
        assert config.calculation["volatility_ratio_threshold"] == Decimal("0.30")

    def test_update_multiple_values(self):
        """Test updating multiple configuration values."""
        config = CapitalGoodsConfig()
        config.database["host"] = "prod.example.com"
        config.database["port"] = 5433
        config.calculation["default_gwp_version"] = "AR6"
        config.performance["batch_size"] = 200

        assert config.database["host"] == "prod.example.com"
        assert config.database["port"] == 5433
        assert config.calculation["default_gwp_version"] == "AR6"
        assert config.performance["batch_size"] == 200


# ============================================================================
# EDGE CASE TESTS
# ============================================================================


class TestConfigEdgeCases:
    """Test configuration edge cases."""

    @pytest.fixture(autouse=True)
    def reset_config(self):
        """Reset config before each test."""
        CapitalGoodsConfig.reset()
        yield
        CapitalGoodsConfig.reset()

    def test_minimum_capitalization_threshold(self):
        """Test minimum capitalization threshold (zero)."""
        config = CapitalGoodsConfig()
        config.calculation["capitalization_threshold_usd"] = Decimal("0.00")
        config.validate()  # Should pass

    def test_maximum_rolling_average_years(self):
        """Test maximum rolling average years."""
        config = CapitalGoodsConfig()
        config.calculation["rolling_average_years"] = 10
        config.validate()  # Should pass

    def test_minimum_volatility_ratio_threshold(self):
        """Test minimum volatility ratio threshold (zero)."""
        config = CapitalGoodsConfig()
        config.calculation["volatility_ratio_threshold"] = Decimal("0.00")
        config.validate()  # Should pass

    def test_maximum_volatility_ratio_threshold(self):
        """Test maximum volatility ratio threshold (one)."""
        config = CapitalGoodsConfig()
        config.calculation["volatility_ratio_threshold"] = Decimal("1.00")
        config.validate()  # Should pass

    def test_very_large_batch_size(self):
        """Test very large batch size."""
        config = CapitalGoodsConfig()
        config.performance["batch_size"] = 10000
        config.validate()  # Should pass

    def test_single_worker(self):
        """Test single worker configuration."""
        config = CapitalGoodsConfig()
        config.performance["max_workers"] = 1
        config.validate()  # Should pass
