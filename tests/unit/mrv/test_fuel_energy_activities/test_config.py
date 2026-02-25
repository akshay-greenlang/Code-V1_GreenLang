# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-MRV-016 Fuel & Energy Activities Agent configuration.

Tests configuration loading, validation, defaults, environment variable handling,
and singleton pattern.
"""

import pytest
import os
from decimal import Decimal
from typing import Dict, Any
from unittest.mock import patch

from greenlang.fuel_energy_activities.config import (
    FuelEnergyActivitiesConfig,
    DatabaseConfig,
    CalculationConfig,
    DQIConfig,
    ComplianceConfig,
    CacheConfig,
    APIConfig,
    MetricsConfig,
    ProvenanceConfig,
)
from greenlang.fuel_energy_activities.models import RegulatoryFramework


# ============================================================================
# DATABASE CONFIG TESTS
# ============================================================================

class TestDatabaseConfig:
    """Test DatabaseConfig."""

    def test_default_config(self):
        """Test DatabaseConfig with default values."""
        config = DatabaseConfig()

        assert config.host == "localhost"
        assert config.port == 5432
        assert config.database == "greenlang"
        assert config.user == "greenlang_user"
        assert config.pool_size == 10
        assert config.max_overflow == 20
        assert config.pool_timeout == 30
        assert config.pool_recycle == 3600

    def test_from_env(self):
        """Test DatabaseConfig from environment variables."""
        env_vars = {
            "FEA_DB_HOST": "prod-db.example.com",
            "FEA_DB_PORT": "5433",
            "FEA_DB_NAME": "production_db",
            "FEA_DB_USER": "prod_user",
            "FEA_DB_PASSWORD": "secure_password",
            "FEA_DB_POOL_SIZE": "20",
            "FEA_DB_MAX_OVERFLOW": "40"
        }

        with patch.dict(os.environ, env_vars):
            config = DatabaseConfig.from_env()

            assert config.host == "prod-db.example.com"
            assert config.port == 5433
            assert config.database == "production_db"
            assert config.user == "prod_user"
            assert config.password == "secure_password"
            assert config.pool_size == 20
            assert config.max_overflow == 40

    def test_connection_string(self):
        """Test DatabaseConfig generates correct connection string."""
        config = DatabaseConfig(
            host="localhost",
            port=5432,
            database="greenlang",
            user="test_user",
            password="test_pass"
        )

        conn_str = config.connection_string()
        assert conn_str == "postgresql://test_user:test_pass@localhost:5432/greenlang"

    def test_connection_string_no_password(self):
        """Test DatabaseConfig connection string without password."""
        config = DatabaseConfig(
            host="localhost",
            port=5432,
            database="greenlang",
            user="test_user",
            password=None
        )

        conn_str = config.connection_string()
        assert conn_str == "postgresql://test_user@localhost:5432/greenlang"


# ============================================================================
# CALCULATION CONFIG TESTS
# ============================================================================

class TestCalculationConfig:
    """Test CalculationConfig."""

    def test_default_config(self):
        """Test CalculationConfig with default values."""
        config = CalculationConfig()

        assert config.default_country == "US"
        assert config.default_quality_tier == "TIER_2"
        assert config.include_biogenic_co2 is False
        assert config.gwp_version == "AR6"
        assert config.ch4_gwp == Decimal("29.8")
        assert config.n2o_gwp == Decimal("273.0")
        assert config.uncertainty_method == "monte_carlo"
        assert config.monte_carlo_iterations == 10000

    def test_from_env(self):
        """Test CalculationConfig from environment variables."""
        env_vars = {
            "FEA_DEFAULT_COUNTRY": "GB",
            "FEA_DEFAULT_QUALITY_TIER": "TIER_3",
            "FEA_INCLUDE_BIOGENIC_CO2": "true",
            "FEA_GWP_VERSION": "AR5",
            "FEA_CH4_GWP": "28.0",
            "FEA_N2O_GWP": "265.0"
        }

        with patch.dict(os.environ, env_vars):
            config = CalculationConfig.from_env()

            assert config.default_country == "GB"
            assert config.default_quality_tier == "TIER_3"
            assert config.include_biogenic_co2 is True
            assert config.gwp_version == "AR5"
            assert config.ch4_gwp == Decimal("28.0")
            assert config.n2o_gwp == Decimal("265.0")

    def test_validation_invalid_gwp_version(self):
        """Test CalculationConfig rejects invalid GWP version."""
        with pytest.raises(ValueError):
            CalculationConfig(gwp_version="AR3")  # Invalid version

    def test_validation_negative_gwp(self):
        """Test CalculationConfig rejects negative GWP values."""
        with pytest.raises(ValueError):
            CalculationConfig(ch4_gwp=Decimal("-10.0"))

    def test_validation_monte_carlo_iterations(self):
        """Test CalculationConfig validates Monte Carlo iterations."""
        with pytest.raises(ValueError):
            CalculationConfig(monte_carlo_iterations=100)  # Too low


# ============================================================================
# DQI CONFIG TESTS
# ============================================================================

class TestDQIConfig:
    """Test DQIConfig."""

    def test_default_config(self):
        """Test DQIConfig with default values."""
        config = DQIConfig()

        assert config.enable_dqi is True
        assert config.min_score == Decimal("0.7")
        assert config.weights["completeness"] == Decimal("0.20")
        assert config.weights["accuracy"] == Decimal("0.25")
        assert config.weights["consistency"] == Decimal("0.15")
        assert config.weights["timeliness"] == Decimal("0.15")
        assert config.weights["reliability"] == Decimal("0.25")

    def test_dqi_weights_sum(self):
        """Test DQI weights sum to 1.0."""
        config = DQIConfig()
        total_weight = sum(config.weights.values())
        assert total_weight == Decimal("1.0")

    def test_from_env(self):
        """Test DQIConfig from environment variables."""
        env_vars = {
            "FEA_DQI_ENABLE": "false",
            "FEA_DQI_MIN_SCORE": "0.8"
        }

        with patch.dict(os.environ, env_vars):
            config = DQIConfig.from_env()

            assert config.enable_dqi is False
            assert config.min_score == Decimal("0.8")

    def test_validation_min_score_range(self):
        """Test DQIConfig validates min_score range."""
        with pytest.raises(ValueError):
            DQIConfig(min_score=Decimal("1.5"))  # > 1.0

        with pytest.raises(ValueError):
            DQIConfig(min_score=Decimal("-0.1"))  # < 0.0


# ============================================================================
# COMPLIANCE CONFIG TESTS
# ============================================================================

class TestComplianceConfig:
    """Test ComplianceConfig."""

    def test_default_config(self):
        """Test ComplianceConfig with default values."""
        config = ComplianceConfig()

        assert config.enable_compliance_checks is True
        assert RegulatoryFramework.GHG_PROTOCOL in config.frameworks
        assert RegulatoryFramework.ISO_14064 in config.frameworks
        assert config.min_coverage_percentage == Decimal("70.0")
        assert config.require_supplier_data is False

    def test_from_env(self):
        """Test ComplianceConfig from environment variables."""
        env_vars = {
            "FEA_COMPLIANCE_ENABLE": "true",
            "FEA_COMPLIANCE_FRAMEWORKS": "GHG_PROTOCOL,CSRD,TCFD",
            "FEA_MIN_COVERAGE_PERCENTAGE": "85.0",
            "FEA_REQUIRE_SUPPLIER_DATA": "true"
        }

        with patch.dict(os.environ, env_vars):
            config = ComplianceConfig.from_env()

            assert config.enable_compliance_checks is True
            assert RegulatoryFramework.GHG_PROTOCOL in config.frameworks
            assert RegulatoryFramework.CSRD in config.frameworks
            assert RegulatoryFramework.TCFD in config.frameworks
            assert config.min_coverage_percentage == Decimal("85.0")
            assert config.require_supplier_data is True

    def test_validation_coverage_percentage(self):
        """Test ComplianceConfig validates coverage percentage."""
        with pytest.raises(ValueError):
            ComplianceConfig(min_coverage_percentage=Decimal("150.0"))  # > 100


# ============================================================================
# CACHE CONFIG TESTS
# ============================================================================

class TestCacheConfig:
    """Test CacheConfig."""

    def test_default_config(self):
        """Test CacheConfig with default values."""
        config = CacheConfig()

        assert config.enable_caching is True
        assert config.cache_backend == "redis"
        assert config.redis_host == "localhost"
        assert config.redis_port == 6379
        assert config.ttl_seconds == 3600
        assert config.max_size == 10000

    def test_from_env(self):
        """Test CacheConfig from environment variables."""
        env_vars = {
            "FEA_CACHE_ENABLE": "true",
            "FEA_CACHE_BACKEND": "redis",
            "FEA_REDIS_HOST": "cache.example.com",
            "FEA_REDIS_PORT": "6380",
            "FEA_CACHE_TTL_SECONDS": "7200",
            "FEA_CACHE_MAX_SIZE": "50000"
        }

        with patch.dict(os.environ, env_vars):
            config = CacheConfig.from_env()

            assert config.enable_caching is True
            assert config.redis_host == "cache.example.com"
            assert config.redis_port == 6380
            assert config.ttl_seconds == 7200
            assert config.max_size == 50000

    def test_validation_ttl_positive(self):
        """Test CacheConfig validates TTL is positive."""
        with pytest.raises(ValueError):
            CacheConfig(ttl_seconds=-100)


# ============================================================================
# API CONFIG TESTS
# ============================================================================

class TestAPIConfig:
    """Test APIConfig."""

    def test_default_config(self):
        """Test APIConfig with default values."""
        config = APIConfig()

        assert config.enable_api is True
        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.timeout_seconds == 30
        assert config.max_request_size_mb == 100
        assert config.rate_limit_per_minute == 100

    def test_from_env(self):
        """Test APIConfig from environment variables."""
        env_vars = {
            "FEA_API_ENABLE": "true",
            "FEA_API_HOST": "127.0.0.1",
            "FEA_API_PORT": "9000",
            "FEA_API_TIMEOUT": "60",
            "FEA_API_MAX_REQUEST_SIZE_MB": "200",
            "FEA_API_RATE_LIMIT": "200"
        }

        with patch.dict(os.environ, env_vars):
            config = APIConfig.from_env()

            assert config.host == "127.0.0.1"
            assert config.port == 9000
            assert config.timeout_seconds == 60
            assert config.max_request_size_mb == 200
            assert config.rate_limit_per_minute == 200


# ============================================================================
# METRICS CONFIG TESTS
# ============================================================================

class TestMetricsConfig:
    """Test MetricsConfig."""

    def test_default_config(self):
        """Test MetricsConfig with default values."""
        config = MetricsConfig()

        assert config.enable_metrics is True
        assert config.metrics_backend == "prometheus"
        assert config.prometheus_port == 9090
        assert config.enable_histograms is True

    def test_from_env(self):
        """Test MetricsConfig from environment variables."""
        env_vars = {
            "FEA_METRICS_ENABLE": "true",
            "FEA_METRICS_BACKEND": "prometheus",
            "FEA_PROMETHEUS_PORT": "9091",
            "FEA_ENABLE_HISTOGRAMS": "false"
        }

        with patch.dict(os.environ, env_vars):
            config = MetricsConfig.from_env()

            assert config.enable_metrics is True
            assert config.prometheus_port == 9091
            assert config.enable_histograms is False


# ============================================================================
# PROVENANCE CONFIG TESTS
# ============================================================================

class TestProvenanceConfig:
    """Test ProvenanceConfig."""

    def test_default_config(self):
        """Test ProvenanceConfig with default values."""
        config = ProvenanceConfig()

        assert config.enable_provenance is True
        assert config.hash_algorithm == "sha256"
        assert config.store_provenance_chain is True
        assert config.max_chain_length == 1000

    def test_from_env(self):
        """Test ProvenanceConfig from environment variables."""
        env_vars = {
            "FEA_PROVENANCE_ENABLE": "true",
            "FEA_PROVENANCE_HASH_ALGORITHM": "sha512",
            "FEA_PROVENANCE_STORE_CHAIN": "false",
            "FEA_PROVENANCE_MAX_CHAIN_LENGTH": "500"
        }

        with patch.dict(os.environ, env_vars):
            config = ProvenanceConfig.from_env()

            assert config.enable_provenance is True
            assert config.hash_algorithm == "sha512"
            assert config.store_provenance_chain is False
            assert config.max_chain_length == 500

    def test_validation_hash_algorithm(self):
        """Test ProvenanceConfig validates hash algorithm."""
        with pytest.raises(ValueError):
            ProvenanceConfig(hash_algorithm="md5")  # Weak algorithm


# ============================================================================
# MAIN CONFIG TESTS
# ============================================================================

class TestFuelEnergyActivitiesConfig:
    """Test FuelEnergyActivitiesConfig (main config)."""

    def test_default_config(self):
        """Test FuelEnergyActivitiesConfig with default values."""
        config = FuelEnergyActivitiesConfig()

        assert isinstance(config.database, DatabaseConfig)
        assert isinstance(config.calculation, CalculationConfig)
        assert isinstance(config.dqi, DQIConfig)
        assert isinstance(config.compliance, ComplianceConfig)
        assert isinstance(config.cache, CacheConfig)
        assert isinstance(config.api, APIConfig)
        assert isinstance(config.metrics, MetricsConfig)
        assert isinstance(config.provenance, ProvenanceConfig)

    def test_from_env(self):
        """Test FuelEnergyActivitiesConfig from environment variables."""
        env_vars = {
            "FEA_DB_HOST": "prod-db.example.com",
            "FEA_DEFAULT_COUNTRY": "GB",
            "FEA_DQI_MIN_SCORE": "0.8",
            "FEA_COMPLIANCE_ENABLE": "true",
            "FEA_CACHE_ENABLE": "true",
            "FEA_API_PORT": "9000",
            "FEA_METRICS_ENABLE": "true",
            "FEA_PROVENANCE_ENABLE": "true"
        }

        with patch.dict(os.environ, env_vars):
            config = FuelEnergyActivitiesConfig.from_env()

            assert config.database.host == "prod-db.example.com"
            assert config.calculation.default_country == "GB"
            assert config.dqi.min_score == Decimal("0.8")
            assert config.compliance.enable_compliance_checks is True
            assert config.cache.enable_caching is True
            assert config.api.port == 9000

    def test_singleton(self):
        """Test FuelEnergyActivitiesConfig implements singleton pattern."""
        config1 = FuelEnergyActivitiesConfig.get_instance()
        config2 = FuelEnergyActivitiesConfig.get_instance()

        assert config1 is config2

    def test_thread_safety(self):
        """Test FuelEnergyActivitiesConfig singleton is thread-safe."""
        import threading

        configs = []

        def get_config():
            config = FuelEnergyActivitiesConfig.get_instance()
            configs.append(config)

        threads = [threading.Thread(target=get_config) for _ in range(10)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # All configs should be the same instance
        assert all(config is configs[0] for config in configs)

    def test_to_dict(self):
        """Test FuelEnergyActivitiesConfig.to_dict()."""
        config = FuelEnergyActivitiesConfig()
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert "database" in config_dict
        assert "calculation" in config_dict
        assert "dqi" in config_dict
        assert "compliance" in config_dict

    def test_to_dict_redacts_password(self):
        """Test FuelEnergyActivitiesConfig.to_dict() redacts password."""
        config = FuelEnergyActivitiesConfig()
        config.database.password = "super_secret_password"

        config_dict = config.to_dict()

        assert config_dict["database"]["password"] == "***REDACTED***"

    def test_validation_errors(self):
        """Test FuelEnergyActivitiesConfig validation catches errors."""
        with pytest.raises(ValueError):
            FuelEnergyActivitiesConfig(
                dqi=DQIConfig(min_score=Decimal("2.0"))  # Invalid > 1.0
            )

    def test_database_config(self):
        """Test FuelEnergyActivitiesConfig database configuration."""
        config = FuelEnergyActivitiesConfig()

        assert config.database.host == "localhost"
        assert config.database.port == 5432
        assert config.database.database == "greenlang"

    def test_calculation_config(self):
        """Test FuelEnergyActivitiesConfig calculation configuration."""
        config = FuelEnergyActivitiesConfig()

        assert config.calculation.default_country == "US"
        assert config.calculation.gwp_version == "AR6"
        assert config.calculation.ch4_gwp == Decimal("29.8")

    def test_dqi_config(self):
        """Test FuelEnergyActivitiesConfig DQI configuration."""
        config = FuelEnergyActivitiesConfig()

        assert config.dqi.enable_dqi is True
        assert config.dqi.min_score == Decimal("0.7")

    def test_compliance_config(self):
        """Test FuelEnergyActivitiesConfig compliance configuration."""
        config = FuelEnergyActivitiesConfig()

        assert config.compliance.enable_compliance_checks is True
        assert len(config.compliance.frameworks) >= 2

    def test_cache_config(self):
        """Test FuelEnergyActivitiesConfig cache configuration."""
        config = FuelEnergyActivitiesConfig()

        assert config.cache.enable_caching is True
        assert config.cache.cache_backend == "redis"

    def test_api_config(self):
        """Test FuelEnergyActivitiesConfig API configuration."""
        config = FuelEnergyActivitiesConfig()

        assert config.api.enable_api is True
        assert config.api.port == 8000

    def test_metrics_config(self):
        """Test FuelEnergyActivitiesConfig metrics configuration."""
        config = FuelEnergyActivitiesConfig()

        assert config.metrics.enable_metrics is True
        assert config.metrics.metrics_backend == "prometheus"

    def test_provenance_config(self):
        """Test FuelEnergyActivitiesConfig provenance configuration."""
        config = FuelEnergyActivitiesConfig()

        assert config.provenance.enable_provenance is True
        assert config.provenance.hash_algorithm == "sha256"
