# -*- coding: utf-8 -*-
"""
Test suite for franchises.config - AGENT-MRV-027.

Tests configuration management for the Franchises Agent (GL-MRV-S3-014)
including default values, environment variable loading, singleton pattern,
thread safety, validation, and serialization.

Coverage:
- Default config values for GeneralConfig, DatabaseConfig, FranchiseSpecificConfig,
  AverageDataConfig, SpendBasedConfig, HybridConfig, ComplianceConfig,
  EFSourceConfig, UncertaintyConfig, CacheConfig, APIConfig,
  ProvenanceConfig, MetricsConfig, HotelConfig, QSRConfig,
  ConvenienceStoreConfig, RetailConfig
- GL_FRN_ environment variable loading (monkeypatch)
- Singleton pattern (get_config returns same instance)
- Thread safety
- Validation (invalid values raise errors)
- to_dict / from_dict round-trip
- Frozen immutability
- _reset_config() function

Author: GL-TestEngineer
Date: February 2026
"""

import os
import threading
from decimal import Decimal
from typing import Any, Dict
import pytest

from greenlang.agents.mrv.franchises.config import (
    get_config,
    GeneralConfig,
    DatabaseConfig,
    FranchiseSpecificConfig,
    AverageDataConfig,
    SpendBasedConfig,
    HybridConfig,
    ComplianceConfig,
    EFSourceConfig,
    UncertaintyConfig,
    CacheConfig,
    APIConfig,
    ProvenanceConfig,
    MetricsConfig,
    HotelConfig,
    QSRConfig,
    ConvenienceStoreConfig,
    RetailConfig,
    _reset_config,
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
        """Test default agent_id is GL-MRV-S3-014."""
        config = GeneralConfig()
        assert config.agent_id == "GL-MRV-S3-014"
        assert config.agent_component == "AGENT-MRV-027"
        assert config.version == "1.0.0"

    def test_general_config_api_prefix(self):
        """Test default API prefix."""
        config = GeneralConfig()
        assert config.api_prefix == "/api/v1/franchises"

    def test_general_config_default_gwp(self):
        """Test default GWP is AR6."""
        config = GeneralConfig()
        assert config.default_gwp == "AR6"

    def test_general_config_default_ef_source(self):
        """Test default EF source."""
        config = GeneralConfig()
        assert config.default_ef_source in ("DEFRA_2024", "EPA_2024", "IEA_2024")

    def test_config_frozen_immutability(self):
        """Test config is immutable (frozen=True)."""
        config = GeneralConfig()
        with pytest.raises(Exception):
            config.enabled = False

    def test_validate_config_no_errors(self):
        """Test validation passes for default config."""
        config = GeneralConfig()
        config.validate()

    def test_validate_config_invalid_log_level(self):
        """Test validation fails for invalid log level."""
        config = GeneralConfig(log_level="INVALID")
        with pytest.raises(ValueError, match="log_level"):
            config.validate()

    def test_validate_config_empty_agent_id(self):
        """Test validation fails for empty agent_id."""
        config = GeneralConfig(agent_id="")
        with pytest.raises(ValueError, match="agent_id"):
            config.validate()

    def test_validate_config_invalid_version(self):
        """Test validation fails for invalid SemVer format."""
        config = GeneralConfig(version="1.0")
        with pytest.raises(ValueError, match="version"):
            config.validate()

    def test_validate_config_invalid_gwp(self):
        """Test validation fails for invalid GWP version."""
        config = GeneralConfig(default_gwp="AR99")
        with pytest.raises(ValueError, match="default_gwp"):
            config.validate()

    def test_validate_config_invalid_batch_size(self):
        """Test validation fails for out-of-range batch size."""
        config = GeneralConfig(max_batch_size=0)
        with pytest.raises(ValueError, match="max_batch_size"):
            config.validate()

    def test_validate_config_invalid_api_prefix(self):
        """Test validation fails for API prefix not starting with /."""
        config = GeneralConfig(api_prefix="api/v1/franchises")
        with pytest.raises(ValueError, match="api_prefix"):
            config.validate()

    def test_to_dict_round_trip(self):
        """Test to_dict produces valid dictionary."""
        config = GeneralConfig()
        d = config.to_dict()
        assert d["agent_id"] == "GL-MRV-S3-014"
        assert d["enabled"] is True

    def test_from_dict_round_trip(self):
        """Test from_dict reconstructs config."""
        config = GeneralConfig()
        d = config.to_dict()
        restored = GeneralConfig.from_dict(d)
        assert restored.agent_id == config.agent_id
        assert restored.enabled == config.enabled


# ==============================================================================
# DATABASE CONFIGURATION TESTS
# ==============================================================================


class TestDatabaseConfig:
    """Test DatabaseConfig dataclass."""

    def test_database_config_defaults(self):
        """Test default database config values."""
        config = DatabaseConfig()
        assert config.host is not None
        assert config.port == 5432

    def test_database_config_table_prefix(self):
        """Test table prefix is gl_frn_."""
        config = DatabaseConfig()
        assert config.table_prefix == "gl_frn_"

    def test_database_config_frozen(self):
        """Test database config is frozen."""
        config = DatabaseConfig()
        with pytest.raises(Exception):
            config.port = 9999


# ==============================================================================
# FRANCHISE-SPECIFIC CONFIGURATION TESTS
# ==============================================================================


class TestFranchiseSpecificConfig:
    """Test FranchiseSpecificConfig dataclass."""

    def test_franchise_specific_defaults(self):
        """Test default franchise-specific config values."""
        config = FranchiseSpecificConfig()
        assert config.enable_cooking_energy is True
        assert config.enable_refrigerant_tracking is True
        assert config.enable_delivery_fleet is True
        assert config.enable_wtt is True

    def test_franchise_specific_frozen(self):
        """Test franchise-specific config is frozen."""
        config = FranchiseSpecificConfig()
        with pytest.raises(Exception):
            config.enable_cooking_energy = False


# ==============================================================================
# AVERAGE-DATA CONFIGURATION TESTS
# ==============================================================================


class TestAverageDataConfig:
    """Test AverageDataConfig dataclass."""

    def test_average_data_defaults(self):
        """Test default average-data config values."""
        config = AverageDataConfig()
        assert config.default_climate_zone == "temperate"
        assert config.enable_hotel_class_adjustment is True
        assert config.enable_climate_adjustment is True

    def test_average_data_frozen(self):
        """Test average-data config is frozen."""
        config = AverageDataConfig()
        with pytest.raises(Exception):
            config.default_climate_zone = "polar"


# ==============================================================================
# SPEND-BASED CONFIGURATION TESTS
# ==============================================================================


class TestSpendBasedConfig:
    """Test SpendBasedConfig dataclass."""

    def test_spend_based_defaults(self):
        """Test default spend-based config values."""
        config = SpendBasedConfig()
        assert config.default_eeio_source == "USEEIO_v2"
        assert config.base_year == 2022

    def test_spend_based_margin_removal(self):
        """Test default margin removal rate."""
        config = SpendBasedConfig()
        assert config.default_margin_rate == Decimal("0.15")

    def test_spend_based_frozen(self):
        """Test spend-based config is frozen."""
        config = SpendBasedConfig()
        with pytest.raises(Exception):
            config.default_eeio_source = "INVALID"


# ==============================================================================
# HYBRID CONFIGURATION TESTS
# ==============================================================================


class TestHybridConfig:
    """Test HybridConfig dataclass."""

    def test_hybrid_defaults(self):
        """Test default hybrid config values."""
        config = HybridConfig()
        assert config.waterfall_order is not None
        waterfall_list = config.get_waterfall_list()
        assert len(waterfall_list) == 3

    def test_hybrid_frozen(self):
        """Test hybrid config is frozen."""
        config = HybridConfig()
        with pytest.raises(Exception):
            config.waterfall_order = ""


# ==============================================================================
# COMPLIANCE CONFIGURATION TESTS
# ==============================================================================


class TestComplianceConfig:
    """Test ComplianceConfig dataclass."""

    def test_compliance_defaults(self):
        """Test default compliance config values."""
        config = ComplianceConfig()
        assert "ghg_protocol" in config.default_frameworks
        assert config.require_dc_check is True

    def test_compliance_frozen(self):
        """Test compliance config is frozen."""
        config = ComplianceConfig()
        with pytest.raises(Exception):
            config.require_dc_check = False


# ==============================================================================
# OTHER CONFIGURATION SECTION TESTS
# ==============================================================================


class TestEFSourceConfig:
    """Test EFSourceConfig dataclass."""

    def test_ef_source_defaults(self):
        """Test default EF source config values."""
        config = EFSourceConfig()
        assert config is not None
        assert config.primary_source == "DEFRA_2024"

    def test_ef_source_frozen(self):
        """Test EF source config is frozen."""
        config = EFSourceConfig()
        with pytest.raises(Exception):
            config.primary_source = "INVALID"


class TestUncertaintyConfig:
    """Test UncertaintyConfig dataclass."""

    def test_uncertainty_defaults(self):
        """Test default uncertainty config values."""
        config = UncertaintyConfig()
        assert config.default_method == "ipcc_tier2"
        assert config.monte_carlo_iterations == 10000

    def test_uncertainty_frozen(self):
        """Test uncertainty config is frozen."""
        config = UncertaintyConfig()
        with pytest.raises(Exception):
            config.monte_carlo_iterations = 0


class TestCacheConfig:
    """Test CacheConfig dataclass."""

    def test_cache_defaults(self):
        """Test default cache config values."""
        config = CacheConfig()
        assert config.enabled is True
        assert config.ef_ttl_seconds > 0

    def test_cache_frozen(self):
        """Test cache config is frozen."""
        config = CacheConfig()
        with pytest.raises(Exception):
            config.enabled = False


class TestAPIConfig:
    """Test APIConfig dataclass."""

    def test_api_defaults(self):
        """Test default API config values."""
        config = APIConfig()
        assert config.port == 8027
        assert config.workers == 4

    def test_api_frozen(self):
        """Test API config is frozen."""
        config = APIConfig()
        with pytest.raises(Exception):
            config.port = 9999


class TestProvenanceConfig:
    """Test ProvenanceConfig dataclass."""

    def test_provenance_defaults(self):
        """Test default provenance config values."""
        config = ProvenanceConfig()
        assert config.hash_algorithm == "sha256"
        assert config.enable_merkle is True

    def test_provenance_frozen(self):
        """Test provenance config is frozen."""
        config = ProvenanceConfig()
        with pytest.raises(Exception):
            config.hash_algorithm = "md5"


class TestMetricsConfig:
    """Test MetricsConfig dataclass."""

    def test_metrics_defaults(self):
        """Test default metrics config values."""
        config = MetricsConfig()
        assert config.enabled is True
        assert config.prefix == "gl_frn_"

    def test_metrics_frozen(self):
        """Test metrics config is frozen."""
        config = MetricsConfig()
        with pytest.raises(Exception):
            config.enabled = False


class TestHotelConfig:
    """Test HotelConfig dataclass."""

    def test_hotel_defaults(self):
        """Test default hotel config values."""
        config = HotelConfig()
        assert config is not None
        assert config.default_class_type == "midscale"

    def test_hotel_frozen(self):
        """Test hotel config is frozen."""
        config = HotelConfig()
        with pytest.raises(Exception):
            config.default_class_type = "invalid"


class TestQSRConfig:
    """Test QSRConfig dataclass."""

    def test_qsr_defaults(self):
        """Test default QSR config values."""
        config = QSRConfig()
        assert config is not None
        assert config.default_cooking_fuel_split == Decimal("0.55")

    def test_qsr_frozen(self):
        """Test QSR config is frozen."""
        config = QSRConfig()
        with pytest.raises(Exception):
            config.default_cooking_fuel_split = Decimal("0")


class TestConvenienceStoreConfig:
    """Test ConvenienceStoreConfig dataclass."""

    def test_convenience_defaults(self):
        """Test default convenience store config values."""
        config = ConvenienceStoreConfig()
        assert config.default_24h_operation is True

    def test_convenience_frozen(self):
        """Test convenience store config is frozen."""
        config = ConvenienceStoreConfig()
        with pytest.raises(Exception):
            config.default_24h_operation = False


class TestRetailConfig:
    """Test RetailConfig dataclass."""

    def test_retail_defaults(self):
        """Test default retail config values."""
        config = RetailConfig()
        assert config is not None
        assert config.enable_lighting_adjustment is True

    def test_retail_frozen(self):
        """Test retail config is frozen."""
        config = RetailConfig()
        with pytest.raises(Exception):
            config.enable_lighting_adjustment = False


# ==============================================================================
# ENVIRONMENT VARIABLE LOADING TESTS
# ==============================================================================


class TestEnvironmentVariableLoading:
    """Test GL_FRN_ environment variable override."""

    def test_env_override_enabled(self, monkeypatch):
        """Test GL_FRN_ENABLED env var override."""
        monkeypatch.setenv("GL_FRN_ENABLED", "false")
        _reset_config()
        config = get_config()
        assert config.general.enabled is False

    def test_env_override_debug(self, monkeypatch):
        """Test GL_FRN_DEBUG env var override."""
        monkeypatch.setenv("GL_FRN_DEBUG", "true")
        _reset_config()
        config = get_config()
        assert config.general.debug is True

    def test_env_override_log_level(self, monkeypatch):
        """Test GL_FRN_LOG_LEVEL env var override."""
        monkeypatch.setenv("GL_FRN_LOG_LEVEL", "DEBUG")
        _reset_config()
        config = get_config()
        assert config.general.log_level == "DEBUG"

    def test_env_override_max_batch_size(self, monkeypatch):
        """Test GL_FRN_MAX_BATCH_SIZE env var override."""
        monkeypatch.setenv("GL_FRN_MAX_BATCH_SIZE", "5000")
        _reset_config()
        config = get_config()
        assert config.general.max_batch_size == 5000


# ==============================================================================
# SINGLETON PATTERN TESTS
# ==============================================================================


class TestSingletonPattern:
    """Test singleton behavior of get_config."""

    def test_singleton_returns_same_instance(self):
        """Test get_config returns the same instance on repeated calls."""
        _reset_config()
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2

    def test_reset_creates_new_instance(self):
        """Test _reset_config creates a new instance."""
        config1 = get_config()
        _reset_config()
        config2 = get_config()
        assert config1 is not config2

    def test_thread_safety(self):
        """Test singleton is thread-safe."""
        _reset_config()
        results = []
        errors = []

        def get_config_thread():
            try:
                config = get_config()
                results.append(id(config))
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=get_config_thread) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(set(results)) == 1


# ==============================================================================
# FROZEN IMMUTABILITY TESTS
# ==============================================================================


class TestFrozenImmutability:
    """Test that all config sections are frozen (immutable)."""

    def test_general_frozen(self):
        """Test GeneralConfig is frozen."""
        _reset_config()
        config = get_config()
        with pytest.raises(Exception):
            config.general.enabled = False

    def test_database_frozen(self):
        """Test DatabaseConfig is frozen."""
        _reset_config()
        config = get_config()
        with pytest.raises(Exception):
            config.database.port = 9999
