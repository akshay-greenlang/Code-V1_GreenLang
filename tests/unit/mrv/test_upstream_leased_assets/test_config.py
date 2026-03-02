# -*- coding: utf-8 -*-
"""
Test suite for upstream_leased_assets.config - AGENT-MRV-021.

Tests configuration management for the Upstream Leased Assets Agent
(GL-MRV-S3-008) including default values, environment variable loading,
singleton pattern, thread safety, validation, and serialization.

Coverage:
- Default config values for GeneralConfig, DatabaseConfig, BuildingConfig,
  VehicleConfig, EquipmentConfig, ITAssetsConfig, AllocationConfig,
  ComplianceConfig, EFSourceConfig, UncertaintyConfig, CacheConfig,
  APIConfig, ProvenanceConfig, MetricsConfig, SpendConfig
- GL_ULA_ environment variable loading (monkeypatch)
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

try:
    from greenlang.upstream_leased_assets.config import (
        get_config,
        GeneralConfig,
        DatabaseConfig,
        BuildingConfig,
        VehicleConfig,
        EquipmentConfig,
        ITAssetsConfig,
        AllocationConfig,
        ComplianceConfig,
        EFSourceConfig,
        UncertaintyConfig,
        CacheConfig,
        APIConfig,
        ProvenanceConfig,
        MetricsConfig,
        SpendConfig,
        UpstreamLeasedConfig,
    )
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

_SKIP = pytest.mark.skipif(
    not _AVAILABLE,
    reason="upstream_leased_assets.config not available",
)

pytestmark = _SKIP


# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture(autouse=True)
def _reset_config_singleton():
    """Reset config singleton before and after every test."""
    if _AVAILABLE:
        try:
            from greenlang.upstream_leased_assets.config import _reset_config
            _reset_config()
        except (ImportError, AttributeError):
            pass
    yield
    if _AVAILABLE:
        try:
            from greenlang.upstream_leased_assets.config import _reset_config
            _reset_config()
        except (ImportError, AttributeError):
            pass


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
        """Test default agent_id is GL-MRV-S3-008."""
        config = GeneralConfig()
        assert config.agent_id == "GL-MRV-S3-008"
        assert config.agent_component == "AGENT-MRV-021"
        assert config.version == "1.0.0"

    def test_general_config_api_prefix(self):
        """Test default API prefix."""
        config = GeneralConfig()
        assert config.api_prefix == "/api/v1/upstream-leased-assets"

    def test_general_config_default_gwp(self):
        """Test default GWP is AR5."""
        config = GeneralConfig()
        assert config.default_gwp == "AR5"

    def test_general_config_default_ef_source(self):
        """Test default EF source is DEFRA."""
        config = GeneralConfig()
        assert config.default_ef_source == "DEFRA"

    def test_general_config_table_prefix(self):
        """Test default table prefix is gl_ula_."""
        config = GeneralConfig()
        assert config.table_prefix == "gl_ula_"

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

    def test_to_dict(self):
        """Test to_dict serialization."""
        config = GeneralConfig()
        d = config.to_dict()
        assert isinstance(d, dict)
        assert d["agent_id"] == "GL-MRV-S3-008"
        assert d["enabled"] is True

    def test_from_dict(self):
        """Test from_dict deserialization."""
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
        assert config.cache_enabled is True
        assert config.cache_ttl_seconds == 3600
        assert config.quantize_decimals == 8

    def test_frozen(self):
        """Test DatabaseConfig is frozen."""
        config = DatabaseConfig()
        with pytest.raises(Exception):
            config.cache_enabled = False

    def test_to_dict(self):
        """Test DatabaseConfig to_dict."""
        config = DatabaseConfig()
        d = config.to_dict()
        assert isinstance(d, dict)
        assert "cache_enabled" in d

    def test_from_dict(self):
        """Test DatabaseConfig from_dict."""
        config = DatabaseConfig()
        d = config.to_dict()
        restored = DatabaseConfig.from_dict(d)
        assert restored.cache_ttl_seconds == config.cache_ttl_seconds


# ==============================================================================
# BUILDING CONFIGURATION TESTS
# ==============================================================================


class TestBuildingConfig:
    """Test BuildingConfig dataclass."""

    def test_building_config_defaults(self):
        """Test default building config values."""
        config = BuildingConfig()
        assert config.default_climate_zone == "temperate"
        assert config.default_allocation_method == "area"
        assert config.include_wtt is True

    def test_max_area_sqm(self):
        """Test maximum floor area setting."""
        config = BuildingConfig()
        assert config.max_floor_area_sqm >= 100000

    def test_default_pue(self):
        """Test default PUE for data centers."""
        config = BuildingConfig()
        assert config.default_pue == Decimal("1.40") or \
            config.default_pue == Decimal("1.58")

    def test_frozen(self):
        """Test BuildingConfig is frozen."""
        config = BuildingConfig()
        with pytest.raises(Exception):
            config.default_climate_zone = "cold"

    def test_validate_no_errors(self):
        """Test validation passes for default building config."""
        config = BuildingConfig()
        config.validate()

    def test_validate_invalid_climate_zone(self):
        """Test validation fails for invalid climate zone."""
        config = BuildingConfig(default_climate_zone="invalid_zone")
        with pytest.raises(ValueError):
            config.validate()

    def test_validate_invalid_allocation_method(self):
        """Test validation fails for invalid allocation method."""
        config = BuildingConfig(default_allocation_method="invalid_method")
        with pytest.raises(ValueError):
            config.validate()

    def test_to_dict(self):
        """Test BuildingConfig to_dict."""
        config = BuildingConfig()
        d = config.to_dict()
        assert "default_climate_zone" in d

    def test_from_dict(self):
        """Test BuildingConfig from_dict."""
        config = BuildingConfig()
        d = config.to_dict()
        restored = BuildingConfig.from_dict(d)
        assert restored.default_climate_zone == config.default_climate_zone


# ==============================================================================
# VEHICLE CONFIGURATION TESTS
# ==============================================================================


class TestVehicleConfig:
    """Test VehicleConfig dataclass."""

    def test_vehicle_config_defaults(self):
        """Test default vehicle config values."""
        config = VehicleConfig()
        assert config.include_wtt is True
        assert config.default_fuel_type == "diesel"

    def test_max_annual_distance(self):
        """Test maximum annual distance setting."""
        config = VehicleConfig()
        assert config.max_annual_distance_km >= 200000

    def test_age_degradation_enabled(self):
        """Test age degradation factor is configurable."""
        config = VehicleConfig()
        assert hasattr(config, 'age_degradation_enabled') or \
            hasattr(config, 'age_factor_enabled')

    def test_frozen(self):
        """Test VehicleConfig is frozen."""
        config = VehicleConfig()
        with pytest.raises(Exception):
            config.include_wtt = False

    def test_validate_no_errors(self):
        """Test validation passes for default vehicle config."""
        config = VehicleConfig()
        config.validate()

    def test_validate_invalid_fuel_type(self):
        """Test validation fails for invalid fuel type."""
        config = VehicleConfig(default_fuel_type="nuclear")
        with pytest.raises(ValueError):
            config.validate()

    def test_to_dict(self):
        """Test VehicleConfig to_dict."""
        config = VehicleConfig()
        d = config.to_dict()
        assert "include_wtt" in d

    def test_from_dict(self):
        """Test VehicleConfig from_dict."""
        config = VehicleConfig()
        d = config.to_dict()
        restored = VehicleConfig.from_dict(d)
        assert restored.include_wtt == config.include_wtt


# ==============================================================================
# EQUIPMENT CONFIGURATION TESTS
# ==============================================================================


class TestEquipmentConfig:
    """Test EquipmentConfig dataclass."""

    def test_equipment_config_defaults(self):
        """Test default equipment config values."""
        config = EquipmentConfig()
        assert config.include_wtt is True
        assert config.max_operating_hours == 8760

    def test_default_load_factor(self):
        """Test default load factor is reasonable."""
        config = EquipmentConfig()
        assert Decimal("0.50") <= config.default_load_factor <= Decimal("0.85")

    def test_frozen(self):
        """Test EquipmentConfig is frozen."""
        config = EquipmentConfig()
        with pytest.raises(Exception):
            config.max_operating_hours = 10000

    def test_validate_no_errors(self):
        """Test validation passes for default equipment config."""
        config = EquipmentConfig()
        config.validate()

    def test_to_dict(self):
        """Test EquipmentConfig to_dict."""
        config = EquipmentConfig()
        d = config.to_dict()
        assert "max_operating_hours" in d

    def test_from_dict(self):
        """Test EquipmentConfig from_dict."""
        config = EquipmentConfig()
        d = config.to_dict()
        restored = EquipmentConfig.from_dict(d)
        assert restored.max_operating_hours == config.max_operating_hours


# ==============================================================================
# IT ASSETS CONFIGURATION TESTS
# ==============================================================================


class TestITAssetsConfig:
    """Test ITAssetsConfig dataclass."""

    def test_it_config_defaults(self):
        """Test default IT assets config values."""
        config = ITAssetsConfig()
        assert config.default_pue == Decimal("1.58") or \
            config.default_pue == Decimal("1.40")
        assert config.include_cooling is True

    def test_default_utilization(self):
        """Test default server utilization is reasonable."""
        config = ITAssetsConfig()
        assert Decimal("0.10") <= config.default_utilization <= Decimal("0.90")

    def test_frozen(self):
        """Test ITAssetsConfig is frozen."""
        config = ITAssetsConfig()
        with pytest.raises(Exception):
            config.include_cooling = False

    def test_validate_no_errors(self):
        """Test validation passes for default IT config."""
        config = ITAssetsConfig()
        config.validate()

    def test_validate_pue_below_one(self):
        """Test validation fails for PUE below 1.0."""
        config = ITAssetsConfig(default_pue=Decimal("0.9"))
        with pytest.raises(ValueError):
            config.validate()

    def test_to_dict(self):
        """Test ITAssetsConfig to_dict."""
        config = ITAssetsConfig()
        d = config.to_dict()
        assert "default_pue" in d

    def test_from_dict(self):
        """Test ITAssetsConfig from_dict."""
        config = ITAssetsConfig()
        d = config.to_dict()
        restored = ITAssetsConfig.from_dict(d)
        assert restored.default_pue == config.default_pue


# ==============================================================================
# ALLOCATION CONFIGURATION TESTS
# ==============================================================================


class TestAllocationConfig:
    """Test AllocationConfig dataclass."""

    def test_allocation_config_defaults(self):
        """Test default allocation config values."""
        config = AllocationConfig()
        assert config.default_method == "area"
        assert config.allow_revenue_allocation is True

    def test_frozen(self):
        """Test AllocationConfig is frozen."""
        config = AllocationConfig()
        with pytest.raises(Exception):
            config.default_method = "headcount"

    def test_validate_no_errors(self):
        """Test validation passes for default allocation config."""
        config = AllocationConfig()
        config.validate()

    def test_validate_invalid_method(self):
        """Test validation fails for invalid allocation method."""
        config = AllocationConfig(default_method="random")
        with pytest.raises(ValueError):
            config.validate()

    def test_to_dict(self):
        """Test AllocationConfig to_dict."""
        config = AllocationConfig()
        d = config.to_dict()
        assert "default_method" in d


# ==============================================================================
# COMPLIANCE CONFIGURATION TESTS
# ==============================================================================


class TestComplianceConfig:
    """Test ComplianceConfig dataclass."""

    def test_compliance_config_defaults(self):
        """Test default compliance config values."""
        config = ComplianceConfig()
        assert config.strict_mode is False
        assert config.materiality_threshold == Decimal("0.01")

    def test_get_frameworks(self):
        """Test get_frameworks returns list of framework strings."""
        config = ComplianceConfig()
        frameworks = config.get_frameworks()
        assert isinstance(frameworks, list)
        assert len(frameworks) >= 1

    def test_frozen(self):
        """Test ComplianceConfig is frozen."""
        config = ComplianceConfig()
        with pytest.raises(Exception):
            config.strict_mode = True

    def test_validate_no_errors(self):
        """Test validation passes for default compliance config."""
        config = ComplianceConfig()
        config.validate()

    def test_to_dict(self):
        """Test ComplianceConfig to_dict."""
        config = ComplianceConfig()
        d = config.to_dict()
        assert "strict_mode" in d


# ==============================================================================
# EF SOURCE CONFIGURATION TESTS
# ==============================================================================


class TestEFSourceConfig:
    """Test EFSourceConfig dataclass."""

    def test_ef_source_config_defaults(self):
        """Test default EF source config values."""
        config = EFSourceConfig()
        assert config.primary_source == "DEFRA"
        assert config.fallback_source == "EPA"

    def test_frozen(self):
        """Test EFSourceConfig is frozen."""
        config = EFSourceConfig()
        with pytest.raises(Exception):
            config.primary_source = "IEA"

    def test_validate_no_errors(self):
        """Test validation passes for default EF source config."""
        config = EFSourceConfig()
        config.validate()

    def test_validate_invalid_source(self):
        """Test validation fails for invalid EF source."""
        config = EFSourceConfig(primary_source="INVALID_SOURCE")
        with pytest.raises(ValueError):
            config.validate()

    def test_to_dict(self):
        """Test EFSourceConfig to_dict."""
        config = EFSourceConfig()
        d = config.to_dict()
        assert "primary_source" in d


# ==============================================================================
# UNCERTAINTY CONFIGURATION TESTS
# ==============================================================================


class TestUncertaintyConfig:
    """Test UncertaintyConfig dataclass."""

    def test_uncertainty_config_defaults(self):
        """Test default uncertainty config values."""
        config = UncertaintyConfig()
        assert config.method == "monte_carlo"
        assert config.iterations >= 1000

    def test_confidence_interval(self):
        """Test default confidence interval is 95%."""
        config = UncertaintyConfig()
        assert config.confidence_level == Decimal("0.95") or \
            config.confidence_level == 95

    def test_frozen(self):
        """Test UncertaintyConfig is frozen."""
        config = UncertaintyConfig()
        with pytest.raises(Exception):
            config.method = "analytical"

    def test_validate_no_errors(self):
        """Test validation passes for default uncertainty config."""
        config = UncertaintyConfig()
        config.validate()

    def test_to_dict(self):
        """Test UncertaintyConfig to_dict."""
        config = UncertaintyConfig()
        d = config.to_dict()
        assert "method" in d


# ==============================================================================
# CACHE CONFIGURATION TESTS
# ==============================================================================


class TestCacheConfig:
    """Test CacheConfig dataclass."""

    def test_cache_config_defaults(self):
        """Test default cache config values."""
        config = CacheConfig()
        assert config.enabled is True
        assert config.ttl_seconds >= 300

    def test_frozen(self):
        """Test CacheConfig is frozen."""
        config = CacheConfig()
        with pytest.raises(Exception):
            config.enabled = False

    def test_to_dict(self):
        """Test CacheConfig to_dict."""
        config = CacheConfig()
        d = config.to_dict()
        assert "enabled" in d


# ==============================================================================
# API CONFIGURATION TESTS
# ==============================================================================


class TestAPIConfig:
    """Test APIConfig dataclass."""

    def test_api_config_defaults(self):
        """Test default API config values."""
        config = APIConfig()
        assert config.prefix == "/api/v1/upstream-leased-assets"
        assert config.tags == ["upstream-leased-assets"]

    def test_rate_limit(self):
        """Test default rate limit is set."""
        config = APIConfig()
        assert config.rate_limit_per_minute >= 60

    def test_frozen(self):
        """Test APIConfig is frozen."""
        config = APIConfig()
        with pytest.raises(Exception):
            config.prefix = "/api/v2/upstream-leased-assets"

    def test_to_dict(self):
        """Test APIConfig to_dict."""
        config = APIConfig()
        d = config.to_dict()
        assert "prefix" in d


# ==============================================================================
# PROVENANCE CONFIGURATION TESTS
# ==============================================================================


class TestProvenanceConfig:
    """Test ProvenanceConfig dataclass."""

    def test_provenance_config_defaults(self):
        """Test default provenance config values."""
        config = ProvenanceConfig()
        assert config.enabled is True
        assert config.hash_algorithm == "sha256"
        assert config.chain_validation is True

    def test_frozen(self):
        """Test ProvenanceConfig is frozen."""
        config = ProvenanceConfig()
        with pytest.raises(Exception):
            config.enabled = False

    def test_to_dict(self):
        """Test ProvenanceConfig to_dict."""
        config = ProvenanceConfig()
        d = config.to_dict()
        assert "hash_algorithm" in d


# ==============================================================================
# METRICS CONFIGURATION TESTS
# ==============================================================================


class TestMetricsConfig:
    """Test MetricsConfig dataclass."""

    def test_metrics_config_defaults(self):
        """Test default metrics config values."""
        config = MetricsConfig()
        assert config.enabled is True
        assert config.prefix == "gl_ula_"

    def test_frozen(self):
        """Test MetricsConfig is frozen."""
        config = MetricsConfig()
        with pytest.raises(Exception):
            config.prefix = "gl_test_"

    def test_to_dict(self):
        """Test MetricsConfig to_dict."""
        config = MetricsConfig()
        d = config.to_dict()
        assert "prefix" in d


# ==============================================================================
# SPEND CONFIGURATION TESTS
# ==============================================================================


class TestSpendConfig:
    """Test SpendConfig dataclass."""

    def test_spend_config_defaults(self):
        """Test default spend config values."""
        config = SpendConfig()
        assert config.base_currency == "USD"
        assert config.cpi_base_year == 2021

    def test_margin_removal(self):
        """Test default margin removal is enabled."""
        config = SpendConfig()
        assert config.margin_removal_enabled is True

    def test_frozen(self):
        """Test SpendConfig is frozen."""
        config = SpendConfig()
        with pytest.raises(Exception):
            config.base_currency = "EUR"

    def test_validate_no_errors(self):
        """Test validation passes for default spend config."""
        config = SpendConfig()
        config.validate()

    def test_validate_invalid_currency(self):
        """Test validation fails for invalid base currency."""
        config = SpendConfig(base_currency="XYZ")
        with pytest.raises(ValueError):
            config.validate()

    def test_to_dict(self):
        """Test SpendConfig to_dict."""
        config = SpendConfig()
        d = config.to_dict()
        assert "base_currency" in d


# ==============================================================================
# MASTER CONFIGURATION TESTS
# ==============================================================================


class TestUpstreamLeasedConfig:
    """Test UpstreamLeasedConfig master configuration."""

    def test_master_config_creation(self):
        """Test creating master config with all 15 sections."""
        config = UpstreamLeasedConfig()
        assert config.general is not None
        assert config.database is not None
        assert config.building is not None
        assert config.vehicle is not None
        assert config.equipment is not None
        assert config.it_assets is not None
        assert config.allocation is not None
        assert config.compliance is not None
        assert config.ef_source is not None
        assert config.uncertainty is not None
        assert config.cache is not None
        assert config.api is not None
        assert config.provenance is not None
        assert config.metrics is not None
        assert config.spend is not None

    def test_master_config_agent_id(self):
        """Test master config agent_id."""
        config = UpstreamLeasedConfig()
        assert config.general.agent_id == "GL-MRV-S3-008"

    def test_master_config_to_dict(self):
        """Test master config to_dict serialization."""
        config = UpstreamLeasedConfig()
        d = config.to_dict()
        assert isinstance(d, dict)
        assert "general" in d
        assert "building" in d
        assert "vehicle" in d

    def test_master_config_from_dict_roundtrip(self):
        """Test master config to_dict / from_dict round-trip."""
        config = UpstreamLeasedConfig()
        d = config.to_dict()
        restored = UpstreamLeasedConfig.from_dict(d)
        assert restored.general.agent_id == config.general.agent_id
        assert restored.building.default_climate_zone == config.building.default_climate_zone

    def test_validate_all(self):
        """Test validating all config sections."""
        config = UpstreamLeasedConfig()
        config.validate_all()


# ==============================================================================
# SINGLETON TESTS
# ==============================================================================


class TestGetConfig:
    """Test get_config singleton pattern."""

    def test_singleton_identity(self):
        """Test get_config returns the same instance."""
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2

    def test_singleton_is_upstream_leased_config(self):
        """Test get_config returns UpstreamLeasedConfig instance."""
        config = get_config()
        assert isinstance(config, UpstreamLeasedConfig)


# ==============================================================================
# ENVIRONMENT VARIABLE TESTS
# ==============================================================================


class TestEnvironmentVariables:
    """Test environment variable loading."""

    def test_gl_ula_debug_env(self, monkeypatch):
        """Test GL_ULA_DEBUG environment variable."""
        monkeypatch.setenv("GL_ULA_DEBUG", "true")
        config = get_config()
        assert config.general.debug is True

    def test_gl_ula_log_level_env(self, monkeypatch):
        """Test GL_ULA_LOG_LEVEL environment variable."""
        monkeypatch.setenv("GL_ULA_LOG_LEVEL", "DEBUG")
        config = get_config()
        assert config.general.log_level == "DEBUG"

    def test_gl_ula_max_batch_size_env(self, monkeypatch):
        """Test GL_ULA_MAX_BATCH_SIZE environment variable."""
        monkeypatch.setenv("GL_ULA_MAX_BATCH_SIZE", "5000")
        config = get_config()
        assert config.general.max_batch_size == 5000

    def test_gl_ula_default_gwp_env(self, monkeypatch):
        """Test GL_ULA_DEFAULT_GWP environment variable."""
        monkeypatch.setenv("GL_ULA_DEFAULT_GWP", "AR6")
        config = get_config()
        assert config.general.default_gwp == "AR6"

    def test_gl_ula_cache_ttl_env(self, monkeypatch):
        """Test GL_ULA_CACHE_TTL environment variable."""
        monkeypatch.setenv("GL_ULA_CACHE_TTL", "7200")
        config = get_config()
        assert config.cache.ttl_seconds == 7200


# ==============================================================================
# THREAD SAFETY TESTS
# ==============================================================================


class TestThreadSafety:
    """Test thread safety of configuration."""

    def test_concurrent_get_config(self):
        """Test concurrent access to get_config returns same instance."""
        configs = []

        def get_cfg():
            configs.append(get_config())

        threads = [threading.Thread(target=get_cfg) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        first = configs[0]
        for cfg in configs[1:]:
            assert cfg is first

    def test_concurrent_read_config_values(self):
        """Test concurrent reads of config values are consistent."""
        config = get_config()
        results = []

        def read_values():
            results.append({
                "agent_id": config.general.agent_id,
                "table_prefix": config.general.table_prefix,
                "default_gwp": config.general.default_gwp,
            })

        threads = [threading.Thread(target=read_values) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for r in results:
            assert r["agent_id"] == "GL-MRV-S3-008"
            assert r["table_prefix"] == "gl_ula_"


# ==============================================================================
# ADDITIONAL VALIDATION TESTS
# ==============================================================================


class TestAdditionalValidation:
    """Additional validation and edge case tests for config sections."""

    def test_general_config_max_batch_size_positive(self):
        """Test max_batch_size must be positive."""
        config = GeneralConfig(max_batch_size=-1)
        with pytest.raises(ValueError):
            config.validate()

    def test_general_config_max_batch_size_zero_raises(self):
        """Test max_batch_size zero raises error."""
        config = GeneralConfig(max_batch_size=0)
        with pytest.raises(ValueError):
            config.validate()

    def test_database_config_negative_ttl_raises(self):
        """Test negative cache TTL raises error."""
        config = DatabaseConfig(cache_ttl_seconds=-100)
        with pytest.raises(ValueError):
            config.validate()

    def test_database_config_quantize_decimals_range(self):
        """Test quantize_decimals must be 0-16."""
        config = DatabaseConfig(quantize_decimals=20)
        with pytest.raises(ValueError):
            config.validate()

    def test_equipment_config_max_hours_over_8760_raises(self):
        """Test max_operating_hours over 8760 raises error."""
        config = EquipmentConfig(max_operating_hours=9000)
        with pytest.raises(ValueError):
            config.validate()

    def test_it_config_pue_over_three_raises(self):
        """Test PUE over 3.0 raises error (unrealistically high)."""
        config = ITAssetsConfig(default_pue=Decimal("3.5"))
        with pytest.raises(ValueError):
            config.validate()

    def test_spend_config_cpi_base_year_range(self):
        """Test CPI base year must be reasonable (2000-2030)."""
        config = SpendConfig(cpi_base_year=1990)
        with pytest.raises(ValueError):
            config.validate()

    def test_uncertainty_config_iterations_positive(self):
        """Test Monte Carlo iterations must be positive."""
        config = UncertaintyConfig(iterations=0)
        with pytest.raises(ValueError):
            config.validate()

    def test_provenance_config_invalid_hash_algorithm(self):
        """Test validation fails for unsupported hash algorithm."""
        config = ProvenanceConfig(hash_algorithm="md5")
        with pytest.raises(ValueError):
            config.validate()
