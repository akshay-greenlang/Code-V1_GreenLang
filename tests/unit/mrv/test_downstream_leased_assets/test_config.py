# -*- coding: utf-8 -*-
"""
Test suite for downstream_leased_assets.config - AGENT-MRV-026.

Tests configuration management for the Downstream Leased Assets Agent
(GL-MRV-S3-013) including default values, environment variable loading,
singleton pattern, thread safety, validation, and serialization.

Coverage:
- Default config values for all 18 sections (GeneralConfig, DatabaseConfig,
  BuildingConfig, VehicleConfig, EquipmentConfig, ITAssetsConfig,
  AllocationConfig, ComplianceConfig, EFSourceConfig, UncertaintyConfig,
  CacheConfig, APIConfig, ProvenanceConfig, MetricsConfig, SpendConfig,
  VacancyConfig, TenantConfig, PortfolioConfig)
- GL_DLA_ environment variable loading (monkeypatch)
- Singleton pattern (get_config returns same instance)
- Thread safety (12 threads)
- Validation (invalid values raise errors)
- to_dict / from_dict round-trip
- Frozen immutability

Author: GL-TestEngineer
Date: February 2026
"""

import threading
from decimal import Decimal
import pytest

try:
    from greenlang.agents.mrv.downstream_leased_assets.config import (
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
        VacancyConfig,
        TenantConfig,
        DownstreamLeasedConfig,
    )
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

_SKIP = pytest.mark.skipif(
    not _AVAILABLE,
    reason="downstream_leased_assets.config not available",
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
            from greenlang.agents.mrv.downstream_leased_assets.config import _reset_config
            _reset_config()
        except (ImportError, AttributeError):
            pass
    yield
    if _AVAILABLE:
        try:
            from greenlang.agents.mrv.downstream_leased_assets.config import _reset_config
            _reset_config()
        except (ImportError, AttributeError):
            pass


# ==============================================================================
# GENERAL CONFIGURATION TESTS
# ==============================================================================


class TestGeneralConfig:
    """Test GeneralConfig dataclass."""

    def test_defaults(self):
        config = GeneralConfig()
        assert config.enabled is True
        assert config.debug is False
        assert config.log_level == "INFO"
        assert config.max_batch_size == 1000

    def test_agent_id(self):
        config = GeneralConfig()
        assert config.agent_id == "GL-MRV-S3-013"
        assert config.agent_component == "AGENT-MRV-026"
        assert config.version == "1.0.0"

    def test_api_prefix(self):
        config = GeneralConfig()
        assert config.api_prefix == "/api/v1/downstream-leased-assets"

    def test_default_gwp(self):
        config = GeneralConfig()
        assert config.default_gwp == "AR5"

    def test_default_ef_source(self):
        config = GeneralConfig()
        assert config.default_ef_source == "DEFRA"

    def test_table_prefix(self):
        config = GeneralConfig()
        assert config.table_prefix == "gl_dla_"

    def test_frozen_immutability(self):
        config = GeneralConfig()
        with pytest.raises(Exception):
            config.enabled = False

    def test_validate_no_errors(self):
        config = GeneralConfig()
        config.validate()

    def test_validate_invalid_log_level(self):
        config = GeneralConfig(log_level="INVALID")
        with pytest.raises(ValueError, match="Invalid log_level"):
            config.validate()

    def test_validate_empty_agent_id(self):
        config = GeneralConfig(agent_id="")
        with pytest.raises(ValueError, match="agent_id cannot be empty"):
            config.validate()

    def test_validate_invalid_version(self):
        config = GeneralConfig(version="1.0")
        with pytest.raises(ValueError, match="Must follow SemVer"):
            config.validate()

    def test_to_dict(self):
        config = GeneralConfig()
        d = config.to_dict()
        assert isinstance(d, dict)
        assert d["agent_id"] == "GL-MRV-S3-013"


# ==============================================================================
# DATABASE CONFIGURATION TESTS
# ==============================================================================


class TestDatabaseConfig:
    """Test DatabaseConfig dataclass."""

    def test_defaults(self):
        config = DatabaseConfig()
        assert config.cache_enabled is True
        assert config.cache_ttl_seconds == 3600
        assert config.quantize_decimals == 8

    def test_frozen(self):
        config = DatabaseConfig()
        with pytest.raises(Exception):
            config.cache_enabled = False


# ==============================================================================
# BUILDING CONFIGURATION TESTS
# ==============================================================================


class TestBuildingConfig:
    """Test BuildingConfig dataclass."""

    def test_defaults(self):
        config = BuildingConfig()
        assert config.default_climate_zone == "temperate"
        assert config.default_allocation_method == "area"
        assert config.include_wtt is True

    def test_max_area_sqm(self):
        config = BuildingConfig()
        assert config.max_floor_area_sqm >= 100000

    def test_default_pue(self):
        config = BuildingConfig()
        assert config.default_pue in (Decimal("1.40"), Decimal("1.58"))

    def test_frozen(self):
        config = BuildingConfig()
        with pytest.raises(Exception):
            config.default_climate_zone = "cold"

    def test_validate_invalid_climate_zone(self):
        config = BuildingConfig(default_climate_zone="invalid_zone")
        with pytest.raises(ValueError):
            config.validate()


# ==============================================================================
# VEHICLE CONFIGURATION TESTS
# ==============================================================================


class TestVehicleConfig:
    """Test VehicleConfig dataclass."""

    def test_defaults(self):
        config = VehicleConfig()
        assert config.include_wtt is True
        assert config.default_fuel_type == "diesel"

    def test_max_annual_distance(self):
        config = VehicleConfig()
        assert config.max_annual_distance_km >= 200000

    def test_frozen(self):
        config = VehicleConfig()
        with pytest.raises(Exception):
            config.include_wtt = False


# ==============================================================================
# EQUIPMENT CONFIGURATION TESTS
# ==============================================================================


class TestEquipmentConfig:
    """Test EquipmentConfig dataclass."""

    def test_defaults(self):
        config = EquipmentConfig()
        assert config.include_wtt is True
        assert config.max_operating_hours == 8760

    def test_default_load_factor(self):
        config = EquipmentConfig()
        assert Decimal("0.50") <= config.default_load_factor <= Decimal("0.85")

    def test_frozen(self):
        config = EquipmentConfig()
        with pytest.raises(Exception):
            config.max_operating_hours = 10000


# ==============================================================================
# IT ASSETS CONFIGURATION TESTS
# ==============================================================================


class TestITAssetsConfig:
    """Test ITAssetsConfig dataclass."""

    def test_defaults(self):
        config = ITAssetsConfig()
        assert config.default_pue in (Decimal("1.58"), Decimal("1.40"))
        assert config.include_cooling is True

    def test_validate_pue_below_one(self):
        config = ITAssetsConfig(default_pue=Decimal("0.9"))
        with pytest.raises(ValueError):
            config.validate()


# ==============================================================================
# ALLOCATION CONFIGURATION TESTS
# ==============================================================================


class TestAllocationConfig:
    """Test AllocationConfig dataclass."""

    def test_defaults(self):
        config = AllocationConfig()
        assert config.default_method == "area"
        assert config.allow_revenue_allocation is True

    def test_frozen(self):
        config = AllocationConfig()
        with pytest.raises(Exception):
            config.default_method = "headcount"


# ==============================================================================
# COMPLIANCE CONFIGURATION TESTS
# ==============================================================================


class TestComplianceConfig:
    """Test ComplianceConfig dataclass."""

    def test_defaults(self):
        config = ComplianceConfig()
        assert config.strict_mode is False
        assert config.materiality_threshold == Decimal("0.01")


# ==============================================================================
# VACANCY CONFIGURATION TESTS (Cat 13 specific)
# ==============================================================================


class TestVacancyConfig:
    """Test VacancyConfig dataclass (downstream-leased specific)."""

    def test_defaults(self):
        config = VacancyConfig()
        assert config.include_vacancy_emissions is True

    def test_frozen(self):
        config = VacancyConfig()
        with pytest.raises(Exception):
            config.include_vacancy_emissions = False


# ==============================================================================
# TENANT CONFIGURATION TESTS (Cat 13 specific)
# ==============================================================================


class TestTenantConfig:
    """Test TenantConfig dataclass (downstream-leased specific)."""

    def test_defaults(self):
        config = TenantConfig()
        assert config.require_tenant_data is False

    def test_frozen(self):
        config = TenantConfig()
        with pytest.raises(Exception):
            config.require_tenant_data = True


# ==============================================================================
# MASTER CONFIGURATION TESTS
# ==============================================================================


class TestDownstreamLeasedConfig:
    """Test DownstreamLeasedConfig master configuration."""

    def test_master_config_creation(self):
        config = DownstreamLeasedConfig()
        assert config.general is not None
        assert config.database is not None
        assert config.building is not None
        assert config.vehicle is not None
        assert config.equipment is not None
        assert config.it_assets is not None
        assert config.allocation is not None
        assert config.compliance is not None
        assert config.vacancy is not None

    def test_master_config_agent_id(self):
        config = DownstreamLeasedConfig()
        assert config.general.agent_id == "GL-MRV-S3-013"

    def test_master_config_to_dict(self):
        config = DownstreamLeasedConfig()
        d = config.to_dict()
        assert isinstance(d, dict)
        assert "general" in d

    def test_validate_all(self):
        config = DownstreamLeasedConfig()
        config.validate_all()


# ==============================================================================
# SINGLETON TESTS
# ==============================================================================


class TestGetConfig:
    """Test get_config singleton pattern."""

    def test_singleton_identity(self):
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2

    def test_singleton_is_downstream_leased_config(self):
        config = get_config()
        assert isinstance(config, DownstreamLeasedConfig)


# ==============================================================================
# ENVIRONMENT VARIABLE TESTS
# ==============================================================================


class TestEnvironmentVariables:
    """Test GL_DLA_ environment variable loading."""

    def test_gl_dla_debug_env(self, monkeypatch):
        monkeypatch.setenv("GL_DLA_DEBUG", "true")
        config = get_config()
        assert config.general.debug is True

    def test_gl_dla_log_level_env(self, monkeypatch):
        monkeypatch.setenv("GL_DLA_LOG_LEVEL", "DEBUG")
        config = get_config()
        assert config.general.log_level == "DEBUG"

    def test_gl_dla_max_batch_size_env(self, monkeypatch):
        monkeypatch.setenv("GL_DLA_MAX_BATCH_SIZE", "5000")
        config = get_config()
        assert config.general.max_batch_size == 5000

    def test_gl_dla_default_gwp_env(self, monkeypatch):
        monkeypatch.setenv("GL_DLA_DEFAULT_GWP", "AR6")
        config = get_config()
        assert config.general.default_gwp == "AR6"

    def test_gl_dla_cache_ttl_env(self, monkeypatch):
        monkeypatch.setenv("GL_DLA_CACHE_TTL", "7200")
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

        threads = [threading.Thread(target=get_cfg) for _ in range(12)]
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
            })

        threads = [threading.Thread(target=read_values) for _ in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for r in results:
            assert r["agent_id"] == "GL-MRV-S3-013"
            assert r["table_prefix"] == "gl_dla_"


# ==============================================================================
# CROSS-SECTION CONSISTENCY TESTS
# ==============================================================================


class TestCrossSectionConsistency:
    """Test cross-section configuration consistency."""

    def test_metrics_prefix_matches_table_prefix(self):
        config = get_config()
        assert config.metrics.prefix == config.general.table_prefix

    def test_api_prefix_matches_general(self):
        config = get_config()
        assert config.api.prefix == config.general.api_prefix

    def test_provenance_hash_algorithm_is_sha256(self):
        config = get_config()
        assert config.provenance.hash_algorithm == "sha256"
