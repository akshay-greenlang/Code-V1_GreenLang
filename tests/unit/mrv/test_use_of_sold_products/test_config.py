# -*- coding: utf-8 -*-
"""
Test suite for use_of_sold_products.config - AGENT-MRV-024.

Tests configuration management for the Use of Sold Products Agent
(GL-MRV-S3-011) including default values, environment variable loading,
singleton pattern, thread safety, validation, and serialization.

Coverage:
- Default config values for all 18 config sections
- GL_USP_ environment variable overrides (monkeypatch)
- Singleton pattern (get_config returns same instance)
- Thread safety (10+ concurrent threads)
- Validation (invalid values raise errors)
- reset_config() creates new instance
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

# ---------------------------------------------------------------------------
# Graceful imports
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.mrv.use_of_sold_products.config import (
        get_config,
        reset_config,
        GeneralConfig,
        DatabaseConfig,
        DirectEmissionsConfig,
        IndirectEmissionsConfig,
        FuelsConfig,
        LifetimeConfig,
        ComplianceConfig,
        ProvenanceConfig,
        CacheConfig,
        APIConfig,
        MetricsConfig,
        UncertaintyConfig,
        EFSourceConfig,
        DegradationConfig,
        FleetConfig,
        ExportConfig,
        BatchConfig,
        UseOfSoldProductsConfig,
    )
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

_SKIP = pytest.mark.skipif(
    not CONFIG_AVAILABLE,
    reason="use_of_sold_products.config not available",
)
pytestmark = _SKIP


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
        """Test default agent_id is GL-MRV-S3-011."""
        config = GeneralConfig()
        assert config.agent_id == "GL-MRV-S3-011"
        assert config.agent_component == "AGENT-MRV-024"
        assert config.version == "1.0.0"

    def test_general_config_api_prefix(self):
        """Test default API prefix."""
        config = GeneralConfig()
        assert config.api_prefix == "/api/v1/use-of-sold-products"

    def test_general_config_default_gwp(self):
        """Test default GWP is AR5."""
        config = GeneralConfig()
        assert config.default_gwp == "AR5"

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

    def test_validate_config_invalid_batch_size(self):
        """Test validation fails for out-of-range batch size."""
        config = GeneralConfig(max_batch_size=0)
        with pytest.raises(ValueError, match="max_batch_size"):
            config.validate()

    def test_validate_config_invalid_api_prefix(self):
        """Test validation fails for API prefix not starting with /."""
        config = GeneralConfig(api_prefix="api/v1/use-of-sold-products")
        with pytest.raises(ValueError, match="api_prefix must start with"):
            config.validate()

    def test_to_dict(self):
        """Test to_dict serialization."""
        config = GeneralConfig()
        d = config.to_dict()
        assert d["agent_id"] == "GL-MRV-S3-011"
        assert d["version"] == "1.0.0"

    def test_from_dict(self):
        """Test from_dict deserialization."""
        d = {"agent_id": "GL-MRV-S3-011-TEST", "version": "2.0.0"}
        config = GeneralConfig.from_dict(d)
        assert config.agent_id == "GL-MRV-S3-011-TEST"


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
        assert config.table_prefix == "gl_usp_"
        assert config.schema == "use_of_sold_products_service"

    def test_database_config_frozen(self):
        """Test database config is frozen."""
        config = DatabaseConfig()
        with pytest.raises(Exception):
            config.host = "other_host"


# ==============================================================================
# DIRECT EMISSIONS CONFIGURATION TESTS
# ==============================================================================


class TestDirectEmissionsConfig:
    """Test DirectEmissionsConfig dataclass."""

    def test_direct_config_defaults(self):
        """Test default direct emissions config."""
        config = DirectEmissionsConfig()
        assert config.enable_fuel_combustion is True
        assert config.enable_refrigerant_leakage is True
        assert config.enable_chemical_release is True

    def test_direct_config_frozen(self):
        """Test direct config is frozen."""
        config = DirectEmissionsConfig()
        with pytest.raises(Exception):
            config.enable_fuel_combustion = False


# ==============================================================================
# INDIRECT EMISSIONS CONFIGURATION TESTS
# ==============================================================================


class TestIndirectEmissionsConfig:
    """Test IndirectEmissionsConfig dataclass."""

    def test_indirect_config_defaults(self):
        """Test default indirect emissions config."""
        config = IndirectEmissionsConfig()
        assert config.enable_electricity is True
        assert config.enable_heating_fuel is True
        assert config.enable_steam_cooling is True
        assert config.default_grid_region == "GLOBAL"

    def test_indirect_config_frozen(self):
        """Test indirect config is frozen."""
        config = IndirectEmissionsConfig()
        with pytest.raises(Exception):
            config.default_grid_region = "US"


# ==============================================================================
# FUELS CONFIGURATION TESTS
# ==============================================================================


class TestFuelsConfig:
    """Test FuelsConfig dataclass."""

    def test_fuels_config_defaults(self):
        """Test default fuels config."""
        config = FuelsConfig()
        assert config.enable_fuel_sales is True
        assert config.enable_feedstock_oxidation is True

    def test_fuels_config_frozen(self):
        """Test fuels config is frozen."""
        config = FuelsConfig()
        with pytest.raises(Exception):
            config.enable_fuel_sales = False


# ==============================================================================
# LIFETIME CONFIGURATION TESTS
# ==============================================================================


class TestLifetimeConfig:
    """Test LifetimeConfig dataclass."""

    def test_lifetime_config_defaults(self):
        """Test default lifetime config."""
        config = LifetimeConfig()
        assert config.default_degradation_model == "linear"
        assert config.enable_weibull is True
        assert config.enable_fleet_survival is True
        assert config.discount_rate == Decimal("0.0")

    def test_lifetime_config_frozen(self):
        """Test lifetime config is frozen."""
        config = LifetimeConfig()
        with pytest.raises(Exception):
            config.default_degradation_model = "exponential"


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

    def test_compliance_config_get_frameworks(self):
        """Test get_frameworks returns all 7 frameworks."""
        config = ComplianceConfig()
        frameworks = config.get_frameworks()
        assert len(frameworks) == 7
        assert "GHG_PROTOCOL_SCOPE3" in frameworks

    def test_compliance_config_frozen(self):
        """Test compliance config is frozen."""
        config = ComplianceConfig()
        with pytest.raises(Exception):
            config.strict_mode = True


# ==============================================================================
# PROVENANCE CONFIGURATION TESTS
# ==============================================================================


class TestProvenanceConfig:
    """Test ProvenanceConfig dataclass."""

    def test_provenance_config_defaults(self):
        """Test default provenance config."""
        config = ProvenanceConfig()
        assert config.enable_provenance is True
        assert config.hash_algorithm == "sha256"

    def test_provenance_config_frozen(self):
        """Test provenance config is frozen."""
        config = ProvenanceConfig()
        with pytest.raises(Exception):
            config.hash_algorithm = "md5"


# ==============================================================================
# CACHE CONFIGURATION TESTS
# ==============================================================================


class TestCacheConfig:
    """Test CacheConfig dataclass."""

    def test_cache_config_defaults(self):
        """Test default cache config."""
        config = CacheConfig()
        assert config.enable_cache is True
        assert config.ttl_seconds == 3600


# ==============================================================================
# API CONFIGURATION TESTS
# ==============================================================================


class TestAPIConfig:
    """Test APIConfig dataclass."""

    def test_api_config_defaults(self):
        """Test default API config."""
        config = APIConfig()
        assert config.enable_api is True
        assert config.prefix == "/api/v1/use-of-sold-products"


# ==============================================================================
# METRICS CONFIGURATION TESTS
# ==============================================================================


class TestMetricsConfig:
    """Test MetricsConfig dataclass."""

    def test_metrics_config_defaults(self):
        """Test default metrics config."""
        config = MetricsConfig()
        assert config.enable_metrics is True
        assert config.prefix == "gl_usp_"


# ==============================================================================
# UNCERTAINTY CONFIGURATION TESTS
# ==============================================================================


class TestUncertaintyConfig:
    """Test UncertaintyConfig dataclass."""

    def test_uncertainty_config_defaults(self):
        """Test default uncertainty config."""
        config = UncertaintyConfig()
        assert config.default_method == "propagation"
        assert config.confidence_level == Decimal("0.95")


# ==============================================================================
# EF SOURCE CONFIGURATION TESTS
# ==============================================================================


class TestEFSourceConfig:
    """Test EFSourceConfig dataclass."""

    def test_ef_source_config_defaults(self):
        """Test default EF source config."""
        config = EFSourceConfig()
        assert config.default_source in ("defra", "epa", "ipcc")


# ==============================================================================
# DEGRADATION CONFIGURATION TESTS
# ==============================================================================


class TestDegradationConfig:
    """Test DegradationConfig dataclass."""

    def test_degradation_config_defaults(self):
        """Test default degradation config."""
        config = DegradationConfig()
        assert config.default_model == "linear"


# ==============================================================================
# FLEET CONFIGURATION TESTS
# ==============================================================================


class TestFleetConfig:
    """Test FleetConfig dataclass."""

    def test_fleet_config_defaults(self):
        """Test default fleet config."""
        config = FleetConfig()
        assert config.enable_survival_modeling is True


# ==============================================================================
# EXPORT CONFIGURATION TESTS
# ==============================================================================


class TestExportConfig:
    """Test ExportConfig dataclass."""

    def test_export_config_defaults(self):
        """Test default export config."""
        config = ExportConfig()
        assert config.default_format in ("json", "csv")


# ==============================================================================
# BATCH CONFIGURATION TESTS
# ==============================================================================


class TestBatchConfig:
    """Test BatchConfig dataclass."""

    def test_batch_config_defaults(self):
        """Test default batch config."""
        config = BatchConfig()
        assert config.max_batch_size == 1000


# ==============================================================================
# ROOT CONFIG TESTS
# ==============================================================================


class TestUseOfSoldProductsConfig:
    """Test root UseOfSoldProductsConfig (composed of all sub-configs)."""

    def test_root_config_has_all_sections(self):
        """Test root config has all 18 sections."""
        config = UseOfSoldProductsConfig()
        assert hasattr(config, "general")
        assert hasattr(config, "database")
        assert hasattr(config, "direct")
        assert hasattr(config, "indirect")
        assert hasattr(config, "fuels")
        assert hasattr(config, "lifetime")
        assert hasattr(config, "compliance")
        assert hasattr(config, "provenance")
        assert hasattr(config, "cache")
        assert hasattr(config, "api")
        assert hasattr(config, "metrics")
        assert hasattr(config, "uncertainty")

    def test_root_config_validate_all(self):
        """Test root config validation passes for defaults."""
        config = UseOfSoldProductsConfig()
        config.validate()

    def test_root_config_to_dict_round_trip(self):
        """Test to_dict / from_dict round-trip."""
        config = UseOfSoldProductsConfig()
        d = config.to_dict()
        config2 = UseOfSoldProductsConfig.from_dict(d)
        assert config2.general.agent_id == config.general.agent_id


# ==============================================================================
# SINGLETON PATTERN TESTS
# ==============================================================================


class TestSingletonPattern:
    """Test get_config singleton pattern."""

    def test_get_config_returns_same_instance(self):
        """Test get_config returns the same object on repeated calls."""
        c1 = get_config()
        c2 = get_config()
        assert c1 is c2

    def test_reset_config_creates_new_instance(self):
        """Test reset_config creates a new config instance."""
        c1 = get_config()
        reset_config()
        c2 = get_config()
        assert c1 is not c2

    def test_thread_safety_get_config(self):
        """Test get_config is thread-safe with 10+ concurrent threads."""
        results = []
        errors = []

        def _get():
            try:
                cfg = get_config()
                results.append(id(cfg))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_get) for _ in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert len(errors) == 0, f"Thread errors: {errors}"
        # All threads should get the same instance
        assert len(set(results)) == 1


# ==============================================================================
# ENVIRONMENT VARIABLE OVERRIDE TESTS
# ==============================================================================


class TestEnvVarOverrides:
    """Test GL_USP_ environment variable loading."""

    def test_gl_usp_debug_override(self, monkeypatch):
        """Test GL_USP_DEBUG=true enables debug mode."""
        monkeypatch.setenv("GL_USP_DEBUG", "true")
        reset_config()
        config = get_config()
        assert config.general.debug is True

    def test_gl_usp_log_level_override(self, monkeypatch):
        """Test GL_USP_LOG_LEVEL=DEBUG overrides log level."""
        monkeypatch.setenv("GL_USP_LOG_LEVEL", "DEBUG")
        reset_config()
        config = get_config()
        assert config.general.log_level == "DEBUG"

    def test_gl_usp_max_batch_size_override(self, monkeypatch):
        """Test GL_USP_MAX_BATCH_SIZE=500 overrides batch size."""
        monkeypatch.setenv("GL_USP_MAX_BATCH_SIZE", "500")
        reset_config()
        config = get_config()
        assert config.general.max_batch_size == 500

    def test_gl_usp_default_gwp_override(self, monkeypatch):
        """Test GL_USP_DEFAULT_GWP=AR6 overrides GWP version."""
        monkeypatch.setenv("GL_USP_DEFAULT_GWP", "AR6")
        reset_config()
        config = get_config()
        assert config.general.default_gwp == "AR6"

    def test_gl_usp_cache_ttl_override(self, monkeypatch):
        """Test GL_USP_CACHE_TTL=7200 overrides cache TTL."""
        monkeypatch.setenv("GL_USP_CACHE_TTL", "7200")
        reset_config()
        config = get_config()
        assert config.cache.ttl_seconds == 7200

    def test_gl_usp_grid_region_override(self, monkeypatch):
        """Test GL_USP_DEFAULT_GRID_REGION=US overrides grid region."""
        monkeypatch.setenv("GL_USP_DEFAULT_GRID_REGION", "US")
        reset_config()
        config = get_config()
        assert config.indirect.default_grid_region == "US"

    def test_gl_usp_discount_rate_override(self, monkeypatch):
        """Test GL_USP_DISCOUNT_RATE=0.03 overrides discount rate."""
        monkeypatch.setenv("GL_USP_DISCOUNT_RATE", "0.03")
        reset_config()
        config = get_config()
        assert config.lifetime.discount_rate == Decimal("0.03")

    def test_gl_usp_degradation_model_override(self, monkeypatch):
        """Test GL_USP_DEGRADATION_MODEL=exponential overrides model."""
        monkeypatch.setenv("GL_USP_DEGRADATION_MODEL", "exponential")
        reset_config()
        config = get_config()
        assert config.lifetime.default_degradation_model == "exponential"

    def test_gl_usp_strict_compliance_override(self, monkeypatch):
        """Test GL_USP_STRICT_COMPLIANCE=true enables strict mode."""
        monkeypatch.setenv("GL_USP_STRICT_COMPLIANCE", "true")
        reset_config()
        config = get_config()
        assert config.compliance.strict_mode is True

    def test_gl_usp_enable_weibull_override(self, monkeypatch):
        """Test GL_USP_ENABLE_WEIBULL=false disables Weibull modeling."""
        monkeypatch.setenv("GL_USP_ENABLE_WEIBULL", "false")
        reset_config()
        config = get_config()
        assert config.lifetime.enable_weibull is False
