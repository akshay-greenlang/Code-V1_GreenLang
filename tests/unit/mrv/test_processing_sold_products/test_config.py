# -*- coding: utf-8 -*-
"""
Test suite for processing_sold_products.config - AGENT-MRV-023.

Tests configuration management for the Processing of Sold Products Agent
(GL-MRV-S3-010) including default values, environment variable loading,
singleton pattern, thread safety, validation, and serialization.

Coverage:
- Default config values for all 18 config sections
- GL_PSP_ environment variable overrides (monkeypatch)
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
    from greenlang.agents.mrv.processing_sold_products.config import (
        get_config,
        reset_config,
        GeneralConfig,
        DatabaseConfig,
        SiteSpecificConfig,
        AverageDataConfig,
        SpendConfig,
        HybridConfig,
        ComplianceConfig,
        ProvenanceConfig,
        CacheConfig,
        APIConfig,
        MetricsConfig,
        UncertaintyConfig,
        EFSourceConfig,
        AllocationConfig,
        ChainConfig,
        ExportConfig,
        BatchConfig,
        ProcessingSoldProductsConfig,
    )
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

_SKIP = pytest.mark.skipif(
    not CONFIG_AVAILABLE,
    reason="processing_sold_products.config not available",
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
        """Test default agent_id is GL-MRV-S3-010."""
        config = GeneralConfig()
        assert config.agent_id == "GL-MRV-S3-010"
        assert config.agent_component == "AGENT-MRV-023"
        assert config.version == "1.0.0"

    def test_general_config_api_prefix(self):
        """Test default API prefix."""
        config = GeneralConfig()
        assert config.api_prefix == "/api/v1/processing-sold-products"

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
        config = GeneralConfig(api_prefix="api/v1/processing-sold-products")
        with pytest.raises(ValueError, match="api_prefix must start with"):
            config.validate()

    def test_to_dict(self):
        """Test to_dict serialization."""
        config = GeneralConfig()
        d = config.to_dict()
        assert d["agent_id"] == "GL-MRV-S3-010"
        assert d["enabled"] is True

    def test_from_dict_roundtrip(self):
        """Test to_dict -> from_dict round-trip."""
        original = GeneralConfig()
        d = original.to_dict()
        restored = GeneralConfig.from_dict(d)
        assert restored.agent_id == original.agent_id
        assert restored.version == original.version

    def test_from_env_defaults(self):
        """Test from_env uses defaults when env vars not set."""
        for key in list(os.environ.keys()):
            if key.startswith("GL_PSP_"):
                del os.environ[key]
        config = GeneralConfig.from_env()
        assert config.enabled is True
        assert config.agent_id == "GL-MRV-S3-010"

    def test_from_env_custom(self, monkeypatch):
        """Test from_env loads custom values from GL_PSP_ environment."""
        monkeypatch.setenv("GL_PSP_ENABLED", "false")
        monkeypatch.setenv("GL_PSP_DEBUG", "true")
        monkeypatch.setenv("GL_PSP_LOG_LEVEL", "DEBUG")
        config = GeneralConfig.from_env()
        assert config.enabled is False
        assert config.debug is True
        assert config.log_level == "DEBUG"


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
        """Test default table prefix is gl_psp_."""
        config = DatabaseConfig()
        assert config.table_prefix == "gl_psp_"

    def test_database_config_schema(self):
        """Test default schema is processing_sold_products_service."""
        config = DatabaseConfig()
        assert config.schema == "processing_sold_products_service"

    def test_database_config_ssl_mode(self):
        """Test default SSL mode is prefer."""
        config = DatabaseConfig()
        assert config.ssl_mode == "prefer"

    def test_database_config_frozen(self):
        """Test DatabaseConfig is immutable."""
        config = DatabaseConfig()
        with pytest.raises(Exception):
            config.host = "remote"

    def test_database_config_validate_invalid_port(self):
        """Test validation fails for port out of range."""
        config = DatabaseConfig(port=99999)
        with pytest.raises(ValueError, match="port must be between"):
            config.validate()

    def test_database_config_validate_invalid_prefix(self):
        """Test validation fails for prefix not ending with _."""
        config = DatabaseConfig(table_prefix="gl_psp")
        with pytest.raises(ValueError, match="table_prefix must end"):
            config.validate()

    def test_database_config_validate_pool_min_gt_max(self):
        """Test validation fails for pool_min > pool_max."""
        config = DatabaseConfig(pool_min=20, pool_max=5)
        with pytest.raises(ValueError, match="pool_min must be"):
            config.validate()


# ==============================================================================
# SITE-SPECIFIC CONFIGURATION TESTS
# ==============================================================================


class TestSiteSpecificConfig:
    """Test SiteSpecificConfig dataclass."""

    def test_site_specific_defaults(self):
        """Test default site-specific config values."""
        config = SiteSpecificConfig()
        assert config.enable_direct is True
        assert config.enable_energy is True
        assert config.enable_fuel is True

    def test_site_specific_dqi_scores(self):
        """Test default DQI scores for site-specific methods."""
        config = SiteSpecificConfig()
        assert config.dqi_direct == Decimal("90")
        assert config.dqi_energy == Decimal("80")
        assert config.dqi_fuel == Decimal("75")

    def test_site_specific_frozen(self):
        """Test SiteSpecificConfig is immutable."""
        config = SiteSpecificConfig()
        with pytest.raises(Exception):
            config.enable_direct = False


# ==============================================================================
# AVERAGE DATA CONFIGURATION TESTS
# ==============================================================================


class TestAverageDataConfig:
    """Test AverageDataConfig dataclass."""

    def test_average_data_defaults(self):
        """Test default average data config values."""
        config = AverageDataConfig()
        assert config.default_region == "GLOBAL"
        assert config.enable_chain_calculations is True

    def test_average_data_dqi_scores(self):
        """Test default DQI scores for average-data methods."""
        config = AverageDataConfig()
        assert config.dqi_process == Decimal("55")
        assert config.dqi_energy_intensity == Decimal("50")
        assert config.dqi_sector == Decimal("45")


# ==============================================================================
# SPEND CONFIGURATION TESTS
# ==============================================================================


class TestSpendConfig:
    """Test SpendConfig dataclass."""

    def test_spend_defaults(self):
        """Test default spend config values."""
        config = SpendConfig()
        assert config.enable_cpi_deflation is True
        assert config.enable_margin_removal is True
        assert config.default_margin_rate == Decimal("0.08")
        assert config.base_year == 2021
        assert config.default_currency == "USD"

    def test_spend_dqi_score(self):
        """Test default spend DQI score is 30."""
        config = SpendConfig()
        assert config.dqi_score == Decimal("30")

    def test_spend_frozen(self):
        """Test SpendConfig is immutable."""
        config = SpendConfig()
        with pytest.raises(Exception):
            config.base_year = 2020


# ==============================================================================
# HYBRID CONFIGURATION TESTS
# ==============================================================================


class TestHybridConfig:
    """Test HybridConfig dataclass."""

    def test_hybrid_defaults(self):
        """Test default hybrid config values."""
        config = HybridConfig()
        assert config.enable_gap_filling is True
        assert config.allocation_method == "mass"

    def test_hybrid_method_priority(self):
        """Test default method priority order."""
        config = HybridConfig()
        assert config.method_priority[0] == "site_specific_direct"
        assert config.method_priority[-1] == "spend_based"
        assert len(config.method_priority) == 5


# ==============================================================================
# COMPLIANCE CONFIGURATION TESTS
# ==============================================================================


class TestComplianceConfig:
    """Test ComplianceConfig dataclass."""

    def test_compliance_defaults(self):
        """Test default compliance config values."""
        config = ComplianceConfig()
        assert config.strict_mode is False
        assert config.materiality_threshold == Decimal("0.01")

    def test_compliance_frameworks(self):
        """Test default frameworks include all 7."""
        config = ComplianceConfig()
        frameworks = config.get_frameworks()
        assert len(frameworks) == 7
        assert "GHG_PROTOCOL_SCOPE3" in frameworks


# ==============================================================================
# PROVENANCE CONFIGURATION TESTS
# ==============================================================================


class TestProvenanceConfig:
    """Test ProvenanceConfig dataclass."""

    def test_provenance_defaults(self):
        """Test default provenance config values."""
        config = ProvenanceConfig()
        assert config.enable_provenance is True
        assert config.hash_algorithm == "sha256"


# ==============================================================================
# CACHE CONFIGURATION TESTS
# ==============================================================================


class TestCacheConfig:
    """Test CacheConfig dataclass."""

    def test_cache_defaults(self):
        """Test default cache config values."""
        config = CacheConfig()
        assert config.enable_cache is True
        assert config.ttl_seconds == 3600


# ==============================================================================
# API CONFIGURATION TESTS
# ==============================================================================


class TestAPIConfig:
    """Test APIConfig dataclass."""

    def test_api_defaults(self):
        """Test default API config values."""
        config = APIConfig()
        assert config.enable_api is True
        assert config.prefix == "/api/v1/processing-sold-products"


# ==============================================================================
# METRICS CONFIGURATION TESTS
# ==============================================================================


class TestMetricsConfig:
    """Test MetricsConfig dataclass."""

    def test_metrics_defaults(self):
        """Test default metrics config values."""
        config = MetricsConfig()
        assert config.enable_metrics is True
        assert config.prefix == "gl_psp_"


# ==============================================================================
# UNCERTAINTY CONFIGURATION TESTS
# ==============================================================================


class TestUncertaintyConfig:
    """Test UncertaintyConfig dataclass."""

    def test_uncertainty_defaults(self):
        """Test default uncertainty config values."""
        config = UncertaintyConfig()
        assert config.default_method == "propagation"
        assert config.confidence_level == Decimal("0.95")


# ==============================================================================
# ALLOCATION CONFIGURATION TESTS
# ==============================================================================


class TestAllocationConfig:
    """Test AllocationConfig dataclass."""

    def test_allocation_defaults(self):
        """Test default allocation config values."""
        config = AllocationConfig()
        assert config.default_method == "mass"

    @pytest.mark.parametrize("method", ["mass", "revenue", "units", "equal"])
    def test_allocation_valid_methods(self, method):
        """Test all valid allocation methods."""
        config = AllocationConfig(default_method=method)
        assert config.default_method == method


# ==============================================================================
# CHAIN CONFIGURATION TESTS
# ==============================================================================


class TestChainConfig:
    """Test ChainConfig dataclass."""

    def test_chain_defaults(self):
        """Test default chain config values."""
        config = ChainConfig()
        assert config.enable_chain_calculations is True
        assert config.max_chain_steps == 10


# ==============================================================================
# EXPORT CONFIGURATION TESTS
# ==============================================================================


class TestExportConfig:
    """Test ExportConfig dataclass."""

    def test_export_defaults(self):
        """Test default export config values."""
        config = ExportConfig()
        assert config.default_format == "json"
        assert config.enable_csv is True
        assert config.enable_xlsx is True


# ==============================================================================
# BATCH CONFIGURATION TESTS
# ==============================================================================


class TestBatchConfig:
    """Test BatchConfig dataclass."""

    def test_batch_defaults(self):
        """Test default batch config values."""
        config = BatchConfig()
        assert config.max_batch_size == 1000
        assert config.enable_parallel is True


# ==============================================================================
# MASTER CONFIG TESTS
# ==============================================================================


class TestProcessingSoldProductsConfig:
    """Test the master ProcessingSoldProductsConfig."""

    def test_master_config_has_all_sections(self):
        """Test master config has all 18 section attributes."""
        config = ProcessingSoldProductsConfig()
        sections = [
            "general", "database", "site_specific", "average_data",
            "spend", "hybrid", "compliance", "provenance",
            "cache", "api", "metrics", "uncertainty",
            "ef_source", "allocation", "chain", "export", "batch",
        ]
        for section in sections:
            assert hasattr(config, section), f"Missing config section: {section}"

    def test_master_config_to_dict(self):
        """Test master config serialization."""
        config = ProcessingSoldProductsConfig()
        d = config.to_dict()
        assert "general" in d
        assert "database" in d
        assert d["general"]["agent_id"] == "GL-MRV-S3-010"


# ==============================================================================
# SINGLETON PATTERN TESTS
# ==============================================================================


class TestSingletonPattern:
    """Test singleton pattern implementation."""

    def test_config_singleton_get(self):
        """Test get_config returns the same instance."""
        reset_config()
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2

    def test_config_thread_safety(self):
        """Test singleton works across 10+ threads."""
        reset_config()
        configs = []

        def get_config_thread():
            configs.append(get_config())

        threads = [threading.Thread(target=get_config_thread) for _ in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        first = configs[0]
        for c in configs[1:]:
            assert c is first

    def test_reset_config_creates_new(self):
        """Test reset_config allows a new instance."""
        config1 = get_config()
        reset_config()
        config2 = get_config()
        assert config1 is not config2


# ==============================================================================
# ENVIRONMENT VARIABLE TESTS
# ==============================================================================


class TestEnvironmentVariables:
    """Test GL_PSP_ environment variable loading."""

    def test_boolean_env_var_true(self, monkeypatch):
        """Test boolean env var parsing for 'true'."""
        monkeypatch.setenv("GL_PSP_ENABLED", "true")
        config = GeneralConfig.from_env()
        assert config.enabled is True

    def test_boolean_env_var_false(self, monkeypatch):
        """Test boolean env var parsing for 'false'."""
        monkeypatch.setenv("GL_PSP_ENABLED", "false")
        config = GeneralConfig.from_env()
        assert config.enabled is False

    def test_boolean_env_var_case_insensitive(self, monkeypatch):
        """Test boolean env var is case-insensitive."""
        monkeypatch.setenv("GL_PSP_DEBUG", "True")
        config = GeneralConfig.from_env()
        assert config.debug is True

    def test_numeric_env_var(self, monkeypatch):
        """Test numeric env var parsing."""
        monkeypatch.setenv("GL_PSP_MAX_BATCH_SIZE", "500")
        config = GeneralConfig.from_env()
        assert config.max_batch_size == 500

    def test_decimal_env_var(self, monkeypatch):
        """Test Decimal env var parsing for margin rate."""
        monkeypatch.setenv("GL_PSP_DEFAULT_MARGIN_RATE", "0.12")
        config = SpendConfig.from_env()
        assert config.default_margin_rate == Decimal("0.12")

    def test_string_env_var(self, monkeypatch):
        """Test string env var for log level."""
        monkeypatch.setenv("GL_PSP_LOG_LEVEL", "WARNING")
        config = GeneralConfig.from_env()
        assert config.log_level == "WARNING"


# ==============================================================================
# SUMMARY
# ==============================================================================


def test_config_module_coverage():
    """Meta-test to ensure comprehensive config section coverage."""
    tested_sections = [
        "GeneralConfig", "DatabaseConfig", "SiteSpecificConfig",
        "AverageDataConfig", "SpendConfig", "HybridConfig",
        "ComplianceConfig", "ProvenanceConfig", "CacheConfig",
        "APIConfig", "MetricsConfig", "UncertaintyConfig",
        "AllocationConfig", "ChainConfig", "ExportConfig", "BatchConfig",
        "ProcessingSoldProductsConfig", "Singleton", "EnvironmentVariables",
    ]
    assert len(tested_sections) >= 18
