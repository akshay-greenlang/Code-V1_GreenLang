# -*- coding: utf-8 -*-
"""
Test suite for investments.config - AGENT-MRV-028.

Tests configuration management for the Investments Agent (GL-MRV-S3-015)
including default values, environment variable loading, singleton pattern,
thread safety, validation, and serialization.

Coverage:
- Default config values for GeneralConfig, DatabaseConfig, EquityConfig,
  DebtConfig, RealAssetsConfig, SovereignConfig, ComplianceConfig,
  PCAFConfig, AttributionConfig, CacheConfig, APIConfig,
  ProvenanceConfig, MetricsConfig, UncertaintyConfig,
  PortfolioConfig, CurrencyConfig, ReportingConfig, BatchConfig
- GL_INV_ environment variable loading (monkeypatch)
- Singleton pattern (get_config returns same instance)
- Thread safety
- Validation (invalid values raise errors)
- to_dict / from_dict round-trip
- Frozen immutability
- _reset_config()

Author: GL-TestEngineer
Date: February 2026
"""

import os
import threading
from decimal import Decimal
from typing import Any, Dict
import pytest

from greenlang.investments.config import (
    get_config,
    GeneralConfig,
    DatabaseConfig,
    EquityConfig,
    DebtConfig,
    RealAssetsConfig,
    SovereignConfig,
    ComplianceConfig,
    PCAFConfig,
    AttributionConfig,
    CacheConfig,
    APIConfig,
    ProvenanceConfig,
    MetricsConfig,
    UncertaintyConfig,
    PortfolioConfig,
    CurrencyConfig,
    ReportingConfig,
    BatchConfig,
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
        """Test default agent_id is GL-MRV-S3-015."""
        config = GeneralConfig()
        assert config.agent_id == "GL-MRV-S3-015"
        assert config.agent_component == "AGENT-MRV-028"
        assert config.version == "1.0.0"

    def test_general_config_api_prefix(self):
        """Test default API prefix."""
        config = GeneralConfig()
        assert config.api_prefix == "/api/v1/investments"

    def test_general_config_default_gwp(self):
        """Test default GWP is AR5."""
        config = GeneralConfig()
        assert config.default_gwp == "AR5"

    def test_general_config_default_attribution(self):
        """Test default attribution method is EVIC."""
        config = GeneralConfig()
        assert config.default_attribution_method == "EVIC"

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

    def test_validate_config_invalid_gwp(self):
        """Test validation fails for invalid GWP version."""
        config = GeneralConfig(default_gwp="AR99")
        with pytest.raises(ValueError):
            config.validate()

    def test_validate_config_invalid_attribution(self):
        """Test validation fails for invalid attribution method."""
        config = GeneralConfig(default_attribution_method="INVALID")
        with pytest.raises(ValueError):
            config.validate()

    def test_validate_config_empty_agent_id(self):
        """Test validation fails for empty agent_id."""
        config = GeneralConfig(agent_id="")
        with pytest.raises(ValueError, match="agent_id"):
            config.validate()

    def test_validate_config_invalid_api_prefix(self):
        """Test validation fails for API prefix not starting with /."""
        config = GeneralConfig(api_prefix="api/v1/investments")
        with pytest.raises(ValueError, match="api_prefix"):
            config.validate()

    def test_validate_config_invalid_batch_size(self):
        """Test validation fails for batch size out of range."""
        config = GeneralConfig(max_batch_size=0)
        with pytest.raises(ValueError, match="max_batch_size"):
            config.validate()

    def test_to_dict(self):
        """Test conversion to dict."""
        config = GeneralConfig()
        d = config.to_dict()
        assert d["agent_id"] == "GL-MRV-S3-015"
        assert d["enabled"] is True

    def test_from_dict_round_trip(self):
        """Test from_dict creates equivalent config."""
        config = GeneralConfig()
        d = config.to_dict()
        config2 = GeneralConfig.from_dict(d)
        assert config2.agent_id == config.agent_id

    def test_from_env(self, monkeypatch):
        """Test loading from GL_INV_ environment variables."""
        monkeypatch.setenv("GL_INV_ENABLED", "false")
        monkeypatch.setenv("GL_INV_DEBUG", "true")
        monkeypatch.setenv("GL_INV_LOG_LEVEL", "DEBUG")
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
        assert config.pool_size == 5
        assert config.max_overflow == 10

    def test_database_config_frozen(self):
        """Test DatabaseConfig is immutable."""
        config = DatabaseConfig()
        with pytest.raises(Exception):
            config.pool_size = 20


# ==============================================================================
# EQUITY CONFIGURATION TESTS
# ==============================================================================


class TestEquityConfig:
    """Test EquityConfig dataclass."""

    def test_equity_config_defaults(self):
        """Test default equity config values."""
        config = EquityConfig()
        assert config.default_evic_source == "BLOOMBERG"
        assert config.include_scope3 is False

    def test_equity_config_frozen(self):
        """Test EquityConfig is immutable."""
        config = EquityConfig()
        with pytest.raises(Exception):
            config.include_scope3 = True

    def test_equity_config_from_env(self, monkeypatch):
        """Test loading equity config from env."""
        monkeypatch.setenv("GL_INV_EQUITY_INCLUDE_SCOPE3", "true")
        config = EquityConfig.from_env()
        assert config.include_scope3 is True


# ==============================================================================
# DEBT CONFIGURATION TESTS
# ==============================================================================


class TestDebtConfig:
    """Test DebtConfig dataclass."""

    def test_debt_config_defaults(self):
        """Test default debt config values."""
        config = DebtConfig()
        assert config.green_bond_discount == Decimal("0.0")

    def test_debt_config_frozen(self):
        """Test DebtConfig is immutable."""
        config = DebtConfig()
        with pytest.raises(Exception):
            config.green_bond_discount = Decimal("0.5")


# ==============================================================================
# REAL ASSETS CONFIGURATION TESTS
# ==============================================================================


class TestRealAssetsConfig:
    """Test RealAssetsConfig dataclass."""

    def test_real_assets_config_defaults(self):
        """Test default real assets config values."""
        config = RealAssetsConfig()
        assert config.default_eui_source == "CRREM"

    def test_real_assets_config_frozen(self):
        """Test RealAssetsConfig is immutable."""
        config = RealAssetsConfig()
        with pytest.raises(Exception):
            config.default_eui_source = "CUSTOM"


# ==============================================================================
# SOVEREIGN CONFIGURATION TESTS
# ==============================================================================


class TestSovereignConfig:
    """Test SovereignConfig dataclass."""

    def test_sovereign_config_defaults(self):
        """Test default sovereign config values."""
        config = SovereignConfig()
        assert config.include_lulucf is False

    def test_sovereign_config_frozen(self):
        """Test SovereignConfig is immutable."""
        config = SovereignConfig()
        with pytest.raises(Exception):
            config.include_lulucf = True


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
        """Test get_frameworks returns all 9 frameworks."""
        config = ComplianceConfig()
        frameworks = config.get_frameworks()
        assert len(frameworks) >= 9

    def test_compliance_config_frozen(self):
        """Test ComplianceConfig is immutable."""
        config = ComplianceConfig()
        with pytest.raises(Exception):
            config.strict_mode = True


# ==============================================================================
# PCAF CONFIGURATION TESTS
# ==============================================================================


class TestPCAFConfig:
    """Test PCAFConfig dataclass."""

    def test_pcaf_config_defaults(self):
        """Test default PCAF config values."""
        config = PCAFConfig()
        assert config.default_quality_score == 5
        assert config.require_quality_reporting is True

    def test_pcaf_config_frozen(self):
        """Test PCAFConfig is immutable."""
        config = PCAFConfig()
        with pytest.raises(Exception):
            config.default_quality_score = 1


# ==============================================================================
# ATTRIBUTION CONFIGURATION TESTS
# ==============================================================================


class TestAttributionConfig:
    """Test AttributionConfig dataclass."""

    def test_attribution_config_defaults(self):
        """Test default attribution config values."""
        config = AttributionConfig()
        assert config.default_method == "EVIC"

    def test_attribution_config_frozen(self):
        """Test AttributionConfig is immutable."""
        config = AttributionConfig()
        with pytest.raises(Exception):
            config.default_method = "EQUITY_SHARE"


# ==============================================================================
# REMAINING CONFIGURATION SECTION TESTS
# ==============================================================================


class TestCacheConfig:
    """Test CacheConfig dataclass."""

    def test_cache_config_defaults(self):
        """Test default cache config values."""
        config = CacheConfig()
        assert config.ttl_seconds == 3600


class TestAPIConfig:
    """Test APIConfig dataclass."""

    def test_api_config_defaults(self):
        """Test default API config values."""
        config = APIConfig()
        assert config.rate_limit_per_minute == 60


class TestProvenanceConfig:
    """Test ProvenanceConfig dataclass."""

    def test_provenance_config_defaults(self):
        """Test default provenance config values."""
        config = ProvenanceConfig()
        assert config.chain_algorithm == "SHA-256"
        assert config.merkle_enabled is True


class TestMetricsConfig:
    """Test MetricsConfig dataclass."""

    def test_metrics_config_defaults(self):
        """Test default metrics config values."""
        config = MetricsConfig()
        assert config.prefix == "gl_inv_"


class TestUncertaintyConfig:
    """Test UncertaintyConfig dataclass."""

    def test_uncertainty_config_defaults(self):
        """Test default uncertainty config values."""
        config = UncertaintyConfig()
        assert config.default_method == "MONTE_CARLO"
        assert config.iterations == 10000


class TestPortfolioConfig:
    """Test PortfolioConfig dataclass."""

    def test_portfolio_config_defaults(self):
        """Test default portfolio config values."""
        config = PortfolioConfig()
        assert config.max_investments == 10000


class TestCurrencyConfig:
    """Test CurrencyConfig dataclass."""

    def test_currency_config_defaults(self):
        """Test default currency config values."""
        config = CurrencyConfig()
        assert config.base_currency == "USD"


class TestReportingConfig:
    """Test ReportingConfig dataclass."""

    def test_reporting_config_defaults(self):
        """Test default reporting config values."""
        config = ReportingConfig()
        assert config.include_pcaf_alignment is True


class TestBatchConfig:
    """Test BatchConfig dataclass."""

    def test_batch_config_defaults(self):
        """Test default batch config values."""
        config = BatchConfig()
        assert config.max_batch_size == 1000


# ==============================================================================
# SINGLETON PATTERN TESTS
# ==============================================================================


class TestSingleton:
    """Test configuration singleton pattern."""

    def test_get_config_singleton(self):
        """Test get_config returns the same instance."""
        _reset_config()
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2

    def test_reset_config(self):
        """Test _reset_config creates a new instance."""
        config1 = get_config()
        _reset_config()
        config2 = get_config()
        assert config1 is not config2

    def test_get_config_thread_safety(self):
        """Test get_config is thread-safe."""
        _reset_config()
        instances = []

        def get_instance():
            instances.append(get_config())

        threads = [threading.Thread(target=get_instance) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(instances) == 10
        assert all(inst is instances[0] for inst in instances)


# ==============================================================================
# ENVIRONMENT VARIABLE OVERRIDE TESTS
# ==============================================================================


class TestEnvVarOverride:
    """Test GL_INV_ environment variable overrides."""

    def test_env_var_enabled(self, monkeypatch):
        """Test GL_INV_ENABLED override."""
        _reset_config()
        monkeypatch.setenv("GL_INV_ENABLED", "false")
        config = GeneralConfig.from_env()
        assert config.enabled is False

    def test_env_var_debug(self, monkeypatch):
        """Test GL_INV_DEBUG override."""
        _reset_config()
        monkeypatch.setenv("GL_INV_DEBUG", "true")
        config = GeneralConfig.from_env()
        assert config.debug is True

    def test_env_var_log_level(self, monkeypatch):
        """Test GL_INV_LOG_LEVEL override."""
        _reset_config()
        monkeypatch.setenv("GL_INV_LOG_LEVEL", "WARNING")
        config = GeneralConfig.from_env()
        assert config.log_level == "WARNING"

    def test_env_var_max_batch_size(self, monkeypatch):
        """Test GL_INV_MAX_BATCH_SIZE override."""
        _reset_config()
        monkeypatch.setenv("GL_INV_MAX_BATCH_SIZE", "5000")
        config = GeneralConfig.from_env()
        assert config.max_batch_size == 5000

    def test_env_var_default_gwp(self, monkeypatch):
        """Test GL_INV_DEFAULT_GWP override."""
        _reset_config()
        monkeypatch.setenv("GL_INV_DEFAULT_GWP", "AR6")
        config = GeneralConfig.from_env()
        assert config.default_gwp == "AR6"
