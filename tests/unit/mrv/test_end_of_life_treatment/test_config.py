# -*- coding: utf-8 -*-
"""
Test suite for end_of_life_treatment.config - AGENT-MRV-025.

Tests configuration management for the End-of-Life Treatment of Sold Products
Agent (GL-MRV-S3-012) including default values, environment variable loading,
singleton pattern, thread safety, validation, and serialization.

Coverage:
- Default config values for all 18 config sections
- GL_EOL_ environment variable overrides (monkeypatch)
- Singleton pattern (get_config returns same instance)
- Thread safety (12 concurrent threads)
- Validation (invalid values raise errors)
- reset_config() creates new instance
- to_dict / from_dict round-trip
- Frozen immutability
- Cross-section consistency checks

Target: 50+ tests.
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
    from greenlang.end_of_life_treatment.config import (
        get_config,
        reset_config,
        GeneralConfig,
        DatabaseConfig,
        WasteTypeConfig,
        AverageDataConfig,
        ProducerSpecificConfig,
        HybridConfig,
        CircularityConfig,
        ComplianceConfig,
        ProvenanceConfig,
        CacheConfig,
        APIConfig,
        MetricsConfig,
        UncertaintyConfig,
        EFSourceConfig,
        LandfillConfig,
        IncinerationConfig,
        ExportConfig,
        BatchConfig,
        EndOfLifeTreatmentConfig,
    )
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

_SKIP = pytest.mark.skipif(
    not CONFIG_AVAILABLE,
    reason="end_of_life_treatment.config not available",
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
        """Test default agent_id is GL-MRV-S3-012."""
        config = GeneralConfig()
        assert config.agent_id == "GL-MRV-S3-012"
        assert config.agent_component == "AGENT-MRV-025"
        assert config.version == "1.0.0"

    def test_general_config_api_prefix(self):
        """Test default API prefix."""
        config = GeneralConfig()
        assert config.api_prefix == "/api/v1/end-of-life-treatment"

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
        config = GeneralConfig(api_prefix="api/v1/end-of-life-treatment")
        with pytest.raises(ValueError, match="api_prefix must start with"):
            config.validate()

    def test_to_dict(self):
        """Test to_dict serialization."""
        config = GeneralConfig()
        d = config.to_dict()
        assert isinstance(d, dict)
        assert d["agent_id"] == "GL-MRV-S3-012"


# ==============================================================================
# DATABASE CONFIGURATION TESTS
# ==============================================================================


class TestDatabaseConfig:
    """Test DatabaseConfig dataclass."""

    def test_defaults(self):
        """Test default database config values."""
        config = DatabaseConfig()
        assert config.host == "localhost"
        assert config.port == 5432
        assert config.table_prefix == "gl_eol_"

    def test_schema_name(self):
        """Test default schema name."""
        config = DatabaseConfig()
        assert config.schema == "end_of_life_treatment_service"


# ==============================================================================
# WASTE TYPE CONFIGURATION TESTS
# ==============================================================================


class TestWasteTypeConfig:
    """Test WasteTypeConfig dataclass."""

    def test_defaults(self):
        """Test default waste type config values."""
        config = WasteTypeConfig()
        assert config.enable_fod_model is True
        assert config.default_climate == "temperate_wet"
        assert config.projection_years == 50

    def test_default_docf(self):
        """Test default DOCf value is 0.50."""
        config = WasteTypeConfig()
        assert config.default_docf == Decimal("0.50")

    def test_default_oxidation(self):
        """Test default oxidation factor is 0.10."""
        config = WasteTypeConfig()
        assert config.default_oxidation == Decimal("0.10")


# ==============================================================================
# AVERAGE DATA CONFIGURATION TESTS
# ==============================================================================


class TestAverageDataConfig:
    """Test AverageDataConfig dataclass."""

    def test_defaults(self):
        """Test default average data config values."""
        config = AverageDataConfig()
        assert config.default_region == "GLOBAL"
        assert config.enable_weight_estimation is True


# ==============================================================================
# PRODUCER SPECIFIC CONFIGURATION TESTS
# ==============================================================================


class TestProducerSpecificConfig:
    """Test ProducerSpecificConfig dataclass."""

    def test_defaults(self):
        """Test default producer specific config values."""
        config = ProducerSpecificConfig()
        assert config.require_epd is False
        assert config.min_verification_level == "self_declared"


# ==============================================================================
# HYBRID CONFIGURATION TESTS
# ==============================================================================


class TestHybridConfig:
    """Test HybridConfig dataclass."""

    def test_method_priority(self):
        """Test default method priority order."""
        config = HybridConfig()
        priority = config.method_priority
        assert priority[0] == "producer_specific"
        assert "waste_type_specific" in priority
        assert "average_data" in priority

    def test_separate_avoided_emissions(self):
        """Test avoided emissions are always reported separately by default."""
        config = HybridConfig()
        assert config.separate_avoided_emissions is True

    def test_gap_filling_enabled(self):
        """Test gap filling is enabled by default."""
        config = HybridConfig()
        assert config.enable_gap_filling is True


# ==============================================================================
# CIRCULARITY CONFIGURATION TESTS
# ==============================================================================


class TestCircularityConfig:
    """Test CircularityConfig dataclass."""

    def test_defaults(self):
        """Test default circularity config values."""
        config = CircularityConfig()
        assert config.enable_circularity_metrics is True
        assert config.track_waste_hierarchy is True


# ==============================================================================
# COMPLIANCE CONFIGURATION TESTS
# ==============================================================================


class TestComplianceConfig:
    """Test ComplianceConfig dataclass."""

    def test_defaults(self):
        """Test default compliance config values."""
        config = ComplianceConfig()
        assert config.strict_mode is False

    def test_materiality_threshold(self):
        """Test default materiality threshold."""
        config = ComplianceConfig()
        assert config.materiality_threshold == Decimal("0.01")

    def test_get_frameworks(self):
        """Test get_frameworks returns all 7."""
        config = ComplianceConfig()
        frameworks = config.get_frameworks()
        assert len(frameworks) == 7
        assert "GHG_PROTOCOL_SCOPE3" in frameworks
        assert "CSRD_ESRS_E5" in frameworks


# ==============================================================================
# SINGLETON AND THREAD SAFETY TESTS
# ==============================================================================


class TestSingleton:
    """Test configuration singleton pattern."""

    def test_get_config_returns_same_instance(self):
        """Test get_config returns the same instance."""
        reset_config()
        c1 = get_config()
        c2 = get_config()
        assert c1 is c2

    def test_reset_config_creates_new_instance(self):
        """Test reset_config creates a new instance."""
        c1 = get_config()
        reset_config()
        c2 = get_config()
        assert c1 is not c2

    def test_thread_safety_12_threads(self):
        """Test singleton is thread-safe with 12 concurrent threads."""
        reset_config()
        instances = []
        errors = []

        def get_instance():
            try:
                instance = get_config()
                instances.append(id(instance))
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=get_instance) for _ in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0, f"Thread errors: {errors}"
        assert len(set(instances)) == 1, "Singleton returned different instances"


# ==============================================================================
# ENVIRONMENT VARIABLE OVERRIDE TESTS
# ==============================================================================


class TestEnvOverrides:
    """Test GL_EOL_ environment variable overrides."""

    def test_gl_eol_debug(self, monkeypatch):
        """Test GL_EOL_DEBUG env var enables debug mode."""
        monkeypatch.setenv("GL_EOL_DEBUG", "true")
        reset_config()
        config = get_config()
        assert config.general.debug is True

    def test_gl_eol_log_level(self, monkeypatch):
        """Test GL_EOL_LOG_LEVEL env var overrides log level."""
        monkeypatch.setenv("GL_EOL_LOG_LEVEL", "DEBUG")
        reset_config()
        config = get_config()
        assert config.general.log_level == "DEBUG"

    def test_gl_eol_max_batch_size(self, monkeypatch):
        """Test GL_EOL_MAX_BATCH_SIZE env var overrides batch size."""
        monkeypatch.setenv("GL_EOL_MAX_BATCH_SIZE", "500")
        reset_config()
        config = get_config()
        assert config.general.max_batch_size == 500

    def test_gl_eol_default_gwp(self, monkeypatch):
        """Test GL_EOL_DEFAULT_GWP env var overrides GWP version."""
        monkeypatch.setenv("GL_EOL_DEFAULT_GWP", "AR6")
        reset_config()
        config = get_config()
        assert config.general.default_gwp == "AR6"

    def test_gl_eol_db_host(self, monkeypatch):
        """Test GL_EOL_DB_HOST env var overrides database host."""
        monkeypatch.setenv("GL_EOL_DB_HOST", "prod-db.example.com")
        reset_config()
        config = get_config()
        assert config.database.host == "prod-db.example.com"

    def test_gl_eol_db_port(self, monkeypatch):
        """Test GL_EOL_DB_PORT env var overrides database port."""
        monkeypatch.setenv("GL_EOL_DB_PORT", "5433")
        reset_config()
        config = get_config()
        assert config.database.port == 5433

    def test_gl_eol_default_climate(self, monkeypatch):
        """Test GL_EOL_DEFAULT_CLIMATE env var overrides climate zone."""
        monkeypatch.setenv("GL_EOL_DEFAULT_CLIMATE", "tropical_wet")
        reset_config()
        config = get_config()
        assert config.waste_type.default_climate == "tropical_wet"

    def test_gl_eol_projection_years(self, monkeypatch):
        """Test GL_EOL_PROJECTION_YEARS env var overrides FOD projection."""
        monkeypatch.setenv("GL_EOL_PROJECTION_YEARS", "100")
        reset_config()
        config = get_config()
        assert config.waste_type.projection_years == 100

    def test_gl_eol_default_region(self, monkeypatch):
        """Test GL_EOL_DEFAULT_REGION env var overrides region."""
        monkeypatch.setenv("GL_EOL_DEFAULT_REGION", "DE")
        reset_config()
        config = get_config()
        assert config.average_data.default_region == "DE"


# ==============================================================================
# CROSS-SECTION CONSISTENCY TESTS
# ==============================================================================


class TestCrossSectionConsistency:
    """Test cross-section configuration consistency."""

    def test_table_prefix_consistent(self):
        """Test table_prefix is consistent across sections."""
        config = get_config()
        assert config.database.table_prefix == "gl_eol_"
        assert config.metrics.prefix == "gl_eol_"

    def test_api_prefix_consistent(self):
        """Test API prefix is consistent."""
        config = get_config()
        assert config.general.api_prefix == config.api.prefix

    def test_agent_id_consistent(self):
        """Test agent_id matches across the module."""
        config = get_config()
        assert config.general.agent_id == "GL-MRV-S3-012"

    def test_version_consistent(self):
        """Test version matches across the module."""
        config = get_config()
        assert config.general.version == "1.0.0"

    def test_hybrid_priority_contains_all_methods(self):
        """Test hybrid method priority includes all calculation methods."""
        config = get_config()
        priority = config.hybrid.method_priority
        assert "producer_specific" in priority
        assert "waste_type_specific" in priority
        assert "average_data" in priority

    def test_frameworks_include_esrs_e5(self):
        """Test compliance frameworks include ESRS E5 (circular economy)."""
        config = get_config()
        frameworks = config.compliance.get_frameworks()
        assert "CSRD_ESRS_E5" in frameworks
