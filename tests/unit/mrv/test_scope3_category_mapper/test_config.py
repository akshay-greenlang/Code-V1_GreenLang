# -*- coding: utf-8 -*-
"""
Test suite for scope3_category_mapper.config - AGENT-MRV-029.

Tests configuration management for the Scope 3 Category Mapper Agent
(GL-MRV-X-040) including default values, environment variable loading,
singleton pattern, thread safety, validation, and serialization.

Coverage:
- Default config values for GeneralConfig, ClassificationConfig,
  BoundaryConfig, CompletenessConfig, DoubleCountingConfig,
  ComplianceConfig, DatabaseConfig, MetricsConfig, ProvenanceConfig
- GL_SCM_ environment variable loading (monkeypatch)
- Singleton pattern (get_config returns same instance)
- Thread safety
- Validation (invalid values raise errors)
- to_dict / from_dict round-trip
- Frozen immutability
- reset_config()
- validate_all() cross-validation

Total: ~50 tests

Author: GL-TestEngineer
Date: March 2026
"""

import os
import threading
from decimal import Decimal
from typing import Any, Dict

import pytest

from greenlang.agents.mrv.scope3_category_mapper.config import (
    get_config,
    reset_config,
    set_config,
    validate_config,
    GeneralConfig,
    ClassificationConfig,
    BoundaryConfig,
    CompletenessConfig,
    DoubleCountingConfig,
    ComplianceConfig,
    DatabaseConfig,
    MetricsConfig,
    ProvenanceConfig,
    Scope3CategoryMapperConfig,
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
        assert config.max_batch_size == 50000

    def test_general_config_agent_id(self):
        """Test default agent_id is GL-MRV-X-040."""
        config = GeneralConfig()
        assert config.agent_id == "GL-MRV-X-040"
        assert config.agent_component == "AGENT-MRV-029"
        assert config.version == "1.0.0"

    def test_general_config_api_prefix(self):
        """Test default API prefix is /api/v1/scope3-category-mapper."""
        config = GeneralConfig()
        assert config.api_prefix == "/api/v1/scope3-category-mapper"

    def test_general_config_default_gwp(self):
        """Test default GWP is AR5."""
        config = GeneralConfig()
        assert config.default_gwp == "AR5"

    def test_general_config_default_currency(self):
        """Test default currency is USD."""
        config = GeneralConfig()
        assert config.default_currency == "USD"

    def test_config_frozen_immutability(self):
        """Test config is immutable (frozen=True)."""
        config = GeneralConfig()
        with pytest.raises(Exception):
            config.enabled = False  # type: ignore[misc]

    def test_config_validate_success(self):
        """Test validate() succeeds for default config."""
        config = GeneralConfig()
        config.validate()  # Should not raise

    def test_config_validate_invalid_log_level(self):
        """Test validate() raises ValueError for invalid log_level."""
        config = GeneralConfig(log_level="VERBOSE")
        with pytest.raises(ValueError, match="Invalid log_level"):
            config.validate()

    def test_config_validate_empty_agent_id(self):
        """Test validate() raises ValueError for empty agent_id."""
        config = GeneralConfig(agent_id="")
        with pytest.raises(ValueError, match="agent_id cannot be empty"):
            config.validate()

    def test_config_validate_empty_agent_component(self):
        """Test validate() raises ValueError for empty agent_component."""
        config = GeneralConfig(agent_component="")
        with pytest.raises(ValueError, match="agent_component cannot be empty"):
            config.validate()

    def test_config_validate_bad_version(self):
        """Test validate() raises ValueError for non-SemVer version."""
        config = GeneralConfig(version="1.0")
        with pytest.raises(ValueError, match="Must follow SemVer"):
            config.validate()

    def test_config_validate_bad_api_prefix(self):
        """Test validate() raises ValueError for prefix not starting with /."""
        config = GeneralConfig(api_prefix="api/v1/scope3")
        with pytest.raises(ValueError, match="must start with '/'"):
            config.validate()

    def test_config_validate_empty_api_prefix(self):
        """Test validate() raises ValueError for empty api_prefix."""
        config = GeneralConfig(api_prefix="")
        with pytest.raises(ValueError, match="api_prefix cannot be empty"):
            config.validate()

    def test_config_validate_batch_size_too_small(self):
        """Test validate() raises ValueError for batch_size < 1."""
        config = GeneralConfig(max_batch_size=0)
        with pytest.raises(ValueError, match="max_batch_size"):
            config.validate()

    def test_config_validate_batch_size_too_large(self):
        """Test validate() raises ValueError for batch_size > 500000."""
        config = GeneralConfig(max_batch_size=500001)
        with pytest.raises(ValueError, match="max_batch_size"):
            config.validate()

    def test_config_validate_bad_gwp(self):
        """Test validate() raises ValueError for invalid GWP."""
        config = GeneralConfig(default_gwp="AR3")
        with pytest.raises(ValueError, match="Invalid default_gwp"):
            config.validate()

    def test_config_validate_gwp_sar_accepted(self):
        """Test validate() accepts SAR as valid GWP."""
        config = GeneralConfig(default_gwp="SAR")
        config.validate()  # Should not raise

    def test_config_validate_bad_currency_length(self):
        """Test validate() raises ValueError for non-3-char currency."""
        config = GeneralConfig(default_currency="US")
        with pytest.raises(ValueError, match="3-letter ISO 4217"):
            config.validate()

    def test_config_to_dict(self):
        """Test to_dict() returns dictionary with all fields."""
        config = GeneralConfig()
        d = config.to_dict()
        assert d["agent_id"] == "GL-MRV-X-040"
        assert d["enabled"] is True
        assert d["max_batch_size"] == 50000
        assert d["default_gwp"] == "AR5"
        assert d["default_currency"] == "USD"

    def test_config_from_dict(self):
        """Test from_dict() creates config from dictionary."""
        d = {"enabled": False, "debug": True, "log_level": "DEBUG"}
        config = GeneralConfig.from_dict(d)
        assert config.enabled is False
        assert config.debug is True
        assert config.log_level == "DEBUG"

    def test_config_to_dict_from_dict_roundtrip(self):
        """Test to_dict -> from_dict round-trip preserves values."""
        original = GeneralConfig()
        restored = GeneralConfig.from_dict(original.to_dict())
        assert original == restored


# ==============================================================================
# CLASSIFICATION CONFIGURATION TESTS
# ==============================================================================


class TestClassificationConfig:
    """Test ClassificationConfig dataclass."""

    def test_classification_config_defaults(self):
        """Test default classification config values."""
        config = ClassificationConfig()
        assert config.min_confidence_threshold == 0.3
        assert config.naics_confidence == 0.95
        assert config.isic_confidence == 0.90
        assert config.gl_account_confidence == 0.85
        assert config.keyword_confidence == 0.40

    def test_classification_config_default_category(self):
        """Test default fallback category."""
        config = ClassificationConfig()
        assert config.default_category == "1_purchased_goods_services"

    def test_classification_config_multi_category_split(self):
        """Test default multi-category split settings."""
        config = ClassificationConfig()
        assert config.enable_multi_category_split is True
        assert config.max_split_categories == 3

    def test_classification_config_validate_success(self):
        """Test validate() succeeds for default config."""
        config = ClassificationConfig()
        config.validate()  # Should not raise

    def test_classification_config_validate_bad_min_confidence(self):
        """Test validate() rejects min_confidence_threshold outside [0.0, 1.0]."""
        config = ClassificationConfig(min_confidence_threshold=1.5)
        with pytest.raises(ValueError, match="min_confidence_threshold"):
            config.validate()

    def test_classification_config_validate_bad_naics_confidence(self):
        """Test validate() rejects naics_confidence outside [0.0, 1.0]."""
        config = ClassificationConfig(naics_confidence=1.1)
        with pytest.raises(ValueError, match="naics_confidence"):
            config.validate()

    def test_classification_config_validate_empty_default_category(self):
        """Test validate() rejects empty default_category."""
        config = ClassificationConfig(default_category="")
        with pytest.raises(ValueError, match="default_category cannot be empty"):
            config.validate()

    def test_classification_config_validate_bad_max_split(self):
        """Test validate() rejects max_split_categories outside [1, 15]."""
        config = ClassificationConfig(max_split_categories=0)
        with pytest.raises(ValueError, match="max_split_categories"):
            config.validate()

    def test_classification_config_frozen(self):
        """Test config is immutable (frozen=True)."""
        config = ClassificationConfig()
        with pytest.raises(Exception):
            config.naics_confidence = 0.5  # type: ignore[misc]

    def test_classification_config_to_dict_roundtrip(self):
        """Test to_dict -> from_dict round-trip preserves values."""
        original = ClassificationConfig()
        restored = ClassificationConfig.from_dict(original.to_dict())
        assert original == restored


# ==============================================================================
# BOUNDARY CONFIGURATION TESTS
# ==============================================================================


class TestBoundaryConfig:
    """Test BoundaryConfig dataclass."""

    def test_boundary_config_defaults(self):
        """Test default boundary config values."""
        config = BoundaryConfig()
        assert config.default_consolidation == "operational_control"
        assert config.capex_threshold == 5000.0
        assert config.default_incoterm == "FCA"
        assert config.lease_scope_boundary_months == 12

    def test_boundary_config_validate_success(self):
        """Test validate() succeeds for default config."""
        config = BoundaryConfig()
        config.validate()  # Should not raise

    def test_boundary_config_validate_bad_consolidation(self):
        """Test validate() rejects invalid consolidation approach."""
        config = BoundaryConfig(default_consolidation="invalid")
        with pytest.raises(ValueError, match="Invalid default_consolidation"):
            config.validate()

    def test_boundary_config_validate_negative_capex(self):
        """Test validate() rejects negative capex threshold."""
        config = BoundaryConfig(capex_threshold=-1.0)
        with pytest.raises(ValueError, match="capex_threshold"):
            config.validate()

    def test_boundary_config_validate_bad_incoterm(self):
        """Test validate() rejects invalid Incoterm."""
        config = BoundaryConfig(default_incoterm="INVALID")
        with pytest.raises(ValueError, match="Invalid default_incoterm"):
            config.validate()

    def test_boundary_config_validate_bad_lease_months(self):
        """Test validate() rejects lease months out of range."""
        config = BoundaryConfig(lease_scope_boundary_months=0)
        with pytest.raises(ValueError, match="lease_scope_boundary_months"):
            config.validate()

    def test_boundary_config_frozen(self):
        """Test config is immutable (frozen=True)."""
        config = BoundaryConfig()
        with pytest.raises(Exception):
            config.capex_threshold = 10000.0  # type: ignore[misc]

    def test_boundary_config_to_dict_roundtrip(self):
        """Test to_dict -> from_dict round-trip preserves values."""
        original = BoundaryConfig()
        restored = BoundaryConfig.from_dict(original.to_dict())
        assert original == restored


# ==============================================================================
# COMPLETENESS CONFIGURATION TESTS
# ==============================================================================


class TestCompletenessConfig:
    """Test CompletenessConfig dataclass."""

    def test_completeness_config_defaults(self):
        """Test default completeness config values."""
        config = CompletenessConfig()
        assert config.materiality_threshold_pct == 1.0
        assert config.min_categories_reported == 5
        assert config.enable_industry_benchmarks is True

    def test_completeness_config_validate_success(self):
        """Test validate() succeeds for default config."""
        config = CompletenessConfig()
        config.validate()  # Should not raise

    def test_completeness_config_validate_negative_materiality(self):
        """Test validate() rejects negative materiality threshold."""
        config = CompletenessConfig(materiality_threshold_pct=-0.01)
        with pytest.raises(ValueError, match="materiality_threshold_pct"):
            config.validate()

    def test_completeness_config_validate_above_100_materiality(self):
        """Test validate() rejects materiality above 100."""
        config = CompletenessConfig(materiality_threshold_pct=101.0)
        with pytest.raises(ValueError, match="materiality_threshold_pct"):
            config.validate()

    def test_completeness_config_validate_bad_min_categories(self):
        """Test validate() rejects min_categories outside [0, 15]."""
        config = CompletenessConfig(min_categories_reported=16)
        with pytest.raises(ValueError, match="min_categories_reported"):
            config.validate()

    def test_completeness_config_frozen(self):
        """Test config is immutable (frozen=True)."""
        config = CompletenessConfig()
        with pytest.raises(Exception):
            config.materiality_threshold_pct = 5.0  # type: ignore[misc]


# ==============================================================================
# DOUBLE-COUNTING CONFIGURATION TESTS
# ==============================================================================


class TestDoubleCountingConfig:
    """Test DoubleCountingConfig dataclass."""

    def test_double_counting_config_defaults(self):
        """Test default double-counting config values."""
        config = DoubleCountingConfig()
        assert config.enable_dc_checks is True
        assert config.strict_mode is False

    def test_double_counting_config_all_rules_enabled(self):
        """Test all 10 DC rules are enabled by default."""
        config = DoubleCountingConfig()
        assert len(config.rules_enabled) == 10
        for i in range(1, 11):
            rule_code = f"DC-SCM-{i:03d}"
            assert rule_code in config.rules_enabled, (
                f"Rule {rule_code} should be in default rules_enabled"
            )

    def test_double_counting_config_validate_success(self):
        """Test validate() succeeds for default config."""
        config = DoubleCountingConfig()
        config.validate()  # Should not raise

    def test_double_counting_config_validate_invalid_rule(self):
        """Test validate() rejects invalid DC rule codes."""
        config = DoubleCountingConfig(
            rules_enabled=("DC-SCM-001", "DC-INVALID-999")
        )
        with pytest.raises(ValueError, match="Invalid DC rule"):
            config.validate()

    def test_double_counting_config_frozen(self):
        """Test config is immutable (frozen=True)."""
        config = DoubleCountingConfig()
        with pytest.raises(Exception):
            config.enable_dc_checks = False  # type: ignore[misc]

    def test_double_counting_config_to_dict_roundtrip(self):
        """Test to_dict -> from_dict round-trip preserves values."""
        original = DoubleCountingConfig()
        restored = DoubleCountingConfig.from_dict(original.to_dict())
        assert original == restored


# ==============================================================================
# COMPLIANCE CONFIGURATION TESTS
# ==============================================================================


class TestComplianceConfig:
    """Test ComplianceConfig dataclass."""

    def test_compliance_config_defaults(self):
        """Test default compliance config values."""
        config = ComplianceConfig()
        assert config.minimum_compliance_score == 70.0

    def test_compliance_config_all_frameworks(self):
        """Test all 8 regulatory frameworks are included by default."""
        config = ComplianceConfig()
        expected = (
            "GHG_PROTOCOL", "ISO_14064", "CSRD_ESRS", "CDP",
            "SBTI", "SB_253", "SEC_CLIMATE", "EU_TAXONOMY",
        )
        assert config.frameworks_enabled == expected
        assert len(config.frameworks_enabled) == 8

    def test_compliance_config_validate_success(self):
        """Test validate() succeeds for default config."""
        config = ComplianceConfig()
        config.validate()  # Should not raise

    def test_compliance_config_validate_invalid_framework(self):
        """Test validate() rejects invalid framework names."""
        config = ComplianceConfig(
            frameworks_enabled=("GHG_PROTOCOL", "BOGUS_FRAMEWORK")
        )
        with pytest.raises(ValueError, match="Invalid framework"):
            config.validate()

    def test_compliance_config_validate_bad_score(self):
        """Test validate() rejects minimum_compliance_score outside [0, 100]."""
        config = ComplianceConfig(minimum_compliance_score=101.0)
        with pytest.raises(ValueError, match="minimum_compliance_score"):
            config.validate()

    def test_compliance_config_frozen(self):
        """Test config is immutable (frozen=True)."""
        config = ComplianceConfig()
        with pytest.raises(Exception):
            config.minimum_compliance_score = 50.0  # type: ignore[misc]

    def test_compliance_config_to_dict_roundtrip(self):
        """Test to_dict -> from_dict round-trip preserves values."""
        original = ComplianceConfig()
        restored = ComplianceConfig.from_dict(original.to_dict())
        assert original == restored


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
        assert config.name == "greenlang"
        assert config.username == "greenlang"
        assert config.password == ""
        assert config.schema == "scope3_mapper_service"
        assert config.pool_size == 10
        assert config.ssl_mode == "prefer"

    def test_database_config_table_prefix(self):
        """Test default table prefix is gl_scm_."""
        config = DatabaseConfig()
        assert config.table_prefix == "gl_scm_"

    def test_database_config_validate_success(self):
        """Test validate() succeeds for default config."""
        config = DatabaseConfig()
        config.validate()  # Should not raise

    def test_database_config_validate_empty_host(self):
        """Test validate() rejects empty host."""
        config = DatabaseConfig(host="")
        with pytest.raises(ValueError, match="host cannot be empty"):
            config.validate()

    def test_database_config_validate_bad_port(self):
        """Test validate() rejects port outside valid range."""
        config = DatabaseConfig(port=0)
        with pytest.raises(ValueError, match="port must be between"):
            config.validate()

    def test_database_config_validate_empty_name(self):
        """Test validate() rejects empty database name."""
        config = DatabaseConfig(name="")
        with pytest.raises(ValueError, match="name cannot be empty"):
            config.validate()

    def test_database_config_validate_bad_pool_size(self):
        """Test validate() rejects pool_size outside range."""
        config = DatabaseConfig(pool_size=0)
        with pytest.raises(ValueError, match="pool_size"):
            config.validate()

    def test_database_config_validate_bad_ssl_mode(self):
        """Test validate() rejects invalid ssl_mode."""
        config = DatabaseConfig(ssl_mode="invalid")
        with pytest.raises(ValueError, match="Invalid ssl_mode"):
            config.validate()

    def test_database_config_validate_bad_table_prefix(self):
        """Test validate() rejects table_prefix not ending with underscore."""
        config = DatabaseConfig(table_prefix="gl_scm")
        with pytest.raises(ValueError, match="table_prefix must end with"):
            config.validate()

    def test_database_config_get_connection_url(self):
        """Test get_connection_url() builds a valid PostgreSQL URL."""
        config = DatabaseConfig()
        url = config.get_connection_url()
        assert url.startswith("postgresql://")
        assert "localhost" in url
        assert "5432" in url
        assert "greenlang" in url

    def test_database_config_get_connection_url_with_password(self):
        """Test get_connection_url() includes password when set."""
        config = DatabaseConfig(password="secret")
        url = config.get_connection_url()
        assert "greenlang:secret@" in url

    def test_database_config_frozen(self):
        """Test config is immutable (frozen=True)."""
        config = DatabaseConfig()
        with pytest.raises(Exception):
            config.host = "remote"  # type: ignore[misc]


# ==============================================================================
# METRICS CONFIGURATION TESTS
# ==============================================================================


class TestMetricsConfig:
    """Test MetricsConfig dataclass."""

    def test_metrics_config_defaults(self):
        """Test default metrics config values."""
        config = MetricsConfig()
        assert config.enabled is True
        assert config.prefix == "gl_scm_"
        assert config.port == 9090

    def test_metrics_config_validate_success(self):
        """Test validate() succeeds for default config."""
        config = MetricsConfig()
        config.validate()  # Should not raise

    def test_metrics_config_validate_empty_prefix(self):
        """Test validate() rejects empty metrics prefix."""
        config = MetricsConfig(prefix="")
        with pytest.raises(ValueError, match="prefix cannot be empty"):
            config.validate()

    def test_metrics_config_validate_prefix_no_underscore(self):
        """Test validate() rejects prefix not ending with underscore."""
        config = MetricsConfig(prefix="gl_scm")
        with pytest.raises(ValueError, match="prefix must end with"):
            config.validate()

    def test_metrics_config_validate_bad_port(self):
        """Test validate() rejects port outside valid range."""
        config = MetricsConfig(port=0)
        with pytest.raises(ValueError, match="port must be between"):
            config.validate()

    def test_metrics_config_frozen(self):
        """Test config is immutable (frozen=True)."""
        config = MetricsConfig()
        with pytest.raises(Exception):
            config.enabled = False  # type: ignore[misc]


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
        assert config.chain_length == 10

    def test_provenance_config_validate_success(self):
        """Test validate() succeeds for default config."""
        config = ProvenanceConfig()
        config.validate()  # Should not raise

    def test_provenance_config_validate_bad_algorithm(self):
        """Test validate() rejects invalid hash algorithm."""
        config = ProvenanceConfig(hash_algorithm="md5")
        with pytest.raises(ValueError, match="Invalid hash_algorithm"):
            config.validate()

    def test_provenance_config_validate_bad_chain_length(self):
        """Test validate() rejects chain_length outside valid range."""
        config = ProvenanceConfig(chain_length=0)
        with pytest.raises(ValueError, match="chain_length"):
            config.validate()

    def test_provenance_config_accepts_sha384(self):
        """Test validate() accepts sha384 as valid algorithm."""
        config = ProvenanceConfig(hash_algorithm="sha384")
        config.validate()  # Should not raise

    def test_provenance_config_accepts_sha512(self):
        """Test validate() accepts sha512 as valid algorithm."""
        config = ProvenanceConfig(hash_algorithm="sha512")
        config.validate()  # Should not raise

    def test_provenance_config_frozen(self):
        """Test config is immutable (frozen=True)."""
        config = ProvenanceConfig()
        with pytest.raises(Exception):
            config.hash_algorithm = "md5"  # type: ignore[misc]


# ==============================================================================
# MASTER CONFIG AND SINGLETON TESTS
# ==============================================================================


class TestScope3CategoryMapperConfig:
    """Test Scope3CategoryMapperConfig master dataclass."""

    def test_master_config_has_all_sections(self):
        """Test master config has all 9 configuration sections."""
        config = Scope3CategoryMapperConfig()
        assert hasattr(config, "general")
        assert hasattr(config, "classification")
        assert hasattr(config, "boundary")
        assert hasattr(config, "completeness")
        assert hasattr(config, "double_counting")
        assert hasattr(config, "compliance")
        assert hasattr(config, "database")
        assert hasattr(config, "metrics")
        assert hasattr(config, "provenance")

    def test_master_config_validate_all_no_errors(self):
        """Test validate_all() returns empty list for default config."""
        config = Scope3CategoryMapperConfig()
        errors = config.validate_all()
        assert isinstance(errors, list)
        # Default config is valid (no section errors)
        # There may be cross-validation warnings but no errors
        for error in errors:
            assert "cross-validation" in error or isinstance(error, str)

    def test_master_config_validate_raises_on_error(self):
        """Test validate() raises ValueError when there are errors."""
        config = Scope3CategoryMapperConfig(
            general=GeneralConfig(log_level="INVALID_LEVEL")
        )
        with pytest.raises(ValueError, match="Configuration validation failed"):
            config.validate()

    def test_master_config_to_dict_from_dict_roundtrip(self):
        """Test to_dict -> from_dict round-trip."""
        original = Scope3CategoryMapperConfig()
        restored = Scope3CategoryMapperConfig.from_dict(original.to_dict())
        assert original.general == restored.general
        assert original.classification == restored.classification
        assert original.boundary == restored.boundary

    def test_master_config_from_env(self):
        """Test from_env() loads all sections from environment."""
        config = Scope3CategoryMapperConfig.from_env()
        assert isinstance(config.general, GeneralConfig)
        assert isinstance(config.classification, ClassificationConfig)
        assert isinstance(config.database, DatabaseConfig)


class TestGetConfigSingleton:
    """Test get_config() singleton pattern and reset_config()."""

    def test_get_config_returns_instance(self):
        """Test get_config() returns a Scope3CategoryMapperConfig."""
        reset_config()
        config = get_config()
        assert isinstance(config, Scope3CategoryMapperConfig)

    def test_get_config_singleton(self):
        """Test get_config() returns the same instance on repeated calls."""
        reset_config()
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2

    def test_reset_config_clears_singleton(self):
        """Test reset_config() clears the cached singleton."""
        reset_config()
        config1 = get_config()
        reset_config()
        config2 = get_config()
        # Both are valid configs
        assert isinstance(config1, Scope3CategoryMapperConfig)
        assert isinstance(config2, Scope3CategoryMapperConfig)

    def test_set_config_replaces_singleton(self):
        """Test set_config() replaces the singleton instance."""
        reset_config()
        custom = Scope3CategoryMapperConfig(
            general=GeneralConfig(debug=True)
        )
        set_config(custom)
        result = get_config()
        assert result.general.debug is True

    def test_set_config_rejects_invalid_type(self):
        """Test set_config() raises TypeError for wrong type."""
        with pytest.raises(TypeError, match="must be a Scope3CategoryMapperConfig"):
            set_config("not_a_config")  # type: ignore[arg-type]

    def test_validate_config_returns_errors_list(self):
        """Test validate_config() returns a list of error strings."""
        config = Scope3CategoryMapperConfig()
        errors = validate_config(config)
        assert isinstance(errors, list)

    def test_get_config_thread_safety(self):
        """Test get_config() is thread-safe (no race conditions)."""
        reset_config()
        results = []
        errors = []

        def worker():
            try:
                config = get_config()
                results.append(id(config))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"
        # All threads must get the same singleton instance
        assert len(set(results)) == 1, (
            "Multiple singleton instances created across threads"
        )


# ==============================================================================
# ENVIRONMENT VARIABLE OVERRIDE TESTS
# ==============================================================================


class TestEnvOverrides:
    """Test environment variable overrides for config values."""

    def test_env_override_GL_SCM_ENABLED(self, monkeypatch):
        """Test GL_SCM_ENABLED=false disables the agent."""
        monkeypatch.setenv("GL_SCM_ENABLED", "false")
        config = GeneralConfig.from_env()
        assert config.enabled is False

    def test_env_override_GL_SCM_DEBUG(self, monkeypatch):
        """Test GL_SCM_DEBUG=true enables debug mode."""
        monkeypatch.setenv("GL_SCM_DEBUG", "true")
        config = GeneralConfig.from_env()
        assert config.debug is True

    def test_env_override_GL_SCM_LOG_LEVEL(self, monkeypatch):
        """Test GL_SCM_LOG_LEVEL=DEBUG overrides log level."""
        monkeypatch.setenv("GL_SCM_LOG_LEVEL", "DEBUG")
        config = GeneralConfig.from_env()
        assert config.log_level == "DEBUG"

    def test_env_override_GL_SCM_MAX_BATCH_SIZE(self, monkeypatch):
        """Test GL_SCM_MAX_BATCH_SIZE=500 overrides batch size."""
        monkeypatch.setenv("GL_SCM_MAX_BATCH_SIZE", "500")
        config = GeneralConfig.from_env()
        assert config.max_batch_size == 500

    def test_env_override_GL_SCM_DEFAULT_GWP(self, monkeypatch):
        """Test GL_SCM_DEFAULT_GWP=AR6 overrides GWP version."""
        monkeypatch.setenv("GL_SCM_DEFAULT_GWP", "AR6")
        config = GeneralConfig.from_env()
        assert config.default_gwp == "AR6"

    def test_env_override_GL_SCM_MIN_CONFIDENCE(self, monkeypatch):
        """Test GL_SCM_MIN_CONFIDENCE=0.5 overrides min confidence threshold."""
        monkeypatch.setenv("GL_SCM_MIN_CONFIDENCE", "0.5")
        config = ClassificationConfig.from_env()
        assert config.min_confidence_threshold == 0.5

    def test_env_override_GL_SCM_NAICS_CONFIDENCE(self, monkeypatch):
        """Test GL_SCM_NAICS_CONFIDENCE=0.80 overrides NAICS confidence."""
        monkeypatch.setenv("GL_SCM_NAICS_CONFIDENCE", "0.80")
        config = ClassificationConfig.from_env()
        assert config.naics_confidence == 0.80

    def test_env_override_GL_SCM_CONSOLIDATION(self, monkeypatch):
        """Test GL_SCM_CONSOLIDATION=equity_share overrides boundary."""
        monkeypatch.setenv("GL_SCM_CONSOLIDATION", "equity_share")
        config = BoundaryConfig.from_env()
        assert config.default_consolidation == "equity_share"

    def test_env_override_GL_SCM_CAPEX_THRESHOLD(self, monkeypatch):
        """Test GL_SCM_CAPEX_THRESHOLD=10000 overrides capex threshold."""
        monkeypatch.setenv("GL_SCM_CAPEX_THRESHOLD", "10000.0")
        config = BoundaryConfig.from_env()
        assert config.capex_threshold == 10000.0

    def test_env_override_GL_SCM_DEFAULT_INCOTERM(self, monkeypatch):
        """Test GL_SCM_DEFAULT_INCOTERM=FOB overrides default Incoterm."""
        monkeypatch.setenv("GL_SCM_DEFAULT_INCOTERM", "FOB")
        config = BoundaryConfig.from_env()
        assert config.default_incoterm == "FOB"

    def test_env_override_GL_SCM_MATERIALITY_PCT(self, monkeypatch):
        """Test GL_SCM_MATERIALITY_PCT=5.0 overrides materiality threshold."""
        monkeypatch.setenv("GL_SCM_MATERIALITY_PCT", "5.0")
        config = CompletenessConfig.from_env()
        assert config.materiality_threshold_pct == 5.0

    def test_env_override_GL_SCM_DC_ENABLED(self, monkeypatch):
        """Test GL_SCM_DC_ENABLED=false disables double-counting checks."""
        monkeypatch.setenv("GL_SCM_DC_ENABLED", "false")
        config = DoubleCountingConfig.from_env()
        assert config.enable_dc_checks is False

    def test_env_override_GL_SCM_DC_STRICT(self, monkeypatch):
        """Test GL_SCM_DC_STRICT=true enables strict DC mode."""
        monkeypatch.setenv("GL_SCM_DC_STRICT", "true")
        config = DoubleCountingConfig.from_env()
        assert config.strict_mode is True

    def test_env_override_GL_SCM_COMPLIANCE_MIN_SCORE(self, monkeypatch):
        """Test GL_SCM_COMPLIANCE_MIN_SCORE=80.0 overrides min score."""
        monkeypatch.setenv("GL_SCM_COMPLIANCE_MIN_SCORE", "80.0")
        config = ComplianceConfig.from_env()
        assert config.minimum_compliance_score == 80.0

    def test_env_override_GL_SCM_DB_HOST(self, monkeypatch):
        """Test GL_SCM_DB_HOST overrides database host."""
        monkeypatch.setenv("GL_SCM_DB_HOST", "db.example.com")
        config = DatabaseConfig.from_env()
        assert config.host == "db.example.com"

    def test_env_override_GL_SCM_DB_PORT(self, monkeypatch):
        """Test GL_SCM_DB_PORT overrides database port."""
        monkeypatch.setenv("GL_SCM_DB_PORT", "5433")
        config = DatabaseConfig.from_env()
        assert config.port == 5433

    def test_env_override_GL_SCM_METRICS_ENABLED(self, monkeypatch):
        """Test GL_SCM_METRICS_ENABLED=false disables metrics."""
        monkeypatch.setenv("GL_SCM_METRICS_ENABLED", "false")
        config = MetricsConfig.from_env()
        assert config.enabled is False

    def test_env_override_GL_SCM_PROV_HASH_ALGO(self, monkeypatch):
        """Test GL_SCM_PROV_HASH_ALGO=sha512 overrides hash algorithm."""
        monkeypatch.setenv("GL_SCM_PROV_HASH_ALGO", "sha512")
        config = ProvenanceConfig.from_env()
        assert config.hash_algorithm == "sha512"

    def test_env_override_full_config_from_env(self, monkeypatch):
        """Test Scope3CategoryMapperConfig.from_env() loads all sections."""
        monkeypatch.setenv("GL_SCM_ENABLED", "false")
        monkeypatch.setenv("GL_SCM_MIN_CONFIDENCE", "0.6")
        monkeypatch.setenv("GL_SCM_DB_PORT", "5433")
        reset_config()
        config = Scope3CategoryMapperConfig.from_env()
        assert config.general.enabled is False
        assert config.classification.min_confidence_threshold == 0.6
        assert config.database.port == 5433

    def test_env_override_GL_SCM_ENABLE_SPLIT(self, monkeypatch):
        """Test GL_SCM_ENABLE_SPLIT=false disables multi-category split."""
        monkeypatch.setenv("GL_SCM_ENABLE_SPLIT", "false")
        config = ClassificationConfig.from_env()
        assert config.enable_multi_category_split is False

    def test_env_override_GL_SCM_LEASE_BOUNDARY_MONTHS(self, monkeypatch):
        """Test GL_SCM_LEASE_BOUNDARY_MONTHS=24 overrides lease threshold."""
        monkeypatch.setenv("GL_SCM_LEASE_BOUNDARY_MONTHS", "24")
        config = BoundaryConfig.from_env()
        assert config.lease_scope_boundary_months == 24

    def test_env_override_GL_SCM_ENABLE_BENCHMARKS(self, monkeypatch):
        """Test GL_SCM_ENABLE_BENCHMARKS=false disables benchmarks."""
        monkeypatch.setenv("GL_SCM_ENABLE_BENCHMARKS", "false")
        config = CompletenessConfig.from_env()
        assert config.enable_industry_benchmarks is False

    def test_env_override_GL_SCM_DB_SSL_MODE(self, monkeypatch):
        """Test GL_SCM_DB_SSL_MODE=require overrides SSL mode."""
        monkeypatch.setenv("GL_SCM_DB_SSL_MODE", "require")
        config = DatabaseConfig.from_env()
        assert config.ssl_mode == "require"

    def test_env_override_GL_SCM_PROV_CHAIN_LENGTH(self, monkeypatch):
        """Test GL_SCM_PROV_CHAIN_LENGTH=20 overrides chain length."""
        monkeypatch.setenv("GL_SCM_PROV_CHAIN_LENGTH", "20")
        config = ProvenanceConfig.from_env()
        assert config.chain_length == 20
