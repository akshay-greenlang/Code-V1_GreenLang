# -*- coding: utf-8 -*-
"""
Unit tests for ThirdPartyAuditManagerConfig -- AGENT-EUDR-024

Tests configuration initialization, validation, environment variable
override, risk weight validation, SLA deadline validation, CAR sub-deadline
validation, escalation thresholds, evidence limits, certification scheme
validation, report settings, authority settings, singleton pattern, and
serialization.

Target: ~80 tests
Author: GreenLang Platform Team
Date: March 2026
"""

import os
from decimal import Decimal
from unittest.mock import patch

import pytest

from greenlang.agents.eudr.third_party_audit_manager.config import (
    ThirdPartyAuditManagerConfig,
    get_config,
    set_config,
    reset_config,
    _ENV_PREFIX,
)


class TestConfigDefaults:
    """Test default configuration values."""

    def test_default_database_url(self):
        cfg = ThirdPartyAuditManagerConfig()
        assert "postgresql" in cfg.database_url

    def test_default_redis_url(self):
        cfg = ThirdPartyAuditManagerConfig()
        assert "redis" in cfg.redis_url

    def test_default_log_level(self):
        cfg = ThirdPartyAuditManagerConfig()
        assert cfg.log_level == "INFO"

    def test_default_pool_size(self):
        cfg = ThirdPartyAuditManagerConfig()
        assert cfg.pool_size == 20

    def test_default_critical_sla_days(self):
        cfg = ThirdPartyAuditManagerConfig()
        assert cfg.critical_sla_days == 30

    def test_default_major_sla_days(self):
        cfg = ThirdPartyAuditManagerConfig()
        assert cfg.major_sla_days == 90

    def test_default_minor_sla_days(self):
        cfg = ThirdPartyAuditManagerConfig()
        assert cfg.minor_sla_days == 365

    def test_default_report_format(self):
        cfg = ThirdPartyAuditManagerConfig()
        assert cfg.default_report_format == "pdf"

    def test_default_report_language(self):
        cfg = ThirdPartyAuditManagerConfig()
        assert cfg.default_report_language == "en"

    def test_default_retention_years(self):
        cfg = ThirdPartyAuditManagerConfig()
        assert cfg.retention_years == 5

    def test_default_provenance_enabled(self):
        cfg = ThirdPartyAuditManagerConfig()
        assert cfg.enable_provenance is True

    def test_default_chain_algorithm(self):
        cfg = ThirdPartyAuditManagerConfig()
        assert cfg.chain_algorithm == "sha256"

    def test_default_metrics_enabled(self):
        cfg = ThirdPartyAuditManagerConfig()
        assert cfg.enable_metrics is True

    def test_default_metrics_prefix(self):
        cfg = ThirdPartyAuditManagerConfig()
        assert cfg.metrics_prefix == "gl_eudr_tam_"


class TestRiskWeightValidation:
    """Test risk weight validation."""

    def test_default_weights_sum_to_one(self):
        cfg = ThirdPartyAuditManagerConfig()
        total = (
            cfg.country_risk_weight
            + cfg.supplier_risk_weight
            + cfg.nc_history_weight
            + cfg.certification_gap_weight
            + cfg.deforestation_alert_weight
        )
        assert abs(total - Decimal("1")) < Decimal("0.001")

    def test_each_weight_positive(self):
        cfg = ThirdPartyAuditManagerConfig()
        assert cfg.country_risk_weight > Decimal("0")
        assert cfg.supplier_risk_weight > Decimal("0")
        assert cfg.nc_history_weight > Decimal("0")
        assert cfg.certification_gap_weight > Decimal("0")
        assert cfg.deforestation_alert_weight > Decimal("0")

    def test_weights_not_summing_to_one_raises(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            ThirdPartyAuditManagerConfig(
                country_risk_weight=Decimal("0.50"),
                supplier_risk_weight=Decimal("0.50"),
                nc_history_weight=Decimal("0.20"),
                certification_gap_weight=Decimal("0.15"),
                deforestation_alert_weight=Decimal("0.15"),
            )

    def test_zero_weight_raises(self):
        with pytest.raises(ValueError):
            ThirdPartyAuditManagerConfig(
                country_risk_weight=Decimal("0"),
                supplier_risk_weight=Decimal("0.40"),
                nc_history_weight=Decimal("0.20"),
                certification_gap_weight=Decimal("0.20"),
                deforestation_alert_weight=Decimal("0.20"),
            )

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError):
            ThirdPartyAuditManagerConfig(
                country_risk_weight=Decimal("-0.10"),
                supplier_risk_weight=Decimal("0.40"),
                nc_history_weight=Decimal("0.20"),
                certification_gap_weight=Decimal("0.30"),
                deforestation_alert_weight=Decimal("0.20"),
            )


class TestSLAValidation:
    """Test SLA deadline validation."""

    def test_sla_ascending_order(self):
        cfg = ThirdPartyAuditManagerConfig()
        assert cfg.critical_sla_days < cfg.major_sla_days < cfg.minor_sla_days

    def test_sla_invalid_order_raises(self):
        with pytest.raises(ValueError, match="SLA deadlines"):
            ThirdPartyAuditManagerConfig(
                critical_sla_days=100,
                major_sla_days=50,
                minor_sla_days=365,
            )

    def test_sla_zero_critical_raises(self):
        with pytest.raises(ValueError):
            ThirdPartyAuditManagerConfig(critical_sla_days=0)

    def test_sla_equal_values_raises(self):
        with pytest.raises(ValueError):
            ThirdPartyAuditManagerConfig(
                critical_sla_days=90,
                major_sla_days=90,
                minor_sla_days=365,
            )


class TestCARSubDeadlineValidation:
    """Test CAR sub-deadline validation."""

    def test_critical_acknowledge_within_sla(self):
        cfg = ThirdPartyAuditManagerConfig()
        assert cfg.critical_acknowledge_days < cfg.critical_sla_days

    def test_major_acknowledge_within_sla(self):
        cfg = ThirdPartyAuditManagerConfig()
        assert cfg.major_acknowledge_days < cfg.major_sla_days

    def test_critical_acknowledge_exceeding_sla_raises(self):
        with pytest.raises(ValueError):
            ThirdPartyAuditManagerConfig(
                critical_acknowledge_days=31,
                critical_sla_days=30,
            )


class TestFrequencyThresholdValidation:
    """Test frequency tier threshold validation."""

    def test_threshold_ordering(self):
        cfg = ThirdPartyAuditManagerConfig()
        assert cfg.standard_risk_threshold < cfg.high_risk_threshold

    def test_invalid_threshold_order_raises(self):
        with pytest.raises(ValueError, match="Frequency thresholds"):
            ThirdPartyAuditManagerConfig(
                high_risk_threshold=Decimal("30"),
                standard_risk_threshold=Decimal("70"),
            )

    def test_recency_cap_minimum_one(self):
        cfg = ThirdPartyAuditManagerConfig()
        assert cfg.recency_multiplier_cap >= Decimal("1")

    def test_recency_cap_below_one_raises(self):
        with pytest.raises(ValueError):
            ThirdPartyAuditManagerConfig(recency_multiplier_cap=Decimal("0.5"))


class TestEscalationValidation:
    """Test escalation threshold validation."""

    def test_escalation_ordering(self):
        cfg = ThirdPartyAuditManagerConfig()
        assert cfg.escalation_stage1_pct < cfg.escalation_stage2_pct

    def test_invalid_escalation_order_raises(self):
        with pytest.raises(ValueError, match="Escalation"):
            ThirdPartyAuditManagerConfig(
                escalation_stage1_pct=Decimal("0.95"),
                escalation_stage2_pct=Decimal("0.80"),
            )


class TestSchemeValidation:
    """Test certification scheme validation."""

    def test_default_schemes(self):
        cfg = ThirdPartyAuditManagerConfig()
        assert "fsc" in cfg.enabled_schemes
        assert "pefc" in cfg.enabled_schemes
        assert "rspo" in cfg.enabled_schemes
        assert "rainforest_alliance" in cfg.enabled_schemes
        assert "iscc" in cfg.enabled_schemes

    def test_invalid_scheme_raises(self):
        with pytest.raises(ValueError, match="Invalid certification scheme"):
            ThirdPartyAuditManagerConfig(enabled_schemes=["fsc", "unknown_scheme"])


class TestReportSettingsValidation:
    """Test report settings validation."""

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match="Invalid default_report_format"):
            ThirdPartyAuditManagerConfig(default_report_format="docx")

    def test_invalid_language_raises(self):
        with pytest.raises(ValueError, match="Invalid default_report_language"):
            ThirdPartyAuditManagerConfig(default_report_language="zh")

    def test_valid_formats(self):
        for fmt in ["pdf", "json", "html", "xlsx", "xml"]:
            cfg = ThirdPartyAuditManagerConfig(default_report_format=fmt)
            assert cfg.default_report_format == fmt

    def test_valid_languages(self):
        for lang in ["en", "fr", "de", "es", "pt"]:
            cfg = ThirdPartyAuditManagerConfig(default_report_language=lang)
            assert cfg.default_report_language == lang


class TestLogLevelValidation:
    """Test log level validation."""

    @pytest.mark.parametrize("level", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    def test_valid_log_levels(self, level):
        cfg = ThirdPartyAuditManagerConfig(log_level=level)
        assert cfg.log_level == level

    def test_invalid_log_level_raises(self):
        with pytest.raises(ValueError, match="Invalid log_level"):
            ThirdPartyAuditManagerConfig(log_level="TRACE")


class TestChainAlgorithmValidation:
    """Test chain algorithm validation."""

    @pytest.mark.parametrize("algo", ["sha256", "sha384", "sha512"])
    def test_valid_algorithms(self, algo):
        cfg = ThirdPartyAuditManagerConfig(chain_algorithm=algo)
        assert cfg.chain_algorithm == algo

    def test_invalid_algorithm_raises(self):
        with pytest.raises(ValueError, match="Invalid chain_algorithm"):
            ThirdPartyAuditManagerConfig(chain_algorithm="md5")


class TestSingletonPattern:
    """Test thread-safe singleton configuration."""

    def test_get_config_returns_instance(self):
        cfg = get_config()
        assert isinstance(cfg, ThirdPartyAuditManagerConfig)

    def test_get_config_returns_same_instance(self):
        cfg1 = get_config()
        cfg2 = get_config()
        assert cfg1 is cfg2

    def test_set_config_overrides(self):
        custom = ThirdPartyAuditManagerConfig(log_level="DEBUG")
        set_config(custom)
        cfg = get_config()
        assert cfg.log_level == "DEBUG"

    def test_reset_config_clears_singleton(self):
        set_config(ThirdPartyAuditManagerConfig(log_level="DEBUG"))
        reset_config()
        cfg = get_config()
        assert cfg.log_level == "INFO"


class TestEnvironmentOverrides:
    """Test environment variable override mechanism."""

    def test_env_prefix(self):
        assert _ENV_PREFIX == "GL_EUDR_TAM_"

    @patch.dict(os.environ, {"GL_EUDR_TAM_LOG_LEVEL": "DEBUG"})
    def test_env_log_level_override(self):
        reset_config()
        cfg = ThirdPartyAuditManagerConfig.from_env()
        assert cfg.log_level == "DEBUG"

    @patch.dict(os.environ, {"GL_EUDR_TAM_POOL_SIZE": "50"})
    def test_env_pool_size_override(self):
        cfg = ThirdPartyAuditManagerConfig.from_env()
        assert cfg.pool_size == 50

    @patch.dict(os.environ, {"GL_EUDR_TAM_CRITICAL_SLA_DAYS": "15"})
    def test_env_sla_override(self):
        cfg = ThirdPartyAuditManagerConfig.from_env()
        assert cfg.critical_sla_days == 15

    @patch.dict(os.environ, {"GL_EUDR_TAM_ENABLE_PROVENANCE": "false"})
    def test_env_provenance_disabled(self):
        cfg = ThirdPartyAuditManagerConfig.from_env()
        assert cfg.enable_provenance is False

    @patch.dict(os.environ, {"GL_EUDR_TAM_DEFAULT_REPORT_FORMAT": "json"})
    def test_env_report_format_override(self):
        cfg = ThirdPartyAuditManagerConfig.from_env()
        assert cfg.default_report_format == "json"


class TestConfigSerialization:
    """Test configuration serialization."""

    def test_to_dict(self):
        cfg = ThirdPartyAuditManagerConfig()
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert "critical_sla_days" in d
        assert "major_sla_days" in d
        assert "minor_sla_days" in d

    def test_to_dict_redacts_secrets(self):
        cfg = ThirdPartyAuditManagerConfig()
        d = cfg.to_dict(redact_secrets=True)
        assert d["database_url"] == "REDACTED"
        assert d["redis_url"] == "REDACTED"

    def test_to_dict_no_redaction(self):
        cfg = ThirdPartyAuditManagerConfig()
        d = cfg.to_dict(redact_secrets=False)
        assert "postgresql" in d["database_url"]
