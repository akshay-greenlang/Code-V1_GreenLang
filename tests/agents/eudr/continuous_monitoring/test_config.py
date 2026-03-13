# -*- coding: utf-8 -*-
"""
Unit tests for ContinuousMonitoringConfig - AGENT-EUDR-033

Tests configuration defaults, environment variable overrides, singleton
pattern, risk level classification, change impact weights, deforestation
severity classification, and configuration validation.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

import os
from decimal import Decimal
from unittest.mock import patch

import pytest

from greenlang.agents.eudr.continuous_monitoring.config import (
    ContinuousMonitoringConfig,
    get_config,
    reset_config,
    _env,
    _env_int,
    _env_float,
    _env_bool,
    _env_decimal,
    _ENV_PREFIX,
)


class TestConfigDefaults:
    """Test all configuration default values."""

    def test_db_host_default(self, sample_config):
        assert sample_config.db_host == "localhost"

    def test_db_port_default(self, sample_config):
        assert sample_config.db_port == 5432

    def test_db_name_default(self, sample_config):
        assert sample_config.db_name == "greenlang"

    def test_db_user_default(self, sample_config):
        assert sample_config.db_user == "gl"

    def test_db_pool_min_default(self, sample_config):
        assert sample_config.db_pool_min == 2

    def test_db_pool_max_default(self, sample_config):
        assert sample_config.db_pool_max == 10

    def test_redis_host_default(self, sample_config):
        assert sample_config.redis_host == "localhost"

    def test_redis_port_default(self, sample_config):
        assert sample_config.redis_port == 6379

    def test_cache_ttl_default(self, sample_config):
        assert sample_config.cache_ttl == 3600

    def test_supply_chain_scan_interval_default(self, sample_config):
        assert sample_config.supply_chain_scan_interval_minutes == 60

    def test_deforestation_check_interval_default(self, sample_config):
        assert sample_config.deforestation_check_interval_minutes == 30

    def test_compliance_audit_interval_default(self, sample_config):
        assert sample_config.compliance_audit_interval_days == 30

    def test_change_detection_sensitivity_default(self, sample_config):
        assert sample_config.change_detection_sensitivity == Decimal("0.10")

    def test_risk_trend_window_days_default(self, sample_config):
        assert sample_config.risk_trend_window_days == 90

    def test_data_freshness_check_interval_default(self, sample_config):
        assert sample_config.data_freshness_check_interval_minutes == 60

    def test_regulatory_check_interval_default(self, sample_config):
        assert sample_config.regulatory_check_interval_hours == 24

    def test_investigation_auto_trigger_threshold(self, sample_config):
        assert sample_config.investigation_auto_trigger_threshold == Decimal("0.75")

    def test_risk_level_thresholds(self, sample_config):
        assert sample_config.risk_level_negligible_max == Decimal("15")
        assert sample_config.risk_level_low_max == Decimal("30")
        assert sample_config.risk_level_moderate_max == Decimal("60")
        assert sample_config.risk_level_high_max == Decimal("80")

    def test_compliance_pass_threshold(self, sample_config):
        assert sample_config.compliance_pass_threshold == Decimal("80.0")

    def test_data_stale_warning_hours(self, sample_config):
        assert sample_config.data_stale_warning_hours == 24

    def test_data_stale_critical_hours(self, sample_config):
        assert sample_config.data_stale_critical_hours == 72

    def test_batch_timeout_seconds_default(self, sample_config):
        assert sample_config.batch_timeout_seconds == 300

    def test_max_concurrent_default(self, sample_config):
        assert sample_config.max_concurrent == 10

    def test_data_refresh_batch_size_default(self, sample_config):
        assert sample_config.data_refresh_batch_size == 100

    def test_retention_years_default(self, sample_config):
        assert sample_config.retention_years == 5

    def test_metrics_prefix(self, sample_config):
        assert sample_config.metrics_prefix == "gl_eudr_cm_"

    def test_provenance_enabled(self, sample_config):
        assert sample_config.provenance_enabled is True

    def test_rate_limits(self, sample_config):
        assert sample_config.rate_limit_anonymous == 10
        assert sample_config.rate_limit_basic == 30
        assert sample_config.rate_limit_standard == 100
        assert sample_config.rate_limit_premium == 500
        assert sample_config.rate_limit_admin == 2000

    def test_circuit_breaker_defaults(self, sample_config):
        assert sample_config.circuit_breaker_failure_threshold == 5
        assert sample_config.circuit_breaker_reset_timeout == 60

    def test_regulatory_sources_configured(self, sample_config):
        assert "eur-lex" in sample_config.regulatory_sources
        assert "eu-commission" in sample_config.regulatory_sources
        assert "national-authorities" in sample_config.regulatory_sources

    def test_regulatory_notification_channels_configured(self, sample_config):
        assert len(sample_config.regulatory_notification_channels) >= 1


class TestConfigEnvOverrides:
    """Test environment variable override functionality."""

    def test_env_prefix(self):
        assert _ENV_PREFIX == "GL_EUDR_CM_"

    def test_env_int_override(self):
        with patch.dict(os.environ, {"GL_EUDR_CM_DB_PORT": "5433"}):
            cfg = ContinuousMonitoringConfig()
            assert cfg.db_port == 5433

    def test_env_bool_override(self):
        with patch.dict(os.environ, {"GL_EUDR_CM_PROVENANCE_ENABLED": "false"}):
            cfg = ContinuousMonitoringConfig()
            assert cfg.provenance_enabled is False

    def test_env_decimal_override(self):
        with patch.dict(os.environ, {"GL_EUDR_CM_INVEST_AUTO_TRIGGER_THRESHOLD": "0.85"}):
            cfg = ContinuousMonitoringConfig()
            assert cfg.investigation_auto_trigger_threshold == Decimal("0.85")

    def test_env_helper_returns_default(self):
        assert _env("NONEXISTENT_KEY_XYZ", "default") == "default"

    def test_env_int_returns_default(self):
        assert _env_int("NONEXISTENT_KEY_XYZ", 42) == 42

    def test_env_float_returns_default(self):
        assert _env_float("NONEXISTENT_KEY_XYZ", 3.14) == 3.14

    def test_env_bool_returns_default(self):
        assert _env_bool("NONEXISTENT_KEY_XYZ", True) is True

    def test_env_bool_false_default(self):
        assert _env_bool("NONEXISTENT_KEY_XYZ", False) is False

    def test_env_decimal_returns_default(self):
        assert _env_decimal("NONEXISTENT_KEY_XYZ", "1.23") == Decimal("1.23")

    def test_env_bool_truthy_values(self):
        for val in ("true", "1", "yes"):
            with patch.dict(os.environ, {"GL_EUDR_CM_TEST_BOOL": val}):
                assert _env_bool("TEST_BOOL", False) is True

    def test_env_bool_falsy_values(self):
        for val in ("false", "0", "no"):
            with patch.dict(os.environ, {"GL_EUDR_CM_TEST_BOOL": val}):
                assert _env_bool("TEST_BOOL", True) is False

    def test_env_string_override(self):
        with patch.dict(os.environ, {"GL_EUDR_CM_DB_HOST": "db.production.local"}):
            cfg = ContinuousMonitoringConfig()
            assert cfg.db_host == "db.production.local"

    def test_env_override_supply_chain_interval(self):
        with patch.dict(os.environ, {"GL_EUDR_CM_SC_SCAN_INTERVAL_MINUTES": "30"}):
            cfg = ContinuousMonitoringConfig()
            assert cfg.supply_chain_scan_interval_minutes == 30


class TestConfigSingleton:
    """Test thread-safe singleton pattern."""

    def test_get_config_returns_instance(self):
        cfg = get_config()
        assert isinstance(cfg, ContinuousMonitoringConfig)

    def test_get_config_returns_same_instance(self):
        cfg1 = get_config()
        cfg2 = get_config()
        assert cfg1 is cfg2

    def test_reset_config_clears_singleton(self):
        cfg1 = get_config()
        reset_config()
        cfg2 = get_config()
        assert cfg1 is not cfg2

    def test_reset_config_allows_new_env_values(self):
        cfg1 = get_config()
        reset_config()
        with patch.dict(os.environ, {"GL_EUDR_CM_DB_PORT": "9999"}):
            cfg2 = get_config()
            assert cfg2.db_port == 9999
        assert cfg1.db_port != 9999


class TestConfigMethods:
    """Test configuration helper methods."""

    def test_get_risk_level_negligible(self, sample_config):
        assert sample_config.get_risk_level(Decimal("10")) == "negligible"

    def test_get_risk_level_low(self, sample_config):
        assert sample_config.get_risk_level(Decimal("20")) == "low"

    def test_get_risk_level_moderate(self, sample_config):
        assert sample_config.get_risk_level(Decimal("40")) == "moderate"

    def test_get_risk_level_high(self, sample_config):
        assert sample_config.get_risk_level(Decimal("70")) == "high"

    def test_get_risk_level_critical(self, sample_config):
        assert sample_config.get_risk_level(Decimal("90")) == "critical"

    def test_get_risk_level_zero(self, sample_config):
        assert sample_config.get_risk_level(Decimal("0")) == "negligible"

    def test_get_risk_level_hundred(self, sample_config):
        assert sample_config.get_risk_level(Decimal("100")) == "critical"

    def test_get_risk_level_at_high_boundary(self, sample_config):
        assert sample_config.get_risk_level(Decimal("80")) == "high"

    def test_get_change_impact_weights_keys(self, sample_config):
        weights = sample_config.get_change_impact_weights()
        assert "compliance" in weights
        assert "risk" in weights
        assert "supply_chain" in weights
        assert "regulatory" in weights

    def test_get_change_impact_weights_sum_to_one(self, sample_config):
        weights = sample_config.get_change_impact_weights()
        total = sum(weights.values())
        assert total == Decimal("1.00")

    def test_get_change_impact_weights_compliance_value(self, sample_config):
        weights = sample_config.get_change_impact_weights()
        assert weights["compliance"] == Decimal("0.35")

    def test_get_change_impact_weights_risk_value(self, sample_config):
        weights = sample_config.get_change_impact_weights()
        assert weights["risk"] == Decimal("0.30")

    def test_get_change_impact_weights_supply_chain_value(self, sample_config):
        weights = sample_config.get_change_impact_weights()
        assert weights["supply_chain"] == Decimal("0.20")

    def test_get_change_impact_weights_regulatory_value(self, sample_config):
        weights = sample_config.get_change_impact_weights()
        assert weights["regulatory"] == Decimal("0.15")

    def test_get_deforestation_severity_negligible(self, sample_config):
        assert sample_config.get_deforestation_severity(Decimal("0")) == "negligible"

    def test_get_deforestation_severity_moderate(self, sample_config):
        assert sample_config.get_deforestation_severity(Decimal("5")) == "moderate"

    def test_get_deforestation_severity_high(self, sample_config):
        assert sample_config.get_deforestation_severity(Decimal("25")) == "high"

    def test_get_deforestation_severity_critical(self, sample_config):
        assert sample_config.get_deforestation_severity(Decimal("60")) == "critical"

    def test_get_deforestation_severity_at_high_boundary(self, sample_config):
        assert sample_config.get_deforestation_severity(Decimal("10")) == "high"

    def test_get_deforestation_severity_at_critical_boundary(self, sample_config):
        assert sample_config.get_deforestation_severity(Decimal("50")) == "critical"
