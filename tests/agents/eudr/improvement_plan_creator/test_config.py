# -*- coding: utf-8 -*-
"""
Unit tests for ImprovementPlanCreatorConfig - AGENT-EUDR-035

Tests default values, environment variable overrides, singleton pattern,
validation logic, and all env helper functions.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

import os
import logging
from decimal import Decimal
from unittest.mock import patch

import pytest

from greenlang.agents.eudr.improvement_plan_creator.config import (
    ImprovementPlanCreatorConfig,
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
    """Test that default configuration values are correct."""

    def test_db_host_default(self, sample_config):
        assert sample_config.db_host == "localhost"

    def test_db_port_default(self, sample_config):
        assert sample_config.db_port == 5432

    def test_db_name_default(self, sample_config):
        assert sample_config.db_name == "greenlang"

    def test_db_user_default(self, sample_config):
        assert sample_config.db_user == "gl"

    def test_db_password_default(self, sample_config):
        assert sample_config.db_password == "gl"

    def test_db_pool_min_default(self, sample_config):
        assert sample_config.db_pool_min == 2

    def test_db_pool_max_default(self, sample_config):
        assert sample_config.db_pool_max == 10

    def test_redis_host_default(self, sample_config):
        assert sample_config.redis_host == "localhost"

    def test_redis_port_default(self, sample_config):
        assert sample_config.redis_port == 6379

    def test_redis_db_default(self, sample_config):
        assert sample_config.redis_db == 0

    def test_cache_ttl_default(self, sample_config):
        assert sample_config.cache_ttl == 3600

    def test_finding_staleness_days_default(self, sample_config):
        assert sample_config.finding_staleness_days == 90

    def test_gap_severity_critical_threshold_default(self, sample_config):
        assert sample_config.gap_severity_critical_threshold == Decimal("0.80")

    def test_gap_severity_high_threshold_default(self, sample_config):
        assert sample_config.gap_severity_high_threshold == Decimal("0.60")

    def test_gap_severity_medium_threshold_default(self, sample_config):
        assert sample_config.gap_severity_medium_threshold == Decimal("0.40")

    def test_max_actions_per_plan_default(self, sample_config):
        assert sample_config.max_actions_per_plan == 50

    def test_default_action_deadline_days_default(self, sample_config):
        assert sample_config.default_action_deadline_days == 30

    def test_five_whys_max_depth_default(self, sample_config):
        assert sample_config.five_whys_max_depth == 5

    def test_fishbone_max_categories_default(self, sample_config):
        assert sample_config.fishbone_max_categories == 8

    def test_risk_score_weight_default(self, sample_config):
        assert sample_config.risk_score_weight == Decimal("0.30")

    def test_compliance_impact_weight_default(self, sample_config):
        assert sample_config.compliance_impact_weight == Decimal("0.25")

    def test_resource_efficiency_weight_default(self, sample_config):
        assert sample_config.resource_efficiency_weight == Decimal("0.20")

    def test_stakeholder_impact_weight_default(self, sample_config):
        assert sample_config.stakeholder_impact_weight == Decimal("0.15")

    def test_time_sensitivity_weight_default(self, sample_config):
        assert sample_config.time_sensitivity_weight == Decimal("0.10")

    def test_milestone_check_interval_days_default(self, sample_config):
        assert sample_config.milestone_check_interval_days == 7

    def test_overdue_alert_threshold_days_default(self, sample_config):
        assert sample_config.overdue_alert_threshold_days == 7

    def test_max_stakeholders_per_action_default(self, sample_config):
        assert sample_config.max_stakeholders_per_action == 10

    def test_notification_channels_enabled_default(self, sample_config):
        assert sample_config.notification_channels_enabled is True

    def test_max_source_agents_default(self, sample_config):
        assert sample_config.max_source_agents == 20

    def test_smart_validation_enabled_default(self, sample_config):
        assert sample_config.smart_validation_enabled is True

    def test_provenance_enabled_default(self, sample_config):
        assert sample_config.provenance_enabled is True

    def test_metrics_prefix_default(self, sample_config):
        assert sample_config.metrics_prefix == "gl_eudr_ipc_"

    def test_rate_limit_anonymous_default(self, sample_config):
        assert sample_config.rate_limit_anonymous == 10

    def test_rate_limit_admin_default(self, sample_config):
        assert sample_config.rate_limit_admin == 2000

    def test_circuit_breaker_failure_threshold_default(self, sample_config):
        assert sample_config.circuit_breaker_failure_threshold == 5

    def test_circuit_breaker_reset_timeout_default(self, sample_config):
        assert sample_config.circuit_breaker_reset_timeout == 60

    def test_max_concurrent_default(self, sample_config):
        assert sample_config.max_concurrent == 10

    def test_batch_timeout_seconds_default(self, sample_config):
        assert sample_config.batch_timeout_seconds == 300

    def test_auto_escalation_enabled_default(self, sample_config):
        assert sample_config.auto_escalation_enabled is True


class TestConfigEnvOverride:
    """Test that environment variable overrides work correctly."""

    def test_env_override_db_host(self):
        with patch.dict(os.environ, {"GL_EUDR_IPC_DB_HOST": "my-db.example.com"}):
            cfg = ImprovementPlanCreatorConfig()
            assert cfg.db_host == "my-db.example.com"

    def test_env_override_db_port(self):
        with patch.dict(os.environ, {"GL_EUDR_IPC_DB_PORT": "5433"}):
            cfg = ImprovementPlanCreatorConfig()
            assert cfg.db_port == 5433

    def test_env_override_finding_staleness(self):
        with patch.dict(os.environ, {"GL_EUDR_IPC_FINDING_STALENESS_DAYS": "180"}):
            cfg = ImprovementPlanCreatorConfig()
            assert cfg.finding_staleness_days == 180

    def test_env_override_gap_critical_threshold(self):
        with patch.dict(os.environ, {"GL_EUDR_IPC_GAP_SEVERITY_CRITICAL": "0.85"}):
            cfg = ImprovementPlanCreatorConfig()
            assert cfg.gap_severity_critical_threshold == Decimal("0.85")

    def test_env_override_bool_true(self):
        with patch.dict(os.environ, {"GL_EUDR_IPC_PROVENANCE_ENABLED": "true"}):
            cfg = ImprovementPlanCreatorConfig()
            assert cfg.provenance_enabled is True

    def test_env_override_bool_false(self):
        with patch.dict(os.environ, {"GL_EUDR_IPC_PROVENANCE_ENABLED": "false"}):
            cfg = ImprovementPlanCreatorConfig()
            assert cfg.provenance_enabled is False

    def test_env_override_max_actions_per_plan(self):
        with patch.dict(os.environ, {"GL_EUDR_IPC_MAX_ACTIONS_PER_PLAN": "500"}):
            cfg = ImprovementPlanCreatorConfig()
            assert cfg.max_actions_per_plan == 500

    def test_env_override_max_concurrent(self):
        with patch.dict(os.environ, {"GL_EUDR_IPC_MAX_CONCURRENT": "20"}):
            cfg = ImprovementPlanCreatorConfig()
            assert cfg.max_concurrent == 20

    def test_env_override_milestone_check_interval(self):
        with patch.dict(os.environ, {"GL_EUDR_IPC_MILESTONE_CHECK_INTERVAL_DAYS": "14"}):
            cfg = ImprovementPlanCreatorConfig()
            assert cfg.milestone_check_interval_days == 14

    def test_env_override_escalation_threshold(self):
        with patch.dict(os.environ, {"GL_EUDR_IPC_ESCALATION_THRESHOLD_DAYS": "21"}):
            cfg = ImprovementPlanCreatorConfig()
            assert cfg.escalation_threshold_days == 21


class TestConfigSingleton:
    """Test singleton pattern via get_config()."""

    def test_get_config_returns_instance(self):
        cfg = get_config()
        assert isinstance(cfg, ImprovementPlanCreatorConfig)

    def test_get_config_returns_same_instance(self):
        cfg1 = get_config()
        cfg2 = get_config()
        assert cfg1 is cfg2

    def test_reset_config_clears_singleton(self):
        cfg1 = get_config()
        reset_config()
        cfg2 = get_config()
        assert cfg1 is not cfg2

    def test_reset_config_allows_env_change(self):
        cfg1 = get_config()
        original_port = cfg1.db_port
        reset_config()
        with patch.dict(os.environ, {"GL_EUDR_IPC_DB_PORT": "9999"}):
            cfg2 = get_config()
            assert cfg2.db_port == 9999


class TestConfigValidation:
    """Test config __post_init__ validation logic."""

    def test_pool_min_exceeds_max_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            cfg = ImprovementPlanCreatorConfig(
                db_pool_min=20,
                db_pool_max=5,
            )
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        pool_warnings = [m for m in warning_msgs if "pool min" in m]
        assert len(pool_warnings) >= 1

    def test_five_whys_depth_out_of_range_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            cfg = ImprovementPlanCreatorConfig(five_whys_max_depth=15)
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        depth_warnings = [m for m in warning_msgs if "5-Whys" in m or "depth" in m.lower()]
        assert len(depth_warnings) >= 1

    def test_priority_weights_sum_to_one_no_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            cfg = ImprovementPlanCreatorConfig()
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        weight_warnings = [m for m in warning_msgs if "weight" in m.lower()]
        assert len(weight_warnings) == 0

    def test_priority_weights_not_summing_to_one_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            cfg = ImprovementPlanCreatorConfig(
                risk_score_weight=Decimal("0.50"),
                compliance_impact_weight=Decimal("0.50"),
                resource_efficiency_weight=Decimal("0.50"),
                stakeholder_impact_weight=Decimal("0.50"),
                time_sensitivity_weight=Decimal("0.50"),
            )
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        weight_warnings = [m for m in warning_msgs if "weight" in m.lower()]
        assert len(weight_warnings) >= 1

    def test_gap_thresholds_valid_order_no_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            cfg = ImprovementPlanCreatorConfig(
                gap_severity_low_threshold=Decimal("0.20"),
                gap_severity_medium_threshold=Decimal("0.40"),
                gap_severity_high_threshold=Decimal("0.60"),
                gap_severity_critical_threshold=Decimal("0.80"),
            )
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        gap_warnings = [m for m in warning_msgs if "severity" in m.lower() and "threshold" in m.lower()]
        assert len(gap_warnings) == 0

    def test_gap_thresholds_invalid_order_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            cfg = ImprovementPlanCreatorConfig(
                gap_severity_low_threshold=Decimal("0.80"),
                gap_severity_medium_threshold=Decimal("0.60"),
                gap_severity_high_threshold=Decimal("0.40"),
                gap_severity_critical_threshold=Decimal("0.20"),
            )
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        gap_warnings = [m for m in warning_msgs if "severity" in m.lower() and "threshold" in m.lower()]
        assert len(gap_warnings) >= 1


class TestConfigEnvHelpers:
    """Test the module-level _env_* helper functions."""

    def test_env_returns_default_when_missing(self):
        result = _env("NONEXISTENT_KEY_12345", "default_val")
        assert result == "default_val"

    def test_env_returns_value_when_set(self):
        with patch.dict(os.environ, {"GL_EUDR_IPC_TEST_KEY": "hello"}):
            result = _env("TEST_KEY", "default")
            assert result == "hello"

    def test_env_int_returns_default(self):
        result = _env_int("NONEXISTENT_INT", 42)
        assert result == 42

    def test_env_int_returns_parsed(self):
        with patch.dict(os.environ, {"GL_EUDR_IPC_TEST_INT": "99"}):
            result = _env_int("TEST_INT", 0)
            assert result == 99

    def test_env_float_returns_default(self):
        result = _env_float("NONEXISTENT_FLOAT", 3.14)
        assert result == 3.14

    def test_env_float_returns_parsed(self):
        with patch.dict(os.environ, {"GL_EUDR_IPC_TEST_FLOAT": "2.71"}):
            result = _env_float("TEST_FLOAT", 0.0)
            assert result == 2.71

    def test_env_bool_returns_default_false(self):
        result = _env_bool("NONEXISTENT_BOOL", False)
        assert result is False

    def test_env_bool_true_values(self):
        for val in ("true", "1", "yes"):
            with patch.dict(os.environ, {"GL_EUDR_IPC_TEST_BOOL": val}):
                result = _env_bool("TEST_BOOL", False)
                assert result is True, f"Expected True for {val!r}"

    def test_env_bool_false_values(self):
        for val in ("false", "0", "no", "anything"):
            with patch.dict(os.environ, {"GL_EUDR_IPC_TEST_BOOL": val}):
                result = _env_bool("TEST_BOOL", True)
                assert result is False, f"Expected False for {val!r}"

    def test_env_decimal_returns_default(self):
        result = _env_decimal("NONEXISTENT_DEC", "99.99")
        assert result == Decimal("99.99")

    def test_env_decimal_returns_parsed(self):
        with patch.dict(os.environ, {"GL_EUDR_IPC_TEST_DEC": "42.5"}):
            result = _env_decimal("TEST_DEC", "0")
            assert result == Decimal("42.5")

    def test_env_prefix_is_correct(self):
        assert _ENV_PREFIX == "GL_EUDR_IPC_"


class TestConfigAttributes:
    """Test config has expected attributes."""

    def test_config_has_db_attributes(self, sample_config):
        assert hasattr(sample_config, 'db_host')
        assert hasattr(sample_config, 'db_port')
        assert hasattr(sample_config, 'db_name')

    def test_config_has_redis_attributes(self, sample_config):
        assert hasattr(sample_config, 'redis_host')
        assert hasattr(sample_config, 'redis_port')
        assert hasattr(sample_config, 'cache_ttl')

    def test_config_has_gap_severity_thresholds(self, sample_config):
        assert hasattr(sample_config, 'gap_severity_critical_threshold')
        assert hasattr(sample_config, 'gap_severity_high_threshold')
        assert hasattr(sample_config, 'gap_severity_medium_threshold')
        assert hasattr(sample_config, 'gap_severity_low_threshold')

    def test_config_has_prioritization_weights(self, sample_config):
        assert hasattr(sample_config, 'risk_score_weight')
        assert hasattr(sample_config, 'compliance_impact_weight')
        assert hasattr(sample_config, 'resource_efficiency_weight')

    def test_config_has_root_cause_settings(self, sample_config):
        assert hasattr(sample_config, 'five_whys_max_depth')
        assert hasattr(sample_config, 'fishbone_max_categories')

    def test_config_has_provenance_settings(self, sample_config):
        assert hasattr(sample_config, 'provenance_enabled')
        assert hasattr(sample_config, 'provenance_algorithm')
        assert sample_config.provenance_algorithm == "sha256"
