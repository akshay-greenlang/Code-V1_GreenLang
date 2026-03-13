# -*- coding: utf-8 -*-
"""
Unit tests for AnnualReviewSchedulerConfig - AGENT-EUDR-034

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

from greenlang.agents.eudr.annual_review_scheduler.config import (
    AnnualReviewSchedulerConfig,
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

    def test_review_cycle_duration_days_default(self, sample_config):
        assert sample_config.review_cycle_duration_days == 120

    def test_preparation_phase_days_default(self, sample_config):
        assert sample_config.preparation_phase_days == 14

    def test_data_collection_phase_days_default(self, sample_config):
        assert sample_config.data_collection_phase_days == 30

    def test_analysis_phase_days_default(self, sample_config):
        assert sample_config.analysis_phase_days == 21

    def test_review_meeting_phase_days_default(self, sample_config):
        assert sample_config.review_meeting_phase_days == 7

    def test_remediation_phase_days_default(self, sample_config):
        assert sample_config.remediation_phase_days == 30

    def test_sign_off_phase_days_default(self, sample_config):
        assert sample_config.sign_off_phase_days == 7

    def test_deadline_warning_days_default(self, sample_config):
        assert sample_config.deadline_warning_days == 7

    def test_deadline_critical_days_default(self, sample_config):
        assert sample_config.deadline_critical_days == 3

    def test_max_review_cycles_per_operator_default(self, sample_config):
        assert sample_config.max_review_cycles_per_operator == 5

    def test_notification_retry_max_default(self, sample_config):
        assert sample_config.notification_retry_max == 3

    def test_notification_retry_delay_seconds_default(self, sample_config):
        assert sample_config.notification_retry_delay_seconds == 300

    def test_year_comparison_lookback_default(self, sample_config):
        assert sample_config.year_comparison_lookback == 3

    def test_compliance_rate_target_default(self, sample_config):
        assert sample_config.compliance_rate_target == Decimal("95.00")

    def test_risk_score_improvement_target_default(self, sample_config):
        assert sample_config.risk_score_improvement_target == Decimal("10.00")

    def test_provenance_enabled_default(self, sample_config):
        assert sample_config.provenance_enabled is True

    def test_metrics_prefix_default(self, sample_config):
        assert sample_config.metrics_prefix == "gl_eudr_ars_"

    def test_rate_limit_anonymous_default(self, sample_config):
        assert sample_config.rate_limit_anonymous == 10

    def test_rate_limit_admin_default(self, sample_config):
        assert sample_config.rate_limit_admin == 2000

    def test_circuit_breaker_failure_threshold_default(self, sample_config):
        assert sample_config.circuit_breaker_failure_threshold == 5

    def test_circuit_breaker_reset_timeout_default(self, sample_config):
        assert sample_config.circuit_breaker_reset_timeout == 60

    def test_circuit_breaker_half_open_max_default(self, sample_config):
        assert sample_config.circuit_breaker_half_open_max == 3

    def test_auto_schedule_enabled_default(self, sample_config):
        assert sample_config.auto_schedule_enabled is True

    def test_calendar_sync_enabled_default(self, sample_config):
        assert sample_config.calendar_sync_enabled is True


class TestConfigEnvOverride:
    """Test that environment variable overrides work correctly."""

    def test_env_override_db_host(self):
        with patch.dict(os.environ, {"GL_EUDR_ARS_DB_HOST": "my-db.example.com"}):
            cfg = AnnualReviewSchedulerConfig()
            assert cfg.db_host == "my-db.example.com"

    def test_env_override_db_port(self):
        with patch.dict(os.environ, {"GL_EUDR_ARS_DB_PORT": "5433"}):
            cfg = AnnualReviewSchedulerConfig()
            assert cfg.db_port == 5433

    def test_env_override_review_cycle_duration(self):
        with patch.dict(os.environ, {"GL_EUDR_ARS_REVIEW_CYCLE_DURATION_DAYS": "90"}):
            cfg = AnnualReviewSchedulerConfig()
            assert cfg.review_cycle_duration_days == 90

    def test_env_override_compliance_rate_target(self):
        with patch.dict(os.environ, {"GL_EUDR_ARS_COMPLIANCE_RATE_TARGET": "98.00"}):
            cfg = AnnualReviewSchedulerConfig()
            assert cfg.compliance_rate_target == Decimal("98.00")

    def test_env_override_bool_true(self):
        with patch.dict(os.environ, {"GL_EUDR_ARS_PROVENANCE_ENABLED": "true"}):
            cfg = AnnualReviewSchedulerConfig()
            assert cfg.provenance_enabled is True

    def test_env_override_bool_false(self):
        with patch.dict(os.environ, {"GL_EUDR_ARS_AUTO_SCHEDULE_ENABLED": "false"}):
            cfg = AnnualReviewSchedulerConfig()
            assert cfg.auto_schedule_enabled is False

    def test_env_override_notification_retry_max(self):
        with patch.dict(os.environ, {"GL_EUDR_ARS_NOTIFICATION_RETRY_MAX": "5"}):
            cfg = AnnualReviewSchedulerConfig()
            assert cfg.notification_retry_max == 5

    def test_env_override_year_comparison_lookback(self):
        with patch.dict(os.environ, {"GL_EUDR_ARS_YEAR_COMPARISON_LOOKBACK": "5"}):
            cfg = AnnualReviewSchedulerConfig()
            assert cfg.year_comparison_lookback == 5


class TestConfigSingleton:
    """Test singleton pattern via get_config()."""

    def test_get_config_returns_instance(self):
        cfg = get_config()
        assert isinstance(cfg, AnnualReviewSchedulerConfig)

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
        with patch.dict(os.environ, {"GL_EUDR_ARS_DB_PORT": "9999"}):
            cfg2 = get_config()
            assert cfg2.db_port == 9999


class TestConfigValidation:
    """Test config __post_init__ validation logic."""

    def test_valid_phase_durations_no_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            cfg = AnnualReviewSchedulerConfig()
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        phase_warnings = [m for m in warning_msgs if "phase duration" in m.lower()]
        assert len(phase_warnings) == 0

    def test_total_phase_exceeds_cycle_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            cfg = AnnualReviewSchedulerConfig(
                review_cycle_duration_days=30,
                preparation_phase_days=14,
                data_collection_phase_days=30,
                analysis_phase_days=21,
            )
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        duration_warnings = [m for m in warning_msgs if "exceeds" in m.lower() or "total phase" in m.lower()]
        assert len(duration_warnings) >= 1

    def test_pool_min_exceeds_max_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            cfg = AnnualReviewSchedulerConfig(
                db_pool_min=20,
                db_pool_max=5,
            )
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        pool_warnings = [m for m in warning_msgs if "pool min" in m]
        assert len(pool_warnings) >= 1

    def test_warning_days_greater_than_critical_is_valid(self, caplog):
        with caplog.at_level(logging.WARNING):
            cfg = AnnualReviewSchedulerConfig(
                deadline_warning_days=7,
                deadline_critical_days=3,
            )
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        deadline_warnings = [m for m in warning_msgs if "warning days" in m.lower()]
        assert len(deadline_warnings) == 0

    def test_warning_days_less_than_critical_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            cfg = AnnualReviewSchedulerConfig(
                deadline_warning_days=2,
                deadline_critical_days=5,
            )
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        deadline_warnings = [m for m in warning_msgs if "warning" in m.lower() and "critical" in m.lower()]
        assert len(deadline_warnings) >= 1

    def test_compliance_target_above_100_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            cfg = AnnualReviewSchedulerConfig(
                compliance_rate_target=Decimal("105.00"),
            )
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        target_warnings = [m for m in warning_msgs if "compliance" in m.lower() and "100" in m]
        assert len(target_warnings) >= 1


class TestConfigEnvHelpers:
    """Test the module-level _env_* helper functions."""

    def test_env_returns_default_when_missing(self):
        result = _env("NONEXISTENT_KEY_12345", "default_val")
        assert result == "default_val"

    def test_env_returns_value_when_set(self):
        with patch.dict(os.environ, {"GL_EUDR_ARS_TEST_KEY": "hello"}):
            result = _env("TEST_KEY", "default")
            assert result == "hello"

    def test_env_int_returns_default(self):
        result = _env_int("NONEXISTENT_INT", 42)
        assert result == 42

    def test_env_int_returns_parsed(self):
        with patch.dict(os.environ, {"GL_EUDR_ARS_TEST_INT": "99"}):
            result = _env_int("TEST_INT", 0)
            assert result == 99

    def test_env_float_returns_default(self):
        result = _env_float("NONEXISTENT_FLOAT", 3.14)
        assert result == 3.14

    def test_env_float_returns_parsed(self):
        with patch.dict(os.environ, {"GL_EUDR_ARS_TEST_FLOAT": "2.71"}):
            result = _env_float("TEST_FLOAT", 0.0)
            assert result == 2.71

    def test_env_bool_returns_default_false(self):
        result = _env_bool("NONEXISTENT_BOOL", False)
        assert result is False

    def test_env_bool_true_values(self):
        for val in ("true", "1", "yes"):
            with patch.dict(os.environ, {"GL_EUDR_ARS_TEST_BOOL": val}):
                result = _env_bool("TEST_BOOL", False)
                assert result is True, f"Expected True for {val!r}"

    def test_env_bool_false_values(self):
        for val in ("false", "0", "no", "anything"):
            with patch.dict(os.environ, {"GL_EUDR_ARS_TEST_BOOL": val}):
                result = _env_bool("TEST_BOOL", True)
                assert result is False, f"Expected False for {val!r}"

    def test_env_decimal_returns_default(self):
        result = _env_decimal("NONEXISTENT_DEC", "99.99")
        assert result == Decimal("99.99")

    def test_env_decimal_returns_parsed(self):
        with patch.dict(os.environ, {"GL_EUDR_ARS_TEST_DEC": "42.5"}):
            result = _env_decimal("TEST_DEC", "0")
            assert result == Decimal("42.5")

    def test_env_prefix_is_correct(self):
        assert _ENV_PREFIX == "GL_EUDR_ARS_"
