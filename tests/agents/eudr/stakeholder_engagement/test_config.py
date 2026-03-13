# -*- coding: utf-8 -*-
"""
Unit tests for StakeholderEngagementConfig - AGENT-EUDR-031

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

from greenlang.agents.eudr.stakeholder_engagement.config import (
    StakeholderEngagementConfig,
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

    # -- Database defaults --

    def test_db_host_default(self, sample_config):
        """Test default database host is localhost."""
        assert sample_config.db_host == "localhost"

    def test_db_port_default(self, sample_config):
        """Test default database port is 5432."""
        assert sample_config.db_port == 5432

    def test_db_name_default(self, sample_config):
        """Test default database name is greenlang."""
        assert sample_config.db_name == "greenlang"

    def test_db_user_default(self, sample_config):
        """Test default database user is gl."""
        assert sample_config.db_user == "gl"

    def test_db_password_default(self, sample_config):
        """Test default database password is gl."""
        assert sample_config.db_password == "gl"

    def test_db_pool_min_default(self, sample_config):
        """Test default database pool min is 2."""
        assert sample_config.db_pool_min == 2

    def test_db_pool_max_default(self, sample_config):
        """Test default database pool max is 10."""
        assert sample_config.db_pool_max == 10

    # -- Redis defaults --

    def test_redis_host_default(self, sample_config):
        """Test default Redis host is localhost."""
        assert sample_config.redis_host == "localhost"

    def test_redis_port_default(self, sample_config):
        """Test default Redis port is 6379."""
        assert sample_config.redis_port == 6379

    def test_redis_db_default(self, sample_config):
        """Test default Redis database is 0."""
        assert sample_config.redis_db == 0

    def test_redis_password_default(self, sample_config):
        """Test default Redis password is empty string."""
        assert sample_config.redis_password == ""

    def test_cache_ttl_default(self, sample_config):
        """Test default cache TTL is 3600 seconds."""
        assert sample_config.cache_ttl == 3600

    # -- Stakeholder mapping defaults --

    def test_max_stakeholders_per_operator_default(self, sample_config):
        """Test default max stakeholders per operator is 500."""
        assert sample_config.max_stakeholders_per_operator == 500

    def test_discovery_radius_km_default(self, sample_config):
        """Test default discovery radius is 50 km."""
        assert sample_config.discovery_radius_km == 50

    def test_enable_auto_discovery_default(self, sample_config):
        """Test default auto-discovery is True."""
        assert sample_config.enable_auto_discovery is True

    def test_rights_classification_enabled_default(self, sample_config):
        """Test default rights classification is enabled."""
        assert sample_config.rights_classification_enabled is True

    # -- FPIC defaults --

    def test_fpic_notification_period_days_default(self, sample_config):
        """Test default FPIC notification period is 30 days."""
        assert sample_config.fpic_notification_period_days == 30

    def test_fpic_deliberation_period_days_default(self, sample_config):
        """Test default FPIC deliberation period is 90 days."""
        assert sample_config.fpic_deliberation_period_days == 90

    def test_fpic_min_consultation_sessions_default(self, sample_config):
        """Test default minimum consultation sessions is 3."""
        assert sample_config.fpic_min_consultation_sessions == 3

    def test_fpic_independent_facilitator_required_default(self, sample_config):
        """Test default independent facilitator required is True."""
        assert sample_config.fpic_independent_facilitator_required is True

    def test_fpic_min_attendance_percentage_default(self, sample_config):
        """Test default minimum attendance percentage is 60%."""
        assert sample_config.fpic_min_attendance_percentage == Decimal("60")

    def test_fpic_consent_validity_years_default(self, sample_config):
        """Test default consent validity is 5 years."""
        assert sample_config.fpic_consent_validity_years == 5

    # -- Grievance defaults --

    def test_grievance_sla_critical_hours_default(self, sample_config):
        """Test default critical grievance SLA is 24 hours."""
        assert sample_config.grievance_sla_critical_hours == 24

    def test_grievance_sla_high_hours_default(self, sample_config):
        """Test default high grievance SLA is 72 hours."""
        assert sample_config.grievance_sla_high_hours == 72

    def test_grievance_sla_standard_days_default(self, sample_config):
        """Test default standard grievance SLA is 14 days."""
        assert sample_config.grievance_sla_standard_days == 14

    def test_grievance_sla_minor_days_default(self, sample_config):
        """Test default minor grievance SLA is 30 days."""
        assert sample_config.grievance_sla_minor_days == 30

    def test_grievance_appeal_window_days_default(self, sample_config):
        """Test default grievance appeal window is 30 days."""
        assert sample_config.grievance_appeal_window_days == 30

    def test_grievance_satisfaction_survey_default(self, sample_config):
        """Test default grievance satisfaction survey is enabled."""
        assert sample_config.grievance_satisfaction_survey is True

    # -- Consultation defaults --

    def test_consultation_min_notice_days_default(self, sample_config):
        """Test default consultation minimum notice is 14 days."""
        assert sample_config.consultation_min_notice_days == 14

    def test_consultation_quorum_percentage_default(self, sample_config):
        """Test default consultation quorum is 50%."""
        assert sample_config.consultation_quorum_percentage == Decimal("50")

    def test_consultation_require_minutes_default(self, sample_config):
        """Test default consultation requires minutes."""
        assert sample_config.consultation_require_minutes is True

    def test_consultation_require_attendance_default(self, sample_config):
        """Test default consultation requires attendance record."""
        assert sample_config.consultation_require_attendance is True

    # -- Communication defaults --

    def test_communication_max_batch_size_default(self, sample_config):
        """Test default communication max batch size is 100."""
        assert sample_config.communication_max_batch_size == 100

    def test_communication_retry_attempts_default(self, sample_config):
        """Test default communication retry attempts is 3."""
        assert sample_config.communication_retry_attempts == 3

    def test_communication_retry_delay_seconds_default(self, sample_config):
        """Test default communication retry delay is 60 seconds."""
        assert sample_config.communication_retry_delay_seconds == 60

    def test_communication_default_language_default(self, sample_config):
        """Test default communication language is en."""
        assert sample_config.communication_default_language == "en"

    # -- Assessment defaults --

    def test_assessment_min_score_default(self, sample_config):
        """Test default assessment minimum score is 0."""
        assert sample_config.assessment_min_score == Decimal("0")

    def test_assessment_max_score_default(self, sample_config):
        """Test default assessment maximum score is 100."""
        assert sample_config.assessment_max_score == Decimal("100")

    def test_assessment_passing_threshold_default(self, sample_config):
        """Test default assessment passing threshold is 60."""
        assert sample_config.assessment_passing_threshold == Decimal("60")

    def test_assessment_high_threshold_default(self, sample_config):
        """Test default assessment high threshold is 80."""
        assert sample_config.assessment_high_threshold == Decimal("80")

    # -- Report & compliance defaults --

    def test_report_format_default(self, sample_config):
        """Test default report format is json."""
        assert sample_config.report_format == "json"

    def test_report_retention_years_default(self, sample_config):
        """Test default report retention years is 5."""
        assert sample_config.report_retention_years == 5

    def test_include_provenance_default(self, sample_config):
        """Test default include provenance is True."""
        assert sample_config.include_provenance is True

    # -- Rate limiting defaults --

    def test_rate_limit_anonymous_default(self, sample_config):
        """Test default rate limit for anonymous is 10."""
        assert sample_config.rate_limit_anonymous == 10

    def test_rate_limit_basic_default(self, sample_config):
        """Test default rate limit for basic is 30."""
        assert sample_config.rate_limit_basic == 30

    def test_rate_limit_standard_default(self, sample_config):
        """Test default rate limit for standard is 100."""
        assert sample_config.rate_limit_standard == 100

    def test_rate_limit_premium_default(self, sample_config):
        """Test default rate limit for premium is 500."""
        assert sample_config.rate_limit_premium == 500

    def test_rate_limit_admin_default(self, sample_config):
        """Test default rate limit for admin is 2000."""
        assert sample_config.rate_limit_admin == 2000

    # -- Circuit breaker defaults --

    def test_circuit_breaker_failure_threshold_default(self, sample_config):
        """Test default circuit breaker failure threshold is 5."""
        assert sample_config.circuit_breaker_failure_threshold == 5

    def test_circuit_breaker_reset_timeout_default(self, sample_config):
        """Test default circuit breaker reset timeout is 60."""
        assert sample_config.circuit_breaker_reset_timeout == 60

    def test_circuit_breaker_half_open_max_default(self, sample_config):
        """Test default circuit breaker half open max is 3."""
        assert sample_config.circuit_breaker_half_open_max == 3

    # -- Batch processing defaults --

    def test_max_concurrent_default(self, sample_config):
        """Test default max concurrent is 10."""
        assert sample_config.max_concurrent == 10

    def test_batch_timeout_seconds_default(self, sample_config):
        """Test default batch timeout is 300 seconds."""
        assert sample_config.batch_timeout_seconds == 300

    # -- Provenance defaults --

    def test_provenance_enabled_default(self, sample_config):
        """Test default provenance enabled is True."""
        assert sample_config.provenance_enabled is True

    def test_provenance_algorithm_default(self, sample_config):
        """Test default provenance algorithm is sha256."""
        assert sample_config.provenance_algorithm == "sha256"

    def test_provenance_chain_enabled_default(self, sample_config):
        """Test default provenance chain enabled is True."""
        assert sample_config.provenance_chain_enabled is True

    def test_provenance_genesis_hash_default(self, sample_config):
        """Test default provenance genesis hash is 64 zeros."""
        assert sample_config.provenance_genesis_hash == "0" * 64

    # -- Metrics defaults --

    def test_metrics_enabled_default(self, sample_config):
        """Test default metrics enabled is True."""
        assert sample_config.metrics_enabled is True

    def test_metrics_prefix_default(self, sample_config):
        """Test default metrics prefix is gl_eudr_set_."""
        assert sample_config.metrics_prefix == "gl_eudr_set_"

    def test_log_level_default(self, sample_config):
        """Test default log level is INFO."""
        assert sample_config.log_level == "INFO"


class TestConfigEnvOverride:
    """Test that environment variable overrides work correctly."""

    def test_env_override_db_host(self):
        """Test environment override for database host."""
        with patch.dict(os.environ, {"GL_EUDR_SET_DB_HOST": "my-db.example.com"}):
            cfg = StakeholderEngagementConfig()
            assert cfg.db_host == "my-db.example.com"

    def test_env_override_db_port(self):
        """Test environment override for database port."""
        with patch.dict(os.environ, {"GL_EUDR_SET_DB_PORT": "5433"}):
            cfg = StakeholderEngagementConfig()
            assert cfg.db_port == 5433

    def test_env_override_max_stakeholders(self):
        """Test environment override for max stakeholders per operator."""
        with patch.dict(os.environ, {"GL_EUDR_SET_MAX_STAKEHOLDERS_PER_OPERATOR": "1000"}):
            cfg = StakeholderEngagementConfig()
            assert cfg.max_stakeholders_per_operator == 1000

    def test_env_override_discovery_radius(self):
        """Test environment override for discovery radius."""
        with patch.dict(os.environ, {"GL_EUDR_SET_DISCOVERY_RADIUS_KM": "100"}):
            cfg = StakeholderEngagementConfig()
            assert cfg.discovery_radius_km == 100

    def test_env_override_fpic_notification_period(self):
        """Test environment override for FPIC notification period."""
        with patch.dict(os.environ, {"GL_EUDR_SET_FPIC_NOTIFICATION_PERIOD_DAYS": "45"}):
            cfg = StakeholderEngagementConfig()
            assert cfg.fpic_notification_period_days == 45

    def test_env_override_fpic_deliberation_period(self):
        """Test environment override for FPIC deliberation period."""
        with patch.dict(os.environ, {"GL_EUDR_SET_FPIC_DELIBERATION_PERIOD_DAYS": "120"}):
            cfg = StakeholderEngagementConfig()
            assert cfg.fpic_deliberation_period_days == 120

    def test_env_override_fpic_min_attendance(self):
        """Test environment override for FPIC minimum attendance percentage."""
        with patch.dict(os.environ, {"GL_EUDR_SET_FPIC_MIN_ATTENDANCE_PERCENTAGE": "75"}):
            cfg = StakeholderEngagementConfig()
            assert cfg.fpic_min_attendance_percentage == Decimal("75")

    def test_env_override_grievance_sla_critical(self):
        """Test environment override for critical grievance SLA."""
        with patch.dict(os.environ, {"GL_EUDR_SET_GRIEVANCE_SLA_CRITICAL_HOURS": "12"}):
            cfg = StakeholderEngagementConfig()
            assert cfg.grievance_sla_critical_hours == 12

    def test_env_override_bool_true(self):
        """Test environment override for boolean value set to true."""
        with patch.dict(os.environ, {"GL_EUDR_SET_INCLUDE_PROVENANCE": "true"}):
            cfg = StakeholderEngagementConfig()
            assert cfg.include_provenance is True

    def test_env_override_bool_false(self):
        """Test environment override for boolean value set to false."""
        with patch.dict(os.environ, {"GL_EUDR_SET_INCLUDE_PROVENANCE": "false"}):
            cfg = StakeholderEngagementConfig()
            assert cfg.include_provenance is False

    def test_env_override_assessment_passing_threshold(self):
        """Test environment override for assessment passing threshold."""
        with patch.dict(os.environ, {"GL_EUDR_SET_ASSESSMENT_PASSING_THRESHOLD": "70"}):
            cfg = StakeholderEngagementConfig()
            assert cfg.assessment_passing_threshold == Decimal("70")

    def test_env_override_report_format(self):
        """Test environment override for report format."""
        with patch.dict(os.environ, {"GL_EUDR_SET_REPORT_FORMAT": "pdf"}):
            cfg = StakeholderEngagementConfig()
            assert cfg.report_format == "pdf"

    def test_env_override_report_retention_years(self):
        """Test environment override for report retention years."""
        with patch.dict(os.environ, {"GL_EUDR_SET_REPORT_RETENTION_YEARS": "7"}):
            cfg = StakeholderEngagementConfig()
            assert cfg.report_retention_years == 7

    def test_env_override_communication_retry_attempts(self):
        """Test environment override for communication retry attempts."""
        with patch.dict(os.environ, {"GL_EUDR_SET_COMMUNICATION_RETRY_ATTEMPTS": "5"}):
            cfg = StakeholderEngagementConfig()
            assert cfg.communication_retry_attempts == 5

    def test_env_override_consultation_quorum(self):
        """Test environment override for consultation quorum percentage."""
        with patch.dict(os.environ, {"GL_EUDR_SET_CONSULTATION_QUORUM_PERCENTAGE": "66.7"}):
            cfg = StakeholderEngagementConfig()
            assert cfg.consultation_quorum_percentage == Decimal("66.7")

    def test_env_override_metrics_prefix(self):
        """Test environment override for metrics prefix."""
        with patch.dict(os.environ, {"GL_EUDR_SET_METRICS_PREFIX": "custom_set_"}):
            cfg = StakeholderEngagementConfig()
            assert cfg.metrics_prefix == "custom_set_"


class TestConfigSingleton:
    """Test singleton pattern via get_config()."""

    def test_get_config_returns_instance(self):
        """Test that get_config returns a StakeholderEngagementConfig instance."""
        cfg = get_config()
        assert isinstance(cfg, StakeholderEngagementConfig)

    def test_get_config_returns_same_instance(self):
        """Test that get_config returns the same singleton instance."""
        cfg1 = get_config()
        cfg2 = get_config()
        assert cfg1 is cfg2

    def test_reset_config_clears_singleton(self):
        """Test that reset_config clears the singleton instance."""
        cfg1 = get_config()
        reset_config()
        cfg2 = get_config()
        assert cfg1 is not cfg2

    def test_reset_config_allows_env_change(self):
        """Test that reset_config allows environment changes to take effect."""
        cfg1 = get_config()
        original_port = cfg1.db_port
        reset_config()
        with patch.dict(os.environ, {"GL_EUDR_SET_DB_PORT": "9999"}):
            cfg2 = get_config()
            assert cfg2.db_port == 9999

    def test_get_config_thread_safe(self):
        """Test that get_config is safe for repeated access."""
        configs = [get_config() for _ in range(10)]
        assert all(c is configs[0] for c in configs)


class TestConfigValidation:
    """Test config __post_init__ validation logic."""

    def test_valid_config_no_warning(self, caplog):
        """Test that valid default config does not produce warnings."""
        with caplog.at_level(logging.WARNING):
            cfg = StakeholderEngagementConfig()
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        # Default config should not produce warnings
        assert len(warning_msgs) == 0

    def test_fpic_notification_period_below_minimum_warns(self, caplog):
        """Test that FPIC notification period below 14 days produces warning."""
        with caplog.at_level(logging.WARNING):
            cfg = StakeholderEngagementConfig()
            cfg.fpic_notification_period_days = 7
            cfg.__post_init__()
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        notification_warnings = [m for m in warning_msgs if "notification period" in m.lower()]
        assert len(notification_warnings) >= 1

    def test_fpic_deliberation_period_below_minimum_warns(self, caplog):
        """Test that FPIC deliberation period below 30 days produces warning."""
        with caplog.at_level(logging.WARNING):
            cfg = StakeholderEngagementConfig()
            cfg.fpic_deliberation_period_days = 14
            cfg.__post_init__()
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        deliberation_warnings = [m for m in warning_msgs if "deliberation period" in m.lower()]
        assert len(deliberation_warnings) >= 1

    def test_fpic_min_attendance_above_100_warns(self, caplog):
        """Test that FPIC min attendance above 100% produces warning."""
        with caplog.at_level(logging.WARNING):
            cfg = StakeholderEngagementConfig()
            cfg.fpic_min_attendance_percentage = Decimal("110")
            cfg.__post_init__()
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        attendance_warnings = [m for m in warning_msgs if "attendance" in m.lower()]
        assert len(attendance_warnings) >= 1

    def test_fpic_min_attendance_below_zero_warns(self, caplog):
        """Test that FPIC min attendance below 0% produces warning."""
        with caplog.at_level(logging.WARNING):
            cfg = StakeholderEngagementConfig()
            cfg.fpic_min_attendance_percentage = Decimal("-10")
            cfg.__post_init__()
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        attendance_warnings = [m for m in warning_msgs if "attendance" in m.lower()]
        assert len(attendance_warnings) >= 1

    def test_grievance_sla_critical_below_one_warns(self, caplog):
        """Test that critical grievance SLA below 1 hour produces warning."""
        with caplog.at_level(logging.WARNING):
            cfg = StakeholderEngagementConfig()
            cfg.grievance_sla_critical_hours = 0
            cfg.__post_init__()
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        sla_warnings = [m for m in warning_msgs if "SLA" in m or "sla" in m.lower()]
        assert len(sla_warnings) >= 1

    def test_assessment_passing_above_max_warns(self, caplog):
        """Test that assessment passing threshold above max score warns."""
        with caplog.at_level(logging.WARNING):
            cfg = StakeholderEngagementConfig()
            cfg.assessment_passing_threshold = Decimal("150")
            cfg.__post_init__()
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        threshold_warnings = [m for m in warning_msgs if "threshold" in m.lower()]
        assert len(threshold_warnings) >= 1

    def test_assessment_passing_below_zero_warns(self, caplog):
        """Test that assessment passing threshold below 0 warns."""
        with caplog.at_level(logging.WARNING):
            cfg = StakeholderEngagementConfig()
            cfg.assessment_passing_threshold = Decimal("-5")
            cfg.__post_init__()
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        threshold_warnings = [m for m in warning_msgs if "threshold" in m.lower()]
        assert len(threshold_warnings) >= 1

    def test_pool_min_exceeds_max_warns(self, caplog):
        """Test that pool min exceeding max produces warning."""
        with caplog.at_level(logging.WARNING):
            cfg = StakeholderEngagementConfig()
            cfg.db_pool_min = 20
            cfg.db_pool_max = 5
            cfg.__post_init__()
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        pool_warnings = [m for m in warning_msgs if "pool" in m.lower()]
        assert len(pool_warnings) >= 1

    def test_retention_years_below_five_warns(self, caplog):
        """Test that retention years below 5 produces warning (EUDR Article 31)."""
        with caplog.at_level(logging.WARNING):
            cfg = StakeholderEngagementConfig()
            cfg.report_retention_years = 3
            cfg.__post_init__()
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        retention_warnings = [m for m in warning_msgs if "retention" in m.lower()]
        assert len(retention_warnings) >= 1

    def test_max_stakeholders_below_one_warns(self, caplog):
        """Test that max stakeholders below 1 produces warning."""
        with caplog.at_level(logging.WARNING):
            cfg = StakeholderEngagementConfig()
            cfg.max_stakeholders_per_operator = 0
            cfg.__post_init__()
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        stakeholder_warnings = [m for m in warning_msgs if "stakeholder" in m.lower()]
        assert len(stakeholder_warnings) >= 1

    def test_invalid_report_format_warns(self, caplog):
        """Test that invalid report format produces warning."""
        with caplog.at_level(logging.WARNING):
            cfg = StakeholderEngagementConfig()
            cfg.report_format = "invalid_format"
            cfg.__post_init__()
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        format_warnings = [m for m in warning_msgs if "format" in m.lower()]
        assert len(format_warnings) >= 1

    def test_consultation_quorum_above_100_warns(self, caplog):
        """Test that consultation quorum above 100% produces warning."""
        with caplog.at_level(logging.WARNING):
            cfg = StakeholderEngagementConfig()
            cfg.consultation_quorum_percentage = Decimal("120")
            cfg.__post_init__()
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        quorum_warnings = [m for m in warning_msgs if "quorum" in m.lower()]
        assert len(quorum_warnings) >= 1

    def test_negative_discovery_radius_warns(self, caplog):
        """Test that negative discovery radius produces warning."""
        with caplog.at_level(logging.WARNING):
            cfg = StakeholderEngagementConfig()
            cfg.discovery_radius_km = -10
            cfg.__post_init__()
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        radius_warnings = [m for m in warning_msgs if "radius" in m.lower()]
        assert len(radius_warnings) >= 1


class TestConfigEnvHelpers:
    """Test the module-level _env_* helper functions."""

    def test_env_returns_default_when_missing(self):
        """Test _env returns default when environment variable is missing."""
        result = _env("NONEXISTENT_KEY_12345", "default_val")
        assert result == "default_val"

    def test_env_returns_value_when_set(self):
        """Test _env returns value when environment variable is set."""
        with patch.dict(os.environ, {"GL_EUDR_SET_TEST_KEY": "hello"}):
            result = _env("TEST_KEY", "default")
            assert result == "hello"

    def test_env_int_returns_default(self):
        """Test _env_int returns default when environment variable is missing."""
        result = _env_int("NONEXISTENT_INT", 42)
        assert result == 42

    def test_env_int_returns_parsed(self):
        """Test _env_int returns parsed integer value."""
        with patch.dict(os.environ, {"GL_EUDR_SET_TEST_INT": "99"}):
            result = _env_int("TEST_INT", 0)
            assert result == 99

    def test_env_float_returns_default(self):
        """Test _env_float returns default when environment variable is missing."""
        result = _env_float("NONEXISTENT_FLOAT", 3.14)
        assert result == 3.14

    def test_env_float_returns_parsed(self):
        """Test _env_float returns parsed float value."""
        with patch.dict(os.environ, {"GL_EUDR_SET_TEST_FLOAT": "2.71"}):
            result = _env_float("TEST_FLOAT", 0.0)
            assert result == 2.71

    def test_env_bool_returns_default_false(self):
        """Test _env_bool returns default False when environment variable is missing."""
        result = _env_bool("NONEXISTENT_BOOL", False)
        assert result is False

    def test_env_bool_true_values(self):
        """Test _env_bool recognizes various true values."""
        for val in ("true", "1", "yes"):
            with patch.dict(os.environ, {"GL_EUDR_SET_TEST_BOOL": val}):
                result = _env_bool("TEST_BOOL", False)
                assert result is True, f"Expected True for {val!r}"

    def test_env_bool_false_values(self):
        """Test _env_bool recognizes various false values."""
        for val in ("false", "0", "no", "anything"):
            with patch.dict(os.environ, {"GL_EUDR_SET_TEST_BOOL": val}):
                result = _env_bool("TEST_BOOL", True)
                assert result is False, f"Expected False for {val!r}"

    def test_env_decimal_returns_default(self):
        """Test _env_decimal returns default when environment variable is missing."""
        result = _env_decimal("NONEXISTENT_DEC", "99.99")
        assert result == Decimal("99.99")

    def test_env_decimal_returns_parsed(self):
        """Test _env_decimal returns parsed Decimal value."""
        with patch.dict(os.environ, {"GL_EUDR_SET_TEST_DEC": "42.5"}):
            result = _env_decimal("TEST_DEC", "0")
            assert result == Decimal("42.5")

    def test_env_prefix_is_correct(self):
        """Test that _ENV_PREFIX is GL_EUDR_SET_."""
        assert _ENV_PREFIX == "GL_EUDR_SET_"
