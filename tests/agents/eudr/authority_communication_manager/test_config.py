# -*- coding: utf-8 -*-
"""
Unit tests for AuthorityCommunicationManagerConfig - AGENT-EUDR-040

Tests all default values, environment variable overrides, 27 member state
configurations, 24 EU language configurations, deadline defaults, penalty
ranges, encryption settings, GDPR compliance, singleton pattern, and
post-init validation warnings.

75+ tests.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

import os
from decimal import Decimal
from unittest.mock import patch

import pytest

from greenlang.agents.eudr.authority_communication_manager.config import (
    AuthorityCommunicationManagerConfig,
    EU_LANGUAGES,
    EU_MEMBER_STATES,
    get_config,
    reset_config,
    _env,
    _env_int,
    _env_float,
    _env_bool,
    _env_decimal,
    _ENV_PREFIX,
)


# ====================================================================
# Default Values
# ====================================================================


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

    def test_redis_password_default(self, sample_config):
        assert sample_config.redis_password == ""

    def test_cache_ttl_default(self, sample_config):
        assert sample_config.cache_ttl == 3600

    def test_message_queue_prefix_default(self, sample_config):
        assert sample_config.message_queue_prefix == "gl_eudr_acm_mq"

    def test_deadline_urgent_hours(self, sample_config):
        assert sample_config.deadline_urgent_hours == 24

    def test_deadline_normal_days(self, sample_config):
        assert sample_config.deadline_normal_days == 5

    def test_deadline_routine_days(self, sample_config):
        assert sample_config.deadline_routine_days == 15

    def test_deadline_appeal_days(self, sample_config):
        assert sample_config.deadline_appeal_days == 30

    def test_reminder_before_deadline_hours(self, sample_config):
        assert sample_config.reminder_before_deadline_hours == 48

    def test_escalation_after_deadline_hours(self, sample_config):
        assert sample_config.escalation_after_deadline_hours == 24

    def test_default_language(self, sample_config):
        assert sample_config.default_language == "en"

    def test_supported_languages_count(self, sample_config):
        assert len(sample_config.supported_languages) == 24

    def test_template_base_path(self, sample_config):
        assert sample_config.template_base_path == "/data/eudr/acm/templates"

    def test_template_fallback_language(self, sample_config):
        assert sample_config.template_fallback_language == "en"

    def test_smtp_host_default(self, sample_config):
        assert sample_config.email_smtp_host == "smtp.greenlang.io"

    def test_smtp_port_default(self, sample_config):
        assert sample_config.email_smtp_port == 587

    def test_email_from_address(self, sample_config):
        assert sample_config.email_from_address == "eudr-compliance@greenlang.io"

    def test_email_reply_to(self, sample_config):
        assert sample_config.email_reply_to == "eudr-support@greenlang.io"

    def test_api_notification_enabled(self, sample_config):
        assert sample_config.api_notification_enabled is True

    def test_portal_notification_enabled(self, sample_config):
        assert sample_config.portal_notification_enabled is True

    def test_encryption_enabled(self, sample_config):
        assert sample_config.encryption_enabled is True

    def test_encryption_algorithm(self, sample_config):
        assert sample_config.encryption_algorithm == "AES-256-GCM"

    def test_encryption_key_id(self, sample_config):
        assert sample_config.encryption_key_id == "eudr-acm-doc-key-v1"

    def test_encryption_key_rotation_days(self, sample_config):
        assert sample_config.encryption_key_rotation_days == 90

    def test_penalty_min_amount(self, sample_config):
        assert sample_config.penalty_min_amount == Decimal("1000")

    def test_penalty_max_amount(self, sample_config):
        assert sample_config.penalty_max_amount == Decimal("10000000")

    def test_penalty_currency(self, sample_config):
        assert sample_config.penalty_currency == "EUR"

    def test_inspection_notice_days(self, sample_config):
        assert sample_config.inspection_notice_days == 5

    def test_inspection_max_duration_hours(self, sample_config):
        assert sample_config.inspection_max_duration_hours == 48

    def test_inspection_follow_up_days(self, sample_config):
        assert sample_config.inspection_follow_up_days == 14

    def test_appeal_window_days(self, sample_config):
        assert sample_config.appeal_window_days == 60

    def test_appeal_max_extensions(self, sample_config):
        assert sample_config.appeal_max_extensions == 2

    def test_appeal_extension_days(self, sample_config):
        assert sample_config.appeal_extension_days == 30

    def test_gdpr_data_retention_days(self, sample_config):
        assert sample_config.gdpr_data_retention_days == 1825

    def test_gdpr_retention_equals_five_years(self, sample_config):
        assert sample_config.gdpr_data_retention_days == 5 * 365

    def test_gdpr_erasure_enabled(self, sample_config):
        assert sample_config.gdpr_erasure_enabled is True

    def test_gdpr_minimization_enabled(self, sample_config):
        assert sample_config.gdpr_minimization_enabled is True

    def test_gdpr_audit_log_retention(self, sample_config):
        assert sample_config.gdpr_audit_log_retention_days == 3650

    def test_rate_limit_anonymous(self, sample_config):
        assert sample_config.rate_limit_anonymous == 10

    def test_rate_limit_basic(self, sample_config):
        assert sample_config.rate_limit_basic == 30

    def test_rate_limit_standard(self, sample_config):
        assert sample_config.rate_limit_standard == 100

    def test_rate_limit_premium(self, sample_config):
        assert sample_config.rate_limit_premium == 500

    def test_rate_limit_admin(self, sample_config):
        assert sample_config.rate_limit_admin == 2000

    def test_circuit_breaker_failure_threshold(self, sample_config):
        assert sample_config.circuit_breaker_failure_threshold == 5

    def test_circuit_breaker_reset_timeout(self, sample_config):
        assert sample_config.circuit_breaker_reset_timeout == 60

    def test_circuit_breaker_half_open_max(self, sample_config):
        assert sample_config.circuit_breaker_half_open_max == 3

    def test_max_concurrent(self, sample_config):
        assert sample_config.max_concurrent == 10

    def test_batch_timeout(self, sample_config):
        assert sample_config.batch_timeout_seconds == 300

    def test_provenance_enabled(self, sample_config):
        assert sample_config.provenance_enabled is True

    def test_provenance_algorithm(self, sample_config):
        assert sample_config.provenance_algorithm == "sha256"

    def test_provenance_chain_enabled(self, sample_config):
        assert sample_config.provenance_chain_enabled is True

    def test_provenance_genesis_hash(self, sample_config):
        assert sample_config.provenance_genesis_hash == "0" * 64

    def test_metrics_enabled(self, sample_config):
        assert sample_config.metrics_enabled is True

    def test_metrics_prefix(self, sample_config):
        assert sample_config.metrics_prefix == "gl_eudr_acm_"

    def test_log_level(self, sample_config):
        assert sample_config.log_level == "INFO"


# ====================================================================
# Upstream Agent URLs
# ====================================================================


class TestUpstreamUrls:
    """Test upstream agent URL defaults."""

    def test_due_diligence_orchestrator_url(self, sample_config):
        assert "eudr-due-diligence" in sample_config.due_diligence_orchestrator_url

    def test_information_gathering_url(self, sample_config):
        assert "eudr-info-gathering" in sample_config.information_gathering_url

    def test_risk_assessment_url(self, sample_config):
        assert "eudr-risk-assessment" in sample_config.risk_assessment_url

    def test_mitigation_designer_url(self, sample_config):
        assert "eudr-mitigation" in sample_config.mitigation_designer_url

    def test_documentation_generator_url(self, sample_config):
        assert "eudr-documentation" in sample_config.documentation_generator_url


# ====================================================================
# EU Member States
# ====================================================================


class TestEUMemberStates:
    """Test 27 EU member state configurations."""

    def test_27_member_states(self):
        assert len(EU_MEMBER_STATES) == 27

    @pytest.mark.parametrize("code", [
        "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI",
        "FR", "DE", "GR", "HU", "IE", "IT", "LV", "LT", "LU",
        "MT", "NL", "PL", "PT", "RO", "SK", "SI", "ES", "SE",
    ])
    def test_member_state_exists(self, code):
        assert code in EU_MEMBER_STATES

    @pytest.mark.parametrize("code", list(EU_MEMBER_STATES.keys()))
    def test_member_state_has_name(self, code):
        assert "name" in EU_MEMBER_STATES[code]
        assert len(EU_MEMBER_STATES[code]["name"]) > 0

    @pytest.mark.parametrize("code", list(EU_MEMBER_STATES.keys()))
    def test_member_state_has_authority(self, code):
        assert "authority" in EU_MEMBER_STATES[code]
        assert len(EU_MEMBER_STATES[code]["authority"]) > 0

    @pytest.mark.parametrize("code", list(EU_MEMBER_STATES.keys()))
    def test_member_state_has_language(self, code):
        assert "language" in EU_MEMBER_STATES[code]

    @pytest.mark.parametrize("code", list(EU_MEMBER_STATES.keys()))
    def test_member_state_has_endpoint(self, code):
        assert "endpoint" in EU_MEMBER_STATES[code]
        assert EU_MEMBER_STATES[code]["endpoint"].startswith("https://")


# ====================================================================
# EU Languages
# ====================================================================


class TestEULanguages:
    """Test 24 official EU language configurations."""

    def test_24_languages(self):
        assert len(EU_LANGUAGES) == 24

    @pytest.mark.parametrize("lang", [
        "bg", "cs", "da", "de", "el", "en", "es", "et",
        "fi", "fr", "ga", "hr", "hu", "it", "lt", "lv",
        "mt", "nl", "pl", "pt", "ro", "sk", "sl", "sv",
    ])
    def test_language_included(self, lang):
        assert lang in EU_LANGUAGES

    def test_supported_languages_match(self, sample_config):
        assert sample_config.supported_languages == EU_LANGUAGES


# ====================================================================
# Environment Variable Overrides
# ====================================================================


class TestConfigEnvOverrides:
    """Test environment variable overrides."""

    def test_env_prefix(self):
        assert _ENV_PREFIX == "GL_EUDR_ACM_"

    def test_env_int_override(self):
        with patch.dict(os.environ, {"GL_EUDR_ACM_DB_PORT": "5433"}):
            cfg = AuthorityCommunicationManagerConfig()
            assert cfg.db_port == 5433

    def test_env_bool_override_true(self):
        with patch.dict(os.environ, {"GL_EUDR_ACM_PROVENANCE_ENABLED": "true"}):
            cfg = AuthorityCommunicationManagerConfig()
            assert cfg.provenance_enabled is True

    def test_env_bool_override_false(self):
        with patch.dict(os.environ, {"GL_EUDR_ACM_ENCRYPTION_ENABLED": "false"}):
            cfg = AuthorityCommunicationManagerConfig()
            assert cfg.encryption_enabled is False

    def test_env_decimal_override(self):
        with patch.dict(os.environ, {"GL_EUDR_ACM_PENALTY_MIN_AMOUNT": "5000"}):
            cfg = AuthorityCommunicationManagerConfig()
            assert cfg.penalty_min_amount == Decimal("5000")

    def test_env_string_override(self):
        with patch.dict(os.environ, {"GL_EUDR_ACM_DB_HOST": "db.production.com"}):
            cfg = AuthorityCommunicationManagerConfig()
            assert cfg.db_host == "db.production.com"

    def test_env_float_override(self):
        with patch.dict(os.environ, {"GL_EUDR_ACM_CACHE_TTL": "7200"}):
            cfg = AuthorityCommunicationManagerConfig()
            assert cfg.cache_ttl == 7200

    def test_env_helper_returns_default(self):
        assert _env("NONEXISTENT_KEY_XYZ", "default_val") == "default_val"

    def test_env_int_returns_default(self):
        assert _env_int("NONEXISTENT_KEY_XYZ", 42) == 42

    def test_env_float_returns_default(self):
        assert _env_float("NONEXISTENT_KEY_XYZ", 3.14) == 3.14

    def test_env_bool_returns_default_true(self):
        assert _env_bool("NONEXISTENT_KEY_XYZ", True) is True

    def test_env_bool_returns_default_false(self):
        assert _env_bool("NONEXISTENT_KEY_XYZ", False) is False

    def test_env_decimal_returns_default(self):
        assert _env_decimal("NONEXISTENT_KEY_XYZ", "99.99") == Decimal("99.99")

    def test_env_bool_yes_value(self):
        with patch.dict(os.environ, {"GL_EUDR_ACM_METRICS_ENABLED": "yes"}):
            cfg = AuthorityCommunicationManagerConfig()
            assert cfg.metrics_enabled is True

    def test_env_bool_one_value(self):
        with patch.dict(os.environ, {"GL_EUDR_ACM_METRICS_ENABLED": "1"}):
            cfg = AuthorityCommunicationManagerConfig()
            assert cfg.metrics_enabled is True

    def test_deadline_override(self):
        with patch.dict(os.environ, {"GL_EUDR_ACM_DEADLINE_URGENT_HOURS": "12"}):
            cfg = AuthorityCommunicationManagerConfig()
            assert cfg.deadline_urgent_hours == 12


# ====================================================================
# Singleton Pattern
# ====================================================================


class TestConfigSingleton:
    """Test thread-safe singleton pattern."""

    def test_get_config_returns_instance(self):
        cfg = get_config()
        assert isinstance(cfg, AuthorityCommunicationManagerConfig)

    def test_get_config_returns_same_instance(self):
        cfg1 = get_config()
        cfg2 = get_config()
        assert cfg1 is cfg2

    def test_reset_config_clears_singleton(self):
        cfg1 = get_config()
        reset_config()
        cfg2 = get_config()
        assert cfg1 is not cfg2


# ====================================================================
# Post-init Validation
# ====================================================================


class TestPostInitValidation:
    """Test configuration post-init validation warnings."""

    def test_valid_config_no_errors(self, sample_config):
        """Default config should be fully valid."""
        assert sample_config.db_pool_min <= sample_config.db_pool_max
        assert sample_config.penalty_min_amount < sample_config.penalty_max_amount
        assert sample_config.default_language in sample_config.supported_languages

    def test_default_language_in_supported(self, sample_config):
        assert sample_config.default_language in sample_config.supported_languages

    def test_penalty_range_valid(self, sample_config):
        assert sample_config.penalty_min_amount < sample_config.penalty_max_amount

    def test_pool_sizing_valid(self, sample_config):
        assert sample_config.db_pool_min <= sample_config.db_pool_max

    def test_reminder_positive(self, sample_config):
        assert sample_config.reminder_before_deadline_hours > 0

    def test_gdpr_retention_minimum_five_years(self, sample_config):
        """EUDR Article 31 requires minimum 5 years record keeping."""
        assert sample_config.gdpr_data_retention_days >= 365 * 5
