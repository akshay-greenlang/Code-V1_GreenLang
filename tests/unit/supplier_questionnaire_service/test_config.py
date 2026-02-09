# -*- coding: utf-8 -*-
"""
Unit tests for SupplierQuestionnaireConfig (AGENT-DATA-008)

Tests configuration defaults, environment variable overrides, singleton
lifecycle (get_config / set_config / reset_config), type coercion for
numeric fields, and thread safety of the config singleton.

Target: 80 tests covering all 23 config fields + lifecycle + threading.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import os
import threading
from unittest.mock import patch

import pytest

from greenlang.supplier_questionnaire.config import (
    SupplierQuestionnaireConfig,
    get_config,
    reset_config,
    set_config,
)


# ============================================================================
# Default value tests for all 23 fields
# ============================================================================


class TestSupplierQuestionnaireConfigDefaults:
    """Verify every default value on a freshly created config."""

    def test_default_database_url_is_empty(self):
        cfg = SupplierQuestionnaireConfig()
        assert cfg.database_url == ""

    def test_default_redis_url_is_empty(self):
        cfg = SupplierQuestionnaireConfig()
        assert cfg.redis_url == ""

    def test_default_log_level_is_info(self):
        cfg = SupplierQuestionnaireConfig()
        assert cfg.log_level == "INFO"

    def test_default_framework_is_custom(self):
        cfg = SupplierQuestionnaireConfig()
        assert cfg.default_framework == "custom"

    def test_default_deadline_days_is_60(self):
        cfg = SupplierQuestionnaireConfig()
        assert cfg.default_deadline_days == 60

    def test_default_max_reminders_is_4(self):
        cfg = SupplierQuestionnaireConfig()
        assert cfg.max_reminders == 4

    def test_default_reminder_gentle_days_is_7(self):
        cfg = SupplierQuestionnaireConfig()
        assert cfg.reminder_gentle_days == 7

    def test_default_reminder_firm_days_is_3(self):
        cfg = SupplierQuestionnaireConfig()
        assert cfg.reminder_firm_days == 3

    def test_default_reminder_urgent_days_is_1(self):
        cfg = SupplierQuestionnaireConfig()
        assert cfg.reminder_urgent_days == 1

    def test_default_min_completion_pct_is_80(self):
        cfg = SupplierQuestionnaireConfig()
        assert cfg.min_completion_pct == 80.0

    def test_default_score_leader_threshold_is_80(self):
        cfg = SupplierQuestionnaireConfig()
        assert cfg.score_leader_threshold == 80

    def test_default_score_advanced_threshold_is_60(self):
        cfg = SupplierQuestionnaireConfig()
        assert cfg.score_advanced_threshold == 60

    def test_default_score_developing_threshold_is_40(self):
        cfg = SupplierQuestionnaireConfig()
        assert cfg.score_developing_threshold == 40

    def test_default_score_lagging_threshold_is_20(self):
        cfg = SupplierQuestionnaireConfig()
        assert cfg.score_lagging_threshold == 20

    def test_default_batch_size_is_100(self):
        cfg = SupplierQuestionnaireConfig()
        assert cfg.batch_size == 100

    def test_default_worker_count_is_4(self):
        cfg = SupplierQuestionnaireConfig()
        assert cfg.worker_count == 4

    def test_default_cache_ttl_seconds_is_1800(self):
        cfg = SupplierQuestionnaireConfig()
        assert cfg.cache_ttl_seconds == 1800

    def test_default_pool_min_size_is_2(self):
        cfg = SupplierQuestionnaireConfig()
        assert cfg.pool_min_size == 2

    def test_default_pool_max_size_is_10(self):
        cfg = SupplierQuestionnaireConfig()
        assert cfg.pool_max_size == 10

    def test_default_retention_days_is_1095(self):
        cfg = SupplierQuestionnaireConfig()
        assert cfg.retention_days == 1095

    def test_default_portal_base_url_is_empty(self):
        cfg = SupplierQuestionnaireConfig()
        assert cfg.portal_base_url == ""

    def test_default_smtp_host_is_empty(self):
        cfg = SupplierQuestionnaireConfig()
        assert cfg.smtp_host == ""

    def test_default_language_is_en(self):
        cfg = SupplierQuestionnaireConfig()
        assert cfg.default_language == "en"


# ============================================================================
# Environment variable override tests for all 23 fields
# ============================================================================


class TestSupplierQuestionnaireConfigEnvOverrides:
    """Verify each field can be overridden via GL_SUPPLIER_QUEST_ env vars."""

    def test_env_database_url(self, monkeypatch):
        monkeypatch.setenv("GL_SUPPLIER_QUEST_DATABASE_URL", "postgresql://db:5432/test")
        cfg = SupplierQuestionnaireConfig.from_env()
        assert cfg.database_url == "postgresql://db:5432/test"

    def test_env_redis_url(self, monkeypatch):
        monkeypatch.setenv("GL_SUPPLIER_QUEST_REDIS_URL", "redis://cache:6379/0")
        cfg = SupplierQuestionnaireConfig.from_env()
        assert cfg.redis_url == "redis://cache:6379/0"

    def test_env_log_level(self, monkeypatch):
        monkeypatch.setenv("GL_SUPPLIER_QUEST_LOG_LEVEL", "DEBUG")
        cfg = SupplierQuestionnaireConfig.from_env()
        assert cfg.log_level == "DEBUG"

    def test_env_default_framework(self, monkeypatch):
        monkeypatch.setenv("GL_SUPPLIER_QUEST_DEFAULT_FRAMEWORK", "cdp_climate")
        cfg = SupplierQuestionnaireConfig.from_env()
        assert cfg.default_framework == "cdp_climate"

    def test_env_default_deadline_days(self, monkeypatch):
        monkeypatch.setenv("GL_SUPPLIER_QUEST_DEFAULT_DEADLINE_DAYS", "90")
        cfg = SupplierQuestionnaireConfig.from_env()
        assert cfg.default_deadline_days == 90

    def test_env_max_reminders(self, monkeypatch):
        monkeypatch.setenv("GL_SUPPLIER_QUEST_MAX_REMINDERS", "6")
        cfg = SupplierQuestionnaireConfig.from_env()
        assert cfg.max_reminders == 6

    def test_env_reminder_gentle_days(self, monkeypatch):
        monkeypatch.setenv("GL_SUPPLIER_QUEST_REMINDER_GENTLE_DAYS", "14")
        cfg = SupplierQuestionnaireConfig.from_env()
        assert cfg.reminder_gentle_days == 14

    def test_env_reminder_firm_days(self, monkeypatch):
        monkeypatch.setenv("GL_SUPPLIER_QUEST_REMINDER_FIRM_DAYS", "5")
        cfg = SupplierQuestionnaireConfig.from_env()
        assert cfg.reminder_firm_days == 5

    def test_env_reminder_urgent_days(self, monkeypatch):
        monkeypatch.setenv("GL_SUPPLIER_QUEST_REMINDER_URGENT_DAYS", "2")
        cfg = SupplierQuestionnaireConfig.from_env()
        assert cfg.reminder_urgent_days == 2

    def test_env_min_completion_pct(self, monkeypatch):
        monkeypatch.setenv("GL_SUPPLIER_QUEST_MIN_COMPLETION_PCT", "90.5")
        cfg = SupplierQuestionnaireConfig.from_env()
        assert cfg.min_completion_pct == 90.5

    def test_env_score_leader_threshold(self, monkeypatch):
        monkeypatch.setenv("GL_SUPPLIER_QUEST_SCORE_LEADER_THRESHOLD", "85")
        cfg = SupplierQuestionnaireConfig.from_env()
        assert cfg.score_leader_threshold == 85

    def test_env_score_advanced_threshold(self, monkeypatch):
        monkeypatch.setenv("GL_SUPPLIER_QUEST_SCORE_ADVANCED_THRESHOLD", "65")
        cfg = SupplierQuestionnaireConfig.from_env()
        assert cfg.score_advanced_threshold == 65

    def test_env_score_developing_threshold(self, monkeypatch):
        monkeypatch.setenv("GL_SUPPLIER_QUEST_SCORE_DEVELOPING_THRESHOLD", "45")
        cfg = SupplierQuestionnaireConfig.from_env()
        assert cfg.score_developing_threshold == 45

    def test_env_score_lagging_threshold(self, monkeypatch):
        monkeypatch.setenv("GL_SUPPLIER_QUEST_SCORE_LAGGING_THRESHOLD", "25")
        cfg = SupplierQuestionnaireConfig.from_env()
        assert cfg.score_lagging_threshold == 25

    def test_env_batch_size(self, monkeypatch):
        monkeypatch.setenv("GL_SUPPLIER_QUEST_BATCH_SIZE", "500")
        cfg = SupplierQuestionnaireConfig.from_env()
        assert cfg.batch_size == 500

    def test_env_worker_count(self, monkeypatch):
        monkeypatch.setenv("GL_SUPPLIER_QUEST_WORKER_COUNT", "8")
        cfg = SupplierQuestionnaireConfig.from_env()
        assert cfg.worker_count == 8

    def test_env_cache_ttl_seconds(self, monkeypatch):
        monkeypatch.setenv("GL_SUPPLIER_QUEST_CACHE_TTL_SECONDS", "3600")
        cfg = SupplierQuestionnaireConfig.from_env()
        assert cfg.cache_ttl_seconds == 3600

    def test_env_pool_min_size(self, monkeypatch):
        monkeypatch.setenv("GL_SUPPLIER_QUEST_POOL_MIN_SIZE", "5")
        cfg = SupplierQuestionnaireConfig.from_env()
        assert cfg.pool_min_size == 5

    def test_env_pool_max_size(self, monkeypatch):
        monkeypatch.setenv("GL_SUPPLIER_QUEST_POOL_MAX_SIZE", "20")
        cfg = SupplierQuestionnaireConfig.from_env()
        assert cfg.pool_max_size == 20

    def test_env_retention_days(self, monkeypatch):
        monkeypatch.setenv("GL_SUPPLIER_QUEST_RETENTION_DAYS", "730")
        cfg = SupplierQuestionnaireConfig.from_env()
        assert cfg.retention_days == 730

    def test_env_portal_base_url(self, monkeypatch):
        monkeypatch.setenv("GL_SUPPLIER_QUEST_PORTAL_BASE_URL", "https://portal.greenlang.io")
        cfg = SupplierQuestionnaireConfig.from_env()
        assert cfg.portal_base_url == "https://portal.greenlang.io"

    def test_env_smtp_host(self, monkeypatch):
        monkeypatch.setenv("GL_SUPPLIER_QUEST_SMTP_HOST", "smtp.greenlang.io")
        cfg = SupplierQuestionnaireConfig.from_env()
        assert cfg.smtp_host == "smtp.greenlang.io"

    def test_env_default_language(self, monkeypatch):
        monkeypatch.setenv("GL_SUPPLIER_QUEST_DEFAULT_LANGUAGE", "de")
        cfg = SupplierQuestionnaireConfig.from_env()
        assert cfg.default_language == "de"


# ============================================================================
# Type coercion and invalid value tests
# ============================================================================


class TestSupplierQuestionnaireConfigTypeCoercion:
    """Verify int/float coercion and fallback on invalid values."""

    def test_invalid_int_falls_back_to_default_deadline_days(self, monkeypatch):
        monkeypatch.setenv("GL_SUPPLIER_QUEST_DEFAULT_DEADLINE_DAYS", "not_a_number")
        cfg = SupplierQuestionnaireConfig.from_env()
        assert cfg.default_deadline_days == 60

    def test_invalid_int_falls_back_to_default_max_reminders(self, monkeypatch):
        monkeypatch.setenv("GL_SUPPLIER_QUEST_MAX_REMINDERS", "abc")
        cfg = SupplierQuestionnaireConfig.from_env()
        assert cfg.max_reminders == 4

    def test_invalid_int_falls_back_to_default_batch_size(self, monkeypatch):
        monkeypatch.setenv("GL_SUPPLIER_QUEST_BATCH_SIZE", "3.14")
        cfg = SupplierQuestionnaireConfig.from_env()
        assert cfg.batch_size == 100

    def test_invalid_float_falls_back_to_default_min_completion_pct(self, monkeypatch):
        monkeypatch.setenv("GL_SUPPLIER_QUEST_MIN_COMPLETION_PCT", "not_float")
        cfg = SupplierQuestionnaireConfig.from_env()
        assert cfg.min_completion_pct == 80.0

    def test_int_from_string_zero(self, monkeypatch):
        monkeypatch.setenv("GL_SUPPLIER_QUEST_WORKER_COUNT", "0")
        cfg = SupplierQuestionnaireConfig.from_env()
        assert cfg.worker_count == 0

    def test_int_from_negative_string(self, monkeypatch):
        monkeypatch.setenv("GL_SUPPLIER_QUEST_REMINDER_GENTLE_DAYS", "-1")
        cfg = SupplierQuestionnaireConfig.from_env()
        assert cfg.reminder_gentle_days == -1

    def test_float_from_integer_string(self, monkeypatch):
        monkeypatch.setenv("GL_SUPPLIER_QUEST_MIN_COMPLETION_PCT", "100")
        cfg = SupplierQuestionnaireConfig.from_env()
        assert cfg.min_completion_pct == 100.0

    def test_invalid_int_falls_back_to_default_cache_ttl(self, monkeypatch):
        monkeypatch.setenv("GL_SUPPLIER_QUEST_CACHE_TTL_SECONDS", "")
        cfg = SupplierQuestionnaireConfig.from_env()
        # Empty string cannot be parsed as int; falls back to default
        assert cfg.cache_ttl_seconds == 1800

    def test_invalid_int_falls_back_to_default_pool_min(self, monkeypatch):
        monkeypatch.setenv("GL_SUPPLIER_QUEST_POOL_MIN_SIZE", "xyz")
        cfg = SupplierQuestionnaireConfig.from_env()
        assert cfg.pool_min_size == 2

    def test_invalid_int_falls_back_to_default_pool_max(self, monkeypatch):
        monkeypatch.setenv("GL_SUPPLIER_QUEST_POOL_MAX_SIZE", "true")
        cfg = SupplierQuestionnaireConfig.from_env()
        assert cfg.pool_max_size == 10

    def test_invalid_int_falls_back_to_default_retention(self, monkeypatch):
        monkeypatch.setenv("GL_SUPPLIER_QUEST_RETENTION_DAYS", "infinite")
        cfg = SupplierQuestionnaireConfig.from_env()
        assert cfg.retention_days == 1095

    def test_invalid_int_falls_back_to_default_score_leader(self, monkeypatch):
        monkeypatch.setenv("GL_SUPPLIER_QUEST_SCORE_LEADER_THRESHOLD", "high")
        cfg = SupplierQuestionnaireConfig.from_env()
        assert cfg.score_leader_threshold == 80

    def test_invalid_int_falls_back_to_default_reminder_firm(self, monkeypatch):
        monkeypatch.setenv("GL_SUPPLIER_QUEST_REMINDER_FIRM_DAYS", "two")
        cfg = SupplierQuestionnaireConfig.from_env()
        assert cfg.reminder_firm_days == 3

    def test_invalid_int_falls_back_to_default_reminder_urgent(self, monkeypatch):
        monkeypatch.setenv("GL_SUPPLIER_QUEST_REMINDER_URGENT_DAYS", "!!")
        cfg = SupplierQuestionnaireConfig.from_env()
        assert cfg.reminder_urgent_days == 1


# ============================================================================
# Singleton lifecycle tests
# ============================================================================


class TestSupplierQuestionnaireConfigSingleton:
    """Test get_config, set_config, reset_config singleton behaviour."""

    def test_get_config_returns_config_instance(self):
        cfg = get_config()
        assert isinstance(cfg, SupplierQuestionnaireConfig)

    def test_get_config_returns_same_instance_on_repeated_calls(self):
        cfg1 = get_config()
        cfg2 = get_config()
        assert cfg1 is cfg2

    def test_set_config_replaces_singleton(self):
        custom = SupplierQuestionnaireConfig(batch_size=999)
        set_config(custom)
        cfg = get_config()
        assert cfg.batch_size == 999
        assert cfg is custom

    def test_reset_config_clears_singleton(self):
        _ = get_config()
        reset_config()
        # After reset, a new call creates a fresh instance
        cfg = get_config()
        assert cfg.batch_size == 100  # default

    def test_set_config_then_reset_restores_defaults(self):
        custom = SupplierQuestionnaireConfig(worker_count=32)
        set_config(custom)
        assert get_config().worker_count == 32
        reset_config()
        assert get_config().worker_count == 4

    def test_get_config_picks_up_env_on_first_call(self, monkeypatch):
        monkeypatch.setenv("GL_SUPPLIER_QUEST_BATCH_SIZE", "42")
        cfg = get_config()
        assert cfg.batch_size == 42

    def test_get_config_ignores_env_on_second_call(self, monkeypatch):
        cfg1 = get_config()  # created with defaults
        monkeypatch.setenv("GL_SUPPLIER_QUEST_BATCH_SIZE", "999")
        cfg2 = get_config()  # should still be the same cached instance
        assert cfg2.batch_size == cfg1.batch_size

    def test_reset_then_get_with_env_picks_up_new_env(self, monkeypatch):
        _ = get_config()
        reset_config()
        monkeypatch.setenv("GL_SUPPLIER_QUEST_WORKER_COUNT", "16")
        cfg = get_config()
        assert cfg.worker_count == 16


# ============================================================================
# Thread safety tests
# ============================================================================


class TestSupplierQuestionnaireConfigThreadSafety:
    """Verify the singleton is thread-safe under concurrent access."""

    def test_concurrent_get_config_returns_same_instance(self):
        results = []

        def _get():
            results.append(id(get_config()))

        threads = [threading.Thread(target=_get) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads should see the same object id
        assert len(set(results)) == 1

    def test_concurrent_set_and_get_does_not_crash(self):
        errors = []

        def _set_and_get(idx: int):
            try:
                if idx % 2 == 0:
                    set_config(SupplierQuestionnaireConfig(batch_size=idx))
                else:
                    _ = get_config()
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_set_and_get, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_reset_and_get_does_not_crash(self):
        errors = []

        def _reset_and_get(idx: int):
            try:
                if idx % 3 == 0:
                    reset_config()
                else:
                    _ = get_config()
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_reset_and_get, args=(i,)) for i in range(30)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# ============================================================================
# Multiple env override combination tests
# ============================================================================


class TestSupplierQuestionnaireConfigMultipleOverrides:
    """Test that multiple env vars can be combined."""

    def test_multiple_string_overrides(self, monkeypatch):
        monkeypatch.setenv("GL_SUPPLIER_QUEST_LOG_LEVEL", "WARNING")
        monkeypatch.setenv("GL_SUPPLIER_QUEST_DEFAULT_LANGUAGE", "fr")
        monkeypatch.setenv("GL_SUPPLIER_QUEST_SMTP_HOST", "mail.example.com")
        cfg = SupplierQuestionnaireConfig.from_env()
        assert cfg.log_level == "WARNING"
        assert cfg.default_language == "fr"
        assert cfg.smtp_host == "mail.example.com"

    def test_multiple_int_overrides(self, monkeypatch):
        monkeypatch.setenv("GL_SUPPLIER_QUEST_BATCH_SIZE", "200")
        monkeypatch.setenv("GL_SUPPLIER_QUEST_WORKER_COUNT", "12")
        monkeypatch.setenv("GL_SUPPLIER_QUEST_RETENTION_DAYS", "365")
        cfg = SupplierQuestionnaireConfig.from_env()
        assert cfg.batch_size == 200
        assert cfg.worker_count == 12
        assert cfg.retention_days == 365

    def test_mixed_string_int_float_overrides(self, monkeypatch):
        monkeypatch.setenv("GL_SUPPLIER_QUEST_DEFAULT_FRAMEWORK", "ecovadis")
        monkeypatch.setenv("GL_SUPPLIER_QUEST_DEFAULT_DEADLINE_DAYS", "30")
        monkeypatch.setenv("GL_SUPPLIER_QUEST_MIN_COMPLETION_PCT", "95.5")
        cfg = SupplierQuestionnaireConfig.from_env()
        assert cfg.default_framework == "ecovadis"
        assert cfg.default_deadline_days == 30
        assert cfg.min_completion_pct == 95.5


# ============================================================================
# Dataclass identity tests
# ============================================================================


class TestSupplierQuestionnaireConfigDataclass:
    """Test dataclass behaviour (equality, repr, etc.)."""

    def test_two_default_instances_are_equal(self):
        a = SupplierQuestionnaireConfig()
        b = SupplierQuestionnaireConfig()
        assert a == b

    def test_instances_with_different_values_are_not_equal(self):
        a = SupplierQuestionnaireConfig(batch_size=1)
        b = SupplierQuestionnaireConfig(batch_size=2)
        assert a != b

    def test_repr_contains_class_name(self):
        cfg = SupplierQuestionnaireConfig()
        assert "SupplierQuestionnaireConfig" in repr(cfg)

    def test_from_env_returns_correct_type(self):
        cfg = SupplierQuestionnaireConfig.from_env()
        assert isinstance(cfg, SupplierQuestionnaireConfig)

    def test_field_count_is_23(self):
        """Confirm the config has exactly 23 fields as documented."""
        import dataclasses
        fields = dataclasses.fields(SupplierQuestionnaireConfig)
        assert len(fields) == 23
