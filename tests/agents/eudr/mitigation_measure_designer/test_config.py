# -*- coding: utf-8 -*-
"""
Unit tests for MitigationMeasureDesignerConfig - AGENT-EUDR-029

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

from greenlang.agents.eudr.mitigation_measure_designer.config import (
    MitigationMeasureDesignerConfig,
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

    def test_negligible_max_default(self, sample_config):
        assert sample_config.negligible_max == Decimal("15")

    def test_low_max_default(self, sample_config):
        assert sample_config.low_max == Decimal("30")

    def test_standard_max_default(self, sample_config):
        assert sample_config.standard_max == Decimal("60")

    def test_high_max_default(self, sample_config):
        assert sample_config.high_max == Decimal("80")

    def test_mitigation_target_score_default(self, sample_config):
        assert sample_config.mitigation_target_score == Decimal("30")

    def test_conservative_factor_default(self, sample_config):
        assert sample_config.conservative_factor == Decimal("0.70")

    def test_moderate_factor_default(self, sample_config):
        assert sample_config.moderate_factor == Decimal("1.00")

    def test_optimistic_factor_default(self, sample_config):
        assert sample_config.optimistic_factor == Decimal("1.30")

    def test_min_effectiveness_threshold_default(self, sample_config):
        assert sample_config.min_effectiveness_threshold == Decimal("5")

    def test_max_effectiveness_cap_default(self, sample_config):
        assert sample_config.max_effectiveness_cap == Decimal("80")

    def test_default_deadline_days(self, sample_config):
        assert sample_config.default_deadline_days == 30

    def test_max_measures_per_strategy(self, sample_config):
        assert sample_config.max_measures_per_strategy == 20

    def test_approval_required_default(self, sample_config):
        assert sample_config.approval_required is True

    def test_provenance_enabled_default(self, sample_config):
        assert sample_config.provenance_enabled is True

    def test_metrics_prefix_default(self, sample_config):
        assert sample_config.metrics_prefix == "gl_eudr_mmd_"

    def test_report_format_default(self, sample_config):
        assert sample_config.report_format == "json"

    def test_rate_limit_anonymous_default(self, sample_config):
        assert sample_config.rate_limit_anonymous == 10

    def test_rate_limit_admin_default(self, sample_config):
        assert sample_config.rate_limit_admin == 2000

    def test_circuit_breaker_defaults(self, sample_config):
        assert sample_config.circuit_breaker_failure_threshold == 5
        assert sample_config.circuit_breaker_reset_timeout == 60
        assert sample_config.circuit_breaker_half_open_max == 3


class TestConfigEnvOverride:
    """Test that environment variable overrides work correctly."""

    def test_env_override_db_host(self):
        with patch.dict(os.environ, {"GL_EUDR_MMD_DB_HOST": "my-db.example.com"}):
            cfg = MitigationMeasureDesignerConfig()
            assert cfg.db_host == "my-db.example.com"

    def test_env_override_db_port(self):
        with patch.dict(os.environ, {"GL_EUDR_MMD_DB_PORT": "5433"}):
            cfg = MitigationMeasureDesignerConfig()
            assert cfg.db_port == 5433

    def test_env_override_negligible_max(self):
        with patch.dict(os.environ, {"GL_EUDR_MMD_NEGLIGIBLE_MAX": "10"}):
            cfg = MitigationMeasureDesignerConfig()
            assert cfg.negligible_max == Decimal("10")

    def test_env_override_bool_true(self):
        with patch.dict(os.environ, {"GL_EUDR_MMD_APPROVAL_REQUIRED": "true"}):
            cfg = MitigationMeasureDesignerConfig()
            assert cfg.approval_required is True

    def test_env_override_bool_false(self):
        with patch.dict(os.environ, {"GL_EUDR_MMD_APPROVAL_REQUIRED": "false"}):
            cfg = MitigationMeasureDesignerConfig()
            assert cfg.approval_required is False


class TestConfigSingleton:
    """Test singleton pattern via get_config()."""

    def test_get_config_returns_instance(self):
        cfg = get_config()
        assert isinstance(cfg, MitigationMeasureDesignerConfig)

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
        with patch.dict(os.environ, {"GL_EUDR_MMD_DB_PORT": "9999"}):
            cfg2 = get_config()
            assert cfg2.db_port == 9999


class TestConfigValidation:
    """Test config __post_init__ validation logic."""

    def test_valid_threshold_ordering_no_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            cfg = MitigationMeasureDesignerConfig()
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        threshold_warnings = [m for m in warning_msgs if "ascending order" in m and "Risk thresholds" in m]
        assert len(threshold_warnings) == 0

    def test_invalid_threshold_ordering_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            cfg = MitigationMeasureDesignerConfig(
                negligible_max=Decimal("50"),
                low_max=Decimal("30"),
                standard_max=Decimal("60"),
                high_max=Decimal("80"),
            )
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        threshold_warnings = [m for m in warning_msgs if "ascending order" in m]
        assert len(threshold_warnings) >= 1

    def test_invalid_effectiveness_ordering_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            cfg = MitigationMeasureDesignerConfig(
                conservative_factor=Decimal("1.50"),
                moderate_factor=Decimal("1.00"),
                optimistic_factor=Decimal("1.30"),
            )
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        factor_warnings = [m for m in warning_msgs if "Effectiveness factors" in m]
        assert len(factor_warnings) >= 1

    def test_mitigation_target_exceeds_low_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            cfg = MitigationMeasureDesignerConfig(
                mitigation_target_score=Decimal("50"),
            )
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        target_warnings = [m for m in warning_msgs if "Mitigation target score" in m]
        assert len(target_warnings) >= 1

    def test_min_effectiveness_ge_max_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            cfg = MitigationMeasureDesignerConfig(
                min_effectiveness_threshold=Decimal("90"),
                max_effectiveness_cap=Decimal("80"),
            )
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        eff_warnings = [m for m in warning_msgs if "Min effectiveness" in m]
        assert len(eff_warnings) >= 1

    def test_pool_min_exceeds_max_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            cfg = MitigationMeasureDesignerConfig(
                db_pool_min=20,
                db_pool_max=5,
            )
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        pool_warnings = [m for m in warning_msgs if "pool min" in m]
        assert len(pool_warnings) >= 1


class TestConfigEnvHelpers:
    """Test the module-level _env_* helper functions."""

    def test_env_returns_default_when_missing(self):
        result = _env("NONEXISTENT_KEY_12345", "default_val")
        assert result == "default_val"

    def test_env_returns_value_when_set(self):
        with patch.dict(os.environ, {"GL_EUDR_MMD_TEST_KEY": "hello"}):
            result = _env("TEST_KEY", "default")
            assert result == "hello"

    def test_env_int_returns_default(self):
        result = _env_int("NONEXISTENT_INT", 42)
        assert result == 42

    def test_env_int_returns_parsed(self):
        with patch.dict(os.environ, {"GL_EUDR_MMD_TEST_INT": "99"}):
            result = _env_int("TEST_INT", 0)
            assert result == 99

    def test_env_float_returns_default(self):
        result = _env_float("NONEXISTENT_FLOAT", 3.14)
        assert result == 3.14

    def test_env_float_returns_parsed(self):
        with patch.dict(os.environ, {"GL_EUDR_MMD_TEST_FLOAT": "2.71"}):
            result = _env_float("TEST_FLOAT", 0.0)
            assert result == 2.71

    def test_env_bool_returns_default_false(self):
        result = _env_bool("NONEXISTENT_BOOL", False)
        assert result is False

    def test_env_bool_true_values(self):
        for val in ("true", "1", "yes"):
            with patch.dict(os.environ, {"GL_EUDR_MMD_TEST_BOOL": val}):
                result = _env_bool("TEST_BOOL", False)
                assert result is True, f"Expected True for {val!r}"

    def test_env_bool_false_values(self):
        for val in ("false", "0", "no", "anything"):
            with patch.dict(os.environ, {"GL_EUDR_MMD_TEST_BOOL": val}):
                result = _env_bool("TEST_BOOL", True)
                assert result is False, f"Expected False for {val!r}"

    def test_env_decimal_returns_default(self):
        result = _env_decimal("NONEXISTENT_DEC", "99.99")
        assert result == Decimal("99.99")

    def test_env_decimal_returns_parsed(self):
        with patch.dict(os.environ, {"GL_EUDR_MMD_TEST_DEC": "42.5"}):
            result = _env_decimal("TEST_DEC", "0")
            assert result == Decimal("42.5")

    def test_env_prefix_is_correct(self):
        assert _ENV_PREFIX == "GL_EUDR_MMD_"
