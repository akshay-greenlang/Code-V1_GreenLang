# -*- coding: utf-8 -*-
"""
Unit tests for EUInformationSystemInterfaceConfig - AGENT-EUDR-036

Tests default values, environment variable overrides, singleton pattern,
validation logic, and all env helper functions.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

import logging
import os
from decimal import Decimal
from unittest.mock import patch

import pytest

from greenlang.agents.eudr.eu_information_system_interface.config import (
    EUInformationSystemInterfaceConfig,
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

    def test_eu_api_base_url_default(self, sample_config):
        assert "eudr-is.europa.eu" in sample_config.eu_api_base_url

    def test_eu_api_version_default(self, sample_config):
        assert sample_config.eu_api_version == "v1"

    def test_eu_api_timeout_default(self, sample_config):
        assert sample_config.eu_api_timeout_seconds == 60

    def test_eu_api_max_retries_default(self, sample_config):
        assert sample_config.eu_api_max_retries == 3

    def test_eu_api_retry_backoff_default(self, sample_config):
        assert sample_config.eu_api_retry_backoff_factor == 1.5

    def test_dds_max_size_bytes_default(self, sample_config):
        assert sample_config.dds_max_size_bytes == 52428800  # 50 MB

    def test_dds_validation_strict_default(self, sample_config):
        assert sample_config.dds_validation_strict is True

    def test_dds_auto_submit_default(self, sample_config):
        assert sample_config.dds_auto_submit is False

    def test_dds_max_commodities_default(self, sample_config):
        assert sample_config.dds_max_commodities_per_statement == 50

    def test_eori_format_pattern_default(self, sample_config):
        assert "^" in sample_config.eori_format_pattern
        assert "A-Z" in sample_config.eori_format_pattern

    def test_registration_expiry_days_default(self, sample_config):
        assert sample_config.registration_expiry_days == 365

    def test_coordinate_precision_default(self, sample_config):
        assert sample_config.coordinate_precision == 6

    def test_coordinate_crs_default(self, sample_config):
        assert sample_config.coordinate_reference_system == "EPSG:4326"

    def test_max_polygon_vertices_default(self, sample_config):
        assert sample_config.max_polygon_vertices == 500

    def test_geolocation_area_threshold_default(self, sample_config):
        assert sample_config.geolocation_area_threshold_ha == Decimal("4.0")

    def test_max_attachments_default(self, sample_config):
        assert sample_config.max_attachments_per_package == 100

    def test_compress_packages_default(self, sample_config):
        assert sample_config.compress_packages is True

    def test_status_poll_interval_default(self, sample_config):
        assert sample_config.status_poll_interval_seconds == 300

    def test_max_poll_attempts_default(self, sample_config):
        assert sample_config.max_poll_attempts == 288

    def test_submission_timeout_hours_default(self, sample_config):
        assert sample_config.submission_timeout_hours == 72

    def test_audit_retention_years_default(self, sample_config):
        assert sample_config.audit_retention_years == 5

    def test_audit_detail_level_default(self, sample_config):
        assert sample_config.audit_detail_level == "full"

    def test_rate_limit_anonymous_default(self, sample_config):
        assert sample_config.rate_limit_anonymous == 10

    def test_rate_limit_admin_default(self, sample_config):
        assert sample_config.rate_limit_admin == 2000

    def test_circuit_breaker_failure_threshold_default(self, sample_config):
        assert sample_config.circuit_breaker_failure_threshold == 5

    def test_circuit_breaker_reset_timeout_default(self, sample_config):
        assert sample_config.circuit_breaker_reset_timeout == 60

    def test_provenance_enabled_default(self, sample_config):
        assert sample_config.provenance_enabled is True

    def test_metrics_prefix_default(self, sample_config):
        assert sample_config.metrics_prefix == "gl_eudr_euis_"

    def test_api_tls_verify_default(self, sample_config):
        assert sample_config.api_tls_verify is True


class TestConfigEnvOverride:
    """Test that environment variable overrides work correctly."""

    def test_env_override_db_host(self):
        with patch.dict(os.environ, {"GL_EUDR_EUIS_DB_HOST": "euis-db.example.com"}):
            cfg = EUInformationSystemInterfaceConfig()
            assert cfg.db_host == "euis-db.example.com"

    def test_env_override_db_port(self):
        with patch.dict(os.environ, {"GL_EUDR_EUIS_DB_PORT": "5433"}):
            cfg = EUInformationSystemInterfaceConfig()
            assert cfg.db_port == 5433

    def test_env_override_eu_api_timeout(self):
        with patch.dict(os.environ, {"GL_EUDR_EUIS_EU_API_TIMEOUT_SECONDS": "120"}):
            cfg = EUInformationSystemInterfaceConfig()
            assert cfg.eu_api_timeout_seconds == 120

    def test_env_override_eu_api_max_retries(self):
        with patch.dict(os.environ, {"GL_EUDR_EUIS_EU_API_MAX_RETRIES": "5"}):
            cfg = EUInformationSystemInterfaceConfig()
            assert cfg.eu_api_max_retries == 5

    def test_env_override_dds_validation_strict_true(self):
        with patch.dict(os.environ, {"GL_EUDR_EUIS_DDS_VALIDATION_STRICT": "true"}):
            cfg = EUInformationSystemInterfaceConfig()
            assert cfg.dds_validation_strict is True

    def test_env_override_dds_validation_strict_false(self):
        with patch.dict(os.environ, {"GL_EUDR_EUIS_DDS_VALIDATION_STRICT": "false"}):
            cfg = EUInformationSystemInterfaceConfig()
            assert cfg.dds_validation_strict is False

    def test_env_override_audit_retention_years(self):
        with patch.dict(os.environ, {"GL_EUDR_EUIS_AUDIT_RETENTION_YEARS": "7"}):
            cfg = EUInformationSystemInterfaceConfig()
            assert cfg.audit_retention_years == 7

    def test_env_override_provenance_enabled(self):
        with patch.dict(os.environ, {"GL_EUDR_EUIS_PROVENANCE_ENABLED": "false"}):
            cfg = EUInformationSystemInterfaceConfig()
            assert cfg.provenance_enabled is False

    def test_env_override_coordinate_precision(self):
        with patch.dict(os.environ, {"GL_EUDR_EUIS_COORDINATE_PRECISION": "8"}):
            cfg = EUInformationSystemInterfaceConfig()
            assert cfg.coordinate_precision == 8


class TestConfigSingleton:
    """Test singleton pattern via get_config()."""

    def test_get_config_returns_instance(self):
        cfg = get_config()
        assert isinstance(cfg, EUInformationSystemInterfaceConfig)

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
        reset_config()
        with patch.dict(os.environ, {"GL_EUDR_EUIS_DB_PORT": "9999"}):
            cfg2 = get_config()
            assert cfg2.db_port == 9999


class TestConfigValidation:
    """Test config __post_init__ validation logic."""

    def test_pool_min_exceeds_max_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            EUInformationSystemInterfaceConfig(
                db_pool_min=20,
                db_pool_max=5,
            )
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        pool_warnings = [m for m in warning_msgs if "pool" in m.lower()]
        assert len(pool_warnings) >= 1

    def test_audit_retention_below_5_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            EUInformationSystemInterfaceConfig(audit_retention_years=3)
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        retention_warnings = [m for m in warning_msgs if "retention" in m.lower() or "article 31" in m.lower()]
        assert len(retention_warnings) >= 1

    def test_valid_config_no_warnings(self, caplog):
        with caplog.at_level(logging.WARNING):
            EUInformationSystemInterfaceConfig()
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_msgs) == 0


class TestConfigEnvHelpers:
    """Test the module-level _env_* helper functions."""

    def test_env_returns_default_when_missing(self):
        result = _env("NONEXISTENT_KEY_12345", "default_val")
        assert result == "default_val"

    def test_env_returns_value_when_set(self):
        with patch.dict(os.environ, {"GL_EUDR_EUIS_TEST_KEY": "hello"}):
            result = _env("TEST_KEY", "default")
            assert result == "hello"

    def test_env_int_returns_default(self):
        result = _env_int("NONEXISTENT_INT", 42)
        assert result == 42

    def test_env_int_returns_parsed(self):
        with patch.dict(os.environ, {"GL_EUDR_EUIS_TEST_INT": "99"}):
            result = _env_int("TEST_INT", 0)
            assert result == 99

    def test_env_float_returns_default(self):
        result = _env_float("NONEXISTENT_FLOAT", 3.14)
        assert result == 3.14

    def test_env_float_returns_parsed(self):
        with patch.dict(os.environ, {"GL_EUDR_EUIS_TEST_FLOAT": "2.71"}):
            result = _env_float("TEST_FLOAT", 0.0)
            assert result == 2.71

    def test_env_bool_returns_default_false(self):
        result = _env_bool("NONEXISTENT_BOOL", False)
        assert result is False

    def test_env_bool_true_values(self):
        for val in ("true", "1", "yes"):
            with patch.dict(os.environ, {"GL_EUDR_EUIS_TEST_BOOL": val}):
                result = _env_bool("TEST_BOOL", False)
                assert result is True, f"Expected True for {val!r}"

    def test_env_bool_false_values(self):
        for val in ("false", "0", "no", "anything"):
            with patch.dict(os.environ, {"GL_EUDR_EUIS_TEST_BOOL": val}):
                result = _env_bool("TEST_BOOL", True)
                assert result is False, f"Expected False for {val!r}"

    def test_env_decimal_returns_default(self):
        result = _env_decimal("NONEXISTENT_DEC", "99.99")
        assert result == Decimal("99.99")

    def test_env_decimal_returns_parsed(self):
        with patch.dict(os.environ, {"GL_EUDR_EUIS_TEST_DEC": "42.5"}):
            result = _env_decimal("TEST_DEC", "0")
            assert result == Decimal("42.5")

    def test_env_prefix_is_correct(self):
        assert _ENV_PREFIX == "GL_EUDR_EUIS_"
