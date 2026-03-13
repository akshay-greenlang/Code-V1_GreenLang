# -*- coding: utf-8 -*-
"""
Unit tests for DDSCreatorConfig - AGENT-EUDR-037

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

from greenlang.agents.eudr.due_diligence_statement_creator.config import (
    DDSCreatorConfig,
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

    def test_dds_template_version_default(self, sample_config):
        assert sample_config.dds_template_version == "1.0"

    def test_dds_max_commodities_default(self, sample_config):
        assert sample_config.dds_max_commodities_per_statement == 50

    def test_dds_max_plots_default(self, sample_config):
        assert sample_config.dds_max_plots_per_commodity == 10000

    def test_dds_reference_number_prefix_default(self, sample_config):
        assert sample_config.dds_reference_number_prefix == "GL-DDS"

    def test_supported_languages_count(self, sample_config):
        assert len(sample_config.supported_languages) == 24

    def test_default_language_default(self, sample_config):
        assert sample_config.default_language == "en"

    def test_translation_cache_ttl_default(self, sample_config):
        assert sample_config.translation_cache_ttl == 86400

    def test_max_attachment_size_mb_default(self, sample_config):
        assert sample_config.max_attachment_size_mb == 25

    def test_max_package_size_mb_default(self, sample_config):
        assert sample_config.max_package_size_mb == 500

    def test_max_attachments_per_statement_default(self, sample_config):
        assert sample_config.max_attachments_per_statement == 100

    def test_signature_algorithm_default(self, sample_config):
        assert sample_config.signature_algorithm == "RSA-SHA256"

    def test_signature_validity_days_default(self, sample_config):
        assert sample_config.signature_validity_days == 365

    def test_require_qualified_signature_default(self, sample_config):
        assert sample_config.require_qualified_signature is True

    def test_geolocation_precision_digits_default(self, sample_config):
        assert sample_config.geolocation_precision_digits == 6

    def test_geolocation_max_polygon_vertices_default(self, sample_config):
        assert sample_config.geolocation_max_polygon_vertices == 5000

    def test_article4_mandatory_field_count_default(self, sample_config):
        assert sample_config.article4_mandatory_field_count == 14

    def test_quantity_tolerance_percent_default(self, sample_config):
        assert sample_config.quantity_tolerance_percent == Decimal("0.5")

    def test_statement_generation_timeout_default(self, sample_config):
        assert sample_config.statement_generation_timeout_seconds == 120

    def test_batch_size_default(self, sample_config):
        assert sample_config.batch_size == 50

    def test_retention_years_default(self, sample_config):
        assert sample_config.retention_years == 5

    def test_deforestation_cutoff_date_default(self, sample_config):
        assert sample_config.deforestation_cutoff_date == "2020-12-31"

    def test_provenance_enabled_default(self, sample_config):
        assert sample_config.provenance_enabled is True

    def test_provenance_algorithm_default(self, sample_config):
        assert sample_config.provenance_algorithm == "sha256"

    def test_metrics_enabled_default(self, sample_config):
        assert sample_config.metrics_enabled is True

    def test_metrics_prefix_default(self, sample_config):
        assert sample_config.metrics_prefix == "gl_eudr_ddsc_"

    def test_rate_limit_anonymous_default(self, sample_config):
        assert sample_config.rate_limit_anonymous == 10

    def test_rate_limit_basic_default(self, sample_config):
        assert sample_config.rate_limit_basic == 30

    def test_rate_limit_standard_default(self, sample_config):
        assert sample_config.rate_limit_standard == 100

    def test_rate_limit_premium_default(self, sample_config):
        assert sample_config.rate_limit_premium == 500

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

    def test_log_level_default(self, sample_config):
        assert sample_config.log_level == "INFO"

    def test_parallel_engine_calls_default(self, sample_config):
        assert sample_config.parallel_engine_calls == 5

    def test_max_risk_level_for_auto_submit_default(self, sample_config):
        assert sample_config.max_risk_level_for_auto_submit == "low"

    def test_include_regulatory_refs_default(self, sample_config):
        assert sample_config.include_regulatory_refs is True

    def test_dds_mandatory_sections_count(self, sample_config):
        assert len(sample_config.dds_mandatory_sections) == 9


class TestConfigEnvOverride:
    """Test that environment variable overrides work correctly."""

    def test_env_override_db_host(self):
        with patch.dict(os.environ, {"GL_EUDR_DDSC_DB_HOST": "my-db.example.com"}):
            cfg = DDSCreatorConfig()
            assert cfg.db_host == "my-db.example.com"

    def test_env_override_db_port(self):
        with patch.dict(os.environ, {"GL_EUDR_DDSC_DB_PORT": "5433"}):
            cfg = DDSCreatorConfig()
            assert cfg.db_port == 5433

    def test_env_override_redis_port(self):
        with patch.dict(os.environ, {"GL_EUDR_DDSC_REDIS_PORT": "6380"}):
            cfg = DDSCreatorConfig()
            assert cfg.redis_port == 6380

    def test_env_override_cache_ttl(self):
        with patch.dict(os.environ, {"GL_EUDR_DDSC_CACHE_TTL": "7200"}):
            cfg = DDSCreatorConfig()
            assert cfg.cache_ttl == 7200

    def test_env_override_geolocation_precision(self):
        with patch.dict(os.environ, {"GL_EUDR_DDSC_GEOLOCATION_PRECISION": "8"}):
            cfg = DDSCreatorConfig()
            assert cfg.geolocation_precision_digits == 8

    def test_env_override_retention_years(self):
        with patch.dict(os.environ, {"GL_EUDR_DDSC_RETENTION_YEARS": "7"}):
            cfg = DDSCreatorConfig()
            assert cfg.retention_years == 7

    def test_env_override_batch_size(self):
        with patch.dict(os.environ, {"GL_EUDR_DDSC_BATCH_SIZE": "100"}):
            cfg = DDSCreatorConfig()
            assert cfg.batch_size == 100

    def test_env_override_bool_true(self):
        with patch.dict(os.environ, {"GL_EUDR_DDSC_PROVENANCE_ENABLED": "true"}):
            cfg = DDSCreatorConfig()
            assert cfg.provenance_enabled is True

    def test_env_override_bool_false(self):
        with patch.dict(os.environ, {"GL_EUDR_DDSC_PROVENANCE_ENABLED": "false"}):
            cfg = DDSCreatorConfig()
            assert cfg.provenance_enabled is False

    def test_env_override_quantity_tolerance(self):
        with patch.dict(os.environ, {"GL_EUDR_DDSC_QUANTITY_TOLERANCE_PERCENT": "1.5"}):
            cfg = DDSCreatorConfig()
            assert cfg.quantity_tolerance_percent == Decimal("1.5")

    def test_env_override_generation_timeout(self):
        with patch.dict(os.environ, {"GL_EUDR_DDSC_GENERATION_TIMEOUT": "60"}):
            cfg = DDSCreatorConfig()
            assert cfg.statement_generation_timeout_seconds == 60

    def test_env_override_max_concurrent(self):
        with patch.dict(os.environ, {"GL_EUDR_DDSC_MAX_CONCURRENT": "20"}):
            cfg = DDSCreatorConfig()
            assert cfg.max_concurrent == 20


class TestConfigSingleton:
    """Test singleton pattern via get_config()."""

    def test_get_config_returns_instance(self):
        cfg = get_config()
        assert isinstance(cfg, DDSCreatorConfig)

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
        with patch.dict(os.environ, {"GL_EUDR_DDSC_DB_PORT": "9999"}):
            cfg2 = get_config()
            assert cfg2.db_port == 9999


class TestConfigValidation:
    """Test config __post_init__ validation logic."""

    def test_pool_min_exceeds_max_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            cfg = DDSCreatorConfig(db_pool_min=20, db_pool_max=5)
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        pool_warnings = [m for m in warning_msgs if "pool min" in m.lower() or "pool" in m.lower()]
        assert len(pool_warnings) >= 1

    def test_retention_years_below_minimum_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            cfg = DDSCreatorConfig(retention_years=2)
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        retention_warnings = [m for m in warning_msgs if "retention" in m.lower()]
        assert len(retention_warnings) >= 1

    def test_geolocation_precision_too_low_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            cfg = DDSCreatorConfig(geolocation_precision_digits=2)
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        geo_warnings = [m for m in warning_msgs if "precision" in m.lower() or "geolocation" in m.lower()]
        assert len(geo_warnings) >= 1

    def test_attachment_exceeds_package_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            cfg = DDSCreatorConfig(max_attachment_size_mb=600, max_package_size_mb=500)
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        size_warnings = [m for m in warning_msgs if "attachment" in m.lower() or "package" in m.lower()]
        assert len(size_warnings) >= 1

    def test_default_language_not_in_supported_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            cfg = DDSCreatorConfig(default_language="xx")
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        lang_warnings = [m for m in warning_msgs if "language" in m.lower()]
        assert len(lang_warnings) >= 1

    def test_signature_validity_too_short_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            cfg = DDSCreatorConfig(signature_validity_days=30)
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        sig_warnings = [m for m in warning_msgs if "signature" in m.lower() or "validity" in m.lower()]
        assert len(sig_warnings) >= 1

    def test_generation_timeout_too_short_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            cfg = DDSCreatorConfig(statement_generation_timeout_seconds=10)
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        timeout_warnings = [m for m in warning_msgs if "timeout" in m.lower()]
        assert len(timeout_warnings) >= 1

    def test_valid_config_logs_info(self, caplog):
        with caplog.at_level(logging.INFO):
            cfg = DDSCreatorConfig()
        info_msgs = [r.message for r in caplog.records if r.levelno == logging.INFO]
        init_msgs = [m for m in info_msgs if "initialized" in m.lower()]
        assert len(init_msgs) >= 1


class TestConfigEnvHelpers:
    """Test the module-level _env_* helper functions."""

    def test_env_returns_default_when_missing(self):
        result = _env("NONEXISTENT_KEY_12345", "default_val")
        assert result == "default_val"

    def test_env_returns_value_when_set(self):
        with patch.dict(os.environ, {"GL_EUDR_DDSC_TEST_KEY": "hello"}):
            result = _env("TEST_KEY", "default")
            assert result == "hello"

    def test_env_int_returns_default(self):
        result = _env_int("NONEXISTENT_INT", 42)
        assert result == 42

    def test_env_int_returns_parsed(self):
        with patch.dict(os.environ, {"GL_EUDR_DDSC_TEST_INT": "99"}):
            result = _env_int("TEST_INT", 0)
            assert result == 99

    def test_env_float_returns_default(self):
        result = _env_float("NONEXISTENT_FLOAT", 3.14)
        assert result == 3.14

    def test_env_float_returns_parsed(self):
        with patch.dict(os.environ, {"GL_EUDR_DDSC_TEST_FLOAT": "2.71"}):
            result = _env_float("TEST_FLOAT", 0.0)
            assert result == 2.71

    def test_env_bool_returns_default_false(self):
        result = _env_bool("NONEXISTENT_BOOL", False)
        assert result is False

    def test_env_bool_true_values(self):
        for val in ("true", "1", "yes"):
            with patch.dict(os.environ, {"GL_EUDR_DDSC_TEST_BOOL": val}):
                result = _env_bool("TEST_BOOL", False)
                assert result is True, f"Expected True for {val!r}"

    def test_env_bool_false_values(self):
        for val in ("false", "0", "no", "anything"):
            with patch.dict(os.environ, {"GL_EUDR_DDSC_TEST_BOOL": val}):
                result = _env_bool("TEST_BOOL", True)
                assert result is False, f"Expected False for {val!r}"

    def test_env_decimal_returns_default(self):
        result = _env_decimal("NONEXISTENT_DEC", "99.99")
        assert result == Decimal("99.99")

    def test_env_decimal_returns_parsed(self):
        with patch.dict(os.environ, {"GL_EUDR_DDSC_TEST_DEC": "42.5"}):
            result = _env_decimal("TEST_DEC", "0")
            assert result == Decimal("42.5")

    def test_env_prefix_is_correct(self):
        assert _ENV_PREFIX == "GL_EUDR_DDSC_"


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

    def test_config_has_dds_template_settings(self, sample_config):
        assert hasattr(sample_config, 'dds_template_version')
        assert hasattr(sample_config, 'dds_max_commodities_per_statement')
        assert hasattr(sample_config, 'dds_max_plots_per_commodity')

    def test_config_has_language_settings(self, sample_config):
        assert hasattr(sample_config, 'supported_languages')
        assert hasattr(sample_config, 'default_language')

    def test_config_has_signature_settings(self, sample_config):
        assert hasattr(sample_config, 'signature_algorithm')
        assert hasattr(sample_config, 'signature_validity_days')
        assert hasattr(sample_config, 'require_qualified_signature')

    def test_config_has_provenance_settings(self, sample_config):
        assert hasattr(sample_config, 'provenance_enabled')
        assert hasattr(sample_config, 'provenance_algorithm')
        assert sample_config.provenance_algorithm == "sha256"


class TestConfigMethods:
    """Test config helper methods."""

    def test_get_rate_limit_anonymous(self, sample_config):
        assert sample_config.get_rate_limit("anonymous") == 10

    def test_get_rate_limit_basic(self, sample_config):
        assert sample_config.get_rate_limit("basic") == 30

    def test_get_rate_limit_standard(self, sample_config):
        assert sample_config.get_rate_limit("standard") == 100

    def test_get_rate_limit_premium(self, sample_config):
        assert sample_config.get_rate_limit("premium") == 500

    def test_get_rate_limit_admin(self, sample_config):
        assert sample_config.get_rate_limit("admin") == 2000

    def test_get_rate_limit_unknown_returns_standard(self, sample_config):
        assert sample_config.get_rate_limit("unknown") == 100

    def test_get_upstream_urls_returns_dict(self, sample_config):
        urls = sample_config.get_upstream_urls()
        assert isinstance(urls, dict)
        assert len(urls) >= 9

    def test_get_upstream_urls_has_supply_chain(self, sample_config):
        urls = sample_config.get_upstream_urls()
        assert "supply_chain_mapper" in urls

    def test_get_upstream_urls_has_geolocation(self, sample_config):
        urls = sample_config.get_upstream_urls()
        assert "geolocation_verification" in urls

    def test_get_upstream_urls_has_eu_is(self, sample_config):
        urls = sample_config.get_upstream_urls()
        assert "eu_information_system" in urls
