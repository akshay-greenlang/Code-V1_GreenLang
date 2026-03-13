# -*- coding: utf-8 -*-
"""
Unit tests for CustomsDeclarationConfig - AGENT-EUDR-039

Tests default values, environment variable overrides, singleton pattern,
validation logic, customs system configurations, tariff settings,
and all env helper functions.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

import os
import logging
from decimal import Decimal
from unittest.mock import patch

import pytest

from greenlang.agents.eudr.customs_declaration_support.config import (
    CustomsDeclarationConfig,
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

    def test_ncts_endpoint_default(self, sample_config):
        assert "ncts" in sample_config.ncts_endpoint.lower() or sample_config.ncts_endpoint != ""

    def test_ais_endpoint_default(self, sample_config):
        assert "ais" in sample_config.ais_endpoint.lower() or sample_config.ais_endpoint != ""

    def test_ncts_timeout_default(self, sample_config):
        assert sample_config.ncts_timeout_seconds == 60

    def test_ais_timeout_default(self, sample_config):
        assert sample_config.ais_timeout_seconds == 60

    def test_mrn_country_code_default(self, sample_config):
        assert len(sample_config.mrn_country_code) == 2

    def test_mrn_year_digits_default(self, sample_config):
        assert sample_config.mrn_year_digits == 2

    def test_customs_office_code_default(self, sample_config):
        assert len(sample_config.customs_office_code) >= 6

    def test_default_currency_default(self, sample_config):
        assert sample_config.default_currency == "EUR"

    def test_vat_rate_default(self, sample_config):
        assert sample_config.default_vat_rate == Decimal("21.0")

    def test_max_line_items_default(self, sample_config):
        assert sample_config.max_line_items_per_declaration == 999

    def test_sad_form_version_default(self, sample_config):
        assert sample_config.sad_form_version == "1.0"

    def test_declaration_retention_years_default(self, sample_config):
        assert sample_config.declaration_retention_years == 5

    def test_batch_size_default(self, sample_config):
        assert sample_config.batch_size == 50

    def test_max_concurrent_default(self, sample_config):
        assert sample_config.max_concurrent == 10

    def test_provenance_enabled_default(self, sample_config):
        assert sample_config.provenance_enabled is True

    def test_provenance_algorithm_default(self, sample_config):
        assert sample_config.provenance_algorithm == "sha256"

    def test_metrics_enabled_default(self, sample_config):
        assert sample_config.metrics_enabled is True

    def test_metrics_prefix_default(self, sample_config):
        assert sample_config.metrics_prefix == "gl_eudr_cds_"

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

    def test_batch_timeout_seconds_default(self, sample_config):
        assert sample_config.batch_timeout_seconds == 300

    def test_log_level_default(self, sample_config):
        assert sample_config.log_level == "INFO"

    def test_parallel_engine_calls_default(self, sample_config):
        assert sample_config.parallel_engine_calls == 5

    def test_exchange_rate_cache_ttl_default(self, sample_config):
        assert sample_config.exchange_rate_cache_ttl == 3600

    def test_supported_currencies_count(self, sample_config):
        assert len(sample_config.supported_currencies) >= 4

    def test_supported_currencies_has_eur(self, sample_config):
        assert "EUR" in sample_config.supported_currencies

    def test_supported_currencies_has_usd(self, sample_config):
        assert "USD" in sample_config.supported_currencies

    def test_supported_currencies_has_gbp(self, sample_config):
        assert "GBP" in sample_config.supported_currencies

    def test_eudr_dds_validation_enabled_default(self, sample_config):
        assert sample_config.eudr_dds_validation_enabled is True

    def test_origin_cross_check_enabled_default(self, sample_config):
        assert sample_config.origin_cross_check_enabled is True

    def test_deforestation_cutoff_date_default(self, sample_config):
        assert sample_config.deforestation_cutoff_date == "2020-12-31"


class TestCustomsSystemConfigDefaults:
    """Test customs system specific configuration defaults."""

    def test_ncts_retry_count_default(self, sample_config):
        assert sample_config.ncts_retry_count == 3

    def test_ais_retry_count_default(self, sample_config):
        assert sample_config.ais_retry_count == 3

    def test_customs_submission_timeout_default(self, sample_config):
        assert sample_config.customs_submission_timeout_seconds == 120

    def test_ncts_message_format_default(self, sample_config):
        assert sample_config.ncts_message_format in ("xml", "json")

    def test_ais_message_format_default(self, sample_config):
        assert sample_config.ais_message_format in ("xml", "json")

    def test_customs_auth_method_default(self, sample_config):
        assert sample_config.customs_auth_method in ("certificate", "oauth2", "api_key")

    def test_sad_form_enabled_default(self, sample_config):
        assert sample_config.sad_form_enabled is True

    def test_mrn_generation_enabled_default(self, sample_config):
        assert sample_config.mrn_generation_enabled is True


class TestTariffConfigDefaults:
    """Test tariff-specific configuration defaults."""

    def test_tariff_database_version_default(self, sample_config):
        assert sample_config.tariff_database_version is not None

    def test_preferential_tariff_enabled_default(self, sample_config):
        assert sample_config.preferential_tariff_enabled is True

    def test_anti_dumping_check_enabled_default(self, sample_config):
        assert sample_config.anti_dumping_check_enabled is True

    def test_currency_conversion_precision_default(self, sample_config):
        assert sample_config.currency_conversion_precision >= 2

    def test_tariff_calculation_timeout_default(self, sample_config):
        assert sample_config.tariff_calculation_timeout_seconds == 30

    def test_cn_code_format_digits_default(self, sample_config):
        assert sample_config.cn_code_format_digits == 8

    def test_hs_code_format_digits_default(self, sample_config):
        assert sample_config.hs_code_format_digits == 6


class TestConfigEnvOverride:
    """Test that environment variable overrides work correctly."""

    def test_env_override_db_host(self):
        with patch.dict(os.environ, {"GL_EUDR_CDS_DB_HOST": "my-db.example.com"}):
            cfg = CustomsDeclarationConfig()
            assert cfg.db_host == "my-db.example.com"

    def test_env_override_db_port(self):
        with patch.dict(os.environ, {"GL_EUDR_CDS_DB_PORT": "5433"}):
            cfg = CustomsDeclarationConfig()
            assert cfg.db_port == 5433

    def test_env_override_redis_port(self):
        with patch.dict(os.environ, {"GL_EUDR_CDS_REDIS_PORT": "6380"}):
            cfg = CustomsDeclarationConfig()
            assert cfg.redis_port == 6380

    def test_env_override_cache_ttl(self):
        with patch.dict(os.environ, {"GL_EUDR_CDS_CACHE_TTL": "7200"}):
            cfg = CustomsDeclarationConfig()
            assert cfg.cache_ttl == 7200

    def test_env_override_ncts_timeout(self):
        with patch.dict(os.environ, {"GL_EUDR_CDS_NCTS_TIMEOUT": "90"}):
            cfg = CustomsDeclarationConfig()
            assert cfg.ncts_timeout_seconds == 90

    def test_env_override_ais_timeout(self):
        with patch.dict(os.environ, {"GL_EUDR_CDS_AIS_TIMEOUT": "120"}):
            cfg = CustomsDeclarationConfig()
            assert cfg.ais_timeout_seconds == 120

    def test_env_override_default_currency(self):
        with patch.dict(os.environ, {"GL_EUDR_CDS_DEFAULT_CURRENCY": "USD"}):
            cfg = CustomsDeclarationConfig()
            assert cfg.default_currency == "USD"

    def test_env_override_vat_rate(self):
        with patch.dict(os.environ, {"GL_EUDR_CDS_DEFAULT_VAT_RATE": "19.0"}):
            cfg = CustomsDeclarationConfig()
            assert cfg.default_vat_rate == Decimal("19.0")

    def test_env_override_batch_size(self):
        with patch.dict(os.environ, {"GL_EUDR_CDS_BATCH_SIZE": "100"}):
            cfg = CustomsDeclarationConfig()
            assert cfg.batch_size == 100

    def test_env_override_bool_true(self):
        with patch.dict(os.environ, {"GL_EUDR_CDS_PROVENANCE_ENABLED": "true"}):
            cfg = CustomsDeclarationConfig()
            assert cfg.provenance_enabled is True

    def test_env_override_bool_false(self):
        with patch.dict(os.environ, {"GL_EUDR_CDS_PROVENANCE_ENABLED": "false"}):
            cfg = CustomsDeclarationConfig()
            assert cfg.provenance_enabled is False

    def test_env_override_max_concurrent(self):
        with patch.dict(os.environ, {"GL_EUDR_CDS_MAX_CONCURRENT": "20"}):
            cfg = CustomsDeclarationConfig()
            assert cfg.max_concurrent == 20

    def test_env_override_retention_years(self):
        with patch.dict(os.environ, {"GL_EUDR_CDS_RETENTION_YEARS": "7"}):
            cfg = CustomsDeclarationConfig()
            assert cfg.declaration_retention_years == 7


class TestConfigSingleton:
    """Test singleton pattern via get_config()."""

    def test_get_config_returns_instance(self):
        cfg = get_config()
        assert isinstance(cfg, CustomsDeclarationConfig)

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
        with patch.dict(os.environ, {"GL_EUDR_CDS_DB_PORT": "9999"}):
            cfg2 = get_config()
            assert cfg2.db_port == 9999


class TestConfigValidation:
    """Test config __post_init__ validation logic."""

    def test_pool_min_exceeds_max_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            cfg = CustomsDeclarationConfig(db_pool_min=20, db_pool_max=5)
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        pool_warnings = [m for m in warning_msgs if "pool" in m.lower()]
        assert len(pool_warnings) >= 1

    def test_retention_years_below_minimum_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            cfg = CustomsDeclarationConfig(declaration_retention_years=2)
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        retention_warnings = [m for m in warning_msgs if "retention" in m.lower()]
        assert len(retention_warnings) >= 1

    def test_ncts_timeout_too_short_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            cfg = CustomsDeclarationConfig(ncts_timeout_seconds=5)
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        timeout_warnings = [m for m in warning_msgs if "timeout" in m.lower() or "ncts" in m.lower()]
        assert len(timeout_warnings) >= 1

    def test_ais_timeout_too_short_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            cfg = CustomsDeclarationConfig(ais_timeout_seconds=5)
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        timeout_warnings = [m for m in warning_msgs if "timeout" in m.lower() or "ais" in m.lower()]
        assert len(timeout_warnings) >= 1

    def test_valid_config_logs_info(self, caplog):
        with caplog.at_level(logging.INFO):
            cfg = CustomsDeclarationConfig()
        info_msgs = [r.message for r in caplog.records if r.levelno == logging.INFO]
        init_msgs = [m for m in info_msgs if "initialized" in m.lower()]
        assert len(init_msgs) >= 1


class TestConfigEnvHelpers:
    """Test the module-level _env_* helper functions."""

    def test_env_returns_default_when_missing(self):
        result = _env("NONEXISTENT_KEY_12345", "default_val")
        assert result == "default_val"

    def test_env_returns_value_when_set(self):
        with patch.dict(os.environ, {"GL_EUDR_CDS_TEST_KEY": "hello"}):
            result = _env("TEST_KEY", "default")
            assert result == "hello"

    def test_env_int_returns_default(self):
        result = _env_int("NONEXISTENT_INT", 42)
        assert result == 42

    def test_env_int_returns_parsed(self):
        with patch.dict(os.environ, {"GL_EUDR_CDS_TEST_INT": "99"}):
            result = _env_int("TEST_INT", 0)
            assert result == 99

    def test_env_float_returns_default(self):
        result = _env_float("NONEXISTENT_FLOAT", 3.14)
        assert result == 3.14

    def test_env_float_returns_parsed(self):
        with patch.dict(os.environ, {"GL_EUDR_CDS_TEST_FLOAT": "2.71"}):
            result = _env_float("TEST_FLOAT", 0.0)
            assert result == 2.71

    def test_env_bool_returns_default_false(self):
        result = _env_bool("NONEXISTENT_BOOL", False)
        assert result is False

    def test_env_bool_true_values(self):
        for val in ("true", "1", "yes"):
            with patch.dict(os.environ, {"GL_EUDR_CDS_TEST_BOOL": val}):
                result = _env_bool("TEST_BOOL", False)
                assert result is True, f"Expected True for {val!r}"

    def test_env_bool_false_values(self):
        for val in ("false", "0", "no", "anything"):
            with patch.dict(os.environ, {"GL_EUDR_CDS_TEST_BOOL": val}):
                result = _env_bool("TEST_BOOL", True)
                assert result is False, f"Expected False for {val!r}"

    def test_env_decimal_returns_default(self):
        result = _env_decimal("NONEXISTENT_DEC", "99.99")
        assert result == Decimal("99.99")

    def test_env_decimal_returns_parsed(self):
        with patch.dict(os.environ, {"GL_EUDR_CDS_TEST_DEC": "42.5"}):
            result = _env_decimal("TEST_DEC", "0")
            assert result == Decimal("42.5")

    def test_env_prefix_is_correct(self):
        assert _ENV_PREFIX == "GL_EUDR_CDS_"


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

    def test_config_has_customs_system_attributes(self, sample_config):
        assert hasattr(sample_config, 'ncts_endpoint')
        assert hasattr(sample_config, 'ais_endpoint')
        assert hasattr(sample_config, 'ncts_timeout_seconds')
        assert hasattr(sample_config, 'ais_timeout_seconds')

    def test_config_has_tariff_attributes(self, sample_config):
        assert hasattr(sample_config, 'default_currency')
        assert hasattr(sample_config, 'default_vat_rate')
        assert hasattr(sample_config, 'cn_code_format_digits')
        assert hasattr(sample_config, 'hs_code_format_digits')

    def test_config_has_provenance_settings(self, sample_config):
        assert hasattr(sample_config, 'provenance_enabled')
        assert hasattr(sample_config, 'provenance_algorithm')
        assert sample_config.provenance_algorithm == "sha256"

    def test_config_has_eudr_compliance_settings(self, sample_config):
        assert hasattr(sample_config, 'eudr_dds_validation_enabled')
        assert hasattr(sample_config, 'origin_cross_check_enabled')
        assert hasattr(sample_config, 'deforestation_cutoff_date')


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
        assert len(urls) >= 5

    def test_get_upstream_urls_has_dds_creator(self, sample_config):
        urls = sample_config.get_upstream_urls()
        assert "dds_creator" in urls

    def test_get_upstream_urls_has_supply_chain(self, sample_config):
        urls = sample_config.get_upstream_urls()
        assert "supply_chain_mapper" in urls

    def test_get_upstream_urls_has_eu_is(self, sample_config):
        urls = sample_config.get_upstream_urls()
        assert "eu_information_system" in urls

    def test_get_upstream_urls_has_country_risk(self, sample_config):
        urls = sample_config.get_upstream_urls()
        assert "country_risk_evaluator" in urls
