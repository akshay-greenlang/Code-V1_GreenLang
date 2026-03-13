# -*- coding: utf-8 -*-
"""
Unit tests for DocumentationGeneratorConfig - AGENT-EUDR-030

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

from greenlang.agents.eudr.documentation_generator.config import (
    DocumentationGeneratorConfig,
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

    def test_dds_reference_prefix_default(self, sample_config):
        """Test default DDS reference prefix is DDS."""
        assert sample_config.dds_reference_prefix == "DDS"

    def test_dds_schema_version_default(self, sample_config):
        """Test default DDS schema version is 1.0."""
        assert sample_config.dds_schema_version == "1.0"

    def test_max_products_per_dds_default(self, sample_config):
        """Test default max products per DDS is 100."""
        assert sample_config.max_products_per_dds == 100

    def test_include_provenance_default(self, sample_config):
        """Test default include provenance is True."""
        assert sample_config.include_provenance is True

    def test_article9_completeness_threshold_default(self, sample_config):
        """Test default Article 9 completeness threshold is 0.95."""
        assert sample_config.article9_completeness_threshold == Decimal("0.95")

    def test_require_polygon_above_4ha_default(self, sample_config):
        """Test default require polygon above 4ha is True."""
        assert sample_config.require_polygon_above_4ha is True

    def test_geolocation_decimal_places_default(self, sample_config):
        """Test default geolocation decimal places is 6."""
        assert sample_config.geolocation_decimal_places == 6

    def test_include_criterion_details_default(self, sample_config):
        """Test default include criterion details is True."""
        assert sample_config.include_criterion_details is True

    def test_include_decomposition_default(self, sample_config):
        """Test default include decomposition is True."""
        assert sample_config.include_decomposition is True

    def test_include_trend_data_default(self, sample_config):
        """Test default include trend data is True."""
        assert sample_config.include_trend_data is True

    def test_include_evidence_summary_default(self, sample_config):
        """Test default include evidence summary is True."""
        assert sample_config.include_evidence_summary is True

    def test_include_timeline_default(self, sample_config):
        """Test default include timeline is True."""
        assert sample_config.include_timeline is True

    def test_include_effectiveness_default(self, sample_config):
        """Test default include effectiveness is True."""
        assert sample_config.include_effectiveness is True

    def test_package_format_default(self, sample_config):
        """Test default package format is json."""
        assert sample_config.package_format == "json"

    def test_include_cross_references_default(self, sample_config):
        """Test default include cross references is True."""
        assert sample_config.include_cross_references is True

    def test_include_table_of_contents_default(self, sample_config):
        """Test default include table of contents is True."""
        assert sample_config.include_table_of_contents is True

    def test_max_package_size_mb_default(self, sample_config):
        """Test default max package size is 500 MB."""
        assert sample_config.max_package_size_mb == 500

    def test_max_versions_per_document_default(self, sample_config):
        """Test default max versions per document is 50."""
        assert sample_config.max_versions_per_document == 50

    def test_retention_years_default(self, sample_config):
        """Test default retention years is 5."""
        assert sample_config.retention_years == 5

    def test_enable_amendment_tracking_default(self, sample_config):
        """Test default enable amendment tracking is True."""
        assert sample_config.enable_amendment_tracking is True

    def test_submission_timeout_seconds_default(self, sample_config):
        """Test default submission timeout is 60 seconds."""
        assert sample_config.submission_timeout_seconds == 60

    def test_max_retries_default(self, sample_config):
        """Test default max retries is 3."""
        assert sample_config.max_retries == 3

    def test_retry_delay_seconds_default(self, sample_config):
        """Test default retry delay is 10 seconds."""
        assert sample_config.retry_delay_seconds == 10

    def test_batch_size_default(self, sample_config):
        """Test default batch size is 10."""
        assert sample_config.batch_size == 10

    def test_eu_information_system_url_default(self, sample_config):
        """Test default EU Information System URL."""
        assert sample_config.eu_information_system_url == "https://eudr-is.europa.eu/api/v1"

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

    def test_circuit_breaker_failure_threshold_default(self, sample_config):
        """Test default circuit breaker failure threshold is 5."""
        assert sample_config.circuit_breaker_failure_threshold == 5

    def test_circuit_breaker_reset_timeout_default(self, sample_config):
        """Test default circuit breaker reset timeout is 60."""
        assert sample_config.circuit_breaker_reset_timeout == 60

    def test_circuit_breaker_half_open_max_default(self, sample_config):
        """Test default circuit breaker half open max is 3."""
        assert sample_config.circuit_breaker_half_open_max == 3

    def test_max_concurrent_default(self, sample_config):
        """Test default max concurrent is 10."""
        assert sample_config.max_concurrent == 10

    def test_batch_timeout_seconds_default(self, sample_config):
        """Test default batch timeout is 300 seconds."""
        assert sample_config.batch_timeout_seconds == 300

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

    def test_metrics_enabled_default(self, sample_config):
        """Test default metrics enabled is True."""
        assert sample_config.metrics_enabled is True

    def test_metrics_prefix_default(self, sample_config):
        """Test default metrics prefix is gl_eudr_dgn_."""
        assert sample_config.metrics_prefix == "gl_eudr_dgn_"

    def test_log_level_default(self, sample_config):
        """Test default log level is INFO."""
        assert sample_config.log_level == "INFO"


class TestConfigEnvOverride:
    """Test that environment variable overrides work correctly."""

    def test_env_override_db_host(self):
        """Test environment override for database host."""
        with patch.dict(os.environ, {"GL_EUDR_DGN_DB_HOST": "my-db.example.com"}):
            cfg = DocumentationGeneratorConfig()
            assert cfg.db_host == "my-db.example.com"

    def test_env_override_db_port(self):
        """Test environment override for database port."""
        with patch.dict(os.environ, {"GL_EUDR_DGN_DB_PORT": "5433"}):
            cfg = DocumentationGeneratorConfig()
            assert cfg.db_port == 5433

    def test_env_override_dds_reference_prefix(self):
        """Test environment override for DDS reference prefix."""
        with patch.dict(os.environ, {"GL_EUDR_DGN_DDS_REFERENCE_PREFIX": "EUDR-DDS"}):
            cfg = DocumentationGeneratorConfig()
            assert cfg.dds_reference_prefix == "EUDR-DDS"

    def test_env_override_max_products_per_dds(self):
        """Test environment override for max products per DDS."""
        with patch.dict(os.environ, {"GL_EUDR_DGN_MAX_PRODUCTS_PER_DDS": "200"}):
            cfg = DocumentationGeneratorConfig()
            assert cfg.max_products_per_dds == 200

    def test_env_override_article9_completeness_threshold(self):
        """Test environment override for Article 9 completeness threshold."""
        with patch.dict(os.environ, {"GL_EUDR_DGN_ARTICLE9_COMPLETENESS_THRESHOLD": "0.90"}):
            cfg = DocumentationGeneratorConfig()
            assert cfg.article9_completeness_threshold == Decimal("0.90")

    def test_env_override_bool_true(self):
        """Test environment override for boolean value set to true."""
        with patch.dict(os.environ, {"GL_EUDR_DGN_INCLUDE_PROVENANCE": "true"}):
            cfg = DocumentationGeneratorConfig()
            assert cfg.include_provenance is True

    def test_env_override_bool_false(self):
        """Test environment override for boolean value set to false."""
        with patch.dict(os.environ, {"GL_EUDR_DGN_INCLUDE_PROVENANCE": "false"}):
            cfg = DocumentationGeneratorConfig()
            assert cfg.include_provenance is False

    def test_env_override_geolocation_decimal_places(self):
        """Test environment override for geolocation decimal places."""
        with patch.dict(os.environ, {"GL_EUDR_DGN_GEOLOCATION_DECIMAL_PLACES": "8"}):
            cfg = DocumentationGeneratorConfig()
            assert cfg.geolocation_decimal_places == 8

    def test_env_override_package_format(self):
        """Test environment override for package format."""
        with patch.dict(os.environ, {"GL_EUDR_DGN_PACKAGE_FORMAT": "xml"}):
            cfg = DocumentationGeneratorConfig()
            assert cfg.package_format == "xml"

    def test_env_override_max_package_size_mb(self):
        """Test environment override for max package size."""
        with patch.dict(os.environ, {"GL_EUDR_DGN_MAX_PACKAGE_SIZE_MB": "1000"}):
            cfg = DocumentationGeneratorConfig()
            assert cfg.max_package_size_mb == 1000

    def test_env_override_retention_years(self):
        """Test environment override for retention years."""
        with patch.dict(os.environ, {"GL_EUDR_DGN_RETENTION_YEARS": "7"}):
            cfg = DocumentationGeneratorConfig()
            assert cfg.retention_years == 7

    def test_env_override_submission_timeout_seconds(self):
        """Test environment override for submission timeout."""
        with patch.dict(os.environ, {"GL_EUDR_DGN_SUBMISSION_TIMEOUT_SECONDS": "120"}):
            cfg = DocumentationGeneratorConfig()
            assert cfg.submission_timeout_seconds == 120

    def test_env_override_max_retries(self):
        """Test environment override for max retries."""
        with patch.dict(os.environ, {"GL_EUDR_DGN_MAX_RETRIES": "5"}):
            cfg = DocumentationGeneratorConfig()
            assert cfg.max_retries == 5


class TestConfigSingleton:
    """Test singleton pattern via get_config()."""

    def test_get_config_returns_instance(self):
        """Test that get_config returns a DocumentationGeneratorConfig instance."""
        cfg = get_config()
        assert isinstance(cfg, DocumentationGeneratorConfig)

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
        with patch.dict(os.environ, {"GL_EUDR_DGN_DB_PORT": "9999"}):
            cfg2 = get_config()
            assert cfg2.db_port == 9999


class TestConfigValidation:
    """Test config __post_init__ validation logic."""

    def test_valid_article9_threshold_no_warning(self, caplog):
        """Test that valid Article 9 threshold does not produce warning."""
        with caplog.at_level(logging.WARNING):
            cfg = DocumentationGeneratorConfig()
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        threshold_warnings = [m for m in warning_msgs if "Article 9 completeness threshold" in m]
        assert len(threshold_warnings) == 0

    def test_article9_threshold_above_one_warns(self, caplog):
        """Test that Article 9 threshold above 1.0 produces warning."""
        with caplog.at_level(logging.WARNING):
            cfg = DocumentationGeneratorConfig()
            cfg.article9_completeness_threshold = Decimal("1.5")
            cfg.__post_init__()
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        threshold_warnings = [m for m in warning_msgs if "Article 9 completeness threshold" in m]
        assert len(threshold_warnings) >= 1

    def test_article9_threshold_below_zero_warns(self, caplog):
        """Test that Article 9 threshold below 0.0 produces warning."""
        with caplog.at_level(logging.WARNING):
            cfg = DocumentationGeneratorConfig()
            cfg.article9_completeness_threshold = Decimal("-0.1")
            cfg.__post_init__()
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        threshold_warnings = [m for m in warning_msgs if "Article 9 completeness threshold" in m]
        assert len(threshold_warnings) >= 1

    def test_geolocation_decimal_places_too_low_warns(self, caplog):
        """Test that geolocation decimal places below 1 produces warning."""
        with caplog.at_level(logging.WARNING):
            cfg = DocumentationGeneratorConfig()
            cfg.geolocation_decimal_places = 0
            cfg.__post_init__()
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        geo_warnings = [m for m in warning_msgs if "Geolocation decimal places" in m]
        assert len(geo_warnings) >= 1

    def test_geolocation_decimal_places_too_high_warns(self, caplog):
        """Test that geolocation decimal places above 15 produces warning."""
        with caplog.at_level(logging.WARNING):
            cfg = DocumentationGeneratorConfig()
            cfg.geolocation_decimal_places = 20
            cfg.__post_init__()
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        geo_warnings = [m for m in warning_msgs if "Geolocation decimal places" in m]
        assert len(geo_warnings) >= 1

    def test_max_products_per_dds_below_one_warns(self, caplog):
        """Test that max products per DDS below 1 produces warning."""
        with caplog.at_level(logging.WARNING):
            cfg = DocumentationGeneratorConfig()
            cfg.max_products_per_dds = 0
            cfg.__post_init__()
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        product_warnings = [m for m in warning_msgs if "Max products per DDS" in m]
        assert len(product_warnings) >= 1

    def test_max_package_size_below_one_warns(self, caplog):
        """Test that max package size below 1 MB produces warning."""
        with caplog.at_level(logging.WARNING):
            cfg = DocumentationGeneratorConfig()
            cfg.max_package_size_mb = 0
            cfg.__post_init__()
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        size_warnings = [m for m in warning_msgs if "Max package size" in m]
        assert len(size_warnings) >= 1

    def test_pool_min_exceeds_max_warns(self, caplog):
        """Test that pool min exceeding max produces warning."""
        with caplog.at_level(logging.WARNING):
            cfg = DocumentationGeneratorConfig()
            cfg.db_pool_min = 20
            cfg.db_pool_max = 5
            cfg.__post_init__()
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        pool_warnings = [m for m in warning_msgs if "pool min" in m]
        assert len(pool_warnings) >= 1

    def test_negative_max_retries_warns(self, caplog):
        """Test that negative max retries produces warning."""
        with caplog.at_level(logging.WARNING):
            cfg = DocumentationGeneratorConfig()
            cfg.max_retries = -1
            cfg.__post_init__()
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        retry_warnings = [m for m in warning_msgs if "Max retries" in m]
        assert len(retry_warnings) >= 1

    def test_retention_years_below_five_warns(self, caplog):
        """Test that retention years below 5 produces warning (EUDR Article 31)."""
        with caplog.at_level(logging.WARNING):
            cfg = DocumentationGeneratorConfig()
            cfg.retention_years = 3
            cfg.__post_init__()
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        retention_warnings = [m for m in warning_msgs if "Retention years" in m]
        assert len(retention_warnings) >= 1

    def test_max_versions_below_one_warns(self, caplog):
        """Test that max versions per document below 1 produces warning."""
        with caplog.at_level(logging.WARNING):
            cfg = DocumentationGeneratorConfig()
            cfg.max_versions_per_document = 0
            cfg.__post_init__()
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        version_warnings = [m for m in warning_msgs if "Max versions per document" in m]
        assert len(version_warnings) >= 1

    def test_invalid_package_format_warns(self, caplog):
        """Test that invalid package format produces warning."""
        with caplog.at_level(logging.WARNING):
            cfg = DocumentationGeneratorConfig()
            cfg.package_format = "invalid_format"
            cfg.__post_init__()
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        format_warnings = [m for m in warning_msgs if "Package format" in m]
        assert len(format_warnings) >= 1


class TestConfigEnvHelpers:
    """Test the module-level _env_* helper functions."""

    def test_env_returns_default_when_missing(self):
        """Test _env returns default when environment variable is missing."""
        result = _env("NONEXISTENT_KEY_12345", "default_val")
        assert result == "default_val"

    def test_env_returns_value_when_set(self):
        """Test _env returns value when environment variable is set."""
        with patch.dict(os.environ, {"GL_EUDR_DGN_TEST_KEY": "hello"}):
            result = _env("TEST_KEY", "default")
            assert result == "hello"

    def test_env_int_returns_default(self):
        """Test _env_int returns default when environment variable is missing."""
        result = _env_int("NONEXISTENT_INT", 42)
        assert result == 42

    def test_env_int_returns_parsed(self):
        """Test _env_int returns parsed integer value."""
        with patch.dict(os.environ, {"GL_EUDR_DGN_TEST_INT": "99"}):
            result = _env_int("TEST_INT", 0)
            assert result == 99

    def test_env_float_returns_default(self):
        """Test _env_float returns default when environment variable is missing."""
        result = _env_float("NONEXISTENT_FLOAT", 3.14)
        assert result == 3.14

    def test_env_float_returns_parsed(self):
        """Test _env_float returns parsed float value."""
        with patch.dict(os.environ, {"GL_EUDR_DGN_TEST_FLOAT": "2.71"}):
            result = _env_float("TEST_FLOAT", 0.0)
            assert result == 2.71

    def test_env_bool_returns_default_false(self):
        """Test _env_bool returns default False when environment variable is missing."""
        result = _env_bool("NONEXISTENT_BOOL", False)
        assert result is False

    def test_env_bool_true_values(self):
        """Test _env_bool recognizes various true values."""
        for val in ("true", "1", "yes"):
            with patch.dict(os.environ, {"GL_EUDR_DGN_TEST_BOOL": val}):
                result = _env_bool("TEST_BOOL", False)
                assert result is True, f"Expected True for {val!r}"

    def test_env_bool_false_values(self):
        """Test _env_bool recognizes various false values."""
        for val in ("false", "0", "no", "anything"):
            with patch.dict(os.environ, {"GL_EUDR_DGN_TEST_BOOL": val}):
                result = _env_bool("TEST_BOOL", True)
                assert result is False, f"Expected False for {val!r}"

    def test_env_decimal_returns_default(self):
        """Test _env_decimal returns default when environment variable is missing."""
        result = _env_decimal("NONEXISTENT_DEC", "99.99")
        assert result == Decimal("99.99")

    def test_env_decimal_returns_parsed(self):
        """Test _env_decimal returns parsed Decimal value."""
        with patch.dict(os.environ, {"GL_EUDR_DGN_TEST_DEC": "42.5"}):
            result = _env_decimal("TEST_DEC", "0")
            assert result == Decimal("42.5")

    def test_env_prefix_is_correct(self):
        """Test that _ENV_PREFIX is GL_EUDR_DGN_."""
        assert _ENV_PREFIX == "GL_EUDR_DGN_"
