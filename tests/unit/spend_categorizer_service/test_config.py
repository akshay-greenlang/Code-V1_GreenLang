# -*- coding: utf-8 -*-
"""
Unit tests for SpendCategorizerConfig (AGENT-DATA-009)

Tests configuration defaults, environment variable overrides, singleton
lifecycle (get_config / set_config / reset_config), type coercion for
numeric fields, and thread safety of the config singleton.

Target: 70+ tests covering all 32 config fields + lifecycle + threading.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import os
import threading
from dataclasses import fields as dataclass_fields
from unittest.mock import patch

import pytest

from greenlang.spend_categorizer.config import (
    SpendCategorizerConfig,
    get_config,
    reset_config,
    set_config,
)


# ============================================================================
# Default value tests for all 32 fields
# ============================================================================


class TestSpendCategorizerConfigDefaults:
    """Verify every default value on a freshly created config."""

    # -- Connections --

    def test_default_database_url_is_empty(self):
        cfg = SpendCategorizerConfig()
        assert cfg.database_url == ""

    def test_default_redis_url_is_empty(self):
        cfg = SpendCategorizerConfig()
        assert cfg.redis_url == ""

    def test_default_s3_bucket_url_is_empty(self):
        cfg = SpendCategorizerConfig()
        assert cfg.s3_bucket_url == ""

    # -- Logging --

    def test_default_log_level_is_info(self):
        cfg = SpendCategorizerConfig()
        assert cfg.log_level == "INFO"

    # -- Classification defaults --

    def test_default_currency_is_usd(self):
        cfg = SpendCategorizerConfig()
        assert cfg.default_currency == "USD"

    def test_default_taxonomy_is_unspsc(self):
        cfg = SpendCategorizerConfig()
        assert cfg.default_taxonomy == "unspsc"

    def test_default_min_confidence(self):
        cfg = SpendCategorizerConfig()
        assert cfg.min_confidence == 0.3

    def test_default_high_confidence_threshold(self):
        cfg = SpendCategorizerConfig()
        assert cfg.high_confidence_threshold == 0.8

    def test_default_medium_confidence_threshold(self):
        cfg = SpendCategorizerConfig()
        assert cfg.medium_confidence_threshold == 0.5

    # -- Emission factor versions --

    def test_default_eeio_version(self):
        cfg = SpendCategorizerConfig()
        assert cfg.eeio_version == "2024"

    def test_default_exiobase_version(self):
        cfg = SpendCategorizerConfig()
        assert cfg.exiobase_version == "3.8.2"

    def test_default_defra_version(self):
        cfg = SpendCategorizerConfig()
        assert cfg.defra_version == "2025"

    def test_default_ecoinvent_version(self):
        cfg = SpendCategorizerConfig()
        assert cfg.ecoinvent_version == "3.10"

    # -- Processing limits --

    def test_default_batch_size(self):
        cfg = SpendCategorizerConfig()
        assert cfg.batch_size == 1000

    def test_default_max_records(self):
        cfg = SpendCategorizerConfig()
        assert cfg.max_records == 100000

    def test_default_dedup_threshold(self):
        cfg = SpendCategorizerConfig()
        assert cfg.dedup_threshold == 0.85

    def test_default_vendor_normalization(self):
        cfg = SpendCategorizerConfig()
        assert cfg.vendor_normalization is True

    def test_default_max_taxonomy_depth(self):
        cfg = SpendCategorizerConfig()
        assert cfg.max_taxonomy_depth == 6

    # -- Cache --

    def test_default_cache_ttl(self):
        cfg = SpendCategorizerConfig()
        assert cfg.cache_ttl == 3600

    def test_default_cache_emission_factors_ttl(self):
        cfg = SpendCategorizerConfig()
        assert cfg.cache_emission_factors_ttl == 86400

    def test_default_cache_taxonomy_ttl(self):
        cfg = SpendCategorizerConfig()
        assert cfg.cache_taxonomy_ttl == 43200

    # -- Feature toggles --

    def test_default_enable_exiobase(self):
        cfg = SpendCategorizerConfig()
        assert cfg.enable_exiobase is True

    def test_default_enable_defra(self):
        cfg = SpendCategorizerConfig()
        assert cfg.enable_defra is True

    def test_default_enable_ecoinvent(self):
        cfg = SpendCategorizerConfig()
        assert cfg.enable_ecoinvent is False

    def test_default_enable_hotspot_analysis(self):
        cfg = SpendCategorizerConfig()
        assert cfg.enable_hotspot_analysis is True

    def test_default_enable_trend_analysis(self):
        cfg = SpendCategorizerConfig()
        assert cfg.enable_trend_analysis is True

    # -- Rate limiting --

    def test_default_rate_limit_rpm(self):
        cfg = SpendCategorizerConfig()
        assert cfg.rate_limit_rpm == 120

    def test_default_rate_limit_burst(self):
        cfg = SpendCategorizerConfig()
        assert cfg.rate_limit_burst == 20

    # -- Provenance --

    def test_default_enable_provenance(self):
        cfg = SpendCategorizerConfig()
        assert cfg.enable_provenance is True

    def test_default_provenance_hash_algorithm(self):
        cfg = SpendCategorizerConfig()
        assert cfg.provenance_hash_algorithm == "sha256"

    # -- Pool sizing --

    def test_default_pool_min_size(self):
        cfg = SpendCategorizerConfig()
        assert cfg.pool_min_size == 2

    def test_default_pool_max_size(self):
        cfg = SpendCategorizerConfig()
        assert cfg.pool_max_size == 10

    def test_default_worker_count(self):
        cfg = SpendCategorizerConfig()
        assert cfg.worker_count == 4


# ============================================================================
# Environment variable override tests
# ============================================================================


class TestSpendCategorizerConfigEnvOverrides:
    """Verify every GL_SPEND_CAT_* env var overrides the correct field."""

    def test_env_database_url(self, monkeypatch):
        monkeypatch.setenv("GL_SPEND_CAT_DATABASE_URL", "postgresql://test:5432/db")
        cfg = SpendCategorizerConfig.from_env()
        assert cfg.database_url == "postgresql://test:5432/db"

    def test_env_redis_url(self, monkeypatch):
        monkeypatch.setenv("GL_SPEND_CAT_REDIS_URL", "redis://test:6379")
        cfg = SpendCategorizerConfig.from_env()
        assert cfg.redis_url == "redis://test:6379"

    def test_env_s3_bucket_url(self, monkeypatch):
        monkeypatch.setenv("GL_SPEND_CAT_S3_BUCKET_URL", "s3://my-bucket")
        cfg = SpendCategorizerConfig.from_env()
        assert cfg.s3_bucket_url == "s3://my-bucket"

    def test_env_log_level(self, monkeypatch):
        monkeypatch.setenv("GL_SPEND_CAT_LOG_LEVEL", "DEBUG")
        cfg = SpendCategorizerConfig.from_env()
        assert cfg.log_level == "DEBUG"

    def test_env_default_currency(self, monkeypatch):
        monkeypatch.setenv("GL_SPEND_CAT_DEFAULT_CURRENCY", "EUR")
        cfg = SpendCategorizerConfig.from_env()
        assert cfg.default_currency == "EUR"

    def test_env_default_taxonomy(self, monkeypatch):
        monkeypatch.setenv("GL_SPEND_CAT_DEFAULT_TAXONOMY", "naics")
        cfg = SpendCategorizerConfig.from_env()
        assert cfg.default_taxonomy == "naics"

    def test_env_min_confidence(self, monkeypatch):
        monkeypatch.setenv("GL_SPEND_CAT_MIN_CONFIDENCE", "0.5")
        cfg = SpendCategorizerConfig.from_env()
        assert cfg.min_confidence == 0.5

    def test_env_high_confidence_threshold(self, monkeypatch):
        monkeypatch.setenv("GL_SPEND_CAT_HIGH_CONFIDENCE_THRESHOLD", "0.9")
        cfg = SpendCategorizerConfig.from_env()
        assert cfg.high_confidence_threshold == 0.9

    def test_env_medium_confidence_threshold(self, monkeypatch):
        monkeypatch.setenv("GL_SPEND_CAT_MEDIUM_CONFIDENCE_THRESHOLD", "0.6")
        cfg = SpendCategorizerConfig.from_env()
        assert cfg.medium_confidence_threshold == 0.6

    def test_env_eeio_version(self, monkeypatch):
        monkeypatch.setenv("GL_SPEND_CAT_EEIO_VERSION", "2025")
        cfg = SpendCategorizerConfig.from_env()
        assert cfg.eeio_version == "2025"

    def test_env_exiobase_version(self, monkeypatch):
        monkeypatch.setenv("GL_SPEND_CAT_EXIOBASE_VERSION", "4.0")
        cfg = SpendCategorizerConfig.from_env()
        assert cfg.exiobase_version == "4.0"

    def test_env_defra_version(self, monkeypatch):
        monkeypatch.setenv("GL_SPEND_CAT_DEFRA_VERSION", "2026")
        cfg = SpendCategorizerConfig.from_env()
        assert cfg.defra_version == "2026"

    def test_env_ecoinvent_version(self, monkeypatch):
        monkeypatch.setenv("GL_SPEND_CAT_ECOINVENT_VERSION", "3.11")
        cfg = SpendCategorizerConfig.from_env()
        assert cfg.ecoinvent_version == "3.11"

    def test_env_batch_size(self, monkeypatch):
        monkeypatch.setenv("GL_SPEND_CAT_BATCH_SIZE", "500")
        cfg = SpendCategorizerConfig.from_env()
        assert cfg.batch_size == 500

    def test_env_max_records(self, monkeypatch):
        monkeypatch.setenv("GL_SPEND_CAT_MAX_RECORDS", "50000")
        cfg = SpendCategorizerConfig.from_env()
        assert cfg.max_records == 50000

    def test_env_dedup_threshold(self, monkeypatch):
        monkeypatch.setenv("GL_SPEND_CAT_DEDUP_THRESHOLD", "0.95")
        cfg = SpendCategorizerConfig.from_env()
        assert cfg.dedup_threshold == 0.95

    def test_env_vendor_normalization_true(self, monkeypatch):
        monkeypatch.setenv("GL_SPEND_CAT_VENDOR_NORMALIZATION", "true")
        cfg = SpendCategorizerConfig.from_env()
        assert cfg.vendor_normalization is True

    def test_env_vendor_normalization_false(self, monkeypatch):
        monkeypatch.setenv("GL_SPEND_CAT_VENDOR_NORMALIZATION", "false")
        cfg = SpendCategorizerConfig.from_env()
        assert cfg.vendor_normalization is False

    def test_env_vendor_normalization_yes(self, monkeypatch):
        monkeypatch.setenv("GL_SPEND_CAT_VENDOR_NORMALIZATION", "yes")
        cfg = SpendCategorizerConfig.from_env()
        assert cfg.vendor_normalization is True

    def test_env_vendor_normalization_one(self, monkeypatch):
        monkeypatch.setenv("GL_SPEND_CAT_VENDOR_NORMALIZATION", "1")
        cfg = SpendCategorizerConfig.from_env()
        assert cfg.vendor_normalization is True

    def test_env_vendor_normalization_zero(self, monkeypatch):
        monkeypatch.setenv("GL_SPEND_CAT_VENDOR_NORMALIZATION", "0")
        cfg = SpendCategorizerConfig.from_env()
        assert cfg.vendor_normalization is False

    def test_env_max_taxonomy_depth(self, monkeypatch):
        monkeypatch.setenv("GL_SPEND_CAT_MAX_TAXONOMY_DEPTH", "8")
        cfg = SpendCategorizerConfig.from_env()
        assert cfg.max_taxonomy_depth == 8

    def test_env_cache_ttl(self, monkeypatch):
        monkeypatch.setenv("GL_SPEND_CAT_CACHE_TTL", "7200")
        cfg = SpendCategorizerConfig.from_env()
        assert cfg.cache_ttl == 7200

    def test_env_cache_emission_factors_ttl(self, monkeypatch):
        monkeypatch.setenv("GL_SPEND_CAT_CACHE_EMISSION_FACTORS_TTL", "172800")
        cfg = SpendCategorizerConfig.from_env()
        assert cfg.cache_emission_factors_ttl == 172800

    def test_env_cache_taxonomy_ttl(self, monkeypatch):
        monkeypatch.setenv("GL_SPEND_CAT_CACHE_TAXONOMY_TTL", "86400")
        cfg = SpendCategorizerConfig.from_env()
        assert cfg.cache_taxonomy_ttl == 86400

    def test_env_enable_exiobase(self, monkeypatch):
        monkeypatch.setenv("GL_SPEND_CAT_ENABLE_EXIOBASE", "false")
        cfg = SpendCategorizerConfig.from_env()
        assert cfg.enable_exiobase is False

    def test_env_enable_defra(self, monkeypatch):
        monkeypatch.setenv("GL_SPEND_CAT_ENABLE_DEFRA", "false")
        cfg = SpendCategorizerConfig.from_env()
        assert cfg.enable_defra is False

    def test_env_enable_ecoinvent(self, monkeypatch):
        monkeypatch.setenv("GL_SPEND_CAT_ENABLE_ECOINVENT", "true")
        cfg = SpendCategorizerConfig.from_env()
        assert cfg.enable_ecoinvent is True

    def test_env_enable_hotspot_analysis(self, monkeypatch):
        monkeypatch.setenv("GL_SPEND_CAT_ENABLE_HOTSPOT_ANALYSIS", "false")
        cfg = SpendCategorizerConfig.from_env()
        assert cfg.enable_hotspot_analysis is False

    def test_env_enable_trend_analysis(self, monkeypatch):
        monkeypatch.setenv("GL_SPEND_CAT_ENABLE_TREND_ANALYSIS", "false")
        cfg = SpendCategorizerConfig.from_env()
        assert cfg.enable_trend_analysis is False

    def test_env_rate_limit_rpm(self, monkeypatch):
        monkeypatch.setenv("GL_SPEND_CAT_RATE_LIMIT_RPM", "200")
        cfg = SpendCategorizerConfig.from_env()
        assert cfg.rate_limit_rpm == 200

    def test_env_rate_limit_burst(self, monkeypatch):
        monkeypatch.setenv("GL_SPEND_CAT_RATE_LIMIT_BURST", "50")
        cfg = SpendCategorizerConfig.from_env()
        assert cfg.rate_limit_burst == 50

    def test_env_enable_provenance(self, monkeypatch):
        monkeypatch.setenv("GL_SPEND_CAT_ENABLE_PROVENANCE", "false")
        cfg = SpendCategorizerConfig.from_env()
        assert cfg.enable_provenance is False

    def test_env_provenance_hash_algorithm(self, monkeypatch):
        monkeypatch.setenv("GL_SPEND_CAT_PROVENANCE_HASH_ALGORITHM", "sha512")
        cfg = SpendCategorizerConfig.from_env()
        assert cfg.provenance_hash_algorithm == "sha512"

    def test_env_pool_min_size(self, monkeypatch):
        monkeypatch.setenv("GL_SPEND_CAT_POOL_MIN_SIZE", "5")
        cfg = SpendCategorizerConfig.from_env()
        assert cfg.pool_min_size == 5

    def test_env_pool_max_size(self, monkeypatch):
        monkeypatch.setenv("GL_SPEND_CAT_POOL_MAX_SIZE", "20")
        cfg = SpendCategorizerConfig.from_env()
        assert cfg.pool_max_size == 20

    def test_env_worker_count(self, monkeypatch):
        monkeypatch.setenv("GL_SPEND_CAT_WORKER_COUNT", "8")
        cfg = SpendCategorizerConfig.from_env()
        assert cfg.worker_count == 8


# ============================================================================
# Type coercion / invalid env var fallback tests
# ============================================================================


class TestSpendCategorizerConfigTypeCoercion:
    """Verify invalid int/float env vars fall back to defaults."""

    def test_invalid_int_batch_size_falls_back(self, monkeypatch):
        monkeypatch.setenv("GL_SPEND_CAT_BATCH_SIZE", "not_a_number")
        cfg = SpendCategorizerConfig.from_env()
        assert cfg.batch_size == 1000

    def test_invalid_int_max_records_falls_back(self, monkeypatch):
        monkeypatch.setenv("GL_SPEND_CAT_MAX_RECORDS", "abc")
        cfg = SpendCategorizerConfig.from_env()
        assert cfg.max_records == 100000

    def test_invalid_float_min_confidence_falls_back(self, monkeypatch):
        monkeypatch.setenv("GL_SPEND_CAT_MIN_CONFIDENCE", "not_float")
        cfg = SpendCategorizerConfig.from_env()
        assert cfg.min_confidence == 0.3

    def test_invalid_float_dedup_threshold_falls_back(self, monkeypatch):
        monkeypatch.setenv("GL_SPEND_CAT_DEDUP_THRESHOLD", "xyz")
        cfg = SpendCategorizerConfig.from_env()
        assert cfg.dedup_threshold == 0.85

    def test_invalid_int_cache_ttl_falls_back(self, monkeypatch):
        monkeypatch.setenv("GL_SPEND_CAT_CACHE_TTL", "")
        cfg = SpendCategorizerConfig.from_env()
        # Empty string -> logged from _env(), but _int parses "" which fails
        # int("") raises ValueError, so falls back to 3600
        assert cfg.cache_ttl == 3600

    def test_invalid_int_rate_limit_rpm_falls_back(self, monkeypatch):
        monkeypatch.setenv("GL_SPEND_CAT_RATE_LIMIT_RPM", "12.5")
        cfg = SpendCategorizerConfig.from_env()
        # int("12.5") raises ValueError
        assert cfg.rate_limit_rpm == 120

    def test_invalid_int_worker_count_falls_back(self, monkeypatch):
        monkeypatch.setenv("GL_SPEND_CAT_WORKER_COUNT", "")
        cfg = SpendCategorizerConfig.from_env()
        assert cfg.worker_count == 4

    def test_invalid_float_high_confidence_falls_back(self, monkeypatch):
        monkeypatch.setenv("GL_SPEND_CAT_HIGH_CONFIDENCE_THRESHOLD", "high")
        cfg = SpendCategorizerConfig.from_env()
        assert cfg.high_confidence_threshold == 0.8

    def test_invalid_int_pool_max_size_falls_back(self, monkeypatch):
        monkeypatch.setenv("GL_SPEND_CAT_POOL_MAX_SIZE", "ten")
        cfg = SpendCategorizerConfig.from_env()
        assert cfg.pool_max_size == 10

    def test_invalid_int_max_taxonomy_depth_falls_back(self, monkeypatch):
        monkeypatch.setenv("GL_SPEND_CAT_MAX_TAXONOMY_DEPTH", "deep")
        cfg = SpendCategorizerConfig.from_env()
        assert cfg.max_taxonomy_depth == 6

    def test_empty_string_database_url_stays_empty(self, monkeypatch):
        monkeypatch.setenv("GL_SPEND_CAT_DATABASE_URL", "")
        cfg = SpendCategorizerConfig.from_env()
        assert cfg.database_url == ""


# ============================================================================
# Singleton lifecycle: get_config / set_config / reset_config
# ============================================================================


class TestSpendCategorizerConfigSingleton:
    """Verify thread-safe singleton accessor behaviour."""

    def test_get_config_returns_instance(self):
        cfg = get_config()
        assert isinstance(cfg, SpendCategorizerConfig)

    def test_get_config_returns_same_instance(self):
        cfg1 = get_config()
        cfg2 = get_config()
        assert cfg1 is cfg2

    def test_set_config_replaces_singleton(self):
        custom = SpendCategorizerConfig(batch_size=999)
        set_config(custom)
        assert get_config().batch_size == 999

    def test_reset_config_clears_singleton(self):
        _ = get_config()
        reset_config()
        # After reset, get_config creates a new instance
        cfg = get_config()
        assert isinstance(cfg, SpendCategorizerConfig)

    def test_reset_config_then_get_config_creates_fresh(self):
        set_config(SpendCategorizerConfig(batch_size=999))
        reset_config()
        cfg = get_config()
        assert cfg.batch_size == 1000  # default value

    def test_get_config_with_env_override(self, monkeypatch):
        monkeypatch.setenv("GL_SPEND_CAT_BATCH_SIZE", "2000")
        reset_config()
        cfg = get_config()
        assert cfg.batch_size == 2000


# ============================================================================
# Thread safety tests
# ============================================================================


class TestSpendCategorizerConfigThreadSafety:
    """Verify config singleton is thread-safe under concurrent access."""

    def test_concurrent_get_config(self):
        """Multiple threads calling get_config should get the same instance."""
        results = []

        def _get():
            results.append(id(get_config()))

        threads = [threading.Thread(target=_get) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(set(results)) == 1, "All threads must receive the same singleton"

    def test_concurrent_set_and_get(self):
        """Concurrent set_config + get_config should not crash."""
        errors = []

        def _set_then_get(batch_val):
            try:
                set_config(SpendCategorizerConfig(batch_size=batch_val))
                cfg = get_config()
                assert isinstance(cfg, SpendCategorizerConfig)
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=_set_then_get, args=(i * 100,))
            for i in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"


# ============================================================================
# Combined / multiple override tests
# ============================================================================


class TestSpendCategorizerConfigMultipleOverrides:
    """Test overriding multiple config fields at once via env."""

    def test_override_connection_and_processing(self, monkeypatch):
        monkeypatch.setenv("GL_SPEND_CAT_DATABASE_URL", "postgresql://prod:5432/gl")
        monkeypatch.setenv("GL_SPEND_CAT_REDIS_URL", "redis://prod:6379")
        monkeypatch.setenv("GL_SPEND_CAT_BATCH_SIZE", "5000")
        monkeypatch.setenv("GL_SPEND_CAT_MAX_RECORDS", "500000")
        cfg = SpendCategorizerConfig.from_env()
        assert cfg.database_url == "postgresql://prod:5432/gl"
        assert cfg.redis_url == "redis://prod:6379"
        assert cfg.batch_size == 5000
        assert cfg.max_records == 500000

    def test_override_all_toggles_off(self, monkeypatch):
        monkeypatch.setenv("GL_SPEND_CAT_ENABLE_EXIOBASE", "false")
        monkeypatch.setenv("GL_SPEND_CAT_ENABLE_DEFRA", "false")
        monkeypatch.setenv("GL_SPEND_CAT_ENABLE_ECOINVENT", "false")
        monkeypatch.setenv("GL_SPEND_CAT_ENABLE_HOTSPOT_ANALYSIS", "false")
        monkeypatch.setenv("GL_SPEND_CAT_ENABLE_TREND_ANALYSIS", "false")
        monkeypatch.setenv("GL_SPEND_CAT_ENABLE_PROVENANCE", "false")
        cfg = SpendCategorizerConfig.from_env()
        assert cfg.enable_exiobase is False
        assert cfg.enable_defra is False
        assert cfg.enable_ecoinvent is False
        assert cfg.enable_hotspot_analysis is False
        assert cfg.enable_trend_analysis is False
        assert cfg.enable_provenance is False

    def test_override_all_thresholds(self, monkeypatch):
        monkeypatch.setenv("GL_SPEND_CAT_MIN_CONFIDENCE", "0.4")
        monkeypatch.setenv("GL_SPEND_CAT_HIGH_CONFIDENCE_THRESHOLD", "0.95")
        monkeypatch.setenv("GL_SPEND_CAT_MEDIUM_CONFIDENCE_THRESHOLD", "0.65")
        monkeypatch.setenv("GL_SPEND_CAT_DEDUP_THRESHOLD", "0.90")
        cfg = SpendCategorizerConfig.from_env()
        assert cfg.min_confidence == 0.4
        assert cfg.high_confidence_threshold == 0.95
        assert cfg.medium_confidence_threshold == 0.65
        assert cfg.dedup_threshold == 0.90

    def test_override_emission_factor_versions(self, monkeypatch):
        monkeypatch.setenv("GL_SPEND_CAT_EEIO_VERSION", "2025")
        monkeypatch.setenv("GL_SPEND_CAT_EXIOBASE_VERSION", "4.0")
        monkeypatch.setenv("GL_SPEND_CAT_DEFRA_VERSION", "2026")
        monkeypatch.setenv("GL_SPEND_CAT_ECOINVENT_VERSION", "4.0")
        cfg = SpendCategorizerConfig.from_env()
        assert cfg.eeio_version == "2025"
        assert cfg.exiobase_version == "4.0"
        assert cfg.defra_version == "2026"
        assert cfg.ecoinvent_version == "4.0"

    def test_override_pool_sizing(self, monkeypatch):
        monkeypatch.setenv("GL_SPEND_CAT_POOL_MIN_SIZE", "5")
        monkeypatch.setenv("GL_SPEND_CAT_POOL_MAX_SIZE", "25")
        monkeypatch.setenv("GL_SPEND_CAT_WORKER_COUNT", "12")
        cfg = SpendCategorizerConfig.from_env()
        assert cfg.pool_min_size == 5
        assert cfg.pool_max_size == 25
        assert cfg.worker_count == 12


# ============================================================================
# Dataclass introspection tests
# ============================================================================


class TestSpendCategorizerConfigDataclass:
    """Test dataclass properties and equality."""

    def test_equality_of_identical_configs(self):
        cfg1 = SpendCategorizerConfig()
        cfg2 = SpendCategorizerConfig()
        assert cfg1 == cfg2

    def test_inequality_when_field_differs(self):
        cfg1 = SpendCategorizerConfig()
        cfg2 = SpendCategorizerConfig(batch_size=999)
        assert cfg1 != cfg2

    def test_repr_contains_class_name(self):
        cfg = SpendCategorizerConfig()
        r = repr(cfg)
        assert "SpendCategorizerConfig" in r

    def test_repr_contains_field_values(self):
        cfg = SpendCategorizerConfig()
        r = repr(cfg)
        assert "batch_size=1000" in r
        assert "default_currency='USD'" in r

    def test_from_env_returns_correct_type(self):
        cfg = SpendCategorizerConfig.from_env()
        assert isinstance(cfg, SpendCategorizerConfig)

    def test_field_count_is_33(self):
        fields = dataclass_fields(SpendCategorizerConfig)
        assert len(fields) == 33, (
            f"Expected 33 config fields, got {len(fields)}: "
            f"{[f.name for f in fields]}"
        )

    def test_all_field_names(self):
        expected_fields = {
            "database_url", "redis_url", "s3_bucket_url", "log_level",
            "default_currency", "default_taxonomy", "min_confidence",
            "high_confidence_threshold", "medium_confidence_threshold",
            "eeio_version", "exiobase_version", "defra_version", "ecoinvent_version",
            "batch_size", "max_records", "dedup_threshold",
            "vendor_normalization", "max_taxonomy_depth",
            "cache_ttl", "cache_emission_factors_ttl", "cache_taxonomy_ttl",
            "enable_exiobase", "enable_defra", "enable_ecoinvent",
            "enable_hotspot_analysis", "enable_trend_analysis",
            "rate_limit_rpm", "rate_limit_burst",
            "enable_provenance", "provenance_hash_algorithm",
            "pool_min_size", "pool_max_size", "worker_count",
        }
        actual_fields = {f.name for f in dataclass_fields(SpendCategorizerConfig)}
        assert actual_fields == expected_fields

    def test_config_is_mutable(self):
        cfg = SpendCategorizerConfig()
        cfg.batch_size = 42
        assert cfg.batch_size == 42
