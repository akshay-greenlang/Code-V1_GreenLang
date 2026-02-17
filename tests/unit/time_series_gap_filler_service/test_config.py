# -*- coding: utf-8 -*-
"""
Unit tests for TimeSeriesGapFillerConfig - AGENT-DATA-014

Tests the config dataclass at greenlang.time_series_gap_filler.config with
70+ tests covering default values for all 22 fields, GL_TSGF_ environment
variable overrides, singleton pattern, thread safety, validation logic,
type coercion, invalid env fallback, repr/str, and edge cases.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-014 Time Series Gap Filler (GL-DATA-X-017)
"""

from __future__ import annotations

import concurrent.futures
import threading
from dataclasses import fields as dc_fields

import pytest

from greenlang.time_series_gap_filler.config import (
    TimeSeriesGapFillerConfig,
    get_config,
    reset_config,
    set_config,
)


# ======================================================================
# 1. Default values -- all 22 fields (24 tests)
# ======================================================================


class TestDefaultValues:
    """Verify every field has the expected default."""

    def test_default_database_url(self, fresh_config):
        assert fresh_config.database_url == ""

    def test_default_redis_url(self, fresh_config):
        assert fresh_config.redis_url == ""

    def test_default_log_level(self, fresh_config):
        assert fresh_config.log_level == "INFO"

    def test_default_batch_size(self, fresh_config):
        assert fresh_config.batch_size == 1000

    def test_default_max_records(self, fresh_config):
        assert fresh_config.max_records == 100_000

    def test_default_max_gap_ratio(self, fresh_config):
        assert fresh_config.max_gap_ratio == 0.5

    def test_default_min_data_points(self, fresh_config):
        assert fresh_config.min_data_points == 10

    def test_default_default_strategy(self, fresh_config):
        assert fresh_config.default_strategy == "auto"

    def test_default_interpolation_method(self, fresh_config):
        assert fresh_config.interpolation_method == "linear"

    def test_default_seasonal_periods(self, fresh_config):
        assert fresh_config.seasonal_periods == 12

    def test_default_smoothing_alpha(self, fresh_config):
        assert fresh_config.smoothing_alpha == pytest.approx(0.3)

    def test_default_smoothing_beta(self, fresh_config):
        assert fresh_config.smoothing_beta == pytest.approx(0.1)

    def test_default_smoothing_gamma(self, fresh_config):
        assert fresh_config.smoothing_gamma == pytest.approx(0.1)

    def test_default_correlation_threshold(self, fresh_config):
        assert fresh_config.correlation_threshold == pytest.approx(0.7)

    def test_default_confidence_threshold(self, fresh_config):
        assert fresh_config.confidence_threshold == pytest.approx(0.6)

    def test_default_enable_seasonal(self, fresh_config):
        assert fresh_config.enable_seasonal is True

    def test_default_short_gap_limit(self, fresh_config):
        assert fresh_config.short_gap_limit == 3

    def test_default_long_gap_limit(self, fresh_config):
        assert fresh_config.long_gap_limit == 12

    def test_default_enable_cross_series(self, fresh_config):
        assert fresh_config.enable_cross_series is True

    def test_default_worker_count(self, fresh_config):
        assert fresh_config.worker_count == 4

    def test_default_pool_min_size(self, fresh_config):
        assert fresh_config.pool_min_size == 2

    def test_default_pool_max_size(self, fresh_config):
        assert fresh_config.pool_max_size == 10

    def test_default_cache_ttl(self, fresh_config):
        assert fresh_config.cache_ttl == 3600

    def test_default_rate_limit_rpm(self, fresh_config):
        assert fresh_config.rate_limit_rpm == 120

    def test_default_enable_provenance(self, fresh_config):
        assert fresh_config.enable_provenance is True

    def test_total_field_count(self, fresh_config):
        """Config dataclass should have exactly 25 fields."""
        assert len(dc_fields(fresh_config)) == 25


# ======================================================================
# 2. Environment variable prefix
# ======================================================================


class TestEnvPrefix:
    """Test the GL_TSGF_ environment variable prefix."""

    def test_env_prefix_used(self, monkeypatch):
        monkeypatch.setenv("GL_TSGF_BATCH_SIZE", "500")
        cfg = TimeSeriesGapFillerConfig.from_env()
        assert cfg.batch_size == 500

    def test_wrong_prefix_ignored(self, monkeypatch):
        monkeypatch.setenv("GL_OTHER_BATCH_SIZE", "999")
        cfg = TimeSeriesGapFillerConfig.from_env()
        assert cfg.batch_size == 1000  # unchanged


# ======================================================================
# 3. Environment variable overrides -- one per field (26 tests)
# ======================================================================


class TestEnvOverrides:
    """Each field can be overridden via GL_TSGF_<FIELD_UPPER>."""

    def test_env_database_url(self, monkeypatch):
        monkeypatch.setenv("GL_TSGF_DATABASE_URL", "postgresql://host/db")
        cfg = TimeSeriesGapFillerConfig.from_env()
        assert cfg.database_url == "postgresql://host/db"

    def test_env_redis_url(self, monkeypatch):
        monkeypatch.setenv("GL_TSGF_REDIS_URL", "redis://host:6379/1")
        cfg = TimeSeriesGapFillerConfig.from_env()
        assert cfg.redis_url == "redis://host:6379/1"

    def test_env_log_level(self, monkeypatch):
        monkeypatch.setenv("GL_TSGF_LOG_LEVEL", "DEBUG")
        cfg = TimeSeriesGapFillerConfig.from_env()
        assert cfg.log_level == "DEBUG"

    def test_env_batch_size(self, monkeypatch):
        monkeypatch.setenv("GL_TSGF_BATCH_SIZE", "2000")
        cfg = TimeSeriesGapFillerConfig.from_env()
        assert cfg.batch_size == 2000

    def test_env_max_records(self, monkeypatch):
        monkeypatch.setenv("GL_TSGF_MAX_RECORDS", "50000")
        cfg = TimeSeriesGapFillerConfig.from_env()
        assert cfg.max_records == 50_000

    def test_env_max_gap_ratio(self, monkeypatch):
        monkeypatch.setenv("GL_TSGF_MAX_GAP_RATIO", "0.8")
        cfg = TimeSeriesGapFillerConfig.from_env()
        assert cfg.max_gap_ratio == pytest.approx(0.8)

    def test_env_min_data_points(self, monkeypatch):
        monkeypatch.setenv("GL_TSGF_MIN_DATA_POINTS", "20")
        cfg = TimeSeriesGapFillerConfig.from_env()
        assert cfg.min_data_points == 20

    def test_env_default_strategy(self, monkeypatch):
        monkeypatch.setenv("GL_TSGF_DEFAULT_STRATEGY", "seasonal")
        cfg = TimeSeriesGapFillerConfig.from_env()
        assert cfg.default_strategy == "seasonal"

    def test_env_interpolation_method(self, monkeypatch):
        monkeypatch.setenv("GL_TSGF_INTERPOLATION_METHOD", "cubic")
        cfg = TimeSeriesGapFillerConfig.from_env()
        assert cfg.interpolation_method == "cubic"

    def test_env_seasonal_periods(self, monkeypatch):
        monkeypatch.setenv("GL_TSGF_SEASONAL_PERIODS", "52")
        cfg = TimeSeriesGapFillerConfig.from_env()
        assert cfg.seasonal_periods == 52

    def test_env_smoothing_alpha(self, monkeypatch):
        monkeypatch.setenv("GL_TSGF_SMOOTHING_ALPHA", "0.5")
        cfg = TimeSeriesGapFillerConfig.from_env()
        assert cfg.smoothing_alpha == pytest.approx(0.5)

    def test_env_smoothing_beta(self, monkeypatch):
        monkeypatch.setenv("GL_TSGF_SMOOTHING_BETA", "0.2")
        cfg = TimeSeriesGapFillerConfig.from_env()
        assert cfg.smoothing_beta == pytest.approx(0.2)

    def test_env_smoothing_gamma(self, monkeypatch):
        monkeypatch.setenv("GL_TSGF_SMOOTHING_GAMMA", "0.4")
        cfg = TimeSeriesGapFillerConfig.from_env()
        assert cfg.smoothing_gamma == pytest.approx(0.4)

    def test_env_correlation_threshold(self, monkeypatch):
        monkeypatch.setenv("GL_TSGF_CORRELATION_THRESHOLD", "0.9")
        cfg = TimeSeriesGapFillerConfig.from_env()
        assert cfg.correlation_threshold == pytest.approx(0.9)

    def test_env_confidence_threshold(self, monkeypatch):
        monkeypatch.setenv("GL_TSGF_CONFIDENCE_THRESHOLD", "0.85")
        cfg = TimeSeriesGapFillerConfig.from_env()
        assert cfg.confidence_threshold == pytest.approx(0.85)

    def test_env_short_gap_limit(self, monkeypatch):
        monkeypatch.setenv("GL_TSGF_SHORT_GAP_LIMIT", "5")
        cfg = TimeSeriesGapFillerConfig.from_env()
        assert cfg.short_gap_limit == 5

    def test_env_long_gap_limit(self, monkeypatch):
        monkeypatch.setenv("GL_TSGF_LONG_GAP_LIMIT", "24")
        cfg = TimeSeriesGapFillerConfig.from_env()
        assert cfg.long_gap_limit == 24

    def test_env_enable_seasonal_true(self, monkeypatch):
        monkeypatch.setenv("GL_TSGF_ENABLE_SEASONAL", "true")
        cfg = TimeSeriesGapFillerConfig.from_env()
        assert cfg.enable_seasonal is True

    def test_env_enable_seasonal_false(self, monkeypatch):
        monkeypatch.setenv("GL_TSGF_ENABLE_SEASONAL", "false")
        cfg = TimeSeriesGapFillerConfig.from_env()
        assert cfg.enable_seasonal is False

    def test_env_enable_cross_series_true(self, monkeypatch):
        monkeypatch.setenv("GL_TSGF_ENABLE_CROSS_SERIES", "1")
        cfg = TimeSeriesGapFillerConfig.from_env()
        assert cfg.enable_cross_series is True

    def test_env_enable_cross_series_false(self, monkeypatch):
        monkeypatch.setenv("GL_TSGF_ENABLE_CROSS_SERIES", "no")
        cfg = TimeSeriesGapFillerConfig.from_env()
        assert cfg.enable_cross_series is False

    def test_env_worker_count(self, monkeypatch):
        monkeypatch.setenv("GL_TSGF_WORKER_COUNT", "8")
        cfg = TimeSeriesGapFillerConfig.from_env()
        assert cfg.worker_count == 8

    def test_env_pool_min_size(self, monkeypatch):
        monkeypatch.setenv("GL_TSGF_POOL_MIN_SIZE", "5")
        cfg = TimeSeriesGapFillerConfig.from_env()
        assert cfg.pool_min_size == 5

    def test_env_pool_max_size(self, monkeypatch):
        monkeypatch.setenv("GL_TSGF_POOL_MAX_SIZE", "20")
        cfg = TimeSeriesGapFillerConfig.from_env()
        assert cfg.pool_max_size == 20

    def test_env_cache_ttl(self, monkeypatch):
        monkeypatch.setenv("GL_TSGF_CACHE_TTL", "7200")
        cfg = TimeSeriesGapFillerConfig.from_env()
        assert cfg.cache_ttl == 7200

    def test_env_rate_limit_rpm(self, monkeypatch):
        monkeypatch.setenv("GL_TSGF_RATE_LIMIT_RPM", "60")
        cfg = TimeSeriesGapFillerConfig.from_env()
        assert cfg.rate_limit_rpm == 60

    def test_env_enable_provenance_true(self, monkeypatch):
        monkeypatch.setenv("GL_TSGF_ENABLE_PROVENANCE", "yes")
        cfg = TimeSeriesGapFillerConfig.from_env()
        assert cfg.enable_provenance is True

    def test_env_enable_provenance_false(self, monkeypatch):
        monkeypatch.setenv("GL_TSGF_ENABLE_PROVENANCE", "0")
        cfg = TimeSeriesGapFillerConfig.from_env()
        assert cfg.enable_provenance is False


# ======================================================================
# 4. Boolean parsing edge cases
# ======================================================================


class TestBooleanParsing:
    """Test boolean environment variable parsing for true/false values."""

    @pytest.mark.parametrize("val", [
        "true", "True", "TRUE", "1", "yes", "Yes", "YES",
    ])
    def test_env_bool_true_values(self, monkeypatch, val):
        monkeypatch.setenv("GL_TSGF_ENABLE_SEASONAL", val)
        cfg = TimeSeriesGapFillerConfig.from_env()
        assert cfg.enable_seasonal is True

    @pytest.mark.parametrize("val", [
        "false", "False", "FALSE", "0", "no", "No", "NO", "anything_else",
    ])
    def test_env_bool_false_values(self, monkeypatch, val):
        monkeypatch.setenv("GL_TSGF_ENABLE_SEASONAL", val)
        cfg = TimeSeriesGapFillerConfig.from_env()
        assert cfg.enable_seasonal is False

    def test_env_bool_empty_string_is_false(self, monkeypatch):
        monkeypatch.setenv("GL_TSGF_ENABLE_PROVENANCE", "")
        cfg = TimeSeriesGapFillerConfig.from_env()
        assert cfg.enable_provenance is False


# ======================================================================
# 5. Singleton: get_config / set_config / reset_config
# ======================================================================


class TestSingleton:
    """Verify singleton accessor semantics."""

    def test_get_config_returns_instance(self):
        cfg = get_config()
        assert isinstance(cfg, TimeSeriesGapFillerConfig)

    def test_get_config_returns_same_instance(self):
        a = get_config()
        b = get_config()
        assert a is b

    def test_reset_config_clears_singleton(self):
        a = get_config()
        reset_config()
        b = get_config()
        assert a is not b

    def test_set_config_replaces_singleton(self):
        custom = TimeSeriesGapFillerConfig(batch_size=42)
        set_config(custom)
        assert get_config() is custom
        assert get_config().batch_size == 42

    def test_set_config_then_reset_returns_fresh(self):
        custom = TimeSeriesGapFillerConfig(batch_size=42)
        set_config(custom)
        reset_config()
        cfg = get_config()
        assert cfg is not custom
        assert cfg.batch_size == 1000  # default


# ======================================================================
# 6. Thread safety
# ======================================================================


class TestThreadSafety:
    """Verify get_config() is safe under concurrent access."""

    def test_concurrent_get_config_returns_same_instance(self):
        """Multiple threads calling get_config() should all get the same object."""
        reset_config()
        results = []
        barrier = threading.Barrier(8)

        def _worker():
            barrier.wait()
            results.append(id(get_config()))

        threads = [threading.Thread(target=_worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(set(results)) == 1, "All threads must see the same singleton"

    def test_concurrent_reset_and_get(self):
        """Reset + get from multiple threads should not raise."""
        errors = []

        def _worker():
            try:
                for _ in range(50):
                    reset_config()
                    cfg = get_config()
                    assert isinstance(cfg, TimeSeriesGapFillerConfig)
            except Exception as exc:
                errors.append(exc)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(_worker) for _ in range(4)]
            concurrent.futures.wait(futures)

        assert errors == [], f"Thread errors: {errors}"


# ======================================================================
# 7. Validation
# ======================================================================


class TestValidation:
    """Verify validate() enforces all constraints."""

    def test_valid_default_config(self, fresh_config):
        """Default config should pass validation without error."""
        fresh_config.validate()

    def test_batch_size_zero_raises(self):
        cfg = TimeSeriesGapFillerConfig(batch_size=0)
        with pytest.raises(ValueError, match="batch_size"):
            cfg.validate()

    def test_batch_size_negative_raises(self):
        cfg = TimeSeriesGapFillerConfig(batch_size=-1)
        with pytest.raises(ValueError, match="batch_size"):
            cfg.validate()

    def test_max_records_zero_raises(self):
        cfg = TimeSeriesGapFillerConfig(max_records=0)
        with pytest.raises(ValueError, match="max_records"):
            cfg.validate()

    def test_batch_size_exceeds_max_records_raises(self):
        cfg = TimeSeriesGapFillerConfig(batch_size=200, max_records=100)
        with pytest.raises(ValueError, match="batch_size must be <= max_records"):
            cfg.validate()

    def test_max_gap_ratio_above_one_raises(self):
        cfg = TimeSeriesGapFillerConfig(max_gap_ratio=1.5)
        with pytest.raises(ValueError, match="max_gap_ratio"):
            cfg.validate()

    def test_max_gap_ratio_negative_raises(self):
        cfg = TimeSeriesGapFillerConfig(max_gap_ratio=-0.1)
        with pytest.raises(ValueError, match="max_gap_ratio"):
            cfg.validate()

    def test_min_data_points_one_raises(self):
        cfg = TimeSeriesGapFillerConfig(min_data_points=1)
        with pytest.raises(ValueError, match="min_data_points"):
            cfg.validate()

    def test_invalid_default_strategy_raises(self):
        cfg = TimeSeriesGapFillerConfig(default_strategy="magic")
        with pytest.raises(ValueError, match="default_strategy"):
            cfg.validate()

    def test_invalid_interpolation_method_raises(self):
        cfg = TimeSeriesGapFillerConfig(interpolation_method="catmull_rom")
        with pytest.raises(ValueError, match="interpolation_method"):
            cfg.validate()

    def test_seasonal_periods_one_raises(self):
        cfg = TimeSeriesGapFillerConfig(seasonal_periods=1)
        with pytest.raises(ValueError, match="seasonal_periods"):
            cfg.validate()

    def test_smoothing_alpha_negative_raises(self):
        cfg = TimeSeriesGapFillerConfig(smoothing_alpha=-0.1)
        with pytest.raises(ValueError, match="smoothing_alpha"):
            cfg.validate()

    def test_smoothing_alpha_above_one_raises(self):
        cfg = TimeSeriesGapFillerConfig(smoothing_alpha=1.1)
        with pytest.raises(ValueError, match="smoothing_alpha"):
            cfg.validate()

    def test_smoothing_beta_negative_raises(self):
        cfg = TimeSeriesGapFillerConfig(smoothing_beta=-0.01)
        with pytest.raises(ValueError, match="smoothing_beta"):
            cfg.validate()

    def test_smoothing_beta_above_one_raises(self):
        cfg = TimeSeriesGapFillerConfig(smoothing_beta=1.5)
        with pytest.raises(ValueError, match="smoothing_beta"):
            cfg.validate()

    def test_smoothing_gamma_negative_raises(self):
        cfg = TimeSeriesGapFillerConfig(smoothing_gamma=-0.5)
        with pytest.raises(ValueError, match="smoothing_gamma"):
            cfg.validate()

    def test_smoothing_gamma_above_one_raises(self):
        cfg = TimeSeriesGapFillerConfig(smoothing_gamma=2.0)
        with pytest.raises(ValueError, match="smoothing_gamma"):
            cfg.validate()

    def test_correlation_threshold_negative_raises(self):
        cfg = TimeSeriesGapFillerConfig(correlation_threshold=-0.1)
        with pytest.raises(ValueError, match="correlation_threshold"):
            cfg.validate()

    def test_correlation_threshold_above_one_raises(self):
        cfg = TimeSeriesGapFillerConfig(correlation_threshold=1.01)
        with pytest.raises(ValueError, match="correlation_threshold"):
            cfg.validate()

    def test_confidence_threshold_negative_raises(self):
        cfg = TimeSeriesGapFillerConfig(confidence_threshold=-0.5)
        with pytest.raises(ValueError, match="confidence_threshold"):
            cfg.validate()

    def test_confidence_threshold_above_one_raises(self):
        cfg = TimeSeriesGapFillerConfig(confidence_threshold=1.5)
        with pytest.raises(ValueError, match="confidence_threshold"):
            cfg.validate()

    def test_worker_count_zero_raises(self):
        cfg = TimeSeriesGapFillerConfig(worker_count=0)
        with pytest.raises(ValueError, match="worker_count"):
            cfg.validate()

    def test_pool_min_size_zero_raises(self):
        cfg = TimeSeriesGapFillerConfig(pool_min_size=0)
        with pytest.raises(ValueError, match="pool_min_size"):
            cfg.validate()

    def test_pool_max_less_than_min_raises(self):
        cfg = TimeSeriesGapFillerConfig(pool_min_size=20, pool_max_size=5)
        with pytest.raises(ValueError, match="pool_max_size"):
            cfg.validate()

    def test_cache_ttl_negative_raises(self):
        cfg = TimeSeriesGapFillerConfig(cache_ttl=-1)
        with pytest.raises(ValueError, match="cache_ttl"):
            cfg.validate()

    def test_cache_ttl_zero_valid(self):
        """cache_ttl=0 means no caching, should be valid."""
        cfg = TimeSeriesGapFillerConfig(cache_ttl=0)
        cfg.validate()  # no error

    def test_rate_limit_rpm_zero_raises(self):
        cfg = TimeSeriesGapFillerConfig(rate_limit_rpm=0)
        with pytest.raises(ValueError, match="rate_limit_rpm"):
            cfg.validate()

    def test_invalid_log_level_raises(self):
        cfg = TimeSeriesGapFillerConfig(log_level="TRACE")
        with pytest.raises(ValueError, match="log_level"):
            cfg.validate()

    def test_multiple_validation_errors_joined(self):
        """Multiple failures should all appear in the error message."""
        cfg = TimeSeriesGapFillerConfig(
            batch_size=0,
            max_records=0,
            smoothing_alpha=-1.0,
        )
        with pytest.raises(ValueError) as exc_info:
            cfg.validate()
        msg = str(exc_info.value)
        assert "batch_size" in msg
        assert "max_records" in msg
        assert "smoothing_alpha" in msg

    def test_valid_strategies_all_pass(self):
        """All documented strategies should pass validation."""
        valid = (
            "auto", "linear", "cubic_spline", "pchip", "akima",
            "polynomial", "seasonal", "trend", "cross_series",
            "moving_average", "exponential_smoothing", "calendar_aware",
        )
        for strategy in valid:
            cfg = TimeSeriesGapFillerConfig(default_strategy=strategy)
            cfg.validate()

    def test_valid_interpolation_methods_all_pass(self):
        """All documented interpolation methods should pass validation."""
        valid = ("linear", "cubic", "spline", "pchip", "akima")
        for method in valid:
            cfg = TimeSeriesGapFillerConfig(interpolation_method=method)
            cfg.validate()

    def test_valid_log_levels_all_pass(self):
        """All standard log levels should pass validation."""
        for level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            cfg = TimeSeriesGapFillerConfig(log_level=level)
            cfg.validate()

    def test_boundary_max_gap_ratio_zero(self):
        cfg = TimeSeriesGapFillerConfig(max_gap_ratio=0.0)
        cfg.validate()

    def test_boundary_max_gap_ratio_one(self):
        cfg = TimeSeriesGapFillerConfig(max_gap_ratio=1.0)
        cfg.validate()

    def test_boundary_smoothing_alpha_zero(self):
        cfg = TimeSeriesGapFillerConfig(smoothing_alpha=0.0)
        cfg.validate()

    def test_boundary_smoothing_alpha_one(self):
        cfg = TimeSeriesGapFillerConfig(smoothing_alpha=1.0)
        cfg.validate()


# ======================================================================
# 8. Type coercion via from_env
# ======================================================================


class TestTypeCoercion:
    """Verify from_env coerces string env vars to correct types."""

    def test_str_to_int_batch_size(self, monkeypatch):
        monkeypatch.setenv("GL_TSGF_BATCH_SIZE", "256")
        cfg = TimeSeriesGapFillerConfig.from_env()
        assert isinstance(cfg.batch_size, int)
        assert cfg.batch_size == 256

    def test_str_to_float_max_gap_ratio(self, monkeypatch):
        monkeypatch.setenv("GL_TSGF_MAX_GAP_RATIO", "0.75")
        cfg = TimeSeriesGapFillerConfig.from_env()
        assert isinstance(cfg.max_gap_ratio, float)
        assert cfg.max_gap_ratio == pytest.approx(0.75)

    def test_str_to_int_rate_limit_rpm(self, monkeypatch):
        monkeypatch.setenv("GL_TSGF_RATE_LIMIT_RPM", "240")
        cfg = TimeSeriesGapFillerConfig.from_env()
        assert isinstance(cfg.rate_limit_rpm, int)
        assert cfg.rate_limit_rpm == 240

    def test_str_to_float_smoothing_gamma(self, monkeypatch):
        monkeypatch.setenv("GL_TSGF_SMOOTHING_GAMMA", "0.55")
        cfg = TimeSeriesGapFillerConfig.from_env()
        assert isinstance(cfg.smoothing_gamma, float)
        assert cfg.smoothing_gamma == pytest.approx(0.55)


# ======================================================================
# 9. Invalid environment values fallback
# ======================================================================


class TestInvalidEnvValues:
    """Invalid env vars should fall back to defaults with a warning."""

    def test_invalid_int_falls_back(self, monkeypatch):
        monkeypatch.setenv("GL_TSGF_BATCH_SIZE", "not_an_int")
        cfg = TimeSeriesGapFillerConfig.from_env()
        assert cfg.batch_size == 1000  # default

    def test_invalid_float_falls_back(self, monkeypatch):
        monkeypatch.setenv("GL_TSGF_MAX_GAP_RATIO", "abc")
        cfg = TimeSeriesGapFillerConfig.from_env()
        assert cfg.max_gap_ratio == pytest.approx(0.5)  # default

    def test_invalid_int_max_records_falls_back(self, monkeypatch):
        monkeypatch.setenv("GL_TSGF_MAX_RECORDS", "xyz")
        cfg = TimeSeriesGapFillerConfig.from_env()
        assert cfg.max_records == 100_000

    def test_empty_string_int_falls_back(self, monkeypatch):
        monkeypatch.setenv("GL_TSGF_WORKER_COUNT", "")
        cfg = TimeSeriesGapFillerConfig.from_env()
        # Empty string will fail int(), should fall back to default
        assert cfg.worker_count == 4

    def test_empty_string_float_falls_back(self, monkeypatch):
        monkeypatch.setenv("GL_TSGF_SMOOTHING_ALPHA", "")
        cfg = TimeSeriesGapFillerConfig.from_env()
        assert cfg.smoothing_alpha == pytest.approx(0.3)

    def test_empty_string_for_str_field_is_accepted(self, monkeypatch):
        """Empty string is a valid string, should replace default."""
        monkeypatch.setenv("GL_TSGF_DATABASE_URL", "")
        cfg = TimeSeriesGapFillerConfig.from_env()
        assert cfg.database_url == ""

    def test_whitespace_int_falls_back(self, monkeypatch):
        monkeypatch.setenv("GL_TSGF_CACHE_TTL", "  ")
        cfg = TimeSeriesGapFillerConfig.from_env()
        assert cfg.cache_ttl == 3600


# ======================================================================
# 10. Repr / str
# ======================================================================


class TestReprStr:
    """Verify dataclass repr and str output."""

    def test_repr_contains_class_name(self, fresh_config):
        r = repr(fresh_config)
        assert "TimeSeriesGapFillerConfig" in r

    def test_repr_contains_batch_size(self, fresh_config):
        r = repr(fresh_config)
        assert "batch_size=1000" in r

    def test_repr_contains_max_records(self, fresh_config):
        r = repr(fresh_config)
        assert "max_records=100000" in r

    def test_repr_contains_strategy(self, fresh_config):
        r = repr(fresh_config)
        assert "default_strategy='auto'" in r

    def test_repr_contains_smoothing_alpha(self, fresh_config):
        r = repr(fresh_config)
        assert "smoothing_alpha" in r

    def test_str_matches_repr(self, fresh_config):
        assert str(fresh_config) == repr(fresh_config)


# ======================================================================
# 11. Explicit construction with overrides
# ======================================================================


class TestExplicitConstruction:
    """Verify explicit keyword construction works correctly."""

    def test_override_single_field(self):
        cfg = TimeSeriesGapFillerConfig(batch_size=2048)
        assert cfg.batch_size == 2048
        assert cfg.max_records == 100_000  # other defaults intact

    def test_override_multiple_fields(self):
        cfg = TimeSeriesGapFillerConfig(
            database_url="pg://host/db",
            redis_url="redis://host",
            batch_size=512,
            enable_seasonal=False,
        )
        assert cfg.database_url == "pg://host/db"
        assert cfg.redis_url == "redis://host"
        assert cfg.batch_size == 512
        assert cfg.enable_seasonal is False

    def test_override_all_smoothing_params(self):
        cfg = TimeSeriesGapFillerConfig(
            smoothing_alpha=0.8,
            smoothing_beta=0.5,
            smoothing_gamma=0.9,
        )
        assert cfg.smoothing_alpha == pytest.approx(0.8)
        assert cfg.smoothing_beta == pytest.approx(0.5)
        assert cfg.smoothing_gamma == pytest.approx(0.9)

    def test_override_pool_sizes(self):
        cfg = TimeSeriesGapFillerConfig(pool_min_size=5, pool_max_size=50)
        assert cfg.pool_min_size == 5
        assert cfg.pool_max_size == 50

    def test_override_thresholds(self):
        cfg = TimeSeriesGapFillerConfig(
            correlation_threshold=0.95,
            confidence_threshold=0.85,
        )
        assert cfg.correlation_threshold == pytest.approx(0.95)
        assert cfg.confidence_threshold == pytest.approx(0.85)


# ======================================================================
# 12. Edge cases
# ======================================================================


class TestEdgeCases:
    """Boundary and edge-case scenarios."""

    def test_batch_size_equals_max_records_is_valid(self):
        cfg = TimeSeriesGapFillerConfig(batch_size=100, max_records=100)
        cfg.validate()

    def test_pool_min_equals_pool_max_is_valid(self):
        cfg = TimeSeriesGapFillerConfig(pool_min_size=5, pool_max_size=5)
        cfg.validate()

    def test_very_large_batch_size_below_max_records(self):
        cfg = TimeSeriesGapFillerConfig(batch_size=999_999, max_records=1_000_000)
        cfg.validate()

    def test_min_data_points_two_is_valid(self):
        cfg = TimeSeriesGapFillerConfig(min_data_points=2)
        cfg.validate()

    def test_seasonal_periods_two_is_valid(self):
        cfg = TimeSeriesGapFillerConfig(seasonal_periods=2)
        cfg.validate()

    def test_from_env_without_any_env_returns_defaults(self):
        """from_env() with no GL_TSGF_ vars set returns all defaults."""
        cfg = TimeSeriesGapFillerConfig.from_env()
        default = TimeSeriesGapFillerConfig()
        for field in dc_fields(default):
            assert getattr(cfg, field.name) == getattr(default, field.name), (
                f"Field {field.name} differs"
            )

    def test_dataclass_is_not_frozen(self, fresh_config):
        """Config fields should be mutable (dataclass is not frozen)."""
        fresh_config.batch_size = 9999
        assert fresh_config.batch_size == 9999

    def test_equality_of_identical_configs(self):
        a = TimeSeriesGapFillerConfig()
        b = TimeSeriesGapFillerConfig()
        assert a == b

    def test_inequality_when_field_differs(self):
        a = TimeSeriesGapFillerConfig()
        b = TimeSeriesGapFillerConfig(batch_size=42)
        assert a != b
