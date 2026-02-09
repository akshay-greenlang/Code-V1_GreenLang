# -*- coding: utf-8 -*-
"""
Unit Tests for DataQualityProfilerConfig - AGENT-DATA-010

Tests the DataQualityProfilerConfig dataclass, all default values,
environment variable overrides (GL_DQ_ prefix), type coercion fallback,
thread-safe singleton management, and equality/repr behaviour.

Target: 100+ tests, 85%+ coverage of greenlang.data_quality_profiler.config

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import dataclasses
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from greenlang.data_quality_profiler.config import (
    DataQualityProfilerConfig,
    get_config,
    reset_config,
    set_config,
)


# ============================================================================
# TestDefaults - verify every default value for all ~35 config fields
# ============================================================================


class TestDefaults:
    """Every field of DataQualityProfilerConfig must have the correct default."""

    def test_default_database_url(self):
        cfg = DataQualityProfilerConfig()
        assert cfg.database_url == ""

    def test_default_redis_url(self):
        cfg = DataQualityProfilerConfig()
        assert cfg.redis_url == ""

    def test_default_s3_bucket_url(self):
        cfg = DataQualityProfilerConfig()
        assert cfg.s3_bucket_url == ""

    def test_default_max_rows_per_profile(self):
        cfg = DataQualityProfilerConfig()
        assert cfg.max_rows_per_profile == 1_000_000

    def test_default_max_columns_per_profile(self):
        cfg = DataQualityProfilerConfig()
        assert cfg.max_columns_per_profile == 500

    def test_default_sample_size_for_stats(self):
        cfg = DataQualityProfilerConfig()
        assert cfg.sample_size_for_stats == 10_000

    def test_default_enable_schema_inference(self):
        cfg = DataQualityProfilerConfig()
        assert cfg.enable_schema_inference is True

    def test_default_enable_cardinality_analysis(self):
        cfg = DataQualityProfilerConfig()
        assert cfg.enable_cardinality_analysis is True

    def test_default_max_unique_values_tracked(self):
        cfg = DataQualityProfilerConfig()
        assert cfg.max_unique_values_tracked == 1000

    def test_default_completeness_weight(self):
        cfg = DataQualityProfilerConfig()
        assert cfg.completeness_weight == pytest.approx(0.20)

    def test_default_validity_weight(self):
        cfg = DataQualityProfilerConfig()
        assert cfg.validity_weight == pytest.approx(0.20)

    def test_default_consistency_weight(self):
        cfg = DataQualityProfilerConfig()
        assert cfg.consistency_weight == pytest.approx(0.20)

    def test_default_timeliness_weight(self):
        cfg = DataQualityProfilerConfig()
        assert cfg.timeliness_weight == pytest.approx(0.15)

    def test_default_uniqueness_weight(self):
        cfg = DataQualityProfilerConfig()
        assert cfg.uniqueness_weight == pytest.approx(0.15)

    def test_default_accuracy_weight(self):
        cfg = DataQualityProfilerConfig()
        assert cfg.accuracy_weight == pytest.approx(0.10)

    def test_default_weights_sum_to_one(self):
        cfg = DataQualityProfilerConfig()
        total = (
            cfg.completeness_weight
            + cfg.validity_weight
            + cfg.consistency_weight
            + cfg.timeliness_weight
            + cfg.uniqueness_weight
            + cfg.accuracy_weight
        )
        assert total == pytest.approx(1.0)

    def test_default_freshness_excellent_hours(self):
        cfg = DataQualityProfilerConfig()
        assert cfg.freshness_excellent_hours == 24

    def test_default_freshness_good_hours(self):
        cfg = DataQualityProfilerConfig()
        assert cfg.freshness_good_hours == 72

    def test_default_freshness_fair_hours(self):
        cfg = DataQualityProfilerConfig()
        assert cfg.freshness_fair_hours == 168

    def test_default_freshness_poor_hours(self):
        cfg = DataQualityProfilerConfig()
        assert cfg.freshness_poor_hours == 720

    def test_default_default_sla_hours(self):
        cfg = DataQualityProfilerConfig()
        assert cfg.default_sla_hours == 48

    def test_default_default_outlier_method(self):
        cfg = DataQualityProfilerConfig()
        assert cfg.default_outlier_method == "iqr"

    def test_default_iqr_multiplier(self):
        cfg = DataQualityProfilerConfig()
        assert cfg.iqr_multiplier == pytest.approx(1.5)

    def test_default_zscore_threshold(self):
        cfg = DataQualityProfilerConfig()
        assert cfg.zscore_threshold == pytest.approx(3.0)

    def test_default_mad_threshold(self):
        cfg = DataQualityProfilerConfig()
        assert cfg.mad_threshold == pytest.approx(3.5)

    def test_default_min_samples_for_anomaly(self):
        cfg = DataQualityProfilerConfig()
        assert cfg.min_samples_for_anomaly == 10

    def test_default_max_rules_per_dataset(self):
        cfg = DataQualityProfilerConfig()
        assert cfg.max_rules_per_dataset == 100

    def test_default_max_gate_conditions(self):
        cfg = DataQualityProfilerConfig()
        assert cfg.max_gate_conditions == 20

    def test_default_default_gate_threshold(self):
        cfg = DataQualityProfilerConfig()
        assert cfg.default_gate_threshold == pytest.approx(0.70)

    def test_default_batch_max_datasets(self):
        cfg = DataQualityProfilerConfig()
        assert cfg.batch_max_datasets == 50

    def test_default_processing_timeout_seconds(self):
        cfg = DataQualityProfilerConfig()
        assert cfg.processing_timeout_seconds == 300

    def test_default_cache_ttl_seconds(self):
        cfg = DataQualityProfilerConfig()
        assert cfg.cache_ttl_seconds == 3600

    def test_default_pool_min_size(self):
        cfg = DataQualityProfilerConfig()
        assert cfg.pool_min_size == 2

    def test_default_pool_max_size(self):
        cfg = DataQualityProfilerConfig()
        assert cfg.pool_max_size == 10

    def test_default_log_level(self):
        cfg = DataQualityProfilerConfig()
        assert cfg.log_level == "INFO"


# ============================================================================
# TestEnvOverrides - verify every GL_DQ_* env var override
# ============================================================================


class TestEnvOverrides:
    """Every field can be overridden via GL_DQ_<FIELD_UPPER> env var."""

    def test_env_database_url(self, monkeypatch):
        monkeypatch.setenv("GL_DQ_DATABASE_URL", "postgresql://test:5432/dq")
        cfg = DataQualityProfilerConfig.from_env()
        assert cfg.database_url == "postgresql://test:5432/dq"

    def test_env_redis_url(self, monkeypatch):
        monkeypatch.setenv("GL_DQ_REDIS_URL", "redis://test:6379/0")
        cfg = DataQualityProfilerConfig.from_env()
        assert cfg.redis_url == "redis://test:6379/0"

    def test_env_s3_bucket_url(self, monkeypatch):
        monkeypatch.setenv("GL_DQ_S3_BUCKET_URL", "s3://my-bucket")
        cfg = DataQualityProfilerConfig.from_env()
        assert cfg.s3_bucket_url == "s3://my-bucket"

    def test_env_max_rows_per_profile(self, monkeypatch):
        monkeypatch.setenv("GL_DQ_MAX_ROWS_PER_PROFILE", "500000")
        cfg = DataQualityProfilerConfig.from_env()
        assert cfg.max_rows_per_profile == 500000

    def test_env_max_columns_per_profile(self, monkeypatch):
        monkeypatch.setenv("GL_DQ_MAX_COLUMNS_PER_PROFILE", "250")
        cfg = DataQualityProfilerConfig.from_env()
        assert cfg.max_columns_per_profile == 250

    def test_env_sample_size_for_stats(self, monkeypatch):
        monkeypatch.setenv("GL_DQ_SAMPLE_SIZE_FOR_STATS", "5000")
        cfg = DataQualityProfilerConfig.from_env()
        assert cfg.sample_size_for_stats == 5000

    def test_env_enable_schema_inference_true(self, monkeypatch):
        monkeypatch.setenv("GL_DQ_ENABLE_SCHEMA_INFERENCE", "true")
        cfg = DataQualityProfilerConfig.from_env()
        assert cfg.enable_schema_inference is True

    def test_env_enable_schema_inference_false(self, monkeypatch):
        monkeypatch.setenv("GL_DQ_ENABLE_SCHEMA_INFERENCE", "false")
        cfg = DataQualityProfilerConfig.from_env()
        assert cfg.enable_schema_inference is False

    def test_env_enable_schema_inference_one(self, monkeypatch):
        monkeypatch.setenv("GL_DQ_ENABLE_SCHEMA_INFERENCE", "1")
        cfg = DataQualityProfilerConfig.from_env()
        assert cfg.enable_schema_inference is True

    def test_env_enable_schema_inference_yes(self, monkeypatch):
        monkeypatch.setenv("GL_DQ_ENABLE_SCHEMA_INFERENCE", "yes")
        cfg = DataQualityProfilerConfig.from_env()
        assert cfg.enable_schema_inference is True

    def test_env_enable_cardinality_analysis_false(self, monkeypatch):
        monkeypatch.setenv("GL_DQ_ENABLE_CARDINALITY_ANALYSIS", "false")
        cfg = DataQualityProfilerConfig.from_env()
        assert cfg.enable_cardinality_analysis is False

    def test_env_max_unique_values_tracked(self, monkeypatch):
        monkeypatch.setenv("GL_DQ_MAX_UNIQUE_VALUES_TRACKED", "2000")
        cfg = DataQualityProfilerConfig.from_env()
        assert cfg.max_unique_values_tracked == 2000

    def test_env_completeness_weight(self, monkeypatch):
        monkeypatch.setenv("GL_DQ_COMPLETENESS_WEIGHT", "0.30")
        cfg = DataQualityProfilerConfig.from_env()
        assert cfg.completeness_weight == pytest.approx(0.30)

    def test_env_validity_weight(self, monkeypatch):
        monkeypatch.setenv("GL_DQ_VALIDITY_WEIGHT", "0.25")
        cfg = DataQualityProfilerConfig.from_env()
        assert cfg.validity_weight == pytest.approx(0.25)

    def test_env_consistency_weight(self, monkeypatch):
        monkeypatch.setenv("GL_DQ_CONSISTENCY_WEIGHT", "0.15")
        cfg = DataQualityProfilerConfig.from_env()
        assert cfg.consistency_weight == pytest.approx(0.15)

    def test_env_timeliness_weight(self, monkeypatch):
        monkeypatch.setenv("GL_DQ_TIMELINESS_WEIGHT", "0.10")
        cfg = DataQualityProfilerConfig.from_env()
        assert cfg.timeliness_weight == pytest.approx(0.10)

    def test_env_uniqueness_weight(self, monkeypatch):
        monkeypatch.setenv("GL_DQ_UNIQUENESS_WEIGHT", "0.12")
        cfg = DataQualityProfilerConfig.from_env()
        assert cfg.uniqueness_weight == pytest.approx(0.12)

    def test_env_accuracy_weight(self, monkeypatch):
        monkeypatch.setenv("GL_DQ_ACCURACY_WEIGHT", "0.08")
        cfg = DataQualityProfilerConfig.from_env()
        assert cfg.accuracy_weight == pytest.approx(0.08)

    def test_env_freshness_excellent_hours(self, monkeypatch):
        monkeypatch.setenv("GL_DQ_FRESHNESS_EXCELLENT_HOURS", "12")
        cfg = DataQualityProfilerConfig.from_env()
        assert cfg.freshness_excellent_hours == 12

    def test_env_freshness_good_hours(self, monkeypatch):
        monkeypatch.setenv("GL_DQ_FRESHNESS_GOOD_HOURS", "48")
        cfg = DataQualityProfilerConfig.from_env()
        assert cfg.freshness_good_hours == 48

    def test_env_freshness_fair_hours(self, monkeypatch):
        monkeypatch.setenv("GL_DQ_FRESHNESS_FAIR_HOURS", "100")
        cfg = DataQualityProfilerConfig.from_env()
        assert cfg.freshness_fair_hours == 100

    def test_env_freshness_poor_hours(self, monkeypatch):
        monkeypatch.setenv("GL_DQ_FRESHNESS_POOR_HOURS", "500")
        cfg = DataQualityProfilerConfig.from_env()
        assert cfg.freshness_poor_hours == 500

    def test_env_default_sla_hours(self, monkeypatch):
        monkeypatch.setenv("GL_DQ_DEFAULT_SLA_HOURS", "24")
        cfg = DataQualityProfilerConfig.from_env()
        assert cfg.default_sla_hours == 24

    def test_env_default_outlier_method(self, monkeypatch):
        monkeypatch.setenv("GL_DQ_DEFAULT_OUTLIER_METHOD", "zscore")
        cfg = DataQualityProfilerConfig.from_env()
        assert cfg.default_outlier_method == "zscore"

    def test_env_iqr_multiplier(self, monkeypatch):
        monkeypatch.setenv("GL_DQ_IQR_MULTIPLIER", "2.0")
        cfg = DataQualityProfilerConfig.from_env()
        assert cfg.iqr_multiplier == pytest.approx(2.0)

    def test_env_zscore_threshold(self, monkeypatch):
        monkeypatch.setenv("GL_DQ_ZSCORE_THRESHOLD", "2.5")
        cfg = DataQualityProfilerConfig.from_env()
        assert cfg.zscore_threshold == pytest.approx(2.5)

    def test_env_mad_threshold(self, monkeypatch):
        monkeypatch.setenv("GL_DQ_MAD_THRESHOLD", "4.0")
        cfg = DataQualityProfilerConfig.from_env()
        assert cfg.mad_threshold == pytest.approx(4.0)

    def test_env_min_samples_for_anomaly(self, monkeypatch):
        monkeypatch.setenv("GL_DQ_MIN_SAMPLES_FOR_ANOMALY", "20")
        cfg = DataQualityProfilerConfig.from_env()
        assert cfg.min_samples_for_anomaly == 20

    def test_env_max_rules_per_dataset(self, monkeypatch):
        monkeypatch.setenv("GL_DQ_MAX_RULES_PER_DATASET", "200")
        cfg = DataQualityProfilerConfig.from_env()
        assert cfg.max_rules_per_dataset == 200

    def test_env_max_gate_conditions(self, monkeypatch):
        monkeypatch.setenv("GL_DQ_MAX_GATE_CONDITIONS", "50")
        cfg = DataQualityProfilerConfig.from_env()
        assert cfg.max_gate_conditions == 50

    def test_env_default_gate_threshold(self, monkeypatch):
        monkeypatch.setenv("GL_DQ_DEFAULT_GATE_THRESHOLD", "0.85")
        cfg = DataQualityProfilerConfig.from_env()
        assert cfg.default_gate_threshold == pytest.approx(0.85)

    def test_env_batch_max_datasets(self, monkeypatch):
        monkeypatch.setenv("GL_DQ_BATCH_MAX_DATASETS", "100")
        cfg = DataQualityProfilerConfig.from_env()
        assert cfg.batch_max_datasets == 100

    def test_env_processing_timeout_seconds(self, monkeypatch):
        monkeypatch.setenv("GL_DQ_PROCESSING_TIMEOUT_SECONDS", "600")
        cfg = DataQualityProfilerConfig.from_env()
        assert cfg.processing_timeout_seconds == 600

    def test_env_cache_ttl_seconds(self, monkeypatch):
        monkeypatch.setenv("GL_DQ_CACHE_TTL_SECONDS", "7200")
        cfg = DataQualityProfilerConfig.from_env()
        assert cfg.cache_ttl_seconds == 7200

    def test_env_pool_min_size(self, monkeypatch):
        monkeypatch.setenv("GL_DQ_POOL_MIN_SIZE", "5")
        cfg = DataQualityProfilerConfig.from_env()
        assert cfg.pool_min_size == 5

    def test_env_pool_max_size(self, monkeypatch):
        monkeypatch.setenv("GL_DQ_POOL_MAX_SIZE", "20")
        cfg = DataQualityProfilerConfig.from_env()
        assert cfg.pool_max_size == 20

    def test_env_log_level(self, monkeypatch):
        monkeypatch.setenv("GL_DQ_LOG_LEVEL", "DEBUG")
        cfg = DataQualityProfilerConfig.from_env()
        assert cfg.log_level == "DEBUG"


# ============================================================================
# TestTypeCoercion - invalid int/float fallback, empty strings
# ============================================================================


class TestTypeCoercion:
    """Invalid environment values must fall back to defaults."""

    def test_invalid_int_falls_back(self, monkeypatch):
        monkeypatch.setenv("GL_DQ_MAX_ROWS_PER_PROFILE", "not_a_number")
        cfg = DataQualityProfilerConfig.from_env()
        assert cfg.max_rows_per_profile == 1_000_000

    def test_invalid_float_falls_back(self, monkeypatch):
        monkeypatch.setenv("GL_DQ_IQR_MULTIPLIER", "abc")
        cfg = DataQualityProfilerConfig.from_env()
        assert cfg.iqr_multiplier == pytest.approx(1.5)

    def test_empty_string_for_int_falls_back(self, monkeypatch):
        monkeypatch.setenv("GL_DQ_BATCH_MAX_DATASETS", "")
        cfg = DataQualityProfilerConfig.from_env()
        # Empty string cannot be parsed as int, falls back to default
        assert cfg.batch_max_datasets == 50

    def test_empty_string_for_float_falls_back(self, monkeypatch):
        monkeypatch.setenv("GL_DQ_ZSCORE_THRESHOLD", "")
        cfg = DataQualityProfilerConfig.from_env()
        assert cfg.zscore_threshold == pytest.approx(3.0)

    def test_float_as_int_truncated(self, monkeypatch):
        monkeypatch.setenv("GL_DQ_MAX_COLUMNS_PER_PROFILE", "3.14")
        cfg = DataQualityProfilerConfig.from_env()
        # "3.14" cannot be parsed by int(), so falls back
        assert cfg.max_columns_per_profile == 500

    def test_bool_no_case(self, monkeypatch):
        monkeypatch.setenv("GL_DQ_ENABLE_SCHEMA_INFERENCE", "TRUE")
        cfg = DataQualityProfilerConfig.from_env()
        assert cfg.enable_schema_inference is True

    def test_bool_zero_is_false(self, monkeypatch):
        monkeypatch.setenv("GL_DQ_ENABLE_CARDINALITY_ANALYSIS", "0")
        cfg = DataQualityProfilerConfig.from_env()
        assert cfg.enable_cardinality_analysis is False

    def test_bool_random_string_is_false(self, monkeypatch):
        monkeypatch.setenv("GL_DQ_ENABLE_SCHEMA_INFERENCE", "maybe")
        cfg = DataQualityProfilerConfig.from_env()
        assert cfg.enable_schema_inference is False

    def test_negative_int_accepted(self, monkeypatch):
        monkeypatch.setenv("GL_DQ_POOL_MIN_SIZE", "-1")
        cfg = DataQualityProfilerConfig.from_env()
        assert cfg.pool_min_size == -1

    def test_negative_float_accepted(self, monkeypatch):
        monkeypatch.setenv("GL_DQ_MAD_THRESHOLD", "-2.5")
        cfg = DataQualityProfilerConfig.from_env()
        assert cfg.mad_threshold == pytest.approx(-2.5)


# ============================================================================
# TestSingleton - get_config, set_config, reset_config
# ============================================================================


class TestSingleton:
    """Thread-safe singleton accessor functions."""

    def test_get_config_returns_instance(self):
        cfg = get_config()
        assert isinstance(cfg, DataQualityProfilerConfig)

    def test_get_config_returns_same_instance(self):
        cfg1 = get_config()
        cfg2 = get_config()
        assert cfg1 is cfg2

    def test_set_config_replaces_instance(self):
        custom = DataQualityProfilerConfig(log_level="DEBUG")
        set_config(custom)
        assert get_config() is custom
        assert get_config().log_level == "DEBUG"

    def test_reset_config_clears_instance(self):
        _ = get_config()
        reset_config()
        # After reset, next get_config creates a new instance
        cfg = get_config()
        assert isinstance(cfg, DataQualityProfilerConfig)

    def test_reset_then_set(self):
        reset_config()
        custom = DataQualityProfilerConfig(pool_max_size=99)
        set_config(custom)
        assert get_config().pool_max_size == 99

    def test_set_config_overrides_previous(self):
        c1 = DataQualityProfilerConfig(pool_min_size=5)
        c2 = DataQualityProfilerConfig(pool_min_size=8)
        set_config(c1)
        assert get_config().pool_min_size == 5
        set_config(c2)
        assert get_config().pool_min_size == 8


# ============================================================================
# TestThreadSafety - concurrent access to singleton
# ============================================================================


class TestThreadSafety:
    """Config singleton is safe under concurrent access."""

    def test_concurrent_get_config(self):
        """20 threads call get_config simultaneously; all must get same instance."""
        reset_config()
        results = []

        def _get():
            return id(get_config())

        with ThreadPoolExecutor(max_workers=20) as pool:
            futures = [pool.submit(_get) for _ in range(20)]
            for f in as_completed(futures):
                results.append(f.result())

        assert len(set(results)) == 1, "All threads must get the same singleton"

    def test_concurrent_set_and_get(self):
        """Concurrent set_config + get_config must not crash."""
        errors = []

        def _set(idx):
            try:
                c = DataQualityProfilerConfig(pool_min_size=idx)
                set_config(c)
                _ = get_config()
            except Exception as exc:
                errors.append(exc)

        with ThreadPoolExecutor(max_workers=20) as pool:
            futures = [pool.submit(_set, i) for i in range(20)]
            for f in as_completed(futures):
                f.result()

        assert len(errors) == 0, f"Concurrent set/get raised errors: {errors}"


# ============================================================================
# TestMultipleOverrides - combined env overrides
# ============================================================================


class TestMultipleOverrides:
    """Multiple env vars overridden simultaneously."""

    def test_all_connections_overridden(self, monkeypatch):
        monkeypatch.setenv("GL_DQ_DATABASE_URL", "pg://a")
        monkeypatch.setenv("GL_DQ_REDIS_URL", "redis://b")
        monkeypatch.setenv("GL_DQ_S3_BUCKET_URL", "s3://c")
        cfg = DataQualityProfilerConfig.from_env()
        assert cfg.database_url == "pg://a"
        assert cfg.redis_url == "redis://b"
        assert cfg.s3_bucket_url == "s3://c"

    def test_mixed_int_and_float_overrides(self, monkeypatch):
        monkeypatch.setenv("GL_DQ_MAX_ROWS_PER_PROFILE", "200000")
        monkeypatch.setenv("GL_DQ_IQR_MULTIPLIER", "2.5")
        monkeypatch.setenv("GL_DQ_ZSCORE_THRESHOLD", "4.0")
        cfg = DataQualityProfilerConfig.from_env()
        assert cfg.max_rows_per_profile == 200000
        assert cfg.iqr_multiplier == pytest.approx(2.5)
        assert cfg.zscore_threshold == pytest.approx(4.0)

    def test_bool_and_str_overrides(self, monkeypatch):
        monkeypatch.setenv("GL_DQ_ENABLE_SCHEMA_INFERENCE", "false")
        monkeypatch.setenv("GL_DQ_LOG_LEVEL", "WARNING")
        cfg = DataQualityProfilerConfig.from_env()
        assert cfg.enable_schema_inference is False
        assert cfg.log_level == "WARNING"

    def test_pool_sizes_overridden(self, monkeypatch):
        monkeypatch.setenv("GL_DQ_POOL_MIN_SIZE", "4")
        monkeypatch.setenv("GL_DQ_POOL_MAX_SIZE", "16")
        cfg = DataQualityProfilerConfig.from_env()
        assert cfg.pool_min_size == 4
        assert cfg.pool_max_size == 16

    def test_all_freshness_overridden(self, monkeypatch):
        monkeypatch.setenv("GL_DQ_FRESHNESS_EXCELLENT_HOURS", "6")
        monkeypatch.setenv("GL_DQ_FRESHNESS_GOOD_HOURS", "24")
        monkeypatch.setenv("GL_DQ_FRESHNESS_FAIR_HOURS", "96")
        monkeypatch.setenv("GL_DQ_FRESHNESS_POOR_HOURS", "360")
        monkeypatch.setenv("GL_DQ_DEFAULT_SLA_HOURS", "12")
        cfg = DataQualityProfilerConfig.from_env()
        assert cfg.freshness_excellent_hours == 6
        assert cfg.freshness_good_hours == 24
        assert cfg.freshness_fair_hours == 96
        assert cfg.freshness_poor_hours == 360
        assert cfg.default_sla_hours == 12


# ============================================================================
# TestDataclass - equality, repr, field count
# ============================================================================


class TestDataclass:
    """DataQualityProfilerConfig is a well-formed Python dataclass."""

    def test_is_dataclass(self):
        assert dataclasses.is_dataclass(DataQualityProfilerConfig)

    def test_field_count(self):
        fields = dataclasses.fields(DataQualityProfilerConfig)
        assert len(fields) == 34

    def test_equality_same_defaults(self):
        a = DataQualityProfilerConfig()
        b = DataQualityProfilerConfig()
        assert a == b

    def test_equality_different_values(self):
        a = DataQualityProfilerConfig(pool_min_size=1)
        b = DataQualityProfilerConfig(pool_min_size=2)
        assert a != b

    def test_repr_contains_class_name(self):
        cfg = DataQualityProfilerConfig()
        assert "DataQualityProfilerConfig" in repr(cfg)

    def test_repr_contains_field_values(self):
        cfg = DataQualityProfilerConfig(log_level="TRACE")
        assert "TRACE" in repr(cfg)

    def test_field_names_match_expected(self):
        expected_names = {
            "database_url", "redis_url", "s3_bucket_url",
            "max_rows_per_profile", "max_columns_per_profile",
            "sample_size_for_stats", "enable_schema_inference",
            "enable_cardinality_analysis", "max_unique_values_tracked",
            "completeness_weight", "validity_weight", "consistency_weight",
            "timeliness_weight", "uniqueness_weight", "accuracy_weight",
            "freshness_excellent_hours", "freshness_good_hours",
            "freshness_fair_hours", "freshness_poor_hours",
            "default_sla_hours", "default_outlier_method",
            "iqr_multiplier", "zscore_threshold", "mad_threshold",
            "min_samples_for_anomaly", "max_rules_per_dataset",
            "max_gate_conditions", "default_gate_threshold",
            "batch_max_datasets", "processing_timeout_seconds",
            "cache_ttl_seconds", "pool_min_size", "pool_max_size",
            "log_level",
        }
        actual_names = {f.name for f in dataclasses.fields(DataQualityProfilerConfig)}
        assert expected_names == actual_names

    def test_from_env_returns_config(self):
        cfg = DataQualityProfilerConfig.from_env()
        assert isinstance(cfg, DataQualityProfilerConfig)


# ============================================================================
# TestModuleExports - __all__ completeness
# ============================================================================


class TestModuleExports:
    """Verify config module exports."""

    def test_all_list_exists(self):
        from greenlang.data_quality_profiler import config as mod
        assert hasattr(mod, "__all__")

    def test_all_contains_config_class(self):
        from greenlang.data_quality_profiler import config as mod
        assert "DataQualityProfilerConfig" in mod.__all__

    def test_all_contains_get_config(self):
        from greenlang.data_quality_profiler import config as mod
        assert "get_config" in mod.__all__

    def test_all_contains_set_config(self):
        from greenlang.data_quality_profiler import config as mod
        assert "set_config" in mod.__all__

    def test_all_contains_reset_config(self):
        from greenlang.data_quality_profiler import config as mod
        assert "reset_config" in mod.__all__

    def test_all_has_four_entries(self):
        from greenlang.data_quality_profiler import config as mod
        assert len(mod.__all__) == 4
