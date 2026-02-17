# -*- coding: utf-8 -*-
"""
Unit tests for MissingValueImputerConfig - AGENT-DATA-012

Tests all 31 configuration fields, GL_MVI_ environment variable overrides,
singleton pattern, thread safety, validation logic, and edge cases.
Target: 70+ tests.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-012 Missing Value Imputer (GL-DATA-X-015)
"""

from __future__ import annotations

import os
import threading
from unittest.mock import patch

import pytest

from greenlang.missing_value_imputer.config import (
    MissingValueImputerConfig,
    get_config,
    reset_config,
    set_config,
)


# =============================================================================
# Test default values for all 31 fields
# =============================================================================


class TestDefaultConfig:
    """Verify every default field value on a freshly constructed config."""

    def test_default_database_url(self):
        cfg = MissingValueImputerConfig()
        assert cfg.database_url == ""

    def test_default_redis_url(self):
        cfg = MissingValueImputerConfig()
        assert cfg.redis_url == ""

    def test_default_s3_bucket_url(self):
        cfg = MissingValueImputerConfig()
        assert cfg.s3_bucket_url == ""

    def test_default_log_level(self):
        cfg = MissingValueImputerConfig()
        assert cfg.log_level == "INFO"

    def test_default_batch_size(self):
        cfg = MissingValueImputerConfig()
        assert cfg.batch_size == 1000

    def test_default_max_records(self):
        cfg = MissingValueImputerConfig()
        assert cfg.max_records == 100_000

    def test_default_strategy(self):
        cfg = MissingValueImputerConfig()
        assert cfg.default_strategy == "auto"

    def test_default_confidence_threshold(self):
        cfg = MissingValueImputerConfig()
        assert cfg.confidence_threshold == 0.7

    def test_default_max_missing_pct(self):
        cfg = MissingValueImputerConfig()
        assert cfg.max_missing_pct == 0.8

    def test_default_enable_statistical(self):
        cfg = MissingValueImputerConfig()
        assert cfg.enable_statistical is True

    def test_default_knn_neighbors(self):
        cfg = MissingValueImputerConfig()
        assert cfg.knn_neighbors == 5

    def test_default_max_knn_dataset_size(self):
        cfg = MissingValueImputerConfig()
        assert cfg.max_knn_dataset_size == 50_000

    def test_default_mice_iterations(self):
        cfg = MissingValueImputerConfig()
        assert cfg.mice_iterations == 10

    def test_default_multiple_imputations(self):
        cfg = MissingValueImputerConfig()
        assert cfg.multiple_imputations == 5

    def test_default_enable_ml_imputation(self):
        cfg = MissingValueImputerConfig()
        assert cfg.enable_ml_imputation is True

    def test_default_enable_timeseries(self):
        cfg = MissingValueImputerConfig()
        assert cfg.enable_timeseries is True

    def test_default_interpolation_method(self):
        cfg = MissingValueImputerConfig()
        assert cfg.interpolation_method == "linear"

    def test_default_seasonal_period(self):
        cfg = MissingValueImputerConfig()
        assert cfg.seasonal_period == 12

    def test_default_trend_window(self):
        cfg = MissingValueImputerConfig()
        assert cfg.trend_window == 6

    def test_default_enable_rule_based(self):
        cfg = MissingValueImputerConfig()
        assert cfg.enable_rule_based is True

    def test_default_validation_split(self):
        cfg = MissingValueImputerConfig()
        assert cfg.validation_split == 0.2

    def test_default_worker_count(self):
        cfg = MissingValueImputerConfig()
        assert cfg.worker_count == 4

    def test_default_pool_min_size(self):
        cfg = MissingValueImputerConfig()
        assert cfg.pool_min_size == 2

    def test_default_pool_max_size(self):
        cfg = MissingValueImputerConfig()
        assert cfg.pool_max_size == 10

    def test_default_cache_ttl(self):
        cfg = MissingValueImputerConfig()
        assert cfg.cache_ttl == 3600

    def test_default_rate_limit_rpm(self):
        cfg = MissingValueImputerConfig()
        assert cfg.rate_limit_rpm == 120

    def test_default_rate_limit_burst(self):
        cfg = MissingValueImputerConfig()
        assert cfg.rate_limit_burst == 20

    def test_default_enable_provenance(self):
        cfg = MissingValueImputerConfig()
        assert cfg.enable_provenance is True

    def test_default_provenance_hash_algorithm(self):
        cfg = MissingValueImputerConfig()
        assert cfg.provenance_hash_algorithm == "sha256"

    def test_default_enable_metrics(self):
        cfg = MissingValueImputerConfig()
        assert cfg.enable_metrics is True

    def test_default_confidence_method(self):
        cfg = MissingValueImputerConfig()
        assert cfg.default_confidence_method == "ensemble"


# =============================================================================
# Test from_env with GL_MVI_ environment variables
# =============================================================================


class TestFromEnv:
    """Test from_env() parsing of GL_MVI_ prefixed environment variables."""

    def test_string_override(self, monkeypatch):
        monkeypatch.setenv("GL_MVI_DATABASE_URL", "postgresql://prod:x@db/prod")
        cfg = MissingValueImputerConfig.from_env()
        assert cfg.database_url == "postgresql://prod:x@db/prod"

    def test_redis_url_override(self, monkeypatch):
        monkeypatch.setenv("GL_MVI_REDIS_URL", "redis://cache:6380/1")
        cfg = MissingValueImputerConfig.from_env()
        assert cfg.redis_url == "redis://cache:6380/1"

    def test_int_override(self, monkeypatch):
        monkeypatch.setenv("GL_MVI_BATCH_SIZE", "5000")
        cfg = MissingValueImputerConfig.from_env()
        assert cfg.batch_size == 5000

    def test_float_override(self, monkeypatch):
        monkeypatch.setenv("GL_MVI_CONFIDENCE_THRESHOLD", "0.85")
        cfg = MissingValueImputerConfig.from_env()
        assert cfg.confidence_threshold == 0.85

    def test_bool_true_override(self, monkeypatch):
        monkeypatch.setenv("GL_MVI_ENABLE_ML_IMPUTATION", "true")
        cfg = MissingValueImputerConfig.from_env()
        assert cfg.enable_ml_imputation is True

    def test_bool_false_override(self, monkeypatch):
        monkeypatch.setenv("GL_MVI_ENABLE_ML_IMPUTATION", "false")
        cfg = MissingValueImputerConfig.from_env()
        assert cfg.enable_ml_imputation is False

    def test_bool_1_override(self, monkeypatch):
        monkeypatch.setenv("GL_MVI_ENABLE_PROVENANCE", "1")
        cfg = MissingValueImputerConfig.from_env()
        assert cfg.enable_provenance is True

    def test_bool_yes_override(self, monkeypatch):
        monkeypatch.setenv("GL_MVI_ENABLE_TIMESERIES", "yes")
        cfg = MissingValueImputerConfig.from_env()
        assert cfg.enable_timeseries is True

    def test_bool_no_override(self, monkeypatch):
        monkeypatch.setenv("GL_MVI_ENABLE_TIMESERIES", "no")
        cfg = MissingValueImputerConfig.from_env()
        assert cfg.enable_timeseries is False

    def test_invalid_int_falls_back(self, monkeypatch):
        monkeypatch.setenv("GL_MVI_BATCH_SIZE", "not_a_number")
        cfg = MissingValueImputerConfig.from_env()
        assert cfg.batch_size == 1000  # default

    def test_invalid_float_falls_back(self, monkeypatch):
        monkeypatch.setenv("GL_MVI_CONFIDENCE_THRESHOLD", "abc")
        cfg = MissingValueImputerConfig.from_env()
        assert cfg.confidence_threshold == 0.7  # default

    def test_max_records_override(self, monkeypatch):
        monkeypatch.setenv("GL_MVI_MAX_RECORDS", "500000")
        cfg = MissingValueImputerConfig.from_env()
        assert cfg.max_records == 500_000

    def test_knn_neighbors_override(self, monkeypatch):
        monkeypatch.setenv("GL_MVI_KNN_NEIGHBORS", "7")
        cfg = MissingValueImputerConfig.from_env()
        assert cfg.knn_neighbors == 7

    def test_mice_iterations_override(self, monkeypatch):
        monkeypatch.setenv("GL_MVI_MICE_ITERATIONS", "20")
        cfg = MissingValueImputerConfig.from_env()
        assert cfg.mice_iterations == 20

    def test_interpolation_method_override(self, monkeypatch):
        monkeypatch.setenv("GL_MVI_INTERPOLATION_METHOD", "cubic")
        cfg = MissingValueImputerConfig.from_env()
        assert cfg.interpolation_method == "cubic"

    def test_seasonal_period_override(self, monkeypatch):
        monkeypatch.setenv("GL_MVI_SEASONAL_PERIOD", "4")
        cfg = MissingValueImputerConfig.from_env()
        assert cfg.seasonal_period == 4

    def test_default_strategy_override(self, monkeypatch):
        monkeypatch.setenv("GL_MVI_DEFAULT_STRATEGY", "median")
        cfg = MissingValueImputerConfig.from_env()
        assert cfg.default_strategy == "median"

    def test_validation_split_override(self, monkeypatch):
        monkeypatch.setenv("GL_MVI_VALIDATION_SPLIT", "0.3")
        cfg = MissingValueImputerConfig.from_env()
        assert cfg.validation_split == 0.3

    def test_cache_ttl_override(self, monkeypatch):
        monkeypatch.setenv("GL_MVI_CACHE_TTL", "7200")
        cfg = MissingValueImputerConfig.from_env()
        assert cfg.cache_ttl == 7200

    def test_log_level_override(self, monkeypatch):
        monkeypatch.setenv("GL_MVI_LOG_LEVEL", "DEBUG")
        cfg = MissingValueImputerConfig.from_env()
        assert cfg.log_level == "DEBUG"

    def test_provenance_hash_algorithm_override(self, monkeypatch):
        monkeypatch.setenv("GL_MVI_PROVENANCE_HASH_ALGORITHM", "sha512")
        cfg = MissingValueImputerConfig.from_env()
        assert cfg.provenance_hash_algorithm == "sha512"

    def test_confidence_method_override(self, monkeypatch):
        monkeypatch.setenv("GL_MVI_DEFAULT_CONFIDENCE_METHOD", "bootstrap")
        cfg = MissingValueImputerConfig.from_env()
        assert cfg.default_confidence_method == "bootstrap"

    def test_multiple_imputations_override(self, monkeypatch):
        monkeypatch.setenv("GL_MVI_MULTIPLE_IMPUTATIONS", "10")
        cfg = MissingValueImputerConfig.from_env()
        assert cfg.multiple_imputations == 10


# =============================================================================
# Singleton pattern tests
# =============================================================================


class TestSingleton:
    """Test get_config, set_config, reset_config thread safety."""

    def test_get_config_returns_instance(self):
        cfg = get_config()
        assert isinstance(cfg, MissingValueImputerConfig)

    def test_get_config_returns_same_instance(self):
        cfg1 = get_config()
        cfg2 = get_config()
        assert cfg1 is cfg2

    def test_set_config_replaces_instance(self):
        custom = MissingValueImputerConfig(batch_size=9999)
        set_config(custom)
        assert get_config().batch_size == 9999

    def test_reset_config_clears_instance(self):
        set_config(MissingValueImputerConfig(batch_size=7777))
        reset_config()
        cfg = get_config()
        # After reset, from_env returns defaults (since no env vars set)
        assert cfg.batch_size == 1000

    def test_thread_safety_get_config(self):
        """Multiple threads calling get_config should get the same instance."""
        results = []

        def worker():
            results.append(id(get_config()))

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(set(results)) == 1

    def test_thread_safety_set_reset(self):
        """Concurrent set and reset should not crash."""
        errors = []

        def setter():
            try:
                for _ in range(20):
                    set_config(MissingValueImputerConfig(batch_size=42))
            except Exception as e:
                errors.append(e)

        def resetter():
            try:
                for _ in range(20):
                    reset_config()
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=setter)
        t2 = threading.Thread(target=resetter)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(errors) == 0


# =============================================================================
# Validation and edge cases
# =============================================================================


class TestValidation:
    """Test boundary conditions and special values."""

    def test_zero_batch_size(self):
        cfg = MissingValueImputerConfig(batch_size=0)
        assert cfg.batch_size == 0

    def test_negative_batch_size_allowed(self):
        """Dataclass does not enforce constraints; just stores values."""
        cfg = MissingValueImputerConfig(batch_size=-1)
        assert cfg.batch_size == -1

    def test_confidence_threshold_zero(self):
        cfg = MissingValueImputerConfig(confidence_threshold=0.0)
        assert cfg.confidence_threshold == 0.0

    def test_confidence_threshold_one(self):
        cfg = MissingValueImputerConfig(confidence_threshold=1.0)
        assert cfg.confidence_threshold == 1.0

    def test_max_missing_pct_zero(self):
        cfg = MissingValueImputerConfig(max_missing_pct=0.0)
        assert cfg.max_missing_pct == 0.0

    def test_max_missing_pct_one(self):
        cfg = MissingValueImputerConfig(max_missing_pct=1.0)
        assert cfg.max_missing_pct == 1.0

    def test_empty_database_url(self):
        cfg = MissingValueImputerConfig(database_url="")
        assert cfg.database_url == ""

    def test_very_large_max_records(self):
        cfg = MissingValueImputerConfig(max_records=10_000_000)
        assert cfg.max_records == 10_000_000

    def test_worker_count_one(self):
        cfg = MissingValueImputerConfig(worker_count=1)
        assert cfg.worker_count == 1

    def test_all_toggles_disabled(self):
        cfg = MissingValueImputerConfig(
            enable_statistical=False,
            enable_ml_imputation=False,
            enable_timeseries=False,
            enable_rule_based=False,
            enable_provenance=False,
            enable_metrics=False,
        )
        assert cfg.enable_statistical is False
        assert cfg.enable_ml_imputation is False
        assert cfg.enable_timeseries is False
        assert cfg.enable_rule_based is False
        assert cfg.enable_provenance is False
        assert cfg.enable_metrics is False

    def test_custom_strategy_name(self):
        cfg = MissingValueImputerConfig(default_strategy="custom_strategy")
        assert cfg.default_strategy == "custom_strategy"
