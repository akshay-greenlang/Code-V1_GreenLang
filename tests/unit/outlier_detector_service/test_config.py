# -*- coding: utf-8 -*-
"""
Unit tests for OutlierDetectorConfig - AGENT-DATA-013

Tests all 28 configuration fields, GL_OD_ environment variable overrides,
singleton pattern, thread safety, validation logic, and edge cases.
Target: 70+ tests.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-013 Outlier Detection (GL-DATA-X-016)
"""

from __future__ import annotations

import os
import threading
from unittest.mock import patch

import pytest

from greenlang.outlier_detector.config import (
    OutlierDetectorConfig,
    get_config,
    reset_config,
    set_config,
)


# =============================================================================
# Test default values for all 28 fields
# =============================================================================


class TestDefaultConfig:
    """Verify every default field value on a freshly constructed config."""

    def test_default_database_url(self):
        cfg = OutlierDetectorConfig()
        assert cfg.database_url == ""

    def test_default_redis_url(self):
        cfg = OutlierDetectorConfig()
        assert cfg.redis_url == ""

    def test_default_s3_bucket_url(self):
        cfg = OutlierDetectorConfig()
        assert cfg.s3_bucket_url == ""

    def test_default_log_level(self):
        cfg = OutlierDetectorConfig()
        assert cfg.log_level == "INFO"

    def test_default_batch_size(self):
        cfg = OutlierDetectorConfig()
        assert cfg.batch_size == 1000

    def test_default_max_records(self):
        cfg = OutlierDetectorConfig()
        assert cfg.max_records == 100_000

    def test_default_iqr_multiplier(self):
        cfg = OutlierDetectorConfig()
        assert cfg.iqr_multiplier == 1.5

    def test_default_zscore_threshold(self):
        cfg = OutlierDetectorConfig()
        assert cfg.zscore_threshold == 3.0

    def test_default_mad_threshold(self):
        cfg = OutlierDetectorConfig()
        assert cfg.mad_threshold == 3.5

    def test_default_grubbs_alpha(self):
        cfg = OutlierDetectorConfig()
        assert cfg.grubbs_alpha == 0.05

    def test_default_lof_neighbors(self):
        cfg = OutlierDetectorConfig()
        assert cfg.lof_neighbors == 20

    def test_default_isolation_trees(self):
        cfg = OutlierDetectorConfig()
        assert cfg.isolation_trees == 100

    def test_default_ensemble_method(self):
        cfg = OutlierDetectorConfig()
        assert cfg.ensemble_method == "weighted_average"

    def test_default_min_consensus(self):
        cfg = OutlierDetectorConfig()
        assert cfg.min_consensus == 2

    def test_default_enable_contextual(self):
        cfg = OutlierDetectorConfig()
        assert cfg.enable_contextual is True

    def test_default_enable_temporal(self):
        cfg = OutlierDetectorConfig()
        assert cfg.enable_temporal is True

    def test_default_enable_multivariate(self):
        cfg = OutlierDetectorConfig()
        assert cfg.enable_multivariate is True

    def test_default_treatment(self):
        cfg = OutlierDetectorConfig()
        assert cfg.default_treatment == "flag"

    def test_default_winsorize_pct(self):
        cfg = OutlierDetectorConfig()
        assert cfg.winsorize_pct == 0.05

    def test_default_worker_count(self):
        cfg = OutlierDetectorConfig()
        assert cfg.worker_count == 4

    def test_default_pool_min_size(self):
        cfg = OutlierDetectorConfig()
        assert cfg.pool_min_size == 2

    def test_default_pool_max_size(self):
        cfg = OutlierDetectorConfig()
        assert cfg.pool_max_size == 10

    def test_default_cache_ttl(self):
        cfg = OutlierDetectorConfig()
        assert cfg.cache_ttl == 3600

    def test_default_rate_limit_rpm(self):
        cfg = OutlierDetectorConfig()
        assert cfg.rate_limit_rpm == 120

    def test_default_rate_limit_burst(self):
        cfg = OutlierDetectorConfig()
        assert cfg.rate_limit_burst == 20

    def test_default_enable_provenance(self):
        cfg = OutlierDetectorConfig()
        assert cfg.enable_provenance is True

    def test_default_provenance_hash_algorithm(self):
        cfg = OutlierDetectorConfig()
        assert cfg.provenance_hash_algorithm == "sha256"

    def test_default_enable_metrics(self):
        cfg = OutlierDetectorConfig()
        assert cfg.enable_metrics is True


# =============================================================================
# Test from_env with GL_OD_ environment variables
# =============================================================================


class TestFromEnv:
    """Test from_env() parsing of GL_OD_ prefixed environment variables."""

    def test_string_override(self, monkeypatch):
        monkeypatch.setenv("GL_OD_DATABASE_URL", "postgresql://prod:x@db/prod")
        cfg = OutlierDetectorConfig.from_env()
        assert cfg.database_url == "postgresql://prod:x@db/prod"

    def test_redis_url_override(self, monkeypatch):
        monkeypatch.setenv("GL_OD_REDIS_URL", "redis://cache:6380/1")
        cfg = OutlierDetectorConfig.from_env()
        assert cfg.redis_url == "redis://cache:6380/1"

    def test_int_override(self, monkeypatch):
        monkeypatch.setenv("GL_OD_BATCH_SIZE", "5000")
        cfg = OutlierDetectorConfig.from_env()
        assert cfg.batch_size == 5000

    def test_float_override(self, monkeypatch):
        monkeypatch.setenv("GL_OD_IQR_MULTIPLIER", "3.0")
        cfg = OutlierDetectorConfig.from_env()
        assert cfg.iqr_multiplier == 3.0

    def test_bool_true_override(self, monkeypatch):
        monkeypatch.setenv("GL_OD_ENABLE_CONTEXTUAL", "true")
        cfg = OutlierDetectorConfig.from_env()
        assert cfg.enable_contextual is True

    def test_bool_false_override(self, monkeypatch):
        monkeypatch.setenv("GL_OD_ENABLE_CONTEXTUAL", "false")
        cfg = OutlierDetectorConfig.from_env()
        assert cfg.enable_contextual is False

    def test_bool_1_override(self, monkeypatch):
        monkeypatch.setenv("GL_OD_ENABLE_PROVENANCE", "1")
        cfg = OutlierDetectorConfig.from_env()
        assert cfg.enable_provenance is True

    def test_bool_yes_override(self, monkeypatch):
        monkeypatch.setenv("GL_OD_ENABLE_TEMPORAL", "yes")
        cfg = OutlierDetectorConfig.from_env()
        assert cfg.enable_temporal is True

    def test_bool_no_override(self, monkeypatch):
        monkeypatch.setenv("GL_OD_ENABLE_TEMPORAL", "no")
        cfg = OutlierDetectorConfig.from_env()
        assert cfg.enable_temporal is False

    def test_invalid_int_falls_back(self, monkeypatch):
        monkeypatch.setenv("GL_OD_BATCH_SIZE", "not_a_number")
        cfg = OutlierDetectorConfig.from_env()
        assert cfg.batch_size == 1000  # default

    def test_invalid_float_falls_back(self, monkeypatch):
        monkeypatch.setenv("GL_OD_IQR_MULTIPLIER", "abc")
        cfg = OutlierDetectorConfig.from_env()
        assert cfg.iqr_multiplier == 1.5  # default

    def test_max_records_override(self, monkeypatch):
        monkeypatch.setenv("GL_OD_MAX_RECORDS", "500000")
        cfg = OutlierDetectorConfig.from_env()
        assert cfg.max_records == 500_000

    def test_zscore_threshold_override(self, monkeypatch):
        monkeypatch.setenv("GL_OD_ZSCORE_THRESHOLD", "2.5")
        cfg = OutlierDetectorConfig.from_env()
        assert cfg.zscore_threshold == 2.5

    def test_mad_threshold_override(self, monkeypatch):
        monkeypatch.setenv("GL_OD_MAD_THRESHOLD", "4.0")
        cfg = OutlierDetectorConfig.from_env()
        assert cfg.mad_threshold == 4.0

    def test_grubbs_alpha_override(self, monkeypatch):
        monkeypatch.setenv("GL_OD_GRUBBS_ALPHA", "0.01")
        cfg = OutlierDetectorConfig.from_env()
        assert cfg.grubbs_alpha == 0.01

    def test_lof_neighbors_override(self, monkeypatch):
        monkeypatch.setenv("GL_OD_LOF_NEIGHBORS", "30")
        cfg = OutlierDetectorConfig.from_env()
        assert cfg.lof_neighbors == 30

    def test_isolation_trees_override(self, monkeypatch):
        monkeypatch.setenv("GL_OD_ISOLATION_TREES", "200")
        cfg = OutlierDetectorConfig.from_env()
        assert cfg.isolation_trees == 200

    def test_ensemble_method_override(self, monkeypatch):
        monkeypatch.setenv("GL_OD_ENSEMBLE_METHOD", "majority_vote")
        cfg = OutlierDetectorConfig.from_env()
        assert cfg.ensemble_method == "majority_vote"

    def test_min_consensus_override(self, monkeypatch):
        monkeypatch.setenv("GL_OD_MIN_CONSENSUS", "3")
        cfg = OutlierDetectorConfig.from_env()
        assert cfg.min_consensus == 3

    def test_default_treatment_override(self, monkeypatch):
        monkeypatch.setenv("GL_OD_DEFAULT_TREATMENT", "cap")
        cfg = OutlierDetectorConfig.from_env()
        assert cfg.default_treatment == "cap"

    def test_winsorize_pct_override(self, monkeypatch):
        monkeypatch.setenv("GL_OD_WINSORIZE_PCT", "0.10")
        cfg = OutlierDetectorConfig.from_env()
        assert cfg.winsorize_pct == 0.10

    def test_cache_ttl_override(self, monkeypatch):
        monkeypatch.setenv("GL_OD_CACHE_TTL", "7200")
        cfg = OutlierDetectorConfig.from_env()
        assert cfg.cache_ttl == 7200

    def test_log_level_override(self, monkeypatch):
        monkeypatch.setenv("GL_OD_LOG_LEVEL", "DEBUG")
        cfg = OutlierDetectorConfig.from_env()
        assert cfg.log_level == "DEBUG"

    def test_provenance_hash_algorithm_override(self, monkeypatch):
        monkeypatch.setenv("GL_OD_PROVENANCE_HASH_ALGORITHM", "sha512")
        cfg = OutlierDetectorConfig.from_env()
        assert cfg.provenance_hash_algorithm == "sha512"

    def test_enable_multivariate_override(self, monkeypatch):
        monkeypatch.setenv("GL_OD_ENABLE_MULTIVARIATE", "false")
        cfg = OutlierDetectorConfig.from_env()
        assert cfg.enable_multivariate is False

    def test_s3_bucket_override(self, monkeypatch):
        monkeypatch.setenv("GL_OD_S3_BUCKET_URL", "s3://prod-bucket")
        cfg = OutlierDetectorConfig.from_env()
        assert cfg.s3_bucket_url == "s3://prod-bucket"


# =============================================================================
# Singleton pattern tests
# =============================================================================


class TestSingleton:
    """Test get_config, set_config, reset_config thread safety."""

    def test_get_config_returns_instance(self):
        cfg = get_config()
        assert isinstance(cfg, OutlierDetectorConfig)

    def test_get_config_returns_same_instance(self):
        cfg1 = get_config()
        cfg2 = get_config()
        assert cfg1 is cfg2

    def test_set_config_replaces_instance(self):
        custom = OutlierDetectorConfig(batch_size=9999)
        set_config(custom)
        assert get_config().batch_size == 9999

    def test_reset_config_clears_instance(self):
        set_config(OutlierDetectorConfig(batch_size=7777))
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
                    set_config(OutlierDetectorConfig(batch_size=42))
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
        cfg = OutlierDetectorConfig(batch_size=0)
        assert cfg.batch_size == 0

    def test_negative_batch_size_allowed(self):
        """Dataclass does not enforce constraints; just stores values."""
        cfg = OutlierDetectorConfig(batch_size=-1)
        assert cfg.batch_size == -1

    def test_iqr_multiplier_zero(self):
        cfg = OutlierDetectorConfig(iqr_multiplier=0.0)
        assert cfg.iqr_multiplier == 0.0

    def test_iqr_multiplier_large(self):
        cfg = OutlierDetectorConfig(iqr_multiplier=10.0)
        assert cfg.iqr_multiplier == 10.0

    def test_zscore_threshold_zero(self):
        cfg = OutlierDetectorConfig(zscore_threshold=0.0)
        assert cfg.zscore_threshold == 0.0

    def test_winsorize_pct_zero(self):
        cfg = OutlierDetectorConfig(winsorize_pct=0.0)
        assert cfg.winsorize_pct == 0.0

    def test_winsorize_pct_half(self):
        cfg = OutlierDetectorConfig(winsorize_pct=0.5)
        assert cfg.winsorize_pct == 0.5

    def test_empty_database_url(self):
        cfg = OutlierDetectorConfig(database_url="")
        assert cfg.database_url == ""

    def test_very_large_max_records(self):
        cfg = OutlierDetectorConfig(max_records=10_000_000)
        assert cfg.max_records == 10_000_000

    def test_worker_count_one(self):
        cfg = OutlierDetectorConfig(worker_count=1)
        assert cfg.worker_count == 1

    def test_all_toggles_disabled(self):
        cfg = OutlierDetectorConfig(
            enable_contextual=False,
            enable_temporal=False,
            enable_multivariate=False,
            enable_provenance=False,
            enable_metrics=False,
        )
        assert cfg.enable_contextual is False
        assert cfg.enable_temporal is False
        assert cfg.enable_multivariate is False
        assert cfg.enable_provenance is False
        assert cfg.enable_metrics is False

    def test_custom_treatment_name(self):
        cfg = OutlierDetectorConfig(default_treatment="custom_strategy")
        assert cfg.default_treatment == "custom_strategy"

    def test_min_consensus_one(self):
        cfg = OutlierDetectorConfig(min_consensus=1)
        assert cfg.min_consensus == 1

    def test_grubbs_alpha_boundary(self):
        cfg = OutlierDetectorConfig(grubbs_alpha=1.0)
        assert cfg.grubbs_alpha == 1.0
