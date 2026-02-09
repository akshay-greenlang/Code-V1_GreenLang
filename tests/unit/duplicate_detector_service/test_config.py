# -*- coding: utf-8 -*-
"""
Unit tests for DuplicateDetectorConfig - AGENT-DATA-011

Tests all 35 configuration fields, GL_DD_ environment variable overrides,
singleton pattern, thread safety, validation logic, and edge cases.
Target: 100+ tests.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-011 Duplicate Detection Agent (GL-DATA-X-014)
"""

from __future__ import annotations

import os
import threading
from unittest.mock import patch

import pytest

from greenlang.duplicate_detector.config import (
    DuplicateDetectorConfig,
    get_config,
    reset_config,
    set_config,
)


# =============================================================================
# Test default values for all 35 fields
# =============================================================================


class TestConfigDefaults:
    """Verify every default field value on a freshly constructed config."""

    def test_default_database_url(self):
        cfg = DuplicateDetectorConfig()
        assert cfg.database_url == ""

    def test_default_redis_url(self):
        cfg = DuplicateDetectorConfig()
        assert cfg.redis_url == ""

    def test_default_s3_bucket(self):
        cfg = DuplicateDetectorConfig()
        assert cfg.s3_bucket == ""

    def test_default_max_records_per_job(self):
        cfg = DuplicateDetectorConfig()
        assert cfg.max_records_per_job == 1_000_000

    def test_default_batch_size(self):
        cfg = DuplicateDetectorConfig()
        assert cfg.default_batch_size == 10_000

    def test_default_fingerprint_algorithm(self):
        cfg = DuplicateDetectorConfig()
        assert cfg.fingerprint_algorithm == "sha256"

    def test_default_fingerprint_normalize(self):
        cfg = DuplicateDetectorConfig()
        assert cfg.fingerprint_normalize is True

    def test_default_blocking_strategy(self):
        cfg = DuplicateDetectorConfig()
        assert cfg.blocking_strategy == "sorted_neighborhood"

    def test_default_blocking_window_size(self):
        cfg = DuplicateDetectorConfig()
        assert cfg.blocking_window_size == 10

    def test_default_blocking_key_size(self):
        cfg = DuplicateDetectorConfig()
        assert cfg.blocking_key_size == 3

    def test_default_canopy_tight_threshold(self):
        cfg = DuplicateDetectorConfig()
        assert cfg.canopy_tight_threshold == 0.8

    def test_default_canopy_loose_threshold(self):
        cfg = DuplicateDetectorConfig()
        assert cfg.canopy_loose_threshold == 0.4

    def test_default_similarity_algorithm(self):
        cfg = DuplicateDetectorConfig()
        assert cfg.default_similarity_algorithm == "jaro_winkler"

    def test_default_ngram_size(self):
        cfg = DuplicateDetectorConfig()
        assert cfg.ngram_size == 3

    def test_default_match_threshold(self):
        cfg = DuplicateDetectorConfig()
        assert cfg.match_threshold == 0.85

    def test_default_possible_threshold(self):
        cfg = DuplicateDetectorConfig()
        assert cfg.possible_threshold == 0.65

    def test_default_non_match_threshold(self):
        cfg = DuplicateDetectorConfig()
        assert cfg.non_match_threshold == 0.40

    def test_default_use_fellegi_sunter(self):
        cfg = DuplicateDetectorConfig()
        assert cfg.use_fellegi_sunter is False

    def test_default_cluster_algorithm(self):
        cfg = DuplicateDetectorConfig()
        assert cfg.cluster_algorithm == "union_find"

    def test_default_cluster_min_quality(self):
        cfg = DuplicateDetectorConfig()
        assert cfg.cluster_min_quality == 0.5

    def test_default_merge_strategy(self):
        cfg = DuplicateDetectorConfig()
        assert cfg.default_merge_strategy == "keep_most_complete"

    def test_default_merge_conflict_resolution(self):
        cfg = DuplicateDetectorConfig()
        assert cfg.merge_conflict_resolution == "most_complete"

    def test_default_pipeline_checkpoint_interval(self):
        cfg = DuplicateDetectorConfig()
        assert cfg.pipeline_checkpoint_interval == 1000

    def test_default_pipeline_timeout_seconds(self):
        cfg = DuplicateDetectorConfig()
        assert cfg.pipeline_timeout_seconds == 3600

    def test_default_max_comparisons_per_block(self):
        cfg = DuplicateDetectorConfig()
        assert cfg.max_comparisons_per_block == 50_000

    def test_default_cache_ttl_seconds(self):
        cfg = DuplicateDetectorConfig()
        assert cfg.cache_ttl_seconds == 3600

    def test_default_cache_enabled(self):
        cfg = DuplicateDetectorConfig()
        assert cfg.cache_enabled is True

    def test_default_pool_min_size(self):
        cfg = DuplicateDetectorConfig()
        assert cfg.pool_min_size == 2

    def test_default_pool_max_size(self):
        cfg = DuplicateDetectorConfig()
        assert cfg.pool_max_size == 10

    def test_default_log_level(self):
        cfg = DuplicateDetectorConfig()
        assert cfg.log_level == "INFO"

    def test_default_enable_metrics(self):
        cfg = DuplicateDetectorConfig()
        assert cfg.enable_metrics is True

    def test_default_max_field_weights(self):
        cfg = DuplicateDetectorConfig()
        assert cfg.max_field_weights == 50

    def test_default_max_rules_per_job(self):
        cfg = DuplicateDetectorConfig()
        assert cfg.max_rules_per_job == 100

    def test_default_comparison_sample_rate(self):
        cfg = DuplicateDetectorConfig()
        assert cfg.comparison_sample_rate == 1.0

    def test_total_field_count(self):
        """Config dataclass must expose exactly 34 fields."""
        cfg = DuplicateDetectorConfig()
        fields = list(cfg.__dataclass_fields__.keys())
        assert len(fields) == 34


# =============================================================================
# Test GL_DD_ env var overrides (from_env)
# =============================================================================


class TestConfigEnvOverrides:
    """Verify that from_env reads GL_DD_ environment variables correctly."""

    # -- String overrides --

    def test_env_database_url(self, monkeypatch):
        monkeypatch.setenv("GL_DD_DATABASE_URL", "postgresql://prod:prod@db/prod")
        cfg = DuplicateDetectorConfig.from_env()
        assert cfg.database_url == "postgresql://prod:prod@db/prod"

    def test_env_redis_url(self, monkeypatch):
        monkeypatch.setenv("GL_DD_REDIS_URL", "redis://prod:6380/1")
        cfg = DuplicateDetectorConfig.from_env()
        assert cfg.redis_url == "redis://prod:6380/1"

    def test_env_s3_bucket(self, monkeypatch):
        monkeypatch.setenv("GL_DD_S3_BUCKET", "prod-dedup-bucket")
        cfg = DuplicateDetectorConfig.from_env()
        assert cfg.s3_bucket == "prod-dedup-bucket"

    def test_env_fingerprint_algorithm(self, monkeypatch):
        monkeypatch.setenv("GL_DD_FINGERPRINT_ALGORITHM", "minhash")
        cfg = DuplicateDetectorConfig.from_env()
        assert cfg.fingerprint_algorithm == "minhash"

    def test_env_blocking_strategy(self, monkeypatch):
        monkeypatch.setenv("GL_DD_BLOCKING_STRATEGY", "canopy")
        cfg = DuplicateDetectorConfig.from_env()
        assert cfg.blocking_strategy == "canopy"

    def test_env_default_similarity_algorithm(self, monkeypatch):
        monkeypatch.setenv("GL_DD_DEFAULT_SIMILARITY_ALGORITHM", "levenshtein")
        cfg = DuplicateDetectorConfig.from_env()
        assert cfg.default_similarity_algorithm == "levenshtein"

    def test_env_cluster_algorithm(self, monkeypatch):
        monkeypatch.setenv("GL_DD_CLUSTER_ALGORITHM", "connected_components")
        cfg = DuplicateDetectorConfig.from_env()
        assert cfg.cluster_algorithm == "connected_components"

    def test_env_default_merge_strategy(self, monkeypatch):
        monkeypatch.setenv("GL_DD_DEFAULT_MERGE_STRATEGY", "golden_record")
        cfg = DuplicateDetectorConfig.from_env()
        assert cfg.default_merge_strategy == "golden_record"

    def test_env_merge_conflict_resolution(self, monkeypatch):
        monkeypatch.setenv("GL_DD_MERGE_CONFLICT_RESOLUTION", "longest")
        cfg = DuplicateDetectorConfig.from_env()
        assert cfg.merge_conflict_resolution == "longest"

    def test_env_log_level(self, monkeypatch):
        monkeypatch.setenv("GL_DD_LOG_LEVEL", "DEBUG")
        cfg = DuplicateDetectorConfig.from_env()
        assert cfg.log_level == "DEBUG"

    # -- Integer overrides --

    def test_env_max_records_per_job(self, monkeypatch):
        monkeypatch.setenv("GL_DD_MAX_RECORDS_PER_JOB", "500000")
        cfg = DuplicateDetectorConfig.from_env()
        assert cfg.max_records_per_job == 500_000

    def test_env_default_batch_size(self, monkeypatch):
        monkeypatch.setenv("GL_DD_DEFAULT_BATCH_SIZE", "2000")
        cfg = DuplicateDetectorConfig.from_env()
        assert cfg.default_batch_size == 2000

    def test_env_blocking_window_size(self, monkeypatch):
        monkeypatch.setenv("GL_DD_BLOCKING_WINDOW_SIZE", "20")
        cfg = DuplicateDetectorConfig.from_env()
        assert cfg.blocking_window_size == 20

    def test_env_blocking_key_size(self, monkeypatch):
        monkeypatch.setenv("GL_DD_BLOCKING_KEY_SIZE", "5")
        cfg = DuplicateDetectorConfig.from_env()
        assert cfg.blocking_key_size == 5

    def test_env_ngram_size(self, monkeypatch):
        monkeypatch.setenv("GL_DD_NGRAM_SIZE", "4")
        cfg = DuplicateDetectorConfig.from_env()
        assert cfg.ngram_size == 4

    def test_env_pipeline_checkpoint_interval(self, monkeypatch):
        monkeypatch.setenv("GL_DD_PIPELINE_CHECKPOINT_INTERVAL", "5000")
        cfg = DuplicateDetectorConfig.from_env()
        assert cfg.pipeline_checkpoint_interval == 5000

    def test_env_pipeline_timeout_seconds(self, monkeypatch):
        monkeypatch.setenv("GL_DD_PIPELINE_TIMEOUT_SECONDS", "7200")
        cfg = DuplicateDetectorConfig.from_env()
        assert cfg.pipeline_timeout_seconds == 7200

    def test_env_max_comparisons_per_block(self, monkeypatch):
        monkeypatch.setenv("GL_DD_MAX_COMPARISONS_PER_BLOCK", "100000")
        cfg = DuplicateDetectorConfig.from_env()
        assert cfg.max_comparisons_per_block == 100_000

    def test_env_cache_ttl_seconds(self, monkeypatch):
        monkeypatch.setenv("GL_DD_CACHE_TTL_SECONDS", "7200")
        cfg = DuplicateDetectorConfig.from_env()
        assert cfg.cache_ttl_seconds == 7200

    def test_env_pool_min_size(self, monkeypatch):
        monkeypatch.setenv("GL_DD_POOL_MIN_SIZE", "5")
        cfg = DuplicateDetectorConfig.from_env()
        assert cfg.pool_min_size == 5

    def test_env_pool_max_size(self, monkeypatch):
        monkeypatch.setenv("GL_DD_POOL_MAX_SIZE", "25")
        cfg = DuplicateDetectorConfig.from_env()
        assert cfg.pool_max_size == 25

    def test_env_max_field_weights(self, monkeypatch):
        monkeypatch.setenv("GL_DD_MAX_FIELD_WEIGHTS", "100")
        cfg = DuplicateDetectorConfig.from_env()
        assert cfg.max_field_weights == 100

    def test_env_max_rules_per_job(self, monkeypatch):
        monkeypatch.setenv("GL_DD_MAX_RULES_PER_JOB", "200")
        cfg = DuplicateDetectorConfig.from_env()
        assert cfg.max_rules_per_job == 200

    # -- Float overrides --

    def test_env_canopy_tight_threshold(self, monkeypatch):
        monkeypatch.setenv("GL_DD_CANOPY_TIGHT_THRESHOLD", "0.9")
        cfg = DuplicateDetectorConfig.from_env()
        assert cfg.canopy_tight_threshold == pytest.approx(0.9)

    def test_env_canopy_loose_threshold(self, monkeypatch):
        monkeypatch.setenv("GL_DD_CANOPY_LOOSE_THRESHOLD", "0.3")
        cfg = DuplicateDetectorConfig.from_env()
        assert cfg.canopy_loose_threshold == pytest.approx(0.3)

    def test_env_match_threshold(self, monkeypatch):
        monkeypatch.setenv("GL_DD_MATCH_THRESHOLD", "0.90")
        cfg = DuplicateDetectorConfig.from_env()
        assert cfg.match_threshold == pytest.approx(0.90)

    def test_env_possible_threshold(self, monkeypatch):
        monkeypatch.setenv("GL_DD_POSSIBLE_THRESHOLD", "0.70")
        cfg = DuplicateDetectorConfig.from_env()
        assert cfg.possible_threshold == pytest.approx(0.70)

    def test_env_non_match_threshold(self, monkeypatch):
        monkeypatch.setenv("GL_DD_NON_MATCH_THRESHOLD", "0.35")
        cfg = DuplicateDetectorConfig.from_env()
        assert cfg.non_match_threshold == pytest.approx(0.35)

    def test_env_cluster_min_quality(self, monkeypatch):
        monkeypatch.setenv("GL_DD_CLUSTER_MIN_QUALITY", "0.6")
        cfg = DuplicateDetectorConfig.from_env()
        assert cfg.cluster_min_quality == pytest.approx(0.6)

    def test_env_comparison_sample_rate(self, monkeypatch):
        monkeypatch.setenv("GL_DD_COMPARISON_SAMPLE_RATE", "0.5")
        cfg = DuplicateDetectorConfig.from_env()
        assert cfg.comparison_sample_rate == pytest.approx(0.5)

    # -- Boolean overrides --

    def test_env_fingerprint_normalize_false(self, monkeypatch):
        monkeypatch.setenv("GL_DD_FINGERPRINT_NORMALIZE", "false")
        cfg = DuplicateDetectorConfig.from_env()
        assert cfg.fingerprint_normalize is False

    def test_env_fingerprint_normalize_true_numeric(self, monkeypatch):
        monkeypatch.setenv("GL_DD_FINGERPRINT_NORMALIZE", "1")
        cfg = DuplicateDetectorConfig.from_env()
        assert cfg.fingerprint_normalize is True

    def test_env_fingerprint_normalize_yes(self, monkeypatch):
        monkeypatch.setenv("GL_DD_FINGERPRINT_NORMALIZE", "yes")
        cfg = DuplicateDetectorConfig.from_env()
        assert cfg.fingerprint_normalize is True

    def test_env_fingerprint_normalize_YES_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("GL_DD_FINGERPRINT_NORMALIZE", "YES")
        cfg = DuplicateDetectorConfig.from_env()
        assert cfg.fingerprint_normalize is True

    def test_env_use_fellegi_sunter_true(self, monkeypatch):
        monkeypatch.setenv("GL_DD_USE_FELLEGI_SUNTER", "true")
        cfg = DuplicateDetectorConfig.from_env()
        assert cfg.use_fellegi_sunter is True

    def test_env_use_fellegi_sunter_false(self, monkeypatch):
        monkeypatch.setenv("GL_DD_USE_FELLEGI_SUNTER", "0")
        cfg = DuplicateDetectorConfig.from_env()
        assert cfg.use_fellegi_sunter is False

    def test_env_cache_enabled_false(self, monkeypatch):
        monkeypatch.setenv("GL_DD_CACHE_ENABLED", "false")
        cfg = DuplicateDetectorConfig.from_env()
        assert cfg.cache_enabled is False

    def test_env_enable_metrics_false(self, monkeypatch):
        monkeypatch.setenv("GL_DD_ENABLE_METRICS", "false")
        cfg = DuplicateDetectorConfig.from_env()
        assert cfg.enable_metrics is False

    def test_env_enable_metrics_true_upper(self, monkeypatch):
        monkeypatch.setenv("GL_DD_ENABLE_METRICS", "TRUE")
        cfg = DuplicateDetectorConfig.from_env()
        assert cfg.enable_metrics is True

    def test_env_bool_non_truthy_string_is_false(self, monkeypatch):
        monkeypatch.setenv("GL_DD_CACHE_ENABLED", "nope")
        cfg = DuplicateDetectorConfig.from_env()
        assert cfg.cache_enabled is False

    def test_env_bool_empty_string_is_false(self, monkeypatch):
        monkeypatch.setenv("GL_DD_CACHE_ENABLED", "")
        cfg = DuplicateDetectorConfig.from_env()
        assert cfg.cache_enabled is False


# =============================================================================
# Test invalid env values (fallback to defaults)
# =============================================================================


class TestConfigInvalidEnvValues:
    """Verify graceful fallback when env vars contain invalid values."""

    def test_invalid_int_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("GL_DD_MAX_RECORDS_PER_JOB", "not_a_number")
        cfg = DuplicateDetectorConfig.from_env()
        assert cfg.max_records_per_job == 1_000_000

    def test_invalid_float_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("GL_DD_MATCH_THRESHOLD", "abc")
        cfg = DuplicateDetectorConfig.from_env()
        assert cfg.match_threshold == pytest.approx(0.85)

    def test_invalid_int_batch_size(self, monkeypatch):
        monkeypatch.setenv("GL_DD_DEFAULT_BATCH_SIZE", "3.14")
        cfg = DuplicateDetectorConfig.from_env()
        # float string is not a valid int, falls back
        assert cfg.default_batch_size == 10_000

    def test_invalid_int_window_size(self, monkeypatch):
        monkeypatch.setenv("GL_DD_BLOCKING_WINDOW_SIZE", "xyz")
        cfg = DuplicateDetectorConfig.from_env()
        assert cfg.blocking_window_size == 10

    def test_invalid_float_canopy_tight(self, monkeypatch):
        monkeypatch.setenv("GL_DD_CANOPY_TIGHT_THRESHOLD", "high")
        cfg = DuplicateDetectorConfig.from_env()
        assert cfg.canopy_tight_threshold == pytest.approx(0.8)

    def test_invalid_float_comparison_sample_rate(self, monkeypatch):
        monkeypatch.setenv("GL_DD_COMPARISON_SAMPLE_RATE", "all")
        cfg = DuplicateDetectorConfig.from_env()
        assert cfg.comparison_sample_rate == pytest.approx(1.0)

    def test_invalid_int_pool_max_size(self, monkeypatch):
        monkeypatch.setenv("GL_DD_POOL_MAX_SIZE", "ten")
        cfg = DuplicateDetectorConfig.from_env()
        assert cfg.pool_max_size == 10

    def test_invalid_int_pipeline_timeout(self, monkeypatch):
        monkeypatch.setenv("GL_DD_PIPELINE_TIMEOUT_SECONDS", "1h")
        cfg = DuplicateDetectorConfig.from_env()
        assert cfg.pipeline_timeout_seconds == 3600

    def test_invalid_float_non_match_threshold(self, monkeypatch):
        monkeypatch.setenv("GL_DD_NON_MATCH_THRESHOLD", "low")
        cfg = DuplicateDetectorConfig.from_env()
        assert cfg.non_match_threshold == pytest.approx(0.40)

    def test_invalid_int_ngram_size(self, monkeypatch):
        monkeypatch.setenv("GL_DD_NGRAM_SIZE", "three")
        cfg = DuplicateDetectorConfig.from_env()
        assert cfg.ngram_size == 3

    def test_invalid_int_checkpoint_interval(self, monkeypatch):
        monkeypatch.setenv("GL_DD_PIPELINE_CHECKPOINT_INTERVAL", "!")
        cfg = DuplicateDetectorConfig.from_env()
        assert cfg.pipeline_checkpoint_interval == 1000

    def test_invalid_int_max_comparisons(self, monkeypatch):
        monkeypatch.setenv("GL_DD_MAX_COMPARISONS_PER_BLOCK", "many")
        cfg = DuplicateDetectorConfig.from_env()
        assert cfg.max_comparisons_per_block == 50_000


# =============================================================================
# Test singleton pattern: get_config / reset_config / set_config
# =============================================================================


class TestConfigSingleton:
    """Verify thread-safe singleton accessor functions."""

    def test_get_config_returns_instance(self):
        cfg = get_config()
        assert isinstance(cfg, DuplicateDetectorConfig)

    def test_get_config_returns_same_instance(self):
        cfg1 = get_config()
        cfg2 = get_config()
        assert cfg1 is cfg2

    def test_reset_config_clears_singleton(self):
        cfg1 = get_config()
        reset_config()
        cfg2 = get_config()
        # After reset, a new instance is created
        assert cfg1 is not cfg2

    def test_set_config_replaces_singleton(self, config):
        set_config(config)
        cfg = get_config()
        assert cfg is config
        assert cfg.database_url == "postgresql://test:test@localhost:5432/testdb"

    def test_set_config_then_get_returns_same(self, config):
        set_config(config)
        assert get_config() is config
        assert get_config() is config  # still the same

    def test_reset_after_set(self, config):
        set_config(config)
        reset_config()
        cfg = get_config()
        assert cfg is not config
        # Should be a fresh from_env instance (defaults since env is clean)
        assert cfg.database_url == ""

    def test_get_config_picks_up_env(self, monkeypatch):
        monkeypatch.setenv("GL_DD_S3_BUCKET", "singleton-test-bucket")
        cfg = get_config()
        assert cfg.s3_bucket == "singleton-test-bucket"


# =============================================================================
# Thread safety
# =============================================================================


class TestConfigThreadSafety:
    """Verify get_config under concurrent access."""

    def test_concurrent_get_config_returns_same_instance(self):
        results = []
        barrier = threading.Barrier(10)

        def worker():
            barrier.wait()
            results.append(get_config())

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads should get the exact same object
        assert len(results) == 10
        first = results[0]
        for cfg in results[1:]:
            assert cfg is first

    def test_concurrent_reset_and_get(self):
        """Reset and get concurrently should not raise."""
        errors = []

        def worker_get():
            try:
                for _ in range(50):
                    get_config()
            except Exception as exc:
                errors.append(exc)

        def worker_reset():
            try:
                for _ in range(50):
                    reset_config()
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=worker_get),
            threading.Thread(target=worker_reset),
            threading.Thread(target=worker_get),
            threading.Thread(target=worker_reset),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_set_config(self, config):
        """Multiple set_config calls should not corrupt state."""
        errors = []

        def worker(idx):
            try:
                c = DuplicateDetectorConfig(s3_bucket=f"bucket-{idx}")
                set_config(c)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # After all sets, get_config should return a valid config
        result = get_config()
        assert isinstance(result, DuplicateDetectorConfig)


# =============================================================================
# Test explicit construction with custom values
# =============================================================================


class TestConfigCustomConstruction:
    """Verify constructing config with non-default values."""

    def test_custom_database_url(self):
        cfg = DuplicateDetectorConfig(database_url="postgresql://custom/db")
        assert cfg.database_url == "postgresql://custom/db"

    def test_custom_match_threshold(self):
        cfg = DuplicateDetectorConfig(match_threshold=0.95)
        assert cfg.match_threshold == pytest.approx(0.95)

    def test_custom_use_fellegi_sunter(self):
        cfg = DuplicateDetectorConfig(use_fellegi_sunter=True)
        assert cfg.use_fellegi_sunter is True

    def test_custom_multiple_fields(self):
        cfg = DuplicateDetectorConfig(
            max_records_per_job=500,
            default_batch_size=50,
            fingerprint_algorithm="minhash",
            blocking_strategy="canopy",
            match_threshold=0.90,
            cluster_algorithm="connected_components",
        )
        assert cfg.max_records_per_job == 500
        assert cfg.default_batch_size == 50
        assert cfg.fingerprint_algorithm == "minhash"
        assert cfg.blocking_strategy == "canopy"
        assert cfg.match_threshold == pytest.approx(0.90)
        assert cfg.cluster_algorithm == "connected_components"

    def test_custom_all_connection_fields(self):
        cfg = DuplicateDetectorConfig(
            database_url="pg://host/db",
            redis_url="redis://host:6379",
            s3_bucket="my-bucket",
        )
        assert cfg.database_url == "pg://host/db"
        assert cfg.redis_url == "redis://host:6379"
        assert cfg.s3_bucket == "my-bucket"


# =============================================================================
# Test from_env with no env vars set (all defaults)
# =============================================================================


class TestConfigFromEnvDefaults:
    """from_env with no GL_DD_ vars should match raw defaults."""

    def test_from_env_matches_raw_defaults(self):
        raw = DuplicateDetectorConfig()
        env = DuplicateDetectorConfig.from_env()

        # Compare all fields
        for field_name in raw.__dataclass_fields__:
            assert getattr(raw, field_name) == getattr(env, field_name), (
                f"Mismatch on field {field_name}"
            )


# =============================================================================
# Test combined env overrides (multiple at once)
# =============================================================================


class TestConfigCombinedEnvOverrides:
    """Set multiple env vars simultaneously and verify all take effect."""

    def test_multiple_env_overrides(self, monkeypatch):
        monkeypatch.setenv("GL_DD_DATABASE_URL", "pg://multi")
        monkeypatch.setenv("GL_DD_MAX_RECORDS_PER_JOB", "250000")
        monkeypatch.setenv("GL_DD_MATCH_THRESHOLD", "0.92")
        monkeypatch.setenv("GL_DD_USE_FELLEGI_SUNTER", "true")
        monkeypatch.setenv("GL_DD_CACHE_ENABLED", "false")
        monkeypatch.setenv("GL_DD_LOG_LEVEL", "WARNING")

        cfg = DuplicateDetectorConfig.from_env()

        assert cfg.database_url == "pg://multi"
        assert cfg.max_records_per_job == 250_000
        assert cfg.match_threshold == pytest.approx(0.92)
        assert cfg.use_fellegi_sunter is True
        assert cfg.cache_enabled is False
        assert cfg.log_level == "WARNING"

    def test_mixed_valid_and_invalid_overrides(self, monkeypatch):
        monkeypatch.setenv("GL_DD_MAX_RECORDS_PER_JOB", "bad")  # invalid
        monkeypatch.setenv("GL_DD_MATCH_THRESHOLD", "0.88")  # valid
        monkeypatch.setenv("GL_DD_POOL_MIN_SIZE", "xxx")  # invalid
        monkeypatch.setenv("GL_DD_FINGERPRINT_NORMALIZE", "1")  # valid

        cfg = DuplicateDetectorConfig.from_env()

        assert cfg.max_records_per_job == 1_000_000  # default fallback
        assert cfg.match_threshold == pytest.approx(0.88)
        assert cfg.pool_min_size == 2  # default fallback
        assert cfg.fingerprint_normalize is True


# =============================================================================
# Test dataclass behavior
# =============================================================================


class TestConfigDataclassBehavior:
    """Verify dataclass features: equality, repr, field access."""

    def test_two_default_configs_are_equal(self):
        c1 = DuplicateDetectorConfig()
        c2 = DuplicateDetectorConfig()
        assert c1 == c2

    def test_different_configs_are_not_equal(self):
        c1 = DuplicateDetectorConfig()
        c2 = DuplicateDetectorConfig(match_threshold=0.99)
        assert c1 != c2

    def test_config_has_repr(self):
        cfg = DuplicateDetectorConfig()
        r = repr(cfg)
        assert "DuplicateDetectorConfig" in r
        assert "match_threshold" in r

    def test_config_fields_are_mutable(self):
        cfg = DuplicateDetectorConfig()
        cfg.match_threshold = 0.99
        assert cfg.match_threshold == pytest.approx(0.99)

    def test_config_is_dataclass(self):
        import dataclasses
        assert dataclasses.is_dataclass(DuplicateDetectorConfig)

    def test_config_field_names(self):
        cfg = DuplicateDetectorConfig()
        names = list(cfg.__dataclass_fields__.keys())
        assert "database_url" in names
        assert "match_threshold" in names
        assert "comparison_sample_rate" in names


# =============================================================================
# Test __all__ exports
# =============================================================================


class TestConfigExports:
    """Verify the config module exports the expected names."""

    def test_all_exports(self):
        import greenlang.duplicate_detector.config as mod
        assert "DuplicateDetectorConfig" in mod.__all__
        assert "get_config" in mod.__all__
        assert "set_config" in mod.__all__
        assert "reset_config" in mod.__all__

    def test_env_prefix_is_gl_dd(self):
        from greenlang.duplicate_detector.config import _ENV_PREFIX
        assert _ENV_PREFIX == "GL_DD_"


# =============================================================================
# Test edge cases on env parsing
# =============================================================================


class TestConfigEnvEdgeCases:
    """Edge cases for environment variable parsing."""

    def test_env_int_zero(self, monkeypatch):
        monkeypatch.setenv("GL_DD_BLOCKING_WINDOW_SIZE", "0")
        cfg = DuplicateDetectorConfig.from_env()
        assert cfg.blocking_window_size == 0

    def test_env_int_negative(self, monkeypatch):
        monkeypatch.setenv("GL_DD_MAX_RECORDS_PER_JOB", "-1")
        cfg = DuplicateDetectorConfig.from_env()
        assert cfg.max_records_per_job == -1

    def test_env_float_zero(self, monkeypatch):
        monkeypatch.setenv("GL_DD_MATCH_THRESHOLD", "0.0")
        cfg = DuplicateDetectorConfig.from_env()
        assert cfg.match_threshold == pytest.approx(0.0)

    def test_env_float_one(self, monkeypatch):
        monkeypatch.setenv("GL_DD_COMPARISON_SAMPLE_RATE", "1.0")
        cfg = DuplicateDetectorConfig.from_env()
        assert cfg.comparison_sample_rate == pytest.approx(1.0)

    def test_env_string_with_spaces(self, monkeypatch):
        monkeypatch.setenv("GL_DD_DATABASE_URL", "  pg://host/db  ")
        cfg = DuplicateDetectorConfig.from_env()
        # Strings are passed as-is (including spaces)
        assert cfg.database_url == "  pg://host/db  "

    def test_env_empty_string(self, monkeypatch):
        monkeypatch.setenv("GL_DD_S3_BUCKET", "")
        cfg = DuplicateDetectorConfig.from_env()
        assert cfg.s3_bucket == ""

    def test_env_large_int(self, monkeypatch):
        monkeypatch.setenv("GL_DD_MAX_RECORDS_PER_JOB", "999999999")
        cfg = DuplicateDetectorConfig.from_env()
        assert cfg.max_records_per_job == 999_999_999

    def test_env_float_high_precision(self, monkeypatch):
        monkeypatch.setenv("GL_DD_MATCH_THRESHOLD", "0.123456789")
        cfg = DuplicateDetectorConfig.from_env()
        assert cfg.match_threshold == pytest.approx(0.123456789)

    def test_env_bool_TRUE_uppercase(self, monkeypatch):
        monkeypatch.setenv("GL_DD_ENABLE_METRICS", "TRUE")
        cfg = DuplicateDetectorConfig.from_env()
        assert cfg.enable_metrics is True

    def test_env_bool_True_mixed_case(self, monkeypatch):
        monkeypatch.setenv("GL_DD_ENABLE_METRICS", "True")
        cfg = DuplicateDetectorConfig.from_env()
        assert cfg.enable_metrics is True

    def test_env_bool_0_is_false(self, monkeypatch):
        monkeypatch.setenv("GL_DD_ENABLE_METRICS", "0")
        cfg = DuplicateDetectorConfig.from_env()
        assert cfg.enable_metrics is False

    def test_env_bool_no_is_false(self, monkeypatch):
        monkeypatch.setenv("GL_DD_CACHE_ENABLED", "no")
        cfg = DuplicateDetectorConfig.from_env()
        assert cfg.cache_enabled is False
