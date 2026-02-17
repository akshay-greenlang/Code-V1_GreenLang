# -*- coding: utf-8 -*-
"""
Unit Tests for SchemaMigrationConfig - AGENT-DATA-017

Tests the SchemaMigrationConfig dataclass, all 25 default values,
environment variable overrides (GL_SM_ prefix), type coercion fallback,
post-init validation constraints, thread-safe singleton management,
serialisation helpers (to_dict, repr), and equality behaviour.

Target: 120+ tests, 85%+ coverage of greenlang.schema_migration.config

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import dataclasses
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from greenlang.schema_migration.config import (
    SchemaMigrationConfig,
    get_config,
    reset_config,
    set_config,
)


# ============================================================================
# TestSchemaMigrationConfigDefaults - verify every default value (25 fields)
# ============================================================================


class TestSchemaMigrationConfigDefaults:
    """Every field of SchemaMigrationConfig must have the correct default."""

    def test_default_database_url(self):
        cfg = SchemaMigrationConfig()
        assert cfg.database_url == ""

    def test_default_redis_url(self):
        cfg = SchemaMigrationConfig()
        assert cfg.redis_url == ""

    def test_default_log_level(self):
        cfg = SchemaMigrationConfig()
        assert cfg.log_level == "INFO"

    def test_default_max_schemas(self):
        cfg = SchemaMigrationConfig()
        assert cfg.max_schemas == 50_000

    def test_default_max_versions_per_schema(self):
        cfg = SchemaMigrationConfig()
        assert cfg.max_versions_per_schema == 1_000

    def test_default_max_migration_batch_size(self):
        cfg = SchemaMigrationConfig()
        assert cfg.max_migration_batch_size == 10_000

    def test_default_migration_timeout_seconds(self):
        cfg = SchemaMigrationConfig()
        assert cfg.migration_timeout_seconds == 3_600

    def test_default_enable_dry_run(self):
        cfg = SchemaMigrationConfig()
        assert cfg.enable_dry_run is True

    def test_default_enable_auto_rollback(self):
        cfg = SchemaMigrationConfig()
        assert cfg.enable_auto_rollback is True

    def test_default_compatibility_default_level(self):
        cfg = SchemaMigrationConfig()
        assert cfg.compatibility_default_level == "backward"

    def test_default_drift_check_interval_minutes(self):
        cfg = SchemaMigrationConfig()
        assert cfg.drift_check_interval_minutes == 60

    def test_default_drift_sample_size(self):
        cfg = SchemaMigrationConfig()
        assert cfg.drift_sample_size == 1_000

    def test_default_enable_provenance(self):
        cfg = SchemaMigrationConfig()
        assert cfg.enable_provenance is True

    def test_default_genesis_hash(self):
        cfg = SchemaMigrationConfig()
        assert cfg.genesis_hash == "greenlang-schema-migration-genesis"

    def test_default_max_workers(self):
        cfg = SchemaMigrationConfig()
        assert cfg.max_workers == 4

    def test_default_pool_size(self):
        cfg = SchemaMigrationConfig()
        assert cfg.pool_size == 5

    def test_default_cache_ttl(self):
        cfg = SchemaMigrationConfig()
        assert cfg.cache_ttl == 300

    def test_default_rate_limit(self):
        cfg = SchemaMigrationConfig()
        assert cfg.rate_limit == 100

    def test_default_checkpoint_interval(self):
        cfg = SchemaMigrationConfig()
        assert cfg.checkpoint_interval == 100

    def test_default_retry_max_attempts(self):
        cfg = SchemaMigrationConfig()
        assert cfg.retry_max_attempts == 3

    def test_default_retry_backoff_base(self):
        cfg = SchemaMigrationConfig()
        assert cfg.retry_backoff_base == pytest.approx(2.0)

    def test_default_field_mapping_min_confidence(self):
        cfg = SchemaMigrationConfig()
        assert cfg.field_mapping_min_confidence == pytest.approx(0.8)

    def test_default_deprecation_warning_days(self):
        cfg = SchemaMigrationConfig()
        assert cfg.deprecation_warning_days == 30

    def test_default_max_change_depth(self):
        cfg = SchemaMigrationConfig()
        assert cfg.max_change_depth == 10

    def test_default_enable_impact_analysis(self):
        cfg = SchemaMigrationConfig()
        assert cfg.enable_impact_analysis is True


# ============================================================================
# TestSchemaMigrationConfigFromEnv - verify every GL_SM_* env var override
# ============================================================================


class TestSchemaMigrationConfigFromEnv:
    """Every field can be overridden via GL_SM_<FIELD_UPPER> env var."""

    def test_env_database_url(self, monkeypatch):
        monkeypatch.setenv("GL_SM_DATABASE_URL", "postgresql://test:5432/sm")
        cfg = SchemaMigrationConfig.from_env()
        assert cfg.database_url == "postgresql://test:5432/sm"

    def test_env_redis_url(self, monkeypatch):
        monkeypatch.setenv("GL_SM_REDIS_URL", "redis://test:6379/0")
        cfg = SchemaMigrationConfig.from_env()
        assert cfg.redis_url == "redis://test:6379/0"

    def test_env_log_level(self, monkeypatch):
        monkeypatch.setenv("GL_SM_LOG_LEVEL", "DEBUG")
        cfg = SchemaMigrationConfig.from_env()
        assert cfg.log_level == "DEBUG"

    def test_env_max_schemas(self, monkeypatch):
        monkeypatch.setenv("GL_SM_MAX_SCHEMAS", "100000")
        cfg = SchemaMigrationConfig.from_env()
        assert cfg.max_schemas == 100_000

    def test_env_max_versions_per_schema(self, monkeypatch):
        monkeypatch.setenv("GL_SM_MAX_VERSIONS_PER_SCHEMA", "5000")
        cfg = SchemaMigrationConfig.from_env()
        assert cfg.max_versions_per_schema == 5_000

    def test_env_max_migration_batch_size(self, monkeypatch):
        monkeypatch.setenv("GL_SM_MAX_MIGRATION_BATCH_SIZE", "50000")
        cfg = SchemaMigrationConfig.from_env()
        assert cfg.max_migration_batch_size == 50_000

    def test_env_migration_timeout_seconds(self, monkeypatch):
        monkeypatch.setenv("GL_SM_MIGRATION_TIMEOUT_SECONDS", "7200")
        cfg = SchemaMigrationConfig.from_env()
        assert cfg.migration_timeout_seconds == 7_200

    def test_env_enable_dry_run_true(self, monkeypatch):
        monkeypatch.setenv("GL_SM_ENABLE_DRY_RUN", "true")
        cfg = SchemaMigrationConfig.from_env()
        assert cfg.enable_dry_run is True

    def test_env_enable_dry_run_false(self, monkeypatch):
        monkeypatch.setenv("GL_SM_ENABLE_DRY_RUN", "false")
        cfg = SchemaMigrationConfig.from_env()
        assert cfg.enable_dry_run is False

    def test_env_enable_auto_rollback_true(self, monkeypatch):
        monkeypatch.setenv("GL_SM_ENABLE_AUTO_ROLLBACK", "1")
        cfg = SchemaMigrationConfig.from_env()
        assert cfg.enable_auto_rollback is True

    def test_env_enable_auto_rollback_false(self, monkeypatch):
        monkeypatch.setenv("GL_SM_ENABLE_AUTO_ROLLBACK", "0")
        cfg = SchemaMigrationConfig.from_env()
        assert cfg.enable_auto_rollback is False

    def test_env_compatibility_default_level(self, monkeypatch):
        monkeypatch.setenv("GL_SM_COMPATIBILITY_DEFAULT_LEVEL", "full")
        cfg = SchemaMigrationConfig.from_env()
        assert cfg.compatibility_default_level == "full"

    def test_env_drift_check_interval_minutes(self, monkeypatch):
        monkeypatch.setenv("GL_SM_DRIFT_CHECK_INTERVAL_MINUTES", "30")
        cfg = SchemaMigrationConfig.from_env()
        assert cfg.drift_check_interval_minutes == 30

    def test_env_drift_sample_size(self, monkeypatch):
        monkeypatch.setenv("GL_SM_DRIFT_SAMPLE_SIZE", "5000")
        cfg = SchemaMigrationConfig.from_env()
        assert cfg.drift_sample_size == 5_000

    def test_env_enable_provenance_true(self, monkeypatch):
        monkeypatch.setenv("GL_SM_ENABLE_PROVENANCE", "yes")
        cfg = SchemaMigrationConfig.from_env()
        assert cfg.enable_provenance is True

    def test_env_enable_provenance_false(self, monkeypatch):
        monkeypatch.setenv("GL_SM_ENABLE_PROVENANCE", "false")
        cfg = SchemaMigrationConfig.from_env()
        assert cfg.enable_provenance is False

    def test_env_genesis_hash(self, monkeypatch):
        monkeypatch.setenv("GL_SM_GENESIS_HASH", "custom-genesis-hash")
        cfg = SchemaMigrationConfig.from_env()
        assert cfg.genesis_hash == "custom-genesis-hash"

    def test_env_max_workers(self, monkeypatch):
        monkeypatch.setenv("GL_SM_MAX_WORKERS", "8")
        cfg = SchemaMigrationConfig.from_env()
        assert cfg.max_workers == 8

    def test_env_pool_size(self, monkeypatch):
        monkeypatch.setenv("GL_SM_POOL_SIZE", "10")
        cfg = SchemaMigrationConfig.from_env()
        assert cfg.pool_size == 10

    def test_env_cache_ttl(self, monkeypatch):
        monkeypatch.setenv("GL_SM_CACHE_TTL", "600")
        cfg = SchemaMigrationConfig.from_env()
        assert cfg.cache_ttl == 600

    def test_env_rate_limit(self, monkeypatch):
        monkeypatch.setenv("GL_SM_RATE_LIMIT", "200")
        cfg = SchemaMigrationConfig.from_env()
        assert cfg.rate_limit == 200

    def test_env_checkpoint_interval(self, monkeypatch):
        monkeypatch.setenv("GL_SM_CHECKPOINT_INTERVAL", "500")
        cfg = SchemaMigrationConfig.from_env()
        assert cfg.checkpoint_interval == 500

    def test_env_retry_max_attempts(self, monkeypatch):
        monkeypatch.setenv("GL_SM_RETRY_MAX_ATTEMPTS", "5")
        cfg = SchemaMigrationConfig.from_env()
        assert cfg.retry_max_attempts == 5

    def test_env_retry_backoff_base(self, monkeypatch):
        monkeypatch.setenv("GL_SM_RETRY_BACKOFF_BASE", "3.0")
        cfg = SchemaMigrationConfig.from_env()
        assert cfg.retry_backoff_base == pytest.approx(3.0)

    def test_env_field_mapping_min_confidence(self, monkeypatch):
        monkeypatch.setenv("GL_SM_FIELD_MAPPING_MIN_CONFIDENCE", "0.9")
        cfg = SchemaMigrationConfig.from_env()
        assert cfg.field_mapping_min_confidence == pytest.approx(0.9)

    def test_env_deprecation_warning_days(self, monkeypatch):
        monkeypatch.setenv("GL_SM_DEPRECATION_WARNING_DAYS", "60")
        cfg = SchemaMigrationConfig.from_env()
        assert cfg.deprecation_warning_days == 60

    def test_env_max_change_depth(self, monkeypatch):
        monkeypatch.setenv("GL_SM_MAX_CHANGE_DEPTH", "20")
        cfg = SchemaMigrationConfig.from_env()
        assert cfg.max_change_depth == 20

    def test_env_enable_impact_analysis_true(self, monkeypatch):
        monkeypatch.setenv("GL_SM_ENABLE_IMPACT_ANALYSIS", "true")
        cfg = SchemaMigrationConfig.from_env()
        assert cfg.enable_impact_analysis is True

    def test_env_enable_impact_analysis_false(self, monkeypatch):
        monkeypatch.setenv("GL_SM_ENABLE_IMPACT_ANALYSIS", "false")
        cfg = SchemaMigrationConfig.from_env()
        assert cfg.enable_impact_analysis is False


# ============================================================================
# TestSchemaMigrationConfigValidation - edge cases and constraint checking
# ============================================================================


class TestSchemaMigrationConfigValidation:
    """Edge cases: negative values, empty strings, boundary values, invalid enums."""

    def test_zero_max_schemas_raises(self):
        with pytest.raises(ValueError, match="max_schemas must be > 0"):
            SchemaMigrationConfig(max_schemas=0)

    def test_negative_max_schemas_raises(self):
        with pytest.raises(ValueError, match="max_schemas must be > 0"):
            SchemaMigrationConfig(max_schemas=-1)

    def test_zero_max_versions_per_schema_raises(self):
        with pytest.raises(ValueError, match="max_versions_per_schema must be > 0"):
            SchemaMigrationConfig(max_versions_per_schema=0)

    def test_zero_max_migration_batch_size_raises(self):
        with pytest.raises(ValueError, match="max_migration_batch_size must be > 0"):
            SchemaMigrationConfig(max_migration_batch_size=0)

    def test_zero_migration_timeout_raises(self):
        with pytest.raises(ValueError, match="migration_timeout_seconds must be > 0"):
            SchemaMigrationConfig(migration_timeout_seconds=0)

    def test_invalid_compatibility_level_raises(self):
        with pytest.raises(ValueError, match="compatibility_default_level"):
            SchemaMigrationConfig(compatibility_default_level="invalid_level")

    def test_valid_compatibility_backward(self):
        cfg = SchemaMigrationConfig(compatibility_default_level="backward")
        assert cfg.compatibility_default_level == "backward"

    def test_valid_compatibility_forward(self):
        cfg = SchemaMigrationConfig(compatibility_default_level="forward")
        assert cfg.compatibility_default_level == "forward"

    def test_valid_compatibility_full(self):
        cfg = SchemaMigrationConfig(compatibility_default_level="full")
        assert cfg.compatibility_default_level == "full"

    def test_valid_compatibility_none(self):
        cfg = SchemaMigrationConfig(compatibility_default_level="none")
        assert cfg.compatibility_default_level == "none"

    def test_valid_compatibility_backward_transitive(self):
        cfg = SchemaMigrationConfig(compatibility_default_level="backward_transitive")
        assert cfg.compatibility_default_level == "backward_transitive"

    def test_valid_compatibility_forward_transitive(self):
        cfg = SchemaMigrationConfig(compatibility_default_level="forward_transitive")
        assert cfg.compatibility_default_level == "forward_transitive"

    def test_valid_compatibility_full_transitive(self):
        cfg = SchemaMigrationConfig(compatibility_default_level="full_transitive")
        assert cfg.compatibility_default_level == "full_transitive"

    def test_zero_drift_check_interval_raises(self):
        with pytest.raises(ValueError, match="drift_check_interval_minutes must be > 0"):
            SchemaMigrationConfig(drift_check_interval_minutes=0)

    def test_zero_drift_sample_size_raises(self):
        with pytest.raises(ValueError, match="drift_sample_size must be > 0"):
            SchemaMigrationConfig(drift_sample_size=0)

    def test_empty_genesis_hash_raises(self):
        with pytest.raises(ValueError, match="genesis_hash must not be empty"):
            SchemaMigrationConfig(genesis_hash="")

    def test_zero_max_workers_raises(self):
        with pytest.raises(ValueError, match="max_workers must be > 0"):
            SchemaMigrationConfig(max_workers=0)

    def test_zero_pool_size_raises(self):
        with pytest.raises(ValueError, match="pool_size must be > 0"):
            SchemaMigrationConfig(pool_size=0)

    def test_zero_cache_ttl_raises(self):
        with pytest.raises(ValueError, match="cache_ttl must be > 0"):
            SchemaMigrationConfig(cache_ttl=0)

    def test_zero_rate_limit_raises(self):
        with pytest.raises(ValueError, match="rate_limit must be > 0"):
            SchemaMigrationConfig(rate_limit=0)

    def test_zero_checkpoint_interval_raises(self):
        with pytest.raises(ValueError, match="checkpoint_interval must be > 0"):
            SchemaMigrationConfig(checkpoint_interval=0)

    def test_zero_retry_max_attempts_raises(self):
        with pytest.raises(ValueError, match="retry_max_attempts must be > 0"):
            SchemaMigrationConfig(retry_max_attempts=0)

    def test_zero_retry_backoff_base_raises(self):
        with pytest.raises(ValueError, match="retry_backoff_base must be > 0"):
            SchemaMigrationConfig(retry_backoff_base=0.0)

    def test_negative_retry_backoff_base_raises(self):
        with pytest.raises(ValueError, match="retry_backoff_base must be > 0"):
            SchemaMigrationConfig(retry_backoff_base=-1.0)

    def test_field_mapping_confidence_below_zero_raises(self):
        with pytest.raises(ValueError, match="field_mapping_min_confidence"):
            SchemaMigrationConfig(field_mapping_min_confidence=-0.1)

    def test_field_mapping_confidence_above_one_raises(self):
        with pytest.raises(ValueError, match="field_mapping_min_confidence"):
            SchemaMigrationConfig(field_mapping_min_confidence=1.1)

    def test_field_mapping_confidence_at_zero_ok(self):
        cfg = SchemaMigrationConfig(field_mapping_min_confidence=0.0)
        assert cfg.field_mapping_min_confidence == pytest.approx(0.0)

    def test_field_mapping_confidence_at_one_ok(self):
        cfg = SchemaMigrationConfig(field_mapping_min_confidence=1.0)
        assert cfg.field_mapping_min_confidence == pytest.approx(1.0)

    def test_negative_deprecation_warning_days_raises(self):
        with pytest.raises(ValueError, match="deprecation_warning_days"):
            SchemaMigrationConfig(deprecation_warning_days=-1)

    def test_zero_deprecation_warning_days_ok(self):
        cfg = SchemaMigrationConfig(deprecation_warning_days=0)
        assert cfg.deprecation_warning_days == 0

    def test_zero_max_change_depth_raises(self):
        with pytest.raises(ValueError, match="max_change_depth must be > 0"):
            SchemaMigrationConfig(max_change_depth=0)

    def test_invalid_log_level_raises(self):
        with pytest.raises(ValueError, match="log_level"):
            SchemaMigrationConfig(log_level="TRACE")

    def test_log_level_normalized_to_upper(self):
        cfg = SchemaMigrationConfig(log_level="debug")
        assert cfg.log_level == "DEBUG"

    def test_log_level_warning_accepted(self):
        cfg = SchemaMigrationConfig(log_level="WARNING")
        assert cfg.log_level == "WARNING"

    def test_log_level_error_accepted(self):
        cfg = SchemaMigrationConfig(log_level="ERROR")
        assert cfg.log_level == "ERROR"

    def test_log_level_critical_accepted(self):
        cfg = SchemaMigrationConfig(log_level="CRITICAL")
        assert cfg.log_level == "CRITICAL"

    def test_boundary_max_schemas_one(self):
        cfg = SchemaMigrationConfig(max_schemas=1)
        assert cfg.max_schemas == 1

    def test_boundary_large_max_schemas(self):
        cfg = SchemaMigrationConfig(max_schemas=10_000_000)
        assert cfg.max_schemas == 10_000_000


# ============================================================================
# TestSchemaMigrationConfigTypeCoercion - invalid env fallbacks
# ============================================================================


class TestSchemaMigrationConfigTypeCoercion:
    """Invalid environment values must fall back to defaults."""

    def test_invalid_int_falls_back(self, monkeypatch):
        monkeypatch.setenv("GL_SM_MAX_SCHEMAS", "not_a_number")
        cfg = SchemaMigrationConfig.from_env()
        assert cfg.max_schemas == 50_000

    def test_invalid_float_falls_back(self, monkeypatch):
        monkeypatch.setenv("GL_SM_RETRY_BACKOFF_BASE", "abc")
        cfg = SchemaMigrationConfig.from_env()
        assert cfg.retry_backoff_base == pytest.approx(2.0)

    def test_empty_string_for_int_falls_back(self, monkeypatch):
        monkeypatch.setenv("GL_SM_MAX_WORKERS", "")
        cfg = SchemaMigrationConfig.from_env()
        assert cfg.max_workers == 4

    def test_empty_string_for_float_falls_back(self, monkeypatch):
        monkeypatch.setenv("GL_SM_FIELD_MAPPING_MIN_CONFIDENCE", "")
        cfg = SchemaMigrationConfig.from_env()
        assert cfg.field_mapping_min_confidence == pytest.approx(0.8)

    def test_float_as_int_truncated(self, monkeypatch):
        monkeypatch.setenv("GL_SM_POOL_SIZE", "3.14")
        cfg = SchemaMigrationConfig.from_env()
        # "3.14" cannot be parsed by int(), so falls back
        assert cfg.pool_size == 5

    def test_bool_no_case(self, monkeypatch):
        monkeypatch.setenv("GL_SM_ENABLE_DRY_RUN", "TRUE")
        cfg = SchemaMigrationConfig.from_env()
        assert cfg.enable_dry_run is True

    def test_bool_one_is_true(self, monkeypatch):
        monkeypatch.setenv("GL_SM_ENABLE_PROVENANCE", "1")
        cfg = SchemaMigrationConfig.from_env()
        assert cfg.enable_provenance is True

    def test_bool_yes_is_true(self, monkeypatch):
        monkeypatch.setenv("GL_SM_ENABLE_AUTO_ROLLBACK", "yes")
        cfg = SchemaMigrationConfig.from_env()
        assert cfg.enable_auto_rollback is True

    def test_bool_zero_is_false(self, monkeypatch):
        monkeypatch.setenv("GL_SM_ENABLE_DRY_RUN", "0")
        cfg = SchemaMigrationConfig.from_env()
        assert cfg.enable_dry_run is False

    def test_bool_random_string_is_false(self, monkeypatch):
        monkeypatch.setenv("GL_SM_ENABLE_PROVENANCE", "maybe")
        cfg = SchemaMigrationConfig.from_env()
        assert cfg.enable_provenance is False

    def test_whitespace_in_int_trimmed(self, monkeypatch):
        monkeypatch.setenv("GL_SM_CACHE_TTL", "  600  ")
        cfg = SchemaMigrationConfig.from_env()
        assert cfg.cache_ttl == 600

    def test_whitespace_in_float_trimmed(self, monkeypatch):
        monkeypatch.setenv("GL_SM_RETRY_BACKOFF_BASE", "  3.5  ")
        cfg = SchemaMigrationConfig.from_env()
        assert cfg.retry_backoff_base == pytest.approx(3.5)

    def test_whitespace_in_str_trimmed(self, monkeypatch):
        monkeypatch.setenv("GL_SM_LOG_LEVEL", "  WARNING  ")
        cfg = SchemaMigrationConfig.from_env()
        assert cfg.log_level == "WARNING"


# ============================================================================
# TestSchemaMigrationConfigSingleton - get_config, set_config, reset_config
# ============================================================================


class TestSchemaMigrationConfigSingleton:
    """Thread-safe singleton accessor functions."""

    def test_get_config_returns_instance(self):
        cfg = get_config()
        assert isinstance(cfg, SchemaMigrationConfig)

    def test_get_config_returns_same_instance(self):
        cfg1 = get_config()
        cfg2 = get_config()
        assert cfg1 is cfg2

    def test_set_config_replaces_instance(self):
        custom = SchemaMigrationConfig(log_level="DEBUG")
        set_config(custom)
        assert get_config() is custom
        assert get_config().log_level == "DEBUG"

    def test_reset_config_clears_instance(self):
        _ = get_config()
        reset_config()
        # After reset, next get_config creates a new instance
        cfg = get_config()
        assert isinstance(cfg, SchemaMigrationConfig)

    def test_reset_then_set(self):
        reset_config()
        custom = SchemaMigrationConfig(max_schemas=99)
        set_config(custom)
        assert get_config().max_schemas == 99

    def test_set_config_overrides_previous(self):
        c1 = SchemaMigrationConfig(pool_size=7)
        c2 = SchemaMigrationConfig(pool_size=12)
        set_config(c1)
        assert get_config().pool_size == 7
        set_config(c2)
        assert get_config().pool_size == 12

    def test_get_config_reads_env_on_first_call(self, monkeypatch):
        monkeypatch.setenv("GL_SM_MAX_WORKERS", "16")
        reset_config()
        cfg = get_config()
        assert cfg.max_workers == 16

    def test_reset_then_get_reads_env_again(self, monkeypatch):
        monkeypatch.setenv("GL_SM_RATE_LIMIT", "500")
        reset_config()
        cfg = get_config()
        assert cfg.rate_limit == 500

    def test_set_config_ignores_env(self, monkeypatch):
        monkeypatch.setenv("GL_SM_CACHE_TTL", "9999")
        custom = SchemaMigrationConfig(cache_ttl=42)
        set_config(custom)
        assert get_config().cache_ttl == 42

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
                c = SchemaMigrationConfig(max_workers=idx + 1)
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
# TestSchemaMigrationConfigMultipleOverrides - combined env overrides
# ============================================================================


class TestSchemaMigrationConfigMultipleOverrides:
    """Multiple env vars overridden simultaneously."""

    def test_all_connections_overridden(self, monkeypatch):
        monkeypatch.setenv("GL_SM_DATABASE_URL", "pg://a")
        monkeypatch.setenv("GL_SM_REDIS_URL", "redis://b")
        cfg = SchemaMigrationConfig.from_env()
        assert cfg.database_url == "pg://a"
        assert cfg.redis_url == "redis://b"

    def test_mixed_int_and_float_overrides(self, monkeypatch):
        monkeypatch.setenv("GL_SM_MAX_SCHEMAS", "200000")
        monkeypatch.setenv("GL_SM_RETRY_BACKOFF_BASE", "4.5")
        monkeypatch.setenv("GL_SM_FIELD_MAPPING_MIN_CONFIDENCE", "0.95")
        cfg = SchemaMigrationConfig.from_env()
        assert cfg.max_schemas == 200_000
        assert cfg.retry_backoff_base == pytest.approx(4.5)
        assert cfg.field_mapping_min_confidence == pytest.approx(0.95)

    def test_bool_and_str_overrides(self, monkeypatch):
        monkeypatch.setenv("GL_SM_ENABLE_DRY_RUN", "false")
        monkeypatch.setenv("GL_SM_ENABLE_AUTO_ROLLBACK", "false")
        monkeypatch.setenv("GL_SM_LOG_LEVEL", "WARNING")
        cfg = SchemaMigrationConfig.from_env()
        assert cfg.enable_dry_run is False
        assert cfg.enable_auto_rollback is False
        assert cfg.log_level == "WARNING"

    def test_pool_and_worker_overrides(self, monkeypatch):
        monkeypatch.setenv("GL_SM_MAX_WORKERS", "16")
        monkeypatch.setenv("GL_SM_POOL_SIZE", "20")
        cfg = SchemaMigrationConfig.from_env()
        assert cfg.max_workers == 16
        assert cfg.pool_size == 20

    def test_drift_overrides(self, monkeypatch):
        monkeypatch.setenv("GL_SM_DRIFT_CHECK_INTERVAL_MINUTES", "15")
        monkeypatch.setenv("GL_SM_DRIFT_SAMPLE_SIZE", "10000")
        cfg = SchemaMigrationConfig.from_env()
        assert cfg.drift_check_interval_minutes == 15
        assert cfg.drift_sample_size == 10_000

    def test_retry_overrides(self, monkeypatch):
        monkeypatch.setenv("GL_SM_RETRY_MAX_ATTEMPTS", "10")
        monkeypatch.setenv("GL_SM_RETRY_BACKOFF_BASE", "1.5")
        monkeypatch.setenv("GL_SM_CHECKPOINT_INTERVAL", "250")
        cfg = SchemaMigrationConfig.from_env()
        assert cfg.retry_max_attempts == 10
        assert cfg.retry_backoff_base == pytest.approx(1.5)
        assert cfg.checkpoint_interval == 250


# ============================================================================
# TestSchemaMigrationConfigRepr - repr/str output and to_dict
# ============================================================================


class TestSchemaMigrationConfigRepr:
    """SchemaMigrationConfig repr, str, and to_dict behaviour."""

    def test_repr_contains_class_name(self):
        cfg = SchemaMigrationConfig()
        assert "SchemaMigrationConfig" in repr(cfg)

    def test_repr_contains_field_values(self):
        cfg = SchemaMigrationConfig(log_level="ERROR")
        assert "ERROR" in repr(cfg)

    def test_repr_redacts_database_url(self):
        cfg = SchemaMigrationConfig(database_url="postgresql://secret:5432/db")
        r = repr(cfg)
        assert "secret" not in r
        assert "***" in r

    def test_repr_redacts_redis_url(self):
        cfg = SchemaMigrationConfig(redis_url="redis://secret:6379/0")
        r = repr(cfg)
        assert "secret" not in r
        assert "***" in r

    def test_repr_shows_empty_url_not_redacted(self):
        cfg = SchemaMigrationConfig(database_url="", redis_url="")
        r = repr(cfg)
        # Empty URLs should not be redacted to "***"
        assert "database_url=''" in r

    def test_to_dict_returns_dict(self):
        cfg = SchemaMigrationConfig()
        d = cfg.to_dict()
        assert isinstance(d, dict)

    def test_to_dict_has_all_keys(self):
        cfg = SchemaMigrationConfig()
        d = cfg.to_dict()
        expected_keys = {
            "database_url", "redis_url", "log_level",
            "max_schemas", "max_versions_per_schema",
            "max_migration_batch_size", "migration_timeout_seconds",
            "enable_dry_run", "enable_auto_rollback",
            "compatibility_default_level",
            "drift_check_interval_minutes", "drift_sample_size",
            "enable_provenance", "genesis_hash",
            "max_workers", "pool_size", "cache_ttl", "rate_limit",
            "checkpoint_interval", "retry_max_attempts", "retry_backoff_base",
            "field_mapping_min_confidence", "deprecation_warning_days",
            "max_change_depth", "enable_impact_analysis",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_redacts_database_url(self):
        cfg = SchemaMigrationConfig(database_url="pg://secret")
        d = cfg.to_dict()
        assert d["database_url"] == "***"

    def test_to_dict_redacts_redis_url(self):
        cfg = SchemaMigrationConfig(redis_url="redis://secret")
        d = cfg.to_dict()
        assert d["redis_url"] == "***"

    def test_to_dict_empty_url_not_redacted(self):
        cfg = SchemaMigrationConfig()
        d = cfg.to_dict()
        assert d["database_url"] == ""
        assert d["redis_url"] == ""


# ============================================================================
# TestSchemaMigrationConfigDataclass - equality, field count
# ============================================================================


class TestSchemaMigrationConfigDataclass:
    """SchemaMigrationConfig is a well-formed Python dataclass."""

    def test_is_dataclass(self):
        assert dataclasses.is_dataclass(SchemaMigrationConfig)

    def test_field_count(self):
        fields = dataclasses.fields(SchemaMigrationConfig)
        assert len(fields) == 25

    def test_equality_same_defaults(self):
        a = SchemaMigrationConfig()
        b = SchemaMigrationConfig()
        assert a == b

    def test_equality_different_values(self):
        a = SchemaMigrationConfig(max_schemas=100)
        b = SchemaMigrationConfig(max_schemas=200)
        assert a != b

    def test_field_names_match_expected(self):
        expected_names = {
            "database_url", "redis_url", "log_level",
            "max_schemas", "max_versions_per_schema",
            "max_migration_batch_size", "migration_timeout_seconds",
            "enable_dry_run", "enable_auto_rollback",
            "compatibility_default_level",
            "drift_check_interval_minutes", "drift_sample_size",
            "enable_provenance", "genesis_hash",
            "max_workers", "pool_size", "cache_ttl", "rate_limit",
            "checkpoint_interval", "retry_max_attempts", "retry_backoff_base",
            "field_mapping_min_confidence", "deprecation_warning_days",
            "max_change_depth", "enable_impact_analysis",
        }
        actual_names = {f.name for f in dataclasses.fields(SchemaMigrationConfig)}
        assert expected_names == actual_names

    def test_from_env_returns_config(self):
        cfg = SchemaMigrationConfig.from_env()
        assert isinstance(cfg, SchemaMigrationConfig)


# ============================================================================
# TestModuleExports - __all__ completeness
# ============================================================================


class TestModuleExports:
    """Verify config module exports."""

    def test_all_list_exists(self):
        from greenlang.schema_migration import config as mod
        assert hasattr(mod, "__all__")

    def test_all_contains_config_class(self):
        from greenlang.schema_migration import config as mod
        assert "SchemaMigrationConfig" in mod.__all__

    def test_all_contains_get_config(self):
        from greenlang.schema_migration import config as mod
        assert "get_config" in mod.__all__

    def test_all_contains_set_config(self):
        from greenlang.schema_migration import config as mod
        assert "set_config" in mod.__all__

    def test_all_contains_reset_config(self):
        from greenlang.schema_migration import config as mod
        assert "reset_config" in mod.__all__

    def test_all_has_four_entries(self):
        from greenlang.schema_migration import config as mod
        assert len(mod.__all__) == 4
