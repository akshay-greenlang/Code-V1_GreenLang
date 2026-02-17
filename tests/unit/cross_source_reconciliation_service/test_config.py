# -*- coding: utf-8 -*-
"""
Unit tests for CrossSourceReconciliationConfig - AGENT-DATA-015

Tests the config dataclass at greenlang.cross_source_reconciliation.config with
80+ tests covering default values for all 25 fields, GL_CSR_ environment
variable overrides, singleton pattern, thread safety, validation logic,
type coercion, invalid env fallback, and edge cases.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-015 Cross-Source Reconciliation (GL-DATA-X-018)
"""

from __future__ import annotations

import concurrent.futures
import threading
from dataclasses import fields as dc_fields

import pytest

from greenlang.cross_source_reconciliation.config import (
    CrossSourceReconciliationConfig,
    get_config,
    reset_config,
    set_config,
)


# ======================================================================
# 1. Default values -- all 25 fields (26 tests incl. count)
# ======================================================================


class TestDefaultValues:
    """Verify every field has the expected default."""

    def test_default_database_url(self, fresh_config):
        assert fresh_config.database_url == "postgresql://localhost:5432/greenlang"

    def test_default_redis_url(self, fresh_config):
        assert fresh_config.redis_url == "redis://localhost:6379/0"

    def test_default_log_level(self, fresh_config):
        assert fresh_config.log_level == "INFO"

    def test_default_batch_size(self, fresh_config):
        assert fresh_config.batch_size == 1000

    def test_default_max_records(self, fresh_config):
        assert fresh_config.max_records == 100_000

    def test_default_max_sources(self, fresh_config):
        assert fresh_config.max_sources == 20

    def test_default_match_threshold(self, fresh_config):
        assert fresh_config.default_match_threshold == pytest.approx(0.85)

    def test_default_tolerance_pct(self, fresh_config):
        assert fresh_config.default_tolerance_pct == pytest.approx(5.0)

    def test_default_tolerance_abs(self, fresh_config):
        assert fresh_config.default_tolerance_abs == pytest.approx(0.01)

    def test_default_resolution_strategy(self, fresh_config):
        assert fresh_config.default_resolution_strategy == "priority_wins"

    def test_default_source_credibility_weight(self, fresh_config):
        assert fresh_config.source_credibility_weight == pytest.approx(0.4)

    def test_default_temporal_alignment_enabled(self, fresh_config):
        assert fresh_config.temporal_alignment_enabled is True

    def test_default_fuzzy_matching_enabled(self, fresh_config):
        assert fresh_config.fuzzy_matching_enabled is True

    def test_default_max_match_candidates(self, fresh_config):
        assert fresh_config.max_match_candidates == 100

    def test_default_enable_golden_records(self, fresh_config):
        assert fresh_config.enable_golden_records is True

    def test_default_max_workers(self, fresh_config):
        assert fresh_config.max_workers == 4

    def test_default_pool_size(self, fresh_config):
        assert fresh_config.pool_size == 5

    def test_default_cache_ttl(self, fresh_config):
        assert fresh_config.cache_ttl == 3600

    def test_default_rate_limit(self, fresh_config):
        assert fresh_config.rate_limit == 100

    def test_default_enable_provenance(self, fresh_config):
        assert fresh_config.enable_provenance is True

    def test_default_manual_review_threshold(self, fresh_config):
        assert fresh_config.manual_review_threshold == pytest.approx(0.6)

    def test_default_critical_discrepancy_pct(self, fresh_config):
        assert fresh_config.critical_discrepancy_pct == pytest.approx(50.0)

    def test_default_high_discrepancy_pct(self, fresh_config):
        assert fresh_config.high_discrepancy_pct == pytest.approx(25.0)

    def test_default_medium_discrepancy_pct(self, fresh_config):
        assert fresh_config.medium_discrepancy_pct == pytest.approx(10.0)

    def test_default_genesis_hash(self, fresh_config):
        assert fresh_config.genesis_hash == "greenlang-cross-source-reconciliation-genesis"

    def test_total_field_count(self, fresh_config):
        """Config dataclass should have exactly 25 fields."""
        assert len(dc_fields(fresh_config)) == 25


# ======================================================================
# 2. Environment variable prefix
# ======================================================================


class TestEnvPrefix:
    """Test the GL_CSR_ environment variable prefix."""

    def test_env_prefix_used(self, monkeypatch):
        monkeypatch.setenv("GL_CSR_BATCH_SIZE", "500")
        cfg = CrossSourceReconciliationConfig.from_env()
        assert cfg.batch_size == 500

    def test_wrong_prefix_ignored(self, monkeypatch):
        monkeypatch.setenv("GL_OTHER_BATCH_SIZE", "999")
        cfg = CrossSourceReconciliationConfig.from_env()
        assert cfg.batch_size == 1000  # unchanged


# ======================================================================
# 3. Environment variable overrides -- one per type
# ======================================================================


class TestEnvOverrides:
    """Each field can be overridden via GL_CSR_<FIELD_UPPER>."""

    # -- String overrides ---------------------------------------------------

    def test_env_database_url(self, monkeypatch):
        monkeypatch.setenv("GL_CSR_DATABASE_URL", "postgresql://custom/db")
        cfg = CrossSourceReconciliationConfig.from_env()
        assert cfg.database_url == "postgresql://custom/db"

    def test_env_redis_url(self, monkeypatch):
        monkeypatch.setenv("GL_CSR_REDIS_URL", "redis://custom:6380/1")
        cfg = CrossSourceReconciliationConfig.from_env()
        assert cfg.redis_url == "redis://custom:6380/1"

    def test_env_log_level(self, monkeypatch):
        monkeypatch.setenv("GL_CSR_LOG_LEVEL", "DEBUG")
        cfg = CrossSourceReconciliationConfig.from_env()
        assert cfg.log_level == "DEBUG"

    def test_env_resolution_strategy(self, monkeypatch):
        monkeypatch.setenv("GL_CSR_DEFAULT_RESOLUTION_STRATEGY", "consensus")
        cfg = CrossSourceReconciliationConfig.from_env()
        assert cfg.default_resolution_strategy == "consensus"

    def test_env_genesis_hash(self, monkeypatch):
        monkeypatch.setenv("GL_CSR_GENESIS_HASH", "custom-genesis")
        cfg = CrossSourceReconciliationConfig.from_env()
        assert cfg.genesis_hash == "custom-genesis"

    # -- Integer overrides --------------------------------------------------

    def test_env_batch_size(self, monkeypatch):
        monkeypatch.setenv("GL_CSR_BATCH_SIZE", "2000")
        cfg = CrossSourceReconciliationConfig.from_env()
        assert cfg.batch_size == 2000

    def test_env_max_records(self, monkeypatch):
        monkeypatch.setenv("GL_CSR_MAX_RECORDS", "50000")
        cfg = CrossSourceReconciliationConfig.from_env()
        assert cfg.max_records == 50000

    def test_env_max_sources(self, monkeypatch):
        monkeypatch.setenv("GL_CSR_MAX_SOURCES", "10")
        cfg = CrossSourceReconciliationConfig.from_env()
        assert cfg.max_sources == 10

    def test_env_max_match_candidates(self, monkeypatch):
        monkeypatch.setenv("GL_CSR_MAX_MATCH_CANDIDATES", "50")
        cfg = CrossSourceReconciliationConfig.from_env()
        assert cfg.max_match_candidates == 50

    def test_env_max_workers(self, monkeypatch):
        monkeypatch.setenv("GL_CSR_MAX_WORKERS", "8")
        cfg = CrossSourceReconciliationConfig.from_env()
        assert cfg.max_workers == 8

    def test_env_pool_size(self, monkeypatch):
        monkeypatch.setenv("GL_CSR_POOL_SIZE", "10")
        cfg = CrossSourceReconciliationConfig.from_env()
        assert cfg.pool_size == 10

    def test_env_cache_ttl(self, monkeypatch):
        monkeypatch.setenv("GL_CSR_CACHE_TTL", "7200")
        cfg = CrossSourceReconciliationConfig.from_env()
        assert cfg.cache_ttl == 7200

    def test_env_rate_limit(self, monkeypatch):
        monkeypatch.setenv("GL_CSR_RATE_LIMIT", "200")
        cfg = CrossSourceReconciliationConfig.from_env()
        assert cfg.rate_limit == 200

    # -- Float overrides ----------------------------------------------------

    def test_env_default_match_threshold(self, monkeypatch):
        monkeypatch.setenv("GL_CSR_DEFAULT_MATCH_THRESHOLD", "0.9")
        cfg = CrossSourceReconciliationConfig.from_env()
        assert cfg.default_match_threshold == pytest.approx(0.9)

    def test_env_default_tolerance_pct(self, monkeypatch):
        monkeypatch.setenv("GL_CSR_DEFAULT_TOLERANCE_PCT", "10.0")
        cfg = CrossSourceReconciliationConfig.from_env()
        assert cfg.default_tolerance_pct == pytest.approx(10.0)

    def test_env_default_tolerance_abs(self, monkeypatch):
        monkeypatch.setenv("GL_CSR_DEFAULT_TOLERANCE_ABS", "0.05")
        cfg = CrossSourceReconciliationConfig.from_env()
        assert cfg.default_tolerance_abs == pytest.approx(0.05)

    def test_env_source_credibility_weight(self, monkeypatch):
        monkeypatch.setenv("GL_CSR_SOURCE_CREDIBILITY_WEIGHT", "0.7")
        cfg = CrossSourceReconciliationConfig.from_env()
        assert cfg.source_credibility_weight == pytest.approx(0.7)

    def test_env_manual_review_threshold(self, monkeypatch):
        monkeypatch.setenv("GL_CSR_MANUAL_REVIEW_THRESHOLD", "0.5")
        cfg = CrossSourceReconciliationConfig.from_env()
        assert cfg.manual_review_threshold == pytest.approx(0.5)

    def test_env_critical_discrepancy_pct(self, monkeypatch):
        monkeypatch.setenv("GL_CSR_CRITICAL_DISCREPANCY_PCT", "60.0")
        cfg = CrossSourceReconciliationConfig.from_env()
        assert cfg.critical_discrepancy_pct == pytest.approx(60.0)

    def test_env_high_discrepancy_pct(self, monkeypatch):
        monkeypatch.setenv("GL_CSR_HIGH_DISCREPANCY_PCT", "30.0")
        cfg = CrossSourceReconciliationConfig.from_env()
        assert cfg.high_discrepancy_pct == pytest.approx(30.0)

    def test_env_medium_discrepancy_pct(self, monkeypatch):
        monkeypatch.setenv("GL_CSR_MEDIUM_DISCREPANCY_PCT", "5.0")
        cfg = CrossSourceReconciliationConfig.from_env()
        assert cfg.medium_discrepancy_pct == pytest.approx(5.0)

    # -- Boolean overrides --------------------------------------------------

    def test_env_temporal_alignment_enabled_false(self, monkeypatch):
        monkeypatch.setenv("GL_CSR_TEMPORAL_ALIGNMENT_ENABLED", "false")
        cfg = CrossSourceReconciliationConfig.from_env()
        assert cfg.temporal_alignment_enabled is False

    def test_env_temporal_alignment_enabled_true(self, monkeypatch):
        monkeypatch.setenv("GL_CSR_TEMPORAL_ALIGNMENT_ENABLED", "true")
        cfg = CrossSourceReconciliationConfig.from_env()
        assert cfg.temporal_alignment_enabled is True

    def test_env_fuzzy_matching_enabled_false(self, monkeypatch):
        monkeypatch.setenv("GL_CSR_FUZZY_MATCHING_ENABLED", "0")
        cfg = CrossSourceReconciliationConfig.from_env()
        assert cfg.fuzzy_matching_enabled is False

    def test_env_enable_golden_records_false(self, monkeypatch):
        monkeypatch.setenv("GL_CSR_ENABLE_GOLDEN_RECORDS", "no")
        cfg = CrossSourceReconciliationConfig.from_env()
        assert cfg.enable_golden_records is False

    def test_env_enable_provenance_false(self, monkeypatch):
        monkeypatch.setenv("GL_CSR_ENABLE_PROVENANCE", "false")
        cfg = CrossSourceReconciliationConfig.from_env()
        assert cfg.enable_provenance is False

    def test_env_bool_yes_accepted(self, monkeypatch):
        monkeypatch.setenv("GL_CSR_ENABLE_PROVENANCE", "yes")
        cfg = CrossSourceReconciliationConfig.from_env()
        assert cfg.enable_provenance is True

    def test_env_bool_one_accepted(self, monkeypatch):
        monkeypatch.setenv("GL_CSR_ENABLE_PROVENANCE", "1")
        cfg = CrossSourceReconciliationConfig.from_env()
        assert cfg.enable_provenance is True

    def test_env_bool_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("GL_CSR_ENABLE_PROVENANCE", "TRUE")
        cfg = CrossSourceReconciliationConfig.from_env()
        assert cfg.enable_provenance is True


# ======================================================================
# 4. Invalid environment variable fallbacks
# ======================================================================


class TestInvalidEnvFallbacks:
    """Non-parseable env values should fall back to defaults."""

    def test_invalid_int_falls_back(self, monkeypatch):
        monkeypatch.setenv("GL_CSR_BATCH_SIZE", "not_an_int")
        cfg = CrossSourceReconciliationConfig.from_env()
        assert cfg.batch_size == 1000  # default

    def test_invalid_float_falls_back(self, monkeypatch):
        monkeypatch.setenv("GL_CSR_DEFAULT_MATCH_THRESHOLD", "bad_float")
        cfg = CrossSourceReconciliationConfig.from_env()
        assert cfg.default_match_threshold == pytest.approx(0.85)

    def test_invalid_int_for_max_records(self, monkeypatch):
        monkeypatch.setenv("GL_CSR_MAX_RECORDS", "abc")
        cfg = CrossSourceReconciliationConfig.from_env()
        assert cfg.max_records == 100_000

    def test_invalid_float_for_tolerance_pct(self, monkeypatch):
        monkeypatch.setenv("GL_CSR_DEFAULT_TOLERANCE_PCT", "xyz")
        cfg = CrossSourceReconciliationConfig.from_env()
        assert cfg.default_tolerance_pct == pytest.approx(5.0)


# ======================================================================
# 5. Validation: thresholds in range
# ======================================================================


class TestValidationThresholds:
    """Test __post_init__ validation raises ValueError for bad inputs."""

    def test_batch_size_zero_raises(self):
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            CrossSourceReconciliationConfig(batch_size=0)

    def test_batch_size_negative_raises(self):
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            CrossSourceReconciliationConfig(batch_size=-5)

    def test_max_records_zero_raises(self):
        with pytest.raises(ValueError, match="max_records must be >= 1"):
            CrossSourceReconciliationConfig(max_records=0)

    def test_batch_size_exceeds_max_records_raises(self):
        with pytest.raises(ValueError, match="batch_size must be <= max_records"):
            CrossSourceReconciliationConfig(batch_size=200, max_records=100)

    def test_max_sources_one_raises(self):
        with pytest.raises(ValueError, match="max_sources must be >= 2"):
            CrossSourceReconciliationConfig(max_sources=1)

    def test_match_threshold_above_one_raises(self):
        with pytest.raises(ValueError, match="default_match_threshold must be between"):
            CrossSourceReconciliationConfig(default_match_threshold=1.5)

    def test_match_threshold_negative_raises(self):
        with pytest.raises(ValueError, match="default_match_threshold must be between"):
            CrossSourceReconciliationConfig(default_match_threshold=-0.1)

    def test_tolerance_pct_negative_raises(self):
        with pytest.raises(ValueError, match="default_tolerance_pct must be >= 0"):
            CrossSourceReconciliationConfig(default_tolerance_pct=-1.0)

    def test_tolerance_abs_negative_raises(self):
        with pytest.raises(ValueError, match="default_tolerance_abs must be >= 0"):
            CrossSourceReconciliationConfig(default_tolerance_abs=-0.5)

    def test_invalid_resolution_strategy_raises(self):
        with pytest.raises(ValueError, match="default_resolution_strategy must be one of"):
            CrossSourceReconciliationConfig(default_resolution_strategy="invalid_strategy")

    def test_source_credibility_weight_above_one_raises(self):
        with pytest.raises(ValueError, match="source_credibility_weight must be between"):
            CrossSourceReconciliationConfig(source_credibility_weight=1.5)

    def test_source_credibility_weight_negative_raises(self):
        with pytest.raises(ValueError, match="source_credibility_weight must be between"):
            CrossSourceReconciliationConfig(source_credibility_weight=-0.1)

    def test_max_match_candidates_zero_raises(self):
        with pytest.raises(ValueError, match="max_match_candidates must be >= 1"):
            CrossSourceReconciliationConfig(max_match_candidates=0)

    def test_max_workers_zero_raises(self):
        with pytest.raises(ValueError, match="max_workers must be >= 1"):
            CrossSourceReconciliationConfig(max_workers=0)

    def test_pool_size_zero_raises(self):
        with pytest.raises(ValueError, match="pool_size must be >= 1"):
            CrossSourceReconciliationConfig(pool_size=0)

    def test_cache_ttl_negative_raises(self):
        with pytest.raises(ValueError, match="cache_ttl must be >= 0"):
            CrossSourceReconciliationConfig(cache_ttl=-1)

    def test_rate_limit_zero_raises(self):
        with pytest.raises(ValueError, match="rate_limit must be >= 1"):
            CrossSourceReconciliationConfig(rate_limit=0)

    def test_manual_review_threshold_above_one_raises(self):
        with pytest.raises(ValueError, match="manual_review_threshold must be between"):
            CrossSourceReconciliationConfig(manual_review_threshold=1.5)

    def test_manual_review_threshold_negative_raises(self):
        with pytest.raises(ValueError, match="manual_review_threshold must be between"):
            CrossSourceReconciliationConfig(manual_review_threshold=-0.1)

    def test_manual_review_exceeds_match_threshold_raises(self):
        with pytest.raises(ValueError, match="manual_review_threshold must be <="):
            CrossSourceReconciliationConfig(
                default_match_threshold=0.7,
                manual_review_threshold=0.8,
            )


# ======================================================================
# 6. Validation: discrepancy severity ordering
# ======================================================================


class TestValidationSeverityOrdering:
    """Discrepancy thresholds must be ordered: medium < high < critical."""

    def test_medium_equals_high_raises(self):
        with pytest.raises(ValueError, match="medium_discrepancy_pct must be < high_discrepancy_pct"):
            CrossSourceReconciliationConfig(
                medium_discrepancy_pct=25.0,
                high_discrepancy_pct=25.0,
            )

    def test_medium_exceeds_high_raises(self):
        with pytest.raises(ValueError, match="medium_discrepancy_pct must be < high_discrepancy_pct"):
            CrossSourceReconciliationConfig(
                medium_discrepancy_pct=30.0,
                high_discrepancy_pct=25.0,
            )

    def test_high_equals_critical_raises(self):
        with pytest.raises(ValueError, match="high_discrepancy_pct must be < critical_discrepancy_pct"):
            CrossSourceReconciliationConfig(
                high_discrepancy_pct=50.0,
                critical_discrepancy_pct=50.0,
            )

    def test_high_exceeds_critical_raises(self):
        with pytest.raises(ValueError, match="high_discrepancy_pct must be < critical_discrepancy_pct"):
            CrossSourceReconciliationConfig(
                high_discrepancy_pct=60.0,
                critical_discrepancy_pct=50.0,
            )

    def test_negative_medium_raises(self):
        with pytest.raises(ValueError, match="medium_discrepancy_pct must be >= 0"):
            CrossSourceReconciliationConfig(medium_discrepancy_pct=-1.0)

    def test_negative_high_raises(self):
        with pytest.raises(ValueError, match="high_discrepancy_pct must be >= 0"):
            CrossSourceReconciliationConfig(high_discrepancy_pct=-1.0)

    def test_negative_critical_raises(self):
        with pytest.raises(ValueError, match="critical_discrepancy_pct must be >= 0"):
            CrossSourceReconciliationConfig(critical_discrepancy_pct=-1.0)

    def test_valid_severity_ordering_accepted(self):
        cfg = CrossSourceReconciliationConfig(
            medium_discrepancy_pct=5.0,
            high_discrepancy_pct=20.0,
            critical_discrepancy_pct=40.0,
        )
        assert cfg.medium_discrepancy_pct < cfg.high_discrepancy_pct
        assert cfg.high_discrepancy_pct < cfg.critical_discrepancy_pct


# ======================================================================
# 7. Validation: log level and genesis hash
# ======================================================================


class TestValidationMisc:
    """Test log level validation and genesis hash validation."""

    def test_invalid_log_level_raises(self):
        with pytest.raises(ValueError, match="log_level must be one of"):
            CrossSourceReconciliationConfig(log_level="TRACE")

    def test_empty_genesis_hash_raises(self):
        with pytest.raises(ValueError, match="genesis_hash must not be empty"):
            CrossSourceReconciliationConfig(genesis_hash="")

    def test_valid_log_levels_accepted(self):
        for level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            cfg = CrossSourceReconciliationConfig(log_level=level)
            assert cfg.log_level == level

    def test_valid_resolution_strategies_accepted(self):
        strategies = [
            "priority_wins", "most_recent_wins", "average",
            "median", "manual_review", "consensus",
        ]
        for strategy in strategies:
            cfg = CrossSourceReconciliationConfig(
                default_resolution_strategy=strategy,
            )
            assert cfg.default_resolution_strategy == strategy


# ======================================================================
# 8. Singleton pattern: get_config / set_config / reset_config
# ======================================================================


class TestSingletonPattern:
    """Test thread-safe singleton accessor functions."""

    def test_get_config_returns_instance(self):
        cfg = get_config()
        assert isinstance(cfg, CrossSourceReconciliationConfig)

    def test_get_config_returns_same_instance(self):
        a = get_config()
        b = get_config()
        assert a is b

    def test_set_config_replaces_singleton(self):
        custom = CrossSourceReconciliationConfig(batch_size=42)
        set_config(custom)
        assert get_config() is custom
        assert get_config().batch_size == 42

    def test_reset_config_clears_singleton(self):
        first = get_config()
        reset_config()
        second = get_config()
        # After reset, a new instance is created
        assert first is not second

    def test_reset_then_get_returns_defaults(self):
        set_config(CrossSourceReconciliationConfig(batch_size=42))
        reset_config()
        cfg = get_config()
        assert cfg.batch_size == 1000  # default


# ======================================================================
# 9. Thread safety
# ======================================================================


class TestThreadSafety:
    """Test that concurrent get_config calls return the same singleton."""

    def test_concurrent_get_config_returns_same_instance(self):
        reset_config()
        results = []
        barrier = threading.Barrier(8)

        def _get():
            barrier.wait()
            results.append(id(get_config()))

        threads = [threading.Thread(target=_get) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(set(results)) == 1  # all same id

    def test_concurrent_get_config_with_thread_pool(self):
        reset_config()
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(get_config) for _ in range(16)]
            configs = [f.result() for f in futures]
        assert all(c is configs[0] for c in configs)


# ======================================================================
# 10. Edge cases and boundary values
# ======================================================================


class TestEdgeCases:
    """Test boundary values that should be accepted."""

    def test_batch_size_one_accepted(self):
        cfg = CrossSourceReconciliationConfig(batch_size=1, max_records=1)
        assert cfg.batch_size == 1

    def test_max_sources_two_accepted(self):
        cfg = CrossSourceReconciliationConfig(max_sources=2)
        assert cfg.max_sources == 2

    def test_match_threshold_zero_accepted(self):
        cfg = CrossSourceReconciliationConfig(
            default_match_threshold=0.0,
            manual_review_threshold=0.0,
        )
        assert cfg.default_match_threshold == 0.0

    def test_match_threshold_one_accepted(self):
        cfg = CrossSourceReconciliationConfig(
            default_match_threshold=1.0,
            manual_review_threshold=0.6,
        )
        assert cfg.default_match_threshold == 1.0

    def test_cache_ttl_zero_accepted(self):
        cfg = CrossSourceReconciliationConfig(cache_ttl=0)
        assert cfg.cache_ttl == 0

    def test_tolerance_zero_accepted(self):
        cfg = CrossSourceReconciliationConfig(
            default_tolerance_pct=0.0,
            default_tolerance_abs=0.0,
        )
        assert cfg.default_tolerance_pct == 0.0
        assert cfg.default_tolerance_abs == 0.0

    def test_manual_review_equals_match_threshold_accepted(self):
        cfg = CrossSourceReconciliationConfig(
            default_match_threshold=0.8,
            manual_review_threshold=0.8,
        )
        assert cfg.manual_review_threshold == cfg.default_match_threshold

    def test_multiple_validation_errors_aggregated(self):
        """Multiple violations produce a combined error message."""
        with pytest.raises(ValueError) as exc_info:
            CrossSourceReconciliationConfig(
                batch_size=0,
                max_records=0,
                max_sources=1,
            )
        msg = str(exc_info.value)
        assert "batch_size" in msg
        assert "max_records" in msg
        assert "max_sources" in msg
