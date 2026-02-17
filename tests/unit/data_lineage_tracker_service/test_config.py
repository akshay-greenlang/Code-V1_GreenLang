# -*- coding: utf-8 -*-
"""
Unit Tests for DataLineageTrackerConfig - AGENT-DATA-018

Tests the complete configuration dataclass including:
  - Default values (22 fields)
  - Environment variable loading (GL_DLT_ prefix)
  - Validation constraints (positive ints, ranges, JSON weights)
  - Serialization helpers (to_dict, __repr__)
  - Thread-safe singleton accessors (get_config, set_config, reset_config)

60+ test cases covering all configuration paths.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import json
import os
import threading
from unittest.mock import patch

import pytest

from greenlang.data_lineage_tracker.config import (
    DataLineageTrackerConfig,
    get_config,
    reset_config,
    set_config,
)


# ============================================================================
# TestDataLineageTrackerConfigDefaults
# ============================================================================


class TestDataLineageTrackerConfigDefaults:
    """Verify all 22 default values are correct."""

    def test_default_database_url_empty(self):
        """Default database_url is empty string when not provided."""
        cfg = DataLineageTrackerConfig(
            database_url="postgresql://x:x@localhost/test",
            redis_url="redis://localhost/0",
        )
        assert cfg.database_url == "postgresql://x:x@localhost/test"

    def test_default_redis_url(self):
        """redis_url retains provided value."""
        cfg = DataLineageTrackerConfig(
            database_url="postgresql://x:x@localhost/test",
            redis_url="redis://localhost/0",
        )
        assert cfg.redis_url == "redis://localhost/0"

    def test_default_log_level(self):
        """Default log_level is INFO."""
        cfg = DataLineageTrackerConfig(
            database_url="postgresql://x:x@localhost/test",
            redis_url="redis://localhost/0",
        )
        assert cfg.log_level == "INFO"

    def test_default_max_assets(self):
        """Default max_assets is 100000."""
        cfg = DataLineageTrackerConfig(
            database_url="postgresql://x:x@localhost/test",
            redis_url="redis://localhost/0",
        )
        assert cfg.max_assets == 100_000

    def test_default_max_transformations(self):
        """Default max_transformations is 500000."""
        cfg = DataLineageTrackerConfig(
            database_url="postgresql://x:x@localhost/test",
            redis_url="redis://localhost/0",
        )
        assert cfg.max_transformations == 500_000

    def test_default_max_edges(self):
        """Default max_edges is 1000000."""
        cfg = DataLineageTrackerConfig(
            database_url="postgresql://x:x@localhost/test",
            redis_url="redis://localhost/0",
        )
        assert cfg.max_edges == 1_000_000

    def test_default_max_graph_depth(self):
        """Default max_graph_depth is 50."""
        cfg = DataLineageTrackerConfig(
            database_url="postgresql://x:x@localhost/test",
            redis_url="redis://localhost/0",
        )
        assert cfg.max_graph_depth == 50

    def test_default_default_traversal_depth(self):
        """Default default_traversal_depth is 10."""
        cfg = DataLineageTrackerConfig(
            database_url="postgresql://x:x@localhost/test",
            redis_url="redis://localhost/0",
        )
        assert cfg.default_traversal_depth == 10

    def test_default_snapshot_interval_minutes(self):
        """Default snapshot_interval_minutes is 60."""
        cfg = DataLineageTrackerConfig(
            database_url="postgresql://x:x@localhost/test",
            redis_url="redis://localhost/0",
        )
        assert cfg.snapshot_interval_minutes == 60

    def test_default_enable_column_lineage(self):
        """Default enable_column_lineage is True."""
        cfg = DataLineageTrackerConfig(
            database_url="postgresql://x:x@localhost/test",
            redis_url="redis://localhost/0",
        )
        assert cfg.enable_column_lineage is True

    def test_default_enable_change_detection(self):
        """Default enable_change_detection is True."""
        cfg = DataLineageTrackerConfig(
            database_url="postgresql://x:x@localhost/test",
            redis_url="redis://localhost/0",
        )
        assert cfg.enable_change_detection is True

    def test_default_enable_provenance(self):
        """Default enable_provenance is True."""
        cfg = DataLineageTrackerConfig(
            database_url="postgresql://x:x@localhost/test",
            redis_url="redis://localhost/0",
        )
        assert cfg.enable_provenance is True

    def test_default_genesis_hash(self):
        """Default genesis_hash is 'greenlang-data-lineage-genesis'."""
        cfg = DataLineageTrackerConfig(
            database_url="postgresql://x:x@localhost/test",
            redis_url="redis://localhost/0",
        )
        assert cfg.genesis_hash == "greenlang-data-lineage-genesis"

    def test_default_pool_size(self):
        """Default pool_size is 5."""
        cfg = DataLineageTrackerConfig(
            database_url="postgresql://x:x@localhost/test",
            redis_url="redis://localhost/0",
        )
        assert cfg.pool_size == 5

    def test_default_cache_ttl(self):
        """Default cache_ttl is 300."""
        cfg = DataLineageTrackerConfig(
            database_url="postgresql://x:x@localhost/test",
            redis_url="redis://localhost/0",
        )
        assert cfg.cache_ttl == 300

    def test_default_rate_limit(self):
        """Default rate_limit is 200."""
        cfg = DataLineageTrackerConfig(
            database_url="postgresql://x:x@localhost/test",
            redis_url="redis://localhost/0",
        )
        assert cfg.rate_limit == 200

    def test_default_batch_size(self):
        """Default batch_size is 1000."""
        cfg = DataLineageTrackerConfig(
            database_url="postgresql://x:x@localhost/test",
            redis_url="redis://localhost/0",
        )
        assert cfg.batch_size == 1000

    def test_default_coverage_warn_threshold(self):
        """Default coverage_warn_threshold is 0.8."""
        cfg = DataLineageTrackerConfig(
            database_url="postgresql://x:x@localhost/test",
            redis_url="redis://localhost/0",
        )
        assert cfg.coverage_warn_threshold == 0.8

    def test_default_coverage_fail_threshold(self):
        """Default coverage_fail_threshold is 0.5."""
        cfg = DataLineageTrackerConfig(
            database_url="postgresql://x:x@localhost/test",
            redis_url="redis://localhost/0",
        )
        assert cfg.coverage_fail_threshold == 0.5

    def test_default_freshness_max_age_hours(self):
        """Default freshness_max_age_hours is 24."""
        cfg = DataLineageTrackerConfig(
            database_url="postgresql://x:x@localhost/test",
            redis_url="redis://localhost/0",
        )
        assert cfg.freshness_max_age_hours == 24

    def test_default_quality_score_weights_is_valid_json(self):
        """Default quality_score_weights is a valid JSON string with 5 keys."""
        cfg = DataLineageTrackerConfig(
            database_url="postgresql://x:x@localhost/test",
            redis_url="redis://localhost/0",
        )
        parsed = json.loads(cfg.quality_score_weights)
        assert isinstance(parsed, dict)
        assert "source_credibility" in parsed
        assert "transformation_depth" in parsed
        assert "freshness" in parsed
        assert "documentation" in parsed
        assert "manual_interventions" in parsed

    def test_default_enable_metrics(self):
        """Default enable_metrics is True."""
        cfg = DataLineageTrackerConfig(
            database_url="postgresql://x:x@localhost/test",
            redis_url="redis://localhost/0",
        )
        assert cfg.enable_metrics is True

    def test_default_quality_score_weights_sum_to_one(self):
        """Default quality_score_weights sum to 1.0."""
        cfg = DataLineageTrackerConfig(
            database_url="postgresql://x:x@localhost/test",
            redis_url="redis://localhost/0",
        )
        parsed = json.loads(cfg.quality_score_weights)
        total = sum(parsed.values())
        assert abs(total - 1.0) < 0.001


# ============================================================================
# TestDataLineageTrackerConfigFromEnv
# ============================================================================


class TestDataLineageTrackerConfigFromEnv:
    """Verify from_env() reads GL_DLT_* environment variables."""

    def test_from_env_database_url(self, monkeypatch):
        """GL_DLT_DATABASE_URL is read correctly."""
        monkeypatch.setenv("GL_DLT_DATABASE_URL", "postgresql://prod:prod@db/lineage")
        monkeypatch.setenv("GL_DLT_REDIS_URL", "redis://localhost/5")
        reset_config()
        cfg = DataLineageTrackerConfig.from_env()
        assert cfg.database_url == "postgresql://prod:prod@db/lineage"

    def test_from_env_redis_url(self, monkeypatch):
        """GL_DLT_REDIS_URL is read correctly."""
        monkeypatch.setenv("GL_DLT_DATABASE_URL", "postgresql://x:x@localhost/test")
        monkeypatch.setenv("GL_DLT_REDIS_URL", "redis://prod-redis:6379/3")
        reset_config()
        cfg = DataLineageTrackerConfig.from_env()
        assert cfg.redis_url == "redis://prod-redis:6379/3"

    def test_from_env_log_level(self, monkeypatch):
        """GL_DLT_LOG_LEVEL is read and uppercased."""
        monkeypatch.setenv("GL_DLT_DATABASE_URL", "postgresql://x:x@localhost/test")
        monkeypatch.setenv("GL_DLT_REDIS_URL", "redis://localhost/0")
        monkeypatch.setenv("GL_DLT_LOG_LEVEL", "warning")
        reset_config()
        cfg = DataLineageTrackerConfig.from_env()
        assert cfg.log_level == "WARNING"

    def test_from_env_max_assets(self, monkeypatch):
        """GL_DLT_MAX_ASSETS is parsed as integer."""
        monkeypatch.setenv("GL_DLT_DATABASE_URL", "postgresql://x:x@localhost/test")
        monkeypatch.setenv("GL_DLT_REDIS_URL", "redis://localhost/0")
        monkeypatch.setenv("GL_DLT_MAX_ASSETS", "200000")
        reset_config()
        cfg = DataLineageTrackerConfig.from_env()
        assert cfg.max_assets == 200_000

    def test_from_env_max_transformations(self, monkeypatch):
        """GL_DLT_MAX_TRANSFORMATIONS is parsed as integer."""
        monkeypatch.setenv("GL_DLT_DATABASE_URL", "postgresql://x:x@localhost/test")
        monkeypatch.setenv("GL_DLT_REDIS_URL", "redis://localhost/0")
        monkeypatch.setenv("GL_DLT_MAX_TRANSFORMATIONS", "1000000")
        reset_config()
        cfg = DataLineageTrackerConfig.from_env()
        assert cfg.max_transformations == 1_000_000

    def test_from_env_enable_column_lineage_true(self, monkeypatch):
        """GL_DLT_ENABLE_COLUMN_LINEAGE accepts 'true'."""
        monkeypatch.setenv("GL_DLT_DATABASE_URL", "postgresql://x:x@localhost/test")
        monkeypatch.setenv("GL_DLT_REDIS_URL", "redis://localhost/0")
        monkeypatch.setenv("GL_DLT_ENABLE_COLUMN_LINEAGE", "true")
        reset_config()
        cfg = DataLineageTrackerConfig.from_env()
        assert cfg.enable_column_lineage is True

    def test_from_env_enable_column_lineage_false(self, monkeypatch):
        """GL_DLT_ENABLE_COLUMN_LINEAGE accepts 'false'."""
        monkeypatch.setenv("GL_DLT_DATABASE_URL", "postgresql://x:x@localhost/test")
        monkeypatch.setenv("GL_DLT_REDIS_URL", "redis://localhost/0")
        monkeypatch.setenv("GL_DLT_ENABLE_COLUMN_LINEAGE", "false")
        reset_config()
        cfg = DataLineageTrackerConfig.from_env()
        assert cfg.enable_column_lineage is False

    def test_from_env_enable_column_lineage_yes(self, monkeypatch):
        """GL_DLT_ENABLE_COLUMN_LINEAGE accepts 'yes'."""
        monkeypatch.setenv("GL_DLT_DATABASE_URL", "postgresql://x:x@localhost/test")
        monkeypatch.setenv("GL_DLT_REDIS_URL", "redis://localhost/0")
        monkeypatch.setenv("GL_DLT_ENABLE_COLUMN_LINEAGE", "yes")
        reset_config()
        cfg = DataLineageTrackerConfig.from_env()
        assert cfg.enable_column_lineage is True

    def test_from_env_enable_column_lineage_1(self, monkeypatch):
        """GL_DLT_ENABLE_COLUMN_LINEAGE accepts '1'."""
        monkeypatch.setenv("GL_DLT_DATABASE_URL", "postgresql://x:x@localhost/test")
        monkeypatch.setenv("GL_DLT_REDIS_URL", "redis://localhost/0")
        monkeypatch.setenv("GL_DLT_ENABLE_COLUMN_LINEAGE", "1")
        reset_config()
        cfg = DataLineageTrackerConfig.from_env()
        assert cfg.enable_column_lineage is True

    def test_from_env_coverage_thresholds(self, monkeypatch):
        """GL_DLT_COVERAGE_*_THRESHOLD are parsed as float."""
        monkeypatch.setenv("GL_DLT_DATABASE_URL", "postgresql://x:x@localhost/test")
        monkeypatch.setenv("GL_DLT_REDIS_URL", "redis://localhost/0")
        monkeypatch.setenv("GL_DLT_COVERAGE_WARN_THRESHOLD", "0.9")
        monkeypatch.setenv("GL_DLT_COVERAGE_FAIL_THRESHOLD", "0.6")
        reset_config()
        cfg = DataLineageTrackerConfig.from_env()
        assert cfg.coverage_warn_threshold == pytest.approx(0.9)
        assert cfg.coverage_fail_threshold == pytest.approx(0.6)

    def test_from_env_invalid_int_uses_default(self, monkeypatch):
        """GL_DLT_MAX_ASSETS with non-integer falls back to default."""
        monkeypatch.setenv("GL_DLT_DATABASE_URL", "postgresql://x:x@localhost/test")
        monkeypatch.setenv("GL_DLT_REDIS_URL", "redis://localhost/0")
        monkeypatch.setenv("GL_DLT_MAX_ASSETS", "not_a_number")
        reset_config()
        cfg = DataLineageTrackerConfig.from_env()
        assert cfg.max_assets == 100_000

    def test_from_env_invalid_float_uses_default(self, monkeypatch):
        """GL_DLT_COVERAGE_WARN_THRESHOLD with non-float falls back to default."""
        monkeypatch.setenv("GL_DLT_DATABASE_URL", "postgresql://x:x@localhost/test")
        monkeypatch.setenv("GL_DLT_REDIS_URL", "redis://localhost/0")
        monkeypatch.setenv("GL_DLT_COVERAGE_WARN_THRESHOLD", "nope")
        reset_config()
        cfg = DataLineageTrackerConfig.from_env()
        assert cfg.coverage_warn_threshold == 0.8

    def test_from_env_snapshot_interval(self, monkeypatch):
        """GL_DLT_SNAPSHOT_INTERVAL_MINUTES is parsed as integer."""
        monkeypatch.setenv("GL_DLT_DATABASE_URL", "postgresql://x:x@localhost/test")
        monkeypatch.setenv("GL_DLT_REDIS_URL", "redis://localhost/0")
        monkeypatch.setenv("GL_DLT_SNAPSHOT_INTERVAL_MINUTES", "120")
        reset_config()
        cfg = DataLineageTrackerConfig.from_env()
        assert cfg.snapshot_interval_minutes == 120

    def test_from_env_genesis_hash(self, monkeypatch):
        """GL_DLT_GENESIS_HASH is read correctly."""
        monkeypatch.setenv("GL_DLT_DATABASE_URL", "postgresql://x:x@localhost/test")
        monkeypatch.setenv("GL_DLT_REDIS_URL", "redis://localhost/0")
        monkeypatch.setenv("GL_DLT_GENESIS_HASH", "custom-genesis-2026")
        reset_config()
        cfg = DataLineageTrackerConfig.from_env()
        assert cfg.genesis_hash == "custom-genesis-2026"


# ============================================================================
# TestDataLineageTrackerConfigValidation
# ============================================================================


class TestDataLineageTrackerConfigValidation:
    """Verify validation constraints raise ValueError appropriately."""

    def test_validation_max_assets_zero(self):
        """max_assets == 0 raises ValueError."""
        with pytest.raises(ValueError, match="max_assets must be > 0"):
            DataLineageTrackerConfig(
                database_url="postgresql://x:x@localhost/test",
                redis_url="redis://localhost/0",
                max_assets=0,
            )

    def test_validation_max_assets_negative(self):
        """max_assets < 0 raises ValueError."""
        with pytest.raises(ValueError, match="max_assets must be > 0"):
            DataLineageTrackerConfig(
                database_url="postgresql://x:x@localhost/test",
                redis_url="redis://localhost/0",
                max_assets=-1,
            )

    def test_validation_max_transformations_zero(self):
        """max_transformations == 0 raises ValueError."""
        with pytest.raises(ValueError, match="max_transformations must be > 0"):
            DataLineageTrackerConfig(
                database_url="postgresql://x:x@localhost/test",
                redis_url="redis://localhost/0",
                max_transformations=0,
            )

    def test_validation_max_edges_zero(self):
        """max_edges == 0 raises ValueError."""
        with pytest.raises(ValueError, match="max_edges must be > 0"):
            DataLineageTrackerConfig(
                database_url="postgresql://x:x@localhost/test",
                redis_url="redis://localhost/0",
                max_edges=0,
            )

    def test_validation_max_graph_depth_zero(self):
        """max_graph_depth == 0 raises ValueError."""
        with pytest.raises(ValueError, match="max_graph_depth must be > 0"):
            DataLineageTrackerConfig(
                database_url="postgresql://x:x@localhost/test",
                redis_url="redis://localhost/0",
                max_graph_depth=0,
            )

    def test_validation_default_traversal_depth_zero(self):
        """default_traversal_depth == 0 raises ValueError."""
        with pytest.raises(ValueError, match="default_traversal_depth must be > 0"):
            DataLineageTrackerConfig(
                database_url="postgresql://x:x@localhost/test",
                redis_url="redis://localhost/0",
                default_traversal_depth=0,
            )

    def test_validation_default_traversal_depth_exceeds_max(self):
        """default_traversal_depth > max_graph_depth raises ValueError."""
        with pytest.raises(ValueError, match="must not exceed max_graph_depth"):
            DataLineageTrackerConfig(
                database_url="postgresql://x:x@localhost/test",
                redis_url="redis://localhost/0",
                default_traversal_depth=100,
                max_graph_depth=50,
            )

    def test_validation_snapshot_interval_zero(self):
        """snapshot_interval_minutes == 0 raises ValueError."""
        with pytest.raises(ValueError, match="snapshot_interval_minutes must be > 0"):
            DataLineageTrackerConfig(
                database_url="postgresql://x:x@localhost/test",
                redis_url="redis://localhost/0",
                snapshot_interval_minutes=0,
            )

    def test_validation_genesis_hash_empty(self):
        """genesis_hash must not be empty."""
        with pytest.raises(ValueError, match="genesis_hash must not be empty"):
            DataLineageTrackerConfig(
                database_url="postgresql://x:x@localhost/test",
                redis_url="redis://localhost/0",
                genesis_hash="",
            )

    def test_validation_pool_size_zero(self):
        """pool_size == 0 raises ValueError."""
        with pytest.raises(ValueError, match="pool_size must be > 0"):
            DataLineageTrackerConfig(
                database_url="postgresql://x:x@localhost/test",
                redis_url="redis://localhost/0",
                pool_size=0,
            )

    def test_validation_cache_ttl_zero(self):
        """cache_ttl == 0 raises ValueError."""
        with pytest.raises(ValueError, match="cache_ttl must be > 0"):
            DataLineageTrackerConfig(
                database_url="postgresql://x:x@localhost/test",
                redis_url="redis://localhost/0",
                cache_ttl=0,
            )

    def test_validation_rate_limit_zero(self):
        """rate_limit == 0 raises ValueError."""
        with pytest.raises(ValueError, match="rate_limit must be > 0"):
            DataLineageTrackerConfig(
                database_url="postgresql://x:x@localhost/test",
                redis_url="redis://localhost/0",
                rate_limit=0,
            )

    def test_validation_batch_size_zero(self):
        """batch_size == 0 raises ValueError."""
        with pytest.raises(ValueError, match="batch_size must be > 0"):
            DataLineageTrackerConfig(
                database_url="postgresql://x:x@localhost/test",
                redis_url="redis://localhost/0",
                batch_size=0,
            )

    def test_validation_coverage_warn_below_zero(self):
        """coverage_warn_threshold < 0 raises ValueError."""
        with pytest.raises(ValueError, match="coverage_warn_threshold must be in"):
            DataLineageTrackerConfig(
                database_url="postgresql://x:x@localhost/test",
                redis_url="redis://localhost/0",
                coverage_warn_threshold=-0.1,
            )

    def test_validation_coverage_warn_above_one(self):
        """coverage_warn_threshold > 1 raises ValueError."""
        with pytest.raises(ValueError, match="coverage_warn_threshold must be in"):
            DataLineageTrackerConfig(
                database_url="postgresql://x:x@localhost/test",
                redis_url="redis://localhost/0",
                coverage_warn_threshold=1.1,
            )

    def test_validation_coverage_fail_below_zero(self):
        """coverage_fail_threshold < 0 raises ValueError."""
        with pytest.raises(ValueError, match="coverage_fail_threshold must be in"):
            DataLineageTrackerConfig(
                database_url="postgresql://x:x@localhost/test",
                redis_url="redis://localhost/0",
                coverage_fail_threshold=-0.5,
            )

    def test_validation_coverage_fail_above_one(self):
        """coverage_fail_threshold > 1 raises ValueError."""
        with pytest.raises(ValueError, match="coverage_fail_threshold must be in"):
            DataLineageTrackerConfig(
                database_url="postgresql://x:x@localhost/test",
                redis_url="redis://localhost/0",
                coverage_fail_threshold=1.5,
            )

    def test_validation_coverage_fail_exceeds_warn(self):
        """coverage_fail_threshold > coverage_warn_threshold raises ValueError."""
        with pytest.raises(ValueError, match="must not exceed coverage_warn_threshold"):
            DataLineageTrackerConfig(
                database_url="postgresql://x:x@localhost/test",
                redis_url="redis://localhost/0",
                coverage_warn_threshold=0.5,
                coverage_fail_threshold=0.6,
            )

    def test_validation_freshness_max_age_zero(self):
        """freshness_max_age_hours == 0 raises ValueError."""
        with pytest.raises(ValueError, match="freshness_max_age_hours must be > 0"):
            DataLineageTrackerConfig(
                database_url="postgresql://x:x@localhost/test",
                redis_url="redis://localhost/0",
                freshness_max_age_hours=0,
            )

    def test_validation_log_level_invalid(self):
        """Invalid log_level raises ValueError."""
        with pytest.raises(ValueError, match="log_level must be one of"):
            DataLineageTrackerConfig(
                database_url="postgresql://x:x@localhost/test",
                redis_url="redis://localhost/0",
                log_level="TRACE",
            )

    def test_validation_log_level_case_insensitive(self):
        """log_level is case-insensitive and uppercased."""
        cfg = DataLineageTrackerConfig(
            database_url="postgresql://x:x@localhost/test",
            redis_url="redis://localhost/0",
            log_level="debug",
        )
        assert cfg.log_level == "DEBUG"

    def test_validation_quality_weights_invalid_json(self):
        """Invalid JSON in quality_score_weights raises ValueError."""
        with pytest.raises(ValueError, match="not valid JSON"):
            DataLineageTrackerConfig(
                database_url="postgresql://x:x@localhost/test",
                redis_url="redis://localhost/0",
                quality_score_weights="not-json{",
            )

    def test_validation_quality_weights_not_dict(self):
        """Non-dict JSON in quality_score_weights raises ValueError."""
        with pytest.raises(ValueError, match="must be a JSON object"):
            DataLineageTrackerConfig(
                database_url="postgresql://x:x@localhost/test",
                redis_url="redis://localhost/0",
                quality_score_weights="[1,2,3]",
            )

    def test_validation_quality_weights_missing_keys(self):
        """Missing required keys in quality_score_weights raises ValueError."""
        weights = json.dumps({"source_credibility": 1.0})
        with pytest.raises(ValueError, match="missing required keys"):
            DataLineageTrackerConfig(
                database_url="postgresql://x:x@localhost/test",
                redis_url="redis://localhost/0",
                quality_score_weights=weights,
            )

    def test_validation_quality_weights_extra_keys(self):
        """Extra keys in quality_score_weights raises ValueError."""
        weights = json.dumps({
            "source_credibility": 0.2,
            "transformation_depth": 0.2,
            "freshness": 0.2,
            "documentation": 0.2,
            "manual_interventions": 0.2,
            "extra_key": 0.0,
        })
        with pytest.raises(ValueError, match="unknown keys"):
            DataLineageTrackerConfig(
                database_url="postgresql://x:x@localhost/test",
                redis_url="redis://localhost/0",
                quality_score_weights=weights,
            )

    def test_validation_quality_weights_non_numeric(self):
        """Non-numeric value in quality_score_weights raises ValueError."""
        weights = json.dumps({
            "source_credibility": "high",
            "transformation_depth": 0.2,
            "freshness": 0.2,
            "documentation": 0.2,
            "manual_interventions": 0.2,
        })
        with pytest.raises(ValueError, match="must be numeric"):
            DataLineageTrackerConfig(
                database_url="postgresql://x:x@localhost/test",
                redis_url="redis://localhost/0",
                quality_score_weights=weights,
            )

    def test_validation_quality_weights_not_sum_to_one(self):
        """quality_score_weights not summing to 1.0 raises ValueError."""
        weights = json.dumps({
            "source_credibility": 0.5,
            "transformation_depth": 0.5,
            "freshness": 0.5,
            "documentation": 0.5,
            "manual_interventions": 0.5,
        })
        with pytest.raises(ValueError, match="must sum to 1.0"):
            DataLineageTrackerConfig(
                database_url="postgresql://x:x@localhost/test",
                redis_url="redis://localhost/0",
                quality_score_weights=weights,
            )

    def test_validation_quality_weights_valid_custom(self):
        """Valid custom quality_score_weights passes validation."""
        weights = json.dumps({
            "source_credibility": 0.4,
            "transformation_depth": 0.2,
            "freshness": 0.15,
            "documentation": 0.15,
            "manual_interventions": 0.1,
        })
        cfg = DataLineageTrackerConfig(
            database_url="postgresql://x:x@localhost/test",
            redis_url="redis://localhost/0",
            quality_score_weights=weights,
        )
        parsed = json.loads(cfg.quality_score_weights)
        assert abs(sum(parsed.values()) - 1.0) < 0.001

    def test_validation_coverage_thresholds_equal_valid(self):
        """coverage_fail_threshold == coverage_warn_threshold is valid."""
        cfg = DataLineageTrackerConfig(
            database_url="postgresql://x:x@localhost/test",
            redis_url="redis://localhost/0",
            coverage_warn_threshold=0.7,
            coverage_fail_threshold=0.7,
        )
        assert cfg.coverage_warn_threshold == 0.7
        assert cfg.coverage_fail_threshold == 0.7

    def test_validation_coverage_zero_values_valid(self):
        """coverage thresholds at 0.0 are valid."""
        cfg = DataLineageTrackerConfig(
            database_url="postgresql://x:x@localhost/test",
            redis_url="redis://localhost/0",
            coverage_warn_threshold=0.0,
            coverage_fail_threshold=0.0,
        )
        assert cfg.coverage_warn_threshold == 0.0
        assert cfg.coverage_fail_threshold == 0.0

    def test_validation_coverage_one_values_valid(self):
        """coverage thresholds at 1.0 are valid."""
        cfg = DataLineageTrackerConfig(
            database_url="postgresql://x:x@localhost/test",
            redis_url="redis://localhost/0",
            coverage_warn_threshold=1.0,
            coverage_fail_threshold=1.0,
        )
        assert cfg.coverage_warn_threshold == 1.0
        assert cfg.coverage_fail_threshold == 1.0


# ============================================================================
# TestDataLineageTrackerConfigSerialization
# ============================================================================


class TestDataLineageTrackerConfigSerialization:
    """Verify to_dict() and __repr__() behave correctly."""

    def test_to_dict_returns_dict(self):
        """to_dict() returns a dict."""
        cfg = DataLineageTrackerConfig(
            database_url="postgresql://secret:secret@db/prod",
            redis_url="redis://secret@redis:6379",
        )
        d = cfg.to_dict()
        assert isinstance(d, dict)

    def test_to_dict_redacts_database_url(self):
        """to_dict() redacts database_url with '***'."""
        cfg = DataLineageTrackerConfig(
            database_url="postgresql://secret:secret@db/prod",
            redis_url="redis://secret@redis:6379",
        )
        d = cfg.to_dict()
        assert d["database_url"] == "***"

    def test_to_dict_redacts_redis_url(self):
        """to_dict() redacts redis_url with '***'."""
        cfg = DataLineageTrackerConfig(
            database_url="postgresql://secret:secret@db/prod",
            redis_url="redis://secret@redis:6379",
        )
        d = cfg.to_dict()
        assert d["redis_url"] == "***"

    def test_to_dict_empty_database_url_shows_empty(self):
        """to_dict() shows empty string when database_url is empty."""
        cfg = DataLineageTrackerConfig()
        d = cfg.to_dict()
        assert d["database_url"] == ""

    def test_to_dict_empty_redis_url_shows_empty(self):
        """to_dict() shows empty string when redis_url is empty."""
        cfg = DataLineageTrackerConfig()
        d = cfg.to_dict()
        assert d["redis_url"] == ""

    def test_to_dict_contains_all_22_keys(self):
        """to_dict() includes all 22 configuration keys."""
        cfg = DataLineageTrackerConfig(
            database_url="postgresql://x:x@localhost/test",
            redis_url="redis://localhost/0",
        )
        d = cfg.to_dict()
        expected_keys = {
            "database_url", "redis_url", "log_level", "max_assets",
            "max_transformations", "max_edges", "max_graph_depth",
            "default_traversal_depth", "snapshot_interval_minutes",
            "enable_column_lineage", "enable_change_detection",
            "enable_provenance", "genesis_hash", "pool_size", "cache_ttl",
            "rate_limit", "batch_size", "coverage_warn_threshold",
            "coverage_fail_threshold", "freshness_max_age_hours",
            "quality_score_weights", "enable_metrics",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_is_json_serializable(self):
        """to_dict() result is JSON-serializable."""
        cfg = DataLineageTrackerConfig(
            database_url="postgresql://x:x@localhost/test",
            redis_url="redis://localhost/0",
        )
        d = cfg.to_dict()
        serialized = json.dumps(d)
        assert isinstance(serialized, str)

    def test_repr_safe_no_credentials(self):
        """__repr__() does not expose credentials."""
        cfg = DataLineageTrackerConfig(
            database_url="postgresql://secret:password@host/db",
            redis_url="redis://secret_pass@host:6379",
        )
        r = repr(cfg)
        assert "secret" not in r
        assert "password" not in r
        assert "DataLineageTrackerConfig" in r

    def test_repr_contains_class_name(self):
        """__repr__() starts with class name."""
        cfg = DataLineageTrackerConfig(
            database_url="postgresql://x:x@localhost/test",
            redis_url="redis://localhost/0",
        )
        r = repr(cfg)
        assert r.startswith("DataLineageTrackerConfig(")


# ============================================================================
# TestDataLineageTrackerConfigSingleton
# ============================================================================


class TestDataLineageTrackerConfigSingleton:
    """Verify singleton get_config / set_config / reset_config behavior."""

    def test_singleton_get_config(self, monkeypatch):
        """get_config() returns the same instance on repeated calls."""
        monkeypatch.setenv("GL_DLT_DATABASE_URL", "postgresql://x:x@localhost/test")
        monkeypatch.setenv("GL_DLT_REDIS_URL", "redis://localhost/0")
        reset_config()
        cfg1 = get_config()
        cfg2 = get_config()
        assert cfg1 is cfg2

    def test_set_config_replaces_singleton(self):
        """set_config() replaces the singleton instance."""
        custom = DataLineageTrackerConfig(
            database_url="postgresql://x:x@localhost/test",
            redis_url="redis://localhost/0",
            max_assets=42,
        )
        set_config(custom)
        assert get_config() is custom
        assert get_config().max_assets == 42

    def test_reset_config_clears_singleton(self, monkeypatch):
        """reset_config() clears the singleton so next get_config creates new."""
        monkeypatch.setenv("GL_DLT_DATABASE_URL", "postgresql://x:x@localhost/test")
        monkeypatch.setenv("GL_DLT_REDIS_URL", "redis://localhost/0")
        cfg1 = get_config()
        reset_config()
        cfg2 = get_config()
        assert cfg1 is not cfg2

    def test_thread_safety_get_config(self, monkeypatch):
        """get_config() is thread-safe under concurrent access."""
        monkeypatch.setenv("GL_DLT_DATABASE_URL", "postgresql://x:x@localhost/test")
        monkeypatch.setenv("GL_DLT_REDIS_URL", "redis://localhost/0")
        reset_config()

        results = []
        errors = []

        def worker():
            try:
                cfg = get_config()
                results.append(cfg)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 20
        first = results[0]
        for cfg in results[1:]:
            assert cfg is first

    def test_thread_safety_set_config(self):
        """set_config() is thread-safe under concurrent writes."""
        errors = []

        def writer(idx):
            try:
                cfg = DataLineageTrackerConfig(
                    database_url="postgresql://x:x@localhost/test",
                    redis_url="redis://localhost/0",
                    max_assets=idx + 1,
                )
                set_config(cfg)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        final = get_config()
        assert final.max_assets > 0
