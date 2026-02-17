# -*- coding: utf-8 -*-
"""
Unit tests for DataFreshnessMonitorConfig - AGENT-DATA-016

Tests the config dataclass at greenlang.data_freshness_monitor.config with
70+ tests covering default values for all 26 fields, GL_DFM_ environment
variable overrides, singleton pattern, thread safety, validation logic,
type coercion, invalid env fallback, to_dict() serialization, from_env()
classmethod, and edge cases.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-016 Data Freshness Monitor (GL-DATA-X-019)
"""

from __future__ import annotations

import concurrent.futures
import threading
from dataclasses import fields as dc_fields

import pytest

from greenlang.data_freshness_monitor.config import (
    DataFreshnessMonitorConfig,
    get_config,
    reset_config,
    set_config,
)


# ======================================================================
# 1. Default values -- all 26 fields (27 tests incl. count)
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

    def test_default_max_datasets(self, fresh_config):
        assert fresh_config.max_datasets == 50_000

    def test_default_sla_warning_hours(self, fresh_config):
        assert fresh_config.default_sla_warning_hours == pytest.approx(24.0)

    def test_default_sla_critical_hours(self, fresh_config):
        assert fresh_config.default_sla_critical_hours == pytest.approx(72.0)

    def test_default_freshness_excellent_hours(self, fresh_config):
        assert fresh_config.freshness_excellent_hours == pytest.approx(1.0)

    def test_default_freshness_good_hours(self, fresh_config):
        assert fresh_config.freshness_good_hours == pytest.approx(6.0)

    def test_default_freshness_fair_hours(self, fresh_config):
        assert fresh_config.freshness_fair_hours == pytest.approx(24.0)

    def test_default_freshness_poor_hours(self, fresh_config):
        assert fresh_config.freshness_poor_hours == pytest.approx(72.0)

    def test_default_check_interval_minutes(self, fresh_config):
        assert fresh_config.check_interval_minutes == 15

    def test_default_alert_throttle_minutes(self, fresh_config):
        assert fresh_config.alert_throttle_minutes == 60

    def test_default_alert_dedup_window_hours(self, fresh_config):
        assert fresh_config.alert_dedup_window_hours == 24

    def test_default_prediction_history_days(self, fresh_config):
        assert fresh_config.prediction_history_days == 90

    def test_default_prediction_min_samples(self, fresh_config):
        assert fresh_config.prediction_min_samples == 5

    def test_default_staleness_pattern_window_days(self, fresh_config):
        assert fresh_config.staleness_pattern_window_days == 30

    def test_default_max_workers(self, fresh_config):
        assert fresh_config.max_workers == 4

    def test_default_pool_size(self, fresh_config):
        assert fresh_config.pool_size == 5

    def test_default_cache_ttl(self, fresh_config):
        assert fresh_config.cache_ttl == 300

    def test_default_rate_limit(self, fresh_config):
        assert fresh_config.rate_limit == 100

    def test_default_enable_provenance(self, fresh_config):
        assert fresh_config.enable_provenance is True

    def test_default_enable_predictions(self, fresh_config):
        assert fresh_config.enable_predictions is True

    def test_default_enable_alerts(self, fresh_config):
        assert fresh_config.enable_alerts is True

    def test_default_escalation_enabled(self, fresh_config):
        assert fresh_config.escalation_enabled is True

    def test_default_genesis_hash(self, fresh_config):
        assert fresh_config.genesis_hash == "greenlang-data-freshness-monitor-genesis"

    def test_total_field_count(self, fresh_config):
        """Config dataclass should have exactly 26 fields."""
        assert len(dc_fields(fresh_config)) == 26


# ======================================================================
# 2. Environment variable prefix
# ======================================================================


class TestEnvPrefix:
    """Test the GL_DFM_ environment variable prefix."""

    def test_env_prefix_used(self, monkeypatch):
        monkeypatch.setenv("GL_DFM_BATCH_SIZE", "500")
        cfg = DataFreshnessMonitorConfig.from_env()
        assert cfg.batch_size == 500

    def test_wrong_prefix_ignored(self, monkeypatch):
        monkeypatch.setenv("GL_OTHER_BATCH_SIZE", "999")
        cfg = DataFreshnessMonitorConfig.from_env()
        assert cfg.batch_size == 1000  # unchanged


# ======================================================================
# 3. Environment variable overrides -- one per type
# ======================================================================


class TestEnvOverrides:
    """Each field can be overridden via GL_DFM_<FIELD_UPPER>."""

    # -- String overrides ---------------------------------------------------

    def test_env_database_url(self, monkeypatch):
        monkeypatch.setenv("GL_DFM_DATABASE_URL", "postgresql://custom/db")
        cfg = DataFreshnessMonitorConfig.from_env()
        assert cfg.database_url == "postgresql://custom/db"

    def test_env_redis_url(self, monkeypatch):
        monkeypatch.setenv("GL_DFM_REDIS_URL", "redis://custom:6380/1")
        cfg = DataFreshnessMonitorConfig.from_env()
        assert cfg.redis_url == "redis://custom:6380/1"

    def test_env_log_level(self, monkeypatch):
        monkeypatch.setenv("GL_DFM_LOG_LEVEL", "DEBUG")
        cfg = DataFreshnessMonitorConfig.from_env()
        assert cfg.log_level == "DEBUG"

    def test_env_genesis_hash(self, monkeypatch):
        monkeypatch.setenv("GL_DFM_GENESIS_HASH", "custom-genesis")
        cfg = DataFreshnessMonitorConfig.from_env()
        assert cfg.genesis_hash == "custom-genesis"

    # -- Integer overrides --------------------------------------------------

    def test_env_batch_size(self, monkeypatch):
        monkeypatch.setenv("GL_DFM_BATCH_SIZE", "2000")
        cfg = DataFreshnessMonitorConfig.from_env()
        assert cfg.batch_size == 2000

    def test_env_max_datasets(self, monkeypatch):
        monkeypatch.setenv("GL_DFM_MAX_DATASETS", "10000")
        cfg = DataFreshnessMonitorConfig.from_env()
        assert cfg.max_datasets == 10000

    def test_env_check_interval_minutes(self, monkeypatch):
        monkeypatch.setenv("GL_DFM_CHECK_INTERVAL_MINUTES", "30")
        cfg = DataFreshnessMonitorConfig.from_env()
        assert cfg.check_interval_minutes == 30

    def test_env_alert_throttle_minutes(self, monkeypatch):
        monkeypatch.setenv("GL_DFM_ALERT_THROTTLE_MINUTES", "120")
        cfg = DataFreshnessMonitorConfig.from_env()
        assert cfg.alert_throttle_minutes == 120

    def test_env_alert_dedup_window_hours(self, monkeypatch):
        monkeypatch.setenv("GL_DFM_ALERT_DEDUP_WINDOW_HOURS", "48")
        cfg = DataFreshnessMonitorConfig.from_env()
        assert cfg.alert_dedup_window_hours == 48

    def test_env_prediction_history_days(self, monkeypatch):
        monkeypatch.setenv("GL_DFM_PREDICTION_HISTORY_DAYS", "180")
        cfg = DataFreshnessMonitorConfig.from_env()
        assert cfg.prediction_history_days == 180

    def test_env_prediction_min_samples(self, monkeypatch):
        monkeypatch.setenv("GL_DFM_PREDICTION_MIN_SAMPLES", "10")
        cfg = DataFreshnessMonitorConfig.from_env()
        assert cfg.prediction_min_samples == 10

    def test_env_staleness_pattern_window_days(self, monkeypatch):
        monkeypatch.setenv("GL_DFM_STALENESS_PATTERN_WINDOW_DAYS", "60")
        cfg = DataFreshnessMonitorConfig.from_env()
        assert cfg.staleness_pattern_window_days == 60

    def test_env_max_workers(self, monkeypatch):
        monkeypatch.setenv("GL_DFM_MAX_WORKERS", "8")
        cfg = DataFreshnessMonitorConfig.from_env()
        assert cfg.max_workers == 8

    def test_env_pool_size(self, monkeypatch):
        monkeypatch.setenv("GL_DFM_POOL_SIZE", "10")
        cfg = DataFreshnessMonitorConfig.from_env()
        assert cfg.pool_size == 10

    def test_env_cache_ttl(self, monkeypatch):
        monkeypatch.setenv("GL_DFM_CACHE_TTL", "600")
        cfg = DataFreshnessMonitorConfig.from_env()
        assert cfg.cache_ttl == 600

    def test_env_rate_limit(self, monkeypatch):
        monkeypatch.setenv("GL_DFM_RATE_LIMIT", "200")
        cfg = DataFreshnessMonitorConfig.from_env()
        assert cfg.rate_limit == 200

    # -- Float overrides ----------------------------------------------------

    def test_env_default_sla_warning_hours(self, monkeypatch):
        monkeypatch.setenv("GL_DFM_DEFAULT_SLA_WARNING_HOURS", "12.0")
        cfg = DataFreshnessMonitorConfig.from_env()
        assert cfg.default_sla_warning_hours == pytest.approx(12.0)

    def test_env_default_sla_critical_hours(self, monkeypatch):
        monkeypatch.setenv("GL_DFM_DEFAULT_SLA_CRITICAL_HOURS", "96.0")
        cfg = DataFreshnessMonitorConfig.from_env()
        assert cfg.default_sla_critical_hours == pytest.approx(96.0)

    def test_env_freshness_excellent_hours(self, monkeypatch):
        monkeypatch.setenv("GL_DFM_FRESHNESS_EXCELLENT_HOURS", "0.5")
        cfg = DataFreshnessMonitorConfig.from_env()
        assert cfg.freshness_excellent_hours == pytest.approx(0.5)

    def test_env_freshness_good_hours(self, monkeypatch):
        monkeypatch.setenv("GL_DFM_FRESHNESS_GOOD_HOURS", "12.0")
        cfg = DataFreshnessMonitorConfig.from_env()
        assert cfg.freshness_good_hours == pytest.approx(12.0)

    def test_env_freshness_fair_hours(self, monkeypatch):
        monkeypatch.setenv("GL_DFM_FRESHNESS_FAIR_HOURS", "48.0")
        cfg = DataFreshnessMonitorConfig.from_env()
        assert cfg.freshness_fair_hours == pytest.approx(48.0)

    def test_env_freshness_poor_hours(self, monkeypatch):
        monkeypatch.setenv("GL_DFM_FRESHNESS_POOR_HOURS", "120.0")
        cfg = DataFreshnessMonitorConfig.from_env()
        assert cfg.freshness_poor_hours == pytest.approx(120.0)

    # -- Boolean overrides --------------------------------------------------

    def test_env_enable_provenance_false(self, monkeypatch):
        monkeypatch.setenv("GL_DFM_ENABLE_PROVENANCE", "false")
        cfg = DataFreshnessMonitorConfig.from_env()
        assert cfg.enable_provenance is False

    def test_env_enable_provenance_true(self, monkeypatch):
        monkeypatch.setenv("GL_DFM_ENABLE_PROVENANCE", "true")
        cfg = DataFreshnessMonitorConfig.from_env()
        assert cfg.enable_provenance is True

    def test_env_enable_predictions_false(self, monkeypatch):
        monkeypatch.setenv("GL_DFM_ENABLE_PREDICTIONS", "0")
        cfg = DataFreshnessMonitorConfig.from_env()
        assert cfg.enable_predictions is False

    def test_env_enable_alerts_false(self, monkeypatch):
        monkeypatch.setenv("GL_DFM_ENABLE_ALERTS", "no")
        cfg = DataFreshnessMonitorConfig.from_env()
        assert cfg.enable_alerts is False

    def test_env_escalation_enabled_false(self, monkeypatch):
        monkeypatch.setenv("GL_DFM_ESCALATION_ENABLED", "false")
        cfg = DataFreshnessMonitorConfig.from_env()
        assert cfg.escalation_enabled is False

    def test_env_bool_yes_accepted(self, monkeypatch):
        monkeypatch.setenv("GL_DFM_ENABLE_PROVENANCE", "yes")
        cfg = DataFreshnessMonitorConfig.from_env()
        assert cfg.enable_provenance is True

    def test_env_bool_one_accepted(self, monkeypatch):
        monkeypatch.setenv("GL_DFM_ENABLE_PROVENANCE", "1")
        cfg = DataFreshnessMonitorConfig.from_env()
        assert cfg.enable_provenance is True

    def test_env_bool_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("GL_DFM_ENABLE_PROVENANCE", "TRUE")
        cfg = DataFreshnessMonitorConfig.from_env()
        assert cfg.enable_provenance is True


# ======================================================================
# 4. Invalid environment variable fallbacks
# ======================================================================


class TestInvalidEnvFallbacks:
    """Non-parseable env values should fall back to defaults."""

    def test_invalid_int_falls_back(self, monkeypatch):
        monkeypatch.setenv("GL_DFM_BATCH_SIZE", "not_an_int")
        cfg = DataFreshnessMonitorConfig.from_env()
        assert cfg.batch_size == 1000  # default

    def test_invalid_float_falls_back(self, monkeypatch):
        monkeypatch.setenv("GL_DFM_DEFAULT_SLA_WARNING_HOURS", "bad_float")
        cfg = DataFreshnessMonitorConfig.from_env()
        assert cfg.default_sla_warning_hours == pytest.approx(24.0)

    def test_invalid_int_for_max_datasets(self, monkeypatch):
        monkeypatch.setenv("GL_DFM_MAX_DATASETS", "abc")
        cfg = DataFreshnessMonitorConfig.from_env()
        assert cfg.max_datasets == 50_000

    def test_invalid_float_for_freshness_excellent(self, monkeypatch):
        monkeypatch.setenv("GL_DFM_FRESHNESS_EXCELLENT_HOURS", "xyz")
        cfg = DataFreshnessMonitorConfig.from_env()
        assert cfg.freshness_excellent_hours == pytest.approx(1.0)

    def test_invalid_int_for_cache_ttl(self, monkeypatch):
        monkeypatch.setenv("GL_DFM_CACHE_TTL", "nope")
        cfg = DataFreshnessMonitorConfig.from_env()
        assert cfg.cache_ttl == 300

    def test_invalid_float_for_sla_critical(self, monkeypatch):
        monkeypatch.setenv("GL_DFM_DEFAULT_SLA_CRITICAL_HOURS", "---")
        cfg = DataFreshnessMonitorConfig.from_env()
        assert cfg.default_sla_critical_hours == pytest.approx(72.0)


# ======================================================================
# 5. Validation: SLA thresholds
# ======================================================================


class TestValidationSLAThresholds:
    """Test __post_init__ validation for SLA warning/critical ordering."""

    def test_sla_warning_zero_raises(self):
        with pytest.raises(ValueError, match="default_sla_warning_hours must be > 0.0"):
            DataFreshnessMonitorConfig(default_sla_warning_hours=0.0)

    def test_sla_warning_negative_raises(self):
        with pytest.raises(ValueError, match="default_sla_warning_hours must be > 0.0"):
            DataFreshnessMonitorConfig(default_sla_warning_hours=-1.0)

    def test_sla_critical_zero_raises(self):
        with pytest.raises(ValueError, match="default_sla_critical_hours must be > 0.0"):
            DataFreshnessMonitorConfig(default_sla_critical_hours=0.0)

    def test_sla_critical_negative_raises(self):
        with pytest.raises(ValueError, match="default_sla_critical_hours must be > 0.0"):
            DataFreshnessMonitorConfig(default_sla_critical_hours=-1.0)

    def test_sla_warning_equals_critical_raises(self):
        with pytest.raises(ValueError, match="default_sla_warning_hours must be < default_sla_critical_hours"):
            DataFreshnessMonitorConfig(
                default_sla_warning_hours=48.0,
                default_sla_critical_hours=48.0,
            )

    def test_sla_warning_exceeds_critical_raises(self):
        with pytest.raises(ValueError, match="default_sla_warning_hours must be < default_sla_critical_hours"):
            DataFreshnessMonitorConfig(
                default_sla_warning_hours=100.0,
                default_sla_critical_hours=50.0,
            )


# ======================================================================
# 6. Validation: freshness tiers ascending
# ======================================================================


class TestValidationFreshnessTiers:
    """Freshness tier boundaries must be strictly ascending."""

    def test_freshness_excellent_zero_raises(self):
        with pytest.raises(ValueError, match="freshness_excellent_hours must be > 0.0"):
            DataFreshnessMonitorConfig(freshness_excellent_hours=0.0)

    def test_freshness_excellent_negative_raises(self):
        with pytest.raises(ValueError, match="freshness_excellent_hours must be > 0.0"):
            DataFreshnessMonitorConfig(freshness_excellent_hours=-1.0)

    def test_freshness_good_zero_raises(self):
        with pytest.raises(ValueError, match="freshness_good_hours must be > 0.0"):
            DataFreshnessMonitorConfig(freshness_good_hours=0.0)

    def test_freshness_fair_zero_raises(self):
        with pytest.raises(ValueError, match="freshness_fair_hours must be > 0.0"):
            DataFreshnessMonitorConfig(freshness_fair_hours=0.0)

    def test_freshness_poor_zero_raises(self):
        with pytest.raises(ValueError, match="freshness_poor_hours must be > 0.0"):
            DataFreshnessMonitorConfig(freshness_poor_hours=0.0)

    def test_excellent_equals_good_raises(self):
        with pytest.raises(ValueError, match="freshness_excellent_hours must be < freshness_good_hours"):
            DataFreshnessMonitorConfig(
                freshness_excellent_hours=6.0,
                freshness_good_hours=6.0,
            )

    def test_excellent_exceeds_good_raises(self):
        with pytest.raises(ValueError, match="freshness_excellent_hours must be < freshness_good_hours"):
            DataFreshnessMonitorConfig(
                freshness_excellent_hours=10.0,
                freshness_good_hours=6.0,
            )

    def test_good_equals_fair_raises(self):
        with pytest.raises(ValueError, match="freshness_good_hours must be < freshness_fair_hours"):
            DataFreshnessMonitorConfig(
                freshness_good_hours=24.0,
                freshness_fair_hours=24.0,
            )

    def test_good_exceeds_fair_raises(self):
        with pytest.raises(ValueError, match="freshness_good_hours must be < freshness_fair_hours"):
            DataFreshnessMonitorConfig(
                freshness_good_hours=48.0,
                freshness_fair_hours=24.0,
            )

    def test_fair_equals_poor_raises(self):
        with pytest.raises(ValueError, match="freshness_fair_hours must be < freshness_poor_hours"):
            DataFreshnessMonitorConfig(
                freshness_fair_hours=72.0,
                freshness_poor_hours=72.0,
            )

    def test_fair_exceeds_poor_raises(self):
        with pytest.raises(ValueError, match="freshness_fair_hours must be < freshness_poor_hours"):
            DataFreshnessMonitorConfig(
                freshness_fair_hours=100.0,
                freshness_poor_hours=72.0,
            )

    def test_valid_ascending_tiers_accepted(self):
        cfg = DataFreshnessMonitorConfig(
            freshness_excellent_hours=0.5,
            freshness_good_hours=4.0,
            freshness_fair_hours=12.0,
            freshness_poor_hours=48.0,
        )
        assert cfg.freshness_excellent_hours < cfg.freshness_good_hours
        assert cfg.freshness_good_hours < cfg.freshness_fair_hours
        assert cfg.freshness_fair_hours < cfg.freshness_poor_hours


# ======================================================================
# 7. Validation: positive integer fields
# ======================================================================


class TestValidationPositiveFields:
    """Test validation of positive-only integer/float fields."""

    def test_batch_size_zero_raises(self):
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            DataFreshnessMonitorConfig(batch_size=0)

    def test_batch_size_negative_raises(self):
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            DataFreshnessMonitorConfig(batch_size=-5)

    def test_max_datasets_zero_raises(self):
        with pytest.raises(ValueError, match="max_datasets must be >= 1"):
            DataFreshnessMonitorConfig(max_datasets=0)

    def test_batch_size_exceeds_max_datasets_raises(self):
        with pytest.raises(ValueError, match="batch_size must be <= max_datasets"):
            DataFreshnessMonitorConfig(batch_size=200, max_datasets=100)

    def test_check_interval_zero_raises(self):
        with pytest.raises(ValueError, match="check_interval_minutes must be >= 1"):
            DataFreshnessMonitorConfig(check_interval_minutes=0)

    def test_alert_throttle_zero_raises(self):
        with pytest.raises(ValueError, match="alert_throttle_minutes must be >= 1"):
            DataFreshnessMonitorConfig(alert_throttle_minutes=0)

    def test_alert_dedup_window_zero_raises(self):
        with pytest.raises(ValueError, match="alert_dedup_window_hours must be >= 1"):
            DataFreshnessMonitorConfig(alert_dedup_window_hours=0)

    def test_prediction_history_days_zero_raises(self):
        with pytest.raises(ValueError, match="prediction_history_days must be >= 1"):
            DataFreshnessMonitorConfig(prediction_history_days=0)

    def test_prediction_min_samples_zero_raises(self):
        with pytest.raises(ValueError, match="prediction_min_samples must be >= 1"):
            DataFreshnessMonitorConfig(prediction_min_samples=0)

    def test_staleness_window_zero_raises(self):
        with pytest.raises(ValueError, match="staleness_pattern_window_days must be >= 1"):
            DataFreshnessMonitorConfig(staleness_pattern_window_days=0)

    def test_staleness_window_exceeds_prediction_history_raises(self):
        with pytest.raises(
            ValueError,
            match="staleness_pattern_window_days must be <= prediction_history_days",
        ):
            DataFreshnessMonitorConfig(
                staleness_pattern_window_days=100,
                prediction_history_days=90,
            )

    def test_max_workers_zero_raises(self):
        with pytest.raises(ValueError, match="max_workers must be >= 1"):
            DataFreshnessMonitorConfig(max_workers=0)

    def test_pool_size_zero_raises(self):
        with pytest.raises(ValueError, match="pool_size must be >= 1"):
            DataFreshnessMonitorConfig(pool_size=0)

    def test_cache_ttl_negative_raises(self):
        with pytest.raises(ValueError, match="cache_ttl must be >= 0"):
            DataFreshnessMonitorConfig(cache_ttl=-1)

    def test_rate_limit_zero_raises(self):
        with pytest.raises(ValueError, match="rate_limit must be >= 1"):
            DataFreshnessMonitorConfig(rate_limit=0)


# ======================================================================
# 8. Validation: log level and genesis hash
# ======================================================================


class TestValidationMisc:
    """Test log level validation and genesis hash validation."""

    def test_invalid_log_level_raises(self):
        with pytest.raises(ValueError, match="log_level must be one of"):
            DataFreshnessMonitorConfig(log_level="TRACE")

    def test_empty_genesis_hash_raises(self):
        with pytest.raises(ValueError, match="genesis_hash must not be empty"):
            DataFreshnessMonitorConfig(genesis_hash="")

    def test_valid_log_levels_accepted(self):
        for level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            cfg = DataFreshnessMonitorConfig(log_level=level)
            assert cfg.log_level == level


# ======================================================================
# 9. Singleton pattern: get_config / set_config / reset_config
# ======================================================================


class TestSingletonPattern:
    """Test thread-safe singleton accessor functions."""

    def test_get_config_returns_instance(self):
        cfg = get_config()
        assert isinstance(cfg, DataFreshnessMonitorConfig)

    def test_get_config_returns_same_instance(self):
        a = get_config()
        b = get_config()
        assert a is b

    def test_set_config_replaces_singleton(self):
        custom = DataFreshnessMonitorConfig(batch_size=42)
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
        set_config(DataFreshnessMonitorConfig(batch_size=42))
        reset_config()
        cfg = get_config()
        assert cfg.batch_size == 1000  # default

    def test_set_config_then_get_returns_custom_values(self):
        custom = DataFreshnessMonitorConfig(
            max_workers=16,
            pool_size=20,
            cache_ttl=900,
        )
        set_config(custom)
        cfg = get_config()
        assert cfg.max_workers == 16
        assert cfg.pool_size == 20
        assert cfg.cache_ttl == 900


# ======================================================================
# 10. Thread safety
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
# 11. to_dict() serialization
# ======================================================================


class TestToDict:
    """Test to_dict() method returns all fields correctly."""

    def test_to_dict_returns_dict(self, fresh_config):
        d = fresh_config.to_dict()
        assert isinstance(d, dict)

    def test_to_dict_has_all_fields(self, fresh_config):
        d = fresh_config.to_dict()
        assert len(d) == 26

    def test_to_dict_field_names(self, fresh_config):
        d = fresh_config.to_dict()
        expected_keys = {f.name for f in dc_fields(fresh_config)}
        assert set(d.keys()) == expected_keys

    def test_to_dict_values_match_attributes(self, fresh_config):
        d = fresh_config.to_dict()
        for field in dc_fields(fresh_config):
            assert d[field.name] == getattr(fresh_config, field.name)

    def test_to_dict_custom_values(self):
        cfg = DataFreshnessMonitorConfig(
            database_url="postgresql://test/db",
            batch_size=500,
            enable_provenance=False,
        )
        d = cfg.to_dict()
        assert d["database_url"] == "postgresql://test/db"
        assert d["batch_size"] == 500
        assert d["enable_provenance"] is False

    def test_to_dict_returns_copy_not_reference(self, fresh_config):
        """Modifying the dict should not alter the config."""
        d = fresh_config.to_dict()
        d["batch_size"] = 9999
        assert fresh_config.batch_size == 1000


# ======================================================================
# 12. from_env() classmethod
# ======================================================================


class TestFromEnv:
    """Test from_env() creates configuration from environment."""

    def test_from_env_returns_instance(self):
        cfg = DataFreshnessMonitorConfig.from_env()
        assert isinstance(cfg, DataFreshnessMonitorConfig)

    def test_from_env_defaults_when_no_env(self):
        cfg = DataFreshnessMonitorConfig.from_env()
        assert cfg.batch_size == 1000
        assert cfg.max_datasets == 50_000

    def test_from_env_respects_multiple_overrides(self, monkeypatch):
        monkeypatch.setenv("GL_DFM_BATCH_SIZE", "250")
        monkeypatch.setenv("GL_DFM_MAX_WORKERS", "16")
        monkeypatch.setenv("GL_DFM_ENABLE_ALERTS", "false")
        cfg = DataFreshnessMonitorConfig.from_env()
        assert cfg.batch_size == 250
        assert cfg.max_workers == 16
        assert cfg.enable_alerts is False

    def test_from_env_applies_validation(self, monkeypatch):
        """from_env should still validate constraints."""
        monkeypatch.setenv("GL_DFM_BATCH_SIZE", "0")
        # The invalid int falls back to default due to _int fallback behavior,
        # but 0 is a valid int parse, so it should raise on validation
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            DataFreshnessMonitorConfig.from_env()


# ======================================================================
# 13. Edge cases and boundary values
# ======================================================================


class TestEdgeCases:
    """Test boundary values that should be accepted."""

    def test_batch_size_one_accepted(self):
        cfg = DataFreshnessMonitorConfig(batch_size=1, max_datasets=1)
        assert cfg.batch_size == 1

    def test_batch_size_equals_max_datasets_accepted(self):
        cfg = DataFreshnessMonitorConfig(batch_size=100, max_datasets=100)
        assert cfg.batch_size == cfg.max_datasets

    def test_cache_ttl_zero_accepted(self):
        cfg = DataFreshnessMonitorConfig(cache_ttl=0)
        assert cfg.cache_ttl == 0

    def test_staleness_equals_prediction_history_accepted(self):
        cfg = DataFreshnessMonitorConfig(
            staleness_pattern_window_days=90,
            prediction_history_days=90,
        )
        assert cfg.staleness_pattern_window_days == cfg.prediction_history_days

    def test_multiple_validation_errors_aggregated(self):
        """Multiple violations produce a combined error message."""
        with pytest.raises(ValueError) as exc_info:
            DataFreshnessMonitorConfig(
                batch_size=0,
                max_datasets=0,
                max_workers=0,
            )
        msg = str(exc_info.value)
        assert "batch_size" in msg
        assert "max_datasets" in msg
        assert "max_workers" in msg

    def test_very_small_freshness_tiers_accepted(self):
        cfg = DataFreshnessMonitorConfig(
            freshness_excellent_hours=0.01,
            freshness_good_hours=0.02,
            freshness_fair_hours=0.03,
            freshness_poor_hours=0.04,
        )
        assert cfg.freshness_excellent_hours == pytest.approx(0.01)

    def test_large_values_accepted(self):
        cfg = DataFreshnessMonitorConfig(
            max_datasets=1_000_000,
            batch_size=100_000,
            prediction_history_days=3650,
            staleness_pattern_window_days=365,
        )
        assert cfg.max_datasets == 1_000_000
