# -*- coding: utf-8 -*-
"""
Unit Tests for ValidationRuleEngineConfig - AGENT-DATA-019

Tests the ValidationRuleEngineConfig dataclass, all 22 default values,
environment variable overrides (GL_VRE_ prefix), type coercion fallback,
post-init validation constraints, thread-safe singleton management,
serialisation helpers (to_dict, repr), and equality behaviour.

Target: 60-80 tests, 85%+ coverage of greenlang.validation_rule_engine.config

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import dataclasses
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from greenlang.validation_rule_engine.config import (
    ValidationRuleEngineConfig,
    get_config,
    reset_config,
    set_config,
)


# ============================================================================
# TestValidationRuleEngineConfigDefaults - verify every default value (22 fields)
# ============================================================================


class TestValidationRuleEngineConfigDefaults:
    """Every field of ValidationRuleEngineConfig must have the correct default."""

    def test_default_database_url(self):
        cfg = ValidationRuleEngineConfig()
        assert cfg.database_url == ""

    def test_default_redis_url(self):
        cfg = ValidationRuleEngineConfig()
        assert cfg.redis_url == ""

    def test_default_log_level(self):
        cfg = ValidationRuleEngineConfig()
        assert cfg.log_level == "INFO"

    def test_default_max_rules(self):
        cfg = ValidationRuleEngineConfig()
        assert cfg.max_rules == 100_000

    def test_default_max_rule_sets(self):
        cfg = ValidationRuleEngineConfig()
        assert cfg.max_rule_sets == 10_000

    def test_default_max_rules_per_set(self):
        cfg = ValidationRuleEngineConfig()
        assert cfg.max_rules_per_set == 500

    def test_default_max_compound_depth(self):
        cfg = ValidationRuleEngineConfig()
        assert cfg.max_compound_depth == 10

    def test_default_pass_threshold(self):
        cfg = ValidationRuleEngineConfig()
        assert cfg.default_pass_threshold == 0.95

    def test_default_warn_threshold(self):
        cfg = ValidationRuleEngineConfig()
        assert cfg.default_warn_threshold == 0.80

    def test_default_evaluation_timeout(self):
        cfg = ValidationRuleEngineConfig()
        assert cfg.evaluation_timeout == 300

    def test_default_batch_size(self):
        cfg = ValidationRuleEngineConfig()
        assert cfg.batch_size == 1000

    def test_default_max_batch_datasets(self):
        cfg = ValidationRuleEngineConfig()
        assert cfg.max_batch_datasets == 100

    def test_default_enable_provenance(self):
        cfg = ValidationRuleEngineConfig()
        assert cfg.enable_provenance is True

    def test_default_genesis_hash(self):
        cfg = ValidationRuleEngineConfig()
        assert cfg.genesis_hash == "greenlang-validation-rule-genesis"

    def test_default_enable_metrics(self):
        cfg = ValidationRuleEngineConfig()
        assert cfg.enable_metrics is True

    def test_default_pool_size(self):
        cfg = ValidationRuleEngineConfig()
        assert cfg.pool_size == 5

    def test_default_cache_ttl(self):
        cfg = ValidationRuleEngineConfig()
        assert cfg.cache_ttl == 300

    def test_default_rate_limit(self):
        cfg = ValidationRuleEngineConfig()
        assert cfg.rate_limit == 200

    def test_default_enable_conflict_detection(self):
        cfg = ValidationRuleEngineConfig()
        assert cfg.enable_conflict_detection is True

    def test_default_enable_short_circuit(self):
        cfg = ValidationRuleEngineConfig()
        assert cfg.enable_short_circuit is True

    def test_default_max_evaluation_rows(self):
        cfg = ValidationRuleEngineConfig()
        assert cfg.max_evaluation_rows == 1_000_000

    def test_default_report_retention_days(self):
        cfg = ValidationRuleEngineConfig()
        assert cfg.report_retention_days == 90


# ============================================================================
# TestFromEnv - environment variable overrides
# ============================================================================


class TestFromEnv:
    """ValidationRuleEngineConfig.from_env() reads GL_VRE_* variables."""

    def test_from_env_database_url(self, monkeypatch):
        monkeypatch.setenv("GL_VRE_DATABASE_URL", "postgres://host/db")
        cfg = ValidationRuleEngineConfig.from_env()
        assert cfg.database_url == "postgres://host/db"

    def test_from_env_redis_url(self, monkeypatch):
        monkeypatch.setenv("GL_VRE_REDIS_URL", "redis://host:6379")
        cfg = ValidationRuleEngineConfig.from_env()
        assert cfg.redis_url == "redis://host:6379"

    def test_from_env_log_level(self, monkeypatch):
        monkeypatch.setenv("GL_VRE_LOG_LEVEL", "DEBUG")
        cfg = ValidationRuleEngineConfig.from_env()
        assert cfg.log_level == "DEBUG"

    def test_from_env_log_level_normalised(self, monkeypatch):
        monkeypatch.setenv("GL_VRE_LOG_LEVEL", "warning")
        cfg = ValidationRuleEngineConfig.from_env()
        assert cfg.log_level == "WARNING"

    def test_from_env_max_rules(self, monkeypatch):
        monkeypatch.setenv("GL_VRE_MAX_RULES", "200000")
        cfg = ValidationRuleEngineConfig.from_env()
        assert cfg.max_rules == 200_000

    def test_from_env_max_rule_sets(self, monkeypatch):
        monkeypatch.setenv("GL_VRE_MAX_RULE_SETS", "5000")
        cfg = ValidationRuleEngineConfig.from_env()
        assert cfg.max_rule_sets == 5_000

    def test_from_env_max_rules_per_set(self, monkeypatch):
        monkeypatch.setenv("GL_VRE_MAX_RULES_PER_SET", "250")
        cfg = ValidationRuleEngineConfig.from_env()
        assert cfg.max_rules_per_set == 250

    def test_from_env_max_compound_depth(self, monkeypatch):
        monkeypatch.setenv("GL_VRE_MAX_COMPOUND_DEPTH", "20")
        cfg = ValidationRuleEngineConfig.from_env()
        assert cfg.max_compound_depth == 20

    def test_from_env_pass_threshold(self, monkeypatch):
        monkeypatch.setenv("GL_VRE_DEFAULT_PASS_THRESHOLD", "0.99")
        cfg = ValidationRuleEngineConfig.from_env()
        assert cfg.default_pass_threshold == pytest.approx(0.99)

    def test_from_env_warn_threshold(self, monkeypatch):
        monkeypatch.setenv("GL_VRE_DEFAULT_WARN_THRESHOLD", "0.70")
        cfg = ValidationRuleEngineConfig.from_env()
        assert cfg.default_warn_threshold == pytest.approx(0.70)

    def test_from_env_evaluation_timeout(self, monkeypatch):
        monkeypatch.setenv("GL_VRE_EVALUATION_TIMEOUT", "600")
        cfg = ValidationRuleEngineConfig.from_env()
        assert cfg.evaluation_timeout == 600

    def test_from_env_batch_size(self, monkeypatch):
        monkeypatch.setenv("GL_VRE_BATCH_SIZE", "5000")
        cfg = ValidationRuleEngineConfig.from_env()
        assert cfg.batch_size == 5000

    def test_from_env_max_batch_datasets(self, monkeypatch):
        monkeypatch.setenv("GL_VRE_MAX_BATCH_DATASETS", "50")
        cfg = ValidationRuleEngineConfig.from_env()
        assert cfg.max_batch_datasets == 50

    def test_from_env_enable_provenance_true(self, monkeypatch):
        monkeypatch.setenv("GL_VRE_ENABLE_PROVENANCE", "true")
        cfg = ValidationRuleEngineConfig.from_env()
        assert cfg.enable_provenance is True

    def test_from_env_enable_provenance_false(self, monkeypatch):
        monkeypatch.setenv("GL_VRE_ENABLE_PROVENANCE", "false")
        cfg = ValidationRuleEngineConfig.from_env()
        assert cfg.enable_provenance is False

    def test_from_env_enable_provenance_yes(self, monkeypatch):
        monkeypatch.setenv("GL_VRE_ENABLE_PROVENANCE", "yes")
        cfg = ValidationRuleEngineConfig.from_env()
        assert cfg.enable_provenance is True

    def test_from_env_enable_provenance_1(self, monkeypatch):
        monkeypatch.setenv("GL_VRE_ENABLE_PROVENANCE", "1")
        cfg = ValidationRuleEngineConfig.from_env()
        assert cfg.enable_provenance is True

    def test_from_env_genesis_hash(self, monkeypatch):
        monkeypatch.setenv("GL_VRE_GENESIS_HASH", "custom-genesis")
        cfg = ValidationRuleEngineConfig.from_env()
        assert cfg.genesis_hash == "custom-genesis"

    def test_from_env_enable_metrics(self, monkeypatch):
        monkeypatch.setenv("GL_VRE_ENABLE_METRICS", "false")
        cfg = ValidationRuleEngineConfig.from_env()
        assert cfg.enable_metrics is False

    def test_from_env_pool_size(self, monkeypatch):
        monkeypatch.setenv("GL_VRE_POOL_SIZE", "10")
        cfg = ValidationRuleEngineConfig.from_env()
        assert cfg.pool_size == 10

    def test_from_env_cache_ttl(self, monkeypatch):
        monkeypatch.setenv("GL_VRE_CACHE_TTL", "600")
        cfg = ValidationRuleEngineConfig.from_env()
        assert cfg.cache_ttl == 600

    def test_from_env_rate_limit(self, monkeypatch):
        monkeypatch.setenv("GL_VRE_RATE_LIMIT", "500")
        cfg = ValidationRuleEngineConfig.from_env()
        assert cfg.rate_limit == 500

    def test_from_env_enable_conflict_detection(self, monkeypatch):
        monkeypatch.setenv("GL_VRE_ENABLE_CONFLICT_DETECTION", "false")
        cfg = ValidationRuleEngineConfig.from_env()
        assert cfg.enable_conflict_detection is False

    def test_from_env_enable_short_circuit(self, monkeypatch):
        monkeypatch.setenv("GL_VRE_ENABLE_SHORT_CIRCUIT", "false")
        cfg = ValidationRuleEngineConfig.from_env()
        assert cfg.enable_short_circuit is False

    def test_from_env_max_evaluation_rows(self, monkeypatch):
        monkeypatch.setenv("GL_VRE_MAX_EVALUATION_ROWS", "500000")
        cfg = ValidationRuleEngineConfig.from_env()
        assert cfg.max_evaluation_rows == 500_000

    def test_from_env_report_retention_days(self, monkeypatch):
        monkeypatch.setenv("GL_VRE_REPORT_RETENTION_DAYS", "365")
        cfg = ValidationRuleEngineConfig.from_env()
        assert cfg.report_retention_days == 365

    def test_from_env_invalid_int_fallback(self, monkeypatch):
        """Invalid integer values fall back to default with no crash."""
        monkeypatch.setenv("GL_VRE_MAX_RULES", "not_a_number")
        cfg = ValidationRuleEngineConfig.from_env()
        assert cfg.max_rules == 100_000  # default

    def test_from_env_invalid_float_fallback(self, monkeypatch):
        """Invalid float values fall back to default with no crash."""
        monkeypatch.setenv("GL_VRE_DEFAULT_PASS_THRESHOLD", "abc")
        cfg = ValidationRuleEngineConfig.from_env()
        assert cfg.default_pass_threshold == 0.95  # default

    def test_from_env_whitespace_trimmed(self, monkeypatch):
        """Whitespace in env values is trimmed."""
        monkeypatch.setenv("GL_VRE_LOG_LEVEL", "  ERROR  ")
        cfg = ValidationRuleEngineConfig.from_env()
        assert cfg.log_level == "ERROR"


# ============================================================================
# TestValidation - post-init constraint errors
# ============================================================================


class TestValidation:
    """Post-init validation must reject invalid configurations."""

    def test_invalid_log_level(self):
        with pytest.raises(ValueError, match="log_level"):
            ValidationRuleEngineConfig(log_level="TRACE")

    def test_invalid_log_level_empty(self):
        with pytest.raises(ValueError, match="log_level"):
            ValidationRuleEngineConfig(log_level="")

    def test_negative_max_rules(self):
        with pytest.raises(ValueError, match="max_rules"):
            ValidationRuleEngineConfig(max_rules=-1)

    def test_zero_max_rules(self):
        with pytest.raises(ValueError, match="max_rules"):
            ValidationRuleEngineConfig(max_rules=0)

    def test_negative_max_rule_sets(self):
        with pytest.raises(ValueError, match="max_rule_sets"):
            ValidationRuleEngineConfig(max_rule_sets=-1)

    def test_zero_max_rule_sets(self):
        with pytest.raises(ValueError, match="max_rule_sets"):
            ValidationRuleEngineConfig(max_rule_sets=0)

    def test_negative_max_rules_per_set(self):
        with pytest.raises(ValueError, match="max_rules_per_set"):
            ValidationRuleEngineConfig(max_rules_per_set=-1)

    def test_zero_max_rules_per_set(self):
        with pytest.raises(ValueError, match="max_rules_per_set"):
            ValidationRuleEngineConfig(max_rules_per_set=0)

    def test_max_rules_per_set_exceeds_max_rules(self):
        with pytest.raises(ValueError, match="max_rules_per_set"):
            ValidationRuleEngineConfig(max_rules=100, max_rules_per_set=200)

    def test_negative_max_compound_depth(self):
        with pytest.raises(ValueError, match="max_compound_depth"):
            ValidationRuleEngineConfig(max_compound_depth=-1)

    def test_zero_max_compound_depth(self):
        with pytest.raises(ValueError, match="max_compound_depth"):
            ValidationRuleEngineConfig(max_compound_depth=0)

    def test_max_compound_depth_over_100(self):
        with pytest.raises(ValueError, match="max_compound_depth"):
            ValidationRuleEngineConfig(max_compound_depth=101)

    def test_pass_threshold_below_zero(self):
        with pytest.raises(ValueError, match="default_pass_threshold"):
            ValidationRuleEngineConfig(default_pass_threshold=-0.1)

    def test_pass_threshold_above_one(self):
        with pytest.raises(ValueError, match="default_pass_threshold"):
            ValidationRuleEngineConfig(default_pass_threshold=1.1)

    def test_warn_threshold_below_zero(self):
        with pytest.raises(ValueError, match="default_warn_threshold"):
            ValidationRuleEngineConfig(default_warn_threshold=-0.1)

    def test_warn_threshold_above_one(self):
        with pytest.raises(ValueError, match="default_warn_threshold"):
            ValidationRuleEngineConfig(default_warn_threshold=1.1)

    def test_warn_threshold_equals_pass_threshold(self):
        with pytest.raises(ValueError, match="default_warn_threshold"):
            ValidationRuleEngineConfig(
                default_pass_threshold=0.90,
                default_warn_threshold=0.90,
            )

    def test_warn_threshold_exceeds_pass_threshold(self):
        with pytest.raises(ValueError, match="default_warn_threshold"):
            ValidationRuleEngineConfig(
                default_pass_threshold=0.80,
                default_warn_threshold=0.90,
            )

    def test_evaluation_timeout_zero(self):
        with pytest.raises(ValueError, match="evaluation_timeout"):
            ValidationRuleEngineConfig(evaluation_timeout=0)

    def test_evaluation_timeout_negative(self):
        with pytest.raises(ValueError, match="evaluation_timeout"):
            ValidationRuleEngineConfig(evaluation_timeout=-10)

    def test_evaluation_timeout_over_3600(self):
        with pytest.raises(ValueError, match="evaluation_timeout"):
            ValidationRuleEngineConfig(evaluation_timeout=3601)

    def test_batch_size_zero(self):
        with pytest.raises(ValueError, match="batch_size"):
            ValidationRuleEngineConfig(batch_size=0)

    def test_batch_size_negative(self):
        with pytest.raises(ValueError, match="batch_size"):
            ValidationRuleEngineConfig(batch_size=-5)

    def test_max_batch_datasets_zero(self):
        with pytest.raises(ValueError, match="max_batch_datasets"):
            ValidationRuleEngineConfig(max_batch_datasets=0)

    def test_genesis_hash_empty(self):
        with pytest.raises(ValueError, match="genesis_hash"):
            ValidationRuleEngineConfig(genesis_hash="")

    def test_pool_size_zero(self):
        with pytest.raises(ValueError, match="pool_size"):
            ValidationRuleEngineConfig(pool_size=0)

    def test_cache_ttl_zero(self):
        with pytest.raises(ValueError, match="cache_ttl"):
            ValidationRuleEngineConfig(cache_ttl=0)

    def test_rate_limit_zero(self):
        with pytest.raises(ValueError, match="rate_limit"):
            ValidationRuleEngineConfig(rate_limit=0)

    def test_max_evaluation_rows_zero(self):
        with pytest.raises(ValueError, match="max_evaluation_rows"):
            ValidationRuleEngineConfig(max_evaluation_rows=0)

    def test_report_retention_days_zero(self):
        with pytest.raises(ValueError, match="report_retention_days"):
            ValidationRuleEngineConfig(report_retention_days=0)

    def test_report_retention_days_over_3650(self):
        with pytest.raises(ValueError, match="report_retention_days"):
            ValidationRuleEngineConfig(report_retention_days=3651)

    def test_valid_boundary_values(self):
        """Config with all valid boundary values must not raise."""
        cfg = ValidationRuleEngineConfig(
            max_rules=1,
            max_rule_sets=1,
            max_rules_per_set=1,
            max_compound_depth=1,
            default_pass_threshold=1.0,
            default_warn_threshold=0.0,
            evaluation_timeout=1,
            batch_size=1,
            max_batch_datasets=1,
            pool_size=1,
            cache_ttl=1,
            rate_limit=1,
            max_evaluation_rows=1,
            report_retention_days=1,
        )
        assert cfg.max_rules == 1

    def test_valid_max_boundary_values(self):
        """Config with max boundary values must not raise."""
        cfg = ValidationRuleEngineConfig(
            max_compound_depth=100,
            evaluation_timeout=3600,
            report_retention_days=3650,
        )
        assert cfg.max_compound_depth == 100
        assert cfg.evaluation_timeout == 3600
        assert cfg.report_retention_days == 3650


# ============================================================================
# TestToDict - serialisation
# ============================================================================


class TestToDict:
    """to_dict() must produce correct keys and redacted values."""

    def test_to_dict_returns_dict(self):
        cfg = ValidationRuleEngineConfig()
        d = cfg.to_dict()
        assert isinstance(d, dict)

    def test_to_dict_key_count(self):
        cfg = ValidationRuleEngineConfig()
        d = cfg.to_dict()
        assert len(d) == 22

    def test_to_dict_redacts_database_url_when_set(self):
        cfg = ValidationRuleEngineConfig(database_url="postgres://secret")
        d = cfg.to_dict()
        assert d["database_url"] == "***"

    def test_to_dict_redacts_redis_url_when_set(self):
        cfg = ValidationRuleEngineConfig(redis_url="redis://secret")
        d = cfg.to_dict()
        assert d["redis_url"] == "***"

    def test_to_dict_empty_database_url_not_redacted(self):
        cfg = ValidationRuleEngineConfig(database_url="")
        d = cfg.to_dict()
        assert d["database_url"] == ""

    def test_to_dict_empty_redis_url_not_redacted(self):
        cfg = ValidationRuleEngineConfig(redis_url="")
        d = cfg.to_dict()
        assert d["redis_url"] == ""

    def test_to_dict_max_rules(self):
        cfg = ValidationRuleEngineConfig(max_rules=50_000)
        d = cfg.to_dict()
        assert d["max_rules"] == 50_000

    def test_to_dict_pass_threshold(self):
        cfg = ValidationRuleEngineConfig()
        d = cfg.to_dict()
        assert d["default_pass_threshold"] == 0.95

    def test_to_dict_warn_threshold(self):
        cfg = ValidationRuleEngineConfig()
        d = cfg.to_dict()
        assert d["default_warn_threshold"] == 0.80

    def test_to_dict_boolean_fields(self):
        cfg = ValidationRuleEngineConfig()
        d = cfg.to_dict()
        assert d["enable_provenance"] is True
        assert d["enable_metrics"] is True
        assert d["enable_conflict_detection"] is True
        assert d["enable_short_circuit"] is True


# ============================================================================
# TestRepr - string representation
# ============================================================================


class TestRepr:
    """__repr__ must produce a valid, credential-safe string."""

    def test_repr_contains_class_name(self):
        cfg = ValidationRuleEngineConfig()
        assert "ValidationRuleEngineConfig" in repr(cfg)

    def test_repr_redacts_database_url(self):
        cfg = ValidationRuleEngineConfig(database_url="postgres://secret")
        r = repr(cfg)
        assert "secret" not in r
        assert "***" in r

    def test_repr_redacts_redis_url(self):
        cfg = ValidationRuleEngineConfig(redis_url="redis://secret")
        r = repr(cfg)
        assert "secret" not in r
        assert "***" in r

    def test_repr_is_string(self):
        cfg = ValidationRuleEngineConfig()
        assert isinstance(repr(cfg), str)


# ============================================================================
# TestSingletonManagement - get_config / set_config / reset_config
# ============================================================================


class TestSingletonManagement:
    """Thread-safe singleton accessor and mutator tests."""

    def test_get_config_returns_instance(self):
        cfg = get_config()
        assert isinstance(cfg, ValidationRuleEngineConfig)

    def test_get_config_same_instance(self):
        a = get_config()
        b = get_config()
        assert a is b

    def test_set_config_replaces_singleton(self):
        custom = ValidationRuleEngineConfig(max_rules=42, max_rules_per_set=10)
        set_config(custom)
        assert get_config().max_rules == 42

    def test_reset_config_clears_singleton(self):
        custom = ValidationRuleEngineConfig(max_rules=99, max_rules_per_set=50)
        set_config(custom)
        assert get_config().max_rules == 99
        reset_config()
        cfg = get_config()
        assert cfg.max_rules == 100_000  # default

    def test_reset_and_reread_env(self, monkeypatch):
        monkeypatch.setenv("GL_VRE_MAX_RULES", "777")
        reset_config()
        cfg = get_config()
        assert cfg.max_rules == 777

    def test_thread_safety_get_config(self):
        """Multiple threads calling get_config simultaneously must get the same instance."""
        results = []

        def _get():
            results.append(id(get_config()))

        threads = [threading.Thread(target=_get) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(set(results)) == 1

    def test_thread_safety_set_and_get(self):
        """set_config + get_config across threads must be consistent."""
        barrier = threading.Barrier(4)
        errors = []

        def _set_and_check(max_rules_val):
            try:
                barrier.wait(timeout=5)
                custom = ValidationRuleEngineConfig(max_rules=max_rules_val)
                set_config(custom)
                # Immediately verify we get a valid config
                cfg = get_config()
                assert isinstance(cfg, ValidationRuleEngineConfig)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=_set_and_check, args=(i * 1000,))
            for i in range(1, 5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0

    def test_thread_pool_get_config(self):
        """ThreadPoolExecutor calls to get_config must all succeed."""
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(get_config) for _ in range(20)]
            configs = [f.result() for f in as_completed(futures)]
        assert all(isinstance(c, ValidationRuleEngineConfig) for c in configs)


# ============================================================================
# TestDataclass - dataclass behaviour
# ============================================================================


class TestDataclass:
    """ValidationRuleEngineConfig is a dataclass with expected properties."""

    def test_is_dataclass(self):
        cfg = ValidationRuleEngineConfig()
        assert dataclasses.is_dataclass(cfg)

    def test_field_count(self):
        fields = dataclasses.fields(ValidationRuleEngineConfig)
        assert len(fields) == 22

    def test_equality(self):
        a = ValidationRuleEngineConfig(max_rules=500)
        b = ValidationRuleEngineConfig(max_rules=500)
        assert a == b

    def test_inequality(self):
        a = ValidationRuleEngineConfig(max_rules=500)
        b = ValidationRuleEngineConfig(max_rules=600)
        assert a != b

    def test_custom_values_propagate(self):
        cfg = ValidationRuleEngineConfig(
            max_rules=50_000,
            max_rule_sets=5_000,
            default_pass_threshold=0.99,
            default_warn_threshold=0.85,
            enable_provenance=False,
            enable_short_circuit=False,
        )
        assert cfg.max_rules == 50_000
        assert cfg.max_rule_sets == 5_000
        assert cfg.default_pass_threshold == 0.99
        assert cfg.default_warn_threshold == 0.85
        assert cfg.enable_provenance is False
        assert cfg.enable_short_circuit is False
