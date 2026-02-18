# -*- coding: utf-8 -*-
"""
Unit tests for RefrigerantsFGasConfig - AGENT-MRV-002

Tests all fields, defaults, validation, environment variable loading,
singleton management, serialization, and thread safety for the
Refrigerants & F-Gas Agent configuration module.

Target: 100+ tests covering:
  - Default values for all 25 fields
  - Custom value construction
  - Environment variable reading (from_env)
  - Type coercion (int, float, bool, str)
  - to_dict credential redaction
  - Validation error paths (negative values, invalid enums)
  - Singleton get_config / set_config / reset_config
  - Thread safety under concurrent access
  - Environment prefix consistency
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict

import pytest

from greenlang.refrigerants_fgas.config import (
    RefrigerantsFGasConfig,
    get_config,
    reset_config,
    set_config,
    _ENV_PREFIX,
)


# ===================================================================
# Default value tests
# ===================================================================


class TestDefaultValues:
    """Verify every field default matches the documented specification."""

    def test_default_database_url(self, default_config: RefrigerantsFGasConfig):
        assert default_config.database_url == "postgresql://localhost:5432/greenlang"

    def test_default_redis_url(self, default_config: RefrigerantsFGasConfig):
        assert default_config.redis_url == "redis://localhost:6379/0"

    def test_default_log_level(self, default_config: RefrigerantsFGasConfig):
        assert default_config.log_level == "INFO"

    def test_default_gwp_source(self, default_config: RefrigerantsFGasConfig):
        assert default_config.default_gwp_source == "AR6"

    def test_default_gwp_timeframe(self, default_config: RefrigerantsFGasConfig):
        assert default_config.default_gwp_timeframe == "100yr"

    def test_default_calculation_method(self, default_config: RefrigerantsFGasConfig):
        assert default_config.default_calculation_method == "equipment_based"

    def test_default_max_refrigerants(self, default_config: RefrigerantsFGasConfig):
        assert default_config.max_refrigerants == 50_000

    def test_default_max_equipment(self, default_config: RefrigerantsFGasConfig):
        assert default_config.max_equipment == 100_000

    def test_default_max_calculations(self, default_config: RefrigerantsFGasConfig):
        assert default_config.max_calculations == 1_000_000

    def test_default_max_blends(self, default_config: RefrigerantsFGasConfig):
        assert default_config.max_blends == 5_000

    def test_default_max_service_events(self, default_config: RefrigerantsFGasConfig):
        assert default_config.max_service_events == 500_000

    def test_default_uncertainty_iterations(self, default_config: RefrigerantsFGasConfig):
        assert default_config.default_uncertainty_iterations == 5_000

    def test_default_confidence_levels(self, default_config: RefrigerantsFGasConfig):
        assert default_config.confidence_levels == "90,95,99"

    def test_default_phase_down_baseline_year(self, default_config: RefrigerantsFGasConfig):
        assert default_config.phase_down_baseline_year == 2015

    def test_default_enable_blend_decomposition(self, default_config: RefrigerantsFGasConfig):
        assert default_config.enable_blend_decomposition is True

    def test_default_enable_lifecycle_tracking(self, default_config: RefrigerantsFGasConfig):
        assert default_config.enable_lifecycle_tracking is True

    def test_default_enable_compliance_checking(self, default_config: RefrigerantsFGasConfig):
        assert default_config.enable_compliance_checking is True

    def test_default_enable_provenance(self, default_config: RefrigerantsFGasConfig):
        assert default_config.enable_provenance is True

    def test_default_genesis_hash(self, default_config: RefrigerantsFGasConfig):
        assert default_config.genesis_hash == "GL-MRV-X-002-REFRIGERANTS-FGAS-GENESIS"

    def test_default_enable_metrics(self, default_config: RefrigerantsFGasConfig):
        assert default_config.enable_metrics is True

    def test_default_pool_size(self, default_config: RefrigerantsFGasConfig):
        assert default_config.pool_size == 5

    def test_default_cache_ttl(self, default_config: RefrigerantsFGasConfig):
        assert default_config.cache_ttl == 300

    def test_default_rate_limit(self, default_config: RefrigerantsFGasConfig):
        assert default_config.rate_limit == 1000

    def test_total_field_count(self, default_config: RefrigerantsFGasConfig):
        """Ensure at least 23 fields are present in to_dict."""
        d = default_config.to_dict()
        assert len(d) >= 23


# ===================================================================
# Custom values tests
# ===================================================================


class TestCustomValues:
    """Verify custom values are accepted and stored correctly."""

    def test_custom_database_url(self, custom_config: RefrigerantsFGasConfig):
        assert custom_config.database_url == "postgresql://test:test@localhost:5432/testdb"

    def test_custom_redis_url(self, custom_config: RefrigerantsFGasConfig):
        assert custom_config.redis_url == "redis://testhost:6379/1"

    def test_custom_log_level(self, custom_config: RefrigerantsFGasConfig):
        assert custom_config.log_level == "DEBUG"

    def test_custom_gwp_source(self, custom_config: RefrigerantsFGasConfig):
        assert custom_config.default_gwp_source == "AR5"

    def test_custom_gwp_timeframe(self, custom_config: RefrigerantsFGasConfig):
        assert custom_config.default_gwp_timeframe == "20yr"

    def test_custom_calculation_method(self, custom_config: RefrigerantsFGasConfig):
        assert custom_config.default_calculation_method == "mass_balance"

    def test_custom_max_refrigerants(self, custom_config: RefrigerantsFGasConfig):
        assert custom_config.max_refrigerants == 10_000

    def test_custom_max_equipment(self, custom_config: RefrigerantsFGasConfig):
        assert custom_config.max_equipment == 20_000

    def test_custom_max_calculations(self, custom_config: RefrigerantsFGasConfig):
        assert custom_config.max_calculations == 500_000

    def test_custom_max_blends(self, custom_config: RefrigerantsFGasConfig):
        assert custom_config.max_blends == 1_000

    def test_custom_max_service_events(self, custom_config: RefrigerantsFGasConfig):
        assert custom_config.max_service_events == 100_000

    def test_custom_uncertainty_iterations(self, custom_config: RefrigerantsFGasConfig):
        assert custom_config.default_uncertainty_iterations == 10_000

    def test_custom_phase_down_baseline_year(self, custom_config: RefrigerantsFGasConfig):
        assert custom_config.phase_down_baseline_year == 2016

    def test_custom_enable_blend_decomposition(self, custom_config: RefrigerantsFGasConfig):
        assert custom_config.enable_blend_decomposition is False

    def test_custom_enable_lifecycle_tracking(self, custom_config: RefrigerantsFGasConfig):
        assert custom_config.enable_lifecycle_tracking is False

    def test_custom_enable_compliance_checking(self, custom_config: RefrigerantsFGasConfig):
        assert custom_config.enable_compliance_checking is False

    def test_custom_enable_metrics(self, custom_config: RefrigerantsFGasConfig):
        assert custom_config.enable_metrics is False

    def test_custom_pool_size(self, custom_config: RefrigerantsFGasConfig):
        assert custom_config.pool_size == 10

    def test_custom_cache_ttl(self, custom_config: RefrigerantsFGasConfig):
        assert custom_config.cache_ttl == 600

    def test_custom_rate_limit(self, custom_config: RefrigerantsFGasConfig):
        assert custom_config.rate_limit == 500


# ===================================================================
# from_env tests
# ===================================================================


class TestFromEnv:
    """Verify from_env reads environment variables correctly."""

    def test_from_env_database_url(self):
        os.environ["GL_REFRIGERANTS_FGAS_DATABASE_URL"] = "postgresql://env:5432/envdb"
        cfg = RefrigerantsFGasConfig.from_env()
        assert cfg.database_url == "postgresql://env:5432/envdb"

    def test_from_env_redis_url(self):
        os.environ["GL_REFRIGERANTS_FGAS_REDIS_URL"] = "redis://envhost:6379/3"
        cfg = RefrigerantsFGasConfig.from_env()
        assert cfg.redis_url == "redis://envhost:6379/3"

    def test_from_env_log_level(self):
        os.environ["GL_REFRIGERANTS_FGAS_LOG_LEVEL"] = "WARNING"
        cfg = RefrigerantsFGasConfig.from_env()
        assert cfg.log_level == "WARNING"

    def test_from_env_gwp_source(self):
        os.environ["GL_REFRIGERANTS_FGAS_DEFAULT_GWP_SOURCE"] = "AR4"
        cfg = RefrigerantsFGasConfig.from_env()
        assert cfg.default_gwp_source == "AR4"

    def test_from_env_gwp_timeframe(self):
        os.environ["GL_REFRIGERANTS_FGAS_DEFAULT_GWP_TIMEFRAME"] = "20yr"
        cfg = RefrigerantsFGasConfig.from_env()
        assert cfg.default_gwp_timeframe == "20yr"

    def test_from_env_calculation_method(self):
        os.environ["GL_REFRIGERANTS_FGAS_DEFAULT_CALCULATION_METHOD"] = "mass_balance"
        cfg = RefrigerantsFGasConfig.from_env()
        assert cfg.default_calculation_method == "mass_balance"

    def test_from_env_max_refrigerants(self):
        os.environ["GL_REFRIGERANTS_FGAS_MAX_REFRIGERANTS"] = "25000"
        cfg = RefrigerantsFGasConfig.from_env()
        assert cfg.max_refrigerants == 25_000

    def test_from_env_max_equipment(self):
        os.environ["GL_REFRIGERANTS_FGAS_MAX_EQUIPMENT"] = "75000"
        cfg = RefrigerantsFGasConfig.from_env()
        assert cfg.max_equipment == 75_000

    def test_from_env_max_calculations(self):
        os.environ["GL_REFRIGERANTS_FGAS_MAX_CALCULATIONS"] = "2000000"
        cfg = RefrigerantsFGasConfig.from_env()
        assert cfg.max_calculations == 2_000_000

    def test_from_env_max_blends(self):
        os.environ["GL_REFRIGERANTS_FGAS_MAX_BLENDS"] = "3000"
        cfg = RefrigerantsFGasConfig.from_env()
        assert cfg.max_blends == 3_000

    def test_from_env_max_service_events(self):
        os.environ["GL_REFRIGERANTS_FGAS_MAX_SERVICE_EVENTS"] = "250000"
        cfg = RefrigerantsFGasConfig.from_env()
        assert cfg.max_service_events == 250_000

    def test_from_env_uncertainty_iterations(self):
        os.environ["GL_REFRIGERANTS_FGAS_DEFAULT_UNCERTAINTY_ITERATIONS"] = "8000"
        cfg = RefrigerantsFGasConfig.from_env()
        assert cfg.default_uncertainty_iterations == 8_000

    def test_from_env_confidence_levels(self):
        os.environ["GL_REFRIGERANTS_FGAS_CONFIDENCE_LEVELS"] = "80,90,95"
        cfg = RefrigerantsFGasConfig.from_env()
        assert cfg.confidence_levels == "80,90,95"

    def test_from_env_phase_down_baseline_year(self):
        os.environ["GL_REFRIGERANTS_FGAS_PHASE_DOWN_BASELINE_YEAR"] = "2020"
        cfg = RefrigerantsFGasConfig.from_env()
        assert cfg.phase_down_baseline_year == 2020

    def test_from_env_enable_blend_decomposition_true(self):
        os.environ["GL_REFRIGERANTS_FGAS_ENABLE_BLEND_DECOMPOSITION"] = "true"
        cfg = RefrigerantsFGasConfig.from_env()
        assert cfg.enable_blend_decomposition is True

    def test_from_env_enable_blend_decomposition_false(self):
        os.environ["GL_REFRIGERANTS_FGAS_ENABLE_BLEND_DECOMPOSITION"] = "false"
        cfg = RefrigerantsFGasConfig.from_env()
        assert cfg.enable_blend_decomposition is False

    def test_from_env_enable_lifecycle_tracking(self):
        os.environ["GL_REFRIGERANTS_FGAS_ENABLE_LIFECYCLE_TRACKING"] = "0"
        cfg = RefrigerantsFGasConfig.from_env()
        assert cfg.enable_lifecycle_tracking is False

    def test_from_env_enable_compliance_checking(self):
        os.environ["GL_REFRIGERANTS_FGAS_ENABLE_COMPLIANCE_CHECKING"] = "yes"
        cfg = RefrigerantsFGasConfig.from_env()
        assert cfg.enable_compliance_checking is True

    def test_from_env_enable_provenance(self):
        os.environ["GL_REFRIGERANTS_FGAS_ENABLE_PROVENANCE"] = "1"
        cfg = RefrigerantsFGasConfig.from_env()
        assert cfg.enable_provenance is True

    def test_from_env_genesis_hash(self):
        os.environ["GL_REFRIGERANTS_FGAS_GENESIS_HASH"] = "custom-genesis"
        cfg = RefrigerantsFGasConfig.from_env()
        assert cfg.genesis_hash == "custom-genesis"

    def test_from_env_enable_metrics(self):
        os.environ["GL_REFRIGERANTS_FGAS_ENABLE_METRICS"] = "false"
        cfg = RefrigerantsFGasConfig.from_env()
        assert cfg.enable_metrics is False

    def test_from_env_pool_size(self):
        os.environ["GL_REFRIGERANTS_FGAS_POOL_SIZE"] = "20"
        cfg = RefrigerantsFGasConfig.from_env()
        assert cfg.pool_size == 20

    def test_from_env_cache_ttl(self):
        os.environ["GL_REFRIGERANTS_FGAS_CACHE_TTL"] = "900"
        cfg = RefrigerantsFGasConfig.from_env()
        assert cfg.cache_ttl == 900

    def test_from_env_rate_limit(self):
        os.environ["GL_REFRIGERANTS_FGAS_RATE_LIMIT"] = "2000"
        cfg = RefrigerantsFGasConfig.from_env()
        assert cfg.rate_limit == 2000

    def test_from_env_defaults_when_no_vars(self):
        """Ensure from_env with no env vars returns defaults."""
        cfg = RefrigerantsFGasConfig.from_env()
        assert cfg.default_gwp_source == "AR6"
        assert cfg.default_gwp_timeframe == "100yr"
        assert cfg.default_calculation_method == "equipment_based"
        assert cfg.max_refrigerants == 50_000


# ===================================================================
# Env var type coercion tests
# ===================================================================


class TestEnvVarTypes:
    """Verify correct type conversion from env var strings."""

    @pytest.mark.parametrize("env_val,expected", [
        ("100", 100),
        ("50000", 50_000),
        ("1", 1),
    ])
    def test_int_conversion(self, env_val: str, expected: int):
        os.environ["GL_REFRIGERANTS_FGAS_MAX_REFRIGERANTS"] = env_val
        cfg = RefrigerantsFGasConfig.from_env()
        assert cfg.max_refrigerants == expected

    def test_invalid_int_falls_back_to_default(self):
        os.environ["GL_REFRIGERANTS_FGAS_MAX_REFRIGERANTS"] = "not_a_number"
        cfg = RefrigerantsFGasConfig.from_env()
        assert cfg.max_refrigerants == 50_000

    @pytest.mark.parametrize("env_val,expected", [
        ("true", True),
        ("True", True),
        ("TRUE", True),
        ("1", True),
        ("yes", True),
        ("Yes", True),
        ("false", False),
        ("False", False),
        ("0", False),
        ("no", False),
        ("", False),
    ])
    def test_bool_conversion(self, env_val: str, expected: bool):
        os.environ["GL_REFRIGERANTS_FGAS_ENABLE_METRICS"] = env_val
        cfg = RefrigerantsFGasConfig.from_env()
        assert cfg.enable_metrics is expected

    def test_string_whitespace_stripping(self):
        os.environ["GL_REFRIGERANTS_FGAS_DEFAULT_GWP_SOURCE"] = "  AR5  "
        cfg = RefrigerantsFGasConfig.from_env()
        assert cfg.default_gwp_source == "AR5"

    def test_int_whitespace_stripping(self):
        os.environ["GL_REFRIGERANTS_FGAS_POOL_SIZE"] = "  15  "
        cfg = RefrigerantsFGasConfig.from_env()
        assert cfg.pool_size == 15


# ===================================================================
# to_dict and serialization tests
# ===================================================================


class TestToDict:
    """Verify to_dict output structure and credential redaction."""

    def test_to_dict_returns_dict(self, default_config: RefrigerantsFGasConfig):
        result = default_config.to_dict()
        assert isinstance(result, dict)

    def test_to_dict_redacts_database_url(self, default_config: RefrigerantsFGasConfig):
        result = default_config.to_dict()
        assert result["database_url"] == "***"

    def test_to_dict_redacts_redis_url(self, default_config: RefrigerantsFGasConfig):
        result = default_config.to_dict()
        assert result["redis_url"] == "***"

    def test_to_dict_includes_log_level(self, default_config: RefrigerantsFGasConfig):
        result = default_config.to_dict()
        assert result["log_level"] == "INFO"

    def test_to_dict_includes_gwp_source(self, default_config: RefrigerantsFGasConfig):
        result = default_config.to_dict()
        assert result["default_gwp_source"] == "AR6"

    def test_to_dict_includes_gwp_timeframe(self, default_config: RefrigerantsFGasConfig):
        result = default_config.to_dict()
        assert result["default_gwp_timeframe"] == "100yr"

    def test_to_dict_includes_calculation_method(self, default_config: RefrigerantsFGasConfig):
        result = default_config.to_dict()
        assert result["default_calculation_method"] == "equipment_based"

    def test_to_dict_includes_all_capacity_limits(self, default_config: RefrigerantsFGasConfig):
        result = default_config.to_dict()
        assert "max_refrigerants" in result
        assert "max_equipment" in result
        assert "max_calculations" in result
        assert "max_blends" in result
        assert "max_service_events" in result

    def test_to_dict_includes_feature_toggles(self, default_config: RefrigerantsFGasConfig):
        result = default_config.to_dict()
        assert "enable_blend_decomposition" in result
        assert "enable_lifecycle_tracking" in result
        assert "enable_compliance_checking" in result
        assert "enable_provenance" in result
        assert "enable_metrics" in result

    def test_to_dict_includes_performance_tuning(self, default_config: RefrigerantsFGasConfig):
        result = default_config.to_dict()
        assert "pool_size" in result
        assert "cache_ttl" in result
        assert "rate_limit" in result

    def test_repr_contains_redacted_url(self, default_config: RefrigerantsFGasConfig):
        rep = repr(default_config)
        assert "***" in rep
        assert "postgresql://" not in rep

    def test_repr_contains_class_name(self, default_config: RefrigerantsFGasConfig):
        rep = repr(default_config)
        assert "RefrigerantsFGasConfig(" in rep

    def test_to_dict_empty_database_url_shows_empty(self):
        """When database_url is empty string, to_dict shows empty."""
        cfg = RefrigerantsFGasConfig(database_url="", redis_url="")
        result = cfg.to_dict()
        assert result["database_url"] == ""
        assert result["redis_url"] == ""


# ===================================================================
# Validation error tests
# ===================================================================


class TestValidationErrors:
    """Verify that invalid configuration values raise ValueError."""

    def test_invalid_log_level_raises(self):
        with pytest.raises(ValueError, match="log_level"):
            RefrigerantsFGasConfig(log_level="TRACE")

    def test_invalid_gwp_source_raises(self):
        with pytest.raises(ValueError, match="default_gwp_source"):
            RefrigerantsFGasConfig(default_gwp_source="AR3")

    def test_invalid_gwp_timeframe_raises(self):
        with pytest.raises(ValueError, match="default_gwp_timeframe"):
            RefrigerantsFGasConfig(default_gwp_timeframe="50yr")

    def test_invalid_calculation_method_raises(self):
        with pytest.raises(ValueError, match="default_calculation_method"):
            RefrigerantsFGasConfig(default_calculation_method="guessing")

    @pytest.mark.parametrize("field_name,value", [
        ("max_refrigerants", 0),
        ("max_refrigerants", -1),
        ("max_equipment", 0),
        ("max_equipment", -100),
        ("max_calculations", 0),
        ("max_blends", -1),
        ("max_service_events", 0),
    ])
    def test_zero_or_negative_capacity_raises(self, field_name: str, value: int):
        with pytest.raises(ValueError, match=field_name):
            RefrigerantsFGasConfig(**{field_name: value})

    @pytest.mark.parametrize("field_name,value", [
        ("max_refrigerants", 10_000_001),
        ("max_equipment", 10_000_001),
        ("max_calculations", 100_000_001),
        ("max_blends", 1_000_001),
        ("max_service_events", 50_000_001),
    ])
    def test_exceeds_upper_limit_raises(self, field_name: str, value: int):
        with pytest.raises(ValueError, match=field_name):
            RefrigerantsFGasConfig(**{field_name: value})

    def test_zero_uncertainty_iterations_raises(self):
        with pytest.raises(ValueError, match="default_uncertainty_iterations"):
            RefrigerantsFGasConfig(default_uncertainty_iterations=0)

    def test_negative_uncertainty_iterations_raises(self):
        with pytest.raises(ValueError, match="default_uncertainty_iterations"):
            RefrigerantsFGasConfig(default_uncertainty_iterations=-100)

    def test_exceeds_max_uncertainty_iterations_raises(self):
        with pytest.raises(ValueError, match="default_uncertainty_iterations"):
            RefrigerantsFGasConfig(default_uncertainty_iterations=1_000_001)

    def test_invalid_confidence_levels_format(self):
        with pytest.raises(ValueError, match="confidence_levels"):
            RefrigerantsFGasConfig(confidence_levels="abc,def")

    def test_confidence_level_zero_raises(self):
        with pytest.raises(ValueError, match="confidence level"):
            RefrigerantsFGasConfig(confidence_levels="0,50")

    def test_confidence_level_100_raises(self):
        with pytest.raises(ValueError, match="confidence level"):
            RefrigerantsFGasConfig(confidence_levels="50,100")

    def test_confidence_level_negative_raises(self):
        with pytest.raises(ValueError, match="confidence level"):
            RefrigerantsFGasConfig(confidence_levels="-5,90")

    def test_phase_down_baseline_year_too_early(self):
        with pytest.raises(ValueError, match="phase_down_baseline_year"):
            RefrigerantsFGasConfig(phase_down_baseline_year=1989)

    def test_phase_down_baseline_year_too_late(self):
        with pytest.raises(ValueError, match="phase_down_baseline_year"):
            RefrigerantsFGasConfig(phase_down_baseline_year=2101)

    def test_empty_genesis_hash_raises(self):
        with pytest.raises(ValueError, match="genesis_hash"):
            RefrigerantsFGasConfig(genesis_hash="")

    def test_zero_pool_size_raises(self):
        with pytest.raises(ValueError, match="pool_size"):
            RefrigerantsFGasConfig(pool_size=0)

    def test_negative_pool_size_raises(self):
        with pytest.raises(ValueError, match="pool_size"):
            RefrigerantsFGasConfig(pool_size=-1)

    def test_zero_cache_ttl_raises(self):
        with pytest.raises(ValueError, match="cache_ttl"):
            RefrigerantsFGasConfig(cache_ttl=0)

    def test_negative_cache_ttl_raises(self):
        with pytest.raises(ValueError, match="cache_ttl"):
            RefrigerantsFGasConfig(cache_ttl=-10)

    def test_zero_rate_limit_raises(self):
        with pytest.raises(ValueError, match="rate_limit"):
            RefrigerantsFGasConfig(rate_limit=0)

    def test_negative_rate_limit_raises(self):
        with pytest.raises(ValueError, match="rate_limit"):
            RefrigerantsFGasConfig(rate_limit=-1)

    def test_multiple_errors_reported(self):
        """Confirm multiple validation errors are collected."""
        with pytest.raises(ValueError) as exc_info:
            RefrigerantsFGasConfig(
                log_level="TRACE",
                max_refrigerants=-1,
                pool_size=0,
            )
        msg = str(exc_info.value)
        assert "log_level" in msg
        assert "max_refrigerants" in msg
        assert "pool_size" in msg


# ===================================================================
# Normalization tests
# ===================================================================


class TestNormalization:
    """Verify field normalization during __post_init__."""

    def test_log_level_normalized_to_upper(self):
        cfg = RefrigerantsFGasConfig(log_level="debug")
        assert cfg.log_level == "DEBUG"

    def test_gwp_source_normalized_to_upper(self):
        cfg = RefrigerantsFGasConfig(default_gwp_source="ar5")
        assert cfg.default_gwp_source == "AR5"

    def test_gwp_timeframe_normalized_to_lower(self):
        cfg = RefrigerantsFGasConfig(default_gwp_timeframe="100YR")
        assert cfg.default_gwp_timeframe == "100yr"

    def test_calculation_method_normalized_to_lower(self):
        cfg = RefrigerantsFGasConfig(default_calculation_method="MASS_BALANCE")
        assert cfg.default_calculation_method == "mass_balance"

    @pytest.mark.parametrize("method", [
        "equipment_based", "EQUIPMENT_BASED",
        "mass_balance", "MASS_BALANCE",
        "screening", "SCREENING",
        "direct_measurement", "DIRECT_MEASUREMENT",
        "top_down", "TOP_DOWN",
    ])
    def test_all_valid_calculation_methods(self, method: str):
        cfg = RefrigerantsFGasConfig(default_calculation_method=method)
        assert cfg.default_calculation_method == method.lower()


# ===================================================================
# Singleton get_config / set_config / reset_config tests
# ===================================================================


class TestSingleton:
    """Verify thread-safe singleton accessor behavior."""

    def test_get_config_returns_instance(self):
        cfg = get_config()
        assert isinstance(cfg, RefrigerantsFGasConfig)

    def test_get_config_returns_same_instance(self):
        cfg1 = get_config()
        cfg2 = get_config()
        assert cfg1 is cfg2

    def test_set_config_replaces_singleton(self):
        new_cfg = RefrigerantsFGasConfig(default_gwp_source="AR4")
        set_config(new_cfg)
        assert get_config().default_gwp_source == "AR4"

    def test_reset_config_clears_singleton(self):
        cfg1 = get_config()
        reset_config()
        cfg2 = get_config()
        # After reset, a new instance should be created
        assert cfg1 is not cfg2

    def test_set_config_then_get(self):
        custom = RefrigerantsFGasConfig(pool_size=42)
        set_config(custom)
        retrieved = get_config()
        assert retrieved.pool_size == 42

    def test_reset_then_from_env(self):
        os.environ["GL_REFRIGERANTS_FGAS_DEFAULT_GWP_SOURCE"] = "AR4"
        reset_config()
        cfg = get_config()
        assert cfg.default_gwp_source == "AR4"


# ===================================================================
# Thread safety tests
# ===================================================================


class TestThreadSafety:
    """Verify concurrent access to singleton does not corrupt state."""

    def test_concurrent_get_config(self):
        """Multiple threads calling get_config should all get the same instance."""
        results = []

        def worker():
            return id(get_config())

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(worker) for _ in range(20)]
            for f in as_completed(futures):
                results.append(f.result())

        assert len(set(results)) == 1, "All threads should get the same singleton"

    def test_concurrent_set_and_get(self):
        """Concurrent set_config and get_config should not raise."""
        errors = []

        def setter():
            try:
                set_config(RefrigerantsFGasConfig())
            except Exception as e:
                errors.append(e)

        def getter():
            try:
                get_config()
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(10):
            threads.append(threading.Thread(target=setter))
            threads.append(threading.Thread(target=getter))

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert len(errors) == 0, f"Thread safety errors: {errors}"


# ===================================================================
# Environment prefix tests
# ===================================================================


class TestEnvPrefix:
    """Verify the environment variable prefix is correct."""

    def test_env_prefix_value(self):
        assert _ENV_PREFIX == "GL_REFRIGERANTS_FGAS_"

    @pytest.mark.parametrize("env_var", [
        "GL_REFRIGERANTS_FGAS_DATABASE_URL",
        "GL_REFRIGERANTS_FGAS_REDIS_URL",
        "GL_REFRIGERANTS_FGAS_LOG_LEVEL",
        "GL_REFRIGERANTS_FGAS_DEFAULT_GWP_SOURCE",
        "GL_REFRIGERANTS_FGAS_DEFAULT_GWP_TIMEFRAME",
        "GL_REFRIGERANTS_FGAS_DEFAULT_CALCULATION_METHOD",
        "GL_REFRIGERANTS_FGAS_MAX_REFRIGERANTS",
        "GL_REFRIGERANTS_FGAS_MAX_EQUIPMENT",
        "GL_REFRIGERANTS_FGAS_MAX_CALCULATIONS",
        "GL_REFRIGERANTS_FGAS_MAX_BLENDS",
        "GL_REFRIGERANTS_FGAS_MAX_SERVICE_EVENTS",
        "GL_REFRIGERANTS_FGAS_DEFAULT_UNCERTAINTY_ITERATIONS",
        "GL_REFRIGERANTS_FGAS_CONFIDENCE_LEVELS",
        "GL_REFRIGERANTS_FGAS_PHASE_DOWN_BASELINE_YEAR",
        "GL_REFRIGERANTS_FGAS_ENABLE_BLEND_DECOMPOSITION",
        "GL_REFRIGERANTS_FGAS_ENABLE_LIFECYCLE_TRACKING",
        "GL_REFRIGERANTS_FGAS_ENABLE_COMPLIANCE_CHECKING",
        "GL_REFRIGERANTS_FGAS_ENABLE_PROVENANCE",
        "GL_REFRIGERANTS_FGAS_GENESIS_HASH",
        "GL_REFRIGERANTS_FGAS_ENABLE_METRICS",
        "GL_REFRIGERANTS_FGAS_POOL_SIZE",
        "GL_REFRIGERANTS_FGAS_CACHE_TTL",
        "GL_REFRIGERANTS_FGAS_RATE_LIMIT",
    ])
    def test_env_var_starts_with_prefix(self, env_var: str):
        assert env_var.startswith(_ENV_PREFIX)


# ===================================================================
# Edge cases
# ===================================================================


class TestEdgeCases:
    """Test boundary and edge-case configurations."""

    def test_minimum_valid_capacity_limits(self):
        cfg = RefrigerantsFGasConfig(
            max_refrigerants=1,
            max_equipment=1,
            max_calculations=1,
            max_blends=1,
            max_service_events=1,
        )
        assert cfg.max_refrigerants == 1

    def test_maximum_valid_capacity_limits(self):
        cfg = RefrigerantsFGasConfig(
            max_refrigerants=10_000_000,
            max_equipment=10_000_000,
            max_calculations=100_000_000,
            max_blends=1_000_000,
            max_service_events=50_000_000,
        )
        assert cfg.max_refrigerants == 10_000_000

    def test_minimum_uncertainty_iterations(self):
        cfg = RefrigerantsFGasConfig(default_uncertainty_iterations=1)
        assert cfg.default_uncertainty_iterations == 1

    def test_maximum_uncertainty_iterations(self):
        cfg = RefrigerantsFGasConfig(default_uncertainty_iterations=1_000_000)
        assert cfg.default_uncertainty_iterations == 1_000_000

    def test_boundary_phase_down_year_1990(self):
        cfg = RefrigerantsFGasConfig(phase_down_baseline_year=1990)
        assert cfg.phase_down_baseline_year == 1990

    def test_boundary_phase_down_year_2100(self):
        cfg = RefrigerantsFGasConfig(phase_down_baseline_year=2100)
        assert cfg.phase_down_baseline_year == 2100

    def test_single_confidence_level(self):
        cfg = RefrigerantsFGasConfig(confidence_levels="95")
        assert cfg.confidence_levels == "95"

    def test_many_confidence_levels(self):
        cfg = RefrigerantsFGasConfig(confidence_levels="50,80,90,95,97.5,99")
        assert "50" in cfg.confidence_levels

    @pytest.mark.parametrize("gwp_source", ["AR4", "AR5", "AR6"])
    def test_all_valid_gwp_sources(self, gwp_source: str):
        cfg = RefrigerantsFGasConfig(default_gwp_source=gwp_source)
        assert cfg.default_gwp_source == gwp_source

    @pytest.mark.parametrize("timeframe", ["100yr", "20yr"])
    def test_all_valid_gwp_timeframes(self, timeframe: str):
        cfg = RefrigerantsFGasConfig(default_gwp_timeframe=timeframe)
        assert cfg.default_gwp_timeframe == timeframe

    @pytest.mark.parametrize("log_level", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    def test_all_valid_log_levels(self, log_level: str):
        cfg = RefrigerantsFGasConfig(log_level=log_level)
        assert cfg.log_level == log_level
