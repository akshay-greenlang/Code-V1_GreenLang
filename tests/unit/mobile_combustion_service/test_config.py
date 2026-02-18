# -*- coding: utf-8 -*-
"""
Unit tests for MobileCombustionConfig - AGENT-MRV-003

Tests all configuration fields, defaults, validation, environment variable
overrides, singleton management, serialization, normalization, and thread
safety of the MobileCombustionConfig dataclass.

Target: 103+ tests
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict

import pytest

from greenlang.mobile_combustion.config import (
    MobileCombustionConfig,
    get_config,
    reset_config,
    set_config,
)


# =========================================================================
# TestDefaultValues - 24 tests
# =========================================================================


class TestDefaultValues:
    """Verify every default field of MobileCombustionConfig."""

    def test_default_database_url(self, default_config: MobileCombustionConfig) -> None:
        assert default_config.database_url == "postgresql://localhost:5432/greenlang"

    def test_default_redis_url(self, default_config: MobileCombustionConfig) -> None:
        assert default_config.redis_url == "redis://localhost:6379/0"

    def test_default_log_level(self, default_config: MobileCombustionConfig) -> None:
        assert default_config.log_level == "INFO"

    def test_default_gwp_source(self, default_config: MobileCombustionConfig) -> None:
        assert default_config.default_gwp_source == "AR6"

    def test_default_calculation_method(self, default_config: MobileCombustionConfig) -> None:
        assert default_config.default_calculation_method == "FUEL_BASED"

    def test_default_monte_carlo_iterations(self, default_config: MobileCombustionConfig) -> None:
        assert default_config.monte_carlo_iterations == 5_000

    def test_default_batch_size(self, default_config: MobileCombustionConfig) -> None:
        assert default_config.batch_size == 100

    def test_default_max_batch_size(self, default_config: MobileCombustionConfig) -> None:
        assert default_config.max_batch_size == 1_000

    def test_default_cache_ttl_seconds(self, default_config: MobileCombustionConfig) -> None:
        assert default_config.cache_ttl_seconds == 3_600

    def test_default_enable_biogenic_tracking(self, default_config: MobileCombustionConfig) -> None:
        assert default_config.enable_biogenic_tracking is True

    def test_default_enable_uncertainty(self, default_config: MobileCombustionConfig) -> None:
        assert default_config.enable_uncertainty is True

    def test_default_enable_compliance(self, default_config: MobileCombustionConfig) -> None:
        assert default_config.enable_compliance is True

    def test_default_enable_fleet_management(self, default_config: MobileCombustionConfig) -> None:
        assert default_config.enable_fleet_management is True

    def test_default_decimal_precision(self, default_config: MobileCombustionConfig) -> None:
        assert default_config.decimal_precision == 8

    def test_default_vehicle_type(self, default_config: MobileCombustionConfig) -> None:
        assert default_config.default_vehicle_type == "PASSENGER_CAR_GASOLINE"

    def test_default_fuel_type(self, default_config: MobileCombustionConfig) -> None:
        assert default_config.default_fuel_type == "GASOLINE"

    def test_default_distance_unit(self, default_config: MobileCombustionConfig) -> None:
        assert default_config.default_distance_unit == "KM"

    def test_default_fuel_economy_unit(self, default_config: MobileCombustionConfig) -> None:
        assert default_config.default_fuel_economy_unit == "L_PER_100KM"

    def test_default_confidence_level_90(self, default_config: MobileCombustionConfig) -> None:
        assert default_config.confidence_level_90 == pytest.approx(0.90)

    def test_default_confidence_level_95(self, default_config: MobileCombustionConfig) -> None:
        assert default_config.confidence_level_95 == pytest.approx(0.95)

    def test_default_confidence_level_99(self, default_config: MobileCombustionConfig) -> None:
        assert default_config.confidence_level_99 == pytest.approx(0.99)

    def test_default_regulatory_framework(self, default_config: MobileCombustionConfig) -> None:
        assert default_config.default_regulatory_framework == "GHG_PROTOCOL"

    def test_default_max_vehicles_per_fleet(self, default_config: MobileCombustionConfig) -> None:
        assert default_config.max_vehicles_per_fleet == 10_000

    def test_default_max_trips_per_query(self, default_config: MobileCombustionConfig) -> None:
        assert default_config.max_trips_per_query == 5_000

    def test_default_calculation_timeout_seconds(self, default_config: MobileCombustionConfig) -> None:
        assert default_config.calculation_timeout_seconds == 30

    def test_default_enable_metrics(self, default_config: MobileCombustionConfig) -> None:
        assert default_config.enable_metrics is True

    def test_default_enable_tracing(self, default_config: MobileCombustionConfig) -> None:
        assert default_config.enable_tracing is True

    def test_default_enable_provenance(self, default_config: MobileCombustionConfig) -> None:
        assert default_config.enable_provenance is True

    def test_default_genesis_hash(self, default_config: MobileCombustionConfig) -> None:
        assert default_config.genesis_hash == "GL-MRV-X-003-MOBILE-COMBUSTION-GENESIS"

    def test_default_pool_size(self, default_config: MobileCombustionConfig) -> None:
        assert default_config.pool_size == 10

    def test_default_rate_limit(self, default_config: MobileCombustionConfig) -> None:
        assert default_config.rate_limit == 1_000


# =========================================================================
# TestCustomValues - 15 tests
# =========================================================================


class TestCustomValues:
    """Verify custom values are accepted and stored correctly."""

    def test_custom_database_url(self, custom_config: MobileCombustionConfig) -> None:
        assert custom_config.database_url == "postgresql://testhost:5432/testdb"

    def test_custom_redis_url(self, custom_config: MobileCombustionConfig) -> None:
        assert custom_config.redis_url == "redis://testhost:6379/1"

    def test_custom_log_level(self, custom_config: MobileCombustionConfig) -> None:
        assert custom_config.log_level == "DEBUG"

    def test_custom_gwp_source(self, custom_config: MobileCombustionConfig) -> None:
        assert custom_config.default_gwp_source == "AR5"

    def test_custom_calculation_method(self, custom_config: MobileCombustionConfig) -> None:
        assert custom_config.default_calculation_method == "DISTANCE_BASED"

    def test_custom_monte_carlo_iterations(self, custom_config: MobileCombustionConfig) -> None:
        assert custom_config.monte_carlo_iterations == 1_000

    def test_custom_batch_size(self, custom_config: MobileCombustionConfig) -> None:
        assert custom_config.batch_size == 50

    def test_custom_max_batch_size(self, custom_config: MobileCombustionConfig) -> None:
        assert custom_config.max_batch_size == 500

    def test_custom_biogenic_tracking_off(self, custom_config: MobileCombustionConfig) -> None:
        assert custom_config.enable_biogenic_tracking is False

    def test_custom_distance_unit(self, custom_config: MobileCombustionConfig) -> None:
        assert custom_config.default_distance_unit == "MILES"

    def test_custom_fuel_economy_unit(self, custom_config: MobileCombustionConfig) -> None:
        assert custom_config.default_fuel_economy_unit == "MPG_US"

    def test_custom_regulatory_framework(self, custom_config: MobileCombustionConfig) -> None:
        assert custom_config.default_regulatory_framework == "ISO_14064"

    def test_custom_pool_size(self, custom_config: MobileCombustionConfig) -> None:
        assert custom_config.pool_size == 5

    def test_custom_rate_limit(self, custom_config: MobileCombustionConfig) -> None:
        assert custom_config.rate_limit == 500

    def test_custom_genesis_hash(self, custom_config: MobileCombustionConfig) -> None:
        assert custom_config.genesis_hash == "TEST-MOBILE-COMBUSTION-GENESIS"


# =========================================================================
# TestFromEnv - 20 tests
# =========================================================================


class TestFromEnv:
    """Verify GL_MOBILE_COMBUSTION_* environment variable reading."""

    def test_env_database_url(self) -> None:
        os.environ["GL_MOBILE_COMBUSTION_DATABASE_URL"] = "postgresql://envhost:5432/envdb"
        cfg = MobileCombustionConfig.from_env()
        assert cfg.database_url == "postgresql://envhost:5432/envdb"

    def test_env_redis_url(self) -> None:
        os.environ["GL_MOBILE_COMBUSTION_REDIS_URL"] = "redis://envhost:6379/2"
        cfg = MobileCombustionConfig.from_env()
        assert cfg.redis_url == "redis://envhost:6379/2"

    def test_env_log_level(self) -> None:
        os.environ["GL_MOBILE_COMBUSTION_LOG_LEVEL"] = "WARNING"
        cfg = MobileCombustionConfig.from_env()
        assert cfg.log_level == "WARNING"

    def test_env_gwp_source(self) -> None:
        os.environ["GL_MOBILE_COMBUSTION_DEFAULT_GWP_SOURCE"] = "AR4"
        cfg = MobileCombustionConfig.from_env()
        assert cfg.default_gwp_source == "AR4"

    def test_env_calculation_method(self) -> None:
        os.environ["GL_MOBILE_COMBUSTION_DEFAULT_CALCULATION_METHOD"] = "SPEND_BASED"
        cfg = MobileCombustionConfig.from_env()
        assert cfg.default_calculation_method == "SPEND_BASED"

    def test_env_monte_carlo_iterations(self) -> None:
        os.environ["GL_MOBILE_COMBUSTION_MONTE_CARLO_ITERATIONS"] = "10000"
        cfg = MobileCombustionConfig.from_env()
        assert cfg.monte_carlo_iterations == 10_000

    def test_env_batch_size(self) -> None:
        os.environ["GL_MOBILE_COMBUSTION_BATCH_SIZE"] = "200"
        cfg = MobileCombustionConfig.from_env()
        assert cfg.batch_size == 200

    def test_env_max_batch_size(self) -> None:
        os.environ["GL_MOBILE_COMBUSTION_MAX_BATCH_SIZE"] = "5000"
        cfg = MobileCombustionConfig.from_env()
        assert cfg.max_batch_size == 5_000

    def test_env_cache_ttl(self) -> None:
        os.environ["GL_MOBILE_COMBUSTION_CACHE_TTL_SECONDS"] = "7200"
        cfg = MobileCombustionConfig.from_env()
        assert cfg.cache_ttl_seconds == 7_200

    def test_env_enable_biogenic_tracking_true(self) -> None:
        os.environ["GL_MOBILE_COMBUSTION_ENABLE_BIOGENIC_TRACKING"] = "true"
        cfg = MobileCombustionConfig.from_env()
        assert cfg.enable_biogenic_tracking is True

    def test_env_enable_biogenic_tracking_false(self) -> None:
        os.environ["GL_MOBILE_COMBUSTION_ENABLE_BIOGENIC_TRACKING"] = "false"
        cfg = MobileCombustionConfig.from_env()
        assert cfg.enable_biogenic_tracking is False

    def test_env_decimal_precision(self) -> None:
        os.environ["GL_MOBILE_COMBUSTION_DECIMAL_PRECISION"] = "12"
        cfg = MobileCombustionConfig.from_env()
        assert cfg.decimal_precision == 12

    def test_env_distance_unit(self) -> None:
        os.environ["GL_MOBILE_COMBUSTION_DEFAULT_DISTANCE_UNIT"] = "NAUTICAL_MILES"
        cfg = MobileCombustionConfig.from_env()
        assert cfg.default_distance_unit == "NAUTICAL_MILES"

    def test_env_fuel_economy_unit(self) -> None:
        os.environ["GL_MOBILE_COMBUSTION_DEFAULT_FUEL_ECONOMY_UNIT"] = "KM_PER_L"
        cfg = MobileCombustionConfig.from_env()
        assert cfg.default_fuel_economy_unit == "KM_PER_L"

    def test_env_confidence_level_90(self) -> None:
        os.environ["GL_MOBILE_COMBUSTION_CONFIDENCE_LEVEL_90"] = "0.85"
        cfg = MobileCombustionConfig.from_env()
        assert cfg.confidence_level_90 == pytest.approx(0.85)

    def test_env_confidence_level_95(self) -> None:
        os.environ["GL_MOBILE_COMBUSTION_CONFIDENCE_LEVEL_95"] = "0.92"
        cfg = MobileCombustionConfig.from_env()
        assert cfg.confidence_level_95 == pytest.approx(0.92)

    def test_env_confidence_level_99(self) -> None:
        os.environ["GL_MOBILE_COMBUSTION_CONFIDENCE_LEVEL_99"] = "0.97"
        cfg = MobileCombustionConfig.from_env()
        assert cfg.confidence_level_99 == pytest.approx(0.97)

    def test_env_regulatory_framework(self) -> None:
        os.environ["GL_MOBILE_COMBUSTION_DEFAULT_REGULATORY_FRAMEWORK"] = "CSRD_ESRS_E1"
        cfg = MobileCombustionConfig.from_env()
        assert cfg.default_regulatory_framework == "CSRD_ESRS_E1"

    def test_env_max_vehicles_per_fleet(self) -> None:
        os.environ["GL_MOBILE_COMBUSTION_MAX_VEHICLES_PER_FLEET"] = "20000"
        cfg = MobileCombustionConfig.from_env()
        assert cfg.max_vehicles_per_fleet == 20_000

    def test_env_pool_size(self) -> None:
        os.environ["GL_MOBILE_COMBUSTION_POOL_SIZE"] = "20"
        cfg = MobileCombustionConfig.from_env()
        assert cfg.pool_size == 20


# =========================================================================
# TestEnvVarTypes - 7 tests
# =========================================================================


class TestEnvVarTypes:
    """Verify type coercion for different env var value types."""

    def test_int_env_var_coercion(self) -> None:
        os.environ["GL_MOBILE_COMBUSTION_BATCH_SIZE"] = "  250  "
        cfg = MobileCombustionConfig.from_env()
        assert cfg.batch_size == 250
        assert isinstance(cfg.batch_size, int)

    def test_bool_env_var_true_variants(self) -> None:
        for true_val in ("true", "True", "TRUE", "1", "yes", "Yes", "YES"):
            os.environ["GL_MOBILE_COMBUSTION_ENABLE_METRICS"] = true_val
            cfg = MobileCombustionConfig.from_env()
            assert cfg.enable_metrics is True, f"Expected True for '{true_val}'"

    def test_bool_env_var_false_variants(self) -> None:
        for false_val in ("false", "False", "FALSE", "0", "no", "No", "NO", "anything"):
            os.environ["GL_MOBILE_COMBUSTION_ENABLE_METRICS"] = false_val
            cfg = MobileCombustionConfig.from_env()
            assert cfg.enable_metrics is False, f"Expected False for '{false_val}'"

    def test_float_env_var_coercion(self) -> None:
        os.environ["GL_MOBILE_COMBUSTION_CONFIDENCE_LEVEL_95"] = "  0.93  "
        cfg = MobileCombustionConfig.from_env()
        assert cfg.confidence_level_95 == pytest.approx(0.93)
        assert isinstance(cfg.confidence_level_95, float)

    def test_string_env_var_strips_whitespace(self) -> None:
        os.environ["GL_MOBILE_COMBUSTION_DEFAULT_GWP_SOURCE"] = "  AR5  "
        cfg = MobileCombustionConfig.from_env()
        assert cfg.default_gwp_source == "AR5"

    def test_invalid_int_env_var_uses_default(self) -> None:
        os.environ["GL_MOBILE_COMBUSTION_BATCH_SIZE"] = "not_a_number"
        cfg = MobileCombustionConfig.from_env()
        assert cfg.batch_size == 100  # default

    def test_invalid_float_env_var_uses_default(self) -> None:
        os.environ["GL_MOBILE_COMBUSTION_CONFIDENCE_LEVEL_90"] = "bad_float"
        cfg = MobileCombustionConfig.from_env()
        assert cfg.confidence_level_90 == pytest.approx(0.90)  # default


# =========================================================================
# TestToDict - 10 tests
# =========================================================================


class TestToDict:
    """Verify to_dict serialization and credential redaction."""

    def test_to_dict_returns_dict(self, default_config: MobileCombustionConfig) -> None:
        result = default_config.to_dict()
        assert isinstance(result, dict)

    def test_database_url_redacted(self, default_config: MobileCombustionConfig) -> None:
        result = default_config.to_dict()
        assert result["database_url"] == "***"

    def test_redis_url_redacted(self, default_config: MobileCombustionConfig) -> None:
        result = default_config.to_dict()
        assert result["redis_url"] == "***"

    def test_empty_database_url_not_redacted(self) -> None:
        cfg = MobileCombustionConfig(database_url="")
        result = cfg.to_dict()
        assert result["database_url"] == ""

    def test_empty_redis_url_not_redacted(self) -> None:
        cfg = MobileCombustionConfig(redis_url="")
        result = cfg.to_dict()
        assert result["redis_url"] == ""

    def test_gwp_source_present(self, default_config: MobileCombustionConfig) -> None:
        result = default_config.to_dict()
        assert result["default_gwp_source"] == "AR6"

    def test_all_expected_keys_present(self, default_config: MobileCombustionConfig) -> None:
        result = default_config.to_dict()
        expected_keys = {
            "database_url", "redis_url", "log_level",
            "default_gwp_source", "default_calculation_method",
            "monte_carlo_iterations", "batch_size", "max_batch_size",
            "cache_ttl_seconds", "enable_biogenic_tracking",
            "enable_uncertainty", "enable_compliance",
            "enable_fleet_management", "decimal_precision",
            "default_vehicle_type", "default_fuel_type",
            "default_distance_unit", "default_fuel_economy_unit",
            "confidence_level_90", "confidence_level_95", "confidence_level_99",
            "default_regulatory_framework",
            "max_vehicles_per_fleet", "max_trips_per_query",
            "calculation_timeout_seconds", "enable_metrics", "enable_tracing",
            "enable_provenance", "genesis_hash", "pool_size", "rate_limit",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_to_dict_bool_values_are_bool(self, default_config: MobileCombustionConfig) -> None:
        result = default_config.to_dict()
        for key in (
            "enable_biogenic_tracking", "enable_uncertainty",
            "enable_compliance", "enable_fleet_management",
            "enable_metrics", "enable_tracing", "enable_provenance",
        ):
            assert isinstance(result[key], bool), f"{key} should be bool"

    def test_to_dict_int_values_are_int(self, default_config: MobileCombustionConfig) -> None:
        result = default_config.to_dict()
        for key in (
            "monte_carlo_iterations", "batch_size", "max_batch_size",
            "cache_ttl_seconds", "decimal_precision",
            "max_vehicles_per_fleet", "max_trips_per_query",
            "calculation_timeout_seconds", "pool_size", "rate_limit",
        ):
            assert isinstance(result[key], int), f"{key} should be int"

    def test_repr_does_not_leak_credentials(self, default_config: MobileCombustionConfig) -> None:
        repr_str = repr(default_config)
        assert "localhost:5432" not in repr_str
        assert "***" in repr_str


# =========================================================================
# TestValidationErrors - 15 tests
# =========================================================================


class TestValidationErrors:
    """Verify that invalid field values raise ValueError."""

    def test_invalid_log_level_raises(self) -> None:
        with pytest.raises(ValueError, match="log_level"):
            MobileCombustionConfig(log_level="TRACE")

    def test_invalid_gwp_source_raises(self) -> None:
        with pytest.raises(ValueError, match="default_gwp_source"):
            MobileCombustionConfig(default_gwp_source="AR3")

    def test_invalid_calculation_method_raises(self) -> None:
        with pytest.raises(ValueError, match="default_calculation_method"):
            MobileCombustionConfig(default_calculation_method="HYBRID_BASED")

    def test_invalid_distance_unit_raises(self) -> None:
        with pytest.raises(ValueError, match="default_distance_unit"):
            MobileCombustionConfig(default_distance_unit="FEET")

    def test_invalid_fuel_economy_unit_raises(self) -> None:
        with pytest.raises(ValueError, match="default_fuel_economy_unit"):
            MobileCombustionConfig(default_fuel_economy_unit="GALLONS_PER_MILE")

    def test_invalid_regulatory_framework_raises(self) -> None:
        with pytest.raises(ValueError, match="default_regulatory_framework"):
            MobileCombustionConfig(default_regulatory_framework="CBAM")

    def test_negative_decimal_precision_raises(self) -> None:
        with pytest.raises(ValueError, match="decimal_precision"):
            MobileCombustionConfig(decimal_precision=-1)

    def test_excessive_decimal_precision_raises(self) -> None:
        with pytest.raises(ValueError, match="decimal_precision"):
            MobileCombustionConfig(decimal_precision=21)

    def test_zero_batch_size_raises(self) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            MobileCombustionConfig(batch_size=0)

    def test_batch_size_exceeds_max_raises(self) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            MobileCombustionConfig(batch_size=2000, max_batch_size=1000)

    def test_zero_monte_carlo_iterations_raises(self) -> None:
        with pytest.raises(ValueError, match="monte_carlo_iterations"):
            MobileCombustionConfig(monte_carlo_iterations=0)

    def test_excessive_monte_carlo_iterations_raises(self) -> None:
        with pytest.raises(ValueError, match="monte_carlo_iterations"):
            MobileCombustionConfig(monte_carlo_iterations=2_000_000)

    def test_zero_cache_ttl_raises(self) -> None:
        with pytest.raises(ValueError, match="cache_ttl_seconds"):
            MobileCombustionConfig(cache_ttl_seconds=0)

    def test_confidence_level_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="confidence_level_90"):
            MobileCombustionConfig(confidence_level_90=0.0)

    def test_confidence_level_one_raises(self) -> None:
        with pytest.raises(ValueError, match="confidence_level_95"):
            MobileCombustionConfig(confidence_level_95=1.0)

    def test_zero_max_vehicles_per_fleet_raises(self) -> None:
        with pytest.raises(ValueError, match="max_vehicles_per_fleet"):
            MobileCombustionConfig(max_vehicles_per_fleet=0)

    def test_excessive_max_vehicles_per_fleet_raises(self) -> None:
        with pytest.raises(ValueError, match="max_vehicles_per_fleet"):
            MobileCombustionConfig(max_vehicles_per_fleet=2_000_000)

    def test_zero_max_trips_per_query_raises(self) -> None:
        with pytest.raises(ValueError, match="max_trips_per_query"):
            MobileCombustionConfig(max_trips_per_query=0)

    def test_excessive_max_trips_per_query_raises(self) -> None:
        with pytest.raises(ValueError, match="max_trips_per_query"):
            MobileCombustionConfig(max_trips_per_query=200_000)

    def test_zero_calculation_timeout_raises(self) -> None:
        with pytest.raises(ValueError, match="calculation_timeout_seconds"):
            MobileCombustionConfig(calculation_timeout_seconds=0)

    def test_excessive_calculation_timeout_raises(self) -> None:
        with pytest.raises(ValueError, match="calculation_timeout_seconds"):
            MobileCombustionConfig(calculation_timeout_seconds=601)

    def test_empty_genesis_hash_raises(self) -> None:
        with pytest.raises(ValueError, match="genesis_hash"):
            MobileCombustionConfig(genesis_hash="")

    def test_zero_pool_size_raises(self) -> None:
        with pytest.raises(ValueError, match="pool_size"):
            MobileCombustionConfig(pool_size=0)

    def test_zero_rate_limit_raises(self) -> None:
        with pytest.raises(ValueError, match="rate_limit"):
            MobileCombustionConfig(rate_limit=0)

    def test_negative_max_batch_size_raises(self) -> None:
        with pytest.raises(ValueError, match="max_batch_size"):
            MobileCombustionConfig(max_batch_size=-1)

    def test_excessive_max_batch_size_raises(self) -> None:
        with pytest.raises(ValueError, match="max_batch_size"):
            MobileCombustionConfig(max_batch_size=1_500_000)


# =========================================================================
# TestNormalization - 5 tests
# =========================================================================


class TestNormalization:
    """Verify that string fields are normalized to uppercase."""

    def test_log_level_normalized_to_uppercase(self) -> None:
        cfg = MobileCombustionConfig(log_level="debug")
        assert cfg.log_level == "DEBUG"

    def test_gwp_source_normalized_to_uppercase(self) -> None:
        cfg = MobileCombustionConfig(default_gwp_source="ar5")
        assert cfg.default_gwp_source == "AR5"

    def test_calculation_method_normalized_to_uppercase(self) -> None:
        cfg = MobileCombustionConfig(default_calculation_method="distance_based")
        assert cfg.default_calculation_method == "DISTANCE_BASED"

    def test_distance_unit_normalized_to_uppercase(self) -> None:
        cfg = MobileCombustionConfig(default_distance_unit="miles")
        assert cfg.default_distance_unit == "MILES"

    def test_fuel_economy_unit_normalized_to_uppercase(self) -> None:
        cfg = MobileCombustionConfig(default_fuel_economy_unit="mpg_us")
        assert cfg.default_fuel_economy_unit == "MPG_US"

    def test_regulatory_framework_normalized_to_uppercase(self) -> None:
        cfg = MobileCombustionConfig(default_regulatory_framework="iso_14064")
        assert cfg.default_regulatory_framework == "ISO_14064"


# =========================================================================
# TestSingleton - 6 tests
# =========================================================================


class TestSingleton:
    """Verify singleton management via get_config, set_config, reset_config."""

    def test_get_config_returns_instance(self) -> None:
        cfg = get_config()
        assert isinstance(cfg, MobileCombustionConfig)

    def test_get_config_returns_same_instance(self) -> None:
        cfg1 = get_config()
        cfg2 = get_config()
        assert cfg1 is cfg2

    def test_set_config_replaces_singleton(self) -> None:
        custom = MobileCombustionConfig(default_gwp_source="AR4")
        set_config(custom)
        assert get_config().default_gwp_source == "AR4"
        assert get_config() is custom

    def test_reset_config_clears_singleton(self) -> None:
        cfg1 = get_config()
        reset_config()
        cfg2 = get_config()
        assert cfg1 is not cfg2

    def test_set_config_then_reset_returns_fresh(self) -> None:
        set_config(MobileCombustionConfig(default_calculation_method="SPEND_BASED"))
        assert get_config().default_calculation_method == "SPEND_BASED"
        reset_config()
        assert get_config().default_calculation_method == "FUEL_BASED"

    def test_get_config_reads_env_after_reset(self) -> None:
        os.environ["GL_MOBILE_COMBUSTION_DEFAULT_GWP_SOURCE"] = "AR5"
        reset_config()
        cfg = get_config()
        assert cfg.default_gwp_source == "AR5"


# =========================================================================
# TestThreadSafety - 2 tests
# =========================================================================


class TestThreadSafety:
    """Verify thread-safe singleton access."""

    def test_concurrent_get_config_returns_same_instance(self) -> None:
        results = []
        barrier = threading.Barrier(8)

        def _get():
            barrier.wait()
            return get_config()

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(_get) for _ in range(8)]
            results = [f.result() for f in as_completed(futures)]

        first = results[0]
        for cfg in results[1:]:
            assert cfg is first

    def test_concurrent_set_and_get_no_crash(self) -> None:
        errors = []

        def _setter():
            try:
                for _ in range(50):
                    set_config(MobileCombustionConfig())
            except Exception as exc:
                errors.append(exc)

        def _getter():
            try:
                for _ in range(50):
                    cfg = get_config()
                    assert isinstance(cfg, MobileCombustionConfig)
            except Exception as exc:
                errors.append(exc)

        threads = []
        for fn in [_setter, _getter, _setter, _getter]:
            t = threading.Thread(target=fn)
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0, f"Thread errors: {errors}"
