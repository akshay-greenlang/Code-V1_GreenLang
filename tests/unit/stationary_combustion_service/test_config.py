# -*- coding: utf-8 -*-
"""
Unit tests for StationaryCombustionConfig - AGENT-MRV-001.

Tests all configuration defaults, validation logic, environment variable
overrides, singleton accessors, and serialization. 50+ tests covering
every config field and validation edge case.

AGENT-MRV-001: Stationary Combustion Agent (GL-MRV-SCOPE1-001)
"""

from __future__ import annotations

import os
import threading
from typing import Any, Dict

import pytest

from greenlang.stationary_combustion.config import (
    StationaryCombustionConfig,
    get_config,
    reset_config,
    set_config,
)


# =============================================================================
# TestStationaryCombustionConfigDefaults - 21 tests for every default value
# =============================================================================


class TestStationaryCombustionConfigDefaults:
    """Verify that every StationaryCombustionConfig field has the documented default."""

    def test_default_database_url(self, default_config: StationaryCombustionConfig):
        assert default_config.database_url == "postgresql://localhost:5432/greenlang"

    def test_default_redis_url(self, default_config: StationaryCombustionConfig):
        assert default_config.redis_url == "redis://localhost:6379/0"

    def test_default_log_level(self, default_config: StationaryCombustionConfig):
        assert default_config.log_level == "INFO"

    def test_default_gwp_source(self, default_config: StationaryCombustionConfig):
        assert default_config.default_gwp_source == "AR6"

    def test_default_tier(self, default_config: StationaryCombustionConfig):
        assert default_config.default_tier == 1

    def test_default_oxidation_factor(self, default_config: StationaryCombustionConfig):
        assert default_config.default_oxidation_factor == 1.0

    def test_default_decimal_precision(self, default_config: StationaryCombustionConfig):
        assert default_config.decimal_precision == 8

    def test_default_max_batch_size(self, default_config: StationaryCombustionConfig):
        assert default_config.max_batch_size == 10_000

    def test_default_max_fuel_types(self, default_config: StationaryCombustionConfig):
        assert default_config.max_fuel_types == 1_000

    def test_default_max_emission_factors(self, default_config: StationaryCombustionConfig):
        assert default_config.max_emission_factors == 10_000

    def test_default_max_equipment_profiles(self, default_config: StationaryCombustionConfig):
        assert default_config.max_equipment_profiles == 5_000

    def test_default_max_calculations(self, default_config: StationaryCombustionConfig):
        assert default_config.max_calculations == 100_000

    def test_default_monte_carlo_iterations(self, default_config: StationaryCombustionConfig):
        assert default_config.monte_carlo_iterations == 5_000

    def test_default_confidence_levels(self, default_config: StationaryCombustionConfig):
        assert default_config.confidence_levels == "90,95,99"

    def test_default_enable_biogenic_tracking(self, default_config: StationaryCombustionConfig):
        assert default_config.enable_biogenic_tracking is True

    def test_default_enable_provenance(self, default_config: StationaryCombustionConfig):
        assert default_config.enable_provenance is True

    def test_default_genesis_hash(self, default_config: StationaryCombustionConfig):
        assert default_config.genesis_hash == "GL-MRV-X-001-STATIONARY-COMBUSTION-GENESIS"

    def test_default_enable_metrics(self, default_config: StationaryCombustionConfig):
        assert default_config.enable_metrics is True

    def test_default_pool_size(self, default_config: StationaryCombustionConfig):
        assert default_config.pool_size == 10

    def test_default_cache_ttl(self, default_config: StationaryCombustionConfig):
        assert default_config.cache_ttl == 3600

    def test_default_rate_limit(self, default_config: StationaryCombustionConfig):
        assert default_config.rate_limit == 1000


# =============================================================================
# TestConfigValidation - __post_init__ validation errors
# =============================================================================


class TestConfigValidation:
    """Verify __post_init__ raises ValueError for invalid field values."""

    def test_invalid_gwp_source_raises(self):
        with pytest.raises(ValueError, match="default_gwp_source"):
            StationaryCombustionConfig(default_gwp_source="AR7")

    def test_invalid_tier_zero_raises(self):
        with pytest.raises(ValueError, match="default_tier"):
            StationaryCombustionConfig(default_tier=0)

    def test_invalid_tier_four_raises(self):
        with pytest.raises(ValueError, match="default_tier"):
            StationaryCombustionConfig(default_tier=4)

    def test_invalid_tier_negative_raises(self):
        with pytest.raises(ValueError, match="default_tier"):
            StationaryCombustionConfig(default_tier=-1)

    def test_negative_oxidation_factor_raises(self):
        with pytest.raises(ValueError, match="default_oxidation_factor"):
            StationaryCombustionConfig(default_oxidation_factor=-0.1)

    def test_oxidation_factor_above_one_raises(self):
        with pytest.raises(ValueError, match="default_oxidation_factor"):
            StationaryCombustionConfig(default_oxidation_factor=1.1)

    def test_negative_decimal_precision_raises(self):
        with pytest.raises(ValueError, match="decimal_precision"):
            StationaryCombustionConfig(decimal_precision=-1)

    def test_decimal_precision_above_20_raises(self):
        with pytest.raises(ValueError, match="decimal_precision"):
            StationaryCombustionConfig(decimal_precision=21)

    def test_negative_max_batch_size_raises(self):
        with pytest.raises(ValueError, match="max_batch_size"):
            StationaryCombustionConfig(max_batch_size=-1)

    def test_zero_max_batch_size_raises(self):
        with pytest.raises(ValueError, match="max_batch_size"):
            StationaryCombustionConfig(max_batch_size=0)

    def test_max_batch_size_above_upper_raises(self):
        with pytest.raises(ValueError, match="max_batch_size"):
            StationaryCombustionConfig(max_batch_size=1_000_001)

    def test_negative_max_fuel_types_raises(self):
        with pytest.raises(ValueError, match="max_fuel_types"):
            StationaryCombustionConfig(max_fuel_types=-5)

    def test_max_fuel_types_above_upper_raises(self):
        with pytest.raises(ValueError, match="max_fuel_types"):
            StationaryCombustionConfig(max_fuel_types=100_001)

    def test_zero_max_emission_factors_raises(self):
        with pytest.raises(ValueError, match="max_emission_factors"):
            StationaryCombustionConfig(max_emission_factors=0)

    def test_max_emission_factors_above_upper_raises(self):
        with pytest.raises(ValueError, match="max_emission_factors"):
            StationaryCombustionConfig(max_emission_factors=1_000_001)

    def test_zero_max_equipment_profiles_raises(self):
        with pytest.raises(ValueError, match="max_equipment_profiles"):
            StationaryCombustionConfig(max_equipment_profiles=0)

    def test_max_equipment_profiles_above_upper_raises(self):
        with pytest.raises(ValueError, match="max_equipment_profiles"):
            StationaryCombustionConfig(max_equipment_profiles=500_001)

    def test_zero_max_calculations_raises(self):
        with pytest.raises(ValueError, match="max_calculations"):
            StationaryCombustionConfig(max_calculations=0)

    def test_max_calculations_above_upper_raises(self):
        with pytest.raises(ValueError, match="max_calculations"):
            StationaryCombustionConfig(max_calculations=10_000_001)

    def test_zero_monte_carlo_iterations_raises(self):
        with pytest.raises(ValueError, match="monte_carlo_iterations"):
            StationaryCombustionConfig(monte_carlo_iterations=0)

    def test_negative_monte_carlo_iterations_raises(self):
        with pytest.raises(ValueError, match="monte_carlo_iterations"):
            StationaryCombustionConfig(monte_carlo_iterations=-100)

    def test_monte_carlo_above_million_raises(self):
        with pytest.raises(ValueError, match="monte_carlo_iterations"):
            StationaryCombustionConfig(monte_carlo_iterations=1_000_001)

    def test_invalid_confidence_levels_non_numeric_raises(self):
        with pytest.raises(ValueError, match="confidence_levels"):
            StationaryCombustionConfig(confidence_levels="abc,def")

    def test_confidence_level_zero_raises(self):
        with pytest.raises(ValueError, match="confidence level"):
            StationaryCombustionConfig(confidence_levels="0,95,99")

    def test_confidence_level_100_raises(self):
        with pytest.raises(ValueError, match="confidence level"):
            StationaryCombustionConfig(confidence_levels="90,100,99")

    def test_confidence_level_negative_raises(self):
        with pytest.raises(ValueError, match="confidence level"):
            StationaryCombustionConfig(confidence_levels="-5,95,99")

    def test_empty_genesis_hash_raises(self):
        with pytest.raises(ValueError, match="genesis_hash"):
            StationaryCombustionConfig(genesis_hash="")

    def test_zero_pool_size_raises(self):
        with pytest.raises(ValueError, match="pool_size"):
            StationaryCombustionConfig(pool_size=0)

    def test_negative_pool_size_raises(self):
        with pytest.raises(ValueError, match="pool_size"):
            StationaryCombustionConfig(pool_size=-1)

    def test_zero_cache_ttl_raises(self):
        with pytest.raises(ValueError, match="cache_ttl"):
            StationaryCombustionConfig(cache_ttl=0)

    def test_negative_cache_ttl_raises(self):
        with pytest.raises(ValueError, match="cache_ttl"):
            StationaryCombustionConfig(cache_ttl=-100)

    def test_zero_rate_limit_raises(self):
        with pytest.raises(ValueError, match="rate_limit"):
            StationaryCombustionConfig(rate_limit=0)

    def test_negative_rate_limit_raises(self):
        with pytest.raises(ValueError, match="rate_limit"):
            StationaryCombustionConfig(rate_limit=-50)

    def test_invalid_log_level_raises(self):
        with pytest.raises(ValueError, match="log_level"):
            StationaryCombustionConfig(log_level="VERBOSE")

    def test_log_level_normalized_to_uppercase(self):
        cfg = StationaryCombustionConfig(log_level="debug")
        assert cfg.log_level == "DEBUG"

    def test_gwp_source_normalized_to_uppercase(self):
        cfg = StationaryCombustionConfig(default_gwp_source="ar5")
        assert cfg.default_gwp_source == "AR5"

    def test_valid_oxidation_factor_zero(self):
        cfg = StationaryCombustionConfig(default_oxidation_factor=0.0)
        assert cfg.default_oxidation_factor == 0.0

    def test_valid_oxidation_factor_one(self):
        cfg = StationaryCombustionConfig(default_oxidation_factor=1.0)
        assert cfg.default_oxidation_factor == 1.0

    def test_valid_decimal_precision_zero(self):
        cfg = StationaryCombustionConfig(decimal_precision=0)
        assert cfg.decimal_precision == 0

    def test_valid_decimal_precision_twenty(self):
        cfg = StationaryCombustionConfig(decimal_precision=20)
        assert cfg.decimal_precision == 20

    def test_multiple_errors_reported_together(self):
        """When multiple validation errors occur, all are reported."""
        with pytest.raises(ValueError) as exc_info:
            StationaryCombustionConfig(
                default_gwp_source="INVALID",
                default_tier=99,
                default_oxidation_factor=-5.0,
                pool_size=0,
            )
        error_message = str(exc_info.value)
        assert "default_gwp_source" in error_message
        assert "default_tier" in error_message
        assert "default_oxidation_factor" in error_message
        assert "pool_size" in error_message


# =============================================================================
# TestConfigFromEnv - from_env() environment variable overrides
# =============================================================================


class TestConfigFromEnv:
    """Verify that from_env() reads and applies all GL_STATIONARY_COMBUSTION_ env vars."""

    def test_from_env_defaults_without_env_vars(self):
        cfg = StationaryCombustionConfig.from_env()
        assert cfg.default_gwp_source == "AR6"
        assert cfg.default_tier == 1

    def test_from_env_database_url(self):
        os.environ["GL_STATIONARY_COMBUSTION_DATABASE_URL"] = "postgresql://custom:5432/db"
        cfg = StationaryCombustionConfig.from_env()
        assert cfg.database_url == "postgresql://custom:5432/db"

    def test_from_env_redis_url(self):
        os.environ["GL_STATIONARY_COMBUSTION_REDIS_URL"] = "redis://custom:6379/2"
        cfg = StationaryCombustionConfig.from_env()
        assert cfg.redis_url == "redis://custom:6379/2"

    def test_from_env_log_level(self):
        os.environ["GL_STATIONARY_COMBUSTION_LOG_LEVEL"] = "DEBUG"
        cfg = StationaryCombustionConfig.from_env()
        assert cfg.log_level == "DEBUG"

    def test_from_env_default_gwp_source(self):
        os.environ["GL_STATIONARY_COMBUSTION_DEFAULT_GWP_SOURCE"] = "AR4"
        cfg = StationaryCombustionConfig.from_env()
        assert cfg.default_gwp_source == "AR4"

    def test_from_env_default_tier(self):
        os.environ["GL_STATIONARY_COMBUSTION_DEFAULT_TIER"] = "3"
        cfg = StationaryCombustionConfig.from_env()
        assert cfg.default_tier == 3

    def test_from_env_default_oxidation_factor(self):
        os.environ["GL_STATIONARY_COMBUSTION_DEFAULT_OXIDATION_FACTOR"] = "0.95"
        cfg = StationaryCombustionConfig.from_env()
        assert cfg.default_oxidation_factor == pytest.approx(0.95)

    def test_from_env_decimal_precision(self):
        os.environ["GL_STATIONARY_COMBUSTION_DECIMAL_PRECISION"] = "12"
        cfg = StationaryCombustionConfig.from_env()
        assert cfg.decimal_precision == 12

    def test_from_env_max_batch_size(self):
        os.environ["GL_STATIONARY_COMBUSTION_MAX_BATCH_SIZE"] = "20000"
        cfg = StationaryCombustionConfig.from_env()
        assert cfg.max_batch_size == 20_000

    def test_from_env_max_fuel_types(self):
        os.environ["GL_STATIONARY_COMBUSTION_MAX_FUEL_TYPES"] = "2000"
        cfg = StationaryCombustionConfig.from_env()
        assert cfg.max_fuel_types == 2_000

    def test_from_env_max_emission_factors(self):
        os.environ["GL_STATIONARY_COMBUSTION_MAX_EMISSION_FACTORS"] = "50000"
        cfg = StationaryCombustionConfig.from_env()
        assert cfg.max_emission_factors == 50_000

    def test_from_env_max_equipment_profiles(self):
        os.environ["GL_STATIONARY_COMBUSTION_MAX_EQUIPMENT_PROFILES"] = "10000"
        cfg = StationaryCombustionConfig.from_env()
        assert cfg.max_equipment_profiles == 10_000

    def test_from_env_max_calculations(self):
        os.environ["GL_STATIONARY_COMBUSTION_MAX_CALCULATIONS"] = "200000"
        cfg = StationaryCombustionConfig.from_env()
        assert cfg.max_calculations == 200_000

    def test_from_env_monte_carlo_iterations(self):
        os.environ["GL_STATIONARY_COMBUSTION_MONTE_CARLO_ITERATIONS"] = "10000"
        cfg = StationaryCombustionConfig.from_env()
        assert cfg.monte_carlo_iterations == 10_000

    def test_from_env_confidence_levels(self):
        os.environ["GL_STATIONARY_COMBUSTION_CONFIDENCE_LEVELS"] = "80,90,95"
        cfg = StationaryCombustionConfig.from_env()
        assert cfg.confidence_levels == "80,90,95"

    def test_from_env_enable_biogenic_tracking_true(self):
        os.environ["GL_STATIONARY_COMBUSTION_ENABLE_BIOGENIC_TRACKING"] = "true"
        cfg = StationaryCombustionConfig.from_env()
        assert cfg.enable_biogenic_tracking is True

    def test_from_env_enable_biogenic_tracking_false(self):
        os.environ["GL_STATIONARY_COMBUSTION_ENABLE_BIOGENIC_TRACKING"] = "false"
        cfg = StationaryCombustionConfig.from_env()
        assert cfg.enable_biogenic_tracking is False

    def test_from_env_enable_provenance_false(self):
        os.environ["GL_STATIONARY_COMBUSTION_ENABLE_PROVENANCE"] = "0"
        cfg = StationaryCombustionConfig.from_env()
        assert cfg.enable_provenance is False

    def test_from_env_genesis_hash(self):
        os.environ["GL_STATIONARY_COMBUSTION_GENESIS_HASH"] = "custom-genesis"
        cfg = StationaryCombustionConfig.from_env()
        assert cfg.genesis_hash == "custom-genesis"

    def test_from_env_enable_metrics_false(self):
        os.environ["GL_STATIONARY_COMBUSTION_ENABLE_METRICS"] = "no"
        cfg = StationaryCombustionConfig.from_env()
        assert cfg.enable_metrics is False

    def test_from_env_pool_size(self):
        os.environ["GL_STATIONARY_COMBUSTION_POOL_SIZE"] = "20"
        cfg = StationaryCombustionConfig.from_env()
        assert cfg.pool_size == 20

    def test_from_env_cache_ttl(self):
        os.environ["GL_STATIONARY_COMBUSTION_CACHE_TTL"] = "7200"
        cfg = StationaryCombustionConfig.from_env()
        assert cfg.cache_ttl == 7200

    def test_from_env_rate_limit(self):
        os.environ["GL_STATIONARY_COMBUSTION_RATE_LIMIT"] = "2000"
        cfg = StationaryCombustionConfig.from_env()
        assert cfg.rate_limit == 2000

    def test_from_env_invalid_int_falls_back_to_default(self):
        """Malformed integer env vars fall back to class default with a warning."""
        os.environ["GL_STATIONARY_COMBUSTION_DEFAULT_TIER"] = "not_a_number"
        cfg = StationaryCombustionConfig.from_env()
        assert cfg.default_tier == 1  # class default

    def test_from_env_invalid_float_falls_back_to_default(self):
        """Malformed float env vars fall back to class default with a warning."""
        os.environ["GL_STATIONARY_COMBUSTION_DEFAULT_OXIDATION_FACTOR"] = "abc"
        cfg = StationaryCombustionConfig.from_env()
        assert cfg.default_oxidation_factor == 1.0  # class default

    def test_from_env_bool_yes_accepted(self):
        os.environ["GL_STATIONARY_COMBUSTION_ENABLE_METRICS"] = "yes"
        cfg = StationaryCombustionConfig.from_env()
        assert cfg.enable_metrics is True

    def test_from_env_bool_1_accepted(self):
        os.environ["GL_STATIONARY_COMBUSTION_ENABLE_METRICS"] = "1"
        cfg = StationaryCombustionConfig.from_env()
        assert cfg.enable_metrics is True

    def test_from_env_whitespace_stripped(self):
        os.environ["GL_STATIONARY_COMBUSTION_LOG_LEVEL"] = "  WARNING  "
        cfg = StationaryCombustionConfig.from_env()
        assert cfg.log_level == "WARNING"


# =============================================================================
# TestConfigSingleton - get_config / set_config / reset_config / thread safety
# =============================================================================


class TestConfigSingleton:
    """Test singleton pattern for StationaryCombustionConfig."""

    def test_get_config_returns_instance(self):
        cfg = get_config()
        assert isinstance(cfg, StationaryCombustionConfig)

    def test_get_config_returns_same_instance(self):
        cfg1 = get_config()
        cfg2 = get_config()
        assert cfg1 is cfg2

    def test_set_config_replaces_singleton(self):
        custom = StationaryCombustionConfig(default_tier=3)
        set_config(custom)
        assert get_config().default_tier == 3

    def test_reset_config_clears_singleton(self):
        custom = StationaryCombustionConfig(default_tier=3)
        set_config(custom)
        assert get_config().default_tier == 3

        reset_config()
        fresh = get_config()
        assert fresh.default_tier == 1  # re-read from env (default)

    def test_set_config_then_get_returns_set_instance(self):
        custom = StationaryCombustionConfig(
            default_gwp_source="AR4",
            default_tier=2,
        )
        set_config(custom)
        cfg = get_config()
        assert cfg.default_gwp_source == "AR4"
        assert cfg.default_tier == 2

    def test_thread_safety_get_config(self):
        """Multiple threads calling get_config concurrently return the same instance."""
        instances = []
        barrier = threading.Barrier(10)

        def worker():
            barrier.wait()
            cfg = get_config()
            instances.append(id(cfg))

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads must have received the same singleton object
        assert len(set(instances)) == 1


# =============================================================================
# TestConfigToDict - to_dict() serialization with credential redaction
# =============================================================================


class TestConfigToDict:
    """Test to_dict() serialization."""

    def test_to_dict_returns_dict(self, default_config: StationaryCombustionConfig):
        d = default_config.to_dict()
        assert isinstance(d, dict)

    def test_to_dict_redacts_database_url(self, default_config: StationaryCombustionConfig):
        d = default_config.to_dict()
        assert d["database_url"] == "***"

    def test_to_dict_redacts_redis_url(self, default_config: StationaryCombustionConfig):
        d = default_config.to_dict()
        assert d["redis_url"] == "***"

    def test_to_dict_includes_all_21_keys(self, default_config: StationaryCombustionConfig):
        d = default_config.to_dict()
        expected_keys = {
            "database_url", "redis_url", "log_level", "default_gwp_source",
            "default_tier", "default_oxidation_factor", "decimal_precision",
            "max_batch_size", "max_fuel_types", "max_emission_factors",
            "max_equipment_profiles", "max_calculations", "monte_carlo_iterations",
            "confidence_levels", "enable_biogenic_tracking", "enable_provenance",
            "genesis_hash", "enable_metrics", "pool_size", "cache_ttl",
            "rate_limit",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_values_match_config(self, default_config: StationaryCombustionConfig):
        d = default_config.to_dict()
        assert d["log_level"] == "INFO"
        assert d["default_gwp_source"] == "AR6"
        assert d["default_tier"] == 1
        assert d["default_oxidation_factor"] == 1.0
        assert d["decimal_precision"] == 8
        assert d["max_batch_size"] == 10_000
        assert d["monte_carlo_iterations"] == 5_000
        assert d["enable_biogenic_tracking"] is True
        assert d["enable_provenance"] is True
        assert d["enable_metrics"] is True
        assert d["pool_size"] == 10
        assert d["cache_ttl"] == 3600
        assert d["rate_limit"] == 1000

    def test_repr_is_credential_safe(self, default_config: StationaryCombustionConfig):
        r = repr(default_config)
        assert "StationaryCombustionConfig(" in r
        assert "postgresql://" not in r
        assert "redis://" not in r
        assert "***" in r

    def test_to_dict_custom_config(self, custom_config: StationaryCombustionConfig):
        d = custom_config.to_dict()
        assert d["default_gwp_source"] == "AR5"
        assert d["default_tier"] == 2
        assert d["default_oxidation_factor"] == pytest.approx(0.99)
        assert d["decimal_precision"] == 10
        assert d["max_batch_size"] == 5_000
        assert d["pool_size"] == 5
        assert d["cache_ttl"] == 1800
        assert d["rate_limit"] == 500
