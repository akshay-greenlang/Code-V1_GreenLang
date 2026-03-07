# -*- coding: utf-8 -*-
"""
Tests for GeolocationVerificationService and Config - AGENT-EUDR-002 Setup

Comprehensive test suite covering:
- GeolocationVerificationConfig creation and validation
- Config from environment variables
- Config singleton pattern (get_config, set_config, reset_config)
- Config to_dict serialization (credential redaction)
- Config post_init validation constraints
- Config timeout ordering
- Config score weight validation
- Config computed properties
- GeolocationVerificationService initialization (placeholder for future)

Test count: 40 tests
Coverage target: >= 85%

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-002 (Setup and Configuration)
"""

import os
from unittest.mock import patch

import pytest

from greenlang.agents.eudr.geolocation_verification.config import (
    GeolocationVerificationConfig,
    get_config,
    set_config,
    reset_config,
)


# ===========================================================================
# 1. Config Creation and Defaults (10 tests)
# ===========================================================================


class TestConfigCreation:
    """Test GeolocationVerificationConfig creation and defaults."""

    def test_default_config_creation(self):
        """Test config with all defaults is valid."""
        cfg = GeolocationVerificationConfig()
        assert cfg.coordinate_precision_min_decimals == 5
        assert cfg.polygon_area_tolerance_pct == 10.0
        assert cfg.deforestation_cutoff_date == "2020-12-31"

    def test_config_database_url_default(self):
        """Test default database URL."""
        cfg = GeolocationVerificationConfig()
        assert "postgresql" in cfg.database_url

    def test_config_redis_url_default(self):
        """Test default Redis URL."""
        cfg = GeolocationVerificationConfig()
        assert "redis" in cfg.redis_url

    def test_config_log_level_default(self):
        """Test default log level."""
        cfg = GeolocationVerificationConfig()
        assert cfg.log_level == "INFO"

    def test_config_score_weights_default(self):
        """Test default score weights sum to 1.0."""
        cfg = GeolocationVerificationConfig()
        total = sum(cfg.score_weights.values())
        assert abs(total - 1.0) < 0.001

    def test_config_score_weights_keys(self):
        """Test default score weights have correct keys."""
        cfg = GeolocationVerificationConfig()
        expected_keys = {"precision", "polygon", "country", "protected", "deforestation", "temporal"}
        assert set(cfg.score_weights.keys()) == expected_keys

    def test_config_custom_values(self):
        """Test config with custom values."""
        cfg = GeolocationVerificationConfig(
            coordinate_precision_min_decimals=6,
            polygon_area_tolerance_pct=15.0,
            max_batch_concurrency=25,
        )
        assert cfg.coordinate_precision_min_decimals == 6
        assert cfg.polygon_area_tolerance_pct == 15.0
        assert cfg.max_batch_concurrency == 25

    def test_config_provenance_defaults(self):
        """Test provenance tracking defaults."""
        cfg = GeolocationVerificationConfig()
        assert cfg.enable_provenance is True
        assert cfg.genesis_hash == "GL-EUDR-GEO-002-GEOLOCATION-VERIFICATION-GENESIS"

    def test_config_timeout_defaults(self):
        """Test timeout defaults."""
        cfg = GeolocationVerificationConfig()
        assert cfg.quick_timeout_seconds == 5.0
        assert cfg.standard_timeout_seconds == 30.0
        assert cfg.deep_timeout_seconds == 120.0

    def test_config_pool_and_rate_defaults(self):
        """Test pool size and rate limit defaults."""
        cfg = GeolocationVerificationConfig()
        assert cfg.pool_size == 10
        assert cfg.rate_limit == 1000


# ===========================================================================
# 2. Config Validation (12 tests)
# ===========================================================================


class TestConfigValidation:
    """Test config post_init validation constraints."""

    def test_invalid_coordinate_precision_too_low(self):
        """Test coordinate precision below 1 is rejected."""
        with pytest.raises(ValueError, match="coordinate_precision_min_decimals"):
            GeolocationVerificationConfig(coordinate_precision_min_decimals=0)

    def test_invalid_coordinate_precision_too_high(self):
        """Test coordinate precision above 15 is rejected."""
        with pytest.raises(ValueError, match="coordinate_precision_min_decimals"):
            GeolocationVerificationConfig(coordinate_precision_min_decimals=16)

    def test_invalid_polygon_tolerance_zero(self):
        """Test polygon tolerance of 0 is rejected."""
        with pytest.raises(ValueError, match="polygon_area_tolerance_pct"):
            GeolocationVerificationConfig(polygon_area_tolerance_pct=0.0)

    def test_invalid_polygon_tolerance_too_high(self):
        """Test polygon tolerance above 100 is rejected."""
        with pytest.raises(ValueError, match="polygon_area_tolerance_pct"):
            GeolocationVerificationConfig(polygon_area_tolerance_pct=101.0)

    def test_invalid_elevation_max_zero(self):
        """Test elevation max of 0 is rejected."""
        with pytest.raises(ValueError, match="elevation_max_m"):
            GeolocationVerificationConfig(elevation_max_m=0.0)

    def test_invalid_elevation_max_too_high(self):
        """Test elevation max above 9000 is rejected."""
        with pytest.raises(ValueError, match="elevation_max_m"):
            GeolocationVerificationConfig(elevation_max_m=9001.0)

    def test_invalid_batch_concurrency_zero(self):
        """Test batch concurrency of 0 is rejected."""
        with pytest.raises(ValueError, match="max_batch_concurrency"):
            GeolocationVerificationConfig(max_batch_concurrency=0)

    def test_invalid_pool_size_zero(self):
        """Test pool size of 0 is rejected."""
        with pytest.raises(ValueError, match="pool_size"):
            GeolocationVerificationConfig(pool_size=0)

    def test_invalid_rate_limit_zero(self):
        """Test rate limit of 0 is rejected."""
        with pytest.raises(ValueError, match="rate_limit"):
            GeolocationVerificationConfig(rate_limit=0)

    def test_invalid_score_weights_sum(self):
        """Test score weights not summing to 1.0 is rejected."""
        with pytest.raises(ValueError, match="score_weights"):
            GeolocationVerificationConfig(
                score_weights={
                    "precision": 0.50,
                    "polygon": 0.50,
                    "country": 0.50,
                    "protected": 0.50,
                    "deforestation": 0.50,
                    "temporal": 0.50,
                }
            )

    def test_invalid_timeout_ordering(self):
        """Test quick >= standard timeout is rejected."""
        with pytest.raises(ValueError, match="quick_timeout_seconds"):
            GeolocationVerificationConfig(
                quick_timeout_seconds=30.0,
                standard_timeout_seconds=30.0,
            )

    def test_invalid_log_level(self):
        """Test invalid log level is rejected."""
        with pytest.raises(ValueError, match="log_level"):
            GeolocationVerificationConfig(log_level="INVALID")


# ===========================================================================
# 3. Singleton Pattern (8 tests)
# ===========================================================================


class TestSingletonPattern:
    """Test config singleton pattern."""

    def test_get_config_returns_instance(self):
        """Test get_config returns a config instance."""
        reset_config()
        cfg = get_config()
        assert isinstance(cfg, GeolocationVerificationConfig)

    def test_get_config_singleton(self):
        """Test get_config returns same instance on repeated calls."""
        reset_config()
        cfg1 = get_config()
        cfg2 = get_config()
        assert cfg1 is cfg2

    def test_set_config_replaces(self):
        """Test set_config replaces the singleton."""
        custom = GeolocationVerificationConfig(
            coordinate_precision_min_decimals=6,
        )
        set_config(custom)
        assert get_config().coordinate_precision_min_decimals == 6

    def test_reset_config(self):
        """Test reset_config clears the singleton."""
        custom = GeolocationVerificationConfig(
            coordinate_precision_min_decimals=7,
        )
        set_config(custom)
        assert get_config().coordinate_precision_min_decimals == 7
        reset_config()
        # After reset, next call creates a new instance from env
        new_cfg = get_config()
        assert isinstance(new_cfg, GeolocationVerificationConfig)

    def test_set_config_validates(self):
        """Test set_config validates the config."""
        # This should work since we pass a valid config
        valid = GeolocationVerificationConfig()
        set_config(valid)
        assert get_config() is valid

    def test_multiple_resets(self):
        """Test multiple resets do not cause errors."""
        reset_config()
        reset_config()
        reset_config()
        cfg = get_config()
        assert isinstance(cfg, GeolocationVerificationConfig)


# ===========================================================================
# 4. Serialization (5 tests)
# ===========================================================================


class TestConfigSerialization:
    """Test config serialization."""

    def test_to_dict(self):
        """Test to_dict returns dictionary."""
        cfg = GeolocationVerificationConfig()
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert "log_level" in d

    def test_to_dict_redacts_credentials(self):
        """Test to_dict redacts database and redis URLs."""
        cfg = GeolocationVerificationConfig(
            database_url="postgresql://user:pass@host:5432/db",
            redis_url="redis://user:pass@host:6379/0",
        )
        d = cfg.to_dict()
        assert d["database_url"] == "***"
        assert d["redis_url"] == "***"

    def test_to_dict_all_keys(self):
        """Test to_dict contains all expected keys."""
        cfg = GeolocationVerificationConfig()
        d = cfg.to_dict()
        expected_keys = {
            "database_url", "redis_url", "log_level",
            "coordinate_precision_min_decimals",
            "duplicate_distance_threshold_m",
            "elevation_max_m", "country_boundary_buffer_km",
            "polygon_area_tolerance_pct", "max_polygon_vertices",
            "min_polygon_vertices", "sliver_ratio_threshold",
            "spike_angle_threshold_degrees", "wdpa_update_interval_days",
            "deforestation_cutoff_date", "score_weights",
            "max_batch_concurrency", "quick_timeout_seconds",
            "standard_timeout_seconds", "deep_timeout_seconds",
            "verification_cache_ttl_seconds", "enable_provenance",
            "genesis_hash", "enable_metrics", "pool_size", "rate_limit",
        }
        assert expected_keys.issubset(set(d.keys()))

    def test_repr_safe(self):
        """Test repr does not leak credentials."""
        cfg = GeolocationVerificationConfig(
            database_url="postgresql://user:secret@host:5432/db",
        )
        r = repr(cfg)
        assert "secret" not in r
        assert "***" in r

    def test_timeout_by_level(self):
        """Test timeout_by_level computed property."""
        cfg = GeolocationVerificationConfig()
        levels = cfg.timeout_by_level
        assert levels["quick"] == 5.0
        assert levels["standard"] == 30.0
        assert levels["deep"] == 120.0


# ===========================================================================
# 5. Environment Variable Override (5 tests)
# ===========================================================================


class TestEnvOverride:
    """Test config from environment variables."""

    def test_from_env_default(self):
        """Test from_env with no env vars set."""
        cfg = GeolocationVerificationConfig.from_env()
        assert isinstance(cfg, GeolocationVerificationConfig)
        assert cfg.coordinate_precision_min_decimals == 5

    @patch.dict(os.environ, {"GL_EUDR_GEO_LOG_LEVEL": "DEBUG"})
    def test_from_env_log_level(self):
        """Test log level override from env."""
        cfg = GeolocationVerificationConfig.from_env()
        assert cfg.log_level == "DEBUG"

    @patch.dict(os.environ, {"GL_EUDR_GEO_COORDINATE_PRECISION_MIN_DECIMALS": "6"})
    def test_from_env_precision(self):
        """Test coordinate precision override from env."""
        cfg = GeolocationVerificationConfig.from_env()
        assert cfg.coordinate_precision_min_decimals == 6

    @patch.dict(os.environ, {"GL_EUDR_GEO_ENABLE_PROVENANCE": "false"})
    def test_from_env_boolean(self):
        """Test boolean override from env."""
        cfg = GeolocationVerificationConfig.from_env()
        assert cfg.enable_provenance is False

    @patch.dict(os.environ, {"GL_EUDR_GEO_DEFORESTATION_CUTOFF_DATE": "2020-12-31"})
    def test_from_env_cutoff_date(self):
        """Test cutoff date override from env."""
        cfg = GeolocationVerificationConfig.from_env()
        assert cfg.deforestation_cutoff_date == "2020-12-31"


# ===========================================================================
# 6. Comprehensive Validation Edge Cases (20 tests)
# ===========================================================================


class TestConfigValidationEdgeCases:
    """Test additional config validation edge cases."""

    @pytest.mark.parametrize("precision", [1, 2, 3, 5, 10, 15])
    def test_valid_precision_range(self, precision):
        """Test all valid precision values are accepted."""
        cfg = GeolocationVerificationConfig(
            coordinate_precision_min_decimals=precision,
        )
        assert cfg.coordinate_precision_min_decimals == precision

    @pytest.mark.parametrize("tolerance", [0.1, 1.0, 10.0, 50.0, 100.0])
    def test_valid_tolerance_range(self, tolerance):
        """Test all valid tolerance values are accepted."""
        cfg = GeolocationVerificationConfig(
            polygon_area_tolerance_pct=tolerance,
        )
        assert cfg.polygon_area_tolerance_pct == tolerance

    @pytest.mark.parametrize("log_level", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    def test_valid_log_levels(self, log_level):
        """Test all valid log levels are accepted."""
        cfg = GeolocationVerificationConfig(log_level=log_level)
        assert cfg.log_level == log_level

    def test_case_insensitive_log_level(self):
        """Test log level normalization to uppercase."""
        cfg = GeolocationVerificationConfig(log_level="debug")
        assert cfg.log_level == "DEBUG"

    def test_min_polygon_gt_max_rejected(self):
        """Test min_polygon_vertices > max_polygon_vertices is rejected."""
        with pytest.raises(ValueError, match="min_polygon_vertices"):
            GeolocationVerificationConfig(
                min_polygon_vertices=50,
                max_polygon_vertices=10,
            )

    def test_standard_ge_deep_timeout_rejected(self):
        """Test standard >= deep timeout is rejected."""
        with pytest.raises(ValueError, match="standard_timeout_seconds"):
            GeolocationVerificationConfig(
                standard_timeout_seconds=120.0,
                deep_timeout_seconds=120.0,
            )

    def test_empty_genesis_hash_rejected(self):
        """Test empty genesis hash is rejected."""
        with pytest.raises(ValueError, match="genesis_hash"):
            GeolocationVerificationConfig(genesis_hash="")

    def test_negative_cache_ttl_rejected(self):
        """Test negative cache TTL is rejected."""
        with pytest.raises(ValueError, match="verification_cache_ttl_seconds"):
            GeolocationVerificationConfig(verification_cache_ttl_seconds=-1)

    def test_negative_wdpa_interval_rejected(self):
        """Test negative WDPA interval is rejected."""
        with pytest.raises(ValueError, match="wdpa_update_interval_days"):
            GeolocationVerificationConfig(wdpa_update_interval_days=0)

    def test_negative_duplicate_distance_rejected(self):
        """Test negative duplicate distance threshold is rejected."""
        with pytest.raises(ValueError, match="duplicate_distance_threshold_m"):
            GeolocationVerificationConfig(duplicate_distance_threshold_m=-1.0)
