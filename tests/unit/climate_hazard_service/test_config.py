# -*- coding: utf-8 -*-
"""
Unit tests for Climate Hazard Connector configuration module.

Tests ClimateHazardConfig dataclass, from_env(), singleton accessors
(get_config, set_config, reset_config), post_init validation, to_dict
serialization, and all 28 environment variable overrides.

AGENT-DATA-020: Climate Hazard Connector
Target: 85%+ coverage of greenlang.climate_hazard.config
"""

from __future__ import annotations

import os
import threading
from unittest.mock import patch

import pytest

from greenlang.climate_hazard.config import (
    ClimateHazardConfig,
    get_config,
    reset_config,
    set_config,
)


# =============================================================================
# Default values
# =============================================================================


class TestClimateHazardConfigDefaults:
    """Verify every default value on the ClimateHazardConfig dataclass."""

    def test_default_database_url(self, default_config: ClimateHazardConfig) -> None:
        assert default_config.database_url == ""

    def test_default_redis_url(self, default_config: ClimateHazardConfig) -> None:
        assert default_config.redis_url == ""

    def test_default_log_level(self, default_config: ClimateHazardConfig) -> None:
        assert default_config.log_level == "INFO"

    def test_default_scenario(self, default_config: ClimateHazardConfig) -> None:
        assert default_config.default_scenario == "SSP2-4.5"

    def test_default_time_horizon(self, default_config: ClimateHazardConfig) -> None:
        assert default_config.default_time_horizon == "MID_TERM"

    def test_default_report_format(self, default_config: ClimateHazardConfig) -> None:
        assert default_config.default_report_format == "json"

    def test_default_max_hazard_sources(self, default_config: ClimateHazardConfig) -> None:
        assert default_config.max_hazard_sources == 50

    def test_default_max_assets(self, default_config: ClimateHazardConfig) -> None:
        assert default_config.max_assets == 10_000

    def test_default_max_risk_indices(self, default_config: ClimateHazardConfig) -> None:
        assert default_config.max_risk_indices == 5_000

    def test_default_risk_weight_probability(self, default_config: ClimateHazardConfig) -> None:
        assert default_config.risk_weight_probability == 0.30

    def test_default_risk_weight_intensity(self, default_config: ClimateHazardConfig) -> None:
        assert default_config.risk_weight_intensity == 0.30

    def test_default_risk_weight_frequency(self, default_config: ClimateHazardConfig) -> None:
        assert default_config.risk_weight_frequency == 0.25

    def test_default_risk_weight_duration(self, default_config: ClimateHazardConfig) -> None:
        assert default_config.risk_weight_duration == 0.15

    def test_default_vuln_weight_exposure(self, default_config: ClimateHazardConfig) -> None:
        assert default_config.vuln_weight_exposure == 0.40

    def test_default_vuln_weight_sensitivity(self, default_config: ClimateHazardConfig) -> None:
        assert default_config.vuln_weight_sensitivity == 0.35

    def test_default_vuln_weight_adaptive(self, default_config: ClimateHazardConfig) -> None:
        assert default_config.vuln_weight_adaptive == 0.25

    def test_default_threshold_extreme(self, default_config: ClimateHazardConfig) -> None:
        assert default_config.threshold_extreme == 80.0

    def test_default_threshold_high(self, default_config: ClimateHazardConfig) -> None:
        assert default_config.threshold_high == 60.0

    def test_default_threshold_medium(self, default_config: ClimateHazardConfig) -> None:
        assert default_config.threshold_medium == 40.0

    def test_default_threshold_low(self, default_config: ClimateHazardConfig) -> None:
        assert default_config.threshold_low == 20.0

    def test_default_max_pipeline_runs(self, default_config: ClimateHazardConfig) -> None:
        assert default_config.max_pipeline_runs == 500

    def test_default_max_reports(self, default_config: ClimateHazardConfig) -> None:
        assert default_config.max_reports == 1_000

    def test_default_enable_provenance(self, default_config: ClimateHazardConfig) -> None:
        assert default_config.enable_provenance is True

    def test_default_genesis_hash(self, default_config: ClimateHazardConfig) -> None:
        assert default_config.genesis_hash == "greenlang-climate-hazard-genesis"

    def test_default_enable_metrics(self, default_config: ClimateHazardConfig) -> None:
        assert default_config.enable_metrics is True

    def test_default_pool_size(self, default_config: ClimateHazardConfig) -> None:
        assert default_config.pool_size == 5

    def test_default_cache_ttl(self, default_config: ClimateHazardConfig) -> None:
        assert default_config.cache_ttl == 300

    def test_default_rate_limit(self, default_config: ClimateHazardConfig) -> None:
        assert default_config.rate_limit == 200

    def test_risk_weights_sum_to_one(self, default_config: ClimateHazardConfig) -> None:
        total = (
            default_config.risk_weight_probability
            + default_config.risk_weight_intensity
            + default_config.risk_weight_frequency
            + default_config.risk_weight_duration
        )
        assert abs(total - 1.0) < 1e-6

    def test_vuln_weights_sum_to_one(self, default_config: ClimateHazardConfig) -> None:
        total = (
            default_config.vuln_weight_exposure
            + default_config.vuln_weight_sensitivity
            + default_config.vuln_weight_adaptive
        )
        assert abs(total - 1.0) < 1e-6


# =============================================================================
# Post-init validation
# =============================================================================


class TestClimateHazardConfigValidation:
    """Test __post_init__ validation catches invalid values."""

    def test_invalid_log_level_raises(self) -> None:
        with pytest.raises(ValueError, match="log_level"):
            ClimateHazardConfig(log_level="VERBOSE")

    def test_invalid_scenario_raises(self) -> None:
        with pytest.raises(ValueError, match="default_scenario"):
            ClimateHazardConfig(default_scenario="SSP99-9.9")

    def test_invalid_time_horizon_raises(self) -> None:
        with pytest.raises(ValueError, match="default_time_horizon"):
            ClimateHazardConfig(default_time_horizon="ULTRA_LONG")

    def test_invalid_report_format_raises(self) -> None:
        with pytest.raises(ValueError, match="default_report_format"):
            ClimateHazardConfig(default_report_format="xml")

    def test_max_hazard_sources_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="max_hazard_sources"):
            ClimateHazardConfig(max_hazard_sources=0)

    def test_max_hazard_sources_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="max_hazard_sources"):
            ClimateHazardConfig(max_hazard_sources=-1)

    def test_max_hazard_sources_over_limit_raises(self) -> None:
        with pytest.raises(ValueError, match="max_hazard_sources"):
            ClimateHazardConfig(max_hazard_sources=1001)

    def test_max_assets_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="max_assets"):
            ClimateHazardConfig(max_assets=0)

    def test_max_assets_over_limit_raises(self) -> None:
        with pytest.raises(ValueError, match="max_assets"):
            ClimateHazardConfig(max_assets=1_000_001)

    def test_max_risk_indices_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="max_risk_indices"):
            ClimateHazardConfig(max_risk_indices=0)

    def test_risk_weight_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="risk_weight_probability"):
            ClimateHazardConfig(risk_weight_probability=-0.1)

    def test_risk_weight_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="risk_weight_intensity"):
            ClimateHazardConfig(risk_weight_intensity=1.5)

    def test_risk_weights_not_summing_to_one_raises(self) -> None:
        with pytest.raises(ValueError, match="risk index weights must sum to 1.0"):
            ClimateHazardConfig(
                risk_weight_probability=0.5,
                risk_weight_intensity=0.5,
                risk_weight_frequency=0.5,
                risk_weight_duration=0.5,
            )

    def test_vuln_weight_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="vuln_weight_exposure"):
            ClimateHazardConfig(vuln_weight_exposure=-0.1)

    def test_vuln_weights_not_summing_to_one_raises(self) -> None:
        with pytest.raises(ValueError, match="vulnerability weights must sum to 1.0"):
            ClimateHazardConfig(
                vuln_weight_exposure=0.5,
                vuln_weight_sensitivity=0.5,
                vuln_weight_adaptive=0.5,
            )

    def test_threshold_extreme_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="threshold_extreme"):
            ClimateHazardConfig(threshold_extreme=101.0)

    def test_threshold_high_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="threshold_high"):
            ClimateHazardConfig(threshold_high=-1.0)

    def test_threshold_extreme_equal_to_high_raises(self) -> None:
        with pytest.raises(ValueError, match="threshold_extreme"):
            ClimateHazardConfig(threshold_extreme=60.0, threshold_high=60.0)

    def test_threshold_extreme_less_than_high_raises(self) -> None:
        with pytest.raises(ValueError, match="threshold_extreme"):
            ClimateHazardConfig(threshold_extreme=50.0, threshold_high=60.0)

    def test_threshold_high_equal_to_medium_raises(self) -> None:
        with pytest.raises(ValueError, match="threshold_high"):
            ClimateHazardConfig(threshold_high=40.0, threshold_medium=40.0)

    def test_threshold_medium_equal_to_low_raises(self) -> None:
        with pytest.raises(ValueError, match="threshold_medium"):
            ClimateHazardConfig(threshold_medium=20.0, threshold_low=20.0)

    def test_threshold_low_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="threshold_low"):
            ClimateHazardConfig(
                threshold_extreme=80.0,
                threshold_high=60.0,
                threshold_medium=40.0,
                threshold_low=0.0,
            )

    def test_max_pipeline_runs_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="max_pipeline_runs"):
            ClimateHazardConfig(max_pipeline_runs=0)

    def test_max_pipeline_runs_over_limit_raises(self) -> None:
        with pytest.raises(ValueError, match="max_pipeline_runs"):
            ClimateHazardConfig(max_pipeline_runs=10_001)

    def test_max_reports_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="max_reports"):
            ClimateHazardConfig(max_reports=0)

    def test_max_reports_over_limit_raises(self) -> None:
        with pytest.raises(ValueError, match="max_reports"):
            ClimateHazardConfig(max_reports=100_001)

    def test_empty_genesis_hash_raises(self) -> None:
        with pytest.raises(ValueError, match="genesis_hash"):
            ClimateHazardConfig(genesis_hash="")

    def test_pool_size_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="pool_size"):
            ClimateHazardConfig(pool_size=0)

    def test_cache_ttl_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="cache_ttl"):
            ClimateHazardConfig(cache_ttl=0)

    def test_rate_limit_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="rate_limit"):
            ClimateHazardConfig(rate_limit=0)

    def test_multiple_errors_collected(self) -> None:
        """Validation collects all errors, not just the first."""
        with pytest.raises(ValueError) as exc_info:
            ClimateHazardConfig(
                max_hazard_sources=0,
                max_assets=0,
                pool_size=0,
            )
        msg = str(exc_info.value)
        assert "max_hazard_sources" in msg
        assert "max_assets" in msg
        assert "pool_size" in msg


# =============================================================================
# Normalization
# =============================================================================


class TestClimateHazardConfigNormalization:
    """Test that post_init normalises enumerated values."""

    def test_log_level_normalized_to_upper(self) -> None:
        cfg = ClimateHazardConfig(log_level="debug")
        assert cfg.log_level == "DEBUG"

    def test_log_level_mixed_case(self) -> None:
        cfg = ClimateHazardConfig(log_level="Warning")
        assert cfg.log_level == "WARNING"

    def test_time_horizon_normalized_to_upper(self) -> None:
        cfg = ClimateHazardConfig(default_time_horizon="short_term")
        assert cfg.default_time_horizon == "SHORT_TERM"

    def test_report_format_normalized_to_lower(self) -> None:
        cfg = ClimateHazardConfig(default_report_format="JSON")
        assert cfg.default_report_format == "json"

    def test_report_format_mixed_case(self) -> None:
        cfg = ClimateHazardConfig(default_report_format="Csv")
        assert cfg.default_report_format == "csv"


# =============================================================================
# Valid scenarios/horizons/formats
# =============================================================================


class TestClimateHazardConfigValidValues:
    """Test that all valid enumerated values are accepted."""

    @pytest.mark.parametrize("scenario", [
        "SSP1-1.9", "SSP1-2.6", "SSP2-4.5", "SSP3-7.0", "SSP5-8.5",
        "RCP2.6", "RCP4.5", "RCP6.0", "RCP8.5",
    ])
    def test_valid_scenario_accepted(self, scenario: str) -> None:
        cfg = ClimateHazardConfig(default_scenario=scenario)
        assert cfg.default_scenario == scenario

    @pytest.mark.parametrize("horizon", ["SHORT_TERM", "MID_TERM", "LONG_TERM"])
    def test_valid_time_horizon_accepted(self, horizon: str) -> None:
        cfg = ClimateHazardConfig(default_time_horizon=horizon)
        assert cfg.default_time_horizon == horizon

    @pytest.mark.parametrize("fmt", ["json", "csv", "pdf"])
    def test_valid_report_format_accepted(self, fmt: str) -> None:
        cfg = ClimateHazardConfig(default_report_format=fmt)
        assert cfg.default_report_format == fmt

    @pytest.mark.parametrize("level", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    def test_valid_log_level_accepted(self, level: str) -> None:
        cfg = ClimateHazardConfig(log_level=level)
        assert cfg.log_level == level


# =============================================================================
# Boundary values
# =============================================================================


class TestClimateHazardConfigBoundaryValues:
    """Test boundary values accepted by validation."""

    def test_max_hazard_sources_at_lower_bound(self) -> None:
        cfg = ClimateHazardConfig(max_hazard_sources=1)
        assert cfg.max_hazard_sources == 1

    def test_max_hazard_sources_at_upper_bound(self) -> None:
        cfg = ClimateHazardConfig(max_hazard_sources=1000)
        assert cfg.max_hazard_sources == 1000

    def test_max_assets_at_lower_bound(self) -> None:
        cfg = ClimateHazardConfig(max_assets=1)
        assert cfg.max_assets == 1

    def test_max_assets_at_upper_bound(self) -> None:
        cfg = ClimateHazardConfig(max_assets=1_000_000)
        assert cfg.max_assets == 1_000_000

    def test_risk_weights_boundary_zero(self) -> None:
        cfg = ClimateHazardConfig(
            risk_weight_probability=0.0,
            risk_weight_intensity=0.0,
            risk_weight_frequency=0.0,
            risk_weight_duration=1.0,
        )
        assert cfg.risk_weight_duration == 1.0

    def test_risk_weights_boundary_one(self) -> None:
        cfg = ClimateHazardConfig(
            risk_weight_probability=1.0,
            risk_weight_intensity=0.0,
            risk_weight_frequency=0.0,
            risk_weight_duration=0.0,
        )
        assert cfg.risk_weight_probability == 1.0

    def test_threshold_extreme_at_100(self) -> None:
        cfg = ClimateHazardConfig(
            threshold_extreme=100.0,
            threshold_high=60.0,
            threshold_medium=40.0,
            threshold_low=20.0,
        )
        assert cfg.threshold_extreme == 100.0

    def test_max_pipeline_runs_at_upper_bound(self) -> None:
        cfg = ClimateHazardConfig(max_pipeline_runs=10_000)
        assert cfg.max_pipeline_runs == 10_000

    def test_max_reports_at_upper_bound(self) -> None:
        cfg = ClimateHazardConfig(max_reports=100_000)
        assert cfg.max_reports == 100_000


# =============================================================================
# to_dict and __repr__
# =============================================================================


class TestClimateHazardConfigSerialization:
    """Test to_dict serialization and __repr__ safety."""

    def test_to_dict_returns_dict(self, default_config: ClimateHazardConfig) -> None:
        d = default_config.to_dict()
        assert isinstance(d, dict)

    def test_to_dict_has_all_fields(self, default_config: ClimateHazardConfig) -> None:
        d = default_config.to_dict()
        expected_keys = {
            "database_url", "redis_url", "log_level",
            "default_scenario", "default_time_horizon", "default_report_format",
            "max_hazard_sources", "max_assets", "max_risk_indices",
            "risk_weight_probability", "risk_weight_intensity",
            "risk_weight_frequency", "risk_weight_duration",
            "vuln_weight_exposure", "vuln_weight_sensitivity", "vuln_weight_adaptive",
            "threshold_extreme", "threshold_high", "threshold_medium", "threshold_low",
            "max_pipeline_runs", "max_reports",
            "enable_provenance", "genesis_hash", "enable_metrics",
            "pool_size", "cache_ttl", "rate_limit",
        }
        assert expected_keys == set(d.keys())

    def test_to_dict_redacts_database_url(self) -> None:
        cfg = ClimateHazardConfig(database_url="postgresql://user:pass@host/db")
        d = cfg.to_dict()
        assert d["database_url"] == "***"

    def test_to_dict_redacts_redis_url(self) -> None:
        cfg = ClimateHazardConfig(redis_url="redis://auth:pass@host:6379/0")
        d = cfg.to_dict()
        assert d["redis_url"] == "***"

    def test_to_dict_empty_urls_not_redacted(self, default_config: ClimateHazardConfig) -> None:
        d = default_config.to_dict()
        assert d["database_url"] == ""
        assert d["redis_url"] == ""

    def test_repr_does_not_leak_credentials(self) -> None:
        cfg = ClimateHazardConfig(
            database_url="postgresql://user:secret@host/db",
            redis_url="redis://auth:secret@host:6379",
        )
        r = repr(cfg)
        assert "secret" not in r
        assert "***" in r

    def test_repr_contains_class_name(self, default_config: ClimateHazardConfig) -> None:
        r = repr(default_config)
        assert r.startswith("ClimateHazardConfig(")


# =============================================================================
# from_env
# =============================================================================


class TestClimateHazardConfigFromEnv:
    """Test ClimateHazardConfig.from_env() reads environment variables."""

    def test_from_env_defaults(self) -> None:
        cfg = ClimateHazardConfig.from_env()
        assert cfg.default_scenario == "SSP2-4.5"
        assert cfg.max_assets == 10_000

    def test_from_env_database_url(self) -> None:
        os.environ["GL_CLIMATE_HAZARD_DATABASE_URL"] = "postgresql://host/db"
        cfg = ClimateHazardConfig.from_env()
        assert cfg.database_url == "postgresql://host/db"

    def test_from_env_redis_url(self) -> None:
        os.environ["GL_CLIMATE_HAZARD_REDIS_URL"] = "redis://host:6379"
        cfg = ClimateHazardConfig.from_env()
        assert cfg.redis_url == "redis://host:6379"

    def test_from_env_log_level(self) -> None:
        os.environ["GL_CLIMATE_HAZARD_LOG_LEVEL"] = "DEBUG"
        cfg = ClimateHazardConfig.from_env()
        assert cfg.log_level == "DEBUG"

    def test_from_env_default_scenario(self) -> None:
        os.environ["GL_CLIMATE_HAZARD_DEFAULT_SCENARIO"] = "SSP5-8.5"
        cfg = ClimateHazardConfig.from_env()
        assert cfg.default_scenario == "SSP5-8.5"

    def test_from_env_default_time_horizon(self) -> None:
        os.environ["GL_CLIMATE_HAZARD_DEFAULT_TIME_HORIZON"] = "LONG_TERM"
        cfg = ClimateHazardConfig.from_env()
        assert cfg.default_time_horizon == "LONG_TERM"

    def test_from_env_default_report_format(self) -> None:
        os.environ["GL_CLIMATE_HAZARD_DEFAULT_REPORT_FORMAT"] = "csv"
        cfg = ClimateHazardConfig.from_env()
        assert cfg.default_report_format == "csv"

    def test_from_env_max_hazard_sources(self) -> None:
        os.environ["GL_CLIMATE_HAZARD_MAX_HAZARD_SOURCES"] = "100"
        cfg = ClimateHazardConfig.from_env()
        assert cfg.max_hazard_sources == 100

    def test_from_env_max_assets(self) -> None:
        os.environ["GL_CLIMATE_HAZARD_MAX_ASSETS"] = "20000"
        cfg = ClimateHazardConfig.from_env()
        assert cfg.max_assets == 20000

    def test_from_env_max_risk_indices(self) -> None:
        os.environ["GL_CLIMATE_HAZARD_MAX_RISK_INDICES"] = "10000"
        cfg = ClimateHazardConfig.from_env()
        assert cfg.max_risk_indices == 10000

    def test_from_env_risk_weight_probability(self) -> None:
        os.environ["GL_CLIMATE_HAZARD_RISK_WEIGHT_PROBABILITY"] = "0.25"
        os.environ["GL_CLIMATE_HAZARD_RISK_WEIGHT_INTENSITY"] = "0.25"
        os.environ["GL_CLIMATE_HAZARD_RISK_WEIGHT_FREQUENCY"] = "0.25"
        os.environ["GL_CLIMATE_HAZARD_RISK_WEIGHT_DURATION"] = "0.25"
        cfg = ClimateHazardConfig.from_env()
        assert cfg.risk_weight_probability == 0.25

    def test_from_env_vuln_weight_exposure(self) -> None:
        os.environ["GL_CLIMATE_HAZARD_VULN_WEIGHT_EXPOSURE"] = "0.5"
        os.environ["GL_CLIMATE_HAZARD_VULN_WEIGHT_SENSITIVITY"] = "0.3"
        os.environ["GL_CLIMATE_HAZARD_VULN_WEIGHT_ADAPTIVE"] = "0.2"
        cfg = ClimateHazardConfig.from_env()
        assert cfg.vuln_weight_exposure == 0.5

    def test_from_env_threshold_extreme(self) -> None:
        os.environ["GL_CLIMATE_HAZARD_THRESHOLD_EXTREME"] = "90.0"
        cfg = ClimateHazardConfig.from_env()
        assert cfg.threshold_extreme == 90.0

    def test_from_env_threshold_high(self) -> None:
        os.environ["GL_CLIMATE_HAZARD_THRESHOLD_HIGH"] = "70.0"
        cfg = ClimateHazardConfig.from_env()
        assert cfg.threshold_high == 70.0

    def test_from_env_threshold_medium(self) -> None:
        os.environ["GL_CLIMATE_HAZARD_THRESHOLD_MEDIUM"] = "50.0"
        cfg = ClimateHazardConfig.from_env()
        assert cfg.threshold_medium == 50.0

    def test_from_env_threshold_low(self) -> None:
        os.environ["GL_CLIMATE_HAZARD_THRESHOLD_LOW"] = "10.0"
        cfg = ClimateHazardConfig.from_env()
        assert cfg.threshold_low == 10.0

    def test_from_env_max_pipeline_runs(self) -> None:
        os.environ["GL_CLIMATE_HAZARD_MAX_PIPELINE_RUNS"] = "2000"
        cfg = ClimateHazardConfig.from_env()
        assert cfg.max_pipeline_runs == 2000

    def test_from_env_max_reports(self) -> None:
        os.environ["GL_CLIMATE_HAZARD_MAX_REPORTS"] = "5000"
        cfg = ClimateHazardConfig.from_env()
        assert cfg.max_reports == 5000

    def test_from_env_enable_provenance_true(self) -> None:
        os.environ["GL_CLIMATE_HAZARD_ENABLE_PROVENANCE"] = "true"
        cfg = ClimateHazardConfig.from_env()
        assert cfg.enable_provenance is True

    def test_from_env_enable_provenance_false(self) -> None:
        os.environ["GL_CLIMATE_HAZARD_ENABLE_PROVENANCE"] = "false"
        cfg = ClimateHazardConfig.from_env()
        assert cfg.enable_provenance is False

    def test_from_env_enable_provenance_one(self) -> None:
        os.environ["GL_CLIMATE_HAZARD_ENABLE_PROVENANCE"] = "1"
        cfg = ClimateHazardConfig.from_env()
        assert cfg.enable_provenance is True

    def test_from_env_enable_provenance_yes(self) -> None:
        os.environ["GL_CLIMATE_HAZARD_ENABLE_PROVENANCE"] = "yes"
        cfg = ClimateHazardConfig.from_env()
        assert cfg.enable_provenance is True

    def test_from_env_enable_provenance_no(self) -> None:
        os.environ["GL_CLIMATE_HAZARD_ENABLE_PROVENANCE"] = "no"
        cfg = ClimateHazardConfig.from_env()
        assert cfg.enable_provenance is False

    def test_from_env_genesis_hash(self) -> None:
        os.environ["GL_CLIMATE_HAZARD_GENESIS_HASH"] = "custom-genesis"
        cfg = ClimateHazardConfig.from_env()
        assert cfg.genesis_hash == "custom-genesis"

    def test_from_env_enable_metrics_true(self) -> None:
        os.environ["GL_CLIMATE_HAZARD_ENABLE_METRICS"] = "true"
        cfg = ClimateHazardConfig.from_env()
        assert cfg.enable_metrics is True

    def test_from_env_enable_metrics_false(self) -> None:
        os.environ["GL_CLIMATE_HAZARD_ENABLE_METRICS"] = "0"
        cfg = ClimateHazardConfig.from_env()
        assert cfg.enable_metrics is False

    def test_from_env_pool_size(self) -> None:
        os.environ["GL_CLIMATE_HAZARD_POOL_SIZE"] = "20"
        cfg = ClimateHazardConfig.from_env()
        assert cfg.pool_size == 20

    def test_from_env_cache_ttl(self) -> None:
        os.environ["GL_CLIMATE_HAZARD_CACHE_TTL"] = "600"
        cfg = ClimateHazardConfig.from_env()
        assert cfg.cache_ttl == 600

    def test_from_env_rate_limit(self) -> None:
        os.environ["GL_CLIMATE_HAZARD_RATE_LIMIT"] = "500"
        cfg = ClimateHazardConfig.from_env()
        assert cfg.rate_limit == 500

    def test_from_env_invalid_int_falls_back_to_default(self) -> None:
        os.environ["GL_CLIMATE_HAZARD_MAX_ASSETS"] = "not_a_number"
        cfg = ClimateHazardConfig.from_env()
        assert cfg.max_assets == 10_000

    def test_from_env_invalid_float_falls_back_to_default(self) -> None:
        os.environ["GL_CLIMATE_HAZARD_RISK_WEIGHT_PROBABILITY"] = "abc"
        cfg = ClimateHazardConfig.from_env()
        assert cfg.risk_weight_probability == 0.30

    def test_from_env_strips_whitespace(self) -> None:
        os.environ["GL_CLIMATE_HAZARD_LOG_LEVEL"] = "  ERROR  "
        cfg = ClimateHazardConfig.from_env()
        assert cfg.log_level == "ERROR"


# =============================================================================
# Singleton accessors
# =============================================================================


class TestClimateHazardConfigSingleton:
    """Test get_config, set_config, reset_config singleton pattern."""

    def test_get_config_returns_config(self) -> None:
        cfg = get_config()
        assert isinstance(cfg, ClimateHazardConfig)

    def test_get_config_returns_same_instance(self) -> None:
        cfg1 = get_config()
        cfg2 = get_config()
        assert cfg1 is cfg2

    def test_set_config_replaces_singleton(self) -> None:
        custom = ClimateHazardConfig(max_assets=500)
        set_config(custom)
        assert get_config().max_assets == 500

    def test_set_config_identity(self) -> None:
        custom = ClimateHazardConfig(default_scenario="SSP5-8.5")
        set_config(custom)
        assert get_config() is custom

    def test_reset_config_clears_singleton(self) -> None:
        cfg1 = get_config()
        reset_config()
        cfg2 = get_config()
        assert cfg1 is not cfg2

    def test_get_config_reads_env_after_reset(self) -> None:
        os.environ["GL_CLIMATE_HAZARD_MAX_ASSETS"] = "999"
        reset_config()
        cfg = get_config()
        assert cfg.max_assets == 999

    def test_get_config_thread_safety(self) -> None:
        """Multiple threads calling get_config should get the same instance."""
        results = []

        def worker():
            results.append(id(get_config()))

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads should have gotten the same instance
        assert len(set(results)) == 1


# =============================================================================
# Custom config
# =============================================================================


class TestClimateHazardConfigCustom:
    """Test creating configs with custom non-default values."""

    def test_custom_config_accepted(self, custom_config: ClimateHazardConfig) -> None:
        assert custom_config.default_scenario == "SSP5-8.5"
        assert custom_config.max_assets == 20000
        assert custom_config.pool_size == 10

    def test_custom_config_to_dict(self, custom_config: ClimateHazardConfig) -> None:
        d = custom_config.to_dict()
        assert d["database_url"] == "***"
        assert d["redis_url"] == "***"
        assert d["default_scenario"] == "SSP5-8.5"
        assert d["max_assets"] == 20000

    def test_equal_weight_distribution(self) -> None:
        cfg = ClimateHazardConfig(
            risk_weight_probability=0.25,
            risk_weight_intensity=0.25,
            risk_weight_frequency=0.25,
            risk_weight_duration=0.25,
        )
        assert cfg.risk_weight_probability == 0.25

    def test_rcp_scenario(self) -> None:
        cfg = ClimateHazardConfig(default_scenario="RCP8.5")
        assert cfg.default_scenario == "RCP8.5"

    def test_pdf_report_format(self) -> None:
        cfg = ClimateHazardConfig(default_report_format="pdf")
        assert cfg.default_report_format == "pdf"
