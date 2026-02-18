# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-MRV-005 Fugitive Emissions Agent Configuration.

Tests the FugitiveEmissionsConfig dataclass, from_env() factory,
singleton accessor (get_config/set_config/reset_config), validation,
boolean/numeric field parsing, serialisation helpers, and edge cases.

Test Classes:
    - TestConfigDefaults            (15 tests)
    - TestConfigFromEnv             (20 tests)
    - TestConfigSingleton           (10 tests)
    - TestConfigValidation          (15 tests)
    - TestConfigBoolFields          (10 tests)
    - TestConfigNumericFields       (10 tests)
    - TestConfigStringFields        (10 tests)
    - TestConfigSerialisation       (10 tests)
    - TestConfigEdgeCases           (10 tests)
    - TestConfigLogLevel            (10 tests)
    - TestConfigMethodologyEnums    (10 tests)

Total: 120+ tests.

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-005 Fugitive Emissions (GL-MRV-SCOPE1-005)
"""

from __future__ import annotations

import os
import threading
from unittest.mock import patch

import pytest

from greenlang.fugitive_emissions.config import (
    FugitiveEmissionsConfig,
    get_config,
    set_config,
    reset_config,
    _ENV_PREFIX,
    _VALID_GWP_SOURCES,
    _VALID_CALCULATION_METHODS,
    _VALID_EF_SOURCES,
    _VALID_LOG_LEVELS,
)


# ==========================================================================
# TestConfigDefaults - 15 tests
# ==========================================================================


class TestConfigDefaults:
    """Verify all default configuration values match specification."""

    def test_default_enabled(self, default_config):
        assert default_config.enabled is True

    def test_default_database_url(self, default_config):
        assert default_config.database_url == ""

    def test_default_redis_url(self, default_config):
        assert default_config.redis_url == ""

    def test_default_max_batch_size(self, default_config):
        assert default_config.max_batch_size == 500

    def test_default_gwp_source(self, default_config):
        assert default_config.default_gwp_source == "AR6"

    def test_default_calculation_method(self, default_config):
        assert default_config.default_calculation_method == "AVERAGE_EMISSION_FACTOR"

    def test_default_emission_factor_source(self, default_config):
        assert default_config.default_emission_factor_source == "EPA"

    def test_default_decimal_precision(self, default_config):
        assert default_config.decimal_precision == 8

    def test_default_monte_carlo_iterations(self, default_config):
        assert default_config.monte_carlo_iterations == 5000

    def test_default_monte_carlo_seed(self, default_config):
        assert default_config.monte_carlo_seed == 42

    def test_default_confidence_levels(self, default_config):
        assert default_config.confidence_levels == "90,95,99"

    def test_default_feature_toggles_all_true(self, default_config):
        assert default_config.enable_ldar_tracking is True
        assert default_config.enable_component_tracking is True
        assert default_config.enable_coal_mine_methane is True
        assert default_config.enable_wastewater is True
        assert default_config.enable_tank_losses is True
        assert default_config.enable_pneumatic_devices is True
        assert default_config.enable_compliance_checking is True
        assert default_config.enable_uncertainty is True
        assert default_config.enable_provenance is True
        assert default_config.enable_metrics is True

    def test_default_max_components(self, default_config):
        assert default_config.max_components == 5000

    def test_default_max_surveys(self, default_config):
        assert default_config.max_surveys == 1000

    def test_default_ldar_leak_threshold_ppm(self, default_config):
        assert default_config.ldar_leak_threshold_ppm == 10000

    def test_default_cache_ttl_seconds(self, default_config):
        assert default_config.cache_ttl_seconds == 3600

    def test_default_api_prefix(self, default_config):
        assert default_config.api_prefix == "/api/v1/fugitive-emissions"

    def test_default_api_page_sizes(self, default_config):
        assert default_config.api_max_page_size == 100
        assert default_config.api_default_page_size == 20

    def test_default_log_level(self, default_config):
        assert default_config.log_level == "INFO"

    def test_default_worker_threads(self, default_config):
        assert default_config.worker_threads == 4

    def test_default_background_tasks(self, default_config):
        assert default_config.enable_background_tasks is True

    def test_default_health_check_interval(self, default_config):
        assert default_config.health_check_interval == 30

    def test_default_genesis_hash(self, default_config):
        assert default_config.genesis_hash == "GL-MRV-X-005-FUGITIVE-EMISSIONS-GENESIS"


# ==========================================================================
# TestConfigFromEnv - 20 tests
# ==========================================================================


class TestConfigFromEnv:
    """Test FugitiveEmissionsConfig.from_env() reads environment variables."""

    def test_from_env_defaults_without_env_vars(self):
        cfg = FugitiveEmissionsConfig.from_env()
        assert cfg.default_gwp_source == "AR6"
        assert cfg.max_batch_size == 500

    def test_from_env_enabled_false(self, monkeypatch):
        monkeypatch.setenv(f"{_ENV_PREFIX}ENABLED", "false")
        cfg = FugitiveEmissionsConfig.from_env()
        assert cfg.enabled is False

    @pytest.mark.parametrize("val,expected", [
        ("true", True), ("1", True), ("yes", True),
        ("false", False), ("0", False), ("no", False),
    ])
    def test_from_env_bool_parsing(self, monkeypatch, val, expected):
        monkeypatch.setenv(f"{_ENV_PREFIX}ENABLE_LDAR_TRACKING", val)
        cfg = FugitiveEmissionsConfig.from_env()
        assert cfg.enable_ldar_tracking is expected

    def test_from_env_max_batch_size(self, monkeypatch):
        monkeypatch.setenv(f"{_ENV_PREFIX}MAX_BATCH_SIZE", "1000")
        cfg = FugitiveEmissionsConfig.from_env()
        assert cfg.max_batch_size == 1000

    def test_from_env_gwp_source(self, monkeypatch):
        monkeypatch.setenv(f"{_ENV_PREFIX}DEFAULT_GWP_SOURCE", "AR5")
        cfg = FugitiveEmissionsConfig.from_env()
        assert cfg.default_gwp_source == "AR5"

    def test_from_env_calculation_method(self, monkeypatch):
        monkeypatch.setenv(
            f"{_ENV_PREFIX}DEFAULT_CALCULATION_METHOD", "DIRECT_MEASUREMENT"
        )
        cfg = FugitiveEmissionsConfig.from_env()
        assert cfg.default_calculation_method == "DIRECT_MEASUREMENT"

    def test_from_env_ef_source(self, monkeypatch):
        monkeypatch.setenv(f"{_ENV_PREFIX}DEFAULT_EMISSION_FACTOR_SOURCE", "IPCC")
        cfg = FugitiveEmissionsConfig.from_env()
        assert cfg.default_emission_factor_source == "IPCC"

    def test_from_env_decimal_precision(self, monkeypatch):
        monkeypatch.setenv(f"{_ENV_PREFIX}DECIMAL_PRECISION", "12")
        cfg = FugitiveEmissionsConfig.from_env()
        assert cfg.decimal_precision == 12

    def test_from_env_monte_carlo_iterations(self, monkeypatch):
        monkeypatch.setenv(f"{_ENV_PREFIX}MONTE_CARLO_ITERATIONS", "10000")
        cfg = FugitiveEmissionsConfig.from_env()
        assert cfg.monte_carlo_iterations == 10000

    def test_from_env_monte_carlo_seed(self, monkeypatch):
        monkeypatch.setenv(f"{_ENV_PREFIX}MONTE_CARLO_SEED", "0")
        cfg = FugitiveEmissionsConfig.from_env()
        assert cfg.monte_carlo_seed == 0

    def test_from_env_confidence_levels(self, monkeypatch):
        monkeypatch.setenv(f"{_ENV_PREFIX}CONFIDENCE_LEVELS", "80,90,95,99")
        cfg = FugitiveEmissionsConfig.from_env()
        assert cfg.confidence_levels == "80,90,95,99"

    def test_from_env_database_url(self, monkeypatch):
        monkeypatch.setenv(
            f"{_ENV_PREFIX}DATABASE_URL",
            "postgresql://user:pass@host:5432/db",
        )
        cfg = FugitiveEmissionsConfig.from_env()
        assert cfg.database_url == "postgresql://user:pass@host:5432/db"

    def test_from_env_redis_url(self, monkeypatch):
        monkeypatch.setenv(f"{_ENV_PREFIX}REDIS_URL", "redis://host:6379/0")
        cfg = FugitiveEmissionsConfig.from_env()
        assert cfg.redis_url == "redis://host:6379/0"

    def test_from_env_api_prefix(self, monkeypatch):
        monkeypatch.setenv(f"{_ENV_PREFIX}API_PREFIX", "/v2/fugitive")
        cfg = FugitiveEmissionsConfig.from_env()
        assert cfg.api_prefix == "/v2/fugitive"

    def test_from_env_log_level(self, monkeypatch):
        monkeypatch.setenv(f"{_ENV_PREFIX}LOG_LEVEL", "DEBUG")
        cfg = FugitiveEmissionsConfig.from_env()
        assert cfg.log_level == "DEBUG"

    def test_from_env_worker_threads(self, monkeypatch):
        monkeypatch.setenv(f"{_ENV_PREFIX}WORKER_THREADS", "8")
        cfg = FugitiveEmissionsConfig.from_env()
        assert cfg.worker_threads == 8

    def test_from_env_genesis_hash(self, monkeypatch):
        monkeypatch.setenv(f"{_ENV_PREFIX}GENESIS_HASH", "CUSTOM-HASH")
        cfg = FugitiveEmissionsConfig.from_env()
        assert cfg.genesis_hash == "CUSTOM-HASH"

    def test_from_env_invalid_int_falls_back(self, monkeypatch):
        monkeypatch.setenv(f"{_ENV_PREFIX}MAX_BATCH_SIZE", "not_a_number")
        cfg = FugitiveEmissionsConfig.from_env()
        assert cfg.max_batch_size == 500  # default

    def test_from_env_strips_whitespace(self, monkeypatch):
        monkeypatch.setenv(f"{_ENV_PREFIX}DEFAULT_GWP_SOURCE", "  AR5  ")
        cfg = FugitiveEmissionsConfig.from_env()
        assert cfg.default_gwp_source == "AR5"

    def test_from_env_case_insensitive_bool(self, monkeypatch):
        monkeypatch.setenv(f"{_ENV_PREFIX}ENABLE_METRICS", "TRUE")
        cfg = FugitiveEmissionsConfig.from_env()
        assert cfg.enable_metrics is True

    def test_from_env_ldar_threshold(self, monkeypatch):
        monkeypatch.setenv(f"{_ENV_PREFIX}LDAR_LEAK_THRESHOLD_PPM", "5000")
        cfg = FugitiveEmissionsConfig.from_env()
        assert cfg.ldar_leak_threshold_ppm == 5000

    def test_from_env_max_components(self, monkeypatch):
        monkeypatch.setenv(f"{_ENV_PREFIX}MAX_COMPONENTS", "10000")
        cfg = FugitiveEmissionsConfig.from_env()
        assert cfg.max_components == 10000

    def test_from_env_max_surveys(self, monkeypatch):
        monkeypatch.setenv(f"{_ENV_PREFIX}MAX_SURVEYS", "2000")
        cfg = FugitiveEmissionsConfig.from_env()
        assert cfg.max_surveys == 2000

    def test_from_env_cache_ttl(self, monkeypatch):
        monkeypatch.setenv(f"{_ENV_PREFIX}CACHE_TTL_SECONDS", "7200")
        cfg = FugitiveEmissionsConfig.from_env()
        assert cfg.cache_ttl_seconds == 7200

    def test_from_env_health_check_interval(self, monkeypatch):
        monkeypatch.setenv(f"{_ENV_PREFIX}HEALTH_CHECK_INTERVAL", "120")
        cfg = FugitiveEmissionsConfig.from_env()
        assert cfg.health_check_interval == 120

    def test_from_env_api_max_page_size(self, monkeypatch):
        monkeypatch.setenv(f"{_ENV_PREFIX}API_MAX_PAGE_SIZE", "200")
        cfg = FugitiveEmissionsConfig.from_env()
        assert cfg.api_max_page_size == 200

    def test_from_env_api_default_page_size(self, monkeypatch):
        monkeypatch.setenv(f"{_ENV_PREFIX}API_DEFAULT_PAGE_SIZE", "50")
        cfg = FugitiveEmissionsConfig.from_env()
        assert cfg.api_default_page_size == 50

    def test_from_env_enable_background_tasks_false(self, monkeypatch):
        monkeypatch.setenv(f"{_ENV_PREFIX}ENABLE_BACKGROUND_TASKS", "false")
        cfg = FugitiveEmissionsConfig.from_env()
        assert cfg.enable_background_tasks is False


# ==========================================================================
# TestConfigSingleton - 10 tests
# ==========================================================================


class TestConfigSingleton:
    """Test get_config / set_config / reset_config singleton pattern."""

    def test_get_config_returns_instance(self):
        cfg = get_config()
        assert isinstance(cfg, FugitiveEmissionsConfig)

    def test_get_config_returns_same_instance(self):
        cfg1 = get_config()
        cfg2 = get_config()
        assert cfg1 is cfg2

    def test_set_config_replaces_instance(self):
        custom = FugitiveEmissionsConfig(max_batch_size=99)
        set_config(custom)
        cfg = get_config()
        assert cfg.max_batch_size == 99

    def test_reset_config_clears_singleton(self):
        get_config()
        reset_config()
        # After reset the next call should create a fresh instance
        cfg = get_config()
        assert isinstance(cfg, FugitiveEmissionsConfig)

    def test_set_then_reset_reverts_to_defaults(self):
        custom = FugitiveEmissionsConfig(max_batch_size=1)
        set_config(custom)
        assert get_config().max_batch_size == 1
        reset_config()
        cfg = get_config()
        assert cfg.max_batch_size == 500

    def test_get_config_thread_safe(self):
        results = []

        def worker():
            cfg = get_config()
            results.append(id(cfg))

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads should see the same singleton
        assert len(set(results)) == 1

    def test_set_config_thread_safe(self):
        configs = []
        for i in range(5):
            configs.append(
                FugitiveEmissionsConfig(worker_threads=i + 1)
            )

        def worker(cfg):
            set_config(cfg)

        threads = [
            threading.Thread(target=worker, args=(c,)) for c in configs
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have one of the configs installed
        final = get_config()
        assert final.worker_threads in range(1, 6)

    def test_multiple_reset_cycles(self):
        for _ in range(5):
            cfg = get_config()
            assert cfg.enabled is True
            reset_config()

    def test_set_config_preserves_validation(self):
        valid = FugitiveEmissionsConfig(decimal_precision=10)
        set_config(valid)
        assert get_config().decimal_precision == 10

    def test_reset_config_idempotent(self):
        reset_config()
        reset_config()
        reset_config()
        cfg = get_config()
        assert isinstance(cfg, FugitiveEmissionsConfig)


# ==========================================================================
# TestConfigValidation - 15 tests
# ==========================================================================


class TestConfigValidation:
    """Test __post_init__ validation catches all invalid values."""

    def test_invalid_gwp_source_raises(self):
        with pytest.raises(ValueError, match="default_gwp_source"):
            FugitiveEmissionsConfig(default_gwp_source="AR99")

    def test_invalid_calculation_method_raises(self):
        with pytest.raises(ValueError, match="default_calculation_method"):
            FugitiveEmissionsConfig(
                default_calculation_method="INVALID_METHOD"
            )

    def test_invalid_ef_source_raises(self):
        with pytest.raises(ValueError, match="default_emission_factor_source"):
            FugitiveEmissionsConfig(
                default_emission_factor_source="UNKNOWN"
            )

    def test_invalid_log_level_raises(self):
        with pytest.raises(ValueError, match="log_level"):
            FugitiveEmissionsConfig(log_level="VERBOSE")

    def test_decimal_precision_negative_raises(self):
        with pytest.raises(ValueError, match="decimal_precision"):
            FugitiveEmissionsConfig(decimal_precision=-1)

    def test_decimal_precision_too_high_raises(self):
        with pytest.raises(ValueError, match="decimal_precision"):
            FugitiveEmissionsConfig(decimal_precision=21)

    def test_max_batch_size_zero_raises(self):
        with pytest.raises(ValueError, match="max_batch_size"):
            FugitiveEmissionsConfig(max_batch_size=0)

    def test_max_batch_size_exceeds_limit_raises(self):
        with pytest.raises(ValueError, match="max_batch_size"):
            FugitiveEmissionsConfig(max_batch_size=100_001)

    def test_ldar_threshold_zero_raises(self):
        with pytest.raises(ValueError, match="ldar_leak_threshold_ppm"):
            FugitiveEmissionsConfig(ldar_leak_threshold_ppm=0)

    def test_ldar_threshold_exceeds_limit_raises(self):
        with pytest.raises(ValueError, match="ldar_leak_threshold_ppm"):
            FugitiveEmissionsConfig(ldar_leak_threshold_ppm=100_001)

    def test_monte_carlo_iterations_zero_raises(self):
        with pytest.raises(ValueError, match="monte_carlo_iterations"):
            FugitiveEmissionsConfig(monte_carlo_iterations=0)

    def test_monte_carlo_iterations_exceeds_limit_raises(self):
        with pytest.raises(ValueError, match="monte_carlo_iterations"):
            FugitiveEmissionsConfig(monte_carlo_iterations=1_000_001)

    def test_monte_carlo_seed_negative_raises(self):
        with pytest.raises(ValueError, match="monte_carlo_seed"):
            FugitiveEmissionsConfig(monte_carlo_seed=-1)

    def test_invalid_confidence_levels_raises(self):
        with pytest.raises(ValueError, match="confidence"):
            FugitiveEmissionsConfig(confidence_levels="abc,def")

    def test_confidence_levels_out_of_range_raises(self):
        with pytest.raises(ValueError, match="confidence level"):
            FugitiveEmissionsConfig(confidence_levels="0,50,100")

    def test_genesis_hash_empty_raises(self):
        with pytest.raises(ValueError, match="genesis_hash"):
            FugitiveEmissionsConfig(genesis_hash="")

    def test_cache_ttl_zero_raises(self):
        with pytest.raises(ValueError, match="cache_ttl_seconds"):
            FugitiveEmissionsConfig(cache_ttl_seconds=0)

    def test_worker_threads_zero_raises(self):
        with pytest.raises(ValueError, match="worker_threads"):
            FugitiveEmissionsConfig(worker_threads=0)

    def test_worker_threads_exceeds_limit_raises(self):
        with pytest.raises(ValueError, match="worker_threads"):
            FugitiveEmissionsConfig(worker_threads=65)

    def test_api_default_page_size_exceeds_max_raises(self):
        with pytest.raises(ValueError, match="api_default_page_size"):
            FugitiveEmissionsConfig(
                api_max_page_size=10, api_default_page_size=20
            )

    def test_health_check_interval_zero_raises(self):
        with pytest.raises(ValueError, match="health_check_interval"):
            FugitiveEmissionsConfig(health_check_interval=0)

    def test_api_max_page_size_zero_raises(self):
        with pytest.raises(ValueError, match="api_max_page_size"):
            FugitiveEmissionsConfig(api_max_page_size=0)

    def test_api_default_page_size_zero_raises(self):
        with pytest.raises(ValueError, match="api_default_page_size"):
            FugitiveEmissionsConfig(api_default_page_size=0)

    def test_max_batch_size_negative_raises(self):
        with pytest.raises(ValueError, match="max_batch_size"):
            FugitiveEmissionsConfig(max_batch_size=-10)

    def test_cache_ttl_negative_raises(self):
        with pytest.raises(ValueError, match="cache_ttl_seconds"):
            FugitiveEmissionsConfig(cache_ttl_seconds=-100)


# ==========================================================================
# TestConfigBoolFields - 10 tests
# ==========================================================================


class TestConfigBoolFields:
    """Test all boolean field overrides and parsing."""

    @pytest.mark.parametrize("field_name,env_suffix", [
        ("enable_ldar_tracking", "ENABLE_LDAR_TRACKING"),
        ("enable_component_tracking", "ENABLE_COMPONENT_TRACKING"),
        ("enable_coal_mine_methane", "ENABLE_COAL_MINE_METHANE"),
        ("enable_wastewater", "ENABLE_WASTEWATER"),
        ("enable_tank_losses", "ENABLE_TANK_LOSSES"),
        ("enable_pneumatic_devices", "ENABLE_PNEUMATIC_DEVICES"),
        ("enable_compliance_checking", "ENABLE_COMPLIANCE_CHECKING"),
        ("enable_uncertainty", "ENABLE_UNCERTAINTY"),
        ("enable_provenance", "ENABLE_PROVENANCE"),
        ("enable_metrics", "ENABLE_METRICS"),
    ])
    def test_bool_field_disabled_via_env(self, monkeypatch, field_name, env_suffix):
        monkeypatch.setenv(f"{_ENV_PREFIX}{env_suffix}", "false")
        cfg = FugitiveEmissionsConfig.from_env()
        assert getattr(cfg, field_name) is False

    @pytest.mark.parametrize("field_name,env_suffix", [
        ("enable_ldar_tracking", "ENABLE_LDAR_TRACKING"),
        ("enable_component_tracking", "ENABLE_COMPONENT_TRACKING"),
        ("enable_coal_mine_methane", "ENABLE_COAL_MINE_METHANE"),
        ("enable_wastewater", "ENABLE_WASTEWATER"),
        ("enable_tank_losses", "ENABLE_TANK_LOSSES"),
        ("enable_pneumatic_devices", "ENABLE_PNEUMATIC_DEVICES"),
        ("enable_compliance_checking", "ENABLE_COMPLIANCE_CHECKING"),
        ("enable_uncertainty", "ENABLE_UNCERTAINTY"),
        ("enable_provenance", "ENABLE_PROVENANCE"),
        ("enable_metrics", "ENABLE_METRICS"),
    ])
    def test_bool_field_enabled_via_env(self, monkeypatch, field_name, env_suffix):
        monkeypatch.setenv(f"{_ENV_PREFIX}{env_suffix}", "true")
        cfg = FugitiveEmissionsConfig.from_env()
        assert getattr(cfg, field_name) is True

    @pytest.mark.parametrize("field_name", [
        "enable_ldar_tracking",
        "enable_component_tracking",
        "enable_coal_mine_methane",
        "enable_wastewater",
        "enable_tank_losses",
        "enable_pneumatic_devices",
        "enable_compliance_checking",
        "enable_uncertainty",
        "enable_provenance",
        "enable_metrics",
    ])
    def test_bool_field_direct_override_false(self, field_name):
        cfg = FugitiveEmissionsConfig(**{field_name: False})
        assert getattr(cfg, field_name) is False

    @pytest.mark.parametrize("val", ["YES", "True", "1"])
    def test_bool_env_truthy_values(self, monkeypatch, val):
        monkeypatch.setenv(f"{_ENV_PREFIX}ENABLE_PROVENANCE", val)
        cfg = FugitiveEmissionsConfig.from_env()
        assert cfg.enable_provenance is True

    @pytest.mark.parametrize("val", ["FALSE", "No", "0", "anything_else"])
    def test_bool_env_falsy_values(self, monkeypatch, val):
        monkeypatch.setenv(f"{_ENV_PREFIX}ENABLE_PROVENANCE", val)
        cfg = FugitiveEmissionsConfig.from_env()
        # Only "true"/"1"/"yes" are truthy
        assert cfg.enable_provenance is False


# ==========================================================================
# TestConfigNumericFields - 10 tests
# ==========================================================================


class TestConfigNumericFields:
    """Test numeric field boundary conditions and parsing."""

    @pytest.mark.parametrize("value,expected", [
        ("1", 1),
        ("100000", 100_000),
        ("500", 500),
    ])
    def test_max_batch_size_valid_range(self, monkeypatch, value, expected):
        monkeypatch.setenv(f"{_ENV_PREFIX}MAX_BATCH_SIZE", value)
        cfg = FugitiveEmissionsConfig.from_env()
        assert cfg.max_batch_size == expected

    def test_decimal_precision_min_boundary(self):
        cfg = FugitiveEmissionsConfig(decimal_precision=0)
        assert cfg.decimal_precision == 0

    def test_decimal_precision_max_boundary(self):
        cfg = FugitiveEmissionsConfig(decimal_precision=20)
        assert cfg.decimal_precision == 20

    def test_monte_carlo_iterations_min_boundary(self):
        cfg = FugitiveEmissionsConfig(monte_carlo_iterations=1)
        assert cfg.monte_carlo_iterations == 1

    def test_monte_carlo_iterations_max_boundary(self):
        cfg = FugitiveEmissionsConfig(monte_carlo_iterations=1_000_000)
        assert cfg.monte_carlo_iterations == 1_000_000

    def test_monte_carlo_seed_zero_allowed(self):
        cfg = FugitiveEmissionsConfig(monte_carlo_seed=0)
        assert cfg.monte_carlo_seed == 0

    def test_max_components_boundary(self):
        cfg = FugitiveEmissionsConfig(max_components=100_000)
        assert cfg.max_components == 100_000

    def test_max_surveys_boundary(self):
        cfg = FugitiveEmissionsConfig(max_surveys=50_000)
        assert cfg.max_surveys == 50_000

    def test_max_components_exceeds_limit_raises(self):
        with pytest.raises(ValueError, match="max_components"):
            FugitiveEmissionsConfig(max_components=100_001)

    def test_max_surveys_exceeds_limit_raises(self):
        with pytest.raises(ValueError, match="max_surveys"):
            FugitiveEmissionsConfig(max_surveys=50_001)

    def test_ldar_threshold_boundary_one(self):
        cfg = FugitiveEmissionsConfig(ldar_leak_threshold_ppm=1)
        assert cfg.ldar_leak_threshold_ppm == 1

    def test_ldar_threshold_boundary_max(self):
        cfg = FugitiveEmissionsConfig(ldar_leak_threshold_ppm=100_000)
        assert cfg.ldar_leak_threshold_ppm == 100_000

    def test_worker_threads_min_boundary(self):
        cfg = FugitiveEmissionsConfig(worker_threads=1)
        assert cfg.worker_threads == 1

    def test_worker_threads_max_boundary(self):
        cfg = FugitiveEmissionsConfig(worker_threads=64)
        assert cfg.worker_threads == 64

    def test_health_check_interval_min_boundary(self):
        cfg = FugitiveEmissionsConfig(health_check_interval=1)
        assert cfg.health_check_interval == 1

    def test_monte_carlo_seed_large_value(self):
        cfg = FugitiveEmissionsConfig(monte_carlo_seed=999999)
        assert cfg.monte_carlo_seed == 999999


# ==========================================================================
# TestConfigStringFields - 10 tests
# ==========================================================================


class TestConfigStringFields:
    """Test string field overrides and normalisation."""

    def test_api_prefix_custom(self):
        cfg = FugitiveEmissionsConfig(api_prefix="/custom/path")
        assert cfg.api_prefix == "/custom/path"

    def test_genesis_hash_custom(self):
        cfg = FugitiveEmissionsConfig(genesis_hash="MY-CUSTOM-GENESIS")
        assert cfg.genesis_hash == "MY-CUSTOM-GENESIS"

    def test_database_url_set(self):
        cfg = FugitiveEmissionsConfig(database_url="postgresql://localhost/test")
        assert cfg.database_url == "postgresql://localhost/test"

    def test_redis_url_set(self):
        cfg = FugitiveEmissionsConfig(redis_url="redis://localhost:6379/0")
        assert cfg.redis_url == "redis://localhost:6379/0"

    def test_confidence_levels_custom(self):
        cfg = FugitiveEmissionsConfig(confidence_levels="50,75,95")
        assert cfg.confidence_levels == "50,75,95"

    @pytest.mark.parametrize("gwp_source", ["ar4", "ar5", "ar6", "ar6_20yr"])
    def test_gwp_source_lowercase_normalised(self, gwp_source):
        cfg = FugitiveEmissionsConfig(default_gwp_source=gwp_source)
        assert cfg.default_gwp_source == gwp_source.upper()

    @pytest.mark.parametrize("method", [
        "average_emission_factor", "screening_ranges",
        "correlation_equation", "engineering_estimate",
        "direct_measurement",
    ])
    def test_calculation_method_lowercase_normalised(self, method):
        cfg = FugitiveEmissionsConfig(default_calculation_method=method)
        assert cfg.default_calculation_method == method.upper()

    @pytest.mark.parametrize("source", ["epa", "ipcc", "defra", "eu_ets", "api", "custom"])
    def test_ef_source_lowercase_normalised(self, source):
        cfg = FugitiveEmissionsConfig(default_emission_factor_source=source)
        assert cfg.default_emission_factor_source == source.upper()

    def test_log_level_mixed_case_normalised(self):
        cfg = FugitiveEmissionsConfig(log_level="Warning")
        assert cfg.log_level == "WARNING"

    def test_api_prefix_empty_string_valid(self):
        cfg = FugitiveEmissionsConfig(api_prefix="")
        assert cfg.api_prefix == ""


# ==========================================================================
# TestConfigSerialisation - 10 tests
# ==========================================================================


class TestConfigSerialisation:
    """Test to_dict() and __repr__() serialization."""

    def test_to_dict_returns_dict(self, default_config):
        d = default_config.to_dict()
        assert isinstance(d, dict)

    def test_to_dict_contains_all_fields(self, default_config):
        d = default_config.to_dict()
        assert "enabled" in d
        assert "default_gwp_source" in d
        assert "default_calculation_method" in d
        assert "decimal_precision" in d
        assert "genesis_hash" in d

    def test_to_dict_redacts_database_url(self):
        cfg = FugitiveEmissionsConfig(
            database_url="postgresql://secret@host/db"
        )
        d = cfg.to_dict()
        assert d["database_url"] == "***"

    def test_to_dict_redacts_redis_url(self):
        cfg = FugitiveEmissionsConfig(redis_url="redis://secret@host/0")
        d = cfg.to_dict()
        assert d["redis_url"] == "***"

    def test_to_dict_empty_urls_not_redacted(self, default_config):
        d = default_config.to_dict()
        assert d["database_url"] == ""
        assert d["redis_url"] == ""

    def test_repr_contains_class_name(self, default_config):
        r = repr(default_config)
        assert "FugitiveEmissionsConfig" in r

    def test_repr_does_not_leak_credentials(self):
        cfg = FugitiveEmissionsConfig(
            database_url="postgresql://user:password@host/db"
        )
        r = repr(cfg)
        assert "password" not in r
        assert "***" in r

    def test_to_dict_gwp_source_value(self, default_config):
        d = default_config.to_dict()
        assert d["default_gwp_source"] == "AR6"

    def test_to_dict_json_serializable(self, default_config):
        import json
        d = default_config.to_dict()
        serialized = json.dumps(d)
        assert isinstance(serialized, str)

    def test_to_dict_field_count(self, default_config):
        d = default_config.to_dict()
        # Should have all the config fields
        assert len(d) >= 25

    def test_to_dict_custom_config_values(self, custom_config):
        d = custom_config.to_dict()
        assert d["default_gwp_source"] == "AR5"
        assert d["default_calculation_method"] == "DIRECT_MEASUREMENT"
        assert d["max_batch_size"] == 100
        assert d["decimal_precision"] == 6
        assert d["enable_metrics"] is False

    def test_repr_contains_gwp_source(self, default_config):
        r = repr(default_config)
        assert "AR6" in r


# ==========================================================================
# TestConfigEdgeCases - 10 tests
# ==========================================================================


class TestConfigEdgeCases:
    """Test edge cases and unusual but valid configurations."""

    def test_confidence_levels_single_value(self):
        cfg = FugitiveEmissionsConfig(confidence_levels="95")
        assert cfg.confidence_levels == "95"

    def test_confidence_levels_many_values(self):
        cfg = FugitiveEmissionsConfig(confidence_levels="50,60,70,80,90,95,99")
        assert cfg.confidence_levels == "50,60,70,80,90,95,99"

    def test_gwp_source_lowercase_normalised(self):
        cfg = FugitiveEmissionsConfig(default_gwp_source="ar6")
        assert cfg.default_gwp_source == "AR6"

    def test_method_lowercase_normalised(self):
        cfg = FugitiveEmissionsConfig(
            default_calculation_method="average_emission_factor"
        )
        assert cfg.default_calculation_method == "AVERAGE_EMISSION_FACTOR"

    def test_ef_source_lowercase_normalised(self):
        cfg = FugitiveEmissionsConfig(
            default_emission_factor_source="epa"
        )
        assert cfg.default_emission_factor_source == "EPA"

    def test_log_level_lowercase_normalised(self):
        cfg = FugitiveEmissionsConfig(log_level="debug")
        assert cfg.log_level == "DEBUG"

    def test_multiple_validation_errors_reported(self):
        with pytest.raises(ValueError) as exc_info:
            FugitiveEmissionsConfig(
                max_batch_size=0,
                decimal_precision=-1,
                worker_threads=0,
            )
        error_text = str(exc_info.value)
        assert "max_batch_size" in error_text
        assert "decimal_precision" in error_text
        assert "worker_threads" in error_text

    def test_api_default_equals_max_is_valid(self):
        cfg = FugitiveEmissionsConfig(
            api_max_page_size=50, api_default_page_size=50
        )
        assert cfg.api_default_page_size == 50

    def test_enabled_false_is_valid(self):
        cfg = FugitiveEmissionsConfig(enabled=False)
        assert cfg.enabled is False

    def test_all_features_disabled_is_valid(self):
        cfg = FugitiveEmissionsConfig(
            enable_ldar_tracking=False,
            enable_component_tracking=False,
            enable_coal_mine_methane=False,
            enable_wastewater=False,
            enable_tank_losses=False,
            enable_pneumatic_devices=False,
            enable_compliance_checking=False,
            enable_uncertainty=False,
            enable_provenance=False,
            enable_metrics=False,
        )
        assert cfg.enable_ldar_tracking is False
        assert cfg.enable_metrics is False

    def test_confidence_levels_decimal_values(self):
        cfg = FugitiveEmissionsConfig(confidence_levels="95.5,99.9")
        assert cfg.confidence_levels == "95.5,99.9"

    def test_confidence_levels_trailing_comma_ignored(self):
        # Trailing comma produces empty string which is filtered by the
        # "if x.strip()" guard in __post_init__
        cfg = FugitiveEmissionsConfig(confidence_levels="90,95,")
        assert cfg.confidence_levels == "90,95,"


# ==========================================================================
# TestConfigLogLevel - 10 tests
# ==========================================================================


class TestConfigLogLevel:
    """Test log level validation and normalization."""

    @pytest.mark.parametrize("level", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    def test_valid_log_levels(self, level):
        cfg = FugitiveEmissionsConfig(log_level=level)
        assert cfg.log_level == level

    @pytest.mark.parametrize("level,expected", [
        ("debug", "DEBUG"),
        ("info", "INFO"),
        ("warning", "WARNING"),
        ("error", "ERROR"),
        ("critical", "CRITICAL"),
    ])
    def test_log_level_case_insensitive(self, level, expected):
        cfg = FugitiveEmissionsConfig(log_level=level)
        assert cfg.log_level == expected

    def test_invalid_log_level_trace_raises(self):
        with pytest.raises(ValueError, match="log_level"):
            FugitiveEmissionsConfig(log_level="TRACE")

    def test_invalid_log_level_warn_raises(self):
        with pytest.raises(ValueError, match="log_level"):
            FugitiveEmissionsConfig(log_level="WARN")

    def test_invalid_log_level_empty_raises(self):
        with pytest.raises(ValueError, match="log_level"):
            FugitiveEmissionsConfig(log_level="")

    def test_log_level_via_env(self, monkeypatch):
        monkeypatch.setenv(f"{_ENV_PREFIX}LOG_LEVEL", "error")
        cfg = FugitiveEmissionsConfig.from_env()
        assert cfg.log_level == "ERROR"

    def test_log_level_via_env_critical(self, monkeypatch):
        monkeypatch.setenv(f"{_ENV_PREFIX}LOG_LEVEL", "CRITICAL")
        cfg = FugitiveEmissionsConfig.from_env()
        assert cfg.log_level == "CRITICAL"


# ==========================================================================
# TestConfigMethodologyEnums - 10 tests
# ==========================================================================


class TestConfigMethodologyEnums:
    """Test all valid methodology enum values are accepted."""

    @pytest.mark.parametrize("gwp_source", list(_VALID_GWP_SOURCES))
    def test_valid_gwp_sources(self, gwp_source):
        cfg = FugitiveEmissionsConfig(default_gwp_source=gwp_source)
        assert cfg.default_gwp_source == gwp_source

    @pytest.mark.parametrize("method", list(_VALID_CALCULATION_METHODS))
    def test_valid_calculation_methods(self, method):
        cfg = FugitiveEmissionsConfig(default_calculation_method=method)
        assert cfg.default_calculation_method == method

    @pytest.mark.parametrize("source", list(_VALID_EF_SOURCES))
    def test_valid_ef_sources(self, source):
        cfg = FugitiveEmissionsConfig(default_emission_factor_source=source)
        assert cfg.default_emission_factor_source == source

    def test_gwp_ar6_20yr(self):
        cfg = FugitiveEmissionsConfig(default_gwp_source="AR6_20YR")
        assert cfg.default_gwp_source == "AR6_20YR"

    def test_all_methods_count(self):
        assert len(_VALID_CALCULATION_METHODS) == 5

    def test_all_ef_sources_count(self):
        assert len(_VALID_EF_SOURCES) == 6

    def test_all_gwp_sources_count(self):
        assert len(_VALID_GWP_SOURCES) == 4

    def test_all_log_levels_count(self):
        assert len(_VALID_LOG_LEVELS) == 5

    @pytest.mark.parametrize("invalid_gwp", ["AR1", "AR2", "AR3", "AR7", "IPCC", ""])
    def test_invalid_gwp_sources_rejected(self, invalid_gwp):
        with pytest.raises(ValueError):
            FugitiveEmissionsConfig(default_gwp_source=invalid_gwp)

    @pytest.mark.parametrize("invalid_method", [
        "MANUAL", "TIER_1", "DEFAULT", "AUTO", "",
    ])
    def test_invalid_calculation_methods_rejected(self, invalid_method):
        with pytest.raises(ValueError):
            FugitiveEmissionsConfig(default_calculation_method=invalid_method)

    @pytest.mark.parametrize("invalid_source", [
        "GHGP", "UNFCCC", "TIER1", "MANUAL", "",
    ])
    def test_invalid_ef_sources_rejected(self, invalid_source):
        with pytest.raises(ValueError):
            FugitiveEmissionsConfig(default_emission_factor_source=invalid_source)
