# -*- coding: utf-8 -*-
"""Unit tests for ProcessEmissionsConfig - AGENT-MRV-004.

Tests cover default values, environment variable loading, singleton
behaviour (get_config / set_config / reset_config), validation logic,
boolean fields, numeric fields, string fields, serialisation, and
edge cases.  Target: 120+ tests with comprehensive parametrize usage.
"""

import os
import threading
import pytest

from greenlang.process_emissions.config import (
    ProcessEmissionsConfig,
    get_config,
    set_config,
    reset_config,
)

_ENV = "GL_PROCESS_EMISSIONS_"


# ============================================================================
# TestConfigDefaults - 15 tests
# ============================================================================


class TestConfigDefaults:
    """Verify every default value on a freshly constructed config."""

    def test_enabled_default(self, default_config):
        assert default_config.enabled is True

    def test_database_url_default(self, default_config):
        assert default_config.database_url == ""

    def test_redis_url_default(self, default_config):
        assert default_config.redis_url == ""

    def test_max_batch_size_default(self, default_config):
        assert default_config.max_batch_size == 500

    def test_default_gwp_source_default(self, default_config):
        assert default_config.default_gwp_source == "AR6"

    def test_default_calculation_tier_default(self, default_config):
        assert default_config.default_calculation_tier == "TIER_1"

    def test_default_calculation_method_default(self, default_config):
        assert default_config.default_calculation_method == "EMISSION_FACTOR"

    def test_default_emission_factor_source_default(self, default_config):
        assert default_config.default_emission_factor_source == "EPA"

    def test_decimal_precision_default(self, default_config):
        assert default_config.decimal_precision == 8

    def test_monte_carlo_iterations_default(self, default_config):
        assert default_config.monte_carlo_iterations == 5_000

    def test_monte_carlo_seed_default(self, default_config):
        assert default_config.monte_carlo_seed == 42

    def test_confidence_levels_default(self, default_config):
        assert default_config.confidence_levels == "90,95,99"

    def test_worker_threads_default(self, default_config):
        assert default_config.worker_threads == 4

    def test_genesis_hash_default(self, default_config):
        assert default_config.genesis_hash == "GL-MRV-X-004-PROCESS-EMISSIONS-GENESIS"

    def test_log_level_default(self, default_config):
        assert default_config.log_level == "INFO"


# ============================================================================
# TestConfigFromEnv - 20 tests
# ============================================================================


class TestConfigFromEnv:
    """Test GL_PROCESS_EMISSIONS_* env-var loading via from_env()."""

    @pytest.fixture(autouse=True)
    def _clean_pe_env(self):
        """Remove all GL_PROCESS_EMISSIONS_* env vars before each test."""
        keys = [k for k in os.environ if k.startswith(_ENV)]
        saved = {k: os.environ.pop(k) for k in keys}
        yield
        for k, v in saved.items():
            os.environ[k] = v

    def test_enabled_env_true(self):
        os.environ[f"{_ENV}ENABLED"] = "true"
        cfg = ProcessEmissionsConfig.from_env()
        assert cfg.enabled is True

    def test_enabled_env_false(self):
        os.environ[f"{_ENV}ENABLED"] = "false"
        cfg = ProcessEmissionsConfig.from_env()
        assert cfg.enabled is False

    def test_enabled_env_yes(self):
        os.environ[f"{_ENV}ENABLED"] = "yes"
        cfg = ProcessEmissionsConfig.from_env()
        assert cfg.enabled is True

    def test_enabled_env_one(self):
        os.environ[f"{_ENV}ENABLED"] = "1"
        cfg = ProcessEmissionsConfig.from_env()
        assert cfg.enabled is True

    def test_enabled_env_zero(self):
        os.environ[f"{_ENV}ENABLED"] = "0"
        cfg = ProcessEmissionsConfig.from_env()
        assert cfg.enabled is False

    def test_max_batch_size_env(self):
        os.environ[f"{_ENV}MAX_BATCH_SIZE"] = "1000"
        cfg = ProcessEmissionsConfig.from_env()
        assert cfg.max_batch_size == 1000

    def test_default_gwp_source_env(self):
        os.environ[f"{_ENV}DEFAULT_GWP_SOURCE"] = "AR5"
        cfg = ProcessEmissionsConfig.from_env()
        assert cfg.default_gwp_source == "AR5"

    def test_default_calculation_tier_env(self):
        os.environ[f"{_ENV}DEFAULT_CALCULATION_TIER"] = "TIER_3"
        cfg = ProcessEmissionsConfig.from_env()
        assert cfg.default_calculation_tier == "TIER_3"

    def test_default_calculation_method_env(self):
        os.environ[f"{_ENV}DEFAULT_CALCULATION_METHOD"] = "MASS_BALANCE"
        cfg = ProcessEmissionsConfig.from_env()
        assert cfg.default_calculation_method == "MASS_BALANCE"

    def test_default_emission_factor_source_env(self):
        os.environ[f"{_ENV}DEFAULT_EMISSION_FACTOR_SOURCE"] = "DEFRA"
        cfg = ProcessEmissionsConfig.from_env()
        assert cfg.default_emission_factor_source == "DEFRA"

    def test_decimal_precision_env(self):
        os.environ[f"{_ENV}DECIMAL_PRECISION"] = "16"
        cfg = ProcessEmissionsConfig.from_env()
        assert cfg.decimal_precision == 16

    def test_monte_carlo_iterations_env(self):
        os.environ[f"{_ENV}MONTE_CARLO_ITERATIONS"] = "20000"
        cfg = ProcessEmissionsConfig.from_env()
        assert cfg.monte_carlo_iterations == 20_000

    def test_monte_carlo_seed_env(self):
        os.environ[f"{_ENV}MONTE_CARLO_SEED"] = "0"
        cfg = ProcessEmissionsConfig.from_env()
        assert cfg.monte_carlo_seed == 0

    def test_confidence_levels_env(self):
        os.environ[f"{_ENV}CONFIDENCE_LEVELS"] = "80,90,95,99"
        cfg = ProcessEmissionsConfig.from_env()
        assert cfg.confidence_levels == "80,90,95,99"

    def test_log_level_env_lowercase(self):
        os.environ[f"{_ENV}LOG_LEVEL"] = "debug"
        cfg = ProcessEmissionsConfig.from_env()
        assert cfg.log_level == "DEBUG"

    def test_worker_threads_env(self):
        os.environ[f"{_ENV}WORKER_THREADS"] = "16"
        cfg = ProcessEmissionsConfig.from_env()
        assert cfg.worker_threads == 16

    def test_database_url_env(self):
        os.environ[f"{_ENV}DATABASE_URL"] = "postgresql://u:p@h/db"
        cfg = ProcessEmissionsConfig.from_env()
        assert cfg.database_url == "postgresql://u:p@h/db"

    def test_redis_url_env(self):
        os.environ[f"{_ENV}REDIS_URL"] = "redis://h:6379/0"
        cfg = ProcessEmissionsConfig.from_env()
        assert cfg.redis_url == "redis://h:6379/0"

    def test_invalid_int_env_falls_back(self):
        os.environ[f"{_ENV}MAX_BATCH_SIZE"] = "not_a_number"
        cfg = ProcessEmissionsConfig.from_env()
        # Falls back to the class-level default
        assert cfg.max_batch_size == 500

    def test_api_prefix_env(self):
        os.environ[f"{_ENV}API_PREFIX"] = "/custom/prefix"
        cfg = ProcessEmissionsConfig.from_env()
        assert cfg.api_prefix == "/custom/prefix"


# ============================================================================
# TestConfigSingleton - 10 tests
# ============================================================================


class TestConfigSingleton:
    """Test get_config(), set_config(), reset_config() singleton behaviour."""

    def test_get_config_returns_instance(self):
        cfg = get_config()
        assert isinstance(cfg, ProcessEmissionsConfig)

    def test_get_config_returns_same_instance(self):
        cfg1 = get_config()
        cfg2 = get_config()
        assert cfg1 is cfg2

    def test_set_config_replaces_singleton(self):
        custom = ProcessEmissionsConfig(default_calculation_tier="TIER_3")
        set_config(custom)
        assert get_config().default_calculation_tier == "TIER_3"

    def test_reset_config_clears_singleton(self):
        _ = get_config()
        reset_config()
        # After reset, get_config() creates a fresh one from env
        cfg = get_config()
        assert isinstance(cfg, ProcessEmissionsConfig)

    def test_reset_then_get_creates_new_instance(self):
        cfg1 = get_config()
        reset_config()
        cfg2 = get_config()
        # They should be different object identities after reset
        assert cfg1 is not cfg2

    def test_set_config_then_reset(self):
        custom = ProcessEmissionsConfig(worker_threads=16)
        set_config(custom)
        assert get_config().worker_threads == 16
        reset_config()
        assert get_config().worker_threads == 4  # back to default

    def test_thread_safety_get_config(self):
        """Multiple threads calling get_config() get the same object."""
        results = []

        def worker():
            results.append(id(get_config()))

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(set(results)) == 1

    def test_thread_safety_set_then_get(self):
        """set_config followed by get_config from another thread is consistent."""
        custom = ProcessEmissionsConfig(decimal_precision=12)
        set_config(custom)
        result = []

        def worker():
            result.append(get_config().decimal_precision)

        t = threading.Thread(target=worker)
        t.start()
        t.join()
        assert result[0] == 12

    def test_thread_safety_concurrent_reset(self):
        """Concurrent reset_config() calls do not raise."""
        _ = get_config()
        threads = [
            threading.Thread(target=reset_config) for _ in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # Should not raise and subsequent get_config should work
        cfg = get_config()
        assert isinstance(cfg, ProcessEmissionsConfig)

    def test_set_config_preserves_custom_values(self):
        cfg = ProcessEmissionsConfig(
            max_batch_size=42,
            cache_ttl_seconds=999,
        )
        set_config(cfg)
        retrieved = get_config()
        assert retrieved.max_batch_size == 42
        assert retrieved.cache_ttl_seconds == 999


# ============================================================================
# TestConfigValidation - 15 tests
# ============================================================================


class TestConfigValidation:
    """Test __post_init__ validation logic for invalid inputs."""

    def test_invalid_log_level_raises(self):
        with pytest.raises(ValueError, match="log_level"):
            ProcessEmissionsConfig(log_level="TRACE")

    def test_invalid_gwp_source_raises(self):
        with pytest.raises(ValueError, match="default_gwp_source"):
            ProcessEmissionsConfig(default_gwp_source="AR99")

    def test_invalid_calculation_tier_raises(self):
        with pytest.raises(ValueError, match="default_calculation_tier"):
            ProcessEmissionsConfig(default_calculation_tier="TIER_9")

    def test_invalid_calculation_method_raises(self):
        with pytest.raises(ValueError, match="default_calculation_method"):
            ProcessEmissionsConfig(default_calculation_method="GUESS")

    def test_invalid_ef_source_raises(self):
        with pytest.raises(ValueError, match="default_emission_factor_source"):
            ProcessEmissionsConfig(default_emission_factor_source="UNKNOWN")

    def test_negative_decimal_precision_raises(self):
        with pytest.raises(ValueError, match="decimal_precision"):
            ProcessEmissionsConfig(decimal_precision=-1)

    def test_excessive_decimal_precision_raises(self):
        with pytest.raises(ValueError, match="decimal_precision"):
            ProcessEmissionsConfig(decimal_precision=21)

    def test_zero_max_batch_size_raises(self):
        with pytest.raises(ValueError, match="max_batch_size"):
            ProcessEmissionsConfig(max_batch_size=0)

    def test_excessive_max_batch_size_raises(self):
        with pytest.raises(ValueError, match="max_batch_size"):
            ProcessEmissionsConfig(max_batch_size=200_000)

    def test_negative_monte_carlo_iterations_raises(self):
        with pytest.raises(ValueError, match="monte_carlo_iterations"):
            ProcessEmissionsConfig(monte_carlo_iterations=-10)

    def test_excessive_monte_carlo_iterations_raises(self):
        with pytest.raises(ValueError, match="monte_carlo_iterations"):
            ProcessEmissionsConfig(monte_carlo_iterations=2_000_000)

    def test_negative_monte_carlo_seed_raises(self):
        with pytest.raises(ValueError, match="monte_carlo_seed"):
            ProcessEmissionsConfig(monte_carlo_seed=-1)

    def test_zero_cache_ttl_raises(self):
        with pytest.raises(ValueError, match="cache_ttl_seconds"):
            ProcessEmissionsConfig(cache_ttl_seconds=0)

    def test_empty_genesis_hash_raises(self):
        with pytest.raises(ValueError, match="genesis_hash"):
            ProcessEmissionsConfig(genesis_hash="")

    def test_multiple_errors_reported(self):
        """When multiple fields are invalid, all errors appear in the message."""
        with pytest.raises(ValueError) as exc_info:
            ProcessEmissionsConfig(
                log_level="TRACE",
                default_gwp_source="AR99",
                max_batch_size=0,
            )
        msg = str(exc_info.value)
        assert "log_level" in msg
        assert "default_gwp_source" in msg
        assert "max_batch_size" in msg


# ============================================================================
# TestConfigBoolFields - 10 tests
# ============================================================================


class TestConfigBoolFields:
    """Test all boolean configuration fields."""

    @pytest.mark.parametrize("field_name,default_value", [
        ("enabled", True),
        ("enable_mass_balance", True),
        ("enable_abatement_tracking", True),
        ("enable_by_product_credits", True),
        ("enable_compliance_checking", True),
        ("enable_uncertainty", True),
        ("enable_provenance", True),
        ("enable_metrics", True),
        ("enable_background_tasks", True),
    ])
    def test_bool_field_default(self, field_name, default_value):
        cfg = ProcessEmissionsConfig()
        assert getattr(cfg, field_name) is default_value

    def test_all_bools_can_be_false(self):
        cfg = ProcessEmissionsConfig(
            enabled=False,
            enable_mass_balance=False,
            enable_abatement_tracking=False,
            enable_by_product_credits=False,
            enable_compliance_checking=False,
            enable_uncertainty=False,
            enable_provenance=False,
            enable_metrics=False,
            enable_background_tasks=False,
        )
        assert cfg.enabled is False
        assert cfg.enable_mass_balance is False
        assert cfg.enable_abatement_tracking is False
        assert cfg.enable_by_product_credits is False
        assert cfg.enable_compliance_checking is False
        assert cfg.enable_uncertainty is False
        assert cfg.enable_provenance is False
        assert cfg.enable_metrics is False
        assert cfg.enable_background_tasks is False


# ============================================================================
# TestConfigNumericFields - 10 tests
# ============================================================================


class TestConfigNumericFields:
    """Test numeric configuration fields, boundaries, and edge cases."""

    @pytest.mark.parametrize("field_name,default_value", [
        ("max_batch_size", 500),
        ("decimal_precision", 8),
        ("monte_carlo_iterations", 5_000),
        ("monte_carlo_seed", 42),
        ("max_material_inputs", 50),
        ("max_process_units", 200),
        ("max_abatement_records", 100),
        ("cache_ttl_seconds", 3600),
        ("api_max_page_size", 100),
        ("api_default_page_size", 20),
    ])
    def test_numeric_field_default(self, field_name, default_value):
        cfg = ProcessEmissionsConfig()
        assert getattr(cfg, field_name) == default_value

    @pytest.mark.parametrize("field_name,min_valid", [
        ("max_batch_size", 1),
        ("decimal_precision", 0),
        ("monte_carlo_iterations", 1),
        ("monte_carlo_seed", 0),
        ("max_material_inputs", 1),
        ("max_process_units", 1),
        ("max_abatement_records", 1),
        ("cache_ttl_seconds", 1),
        ("worker_threads", 1),
        ("health_check_interval", 1),
    ])
    def test_numeric_field_minimum_valid(self, field_name, min_valid):
        cfg = ProcessEmissionsConfig(**{field_name: min_valid})
        assert getattr(cfg, field_name) == min_valid

    @pytest.mark.parametrize("field_name,max_valid", [
        ("max_batch_size", 100_000),
        ("decimal_precision", 20),
        ("monte_carlo_iterations", 1_000_000),
        ("max_material_inputs", 1_000),
        ("max_process_units", 50_000),
        ("max_abatement_records", 10_000),
        ("worker_threads", 64),
    ])
    def test_numeric_field_maximum_valid(self, field_name, max_valid):
        cfg = ProcessEmissionsConfig(**{field_name: max_valid})
        assert getattr(cfg, field_name) == max_valid

    def test_api_default_page_size_must_be_lte_max(self):
        with pytest.raises(ValueError, match="api_default_page_size"):
            ProcessEmissionsConfig(
                api_max_page_size=10,
                api_default_page_size=20,
            )

    def test_api_default_page_size_equal_to_max_valid(self):
        cfg = ProcessEmissionsConfig(
            api_max_page_size=50,
            api_default_page_size=50,
        )
        assert cfg.api_default_page_size == 50
        assert cfg.api_max_page_size == 50

    def test_zero_worker_threads_raises(self):
        with pytest.raises(ValueError, match="worker_threads"):
            ProcessEmissionsConfig(worker_threads=0)

    def test_excessive_worker_threads_raises(self):
        with pytest.raises(ValueError, match="worker_threads"):
            ProcessEmissionsConfig(worker_threads=65)

    def test_zero_health_check_interval_raises(self):
        with pytest.raises(ValueError, match="health_check_interval"):
            ProcessEmissionsConfig(health_check_interval=0)

    def test_negative_max_material_inputs_raises(self):
        with pytest.raises(ValueError, match="max_material_inputs"):
            ProcessEmissionsConfig(max_material_inputs=-5)

    def test_negative_api_max_page_size_raises(self):
        with pytest.raises(ValueError, match="api_max_page_size"):
            ProcessEmissionsConfig(api_max_page_size=-1)


# ============================================================================
# TestConfigStringFields - 10 tests
# ============================================================================


class TestConfigStringFields:
    """Test string configuration fields including normalisation."""

    @pytest.mark.parametrize("gwp", ["AR4", "AR5", "AR6", "AR6_20YR"])
    def test_valid_gwp_sources(self, gwp):
        cfg = ProcessEmissionsConfig(default_gwp_source=gwp)
        assert cfg.default_gwp_source == gwp

    @pytest.mark.parametrize("tier", ["TIER_1", "TIER_2", "TIER_3"])
    def test_valid_calculation_tiers(self, tier):
        cfg = ProcessEmissionsConfig(default_calculation_tier=tier)
        assert cfg.default_calculation_tier == tier

    @pytest.mark.parametrize("method", [
        "EMISSION_FACTOR", "MASS_BALANCE", "STOICHIOMETRIC", "DIRECT_MEASUREMENT",
    ])
    def test_valid_calculation_methods(self, method):
        cfg = ProcessEmissionsConfig(default_calculation_method=method)
        assert cfg.default_calculation_method == method

    @pytest.mark.parametrize("src", ["EPA", "IPCC", "DEFRA", "EU_ETS", "CUSTOM"])
    def test_valid_ef_sources(self, src):
        cfg = ProcessEmissionsConfig(default_emission_factor_source=src)
        assert cfg.default_emission_factor_source == src

    @pytest.mark.parametrize("level", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    def test_valid_log_levels(self, level):
        cfg = ProcessEmissionsConfig(log_level=level)
        assert cfg.log_level == level

    def test_gwp_source_normalised_to_upper(self):
        cfg = ProcessEmissionsConfig(default_gwp_source="ar6")
        assert cfg.default_gwp_source == "AR6"

    def test_calculation_tier_normalised_to_upper(self):
        cfg = ProcessEmissionsConfig(default_calculation_tier="tier_2")
        assert cfg.default_calculation_tier == "TIER_2"

    def test_calculation_method_normalised_to_upper(self):
        cfg = ProcessEmissionsConfig(default_calculation_method="mass_balance")
        assert cfg.default_calculation_method == "MASS_BALANCE"

    def test_ef_source_normalised_to_upper(self):
        cfg = ProcessEmissionsConfig(default_emission_factor_source="defra")
        assert cfg.default_emission_factor_source == "DEFRA"

    def test_log_level_normalised_to_upper(self):
        cfg = ProcessEmissionsConfig(log_level="warning")
        assert cfg.log_level == "WARNING"


# ============================================================================
# TestConfigConfidenceLevels - 8 tests
# ============================================================================


class TestConfigConfidenceLevels:
    """Test confidence_levels parsing and validation."""

    def test_default_confidence_levels(self):
        cfg = ProcessEmissionsConfig()
        assert cfg.confidence_levels == "90,95,99"

    def test_custom_confidence_levels(self):
        cfg = ProcessEmissionsConfig(confidence_levels="80,90,95")
        assert cfg.confidence_levels == "80,90,95"

    def test_single_confidence_level(self):
        cfg = ProcessEmissionsConfig(confidence_levels="95")
        assert cfg.confidence_levels == "95"

    def test_float_confidence_levels(self):
        cfg = ProcessEmissionsConfig(confidence_levels="90.5,95.0,99.9")
        assert cfg.confidence_levels == "90.5,95.0,99.9"

    def test_confidence_level_zero_raises(self):
        with pytest.raises(ValueError, match="confidence level"):
            ProcessEmissionsConfig(confidence_levels="0,50,90")

    def test_confidence_level_100_raises(self):
        with pytest.raises(ValueError, match="confidence level"):
            ProcessEmissionsConfig(confidence_levels="90,100")

    def test_confidence_level_negative_raises(self):
        with pytest.raises(ValueError, match="confidence level"):
            ProcessEmissionsConfig(confidence_levels="-5,90")

    def test_non_numeric_confidence_level_raises(self):
        with pytest.raises(ValueError, match="confidence_levels"):
            ProcessEmissionsConfig(confidence_levels="high,medium")


# ============================================================================
# TestConfigToDict - 8 tests
# ============================================================================


class TestConfigToDict:
    """Test to_dict() serialisation and redaction."""

    def test_to_dict_returns_dict(self, default_config):
        d = default_config.to_dict()
        assert isinstance(d, dict)

    def test_to_dict_contains_all_fields(self, default_config):
        d = default_config.to_dict()
        expected_keys = {
            "enabled", "database_url", "redis_url", "max_batch_size",
            "default_gwp_source", "default_calculation_tier",
            "default_calculation_method", "default_emission_factor_source",
            "decimal_precision", "monte_carlo_iterations", "monte_carlo_seed",
            "confidence_levels", "enable_mass_balance",
            "enable_abatement_tracking", "enable_by_product_credits",
            "enable_compliance_checking", "enable_uncertainty",
            "enable_provenance", "enable_metrics", "max_material_inputs",
            "max_process_units", "max_abatement_records",
            "cache_ttl_seconds", "api_prefix", "api_max_page_size",
            "api_default_page_size", "log_level", "worker_threads",
            "enable_background_tasks", "health_check_interval",
            "genesis_hash",
        }
        assert expected_keys.issubset(set(d.keys()))

    def test_database_url_redacted_when_set(self):
        cfg = ProcessEmissionsConfig(
            database_url="postgresql://secret:secret@h/db"
        )
        d = cfg.to_dict()
        assert d["database_url"] == "***"

    def test_database_url_empty_when_not_set(self, default_config):
        d = default_config.to_dict()
        assert d["database_url"] == ""

    def test_redis_url_redacted_when_set(self):
        cfg = ProcessEmissionsConfig(redis_url="redis://secret@h:6379/0")
        d = cfg.to_dict()
        assert d["redis_url"] == "***"

    def test_redis_url_empty_when_not_set(self, default_config):
        d = default_config.to_dict()
        assert d["redis_url"] == ""

    def test_to_dict_values_match_attributes(self, default_config):
        d = default_config.to_dict()
        assert d["enabled"] == default_config.enabled
        assert d["max_batch_size"] == default_config.max_batch_size
        assert d["default_gwp_source"] == default_config.default_gwp_source

    def test_to_dict_json_serialisable(self, default_config):
        import json
        d = default_config.to_dict()
        serialised = json.dumps(d)
        assert isinstance(serialised, str)
        assert len(serialised) > 10


# ============================================================================
# TestConfigRepr - 4 tests
# ============================================================================


class TestConfigRepr:
    """Test __repr__ method."""

    def test_repr_contains_class_name(self, default_config):
        r = repr(default_config)
        assert r.startswith("ProcessEmissionsConfig(")

    def test_repr_redacts_database_url(self):
        cfg = ProcessEmissionsConfig(
            database_url="postgresql://u:p@h/db"
        )
        r = repr(cfg)
        assert "postgresql" not in r
        assert "***" in r

    def test_repr_contains_key_values(self, default_config):
        r = repr(default_config)
        assert "enabled" in r
        assert "default_gwp_source" in r

    def test_repr_is_string(self, default_config):
        assert isinstance(repr(default_config), str)


# ============================================================================
# TestConfigEdgeCases - 12 tests
# ============================================================================


class TestConfigEdgeCases:
    """Edge cases and boundary conditions."""

    def test_decimal_precision_zero_valid(self):
        cfg = ProcessEmissionsConfig(decimal_precision=0)
        assert cfg.decimal_precision == 0

    def test_decimal_precision_twenty_valid(self):
        cfg = ProcessEmissionsConfig(decimal_precision=20)
        assert cfg.decimal_precision == 20

    def test_max_batch_size_one_valid(self):
        cfg = ProcessEmissionsConfig(max_batch_size=1)
        assert cfg.max_batch_size == 1

    def test_monte_carlo_seed_zero_valid(self):
        cfg = ProcessEmissionsConfig(monte_carlo_seed=0)
        assert cfg.monte_carlo_seed == 0

    def test_large_cache_ttl_valid(self):
        cfg = ProcessEmissionsConfig(cache_ttl_seconds=86400)
        assert cfg.cache_ttl_seconds == 86400

    def test_api_prefix_custom(self):
        cfg = ProcessEmissionsConfig(api_prefix="/v3/pe")
        assert cfg.api_prefix == "/v3/pe"

    def test_genesis_hash_custom(self):
        cfg = ProcessEmissionsConfig(genesis_hash="CUSTOM-GENESIS-42")
        assert cfg.genesis_hash == "CUSTOM-GENESIS-42"

    def test_dataclass_equality(self):
        cfg1 = ProcessEmissionsConfig()
        cfg2 = ProcessEmissionsConfig()
        assert cfg1 == cfg2

    def test_dataclass_inequality(self):
        cfg1 = ProcessEmissionsConfig(max_batch_size=100)
        cfg2 = ProcessEmissionsConfig(max_batch_size=200)
        assert cfg1 != cfg2

    def test_config_is_mutable_after_creation(self, default_config):
        """Dataclass fields are mutable (not frozen)."""
        default_config.max_batch_size = 999
        assert default_config.max_batch_size == 999

    def test_negative_api_default_page_size_raises(self):
        with pytest.raises(ValueError, match="api_default_page_size"):
            ProcessEmissionsConfig(api_default_page_size=-1)

    def test_max_material_inputs_upper_bound(self):
        with pytest.raises(ValueError, match="max_material_inputs"):
            ProcessEmissionsConfig(max_material_inputs=1001)
