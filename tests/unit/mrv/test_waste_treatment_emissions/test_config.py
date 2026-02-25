# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-MRV-007 Waste Treatment Emissions Agent Configuration.

Tests WasteTreatmentConfig default values, environment variable overrides,
thread-safe singleton, reset_config, validation of ranges, and all
feature toggle flags.

Target: 40+ tests, 85%+ coverage.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import os
import threading
from unittest.mock import patch

import pytest

from greenlang.waste_treatment_emissions.config import (
    WasteTreatmentConfig,
    get_config,
    set_config,
    reset_config,
    _WasteTreatmentConfigHolder,
)


# ===========================================================================
# Default Value Tests
# ===========================================================================


class TestConfigDefaults:
    """Tests for WasteTreatmentConfig default values."""

    def test_enabled_default_true(self):
        """Default enabled is True."""
        cfg = WasteTreatmentConfig()
        assert cfg.enabled is True

    def test_default_gwp_source(self):
        """Default GWP source is AR6."""
        cfg = WasteTreatmentConfig()
        assert cfg.default_gwp_source == "AR6"

    def test_default_calculation_method(self):
        """Default calculation method is IPCC_TIER_2."""
        cfg = WasteTreatmentConfig()
        assert cfg.default_calculation_method == "IPCC_TIER_2"

    def test_default_emission_factor_source(self):
        """Default emission factor source is IPCC_2019."""
        cfg = WasteTreatmentConfig()
        assert cfg.default_emission_factor_source == "IPCC_2019"

    def test_default_decimal_precision(self):
        """Default decimal precision is 8."""
        cfg = WasteTreatmentConfig()
        assert cfg.decimal_precision == 8

    def test_default_max_batch_size(self):
        """Default max batch size is 10,000."""
        cfg = WasteTreatmentConfig()
        assert cfg.max_batch_size == 10_000

    def test_default_doc_f(self):
        """Default DOCf is 0.5."""
        cfg = WasteTreatmentConfig()
        assert cfg.default_doc_f == 0.5

    def test_default_mcf(self):
        """Default MCF is 1.0."""
        cfg = WasteTreatmentConfig()
        assert cfg.default_mcf == 1.0

    def test_default_f_ch4(self):
        """Default F(CH4) is 0.5."""
        cfg = WasteTreatmentConfig()
        assert cfg.default_f_ch4 == 0.5

    def test_default_oxidation_factor(self):
        """Default oxidation factor is 0.1."""
        cfg = WasteTreatmentConfig()
        assert cfg.default_oxidation_factor == 0.1

    def test_default_collection_efficiency(self):
        """Default gas collection efficiency is 0.75."""
        cfg = WasteTreatmentConfig()
        assert cfg.default_collection_efficiency == 0.75

    def test_default_flare_efficiency(self):
        """Default flare destruction efficiency is 0.98."""
        cfg = WasteTreatmentConfig()
        assert cfg.default_flare_efficiency == 0.98

    def test_default_utilization_efficiency(self):
        """Default utilization efficiency is 0.95."""
        cfg = WasteTreatmentConfig()
        assert cfg.default_utilization_efficiency == 0.95

    def test_default_climate_zone(self):
        """Default climate zone is temperate."""
        cfg = WasteTreatmentConfig()
        assert cfg.default_climate_zone == "temperate"

    def test_default_decay_rate(self):
        """Default decay rate is 0.05."""
        cfg = WasteTreatmentConfig()
        assert cfg.default_decay_rate == 0.05

    def test_default_composting_ch4_ef(self):
        """Default composting CH4 EF is 4.0 g/kg."""
        cfg = WasteTreatmentConfig()
        assert cfg.composting_ch4_ef == 4.0

    def test_default_composting_n2o_ef(self):
        """Default composting N2O EF is 0.24 g/kg."""
        cfg = WasteTreatmentConfig()
        assert cfg.composting_n2o_ef == 0.24

    def test_default_ad_ch4_ef(self):
        """Default AD CH4 EF is 0.8 g/kg."""
        cfg = WasteTreatmentConfig()
        assert cfg.ad_ch4_ef == 0.8

    def test_default_digestion_efficiency(self):
        """Default digestion efficiency is 0.70."""
        cfg = WasteTreatmentConfig()
        assert cfg.default_digestion_efficiency == 0.70

    def test_default_biogas_ch4_fraction(self):
        """Default biogas CH4 fraction is 0.60."""
        cfg = WasteTreatmentConfig()
        assert cfg.default_biogas_ch4_fraction == 0.60

    def test_default_incineration_of(self):
        """Default incineration oxidation factor is 1.0."""
        cfg = WasteTreatmentConfig()
        assert cfg.default_incineration_of == 1.0

    def test_default_energy_recovery_efficiency(self):
        """Default energy recovery efficiency is 0.25."""
        cfg = WasteTreatmentConfig()
        assert cfg.default_energy_recovery_efficiency == 0.25

    def test_default_open_burning_of(self):
        """Default open burning oxidation factor is 0.58."""
        cfg = WasteTreatmentConfig()
        assert cfg.default_open_burning_of == 0.58

    def test_default_bod_ch4_capacity(self):
        """Default BOD CH4 capacity is 0.6."""
        cfg = WasteTreatmentConfig()
        assert cfg.default_bod_ch4_capacity == 0.6

    def test_default_cod_ch4_capacity(self):
        """Default COD CH4 capacity is 0.25."""
        cfg = WasteTreatmentConfig()
        assert cfg.default_cod_ch4_capacity == 0.25

    def test_default_ww_mcf(self):
        """Default wastewater MCF is 0.3."""
        cfg = WasteTreatmentConfig()
        assert cfg.default_ww_mcf == 0.3

    def test_default_n2o_ef_plant(self):
        """Default plant N2O EF is 0.016."""
        cfg = WasteTreatmentConfig()
        assert cfg.default_n2o_ef_plant == 0.016

    def test_default_n2o_ef_effluent(self):
        """Default effluent N2O EF is 0.005."""
        cfg = WasteTreatmentConfig()
        assert cfg.default_n2o_ef_effluent == 0.005

    def test_default_monte_carlo_iterations(self):
        """Default Monte Carlo iterations is 5,000."""
        cfg = WasteTreatmentConfig()
        assert cfg.monte_carlo_iterations == 5_000

    def test_default_monte_carlo_seed(self):
        """Default Monte Carlo seed is 42."""
        cfg = WasteTreatmentConfig()
        assert cfg.monte_carlo_seed == 42

    def test_default_confidence_levels(self):
        """Default confidence levels is '90,95,99'."""
        cfg = WasteTreatmentConfig()
        assert cfg.confidence_levels == "90,95,99"

    def test_default_api_prefix(self):
        """Default API prefix is /api/v1/waste-treatment-emissions."""
        cfg = WasteTreatmentConfig()
        assert cfg.api_prefix == "/api/v1/waste-treatment-emissions"

    def test_default_worker_threads(self):
        """Default worker threads is 4."""
        cfg = WasteTreatmentConfig()
        assert cfg.worker_threads == 4

    def test_default_genesis_hash(self):
        """Default genesis hash contains WASTE-TREATMENT."""
        cfg = WasteTreatmentConfig()
        assert "WASTE-TREATMENT" in cfg.genesis_hash

    def test_default_log_level(self):
        """Default log level is INFO."""
        cfg = WasteTreatmentConfig()
        assert cfg.log_level == "INFO"


# ===========================================================================
# Environment Variable Override Tests
# ===========================================================================


class TestConfigEnvironment:
    """Tests for environment variable overrides."""

    def test_from_env_default_values(self, clean_env):
        """from_env with no env vars returns defaults."""
        cfg = WasteTreatmentConfig.from_env()
        assert cfg.default_gwp_source == "AR6"
        assert cfg.default_calculation_method == "IPCC_TIER_2"

    def test_override_gwp_source(self, clean_env):
        """GL_WASTE_TREATMENT_DEFAULT_GWP_SOURCE overrides default."""
        os.environ["GL_WASTE_TREATMENT_DEFAULT_GWP_SOURCE"] = "AR5"
        cfg = WasteTreatmentConfig.from_env()
        assert cfg.default_gwp_source == "AR5"

    def test_override_calculation_method(self, clean_env):
        """GL_WASTE_TREATMENT_DEFAULT_CALCULATION_METHOD overrides default."""
        os.environ["GL_WASTE_TREATMENT_DEFAULT_CALCULATION_METHOD"] = "FOD"
        cfg = WasteTreatmentConfig.from_env()
        assert cfg.default_calculation_method == "FOD"

    def test_override_max_batch_size(self, clean_env):
        """GL_WASTE_TREATMENT_MAX_BATCH_SIZE overrides default."""
        os.environ["GL_WASTE_TREATMENT_MAX_BATCH_SIZE"] = "5000"
        cfg = WasteTreatmentConfig.from_env()
        assert cfg.max_batch_size == 5000

    def test_override_boolean_enabled_false(self, clean_env):
        """GL_WASTE_TREATMENT_ENABLED=false disables agent."""
        os.environ["GL_WASTE_TREATMENT_ENABLED"] = "false"
        cfg = WasteTreatmentConfig.from_env()
        assert cfg.enabled is False

    def test_override_boolean_true_variants(self, clean_env):
        """Boolean override accepts true/1/yes."""
        for val in ["true", "1", "yes", "True", "YES"]:
            os.environ["GL_WASTE_TREATMENT_ENABLE_BIOLOGICAL"] = val
            cfg = WasteTreatmentConfig.from_env()
            assert cfg.enable_biological is True

    def test_override_float_composting_ch4(self, clean_env):
        """GL_WASTE_TREATMENT_COMPOSTING_CH4_EF overrides default."""
        os.environ["GL_WASTE_TREATMENT_COMPOSTING_CH4_EF"] = "8.0"
        cfg = WasteTreatmentConfig.from_env()
        assert cfg.composting_ch4_ef == 8.0

    def test_invalid_int_falls_back_to_default(self, clean_env):
        """Invalid integer value falls back to default."""
        os.environ["GL_WASTE_TREATMENT_MAX_BATCH_SIZE"] = "not_a_number"
        cfg = WasteTreatmentConfig.from_env()
        assert cfg.max_batch_size == 10_000

    def test_invalid_float_falls_back_to_default(self, clean_env):
        """Invalid float value falls back to default."""
        os.environ["GL_WASTE_TREATMENT_COMPOSTING_CH4_EF"] = "abc"
        cfg = WasteTreatmentConfig.from_env()
        assert cfg.composting_ch4_ef == 4.0

    def test_override_decimal_precision(self, clean_env):
        """GL_WASTE_TREATMENT_DECIMAL_PRECISION overrides default."""
        os.environ["GL_WASTE_TREATMENT_DECIMAL_PRECISION"] = "12"
        cfg = WasteTreatmentConfig.from_env()
        assert cfg.decimal_precision == 12


# ===========================================================================
# Singleton Tests
# ===========================================================================


class TestConfigSingleton:
    """Tests for thread-safe singleton configuration access."""

    def test_get_config_returns_instance(self):
        """get_config returns a WasteTreatmentConfig instance."""
        cfg = get_config()
        assert isinstance(cfg, WasteTreatmentConfig)

    def test_get_config_returns_same_instance(self):
        """get_config returns the same instance on subsequent calls."""
        cfg1 = get_config()
        cfg2 = get_config()
        assert cfg1 is cfg2

    def test_set_config_replaces_singleton(self):
        """set_config replaces the singleton instance."""
        custom_cfg = WasteTreatmentConfig(
            default_calculation_method="FOD",
        )
        set_config(custom_cfg)
        assert get_config().default_calculation_method == "FOD"

    def test_reset_config_clears_singleton(self):
        """reset_config clears the singleton so next get_config re-creates."""
        cfg1 = get_config()
        reset_config()
        cfg2 = get_config()
        # They should be different instances after reset
        assert cfg1 is not cfg2

    def test_thread_safe_singleton(self):
        """Singleton is thread-safe under concurrent access."""
        results = []

        def get_in_thread():
            cfg = get_config()
            results.append(id(cfg))

        threads = [threading.Thread(target=get_in_thread) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads should get the same instance
        assert len(set(results)) == 1


# ===========================================================================
# Validation Tests
# ===========================================================================


class TestConfigValidation:
    """Tests for configuration validation rules."""

    def test_invalid_gwp_source_rejected(self):
        """Invalid GWP source raises ValueError."""
        with pytest.raises(ValueError, match="default_gwp_source"):
            WasteTreatmentConfig(default_gwp_source="AR99")

    def test_invalid_calculation_method_rejected(self):
        """Invalid calculation method raises ValueError."""
        with pytest.raises(ValueError, match="default_calculation_method"):
            WasteTreatmentConfig(default_calculation_method="INVALID")

    def test_invalid_ef_source_rejected(self):
        """Invalid emission factor source raises ValueError."""
        with pytest.raises(ValueError, match="default_emission_factor_source"):
            WasteTreatmentConfig(default_emission_factor_source="BOGUS")

    def test_invalid_climate_zone_rejected(self):
        """Invalid climate zone raises ValueError."""
        with pytest.raises(ValueError, match="default_climate_zone"):
            WasteTreatmentConfig(default_climate_zone="mars")

    def test_invalid_log_level_rejected(self):
        """Invalid log level raises ValueError."""
        with pytest.raises(ValueError, match="log_level"):
            WasteTreatmentConfig(log_level="VERBOSE")

    def test_negative_decimal_precision_rejected(self):
        """Negative decimal precision raises ValueError."""
        with pytest.raises(ValueError, match="decimal_precision"):
            WasteTreatmentConfig(decimal_precision=-1)

    def test_excessive_decimal_precision_rejected(self):
        """Decimal precision > 20 raises ValueError."""
        with pytest.raises(ValueError, match="decimal_precision"):
            WasteTreatmentConfig(decimal_precision=21)

    def test_negative_doc_f_rejected(self):
        """Negative DOCf raises ValueError."""
        with pytest.raises(ValueError, match="default_doc_f"):
            WasteTreatmentConfig(default_doc_f=-0.1)

    def test_doc_f_over_one_rejected(self):
        """DOCf > 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="default_doc_f"):
            WasteTreatmentConfig(default_doc_f=1.5)

    def test_zero_decay_rate_rejected(self):
        """Zero decay rate raises ValueError."""
        with pytest.raises(ValueError, match="default_decay_rate"):
            WasteTreatmentConfig(default_decay_rate=0.0)

    def test_decay_rate_over_one_rejected(self):
        """Decay rate > 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="default_decay_rate"):
            WasteTreatmentConfig(default_decay_rate=1.5)

    def test_negative_composting_ch4_ef_rejected(self):
        """Negative composting CH4 EF raises ValueError."""
        with pytest.raises(ValueError, match="composting_ch4_ef"):
            WasteTreatmentConfig(composting_ch4_ef=-1.0)

    def test_zero_max_batch_size_rejected(self):
        """Zero max batch size raises ValueError."""
        with pytest.raises(ValueError, match="max_batch_size"):
            WasteTreatmentConfig(max_batch_size=0)

    def test_negative_monte_carlo_iterations_rejected(self):
        """Zero Monte Carlo iterations raises ValueError."""
        with pytest.raises(ValueError, match="monte_carlo_iterations"):
            WasteTreatmentConfig(monte_carlo_iterations=0)

    def test_zero_cache_ttl_rejected(self):
        """Zero cache TTL raises ValueError."""
        with pytest.raises(ValueError, match="cache_ttl_seconds"):
            WasteTreatmentConfig(cache_ttl_seconds=0)

    def test_api_default_exceeds_max_rejected(self):
        """api_default_page_size > api_max_page_size raises ValueError."""
        with pytest.raises(ValueError, match="api_default_page_size"):
            WasteTreatmentConfig(
                api_default_page_size=200,
                api_max_page_size=100,
            )

    def test_zero_worker_threads_rejected(self):
        """Zero worker threads raises ValueError."""
        with pytest.raises(ValueError, match="worker_threads"):
            WasteTreatmentConfig(worker_threads=0)

    def test_excessive_worker_threads_rejected(self):
        """Worker threads > 64 raises ValueError."""
        with pytest.raises(ValueError, match="worker_threads"):
            WasteTreatmentConfig(worker_threads=65)

    def test_empty_genesis_hash_rejected(self):
        """Empty genesis hash raises ValueError."""
        with pytest.raises(ValueError, match="genesis_hash"):
            WasteTreatmentConfig(genesis_hash="")

    def test_invalid_confidence_levels_rejected(self):
        """Non-numeric confidence levels raise ValueError."""
        with pytest.raises(ValueError, match="confidence_levels"):
            WasteTreatmentConfig(confidence_levels="abc,def")

    def test_confidence_level_out_of_range_rejected(self):
        """Confidence level >= 100 raises ValueError."""
        with pytest.raises(ValueError, match="confidence level"):
            WasteTreatmentConfig(confidence_levels="90,100,150")

    def test_negative_bod_ch4_capacity_rejected(self):
        """Zero BOD CH4 capacity raises ValueError."""
        with pytest.raises(ValueError, match="default_bod_ch4_capacity"):
            WasteTreatmentConfig(default_bod_ch4_capacity=0.0)


# ===========================================================================
# Feature Toggle Tests
# ===========================================================================


class TestConfigFeatureToggles:
    """Tests for feature toggle flags."""

    def test_enable_biological_default_true(self):
        """Default enable_biological is True."""
        cfg = WasteTreatmentConfig()
        assert cfg.enable_biological is True

    def test_enable_thermal_default_true(self):
        """Default enable_thermal is True."""
        cfg = WasteTreatmentConfig()
        assert cfg.enable_thermal is True

    def test_enable_wastewater_default_true(self):
        """Default enable_wastewater is True."""
        cfg = WasteTreatmentConfig()
        assert cfg.enable_wastewater is True

    def test_enable_methane_recovery_default_true(self):
        """Default enable_methane_recovery is True."""
        cfg = WasteTreatmentConfig()
        assert cfg.enable_methane_recovery is True

    def test_enable_energy_recovery_default_true(self):
        """Default enable_energy_recovery is True."""
        cfg = WasteTreatmentConfig()
        assert cfg.enable_energy_recovery is True

    def test_enable_compliance_checking_default_true(self):
        """Default enable_compliance_checking is True."""
        cfg = WasteTreatmentConfig()
        assert cfg.enable_compliance_checking is True

    def test_enable_uncertainty_default_true(self):
        """Default enable_uncertainty is True."""
        cfg = WasteTreatmentConfig()
        assert cfg.enable_uncertainty is True

    def test_enable_provenance_default_true(self):
        """Default enable_provenance is True."""
        cfg = WasteTreatmentConfig()
        assert cfg.enable_provenance is True

    def test_enable_metrics_default_true(self):
        """Default enable_metrics is True."""
        cfg = WasteTreatmentConfig()
        assert cfg.enable_metrics is True

    def test_separate_biogenic_co2_default_true(self):
        """Default separate_biogenic_co2 is True."""
        cfg = WasteTreatmentConfig()
        assert cfg.separate_biogenic_co2 is True

    def test_enable_auth_default_true(self):
        """Default enable_auth is True."""
        cfg = WasteTreatmentConfig()
        assert cfg.enable_auth is True

    def test_enable_tracing_default_true(self):
        """Default enable_tracing is True."""
        cfg = WasteTreatmentConfig()
        assert cfg.enable_tracing is True

    def test_enable_background_tasks_default_true(self):
        """Default enable_background_tasks is True."""
        cfg = WasteTreatmentConfig()
        assert cfg.enable_background_tasks is True


# ===========================================================================
# Serialization Tests
# ===========================================================================


class TestConfigSerialization:
    """Tests for configuration serialization helpers."""

    def test_to_dict_returns_dict(self):
        """to_dict returns a plain dictionary."""
        cfg = WasteTreatmentConfig()
        d = cfg.to_dict()
        assert isinstance(d, dict)

    def test_to_dict_redacts_database_url(self):
        """to_dict redacts database_url when set."""
        cfg = WasteTreatmentConfig(
            database_url="postgresql://user:pass@host/db",
        )
        d = cfg.to_dict()
        assert d["database_url"] == "***"

    def test_to_dict_empty_database_url(self):
        """to_dict shows empty string when database_url is empty."""
        cfg = WasteTreatmentConfig()
        d = cfg.to_dict()
        assert d["database_url"] == ""

    def test_repr_does_not_leak_credentials(self):
        """repr output redacts sensitive connection strings."""
        cfg = WasteTreatmentConfig(
            database_url="postgresql://secret@host/db",
            redis_url="redis://secret@host:6379",
        )
        r = repr(cfg)
        assert "secret" not in r
        assert "***" in r

    def test_to_dict_contains_all_keys(self):
        """to_dict contains all expected configuration keys."""
        cfg = WasteTreatmentConfig()
        d = cfg.to_dict()
        expected_keys = {
            "enabled", "database_url", "redis_url", "max_batch_size",
            "default_gwp_source", "default_calculation_method",
            "default_emission_factor_source", "decimal_precision",
            "enable_biological", "enable_thermal", "enable_wastewater",
            "enable_provenance", "enable_metrics",
        }
        for key in expected_keys:
            assert key in d, f"Missing key '{key}' in to_dict output"
