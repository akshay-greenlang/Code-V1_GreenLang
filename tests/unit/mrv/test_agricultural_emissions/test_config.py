# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-MRV-008 Agricultural Emissions Agent Configuration.

Tests AgriculturalEmissionsConfig default values, environment variable
overrides with GL_AGRICULTURAL_ prefix, thread-safe singleton access,
reset_config, validation of all enum/range constraints, feature toggles,
serialization helpers, and all 64 configuration fields.

Target: 60+ tests, 85%+ coverage of
    greenlang.agricultural_emissions.config

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import os
import threading
from unittest.mock import patch

import pytest

try:
    from greenlang.agricultural_emissions.config import (
        AgriculturalEmissionsConfig,
        get_config,
        set_config,
        reset_config,
        _AgriculturalEmissionsConfigHolder,
    )
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

_SKIP = pytest.mark.skipif(not CONFIG_AVAILABLE, reason="Config not available")


# ===========================================================================
# Default Value Tests -- Feature flag, connections, methodology
# ===========================================================================


@_SKIP
class TestAgriculturalConfigDefaults:
    """Tests for AgriculturalEmissionsConfig default values."""

    def test_enabled_default_true(self):
        """Default enabled is True."""
        cfg = AgriculturalEmissionsConfig()
        assert cfg.enabled is True

    def test_database_url_default_empty(self):
        """Default database_url is empty string."""
        cfg = AgriculturalEmissionsConfig()
        assert cfg.database_url == ""

    def test_redis_url_default_empty(self):
        """Default redis_url is empty string."""
        cfg = AgriculturalEmissionsConfig()
        assert cfg.redis_url == ""

    def test_default_gwp_source(self):
        """Default GWP source is AR6."""
        cfg = AgriculturalEmissionsConfig()
        assert cfg.default_gwp_source == "AR6"

    def test_default_calculation_method(self):
        """Default calculation method is IPCC_TIER_1."""
        cfg = AgriculturalEmissionsConfig()
        assert cfg.default_calculation_method == "IPCC_TIER_1"

    def test_default_emission_factor_source(self):
        """Default emission factor source is IPCC_2019."""
        cfg = AgriculturalEmissionsConfig()
        assert cfg.default_emission_factor_source == "IPCC_2019"

    def test_default_decimal_precision(self):
        """Default decimal precision is 8."""
        cfg = AgriculturalEmissionsConfig()
        assert cfg.decimal_precision == 8

    def test_default_climate_zone(self):
        """Default climate zone is COOL_TEMPERATE_WET."""
        cfg = AgriculturalEmissionsConfig()
        assert cfg.default_climate_zone == "COOL_TEMPERATE_WET"

    def test_default_max_batch_size(self):
        """Default max batch size is 10000."""
        cfg = AgriculturalEmissionsConfig()
        assert cfg.max_batch_size == 10_000

    def test_default_monte_carlo_iterations(self):
        """Default Monte Carlo iterations is 5000."""
        cfg = AgriculturalEmissionsConfig()
        assert cfg.monte_carlo_iterations == 5_000

    def test_default_monte_carlo_seed(self):
        """Default Monte Carlo seed is 42."""
        cfg = AgriculturalEmissionsConfig()
        assert cfg.monte_carlo_seed == 42

    def test_default_confidence_levels(self):
        """Default confidence levels is '90,95,99'."""
        cfg = AgriculturalEmissionsConfig()
        assert cfg.confidence_levels == "90,95,99"

    def test_default_log_level(self):
        """Default log level is INFO."""
        cfg = AgriculturalEmissionsConfig()
        assert cfg.log_level == "INFO"

    def test_default_worker_threads(self):
        """Default worker threads is 4."""
        cfg = AgriculturalEmissionsConfig()
        assert cfg.worker_threads == 4

    def test_default_api_prefix(self):
        """Default API prefix is /api/v1/agricultural-emissions."""
        cfg = AgriculturalEmissionsConfig()
        assert cfg.api_prefix == "/api/v1/agricultural-emissions"

    def test_default_api_max_page_size(self):
        """Default API max page size is 100."""
        cfg = AgriculturalEmissionsConfig()
        assert cfg.api_max_page_size == 100

    def test_default_api_default_page_size(self):
        """Default API default page size is 20."""
        cfg = AgriculturalEmissionsConfig()
        assert cfg.api_default_page_size == 20

    def test_default_genesis_hash(self):
        """Default genesis hash is the expected anchor string."""
        cfg = AgriculturalEmissionsConfig()
        assert cfg.genesis_hash == "GL-MRV-X-008-AGRICULTURAL-EMISSIONS-GENESIS"


# ===========================================================================
# Default Value Tests -- Enteric fermentation parameters
# ===========================================================================


@_SKIP
class TestEntericFermentationDefaults:
    """Tests for enteric fermentation parameter defaults."""

    def test_default_ym_pct(self):
        """Default Ym% is 6.5."""
        cfg = AgriculturalEmissionsConfig()
        assert cfg.default_ym_pct == 6.5

    def test_default_de_pct(self):
        """Default DE% is 65.0."""
        cfg = AgriculturalEmissionsConfig()
        assert cfg.default_de_pct == 65.0

    def test_default_cfi_dairy(self):
        """Default CFi for dairy is 0.386."""
        cfg = AgriculturalEmissionsConfig()
        assert cfg.default_cfi_dairy == 0.386

    def test_default_cfi_non_dairy(self):
        """Default CFi for non-dairy is 0.322."""
        cfg = AgriculturalEmissionsConfig()
        assert cfg.default_cfi_non_dairy == 0.322

    def test_default_activity_coefficient(self):
        """Default activity coefficient (Ca) is 0.0."""
        cfg = AgriculturalEmissionsConfig()
        assert cfg.default_activity_coefficient == 0.0

    def test_default_pregnancy_factor(self):
        """Default pregnancy factor is 0.10."""
        cfg = AgriculturalEmissionsConfig()
        assert cfg.default_pregnancy_factor == 0.10


# ===========================================================================
# Default Value Tests -- Manure management parameters
# ===========================================================================


@_SKIP
class TestManureManagementDefaults:
    """Tests for manure management parameter defaults."""

    def test_default_manure_mcf(self):
        """Default MCF is 0.10 (solid storage)."""
        cfg = AgriculturalEmissionsConfig()
        assert cfg.default_manure_mcf == 0.10

    def test_default_manure_bo_dairy(self):
        """Default Bo for dairy cattle is 0.24."""
        cfg = AgriculturalEmissionsConfig()
        assert cfg.default_manure_bo_dairy == 0.24

    def test_default_manure_bo_swine(self):
        """Default Bo for swine is 0.48."""
        cfg = AgriculturalEmissionsConfig()
        assert cfg.default_manure_bo_swine == 0.48

    def test_default_vs_dairy(self):
        """Default VS for dairy is 5.4 kg/head/day."""
        cfg = AgriculturalEmissionsConfig()
        assert cfg.default_vs_dairy == 5.4

    def test_default_temperature_c(self):
        """Default temperature is 15.0 C."""
        cfg = AgriculturalEmissionsConfig()
        assert cfg.default_temperature_c == 15.0


# ===========================================================================
# Default Value Tests -- Soils, indirect N2O, liming, rice, burning
# ===========================================================================


@_SKIP
class TestSoilsAndOtherDefaults:
    """Tests for soils, liming, rice, and field burning parameter defaults."""

    def test_default_ef1(self):
        """Default EF1 is 0.01."""
        cfg = AgriculturalEmissionsConfig()
        assert cfg.default_ef1 == 0.01

    def test_default_ef2_cg(self):
        """Default EF2 (cropland/grassland) is 8.0."""
        cfg = AgriculturalEmissionsConfig()
        assert cfg.default_ef2_cg == 8.0

    def test_default_ef2_f(self):
        """Default EF2 (forest land) is 2.5."""
        cfg = AgriculturalEmissionsConfig()
        assert cfg.default_ef2_f == 2.5

    def test_default_ef3_prp_cattle(self):
        """Default EF3_PRP for cattle is 0.02."""
        cfg = AgriculturalEmissionsConfig()
        assert cfg.default_ef3_prp_cattle == 0.02

    def test_default_ef3_prp_other(self):
        """Default EF3_PRP for other animals is 0.01."""
        cfg = AgriculturalEmissionsConfig()
        assert cfg.default_ef3_prp_other == 0.01

    def test_default_frac_gasf(self):
        """Default FRAC_GASF is 0.10."""
        cfg = AgriculturalEmissionsConfig()
        assert cfg.default_frac_gasf == 0.10

    def test_default_frac_gasm(self):
        """Default FRAC_GASM is 0.20."""
        cfg = AgriculturalEmissionsConfig()
        assert cfg.default_frac_gasm == 0.20

    def test_default_frac_leach(self):
        """Default FRAC_LEACH is 0.30."""
        cfg = AgriculturalEmissionsConfig()
        assert cfg.default_frac_leach == 0.30

    def test_default_ef4(self):
        """Default EF4 is 0.01."""
        cfg = AgriculturalEmissionsConfig()
        assert cfg.default_ef4 == 0.01

    def test_default_ef5(self):
        """Default EF5 is 0.0075."""
        cfg = AgriculturalEmissionsConfig()
        assert cfg.default_ef5 == 0.0075

    def test_default_limestone_ef(self):
        """Default limestone EF is 0.12."""
        cfg = AgriculturalEmissionsConfig()
        assert cfg.default_limestone_ef == 0.12

    def test_default_dolomite_ef(self):
        """Default dolomite EF is 0.13."""
        cfg = AgriculturalEmissionsConfig()
        assert cfg.default_dolomite_ef == 0.13

    def test_default_urea_ef(self):
        """Default urea EF is 0.20."""
        cfg = AgriculturalEmissionsConfig()
        assert cfg.default_urea_ef == 0.20

    def test_default_rice_baseline_ef(self):
        """Default rice baseline EF is 1.30 kg CH4/ha/day."""
        cfg = AgriculturalEmissionsConfig()
        assert cfg.default_rice_baseline_ef == 1.30

    def test_default_rice_cultivation_days(self):
        """Default rice cultivation days is 120."""
        cfg = AgriculturalEmissionsConfig()
        assert cfg.default_rice_cultivation_days == 120

    def test_default_water_regime(self):
        """Default water regime is continuously_flooded."""
        cfg = AgriculturalEmissionsConfig()
        assert cfg.default_water_regime == "continuously_flooded"

    def test_default_combustion_factor(self):
        """Default combustion factor is 0.80."""
        cfg = AgriculturalEmissionsConfig()
        assert cfg.default_combustion_factor == 0.80

    def test_default_burn_fraction(self):
        """Default burn fraction is 0.25."""
        cfg = AgriculturalEmissionsConfig()
        assert cfg.default_burn_fraction == 0.25


# ===========================================================================
# Default Value Tests -- Feature toggles
# ===========================================================================


@_SKIP
class TestFeatureToggleDefaults:
    """Tests that all feature toggles default to True."""

    def test_enable_enteric_default_true(self):
        """enable_enteric defaults to True."""
        cfg = AgriculturalEmissionsConfig()
        assert cfg.enable_enteric is True

    def test_enable_manure_default_true(self):
        """enable_manure defaults to True."""
        cfg = AgriculturalEmissionsConfig()
        assert cfg.enable_manure is True

    def test_enable_soils_default_true(self):
        """enable_soils defaults to True."""
        cfg = AgriculturalEmissionsConfig()
        assert cfg.enable_soils is True

    def test_enable_rice_default_true(self):
        """enable_rice defaults to True."""
        cfg = AgriculturalEmissionsConfig()
        assert cfg.enable_rice is True

    def test_enable_field_burning_default_true(self):
        """enable_field_burning defaults to True."""
        cfg = AgriculturalEmissionsConfig()
        assert cfg.enable_field_burning is True

    def test_enable_compliance_checking_default_true(self):
        """enable_compliance_checking defaults to True."""
        cfg = AgriculturalEmissionsConfig()
        assert cfg.enable_compliance_checking is True

    def test_enable_uncertainty_default_true(self):
        """enable_uncertainty defaults to True."""
        cfg = AgriculturalEmissionsConfig()
        assert cfg.enable_uncertainty is True

    def test_enable_provenance_default_true(self):
        """enable_provenance defaults to True."""
        cfg = AgriculturalEmissionsConfig()
        assert cfg.enable_provenance is True

    def test_enable_metrics_default_true(self):
        """enable_metrics defaults to True."""
        cfg = AgriculturalEmissionsConfig()
        assert cfg.enable_metrics is True

    def test_separate_biogenic_ch4_default_true(self):
        """separate_biogenic_ch4 defaults to True."""
        cfg = AgriculturalEmissionsConfig()
        assert cfg.separate_biogenic_ch4 is True

    def test_enable_auth_default_true(self):
        """enable_auth defaults to True."""
        cfg = AgriculturalEmissionsConfig()
        assert cfg.enable_auth is True

    def test_enable_tracing_default_true(self):
        """enable_tracing defaults to True."""
        cfg = AgriculturalEmissionsConfig()
        assert cfg.enable_tracing is True

    def test_enable_background_tasks_default_true(self):
        """enable_background_tasks defaults to True."""
        cfg = AgriculturalEmissionsConfig()
        assert cfg.enable_background_tasks is True


# ===========================================================================
# Default Value Tests -- Capacity and cache
# ===========================================================================


@_SKIP
class TestCapacityDefaults:
    """Tests for capacity limit and cache defaults."""

    def test_max_farms_default(self):
        """Default max_farms is 10000."""
        cfg = AgriculturalEmissionsConfig()
        assert cfg.max_farms == 10_000

    def test_max_livestock_records_default(self):
        """Default max_livestock_records is 100000."""
        cfg = AgriculturalEmissionsConfig()
        assert cfg.max_livestock_records == 100_000

    def test_cache_ttl_seconds_default(self):
        """Default cache_ttl_seconds is 3600."""
        cfg = AgriculturalEmissionsConfig()
        assert cfg.cache_ttl_seconds == 3600

    def test_health_check_interval_default(self):
        """Default health_check_interval is 30."""
        cfg = AgriculturalEmissionsConfig()
        assert cfg.health_check_interval == 30


# ===========================================================================
# Environment Variable Override Tests
# ===========================================================================


@_SKIP
class TestAgriculturalConfigEnvOverrides:
    """Tests for environment variable overrides with GL_AGRICULTURAL_ prefix."""

    def test_override_enabled_false(self, clean_env):
        """GL_AGRICULTURAL_ENABLED=false disables the agent."""
        os.environ["GL_AGRICULTURAL_ENABLED"] = "false"
        cfg = AgriculturalEmissionsConfig.from_env()
        assert cfg.enabled is False

    def test_override_gwp_source(self, clean_env):
        """GL_AGRICULTURAL_DEFAULT_GWP_SOURCE overrides GWP source."""
        os.environ["GL_AGRICULTURAL_DEFAULT_GWP_SOURCE"] = "AR5"
        cfg = AgriculturalEmissionsConfig.from_env()
        assert cfg.default_gwp_source == "AR5"

    def test_override_calculation_method(self, clean_env):
        """GL_AGRICULTURAL_DEFAULT_CALCULATION_METHOD overrides method."""
        os.environ["GL_AGRICULTURAL_DEFAULT_CALCULATION_METHOD"] = "IPCC_TIER_2"
        cfg = AgriculturalEmissionsConfig.from_env()
        assert cfg.default_calculation_method == "IPCC_TIER_2"

    def test_override_emission_factor_source(self, clean_env):
        """GL_AGRICULTURAL_DEFAULT_EMISSION_FACTOR_SOURCE overrides EF source."""
        os.environ["GL_AGRICULTURAL_DEFAULT_EMISSION_FACTOR_SOURCE"] = "DEFRA"
        cfg = AgriculturalEmissionsConfig.from_env()
        assert cfg.default_emission_factor_source == "DEFRA"

    def test_override_max_batch_size(self, clean_env):
        """GL_AGRICULTURAL_MAX_BATCH_SIZE overrides max_batch_size."""
        os.environ["GL_AGRICULTURAL_MAX_BATCH_SIZE"] = "5000"
        cfg = AgriculturalEmissionsConfig.from_env()
        assert cfg.max_batch_size == 5000

    def test_override_decimal_precision(self, clean_env):
        """GL_AGRICULTURAL_DECIMAL_PRECISION overrides decimal_precision."""
        os.environ["GL_AGRICULTURAL_DECIMAL_PRECISION"] = "12"
        cfg = AgriculturalEmissionsConfig.from_env()
        assert cfg.decimal_precision == 12

    def test_override_ym_pct(self, clean_env):
        """GL_AGRICULTURAL_DEFAULT_YM_PCT overrides Ym%."""
        os.environ["GL_AGRICULTURAL_DEFAULT_YM_PCT"] = "5.5"
        cfg = AgriculturalEmissionsConfig.from_env()
        assert cfg.default_ym_pct == 5.5

    def test_override_de_pct(self, clean_env):
        """GL_AGRICULTURAL_DEFAULT_DE_PCT overrides DE%."""
        os.environ["GL_AGRICULTURAL_DEFAULT_DE_PCT"] = "70.0"
        cfg = AgriculturalEmissionsConfig.from_env()
        assert cfg.default_de_pct == 70.0

    def test_override_climate_zone(self, clean_env):
        """GL_AGRICULTURAL_DEFAULT_CLIMATE_ZONE overrides climate zone."""
        os.environ["GL_AGRICULTURAL_DEFAULT_CLIMATE_ZONE"] = "TROPICAL_WET"
        cfg = AgriculturalEmissionsConfig.from_env()
        assert cfg.default_climate_zone == "TROPICAL_WET"

    def test_override_water_regime(self, clean_env):
        """GL_AGRICULTURAL_DEFAULT_WATER_REGIME overrides water regime."""
        os.environ["GL_AGRICULTURAL_DEFAULT_WATER_REGIME"] = "rainfed_regular"
        cfg = AgriculturalEmissionsConfig.from_env()
        assert cfg.default_water_regime == "rainfed_regular"

    def test_override_monte_carlo_iterations(self, clean_env):
        """GL_AGRICULTURAL_MONTE_CARLO_ITERATIONS overrides iterations."""
        os.environ["GL_AGRICULTURAL_MONTE_CARLO_ITERATIONS"] = "10000"
        cfg = AgriculturalEmissionsConfig.from_env()
        assert cfg.monte_carlo_iterations == 10_000

    def test_override_monte_carlo_seed(self, clean_env):
        """GL_AGRICULTURAL_MONTE_CARLO_SEED overrides seed."""
        os.environ["GL_AGRICULTURAL_MONTE_CARLO_SEED"] = "0"
        cfg = AgriculturalEmissionsConfig.from_env()
        assert cfg.monte_carlo_seed == 0

    def test_override_log_level(self, clean_env):
        """GL_AGRICULTURAL_LOG_LEVEL overrides log level."""
        os.environ["GL_AGRICULTURAL_LOG_LEVEL"] = "debug"
        cfg = AgriculturalEmissionsConfig.from_env()
        assert cfg.log_level == "DEBUG"

    def test_override_enable_enteric(self, clean_env):
        """GL_AGRICULTURAL_ENABLE_ENTERIC=false disables enteric engine."""
        os.environ["GL_AGRICULTURAL_ENABLE_ENTERIC"] = "false"
        cfg = AgriculturalEmissionsConfig.from_env()
        assert cfg.enable_enteric is False

    def test_override_enable_manure(self, clean_env):
        """GL_AGRICULTURAL_ENABLE_MANURE=0 disables manure engine."""
        os.environ["GL_AGRICULTURAL_ENABLE_MANURE"] = "0"
        cfg = AgriculturalEmissionsConfig.from_env()
        assert cfg.enable_manure is False

    def test_override_enable_rice(self, clean_env):
        """GL_AGRICULTURAL_ENABLE_RICE=no disables rice engine."""
        os.environ["GL_AGRICULTURAL_ENABLE_RICE"] = "no"
        cfg = AgriculturalEmissionsConfig.from_env()
        assert cfg.enable_rice is False

    def test_override_separate_biogenic_ch4(self, clean_env):
        """GL_AGRICULTURAL_SEPARATE_BIOGENIC_CH4=false disables separation."""
        os.environ["GL_AGRICULTURAL_SEPARATE_BIOGENIC_CH4"] = "false"
        cfg = AgriculturalEmissionsConfig.from_env()
        assert cfg.separate_biogenic_ch4 is False

    def test_invalid_integer_falls_back_to_default(self, clean_env):
        """Invalid integer value falls back to default with warning."""
        os.environ["GL_AGRICULTURAL_MAX_BATCH_SIZE"] = "not_a_number"
        cfg = AgriculturalEmissionsConfig.from_env()
        assert cfg.max_batch_size == 10_000

    def test_invalid_float_falls_back_to_default(self, clean_env):
        """Invalid float value falls back to default with warning."""
        os.environ["GL_AGRICULTURAL_DEFAULT_YM_PCT"] = "abc"
        cfg = AgriculturalEmissionsConfig.from_env()
        assert cfg.default_ym_pct == 6.5

    def test_boolean_accepts_yes(self, clean_env):
        """Boolean env vars accept 'yes' as True."""
        os.environ["GL_AGRICULTURAL_ENABLE_METRICS"] = "yes"
        cfg = AgriculturalEmissionsConfig.from_env()
        assert cfg.enable_metrics is True

    def test_boolean_accepts_1(self, clean_env):
        """Boolean env vars accept '1' as True."""
        os.environ["GL_AGRICULTURAL_ENABLE_PROVENANCE"] = "1"
        cfg = AgriculturalEmissionsConfig.from_env()
        assert cfg.enable_provenance is True

    def test_boolean_accepts_true_case_insensitive(self, clean_env):
        """Boolean env vars accept 'TRUE' (case insensitive) as True."""
        os.environ["GL_AGRICULTURAL_ENABLE_UNCERTAINTY"] = "TRUE"
        cfg = AgriculturalEmissionsConfig.from_env()
        assert cfg.enable_uncertainty is True

    def test_override_manure_mcf(self, clean_env):
        """GL_AGRICULTURAL_DEFAULT_MANURE_MCF overrides MCF."""
        os.environ["GL_AGRICULTURAL_DEFAULT_MANURE_MCF"] = "0.39"
        cfg = AgriculturalEmissionsConfig.from_env()
        assert cfg.default_manure_mcf == 0.39

    def test_override_rice_cultivation_days(self, clean_env):
        """GL_AGRICULTURAL_DEFAULT_RICE_CULTIVATION_DAYS overrides days."""
        os.environ["GL_AGRICULTURAL_DEFAULT_RICE_CULTIVATION_DAYS"] = "90"
        cfg = AgriculturalEmissionsConfig.from_env()
        assert cfg.default_rice_cultivation_days == 90

    def test_override_confidence_levels(self, clean_env):
        """GL_AGRICULTURAL_CONFIDENCE_LEVELS overrides levels string."""
        os.environ["GL_AGRICULTURAL_CONFIDENCE_LEVELS"] = "80,90,95"
        cfg = AgriculturalEmissionsConfig.from_env()
        assert cfg.confidence_levels == "80,90,95"


# ===========================================================================
# Validation Tests -- Enum values
# ===========================================================================


@_SKIP
class TestAgriculturalConfigEnumValidation:
    """Tests for enum/string field validation."""

    def test_invalid_gwp_source_raises(self):
        """Invalid GWP source raises ValueError."""
        with pytest.raises(ValueError, match="default_gwp_source"):
            AgriculturalEmissionsConfig(default_gwp_source="INVALID")

    def test_invalid_calculation_method_raises(self):
        """Invalid calculation method raises ValueError."""
        with pytest.raises(ValueError, match="default_calculation_method"):
            AgriculturalEmissionsConfig(default_calculation_method="BAD_METHOD")

    def test_invalid_emission_factor_source_raises(self):
        """Invalid emission factor source raises ValueError."""
        with pytest.raises(ValueError, match="default_emission_factor_source"):
            AgriculturalEmissionsConfig(default_emission_factor_source="UNKNOWN")

    def test_invalid_climate_zone_raises(self):
        """Invalid climate zone raises ValueError."""
        with pytest.raises(ValueError, match="default_climate_zone"):
            AgriculturalEmissionsConfig(default_climate_zone="MARS")

    def test_invalid_water_regime_raises(self):
        """Invalid water regime raises ValueError."""
        with pytest.raises(ValueError, match="default_water_regime"):
            AgriculturalEmissionsConfig(default_water_regime="flooded_always")

    def test_invalid_log_level_raises(self):
        """Invalid log level raises ValueError."""
        with pytest.raises(ValueError, match="log_level"):
            AgriculturalEmissionsConfig(log_level="VERBOSE")

    def test_gwp_source_case_normalised(self):
        """GWP source is normalised to uppercase."""
        cfg = AgriculturalEmissionsConfig(default_gwp_source="ar5")
        assert cfg.default_gwp_source == "AR5"

    def test_calculation_method_case_normalised(self):
        """Calculation method is normalised to uppercase."""
        cfg = AgriculturalEmissionsConfig(default_calculation_method="ipcc_tier_2")
        assert cfg.default_calculation_method == "IPCC_TIER_2"

    def test_water_regime_case_normalised(self):
        """Water regime is normalised to lowercase."""
        cfg = AgriculturalEmissionsConfig(default_water_regime="RAINFED_REGULAR")
        assert cfg.default_water_regime == "rainfed_regular"

    def test_log_level_case_normalised(self):
        """Log level is normalised to uppercase."""
        cfg = AgriculturalEmissionsConfig(log_level="debug")
        assert cfg.log_level == "DEBUG"


# ===========================================================================
# Validation Tests -- Numeric ranges
# ===========================================================================


@_SKIP
class TestAgriculturalConfigNumericValidation:
    """Tests for numeric range validation."""

    def test_negative_decimal_precision_raises(self):
        """Negative decimal precision raises ValueError."""
        with pytest.raises(ValueError, match="decimal_precision"):
            AgriculturalEmissionsConfig(decimal_precision=-1)

    def test_decimal_precision_above_20_raises(self):
        """Decimal precision > 20 raises ValueError."""
        with pytest.raises(ValueError, match="decimal_precision"):
            AgriculturalEmissionsConfig(decimal_precision=21)

    def test_ym_pct_negative_raises(self):
        """Negative Ym% raises ValueError."""
        with pytest.raises(ValueError, match="default_ym_pct"):
            AgriculturalEmissionsConfig(default_ym_pct=-1.0)

    def test_ym_pct_above_25_raises(self):
        """Ym% above 25.0 raises ValueError."""
        with pytest.raises(ValueError, match="default_ym_pct"):
            AgriculturalEmissionsConfig(default_ym_pct=26.0)

    def test_de_pct_negative_raises(self):
        """Negative DE% raises ValueError."""
        with pytest.raises(ValueError, match="default_de_pct"):
            AgriculturalEmissionsConfig(default_de_pct=-1.0)

    def test_de_pct_above_100_raises(self):
        """DE% above 100.0 raises ValueError."""
        with pytest.raises(ValueError, match="default_de_pct"):
            AgriculturalEmissionsConfig(default_de_pct=101.0)

    def test_manure_mcf_above_1_raises(self):
        """MCF above 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="default_manure_mcf"):
            AgriculturalEmissionsConfig(default_manure_mcf=1.1)

    def test_manure_bo_dairy_zero_raises(self):
        """Bo dairy of 0.0 raises ValueError (must be > 0)."""
        with pytest.raises(ValueError, match="default_manure_bo_dairy"):
            AgriculturalEmissionsConfig(default_manure_bo_dairy=0.0)

    def test_manure_bo_swine_above_1_raises(self):
        """Bo swine above 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="default_manure_bo_swine"):
            AgriculturalEmissionsConfig(default_manure_bo_swine=1.5)

    def test_vs_dairy_zero_raises(self):
        """VS dairy of 0.0 raises ValueError (must be > 0)."""
        with pytest.raises(ValueError, match="default_vs_dairy"):
            AgriculturalEmissionsConfig(default_vs_dairy=0.0)

    def test_vs_dairy_above_50_raises(self):
        """VS dairy above 50.0 raises ValueError."""
        with pytest.raises(ValueError, match="default_vs_dairy"):
            AgriculturalEmissionsConfig(default_vs_dairy=51.0)

    def test_temperature_below_minus_60_raises(self):
        """Temperature below -60.0 raises ValueError."""
        with pytest.raises(ValueError, match="default_temperature_c"):
            AgriculturalEmissionsConfig(default_temperature_c=-61.0)

    def test_temperature_above_60_raises(self):
        """Temperature above 60.0 raises ValueError."""
        with pytest.raises(ValueError, match="default_temperature_c"):
            AgriculturalEmissionsConfig(default_temperature_c=61.0)

    def test_rice_cultivation_days_zero_raises(self):
        """Zero rice cultivation days raises ValueError."""
        with pytest.raises(ValueError, match="default_rice_cultivation_days"):
            AgriculturalEmissionsConfig(default_rice_cultivation_days=0)

    def test_rice_cultivation_days_above_365_raises(self):
        """Rice cultivation days above 365 raises ValueError."""
        with pytest.raises(ValueError, match="default_rice_cultivation_days"):
            AgriculturalEmissionsConfig(default_rice_cultivation_days=366)

    def test_rice_baseline_ef_negative_raises(self):
        """Negative rice baseline EF raises ValueError."""
        with pytest.raises(ValueError, match="default_rice_baseline_ef"):
            AgriculturalEmissionsConfig(default_rice_baseline_ef=-0.1)

    def test_rice_baseline_ef_above_20_raises(self):
        """Rice baseline EF above 20.0 raises ValueError."""
        with pytest.raises(ValueError, match="default_rice_baseline_ef"):
            AgriculturalEmissionsConfig(default_rice_baseline_ef=20.1)

    def test_combustion_factor_negative_raises(self):
        """Negative combustion factor raises ValueError."""
        with pytest.raises(ValueError, match="default_combustion_factor"):
            AgriculturalEmissionsConfig(default_combustion_factor=-0.1)

    def test_burn_fraction_above_1_raises(self):
        """Burn fraction above 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="default_burn_fraction"):
            AgriculturalEmissionsConfig(default_burn_fraction=1.1)

    def test_max_batch_size_zero_raises(self):
        """Zero max_batch_size raises ValueError."""
        with pytest.raises(ValueError, match="max_batch_size"):
            AgriculturalEmissionsConfig(max_batch_size=0)

    def test_max_batch_size_above_100k_raises(self):
        """max_batch_size above 100000 raises ValueError."""
        with pytest.raises(ValueError, match="max_batch_size"):
            AgriculturalEmissionsConfig(max_batch_size=100_001)

    def test_max_farms_zero_raises(self):
        """Zero max_farms raises ValueError."""
        with pytest.raises(ValueError, match="max_farms"):
            AgriculturalEmissionsConfig(max_farms=0)

    def test_max_livestock_records_zero_raises(self):
        """Zero max_livestock_records raises ValueError."""
        with pytest.raises(ValueError, match="max_livestock_records"):
            AgriculturalEmissionsConfig(max_livestock_records=0)

    def test_monte_carlo_iterations_zero_raises(self):
        """Zero Monte Carlo iterations raises ValueError."""
        with pytest.raises(ValueError, match="monte_carlo_iterations"):
            AgriculturalEmissionsConfig(monte_carlo_iterations=0)

    def test_monte_carlo_iterations_above_1m_raises(self):
        """Monte Carlo iterations above 1000000 raises ValueError."""
        with pytest.raises(ValueError, match="monte_carlo_iterations"):
            AgriculturalEmissionsConfig(monte_carlo_iterations=1_000_001)

    def test_monte_carlo_seed_negative_raises(self):
        """Negative Monte Carlo seed raises ValueError."""
        with pytest.raises(ValueError, match="monte_carlo_seed"):
            AgriculturalEmissionsConfig(monte_carlo_seed=-1)

    def test_cache_ttl_zero_raises(self):
        """Zero cache TTL raises ValueError."""
        with pytest.raises(ValueError, match="cache_ttl_seconds"):
            AgriculturalEmissionsConfig(cache_ttl_seconds=0)

    def test_worker_threads_zero_raises(self):
        """Zero worker threads raises ValueError."""
        with pytest.raises(ValueError, match="worker_threads"):
            AgriculturalEmissionsConfig(worker_threads=0)

    def test_worker_threads_above_64_raises(self):
        """Worker threads > 64 raises ValueError."""
        with pytest.raises(ValueError, match="worker_threads"):
            AgriculturalEmissionsConfig(worker_threads=65)

    def test_health_check_interval_zero_raises(self):
        """Zero health_check_interval raises ValueError."""
        with pytest.raises(ValueError, match="health_check_interval"):
            AgriculturalEmissionsConfig(health_check_interval=0)

    def test_empty_genesis_hash_raises(self):
        """Empty genesis hash raises ValueError."""
        with pytest.raises(ValueError, match="genesis_hash"):
            AgriculturalEmissionsConfig(genesis_hash="")

    def test_api_default_page_exceeds_max_raises(self):
        """Default page size exceeding max raises ValueError."""
        with pytest.raises(ValueError, match="api_default_page_size"):
            AgriculturalEmissionsConfig(
                api_default_page_size=200, api_max_page_size=100
            )

    def test_api_max_page_size_zero_raises(self):
        """Zero api_max_page_size raises ValueError."""
        with pytest.raises(ValueError, match="api_max_page_size"):
            AgriculturalEmissionsConfig(api_max_page_size=0)


# ===========================================================================
# Validation Tests -- Confidence levels
# ===========================================================================


@_SKIP
class TestConfidenceLevelValidation:
    """Tests for confidence_levels string validation."""

    def test_non_numeric_confidence_levels_raises(self):
        """Non-numeric confidence levels raise ValueError."""
        with pytest.raises(ValueError, match="confidence_levels"):
            AgriculturalEmissionsConfig(confidence_levels="abc,def")

    def test_confidence_level_zero_raises(self):
        """Confidence level of 0 raises ValueError."""
        with pytest.raises(ValueError, match="confidence level"):
            AgriculturalEmissionsConfig(confidence_levels="0,50")

    def test_confidence_level_100_raises(self):
        """Confidence level of 100 raises ValueError."""
        with pytest.raises(ValueError, match="confidence level"):
            AgriculturalEmissionsConfig(confidence_levels="50,100")

    def test_valid_confidence_levels_accepted(self):
        """Valid confidence levels are accepted without error."""
        cfg = AgriculturalEmissionsConfig(confidence_levels="80,90,95,99")
        assert cfg.confidence_levels == "80,90,95,99"

    def test_single_confidence_level_accepted(self):
        """Single confidence level string is accepted."""
        cfg = AgriculturalEmissionsConfig(confidence_levels="95")
        assert cfg.confidence_levels == "95"


# ===========================================================================
# Validation Tests -- Boundary values that should pass
# ===========================================================================


@_SKIP
class TestBoundaryValuesAccepted:
    """Tests that boundary-valid values are accepted by validation."""

    def test_decimal_precision_zero_accepted(self):
        """Decimal precision of 0 is valid."""
        cfg = AgriculturalEmissionsConfig(decimal_precision=0)
        assert cfg.decimal_precision == 0

    def test_decimal_precision_20_accepted(self):
        """Decimal precision of 20 is valid."""
        cfg = AgriculturalEmissionsConfig(decimal_precision=20)
        assert cfg.decimal_precision == 20

    def test_ym_pct_zero_accepted(self):
        """Ym% of 0.0 is valid."""
        cfg = AgriculturalEmissionsConfig(default_ym_pct=0.0)
        assert cfg.default_ym_pct == 0.0

    def test_ym_pct_25_accepted(self):
        """Ym% of 25.0 is valid."""
        cfg = AgriculturalEmissionsConfig(default_ym_pct=25.0)
        assert cfg.default_ym_pct == 25.0

    def test_temperature_minus_60_accepted(self):
        """Temperature of -60.0 is valid."""
        cfg = AgriculturalEmissionsConfig(default_temperature_c=-60.0)
        assert cfg.default_temperature_c == -60.0

    def test_temperature_60_accepted(self):
        """Temperature of 60.0 is valid."""
        cfg = AgriculturalEmissionsConfig(default_temperature_c=60.0)
        assert cfg.default_temperature_c == 60.0

    def test_rice_cultivation_days_1_accepted(self):
        """Rice cultivation days of 1 is valid."""
        cfg = AgriculturalEmissionsConfig(default_rice_cultivation_days=1)
        assert cfg.default_rice_cultivation_days == 1

    def test_rice_cultivation_days_365_accepted(self):
        """Rice cultivation days of 365 is valid."""
        cfg = AgriculturalEmissionsConfig(default_rice_cultivation_days=365)
        assert cfg.default_rice_cultivation_days == 365


# ===========================================================================
# Singleton Tests
# ===========================================================================


@_SKIP
class TestConfigSingleton:
    """Tests for thread-safe singleton access."""

    def test_get_config_returns_instance(self, clean_env):
        """get_config returns an AgriculturalEmissionsConfig instance."""
        cfg = get_config()
        assert isinstance(cfg, AgriculturalEmissionsConfig)

    def test_get_config_returns_same_instance(self, clean_env):
        """get_config returns the same instance on repeated calls."""
        cfg1 = get_config()
        cfg2 = get_config()
        assert cfg1 is cfg2

    def test_set_config_replaces_singleton(self, clean_env):
        """set_config replaces the singleton."""
        custom = AgriculturalEmissionsConfig(
            default_calculation_method="IPCC_TIER_2"
        )
        set_config(custom)
        assert get_config().default_calculation_method == "IPCC_TIER_2"

    def test_reset_config_clears_singleton(self, clean_env):
        """reset_config clears the singleton so next get creates fresh."""
        cfg1 = get_config()
        reset_config()
        cfg2 = get_config()
        assert cfg1 is not cfg2

    def test_thread_safe_get_config(self, clean_env):
        """get_config is thread-safe under concurrent access."""
        results = []
        errors = []

        def worker():
            try:
                cfg = get_config()
                results.append(id(cfg))
            except Exception as exc:
                errors.append(str(exc))

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(set(results)) == 1

    def test_get_config_reads_env_vars(self, clean_env):
        """get_config reads GL_AGRICULTURAL_ env vars on first call."""
        os.environ["GL_AGRICULTURAL_DEFAULT_GWP_SOURCE"] = "AR4"
        cfg = get_config()
        assert cfg.default_gwp_source == "AR4"


# ===========================================================================
# Serialization Tests
# ===========================================================================


@_SKIP
class TestConfigSerialization:
    """Tests for AgriculturalEmissionsConfig serialization helpers."""

    def test_to_dict_redacts_database_url(self):
        """to_dict redacts database_url when it has a value."""
        cfg = AgriculturalEmissionsConfig(
            database_url="postgresql://user:pass@host/db"
        )
        d = cfg.to_dict()
        assert d["database_url"] == "***"

    def test_to_dict_redacts_redis_url(self):
        """to_dict redacts redis_url when it has a value."""
        cfg = AgriculturalEmissionsConfig(redis_url="redis://secret@host:6379")
        d = cfg.to_dict()
        assert d["redis_url"] == "***"

    def test_to_dict_empty_database_url(self):
        """to_dict shows empty string when database_url is empty."""
        cfg = AgriculturalEmissionsConfig()
        d = cfg.to_dict()
        assert d["database_url"] == ""

    def test_to_dict_empty_redis_url(self):
        """to_dict shows empty string when redis_url is empty."""
        cfg = AgriculturalEmissionsConfig()
        d = cfg.to_dict()
        assert d["redis_url"] == ""

    def test_repr_is_credential_safe(self):
        """repr does not leak connection strings."""
        cfg = AgriculturalEmissionsConfig(
            database_url="postgresql://secret@host/db"
        )
        text = repr(cfg)
        assert "secret" not in text
        assert "***" in text

    def test_to_dict_has_all_core_fields(self):
        """to_dict includes all core configuration fields."""
        cfg = AgriculturalEmissionsConfig()
        d = cfg.to_dict()
        assert "enabled" in d
        assert "default_gwp_source" in d
        assert "default_calculation_method" in d
        assert "default_emission_factor_source" in d
        assert "decimal_precision" in d
        assert "default_climate_zone" in d
        assert "enable_provenance" in d
        assert "genesis_hash" in d
        assert "enable_enteric" in d
        assert "enable_manure" in d
        assert "enable_soils" in d
        assert "enable_rice" in d
        assert "enable_field_burning" in d
        assert "separate_biogenic_ch4" in d
        assert "monte_carlo_iterations" in d
        assert "confidence_levels" in d

    def test_to_dict_has_enteric_fields(self):
        """to_dict includes enteric fermentation parameter fields."""
        cfg = AgriculturalEmissionsConfig()
        d = cfg.to_dict()
        assert "default_ym_pct" in d
        assert "default_de_pct" in d
        assert "default_cfi_dairy" in d
        assert "default_cfi_non_dairy" in d
        assert "default_activity_coefficient" in d
        assert "default_pregnancy_factor" in d

    def test_to_dict_has_manure_fields(self):
        """to_dict includes manure management parameter fields."""
        cfg = AgriculturalEmissionsConfig()
        d = cfg.to_dict()
        assert "default_manure_mcf" in d
        assert "default_manure_bo_dairy" in d
        assert "default_manure_bo_swine" in d
        assert "default_vs_dairy" in d
        assert "default_temperature_c" in d

    def test_to_dict_has_soils_fields(self):
        """to_dict includes agricultural soils parameter fields."""
        cfg = AgriculturalEmissionsConfig()
        d = cfg.to_dict()
        assert "default_ef1" in d
        assert "default_ef2_cg" in d
        assert "default_ef2_f" in d
        assert "default_ef3_prp_cattle" in d
        assert "default_ef3_prp_other" in d
        assert "default_frac_gasf" in d
        assert "default_frac_gasm" in d
        assert "default_frac_leach" in d

    def test_to_dict_has_indirect_n2o_fields(self):
        """to_dict includes indirect N2O parameter fields."""
        cfg = AgriculturalEmissionsConfig()
        d = cfg.to_dict()
        assert "default_ef4" in d
        assert "default_ef5" in d

    def test_to_dict_has_liming_and_rice_fields(self):
        """to_dict includes liming, urea, rice, and burning fields."""
        cfg = AgriculturalEmissionsConfig()
        d = cfg.to_dict()
        assert "default_limestone_ef" in d
        assert "default_dolomite_ef" in d
        assert "default_urea_ef" in d
        assert "default_rice_baseline_ef" in d
        assert "default_rice_cultivation_days" in d
        assert "default_water_regime" in d
        assert "default_combustion_factor" in d
        assert "default_burn_fraction" in d

    def test_repr_contains_class_name(self):
        """repr includes the class name."""
        cfg = AgriculturalEmissionsConfig()
        text = repr(cfg)
        assert "AgriculturalEmissionsConfig" in text


# ===========================================================================
# Multiple validation errors test
# ===========================================================================


@_SKIP
class TestMultipleValidationErrors:
    """Tests that multiple errors are reported in a single ValueError."""

    def test_multiple_errors_reported_at_once(self):
        """Multiple validation failures produce a single exception."""
        with pytest.raises(ValueError) as exc_info:
            AgriculturalEmissionsConfig(
                default_gwp_source="INVALID",
                decimal_precision=-5,
                worker_threads=0,
            )
        msg = str(exc_info.value)
        assert "default_gwp_source" in msg
        assert "decimal_precision" in msg
        assert "worker_threads" in msg

    def test_soil_fraction_fields_validated(self):
        """All soil fraction fields are validated in one pass."""
        with pytest.raises(ValueError) as exc_info:
            AgriculturalEmissionsConfig(
                default_frac_gasf=-0.1,
                default_frac_gasm=-0.2,
                default_frac_leach=-0.3,
            )
        msg = str(exc_info.value)
        assert "default_frac_gasf" in msg
        assert "default_frac_gasm" in msg
        assert "default_frac_leach" in msg
