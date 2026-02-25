# -*- coding: utf-8 -*-
"""
Unit tests for Steam/Heat Purchase Agent configuration - AGENT-MRV-011.

Tests the SteamHeatPurchaseConfig singleton class and module-level
convenience functions: get_config, reset_config, set_config, validate_config.

Coverage targets:
- Singleton pattern (same instance, reset creates new)
- All 9 configuration sections with default values
- GL_SHP_ environment variable override (monkeypatch)
- Validation passes with defaults, fails with invalid values
- to_dict() returns all sections
- get_config() module function
- Each accessor method (get_steam_config, get_district_heating_config, etc.)

Test count target: ~130 tests.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List

import pytest

try:
    from greenlang.steam_heat_purchase.config import (
        SteamHeatPurchaseConfig,
        get_config,
        reset_config,
        set_config,
        validate_config,
        _ENV_PREFIX,
        _VALID_LOG_LEVELS,
        _VALID_GWP_SOURCES,
        _VALID_DATA_QUALITY_TIERS,
        _VALID_CHP_ALLOC_METHODS,
        _VALID_FRAMEWORKS,
    )

    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not CONFIG_AVAILABLE,
    reason="greenlang.steam_heat_purchase.config not importable",
)


# ===================================================================
# Helpers
# ===================================================================


def _fresh_config() -> SteamHeatPurchaseConfig:
    """Create a fresh configuration instance after reset."""
    SteamHeatPurchaseConfig.reset()
    return SteamHeatPurchaseConfig()


# ===================================================================
# Section 1: Singleton pattern
# ===================================================================


class TestSingletonPattern:
    """Tests for the singleton pattern implementation."""

    def test_same_instance_returned(self):
        SteamHeatPurchaseConfig.reset()
        cfg1 = SteamHeatPurchaseConfig()
        cfg2 = SteamHeatPurchaseConfig()
        assert cfg1 is cfg2

    def test_reset_creates_new_instance(self):
        cfg1 = SteamHeatPurchaseConfig()
        SteamHeatPurchaseConfig.reset()
        cfg2 = SteamHeatPurchaseConfig()
        assert cfg1 is not cfg2

    def test_get_config_returns_singleton(self):
        SteamHeatPurchaseConfig.reset()
        cfg1 = get_config()
        cfg2 = get_config()
        assert cfg1 is cfg2

    def test_get_config_matches_direct_construction(self):
        SteamHeatPurchaseConfig.reset()
        cfg1 = SteamHeatPurchaseConfig()
        cfg2 = get_config()
        assert cfg1 is cfg2

    def test_reset_config_function(self):
        cfg1 = get_config()
        reset_config()
        cfg2 = get_config()
        assert cfg1 is not cfg2


# ===================================================================
# Section 2: Default values (9 sections)
# ===================================================================


class TestDefaultGeneralSettings:
    """Tests for default general settings (Section 1)."""

    def test_service_name_default(self):
        cfg = _fresh_config()
        assert cfg.service_name == "steam-heat-purchase-service"

    def test_service_version_default(self):
        cfg = _fresh_config()
        assert cfg.service_version == "1.0.0"

    def test_log_level_default(self):
        cfg = _fresh_config()
        assert cfg.log_level == "INFO"

    def test_debug_mode_default(self):
        cfg = _fresh_config()
        assert cfg.debug_mode is False

    def test_tenant_id_default(self):
        cfg = _fresh_config()
        assert cfg.tenant_id == "default"


class TestDefaultDatabaseSettings:
    """Tests for default database settings (Section 2)."""

    def test_db_host_default(self):
        cfg = _fresh_config()
        assert cfg.db_host == "localhost"

    def test_db_port_default(self):
        cfg = _fresh_config()
        assert cfg.db_port == 5432

    def test_db_name_default(self):
        cfg = _fresh_config()
        assert cfg.db_name == "greenlang"

    def test_db_user_default(self):
        cfg = _fresh_config()
        assert cfg.db_user == "greenlang"

    def test_db_password_default_empty(self):
        cfg = _fresh_config()
        assert cfg.db_password == ""

    def test_db_pool_min_default(self):
        cfg = _fresh_config()
        assert cfg.db_pool_min == 2

    def test_db_pool_max_default(self):
        cfg = _fresh_config()
        assert cfg.db_pool_max == 10

    def test_db_ssl_mode_default(self):
        cfg = _fresh_config()
        assert cfg.db_ssl_mode == "prefer"


class TestDefaultCalculationSettings:
    """Tests for default calculation settings (Section 3)."""

    def test_calc_decimal_places_default(self):
        cfg = _fresh_config()
        assert cfg.calc_decimal_places == 8

    def test_calc_max_batch_size_default(self):
        cfg = _fresh_config()
        assert cfg.calc_max_batch_size == 10000

    def test_calc_default_gwp_source(self):
        cfg = _fresh_config()
        assert cfg.calc_default_gwp_source == "AR6"

    def test_calc_default_data_quality_tier(self):
        cfg = _fresh_config()
        assert cfg.calc_default_data_quality_tier == "TIER_1"

    def test_calc_default_boiler_efficiency(self):
        cfg = _fresh_config()
        assert cfg.calc_default_boiler_efficiency == 0.85

    def test_calc_default_distribution_loss_pct(self):
        cfg = _fresh_config()
        assert cfg.calc_default_distribution_loss_pct == 0.12

    def test_calc_condensate_return_default_pct(self):
        cfg = _fresh_config()
        assert cfg.calc_condensate_return_default_pct == 0.0

    def test_calc_default_chp_alloc_method(self):
        cfg = _fresh_config()
        assert cfg.calc_default_chp_alloc_method == "EFFICIENCY"

    def test_calc_default_ambient_temp_c(self):
        cfg = _fresh_config()
        assert cfg.calc_default_ambient_temp_c == 25.0

    def test_calc_max_trace_steps(self):
        cfg = _fresh_config()
        assert cfg.calc_max_trace_steps == 200

    def test_calc_enable_biogenic_separation(self):
        cfg = _fresh_config()
        assert cfg.calc_enable_biogenic_separation is True

    def test_calc_enable_condensate_adjustment(self):
        cfg = _fresh_config()
        assert cfg.calc_enable_condensate_adjustment is True


class TestDefaultSteamSettings:
    """Tests for default steam settings (Section 4)."""

    def test_steam_default_fuel_type(self):
        cfg = _fresh_config()
        assert cfg.steam_default_fuel_type == "natural_gas"

    def test_steam_default_steam_pressure(self):
        cfg = _fresh_config()
        assert cfg.steam_default_steam_pressure == "MEDIUM"

    def test_steam_default_steam_quality(self):
        cfg = _fresh_config()
        assert cfg.steam_default_steam_quality == "SATURATED"

    def test_steam_enable_multi_fuel_blend(self):
        cfg = _fresh_config()
        assert cfg.steam_enable_multi_fuel_blend is True

    def test_steam_max_fuel_types_per_blend(self):
        cfg = _fresh_config()
        assert cfg.steam_max_fuel_types_per_blend == 5


class TestDefaultDistrictHeatingSettings:
    """Tests for default district heating settings (Section 5)."""

    def test_dh_default_region(self):
        cfg = _fresh_config()
        assert cfg.dh_default_region == "global_default"

    def test_dh_default_network_type(self):
        cfg = _fresh_config()
        assert cfg.dh_default_network_type == "MUNICIPAL"

    def test_dh_enable_distribution_loss(self):
        cfg = _fresh_config()
        assert cfg.dh_enable_distribution_loss is True

    def test_dh_enable_supplier_ef(self):
        cfg = _fresh_config()
        assert cfg.dh_enable_supplier_ef is True


class TestDefaultDistrictCoolingSettings:
    """Tests for default district cooling settings (Section 6)."""

    def test_dc_default_technology(self):
        cfg = _fresh_config()
        assert cfg.dc_default_technology == "centrifugal_chiller"

    def test_dc_default_cop(self):
        cfg = _fresh_config()
        assert cfg.dc_default_cop == 6.0

    def test_dc_default_grid_ef_kwh(self):
        cfg = _fresh_config()
        assert cfg.dc_default_grid_ef_kwh == 0.436

    def test_dc_enable_free_cooling_adjustment(self):
        cfg = _fresh_config()
        assert cfg.dc_enable_free_cooling_adjustment is True

    def test_dc_enable_thermal_storage_losses(self):
        cfg = _fresh_config()
        assert cfg.dc_enable_thermal_storage_losses is True


class TestDefaultCHPSettings:
    """Tests for default CHP settings (Section 7)."""

    def test_chp_default_electrical_efficiency(self):
        cfg = _fresh_config()
        assert cfg.chp_default_electrical_efficiency == 0.35

    def test_chp_default_thermal_efficiency(self):
        cfg = _fresh_config()
        assert cfg.chp_default_thermal_efficiency == 0.45

    def test_chp_default_alloc_method(self):
        cfg = _fresh_config()
        assert cfg.chp_default_alloc_method == "EFFICIENCY"

    def test_chp_reference_electrical_efficiency(self):
        cfg = _fresh_config()
        assert cfg.chp_reference_electrical_efficiency == 0.525

    def test_chp_reference_thermal_efficiency(self):
        cfg = _fresh_config()
        assert cfg.chp_reference_thermal_efficiency == 0.90

    def test_chp_enable_primary_energy_savings(self):
        cfg = _fresh_config()
        assert cfg.chp_enable_primary_energy_savings is True


class TestDefaultUncertaintySettings:
    """Tests for default uncertainty settings (Section 8)."""

    def test_unc_default_method(self):
        cfg = _fresh_config()
        assert cfg.unc_default_method == "monte_carlo"

    def test_unc_default_iterations(self):
        cfg = _fresh_config()
        assert cfg.unc_default_iterations == 10000

    def test_unc_default_confidence_level(self):
        cfg = _fresh_config()
        assert cfg.unc_default_confidence_level == 0.95

    def test_unc_seed(self):
        cfg = _fresh_config()
        assert cfg.unc_seed == 42

    def test_unc_activity_data_uncertainty_pct(self):
        cfg = _fresh_config()
        assert cfg.unc_activity_data_uncertainty_pct == 5.0

    def test_unc_emission_factor_uncertainty_pct(self):
        cfg = _fresh_config()
        assert cfg.unc_emission_factor_uncertainty_pct == 10.0

    def test_unc_efficiency_uncertainty_pct(self):
        cfg = _fresh_config()
        assert cfg.unc_efficiency_uncertainty_pct == 5.0

    def test_unc_cop_uncertainty_pct(self):
        cfg = _fresh_config()
        assert cfg.unc_cop_uncertainty_pct == 8.0

    def test_unc_chp_allocation_uncertainty_pct(self):
        cfg = _fresh_config()
        assert cfg.unc_chp_allocation_uncertainty_pct == 10.0


class TestDefaultComplianceSettings:
    """Tests for default compliance settings (Section 9)."""

    def test_comp_enabled_frameworks_default(self):
        cfg = _fresh_config()
        expected = [
            "ghg_protocol_scope2",
            "iso_14064",
            "csrd_esrs",
            "cdp",
            "sbti",
            "eu_eed",
            "epa_mrr",
        ]
        assert cfg.comp_enabled_frameworks == expected

    def test_comp_strict_mode_default(self):
        cfg = _fresh_config()
        assert cfg.comp_strict_mode is False

    def test_comp_require_all_frameworks_default(self):
        cfg = _fresh_config()
        assert cfg.comp_require_all_frameworks is False


class TestDefaultObservabilitySettings:
    """Tests for default observability and other settings."""

    def test_enable_metrics_default(self):
        cfg = _fresh_config()
        assert cfg.enable_metrics is True

    def test_metrics_prefix_default(self):
        cfg = _fresh_config()
        assert cfg.metrics_prefix == "gl_shp_"

    def test_enable_tracing_default(self):
        cfg = _fresh_config()
        assert cfg.enable_tracing is True

    def test_enable_provenance_default(self):
        cfg = _fresh_config()
        assert cfg.enable_provenance is True

    def test_genesis_hash_default(self):
        cfg = _fresh_config()
        assert cfg.genesis_hash == "GL-MRV-X-022-STEAM-HEAT-PURCHASE-GENESIS"

    def test_enable_auth_default(self):
        cfg = _fresh_config()
        assert cfg.enable_auth is True

    def test_worker_threads_default(self):
        cfg = _fresh_config()
        assert cfg.worker_threads == 4

    def test_enable_background_tasks_default(self):
        cfg = _fresh_config()
        assert cfg.enable_background_tasks is True

    def test_health_check_interval_default(self):
        cfg = _fresh_config()
        assert cfg.health_check_interval == 30

    def test_api_prefix_default(self):
        cfg = _fresh_config()
        assert cfg.api_prefix == "/api/v1/steam-heat-purchase"

    def test_api_rate_limit_default(self):
        cfg = _fresh_config()
        assert cfg.api_rate_limit == 100

    def test_cors_origins_default(self):
        cfg = _fresh_config()
        assert cfg.cors_origins == ["*"]

    def test_enable_docs_default(self):
        cfg = _fresh_config()
        assert cfg.enable_docs is True

    def test_enabled_default(self):
        cfg = _fresh_config()
        assert cfg.enabled is True


# ===================================================================
# Section 3: Environment variable override
# ===================================================================


class TestEnvVarOverride:
    """Tests for GL_SHP_ environment variable overrides."""

    def test_service_name_override(self, monkeypatch):
        monkeypatch.setenv("GL_SHP_SERVICE_NAME", "custom-service")
        cfg = _fresh_config()
        assert cfg.service_name == "custom-service"

    def test_log_level_override(self, monkeypatch):
        monkeypatch.setenv("GL_SHP_LOG_LEVEL", "DEBUG")
        cfg = _fresh_config()
        assert cfg.log_level == "DEBUG"

    def test_debug_mode_true(self, monkeypatch):
        monkeypatch.setenv("GL_SHP_DEBUG_MODE", "true")
        cfg = _fresh_config()
        assert cfg.debug_mode is True

    def test_debug_mode_yes(self, monkeypatch):
        monkeypatch.setenv("GL_SHP_DEBUG_MODE", "yes")
        cfg = _fresh_config()
        assert cfg.debug_mode is True

    def test_debug_mode_1(self, monkeypatch):
        monkeypatch.setenv("GL_SHP_DEBUG_MODE", "1")
        cfg = _fresh_config()
        assert cfg.debug_mode is True

    def test_debug_mode_false_string(self, monkeypatch):
        monkeypatch.setenv("GL_SHP_DEBUG_MODE", "false")
        cfg = _fresh_config()
        assert cfg.debug_mode is False

    def test_db_port_override(self, monkeypatch):
        monkeypatch.setenv("GL_SHP_DB_PORT", "5433")
        cfg = _fresh_config()
        assert cfg.db_port == 5433

    def test_db_port_invalid_uses_default(self, monkeypatch):
        monkeypatch.setenv("GL_SHP_DB_PORT", "not_a_number")
        cfg = _fresh_config()
        assert cfg.db_port == 5432

    def test_calc_gwp_source_override(self, monkeypatch):
        monkeypatch.setenv("GL_SHP_CALC_DEFAULT_GWP_SOURCE", "AR5")
        cfg = _fresh_config()
        assert cfg.calc_default_gwp_source == "AR5"

    def test_calc_boiler_efficiency_override(self, monkeypatch):
        monkeypatch.setenv("GL_SHP_CALC_DEFAULT_BOILER_EFFICIENCY", "0.92")
        cfg = _fresh_config()
        assert cfg.calc_default_boiler_efficiency == 0.92

    def test_calc_boiler_efficiency_invalid_uses_default(self, monkeypatch):
        monkeypatch.setenv("GL_SHP_CALC_DEFAULT_BOILER_EFFICIENCY", "abc")
        cfg = _fresh_config()
        assert cfg.calc_default_boiler_efficiency == 0.85

    def test_comp_frameworks_override(self, monkeypatch):
        monkeypatch.setenv("GL_SHP_COMP_ENABLED_FRAMEWORKS", "cdp,sbti,ghg_protocol_scope2")
        cfg = _fresh_config()
        assert cfg.comp_enabled_frameworks == ["cdp", "sbti", "ghg_protocol_scope2"]

    def test_comp_frameworks_empty_uses_default(self, monkeypatch):
        monkeypatch.setenv("GL_SHP_COMP_ENABLED_FRAMEWORKS", "")
        cfg = _fresh_config()
        # Empty parsed list should fall back to default
        assert len(cfg.comp_enabled_frameworks) == 7

    def test_cors_origins_override(self, monkeypatch):
        monkeypatch.setenv("GL_SHP_CORS_ORIGINS", "http://localhost:3000,http://example.com")
        cfg = _fresh_config()
        assert cfg.cors_origins == [
            "http://localhost:3000",
            "http://example.com",
        ]

    def test_enabled_false(self, monkeypatch):
        monkeypatch.setenv("GL_SHP_ENABLED", "false")
        cfg = _fresh_config()
        assert cfg.enabled is False

    def test_unc_seed_override(self, monkeypatch):
        monkeypatch.setenv("GL_SHP_UNC_SEED", "123")
        cfg = _fresh_config()
        assert cfg.unc_seed == 123


# ===================================================================
# Section 4: Validation
# ===================================================================


class TestValidation:
    """Tests for the validate() method."""

    def test_defaults_pass_validation(self):
        cfg = _fresh_config()
        errors = cfg.validate()
        assert errors == []

    def test_validate_config_function(self):
        reset_config()
        errors = validate_config()
        assert errors == []

    def test_invalid_log_level(self):
        cfg = _fresh_config()
        cfg.log_level = "TRACE"
        errors = cfg.validate()
        assert any("log_level" in e for e in errors)

    def test_invalid_gwp_source(self):
        cfg = _fresh_config()
        cfg.calc_default_gwp_source = "AR99"
        errors = cfg.validate()
        assert any("calc_default_gwp_source" in e for e in errors)

    def test_invalid_data_quality_tier(self):
        cfg = _fresh_config()
        cfg.calc_default_data_quality_tier = "TIER_99"
        errors = cfg.validate()
        assert any("calc_default_data_quality_tier" in e for e in errors)

    def test_boiler_efficiency_too_high(self):
        cfg = _fresh_config()
        cfg.calc_default_boiler_efficiency = 1.5
        errors = cfg.validate()
        assert any("calc_default_boiler_efficiency" in e for e in errors)

    def test_boiler_efficiency_zero(self):
        cfg = _fresh_config()
        cfg.calc_default_boiler_efficiency = 0.0
        errors = cfg.validate()
        assert any("calc_default_boiler_efficiency" in e for e in errors)

    def test_distribution_loss_negative(self):
        cfg = _fresh_config()
        cfg.calc_default_distribution_loss_pct = -0.1
        errors = cfg.validate()
        assert any("calc_default_distribution_loss_pct" in e for e in errors)

    def test_distribution_loss_too_high(self):
        cfg = _fresh_config()
        cfg.calc_default_distribution_loss_pct = 1.5
        errors = cfg.validate()
        assert any("calc_default_distribution_loss_pct" in e for e in errors)

    def test_invalid_chp_alloc_method(self):
        cfg = _fresh_config()
        cfg.calc_default_chp_alloc_method = "UNKNOWN_METHOD"
        errors = cfg.validate()
        assert any("calc_default_chp_alloc_method" in e for e in errors)

    def test_db_port_negative(self):
        cfg = _fresh_config()
        cfg.db_port = -1
        errors = cfg.validate()
        assert any("db_port" in e for e in errors)

    def test_db_port_too_high(self):
        cfg = _fresh_config()
        cfg.db_port = 99999
        errors = cfg.validate()
        assert any("db_port" in e for e in errors)

    def test_pool_min_exceeds_max(self):
        cfg = _fresh_config()
        cfg.db_pool_min = 20
        cfg.db_pool_max = 5
        errors = cfg.validate()
        assert any("db_pool_min" in e for e in errors)

    def test_invalid_ssl_mode(self):
        cfg = _fresh_config()
        cfg.db_ssl_mode = "invalid_ssl"
        errors = cfg.validate()
        assert any("db_ssl_mode" in e for e in errors)

    def test_invalid_steam_fuel_type(self):
        cfg = _fresh_config()
        cfg.steam_default_fuel_type = "plutonium"
        errors = cfg.validate()
        assert any("steam_default_fuel_type" in e for e in errors)

    def test_invalid_steam_pressure(self):
        cfg = _fresh_config()
        cfg.steam_default_steam_pressure = "MEGA"
        errors = cfg.validate()
        assert any("steam_default_steam_pressure" in e for e in errors)

    def test_invalid_steam_quality(self):
        cfg = _fresh_config()
        cfg.steam_default_steam_quality = "DRY"
        errors = cfg.validate()
        assert any("steam_default_steam_quality" in e for e in errors)

    def test_invalid_network_type(self):
        cfg = _fresh_config()
        cfg.dh_default_network_type = "URBAN"
        errors = cfg.validate()
        assert any("dh_default_network_type" in e for e in errors)

    def test_invalid_cooling_technology(self):
        cfg = _fresh_config()
        cfg.dc_default_technology = "evaporative_cooler"
        errors = cfg.validate()
        assert any("dc_default_technology" in e for e in errors)

    def test_cop_zero(self):
        cfg = _fresh_config()
        cfg.dc_default_cop = 0.0
        errors = cfg.validate()
        assert any("dc_default_cop" in e for e in errors)

    def test_cop_too_high(self):
        cfg = _fresh_config()
        cfg.dc_default_cop = 50.0
        errors = cfg.validate()
        assert any("dc_default_cop" in e for e in errors)

    def test_chp_combined_efficiency_above_1(self):
        cfg = _fresh_config()
        cfg.chp_default_electrical_efficiency = 0.60
        cfg.chp_default_thermal_efficiency = 0.60
        errors = cfg.validate()
        assert any("Combined CHP" in e for e in errors)

    def test_invalid_uncertainty_method(self):
        cfg = _fresh_config()
        cfg.unc_default_method = "bayesian"
        errors = cfg.validate()
        assert any("unc_default_method" in e for e in errors)

    def test_iterations_negative(self):
        cfg = _fresh_config()
        cfg.unc_default_iterations = -100
        errors = cfg.validate()
        assert any("unc_default_iterations" in e for e in errors)

    def test_confidence_level_above_1(self):
        cfg = _fresh_config()
        cfg.unc_default_confidence_level = 1.5
        errors = cfg.validate()
        assert any("unc_default_confidence_level" in e for e in errors)

    def test_invalid_framework_in_list(self):
        cfg = _fresh_config()
        cfg.comp_enabled_frameworks = ["unknown_framework"]
        errors = cfg.validate()
        assert any("not valid" in e for e in errors)

    def test_empty_frameworks_list(self):
        cfg = _fresh_config()
        cfg.comp_enabled_frameworks = []
        errors = cfg.validate()
        assert any("comp_enabled_frameworks" in e for e in errors)

    def test_require_all_without_strict(self):
        cfg = _fresh_config()
        cfg.comp_require_all_frameworks = True
        cfg.comp_strict_mode = False
        errors = cfg.validate()
        assert any("comp_require_all_frameworks" in e for e in errors)

    def test_worker_threads_zero(self):
        cfg = _fresh_config()
        cfg.worker_threads = 0
        errors = cfg.validate()
        assert any("worker_threads" in e for e in errors)

    def test_health_check_interval_zero(self):
        cfg = _fresh_config()
        cfg.health_check_interval = 0
        errors = cfg.validate()
        assert any("health_check_interval" in e for e in errors)

    def test_api_prefix_no_slash(self):
        cfg = _fresh_config()
        cfg.api_prefix = "no-slash"
        errors = cfg.validate()
        assert any("api_prefix" in e for e in errors)

    def test_api_rate_limit_zero(self):
        cfg = _fresh_config()
        cfg.api_rate_limit = 0
        errors = cfg.validate()
        assert any("api_rate_limit" in e for e in errors)

    def test_empty_cors_origins(self):
        cfg = _fresh_config()
        cfg.cors_origins = []
        errors = cfg.validate()
        assert any("cors_origins" in e for e in errors)


# ===================================================================
# Section 5: Serialisation (to_dict / from_dict)
# ===================================================================


class TestSerialisation:
    """Tests for to_dict() and from_dict() methods."""

    def test_to_dict_returns_dict(self):
        cfg = _fresh_config()
        d = cfg.to_dict()
        assert isinstance(d, dict)

    def test_to_dict_contains_all_sections(self):
        cfg = _fresh_config()
        d = cfg.to_dict()
        # Spot-check keys from every section
        assert "service_name" in d
        assert "db_host" in d
        assert "calc_default_gwp_source" in d
        assert "steam_default_fuel_type" in d
        assert "dh_default_region" in d
        assert "dc_default_technology" in d
        assert "chp_default_alloc_method" in d
        assert "unc_default_method" in d
        assert "comp_enabled_frameworks" in d
        assert "enable_metrics" in d
        assert "enable_provenance" in d
        assert "enable_auth" in d
        assert "api_prefix" in d
        assert "enabled" in d

    def test_to_dict_password_redacted(self):
        cfg = _fresh_config()
        cfg.db_password = "supersecret"
        d = cfg.to_dict()
        assert d["db_password"] == "***"

    def test_to_dict_empty_password_not_redacted(self):
        cfg = _fresh_config()
        cfg.db_password = ""
        d = cfg.to_dict()
        assert d["db_password"] == ""

    def test_from_dict_applies_overrides(self):
        d = {"calc_default_gwp_source": "AR5", "calc_decimal_places": 12}
        cfg = SteamHeatPurchaseConfig.from_dict(d)
        assert cfg.calc_default_gwp_source == "AR5"
        assert cfg.calc_decimal_places == 12

    def test_from_dict_preserves_non_specified_defaults(self):
        d = {"calc_default_gwp_source": "AR5"}
        cfg = SteamHeatPurchaseConfig.from_dict(d)
        assert cfg.db_host == "localhost"

    def test_from_dict_skip_redacted_password(self):
        d = {"db_password": "***"}
        cfg = SteamHeatPurchaseConfig.from_dict(d)
        # Redacted password should not overwrite
        assert cfg.db_password == ""

    def test_set_config_with_overrides(self):
        cfg = set_config(calc_default_gwp_source="AR4")
        assert cfg.calc_default_gwp_source == "AR4"

    def test_set_config_resets_first(self):
        cfg1 = get_config()
        cfg2 = set_config()
        assert cfg1 is not cfg2


# ===================================================================
# Section 6: Accessor methods
# ===================================================================


class TestAccessorMethods:
    """Tests for configuration accessor methods."""

    def test_get_db_url(self):
        cfg = _fresh_config()
        url = cfg.get_db_url()
        assert url.startswith("postgresql://")
        assert "greenlang" in url

    def test_get_db_url_with_password(self):
        cfg = _fresh_config()
        cfg.db_password = "s3cr3t"
        url = cfg.get_db_url()
        assert "s3cr3t" in url

    def test_get_async_db_url(self):
        cfg = _fresh_config()
        url = cfg.get_async_db_url()
        assert url.startswith("postgresql+asyncpg://")

    def test_get_enabled_frameworks(self):
        cfg = _fresh_config()
        fws = cfg.get_enabled_frameworks()
        assert isinstance(fws, list)
        assert len(fws) == 7

    def test_get_enabled_frameworks_returns_copy(self):
        cfg = _fresh_config()
        fws = cfg.get_enabled_frameworks()
        fws.append("extra")
        assert "extra" not in cfg.get_enabled_frameworks()

    def test_is_framework_enabled_true(self):
        cfg = _fresh_config()
        assert cfg.is_framework_enabled("cdp") is True

    def test_is_framework_enabled_case_insensitive(self):
        cfg = _fresh_config()
        assert cfg.is_framework_enabled("CDP") is True

    def test_is_framework_enabled_false(self):
        cfg = _fresh_config()
        assert cfg.is_framework_enabled("unknown_framework") is False

    def test_get_rounding_mode(self):
        cfg = _fresh_config()
        mode = cfg.get_rounding_mode()
        assert mode is not None

    def test_get_steam_config(self):
        cfg = _fresh_config()
        steam = cfg.get_steam_config()
        assert steam["default_fuel_type"] == "natural_gas"
        assert steam["default_steam_pressure"] == "MEDIUM"
        assert steam["default_steam_quality"] == "SATURATED"
        assert steam["enable_multi_fuel_blend"] is True
        assert steam["max_fuel_types_per_blend"] == 5
        assert steam["default_boiler_efficiency"] == 0.85

    def test_get_district_heating_config(self):
        cfg = _fresh_config()
        dh = cfg.get_district_heating_config()
        assert dh["default_region"] == "global_default"
        assert dh["default_network_type"] == "MUNICIPAL"
        assert dh["enable_distribution_loss"] is True
        assert dh["enable_supplier_ef"] is True

    def test_get_district_cooling_config(self):
        cfg = _fresh_config()
        dc = cfg.get_district_cooling_config()
        assert dc["default_technology"] == "centrifugal_chiller"
        assert dc["default_cop"] == 6.0
        assert dc["default_grid_ef_kwh"] == 0.436

    def test_get_chp_config(self):
        cfg = _fresh_config()
        chp = cfg.get_chp_config()
        assert chp["default_alloc_method"] == "EFFICIENCY"
        assert chp["default_electrical_efficiency"] == 0.35
        assert chp["default_thermal_efficiency"] == 0.45
        assert chp["reference_electrical_efficiency"] == 0.525
        assert chp["reference_thermal_efficiency"] == 0.90

    def test_get_uncertainty_params(self):
        cfg = _fresh_config()
        params = cfg.get_uncertainty_params()
        assert params["default_method"] == "monte_carlo"
        assert params["iterations"] == 10000
        assert params["confidence_level"] == 0.95
        assert params["seed"] == 42

    def test_get_calculation_config(self):
        cfg = _fresh_config()
        calc = cfg.get_calculation_config()
        assert calc["gwp_source"] == "AR6"
        assert calc["data_quality_tier"] == "TIER_1"
        assert calc["decimal_places"] == 8

    def test_get_db_pool_params(self):
        cfg = _fresh_config()
        params = cfg.get_db_pool_params()
        assert params["min_size"] == 2
        assert params["max_size"] == 10
        assert "conninfo" in params

    def test_get_api_config(self):
        cfg = _fresh_config()
        api = cfg.get_api_config()
        assert api["prefix"] == "/api/v1/steam-heat-purchase"
        assert api["rate_limit"] == 100
        assert api["cors_origins"] == ["*"]
        assert api["enable_docs"] is True
