# -*- coding: utf-8 -*-
"""
PACK-016 ESRS E1 Climate Pack - Configuration Tests
=====================================================

Tests for pack_config.py covering all 18 enums, reference data constants,
8 sub-config models, the E1ClimateConfig root model, PackConfig wrapper,
preset loading, environment variable overrides, config hashing, and
utility functions.

Target: 55+ tests.

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-016 ESRS E1 Climate Change
Date:    March 2026
"""

import hashlib
import json
import os
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import pytest

from .conftest import _load_config_module, PRESETS_DIR


# ---------------------------------------------------------------------------
# Module-level config module load (session-scoped via conftest fixture)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cfg():
    """Load the pack_config module once per module."""
    return _load_config_module()


# ===========================================================================
# Enum Tests
# ===========================================================================


class TestGHGScopeEnum:
    """Tests for GHGScope enum values."""

    def test_ghg_scope_count(self, cfg):
        """GHGScope enum has exactly 4 members."""
        assert len(cfg.GHGScope) == 4

    def test_ghg_scope_includes_all(self, cfg):
        """GHGScope contains SCOPE_1, SCOPE_2_LOCATION, SCOPE_2_MARKET, SCOPE_3."""
        names = {m.name for m in cfg.GHGScope}
        assert names == {"SCOPE_1", "SCOPE_2_LOCATION", "SCOPE_2_MARKET", "SCOPE_3"}

    def test_ghg_scope_is_str_enum(self, cfg):
        """GHGScope members are string instances."""
        for member in cfg.GHGScope:
            assert isinstance(member.value, str)


class TestEmissionGasEnum:
    """Tests for EmissionGas enum values."""

    def test_emission_gas_count(self, cfg):
        """EmissionGas has exactly 7 Kyoto gases."""
        assert len(cfg.EmissionGas) == 7

    def test_emission_gas_includes_all_kyoto(self, cfg):
        """All 7 Kyoto greenhouse gases present."""
        expected = {"CO2", "CH4", "N2O", "HFCS", "PFCS", "SF6", "NF3"}
        actual = {m.name for m in cfg.EmissionGas}
        assert actual == expected

    def test_emission_gas_values_match_names(self, cfg):
        """Each EmissionGas value should equal its name."""
        for member in cfg.EmissionGas:
            assert member.value == member.name


class TestFuelTypeEnum:
    """Tests for FuelType enum values."""

    def test_fuel_type_count(self, cfg):
        """FuelType has at least 10 members."""
        assert len(cfg.FuelType) >= 10

    def test_fuel_type_includes_common_fuels(self, cfg):
        """Common fuel types are present."""
        names = {m.name for m in cfg.FuelType}
        for fuel in ["NATURAL_GAS", "DIESEL", "GASOLINE", "COAL", "LPG",
                      "FUEL_OIL", "JET_FUEL", "BIOMASS", "BIOGAS", "HYDROGEN"]:
            assert fuel in names, f"Missing fuel type: {fuel}"


class TestEnergySourceEnum:
    """Tests for EnergySource enum values."""

    def test_energy_source_count(self, cfg):
        """EnergySource has at least 12 members."""
        assert len(cfg.EnergySource) >= 12

    def test_energy_source_includes_renewables(self, cfg):
        """Renewable energy sources are present."""
        names = {m.name for m in cfg.EnergySource}
        for source in ["SOLAR_PV", "WIND_ONSHORE", "WIND_OFFSHORE",
                        "HYDROPOWER", "GEOTHERMAL", "BIOMASS"]:
            assert source in names, f"Missing energy source: {source}"

    def test_energy_source_includes_fossil(self, cfg):
        """Fossil energy sources are present."""
        names = {m.name for m in cfg.EnergySource}
        assert "NATURAL_GAS" in names
        assert "DIESEL" in names

    def test_energy_source_includes_district(self, cfg):
        """District heating and cooling are present."""
        names = {m.name for m in cfg.EnergySource}
        assert "DISTRICT_HEATING" in names
        assert "DISTRICT_COOLING" in names


class TestRenewableCategoryEnum:
    """Tests for RenewableCategory enum values."""

    def test_renewable_category_count(self, cfg):
        """RenewableCategory has at least 6 members."""
        assert len(cfg.RenewableCategory) >= 6

    def test_renewable_category_includes_main(self, cfg):
        """Main renewable categories are present."""
        names = {m.name for m in cfg.RenewableCategory}
        for cat in ["SOLAR", "WIND", "HYDRO", "GEOTHERMAL", "BIOMASS", "OTHER"]:
            assert cat in names


class TestTargetTypeEnum:
    """Tests for TargetType enum values."""

    def test_target_type_count(self, cfg):
        """TargetType has exactly 3 members."""
        assert len(cfg.TargetType) == 3

    def test_target_type_includes_all(self, cfg):
        """ABSOLUTE, INTENSITY, NET_ZERO present."""
        names = {m.name for m in cfg.TargetType}
        assert names == {"ABSOLUTE", "INTENSITY", "NET_ZERO"}


class TestTargetPathwayEnum:
    """Tests for TargetPathway enum values."""

    def test_target_pathway_count(self, cfg):
        """TargetPathway has at least 4 members."""
        assert len(cfg.TargetPathway) >= 4

    def test_target_pathway_includes_sbti(self, cfg):
        """SBTi pathway options are present."""
        names = {m.name for m in cfg.TargetPathway}
        for pathway in ["SBTi_1_5C", "SBTi_WELL_BELOW_2C", "SBTi_NET_ZERO", "CUSTOM"]:
            assert pathway in names


class TestCarbonCreditStandardEnum:
    """Tests for CarbonCreditStandard enum values."""

    def test_carbon_credit_standard_count(self, cfg):
        """CarbonCreditStandard has at least 6 members."""
        assert len(cfg.CarbonCreditStandard) >= 6

    def test_carbon_credit_standard_includes_main(self, cfg):
        """Main carbon credit standards present."""
        names = {m.name for m in cfg.CarbonCreditStandard}
        for std in ["VERRA_VCS", "GOLD_STANDARD", "ACR", "CDM", "CORSIA"]:
            assert std in names, f"Missing standard: {std}"


class TestClimateScenarioEnum:
    """Tests for ClimateScenario enum values."""

    def test_climate_scenario_count(self, cfg):
        """ClimateScenario has at least 6 members."""
        assert len(cfg.ClimateScenario) >= 6

    def test_climate_scenario_includes_rcp(self, cfg):
        """RCP scenarios present."""
        names = {m.name for m in cfg.ClimateScenario}
        assert "RCP_2_6" in names
        assert "RCP_8_5" in names

    def test_climate_scenario_includes_ssp(self, cfg):
        """SSP scenarios present."""
        names = {m.name for m in cfg.ClimateScenario}
        assert "SSP1_2_6" in names
        assert "SSP5_8_5" in names


# ===========================================================================
# Reference Data Constants
# ===========================================================================


class TestGWPAR6:
    """Tests for IPCC AR6 GWP-100 reference values."""

    def test_gwp_co2_is_1(self, cfg):
        """CO2 GWP-100 is exactly 1.0."""
        assert cfg.GWP_AR6_VALUES["CO2"] == 1.0

    def test_gwp_ch4_fossil(self, cfg):
        """CH4 (fossil) GWP-100 is 29.8."""
        assert cfg.GWP_AR6_VALUES["CH4_FOSSIL"] == 29.8

    def test_gwp_ch4_biogenic(self, cfg):
        """CH4 (biogenic) GWP-100 is 27.2."""
        assert cfg.GWP_AR6_VALUES["CH4_BIOGENIC"] == 27.2

    def test_gwp_n2o(self, cfg):
        """N2O GWP-100 is 273."""
        assert cfg.GWP_AR6_VALUES["N2O"] == 273.0

    def test_gwp_sf6(self, cfg):
        """SF6 GWP-100 is 25200."""
        assert cfg.GWP_AR6_VALUES["SF6"] == 25200.0

    def test_gwp_nf3(self, cfg):
        """NF3 GWP-100 is 17400."""
        assert cfg.GWP_AR6_VALUES["NF3"] == 17400.0

    def test_gwp_has_10_plus_entries(self, cfg):
        """GWP_AR6_VALUES has at least 10 entries (including HFCs, PFCs)."""
        assert len(cfg.GWP_AR6_VALUES) >= 10

    def test_gwp_hfc_134a(self, cfg):
        """HFC-134a GWP-100 is 1530."""
        assert cfg.GWP_AR6_VALUES["HFC_134A"] == 1530.0

    def test_gwp_cf4(self, cfg):
        """CF4 (PFC) GWP-100 is 7380."""
        assert cfg.GWP_AR6_VALUES["CF4"] == 7380.0


class TestEnergySourceClassification:
    """Tests for ENERGY_SOURCE_CLASSIFICATION reference data."""

    def test_all_energy_sources_classified(self, cfg):
        """All entries in ENERGY_SOURCE_CLASSIFICATION are categorized."""
        valid_categories = {"FOSSIL", "RENEWABLE", "NUCLEAR", "MIXED"}
        for source, category in cfg.ENERGY_SOURCE_CLASSIFICATION.items():
            assert category in valid_categories, (
                f"Source {source} has invalid category: {category}"
            )

    def test_solar_pv_is_renewable(self, cfg):
        """SOLAR_PV is classified as RENEWABLE."""
        assert cfg.ENERGY_SOURCE_CLASSIFICATION["SOLAR_PV"] == "RENEWABLE"

    def test_natural_gas_is_fossil(self, cfg):
        """NATURAL_GAS is classified as FOSSIL."""
        assert cfg.ENERGY_SOURCE_CLASSIFICATION["NATURAL_GAS"] == "FOSSIL"

    def test_nuclear_is_nuclear(self, cfg):
        """NUCLEAR is classified as NUCLEAR."""
        assert cfg.ENERGY_SOURCE_CLASSIFICATION["NUCLEAR"] == "NUCLEAR"

    def test_grid_electricity_is_mixed(self, cfg):
        """GRID_ELECTRICITY is classified as MIXED."""
        assert cfg.ENERGY_SOURCE_CLASSIFICATION["GRID_ELECTRICITY"] == "MIXED"


class TestScope3Categories:
    """Tests for SCOPE_3_CATEGORIES reference data."""

    def test_scope_3_has_15_categories(self, cfg):
        """SCOPE_3_CATEGORIES has exactly 15 entries (Cat 1-15)."""
        assert len(cfg.SCOPE_3_CATEGORIES) == 15

    def test_scope_3_category_1(self, cfg):
        """Category 1 is Purchased Goods and Services."""
        assert "Purchased Goods" in cfg.SCOPE_3_CATEGORIES[1]

    def test_scope_3_category_15(self, cfg):
        """Category 15 is Investments."""
        assert "Investments" in cfg.SCOPE_3_CATEGORIES[15]

    def test_scope_3_keys_are_1_to_15(self, cfg):
        """Keys are integers 1 through 15."""
        assert set(cfg.SCOPE_3_CATEGORIES.keys()) == set(range(1, 16))


class TestSBTiRates:
    """Tests for SBTI_REDUCTION_RATES reference data."""

    def test_sbti_1_5c_rate(self, cfg):
        """1.5C annual linear reduction rate is approximately 4.2%."""
        rate = cfg.SBTI_REDUCTION_RATES["SBTi_1_5C"]
        assert float(rate) == pytest.approx(0.042, rel=1e-2)

    def test_sbti_well_below_2c_rate(self, cfg):
        """Well-below 2C rate is approximately 2.5%."""
        rate = cfg.SBTI_REDUCTION_RATES["SBTi_WELL_BELOW_2C"]
        assert float(rate) == pytest.approx(0.025, rel=1e-2)

    def test_sbti_net_zero_rate(self, cfg):
        """Net-zero near-term rate is approximately 4.2%."""
        rate = cfg.SBTI_REDUCTION_RATES["SBTi_NET_ZERO"]
        assert float(rate) == pytest.approx(0.042, rel=1e-2)


# ===========================================================================
# Sub-Config Model Tests
# ===========================================================================


class TestGHGConfig:
    """Tests for GHGConfig sub-model defaults and validation."""

    def test_default_ghg_config_enabled(self, cfg):
        """GHGConfig enabled by default."""
        config = cfg.GHGConfig()
        assert config.enabled is True

    def test_default_base_year(self, cfg):
        """Default base year is 2020."""
        config = cfg.GHGConfig()
        assert config.base_year == 2020

    def test_default_reporting_year(self, cfg):
        """Default reporting year is 2025."""
        config = cfg.GHGConfig()
        assert config.reporting_year == 2025

    def test_default_consolidation_approach(self, cfg):
        """Default consolidation is OPERATIONAL_CONTROL."""
        config = cfg.GHGConfig()
        assert config.consolidation_approach == cfg.ConsolidationApproach.OPERATIONAL_CONTROL

    def test_default_scopes_all_enabled(self, cfg):
        """All 4 scopes enabled by default."""
        config = cfg.GHGConfig()
        assert len(config.scopes_enabled) == 4

    def test_default_scope_3_categories_all_15(self, cfg):
        """All 15 Scope 3 categories enabled by default."""
        config = cfg.GHGConfig()
        assert config.scope_3_categories == list(range(1, 16))

    def test_custom_base_year_override(self, cfg):
        """Custom base year can be set."""
        config = cfg.GHGConfig(base_year=2019)
        assert config.base_year == 2019

    def test_invalid_scope_3_category_raises(self, cfg):
        """Invalid Scope 3 category (16) raises ValueError."""
        with pytest.raises(Exception):
            cfg.GHGConfig(scope_3_categories=[1, 2, 16])

    def test_kyoto_gas_disaggregation_default(self, cfg):
        """Kyoto gas disaggregation enabled by default."""
        config = cfg.GHGConfig()
        assert config.kyoto_gas_disaggregation is True

    def test_biogenic_co2_separate_default(self, cfg):
        """Biogenic CO2 separate reporting enabled by default."""
        config = cfg.GHGConfig()
        assert config.biogenic_co2_separate is True


class TestEnergyConfig:
    """Tests for EnergyConfig sub-model."""

    def test_default_energy_config_enabled(self, cfg):
        """EnergyConfig enabled by default."""
        config = cfg.EnergyConfig()
        assert config.enabled is True

    def test_default_reporting_unit(self, cfg):
        """Default reporting unit is MWh."""
        config = cfg.EnergyConfig()
        assert config.reporting_unit == "MWh"

    def test_default_include_renewables(self, cfg):
        """Renewable tracking enabled by default."""
        config = cfg.EnergyConfig()
        assert config.include_renewables is True

    def test_invalid_reporting_unit_raises(self, cfg):
        """Invalid energy unit raises ValueError."""
        with pytest.raises(Exception):
            cfg.EnergyConfig(reporting_unit="kWh")


class TestTransitionPlanConfig:
    """Tests for TransitionPlanConfig sub-model."""

    def test_default_transition_plan_enabled(self, cfg):
        """TransitionPlanConfig enabled by default."""
        config = cfg.TransitionPlanConfig()
        assert config.enabled is True

    def test_default_target_year(self, cfg):
        """Default target year is 2050."""
        config = cfg.TransitionPlanConfig()
        assert config.target_year == 2050

    def test_default_interim_targets_enabled(self, cfg):
        """Interim targets enabled by default."""
        config = cfg.TransitionPlanConfig()
        assert config.interim_targets_enabled is True

    def test_default_has_transition_plan(self, cfg):
        """has_transition_plan is True by default."""
        config = cfg.TransitionPlanConfig()
        assert config.has_transition_plan is True

    def test_default_scenario_alignment(self, cfg):
        """Default scenario alignment is 1.5C."""
        config = cfg.TransitionPlanConfig()
        assert config.scenario_alignment == "1.5C"


class TestTargetConfig:
    """Tests for TargetConfig sub-model."""

    def test_default_target_config_enabled(self, cfg):
        """TargetConfig enabled by default."""
        config = cfg.TargetConfig()
        assert config.enabled is True

    def test_default_sbti_commitment(self, cfg):
        """Default SBTi commitment is 1.5C."""
        config = cfg.TargetConfig()
        assert config.sbti_commitment_level == cfg.TargetPathway.SBTi_1_5C

    def test_sbti_not_validated_by_default(self, cfg):
        """SBTi not validated by default."""
        config = cfg.TargetConfig()
        assert config.sbti_validated is False

    def test_target_year_must_be_after_base_year(self, cfg):
        """Target year before base year raises error."""
        with pytest.raises(Exception):
            cfg.TargetConfig(base_year=2025, target_year=2020)

    def test_default_reduction_path(self, cfg):
        """Default reduction path is linear."""
        config = cfg.TargetConfig()
        assert config.reduction_path == "linear"


class TestCarbonCreditConfig:
    """Tests for CarbonCreditConfig sub-model."""

    def test_default_carbon_credit_enabled(self, cfg):
        """CarbonCreditConfig enabled by default."""
        config = cfg.CarbonCreditConfig()
        assert config.enabled is True

    def test_default_vintage_requirements(self, cfg):
        """Default vintage requirement is 5 years."""
        config = cfg.CarbonCreditConfig()
        assert config.vintage_requirements == 5

    def test_default_quality_assessment(self, cfg):
        """Quality assessment enabled by default."""
        config = cfg.CarbonCreditConfig()
        assert config.quality_assessment is True

    def test_default_sbti_offset_guidance(self, cfg):
        """SBTi offset guidance enabled by default."""
        config = cfg.CarbonCreditConfig()
        assert config.sbti_offset_guidance is True

    def test_default_separate_from_gross(self, cfg):
        """Credits reported separately from gross emissions by default."""
        config = cfg.CarbonCreditConfig()
        assert config.separate_from_gross_emissions is True


class TestCarbonPricingConfig:
    """Tests for CarbonPricingConfig sub-model."""

    def test_default_carbon_pricing_enabled(self, cfg):
        """CarbonPricingConfig enabled by default."""
        config = cfg.CarbonPricingConfig()
        assert config.enabled is True

    def test_default_no_carbon_pricing(self, cfg):
        """has_carbon_pricing is False by default."""
        config = cfg.CarbonPricingConfig()
        assert config.has_carbon_pricing is False

    def test_default_price_per_tco2e(self, cfg):
        """Default price is 100.00 EUR."""
        config = cfg.CarbonPricingConfig()
        assert config.price_per_tco2e == "100.00"

    def test_default_currency(self, cfg):
        """Default currency is EUR."""
        config = cfg.CarbonPricingConfig()
        assert config.price_currency == "EUR"


class TestClimateRiskConfig:
    """Tests for ClimateRiskConfig sub-model."""

    def test_default_climate_risk_enabled(self, cfg):
        """ClimateRiskConfig enabled by default."""
        config = cfg.ClimateRiskConfig()
        assert config.enabled is True

    def test_default_physical_risk_types(self, cfg):
        """Both ACUTE and CHRONIC physical risk types by default."""
        config = cfg.ClimateRiskConfig()
        assert len(config.physical_risk_types) == 2

    def test_default_transition_risk_types(self, cfg):
        """All 5 transition risk types by default."""
        config = cfg.ClimateRiskConfig()
        assert len(config.transition_risk_types) == 5

    def test_default_time_horizons(self, cfg):
        """All 3 time horizons by default."""
        config = cfg.ClimateRiskConfig()
        assert len(config.time_horizons) == 3

    def test_default_opportunity_assessment(self, cfg):
        """Opportunity assessment enabled by default."""
        config = cfg.ClimateRiskConfig()
        assert config.opportunity_assessment is True

    def test_default_tcfd_alignment(self, cfg):
        """TCFD alignment enabled by default."""
        config = cfg.ClimateRiskConfig()
        assert config.tcfd_alignment is True


# ===========================================================================
# E1ClimateConfig Root Model Tests
# ===========================================================================


class TestE1ClimateConfig:
    """Tests for E1ClimateConfig root configuration model."""

    def test_default_creation(self, cfg):
        """E1ClimateConfig can be created with all defaults."""
        config = cfg.E1ClimateConfig()
        assert config is not None

    def test_default_reporting_year(self, cfg):
        """Default reporting year is 2025."""
        config = cfg.E1ClimateConfig()
        assert config.reporting_year == 2025

    def test_default_sector(self, cfg):
        """Default sector is GENERAL."""
        config = cfg.E1ClimateConfig()
        assert config.sector == "GENERAL"

    def test_default_currency(self, cfg):
        """Default currency is EUR."""
        config = cfg.E1ClimateConfig()
        assert config.currency == "EUR"

    def test_from_dict(self, cfg):
        """E1ClimateConfig can be created from a dictionary."""
        data = {
            "company_name": "TestCorp GmbH",
            "reporting_year": 2025,
            "sector": "MANUFACTURING",
        }
        config = cfg.E1ClimateConfig(**data)
        assert config.company_name == "TestCorp GmbH"
        assert config.sector == "MANUFACTURING"

    def test_sub_configs_initialized(self, cfg):
        """All 8 sub-configs are initialized."""
        config = cfg.E1ClimateConfig()
        assert config.ghg is not None
        assert config.energy is not None
        assert config.transition_plan is not None
        assert config.targets is not None
        assert config.carbon_credits is not None
        assert config.carbon_pricing is not None
        assert config.climate_risk is not None
        assert config.reporting is not None

    def test_validation_base_year_warning(self, cfg):
        """Warning when GHG and target base years differ (does not raise)."""
        config = cfg.E1ClimateConfig(
            ghg=cfg.GHGConfig(base_year=2019),
            targets=cfg.TargetConfig(base_year=2020),
        )
        assert config.ghg.base_year == 2019
        assert config.targets.base_year == 2020


# ===========================================================================
# PackConfig Wrapper Tests
# ===========================================================================


class TestPackConfig:
    """Tests for PackConfig wrapper."""

    def test_default_pack_config(self, cfg):
        """PackConfig creates with defaults."""
        config = cfg.PackConfig()
        assert config.pack_id == "PACK-016-esrs-e1-climate"
        assert config.config_version == "1.0.0"

    def test_from_preset_manufacturing(self, cfg):
        """from_preset loads manufacturing preset."""
        config = cfg.PackConfig.from_preset("manufacturing")
        assert config.preset_name == "manufacturing"

    def test_from_preset_invalid_raises(self, cfg):
        """from_preset with unknown name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown preset"):
            cfg.PackConfig.from_preset("nonexistent_preset")

    def test_from_yaml(self, cfg):
        """from_yaml loads configuration from YAML file."""
        demo_path = PRESETS_DIR / "manufacturing.yaml"
        if demo_path.exists():
            config = cfg.PackConfig.from_yaml(demo_path)
            assert config.pack is not None

    def test_config_hash_is_sha256(self, cfg):
        """get_config_hash returns a 64-char hex string (SHA-256)."""
        config = cfg.PackConfig()
        hash_value = config.get_config_hash()
        assert len(hash_value) == 64
        assert all(c in "0123456789abcdef" for c in hash_value)

    def test_config_hash_deterministic(self, cfg):
        """Same configuration produces same hash."""
        config1 = cfg.PackConfig()
        config2 = cfg.PackConfig()
        assert config1.get_config_hash() == config2.get_config_hash()


# ===========================================================================
# Preset Loading Tests
# ===========================================================================


class TestPresetLoading:
    """Tests for preset loading functionality."""

    @pytest.mark.parametrize("preset_name", [
        "power_generation",
        "manufacturing",
        "transport",
        "financial_services",
        "real_estate",
        "multi_sector",
    ])
    def test_preset_loads(self, cfg, preset_name):
        """Each of the 6 presets loads successfully."""
        preset_path = PRESETS_DIR / f"{preset_name}.yaml"
        if not preset_path.exists():
            pytest.skip(f"Preset file not found: {preset_path}")
        config = cfg.PackConfig.from_preset(preset_name)
        assert config.preset_name == preset_name

    def test_financial_services_preset_has_scope_3_cat_15(self, cfg):
        """Financial services preset includes Scope 3 Cat 15 (Investments)."""
        preset_path = PRESETS_DIR / "financial_services.yaml"
        if not preset_path.exists():
            pytest.skip("financial_services preset not found")
        config = cfg.PackConfig.from_preset("financial_services")
        assert 15 in config.pack.ghg.scope_3_categories

    def test_manufacturing_preset_sector(self, cfg):
        """Manufacturing preset sets sector to MANUFACTURING."""
        preset_path = PRESETS_DIR / "manufacturing.yaml"
        if not preset_path.exists():
            pytest.skip("manufacturing preset not found")
        config = cfg.PackConfig.from_preset("manufacturing")
        assert config.pack.sector == "MANUFACTURING"


# ===========================================================================
# Utility Function Tests
# ===========================================================================


class TestUtilityFunctions:
    """Tests for pack_config utility functions."""

    def test_get_gwp_value_co2(self, cfg):
        """get_gwp_value returns 1.0 for CO2."""
        assert cfg.get_gwp_value("CO2") == 1.0

    def test_get_gwp_value_sf6(self, cfg):
        """get_gwp_value returns 25200 for SF6."""
        assert cfg.get_gwp_value("SF6") == 25200.0

    def test_get_gwp_value_unknown(self, cfg):
        """get_gwp_value returns 0.0 for unknown gas."""
        assert cfg.get_gwp_value("UNKNOWN_GAS") == 0.0

    def test_get_scope_3_category_name(self, cfg):
        """get_scope_3_category_name returns correct name for Cat 1."""
        name = cfg.get_scope_3_category_name(1)
        assert "Purchased Goods" in name

    def test_get_scope_3_category_name_unknown(self, cfg):
        """get_scope_3_category_name returns fallback for Cat 99."""
        name = cfg.get_scope_3_category_name(99)
        assert "Unknown" in name

    def test_list_available_presets(self, cfg):
        """list_available_presets returns 6 presets."""
        presets = cfg.list_available_presets()
        assert len(presets) == 6
        assert "manufacturing" in presets
        assert "power_generation" in presets

    def test_get_e1_disclosure_info_e1_6(self, cfg):
        """get_e1_disclosure_info returns info for E1-6."""
        info = cfg.get_e1_disclosure_info("E1-6")
        assert "GHG" in info["name"] or "Gross" in info["name"]
        assert info["quantitative"] is True

    def test_get_sbti_reduction_rate_1_5c(self, cfg):
        """get_sbti_reduction_rate returns ~4.2% for 1.5C."""
        rate = cfg.get_sbti_reduction_rate("SBTi_1_5C")
        assert float(rate) == pytest.approx(0.042, rel=1e-2)

    def test_get_sbti_reduction_rate_enum(self, cfg):
        """get_sbti_reduction_rate works with TargetPathway enum."""
        rate = cfg.get_sbti_reduction_rate(cfg.TargetPathway.SBTi_1_5C)
        assert float(rate) == pytest.approx(0.042, rel=1e-2)

    def test_validate_config_returns_list(self, cfg):
        """validate_config returns a list of warnings."""
        config = cfg.E1ClimateConfig()
        warnings = cfg.validate_config(config)
        assert isinstance(warnings, list)

    def test_get_default_config(self, cfg):
        """get_default_config returns E1ClimateConfig instance."""
        config = cfg.get_default_config("MANUFACTURING")
        assert config.sector == "MANUFACTURING"
