# -*- coding: utf-8 -*-
"""
Unit tests for Steam/Heat Purchase Agent data models - AGENT-MRV-011.

Tests all 18 enumerations, 6 constant tables, and 20 Pydantic data models
defined in ``greenlang.agents.mrv.steam_heat_purchase.models``.

Coverage targets:
- Every enum: value membership, str conversion, invalid value rejection
- Every constant table: key count, value correctness, Decimal types
- Every Pydantic model: creation, frozen immutability, validators, defaults

Test count target: ~150 tests.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List

import pytest

try:
    from greenlang.agents.mrv.steam_heat_purchase.models import (
        # Enums (18)
        EnergyType,
        FuelType,
        CoolingTechnology,
        CHPAllocMethod,
        CalculationMethod,
        EmissionGas,
        GWPSource,
        ComplianceStatus,
        DataQualityTier,
        EnergyUnit,
        TemperatureUnit,
        SteamPressure,
        SteamQuality,
        NetworkType,
        FacilityType,
        ReportingPeriod,
        AggregationType,
        BatchStatus,
        # Constants
        GWP_VALUES,
        FUEL_EMISSION_FACTORS,
        DISTRICT_HEATING_FACTORS,
        COOLING_SYSTEM_FACTORS,
        COOLING_ENERGY_SOURCE,
        CHP_DEFAULT_EFFICIENCIES,
        UNIT_CONVERSIONS,
        VERSION,
        MAX_CALCULATIONS_PER_BATCH,
        MAX_GASES_PER_RESULT,
        MAX_TRACE_STEPS,
        MAX_FACILITIES_PER_TENANT,
        DEFAULT_MONTE_CARLO_ITERATIONS,
        DEFAULT_CONFIDENCE_LEVEL,
        TABLE_PREFIX,
        # Models (20)
        FuelEmissionFactor,
        DistrictHeatingFactor,
        CoolingSystemFactor,
        CHPParameters,
        FacilityInfo,
        SteamSupplier,
        SteamCalculationRequest,
        HeatingCalculationRequest,
        CoolingCalculationRequest,
        CHPAllocationRequest,
        GasEmissionDetail,
        CalculationResult,
        CHPAllocationResult,
        BatchCalculationRequest,
        BatchCalculationResult,
        UncertaintyRequest,
        UncertaintyResult,
        ComplianceCheckResult,
        AggregationRequest,
        AggregationResult,
    )

    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not MODELS_AVAILABLE,
    reason="greenlang.agents.mrv.steam_heat_purchase.models not importable",
)


# ===================================================================
# Section 1: Enumeration tests (18 enums)
# ===================================================================


class TestEnergyType:
    """Tests for the EnergyType enumeration."""

    def test_all_values_present(self):
        members = {e.value for e in EnergyType}
        expected = {
            "steam",
            "district_heating",
            "district_cooling",
            "chp_steam",
            "chp_heating",
        }
        assert members == expected

    def test_member_count(self):
        assert len(EnergyType) == 5

    def test_str_conversion(self):
        assert str(EnergyType.STEAM) == "EnergyType.STEAM" or "steam" in str(
            EnergyType.STEAM
        )
        assert EnergyType.STEAM.value == "steam"

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            EnergyType("nonexistent_type")


class TestFuelType:
    """Tests for the FuelType enumeration."""

    def test_all_14_values_present(self):
        members = {e.value for e in FuelType}
        expected = {
            "natural_gas",
            "fuel_oil_2",
            "fuel_oil_6",
            "coal_bituminous",
            "coal_subbituminous",
            "coal_lignite",
            "lpg",
            "biomass_wood",
            "biomass_biogas",
            "municipal_waste",
            "waste_heat",
            "geothermal",
            "solar_thermal",
            "electric",
        }
        assert members == expected

    def test_member_count(self):
        assert len(FuelType) == 14

    def test_str_is_string(self):
        assert isinstance(FuelType.NATURAL_GAS.value, str)

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            FuelType("hydrogen")


class TestCoolingTechnology:
    """Tests for the CoolingTechnology enumeration."""

    def test_all_9_values_present(self):
        members = {e.value for e in CoolingTechnology}
        expected = {
            "centrifugal_chiller",
            "screw_chiller",
            "reciprocating_chiller",
            "absorption_single",
            "absorption_double",
            "absorption_triple",
            "free_cooling",
            "ice_storage",
            "thermal_storage",
        }
        assert members == expected

    def test_member_count(self):
        assert len(CoolingTechnology) == 9

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            CoolingTechnology("evaporative")


class TestCHPAllocMethod:
    """Tests for the CHPAllocMethod enumeration."""

    def test_all_3_values_present(self):
        members = {e.value for e in CHPAllocMethod}
        expected = {"efficiency", "energy", "exergy"}
        assert members == expected

    def test_member_count(self):
        assert len(CHPAllocMethod) == 3

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            CHPAllocMethod("economic")


class TestCalculationMethod:
    """Tests for the CalculationMethod enumeration."""

    def test_all_4_values_present(self):
        members = {e.value for e in CalculationMethod}
        expected = {"direct_ef", "fuel_based", "cop_based", "chp_allocated"}
        assert members == expected

    def test_member_count(self):
        assert len(CalculationMethod) == 4

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            CalculationMethod("manual")


class TestEmissionGas:
    """Tests for the EmissionGas enumeration."""

    def test_all_5_values_present(self):
        members = {e.value for e in EmissionGas}
        expected = {"CO2", "CH4", "N2O", "CO2e", "biogenic_CO2"}
        assert members == expected

    def test_member_count(self):
        assert len(EmissionGas) == 5

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            EmissionGas("SF6")


class TestGWPSource:
    """Tests for the GWPSource enumeration."""

    def test_all_4_values_present(self):
        members = {e.value for e in GWPSource}
        expected = {"AR4", "AR5", "AR6", "AR6_20YR"}
        assert members == expected

    def test_member_count(self):
        assert len(GWPSource) == 4

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            GWPSource("AR3")


class TestComplianceStatus:
    """Tests for the ComplianceStatus enumeration."""

    def test_all_4_values_present(self):
        members = {e.value for e in ComplianceStatus}
        expected = {"compliant", "non_compliant", "partial", "not_applicable"}
        assert members == expected

    def test_member_count(self):
        assert len(ComplianceStatus) == 4

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            ComplianceStatus("unknown")


class TestDataQualityTier:
    """Tests for the DataQualityTier enumeration."""

    def test_all_3_values_present(self):
        members = {e.value for e in DataQualityTier}
        expected = {"tier_1", "tier_2", "tier_3"}
        assert members == expected

    def test_member_count(self):
        assert len(DataQualityTier) == 3

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            DataQualityTier("tier_4")


class TestEnergyUnit:
    """Tests for the EnergyUnit enumeration."""

    def test_all_6_values_present(self):
        members = {e.value for e in EnergyUnit}
        expected = {"gj", "mwh", "kwh", "mmbtu", "therm", "mj"}
        assert members == expected

    def test_member_count(self):
        assert len(EnergyUnit) == 6

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            EnergyUnit("btu")


class TestTemperatureUnit:
    """Tests for the TemperatureUnit enumeration."""

    def test_all_3_values_present(self):
        members = {e.value for e in TemperatureUnit}
        expected = {"celsius", "fahrenheit", "kelvin"}
        assert members == expected

    def test_member_count(self):
        assert len(TemperatureUnit) == 3

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            TemperatureUnit("rankine")


class TestSteamPressure:
    """Tests for the SteamPressure enumeration."""

    def test_all_4_values_present(self):
        members = {e.value for e in SteamPressure}
        expected = {"low", "medium", "high", "very_high"}
        assert members == expected

    def test_member_count(self):
        assert len(SteamPressure) == 4

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            SteamPressure("ultra_high")


class TestSteamQuality:
    """Tests for the SteamQuality enumeration."""

    def test_all_3_values_present(self):
        members = {e.value for e in SteamQuality}
        expected = {"saturated", "superheated", "wet"}
        assert members == expected

    def test_member_count(self):
        assert len(SteamQuality) == 3

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            SteamQuality("dry")


class TestNetworkType:
    """Tests for the NetworkType enumeration."""

    def test_all_4_values_present(self):
        members = {e.value for e in NetworkType}
        expected = {"municipal", "industrial", "campus", "mixed"}
        assert members == expected

    def test_member_count(self):
        assert len(NetworkType) == 4

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            NetworkType("private")


class TestFacilityType:
    """Tests for the FacilityType enumeration."""

    def test_all_6_values_present(self):
        members = {e.value for e in FacilityType}
        expected = {
            "industrial",
            "commercial",
            "institutional",
            "residential",
            "data_center",
            "campus",
        }
        assert members == expected

    def test_member_count(self):
        assert len(FacilityType) == 6

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            FacilityType("warehouse")


class TestReportingPeriod:
    """Tests for the ReportingPeriod enumeration."""

    def test_all_3_values_present(self):
        members = {e.value for e in ReportingPeriod}
        expected = {"monthly", "quarterly", "annual"}
        assert members == expected

    def test_member_count(self):
        assert len(ReportingPeriod) == 3

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            ReportingPeriod("weekly")


class TestAggregationType:
    """Tests for the AggregationType enumeration."""

    def test_all_5_values_present(self):
        members = {e.value for e in AggregationType}
        expected = {
            "by_facility",
            "by_fuel",
            "by_energy_type",
            "by_supplier",
            "by_period",
        }
        assert members == expected

    def test_member_count(self):
        assert len(AggregationType) == 5

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            AggregationType("by_country")


class TestBatchStatus:
    """Tests for the BatchStatus enumeration."""

    def test_all_5_values_present(self):
        members = {e.value for e in BatchStatus}
        expected = {"pending", "running", "completed", "failed", "partial"}
        assert members == expected

    def test_member_count(self):
        assert len(BatchStatus) == 5

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            BatchStatus("cancelled")


# ===================================================================
# Section 2: Constant table tests (6 tables)
# ===================================================================


class TestGWPValues:
    """Tests for the GWP_VALUES constant table."""

    def test_has_four_sources(self):
        assert set(GWP_VALUES.keys()) == {"AR4", "AR5", "AR6", "AR6_20YR"}

    def test_each_source_has_three_gases(self):
        for source, gases in GWP_VALUES.items():
            assert set(gases.keys()) == {"CO2", "CH4", "N2O"}, (
                f"{source} missing gas keys"
            )

    def test_co2_always_one(self):
        for source in GWP_VALUES:
            assert GWP_VALUES[source]["CO2"] == Decimal("1")

    def test_ar4_ch4_is_25(self):
        assert GWP_VALUES["AR4"]["CH4"] == Decimal("25")

    def test_ar4_n2o_is_298(self):
        assert GWP_VALUES["AR4"]["N2O"] == Decimal("298")

    def test_ar5_ch4_is_28(self):
        assert GWP_VALUES["AR5"]["CH4"] == Decimal("28")

    def test_ar5_n2o_is_265(self):
        assert GWP_VALUES["AR5"]["N2O"] == Decimal("265")

    def test_ar6_ch4_is_27_9(self):
        assert GWP_VALUES["AR6"]["CH4"] == Decimal("27.9")

    def test_ar6_n2o_is_273(self):
        assert GWP_VALUES["AR6"]["N2O"] == Decimal("273")

    def test_ar6_20yr_ch4_is_81_2(self):
        assert GWP_VALUES["AR6_20YR"]["CH4"] == Decimal("81.2")

    def test_ar6_20yr_n2o_is_273(self):
        assert GWP_VALUES["AR6_20YR"]["N2O"] == Decimal("273")

    def test_all_values_are_decimal(self):
        for source in GWP_VALUES:
            for gas, val in GWP_VALUES[source].items():
                assert isinstance(val, Decimal), (
                    f"{source}/{gas} is {type(val)}"
                )


class TestFuelEmissionFactors:
    """Tests for the FUEL_EMISSION_FACTORS constant table."""

    def test_has_14_fuel_types(self):
        assert len(FUEL_EMISSION_FACTORS) == 14

    def test_all_fuel_enum_values_covered(self):
        expected_keys = {e.value for e in FuelType}
        assert set(FUEL_EMISSION_FACTORS.keys()) == expected_keys

    def test_each_entry_has_five_fields(self):
        required = {"co2_ef", "ch4_ef", "n2o_ef", "default_efficiency", "is_biogenic"}
        for fuel, factors in FUEL_EMISSION_FACTORS.items():
            assert required.issubset(set(factors.keys())), (
                f"{fuel} missing fields"
            )

    def test_natural_gas_co2_ef(self):
        assert FUEL_EMISSION_FACTORS["natural_gas"]["co2_ef"] == Decimal("56.100")

    def test_biomass_wood_is_biogenic(self):
        assert FUEL_EMISSION_FACTORS["biomass_wood"]["is_biogenic"] == Decimal("1")

    def test_natural_gas_not_biogenic(self):
        assert FUEL_EMISSION_FACTORS["natural_gas"]["is_biogenic"] == Decimal("0")

    def test_waste_heat_zero_emissions(self):
        wh = FUEL_EMISSION_FACTORS["waste_heat"]
        assert wh["co2_ef"] == Decimal("0.000")
        assert wh["ch4_ef"] == Decimal("0.000")
        assert wh["n2o_ef"] == Decimal("0.000")

    def test_all_efficiencies_between_0_and_1(self):
        for fuel, factors in FUEL_EMISSION_FACTORS.items():
            eff = factors["default_efficiency"]
            assert Decimal("0") < eff <= Decimal("1"), (
                f"{fuel} efficiency {eff} out of range"
            )

    def test_all_values_are_decimal(self):
        for fuel, factors in FUEL_EMISSION_FACTORS.items():
            for key, val in factors.items():
                assert isinstance(val, Decimal), (
                    f"{fuel}/{key} is {type(val)}"
                )


class TestDistrictHeatingFactors:
    """Tests for the DISTRICT_HEATING_FACTORS constant table."""

    def test_has_13_regions(self):
        assert len(DISTRICT_HEATING_FACTORS) == 13

    def test_expected_regions_present(self):
        expected_regions = {
            "denmark",
            "sweden",
            "finland",
            "germany",
            "poland",
            "netherlands",
            "france",
            "uk",
            "us",
            "china",
            "japan",
            "south_korea",
            "global_default",
        }
        assert set(DISTRICT_HEATING_FACTORS.keys()) == expected_regions

    def test_each_entry_has_ef_and_loss(self):
        for region, factors in DISTRICT_HEATING_FACTORS.items():
            assert "ef_kgco2e_per_gj" in factors, f"{region} missing ef"
            assert "distribution_loss_pct" in factors, f"{region} missing loss"

    def test_germany_ef(self):
        assert (
            DISTRICT_HEATING_FACTORS["germany"]["ef_kgco2e_per_gj"]
            == Decimal("72.0")
        )

    def test_sweden_lowest_ef_among_nordics(self):
        se = DISTRICT_HEATING_FACTORS["sweden"]["ef_kgco2e_per_gj"]
        dk = DISTRICT_HEATING_FACTORS["denmark"]["ef_kgco2e_per_gj"]
        fi = DISTRICT_HEATING_FACTORS["finland"]["ef_kgco2e_per_gj"]
        assert se < dk
        assert se < fi

    def test_all_loss_pct_between_0_and_1(self):
        for region, factors in DISTRICT_HEATING_FACTORS.items():
            loss = factors["distribution_loss_pct"]
            assert Decimal("0") <= loss <= Decimal("1"), (
                f"{region} loss {loss} out of range"
            )

    def test_all_values_are_decimal(self):
        for region, factors in DISTRICT_HEATING_FACTORS.items():
            for key, val in factors.items():
                assert isinstance(val, Decimal), (
                    f"{region}/{key} is {type(val)}"
                )


class TestCoolingSystemFactors:
    """Tests for the COOLING_SYSTEM_FACTORS constant table."""

    def test_has_9_technologies(self):
        assert len(COOLING_SYSTEM_FACTORS) == 9

    def test_all_tech_enum_values_covered(self):
        expected_keys = {e.value for e in CoolingTechnology}
        assert set(COOLING_SYSTEM_FACTORS.keys()) == expected_keys

    def test_each_entry_has_four_fields(self):
        required = {"cop_min", "cop_max", "cop_default", "energy_source"}
        for tech, factors in COOLING_SYSTEM_FACTORS.items():
            assert required.issubset(set(factors.keys())), (
                f"{tech} missing fields"
            )

    def test_centrifugal_default_cop_is_6(self):
        assert (
            COOLING_SYSTEM_FACTORS["centrifugal_chiller"]["cop_default"]
            == Decimal("6.0")
        )

    def test_cop_min_le_cop_default_le_cop_max(self):
        for tech, factors in COOLING_SYSTEM_FACTORS.items():
            assert factors["cop_min"] <= factors["cop_default"] <= factors["cop_max"], (
                f"{tech} COP ordering violated"
            )

    def test_free_cooling_highest_cop(self):
        fc_max = COOLING_SYSTEM_FACTORS["free_cooling"]["cop_max"]
        for tech, factors in COOLING_SYSTEM_FACTORS.items():
            if tech != "free_cooling":
                assert fc_max >= factors["cop_max"], (
                    f"free_cooling cop_max not highest vs {tech}"
                )

    def test_cooling_energy_source_companion(self):
        for tech in COOLING_SYSTEM_FACTORS:
            assert tech in COOLING_ENERGY_SOURCE

    def test_absorption_uses_heat(self):
        for tech in ("absorption_single", "absorption_double", "absorption_triple"):
            assert COOLING_ENERGY_SOURCE[tech] == "heat"

    def test_electric_chillers_use_electricity(self):
        for tech in ("centrifugal_chiller", "screw_chiller", "reciprocating_chiller"):
            assert COOLING_ENERGY_SOURCE[tech] == "electricity"


class TestCHPDefaultEfficiencies:
    """Tests for the CHP_DEFAULT_EFFICIENCIES constant table."""

    def test_has_5_fuel_types(self):
        assert len(CHP_DEFAULT_EFFICIENCIES) == 5

    def test_expected_keys(self):
        expected = {"natural_gas", "coal", "biomass", "fuel_oil", "municipal_waste"}
        assert set(CHP_DEFAULT_EFFICIENCIES.keys()) == expected

    def test_each_entry_has_three_efficiencies(self):
        required = {
            "electrical_efficiency",
            "thermal_efficiency",
            "overall_efficiency",
        }
        for fuel, eff in CHP_DEFAULT_EFFICIENCIES.items():
            assert required.issubset(set(eff.keys())), (
                f"{fuel} missing efficiency fields"
            )

    def test_natural_gas_values(self):
        ng = CHP_DEFAULT_EFFICIENCIES["natural_gas"]
        assert ng["electrical_efficiency"] == Decimal("0.35")
        assert ng["thermal_efficiency"] == Decimal("0.45")
        assert ng["overall_efficiency"] == Decimal("0.80")

    def test_overall_ge_sum_of_components(self):
        for fuel, eff in CHP_DEFAULT_EFFICIENCIES.items():
            component_sum = (
                eff["electrical_efficiency"] + eff["thermal_efficiency"]
            )
            assert eff["overall_efficiency"] >= component_sum, (
                f"{fuel} overall < elec + thermal"
            )


class TestUnitConversions:
    """Tests for the UNIT_CONVERSIONS constant table."""

    def test_has_9_factors(self):
        assert len(UNIT_CONVERSIONS) == 9

    def test_expected_keys(self):
        expected = {
            "gj_to_mwh",
            "mwh_to_gj",
            "gj_to_kwh",
            "gj_to_mmbtu",
            "mmbtu_to_gj",
            "therm_to_gj",
            "gj_to_therm",
            "mj_to_gj",
            "gj_to_mj",
        }
        assert set(UNIT_CONVERSIONS.keys()) == expected

    def test_mwh_to_gj_is_3_6(self):
        assert UNIT_CONVERSIONS["mwh_to_gj"] == Decimal("3.6")

    def test_mj_to_gj_is_0_001(self):
        assert UNIT_CONVERSIONS["mj_to_gj"] == Decimal("0.001")

    def test_gj_to_mj_is_1000(self):
        assert UNIT_CONVERSIONS["gj_to_mj"] == Decimal("1000.0")

    def test_reciprocal_mwh_gj(self):
        product = UNIT_CONVERSIONS["gj_to_mwh"] * UNIT_CONVERSIONS["mwh_to_gj"]
        assert abs(product - Decimal("1")) < Decimal("0.001")

    def test_reciprocal_mmbtu_gj(self):
        product = (
            UNIT_CONVERSIONS["gj_to_mmbtu"] * UNIT_CONVERSIONS["mmbtu_to_gj"]
        )
        assert abs(product - Decimal("1")) < Decimal("0.001")

    def test_all_values_are_decimal(self):
        for key, val in UNIT_CONVERSIONS.items():
            assert isinstance(val, Decimal), f"{key} is {type(val)}"


class TestModuleConstants:
    """Tests for module-level constants."""

    def test_version(self):
        assert VERSION == "1.0.0"

    def test_max_calculations_per_batch(self):
        assert MAX_CALCULATIONS_PER_BATCH == 10_000

    def test_max_gases_per_result(self):
        assert MAX_GASES_PER_RESULT == 10

    def test_max_trace_steps(self):
        assert MAX_TRACE_STEPS == 200

    def test_max_facilities_per_tenant(self):
        assert MAX_FACILITIES_PER_TENANT == 50_000

    def test_default_monte_carlo_iterations(self):
        assert DEFAULT_MONTE_CARLO_ITERATIONS == 10_000

    def test_default_confidence_level(self):
        assert DEFAULT_CONFIDENCE_LEVEL == Decimal("0.95")

    def test_table_prefix(self):
        assert TABLE_PREFIX == "gl_shp_"


# ===================================================================
# Section 3: Pydantic model tests (20 models)
# ===================================================================


class TestFuelEmissionFactorModel:
    """Tests for the FuelEmissionFactor Pydantic model."""

    def test_creation(self):
        m = FuelEmissionFactor(
            fuel_type=FuelType.NATURAL_GAS,
            co2_ef_per_gj=Decimal("56.1"),
            ch4_ef_per_gj=Decimal("0.001"),
            n2o_ef_per_gj=Decimal("0.0001"),
            default_efficiency=Decimal("0.85"),
            is_biogenic=False,
        )
        assert m.fuel_type == FuelType.NATURAL_GAS
        assert m.co2_ef_per_gj == Decimal("56.1")

    def test_frozen_immutability(self):
        m = FuelEmissionFactor(
            fuel_type=FuelType.LPG,
            co2_ef_per_gj=Decimal("63.1"),
            ch4_ef_per_gj=Decimal("0.001"),
            n2o_ef_per_gj=Decimal("0.0001"),
            default_efficiency=Decimal("0.85"),
            is_biogenic=False,
        )
        with pytest.raises(Exception):
            m.co2_ef_per_gj = Decimal("99")

    def test_negative_co2_ef_raises(self):
        with pytest.raises(Exception):
            FuelEmissionFactor(
                fuel_type=FuelType.NATURAL_GAS,
                co2_ef_per_gj=Decimal("-1"),
                ch4_ef_per_gj=Decimal("0.001"),
                n2o_ef_per_gj=Decimal("0.0001"),
                default_efficiency=Decimal("0.85"),
                is_biogenic=False,
            )

    def test_zero_efficiency_raises(self):
        with pytest.raises(Exception):
            FuelEmissionFactor(
                fuel_type=FuelType.NATURAL_GAS,
                co2_ef_per_gj=Decimal("56.1"),
                ch4_ef_per_gj=Decimal("0.001"),
                n2o_ef_per_gj=Decimal("0.0001"),
                default_efficiency=Decimal("0"),
                is_biogenic=False,
            )

    def test_efficiency_above_1_raises(self):
        with pytest.raises(Exception):
            FuelEmissionFactor(
                fuel_type=FuelType.NATURAL_GAS,
                co2_ef_per_gj=Decimal("56.1"),
                ch4_ef_per_gj=Decimal("0.001"),
                n2o_ef_per_gj=Decimal("0.0001"),
                default_efficiency=Decimal("1.01"),
                is_biogenic=False,
            )


class TestDistrictHeatingFactorModel:
    """Tests for the DistrictHeatingFactor Pydantic model."""

    def test_creation(self):
        m = DistrictHeatingFactor(
            region="germany",
            ef_kgco2e_per_gj=Decimal("72.0"),
            distribution_loss_pct=Decimal("0.12"),
        )
        assert m.region == "germany"

    def test_region_lowercased(self):
        m = DistrictHeatingFactor(
            region="GERMANY",
            ef_kgco2e_per_gj=Decimal("72.0"),
            distribution_loss_pct=Decimal("0.12"),
        )
        assert m.region == "germany"

    def test_default_network_type(self):
        m = DistrictHeatingFactor(
            region="uk",
            ef_kgco2e_per_gj=Decimal("65.0"),
            distribution_loss_pct=Decimal("0.12"),
        )
        assert m.network_type == NetworkType.MUNICIPAL

    def test_empty_region_raises(self):
        with pytest.raises(Exception):
            DistrictHeatingFactor(
                region="",
                ef_kgco2e_per_gj=Decimal("72.0"),
                distribution_loss_pct=Decimal("0.12"),
            )

    def test_loss_pct_above_1_raises(self):
        with pytest.raises(Exception):
            DistrictHeatingFactor(
                region="test",
                ef_kgco2e_per_gj=Decimal("72.0"),
                distribution_loss_pct=Decimal("1.5"),
            )

    def test_frozen_immutability(self):
        m = DistrictHeatingFactor(
            region="finland",
            ef_kgco2e_per_gj=Decimal("55.0"),
            distribution_loss_pct=Decimal("0.09"),
        )
        with pytest.raises(Exception):
            m.region = "sweden"


class TestCoolingSystemFactorModel:
    """Tests for the CoolingSystemFactor Pydantic model."""

    def test_creation(self):
        m = CoolingSystemFactor(
            technology=CoolingTechnology.CENTRIFUGAL_CHILLER,
            cop_min=Decimal("5.0"),
            cop_max=Decimal("7.0"),
            cop_default=Decimal("6.0"),
            energy_source="electricity",
        )
        assert m.technology == CoolingTechnology.CENTRIFUGAL_CHILLER

    def test_cop_max_less_than_min_raises(self):
        with pytest.raises(Exception):
            CoolingSystemFactor(
                technology=CoolingTechnology.SCREW_CHILLER,
                cop_min=Decimal("5.0"),
                cop_max=Decimal("4.0"),
                cop_default=Decimal("4.5"),
                energy_source="electricity",
            )

    def test_cop_default_below_min_raises(self):
        with pytest.raises(Exception):
            CoolingSystemFactor(
                technology=CoolingTechnology.SCREW_CHILLER,
                cop_min=Decimal("4.0"),
                cop_max=Decimal("5.5"),
                cop_default=Decimal("3.0"),
                energy_source="electricity",
            )

    def test_cop_default_above_max_raises(self):
        with pytest.raises(Exception):
            CoolingSystemFactor(
                technology=CoolingTechnology.SCREW_CHILLER,
                cop_min=Decimal("4.0"),
                cop_max=Decimal("5.5"),
                cop_default=Decimal("6.0"),
                energy_source="electricity",
            )


class TestCHPParametersModel:
    """Tests for the CHPParameters Pydantic model."""

    def test_creation(self):
        m = CHPParameters(
            chp_id="chp-001",
            electrical_efficiency=Decimal("0.35"),
            thermal_efficiency=Decimal("0.45"),
            fuel_type=FuelType.NATURAL_GAS,
            power_output_mw=Decimal("10"),
            heat_output_mw=Decimal("15"),
            overall_efficiency=Decimal("0.80"),
        )
        assert m.chp_id == "chp-001"

    def test_overall_less_than_components_raises(self):
        with pytest.raises(Exception):
            CHPParameters(
                chp_id="chp-bad",
                electrical_efficiency=Decimal("0.35"),
                thermal_efficiency=Decimal("0.45"),
                fuel_type=FuelType.NATURAL_GAS,
                power_output_mw=Decimal("10"),
                heat_output_mw=Decimal("15"),
                overall_efficiency=Decimal("0.50"),
            )

    def test_empty_chp_id_raises(self):
        with pytest.raises(Exception):
            CHPParameters(
                chp_id="",
                electrical_efficiency=Decimal("0.35"),
                thermal_efficiency=Decimal("0.45"),
                fuel_type=FuelType.NATURAL_GAS,
                power_output_mw=Decimal("10"),
                heat_output_mw=Decimal("15"),
                overall_efficiency=Decimal("0.80"),
            )

    def test_frozen_immutability(self):
        m = CHPParameters(
            chp_id="chp-frozen",
            electrical_efficiency=Decimal("0.35"),
            thermal_efficiency=Decimal("0.45"),
            fuel_type=FuelType.NATURAL_GAS,
            power_output_mw=Decimal("10"),
            heat_output_mw=Decimal("15"),
            overall_efficiency=Decimal("0.80"),
        )
        with pytest.raises(Exception):
            m.chp_id = "chp-changed"


class TestFacilityInfoModel:
    """Tests for the FacilityInfo Pydantic model."""

    def test_creation(self):
        m = FacilityInfo(
            name="Test Facility",
            facility_type=FacilityType.INDUSTRIAL,
            country="DE",
            region="germany",
        )
        assert m.name == "Test Facility"
        assert m.facility_id is not None

    def test_country_uppercased(self):
        m = FacilityInfo(
            name="Test",
            facility_type=FacilityType.COMMERCIAL,
            country="de",
            region="germany",
        )
        assert m.country == "DE"

    def test_region_lowercased(self):
        m = FacilityInfo(
            name="Test",
            facility_type=FacilityType.COMMERCIAL,
            country="DE",
            region="GERMANY",
        )
        assert m.region == "germany"

    def test_default_tenant_id(self):
        m = FacilityInfo(
            name="Test",
            facility_type=FacilityType.INDUSTRIAL,
            country="US",
            region="us",
        )
        assert m.tenant_id == "default"

    def test_too_many_suppliers_raises(self):
        with pytest.raises(Exception):
            FacilityInfo(
                name="Test",
                facility_type=FacilityType.INDUSTRIAL,
                country="US",
                region="us",
                steam_suppliers=["s" + str(i) for i in range(101)],
            )

    def test_created_at_is_utc(self):
        m = FacilityInfo(
            name="Test",
            facility_type=FacilityType.INDUSTRIAL,
            country="US",
            region="us",
        )
        assert m.created_at.tzinfo is not None


class TestSteamSupplierModel:
    """Tests for the SteamSupplier Pydantic model."""

    def test_creation(self):
        m = SteamSupplier(name="Supplier A", country="US")
        assert m.name == "Supplier A"
        assert m.country == "US"
        assert m.verified is False

    def test_country_uppercased(self):
        m = SteamSupplier(name="Supplier B", country="de")
        assert m.country == "DE"

    def test_negative_fuel_mix_fraction_raises(self):
        with pytest.raises(Exception):
            SteamSupplier(
                name="Bad",
                country="US",
                fuel_mix={"natural_gas": Decimal("-0.5")},
            )

    def test_default_data_quality_tier(self):
        m = SteamSupplier(name="Test", country="US")
        assert m.data_quality_tier == DataQualityTier.TIER_1


class TestSteamCalculationRequestModel:
    """Tests for the SteamCalculationRequest Pydantic model."""

    def test_creation(self):
        m = SteamCalculationRequest(
            facility_id="fac-001",
            consumption_gj=Decimal("1000"),
        )
        assert m.consumption_gj == Decimal("1000")

    def test_default_energy_type_is_steam(self):
        m = SteamCalculationRequest(
            facility_id="fac-001",
            consumption_gj=Decimal("500"),
        )
        assert m.energy_type == EnergyType.STEAM

    def test_default_gwp_source_is_ar6(self):
        m = SteamCalculationRequest(
            facility_id="fac-001",
            consumption_gj=Decimal("500"),
        )
        assert m.gwp_source == GWPSource.AR6

    def test_zero_consumption_raises(self):
        with pytest.raises(Exception):
            SteamCalculationRequest(
                facility_id="fac-001",
                consumption_gj=Decimal("0"),
            )

    def test_negative_consumption_raises(self):
        with pytest.raises(Exception):
            SteamCalculationRequest(
                facility_id="fac-001",
                consumption_gj=Decimal("-100"),
            )

    def test_condensate_return_default_zero(self):
        m = SteamCalculationRequest(
            facility_id="fac-001",
            consumption_gj=Decimal("100"),
        )
        assert m.condensate_return_pct == Decimal("0")

    def test_frozen_immutability(self):
        m = SteamCalculationRequest(
            facility_id="fac-001",
            consumption_gj=Decimal("1000"),
        )
        with pytest.raises(Exception):
            m.consumption_gj = Decimal("2000")


class TestHeatingCalculationRequestModel:
    """Tests for the HeatingCalculationRequest Pydantic model."""

    def test_creation(self):
        m = HeatingCalculationRequest(
            facility_id="fac-002",
            consumption_gj=Decimal("500"),
            region="germany",
        )
        assert m.consumption_gj == Decimal("500")

    def test_region_lowercased(self):
        m = HeatingCalculationRequest(
            facility_id="fac-002",
            consumption_gj=Decimal("500"),
            region="GERMANY",
        )
        assert m.region == "germany"

    def test_default_network_type(self):
        m = HeatingCalculationRequest(
            facility_id="fac-002",
            consumption_gj=Decimal("500"),
            region="sweden",
        )
        assert m.network_type == NetworkType.MUNICIPAL


class TestCoolingCalculationRequestModel:
    """Tests for the CoolingCalculationRequest Pydantic model."""

    def test_creation(self):
        m = CoolingCalculationRequest(
            facility_id="fac-003",
            cooling_output_gj=Decimal("300"),
            technology=CoolingTechnology.CENTRIFUGAL_CHILLER,
        )
        assert m.cooling_output_gj == Decimal("300")

    def test_zero_cooling_output_raises(self):
        with pytest.raises(Exception):
            CoolingCalculationRequest(
                facility_id="fac-003",
                cooling_output_gj=Decimal("0"),
                technology=CoolingTechnology.SCREW_CHILLER,
            )


class TestCHPAllocationRequestModel:
    """Tests for the CHPAllocationRequest Pydantic model."""

    def test_creation(self):
        m = CHPAllocationRequest(
            facility_id="fac-004",
            total_fuel_gj=Decimal("2000"),
            fuel_type=FuelType.NATURAL_GAS,
            heat_output_gj=Decimal("900"),
            power_output_gj=Decimal("700"),
        )
        assert m.total_fuel_gj == Decimal("2000")

    def test_default_method_is_efficiency(self):
        m = CHPAllocationRequest(
            facility_id="fac-004",
            total_fuel_gj=Decimal("2000"),
            fuel_type=FuelType.NATURAL_GAS,
            heat_output_gj=Decimal("900"),
            power_output_gj=Decimal("700"),
        )
        assert m.method == CHPAllocMethod.EFFICIENCY

    def test_steam_temp_below_ambient_raises(self):
        with pytest.raises(Exception):
            CHPAllocationRequest(
                facility_id="fac-004",
                total_fuel_gj=Decimal("2000"),
                fuel_type=FuelType.NATURAL_GAS,
                heat_output_gj=Decimal("900"),
                power_output_gj=Decimal("700"),
                steam_temperature_c=Decimal("20"),
                ambient_temperature_c=Decimal("25"),
            )

    def test_default_ambient_temperature(self):
        m = CHPAllocationRequest(
            facility_id="fac-004",
            total_fuel_gj=Decimal("2000"),
            fuel_type=FuelType.NATURAL_GAS,
            heat_output_gj=Decimal("900"),
            power_output_gj=Decimal("700"),
        )
        assert m.ambient_temperature_c == Decimal("25")


class TestGasEmissionDetailModel:
    """Tests for the GasEmissionDetail Pydantic model."""

    def test_creation(self):
        m = GasEmissionDetail(
            gas=EmissionGas.CO2,
            emission_kg=Decimal("56100"),
            gwp_value=Decimal("1"),
            gwp_source=GWPSource.AR6,
            co2e_kg=Decimal("56100"),
        )
        assert m.gas == EmissionGas.CO2

    def test_negative_emission_raises(self):
        with pytest.raises(Exception):
            GasEmissionDetail(
                gas=EmissionGas.CO2,
                emission_kg=Decimal("-1"),
                gwp_value=Decimal("1"),
                gwp_source=GWPSource.AR6,
                co2e_kg=Decimal("-1"),
            )


class TestCalculationResultModel:
    """Tests for the CalculationResult Pydantic model."""

    def _make_result(self, **kwargs) -> CalculationResult:
        defaults = {
            "energy_type": EnergyType.STEAM,
            "calculation_method": CalculationMethod.FUEL_BASED,
            "total_co2e_kg": Decimal("1000"),
            "fossil_co2e_kg": Decimal("1000"),
            "consumption_gj": Decimal("100"),
            "effective_ef_kgco2e_per_gj": Decimal("10"),
            "data_quality_tier": DataQualityTier.TIER_1,
        }
        defaults.update(kwargs)
        return CalculationResult(**defaults)

    def test_creation(self):
        m = self._make_result()
        assert m.status == "SUCCESS"
        assert m.calc_id is not None

    def test_default_biogenic_zero(self):
        m = self._make_result()
        assert m.biogenic_co2_kg == Decimal("0")

    def test_too_many_gas_details_raises(self):
        details = [
            GasEmissionDetail(
                gas=EmissionGas.CO2,
                emission_kg=Decimal("1"),
                gwp_value=Decimal("1"),
                gwp_source=GWPSource.AR6,
                co2e_kg=Decimal("1"),
            )
            for _ in range(MAX_GASES_PER_RESULT + 1)
        ]
        with pytest.raises(Exception):
            self._make_result(gas_details=details)

    def test_too_many_trace_steps_raises(self):
        steps = ["step"] * (MAX_TRACE_STEPS + 1)
        with pytest.raises(Exception):
            self._make_result(trace=steps)


class TestCHPAllocationResultModel:
    """Tests for the CHPAllocationResult Pydantic model."""

    def test_creation(self):
        m = CHPAllocationResult(
            method=CHPAllocMethod.EFFICIENCY,
            heat_share=Decimal("0.5625"),
            power_share=Decimal("0.4375"),
            heat_emissions_kgco2e=Decimal("5000"),
            power_emissions_kgco2e=Decimal("3888"),
            total_fuel_emissions_kgco2e=Decimal("8888"),
        )
        assert m.method == CHPAllocMethod.EFFICIENCY

    def test_default_cooling_share_zero(self):
        m = CHPAllocationResult(
            method=CHPAllocMethod.ENERGY,
            heat_share=Decimal("0.5"),
            power_share=Decimal("0.5"),
            heat_emissions_kgco2e=Decimal("5000"),
            power_emissions_kgco2e=Decimal("5000"),
            total_fuel_emissions_kgco2e=Decimal("10000"),
        )
        assert m.cooling_share == Decimal("0")
        assert m.cooling_emissions_kgco2e == Decimal("0")


class TestBatchCalculationRequestModel:
    """Tests for the BatchCalculationRequest Pydantic model."""

    def test_creation(self):
        m = BatchCalculationRequest(
            requests=["req1"],
        )
        assert len(m.requests) == 1

    def test_empty_requests_raises(self):
        with pytest.raises(Exception):
            BatchCalculationRequest(requests=[])

    def test_exceeding_max_batch_raises(self):
        with pytest.raises(Exception):
            BatchCalculationRequest(
                requests=list(range(MAX_CALCULATIONS_PER_BATCH + 1)),
            )


class TestBatchCalculationResultModel:
    """Tests for the BatchCalculationResult Pydantic model."""

    def test_creation(self):
        m = BatchCalculationResult(
            total_co2e_kg=Decimal("10000"),
            total_fossil_co2e_kg=Decimal("9000"),
            total_biogenic_co2_kg=Decimal("1000"),
            success_count=10,
            failure_count=0,
            status=BatchStatus.COMPLETED,
        )
        assert m.batch_id is not None
        assert m.success_count == 10


class TestUncertaintyRequestModel:
    """Tests for the UncertaintyRequest Pydantic model."""

    def _make_calc_result(self) -> CalculationResult:
        return CalculationResult(
            energy_type=EnergyType.STEAM,
            calculation_method=CalculationMethod.FUEL_BASED,
            total_co2e_kg=Decimal("1000"),
            fossil_co2e_kg=Decimal("1000"),
            consumption_gj=Decimal("100"),
            effective_ef_kgco2e_per_gj=Decimal("10"),
            data_quality_tier=DataQualityTier.TIER_1,
        )

    def test_creation(self):
        cr = self._make_calc_result()
        m = UncertaintyRequest(calc_result=cr)
        assert m.method == "monte_carlo"
        assert m.iterations == DEFAULT_MONTE_CARLO_ITERATIONS

    def test_invalid_method_raises(self):
        cr = self._make_calc_result()
        with pytest.raises(Exception):
            UncertaintyRequest(calc_result=cr, method="bayesian")

    def test_iterations_below_100_raises(self):
        cr = self._make_calc_result()
        with pytest.raises(Exception):
            UncertaintyRequest(calc_result=cr, iterations=50)


class TestUncertaintyResultModel:
    """Tests for the UncertaintyResult Pydantic model."""

    def test_creation(self):
        m = UncertaintyResult(
            mean_co2e_kg=Decimal("1000"),
            std_dev_kg=Decimal("50"),
            ci_lower_kg=Decimal("900"),
            ci_upper_kg=Decimal("1100"),
            confidence_level=Decimal("0.95"),
            method="monte_carlo",
            relative_uncertainty_pct=Decimal("10"),
        )
        assert m.mean_co2e_kg == Decimal("1000")


class TestComplianceCheckResultModel:
    """Tests for the ComplianceCheckResult Pydantic model."""

    def test_creation(self):
        m = ComplianceCheckResult(
            framework="GHG_PROTOCOL",
            status=ComplianceStatus.COMPLIANT,
            total_requirements=10,
            met_requirements=10,
            score_pct=Decimal("100"),
        )
        assert m.framework == "GHG_PROTOCOL"

    def test_met_exceeds_total_raises(self):
        with pytest.raises(Exception):
            ComplianceCheckResult(
                framework="GHG_PROTOCOL",
                status=ComplianceStatus.COMPLIANT,
                total_requirements=5,
                met_requirements=10,
                score_pct=Decimal("100"),
            )

    def test_too_many_findings_raises(self):
        with pytest.raises(Exception):
            ComplianceCheckResult(
                framework="GHG_PROTOCOL",
                status=ComplianceStatus.NON_COMPLIANT,
                total_requirements=10,
                met_requirements=5,
                score_pct=Decimal("50"),
                findings=["finding"] * 501,
            )


class TestAggregationRequestModel:
    """Tests for the AggregationRequest Pydantic model."""

    def test_creation(self):
        m = AggregationRequest(
            calc_ids=["id-1", "id-2"],
            aggregation_type=AggregationType.BY_FACILITY,
        )
        assert len(m.calc_ids) == 2

    def test_empty_calc_ids_raises(self):
        with pytest.raises(Exception):
            AggregationRequest(
                calc_ids=[],
                aggregation_type=AggregationType.BY_FACILITY,
            )


class TestAggregationResultModel:
    """Tests for the AggregationResult Pydantic model."""

    def test_creation(self):
        m = AggregationResult(
            aggregation_type=AggregationType.BY_FUEL,
            total_co2e_kg=Decimal("50000"),
            total_fossil_co2e_kg=Decimal("45000"),
            total_biogenic_co2_kg=Decimal("5000"),
            count=10,
        )
        assert m.aggregation_id is not None
        assert m.count == 10

    def test_default_provenance_hash_empty(self):
        m = AggregationResult(
            aggregation_type=AggregationType.BY_PERIOD,
            total_co2e_kg=Decimal("1000"),
            total_fossil_co2e_kg=Decimal("1000"),
            total_biogenic_co2_kg=Decimal("0"),
            count=1,
        )
        assert m.provenance_hash == ""
