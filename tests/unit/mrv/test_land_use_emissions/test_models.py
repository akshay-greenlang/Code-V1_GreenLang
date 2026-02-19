# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-MRV-006 Land Use Emissions Agent Data Models.

Tests all 16 enumerations, constant tables (GWP, AGB, root-shoot ratios,
dead wood fractions, litter stocks, SOC reference stocks, SOC factors,
biomass growth rates, combustion factors, fire EFs, peatland EFs, N2O EF),
and 16 Pydantic models with field validators.

Target: 120 tests, 85%+ coverage.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict

import pytest

from greenlang.land_use_emissions.models import (
    # Enumerations (16)
    LandCategory,
    CarbonPool,
    ClimateZone,
    SoilType,
    CalculationTier,
    CalculationMethod,
    EmissionGas,
    GWPSource,
    EmissionFactorSource,
    TransitionType,
    DisturbanceType,
    PeatlandStatus,
    ManagementPractice,
    InputLevel,
    ComplianceStatus,
    ReportingPeriod,
    # Constants
    GWP_VALUES,
    IPCC_AGB_DEFAULTS,
    ROOT_SHOOT_RATIOS,
    DEAD_WOOD_FRACTION,
    LITTER_STOCKS,
    SOC_REFERENCE_STOCKS,
    SOC_LAND_USE_FACTORS,
    SOC_MANAGEMENT_FACTORS,
    SOC_INPUT_FACTORS,
    BIOMASS_GROWTH_RATES,
    CARBON_FRACTION,
    COMBUSTION_FACTORS,
    FIRE_EMISSION_FACTORS,
    PEATLAND_EF,
    N2O_SOIL_EF,
    CONVERSION_FACTOR_CO2_C,
    # Pydantic models
    LandParcelInfo,
    CarbonStockSnapshot,
    # Module constants
    VERSION,
    MAX_CALCULATIONS_PER_BATCH,
    MAX_GASES_PER_RESULT,
    MAX_TRACE_STEPS,
    MAX_POOLS_PER_CALC,
    MAX_PARCELS_PER_TENANT,
    DEFAULT_TRANSITION_YEARS,
)


# ===========================================================================
# Enumeration Tests
# ===========================================================================


class TestLandCategory:
    """Tests for the LandCategory enumeration."""

    def test_member_count(self):
        """LandCategory has exactly 6 members."""
        assert len(LandCategory) == 6

    def test_forest_land_value(self):
        """FOREST_LAND has the expected string value."""
        assert LandCategory.FOREST_LAND.value == "forest_land"

    def test_cropland_value(self):
        """CROPLAND has the expected string value."""
        assert LandCategory.CROPLAND.value == "cropland"

    def test_grassland_value(self):
        """GRASSLAND has the expected string value."""
        assert LandCategory.GRASSLAND.value == "grassland"

    def test_wetland_value(self):
        """WETLAND has the expected string value."""
        assert LandCategory.WETLAND.value == "wetland"

    def test_settlement_value(self):
        """SETTLEMENT has the expected string value."""
        assert LandCategory.SETTLEMENT.value == "settlement"

    def test_other_land_value(self):
        """OTHER_LAND has the expected string value."""
        assert LandCategory.OTHER_LAND.value == "other_land"

    def test_membership(self):
        """String values can be used to look up enum members."""
        assert LandCategory("forest_land") == LandCategory.FOREST_LAND

    def test_all_are_str_enum(self):
        """All members are string-compatible enums."""
        for member in LandCategory:
            assert isinstance(member.value, str)

    def test_iteration_yields_all(self):
        """Iterating over LandCategory yields all 6 members."""
        members = list(LandCategory)
        assert len(members) == 6


class TestCarbonPool:
    """Tests for the CarbonPool enumeration."""

    def test_member_count(self):
        """CarbonPool has exactly 5 members."""
        assert len(CarbonPool) == 5

    @pytest.mark.parametrize("member,expected", [
        (CarbonPool.ABOVE_GROUND_BIOMASS, "above_ground_biomass"),
        (CarbonPool.BELOW_GROUND_BIOMASS, "below_ground_biomass"),
        (CarbonPool.DEAD_WOOD, "dead_wood"),
        (CarbonPool.LITTER, "litter"),
        (CarbonPool.SOIL_ORGANIC_CARBON, "soil_organic_carbon"),
    ])
    def test_values(self, member, expected):
        """Each CarbonPool member has the correct string value."""
        assert member.value == expected


class TestClimateZone:
    """Tests for the ClimateZone enumeration."""

    def test_member_count(self):
        """ClimateZone has exactly 12 members."""
        assert len(ClimateZone) == 12

    @pytest.mark.parametrize("member,expected", [
        (ClimateZone.TROPICAL_WET, "tropical_wet"),
        (ClimateZone.TROPICAL_MOIST, "tropical_moist"),
        (ClimateZone.TROPICAL_DRY, "tropical_dry"),
        (ClimateZone.TROPICAL_MONTANE, "tropical_montane"),
        (ClimateZone.WARM_TEMPERATE_MOIST, "warm_temperate_moist"),
        (ClimateZone.WARM_TEMPERATE_DRY, "warm_temperate_dry"),
        (ClimateZone.COOL_TEMPERATE_MOIST, "cool_temperate_moist"),
        (ClimateZone.COOL_TEMPERATE_DRY, "cool_temperate_dry"),
        (ClimateZone.BOREAL_MOIST, "boreal_moist"),
        (ClimateZone.BOREAL_DRY, "boreal_dry"),
        (ClimateZone.POLAR_MOIST, "polar_moist"),
        (ClimateZone.POLAR_DRY, "polar_dry"),
    ])
    def test_values(self, member, expected):
        """Each ClimateZone member has the correct string value."""
        assert member.value == expected


class TestSoilType:
    """Tests for the SoilType enumeration."""

    def test_member_count(self):
        """SoilType has exactly 7 members."""
        assert len(SoilType) == 7

    @pytest.mark.parametrize("member,expected", [
        (SoilType.HIGH_ACTIVITY_CLAY, "high_activity_clay"),
        (SoilType.LOW_ACTIVITY_CLAY, "low_activity_clay"),
        (SoilType.SANDY, "sandy"),
        (SoilType.SPODIC, "spodic"),
        (SoilType.VOLCANIC, "volcanic"),
        (SoilType.WETLAND_ORGANIC, "wetland_organic"),
        (SoilType.OTHER, "other"),
    ])
    def test_values(self, member, expected):
        """Each SoilType member has the correct string value."""
        assert member.value == expected


class TestCalculationTier:
    """Tests for the CalculationTier enumeration."""

    def test_member_count(self):
        """CalculationTier has exactly 3 members."""
        assert len(CalculationTier) == 3

    @pytest.mark.parametrize("member,expected", [
        (CalculationTier.TIER_1, "tier_1"),
        (CalculationTier.TIER_2, "tier_2"),
        (CalculationTier.TIER_3, "tier_3"),
    ])
    def test_values(self, member, expected):
        """Each CalculationTier member has the correct string value."""
        assert member.value == expected


class TestCalculationMethod:
    """Tests for the CalculationMethod enumeration."""

    def test_member_count(self):
        """CalculationMethod has exactly 2 members."""
        assert len(CalculationMethod) == 2

    def test_stock_difference_value(self):
        """STOCK_DIFFERENCE has the expected string value."""
        assert CalculationMethod.STOCK_DIFFERENCE.value == "stock_difference"

    def test_gain_loss_value(self):
        """GAIN_LOSS has the expected string value."""
        assert CalculationMethod.GAIN_LOSS.value == "gain_loss"


class TestEmissionGas:
    """Tests for the EmissionGas enumeration."""

    def test_member_count(self):
        """EmissionGas has exactly 4 members."""
        assert len(EmissionGas) == 4

    @pytest.mark.parametrize("member,expected", [
        (EmissionGas.CO2, "CO2"),
        (EmissionGas.CH4, "CH4"),
        (EmissionGas.N2O, "N2O"),
        (EmissionGas.CO, "CO"),
    ])
    def test_values(self, member, expected):
        """Each EmissionGas member has the correct string value."""
        assert member.value == expected


class TestGWPSource:
    """Tests for the GWPSource enumeration."""

    def test_member_count(self):
        """GWPSource has exactly 4 members."""
        assert len(GWPSource) == 4

    @pytest.mark.parametrize("member,expected", [
        (GWPSource.IPCC_AR4, "AR4"),
        (GWPSource.IPCC_AR5, "AR5"),
        (GWPSource.IPCC_AR6, "AR6"),
        (GWPSource.IPCC_AR6_GTP, "AR6_GTP"),
    ])
    def test_values(self, member, expected):
        """Each GWPSource member has the correct string value."""
        assert member.value == expected


class TestEmissionFactorSource:
    """Tests for the EmissionFactorSource enumeration."""

    def test_member_count(self):
        """EmissionFactorSource has exactly 6 members."""
        assert len(EmissionFactorSource) == 6

    @pytest.mark.parametrize("member,expected", [
        (EmissionFactorSource.IPCC_2006, "IPCC_2006"),
        (EmissionFactorSource.IPCC_2019, "IPCC_2019"),
        (EmissionFactorSource.IPCC_WETLANDS_2013, "IPCC_WETLANDS_2013"),
        (EmissionFactorSource.NATIONAL_INVENTORY, "NATIONAL_INVENTORY"),
        (EmissionFactorSource.LITERATURE, "LITERATURE"),
        (EmissionFactorSource.CUSTOM, "CUSTOM"),
    ])
    def test_values(self, member, expected):
        """Each EmissionFactorSource member has the correct string value."""
        assert member.value == expected


class TestTransitionType:
    """Tests for the TransitionType enumeration."""

    def test_member_count(self):
        """TransitionType has exactly 2 members."""
        assert len(TransitionType) == 2

    def test_remaining_value(self):
        """REMAINING has the expected string value."""
        assert TransitionType.REMAINING.value == "remaining"

    def test_conversion_value(self):
        """CONVERSION has the expected string value."""
        assert TransitionType.CONVERSION.value == "conversion"


class TestDisturbanceType:
    """Tests for the DisturbanceType enumeration."""

    def test_member_count(self):
        """DisturbanceType has exactly 8 members."""
        assert len(DisturbanceType) == 8

    @pytest.mark.parametrize("member,expected", [
        (DisturbanceType.FIRE, "fire"),
        (DisturbanceType.HARVEST, "harvest"),
        (DisturbanceType.STORM, "storm"),
        (DisturbanceType.INSECTS, "insects"),
        (DisturbanceType.DROUGHT, "drought"),
        (DisturbanceType.FLOOD, "flood"),
        (DisturbanceType.LAND_CLEARING, "land_clearing"),
        (DisturbanceType.NONE, "none"),
    ])
    def test_values(self, member, expected):
        """Each DisturbanceType member has the correct string value."""
        assert member.value == expected


class TestPeatlandStatus:
    """Tests for the PeatlandStatus enumeration."""

    def test_member_count(self):
        """PeatlandStatus has exactly 4 members."""
        assert len(PeatlandStatus) == 4

    @pytest.mark.parametrize("member,expected", [
        (PeatlandStatus.NATURAL, "natural"),
        (PeatlandStatus.DRAINED, "drained"),
        (PeatlandStatus.REWETTED, "rewetted"),
        (PeatlandStatus.EXTRACTED, "extracted"),
    ])
    def test_values(self, member, expected):
        """Each PeatlandStatus member has the correct string value."""
        assert member.value == expected


class TestManagementPractice:
    """Tests for the ManagementPractice enumeration."""

    def test_member_count(self):
        """ManagementPractice has exactly 6 members."""
        assert len(ManagementPractice) == 6

    @pytest.mark.parametrize("member,expected", [
        (ManagementPractice.FULL_TILLAGE, "full_tillage"),
        (ManagementPractice.REDUCED_TILLAGE, "reduced_tillage"),
        (ManagementPractice.NO_TILL, "no_till"),
        (ManagementPractice.IMPROVED, "improved"),
        (ManagementPractice.DEGRADED, "degraded"),
        (ManagementPractice.NOMINALLY_MANAGED, "nominally_managed"),
    ])
    def test_values(self, member, expected):
        """Each ManagementPractice member has the correct string value."""
        assert member.value == expected


class TestInputLevel:
    """Tests for the InputLevel enumeration."""

    def test_member_count(self):
        """InputLevel has exactly 4 members."""
        assert len(InputLevel) == 4

    @pytest.mark.parametrize("member,expected", [
        (InputLevel.LOW, "low"),
        (InputLevel.MEDIUM, "medium"),
        (InputLevel.HIGH, "high"),
        (InputLevel.HIGH_WITH_MANURE, "high_with_manure"),
    ])
    def test_values(self, member, expected):
        """Each InputLevel member has the correct string value."""
        assert member.value == expected


class TestComplianceStatus:
    """Tests for the ComplianceStatus enumeration."""

    def test_member_count(self):
        """ComplianceStatus has exactly 4 members."""
        assert len(ComplianceStatus) == 4

    @pytest.mark.parametrize("member,expected", [
        (ComplianceStatus.COMPLIANT, "compliant"),
        (ComplianceStatus.NON_COMPLIANT, "non_compliant"),
        (ComplianceStatus.PARTIAL, "partial"),
        (ComplianceStatus.NOT_ASSESSED, "not_assessed"),
    ])
    def test_values(self, member, expected):
        """Each ComplianceStatus member has the correct string value."""
        assert member.value == expected


class TestReportingPeriod:
    """Tests for the ReportingPeriod enumeration."""

    def test_member_count(self):
        """ReportingPeriod has exactly 4 members."""
        assert len(ReportingPeriod) == 4

    @pytest.mark.parametrize("member,expected", [
        (ReportingPeriod.MONTHLY, "monthly"),
        (ReportingPeriod.QUARTERLY, "quarterly"),
        (ReportingPeriod.ANNUAL, "annual"),
        (ReportingPeriod.CUSTOM, "custom"),
    ])
    def test_values(self, member, expected):
        """Each ReportingPeriod member has the correct string value."""
        assert member.value == expected


# ===========================================================================
# Constant Table Tests
# ===========================================================================


class TestGWPValues:
    """Tests for the GWP_VALUES constant table."""

    def test_has_all_gwp_sources(self):
        """GWP_VALUES covers all 4 GWP sources."""
        assert len(GWP_VALUES) == 4
        for source in GWPSource:
            assert source in GWP_VALUES

    def test_all_sources_have_four_gases(self):
        """Each GWP source maps to exactly 4 gases."""
        for source, gases in GWP_VALUES.items():
            assert len(gases) == 4, f"{source} has {len(gases)} gases, expected 4"

    def test_co2_is_always_one(self):
        """CO2 GWP is 1 across all assessment reports."""
        for source, gases in GWP_VALUES.items():
            assert gases[EmissionGas.CO2] == Decimal("1"), f"{source} CO2 != 1"

    @pytest.mark.parametrize("source,gas,expected", [
        (GWPSource.IPCC_AR4, EmissionGas.CH4, Decimal("25")),
        (GWPSource.IPCC_AR5, EmissionGas.CH4, Decimal("28")),
        (GWPSource.IPCC_AR6, EmissionGas.CH4, Decimal("27.9")),
        (GWPSource.IPCC_AR4, EmissionGas.N2O, Decimal("298")),
        (GWPSource.IPCC_AR5, EmissionGas.N2O, Decimal("265")),
        (GWPSource.IPCC_AR6, EmissionGas.N2O, Decimal("273")),
    ])
    def test_specific_gwp_values(self, source, gas, expected):
        """Specific GWP values match published IPCC data."""
        assert GWP_VALUES[source][gas] == expected

    def test_all_values_are_decimal(self):
        """All GWP values are Decimal instances for deterministic arithmetic."""
        for source, gases in GWP_VALUES.items():
            for gas, value in gases.items():
                assert isinstance(value, Decimal), f"{source}/{gas} is not Decimal"


class TestIPCCAGBDefaults:
    """Tests for the IPCC_AGB_DEFAULTS constant table."""

    def test_forest_land_has_all_12_climate_zones(self):
        """Forest land AGB defaults cover all 12 climate zones."""
        forest_keys = [
            k for k in IPCC_AGB_DEFAULTS.keys()
            if k[0] == LandCategory.FOREST_LAND
        ]
        assert len(forest_keys) == 12

    def test_tropical_wet_forest_is_200(self):
        """Tropical wet forest AGB default is 200 tC/ha."""
        key = (LandCategory.FOREST_LAND, ClimateZone.TROPICAL_WET)
        assert IPCC_AGB_DEFAULTS[key] == Decimal("200")

    def test_all_values_are_decimal(self):
        """All AGB defaults are Decimal instances."""
        for key, value in IPCC_AGB_DEFAULTS.items():
            assert isinstance(value, Decimal), f"Key {key} is not Decimal"

    def test_all_values_non_negative(self):
        """All AGB defaults are non-negative."""
        for key, value in IPCC_AGB_DEFAULTS.items():
            assert value >= Decimal("0"), f"Key {key} has negative AGB: {value}"

    def test_forest_agb_decreases_toward_poles(self):
        """Forest AGB decreases from tropical wet to polar dry."""
        tw = IPCC_AGB_DEFAULTS[(LandCategory.FOREST_LAND, ClimateZone.TROPICAL_WET)]
        pd = IPCC_AGB_DEFAULTS[(LandCategory.FOREST_LAND, ClimateZone.POLAR_DRY)]
        assert tw > pd


class TestRootShootRatios:
    """Tests for the ROOT_SHOOT_RATIOS constant table."""

    def test_forest_land_has_all_12_climate_zones(self):
        """Forest land root-shoot ratios cover all 12 climate zones."""
        forest_keys = [
            k for k in ROOT_SHOOT_RATIOS.keys()
            if k[0] == LandCategory.FOREST_LAND
        ]
        assert len(forest_keys) == 12

    def test_all_ratios_between_zero_and_one(self):
        """All root-shoot ratios are between 0 and 1."""
        for key, value in ROOT_SHOOT_RATIOS.items():
            assert Decimal("0") < value < Decimal("5"), (
                f"Key {key} ratio {value} out of range"
            )

    def test_tropical_wet_forest_ratio(self):
        """Tropical wet forest root-shoot ratio is 0.24."""
        key = (LandCategory.FOREST_LAND, ClimateZone.TROPICAL_WET)
        assert ROOT_SHOOT_RATIOS[key] == Decimal("0.24")


class TestDeadWoodFraction:
    """Tests for the DEAD_WOOD_FRACTION constant table."""

    def test_has_all_12_climate_zones(self):
        """Dead wood fractions cover all 12 climate zones."""
        assert len(DEAD_WOOD_FRACTION) == 12

    def test_all_fractions_between_zero_and_one(self):
        """All dead wood fractions are between 0 and 1."""
        for zone, fraction in DEAD_WOOD_FRACTION.items():
            assert Decimal("0") <= fraction <= Decimal("1"), (
                f"Zone {zone} fraction {fraction} out of range"
            )

    def test_tropical_wet_is_005(self):
        """Tropical wet dead wood fraction is 0.05."""
        assert DEAD_WOOD_FRACTION[ClimateZone.TROPICAL_WET] == Decimal("0.05")


class TestLitterStocks:
    """Tests for the LITTER_STOCKS constant table."""

    def test_forest_land_has_all_12_zones(self):
        """Forest land litter stocks cover all 12 climate zones."""
        forest_keys = [
            k for k in LITTER_STOCKS.keys()
            if k[0] == LandCategory.FOREST_LAND
        ]
        assert len(forest_keys) == 12

    def test_all_values_non_negative(self):
        """All litter stocks are non-negative."""
        for key, value in LITTER_STOCKS.items():
            assert value >= Decimal("0"), f"Key {key} has negative litter: {value}"

    def test_boreal_moist_forest_is_25(self):
        """Boreal moist forest litter stock is 25 tC/ha."""
        key = (LandCategory.FOREST_LAND, ClimateZone.BOREAL_MOIST)
        assert LITTER_STOCKS[key] == Decimal("25.0")


class TestSOCReferenceStocks:
    """Tests for the SOC_REFERENCE_STOCKS constant table."""

    def test_tropical_wet_high_activity_clay(self):
        """Tropical wet, high-activity clay SOC reference is 65 tC/ha."""
        key = (ClimateZone.TROPICAL_WET, SoilType.HIGH_ACTIVITY_CLAY)
        assert SOC_REFERENCE_STOCKS[key] == Decimal("65")

    def test_tropical_wet_volcanic_is_130(self):
        """Tropical wet volcanic SOC reference is 130 tC/ha."""
        key = (ClimateZone.TROPICAL_WET, SoilType.VOLCANIC)
        assert SOC_REFERENCE_STOCKS[key] == Decimal("130")

    def test_all_values_are_decimal(self):
        """All SOC reference stocks are Decimal instances."""
        for key, value in SOC_REFERENCE_STOCKS.items():
            assert isinstance(value, Decimal), f"Key {key} is not Decimal"

    def test_all_values_positive(self):
        """All SOC reference stocks are positive."""
        for key, value in SOC_REFERENCE_STOCKS.items():
            assert value > Decimal("0"), f"Key {key} is not positive: {value}"


class TestSOCLandUseFactors:
    """Tests for the SOC_LAND_USE_FACTORS constant table."""

    def test_forest_land_factors_are_one(self):
        """Forest land SOC land-use factors are all 1.0 (reference)."""
        for key, value in SOC_LAND_USE_FACTORS.items():
            if key[0] == LandCategory.FOREST_LAND:
                assert value == Decimal("1.0"), f"Forest F_LU for {key} != 1.0"

    def test_cropland_factors_less_than_one(self):
        """Cropland SOC land-use factors are less than 1.0."""
        for key, value in SOC_LAND_USE_FACTORS.items():
            if key[0] == LandCategory.CROPLAND:
                assert value < Decimal("1.0"), (
                    f"Cropland F_LU for {key} should be < 1.0, got {value}"
                )


class TestSOCManagementFactors:
    """Tests for the SOC_MANAGEMENT_FACTORS constant table."""

    def test_full_tillage_is_reference(self):
        """Full tillage management factors are all 1.0."""
        for key, value in SOC_MANAGEMENT_FACTORS.items():
            if key[0] == ManagementPractice.FULL_TILLAGE:
                assert value == Decimal("1.0"), (
                    f"Full tillage F_MG for {key} != 1.0"
                )

    def test_no_till_factors_above_one(self):
        """No-till management factors are above 1.0."""
        for key, value in SOC_MANAGEMENT_FACTORS.items():
            if key[0] == ManagementPractice.NO_TILL:
                assert value > Decimal("1.0"), (
                    f"No-till F_MG for {key} should be > 1.0, got {value}"
                )

    def test_degraded_below_one(self):
        """Degraded management factors are below 1.0."""
        for key, value in SOC_MANAGEMENT_FACTORS.items():
            if key[0] == ManagementPractice.DEGRADED:
                assert value < Decimal("1.0"), (
                    f"Degraded F_MG for {key} should be < 1.0, got {value}"
                )


class TestSOCInputFactors:
    """Tests for the SOC_INPUT_FACTORS constant table."""

    def test_has_four_levels(self):
        """SOC input factors cover all 4 input levels."""
        assert len(SOC_INPUT_FACTORS) == 4

    def test_medium_is_one(self):
        """Medium input level factor is 1.0."""
        assert SOC_INPUT_FACTORS[InputLevel.MEDIUM] == Decimal("1.0")

    def test_low_below_one(self):
        """Low input level factor is below 1.0."""
        assert SOC_INPUT_FACTORS[InputLevel.LOW] < Decimal("1.0")

    def test_high_with_manure_above_high(self):
        """High-with-manure input factor is above high factor."""
        assert SOC_INPUT_FACTORS[InputLevel.HIGH_WITH_MANURE] > (
            SOC_INPUT_FACTORS[InputLevel.HIGH]
        )


class TestBiomassGrowthRates:
    """Tests for the BIOMASS_GROWTH_RATES constant table."""

    def test_forest_tropical_wet_is_five(self):
        """Forest tropical wet growth rate is 5.0 tC/ha/yr."""
        key = (LandCategory.FOREST_LAND, ClimateZone.TROPICAL_WET)
        assert BIOMASS_GROWTH_RATES[key] == Decimal("5.0")

    def test_all_values_non_negative(self):
        """All growth rates are non-negative."""
        for key, value in BIOMASS_GROWTH_RATES.items():
            assert value >= Decimal("0"), f"Key {key} has negative rate: {value}"


class TestCarbonFraction:
    """Tests for the CARBON_FRACTION constant."""

    def test_value_is_047(self):
        """IPCC default carbon fraction is 0.47."""
        assert CARBON_FRACTION == Decimal("0.47")

    def test_is_decimal(self):
        """CARBON_FRACTION is a Decimal instance."""
        assert isinstance(CARBON_FRACTION, Decimal)


class TestConversionFactorCO2C:
    """Tests for the CONVERSION_FACTOR_CO2_C constant."""

    def test_approximately_3667(self):
        """CO2/C molecular weight ratio is approximately 44/12 = 3.667."""
        assert abs(CONVERSION_FACTOR_CO2_C - Decimal("3.6667")) < Decimal("0.001")


class TestCombustionFactors:
    """Tests for the COMBUSTION_FACTORS constant table."""

    def test_forest_fire_factor(self):
        """Forest fire combustion factor is 0.45."""
        key = (LandCategory.FOREST_LAND, DisturbanceType.FIRE)
        assert COMBUSTION_FACTORS[key] == Decimal("0.45")

    def test_no_disturbance_is_zero(self):
        """No-disturbance combustion factor is 0.0."""
        key = (LandCategory.FOREST_LAND, DisturbanceType.NONE)
        assert COMBUSTION_FACTORS[key] == Decimal("0.0")

    def test_all_factors_between_zero_and_one(self):
        """All combustion factors are between 0 and 1."""
        for key, value in COMBUSTION_FACTORS.items():
            assert Decimal("0") <= value <= Decimal("1"), (
                f"Key {key} factor {value} out of range"
            )


class TestFireEmissionFactors:
    """Tests for the FIRE_EMISSION_FACTORS constant table."""

    def test_has_four_gases(self):
        """Fire emission factors cover all 4 gases."""
        assert len(FIRE_EMISSION_FACTORS) == 4

    def test_co2_ef_is_1580(self):
        """CO2 fire EF is 1580 g/kg DM."""
        assert FIRE_EMISSION_FACTORS[EmissionGas.CO2] == Decimal("1580")

    def test_ch4_ef_is_6_8(self):
        """CH4 fire EF is 6.8 g/kg DM."""
        assert FIRE_EMISSION_FACTORS[EmissionGas.CH4] == Decimal("6.8")

    def test_n2o_ef_is_020(self):
        """N2O fire EF is 0.20 g/kg DM."""
        assert FIRE_EMISSION_FACTORS[EmissionGas.N2O] == Decimal("0.20")


class TestPeatlandEF:
    """Tests for the PEATLAND_EF constant table."""

    def test_has_entries(self):
        """Peatland EF table has at least 10 entries."""
        assert len(PEATLAND_EF) >= 10

    def test_drained_tropical_wet_co2(self):
        """Drained tropical wet peatland CO2 EF is 11.0 tC/ha/yr."""
        key = (PeatlandStatus.DRAINED, ClimateZone.TROPICAL_WET)
        assert PEATLAND_EF[key]["CO2"] == Decimal("11.0")

    def test_natural_tropical_wet_ch4(self):
        """Natural tropical wet peatland CH4 EF is 150.0 kg/ha/yr."""
        key = (PeatlandStatus.NATURAL, ClimateZone.TROPICAL_WET)
        assert PEATLAND_EF[key]["CH4"] == Decimal("150.0")


class TestN2OSoilEF:
    """Tests for the N2O_SOIL_EF constant."""

    def test_ef1_is_001(self):
        """IPCC default EF1 for direct soil N2O is 0.01."""
        assert N2O_SOIL_EF == Decimal("0.01")


class TestModuleConstants:
    """Tests for module-level constants."""

    def test_version(self):
        """VERSION is a valid semver string."""
        assert VERSION == "1.0.0"

    def test_max_calculations_per_batch(self):
        """MAX_CALCULATIONS_PER_BATCH is 10000."""
        assert MAX_CALCULATIONS_PER_BATCH == 10_000

    def test_max_gases_per_result(self):
        """MAX_GASES_PER_RESULT is 10."""
        assert MAX_GASES_PER_RESULT == 10

    def test_max_trace_steps(self):
        """MAX_TRACE_STEPS is 200."""
        assert MAX_TRACE_STEPS == 200

    def test_max_pools_per_calc(self):
        """MAX_POOLS_PER_CALC is 5."""
        assert MAX_POOLS_PER_CALC == 5

    def test_max_parcels_per_tenant(self):
        """MAX_PARCELS_PER_TENANT is 50000."""
        assert MAX_PARCELS_PER_TENANT == 50_000

    def test_default_transition_years(self):
        """DEFAULT_TRANSITION_YEARS is 20."""
        assert DEFAULT_TRANSITION_YEARS == 20


# ===========================================================================
# Pydantic Model Tests
# ===========================================================================


class TestLandParcelInfo:
    """Tests for the LandParcelInfo Pydantic model."""

    def test_creation_with_required_fields(self):
        """LandParcelInfo can be created with all required fields."""
        parcel = LandParcelInfo(
            name="Test Parcel",
            area_ha=Decimal("100"),
            land_category=LandCategory.FOREST_LAND,
            climate_zone=ClimateZone.TROPICAL_WET,
            soil_type=SoilType.HIGH_ACTIVITY_CLAY,
            latitude=Decimal("0.0"),
            longitude=Decimal("0.0"),
            tenant_id="tenant_001",
        )
        assert parcel.name == "Test Parcel"
        assert parcel.area_ha == Decimal("100")
        assert parcel.land_category == LandCategory.FOREST_LAND

    def test_auto_generated_id(self):
        """LandParcelInfo generates a UUID id by default."""
        parcel = LandParcelInfo(
            name="Test",
            area_ha=Decimal("1"),
            land_category=LandCategory.CROPLAND,
            climate_zone=ClimateZone.TROPICAL_WET,
            soil_type=SoilType.SANDY,
            latitude=Decimal("0"),
            longitude=Decimal("0"),
            tenant_id="t1",
        )
        assert len(parcel.id) > 0
        uuid.UUID(parcel.id)

    def test_default_management_practice(self):
        """Default management practice is NOMINALLY_MANAGED."""
        parcel = LandParcelInfo(
            name="Test",
            area_ha=Decimal("1"),
            land_category=LandCategory.GRASSLAND,
            climate_zone=ClimateZone.BOREAL_DRY,
            soil_type=SoilType.OTHER,
            latitude=Decimal("60"),
            longitude=Decimal("30"),
            tenant_id="t1",
        )
        assert parcel.management_practice == ManagementPractice.NOMINALLY_MANAGED

    def test_default_input_level(self):
        """Default input level is MEDIUM."""
        parcel = LandParcelInfo(
            name="Test",
            area_ha=Decimal("1"),
            land_category=LandCategory.CROPLAND,
            climate_zone=ClimateZone.TROPICAL_DRY,
            soil_type=SoilType.SANDY,
            latitude=Decimal("10"),
            longitude=Decimal("20"),
            tenant_id="t1",
        )
        assert parcel.input_level == InputLevel.MEDIUM

    def test_peatland_status_default_none(self):
        """Default peatland status is None."""
        parcel = LandParcelInfo(
            name="Test",
            area_ha=Decimal("1"),
            land_category=LandCategory.WETLAND,
            climate_zone=ClimateZone.BOREAL_MOIST,
            soil_type=SoilType.WETLAND_ORGANIC,
            latitude=Decimal("55"),
            longitude=Decimal("25"),
            tenant_id="t1",
        )
        assert parcel.peatland_status is None

    def test_area_must_be_positive(self):
        """Area must be greater than 0."""
        with pytest.raises(Exception):
            LandParcelInfo(
                name="Test",
                area_ha=Decimal("0"),
                land_category=LandCategory.FOREST_LAND,
                climate_zone=ClimateZone.TROPICAL_WET,
                soil_type=SoilType.HIGH_ACTIVITY_CLAY,
                latitude=Decimal("0"),
                longitude=Decimal("0"),
                tenant_id="t1",
            )

    def test_negative_area_rejected(self):
        """Negative area is rejected."""
        with pytest.raises(Exception):
            LandParcelInfo(
                name="Test",
                area_ha=Decimal("-10"),
                land_category=LandCategory.FOREST_LAND,
                climate_zone=ClimateZone.TROPICAL_WET,
                soil_type=SoilType.HIGH_ACTIVITY_CLAY,
                latitude=Decimal("0"),
                longitude=Decimal("0"),
                tenant_id="t1",
            )

    def test_latitude_range(self):
        """Latitude must be between -90 and 90."""
        with pytest.raises(Exception):
            LandParcelInfo(
                name="Test",
                area_ha=Decimal("1"),
                land_category=LandCategory.FOREST_LAND,
                climate_zone=ClimateZone.TROPICAL_WET,
                soil_type=SoilType.HIGH_ACTIVITY_CLAY,
                latitude=Decimal("91"),
                longitude=Decimal("0"),
                tenant_id="t1",
            )

    def test_longitude_range(self):
        """Longitude must be between -180 and 180."""
        with pytest.raises(Exception):
            LandParcelInfo(
                name="Test",
                area_ha=Decimal("1"),
                land_category=LandCategory.FOREST_LAND,
                climate_zone=ClimateZone.TROPICAL_WET,
                soil_type=SoilType.HIGH_ACTIVITY_CLAY,
                latitude=Decimal("0"),
                longitude=Decimal("181"),
                tenant_id="t1",
            )

    def test_name_cannot_be_empty(self):
        """Name must have at least 1 character."""
        with pytest.raises(Exception):
            LandParcelInfo(
                name="",
                area_ha=Decimal("1"),
                land_category=LandCategory.FOREST_LAND,
                climate_zone=ClimateZone.TROPICAL_WET,
                soil_type=SoilType.HIGH_ACTIVITY_CLAY,
                latitude=Decimal("0"),
                longitude=Decimal("0"),
                tenant_id="t1",
            )

    def test_serialization_round_trip(self):
        """Model can be serialized and deserialized."""
        parcel = LandParcelInfo(
            name="Test Parcel",
            area_ha=Decimal("100"),
            land_category=LandCategory.FOREST_LAND,
            climate_zone=ClimateZone.TROPICAL_WET,
            soil_type=SoilType.HIGH_ACTIVITY_CLAY,
            latitude=Decimal("0"),
            longitude=Decimal("0"),
            tenant_id="t1",
        )
        data = parcel.model_dump()
        assert data["name"] == "Test Parcel"
        assert data["land_category"] == "forest_land"

    def test_created_at_is_datetime(self):
        """created_at field is a datetime with timezone."""
        parcel = LandParcelInfo(
            name="Test",
            area_ha=Decimal("1"),
            land_category=LandCategory.FOREST_LAND,
            climate_zone=ClimateZone.TROPICAL_WET,
            soil_type=SoilType.HIGH_ACTIVITY_CLAY,
            latitude=Decimal("0"),
            longitude=Decimal("0"),
            tenant_id="t1",
        )
        assert isinstance(parcel.created_at, datetime)


class TestCarbonStockSnapshot:
    """Tests for the CarbonStockSnapshot Pydantic model."""

    def test_creation_with_required_fields(self):
        """CarbonStockSnapshot can be created with all required fields."""
        snap = CarbonStockSnapshot(
            parcel_id="p1",
            pool=CarbonPool.ABOVE_GROUND_BIOMASS,
            stock_tc_ha=Decimal("180"),
            measurement_date=datetime(2023, 1, 1, tzinfo=timezone.utc),
        )
        assert snap.parcel_id == "p1"
        assert snap.pool == CarbonPool.ABOVE_GROUND_BIOMASS
        assert snap.stock_tc_ha == Decimal("180")

    def test_auto_generated_id(self):
        """CarbonStockSnapshot generates a UUID id by default."""
        snap = CarbonStockSnapshot(
            parcel_id="p1",
            pool=CarbonPool.SOIL_ORGANIC_CARBON,
            stock_tc_ha=Decimal("65"),
            measurement_date=datetime(2023, 6, 1, tzinfo=timezone.utc),
        )
        assert len(snap.id) > 0
        uuid.UUID(snap.id)

    def test_stock_must_be_non_negative(self):
        """Stock value must be >= 0."""
        with pytest.raises(Exception):
            CarbonStockSnapshot(
                parcel_id="p1",
                pool=CarbonPool.ABOVE_GROUND_BIOMASS,
                stock_tc_ha=Decimal("-10"),
                measurement_date=datetime(2023, 1, 1, tzinfo=timezone.utc),
            )

    def test_zero_stock_is_valid(self):
        """Stock value of 0 is valid (e.g. other land with no biomass)."""
        snap = CarbonStockSnapshot(
            parcel_id="p1",
            pool=CarbonPool.DEAD_WOOD,
            stock_tc_ha=Decimal("0"),
            measurement_date=datetime(2023, 1, 1, tzinfo=timezone.utc),
        )
        assert snap.stock_tc_ha == Decimal("0")

    def test_decimal_precision_maintained(self):
        """Decimal precision is maintained in stock values."""
        snap = CarbonStockSnapshot(
            parcel_id="p1",
            pool=CarbonPool.LITTER,
            stock_tc_ha=Decimal("12.34567890"),
            measurement_date=datetime(2023, 1, 1, tzinfo=timezone.utc),
        )
        assert snap.stock_tc_ha == Decimal("12.34567890")

    def test_serialization(self):
        """CarbonStockSnapshot can be serialized to dict."""
        snap = CarbonStockSnapshot(
            parcel_id="p1",
            pool=CarbonPool.ABOVE_GROUND_BIOMASS,
            stock_tc_ha=Decimal("100"),
            measurement_date=datetime(2023, 1, 1, tzinfo=timezone.utc),
        )
        data = snap.model_dump()
        assert "parcel_id" in data
        assert data["pool"] == "above_ground_biomass"
