# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-MRV-009 Scope 2 Location-Based Emissions Agent Data Models.

Tests all 18 enumerations, 10 constant tables (GWP_VALUES, EGRID_FACTORS,
IEA_COUNTRY_FACTORS, EU_COUNTRY_FACTORS, DEFRA_FACTORS, TD_LOSS_FACTORS,
STEAM_DEFAULT_EF, HEAT_DEFAULT_EF, COOLING_DEFAULT_EF, UNIT_CONVERSIONS),
module-level constants, and 18 Pydantic data models with field validators.

Target: 100+ tests, 85%+ coverage of models.py.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict

import pytest

# ---------------------------------------------------------------------------
# Conditional import guard
# ---------------------------------------------------------------------------

try:
    from greenlang.scope2_location.models import (
        # Enumerations (18)
        EnergyType,
        EnergyUnit,
        GridRegionSource,
        CalculationMethod,
        EmissionGas,
        GWPSource,
        EmissionFactorSource,
        DataQualityTier,
        FacilityType,
        GridRegionType,
        TDLossMethod,
        TimeGranularity,
        ComplianceStatus,
        ReportingPeriod,
        ConsumptionDataSource,
        SteamType,
        CoolingType,
        HeatingType,
        # Constants
        GWP_VALUES,
        EGRID_FACTORS,
        IEA_COUNTRY_FACTORS,
        EU_COUNTRY_FACTORS,
        DEFRA_FACTORS,
        TD_LOSS_FACTORS,
        STEAM_DEFAULT_EF,
        HEAT_DEFAULT_EF,
        COOLING_DEFAULT_EF,
        UNIT_CONVERSIONS,
        # Module constants
        VERSION,
        MAX_CALCULATIONS_PER_BATCH,
        MAX_GASES_PER_RESULT,
        MAX_TRACE_STEPS,
        MAX_FACILITIES_PER_TENANT,
        MAX_ENERGY_RECORDS_PER_CALC,
        DEFAULT_MONTE_CARLO_ITERATIONS,
        DEFAULT_CONFIDENCE_LEVEL,
        TABLE_PREFIX,
        # Pydantic models
        FacilityInfo,
        GridRegion,
        GridEmissionFactor,
        EnergyConsumption,
        ElectricityConsumptionRequest,
        SteamHeatCoolingRequest,
        TransmissionLossInput,
        CalculationRequest,
        GasEmissionDetail,
        CalculationResult,
        BatchCalculationRequest,
        BatchCalculationResult,
        ComplianceCheckResult,
        UncertaintyRequest,
        UncertaintyResult,
        AggregationResult,
        GridFactorLookupResult,
        TDLossResult,
    )
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False

_SKIP = pytest.mark.skipif(not MODELS_AVAILABLE, reason="models not available")


# ===========================================================================
# Enumeration Tests (18 enums)
# ===========================================================================


@_SKIP
class TestEnergyType:
    """Tests for the EnergyType enumeration."""

    def test_member_count(self):
        """EnergyType has exactly 4 members."""
        assert len(EnergyType) == 4

    @pytest.mark.parametrize("member,expected", [
        (EnergyType.ELECTRICITY, "electricity"),
        (EnergyType.STEAM, "steam"),
        (EnergyType.HEATING, "heating"),
        (EnergyType.COOLING, "cooling"),
    ])
    def test_values(self, member, expected):
        """Each EnergyType member has the correct string value."""
        assert member.value == expected

    def test_membership_from_string(self):
        """EnergyType can be looked up from string value."""
        assert EnergyType("electricity") == EnergyType.ELECTRICITY

    def test_all_are_str_enum(self):
        """All EnergyType members are string-compatible."""
        for member in EnergyType:
            assert isinstance(member.value, str)


@_SKIP
class TestEnergyUnit:
    """Tests for the EnergyUnit enumeration."""

    def test_member_count(self):
        """EnergyUnit has exactly 5 members."""
        assert len(EnergyUnit) == 5

    @pytest.mark.parametrize("member,expected", [
        (EnergyUnit.KWH, "kwh"),
        (EnergyUnit.MWH, "mwh"),
        (EnergyUnit.GJ, "gj"),
        (EnergyUnit.MMBTU, "mmbtu"),
        (EnergyUnit.THERMS, "therms"),
    ])
    def test_values(self, member, expected):
        """Each EnergyUnit member has the correct string value."""
        assert member.value == expected


@_SKIP
class TestGridRegionSource:
    """Tests for the GridRegionSource enumeration."""

    def test_member_count(self):
        """GridRegionSource has exactly 6 members."""
        assert len(GridRegionSource) == 6

    @pytest.mark.parametrize("member,expected", [
        (GridRegionSource.EGRID, "egrid"),
        (GridRegionSource.IEA, "iea"),
        (GridRegionSource.EU_EEA, "eu_eea"),
        (GridRegionSource.DEFRA, "defra"),
        (GridRegionSource.NATIONAL, "national"),
        (GridRegionSource.CUSTOM, "custom"),
    ])
    def test_values(self, member, expected):
        """Each GridRegionSource member has the correct string value."""
        assert member.value == expected


@_SKIP
class TestCalculationMethod:
    """Tests for the CalculationMethod enumeration."""

    def test_member_count(self):
        """CalculationMethod has exactly 6 members."""
        assert len(CalculationMethod) == 6

    @pytest.mark.parametrize("member,expected", [
        (CalculationMethod.IPCC_TIER_1, "ipcc_tier_1"),
        (CalculationMethod.IPCC_TIER_2, "ipcc_tier_2"),
        (CalculationMethod.IPCC_TIER_3, "ipcc_tier_3"),
        (CalculationMethod.MASS_BALANCE, "mass_balance"),
        (CalculationMethod.DIRECT_MEASUREMENT, "direct_measurement"),
        (CalculationMethod.SPEND_BASED, "spend_based"),
    ])
    def test_values(self, member, expected):
        """Each CalculationMethod member has the correct string value."""
        assert member.value == expected


@_SKIP
class TestEmissionGas:
    """Tests for the EmissionGas enumeration."""

    def test_member_count(self):
        """EmissionGas has exactly 3 members."""
        assert len(EmissionGas) == 3

    @pytest.mark.parametrize("member,expected", [
        (EmissionGas.CO2, "CO2"),
        (EmissionGas.CH4, "CH4"),
        (EmissionGas.N2O, "N2O"),
    ])
    def test_values(self, member, expected):
        """Each EmissionGas member has the correct uppercase value."""
        assert member.value == expected


@_SKIP
class TestGWPSource:
    """Tests for the GWPSource enumeration."""

    def test_member_count(self):
        """GWPSource has exactly 4 members."""
        assert len(GWPSource) == 4

    @pytest.mark.parametrize("member,expected", [
        (GWPSource.AR4, "AR4"),
        (GWPSource.AR5, "AR5"),
        (GWPSource.AR6, "AR6"),
        (GWPSource.AR6_20YR, "AR6_20YR"),
    ])
    def test_values(self, member, expected):
        """Each GWPSource member has the correct value."""
        assert member.value == expected


@_SKIP
class TestEmissionFactorSource:
    """Tests for the EmissionFactorSource enumeration."""

    def test_member_count(self):
        """EmissionFactorSource has exactly 7 members."""
        assert len(EmissionFactorSource) == 7

    @pytest.mark.parametrize("member,expected", [
        (EmissionFactorSource.EGRID, "egrid"),
        (EmissionFactorSource.IEA, "iea"),
        (EmissionFactorSource.DEFRA, "defra"),
        (EmissionFactorSource.EU_EEA, "eu_eea"),
        (EmissionFactorSource.NATIONAL, "national"),
        (EmissionFactorSource.CUSTOM, "custom"),
        (EmissionFactorSource.IPCC, "ipcc"),
    ])
    def test_values(self, member, expected):
        """Each EmissionFactorSource member has the correct value."""
        assert member.value == expected


@_SKIP
class TestDataQualityTier:
    """Tests for the DataQualityTier enumeration."""

    def test_member_count(self):
        """DataQualityTier has exactly 3 members."""
        assert len(DataQualityTier) == 3

    @pytest.mark.parametrize("member,expected", [
        (DataQualityTier.TIER_1, "tier_1"),
        (DataQualityTier.TIER_2, "tier_2"),
        (DataQualityTier.TIER_3, "tier_3"),
    ])
    def test_values(self, member, expected):
        """Each DataQualityTier member has the correct value."""
        assert member.value == expected


@_SKIP
class TestFacilityType:
    """Tests for the FacilityType enumeration."""

    def test_member_count(self):
        """FacilityType has exactly 8 members."""
        assert len(FacilityType) == 8

    @pytest.mark.parametrize("member,expected", [
        (FacilityType.OFFICE, "office"),
        (FacilityType.WAREHOUSE, "warehouse"),
        (FacilityType.MANUFACTURING, "manufacturing"),
        (FacilityType.RETAIL, "retail"),
        (FacilityType.DATA_CENTER, "data_center"),
        (FacilityType.HOSPITAL, "hospital"),
        (FacilityType.SCHOOL, "school"),
        (FacilityType.OTHER, "other"),
    ])
    def test_values(self, member, expected):
        """Each FacilityType member has the correct value."""
        assert member.value == expected


@_SKIP
class TestGridRegionType:
    """Tests for the GridRegionType enumeration."""

    def test_member_count(self):
        """GridRegionType has exactly 4 members."""
        assert len(GridRegionType) == 4

    @pytest.mark.parametrize("member,expected", [
        (GridRegionType.COUNTRY, "country"),
        (GridRegionType.SUBREGION, "subregion"),
        (GridRegionType.STATE, "state"),
        (GridRegionType.CUSTOM, "custom"),
    ])
    def test_values(self, member, expected):
        """Each GridRegionType member has the correct value."""
        assert member.value == expected


@_SKIP
class TestTDLossMethod:
    """Tests for the TDLossMethod enumeration."""

    def test_member_count(self):
        """TDLossMethod has exactly 3 members."""
        assert len(TDLossMethod) == 3

    @pytest.mark.parametrize("member,expected", [
        (TDLossMethod.COUNTRY_AVERAGE, "country_average"),
        (TDLossMethod.REGIONAL, "regional"),
        (TDLossMethod.CUSTOM, "custom"),
    ])
    def test_values(self, member, expected):
        """Each TDLossMethod member has the correct value."""
        assert member.value == expected


@_SKIP
class TestTimeGranularity:
    """Tests for the TimeGranularity enumeration."""

    def test_member_count(self):
        """TimeGranularity has exactly 3 members."""
        assert len(TimeGranularity) == 3

    @pytest.mark.parametrize("member,expected", [
        (TimeGranularity.ANNUAL, "annual"),
        (TimeGranularity.MONTHLY, "monthly"),
        (TimeGranularity.HOURLY, "hourly"),
    ])
    def test_values(self, member, expected):
        """Each TimeGranularity member has the correct value."""
        assert member.value == expected


@_SKIP
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
        """Each ComplianceStatus member has the correct value."""
        assert member.value == expected


@_SKIP
class TestReportingPeriod:
    """Tests for the ReportingPeriod enumeration."""

    def test_member_count(self):
        """ReportingPeriod has exactly 4 members."""
        assert len(ReportingPeriod) == 4

    @pytest.mark.parametrize("member,expected", [
        (ReportingPeriod.ANNUAL, "annual"),
        (ReportingPeriod.QUARTERLY, "quarterly"),
        (ReportingPeriod.MONTHLY, "monthly"),
        (ReportingPeriod.CUSTOM, "custom"),
    ])
    def test_values(self, member, expected):
        """Each ReportingPeriod member has the correct value."""
        assert member.value == expected


@_SKIP
class TestConsumptionDataSource:
    """Tests for the ConsumptionDataSource enumeration."""

    def test_member_count(self):
        """ConsumptionDataSource has exactly 4 members."""
        assert len(ConsumptionDataSource) == 4

    @pytest.mark.parametrize("member,expected", [
        (ConsumptionDataSource.METER, "meter"),
        (ConsumptionDataSource.INVOICE, "invoice"),
        (ConsumptionDataSource.ESTIMATE, "estimate"),
        (ConsumptionDataSource.BENCHMARK, "benchmark"),
    ])
    def test_values(self, member, expected):
        """Each ConsumptionDataSource member has the correct value."""
        assert member.value == expected


@_SKIP
class TestSteamType:
    """Tests for the SteamType enumeration."""

    def test_member_count(self):
        """SteamType has exactly 3 members."""
        assert len(SteamType) == 3

    @pytest.mark.parametrize("member,expected", [
        (SteamType.NATURAL_GAS, "natural_gas"),
        (SteamType.COAL, "coal"),
        (SteamType.BIOMASS, "biomass"),
    ])
    def test_values(self, member, expected):
        """Each SteamType member has the correct value."""
        assert member.value == expected


@_SKIP
class TestCoolingType:
    """Tests for the CoolingType enumeration."""

    def test_member_count(self):
        """CoolingType has exactly 3 members."""
        assert len(CoolingType) == 3

    @pytest.mark.parametrize("member,expected", [
        (CoolingType.ELECTRIC_CHILLER, "electric_chiller"),
        (CoolingType.ABSORPTION, "absorption"),
        (CoolingType.DISTRICT, "district"),
    ])
    def test_values(self, member, expected):
        """Each CoolingType member has the correct value."""
        assert member.value == expected


@_SKIP
class TestHeatingType:
    """Tests for the HeatingType enumeration."""

    def test_member_count(self):
        """HeatingType has exactly 3 members."""
        assert len(HeatingType) == 3

    @pytest.mark.parametrize("member,expected", [
        (HeatingType.DISTRICT, "district"),
        (HeatingType.GAS_BOILER, "gas_boiler"),
        (HeatingType.ELECTRIC, "electric"),
    ])
    def test_values(self, member, expected):
        """Each HeatingType member has the correct value."""
        assert member.value == expected


# ===========================================================================
# Constant Table Tests
# ===========================================================================


@_SKIP
class TestGWPValues:
    """Tests for the GWP_VALUES constant table."""

    def test_has_all_four_gwp_sources(self):
        """GWP_VALUES covers all 4 GWP assessment report editions."""
        assert len(GWP_VALUES) == 4
        for key in ("AR4", "AR5", "AR6", "AR6_20YR"):
            assert key in GWP_VALUES

    def test_each_source_has_three_gases(self):
        """Each GWP source maps to exactly 3 gases (CO2, CH4, N2O)."""
        for source, gases in GWP_VALUES.items():
            assert len(gases) == 3, f"{source} has {len(gases)} gases"
            assert "CO2" in gases
            assert "CH4" in gases
            assert "N2O" in gases

    def test_co2_is_always_one(self):
        """CO2 GWP is 1 across all assessment reports."""
        for source, gases in GWP_VALUES.items():
            assert gases["CO2"] == Decimal("1"), f"{source} CO2 != 1"

    @pytest.mark.parametrize("source,gas,expected", [
        ("AR4", "CH4", Decimal("25")),
        ("AR5", "CH4", Decimal("28")),
        ("AR6", "CH4", Decimal("27.9")),
        ("AR6_20YR", "CH4", Decimal("81.2")),
        ("AR4", "N2O", Decimal("298")),
        ("AR5", "N2O", Decimal("265")),
        ("AR6", "N2O", Decimal("273")),
        ("AR6_20YR", "N2O", Decimal("273")),
    ])
    def test_specific_gwp_values(self, source, gas, expected):
        """Specific GWP values match published IPCC data."""
        assert GWP_VALUES[source][gas] == expected

    def test_all_values_are_decimal(self):
        """All GWP values are Decimal instances for deterministic arithmetic."""
        for source, gases in GWP_VALUES.items():
            for gas, value in gases.items():
                assert isinstance(value, Decimal), f"{source}/{gas} not Decimal"


@_SKIP
class TestEGridFactors:
    """Tests for the EGRID_FACTORS constant table."""

    def test_has_27_subregions(self):
        """EGRID_FACTORS covers all 27 eGRID subregions."""
        assert len(EGRID_FACTORS) == 27

    def test_each_subregion_has_three_gases(self):
        """Each subregion has CO2, CH4, and N2O factors."""
        for subregion, gases in EGRID_FACTORS.items():
            assert "CO2" in gases, f"{subregion} missing CO2"
            assert "CH4" in gases, f"{subregion} missing CH4"
            assert "N2O" in gases, f"{subregion} missing N2O"

    @pytest.mark.parametrize("subregion,gas,expected", [
        ("CAMX", "CO2", Decimal("225.30")),
        ("CAMX", "CH4", Decimal("0.026")),
        ("CAMX", "N2O", Decimal("0.003")),
        ("ERCT", "CO2", Decimal("380.10")),
        ("NYUP", "CO2", Decimal("115.30")),
        ("SRMW", "CO2", Decimal("629.40")),
    ])
    def test_specific_egrid_factors(self, subregion, gas, expected):
        """Specific eGRID factors match EPA eGRID2022 data."""
        assert EGRID_FACTORS[subregion][gas] == expected

    def test_all_values_are_decimal(self):
        """All eGRID values are Decimal instances."""
        for subregion, gases in EGRID_FACTORS.items():
            for gas, value in gases.items():
                assert isinstance(value, Decimal), f"{subregion}/{gas} not Decimal"

    def test_all_co2_values_positive(self):
        """All CO2 emission rates are positive (no zero-carbon grid)."""
        for subregion, gases in EGRID_FACTORS.items():
            assert gases["CO2"] > Decimal("0"), f"{subregion} CO2 <= 0"

    def test_co2_dominates_ch4_n2o(self):
        """CO2 emission rate is much larger than CH4 and N2O for all subregions."""
        for subregion, gases in EGRID_FACTORS.items():
            assert gases["CO2"] > gases["CH4"] * Decimal("100"), (
                f"{subregion} CO2 not >> CH4"
            )
            assert gases["CO2"] > gases["N2O"] * Decimal("100"), (
                f"{subregion} CO2 not >> N2O"
            )

    def test_known_subregion_acronyms(self):
        """Known eGRID subregion acronyms are present."""
        expected_subregions = [
            "AKGD", "AKMS", "AZNM", "CAMX", "ERCT", "FRCC",
            "HIMS", "HIOA", "MROE", "MROW", "NEWE", "NWPP",
            "NYCW", "NYLI", "NYUP", "PRMS", "RFCE", "RFCM",
            "RFCW", "RMPA", "SPNO", "SPSO", "SRMV", "SRMW",
            "SRSO", "SRTV", "SRVC",
        ]
        for subregion in expected_subregions:
            assert subregion in EGRID_FACTORS, f"{subregion} missing"


@_SKIP
class TestIEACountryFactors:
    """Tests for the IEA_COUNTRY_FACTORS constant table."""

    def test_has_more_than_100_countries(self):
        """IEA factors cover more than 100 countries."""
        assert len(IEA_COUNTRY_FACTORS) > 100

    def test_all_values_are_decimal(self):
        """All IEA factors are Decimal instances (tCO2/MWh)."""
        for country, factor in IEA_COUNTRY_FACTORS.items():
            assert isinstance(factor, Decimal), f"{country} not Decimal"

    def test_all_values_non_negative(self):
        """All IEA factors are non-negative (some hydro countries = 0)."""
        for country, factor in IEA_COUNTRY_FACTORS.items():
            assert factor >= Decimal("0"), f"{country} has negative factor"

    @pytest.mark.parametrize("country,expected", [
        ("US", Decimal("0.379")),
        ("GB", Decimal("0.212")),
        ("DE", Decimal("0.338")),
        ("FR", Decimal("0.056")),
        ("CN", Decimal("0.555")),
        ("IN", Decimal("0.708")),
        ("JP", Decimal("0.457")),
        ("AU", Decimal("0.656")),
        ("SE", Decimal("0.008")),
        ("ZA", Decimal("0.928")),
    ])
    def test_specific_country_factors(self, country, expected):
        """Specific country factors match IEA 2023 data."""
        assert IEA_COUNTRY_FACTORS[country] == expected

    def test_zero_emission_countries(self):
        """Countries with predominantly hydro/renewable have zero factors."""
        zero_countries = ["IS", "PY", "NP", "ET"]
        for country in zero_countries:
            assert IEA_COUNTRY_FACTORS[country] == Decimal("0.000"), (
                f"{country} should be 0.000"
            )

    def test_keys_are_uppercase_iso_alpha2(self):
        """All keys are 2-character uppercase ISO country codes."""
        for country in IEA_COUNTRY_FACTORS:
            assert len(country) == 2, f"{country} not 2 chars"
            assert country == country.upper(), f"{country} not uppercase"


@_SKIP
class TestEUCountryFactors:
    """Tests for the EU_COUNTRY_FACTORS constant table."""

    def test_has_27_eu_members(self):
        """EU factors cover all 27 EU member states."""
        assert len(EU_COUNTRY_FACTORS) == 27

    def test_all_values_are_decimal(self):
        """All EU factors are Decimal instances."""
        for country, factor in EU_COUNTRY_FACTORS.items():
            assert isinstance(factor, Decimal)

    @pytest.mark.parametrize("country,expected", [
        ("DE", Decimal("0.338")),
        ("FR", Decimal("0.056")),
        ("PL", Decimal("0.635")),
        ("SE", Decimal("0.008")),
    ])
    def test_specific_eu_factors(self, country, expected):
        """Specific EU factors match EEA data."""
        assert EU_COUNTRY_FACTORS[country] == expected


@_SKIP
class TestDEFRAFactors:
    """Tests for the DEFRA_FACTORS constant table."""

    def test_has_six_entries(self):
        """DEFRA factors table has exactly 6 entries."""
        assert len(DEFRA_FACTORS) == 6

    def test_all_values_are_decimal(self):
        """All DEFRA factors are Decimal instances."""
        for key, value in DEFRA_FACTORS.items():
            assert isinstance(value, Decimal), f"{key} not Decimal"

    @pytest.mark.parametrize("key,expected", [
        ("electricity_generation", Decimal("0.20707")),
        ("electricity_td", Decimal("0.01879")),
        ("electricity_total", Decimal("0.22586")),
        ("steam", Decimal("0.07050")),
        ("heating", Decimal("0.04350")),
        ("cooling", Decimal("0.03210")),
    ])
    def test_specific_defra_values(self, key, expected):
        """Specific DEFRA factors match 2024 edition data."""
        assert DEFRA_FACTORS[key] == expected

    def test_total_equals_generation_plus_td(self):
        """Electricity total equals generation + T&D."""
        total = DEFRA_FACTORS["electricity_generation"] + DEFRA_FACTORS["electricity_td"]
        assert total == DEFRA_FACTORS["electricity_total"]


@_SKIP
class TestTDLossFactors:
    """Tests for the TD_LOSS_FACTORS constant table."""

    def test_has_more_than_50_countries(self):
        """TD loss factors cover more than 50 countries."""
        assert len(TD_LOSS_FACTORS) > 50

    def test_all_values_between_zero_and_one(self):
        """All T&D loss factors are between 0 and 1 (fractions)."""
        for country, loss in TD_LOSS_FACTORS.items():
            assert Decimal("0") <= loss <= Decimal("1"), (
                f"{country} loss {loss} out of range"
            )

    @pytest.mark.parametrize("country,expected", [
        ("US", Decimal("0.050")),
        ("GB", Decimal("0.077")),
        ("DE", Decimal("0.040")),
        ("IN", Decimal("0.194")),
        ("NG", Decimal("0.216")),
    ])
    def test_specific_td_loss_values(self, country, expected):
        """Specific T&D loss values match World Bank data."""
        assert TD_LOSS_FACTORS[country] == expected


@_SKIP
class TestSteamDefaultEF:
    """Tests for the STEAM_DEFAULT_EF constant table."""

    def test_has_four_fuel_sources(self):
        """Steam default EFs cover 4 fuel sources."""
        assert len(STEAM_DEFAULT_EF) == 4
        for key in ("natural_gas", "coal", "biomass", "oil"):
            assert key in STEAM_DEFAULT_EF

    def test_biomass_is_zero(self):
        """Biomass steam EF is 0 (biogenic CO2 reported separately)."""
        assert STEAM_DEFAULT_EF["biomass"] == Decimal("0.00")

    def test_coal_higher_than_gas(self):
        """Coal steam EF is higher than natural gas EF."""
        assert STEAM_DEFAULT_EF["coal"] > STEAM_DEFAULT_EF["natural_gas"]


@_SKIP
class TestHeatDefaultEF:
    """Tests for the HEAT_DEFAULT_EF constant table."""

    def test_has_three_types(self):
        """Heat default EFs cover 3 heating types."""
        assert len(HEAT_DEFAULT_EF) == 3

    def test_electric_is_zero(self):
        """Electric heating placeholder EF is 0 (grid factor used instead)."""
        assert HEAT_DEFAULT_EF["electric"] == Decimal("0.00")


@_SKIP
class TestCoolingDefaultEF:
    """Tests for the COOLING_DEFAULT_EF constant table."""

    def test_has_three_types(self):
        """Cooling default EFs cover 3 cooling types."""
        assert len(COOLING_DEFAULT_EF) == 3

    def test_electric_chiller_is_zero(self):
        """Electric chiller placeholder EF is 0 (grid factor used instead)."""
        assert COOLING_DEFAULT_EF["electric_chiller"] == Decimal("0.00")


@_SKIP
class TestUnitConversions:
    """Tests for the UNIT_CONVERSIONS constant table."""

    def test_has_eight_entries(self):
        """UNIT_CONVERSIONS has exactly 8 conversion factors."""
        assert len(UNIT_CONVERSIONS) == 8

    def test_all_values_are_decimal(self):
        """All unit conversion factors are Decimal instances."""
        for key, value in UNIT_CONVERSIONS.items():
            assert isinstance(value, Decimal), f"{key} not Decimal"

    @pytest.mark.parametrize("key,expected", [
        ("MWH_TO_GJ", Decimal("3.6")),
        ("GJ_TO_MWH", Decimal("0.277778")),
        ("MMBTU_TO_GJ", Decimal("1.05506")),
        ("KWH_TO_MWH", Decimal("0.001")),
        ("MWH_TO_KWH", Decimal("1000")),
    ])
    def test_specific_conversions(self, key, expected):
        """Specific unit conversion factors are correct."""
        assert UNIT_CONVERSIONS[key] == expected

    def test_kwh_mwh_inverse(self):
        """KWH_TO_MWH and MWH_TO_KWH are reciprocals."""
        product = UNIT_CONVERSIONS["KWH_TO_MWH"] * UNIT_CONVERSIONS["MWH_TO_KWH"]
        assert product == Decimal("1")


# ===========================================================================
# Module Constant Tests
# ===========================================================================


@_SKIP
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

    def test_max_facilities_per_tenant(self):
        """MAX_FACILITIES_PER_TENANT is 50000."""
        assert MAX_FACILITIES_PER_TENANT == 50_000

    def test_max_energy_records_per_calc(self):
        """MAX_ENERGY_RECORDS_PER_CALC is 1000."""
        assert MAX_ENERGY_RECORDS_PER_CALC == 1_000

    def test_default_monte_carlo_iterations(self):
        """DEFAULT_MONTE_CARLO_ITERATIONS is 10000."""
        assert DEFAULT_MONTE_CARLO_ITERATIONS == 10_000

    def test_default_confidence_level(self):
        """DEFAULT_CONFIDENCE_LEVEL is 0.95."""
        assert DEFAULT_CONFIDENCE_LEVEL == Decimal("0.95")

    def test_table_prefix(self):
        """TABLE_PREFIX is gl_s2l_."""
        assert TABLE_PREFIX == "gl_s2l_"


# ===========================================================================
# Pydantic Model Tests
# ===========================================================================


@_SKIP
class TestFacilityInfo:
    """Tests for the FacilityInfo Pydantic model."""

    def test_creation_with_required_fields(self):
        """FacilityInfo can be created with all required fields."""
        facility = FacilityInfo(
            name="Test Office",
            facility_type=FacilityType.OFFICE,
            country_code="US",
            grid_region_id="CAMX",
            tenant_id="t1",
        )
        assert facility.name == "Test Office"
        assert facility.facility_type == FacilityType.OFFICE
        assert facility.country_code == "US"

    def test_auto_generated_facility_id(self):
        """FacilityInfo generates a UUID facility_id by default."""
        facility = FacilityInfo(
            name="Test",
            facility_type=FacilityType.WAREHOUSE,
            country_code="DE",
            grid_region_id="EU-DE",
            tenant_id="t1",
        )
        assert len(facility.facility_id) > 0
        uuid.UUID(facility.facility_id)  # should not raise

    def test_country_code_uppercased(self):
        """Country code is normalized to uppercase."""
        facility = FacilityInfo(
            name="Test",
            facility_type=FacilityType.OFFICE,
            country_code="gb",
            grid_region_id="IEA-GB",
            tenant_id="t1",
        )
        assert facility.country_code == "GB"

    def test_egrid_subregion_uppercased(self):
        """eGRID subregion code is normalized to uppercase."""
        facility = FacilityInfo(
            name="Test",
            facility_type=FacilityType.OFFICE,
            country_code="US",
            grid_region_id="CAMX",
            egrid_subregion="camx",
            tenant_id="t1",
        )
        assert facility.egrid_subregion == "CAMX"

    def test_optional_latitude_longitude(self):
        """Latitude and longitude default to None."""
        facility = FacilityInfo(
            name="Test",
            facility_type=FacilityType.RETAIL,
            country_code="US",
            grid_region_id="ERCT",
            tenant_id="t1",
        )
        assert facility.latitude is None
        assert facility.longitude is None

    def test_latitude_range_validation(self):
        """Latitude must be between -90 and 90."""
        with pytest.raises(Exception):
            FacilityInfo(
                name="Test",
                facility_type=FacilityType.OFFICE,
                country_code="US",
                grid_region_id="X",
                tenant_id="t1",
                latitude=Decimal("91"),
            )

    def test_longitude_range_validation(self):
        """Longitude must be between -180 and 180."""
        with pytest.raises(Exception):
            FacilityInfo(
                name="Test",
                facility_type=FacilityType.OFFICE,
                country_code="US",
                grid_region_id="X",
                tenant_id="t1",
                longitude=Decimal("181"),
            )

    def test_empty_name_rejected(self):
        """Empty name is rejected by min_length=1 constraint."""
        with pytest.raises(Exception):
            FacilityInfo(
                name="",
                facility_type=FacilityType.OFFICE,
                country_code="US",
                grid_region_id="X",
                tenant_id="t1",
            )

    def test_model_is_frozen(self):
        """FacilityInfo instances are immutable (frozen=True)."""
        facility = FacilityInfo(
            name="Test",
            facility_type=FacilityType.OFFICE,
            country_code="US",
            grid_region_id="X",
            tenant_id="t1",
        )
        with pytest.raises(Exception):
            facility.name = "Changed"

    def test_serialization_round_trip(self):
        """FacilityInfo can be serialized to dict and back."""
        facility = FacilityInfo(
            name="Test Office",
            facility_type=FacilityType.DATA_CENTER,
            country_code="US",
            grid_region_id="CAMX",
            tenant_id="t1",
        )
        data = facility.model_dump()
        assert data["name"] == "Test Office"
        assert data["facility_type"] == "data_center"
        assert data["country_code"] == "US"


@_SKIP
class TestGridRegion:
    """Tests for the GridRegion Pydantic model."""

    def test_creation(self):
        """GridRegion can be created with required fields."""
        region = GridRegion(
            name="CAMX",
            region_type=GridRegionType.SUBREGION,
            source=GridRegionSource.EGRID,
            country_code="US",
        )
        assert region.name == "CAMX"
        assert region.region_type == GridRegionType.SUBREGION

    def test_auto_generated_id(self):
        """GridRegion generates a UUID region_id by default."""
        region = GridRegion(
            name="Test",
            region_type=GridRegionType.COUNTRY,
            source=GridRegionSource.IEA,
            country_code="GB",
        )
        uuid.UUID(region.region_id)


@_SKIP
class TestGridEmissionFactor:
    """Tests for the GridEmissionFactor Pydantic model."""

    def test_creation(self):
        """GridEmissionFactor can be created with valid data."""
        factor = GridEmissionFactor(
            region_id="CAMX",
            source=EmissionFactorSource.EGRID,
            year=2022,
            co2_kg_per_mwh=Decimal("225.30"),
            ch4_kg_per_mwh=Decimal("0.026"),
            n2o_kg_per_mwh=Decimal("0.003"),
            total_co2e_kg_per_mwh=Decimal("226.50"),
        )
        assert factor.co2_kg_per_mwh == Decimal("225.30")

    def test_year_range_validation(self):
        """Year must be between 1990 and 2100."""
        with pytest.raises(Exception):
            GridEmissionFactor(
                region_id="X",
                source=EmissionFactorSource.EGRID,
                year=1980,
                co2_kg_per_mwh=Decimal("0"),
                ch4_kg_per_mwh=Decimal("0"),
                n2o_kg_per_mwh=Decimal("0"),
                total_co2e_kg_per_mwh=Decimal("0"),
            )

    def test_negative_ef_rejected(self):
        """Negative emission factor values are rejected."""
        with pytest.raises(Exception):
            GridEmissionFactor(
                region_id="X",
                source=EmissionFactorSource.IEA,
                year=2023,
                co2_kg_per_mwh=Decimal("-1"),
                ch4_kg_per_mwh=Decimal("0"),
                n2o_kg_per_mwh=Decimal("0"),
                total_co2e_kg_per_mwh=Decimal("0"),
            )


@_SKIP
class TestEnergyConsumption:
    """Tests for the EnergyConsumption Pydantic model."""

    def test_creation(self):
        """EnergyConsumption can be created with valid data."""
        consumption = EnergyConsumption(
            facility_id="fac-001",
            energy_type=EnergyType.ELECTRICITY,
            quantity=Decimal("1000"),
            unit=EnergyUnit.MWH,
            period_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            period_end=datetime(2024, 12, 31, tzinfo=timezone.utc),
        )
        assert consumption.quantity == Decimal("1000")
        assert consumption.data_source == ConsumptionDataSource.INVOICE

    def test_period_end_before_start_rejected(self):
        """period_end must be after period_start."""
        with pytest.raises(Exception):
            EnergyConsumption(
                facility_id="fac-001",
                energy_type=EnergyType.ELECTRICITY,
                quantity=Decimal("100"),
                unit=EnergyUnit.MWH,
                period_start=datetime(2024, 12, 31, tzinfo=timezone.utc),
                period_end=datetime(2024, 1, 1, tzinfo=timezone.utc),
            )


@_SKIP
class TestElectricityConsumptionRequest:
    """Tests for the ElectricityConsumptionRequest Pydantic model."""

    def test_creation(self):
        """ElectricityConsumptionRequest can be created with valid data."""
        req = ElectricityConsumptionRequest(
            facility_id="fac-001",
            consumption_mwh=Decimal("1000"),
            period_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            period_end=datetime(2024, 12, 31, tzinfo=timezone.utc),
        )
        assert req.consumption_mwh == Decimal("1000")
        assert req.gwp_source == GWPSource.AR6
        assert req.include_td_losses is True
        assert req.time_granularity == TimeGranularity.ANNUAL

    def test_egrid_subregion_uppercased(self):
        """eGRID subregion is uppercased by validator."""
        req = ElectricityConsumptionRequest(
            facility_id="fac-001",
            consumption_mwh=Decimal("100"),
            period_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            period_end=datetime(2024, 12, 31, tzinfo=timezone.utc),
            egrid_subregion="camx",
        )
        assert req.egrid_subregion == "CAMX"

    def test_country_code_uppercased(self):
        """Country code is uppercased by validator."""
        req = ElectricityConsumptionRequest(
            facility_id="fac-001",
            consumption_mwh=Decimal("100"),
            period_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            period_end=datetime(2024, 12, 31, tzinfo=timezone.utc),
            country_code="us",
        )
        assert req.country_code == "US"


@_SKIP
class TestSteamHeatCoolingRequest:
    """Tests for the SteamHeatCoolingRequest Pydantic model."""

    def test_creation_steam(self):
        """SteamHeatCoolingRequest can be created for steam."""
        req = SteamHeatCoolingRequest(
            facility_id="fac-001",
            energy_type=EnergyType.STEAM,
            consumption_gj=Decimal("500"),
            period_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            period_end=datetime(2024, 12, 31, tzinfo=timezone.utc),
            steam_type=SteamType.NATURAL_GAS,
            country_code="GB",
        )
        assert req.energy_type == EnergyType.STEAM
        assert req.consumption_gj == Decimal("500")

    def test_electricity_energy_type_rejected(self):
        """ELECTRICITY energy type is rejected for SteamHeatCoolingRequest."""
        with pytest.raises(Exception):
            SteamHeatCoolingRequest(
                facility_id="fac-001",
                energy_type=EnergyType.ELECTRICITY,
                consumption_gj=Decimal("100"),
                period_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                period_end=datetime(2024, 12, 31, tzinfo=timezone.utc),
                country_code="US",
            )

    def test_country_code_uppercased(self):
        """Country code is uppercased by validator."""
        req = SteamHeatCoolingRequest(
            facility_id="fac-001",
            energy_type=EnergyType.COOLING,
            consumption_gj=Decimal("100"),
            period_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            period_end=datetime(2024, 12, 31, tzinfo=timezone.utc),
            country_code="sg",
        )
        assert req.country_code == "SG"


@_SKIP
class TestTransmissionLossInput:
    """Tests for the TransmissionLossInput Pydantic model."""

    def test_creation(self):
        """TransmissionLossInput can be created with valid data."""
        tl = TransmissionLossInput(
            country_code="US",
        )
        assert tl.country_code == "US"
        assert tl.method == TDLossMethod.COUNTRY_AVERAGE
        assert tl.include_upstream is False

    def test_custom_td_loss_bounds(self):
        """custom_td_loss must be between 0 and 1."""
        with pytest.raises(Exception):
            TransmissionLossInput(
                country_code="US",
                custom_td_loss=Decimal("1.5"),
            )

    def test_country_code_uppercased(self):
        """Country code is uppercased by validator."""
        tl = TransmissionLossInput(country_code="gb")
        assert tl.country_code == "GB"


@_SKIP
class TestGasEmissionDetail:
    """Tests for the GasEmissionDetail Pydantic model."""

    def test_creation(self):
        """GasEmissionDetail can be created with valid data."""
        detail = GasEmissionDetail(
            gas=EmissionGas.CO2,
            emission_kg=Decimal("225300"),
            gwp_factor=Decimal("1"),
            co2e_kg=Decimal("225300"),
        )
        assert detail.gas == EmissionGas.CO2
        assert detail.emission_kg == Decimal("225300")


@_SKIP
class TestCalculationResult:
    """Tests for the CalculationResult Pydantic model."""

    def test_creation(self):
        """CalculationResult can be created with valid data."""
        result = CalculationResult(
            calculation_id="calc-001",
            facility_id="fac-001",
            energy_type=EnergyType.ELECTRICITY,
            consumption_value=Decimal("1000"),
            consumption_unit=EnergyUnit.MWH,
            grid_region="CAMX",
            emission_factor_source=EmissionFactorSource.EGRID,
            ef_co2e_per_mwh=Decimal("226.50"),
            total_co2e_kg=Decimal("226500"),
            total_co2e_tonnes=Decimal("226.50"),
            provenance_hash="a" * 64,
        )
        assert result.total_co2e_tonnes == Decimal("226.50")
        assert len(result.provenance_hash) == 64

    def test_provenance_hash_length_validation(self):
        """Provenance hash must be exactly 64 characters (SHA-256)."""
        with pytest.raises(Exception):
            CalculationResult(
                calculation_id="calc-001",
                facility_id="fac-001",
                energy_type=EnergyType.ELECTRICITY,
                consumption_value=Decimal("100"),
                consumption_unit=EnergyUnit.MWH,
                grid_region="CAMX",
                emission_factor_source=EmissionFactorSource.EGRID,
                ef_co2e_per_mwh=Decimal("226"),
                total_co2e_kg=Decimal("22600"),
                total_co2e_tonnes=Decimal("22.6"),
                provenance_hash="short",
            )


@_SKIP
class TestUncertaintyRequest:
    """Tests for the UncertaintyRequest Pydantic model."""

    def test_creation_monte_carlo(self):
        """UncertaintyRequest defaults to monte_carlo method."""
        req = UncertaintyRequest(calculation_id="calc-001")
        assert req.method == "monte_carlo"
        assert req.iterations == 10_000
        assert req.confidence_level == Decimal("0.95")

    def test_invalid_method_rejected(self):
        """Invalid uncertainty method is rejected."""
        with pytest.raises(Exception):
            UncertaintyRequest(
                calculation_id="calc-001",
                method="invalid_method",
            )

    def test_analytical_method(self):
        """Analytical method is accepted."""
        req = UncertaintyRequest(
            calculation_id="calc-001",
            method="analytical",
        )
        assert req.method == "analytical"


@_SKIP
class TestUncertaintyResult:
    """Tests for the UncertaintyResult Pydantic model."""

    def test_creation(self):
        """UncertaintyResult can be created with valid data."""
        result = UncertaintyResult(
            calculation_id="calc-001",
            method="monte_carlo",
            mean_co2e=Decimal("226.50"),
            std_dev=Decimal("15.30"),
            ci_lower=Decimal("196.50"),
            ci_upper=Decimal("256.50"),
            confidence_level=Decimal("0.95"),
            iterations=10000,
        )
        assert result.mean_co2e == Decimal("226.50")
        assert result.ci_lower < result.ci_upper


@_SKIP
class TestComplianceCheckResult:
    """Tests for the ComplianceCheckResult Pydantic model."""

    def test_creation(self):
        """ComplianceCheckResult can be created with valid data."""
        check = ComplianceCheckResult(
            calculation_id="calc-001",
            framework="GHG_PROTOCOL",
            status=ComplianceStatus.COMPLIANT,
        )
        assert check.framework == "GHG_PROTOCOL"
        assert check.status == ComplianceStatus.COMPLIANT
        assert check.findings == []
        assert check.recommendations == []


@_SKIP
class TestBatchCalculationRequest:
    """Tests for the BatchCalculationRequest Pydantic model."""

    def test_creation(self):
        """BatchCalculationRequest can be created with one request."""
        req = BatchCalculationRequest(
            tenant_id="t1",
            requests=[
                CalculationRequest(
                    tenant_id="t1",
                    facility_id="fac-001",
                ),
            ],
        )
        assert len(req.requests) == 1

    def test_empty_requests_rejected(self):
        """Empty requests list is rejected (min_length=1)."""
        with pytest.raises(Exception):
            BatchCalculationRequest(
                tenant_id="t1",
                requests=[],
            )


@_SKIP
class TestAggregationResult:
    """Tests for the AggregationResult Pydantic model."""

    def test_creation(self):
        """AggregationResult can be created with valid data."""
        agg = AggregationResult(
            group_by="energy_type",
            period="2024",
            total_co2e_tonnes=Decimal("500"),
            facility_count=5,
        )
        assert agg.total_co2e_tonnes == Decimal("500")
        assert agg.facility_count == 5


@_SKIP
class TestGridFactorLookupResult:
    """Tests for the GridFactorLookupResult Pydantic model."""

    def test_creation(self):
        """GridFactorLookupResult can be created with valid data."""
        result = GridFactorLookupResult(
            region_id="CAMX",
            source=EmissionFactorSource.EGRID,
            year=2022,
            co2_kg_per_mwh=Decimal("225.30"),
            ch4_kg_per_mwh=Decimal("0.026"),
            n2o_kg_per_mwh=Decimal("0.003"),
            total_co2e_kg_per_mwh=Decimal("226.50"),
        )
        assert result.quality_tier == DataQualityTier.TIER_1


@_SKIP
class TestTDLossResult:
    """Tests for the TDLossResult Pydantic model."""

    def test_creation(self):
        """TDLossResult can be created with valid data."""
        result = TDLossResult(
            country_code="US",
            td_loss_pct=Decimal("0.050"),
            method=TDLossMethod.COUNTRY_AVERAGE,
            gross_consumption=Decimal("1052.63"),
            net_consumption=Decimal("1000"),
            loss_emissions_kg=Decimal("11831"),
        )
        assert result.td_loss_pct == Decimal("0.050")
        assert result.gross_consumption > result.net_consumption
