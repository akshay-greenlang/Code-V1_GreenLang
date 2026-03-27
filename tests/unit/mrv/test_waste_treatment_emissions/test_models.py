# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-MRV-007 Waste Treatment Emissions Agent Data Models.

Tests all 16 enumerations, constant tables (GWP, conversion factors,
DOC values, carbon content/fossil fractions, composting EFs,
incineration EFs, wastewater MCF), and 18 Pydantic models with
field validators.

Target: 120+ tests, 85%+ coverage.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict

import pytest

from greenlang.agents.mrv.waste_treatment_emissions.models import (
    # Enumerations (16)
    WasteCategory,
    TreatmentMethod,
    CompostingType,
    IncineratorType,
    WastewaterSystem,
    CalculationMethod,
    EmissionGas,
    GWPSource,
    EmissionFactorSource,
    DataQualityTier,
    FacilityType,
    BiogasComponent,
    ClimateZone,
    ComplianceStatus,
    ReportingPeriod,
    EmissionScope,
    # Constants
    GWP_VALUES,
    CONVERSION_FACTOR_CO2_C,
    CH4_C_RATIO,
    N2O_N_RATIO,
    CH4_DENSITY_STP,
    IPCC_DOC_VALUES,
    IPCC_CARBON_CONTENT,
    IPCC_COMPOSTING_EF,
    IPCC_INCINERATION_EF,
    IPCC_WASTEWATER_MCF,
    IPCC_MCF_VALUES,
    INCINERATION_NCV,
    HALF_LIFE_VALUES,
    WASTE_DEGRADABILITY_CLASS,
    OPEN_BURNING_EF,
    BMP_DEFAULTS,
    VS_FRACTION,
    BIOGAS_CH4_FRACTION,
    ADVANCED_THERMAL_EF,
    WASTEWATER_BO,
    WASTEWATER_N2O_EF,
    # Module constants
    VERSION,
    MAX_CALCULATIONS_PER_BATCH,
    MAX_GASES_PER_RESULT,
    MAX_TRACE_STEPS,
    MAX_STREAMS_PER_CALC,
    MAX_FACILITIES_PER_TENANT,
    DEFAULT_DOCf,
    DEFAULT_F_CH4_LFG,
    DEFAULT_OXIDATION_FACTOR,
    DEFAULT_OPEN_BURN_OXIDATION,
    DEFAULT_FLARE_DESTRUCTION_EFF,
    # Data models
    WasteComposition,
    TreatmentFacilityInfo,
    WasteStreamInfo,
    CalculationRequest,
    CalculationResult,
    GasEmissionDetail,
    BatchCalculationRequest,
    BatchCalculationResult,
    BiologicalTreatmentInput,
    ThermalTreatmentInput,
    WastewaterTreatmentInput,
    MethaneRecoveryRecord,
    EnergyRecoveryRecord,
)


# ===========================================================================
# Module Constants Tests
# ===========================================================================


class TestModuleConstants:
    """Tests for module-level constants."""

    def test_version_is_string(self):
        """VERSION is a valid version string."""
        assert isinstance(VERSION, str)
        assert VERSION == "1.0.0"

    def test_max_calculations_per_batch(self):
        """Maximum batch size is 10,000."""
        assert MAX_CALCULATIONS_PER_BATCH == 10_000

    def test_max_gases_per_result(self):
        """Maximum gas entries per result is 10."""
        assert MAX_GASES_PER_RESULT == 10

    def test_max_trace_steps(self):
        """Maximum trace steps is 200."""
        assert MAX_TRACE_STEPS == 200

    def test_max_streams_per_calc(self):
        """Maximum waste streams per calculation is 50."""
        assert MAX_STREAMS_PER_CALC == 50

    def test_max_facilities_per_tenant(self):
        """Maximum facilities per tenant is 10,000."""
        assert MAX_FACILITIES_PER_TENANT == 10_000

    def test_default_docf(self):
        """Default DOCf is 0.5."""
        assert DEFAULT_DOCf == Decimal("0.5")

    def test_default_f_ch4_lfg(self):
        """Default CH4 fraction in landfill gas is 0.5."""
        assert DEFAULT_F_CH4_LFG == Decimal("0.5")

    def test_default_oxidation_factor(self):
        """Default oxidation factor for incinerators is 1.0."""
        assert DEFAULT_OXIDATION_FACTOR == Decimal("1.0")

    def test_default_open_burn_oxidation(self):
        """Default open burning oxidation factor is 0.58."""
        assert DEFAULT_OPEN_BURN_OXIDATION == Decimal("0.58")

    def test_default_flare_destruction_eff(self):
        """Default flare destruction efficiency is 0.98."""
        assert DEFAULT_FLARE_DESTRUCTION_EFF == Decimal("0.98")


# ===========================================================================
# Enumeration Tests
# ===========================================================================


class TestWasteCategory:
    """Tests for the WasteCategory enumeration."""

    def test_member_count(self):
        """WasteCategory has exactly 19 members."""
        assert len(WasteCategory) == 19

    @pytest.mark.parametrize("member,value", [
        ("MSW", "msw"),
        ("INDUSTRIAL", "industrial"),
        ("CONSTRUCTION_DEMOLITION", "construction_demolition"),
        ("ORGANIC", "organic"),
        ("FOOD", "food"),
        ("YARD", "yard"),
        ("PAPER", "paper"),
        ("CARDBOARD", "cardboard"),
        ("PLASTIC", "plastic"),
        ("METAL", "metal"),
        ("GLASS", "glass"),
        ("TEXTILES", "textiles"),
        ("WOOD", "wood"),
        ("RUBBER", "rubber"),
        ("E_WASTE", "e_waste"),
        ("HAZARDOUS", "hazardous"),
        ("MEDICAL", "medical"),
        ("SLUDGE", "sludge"),
        ("MIXED", "mixed"),
    ])
    def test_waste_category_values(self, member: str, value: str):
        """Each WasteCategory member has the expected string value."""
        assert WasteCategory[member].value == value

    def test_waste_category_is_str_enum(self):
        """WasteCategory inherits from str."""
        assert isinstance(WasteCategory.MSW, str)
        assert WasteCategory.MSW == "msw"


class TestTreatmentMethod:
    """Tests for the TreatmentMethod enumeration."""

    def test_member_count(self):
        """TreatmentMethod has exactly 15 members."""
        assert len(TreatmentMethod) == 15

    @pytest.mark.parametrize("member,value", [
        ("LANDFILL", "landfill"),
        ("LANDFILL_GAS_CAPTURE", "landfill_gas_capture"),
        ("INCINERATION", "incineration"),
        ("INCINERATION_ENERGY_RECOVERY", "incineration_energy_recovery"),
        ("RECYCLING", "recycling"),
        ("COMPOSTING", "composting"),
        ("ANAEROBIC_DIGESTION", "anaerobic_digestion"),
        ("MBT", "mbt"),
        ("PYROLYSIS", "pyrolysis"),
        ("GASIFICATION", "gasification"),
        ("CHEMICAL_TREATMENT", "chemical_treatment"),
        ("THERMAL_TREATMENT", "thermal_treatment"),
        ("BIOLOGICAL_TREATMENT", "biological_treatment"),
        ("OPEN_BURNING", "open_burning"),
        ("OPEN_DUMPING", "open_dumping"),
    ])
    def test_treatment_method_values(self, member: str, value: str):
        """Each TreatmentMethod member has the expected string value."""
        assert TreatmentMethod[member].value == value


class TestCompostingType:
    """Tests for the CompostingType enumeration."""

    def test_member_count(self):
        """CompostingType has exactly 5 members."""
        assert len(CompostingType) == 5

    @pytest.mark.parametrize("member,value", [
        ("WINDROW", "windrow"),
        ("IN_VESSEL", "in_vessel"),
        ("AERATED_STATIC_PILE", "aerated_static_pile"),
        ("VERMICOMPOSTING", "vermicomposting"),
        ("HOME_COMPOSTING", "home_composting"),
    ])
    def test_composting_type_values(self, member: str, value: str):
        """Each CompostingType member has the expected string value."""
        assert CompostingType[member].value == value


class TestIncineratorType:
    """Tests for the IncineratorType enumeration."""

    def test_member_count(self):
        """IncineratorType has exactly 6 members."""
        assert len(IncineratorType) == 6

    @pytest.mark.parametrize("member,value", [
        ("STOKER_GRATE", "stoker_grate"),
        ("FLUIDIZED_BED", "fluidized_bed"),
        ("ROTARY_KILN", "rotary_kiln"),
        ("SEMI_CONTINUOUS", "semi_continuous"),
        ("BATCH_TYPE", "batch_type"),
        ("MODULAR", "modular"),
    ])
    def test_incinerator_type_values(self, member: str, value: str):
        """Each IncineratorType member has the expected string value."""
        assert IncineratorType[member].value == value


class TestWastewaterSystem:
    """Tests for the WastewaterSystem enumeration."""

    def test_member_count(self):
        """WastewaterSystem has exactly 8 members."""
        assert len(WastewaterSystem) == 8

    @pytest.mark.parametrize("member,value", [
        ("AEROBIC_WELL_MANAGED", "aerobic_well_managed"),
        ("AEROBIC_OVERLOADED", "aerobic_overloaded"),
        ("ANAEROBIC_REACTOR", "anaerobic_reactor"),
        ("ANAEROBIC_REACTOR_WITH_RECOVERY", "anaerobic_reactor_with_recovery"),
        ("ANAEROBIC_SHALLOW_LAGOON", "anaerobic_shallow_lagoon"),
        ("ANAEROBIC_DEEP_LAGOON", "anaerobic_deep_lagoon"),
        ("SEPTIC_SYSTEM", "septic_system"),
        ("UNTREATED_DISCHARGE", "untreated_discharge"),
    ])
    def test_wastewater_system_values(self, member: str, value: str):
        """Each WastewaterSystem member has the expected string value."""
        assert WastewaterSystem[member].value == value


class TestCalculationMethod:
    """Tests for the CalculationMethod enumeration."""

    def test_member_count(self):
        """CalculationMethod has exactly 7 members."""
        assert len(CalculationMethod) == 7

    @pytest.mark.parametrize("member,value", [
        ("IPCC_FOD", "ipcc_fod"),
        ("IPCC_TIER_1", "ipcc_tier_1"),
        ("IPCC_TIER_2", "ipcc_tier_2"),
        ("IPCC_TIER_3", "ipcc_tier_3"),
        ("MASS_BALANCE", "mass_balance"),
        ("DIRECT_MEASUREMENT", "direct_measurement"),
        ("SPEND_BASED", "spend_based"),
    ])
    def test_calculation_method_values(self, member: str, value: str):
        """Each CalculationMethod member has the expected string value."""
        assert CalculationMethod[member].value == value


class TestEmissionGas:
    """Tests for the EmissionGas enumeration."""

    def test_member_count(self):
        """EmissionGas has exactly 4 members."""
        assert len(EmissionGas) == 4

    @pytest.mark.parametrize("member,value", [
        ("CO2", "CO2"),
        ("CH4", "CH4"),
        ("N2O", "N2O"),
        ("CO", "CO"),
    ])
    def test_emission_gas_values(self, member: str, value: str):
        """Each EmissionGas member has the expected string value."""
        assert EmissionGas[member].value == value


class TestGWPSource:
    """Tests for the GWPSource enumeration."""

    def test_member_count(self):
        """GWPSource has exactly 4 members."""
        assert len(GWPSource) == 4

    @pytest.mark.parametrize("member,value", [
        ("AR4", "AR4"),
        ("AR5", "AR5"),
        ("AR6", "AR6"),
        ("AR6_20YR", "AR6_20YR"),
    ])
    def test_gwp_source_values(self, member: str, value: str):
        """Each GWPSource member has the expected string value."""
        assert GWPSource[member].value == value


class TestEmissionFactorSource:
    """Tests for the EmissionFactorSource enumeration."""

    def test_member_count(self):
        """EmissionFactorSource has exactly 7 members."""
        assert len(EmissionFactorSource) == 7

    @pytest.mark.parametrize("member,value", [
        ("IPCC_2006", "IPCC_2006"),
        ("IPCC_2019", "IPCC_2019"),
        ("EPA_AP42", "EPA_AP42"),
        ("DEFRA", "DEFRA"),
        ("ECOINVENT", "ECOINVENT"),
        ("NATIONAL", "NATIONAL"),
        ("CUSTOM", "CUSTOM"),
    ])
    def test_ef_source_values(self, member: str, value: str):
        """Each EmissionFactorSource member has the expected string value."""
        assert EmissionFactorSource[member].value == value


class TestDataQualityTier:
    """Tests for the DataQualityTier enumeration."""

    def test_member_count(self):
        """DataQualityTier has exactly 3 members."""
        assert len(DataQualityTier) == 3

    @pytest.mark.parametrize("member,value", [
        ("TIER_1", "tier_1"),
        ("TIER_2", "tier_2"),
        ("TIER_3", "tier_3"),
    ])
    def test_data_quality_tier_values(self, member: str, value: str):
        """Each DataQualityTier member has the expected string value."""
        assert DataQualityTier[member].value == value


class TestFacilityType:
    """Tests for the FacilityType enumeration."""

    def test_member_count(self):
        """FacilityType has 9 members."""
        assert len(FacilityType) == 9

    def test_industrial_onsite_value(self):
        """INDUSTRIAL_ONSITE has expected value."""
        assert FacilityType.INDUSTRIAL_ONSITE.value == "industrial_onsite"

    def test_multi_stream_value(self):
        """MULTI_STREAM has expected value."""
        assert FacilityType.MULTI_STREAM.value == "multi_stream"


class TestBiogasComponent:
    """Tests for the BiogasComponent enumeration."""

    def test_member_count(self):
        """BiogasComponent has 7 members."""
        assert len(BiogasComponent) == 7

    def test_methane_value(self):
        """METHANE has expected value."""
        assert BiogasComponent.METHANE.value == "methane"


class TestClimateZone:
    """Tests for the ClimateZone enumeration."""

    def test_member_count(self):
        """ClimateZone has 5 members."""
        assert len(ClimateZone) == 5

    @pytest.mark.parametrize("member,value", [
        ("TROPICAL", "tropical"),
        ("SUBTROPICAL", "subtropical"),
        ("TEMPERATE", "temperate"),
        ("BOREAL", "boreal"),
        ("POLAR", "polar"),
    ])
    def test_climate_zone_values(self, member: str, value: str):
        """Each ClimateZone member has the expected string value."""
        assert ClimateZone[member].value == value


class TestComplianceStatus:
    """Tests for the ComplianceStatus enumeration."""

    def test_member_count(self):
        """ComplianceStatus has 4 members."""
        assert len(ComplianceStatus) == 4

    @pytest.mark.parametrize("member,value", [
        ("COMPLIANT", "compliant"),
        ("NON_COMPLIANT", "non_compliant"),
        ("PARTIAL", "partial"),
        ("NOT_ASSESSED", "not_assessed"),
    ])
    def test_compliance_status_values(self, member: str, value: str):
        """Each ComplianceStatus member has the expected string value."""
        assert ComplianceStatus[member].value == value


class TestReportingPeriod:
    """Tests for the ReportingPeriod enumeration."""

    def test_member_count(self):
        """ReportingPeriod has 4 members."""
        assert len(ReportingPeriod) == 4

    @pytest.mark.parametrize("member,value", [
        ("ANNUAL", "annual"),
        ("QUARTERLY", "quarterly"),
        ("MONTHLY", "monthly"),
        ("AD_HOC", "ad_hoc"),
    ])
    def test_reporting_period_values(self, member: str, value: str):
        """Each ReportingPeriod member has the expected string value."""
        assert ReportingPeriod[member].value == value


class TestEmissionScope:
    """Tests for the EmissionScope enumeration."""

    def test_member_count(self):
        """EmissionScope has 3 members."""
        assert len(EmissionScope) == 3

    @pytest.mark.parametrize("member,value", [
        ("SCOPE_1", "scope_1"),
        ("SCOPE_2", "scope_2"),
        ("SCOPE_3", "scope_3"),
    ])
    def test_emission_scope_values(self, member: str, value: str):
        """Each EmissionScope member has the expected string value."""
        assert EmissionScope[member].value == value


# ===========================================================================
# GWP Values Tests
# ===========================================================================


class TestGWPValues:
    """Tests for IPCC GWP value constant tables."""

    def test_four_gwp_sources_present(self):
        """GWP_VALUES contains all 4 IPCC assessment report sources."""
        assert GWPSource.AR4 in GWP_VALUES
        assert GWPSource.AR5 in GWP_VALUES
        assert GWPSource.AR6 in GWP_VALUES
        assert GWPSource.AR6_20YR in GWP_VALUES

    def test_ar4_ch4_gwp(self):
        """AR4 CH4 GWP is 25."""
        assert GWP_VALUES[GWPSource.AR4]["CH4"] == Decimal("25")

    def test_ar5_ch4_gwp(self):
        """AR5 CH4 GWP is 28."""
        assert GWP_VALUES[GWPSource.AR5]["CH4"] == Decimal("28")

    def test_ar6_ch4_fossil_gwp(self):
        """AR6 fossil CH4 GWP is 29.8."""
        assert GWP_VALUES[GWPSource.AR6]["CH4"] == Decimal("29.8")

    def test_ar6_ch4_biogenic_gwp(self):
        """AR6 biogenic CH4 GWP is 27.0."""
        assert GWP_VALUES[GWPSource.AR6]["CH4_biogenic"] == Decimal("27.0")

    def test_ar6_n2o_gwp(self):
        """AR6 N2O GWP is 273."""
        assert GWP_VALUES[GWPSource.AR6]["N2O"] == Decimal("273")

    def test_ar4_n2o_gwp(self):
        """AR4 N2O GWP is 298."""
        assert GWP_VALUES[GWPSource.AR4]["N2O"] == Decimal("298")

    def test_ar6_20yr_ch4_gwp(self):
        """AR6 20-year CH4 GWP is 82.5."""
        assert GWP_VALUES[GWPSource.AR6_20YR]["CH4"] == Decimal("82.5")

    def test_ar6_20yr_ch4_biogenic_gwp(self):
        """AR6 20-year biogenic CH4 GWP is 80.8."""
        assert GWP_VALUES[GWPSource.AR6_20YR]["CH4_biogenic"] == Decimal("80.8")

    def test_co2_gwp_always_one(self):
        """CO2 GWP is 1 for all assessment reports."""
        for source in GWPSource:
            assert GWP_VALUES[source]["CO2"] == Decimal("1")

    def test_ar6_co_gwp(self):
        """AR6 CO GWP is 4.06."""
        assert GWP_VALUES[GWPSource.AR6]["CO"] == Decimal("4.06")


# ===========================================================================
# Conversion Factors Tests
# ===========================================================================


class TestConversionFactors:
    """Tests for molecular weight conversion factors."""

    def test_co2_c_ratio(self):
        """CO2:C ratio is 3.66667 (approximately 44/12)."""
        assert CONVERSION_FACTOR_CO2_C == Decimal("3.66667")
        # Verify it is approximately 44/12
        expected = Decimal("44") / Decimal("12")
        assert abs(CONVERSION_FACTOR_CO2_C - expected) < Decimal("0.001")

    def test_ch4_c_ratio(self):
        """CH4:C ratio is 1.33333 (approximately 16/12)."""
        assert CH4_C_RATIO == Decimal("1.33333")
        expected = Decimal("16") / Decimal("12")
        assert abs(CH4_C_RATIO - expected) < Decimal("0.001")

    def test_n2o_n_ratio(self):
        """N2O:N ratio is 1.57143 (approximately 44/28)."""
        assert N2O_N_RATIO == Decimal("1.57143")
        expected = Decimal("44") / Decimal("28")
        assert abs(N2O_N_RATIO - expected) < Decimal("0.001")

    def test_ch4_density_stp(self):
        """CH4 density at STP is 0.0007168 tonnes/m3."""
        assert CH4_DENSITY_STP == Decimal("0.0007168")


# ===========================================================================
# IPCC DOC Values Tests
# ===========================================================================


class TestIPCCDOCValues:
    """Tests for IPCC Degradable Organic Carbon fraction table."""

    def test_all_19_waste_types_present(self):
        """IPCC_DOC_VALUES contains all 19 waste categories."""
        assert len(IPCC_DOC_VALUES) == 19
        for cat in WasteCategory:
            assert cat in IPCC_DOC_VALUES, f"Missing DOC for {cat.name}"

    def test_paper_doc_is_040(self):
        """Paper DOC is 0.40."""
        assert IPCC_DOC_VALUES[WasteCategory.PAPER] == Decimal("0.40")

    def test_wood_doc_is_043(self):
        """Wood DOC is 0.43."""
        assert IPCC_DOC_VALUES[WasteCategory.WOOD] == Decimal("0.43")

    def test_food_doc_is_015(self):
        """Food waste DOC is 0.15."""
        assert IPCC_DOC_VALUES[WasteCategory.FOOD] == Decimal("0.15")

    def test_plastic_doc_is_zero(self):
        """Plastic DOC is 0.0 (not biodegradable)."""
        assert IPCC_DOC_VALUES[WasteCategory.PLASTIC] == Decimal("0.0")

    def test_metal_doc_is_zero(self):
        """Metal DOC is 0.0 (inorganic)."""
        assert IPCC_DOC_VALUES[WasteCategory.METAL] == Decimal("0.0")

    def test_glass_doc_is_zero(self):
        """Glass DOC is 0.0 (inorganic)."""
        assert IPCC_DOC_VALUES[WasteCategory.GLASS] == Decimal("0.0")

    def test_yard_doc_is_020(self):
        """Yard waste DOC is 0.20."""
        assert IPCC_DOC_VALUES[WasteCategory.YARD] == Decimal("0.20")

    def test_textiles_doc_is_024(self):
        """Textiles DOC is 0.24."""
        assert IPCC_DOC_VALUES[WasteCategory.TEXTILES] == Decimal("0.24")


# ===========================================================================
# IPCC Carbon Content Tests
# ===========================================================================


class TestIPCCCarbonContent:
    """Tests for IPCC carbon content and fossil fraction table."""

    def test_all_19_waste_types_present(self):
        """IPCC_CARBON_CONTENT contains all 19 waste categories."""
        assert len(IPCC_CARBON_CONTENT) == 19
        for cat in WasteCategory:
            assert cat in IPCC_CARBON_CONTENT, f"Missing carbon data for {cat.name}"

    def test_each_entry_has_required_keys(self):
        """Each carbon content entry has carbon_content_wet and fossil_carbon_fraction."""
        for cat, data in IPCC_CARBON_CONTENT.items():
            assert "carbon_content_wet" in data, f"Missing carbon_content_wet for {cat}"
            assert "fossil_carbon_fraction" in data, f"Missing fossil_fraction for {cat}"
            assert "dry_matter_fraction" in data, f"Missing dry_matter_fraction for {cat}"

    def test_plastic_is_100pct_fossil(self):
        """Plastic has 100% fossil carbon fraction."""
        assert IPCC_CARBON_CONTENT[WasteCategory.PLASTIC]["fossil_carbon_fraction"] == Decimal("1.0")

    def test_food_is_0pct_fossil(self):
        """Food waste has 0% fossil carbon fraction."""
        assert IPCC_CARBON_CONTENT[WasteCategory.FOOD]["fossil_carbon_fraction"] == Decimal("0.0")

    def test_wood_is_0pct_fossil(self):
        """Wood has 0% fossil carbon fraction (biogenic)."""
        assert IPCC_CARBON_CONTENT[WasteCategory.WOOD]["fossil_carbon_fraction"] == Decimal("0.0")

    def test_textiles_is_80pct_fossil(self):
        """Textiles has 80% fossil carbon fraction (synthetic dominated)."""
        assert IPCC_CARBON_CONTENT[WasteCategory.TEXTILES]["fossil_carbon_fraction"] == Decimal("0.80")

    def test_metal_has_zero_carbon(self):
        """Metal has 0% carbon content."""
        assert IPCC_CARBON_CONTENT[WasteCategory.METAL]["carbon_content_wet"] == Decimal("0.0")

    def test_glass_has_zero_carbon(self):
        """Glass has 0% carbon content."""
        assert IPCC_CARBON_CONTENT[WasteCategory.GLASS]["carbon_content_wet"] == Decimal("0.0")

    def test_plastic_high_carbon(self):
        """Plastic has high carbon content (0.67)."""
        assert IPCC_CARBON_CONTENT[WasteCategory.PLASTIC]["carbon_content_wet"] == Decimal("0.67")

    def test_values_in_valid_range(self):
        """All carbon content and fossil fractions are in [0, 1]."""
        for cat, data in IPCC_CARBON_CONTENT.items():
            assert Decimal("0") <= data["carbon_content_wet"] <= Decimal("1"), (
                f"carbon_content_wet out of range for {cat}"
            )
            assert Decimal("0") <= data["fossil_carbon_fraction"] <= Decimal("1"), (
                f"fossil_carbon_fraction out of range for {cat}"
            )


# ===========================================================================
# IPCC Composting EF Tests
# ===========================================================================


class TestIPCCCompostingEF:
    """Tests for IPCC composting and biological treatment emission factors."""

    def test_composting_well_managed_ch4(self):
        """Well-managed composting CH4 EF is 4.0 g/kg."""
        assert IPCC_COMPOSTING_EF["composting_well_managed"]["CH4"] == Decimal("4.0")

    def test_composting_well_managed_n2o(self):
        """Well-managed composting N2O EF is 0.24 g/kg."""
        assert IPCC_COMPOSTING_EF["composting_well_managed"]["N2O"] == Decimal("0.24")

    def test_composting_poorly_managed_ch4(self):
        """Poorly managed composting CH4 EF is 10.0 g/kg."""
        assert IPCC_COMPOSTING_EF["composting_poorly_managed"]["CH4"] == Decimal("10.0")

    def test_composting_poorly_managed_n2o(self):
        """Poorly managed composting N2O EF is 0.6 g/kg."""
        assert IPCC_COMPOSTING_EF["composting_poorly_managed"]["N2O"] == Decimal("0.6")

    def test_ad_vented_ch4(self):
        """AD vented CH4 EF is 2.0 g/kg."""
        assert IPCC_COMPOSTING_EF["anaerobic_digestion_vented"]["CH4"] == Decimal("2.0")

    def test_ad_flared_ch4(self):
        """AD flared CH4 EF is 0.8 g/kg."""
        assert IPCC_COMPOSTING_EF["anaerobic_digestion_flared"]["CH4"] == Decimal("0.8")

    def test_mbt_aerobic_ch4(self):
        """MBT aerobic CH4 EF is 4.0 g/kg."""
        assert IPCC_COMPOSTING_EF["mbt_aerobic"]["CH4"] == Decimal("4.0")

    def test_mbt_aerobic_n2o(self):
        """MBT aerobic N2O EF is 0.3 g/kg."""
        assert IPCC_COMPOSTING_EF["mbt_aerobic"]["N2O"] == Decimal("0.3")

    def test_all_entries_have_ch4_and_n2o(self):
        """All composting EF entries have CH4 and N2O keys."""
        for key, data in IPCC_COMPOSTING_EF.items():
            assert "CH4" in data, f"Missing CH4 for {key}"
            assert "N2O" in data, f"Missing N2O for {key}"


# ===========================================================================
# IPCC Incineration EF Tests
# ===========================================================================


class TestIPCCIncinerationEF:
    """Tests for IPCC incineration emission factors by technology."""

    def test_stoker_grate_n2o(self):
        """Stoker/grate N2O EF is 50 kg/Gg waste."""
        assert IPCC_INCINERATION_EF[IncineratorType.STOKER_GRATE]["N2O"] == Decimal("50")

    def test_stoker_grate_ch4(self):
        """Stoker/grate CH4 EF is 0.2 kg/Gg waste."""
        assert IPCC_INCINERATION_EF[IncineratorType.STOKER_GRATE]["CH4"] == Decimal("0.2")

    def test_fluidized_bed_n2o(self):
        """Fluidized bed N2O EF is 56 kg/Gg waste."""
        assert IPCC_INCINERATION_EF[IncineratorType.FLUIDIZED_BED]["N2O"] == Decimal("56")

    def test_fluidized_bed_ch4(self):
        """Fluidized bed CH4 EF is 0.68 kg/Gg waste."""
        assert IPCC_INCINERATION_EF[IncineratorType.FLUIDIZED_BED]["CH4"] == Decimal("0.68")

    def test_batch_type_ch4(self):
        """Batch type CH4 EF is 60 kg/Gg waste (highest)."""
        assert IPCC_INCINERATION_EF[IncineratorType.BATCH_TYPE]["CH4"] == Decimal("60")

    def test_batch_highest_ch4(self):
        """Batch type has the highest CH4 emission factor."""
        batch_ch4 = IPCC_INCINERATION_EF[IncineratorType.BATCH_TYPE]["CH4"]
        for itype, data in IPCC_INCINERATION_EF.items():
            assert data["CH4"] <= batch_ch4, (
                f"{itype} CH4 ({data['CH4']}) exceeds BATCH_TYPE ({batch_ch4})"
            )

    def test_all_incinerator_types_present(self):
        """IPCC_INCINERATION_EF contains all 6 incinerator types."""
        for itype in IncineratorType:
            assert itype in IPCC_INCINERATION_EF, f"Missing EF for {itype.name}"


# ===========================================================================
# IPCC Wastewater MCF Tests
# ===========================================================================


class TestIPCCWastewaterMCF:
    """Tests for IPCC wastewater methane correction factors."""

    def test_all_8_systems_present(self):
        """IPCC_WASTEWATER_MCF contains all 8 wastewater system types."""
        assert len(IPCC_WASTEWATER_MCF) == 8
        for sys in WastewaterSystem:
            assert sys in IPCC_WASTEWATER_MCF, f"Missing MCF for {sys.name}"

    def test_aerobic_well_managed_mcf_is_zero(self):
        """Well-managed aerobic treatment MCF is 0.0."""
        assert IPCC_WASTEWATER_MCF[WastewaterSystem.AEROBIC_WELL_MANAGED] == Decimal("0.0")

    def test_aerobic_overloaded_mcf_is_03(self):
        """Overloaded aerobic treatment MCF is 0.3."""
        assert IPCC_WASTEWATER_MCF[WastewaterSystem.AEROBIC_OVERLOADED] == Decimal("0.3")

    def test_anaerobic_reactor_mcf_is_08(self):
        """Anaerobic reactor MCF is 0.8."""
        assert IPCC_WASTEWATER_MCF[WastewaterSystem.ANAEROBIC_REACTOR] == Decimal("0.8")

    def test_septic_system_mcf_is_05(self):
        """Septic system MCF is 0.5."""
        assert IPCC_WASTEWATER_MCF[WastewaterSystem.SEPTIC_SYSTEM] == Decimal("0.5")

    def test_untreated_mcf_is_01(self):
        """Untreated discharge MCF is 0.1."""
        assert IPCC_WASTEWATER_MCF[WastewaterSystem.UNTREATED_DISCHARGE] == Decimal("0.1")

    def test_mcf_values_in_valid_range(self):
        """All MCF values are in [0.0, 1.0]."""
        for sys, mcf in IPCC_WASTEWATER_MCF.items():
            assert Decimal("0") <= mcf <= Decimal("1"), (
                f"MCF out of range for {sys}: {mcf}"
            )


# ===========================================================================
# WasteComposition Model Tests
# ===========================================================================


class TestWasteComposition:
    """Tests for the WasteComposition Pydantic model."""

    def test_valid_composition(self):
        """Valid composition with fractions summing to 1.0 is accepted."""
        comp = WasteComposition(
            name="Test MSW",
            fractions={"msw": Decimal("0.5"), "food": Decimal("0.5")},
        )
        assert comp.name == "Test MSW"
        assert sum(comp.fractions.values()) == Decimal("1.0")

    def test_empty_fractions_rejected(self):
        """Empty fractions dict raises ValidationError."""
        with pytest.raises(Exception):
            WasteComposition(name="Empty", fractions={})

    def test_fractions_not_summing_to_one_rejected(self):
        """Fractions far from 1.0 raise ValidationError."""
        with pytest.raises(Exception):
            WasteComposition(
                name="Bad Sum",
                fractions={"msw": Decimal("0.2"), "food": Decimal("0.1")},
            )

    def test_negative_fraction_rejected(self):
        """Negative fraction values raise ValidationError."""
        with pytest.raises(Exception):
            WasteComposition(
                name="Negative",
                fractions={"msw": Decimal("-0.5"), "food": Decimal("1.5")},
            )

    def test_fraction_over_one_rejected(self):
        """Fraction over 1.0 raises ValidationError."""
        with pytest.raises(Exception):
            WasteComposition(
                name="Over",
                fractions={"msw": Decimal("1.5")},
            )

    def test_default_moisture_content(self):
        """Default moisture content is 0.30."""
        comp = WasteComposition(
            name="Test",
            fractions={"msw": Decimal("1.0")},
        )
        assert comp.moisture_content == Decimal("0.30")


# ===========================================================================
# CalculationRequest Model Tests
# ===========================================================================


class TestCalculationRequest:
    """Tests for the CalculationRequest Pydantic model."""

    def test_valid_request_creation(self):
        """Valid CalculationRequest is created with required fields."""
        req = CalculationRequest(
            waste_category=WasteCategory.MSW,
            treatment_method=TreatmentMethod.INCINERATION,
            waste_mass_tonnes=Decimal("500"),
        )
        assert req.waste_category == WasteCategory.MSW
        assert req.treatment_method == TreatmentMethod.INCINERATION
        assert req.waste_mass_tonnes == Decimal("500")

    def test_default_calculation_method(self):
        """Default calculation method is IPCC_TIER_1."""
        req = CalculationRequest(
            waste_category=WasteCategory.MSW,
            treatment_method=TreatmentMethod.INCINERATION,
            waste_mass_tonnes=Decimal("100"),
        )
        assert req.calculation_method == CalculationMethod.IPCC_TIER_1

    def test_default_gwp_source(self):
        """Default GWP source is AR6."""
        req = CalculationRequest(
            waste_category=WasteCategory.MSW,
            treatment_method=TreatmentMethod.INCINERATION,
            waste_mass_tonnes=Decimal("100"),
        )
        assert req.gwp_source == GWPSource.AR6

    def test_default_scope(self):
        """Default scope is SCOPE_1."""
        req = CalculationRequest(
            waste_category=WasteCategory.MSW,
            treatment_method=TreatmentMethod.INCINERATION,
            waste_mass_tonnes=Decimal("100"),
        )
        assert req.scope == EmissionScope.SCOPE_1

    def test_auto_generated_id(self):
        """Request ID is auto-generated as UUID."""
        req = CalculationRequest(
            waste_category=WasteCategory.MSW,
            treatment_method=TreatmentMethod.INCINERATION,
            waste_mass_tonnes=Decimal("100"),
        )
        assert req.id is not None
        assert len(req.id) > 0

    def test_zero_mass_rejected(self):
        """Zero waste mass is rejected."""
        with pytest.raises(Exception):
            CalculationRequest(
                waste_category=WasteCategory.MSW,
                treatment_method=TreatmentMethod.INCINERATION,
                waste_mass_tonnes=Decimal("0"),
            )

    def test_negative_mass_rejected(self):
        """Negative waste mass is rejected."""
        with pytest.raises(Exception):
            CalculationRequest(
                waste_category=WasteCategory.MSW,
                treatment_method=TreatmentMethod.INCINERATION,
                waste_mass_tonnes=Decimal("-10"),
            )

    def test_default_reference_year(self):
        """Default reference year is 2025."""
        req = CalculationRequest(
            waste_category=WasteCategory.MSW,
            treatment_method=TreatmentMethod.INCINERATION,
            waste_mass_tonnes=Decimal("100"),
        )
        assert req.reference_year == 2025


# ===========================================================================
# CalculationResult Model Tests
# ===========================================================================


class TestCalculationResult:
    """Tests for the CalculationResult Pydantic model."""

    def test_valid_result_creation(self):
        """Valid CalculationResult is created with required fields."""
        result = CalculationResult(
            total_co2e=Decimal("150.5"),
            net_co2e=Decimal("150.5"),
            waste_category=WasteCategory.MSW,
            treatment_method=TreatmentMethod.INCINERATION,
            calculation_method=CalculationMethod.IPCC_TIER_2,
        )
        assert result.total_co2e == Decimal("150.5")
        assert result.net_co2e == Decimal("150.5")

    def test_auto_generated_result_id(self):
        """Result ID is auto-generated."""
        result = CalculationResult(
            total_co2e=Decimal("10"),
            net_co2e=Decimal("10"),
            waste_category=WasteCategory.FOOD,
            treatment_method=TreatmentMethod.COMPOSTING,
            calculation_method=CalculationMethod.IPCC_TIER_1,
        )
        assert result.id is not None
        assert len(result.id) > 0

    def test_default_energy_offset_is_zero(self):
        """Default energy offset is 0."""
        result = CalculationResult(
            total_co2e=Decimal("10"),
            net_co2e=Decimal("10"),
            waste_category=WasteCategory.FOOD,
            treatment_method=TreatmentMethod.COMPOSTING,
            calculation_method=CalculationMethod.IPCC_TIER_1,
        )
        assert result.energy_offset_tco2e == Decimal("0")

    def test_trace_steps_max_validation(self):
        """More than MAX_TRACE_STEPS trace steps raises error."""
        with pytest.raises(Exception):
            CalculationResult(
                total_co2e=Decimal("10"),
                net_co2e=Decimal("10"),
                waste_category=WasteCategory.FOOD,
                treatment_method=TreatmentMethod.COMPOSTING,
                calculation_method=CalculationMethod.IPCC_TIER_1,
                trace_steps=["step"] * (MAX_TRACE_STEPS + 1),
            )

    def test_gas_details_max_validation(self):
        """More than MAX_GASES_PER_RESULT gas details raises error."""
        details = [
            GasEmissionDetail(
                gas=EmissionGas.CO2,
                emission_mass_tonnes=Decimal("1"),
                emission_tco2e=Decimal("1"),
            )
            for _ in range(MAX_GASES_PER_RESULT + 1)
        ]
        with pytest.raises(Exception):
            CalculationResult(
                total_co2e=Decimal("10"),
                net_co2e=Decimal("10"),
                waste_category=WasteCategory.MSW,
                treatment_method=TreatmentMethod.INCINERATION,
                calculation_method=CalculationMethod.IPCC_TIER_2,
                gas_details=details,
            )


# ===========================================================================
# BatchCalculationRequest Model Tests
# ===========================================================================


class TestBatchCalculationRequest:
    """Tests for the BatchCalculationRequest Pydantic model."""

    def test_valid_batch_creation(self):
        """Valid batch with 1 calculation is accepted."""
        calc = CalculationRequest(
            waste_category=WasteCategory.MSW,
            treatment_method=TreatmentMethod.INCINERATION,
            waste_mass_tonnes=Decimal("100"),
        )
        batch = BatchCalculationRequest(calculations=[calc])
        assert len(batch.calculations) == 1

    def test_empty_batch_rejected(self):
        """Empty calculations list is rejected."""
        with pytest.raises(Exception):
            BatchCalculationRequest(calculations=[])

    def test_auto_generated_batch_id(self):
        """Batch ID is auto-generated."""
        calc = CalculationRequest(
            waste_category=WasteCategory.MSW,
            treatment_method=TreatmentMethod.INCINERATION,
            waste_mass_tonnes=Decimal("100"),
        )
        batch = BatchCalculationRequest(calculations=[calc])
        assert batch.id is not None
        assert len(batch.id) > 0

    def test_default_gwp_source(self):
        """Default batch GWP source is AR6."""
        calc = CalculationRequest(
            waste_category=WasteCategory.MSW,
            treatment_method=TreatmentMethod.INCINERATION,
            waste_mass_tonnes=Decimal("100"),
        )
        batch = BatchCalculationRequest(calculations=[calc])
        assert batch.gwp_source == GWPSource.AR6


# ===========================================================================
# BiologicalTreatmentInput Model Tests
# ===========================================================================


class TestBiologicalTreatmentInput:
    """Tests for the BiologicalTreatmentInput Pydantic model."""

    def test_valid_composting_input(self):
        """Valid composting input is accepted."""
        inp = BiologicalTreatmentInput(
            composting_type=CompostingType.WINDROW,
            is_well_managed=True,
            volatile_solids_fraction=Decimal("0.80"),
            ch4_recovery_fraction=Decimal("0.10"),
        )
        assert inp.composting_type == CompostingType.WINDROW
        assert inp.is_well_managed is True
        assert inp.volatile_solids_fraction == Decimal("0.80")
        assert inp.ch4_recovery_fraction == Decimal("0.10")

    def test_valid_ad_input(self):
        """Valid anaerobic digestion input is accepted."""
        inp = BiologicalTreatmentInput(
            volatile_solids_fraction=Decimal("0.87"),
            bmp=Decimal("400"),
            digestion_efficiency=Decimal("0.75"),
            ch4_fraction_biogas=Decimal("0.60"),
        )
        assert inp.volatile_solids_fraction == Decimal("0.87")
        assert inp.bmp == Decimal("400")
        assert inp.digestion_efficiency == Decimal("0.75")

    def test_default_is_well_managed(self):
        """Default is_well_managed is True."""
        inp = BiologicalTreatmentInput()
        assert inp.is_well_managed is True

    def test_default_ch4_recovery_fraction(self):
        """Default ch4_recovery_fraction is 0.0."""
        inp = BiologicalTreatmentInput()
        assert inp.ch4_recovery_fraction == Decimal("0.0")

    def test_default_digestion_efficiency(self):
        """Default digestion_efficiency is 0.7."""
        inp = BiologicalTreatmentInput()
        assert inp.digestion_efficiency == Decimal("0.7")

    def test_default_ch4_fraction_biogas(self):
        """Default ch4_fraction_biogas is 0.60."""
        inp = BiologicalTreatmentInput()
        assert inp.ch4_fraction_biogas == Decimal("0.60")

    def test_default_biogas_capture_efficiency(self):
        """Default biogas_capture_efficiency is 0.95."""
        inp = BiologicalTreatmentInput()
        assert inp.biogas_capture_efficiency == Decimal("0.95")

    def test_default_flare_destruction_efficiency(self):
        """Default flare_destruction_efficiency is 0.98."""
        inp = BiologicalTreatmentInput()
        assert inp.flare_destruction_efficiency == Decimal("0.98")

    def test_ch4_recovery_fraction_range(self):
        """ch4_recovery_fraction outside 0-1 is rejected."""
        with pytest.raises(Exception):
            BiologicalTreatmentInput(ch4_recovery_fraction=Decimal("1.5"))

    def test_volatile_solids_fraction_range(self):
        """volatile_solids_fraction outside 0-1 is rejected."""
        with pytest.raises(Exception):
            BiologicalTreatmentInput(volatile_solids_fraction=Decimal("2.0"))

    def test_biogas_flare_fraction_alone_exceeds_one_rejected(self):
        """Biogas flare fraction > 1.0 is rejected by range constraint."""
        with pytest.raises(Exception):
            BiologicalTreatmentInput(
                biogas_flare_fraction=Decimal("1.5"),
            )

    def test_biogas_utilization_fraction_exceeds_one_rejected(self):
        """Biogas utilization fraction > 1.0 is rejected by range constraint."""
        with pytest.raises(Exception):
            BiologicalTreatmentInput(
                biogas_utilization_fraction=Decimal("1.5"),
            )

    def test_biogas_vent_fraction_exceeds_one_rejected(self):
        """Biogas vent fraction > 1.0 is rejected by range constraint."""
        with pytest.raises(Exception):
            BiologicalTreatmentInput(
                biogas_vent_fraction=Decimal("1.5"),
            )

    def test_residence_time_valid(self):
        """Valid residence_time_days is accepted."""
        inp = BiologicalTreatmentInput(residence_time_days=30)
        assert inp.residence_time_days == 30

    def test_residence_time_zero_rejected(self):
        """Zero residence_time_days is rejected."""
        with pytest.raises(Exception):
            BiologicalTreatmentInput(residence_time_days=0)

    def test_temperature_celsius_range(self):
        """Valid temperature_celsius within range is accepted."""
        inp = BiologicalTreatmentInput(temperature_celsius=Decimal("55"))
        assert inp.temperature_celsius == Decimal("55")


# ===========================================================================
# ThermalTreatmentInput Model Tests
# ===========================================================================


class TestThermalTreatmentInput:
    """Tests for the ThermalTreatmentInput Pydantic model."""

    def test_valid_incineration_input(self):
        """Valid incineration input is accepted."""
        inp = ThermalTreatmentInput(
            incinerator_type=IncineratorType.STOKER_GRATE,
            oxidation_factor=Decimal("1.0"),
        )
        assert inp.incinerator_type == IncineratorType.STOKER_GRATE

    def test_default_oxidation_factor(self):
        """Default oxidation factor is 1.0."""
        inp = ThermalTreatmentInput()
        assert inp.oxidation_factor == Decimal("1.0")


# ===========================================================================
# WastewaterTreatmentInput Model Tests
# ===========================================================================


class TestWastewaterTreatmentInput:
    """Tests for the WastewaterTreatmentInput Pydantic model."""

    def test_valid_wastewater_input_with_bod(self):
        """Valid wastewater input with BOD basis is accepted."""
        inp = WastewaterTreatmentInput(
            system_type=WastewaterSystem.AEROBIC_OVERLOADED,
            total_organic_waste_kg=Decimal("50000"),
            organic_basis="bod",
        )
        assert inp.system_type == WastewaterSystem.AEROBIC_OVERLOADED
        assert inp.total_organic_waste_kg == Decimal("50000")
        assert inp.organic_basis == "bod"

    def test_valid_wastewater_input_with_cod(self):
        """Valid wastewater input with COD basis is accepted."""
        inp = WastewaterTreatmentInput(
            system_type=WastewaterSystem.ANAEROBIC_REACTOR,
            total_organic_waste_kg=Decimal("120000"),
            organic_basis="cod",
        )
        assert inp.system_type == WastewaterSystem.ANAEROBIC_REACTOR
        assert inp.organic_basis == "cod"

    def test_system_type_is_required(self):
        """system_type is required (no default)."""
        with pytest.raises(Exception):
            WastewaterTreatmentInput(
                total_organic_waste_kg=Decimal("1000"),
            )

    def test_total_organic_waste_is_required(self):
        """total_organic_waste_kg is required."""
        with pytest.raises(Exception):
            WastewaterTreatmentInput(
                system_type=WastewaterSystem.AEROBIC_WELL_MANAGED,
            )

    def test_default_organic_basis_is_cod(self):
        """Default organic_basis is 'cod'."""
        inp = WastewaterTreatmentInput(
            system_type=WastewaterSystem.AEROBIC_WELL_MANAGED,
            total_organic_waste_kg=Decimal("1000"),
        )
        assert inp.organic_basis == "cod"

    def test_default_sludge_removal_zero(self):
        """Default sludge_removal_kg is 0."""
        inp = WastewaterTreatmentInput(
            system_type=WastewaterSystem.AEROBIC_WELL_MANAGED,
            total_organic_waste_kg=Decimal("1000"),
        )
        assert inp.sludge_removal_kg == Decimal("0")

    def test_default_ch4_recovered_zero(self):
        """Default ch4_recovered_tonnes is 0."""
        inp = WastewaterTreatmentInput(
            system_type=WastewaterSystem.AEROBIC_WELL_MANAGED,
            total_organic_waste_kg=Decimal("1000"),
        )
        assert inp.ch4_recovered_tonnes == Decimal("0")

    def test_default_protein_consumption(self):
        """Default protein_consumption_kg_per_person is 25.0."""
        inp = WastewaterTreatmentInput(
            system_type=WastewaterSystem.AEROBIC_WELL_MANAGED,
            total_organic_waste_kg=Decimal("1000"),
        )
        assert inp.protein_consumption_kg_per_person == Decimal("25.0")

    def test_default_nitrogen_fraction_protein(self):
        """Default nitrogen_fraction_protein is 0.16."""
        inp = WastewaterTreatmentInput(
            system_type=WastewaterSystem.AEROBIC_WELL_MANAGED,
            total_organic_waste_kg=Decimal("1000"),
        )
        assert inp.nitrogen_fraction_protein == Decimal("0.16")

    def test_invalid_organic_basis_rejected(self):
        """Invalid organic_basis is rejected."""
        with pytest.raises(Exception):
            WastewaterTreatmentInput(
                system_type=WastewaterSystem.AEROBIC_WELL_MANAGED,
                total_organic_waste_kg=Decimal("1000"),
                organic_basis="toc",
            )

    def test_negative_total_organic_waste_rejected(self):
        """Negative total_organic_waste_kg is rejected."""
        with pytest.raises(Exception):
            WastewaterTreatmentInput(
                system_type=WastewaterSystem.AEROBIC_WELL_MANAGED,
                total_organic_waste_kg=Decimal("-100"),
            )


# ===========================================================================
# Additional Constant Table Tests
# ===========================================================================


class TestAdditionalConstantTables:
    """Tests for additional IPCC constant tables."""

    def test_ipcc_mcf_values_present(self):
        """IPCC_MCF_VALUES contains expected landfill types."""
        assert "managed_anaerobic" in IPCC_MCF_VALUES
        assert IPCC_MCF_VALUES["managed_anaerobic"] == Decimal("1.0")

    def test_ipcc_mcf_managed_semi_aerobic(self):
        """Managed semi-aerobic MCF is 0.5."""
        assert IPCC_MCF_VALUES["managed_semi_aerobic"] == Decimal("0.5")

    def test_incineration_ncv_all_categories(self):
        """INCINERATION_NCV covers all 19 waste categories."""
        assert len(INCINERATION_NCV) == 19
        for cat in WasteCategory:
            assert cat in INCINERATION_NCV, f"Missing NCV for {cat.name}"

    def test_plastic_highest_ncv(self):
        """Plastic has the highest NCV at 32 GJ/tonne."""
        assert INCINERATION_NCV[WasteCategory.PLASTIC] == Decimal("32.0")

    def test_metal_zero_ncv(self):
        """Metal has 0 GJ/tonne NCV."""
        assert INCINERATION_NCV[WasteCategory.METAL] == Decimal("0.0")

    def test_half_life_values_20_entries(self):
        """HALF_LIFE_VALUES contains 20 entries (5 zones x 4 classes)."""
        assert len(HALF_LIFE_VALUES) == 20

    def test_tropical_rapidly_degrading_half_life(self):
        """Tropical rapidly degrading half-life is 3 years."""
        assert HALF_LIFE_VALUES[(ClimateZone.TROPICAL, "rapidly_degrading")] == Decimal("3")

    def test_polar_very_slowly_degrading_half_life(self):
        """Polar very slowly degrading half-life is 70 years."""
        assert HALF_LIFE_VALUES[(ClimateZone.POLAR, "very_slowly_degrading")] == Decimal("70")

    def test_waste_degradability_class_all_categories(self):
        """WASTE_DEGRADABILITY_CLASS covers all 19 waste categories."""
        assert len(WASTE_DEGRADABILITY_CLASS) == 19
        for cat in WasteCategory:
            assert cat in WASTE_DEGRADABILITY_CLASS, f"Missing class for {cat.name}"

    def test_food_is_rapidly_degrading(self):
        """Food waste is classified as rapidly degrading."""
        assert WASTE_DEGRADABILITY_CLASS[WasteCategory.FOOD] == "rapidly_degrading"

    def test_plastic_is_very_slowly_degrading(self):
        """Plastic is classified as very slowly degrading."""
        assert WASTE_DEGRADABILITY_CLASS[WasteCategory.PLASTIC] == "very_slowly_degrading"

    def test_open_burning_ef_four_gases(self):
        """OPEN_BURNING_EF contains all 4 emission gases."""
        assert len(OPEN_BURNING_EF) == 4
        for gas in EmissionGas:
            assert gas in OPEN_BURNING_EF, f"Missing open burning EF for {gas.name}"

    def test_bmp_defaults_food(self):
        """BMP default for food waste is 400 m3 CH4/tonne VS."""
        assert BMP_DEFAULTS[WasteCategory.FOOD] == Decimal("400")

    def test_vs_fraction_food(self):
        """VS fraction for food waste is 0.87."""
        assert VS_FRACTION[WasteCategory.FOOD] == Decimal("0.87")

    def test_biogas_ch4_fraction_sludge(self):
        """Biogas CH4 fraction for sludge is 0.65."""
        assert BIOGAS_CH4_FRACTION[WasteCategory.SLUDGE] == Decimal("0.65")

    def test_advanced_thermal_ef_pyrolysis(self):
        """Advanced thermal EF for pyrolysis includes CO2_fossil."""
        assert "CO2_fossil" in ADVANCED_THERMAL_EF["pyrolysis"]
        assert ADVANCED_THERMAL_EF["pyrolysis"]["CO2_fossil"] == Decimal("500")

    def test_wastewater_bo_domestic_bod(self):
        """Domestic BOD Bo value is 0.6."""
        assert WASTEWATER_BO["domestic_bod"] == Decimal("0.6")

    def test_wastewater_bo_domestic_cod(self):
        """Domestic COD Bo value is 0.25."""
        assert WASTEWATER_BO["domestic_cod"] == Decimal("0.25")

    def test_wastewater_n2o_ef_plant(self):
        """Wastewater plant N2O EF is 0.016."""
        assert WASTEWATER_N2O_EF["plant_ef"] == Decimal("0.016")

    def test_wastewater_n2o_ef_effluent(self):
        """Wastewater effluent N2O EF is 0.005."""
        assert WASTEWATER_N2O_EF["effluent_ef"] == Decimal("0.005")
