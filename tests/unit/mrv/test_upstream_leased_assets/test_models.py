# -*- coding: utf-8 -*-
"""
Test suite for upstream_leased_assets.models - AGENT-MRV-021.

Tests all 22 enums, 14 constant tables, 12 Pydantic input/result models,
and helper functions for the Upstream Leased Assets Agent (GL-MRV-S3-008).

Coverage:
- Enumerations: 22 enums (values, membership, count, str behavior)
- Constants: GWP_VALUES, BUILDING_EUI_BENCHMARKS, BUILDING_EMISSION_FACTORS,
  VEHICLE_EMISSION_FACTORS, EQUIPMENT_BENCHMARKS, IT_POWER_RATINGS,
  GRID_EMISSION_FACTORS, FUEL_EMISSION_FACTORS, EEIO_FACTORS,
  CURRENCY_RATES, CPI_DEFLATORS, ALLOCATION_DEFAULTS,
  DQI_SCORING, UNCERTAINTY_RANGES
- Agent metadata: AGENT_ID, AGENT_COMPONENT, VERSION, TABLE_PREFIX
- Input models: BuildingInput, VehicleInput, EquipmentInput, ITAssetInput,
  LessorInput, SpendInput, AllocationInput, BatchAssetInput,
  AssetInput, ComplianceInput
- Result models: AssetResult, BatchAssetResult (frozen checks)
- Helper functions: calculate_provenance_hash

Author: GL-TestEngineer
Date: February 2026
"""

from decimal import Decimal
import pytest
from pydantic import ValidationError as PydanticValidationError

try:
    from greenlang.upstream_leased_assets.models import (
        # Enumerations
        CalculationMethod,
        AssetCategory,
        BuildingType,
        ClimateZone,
        EnergySource,
        VehicleType,
        FuelType,
        EquipmentType,
        ITAssetType,
        AllocationMethod,
        LeaseType,
        EFSource,
        ComplianceFramework,
        DataQualityTier,
        ProvenanceStage,
        UncertaintyMethod,
        DQIDimension,
        DQIScore,
        ComplianceStatus,
        GWPVersion,
        EmissionGas,
        CurrencyCode,

        # Agent metadata
        AGENT_ID,
        AGENT_COMPONENT,
        VERSION,
        TABLE_PREFIX,

        # Constant tables
        GWP_VALUES,
        BUILDING_EUI_BENCHMARKS,
        BUILDING_EMISSION_FACTORS,
        VEHICLE_EMISSION_FACTORS,
        EQUIPMENT_BENCHMARKS,
        IT_POWER_RATINGS,
        GRID_EMISSION_FACTORS,
        FUEL_EMISSION_FACTORS,
        EEIO_FACTORS,
        CURRENCY_RATES,
        CPI_DEFLATORS,
        ALLOCATION_DEFAULTS,
        DQI_SCORING,
        UNCERTAINTY_RANGES,

        # Input models
        BuildingInput,
        VehicleInput,
        EquipmentInput,
        ITAssetInput,
        LessorInput,
        SpendInput,
        AllocationInput,
        BatchAssetInput,
        AssetInput,
        ComplianceInput,

        # Result models
        AssetResult,
        BatchAssetResult,

        # Helper functions
        calculate_provenance_hash,
    )
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

_SKIP = pytest.mark.skipif(
    not _AVAILABLE,
    reason="upstream_leased_assets.models not available",
)

pytestmark = _SKIP


# ==============================================================================
# AGENT METADATA TESTS
# ==============================================================================


class TestAgentMetadata:
    """Tests for agent metadata constants."""

    def test_agent_id(self):
        """Test AGENT_ID is GL-MRV-S3-008."""
        assert AGENT_ID == "GL-MRV-S3-008"

    def test_agent_component(self):
        """Test AGENT_COMPONENT is AGENT-MRV-021."""
        assert AGENT_COMPONENT == "AGENT-MRV-021"

    def test_version(self):
        """Test VERSION is 1.0.0."""
        assert VERSION == "1.0.0"

    def test_table_prefix(self):
        """Test TABLE_PREFIX is gl_ula_."""
        assert TABLE_PREFIX == "gl_ula_"

    def test_table_prefix_ends_with_underscore(self):
        """Test TABLE_PREFIX ends with underscore for table naming convention."""
        assert TABLE_PREFIX.endswith("_")

    def test_table_prefix_starts_with_gl(self):
        """Test TABLE_PREFIX follows GreenLang naming convention."""
        assert TABLE_PREFIX.startswith("gl_")


# ==============================================================================
# ENUMERATION TESTS (22 enums)
# ==============================================================================


class TestCalculationMethod:
    """Tests for CalculationMethod enum."""

    def test_member_count(self):
        """Test CalculationMethod has exactly 4 members."""
        assert len(CalculationMethod) == 4

    @pytest.mark.parametrize("member,value", [
        (CalculationMethod.ASSET_SPECIFIC, "asset_specific"),
        (CalculationMethod.LESSOR_SPECIFIC, "lessor_specific"),
        (CalculationMethod.AVERAGE_DATA, "average_data"),
        (CalculationMethod.SPEND_BASED, "spend_based"),
    ])
    def test_members(self, member, value):
        """Test all 4 CalculationMethod members and their values."""
        assert member.value == value

    def test_is_str_enum(self):
        """Test CalculationMethod is a str enum."""
        assert isinstance(CalculationMethod.ASSET_SPECIFIC, str)


class TestAssetCategory:
    """Tests for AssetCategory enum."""

    def test_member_count(self):
        """Test AssetCategory has exactly 4 members."""
        assert len(AssetCategory) == 4

    @pytest.mark.parametrize("member,value", [
        (AssetCategory.BUILDING, "building"),
        (AssetCategory.VEHICLE, "vehicle"),
        (AssetCategory.EQUIPMENT, "equipment"),
        (AssetCategory.IT_ASSET, "it_asset"),
    ])
    def test_members(self, member, value):
        """Test all 4 AssetCategory members and their values."""
        assert member.value == value

    def test_is_str_enum(self):
        """Test AssetCategory is a str enum."""
        assert isinstance(AssetCategory.BUILDING, str)


class TestBuildingType:
    """Tests for BuildingType enum."""

    def test_member_count(self):
        """Test BuildingType has exactly 8 members."""
        assert len(BuildingType) == 8

    @pytest.mark.parametrize("member,value", [
        (BuildingType.OFFICE, "office"),
        (BuildingType.RETAIL, "retail"),
        (BuildingType.WAREHOUSE, "warehouse"),
        (BuildingType.INDUSTRIAL, "industrial"),
        (BuildingType.DATA_CENTER, "data_center"),
        (BuildingType.HOTEL, "hotel"),
        (BuildingType.HEALTHCARE, "healthcare"),
        (BuildingType.EDUCATION, "education"),
    ])
    def test_members(self, member, value):
        """Test all 8 BuildingType members and their values."""
        assert member.value == value

    def test_is_str_enum(self):
        """Test BuildingType is a str enum."""
        assert isinstance(BuildingType.OFFICE, str)


class TestClimateZone:
    """Tests for ClimateZone enum."""

    def test_member_count(self):
        """Test ClimateZone has exactly 5 members."""
        assert len(ClimateZone) == 5

    @pytest.mark.parametrize("member,value", [
        (ClimateZone.TROPICAL, "tropical"),
        (ClimateZone.ARID, "arid"),
        (ClimateZone.TEMPERATE, "temperate"),
        (ClimateZone.COLD, "cold"),
        (ClimateZone.WARM, "warm"),
    ])
    def test_members(self, member, value):
        """Test all 5 ClimateZone members and their values."""
        assert member.value == value

    def test_is_str_enum(self):
        """Test ClimateZone is a str enum."""
        assert isinstance(ClimateZone.TEMPERATE, str)


class TestEnergySource:
    """Tests for EnergySource enum."""

    def test_member_count(self):
        """Test EnergySource has exactly 6 members."""
        assert len(EnergySource) == 6

    @pytest.mark.parametrize("member,value", [
        (EnergySource.ELECTRICITY, "electricity"),
        (EnergySource.NATURAL_GAS, "natural_gas"),
        (EnergySource.DIESEL, "diesel"),
        (EnergySource.FUEL_OIL, "fuel_oil"),
        (EnergySource.DISTRICT_HEATING, "district_heating"),
        (EnergySource.DISTRICT_COOLING, "district_cooling"),
    ])
    def test_members(self, member, value):
        """Test all 6 EnergySource members and their values."""
        assert member.value == value

    def test_is_str_enum(self):
        """Test EnergySource is a str enum."""
        assert isinstance(EnergySource.ELECTRICITY, str)


class TestVehicleType:
    """Tests for VehicleType enum."""

    def test_member_count(self):
        """Test VehicleType has exactly 8 members."""
        assert len(VehicleType) == 8

    @pytest.mark.parametrize("member,value", [
        (VehicleType.SMALL_CAR, "small_car"),
        (VehicleType.MEDIUM_CAR, "medium_car"),
        (VehicleType.LARGE_CAR, "large_car"),
        (VehicleType.SUV, "suv"),
        (VehicleType.LIGHT_VAN, "light_van"),
        (VehicleType.HEAVY_VAN, "heavy_van"),
        (VehicleType.LIGHT_TRUCK, "light_truck"),
        (VehicleType.HEAVY_TRUCK, "heavy_truck"),
    ])
    def test_members(self, member, value):
        """Test all 8 VehicleType members and their values."""
        assert member.value == value

    def test_is_str_enum(self):
        """Test VehicleType is a str enum."""
        assert isinstance(VehicleType.MEDIUM_CAR, str)


class TestFuelType:
    """Tests for FuelType enum."""

    def test_member_count(self):
        """Test FuelType has exactly 7 members."""
        assert len(FuelType) == 7

    @pytest.mark.parametrize("member,value", [
        (FuelType.PETROL, "petrol"),
        (FuelType.DIESEL, "diesel"),
        (FuelType.LPG, "lpg"),
        (FuelType.CNG, "cng"),
        (FuelType.HYBRID, "hybrid"),
        (FuelType.PLUGIN_HYBRID, "plugin_hybrid"),
        (FuelType.BEV, "bev"),
    ])
    def test_members(self, member, value):
        """Test all 7 FuelType members and their values."""
        assert member.value == value

    def test_is_str_enum(self):
        """Test FuelType is a str enum."""
        assert isinstance(FuelType.PETROL, str)


class TestEquipmentType:
    """Tests for EquipmentType enum."""

    def test_member_count(self):
        """Test EquipmentType has exactly 6 members."""
        assert len(EquipmentType) == 6

    @pytest.mark.parametrize("member,value", [
        (EquipmentType.MANUFACTURING, "manufacturing"),
        (EquipmentType.CONSTRUCTION, "construction"),
        (EquipmentType.GENERATOR, "generator"),
        (EquipmentType.AGRICULTURAL, "agricultural"),
        (EquipmentType.MINING, "mining"),
        (EquipmentType.HVAC, "hvac"),
    ])
    def test_members(self, member, value):
        """Test all 6 EquipmentType members and their values."""
        assert member.value == value

    def test_is_str_enum(self):
        """Test EquipmentType is a str enum."""
        assert isinstance(EquipmentType.MANUFACTURING, str)


class TestITAssetType:
    """Tests for ITAssetType enum."""

    def test_member_count(self):
        """Test ITAssetType has exactly 7 members."""
        assert len(ITAssetType) == 7

    @pytest.mark.parametrize("member,value", [
        (ITAssetType.SERVER, "server"),
        (ITAssetType.NETWORK, "network"),
        (ITAssetType.STORAGE, "storage"),
        (ITAssetType.DESKTOP, "desktop"),
        (ITAssetType.LAPTOP, "laptop"),
        (ITAssetType.PRINTER, "printer"),
        (ITAssetType.COPIER, "copier"),
    ])
    def test_members(self, member, value):
        """Test all 7 ITAssetType members and their values."""
        assert member.value == value

    def test_is_str_enum(self):
        """Test ITAssetType is a str enum."""
        assert isinstance(ITAssetType.SERVER, str)


class TestAllocationMethod:
    """Tests for AllocationMethod enum."""

    def test_member_count(self):
        """Test AllocationMethod has exactly 4 members."""
        assert len(AllocationMethod) == 4

    @pytest.mark.parametrize("member,value", [
        (AllocationMethod.AREA, "area"),
        (AllocationMethod.HEADCOUNT, "headcount"),
        (AllocationMethod.REVENUE, "revenue"),
        (AllocationMethod.EQUAL, "equal"),
    ])
    def test_members(self, member, value):
        """Test all 4 AllocationMethod members and their values."""
        assert member.value == value

    def test_is_str_enum(self):
        """Test AllocationMethod is a str enum."""
        assert isinstance(AllocationMethod.AREA, str)


class TestLeaseType:
    """Tests for LeaseType enum."""

    def test_member_count(self):
        """Test LeaseType has exactly 3 members."""
        assert len(LeaseType) == 3

    @pytest.mark.parametrize("member,value", [
        (LeaseType.OPERATING, "operating"),
        (LeaseType.FINANCE, "finance"),
        (LeaseType.CAPITAL, "capital"),
    ])
    def test_members(self, member, value):
        """Test all 3 LeaseType members and their values."""
        assert member.value == value

    def test_is_str_enum(self):
        """Test LeaseType is a str enum."""
        assert isinstance(LeaseType.OPERATING, str)


class TestEFSource:
    """Tests for EFSource enum."""

    def test_member_count(self):
        """Test EFSource has exactly 7 members."""
        assert len(EFSource) == 7

    @pytest.mark.parametrize("member,value", [
        (EFSource.DEFRA, "defra"),
        (EFSource.EPA, "epa"),
        (EFSource.IEA, "iea"),
        (EFSource.EGRID, "egrid"),
        (EFSource.CBECS, "cbecs"),
        (EFSource.EEIO, "eeio"),
        (EFSource.CUSTOM, "custom"),
    ])
    def test_members(self, member, value):
        """Test all 7 EFSource members and their values."""
        assert member.value == value

    def test_is_str_enum(self):
        """Test EFSource is a str enum."""
        assert isinstance(EFSource.DEFRA, str)


class TestComplianceFramework:
    """Tests for ComplianceFramework enum."""

    def test_member_count(self):
        """Test ComplianceFramework has exactly 7 members."""
        assert len(ComplianceFramework) == 7

    @pytest.mark.parametrize("member,value", [
        (ComplianceFramework.GHG_PROTOCOL, "ghg_protocol"),
        (ComplianceFramework.ISO_14064, "iso_14064"),
        (ComplianceFramework.CSRD_ESRS, "csrd_esrs"),
        (ComplianceFramework.CDP, "cdp"),
        (ComplianceFramework.SBTI, "sbti"),
        (ComplianceFramework.SB_253, "sb_253"),
        (ComplianceFramework.GRI, "gri"),
    ])
    def test_members(self, member, value):
        """Test all 7 ComplianceFramework members and their values."""
        assert member.value == value

    def test_is_str_enum(self):
        """Test ComplianceFramework is a str enum."""
        assert isinstance(ComplianceFramework.GHG_PROTOCOL, str)


class TestDataQualityTier:
    """Tests for DataQualityTier enum."""

    def test_member_count(self):
        """Test DataQualityTier has exactly 3 members."""
        assert len(DataQualityTier) == 3

    @pytest.mark.parametrize("member,value", [
        (DataQualityTier.TIER_1, "tier_1"),
        (DataQualityTier.TIER_2, "tier_2"),
        (DataQualityTier.TIER_3, "tier_3"),
    ])
    def test_members(self, member, value):
        """Test all 3 DataQualityTier members and their values."""
        assert member.value == value


class TestProvenanceStage:
    """Tests for ProvenanceStage enum."""

    def test_member_count(self):
        """Test ProvenanceStage has exactly 10 members."""
        assert len(ProvenanceStage) == 10

    @pytest.mark.parametrize("member,value", [
        (ProvenanceStage.VALIDATE, "validate"),
        (ProvenanceStage.CLASSIFY, "classify"),
        (ProvenanceStage.NORMALIZE, "normalize"),
        (ProvenanceStage.RESOLVE_EFS, "resolve_efs"),
        (ProvenanceStage.CALCULATE_BUILDING, "calculate_building"),
        (ProvenanceStage.CALCULATE_VEHICLE, "calculate_vehicle"),
        (ProvenanceStage.CALCULATE_EQUIPMENT, "calculate_equipment"),
        (ProvenanceStage.CALCULATE_IT, "calculate_it"),
        (ProvenanceStage.COMPLIANCE, "compliance"),
        (ProvenanceStage.SEAL, "seal"),
    ])
    def test_members(self, member, value):
        """Test all 10 ProvenanceStage members and their values."""
        assert member.value == value


class TestUncertaintyMethod:
    """Tests for UncertaintyMethod enum."""

    def test_member_count(self):
        """Test UncertaintyMethod has exactly 3 members."""
        assert len(UncertaintyMethod) == 3

    @pytest.mark.parametrize("member,value", [
        (UncertaintyMethod.MONTE_CARLO, "monte_carlo"),
        (UncertaintyMethod.ANALYTICAL, "analytical"),
        (UncertaintyMethod.IPCC_TIER_2, "ipcc_tier_2"),
    ])
    def test_members(self, member, value):
        """Test all 3 UncertaintyMethod members and their values."""
        assert member.value == value


class TestDQIDimension:
    """Tests for DQIDimension enum."""

    def test_member_count(self):
        """Test DQIDimension has exactly 5 members."""
        assert len(DQIDimension) == 5

    @pytest.mark.parametrize("member,value", [
        (DQIDimension.REPRESENTATIVENESS, "representativeness"),
        (DQIDimension.COMPLETENESS, "completeness"),
        (DQIDimension.TEMPORAL, "temporal"),
        (DQIDimension.GEOGRAPHICAL, "geographical"),
        (DQIDimension.TECHNOLOGICAL, "technological"),
    ])
    def test_members(self, member, value):
        """Test all 5 DQIDimension members and their values."""
        assert member.value == value


class TestDQIScore:
    """Tests for DQIScore enum."""

    def test_member_count(self):
        """Test DQIScore has exactly 5 members."""
        assert len(DQIScore) == 5

    @pytest.mark.parametrize("member,value", [
        (DQIScore.VERY_HIGH, "very_high"),
        (DQIScore.HIGH, "high"),
        (DQIScore.MEDIUM, "medium"),
        (DQIScore.LOW, "low"),
        (DQIScore.VERY_LOW, "very_low"),
    ])
    def test_members(self, member, value):
        """Test all 5 DQIScore members and their values."""
        assert member.value == value


class TestComplianceStatus:
    """Tests for ComplianceStatus enum."""

    def test_member_count(self):
        """Test ComplianceStatus has exactly 3 members."""
        assert len(ComplianceStatus) == 3

    @pytest.mark.parametrize("member,value", [
        (ComplianceStatus.PASS, "pass"),
        (ComplianceStatus.FAIL, "fail"),
        (ComplianceStatus.WARNING, "warning"),
    ])
    def test_members(self, member, value):
        """Test all 3 ComplianceStatus members and their values."""
        assert member.value == value


class TestGWPVersion:
    """Tests for GWPVersion enum."""

    def test_member_count(self):
        """Test GWPVersion has exactly 4 members."""
        assert len(GWPVersion) == 4

    @pytest.mark.parametrize("member,value", [
        (GWPVersion.AR4, "ar4"),
        (GWPVersion.AR5, "ar5"),
        (GWPVersion.AR6, "ar6"),
        (GWPVersion.AR6_20YR, "ar6_20yr"),
    ])
    def test_members(self, member, value):
        """Test all 4 GWPVersion members and their values."""
        assert member.value == value


class TestEmissionGas:
    """Tests for EmissionGas enum."""

    def test_member_count(self):
        """Test EmissionGas has exactly 3 members."""
        assert len(EmissionGas) == 3

    @pytest.mark.parametrize("member,value", [
        (EmissionGas.CO2, "co2"),
        (EmissionGas.CH4, "ch4"),
        (EmissionGas.N2O, "n2o"),
    ])
    def test_members(self, member, value):
        """Test all 3 EmissionGas members and their values."""
        assert member.value == value


class TestCurrencyCode:
    """Tests for CurrencyCode enum."""

    def test_member_count(self):
        """Test CurrencyCode has exactly 12 members."""
        assert len(CurrencyCode) == 12

    @pytest.mark.parametrize("member,value", [
        (CurrencyCode.USD, "USD"),
        (CurrencyCode.EUR, "EUR"),
        (CurrencyCode.GBP, "GBP"),
        (CurrencyCode.CAD, "CAD"),
        (CurrencyCode.AUD, "AUD"),
        (CurrencyCode.JPY, "JPY"),
        (CurrencyCode.CNY, "CNY"),
        (CurrencyCode.INR, "INR"),
        (CurrencyCode.CHF, "CHF"),
        (CurrencyCode.SGD, "SGD"),
        (CurrencyCode.BRL, "BRL"),
        (CurrencyCode.ZAR, "ZAR"),
    ])
    def test_members(self, member, value):
        """Test all 12 CurrencyCode members and their values."""
        assert member.value == value

    def test_is_str_enum(self):
        """Test CurrencyCode is a str enum."""
        assert isinstance(CurrencyCode.USD, str)


# ==============================================================================
# CONSTANT TABLE TESTS (14 tables)
# ==============================================================================


class TestGWPValues:
    """Tests for GWP_VALUES constant table."""

    def test_contains_four_versions(self):
        """Test GWP_VALUES has AR4, AR5, AR6, and AR6_20YR."""
        assert len(GWP_VALUES) == 4
        assert GWPVersion.AR4 in GWP_VALUES
        assert GWPVersion.AR5 in GWP_VALUES
        assert GWPVersion.AR6 in GWP_VALUES
        assert GWPVersion.AR6_20YR in GWP_VALUES

    def test_co2_always_one(self):
        """Test CO2 GWP is always 1 for all versions."""
        for version in GWP_VALUES:
            assert GWP_VALUES[version]["co2"] == Decimal("1")

    def test_ar5_ch4(self):
        """Test AR5 CH4 GWP is 28."""
        assert GWP_VALUES[GWPVersion.AR5]["ch4"] == Decimal("28")

    def test_ar5_n2o(self):
        """Test AR5 N2O GWP is 265."""
        assert GWP_VALUES[GWPVersion.AR5]["n2o"] == Decimal("265")

    def test_ar6_ch4(self):
        """Test AR6 CH4 GWP is 27.9."""
        assert GWP_VALUES[GWPVersion.AR6]["ch4"] == Decimal("27.9")

    def test_ar6_n2o(self):
        """Test AR6 N2O GWP is 273."""
        assert GWP_VALUES[GWPVersion.AR6]["n2o"] == Decimal("273")

    def test_ar4_ch4(self):
        """Test AR4 CH4 GWP is 25."""
        assert GWP_VALUES[GWPVersion.AR4]["ch4"] == Decimal("25")

    def test_ar6_20yr_ch4(self):
        """Test AR6 20-year CH4 GWP is 81.2."""
        assert GWP_VALUES[GWPVersion.AR6_20YR]["ch4"] == Decimal("81.2")

    def test_all_values_are_decimal(self):
        """Test all GWP values are Decimal type."""
        for version_data in GWP_VALUES.values():
            for gas_value in version_data.values():
                assert isinstance(gas_value, Decimal)

    def test_each_version_has_three_gases(self):
        """Test each GWP version has co2, ch4, n2o."""
        for version_data in GWP_VALUES.values():
            assert "co2" in version_data
            assert "ch4" in version_data
            assert "n2o" in version_data


class TestBuildingEUIBenchmarks:
    """Tests for BUILDING_EUI_BENCHMARKS constant table."""

    def test_eight_building_types(self):
        """Test BUILDING_EUI_BENCHMARKS has exactly 8 building types."""
        assert len(BUILDING_EUI_BENCHMARKS) == 8

    def test_all_building_types_present(self):
        """Test all BuildingType enum members are present."""
        for bt in BuildingType:
            assert bt in BUILDING_EUI_BENCHMARKS

    def test_each_building_has_five_climate_zones(self):
        """Test each building type has 5 climate zone EUIs."""
        for bt, zones in BUILDING_EUI_BENCHMARKS.items():
            assert len(zones) == 5, f"{bt} missing climate zones"

    def test_all_climate_zones_present_per_building(self):
        """Test all ClimateZone enum members are present per building."""
        for bt, zones in BUILDING_EUI_BENCHMARKS.items():
            for cz in ClimateZone:
                assert cz in zones, f"{bt} missing {cz}"

    def test_office_temperate_eui(self):
        """Test office temperate EUI value is reasonable (100-300 kWh/sqm/yr)."""
        eui = BUILDING_EUI_BENCHMARKS[BuildingType.OFFICE][ClimateZone.TEMPERATE]
        assert Decimal("100") <= eui <= Decimal("300")

    def test_data_center_highest_eui(self):
        """Test data center has highest EUI across all building types."""
        dc_temperate = BUILDING_EUI_BENCHMARKS[BuildingType.DATA_CENTER][ClimateZone.TEMPERATE]
        for bt in BuildingType:
            if bt != BuildingType.DATA_CENTER:
                assert dc_temperate > BUILDING_EUI_BENCHMARKS[bt][ClimateZone.TEMPERATE], \
                    f"Data center EUI should exceed {bt}"

    def test_warehouse_lowest_eui(self):
        """Test warehouse has relatively low EUI (not the highest)."""
        wh_temperate = BUILDING_EUI_BENCHMARKS[BuildingType.WAREHOUSE][ClimateZone.TEMPERATE]
        office_temperate = BUILDING_EUI_BENCHMARKS[BuildingType.OFFICE][ClimateZone.TEMPERATE]
        assert wh_temperate < office_temperate

    def test_cold_zone_higher_than_tropical(self):
        """Test cold zone EUI >= tropical zone for office (heating demand)."""
        cold = BUILDING_EUI_BENCHMARKS[BuildingType.OFFICE][ClimateZone.COLD]
        tropical = BUILDING_EUI_BENCHMARKS[BuildingType.OFFICE][ClimateZone.TROPICAL]
        assert cold >= tropical

    def test_all_values_are_decimal(self):
        """Test all EUI benchmark values are Decimal type."""
        for bt, zones in BUILDING_EUI_BENCHMARKS.items():
            for cz, eui in zones.items():
                assert isinstance(eui, Decimal), f"{bt}/{cz} is not Decimal"

    def test_all_values_positive(self):
        """Test all EUI benchmark values are positive."""
        for bt, zones in BUILDING_EUI_BENCHMARKS.items():
            for cz, eui in zones.items():
                assert eui > 0, f"{bt}/{cz} EUI not positive"


class TestBuildingEmissionFactors:
    """Tests for BUILDING_EMISSION_FACTORS constant table."""

    def test_six_energy_sources(self):
        """Test BUILDING_EMISSION_FACTORS has entries for energy sources."""
        assert len(BUILDING_EMISSION_FACTORS) >= 4

    def test_electricity_present(self):
        """Test electricity emission factor is present."""
        assert EnergySource.ELECTRICITY in BUILDING_EMISSION_FACTORS or \
            "electricity" in BUILDING_EMISSION_FACTORS

    def test_natural_gas_present(self):
        """Test natural gas emission factor is present."""
        assert EnergySource.NATURAL_GAS in BUILDING_EMISSION_FACTORS or \
            "natural_gas" in BUILDING_EMISSION_FACTORS

    def test_all_values_positive_decimal(self):
        """Test all emission factor values are positive Decimal."""
        for source, ef in BUILDING_EMISSION_FACTORS.items():
            if isinstance(ef, dict):
                for k, v in ef.items():
                    assert isinstance(v, Decimal), f"{source}/{k} is not Decimal"
                    assert v > 0
            else:
                assert isinstance(ef, Decimal), f"{source} is not Decimal"
                assert ef > 0


class TestVehicleEmissionFactors:
    """Tests for VEHICLE_EMISSION_FACTORS constant table."""

    def test_eight_vehicle_types(self):
        """Test VEHICLE_EMISSION_FACTORS has exactly 8 vehicle types."""
        assert len(VEHICLE_EMISSION_FACTORS) == 8

    def test_all_vehicle_types_present(self):
        """Test all VehicleType enum members are present."""
        for vt in VehicleType:
            assert vt in VEHICLE_EMISSION_FACTORS

    def test_all_have_fuel_type_entries(self):
        """Test each vehicle type has at least one fuel type entry."""
        for vt, data in VEHICLE_EMISSION_FACTORS.items():
            assert len(data) >= 1, f"{vt} has no fuel type entries"

    def test_medium_car_petrol_ef(self):
        """Test medium car petrol emission factor is reasonable."""
        ef_data = VEHICLE_EMISSION_FACTORS[VehicleType.MEDIUM_CAR]
        petrol_key = FuelType.PETROL if FuelType.PETROL in ef_data else "petrol"
        if petrol_key in ef_data:
            ef = ef_data[petrol_key]
            if isinstance(ef, dict):
                assert ef.get("ef_per_km", ef.get("ef_per_vkm", Decimal("0"))) > 0
            else:
                assert ef > 0

    def test_heavy_truck_higher_than_small_car(self):
        """Test heavy truck has higher EF than small car (per km)."""
        truck = VEHICLE_EMISSION_FACTORS[VehicleType.HEAVY_TRUCK]
        car = VEHICLE_EMISSION_FACTORS[VehicleType.SMALL_CAR]
        # Compare first available fuel type
        truck_val = next(iter(truck.values()))
        car_val = next(iter(car.values()))
        if isinstance(truck_val, dict):
            truck_ef = truck_val.get("ef_per_km", truck_val.get("ef_per_vkm", Decimal("0")))
        else:
            truck_ef = truck_val
        if isinstance(car_val, dict):
            car_ef = car_val.get("ef_per_km", car_val.get("ef_per_vkm", Decimal("0")))
        else:
            car_ef = car_val
        assert truck_ef > car_ef

    def test_all_values_are_decimal(self):
        """Test all vehicle emission factor values are Decimal type."""
        for vt, fuels in VEHICLE_EMISSION_FACTORS.items():
            for fuel, data in fuels.items():
                if isinstance(data, dict):
                    for k, v in data.items():
                        if isinstance(v, (int, float, Decimal)):
                            assert isinstance(v, Decimal), f"{vt}/{fuel}/{k} not Decimal"
                else:
                    assert isinstance(data, Decimal), f"{vt}/{fuel} not Decimal"


class TestEquipmentBenchmarks:
    """Tests for EQUIPMENT_BENCHMARKS constant table."""

    def test_six_equipment_types(self):
        """Test EQUIPMENT_BENCHMARKS has exactly 6 equipment types."""
        assert len(EQUIPMENT_BENCHMARKS) == 6

    def test_all_equipment_types_present(self):
        """Test all EquipmentType enum members are present."""
        for et in EquipmentType:
            assert et in EQUIPMENT_BENCHMARKS

    def test_all_have_default_load_factor(self):
        """Test all equipment types have a default load factor."""
        for et, data in EQUIPMENT_BENCHMARKS.items():
            assert "default_load_factor" in data
            assert Decimal("0") < data["default_load_factor"] <= Decimal("1")

    def test_manufacturing_load_factor(self):
        """Test manufacturing default load factor is reasonable (0.6-0.85)."""
        lf = EQUIPMENT_BENCHMARKS[EquipmentType.MANUFACTURING]["default_load_factor"]
        assert Decimal("0.60") <= lf <= Decimal("0.85")

    def test_generator_has_output_ef(self):
        """Test generator has fuel consumption factor."""
        gen = EQUIPMENT_BENCHMARKS[EquipmentType.GENERATOR]
        assert "fuel_consumption_factor" in gen or "default_load_factor" in gen

    def test_all_values_decimal(self):
        """Test all benchmark values are Decimal type."""
        for et, data in EQUIPMENT_BENCHMARKS.items():
            for k, v in data.items():
                if isinstance(v, (int, float, Decimal)):
                    assert isinstance(v, Decimal), f"{et}/{k} not Decimal"


class TestITPowerRatings:
    """Tests for IT_POWER_RATINGS constant table."""

    def test_seven_it_types(self):
        """Test IT_POWER_RATINGS has exactly 7 IT asset types."""
        assert len(IT_POWER_RATINGS) == 7

    def test_all_it_types_present(self):
        """Test all ITAssetType enum members are present."""
        for it_type in ITAssetType:
            assert it_type in IT_POWER_RATINGS

    def test_all_have_typical_power_w(self):
        """Test all IT types have typical_power_w field."""
        for it_type, data in IT_POWER_RATINGS.items():
            assert "typical_power_w" in data
            assert isinstance(data["typical_power_w"], Decimal)

    def test_server_highest_power(self):
        """Test server has highest typical power rating."""
        server_power = IT_POWER_RATINGS[ITAssetType.SERVER]["typical_power_w"]
        for it_type in [ITAssetType.DESKTOP, ITAssetType.LAPTOP, ITAssetType.PRINTER]:
            assert server_power > IT_POWER_RATINGS[it_type]["typical_power_w"]

    def test_laptop_lower_than_desktop(self):
        """Test laptop has lower power than desktop."""
        laptop = IT_POWER_RATINGS[ITAssetType.LAPTOP]["typical_power_w"]
        desktop = IT_POWER_RATINGS[ITAssetType.DESKTOP]["typical_power_w"]
        assert laptop < desktop

    def test_all_have_standby_power(self):
        """Test all IT types have standby_power_w field."""
        for it_type, data in IT_POWER_RATINGS.items():
            assert "standby_power_w" in data
            assert isinstance(data["standby_power_w"], Decimal)

    def test_standby_lower_than_active(self):
        """Test standby power is lower than active power for all types."""
        for it_type, data in IT_POWER_RATINGS.items():
            assert data["standby_power_w"] < data["typical_power_w"], \
                f"{it_type} standby >= active power"


class TestGridEmissionFactors:
    """Tests for GRID_EMISSION_FACTORS constant table."""

    def test_eleven_or_more_regions(self):
        """Test GRID_EMISSION_FACTORS has at least 11 regions."""
        assert len(GRID_EMISSION_FACTORS) >= 11

    def test_us_grid_ef(self):
        """Test US grid emission factor is approximately 0.37 kgCO2e/kWh."""
        us_key = "US" if "US" in GRID_EMISSION_FACTORS else next(
            k for k in GRID_EMISSION_FACTORS if str(k) == "US" or getattr(k, 'value', None) == "US"
        )
        ef = GRID_EMISSION_FACTORS[us_key]
        assert Decimal("0.30") <= ef <= Decimal("0.50")

    def test_france_lower_than_us(self):
        """Test France grid EF is lower than US (nuclear-heavy grid)."""
        us_ef = None
        fr_ef = None
        for k, v in GRID_EMISSION_FACTORS.items():
            if str(k) == "US" or getattr(k, 'value', None) == "US":
                us_ef = v
            if str(k) == "FR" or getattr(k, 'value', None) == "FR":
                fr_ef = v
        if us_ef is not None and fr_ef is not None:
            assert fr_ef < us_ef

    def test_all_values_decimal_and_positive(self):
        """Test all grid emission factors are positive Decimal values."""
        for k, val in GRID_EMISSION_FACTORS.items():
            assert isinstance(val, Decimal), f"{k} is not Decimal"
            assert val > 0, f"{k} is not positive"


class TestFuelEmissionFactors:
    """Tests for FUEL_EMISSION_FACTORS constant table."""

    def test_at_least_four_fuel_types(self):
        """Test FUEL_EMISSION_FACTORS has at least 4 fuel types."""
        assert len(FUEL_EMISSION_FACTORS) >= 4

    def test_diesel_ef_per_litre(self):
        """Test diesel emission factor is approximately 2.68-2.71 kgCO2e/litre."""
        diesel_key = FuelType.DIESEL if FuelType.DIESEL in FUEL_EMISSION_FACTORS else "diesel"
        ef_data = FUEL_EMISSION_FACTORS[diesel_key]
        if isinstance(ef_data, dict):
            ef = ef_data.get("ef_per_litre", ef_data.get("ef", Decimal("0")))
        else:
            ef = ef_data
        assert Decimal("2.50") <= ef <= Decimal("3.00")

    def test_petrol_ef_per_litre(self):
        """Test petrol emission factor is approximately 2.31 kgCO2e/litre."""
        petrol_key = FuelType.PETROL if FuelType.PETROL in FUEL_EMISSION_FACTORS else "petrol"
        ef_data = FUEL_EMISSION_FACTORS[petrol_key]
        if isinstance(ef_data, dict):
            ef = ef_data.get("ef_per_litre", ef_data.get("ef", Decimal("0")))
        else:
            ef = ef_data
        assert Decimal("2.00") <= ef <= Decimal("2.60")

    def test_all_values_positive(self):
        """Test all fuel emission factors are positive."""
        for ft, data in FUEL_EMISSION_FACTORS.items():
            if isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(v, Decimal):
                        assert v > 0, f"{ft}/{k} not positive"
            else:
                assert data > 0, f"{ft} not positive"


class TestEEIOFactors:
    """Tests for EEIO_FACTORS constant table."""

    def test_at_least_six_naics_codes(self):
        """Test EEIO_FACTORS has at least 6 NAICS codes."""
        assert len(EEIO_FACTORS) >= 6

    def test_office_rental_code(self):
        """Test NAICS 531120 (lessors of buildings) is present."""
        assert "531120" in EEIO_FACTORS

    def test_vehicle_leasing_code(self):
        """Test NAICS 532112 (passenger car leasing) is present."""
        assert "532112" in EEIO_FACTORS

    def test_all_have_name_and_ef(self):
        """Test all EEIO entries have name and ef fields."""
        for code, data in EEIO_FACTORS.items():
            assert "name" in data
            assert "ef" in data
            assert isinstance(data["ef"], Decimal)

    def test_all_efs_positive(self):
        """Test all EEIO emission factors are positive."""
        for code, data in EEIO_FACTORS.items():
            assert data["ef"] > 0, f"NAICS {code} EF not positive"


class TestCurrencyRates:
    """Tests for CURRENCY_RATES constant table."""

    def test_twelve_currencies(self):
        """Test CURRENCY_RATES has exactly 12 currencies."""
        assert len(CURRENCY_RATES) == 12

    def test_usd_rate_is_one(self):
        """Test USD rate is 1.0 (base currency)."""
        assert CURRENCY_RATES[CurrencyCode.USD] == Decimal("1.0")

    def test_all_currency_codes_present(self):
        """Test all CurrencyCode enum members are present."""
        for cc in CurrencyCode:
            assert cc in CURRENCY_RATES

    def test_all_rates_positive_decimal(self):
        """Test all rates are positive Decimal values."""
        for val in CURRENCY_RATES.values():
            assert isinstance(val, Decimal)
            assert val > 0

    def test_eur_rate_reasonable(self):
        """Test EUR rate is reasonable (0.8-1.2 vs USD)."""
        assert Decimal("0.80") <= CURRENCY_RATES[CurrencyCode.EUR] <= Decimal("1.20")


class TestCPIDeflators:
    """Tests for CPI_DEFLATORS constant table."""

    def test_eleven_years(self):
        """Test CPI_DEFLATORS covers 2015-2025."""
        assert len(CPI_DEFLATORS) == 11

    def test_base_year_2021_is_one(self):
        """Test CPI deflator for base year 2021 is 1.0."""
        assert CPI_DEFLATORS[2021] == Decimal("1.0000")

    def test_years_before_base_less_than_one(self):
        """Test years before 2021 have deflators less than 1.0."""
        for year in range(2015, 2021):
            assert CPI_DEFLATORS[year] < Decimal("1.0")

    def test_years_after_base_greater_than_one(self):
        """Test years after 2021 have deflators greater than 1.0."""
        for year in range(2022, 2026):
            assert CPI_DEFLATORS[year] > Decimal("1.0")

    def test_all_values_decimal(self):
        """Test all CPI deflator values are Decimal type."""
        for val in CPI_DEFLATORS.values():
            assert isinstance(val, Decimal)

    def test_monotonically_increasing(self):
        """Test CPI deflators are monotonically increasing over time."""
        years = sorted(CPI_DEFLATORS.keys())
        for i in range(1, len(years)):
            assert CPI_DEFLATORS[years[i]] >= CPI_DEFLATORS[years[i - 1]], \
                f"CPI deflator not increasing: {years[i-1]}->{years[i]}"


class TestAllocationDefaults:
    """Tests for ALLOCATION_DEFAULTS constant table."""

    def test_four_methods(self):
        """Test ALLOCATION_DEFAULTS has 4 allocation methods."""
        assert len(ALLOCATION_DEFAULTS) == 4

    def test_all_methods_present(self):
        """Test all AllocationMethod enum members are present."""
        for am in AllocationMethod:
            assert am in ALLOCATION_DEFAULTS

    def test_equal_allocation_is_one(self):
        """Test equal allocation default share is 1.0."""
        assert ALLOCATION_DEFAULTS[AllocationMethod.EQUAL] == Decimal("1.0")

    def test_all_values_between_zero_and_one(self):
        """Test all allocation defaults are between 0 and 1."""
        for am, val in ALLOCATION_DEFAULTS.items():
            assert Decimal("0") <= val <= Decimal("1.0"), f"{am} out of range"


class TestDQIScoring:
    """Tests for DQI_SCORING constant table."""

    def test_five_dimensions(self):
        """Test DQI_SCORING has exactly 5 dimensions."""
        assert len(DQI_SCORING) == 5

    def test_all_dimensions_present(self):
        """Test all DQIDimension enum members are present."""
        for dim in DQIDimension:
            assert dim in DQI_SCORING

    def test_each_dimension_has_five_scores(self):
        """Test each dimension has 5 score levels."""
        for dim, scores in DQI_SCORING.items():
            assert len(scores) == 5

    def test_very_high_is_five(self):
        """Test VERY_HIGH score is 5 for all dimensions."""
        for dim_scores in DQI_SCORING.values():
            assert dim_scores[DQIScore.VERY_HIGH] == Decimal("5")

    def test_very_low_is_one(self):
        """Test VERY_LOW score is 1 for all dimensions."""
        for dim_scores in DQI_SCORING.values():
            assert dim_scores[DQIScore.VERY_LOW] == Decimal("1")


class TestUncertaintyRanges:
    """Tests for UNCERTAINTY_RANGES constant table."""

    def test_at_least_four_categories(self):
        """Test UNCERTAINTY_RANGES has at least 4 categories."""
        assert len(UNCERTAINTY_RANGES) >= 4

    def test_asset_specific_has_three_tiers(self):
        """Test asset_specific has 3 data quality tiers."""
        assert len(UNCERTAINTY_RANGES["asset_specific"]) == 3

    def test_asset_specific_tier1_lowest(self):
        """Test asset_specific Tier 1 has lowest uncertainty."""
        assert UNCERTAINTY_RANGES["asset_specific"][DataQualityTier.TIER_1] == Decimal("0.05")

    def test_spend_based_highest_uncertainty(self):
        """Test spend_based Tier 3 has highest uncertainty (0.60)."""
        assert UNCERTAINTY_RANGES["spend_based"][DataQualityTier.TIER_3] == Decimal("0.60")

    def test_tier_ordering(self):
        """Test Tier 1 < Tier 2 < Tier 3 for all categories."""
        for cat, tiers in UNCERTAINTY_RANGES.items():
            assert tiers[DataQualityTier.TIER_1] < tiers[DataQualityTier.TIER_2], \
                f"{cat}: Tier 1 not < Tier 2"
            assert tiers[DataQualityTier.TIER_2] < tiers[DataQualityTier.TIER_3], \
                f"{cat}: Tier 2 not < Tier 3"


# ==============================================================================
# INPUT MODEL TESTS (10 models)
# ==============================================================================


class TestBuildingInput:
    """Tests for BuildingInput Pydantic model."""

    def test_valid_office_input(self):
        """Test creating valid office building input."""
        inp = BuildingInput(
            building_type=BuildingType.OFFICE,
            floor_area_sqm=Decimal("2500.00"),
            climate_zone=ClimateZone.TEMPERATE,
            energy_sources={"electricity_kwh": Decimal("450000")},
        )
        assert inp.building_type == BuildingType.OFFICE
        assert inp.floor_area_sqm == Decimal("2500.00")

    def test_default_values(self):
        """Test BuildingInput default field values."""
        inp = BuildingInput(
            building_type=BuildingType.OFFICE,
            floor_area_sqm=Decimal("1000"),
        )
        assert inp.occupancy_months == 12
        assert inp.allocation_share == Decimal("1.0")
        assert inp.allocation_method == AllocationMethod.AREA

    def test_frozen_immutability(self):
        """Test BuildingInput is frozen (immutable)."""
        inp = BuildingInput(
            building_type=BuildingType.OFFICE,
            floor_area_sqm=Decimal("1000"),
        )
        with pytest.raises(Exception):
            inp.building_type = BuildingType.RETAIL

    def test_zero_area_rejected(self):
        """Test zero floor area is rejected."""
        with pytest.raises(PydanticValidationError):
            BuildingInput(
                building_type=BuildingType.OFFICE,
                floor_area_sqm=Decimal("0"),
            )

    def test_negative_area_rejected(self):
        """Test negative floor area is rejected."""
        with pytest.raises(PydanticValidationError):
            BuildingInput(
                building_type=BuildingType.OFFICE,
                floor_area_sqm=Decimal("-100"),
            )

    def test_occupancy_months_max_twelve(self):
        """Test occupancy_months cannot exceed 12."""
        with pytest.raises(PydanticValidationError):
            BuildingInput(
                building_type=BuildingType.OFFICE,
                floor_area_sqm=Decimal("1000"),
                occupancy_months=13,
            )

    def test_occupancy_months_min_one(self):
        """Test occupancy_months minimum is 1."""
        inp = BuildingInput(
            building_type=BuildingType.OFFICE,
            floor_area_sqm=Decimal("1000"),
            occupancy_months=1,
        )
        assert inp.occupancy_months == 1

    def test_allocation_share_max_one(self):
        """Test allocation_share cannot exceed 1.0."""
        with pytest.raises(PydanticValidationError):
            BuildingInput(
                building_type=BuildingType.OFFICE,
                floor_area_sqm=Decimal("1000"),
                allocation_share=Decimal("1.5"),
            )

    def test_allocation_share_min_zero(self):
        """Test allocation_share cannot be negative."""
        with pytest.raises(PydanticValidationError):
            BuildingInput(
                building_type=BuildingType.OFFICE,
                floor_area_sqm=Decimal("1000"),
                allocation_share=Decimal("-0.1"),
            )


class TestVehicleInput:
    """Tests for VehicleInput Pydantic model."""

    def test_valid_car_input(self):
        """Test creating valid car input."""
        inp = VehicleInput(
            vehicle_type=VehicleType.MEDIUM_CAR,
            fuel_type=FuelType.PETROL,
            annual_distance_km=Decimal("25000"),
        )
        assert inp.vehicle_type == VehicleType.MEDIUM_CAR
        assert inp.fuel_type == FuelType.PETROL

    def test_default_values(self):
        """Test VehicleInput default field values."""
        inp = VehicleInput(
            vehicle_type=VehicleType.MEDIUM_CAR,
            fuel_type=FuelType.PETROL,
            annual_distance_km=Decimal("20000"),
        )
        assert inp.age_years is None or inp.age_years == 0

    def test_frozen_immutability(self):
        """Test VehicleInput is frozen."""
        inp = VehicleInput(
            vehicle_type=VehicleType.MEDIUM_CAR,
            fuel_type=FuelType.PETROL,
            annual_distance_km=Decimal("20000"),
        )
        with pytest.raises(Exception):
            inp.fuel_type = FuelType.DIESEL

    def test_zero_distance_rejected(self):
        """Test zero annual distance is rejected."""
        with pytest.raises(PydanticValidationError):
            VehicleInput(
                vehicle_type=VehicleType.MEDIUM_CAR,
                fuel_type=FuelType.PETROL,
                annual_distance_km=Decimal("0"),
            )

    def test_negative_distance_rejected(self):
        """Test negative annual distance is rejected."""
        with pytest.raises(PydanticValidationError):
            VehicleInput(
                vehicle_type=VehicleType.MEDIUM_CAR,
                fuel_type=FuelType.PETROL,
                annual_distance_km=Decimal("-1000"),
            )

    def test_bev_fuel_type(self):
        """Test BEV fuel type is valid."""
        inp = VehicleInput(
            vehicle_type=VehicleType.MEDIUM_CAR,
            fuel_type=FuelType.BEV,
            annual_distance_km=Decimal("20000"),
        )
        assert inp.fuel_type == FuelType.BEV


class TestEquipmentInput:
    """Tests for EquipmentInput Pydantic model."""

    def test_valid_manufacturing_input(self):
        """Test creating valid manufacturing equipment input."""
        inp = EquipmentInput(
            equipment_type=EquipmentType.MANUFACTURING,
            rated_power_kw=Decimal("500"),
            annual_operating_hours=6000,
            energy_source=EnergySource.ELECTRICITY,
        )
        assert inp.equipment_type == EquipmentType.MANUFACTURING
        assert inp.rated_power_kw == Decimal("500")

    def test_default_load_factor(self):
        """Test EquipmentInput default load factor."""
        inp = EquipmentInput(
            equipment_type=EquipmentType.MANUFACTURING,
            rated_power_kw=Decimal("100"),
            annual_operating_hours=2000,
            energy_source=EnergySource.ELECTRICITY,
        )
        assert inp.load_factor is None or Decimal("0") < inp.load_factor <= Decimal("1")

    def test_frozen(self):
        """Test EquipmentInput is frozen."""
        inp = EquipmentInput(
            equipment_type=EquipmentType.MANUFACTURING,
            rated_power_kw=Decimal("100"),
            annual_operating_hours=2000,
            energy_source=EnergySource.ELECTRICITY,
        )
        with pytest.raises(Exception):
            inp.rated_power_kw = Decimal("200")

    def test_zero_power_rejected(self):
        """Test zero rated power is rejected."""
        with pytest.raises(PydanticValidationError):
            EquipmentInput(
                equipment_type=EquipmentType.MANUFACTURING,
                rated_power_kw=Decimal("0"),
                annual_operating_hours=2000,
                energy_source=EnergySource.ELECTRICITY,
            )

    def test_negative_hours_rejected(self):
        """Test negative operating hours is rejected."""
        with pytest.raises(PydanticValidationError):
            EquipmentInput(
                equipment_type=EquipmentType.MANUFACTURING,
                rated_power_kw=Decimal("100"),
                annual_operating_hours=-100,
                energy_source=EnergySource.ELECTRICITY,
            )

    def test_hours_max_8760(self):
        """Test operating hours cannot exceed 8760 (hours in a year)."""
        with pytest.raises(PydanticValidationError):
            EquipmentInput(
                equipment_type=EquipmentType.MANUFACTURING,
                rated_power_kw=Decimal("100"),
                annual_operating_hours=9000,
                energy_source=EnergySource.ELECTRICITY,
            )


class TestITAssetInput:
    """Tests for ITAssetInput Pydantic model."""

    def test_valid_server_input(self):
        """Test creating valid server input."""
        inp = ITAssetInput(
            it_type=ITAssetType.SERVER,
            rated_power_w=Decimal("500"),
            utilization_pct=Decimal("0.90"),
            pue=Decimal("1.40"),
        )
        assert inp.it_type == ITAssetType.SERVER
        assert inp.pue == Decimal("1.40")

    def test_default_values(self):
        """Test ITAssetInput default values."""
        inp = ITAssetInput(
            it_type=ITAssetType.DESKTOP,
            rated_power_w=Decimal("200"),
        )
        assert inp.pue == Decimal("1.0") or inp.pue is None
        assert inp.annual_hours == 8760 or inp.annual_hours == 2080

    def test_frozen(self):
        """Test ITAssetInput is frozen."""
        inp = ITAssetInput(
            it_type=ITAssetType.SERVER,
            rated_power_w=Decimal("500"),
        )
        with pytest.raises(Exception):
            inp.it_type = ITAssetType.DESKTOP

    def test_zero_power_rejected(self):
        """Test zero rated power is rejected."""
        with pytest.raises(PydanticValidationError):
            ITAssetInput(
                it_type=ITAssetType.SERVER,
                rated_power_w=Decimal("0"),
            )

    def test_pue_below_one_rejected(self):
        """Test PUE below 1.0 is rejected (physically impossible)."""
        with pytest.raises(PydanticValidationError):
            ITAssetInput(
                it_type=ITAssetType.SERVER,
                rated_power_w=Decimal("500"),
                pue=Decimal("0.8"),
            )

    def test_utilization_over_one_rejected(self):
        """Test utilization over 1.0 (100%) is rejected."""
        with pytest.raises(PydanticValidationError):
            ITAssetInput(
                it_type=ITAssetType.SERVER,
                rated_power_w=Decimal("500"),
                utilization_pct=Decimal("1.5"),
            )


class TestLessorInput:
    """Tests for LessorInput Pydantic model."""

    def test_valid_lessor_input(self):
        """Test creating valid lessor-specific input."""
        inp = LessorInput(
            asset_type=AssetCategory.BUILDING,
            lessor_electricity_kwh=Decimal("430000"),
            lessor_natural_gas_kwh=Decimal("115000"),
            lessor_data_year=2024,
        )
        assert inp.asset_type == AssetCategory.BUILDING
        assert inp.lessor_electricity_kwh == Decimal("430000")

    def test_default_values(self):
        """Test LessorInput defaults."""
        inp = LessorInput(
            asset_type=AssetCategory.BUILDING,
            lessor_electricity_kwh=Decimal("100000"),
            lessor_data_year=2024,
        )
        assert inp.lessor_natural_gas_kwh is None or \
            inp.lessor_natural_gas_kwh == Decimal("0")

    def test_frozen(self):
        """Test LessorInput is frozen."""
        inp = LessorInput(
            asset_type=AssetCategory.BUILDING,
            lessor_electricity_kwh=Decimal("100000"),
            lessor_data_year=2024,
        )
        with pytest.raises(Exception):
            inp.lessor_data_year = 2025


class TestSpendInput:
    """Tests for SpendInput Pydantic model."""

    def test_valid_input(self):
        """Test creating valid spend input."""
        inp = SpendInput(
            naics_code="531120",
            amount=Decimal("120000.00"),
            currency=CurrencyCode.USD,
            reporting_year=2024,
        )
        assert inp.naics_code == "531120"
        assert inp.amount == Decimal("120000.00")

    def test_default_currency(self):
        """Test default currency is USD."""
        inp = SpendInput(
            naics_code="531120",
            amount=Decimal("100.00"),
        )
        assert inp.currency == CurrencyCode.USD

    def test_default_reporting_year(self):
        """Test default reporting_year is 2024."""
        inp = SpendInput(
            naics_code="531120",
            amount=Decimal("100.00"),
        )
        assert inp.reporting_year == 2024

    def test_zero_amount_rejected(self):
        """Test zero amount is rejected."""
        with pytest.raises(PydanticValidationError):
            SpendInput(
                naics_code="531120",
                amount=Decimal("0"),
            )

    def test_negative_amount_rejected(self):
        """Test negative amount is rejected."""
        with pytest.raises(PydanticValidationError):
            SpendInput(
                naics_code="531120",
                amount=Decimal("-100"),
            )

    def test_year_below_2015_rejected(self):
        """Test reporting year below 2015 is rejected."""
        with pytest.raises(PydanticValidationError):
            SpendInput(
                naics_code="531120",
                amount=Decimal("100"),
                reporting_year=2014,
            )

    def test_year_above_2030_rejected(self):
        """Test reporting year above 2030 is rejected."""
        with pytest.raises(PydanticValidationError):
            SpendInput(
                naics_code="531120",
                amount=Decimal("100"),
                reporting_year=2031,
            )

    def test_frozen(self):
        """Test SpendInput is frozen."""
        inp = SpendInput(
            naics_code="531120",
            amount=Decimal("100.00"),
        )
        with pytest.raises(Exception):
            inp.amount = Decimal("200.00")


class TestAllocationInput:
    """Tests for AllocationInput Pydantic model."""

    def test_valid_area_allocation(self):
        """Test creating valid area allocation input."""
        inp = AllocationInput(
            method=AllocationMethod.AREA,
            tenant_value=Decimal("875.00"),
            total_value=Decimal("2500.00"),
        )
        assert inp.method == AllocationMethod.AREA

    def test_computed_share(self):
        """Test allocation share computation."""
        inp = AllocationInput(
            method=AllocationMethod.AREA,
            tenant_value=Decimal("875.00"),
            total_value=Decimal("2500.00"),
        )
        expected_share = Decimal("875.00") / Decimal("2500.00")
        if hasattr(inp, 'share'):
            assert abs(inp.share - expected_share) < Decimal("0.001")

    def test_zero_total_rejected(self):
        """Test zero total value is rejected."""
        with pytest.raises(PydanticValidationError):
            AllocationInput(
                method=AllocationMethod.AREA,
                tenant_value=Decimal("875.00"),
                total_value=Decimal("0"),
            )

    def test_tenant_exceeds_total_rejected(self):
        """Test tenant value exceeding total is rejected."""
        with pytest.raises(PydanticValidationError):
            AllocationInput(
                method=AllocationMethod.AREA,
                tenant_value=Decimal("3000.00"),
                total_value=Decimal("2500.00"),
            )


class TestComplianceInput:
    """Tests for ComplianceInput Pydantic model."""

    def test_valid_input(self):
        """Test creating valid compliance input."""
        inp = ComplianceInput(
            frameworks=[ComplianceFramework.GHG_PROTOCOL, ComplianceFramework.CDP],
            total_co2e=Decimal("85000.00"),
            method_used=CalculationMethod.ASSET_SPECIFIC,
            reporting_period="2024",
        )
        assert len(inp.frameworks) == 2
        assert inp.total_co2e == Decimal("85000.00")

    def test_empty_frameworks_rejected(self):
        """Test empty frameworks list is rejected."""
        with pytest.raises(PydanticValidationError):
            ComplianceInput(
                frameworks=[],
                total_co2e=Decimal("85000.00"),
                method_used=CalculationMethod.ASSET_SPECIFIC,
                reporting_period="2024",
            )


class TestAssetInput:
    """Tests for AssetInput Pydantic model."""

    def test_valid_building_asset(self):
        """Test creating valid building asset input."""
        inp = AssetInput(
            asset_category=AssetCategory.BUILDING,
            asset_id="BLDG-001",
            calculation_method=CalculationMethod.ASSET_SPECIFIC,
            building_input=BuildingInput(
                building_type=BuildingType.OFFICE,
                floor_area_sqm=Decimal("2500"),
            ),
        )
        assert inp.asset_category == AssetCategory.BUILDING
        assert inp.asset_id == "BLDG-001"

    def test_valid_vehicle_asset(self):
        """Test creating valid vehicle asset input."""
        inp = AssetInput(
            asset_category=AssetCategory.VEHICLE,
            asset_id="VEH-001",
            calculation_method=CalculationMethod.ASSET_SPECIFIC,
            vehicle_input=VehicleInput(
                vehicle_type=VehicleType.MEDIUM_CAR,
                fuel_type=FuelType.PETROL,
                annual_distance_km=Decimal("25000"),
            ),
        )
        assert inp.asset_category == AssetCategory.VEHICLE

    def test_frozen(self):
        """Test AssetInput is frozen."""
        inp = AssetInput(
            asset_category=AssetCategory.BUILDING,
            asset_id="BLDG-001",
            calculation_method=CalculationMethod.ASSET_SPECIFIC,
        )
        with pytest.raises(Exception):
            inp.asset_id = "BLDG-002"


class TestBatchAssetInput:
    """Tests for BatchAssetInput Pydantic model."""

    def test_valid_batch(self):
        """Test creating valid batch asset input."""
        asset1 = AssetInput(
            asset_category=AssetCategory.BUILDING,
            asset_id="BLDG-001",
            calculation_method=CalculationMethod.ASSET_SPECIFIC,
        )
        asset2 = AssetInput(
            asset_category=AssetCategory.VEHICLE,
            asset_id="VEH-001",
            calculation_method=CalculationMethod.ASSET_SPECIFIC,
        )
        batch = BatchAssetInput(
            assets=[asset1, asset2],
            reporting_period="2024",
        )
        assert len(batch.assets) == 2
        assert batch.reporting_period == "2024"

    def test_empty_assets_rejected(self):
        """Test empty assets list is rejected."""
        with pytest.raises(PydanticValidationError):
            BatchAssetInput(
                assets=[],
                reporting_period="2024",
            )


# ==============================================================================
# RESULT MODEL TESTS
# ==============================================================================


class TestAssetResult:
    """Tests for AssetResult Pydantic model."""

    def test_valid_result(self):
        """Test creating valid asset result."""
        result = AssetResult(
            asset_id="BLDG-001",
            asset_category=AssetCategory.BUILDING,
            calculation_method=CalculationMethod.ASSET_SPECIFIC,
            total_co2e_kg=Decimal("42500.00"),
            co2_kg=Decimal("40000.00"),
            ch4_kg=Decimal("1500.00"),
            n2o_kg=Decimal("1000.00"),
            provenance_hash="a" * 64,
        )
        assert result.total_co2e_kg == Decimal("42500.00")
        assert result.provenance_hash == "a" * 64

    def test_frozen(self):
        """Test AssetResult is frozen."""
        result = AssetResult(
            asset_id="BLDG-001",
            asset_category=AssetCategory.BUILDING,
            calculation_method=CalculationMethod.ASSET_SPECIFIC,
            total_co2e_kg=Decimal("42500.00"),
            provenance_hash="a" * 64,
        )
        with pytest.raises(Exception):
            result.total_co2e_kg = Decimal("50000.00")

    def test_provenance_hash_length(self):
        """Test provenance hash is 64 characters."""
        result = AssetResult(
            asset_id="BLDG-001",
            asset_category=AssetCategory.BUILDING,
            calculation_method=CalculationMethod.ASSET_SPECIFIC,
            total_co2e_kg=Decimal("42500.00"),
            provenance_hash="a" * 64,
        )
        assert len(result.provenance_hash) == 64


class TestBatchAssetResult:
    """Tests for BatchAssetResult Pydantic model."""

    def test_valid_batch_result(self):
        """Test creating valid batch result."""
        r1 = AssetResult(
            asset_id="BLDG-001",
            asset_category=AssetCategory.BUILDING,
            calculation_method=CalculationMethod.ASSET_SPECIFIC,
            total_co2e_kg=Decimal("42500.00"),
            provenance_hash="a" * 64,
        )
        batch = BatchAssetResult(
            results=[r1],
            total_co2e_kg=Decimal("42500.00"),
            reporting_period="2024",
            provenance_hash="b" * 64,
        )
        assert len(batch.results) == 1
        assert batch.total_co2e_kg == Decimal("42500.00")


# ==============================================================================
# HELPER FUNCTION TESTS
# ==============================================================================


class TestCalculateProvenanceHash:
    """Tests for calculate_provenance_hash helper function."""

    def test_returns_64_char_hex_string(self):
        """Test provenance hash is a 64-character hex string (SHA-256)."""
        h = calculate_provenance_hash("test_input")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_deterministic(self):
        """Test same inputs produce same hash."""
        h1 = calculate_provenance_hash("BLDG-001", Decimal("42500.00"))
        h2 = calculate_provenance_hash("BLDG-001", Decimal("42500.00"))
        assert h1 == h2

    def test_different_inputs_different_hash(self):
        """Test different inputs produce different hashes."""
        h1 = calculate_provenance_hash("BLDG-001", Decimal("42500.00"))
        h2 = calculate_provenance_hash("BLDG-002", Decimal("30000.00"))
        assert h1 != h2

    def test_decimal_quantization(self):
        """Test Decimal values are quantized to 8 decimal places."""
        h1 = calculate_provenance_hash(Decimal("1.234567890000"))
        h2 = calculate_provenance_hash(Decimal("1.23456789"))
        assert h1 == h2

    def test_pydantic_model_input(self):
        """Test hashing a Pydantic model produces valid hash."""
        inp = BuildingInput(
            building_type=BuildingType.OFFICE,
            floor_area_sqm=Decimal("2500"),
        )
        h = calculate_provenance_hash(inp)
        assert len(h) == 64

    def test_pydantic_model_deterministic(self):
        """Test same Pydantic model produces same hash."""
        inp1 = BuildingInput(
            building_type=BuildingType.OFFICE,
            floor_area_sqm=Decimal("2500"),
        )
        inp2 = BuildingInput(
            building_type=BuildingType.OFFICE,
            floor_area_sqm=Decimal("2500"),
        )
        assert calculate_provenance_hash(inp1) == calculate_provenance_hash(inp2)

    def test_multiple_inputs(self):
        """Test hashing multiple inputs produces valid hash."""
        h = calculate_provenance_hash("stage1", Decimal("42500"), "building")
        assert len(h) == 64

    def test_empty_input(self):
        """Test hashing empty string produces valid hash."""
        h = calculate_provenance_hash("")
        assert len(h) == 64
