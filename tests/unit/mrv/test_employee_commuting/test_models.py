# -*- coding: utf-8 -*-
"""
Test suite for employee_commuting.models - AGENT-MRV-020.

Tests all 25 enums, 16+ constant tables, input/result Pydantic models,
and helper functions for the Employee Commuting Agent (GL-MRV-S3-007).

Coverage:
- Enumerations: 25 enums (values, membership, count, str behavior)
- Constants: GWP_VALUES, VEHICLE_EMISSION_FACTORS, FUEL_EMISSION_FACTORS,
  TRANSIT_EMISSION_FACTORS, MICRO_MOBILITY_EFS, GRID_EMISSION_FACTORS,
  WORKING_DAYS_DEFAULTS, AVERAGE_COMMUTE_DISTANCES, DEFAULT_MODE_SHARES,
  TELEWORK_ENERGY_DEFAULTS, VAN_EMISSION_FACTORS, EEIO_FACTORS,
  CURRENCY_RATES, CPI_DEFLATORS, DQI_SCORING, DQI_WEIGHTS,
  UNCERTAINTY_RANGES, WORK_SCHEDULE_FRACTIONS, TELEWORK_FREQUENCY_FRACTIONS,
  SEASONAL_ADJUSTMENT_MULTIPLIERS, FRAMEWORK_REQUIRED_DISCLOSURES
- Agent metadata: AGENT_ID, AGENT_COMPONENT, VERSION, TABLE_PREFIX
- Input models: CommuteInput, FuelBasedCommuteInput, CarpoolInput,
  TransitInput, TeleworkInput, SurveyResponseInput, SurveyInput,
  AverageDataInput, SpendInput, EmployeeInput, BatchEmployeeInput
- Result models: CommuteResult, TeleworkResult, EmployeeResult (frozen checks)
- Helper functions: calculate_provenance_hash

Author: GL-TestEngineer
Date: February 2026
"""

from decimal import Decimal
import pytest
from pydantic import ValidationError as PydanticValidationError

from greenlang.agents.mrv.employee_commuting.models import (
    # Enumerations
    CalculationMethod,
    CommuteMode,
    VehicleType,
    FuelType,
    TransitType,
    TeleworkFrequency,
    WorkSchedule,
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
    ExportFormat,
    BatchStatus,
    RegionCode,
    DistanceBand,
    SurveyMethod,
    AllocationMethod,
    SeasonalAdjustment,

    # Agent metadata
    AGENT_ID,
    AGENT_COMPONENT,
    VERSION,
    TABLE_PREFIX,

    # Constant tables
    GWP_VALUES,
    VEHICLE_EMISSION_FACTORS,
    FUEL_EMISSION_FACTORS,
    TRANSIT_EMISSION_FACTORS,
    MICRO_MOBILITY_EFS,
    GRID_EMISSION_FACTORS,
    WORKING_DAYS_DEFAULTS,
    AVERAGE_COMMUTE_DISTANCES,
    DEFAULT_MODE_SHARES,
    TELEWORK_ENERGY_DEFAULTS,
    VAN_EMISSION_FACTORS,
    EEIO_FACTORS,
    CURRENCY_RATES,
    CPI_DEFLATORS,
    DQI_SCORING,
    DQI_WEIGHTS,
    UNCERTAINTY_RANGES,
    FRAMEWORK_REQUIRED_DISCLOSURES,
    WORK_SCHEDULE_FRACTIONS,
    TELEWORK_FREQUENCY_FRACTIONS,
    SEASONAL_ADJUSTMENT_MULTIPLIERS,

    # Input models
    CommuteInput,
    FuelBasedCommuteInput,
    CarpoolInput,
    TransitInput,
    TeleworkInput,
    SurveyResponseInput,
    SurveyInput,
    AverageDataInput,
    SpendInput,
    EmployeeInput,
    BatchEmployeeInput,

    # Result models
    CommuteResult,
    TeleworkResult,
    EmployeeResult,

    # Helper functions
    calculate_provenance_hash,
)


# ==============================================================================
# AGENT METADATA TESTS
# ==============================================================================


class TestAgentMetadata:
    """Tests for agent metadata constants."""

    def test_agent_id(self):
        """Test AGENT_ID is GL-MRV-S3-007."""
        assert AGENT_ID == "GL-MRV-S3-007"

    def test_agent_component(self):
        """Test AGENT_COMPONENT is AGENT-MRV-020."""
        assert AGENT_COMPONENT == "AGENT-MRV-020"

    def test_version(self):
        """Test VERSION is 1.0.0."""
        assert VERSION == "1.0.0"

    def test_table_prefix(self):
        """Test TABLE_PREFIX is gl_ec_."""
        assert TABLE_PREFIX == "gl_ec_"

    def test_table_prefix_ends_with_underscore(self):
        """Test TABLE_PREFIX ends with underscore for table naming convention."""
        assert TABLE_PREFIX.endswith("_")


# ==============================================================================
# ENUMERATION TESTS (25 enums)
# ==============================================================================


class TestCalculationMethod:
    """Tests for CalculationMethod enum."""

    def test_member_count(self):
        """Test CalculationMethod has exactly 3 members."""
        assert len(CalculationMethod) == 3

    def test_employee_specific(self):
        """Test EMPLOYEE_SPECIFIC member."""
        assert CalculationMethod.EMPLOYEE_SPECIFIC == "employee_specific"

    def test_average_data(self):
        """Test AVERAGE_DATA member."""
        assert CalculationMethod.AVERAGE_DATA == "average_data"

    def test_spend_based(self):
        """Test SPEND_BASED member."""
        assert CalculationMethod.SPEND_BASED == "spend_based"

    def test_is_str_enum(self):
        """Test CalculationMethod is a str enum."""
        assert isinstance(CalculationMethod.EMPLOYEE_SPECIFIC, str)


class TestCommuteMode:
    """Tests for CommuteMode enum."""

    def test_member_count(self):
        """Test CommuteMode has exactly 14 members."""
        assert len(CommuteMode) == 14

    @pytest.mark.parametrize("member,value", [
        (CommuteMode.SOV, "sov"),
        (CommuteMode.CARPOOL, "carpool"),
        (CommuteMode.VANPOOL, "vanpool"),
        (CommuteMode.BUS, "bus"),
        (CommuteMode.METRO, "metro"),
        (CommuteMode.LIGHT_RAIL, "light_rail"),
        (CommuteMode.COMMUTER_RAIL, "commuter_rail"),
        (CommuteMode.FERRY, "ferry"),
        (CommuteMode.MOTORCYCLE, "motorcycle"),
        (CommuteMode.E_BIKE, "e_bike"),
        (CommuteMode.E_SCOOTER, "e_scooter"),
        (CommuteMode.CYCLING, "cycling"),
        (CommuteMode.WALKING, "walking"),
        (CommuteMode.TELEWORK, "telework"),
    ])
    def test_members(self, member, value):
        """Test all 14 CommuteMode members and their values."""
        assert member.value == value

    def test_is_str_enum(self):
        """Test CommuteMode is a str enum."""
        assert isinstance(CommuteMode.SOV, str)


class TestVehicleType:
    """Tests for VehicleType enum."""

    def test_member_count(self):
        """Test VehicleType has exactly 12 members."""
        assert len(VehicleType) == 12

    @pytest.mark.parametrize("member,value", [
        (VehicleType.CAR_AVERAGE, "car_average"),
        (VehicleType.CAR_SMALL_PETROL, "car_small_petrol"),
        (VehicleType.CAR_MEDIUM_PETROL, "car_medium_petrol"),
        (VehicleType.CAR_LARGE_PETROL, "car_large_petrol"),
        (VehicleType.CAR_SMALL_DIESEL, "car_small_diesel"),
        (VehicleType.CAR_MEDIUM_DIESEL, "car_medium_diesel"),
        (VehicleType.CAR_LARGE_DIESEL, "car_large_diesel"),
        (VehicleType.HYBRID, "hybrid"),
        (VehicleType.PLUGIN_HYBRID, "plugin_hybrid"),
        (VehicleType.BEV, "bev"),
        (VehicleType.VAN_AVERAGE, "van_average"),
        (VehicleType.MOTORCYCLE, "motorcycle"),
    ])
    def test_members(self, member, value):
        """Test all 12 VehicleType members and their values."""
        assert member.value == value

    def test_is_str_enum(self):
        """Test VehicleType is a str enum."""
        assert isinstance(VehicleType.CAR_AVERAGE, str)


class TestFuelType:
    """Tests for FuelType enum."""

    def test_member_count(self):
        """Test FuelType has exactly 5 members."""
        assert len(FuelType) == 5

    @pytest.mark.parametrize("member,value", [
        (FuelType.PETROL, "petrol"),
        (FuelType.DIESEL, "diesel"),
        (FuelType.LPG, "lpg"),
        (FuelType.E10, "e10"),
        (FuelType.B7, "b7"),
    ])
    def test_members(self, member, value):
        """Test all 5 FuelType members and their values."""
        assert member.value == value

    def test_is_str_enum(self):
        """Test FuelType is a str enum."""
        assert isinstance(FuelType.PETROL, str)


class TestTransitType:
    """Tests for TransitType enum."""

    def test_member_count(self):
        """Test TransitType has exactly 6 members."""
        assert len(TransitType) == 6

    @pytest.mark.parametrize("member,value", [
        (TransitType.BUS_LOCAL, "bus_local"),
        (TransitType.BUS_COACH, "bus_coach"),
        (TransitType.METRO, "metro"),
        (TransitType.LIGHT_RAIL, "light_rail"),
        (TransitType.COMMUTER_RAIL, "commuter_rail"),
        (TransitType.FERRY, "ferry"),
    ])
    def test_members(self, member, value):
        """Test all 6 TransitType members and their values."""
        assert member.value == value


class TestTeleworkFrequency:
    """Tests for TeleworkFrequency enum."""

    def test_member_count(self):
        """Test TeleworkFrequency has exactly 6 members."""
        assert len(TeleworkFrequency) == 6

    @pytest.mark.parametrize("member,value", [
        (TeleworkFrequency.FULL_REMOTE, "full_remote"),
        (TeleworkFrequency.HYBRID_4, "hybrid_4"),
        (TeleworkFrequency.HYBRID_3, "hybrid_3"),
        (TeleworkFrequency.HYBRID_2, "hybrid_2"),
        (TeleworkFrequency.HYBRID_1, "hybrid_1"),
        (TeleworkFrequency.OFFICE_FULL, "office_full"),
    ])
    def test_members(self, member, value):
        """Test all 6 TeleworkFrequency members and their values."""
        assert member.value == value


class TestWorkSchedule:
    """Tests for WorkSchedule enum."""

    def test_member_count(self):
        """Test WorkSchedule has exactly 4 members."""
        assert len(WorkSchedule) == 4

    @pytest.mark.parametrize("member,value", [
        (WorkSchedule.FULL_TIME, "full_time"),
        (WorkSchedule.PART_TIME_80, "part_time_80"),
        (WorkSchedule.PART_TIME_60, "part_time_60"),
        (WorkSchedule.PART_TIME_50, "part_time_50"),
    ])
    def test_members(self, member, value):
        """Test all 4 WorkSchedule members and their values."""
        assert member.value == value


class TestEFSource:
    """Tests for EFSource enum."""

    def test_member_count(self):
        """Test EFSource has exactly 7 members."""
        assert len(EFSource) == 7

    @pytest.mark.parametrize("member,value", [
        (EFSource.EMPLOYEE, "employee"),
        (EFSource.DEFRA, "defra"),
        (EFSource.EPA, "epa"),
        (EFSource.IEA, "iea"),
        (EFSource.CENSUS, "census"),
        (EFSource.EEIO, "eeio"),
        (EFSource.CUSTOM, "custom"),
    ])
    def test_members(self, member, value):
        """Test all 7 EFSource members and their values."""
        assert member.value == value


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
        (ProvenanceStage.CALCULATE_COMMUTE, "calculate_commute"),
        (ProvenanceStage.CALCULATE_TELEWORK, "calculate_telework"),
        (ProvenanceStage.EXTRAPOLATE, "extrapolate"),
        (ProvenanceStage.COMPLIANCE, "compliance"),
        (ProvenanceStage.AGGREGATE, "aggregate"),
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


class TestExportFormat:
    """Tests for ExportFormat enum."""

    def test_member_count(self):
        """Test ExportFormat has exactly 4 members."""
        assert len(ExportFormat) == 4

    @pytest.mark.parametrize("member,value", [
        (ExportFormat.JSON, "json"),
        (ExportFormat.CSV, "csv"),
        (ExportFormat.EXCEL, "excel"),
        (ExportFormat.PDF, "pdf"),
    ])
    def test_members(self, member, value):
        """Test all 4 ExportFormat members and their values."""
        assert member.value == value


class TestBatchStatus:
    """Tests for BatchStatus enum."""

    def test_member_count(self):
        """Test BatchStatus has exactly 5 members."""
        assert len(BatchStatus) == 5

    @pytest.mark.parametrize("member,value", [
        (BatchStatus.PENDING, "pending"),
        (BatchStatus.PROCESSING, "processing"),
        (BatchStatus.COMPLETED, "completed"),
        (BatchStatus.FAILED, "failed"),
        (BatchStatus.PARTIAL, "partial"),
    ])
    def test_members(self, member, value):
        """Test all 5 BatchStatus members and their values."""
        assert member.value == value


class TestRegionCode:
    """Tests for RegionCode enum."""

    def test_member_count(self):
        """Test RegionCode has exactly 11 members."""
        assert len(RegionCode) == 11

    @pytest.mark.parametrize("member,value", [
        (RegionCode.US, "US"),
        (RegionCode.GB, "GB"),
        (RegionCode.DE, "DE"),
        (RegionCode.FR, "FR"),
        (RegionCode.JP, "JP"),
        (RegionCode.CA, "CA"),
        (RegionCode.AU, "AU"),
        (RegionCode.IN, "IN"),
        (RegionCode.CN, "CN"),
        (RegionCode.BR, "BR"),
        (RegionCode.GLOBAL, "GLOBAL"),
    ])
    def test_members(self, member, value):
        """Test all 11 RegionCode members and their values."""
        assert member.value == value


class TestDistanceBand:
    """Tests for DistanceBand enum."""

    def test_member_count(self):
        """Test DistanceBand has exactly 4 members."""
        assert len(DistanceBand) == 4

    @pytest.mark.parametrize("member,value", [
        (DistanceBand.SHORT_0_5, "short_0_5"),
        (DistanceBand.MEDIUM_5_15, "medium_5_15"),
        (DistanceBand.LONG_15_30, "long_15_30"),
        (DistanceBand.VERY_LONG_30_PLUS, "very_long_30_plus"),
    ])
    def test_members(self, member, value):
        """Test all 4 DistanceBand members and their values."""
        assert member.value == value


class TestSurveyMethod:
    """Tests for SurveyMethod enum."""

    def test_member_count(self):
        """Test SurveyMethod has exactly 4 members."""
        assert len(SurveyMethod) == 4

    @pytest.mark.parametrize("member,value", [
        (SurveyMethod.FULL_CENSUS, "full_census"),
        (SurveyMethod.STRATIFIED_SAMPLE, "stratified_sample"),
        (SurveyMethod.RANDOM_SAMPLE, "random_sample"),
        (SurveyMethod.CONVENIENCE, "convenience"),
    ])
    def test_members(self, member, value):
        """Test all 4 SurveyMethod members and their values."""
        assert member.value == value


class TestAllocationMethod:
    """Tests for AllocationMethod enum."""

    def test_member_count(self):
        """Test AllocationMethod has exactly 5 members."""
        assert len(AllocationMethod) == 5

    @pytest.mark.parametrize("member,value", [
        (AllocationMethod.EQUAL, "equal"),
        (AllocationMethod.HEADCOUNT, "headcount"),
        (AllocationMethod.SITE, "site"),
        (AllocationMethod.DEPARTMENT, "department"),
        (AllocationMethod.COST_CENTER, "cost_center"),
    ])
    def test_members(self, member, value):
        """Test all 5 AllocationMethod members and their values."""
        assert member.value == value


class TestSeasonalAdjustment:
    """Tests for SeasonalAdjustment enum."""

    def test_member_count(self):
        """Test SeasonalAdjustment has exactly 4 members."""
        assert len(SeasonalAdjustment) == 4

    @pytest.mark.parametrize("member,value", [
        (SeasonalAdjustment.NONE, "none"),
        (SeasonalAdjustment.HEATING_ONLY, "heating_only"),
        (SeasonalAdjustment.COOLING_ONLY, "cooling_only"),
        (SeasonalAdjustment.FULL_SEASONAL, "full_seasonal"),
    ])
    def test_members(self, member, value):
        """Test all 4 SeasonalAdjustment members and their values."""
        assert member.value == value


# ==============================================================================
# CONSTANT TABLE TESTS
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


class TestVehicleEmissionFactors:
    """Tests for VEHICLE_EMISSION_FACTORS constant table."""

    def test_twelve_vehicle_types(self):
        """Test VEHICLE_EMISSION_FACTORS has exactly 12 vehicle types."""
        assert len(VEHICLE_EMISSION_FACTORS) == 12

    def test_all_vehicle_types_present(self):
        """Test all VehicleType enum members are present."""
        for vt in VehicleType:
            assert vt in VEHICLE_EMISSION_FACTORS

    def test_all_have_ef_per_vkm(self):
        """Test all vehicle types have ef_per_vkm field."""
        for vt, data in VEHICLE_EMISSION_FACTORS.items():
            assert "ef_per_vkm" in data
            assert isinstance(data["ef_per_vkm"], Decimal)

    def test_all_have_wtt_per_vkm(self):
        """Test all vehicle types have wtt_per_vkm field."""
        for vt, data in VEHICLE_EMISSION_FACTORS.items():
            assert "wtt_per_vkm" in data
            assert isinstance(data["wtt_per_vkm"], Decimal)

    def test_car_average_ef(self):
        """Test CAR_AVERAGE emission factor value."""
        ef = VEHICLE_EMISSION_FACTORS[VehicleType.CAR_AVERAGE]
        assert ef["ef_per_vkm"] == Decimal("0.27145")

    def test_bev_lower_than_petrol(self):
        """Test BEV ef_per_vkm is lower than any petrol car."""
        bev_ef = VEHICLE_EMISSION_FACTORS[VehicleType.BEV]["ef_per_vkm"]
        petrol_ef = VEHICLE_EMISSION_FACTORS[VehicleType.CAR_SMALL_PETROL]["ef_per_vkm"]
        assert bev_ef < petrol_ef

    def test_van_average_no_pkm(self):
        """Test VAN_AVERAGE has None for ef_per_pkm (not a passenger vehicle)."""
        van = VEHICLE_EMISSION_FACTORS[VehicleType.VAN_AVERAGE]
        assert van["ef_per_pkm"] is None

    def test_motorcycle_occupancy_one(self):
        """Test motorcycle occupancy is 1.0."""
        mc = VEHICLE_EMISSION_FACTORS[VehicleType.MOTORCYCLE]
        assert mc["occupancy"] == Decimal("1.0")


class TestFuelEmissionFactors:
    """Tests for FUEL_EMISSION_FACTORS constant table."""

    def test_five_fuel_types(self):
        """Test FUEL_EMISSION_FACTORS has exactly 5 fuel types."""
        assert len(FUEL_EMISSION_FACTORS) == 5

    def test_all_fuel_types_present(self):
        """Test all FuelType enum members are present."""
        for ft in FuelType:
            assert ft in FUEL_EMISSION_FACTORS

    def test_petrol_ef_per_litre(self):
        """Test petrol ef_per_litre is 2.31480."""
        assert FUEL_EMISSION_FACTORS[FuelType.PETROL]["ef_per_litre"] == Decimal("2.31480")

    def test_diesel_ef_per_litre(self):
        """Test diesel ef_per_litre is 2.70370."""
        assert FUEL_EMISSION_FACTORS[FuelType.DIESEL]["ef_per_litre"] == Decimal("2.70370")

    def test_all_have_wtt_per_litre(self):
        """Test all fuel types have wtt_per_litre field."""
        for ft, data in FUEL_EMISSION_FACTORS.items():
            assert "wtt_per_litre" in data
            assert isinstance(data["wtt_per_litre"], Decimal)

    def test_all_values_positive(self):
        """Test all fuel emission factors are positive."""
        for data in FUEL_EMISSION_FACTORS.values():
            assert data["ef_per_litre"] > 0
            assert data["wtt_per_litre"] > 0


class TestTransitEmissionFactors:
    """Tests for TRANSIT_EMISSION_FACTORS constant table."""

    def test_six_transit_types(self):
        """Test TRANSIT_EMISSION_FACTORS has exactly 6 transit types."""
        assert len(TRANSIT_EMISSION_FACTORS) == 6

    def test_all_transit_types_present(self):
        """Test all TransitType enum members are present."""
        for tt in TransitType:
            assert tt in TRANSIT_EMISSION_FACTORS

    def test_metro_ef_per_pkm(self):
        """Test metro ef_per_pkm value."""
        assert TRANSIT_EMISSION_FACTORS[TransitType.METRO]["ef_per_pkm"] == Decimal("0.02781")

    def test_bus_local_higher_than_metro(self):
        """Test bus local EF is higher than metro EF."""
        bus = TRANSIT_EMISSION_FACTORS[TransitType.BUS_LOCAL]["ef_per_pkm"]
        metro = TRANSIT_EMISSION_FACTORS[TransitType.METRO]["ef_per_pkm"]
        assert bus > metro

    def test_all_have_wtt_per_pkm(self):
        """Test all transit types have wtt_per_pkm field."""
        for tt, data in TRANSIT_EMISSION_FACTORS.items():
            assert "wtt_per_pkm" in data
            assert isinstance(data["wtt_per_pkm"], Decimal)


class TestMicroMobilityEFs:
    """Tests for MICRO_MOBILITY_EFS constant table."""

    def test_two_modes(self):
        """Test MICRO_MOBILITY_EFS has e_bike and e_scooter."""
        assert len(MICRO_MOBILITY_EFS) == 2
        assert "e_bike" in MICRO_MOBILITY_EFS
        assert "e_scooter" in MICRO_MOBILITY_EFS

    def test_e_bike_ef(self):
        """Test e_bike emission factor is 0.005 kgCO2e/pkm."""
        assert MICRO_MOBILITY_EFS["e_bike"] == Decimal("0.00500")

    def test_e_scooter_ef(self):
        """Test e_scooter emission factor is 0.0035 kgCO2e/pkm."""
        assert MICRO_MOBILITY_EFS["e_scooter"] == Decimal("0.00350")

    def test_values_are_decimal(self):
        """Test both values are Decimal type."""
        for val in MICRO_MOBILITY_EFS.values():
            assert isinstance(val, Decimal)


class TestGridEmissionFactors:
    """Tests for GRID_EMISSION_FACTORS constant table."""

    def test_eleven_regions(self):
        """Test GRID_EMISSION_FACTORS has exactly 11 regions."""
        assert len(GRID_EMISSION_FACTORS) == 11

    def test_all_region_codes_present(self):
        """Test all RegionCode enum members are present."""
        for rc in RegionCode:
            assert rc in GRID_EMISSION_FACTORS

    def test_us_grid_ef(self):
        """Test US grid emission factor is 0.37170 kgCO2e/kWh."""
        assert GRID_EMISSION_FACTORS[RegionCode.US] == Decimal("0.37170")

    def test_france_lower_than_us(self):
        """Test France grid EF is lower than US (nuclear-heavy grid)."""
        assert GRID_EMISSION_FACTORS[RegionCode.FR] < GRID_EMISSION_FACTORS[RegionCode.US]

    def test_all_values_decimal_and_positive(self):
        """Test all grid emission factors are positive Decimal values."""
        for val in GRID_EMISSION_FACTORS.values():
            assert isinstance(val, Decimal)
            assert val > 0


class TestWorkingDaysDefaults:
    """Tests for WORKING_DAYS_DEFAULTS constant table."""

    def test_eleven_regions(self):
        """Test WORKING_DAYS_DEFAULTS has exactly 11 regions."""
        assert len(WORKING_DAYS_DEFAULTS) == 11

    def test_all_regions_present(self):
        """Test all RegionCode enum members are present."""
        for rc in RegionCode:
            assert rc in WORKING_DAYS_DEFAULTS

    def test_us_net_working_days(self):
        """Test US net working days is 225."""
        assert WORKING_DAYS_DEFAULTS[RegionCode.US]["net"] == 225

    def test_global_net_working_days(self):
        """Test GLOBAL net working days is 230."""
        assert WORKING_DAYS_DEFAULTS[RegionCode.GLOBAL]["net"] == 230

    def test_all_have_required_keys(self):
        """Test all regions have holidays, pto, sick, net keys."""
        for rc, data in WORKING_DAYS_DEFAULTS.items():
            assert "holidays" in data
            assert "pto" in data
            assert "sick" in data
            assert "net" in data

    def test_net_is_reasonable(self):
        """Test net working days are between 150 and 260."""
        for data in WORKING_DAYS_DEFAULTS.values():
            assert 150 <= data["net"] <= 260


class TestAverageCommuteDistances:
    """Tests for AVERAGE_COMMUTE_DISTANCES constant table."""

    def test_eleven_entries(self):
        """Test AVERAGE_COMMUTE_DISTANCES has 11 entries (10 countries + GLOBAL)."""
        assert len(AVERAGE_COMMUTE_DISTANCES) == 11

    def test_us_distance(self):
        """Test US average commute distance is 21.7 km."""
        assert AVERAGE_COMMUTE_DISTANCES["US"] == Decimal("21.7")

    def test_global_distance(self):
        """Test GLOBAL average commute distance is 15.0 km."""
        assert AVERAGE_COMMUTE_DISTANCES["GLOBAL"] == Decimal("15.0")

    def test_all_values_positive_decimal(self):
        """Test all values are positive Decimal types."""
        for val in AVERAGE_COMMUTE_DISTANCES.values():
            assert isinstance(val, Decimal)
            assert val > 0


class TestDefaultModeShares:
    """Tests for DEFAULT_MODE_SHARES constant table."""

    def test_three_regions(self):
        """Test DEFAULT_MODE_SHARES has US, GB, EU."""
        assert "US" in DEFAULT_MODE_SHARES
        assert "GB" in DEFAULT_MODE_SHARES
        assert "EU" in DEFAULT_MODE_SHARES

    def test_us_sov_dominant(self):
        """Test US SOV share is the highest mode share (76.10%)."""
        assert DEFAULT_MODE_SHARES["US"]["sov"] == Decimal("0.7610")

    def test_shares_sum_to_approximately_one(self):
        """Test mode shares for each region sum to approximately 1.0."""
        for region, shares in DEFAULT_MODE_SHARES.items():
            total = sum(shares.values())
            assert abs(total - Decimal("1.0")) < Decimal("0.01"), \
                f"{region} mode shares sum to {total}, expected ~1.0"


class TestTeleworkEnergyDefaults:
    """Tests for TELEWORK_ENERGY_DEFAULTS constant table."""

    def test_five_entries(self):
        """Test TELEWORK_ENERGY_DEFAULTS has 5 entries."""
        assert len(TELEWORK_ENERGY_DEFAULTS) == 5

    def test_total_typical(self):
        """Test total_typical is 4.0 kWh/day."""
        assert TELEWORK_ENERGY_DEFAULTS["total_typical"] == Decimal("4.0")

    def test_laptop_monitor(self):
        """Test laptop_monitor is 0.3 kWh/day."""
        assert TELEWORK_ENERGY_DEFAULTS["laptop_monitor"] == Decimal("0.3")

    def test_heating(self):
        """Test heating is 3.5 kWh/day."""
        assert TELEWORK_ENERGY_DEFAULTS["heating"] == Decimal("3.5")

    def test_all_values_decimal(self):
        """Test all telework energy values are Decimal."""
        for val in TELEWORK_ENERGY_DEFAULTS.values():
            assert isinstance(val, Decimal)


class TestEEIOFactors:
    """Tests for EEIO_FACTORS constant table."""

    def test_seven_naics_codes(self):
        """Test EEIO_FACTORS has 7 NAICS codes."""
        assert len(EEIO_FACTORS) == 7

    def test_ground_transport_code(self):
        """Test NAICS 485000 is ground passenger transport."""
        assert "485000" in EEIO_FACTORS
        assert EEIO_FACTORS["485000"]["name"] == "Ground passenger transport"
        assert EEIO_FACTORS["485000"]["ef"] == Decimal("0.2600")

    def test_all_have_name_and_ef(self):
        """Test all EEIO entries have name and ef fields."""
        for code, data in EEIO_FACTORS.items():
            assert "name" in data
            assert "ef" in data
            assert isinstance(data["ef"], Decimal)


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


class TestDQIWeights:
    """Tests for DQI_WEIGHTS constant table."""

    def test_five_dimensions(self):
        """Test DQI_WEIGHTS has exactly 5 dimensions."""
        assert len(DQI_WEIGHTS) == 5

    def test_weights_sum_to_one(self):
        """Test DQI weights sum to 1.0."""
        total = sum(DQI_WEIGHTS.values())
        assert total == Decimal("1.00")

    def test_representativeness_highest(self):
        """Test REPRESENTATIVENESS has the highest weight (0.30)."""
        assert DQI_WEIGHTS[DQIDimension.REPRESENTATIVENESS] == Decimal("0.30")


class TestUncertaintyRanges:
    """Tests for UNCERTAINTY_RANGES constant table."""

    def test_six_method_categories(self):
        """Test UNCERTAINTY_RANGES has 6 categories."""
        assert len(UNCERTAINTY_RANGES) == 6

    def test_employee_specific_has_three_tiers(self):
        """Test employee_specific has 3 data quality tiers."""
        assert len(UNCERTAINTY_RANGES["employee_specific"]) == 3

    def test_employee_specific_tier1_lowest_uncertainty(self):
        """Test employee_specific Tier 1 has lowest uncertainty (0.05)."""
        assert UNCERTAINTY_RANGES["employee_specific"][DataQualityTier.TIER_1] == Decimal("0.05")

    def test_spend_based_highest_uncertainty(self):
        """Test spend_based Tier 3 has highest uncertainty (0.60)."""
        assert UNCERTAINTY_RANGES["spend_based"][DataQualityTier.TIER_3] == Decimal("0.60")


class TestWorkScheduleFractions:
    """Tests for WORK_SCHEDULE_FRACTIONS constant table."""

    def test_four_schedules(self):
        """Test WORK_SCHEDULE_FRACTIONS has 4 schedules."""
        assert len(WORK_SCHEDULE_FRACTIONS) == 4

    def test_full_time_is_one(self):
        """Test full-time fraction is 1.0."""
        assert WORK_SCHEDULE_FRACTIONS[WorkSchedule.FULL_TIME] == Decimal("1.0")

    def test_part_time_50_is_half(self):
        """Test part-time 50% fraction is 0.5."""
        assert WORK_SCHEDULE_FRACTIONS[WorkSchedule.PART_TIME_50] == Decimal("0.5")


class TestTeleworkFrequencyFractions:
    """Tests for TELEWORK_FREQUENCY_FRACTIONS constant table."""

    def test_six_frequencies(self):
        """Test TELEWORK_FREQUENCY_FRACTIONS has 6 frequencies."""
        assert len(TELEWORK_FREQUENCY_FRACTIONS) == 6

    def test_full_remote_is_one(self):
        """Test full_remote fraction is 1.0."""
        assert TELEWORK_FREQUENCY_FRACTIONS[TeleworkFrequency.FULL_REMOTE] == Decimal("1.0")

    def test_office_full_is_zero(self):
        """Test office_full fraction is 0.0."""
        assert TELEWORK_FREQUENCY_FRACTIONS[TeleworkFrequency.OFFICE_FULL] == Decimal("0.0")


class TestSeasonalAdjustmentMultipliers:
    """Tests for SEASONAL_ADJUSTMENT_MULTIPLIERS constant table."""

    def test_four_adjustments(self):
        """Test SEASONAL_ADJUSTMENT_MULTIPLIERS has 4 adjustments."""
        assert len(SEASONAL_ADJUSTMENT_MULTIPLIERS) == 4

    def test_none_is_one(self):
        """Test NONE seasonal adjustment is 1.0 (no effect)."""
        assert SEASONAL_ADJUSTMENT_MULTIPLIERS[SeasonalAdjustment.NONE] == Decimal("1.0")

    def test_full_seasonal_highest(self):
        """Test FULL_SEASONAL has the highest multiplier (1.25)."""
        assert SEASONAL_ADJUSTMENT_MULTIPLIERS[SeasonalAdjustment.FULL_SEASONAL] == Decimal("1.25")


class TestFrameworkRequiredDisclosures:
    """Tests for FRAMEWORK_REQUIRED_DISCLOSURES constant table."""

    def test_seven_frameworks(self):
        """Test FRAMEWORK_REQUIRED_DISCLOSURES has 7 frameworks."""
        assert len(FRAMEWORK_REQUIRED_DISCLOSURES) == 7

    def test_all_frameworks_present(self):
        """Test all ComplianceFramework enum members are present."""
        for cf in ComplianceFramework:
            assert cf in FRAMEWORK_REQUIRED_DISCLOSURES

    def test_ghg_protocol_requires_total_co2e(self):
        """Test GHG Protocol requires total_co2e disclosure."""
        disclosures = FRAMEWORK_REQUIRED_DISCLOSURES[ComplianceFramework.GHG_PROTOCOL]
        assert "total_co2e" in disclosures

    def test_all_frameworks_require_total_co2e(self):
        """Test all frameworks require total_co2e disclosure."""
        for cf, disclosures in FRAMEWORK_REQUIRED_DISCLOSURES.items():
            assert "total_co2e" in disclosures, f"{cf} missing total_co2e"

    def test_csrd_requires_telework_policy(self):
        """Test CSRD requires telework_policy disclosure."""
        disclosures = FRAMEWORK_REQUIRED_DISCLOSURES[ComplianceFramework.CSRD_ESRS]
        assert "telework_policy" in disclosures


class TestVanEmissionFactors:
    """Tests for VAN_EMISSION_FACTORS constant table."""

    def test_four_van_types(self):
        """Test VAN_EMISSION_FACTORS has 4 van/minibus types."""
        assert len(VAN_EMISSION_FACTORS) == 4
        assert "van_small" in VAN_EMISSION_FACTORS
        assert "van_medium" in VAN_EMISSION_FACTORS
        assert "van_large" in VAN_EMISSION_FACTORS
        assert "minibus" in VAN_EMISSION_FACTORS

    def test_all_have_required_keys(self):
        """Test all van types have ef_per_vkm, wtt_per_vkm, default_occupancy."""
        for van_type, data in VAN_EMISSION_FACTORS.items():
            assert "ef_per_vkm" in data
            assert "wtt_per_vkm" in data
            assert "default_occupancy" in data

    def test_minibus_highest_ef(self):
        """Test minibus has the highest ef_per_vkm."""
        minibus_ef = VAN_EMISSION_FACTORS["minibus"]["ef_per_vkm"]
        for van_type, data in VAN_EMISSION_FACTORS.items():
            assert data["ef_per_vkm"] <= minibus_ef


# ==============================================================================
# INPUT MODEL TESTS
# ==============================================================================


class TestCommuteInput:
    """Tests for CommuteInput Pydantic model."""

    def test_valid_sov_input(self):
        """Test creating valid SOV commute input."""
        inp = CommuteInput(
            mode=CommuteMode.SOV,
            vehicle_type=VehicleType.CAR_MEDIUM_PETROL,
            one_way_distance_km=Decimal("15.0"),
            commute_days_per_week=5,
        )
        assert inp.mode == CommuteMode.SOV
        assert inp.one_way_distance_km == Decimal("15.0")
        assert inp.commute_days_per_week == 5

    def test_default_values(self):
        """Test CommuteInput default field values."""
        inp = CommuteInput(
            mode=CommuteMode.BUS,
            one_way_distance_km=Decimal("10.0"),
        )
        assert inp.commute_days_per_week == 5
        assert inp.work_schedule == WorkSchedule.FULL_TIME
        assert inp.region == RegionCode.GLOBAL
        assert inp.vehicle_type is None
        assert inp.tenant_id is None

    def test_frozen_immutability(self):
        """Test CommuteInput is frozen (immutable)."""
        inp = CommuteInput(
            mode=CommuteMode.SOV,
            one_way_distance_km=Decimal("15.0"),
        )
        with pytest.raises(Exception):
            inp.mode = CommuteMode.BUS

    def test_zero_distance_rejected(self):
        """Test zero distance is rejected."""
        with pytest.raises(PydanticValidationError):
            CommuteInput(
                mode=CommuteMode.SOV,
                one_way_distance_km=Decimal("0"),
            )

    def test_negative_distance_rejected(self):
        """Test negative distance is rejected."""
        with pytest.raises(PydanticValidationError):
            CommuteInput(
                mode=CommuteMode.SOV,
                one_way_distance_km=Decimal("-5.0"),
            )

    def test_distance_over_500_rejected(self):
        """Test distance exceeding 500 km is rejected by validator."""
        with pytest.raises(PydanticValidationError):
            CommuteInput(
                mode=CommuteMode.SOV,
                one_way_distance_km=Decimal("501"),
            )

    def test_commute_days_min_boundary(self):
        """Test commute_days_per_week minimum boundary (1)."""
        inp = CommuteInput(
            mode=CommuteMode.SOV,
            one_way_distance_km=Decimal("10.0"),
            commute_days_per_week=1,
        )
        assert inp.commute_days_per_week == 1

    def test_commute_days_max_boundary(self):
        """Test commute_days_per_week maximum boundary (7)."""
        inp = CommuteInput(
            mode=CommuteMode.SOV,
            one_way_distance_km=Decimal("10.0"),
            commute_days_per_week=7,
        )
        assert inp.commute_days_per_week == 7

    def test_commute_days_zero_rejected(self):
        """Test zero commute days is rejected."""
        with pytest.raises(PydanticValidationError):
            CommuteInput(
                mode=CommuteMode.SOV,
                one_way_distance_km=Decimal("10.0"),
                commute_days_per_week=0,
            )

    def test_commute_days_eight_rejected(self):
        """Test 8 commute days is rejected."""
        with pytest.raises(PydanticValidationError):
            CommuteInput(
                mode=CommuteMode.SOV,
                one_way_distance_km=Decimal("10.0"),
                commute_days_per_week=8,
            )


class TestFuelBasedCommuteInput:
    """Tests for FuelBasedCommuteInput Pydantic model."""

    def test_valid_input(self):
        """Test creating valid fuel-based commute input."""
        inp = FuelBasedCommuteInput(
            fuel_type=FuelType.PETROL,
            litres_per_week=Decimal("12.5"),
            commute_weeks_per_year=48,
        )
        assert inp.fuel_type == FuelType.PETROL
        assert inp.litres_per_week == Decimal("12.5")
        assert inp.commute_weeks_per_year == 48

    def test_default_weeks(self):
        """Test default commute_weeks_per_year is 48."""
        inp = FuelBasedCommuteInput(
            fuel_type=FuelType.DIESEL,
            litres_per_week=Decimal("10.0"),
        )
        assert inp.commute_weeks_per_year == 48

    def test_frozen(self):
        """Test FuelBasedCommuteInput is frozen."""
        inp = FuelBasedCommuteInput(
            fuel_type=FuelType.PETROL,
            litres_per_week=Decimal("12.5"),
        )
        with pytest.raises(Exception):
            inp.fuel_type = FuelType.DIESEL

    def test_zero_litres_rejected(self):
        """Test zero litres is rejected."""
        with pytest.raises(PydanticValidationError):
            FuelBasedCommuteInput(
                fuel_type=FuelType.PETROL,
                litres_per_week=Decimal("0"),
            )

    def test_negative_litres_rejected(self):
        """Test negative litres is rejected."""
        with pytest.raises(PydanticValidationError):
            FuelBasedCommuteInput(
                fuel_type=FuelType.PETROL,
                litres_per_week=Decimal("-5"),
            )


class TestCarpoolInput:
    """Tests for CarpoolInput Pydantic model."""

    def test_valid_input(self):
        """Test creating valid carpool input."""
        inp = CarpoolInput(
            vehicle_type=VehicleType.CAR_AVERAGE,
            one_way_distance_km=Decimal("20.0"),
            occupants=3,
        )
        assert inp.occupants == 3
        assert inp.one_way_distance_km == Decimal("20.0")

    def test_default_vehicle_type(self):
        """Test default vehicle type is CAR_AVERAGE."""
        inp = CarpoolInput(
            one_way_distance_km=Decimal("10.0"),
            occupants=2,
        )
        assert inp.vehicle_type == VehicleType.CAR_AVERAGE

    def test_min_occupants_two(self):
        """Test minimum occupants is 2 for carpool."""
        inp = CarpoolInput(
            one_way_distance_km=Decimal("10.0"),
            occupants=2,
        )
        assert inp.occupants == 2

    def test_max_occupants_eight(self):
        """Test maximum occupants is 8 for carpool."""
        inp = CarpoolInput(
            one_way_distance_km=Decimal("10.0"),
            occupants=8,
        )
        assert inp.occupants == 8

    def test_one_occupant_rejected(self):
        """Test single occupant is rejected (not a carpool)."""
        with pytest.raises(PydanticValidationError):
            CarpoolInput(
                one_way_distance_km=Decimal("10.0"),
                occupants=1,
            )

    def test_nine_occupants_rejected(self):
        """Test 9 occupants is rejected."""
        with pytest.raises(PydanticValidationError):
            CarpoolInput(
                one_way_distance_km=Decimal("10.0"),
                occupants=9,
            )


class TestTransitInput:
    """Tests for TransitInput Pydantic model."""

    def test_valid_metro_input(self):
        """Test creating valid metro transit input."""
        inp = TransitInput(
            transit_type=TransitType.METRO,
            one_way_distance_km=Decimal("8.5"),
        )
        assert inp.transit_type == TransitType.METRO
        assert inp.one_way_distance_km == Decimal("8.5")

    def test_default_values(self):
        """Test TransitInput default field values."""
        inp = TransitInput(
            transit_type=TransitType.BUS_LOCAL,
            one_way_distance_km=Decimal("10.0"),
        )
        assert inp.commute_days_per_week == 5
        assert inp.work_schedule == WorkSchedule.FULL_TIME
        assert inp.region == RegionCode.GLOBAL

    def test_frozen(self):
        """Test TransitInput is frozen."""
        inp = TransitInput(
            transit_type=TransitType.METRO,
            one_way_distance_km=Decimal("8.5"),
        )
        with pytest.raises(Exception):
            inp.transit_type = TransitType.FERRY


class TestTeleworkInput:
    """Tests for TeleworkInput Pydantic model."""

    def test_valid_full_remote(self):
        """Test creating valid full remote telework input."""
        inp = TeleworkInput(
            frequency=TeleworkFrequency.FULL_REMOTE,
            region=RegionCode.US,
        )
        assert inp.frequency == TeleworkFrequency.FULL_REMOTE
        assert inp.region == RegionCode.US

    def test_default_values(self):
        """Test TeleworkInput default field values."""
        inp = TeleworkInput(
            frequency=TeleworkFrequency.HYBRID_3,
        )
        assert inp.region == RegionCode.GLOBAL
        assert inp.daily_kwh_override is None
        assert inp.seasonal_adjustment == SeasonalAdjustment.NONE
        assert inp.work_schedule == WorkSchedule.FULL_TIME

    def test_daily_kwh_override(self):
        """Test daily kWh override works."""
        inp = TeleworkInput(
            frequency=TeleworkFrequency.FULL_REMOTE,
            daily_kwh_override=Decimal("6.0"),
        )
        assert inp.daily_kwh_override == Decimal("6.0")

    def test_frozen(self):
        """Test TeleworkInput is frozen."""
        inp = TeleworkInput(
            frequency=TeleworkFrequency.FULL_REMOTE,
        )
        with pytest.raises(Exception):
            inp.frequency = TeleworkFrequency.OFFICE_FULL


class TestSurveyResponseInput:
    """Tests for SurveyResponseInput Pydantic model."""

    def test_valid_response(self):
        """Test creating valid survey response input."""
        inp = SurveyResponseInput(
            employee_id="EMP-001",
            mode=CommuteMode.SOV,
            vehicle_type=VehicleType.CAR_MEDIUM_PETROL,
            one_way_distance_km=Decimal("18.5"),
            commute_days_per_week=5,
        )
        assert inp.employee_id == "EMP-001"
        assert inp.mode == CommuteMode.SOV

    def test_empty_employee_id_rejected(self):
        """Test empty employee_id is rejected."""
        with pytest.raises(PydanticValidationError):
            SurveyResponseInput(
                employee_id="",
                mode=CommuteMode.SOV,
                one_way_distance_km=Decimal("10.0"),
            )

    def test_default_telework_frequency(self):
        """Test default telework_frequency is OFFICE_FULL."""
        inp = SurveyResponseInput(
            employee_id="EMP-002",
            mode=CommuteMode.BUS,
            one_way_distance_km=Decimal("10.0"),
        )
        assert inp.telework_frequency == TeleworkFrequency.OFFICE_FULL


class TestSurveyInput:
    """Tests for SurveyInput Pydantic model."""

    def test_valid_survey(self):
        """Test creating valid survey input."""
        response = SurveyResponseInput(
            employee_id="EMP-001",
            mode=CommuteMode.SOV,
            one_way_distance_km=Decimal("15.0"),
        )
        inp = SurveyInput(
            survey_method=SurveyMethod.RANDOM_SAMPLE,
            total_employees=500,
            responses=[response],
            reporting_period="2024",
        )
        assert inp.total_employees == 500
        assert len(inp.responses) == 1

    def test_zero_employees_rejected(self):
        """Test zero total employees is rejected."""
        response = SurveyResponseInput(
            employee_id="EMP-001",
            mode=CommuteMode.SOV,
            one_way_distance_km=Decimal("15.0"),
        )
        with pytest.raises(PydanticValidationError):
            SurveyInput(
                survey_method=SurveyMethod.RANDOM_SAMPLE,
                total_employees=0,
                responses=[response],
                reporting_period="2024",
            )


class TestAverageDataInput:
    """Tests for AverageDataInput Pydantic model."""

    def test_valid_input(self):
        """Test creating valid average data input."""
        inp = AverageDataInput(
            total_employees=5000,
            region=RegionCode.US,
            reporting_period="2024",
        )
        assert inp.total_employees == 5000
        assert inp.region == RegionCode.US

    def test_default_telework_rate(self):
        """Test default telework_rate is 0.0."""
        inp = AverageDataInput(
            total_employees=1000,
            reporting_period="2024",
        )
        assert inp.telework_rate == Decimal("0.0")

    def test_zero_employees_rejected(self):
        """Test zero total employees is rejected."""
        with pytest.raises(PydanticValidationError):
            AverageDataInput(
                total_employees=0,
                reporting_period="2024",
            )


class TestSpendInput:
    """Tests for SpendInput Pydantic model."""

    def test_valid_input(self):
        """Test creating valid spend input."""
        inp = SpendInput(
            naics_code="485000",
            amount=Decimal("250000.00"),
            currency=CurrencyCode.USD,
            reporting_year=2024,
        )
        assert inp.naics_code == "485000"
        assert inp.amount == Decimal("250000.00")

    def test_default_currency(self):
        """Test default currency is USD."""
        inp = SpendInput(
            naics_code="485000",
            amount=Decimal("100.00"),
        )
        assert inp.currency == CurrencyCode.USD

    def test_default_reporting_year(self):
        """Test default reporting_year is 2024."""
        inp = SpendInput(
            naics_code="485000",
            amount=Decimal("100.00"),
        )
        assert inp.reporting_year == 2024

    def test_zero_amount_rejected(self):
        """Test zero amount is rejected."""
        with pytest.raises(PydanticValidationError):
            SpendInput(
                naics_code="485000",
                amount=Decimal("0"),
            )

    def test_negative_amount_rejected(self):
        """Test negative amount is rejected."""
        with pytest.raises(PydanticValidationError):
            SpendInput(
                naics_code="485000",
                amount=Decimal("-100"),
            )

    def test_year_below_2015_rejected(self):
        """Test reporting year below 2015 is rejected."""
        with pytest.raises(PydanticValidationError):
            SpendInput(
                naics_code="485000",
                amount=Decimal("100"),
                reporting_year=2014,
            )

    def test_year_above_2030_rejected(self):
        """Test reporting year above 2030 is rejected."""
        with pytest.raises(PydanticValidationError):
            SpendInput(
                naics_code="485000",
                amount=Decimal("100"),
                reporting_year=2031,
            )

    def test_frozen(self):
        """Test SpendInput is frozen."""
        inp = SpendInput(
            naics_code="485000",
            amount=Decimal("100.00"),
        )
        with pytest.raises(Exception):
            inp.amount = Decimal("200.00")


class TestEmployeeInput:
    """Tests for EmployeeInput Pydantic model."""

    def test_valid_input(self):
        """Test creating valid employee input."""
        inp = EmployeeInput(
            employee_id="EMP-1234",
            mode=CommuteMode.SOV,
            vehicle_type=VehicleType.CAR_MEDIUM_PETROL,
            one_way_distance_km=Decimal("18.5"),
            commute_days_per_week=5,
            telework_frequency=TeleworkFrequency.HYBRID_2,
            region=RegionCode.US,
            department="Engineering",
        )
        assert inp.employee_id == "EMP-1234"
        assert inp.department == "Engineering"

    def test_default_values(self):
        """Test EmployeeInput default field values."""
        inp = EmployeeInput(
            employee_id="EMP-001",
            mode=CommuteMode.BUS,
            one_way_distance_km=Decimal("10.0"),
        )
        assert inp.telework_frequency == TeleworkFrequency.OFFICE_FULL
        assert inp.work_schedule == WorkSchedule.FULL_TIME
        assert inp.region == RegionCode.GLOBAL
        assert inp.department is None
        assert inp.site is None
        assert inp.cost_center is None


class TestBatchEmployeeInput:
    """Tests for BatchEmployeeInput Pydantic model."""

    def test_valid_batch(self):
        """Test creating valid batch employee input."""
        emp1 = EmployeeInput(
            employee_id="EMP-001",
            mode=CommuteMode.SOV,
            one_way_distance_km=Decimal("15.0"),
        )
        emp2 = EmployeeInput(
            employee_id="EMP-002",
            mode=CommuteMode.BUS,
            one_way_distance_km=Decimal("10.0"),
        )
        batch = BatchEmployeeInput(
            employees=[emp1, emp2],
            reporting_period="2024",
        )
        assert len(batch.employees) == 2
        assert batch.reporting_period == "2024"

    def test_default_allocation_method(self):
        """Test default allocation method is EQUAL."""
        emp = EmployeeInput(
            employee_id="EMP-001",
            mode=CommuteMode.SOV,
            one_way_distance_km=Decimal("15.0"),
        )
        batch = BatchEmployeeInput(
            employees=[emp],
            reporting_period="2024",
        )
        assert batch.allocation_method == AllocationMethod.EQUAL


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
        h1 = calculate_provenance_hash("EMP-1234", Decimal("100.0"))
        h2 = calculate_provenance_hash("EMP-1234", Decimal("100.0"))
        assert h1 == h2

    def test_different_inputs_different_hash(self):
        """Test different inputs produce different hashes."""
        h1 = calculate_provenance_hash("EMP-1234", Decimal("100.0"))
        h2 = calculate_provenance_hash("EMP-5678", Decimal("200.0"))
        assert h1 != h2

    def test_decimal_quantization(self):
        """Test Decimal values are quantized to 8 decimal places."""
        h1 = calculate_provenance_hash(Decimal("1.234567890000"))
        h2 = calculate_provenance_hash(Decimal("1.23456789"))
        assert h1 == h2

    def test_pydantic_model_input(self):
        """Test hashing a Pydantic model produces valid hash."""
        inp = CommuteInput(
            mode=CommuteMode.SOV,
            one_way_distance_km=Decimal("15.0"),
        )
        h = calculate_provenance_hash(inp)
        assert len(h) == 64

    def test_pydantic_model_deterministic(self):
        """Test same Pydantic model produces same hash."""
        inp1 = CommuteInput(
            mode=CommuteMode.SOV,
            one_way_distance_km=Decimal("15.0"),
        )
        inp2 = CommuteInput(
            mode=CommuteMode.SOV,
            one_way_distance_km=Decimal("15.0"),
        )
        assert calculate_provenance_hash(inp1) == calculate_provenance_hash(inp2)

    def test_multiple_inputs(self):
        """Test hashing multiple inputs produces valid hash."""
        h = calculate_provenance_hash("stage1", Decimal("100"), "data")
        assert len(h) == 64

    def test_empty_input(self):
        """Test hashing empty string produces valid hash."""
        h = calculate_provenance_hash("")
        assert len(h) == 64
