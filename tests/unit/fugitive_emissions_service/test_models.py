# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-MRV-005 Fugitive Emissions Agent Data Models.

Tests all 16 enumerations, constant lookup tables (GWP_VALUES,
EPA_COMPONENT_EMISSION_FACTORS, EPA_CORRELATION_COEFFICIENTS,
IPCC_COAL_EMISSION_FACTORS, COAL_METHANE_FACTORS, WASTEWATER_MCF,
PNEUMATIC_RATES_M3_PER_DAY, SOURCE_CATEGORY_MAP, SOURCE_DEFAULT_GASES),
and Pydantic v2 models (FugitiveSourceInfo, ComponentRecord,
EmissionFactorRecord, SurveyRecord, LeakRecord, RepairRecord,
CalculationRequest).

Test Classes:
    - TestFugitiveSourceCategory           (15 tests)
    - TestFugitiveSourceType               (25 tests)
    - TestComponentType                     (9 tests)
    - TestServiceType                       (5 tests)
    - TestEmissionGas                       (5 tests)
    - TestCalculationMethod                 (6 tests)
    - TestEmissionFactorSource              (7 tests)
    - TestGWPSource                         (5 tests)
    - TestSurveyType                        (5 tests)
    - TestLeakStatus                        (6 tests)
    - TestCoalRank                          (5 tests)
    - TestWastewaterType                    (6 tests)
    - TestComplianceStatus                  (5 tests)
    - TestReportingPeriod                   (4 tests)
    - TestUnitType                          (5 tests)
    - TestTankType                          (5 tests)
    - TestGWPValuesConstant                 (12 tests)
    - TestEPAComponentEmissionFactors       (10 tests)
    - TestEPACorrelationCoefficients        (6 tests)
    - TestIPCCCoalEmissionFactors           (8 tests)
    - TestCoalMethaneFactors                (5 tests)
    - TestWastewaterMCF                     (6 tests)
    - TestPneumaticRates                    (4 tests)
    - TestSourceCategoryMap                 (4 tests)
    - TestSourceDefaultGases                (4 tests)
    - TestFugitiveSourceInfoModel           (6 tests)
    - TestComponentRecordModel              (6 tests)
    - TestEmissionFactorRecordModel         (5 tests)
    - TestSurveyRecordModel                 (5 tests)
    - TestLeakRecordModel                   (4 tests)
    - TestRepairRecordModel                 (4 tests)
    - TestCalculationRequestModel           (5 tests)

Total: 150+ tests.

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-005 Fugitive Emissions (GL-MRV-SCOPE1-005)
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from decimal import Decimal

import pytest
from pydantic import ValidationError

from greenlang.fugitive_emissions.models import (
    # Enumerations
    FugitiveSourceCategory,
    FugitiveSourceType,
    ComponentType,
    ServiceType,
    EmissionGas,
    CalculationMethod,
    EmissionFactorSource,
    GWPSource,
    SurveyType,
    LeakStatus,
    CoalRank,
    WastewaterType,
    ComplianceStatus,
    ReportingPeriod,
    UnitType,
    TankType,
    # Constants
    GWP_VALUES,
    EPA_COMPONENT_EMISSION_FACTORS,
    EPA_CORRELATION_COEFFICIENTS,
    IPCC_COAL_EMISSION_FACTORS,
    COAL_METHANE_FACTORS,
    WASTEWATER_MCF,
    PNEUMATIC_RATES_M3_PER_DAY,
    SOURCE_CATEGORY_MAP,
    SOURCE_DEFAULT_GASES,
    # Module-level constants
    VERSION,
    MAX_CALCULATIONS_PER_BATCH,
    MAX_GASES_PER_RESULT,
    MAX_TRACE_STEPS,
    MAX_COMPONENTS_PER_CALC,
    MAX_SURVEYS_PER_FACILITY,
    MAX_LEAKS_PER_SURVEY,
    DEFAULT_LEAK_THRESHOLD_PPM,
    DEFAULT_REPAIR_DEADLINE_DAYS,
    MAX_DELAY_OF_REPAIR_DAYS,
    # Models
    FugitiveSourceInfo,
    ComponentRecord,
    EmissionFactorRecord,
    SurveyRecord,
    LeakRecord,
    RepairRecord,
    CalculationRequest,
)


# ==========================================================================
# Module-level constants tests
# ==========================================================================


class TestModuleConstants:
    """Test module-level constants are correctly set."""

    def test_version(self):
        assert VERSION == "1.0.0"

    def test_max_calculations_per_batch(self):
        assert MAX_CALCULATIONS_PER_BATCH == 10_000

    def test_max_gases_per_result(self):
        assert MAX_GASES_PER_RESULT == 10

    def test_max_trace_steps(self):
        assert MAX_TRACE_STEPS == 200

    def test_max_components_per_calc(self):
        assert MAX_COMPONENTS_PER_CALC == 50_000

    def test_max_surveys_per_facility(self):
        assert MAX_SURVEYS_PER_FACILITY == 5_000

    def test_max_leaks_per_survey(self):
        assert MAX_LEAKS_PER_SURVEY == 10_000

    def test_default_leak_threshold_ppm(self):
        assert DEFAULT_LEAK_THRESHOLD_PPM == 10_000

    def test_default_repair_deadline_days(self):
        assert DEFAULT_REPAIR_DEADLINE_DAYS == 15

    def test_max_delay_of_repair_days(self):
        assert MAX_DELAY_OF_REPAIR_DAYS == 365


# ==========================================================================
# Enum tests
# ==========================================================================


class TestFugitiveSourceCategory:
    """Test FugitiveSourceCategory enum values and membership."""

    @pytest.mark.parametrize("member,value", [
        (FugitiveSourceCategory.OIL_GAS_PRODUCTION, "oil_gas_production"),
        (FugitiveSourceCategory.OIL_GAS_PROCESSING, "oil_gas_processing"),
        (FugitiveSourceCategory.GAS_TRANSMISSION, "gas_transmission"),
        (FugitiveSourceCategory.GAS_DISTRIBUTION, "gas_distribution"),
        (FugitiveSourceCategory.CRUDE_OIL, "crude_oil"),
        (FugitiveSourceCategory.LNG, "lng"),
        (FugitiveSourceCategory.COAL_UNDERGROUND, "coal_underground"),
        (FugitiveSourceCategory.COAL_SURFACE, "coal_surface"),
        (FugitiveSourceCategory.COAL_POST_MINING, "coal_post_mining"),
        (FugitiveSourceCategory.WASTEWATER_INDUSTRIAL, "wastewater_industrial"),
        (FugitiveSourceCategory.WASTEWATER_MUNICIPAL, "wastewater_municipal"),
        (FugitiveSourceCategory.EQUIPMENT_LEAKS, "equipment_leaks"),
        (FugitiveSourceCategory.TANK_STORAGE, "tank_storage"),
        (FugitiveSourceCategory.PNEUMATIC_DEVICES, "pneumatic_devices"),
        (FugitiveSourceCategory.OTHER, "other"),
    ])
    def test_member_value(self, member, value):
        assert member.value == value

    def test_total_members(self):
        assert len(FugitiveSourceCategory) == 15

    def test_is_str_enum(self):
        assert isinstance(FugitiveSourceCategory.OTHER, str)

    def test_str_lookup(self):
        assert FugitiveSourceCategory("oil_gas_production") == FugitiveSourceCategory.OIL_GAS_PRODUCTION

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            FugitiveSourceCategory("nonexistent_category")


class TestFugitiveSourceType:
    """Test FugitiveSourceType enum has all 25 values."""

    @pytest.mark.parametrize("member,value", [
        (FugitiveSourceType.WELLHEAD, "wellhead"),
        (FugitiveSourceType.SEPARATOR, "separator"),
        (FugitiveSourceType.DEHYDRATOR, "dehydrator"),
        (FugitiveSourceType.PNEUMATIC_CONTROLLER_HIGH, "pneumatic_controller_high"),
        (FugitiveSourceType.PNEUMATIC_CONTROLLER_LOW, "pneumatic_controller_low"),
        (FugitiveSourceType.PNEUMATIC_CONTROLLER_INTERMITTENT, "pneumatic_controller_intermittent"),
        (FugitiveSourceType.COMPRESSOR_CENTRIFUGAL, "compressor_centrifugal"),
        (FugitiveSourceType.COMPRESSOR_RECIPROCATING, "compressor_reciprocating"),
        (FugitiveSourceType.ACID_GAS_REMOVAL, "acid_gas_removal"),
        (FugitiveSourceType.GLYCOL_DEHYDRATOR, "glycol_dehydrator"),
        (FugitiveSourceType.PIPELINE_MAIN, "pipeline_main"),
        (FugitiveSourceType.PIPELINE_SERVICE, "pipeline_service"),
        (FugitiveSourceType.METER_REGULATOR, "meter_regulator"),
        (FugitiveSourceType.TANK_FIXED_ROOF, "tank_fixed_roof"),
        (FugitiveSourceType.TANK_FLOATING_ROOF, "tank_floating_roof"),
        (FugitiveSourceType.COAL_MINE_UNDERGROUND, "coal_mine_underground"),
        (FugitiveSourceType.COAL_MINE_SURFACE, "coal_mine_surface"),
        (FugitiveSourceType.COAL_HANDLING, "coal_handling"),
        (FugitiveSourceType.ABANDONED_MINE, "abandoned_mine"),
        (FugitiveSourceType.WASTEWATER_LAGOON, "wastewater_lagoon"),
        (FugitiveSourceType.WASTEWATER_DIGESTER, "wastewater_digester"),
        (FugitiveSourceType.WASTEWATER_AEROBIC, "wastewater_aerobic"),
        (FugitiveSourceType.VALVE_GAS, "valve_gas"),
        (FugitiveSourceType.PUMP_SEAL, "pump_seal"),
        (FugitiveSourceType.COMPRESSOR_SEAL, "compressor_seal"),
    ])
    def test_source_type_value(self, member, value):
        assert member.value == value

    def test_total_source_types(self):
        assert len(FugitiveSourceType) == 26

    def test_flange_connector(self):
        assert FugitiveSourceType.FLANGE_CONNECTOR.value == "flange_connector"

    def test_is_str_enum(self):
        assert isinstance(FugitiveSourceType.WELLHEAD, str)

    def test_str_lookup(self):
        assert FugitiveSourceType("valve_gas") == FugitiveSourceType.VALVE_GAS


class TestComponentType:
    """Test ComponentType enum for all 9 equipment component types."""

    @pytest.mark.parametrize("member,value", [
        (ComponentType.VALVE, "valve"),
        (ComponentType.PUMP, "pump"),
        (ComponentType.COMPRESSOR, "compressor"),
        (ComponentType.PRESSURE_RELIEF_DEVICE, "pressure_relief_device"),
        (ComponentType.CONNECTOR, "connector"),
        (ComponentType.OPEN_ENDED_LINE, "open_ended_line"),
        (ComponentType.SAMPLING_CONNECTION, "sampling_connection"),
        (ComponentType.FLANGE, "flange"),
        (ComponentType.OTHER, "other"),
    ])
    def test_component_type_value(self, member, value):
        assert member.value == value

    def test_total_component_types(self):
        assert len(ComponentType) == 9

    def test_is_str_enum(self):
        assert isinstance(ComponentType.VALVE, str)


class TestServiceType:
    """Test ServiceType enum for 4 service classifications."""

    @pytest.mark.parametrize("member,value", [
        (ServiceType.GAS, "gas"),
        (ServiceType.LIGHT_LIQUID, "light_liquid"),
        (ServiceType.HEAVY_LIQUID, "heavy_liquid"),
        (ServiceType.HYDROGEN, "hydrogen"),
    ])
    def test_service_type_value(self, member, value):
        assert member.value == value

    def test_total_service_types(self):
        assert len(ServiceType) == 4

    def test_is_str_enum(self):
        assert isinstance(ServiceType.GAS, str)


class TestEmissionGas:
    """Test EmissionGas enum for 4 greenhouse gas species."""

    @pytest.mark.parametrize("member,value", [
        (EmissionGas.CH4, "CH4"),
        (EmissionGas.CO2, "CO2"),
        (EmissionGas.N2O, "N2O"),
        (EmissionGas.VOC, "VOC"),
    ])
    def test_emission_gas_value(self, member, value):
        assert member.value == value

    def test_total_gases(self):
        assert len(EmissionGas) == 4

    def test_is_str_enum(self):
        assert isinstance(EmissionGas.CH4, str)

    def test_ch4_uppercase(self):
        assert EmissionGas("CH4") == EmissionGas.CH4

    def test_voc_not_typical_ghg(self):
        """VOC is tracked for air quality but is a valid enum member."""
        assert EmissionGas.VOC.value == "VOC"


class TestCalculationMethod:
    """Test CalculationMethod enum for all 5 EPA methods."""

    @pytest.mark.parametrize("member,value", [
        (CalculationMethod.AVERAGE_EMISSION_FACTOR, "AVERAGE_EMISSION_FACTOR"),
        (CalculationMethod.SCREENING_RANGES, "SCREENING_RANGES"),
        (CalculationMethod.CORRELATION_EQUATION, "CORRELATION_EQUATION"),
        (CalculationMethod.ENGINEERING_ESTIMATE, "ENGINEERING_ESTIMATE"),
        (CalculationMethod.DIRECT_MEASUREMENT, "DIRECT_MEASUREMENT"),
    ])
    def test_method_value(self, member, value):
        assert member.value == value

    def test_total_methods(self):
        assert len(CalculationMethod) == 5

    def test_is_str_enum(self):
        assert isinstance(CalculationMethod.AVERAGE_EMISSION_FACTOR, str)


class TestEmissionFactorSource:
    """Test EmissionFactorSource enum for 6 source authorities."""

    @pytest.mark.parametrize("member,value", [
        (EmissionFactorSource.EPA, "EPA"),
        (EmissionFactorSource.IPCC, "IPCC"),
        (EmissionFactorSource.DEFRA, "DEFRA"),
        (EmissionFactorSource.EU_ETS, "EU_ETS"),
        (EmissionFactorSource.API, "API"),
        (EmissionFactorSource.CUSTOM, "CUSTOM"),
    ])
    def test_ef_source_value(self, member, value):
        assert member.value == value

    def test_total_sources(self):
        assert len(EmissionFactorSource) == 6

    def test_is_str_enum(self):
        assert isinstance(EmissionFactorSource.EPA, str)


class TestGWPSource:
    """Test GWPSource enum for 4 IPCC assessment report editions."""

    @pytest.mark.parametrize("member,value", [
        (GWPSource.AR4, "AR4"),
        (GWPSource.AR5, "AR5"),
        (GWPSource.AR6, "AR6"),
        (GWPSource.AR6_20YR, "AR6_20YR"),
    ])
    def test_gwp_source_value(self, member, value):
        assert member.value == value

    def test_total_gwp_sources(self):
        assert len(GWPSource) == 4

    def test_is_str_enum(self):
        assert isinstance(GWPSource.AR6, str)

    def test_ar6_20yr_distinct_from_ar6(self):
        assert GWPSource.AR6 != GWPSource.AR6_20YR

    def test_str_lookup(self):
        assert GWPSource("AR6_20YR") == GWPSource.AR6_20YR


class TestSurveyType:
    """Test SurveyType enum for 4 LDAR survey methodologies."""

    @pytest.mark.parametrize("member,value", [
        (SurveyType.OGI, "OGI"),
        (SurveyType.METHOD_21, "METHOD_21"),
        (SurveyType.AVO, "AVO"),
        (SurveyType.HI_FLOW, "HI_FLOW"),
    ])
    def test_survey_type_value(self, member, value):
        assert member.value == value

    def test_total_survey_types(self):
        assert len(SurveyType) == 4

    def test_is_str_enum(self):
        assert isinstance(SurveyType.OGI, str)


class TestLeakStatus:
    """Test LeakStatus enum for 5 component leak states."""

    @pytest.mark.parametrize("member,value", [
        (LeakStatus.NO_LEAK, "no_leak"),
        (LeakStatus.LEAK_DETECTED, "leak_detected"),
        (LeakStatus.REPAIR_PENDING, "repair_pending"),
        (LeakStatus.REPAIRED, "repaired"),
        (LeakStatus.DELAY_OF_REPAIR, "delay_of_repair"),
    ])
    def test_leak_status_value(self, member, value):
        assert member.value == value

    def test_total_statuses(self):
        assert len(LeakStatus) == 5

    def test_is_str_enum(self):
        assert isinstance(LeakStatus.NO_LEAK, str)


class TestCoalRank:
    """Test CoalRank enum for 4 coal classifications."""

    @pytest.mark.parametrize("member,value", [
        (CoalRank.ANTHRACITE, "anthracite"),
        (CoalRank.BITUMINOUS, "bituminous"),
        (CoalRank.SUBBITUMINOUS, "subbituminous"),
        (CoalRank.LIGNITE, "lignite"),
    ])
    def test_coal_rank_value(self, member, value):
        assert member.value == value

    def test_total_ranks(self):
        assert len(CoalRank) == 4

    def test_is_str_enum(self):
        assert isinstance(CoalRank.ANTHRACITE, str)


class TestWastewaterType:
    """Test WastewaterType enum for 5 treatment system types."""

    @pytest.mark.parametrize("member,value", [
        (WastewaterType.AEROBIC, "aerobic"),
        (WastewaterType.ANAEROBIC_LAGOON, "anaerobic_lagoon"),
        (WastewaterType.ANAEROBIC_DIGESTER, "anaerobic_digester"),
        (WastewaterType.FACULTATIVE, "facultative"),
        (WastewaterType.SEPTIC, "septic"),
    ])
    def test_wastewater_type_value(self, member, value):
        assert member.value == value

    def test_total_types(self):
        assert len(WastewaterType) == 5

    def test_is_str_enum(self):
        assert isinstance(WastewaterType.AEROBIC, str)


class TestComplianceStatus:
    """Test ComplianceStatus enum for 4 compliance states."""

    @pytest.mark.parametrize("member,value", [
        (ComplianceStatus.COMPLIANT, "compliant"),
        (ComplianceStatus.NON_COMPLIANT, "non_compliant"),
        (ComplianceStatus.PARTIAL, "partial"),
        (ComplianceStatus.NOT_CHECKED, "not_checked"),
    ])
    def test_compliance_status_value(self, member, value):
        assert member.value == value

    def test_total_statuses(self):
        assert len(ComplianceStatus) == 4

    def test_is_str_enum(self):
        assert isinstance(ComplianceStatus.COMPLIANT, str)


class TestReportingPeriod:
    """Test ReportingPeriod enum for 3 temporal aggregations."""

    @pytest.mark.parametrize("member,value", [
        (ReportingPeriod.MONTHLY, "monthly"),
        (ReportingPeriod.QUARTERLY, "quarterly"),
        (ReportingPeriod.ANNUAL, "annual"),
    ])
    def test_reporting_period_value(self, member, value):
        assert member.value == value

    def test_total_periods(self):
        assert len(ReportingPeriod) == 3

    def test_is_str_enum(self):
        assert isinstance(ReportingPeriod.MONTHLY, str)


class TestUnitType:
    """Test UnitType enum for 4 physical unit categories."""

    @pytest.mark.parametrize("member,value", [
        (UnitType.MASS, "mass"),
        (UnitType.VOLUME, "volume"),
        (UnitType.COUNT, "count"),
        (UnitType.TIME, "time"),
    ])
    def test_unit_type_value(self, member, value):
        assert member.value == value

    def test_total_unit_types(self):
        assert len(UnitType) == 4

    def test_is_str_enum(self):
        assert isinstance(UnitType.MASS, str)


class TestTankType:
    """Test TankType enum for 4 storage tank classifications."""

    @pytest.mark.parametrize("member,value", [
        (TankType.FIXED_ROOF, "fixed_roof"),
        (TankType.FLOATING_ROOF_EXTERNAL, "floating_roof_external"),
        (TankType.FLOATING_ROOF_INTERNAL, "floating_roof_internal"),
        (TankType.PRESSURIZED, "pressurized"),
    ])
    def test_tank_type_value(self, member, value):
        assert member.value == value

    def test_total_tank_types(self):
        assert len(TankType) == 4

    def test_is_str_enum(self):
        assert isinstance(TankType.FIXED_ROOF, str)


# ==========================================================================
# Constant lookup table tests
# ==========================================================================


class TestGWPValuesConstant:
    """Test GWP_VALUES lookup table covers all IPCC assessment reports."""

    def test_gwp_values_has_four_sources(self):
        assert len(GWP_VALUES) == 4

    def test_gwp_ar4_ch4(self):
        assert GWP_VALUES["AR4"]["CH4"] == Decimal("25")

    def test_gwp_ar4_co2(self):
        assert GWP_VALUES["AR4"]["CO2"] == Decimal("1")

    def test_gwp_ar4_n2o(self):
        assert GWP_VALUES["AR4"]["N2O"] == Decimal("298")

    def test_gwp_ar5_ch4(self):
        assert GWP_VALUES["AR5"]["CH4"] == Decimal("28")

    def test_gwp_ar5_n2o(self):
        assert GWP_VALUES["AR5"]["N2O"] == Decimal("265")

    def test_gwp_ar6_ch4(self):
        assert GWP_VALUES["AR6"]["CH4"] == Decimal("29.8")

    def test_gwp_ar6_n2o(self):
        assert GWP_VALUES["AR6"]["N2O"] == Decimal("273")

    def test_gwp_ar6_20yr_ch4(self):
        assert GWP_VALUES["AR6_20YR"]["CH4"] == Decimal("82.5")

    def test_gwp_co2_always_one(self):
        for source in GWP_VALUES:
            assert GWP_VALUES[source]["CO2"] == Decimal("1")

    def test_gwp_all_values_positive(self):
        for source, gases in GWP_VALUES.items():
            for gas, gwp in gases.items():
                assert gwp > Decimal("0"), f"{source}/{gas} GWP must be > 0"

    def test_gwp_all_values_are_decimal(self):
        for source, gases in GWP_VALUES.items():
            for gas, gwp in gases.items():
                assert isinstance(gwp, Decimal), (
                    f"{source}/{gas} must be Decimal, got {type(gwp)}"
                )

    def test_gwp_ar6_20yr_ch4_greater_than_ar6(self):
        assert GWP_VALUES["AR6_20YR"]["CH4"] > GWP_VALUES["AR6"]["CH4"]

    def test_gwp_n2o_same_ar6_and_ar6_20yr(self):
        assert GWP_VALUES["AR6"]["N2O"] == GWP_VALUES["AR6_20YR"]["N2O"]

    @pytest.mark.parametrize("source", ["AR4", "AR5", "AR6", "AR6_20YR"])
    def test_each_source_has_three_gases(self, source):
        assert len(GWP_VALUES[source]) == 3
        assert set(GWP_VALUES[source].keys()) == {"CH4", "CO2", "N2O"}


class TestEPAComponentEmissionFactors:
    """Test EPA_COMPONENT_EMISSION_FACTORS lookup table."""

    def test_valve_gas_ef(self):
        assert EPA_COMPONENT_EMISSION_FACTORS[("valve", "gas")] == Decimal("0.00597")

    def test_valve_light_liquid_ef(self):
        assert EPA_COMPONENT_EMISSION_FACTORS[("valve", "light_liquid")] == Decimal("0.00403")

    def test_valve_heavy_liquid_ef(self):
        assert EPA_COMPONENT_EMISSION_FACTORS[("valve", "heavy_liquid")] == Decimal("0.00023")

    def test_pump_light_liquid_ef(self):
        assert EPA_COMPONENT_EMISSION_FACTORS[("pump", "light_liquid")] == Decimal("0.01140")

    def test_compressor_gas_ef(self):
        assert EPA_COMPONENT_EMISSION_FACTORS[("compressor", "gas")] == Decimal("0.22800")

    def test_connector_gas_ef(self):
        assert EPA_COMPONENT_EMISSION_FACTORS[("connector", "gas")] == Decimal("0.00183")

    def test_flange_gas_ef(self):
        assert EPA_COMPONENT_EMISSION_FACTORS[("flange", "gas")] == Decimal("0.00083")

    def test_all_factors_positive(self):
        for key, ef in EPA_COMPONENT_EMISSION_FACTORS.items():
            assert ef >= Decimal("0"), f"EF for {key} must be >= 0"

    def test_all_factors_are_decimal(self):
        for key, ef in EPA_COMPONENT_EMISSION_FACTORS.items():
            assert isinstance(ef, Decimal), (
                f"EF for {key} must be Decimal, got {type(ef)}"
            )

    def test_factor_count_minimum(self):
        # At least 36 factors from source code (9 types x 4 services)
        assert len(EPA_COMPONENT_EMISSION_FACTORS) >= 17

    @pytest.mark.parametrize("component,service", [
        ("valve", "gas"), ("valve", "light_liquid"), ("valve", "heavy_liquid"),
        ("pump", "gas"), ("pump", "light_liquid"),
        ("compressor", "gas"),
        ("connector", "gas"), ("connector", "light_liquid"),
        ("flange", "gas"),
        ("open_ended_line", "gas"),
        ("sampling_connection", "gas"),
        ("other", "gas"),
    ])
    def test_key_exists(self, component, service):
        assert (component, service) in EPA_COMPONENT_EMISSION_FACTORS

    def test_compressor_ef_highest_for_gas_service(self):
        """Compressors should have the highest EF among component types for gas."""
        compressor_ef = EPA_COMPONENT_EMISSION_FACTORS[("compressor", "gas")]
        valve_ef = EPA_COMPONENT_EMISSION_FACTORS[("valve", "gas")]
        connector_ef = EPA_COMPONENT_EMISSION_FACTORS[("connector", "gas")]
        assert compressor_ef > valve_ef
        assert compressor_ef > connector_ef

    def test_heavy_liquid_valve_lower_than_gas(self):
        """Heavy liquid service valve EF should be lower than gas service."""
        heavy = EPA_COMPONENT_EMISSION_FACTORS[("valve", "heavy_liquid")]
        gas = EPA_COMPONENT_EMISSION_FACTORS[("valve", "gas")]
        assert heavy < gas


class TestEPACorrelationCoefficients:
    """Test EPA_CORRELATION_COEFFICIENTS lookup table."""

    def test_valve_gas_coefficients(self):
        slope, intercept = EPA_CORRELATION_COEFFICIENTS[("valve", "gas")]
        assert slope == Decimal("0.7240")
        assert intercept == Decimal("-6.5850")

    def test_pump_gas_coefficients(self):
        slope, intercept = EPA_CORRELATION_COEFFICIENTS[("pump", "gas")]
        assert slope == Decimal("0.8530")
        assert intercept == Decimal("-6.1440")

    def test_compressor_gas_coefficients(self):
        slope, intercept = EPA_CORRELATION_COEFFICIENTS[("compressor", "gas")]
        assert slope == Decimal("0.7060")
        assert intercept == Decimal("-5.2310")

    def test_all_coefficients_are_tuples(self):
        for key, coeff in EPA_CORRELATION_COEFFICIENTS.items():
            assert isinstance(coeff, tuple), (
                f"Coefficients for {key} must be a tuple, got {type(coeff)}"
            )
            assert len(coeff) == 2

    def test_all_coefficient_values_are_decimal(self):
        for key, (slope, intercept) in EPA_CORRELATION_COEFFICIENTS.items():
            assert isinstance(slope, Decimal), (
                f"Slope for {key} must be Decimal"
            )
            assert isinstance(intercept, Decimal), (
                f"Intercept for {key} must be Decimal"
            )

    def test_slopes_are_positive(self):
        for key, (slope, intercept) in EPA_CORRELATION_COEFFICIENTS.items():
            assert slope > Decimal("0"), (
                f"Slope for {key} must be positive, got {slope}"
            )

    def test_minimum_coefficient_count(self):
        assert len(EPA_CORRELATION_COEFFICIENTS) >= 8


class TestIPCCCoalEmissionFactors:
    """Test IPCC_COAL_EMISSION_FACTORS by mining type and coal rank."""

    def test_underground_anthracite(self):
        assert IPCC_COAL_EMISSION_FACTORS["underground"]["anthracite"] == Decimal("18.0")

    def test_underground_bituminous(self):
        assert IPCC_COAL_EMISSION_FACTORS["underground"]["bituminous"] == Decimal("10.0")

    def test_surface_anthracite(self):
        assert IPCC_COAL_EMISSION_FACTORS["surface"]["anthracite"] == Decimal("1.2")

    def test_post_mining_bituminous(self):
        assert IPCC_COAL_EMISSION_FACTORS["post_mining"]["bituminous"] == Decimal("1.5")

    def test_has_three_mining_types(self):
        assert set(IPCC_COAL_EMISSION_FACTORS.keys()) == {
            "underground", "surface", "post_mining"
        }

    @pytest.mark.parametrize("mining_type", ["underground", "surface", "post_mining"])
    def test_each_type_has_four_ranks(self, mining_type):
        factors = IPCC_COAL_EMISSION_FACTORS[mining_type]
        assert set(factors.keys()) == {
            "anthracite", "bituminous", "subbituminous", "lignite"
        }

    def test_underground_higher_than_surface(self):
        """Underground mining should produce more CH4 than surface mining."""
        for rank in ["anthracite", "bituminous", "subbituminous", "lignite"]:
            assert (
                IPCC_COAL_EMISSION_FACTORS["underground"][rank]
                > IPCC_COAL_EMISSION_FACTORS["surface"][rank]
            )

    def test_all_values_positive(self):
        for mining_type, ranks in IPCC_COAL_EMISSION_FACTORS.items():
            for rank, factor in ranks.items():
                assert factor > Decimal("0"), (
                    f"{mining_type}/{rank} factor must be > 0"
                )


class TestCoalMethaneFactors:
    """Test COAL_METHANE_FACTORS constant by coal rank."""

    def test_anthracite(self):
        assert COAL_METHANE_FACTORS["anthracite"] == Decimal("18.0")

    def test_bituminous(self):
        assert COAL_METHANE_FACTORS["bituminous"] == Decimal("10.0")

    def test_subbituminous(self):
        assert COAL_METHANE_FACTORS["subbituminous"] == Decimal("3.0")

    def test_lignite(self):
        assert COAL_METHANE_FACTORS["lignite"] == Decimal("1.0")

    def test_all_values_positive(self):
        for rank, factor in COAL_METHANE_FACTORS.items():
            assert factor > Decimal("0"), f"{rank} factor must be > 0"

    def test_anthracite_highest(self):
        max_factor = max(COAL_METHANE_FACTORS.values())
        assert COAL_METHANE_FACTORS["anthracite"] == max_factor

    def test_lignite_lowest(self):
        min_factor = min(COAL_METHANE_FACTORS.values())
        assert COAL_METHANE_FACTORS["lignite"] == min_factor


class TestWastewaterMCF:
    """Test WASTEWATER_MCF constant for treatment types."""

    def test_aerobic_mcf(self):
        assert WASTEWATER_MCF["aerobic"] == Decimal("0.0")

    def test_anaerobic_lagoon_mcf(self):
        assert WASTEWATER_MCF["anaerobic_lagoon"] == Decimal("0.8")

    def test_anaerobic_digester_mcf(self):
        assert WASTEWATER_MCF["anaerobic_digester"] == Decimal("0.8")

    def test_facultative_mcf(self):
        assert WASTEWATER_MCF["facultative"] == Decimal("0.2")

    def test_septic_mcf(self):
        assert WASTEWATER_MCF["septic"] == Decimal("0.5")

    def test_mcf_range_0_to_1(self):
        for type_name, mcf in WASTEWATER_MCF.items():
            assert Decimal("0") <= mcf <= Decimal("1"), (
                f"MCF for {type_name} must be in [0, 1], got {mcf}"
            )

    def test_total_treatment_types(self):
        assert len(WASTEWATER_MCF) == 5

    def test_anaerobic_highest_mcf(self):
        """Anaerobic systems should have the highest MCF."""
        assert WASTEWATER_MCF["anaerobic_lagoon"] >= WASTEWATER_MCF["facultative"]
        assert WASTEWATER_MCF["anaerobic_lagoon"] >= WASTEWATER_MCF["septic"]


class TestPneumaticRates:
    """Test PNEUMATIC_RATES_M3_PER_DAY constant."""

    def test_high_bleed_rate(self):
        assert PNEUMATIC_RATES_M3_PER_DAY["HIGH_BLEED"] == Decimal("37.8")

    def test_low_bleed_rate(self):
        assert PNEUMATIC_RATES_M3_PER_DAY["LOW_BLEED"] == Decimal("0.945")

    def test_intermittent_rate(self):
        assert PNEUMATIC_RATES_M3_PER_DAY["INTERMITTENT"] == Decimal("9.18")

    def test_all_rates_non_negative(self):
        for device, rate in PNEUMATIC_RATES_M3_PER_DAY.items():
            assert rate >= Decimal("0"), f"Rate for {device} must be >= 0"

    def test_high_bleed_highest(self):
        assert PNEUMATIC_RATES_M3_PER_DAY["HIGH_BLEED"] > PNEUMATIC_RATES_M3_PER_DAY["LOW_BLEED"]
        assert PNEUMATIC_RATES_M3_PER_DAY["HIGH_BLEED"] > PNEUMATIC_RATES_M3_PER_DAY["INTERMITTENT"]

    def test_total_device_types(self):
        assert len(PNEUMATIC_RATES_M3_PER_DAY) == 3


# ==========================================================================
# Source category map and default gas tests
# ==========================================================================


class TestSourceCategoryMap:
    """Test SOURCE_CATEGORY_MAP completeness."""

    def test_has_all_categories(self):
        expected = {
            "oil_gas_production", "oil_gas_processing", "gas_transmission",
            "gas_distribution", "crude_oil", "lng", "coal_underground",
            "coal_surface", "coal_post_mining", "wastewater_industrial",
            "wastewater_municipal", "equipment_leaks", "tank_storage",
            "pneumatic_devices", "other",
        }
        assert set(SOURCE_CATEGORY_MAP.keys()) == expected

    def test_equipment_leaks_has_members(self):
        members = SOURCE_CATEGORY_MAP["equipment_leaks"]
        assert "valve_gas" in members
        assert "pump_seal" in members

    def test_other_is_empty(self):
        assert SOURCE_CATEGORY_MAP["other"] == []

    def test_all_values_are_lists(self):
        for cat, members in SOURCE_CATEGORY_MAP.items():
            assert isinstance(members, list), f"{cat} must map to a list"

    def test_coal_underground_has_member(self):
        assert "coal_mine_underground" in SOURCE_CATEGORY_MAP["coal_underground"]

    def test_tank_storage_members(self):
        members = SOURCE_CATEGORY_MAP["tank_storage"]
        assert "tank_fixed_roof" in members
        assert "tank_floating_roof" in members

    def test_pneumatic_devices_members(self):
        members = SOURCE_CATEGORY_MAP["pneumatic_devices"]
        assert "pneumatic_controller_high" in members
        assert "pneumatic_controller_low" in members
        assert "pneumatic_controller_intermittent" in members

    def test_total_categories(self):
        assert len(SOURCE_CATEGORY_MAP) == 15


class TestSourceDefaultGases:
    """Test SOURCE_DEFAULT_GASES completeness."""

    def test_valve_gas_defaults(self):
        assert "CH4" in SOURCE_DEFAULT_GASES["valve_gas"]
        assert "VOC" in SOURCE_DEFAULT_GASES["valve_gas"]

    def test_coal_mine_underground_defaults(self):
        gases = SOURCE_DEFAULT_GASES["coal_mine_underground"]
        assert "CH4" in gases
        assert "CO2" in gases

    def test_wastewater_aerobic_defaults(self):
        assert "N2O" in SOURCE_DEFAULT_GASES["wastewater_aerobic"]

    def test_all_source_types_have_defaults(self):
        for member in FugitiveSourceType:
            assert member.value in SOURCE_DEFAULT_GASES, (
                f"Missing default gases for {member.value}"
            )

    def test_all_default_gases_are_valid_enum_values(self):
        valid_gas_values = {g.value for g in EmissionGas}
        for source, gases in SOURCE_DEFAULT_GASES.items():
            for gas in gases:
                assert gas in valid_gas_values, (
                    f"Invalid gas '{gas}' for source '{source}'"
                )

    def test_all_sources_have_at_least_one_gas(self):
        for source, gases in SOURCE_DEFAULT_GASES.items():
            assert len(gases) >= 1, (
                f"Source '{source}' must have at least one default gas"
            )

    def test_pneumatic_controller_all_ch4(self):
        """All pneumatic controllers should default to CH4."""
        for key in ["pneumatic_controller_high", "pneumatic_controller_low",
                     "pneumatic_controller_intermittent"]:
            assert "CH4" in SOURCE_DEFAULT_GASES[key]


# ==========================================================================
# Pydantic model tests
# ==========================================================================


class TestFugitiveSourceInfoModel:
    """Test FugitiveSourceInfo Pydantic model."""

    def test_create_valid_source_info(self):
        info = FugitiveSourceInfo(
            source_type=FugitiveSourceType.VALVE_GAS,
            category=FugitiveSourceCategory.EQUIPMENT_LEAKS,
            name="Valve Gas Service",
            description="Test valve",
            primary_gases=[EmissionGas.CH4, EmissionGas.VOC],
            applicable_methods=[CalculationMethod.AVERAGE_EMISSION_FACTOR],
        )
        assert info.source_type == FugitiveSourceType.VALVE_GAS
        assert info.name == "Valve Gas Service"

    def test_frozen_model(self):
        info = FugitiveSourceInfo(
            source_type=FugitiveSourceType.WELLHEAD,
            category=FugitiveSourceCategory.OIL_GAS_PRODUCTION,
            name="Wellhead",
        )
        with pytest.raises(ValidationError):
            info.name = "Changed"

    def test_name_min_length(self):
        with pytest.raises(ValidationError):
            FugitiveSourceInfo(
                source_type=FugitiveSourceType.WELLHEAD,
                category=FugitiveSourceCategory.OIL_GAS_PRODUCTION,
                name="",
            )

    def test_default_supports_ldar_false(self):
        info = FugitiveSourceInfo(
            source_type=FugitiveSourceType.WELLHEAD,
            category=FugitiveSourceCategory.OIL_GAS_PRODUCTION,
            name="Wellhead",
        )
        assert info.supports_ldar is False

    def test_default_supports_direct_measurement_false(self):
        info = FugitiveSourceInfo(
            source_type=FugitiveSourceType.WELLHEAD,
            category=FugitiveSourceCategory.OIL_GAS_PRODUCTION,
            name="Wellhead",
        )
        assert info.supports_direct_measurement is False

    def test_optional_fields_default_none(self):
        info = FugitiveSourceInfo(
            source_type=FugitiveSourceType.WELLHEAD,
            category=FugitiveSourceCategory.OIL_GAS_PRODUCTION,
            name="Wellhead",
        )
        assert info.ipcc_reference is None
        assert info.epa_subpart is None

    def test_primary_gases_default_empty(self):
        info = FugitiveSourceInfo(
            source_type=FugitiveSourceType.WELLHEAD,
            category=FugitiveSourceCategory.OIL_GAS_PRODUCTION,
            name="Wellhead",
        )
        assert info.primary_gases == []

    def test_applicable_methods_default_empty(self):
        info = FugitiveSourceInfo(
            source_type=FugitiveSourceType.WELLHEAD,
            category=FugitiveSourceCategory.OIL_GAS_PRODUCTION,
            name="Wellhead",
        )
        assert info.applicable_methods == []

    def test_with_ipcc_reference(self):
        info = FugitiveSourceInfo(
            source_type=FugitiveSourceType.COAL_MINE_UNDERGROUND,
            category=FugitiveSourceCategory.COAL_UNDERGROUND,
            name="Underground Coal Mine",
            ipcc_reference="Volume 2, Chapter 4, Section 4.1",
            epa_subpart="Subpart FF",
        )
        assert info.ipcc_reference == "Volume 2, Chapter 4, Section 4.1"
        assert info.epa_subpart == "Subpart FF"


class TestComponentRecordModel:
    """Test ComponentRecord Pydantic model."""

    def test_create_valid_component(self):
        rec = ComponentRecord(
            tag_number="V-101-FLG-001",
            component_type=ComponentType.VALVE,
            service_type=ServiceType.GAS,
            facility_id="FAC-001",
        )
        assert rec.tag_number == "V-101-FLG-001"
        assert rec.component_type == ComponentType.VALVE
        assert rec.service_type == ServiceType.GAS

    def test_default_leak_status(self):
        rec = ComponentRecord(
            tag_number="V-001",
            component_type=ComponentType.PUMP,
            service_type=ServiceType.LIGHT_LIQUID,
            facility_id="FAC-001",
        )
        assert rec.leak_status == LeakStatus.NO_LEAK

    def test_default_operating_hours(self):
        rec = ComponentRecord(
            tag_number="V-001",
            component_type=ComponentType.VALVE,
            service_type=ServiceType.GAS,
            facility_id="FAC-001",
        )
        assert rec.operating_hours_per_year == Decimal("8760")

    def test_component_id_auto_generated(self):
        rec = ComponentRecord(
            tag_number="V-001",
            component_type=ComponentType.VALVE,
            service_type=ServiceType.GAS,
            facility_id="FAC-001",
        )
        assert rec.component_id.startswith("comp_")

    def test_tag_number_min_length(self):
        with pytest.raises(ValidationError):
            ComponentRecord(
                tag_number="",
                component_type=ComponentType.VALVE,
                service_type=ServiceType.GAS,
                facility_id="FAC-001",
            )

    def test_screening_value_non_negative(self):
        with pytest.raises(ValidationError):
            ComponentRecord(
                tag_number="V-001",
                component_type=ComponentType.VALVE,
                service_type=ServiceType.GAS,
                facility_id="FAC-001",
                screening_value_ppm=-1.0,
            )

    def test_measured_rate_non_negative(self):
        with pytest.raises(ValidationError):
            ComponentRecord(
                tag_number="V-001",
                component_type=ComponentType.VALVE,
                service_type=ServiceType.GAS,
                facility_id="FAC-001",
                measured_rate_kg_hr=-0.5,
            )

    def test_frozen_model(self):
        rec = ComponentRecord(
            tag_number="V-001",
            component_type=ComponentType.VALVE,
            service_type=ServiceType.GAS,
            facility_id="FAC-001",
        )
        with pytest.raises(ValidationError):
            rec.tag_number = "V-002"

    def test_optional_location(self):
        rec = ComponentRecord(
            tag_number="V-001",
            component_type=ComponentType.VALVE,
            service_type=ServiceType.GAS,
            facility_id="FAC-001",
            location="Process Unit A, Level 2",
        )
        assert rec.location == "Process Unit A, Level 2"


class TestEmissionFactorRecordModel:
    """Test EmissionFactorRecord Pydantic model."""

    def test_create_valid_ef_record(self):
        rec = EmissionFactorRecord(
            source_type=FugitiveSourceType.VALVE_GAS,
            gas=EmissionGas.CH4,
            factor_value=Decimal("0.00597"),
            factor_unit="kg/hr/component",
        )
        assert rec.factor_value == Decimal("0.00597")
        assert rec.source == EmissionFactorSource.EPA

    def test_factor_value_must_be_positive(self):
        with pytest.raises(ValidationError):
            EmissionFactorRecord(
                source_type=FugitiveSourceType.VALVE_GAS,
                gas=EmissionGas.CH4,
                factor_value=Decimal("0"),
                factor_unit="kg/hr",
            )

    def test_default_method(self):
        rec = EmissionFactorRecord(
            source_type=FugitiveSourceType.VALVE_GAS,
            gas=EmissionGas.CH4,
            factor_value=Decimal("1.0"),
            factor_unit="kg/hr",
        )
        assert rec.method == CalculationMethod.AVERAGE_EMISSION_FACTOR

    def test_default_geography_global(self):
        rec = EmissionFactorRecord(
            source_type=FugitiveSourceType.VALVE_GAS,
            gas=EmissionGas.CH4,
            factor_value=Decimal("1.0"),
            factor_unit="kg/hr",
        )
        assert rec.geography == "GLOBAL"

    def test_expiry_before_effective_raises(self):
        now = datetime.now(timezone.utc)
        with pytest.raises(ValidationError, match="expiry_date must be after"):
            EmissionFactorRecord(
                source_type=FugitiveSourceType.VALVE_GAS,
                gas=EmissionGas.CH4,
                factor_value=Decimal("1.0"),
                factor_unit="kg/hr",
                effective_date=now,
                expiry_date=now - timedelta(days=1),
            )

    def test_factor_id_auto_generated(self):
        rec = EmissionFactorRecord(
            source_type=FugitiveSourceType.VALVE_GAS,
            gas=EmissionGas.CH4,
            factor_value=Decimal("1.0"),
            factor_unit="kg/hr",
        )
        assert rec.factor_id.startswith("fef_")

    def test_uncertainty_pct_range(self):
        rec = EmissionFactorRecord(
            source_type=FugitiveSourceType.VALVE_GAS,
            gas=EmissionGas.CH4,
            factor_value=Decimal("1.0"),
            factor_unit="kg/hr",
            uncertainty_pct=Decimal("50.0"),
        )
        assert rec.uncertainty_pct == Decimal("50.0")

    def test_uncertainty_pct_out_of_range(self):
        with pytest.raises(ValidationError):
            EmissionFactorRecord(
                source_type=FugitiveSourceType.VALVE_GAS,
                gas=EmissionGas.CH4,
                factor_value=Decimal("1.0"),
                factor_unit="kg/hr",
                uncertainty_pct=Decimal("101.0"),
            )


class TestSurveyRecordModel:
    """Test SurveyRecord Pydantic model."""

    def test_create_valid_survey(self):
        rec = SurveyRecord(
            survey_type=SurveyType.OGI,
            survey_date=datetime.now(timezone.utc),
            facility_id="FAC-001",
            components_surveyed=500,
            leaks_detected=10,
            coverage_pct=95.0,
        )
        assert rec.survey_type == SurveyType.OGI
        assert rec.leaks_detected == 10

    def test_leaks_cannot_exceed_surveyed(self):
        with pytest.raises(ValidationError, match="leaks_detected"):
            SurveyRecord(
                survey_type=SurveyType.METHOD_21,
                survey_date=datetime.now(timezone.utc),
                facility_id="FAC-001",
                components_surveyed=100,
                leaks_detected=101,
                coverage_pct=100.0,
            )

    def test_coverage_range_validation(self):
        with pytest.raises(ValidationError):
            SurveyRecord(
                survey_type=SurveyType.OGI,
                survey_date=datetime.now(timezone.utc),
                facility_id="FAC-001",
                components_surveyed=100,
                leaks_detected=5,
                coverage_pct=101.0,
            )

    def test_survey_id_auto_generated(self):
        rec = SurveyRecord(
            survey_type=SurveyType.AVO,
            survey_date=datetime.now(timezone.utc),
            facility_id="FAC-001",
            components_surveyed=200,
            leaks_detected=0,
            coverage_pct=50.0,
        )
        assert rec.survey_id.startswith("srv_")

    def test_zero_leaks_valid(self):
        rec = SurveyRecord(
            survey_type=SurveyType.HI_FLOW,
            survey_date=datetime.now(timezone.utc),
            facility_id="FAC-001",
            components_surveyed=50,
            leaks_detected=0,
            coverage_pct=80.0,
        )
        assert rec.leaks_detected == 0

    def test_coverage_zero_valid(self):
        rec = SurveyRecord(
            survey_type=SurveyType.OGI,
            survey_date=datetime.now(timezone.utc),
            facility_id="FAC-001",
            components_surveyed=0,
            leaks_detected=0,
            coverage_pct=0.0,
        )
        assert rec.coverage_pct == 0.0

    def test_negative_coverage_invalid(self):
        with pytest.raises(ValidationError):
            SurveyRecord(
                survey_type=SurveyType.OGI,
                survey_date=datetime.now(timezone.utc),
                facility_id="FAC-001",
                components_surveyed=100,
                leaks_detected=0,
                coverage_pct=-1.0,
            )


class TestLeakRecordModel:
    """Test LeakRecord Pydantic model."""

    def test_create_valid_leak(self):
        rec = LeakRecord(
            component_id="V-101",
            detection_date=datetime.now(timezone.utc),
        )
        assert rec.leak_status == LeakStatus.LEAK_DETECTED

    def test_leak_id_auto_generated(self):
        rec = LeakRecord(
            component_id="V-101",
            detection_date=datetime.now(timezone.utc),
        )
        assert rec.leak_id.startswith("leak_")

    def test_repair_deadline_before_detection_raises(self):
        now = datetime.now(timezone.utc)
        with pytest.raises(ValidationError, match="repair_deadline must be after"):
            LeakRecord(
                component_id="V-101",
                detection_date=now,
                repair_deadline=now - timedelta(days=1),
            )

    def test_screening_value_non_negative(self):
        with pytest.raises(ValidationError):
            LeakRecord(
                component_id="V-101",
                detection_date=datetime.now(timezone.utc),
                screening_value_ppm=-100.0,
            )

    def test_valid_repair_deadline_after_detection(self):
        now = datetime.now(timezone.utc)
        rec = LeakRecord(
            component_id="V-101",
            detection_date=now,
            repair_deadline=now + timedelta(days=15),
        )
        assert rec.repair_deadline > rec.detection_date

    def test_delay_of_repair_justification(self):
        rec = LeakRecord(
            component_id="V-101",
            detection_date=datetime.now(timezone.utc),
            leak_status=LeakStatus.DELAY_OF_REPAIR,
            delay_of_repair_justification="Equipment shutdown required for repair",
        )
        assert rec.delay_of_repair_justification is not None

    def test_measured_rate_non_negative(self):
        with pytest.raises(ValidationError):
            LeakRecord(
                component_id="V-101",
                detection_date=datetime.now(timezone.utc),
                measured_rate_kg_hr=-0.01,
            )


class TestRepairRecordModel:
    """Test RepairRecord Pydantic model."""

    def test_create_valid_repair(self):
        rec = RepairRecord(
            leak_id="leak_abc123",
            repair_date=datetime.now(timezone.utc),
            repair_method="Replace packing",
        )
        assert rec.repair_method == "Replace packing"

    def test_repair_id_auto_generated(self):
        rec = RepairRecord(
            leak_id="leak_abc123",
            repair_date=datetime.now(timezone.utc),
            repair_method="Tighten bolts",
        )
        assert rec.repair_id.startswith("repr_")

    def test_default_is_verified_false(self):
        rec = RepairRecord(
            leak_id="leak_abc123",
            repair_date=datetime.now(timezone.utc),
            repair_method="Replace component",
        )
        assert rec.is_verified is False

    def test_repair_cost_non_negative(self):
        with pytest.raises(ValidationError):
            RepairRecord(
                leak_id="leak_abc123",
                repair_date=datetime.now(timezone.utc),
                repair_method="Test",
                repair_cost_usd=-100.0,
            )

    def test_with_all_optional_fields(self):
        rec = RepairRecord(
            leak_id="leak_abc123",
            repair_date=datetime.now(timezone.utc),
            repair_method="Gasket replacement",
            component_id="V-101",
            pre_repair_rate_kg_hr=0.5,
            post_repair_ppm=100.0,
            post_repair_rate_kg_hr=0.001,
            emissions_reduced_kg_hr=0.499,
            repair_cost_usd=1500.0,
            is_verified=True,
            notes="Gasket replaced with PTFE material",
        )
        assert rec.is_verified is True
        assert rec.repair_cost_usd == 1500.0

    def test_post_repair_ppm_non_negative(self):
        with pytest.raises(ValidationError):
            RepairRecord(
                leak_id="leak_abc123",
                repair_date=datetime.now(timezone.utc),
                repair_method="Test",
                post_repair_ppm=-50.0,
            )

    def test_emissions_reduced_non_negative(self):
        with pytest.raises(ValidationError):
            RepairRecord(
                leak_id="leak_abc123",
                repair_date=datetime.now(timezone.utc),
                repair_method="Test",
                emissions_reduced_kg_hr=-0.1,
            )


class TestCalculationRequestModel:
    """Test CalculationRequest Pydantic model."""

    def test_create_valid_request(self):
        req = CalculationRequest(
            source_type=FugitiveSourceType.VALVE_GAS,
            calculation_method=CalculationMethod.AVERAGE_EMISSION_FACTOR,
            activity_data=Decimal("100"),
            activity_unit="count",
        )
        assert req.source_type == FugitiveSourceType.VALVE_GAS
        assert req.activity_data == Decimal("100")

    def test_default_gwp_source(self):
        req = CalculationRequest(
            source_type=FugitiveSourceType.VALVE_GAS,
            calculation_method=CalculationMethod.AVERAGE_EMISSION_FACTOR,
            activity_data=Decimal("100"),
            activity_unit="count",
        )
        assert req.gwp_source == GWPSource.AR6

    def test_default_gas_composition_ch4(self):
        req = CalculationRequest(
            source_type=FugitiveSourceType.VALVE_GAS,
            calculation_method=CalculationMethod.AVERAGE_EMISSION_FACTOR,
            activity_data=Decimal("100"),
            activity_unit="count",
        )
        assert req.gas_composition_ch4 == Decimal("0.80")

    def test_all_calculation_methods_accepted(self):
        for method in CalculationMethod:
            req = CalculationRequest(
                source_type=FugitiveSourceType.VALVE_GAS,
                calculation_method=method,
                activity_data=Decimal("1"),
                activity_unit="count",
            )
            assert req.calculation_method == method

    def test_all_source_types_accepted(self):
        for st in FugitiveSourceType:
            req = CalculationRequest(
                source_type=st,
                calculation_method=CalculationMethod.AVERAGE_EMISSION_FACTOR,
                activity_data=Decimal("1"),
                activity_unit="count",
            )
            assert req.source_type == st

    def test_activity_data_zero_invalid(self):
        with pytest.raises(ValidationError):
            CalculationRequest(
                source_type=FugitiveSourceType.VALVE_GAS,
                calculation_method=CalculationMethod.AVERAGE_EMISSION_FACTOR,
                activity_data=Decimal("0"),
                activity_unit="count",
            )

    def test_activity_data_negative_invalid(self):
        with pytest.raises(ValidationError):
            CalculationRequest(
                source_type=FugitiveSourceType.VALVE_GAS,
                calculation_method=CalculationMethod.AVERAGE_EMISSION_FACTOR,
                activity_data=Decimal("-10"),
                activity_unit="count",
            )

    def test_default_recovery_rate_zero(self):
        req = CalculationRequest(
            source_type=FugitiveSourceType.VALVE_GAS,
            activity_data=Decimal("100"),
            activity_unit="count",
        )
        assert req.recovery_rate == Decimal("0.0")

    def test_default_operating_hours(self):
        req = CalculationRequest(
            source_type=FugitiveSourceType.VALVE_GAS,
            activity_data=Decimal("100"),
            activity_unit="count",
        )
        assert req.operating_hours == Decimal("8760")

    def test_coal_rank_optional(self):
        req = CalculationRequest(
            source_type=FugitiveSourceType.COAL_MINE_UNDERGROUND,
            calculation_method=CalculationMethod.ENGINEERING_ESTIMATE,
            activity_data=Decimal("10000"),
            activity_unit="tonnes",
            coal_rank=CoalRank.BITUMINOUS,
        )
        assert req.coal_rank == CoalRank.BITUMINOUS

    def test_wastewater_type_optional(self):
        req = CalculationRequest(
            source_type=FugitiveSourceType.WASTEWATER_LAGOON,
            calculation_method=CalculationMethod.ENGINEERING_ESTIMATE,
            activity_data=Decimal("5000"),
            activity_unit="kg_bod",
            wastewater_type=WastewaterType.ANAEROBIC_LAGOON,
        )
        assert req.wastewater_type == WastewaterType.ANAEROBIC_LAGOON

    def test_frozen_model(self):
        req = CalculationRequest(
            source_type=FugitiveSourceType.VALVE_GAS,
            activity_data=Decimal("100"),
            activity_unit="count",
        )
        with pytest.raises(ValidationError):
            req.activity_data = Decimal("200")

    def test_component_counts_optional(self):
        req = CalculationRequest(
            source_type=FugitiveSourceType.VALVE_GAS,
            activity_data=Decimal("100"),
            activity_unit="count",
            component_counts={"valve:gas": 100, "connector:gas": 50},
        )
        assert req.component_counts["valve:gas"] == 100
        assert req.component_counts["connector:gas"] == 50

    def test_gas_composition_range(self):
        with pytest.raises(ValidationError):
            CalculationRequest(
                source_type=FugitiveSourceType.VALVE_GAS,
                activity_data=Decimal("100"),
                activity_unit="count",
                gas_composition_ch4=Decimal("1.5"),
            )
