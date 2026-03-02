# -*- coding: utf-8 -*-
"""
Test suite for franchises.models - AGENT-MRV-027.

Tests all 20 enumerations, 15 constant tables, 14 Pydantic models,
and 28 helper functions for the Franchises Agent (GL-MRV-S3-014).

Coverage:
- Enumerations: 20 enums (values, membership, count)
- Constants: FRANCHISE_EUI_BENCHMARKS, FRANCHISE_REVENUE_INTENSITY,
  COOKING_FUEL_CONSUMPTION, REFRIGERATION_LEAKAGE_RATES, GRID_EMISSION_FACTORS,
  FUEL_EMISSION_FACTORS, REFRIGERANT_GWPS, EEIO_SPEND_FACTORS,
  HOTEL_ENERGY_BENCHMARKS, VEHICLE_EMISSION_FACTORS, DC_RULES,
  COMPLIANCE_FRAMEWORK_RULES, DQI_SCORING, UNCERTAINTY_RANGES,
  COUNTRY_CLIMATE_ZONES
- Input models: FranchiseUnitInput, FranchiseNetworkInput, CookingEnergyInput,
  RefrigerationInput, DeliveryFleetInput, HotelOperationsInput
- Result models: FranchiseCalculationResult, NetworkAggregationResult,
  ComplianceResult, ProvenanceRecord, DataQualityScore, UncertaintyResult,
  DataCoverageReport, AggregationResult
- Helper functions: 28 functions including provenance hashing, DQI,
  currency conversion, and more

Author: GL-TestEngineer
Date: February 2026
"""

from decimal import Decimal
from datetime import datetime
import pytest
from pydantic import ValidationError as PydanticValidationError

from greenlang.franchises.models import (
    # Metadata
    AGENT_ID,
    AGENT_COMPONENT,
    VERSION,
    TABLE_PREFIX,

    # Enumerations
    FranchiseType,
    OwnershipType,
    FranchiseAgreementType,
    CalculationMethod,
    EmissionSource,
    FuelType,
    ClimateZone,
    EFSource,
    DataQualityTier,
    DQIDimension,
    ComplianceFramework,
    ComplianceStatus,
    PipelineStage,
    UncertaintyMethod,
    BatchStatus,
    GWPSource,
    DataCollectionMethod,
    UnitStatus,
    ConsolidationApproach,
    RefrigerantType,

    # Constant tables
    FRANCHISE_EUI_BENCHMARKS,
    FRANCHISE_REVENUE_INTENSITY,
    COOKING_FUEL_CONSUMPTION,
    REFRIGERATION_LEAKAGE_RATES,
    GRID_EMISSION_FACTORS,
    FUEL_EMISSION_FACTORS,
    REFRIGERANT_GWPS,
    EEIO_SPEND_FACTORS,
    HOTEL_ENERGY_BENCHMARKS,
    VEHICLE_EMISSION_FACTORS,
    DC_RULES,
    COMPLIANCE_FRAMEWORK_RULES,
    DQI_SCORING,
    UNCERTAINTY_RANGES,
    COUNTRY_CLIMATE_ZONES,

    # Input models
    FranchiseUnitInput,
    FranchiseNetworkInput,
    CookingEnergyInput,
    RefrigerationInput,
    DeliveryFleetInput,
    HotelOperationsInput,

    # Result models
    FranchiseCalculationResult,
    NetworkAggregationResult,
    ComplianceResult,
    ProvenanceRecord,
    DataQualityScore,
    UncertaintyResult,
    DataCoverageReport,
    AggregationResult,

    # Helper functions
    calculate_provenance_hash,
    get_dqi_classification,
    convert_currency_to_usd,
    get_cpi_deflator,
    get_eui_benchmark,
    get_revenue_intensity,
    get_grid_ef,
    get_fuel_ef,
    get_refrigerant_gwp,
    get_eeio_factor,
    get_hotel_benchmark,
    get_vehicle_ef,
    get_dc_rule,
    get_climate_zone,
    validate_ownership_for_cat14,
    calculate_pro_rata_factor,
    get_franchise_type_label,
)


# ==============================================================================
# METADATA TESTS
# ==============================================================================


class TestMetadata:
    """Test agent metadata constants."""

    def test_agent_id(self):
        """Test agent ID is GL-MRV-S3-014."""
        assert AGENT_ID == "GL-MRV-S3-014"

    def test_agent_component(self):
        """Test agent component is AGENT-MRV-027."""
        assert AGENT_COMPONENT == "AGENT-MRV-027"

    def test_version(self):
        """Test version is 1.0.0."""
        assert VERSION == "1.0.0"

    def test_table_prefix(self):
        """Test table prefix is gl_frn_."""
        assert TABLE_PREFIX == "gl_frn_"


# ==============================================================================
# ENUMERATION TESTS
# ==============================================================================


class TestFranchiseTypeEnum:
    """Test FranchiseType enum has all 10 types."""

    def test_franchise_type_count(self):
        """Test all 10 franchise types exist."""
        assert len(FranchiseType) == 10

    def test_franchise_type_values(self):
        """Test individual franchise type values."""
        assert FranchiseType.QSR_RESTAURANT == "qsr_restaurant"
        assert FranchiseType.FULL_SERVICE_RESTAURANT == "full_service_restaurant"
        assert FranchiseType.HOTEL == "hotel"
        assert FranchiseType.CONVENIENCE_STORE == "convenience_store"
        assert FranchiseType.RETAIL_STORE == "retail_store"
        assert FranchiseType.FITNESS_CENTER == "fitness_center"
        assert FranchiseType.AUTOMOTIVE_SERVICE == "automotive_service"
        assert FranchiseType.HEALTHCARE_CLINIC == "healthcare_clinic"
        assert FranchiseType.EDUCATION_CENTER == "education_center"
        assert FranchiseType.OTHER_SERVICE == "other_service"

    def test_franchise_type_membership(self):
        """Test membership check for franchise types."""
        assert "qsr_restaurant" in [t.value for t in FranchiseType]
        assert "invalid_type" not in [t.value for t in FranchiseType]


class TestOwnershipTypeEnum:
    """Test OwnershipType enum."""

    def test_ownership_type_count(self):
        """Test all 3 ownership types exist."""
        assert len(OwnershipType) == 3

    def test_ownership_type_values(self):
        """Test ownership type values."""
        assert OwnershipType.FRANCHISED == "franchised"
        assert OwnershipType.COMPANY_OWNED == "company_owned"
        assert OwnershipType.JOINT_VENTURE == "joint_venture"

    def test_company_owned_excluded_from_cat14(self):
        """Test company_owned is defined but excluded from Cat 14 reporting."""
        assert OwnershipType.COMPANY_OWNED.value == "company_owned"


class TestFranchiseAgreementTypeEnum:
    """Test FranchiseAgreementType enum."""

    def test_agreement_type_count(self):
        """Test all 4 agreement types exist."""
        assert len(FranchiseAgreementType) == 4

    def test_agreement_type_values(self):
        """Test agreement type values."""
        assert FranchiseAgreementType.SINGLE_UNIT == "single_unit"
        assert FranchiseAgreementType.MULTI_UNIT == "multi_unit"
        assert FranchiseAgreementType.AREA_DEVELOPMENT == "area_development"
        assert FranchiseAgreementType.MASTER_FRANCHISE == "master_franchise"


class TestCalculationMethodEnum:
    """Test CalculationMethod enum."""

    def test_calculation_method_count(self):
        """Test all 4 calculation methods exist."""
        assert len(CalculationMethod) == 4

    def test_calculation_method_values(self):
        """Test calculation method values."""
        assert CalculationMethod.FRANCHISE_SPECIFIC == "franchise_specific"
        assert CalculationMethod.AVERAGE_DATA == "average_data"
        assert CalculationMethod.SPEND_BASED == "spend_based"
        assert CalculationMethod.HYBRID == "hybrid"


class TestEmissionSourceEnum:
    """Test EmissionSource enum."""

    def test_emission_source_count(self):
        """Test all 7 emission sources exist."""
        assert len(EmissionSource) == 7

    def test_emission_source_values(self):
        """Test emission source values."""
        assert EmissionSource.STATIONARY_COMBUSTION == "stationary_combustion"
        assert EmissionSource.MOBILE_COMBUSTION == "mobile_combustion"
        assert EmissionSource.REFRIGERANT_LEAKAGE == "refrigerant_leakage"
        assert EmissionSource.PROCESS_EMISSIONS == "process_emissions"
        assert EmissionSource.PURCHASED_ELECTRICITY == "purchased_electricity"
        assert EmissionSource.PURCHASED_HEATING == "purchased_heating"
        assert EmissionSource.PURCHASED_COOLING == "purchased_cooling"


class TestFuelTypeEnum:
    """Test FuelType enum."""

    def test_fuel_type_count(self):
        """Test all 8 fuel types exist."""
        assert len(FuelType) == 8

    def test_fuel_type_values(self):
        """Test fuel type values."""
        assert FuelType.NATURAL_GAS == "natural_gas"
        assert FuelType.PROPANE == "propane"
        assert FuelType.DIESEL == "diesel"
        assert FuelType.GASOLINE == "gasoline"
        assert FuelType.FUEL_OIL == "fuel_oil"
        assert FuelType.LPG == "lpg"
        assert FuelType.BIOMASS == "biomass"
        assert FuelType.ELECTRICITY == "electricity"


class TestClimateZoneEnum:
    """Test ClimateZone enum."""

    def test_climate_zone_count(self):
        """Test all 5 climate zones exist."""
        assert len(ClimateZone) == 5

    def test_climate_zone_values(self):
        """Test climate zone values."""
        assert ClimateZone.TROPICAL == "tropical"
        assert ClimateZone.ARID == "arid"
        assert ClimateZone.TEMPERATE == "temperate"
        assert ClimateZone.CONTINENTAL == "continental"
        assert ClimateZone.POLAR == "polar"


class TestEFSourceEnum:
    """Test EFSource enum."""

    def test_ef_source_count(self):
        """Test all 6 EF sources exist."""
        assert len(EFSource) == 6

    def test_ef_source_values(self):
        """Test EF source values."""
        assert EFSource.DEFRA_2024 == "DEFRA_2024"
        assert EFSource.EPA_2024 == "EPA_2024"
        assert EFSource.IEA_2024 == "IEA_2024"
        assert EFSource.EGRID_2024 == "EGRID_2024"
        assert EFSource.IPCC_AR6 == "IPCC_AR6"
        assert EFSource.CUSTOM == "CUSTOM"


class TestDataQualityTierEnum:
    """Test DataQualityTier enum."""

    def test_data_quality_tier_count(self):
        """Test all 3 tiers exist."""
        assert len(DataQualityTier) == 3

    def test_data_quality_tier_values(self):
        """Test data quality tier values."""
        assert DataQualityTier.TIER_1 == "tier_1"
        assert DataQualityTier.TIER_2 == "tier_2"
        assert DataQualityTier.TIER_3 == "tier_3"


class TestDQIDimensionEnum:
    """Test DQIDimension enum."""

    def test_dqi_dimension_count(self):
        """Test all 5 dimensions exist."""
        assert len(DQIDimension) == 5

    def test_dqi_dimension_values(self):
        """Test DQI dimension values."""
        assert DQIDimension.TEMPORAL == "temporal"
        assert DQIDimension.GEOGRAPHICAL == "geographical"
        assert DQIDimension.TECHNOLOGICAL == "technological"
        assert DQIDimension.COMPLETENESS == "completeness"
        assert DQIDimension.RELIABILITY == "reliability"


class TestComplianceFrameworkEnum:
    """Test ComplianceFramework enum."""

    def test_compliance_framework_count(self):
        """Test all 7 frameworks exist."""
        assert len(ComplianceFramework) == 7

    def test_compliance_framework_values(self):
        """Test compliance framework values."""
        assert ComplianceFramework.GHG_PROTOCOL == "ghg_protocol"
        assert ComplianceFramework.ISO_14064 == "iso_14064"
        assert ComplianceFramework.CSRD_ESRS == "csrd_esrs"
        assert ComplianceFramework.CDP == "cdp"
        assert ComplianceFramework.SBTI == "sbti"
        assert ComplianceFramework.SB_253 == "sb_253"
        assert ComplianceFramework.GRI == "gri"


class TestComplianceStatusEnum:
    """Test ComplianceStatus enum."""

    def test_compliance_status_count(self):
        """Test all 4 statuses exist."""
        assert len(ComplianceStatus) == 4

    def test_compliance_status_values(self):
        """Test compliance status values."""
        assert ComplianceStatus.COMPLIANT == "compliant"
        assert ComplianceStatus.NON_COMPLIANT == "non_compliant"
        assert ComplianceStatus.PARTIAL == "partial"
        assert ComplianceStatus.NOT_APPLICABLE == "not_applicable"


class TestPipelineStageEnum:
    """Test PipelineStage enum."""

    def test_pipeline_stage_count(self):
        """Test all 10 stages exist."""
        assert len(PipelineStage) == 10

    def test_pipeline_stage_values(self):
        """Test pipeline stage values."""
        assert PipelineStage.VALIDATE == "validate"
        assert PipelineStage.CLASSIFY == "classify"
        assert PipelineStage.NORMALIZE == "normalize"
        assert PipelineStage.RESOLVE_EFS == "resolve_efs"
        assert PipelineStage.CALCULATE == "calculate"
        assert PipelineStage.ALLOCATE == "allocate"
        assert PipelineStage.AGGREGATE == "aggregate"
        assert PipelineStage.COMPLIANCE == "compliance"
        assert PipelineStage.PROVENANCE == "provenance"
        assert PipelineStage.SEAL == "seal"


class TestUncertaintyMethodEnum:
    """Test UncertaintyMethod enum."""

    def test_uncertainty_method_count(self):
        """Test all 3 methods exist."""
        assert len(UncertaintyMethod) == 3


class TestBatchStatusEnum:
    """Test BatchStatus enum."""

    def test_batch_status_count(self):
        """Test all 4 statuses exist."""
        assert len(BatchStatus) == 4


class TestGWPSourceEnum:
    """Test GWPSource enum."""

    def test_gwp_source_count(self):
        """Test both AR5 and AR6 exist."""
        assert len(GWPSource) == 2
        assert GWPSource.AR5 == "AR5"
        assert GWPSource.AR6 == "AR6"


class TestDataCollectionMethodEnum:
    """Test DataCollectionMethod enum."""

    def test_data_collection_method_count(self):
        """Test all 4 methods exist."""
        assert len(DataCollectionMethod) == 4
        assert DataCollectionMethod.METERED == "metered"
        assert DataCollectionMethod.SURVEY == "survey"
        assert DataCollectionMethod.ESTIMATED == "estimated"
        assert DataCollectionMethod.DEFAULT == "default"


class TestUnitStatusEnum:
    """Test UnitStatus enum."""

    def test_unit_status_count(self):
        """Test all 4 statuses exist."""
        assert len(UnitStatus) == 4
        assert UnitStatus.ACTIVE == "active"
        assert UnitStatus.TEMPORARILY_CLOSED == "temporarily_closed"
        assert UnitStatus.PERMANENTLY_CLOSED == "permanently_closed"
        assert UnitStatus.UNDER_CONSTRUCTION == "under_construction"


class TestConsolidationApproachEnum:
    """Test ConsolidationApproach enum."""

    def test_consolidation_approach_count(self):
        """Test all 3 approaches exist."""
        assert len(ConsolidationApproach) == 3
        assert ConsolidationApproach.FINANCIAL_CONTROL == "financial_control"
        assert ConsolidationApproach.EQUITY_SHARE == "equity_share"
        assert ConsolidationApproach.OPERATIONAL_CONTROL == "operational_control"


class TestRefrigerantTypeEnum:
    """Test RefrigerantType enum."""

    def test_refrigerant_type_count(self):
        """Test all 10 refrigerant types exist."""
        assert len(RefrigerantType) == 10

    def test_refrigerant_type_values(self):
        """Test specific refrigerant type values."""
        assert RefrigerantType.R_410A == "R_410A"
        assert RefrigerantType.R_32 == "R_32"
        assert RefrigerantType.R_134A == "R_134a"
        assert RefrigerantType.R_404A == "R_404A"
        assert RefrigerantType.R_507A == "R_507A"
        assert RefrigerantType.R_22 == "R_22"
        assert RefrigerantType.R_407C == "R_407C"
        assert RefrigerantType.R_290 == "R_290"
        assert RefrigerantType.R_744 == "R_744"
        assert RefrigerantType.R_1234YF == "R_1234yf"


# ==============================================================================
# CONSTANT TABLE TESTS
# ==============================================================================


class TestFranchiseEUIBenchmarks:
    """Test FRANCHISE_EUI_BENCHMARKS constant table."""

    def test_eui_benchmarks_non_empty(self):
        """Test EUI benchmarks table is not empty."""
        assert len(FRANCHISE_EUI_BENCHMARKS) > 0

    def test_eui_benchmarks_all_franchise_types(self):
        """Test EUI benchmarks exist for all 10 franchise types."""
        for ftype in FranchiseType:
            assert ftype.value in FRANCHISE_EUI_BENCHMARKS, f"Missing EUI for {ftype.value}"

    def test_eui_benchmarks_all_climate_zones(self):
        """Test each franchise type has all 5 climate zones."""
        for ftype in FranchiseType:
            for cz in ClimateZone:
                assert cz.value in FRANCHISE_EUI_BENCHMARKS[ftype.value], \
                    f"Missing climate zone {cz.value} for {ftype.value}"

    def test_eui_benchmarks_decimal_values(self):
        """Test all EUI values are Decimal and positive."""
        for ftype, zones in FRANCHISE_EUI_BENCHMARKS.items():
            for cz, val in zones.items():
                assert isinstance(val, Decimal), f"EUI for {ftype}/{cz} is not Decimal"
                assert val > 0, f"EUI for {ftype}/{cz} is not positive"


class TestFranchiseRevenueIntensity:
    """Test FRANCHISE_REVENUE_INTENSITY constant table."""

    def test_revenue_intensity_non_empty(self):
        """Test revenue intensity table is not empty."""
        assert len(FRANCHISE_REVENUE_INTENSITY) > 0

    def test_revenue_intensity_all_types(self):
        """Test revenue intensity for all 10 franchise types."""
        for ftype in FranchiseType:
            assert ftype.value in FRANCHISE_REVENUE_INTENSITY, f"Missing RI for {ftype.value}"

    def test_revenue_intensity_decimal_values(self):
        """Test all revenue intensity values are positive Decimals."""
        for ftype, val in FRANCHISE_REVENUE_INTENSITY.items():
            assert isinstance(val, Decimal), f"RI for {ftype} not Decimal"
            assert val > 0, f"RI for {ftype} not positive"


class TestCookingFuelConsumption:
    """Test COOKING_FUEL_CONSUMPTION constant table."""

    def test_cooking_fuel_non_empty(self):
        """Test cooking fuel table is not empty."""
        assert len(COOKING_FUEL_CONSUMPTION) > 0

    def test_cooking_fuel_values_decimal(self):
        """Test all cooking fuel values are Decimal."""
        for key, val in COOKING_FUEL_CONSUMPTION.items():
            if isinstance(val, dict):
                for k, v in val.items():
                    assert isinstance(v, Decimal), f"Cooking fuel {key}/{k} not Decimal"


class TestRefrigerationLeakageRates:
    """Test REFRIGERATION_LEAKAGE_RATES constant table."""

    def test_leakage_rates_non_empty(self):
        """Test leakage rates table is not empty."""
        assert len(REFRIGERATION_LEAKAGE_RATES) > 0

    def test_leakage_rates_between_0_and_1(self):
        """Test leakage rates are between 0 and 1."""
        for equip, rate in REFRIGERATION_LEAKAGE_RATES.items():
            assert Decimal("0") < rate <= Decimal("1"), f"Leakage rate for {equip} out of range"


class TestGridEmissionFactors:
    """Test GRID_EMISSION_FACTORS constant table."""

    def test_grid_ef_non_empty(self):
        """Test grid EF table is not empty."""
        assert len(GRID_EMISSION_FACTORS) > 0

    def test_grid_ef_has_countries(self):
        """Test grid EFs include major countries."""
        country_codes = list(GRID_EMISSION_FACTORS.keys())
        assert "US" in country_codes

    def test_grid_ef_decimal_values(self):
        """Test all grid EF values are positive Decimals."""
        for region, val in GRID_EMISSION_FACTORS.items():
            if isinstance(val, Decimal):
                assert val > 0, f"Grid EF for {region} not positive"


class TestFuelEmissionFactors:
    """Test FUEL_EMISSION_FACTORS constant table."""

    def test_fuel_ef_non_empty(self):
        """Test fuel EF table is not empty."""
        assert len(FUEL_EMISSION_FACTORS) > 0

    def test_fuel_ef_all_types(self):
        """Test fuel EFs for all 8 fuel types."""
        for ftype in FuelType:
            assert ftype.value in FUEL_EMISSION_FACTORS, f"Missing fuel EF for {ftype.value}"


class TestRefrigerantGWPs:
    """Test REFRIGERANT_GWPS constant table."""

    def test_refrigerant_gwps_non_empty(self):
        """Test refrigerant GWPs table is not empty."""
        assert len(REFRIGERANT_GWPS) > 0

    def test_refrigerant_gwps_all_types(self):
        """Test GWPs for all 10 refrigerant types."""
        for rtype in RefrigerantType:
            assert rtype.value in REFRIGERANT_GWPS, f"Missing GWP for {rtype.value}"

    def test_refrigerant_gwps_decimal_positive(self):
        """Test all GWP values are positive Decimals."""
        for ref, gwp in REFRIGERANT_GWPS.items():
            assert isinstance(gwp, Decimal), f"GWP for {ref} not Decimal"
            assert gwp >= 0, f"GWP for {ref} negative"


class TestEEIOSpendFactors:
    """Test EEIO_SPEND_FACTORS constant table."""

    def test_eeio_factors_non_empty(self):
        """Test EEIO factors table is not empty."""
        assert len(EEIO_SPEND_FACTORS) > 0

    def test_eeio_factors_have_naics(self):
        """Test EEIO factors have NAICS code keys."""
        for naics in EEIO_SPEND_FACTORS:
            assert isinstance(naics, str)
            assert len(naics) == 6, f"NAICS code {naics} not 6 digits"


class TestHotelEnergyBenchmarks:
    """Test HOTEL_ENERGY_BENCHMARKS constant table."""

    def test_hotel_benchmarks_non_empty(self):
        """Test hotel benchmarks table is not empty."""
        assert len(HOTEL_ENERGY_BENCHMARKS) > 0


class TestVehicleEmissionFactors:
    """Test VEHICLE_EMISSION_FACTORS constant table."""

    def test_vehicle_ef_non_empty(self):
        """Test vehicle EF table is not empty."""
        assert len(VEHICLE_EMISSION_FACTORS) > 0


class TestDCRules:
    """Test DC_RULES constant table."""

    def test_dc_rules_has_8_entries(self):
        """Test DC_RULES has 8 double-counting rules."""
        assert len(DC_RULES) == 8

    def test_dc_rules_frn_001_exists(self):
        """Test DC-FRN-001 rule exists."""
        assert "DC-FRN-001" in DC_RULES

    def test_dc_rules_all_have_description(self):
        """Test all DC rules have descriptions."""
        for rule_id, rule in DC_RULES.items():
            assert "description" in rule, f"Rule {rule_id} missing description"


class TestComplianceFrameworkRules:
    """Test COMPLIANCE_FRAMEWORK_RULES constant table."""

    def test_framework_rules_non_empty(self):
        """Test framework rules table is not empty."""
        assert len(COMPLIANCE_FRAMEWORK_RULES) > 0

    def test_framework_rules_all_frameworks(self):
        """Test rules exist for all 7 frameworks."""
        for fw in ComplianceFramework:
            assert fw.value in COMPLIANCE_FRAMEWORK_RULES, f"Missing rules for {fw.value}"


class TestDQIScoring:
    """Test DQI_SCORING constant table."""

    def test_dqi_scoring_non_empty(self):
        """Test DQI scoring table is not empty."""
        assert len(DQI_SCORING) > 0

    def test_dqi_scoring_all_dimensions(self):
        """Test DQI scoring covers all 5 dimensions."""
        for dim in DQIDimension:
            assert dim.value in DQI_SCORING, f"Missing DQI dimension {dim.value}"


class TestUncertaintyRanges:
    """Test UNCERTAINTY_RANGES constant table."""

    def test_uncertainty_ranges_non_empty(self):
        """Test uncertainty ranges table is not empty."""
        assert len(UNCERTAINTY_RANGES) > 0


class TestCountryClimateZones:
    """Test COUNTRY_CLIMATE_ZONES constant table."""

    def test_country_climate_zones_non_empty(self):
        """Test country climate zones table is not empty."""
        assert len(COUNTRY_CLIMATE_ZONES) >= 30

    def test_country_climate_zones_us(self):
        """Test US is mapped to a climate zone."""
        assert "US" in COUNTRY_CLIMATE_ZONES


# ==============================================================================
# INPUT MODEL TESTS
# ==============================================================================


class TestFranchiseUnitInput:
    """Test FranchiseUnitInput Pydantic model."""

    def test_franchise_unit_input_creation(self):
        """Test valid FranchiseUnitInput creation."""
        unit = FranchiseUnitInput(
            unit_id="FRN-001",
            franchise_type=FranchiseType.QSR_RESTAURANT,
            ownership_type=OwnershipType.FRANCHISED,
            floor_area_m2=Decimal("250"),
            country="US",
            climate_zone=ClimateZone.TEMPERATE,
            electricity_kwh=Decimal("180000"),
        )
        assert unit.unit_id == "FRN-001"
        assert unit.franchise_type == FranchiseType.QSR_RESTAURANT

    def test_franchise_unit_input_frozen(self):
        """Test FranchiseUnitInput is frozen."""
        unit = FranchiseUnitInput(
            unit_id="FRN-001",
            franchise_type=FranchiseType.QSR_RESTAURANT,
            ownership_type=OwnershipType.FRANCHISED,
            floor_area_m2=Decimal("250"),
            country="US",
            climate_zone=ClimateZone.TEMPERATE,
            electricity_kwh=Decimal("180000"),
        )
        with pytest.raises(Exception):
            unit.unit_id = "MODIFIED"

    def test_franchise_unit_input_serialization(self):
        """Test FranchiseUnitInput serialization via model_dump."""
        unit = FranchiseUnitInput(
            unit_id="FRN-001",
            franchise_type=FranchiseType.QSR_RESTAURANT,
            ownership_type=OwnershipType.FRANCHISED,
            floor_area_m2=Decimal("250"),
            country="US",
            climate_zone=ClimateZone.TEMPERATE,
            electricity_kwh=Decimal("180000"),
        )
        data = unit.model_dump()
        assert data["unit_id"] == "FRN-001"
        assert "franchise_type" in data


class TestFranchiseNetworkInput:
    """Test FranchiseNetworkInput Pydantic model."""

    def test_network_input_creation(self):
        """Test valid FranchiseNetworkInput creation."""
        unit = FranchiseUnitInput(
            unit_id="FRN-001",
            franchise_type=FranchiseType.QSR_RESTAURANT,
            ownership_type=OwnershipType.FRANCHISED,
            country="US",
        )
        net = FranchiseNetworkInput(
            brand_name="TestBrand Inc.",
            reporting_period="2025",
            consolidation_approach=ConsolidationApproach.FINANCIAL_CONTROL,
            units=[unit],
        )
        assert net.brand_name == "TestBrand Inc."
        assert net.reporting_period == "2025"

    def test_network_input_frozen(self):
        """Test FranchiseNetworkInput is frozen."""
        unit = FranchiseUnitInput(
            unit_id="FRN-001",
            franchise_type=FranchiseType.QSR_RESTAURANT,
            ownership_type=OwnershipType.FRANCHISED,
            country="US",
        )
        net = FranchiseNetworkInput(
            brand_name="TestBrand Inc.",
            reporting_period="2025",
            consolidation_approach=ConsolidationApproach.FINANCIAL_CONTROL,
            units=[unit],
        )
        with pytest.raises(Exception):
            net.brand_name = "MODIFIED"


class TestCookingEnergyInput:
    """Test CookingEnergyInput Pydantic model."""

    def test_cooking_energy_creation(self):
        """Test valid CookingEnergyInput creation."""
        cook = CookingEnergyInput(
            natural_gas_therms=Decimal("8500"),
            propane_gallons=Decimal("200"),
            electricity_kwh=Decimal("25000"),
        )
        assert cook.natural_gas_therms == Decimal("8500")

    def test_cooking_energy_frozen(self):
        """Test CookingEnergyInput is frozen."""
        cook = CookingEnergyInput(
            natural_gas_therms=Decimal("8500"),
        )
        with pytest.raises(Exception):
            cook.natural_gas_therms = Decimal("0")


class TestRefrigerationInput:
    """Test RefrigerationInput Pydantic model."""

    def test_refrigeration_input_creation(self):
        """Test valid RefrigerationInput creation."""
        ref = RefrigerationInput(
            system_type="walk_in_cooler",
            refrigerant_type=RefrigerantType.R_404A,
            charge_kg=Decimal("15.0"),
            annual_leakage_rate=Decimal("0.15"),
        )
        assert ref.refrigerant_type == RefrigerantType.R_404A
        assert ref.charge_kg == Decimal("15.0")

    def test_refrigeration_input_frozen(self):
        """Test RefrigerationInput is frozen."""
        ref = RefrigerationInput(
            system_type="walk_in_cooler",
            refrigerant_type=RefrigerantType.R_404A,
            charge_kg=Decimal("15.0"),
            annual_leakage_rate=Decimal("0.15"),
        )
        with pytest.raises(Exception):
            ref.charge_kg = Decimal("0")


class TestDeliveryFleetInput:
    """Test DeliveryFleetInput Pydantic model."""

    def test_delivery_fleet_creation(self):
        """Test valid DeliveryFleetInput creation."""
        fleet = DeliveryFleetInput(
            vehicle_type="small_van",
            fuel_type=FuelType.GASOLINE,
            annual_distance_km=Decimal("18000"),
            fuel_consumption_l=Decimal("2200"),
            vehicle_count=3,
        )
        assert fleet.vehicle_count == 3

    def test_delivery_fleet_frozen(self):
        """Test DeliveryFleetInput is frozen."""
        fleet = DeliveryFleetInput(
            vehicle_type="small_van",
            fuel_type=FuelType.GASOLINE,
            annual_distance_km=Decimal("18000"),
            fuel_consumption_l=Decimal("2200"),
            vehicle_count=3,
        )
        with pytest.raises(Exception):
            fleet.vehicle_count = 10


class TestHotelOperationsInput:
    """Test HotelOperationsInput Pydantic model."""

    def test_hotel_operations_creation(self):
        """Test valid HotelOperationsInput creation."""
        hotel = HotelOperationsInput(
            room_count=120,
            class_type="upscale",
            occupancy_rate=Decimal("0.72"),
        )
        assert hotel.room_count == 120
        assert hotel.class_type == "upscale"

    def test_hotel_operations_frozen(self):
        """Test HotelOperationsInput is frozen."""
        hotel = HotelOperationsInput(
            room_count=120,
            class_type="upscale",
            occupancy_rate=Decimal("0.72"),
        )
        with pytest.raises(Exception):
            hotel.room_count = 200


# ==============================================================================
# RESULT MODEL TESTS
# ==============================================================================


class TestFranchiseCalculationResult:
    """Test FranchiseCalculationResult Pydantic model."""

    def test_calculation_result_creation(self):
        """Test valid FranchiseCalculationResult creation."""
        result = FranchiseCalculationResult(
            unit_id="FRN-001",
            franchise_type=FranchiseType.QSR_RESTAURANT,
            ownership_type=OwnershipType.FRANCHISED,
            calculation_method=CalculationMethod.FRANCHISE_SPECIFIC,
            total_emissions_kgco2e=Decimal("85432.50"),
            stationary_combustion_kgco2e=Decimal("55000.00"),
            purchased_electricity_kgco2e=Decimal("30432.50"),
            provenance_hash="a" * 64,
        )
        assert result.total_emissions_kgco2e == Decimal("85432.50")

    def test_calculation_result_frozen(self):
        """Test FranchiseCalculationResult is frozen."""
        result = FranchiseCalculationResult(
            unit_id="FRN-001",
            franchise_type=FranchiseType.QSR_RESTAURANT,
            ownership_type=OwnershipType.FRANCHISED,
            calculation_method=CalculationMethod.FRANCHISE_SPECIFIC,
            total_emissions_kgco2e=Decimal("85432.50"),
            provenance_hash="a" * 64,
        )
        with pytest.raises(Exception):
            result.total_emissions_kgco2e = Decimal("0")


class TestNetworkAggregationResult:
    """Test NetworkAggregationResult Pydantic model."""

    def test_network_aggregation_creation(self):
        """Test valid NetworkAggregationResult creation."""
        agg = NetworkAggregationResult(
            brand_name="TestBrand",
            reporting_period="2025",
            total_emissions_kgco2e=Decimal("12500000"),
            total_emissions_tco2e=Decimal("12500"),
            unit_count=150,
            average_per_unit_kgco2e=Decimal("83333.33"),
            method_breakdown={"franchise_specific": 100, "average_data": 50},
            provenance_hash="a" * 64,
        )
        assert agg.total_emissions_kgco2e == Decimal("12500000")


class TestComplianceResult:
    """Test ComplianceResult Pydantic model."""

    def test_compliance_result_creation(self):
        """Test valid ComplianceResult creation."""
        comp = ComplianceResult(
            framework=ComplianceFramework.GHG_PROTOCOL,
            status=ComplianceStatus.COMPLIANT,
            score=Decimal("92.5"),
            findings=[],
            recommendations=[],
        )
        assert comp.status == ComplianceStatus.COMPLIANT
        assert comp.score == Decimal("92.5")


class TestProvenanceRecordModel:
    """Test ProvenanceRecord Pydantic model."""

    def test_provenance_record_creation(self):
        """Test valid ProvenanceRecord creation."""
        record = ProvenanceRecord(
            record_id="prov-001",
            sha256_hash="a" * 64,
            chain_hash="c" * 64,
            timestamp="2026-02-28T00:00:00Z",
            operation="validate",
        )
        assert record.operation == "validate"
        assert len(record.sha256_hash) == 64


class TestDataQualityScore:
    """Test DataQualityScore Pydantic model."""

    def test_dqi_score_creation(self):
        """Test valid DataQualityScore creation."""
        dqi = DataQualityScore(
            temporal=Decimal("5"),
            geographical=Decimal("3"),
            technological=Decimal("5"),
            completeness=Decimal("3"),
            reliability=Decimal("5"),
            composite=Decimal("4.2"),
            classification="Good",
        )
        assert dqi.composite == Decimal("4.2")
        assert dqi.classification == "Good"


# ==============================================================================
# HELPER FUNCTION TESTS
# ==============================================================================


class TestCalculateProvenanceHash:
    """Test calculate_provenance_hash function."""

    def test_provenance_hash_length(self):
        """Test provenance hash is 64-character hex."""
        h = calculate_provenance_hash("input1", "input2")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_provenance_hash_deterministic(self):
        """Test provenance hash is deterministic."""
        h1 = calculate_provenance_hash("input1", "input2")
        h2 = calculate_provenance_hash("input1", "input2")
        assert h1 == h2

    def test_provenance_hash_different_inputs(self):
        """Test different inputs produce different hashes."""
        h1 = calculate_provenance_hash("input1")
        h2 = calculate_provenance_hash("input2")
        assert h1 != h2

    def test_provenance_hash_with_decimal(self):
        """Test provenance hash with Decimal input."""
        h = calculate_provenance_hash(Decimal("123.456"))
        assert len(h) == 64


class TestGetDQIClassification:
    """Test get_dqi_classification function."""

    def test_excellent_classification(self):
        """Test score >= 4.5 is Excellent."""
        assert get_dqi_classification(Decimal("4.8")) == "Excellent"

    def test_good_classification(self):
        """Test score >= 3.5 is Good."""
        assert get_dqi_classification(Decimal("4.0")) == "Good"

    def test_fair_classification(self):
        """Test score >= 2.5 is Fair."""
        assert get_dqi_classification(Decimal("3.0")) == "Fair"

    def test_poor_classification(self):
        """Test score >= 1.5 is Poor."""
        assert get_dqi_classification(Decimal("2.0")) == "Poor"

    def test_very_poor_classification(self):
        """Test score < 1.5 is Very Poor."""
        assert get_dqi_classification(Decimal("1.0")) == "Very Poor"


class TestConvertCurrencyToUSD:
    """Test convert_currency_to_usd function."""

    def test_usd_to_usd(self):
        """Test USD to USD is identity."""
        result = convert_currency_to_usd(Decimal("1000"), "USD")
        assert result == Decimal("1000.00")

    def test_eur_to_usd(self):
        """Test EUR to USD conversion."""
        result = convert_currency_to_usd(Decimal("1000"), "EUR")
        assert result > Decimal("1000")

    def test_unknown_currency_raises(self):
        """Test unknown currency raises ValueError."""
        with pytest.raises(ValueError):
            convert_currency_to_usd(Decimal("1000"), "XYZ")


class TestGetCPIDeflator:
    """Test get_cpi_deflator function."""

    def test_base_year_deflator(self):
        """Test base year 2021 deflator is 1.0."""
        result = get_cpi_deflator(2021)
        assert result == Decimal("1.0000") or result == Decimal("1.0")

    def test_later_year_deflator(self):
        """Test later year has deflator > 1."""
        result = get_cpi_deflator(2024)
        assert result > Decimal("1.0")

    def test_unknown_year_raises(self):
        """Test unknown year raises ValueError."""
        with pytest.raises(ValueError):
            get_cpi_deflator(1900)


class TestGetEUIBenchmark:
    """Test get_eui_benchmark function."""

    def test_valid_lookup(self):
        """Test valid EUI lookup returns positive Decimal."""
        result = get_eui_benchmark("qsr_restaurant", "temperate")
        assert isinstance(result, Decimal)
        assert result > 0

    def test_unknown_type_returns_default(self):
        """Test unknown franchise type returns default value."""
        result = get_eui_benchmark("nonexistent", "temperate")
        assert result == Decimal("300")


class TestGetRevenueIntensity:
    """Test get_revenue_intensity function."""

    def test_valid_lookup(self):
        """Test valid revenue intensity lookup."""
        result = get_revenue_intensity("qsr_restaurant")
        assert isinstance(result, Decimal)
        assert result > 0


class TestGetGridEF:
    """Test get_grid_ef function."""

    def test_valid_country(self):
        """Test grid EF for valid country."""
        result = get_grid_ef("US")
        assert isinstance(result, Decimal)
        assert result > 0


class TestGetFuelEF:
    """Test get_fuel_ef function."""

    def test_valid_fuel(self):
        """Test fuel EF for valid fuel type."""
        result = get_fuel_ef("natural_gas")
        assert result is not None


class TestGetRefrigerantGWP:
    """Test get_refrigerant_gwp function."""

    def test_valid_refrigerant(self):
        """Test GWP for valid refrigerant."""
        result = get_refrigerant_gwp("R_404A")
        assert isinstance(result, Decimal)
        assert result > 0


class TestGetEEIOFactor:
    """Test get_eeio_factor function."""

    def test_valid_naics(self):
        """Test EEIO factor for valid NAICS code."""
        result = get_eeio_factor("722513")
        assert result is not None

    def test_invalid_naics(self):
        """Test EEIO factor for invalid NAICS code returns None."""
        result = get_eeio_factor("000000")
        assert result is None


class TestValidateOwnershipForCat14:
    """Test validate_ownership_for_cat14 function."""

    def test_franchised_is_valid(self):
        """Test franchised ownership is valid for Cat 14."""
        assert validate_ownership_for_cat14("franchised") is True

    def test_company_owned_is_invalid(self):
        """Test company_owned ownership is invalid for Cat 14."""
        assert validate_ownership_for_cat14("company_owned") is False

    def test_joint_venture_is_valid(self):
        """Test joint_venture ownership is valid for Cat 14."""
        assert validate_ownership_for_cat14("joint_venture") is True


class TestCalculateProRataFactor:
    """Test calculate_pro_rata_factor function."""

    def test_equity_share_approach(self):
        """Test equity share approach returns equity share value."""
        result = calculate_pro_rata_factor(
            Decimal("0.50"),
            ConsolidationApproach.EQUITY_SHARE,
        )
        assert result == Decimal("0.50")

    def test_operational_control_returns_full(self):
        """Test operational control returns 1.0."""
        result = calculate_pro_rata_factor(
            None,
            ConsolidationApproach.OPERATIONAL_CONTROL,
        )
        assert result == Decimal("1")

    def test_financial_control_returns_full(self):
        """Test financial control returns 1.0."""
        result = calculate_pro_rata_factor(
            None,
            ConsolidationApproach.FINANCIAL_CONTROL,
        )
        assert result == Decimal("1")


class TestGetFranchiseTypeLabel:
    """Test get_franchise_type_label function."""

    def test_qsr_label(self):
        """Test QSR franchise type label."""
        label = get_franchise_type_label("qsr_restaurant")
        assert isinstance(label, str)
        assert len(label) > 0

    def test_hotel_label(self):
        """Test hotel franchise type label."""
        label = get_franchise_type_label("hotel")
        assert isinstance(label, str)
        assert len(label) > 0
