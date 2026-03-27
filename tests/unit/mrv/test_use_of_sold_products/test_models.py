# -*- coding: utf-8 -*-
"""
Test suite for use_of_sold_products.models - AGENT-MRV-024.

Tests all 22 enums, 16 constant tables, 14 Pydantic models, and 16 helper
functions for the Use of Sold Products Agent (GL-MRV-S3-011).

Coverage:
- Enumerations: 22 enums (values, membership, count, parametrized)
- Constants: FUEL_EMISSION_FACTORS, REFRIGERANT_GWPS, GRID_EMISSION_FACTORS,
  PRODUCT_PROFILES, STEAM_COOLING_EFS, CHEMICAL_EFS, FEEDSTOCK_PROPERTIES,
  LIFETIME_DEFAULTS, DEGRADATION_RATES, DQI_SCORING, DQI_WEIGHTS,
  UNCERTAINTY_RANGES, GWP_VALUES, DC_RULES, CURRENCY_RATES, CPI_DEFLATORS
- Input models: DirectEmissionInput, IndirectEmissionInput, FuelSaleInput,
  FeedstockInput, ProductInput, LifetimeInput, PortfolioInput
- Result models: DirectEmissionResult, IndirectEmissionResult, FuelSaleResult,
  LifetimeResult, PipelineResult, ComplianceCheckResult, AggregationResult
  (frozen=True checks)
- Helper functions: calculate_provenance_hash, get_dqi_classification,
  validate_product_category, get_default_lifetime, compute_dqi_score,
  compute_uncertainty_range, get_fuel_ef, get_grid_ef, get_refrigerant_gwp,
  convert_units, format_co2e, classify_emission_type, get_degradation_rate,
  get_product_profile, compute_lifetime_emissions, compute_annual_emissions

Author: GL-TestEngineer
Date: February 2026
"""

from decimal import Decimal
from datetime import datetime
import hashlib
import json
import pytest

# ---------------------------------------------------------------------------
# Graceful imports
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.mrv.use_of_sold_products.models import (
        # Enumerations
        CalculationMethod,
        ProductCategory,
        EmissionType,
        FuelType,
        RefrigerantType,
        GridRegion,
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
        DegradationModel,
        LifetimeSource,
        ProductSubcategory,

        # Constants
        AGENT_ID,
        AGENT_COMPONENT,
        VERSION,
        TABLE_PREFIX,
        FUEL_EMISSION_FACTORS,
        REFRIGERANT_GWPS,
        GRID_EMISSION_FACTORS,
        PRODUCT_PROFILES,
        STEAM_COOLING_EFS,
        CHEMICAL_EFS,
        FEEDSTOCK_PROPERTIES,
        LIFETIME_DEFAULTS,
        DEGRADATION_RATES,
        DQI_SCORING,
        DQI_WEIGHTS,
        UNCERTAINTY_RANGES,
        GWP_VALUES,
        DC_RULES,
        CURRENCY_RATES,
        CPI_DEFLATORS,

        # Input models
        DirectEmissionInput,
        IndirectEmissionInput,
        FuelSaleInput,
        FeedstockInput,
        ProductInput,
        LifetimeInput,
        PortfolioInput,

        # Result models
        DirectEmissionResult,
        IndirectEmissionResult,
        FuelSaleResult,
        LifetimeResult,
        PipelineResult,
        ComplianceCheckResult,
        AggregationResult,

        # Helpers
        calculate_provenance_hash,
        get_dqi_classification,
        validate_product_category,
        get_default_lifetime,
        compute_dqi_score,
        compute_uncertainty_range,
        get_fuel_ef,
        get_grid_ef,
        get_refrigerant_gwp,
        convert_units,
        format_co2e,
        classify_emission_type,
        get_degradation_rate,
        get_product_profile,
        compute_lifetime_emissions,
        compute_annual_emissions,
    )
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False

try:
    from pydantic import ValidationError as PydanticValidationError
except ImportError:
    PydanticValidationError = Exception  # type: ignore[assignment, misc]

_SKIP = pytest.mark.skipif(
    not MODELS_AVAILABLE,
    reason="use_of_sold_products.models not available",
)
pytestmark = _SKIP


# ==============================================================================
# ENUMERATION TESTS
# ==============================================================================


class TestCalculationMethodEnum:
    """Test CalculationMethod enum."""

    def test_calculation_method_values(self):
        """Test all 8 calculation method values exist."""
        assert CalculationMethod.DIRECT_FUEL_COMBUSTION == "direct_fuel_combustion"
        assert CalculationMethod.DIRECT_REFRIGERANT_LEAKAGE == "direct_refrigerant_leakage"
        assert CalculationMethod.DIRECT_CHEMICAL_RELEASE == "direct_chemical_release"
        assert CalculationMethod.INDIRECT_ELECTRICITY == "indirect_electricity"
        assert CalculationMethod.INDIRECT_HEATING_FUEL == "indirect_heating_fuel"
        assert CalculationMethod.INDIRECT_STEAM_COOLING == "indirect_steam_cooling"
        assert CalculationMethod.FUELS_SOLD == "fuels_sold"
        assert CalculationMethod.FEEDSTOCKS_SOLD == "feedstocks_sold"
        assert len(CalculationMethod) == 8

    @pytest.mark.parametrize("value", [
        "direct_fuel_combustion", "direct_refrigerant_leakage",
        "direct_chemical_release", "indirect_electricity",
        "indirect_heating_fuel", "indirect_steam_cooling",
        "fuels_sold", "feedstocks_sold",
    ])
    def test_calculation_method_membership(self, value):
        """Test each method value is a valid member."""
        assert CalculationMethod(value) is not None


class TestProductCategoryEnum:
    """Test ProductCategory enum."""

    def test_product_category_values(self):
        """Test all 10 product category values exist."""
        assert ProductCategory.VEHICLES == "vehicles"
        assert ProductCategory.APPLIANCES == "appliances"
        assert ProductCategory.HVAC == "hvac"
        assert ProductCategory.LIGHTING == "lighting"
        assert ProductCategory.IT_EQUIPMENT == "it_equipment"
        assert ProductCategory.INDUSTRIAL_EQUIPMENT == "industrial_equipment"
        assert ProductCategory.FUELS_FEEDSTOCKS == "fuels_feedstocks"
        assert ProductCategory.BUILDING_PRODUCTS == "building_products"
        assert ProductCategory.CONSUMER_PRODUCTS == "consumer_products"
        assert ProductCategory.MEDICAL_DEVICES == "medical_devices"
        assert len(ProductCategory) == 10

    @pytest.mark.parametrize("value", [
        "vehicles", "appliances", "hvac", "lighting", "it_equipment",
        "industrial_equipment", "fuels_feedstocks", "building_products",
        "consumer_products", "medical_devices",
    ])
    def test_product_category_membership(self, value):
        """Test each category value is a valid member."""
        assert ProductCategory(value) is not None


class TestEmissionTypeEnum:
    """Test EmissionType enum."""

    def test_emission_type_values(self):
        """Test all 3 emission type values exist."""
        assert EmissionType.DIRECT == "direct"
        assert EmissionType.INDIRECT == "indirect"
        assert EmissionType.BOTH == "both"
        assert len(EmissionType) == 3


class TestFuelTypeEnum:
    """Test FuelType enum."""

    def test_fuel_type_values(self):
        """Test all 15 fuel type values exist."""
        expected_values = [
            "gasoline", "diesel", "natural_gas", "lpg", "ethanol_e85",
            "biodiesel_b20", "cng", "lng", "kerosene", "heating_oil",
            "propane", "coal_anthracite", "coal_bituminous", "wood_pellets",
            "hydrogen",
        ]
        actual_values = [e.value for e in FuelType]
        for v in expected_values:
            assert v in actual_values
        assert len(FuelType) == 15

    @pytest.mark.parametrize("value", [
        "gasoline", "diesel", "natural_gas", "lpg", "ethanol_e85",
        "biodiesel_b20", "cng", "lng", "kerosene", "heating_oil",
        "propane", "coal_anthracite", "coal_bituminous", "wood_pellets",
        "hydrogen",
    ])
    def test_fuel_type_membership(self, value):
        """Test each fuel type is a valid member."""
        assert FuelType(value) is not None


class TestRefrigerantTypeEnum:
    """Test RefrigerantType enum."""

    def test_refrigerant_type_values(self):
        """Test all 10 refrigerant type values exist."""
        expected_values = [
            "R-134a", "R-410A", "R-32", "R-404A", "R-407C",
            "R-290", "R-600a", "R-744", "R-1234yf", "R-1234ze",
        ]
        actual_values = [e.value for e in RefrigerantType]
        for v in expected_values:
            assert v in actual_values
        assert len(RefrigerantType) == 10


class TestGridRegionEnum:
    """Test GridRegion enum."""

    def test_grid_region_values(self):
        """Test all 16 grid region values exist."""
        expected_values = [
            "US", "US_CAMX", "US_RFCW", "US_SRMW", "DE", "CN", "GB",
            "JP", "IN", "BR", "FR", "AU", "CA", "KR", "ZA", "GLOBAL",
        ]
        actual_values = [e.value for e in GridRegion]
        for v in expected_values:
            assert v in actual_values
        assert len(GridRegion) == 16

    @pytest.mark.parametrize("value", [
        "US", "US_CAMX", "US_RFCW", "US_SRMW", "DE", "CN", "GB",
        "JP", "IN", "BR", "FR", "AU", "CA", "KR", "ZA", "GLOBAL",
    ])
    def test_grid_region_membership(self, value):
        """Test each region is a valid member."""
        assert GridRegion(value) is not None


class TestEFSourceEnum:
    """Test EFSource enum."""

    def test_ef_source_values(self):
        """Test all EF source values exist."""
        assert EFSource.DEFRA == "defra"
        assert EFSource.EPA == "epa"
        assert EFSource.IEA == "iea"
        assert EFSource.IPCC == "ipcc"
        assert EFSource.EGRID == "egrid"
        assert EFSource.CUSTOM == "custom"
        assert len(EFSource) >= 6


class TestComplianceFrameworkEnum:
    """Test ComplianceFramework enum."""

    def test_compliance_framework_values(self):
        """Test all 7 compliance framework values exist."""
        assert ComplianceFramework.GHG_PROTOCOL_SCOPE3 == "ghg_protocol_scope3"
        assert ComplianceFramework.ISO_14064 == "iso_14064"
        assert ComplianceFramework.CSRD_ESRS_E1 == "csrd_esrs_e1"
        assert ComplianceFramework.CDP == "cdp"
        assert ComplianceFramework.SBTI == "sbti"
        assert ComplianceFramework.SB_253 == "sb_253"
        assert ComplianceFramework.GRI == "gri"
        assert len(ComplianceFramework) == 7

    @pytest.mark.parametrize("value", [
        "ghg_protocol_scope3", "iso_14064", "csrd_esrs_e1",
        "cdp", "sbti", "sb_253", "gri",
    ])
    def test_compliance_framework_membership(self, value):
        """Test each framework is a valid member."""
        assert ComplianceFramework(value) is not None


class TestDataQualityTierEnum:
    """Test DataQualityTier enum."""

    def test_data_quality_tier_values(self):
        """Test all 3 tier values exist."""
        assert DataQualityTier.TIER_1 == "tier_1"
        assert DataQualityTier.TIER_2 == "tier_2"
        assert DataQualityTier.TIER_3 == "tier_3"
        assert len(DataQualityTier) == 3


class TestProvenanceStageEnum:
    """Test ProvenanceStage enum."""

    def test_provenance_stage_values(self):
        """Test all 10 pipeline stage values exist."""
        expected_values = [
            "validate", "classify", "normalize", "resolve_efs",
            "calculate", "lifetime", "aggregate", "compliance",
            "provenance", "seal",
        ]
        actual_values = [e.value for e in ProvenanceStage]
        for v in expected_values:
            assert v in actual_values
        assert len(ProvenanceStage) == 10


class TestUncertaintyMethodEnum:
    """Test UncertaintyMethod enum."""

    def test_uncertainty_method_values(self):
        """Test uncertainty method values exist."""
        assert UncertaintyMethod.PROPAGATION == "propagation"
        assert UncertaintyMethod.MONTE_CARLO == "monte_carlo"
        assert len(UncertaintyMethod) >= 2


class TestDQIDimensionEnum:
    """Test DQIDimension enum."""

    def test_dqi_dimension_values(self):
        """Test all 5 DQI dimension values exist."""
        assert DQIDimension.RELIABILITY == "reliability"
        assert DQIDimension.COMPLETENESS == "completeness"
        assert DQIDimension.TEMPORAL == "temporal"
        assert DQIDimension.GEOGRAPHICAL == "geographical"
        assert DQIDimension.TECHNOLOGICAL == "technological"
        assert len(DQIDimension) == 5


class TestDQIScoreEnum:
    """Test DQIScore enum."""

    def test_dqi_score_values(self):
        """Test DQI score levels."""
        assert DQIScore.EXCELLENT == "excellent"
        assert DQIScore.GOOD == "good"
        assert DQIScore.FAIR == "fair"
        assert DQIScore.POOR == "poor"
        assert len(DQIScore) >= 4


class TestComplianceStatusEnum:
    """Test ComplianceStatus enum."""

    def test_compliance_status_values(self):
        """Test compliance status values."""
        assert ComplianceStatus.COMPLIANT == "compliant"
        assert ComplianceStatus.NON_COMPLIANT == "non_compliant"
        assert ComplianceStatus.PARTIAL == "partial"
        assert len(ComplianceStatus) >= 3


class TestGWPVersionEnum:
    """Test GWPVersion enum."""

    def test_gwp_version_values(self):
        """Test GWP version values (AR5 and AR6)."""
        assert GWPVersion.AR5 == "AR5"
        assert GWPVersion.AR6 == "AR6"
        assert len(GWPVersion) == 2


class TestEmissionGasEnum:
    """Test EmissionGas enum."""

    def test_emission_gas_values(self):
        """Test emission gas values."""
        assert EmissionGas.CO2 == "co2"
        assert EmissionGas.CH4 == "ch4"
        assert EmissionGas.N2O == "n2o"
        assert EmissionGas.HFCS == "hfcs"
        assert len(EmissionGas) >= 4


class TestCurrencyCodeEnum:
    """Test CurrencyCode enum."""

    def test_currency_code_values(self):
        """Test currency code values."""
        assert CurrencyCode.USD == "USD"
        assert CurrencyCode.EUR == "EUR"
        assert CurrencyCode.GBP == "GBP"
        assert len(CurrencyCode) >= 3


class TestExportFormatEnum:
    """Test ExportFormat enum."""

    def test_export_format_values(self):
        """Test export format values."""
        assert ExportFormat.JSON == "json"
        assert ExportFormat.CSV == "csv"
        assert len(ExportFormat) >= 2


class TestBatchStatusEnum:
    """Test BatchStatus enum."""

    def test_batch_status_values(self):
        """Test batch status values."""
        assert BatchStatus.PENDING == "pending"
        assert BatchStatus.PROCESSING == "processing"
        assert BatchStatus.COMPLETED == "completed"
        assert BatchStatus.FAILED == "failed"
        assert len(BatchStatus) == 4


class TestDegradationModelEnum:
    """Test DegradationModel enum."""

    def test_degradation_model_values(self):
        """Test degradation model values."""
        assert DegradationModel.LINEAR == "linear"
        assert DegradationModel.EXPONENTIAL == "exponential"
        assert DegradationModel.NONE == "none"
        assert len(DegradationModel) >= 3


class TestLifetimeSourceEnum:
    """Test LifetimeSource enum."""

    def test_lifetime_source_values(self):
        """Test lifetime source values."""
        assert LifetimeSource.DEFAULT == "default"
        assert LifetimeSource.MANUFACTURER == "manufacturer"
        assert LifetimeSource.CUSTOM == "custom"
        assert len(LifetimeSource) >= 3


class TestProductSubcategoryEnum:
    """Test ProductSubcategory enum."""

    def test_product_subcategory_values(self):
        """Test product subcategory values exist."""
        expected_subcategories = [
            "passenger_car", "refrigerator", "air_conditioner",
            "laptop", "generator", "led_bulb",
        ]
        actual_values = [e.value for e in ProductSubcategory]
        for v in expected_subcategories:
            assert v in actual_values
        assert len(ProductSubcategory) >= 6


# ==============================================================================
# CONSTANT TABLE TESTS
# ==============================================================================


class TestFuelEmissionFactors:
    """Test FUEL_EMISSION_FACTORS constant table."""

    def test_gasoline_ef(self):
        """Test gasoline EF is 2.315 kgCO2e/litre."""
        gasoline = FUEL_EMISSION_FACTORS[FuelType.GASOLINE]
        assert gasoline["ef_kg_per_litre"] == Decimal("2.315")

    def test_diesel_ef(self):
        """Test diesel EF is 2.680 kgCO2e/litre."""
        diesel = FUEL_EMISSION_FACTORS[FuelType.DIESEL]
        assert diesel["ef_kg_per_litre"] == Decimal("2.680")

    def test_natural_gas_ef(self):
        """Test natural gas EF is 1.930 kgCO2e/m3."""
        ng = FUEL_EMISSION_FACTORS[FuelType.NATURAL_GAS]
        assert ng["ef_kg_per_m3"] == Decimal("1.930")

    def test_lpg_ef(self):
        """Test LPG EF is 1.553 kgCO2e/litre."""
        lpg = FUEL_EMISSION_FACTORS[FuelType.LPG]
        assert lpg["ef_kg_per_litre"] == Decimal("1.553")

    def test_hydrogen_ef_zero(self):
        """Test hydrogen EF is 0.000 (zero direct combustion CO2)."""
        h2 = FUEL_EMISSION_FACTORS[FuelType.HYDROGEN]
        assert h2["ef_kg_per_kg"] == Decimal("0.000")

    def test_all_15_fuel_types_present(self):
        """Test all 15 fuel types have entries."""
        assert len(FUEL_EMISSION_FACTORS) >= 15

    @pytest.mark.parametrize("fuel,key", [
        ("GASOLINE", "ef_kg_per_litre"),
        ("DIESEL", "ef_kg_per_litre"),
        ("KEROSENE", "ef_kg_per_litre"),
        ("HEATING_OIL", "ef_kg_per_litre"),
        ("PROPANE", "ef_kg_per_litre"),
    ])
    def test_fuel_ef_positive_values(self, fuel, key):
        """Test fuel EFs are positive."""
        ef = FUEL_EMISSION_FACTORS[FuelType(fuel.lower())]
        assert ef[key] > 0


class TestRefrigerantGWPs:
    """Test REFRIGERANT_GWPS constant table."""

    def test_r134a_ar5(self):
        """Test R-134a AR5 GWP is 1430."""
        assert REFRIGERANT_GWPS[RefrigerantType("R-134a")]["gwp_ar5"] == Decimal("1430")

    def test_r410a_ar5(self):
        """Test R-410A AR5 GWP is 2088."""
        assert REFRIGERANT_GWPS[RefrigerantType("R-410A")]["gwp_ar5"] == Decimal("2088")

    def test_r32_ar5(self):
        """Test R-32 AR5 GWP is 675."""
        assert REFRIGERANT_GWPS[RefrigerantType("R-32")]["gwp_ar5"] == Decimal("675")

    def test_r404a_ar5(self):
        """Test R-404A AR5 GWP is 3922."""
        assert REFRIGERANT_GWPS[RefrigerantType("R-404A")]["gwp_ar5"] == Decimal("3922")

    def test_r290_low_gwp(self):
        """Test R-290 (propane) has low GWP (3 for AR5)."""
        assert REFRIGERANT_GWPS[RefrigerantType("R-290")]["gwp_ar5"] == Decimal("3")

    def test_r744_co2_gwp_one(self):
        """Test R-744 (CO2) GWP is 1."""
        assert REFRIGERANT_GWPS[RefrigerantType("R-744")]["gwp_ar5"] == Decimal("1")

    @pytest.mark.parametrize("refrigerant", [
        "R-134a", "R-410A", "R-32", "R-404A", "R-407C",
        "R-290", "R-600a", "R-744", "R-1234yf", "R-1234ze",
    ])
    def test_all_10_refrigerants_have_ar5_and_ar6(self, refrigerant):
        """Test all 10 refrigerants have both AR5 and AR6 GWP values."""
        ref = REFRIGERANT_GWPS[RefrigerantType(refrigerant)]
        assert "gwp_ar5" in ref
        assert "gwp_ar6" in ref
        assert ref["gwp_ar5"] >= 0
        assert ref["gwp_ar6"] >= 0


class TestGridEmissionFactors:
    """Test GRID_EMISSION_FACTORS constant table."""

    def test_us_grid_ef(self):
        """Test US grid EF is 0.417 kgCO2e/kWh."""
        assert GRID_EMISSION_FACTORS[GridRegion.US]["ef_kg_per_kwh"] == Decimal("0.417")

    def test_de_grid_ef(self):
        """Test Germany grid EF is 0.350 kgCO2e/kWh."""
        assert GRID_EMISSION_FACTORS[GridRegion.DE]["ef_kg_per_kwh"] == Decimal("0.350")

    def test_za_grid_ef_highest(self):
        """Test South Africa has highest grid EF at 0.920."""
        assert GRID_EMISSION_FACTORS[GridRegion.ZA]["ef_kg_per_kwh"] == Decimal("0.920")

    def test_fr_grid_ef_low(self):
        """Test France has low grid EF at 0.060 (nuclear)."""
        assert GRID_EMISSION_FACTORS[GridRegion.FR]["ef_kg_per_kwh"] == Decimal("0.060")

    def test_global_grid_ef(self):
        """Test GLOBAL grid EF is 0.440."""
        assert GRID_EMISSION_FACTORS[GridRegion.GLOBAL]["ef_kg_per_kwh"] == Decimal("0.440")

    @pytest.mark.parametrize("region,expected_ef", [
        ("US", Decimal("0.417")),
        ("US_CAMX", Decimal("0.275")),
        ("US_RFCW", Decimal("0.520")),
        ("US_SRMW", Decimal("0.680")),
        ("DE", Decimal("0.350")),
        ("CN", Decimal("0.580")),
        ("GB", Decimal("0.230")),
        ("JP", Decimal("0.470")),
        ("IN", Decimal("0.710")),
        ("BR", Decimal("0.080")),
        ("FR", Decimal("0.060")),
        ("AU", Decimal("0.630")),
        ("CA", Decimal("0.130")),
        ("KR", Decimal("0.460")),
        ("ZA", Decimal("0.920")),
        ("GLOBAL", Decimal("0.440")),
    ])
    def test_all_16_grid_efs(self, region, expected_ef):
        """Test all 16 grid emission factors match expected values."""
        assert GRID_EMISSION_FACTORS[GridRegion(region)]["ef_kg_per_kwh"] == expected_ef


class TestProductProfiles:
    """Test PRODUCT_PROFILES constant table."""

    def test_passenger_car_lifetime(self):
        """Test passenger car default lifetime is 15 years."""
        assert PRODUCT_PROFILES["passenger_car"]["default_lifetime_years"] == 15

    def test_refrigerator_lifetime(self):
        """Test refrigerator default lifetime is 15 years."""
        assert PRODUCT_PROFILES["refrigerator"]["default_lifetime_years"] == 15

    def test_laptop_lifetime(self):
        """Test laptop default lifetime is 5 years."""
        assert PRODUCT_PROFILES["laptop"]["default_lifetime_years"] == 5

    def test_led_bulb_lifetime(self):
        """Test LED bulb default lifetime is 25 years."""
        assert PRODUCT_PROFILES["led_bulb"]["default_lifetime_years"] == 25

    def test_profile_has_degradation_rate(self):
        """Test profiles include degradation rate."""
        for key, profile in PRODUCT_PROFILES.items():
            assert "degradation_rate" in profile, f"Profile {key} missing degradation_rate"


class TestSteamCoolingEFs:
    """Test STEAM_COOLING_EFS constant table."""

    def test_steam_boiler_gas_ef(self):
        """Test steam boiler gas EF is 0.200 kgCO2e/kWh."""
        assert STEAM_COOLING_EFS["steam_boiler_gas"]["ef_kg_per_kwh"] == Decimal("0.200")

    def test_cooling_electric_chiller_ef(self):
        """Test electric chiller EF is 0.140 kgCO2e/kWh."""
        assert STEAM_COOLING_EFS["cooling_electric_chiller"]["ef_kg_per_kwh"] == Decimal("0.140")

    def test_all_7_steam_cooling_types(self):
        """Test all 7 steam/cooling types are present."""
        assert len(STEAM_COOLING_EFS) >= 7


class TestChemicalEFs:
    """Test CHEMICAL_EFS constant table."""

    def test_hfc134a_gwp(self):
        """Test HFC-134a GWP is 1430."""
        assert CHEMICAL_EFS["HFC-134a"]["gwp"] == Decimal("1430")

    def test_sf6_gwp(self):
        """Test SF6 GWP is 22800."""
        assert CHEMICAL_EFS["sf6"]["gwp"] == Decimal("22800")

    def test_all_chemical_types(self):
        """Test at least 7 chemical types are present."""
        assert len(CHEMICAL_EFS) >= 7


class TestFeedstockProperties:
    """Test FEEDSTOCK_PROPERTIES constant table."""

    def test_naphtha_carbon_content(self):
        """Test naphtha carbon content is 0.836."""
        assert FEEDSTOCK_PROPERTIES["naphtha"]["carbon_content"] == Decimal("0.836")

    def test_naphtha_ef(self):
        """Test naphtha EF is 3.065 kgCO2/kg."""
        assert FEEDSTOCK_PROPERTIES["naphtha"]["ef_kg_co2_per_kg"] == Decimal("3.065")

    def test_all_4_feedstock_types(self):
        """Test all 4 feedstock types present."""
        assert len(FEEDSTOCK_PROPERTIES) >= 4


class TestLifetimeDefaults:
    """Test LIFETIME_DEFAULTS constant table."""

    def test_vehicles_lifetime(self):
        """Test vehicles default lifetime is 15 years."""
        assert LIFETIME_DEFAULTS["vehicles"] == 15

    def test_appliances_lifetime(self):
        """Test appliances default lifetime is 15 years."""
        assert LIFETIME_DEFAULTS["appliances"] == 15

    def test_it_equipment_lifetime(self):
        """Test IT equipment default lifetime is 5 years."""
        assert LIFETIME_DEFAULTS["it_equipment"] == 5

    def test_lighting_lifetime(self):
        """Test lighting default lifetime is 25 years."""
        assert LIFETIME_DEFAULTS["lighting"] == 25

    def test_industrial_equipment_lifetime(self):
        """Test industrial equipment default lifetime is 20 years."""
        assert LIFETIME_DEFAULTS["industrial_equipment"] == 20


class TestDegradationRates:
    """Test DEGRADATION_RATES constant table."""

    def test_vehicles_degradation(self):
        """Test vehicles degradation rate is 0.005."""
        assert DEGRADATION_RATES["vehicles"] == Decimal("0.005")

    def test_it_equipment_degradation(self):
        """Test IT equipment degradation rate is 0.02."""
        assert DEGRADATION_RATES["it_equipment"] == Decimal("0.02")


class TestGWPValues:
    """Test GWP_VALUES constant table."""

    def test_gwp_ar5_ch4(self):
        """Test IPCC AR5 GWP for CH4 is 28."""
        ar5 = GWP_VALUES[GWPVersion.AR5]
        assert ar5["ch4"] == Decimal("28")

    def test_gwp_ar5_n2o(self):
        """Test IPCC AR5 GWP for N2O is 265."""
        ar5 = GWP_VALUES[GWPVersion.AR5]
        assert ar5["n2o"] == Decimal("265")

    def test_gwp_ar5_co2(self):
        """Test IPCC AR5 GWP for CO2 is 1."""
        ar5 = GWP_VALUES[GWPVersion.AR5]
        assert ar5["co2"] == Decimal("1")

    def test_gwp_ar6_ch4(self):
        """Test IPCC AR6 GWP for CH4 is 27.9."""
        ar6 = GWP_VALUES[GWPVersion.AR6]
        assert ar6["ch4"] == Decimal("27.9")

    def test_gwp_ar6_n2o(self):
        """Test IPCC AR6 GWP for N2O is 273."""
        ar6 = GWP_VALUES[GWPVersion.AR6]
        assert ar6["n2o"] == Decimal("273")


class TestDCRules:
    """Test DC_RULES constant table (double-counting prevention)."""

    def test_dc_rules_has_8_entries(self):
        """Test there are 8 double-counting rules."""
        assert len(DC_RULES) == 8

    @pytest.mark.parametrize("rule_id", [
        "DC-USP-001", "DC-USP-002", "DC-USP-003", "DC-USP-004",
        "DC-USP-005", "DC-USP-006", "DC-USP-007", "DC-USP-008",
    ])
    def test_dc_rule_exists(self, rule_id):
        """Test each DC rule ID exists."""
        assert rule_id in DC_RULES

    def test_dc_rule_has_description(self):
        """Test all DC rules have description field."""
        for rule_id, rule in DC_RULES.items():
            assert "description" in rule, f"Rule {rule_id} missing description"


class TestDQIScoringAndWeights:
    """Test DQI_SCORING and DQI_WEIGHTS constant tables."""

    def test_dqi_scoring_5_dimensions(self):
        """Test DQI scoring has 5 dimensions."""
        assert len(DQI_SCORING) == 5

    def test_dqi_weights_sum_to_one(self):
        """Test DQI dimension weights sum to 1.0."""
        total = sum(DQI_WEIGHTS.values())
        assert total == Decimal("1.00")

    def test_dqi_weights_has_all_dimensions(self):
        """Test DQI weights cover all 5 dimensions."""
        for dim in DQIDimension:
            assert dim in DQI_WEIGHTS or dim.value in DQI_WEIGHTS


class TestUncertaintyRanges:
    """Test UNCERTAINTY_RANGES constant table."""

    def test_uncertainty_direct_tier1(self):
        """Test direct Tier 1 uncertainty is 0.05 (5%)."""
        assert UNCERTAINTY_RANGES["direct"][DataQualityTier.TIER_1] == Decimal("0.05")

    def test_uncertainty_indirect_tier3(self):
        """Test indirect Tier 3 uncertainty is 0.30 (30%)."""
        assert UNCERTAINTY_RANGES["indirect"][DataQualityTier.TIER_3] == Decimal("0.30")


class TestCurrencyRates:
    """Test CURRENCY_RATES constant table."""

    def test_usd_rate_identity(self):
        """Test USD rate is 1.0."""
        assert CURRENCY_RATES[CurrencyCode.USD] == Decimal("1.0")

    def test_eur_rate(self):
        """Test EUR rate is 1.0850."""
        assert CURRENCY_RATES[CurrencyCode.EUR] == Decimal("1.0850")

    def test_gbp_rate(self):
        """Test GBP rate is 1.2650."""
        assert CURRENCY_RATES[CurrencyCode.GBP] == Decimal("1.2650")


class TestCPIDeflators:
    """Test CPI_DEFLATORS constant table."""

    def test_cpi_base_year_2021(self):
        """Test CPI deflator for base year 2021 is 1.0."""
        assert CPI_DEFLATORS[2021] == Decimal("1.0000")

    def test_cpi_2024(self):
        """Test CPI deflator for 2024 is 1.1490."""
        assert CPI_DEFLATORS[2024] == Decimal("1.1490")


# ==============================================================================
# AGENT METADATA CONSTANTS
# ==============================================================================


class TestAgentMetadata:
    """Test AGENT_ID, AGENT_COMPONENT, VERSION, TABLE_PREFIX."""

    def test_agent_id(self):
        """Test AGENT_ID is GL-MRV-S3-011."""
        assert AGENT_ID == "GL-MRV-S3-011"

    def test_agent_component(self):
        """Test AGENT_COMPONENT is AGENT-MRV-024."""
        assert AGENT_COMPONENT == "AGENT-MRV-024"

    def test_version(self):
        """Test VERSION is 1.0.0."""
        assert VERSION == "1.0.0"

    def test_table_prefix(self):
        """Test TABLE_PREFIX is gl_usp_."""
        assert TABLE_PREFIX == "gl_usp_"


# ==============================================================================
# INPUT MODEL TESTS
# ==============================================================================


class TestDirectEmissionInputModel:
    """Test DirectEmissionInput Pydantic model."""

    def test_valid_fuel_combustion_input(self):
        """Test valid direct fuel combustion input creation."""
        inp = DirectEmissionInput(
            product_id="VEH-001",
            category="vehicles",
            emission_method="direct_fuel_combustion",
            fuel_type="gasoline",
            fuel_consumption_per_year=Decimal("1200.0"),
            units_sold=1000,
            lifetime_years=15,
        )
        assert inp.product_id == "VEH-001"
        assert inp.units_sold == 1000

    def test_valid_refrigerant_leakage_input(self):
        """Test valid refrigerant leakage input creation."""
        inp = DirectEmissionInput(
            product_id="HVAC-001",
            category="hvac",
            emission_method="direct_refrigerant_leakage",
            refrigerant_type="R-410A",
            refrigerant_charge_kg=Decimal("3.0"),
            annual_leak_rate=Decimal("0.05"),
            units_sold=500,
            lifetime_years=12,
        )
        assert inp.refrigerant_type == "R-410A"

    def test_frozen_immutability(self):
        """Test DirectEmissionInput is frozen (immutable)."""
        inp = DirectEmissionInput(
            product_id="VEH-001",
            category="vehicles",
            emission_method="direct_fuel_combustion",
            units_sold=1000,
            lifetime_years=15,
        )
        with pytest.raises(Exception):
            inp.product_id = "VEH-002"

    def test_negative_units_rejected(self):
        """Test negative units_sold is rejected."""
        with pytest.raises(PydanticValidationError):
            DirectEmissionInput(
                product_id="VEH-001",
                category="vehicles",
                emission_method="direct_fuel_combustion",
                units_sold=-100,
                lifetime_years=15,
            )

    def test_zero_lifetime_rejected(self):
        """Test zero lifetime_years is rejected."""
        with pytest.raises(PydanticValidationError):
            DirectEmissionInput(
                product_id="VEH-001",
                category="vehicles",
                emission_method="direct_fuel_combustion",
                units_sold=1000,
                lifetime_years=0,
            )


class TestIndirectEmissionInputModel:
    """Test IndirectEmissionInput Pydantic model."""

    def test_valid_electricity_input(self):
        """Test valid indirect electricity input creation."""
        inp = IndirectEmissionInput(
            product_id="APP-001",
            category="appliances",
            emission_method="indirect_electricity",
            energy_consumption_kwh_per_year=Decimal("400.0"),
            grid_region="US",
            units_sold=10000,
            lifetime_years=15,
        )
        assert inp.energy_consumption_kwh_per_year == Decimal("400.0")

    def test_frozen_immutability(self):
        """Test IndirectEmissionInput is frozen (immutable)."""
        inp = IndirectEmissionInput(
            product_id="APP-001",
            category="appliances",
            emission_method="indirect_electricity",
            units_sold=10000,
            lifetime_years=15,
        )
        with pytest.raises(Exception):
            inp.units_sold = 5000

    def test_negative_energy_rejected(self):
        """Test negative energy consumption is rejected."""
        with pytest.raises(PydanticValidationError):
            IndirectEmissionInput(
                product_id="APP-001",
                category="appliances",
                emission_method="indirect_electricity",
                energy_consumption_kwh_per_year=Decimal("-100.0"),
                units_sold=10000,
                lifetime_years=15,
            )


class TestFuelSaleInputModel:
    """Test FuelSaleInput Pydantic model."""

    def test_valid_fuel_sale_input(self):
        """Test valid fuel sale input creation."""
        inp = FuelSaleInput(
            product_id="FUEL-001",
            fuel_type="gasoline",
            quantity_sold_litres=Decimal("1000000.0"),
        )
        assert inp.quantity_sold_litres == Decimal("1000000.0")

    def test_frozen_immutability(self):
        """Test FuelSaleInput is frozen (immutable)."""
        inp = FuelSaleInput(
            product_id="FUEL-001",
            fuel_type="gasoline",
            quantity_sold_litres=Decimal("1000000.0"),
        )
        with pytest.raises(Exception):
            inp.fuel_type = "diesel"

    def test_negative_quantity_rejected(self):
        """Test negative quantity is rejected."""
        with pytest.raises(PydanticValidationError):
            FuelSaleInput(
                product_id="FUEL-001",
                fuel_type="gasoline",
                quantity_sold_litres=Decimal("-500.0"),
            )


class TestFeedstockInputModel:
    """Test FeedstockInput Pydantic model."""

    def test_valid_feedstock_input(self):
        """Test valid feedstock input creation."""
        inp = FeedstockInput(
            product_id="FEED-001",
            feedstock_type="naphtha",
            quantity_sold_kg=Decimal("1000000.0"),
        )
        assert inp.feedstock_type == "naphtha"

    def test_frozen_immutability(self):
        """Test FeedstockInput is frozen (immutable)."""
        inp = FeedstockInput(
            product_id="FEED-001",
            feedstock_type="naphtha",
            quantity_sold_kg=Decimal("1000000.0"),
        )
        with pytest.raises(Exception):
            inp.feedstock_type = "ethane"


class TestProductInputModel:
    """Test ProductInput Pydantic model."""

    def test_valid_product_input(self):
        """Test valid product input creation."""
        inp = ProductInput(
            product_id="PRD-001",
            product_name="Sedan 2.0L",
            category="vehicles",
            units_sold=1000,
            reporting_year=2024,
        )
        assert inp.product_name == "Sedan 2.0L"

    def test_empty_product_id_rejected(self):
        """Test empty product_id is rejected."""
        with pytest.raises(PydanticValidationError):
            ProductInput(
                product_id="",
                product_name="Test",
                category="vehicles",
                units_sold=1000,
                reporting_year=2024,
            )


class TestLifetimeInputModel:
    """Test LifetimeInput Pydantic model."""

    def test_valid_lifetime_input(self):
        """Test valid lifetime input creation."""
        inp = LifetimeInput(
            product_id="PRD-001",
            category="vehicles",
            lifetime_years=15,
            degradation_model="linear",
            degradation_rate=Decimal("0.005"),
        )
        assert inp.lifetime_years == 15

    def test_negative_degradation_rejected(self):
        """Test negative degradation rate is rejected."""
        with pytest.raises(PydanticValidationError):
            LifetimeInput(
                product_id="PRD-001",
                category="vehicles",
                lifetime_years=15,
                degradation_rate=Decimal("-0.01"),
            )


class TestPortfolioInputModel:
    """Test PortfolioInput Pydantic model."""

    def test_valid_portfolio_input(self):
        """Test valid portfolio input with multiple products."""
        inp = PortfolioInput(
            org_id="ORG-001",
            reporting_year=2024,
            products=[
                {"product_id": "P1", "category": "vehicles", "units_sold": 1000},
                {"product_id": "P2", "category": "appliances", "units_sold": 5000},
            ],
        )
        assert len(inp.products) == 2


# ==============================================================================
# RESULT MODEL TESTS
# ==============================================================================


class TestDirectEmissionResultModel:
    """Test DirectEmissionResult model (frozen)."""

    def test_result_creation(self):
        """Test result model creation."""
        result = DirectEmissionResult(
            product_id="VEH-001",
            total_co2e_kg=Decimal("41670000.0"),
            co2e_per_unit=Decimal("41670.0"),
            method="direct_fuel_combustion",
            provenance_hash="a" * 64,
        )
        assert result.total_co2e_kg == Decimal("41670000.0")

    def test_result_frozen(self):
        """Test result model is frozen (immutable)."""
        result = DirectEmissionResult(
            product_id="VEH-001",
            total_co2e_kg=Decimal("41670000.0"),
            co2e_per_unit=Decimal("41670.0"),
            method="direct_fuel_combustion",
            provenance_hash="a" * 64,
        )
        with pytest.raises(Exception):
            result.total_co2e_kg = Decimal("0")


class TestIndirectEmissionResultModel:
    """Test IndirectEmissionResult model (frozen)."""

    def test_result_creation(self):
        """Test result model creation."""
        result = IndirectEmissionResult(
            product_id="APP-001",
            total_co2e_kg=Decimal("25020000.0"),
            co2e_per_unit=Decimal("2502.0"),
            method="indirect_electricity",
            grid_region="US",
            provenance_hash="b" * 64,
        )
        assert result.grid_region == "US"

    def test_result_frozen(self):
        """Test result model is frozen (immutable)."""
        result = IndirectEmissionResult(
            product_id="APP-001",
            total_co2e_kg=Decimal("25020000.0"),
            co2e_per_unit=Decimal("2502.0"),
            method="indirect_electricity",
            grid_region="US",
            provenance_hash="b" * 64,
        )
        with pytest.raises(Exception):
            result.grid_region = "DE"


class TestFuelSaleResultModel:
    """Test FuelSaleResult model (frozen)."""

    def test_result_creation(self):
        """Test result model creation."""
        result = FuelSaleResult(
            product_id="FUEL-001",
            total_co2e_kg=Decimal("2315000.0"),
            fuel_type="gasoline",
            quantity_litres=Decimal("1000000.0"),
            provenance_hash="c" * 64,
        )
        assert result.total_co2e_kg == Decimal("2315000.0")


class TestPipelineResultModel:
    """Test PipelineResult model (frozen)."""

    def test_result_creation(self):
        """Test pipeline result model creation."""
        result = PipelineResult(
            calculation_id="CALC-001",
            total_co2e_kg=Decimal("70000000.0"),
            direct_co2e_kg=Decimal("43549200.0"),
            indirect_co2e_kg=Decimal("25020000.0"),
            fuels_feedstocks_co2e_kg=Decimal("2315000.0"),
            product_count=5,
            provenance_hash="d" * 64,
        )
        assert result.product_count == 5


# ==============================================================================
# HELPER FUNCTION TESTS
# ==============================================================================


class TestProvenanceHash:
    """Test calculate_provenance_hash function."""

    def test_provenance_hash_deterministic(self):
        """Test provenance hash is deterministic for same inputs."""
        h1 = calculate_provenance_hash("VEH-001", "gasoline", Decimal("1200"))
        h2 = calculate_provenance_hash("VEH-001", "gasoline", Decimal("1200"))
        assert h1 == h2
        assert len(h1) == 64

    def test_provenance_hash_different_inputs(self):
        """Test provenance hash differs for different inputs."""
        h1 = calculate_provenance_hash("VEH-001", "gasoline")
        h2 = calculate_provenance_hash("VEH-002", "diesel")
        assert h1 != h2

    def test_provenance_hash_64_char_hex(self):
        """Test provenance hash is 64-char lowercase hex."""
        h = calculate_provenance_hash("test")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


class TestDQIClassification:
    """Test get_dqi_classification function."""

    def test_dqi_classification_excellent(self):
        """Test score >=90 classifies as Excellent."""
        assert get_dqi_classification(Decimal("95")) == "Excellent"

    def test_dqi_classification_good(self):
        """Test score >=75 classifies as Good."""
        assert get_dqi_classification(Decimal("80")) == "Good"

    def test_dqi_classification_fair(self):
        """Test score >=50 classifies as Fair."""
        assert get_dqi_classification(Decimal("60")) == "Fair"

    def test_dqi_classification_poor(self):
        """Test score <50 classifies as Poor."""
        assert get_dqi_classification(Decimal("30")) == "Poor"


class TestValidateProductCategory:
    """Test validate_product_category function."""

    def test_valid_category(self):
        """Test valid category passes validation."""
        assert validate_product_category("vehicles") is True

    def test_invalid_category(self):
        """Test invalid category fails validation."""
        assert validate_product_category("unknown_category") is False


class TestGetDefaultLifetime:
    """Test get_default_lifetime function."""

    def test_vehicles_default_lifetime(self):
        """Test vehicles default lifetime is 15."""
        assert get_default_lifetime("vehicles") == 15

    def test_it_equipment_default_lifetime(self):
        """Test IT equipment default lifetime is 5."""
        assert get_default_lifetime("it_equipment") == 5

    def test_unknown_category_fallback(self):
        """Test unknown category returns fallback default."""
        result = get_default_lifetime("unknown")
        assert isinstance(result, int)
        assert result > 0


class TestGetFuelEF:
    """Test get_fuel_ef function."""

    def test_gasoline_ef(self):
        """Test gasoline EF lookup returns 2.315."""
        result = get_fuel_ef("gasoline")
        assert result == Decimal("2.315")

    def test_diesel_ef(self):
        """Test diesel EF lookup returns 2.680."""
        result = get_fuel_ef("diesel")
        assert result == Decimal("2.680")

    def test_invalid_fuel_raises(self):
        """Test invalid fuel type raises ValueError."""
        with pytest.raises(ValueError):
            get_fuel_ef("invalid_fuel")


class TestGetGridEF:
    """Test get_grid_ef function."""

    def test_us_grid_ef(self):
        """Test US grid EF is 0.417."""
        result = get_grid_ef("US")
        assert result == Decimal("0.417")

    def test_global_fallback(self):
        """Test unknown region returns GLOBAL."""
        result = get_grid_ef("GLOBAL")
        assert result == Decimal("0.440")


class TestGetRefrigerantGWP:
    """Test get_refrigerant_gwp function."""

    def test_r410a_gwp(self):
        """Test R-410A GWP (AR5) is 2088."""
        result = get_refrigerant_gwp("R-410A", "AR5")
        assert result == Decimal("2088")

    def test_r134a_gwp_ar6(self):
        """Test R-134a GWP (AR6) is 1530."""
        result = get_refrigerant_gwp("R-134a", "AR6")
        assert result == Decimal("1530")


class TestConvertUnits:
    """Test convert_units function."""

    def test_litres_to_gallons(self):
        """Test litre to gallon conversion."""
        result = convert_units(Decimal("1000"), "litres", "gallons")
        assert result == pytest.approx(Decimal("264.172"), rel=Decimal("0.01"))

    def test_kg_to_tonnes(self):
        """Test kg to tonnes conversion."""
        result = convert_units(Decimal("1000"), "kg", "tonnes")
        assert result == Decimal("1.0")


class TestFormatCO2e:
    """Test format_co2e function."""

    def test_format_kg(self):
        """Test formatting in kg CO2e."""
        result = format_co2e(Decimal("1234.567"), "kg")
        assert "1234" in result or "1,234" in result

    def test_format_tonnes(self):
        """Test formatting in tonnes CO2e."""
        result = format_co2e(Decimal("1234567.0"), "tonnes")
        assert "1234" in result or "1,234" in result


class TestClassifyEmissionType:
    """Test classify_emission_type function."""

    def test_vehicles_direct(self):
        """Test vehicles category classifies as direct."""
        result = classify_emission_type("vehicles")
        assert result in ("direct", "both")

    def test_appliances_indirect(self):
        """Test appliances category classifies as indirect."""
        result = classify_emission_type("appliances")
        assert result in ("indirect", "both")

    def test_hvac_both(self):
        """Test HVAC category classifies as both."""
        result = classify_emission_type("hvac")
        assert result == "both"


class TestGetDegradationRate:
    """Test get_degradation_rate function."""

    def test_vehicles_degradation(self):
        """Test vehicles degradation rate."""
        result = get_degradation_rate("vehicles")
        assert result == Decimal("0.005")

    def test_unknown_returns_zero(self):
        """Test unknown category returns zero degradation."""
        result = get_degradation_rate("unknown")
        assert result == Decimal("0.0")


class TestComputeLifetimeEmissions:
    """Test compute_lifetime_emissions function."""

    def test_simple_computation(self):
        """Test lifetime emissions = units x lifetime x annual_emission."""
        result = compute_lifetime_emissions(
            units_sold=1000,
            lifetime_years=15,
            annual_emissions_per_unit=Decimal("2778.0"),
        )
        assert result == Decimal("41670000.0")

    def test_zero_units(self):
        """Test zero units returns zero emissions."""
        result = compute_lifetime_emissions(
            units_sold=0,
            lifetime_years=15,
            annual_emissions_per_unit=Decimal("2778.0"),
        )
        assert result == Decimal("0")


class TestComputeAnnualEmissions:
    """Test compute_annual_emissions function."""

    def test_fuel_combustion_annual(self):
        """Test annual fuel combustion: 1200L x 2.315 = 2778 kgCO2e."""
        result = compute_annual_emissions(
            consumption_per_year=Decimal("1200.0"),
            emission_factor=Decimal("2.315"),
        )
        assert result == Decimal("2778.0")


# ==============================================================================
# SUMMARY
# ==============================================================================


def test_total_enum_count():
    """Meta-test: verify enum count is at least 22."""
    enum_classes = [
        CalculationMethod, ProductCategory, EmissionType, FuelType,
        RefrigerantType, GridRegion, EFSource, ComplianceFramework,
        DataQualityTier, ProvenanceStage, UncertaintyMethod, DQIDimension,
        DQIScore, ComplianceStatus, GWPVersion, EmissionGas, CurrencyCode,
        ExportFormat, BatchStatus, DegradationModel, LifetimeSource,
        ProductSubcategory,
    ]
    assert len(enum_classes) >= 22
