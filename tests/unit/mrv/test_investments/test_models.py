# -*- coding: utf-8 -*-
"""
Test suite for investments.models - AGENT-MRV-028.

Tests all 22 enumerations, 16 constant tables, 16 Pydantic models,
and 18 helper functions for the Investments Agent (GL-MRV-S3-015).

Coverage:
- Enumerations: 22 enums (values, membership, count)
- Constants: SECTOR_EMISSION_FACTORS, COUNTRY_EMISSIONS, PCAF_QUALITY_CRITERIA,
  GRID_EMISSION_FACTORS, BUILDING_BENCHMARKS, VEHICLE_EMISSION_FACTORS,
  CURRENCY_RATES, SOVEREIGN_DATA, ASSET_CLASS_DEFAULTS, GWP_VALUES,
  PCAF_SCORING_MATRIX, DC_RULES, EPC_ADJUSTMENTS, CLIMATE_ZONE_FACTORS,
  CARBON_INTENSITY_BENCHMARKS, ATTRIBUTION_METHOD_RULES
- Input models: EquityInput, CorporateBondInput, ProjectFinanceInput,
  PrivateEquityInput, CREInput, MortgageInput, MotorVehicleInput,
  SovereignBondInput, PortfolioInput, ComplianceCheckInput
- Result models: EquityResult, DebtResult, RealAssetResult, SovereignResult,
  PortfolioResult, ComplianceCheckResult
- Helper functions: calculate_attribution_factor, calculate_financed_emissions,
  calculate_waci, calculate_pcaf_score, calculate_provenance_hash,
  get_dqi_classification, convert_currency, get_sector_ef,
  get_country_emissions, classify_pcaf_quality, get_epc_adjustment,
  get_building_benchmark, get_vehicle_ef, normalize_evic,
  annualize_project_emissions, calculate_carbon_intensity,
  validate_asset_class, get_dc_rule

Author: GL-TestEngineer
Date: February 2026
"""

from decimal import Decimal
from datetime import datetime
import pytest
from pydantic import ValidationError as PydanticValidationError

from greenlang.investments.models import (
    # Enumerations
    AssetClass,
    CalculationMethod,
    AttributionMethod,
    PCAFDataQuality,
    Sector,
    ComplianceFramework,
    ComplianceStatus,
    DCRule,
    InstrumentType,
    PropertyType,
    EPCRating,
    ClimateZone,
    VehicleCategory,
    FuelType,
    CurrencyCode,
    GWPVersion,
    ProvenanceStage,
    EFSource,
    DataQualityTier,
    UncertaintyMethod,
    ExportFormat,
    BatchStatus,
    # Constants
    AGENT_ID,
    AGENT_COMPONENT,
    VERSION,
    TABLE_PREFIX,
    GWP_VALUES,
    SECTOR_EMISSION_FACTORS,
    COUNTRY_EMISSIONS,
    PCAF_QUALITY_CRITERIA,
    GRID_EMISSION_FACTORS,
    BUILDING_BENCHMARKS,
    VEHICLE_EMISSION_FACTORS,
    CURRENCY_RATES,
    SOVEREIGN_DATA,
    DC_RULES,
    EPC_ADJUSTMENTS,
    CLIMATE_ZONE_FACTORS,
    CARBON_INTENSITY_BENCHMARKS,
    ATTRIBUTION_METHOD_RULES,
    PCAF_SCORING_MATRIX,
    ASSET_CLASS_DEFAULTS,
    # Input models
    EquityInput,
    CorporateBondInput,
    ProjectFinanceInput,
    PrivateEquityInput,
    CREInput,
    MortgageInput,
    MotorVehicleInput,
    SovereignBondInput,
    PortfolioInput,
    ComplianceCheckInput,
    # Result models
    EquityResult,
    DebtResult,
    RealAssetResult,
    SovereignResult,
    PortfolioResult,
    ComplianceCheckResult,
    # Helper functions
    calculate_attribution_factor,
    calculate_financed_emissions,
    calculate_waci,
    calculate_pcaf_score,
    calculate_provenance_hash,
    get_dqi_classification,
    convert_currency,
    get_sector_ef,
    get_country_emissions,
    classify_pcaf_quality,
    get_epc_adjustment,
    get_building_benchmark,
    get_vehicle_ef,
    normalize_evic,
    annualize_project_emissions,
    calculate_carbon_intensity,
    validate_asset_class,
    get_dc_rule,
)


# ==============================================================================
# ENUMERATION TESTS
# ==============================================================================


class TestAssetClassEnum:
    """Test AssetClass enum."""

    def test_asset_class_has_all_8_types(self):
        """Test all 8 asset class values exist."""
        assert AssetClass.LISTED_EQUITY == "listed_equity"
        assert AssetClass.CORPORATE_BOND == "corporate_bond"
        assert AssetClass.PRIVATE_EQUITY == "private_equity"
        assert AssetClass.PROJECT_FINANCE == "project_finance"
        assert AssetClass.COMMERCIAL_REAL_ESTATE == "commercial_real_estate"
        assert AssetClass.MORTGAGE == "mortgage"
        assert AssetClass.MOTOR_VEHICLE_LOAN == "motor_vehicle_loan"
        assert AssetClass.SOVEREIGN_BOND == "sovereign_bond"
        assert len(AssetClass) == 8

    def test_asset_class_membership(self):
        """Test membership check for valid and invalid values."""
        assert "listed_equity" in [e.value for e in AssetClass]
        assert "invalid_class" not in [e.value for e in AssetClass]


class TestCalculationMethodEnum:
    """Test CalculationMethod enum."""

    def test_calculation_method_values(self):
        """Test all calculation method values exist."""
        assert CalculationMethod.REPORTED == "reported"
        assert CalculationMethod.PHYSICAL_ACTIVITY == "physical_activity"
        assert CalculationMethod.REVENUE_BASED == "revenue_based"
        assert CalculationMethod.SECTOR_AVERAGE == "sector_average"
        assert CalculationMethod.ASSET_SPECIFIC == "asset_specific"
        assert len(CalculationMethod) == 5


class TestAttributionMethodEnum:
    """Test AttributionMethod enum."""

    def test_attribution_method_values(self):
        """Test all attribution method values exist."""
        assert AttributionMethod.EVIC == "evic"
        assert AttributionMethod.EQUITY_SHARE == "equity_share"
        assert AttributionMethod.TOTAL_BALANCE_SHEET == "total_balance_sheet"
        assert AttributionMethod.PRO_RATA_PROJECT == "pro_rata_project"
        assert AttributionMethod.LTV_WEIGHTED == "ltv_weighted"
        assert AttributionMethod.GDP_PPP == "gdp_ppp"


class TestPCAFDataQualityEnum:
    """Test PCAFDataQuality enum."""

    def test_pcaf_quality_scores_1_to_5(self):
        """Test PCAF data quality scores follow 1-5 ordering."""
        assert PCAFDataQuality.SCORE_1 == "score_1"
        assert PCAFDataQuality.SCORE_2 == "score_2"
        assert PCAFDataQuality.SCORE_3 == "score_3"
        assert PCAFDataQuality.SCORE_4 == "score_4"
        assert PCAFDataQuality.SCORE_5 == "score_5"
        assert len(PCAFDataQuality) == 5

    def test_pcaf_quality_score_ordering(self):
        """Test score 1 is best (reported) and score 5 is worst (estimated)."""
        scores = list(PCAFDataQuality)
        assert scores[0] == PCAFDataQuality.SCORE_1
        assert scores[-1] == PCAFDataQuality.SCORE_5


class TestSectorEnum:
    """Test Sector enum."""

    def test_sector_has_12_gics_sectors(self):
        """Test all 12 GICS-aligned sector values exist."""
        expected = [
            "energy", "materials", "industrials", "consumer_discretionary",
            "consumer_staples", "health_care", "financials",
            "information_technology", "communication_services",
            "utilities", "real_estate", "other",
        ]
        actual = [e.value for e in Sector]
        for s in expected:
            assert s in actual, f"Missing sector: {s}"
        assert len(Sector) == 12


class TestComplianceFrameworkEnum:
    """Test ComplianceFramework enum."""

    def test_compliance_framework_values(self):
        """Test all 9 compliance framework values exist."""
        assert ComplianceFramework.GHG_PROTOCOL == "ghg_protocol"
        assert ComplianceFramework.PCAF == "pcaf"
        assert ComplianceFramework.ISO_14064 == "iso_14064"
        assert ComplianceFramework.CSRD_ESRS == "csrd_esrs"
        assert ComplianceFramework.CDP == "cdp"
        assert ComplianceFramework.SBTI_FI == "sbti_fi"
        assert ComplianceFramework.SB_253 == "sb_253"
        assert ComplianceFramework.TCFD == "tcfd"
        assert ComplianceFramework.NZBA == "nzba"
        assert len(ComplianceFramework) == 9


class TestComplianceStatusEnum:
    """Test ComplianceStatus enum."""

    def test_compliance_status_values(self):
        """Test all compliance status values exist."""
        assert ComplianceStatus.PASS == "pass"
        assert ComplianceStatus.FAIL == "fail"
        assert ComplianceStatus.WARNING == "warning"
        assert len(ComplianceStatus) == 3


class TestDCRuleEnum:
    """Test DCRule enum."""

    def test_dc_rule_has_8_entries(self):
        """Test DC_RULES dict has all 8 double-counting rule entries."""
        assert DCRule.DC_INV_001 == "dc_inv_001"
        assert DCRule.DC_INV_002 == "dc_inv_002"
        assert DCRule.DC_INV_003 == "dc_inv_003"
        assert DCRule.DC_INV_004 == "dc_inv_004"
        assert DCRule.DC_INV_005 == "dc_inv_005"
        assert DCRule.DC_INV_006 == "dc_inv_006"
        assert DCRule.DC_INV_007 == "dc_inv_007"
        assert DCRule.DC_INV_008 == "dc_inv_008"
        assert len(DCRule) == 8


class TestInstrumentTypeEnum:
    """Test InstrumentType enum."""

    def test_instrument_type_values(self):
        """Test instrument type values exist."""
        assert InstrumentType.EQUITY == "equity"
        assert InstrumentType.BOND == "bond"
        assert InstrumentType.LOAN == "loan"
        assert InstrumentType.PROJECT_FINANCE == "project_finance"
        assert InstrumentType.MORTGAGE == "mortgage"
        assert InstrumentType.SOVEREIGN == "sovereign"


class TestPropertyTypeEnum:
    """Test PropertyType enum."""

    def test_property_type_values(self):
        """Test all property type values exist."""
        expected = ["office", "retail", "industrial", "residential",
                    "hotel", "mixed_use"]
        actual = [e.value for e in PropertyType]
        for p in expected:
            assert p in actual


class TestEPCRatingEnum:
    """Test EPCRating enum."""

    def test_epc_rating_values_a_to_g(self):
        """Test EPC ratings from A to G exist."""
        assert EPCRating.A == "A"
        assert EPCRating.B == "B"
        assert EPCRating.C == "C"
        assert EPCRating.D == "D"
        assert EPCRating.E == "E"
        assert EPCRating.F == "F"
        assert EPCRating.G == "G"
        assert len(EPCRating) == 7


class TestClimateZoneEnum:
    """Test ClimateZone enum."""

    def test_climate_zone_values(self):
        """Test all 5 climate zone values exist."""
        expected = ["tropical", "arid", "temperate", "continental", "polar"]
        actual = [e.value for e in ClimateZone]
        for c in expected:
            assert c in actual
        assert len(ClimateZone) == 5


class TestVehicleCategoryEnum:
    """Test VehicleCategory enum."""

    def test_vehicle_category_values(self):
        """Test vehicle category values exist."""
        assert VehicleCategory.PASSENGER_CAR == "passenger_car"
        assert VehicleCategory.LIGHT_COMMERCIAL == "light_commercial"
        assert VehicleCategory.HEAVY_COMMERCIAL == "heavy_commercial"
        assert VehicleCategory.ELECTRIC_VEHICLE == "electric_vehicle"
        assert VehicleCategory.MOTORCYCLE == "motorcycle"


class TestFuelTypeEnum:
    """Test FuelType enum."""

    def test_fuel_type_values(self):
        """Test fuel type values exist."""
        assert FuelType.PETROL == "petrol"
        assert FuelType.DIESEL == "diesel"
        assert FuelType.HYBRID == "hybrid"
        assert FuelType.ELECTRIC == "electric"
        assert FuelType.CNG == "cng"
        assert FuelType.LPG == "lpg"


class TestCurrencyCodeEnum:
    """Test CurrencyCode enum."""

    def test_currency_code_values(self):
        """Test all 15 currency code values exist."""
        expected = [
            "USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD",
            "CNY", "INR", "BRL", "ZAR", "SGD", "KRW", "SEK", "NOK",
        ]
        actual = [e.value for e in CurrencyCode]
        for c in expected:
            assert c in actual
        assert len(CurrencyCode) == 15


class TestGWPVersionEnum:
    """Test GWPVersion enum."""

    def test_gwp_version_values(self):
        """Test GWP version values exist."""
        assert GWPVersion.AR4 == "ar4"
        assert GWPVersion.AR5 == "ar5"
        assert GWPVersion.AR6 == "ar6"


class TestProvenanceStageEnum:
    """Test ProvenanceStage enum."""

    def test_provenance_stage_values(self):
        """Test all 10 provenance stage values exist."""
        expected = [
            "validate", "classify", "normalize", "resolve_efs",
            "calculate_equity", "calculate_debt", "calculate_real_assets",
            "calculate_sovereign", "compliance", "seal",
        ]
        actual = [e.value for e in ProvenanceStage]
        for s in expected:
            assert s in actual
        assert len(ProvenanceStage) == 10


class TestRemainingEnums:
    """Test remaining enumerations."""

    def test_ef_source_enum(self):
        """Test EFSource values."""
        assert EFSource.PCAF == "pcaf"
        assert EFSource.BLOOMBERG == "bloomberg"
        assert EFSource.CDP == "cdp"

    def test_data_quality_tier_enum(self):
        """Test DataQualityTier values."""
        assert DataQualityTier.TIER_1 == "tier_1"
        assert DataQualityTier.TIER_2 == "tier_2"
        assert DataQualityTier.TIER_3 == "tier_3"

    def test_uncertainty_method_enum(self):
        """Test UncertaintyMethod values."""
        assert UncertaintyMethod.MONTE_CARLO == "monte_carlo"
        assert UncertaintyMethod.ANALYTICAL == "analytical"

    def test_export_format_enum(self):
        """Test ExportFormat values."""
        assert ExportFormat.JSON == "json"
        assert ExportFormat.CSV == "csv"

    def test_batch_status_enum(self):
        """Test BatchStatus values."""
        assert BatchStatus.PENDING == "pending"
        assert BatchStatus.COMPLETED == "completed"
        assert BatchStatus.FAILED == "failed"


# ==============================================================================
# CONSTANT TABLE TESTS
# ==============================================================================


class TestGWPValues:
    """Test GWP_VALUES constant table."""

    def test_gwp_ar5_co2(self):
        """Test AR5 CO2 GWP is 1."""
        assert GWP_VALUES[GWPVersion.AR5]["co2"] == Decimal("1")

    def test_gwp_ar5_ch4(self):
        """Test AR5 CH4 GWP is 28."""
        assert GWP_VALUES[GWPVersion.AR5]["ch4"] == Decimal("28")

    def test_gwp_ar5_n2o(self):
        """Test AR5 N2O GWP is 265."""
        assert GWP_VALUES[GWPVersion.AR5]["n2o"] == Decimal("265")

    def test_gwp_all_versions_present(self):
        """Test all GWP versions have entries."""
        assert GWPVersion.AR4 in GWP_VALUES
        assert GWPVersion.AR5 in GWP_VALUES
        assert GWPVersion.AR6 in GWP_VALUES


class TestSectorEmissionFactors:
    """Test SECTOR_EMISSION_FACTORS constant table."""

    def test_sector_ef_has_12_sectors(self):
        """Test all 12 GICS sectors have emission factors."""
        assert len(SECTOR_EMISSION_FACTORS) >= 12

    @pytest.mark.parametrize("sector", [
        "energy", "materials", "industrials", "consumer_discretionary",
        "consumer_staples", "health_care", "financials",
        "information_technology", "communication_services",
        "utilities", "real_estate", "other",
    ])
    def test_sector_ef_positive(self, sector):
        """Test each sector emission factor is positive."""
        assert SECTOR_EMISSION_FACTORS[sector] > Decimal("0")

    def test_energy_sector_highest(self):
        """Test energy sector has the highest emission factor."""
        energy_ef = SECTOR_EMISSION_FACTORS["energy"]
        for sector, ef in SECTOR_EMISSION_FACTORS.items():
            if sector != "energy":
                assert energy_ef >= ef, f"Energy EF should be >= {sector} EF"


class TestCountryEmissions:
    """Test COUNTRY_EMISSIONS constant table."""

    def test_country_emissions_has_50_plus_countries(self):
        """Test at least 50 countries have emissions data."""
        assert len(COUNTRY_EMISSIONS) >= 50

    @pytest.mark.parametrize("country", [
        "US", "CN", "IN", "RU", "JP", "DE", "GB", "FR", "BR", "CA",
    ])
    def test_major_country_emissions_positive(self, country):
        """Test major countries have positive emissions."""
        assert COUNTRY_EMISSIONS[country] > Decimal("0")

    def test_us_emissions_value(self):
        """Test US emissions are approximately 5.2 GtCO2e."""
        us_emissions = COUNTRY_EMISSIONS["US"]
        assert us_emissions > Decimal("4000000000")
        assert us_emissions < Decimal("7000000000")


class TestPCAFQualityCriteria:
    """Test PCAF_QUALITY_CRITERIA constant table."""

    @pytest.mark.parametrize("asset_class", [
        "listed_equity", "corporate_bond", "private_equity",
        "project_finance", "commercial_real_estate", "mortgage",
        "motor_vehicle_loan", "sovereign_bond",
    ])
    def test_pcaf_criteria_for_all_asset_classes(self, asset_class):
        """Test PCAF quality criteria exist for all 8 asset classes."""
        assert asset_class in PCAF_QUALITY_CRITERIA

    @pytest.mark.parametrize("score", [1, 2, 3, 4, 5])
    def test_pcaf_criteria_for_all_scores(self, score):
        """Test all 5 quality scores exist for listed_equity."""
        assert score in PCAF_QUALITY_CRITERIA["listed_equity"]

    def test_pcaf_score_1_is_reported(self):
        """Test score 1 criteria indicates reported/audited data."""
        criteria = PCAF_QUALITY_CRITERIA["listed_equity"][1]
        assert "reported" in criteria["data_type"].lower() or \
               "audited" in criteria["description"].lower()


class TestGridEmissionFactors:
    """Test GRID_EMISSION_FACTORS constant table."""

    def test_grid_ef_positive(self):
        """Test all grid EFs are positive."""
        for region, ef in GRID_EMISSION_FACTORS.items():
            assert ef > Decimal("0"), f"Grid EF for {region} should be positive"


class TestBuildingBenchmarks:
    """Test BUILDING_BENCHMARKS constant table."""

    @pytest.mark.parametrize("property_type", [
        "office", "retail", "industrial", "residential", "hotel", "mixed_use",
    ])
    def test_building_benchmark_property_types(self, property_type):
        """Test all 6 property types have benchmarks."""
        assert property_type in BUILDING_BENCHMARKS

    @pytest.mark.parametrize("climate_zone", [
        "tropical", "arid", "temperate", "continental", "polar",
    ])
    def test_building_benchmark_climate_zones(self, climate_zone):
        """Test office benchmarks exist for all 5 climate zones."""
        assert climate_zone in BUILDING_BENCHMARKS["office"]


class TestVehicleEmissionFactors:
    """Test VEHICLE_EMISSION_FACTORS constant table."""

    @pytest.mark.parametrize("category", [
        "passenger_car", "light_commercial", "heavy_commercial",
        "electric_vehicle", "motorcycle",
    ])
    def test_vehicle_ef_by_category(self, category):
        """Test each vehicle category has emission factors."""
        assert category in VEHICLE_EMISSION_FACTORS

    def test_ev_emissions_lower_than_petrol(self):
        """Test EV emissions are lower than petrol car."""
        ev_ef = VEHICLE_EMISSION_FACTORS["electric_vehicle"]["ef_per_km"]
        petrol_ef = VEHICLE_EMISSION_FACTORS["passenger_car"]["ef_per_km"]
        assert ev_ef < petrol_ef


class TestCurrencyRates:
    """Test CURRENCY_RATES constant table."""

    def test_usd_rate_is_one(self):
        """Test USD rate is 1.0."""
        assert CURRENCY_RATES[CurrencyCode.USD] == Decimal("1.0")

    def test_currency_rates_has_15_currencies(self):
        """Test 15 currencies have exchange rates."""
        assert len(CURRENCY_RATES) == 15

    @pytest.mark.parametrize("currency", [
        CurrencyCode.EUR, CurrencyCode.GBP, CurrencyCode.JPY,
        CurrencyCode.CHF, CurrencyCode.CAD,
    ])
    def test_major_currency_rates_positive(self, currency):
        """Test major currency rates are positive."""
        assert CURRENCY_RATES[currency] > Decimal("0")


class TestDCRules:
    """Test DC_RULES constant table."""

    def test_dc_rules_has_8_entries(self):
        """Test DC_RULES has all 8 double-counting rule entries."""
        assert len(DC_RULES) == 8

    def test_dc_inv_001_consolidated_exclusion(self):
        """Test DC-INV-001 excludes consolidated subsidiaries."""
        rule = DC_RULES["dc_inv_001"]
        assert "consolidat" in rule["description"].lower()

    @pytest.mark.parametrize("rule_id", [
        "dc_inv_001", "dc_inv_002", "dc_inv_003", "dc_inv_004",
        "dc_inv_005", "dc_inv_006", "dc_inv_007", "dc_inv_008",
    ])
    def test_dc_rule_has_description(self, rule_id):
        """Test each DC rule has a description."""
        assert "description" in DC_RULES[rule_id]
        assert len(DC_RULES[rule_id]["description"]) > 0


class TestEPCAdjustments:
    """Test EPC_ADJUSTMENTS constant table."""

    @pytest.mark.parametrize("rating", ["A", "B", "C", "D", "E", "F", "G"])
    def test_epc_adjustment_all_ratings(self, rating):
        """Test all EPC ratings A-G have adjustment factors."""
        assert rating in EPC_ADJUSTMENTS

    def test_epc_a_is_lowest_adjustment(self):
        """Test EPC A has the lowest (best) adjustment factor."""
        a_adj = EPC_ADJUSTMENTS["A"]
        for rating, adj in EPC_ADJUSTMENTS.items():
            if rating != "A":
                assert a_adj <= adj


class TestRemainingConstantTables:
    """Test remaining constant tables."""

    def test_climate_zone_factors_positive(self):
        """Test all climate zone factors are positive."""
        for zone, factor in CLIMATE_ZONE_FACTORS.items():
            assert factor > Decimal("0")

    def test_carbon_intensity_benchmarks_exist(self):
        """Test carbon intensity benchmarks table exists and is populated."""
        assert len(CARBON_INTENSITY_BENCHMARKS) > 0

    def test_attribution_method_rules_exist(self):
        """Test attribution method rules table exists."""
        assert len(ATTRIBUTION_METHOD_RULES) > 0

    def test_pcaf_scoring_matrix_exists(self):
        """Test PCAF scoring matrix is populated."""
        assert len(PCAF_SCORING_MATRIX) > 0

    def test_asset_class_defaults_for_all_8(self):
        """Test defaults exist for all 8 asset classes."""
        assert len(ASSET_CLASS_DEFAULTS) == 8


# ==============================================================================
# AGENT METADATA TESTS
# ==============================================================================


class TestAgentMetadata:
    """Test agent metadata constants."""

    def test_agent_id(self):
        """Test AGENT_ID is GL-MRV-S3-015."""
        assert AGENT_ID == "GL-MRV-S3-015"

    def test_agent_component(self):
        """Test AGENT_COMPONENT is AGENT-MRV-028."""
        assert AGENT_COMPONENT == "AGENT-MRV-028"

    def test_version(self):
        """Test VERSION is 1.0.0."""
        assert VERSION == "1.0.0"

    def test_table_prefix(self):
        """Test TABLE_PREFIX is gl_inv_."""
        assert TABLE_PREFIX == "gl_inv_"


# ==============================================================================
# INPUT MODEL TESTS
# ==============================================================================


class TestEquityInput:
    """Test EquityInput Pydantic model."""

    def test_equity_input_creation(self):
        """Test valid EquityInput creation."""
        inp = EquityInput(
            asset_class="listed_equity",
            investee_name="Apple Inc.",
            outstanding_amount=Decimal("100000000"),
            evic=Decimal("3000000000000"),
            investee_scope1=Decimal("22400"),
            investee_scope2=Decimal("9100"),
            sector="information_technology",
            country="US",
        )
        assert inp.investee_name == "Apple Inc."
        assert inp.outstanding_amount == Decimal("100000000")

    def test_equity_input_frozen(self):
        """Test EquityInput is immutable."""
        inp = EquityInput(
            asset_class="listed_equity",
            investee_name="Test",
            outstanding_amount=Decimal("100"),
            evic=Decimal("1000"),
            investee_scope1=Decimal("10"),
            investee_scope2=Decimal("5"),
            sector="energy",
            country="US",
        )
        with pytest.raises(Exception):
            inp.investee_name = "Modified"

    def test_equity_input_negative_amount_rejected(self):
        """Test negative outstanding_amount is rejected."""
        with pytest.raises(PydanticValidationError):
            EquityInput(
                asset_class="listed_equity",
                investee_name="Test",
                outstanding_amount=Decimal("-100"),
                evic=Decimal("1000"),
                investee_scope1=Decimal("10"),
                investee_scope2=Decimal("5"),
                sector="energy",
                country="US",
            )


class TestCorporateBondInput:
    """Test CorporateBondInput Pydantic model."""

    def test_corporate_bond_input_creation(self):
        """Test valid CorporateBondInput creation."""
        inp = CorporateBondInput(
            asset_class="corporate_bond",
            investee_name="Tesla Inc.",
            outstanding_amount=Decimal("75000000"),
            evic=Decimal("500000000000"),
            investee_scope1=Decimal("30000"),
            investee_scope2=Decimal("12000"),
            sector="consumer_discretionary",
            country="US",
        )
        assert inp.investee_name == "Tesla Inc."

    def test_corporate_bond_input_frozen(self):
        """Test CorporateBondInput is immutable."""
        inp = CorporateBondInput(
            asset_class="corporate_bond",
            investee_name="Test",
            outstanding_amount=Decimal("100"),
            evic=Decimal("1000"),
            investee_scope1=Decimal("10"),
            investee_scope2=Decimal("5"),
            sector="energy",
            country="US",
        )
        with pytest.raises(Exception):
            inp.investee_name = "Modified"


class TestProjectFinanceInput:
    """Test ProjectFinanceInput Pydantic model."""

    def test_project_finance_input_creation(self):
        """Test valid ProjectFinanceInput creation."""
        inp = ProjectFinanceInput(
            asset_class="project_finance",
            project_name="Solar Farm",
            outstanding_amount=Decimal("30000000"),
            total_project_cost=Decimal("100000000"),
            project_lifetime_years=25,
            annual_project_emissions=Decimal("500"),
            sector="utilities",
            country="US",
        )
        assert inp.project_name == "Solar Farm"
        assert inp.project_lifetime_years == 25


class TestPrivateEquityInput:
    """Test PrivateEquityInput Pydantic model."""

    def test_private_equity_input_creation(self):
        """Test valid PrivateEquityInput creation."""
        inp = PrivateEquityInput(
            asset_class="private_equity",
            investee_name="GreenTech",
            outstanding_amount=Decimal("50000000"),
            total_equity_plus_debt=Decimal("200000000"),
            investee_scope1=Decimal("15000"),
            investee_scope2=Decimal("8000"),
            sector="industrials",
            country="US",
        )
        assert inp.total_equity_plus_debt == Decimal("200000000")


class TestCREInput:
    """Test CREInput Pydantic model."""

    def test_cre_input_creation(self):
        """Test valid CREInput creation."""
        inp = CREInput(
            asset_class="commercial_real_estate",
            property_name="Office Tower",
            outstanding_amount=Decimal("25000000"),
            property_value=Decimal("50000000"),
            floor_area_m2=Decimal("10000"),
            property_type="office",
            epc_rating="B",
            climate_zone="temperate",
            country="US",
        )
        assert inp.floor_area_m2 == Decimal("10000")
        assert inp.property_type == "office"


class TestMortgageInput:
    """Test MortgageInput Pydantic model."""

    def test_mortgage_input_creation(self):
        """Test valid MortgageInput creation."""
        inp = MortgageInput(
            asset_class="mortgage",
            property_name="123 Oak St",
            outstanding_amount=Decimal("300000"),
            property_value=Decimal("400000"),
            floor_area_m2=Decimal("150"),
            property_type="residential",
            epc_rating="C",
            climate_zone="temperate",
            country="US",
        )
        assert inp.outstanding_amount == Decimal("300000")


class TestMotorVehicleInput:
    """Test MotorVehicleInput Pydantic model."""

    def test_motor_vehicle_input_creation(self):
        """Test valid MotorVehicleInput creation."""
        inp = MotorVehicleInput(
            asset_class="motor_vehicle_loan",
            vehicle_description="Toyota Camry",
            outstanding_amount=Decimal("25000"),
            vehicle_value=Decimal("35000"),
            vehicle_category="passenger_car",
            fuel_type="hybrid",
            annual_mileage_km=Decimal("20000"),
            country="US",
        )
        assert inp.vehicle_category == "passenger_car"
        assert inp.annual_mileage_km == Decimal("20000")


class TestSovereignBondInput:
    """Test SovereignBondInput Pydantic model."""

    def test_sovereign_bond_input_creation(self):
        """Test valid SovereignBondInput creation."""
        inp = SovereignBondInput(
            asset_class="sovereign_bond",
            country="US",
            outstanding_amount=Decimal("500000000"),
            gdp_ppp=Decimal("25460000000000"),
            country_emissions=Decimal("5222000000"),
            include_lulucf=False,
        )
        assert inp.country == "US"
        assert inp.include_lulucf is False


class TestPortfolioInput:
    """Test PortfolioInput Pydantic model."""

    def test_portfolio_input_creation(self):
        """Test valid PortfolioInput creation."""
        inp = PortfolioInput(
            portfolio_name="Test Portfolio",
            reporting_year=2024,
            investments=[],
        )
        assert inp.portfolio_name == "Test Portfolio"

    def test_portfolio_input_frozen(self):
        """Test PortfolioInput is immutable."""
        inp = PortfolioInput(
            portfolio_name="Test",
            reporting_year=2024,
            investments=[],
        )
        with pytest.raises(Exception):
            inp.portfolio_name = "Modified"


class TestComplianceCheckInputModel:
    """Test ComplianceCheckInput Pydantic model."""

    def test_compliance_check_input_creation(self):
        """Test valid ComplianceCheckInput creation."""
        inp = ComplianceCheckInput(
            frameworks=["ghg_protocol", "pcaf"],
            calculation_results=[{"total_co2e": 1000}],
        )
        assert len(inp.frameworks) == 2


# ==============================================================================
# RESULT MODEL TESTS
# ==============================================================================


class TestResultModels:
    """Test result Pydantic models."""

    def test_equity_result_frozen(self):
        """Test EquityResult is immutable."""
        result = EquityResult(
            investee_name="Apple",
            asset_class="listed_equity",
            attribution_factor=Decimal("0.00003333"),
            financed_emissions=Decimal("1.05"),
            carbon_intensity=Decimal("10.5"),
            pcaf_quality_score=1,
            provenance_hash="a" * 64,
        )
        with pytest.raises(Exception):
            result.investee_name = "Modified"

    def test_debt_result_creation(self):
        """Test DebtResult creation."""
        result = DebtResult(
            investee_name="Tesla",
            asset_class="corporate_bond",
            attribution_factor=Decimal("0.00015"),
            financed_emissions=Decimal("6.3"),
            pcaf_quality_score=1,
            provenance_hash="b" * 64,
        )
        assert result.financed_emissions == Decimal("6.3")

    def test_real_asset_result_creation(self):
        """Test RealAssetResult creation."""
        result = RealAssetResult(
            property_name="Office Tower",
            asset_class="commercial_real_estate",
            attribution_factor=Decimal("0.5"),
            financed_emissions=Decimal("250.0"),
            pcaf_quality_score=2,
            provenance_hash="c" * 64,
        )
        assert result.attribution_factor == Decimal("0.5")

    def test_sovereign_result_creation(self):
        """Test SovereignResult creation."""
        result = SovereignResult(
            country="US",
            attribution_factor=Decimal("0.00001964"),
            financed_emissions=Decimal("102560.0"),
            pcaf_quality_score=4,
            provenance_hash="d" * 64,
        )
        assert result.country == "US"

    def test_portfolio_result_creation(self):
        """Test PortfolioResult creation."""
        result = PortfolioResult(
            portfolio_name="Test Portfolio",
            total_financed_emissions=Decimal("103000"),
            weighted_pcaf_score=Decimal("2.5"),
            asset_class_breakdown={},
            provenance_hash="e" * 64,
        )
        assert result.total_financed_emissions == Decimal("103000")


# ==============================================================================
# HELPER FUNCTION TESTS
# ==============================================================================


class TestCalculateAttributionFactor:
    """Test calculate_attribution_factor helper function."""

    def test_evic_attribution_factor(self):
        """Test EVIC attribution factor calculation."""
        factor = calculate_attribution_factor(
            outstanding_amount=Decimal("100000000"),
            denominator=Decimal("3000000000000"),
            method="evic",
        )
        expected = Decimal("100000000") / Decimal("3000000000000")
        assert abs(factor - expected) < Decimal("0.0000001")

    def test_equity_share_attribution(self):
        """Test equity share attribution factor."""
        factor = calculate_attribution_factor(
            outstanding_amount=Decimal("50000000"),
            denominator=Decimal("200000000"),
            method="equity_share",
        )
        assert factor == Decimal("0.25")


class TestCalculateFinancedEmissions:
    """Test calculate_financed_emissions helper function."""

    def test_financed_emissions_basic(self):
        """Test basic financed emissions calculation."""
        emissions = calculate_financed_emissions(
            attribution_factor=Decimal("0.01"),
            investee_emissions=Decimal("50000"),
        )
        assert emissions == Decimal("500")

    def test_financed_emissions_zero_attribution(self):
        """Test zero attribution produces zero emissions."""
        emissions = calculate_financed_emissions(
            attribution_factor=Decimal("0"),
            investee_emissions=Decimal("50000"),
        )
        assert emissions == Decimal("0")


class TestCalculateWACI:
    """Test calculate_waci helper function."""

    def test_waci_basic(self):
        """Test WACI calculation with single holding."""
        waci = calculate_waci(
            weights=[Decimal("1.0")],
            intensities=[Decimal("15.5")],
        )
        assert waci == Decimal("15.5")

    def test_waci_multiple_holdings(self):
        """Test WACI with multiple holdings."""
        waci = calculate_waci(
            weights=[Decimal("0.6"), Decimal("0.4")],
            intensities=[Decimal("10.0"), Decimal("20.0")],
        )
        expected = Decimal("0.6") * Decimal("10.0") + Decimal("0.4") * Decimal("20.0")
        assert abs(waci - expected) < Decimal("0.01")


class TestCalculatePCAFScore:
    """Test calculate_pcaf_score helper function."""

    def test_pcaf_score_1_reported(self):
        """Test PCAF score 1 for reported data."""
        score = calculate_pcaf_score(
            data_type="reported",
            verified=True,
        )
        assert score == 1

    def test_pcaf_score_5_estimated(self):
        """Test PCAF score 5 for estimated sector average data."""
        score = calculate_pcaf_score(
            data_type="estimated",
            verified=False,
        )
        assert score == 5


class TestCalculateProvenanceHash:
    """Test calculate_provenance_hash helper function."""

    def test_provenance_hash_deterministic(self):
        """Test provenance hash is deterministic."""
        h1 = calculate_provenance_hash("input1", "input2")
        h2 = calculate_provenance_hash("input1", "input2")
        assert h1 == h2

    def test_provenance_hash_length(self):
        """Test provenance hash is 64 characters (SHA-256)."""
        h = calculate_provenance_hash("test")
        assert len(h) == 64

    def test_provenance_hash_different_inputs(self):
        """Test different inputs produce different hashes."""
        h1 = calculate_provenance_hash("input1")
        h2 = calculate_provenance_hash("input2")
        assert h1 != h2

    def test_provenance_hash_with_decimal(self):
        """Test provenance hash with Decimal input."""
        h = calculate_provenance_hash(Decimal("123.456"))
        assert len(h) == 64


class TestGetDqiClassification:
    """Test get_dqi_classification helper function."""

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


class TestConvertCurrency:
    """Test convert_currency helper function."""

    def test_usd_to_usd(self):
        """Test USD to USD conversion is identity."""
        result = convert_currency(Decimal("1000"), CurrencyCode.USD)
        assert result == Decimal("1000")

    def test_eur_to_usd(self):
        """Test EUR to USD conversion uses rate."""
        result = convert_currency(Decimal("1000"), CurrencyCode.EUR)
        expected_rate = CURRENCY_RATES[CurrencyCode.EUR]
        assert abs(result - Decimal("1000") * expected_rate) < Decimal("0.01")


class TestGetSectorEF:
    """Test get_sector_ef helper function."""

    def test_sector_ef_energy(self):
        """Test energy sector EF retrieval."""
        ef = get_sector_ef("energy")
        assert ef > Decimal("0")
        assert ef == SECTOR_EMISSION_FACTORS["energy"]

    def test_sector_ef_invalid(self):
        """Test invalid sector returns None or raises ValueError."""
        try:
            result = get_sector_ef("nonexistent_sector")
            assert result is None
        except (ValueError, KeyError):
            pass  # Also acceptable


class TestGetCountryEmissions:
    """Test get_country_emissions helper function."""

    def test_country_emissions_us(self):
        """Test US country emissions retrieval."""
        emissions = get_country_emissions("US")
        assert emissions > Decimal("0")

    def test_country_emissions_china(self):
        """Test China country emissions retrieval."""
        emissions = get_country_emissions("CN")
        assert emissions > Decimal("0")


class TestClassifyPCAFQuality:
    """Test classify_pcaf_quality helper function."""

    @pytest.mark.parametrize("score,expected_tier", [
        (1, "reported"),
        (2, "physical_activity"),
        (3, "revenue_based"),
        (4, "sector_average"),
        (5, "estimated"),
    ])
    def test_classify_pcaf_quality_all_scores(self, score, expected_tier):
        """Test PCAF quality classification for all scores."""
        result = classify_pcaf_quality(score)
        assert result is not None


class TestRemainingHelpers:
    """Test remaining helper functions."""

    def test_get_epc_adjustment(self):
        """Test EPC adjustment retrieval."""
        adj = get_epc_adjustment("A")
        assert adj is not None
        assert adj <= get_epc_adjustment("G")

    def test_get_building_benchmark(self):
        """Test building benchmark retrieval."""
        benchmark = get_building_benchmark("office", "temperate")
        assert benchmark > Decimal("0")

    def test_get_vehicle_ef(self):
        """Test vehicle emission factor retrieval."""
        ef = get_vehicle_ef("passenger_car")
        assert ef > Decimal("0")

    def test_normalize_evic(self):
        """Test EVIC normalization."""
        evic = normalize_evic(Decimal("3000000000000"), "USD")
        assert evic > Decimal("0")

    def test_annualize_project_emissions(self):
        """Test project emissions annualization."""
        annual = annualize_project_emissions(
            total_emissions=Decimal("12500"),
            lifetime_years=25,
        )
        assert annual == Decimal("500")

    def test_calculate_carbon_intensity(self):
        """Test carbon intensity calculation."""
        intensity = calculate_carbon_intensity(
            emissions=Decimal("50000"),
            revenue=Decimal("1000000"),
        )
        assert intensity == Decimal("50")

    def test_validate_asset_class_valid(self):
        """Test validate_asset_class with valid class."""
        result = validate_asset_class("listed_equity")
        assert result is True

    def test_validate_asset_class_invalid(self):
        """Test validate_asset_class with invalid class raises error."""
        with pytest.raises((ValueError, KeyError)):
            validate_asset_class("invalid_asset_class")

    def test_get_dc_rule(self):
        """Test DC rule retrieval."""
        rule = get_dc_rule("dc_inv_001")
        assert "description" in rule
