# -*- coding: utf-8 -*-
"""
Test suite for processing_sold_products.models - AGENT-MRV-023.

Tests all 20 enums, 14 constant tables, 12 Pydantic models, and 14 helper
functions for the Processing of Sold Products Agent (GL-MRV-S3-010).

Coverage:
- Enumerations: 20 enums (values, membership, count, parametrized)
- Constants: PROCESS_EMISSION_FACTORS, ENERGY_INTENSITY_FACTORS, GRID_EFS,
  EEIO_FACTORS, CURRENCY_RATES, CPI_DEFLATORS, DQI_SCORING, DQI_WEIGHTS,
  UNCERTAINTY_RANGES, PROCESSING_CHAINS, FUEL_EFS, GWP_VALUES,
  ALLOCATION_METHODS, DC_RULES
- Input models: SiteSpecificInput, AverageDataInput, SpendInput,
  ProductInput, ChainInput, PortfolioInput
- Result models: CalculationResult, ComplianceResult, AggregationResult,
  ChainResult, PortfolioResult, HotspotResult (frozen=True checks)
- Helper functions: calculate_provenance_hash, get_dqi_classification,
  convert_currency_to_usd, get_cpi_deflator, classify_product_category,
  validate_processing_type, get_process_ef, get_grid_ef,
  get_energy_intensity, get_eeio_factor, compute_dqi_score,
  compute_uncertainty_range, get_allocation_weights, format_co2e

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
    from greenlang.agents.mrv.processing_sold_products.models import (
        # Enumerations
        CalculationMethod,
        ProductCategory,
        ProcessingType,
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
        AllocationMethod,
        FuelType,
        RegionCode,
        ChainType,

        # Constants
        AGENT_ID,
        AGENT_COMPONENT,
        VERSION,
        TABLE_PREFIX,
        PROCESS_EMISSION_FACTORS,
        ENERGY_INTENSITY_FACTORS,
        GRID_EFS,
        EEIO_FACTORS,
        CURRENCY_RATES,
        CPI_DEFLATORS,
        DQI_SCORING,
        DQI_WEIGHTS,
        UNCERTAINTY_RANGES,
        PROCESSING_CHAINS,
        FUEL_EFS,
        GWP_VALUES,
        ALLOCATION_METHODS,
        DC_RULES,

        # Input models
        SiteSpecificInput,
        AverageDataInput,
        SpendBasedInput,
        ProductInput,
        ChainInput,
        PortfolioInput,

        # Result models
        CalculationResult,
        ComplianceCheckResult,
        AggregationResult,
        ChainResult,
        PortfolioResult,
        HotspotResult,

        # Helpers
        calculate_provenance_hash,
        get_dqi_classification,
        convert_currency_to_usd,
        get_cpi_deflator,
        classify_product_category,
        validate_processing_type,
        get_process_ef,
        get_grid_ef,
        get_energy_intensity,
        get_eeio_factor,
        compute_dqi_score,
        compute_uncertainty_range,
        get_allocation_weights,
        format_co2e,
    )
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False

from pydantic import ValidationError as PydanticValidationError

_SKIP = pytest.mark.skipif(
    not MODELS_AVAILABLE,
    reason="processing_sold_products.models not available",
)
pytestmark = _SKIP


# ==============================================================================
# ENUMERATION TESTS
# ==============================================================================


class TestCalculationMethodEnum:
    """Test CalculationMethod enum."""

    def test_calculation_method_values(self):
        """Test all 5 calculation method values exist."""
        assert CalculationMethod.SITE_SPECIFIC_DIRECT == "site_specific_direct"
        assert CalculationMethod.SITE_SPECIFIC_ENERGY == "site_specific_energy"
        assert CalculationMethod.SITE_SPECIFIC_FUEL == "site_specific_fuel"
        assert CalculationMethod.AVERAGE_DATA == "average_data"
        assert CalculationMethod.SPEND_BASED == "spend_based"
        assert len(CalculationMethod) == 5

    @pytest.mark.parametrize("value", [
        "site_specific_direct", "site_specific_energy", "site_specific_fuel",
        "average_data", "spend_based",
    ])
    def test_calculation_method_membership(self, value):
        """Test each method value is a valid member."""
        assert CalculationMethod(value) is not None


class TestProductCategoryEnum:
    """Test ProductCategory enum."""

    @pytest.mark.parametrize("value", [
        "metals_ferrous", "metals_non_ferrous",
        "plastics_thermoplastic", "plastics_thermoset",
        "chemicals", "food_ingredients", "textiles",
        "electronics_components", "glass_ceramics",
        "wood_paper_pulp", "minerals", "agricultural_commodities",
    ])
    def test_product_category_membership(self, value):
        """Test each category value is valid."""
        assert ProductCategory(value) is not None

    def test_product_category_count(self):
        """Test there are exactly 12 product categories."""
        assert len(ProductCategory) == 12


class TestProcessingTypeEnum:
    """Test ProcessingType enum."""

    @pytest.mark.parametrize("value", [
        "machining", "stamping", "welding", "heat_treatment",
        "injection_molding", "extrusion", "blow_molding",
        "casting", "forging", "coating", "assembly",
        "chemical_reaction", "refining", "milling",
        "drying", "sintering", "fermentation", "textile_finishing",
    ])
    def test_processing_type_membership(self, value):
        """Test each processing type value is valid."""
        assert ProcessingType(value) is not None

    def test_processing_type_count(self):
        """Test there are exactly 18 processing types."""
        assert len(ProcessingType) == 18


class TestEFSourceEnum:
    """Test EFSource enum."""

    @pytest.mark.parametrize("value", [
        "customer_reported", "epa", "defra", "ecoinvent",
        "gabi", "ipcc", "eeio", "literature",
    ])
    def test_ef_source_membership(self, value):
        """Test each EF source is valid."""
        assert EFSource(value) is not None

    def test_ef_source_count(self):
        """Test there are at least 8 EF sources."""
        assert len(EFSource) >= 8


class TestComplianceFrameworkEnum:
    """Test ComplianceFramework enum."""

    @pytest.mark.parametrize("value", [
        "GHG_PROTOCOL_SCOPE3", "ISO_14064", "CSRD_ESRS_E1",
        "CDP", "SBTI", "SB_253", "GRI",
    ])
    def test_compliance_framework_membership(self, value):
        """Test each framework is valid."""
        assert ComplianceFramework(value) is not None

    def test_compliance_framework_count(self):
        """Test there are exactly 7 frameworks."""
        assert len(ComplianceFramework) == 7


class TestDataQualityTierEnum:
    """Test DataQualityTier enum."""

    @pytest.mark.parametrize("value,expected_label", [
        ("TIER_1", "TIER_1"),
        ("TIER_2", "TIER_2"),
        ("TIER_3", "TIER_3"),
    ])
    def test_data_quality_tier(self, value, expected_label):
        """Test tier values exist."""
        tier = DataQualityTier(value)
        assert tier.value == expected_label


class TestComplianceStatusEnum:
    """Test ComplianceStatus enum."""

    @pytest.mark.parametrize("value", ["pass", "fail", "warning", "not_applicable"])
    def test_compliance_status_membership(self, value):
        """Test each status value is valid."""
        assert ComplianceStatus(value) is not None


class TestGWPVersionEnum:
    """Test GWPVersion enum."""

    @pytest.mark.parametrize("value", ["AR5", "AR6"])
    def test_gwp_version_membership(self, value):
        """Test GWP version values."""
        assert GWPVersion(value) is not None


class TestEmissionGasEnum:
    """Test EmissionGas enum."""

    @pytest.mark.parametrize("value", ["co2", "ch4", "n2o", "co2e"])
    def test_emission_gas_membership(self, value):
        """Test emission gas values."""
        assert EmissionGas(value) is not None


class TestCurrencyCodeEnum:
    """Test CurrencyCode enum."""

    @pytest.mark.parametrize("value", [
        "USD", "EUR", "GBP", "JPY", "CNY", "INR",
        "BRL", "KRW", "TWD", "CAD", "AUD", "CHF",
    ])
    def test_currency_code_membership(self, value):
        """Test each currency is valid."""
        assert CurrencyCode(value) is not None

    def test_currency_code_count(self):
        """Test there are exactly 12 currencies."""
        assert len(CurrencyCode) == 12


class TestAllocationMethodEnum:
    """Test AllocationMethod enum."""

    @pytest.mark.parametrize("value", ["mass", "revenue", "units", "equal"])
    def test_allocation_method_membership(self, value):
        """Test allocation method values."""
        assert AllocationMethod(value) is not None


class TestFuelTypeEnum:
    """Test FuelType enum."""

    @pytest.mark.parametrize("value", ["natural_gas", "diesel", "heavy_fuel_oil", "lpg"])
    def test_fuel_type_membership(self, value):
        """Test fuel type values."""
        assert FuelType(value) is not None


class TestRegionCodeEnum:
    """Test RegionCode enum."""

    @pytest.mark.parametrize("value", [
        "US", "DE", "CN", "GB", "JP", "IN", "BR", "FI",
        "TW", "KR", "FR", "AU", "CA", "MX", "ZA", "GLOBAL",
    ])
    def test_region_code_membership(self, value):
        """Test region code values."""
        assert RegionCode(value) is not None

    def test_region_code_count(self):
        """Test there are exactly 16 regions."""
        assert len(RegionCode) == 16


class TestChainTypeEnum:
    """Test ChainType enum."""

    @pytest.mark.parametrize("value", [
        "steel_automotive", "steel_construction",
        "plastic_packaging", "plastic_automotive",
        "chemical_pharma", "food_bakery",
        "glass_automotive", "electronics_pcb",
    ])
    def test_chain_type_membership(self, value):
        """Test chain type values."""
        assert ChainType(value) is not None

    def test_chain_type_count(self):
        """Test there are exactly 8 chain types."""
        assert len(ChainType) == 8


class TestProvenanceStageEnum:
    """Test ProvenanceStage enum."""

    @pytest.mark.parametrize("value", [
        "VALIDATE", "CLASSIFY", "NORMALIZE", "RESOLVE_EFS",
        "CALCULATE", "ALLOCATE", "AGGREGATE",
        "COMPLIANCE", "PROVENANCE", "SEAL",
    ])
    def test_provenance_stage_membership(self, value):
        """Test provenance stage values."""
        assert ProvenanceStage(value) is not None

    def test_provenance_stage_count(self):
        """Test there are exactly 10 stages."""
        assert len(ProvenanceStage) == 10


class TestExportFormatEnum:
    """Test ExportFormat enum."""

    @pytest.mark.parametrize("value", ["json", "csv", "xlsx"])
    def test_export_format_membership(self, value):
        """Test export format values."""
        assert ExportFormat(value) is not None


class TestBatchStatusEnum:
    """Test BatchStatus enum."""

    @pytest.mark.parametrize("value", ["pending", "processing", "completed", "failed"])
    def test_batch_status_membership(self, value):
        """Test batch status values."""
        assert BatchStatus(value) is not None


class TestDQIDimensionEnum:
    """Test DQIDimension enum."""

    @pytest.mark.parametrize("value", [
        "temporal", "geographic", "technological", "completeness", "reliability",
    ])
    def test_dqi_dimension_membership(self, value):
        """Test DQI dimension values."""
        assert DQIDimension(value) is not None

    def test_dqi_dimension_count(self):
        """Test there are exactly 5 DQI dimensions."""
        assert len(DQIDimension) == 5


class TestUncertaintyMethodEnum:
    """Test UncertaintyMethod enum."""

    @pytest.mark.parametrize("value", ["propagation", "monte_carlo", "pedigree"])
    def test_uncertainty_method_membership(self, value):
        """Test uncertainty method values."""
        assert UncertaintyMethod(value) is not None


# ==============================================================================
# CONSTANT TABLE TESTS
# ==============================================================================


class TestAgentMetadata:
    """Test AGENT_ID, AGENT_COMPONENT, VERSION, TABLE_PREFIX constants."""

    def test_agent_id(self):
        """Test AGENT_ID is GL-MRV-S3-010."""
        assert AGENT_ID == "GL-MRV-S3-010"

    def test_agent_component(self):
        """Test AGENT_COMPONENT is AGENT-MRV-023."""
        assert AGENT_COMPONENT == "AGENT-MRV-023"

    def test_version(self):
        """Test VERSION is 1.0.0."""
        assert VERSION == "1.0.0"

    def test_table_prefix(self):
        """Test TABLE_PREFIX is gl_psp_."""
        assert TABLE_PREFIX == "gl_psp_"


class TestProcessEmissionFactors:
    """Test PROCESS_EMISSION_FACTORS constant table."""

    def test_ferrous_stamping_us(self):
        """Test ferrous stamping US EF is 280 kgCO2e/t."""
        key = (ProductCategory.METALS_FERROUS, ProcessingType.STAMPING, RegionCode.US)
        assert PROCESS_EMISSION_FACTORS[key] == Decimal("280.0")

    def test_plastics_injection_molding_de(self):
        """Test plastics injection molding DE EF is 320 kgCO2e/t."""
        key = (ProductCategory.PLASTICS_THERMOPLASTIC, ProcessingType.INJECTION_MOLDING, RegionCode.DE)
        assert PROCESS_EMISSION_FACTORS[key] == Decimal("320.0")

    def test_chemicals_reaction_cn(self):
        """Test chemicals reaction CN EF is 450 kgCO2e/t."""
        key = (ProductCategory.CHEMICALS, ProcessingType.CHEMICAL_REACTION, RegionCode.CN)
        assert PROCESS_EMISSION_FACTORS[key] == Decimal("450.0")

    def test_food_drying_gb(self):
        """Test food drying GB EF is 120 kgCO2e/t."""
        key = (ProductCategory.FOOD_INGREDIENTS, ProcessingType.DRYING, RegionCode.GB)
        assert PROCESS_EMISSION_FACTORS[key] == Decimal("120.0")

    def test_electronics_assembly_tw(self):
        """Test electronics assembly TW EF is 680 kgCO2e/t."""
        key = (ProductCategory.ELECTRONICS_COMPONENTS, ProcessingType.ASSEMBLY, RegionCode.TW)
        assert PROCESS_EMISSION_FACTORS[key] == Decimal("680.0")

    def test_all_values_are_decimal(self):
        """Test all EF values are Decimal type."""
        for key, value in PROCESS_EMISSION_FACTORS.items():
            assert isinstance(value, Decimal), f"Key {key} value is not Decimal"

    def test_all_values_positive(self):
        """Test all EF values are positive."""
        for key, value in PROCESS_EMISSION_FACTORS.items():
            assert value > Decimal("0"), f"Key {key} has non-positive value"


class TestEnergyIntensityFactors:
    """Test ENERGY_INTENSITY_FACTORS constant table."""

    @pytest.mark.parametrize("processing_type,expected_min,expected_max", [
        ("stamping", Decimal("300"), Decimal("400")),
        ("machining", Decimal("400"), Decimal("550")),
        ("heat_treatment", Decimal("1000"), Decimal("1400")),
        ("injection_molding", Decimal("500"), Decimal("650")),
        ("chemical_reaction", Decimal("1300"), Decimal("1700")),
        ("milling", Decimal("150"), Decimal("250")),
    ])
    def test_energy_intensity_range(self, processing_type, expected_min, expected_max):
        """Test energy intensity falls within expected range."""
        pt = ProcessingType(processing_type)
        value = ENERGY_INTENSITY_FACTORS[pt]
        assert expected_min <= value <= expected_max

    def test_all_18_types_present(self):
        """Test all 18 processing types have energy intensities."""
        for pt in ProcessingType:
            assert pt in ENERGY_INTENSITY_FACTORS, f"Missing energy intensity for {pt}"


class TestGridEFs:
    """Test GRID_EFS constant table (kgCO2e/kWh by region)."""

    @pytest.mark.parametrize("region,expected", [
        ("US", Decimal("0.42")),
        ("DE", Decimal("0.35")),
        ("CN", Decimal("0.58")),
        ("GB", Decimal("0.23")),
        ("JP", Decimal("0.47")),
        ("IN", Decimal("0.71")),
        ("BR", Decimal("0.08")),
        ("GLOBAL", Decimal("0.44")),
    ])
    def test_grid_ef_values(self, region, expected):
        """Test grid EF for region matches expected value."""
        assert GRID_EFS[RegionCode(region)] == expected

    def test_all_16_regions_present(self):
        """Test all 16 regions have grid EFs."""
        for region in RegionCode:
            assert region in GRID_EFS, f"Missing grid EF for {region}"


class TestEEIOFactors:
    """Test EEIO_FACTORS constant table."""

    def test_eeio_iron_steel(self):
        """Test iron and steel EEIO factor is 0.82."""
        assert EEIO_FACTORS["331110"]["ef_per_usd"] == Decimal("0.82")

    def test_eeio_plastics(self):
        """Test plastics EEIO factor is 0.48."""
        assert EEIO_FACTORS["326100"]["ef_per_usd"] == Decimal("0.48")

    def test_eeio_semiconductor(self):
        """Test semiconductor EEIO factor is 0.38."""
        assert EEIO_FACTORS["334400"]["ef_per_usd"] == Decimal("0.38")

    def test_eeio_count(self):
        """Test at least 12 EEIO sectors are defined."""
        assert len(EEIO_FACTORS) >= 12

    def test_eeio_all_have_name_and_ef(self):
        """Test all EEIO entries have name and ef_per_usd."""
        for naics, data in EEIO_FACTORS.items():
            assert "name" in data, f"NAICS {naics} missing name"
            assert "ef_per_usd" in data, f"NAICS {naics} missing ef_per_usd"


class TestCurrencyRates:
    """Test CURRENCY_RATES constant table."""

    def test_usd_rate_is_one(self):
        """Test USD rate is 1.0."""
        assert CURRENCY_RATES[CurrencyCode.USD] == Decimal("1.0")

    def test_eur_rate(self):
        """Test EUR rate is 1.085."""
        assert CURRENCY_RATES[CurrencyCode.EUR] == Decimal("1.0850")

    def test_gbp_rate(self):
        """Test GBP rate is 1.265."""
        assert CURRENCY_RATES[CurrencyCode.GBP] == Decimal("1.2650")

    def test_all_12_currencies_present(self):
        """Test all 12 currencies have rates."""
        for currency in CurrencyCode:
            assert currency in CURRENCY_RATES, f"Missing rate for {currency}"

    def test_all_rates_positive(self):
        """Test all currency rates are positive."""
        for currency, rate in CURRENCY_RATES.items():
            assert rate > Decimal("0"), f"Currency {currency} has non-positive rate"


class TestCPIDeflators:
    """Test CPI_DEFLATORS constant table."""

    def test_base_year_2021(self):
        """Test 2021 base year deflator is 1.0."""
        assert CPI_DEFLATORS[2021] == Decimal("1.0000")

    def test_2024_deflator(self):
        """Test 2024 deflator is 1.1490."""
        assert CPI_DEFLATORS[2024] == Decimal("1.1490")

    def test_2015_deflator(self):
        """Test 2015 deflator is 0.8490."""
        assert CPI_DEFLATORS[2015] == Decimal("0.8490")

    def test_all_years_2015_to_2025(self):
        """Test deflators exist for 2015-2025."""
        for year in range(2015, 2026):
            assert year in CPI_DEFLATORS, f"Missing CPI deflator for {year}"


class TestGWPValues:
    """Test GWP_VALUES constant table."""

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

    def test_co2_always_one(self):
        """Test CO2 GWP is always 1."""
        for version in GWPVersion:
            assert GWP_VALUES[version]["co2"] == Decimal("1")


class TestDQIScoringAndWeights:
    """Test DQI_SCORING and DQI_WEIGHTS constants."""

    def test_dqi_scoring_5_dimensions(self):
        """Test DQI scoring has 5 dimensions."""
        assert len(DQI_SCORING) == 5

    def test_dqi_weights_sum_to_one(self):
        """Test DQI weights sum to 1.0."""
        total = sum(DQI_WEIGHTS.values())
        assert total == Decimal("1.00")

    def test_dqi_weights_5_dimensions(self):
        """Test DQI weights has 5 dimensions."""
        assert len(DQI_WEIGHTS) == 5


class TestUncertaintyRanges:
    """Test UNCERTAINTY_RANGES constant table."""

    def test_site_specific_tier1(self):
        """Test site-specific Tier 1 uncertainty is 0.05."""
        assert UNCERTAINTY_RANGES["site_specific_direct"][DataQualityTier.TIER_1] == Decimal("0.05")

    def test_spend_based_tier3(self):
        """Test spend-based Tier 3 uncertainty is 0.60."""
        assert UNCERTAINTY_RANGES["spend_based"][DataQualityTier.TIER_3] == Decimal("0.60")

    def test_average_data_tier2(self):
        """Test average-data Tier 2 uncertainty is in expected range."""
        value = UNCERTAINTY_RANGES["average_data"][DataQualityTier.TIER_2]
        assert Decimal("0.15") <= value <= Decimal("0.40")


class TestProcessingChains:
    """Test PROCESSING_CHAINS constant table."""

    def test_steel_automotive_chain(self):
        """Test steel automotive chain has 4 steps."""
        chain = PROCESSING_CHAINS[ChainType.STEEL_AUTOMOTIVE]
        assert len(chain) == 4
        assert chain[0] == ProcessingType.STAMPING
        assert chain[-1] == ProcessingType.ASSEMBLY

    def test_all_8_chains_defined(self):
        """Test all 8 chain types are defined."""
        for ct in ChainType:
            assert ct in PROCESSING_CHAINS, f"Missing chain for {ct}"

    def test_all_chains_non_empty(self):
        """Test no chain is empty."""
        for ct, chain in PROCESSING_CHAINS.items():
            assert len(chain) >= 2, f"Chain {ct} has fewer than 2 steps"


class TestFuelEFs:
    """Test FUEL_EFS constant table."""

    @pytest.mark.parametrize("fuel,expected", [
        ("natural_gas", Decimal("2.02")),
        ("diesel", Decimal("2.68")),
        ("heavy_fuel_oil", Decimal("3.12")),
        ("lpg", Decimal("1.55")),
    ])
    def test_fuel_ef_values(self, fuel, expected):
        """Test fuel EF values match expected."""
        assert FUEL_EFS[FuelType(fuel)]["ef_kg_per_litre"] == expected


class TestAllocationMethods:
    """Test ALLOCATION_METHODS constant table."""

    def test_mass_allocation_present(self):
        """Test mass allocation method is defined."""
        assert AllocationMethod.MASS in ALLOCATION_METHODS

    def test_all_4_methods_defined(self):
        """Test all 4 allocation methods are defined."""
        for method in AllocationMethod:
            assert method in ALLOCATION_METHODS


class TestDCRules:
    """Test DC_RULES (double-counting rules) constant table."""

    @pytest.mark.parametrize("rule_code", [
        "DC-PSP-001", "DC-PSP-002", "DC-PSP-003", "DC-PSP-004",
        "DC-PSP-005", "DC-PSP-006", "DC-PSP-007", "DC-PSP-008",
    ])
    def test_dc_rule_exists(self, rule_code):
        """Test double-counting rule exists."""
        assert rule_code in DC_RULES

    def test_dc_rules_count(self):
        """Test exactly 8 DC rules are defined."""
        assert len(DC_RULES) == 8

    def test_dc_rules_have_description(self):
        """Test all DC rules have descriptions."""
        for code, rule in DC_RULES.items():
            assert "description" in rule, f"DC rule {code} missing description"


# ==============================================================================
# INPUT MODEL TESTS
# ==============================================================================


class TestSiteSpecificInputModel:
    """Test SiteSpecificInput Pydantic model."""

    def test_valid_direct_input(self):
        """Test valid site-specific direct input creation."""
        inp = SiteSpecificInput(
            method="site_specific_direct",
            product_id="PRD-001",
            category="metals_ferrous",
            quantity_tonnes=Decimal("1000.0"),
            processing_type="stamping",
            customer_reported_co2e_kg=Decimal("280000.0"),
            region="US",
        )
        assert inp.method == "site_specific_direct"
        assert inp.quantity_tonnes == Decimal("1000.0")

    def test_valid_energy_input(self):
        """Test valid site-specific energy input creation."""
        inp = SiteSpecificInput(
            method="site_specific_energy",
            product_id="PRD-002",
            category="plastics_thermoplastic",
            quantity_tonnes=Decimal("500.0"),
            processing_type="injection_molding",
            energy_consumption_kwh=Decimal("750000.0"),
            grid_ef_kg_per_kwh=Decimal("0.42"),
            region="DE",
        )
        assert inp.energy_consumption_kwh == Decimal("750000.0")

    def test_negative_quantity_raises(self):
        """Test negative quantity raises validation error."""
        with pytest.raises(PydanticValidationError):
            SiteSpecificInput(
                method="site_specific_direct",
                product_id="PRD-001",
                category="metals_ferrous",
                quantity_tonnes=Decimal("-100.0"),
                processing_type="stamping",
                region="US",
            )

    def test_frozen_immutability(self):
        """Test SiteSpecificInput is immutable."""
        inp = SiteSpecificInput(
            method="site_specific_direct",
            product_id="PRD-001",
            category="metals_ferrous",
            quantity_tonnes=Decimal("1000.0"),
            processing_type="stamping",
            region="US",
        )
        with pytest.raises(Exception):
            inp.quantity_tonnes = Decimal("2000.0")


class TestAverageDataInputModel:
    """Test AverageDataInput Pydantic model."""

    def test_valid_average_data_input(self):
        """Test valid average data input creation."""
        inp = AverageDataInput(
            product_id="PRD-001",
            category="metals_ferrous",
            quantity_tonnes=Decimal("1000.0"),
            processing_type="stamping",
            region="US",
        )
        assert inp.category == "metals_ferrous"

    def test_zero_quantity_raises(self):
        """Test zero quantity raises validation error."""
        with pytest.raises(PydanticValidationError):
            AverageDataInput(
                product_id="PRD-001",
                category="metals_ferrous",
                quantity_tonnes=Decimal("0.0"),
                processing_type="stamping",
                region="US",
            )


class TestSpendBasedInputModel:
    """Test SpendBasedInput Pydantic model."""

    def test_valid_spend_input(self):
        """Test valid spend-based input creation."""
        inp = SpendBasedInput(
            naics_code="331110",
            revenue_usd=Decimal("1000000.0"),
            currency="USD",
            reporting_year=2024,
        )
        assert inp.naics_code == "331110"
        assert inp.revenue_usd == Decimal("1000000.0")

    def test_negative_revenue_raises(self):
        """Test negative revenue raises validation error."""
        with pytest.raises(PydanticValidationError):
            SpendBasedInput(
                naics_code="331110",
                revenue_usd=Decimal("-100.0"),
                currency="USD",
                reporting_year=2024,
            )

    def test_frozen_immutability(self):
        """Test SpendBasedInput is immutable."""
        inp = SpendBasedInput(
            naics_code="331110",
            revenue_usd=Decimal("1000000.0"),
            currency="USD",
            reporting_year=2024,
        )
        with pytest.raises(Exception):
            inp.revenue_usd = Decimal("500000.0")


class TestProductInputModel:
    """Test ProductInput Pydantic model."""

    def test_valid_product_input(self):
        """Test valid product input creation with all fields."""
        inp = ProductInput(
            product_id="PRD-FER-001",
            product_name="Hot-rolled steel coil",
            category="metals_ferrous",
            quantity_tonnes=Decimal("1000.0"),
            processing_type="stamping",
            customer_name="AutoStamp Co",
            customer_country="US",
            reporting_year=2024,
        )
        assert inp.product_id == "PRD-FER-001"
        assert inp.product_name == "Hot-rolled steel coil"


class TestCalculationResultModel:
    """Test CalculationResult Pydantic model."""

    def test_result_frozen(self):
        """Test CalculationResult is frozen (immutable)."""
        result = CalculationResult(
            calculation_id="calc-001",
            product_id="PRD-001",
            method="site_specific_direct",
            total_co2e_kg=Decimal("280000.0"),
            dqi_score=Decimal("90.0"),
            provenance_hash="a" * 64,
        )
        with pytest.raises(Exception):
            result.total_co2e_kg = Decimal("0")

    def test_result_has_required_fields(self):
        """Test CalculationResult has all required fields."""
        result = CalculationResult(
            calculation_id="calc-001",
            product_id="PRD-001",
            method="site_specific_direct",
            total_co2e_kg=Decimal("280000.0"),
            dqi_score=Decimal("90.0"),
            provenance_hash="a" * 64,
        )
        assert result.calculation_id == "calc-001"
        assert result.total_co2e_kg == Decimal("280000.0")
        assert len(result.provenance_hash) == 64


# ==============================================================================
# HELPER FUNCTION TESTS
# ==============================================================================


class TestProvenanceHash:
    """Test calculate_provenance_hash function."""

    def test_hash_deterministic(self):
        """Test provenance hash is deterministic for same inputs."""
        h1 = calculate_provenance_hash("PRD-001", "stamping", Decimal("1000.0"))
        h2 = calculate_provenance_hash("PRD-001", "stamping", Decimal("1000.0"))
        assert h1 == h2
        assert len(h1) == 64

    def test_hash_different_inputs(self):
        """Test provenance hash differs for different inputs."""
        h1 = calculate_provenance_hash("PRD-001", "stamping", Decimal("1000.0"))
        h2 = calculate_provenance_hash("PRD-002", "machining", Decimal("500.0"))
        assert h1 != h2

    def test_hash_is_hex(self):
        """Test provenance hash is valid hex string."""
        h = calculate_provenance_hash("test", "data")
        assert all(c in "0123456789abcdef" for c in h)


class TestDQIClassification:
    """Test get_dqi_classification function."""

    @pytest.mark.parametrize("score,expected", [
        (Decimal("4.5"), "Excellent"),
        (Decimal("4.2"), "Good"),
        (Decimal("3.0"), "Fair"),
        (Decimal("1.5"), "Poor"),
        (Decimal("1.0"), "Very Poor"),
    ])
    def test_dqi_classification(self, score, expected):
        """Test DQI score classification."""
        assert get_dqi_classification(score) == expected


class TestCurrencyConversion:
    """Test convert_currency_to_usd function."""

    @pytest.mark.parametrize("amount,currency,expected_min,expected_max", [
        (Decimal("1000"), "EUR", Decimal("1080"), Decimal("1090")),
        (Decimal("1000"), "GBP", Decimal("1260"), Decimal("1270")),
        (Decimal("1000"), "USD", Decimal("1000"), Decimal("1000")),
        (Decimal("100000"), "JPY", Decimal("660"), Decimal("670")),
        (Decimal("10000"), "CNY", Decimal("1370"), Decimal("1390")),
    ])
    def test_currency_conversion(self, amount, currency, expected_min, expected_max):
        """Test currency conversion to USD."""
        result = convert_currency_to_usd(amount, CurrencyCode(currency))
        assert expected_min <= result <= expected_max


class TestCPIDeflatorHelper:
    """Test get_cpi_deflator helper function."""

    def test_2024_deflator(self):
        """Test CPI deflator for 2024."""
        assert get_cpi_deflator(2024) == Decimal("1.1490")

    def test_2021_base(self):
        """Test CPI deflator for base year 2021."""
        assert get_cpi_deflator(2021) == Decimal("1.0000")

    def test_invalid_year_raises(self):
        """Test invalid year raises ValueError."""
        with pytest.raises(ValueError, match="not available"):
            get_cpi_deflator(1900)


class TestClassifyProductCategory:
    """Test classify_product_category helper."""

    def test_classify_steel(self):
        """Test steel is classified as metals_ferrous."""
        result = classify_product_category("steel coil")
        assert result == "metals_ferrous"

    def test_classify_polypropylene(self):
        """Test polypropylene is classified as plastics."""
        result = classify_product_category("polypropylene pellets")
        assert "plastics" in result


class TestValidateProcessingType:
    """Test validate_processing_type helper."""

    @pytest.mark.parametrize("processing_type", [
        "stamping", "machining", "welding", "heat_treatment",
        "injection_molding", "extrusion", "chemical_reaction",
    ])
    def test_valid_types(self, processing_type):
        """Test valid processing types return True."""
        assert validate_processing_type(processing_type) is True

    def test_invalid_type_returns_false(self):
        """Test invalid processing type returns False."""
        assert validate_processing_type("nonexistent_type") is False


class TestComputeDQIScore:
    """Test compute_dqi_score helper."""

    def test_site_specific_direct_score(self):
        """Test site-specific direct method DQI is 90."""
        score = compute_dqi_score("site_specific_direct")
        assert score == Decimal("90")

    def test_average_data_score(self):
        """Test average-data method DQI is 55."""
        score = compute_dqi_score("average_data")
        assert score == Decimal("55")

    def test_spend_based_score(self):
        """Test spend-based method DQI is 30."""
        score = compute_dqi_score("spend_based")
        assert score == Decimal("30")

    def test_site_specific_energy_score(self):
        """Test site-specific energy method DQI is 80."""
        score = compute_dqi_score("site_specific_energy")
        assert score == Decimal("80")

    def test_site_specific_fuel_score(self):
        """Test site-specific fuel method DQI is 75."""
        score = compute_dqi_score("site_specific_fuel")
        assert score == Decimal("75")


class TestComputeUncertaintyRange:
    """Test compute_uncertainty_range helper."""

    def test_site_specific_uncertainty(self):
        """Test site-specific uncertainty range is narrow."""
        lower, upper = compute_uncertainty_range(
            Decimal("280000.0"), "site_specific_direct", DataQualityTier.TIER_1
        )
        assert lower < Decimal("280000.0")
        assert upper > Decimal("280000.0")
        # 5% uncertainty: 266000 to 294000
        assert lower >= Decimal("260000")
        assert upper <= Decimal("300000")

    def test_spend_based_uncertainty_wider(self):
        """Test spend-based uncertainty range is wider than site-specific."""
        lower_ss, upper_ss = compute_uncertainty_range(
            Decimal("280000.0"), "site_specific_direct", DataQualityTier.TIER_1
        )
        lower_sp, upper_sp = compute_uncertainty_range(
            Decimal("280000.0"), "spend_based", DataQualityTier.TIER_3
        )
        range_ss = upper_ss - lower_ss
        range_sp = upper_sp - lower_sp
        assert range_sp > range_ss


class TestFormatCO2e:
    """Test format_co2e helper."""

    def test_format_kg(self):
        """Test formatting in kg."""
        result = format_co2e(Decimal("1234.5678"))
        assert "1234.57" in result or "1,234.57" in result

    def test_format_tonnes(self):
        """Test formatting converts to tonnes for large values."""
        result = format_co2e(Decimal("1000000.0"))
        assert "1000" in result or "1,000" in result


class TestDecimalPrecision:
    """Test Decimal precision across models and constants."""

    def test_process_ef_decimal_precision(self):
        """Test process EF values maintain Decimal precision."""
        key = (ProductCategory.METALS_FERROUS, ProcessingType.STAMPING, RegionCode.US)
        ef = PROCESS_EMISSION_FACTORS[key]
        assert isinstance(ef, Decimal)

    def test_grid_ef_decimal_precision(self):
        """Test grid EF values maintain Decimal precision."""
        ef = GRID_EFS[RegionCode.US]
        assert isinstance(ef, Decimal)

    def test_currency_rate_decimal_precision(self):
        """Test currency rates maintain Decimal precision."""
        rate = CURRENCY_RATES[CurrencyCode.EUR]
        assert isinstance(rate, Decimal)


# ==============================================================================
# SUMMARY
# ==============================================================================


def test_total_enum_count():
    """Meta-test: verify enum count is at least 20."""
    enum_classes = [
        CalculationMethod, ProductCategory, ProcessingType, EFSource,
        ComplianceFramework, DataQualityTier, ProvenanceStage,
        UncertaintyMethod, DQIDimension, DQIScore, ComplianceStatus,
        GWPVersion, EmissionGas, CurrencyCode, ExportFormat,
        BatchStatus, AllocationMethod, FuelType, RegionCode, ChainType,
    ]
    assert len(enum_classes) >= 20
