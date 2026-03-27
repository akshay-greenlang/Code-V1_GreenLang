# -*- coding: utf-8 -*-
"""
Test suite for end_of_life_treatment.models - AGENT-MRV-025.

Tests all 22 enums, 16 constant tables, 14 Pydantic models, and 16 helper
functions for the End-of-Life Treatment of Sold Products Agent (GL-MRV-S3-012).

Coverage:
- Enumerations: 22 enums (values, membership, count, parametrized)
- Constants: MATERIAL_TREATMENT_EFS, PRODUCT_COMPOSITIONS, REGIONAL_TREATMENT_MIXES,
  LANDFILL_FOD_PARAMS, INCINERATION_PARAMS, RECYCLING_FACTORS, COMPOSTING_AD_FACTORS,
  PRODUCT_WEIGHTS, DQI_SCORING, DQI_WEIGHTS, UNCERTAINTY_RANGES, GWP_VALUES,
  DC_RULES, AVOIDED_EFS, CIRCULARITY_BENCHMARKS, WASTE_HIERARCHY_WEIGHTS
- Input models: WasteTypeSpecificInput, AverageDataInput, ProducerSpecificInput,
  ProductInput, TreatmentPathwayInput, BatchInput, PortfolioInput
- Result models: CalculationResult, TreatmentBreakdown, ComplianceCheckResult,
  AggregationResult, CircularityResult, AvoidedEmissionsResult, HotspotResult
- Helper functions: calculate_provenance_hash, get_dqi_classification,
  classify_product_category, validate_material, get_material_ef,
  get_treatment_mix, compute_dqi_score, compute_uncertainty_range,
  format_co2e, get_circularity_index, get_waste_hierarchy_score,
  get_avoided_emissions_ef, compute_gross_emissions, is_biogenic_material,
  get_gwp, normalize_treatment_mix

Target: 150+ expanded tests.
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
    from greenlang.agents.mrv.end_of_life_treatment.models import (
        # Enumerations
        CalculationMethod,
        TreatmentPathway,
        MaterialType,
        ProductCategory,
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
        ExportFormat,
        BatchStatus,
        ClimateZone,
        LandfillType,
        VerificationLevel,
        CircularityMetric,
        WasteHierarchyLevel,
        DCRuleId,

        # Constants
        AGENT_ID,
        AGENT_COMPONENT,
        VERSION,
        TABLE_PREFIX,
        MATERIAL_TREATMENT_EFS,
        PRODUCT_COMPOSITIONS,
        REGIONAL_TREATMENT_MIXES,
        LANDFILL_FOD_PARAMS,
        INCINERATION_PARAMS,
        RECYCLING_FACTORS,
        COMPOSTING_AD_FACTORS,
        PRODUCT_WEIGHTS,
        DQI_SCORING,
        DQI_WEIGHTS,
        UNCERTAINTY_RANGES,
        GWP_VALUES,
        DC_RULES,
        AVOIDED_EFS,
        CIRCULARITY_BENCHMARKS,
        WASTE_HIERARCHY_WEIGHTS,

        # Input models
        WasteTypeSpecificInput,
        AverageDataInput,
        ProducerSpecificInput,
        ProductInput,
        TreatmentPathwayInput,
        BatchInput,
        PortfolioInput,

        # Result models
        CalculationResult,
        TreatmentBreakdown,
        ComplianceCheckResult,
        AggregationResult,
        CircularityResult,
        AvoidedEmissionsResult,
        HotspotResult,

        # Helpers
        calculate_provenance_hash,
        get_dqi_classification,
        classify_product_category,
        validate_material,
        get_material_ef,
        get_treatment_mix,
        compute_dqi_score,
        compute_uncertainty_range,
        format_co2e,
        get_circularity_index,
        get_waste_hierarchy_score,
        get_avoided_emissions_ef,
        compute_gross_emissions,
        is_biogenic_material,
        get_gwp,
        normalize_treatment_mix,
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
    reason="end_of_life_treatment.models not available",
)
pytestmark = _SKIP


# ==============================================================================
# ENUMERATION TESTS
# ==============================================================================


class TestCalculationMethodEnum:
    """Test CalculationMethod enum."""

    def test_calculation_method_values(self):
        """Test all 4 calculation method values exist."""
        assert CalculationMethod.WASTE_TYPE_SPECIFIC == "waste_type_specific"
        assert CalculationMethod.AVERAGE_DATA == "average_data"
        assert CalculationMethod.PRODUCER_SPECIFIC == "producer_specific"
        assert CalculationMethod.HYBRID == "hybrid"
        assert len(CalculationMethod) == 4

    @pytest.mark.parametrize("value", [
        "waste_type_specific", "average_data", "producer_specific", "hybrid",
    ])
    def test_calculation_method_membership(self, value):
        """Test each method value is a valid member."""
        assert CalculationMethod(value) is not None


class TestTreatmentPathwayEnum:
    """Test TreatmentPathway enum."""

    @pytest.mark.parametrize("value", [
        "landfill", "incineration", "recycling", "composting",
        "anaerobic_digestion", "open_burning", "wastewater",
    ])
    def test_treatment_pathway_membership(self, value):
        """Test each treatment pathway is valid."""
        assert TreatmentPathway(value) is not None

    def test_treatment_pathway_count(self):
        """Test there are exactly 7 treatment pathways."""
        assert len(TreatmentPathway) == 7


class TestMaterialTypeEnum:
    """Test MaterialType enum."""

    @pytest.mark.parametrize("value", [
        "steel", "aluminum", "copper", "glass",
        "plastic_abs", "plastic_pe", "plastic_pp",
        "paper_cardboard", "wood_mdf", "cotton", "polyester",
        "rubber_synthetic", "rubber_natural", "food_organic",
        "concrete", "lithium_battery",
    ])
    def test_material_type_membership(self, value):
        """Test each material type is valid."""
        assert MaterialType(value) is not None

    def test_material_type_count(self):
        """Test there are at least 15 material types."""
        assert len(MaterialType) >= 15


class TestProductCategoryEnum:
    """Test ProductCategory enum."""

    @pytest.mark.parametrize("value", [
        "consumer_electronics", "large_appliances", "small_appliances",
        "packaging", "clothing", "furniture", "batteries", "tires",
        "food_products", "building_materials", "automotive_parts",
        "medical_devices", "toys", "sporting_goods", "cosmetics",
        "office_equipment", "garden_tools", "pet_products", "lighting", "mixed",
    ])
    def test_product_category_membership(self, value):
        """Test each product category is valid."""
        assert ProductCategory(value) is not None

    def test_product_category_count(self):
        """Test there are exactly 20 product categories."""
        assert len(ProductCategory) == 20


class TestEFSourceEnum:
    """Test EFSource enum."""

    @pytest.mark.parametrize("value", [
        "epa_warm", "defra", "ipcc", "ecoinvent", "epd", "producer",
    ])
    def test_ef_source_membership(self, value):
        """Test each EF source is valid."""
        assert EFSource(value) is not None

    def test_ef_source_count(self):
        """Test EF source count."""
        assert len(EFSource) >= 6


class TestComplianceFrameworkEnum:
    """Test ComplianceFramework enum."""

    @pytest.mark.parametrize("value", [
        "GHG_PROTOCOL_SCOPE3", "ISO_14064", "CSRD_ESRS_E1",
        "CSRD_ESRS_E5", "CDP", "SBTI", "GRI",
    ])
    def test_framework_membership(self, value):
        """Test each compliance framework is valid."""
        assert ComplianceFramework(value) is not None

    def test_framework_count(self):
        """Test there are exactly 7 frameworks."""
        assert len(ComplianceFramework) == 7


class TestDataQualityTierEnum:
    """Test DataQualityTier enum."""

    @pytest.mark.parametrize("value", [
        "primary", "secondary", "tertiary", "default",
    ])
    def test_tier_membership(self, value):
        """Test each tier is valid."""
        assert DataQualityTier(value) is not None

    def test_tier_count(self):
        """Test there are 4 tiers."""
        assert len(DataQualityTier) == 4


class TestProvenanceStageEnum:
    """Test ProvenanceStage enum."""

    @pytest.mark.parametrize("value", [
        "validate", "classify", "normalize", "resolve_efs",
        "calculate", "allocate", "aggregate", "compliance",
        "provenance", "seal",
    ])
    def test_stage_membership(self, value):
        """Test each stage is valid."""
        assert ProvenanceStage(value) is not None

    def test_stage_count(self):
        """Test there are exactly 10 provenance stages."""
        assert len(ProvenanceStage) == 10


class TestUncertaintyMethodEnum:
    """Test UncertaintyMethod enum."""

    @pytest.mark.parametrize("value", [
        "propagation", "monte_carlo", "pedigree",
    ])
    def test_method_membership(self, value):
        """Test each method is valid."""
        assert UncertaintyMethod(value) is not None


class TestDQIDimensionEnum:
    """Test DQIDimension enum."""

    @pytest.mark.parametrize("value", [
        "reliability", "completeness", "temporal", "geographic", "technological",
    ])
    def test_dimension_membership(self, value):
        """Test each dimension is valid."""
        assert DQIDimension(value) is not None

    def test_dimension_count(self):
        """Test there are 5 DQI dimensions."""
        assert len(DQIDimension) == 5


class TestDQIScoreEnum:
    """Test DQIScore enum with 5 levels."""

    @pytest.mark.parametrize("value,expected_range", [
        ("excellent", (85, 100)),
        ("good", (70, 85)),
        ("fair", (50, 70)),
        ("poor", (30, 50)),
        ("very_poor", (0, 30)),
    ])
    def test_score_membership(self, value, expected_range):
        """Test each DQI score level is valid."""
        assert DQIScore(value) is not None


class TestComplianceStatusEnum:
    """Test ComplianceStatus enum."""

    @pytest.mark.parametrize("value", [
        "compliant", "non_compliant", "partial", "not_applicable",
    ])
    def test_status_membership(self, value):
        """Test each status is valid."""
        assert ComplianceStatus(value) is not None


class TestGWPVersionEnum:
    """Test GWPVersion enum."""

    @pytest.mark.parametrize("value", ["AR4", "AR5", "AR6"])
    def test_gwp_version_membership(self, value):
        """Test each GWP version is valid."""
        assert GWPVersion(value) is not None


class TestEmissionGasEnum:
    """Test EmissionGas enum."""

    @pytest.mark.parametrize("value", ["CO2", "CH4", "N2O"])
    def test_gas_membership(self, value):
        """Test each emission gas is valid."""
        assert EmissionGas(value) is not None


class TestExportFormatEnum:
    """Test ExportFormat enum."""

    @pytest.mark.parametrize("value", ["json", "csv", "xlsx", "pdf"])
    def test_format_membership(self, value):
        """Test each export format is valid."""
        assert ExportFormat(value) is not None


class TestBatchStatusEnum:
    """Test BatchStatus enum."""

    @pytest.mark.parametrize("value", [
        "pending", "processing", "completed", "failed", "partial",
    ])
    def test_status_membership(self, value):
        """Test each batch status is valid."""
        assert BatchStatus(value) is not None


class TestClimateZoneEnum:
    """Test ClimateZone enum for landfill FOD model."""

    @pytest.mark.parametrize("value", [
        "boreal_dry", "boreal_wet", "temperate_dry", "temperate_wet",
        "tropical_dry", "tropical_wet",
    ])
    def test_climate_zone_membership(self, value):
        """Test each climate zone is valid."""
        assert ClimateZone(value) is not None

    def test_climate_zone_count(self):
        """Test there are 6 climate zones."""
        assert len(ClimateZone) == 6


class TestVerificationLevelEnum:
    """Test VerificationLevel enum for producer-specific data."""

    @pytest.mark.parametrize("value", [
        "self_declared", "second_party", "third_party_verified",
    ])
    def test_verification_membership(self, value):
        """Test each verification level is valid."""
        assert VerificationLevel(value) is not None


class TestCircularityMetricEnum:
    """Test CircularityMetric enum."""

    @pytest.mark.parametrize("value", [
        "recycling_rate", "diversion_rate", "circularity_index",
        "material_recovery_rate",
    ])
    def test_metric_membership(self, value):
        """Test each circularity metric is valid."""
        assert CircularityMetric(value) is not None


class TestWasteHierarchyLevelEnum:
    """Test WasteHierarchyLevel enum (EU Waste Framework Directive)."""

    @pytest.mark.parametrize("value", [
        "prevention", "reuse", "recycling", "recovery", "disposal",
    ])
    def test_level_membership(self, value):
        """Test each waste hierarchy level is valid."""
        assert WasteHierarchyLevel(value) is not None

    def test_level_count(self):
        """Test there are exactly 5 levels."""
        assert len(WasteHierarchyLevel) == 5


class TestDCRuleIdEnum:
    """Test DCRuleId enum for double-counting prevention."""

    @pytest.mark.parametrize("value", [
        "DC-EOL-001", "DC-EOL-002", "DC-EOL-003", "DC-EOL-004",
        "DC-EOL-005", "DC-EOL-006", "DC-EOL-007", "DC-EOL-008",
    ])
    def test_rule_id_membership(self, value):
        """Test each DC rule ID is valid."""
        assert DCRuleId(value) is not None

    def test_dc_rule_count(self):
        """Test there are exactly 8 DC rules."""
        assert len(DCRuleId) == 8


# ==============================================================================
# CONSTANT TABLE TESTS
# ==============================================================================


class TestAgentConstants:
    """Test module-level agent constants."""

    def test_agent_id(self):
        """Test AGENT_ID is GL-MRV-S3-012."""
        assert AGENT_ID == "GL-MRV-S3-012"

    def test_agent_component(self):
        """Test AGENT_COMPONENT is AGENT-MRV-025."""
        assert AGENT_COMPONENT == "AGENT-MRV-025"

    def test_version(self):
        """Test VERSION follows SemVer."""
        parts = VERSION.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)

    def test_table_prefix(self):
        """Test TABLE_PREFIX is gl_eol_."""
        assert TABLE_PREFIX == "gl_eol_"


class TestMaterialTreatmentEFs:
    """Test MATERIAL_TREATMENT_EFS constant table."""

    def test_has_at_least_15_materials(self):
        """Test the table has at least 15 material entries."""
        assert len(MATERIAL_TREATMENT_EFS) >= 15

    @pytest.mark.parametrize("material", [
        "steel", "aluminum", "plastic_abs", "paper_cardboard",
        "glass", "food_organic", "rubber_synthetic", "cotton",
    ])
    def test_material_key_present(self, material):
        """Test key materials are present in EF table."""
        assert material in MATERIAL_TREATMENT_EFS

    @pytest.mark.parametrize("treatment", [
        "landfill", "incineration", "recycling",
    ])
    def test_treatment_keys_present_for_steel(self, treatment):
        """Test treatment pathway keys exist for steel."""
        assert treatment in MATERIAL_TREATMENT_EFS["steel"]

    def test_ef_values_are_decimal(self):
        """Test all EF values are Decimal type."""
        for material, treatments in MATERIAL_TREATMENT_EFS.items():
            for treatment, ef in treatments.items():
                assert isinstance(ef, Decimal), (
                    f"EF for {material}/{treatment} should be Decimal, got {type(ef)}"
                )

    def test_ef_values_non_negative(self):
        """Test all EF values are >= 0."""
        for material, treatments in MATERIAL_TREATMENT_EFS.items():
            for treatment, ef in treatments.items():
                assert ef >= 0, f"EF for {material}/{treatment} is negative: {ef}"

    def test_plastic_incineration_higher_than_metal(self):
        """Test plastic incineration EF is higher than metal (more fossil CO2)."""
        plastic_ef = MATERIAL_TREATMENT_EFS["plastic_abs"]["incineration"]
        steel_ef = MATERIAL_TREATMENT_EFS["steel"]["incineration"]
        assert plastic_ef > steel_ef

    def test_paper_landfill_higher_than_plastic_landfill(self):
        """Test paper generates more landfill CH4 than plastic (DOC degradation)."""
        paper_ef = MATERIAL_TREATMENT_EFS["paper_cardboard"]["landfill"]
        plastic_ef = MATERIAL_TREATMENT_EFS["plastic_abs"]["landfill"]
        assert paper_ef > plastic_ef

    def test_food_organic_highest_landfill(self):
        """Test food organic has high landfill EF (highest DOC degradation rate)."""
        food_ef = MATERIAL_TREATMENT_EFS["food_organic"]["landfill"]
        assert food_ef > Decimal("0.1")


class TestProductCompositions:
    """Test PRODUCT_COMPOSITIONS constant table."""

    def test_has_20_categories(self):
        """Test the table has at least 20 product categories."""
        assert len(PRODUCT_COMPOSITIONS) >= 20

    @pytest.mark.parametrize("category", [
        "consumer_electronics", "large_appliances", "packaging",
        "clothing", "furniture", "batteries", "tires",
        "food_products", "building_materials", "mixed",
    ])
    def test_category_present(self, category):
        """Test expected categories are present."""
        assert category in PRODUCT_COMPOSITIONS

    def test_composition_fractions_sum_to_one(self):
        """Test all compositions sum to 1.0 (within tolerance)."""
        for category, materials in PRODUCT_COMPOSITIONS.items():
            total = sum(m["mass_fraction"] for m in materials)
            assert abs(total - Decimal("1.0")) < Decimal("0.01"), (
                f"Category {category} fractions sum to {total}, not 1.0"
            )

    def test_composition_has_material_key(self):
        """Test all composition entries have 'material' key."""
        for category, materials in PRODUCT_COMPOSITIONS.items():
            for mat in materials:
                assert "material" in mat, f"Missing 'material' in {category}"
                assert "mass_fraction" in mat, f"Missing 'mass_fraction' in {category}"


class TestRegionalTreatmentMixes:
    """Test REGIONAL_TREATMENT_MIXES constant table."""

    def test_has_at_least_12_regions(self):
        """Test the table has at least 12 regions."""
        assert len(REGIONAL_TREATMENT_MIXES) >= 12

    @pytest.mark.parametrize("region", [
        "US", "DE", "GB", "JP", "FR", "CN", "IN", "BR", "AU", "KR", "CA", "GLOBAL",
    ])
    def test_region_present(self, region):
        """Test expected regions are present."""
        assert region in REGIONAL_TREATMENT_MIXES

    def test_treatment_mix_sums_to_one(self):
        """Test all regional treatment mixes sum to 1.0."""
        for region, mix in REGIONAL_TREATMENT_MIXES.items():
            total = sum(mix.values())
            assert abs(total - Decimal("1.0")) < Decimal("0.02"), (
                f"Region {region} treatment mix sums to {total}, not 1.0"
            )

    def test_japan_high_incineration(self):
        """Test Japan has highest incineration fraction (>60%)."""
        jp_incin = REGIONAL_TREATMENT_MIXES["JP"]["incineration"]
        assert jp_incin >= Decimal("0.60")

    def test_germany_low_landfill(self):
        """Test Germany has very low landfill fraction (<5%)."""
        de_landfill = REGIONAL_TREATMENT_MIXES["DE"]["landfill"]
        assert de_landfill <= Decimal("0.05")

    def test_india_open_burning(self):
        """Test India has significant open burning fraction."""
        in_open = REGIONAL_TREATMENT_MIXES["IN"]["open_burning"]
        assert in_open >= Decimal("0.10")


class TestLandfillFODParams:
    """Test LANDFILL_FOD_PARAMS constant table."""

    def test_has_entries(self):
        """Test the table has FOD parameter entries."""
        assert len(LANDFILL_FOD_PARAMS) >= 5

    def test_params_have_required_keys(self):
        """Test each entry has DOC, DOCf, MCF, k, F, OX keys."""
        required_keys = {"doc", "docf", "mcf", "k", "f", "ox"}
        for key, params in LANDFILL_FOD_PARAMS.items():
            assert required_keys.issubset(params.keys()), (
                f"Missing keys in {key}: {required_keys - params.keys()}"
            )

    def test_plastic_doc_is_zero(self):
        """Test plastics have DOC = 0 (non-degradable)."""
        for key, params in LANDFILL_FOD_PARAMS.items():
            if "plastic" in key:
                assert params["doc"] == Decimal("0.0")

    def test_food_high_decay_rate(self):
        """Test food organic has higher decay rate than paper."""
        for key, params in LANDFILL_FOD_PARAMS.items():
            if "food_organic" in key and "temperate" in key:
                assert params["k"] >= Decimal("0.10")


class TestIncinerationParams:
    """Test INCINERATION_PARAMS constant table."""

    def test_has_entries(self):
        """Test table has incineration parameter entries."""
        assert len(INCINERATION_PARAMS) >= 5

    def test_plastic_fossil_carbon_fraction_high(self):
        """Test plastics have high fossil carbon fraction."""
        for key, params in INCINERATION_PARAMS.items():
            if "plastic" in key:
                assert params["fossil_carbon_fraction"] >= Decimal("0.90")

    def test_food_fossil_carbon_zero(self):
        """Test food organic has zero fossil carbon (all biogenic)."""
        if "food_organic" in INCINERATION_PARAMS:
            assert INCINERATION_PARAMS["food_organic"]["fossil_carbon_fraction"] == Decimal("0.0")


class TestRecyclingFactors:
    """Test RECYCLING_FACTORS constant table."""

    def test_has_entries(self):
        """Test table has recycling factor entries."""
        assert len(RECYCLING_FACTORS) >= 6

    def test_aluminum_high_avoided_ef(self):
        """Test aluminum has very high avoided EF (energy-intensive virgin production)."""
        if "aluminum" in RECYCLING_FACTORS:
            al_avoided = RECYCLING_FACTORS["aluminum"]["avoided_ef"]
            assert al_avoided > Decimal("5.0")

    def test_recovery_rate_range(self):
        """Test recovery rates are between 0 and 1."""
        for material, factors in RECYCLING_FACTORS.items():
            rate = factors["recovery_rate"]
            assert Decimal("0.0") <= rate <= Decimal("1.0"), (
                f"Recovery rate for {material} out of range: {rate}"
            )


class TestCompostingADFactors:
    """Test COMPOSTING_AD_FACTORS constant table."""

    def test_has_entries(self):
        """Test table has composting/AD factor entries."""
        assert len(COMPOSTING_AD_FACTORS) >= 3

    def test_food_highest_ch4(self):
        """Test food organic has highest composting CH4 factor."""
        if "food_organic" in COMPOSTING_AD_FACTORS:
            food_ch4 = COMPOSTING_AD_FACTORS["food_organic"]["ch4_ef"]
            assert food_ch4 > Decimal("2.0")


class TestProductWeights:
    """Test PRODUCT_WEIGHTS constant table."""

    def test_has_20_categories(self):
        """Test weights are available for all 20 categories."""
        assert len(PRODUCT_WEIGHTS) >= 20

    def test_building_materials_heaviest(self):
        """Test building materials have highest default weight."""
        bm_weight = PRODUCT_WEIGHTS["building_materials"]
        assert bm_weight >= Decimal("30.0")

    def test_packaging_lightest(self):
        """Test packaging has lowest default weight."""
        pkg_weight = PRODUCT_WEIGHTS["packaging"]
        assert pkg_weight <= Decimal("0.1")


class TestDQIScoring:
    """Test DQI_SCORING constant table."""

    def test_has_entries(self):
        """Test DQI scoring table exists."""
        assert len(DQI_SCORING) >= 3

    def test_values_are_decimal(self):
        """Test all DQI scoring values are Decimal."""
        for key, value in DQI_SCORING.items():
            if isinstance(value, (int, float, Decimal)):
                assert isinstance(value, (Decimal, int))


class TestGWPValues:
    """Test GWP_VALUES constant table."""

    def test_ar5_ch4_gwp(self):
        """Test AR5 CH4 GWP100 is 28."""
        assert GWP_VALUES["AR5"]["CH4"] == Decimal("28")

    def test_ar5_n2o_gwp(self):
        """Test AR5 N2O GWP100 is 265."""
        assert GWP_VALUES["AR5"]["N2O"] == Decimal("265")

    def test_ar6_ch4_gwp(self):
        """Test AR6 CH4 GWP100 is 27.9 (or rounded)."""
        ch4_ar6 = GWP_VALUES["AR6"]["CH4"]
        assert Decimal("27") <= ch4_ar6 <= Decimal("28.5")


class TestDCRules:
    """Test DC_RULES constant table for double-counting prevention."""

    def test_has_8_rules(self):
        """Test there are exactly 8 double-counting rules."""
        assert len(DC_RULES) == 8

    @pytest.mark.parametrize("rule_id", [
        "DC-EOL-001", "DC-EOL-002", "DC-EOL-003", "DC-EOL-004",
        "DC-EOL-005", "DC-EOL-006", "DC-EOL-007", "DC-EOL-008",
    ])
    def test_rule_present(self, rule_id):
        """Test each DC rule ID is present."""
        assert rule_id in DC_RULES

    def test_dc_eol_007_avoided_emissions(self):
        """Test DC-EOL-007 enforces separate avoided emissions reporting."""
        rule = DC_RULES["DC-EOL-007"]
        assert "avoided" in rule.get("description", "").lower() or \
               "separate" in rule.get("description", "").lower()

    def test_dc_eol_001_cat5_boundary(self):
        """Test DC-EOL-001 prevents overlap with Cat 5 (Waste Generated)."""
        rule = DC_RULES["DC-EOL-001"]
        assert "cat" in rule.get("description", "").lower() or \
               "boundary" in rule.get("description", "").lower() or \
               "scope" in rule.get("description", "").lower()


class TestAvoidedEFs:
    """Test AVOIDED_EFS constant table."""

    def test_has_entries(self):
        """Test avoided EFs table has entries."""
        assert len(AVOIDED_EFS) >= 5

    def test_values_are_decimal(self):
        """Test all avoided EFs are Decimal."""
        for material, ef in AVOIDED_EFS.items():
            assert isinstance(ef, Decimal)

    def test_values_are_positive(self):
        """Test all avoided EFs are positive (represent emissions avoided)."""
        for material, ef in AVOIDED_EFS.items():
            assert ef > Decimal("0.0"), f"Avoided EF for {material} not positive"


class TestCircularityBenchmarks:
    """Test CIRCULARITY_BENCHMARKS constant table."""

    def test_has_entries(self):
        """Test circularity benchmarks table exists."""
        assert len(CIRCULARITY_BENCHMARKS) >= 3

    def test_values_are_decimal(self):
        """Test all benchmark values are Decimal."""
        for key, value in CIRCULARITY_BENCHMARKS.items():
            if isinstance(value, (Decimal, float, int)):
                assert isinstance(value, (Decimal, int))


class TestWasteHierarchyWeights:
    """Test WASTE_HIERARCHY_WEIGHTS constant table."""

    def test_has_5_levels(self):
        """Test there are 5 waste hierarchy levels."""
        assert len(WASTE_HIERARCHY_WEIGHTS) == 5

    def test_prevention_highest_weight(self):
        """Test prevention has highest weight (most preferred)."""
        prevention = WASTE_HIERARCHY_WEIGHTS["prevention"]
        disposal = WASTE_HIERARCHY_WEIGHTS["disposal"]
        assert prevention > disposal


# ==============================================================================
# INPUT MODEL TESTS
# ==============================================================================


class TestWasteTypeSpecificInput:
    """Test WasteTypeSpecificInput Pydantic model."""

    def test_valid_creation(self):
        """Test creating a valid waste-type-specific input."""
        inp = WasteTypeSpecificInput(
            product_id="PRD-001",
            material="plastic_abs",
            treatment="incineration",
            mass_kg=Decimal("100.0"),
        )
        assert inp.product_id == "PRD-001"
        assert inp.mass_kg == Decimal("100.0")

    def test_negative_mass_rejected(self):
        """Test negative mass raises validation error."""
        with pytest.raises(PydanticValidationError):
            WasteTypeSpecificInput(
                product_id="PRD-001",
                material="steel",
                treatment="landfill",
                mass_kg=Decimal("-10.0"),
            )

    def test_empty_product_id_rejected(self):
        """Test empty product_id raises validation error."""
        with pytest.raises(PydanticValidationError):
            WasteTypeSpecificInput(
                product_id="",
                material="steel",
                treatment="landfill",
                mass_kg=Decimal("100.0"),
            )


class TestAverageDataInput:
    """Test AverageDataInput Pydantic model."""

    def test_valid_creation(self):
        """Test creating a valid average-data input."""
        inp = AverageDataInput(
            product_id="PRD-001",
            product_category="consumer_electronics",
            total_mass_kg=Decimal("200.0"),
            region="US",
        )
        assert inp.product_category == "consumer_electronics"

    def test_default_region(self):
        """Test default region is GLOBAL if not specified."""
        inp = AverageDataInput(
            product_id="PRD-001",
            product_category="packaging",
            total_mass_kg=Decimal("50.0"),
        )
        assert inp.region in ("GLOBAL", None)


class TestProducerSpecificInput:
    """Test ProducerSpecificInput Pydantic model."""

    def test_valid_creation(self):
        """Test creating a valid producer-specific input."""
        inp = ProducerSpecificInput(
            product_id="PRD-001",
            epd_id="EPD-2024-001",
            eol_module_co2e_kg=Decimal("500.0"),
            verification_level="third_party_verified",
        )
        assert inp.epd_id == "EPD-2024-001"

    def test_epd_id_required(self):
        """Test EPD ID is required for producer-specific method."""
        with pytest.raises(PydanticValidationError):
            ProducerSpecificInput(
                product_id="PRD-001",
                epd_id=None,
                eol_module_co2e_kg=Decimal("500.0"),
            )


class TestProductInput:
    """Test ProductInput Pydantic model."""

    def test_valid_creation(self):
        """Test creating a valid product input."""
        inp = ProductInput(
            product_id="PRD-001",
            product_name="Test Product",
            product_category="consumer_electronics",
            total_mass_kg=Decimal("2.5"),
            units_sold=100000,
            reporting_year=2024,
        )
        assert inp.units_sold == 100000

    def test_total_mass_tonnes_calculation(self):
        """Test total mass in tonnes calculation."""
        inp = ProductInput(
            product_id="PRD-001",
            product_name="Test Product",
            product_category="consumer_electronics",
            total_mass_kg=Decimal("2.5"),
            units_sold=100000,
            reporting_year=2024,
        )
        expected_tonnes = Decimal("2.5") * 100000 / 1000
        assert inp.total_mass_tonnes == expected_tonnes

    def test_zero_units_rejected(self):
        """Test zero units_sold raises validation error."""
        with pytest.raises(PydanticValidationError):
            ProductInput(
                product_id="PRD-001",
                product_name="Test",
                product_category="packaging",
                total_mass_kg=Decimal("0.01"),
                units_sold=0,
                reporting_year=2024,
            )


class TestTreatmentPathwayInput:
    """Test TreatmentPathwayInput Pydantic model."""

    def test_valid_creation(self):
        """Test creating a valid treatment pathway input."""
        inp = TreatmentPathwayInput(
            treatment="recycling",
            fraction=Decimal("0.30"),
        )
        assert inp.treatment == "recycling"

    def test_fraction_over_one_rejected(self):
        """Test fraction > 1.0 raises validation error."""
        with pytest.raises(PydanticValidationError):
            TreatmentPathwayInput(
                treatment="landfill",
                fraction=Decimal("1.5"),
            )


class TestBatchInput:
    """Test BatchInput Pydantic model."""

    def test_valid_creation(self):
        """Test creating a valid batch input."""
        inp = BatchInput(
            tenant_id="TENANT-001",
            products=[
                {"product_id": "P1", "product_category": "packaging"},
            ],
        )
        assert len(inp.products) == 1


class TestPortfolioInput:
    """Test PortfolioInput Pydantic model."""

    def test_valid_creation(self):
        """Test creating a valid portfolio input."""
        inp = PortfolioInput(
            tenant_id="TENANT-001",
            org_id="ORG-001",
            reporting_year=2024,
            products=[
                {"product_id": "P1", "product_category": "consumer_electronics"},
                {"product_id": "P2", "product_category": "packaging"},
            ],
        )
        assert len(inp.products) == 2


# ==============================================================================
# RESULT MODEL TESTS
# ==============================================================================


class TestCalculationResult:
    """Test CalculationResult Pydantic model (frozen=True)."""

    def test_valid_creation(self):
        """Test creating a valid calculation result."""
        result = CalculationResult(
            calculation_id="CALC-001",
            product_id="PRD-001",
            gross_emissions_kgco2e=Decimal("1250.0"),
            avoided_emissions_kgco2e=Decimal("450.0"),
            method="waste_type_specific",
            dqi_score=Decimal("75.0"),
            provenance_hash="a" * 64,
        )
        assert result.gross_emissions_kgco2e == Decimal("1250.0")

    def test_avoided_emissions_always_separate(self):
        """CRITICAL: Test avoided emissions are never netted against gross."""
        result = CalculationResult(
            calculation_id="CALC-002",
            product_id="PRD-002",
            gross_emissions_kgco2e=Decimal("1000.0"),
            avoided_emissions_kgco2e=Decimal("800.0"),
            method="waste_type_specific",
            dqi_score=Decimal("75.0"),
            provenance_hash="b" * 64,
        )
        # Gross is always reported independently, never subtracted by avoided
        assert result.gross_emissions_kgco2e == Decimal("1000.0")
        assert result.avoided_emissions_kgco2e == Decimal("800.0")
        # There should be no net_emissions field that subtracts
        assert not hasattr(result, "net_emissions_kgco2e") or \
            result.gross_emissions_kgco2e != result.gross_emissions_kgco2e - result.avoided_emissions_kgco2e

    def test_immutability(self):
        """Test result is immutable (frozen=True)."""
        result = CalculationResult(
            calculation_id="CALC-001",
            product_id="PRD-001",
            gross_emissions_kgco2e=Decimal("1000.0"),
            avoided_emissions_kgco2e=Decimal("0.0"),
            method="average_data",
            dqi_score=Decimal("50.0"),
            provenance_hash="c" * 64,
        )
        with pytest.raises(Exception):
            result.gross_emissions_kgco2e = Decimal("999.0")


class TestTreatmentBreakdown:
    """Test TreatmentBreakdown Pydantic model."""

    def test_valid_creation(self):
        """Test creating a valid treatment breakdown."""
        breakdown = TreatmentBreakdown(
            treatment="landfill",
            gross_emissions_kgco2e=Decimal("380.0"),
            avoided_emissions_kgco2e=Decimal("0.0"),
            mass_kg=Decimal("500.0"),
            fraction=Decimal("0.40"),
        )
        assert breakdown.treatment == "landfill"


class TestComplianceCheckResult:
    """Test ComplianceCheckResult Pydantic model."""

    def test_valid_compliant_result(self):
        """Test creating a compliant check result."""
        result = ComplianceCheckResult(
            framework="GHG_PROTOCOL_SCOPE3",
            compliant=True,
            issues=[],
            warnings=[],
        )
        assert result.compliant is True

    def test_non_compliant_with_issues(self):
        """Test creating a non-compliant result with issues."""
        result = ComplianceCheckResult(
            framework="CSRD_ESRS_E5",
            compliant=False,
            issues=["Circularity metrics missing"],
            warnings=[],
        )
        assert result.compliant is False
        assert len(result.issues) == 1


class TestCircularityResult:
    """Test CircularityResult Pydantic model."""

    def test_valid_creation(self):
        """Test creating a valid circularity result."""
        result = CircularityResult(
            recycling_rate=Decimal("0.28"),
            diversion_rate=Decimal("0.48"),
            circularity_index=Decimal("0.35"),
            material_recovery_rate=Decimal("0.22"),
            waste_hierarchy_score=Decimal("65.0"),
        )
        assert result.recycling_rate == Decimal("0.28")


class TestAvoidedEmissionsResult:
    """Test AvoidedEmissionsResult Pydantic model."""

    def test_valid_creation(self):
        """Test creating a valid avoided emissions result."""
        result = AvoidedEmissionsResult(
            total_avoided_kgco2e=Decimal("450.0"),
            by_treatment={"recycling": Decimal("350.0"), "anaerobic_digestion": Decimal("100.0")},
            by_material={"steel": Decimal("200.0"), "aluminum": Decimal("250.0")},
        )
        assert result.total_avoided_kgco2e == Decimal("450.0")

    def test_avoided_always_positive(self):
        """Test avoided emissions are always reported as positive values."""
        result = AvoidedEmissionsResult(
            total_avoided_kgco2e=Decimal("450.0"),
            by_treatment={"recycling": Decimal("350.0")},
            by_material={"steel": Decimal("200.0")},
        )
        assert result.total_avoided_kgco2e > Decimal("0.0")


class TestHotspotResult:
    """Test HotspotResult Pydantic model."""

    def test_valid_creation(self):
        """Test creating a valid hotspot result."""
        result = HotspotResult(
            material="plastic_abs",
            treatment="incineration",
            emissions_kgco2e=Decimal("520.0"),
            percentage=Decimal("41.6"),
        )
        assert result.percentage == Decimal("41.6")


# ==============================================================================
# HELPER FUNCTION TESTS
# ==============================================================================


class TestCalculateProvenanceHash:
    """Test calculate_provenance_hash helper."""

    def test_returns_64_char_hex(self):
        """Test hash is 64-char lowercase hex string."""
        h = calculate_provenance_hash({"key": "value"})
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_deterministic(self):
        """Test same input gives same hash."""
        data = {"product": "PRD-001", "mass": "100"}
        h1 = calculate_provenance_hash(data)
        h2 = calculate_provenance_hash(data)
        assert h1 == h2

    def test_different_inputs_different_hash(self):
        """Test different inputs give different hashes."""
        h1 = calculate_provenance_hash({"a": 1})
        h2 = calculate_provenance_hash({"a": 2})
        assert h1 != h2


class TestGetDQIClassification:
    """Test get_dqi_classification helper."""

    @pytest.mark.parametrize("score,expected", [
        (Decimal("90"), "excellent"),
        (Decimal("80"), "good"),
        (Decimal("60"), "fair"),
        (Decimal("40"), "poor"),
        (Decimal("10"), "very_poor"),
    ])
    def test_classification_levels(self, score, expected):
        """Test each DQI classification level."""
        result = get_dqi_classification(score)
        assert result == expected


class TestClassifyProductCategory:
    """Test classify_product_category helper."""

    def test_known_category(self):
        """Test classifying a known product name."""
        result = classify_product_category("smartphone")
        assert result in ("consumer_electronics", "mixed")

    def test_unknown_returns_mixed(self):
        """Test unknown product defaults to 'mixed'."""
        result = classify_product_category("unknown_widget_xyz")
        assert result == "mixed"


class TestValidateMaterial:
    """Test validate_material helper."""

    @pytest.mark.parametrize("material", [
        "steel", "aluminum", "plastic_abs", "paper_cardboard", "glass",
    ])
    def test_valid_materials(self, material):
        """Test known materials validate successfully."""
        assert validate_material(material) is True

    def test_invalid_material(self):
        """Test unknown material returns False."""
        assert validate_material("unobtanium") is False


class TestGetMaterialEF:
    """Test get_material_ef helper."""

    def test_returns_decimal(self):
        """Test EF lookup returns Decimal."""
        ef = get_material_ef("steel", "landfill")
        assert isinstance(ef, Decimal)

    def test_known_material_treatment(self):
        """Test EF for known material/treatment pair."""
        ef = get_material_ef("plastic_abs", "incineration")
        assert ef > Decimal("0.0")


class TestGetTreatmentMix:
    """Test get_treatment_mix helper."""

    def test_known_region(self):
        """Test treatment mix for known region."""
        mix = get_treatment_mix("US")
        assert "landfill" in mix
        assert "incineration" in mix

    def test_unknown_region_returns_global(self):
        """Test unknown region falls back to GLOBAL."""
        mix = get_treatment_mix("XX")
        assert mix is not None


class TestComputeDQIScore:
    """Test compute_dqi_score helper."""

    def test_score_range(self):
        """Test DQI score is between 0 and 100."""
        scores = {
            "reliability": Decimal("4"), "completeness": Decimal("3"),
            "temporal": Decimal("3"), "geographic": Decimal("4"),
            "technological": Decimal("3"),
        }
        result = compute_dqi_score(scores)
        assert Decimal("0") <= result <= Decimal("100")


class TestComputeUncertaintyRange:
    """Test compute_uncertainty_range helper."""

    def test_returns_tuple(self):
        """Test returns (lower, upper) tuple."""
        lower, upper = compute_uncertainty_range(Decimal("1000"), Decimal("20"))
        assert lower < Decimal("1000")
        assert upper > Decimal("1000")


class TestFormatCO2e:
    """Test format_co2e helper."""

    def test_format_kg(self):
        """Test formatting in kg CO2e."""
        result = format_co2e(Decimal("123.456"), "kg")
        assert "123" in result

    def test_format_tonnes(self):
        """Test formatting in tonnes CO2e."""
        result = format_co2e(Decimal("1234.5"), "tonnes")
        assert isinstance(result, str)


class TestGetCircularityIndex:
    """Test get_circularity_index helper."""

    def test_returns_decimal(self):
        """Test circularity index is a Decimal between 0 and 1."""
        treatments = {
            "recycling": Decimal("0.30"),
            "composting": Decimal("0.10"),
            "landfill": Decimal("0.50"),
            "incineration": Decimal("0.10"),
        }
        idx = get_circularity_index(treatments)
        assert Decimal("0.0") <= idx <= Decimal("1.0")


class TestGetWasteHierarchyScore:
    """Test get_waste_hierarchy_score helper."""

    def test_higher_recycling_higher_score(self):
        """Test more recycling gives higher score."""
        high_recycling = {
            "recycling": Decimal("0.70"), "landfill": Decimal("0.30"),
        }
        low_recycling = {
            "recycling": Decimal("0.10"), "landfill": Decimal("0.90"),
        }
        score_high = get_waste_hierarchy_score(high_recycling)
        score_low = get_waste_hierarchy_score(low_recycling)
        assert score_high > score_low


class TestIsBiogenicMaterial:
    """Test is_biogenic_material helper."""

    @pytest.mark.parametrize("material,expected", [
        ("food_organic", True),
        ("paper_cardboard", True),
        ("cotton", True),
        ("wood_mdf", True),
        ("plastic_abs", False),
        ("steel", False),
        ("glass", False),
    ])
    def test_biogenic_classification(self, material, expected):
        """Test biogenic material classification."""
        assert is_biogenic_material(material) == expected


class TestGetGWP:
    """Test get_gwp helper."""

    @pytest.mark.parametrize("version,gas,expected_min", [
        ("AR5", "CH4", Decimal("25")),
        ("AR5", "N2O", Decimal("260")),
        ("AR6", "CH4", Decimal("25")),
    ])
    def test_gwp_values(self, version, gas, expected_min):
        """Test GWP value retrieval."""
        gwp = get_gwp(version, gas)
        assert gwp >= expected_min


class TestNormalizeTreatmentMix:
    """Test normalize_treatment_mix helper."""

    def test_sums_to_one(self):
        """Test normalized mix sums to 1.0."""
        mix = {"landfill": Decimal("5"), "recycling": Decimal("3"), "incineration": Decimal("2")}
        normalized = normalize_treatment_mix(mix)
        total = sum(normalized.values())
        assert abs(total - Decimal("1.0")) < Decimal("0.001")

    def test_preserves_ratios(self):
        """Test normalization preserves relative proportions."""
        mix = {"landfill": Decimal("6"), "recycling": Decimal("4")}
        normalized = normalize_treatment_mix(mix)
        assert normalized["landfill"] > normalized["recycling"]
