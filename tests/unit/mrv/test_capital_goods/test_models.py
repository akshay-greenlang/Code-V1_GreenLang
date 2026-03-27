# -*- coding: utf-8 -*-
"""Unit tests for Capital Goods Agent models and constants."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from greenlang.agents.mrv.capital_goods.models import (
    # Enums
    AssetCategory,
    AssetSubCategory,
    CalculationMethod,
    ConstructionMaterial,
    DataQualityIndicator,
    EmissionIntensityUnit,
    ManufacturingProcessType,
    RegulatoryFramework,
    SectorClassification,
    SupplierDataQuality,
    SupplierDataType,
    # Models
    CapitalAssetRecord,
    CapExSpendRecord,
    PhysicalRecord,
    SupplierRecord,
    CalculationRequest,
    BatchRequest,
    SpendBasedResult,
    AverageDataResult,
    SupplierSpecificResult,
    HybridResult,
    # Constants
    GWP_VALUES,
    CAPITAL_EEIO_EMISSION_FACTORS,
    ASSET_USEFUL_LIFE_RANGES,
    DQI_SCORING,
    FRAMEWORK_REQUIRED_DISCLOSURES,
)


# ============================================================================
# ENUM TESTS
# ============================================================================


class TestEnums:
    """Test all enum definitions."""

    def test_asset_category_members(self):
        """Test AssetCategory enum has expected members."""
        assert AssetCategory.BUILDINGS.value == "buildings"
        assert AssetCategory.MACHINERY.value == "machinery"
        assert AssetCategory.VEHICLES.value == "vehicles"
        assert AssetCategory.IT_EQUIPMENT.value == "it_equipment"
        assert AssetCategory.OFFICE_EQUIPMENT.value == "office_equipment"
        assert AssetCategory.FURNITURE.value == "furniture"
        assert AssetCategory.OTHER.value == "other"
        assert len(AssetCategory) == 7

    def test_asset_subcategory_members(self):
        """Test AssetSubCategory enum has expected members."""
        assert AssetSubCategory.COMMERCIAL_BUILDING.value == "commercial_building"
        assert AssetSubCategory.INDUSTRIAL_FACILITY.value == "industrial_facility"
        assert AssetSubCategory.PRODUCTION_EQUIPMENT.value == "production_equipment"
        assert AssetSubCategory.COMMERCIAL_VEHICLE.value == "commercial_vehicle"
        assert AssetSubCategory.SERVERS.value == "servers"
        assert len(AssetSubCategory) >= 5

    def test_calculation_method_members(self):
        """Test CalculationMethod enum has expected members."""
        assert CalculationMethod.SPEND_BASED.value == "spend_based"
        assert CalculationMethod.AVERAGE_DATA.value == "average_data"
        assert CalculationMethod.SUPPLIER_SPECIFIC.value == "supplier_specific"
        assert CalculationMethod.HYBRID.value == "hybrid"
        assert len(CalculationMethod) == 4

    def test_construction_material_members(self):
        """Test ConstructionMaterial enum has expected members."""
        assert ConstructionMaterial.STEEL.value == "steel"
        assert ConstructionMaterial.CONCRETE.value == "concrete"
        assert ConstructionMaterial.ALUMINUM.value == "aluminum"
        assert ConstructionMaterial.GLASS.value == "glass"
        assert ConstructionMaterial.WOOD.value == "wood"
        assert ConstructionMaterial.PLASTICS.value == "plastics"
        assert ConstructionMaterial.ELECTRONICS.value == "electronics"
        assert len(ConstructionMaterial) >= 7

    def test_data_quality_indicator_members(self):
        """Test DataQualityIndicator enum has expected members."""
        assert DataQualityIndicator.TEMPORAL_CORRELATION.value == "temporal_correlation"
        assert DataQualityIndicator.GEOGRAPHICAL_CORRELATION.value == "geographical_correlation"
        assert DataQualityIndicator.TECHNOLOGICAL_CORRELATION.value == "technological_correlation"
        assert DataQualityIndicator.COMPLETENESS.value == "completeness"
        assert DataQualityIndicator.RELIABILITY.value == "reliability"
        assert len(DataQualityIndicator) == 5

    def test_emission_intensity_unit_members(self):
        """Test EmissionIntensityUnit enum has expected members."""
        assert EmissionIntensityUnit.KG_CO2E_PER_USD.value == "kgCO2e/USD"
        assert EmissionIntensityUnit.TCO2E_PER_USD.value == "tCO2e/USD"
        assert EmissionIntensityUnit.KG_CO2E_PER_KG.value == "kgCO2e/kg"
        assert EmissionIntensityUnit.TCO2E_PER_UNIT.value == "tCO2e/unit"
        assert len(EmissionIntensityUnit) >= 4

    def test_manufacturing_process_type_members(self):
        """Test ManufacturingProcessType enum has expected members."""
        assert ManufacturingProcessType.PRIMARY_STEEL_BOF.value == "primary_steel_bof"
        assert ManufacturingProcessType.PRIMARY_STEEL_EAF.value == "primary_steel_eaf"
        assert ManufacturingProcessType.RECYCLED_STEEL.value == "recycled_steel"
        assert ManufacturingProcessType.PRIMARY_ALUMINUM.value == "primary_aluminum"
        assert ManufacturingProcessType.RECYCLED_ALUMINUM.value == "recycled_aluminum"
        assert len(ManufacturingProcessType) >= 5

    def test_regulatory_framework_members(self):
        """Test RegulatoryFramework enum has expected members."""
        assert RegulatoryFramework.GHG_PROTOCOL_SCOPE3.value == "ghg_protocol_scope3"
        assert RegulatoryFramework.ISO_14064_1.value == "iso_14064_1"
        assert RegulatoryFramework.CSRD_ESRS_E1.value == "csrd_esrs_e1"
        assert RegulatoryFramework.CDP_CLIMATE.value == "cdp_climate"
        assert RegulatoryFramework.TCFD.value == "tcfd"
        assert RegulatoryFramework.SBTI_NET_ZERO.value == "sbti_net_zero"
        assert len(RegulatoryFramework) == 6

    def test_sector_classification_members(self):
        """Test SectorClassification enum has expected members."""
        assert SectorClassification.NAICS.value == "naics"
        assert SectorClassification.ISIC.value == "isic"
        assert SectorClassification.NACE.value == "nace"
        assert SectorClassification.UNSPSC.value == "unspsc"
        assert len(SectorClassification) == 4

    def test_supplier_data_quality_members(self):
        """Test SupplierDataQuality enum has expected members."""
        assert SupplierDataQuality.VERY_HIGH.value == "very_high"
        assert SupplierDataQuality.HIGH.value == "high"
        assert SupplierDataQuality.MEDIUM.value == "medium"
        assert SupplierDataQuality.LOW.value == "low"
        assert SupplierDataQuality.VERY_LOW.value == "very_low"
        assert len(SupplierDataQuality) == 5

    def test_supplier_data_type_members(self):
        """Test SupplierDataType enum has expected members."""
        assert SupplierDataType.EPD.value == "epd"
        assert SupplierDataType.PRODUCT_CARBON_FOOTPRINT.value == "product_carbon_footprint"
        assert SupplierDataType.CDP_DISCLOSURE.value == "cdp_disclosure"
        assert SupplierDataType.LCA_REPORT.value == "lca_report"
        assert SupplierDataType.SUPPLIER_QUESTIONNAIRE.value == "supplier_questionnaire"
        assert len(SupplierDataType) >= 5


# ============================================================================
# CAPITAL ASSET RECORD TESTS
# ============================================================================


class TestCapitalAssetRecord:
    """Test CapitalAssetRecord model."""

    def test_create_valid_asset_record(self, sample_asset_building):
        """Test creating a valid asset record."""
        assert sample_asset_building.asset_id == "BLDG-2026-001"
        assert sample_asset_building.asset_name == "Corporate Headquarters Building"
        assert sample_asset_building.asset_category == AssetCategory.BUILDINGS
        assert sample_asset_building.acquisition_cost_usd == Decimal("12500000.00")
        assert sample_asset_building.useful_life_years == 40

    def test_asset_record_is_frozen(self, sample_asset_building):
        """Test that asset record is immutable (frozen)."""
        with pytest.raises(ValidationError):
            sample_asset_building.asset_name = "Modified Name"

    def test_asset_record_negative_cost_fails(self):
        """Test that negative acquisition cost fails validation."""
        with pytest.raises(ValidationError):
            CapitalAssetRecord(
                asset_id="INVALID-001",
                asset_name="Invalid Asset",
                asset_category=AssetCategory.BUILDINGS,
                asset_subcategory=AssetSubCategory.COMMERCIAL_BUILDING,
                acquisition_date=date(2026, 1, 1),
                acquisition_cost_usd=Decimal("-1000.00"),
                useful_life_years=40,
                depreciation_method="straight-line",
                currency_code="USD",
                reporting_year=2026,
            )

    def test_asset_record_zero_useful_life_fails(self):
        """Test that zero useful life fails validation."""
        with pytest.raises(ValidationError):
            CapitalAssetRecord(
                asset_id="INVALID-002",
                asset_name="Invalid Asset",
                asset_category=AssetCategory.BUILDINGS,
                asset_subcategory=AssetSubCategory.COMMERCIAL_BUILDING,
                acquisition_date=date(2026, 1, 1),
                acquisition_cost_usd=Decimal("1000.00"),
                useful_life_years=0,
                depreciation_method="straight-line",
                currency_code="USD",
                reporting_year=2026,
            )

    def test_asset_record_invalid_currency_fails(self):
        """Test that invalid currency code fails validation."""
        with pytest.raises(ValidationError):
            CapitalAssetRecord(
                asset_id="INVALID-003",
                asset_name="Invalid Asset",
                asset_category=AssetCategory.BUILDINGS,
                asset_subcategory=AssetSubCategory.COMMERCIAL_BUILDING,
                acquisition_date=date(2026, 1, 1),
                acquisition_cost_usd=Decimal("1000.00"),
                useful_life_years=40,
                depreciation_method="straight-line",
                currency_code="INVALID",
                reporting_year=2026,
            )

    def test_asset_record_serialization(self, sample_asset_building):
        """Test asset record serialization to dict."""
        data = sample_asset_building.model_dump()
        assert data["asset_id"] == "BLDG-2026-001"
        assert data["asset_category"] == "buildings"
        assert float(data["acquisition_cost_usd"]) == 12500000.00

    def test_asset_record_json_roundtrip(self, sample_asset_building):
        """Test asset record JSON serialization round-trip."""
        json_str = sample_asset_building.model_dump_json()
        restored = CapitalAssetRecord.model_validate_json(json_str)
        assert restored.asset_id == sample_asset_building.asset_id
        assert restored.acquisition_cost_usd == sample_asset_building.acquisition_cost_usd


# ============================================================================
# CAPEX SPEND RECORD TESTS
# ============================================================================


class TestCapExSpendRecord:
    """Test CapExSpendRecord model."""

    def test_create_valid_capex_record(self, sample_capex_construction):
        """Test creating a valid CapEx spend record."""
        assert sample_capex_construction.spend_id == "CAPEX-2026-001"
        assert sample_capex_construction.spend_amount_usd == Decimal("12500000.00")
        assert sample_capex_construction.sector_code == "236220"
        assert sample_capex_construction.sector_classification == SectorClassification.NAICS

    def test_capex_record_is_frozen(self, sample_capex_construction):
        """Test that CapEx record is immutable."""
        with pytest.raises(ValidationError):
            sample_capex_construction.vendor_name = "Modified Vendor"

    def test_capex_record_negative_spend_fails(self):
        """Test that negative spend amount fails validation."""
        with pytest.raises(ValidationError):
            CapExSpendRecord(
                spend_id="INVALID-001",
                spend_description="Invalid spend",
                spend_amount_usd=Decimal("-1000.00"),
                spend_date=date(2026, 1, 1),
                vendor_name="Test Vendor",
                vendor_country="USA",
                sector_classification=SectorClassification.NAICS,
                sector_code="123456",
                currency_code="USD",
                reporting_year=2026,
            )

    def test_capex_record_valid_naics_code(self):
        """Test that valid NAICS code is accepted."""
        record = CapExSpendRecord(
            spend_id="VALID-001",
            spend_description="Valid spend",
            spend_amount_usd=Decimal("1000.00"),
            spend_date=date(2026, 1, 1),
            vendor_name="Test Vendor",
            vendor_country="USA",
            sector_classification=SectorClassification.NAICS,
            sector_code="236220",
            currency_code="USD",
            reporting_year=2026,
        )
        assert record.sector_code == "236220"


# ============================================================================
# PHYSICAL RECORD TESTS
# ============================================================================


class TestPhysicalRecord:
    """Test PhysicalRecord model."""

    def test_create_valid_physical_record(self, sample_physical_steel):
        """Test creating a valid physical record."""
        assert sample_physical_steel.material_id == "MAT-STEEL-001"
        assert sample_physical_steel.material_type == ConstructionMaterial.STEEL
        assert sample_physical_steel.quantity == Decimal("125.5")
        assert sample_physical_steel.unit == "metric_ton"

    def test_physical_record_is_frozen(self, sample_physical_steel):
        """Test that physical record is immutable."""
        with pytest.raises(ValidationError):
            sample_physical_steel.quantity = Decimal("200.0")

    def test_physical_record_negative_quantity_fails(self):
        """Test that negative quantity fails validation."""
        with pytest.raises(ValidationError):
            PhysicalRecord(
                material_id="INVALID-001",
                material_name="Invalid Material",
                material_type=ConstructionMaterial.STEEL,
                quantity=Decimal("-10.0"),
                unit="metric_ton",
                unit_cost_usd=Decimal("100.00"),
                total_cost_usd=Decimal("1000.00"),
                acquisition_date=date(2026, 1, 1),
                supplier_name="Test Supplier",
                supplier_country="USA",
                currency_code="USD",
                reporting_year=2026,
            )

    def test_physical_record_manufacturing_process(self):
        """Test physical record with manufacturing process type."""
        record = PhysicalRecord(
            material_id="MAT-STEEL-002",
            material_name="Recycled Steel",
            material_type=ConstructionMaterial.STEEL,
            quantity=Decimal("50.0"),
            unit="metric_ton",
            unit_cost_usd=Decimal("700.00"),
            total_cost_usd=Decimal("35000.00"),
            acquisition_date=date(2026, 1, 1),
            supplier_name="Test Supplier",
            supplier_country="USA",
            manufacturing_process=ManufacturingProcessType.RECYCLED_STEEL,
            currency_code="USD",
            reporting_year=2026,
        )
        assert record.manufacturing_process == ManufacturingProcessType.RECYCLED_STEEL


# ============================================================================
# SUPPLIER RECORD TESTS
# ============================================================================


class TestSupplierRecord:
    """Test SupplierRecord model."""

    def test_create_valid_supplier_epd(self, sample_supplier_epd):
        """Test creating a valid supplier record with EPD."""
        assert sample_supplier_epd.supplier_id == "SUP-001"
        assert sample_supplier_epd.data_type == SupplierDataType.EPD
        assert sample_supplier_epd.data_quality == SupplierDataQuality.HIGH
        assert sample_supplier_epd.emission_value == Decimal("1.85")
        assert sample_supplier_epd.emission_unit == EmissionIntensityUnit.KG_CO2E_PER_KG

    def test_supplier_record_is_frozen(self, sample_supplier_epd):
        """Test that supplier record is immutable."""
        with pytest.raises(ValidationError):
            sample_supplier_epd.data_quality = SupplierDataQuality.MEDIUM

    def test_supplier_record_negative_emission_fails(self):
        """Test that negative emission value fails validation."""
        with pytest.raises(ValidationError):
            SupplierRecord(
                supplier_id="INVALID-001",
                supplier_name="Invalid Supplier",
                product_name="Invalid Product",
                product_category=AssetCategory.BUILDINGS,
                data_type=SupplierDataType.EPD,
                emission_value=Decimal("-1.0"),
                emission_unit=EmissionIntensityUnit.KG_CO2E_PER_KG,
                data_quality=SupplierDataQuality.HIGH,
                data_year=2025,
                reporting_year=2026,
            )

    def test_supplier_record_pcf_type(self, sample_supplier_pcf):
        """Test supplier record with PCF data type."""
        assert sample_supplier_pcf.data_type == SupplierDataType.PRODUCT_CARBON_FOOTPRINT
        assert sample_supplier_pcf.verification_status == "self-reported"


# ============================================================================
# CALCULATION REQUEST TESTS
# ============================================================================


class TestCalculationRequest:
    """Test CalculationRequest model."""

    def test_create_spend_based_request(self, sample_calculation_request_spend):
        """Test creating spend-based calculation request."""
        assert sample_calculation_request_spend.request_id == "REQ-2026-001"
        assert sample_calculation_request_spend.calculation_method == CalculationMethod.SPEND_BASED
        assert sample_calculation_request_spend.capex_spend_record is not None
        assert sample_calculation_request_spend.gwp_version == "AR5"

    def test_create_average_data_request(self, sample_calculation_request_average):
        """Test creating average-data calculation request."""
        assert sample_calculation_request_average.calculation_method == CalculationMethod.AVERAGE_DATA
        assert sample_calculation_request_average.asset_record is not None

    def test_create_supplier_specific_request(self, sample_calculation_request_supplier):
        """Test creating supplier-specific calculation request."""
        assert sample_calculation_request_supplier.calculation_method == CalculationMethod.SUPPLIER_SPECIFIC
        assert sample_calculation_request_supplier.asset_record is not None
        assert sample_calculation_request_supplier.supplier_record is not None

    def test_create_hybrid_request(self, sample_calculation_request_hybrid):
        """Test creating hybrid calculation request."""
        assert sample_calculation_request_hybrid.calculation_method == CalculationMethod.HYBRID
        assert sample_calculation_request_hybrid.asset_record is not None
        assert sample_calculation_request_hybrid.capex_spend_record is not None
        assert sample_calculation_request_hybrid.physical_records is not None
        assert len(sample_calculation_request_hybrid.physical_records) > 0

    def test_calculation_request_invalid_gwp_version_fails(self, sample_capex_construction):
        """Test that invalid GWP version fails validation."""
        with pytest.raises(ValidationError):
            CalculationRequest(
                request_id="INVALID-001",
                calculation_method=CalculationMethod.SPEND_BASED,
                capex_spend_record=sample_capex_construction,
                gwp_version="INVALID",
                reporting_year=2026,
            )


# ============================================================================
# BATCH REQUEST TESTS
# ============================================================================


class TestBatchRequest:
    """Test BatchRequest model."""

    def test_create_valid_batch_request(self, sample_batch_request):
        """Test creating a valid batch request."""
        assert sample_batch_request.batch_id == "BATCH-2026-001"
        assert len(sample_batch_request.requests) == 2
        assert sample_batch_request.parallel_processing is True
        assert sample_batch_request.max_workers == 4

    def test_batch_request_empty_requests_fails(self):
        """Test that empty requests list fails validation."""
        with pytest.raises(ValidationError):
            BatchRequest(
                batch_id="INVALID-001",
                requests=[],
                reporting_year=2026,
            )


# ============================================================================
# RESULT MODEL TESTS
# ============================================================================


class TestSpendBasedResult:
    """Test SpendBasedResult model."""

    def test_create_valid_spend_based_result(self, sample_spend_based_result):
        """Test creating a valid spend-based result."""
        assert sample_spend_based_result.request_id == "REQ-2026-001"
        assert sample_spend_based_result.calculation_method == CalculationMethod.SPEND_BASED
        assert sample_spend_based_result.total_emissions_tco2e == Decimal("850.25")
        assert sample_spend_based_result.scope3_category == "3.2"
        assert sample_spend_based_result.emission_factor == Decimal("0.068")

    def test_spend_based_result_serialization(self, sample_spend_based_result):
        """Test spend-based result serialization."""
        data = sample_spend_based_result.model_dump()
        assert data["request_id"] == "REQ-2026-001"
        assert data["calculation_method"] == "spend_based"


class TestAverageDataResult:
    """Test AverageDataResult model."""

    def test_create_valid_average_data_result(self, sample_average_data_result):
        """Test creating a valid average-data result."""
        assert sample_average_data_result.request_id == "REQ-2026-042"
        assert sample_average_data_result.calculation_method == CalculationMethod.AVERAGE_DATA
        assert sample_average_data_result.total_emissions_tco2e == Decimal("24.5")
        assert sample_average_data_result.asset_category == AssetCategory.MACHINERY

    def test_average_data_result_serialization(self, sample_average_data_result):
        """Test average-data result serialization."""
        data = sample_average_data_result.model_dump()
        assert data["asset_category"] == "machinery"


class TestSupplierSpecificResult:
    """Test SupplierSpecificResult model."""

    def test_create_valid_supplier_specific_result(self, sample_supplier_specific_result):
        """Test creating a valid supplier-specific result."""
        assert sample_supplier_specific_result.request_id == "REQ-2026-156"
        assert sample_supplier_specific_result.calculation_method == CalculationMethod.SUPPLIER_SPECIFIC
        assert sample_supplier_specific_result.supplier_name == "Ford Motor Company"
        assert sample_supplier_specific_result.supplier_data_quality == SupplierDataQuality.HIGH

    def test_supplier_specific_result_serialization(self, sample_supplier_specific_result):
        """Test supplier-specific result serialization."""
        data = sample_supplier_specific_result.model_dump()
        assert data["supplier_data_type"] == "cdp_disclosure"


class TestHybridResult:
    """Test HybridResult model."""

    def test_create_valid_hybrid_result(self, sample_hybrid_result):
        """Test creating a valid hybrid result."""
        assert sample_hybrid_result.request_id == "REQ-2026-HYBRID-001"
        assert sample_hybrid_result.calculation_method == CalculationMethod.HYBRID
        assert sample_hybrid_result.physical_portion_tco2e == Decimal("232.18")
        assert sample_hybrid_result.spend_portion_tco2e == Decimal("643.32")
        assert sample_hybrid_result.physical_coverage_percentage == Decimal("26.5")

    def test_hybrid_result_total_equals_sum(self, sample_hybrid_result):
        """Test that total emissions equals sum of portions."""
        expected_total = sample_hybrid_result.physical_portion_tco2e + sample_hybrid_result.spend_portion_tco2e
        assert sample_hybrid_result.total_emissions_tco2e == expected_total


# ============================================================================
# CONSTANTS TESTS
# ============================================================================


class TestConstants:
    """Test constant dictionaries and tables."""

    def test_gwp_values_structure(self):
        """Test GWP_VALUES constant structure."""
        assert "AR5" in GWP_VALUES
        assert "AR6" in GWP_VALUES
        assert "CO2" in GWP_VALUES["AR5"]
        assert "CH4" in GWP_VALUES["AR5"]
        assert "N2O" in GWP_VALUES["AR5"]
        assert GWP_VALUES["AR5"]["CO2"] == Decimal("1")
        assert GWP_VALUES["AR5"]["CH4"] == Decimal("28")
        assert GWP_VALUES["AR5"]["N2O"] == Decimal("265")

    def test_capital_eeio_emission_factors_structure(self):
        """Test CAPITAL_EEIO_EMISSION_FACTORS constant structure."""
        assert SectorClassification.NAICS in CAPITAL_EEIO_EMISSION_FACTORS
        naics_factors = CAPITAL_EEIO_EMISSION_FACTORS[SectorClassification.NAICS]
        assert "236220" in naics_factors  # Commercial building construction
        assert "factor" in naics_factors["236220"]
        assert "unit" in naics_factors["236220"]
        assert "description" in naics_factors["236220"]
        assert isinstance(naics_factors["236220"]["factor"], Decimal)

    def test_asset_useful_life_ranges_structure(self):
        """Test ASSET_USEFUL_LIFE_RANGES constant structure."""
        assert AssetCategory.BUILDINGS in ASSET_USEFUL_LIFE_RANGES
        building_range = ASSET_USEFUL_LIFE_RANGES[AssetCategory.BUILDINGS]
        assert "min_years" in building_range
        assert "max_years" in building_range
        assert "typical_years" in building_range
        assert building_range["min_years"] <= building_range["typical_years"] <= building_range["max_years"]

    def test_dqi_scoring_structure(self):
        """Test DQI_SCORING constant structure."""
        assert DataQualityIndicator.TEMPORAL_CORRELATION in DQI_SCORING
        temporal_scores = DQI_SCORING[DataQualityIndicator.TEMPORAL_CORRELATION]
        assert 1 in temporal_scores
        assert 2 in temporal_scores
        assert 3 in temporal_scores
        assert 4 in temporal_scores
        assert 5 in temporal_scores
        assert "description" in temporal_scores[1]
        assert "criteria" in temporal_scores[1]

    def test_dqi_scoring_values(self):
        """Test DQI_SCORING has valid score values."""
        for indicator in DataQualityIndicator:
            assert indicator in DQI_SCORING
            scores = DQI_SCORING[indicator]
            assert all(score in [1, 2, 3, 4, 5] for score in scores.keys())

    def test_framework_required_disclosures_structure(self):
        """Test FRAMEWORK_REQUIRED_DISCLOSURES constant structure."""
        assert RegulatoryFramework.GHG_PROTOCOL_SCOPE3 in FRAMEWORK_REQUIRED_DISCLOSURES
        ghg_disclosures = FRAMEWORK_REQUIRED_DISCLOSURES[RegulatoryFramework.GHG_PROTOCOL_SCOPE3]
        assert "required_fields" in ghg_disclosures
        assert "optional_fields" in ghg_disclosures
        assert "calculation_methods" in ghg_disclosures
        assert isinstance(ghg_disclosures["required_fields"], list)
        assert isinstance(ghg_disclosures["calculation_methods"], list)

    def test_framework_required_disclosures_all_frameworks(self):
        """Test all regulatory frameworks have disclosure requirements."""
        for framework in RegulatoryFramework:
            assert framework in FRAMEWORK_REQUIRED_DISCLOSURES
            disclosure = FRAMEWORK_REQUIRED_DISCLOSURES[framework]
            assert len(disclosure["required_fields"]) > 0


# ============================================================================
# EDGE CASE TESTS
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_large_acquisition_cost(self):
        """Test handling very large acquisition costs."""
        record = CapitalAssetRecord(
            asset_id="LARGE-001",
            asset_name="Mega Facility",
            asset_category=AssetCategory.BUILDINGS,
            asset_subcategory=AssetSubCategory.INDUSTRIAL_FACILITY,
            acquisition_date=date(2026, 1, 1),
            acquisition_cost_usd=Decimal("999999999999.99"),
            useful_life_years=50,
            depreciation_method="straight-line",
            currency_code="USD",
            reporting_year=2026,
        )
        assert record.acquisition_cost_usd == Decimal("999999999999.99")

    def test_very_small_emission_value(self):
        """Test handling very small emission values."""
        result = SpendBasedResult(
            request_id="SMALL-001",
            calculation_method=CalculationMethod.SPEND_BASED,
            total_emissions_tco2e=Decimal("0.0001"),
            scope3_category="3.2",
            spend_amount_usd=Decimal("100.00"),
            emission_factor=Decimal("0.000001"),
            emission_factor_unit="tCO2e/USD",
            sector_code="999999",
            sector_classification=SectorClassification.NAICS,
            data_quality_score=Decimal("3.0"),
            gwp_version="AR5",
        )
        assert result.total_emissions_tco2e == Decimal("0.0001")

    def test_maximum_useful_life(self):
        """Test asset with maximum useful life."""
        record = CapitalAssetRecord(
            asset_id="MAX-LIFE-001",
            asset_name="Long-Lived Asset",
            asset_category=AssetCategory.BUILDINGS,
            asset_subcategory=AssetSubCategory.COMMERCIAL_BUILDING,
            acquisition_date=date(2026, 1, 1),
            acquisition_cost_usd=Decimal("1000000.00"),
            useful_life_years=100,
            depreciation_method="straight-line",
            currency_code="USD",
            reporting_year=2026,
        )
        assert record.useful_life_years == 100

    def test_minimum_useful_life(self):
        """Test asset with minimum useful life."""
        record = CapitalAssetRecord(
            asset_id="MIN-LIFE-001",
            asset_name="Short-Lived Asset",
            asset_category=AssetCategory.IT_EQUIPMENT,
            asset_subcategory=AssetSubCategory.SERVERS,
            acquisition_date=date(2026, 1, 1),
            acquisition_cost_usd=Decimal("10000.00"),
            useful_life_years=1,
            depreciation_method="straight-line",
            currency_code="USD",
            reporting_year=2026,
        )
        assert record.useful_life_years == 1
