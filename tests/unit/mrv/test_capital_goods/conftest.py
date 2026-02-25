# -*- coding: utf-8 -*-
"""Shared pytest fixtures for Capital Goods Agent tests."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine

from greenlang.capital_goods.models import (
    AssetCategory,
    AssetSubCategory,
    BatchRequest,
    CalculationMethod,
    CalculationRequest,
    CapExSpendRecord,
    CapitalAssetRecord,
    ConstructionMaterial,
    EmissionIntensityUnit,
    ManufacturingProcessType,
    PhysicalRecord,
    SectorClassification,
    SupplierDataQuality,
    SupplierRecord,
    SupplierDataType,
    AverageDataResult,
    SpendBasedResult,
    SupplierSpecificResult,
    HybridResult,
)


# ============================================================================
# ASSET RECORD FIXTURES
# ============================================================================


@pytest.fixture
def sample_asset_building() -> CapitalAssetRecord:
    """Sample capital asset record for a commercial building."""
    return CapitalAssetRecord(
        asset_id="BLDG-2026-001",
        asset_name="Corporate Headquarters Building",
        asset_category=AssetCategory.BUILDINGS,
        asset_subcategory=AssetSubCategory.COMMERCIAL_BUILDING,
        acquisition_date=date(2026, 1, 15),
        acquisition_cost_usd=Decimal("12500000.00"),
        useful_life_years=40,
        depreciation_method="straight-line",
        currency_code="USD",
        reporting_year=2026,
        reporting_period_start=date(2026, 1, 1),
        reporting_period_end=date(2026, 12, 31),
        operational_unit_id="OU-HQ-001",
        facility_id="FAC-USA-NYC-001",
        country_code="USA",
        region="North America",
        gl_account="1500-Buildings",
        cost_center="CC-FACILITIES-001",
    )


@pytest.fixture
def sample_asset_machinery() -> CapitalAssetRecord:
    """Sample capital asset record for industrial machinery."""
    return CapitalAssetRecord(
        asset_id="MACH-2026-042",
        asset_name="CNC Milling Machine Model X3000",
        asset_category=AssetCategory.MACHINERY,
        asset_subcategory=AssetSubCategory.PRODUCTION_EQUIPMENT,
        acquisition_date=date(2026, 3, 10),
        acquisition_cost_usd=Decimal("450000.00"),
        useful_life_years=15,
        depreciation_method="double-declining-balance",
        currency_code="USD",
        reporting_year=2026,
        reporting_period_start=date(2026, 1, 1),
        reporting_period_end=date(2026, 12, 31),
        operational_unit_id="OU-MFG-001",
        facility_id="FAC-USA-DET-003",
        country_code="USA",
        region="North America",
        gl_account="1520-Machinery",
        cost_center="CC-MANUFACTURING-001",
        manufacturer="Haas Automation",
        model_number="X3000-CNC",
        serial_number="X3000-2026-001234",
    )


@pytest.fixture
def sample_asset_vehicle() -> CapitalAssetRecord:
    """Sample capital asset record for a fleet vehicle."""
    return CapitalAssetRecord(
        asset_id="VEH-2026-156",
        asset_name="Delivery Truck - Ford F-150",
        asset_category=AssetCategory.VEHICLES,
        asset_subcategory=AssetSubCategory.COMMERCIAL_VEHICLE,
        acquisition_date=date(2026, 2, 20),
        acquisition_cost_usd=Decimal("55000.00"),
        useful_life_years=8,
        depreciation_method="straight-line",
        currency_code="USD",
        reporting_year=2026,
        reporting_period_start=date(2026, 1, 1),
        reporting_period_end=date(2026, 12, 31),
        operational_unit_id="OU-LOGISTICS-001",
        facility_id="FAC-USA-CHI-002",
        country_code="USA",
        region="North America",
        gl_account="1540-Vehicles",
        cost_center="CC-LOGISTICS-001",
        manufacturer="Ford Motor Company",
        model_number="F-150",
        serial_number="1FTFW1E50NFA12345",
    )


@pytest.fixture
def sample_asset_it_equipment() -> CapitalAssetRecord:
    """Sample capital asset record for IT equipment."""
    return CapitalAssetRecord(
        asset_id="IT-2026-892",
        asset_name="Dell Server PowerEdge R750",
        asset_category=AssetCategory.IT_EQUIPMENT,
        asset_subcategory=AssetSubCategory.SERVERS,
        acquisition_date=date(2026, 1, 5),
        acquisition_cost_usd=Decimal("18500.00"),
        useful_life_years=5,
        depreciation_method="straight-line",
        currency_code="USD",
        reporting_year=2026,
        reporting_period_start=date(2026, 1, 1),
        reporting_period_end=date(2026, 12, 31),
        operational_unit_id="OU-IT-001",
        facility_id="FAC-USA-NYC-001",
        country_code="USA",
        region="North America",
        gl_account="1560-IT-Equipment",
        cost_center="CC-IT-001",
        manufacturer="Dell Technologies",
        model_number="PowerEdge R750",
        serial_number="R750-2026-ABCD1234",
    )


# ============================================================================
# CAPEX SPEND RECORD FIXTURES
# ============================================================================


@pytest.fixture
def sample_capex_construction() -> CapExSpendRecord:
    """Sample CapEx spend record for construction services."""
    return CapExSpendRecord(
        spend_id="CAPEX-2026-001",
        spend_description="Commercial building construction services",
        spend_amount_usd=Decimal("12500000.00"),
        spend_date=date(2026, 1, 15),
        vendor_name="ABC Construction LLC",
        vendor_country="USA",
        sector_classification=SectorClassification.NAICS,
        sector_code="236220",  # Commercial building construction
        sector_description="Commercial and Institutional Building Construction",
        currency_code="USD",
        reporting_year=2026,
        reporting_period_start=date(2026, 1, 1),
        reporting_period_end=date(2026, 12, 31),
        operational_unit_id="OU-HQ-001",
        cost_center="CC-FACILITIES-001",
        gl_account="1500-Buildings",
        purchase_order_number="PO-2026-1234",
    )


@pytest.fixture
def sample_capex_machinery() -> CapExSpendRecord:
    """Sample CapEx spend record for machinery purchase."""
    return CapExSpendRecord(
        spend_id="CAPEX-2026-042",
        spend_description="CNC milling machine purchase",
        spend_amount_usd=Decimal("450000.00"),
        spend_date=date(2026, 3, 10),
        vendor_name="Haas Automation Inc",
        vendor_country="USA",
        sector_classification=SectorClassification.NAICS,
        sector_code="333517",  # Machine tool manufacturing
        sector_description="Machine Tool Manufacturing",
        currency_code="USD",
        reporting_year=2026,
        reporting_period_start=date(2026, 1, 1),
        reporting_period_end=date(2026, 12, 31),
        operational_unit_id="OU-MFG-001",
        cost_center="CC-MANUFACTURING-001",
        gl_account="1520-Machinery",
        purchase_order_number="PO-2026-5678",
    )


# ============================================================================
# PHYSICAL RECORD FIXTURES
# ============================================================================


@pytest.fixture
def sample_physical_steel() -> PhysicalRecord:
    """Sample physical record for steel materials."""
    return PhysicalRecord(
        material_id="MAT-STEEL-001",
        material_name="Structural Steel I-Beams",
        material_type=ConstructionMaterial.STEEL,
        quantity=Decimal("125.5"),
        unit="metric_ton",
        unit_cost_usd=Decimal("850.00"),
        total_cost_usd=Decimal("106675.00"),
        acquisition_date=date(2026, 1, 20),
        supplier_name="US Steel Corporation",
        supplier_country="USA",
        manufacturing_process=ManufacturingProcessType.PRIMARY_STEEL_BOF,
        currency_code="USD",
        reporting_year=2026,
        reporting_period_start=date(2026, 1, 1),
        reporting_period_end=date(2026, 12, 31),
        operational_unit_id="OU-HQ-001",
        cost_center="CC-FACILITIES-001",
        purchase_order_number="PO-2026-STEEL-001",
    )


@pytest.fixture
def sample_physical_concrete() -> PhysicalRecord:
    """Sample physical record for concrete materials."""
    return PhysicalRecord(
        material_id="MAT-CONC-001",
        material_name="Ready-Mix Concrete C30/37",
        material_type=ConstructionMaterial.CONCRETE,
        quantity=Decimal("850.0"),
        unit="cubic_meter",
        unit_cost_usd=Decimal("125.00"),
        total_cost_usd=Decimal("106250.00"),
        acquisition_date=date(2026, 1, 25),
        supplier_name="CEMEX USA",
        supplier_country="USA",
        currency_code="USD",
        reporting_year=2026,
        reporting_period_start=date(2026, 1, 1),
        reporting_period_end=date(2026, 12, 31),
        operational_unit_id="OU-HQ-001",
        cost_center="CC-FACILITIES-001",
        purchase_order_number="PO-2026-CONC-001",
    )


@pytest.fixture
def sample_physical_electronics() -> PhysicalRecord:
    """Sample physical record for electronic components."""
    return PhysicalRecord(
        material_id="MAT-ELEC-001",
        material_name="Server motherboards and components",
        material_type=ConstructionMaterial.ELECTRONICS,
        quantity=Decimal("45.0"),
        unit="kilogram",
        unit_cost_usd=Decimal("220.00"),
        total_cost_usd=Decimal("9900.00"),
        acquisition_date=date(2026, 1, 5),
        supplier_name="Intel Corporation",
        supplier_country="USA",
        currency_code="USD",
        reporting_year=2026,
        reporting_period_start=date(2026, 1, 1),
        reporting_period_end=date(2026, 12, 31),
        operational_unit_id="OU-IT-001",
        cost_center="CC-IT-001",
        purchase_order_number="PO-2026-ELEC-001",
    )


# ============================================================================
# SUPPLIER RECORD FIXTURES
# ============================================================================


@pytest.fixture
def sample_supplier_epd() -> SupplierRecord:
    """Sample supplier record with EPD data."""
    return SupplierRecord(
        supplier_id="SUP-001",
        supplier_name="US Steel Corporation",
        product_name="Structural Steel I-Beams Grade A992",
        product_category=AssetCategory.BUILDINGS,
        data_type=SupplierDataType.EPD,
        emission_value=Decimal("1.85"),
        emission_unit=EmissionIntensityUnit.KG_CO2E_PER_KG,
        data_quality=SupplierDataQuality.HIGH,
        data_source_name="EPD International",
        data_source_url="https://www.environdec.com/EPD12345",
        data_year=2025,
        verification_status="third-party-verified",
        gwp_version="AR5",
        reporting_year=2026,
    )


@pytest.fixture
def sample_supplier_pcf() -> SupplierRecord:
    """Sample supplier record with Product Carbon Footprint."""
    return SupplierRecord(
        supplier_id="SUP-042",
        supplier_name="Haas Automation Inc",
        product_name="CNC Milling Machine X3000",
        product_category=AssetCategory.MACHINERY,
        data_type=SupplierDataType.PRODUCT_CARBON_FOOTPRINT,
        emission_value=Decimal("24.5"),
        emission_unit=EmissionIntensityUnit.TCO2E_PER_UNIT,
        data_quality=SupplierDataQuality.MEDIUM,
        data_source_name="Supplier PCF Report",
        data_year=2025,
        verification_status="self-reported",
        gwp_version="AR6",
        reporting_year=2026,
    )


@pytest.fixture
def sample_supplier_cdp() -> SupplierRecord:
    """Sample supplier record with CDP disclosure."""
    return SupplierRecord(
        supplier_id="SUP-156",
        supplier_name="Ford Motor Company",
        product_name="F-150 Commercial Truck",
        product_category=AssetCategory.VEHICLES,
        data_type=SupplierDataType.CDP_DISCLOSURE,
        emission_value=Decimal("18.2"),
        emission_unit=EmissionIntensityUnit.TCO2E_PER_UNIT,
        data_quality=SupplierDataQuality.HIGH,
        data_source_name="CDP Climate Change Questionnaire 2025",
        data_source_url="https://www.cdp.net/en/responses/12345",
        data_year=2025,
        verification_status="third-party-verified",
        gwp_version="AR5",
        reporting_year=2026,
    )


# ============================================================================
# CALCULATION REQUEST FIXTURES
# ============================================================================


@pytest.fixture
def sample_calculation_request_spend(sample_capex_construction: CapExSpendRecord) -> CalculationRequest:
    """Sample calculation request using spend-based method."""
    return CalculationRequest(
        request_id="REQ-2026-001",
        calculation_method=CalculationMethod.SPEND_BASED,
        capex_spend_record=sample_capex_construction,
        gwp_version="AR5",
        include_biogenic=False,
        apply_uncertainty_analysis=True,
        reporting_year=2026,
    )


@pytest.fixture
def sample_calculation_request_average(sample_asset_machinery: CapitalAssetRecord) -> CalculationRequest:
    """Sample calculation request using average-data method."""
    return CalculationRequest(
        request_id="REQ-2026-042",
        calculation_method=CalculationMethod.AVERAGE_DATA,
        asset_record=sample_asset_machinery,
        gwp_version="AR5",
        include_biogenic=False,
        apply_uncertainty_analysis=True,
        reporting_year=2026,
    )


@pytest.fixture
def sample_calculation_request_supplier(
    sample_asset_vehicle: CapitalAssetRecord,
    sample_supplier_cdp: SupplierRecord,
) -> CalculationRequest:
    """Sample calculation request using supplier-specific method."""
    return CalculationRequest(
        request_id="REQ-2026-156",
        calculation_method=CalculationMethod.SUPPLIER_SPECIFIC,
        asset_record=sample_asset_vehicle,
        supplier_record=sample_supplier_cdp,
        gwp_version="AR5",
        include_biogenic=False,
        apply_uncertainty_analysis=False,
        reporting_year=2026,
    )


@pytest.fixture
def sample_calculation_request_hybrid(
    sample_asset_building: CapitalAssetRecord,
    sample_capex_construction: CapExSpendRecord,
) -> CalculationRequest:
    """Sample calculation request using hybrid method."""
    physical_records = [
        PhysicalRecord(
            material_id="MAT-STEEL-001",
            material_name="Structural Steel",
            material_type=ConstructionMaterial.STEEL,
            quantity=Decimal("125.5"),
            unit="metric_ton",
            unit_cost_usd=Decimal("850.00"),
            total_cost_usd=Decimal("106675.00"),
            acquisition_date=date(2026, 1, 20),
            supplier_name="US Steel Corporation",
            supplier_country="USA",
            currency_code="USD",
            reporting_year=2026,
        )
    ]

    return CalculationRequest(
        request_id="REQ-2026-HYBRID-001",
        calculation_method=CalculationMethod.HYBRID,
        asset_record=sample_asset_building,
        capex_spend_record=sample_capex_construction,
        physical_records=physical_records,
        gwp_version="AR5",
        include_biogenic=False,
        apply_uncertainty_analysis=True,
        reporting_year=2026,
    )


@pytest.fixture
def sample_batch_request(
    sample_calculation_request_spend: CalculationRequest,
    sample_calculation_request_average: CalculationRequest,
) -> BatchRequest:
    """Sample batch calculation request."""
    return BatchRequest(
        batch_id="BATCH-2026-001",
        requests=[
            sample_calculation_request_spend,
            sample_calculation_request_average,
        ],
        parallel_processing=True,
        max_workers=4,
        reporting_year=2026,
    )


# ============================================================================
# RESULT FIXTURES
# ============================================================================


@pytest.fixture
def sample_spend_based_result() -> SpendBasedResult:
    """Sample spend-based calculation result."""
    return SpendBasedResult(
        request_id="REQ-2026-001",
        calculation_method=CalculationMethod.SPEND_BASED,
        total_emissions_tco2e=Decimal("850.25"),
        scope3_category="3.2",
        spend_amount_usd=Decimal("12500000.00"),
        emission_factor=Decimal("0.068"),
        emission_factor_unit="tCO2e/USD",
        sector_code="236220",
        sector_classification=SectorClassification.NAICS,
        data_quality_score=Decimal("3.2"),
        uncertainty_percentage=Decimal("45.0"),
        gwp_version="AR5",
        calculation_timestamp=datetime(2026, 2, 25, 10, 30, 0),
    )


@pytest.fixture
def sample_average_data_result() -> AverageDataResult:
    """Sample average-data calculation result."""
    return AverageDataResult(
        request_id="REQ-2026-042",
        calculation_method=CalculationMethod.AVERAGE_DATA,
        total_emissions_tco2e=Decimal("24.5"),
        scope3_category="3.2",
        asset_category=AssetCategory.MACHINERY,
        asset_subcategory=AssetSubCategory.PRODUCTION_EQUIPMENT,
        emission_factor=Decimal("54.44"),
        emission_factor_unit="kgCO2e/kg",
        estimated_mass_kg=Decimal("450.0"),
        data_quality_score=Decimal("3.8"),
        uncertainty_percentage=Decimal("35.0"),
        gwp_version="AR5",
        calculation_timestamp=datetime(2026, 2, 25, 10, 35, 0),
    )


@pytest.fixture
def sample_supplier_specific_result() -> SupplierSpecificResult:
    """Sample supplier-specific calculation result."""
    return SupplierSpecificResult(
        request_id="REQ-2026-156",
        calculation_method=CalculationMethod.SUPPLIER_SPECIFIC,
        total_emissions_tco2e=Decimal("18.2"),
        scope3_category="3.2",
        supplier_name="Ford Motor Company",
        product_name="F-150 Commercial Truck",
        supplier_data_type=SupplierDataType.CDP_DISCLOSURE,
        supplier_data_quality=SupplierDataQuality.HIGH,
        verification_status="third-party-verified",
        data_quality_score=Decimal("4.5"),
        uncertainty_percentage=Decimal("15.0"),
        gwp_version="AR5",
        calculation_timestamp=datetime(2026, 2, 25, 10, 40, 0),
    )


@pytest.fixture
def sample_hybrid_result() -> HybridResult:
    """Sample hybrid calculation result."""
    return HybridResult(
        request_id="REQ-2026-HYBRID-001",
        calculation_method=CalculationMethod.HYBRID,
        total_emissions_tco2e=Decimal("875.50"),
        scope3_category="3.2",
        physical_portion_tco2e=Decimal("232.18"),
        spend_portion_tco2e=Decimal("643.32"),
        physical_coverage_percentage=Decimal("26.5"),
        data_quality_score=Decimal("4.1"),
        uncertainty_percentage=Decimal("25.0"),
        gwp_version="AR5",
        calculation_timestamp=datetime(2026, 2, 25, 10, 45, 0),
    )


# ============================================================================
# CONFIG AND SERVICE FIXTURES
# ============================================================================


@pytest.fixture
def sample_config() -> dict[str, Any]:
    """Sample configuration dictionary."""
    return {
        "database": {
            "host": "localhost",
            "port": 5432,
            "database": "greenlang_test",
            "user": "test_user",
            "password": "test_password",
        },
        "calculation": {
            "default_gwp_version": "AR5",
            "capitalization_threshold_usd": Decimal("5000.00"),
            "apply_uncertainty_by_default": True,
            "rolling_average_years": 3,
            "volatility_ratio_threshold": Decimal("0.25"),
        },
        "data_quality": {
            "min_acceptable_score": Decimal("2.0"),
            "high_quality_threshold": Decimal("4.0"),
            "require_verification_for_supplier_data": False,
        },
        "performance": {
            "batch_size": 100,
            "max_workers": 4,
            "cache_ttl_seconds": 3600,
            "query_timeout_seconds": 30,
        },
    }


@pytest.fixture
async def mock_db_engine() -> AsyncIterator[AsyncEngine]:
    """Mock async database engine for testing."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    yield engine
    await engine.dispose()


@pytest.fixture
def mock_db_session() -> AsyncSession:
    """Mock async database session."""
    session = AsyncMock(spec=AsyncSession)
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.close = AsyncMock()
    return session


@pytest.fixture
def mock_capital_goods_service() -> MagicMock:
    """Mock Capital Goods Service."""
    service = MagicMock()
    service.calculate = AsyncMock()
    service.calculate_batch = AsyncMock()
    service.get_emission_factor = AsyncMock(return_value=Decimal("0.068"))
    service.validate_data_quality = MagicMock(return_value=True)
    return service
