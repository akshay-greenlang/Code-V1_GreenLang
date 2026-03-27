"""
Pytest fixtures for Purchased Goods & Services Agent test suite.

Provides reusable test data for all test modules.
"""

import pytest
from decimal import Decimal
from datetime import date, datetime
from typing import List

from greenlang.agents.mrv.purchased_goods_services.models import (
    ProcurementItem,
    SupplierRecord,
    SpendBasedResult,
    AverageDataResult,
    SupplierSpecificResult,
    HybridResult,
    BatchRequest,
    DQIAssessment,
    CalculationMethod,
    EEIODatabase,
    PhysicalEFSource,
    SupplierDataSource,
    AllocationMethod,
    MaterialCategory,
    CurrencyCode,
    DQIDimension,
    ComplianceFramework,
    ProcurementType,
    CoverageLevel,
    ExportFormat,
    SpendRecord,
    PhysicalRecord,
    ComplianceDisclosure,
)


@pytest.fixture
def sample_procurement_item() -> ProcurementItem:
    """Basic goods procurement item (steel beams)."""
    return ProcurementItem(
        item_id="ITEM-001",
        item_name="Hot-rolled steel beams",
        procurement_type=ProcurementType.GOODS,
        category=MaterialCategory.METALS_STEEL,
        supplier_id="SUP-STEEL-123",
        supplier_name="SteelCorp Inc.",
        quantity=Decimal("15000.00"),
        unit="kg",
        unit_cost=Decimal("2.50"),
        total_spend=Decimal("37500.00"),
        currency=CurrencyCode.USD,
        period_start=date(2024, 1, 1),
        period_end=date(2024, 12, 31),
        country_origin="USA",
        calculation_method=CalculationMethod.SPEND_BASED,
    )


@pytest.fixture
def sample_service_item() -> ProcurementItem:
    """Professional service procurement item."""
    return ProcurementItem(
        item_id="ITEM-SVC-001",
        item_name="Management consulting services",
        procurement_type=ProcurementType.SERVICES,
        category=MaterialCategory.BUSINESS_SERVICES,
        supplier_id="SUP-CONSULT-456",
        supplier_name="Consulting Partners LLC",
        quantity=Decimal("1000.00"),
        unit="hours",
        unit_cost=Decimal("150.00"),
        total_spend=Decimal("150000.00"),
        currency=CurrencyCode.USD,
        period_start=date(2024, 1, 1),
        period_end=date(2024, 12, 31),
        country_origin="USA",
        calculation_method=CalculationMethod.SPEND_BASED,
    )


@pytest.fixture
def sample_supplier_record() -> SupplierRecord:
    """Supplier record with EPD data."""
    return SupplierRecord(
        supplier_id="SUP-CONCRETE-789",
        supplier_name="ConcreteCo Ltd.",
        data_source=SupplierDataSource.EPD,
        emission_factor=Decimal("0.245"),
        emission_factor_unit="kgCO2e/kg",
        validity_start=date(2024, 1, 1),
        validity_end=date(2026, 12, 31),
        verification_body="NSF International",
        epd_number="EPD-CONC-2024-001",
        data_quality_score=Decimal("4.5"),
        coverage_percentage=Decimal("95.0"),
    )


@pytest.fixture
def sample_spend_based_result() -> SpendBasedResult:
    """Spend-based calculation result."""
    return SpendBasedResult(
        item_id="ITEM-001",
        method=CalculationMethod.SPEND_BASED,
        total_emissions=Decimal("12500.00"),
        emissions_unit="kgCO2e",
        eeio_database=EEIODatabase.EXIOBASE,
        eeio_sector="Iron and steel",
        emission_factor=Decimal("0.833"),
        emission_factor_unit="kgCO2e/USD",
        spend_amount=Decimal("37500.00"),
        spend_currency=CurrencyCode.USD,
        dqi_assessment=DQIAssessment(
            technological_representativeness=3,
            temporal_representativeness=4,
            geographical_representativeness=4,
            completeness=5,
            reliability=4,
            overall_score=Decimal("4.0"),
        ),
        calculation_timestamp=datetime(2024, 1, 15, 10, 30, 0),
        provenance_hash="a1b2c3d4e5f6",
    )


@pytest.fixture
def sample_average_data_result() -> AverageDataResult:
    """Average-data calculation result."""
    return AverageDataResult(
        item_id="ITEM-002",
        method=CalculationMethod.AVERAGE_DATA,
        total_emissions=Decimal("8750.00"),
        emissions_unit="kgCO2e",
        ef_source=PhysicalEFSource.IPCC_2006,
        emission_factor=Decimal("0.583"),
        emission_factor_unit="kgCO2e/kg",
        quantity=Decimal("15000.00"),
        quantity_unit="kg",
        dqi_assessment=DQIAssessment(
            technological_representativeness=4,
            temporal_representativeness=5,
            geographical_representativeness=3,
            completeness=4,
            reliability=5,
            overall_score=Decimal("4.2"),
        ),
        calculation_timestamp=datetime(2024, 1, 15, 10, 35, 0),
        provenance_hash="b2c3d4e5f6a7",
    )


@pytest.fixture
def sample_supplier_specific_result() -> SupplierSpecificResult:
    """Supplier-specific calculation result."""
    return SupplierSpecificResult(
        item_id="ITEM-003",
        method=CalculationMethod.SUPPLIER_SPECIFIC,
        total_emissions=Decimal("6200.00"),
        emissions_unit="kgCO2e",
        supplier_id="SUP-CONCRETE-789",
        supplier_name="ConcreteCo Ltd.",
        data_source=SupplierDataSource.EPD,
        emission_factor=Decimal("0.245"),
        emission_factor_unit="kgCO2e/kg",
        quantity=Decimal("25300.00"),
        quantity_unit="kg",
        verification_status="Verified by NSF International",
        epd_number="EPD-CONC-2024-001",
        dqi_assessment=DQIAssessment(
            technological_representativeness=5,
            temporal_representativeness=5,
            geographical_representativeness=5,
            completeness=5,
            reliability=5,
            overall_score=Decimal("5.0"),
        ),
        calculation_timestamp=datetime(2024, 1, 15, 10, 40, 0),
        provenance_hash="c3d4e5f6a7b8",
    )


@pytest.fixture
def sample_hybrid_result() -> HybridResult:
    """Hybrid calculation result."""
    return HybridResult(
        item_id="ITEM-HYBRID-001",
        method=CalculationMethod.HYBRID,
        total_emissions=Decimal("18500.00"),
        emissions_unit="kgCO2e",
        supplier_specific_emissions=Decimal("12000.00"),
        supplier_specific_coverage=Decimal("65.0"),
        average_data_emissions=Decimal("6500.00"),
        average_data_coverage=Decimal("35.0"),
        coverage_level=CoverageLevel.MEDIUM,
        supplier_records_used=5,
        average_data_records_used=3,
        dqi_assessment=DQIAssessment(
            technological_representativeness=4,
            temporal_representativeness=4,
            geographical_representativeness=4,
            completeness=4,
            reliability=4,
            overall_score=Decimal("4.0"),
        ),
        calculation_timestamp=datetime(2024, 1, 15, 10, 45, 0),
        provenance_hash="d4e5f6a7b8c9",
    )


@pytest.fixture
def sample_batch_request() -> BatchRequest:
    """Batch request for multiple procurement items."""
    return BatchRequest(
        request_id="BATCH-2024-Q1",
        tenant_id="tenant-acme-corp",
        organization_id="org-acme-123",
        facility_id="facility-main-plant",
        reporting_period_start=date(2024, 1, 1),
        reporting_period_end=date(2024, 3, 31),
        item_ids=["ITEM-001", "ITEM-002", "ITEM-003", "ITEM-004", "ITEM-005"],
        calculation_methods=[
            CalculationMethod.SUPPLIER_SPECIFIC,
            CalculationMethod.AVERAGE_DATA,
            CalculationMethod.SPEND_BASED,
        ],
        compliance_frameworks=[
            ComplianceFramework.GHG_PROTOCOL,
            ComplianceFramework.ISO_14064_1,
        ],
        export_format=ExportFormat.JSON,
    )


@pytest.fixture
def sample_procurement_items() -> List[ProcurementItem]:
    """List of 10 procurement items across various sectors."""
    return [
        ProcurementItem(
            item_id="ITEM-MULTI-001",
            item_name="Cement",
            procurement_type=ProcurementType.GOODS,
            category=MaterialCategory.CEMENT_LIME,
            supplier_id="SUP-001",
            supplier_name="CementCo",
            quantity=Decimal("50000.00"),
            unit="kg",
            unit_cost=Decimal("0.12"),
            total_spend=Decimal("6000.00"),
            currency=CurrencyCode.USD,
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
            country_origin="USA",
            calculation_method=CalculationMethod.AVERAGE_DATA,
        ),
        ProcurementItem(
            item_id="ITEM-MULTI-002",
            item_name="Aluminum sheets",
            procurement_type=ProcurementType.GOODS,
            category=MaterialCategory.METALS_ALUMINUM,
            supplier_id="SUP-002",
            supplier_name="AluminumCo",
            quantity=Decimal("8000.00"),
            unit="kg",
            unit_cost=Decimal("3.50"),
            total_spend=Decimal("28000.00"),
            currency=CurrencyCode.USD,
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
            country_origin="Canada",
            calculation_method=CalculationMethod.SUPPLIER_SPECIFIC,
        ),
        ProcurementItem(
            item_id="ITEM-MULTI-003",
            item_name="Plastic pellets (HDPE)",
            procurement_type=ProcurementType.GOODS,
            category=MaterialCategory.PLASTICS,
            supplier_id="SUP-003",
            supplier_name="PlasticsCo",
            quantity=Decimal("12000.00"),
            unit="kg",
            unit_cost=Decimal("1.80"),
            total_spend=Decimal("21600.00"),
            currency=CurrencyCode.USD,
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
            country_origin="Germany",
            calculation_method=CalculationMethod.AVERAGE_DATA,
        ),
        ProcurementItem(
            item_id="ITEM-MULTI-004",
            item_name="Paper packaging",
            procurement_type=ProcurementType.GOODS,
            category=MaterialCategory.PAPER_CARDBOARD,
            supplier_id="SUP-004",
            supplier_name="PaperCo",
            quantity=Decimal("25000.00"),
            unit="kg",
            unit_cost=Decimal("0.45"),
            total_spend=Decimal("11250.00"),
            currency=CurrencyCode.USD,
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
            country_origin="Sweden",
            calculation_method=CalculationMethod.SPEND_BASED,
        ),
        ProcurementItem(
            item_id="ITEM-MULTI-005",
            item_name="IT hardware",
            procurement_type=ProcurementType.GOODS,
            category=MaterialCategory.ELECTRONICS,
            supplier_id="SUP-005",
            supplier_name="TechCo",
            quantity=Decimal("50.00"),
            unit="units",
            unit_cost=Decimal("800.00"),
            total_spend=Decimal("40000.00"),
            currency=CurrencyCode.USD,
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
            country_origin="Taiwan",
            calculation_method=CalculationMethod.SPEND_BASED,
        ),
        ProcurementItem(
            item_id="ITEM-MULTI-006",
            item_name="Freight transport",
            procurement_type=ProcurementType.SERVICES,
            category=MaterialCategory.LOGISTICS,
            supplier_id="SUP-006",
            supplier_name="LogisticsCo",
            quantity=Decimal("150000.00"),
            unit="tonne-km",
            unit_cost=Decimal("0.08"),
            total_spend=Decimal("12000.00"),
            currency=CurrencyCode.USD,
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
            country_origin="USA",
            calculation_method=CalculationMethod.AVERAGE_DATA,
        ),
        ProcurementItem(
            item_id="ITEM-MULTI-007",
            item_name="Legal services",
            procurement_type=ProcurementType.SERVICES,
            category=MaterialCategory.BUSINESS_SERVICES,
            supplier_id="SUP-007",
            supplier_name="LawFirm LLC",
            quantity=Decimal("500.00"),
            unit="hours",
            unit_cost=Decimal("300.00"),
            total_spend=Decimal("150000.00"),
            currency=CurrencyCode.USD,
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
            country_origin="USA",
            calculation_method=CalculationMethod.SPEND_BASED,
        ),
        ProcurementItem(
            item_id="ITEM-MULTI-008",
            item_name="Organic chemicals",
            procurement_type=ProcurementType.GOODS,
            category=MaterialCategory.CHEMICALS_ORGANIC,
            supplier_id="SUP-008",
            supplier_name="ChemicalsCo",
            quantity=Decimal("18000.00"),
            unit="kg",
            unit_cost=Decimal("2.20"),
            total_spend=Decimal("39600.00"),
            currency=CurrencyCode.EUR,
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
            country_origin="Netherlands",
            calculation_method=CalculationMethod.AVERAGE_DATA,
        ),
        ProcurementItem(
            item_id="ITEM-MULTI-009",
            item_name="Glass sheets",
            procurement_type=ProcurementType.GOODS,
            category=MaterialCategory.GLASS,
            supplier_id="SUP-009",
            supplier_name="GlassCo",
            quantity=Decimal("22000.00"),
            unit="kg",
            unit_cost=Decimal("1.10"),
            total_spend=Decimal("24200.00"),
            currency=CurrencyCode.USD,
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
            country_origin="USA",
            calculation_method=CalculationMethod.SPEND_BASED,
        ),
        ProcurementItem(
            item_id="ITEM-MULTI-010",
            item_name="Textiles",
            procurement_type=ProcurementType.GOODS,
            category=MaterialCategory.TEXTILES,
            supplier_id="SUP-010",
            supplier_name="TextileCo",
            quantity=Decimal("5000.00"),
            unit="kg",
            unit_cost=Decimal("8.50"),
            total_spend=Decimal("42500.00"),
            currency=CurrencyCode.USD,
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
            country_origin="India",
            calculation_method=CalculationMethod.HYBRID,
        ),
    ]


@pytest.fixture
def sample_disclosures() -> List[ComplianceDisclosure]:
    """Compliance disclosure data for multiple frameworks."""
    return [
        ComplianceDisclosure(
            framework=ComplianceFramework.GHG_PROTOCOL,
            disclosure_id="GHG-SCOPE3-CAT1-2024",
            total_emissions=Decimal("125000.00"),
            emissions_unit="tCO2e",
            spend_based_percentage=Decimal("35.0"),
            average_data_percentage=Decimal("25.0"),
            supplier_specific_percentage=Decimal("40.0"),
            data_quality_rating="Medium",
            coverage_completeness=Decimal("92.0"),
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
        ),
        ComplianceDisclosure(
            framework=ComplianceFramework.CSRD,
            disclosure_id="CSRD-E1-2024",
            total_emissions=Decimal("125000.00"),
            emissions_unit="tCO2e",
            spend_based_percentage=Decimal("35.0"),
            average_data_percentage=Decimal("25.0"),
            supplier_specific_percentage=Decimal("40.0"),
            data_quality_rating="Medium",
            coverage_completeness=Decimal("92.0"),
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
        ),
    ]
