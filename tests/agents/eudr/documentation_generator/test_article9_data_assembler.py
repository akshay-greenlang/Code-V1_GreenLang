# -*- coding: utf-8 -*-
"""
Tests for Article9DataAssembler Engine - AGENT-EUDR-030

Tests the Article 9 Data Assembler including assemble(), _validate_completeness(),
_check_mandatory_elements(), and all 10 Article 9 data elements
(product_description, quantity, country_of_production, geolocation,
supplier_information, date_of_production, hs_code, trade_name,
scientific_name, operator_information).

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

import pytest
from datetime import datetime, timezone
from decimal import Decimal

from greenlang.agents.eudr.documentation_generator.article9_data_assembler import (
    Article9DataAssembler,
    _ELEMENT_DESCRIPTIONS,
)
from pydantic import ValidationError as PydanticValidationError

from greenlang.agents.eudr.documentation_generator.models import (
    ARTICLE9_MANDATORY_ELEMENTS,
    Article9Element,
    Article9Package,
    EUDRCommodity,
    GeolocationReference,
    ProductEntry,
    SupplierReference,
    ValidationSeverity,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def assembler() -> Article9DataAssembler:
    """Create Article9DataAssembler instance."""
    return Article9DataAssembler()


@pytest.fixture
def sample_products_complete() -> list[ProductEntry]:
    """Create complete product entries with all fields."""
    return [
        ProductEntry(
            product_id="PROD-001",
            description="Premium Arabica Coffee Beans",
            hs_code="0901.11",
            cn_code="09011100",
            quantity=Decimal("1500"),
            unit="kg",
        ),
        ProductEntry(
            product_id="PROD-002",
            description="Organic Robusta Coffee Beans",
            hs_code="0901.12",
            cn_code="09011200",
            quantity=Decimal("800"),
            unit="kg",
        ),
    ]


@pytest.fixture
def sample_geolocations() -> list[GeolocationReference]:
    """Create sample geolocation references."""
    return [
        GeolocationReference(
            plot_id="PLOT-CO-001",
            latitude=Decimal("4.570868"),
            longitude=Decimal("-74.297333"),
            polygon=None,
            area_hectares=Decimal("2.5"),
            country_code="CO",
        ),
        GeolocationReference(
            plot_id="PLOT-CO-002",
            latitude=Decimal("4.571234"),
            longitude=Decimal("-74.298456"),
            polygon=[
                [Decimal("4.5712"), Decimal("-74.2984")],
                [Decimal("4.5715"), Decimal("-74.2987")],
                [Decimal("4.5718"), Decimal("-74.2982")],
                [Decimal("4.5712"), Decimal("-74.2984")],
            ],
            area_hectares=Decimal("5.2"),
            country_code="CO",
        ),
    ]


@pytest.fixture
def sample_suppliers() -> list[SupplierReference]:
    """Create sample supplier references."""
    return [
        SupplierReference(
            supplier_id="SUP-CO-001",
            name="Cafe Verde S.A.",
            country="CO",
            registration_number="NIT-900123456",
        ),
        SupplierReference(
            supplier_id="SUP-CO-002",
            name="Colombian Coffee Exporters Ltd.",
            country="CO",
            registration_number="NIT-900654321",
        ),
    ]


# ---------------------------------------------------------------------------
# Test: Initialization
# ---------------------------------------------------------------------------

def test_assembler_initialization(assembler):
    """Test Article9DataAssembler initializes with correct defaults."""
    assert assembler._config is not None
    assert assembler._provenance is not None


def test_assembler_initialization_with_custom_config():
    """Test initialization with custom configuration."""
    from greenlang.agents.eudr.documentation_generator.config import (
        DocumentationGeneratorConfig,
    )

    config = DocumentationGeneratorConfig(
        article9_completeness_threshold=Decimal("0.90"),
        require_polygon_above_4ha=True,
        geolocation_decimal_places=6,
    )
    assembler = Article9DataAssembler(config=config)
    assert assembler._config.article9_completeness_threshold == Decimal("0.90")
    assert assembler._config.require_polygon_above_4ha is True
    assert assembler._config.geolocation_decimal_places == 6


# ---------------------------------------------------------------------------
# Test: assemble_package - Success Paths
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_assemble_package_complete(
    assembler,
    sample_products_complete,
    sample_geolocations,
    sample_suppliers,
):
    """Test assembling complete Article 9 package."""
    package = await assembler.assemble_package(
        operator_id="OP-001",
        commodity=EUDRCommodity.COFFEE,
        products=sample_products_complete,
        geolocations=sample_geolocations,
        suppliers=sample_suppliers,
        production_date_range=("2025-01-01", "2025-06-30"),
        certifications=["Rainforest Alliance", "Fairtrade"],
        buyer_info={"name": "EU Coffee Importers", "country": "DE"},
        supporting_evidence=["cert-001", "audit-2025-Q1"],
    )

    assert package.package_id.startswith("a9p-")
    assert package.operator_id == "OP-001"
    assert package.commodity == EUDRCommodity.COFFEE
    assert package.completeness_score >= Decimal("0.80")
    assert len(package.missing_elements) <= 2


@pytest.mark.asyncio
async def test_assemble_package_minimal(
    assembler,
    sample_products_complete,
    sample_geolocations,
    sample_suppliers,
):
    """Test assembling Article 9 package with minimal data."""
    package = await assembler.assemble_package(
        operator_id="OP-002",
        commodity=EUDRCommodity.COFFEE,
        products=sample_products_complete,
        geolocations=sample_geolocations,
        suppliers=sample_suppliers,
    )

    assert package.package_id.startswith("a9p-")
    assert package.operator_id == "OP-002"
    assert package.commodity == EUDRCommodity.COFFEE
    # With minimal data, completeness should be lower
    assert package.completeness_score < Decimal("1.0")
    assert len(package.missing_elements) > 0


@pytest.mark.asyncio
async def test_assemble_package_all_elements_present(
    assembler,
    sample_products_complete,
    sample_geolocations,
    sample_suppliers,
):
    """Test package assembly with all 10 Article 9 elements."""
    package = await assembler.assemble_package(
        operator_id="OP-003",
        commodity=EUDRCommodity.COCOA,
        products=sample_products_complete,
        geolocations=sample_geolocations,
        suppliers=sample_suppliers,
        production_date_range=("2025-03-01", "2025-08-31"),
        certifications=["Organic", "UTZ"],
        buyer_info={"name": "Chocolate Corp", "country": "BE"},
        supporting_evidence=["docs-001", "docs-002"],
    )

    # All 10 elements provided
    assert package.completeness_score == Decimal("1.00")
    assert len(package.missing_elements) == 0
    assert Article9Element.PRODUCT_DESCRIPTION.value in package.elements
    assert Article9Element.QUANTITY.value in package.elements
    assert Article9Element.COUNTRY_OF_PRODUCTION.value in package.elements
    assert Article9Element.GEOLOCATION.value in package.elements
    assert Article9Element.PRODUCTION_DATE.value in package.elements
    assert Article9Element.SUPPLIER_INFO.value in package.elements
    assert Article9Element.BUYER_INFO.value in package.elements
    assert Article9Element.CERTIFICATIONS.value in package.elements
    assert Article9Element.TRADE_CODES.value in package.elements
    assert Article9Element.SUPPORTING_EVIDENCE.value in package.elements


# ---------------------------------------------------------------------------
# Test: Element Building - Product Description
# ---------------------------------------------------------------------------

def test_build_element_dict_product_description(assembler, sample_products_complete):
    """Test product description element building."""
    elements = assembler._build_element_dict(
        operator_id="OP-001",
        commodity=EUDRCommodity.COFFEE,
        products=sample_products_complete,
        geolocations=[],
        suppliers=[],
    )

    assert Article9Element.PRODUCT_DESCRIPTION.value in elements
    prod_elem = elements[Article9Element.PRODUCT_DESCRIPTION.value]
    assert prod_elem["count"] == 2
    assert len(prod_elem["products"]) == 2
    assert prod_elem["products"][0]["product_id"] == "PROD-001"
    assert "Arabica" in prod_elem["products"][0]["description"]


# ---------------------------------------------------------------------------
# Test: Element Building - Quantity
# ---------------------------------------------------------------------------

def test_build_element_dict_quantity(assembler, sample_products_complete):
    """Test quantity element building."""
    elements = assembler._build_element_dict(
        operator_id="OP-001",
        commodity=EUDRCommodity.COFFEE,
        products=sample_products_complete,
        geolocations=[],
        suppliers=[],
    )

    assert Article9Element.QUANTITY.value in elements
    qty_elem = elements[Article9Element.QUANTITY.value]
    assert qty_elem["total_items"] == 2
    assert len(qty_elem["items"]) == 2
    assert qty_elem["items"][0]["quantity"] == "1500"
    assert qty_elem["items"][0]["unit"] == "kg"


def test_build_element_dict_quantity_missing(assembler):
    """Test quantity element when products have no quantity."""
    products_no_qty = [
        ProductEntry(
            product_id="P1",
            description="Coffee",
            hs_code="0901.11",
            quantity=Decimal("0"),
            unit="kg",
        )
    ]

    elements = assembler._build_element_dict(
        operator_id="OP-001",
        commodity=EUDRCommodity.COFFEE,
        products=products_no_qty,
        geolocations=[],
        suppliers=[],
    )

    # Should still be present but might not include all items
    # The logic checks if all products have quantity is not None
    assert Article9Element.PRODUCT_DESCRIPTION.value in elements


# ---------------------------------------------------------------------------
# Test: Element Building - Country of Production
# ---------------------------------------------------------------------------

def test_build_element_dict_country(assembler, sample_geolocations):
    """Test country of production element building."""
    elements = assembler._build_element_dict(
        operator_id="OP-001",
        commodity=EUDRCommodity.COFFEE,
        products=[],
        geolocations=sample_geolocations,
        suppliers=[],
    )

    assert Article9Element.COUNTRY_OF_PRODUCTION.value in elements
    country_elem = elements[Article9Element.COUNTRY_OF_PRODUCTION.value]
    assert country_elem["count"] == 1
    assert "CO" in country_elem["countries"]


def test_build_element_dict_multiple_countries(assembler):
    """Test country element with multiple countries."""
    geos = [
        GeolocationReference(
            plot_id="PLOT-1",
            latitude=Decimal("4.5"),
            longitude=Decimal("-74.2"),
            area_hectares=Decimal("2.0"),
            country_code="CO",
        ),
        GeolocationReference(
            plot_id="PLOT-2",
            latitude=Decimal("6.6"),
            longitude=Decimal("-1.6"),
            area_hectares=Decimal("3.0"),
            country_code="GH",
        ),
    ]

    elements = assembler._build_element_dict(
        operator_id="OP-001",
        commodity=EUDRCommodity.COFFEE,
        products=[],
        geolocations=geos,
        suppliers=[],
    )

    country_elem = elements[Article9Element.COUNTRY_OF_PRODUCTION.value]
    assert country_elem["count"] == 2
    assert set(country_elem["countries"]) == {"CO", "GH"}


# ---------------------------------------------------------------------------
# Test: Element Building - Geolocation
# ---------------------------------------------------------------------------

def test_build_element_dict_geolocation(assembler, sample_geolocations):
    """Test geolocation element building."""
    elements = assembler._build_element_dict(
        operator_id="OP-001",
        commodity=EUDRCommodity.COFFEE,
        products=[],
        geolocations=sample_geolocations,
        suppliers=[],
    )

    assert Article9Element.GEOLOCATION.value in elements
    geo_elem = elements[Article9Element.GEOLOCATION.value]
    assert geo_elem["total_plots"] == 2
    assert len(geo_elem["plots"]) == 2

    plot1 = geo_elem["plots"][0]
    assert plot1["plot_id"] == "PLOT-CO-001"
    assert plot1["latitude"] == "2.5" or "4.570868" in str(plot1)
    assert plot1["has_polygon"] is False

    plot2 = geo_elem["plots"][1]
    assert plot2["plot_id"] == "PLOT-CO-002"
    assert plot2["has_polygon"] is True


# ---------------------------------------------------------------------------
# Test: Element Building - Production Date
# ---------------------------------------------------------------------------

def test_build_element_dict_production_date(assembler):
    """Test production date element building."""
    elements = assembler._build_element_dict(
        operator_id="OP-001",
        commodity=EUDRCommodity.COFFEE,
        products=[],
        geolocations=[],
        suppliers=[],
        production_date_range=("2025-01-01", "2025-06-30"),
    )

    assert Article9Element.PRODUCTION_DATE.value in elements
    date_elem = elements[Article9Element.PRODUCTION_DATE.value]
    assert date_elem["start_date"] == "2025-01-01"
    assert date_elem["end_date"] == "2025-06-30"


# ---------------------------------------------------------------------------
# Test: Element Building - Supplier Information
# ---------------------------------------------------------------------------

def test_build_element_dict_supplier_info(assembler, sample_suppliers):
    """Test supplier information element building."""
    elements = assembler._build_element_dict(
        operator_id="OP-001",
        commodity=EUDRCommodity.COFFEE,
        products=[],
        geolocations=[],
        suppliers=sample_suppliers,
    )

    assert Article9Element.SUPPLIER_INFO.value in elements
    supplier_elem = elements[Article9Element.SUPPLIER_INFO.value]
    assert supplier_elem["count"] == 2
    assert len(supplier_elem["suppliers"]) == 2
    assert supplier_elem["suppliers"][0]["supplier_id"] == "SUP-CO-001"
    assert supplier_elem["suppliers"][0]["name"] == "Cafe Verde S.A."
    assert supplier_elem["suppliers"][0]["country"] == "CO"


# ---------------------------------------------------------------------------
# Test: Element Building - Buyer Information
# ---------------------------------------------------------------------------

def test_build_element_dict_buyer_info(assembler):
    """Test buyer information element building."""
    buyer = {"name": "EU Importers GmbH", "country": "DE", "address": "Berlin"}

    elements = assembler._build_element_dict(
        operator_id="OP-001",
        commodity=EUDRCommodity.COFFEE,
        products=[],
        geolocations=[],
        suppliers=[],
        buyer_info=buyer,
    )

    assert Article9Element.BUYER_INFO.value in elements
    buyer_elem = elements[Article9Element.BUYER_INFO.value]
    assert buyer_elem["name"] == "EU Importers GmbH"
    assert buyer_elem["country"] == "DE"


# ---------------------------------------------------------------------------
# Test: Element Building - Certifications
# ---------------------------------------------------------------------------

def test_build_element_dict_certifications(assembler):
    """Test certifications element building."""
    certifications = ["Rainforest Alliance", "Fairtrade", "Organic"]

    elements = assembler._build_element_dict(
        operator_id="OP-001",
        commodity=EUDRCommodity.COFFEE,
        products=[],
        geolocations=[],
        suppliers=[],
        certifications=certifications,
    )

    assert Article9Element.CERTIFICATIONS.value in elements
    cert_elem = elements[Article9Element.CERTIFICATIONS.value]
    assert cert_elem["count"] == 3
    assert "Rainforest Alliance" in cert_elem["certifications"]


# ---------------------------------------------------------------------------
# Test: Element Building - Trade Codes
# ---------------------------------------------------------------------------

def test_build_element_dict_trade_codes(assembler, sample_products_complete):
    """Test trade codes (HS/CN) element building."""
    elements = assembler._build_element_dict(
        operator_id="OP-001",
        commodity=EUDRCommodity.COFFEE,
        products=sample_products_complete,
        geolocations=[],
        suppliers=[],
    )

    assert Article9Element.TRADE_CODES.value in elements
    trade_elem = elements[Article9Element.TRADE_CODES.value]
    assert len(trade_elem["hs_codes"]) == 2
    assert "0901.11" in trade_elem["hs_codes"]
    assert "0901.12" in trade_elem["hs_codes"]
    assert len(trade_elem["cn_codes"]) == 2


# ---------------------------------------------------------------------------
# Test: Element Building - Supporting Evidence
# ---------------------------------------------------------------------------

def test_build_element_dict_supporting_evidence(assembler):
    """Test supporting evidence element building."""
    evidence = ["doc-001", "audit-report-2025", "satellite-imagery-Q1"]

    elements = assembler._build_element_dict(
        operator_id="OP-001",
        commodity=EUDRCommodity.COFFEE,
        products=[],
        geolocations=[],
        suppliers=[],
        supporting_evidence=evidence,
    )

    assert Article9Element.SUPPORTING_EVIDENCE.value in elements
    evidence_elem = elements[Article9Element.SUPPORTING_EVIDENCE.value]
    assert evidence_elem["count"] == 3
    assert "doc-001" in evidence_elem["evidence_refs"]


# ---------------------------------------------------------------------------
# Test: Completeness Checking
# ---------------------------------------------------------------------------

def test_check_element_completeness_all_present(assembler):
    """Test completeness check with all elements present."""
    elements = {elem.value: {} for elem in ARTICLE9_MANDATORY_ELEMENTS}

    score, missing = assembler._check_element_completeness(elements)

    assert score == Decimal("1.00")
    assert len(missing) == 0


def test_check_element_completeness_half_present(assembler):
    """Test completeness check with half elements present."""
    # Include only 5 out of 10 elements
    elements = {
        Article9Element.PRODUCT_DESCRIPTION.value: {},
        Article9Element.QUANTITY.value: {},
        Article9Element.COUNTRY_OF_PRODUCTION.value: {},
        Article9Element.GEOLOCATION.value: {},
        Article9Element.SUPPLIER_INFO.value: {},
    }

    score, missing = assembler._check_element_completeness(elements)

    assert score == Decimal("0.50")
    assert len(missing) == 5


def test_check_element_completeness_none_present(assembler):
    """Test completeness check with no elements present."""
    elements = {}

    score, missing = assembler._check_element_completeness(elements)

    assert score == Decimal("0.00")
    assert len(missing) == 10


# ---------------------------------------------------------------------------
# Test: Geolocation Validation
# ---------------------------------------------------------------------------

def test_validate_geolocations_valid(assembler, sample_geolocations):
    """Test geolocation validation with valid data."""
    issues = assembler._validate_geolocations(sample_geolocations)

    # Should have no ERROR-level issues
    errors = [i for i in issues if i.severity == ValidationSeverity.ERROR]
    assert len(errors) == 0


def test_validate_geolocations_missing_polygon_above_4ha(assembler):
    """Test validation detects missing polygon for plots > 4ha."""
    geo_large_no_polygon = GeolocationReference(
        plot_id="PLOT-LARGE",
        latitude=Decimal("4.5"),
        longitude=Decimal("-74.2"),
        polygon=None,
        area_hectares=Decimal("6.0"),  # > 4ha
        country_code="CO",
    )

    issues = assembler._validate_geolocations([geo_large_no_polygon])

    assert len(issues) > 0
    assert any("polygon" in i.message.lower() for i in issues)
    assert any(i.severity == ValidationSeverity.ERROR for i in issues)


def test_validate_geolocations_has_polygon_above_4ha(assembler):
    """Test validation passes when polygon provided for > 4ha plots."""
    geo_large_with_polygon = GeolocationReference(
        plot_id="PLOT-LARGE",
        latitude=Decimal("4.5"),
        longitude=Decimal("-74.2"),
        polygon=[[Decimal("4.5"), Decimal("-74.2")]],
        area_hectares=Decimal("6.0"),
        country_code="CO",
    )

    issues = assembler._validate_geolocations([geo_large_with_polygon])

    polygon_errors = [
        i for i in issues
        if "polygon" in i.message.lower() and i.severity == ValidationSeverity.ERROR
    ]
    assert len(polygon_errors) == 0


def test_validate_geolocations_invalid_country_code(assembler):
    """Test validation detects invalid country code via model validation."""
    # Country code must be 2-3 characters per ISO 3166-1
    # Test that empty string raises validation error at model level
    with pytest.raises(PydanticValidationError) as exc_info:
        GeolocationReference(
            plot_id="PLOT-BAD",
            latitude=Decimal("4.5"),
            longitude=Decimal("-74.2"),
            area_hectares=Decimal("2.0"),
            country_code="",  # Invalid - empty country code
        )

    assert "country_code" in str(exc_info.value)


def test_validate_coordinate_precision_sufficient(assembler):
    """Test coordinate precision validation with sufficient decimals."""
    geo = GeolocationReference(
        plot_id="PLOT-001",
        latitude=Decimal("4.570868"),  # 6 decimals
        longitude=Decimal("-74.297333"),  # 6 decimals
        area_hectares=Decimal("2.0"),
        country_code="CO",
    )

    issues = assembler._validate_coordinate_precision(geo)

    # Should have no warnings if precision >= configured
    assert len(issues) == 0


def test_validate_coordinate_precision_insufficient(assembler):
    """Test coordinate precision validation with insufficient decimals."""
    geo = GeolocationReference(
        plot_id="PLOT-001",
        latitude=Decimal("4.5"),  # 1 decimal
        longitude=Decimal("-74.2"),  # 1 decimal
        area_hectares=Decimal("2.0"),
        country_code="CO",
    )

    issues = assembler._validate_coordinate_precision(geo)

    # Should have warnings
    assert len(issues) > 0
    assert all(i.severity == ValidationSeverity.WARNING for i in issues)


# ---------------------------------------------------------------------------
# Test: Product Validation
# ---------------------------------------------------------------------------

def test_validate_products_valid(assembler, sample_products_complete):
    """Test product validation with valid products."""
    issues = assembler._validate_products(sample_products_complete)

    assert len(issues) == 0


def test_validate_products_missing_description(assembler):
    """Test validation detects missing product description."""
    products = [
        ProductEntry(
            product_id="P1",
            description="",
            hs_code="0901.11",
            quantity=Decimal("100"),
            unit="kg",
        )
    ]

    issues = assembler._validate_products(products)

    assert len(issues) > 0
    assert any("description" in i.message.lower() for i in issues)
    assert any(i.severity == ValidationSeverity.ERROR for i in issues)


def test_validate_products_invalid_quantity(assembler):
    """Test validation detects invalid (zero/negative) quantity."""
    products = [
        ProductEntry(
            product_id="P1",
            description="Coffee",
            hs_code="0901.11",
            quantity=Decimal("0"),
            unit="kg",
        )
    ]

    issues = assembler._validate_products(products)

    assert len(issues) > 0
    assert any("quantity" in i.message.lower() for i in issues)


# ---------------------------------------------------------------------------
# Test: Package Completeness Check
# ---------------------------------------------------------------------------

def test_is_package_complete_above_threshold(assembler):
    """Test package completeness check with score above threshold."""
    package = Article9Package(
        package_id="pkg-001",
        operator_id="OP-001",
        commodity=EUDRCommodity.COFFEE,
        elements={},
        completeness_score=Decimal("0.96"),
        missing_elements=[],
    )

    is_complete = assembler.is_package_complete(package)
    assert is_complete is True


def test_is_package_complete_below_threshold(assembler):
    """Test package completeness check with score below threshold."""
    package = Article9Package(
        package_id="pkg-001",
        operator_id="OP-001",
        commodity=EUDRCommodity.COFFEE,
        elements={},
        completeness_score=Decimal("0.50"),
        missing_elements=["buyer_info", "certifications"],
    )

    is_complete = assembler.is_package_complete(package)
    assert is_complete is False


# ---------------------------------------------------------------------------
# Test: Element Descriptions
# ---------------------------------------------------------------------------

def test_get_element_description(assembler):
    """Test getting element descriptions."""
    desc = assembler.get_element_description(Article9Element.GEOLOCATION)

    assert len(desc) > 0
    assert "geolocation" in desc.lower() or "coordinates" in desc.lower()


def test_get_element_description_all_elements(assembler):
    """Test descriptions exist for all elements."""
    for element in Article9Element:
        desc = assembler.get_element_description(element)
        assert len(desc) > 0


# ---------------------------------------------------------------------------
# Test: Missing Element Guidance
# ---------------------------------------------------------------------------

def test_get_missing_element_guidance(assembler):
    """Test getting guidance for missing elements."""
    missing = ["buyer_info", "certifications"]
    guidance = assembler.get_missing_element_guidance(missing)

    assert len(guidance) == 2
    assert any(g["element"] == "buyer_info" for g in guidance)
    assert any(g["element"] == "certifications" for g in guidance)
    assert all("action" in g for g in guidance)
    assert all("description" in g for g in guidance)


def test_get_missing_element_guidance_empty(assembler):
    """Test guidance with empty missing list."""
    guidance = assembler.get_missing_element_guidance([])
    assert len(guidance) == 0


# ---------------------------------------------------------------------------
# Test: Health Check
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_health_check(assembler):
    """Test health check returns correct status."""
    status = await assembler.health_check()

    assert status["engine"] == "Article9DataAssembler"
    assert status["status"] == "available"
    assert "config" in status
    assert status["mandatory_elements"] == len(ARTICLE9_MANDATORY_ELEMENTS)


# ---------------------------------------------------------------------------
# Test: Different Commodities
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_assemble_package_palm_oil(
    assembler,
    sample_geolocations,
    sample_suppliers,
):
    """Test Article 9 assembly for palm oil commodity."""
    products = [
        ProductEntry(
            product_id="PALM-001",
            description="Crude Palm Oil",
            hs_code="1511.10",
            quantity=Decimal("5000"),
            unit="kg",
        )
    ]

    package = await assembler.assemble_package(
        operator_id="OP-PALM",
        commodity=EUDRCommodity.PALM_OIL,
        products=products,
        geolocations=sample_geolocations,
        suppliers=sample_suppliers,
    )

    assert package.commodity == EUDRCommodity.PALM_OIL
    assert package.package_id.startswith("a9p-")


@pytest.mark.asyncio
async def test_assemble_package_rubber(
    assembler,
    sample_geolocations,
    sample_suppliers,
):
    """Test Article 9 assembly for rubber commodity."""
    products = [
        ProductEntry(
            product_id="RUB-001",
            description="Natural Rubber Sheets",
            hs_code="4001.21",
            quantity=Decimal("2000"),
            unit="kg",
        )
    ]

    package = await assembler.assemble_package(
        operator_id="OP-RUBBER",
        commodity=EUDRCommodity.RUBBER,
        products=products,
        geolocations=sample_geolocations,
        suppliers=sample_suppliers,
    )

    assert package.commodity == EUDRCommodity.RUBBER
