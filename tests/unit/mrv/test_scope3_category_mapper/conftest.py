# -*- coding: utf-8 -*-
"""
Shared fixtures for Scope 3 Category Mapper tests (AGENT-MRV-029).

Provides comprehensive test fixtures for all test modules:
- Spend records (single and batch across all 15 categories)
- Purchase order, BOM, travel, waste, lease, logistics, energy records
- Investment and franchise records
- Organization contexts (manufacturer, financial, retailer)
- Classification results (single and batch for all 15 categories)
- Completeness reports and compliance assessments
- Engine lifecycle management (singleton resets)
- Mock engine and config fixtures

Author: GL-TestEngineer
Date: March 2026
"""

import pytest
from decimal import Decimal
from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock


# ============================================================================
# ENGINE LIFECYCLE FIXTURES
# ============================================================================


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset all singleton engines before and after each test."""
    # Reset completeness screener
    try:
        from greenlang.agents.mrv.scope3_category_mapper.completeness_screener import (
            CompletenessScreenerEngine,
        )
        CompletenessScreenerEngine.reset_instance()
    except (ImportError, AttributeError):
        pass
    # Reset config singleton
    try:
        from greenlang.agents.mrv.scope3_category_mapper.config import reset_config
        reset_config()
    except (ImportError, AttributeError):
        pass
    yield
    try:
        from greenlang.agents.mrv.scope3_category_mapper.completeness_screener import (
            CompletenessScreenerEngine,
        )
        CompletenessScreenerEngine.reset_instance()
    except (ImportError, AttributeError):
        pass
    try:
        from greenlang.agents.mrv.scope3_category_mapper.config import reset_config
        reset_config()
    except (ImportError, AttributeError):
        pass


# ============================================================================
# SPEND RECORD FIXTURES
# ============================================================================


@pytest.fixture
def sample_spend_record() -> Dict[str, Any]:
    """Single spend record with amount, currency, GL account, NAICS code."""
    return {
        "record_id": "SPD-001",
        "amount": Decimal("12500.00"),
        "currency": "USD",
        "gl_account": "5000",
        "naics_code": "331",
        "description": "Raw steel purchase for manufacturing",
        "supplier_name": "US Steel Corp",
        "source_type": "purchase_order",
        "posting_date": "2025-03-15",
    }


@pytest.fixture
def sample_spend_records_batch() -> List[Dict[str, Any]]:
    """
    Twenty spend records covering many Scope 3 categories.

    Includes records that should classify to Cat 1-15 to validate
    broad coverage of the classification engine.
    """
    records = [
        # Cat 1 - Purchased Goods & Services
        {"record_id": "B-001", "description": "Raw materials purchase - steel",
         "amount": Decimal("50000.00"), "currency": "USD",
         "naics_code": "331", "gl_account": "5000",
         "supplier_name": "SteelCo", "source_type": "purchase_order"},
        # Cat 1 - Purchased Goods (services)
        {"record_id": "B-002", "description": "Legal advisory fees",
         "amount": Decimal("9000.00"), "currency": "USD",
         "naics_code": "5411", "gl_account": "6300",
         "supplier_name": "LawFirm LLP", "source_type": "invoice"},
        # Cat 2 - Capital Goods
        {"record_id": "B-003", "description": "CNC machine acquisition",
         "amount": Decimal("250000.00"), "currency": "USD",
         "naics_code": "333", "gl_account": "7000",
         "supplier_name": "MachineryCo", "source_type": "purchase_order"},
        # Cat 3 - Fuel & Energy Activities
        {"record_id": "B-004", "description": "Electricity bill Jan-Mar 2025",
         "amount": Decimal("6000.00"), "currency": "USD",
         "naics_code": "2211", "gl_account": "6600",
         "supplier_name": "PowerGrid", "source_type": "utility_bill"},
        # Cat 4 - Upstream Transport
        {"record_id": "B-005", "description": "Inbound freight charges Q1",
         "amount": Decimal("12500.00"), "currency": "USD",
         "naics_code": "484", "gl_account": "5300",
         "supplier_name": "ABC Logistics", "source_type": "invoice"},
        # Cat 5 - Waste Generated
        {"record_id": "B-006", "description": "Hazardous waste disposal",
         "amount": Decimal("3000.00"), "currency": "USD",
         "naics_code": "562", "gl_account": "6550",
         "supplier_name": "WasteMgmt", "source_type": "invoice"},
        # Cat 6 - Business Travel (air)
        {"record_id": "B-007", "description": "Employee airfare SFO-NYC",
         "amount": Decimal("3200.00"), "currency": "USD",
         "naics_code": "481", "gl_account": "6410",
         "supplier_name": "Delta Airlines", "source_type": "expense_report"},
        # Cat 6 - Business Travel (hotel)
        {"record_id": "B-008", "description": "Hotel accommodation NYC",
         "amount": Decimal("1800.00"), "currency": "USD",
         "naics_code": "721", "gl_account": "6420",
         "supplier_name": "Marriott", "source_type": "expense_report"},
        # Cat 7 - Employee Commuting
        {"record_id": "B-009", "description": "Commuter rail pass subsidy",
         "amount": Decimal("600.00"), "currency": "USD",
         "naics_code": None, "gl_account": "8400",
         "supplier_name": "MetroNorth", "source_type": "payroll"},
        # Cat 8 - Upstream Leased Assets
        {"record_id": "B-010", "description": "Office rent Q1 2025",
         "amount": Decimal("15000.00"), "currency": "USD",
         "naics_code": "531", "gl_account": "6700",
         "supplier_name": "RealEstate LLC", "source_type": "lease"},
        # Cat 9 - Downstream Transport
        {"record_id": "B-011", "description": "Outbound freight to customer",
         "amount": Decimal("7500.00"), "currency": "USD",
         "naics_code": "484", "gl_account": "8100",
         "supplier_name": "FedEx", "source_type": "invoice"},
        # Cat 10 - Processing of Sold Products
        {"record_id": "B-012", "description": "Toll processing fee",
         "amount": Decimal("4500.00"), "currency": "USD",
         "naics_code": None, "gl_account": "8050",
         "supplier_name": "ProcessorCo", "source_type": "invoice"},
        # Cat 11 - Use of Sold Products
        {"record_id": "B-013", "description": "Product energy use analysis",
         "amount": Decimal("2000.00"), "currency": "USD",
         "naics_code": None, "gl_account": "8060",
         "supplier_name": "TestLab", "source_type": "invoice"},
        # Cat 12 - End-of-Life Treatment
        {"record_id": "B-014", "description": "Recycling service fee",
         "amount": Decimal("800.00"), "currency": "USD",
         "naics_code": None, "gl_account": "8070",
         "supplier_name": "GreenRecycle", "source_type": "invoice"},
        # Cat 13 - Downstream Leased Assets
        {"record_id": "B-015", "description": "Copier lease payment",
         "amount": Decimal("500.00"), "currency": "USD",
         "naics_code": "532", "gl_account": "8080",
         "supplier_name": "Xerox", "source_type": "lease"},
        # Cat 14 - Franchises
        {"record_id": "B-016", "description": "Franchise royalty payment",
         "amount": Decimal("5000.00"), "currency": "USD",
         "naics_code": None, "gl_account": "8200",
         "supplier_name": "BrandCo", "source_type": "invoice"},
        # Cat 15 - Investments
        {"record_id": "B-017", "description": "Bond investment Q1",
         "amount": Decimal("100000.00"), "currency": "USD",
         "naics_code": "523", "gl_account": "8300",
         "supplier_name": "Goldman Sachs", "source_type": "journal"},
        # Additional Cat 1 variants
        {"record_id": "B-018", "description": "Chemical reagents purchase",
         "amount": Decimal("4500.00"), "currency": "USD",
         "naics_code": "325", "gl_account": "5000",
         "supplier_name": "LabChem", "source_type": "purchase_order"},
        # Cat 2 - Capital (fleet)
        {"record_id": "B-019", "description": "Fleet vehicle acquisition",
         "amount": Decimal("45000.00"), "currency": "USD",
         "naics_code": "336", "gl_account": "7000",
         "supplier_name": "Ford Motor", "source_type": "purchase_order"},
        # Cat 7 - Employee Commuting (shuttle)
        {"record_id": "B-020", "description": "Employee shuttle service",
         "amount": Decimal("1200.00"), "currency": "USD",
         "naics_code": None, "gl_account": "8400",
         "supplier_name": "TransitCo", "source_type": "payroll"},
    ]
    return records


# ============================================================================
# PURCHASE ORDER FIXTURE
# ============================================================================


@pytest.fixture
def sample_purchase_order() -> Dict[str, Any]:
    """Purchase order with line items spanning multiple categories."""
    return {
        "po_number": "PO-2025-001234",
        "supplier_name": "MegaSupply Corp",
        "supplier_country": "US",
        "po_date": "2025-03-01",
        "currency": "USD",
        "total_amount": Decimal("175000.00"),
        "line_items": [
            {
                "line_number": 1,
                "description": "Steel plate 10mm",
                "quantity": 500,
                "unit": "kg",
                "unit_price": Decimal("2.50"),
                "amount": Decimal("1250.00"),
                "naics_code": "331",
            },
            {
                "line_number": 2,
                "description": "Industrial pump assembly",
                "quantity": 1,
                "unit": "each",
                "unit_price": Decimal("75000.00"),
                "amount": Decimal("75000.00"),
                "naics_code": "333",
            },
            {
                "line_number": 3,
                "description": "Safety equipment",
                "quantity": 50,
                "unit": "sets",
                "unit_price": Decimal("120.00"),
                "amount": Decimal("6000.00"),
                "naics_code": "339",
            },
        ],
    }


# ============================================================================
# BOM RECORD FIXTURE
# ============================================================================


@pytest.fixture
def sample_bom_record() -> Dict[str, Any]:
    """Bill of Materials record for a manufactured product."""
    return {
        "bom_id": "BOM-2025-00456",
        "product_name": "Industrial Pump Model X-200",
        "product_code": "PUMP-X200",
        "components": [
            {
                "material": "Stainless steel casing",
                "weight_kg": Decimal("25.0"),
                "naics_code": "331",
                "supplier": "SteelWorks Inc",
                "origin_country": "US",
            },
            {
                "material": "Electric motor 5kW",
                "weight_kg": Decimal("12.0"),
                "naics_code": "335",
                "supplier": "MotorTech Ltd",
                "origin_country": "DE",
            },
            {
                "material": "Rubber seals",
                "weight_kg": Decimal("0.5"),
                "naics_code": "326",
                "supplier": "RubberCo",
                "origin_country": "JP",
            },
        ],
        "total_weight_kg": Decimal("37.5"),
    }


# ============================================================================
# TRAVEL RECORD FIXTURE
# ============================================================================


@pytest.fixture
def sample_travel_record() -> Dict[str, Any]:
    """Business travel record -- air travel segment."""
    return {
        "travel_id": "TRV-2025-00789",
        "travel_type": "air",
        "traveler_name": "Jane Smith",
        "origin": "SFO",
        "destination": "JFK",
        "departure_date": "2025-04-10",
        "return_date": "2025-04-14",
        "cabin_class": "economy",
        "distance_km": Decimal("4150.0"),
        "airline": "United Airlines",
        "booking_cost": Decimal("850.00"),
        "currency": "USD",
        "hotel_nights": 4,
        "hotel_city": "New York",
        "hotel_cost_per_night": Decimal("225.00"),
        "purpose": "Client meeting",
    }


# ============================================================================
# WASTE RECORD FIXTURE
# ============================================================================


@pytest.fixture
def sample_waste_record() -> Dict[str, Any]:
    """Waste disposal record for operations waste."""
    return {
        "waste_id": "WST-2025-00123",
        "waste_type": "general_industrial",
        "waste_category": "non_hazardous",
        "weight_tonnes": Decimal("5.20"),
        "treatment_method": "landfill",
        "disposal_date": "2025-03-20",
        "facility_name": "Metro Landfill Inc",
        "manifest_number": "MAN-2025-4567",
        "origin_site": "Plant A - Chicago",
        "cost": Decimal("1250.00"),
        "currency": "USD",
    }


# ============================================================================
# LEASE RECORD FIXTURE
# ============================================================================


@pytest.fixture
def sample_lease_record() -> Dict[str, Any]:
    """Upstream lease record for office space."""
    return {
        "lease_id": "LSE-2025-00456",
        "asset_type": "building",
        "asset_description": "Office space - Floor 12, 500 Market St",
        "lessor_name": "RealEstate Holdings LLC",
        "lease_start_date": "2024-01-01",
        "lease_end_date": "2028-12-31",
        "lease_duration_months": 60,
        "monthly_payment": Decimal("15000.00"),
        "annual_payment": Decimal("180000.00"),
        "currency": "USD",
        "floor_area_m2": Decimal("500.0"),
        "building_type": "commercial_office",
        "location_city": "San Francisco",
        "location_country": "US",
        "energy_included": False,
        "lease_classification": "operating",
    }


# ============================================================================
# LOGISTICS RECORD FIXTURE
# ============================================================================


@pytest.fixture
def sample_logistics_record() -> Dict[str, Any]:
    """Logistics record with Incoterm for upstream/downstream routing."""
    return {
        "logistics_id": "LOG-2025-00789",
        "shipment_type": "freight",
        "transport_mode": "road",
        "origin": "Chicago, IL",
        "destination": "Detroit, MI",
        "distance_km": Decimal("450.0"),
        "weight_tonnes": Decimal("8.50"),
        "tonne_km": Decimal("3825.00"),
        "carrier": "ABC Trucking",
        "incoterm": "FOB",
        "direction": "inbound",
        "shipping_date": "2025-03-18",
        "delivery_date": "2025-03-19",
        "cost": Decimal("2800.00"),
        "currency": "USD",
        "vehicle_type": "heavy_truck",
    }


# ============================================================================
# ENERGY RECORD FIXTURE
# ============================================================================


@pytest.fixture
def sample_energy_record() -> Dict[str, Any]:
    """Energy consumption record for fuel and energy activities."""
    return {
        "energy_id": "ENR-2025-00345",
        "energy_type": "electricity",
        "source": "grid",
        "quantity": Decimal("250000.0"),
        "unit": "kWh",
        "period_start": "2025-01-01",
        "period_end": "2025-03-31",
        "supplier": "Pacific Gas & Electric",
        "grid_region": "CAMX",
        "cost": Decimal("32500.00"),
        "currency": "USD",
        "renewable_pct": Decimal("35.0"),
        "facility": "Plant A - San Francisco",
    }


# ============================================================================
# INVESTMENT RECORD FIXTURE
# ============================================================================


@pytest.fixture
def sample_investment_record() -> Dict[str, Any]:
    """Investment record for Category 15."""
    return {
        "investment_id": "INV-2025-00678",
        "asset_class": "listed_equity",
        "investee_name": "TechCorp Inc.",
        "isin": "US1234567890",
        "outstanding_amount": Decimal("50000000.00"),
        "evic": Decimal("500000000000.00"),
        "investee_scope1": Decimal("15000.0"),
        "investee_scope2": Decimal("8000.0"),
        "sector": "information_technology",
        "country": "US",
        "currency": "USD",
        "reporting_year": 2024,
        "pcaf_quality_score": 2,
    }


# ============================================================================
# FRANCHISE RECORD FIXTURE
# ============================================================================


@pytest.fixture
def sample_franchise_record() -> Dict[str, Any]:
    """Franchise record for Category 14."""
    return {
        "franchise_id": "FRN-2025-00123",
        "franchise_name": "QuickBurger Unit #4521",
        "franchise_type": "quick_service_restaurant",
        "location_city": "Dallas",
        "location_state": "TX",
        "location_country": "US",
        "floor_area_m2": Decimal("200.0"),
        "annual_revenue": Decimal("1250000.00"),
        "annual_energy_kwh": Decimal("180000.0"),
        "annual_natural_gas_therms": Decimal("12000.0"),
        "employee_count": 25,
        "operating_hours_per_week": 112,
        "currency": "USD",
        "reporting_year": 2024,
    }


# ============================================================================
# ORGANIZATION CONTEXT FIXTURES
# ============================================================================


@pytest.fixture
def sample_manufacturer_context() -> Dict[str, Any]:
    """Organization context for a manufacturing company."""
    return {
        "org_id": "ORG-001",
        "org_name": "AcmeManufacturing Inc.",
        "company_type": "manufacturer",
        "industry_sector": "industrials",
        "naics_primary": "332",
        "employee_count": 5000,
        "annual_revenue": Decimal("750000000.00"),
        "currency": "USD",
        "headquarters_country": "US",
        "consolidation_approach": "operational_control",
        "reporting_year": 2024,
        "facilities": [
            {"name": "Plant A", "city": "Chicago", "state": "IL", "type": "manufacturing"},
            {"name": "Plant B", "city": "Detroit", "state": "MI", "type": "manufacturing"},
            {"name": "HQ", "city": "New York", "state": "NY", "type": "office"},
        ],
    }


@pytest.fixture
def sample_financial_context() -> Dict[str, Any]:
    """Organization context for a financial institution."""
    return {
        "org_id": "ORG-002",
        "org_name": "GreenBank Financial",
        "company_type": "financial",
        "industry_sector": "financials",
        "naics_primary": "522",
        "employee_count": 12000,
        "annual_revenue": Decimal("8500000000.00"),
        "currency": "USD",
        "headquarters_country": "US",
        "consolidation_approach": "financial_control",
        "reporting_year": 2024,
        "aum": Decimal("150000000000.00"),
        "facilities": [
            {"name": "HQ", "city": "New York", "state": "NY", "type": "office"},
            {"name": "Branch Network", "city": "Various", "state": "Multi", "type": "retail"},
        ],
    }


@pytest.fixture
def sample_retailer_context() -> Dict[str, Any]:
    """Organization context for a retail company."""
    return {
        "org_id": "ORG-003",
        "org_name": "MegaMart Retail",
        "company_type": "retailer",
        "industry_sector": "consumer_staples",
        "naics_primary": "452",
        "employee_count": 85000,
        "annual_revenue": Decimal("42000000000.00"),
        "currency": "USD",
        "headquarters_country": "US",
        "consolidation_approach": "operational_control",
        "reporting_year": 2024,
        "store_count": 2500,
        "facilities": [
            {"name": "HQ", "city": "Bentonville", "state": "AR", "type": "office"},
            {"name": "DC East", "city": "Atlanta", "state": "GA", "type": "distribution"},
            {"name": "DC West", "city": "Phoenix", "state": "AZ", "type": "distribution"},
        ],
    }


# ============================================================================
# CLASSIFICATION RESULT FIXTURES
# ============================================================================


@pytest.fixture
def sample_classification_result() -> Dict[str, Any]:
    """A single ClassificationResult-compatible dict for Cat 1."""
    return {
        "record_id": "SPD-001",
        "primary_category": "cat_1_purchased_goods",
        "category_number": 1,
        "category_name": "Purchased Goods & Services",
        "confidence": 0.92,
        "classification_method": "naics_lookup",
        "secondary_candidates": [
            {"category": "cat_2_capital_goods", "confidence": 0.15},
        ],
        "data_quality_tier": "tier_3",
        "provenance_hash": "a1b2c3d4e5f6" + "0" * 52,
        "processing_time_ms": 2.3,
    }


@pytest.fixture
def sample_classification_results_batch() -> List[Dict[str, Any]]:
    """Fifteen classification results -- one per Scope 3 category."""
    categories = [
        ("cat_1_purchased_goods", 1, "Purchased Goods & Services", 0.95),
        ("cat_2_capital_goods", 2, "Capital Goods", 0.88),
        ("cat_3_fuel_energy", 3, "Fuel- and Energy-Related Activities", 0.91),
        ("cat_4_upstream_transport", 4, "Upstream Transportation & Distribution", 0.87),
        ("cat_5_waste", 5, "Waste Generated in Operations", 0.93),
        ("cat_6_business_travel", 6, "Business Travel", 0.96),
        ("cat_7_employee_commuting", 7, "Employee Commuting", 0.84),
        ("cat_8_upstream_leased", 8, "Upstream Leased Assets", 0.79),
        ("cat_9_downstream_transport", 9, "Downstream Transportation & Distribution", 0.85),
        ("cat_10_processing_sold", 10, "Processing of Sold Products", 0.77),
        ("cat_11_use_sold", 11, "Use of Sold Products", 0.82),
        ("cat_12_end_of_life", 12, "End-of-Life Treatment of Sold Products", 0.80),
        ("cat_13_downstream_leased", 13, "Downstream Leased Assets", 0.76),
        ("cat_14_franchises", 14, "Franchises", 0.90),
        ("cat_15_investments", 15, "Investments", 0.94),
    ]
    results = []
    for i, (cat_val, cat_num, cat_name, conf) in enumerate(categories):
        results.append({
            "record_id": f"CLF-{i+1:03d}",
            "primary_category": cat_val,
            "category_number": cat_num,
            "category_name": cat_name,
            "confidence": conf,
            "classification_method": "rule_based",
            "secondary_candidates": [],
            "data_quality_tier": "tier_3",
            "provenance_hash": f"{cat_num:02d}" + "a" * 62,
            "processing_time_ms": 1.5 + i * 0.1,
        })
    return results


# ============================================================================
# COMPLETENESS REPORT FIXTURE
# ============================================================================


@pytest.fixture
def sample_completeness_report() -> Dict[str, Any]:
    """A completeness report dict for a manufacturer with partial coverage."""
    return {
        "company_type": "manufacturer",
        "total_categories": 15,
        "categories_reported": 8,
        "categories_material": 8,
        "categories_material_reported": 5,
        "categories_relevant": 4,
        "categories_not_relevant": 3,
        "completeness_score": Decimal("62.50"),
        "gaps": [
            "Category 2 (Capital Goods) is material but has no reported data.",
            "Category 10 (Processing of Sold Products) is material but has no reported data.",
            "Category 12 (End-of-Life Treatment) is material but has no reported data.",
        ],
        "recommended_actions": [
            "[CRITICAL] Collect data for Category 2 (Capital Goods).",
            "[CRITICAL] Collect data for Category 10 (Processing of Sold Products).",
            "[CRITICAL] Collect data for Category 12 (End-of-Life Treatment).",
        ],
        "provenance_hash": "abcdef1234567890" * 4,
        "assessed_at": "2025-03-15T12:00:00+00:00",
        "processing_time_ms": 15.7,
    }


# ============================================================================
# MOCK CONFIG FIXTURE
# ============================================================================


@pytest.fixture
def mock_config() -> MagicMock:
    """Mock Scope3CategoryMapperConfig with default values."""
    config = MagicMock()
    config.general.enabled = True
    config.general.debug = False
    config.general.log_level = "INFO"
    config.general.agent_id = "GL-MRV-X-040"
    config.general.agent_component = "AGENT-MRV-029"
    config.general.version = "1.0.0"
    config.general.api_prefix = "/api/v1/scope3-mapper"
    config.general.max_batch_size = 5000
    config.general.default_gwp = "AR5"
    config.classification.min_confidence = 0.7
    config.classification.high_confidence_threshold = 0.9
    config.classification.max_candidates = 3
    config.classification.enable_naics_lookup = True
    config.classification.enable_ml_classifier = False
    config.boundary.default_consolidation = "operational_control"
    config.boundary.capex_threshold = Decimal("5000.00")
    config.completeness.materiality_threshold_pct = Decimal("1.00")
    config.completeness.benchmark_tolerance_pct = Decimal("10.00")
    config.double_counting.enabled = True
    config.double_counting.strict_mode = False
    config.compliance.enabled = True
    config.compliance.strict_mode = False
    config.compliance.get_frameworks.return_value = [
        "cdp", "csrd_esrs", "ghg_protocol", "iso_14064",
        "issb_s2", "sb_253", "sbti", "sec_climate",
    ]
    config.database.pool_size = 5
    config.database.table_prefix = "gl_scm_"
    config.metrics.prefix = "gl_scm_"
    config.provenance.chain_algorithm = "SHA-256"
    config.provenance.merkle_enabled = True
    config.cache.ttl_seconds = 3600
    config.routing.enabled = True
    config.routing.max_retries = 3
    return config
