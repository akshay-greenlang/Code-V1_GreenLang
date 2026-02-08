# -*- coding: utf-8 -*-
"""
Pytest Fixtures for ERP/Finance Connector Service Unit Tests (AGENT-DATA-003)
=============================================================================

Provides shared fixtures for testing the ERP connector config, models,
connection manager, spend extractor, purchase order engine, inventory
tracker, scope 3 mapper, emissions calculator, currency converter,
provenance tracker, metrics, setup facade, and API router.

All tests are self-contained with no external dependencies.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import os
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Environment cleanup fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_erp_connector_env(monkeypatch):
    """Remove any GL_ERP_CONNECTOR_ env vars between tests."""
    prefix = "GL_ERP_CONNECTOR_"
    for key in list(os.environ.keys()):
        if key.startswith(prefix):
            monkeypatch.delenv(key, raising=False)


# ---------------------------------------------------------------------------
# Sample Vendor Data Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_vendors() -> List[Dict[str, Any]]:
    """Realistic ERP vendor data (10 vendors with SAP/Oracle codes)."""
    return [
        {"vendor_id": "V-SAP-10001", "vendor_name": "EcoSteel GmbH", "category": "raw_materials", "country": "DE", "erp_code": "LFA1-10001"},
        {"vendor_id": "V-SAP-10002", "vendor_name": "GreenLogistics AG", "category": "transportation", "country": "NL", "erp_code": "LFA1-10002"},
        {"vendor_id": "V-ORA-20001", "vendor_name": "CleanEnergy Corp", "category": "energy", "country": "US", "erp_code": "AP_SUPPLIERS-20001"},
        {"vendor_id": "V-ORA-20002", "vendor_name": "SustainPack Ltd", "category": "packaging", "country": "UK", "erp_code": "AP_SUPPLIERS-20002"},
        {"vendor_id": "V-SAP-10003", "vendor_name": "BioChemicals SA", "category": "chemicals", "country": "FR", "erp_code": "LFA1-10003"},
        {"vendor_id": "V-SAP-10004", "vendor_name": "CircularWaste Inc", "category": "waste_management", "country": "US", "erp_code": "LFA1-10004"},
        {"vendor_id": "V-NET-30001", "vendor_name": "CloudIT Services", "category": "it_services", "country": "IN", "erp_code": "NS-VENDOR-30001"},
        {"vendor_id": "V-DYN-40001", "vendor_name": "FacilityMgmt Co", "category": "facilities", "country": "US", "erp_code": "DYN-40001"},
        {"vendor_id": "V-SAP-10005", "vendor_name": "RenewableParts Pty", "category": "raw_materials", "country": "AU", "erp_code": "LFA1-10005"},
        {"vendor_id": "V-ORA-20003", "vendor_name": "BusinessTravel GmbH", "category": "travel", "country": "DE", "erp_code": "AP_SUPPLIERS-20003"},
    ]


# ---------------------------------------------------------------------------
# Sample Spend Records Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_spend_records() -> List[Dict[str, Any]]:
    """Realistic spend records from ERP system."""
    return [
        {"record_id": "SPD-001", "vendor_id": "V-SAP-10001", "amount": 125000.0, "currency": "EUR", "date": "2025-06-15", "category": "raw_materials", "description": "Hot-rolled steel coils", "cost_center": "CC-MFG-001"},
        {"record_id": "SPD-002", "vendor_id": "V-SAP-10002", "amount": 45000.0, "currency": "EUR", "date": "2025-06-16", "category": "transportation", "description": "Freight forwarding Q2", "cost_center": "CC-LOG-001"},
        {"record_id": "SPD-003", "vendor_id": "V-ORA-20001", "amount": 78500.0, "currency": "USD", "date": "2025-06-17", "category": "energy", "description": "Renewable energy certificates", "cost_center": "CC-OPS-001"},
        {"record_id": "SPD-004", "vendor_id": "V-ORA-20002", "amount": 12300.0, "currency": "GBP", "date": "2025-06-18", "category": "packaging", "description": "Biodegradable packaging", "cost_center": "CC-MFG-002"},
        {"record_id": "SPD-005", "vendor_id": "V-SAP-10003", "amount": 34200.0, "currency": "EUR", "date": "2025-06-19", "category": "chemicals", "description": "Bio-based solvents", "cost_center": "CC-MFG-003"},
    ]


# ---------------------------------------------------------------------------
# Sample Purchase Orders Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_purchase_orders() -> List[Dict[str, Any]]:
    """Realistic purchase orders with line items."""
    return [
        {
            "po_number": "PO-2025-0001",
            "vendor_id": "V-SAP-10001",
            "status": "open",
            "total_value": 250000.0,
            "currency": "EUR",
            "created_date": "2025-06-01",
            "delivery_date": "2025-07-15",
            "line_items": [
                {"item_number": 10, "material": "STEEL-HR-001", "quantity": 50.0, "unit": "tonnes", "unit_price": 4000.0, "amount": 200000.0},
                {"item_number": 20, "material": "STEEL-CR-002", "quantity": 10.0, "unit": "tonnes", "unit_price": 5000.0, "amount": 50000.0},
            ],
        },
        {
            "po_number": "PO-2025-0002",
            "vendor_id": "V-ORA-20001",
            "status": "closed",
            "total_value": 78500.0,
            "currency": "USD",
            "created_date": "2025-05-15",
            "delivery_date": "2025-06-30",
            "line_items": [
                {"item_number": 10, "material": "REC-SOLAR-001", "quantity": 500.0, "unit": "MWh", "unit_price": 157.0, "amount": 78500.0},
            ],
        },
    ]


# ---------------------------------------------------------------------------
# Sample Inventory Items Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_inventory_items() -> List[Dict[str, Any]]:
    """Realistic inventory items from ERP warehouse."""
    return [
        {"item_id": "INV-001", "material": "STEEL-HR-001", "description": "Hot-rolled steel coil", "warehouse": "WH-MAIN", "quantity": 120.0, "unit": "tonnes", "unit_cost": 4000.0, "material_group": "raw_materials"},
        {"item_id": "INV-002", "material": "PACK-BIO-001", "description": "Biodegradable packaging rolls", "warehouse": "WH-MAIN", "quantity": 5000.0, "unit": "rolls", "unit_cost": 2.50, "material_group": "packaging"},
        {"item_id": "INV-003", "material": "CHEM-SOL-001", "description": "Bio-based solvent", "warehouse": "WH-CHEM", "quantity": 800.0, "unit": "liters", "unit_cost": 42.75, "material_group": "chemicals"},
        {"item_id": "INV-004", "material": "STEEL-CR-002", "description": "Cold-rolled steel sheet", "warehouse": "WH-MAIN", "quantity": 45.0, "unit": "tonnes", "unit_cost": 5000.0, "material_group": "raw_materials"},
        {"item_id": "INV-005", "material": "ELEC-COMP-001", "description": "Electronic components", "warehouse": "WH-ELEC", "quantity": 10000.0, "unit": "pcs", "unit_cost": 1.20, "material_group": "components"},
    ]


# ---------------------------------------------------------------------------
# Mock Prometheus Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_prometheus():
    """Mock prometheus_client for metrics testing."""
    mock_counter = MagicMock()
    mock_counter.labels.return_value = mock_counter
    mock_histogram = MagicMock()
    mock_histogram.labels.return_value = mock_histogram
    mock_gauge = MagicMock()
    mock_gauge.labels.return_value = mock_gauge

    mock_prom = MagicMock()
    mock_prom.Counter.return_value = mock_counter
    mock_prom.Histogram.return_value = mock_histogram
    mock_prom.Gauge.return_value = mock_gauge
    mock_prom.generate_latest.return_value = (
        b"# HELP test_metric\n# TYPE test_metric counter\n"
    )
    return mock_prom
