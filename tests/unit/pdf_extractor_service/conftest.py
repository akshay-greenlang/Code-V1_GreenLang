# -*- coding: utf-8 -*-
"""
Pytest Fixtures for PDF & Invoice Extractor Service Unit Tests (AGENT-DATA-001)
================================================================================

Provides shared fixtures for testing the PDF extractor config, models,
document parser, OCR engine, field extractor, invoice/manifest processors,
document classifier, validation engine, provenance tracker, metrics,
setup facade, and API router.

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
def _clean_pdf_extractor_env(monkeypatch):
    """Remove any GL_PDF_EXTRACTOR_ env vars between tests."""
    prefix = "GL_PDF_EXTRACTOR_"
    for key in list(os.environ.keys()):
        if key.startswith(prefix):
            monkeypatch.delenv(key, raising=False)


# ---------------------------------------------------------------------------
# Sample Document Text Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_invoice_text() -> str:
    """Realistic invoice text for extraction tests."""
    return """
    INVOICE
    Invoice Number: INV-2025-001234
    Invoice Date: 2025-06-15
    Due Date: 2025-07-15
    PO Number: PO-98765

    Bill To:
    GreenCorp Ltd.
    123 Sustainability Drive
    London, UK EC1A 1BB

    Ship To:
    GreenCorp Warehouse
    456 Carbon Neutral Way
    Manchester, UK M1 2AB

    Vendor:
    EcoSupply Partners Inc.
    789 Green Avenue
    Berlin, Germany 10117

    Item       Description                  Qty    Unit Price    Amount
    ----       -----------                  ---    ----------    ------
    CARB-001   Carbon Offset Credits        100    $25.00        $2,500.00
    RENEW-002  Renewable Energy Certs       50     $15.50        $775.00
    CONS-003   Sustainability Consulting    40     $150.00       $6,000.00

    Subtotal:    $9,275.00
    Tax (20%):   $1,855.00
    Total:       $11,130.00

    Payment Terms: Net 30
    Bank: HSBC Business Account
    Account: 12345678
    Sort Code: 40-20-30
    """


@pytest.fixture
def sample_manifest_text() -> str:
    """Realistic shipping manifest / Bill of Lading text."""
    return """
    BILL OF LADING
    BOL Number: BOL-2025-56789
    Date: 2025-06-20

    Shipper:
    EcoMaterials GmbH
    Am Hafen 12
    Hamburg, Germany 20457

    Consignee:
    GreenCorp Ltd.
    123 Sustainability Drive
    London, UK EC1A 1BB

    Carrier:
    MaerskLine Container Shipping
    Vessel: MV Green Future
    Voyage: GF-2025-042

    Port of Loading: Hamburg, Germany (DEHAM)
    Port of Discharge: Felixstowe, UK (GBFXT)

    Container: MSKU1234567 (40ft HC)

    Cargo Details:
    Item    Description              Packages   Weight (kg)   Volume (m3)
    1       Recycled Steel Coils     20         18,500        12.5
    2       Solar Panel Components   15         4,200         8.3
    3       Bio-degradable Packaging 50         2,800         15.0

    Total Packages: 85
    Total Gross Weight: 25,500 kg
    Total Volume: 35.8 m3

    Freight Terms: CIF London
    """


@pytest.fixture
def sample_utility_bill_text() -> str:
    """Realistic utility bill text."""
    return """
    UTILITY BILL - ELECTRICITY
    Account Number: ELEC-2025-78901
    Billing Period: 2025-05-01 to 2025-05-31
    Statement Date: 2025-06-05

    Customer:
    GreenCorp Manufacturing Ltd.
    Industrial Park Unit 7
    Birmingham, UK B1 1AA

    Meter Number: E-MTR-445566
    Previous Reading: 145,230 kWh
    Current Reading: 152,780 kWh
    Consumption: 7,550 kWh

    Rate Breakdown:
    Peak (07:00-23:00):     5,200 kWh @ $0.18/kWh = $936.00
    Off-Peak (23:00-07:00): 2,350 kWh @ $0.09/kWh = $211.50

    Supply Charge:   $45.00
    Subtotal:        $1,192.50
    Tax (5%):        $59.63
    Total Due:       $1,252.13

    Due Date: 2025-06-25
    """


@pytest.fixture
def sample_receipt_text() -> str:
    """Simple receipt text for testing."""
    return """
    RECEIPT
    Receipt No: REC-2025-3456
    Date: 2025-06-10
    Vendor: Office Supply Co.
    Total: $127.45
    Payment: Visa ending 4242
    """


@pytest.fixture
def sample_purchase_order_text() -> str:
    """Realistic purchase order text."""
    return """
    PURCHASE ORDER
    PO Number: PO-2025-11223
    Date: 2025-06-01
    Delivery Date: 2025-06-30

    Buyer:
    GreenCorp Ltd.

    Supplier:
    EcoSupply Partners Inc.

    Item   Description              Qty   Unit Price   Total
    1      Carbon Offset Credits    200   $25.00       $5,000.00
    2      Renewable Energy Certs   100   $15.50       $1,550.00

    Total: $6,550.00
    """


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
