# -*- coding: utf-8 -*-
"""
Pytest Fixtures for Excel & CSV Normalizer Service Unit Tests (AGENT-DATA-002)
================================================================================

Provides shared fixtures for testing the Excel normalizer config, models,
excel parser, CSV parser, column mapper, data type detector, schema validator,
data quality scorer, transform engine, provenance tracker, metrics,
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
def _clean_excel_normalizer_env(monkeypatch):
    """Remove any GL_EXCEL_NORMALIZER_ env vars between tests."""
    prefix = "GL_EXCEL_NORMALIZER_"
    for key in list(os.environ.keys()):
        if key.startswith(prefix):
            monkeypatch.delenv(key, raising=False)


# ---------------------------------------------------------------------------
# Sample Excel Data Fixtures (list of lists)
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_excel_data() -> List[List[Any]]:
    """Realistic emissions reporting data in list-of-lists format."""
    return [
        ["Facility Name", "Reporting Year", "Scope 1 Emissions (tCO2e)", "Scope 2 Emissions (tCO2e)",
         "Energy Consumption (MWh)", "Fuel Type", "Country", "Verification Status"],
        ["GreenCorp London HQ", 2025, 1250.5, 890.3, 4500.0, "Natural Gas", "GB", "Verified"],
        ["GreenCorp Berlin Plant", 2025, 3400.0, 1200.8, 12000.0, "Coal", "DE", "Pending"],
        ["GreenCorp NYC Office", 2025, 450.2, 620.1, 2100.0, "Grid Electricity", "US", "Verified"],
        ["GreenCorp Tokyo Warehouse", 2025, 780.0, 510.0, 3300.0, "Diesel", "JP", "Not Verified"],
        ["GreenCorp Sydney Lab", 2025, None, 340.0, 1500.0, "Solar", "AU", "Verified"],
    ]


@pytest.fixture
def sample_csv_text() -> str:
    """Realistic CSV text for emissions data."""
    return (
        "facility_name,year,scope1_co2e_tonnes,scope2_co2e_tonnes,energy_mwh,fuel_type,country\n"
        "London HQ,2025,1250.5,890.3,4500.0,Natural Gas,GB\n"
        "Berlin Plant,2025,3400.0,1200.8,12000.0,Coal,DE\n"
        "NYC Office,2025,450.2,620.1,2100.0,Grid Electricity,US\n"
        "Tokyo Warehouse,2025,780.0,510.0,3300.0,Diesel,JP\n"
        "Sydney Lab,2025,,340.0,1500.0,Solar,AU\n"
    )


# ---------------------------------------------------------------------------
# Sample Header Fixtures by Category
# ---------------------------------------------------------------------------


@pytest.fixture
def energy_headers() -> List[str]:
    """Column headers for energy consumption data."""
    return [
        "Facility", "Year", "Electricity Consumption (kWh)", "Natural Gas (therms)",
        "Diesel Fuel (litres)", "Renewable Energy (%)", "Peak Demand (kW)",
        "Grid Emission Factor (kgCO2e/kWh)",
    ]


@pytest.fixture
def transport_headers() -> List[str]:
    """Column headers for transport and logistics data."""
    return [
        "Vehicle ID", "Route", "Distance (km)", "Fuel Used (litres)",
        "Fuel Type", "Cargo Weight (tonnes)", "Mode of Transport",
        "CO2 Emissions (kgCO2e)",
    ]


@pytest.fixture
def waste_headers() -> List[str]:
    """Column headers for waste management data."""
    return [
        "Waste Category", "Disposal Method", "Weight (kg)", "Recycled (%)",
        "Landfill (kg)", "Incinerated (kg)", "Hazardous", "Treatment Facility",
    ]


@pytest.fixture
def emissions_headers() -> List[str]:
    """Column headers for direct emissions reporting."""
    return [
        "Source", "Scope", "GHG Type", "Emission Factor",
        "Activity Data", "Unit", "Total Emissions (tCO2e)", "Uncertainty (%)",
    ]


@pytest.fixture
def procurement_headers() -> List[str]:
    """Column headers for procurement and Scope 3 data."""
    return [
        "Supplier", "Product Category", "Spend (USD)", "Emission Factor (kgCO2e/USD)",
        "Quantity", "Unit", "Origin Country", "Scope 3 Category",
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


# ---------------------------------------------------------------------------
# Config Reset Fixture
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_excel_normalizer_config():
    """Reset the config singleton between tests to avoid state leaks."""
    yield
    # Inline reset to avoid importing from greenlang.excel_normalizer
    import threading
    globals().setdefault("_config_instance", None)
