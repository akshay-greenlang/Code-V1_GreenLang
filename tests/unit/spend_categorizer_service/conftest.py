# -*- coding: utf-8 -*-
"""
Pytest Fixtures for Spend Data Categorizer Service Unit Tests (AGENT-DATA-009)
==============================================================================

Provides shared fixtures for testing the spend categorizer config, models,
provenance tracker, metrics, engines, setup facade, and API router.

All tests are self-contained with no external dependencies.

Includes a module-level stub for greenlang.spend_categorizer.__init__ to bypass
engine imports that are not yet built, allowing direct submodule imports to work.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import importlib
import os
import sys
import types
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Stub the spend_categorizer package to bypass broken __init__ imports.
# This must run BEFORE any test module imports from greenlang.spend_categorizer.
# We register a lightweight package module that exposes the real submodules
# (config, models, provenance, metrics) without triggering the full __init__
# that tries to load engines not yet built.
# ---------------------------------------------------------------------------

_PKG_NAME = "greenlang.spend_categorizer"

if _PKG_NAME not in sys.modules:
    # Only stub if the package has not been loaded yet
    import greenlang  # ensure parent exists

    _stub = types.ModuleType(_PKG_NAME)
    _stub.__path__ = [
        os.path.join(os.path.dirname(greenlang.__file__), "spend_categorizer")
    ]
    _stub.__package__ = _PKG_NAME
    _stub.__file__ = os.path.join(
        _stub.__path__[0], "__init__.py"
    )
    sys.modules[_PKG_NAME] = _stub
else:
    # If already loaded (possibly with errors), just leave it
    pass


# ---------------------------------------------------------------------------
# Environment cleanup fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_spend_cat_env(monkeypatch):
    """Remove any GL_SPEND_CAT_ env vars between tests and reset config singleton."""
    prefix = "GL_SPEND_CAT_"
    for key in list(os.environ.keys()):
        if key.startswith(prefix):
            monkeypatch.delenv(key, raising=False)

    # Reset the config singleton so each test starts fresh
    from greenlang.spend_categorizer.config import reset_config

    reset_config()


# ---------------------------------------------------------------------------
# Sample Spend Records Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_spend_records() -> List[Dict[str, Any]]:
    """Realistic spend records (10 records across vendors, categories, currencies)."""
    return [
        {
            "vendor_id": "V-001",
            "vendor_name": "Office Depot Inc",
            "transaction_date": "2025-03-15",
            "amount": 12500.00,
            "currency": "USD",
            "description": "Office supplies and furniture",
            "cost_center": "CC-100",
            "gl_account": "6100",
            "naics_code": "424120",
        },
        {
            "vendor_id": "V-002",
            "vendor_name": "DHL Logistics GmbH",
            "transaction_date": "2025-03-20",
            "amount": 85000.00,
            "currency": "EUR",
            "description": "Freight and transportation services Q1",
            "cost_center": "CC-200",
            "gl_account": "6300",
            "naics_code": "484110",
        },
        {
            "vendor_id": "V-003",
            "vendor_name": "Amazon Web Services",
            "transaction_date": "2025-04-01",
            "amount": 45000.00,
            "currency": "USD",
            "description": "Cloud computing services",
            "cost_center": "CC-300",
            "gl_account": "6500",
            "naics_code": "518210",
        },
        {
            "vendor_id": "V-004",
            "vendor_name": "Acme Chemical Corp",
            "transaction_date": "2025-04-10",
            "amount": 220000.00,
            "currency": "USD",
            "description": "Industrial chemicals for manufacturing",
            "cost_center": "CC-400",
            "gl_account": "5100",
            "naics_code": "325110",
        },
        {
            "vendor_id": "V-005",
            "vendor_name": "EcoSteel Ltd",
            "transaction_date": "2025-04-15",
            "amount": 350000.00,
            "currency": "GBP",
            "description": "Steel beams and structural materials",
            "cost_center": "CC-400",
            "gl_account": "5200",
            "naics_code": "331110",
        },
        {
            "vendor_id": "V-006",
            "vendor_name": "CleanEnergy Solutions",
            "transaction_date": "2025-05-01",
            "amount": 18000.00,
            "currency": "USD",
            "description": "Renewable energy certificates",
            "cost_center": "CC-500",
            "gl_account": "6800",
            "naics_code": "221114",
        },
        {
            "vendor_id": "V-007",
            "vendor_name": "JetBlue Airways",
            "transaction_date": "2025-05-10",
            "amount": 6500.00,
            "currency": "USD",
            "description": "Business travel flights domestic",
            "cost_center": "CC-600",
            "gl_account": "6700",
            "naics_code": "481111",
        },
        {
            "vendor_id": "V-008",
            "vendor_name": "Waste Management Inc",
            "transaction_date": "2025-05-15",
            "amount": 9800.00,
            "currency": "USD",
            "description": "Waste disposal and recycling services",
            "cost_center": "CC-700",
            "gl_account": "6900",
            "naics_code": "562111",
        },
        {
            "vendor_id": "V-009",
            "vendor_name": "Siemens AG",
            "transaction_date": "2025-06-01",
            "amount": 150000.00,
            "currency": "EUR",
            "description": "Capital equipment - industrial automation",
            "cost_center": "CC-800",
            "gl_account": "1500",
            "naics_code": "335314",
        },
        {
            "vendor_id": "V-010",
            "vendor_name": "Deloitte Consulting",
            "transaction_date": "2025-06-15",
            "amount": 75000.00,
            "currency": "USD",
            "description": "Sustainability consulting services",
            "cost_center": "CC-900",
            "gl_account": "6600",
            "naics_code": "541611",
        },
    ]


# ---------------------------------------------------------------------------
# Sample Taxonomy Codes Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_taxonomy_codes() -> List[Dict[str, Any]]:
    """Sample UNSPSC, NAICS, and eCl@ss taxonomy codes."""
    return [
        {"code": "43211500", "system": "unspsc", "level": 4, "description": "Desktop computers"},
        {"code": "44121600", "system": "unspsc", "level": 4, "description": "Office supplies"},
        {"code": "78101800", "system": "unspsc", "level": 4, "description": "Freight services"},
        {"code": "484110", "system": "naics", "level": 3, "description": "General freight trucking, long-distance"},
        {"code": "325110", "system": "naics", "level": 3, "description": "Petrochemical manufacturing"},
        {"code": "331110", "system": "naics", "level": 3, "description": "Iron and steel mills"},
        {"code": "27-02-30-01", "system": "eclass", "level": 4, "description": "Steel beam"},
        {"code": "27-01-10-01", "system": "eclass", "level": 4, "description": "Iron ingot"},
        {"code": "221114", "system": "naics", "level": 3, "description": "Solar electric power generation"},
        {"code": "541611", "system": "naics", "level": 3, "description": "Administrative management consulting"},
    ]


# ---------------------------------------------------------------------------
# Sample Emission Factors Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_emission_factors() -> List[Dict[str, Any]]:
    """EPA EEIO, EXIOBASE, and DEFRA emission factor samples."""
    return [
        {"taxonomy_code": "424120", "source": "epa_eeio", "value": 0.30, "unit": "kgCO2e/USD", "region": "US", "year": 2024},
        {"taxonomy_code": "484110", "source": "epa_eeio", "value": 0.85, "unit": "kgCO2e/USD", "region": "US", "year": 2024},
        {"taxonomy_code": "518210", "source": "epa_eeio", "value": 0.14, "unit": "kgCO2e/USD", "region": "US", "year": 2024},
        {"taxonomy_code": "325110", "source": "epa_eeio", "value": 1.20, "unit": "kgCO2e/USD", "region": "US", "year": 2024},
        {"taxonomy_code": "331110", "source": "epa_eeio", "value": 1.85, "unit": "kgCO2e/USD", "region": "US", "year": 2024},
        {"taxonomy_code": "484110", "source": "exiobase", "value": 0.78, "unit": "kgCO2e/EUR", "region": "EU", "year": 2024},
        {"taxonomy_code": "331110", "source": "exiobase", "value": 1.70, "unit": "kgCO2e/EUR", "region": "EU", "year": 2024},
        {"taxonomy_code": "484110", "source": "defra", "value": 0.82, "unit": "kgCO2e/USD", "region": "UK", "year": 2025},
        {"taxonomy_code": "562111", "source": "defra", "value": 0.45, "unit": "kgCO2e/USD", "region": "UK", "year": 2025},
        {"taxonomy_code": "481111", "source": "defra", "value": 0.95, "unit": "kgCO2e/USD", "region": "UK", "year": 2025},
    ]


# ---------------------------------------------------------------------------
# Sample Categorization Rules Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_rules() -> List[Dict[str, Any]]:
    """Five custom categorization rules for spend classification."""
    return [
        {
            "name": "Office Supplies Rule",
            "match_type": "contains",
            "match_field": "description",
            "pattern": "office supplies",
            "target_taxonomy_system": "unspsc",
            "target_taxonomy_code": "44121600",
            "priority": "high",
            "confidence_boost": 0.15,
        },
        {
            "name": "Freight Rule",
            "match_type": "regex",
            "match_field": "description",
            "pattern": r"freight|shipping|logistics|transportation",
            "target_taxonomy_system": "naics",
            "target_taxonomy_code": "484110",
            "priority": "medium",
            "confidence_boost": 0.10,
        },
        {
            "name": "Cloud Computing Rule",
            "match_type": "exact",
            "match_field": "vendor_name",
            "pattern": "Amazon Web Services",
            "target_taxonomy_system": "naics",
            "target_taxonomy_code": "518210",
            "priority": "critical",
            "confidence_boost": 0.25,
        },
        {
            "name": "Chemical Vendor Rule",
            "match_type": "fuzzy",
            "match_field": "vendor_name",
            "pattern": "chemical",
            "target_taxonomy_system": "naics",
            "target_taxonomy_code": "325110",
            "priority": "low",
            "confidence_boost": 0.05,
        },
        {
            "name": "Travel Prefix Rule",
            "match_type": "starts_with",
            "match_field": "description",
            "pattern": "Business travel",
            "target_taxonomy_system": "naics",
            "target_taxonomy_code": "481111",
            "priority": "default",
            "confidence_boost": 0.0,
        },
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
    return mock_prom
