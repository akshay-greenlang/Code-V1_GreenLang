# -*- coding: utf-8 -*-
"""
Pytest Fixtures for Citations & Evidence Service Unit Tests (AGENT-FOUND-005)
=============================================================================

Provides shared fixtures for testing the citations service config, models,
registry, evidence manager, verification engine, provenance tracker,
export/import manager, metrics, setup facade, and API router.

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
def _clean_citations_env(monkeypatch):
    """Remove any GL_CITATIONS_ env vars between tests."""
    prefix = "GL_CITATIONS_"
    for key in list(os.environ.keys()):
        if key.startswith(prefix):
            monkeypatch.delenv(key, raising=False)


# ---------------------------------------------------------------------------
# Sample Data Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_citation_metadata() -> Dict[str, Any]:
    """Sample DEFRA 2024 citation metadata."""
    return {
        "title": "UK Government GHG Conversion Factors for Company Reporting",
        "authors": ["DEFRA", "BEIS"],
        "publication_date": "2024-06-01",
        "version": "2024",
        "publisher": "UK Department for Environment, Food & Rural Affairs",
        "url": "https://www.gov.uk/government/collections/government-conversion-factors-for-company-reporting",
    }


@pytest.fixture
def sample_citation_data() -> Dict[str, Any]:
    """Sample emission factor citation data."""
    return {
        "citation_id": "defra-2024-ghg-001",
        "citation_type": "emission_factor",
        "source_authority": "defra",
        "metadata": {
            "title": "UK Government GHG Conversion Factors for Company Reporting",
            "authors": ["DEFRA", "BEIS"],
            "publication_date": "2024-06-01",
            "version": "2024",
            "publisher": "UK DEFRA",
            "url": "https://www.gov.uk/ghg-conversion-factors",
        },
        "effective_date": "2024-01-01",
        "expiration_date": "2025-12-31",
        "regulatory_frameworks": ["csrd", "cbam"],
        "key_values": {
            "diesel_combustion_kgco2e_per_litre": 2.68,
            "natural_gas_kgco2e_per_kwh": 0.18,
        },
    }


@pytest.fixture
def sample_scientific_citation_data() -> Dict[str, Any]:
    """Sample scientific citation with DOI."""
    return {
        "citation_id": "ipcc-ar6-wg3-2022",
        "citation_type": "scientific",
        "source_authority": "ipcc",
        "metadata": {
            "title": "Climate Change 2022: Mitigation of Climate Change",
            "authors": ["IPCC Working Group III"],
            "publication_date": "2022-04-04",
            "version": "AR6",
            "publisher": "Cambridge University Press",
            "doi": "10.1017/9781009157926",
        },
        "effective_date": "2022-04-04",
        "key_values": {"gwp_ch4_100yr": 27.9, "gwp_n2o_100yr": 273},
    }


@pytest.fixture
def sample_regulatory_citation_data() -> Dict[str, Any]:
    """Sample CSRD regulatory citation."""
    return {
        "citation_id": "csrd-directive-2022-2464",
        "citation_type": "regulatory",
        "source_authority": "eu_commission",
        "metadata": {
            "title": "Directive (EU) 2022/2464 - Corporate Sustainability Reporting Directive",
            "authors": ["European Parliament", "Council of the EU"],
            "publication_date": "2022-12-14",
            "version": "2022/2464",
            "publisher": "Official Journal of the European Union",
        },
        "effective_date": "2024-01-01",
        "regulatory_frameworks": ["csrd"],
    }


@pytest.fixture
def sample_evidence_item_data() -> Dict[str, Any]:
    """Sample calculation evidence item."""
    return {
        "evidence_id": "ev-calc-001",
        "evidence_type": "calculation",
        "description": "Scope 1 diesel combustion emissions calculation",
        "data": {
            "fuel_type": "diesel",
            "quantity_litres": 10000,
            "emission_factor_kgco2e_per_litre": 2.68,
            "total_emissions_kgco2e": 26800.0,
        },
        "citation_ids": ["defra-2024-ghg-001"],
        "source_system": "greenlang",
        "source_agent": "GL-FOUND-X-005",
    }


@pytest.fixture
def sample_methodology_data() -> Dict[str, Any]:
    """Sample GHG Protocol methodology reference."""
    return {
        "reference_id": "ghg-protocol-corporate",
        "name": "GHG Protocol Corporate Standard",
        "standard": "GHG Protocol",
        "version": "Revised Edition",
        "section": "Chapter 6: Identifying and Calculating GHG Emissions",
        "description": "Corporate Accounting and Reporting Standard",
        "scope_1_applicable": True,
        "scope_2_applicable": True,
        "scope_3_applicable": False,
    }


@pytest.fixture
def sample_regulatory_requirement_data() -> Dict[str, Any]:
    """Sample CSRD Article 29b regulatory requirement."""
    return {
        "requirement_id": "csrd-art-29b",
        "framework": "csrd",
        "article": "Article 29b",
        "requirement_text": "Undertakings shall include in the management report information necessary to understand the undertaking's impacts on sustainability matters.",
        "effective_date": "2024-01-01",
        "compliance_deadline": "2025-01-01",
        "applies_to_scope_1": True,
        "applies_to_scope_2": True,
        "applies_to_scope_3": True,
    }


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
