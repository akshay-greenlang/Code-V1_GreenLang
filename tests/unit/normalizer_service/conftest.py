# -*- coding: utf-8 -*-
"""
Pytest Fixtures for Normalizer Service Unit Tests (AGENT-FOUND-003)
====================================================================

Provides shared fixtures for testing the normalizer service config,
converter, entity resolver, dimensional analyzer, provenance tracker,
metrics, setup facade, and API router.

All tests are self-contained with no external dependencies.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import os
from decimal import Decimal
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Configuration Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def normalizer_config():
    """Default NormalizerConfig for testing."""
    from tests.unit.normalizer_service.test_config import NormalizerConfig

    return NormalizerConfig()


# ---------------------------------------------------------------------------
# Converter Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def unit_converter():
    """UnitConverter instance for testing."""
    from tests.unit.normalizer_service.test_converter import UnitConverter

    return UnitConverter()


# ---------------------------------------------------------------------------
# Entity Resolver Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def entity_resolver():
    """EntityResolver instance for testing."""
    from tests.unit.normalizer_service.test_entity_resolver import EntityResolver

    return EntityResolver()


# ---------------------------------------------------------------------------
# Dimensional Analyzer Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dimensional_analyzer():
    """DimensionalAnalyzer instance for testing."""
    from tests.unit.normalizer_service.test_dimensional import DimensionalAnalyzer

    return DimensionalAnalyzer()


# ---------------------------------------------------------------------------
# Sample Data Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_energy_conversion() -> Dict[str, Any]:
    """Sample energy conversion request data."""
    return {
        "value": 100,
        "from_unit": "kWh",
        "to_unit": "MWh",
    }


@pytest.fixture
def sample_mass_conversion() -> Dict[str, Any]:
    """Sample mass conversion request data."""
    return {
        "value": 1000,
        "from_unit": "kg",
        "to_unit": "t",
    }


@pytest.fixture
def sample_emissions_conversion() -> Dict[str, Any]:
    """Sample emissions conversion request data."""
    return {
        "value": 1,
        "from_unit": "tCO2e",
        "to_unit": "kgCO2e",
    }


@pytest.fixture
def sample_fuel_names() -> List[str]:
    """Sample fuel name variants for entity resolution testing."""
    return ["Natural Gas", "Nat Gas", "natural-gas", "NG"]


@pytest.fixture
def sample_material_names() -> List[str]:
    """Sample material name variants for entity resolution testing."""
    return ["Portland Cement", "OPC", "CEM I"]


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
# Environment cleanup fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_normalizer_env(monkeypatch):
    """Remove any GL_NORMALIZER_ env vars between tests."""
    prefix = "GL_NORMALIZER_"
    for key in list(os.environ.keys()):
        if key.startswith(prefix):
            monkeypatch.delenv(key, raising=False)
