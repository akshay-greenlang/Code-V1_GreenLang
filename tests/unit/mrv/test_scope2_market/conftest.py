# -*- coding: utf-8 -*-
"""
Shared fixtures for AGENT-MRV-010 Scope 2 Market-Based Emissions Agent tests.

Provides common fixtures for models, config, metrics, and sample data
used across all test modules in the test_scope2_market package.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Tenant and ID fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_tenant_id() -> str:
    """Return a deterministic tenant ID for testing."""
    return "tenant_test_001"


@pytest.fixture
def mock_facility_id() -> str:
    """Return a deterministic facility ID for testing."""
    return "fac_test_001"


@pytest.fixture
def mock_calculation_id() -> str:
    """Return a deterministic calculation ID for testing."""
    return "calc_test_001"


@pytest.fixture
def mock_instrument_id() -> str:
    """Return a deterministic instrument ID for testing."""
    return "inst_test_001"


# ---------------------------------------------------------------------------
# Config singleton reset
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_config_singleton():
    """Reset the Scope2MarketConfig singleton before and after each test.

    Ensures test isolation by preventing state leakage between tests.
    The singleton is reset before yield (test setup) and after yield
    (test teardown).
    """
    try:
        from greenlang.scope2_market.config import Scope2MarketConfig
        Scope2MarketConfig._instance = None
        Scope2MarketConfig._initialized = False
    except ImportError:
        pass
    yield
    try:
        from greenlang.scope2_market.config import Scope2MarketConfig
        Scope2MarketConfig._instance = None
        Scope2MarketConfig._initialized = False
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Metrics singleton reset
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_metrics_singleton():
    """Reset the Scope2MarketMetrics singleton before and after each test.

    Prevents duplicated timeseries registration errors and ensures
    test isolation for metrics tests. Also resets the module-level
    _default_metrics variable used by get_metrics().
    """
    try:
        import greenlang.scope2_market.metrics as _metrics_mod
        _metrics_mod.Scope2MarketMetrics._reset()
        _metrics_mod._default_metrics = None
    except (ImportError, AttributeError):
        pass
    yield
    try:
        import greenlang.scope2_market.metrics as _metrics_mod
        _metrics_mod.Scope2MarketMetrics._reset()
        _metrics_mod._default_metrics = None
    except (ImportError, AttributeError):
        pass


# ---------------------------------------------------------------------------
# Environment variable cleanup
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Remove all GL_S2M_ environment variables before each test.

    Uses monkeypatch to ensure automatic cleanup after each test,
    preventing environment variable leakage between tests.
    """
    for key in list(os.environ.keys()):
        if key.startswith("GL_S2M_"):
            monkeypatch.delenv(key, raising=False)


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_config():
    """Return a Scope2MarketConfig with default values for testing."""
    try:
        from greenlang.scope2_market.config import Scope2MarketConfig
        return Scope2MarketConfig()
    except ImportError:
        pytest.skip("Scope2MarketConfig not available")


# ---------------------------------------------------------------------------
# Sample data fixtures -- Energy Purchase
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_energy_purchase():
    """Return a sample energy purchase dictionary for testing.

    Represents a US facility (CAMX subregion) consuming 1000 MWh of
    grid electricity during calendar year 2025.
    """
    return {
        "purchase_id": "pur_test_001",
        "facility_id": "fac_test_001",
        "quantity_mwh": 1000,
        "unit": "mwh",
        "region": "US-CAMX",
        "period_start": datetime(2025, 1, 1, tzinfo=timezone.utc),
        "period_end": datetime(2025, 12, 31, tzinfo=timezone.utc),
    }


# ---------------------------------------------------------------------------
# Sample data fixtures -- Contractual Instrument
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_instrument():
    """Return a sample contractual instrument dictionary for testing.

    Represents a 500 MWh US wind REC tracked in Green-e with zero
    emission factor (renewable), vintage year 2025.
    """
    return {
        "type": "rec",
        "quantity_mwh": 500,
        "energy_source": "wind",
        "emission_factor": 0.0,
        "vintage_year": 2025,
        "tracking_system": "green_e",
        "certificate_id": "CERT-001",
        "region": "US-WECC",
        "status": "active",
    }


# ---------------------------------------------------------------------------
# Sample data fixtures -- Calculation Result
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_calculation_result():
    """Return a sample market-based calculation result for testing.

    Represents a partially covered facility: 500 MWh covered by
    wind RECs (zero emissions) and 500 MWh uncovered using the
    US-CAMX residual mix factor. Total: 210.0 tCO2e.
    """
    return {
        "facility_id": "fac_test_001",
        "total_mwh": 1000,
        "covered_mwh": 500,
        "uncovered_mwh": 500,
        "coverage_pct": 50.0,
        "covered_co2e_kg": 0.0,
        "uncovered_co2e_kg": 210000,
        "total_co2e_kg": 210000,
        "total_co2e_tonnes": 210.0,
        "gwp_source": "AR5",
        "gas_breakdown": [
            {"gas": "CO2", "co2e_kg": 199500},
            {"gas": "CH4", "co2e_kg": 6300},
            {"gas": "N2O", "co2e_kg": 4200},
        ],
        "provenance_hash": "abc123",
    }


# ---------------------------------------------------------------------------
# Sample data fixtures -- Calculation Request
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_calculation_request(sample_energy_purchase):
    """Return a sample calculation request for testing.

    Includes one energy purchase, AR5 GWP source, and
    ghg_protocol_scope2 compliance framework.
    """
    return {
        "tenant_id": "tenant_test_001",
        "facility_id": "fac_test_001",
        "energy_purchases": [sample_energy_purchase],
        "gwp_source": "AR5",
        "compliance_frameworks": ["ghg_protocol_scope2"],
    }


# ---------------------------------------------------------------------------
# Sample data fixtures -- Facility
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_facility():
    """Return a sample FacilityInfo dictionary for a US office building."""
    return {
        "facility_id": "fac_test_001",
        "name": "Test Office Building",
        "facility_type": "office",
        "country_code": "US",
        "grid_region": "US-CAMX",
        "tenant_id": "tenant_test_001",
    }


# ---------------------------------------------------------------------------
# Sample data fixtures -- Compliance
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_compliance_frameworks() -> List[str]:
    """Return a list of compliance frameworks for testing."""
    return [
        "ghg_protocol_scope2",
        "csrd_esrs",
        "cdp",
        "iso_14064",
        "re100",
    ]


# ---------------------------------------------------------------------------
# Sample data fixtures -- Uncertainty
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_uncertainty_request():
    """Return a sample uncertainty quantification request dict."""
    return {
        "calculation_id": "calc_test_001",
        "method": "monte_carlo",
        "iterations": 1000,
        "confidence_level": Decimal("0.95"),
    }


@pytest.fixture
def sample_uncertainty_request_analytical():
    """Return a sample analytical uncertainty request dict."""
    return {
        "calculation_id": "calc_test_002",
        "method": "analytical",
        "iterations": 0,
        "confidence_level": Decimal("0.90"),
    }
