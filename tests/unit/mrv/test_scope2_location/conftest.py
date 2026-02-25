# -*- coding: utf-8 -*-
"""
Shared fixtures for AGENT-MRV-009 Scope 2 Location-Based Emissions Agent tests.

Provides common fixtures for models, config, metrics, provenance,
grid factor database, electricity emissions, steam/heat/cooling,
transmission loss, uncertainty quantifier, compliance checker,
and pipeline engine test modules.

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


# ---------------------------------------------------------------------------
# Config singleton reset
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_config_singleton():
    """Reset the Scope2LocationConfig singleton before and after each test.

    Ensures test isolation by preventing state leakage between tests.
    The singleton is reset before yield (test setup) and after yield
    (test teardown).
    """
    try:
        from greenlang.scope2_location.config import Scope2LocationConfig
        Scope2LocationConfig._instance = None
        Scope2LocationConfig._initialized = False
    except ImportError:
        pass
    yield
    try:
        from greenlang.scope2_location.config import Scope2LocationConfig
        Scope2LocationConfig._instance = None
        Scope2LocationConfig._initialized = False
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Metrics singleton reset
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_metrics_singleton():
    """Reset the Scope2LocationMetrics singleton before and after each test.

    Prevents duplicated timeseries registration errors and ensures
    test isolation for metrics tests. Also resets the module-level
    _default_metrics variable used by get_metrics().
    """
    try:
        import greenlang.scope2_location.metrics as _metrics_mod
        _metrics_mod.Scope2LocationMetrics._reset()
        _metrics_mod._default_metrics = None
    except (ImportError, AttributeError):
        pass
    yield
    try:
        import greenlang.scope2_location.metrics as _metrics_mod
        _metrics_mod.Scope2LocationMetrics._reset()
        _metrics_mod._default_metrics = None
    except (ImportError, AttributeError):
        pass


# ---------------------------------------------------------------------------
# Environment variable cleanup
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Remove all GL_S2L_ environment variables before each test.

    Uses monkeypatch to ensure automatic cleanup after each test,
    preventing environment variable leakage between tests.
    """
    for key in list(os.environ.keys()):
        if key.startswith("GL_S2L_"):
            monkeypatch.delenv(key, raising=False)


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_config():
    """Return a Scope2LocationConfig with default values for testing."""
    try:
        from greenlang.scope2_location.config import Scope2LocationConfig
        return Scope2LocationConfig()
    except ImportError:
        pytest.skip("Scope2LocationConfig not available")


# ---------------------------------------------------------------------------
# Engine fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def grid_factor_db():
    """Create a GridEmissionFactorDatabaseEngine instance for testing."""
    try:
        from greenlang.scope2_location.grid_factor_database import (
            GridEmissionFactorDatabaseEngine,
        )
        return GridEmissionFactorDatabaseEngine()
    except ImportError:
        pytest.skip("GridEmissionFactorDatabaseEngine not available")


@pytest.fixture
def electricity_engine(grid_factor_db):
    """Create an ElectricityEmissionsEngine instance for testing."""
    try:
        from greenlang.scope2_location.electricity_emissions import (
            ElectricityEmissionsEngine,
        )
        return ElectricityEmissionsEngine(grid_factor_db)
    except ImportError:
        pytest.skip("ElectricityEmissionsEngine not available")


@pytest.fixture
def steam_heat_cool_engine(grid_factor_db):
    """Create a SteamHeatCoolingEngine instance for testing."""
    try:
        from greenlang.scope2_location.steam_heat_cooling import (
            SteamHeatCoolingEngine,
        )
        return SteamHeatCoolingEngine(grid_factor_db)
    except ImportError:
        pytest.skip("SteamHeatCoolingEngine not available")


@pytest.fixture
def transmission_engine():
    """Create a TransmissionLossEngine instance for testing."""
    try:
        from greenlang.scope2_location.transmission_loss import (
            TransmissionLossEngine,
        )
        return TransmissionLossEngine()
    except ImportError:
        pytest.skip("TransmissionLossEngine not available")


@pytest.fixture
def uncertainty_engine():
    """Create an UncertaintyQuantifierEngine instance for testing."""
    try:
        from greenlang.scope2_location.uncertainty_quantifier import (
            UncertaintyQuantifierEngine,
        )
        return UncertaintyQuantifierEngine()
    except ImportError:
        pytest.skip("UncertaintyQuantifierEngine not available")


@pytest.fixture
def compliance_engine():
    """Create a ComplianceCheckerEngine instance for testing."""
    try:
        from greenlang.scope2_location.compliance_checker import (
            ComplianceCheckerEngine,
        )
        return ComplianceCheckerEngine()
    except ImportError:
        pytest.skip("ComplianceCheckerEngine not available")


@pytest.fixture
def pipeline_engine(
    grid_factor_db,
    electricity_engine,
    steam_heat_cool_engine,
    transmission_engine,
    uncertainty_engine,
    compliance_engine,
):
    """Create a Scope2LocationPipelineEngine with all sub-engines."""
    try:
        from greenlang.scope2_location.scope2_location_pipeline import (
            Scope2LocationPipelineEngine,
        )
        return Scope2LocationPipelineEngine(
            grid_factor_db,
            electricity_engine,
            steam_heat_cool_engine,
            transmission_engine,
            uncertainty_engine,
            compliance_engine,
        )
    except ImportError:
        pytest.skip("Scope2LocationPipelineEngine not available")


@pytest.fixture
def provenance_tracker():
    """Create a fresh ProvenanceTracker instance for testing."""
    try:
        from greenlang.scope2_location.provenance import ProvenanceTracker
        return ProvenanceTracker(max_entries=1000)
    except ImportError:
        pytest.skip("ProvenanceTracker not available")


# ---------------------------------------------------------------------------
# Sample data fixtures -- Electricity
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_electricity_request():
    """Return a sample electricity consumption calculation request dict.

    Represents a typical US facility (CAMX eGRID subregion) consuming
    1000 MWh of grid electricity with T&D loss inclusion.
    """
    return {
        "calculation_id": "calc-001",
        "tenant_id": "tenant-001",
        "facility_id": "fac-001",
        "energy_type": "electricity",
        "consumption_value": Decimal("1000"),
        "consumption_unit": "mwh",
        "country_code": "US",
        "egrid_subregion": "CAMX",
        "gwp_source": "AR5",
        "include_td_losses": True,
    }


@pytest.fixture
def sample_electricity_request_gb():
    """Return a sample UK electricity request using IEA country factors."""
    return {
        "calculation_id": "calc-003",
        "tenant_id": "tenant-001",
        "facility_id": "fac-003",
        "energy_type": "electricity",
        "consumption_value": Decimal("500"),
        "consumption_unit": "mwh",
        "country_code": "GB",
        "gwp_source": "AR6",
        "include_td_losses": True,
    }


@pytest.fixture
def sample_electricity_request_de():
    """Return a sample German electricity request using EU factors."""
    return {
        "calculation_id": "calc-004",
        "tenant_id": "tenant-001",
        "facility_id": "fac-004",
        "energy_type": "electricity",
        "consumption_value": Decimal("2000"),
        "consumption_unit": "mwh",
        "country_code": "DE",
        "gwp_source": "AR5",
        "include_td_losses": False,
    }


# ---------------------------------------------------------------------------
# Sample data fixtures -- Steam / Heat / Cooling
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_steam_request():
    """Return a sample steam consumption request dict.

    Represents a UK facility consuming 500 GJ of natural-gas-fired steam.
    """
    return {
        "calculation_id": "calc-002",
        "tenant_id": "tenant-001",
        "facility_id": "fac-001",
        "energy_type": "steam",
        "consumption_value": Decimal("500"),
        "consumption_unit": "gj",
        "country_code": "GB",
        "steam_type": "natural_gas",
    }


@pytest.fixture
def sample_heating_request():
    """Return a sample district heating request dict."""
    return {
        "calculation_id": "calc-005",
        "tenant_id": "tenant-001",
        "facility_id": "fac-002",
        "energy_type": "heating",
        "consumption_value": Decimal("200"),
        "consumption_unit": "gj",
        "country_code": "DE",
        "heating_type": "district",
    }


@pytest.fixture
def sample_cooling_request():
    """Return a sample absorption cooling request dict."""
    return {
        "calculation_id": "calc-006",
        "tenant_id": "tenant-001",
        "facility_id": "fac-002",
        "energy_type": "cooling",
        "consumption_value": Decimal("100"),
        "consumption_unit": "gj",
        "country_code": "SG",
        "cooling_type": "absorption",
    }


# ---------------------------------------------------------------------------
# Sample data fixtures -- Facility
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_facility():
    """Return a sample FacilityInfo dictionary for a US office building."""
    return {
        "facility_id": "fac-001",
        "name": "Test Office Building",
        "facility_type": "office",
        "country_code": "US",
        "grid_region_id": "EGRID-CAMX",
        "egrid_subregion": "CAMX",
        "latitude": Decimal("34.0522"),
        "longitude": Decimal("-118.2437"),
        "tenant_id": "tenant-001",
    }


@pytest.fixture
def sample_facility_gb():
    """Return a sample FacilityInfo dictionary for a UK data center."""
    return {
        "facility_id": "fac-003",
        "name": "London Data Center",
        "facility_type": "data_center",
        "country_code": "GB",
        "grid_region_id": "IEA-GB",
        "egrid_subregion": None,
        "latitude": Decimal("51.5074"),
        "longitude": Decimal("-0.1278"),
        "tenant_id": "tenant-001",
    }


@pytest.fixture
def sample_facility_de():
    """Return a sample FacilityInfo dictionary for a German warehouse."""
    return {
        "facility_id": "fac-004",
        "name": "Munich Warehouse",
        "facility_type": "warehouse",
        "country_code": "DE",
        "grid_region_id": "EU-DE",
        "egrid_subregion": None,
        "latitude": Decimal("48.1351"),
        "longitude": Decimal("11.5820"),
        "tenant_id": "tenant-001",
    }


# ---------------------------------------------------------------------------
# Sample data fixtures -- Transmission & Distribution Loss
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_td_loss_input_us():
    """Return a US T&D loss input dict using country average method."""
    return {
        "country_code": "US",
        "method": "country_average",
        "custom_td_loss": None,
        "include_upstream": False,
    }


@pytest.fixture
def sample_td_loss_input_custom():
    """Return a custom T&D loss input dict."""
    return {
        "country_code": "US",
        "method": "custom",
        "custom_td_loss": Decimal("0.08"),
        "include_upstream": True,
    }


# ---------------------------------------------------------------------------
# Sample data fixtures -- Batch Calculation
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_batch_request(sample_electricity_request, sample_steam_request):
    """Return a sample batch calculation request with two sub-requests."""
    return {
        "batch_id": "batch-001",
        "tenant_id": "tenant-001",
        "requests": [
            sample_electricity_request,
            sample_steam_request,
        ],
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
    ]


# ---------------------------------------------------------------------------
# Sample data fixtures -- Uncertainty
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_uncertainty_request():
    """Return a sample uncertainty quantification request dict."""
    return {
        "calculation_id": "calc-001",
        "method": "monte_carlo",
        "iterations": 1000,
        "confidence_level": Decimal("0.95"),
    }


@pytest.fixture
def sample_uncertainty_request_analytical():
    """Return a sample analytical uncertainty request dict."""
    return {
        "calculation_id": "calc-002",
        "method": "analytical",
        "iterations": 0,
        "confidence_level": Decimal("0.90"),
    }


# ---------------------------------------------------------------------------
# Known calculation values for regression testing
# ---------------------------------------------------------------------------


@pytest.fixture
def known_camx_electricity_result():
    """Return known expected results for CAMX 1000 MWh electricity calc.

    CAMX emission factors (eGRID2022):
        CO2: 225.30 kg/MWh
        CH4: 0.026 kg/MWh
        N2O: 0.003 kg/MWh
    US T&D loss: 5.0%
    AR5 GWP: CO2=1, CH4=28, N2O=265
    """
    return {
        "co2_kg": Decimal("225300"),       # 1000 * 225.30
        "ch4_kg": Decimal("26"),           # 1000 * 0.026
        "n2o_kg": Decimal("3"),            # 1000 * 0.003
        "co2e_ch4_kg": Decimal("728"),     # 26 * 28
        "co2e_n2o_kg": Decimal("795"),     # 3 * 265
        "total_co2e_kg_no_td": Decimal("226823"),  # 225300 + 728 + 795
        "td_loss_pct": Decimal("0.050"),
    }


@pytest.fixture
def known_gb_electricity_result():
    """Return known expected results for GB 500 MWh electricity calc.

    IEA GB factor: 0.212 tCO2/MWh = 212 kgCO2/MWh
    GB T&D loss: 7.7%
    """
    return {
        "co2_kg_per_mwh": Decimal("212"),
        "total_co2_kg_no_td": Decimal("106000"),  # 500 * 212
        "td_loss_pct": Decimal("0.077"),
    }
