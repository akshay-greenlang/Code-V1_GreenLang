# -*- coding: utf-8 -*-
"""
Shared pytest fixtures for Refrigerants & F-Gas integration tests.

Provides reusable fixtures for service instances and sample inputs
used across end-to-end and full-pipeline integration tests.

AGENT-MRV-002: Refrigerants & F-Gas Agent (GL-MRV-SCOPE1-002)
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List

import pytest

from greenlang.refrigerants_fgas.setup import RefrigerantsFGasService


# ---------------------------------------------------------------------------
# Override parent conftest autouse fixtures that are incompatible
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def mock_agents():
    """Override the parent integration conftest ``mock_agents`` fixture.

    The parent conftest patches ``greenlang.agents.registry.get_agent``
    which is not relevant to the refrigerants-fgas service tests and may
    fail if the module is not importable.  This override is a no-op.
    """
    yield {}

# ---------------------------------------------------------------------------
# Optional imports for config/provenance reset
# ---------------------------------------------------------------------------

try:
    from greenlang.refrigerants_fgas.config import (
        RefrigerantsFGasConfig,
        reset_config,
    )
except ImportError:
    RefrigerantsFGasConfig = None  # type: ignore[assignment, misc]

    def reset_config() -> None:
        """No-op fallback when config module is unavailable."""

try:
    from greenlang.refrigerants_fgas.provenance import reset_provenance_tracker
except ImportError:

    def reset_provenance_tracker() -> None:
        """No-op fallback when provenance module is unavailable."""


# ---------------------------------------------------------------------------
# Environment variable cleanup
# ---------------------------------------------------------------------------

_ENV_VARS = [
    "GL_REFRIGERANTS_FGAS_DATABASE_URL",
    "GL_REFRIGERANTS_FGAS_REDIS_URL",
    "GL_REFRIGERANTS_FGAS_LOG_LEVEL",
    "GL_REFRIGERANTS_FGAS_DEFAULT_GWP_SOURCE",
    "GL_REFRIGERANTS_FGAS_DEFAULT_GWP_TIMEFRAME",
    "GL_REFRIGERANTS_FGAS_DEFAULT_CALCULATION_METHOD",
    "GL_REFRIGERANTS_FGAS_ENABLE_PROVENANCE",
    "GL_REFRIGERANTS_FGAS_GENESIS_HASH",
    "GL_REFRIGERANTS_FGAS_ENABLE_METRICS",
]


@pytest.fixture(autouse=True)
def clean_env() -> Generator[None, None, None]:
    """Remove GL_REFRIGERANTS_FGAS_ env vars and reset singletons."""
    saved: Dict[str, str] = {}
    for var in _ENV_VARS:
        val = os.environ.pop(var, None)
        if val is not None:
            saved[var] = val

    reset_config()
    reset_provenance_tracker()

    yield

    for var in _ENV_VARS:
        os.environ.pop(var, None)
    for var, val in saved.items():
        os.environ[var] = val

    reset_config()
    reset_provenance_tracker()


# ---------------------------------------------------------------------------
# Service fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def service() -> RefrigerantsFGasService:
    """Return a fresh RefrigerantsFGasService with default config.

    The service is started and ready for use.
    """
    svc = RefrigerantsFGasService()
    svc.startup()
    return svc


@pytest.fixture
def configured_service() -> RefrigerantsFGasService:
    """Return a RefrigerantsFGasService with custom config (if available)."""
    config = None
    if RefrigerantsFGasConfig is not None:
        config = RefrigerantsFGasConfig(
            enable_provenance=True,
            genesis_hash="integration-test-genesis",
            enable_blend_decomposition=True,
            enable_lifecycle_tracking=True,
            enable_compliance_checking=True,
        )
    svc = RefrigerantsFGasService(config=config)
    svc.startup()
    return svc


# ---------------------------------------------------------------------------
# Sample input fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_equipment_based_input() -> Dict[str, Any]:
    """Return a valid equipment-based calculation input dict.

    Models a commercial air conditioning unit charged with R-410A:
    - 25 kg charge at 6% default leak rate for COMMERCIAL_AC
    - GWP AR6 (R-410A = 2256 tCO2e/tonne)
    - Expected emissions: 25 * 0.06 * 2256 / 1000 = ~3.384 tCO2e
    """
    return {
        "refrigerant_type": "R_410A",
        "charge_kg": 25.0,
        "method": "equipment_based",
        "gwp_source": "AR6",
        "equipment_type": "COMMERCIAL_AC",
        "equipment_id": "eq_integ_001",
        "facility_id": "fac_integ_001",
    }


@pytest.fixture
def sample_mass_balance_input() -> Dict[str, Any]:
    """Return a valid mass-balance calculation input dict.

    Models an R-134A inventory reconciliation:
    - Start: 500 kg, Purchases: 100 kg, Recovery: 50 kg, End: 450 kg
    - Emissions = (500 + 100 - 50 - 450) = 100 kg
    - GWP AR6 (R-134A = 1530): 100 * 1530 / 1000 = 153 tCO2e
    """
    return {
        "refrigerant_type": "R_134A",
        "charge_kg": 500.0,
        "method": "mass_balance",
        "gwp_source": "AR6",
        "facility_id": "fac_integ_002",
        "mass_balance_data": {
            "inventory_start_kg": 500.0,
            "purchases_kg": 100.0,
            "recovery_kg": 50.0,
            "inventory_end_kg": 450.0,
        },
    }


@pytest.fixture
def sample_screening_input() -> Dict[str, Any]:
    """Return a valid screening calculation input dict.

    Models a portfolio-level screening estimate:
    - Activity data: 1000 units, screening factor: 0.02
    - Emissions = 1000 * 0.02 = 20 kg
    - GWP AR6 (R-407C = 1908): 20 * 1908 / 1000 = ~38.16 tCO2e
    """
    return {
        "refrigerant_type": "R_407C",
        "charge_kg": 10.0,
        "method": "screening",
        "gwp_source": "AR6",
        "activity_data": 1000.0,
        "screening_factor": 0.02,
        "facility_id": "fac_integ_003",
    }


@pytest.fixture
def sample_sf6_input() -> Dict[str, Any]:
    """Return a valid SF6 switchgear calculation input dict.

    Models electrical switchgear SF6 insulation gas:
    - 15 kg charge at 0.5% default leak rate for SWITCHGEAR
    - GWP AR6 (SF6 = 25200): 15 * 0.005 * 25200 / 1000 = ~1.89 tCO2e
    """
    return {
        "refrigerant_type": "SF6",
        "charge_kg": 15.0,
        "method": "equipment_based",
        "gwp_source": "AR6",
        "equipment_type": "SWITCHGEAR",
        "equipment_id": "eq_sf6_001",
        "facility_id": "fac_substation_001",
    }


@pytest.fixture
def sample_hfo_input() -> Dict[str, Any]:
    """Return a valid HFO (low-GWP) calculation input dict.

    Models an HFO-1234yf system (automotive or commercial):
    - 5 kg charge with custom 10% leak rate
    - GWP AR6 (R-1234yf = 0.501): near-zero emissions
    """
    return {
        "refrigerant_type": "R_1234YF",
        "charge_kg": 5.0,
        "method": "equipment_based",
        "gwp_source": "AR6",
        "equipment_type": "COMMERCIAL_AC",
        "custom_leak_rate_pct": 10.0,
        "facility_id": "fac_hfo_001",
    }


@pytest.fixture
def sample_r404a_input() -> Dict[str, Any]:
    """Return a valid R-404A blend calculation input dict.

    R-404A is a ternary blend: R-125 (44%), R-143a (52%), R-134a (4%).
    High-GWP refrigerant commonly found in commercial refrigeration.
    """
    return {
        "refrigerant_type": "R_404A",
        "charge_kg": 80.0,
        "method": "equipment_based",
        "gwp_source": "AR6",
        "equipment_type": "COMMERCIAL_REFRIGERATION",
        "equipment_id": "eq_r404a_001",
        "facility_id": "fac_supermarket_001",
    }


@pytest.fixture
def sample_r410a_input() -> Dict[str, Any]:
    """Return a valid R-410A blend calculation input dict.

    R-410A is a binary blend: R-32 (50%), R-125 (50%).
    Standard HVAC refrigerant being phased down under Kigali.
    """
    return {
        "refrigerant_type": "R_410A",
        "charge_kg": 30.0,
        "method": "equipment_based",
        "gwp_source": "AR6",
        "equipment_type": "COMMERCIAL_AC",
        "equipment_id": "eq_r410a_001",
        "facility_id": "fac_office_001",
    }
