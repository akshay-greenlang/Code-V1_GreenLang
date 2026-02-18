# -*- coding: utf-8 -*-
"""
Shared pytest fixtures for Refrigerants & F-Gas Agent unit tests.

Provides reusable fixtures for configuration, provenance tracking,
sample models, and test cleanup across all test modules in this
test package.

AGENT-MRV-002: Refrigerants & F-Gas Agent (GL-MRV-SCOPE1-002)
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List

import pytest

from greenlang.refrigerants_fgas.config import (
    RefrigerantsFGasConfig,
    reset_config,
    set_config,
)
from greenlang.refrigerants_fgas.provenance import (
    ProvenanceEntry,
    ProvenanceTracker,
    reset_provenance_tracker,
)
from greenlang.refrigerants_fgas.models import (
    EquipmentProfile,
    EquipmentType,
    EquipmentStatus,
    RefrigerantType,
    ServiceEventType,
    LifecycleStage,
    CalculationMethod,
    GWPSource,
    GWPTimeframe,
    MassBalanceData,
)


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
    "GL_REFRIGERANTS_FGAS_MAX_REFRIGERANTS",
    "GL_REFRIGERANTS_FGAS_MAX_EQUIPMENT",
    "GL_REFRIGERANTS_FGAS_MAX_CALCULATIONS",
    "GL_REFRIGERANTS_FGAS_MAX_BLENDS",
    "GL_REFRIGERANTS_FGAS_MAX_SERVICE_EVENTS",
    "GL_REFRIGERANTS_FGAS_DEFAULT_UNCERTAINTY_ITERATIONS",
    "GL_REFRIGERANTS_FGAS_CONFIDENCE_LEVELS",
    "GL_REFRIGERANTS_FGAS_PHASE_DOWN_BASELINE_YEAR",
    "GL_REFRIGERANTS_FGAS_ENABLE_BLEND_DECOMPOSITION",
    "GL_REFRIGERANTS_FGAS_ENABLE_LIFECYCLE_TRACKING",
    "GL_REFRIGERANTS_FGAS_ENABLE_COMPLIANCE_CHECKING",
    "GL_REFRIGERANTS_FGAS_ENABLE_PROVENANCE",
    "GL_REFRIGERANTS_FGAS_GENESIS_HASH",
    "GL_REFRIGERANTS_FGAS_ENABLE_METRICS",
    "GL_REFRIGERANTS_FGAS_POOL_SIZE",
    "GL_REFRIGERANTS_FGAS_CACHE_TTL",
    "GL_REFRIGERANTS_FGAS_RATE_LIMIT",
]


@pytest.fixture(autouse=True)
def clean_env() -> Generator[None, None, None]:
    """Remove all GL_REFRIGERANTS_FGAS_ env vars before and after each test.

    Also resets the config and provenance singletons so that no state
    leaks between test cases.
    """
    saved: Dict[str, str] = {}
    for var in _ENV_VARS:
        val = os.environ.pop(var, None)
        if val is not None:
            saved[var] = val

    reset_config()
    reset_provenance_tracker()

    yield

    # Restore original env vars
    for var in _ENV_VARS:
        os.environ.pop(var, None)
    for var, val in saved.items():
        os.environ[var] = val

    reset_config()
    reset_provenance_tracker()


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_config() -> RefrigerantsFGasConfig:
    """Return a RefrigerantsFGasConfig with all default values."""
    return RefrigerantsFGasConfig()


@pytest.fixture
def custom_config() -> RefrigerantsFGasConfig:
    """Return a RefrigerantsFGasConfig with non-default customizations."""
    return RefrigerantsFGasConfig(
        database_url="postgresql://test:test@localhost:5432/testdb",
        redis_url="redis://testhost:6379/1",
        log_level="DEBUG",
        default_gwp_source="AR5",
        default_gwp_timeframe="20yr",
        default_calculation_method="mass_balance",
        max_refrigerants=10_000,
        max_equipment=20_000,
        max_calculations=500_000,
        max_blends=1_000,
        max_service_events=100_000,
        default_uncertainty_iterations=10_000,
        confidence_levels="90,95,99",
        phase_down_baseline_year=2016,
        enable_blend_decomposition=False,
        enable_lifecycle_tracking=False,
        enable_compliance_checking=False,
        enable_provenance=True,
        genesis_hash="test-refrigerants-fgas-genesis",
        enable_metrics=False,
        pool_size=10,
        cache_ttl=600,
        rate_limit=500,
    )


# ---------------------------------------------------------------------------
# Provenance fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tracker() -> ProvenanceTracker:
    """Return a fresh ProvenanceTracker with the default genesis hash."""
    return ProvenanceTracker()


@pytest.fixture
def custom_tracker() -> ProvenanceTracker:
    """Return a ProvenanceTracker with a custom genesis hash."""
    return ProvenanceTracker(genesis_hash="custom-test-genesis-hash")


@pytest.fixture
def populated_tracker() -> ProvenanceTracker:
    """Return a ProvenanceTracker pre-populated with 5+ entries.

    Records entries spanning different entity types and actions to
    exercise filtering and chain verification.
    """
    t = ProvenanceTracker(genesis_hash="populated-tracker-genesis")
    t.record("refrigerant", "register", "R_410A", data={"gwp": 2088})
    t.record("refrigerant", "register", "R_134A", data={"gwp": 1430})
    t.record("equipment", "register", "eq_chiller_01", data={"charge_kg": 50.0})
    t.record("service_event", "service", "svc_recharge_01", data={"added_kg": 5.0})
    t.record("calculation", "calculate", "calc_001", data={"tco2e": 10.44})
    t.record("blend", "decompose", "R_404A", data={"components": 3})
    t.record("compliance", "check_compliance", "comp_001", data={"status": "compliant"})
    return t


@pytest.fixture
def sample_provenance_entry() -> ProvenanceEntry:
    """Return a sample ProvenanceEntry for direct model testing."""
    return ProvenanceEntry(
        entity_type="refrigerant",
        entity_id="R_410A",
        action="register",
        hash_value="a" * 64,
        parent_hash="b" * 64,
        timestamp="2026-02-01T00:00:00+00:00",
        metadata={"data_hash": "c" * 64, "gwp_source": "AR6"},
    )


# ---------------------------------------------------------------------------
# Sample model fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_equipment_profile() -> Dict[str, Any]:
    """Return a dictionary suitable for constructing an EquipmentProfile."""
    return {
        "equipment_id": "eq_test_001",
        "equipment_type": EquipmentType.COMMERCIAL_AC,
        "refrigerant_type": RefrigerantType.R_410A,
        "charge_kg": 25.0,
        "equipment_count": 3,
        "status": EquipmentStatus.ACTIVE,
        "installation_date": datetime(2022, 6, 15, tzinfo=timezone.utc),
        "location": "Building A, Floor 2",
        "custom_leak_rate": 0.05,
    }


@pytest.fixture
def sample_mass_balance_data() -> Dict[str, Any]:
    """Return a dictionary suitable for constructing a MassBalanceData."""
    return {
        "refrigerant_type": RefrigerantType.R_134A,
        "beginning_inventory_kg": 500.0,
        "purchases_kg": 100.0,
        "sales_kg": 20.0,
        "acquisitions_kg": 10.0,
        "divestitures_kg": 5.0,
        "ending_inventory_kg": 450.0,
        "capacity_change_kg": 30.0,
    }


# ---------------------------------------------------------------------------
# Time fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def utcnow() -> datetime:
    """Return a fixed UTC datetime for deterministic testing."""
    return datetime(2026, 2, 1, 12, 0, 0, tzinfo=timezone.utc)
