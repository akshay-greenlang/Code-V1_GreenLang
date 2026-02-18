# -*- coding: utf-8 -*-
"""
Shared pytest fixtures for Stationary Combustion Agent unit tests.

Provides reusable fixtures for configuration, engine instances,
provenance tracking, sample models, and test cleanup across all
test modules in this test package.

AGENT-MRV-001: Stationary Combustion Agent (GL-MRV-SCOPE1-001)
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List

import pytest

from greenlang.stationary_combustion.config import (
    StationaryCombustionConfig,
    reset_config,
    set_config,
)
from greenlang.stationary_combustion.provenance import (
    ProvenanceEntry,
    ProvenanceTracker,
    reset_provenance_tracker,
)
from greenlang.stationary_combustion.models import (
    CombustionInput,
    FuelType,
    UnitType,
    HeatingValueBasis,
    CalculationTier,
)


# ---------------------------------------------------------------------------
# Environment variable cleanup
# ---------------------------------------------------------------------------

_ENV_VARS = [
    "GL_STATIONARY_COMBUSTION_DATABASE_URL",
    "GL_STATIONARY_COMBUSTION_REDIS_URL",
    "GL_STATIONARY_COMBUSTION_LOG_LEVEL",
    "GL_STATIONARY_COMBUSTION_DEFAULT_GWP_SOURCE",
    "GL_STATIONARY_COMBUSTION_DEFAULT_TIER",
    "GL_STATIONARY_COMBUSTION_DEFAULT_OXIDATION_FACTOR",
    "GL_STATIONARY_COMBUSTION_DECIMAL_PRECISION",
    "GL_STATIONARY_COMBUSTION_MAX_BATCH_SIZE",
    "GL_STATIONARY_COMBUSTION_MAX_FUEL_TYPES",
    "GL_STATIONARY_COMBUSTION_MAX_EMISSION_FACTORS",
    "GL_STATIONARY_COMBUSTION_MAX_EQUIPMENT_PROFILES",
    "GL_STATIONARY_COMBUSTION_MAX_CALCULATIONS",
    "GL_STATIONARY_COMBUSTION_MONTE_CARLO_ITERATIONS",
    "GL_STATIONARY_COMBUSTION_CONFIDENCE_LEVELS",
    "GL_STATIONARY_COMBUSTION_ENABLE_BIOGENIC_TRACKING",
    "GL_STATIONARY_COMBUSTION_ENABLE_PROVENANCE",
    "GL_STATIONARY_COMBUSTION_GENESIS_HASH",
    "GL_STATIONARY_COMBUSTION_ENABLE_METRICS",
    "GL_STATIONARY_COMBUSTION_POOL_SIZE",
    "GL_STATIONARY_COMBUSTION_CACHE_TTL",
    "GL_STATIONARY_COMBUSTION_RATE_LIMIT",
]


@pytest.fixture(autouse=True)
def clean_env() -> Generator[None, None, None]:
    """Remove all GL_STATIONARY_COMBUSTION_ env vars before and after each test.

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
def default_config() -> StationaryCombustionConfig:
    """Return a StationaryCombustionConfig with all default values."""
    return StationaryCombustionConfig()


@pytest.fixture
def custom_config() -> StationaryCombustionConfig:
    """Return a StationaryCombustionConfig with non-default customizations."""
    return StationaryCombustionConfig(
        database_url="postgresql://test:test@localhost:5432/testdb",
        redis_url="redis://testhost:6379/1",
        log_level="DEBUG",
        default_gwp_source="AR5",
        default_tier=2,
        default_oxidation_factor=0.99,
        decimal_precision=10,
        max_batch_size=5_000,
        max_fuel_types=500,
        max_emission_factors=5_000,
        max_equipment_profiles=2_500,
        max_calculations=50_000,
        monte_carlo_iterations=10_000,
        confidence_levels="90,95,99",
        enable_biogenic_tracking=True,
        enable_provenance=True,
        genesis_hash="test-genesis-hash",
        enable_metrics=False,
        pool_size=5,
        cache_ttl=1800,
        rate_limit=500,
    )


# ---------------------------------------------------------------------------
# Engine fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fuel_database():
    """Return a FuelDatabaseEngine instance."""
    from greenlang.stationary_combustion.fuel_database import FuelDatabaseEngine

    return FuelDatabaseEngine(config={"enable_provenance": False})


@pytest.fixture
def combustion_calculator(fuel_database):
    """Return a CombustionCalculatorEngine with a test fuel_database."""
    from greenlang.stationary_combustion.combustion_calculator import (
        CombustionCalculatorEngine,
    )

    return CombustionCalculatorEngine(
        fuel_database=fuel_database,
        config={"enable_provenance": False, "decimal_precision": 8},
    )


@pytest.fixture
def equipment_profiler():
    """Return an EquipmentProfilerEngine instance."""
    from greenlang.stationary_combustion.equipment_profiler import (
        EquipmentProfilerEngine,
    )

    return EquipmentProfilerEngine(config={"enable_provenance": False})


@pytest.fixture
def emission_factor_selector(fuel_database):
    """Return an EmissionFactorSelectorEngine with a test fuel_database."""
    from greenlang.stationary_combustion.emission_factor_selector import (
        EmissionFactorSelectorEngine,
    )

    return EmissionFactorSelectorEngine(fuel_database=fuel_database)


@pytest.fixture
def uncertainty_quantifier():
    """Return an UncertaintyQuantifierEngine instance."""
    from greenlang.stationary_combustion.uncertainty_quantifier import (
        UncertaintyQuantifierEngine,
    )

    return UncertaintyQuantifierEngine()


@pytest.fixture
def audit_trail_engine():
    """Return an AuditTrailEngine instance."""
    from greenlang.stationary_combustion.audit_trail import AuditTrailEngine

    return AuditTrailEngine()


@pytest.fixture
def pipeline(
    fuel_database,
    combustion_calculator,
    equipment_profiler,
    emission_factor_selector,
    uncertainty_quantifier,
    audit_trail_engine,
    default_config,
):
    """Return a StationaryCombustionPipelineEngine with all engines wired."""
    from greenlang.stationary_combustion.combustion_pipeline import (
        StationaryCombustionPipelineEngine,
    )

    return StationaryCombustionPipelineEngine(
        fuel_database=fuel_database,
        calculator=combustion_calculator,
        equipment_profiler=equipment_profiler,
        factor_selector=emission_factor_selector,
        uncertainty_engine=uncertainty_quantifier,
        audit_engine=audit_trail_engine,
        config=default_config,
    )


# ---------------------------------------------------------------------------
# Sample input fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_combustion_input() -> CombustionInput:
    """Return a valid CombustionInput for natural gas, 1000 m3."""
    return CombustionInput(
        fuel_type=FuelType.NATURAL_GAS,
        quantity=1000.0,
        unit=UnitType.CUBIC_METERS,
        period_start=datetime(2025, 1, 1, tzinfo=timezone.utc),
        period_end=datetime(2025, 12, 31, tzinfo=timezone.utc),
        heating_value_basis=HeatingValueBasis.HHV,
    )


@pytest.fixture
def sample_batch_inputs() -> List[CombustionInput]:
    """Return a list of 5 CombustionInput records for batch testing."""
    period_start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    period_end = datetime(2025, 12, 31, tzinfo=timezone.utc)

    return [
        CombustionInput(
            fuel_type=FuelType.NATURAL_GAS,
            quantity=1000.0,
            unit=UnitType.CUBIC_METERS,
            period_start=period_start,
            period_end=period_end,
            facility_id="facility_001",
        ),
        CombustionInput(
            fuel_type=FuelType.DIESEL,
            quantity=500.0,
            unit=UnitType.LITERS,
            period_start=period_start,
            period_end=period_end,
            facility_id="facility_001",
        ),
        CombustionInput(
            fuel_type=FuelType.COAL_BITUMINOUS,
            quantity=200.0,
            unit=UnitType.TONNES,
            period_start=period_start,
            period_end=period_end,
            facility_id="facility_002",
        ),
        CombustionInput(
            fuel_type=FuelType.WOOD,
            quantity=300.0,
            unit=UnitType.TONNES,
            period_start=period_start,
            period_end=period_end,
            facility_id="facility_002",
        ),
        CombustionInput(
            fuel_type=FuelType.PROPANE,
            quantity=150.0,
            unit=UnitType.GALLONS,
            period_start=period_start,
            period_end=period_end,
            facility_id="facility_003",
        ),
    ]


# ---------------------------------------------------------------------------
# Provenance fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def provenance_tracker() -> ProvenanceTracker:
    """Return a fresh ProvenanceTracker with the default genesis hash."""
    return ProvenanceTracker()
