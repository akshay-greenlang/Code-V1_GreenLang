# -*- coding: utf-8 -*-
"""
Shared fixtures for AGENT-MRV-006 Land Use Emissions Agent tests.

Provides common fixtures for models, config, metrics, provenance,
land use database, carbon stock calculator, and land use change tracker
test modules.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Tenant fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_tenant_id() -> str:
    """Return a deterministic tenant ID for testing."""
    return "tenant_test_001"


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_config():
    """Return a LandUseConfig with default values for testing."""
    from greenlang.land_use_emissions.config import LandUseConfig
    return LandUseConfig()


@pytest.fixture(autouse=True)
def reset_config_singleton():
    """Reset the config singleton before and after each test."""
    try:
        from greenlang.land_use_emissions.config import reset_config
        reset_config()
    except ImportError:
        pass
    yield
    try:
        from greenlang.land_use_emissions.config import reset_config
        reset_config()
    except ImportError:
        pass


@pytest.fixture
def clean_env():
    """Context manager to clean GL_LAND_USE_ env vars for a test."""
    saved = {}
    prefix = "GL_LAND_USE_"
    for key in list(os.environ.keys()):
        if key.startswith(prefix):
            saved[key] = os.environ.pop(key)
    yield
    for key in list(os.environ.keys()):
        if key.startswith(prefix):
            del os.environ[key]
    os.environ.update(saved)


# ---------------------------------------------------------------------------
# Parcel fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_parcel() -> Dict[str, Any]:
    """Return a sample land parcel dictionary for testing."""
    return {
        "id": "parcel-test-001",
        "name": "Test Forest Parcel Alpha",
        "area_ha": Decimal("100.5"),
        "land_category": "forest_land",
        "climate_zone": "tropical_wet",
        "soil_type": "high_activity_clay",
        "latitude": Decimal("-3.456"),
        "longitude": Decimal("28.789"),
        "tenant_id": "tenant_test_001",
        "country_code": "CD",
    }


# ---------------------------------------------------------------------------
# Calculation request fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_calculation_request() -> Dict[str, Any]:
    """Return a sample stock-difference calculation request."""
    return {
        "land_category": "FOREST_LAND",
        "climate_zone": "TROPICAL_WET",
        "area_ha": 1000,
        "c_t1": {
            "AGB": 180,
            "BGB": 43,
            "DEAD_WOOD": 14,
            "LITTER": 5,
        },
        "c_t2": {
            "AGB": 170,
            "BGB": 40,
            "DEAD_WOOD": 13,
            "LITTER": 5,
        },
        "year_t1": 2020,
        "year_t2": 2025,
    }


@pytest.fixture
def sample_gain_loss_request() -> Dict[str, Any]:
    """Return a sample gain-loss calculation request."""
    return {
        "land_category": "FOREST_LAND",
        "climate_zone": "TROPICAL_WET",
        "area_ha": 500,
        "harvest_volume_m3": 200,
        "fuelwood_volume_m3": 50,
        "disturbance_area_ha": 0,
        "disturbance_type": "",
    }


# ---------------------------------------------------------------------------
# Transition record fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_transition_record() -> Dict[str, Any]:
    """Return a sample land-use transition record request."""
    return {
        "parcel_id": "parcel-test-001",
        "from_category": "FOREST_LAND",
        "to_category": "CROPLAND",
        "area_ha": 50.0,
        "transition_date": "2023-06-15",
        "notes": "Test deforestation event",
    }


# ---------------------------------------------------------------------------
# Carbon stock snapshot fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_carbon_stock_snapshot() -> Dict[str, Any]:
    """Return a sample carbon stock snapshot dictionary."""
    return {
        "id": "snap-test-001",
        "parcel_id": "parcel-test-001",
        "pool": "above_ground_biomass",
        "stock_tc_ha": Decimal("180.00"),
        "measurement_date": datetime(2023, 1, 1, tzinfo=timezone.utc),
        "tier": "tier_1",
        "source": "IPCC_2006",
        "uncertainty_pct": Decimal("30.0"),
        "notes": "Initial inventory measurement",
    }


# ---------------------------------------------------------------------------
# Engine fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def land_use_database_engine():
    """Create a LandUseDatabaseEngine instance for testing."""
    from greenlang.land_use_emissions.land_use_database import LandUseDatabaseEngine
    return LandUseDatabaseEngine()


@pytest.fixture
def carbon_stock_calculator(land_use_database_engine):
    """Create a CarbonStockCalculatorEngine instance for testing."""
    from greenlang.land_use_emissions.carbon_stock_calculator import (
        CarbonStockCalculatorEngine,
    )
    return CarbonStockCalculatorEngine(land_use_database=land_use_database_engine)


@pytest.fixture
def land_use_change_tracker():
    """Create a LandUseChangeTrackerEngine instance for testing."""
    from greenlang.land_use_emissions.land_use_change_tracker import (
        LandUseChangeTrackerEngine,
    )
    return LandUseChangeTrackerEngine()


@pytest.fixture
def provenance_tracker():
    """Create a fresh ProvenanceTracker instance for testing."""
    from greenlang.land_use_emissions.provenance import ProvenanceTracker
    return ProvenanceTracker(max_entries=1000)
