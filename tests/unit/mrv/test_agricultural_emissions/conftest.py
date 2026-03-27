# -*- coding: utf-8 -*-
"""
Shared fixtures for AGENT-MRV-008 Agricultural Emissions Agent tests.

Provides common fixtures for models, config, metrics, provenance,
agricultural database, enteric fermentation, manure management,
cropland emissions, rice cultivation, field burning, compliance
checker, uncertainty quantifier, and pipeline test modules.

Fixtures cover:
- Config singleton reset (autouse) to prevent state leakage
- Environment variable isolation for GL_AGRICULTURAL_ prefix
- Tenant and farm registration data
- Livestock population and herd records
- Enteric fermentation calculation requests (Tier 1 / Tier 2)
- Manure management calculation requests (single + multi-AWMS)
- Cropland N2O calculation requests (synthetic N, organic N, liming)
- Rice cultivation CH4 calculation requests (water regime, amendments)
- Field burning calculation requests (crop residue combustion)
- Batch calculation request (5 mixed-source requests)
- Engine instances with graceful import fallback (MagicMock)
- Provenance tracker for SHA-256 chain testing
- Service instance mock with config attributes

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-008 Agricultural Emissions (GL-MRV-SCOPE1-008)
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
# Config singleton reset (autouse)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_config_singleton():
    """Reset the config singleton before and after each test.

    This autouse fixture prevents state leakage between tests by
    resetting the AgriculturalEmissionsConfig singleton in the
    holder class.
    """
    try:
        from greenlang.agents.mrv.agricultural_emissions.config import reset_config
        reset_config()
    except ImportError:
        pass
    yield
    try:
        from greenlang.agents.mrv.agricultural_emissions.config import reset_config
        reset_config()
    except ImportError:
        pass


@pytest.fixture
def clean_env():
    """Context manager to clean GL_AGRICULTURAL_ env vars for a test.

    Saves any existing GL_AGRICULTURAL_* environment variables,
    removes them for the test duration, and restores them afterward.
    This ensures tests that manipulate env vars do not leak into
    subsequent tests.
    """
    saved = {}
    prefix = "GL_AGRICULTURAL_"
    for key in list(os.environ.keys()):
        if key.startswith(prefix):
            saved[key] = os.environ.pop(key)
    yield
    for key in list(os.environ.keys()):
        if key.startswith(prefix):
            del os.environ[key]
    os.environ.update(saved)


# ---------------------------------------------------------------------------
# Metrics singleton reset (autouse)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_metrics_singleton():
    """Reset MetricsCollector singleton before each test."""
    try:
        from greenlang.agents.mrv.agricultural_emissions.metrics import MetricsCollector
        MetricsCollector._instance = None
    except ImportError:
        pass
    yield
    try:
        from greenlang.agents.mrv.agricultural_emissions.metrics import MetricsCollector
        MetricsCollector._instance = None
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Provenance singleton reset (autouse)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_provenance_singleton():
    """Reset provenance singleton before and after each test."""
    try:
        from greenlang.agents.mrv.agricultural_emissions.provenance import (
            reset_provenance_tracker,
        )
        reset_provenance_tracker()
    except ImportError:
        pass
    yield
    try:
        from greenlang.agents.mrv.agricultural_emissions.provenance import (
            reset_provenance_tracker,
        )
        reset_provenance_tracker()
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Tenant fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_tenant_id() -> str:
    """Return a deterministic tenant ID for testing."""
    return "tenant_ag_test_001"


# ---------------------------------------------------------------------------
# Farm fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_farm(mock_tenant_id: str) -> Dict[str, Any]:
    """Return a sample agricultural farm dictionary for testing.

    Represents a mixed crop-livestock farm in western Europe with
    dairy cattle, cropland, and pasture areas.
    """
    return {
        "farm_id": "farm-ag-001",
        "name": "Test Dairy and Crop Farm",
        "farm_type": "mixed_crop_livestock",
        "latitude": Decimal("51.5074"),
        "longitude": Decimal("-0.1278"),
        "climate_zone": "cool_temperate_wet",
        "country_code": "GB",
        "region": "developed",
        "total_area_ha": Decimal("500"),
        "cropland_area_ha": Decimal("200"),
        "pasture_area_ha": Decimal("250"),
        "soil_type": "mineral",
        "organic_soil_area_ha": Decimal("0"),
        "tenant_id": mock_tenant_id,
        "is_active": True,
        "operating_since": "2010-01-01",
        "status": "active",
    }


# ---------------------------------------------------------------------------
# Livestock fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_livestock(mock_tenant_id: str) -> Dict[str, Any]:
    """Return a sample livestock population dictionary for testing.

    Represents a dairy cattle herd of 200 head in western Europe
    for the reference year 2025.
    """
    return {
        "herd_id": "herd-ag-001",
        "farm_id": "farm-ag-001",
        "animal_type": "dairy_cattle",
        "head_count": 200,
        "typical_animal_mass_kg": Decimal("600"),
        "region": "developed",
        "data_quality_tier": "tier_1",
        "reference_year": 2025,
        "tenant_id": mock_tenant_id,
        "notes": "Holstein Friesian dairy herd",
    }


# ---------------------------------------------------------------------------
# Enteric fermentation request fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_enteric_request(mock_tenant_id: str) -> Dict[str, Any]:
    """Return a sample enteric fermentation calculation request.

    Tier 1 request for 200 dairy cattle in a developed region
    using AR6 GWP values.
    """
    return {
        "tenant_id": mock_tenant_id,
        "farm_id": "farm-ag-001",
        "animal_type": "dairy_cattle",
        "head_count": 200,
        "region": "developed",
        "calculation_method": "ipcc_tier_1",
        "gwp_source": "AR6",
        "ef_source": "IPCC_2019",
        "reference_year": 2025,
    }


# ---------------------------------------------------------------------------
# Manure management request fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_manure_request(mock_tenant_id: str) -> Dict[str, Any]:
    """Return a sample manure management calculation request.

    Request for 200 dairy cattle with 60% liquid slurry (no crust)
    and 40% solid storage in temperate climate.
    """
    return {
        "tenant_id": mock_tenant_id,
        "farm_id": "farm-ag-001",
        "animal_type": "dairy_cattle",
        "head_count": 200,
        "region": "developed",
        "manure_allocations": [
            {
                "manure_system": "liquid_slurry_no_crust",
                "allocation_fraction": Decimal("0.60"),
                "temperature_class": "temperate",
            },
            {
                "manure_system": "solid_storage",
                "allocation_fraction": Decimal("0.40"),
                "temperature_class": "temperate",
            },
        ],
        "calculation_method": "ipcc_tier_1",
        "gwp_source": "AR6",
        "ef_source": "IPCC_2019",
        "reference_year": 2025,
    }


# ---------------------------------------------------------------------------
# Cropland N2O request fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_cropland_request(mock_tenant_id: str) -> Dict[str, Any]:
    """Return a sample cropland N2O calculation request.

    Wheat cropland of 50 ha receiving synthetic N fertilizer,
    including indirect N2O from volatilisation and leaching.
    """
    return {
        "tenant_id": mock_tenant_id,
        "farm_id": "farm-ag-001",
        "crop_type": "wheat",
        "area_ha": Decimal("50"),
        "crop_yield_tonnes_per_ha": Decimal("7.5"),
        "synthetic_n_kg": Decimal("5000"),
        "organic_n_kg": Decimal("0"),
        "crop_residue_n_kg": None,
        "sewage_sludge_n_kg": Decimal("0"),
        "limestone_tonnes": Decimal("5"),
        "dolomite_tonnes": Decimal("0"),
        "urea_tonnes": Decimal("2"),
        "fraction_burned": Decimal("0"),
        "soil_type": "mineral",
        "include_indirect_n2o": True,
        "calculation_method": "ipcc_tier_1",
        "gwp_source": "AR6",
        "ef_source": "IPCC_2019",
        "reference_year": 2025,
    }


# ---------------------------------------------------------------------------
# Rice cultivation request fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_rice_request(mock_tenant_id: str) -> Dict[str, Any]:
    """Return a sample rice cultivation CH4 calculation request.

    10 ha continuously flooded rice paddy with 120-day growing season
    and straw (short pre-incorporation) organic amendment.
    """
    return {
        "tenant_id": mock_tenant_id,
        "farm_id": "farm-ag-001",
        "field_name": "Paddy Field A",
        "area_ha": Decimal("10"),
        "water_regime": "continuously_flooded",
        "pre_season_flooding": "not_known",
        "growing_season_days": 120,
        "organic_amendments": ["straw_short"],
        "organic_amendment_rates": [Decimal("2.0")],
        "cultivations_per_year": 1,
        "calculation_method": "ipcc_tier_1",
        "gwp_source": "AR6",
        "ef_source": "IPCC_2019",
        "reference_year": 2025,
    }


# ---------------------------------------------------------------------------
# Field burning request fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_field_burning_request(mock_tenant_id: str) -> Dict[str, Any]:
    """Return a sample field burning calculation request.

    Wheat residue burning: 500 tonnes crop production with 25%
    of residues burned in the field.
    """
    return {
        "tenant_id": mock_tenant_id,
        "farm_id": "farm-ag-001",
        "crop_type": "wheat",
        "crop_production_tonnes": Decimal("500"),
        "fraction_burned": Decimal("0.25"),
        "calculation_method": "ipcc_tier_1",
        "gwp_source": "AR6",
        "ef_source": "IPCC_2019",
        "reference_year": 2025,
    }


# ---------------------------------------------------------------------------
# Batch calculation request fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_batch_request(
    sample_enteric_request: Dict[str, Any],
    sample_manure_request: Dict[str, Any],
    sample_cropland_request: Dict[str, Any],
    sample_rice_request: Dict[str, Any],
    sample_field_burning_request: Dict[str, Any],
) -> Dict[str, Any]:
    """Return a batch of 5 mixed-source calculation requests.

    Includes one of each: enteric fermentation, manure management,
    cropland N2O, rice cultivation CH4, and field burning.
    """
    requests = []
    for i, req in enumerate([
        sample_enteric_request,
        sample_manure_request,
        sample_cropland_request,
        sample_rice_request,
        sample_field_burning_request,
    ]):
        r = dict(req)
        r["request_id"] = f"batch-req-{i:03d}"
        requests.append(r)
    return {
        "batch_id": "batch-ag-test-001",
        "tenant_id": "tenant_ag_test_001",
        "calculations": requests,
    }


# ---------------------------------------------------------------------------
# Engine fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db_engine():
    """Create an AgriculturalDatabaseEngine instance for testing.

    Falls back to MagicMock if the module is not importable.
    """
    try:
        from greenlang.agents.mrv.agricultural_emissions.agricultural_database import (
            AgriculturalDatabaseEngine,
        )
        return AgriculturalDatabaseEngine()
    except ImportError:
        return MagicMock(name="AgriculturalDatabaseEngine")


@pytest.fixture
def enteric_engine(db_engine):
    """Create an EntericFermentationEngine instance for testing.

    Falls back to MagicMock if the module is not importable.
    """
    try:
        from greenlang.agents.mrv.agricultural_emissions.enteric_fermentation import (
            EntericFermentationEngine,
        )
        return EntericFermentationEngine(database=db_engine)
    except ImportError:
        return MagicMock(name="EntericFermentationEngine")


@pytest.fixture
def manure_engine(db_engine):
    """Create a ManureManagementEngine instance for testing.

    Falls back to MagicMock if the module is not importable.
    """
    try:
        from greenlang.agents.mrv.agricultural_emissions.manure_management import (
            ManureManagementEngine,
        )
        return ManureManagementEngine(database=db_engine)
    except ImportError:
        return MagicMock(name="ManureManagementEngine")


@pytest.fixture
def cropland_engine(db_engine):
    """Create a CroplandEmissionsEngine instance for testing.

    Falls back to MagicMock if the module is not importable.
    """
    try:
        from greenlang.agents.mrv.agricultural_emissions.cropland_emissions import (
            CroplandEmissionsEngine,
        )
        return CroplandEmissionsEngine(database=db_engine)
    except ImportError:
        return MagicMock(name="CroplandEmissionsEngine")


# ---------------------------------------------------------------------------
# Provenance fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def provenance_tracker():
    """Create a fresh ProvenanceTracker for testing.

    Falls back to MagicMock if the module is not importable.
    """
    try:
        from greenlang.agents.mrv.agricultural_emissions.provenance import ProvenanceTracker
        return ProvenanceTracker(max_entries=1000)
    except ImportError:
        return MagicMock(name="ProvenanceTracker")


# ---------------------------------------------------------------------------
# Service instance fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def service_instance():
    """Return a fresh AgriculturalEmissionsService-like mock.

    Since the full service may not yet be fully wired, this fixture
    returns a MagicMock with commonly expected attributes matching
    the AgriculturalEmissionsConfig defaults.
    """
    service = MagicMock(name="AgriculturalEmissionsService")
    service.config = MagicMock()
    service.config.enabled = True
    service.config.default_gwp_source = "AR6"
    service.config.default_calculation_method = "IPCC_TIER_1"
    service.config.default_emission_factor_source = "IPCC_2019"
    service.config.default_climate_zone = "COOL_TEMPERATE_WET"
    service.config.max_batch_size = 10_000
    service.config.decimal_precision = 8
    service.config.enable_enteric = True
    service.config.enable_manure = True
    service.config.enable_soils = True
    service.config.enable_rice = True
    service.config.enable_field_burning = True
    service.config.enable_compliance_checking = True
    service.config.enable_uncertainty = True
    service.config.enable_provenance = True
    service.config.enable_metrics = True
    service.config.separate_biogenic_ch4 = True
    service.config.monte_carlo_iterations = 5_000
    service.config.monte_carlo_seed = 42
    service.config.default_ym_pct = 6.5
    service.config.default_de_pct = 65.0
    service.config.default_ef1 = 0.01
    service.config.default_frac_gasf = 0.10
    service.config.default_frac_gasm = 0.20
    service.config.default_frac_leach = 0.30
    service.config.default_limestone_ef = 0.12
    service.config.default_dolomite_ef = 0.13
    service.config.default_urea_ef = 0.20
    service.config.default_rice_baseline_ef = 1.30
    service.config.default_rice_cultivation_days = 120
    service.config.default_water_regime = "continuously_flooded"
    service.config.default_combustion_factor = 0.80
    service.config.default_burn_fraction = 0.25
    return service
