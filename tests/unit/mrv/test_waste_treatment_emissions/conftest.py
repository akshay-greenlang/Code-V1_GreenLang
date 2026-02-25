# -*- coding: utf-8 -*-
"""
Shared fixtures for AGENT-MRV-007 Waste Treatment Emissions Agent tests.

Provides common fixtures for models, config, metrics, provenance,
waste treatment database, biological treatment, thermal treatment,
wastewater treatment, and pipeline test modules.

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
# Config singleton reset (autouse)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_config_singleton():
    """Reset the config singleton before and after each test.

    This autouse fixture prevents state leakage between tests by
    resetting the WasteTreatmentConfig singleton in the holder class.
    """
    try:
        from greenlang.waste_treatment_emissions.config import reset_config
        reset_config()
    except ImportError:
        pass
    yield
    try:
        from greenlang.waste_treatment_emissions.config import reset_config
        reset_config()
    except ImportError:
        pass


@pytest.fixture
def clean_env():
    """Context manager to clean GL_WASTE_TREATMENT_ env vars for a test."""
    saved = {}
    prefix = "GL_WASTE_TREATMENT_"
    for key in list(os.environ.keys()):
        if key.startswith(prefix):
            saved[key] = os.environ.pop(key)
    yield
    for key in list(os.environ.keys()):
        if key.startswith(prefix):
            del os.environ[key]
    os.environ.update(saved)


# ---------------------------------------------------------------------------
# Tenant and facility fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_tenant_id() -> str:
    """Return a deterministic tenant ID for testing."""
    return "tenant_wt_test_001"


@pytest.fixture
def sample_facility() -> Dict[str, Any]:
    """Return a sample treatment facility dictionary for testing."""
    return {
        "facility_id": "fac-wt-001",
        "name": "Test Municipal Waste Treatment Plant",
        "facility_type": "municipal_treatment",
        "treatment_methods": [
            "incineration",
            "composting",
            "anaerobic_digestion",
        ],
        "capacity_tonnes_per_year": Decimal("50000"),
        "latitude": Decimal("51.5074"),
        "longitude": Decimal("-0.1278"),
        "country_code": "GB",
        "region": "Europe",
        "tenant_id": "tenant_wt_test_001",
        "operating_since": "2015-01-01",
        "status": "active",
    }


@pytest.fixture
def sample_waste_stream() -> Dict[str, Any]:
    """Return a sample waste stream dictionary for testing."""
    return {
        "stream_id": "ws-001",
        "facility_id": "fac-wt-001",
        "waste_category": "msw",
        "treatment_method": "incineration",
        "mass_tonnes": Decimal("1000"),
        "moisture_content": Decimal("0.35"),
        "dry_matter_fraction": Decimal("0.65"),
        "carbon_content_wet": Decimal("0.20"),
        "fossil_carbon_fraction": Decimal("0.40"),
        "doc_fraction": Decimal("0.15"),
        "reporting_year": 2025,
    }


# ---------------------------------------------------------------------------
# Calculation request fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_calculation_request() -> Dict[str, Any]:
    """Return a basic waste treatment calculation request."""
    return {
        "tenant_id": "tenant_wt_test_001",
        "facility_id": "fac-wt-001",
        "waste_category": "msw",
        "treatment_method": "incineration",
        "waste_mass_tonnes": Decimal("500"),
        "calculation_method": "ipcc_tier_2",
        "gwp_source": "AR6",
        "ef_source": "IPCC_2019",
        "reporting_year": 2025,
    }


@pytest.fixture
def sample_composting_request() -> Dict[str, Any]:
    """Return a composting-specific calculation request."""
    return {
        "tenant_id": "tenant_wt_test_001",
        "facility_id": "fac-wt-001",
        "waste_category": "food",
        "treatment_method": "composting",
        "waste_mass_tonnes": Decimal("200"),
        "calculation_method": "ipcc_tier_1",
        "gwp_source": "AR6",
        "ef_source": "IPCC_2019",
        "reporting_year": 2025,
        "biological_input": {
            "composting_type": "windrow",
            "moisture_content": Decimal("0.60"),
            "volatile_solids_fraction": Decimal("0.80"),
            "well_managed": True,
        },
    }


@pytest.fixture
def sample_incineration_request() -> Dict[str, Any]:
    """Return an incineration-specific calculation request."""
    return {
        "tenant_id": "tenant_wt_test_001",
        "facility_id": "fac-wt-001",
        "waste_category": "msw",
        "treatment_method": "incineration_energy_recovery",
        "waste_mass_tonnes": Decimal("1000"),
        "calculation_method": "ipcc_tier_2",
        "gwp_source": "AR6",
        "ef_source": "IPCC_2019",
        "reporting_year": 2025,
        "thermal_input": {
            "incinerator_type": "stoker_grate",
            "dry_matter_content": Decimal("0.65"),
            "oxidation_factor": Decimal("1.0"),
            "separate_biogenic": True,
            "energy_recovery_efficiency": Decimal("0.25"),
            "net_calorific_value": Decimal("9.0"),
        },
    }


@pytest.fixture
def sample_ad_request() -> Dict[str, Any]:
    """Return an anaerobic digestion calculation request."""
    return {
        "tenant_id": "tenant_wt_test_001",
        "facility_id": "fac-wt-001",
        "waste_category": "food",
        "treatment_method": "anaerobic_digestion",
        "waste_mass_tonnes": Decimal("300"),
        "calculation_method": "ipcc_tier_2",
        "gwp_source": "AR6",
        "ef_source": "IPCC_2019",
        "reporting_year": 2025,
        "biological_input": {
            "composting_type": None,
            "moisture_content": Decimal("0.70"),
            "volatile_solids_fraction": Decimal("0.87"),
            "bmp": Decimal("400"),
            "ch4_recovery_fraction": Decimal("0.0"),
            "well_managed": True,
        },
    }


@pytest.fixture
def sample_wastewater_request() -> Dict[str, Any]:
    """Return a wastewater treatment calculation request."""
    return {
        "tenant_id": "tenant_wt_test_001",
        "facility_id": "fac-wt-001",
        "waste_category": "sludge",
        "treatment_method": "biological_treatment",
        "waste_mass_tonnes": Decimal("0"),
        "calculation_method": "ipcc_tier_2",
        "gwp_source": "AR6",
        "ef_source": "IPCC_2019",
        "reporting_year": 2025,
        "wastewater_input": {
            "system_type": "aerobic_overloaded",
            "bod_load_kg": Decimal("50000"),
            "cod_load_kg": Decimal("120000"),
            "sludge_removal_fraction": Decimal("0.10"),
            "nitrogen_load_kg": Decimal("5000"),
            "ch4_recovered_tonnes": Decimal("0"),
        },
    }


@pytest.fixture
def sample_batch_request(
    sample_calculation_request: Dict[str, Any],
) -> Dict[str, Any]:
    """Return a batch of 5 calculation requests."""
    requests = []
    for i in range(5):
        req = dict(sample_calculation_request)
        req["request_id"] = f"batch-req-{i:03d}"
        req["waste_mass_tonnes"] = Decimal(str(100 * (i + 1)))
        requests.append(req)
    return {
        "batch_id": "batch-test-001",
        "tenant_id": "tenant_wt_test_001",
        "calculations": requests,
    }


@pytest.fixture
def sample_waste_composition() -> Dict[str, Any]:
    """Return a sample MSW composition breakdown summing to 100%."""
    return {
        "food_waste_pct": Decimal("30"),
        "paper_pct": Decimal("15"),
        "cardboard_pct": Decimal("8"),
        "plastic_pct": Decimal("12"),
        "textiles_pct": Decimal("3"),
        "rubber_pct": Decimal("1"),
        "wood_pct": Decimal("5"),
        "yard_waste_pct": Decimal("10"),
        "glass_pct": Decimal("5"),
        "metal_pct": Decimal("4"),
        "sludge_pct": Decimal("0"),
        "other_pct": Decimal("7"),
    }


# ---------------------------------------------------------------------------
# Engine fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db_engine():
    """Create a WasteTreatmentDatabaseEngine instance for testing."""
    try:
        from greenlang.waste_treatment_emissions.waste_treatment_database import (
            WasteTreatmentDatabaseEngine,
        )
        return WasteTreatmentDatabaseEngine()
    except ImportError:
        return MagicMock(name="WasteTreatmentDatabaseEngine")


@pytest.fixture
def bio_engine(db_engine):
    """Create a BiologicalTreatmentEngine instance for testing."""
    try:
        from greenlang.waste_treatment_emissions.biological_treatment import (
            BiologicalTreatmentEngine,
        )
        return BiologicalTreatmentEngine(database=db_engine)
    except ImportError:
        return MagicMock(name="BiologicalTreatmentEngine")


@pytest.fixture
def thermal_engine(db_engine):
    """Create a ThermalTreatmentEngine instance for testing."""
    try:
        from greenlang.waste_treatment_emissions.thermal_treatment import (
            ThermalTreatmentEngine,
        )
        return ThermalTreatmentEngine(database=db_engine)
    except ImportError:
        return MagicMock(name="ThermalTreatmentEngine")


# ---------------------------------------------------------------------------
# Provenance fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def provenance_tracker():
    """Create a fresh ProvenanceTracker for testing."""
    try:
        from greenlang.waste_treatment_emissions.provenance import ProvenanceTracker
        return ProvenanceTracker(max_entries=1000)
    except ImportError:
        return MagicMock(name="ProvenanceTracker")


# ---------------------------------------------------------------------------
# Service instance fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def service_instance():
    """Return a fresh WasteTreatmentEmissionsService-like mock.

    Since the full service may not yet be implemented, this fixture
    returns a MagicMock with commonly expected attributes.
    """
    service = MagicMock(name="WasteTreatmentEmissionsService")
    service.config = MagicMock()
    service.config.enabled = True
    service.config.default_gwp_source = "AR6"
    service.config.default_calculation_method = "IPCC_TIER_2"
    service.config.default_emission_factor_source = "IPCC_2019"
    service.config.max_batch_size = 10_000
    service.config.enable_biological = True
    service.config.enable_thermal = True
    service.config.enable_wastewater = True
    service.config.enable_provenance = True
    service.config.enable_metrics = True
    return service
