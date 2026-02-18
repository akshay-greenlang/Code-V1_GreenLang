# -*- coding: utf-8 -*-
"""
Integration test fixtures for AGENT-MRV-005 Fugitive Emissions Agent.

Provides heavyweight fixtures that create real (non-mock) service instances,
pre-populated services, and complete sample request dictionaries for
end-to-end and full-pipeline integration testing.

Fixtures (14):
    Service:
        service                   - fresh FugitiveEmissionsService
        populated_service         - service with pre-loaded components + calcs
    Requests:
        equipment_leak_request    - EQUIPMENT_LEAK avg EF request
        coal_mine_request         - COAL_MINE_METHANE request
        wastewater_request        - WASTEWATER treatment request
        pneumatic_request         - PNEUMATIC_DEVICE request
        tank_loss_request         - TANK_LOSS request
        batch_requests            - list of 5 heterogeneous requests
    Components:
        sample_components         - list of component registration dicts
    Surveys:
        sample_survey             - LDAR survey dict
    Factors:
        sample_factor             - custom emission factor dict
    Repairs:
        sample_repair_data        - repair registration dict
    Compliance:
        compliance_frameworks     - list of all 7 framework names
    Uncertainty:
        uncertainty_config        - uncertainty analysis dict

~370 lines
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest


# ---------------------------------------------------------------------------
# Service fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def service():
    """Create a fresh FugitiveEmissionsService instance for integration testing.

    This is a REAL service instance (not a mock) that will exercise all
    7 engines through their actual code paths.
    """
    from greenlang.fugitive_emissions.setup import FugitiveEmissionsService

    return FugitiveEmissionsService()


@pytest.fixture
def populated_service(service, sample_components):
    """Create a service pre-populated with components, a calculation,
    and a survey for tests that need existing state.
    """
    # Register components
    for comp in sample_components:
        service.register_component(comp)

    # Run a calculation to seed the history
    service.calculate({
        "source_type": "EQUIPMENT_LEAK",
        "facility_id": "FAC-001",
        "calculation_method": "AVERAGE_EMISSION_FACTOR",
    })

    # Register a survey
    service.register_survey({
        "survey_type": "OGI",
        "facility_id": "FAC-001",
        "survey_date": "2026-01-15",
        "components_surveyed": 100,
        "leaks_found": 3,
    })

    # Register an emission factor
    service.register_factor({
        "source_type": "EQUIPMENT_LEAK",
        "component_type": "valve",
        "gas": "CH4",
        "value": 0.00597,
        "source": "EPA",
    })

    # Register a repair
    service.register_repair({
        "component_id": "COMP-INT-001",
        "repair_type": "minor",
        "repair_date": "2026-02-01",
    })

    return service


# ---------------------------------------------------------------------------
# Request fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def equipment_leak_request() -> Dict[str, Any]:
    """Standard EQUIPMENT_LEAK average emission factor request."""
    return {
        "source_type": "EQUIPMENT_LEAK",
        "facility_id": "FAC-INT-001",
        "calculation_method": "AVERAGE_EMISSION_FACTOR",
        "component_counts": {"valve|gas": 100, "pump|light_liquid": 20},
        "gas_composition_ch4": 0.95,
        "gas_composition_voc": 0.02,
        "gwp_source": "AR6",
        "operating_hours": 8760,
    }


@pytest.fixture
def coal_mine_request() -> Dict[str, Any]:
    """COAL_MINE_METHANE engineering estimate request."""
    return {
        "source_type": "COAL_MINE_METHANE",
        "facility_id": "FAC-MINE-INT-001",
        "calculation_method": "AVERAGE_EMISSION_FACTOR",
        "coal_production_tonnes": 50000,
        "coal_rank": "BITUMINOUS",
        "mine_type": "UNDERGROUND",
        "gwp_source": "AR6",
    }


@pytest.fixture
def wastewater_request() -> Dict[str, Any]:
    """WASTEWATER treatment engineering estimate request."""
    return {
        "source_type": "WASTEWATER",
        "facility_id": "FAC-WW-INT-001",
        "calculation_method": "AVERAGE_EMISSION_FACTOR",
        "bod_load_kg": 100000,
        "treatment_type": "ANAEROBIC_LAGOON",
        "nitrogen_load_kg": 5000,
        "gwp_source": "AR6",
    }


@pytest.fixture
def pneumatic_request() -> Dict[str, Any]:
    """PNEUMATIC_DEVICE inventory-based request."""
    return {
        "source_type": "PNEUMATIC_DEVICE",
        "facility_id": "FAC-PD-INT-001",
        "calculation_method": "AVERAGE_EMISSION_FACTOR",
        "device_counts": {"high_bleed": 10, "low_bleed": 25, "intermittent": 15},
        "gwp_source": "AR6",
    }


@pytest.fixture
def tank_loss_request() -> Dict[str, Any]:
    """TANK_LOSS storage loss request (AP-42 parameters)."""
    return {
        "source_type": "TANK_LOSS",
        "facility_id": "FAC-TK-INT-001",
        "calculation_method": "AVERAGE_EMISSION_FACTOR",
        "tank_type": "fixed_roof_vertical",
        "tank_parameters": {
            "tank_id": "TK-INT-001",
            "diameter_ft": 50.0,
            "height_ft": 40.0,
            "liquid_height_ft": 20.0,
            "vapor_pressure_psia": 1.5,
            "molecular_weight": 68.0,
            "annual_throughput_gal": 500000,
        },
        "gwp_source": "AR6",
    }


@pytest.fixture
def batch_requests(
    equipment_leak_request,
    coal_mine_request,
    wastewater_request,
    pneumatic_request,
    tank_loss_request,
) -> List[Dict[str, Any]]:
    """A heterogeneous list of 5 calculation requests for batch testing."""
    return [
        equipment_leak_request,
        coal_mine_request,
        wastewater_request,
        pneumatic_request,
        tank_loss_request,
    ]


# ---------------------------------------------------------------------------
# Component fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_components() -> List[Dict[str, Any]]:
    """A set of 10 equipment components across types and facilities."""
    return [
        {
            "tag_number": "V-INT-001",
            "component_type": "valve",
            "service_type": "gas",
            "facility_id": "FAC-001",
        },
        {
            "tag_number": "V-INT-002",
            "component_type": "valve",
            "service_type": "gas",
            "facility_id": "FAC-001",
        },
        {
            "tag_number": "V-INT-003",
            "component_type": "valve",
            "service_type": "light_liquid",
            "facility_id": "FAC-001",
        },
        {
            "tag_number": "P-INT-001",
            "component_type": "pump",
            "service_type": "light_liquid",
            "facility_id": "FAC-001",
        },
        {
            "tag_number": "P-INT-002",
            "component_type": "pump",
            "service_type": "heavy_liquid",
            "facility_id": "FAC-001",
        },
        {
            "tag_number": "C-INT-001",
            "component_type": "compressor",
            "service_type": "gas",
            "facility_id": "FAC-001",
        },
        {
            "tag_number": "F-INT-001",
            "component_type": "connector",
            "service_type": "gas",
            "facility_id": "FAC-001",
        },
        {
            "tag_number": "F-INT-002",
            "component_type": "connector",
            "service_type": "gas",
            "facility_id": "FAC-001",
        },
        {
            "tag_number": "F-INT-003",
            "component_type": "connector",
            "service_type": "gas",
            "facility_id": "FAC-002",
        },
        {
            "tag_number": "OEL-INT-001",
            "component_type": "open_ended_line",
            "service_type": "gas",
            "facility_id": "FAC-002",
        },
    ]


# ---------------------------------------------------------------------------
# Survey fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_survey() -> Dict[str, Any]:
    """An LDAR survey registration dict."""
    return {
        "survey_type": "OGI",
        "facility_id": "FAC-001",
        "survey_date": "2026-03-15",
        "components_surveyed": 200,
        "leaks_found": 5,
        "threshold_ppm": 500,
    }


# ---------------------------------------------------------------------------
# Factor fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_factor() -> Dict[str, Any]:
    """A custom emission factor dict."""
    return {
        "source_type": "EQUIPMENT_LEAK",
        "component_type": "valve",
        "gas": "CH4",
        "value": 0.00597,
        "source": "EPA",
    }


# ---------------------------------------------------------------------------
# Repair fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_repair_data() -> Dict[str, Any]:
    """Repair registration dict."""
    return {
        "component_id": "COMP-INT-001",
        "repair_type": "minor",
        "repair_date": "2026-03-15",
        "pre_repair_rate_ppm": 15000,
        "post_repair_rate_ppm": 50,
        "cost_usd": 250.0,
        "notes": "Tightened valve packing gland",
    }


# ---------------------------------------------------------------------------
# Compliance fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def compliance_frameworks() -> List[str]:
    """All 7 regulatory frameworks supported by AGENT-MRV-005."""
    return [
        "GHG_PROTOCOL",
        "ISO_14064",
        "CSRD",
        "EPA_SUBPART_W",
        "EPA_LDAR",
        "EU_METHANE_REG",
        "UK_SECR",
    ]


# ---------------------------------------------------------------------------
# Uncertainty fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def uncertainty_config() -> Dict[str, Any]:
    """Uncertainty analysis configuration dict."""
    return {
        "calculation_id": "",  # filled in test after a calculation
        "method": "monte_carlo",
        "iterations": 1000,
    }
