# -*- coding: utf-8 -*-
"""
Shared fixtures for Process Emissions Agent integration tests.

AGENT-MRV-004: Process Emissions Agent (GL-MRV-SCOPE1-004)

Provides reusable fixtures for:
- NetworkBlocker: Ensures no external network calls during tests
- Mock agents: MagicMock-based fakes for upstream engine dependencies
- Full service and pipeline instances wired for integration testing
- Sample data generators for multi-stage pipeline workflows
"""

from __future__ import annotations

import os
import socket
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Network safety
# ---------------------------------------------------------------------------


class NetworkBlocker:
    """Context manager that blocks all outgoing network connections.

    Prevents integration tests from making real HTTP, gRPC, or database
    calls.  Monkeypatches ``socket.socket.connect`` and
    ``socket.socket.connect_ex`` to raise ``ConnectionRefusedError``.
    """

    _original_connect = None
    _original_connect_ex = None

    def __enter__(self):
        self._original_connect = socket.socket.connect
        self._original_connect_ex = socket.socket.connect_ex

        def _blocked_connect(*args, **kwargs):
            raise ConnectionRefusedError(
                "Network access is blocked in integration tests"
            )

        socket.socket.connect = _blocked_connect
        socket.socket.connect_ex = _blocked_connect
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._original_connect is not None:
            socket.socket.connect = self._original_connect
        if self._original_connect_ex is not None:
            socket.socket.connect_ex = self._original_connect_ex
        return False


@pytest.fixture(autouse=True, scope="session")
def block_network():
    """Block all network access for the entire integration test session."""
    with NetworkBlocker():
        yield


# ---------------------------------------------------------------------------
# Environment hygiene
# ---------------------------------------------------------------------------


_GL_PE_PREFIX = "GL_PROCESS_EMISSIONS_"


@pytest.fixture(autouse=True, scope="session")
def clean_env():
    """Remove all GL_PROCESS_EMISSIONS_* env vars before tests."""
    keys_to_remove = [
        key for key in os.environ if key.startswith(_GL_PE_PREFIX)
    ]
    saved = {}
    for key in keys_to_remove:
        saved[key] = os.environ.pop(key)
    yield
    for key, val in saved.items():
        os.environ[key] = val


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset config and provenance singletons between tests."""
    from greenlang.process_emissions.config import reset_config
    from greenlang.process_emissions.provenance import reset_provenance_tracker
    reset_config()
    reset_provenance_tracker()
    yield
    reset_config()
    reset_provenance_tracker()


# ---------------------------------------------------------------------------
# Engine fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def process_database_engine():
    """Real ProcessDatabaseEngine for integration testing."""
    from greenlang.process_emissions.process_database import (
        ProcessDatabaseEngine,
    )
    return ProcessDatabaseEngine()


@pytest.fixture
def emission_calculator_engine(process_database_engine):
    """Real EmissionCalculatorEngine wired to the process database."""
    from greenlang.process_emissions.emission_calculator import (
        EmissionCalculatorEngine,
    )
    return EmissionCalculatorEngine(
        process_database=process_database_engine,
    )


@pytest.fixture
def abatement_tracker_engine():
    """Real AbatementTrackerEngine for integration testing."""
    from greenlang.process_emissions.abatement_tracker import (
        AbatementTrackerEngine,
    )
    return AbatementTrackerEngine()


@pytest.fixture
def uncertainty_engine():
    """Real UncertaintyQuantifierEngine for integration testing."""
    from greenlang.process_emissions.uncertainty_quantifier import (
        UncertaintyQuantifierEngine,
    )
    return UncertaintyQuantifierEngine()


@pytest.fixture
def compliance_engine():
    """Real ComplianceCheckerEngine for integration testing."""
    from greenlang.process_emissions.compliance_checker import (
        ComplianceCheckerEngine,
    )
    return ComplianceCheckerEngine()


@pytest.fixture
def pipeline_engine(
    process_database_engine,
    emission_calculator_engine,
    abatement_tracker_engine,
    uncertainty_engine,
    compliance_engine,
):
    """Real ProcessEmissionsPipelineEngine with all engines wired."""
    from greenlang.process_emissions.process_emissions_pipeline import (
        ProcessEmissionsPipelineEngine,
    )
    return ProcessEmissionsPipelineEngine(
        process_database=process_database_engine,
        emission_calculator=emission_calculator_engine,
        abatement_tracker=abatement_tracker_engine,
        uncertainty_engine=uncertainty_engine,
        compliance_checker=compliance_engine,
    )


@pytest.fixture
def service():
    """Real ProcessEmissionsService facade for integration testing."""
    from greenlang.process_emissions.setup import ProcessEmissionsService
    return ProcessEmissionsService()


# ---------------------------------------------------------------------------
# Sample data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cement_pipeline_request() -> Dict[str, Any]:
    """Full cement production pipeline request."""
    return {
        "process_type": "CEMENT_CLINKER",
        "production_quantity": 100000.0,
        "production_unit": "tonne",
        "calculation_method": "EMISSION_FACTOR",
        "facility_id": "INT-FAC-001",
        "period_start": "2025-01-01",
        "period_end": "2025-12-31",
    }


@pytest.fixture
def iron_steel_pipeline_request() -> Dict[str, Any]:
    """Full iron/steel BF-BOF pipeline request."""
    return {
        "process_type": "IRON_STEEL_BF_BOF",
        "production_quantity": 50000.0,
        "production_unit": "tonne",
        "calculation_method": "MASS_BALANCE",
        "production_route": "BF_BOF",
        "facility_id": "INT-FAC-002",
        "period_start": "2025-01-01",
        "period_end": "2025-12-31",
    }


@pytest.fixture
def semiconductor_pipeline_request() -> Dict[str, Any]:
    """Full semiconductor manufacturing pipeline request."""
    return {
        "process_type": "SEMICONDUCTOR_MANUFACTURING",
        "production_quantity": 10000.0,
        "production_unit": "wafer_start",
        "calculation_method": "EMISSION_FACTOR",
        "facility_id": "INT-FAC-003",
        "period_start": "2025-01-01",
        "period_end": "2025-12-31",
    }


@pytest.fixture
def nitric_acid_pipeline_request() -> Dict[str, Any]:
    """Full nitric acid pipeline request."""
    return {
        "process_type": "NITRIC_ACID",
        "production_quantity": 80000.0,
        "production_unit": "tonne",
        "calculation_method": "EMISSION_FACTOR",
        "facility_id": "INT-FAC-004",
        "period_start": "2025-01-01",
        "period_end": "2025-12-31",
    }


@pytest.fixture
def aluminium_pipeline_request() -> Dict[str, Any]:
    """Full aluminium prebake pipeline request."""
    return {
        "process_type": "ALUMINIUM_PREBAKE",
        "production_quantity": 25000.0,
        "production_unit": "tonne",
        "calculation_method": "EMISSION_FACTOR",
        "production_route": "PREBAKE",
        "facility_id": "INT-FAC-005",
        "period_start": "2025-01-01",
        "period_end": "2025-12-31",
    }


@pytest.fixture
def multi_process_batch_requests() -> List[Dict[str, Any]]:
    """Batch of diverse process type requests for integration tests."""
    return [
        {
            "process_type": "CEMENT_CLINKER",
            "production_quantity": 50000.0,
        },
        {
            "process_type": "LIME_PRODUCTION",
            "production_quantity": 20000.0,
        },
        {
            "process_type": "AMMONIA_PRODUCTION",
            "production_quantity": 30000.0,
        },
        {
            "process_type": "IRON_STEEL_BF_BOF",
            "production_quantity": 10000.0,
        },
        {
            "process_type": "ALUMINIUM_PREBAKE",
            "production_quantity": 5000.0,
        },
    ]


@pytest.fixture
def compliance_check_data() -> Dict[str, Any]:
    """Data dictionary suitable for compliance checking."""
    return {
        "process_type": "CEMENT_PRODUCTION",
        "calculation_method": "EMISSION_FACTOR",
        "tier": "TIER_2",
        "total_co2e_tonnes": 52500.0,
        "emissions_by_gas": [{"gas": "CO2", "co2e_tonnes": 52500.0}],
        "emission_factor_source": "IPCC",
        "gwp_source": "AR6",
        "provenance_hash": "integration_test_hash_" + "a" * 48,
        "calculation_trace": [{"step": "emission_factor_lookup"}],
        "facility_id": "INT-FAC-001",
        "organization_id": "INT-ORG-001",
        "production_quantity_tonnes": 100000.0,
        "period_start": "2025-01-01",
        "period_end": "2025-12-31",
    }
