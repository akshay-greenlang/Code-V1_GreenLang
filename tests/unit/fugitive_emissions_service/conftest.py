# -*- coding: utf-8 -*-
"""
Shared fixtures for AGENT-MRV-005 Fugitive Emissions Agent unit tests.

Provides reusable test fixtures for configuration, sample calculation
requests, engine instances, and provenance tracking. The ``clean_env``
fixture is session-autouse and clears all ``GL_FUGITIVE_EMISSIONS_*``
environment variables between tests to prevent state leakage.

Fixtures (22):
    Environment:
        clean_env           - autouse, clears GL_FUGITIVE_EMISSIONS_* env vars
    Configuration:
        default_config      - FugitiveEmissionsConfig with all defaults
        custom_config       - FugitiveEmissionsConfig with custom overrides
    Provenance:
        tracker             - fresh ProvenanceTracker instance
        tracker_small       - ProvenanceTracker with low max_entries (5)
    Calculation:
        sample_calculation_request  - canonical VALVE_GAS avg EF request dict
        sample_batch_requests       - list of 5 diverse request dicts
    Engines:
        source_database_engine      - FugitiveSourceDatabaseEngine
        source_database_engine_ar5  - FugitiveSourceDatabaseEngine (AR5)
        emission_calculator_engine  - EmissionCalculatorEngine (or None)
        leak_detection_engine       - LeakDetectionEngine (or None)
        equipment_engine            - EquipmentComponentEngine (or None)
        uncertainty_engine          - UncertaintyQuantifierEngine (or None)
        compliance_engine           - ComplianceCheckerEngine (or None)
        pipeline_engine             - FugitiveEmissionsPipeline (or None)
    Mock Service:
        mock_service                - MagicMock with engine attributes

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-005 Fugitive Emissions (GL-MRV-SCOPE1-005)
"""

from __future__ import annotations

import os
import copy
from decimal import Decimal
from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Environment cleanup (autouse)
# ---------------------------------------------------------------------------

_ENV_PREFIX = "GL_FUGITIVE_EMISSIONS_"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Clear all GL_FUGITIVE_EMISSIONS_* environment variables before each test.

    This fixture is autouse so that every test in this directory starts
    with a clean slate. It also resets the config singleton to avoid
    cross-test contamination.
    """
    keys_to_remove = [
        key for key in os.environ if key.startswith(_ENV_PREFIX)
    ]
    for key in keys_to_remove:
        monkeypatch.delenv(key, raising=False)

    # Reset the config singleton so each test gets a fresh one
    try:
        from greenlang.fugitive_emissions.config import reset_config
        reset_config()
    except ImportError:
        pass

    yield

    # Post-test cleanup
    try:
        from greenlang.fugitive_emissions.config import reset_config
        reset_config()
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Configuration fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_config():
    """Create a default FugitiveEmissionsConfig with all defaults."""
    from greenlang.fugitive_emissions.config import FugitiveEmissionsConfig
    return FugitiveEmissionsConfig()


@pytest.fixture
def custom_config():
    """Create a FugitiveEmissionsConfig with custom overrides for testing."""
    from greenlang.fugitive_emissions.config import FugitiveEmissionsConfig
    return FugitiveEmissionsConfig(
        enabled=True,
        max_batch_size=100,
        default_gwp_source="AR5",
        default_calculation_method="DIRECT_MEASUREMENT",
        default_emission_factor_source="IPCC",
        decimal_precision=6,
        monte_carlo_iterations=1000,
        monte_carlo_seed=99,
        confidence_levels="90,95",
        enable_ldar_tracking=False,
        enable_component_tracking=False,
        enable_coal_mine_methane=False,
        enable_wastewater=False,
        enable_tank_losses=False,
        enable_pneumatic_devices=False,
        enable_compliance_checking=False,
        enable_uncertainty=False,
        enable_provenance=False,
        enable_metrics=False,
        max_components=1000,
        max_surveys=500,
        ldar_leak_threshold_ppm=5000,
        cache_ttl_seconds=1800,
        api_prefix="/api/v2/fugitive",
        api_max_page_size=50,
        api_default_page_size=10,
        log_level="DEBUG",
        worker_threads=2,
        enable_background_tasks=False,
        health_check_interval=60,
        genesis_hash="TEST-GENESIS-HASH",
    )


# ---------------------------------------------------------------------------
# Calculation request fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_calculation_request() -> Dict[str, Any]:
    """Create a sample VALVE_GAS average emission factor request with 100 valves.

    This is the canonical 'happy path' test fixture representing:
    - Source type: VALVE_GAS (equipment leak from gas service valve)
    - Method: AVERAGE_EMISSION_FACTOR (EPA Tier 1)
    - Activity data: 100 valves operating 8760 hr/yr
    - Gas composition: 95% CH4 (default pipeline gas)
    - GWP source: AR6
    """
    return {
        "source_type": "valve_gas",
        "calculation_method": "AVERAGE_EMISSION_FACTOR",
        "activity_data": 100,
        "activity_unit": "count",
        "component_counts": {"valve:gas": 100},
        "gas_composition_ch4": 0.95,
        "gas_composition_voc": 0.02,
        "gwp_source": "AR6",
        "ef_source": "EPA",
        "operating_hours": 8760,
        "facility_id": "FAC-TEST-001",
        "reporting_period": "annual",
    }


@pytest.fixture
def sample_batch_requests() -> List[Dict[str, Any]]:
    """Create a batch of 5 different calculation request types.

    Returns a list covering:
    1. Valve gas service (average EF)
    2. Pump light liquid (screening ranges)
    3. Compressor gas (correlation equation)
    4. Coal mine underground (engineering estimate)
    5. Wastewater lagoon (engineering estimate)
    """
    return [
        {
            "source_type": "valve_gas",
            "calculation_method": "AVERAGE_EMISSION_FACTOR",
            "activity_data": 100,
            "activity_unit": "count",
            "component_counts": {"valve:gas": 100},
            "gas_composition_ch4": 0.95,
            "gwp_source": "AR6",
            "ef_source": "EPA",
        },
        {
            "source_type": "pump_seal",
            "calculation_method": "SCREENING_RANGES",
            "activity_data": 25,
            "activity_unit": "count",
            "component_counts": {"pump:light_liquid": 25},
            "gas_composition_ch4": 0.90,
            "screening_value_ppm": 15000,
            "gwp_source": "AR6",
            "ef_source": "EPA",
        },
        {
            "source_type": "compressor_seal",
            "calculation_method": "CORRELATION_EQUATION",
            "activity_data": 5,
            "activity_unit": "count",
            "component_counts": {"compressor:gas": 5},
            "gas_composition_ch4": 0.95,
            "screening_value_ppm": 50000,
            "gwp_source": "AR6",
            "ef_source": "EPA",
        },
        {
            "source_type": "coal_mine_underground",
            "calculation_method": "ENGINEERING_ESTIMATE",
            "activity_data": 10000,
            "activity_unit": "tonnes",
            "coal_rank": "BITUMINOUS",
            "gwp_source": "AR6",
        },
        {
            "source_type": "wastewater_lagoon",
            "calculation_method": "ENGINEERING_ESTIMATE",
            "activity_data": 5000,
            "activity_unit": "kg_bod",
            "treatment_type": "ANAEROBIC_LAGOON_DEEP",
            "gwp_source": "AR6",
        },
    ]


# ---------------------------------------------------------------------------
# Engine fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def source_database_engine():
    """Create a FugitiveSourceDatabaseEngine instance for testing."""
    from greenlang.fugitive_emissions.fugitive_source_database import (
        FugitiveSourceDatabaseEngine,
    )
    return FugitiveSourceDatabaseEngine()


@pytest.fixture
def source_database_engine_ar5():
    """Create a FugitiveSourceDatabaseEngine with AR5 GWP default."""
    from greenlang.fugitive_emissions.fugitive_source_database import (
        FugitiveSourceDatabaseEngine,
    )
    return FugitiveSourceDatabaseEngine(config={"default_gwp_source": "AR5"})


@pytest.fixture
def emission_calculator_engine():
    """Create a mock EmissionCalculatorEngine for testing.

    Returns None if the class is not available, allowing tests that
    do not require it to still pass.
    """
    try:
        from greenlang.fugitive_emissions.emission_calculator import (
            EmissionCalculatorEngine,
        )
        return EmissionCalculatorEngine()
    except (ImportError, Exception):
        return None


@pytest.fixture
def leak_detection_engine():
    """Create a mock LeakDetectionEngine for testing."""
    try:
        from greenlang.fugitive_emissions.leak_detection import (
            LeakDetectionEngine,
        )
        return LeakDetectionEngine()
    except (ImportError, Exception):
        return None


@pytest.fixture
def equipment_engine():
    """Create a mock EquipmentComponentEngine for testing."""
    try:
        from greenlang.fugitive_emissions.equipment_component import (
            EquipmentComponentEngine,
        )
        return EquipmentComponentEngine()
    except (ImportError, Exception):
        return None


@pytest.fixture
def uncertainty_engine():
    """Create a mock UncertaintyQuantifierEngine for testing."""
    try:
        from greenlang.fugitive_emissions.uncertainty_quantifier import (
            UncertaintyQuantifierEngine,
        )
        return UncertaintyQuantifierEngine()
    except (ImportError, Exception):
        return None


@pytest.fixture
def compliance_engine():
    """Create a mock ComplianceCheckerEngine for testing."""
    try:
        from greenlang.fugitive_emissions.compliance_checker import (
            ComplianceCheckerEngine,
        )
        return ComplianceCheckerEngine()
    except (ImportError, Exception):
        return None


@pytest.fixture
def pipeline_engine():
    """Create a mock FugitiveEmissionsPipeline for testing."""
    try:
        from greenlang.fugitive_emissions.fugitive_emissions_pipeline import (
            FugitiveEmissionsPipeline,
        )
        return FugitiveEmissionsPipeline()
    except (ImportError, Exception):
        return None


# ---------------------------------------------------------------------------
# Provenance tracker fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def tracker():
    """Create a fresh ProvenanceTracker instance for testing."""
    from greenlang.fugitive_emissions.provenance import ProvenanceTracker
    return ProvenanceTracker()


@pytest.fixture
def tracker_small():
    """Create a ProvenanceTracker with low max_entries for eviction testing."""
    from greenlang.fugitive_emissions.provenance import ProvenanceTracker
    return ProvenanceTracker(max_entries=5)


# ---------------------------------------------------------------------------
# Mock service fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_service():
    """Create a MagicMock service object with engine-like attributes.

    This mock emulates a high-level fugitive emissions service orchestrator
    with references to all sub-engines, useful for integration-style tests
    that need to verify inter-engine coordination without importing every
    real engine class.
    """
    service = MagicMock()
    service.source_database = MagicMock()
    service.emission_calculator = MagicMock()
    service.leak_detection = MagicMock()
    service.equipment_component = MagicMock()
    service.uncertainty_quantifier = MagicMock()
    service.compliance_checker = MagicMock()
    service.pipeline = MagicMock()
    service.provenance_tracker = MagicMock()
    service.config = MagicMock()
    service.config.enabled = True
    service.config.default_gwp_source = "AR6"
    service.config.default_calculation_method = "AVERAGE_EMISSION_FACTOR"
    service.config.default_emission_factor_source = "EPA"
    service.config.max_batch_size = 500
    service.config.decimal_precision = 8
    service.config.enable_provenance = True
    service.config.enable_metrics = True
    return service
