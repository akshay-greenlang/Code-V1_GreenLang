# -*- coding: utf-8 -*-
"""Shared fixtures for Process Emissions Agent tests.

Provides reusable pytest fixtures for configuration, engine instances,
sample data, and mock objects used across test_config, test_models,
and test_metrics test modules.

Agent: AGENT-MRV-004 (GL-MRV-SCOPE1-004)
"""

import os
import pytest
from unittest.mock import MagicMock, patch
from decimal import Decimal


# ---------------------------------------------------------------------------
# Environment hygiene
# ---------------------------------------------------------------------------

_GL_PE_PREFIX = "GL_PROCESS_EMISSIONS_"


@pytest.fixture(autouse=True, scope="session")
def clean_env():
    """Remove all GL_PROCESS_EMISSIONS_* env vars before the test session.

    This prevents developer-local or CI-level env vars from leaking into
    tests and causing non-deterministic behaviour.  Runs once per session.
    """
    keys_to_remove = [
        key for key in os.environ if key.startswith(_GL_PE_PREFIX)
    ]
    saved = {}
    for key in keys_to_remove:
        saved[key] = os.environ.pop(key)
    yield
    # Restore any removed keys after session
    for key, val in saved.items():
        os.environ[key] = val


@pytest.fixture(autouse=True)
def reset_config_singleton():
    """Reset the config singleton before and after every test.

    Prevents state leakage between tests that call get_config(),
    set_config(), or modify environment variables.  Also cleans up
    any GL_PROCESS_EMISSIONS_* env vars created during the test to
    prevent cross-test contamination.
    """
    from greenlang.process_emissions.config import reset_config
    reset_config()
    # Snapshot env vars before the test
    pre_keys = {k for k in os.environ if k.startswith(_GL_PE_PREFIX)}
    yield
    reset_config()
    # Remove any GL_PROCESS_EMISSIONS_* env vars that were created
    # during the test to prevent leakage into subsequent tests
    post_keys = {k for k in os.environ if k.startswith(_GL_PE_PREFIX)}
    for key in post_keys - pre_keys:
        os.environ.pop(key, None)


@pytest.fixture(autouse=True)
def reset_provenance_singleton():
    """Reset the provenance tracker singleton before and after every test."""
    from greenlang.process_emissions.provenance import reset_provenance_tracker
    reset_provenance_tracker()
    yield
    reset_provenance_tracker()


# ---------------------------------------------------------------------------
# Configuration fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def default_config():
    """Return a ProcessEmissionsConfig with all default values."""
    from greenlang.process_emissions.config import ProcessEmissionsConfig
    return ProcessEmissionsConfig()


@pytest.fixture()
def custom_config():
    """Return a ProcessEmissionsConfig with non-default values.

    Useful for testing that overrides propagate correctly.
    """
    from greenlang.process_emissions.config import ProcessEmissionsConfig
    return ProcessEmissionsConfig(
        enabled=False,
        database_url="postgresql://test:test@localhost:5432/pe_test",
        redis_url="redis://localhost:6379/5",
        max_batch_size=250,
        default_gwp_source="AR5",
        default_calculation_tier="TIER_2",
        default_calculation_method="MASS_BALANCE",
        default_emission_factor_source="IPCC",
        decimal_precision=12,
        monte_carlo_iterations=10_000,
        monte_carlo_seed=99,
        confidence_levels="90,95",
        enable_mass_balance=False,
        enable_abatement_tracking=False,
        enable_by_product_credits=False,
        enable_compliance_checking=False,
        enable_uncertainty=False,
        enable_provenance=False,
        enable_metrics=False,
        max_material_inputs=25,
        max_process_units=100,
        max_abatement_records=50,
        cache_ttl_seconds=1800,
        api_prefix="/api/v2/process-emissions",
        api_max_page_size=50,
        api_default_page_size=10,
        log_level="DEBUG",
        worker_threads=8,
        enable_background_tasks=False,
        health_check_interval=60,
        genesis_hash="TEST-GENESIS-HASH",
    )


# ---------------------------------------------------------------------------
# Provenance fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tracker():
    """Return a fresh ProvenanceTracker instance (not the singleton)."""
    from greenlang.process_emissions.provenance import ProvenanceTracker
    return ProvenanceTracker(
        genesis_hash="TEST-PROCESS-EMISSIONS-GENESIS",
        max_entries=500,
    )


# ---------------------------------------------------------------------------
# Sample data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_calculation_request():
    """Return a dict representing a typical single calculation request.

    Uses cement production as the canonical example because it is the
    most common industrial process emission type.
    """
    return {
        "process_type": "CEMENT_PRODUCTION",
        "activity_data": 1_000_000,
        "activity_unit": "tonne_clinker",
        "calculation_method": "EMISSION_FACTOR",
        "calculation_tier": "TIER_1",
        "gwp_source": "AR6",
        "ef_source": "IPCC",
        "production_quantity_tonnes": 1_000_000.0,
        "facility_id": "FAC-001",
        "unit_id": "PU-KILN-001",
        "period_start": "2025-01-01T00:00:00+00:00",
        "period_end": "2025-12-31T23:59:59+00:00",
    }


@pytest.fixture()
def sample_batch_requests():
    """Return a list of 5 different process type calculation request dicts.

    Covers one representative from each of the major process categories:
    mineral, chemical, metal, electronics, other.
    """
    base_period_start = "2025-01-01T00:00:00+00:00"
    base_period_end = "2025-12-31T23:59:59+00:00"
    return [
        {
            "process_type": "CEMENT_PRODUCTION",
            "production_quantity_tonnes": 500_000.0,
            "calculation_method": "EMISSION_FACTOR",
            "calculation_tier": "TIER_1",
            "gwp_source": "AR6",
            "ef_source": "IPCC",
            "period_start": base_period_start,
            "period_end": base_period_end,
        },
        {
            "process_type": "NITRIC_ACID",
            "production_quantity_tonnes": 100_000.0,
            "calculation_method": "EMISSION_FACTOR",
            "calculation_tier": "TIER_1",
            "gwp_source": "AR6",
            "ef_source": "EPA",
            "period_start": base_period_start,
            "period_end": base_period_end,
        },
        {
            "process_type": "IRON_STEEL",
            "production_quantity_tonnes": 200_000.0,
            "calculation_method": "MASS_BALANCE",
            "calculation_tier": "TIER_2",
            "gwp_source": "AR6",
            "ef_source": "IPCC",
            "period_start": base_period_start,
            "period_end": base_period_end,
        },
        {
            "process_type": "SEMICONDUCTOR",
            "production_quantity_tonnes": 50.0,
            "calculation_method": "EMISSION_FACTOR",
            "calculation_tier": "TIER_1",
            "gwp_source": "AR6",
            "ef_source": "EPA",
            "period_start": base_period_start,
            "period_end": base_period_end,
        },
        {
            "process_type": "FOOD_DRINK",
            "production_quantity_tonnes": 10_000.0,
            "calculation_method": "EMISSION_FACTOR",
            "calculation_tier": "TIER_1",
            "gwp_source": "AR5",
            "ef_source": "DEFRA",
            "period_start": base_period_start,
            "period_end": base_period_end,
        },
    ]


@pytest.fixture()
def sample_materials():
    """Return a list of MaterialInput-like dicts for mass balance testing.

    Includes carbonates (limestone, dolomite), carbon sources (coke),
    and metal ores (iron ore, bauxite).
    """
    return [
        {
            "material_type": "limestone",
            "quantity_tonnes": 1_300_000.0,
            "carbon_content_fraction": 0.12,
            "carbonate_type": "calcite",
            "purity_fraction": 0.95,
            "moisture_fraction": 0.02,
        },
        {
            "material_type": "dolomite",
            "quantity_tonnes": 50_000.0,
            "carbon_content_fraction": 0.13,
            "carbonate_type": "dolomite",
            "purity_fraction": 0.90,
            "moisture_fraction": 0.03,
        },
        {
            "material_type": "coke",
            "quantity_tonnes": 20_000.0,
            "carbon_content_fraction": 0.85,
            "carbonate_type": None,
            "purity_fraction": 1.0,
            "moisture_fraction": 0.0,
        },
        {
            "material_type": "iron_ore",
            "quantity_tonnes": 500_000.0,
            "carbon_content_fraction": 0.0,
            "carbonate_type": None,
            "purity_fraction": 0.65,
            "moisture_fraction": 0.05,
        },
        {
            "material_type": "bauxite",
            "quantity_tonnes": 100_000.0,
            "carbon_content_fraction": 0.0,
            "carbonate_type": None,
            "purity_fraction": 0.55,
            "moisture_fraction": 0.08,
        },
    ]


# ---------------------------------------------------------------------------
# Engine fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def process_database_engine():
    """Return a ProcessDatabaseEngine instance with default config."""
    from greenlang.process_emissions.process_database import ProcessDatabaseEngine
    return ProcessDatabaseEngine()


@pytest.fixture()
def emission_calculator_engine(process_database_engine):
    """Return an EmissionCalculatorEngine wired to the process database."""
    from greenlang.process_emissions.emission_calculator import EmissionCalculatorEngine
    return EmissionCalculatorEngine(process_database=process_database_engine)


@pytest.fixture()
def material_balance_engine():
    """Return a MaterialBalanceEngine with default precision."""
    from greenlang.process_emissions.material_balance import MaterialBalanceEngine
    return MaterialBalanceEngine(precision=8)


@pytest.fixture()
def abatement_tracker_engine():
    """Return an AbatementTrackerEngine with default config."""
    from greenlang.process_emissions.abatement_tracker import AbatementTrackerEngine
    return AbatementTrackerEngine()


@pytest.fixture()
def uncertainty_engine():
    """Return an UncertaintyQuantifierEngine instance."""
    from greenlang.process_emissions.uncertainty_quantifier import UncertaintyQuantifierEngine
    return UncertaintyQuantifierEngine()


@pytest.fixture()
def compliance_engine():
    """Return a ComplianceCheckerEngine instance."""
    from greenlang.process_emissions.compliance_checker import ComplianceCheckerEngine
    return ComplianceCheckerEngine()


@pytest.fixture()
def pipeline_engine():
    """Return a ProcessEmissionsPipelineEngine with all engines wired."""
    from greenlang.process_emissions.process_emissions_pipeline import (
        ProcessEmissionsPipelineEngine,
    )
    return ProcessEmissionsPipelineEngine()


@pytest.fixture()
def mock_service():
    """Return a MagicMock standing in for ProcessEmissionsService.

    Pre-configures commonly accessed attributes and methods so that
    tests can focus on interactions without instantiating real engines.
    """
    service = MagicMock()
    service.config = MagicMock()
    service.config.enabled = True
    service.config.default_gwp_source = "AR6"
    service.config.default_calculation_tier = "TIER_1"
    service.config.default_calculation_method = "EMISSION_FACTOR"
    service.config.default_emission_factor_source = "EPA"
    service.config.decimal_precision = 8
    service.config.max_batch_size = 500
    service.config.enable_mass_balance = True
    service.config.enable_abatement_tracking = True
    service.config.enable_by_product_credits = True
    service.config.enable_compliance_checking = True
    service.config.enable_uncertainty = True
    service.config.enable_provenance = True
    service.config.enable_metrics = True

    # Mock methods return sane defaults
    service.calculate.return_value = {
        "success": True,
        "total_co2e_tonnes": 440_000.0,
    }
    service.calculate_batch.return_value = {
        "success": True,
        "calculation_count": 5,
        "total_co2e_tonnes": 1_200_000.0,
    }
    service.health_check.return_value = {"status": "healthy"}
    return service
