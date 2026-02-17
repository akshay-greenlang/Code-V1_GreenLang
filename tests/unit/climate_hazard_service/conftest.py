# -*- coding: utf-8 -*-
"""
Shared pytest fixtures for Climate Hazard Connector unit tests.

Provides reusable fixtures for configuration, provenance tracking,
sample models, and test cleanup across all test modules in this
test package.

AGENT-DATA-020: Climate Hazard Connector
"""

from __future__ import annotations

import hashlib
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Generator
from unittest.mock import patch

import pytest

from greenlang.climate_hazard.config import (
    ClimateHazardConfig,
    reset_config,
    set_config,
)
from greenlang.climate_hazard.provenance import (
    ProvenanceEntry,
    ProvenanceTracker,
    reset_provenance_tracker,
)


# ---------------------------------------------------------------------------
# Environment variable cleanup
# ---------------------------------------------------------------------------

_ENV_VARS = [
    "GL_CLIMATE_HAZARD_DATABASE_URL",
    "GL_CLIMATE_HAZARD_REDIS_URL",
    "GL_CLIMATE_HAZARD_LOG_LEVEL",
    "GL_CLIMATE_HAZARD_DEFAULT_SCENARIO",
    "GL_CLIMATE_HAZARD_DEFAULT_TIME_HORIZON",
    "GL_CLIMATE_HAZARD_DEFAULT_REPORT_FORMAT",
    "GL_CLIMATE_HAZARD_MAX_HAZARD_SOURCES",
    "GL_CLIMATE_HAZARD_MAX_ASSETS",
    "GL_CLIMATE_HAZARD_MAX_RISK_INDICES",
    "GL_CLIMATE_HAZARD_RISK_WEIGHT_PROBABILITY",
    "GL_CLIMATE_HAZARD_RISK_WEIGHT_INTENSITY",
    "GL_CLIMATE_HAZARD_RISK_WEIGHT_FREQUENCY",
    "GL_CLIMATE_HAZARD_RISK_WEIGHT_DURATION",
    "GL_CLIMATE_HAZARD_VULN_WEIGHT_EXPOSURE",
    "GL_CLIMATE_HAZARD_VULN_WEIGHT_SENSITIVITY",
    "GL_CLIMATE_HAZARD_VULN_WEIGHT_ADAPTIVE",
    "GL_CLIMATE_HAZARD_THRESHOLD_EXTREME",
    "GL_CLIMATE_HAZARD_THRESHOLD_HIGH",
    "GL_CLIMATE_HAZARD_THRESHOLD_MEDIUM",
    "GL_CLIMATE_HAZARD_THRESHOLD_LOW",
    "GL_CLIMATE_HAZARD_MAX_PIPELINE_RUNS",
    "GL_CLIMATE_HAZARD_MAX_REPORTS",
    "GL_CLIMATE_HAZARD_ENABLE_PROVENANCE",
    "GL_CLIMATE_HAZARD_GENESIS_HASH",
    "GL_CLIMATE_HAZARD_ENABLE_METRICS",
    "GL_CLIMATE_HAZARD_POOL_SIZE",
    "GL_CLIMATE_HAZARD_CACHE_TTL",
    "GL_CLIMATE_HAZARD_RATE_LIMIT",
]


@pytest.fixture(autouse=True)
def clean_env() -> Generator[None, None, None]:
    """Remove all GL_CLIMATE_HAZARD_ env vars before and after each test.

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
def default_config() -> ClimateHazardConfig:
    """Return a ClimateHazardConfig with all default values."""
    return ClimateHazardConfig()


@pytest.fixture
def custom_config() -> ClimateHazardConfig:
    """Return a ClimateHazardConfig with non-default customizations."""
    return ClimateHazardConfig(
        database_url="postgresql://test:test@localhost:5432/testdb",
        redis_url="redis://localhost:6379/0",
        log_level="DEBUG",
        default_scenario="SSP5-8.5",
        default_time_horizon="LONG_TERM",
        default_report_format="csv",
        max_hazard_sources=100,
        max_assets=20000,
        max_risk_indices=10000,
        risk_weight_probability=0.25,
        risk_weight_intensity=0.25,
        risk_weight_frequency=0.25,
        risk_weight_duration=0.25,
        vuln_weight_exposure=0.40,
        vuln_weight_sensitivity=0.35,
        vuln_weight_adaptive=0.25,
        threshold_extreme=90.0,
        threshold_high=70.0,
        threshold_medium=50.0,
        threshold_low=30.0,
        max_pipeline_runs=1000,
        max_reports=5000,
        enable_provenance=True,
        genesis_hash="test-genesis",
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
    return ProvenanceTracker(genesis_hash="test-custom-genesis")


@pytest.fixture
def populated_tracker() -> ProvenanceTracker:
    """Return a ProvenanceTracker with several pre-recorded entries."""
    t = ProvenanceTracker(genesis_hash="test-populated")
    t.record("hazard_source", "register_source", "src_001", {"name": "NOAA"})
    t.record("hazard_data", "ingest_data", "data_001", {"count": 100})
    t.record("risk_index", "calculate_risk", "risk_001", {"score": 75.0})
    t.record("asset", "register_asset", "asset_001", {"type": "facility"})
    t.record("exposure", "assess_exposure", "exp_001", {"level": "high"})
    return t


@pytest.fixture
def sample_provenance_entry() -> ProvenanceEntry:
    """Return a single ProvenanceEntry for testing serialization."""
    return ProvenanceEntry(
        entity_type="hazard_source",
        entity_id="src_test_001",
        action="register_source",
        hash_value="a" * 64,
        parent_hash="b" * 64,
        timestamp="2026-02-17T00:00:00+00:00",
        metadata={"data_hash": "c" * 64, "extra": "value"},
    )


# ---------------------------------------------------------------------------
# Model helper fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_location_data() -> Dict[str, Any]:
    """Return minimal valid data for a Location model."""
    return {
        "latitude": 51.5074,
        "longitude": -0.1278,
    }


@pytest.fixture
def sample_location_full_data() -> Dict[str, Any]:
    """Return fully populated data for a Location model."""
    return {
        "latitude": 48.8566,
        "longitude": 2.3522,
        "elevation_m": 35.0,
        "name": "Paris",
        "country_code": "FR",
    }


@pytest.fixture
def utcnow() -> datetime:
    """Return a fixed UTC datetime for deterministic testing."""
    return datetime(2026, 2, 17, 0, 0, 0, tzinfo=timezone.utc)
