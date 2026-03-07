# -*- coding: utf-8 -*-
"""
Shared pytest fixtures for AGENT-EUDR-002 Geolocation Verification test suite.

Provides reusable fixtures for coordinate inputs, polygon inputs,
verification engine instances, mock configurations, and shared
test data used across all test modules.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-002 Geolocation Verification Agent (GL-EUDR-GEO-002)
"""

import hashlib
import json
import math
import uuid
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from greenlang.agents.eudr.geolocation_verification.config import (
    GeolocationVerificationConfig,
    get_config,
    set_config,
    reset_config,
)
from greenlang.agents.eudr.geolocation_verification.models import (
    BoundaryChange,
    ChangeType,
    CoordinateInput,
    CoordinateValidationResult,
    DeforestationVerificationResult,
    GeolocationAccuracyScore,
    IssueSeverity,
    PolygonInput,
    PolygonVerificationResult,
    ProtectedAreaCheckResult,
    QualityTier,
    RepairSuggestion,
    TemporalChangeResult,
    ValidationIssue,
)
from greenlang.agents.eudr.geolocation_verification.coordinate_validator import (
    CoordinateValidator,
)
from greenlang.agents.eudr.geolocation_verification.polygon_verifier import (
    PolygonTopologyVerifier,
)
from greenlang.agents.eudr.geolocation_verification.protected_area_checker import (
    ProtectedAreaChecker,
)
from greenlang.agents.eudr.geolocation_verification.deforestation_verifier import (
    DeforestationCutoffVerifier,
)
from greenlang.agents.eudr.geolocation_verification.accuracy_scorer import (
    AccuracyScoringEngine,
)
from greenlang.agents.eudr.geolocation_verification.temporal_analyzer import (
    TemporalConsistencyAnalyzer,
)
from greenlang.agents.eudr.geolocation_verification.batch_pipeline import (
    BatchVerificationPipeline,
)
from greenlang.agents.eudr.geolocation_verification.article9_reporter import (
    Article9ComplianceReporter,
)


# ---------------------------------------------------------------------------
# Deterministic UUID helper
# ---------------------------------------------------------------------------


class DeterministicUUID:
    """Generate sequential identifiers for deterministic testing."""

    def __init__(self, prefix: str = "test"):
        self._counter = 0
        self._prefix = prefix

    def next(self) -> str:
        self._counter += 1
        return f"{self._prefix}-{self._counter:08d}"

    def reset(self):
        self._counter = 0


# ---------------------------------------------------------------------------
# Configuration Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_config():
    """Create a GeolocationVerificationConfig with test defaults."""
    return GeolocationVerificationConfig(
        database_url="postgresql://localhost:5432/greenlang_test",
        redis_url="redis://localhost:6379/1",
        log_level="DEBUG",
        coordinate_precision_min_decimals=5,
        duplicate_distance_threshold_m=10.0,
        elevation_max_m=6000.0,
        country_boundary_buffer_km=5.0,
        polygon_area_tolerance_pct=10.0,
        max_polygon_vertices=100_000,
        min_polygon_vertices=4,
        sliver_ratio_threshold=0.001,
        spike_angle_threshold_degrees=1.0,
        wdpa_update_interval_days=90,
        deforestation_cutoff_date="2020-12-31",
        score_weights={
            "precision": 0.20,
            "polygon": 0.20,
            "country": 0.15,
            "protected": 0.15,
            "deforestation": 0.15,
            "temporal": 0.15,
        },
        max_batch_concurrency=10,
        quick_timeout_seconds=5.0,
        standard_timeout_seconds=30.0,
        deep_timeout_seconds=120.0,
        verification_cache_ttl_seconds=300,
        enable_provenance=True,
        genesis_hash="GL-EUDR-GEO-002-TEST-GENESIS",
        enable_metrics=False,
        pool_size=5,
        rate_limit=500,
    )


@pytest.fixture(autouse=True)
def reset_singleton_config():
    """Reset the singleton config after each test to avoid cross-test leaks."""
    yield
    reset_config()


@pytest.fixture
def uuid_gen():
    """Create a deterministic UUID generator."""
    return DeterministicUUID()


# ---------------------------------------------------------------------------
# Coordinate Input Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def valid_coordinate_brazil():
    """Valid coordinate in Para, Brazil (Amazon region)."""
    return CoordinateInput(
        lat=-3.1234567,
        lon=-60.0234567,
        declared_country="BR",
        commodity="cocoa",
        plot_id="PLOT-BR-001",
    )


@pytest.fixture
def valid_coordinate_indonesia():
    """Valid coordinate in Kalimantan, Indonesia."""
    return CoordinateInput(
        lat=-2.5678901,
        lon=111.7654321,
        declared_country="ID",
        commodity="oil_palm",
        plot_id="PLOT-ID-001",
    )


@pytest.fixture
def valid_coordinate_ghana():
    """Valid coordinate in Ashanti, Ghana."""
    return CoordinateInput(
        lat=6.1234567,
        lon=-1.6234567,
        declared_country="GH",
        commodity="cocoa",
        plot_id="PLOT-GH-001",
    )


@pytest.fixture
def invalid_coordinate_ocean():
    """Coordinate in the middle of the Atlantic Ocean (not on land)."""
    return CoordinateInput(
        lat=0.0,
        lon=-30.0,
        declared_country="BR",
        commodity="soya",
        plot_id="PLOT-OCEAN-001",
    )


@pytest.fixture
def coordinate_north_pole():
    """Coordinate at the North Pole (boundary)."""
    return CoordinateInput(lat=90.0, lon=0.0, declared_country="", plot_id="POLE-N")


@pytest.fixture
def coordinate_south_pole():
    """Coordinate at the South Pole (boundary)."""
    return CoordinateInput(lat=-90.0, lon=0.0, declared_country="", plot_id="POLE-S")


@pytest.fixture
def coordinate_dateline():
    """Coordinate on the International Date Line."""
    return CoordinateInput(lat=0.0, lon=180.0, declared_country="", plot_id="DATELINE")


@pytest.fixture
def coordinate_prime_meridian():
    """Coordinate on the Prime Meridian at the equator."""
    return CoordinateInput(lat=0.0, lon=0.0, declared_country="", plot_id="PRIME-MER")


# ---------------------------------------------------------------------------
# Polygon Input Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def valid_polygon_small():
    """Valid small triangle polygon (~2 ha) in Brazil.

    Triangle with vertices near -3.12, -60.02 area.
    Approximately 140m x 285m = ~2 ha at this latitude.
    CCW winding order. Ring is closed (first == last).
    """
    return PolygonInput(
        vertices=[
            (-3.1200, -60.0200),
            (-3.1200, -60.0180),
            (-3.1215, -60.0190),
            (-3.1200, -60.0200),  # closure
        ],
        declared_area_ha=2.0,
        commodity="cocoa",
        plot_id="POLY-SM-001",
    )


@pytest.fixture
def valid_polygon_large():
    """Valid larger pentagon polygon (~10 ha) in Indonesia.

    Pentagon shape near -2.57, 111.77 area.
    CCW winding order. Ring is closed.
    """
    return PolygonInput(
        vertices=[
            (-2.5700, 111.7700),
            (-2.5700, 111.7740),
            (-2.5720, 111.7750),
            (-2.5740, 111.7730),
            (-2.5730, 111.7700),
            (-2.5700, 111.7700),  # closure
        ],
        declared_area_ha=10.0,
        commodity="oil_palm",
        plot_id="POLY-LG-001",
    )


@pytest.fixture
def invalid_polygon_self_intersecting():
    """Self-intersecting polygon (bowtie / figure-eight shape).

    Vertices cross over each other, creating two separate lobes.
    """
    return PolygonInput(
        vertices=[
            (-3.1200, -60.0200),
            (-3.1200, -60.0180),
            (-3.1220, -60.0200),  # crosses previous edge
            (-3.1220, -60.0180),  # crosses first edge
            (-3.1200, -60.0200),  # closure
        ],
        declared_area_ha=2.0,
        commodity="cocoa",
        plot_id="POLY-SELF-INT",
    )


@pytest.fixture
def invalid_polygon_unclosed():
    """Polygon that is not properly closed (last vertex != first)."""
    return PolygonInput(
        vertices=[
            (-3.1200, -60.0200),
            (-3.1200, -60.0180),
            (-3.1215, -60.0190),
            # missing closure vertex
        ],
        declared_area_ha=2.0,
        commodity="cocoa",
        plot_id="POLY-UNCLOSED",
    )


@pytest.fixture
def polygon_degenerate_point():
    """Degenerate polygon with all vertices at the same point."""
    pt = (-3.12, -60.02)
    return PolygonInput(
        vertices=[pt, pt, pt, pt],
        declared_area_ha=0.0,
        commodity="cocoa",
        plot_id="POLY-DEGEN",
    )


@pytest.fixture
def polygon_sliver():
    """Very thin sliver polygon (extremely low area-to-perimeter ratio)."""
    return PolygonInput(
        vertices=[
            (-3.1200, -60.0200),
            (-3.1200, -60.0000),  # Very long edge
            (-3.12001, -60.0100),  # Very narrow
            (-3.1200, -60.0200),
        ],
        declared_area_ha=0.1,
        commodity="cocoa",
        plot_id="POLY-SLIVER",
    )


@pytest.fixture
def polygon_complex_20_vertices():
    """Complex polygon with 20 vertices forming a rough circle."""
    import math as _math

    center_lat, center_lon = -2.57, 111.77
    radius = 0.003  # ~330m at equator
    n = 20
    vertices = []
    for i in range(n):
        angle = 2 * _math.pi * i / n
        lat = center_lat + radius * _math.cos(angle)
        lon = center_lon + radius * _math.sin(angle)
        vertices.append((round(lat, 7), round(lon, 7)))
    vertices.append(vertices[0])  # close ring
    return PolygonInput(
        vertices=vertices,
        declared_area_ha=25.0,
        commodity="oil_palm",
        plot_id="POLY-COMPLEX-20",
    )


# ---------------------------------------------------------------------------
# Verification Request Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_verification_request(valid_coordinate_brazil, valid_polygon_small):
    """A sample verification request combining coordinate and polygon data."""
    return {
        "plot_id": "PLOT-BR-001",
        "coordinate": valid_coordinate_brazil,
        "polygon": valid_polygon_small,
        "declared_country": "BR",
        "commodity": "cocoa",
        "operator_id": "OP-TEST-001",
        "verification_level": "standard",
    }


@pytest.fixture
def sample_batch_request():
    """A batch request with 10 plots spanning multiple countries."""
    plots = []
    # 4 Brazil plots
    for i in range(4):
        plots.append({
            "plot_id": f"BATCH-BR-{i+1:03d}",
            "coordinate": CoordinateInput(
                lat=-3.12 + i * 0.01,
                lon=-60.02 + i * 0.005,
                declared_country="BR",
                commodity="cocoa",
                plot_id=f"BATCH-BR-{i+1:03d}",
            ),
            "polygon": PolygonInput(
                vertices=[
                    (-3.12 + i * 0.01, -60.02 + i * 0.005),
                    (-3.12 + i * 0.01, -60.018 + i * 0.005),
                    (-3.1215 + i * 0.01, -60.019 + i * 0.005),
                    (-3.12 + i * 0.01, -60.02 + i * 0.005),
                ],
                declared_area_ha=2.0,
                commodity="cocoa",
                plot_id=f"BATCH-BR-{i+1:03d}",
            ),
            "declared_country": "BR",
            "commodity": "cocoa",
        })
    # 3 Indonesia plots
    for i in range(3):
        plots.append({
            "plot_id": f"BATCH-ID-{i+1:03d}",
            "coordinate": CoordinateInput(
                lat=-2.57 + i * 0.01,
                lon=111.77 + i * 0.005,
                declared_country="ID",
                commodity="oil_palm",
                plot_id=f"BATCH-ID-{i+1:03d}",
            ),
            "polygon": PolygonInput(
                vertices=[
                    (-2.57 + i * 0.01, 111.77 + i * 0.005),
                    (-2.57 + i * 0.01, 111.772 + i * 0.005),
                    (-2.5715 + i * 0.01, 111.771 + i * 0.005),
                    (-2.57 + i * 0.01, 111.77 + i * 0.005),
                ],
                declared_area_ha=5.0,
                commodity="oil_palm",
                plot_id=f"BATCH-ID-{i+1:03d}",
            ),
            "declared_country": "ID",
            "commodity": "oil_palm",
        })
    # 3 Ghana plots
    for i in range(3):
        plots.append({
            "plot_id": f"BATCH-GH-{i+1:03d}",
            "coordinate": CoordinateInput(
                lat=6.12 + i * 0.01,
                lon=-1.62 + i * 0.005,
                declared_country="GH",
                commodity="cocoa",
                plot_id=f"BATCH-GH-{i+1:03d}",
            ),
            "polygon": PolygonInput(
                vertices=[
                    (6.12 + i * 0.01, -1.62 + i * 0.005),
                    (6.12 + i * 0.01, -1.618 + i * 0.005),
                    (6.1215 + i * 0.01, -1.619 + i * 0.005),
                    (6.12 + i * 0.01, -1.62 + i * 0.005),
                ],
                declared_area_ha=1.5,
                commodity="cocoa",
                plot_id=f"BATCH-GH-{i+1:03d}",
            ),
            "declared_country": "GH",
            "commodity": "cocoa",
        })
    return plots


# ---------------------------------------------------------------------------
# Engine Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def coordinate_validator(mock_config):
    """Create a CoordinateValidator instance for testing."""
    return CoordinateValidator(config=mock_config)


@pytest.fixture
def polygon_verifier(mock_config):
    """Create a PolygonTopologyVerifier instance for testing."""
    return PolygonTopologyVerifier(config=mock_config)


@pytest.fixture
def protected_checker(mock_config):
    """Create a ProtectedAreaChecker instance for testing."""
    return ProtectedAreaChecker(config=mock_config)


@pytest.fixture
def deforestation_verifier(mock_config):
    """Create a DeforestationCutoffVerifier instance for testing."""
    return DeforestationCutoffVerifier(config=mock_config)


@pytest.fixture
def accuracy_scorer(mock_config):
    """Create an AccuracyScoringEngine instance for testing."""
    return AccuracyScoringEngine(config=mock_config)


@pytest.fixture
def temporal_analyzer(mock_config):
    """Create a TemporalConsistencyAnalyzer instance for testing."""
    return TemporalConsistencyAnalyzer(config=mock_config)


@pytest.fixture
def batch_pipeline(mock_config):
    """Create a BatchVerificationPipeline instance for testing."""
    return BatchVerificationPipeline(config=mock_config)


@pytest.fixture
def article9_reporter(mock_config):
    """Create an Article9ComplianceReporter instance for testing."""
    return Article9ComplianceReporter(config=mock_config)


# ---------------------------------------------------------------------------
# Protected Area Mock Data
# ---------------------------------------------------------------------------


MOCK_PROTECTED_AREAS = [
    {
        "name": "Amazonia National Park",
        "country": "BR",
        "iucn_category": "II",
        "protection_level": "strict",
        "bounds": {
            "min_lat": -4.0,
            "max_lat": -2.0,
            "min_lon": -58.0,
            "max_lon": -56.0,
        },
    },
    {
        "name": "Tanjung Puting National Park",
        "country": "ID",
        "iucn_category": "II",
        "protection_level": "strict",
        "bounds": {
            "min_lat": -3.5,
            "max_lat": -2.5,
            "min_lon": 111.5,
            "max_lon": 112.5,
        },
    },
    {
        "name": "Kakum National Park",
        "country": "GH",
        "iucn_category": "II",
        "protection_level": "strict",
        "bounds": {
            "min_lat": 5.3,
            "max_lat": 5.5,
            "min_lon": -1.5,
            "max_lon": -1.3,
        },
    },
]


# ---------------------------------------------------------------------------
# Satellite / Deforestation Mock Data
# ---------------------------------------------------------------------------


MOCK_DEFORESTATION_ALERTS = {
    "post_cutoff": {
        "plot_id": "PLOT-DEFOREST-001",
        "alerts": [
            {
                "date": "2021-03-15",
                "source": "GFW",
                "confidence": 0.92,
                "area_ha": 1.5,
            },
            {
                "date": "2021-07-20",
                "source": "PRODES",
                "confidence": 0.88,
                "area_ha": 2.0,
            },
        ],
    },
    "pre_cutoff_only": {
        "plot_id": "PLOT-PRECUT-001",
        "alerts": [
            {
                "date": "2019-06-10",
                "source": "GFW",
                "confidence": 0.95,
                "area_ha": 3.0,
            },
        ],
    },
    "no_alerts": {
        "plot_id": "PLOT-CLEAN-001",
        "alerts": [],
    },
}


# ---------------------------------------------------------------------------
# Shared constants for test assertions
# ---------------------------------------------------------------------------


EUDR_DEFORESTATION_CUTOFF = "2020-12-31"

ALL_QUALITY_TIERS = ["gold", "silver", "bronze", "fail"]

ALL_ISSUE_SEVERITIES = ["critical", "high", "medium", "low", "info"]

ALL_CHANGE_TYPES = ["expansion", "contraction", "shift", "reshape", "stable"]

# EUDR Article 9 commodities
EUDR_COMMODITIES = ["cattle", "cocoa", "coffee", "oil_palm", "rubber", "soya", "wood"]

# Countries with high deforestation risk for EUDR
HIGH_RISK_COUNTRIES = ["BR", "ID", "CO", "MY", "PY", "CM", "CI", "NG"]

# Score thresholds for quality tier classification
GOLD_THRESHOLD = 85
SILVER_THRESHOLD = 70
BRONZE_THRESHOLD = 50

# SHA-256 hash length
SHA256_HEX_LENGTH = 64
