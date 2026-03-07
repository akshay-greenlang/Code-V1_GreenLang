# -*- coding: utf-8 -*-
"""
Shared pytest fixtures for AGENT-EUDR-003 Satellite Monitoring test suite.

Provides reusable fixtures for polygon coordinates, spectral band data,
satellite scene metadata, baseline snapshots, engine instances, and
shared test data used across all test modules.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-003 Satellite Monitoring Agent (GL-EUDR-SAT-003)
"""

import hashlib
import json
import math
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from greenlang.agents.eudr.satellite_monitoring.config import (
    SatelliteMonitoringConfig,
    get_config,
    set_config,
    reset_config,
)
from greenlang.agents.eudr.satellite_monitoring.imagery_acquisition import (
    ImageryAcquisitionEngine,
    SceneMetadata,
    DataQualityAssessment,
    SENTINEL2_BAND_SPECS,
    LANDSAT_BAND_SPECS,
    TILE_GRID,
)
from greenlang.agents.eudr.satellite_monitoring.reference_data.forest_thresholds import (
    BIOME_NDVI_THRESHOLDS,
    BIOME_EVI_THRESHOLDS,
    COMMODITY_BIOME_MAP,
    CLASSIFICATION_LEVELS,
    classify_ndvi,
    classify_evi,
    get_forest_threshold,
    get_biome_for_commodity,
    get_all_biomes,
    is_forest_cover,
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
# Provenance hash helper
# ---------------------------------------------------------------------------


def compute_test_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash for test assertions."""
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Configuration Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_config():
    """Create a SatelliteMonitoringConfig with test defaults."""
    return SatelliteMonitoringConfig(
        database_url="postgresql://localhost:5432/greenlang_test",
        redis_url="redis://localhost:6379/1",
        log_level="DEBUG",
        sentinel2_client_id="test-s2-client",
        sentinel2_client_secret="test-s2-secret",
        landsat_api_key="test-landsat-key",
        gfw_api_key="test-gfw-key",
        cutoff_date="2020-12-31",
        baseline_window_days=90,
        cloud_cover_max=20.0,
        cloud_cover_absolute_max=50.0,
        ndvi_deforestation_threshold=-0.15,
        ndvi_degradation_threshold=-0.05,
        regrowth_threshold=0.10,
        min_change_area_ha=0.1,
        sentinel2_weight=0.50,
        landsat_weight=0.30,
        gfw_weight=0.20,
        monitoring_max_concurrency=10,
        cache_ttl_seconds=300,
        baseline_cache_ttl_seconds=86400,
        quick_timeout_seconds=5.0,
        standard_timeout_seconds=30.0,
        deep_timeout_seconds=120.0,
        max_batch_size=100,
        alert_confidence_threshold=0.7,
        seasonal_adjustment_enabled=True,
        sar_enabled=True,
        enable_provenance=True,
        genesis_hash="GL-EUDR-SAT-003-TEST-GENESIS",
        enable_metrics=False,
        pool_size=5,
        rate_limit=500,
    )


@pytest.fixture
def strict_config():
    """Create a SatelliteMonitoringConfig with strict/sensitive thresholds."""
    return SatelliteMonitoringConfig(
        database_url="postgresql://localhost:5432/greenlang_test",
        redis_url="redis://localhost:6379/1",
        log_level="DEBUG",
        cutoff_date="2020-12-31",
        baseline_window_days=30,
        cloud_cover_max=10.0,
        cloud_cover_absolute_max=25.0,
        ndvi_deforestation_threshold=-0.08,
        ndvi_degradation_threshold=-0.03,
        regrowth_threshold=0.05,
        min_change_area_ha=0.01,
        sentinel2_weight=0.50,
        landsat_weight=0.30,
        gfw_weight=0.20,
        monitoring_max_concurrency=5,
        cache_ttl_seconds=60,
        baseline_cache_ttl_seconds=3600,
        quick_timeout_seconds=2.0,
        standard_timeout_seconds=10.0,
        deep_timeout_seconds=60.0,
        max_batch_size=50,
        alert_confidence_threshold=0.5,
        seasonal_adjustment_enabled=False,
        sar_enabled=False,
        enable_provenance=True,
        genesis_hash="GL-EUDR-SAT-003-TEST-STRICT-GENESIS",
        enable_metrics=False,
        pool_size=2,
        rate_limit=100,
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
# Polygon / Coordinate Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def amazon_polygon():
    """Polygon vertices for a plot in the Brazilian Amazon.

    Triangle near -3.12, -60.02 in Para state.
    Approximately 150m x 300m ~ 4.5 ha at this latitude.
    Ring is closed (first == last).
    """
    return [
        (-3.1200, -60.0200),
        (-3.1200, -60.0170),
        (-3.1225, -60.0185),
        (-3.1200, -60.0200),  # closure
    ]


@pytest.fixture
def borneo_polygon():
    """Polygon vertices for a palm oil plot in Indonesian Borneo (Kalimantan).

    Quadrilateral near -1.50, 110.40.
    """
    return [
        (-1.5000, 110.4000),
        (-1.5000, 110.4040),
        (-1.5030, 110.4040),
        (-1.5030, 110.4000),
        (-1.5000, 110.4000),  # closure
    ]


@pytest.fixture
def ghana_polygon():
    """Polygon vertices for a cocoa plot in Ashanti Region, Ghana.

    Triangle near 6.50, -1.60.
    """
    return [
        (6.5000, -1.6000),
        (6.5000, -1.5970),
        (6.5020, -1.5985),
        (6.5000, -1.6000),  # closure
    ]


@pytest.fixture
def congo_polygon():
    """Polygon vertices for a wood plot in the DRC (Congo Basin).

    Quadrilateral near 0.50, 20.50.
    """
    return [
        (0.5000, 20.5000),
        (0.5000, 20.5050),
        (0.5040, 20.5050),
        (0.5040, 20.5000),
        (0.5000, 20.5000),  # closure
    ]


@pytest.fixture
def small_polygon():
    """Very small polygon (~0.5 ha) for minimum-area tests."""
    return [
        (-3.1200, -60.0200),
        (-3.1200, -60.0195),
        (-3.1205, -60.01975),
        (-3.1200, -60.0200),
    ]


@pytest.fixture
def large_polygon():
    """Large polygon (~100 ha) for maximum-area tests."""
    return [
        (-3.0000, -60.0000),
        (-3.0000, -59.9800),
        (-3.0100, -59.9800),
        (-3.0100, -60.0000),
        (-3.0000, -60.0000),
    ]


@pytest.fixture
def invalid_polygon():
    """Empty polygon (no vertices) for error tests."""
    return []


# ---------------------------------------------------------------------------
# Band Data Fixtures (surface reflectance values in DN)
# ---------------------------------------------------------------------------


@pytest.fixture
def healthy_forest_bands():
    """Spectral band data for a healthy tropical forest pixel.

    Red (B04) = 500 DN -> low reflectance (chlorophyll absorption)
    NIR (B08) = 3500 DN -> high reflectance (leaf structure)
    NDVI = (3500-500)/(3500+500) = 3000/4000 = 0.75
    """
    return {
        "B02": 800,     # Blue
        "B03": 1200,    # Green
        "B04": 500,     # Red
        "B05": 1800,    # Red Edge 1
        "B06": 2800,    # Red Edge 2
        "B07": 3200,    # Red Edge 3
        "B08": 3500,    # NIR
        "B8A": 3400,    # Narrow NIR
        "B11": 1500,    # SWIR 1
        "B12": 800,     # SWIR 2
    }


@pytest.fixture
def degraded_forest_bands():
    """Spectral band data for a degraded forest pixel.

    Red (B04) = 1500 DN
    NIR (B08) = 2500 DN
    NDVI = (2500-1500)/(2500+1500) = 1000/4000 = 0.25
    """
    return {
        "B02": 1200,
        "B03": 1500,
        "B04": 1500,
        "B05": 1800,
        "B06": 2200,
        "B07": 2400,
        "B08": 2500,
        "B8A": 2400,
        "B11": 1800,
        "B12": 1200,
    }


@pytest.fixture
def deforested_bands():
    """Spectral band data for a deforested (bare soil) pixel.

    Red (B04) = 2500 DN
    NIR (B08) = 1500 DN
    NDVI = (1500-2500)/(1500+2500) = -1000/4000 = -0.25
    """
    return {
        "B02": 1800,
        "B03": 2200,
        "B04": 2500,
        "B05": 2000,
        "B06": 1800,
        "B07": 1600,
        "B08": 1500,
        "B8A": 1400,
        "B11": 2500,
        "B12": 2000,
    }


@pytest.fixture
def cloud_contaminated_bands():
    """Spectral band data for a cloud-contaminated pixel.

    All bands have very high reflectance typical of thick clouds.
    """
    return {
        "B02": 8000,
        "B03": 8200,
        "B04": 8500,
        "B05": 8000,
        "B06": 7800,
        "B07": 7500,
        "B08": 7000,
        "B8A": 6800,
        "B11": 4000,
        "B12": 2500,
    }


@pytest.fixture
def water_bands():
    """Spectral band data for a water body pixel.

    Very low reflectance across all bands, NIR < Red.
    NDVI is typically negative for water.
    """
    return {
        "B02": 500,
        "B03": 400,
        "B04": 300,
        "B05": 200,
        "B06": 150,
        "B07": 120,
        "B08": 100,
        "B8A": 90,
        "B11": 50,
        "B12": 30,
    }


# ---------------------------------------------------------------------------
# Scene Metadata Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sentinel2_scene():
    """SceneMetadata for a Sentinel-2 scene over the Amazon."""
    return SceneMetadata(
        scene_id="S2A_20201231_T20MQS",
        source="sentinel2",
        acquisition_date=date(2020, 12, 31),
        cloud_cover_pct=8.5,
        spatial_coverage_pct=98.0,
        tile_id="T20MQS",
        resolution_m=10,
        sun_elevation_deg=65.0,
        sun_azimuth_deg=140.0,
        processing_level="L2A",
        bands_available=list(SENTINEL2_BAND_SPECS.keys()),
        file_size_mb=750.0,
        quality_score=0.0,
        provenance_hash="",
    )


@pytest.fixture
def landsat_scene():
    """SceneMetadata for a Landsat-8 scene over the Amazon."""
    return SceneMetadata(
        scene_id="LC08_20201215_P001R060",
        source="landsat8",
        acquisition_date=date(2020, 12, 15),
        cloud_cover_pct=12.0,
        spatial_coverage_pct=95.0,
        tile_id="P001R060",
        resolution_m=30,
        sun_elevation_deg=58.0,
        sun_azimuth_deg=135.0,
        processing_level="L1TP",
        bands_available=list(LANDSAT_BAND_SPECS.keys()),
        file_size_mb=1050.0,
        quality_score=0.0,
        provenance_hash="",
    )


@pytest.fixture
def cloudy_scene():
    """SceneMetadata for a heavily cloudy scene (45% cloud cover)."""
    return SceneMetadata(
        scene_id="S2A_20210315_T20MQS_CLOUDY",
        source="sentinel2",
        acquisition_date=date(2021, 3, 15),
        cloud_cover_pct=45.0,
        spatial_coverage_pct=85.0,
        tile_id="T20MQS",
        resolution_m=10,
        sun_elevation_deg=55.0,
        sun_azimuth_deg=145.0,
        processing_level="L2A",
        bands_available=list(SENTINEL2_BAND_SPECS.keys()),
        file_size_mb=720.0,
        quality_score=0.0,
        provenance_hash="",
    )


@pytest.fixture
def scene_list(sentinel2_scene, landsat_scene, cloudy_scene):
    """List of three scenes for best-scene selection tests."""
    return [sentinel2_scene, landsat_scene, cloudy_scene]


# ---------------------------------------------------------------------------
# Baseline Snapshot Fixtures
# ---------------------------------------------------------------------------


@dataclass
class BaselineSnapshot:
    """Test-only baseline data class for fixture use."""

    plot_id: str = ""
    biome: str = "tropical_rainforest"
    cutoff_date: str = "2020-12-31"
    ndvi_mean: float = 0.0
    ndvi_std: float = 0.0
    evi_mean: float = 0.0
    evi_std: float = 0.0
    forest_percentage: float = 0.0
    total_area_ha: float = 0.0
    cloud_free_percentage: float = 0.0
    scenes_used: int = 0
    composite_method: str = "median"
    quality_score: float = 0.0
    provenance_hash: str = ""
    established_at: Optional[str] = None


@pytest.fixture
def amazon_baseline():
    """BaselineSnapshot for an Amazon plot at the EUDR cutoff date."""
    return BaselineSnapshot(
        plot_id="PLOT-BR-001",
        biome="tropical_rainforest",
        cutoff_date="2020-12-31",
        ndvi_mean=0.72,
        ndvi_std=0.05,
        evi_mean=0.48,
        evi_std=0.04,
        forest_percentage=95.0,
        total_area_ha=4.5,
        cloud_free_percentage=92.0,
        scenes_used=6,
        composite_method="median",
        quality_score=88.5,
        provenance_hash=compute_test_hash({
            "plot_id": "PLOT-BR-001",
            "cutoff_date": "2020-12-31",
            "ndvi_mean": 0.72,
        }),
        established_at="2026-03-01T00:00:00+00:00",
    )


@pytest.fixture
def borneo_baseline():
    """BaselineSnapshot for a Borneo palm oil plot."""
    return BaselineSnapshot(
        plot_id="PLOT-ID-001",
        biome="tropical_rainforest",
        cutoff_date="2020-12-31",
        ndvi_mean=0.68,
        ndvi_std=0.06,
        evi_mean=0.45,
        evi_std=0.05,
        forest_percentage=88.0,
        total_area_ha=8.0,
        cloud_free_percentage=78.0,
        scenes_used=4,
        composite_method="median",
        quality_score=75.0,
        provenance_hash=compute_test_hash({
            "plot_id": "PLOT-ID-001",
            "cutoff_date": "2020-12-31",
            "ndvi_mean": 0.68,
        }),
        established_at="2026-03-01T00:00:00+00:00",
    )


# ---------------------------------------------------------------------------
# Change Detection Result Fixture Helpers
# ---------------------------------------------------------------------------


@dataclass
class ChangeDetectionResult:
    """Test-only change detection result for fixture use."""

    plot_id: str = ""
    baseline_ndvi: float = 0.0
    current_ndvi: float = 0.0
    ndvi_delta: float = 0.0
    classification: str = "no_change"
    confidence: float = 0.0
    change_area_ha: float = 0.0
    deforestation_detected: bool = False
    detection_method: str = "ndvi_differencing"
    provenance_hash: str = ""


@dataclass
class SatelliteAlert:
    """Test-only alert model for fixture use."""

    alert_id: str = ""
    plot_id: str = ""
    severity: str = "info"
    change_type: str = "no_change"
    ndvi_drop: float = 0.0
    confidence: float = 0.0
    area_affected_ha: float = 0.0
    detected_at: Optional[str] = None
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[str] = None
    provenance_hash: str = ""


@dataclass
class EvidencePackage:
    """Test-only evidence package model for fixture use."""

    evidence_id: str = ""
    plot_id: str = ""
    compliance_status: str = "unknown"
    format: str = "json"
    baseline_snapshot: Optional[Dict[str, Any]] = None
    change_results: Optional[List[Dict[str, Any]]] = None
    alert_history: Optional[List[Dict[str, Any]]] = None
    generated_at: Optional[str] = None
    provenance_hash: str = ""


@dataclass
class FusionResult:
    """Test-only multi-source fusion result for fixture use."""

    plot_id: str = ""
    sentinel2_result: Optional[str] = None
    landsat_result: Optional[str] = None
    gfw_result: Optional[str] = None
    fused_classification: str = "no_change"
    agreement_score: float = 0.0
    compliance_status: str = "unknown"
    confidence: float = 0.0
    quality_score: float = 0.0
    provenance_hash: str = ""


@dataclass
class CloudGapFillResult:
    """Test-only cloud gap fill result for fixture use."""

    scene_id: str = ""
    original_cloud_pct: float = 0.0
    filled_cloud_pct: float = 0.0
    fill_method: str = "temporal_composite"
    pixels_filled: int = 0
    total_pixels: int = 0
    quality_score: float = 0.0
    provenance_hash: str = ""


# ---------------------------------------------------------------------------
# Engine Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def imagery_engine(default_config):
    """Create an ImageryAcquisitionEngine instance for testing."""
    return ImageryAcquisitionEngine(config=default_config)


@pytest.fixture
def spectral_calculator(default_config):
    """Create a mock SpectralIndexCalculator for testing.

    Returns a MagicMock since SpectralIndexCalculator may have
    additional constructor dependencies beyond config.
    """
    calc = MagicMock()
    calc.config = default_config
    calc.compute_ndvi = MagicMock(side_effect=lambda red, nir: (
        (nir - red) / (nir + red) if (nir + red) != 0 else 0.0
    ))
    calc.compute_evi = MagicMock(side_effect=lambda blue, red, nir: (
        2.5 * (nir - red) / (nir + 6.0 * red - 7.5 * blue + 1.0)
        if (nir + 6.0 * red - 7.5 * blue + 1.0) != 0 else 0.0
    ))
    calc.compute_nbr = MagicMock(side_effect=lambda nir, swir2: (
        (nir - swir2) / (nir + swir2) if (nir + swir2) != 0 else 0.0
    ))
    calc.compute_ndmi = MagicMock(side_effect=lambda nir, swir1: (
        (nir - swir1) / (nir + swir1) if (nir + swir1) != 0 else 0.0
    ))
    calc.compute_savi = MagicMock(side_effect=lambda red, nir, L=0.5: (
        ((nir - red) / (nir + red + L)) * (1.0 + L)
        if (nir + red + L) != 0 else 0.0
    ))
    return calc


@pytest.fixture
def baseline_manager(default_config):
    """Create a mock BaselineManager for testing."""
    mgr = MagicMock()
    mgr.config = default_config
    return mgr


@pytest.fixture
def change_detector(default_config):
    """Create a mock ForestChangeDetector for testing."""
    detector = MagicMock()
    detector.config = default_config
    return detector


@pytest.fixture
def fusion_engine(default_config):
    """Create a mock DataFusionEngine for testing."""
    engine = MagicMock()
    engine.config = default_config
    return engine


@pytest.fixture
def cloud_filler(default_config):
    """Create a mock CloudGapFiller for testing."""
    filler = MagicMock()
    filler.config = default_config
    return filler


@pytest.fixture
def continuous_monitor(default_config):
    """Create a mock ContinuousMonitor for testing."""
    monitor = MagicMock()
    monitor.config = default_config
    return monitor


@pytest.fixture
def alert_generator(default_config):
    """Create a mock AlertGenerator for testing."""
    gen = MagicMock()
    gen.config = default_config
    return gen


# ---------------------------------------------------------------------------
# Shared Constants for Test Assertions
# ---------------------------------------------------------------------------


EUDR_DEFORESTATION_CUTOFF = "2020-12-31"

SHA256_HEX_LENGTH = 64

# EUDR Article 9 commodities
EUDR_COMMODITIES = [
    "cattle", "cocoa", "coffee", "oil_palm", "rubber", "soya", "wood",
]

# Countries with high deforestation risk for EUDR monitoring
HIGH_RISK_COUNTRIES = [
    "BR", "ID", "CO", "MY", "PY", "CM", "CI", "GH", "NG", "CD",
    "CG", "PE", "BO", "VN", "TH", "MM",
]

# Satellite sources supported
SATELLITE_SOURCES = ["sentinel2", "landsat8", "landsat9"]

# Spectral index names
SPECTRAL_INDICES = ["ndvi", "evi", "nbr", "ndmi", "savi"]

# Forest classification levels
FOREST_CLASSIFICATIONS = [
    "dense_forest", "forest", "shrubland", "sparse_vegetation", "non_vegetated",
]

# Change detection classifications
CHANGE_CLASSIFICATIONS = [
    "deforestation", "degradation", "regrowth", "no_change",
]

# Alert severity levels
ALERT_SEVERITIES = ["critical", "warning", "info"]

# Monitoring intervals
MONITORING_INTERVALS = [
    "daily", "weekly", "biweekly", "monthly", "quarterly",
]

# Evidence formats
EVIDENCE_FORMATS = ["json", "csv", "pdf"]

# Cloud fill methods
CLOUD_FILL_METHODS = [
    "temporal_composite", "sar_fusion", "interpolation", "nearest_clear",
]

# Detection methods
DETECTION_METHODS = [
    "ndvi_differencing", "spectral_angle", "time_series_break",
]

# Biomes list
ALL_BIOMES = sorted(BIOME_NDVI_THRESHOLDS.keys())

# EUDR-relevant countries with known MGRS tile coverage
EUDR_COUNTRIES_WITH_TILES = {
    "BR": [(-3.0, -62.0), (-8.0, -63.0), (-3.0, -57.0)],
    "ID": [(-0.5, 110.0), (-1.0, 113.0), (-4.0, 106.0)],
    "MY": [(4.0, 110.0)],
    "GH": [(6.0, -2.0), (10.0, -2.0)],
    "CI": [(6.0, -6.0)],
    "CO": [(2.0, -76.0)],
    "CD": [(0.0, 20.0), (4.0, 20.0)],
    "CG": [(0.0, 20.0)],
    "CM": [(4.0, 10.0)],
    "PY": [(-22.0, -60.0)],
    "PE": [(-5.0, -76.0)],
    "ET": [(0.0, 38.0)],
    "KE": [(0.0, 38.0)],
    "NG": [(8.0, 4.0)],
    "GA": [(0.0, 10.0)],
}
