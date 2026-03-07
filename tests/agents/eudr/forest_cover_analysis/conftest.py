# -*- coding: utf-8 -*-
"""
Shared pytest fixtures for AGENT-EUDR-004 Forest Cover Analysis test suite.

Provides reusable fixtures for polygons (WKT and coordinate lists),
spectral band data, NDVI/canopy density values, classification results,
historical records, height estimates, fragmentation metrics, biomass
estimates, deforestation-free verdicts, engine mocks, and configuration
objects used across all test modules.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-004 Forest Cover Analysis Agent (GL-EUDR-FCA-004)
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

from greenlang.agents.eudr.forest_cover_analysis.config import (
    ForestCoverConfig,
    get_config,
    set_config,
    reset_config,
)
from greenlang.agents.eudr.forest_cover_analysis.provenance import (
    ProvenanceTracker,
    ProvenanceEntry,
    VALID_ENTITY_TYPES,
    VALID_ACTIONS,
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
# Shared Constants
# ---------------------------------------------------------------------------

SHA256_HEX_LENGTH = 64

EUDR_DEFORESTATION_CUTOFF = "2020-12-31"

EUDR_COMMODITIES = [
    "cattle", "cocoa", "coffee", "oil_palm", "rubber", "soya", "wood",
]

# FAO forest definition constants
FAO_CANOPY_COVER_PCT = 10.0  # percent
FAO_TREE_HEIGHT_M = 5.0  # metres
FAO_MIN_AREA_HA = 0.5  # hectares

# All 16 biome names recognised by the reference data
ALL_BIOMES = [
    "boreal_forest",
    "cerrado_savanna",
    "dry_woodland",
    "mangrove",
    "montane_cloud_forest",
    "montane_dry_forest",
    "peat_swamp_forest",
    "temperate_deciduous",
    "temperate_forest",
    "temperate_rainforest",
    "thorn_forest",
    "tropical_dry_forest",
    "tropical_moist_forest",
    "tropical_rainforest",
    "tropical_savanna",
    "woodland_savanna",
]

# 10 forest types for classification
FOREST_TYPES = [
    "PRIMARY_TROPICAL",
    "SECONDARY_TROPICAL",
    "MANGROVE",
    "PEAT_SWAMP",
    "TEMPERATE_BROADLEAF",
    "TEMPERATE_CONIFEROUS",
    "BOREAL_CONIFEROUS",
    "MONTANE_CLOUD",
    "PLANTATION",
    "AGROFORESTRY",
]

# 6 canopy density classes
CANOPY_DENSITY_CLASSES = [
    "VERY_HIGH",   # > 80%
    "HIGH",        # 60-80%
    "MODERATE",    # 40-60%
    "LOW",         # 20-40%
    "VERY_LOW",    # 10-20%
    "SPARSE",      # < 10%
]

# Canopy density class boundary values (lower bound inclusive)
DENSITY_CLASS_BOUNDARIES = {
    "VERY_HIGH": 80.0,
    "HIGH": 60.0,
    "MODERATE": 40.0,
    "LOW": 20.0,
    "VERY_LOW": 10.0,
    "SPARSE": 0.0,
}

# 4 deforestation-free verdicts
VERDICTS = [
    "DEFORESTATION_FREE",
    "DEFORESTED",
    "DEGRADED",
    "INCONCLUSIVE",
]

# 4 density estimation methods
DENSITY_METHODS = [
    "spectral_unmixing",
    "ndvi_regression",
    "dimidiation",
    "sub_pixel",
]

# Data quality tiers
DATA_QUALITY_TIERS = {
    "GOLD": 90.0,
    "SILVER": 70.0,
    "BRONZE": 50.0,
    "INSUFFICIENT": 0.0,
}

# Multi-source fusion weights for historical reconstruction
RECONSTRUCTION_SOURCE_WEIGHTS = {
    "landsat": 0.30,
    "sentinel2": 0.30,
    "hansen": 0.25,
    "jaxa": 0.15,
}

# Canopy height source weights for fusion
HEIGHT_SOURCE_WEIGHTS = {
    "gedi": 0.35,
    "icesat2": 0.30,
    "eth_global": 0.20,
    "meta_global": 0.10,
    "texture_proxy": 0.05,
}

# EUDR regulatory articles referenced by verdict
VERDICT_REGULATORY_REFS = {
    "DEFORESTATION_FREE": ["EUDR Art. 3(a)", "EUDR Art. 10(1)"],
    "DEFORESTED": ["EUDR Art. 3(b)", "EUDR Art. 10(2)"],
    "DEGRADED": ["EUDR Art. 2(6)", "EUDR Art. 10(2)"],
    "INCONCLUSIVE": ["EUDR Art. 10(3)", "EUDR Art. 11(1)"],
}


# ---------------------------------------------------------------------------
# Test-only Dataclass Models (mirrors of production models)
# ---------------------------------------------------------------------------


@dataclass
class CanopyDensityResult:
    """Test-only result of canopy density mapping for a single plot."""

    plot_id: str = ""
    biome: str = "tropical_rainforest"
    method: str = "ndvi_regression"
    canopy_density_pct: float = 0.0
    density_class: str = "SPARSE"
    forest_fractions: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    cloud_cover_pct: float = 0.0
    meets_fao_threshold: bool = False
    area_ha: float = 0.0
    provenance_hash: str = ""


@dataclass
class ForestClassificationResult:
    """Test-only result of forest type classification for a single plot."""

    plot_id: str = ""
    forest_type: str = "PRIMARY_TROPICAL"
    spectral_class: str = ""
    phenological_class: str = ""
    structural_class: str = ""
    ensemble_class: str = ""
    is_forest_per_eudr: bool = True
    commodity_exclusion_applied: bool = False
    confidence: float = 0.0
    inter_method_agreement: float = 0.0
    provenance_hash: str = ""


@dataclass
class HistoricalCoverRecord:
    """Test-only record of reconstructed historical forest cover."""

    plot_id: str = ""
    cutoff_date: str = EUDR_DEFORESTATION_CUTOFF
    was_forest: bool = True
    canopy_density_pct: float = 0.0
    ndvi_mean: float = 0.0
    hansen_tree_cover_pct: float = 0.0
    cross_validation_score: float = 0.0
    sources_used: List[str] = field(default_factory=list)
    source_weights: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    provenance_hash: str = ""


@dataclass
class DeforestationFreeResult:
    """Test-only result of deforestation-free verification."""

    plot_id: str = ""
    verdict: str = "INCONCLUSIVE"
    cutoff_was_forest: bool = False
    current_is_forest: bool = False
    canopy_change_pct: float = 0.0
    degradation_threshold_pct: float = 30.0
    commodity: str = ""
    commodity_exclusion_applied: bool = False
    confidence: float = 0.0
    confidence_min: float = 0.6
    evidence_package: Dict[str, Any] = field(default_factory=dict)
    regulatory_references: List[str] = field(default_factory=list)
    provenance_hash: str = ""


@dataclass
class CanopyHeightEstimate:
    """Test-only result of canopy height estimation."""

    plot_id: str = ""
    height_m: float = 0.0
    gedi_height_m: Optional[float] = None
    icesat2_height_m: Optional[float] = None
    texture_height_m: Optional[float] = None
    eth_height_m: Optional[float] = None
    meta_height_m: Optional[float] = None
    fused_height_m: float = 0.0
    uncertainty_m: float = 0.0
    meets_fao_threshold: bool = False
    sources_available: int = 0
    provenance_hash: str = ""


@dataclass
class FragmentationMetrics:
    """Test-only result of landscape fragmentation analysis."""

    plot_id: str = ""
    num_patches: int = 0
    mean_patch_area_ha: float = 0.0
    edge_density_m_per_ha: float = 0.0
    core_area_pct: float = 0.0
    nearest_neighbour_m: float = 0.0
    perimeter_area_ratio: float = 0.0
    effective_mesh_size_ha: float = 0.0
    fragmentation_class: str = "intact"
    risk_score: float = 0.0
    provenance_hash: str = ""


@dataclass
class BiomassEstimate:
    """Test-only result of above-ground biomass estimation."""

    plot_id: str = ""
    biome: str = "tropical_rainforest"
    agb_mg_per_ha: float = 0.0
    carbon_stock_mg_per_ha: float = 0.0
    esa_cci_agb: Optional[float] = None
    gedi_l4a_agb: Optional[float] = None
    sar_agb: Optional[float] = None
    sar_saturated: bool = False
    ndvi_allometric_agb: Optional[float] = None
    fused_agb: float = 0.0
    uncertainty_mg_per_ha: float = 0.0
    change_from_cutoff_pct: float = 0.0
    provenance_hash: str = ""


# ---------------------------------------------------------------------------
# Configuration Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_config():
    """Create a ForestCoverConfig with test defaults."""
    return ForestCoverConfig(
        database_url="postgresql://localhost:5432/greenlang_test",
        redis_url="redis://localhost:6379/1",
        log_level="DEBUG",
        canopy_cover_threshold=10.0,
        tree_height_threshold=5.0,
        min_forest_area_ha=0.5,
        degradation_threshold=30.0,
        cutoff_date="2020-12-31",
        baseline_window_years=3,
        gedi_api_key="test-gedi-key",
        esa_cci_api_key="test-esa-key",
        hansen_gfc_version="v1.11",
        max_batch_size=100,
        analysis_concurrency=4,
        cache_ttl_seconds=300,
        biomass_cache_ttl_seconds=3600,
        confidence_min=0.6,
        genesis_hash="GL-EUDR-FCA-004-TEST-GENESIS",
        enable_metrics=False,
    )


@pytest.fixture
def strict_config():
    """Create a ForestCoverConfig with strict/conservative thresholds."""
    return ForestCoverConfig(
        database_url="postgresql://localhost:5432/greenlang_test",
        redis_url="redis://localhost:6379/1",
        log_level="DEBUG",
        canopy_cover_threshold=15.0,
        tree_height_threshold=7.0,
        min_forest_area_ha=1.0,
        degradation_threshold=20.0,
        cutoff_date="2020-12-31",
        baseline_window_years=5,
        gedi_api_key="test-gedi-strict",
        esa_cci_api_key="test-esa-strict",
        hansen_gfc_version="v1.11",
        max_batch_size=50,
        analysis_concurrency=2,
        cache_ttl_seconds=60,
        biomass_cache_ttl_seconds=1800,
        confidence_min=0.8,
        genesis_hash="GL-EUDR-FCA-004-TEST-STRICT-GENESIS",
        enable_metrics=False,
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
def sample_polygon_wkt():
    """Valid WKT polygon for a plot in the Brazilian Amazon.

    Roughly 4.5 ha near -3.12, -60.02.
    """
    return (
        "POLYGON(("
        "-60.0200 -3.1200, "
        "-60.0170 -3.1200, "
        "-60.0170 -3.1230, "
        "-60.0200 -3.1230, "
        "-60.0200 -3.1200"
        "))"
    )


@pytest.fixture
def sample_polygon_coords():
    """Coordinate list for a plot in the Brazilian Amazon (closed ring)."""
    return [
        (-3.1200, -60.0200),
        (-3.1200, -60.0170),
        (-3.1230, -60.0170),
        (-3.1230, -60.0200),
        (-3.1200, -60.0200),
    ]


@pytest.fixture
def sample_polygon_small():
    """Small polygon below FAO 0.5 ha threshold (~0.3 ha)."""
    return [
        (-3.1200, -60.0200),
        (-3.1200, -60.0195),
        (-3.1204, -60.0195),
        (-3.1204, -60.0200),
        (-3.1200, -60.0200),
    ]


@pytest.fixture
def sample_polygon_large():
    """Large polygon (~120 ha) for stress / boundary tests."""
    return [
        (-3.0000, -60.0000),
        (-3.0000, -59.9800),
        (-3.0120, -59.9800),
        (-3.0120, -60.0000),
        (-3.0000, -60.0000),
    ]


@pytest.fixture
def borneo_polygon():
    """Palm oil plot in Indonesian Borneo (Kalimantan), ~8 ha."""
    return [
        (-1.5000, 110.4000),
        (-1.5000, 110.4040),
        (-1.5030, 110.4040),
        (-1.5030, 110.4000),
        (-1.5000, 110.4000),
    ]


@pytest.fixture
def ghana_polygon():
    """Cocoa plot in Ashanti Region, Ghana, ~3 ha."""
    return [
        (6.5000, -1.6000),
        (6.5000, -1.5970),
        (6.5020, -1.5985),
        (6.5000, -1.6000),
    ]


# ---------------------------------------------------------------------------
# NDVI / Spectral Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_ndvi_forest():
    """NDVI value typical of healthy tropical forest (0.7-0.9 range)."""
    return 0.78


@pytest.fixture
def sample_ndvi_non_forest():
    """NDVI value typical of bare soil / non-forest (0.1-0.3 range)."""
    return 0.18


@pytest.fixture
def sample_ndvi_degraded():
    """NDVI value typical of degraded forest (0.35-0.45)."""
    return 0.40


@pytest.fixture
def sample_spectral_bands():
    """6-band reflectance values for a healthy forest pixel.

    Bands: blue, green, red, nir, swir1, swir2.
    Red low + NIR high = high NDVI (forest).
    """
    return {
        "blue": 0.030,
        "green": 0.055,
        "red": 0.025,
        "nir": 0.350,
        "swir1": 0.120,
        "swir2": 0.060,
    }


@pytest.fixture
def sample_spectral_bands_non_forest():
    """6-band reflectance values for a non-forest (bare soil) pixel.

    Red high + NIR moderate/low = low NDVI.
    """
    return {
        "blue": 0.100,
        "green": 0.140,
        "red": 0.180,
        "nir": 0.200,
        "swir1": 0.250,
        "swir2": 0.200,
    }


@pytest.fixture
def sample_spectral_bands_cloud():
    """6-band reflectance for a cloud-contaminated pixel (all high)."""
    return {
        "blue": 0.550,
        "green": 0.560,
        "red": 0.570,
        "nir": 0.520,
        "swir1": 0.350,
        "swir2": 0.220,
    }


# ---------------------------------------------------------------------------
# Result Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_canopy_density_result():
    """CanopyDensityResult for a healthy tropical forest plot."""
    return CanopyDensityResult(
        plot_id="PLOT-BR-001",
        biome="tropical_rainforest",
        method="ndvi_regression",
        canopy_density_pct=72.5,
        density_class="HIGH",
        forest_fractions={"forest": 0.72, "soil": 0.15, "shadow": 0.13},
        confidence=0.88,
        cloud_cover_pct=5.0,
        meets_fao_threshold=True,
        area_ha=4.5,
        provenance_hash=compute_test_hash({
            "plot_id": "PLOT-BR-001",
            "canopy_density_pct": 72.5,
        }),
    )


@pytest.fixture
def sample_classification_result():
    """ForestClassificationResult for a primary tropical forest plot."""
    return ForestClassificationResult(
        plot_id="PLOT-BR-001",
        forest_type="PRIMARY_TROPICAL",
        spectral_class="PRIMARY_TROPICAL",
        phenological_class="PRIMARY_TROPICAL",
        structural_class="PRIMARY_TROPICAL",
        ensemble_class="PRIMARY_TROPICAL",
        is_forest_per_eudr=True,
        commodity_exclusion_applied=False,
        confidence=0.92,
        inter_method_agreement=0.95,
        provenance_hash=compute_test_hash({
            "plot_id": "PLOT-BR-001",
            "forest_type": "PRIMARY_TROPICAL",
        }),
    )


@pytest.fixture
def sample_historical_record():
    """HistoricalCoverRecord showing forest present at cutoff (was_forest=True)."""
    return HistoricalCoverRecord(
        plot_id="PLOT-BR-001",
        cutoff_date=EUDR_DEFORESTATION_CUTOFF,
        was_forest=True,
        canopy_density_pct=75.0,
        ndvi_mean=0.72,
        hansen_tree_cover_pct=80.0,
        cross_validation_score=0.90,
        sources_used=["landsat", "sentinel2", "hansen", "jaxa"],
        source_weights=RECONSTRUCTION_SOURCE_WEIGHTS.copy(),
        confidence=0.88,
        provenance_hash=compute_test_hash({
            "plot_id": "PLOT-BR-001",
            "was_forest": True,
            "cutoff_date": EUDR_DEFORESTATION_CUTOFF,
        }),
    )


@pytest.fixture
def sample_historical_record_no_forest():
    """HistoricalCoverRecord showing no forest at cutoff (was_forest=False)."""
    return HistoricalCoverRecord(
        plot_id="PLOT-BR-002",
        cutoff_date=EUDR_DEFORESTATION_CUTOFF,
        was_forest=False,
        canopy_density_pct=5.0,
        ndvi_mean=0.15,
        hansen_tree_cover_pct=3.0,
        cross_validation_score=0.92,
        sources_used=["landsat", "sentinel2", "hansen"],
        source_weights={"landsat": 0.35, "sentinel2": 0.35, "hansen": 0.30},
        confidence=0.90,
        provenance_hash=compute_test_hash({
            "plot_id": "PLOT-BR-002",
            "was_forest": False,
            "cutoff_date": EUDR_DEFORESTATION_CUTOFF,
        }),
    )


@pytest.fixture
def sample_height_estimate():
    """CanopyHeightEstimate with height=25m (above FAO 5m threshold)."""
    return CanopyHeightEstimate(
        plot_id="PLOT-BR-001",
        height_m=25.0,
        gedi_height_m=26.5,
        icesat2_height_m=24.0,
        texture_height_m=22.0,
        eth_height_m=25.5,
        meta_height_m=26.0,
        fused_height_m=25.0,
        uncertainty_m=2.5,
        meets_fao_threshold=True,
        sources_available=5,
        provenance_hash=compute_test_hash({
            "plot_id": "PLOT-BR-001",
            "height_m": 25.0,
        }),
    )


@pytest.fixture
def sample_fragmentation_metrics():
    """FragmentationMetrics for a moderately fragmented plot."""
    return FragmentationMetrics(
        plot_id="PLOT-BR-001",
        num_patches=3,
        mean_patch_area_ha=1.5,
        edge_density_m_per_ha=120.0,
        core_area_pct=65.0,
        nearest_neighbour_m=50.0,
        perimeter_area_ratio=0.08,
        effective_mesh_size_ha=2.0,
        fragmentation_class="moderate",
        risk_score=0.4,
        provenance_hash=compute_test_hash({
            "plot_id": "PLOT-BR-001",
            "num_patches": 3,
        }),
    )


@pytest.fixture
def sample_biomass_estimate():
    """BiomassEstimate with AGB=200 Mg/ha (typical tropical forest)."""
    return BiomassEstimate(
        plot_id="PLOT-BR-001",
        biome="tropical_rainforest",
        agb_mg_per_ha=200.0,
        carbon_stock_mg_per_ha=200.0 * 0.47,  # 94.0
        esa_cci_agb=195.0,
        gedi_l4a_agb=210.0,
        sar_agb=180.0,
        sar_saturated=False,
        ndvi_allometric_agb=190.0,
        fused_agb=200.0,
        uncertainty_mg_per_ha=25.0,
        change_from_cutoff_pct=-2.5,
        provenance_hash=compute_test_hash({
            "plot_id": "PLOT-BR-001",
            "agb_mg_per_ha": 200.0,
        }),
    )


@pytest.fixture
def sample_deforestation_free_result():
    """DeforestationFreeResult with verdict=DEFORESTATION_FREE."""
    return DeforestationFreeResult(
        plot_id="PLOT-BR-001",
        verdict="DEFORESTATION_FREE",
        cutoff_was_forest=True,
        current_is_forest=True,
        canopy_change_pct=-5.0,
        degradation_threshold_pct=30.0,
        commodity="soya",
        commodity_exclusion_applied=False,
        confidence=0.92,
        confidence_min=0.6,
        evidence_package={
            "before_ndvi": 0.75,
            "after_ndvi": 0.72,
            "spectral_comparison": "stable",
        },
        regulatory_references=VERDICT_REGULATORY_REFS["DEFORESTATION_FREE"],
        provenance_hash=compute_test_hash({
            "plot_id": "PLOT-BR-001",
            "verdict": "DEFORESTATION_FREE",
        }),
    )


# ---------------------------------------------------------------------------
# Reference Data Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def biome_list():
    """Return all 16 biome names."""
    return ALL_BIOMES[:]


@pytest.fixture
def commodity_list():
    """Return all 7 EUDR commodities."""
    return EUDR_COMMODITIES[:]


# ---------------------------------------------------------------------------
# Engine Mock Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_canopy_density_mapper(sample_config):
    """Mocked CanopyDensityMapper engine."""
    mapper = MagicMock()
    mapper.config = sample_config
    mapper.analyze_plot = MagicMock(return_value=CanopyDensityResult(
        plot_id="PLOT-MOCK",
        canopy_density_pct=70.0,
        density_class="HIGH",
        confidence=0.85,
        meets_fao_threshold=True,
        area_ha=4.0,
    ))
    return mapper


@pytest.fixture
def mock_forest_type_classifier(sample_config):
    """Mocked ForestTypeClassifier engine."""
    classifier = MagicMock()
    classifier.config = sample_config
    classifier.classify_plot = MagicMock(return_value=ForestClassificationResult(
        plot_id="PLOT-MOCK",
        forest_type="PRIMARY_TROPICAL",
        is_forest_per_eudr=True,
        confidence=0.90,
    ))
    return classifier


@pytest.fixture
def mock_historical_reconstructor(sample_config):
    """Mocked HistoricalReconstructor engine."""
    reconstructor = MagicMock()
    reconstructor.config = sample_config
    reconstructor.reconstruct_plot = MagicMock(
        return_value=HistoricalCoverRecord(
            plot_id="PLOT-MOCK",
            was_forest=True,
            canopy_density_pct=75.0,
            confidence=0.88,
        )
    )
    return reconstructor


@pytest.fixture
def mock_deforestation_free_verifier(sample_config):
    """Mocked DeforestationFreeVerifier engine."""
    verifier = MagicMock()
    verifier.config = sample_config
    verifier.verify_plot = MagicMock(return_value=DeforestationFreeResult(
        plot_id="PLOT-MOCK",
        verdict="DEFORESTATION_FREE",
        confidence=0.92,
    ))
    return verifier


@pytest.fixture
def mock_canopy_height_modeler(sample_config):
    """Mocked CanopyHeightModeler engine."""
    modeler = MagicMock()
    modeler.config = sample_config
    modeler.estimate_plot_height = MagicMock(
        return_value=CanopyHeightEstimate(
            plot_id="PLOT-MOCK",
            height_m=25.0,
            fused_height_m=25.0,
            meets_fao_threshold=True,
        )
    )
    return modeler


@pytest.fixture
def mock_fragmentation_analyzer(sample_config):
    """Mocked FragmentationAnalyzer engine."""
    analyzer = MagicMock()
    analyzer.config = sample_config
    analyzer.analyze_plot = MagicMock(return_value=FragmentationMetrics(
        plot_id="PLOT-MOCK",
        num_patches=2,
        fragmentation_class="moderate",
    ))
    return analyzer


@pytest.fixture
def mock_biomass_estimator(sample_config):
    """Mocked BiomassEstimator engine."""
    estimator = MagicMock()
    estimator.config = sample_config
    estimator.estimate_plot_biomass = MagicMock(
        return_value=BiomassEstimate(
            plot_id="PLOT-MOCK",
            agb_mg_per_ha=200.0,
            carbon_stock_mg_per_ha=94.0,
        )
    )
    return estimator


@pytest.fixture
def mock_engines(
    mock_canopy_density_mapper,
    mock_forest_type_classifier,
    mock_historical_reconstructor,
    mock_deforestation_free_verifier,
    mock_canopy_height_modeler,
    mock_fragmentation_analyzer,
    mock_biomass_estimator,
):
    """All mocked engine instances for integration tests."""
    return {
        "canopy_density_mapper": mock_canopy_density_mapper,
        "forest_type_classifier": mock_forest_type_classifier,
        "historical_reconstructor": mock_historical_reconstructor,
        "deforestation_free_verifier": mock_deforestation_free_verifier,
        "canopy_height_modeler": mock_canopy_height_modeler,
        "fragmentation_analyzer": mock_fragmentation_analyzer,
        "biomass_estimator": mock_biomass_estimator,
    }


# ---------------------------------------------------------------------------
# Computation Helpers for Tests
# ---------------------------------------------------------------------------


def compute_ndvi(red: float, nir: float) -> float:
    """Compute NDVI from red and NIR reflectance values."""
    if (nir + red) == 0.0:
        return 0.0
    return (nir - red) / (nir + red)


def compute_dimidiation_fvc(ndvi: float, ndvi_soil: float, ndvi_veg: float) -> float:
    """Compute fractional vegetation cover using the dimidiation model.

    FVC = ((NDVI - NDVIsoil) / (NDVIveg - NDVIsoil))^2

    The intermediate ratio is clamped to [0, 1] before squaring to
    ensure physically meaningful results (NDVI below bare soil or
    above full vegetation canopy are bounded).
    """
    if ndvi_veg == ndvi_soil:
        return 0.0
    ratio = (ndvi - ndvi_soil) / (ndvi_veg - ndvi_soil)
    ratio = max(0.0, min(1.0, ratio))
    return ratio ** 2


def classify_density(density_pct: float) -> str:
    """Classify canopy density into 6 classes based on percentage."""
    if density_pct >= 80.0:
        return "VERY_HIGH"
    elif density_pct >= 60.0:
        return "HIGH"
    elif density_pct >= 40.0:
        return "MODERATE"
    elif density_pct >= 20.0:
        return "LOW"
    elif density_pct >= 10.0:
        return "VERY_LOW"
    else:
        return "SPARSE"


def check_fao_forest(
    canopy_density_pct: float,
    area_ha: float,
    height_m: float = 5.0,
    canopy_threshold: float = FAO_CANOPY_COVER_PCT,
    area_threshold: float = FAO_MIN_AREA_HA,
    height_threshold: float = FAO_TREE_HEIGHT_M,
) -> bool:
    """Check if a plot meets the FAO forest definition."""
    return (
        canopy_density_pct >= canopy_threshold
        and area_ha >= area_threshold
        and height_m >= height_threshold
    )


def compute_canopy_change_pct(
    cutoff_density: float,
    current_density: float,
) -> float:
    """Compute percentage change in canopy density from cutoff to current."""
    if cutoff_density == 0.0:
        return 0.0
    return ((current_density - cutoff_density) / cutoff_density) * 100.0


def fuse_weighted(
    values: Dict[str, float],
    weights: Dict[str, float],
) -> float:
    """Compute weighted fusion of values, re-normalizing for available sources."""
    available = {k: v for k, v in values.items() if k in weights}
    if not available:
        return 0.0
    total_weight = sum(weights[k] for k in available)
    if total_weight == 0.0:
        return 0.0
    return sum(v * weights[k] / total_weight for k, v in available.items())


def weighted_rms_uncertainty(
    uncertainties: Dict[str, float],
    weights: Dict[str, float],
) -> float:
    """Compute weighted RMS uncertainty from multiple sources."""
    available = {k: v for k, v in uncertainties.items() if k in weights}
    if not available:
        return 0.0
    total_weight = sum(weights[k] for k in available)
    if total_weight == 0.0:
        return 0.0
    sum_sq = sum(
        (weights[k] / total_weight) ** 2 * v ** 2
        for k, v in available.items()
    )
    return math.sqrt(sum_sq)
