# -*- coding: utf-8 -*-
"""
Shared pytest fixtures for AGENT-EUDR-005 Land Use Change Detector test suite.

Provides reusable fixtures for configuration objects, engine instances,
spectral data, coordinate locations, NDVI time series, region bounds,
classification results, transition results, trajectory results, cutoff
verification results, cropland expansion results, conversion risk
results, urban encroachment results, and provenance tracking used
across all test modules.

Fixture count: 34+ fixtures
Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-005 Land Use Change Detector Agent (GL-EUDR-LUC-005)
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

from greenlang.agents.eudr.land_use_change.config import (
    LandUseChangeConfig,
    get_config,
    set_config,
    reset_config,
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
EUDR_CUTOFF_DATE = date(2020, 12, 31)

EUDR_COMMODITIES = [
    "cattle", "cocoa", "coffee", "palm_oil", "rubber", "soya", "wood",
]

# 10 IPCC land use categories
LAND_USE_CATEGORIES = [
    "forest",
    "shrubland",
    "grassland",
    "cropland",
    "wetland",
    "water",
    "urban",
    "bare_soil",
    "snow_ice",
    "other",
]

# 5 classification methods
CLASSIFICATION_METHODS = [
    "spectral",
    "vegetation_index",
    "phenology",
    "texture",
    "ensemble",
]

# Transition types
TRANSITION_TYPES = [
    "deforestation",
    "degradation",
    "reforestation",
    "urbanization",
    "agricultural_expansion",
    "wetland_drainage",
    "stable",
    "unknown",
]

# Trajectory types
TRAJECTORY_TYPES = [
    "stable",
    "abrupt_change",
    "gradual_change",
    "oscillating",
    "recovery",
]

# Compliance verdicts
COMPLIANCE_VERDICTS = [
    "compliant",
    "non_compliant",
    "degraded",
    "inconclusive",
    "pre_existing_agriculture",
]

# Conversion types for cropland expansion
CONVERSION_TYPES = [
    "palm_oil_conversion",
    "rubber_conversion",
    "cocoa_conversion",
    "coffee_conversion",
    "soya_conversion",
    "pasture_conversion",
    "timber_plantation_conversion",
]

# Risk tiers
RISK_TIERS = [
    "low",
    "moderate",
    "high",
    "critical",
]

# 8 risk factors
RISK_FACTORS = [
    "transition_magnitude",
    "proximity_to_forest",
    "historical_deforestation_rate",
    "commodity_pressure",
    "governance_score",
    "protected_area_proximity",
    "road_infrastructure_proximity",
    "population_density_change",
]

# Default risk weights (must sum to 1.0)
DEFAULT_RISK_WEIGHTS = {
    "transition_magnitude": 0.20,
    "proximity_to_forest": 0.15,
    "historical_deforestation_rate": 0.15,
    "commodity_pressure": 0.12,
    "governance_score": 0.10,
    "protected_area_proximity": 0.10,
    "road_infrastructure_proximity": 0.08,
    "population_density_change": 0.10,
}

# Infrastructure types for urban encroachment
INFRASTRUCTURE_TYPES = [
    "road_construction",
    "building_expansion",
    "mining_activity",
    "industrial_development",
    "residential_growth",
]

# Report types
REPORT_TYPES = [
    "full",
    "summary",
    "compliance",
    "evidence",
]

# Report formats
REPORT_FORMATS = [
    "json",
    "pdf",
    "csv",
    "eudr_xml",
]

# Scale classifications for cropland expansion
EXPANSION_SCALES = [
    "smallholder",
    "medium",
    "industrial",
]

# Sentinel-2 spectral bands
SPECTRAL_BANDS = [
    "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12",
]

# Ensemble method weights (must sum to 1.0)
ENSEMBLE_WEIGHTS = {
    "spectral": 0.25,
    "vegetation_index": 0.25,
    "phenology": 0.20,
    "texture": 0.15,
    "ensemble": 0.15,
}

# EUDR regulatory references by verdict
VERDICT_REGULATORY_REFS = {
    "compliant": ["EUDR Art. 3(a)", "EUDR Art. 10(1)"],
    "non_compliant": ["EUDR Art. 3(b)", "EUDR Art. 10(2)"],
    "degraded": ["EUDR Art. 2(6)", "EUDR Art. 10(2)"],
    "inconclusive": ["EUDR Art. 10(3)", "EUDR Art. 11(1)"],
    "pre_existing_agriculture": ["EUDR Art. 2(4)", "EUDR Art. 10(1)"],
}


# ---------------------------------------------------------------------------
# Test-only Dataclass Models (mirrors of production models)
# ---------------------------------------------------------------------------


@dataclass
class SpectralData:
    """Test-only spectral reflectance data for a single pixel."""

    blue: float = 0.0
    green: float = 0.0
    red: float = 0.0
    red_edge_1: float = 0.0
    red_edge_2: float = 0.0
    red_edge_3: float = 0.0
    nir: float = 0.0
    narrow_nir: float = 0.0
    swir1: float = 0.0
    swir2: float = 0.0


@dataclass
class VegetationIndices:
    """Test-only vegetation indices computed from spectral data."""

    ndvi: float = 0.0
    evi: float = 0.0
    savi: float = 0.0
    ndmi: float = 0.0
    nbr: float = 0.0


@dataclass
class TextureFeatures:
    """Test-only GLCM texture features."""

    contrast: float = 0.0
    dissimilarity: float = 0.0
    homogeneity: float = 0.0
    energy: float = 0.0
    correlation: float = 0.0
    entropy: float = 0.0


@dataclass
class PhenologyTimeSeries:
    """Test-only phenology time series for seasonal classification."""

    dates: List[str] = field(default_factory=list)
    ndvi_values: List[float] = field(default_factory=list)
    period_months: int = 12
    peak_ndvi: float = 0.0
    trough_ndvi: float = 0.0
    amplitude: float = 0.0


@dataclass
class LandUseClassification:
    """Test-only result of land use classification for a single plot."""

    plot_id: str = ""
    category: str = "other"
    method: str = "ensemble"
    confidence: float = 0.0
    spectral_class: str = ""
    vi_class: str = ""
    phenology_class: str = ""
    texture_class: str = ""
    ensemble_class: str = ""
    all_method_results: Dict[str, str] = field(default_factory=dict)
    commodity_context: str = ""
    article_2_4_applies: bool = False
    provenance_hash: str = ""


@dataclass
class LandUseTransition:
    """Test-only result of land use transition detection."""

    plot_id: str = ""
    from_category: str = ""
    to_category: str = ""
    transition_type: str = "stable"
    transition_date: str = ""
    confidence: float = 0.0
    area_ha: float = 0.0
    is_deforestation: bool = False
    is_degradation: bool = False
    evidence: Dict[str, Any] = field(default_factory=dict)
    provenance_hash: str = ""


@dataclass
class TransitionMatrix:
    """Test-only transition matrix for a region."""

    region_id: str = ""
    period_start: str = ""
    period_end: str = ""
    matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)
    total_area_ha: float = 0.0
    provenance_hash: str = ""


@dataclass
class TemporalTrajectory:
    """Test-only temporal trajectory analysis result."""

    plot_id: str = ""
    trajectory_type: str = "stable"
    dates: List[str] = field(default_factory=list)
    ndvi_values: List[float] = field(default_factory=list)
    change_date: Optional[str] = None
    change_date_range: Optional[Tuple[str, str]] = None
    oscillation_period_months: Optional[int] = None
    recovery_completeness: Optional[float] = None
    confidence: float = 0.0
    is_natural_disturbance: bool = False
    visualization_data: Dict[str, Any] = field(default_factory=dict)
    provenance_hash: str = ""


@dataclass
class CutoffVerification:
    """Test-only cutoff date verification result."""

    plot_id: str = ""
    verdict: str = "inconclusive"
    cutoff_category: str = ""
    current_category: str = ""
    cutoff_confidence: float = 0.0
    current_confidence: float = 0.0
    transition_detected: bool = False
    transition_date: Optional[str] = None
    commodity: str = ""
    article_2_4_applies: bool = False
    cross_validation_score: float = 0.0
    evidence: Dict[str, Any] = field(default_factory=dict)
    regulatory_references: List[str] = field(default_factory=list)
    provenance_hash: str = ""


@dataclass
class CroplandExpansion:
    """Test-only cropland expansion detection result."""

    plot_id: str = ""
    conversion_type: str = ""
    from_category: str = ""
    to_category: str = "cropland"
    commodity: str = ""
    scale: str = "smallholder"
    expansion_rate_ha_per_year: float = 0.0
    area_converted_ha: float = 0.0
    is_hotspot: bool = False
    leapfrog_pattern: bool = False
    confidence: float = 0.0
    provenance_hash: str = ""


@dataclass
class ConversionRiskAssessment:
    """Test-only conversion risk assessment result."""

    plot_id: str = ""
    risk_tier: str = "low"
    composite_score: float = 0.0
    factor_scores: Dict[str, float] = field(default_factory=dict)
    factor_weights: Dict[str, float] = field(default_factory=dict)
    conversion_probability_6m: float = 0.0
    conversion_probability_12m: float = 0.0
    conversion_probability_24m: float = 0.0
    is_deforestation_frontier: bool = False
    heatmap_data: Dict[str, Any] = field(default_factory=dict)
    provenance_hash: str = ""


@dataclass
class UrbanEncroachment:
    """Test-only urban encroachment analysis result."""

    plot_id: str = ""
    infrastructure_type: str = ""
    expansion_rate_ha_per_year: float = 0.0
    pressure_corridors: List[Dict[str, Any]] = field(default_factory=list)
    time_to_conversion_years: Optional[float] = None
    urban_proximity_km: float = 0.0
    buffer_zone_risk: float = 0.0
    confidence: float = 0.0
    provenance_hash: str = ""


@dataclass
class ComplianceReport:
    """Test-only compliance report result."""

    report_id: str = ""
    plot_id: str = ""
    report_type: str = "full"
    report_format: str = "json"
    verdict: str = "inconclusive"
    summary: str = ""
    created_at: str = ""
    provenance_hash: str = ""
    regulatory_framework: str = "EUDR EU 2023/1115"


@dataclass
class ProvenanceRecord:
    """Test-only provenance record."""

    entity_type: str = ""
    entity_id: str = ""
    action: str = ""
    hash_value: str = ""
    parent_hash: str = ""
    timestamp: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Configuration Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_config():
    """Create a LandUseChangeConfig with test defaults."""
    return LandUseChangeConfig(
        database_url="postgresql://localhost:5432/greenlang_test",
        redis_url="redis://localhost:6379/1",
        log_level="DEBUG",
        num_classes=10,
        default_method="ensemble",
        min_confidence=0.60,
        min_transition_area_ha=0.1,
        transition_date_granularity="monthly",
        deforestation_precision_target=0.90,
        min_temporal_depth_years=3,
        max_time_steps=60,
        cutoff_date="2020-12-31",
        search_window_days=60,
        conservative_bias=True,
        default_buffer_km=10.0,
        max_buffer_km=50.0,
        batch_size=100,
        max_concurrent_jobs=4,
        cache_ttl_seconds=300,
        genesis_hash="GL-EUDR-LUC-005-TEST-GENESIS",
        enable_metrics=False,
    )


@pytest.fixture
def strict_config():
    """Create a LandUseChangeConfig with strict/conservative thresholds."""
    return LandUseChangeConfig(
        database_url="postgresql://localhost:5432/greenlang_test",
        redis_url="redis://localhost:6379/1",
        log_level="DEBUG",
        num_classes=10,
        default_method="ensemble",
        min_confidence=0.80,
        min_transition_area_ha=0.01,
        transition_date_granularity="weekly",
        deforestation_precision_target=0.95,
        min_temporal_depth_years=5,
        max_time_steps=120,
        cutoff_date="2020-12-31",
        search_window_days=30,
        conservative_bias=True,
        default_buffer_km=5.0,
        max_buffer_km=50.0,
        batch_size=50,
        max_concurrent_jobs=2,
        cache_ttl_seconds=60,
        genesis_hash="GL-EUDR-LUC-005-TEST-STRICT-GENESIS",
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
# Engine Mock Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def classifier(sample_config):
    """Mocked LandUseClassifier engine."""
    c = MagicMock()
    c.config = sample_config
    c.classify = MagicMock(return_value=LandUseClassification(
        plot_id="PLOT-MOCK",
        category="forest",
        method="ensemble",
        confidence=0.90,
        spectral_class="forest",
        vi_class="forest",
        phenology_class="forest",
        texture_class="forest",
        ensemble_class="forest",
    ))
    c.classify_batch = MagicMock(return_value=[
        LandUseClassification(plot_id=f"PLOT-{i}", category="forest", confidence=0.85)
        for i in range(10)
    ])
    return c


@pytest.fixture
def transition_detector(sample_config):
    """Mocked TransitionDetector engine."""
    td = MagicMock()
    td.config = sample_config
    td.detect = MagicMock(return_value=LandUseTransition(
        plot_id="PLOT-MOCK",
        from_category="forest",
        to_category="forest",
        transition_type="stable",
        confidence=0.92,
    ))
    td.detect_batch = MagicMock(return_value=[])
    td.generate_transition_matrix = MagicMock(return_value=TransitionMatrix(
        region_id="REGION-MOCK",
        period_start="2020-01-01",
        period_end="2024-01-01",
    ))
    return td


@pytest.fixture
def trajectory_analyzer(sample_config):
    """Mocked TemporalTrajectoryAnalyzer engine."""
    ta = MagicMock()
    ta.config = sample_config
    ta.analyze = MagicMock(return_value=TemporalTrajectory(
        plot_id="PLOT-MOCK",
        trajectory_type="stable",
        confidence=0.88,
    ))
    ta.analyze_batch = MagicMock(return_value=[])
    return ta


@pytest.fixture
def cutoff_verifier(sample_config):
    """Mocked CutoffDateVerifier engine."""
    cv = MagicMock()
    cv.config = sample_config
    cv.verify = MagicMock(return_value=CutoffVerification(
        plot_id="PLOT-MOCK",
        verdict="compliant",
        cutoff_category="forest",
        current_category="forest",
        cutoff_confidence=0.90,
        current_confidence=0.88,
    ))
    cv.verify_batch = MagicMock(return_value=[])
    return cv


@pytest.fixture
def cropland_detector(sample_config):
    """Mocked CroplandExpansionDetector engine."""
    cd = MagicMock()
    cd.config = sample_config
    cd.detect = MagicMock(return_value=CroplandExpansion(
        plot_id="PLOT-MOCK",
        conversion_type="palm_oil_conversion",
        from_category="forest",
        to_category="cropland",
        commodity="palm_oil",
        scale="industrial",
        confidence=0.85,
    ))
    cd.detect_batch = MagicMock(return_value=[])
    return cd


@pytest.fixture
def risk_assessor(sample_config):
    """Mocked ConversionRiskAssessor engine."""
    ra = MagicMock()
    ra.config = sample_config
    ra.assess = MagicMock(return_value=ConversionRiskAssessment(
        plot_id="PLOT-MOCK",
        risk_tier="moderate",
        composite_score=0.55,
    ))
    ra.assess_batch = MagicMock(return_value=[])
    return ra


@pytest.fixture
def urban_analyzer(sample_config):
    """Mocked UrbanEncroachmentAnalyzer engine."""
    ua = MagicMock()
    ua.config = sample_config
    ua.analyze = MagicMock(return_value=UrbanEncroachment(
        plot_id="PLOT-MOCK",
        infrastructure_type="road_construction",
        expansion_rate_ha_per_year=5.0,
        confidence=0.80,
    ))
    ua.analyze_batch = MagicMock(return_value=[])
    return ua


@pytest.fixture
def reporter(sample_config):
    """Mocked ComplianceReporter engine."""
    rp = MagicMock()
    rp.config = sample_config
    rp.generate = MagicMock(return_value=ComplianceReport(
        report_id="RPT-MOCK",
        plot_id="PLOT-MOCK",
        verdict="compliant",
    ))
    return rp


@pytest.fixture
def land_use_service(
    classifier,
    transition_detector,
    trajectory_analyzer,
    cutoff_verifier,
    cropland_detector,
    risk_assessor,
    urban_analyzer,
    reporter,
):
    """Mocked LandUseChangeService with all engine instances."""
    svc = MagicMock()
    svc.classifier = classifier
    svc.transition_detector = transition_detector
    svc.trajectory_analyzer = trajectory_analyzer
    svc.cutoff_verifier = cutoff_verifier
    svc.cropland_detector = cropland_detector
    svc.risk_assessor = risk_assessor
    svc.urban_analyzer = urban_analyzer
    svc.reporter = reporter
    return svc


@pytest.fixture
def provenance_tracker():
    """Create a fresh ProvenanceTracker-like mock for testing."""
    tracker = MagicMock()
    tracker.genesis_hash = compute_test_hash({"genesis": "GL-EUDR-LUC-005"})
    tracker._entries = []
    tracker.entry_count = 0

    def record_side_effect(entity_type, action, entity_id, data=None, metadata=None):
        entry = ProvenanceRecord(
            entity_type=entity_type,
            entity_id=entity_id,
            action=action,
            hash_value=compute_test_hash({
                "entity_type": entity_type,
                "entity_id": entity_id,
                "action": action,
            }),
            parent_hash=tracker.genesis_hash if not tracker._entries else tracker._entries[-1].hash_value,
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata=metadata or {},
        )
        tracker._entries.append(entry)
        tracker.entry_count = len(tracker._entries)
        return entry

    tracker.record = MagicMock(side_effect=record_side_effect)
    tracker.verify_chain = MagicMock(return_value=True)
    tracker.get_entries = MagicMock(return_value=[])
    tracker.build_hash = MagicMock(side_effect=lambda d: compute_test_hash(d))
    tracker.clear = MagicMock()
    return tracker


# ---------------------------------------------------------------------------
# Spectral Data Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def forest_spectral_data():
    """Typical tropical forest spectral values.

    Low red reflectance (chlorophyll absorption), high NIR (leaf structure).
    NDVI = (0.350 - 0.025) / (0.350 + 0.025) = 0.867
    """
    return SpectralData(
        blue=0.030,
        green=0.055,
        red=0.025,
        red_edge_1=0.120,
        red_edge_2=0.250,
        red_edge_3=0.300,
        nir=0.350,
        narrow_nir=0.340,
        swir1=0.120,
        swir2=0.060,
    )


@pytest.fixture
def cropland_spectral_data():
    """Typical cropland spectral values during growing season.

    Moderate red, moderate NIR.
    NDVI = (0.250 - 0.100) / (0.250 + 0.100) = 0.429
    """
    return SpectralData(
        blue=0.060,
        green=0.080,
        red=0.100,
        red_edge_1=0.130,
        red_edge_2=0.180,
        red_edge_3=0.210,
        nir=0.250,
        narrow_nir=0.240,
        swir1=0.200,
        swir2=0.150,
    )


@pytest.fixture
def palm_oil_spectral_data():
    """Oil palm plantation spectral values.

    Similar to forest but more uniform, slightly lower NIR.
    NDVI = (0.310 - 0.040) / (0.310 + 0.040) = 0.771
    """
    return SpectralData(
        blue=0.035,
        green=0.060,
        red=0.040,
        red_edge_1=0.125,
        red_edge_2=0.230,
        red_edge_3=0.275,
        nir=0.310,
        narrow_nir=0.300,
        swir1=0.140,
        swir2=0.075,
    )


@pytest.fixture
def urban_spectral_data():
    """Urban settlement spectral values.

    High reflectance across visible bands, moderate NIR.
    NDVI = (0.150 - 0.180) / (0.150 + 0.180) = -0.091
    """
    return SpectralData(
        blue=0.120,
        green=0.150,
        red=0.180,
        red_edge_1=0.170,
        red_edge_2=0.165,
        red_edge_3=0.160,
        nir=0.150,
        narrow_nir=0.145,
        swir1=0.220,
        swir2=0.190,
    )


@pytest.fixture
def water_spectral_data():
    """Water body spectral values.

    Very low reflectance, NIR << visible.
    NDVI = (0.020 - 0.060) / (0.020 + 0.060) = -0.500
    """
    return SpectralData(
        blue=0.080,
        green=0.070,
        red=0.060,
        red_edge_1=0.040,
        red_edge_2=0.030,
        red_edge_3=0.025,
        nir=0.020,
        narrow_nir=0.018,
        swir1=0.010,
        swir2=0.005,
    )


@pytest.fixture
def grassland_spectral_data():
    """Grassland spectral values.

    Moderate red, moderate NIR, lower than forest.
    NDVI = (0.220 - 0.090) / (0.220 + 0.090) = 0.419
    """
    return SpectralData(
        blue=0.050,
        green=0.070,
        red=0.090,
        red_edge_1=0.120,
        red_edge_2=0.170,
        red_edge_3=0.195,
        nir=0.220,
        narrow_nir=0.210,
        swir1=0.180,
        swir2=0.130,
    )


@pytest.fixture
def rubber_spectral_data():
    """Rubber plantation spectral values.

    Similar to forest but seasonal leaf-off signature.
    NDVI = (0.300 - 0.035) / (0.300 + 0.035) = 0.791
    """
    return SpectralData(
        blue=0.032,
        green=0.058,
        red=0.035,
        red_edge_1=0.118,
        red_edge_2=0.225,
        red_edge_3=0.268,
        nir=0.300,
        narrow_nir=0.292,
        swir1=0.135,
        swir2=0.070,
    )


# ---------------------------------------------------------------------------
# Coordinate Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def brazil_soya_plot():
    """Mato Grosso soya area coordinates (lat, lon)."""
    return (-12.5, -55.3)


@pytest.fixture
def indonesia_palm_oil_plot():
    """Sumatra palm oil area coordinates (lat, lon)."""
    return (1.5, 103.5)


@pytest.fixture
def ghana_cocoa_plot():
    """Ashanti Region cocoa area coordinates (lat, lon)."""
    return (6.7, -1.6)


@pytest.fixture
def congo_forest_plot():
    """Congo Basin intact forest coordinates (lat, lon)."""
    return (0.5, 25.0)


@pytest.fixture
def malaysia_rubber_plot():
    """Pahang rubber area coordinates (lat, lon)."""
    return (4.2, 103.4)


@pytest.fixture
def ethiopia_coffee_plot():
    """Kaffa coffee area coordinates (lat, lon)."""
    return (7.5, 36.5)


@pytest.fixture
def germany_urban_plot():
    """Berlin peri-urban coordinates (lat, lon)."""
    return (52.5, 13.4)


# ---------------------------------------------------------------------------
# Time Series Fixtures
# ---------------------------------------------------------------------------


def _generate_monthly_dates(start_year: int, num_months: int) -> List[str]:
    """Generate a list of monthly ISO date strings."""
    dates = []
    for i in range(num_months):
        year = start_year + (i // 12)
        month = (i % 12) + 1
        dates.append(f"{year}-{month:02d}-01")
    return dates


@pytest.fixture
def stable_forest_ndvi_series():
    """Stable NDVI ~0.7 for 5 years (60 monthly observations).

    Small random-like noise simulated via deterministic sinusoidal pattern.
    """
    num_months = 60
    dates = _generate_monthly_dates(2018, num_months)
    values = [
        round(0.70 + 0.03 * math.sin(i * 0.5), 4)
        for i in range(num_months)
    ]
    return {"dates": dates, "ndvi_values": values}


@pytest.fixture
def abrupt_deforestation_series():
    """NDVI drops from 0.7 to 0.2 in 1 month (month 30 of 60).

    Simulates rapid deforestation event.
    """
    num_months = 60
    dates = _generate_monthly_dates(2018, num_months)
    values = []
    for i in range(num_months):
        if i < 30:
            values.append(round(0.70 + 0.02 * math.sin(i * 0.5), 4))
        else:
            values.append(round(0.20 + 0.02 * math.sin(i * 0.5), 4))
    return {"dates": dates, "ndvi_values": values}


@pytest.fixture
def gradual_conversion_series():
    """NDVI declines from 0.7 to 0.3 over 12 months (months 24-36).

    Simulates gradual forest-to-cropland conversion.
    """
    num_months = 60
    dates = _generate_monthly_dates(2018, num_months)
    values = []
    for i in range(num_months):
        if i < 24:
            values.append(round(0.70 + 0.02 * math.sin(i * 0.5), 4))
        elif i <= 36:
            frac = (i - 24) / 12.0
            val = 0.70 - 0.40 * frac
            values.append(round(val + 0.02 * math.sin(i * 0.5), 4))
        else:
            values.append(round(0.30 + 0.02 * math.sin(i * 0.5), 4))
    return {"dates": dates, "ndvi_values": values}


@pytest.fixture
def oscillating_crop_series():
    """NDVI oscillates between 0.2 and 0.7 annually.

    Simulates seasonal crop cycle with planting and harvest.
    """
    num_months = 60
    dates = _generate_monthly_dates(2018, num_months)
    values = [
        round(0.45 + 0.25 * math.sin(2 * math.pi * i / 12.0), 4)
        for i in range(num_months)
    ]
    return {"dates": dates, "ndvi_values": values}


@pytest.fixture
def recovery_series():
    """NDVI drops then recovers to 0.5+.

    Simulates disturbance at month 20 with gradual recovery.
    """
    num_months = 60
    dates = _generate_monthly_dates(2018, num_months)
    values = []
    for i in range(num_months):
        if i < 20:
            values.append(round(0.70 + 0.02 * math.sin(i * 0.5), 4))
        elif i == 20:
            values.append(0.25)
        else:
            recovery_months = i - 20
            target = min(0.55, 0.25 + recovery_months * 0.01)
            values.append(round(target + 0.02 * math.sin(i * 0.5), 4))
    return {"dates": dates, "ndvi_values": values}


# ---------------------------------------------------------------------------
# Region Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def amazon_region_bounds():
    """Amazon region bounding box: (lat_min, lon_min, lat_max, lon_max)."""
    return (-15.0, -62.0, -2.0, -50.0)


@pytest.fixture
def se_asia_region_bounds():
    """Southeast Asia region bounding box: (lat_min, lon_min, lat_max, lon_max)."""
    return (-5.0, 95.0, 10.0, 115.0)


# ---------------------------------------------------------------------------
# Computation Helpers for Tests
# ---------------------------------------------------------------------------


def compute_ndvi(red: float, nir: float) -> float:
    """Compute NDVI from red and NIR reflectance values."""
    if (nir + red) == 0.0:
        return 0.0
    return (nir - red) / (nir + red)


def compute_evi(blue: float, red: float, nir: float) -> float:
    """Compute EVI from blue, red, and NIR reflectance values."""
    denom = nir + 6.0 * red - 7.5 * blue + 1.0
    if denom == 0.0:
        return 0.0
    return 2.5 * (nir - red) / denom


def classify_risk_tier(score: float) -> str:
    """Classify composite risk score into tier."""
    if score >= 0.75:
        return "critical"
    elif score >= 0.50:
        return "high"
    elif score >= 0.25:
        return "moderate"
    else:
        return "low"


def determine_verdict(
    cutoff_was_forest: bool,
    current_is_forest: bool,
    confidence: float,
    min_confidence: float = 0.60,
    conservative_bias: bool = True,
) -> str:
    """Determine EUDR compliance verdict from classification states."""
    if confidence < min_confidence:
        return "inconclusive"
    if not cutoff_was_forest:
        return "pre_existing_agriculture"
    if cutoff_was_forest and current_is_forest:
        return "compliant"
    if cutoff_was_forest and not current_is_forest:
        return "non_compliant"
    if conservative_bias:
        return "inconclusive"
    return "inconclusive"


def is_deforestation_transition(from_cat: str, to_cat: str) -> bool:
    """Check if a transition constitutes deforestation."""
    forest_categories = {"forest"}
    agriculture_categories = {"cropland", "grassland"}
    return from_cat in forest_categories and to_cat in agriculture_categories


def is_degradation_transition(from_cat: str, to_cat: str) -> bool:
    """Check if a transition constitutes degradation."""
    return from_cat == "forest" and to_cat == "shrubland"


def weighted_composite(
    values: Dict[str, float],
    weights: Dict[str, float],
) -> float:
    """Compute weighted composite score."""
    available = {k: v for k, v in values.items() if k in weights}
    if not available:
        return 0.0
    total_weight = sum(weights[k] for k in available)
    if total_weight == 0.0:
        return 0.0
    return sum(v * weights[k] / total_weight for k, v in available.items())
