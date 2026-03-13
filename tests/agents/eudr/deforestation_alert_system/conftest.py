# -*- coding: utf-8 -*-
"""
Shared pytest fixtures for AGENT-EUDR-020 Deforestation Alert System test suite.

Provides reusable fixtures for configuration objects, engine instances,
satellite detection samples, deforestation alert samples, severity scores,
buffer zones, historical baselines, workflow states, coordinate locations,
NDVI thresholds, provenance tracking helpers, and shared constants used
across all test modules.

Fixture count: 30+ fixtures
Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-020 Deforestation Alert System Agent (GL-EUDR-DAS-020)
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

from greenlang.agents.eudr.deforestation_alert_system.config import (
    DeforestationAlertSystemConfig,
    get_config,
    set_config,
    reset_config,
)
from greenlang.agents.eudr.deforestation_alert_system.models import (
    SatelliteSource,
    ChangeType,
    AlertSeverity,
    AlertStatus,
    BufferType,
    CutoffResult,
    ComplianceOutcome,
    WorkflowAction,
    EUDRCommodity,
    SpectralIndex,
    EvidenceQuality,
    RemediationAction,
    SatelliteDetection,
    DeforestationAlert,
    SeverityScore,
    SpatialBuffer,
    BufferViolation,
    CutoffVerification,
    HistoricalBaseline,
    BaselineComparison,
    WorkflowState,
    WorkflowTransition,
    ComplianceImpact,
    AuditLogEntry,
    EUDR_CUTOFF_DATE,
    SUPPORTED_SATELLITE_SOURCES,
    SUPPORTED_SPECTRAL_INDICES,
    SUPPORTED_COMMODITIES,
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
EUDR_CUTOFF_DATE_OBJ = date(2020, 12, 31)

EUDR_COMMODITIES = [
    "cattle", "cocoa", "coffee", "palm_oil", "rubber", "soya", "wood",
]

# Satellite sources
SATELLITE_SOURCES = [
    "sentinel2", "landsat8", "landsat9", "glad", "hansen_gfc", "radd",
]

# Severity levels in descending order
SEVERITY_LEVELS = ["critical", "high", "medium", "low", "informational"]

# Alert statuses
ALERT_STATUSES = [
    "pending", "triaged", "investigating", "resolved",
    "escalated", "false_positive", "expired",
]

# Change types
CHANGE_TYPES = [
    "deforestation", "degradation", "fire", "logging",
    "clearing", "regrowth", "no_change",
]

# NDVI thresholds for change classification
NDVI_THRESHOLDS = {
    "deforestation": Decimal("-0.15"),
    "degradation": Decimal("-0.05"),
    "regrowth": Decimal("0.10"),
}

# Severity score thresholds (total score -> severity level)
SEVERITY_THRESHOLDS = {
    "critical": Decimal("80"),
    "high": Decimal("60"),
    "medium": Decimal("40"),
    "low": Decimal("20"),
    "informational": Decimal("0"),
}

# Default severity weights
DEFAULT_SEVERITY_WEIGHTS = {
    "area": Decimal("0.25"),
    "rate": Decimal("0.20"),
    "proximity": Decimal("0.25"),
    "protected": Decimal("0.15"),
    "timing": Decimal("0.15"),
}

# Area scoring thresholds (ha -> score)
AREA_SCORE_MAP = {
    50: 100,   # >= 50 ha
    10: 80,    # >= 10 ha
    1: 50,     # >= 1 ha
    0.5: 30,   # >= 0.5 ha
    0: 10,     # < 0.5 ha
}

# Proximity scoring thresholds (km -> score)
PROXIMITY_SCORE_MAP = {
    1: 100,    # < 1 km
    5: 80,     # < 5 km
    25: 50,    # < 25 km
    50: 30,    # < 50 km
}

# High-risk EUDR countries
HIGH_RISK_COUNTRIES = [
    "BR", "ID", "CO", "MY", "PY", "CM", "CI", "GH", "NG", "CD",
    "CG", "PE", "BO", "VN", "TH", "MM",
]


# ---------------------------------------------------------------------------
# Configuration Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_config():
    """Create a DeforestationAlertSystemConfig with test defaults."""
    return DeforestationAlertSystemConfig(
        database_url="postgresql://localhost:5432/greenlang_test",
        redis_url="redis://localhost:6379/1",
        log_level="DEBUG",
        sentinel2_enabled=True,
        landsat_enabled=True,
        glad_enabled=True,
        hansen_gfc_enabled=True,
        radd_enabled=True,
        ndvi_change_threshold=Decimal("-0.15"),
        evi_change_threshold=Decimal("-0.12"),
        min_clearing_area_ha=Decimal("0.5"),
        confidence_threshold=Decimal("0.75"),
        dedup_window_hours=72,
        critical_area_threshold_ha=Decimal("50"),
        high_area_threshold_ha=Decimal("10"),
        medium_area_threshold_ha=Decimal("1"),
        proximity_critical_km=Decimal("1"),
        proximity_high_km=Decimal("5"),
        proximity_medium_km=Decimal("25"),
        protected_area_multiplier=Decimal("1.5"),
        post_cutoff_multiplier=Decimal("2.0"),
        default_buffer_radius_km=Decimal("10"),
        min_buffer_km=Decimal("1"),
        max_buffer_km=Decimal("50"),
        cutoff_date="2020-12-31",
        baseline_start_year=2018,
        baseline_end_year=2020,
        min_baseline_samples=3,
        canopy_cover_threshold_pct=Decimal("10"),
        auto_triage_enabled=True,
        sla_triage_hours=4,
        sla_investigation_hours=48,
        sla_resolution_hours=168,
        enable_provenance=True,
        genesis_hash="GL-EUDR-DAS-020-TEST-GENESIS",
        enable_metrics=False,
        batch_max_size=100,
        batch_concurrency=4,
    )


@pytest.fixture
def strict_config():
    """Create a DeforestationAlertSystemConfig with strict thresholds."""
    return DeforestationAlertSystemConfig(
        database_url="postgresql://localhost:5432/greenlang_test",
        redis_url="redis://localhost:6379/1",
        log_level="DEBUG",
        ndvi_change_threshold=Decimal("-0.10"),
        evi_change_threshold=Decimal("-0.08"),
        min_clearing_area_ha=Decimal("0.1"),
        confidence_threshold=Decimal("0.90"),
        dedup_window_hours=24,
        critical_area_threshold_ha=Decimal("50"),
        high_area_threshold_ha=Decimal("10"),
        medium_area_threshold_ha=Decimal("1"),
        proximity_critical_km=Decimal("1"),
        proximity_high_km=Decimal("5"),
        proximity_medium_km=Decimal("25"),
        protected_area_multiplier=Decimal("2.0"),
        post_cutoff_multiplier=Decimal("2.5"),
        default_buffer_radius_km=Decimal("5"),
        min_buffer_km=Decimal("1"),
        max_buffer_km=Decimal("50"),
        cutoff_date="2020-12-31",
        enable_provenance=True,
        genesis_hash="GL-EUDR-DAS-020-TEST-STRICT-GENESIS",
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
# Mock Fixtures (Provenance, Metrics)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_provenance():
    """Create a mock ProvenanceTracker for testing."""
    tracker = MagicMock()
    tracker.genesis_hash = compute_test_hash({"genesis": "GL-EUDR-DAS-020"})
    tracker._entries = []
    tracker.entry_count = 0

    def record_side_effect(entity_type, action, entity_id, data=None, metadata=None):
        entry = {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "action": action,
            "hash_value": compute_test_hash({
                "entity_type": entity_type,
                "entity_id": entity_id,
                "action": action,
            }),
            "parent_hash": (
                tracker.genesis_hash
                if not tracker._entries
                else tracker._entries[-1]["hash_value"]
            ),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        }
        tracker._entries.append(entry)
        tracker.entry_count = len(tracker._entries)
        return entry

    tracker.record = MagicMock(side_effect=record_side_effect)
    tracker.verify_chain = MagicMock(return_value=True)
    tracker.get_entries = MagicMock(return_value=[])
    tracker.build_hash = MagicMock(side_effect=lambda d: compute_test_hash(d))
    tracker.clear = MagicMock()
    return tracker


@pytest.fixture
def mock_metrics():
    """Create a mock MetricsCollector for testing."""
    metrics = MagicMock()
    metrics.increment = MagicMock()
    metrics.observe = MagicMock()
    metrics.set_gauge = MagicMock()
    metrics.start_timer = MagicMock(return_value=MagicMock())
    return metrics


# ---------------------------------------------------------------------------
# Sample Detection Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_detection():
    """Sample SatelliteDetection: Brazil Amazon, -3.1, -60.0, 5.5ha deforestation."""
    return SatelliteDetection(
        detection_id="det-sentinel2-2025-001",
        source=SatelliteSource.SENTINEL2,
        timestamp=datetime(2025, 6, 15, 10, 30, 0, tzinfo=timezone.utc),
        latitude=Decimal("-3.1"),
        longitude=Decimal("-60.0"),
        area_ha=Decimal("5.5"),
        change_type=ChangeType.DEFORESTATION,
        confidence=Decimal("0.92"),
        spectral_indices={
            "ndvi_before": Decimal("0.75"),
            "ndvi_after": Decimal("0.15"),
            "ndvi_change": Decimal("-0.60"),
        },
        cloud_cover_pct=Decimal("8.5"),
        resolution_m=10,
        tile_id="T20MQS",
        provenance_hash=compute_test_hash({
            "detection_id": "det-sentinel2-2025-001",
            "source": "sentinel2",
            "latitude": "-3.1",
            "longitude": "-60.0",
        }),
    )


@pytest.fixture
def sample_detections():
    """List of 5 sample detections in different countries."""
    detections = []
    specs = [
        ("det-001", SatelliteSource.SENTINEL2, Decimal("-3.1"), Decimal("-60.0"),
         "BR", Decimal("5.5"), ChangeType.DEFORESTATION, Decimal("0.92")),
        ("det-002", SatelliteSource.LANDSAT8, Decimal("-1.5"), Decimal("116.0"),
         "ID", Decimal("12.0"), ChangeType.CLEARING, Decimal("0.88")),
        ("det-003", SatelliteSource.GLAD, Decimal("6.5"), Decimal("-1.6"),
         "GH", Decimal("2.3"), ChangeType.DEGRADATION, Decimal("0.80")),
        ("det-004", SatelliteSource.RADD, Decimal("0.5"), Decimal("25.0"),
         "CD", Decimal("8.0"), ChangeType.LOGGING, Decimal("0.85")),
        ("det-005", SatelliteSource.HANSEN_GFC, Decimal("-12.5"), Decimal("-55.3"),
         "BR", Decimal("0.3"), ChangeType.FIRE, Decimal("0.78")),
    ]
    for det_id, source, lat, lon, country, area, change, conf in specs:
        detections.append(SatelliteDetection(
            detection_id=det_id,
            source=source,
            timestamp=datetime(2025, 6, 15, 10, 30, 0, tzinfo=timezone.utc),
            latitude=lat,
            longitude=lon,
            area_ha=area,
            change_type=change,
            confidence=conf,
            spectral_indices={},
            provenance_hash=compute_test_hash({
                "detection_id": det_id,
                "source": source.value,
            }),
        ))
    return detections


# ---------------------------------------------------------------------------
# Sample Alert Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_alert():
    """Sample DeforestationAlert: CRITICAL, post-cutoff, 5.5ha, Brazil."""
    return DeforestationAlert(
        alert_id="alert-001",
        detection_id="det-sentinel2-2025-001",
        severity=AlertSeverity.CRITICAL,
        status=AlertStatus.PENDING,
        title="Critical deforestation detected in Brazilian Amazon",
        description="5.5 hectares of forest loss detected near supply chain plot",
        area_ha=Decimal("5.5"),
        latitude=Decimal("-3.1"),
        longitude=Decimal("-60.0"),
        country_code="BR",
        affected_plots=["PLOT-BR-001", "PLOT-BR-002"],
        affected_commodities=[EUDRCommodity.SOYA, EUDRCommodity.CATTLE],
        proximity_km=Decimal("2.5"),
        is_post_cutoff=True,
        detection_sources=[SatelliteSource.SENTINEL2],
        provenance_hash=compute_test_hash({
            "alert_id": "alert-001",
            "severity": "critical",
            "area_ha": "5.5",
        }),
    )


@pytest.fixture
def sample_alerts():
    """List of alerts at different severity levels."""
    alerts = []
    specs = [
        ("alert-crit", AlertSeverity.CRITICAL, Decimal("55.0"), "BR", True),
        ("alert-high", AlertSeverity.HIGH, Decimal("15.0"), "ID", True),
        ("alert-med", AlertSeverity.MEDIUM, Decimal("3.0"), "GH", False),
        ("alert-low", AlertSeverity.LOW, Decimal("0.8"), "CD", False),
        ("alert-info", AlertSeverity.INFORMATIONAL, Decimal("0.2"), "CO", False),
    ]
    for aid, sev, area, country, post_cutoff in specs:
        alerts.append(DeforestationAlert(
            alert_id=aid,
            detection_id=f"det-for-{aid}",
            severity=sev,
            status=AlertStatus.PENDING,
            title=f"{sev.value.title()} alert in {country}",
            area_ha=area,
            latitude=Decimal("-3.0"),
            longitude=Decimal("-60.0"),
            country_code=country,
            is_post_cutoff=post_cutoff,
            provenance_hash=compute_test_hash({
                "alert_id": aid,
                "severity": sev.value,
            }),
        ))
    return alerts


# ---------------------------------------------------------------------------
# Buffer Zone Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_buffer_zones():
    """List of buffer zones around test plots."""
    buffers = []
    specs = [
        ("buf-001", "PLOT-BR-001", Decimal("-3.1"), Decimal("-60.0"),
         Decimal("10"), BufferType.CIRCULAR, "BR"),
        ("buf-002", "PLOT-ID-001", Decimal("-1.5"), Decimal("116.0"),
         Decimal("5"), BufferType.ADAPTIVE, "ID"),
        ("buf-003", "PLOT-GH-001", Decimal("6.5"), Decimal("-1.6"),
         Decimal("8"), BufferType.POLYGON, "GH"),
    ]
    for bid, pid, lat, lon, radius, btype, country in specs:
        buffers.append(SpatialBuffer(
            buffer_id=bid,
            plot_id=pid,
            center_lat=lat,
            center_lon=lon,
            radius_km=radius,
            buffer_type=btype,
            active=True,
            commodities=[EUDRCommodity.SOYA],
            country_code=country,
            provenance_hash=compute_test_hash({
                "buffer_id": bid,
                "plot_id": pid,
            }),
        ))
    return buffers


# ---------------------------------------------------------------------------
# Historical Baseline Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_baseline():
    """Historical baseline: 2018-2020, 85% canopy, 120ha forest."""
    return HistoricalBaseline(
        baseline_id="baseline-001",
        plot_id="PLOT-BR-001",
        latitude=Decimal("-3.1"),
        longitude=Decimal("-60.0"),
        baseline_period="2018-2020",
        canopy_cover_pct=Decimal("85"),
        forest_area_ha=Decimal("120"),
        reference_images=[
            "S2A_20180615_T20MQS",
            "S2A_20190301_T20MQS",
            "S2A_20200915_T20MQS",
        ],
        num_observations=6,
        provenance_hash=compute_test_hash({
            "baseline_id": "baseline-001",
            "plot_id": "PLOT-BR-001",
            "canopy_cover_pct": "85",
        }),
    )


# ---------------------------------------------------------------------------
# Workflow State Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_workflow_state():
    """WorkflowState in PENDING status."""
    return WorkflowState(
        state_id="wf-state-001",
        alert_id="alert-001",
        current_status=AlertStatus.PENDING,
        assigned_to=None,
        priority=2,
        sla_deadline=datetime(2025, 6, 16, 14, 30, 0, tzinfo=timezone.utc),
        escalation_level=0,
        notes=["Auto-generated from satellite detection"],
        transitions=[],
        provenance_hash=compute_test_hash({
            "state_id": "wf-state-001",
            "alert_id": "alert-001",
        }),
    )


# ---------------------------------------------------------------------------
# EUDR Cutoff Date Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def eudr_cutoff_date():
    """EUDR cutoff date: 31 December 2020."""
    return date(2020, 12, 31)


# ---------------------------------------------------------------------------
# Sample Coordinates
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_coordinates():
    """Dictionary of test locations with (lat, lon) tuples."""
    return {
        "brazil_amazon": (Decimal("-3.1"), Decimal("-60.0")),
        "indonesia_borneo": (Decimal("-1.5"), Decimal("116.0")),
        "drc_congo": (Decimal("0.5"), Decimal("25.0")),
        "denmark_copenhagen": (Decimal("55.7"), Decimal("12.6")),
        "ghana_ashanti": (Decimal("6.5"), Decimal("-1.6")),
        "brazil_mato_grosso": (Decimal("-12.5"), Decimal("-55.3")),
    }


# ---------------------------------------------------------------------------
# Engine Mock Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def satellite_detector(mock_config):
    """Create a mock SatelliteChangeDetector for testing."""
    detector = MagicMock()
    detector.config = mock_config
    detector.detect_changes = MagicMock(return_value=[])
    detector.scan_area = MagicMock(return_value=[])
    detector.get_available_sources = MagicMock(
        return_value=["sentinel2", "landsat8", "glad"]
    )
    detector.calculate_ndvi = MagicMock(
        side_effect=lambda red, nir: (
            (nir - red) / (nir + red) if (nir + red) != 0 else Decimal("0")
        )
    )
    detector.calculate_evi = MagicMock(
        side_effect=lambda blue, red, nir: (
            Decimal("2.5") * (nir - red) / (nir + Decimal("6") * red
            - Decimal("7.5") * blue + Decimal("1"))
            if (nir + Decimal("6") * red - Decimal("7.5") * blue
                + Decimal("1")) != 0
            else Decimal("0")
        )
    )
    return detector


@pytest.fixture
def alert_generator(mock_config):
    """Create a mock AlertGenerator for testing."""
    gen = MagicMock()
    gen.config = mock_config
    gen.generate_alert = MagicMock(return_value=None)
    gen.generate_batch = MagicMock(return_value=[])
    gen.get_alert = MagicMock(return_value=None)
    gen.list_alerts = MagicMock(return_value=[])
    gen.get_alert_summary = MagicMock(return_value={})
    gen.get_alert_statistics = MagicMock(return_value={})
    return gen


@pytest.fixture
def severity_classifier(mock_config):
    """Create a mock SeverityClassifier for testing."""
    cls = MagicMock()
    cls.config = mock_config
    cls.classify = MagicMock(return_value=None)
    cls.reclassify = MagicMock(return_value=None)
    cls.get_thresholds = MagicMock(return_value={})
    cls.get_distribution = MagicMock(return_value={})
    return cls


@pytest.fixture
def buffer_monitor(mock_config):
    """Create a mock SpatialBufferMonitor for testing."""
    monitor = MagicMock()
    monitor.config = mock_config
    monitor.check_buffer = MagicMock(return_value=[])
    monitor.create_buffer = MagicMock(return_value=None)
    monitor.update_buffer = MagicMock(return_value=None)
    monitor.list_buffers = MagicMock(return_value=[])
    return monitor


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


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute Haversine distance between two points in kilometers."""
    R = 6371.0  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def score_area(area_ha: float) -> int:
    """Score area component for severity classification."""
    if area_ha >= 50:
        return 100
    elif area_ha >= 10:
        return 80
    elif area_ha >= 1:
        return 50
    elif area_ha >= 0.5:
        return 30
    else:
        return 10


def score_proximity(distance_km: float) -> int:
    """Score proximity component for severity classification."""
    if distance_km < 1:
        return 100
    elif distance_km < 5:
        return 80
    elif distance_km < 25:
        return 50
    elif distance_km < 50:
        return 30
    else:
        return 10


def determine_severity(total_score: float) -> str:
    """Determine severity level from weighted total score."""
    if total_score >= 80:
        return "critical"
    elif total_score >= 60:
        return "high"
    elif total_score >= 40:
        return "medium"
    elif total_score >= 20:
        return "low"
    else:
        return "informational"


def classify_ndvi_change(
    ndvi_before: float,
    ndvi_after: float,
    deforestation_threshold: float = -0.15,
    degradation_threshold: float = -0.05,
    regrowth_threshold: float = 0.10,
) -> str:
    """Classify NDVI change into categories."""
    delta = ndvi_after - ndvi_before
    if delta <= deforestation_threshold:
        return "deforestation"
    elif delta <= degradation_threshold:
        return "degradation"
    elif delta >= regrowth_threshold:
        return "regrowth"
    else:
        return "no_change"


def is_post_cutoff(detection_date: date, cutoff: date = None) -> bool:
    """Determine if a detection date is after the EUDR cutoff."""
    if cutoff is None:
        cutoff = date(2020, 12, 31)
    return detection_date > cutoff
