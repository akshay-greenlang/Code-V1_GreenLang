# -*- coding: utf-8 -*-
"""
Shared pytest fixtures for AGENT-EUDR-022 Protected Area Validator test suite.

Provides reusable fixtures for configuration objects, engine instances,
protected area samples, plot data, overlap detection samples, buffer zone
samples, IUCN category data, designation status data, risk scoring data,
violation alert samples, compliance report samples, provenance tracking
helpers, mock PostGIS functions, mock WDPA API, mock Redis cache, mock
authentication, and shared constants used across all test modules.

Fixture count: 60+ fixtures
Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-022 Protected Area Validator (GL-EUDR-PAV-022)
"""

import hashlib
import json
import math
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


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
# Haversine distance helper
# ---------------------------------------------------------------------------


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute Haversine distance between two points in kilometers."""
    R = 6371.0
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


# ---------------------------------------------------------------------------
# Shared Constants
# ---------------------------------------------------------------------------

SHA256_HEX_LENGTH = 64

EUDR_CUTOFF_DATE_STR = "2020-12-31"
EUDR_CUTOFF_DATE_OBJ = date(2020, 12, 31)

ALL_COMMODITIES = [
    "cattle", "cocoa", "coffee", "palm_oil", "rubber", "soya", "wood",
]

# IUCN Protected Area Categories (I through VI)
IUCN_CATEGORIES = [
    "Ia",   # Strict Nature Reserve
    "Ib",   # Wilderness Area
    "II",   # National Park
    "III",  # Natural Monument
    "IV",   # Habitat/Species Management Area
    "V",    # Protected Landscape/Seascape
    "VI",   # Protected Area with Sustainable Use
]

# IUCN category base risk scores (higher for more strictly protected)
IUCN_CATEGORY_RISK_SCORES = {
    "Ia": Decimal("100"),
    "Ib": Decimal("95"),
    "II": Decimal("90"),
    "III": Decimal("80"),
    "IV": Decimal("70"),
    "V": Decimal("55"),
    "VI": Decimal("40"),
}

# Overlap type ordering (severity descending)
OVERLAP_TYPES = ["DIRECT", "PARTIAL", "BUFFER", "ADJACENT", "PROXIMATE", "NONE"]

# Overlap type base scores
OVERLAP_TYPE_SCORES = {
    "DIRECT": Decimal("100"),
    "PARTIAL": Decimal("80"),
    "BUFFER": Decimal("60"),
    "ADJACENT": Decimal("45"),
    "PROXIMATE": Decimal("25"),
    "NONE": Decimal("0"),
}

# Overlap type multipliers for risk scoring
OVERLAP_TYPE_MULTIPLIERS = {
    "DIRECT": Decimal("1.00"),
    "PARTIAL": Decimal("0.80"),
    "BUFFER": Decimal("0.60"),
    "ADJACENT": Decimal("0.45"),
    "PROXIMATE": Decimal("0.25"),
    "NONE": Decimal("0.00"),
}

# Risk levels descending
RISK_LEVELS = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]

# Severity levels
SEVERITY_LEVELS = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]

# Risk score thresholds
RISK_THRESHOLD_CRITICAL = Decimal("80")
RISK_THRESHOLD_HIGH = Decimal("60")
RISK_THRESHOLD_MEDIUM = Decimal("40")
RISK_THRESHOLD_LOW = Decimal("20")

# Buffer zone distances in kilometers
BUFFER_RING_DISTANCES = [1, 5, 10, 25, 50]

# Default risk scoring weights (5 factors, sum to 1.00)
DEFAULT_RISK_WEIGHTS = {
    "iucn_category": Decimal("0.30"),
    "overlap_type": Decimal("0.25"),
    "buffer_proximity": Decimal("0.20"),
    "deforestation_correlation": Decimal("0.15"),
    "certification_overlay": Decimal("0.10"),
}

# Designation statuses
DESIGNATION_STATUSES = [
    "designated", "proposed", "inscribed", "adopted",
    "established", "degazetted", "downgraded", "downsized",
]

# Governance types
GOVERNANCE_TYPES = [
    "government", "shared", "private", "indigenous_community",
]

# Protected area data sources
DATA_SOURCES = [
    "wdpa", "oecm", "national_registry", "raisg", "mapbiomas",
]

# Report types
ALL_REPORT_TYPES = [
    "protected_area_compliance",
    "dds_section",
    "overlap_summary",
    "buffer_analysis",
    "risk_assessment",
    "violation_report",
    "trend_report",
    "executive_summary",
]

# Report formats
ALL_REPORT_FORMATS = ["PDF", "JSON", "HTML", "CSV", "XLSX"]

# Report languages
ALL_REPORT_LANGUAGES = ["en", "fr", "de", "es", "pt"]

# Compliance outcome statuses
COMPLIANCE_OUTCOMES = [
    "compliant",
    "non_compliant",
    "conditional",
    "requires_review",
    "low_risk",
    "multi_jurisdiction",
]

# High-risk EUDR countries for protected areas
HIGH_RISK_COUNTRIES = [
    "BR", "ID", "CO", "MY", "PY", "CM", "CI", "GH", "NG", "CD",
    "CG", "PE", "BO", "VN", "TH", "MM",
]

# Low-risk EUDR countries
LOW_RISK_COUNTRIES = [
    "SE", "FI", "DE", "NO", "DK", "CH", "NZ", "AT",
]

# PADDD event types (Protection Downgrade/Downsize/Degazettement)
PADDD_EVENT_TYPES = [
    "downgrade", "downsize", "degazettement",
]

# Certification schemes
CERTIFICATION_SCHEMES = [
    "fsc", "pefc", "rspo", "rainforest_alliance", "iscc",
    "fairtrade", "organic", "none",
]

# Dedup window for violations (hours)
VIOLATION_DEDUP_WINDOW_HOURS = 72

# Maximum batch size
MAX_BATCH_SIZE = 10000


# ---------------------------------------------------------------------------
# Configuration Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_config():
    """Create a ProtectedAreaValidatorConfig dict with test defaults."""
    return {
        "database_url": "postgresql://localhost:5432/greenlang_test",
        "redis_url": "redis://localhost:6379/1",
        "log_level": "DEBUG",
        "pool_size": 2,
        # WDPA integration
        "wdpa_api_url": "https://api.protectedplanet.net/v3",
        "wdpa_api_key": "test-wdpa-key",
        "wdpa_cache_ttl_hours": 24,
        "wdpa_sync_interval_hours": 168,
        # Spatial analysis
        "srid": 4326,
        "default_buffer_km": Decimal("10"),
        "min_buffer_km": Decimal("1"),
        "max_buffer_km": Decimal("50"),
        "buffer_ring_distances": [1, 5, 10, 25, 50],
        "overlap_min_area_ha": Decimal("0.01"),
        "adjacency_threshold_km": Decimal("5"),
        "proximity_threshold_km": Decimal("25"),
        # Risk scoring
        "iucn_category_weight": Decimal("0.30"),
        "overlap_type_weight": Decimal("0.25"),
        "buffer_proximity_weight": Decimal("0.20"),
        "deforestation_correlation_weight": Decimal("0.15"),
        "certification_overlay_weight": Decimal("0.10"),
        "risk_threshold_critical": Decimal("80"),
        "risk_threshold_high": Decimal("60"),
        "risk_threshold_medium": Decimal("40"),
        "risk_threshold_low": Decimal("20"),
        # Violation detection
        "violation_dedup_window_hours": 72,
        "auto_escalation_enabled": True,
        "sla_triage_hours": 4,
        "sla_investigation_hours": 48,
        "sla_resolution_hours": 168,
        # Batch processing
        "batch_max_size": 10000,
        "batch_concurrency": 4,
        "batch_timeout_s": 300,
        # Provenance
        "enable_provenance": True,
        "genesis_hash": "GL-EUDR-PAV-022-TEST-GENESIS",
        "chain_algorithm": "sha256",
        # Metrics
        "enable_metrics": False,
        # Reporting
        "default_language": "en",
        "report_retention_days": 1825,
        "max_report_size_mb": 50,
    }


@pytest.fixture
def strict_config():
    """Create a ProtectedAreaValidatorConfig with strict thresholds."""
    return {
        "database_url": "postgresql://localhost:5432/greenlang_test",
        "redis_url": "redis://localhost:6379/1",
        "log_level": "DEBUG",
        "default_buffer_km": Decimal("5"),
        "overlap_min_area_ha": Decimal("0.001"),
        "adjacency_threshold_km": Decimal("2"),
        "proximity_threshold_km": Decimal("10"),
        "risk_threshold_critical": Decimal("75"),
        "risk_threshold_high": Decimal("55"),
        "risk_threshold_medium": Decimal("35"),
        "risk_threshold_low": Decimal("15"),
        "violation_dedup_window_hours": 24,
        "enable_provenance": True,
        "genesis_hash": "GL-EUDR-PAV-022-TEST-STRICT-GENESIS",
        "enable_metrics": False,
    }


@pytest.fixture
def uuid_gen():
    """Create a deterministic UUID generator."""
    return DeterministicUUID()


# ---------------------------------------------------------------------------
# Mock Fixtures (Provenance, Metrics, Database, Redis, Auth, WDPA)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_provenance():
    """Create a mock ProvenanceTracker for testing."""
    tracker = MagicMock()
    tracker.genesis_hash = compute_test_hash({"genesis": "GL-EUDR-PAV-022"})
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
    metrics.labels = MagicMock(return_value=metrics)
    return metrics


@pytest.fixture
def mock_db_pool():
    """Create a mock database connection pool with PostGIS support."""
    pool = MagicMock()
    conn = MagicMock()
    cursor = MagicMock()

    cursor.fetchone = MagicMock(return_value=None)
    cursor.fetchall = MagicMock(return_value=[])
    cursor.rowcount = 0
    cursor.__aenter__ = AsyncMock(return_value=cursor)
    cursor.__aexit__ = AsyncMock(return_value=False)

    conn.cursor = MagicMock(return_value=cursor)
    conn.execute = AsyncMock()
    conn.commit = AsyncMock()
    conn.__aenter__ = AsyncMock(return_value=conn)
    conn.__aexit__ = AsyncMock(return_value=False)

    pool.connection = MagicMock(return_value=conn)
    pool.getconn = AsyncMock(return_value=conn)
    pool.putconn = AsyncMock()
    return pool


@pytest.fixture
def mock_redis():
    """Create a mock Redis client."""
    redis = MagicMock()
    redis.get = AsyncMock(return_value=None)
    redis.set = AsyncMock(return_value=True)
    redis.delete = AsyncMock(return_value=1)
    redis.exists = AsyncMock(return_value=0)
    redis.expire = AsyncMock(return_value=True)
    redis.keys = AsyncMock(return_value=[])
    redis.pipeline = MagicMock(return_value=redis)
    redis.execute = AsyncMock(return_value=[])
    return redis


@pytest.fixture
def mock_auth():
    """Create mock authentication middleware."""
    auth = MagicMock()
    auth.validate_token = MagicMock(return_value={
        "sub": "test-user-001",
        "role": "eudr_analyst",
        "permissions": [
            "eudr-pav:protected-areas:read",
            "eudr-pav:protected-areas:write",
            "eudr-pav:overlaps:read",
            "eudr-pav:overlaps:detect",
            "eudr-pav:buffer-zones:read",
            "eudr-pav:designations:read",
            "eudr-pav:violations:read",
            "eudr-pav:violations:write",
            "eudr-pav:reports:read",
            "eudr-pav:reports:generate",
        ],
    })
    auth.require_permission = MagicMock(return_value=True)
    return auth


@pytest.fixture
def mock_wdpa_api():
    """Create a mock WDPA (World Database on Protected Areas) API client."""
    wdpa = MagicMock()
    wdpa.search_protected_areas = AsyncMock(return_value=[])
    wdpa.get_protected_area = AsyncMock(return_value=None)
    wdpa.get_protected_areas_in_bbox = AsyncMock(return_value=[])
    wdpa.get_protected_areas_by_country = AsyncMock(return_value=[])
    wdpa.sync_updates = AsyncMock(return_value={"added": 0, "updated": 0})
    wdpa.get_coverage_statistics = AsyncMock(return_value={})
    return wdpa


@pytest.fixture
def mock_postgis():
    """Create mock PostGIS spatial function responses."""
    postgis = MagicMock()
    postgis.st_intersects = MagicMock(return_value=False)
    postgis.st_intersection = MagicMock(return_value=None)
    postgis.st_area = MagicMock(return_value=Decimal("0"))
    postgis.st_buffer = MagicMock(return_value=None)
    postgis.st_dwithin = MagicMock(return_value=False)
    postgis.st_distance = MagicMock(return_value=Decimal("50000"))
    postgis.st_contains = MagicMock(return_value=False)
    postgis.st_within = MagicMock(return_value=False)
    postgis.st_centroid = MagicMock(return_value={"lat": 0.0, "lon": 0.0})
    postgis.st_makepoint = MagicMock(return_value="POINT(0 0)")
    postgis.st_setsrid = MagicMock(return_value="SRID=4326;POINT(0 0)")
    postgis.st_transform = MagicMock(return_value=None)
    postgis.st_geomfromgeojson = MagicMock(return_value=None)
    return postgis


# ---------------------------------------------------------------------------
# Sample Protected Area Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_protected_area():
    """Sample protected area: Amazonia National Park, Brazil, IUCN II."""
    return {
        "area_id": "pa-001",
        "wdpa_id": "350001",
        "name": "Amazonia National Park",
        "original_name": "Parque Nacional da Amazonia",
        "country_code": "BR",
        "iucn_category": "II",
        "designation": "National Park",
        "designation_type": "national",
        "governance_type": "government",
        "management_authority": "ICMBio",
        "status": "designated",
        "status_year": 1974,
        "area_hectares": Decimal("1089140"),
        "marine_area_hectares": Decimal("0"),
        "reported_area_hectares": Decimal("1089140"),
        "gis_area_hectares": Decimal("1085320"),
        "latitude": Decimal("-4.5"),
        "longitude": Decimal("-56.5"),
        "boundary_geojson": {
            "type": "Polygon",
            "coordinates": [[
                [-57.0, -5.0], [-57.0, -4.0],
                [-56.0, -4.0], [-56.0, -5.0],
                [-57.0, -5.0],
            ]],
        },
        "data_source": "wdpa",
        "last_updated": date(2025, 6, 1),
        "world_heritage": False,
        "ramsar_site": False,
        "biosphere_reserve": False,
        "provenance_hash": compute_test_hash({
            "area_id": "pa-001",
            "wdpa_id": "350001",
            "country_code": "BR",
        }),
    }


@pytest.fixture
def sample_protected_areas():
    """List of 6 sample protected areas across IUCN categories and countries."""
    specs = [
        ("pa-001", "350001", "Amazonia National Park", "BR", "II",
         Decimal("1089140"), Decimal("-4.5"), Decimal("-56.5"),
         "designated", "government", False),
        ("pa-002", "900100", "Virunga National Park", "CD", "II",
         Decimal("790000"), Decimal("0.5"), Decimal("29.5"),
         "designated", "government", True),
        ("pa-003", "450200", "Tanjung Puting National Park", "ID", "II",
         Decimal("415040"), Decimal("-2.8"), Decimal("111.9"),
         "designated", "government", False),
        ("pa-004", "555001", "Bialowieza Forest", "PL", "Ia",
         Decimal("10502"), Decimal("52.7"), Decimal("23.9"),
         "designated", "government", True),
        ("pa-005", "660001", "Tai National Park", "CI", "II",
         Decimal("454000"), Decimal("5.8"), Decimal("-7.1"),
         "designated", "government", True),
        ("pa-006", "770001", "Monte Verde Cloud Forest", "CR", "VI",
         Decimal("10500"), Decimal("10.3"), Decimal("-84.8"),
         "designated", "private", False),
    ]
    areas = []
    for (aid, wid, name, country, iucn, area_ha, lat, lon,
         status, governance, whs) in specs:
        areas.append({
            "area_id": aid,
            "wdpa_id": wid,
            "name": name,
            "country_code": country,
            "iucn_category": iucn,
            "designation": "National Park" if iucn == "II" else "Reserve",
            "governance_type": governance,
            "status": status,
            "area_hectares": area_ha,
            "latitude": lat,
            "longitude": lon,
            "world_heritage": whs,
            "data_source": "wdpa",
            "provenance_hash": compute_test_hash({
                "area_id": aid,
                "wdpa_id": wid,
                "country_code": country,
            }),
        })
    return areas


@pytest.fixture
def sample_iucn_ia_area():
    """Sample IUCN Ia Strict Nature Reserve."""
    return {
        "area_id": "pa-ia-001",
        "wdpa_id": "800001",
        "name": "Strict Reserve Zazamarotra",
        "country_code": "MG",
        "iucn_category": "Ia",
        "status": "designated",
        "area_hectares": Decimal("5600"),
        "latitude": Decimal("-15.4"),
        "longitude": Decimal("47.5"),
        "governance_type": "government",
        "world_heritage": False,
        "provenance_hash": compute_test_hash({
            "area_id": "pa-ia-001",
            "iucn_category": "Ia",
        }),
    }


@pytest.fixture
def sample_iucn_vi_area():
    """Sample IUCN VI Protected Area with Sustainable Use."""
    return {
        "area_id": "pa-vi-001",
        "wdpa_id": "800010",
        "name": "Resex Chico Mendes",
        "country_code": "BR",
        "iucn_category": "VI",
        "status": "designated",
        "area_hectares": Decimal("931537"),
        "latitude": Decimal("-10.1"),
        "longitude": Decimal("-69.1"),
        "governance_type": "shared",
        "world_heritage": False,
        "provenance_hash": compute_test_hash({
            "area_id": "pa-vi-001",
            "iucn_category": "VI",
        }),
    }


@pytest.fixture
def sample_world_heritage_site():
    """Sample UNESCO World Heritage Site."""
    return {
        "area_id": "pa-whs-001",
        "wdpa_id": "900100",
        "name": "Virunga National Park",
        "country_code": "CD",
        "iucn_category": "II",
        "status": "inscribed",
        "status_year": 1979,
        "area_hectares": Decimal("790000"),
        "latitude": Decimal("0.5"),
        "longitude": Decimal("29.5"),
        "world_heritage": True,
        "governance_type": "government",
        "provenance_hash": compute_test_hash({
            "area_id": "pa-whs-001",
            "world_heritage": True,
        }),
    }


# ---------------------------------------------------------------------------
# Sample Plot Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_plot():
    """Sample production plot: Brazil, soya, 150 hectares."""
    return {
        "plot_id": "plot-001",
        "latitude": Decimal("-4.5"),
        "longitude": Decimal("-56.5"),
        "area_hectares": Decimal("150"),
        "country_code": "BR",
        "commodity": "soya",
        "supplier_id": "sup-001",
        "boundary_geojson": {
            "type": "Polygon",
            "coordinates": [[
                [-56.55, -4.55],
                [-56.55, -4.45],
                [-56.45, -4.45],
                [-56.45, -4.55],
                [-56.55, -4.55],
            ]],
        },
        "provenance_hash": compute_test_hash({
            "plot_id": "plot-001",
            "country_code": "BR",
        }),
    }


@pytest.fixture
def sample_plots():
    """List of 5 sample production plots in different countries."""
    specs = [
        ("plot-001", Decimal("-4.5"), Decimal("-56.5"), "BR", "soya",
         Decimal("150")),
        ("plot-002", Decimal("-2.8"), Decimal("111.9"), "ID", "palm_oil",
         Decimal("200")),
        ("plot-003", Decimal("5.8"), Decimal("-7.1"), "CI", "cocoa",
         Decimal("50")),
        ("plot-004", Decimal("0.5"), Decimal("29.5"), "CD", "wood",
         Decimal("300")),
        ("plot-005", Decimal("-10.1"), Decimal("-69.1"), "BR", "cattle",
         Decimal("500")),
    ]
    plots = []
    for pid, lat, lon, country, commodity, area in specs:
        plots.append({
            "plot_id": pid,
            "latitude": lat,
            "longitude": lon,
            "country_code": country,
            "commodity": commodity,
            "area_hectares": area,
            "boundary_geojson": {
                "type": "Polygon",
                "coordinates": [[
                    [float(lon) - 0.05, float(lat) - 0.05],
                    [float(lon) - 0.05, float(lat) + 0.05],
                    [float(lon) + 0.05, float(lat) + 0.05],
                    [float(lon) + 0.05, float(lat) - 0.05],
                    [float(lon) - 0.05, float(lat) - 0.05],
                ]],
            },
            "provenance_hash": compute_test_hash({
                "plot_id": pid,
                "country_code": country,
            }),
        })
    return plots


# ---------------------------------------------------------------------------
# Sample Overlap Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_overlap_direct():
    """Overlap: DIRECT overlap, plot fully inside protected area."""
    return {
        "overlap_id": "ov-001",
        "plot_id": "plot-001",
        "area_id": "pa-001",
        "overlap_type": "DIRECT",
        "overlap_area_hectares": Decimal("150"),
        "overlap_pct_of_plot": Decimal("100.0"),
        "overlap_pct_of_area": Decimal("0.014"),
        "distance_meters": Decimal("0"),
        "iucn_category": "II",
        "risk_score": Decimal("92.50"),
        "risk_level": "CRITICAL",
        "provenance_hash": compute_test_hash({
            "overlap_id": "ov-001",
            "overlap_type": "DIRECT",
        }),
    }


@pytest.fixture
def sample_overlap_partial():
    """Overlap: PARTIAL overlap, plot partially inside protected area."""
    return {
        "overlap_id": "ov-002",
        "plot_id": "plot-002",
        "area_id": "pa-003",
        "overlap_type": "PARTIAL",
        "overlap_area_hectares": Decimal("85"),
        "overlap_pct_of_plot": Decimal("42.5"),
        "overlap_pct_of_area": Decimal("0.020"),
        "distance_meters": Decimal("0"),
        "iucn_category": "II",
        "risk_score": Decimal("72.00"),
        "risk_level": "HIGH",
        "provenance_hash": compute_test_hash({
            "overlap_id": "ov-002",
            "overlap_type": "PARTIAL",
        }),
    }


@pytest.fixture
def sample_overlap_buffer():
    """Overlap: BUFFER overlap, plot within buffer zone of protected area."""
    return {
        "overlap_id": "ov-003",
        "plot_id": "plot-003",
        "area_id": "pa-005",
        "overlap_type": "BUFFER",
        "overlap_area_hectares": Decimal("0"),
        "overlap_pct_of_plot": Decimal("0"),
        "distance_meters": Decimal("3500"),
        "iucn_category": "II",
        "buffer_ring_km": 5,
        "risk_score": Decimal("55.00"),
        "risk_level": "MEDIUM",
        "provenance_hash": compute_test_hash({
            "overlap_id": "ov-003",
            "overlap_type": "BUFFER",
        }),
    }


@pytest.fixture
def sample_overlap_adjacent():
    """Overlap: ADJACENT, plot near protected area boundary."""
    return {
        "overlap_id": "ov-004",
        "plot_id": "plot-004",
        "area_id": "pa-002",
        "overlap_type": "ADJACENT",
        "distance_meters": Decimal("4200"),
        "iucn_category": "II",
        "risk_score": Decimal("42.00"),
        "risk_level": "MEDIUM",
        "provenance_hash": compute_test_hash({
            "overlap_id": "ov-004",
            "overlap_type": "ADJACENT",
        }),
    }


@pytest.fixture
def sample_overlap_proximate():
    """Overlap: PROXIMATE, plot within proximity threshold."""
    return {
        "overlap_id": "ov-005",
        "plot_id": "plot-005",
        "area_id": "pa-006",
        "overlap_type": "PROXIMATE",
        "distance_meters": Decimal("18000"),
        "iucn_category": "VI",
        "risk_score": Decimal("22.00"),
        "risk_level": "LOW",
        "provenance_hash": compute_test_hash({
            "overlap_id": "ov-005",
            "overlap_type": "PROXIMATE",
        }),
    }


@pytest.fixture
def sample_overlap_none():
    """Overlap: NONE, no protected area nearby."""
    return {
        "overlap_id": "ov-006",
        "plot_id": "plot-006",
        "area_id": None,
        "overlap_type": "NONE",
        "distance_meters": Decimal("100000"),
        "risk_score": Decimal("0"),
        "risk_level": "INFO",
        "provenance_hash": compute_test_hash({
            "overlap_id": "ov-006",
            "overlap_type": "NONE",
        }),
    }


# ---------------------------------------------------------------------------
# Sample Buffer Zone Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_buffer_zones():
    """List of multi-ring buffer zones around a protected area."""
    buffers = []
    for radius_km in BUFFER_RING_DISTANCES:
        buffers.append({
            "buffer_id": f"buf-{radius_km:02d}km",
            "area_id": "pa-001",
            "radius_km": radius_km,
            "buffer_type": "circular",
            "area_hectares": Decimal(str(
                math.pi * (radius_km * 100) ** 2 / 10000
            )).quantize(Decimal("0.01")),
            "encroachment_count": 0,
            "active": True,
            "provenance_hash": compute_test_hash({
                "buffer_id": f"buf-{radius_km:02d}km",
                "area_id": "pa-001",
            }),
        })
    return buffers


@pytest.fixture
def sample_buffer_violation():
    """Sample buffer zone encroachment violation."""
    return {
        "violation_id": "bv-001",
        "buffer_id": "buf-05km",
        "area_id": "pa-001",
        "plot_id": "plot-001",
        "distance_to_boundary_meters": Decimal("3200"),
        "buffer_ring_km": 5,
        "encroachment_area_hectares": Decimal("12.5"),
        "detected_at": datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
        "severity": "MEDIUM",
        "provenance_hash": compute_test_hash({
            "violation_id": "bv-001",
            "buffer_ring_km": 5,
        }),
    }


# ---------------------------------------------------------------------------
# Sample Designation Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_designation():
    """Sample protected area designation record."""
    return {
        "designation_id": "des-001",
        "area_id": "pa-001",
        "designation_type": "National Park",
        "iucn_category": "II",
        "designated_date": date(1974, 2, 19),
        "legal_instrument": "Federal Decree No. 73.683",
        "governing_body": "ICMBio",
        "governance_type": "government",
        "management_plan_exists": True,
        "management_plan_year": 2010,
        "management_effectiveness_score": Decimal("72.5"),
        "is_active": True,
        "paddd_events": [],
        "provenance_hash": compute_test_hash({
            "designation_id": "des-001",
            "area_id": "pa-001",
        }),
    }


@pytest.fixture
def sample_paddd_event():
    """Sample PADDD (Protection Downgrade/Downsize/Degazettement) event."""
    return {
        "paddd_id": "paddd-001",
        "area_id": "pa-007",
        "event_type": "downsize",
        "event_date": date(2022, 6, 15),
        "area_affected_hectares": Decimal("15000"),
        "area_remaining_hectares": Decimal("85000"),
        "legal_instrument": "State Decree 2022-145",
        "reason": "Infrastructure development corridor",
        "reversible": True,
        "provenance_hash": compute_test_hash({
            "paddd_id": "paddd-001",
            "event_type": "downsize",
        }),
    }


# ---------------------------------------------------------------------------
# Sample Violation Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_violation():
    """Sample encroachment violation: CRITICAL severity."""
    return {
        "violation_id": "viol-001",
        "area_id": "pa-001",
        "plot_id": "plot-001",
        "violation_type": "encroachment",
        "severity": "CRITICAL",
        "status": "pending",
        "title": "Direct encroachment into Amazonia National Park",
        "description": "Production plot fully within IUCN II protected area",
        "overlap_area_hectares": Decimal("150"),
        "distance_to_boundary_meters": Decimal("0"),
        "iucn_category": "II",
        "country_code": "BR",
        "detected_at": datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
        "affected_commodities": ["soya"],
        "affected_suppliers": ["sup-001"],
        "supply_chain_correlation": True,
        "provenance_hash": compute_test_hash({
            "violation_id": "viol-001",
            "violation_type": "encroachment",
        }),
    }


@pytest.fixture
def sample_violations():
    """List of 5 violations at different severity levels."""
    specs = [
        ("viol-001", "encroachment", "BR", "CRITICAL", "II",
         Decimal("150"), Decimal("0")),
        ("viol-002", "encroachment", "ID", "HIGH", "II",
         Decimal("85"), Decimal("0")),
        ("viol-003", "buffer_intrusion", "CI", "MEDIUM", "II",
         Decimal("0"), Decimal("3500")),
        ("viol-004", "proximity_alert", "CD", "LOW", "II",
         Decimal("0"), Decimal("18000")),
        ("viol-005", "monitoring_flag", "CR", "INFO", "VI",
         Decimal("0"), Decimal("45000")),
    ]
    violations = []
    for vid, vtype, country, severity, iucn, area, dist in specs:
        violations.append({
            "violation_id": vid,
            "violation_type": vtype,
            "country_code": country,
            "severity": severity,
            "iucn_category": iucn,
            "overlap_area_hectares": area,
            "distance_to_boundary_meters": dist,
            "status": "pending",
            "provenance_hash": compute_test_hash({
                "violation_id": vid,
                "severity": severity,
            }),
        })
    return violations


# ---------------------------------------------------------------------------
# Sample Report Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_report():
    """Sample compliance report: JSON format, EN language."""
    return {
        "report_id": "rpt-001",
        "report_type": "protected_area_compliance",
        "title": "Protected Area Compliance Report Q1 2026",
        "format": "JSON",
        "language": "en",
        "scope_type": "operator",
        "scope_ids": ["op-001"],
        "generated_at": datetime(2026, 3, 1, 0, 0, 0, tzinfo=timezone.utc),
        "provenance_hash": compute_test_hash({
            "report_id": "rpt-001",
            "report_type": "protected_area_compliance",
        }),
    }


# ---------------------------------------------------------------------------
# Sample Coordinates and Locations
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_coordinates():
    """Dictionary of test locations with (lat, lon) tuples."""
    return {
        "brazil_amazonia_np": (Decimal("-4.5"), Decimal("-56.5")),
        "brazil_mato_grosso": (Decimal("-12.5"), Decimal("-55.3")),
        "indonesia_tanjung_puting": (Decimal("-2.8"), Decimal("111.9")),
        "drc_virunga": (Decimal("0.5"), Decimal("29.5")),
        "cote_ivoire_tai": (Decimal("5.8"), Decimal("-7.1")),
        "costa_rica_monteverde": (Decimal("10.3"), Decimal("-84.8")),
        "denmark_copenhagen": (Decimal("55.7"), Decimal("12.6")),
        "north_pole": (Decimal("90.0"), Decimal("0.0")),
        "south_pole": (Decimal("-90.0"), Decimal("0.0")),
        "antimeridian": (Decimal("0.0"), Decimal("180.0")),
    }


# ---------------------------------------------------------------------------
# Risk Scoring Computation Helpers
# ---------------------------------------------------------------------------


def compute_risk_score(
    iucn_category: str,
    overlap_type: str,
    buffer_proximity_score: Decimal,
    deforestation_correlation_score: Decimal,
    certification_overlay_score: Decimal,
    weights: Optional[Dict[str, Decimal]] = None,
) -> Decimal:
    """Compute 5-factor deterministic risk score using Decimal arithmetic.

    Args:
        iucn_category: IUCN category string (Ia, Ib, II-VI).
        overlap_type: Overlap type (DIRECT, PARTIAL, BUFFER, ADJACENT, PROXIMATE, NONE).
        buffer_proximity_score: Buffer proximity factor score [0-100].
        deforestation_correlation_score: Deforestation correlation factor [0-100].
        certification_overlay_score: Certification scheme reduction factor [0-100].
        weights: Optional weight overrides.

    Returns:
        Weighted composite risk score as Decimal [0-100].
    """
    w = weights or DEFAULT_RISK_WEIGHTS
    iucn_score = IUCN_CATEGORY_RISK_SCORES.get(iucn_category, Decimal("50"))
    ot_score = OVERLAP_TYPE_SCORES.get(overlap_type, Decimal("0"))

    total = (
        iucn_score * w["iucn_category"]
        + ot_score * w["overlap_type"]
        + buffer_proximity_score * w["buffer_proximity"]
        + deforestation_correlation_score * w["deforestation_correlation"]
        + certification_overlay_score * w["certification_overlay"]
    )
    return total.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def classify_risk_level(score: Decimal) -> str:
    """Classify risk level from composite score.

    Thresholds (default):
        >= 80 -> CRITICAL
        >= 60 -> HIGH
        >= 40 -> MEDIUM
        >= 20 -> LOW
        < 20  -> INFO
    """
    if score >= RISK_THRESHOLD_CRITICAL:
        return "CRITICAL"
    elif score >= RISK_THRESHOLD_HIGH:
        return "HIGH"
    elif score >= RISK_THRESHOLD_MEDIUM:
        return "MEDIUM"
    elif score >= RISK_THRESHOLD_LOW:
        return "LOW"
    else:
        return "INFO"


def compute_buffer_proximity_score(
    distance_meters: Decimal,
    buffer_rings: Optional[List[int]] = None,
) -> Decimal:
    """Compute buffer proximity score based on distance to protected area.

    Closer distance = higher score.
    0 m    -> 100
    < 1 km -> 90
    < 5 km -> 75
    < 10 km -> 60
    < 25 km -> 40
    < 50 km -> 20
    >= 50 km -> 0
    """
    d_km = distance_meters / Decimal("1000")
    if d_km <= Decimal("0"):
        return Decimal("100")
    elif d_km < Decimal("1"):
        return Decimal("90")
    elif d_km < Decimal("5"):
        return Decimal("75")
    elif d_km < Decimal("10"):
        return Decimal("60")
    elif d_km < Decimal("25"):
        return Decimal("40")
    elif d_km < Decimal("50"):
        return Decimal("20")
    else:
        return Decimal("0")


def classify_severity(
    overlap_type: str,
    iucn_category: str,
    world_heritage: bool = False,
) -> str:
    """Classify violation severity from overlap type and IUCN category.

    CRITICAL: Direct/Partial overlap with Ia/Ib/II or any World Heritage
    HIGH: Direct/Partial with III/IV or Buffer with Ia/Ib/II
    MEDIUM: Buffer/Adjacent with III-VI
    LOW: Proximate with any category
    INFO: No overlap
    """
    if overlap_type == "NONE":
        return "INFO"

    strict_categories = {"Ia", "Ib", "II"}

    if world_heritage and overlap_type in ("DIRECT", "PARTIAL"):
        return "CRITICAL"

    if overlap_type in ("DIRECT", "PARTIAL"):
        if iucn_category in strict_categories:
            return "CRITICAL"
        elif iucn_category in ("III", "IV"):
            return "HIGH"
        else:
            return "MEDIUM"

    if overlap_type == "BUFFER":
        if iucn_category in strict_categories:
            return "HIGH"
        else:
            return "MEDIUM"

    if overlap_type == "ADJACENT":
        return "MEDIUM" if iucn_category in strict_categories else "LOW"

    if overlap_type == "PROXIMATE":
        return "LOW"

    return "INFO"


def compute_compliance_status(
    risk_level: str,
    certification_present: bool = False,
    iucn_vi_managed_use: bool = False,
) -> str:
    """Determine compliance status from risk level and mitigating factors.

    CRITICAL -> non_compliant
    HIGH -> non_compliant (unless IUCN VI managed use -> conditional)
    MEDIUM -> conditional (if certified) or requires_review
    LOW -> low_risk
    INFO -> compliant
    """
    if risk_level == "CRITICAL":
        return "non_compliant"
    elif risk_level == "HIGH":
        if iucn_vi_managed_use:
            return "conditional"
        return "non_compliant"
    elif risk_level == "MEDIUM":
        if certification_present:
            return "conditional"
        return "requires_review"
    elif risk_level == "LOW":
        return "low_risk"
    else:
        return "compliant"
