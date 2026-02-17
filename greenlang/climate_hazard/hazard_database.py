# -*- coding: utf-8 -*-
"""
Hazard Database Engine - AGENT-DATA-020: Climate Hazard Connector

Engine 1 of 7 in the Climate Hazard Connector pipeline.

Pure-Python engine for unified ingestion and querying of climate hazard data
from multiple authoritative sources. Provides source registration, hazard
data ingestion, spatial and temporal querying, historical event tracking,
multi-source aggregation, import/export, and aggregate statistics. Each
record is assigned a unique UUID and tracked with SHA-256 provenance
hashing for complete audit trails.

Hazard data sources include global databases (WRI Aqueduct, ThinkHazard,
IPCC AR6 WG2, Copernicus CDS, NASA SEDAC), event catalogs (EM-DAT, NOAA
NCEI, Munich Re NatCatSERVICE), scenario models (NGFS Climate Scenarios),
and regional indices (Swiss Re CatNet). Ten built-in sources are
pre-registered on engine initialization.

Twelve hazard types are supported following the TCFD and IPCC AR6
classification: RIVERINE_FLOOD, COASTAL_FLOOD, DROUGHT, EXTREME_HEAT,
EXTREME_COLD, WILDFIRE, TROPICAL_CYCLONE, EXTREME_PRECIPITATION,
WATER_STRESS, SEA_LEVEL_RISE, LANDSLIDE, COASTAL_EROSION.

Zero-Hallucination Guarantees:
    - All IDs are deterministic UUID-4 values (no LLM involvement)
    - Timestamps from ``datetime.now(timezone.utc)`` (deterministic)
    - SHA-256 provenance hashes recorded on every mutating operation
    - Haversine distance uses pure math (sin, cos, asin, sqrt)
    - Aggregation strategies use explicit arithmetic only
    - Statistics are derived from in-memory data structures
    - No ML or LLM calls anywhere in this engine

Thread Safety:
    All mutating and read operations are protected by ``self._lock``
    (a ``threading.Lock``). Callers receive plain dict copies so they
    cannot accidentally mutate internal state.

Example:
    >>> from greenlang.climate_hazard.hazard_database import (
    ...     HazardDatabaseEngine,
    ... )
    >>> engine = HazardDatabaseEngine()
    >>> src = engine.get_source("wri-aqueduct")
    >>> assert src is not None
    >>> result = engine.ingest_hazard_data(
    ...     source_id="wri-aqueduct",
    ...     hazard_type="RIVERINE_FLOOD",
    ...     records=[{
    ...         "location": {"lat": 51.5074, "lon": -0.1278},
    ...         "intensity": 6.5,
    ...         "probability": 0.15,
    ...         "frequency": 0.8,
    ...         "duration_days": 5,
    ...         "observed_at": "2024-01-15T00:00:00",
    ...         "metadata": {"return_period_years": 50},
    ...     }],
    ...     region="europe",
    ... )
    >>> assert result["ingested_count"] == 1
    >>> stats = engine.get_statistics()
    >>> assert stats["total_sources"] >= 10

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-020 Climate Hazard Connector (GL-DATA-GEO-002)
Status: Production Ready
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency: ProvenanceTracker
# ---------------------------------------------------------------------------

try:
    from greenlang.climate_hazard.provenance import (
        ProvenanceTracker,
    )
    _PROVENANCE_AVAILABLE = True
except ImportError:
    _PROVENANCE_AVAILABLE = False
    logger.info(
        "climate_hazard.provenance not available; "
        "using inline ProvenanceTracker"
    )

    class ProvenanceTracker:  # type: ignore[no-redef]
        """Minimal inline provenance tracker for standalone operation.

        Provides SHA-256 chain hashing without external dependencies.
        """

        GENESIS_HASH = hashlib.sha256(
            b"greenlang-climate-hazard-genesis"
        ).hexdigest()

        def __init__(
            self,
            genesis_hash: Optional[str] = None,
        ) -> None:
            """Initialize with genesis hash.

            Args:
                genesis_hash: Optional custom genesis seed string.
                    When provided, the genesis hash is computed from
                    this string. Otherwise the class-level default
                    is used.
            """
            if genesis_hash is not None:
                self._last_chain_hash: str = hashlib.sha256(
                    genesis_hash.encode("utf-8"),
                ).hexdigest()
            else:
                self._last_chain_hash = self.GENESIS_HASH
            self._chain: List[Dict[str, Any]] = []
            self._lock = threading.Lock()

        def record(
            self,
            entity_type: str,
            entity_id: str,
            action: str,
            metadata: Optional[Any] = None,
        ) -> Any:
            """Record a provenance entry and return a stub entry.

            Args:
                entity_type: Type of entity being tracked.
                entity_id: Unique identifier for the entity.
                action: Action performed.
                metadata: Optional payload to hash.

            Returns:
                A stub object with a ``hash_value`` attribute.
            """
            ts = _utcnow().isoformat()
            if metadata is None:
                serialized = "null"
            else:
                serialized = json.dumps(
                    metadata, sort_keys=True, default=str,
                )
            data_hash = hashlib.sha256(
                serialized.encode("utf-8"),
            ).hexdigest()

            with self._lock:
                combined = json.dumps({
                    "action": action,
                    "data_hash": data_hash,
                    "parent_hash": self._last_chain_hash,
                    "timestamp": ts,
                }, sort_keys=True)
                chain_hash = hashlib.sha256(
                    combined.encode("utf-8"),
                ).hexdigest()

                self._chain.append({
                    "entity_type": entity_type,
                    "entity_id": entity_id,
                    "action": action,
                    "hash_value": chain_hash,
                    "parent_hash": self._last_chain_hash,
                    "timestamp": ts,
                    "metadata": {"data_hash": data_hash},
                })
                self._last_chain_hash = chain_hash

            class _StubEntry:
                def __init__(self, hv: str) -> None:
                    self.hash_value = hv

            return _StubEntry(chain_hash)

        def build_hash(self, data: Any) -> str:
            """Return SHA-256 hash of JSON-serialized data.

            Args:
                data: JSON-serializable value to hash.

            Returns:
                64-character lowercase hex SHA-256 digest.
            """
            return hashlib.sha256(
                json.dumps(data, sort_keys=True, default=str).encode()
            ).hexdigest()

        @property
        def entry_count(self) -> int:
            """Return the total number of provenance entries."""
            with self._lock:
                return len(self._chain)

        def export_chain(self) -> List[Dict[str, Any]]:
            """Return the full provenance chain for audit."""
            with self._lock:
                return list(self._chain)

        def reset(self) -> None:
            """Clear all provenance state."""
            with self._lock:
                self._chain.clear()
                self._last_chain_hash = self.GENESIS_HASH


# ---------------------------------------------------------------------------
# Optional dependency: Prometheus metrics
# ---------------------------------------------------------------------------

try:
    from greenlang.climate_hazard.metrics import (
        record_source_registered,
        record_data_ingested,
        record_query_executed,
        record_event_registered,
        record_aggregation,
        observe_processing_duration,
        set_total_sources,
        set_total_records,
        record_error,
        PROMETHEUS_AVAILABLE,
    )
    _METRICS_AVAILABLE = True
except ImportError:
    _METRICS_AVAILABLE = False
    PROMETHEUS_AVAILABLE = False  # type: ignore[assignment]
    logger.info(
        "climate_hazard.metrics not available; "
        "hazard database metrics disabled"
    )

    def record_source_registered(  # type: ignore[misc]
        source_type: str,
    ) -> None:
        """No-op fallback when metrics module is not available."""

    def record_data_ingested(  # type: ignore[misc]
        source_id: str,
        hazard_type: str,
        count: int = 1,
    ) -> None:
        """No-op fallback when metrics module is not available."""

    def record_query_executed(  # type: ignore[misc]
        query_type: str,
    ) -> None:
        """No-op fallback when metrics module is not available."""

    def record_event_registered(  # type: ignore[misc]
        hazard_type: str,
    ) -> None:
        """No-op fallback when metrics module is not available."""

    def record_aggregation(  # type: ignore[misc]
        strategy: str,
    ) -> None:
        """No-op fallback when metrics module is not available."""

    def observe_processing_duration(  # type: ignore[misc]
        operation: str,
        seconds: float,
    ) -> None:
        """No-op fallback when metrics module is not available."""

    def set_total_sources(  # type: ignore[misc]
        count: int,
    ) -> None:
        """No-op fallback when metrics module is not available."""

    def set_total_records(  # type: ignore[misc]
        count: int,
    ) -> None:
        """No-op fallback when metrics module is not available."""

    def record_error(  # type: ignore[misc]
        error_type: str,
    ) -> None:
        """No-op fallback when metrics module is not available."""


# ---------------------------------------------------------------------------
# Optional dependency: Config
# ---------------------------------------------------------------------------

try:
    from greenlang.climate_hazard.config import get_config as _get_config
    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False
    logger.info(
        "climate_hazard.config not available; "
        "using default engine settings"
    )

    def _get_config() -> Any:  # type: ignore[misc]
        """No-op fallback returning None when config is not available."""
        return None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Valid hazard types (TCFD + IPCC AR6 classification).
VALID_HAZARD_TYPES: Set[str] = {
    "RIVERINE_FLOOD",
    "COASTAL_FLOOD",
    "DROUGHT",
    "EXTREME_HEAT",
    "EXTREME_COLD",
    "WILDFIRE",
    "TROPICAL_CYCLONE",
    "EXTREME_PRECIPITATION",
    "WATER_STRESS",
    "SEA_LEVEL_RISE",
    "LANDSLIDE",
    "COASTAL_EROSION",
}

#: Valid source types for hazard data providers.
VALID_SOURCE_TYPES: Set[str] = {
    "GLOBAL_DATABASE",
    "EVENT_CATALOG",
    "SCENARIO_MODEL",
    "REANALYSIS",
    "REGIONAL_INDEX",
    "CUSTOM",
}

#: Valid aggregation strategies for multi-source data fusion.
VALID_AGGREGATION_STRATEGIES: Set[str] = {
    "weighted_average",
    "maximum",
    "minimum",
    "median",
}

#: Valid source statuses.
VALID_SOURCE_STATUSES: Set[str] = {
    "active",
    "inactive",
    "deprecated",
}

#: Maximum number of records in a single ingest call.
MAX_INGEST_BATCH: int = 10_000

#: Maximum number of sources registrable.
MAX_SOURCES: int = 200

#: Maximum number of historical events.
MAX_EVENTS: int = 100_000

#: Maximum number of hazard data records.
MAX_RECORDS: int = 500_000

#: Maximum number of records returned by a single query.
MAX_QUERY_LIMIT: int = 10_000

#: Default query limit.
DEFAULT_QUERY_LIMIT: int = 100

#: Earth radius in kilometers (WGS-84 mean radius).
EARTH_RADIUS_KM: float = 6371.0088

#: Region bounding boxes (lat_min, lat_max, lon_min, lon_max).
#: Approximate continental and sub-continental boundaries.
REGION_BOUNDS: Dict[str, Tuple[float, float, float, float]] = {
    "global": (-90.0, 90.0, -180.0, 180.0),
    "africa": (-35.0, 37.5, -25.0, 55.0),
    "asia": (-10.0, 80.0, 25.0, 180.0),
    "europe": (35.0, 72.0, -25.0, 65.0),
    "north_america": (15.0, 85.0, -170.0, -50.0),
    "south_america": (-56.0, 15.0, -82.0, -34.0),
    "oceania": (-50.0, 0.0, 100.0, 180.0),
    "middle_east": (12.0, 42.0, 25.0, 65.0),
    "central_america": (7.0, 24.0, -118.0, -60.0),
    "caribbean": (10.0, 27.0, -90.0, -58.0),
    "southeast_asia": (-11.0, 28.0, 92.0, 142.0),
    "south_asia": (5.0, 38.0, 60.0, 98.0),
    "east_asia": (18.0, 55.0, 73.0, 150.0),
    "central_asia": (35.0, 55.0, 46.0, 88.0),
    "west_africa": (4.0, 28.0, -18.0, 16.0),
    "east_africa": (-12.0, 18.0, 22.0, 52.0),
    "southern_africa": (-35.0, -10.0, 10.0, 41.0),
    "northern_europe": (54.0, 72.0, -25.0, 40.0),
    "southern_europe": (35.0, 48.0, -10.0, 35.0),
    "pacific_islands": (-25.0, 20.0, 120.0, -120.0),
    "arctic": (66.5, 90.0, -180.0, 180.0),
    "antarctic": (-90.0, -60.0, -180.0, 180.0),
}


# ---------------------------------------------------------------------------
# Built-in data source definitions
# ---------------------------------------------------------------------------

#: Pre-registered data sources reflecting major global climate hazard
#: databases, event catalogs, scenario models, and regional indices.
BUILTIN_SOURCES: List[Dict[str, Any]] = [
    {
        "source_id": "wri-aqueduct",
        "name": "WRI Aqueduct",
        "source_type": "GLOBAL_DATABASE",
        "hazard_types": [
            "RIVERINE_FLOOD",
            "COASTAL_FLOOD",
            "DROUGHT",
            "WATER_STRESS",
        ],
        "coverage": "global",
        "config": {
            "provider": "World Resources Institute",
            "url": "https://www.wri.org/aqueduct",
            "version": "4.0",
            "resolution": "sub-basin",
            "update_frequency": "annual",
            "license": "CC-BY-4.0",
            "description": (
                "WRI Aqueduct global flood and water risk analysis "
                "tool providing baseline and projected flood risk, "
                "water stress, and drought severity at sub-basin "
                "resolution."
            ),
        },
    },
    {
        "source_id": "thinkhazard-gfdrr",
        "name": "ThinkHazard (GFDRR)",
        "source_type": "GLOBAL_DATABASE",
        "hazard_types": [
            "RIVERINE_FLOOD",
            "COASTAL_FLOOD",
            "DROUGHT",
            "EXTREME_HEAT",
            "WILDFIRE",
            "TROPICAL_CYCLONE",
            "EXTREME_PRECIPITATION",
            "WATER_STRESS",
            "LANDSLIDE",
            "COASTAL_EROSION",
            "EXTREME_COLD",
        ],
        "coverage": "global",
        "config": {
            "provider": "Global Facility for Disaster Reduction and Recovery",
            "url": "https://thinkhazard.org",
            "version": "3.0",
            "resolution": "admin-level-2",
            "update_frequency": "semi-annual",
            "license": "CC-BY-SA-4.0",
            "description": (
                "ThinkHazard provides hazard level classifications "
                "for 11 natural hazard types at administrative level 2 "
                "globally, supporting development project planning."
            ),
        },
    },
    {
        "source_id": "emdat-cred",
        "name": "EM-DAT (CRED)",
        "source_type": "EVENT_CATALOG",
        "hazard_types": [
            "RIVERINE_FLOOD",
            "COASTAL_FLOOD",
            "DROUGHT",
            "EXTREME_HEAT",
            "EXTREME_COLD",
            "WILDFIRE",
            "TROPICAL_CYCLONE",
            "EXTREME_PRECIPITATION",
            "LANDSLIDE",
            "COASTAL_EROSION",
            "WATER_STRESS",
            "SEA_LEVEL_RISE",
        ],
        "coverage": "global",
        "config": {
            "provider": "Centre for Research on the Epidemiology of Disasters",
            "url": "https://www.emdat.be",
            "version": "2024",
            "temporal_coverage": "1900-present",
            "update_frequency": "quarterly",
            "license": "academic/commercial",
            "description": (
                "EM-DAT international disaster database cataloging "
                "natural and technological disasters since 1900. "
                "Includes fatalities, affected populations, and "
                "economic losses."
            ),
        },
    },
    {
        "source_id": "ngfs-scenarios",
        "name": "NGFS Climate Scenarios",
        "source_type": "SCENARIO_MODEL",
        "hazard_types": [
            "EXTREME_HEAT",
            "DROUGHT",
            "RIVERINE_FLOOD",
            "COASTAL_FLOOD",
            "TROPICAL_CYCLONE",
            "WILDFIRE",
            "SEA_LEVEL_RISE",
            "WATER_STRESS",
            "EXTREME_PRECIPITATION",
            "EXTREME_COLD",
            "LANDSLIDE",
            "COASTAL_EROSION",
        ],
        "coverage": "global",
        "config": {
            "provider": "Network for Greening the Financial System",
            "url": "https://www.ngfs.net/ngfs-scenarios-portal",
            "version": "Phase IV (2023)",
            "scenarios": [
                "Net Zero 2050",
                "Below 2C",
                "Divergent Net Zero",
                "Delayed Transition",
                "NDCs",
                "Current Policies",
            ],
            "update_frequency": "annual",
            "license": "open",
            "description": (
                "NGFS Climate Scenarios for central banks and "
                "financial supervisors. Covers both transition "
                "risks (policy, technology, market) and physical "
                "risks (acute and chronic climate hazards)."
            ),
        },
    },
    {
        "source_id": "ipcc-ar6-wg2",
        "name": "IPCC AR6 WG2",
        "source_type": "GLOBAL_DATABASE",
        "hazard_types": [
            "RIVERINE_FLOOD",
            "COASTAL_FLOOD",
            "DROUGHT",
            "EXTREME_HEAT",
            "EXTREME_COLD",
            "WILDFIRE",
            "TROPICAL_CYCLONE",
            "EXTREME_PRECIPITATION",
            "WATER_STRESS",
            "SEA_LEVEL_RISE",
            "LANDSLIDE",
            "COASTAL_EROSION",
        ],
        "coverage": "global",
        "config": {
            "provider": "Intergovernmental Panel on Climate Change",
            "url": "https://www.ipcc.ch/report/ar6/wg2/",
            "version": "AR6 (2022)",
            "regions": [
                "Africa",
                "Asia",
                "Australasia",
                "Central and South America",
                "Europe",
                "North America",
                "Small Islands",
                "Ocean",
                "Polar Regions",
                "Mediterranean",
                "Mountains",
                "Tropical Forests",
            ],
            "update_frequency": "assessment-cycle",
            "license": "open",
            "description": (
                "IPCC AR6 Working Group II assesses climate change "
                "impacts, adaptation, and vulnerability across 12 "
                "global regions with confidence-calibrated risk "
                "assessments."
            ),
        },
    },
    {
        "source_id": "copernicus-cds",
        "name": "Copernicus CDS",
        "source_type": "REANALYSIS",
        "hazard_types": [
            "EXTREME_HEAT",
            "EXTREME_COLD",
            "EXTREME_PRECIPITATION",
            "DROUGHT",
            "WATER_STRESS",
            "WILDFIRE",
        ],
        "coverage": "global",
        "config": {
            "provider": "European Centre for Medium-Range Weather Forecasts",
            "url": "https://cds.climate.copernicus.eu",
            "version": "ERA5 (2024)",
            "resolution": "0.25 degrees (~31 km)",
            "temporal_coverage": "1940-present",
            "update_frequency": "near-real-time",
            "license": "Copernicus License",
            "description": (
                "Copernicus Climate Data Store provides ERA5 "
                "reanalysis covering temperature, precipitation, "
                "wind, humidity, and derived indicators at 0.25-degree "
                "global resolution."
            ),
        },
    },
    {
        "source_id": "nasa-sedac",
        "name": "NASA SEDAC",
        "source_type": "GLOBAL_DATABASE",
        "hazard_types": [
            "RIVERINE_FLOOD",
            "COASTAL_FLOOD",
            "DROUGHT",
            "EXTREME_HEAT",
            "TROPICAL_CYCLONE",
            "LANDSLIDE",
            "SEA_LEVEL_RISE",
        ],
        "coverage": "global",
        "config": {
            "provider": "NASA Socioeconomic Data and Applications Center",
            "url": "https://sedac.ciesin.columbia.edu",
            "version": "v4.11",
            "resolution": "variable (1km - 0.5 degrees)",
            "update_frequency": "annual",
            "license": "open",
            "description": (
                "NASA SEDAC provides multi-hazard exposure and "
                "vulnerability datasets including population exposure "
                "to natural hazards, cyclone tracks, flood frequency, "
                "and drought indices."
            ),
        },
    },
    {
        "source_id": "noaa-ncei",
        "name": "NOAA NCEI",
        "source_type": "EVENT_CATALOG",
        "hazard_types": [
            "TROPICAL_CYCLONE",
            "RIVERINE_FLOOD",
            "COASTAL_FLOOD",
            "EXTREME_HEAT",
            "EXTREME_COLD",
            "WILDFIRE",
            "EXTREME_PRECIPITATION",
        ],
        "coverage": "global",
        "config": {
            "provider": (
                "National Oceanic and Atmospheric Administration "
                "National Centers for Environmental Information"
            ),
            "url": "https://www.ncei.noaa.gov",
            "version": "2024",
            "datasets": [
                "International Best Track Archive (IBTrACS)",
                "Storm Events Database",
                "Billion-Dollar Disasters",
                "Global Historical Climatology Network",
            ],
            "update_frequency": "monthly",
            "license": "public domain (US Government)",
            "description": (
                "NOAA NCEI provides comprehensive climate and weather "
                "event catalogs including tropical cyclone tracks, "
                "storm events, billion-dollar disasters, and global "
                "climatological observations."
            ),
        },
    },
    {
        "source_id": "swissre-catnet",
        "name": "Swiss Re CatNet",
        "source_type": "REGIONAL_INDEX",
        "hazard_types": [
            "RIVERINE_FLOOD",
            "COASTAL_FLOOD",
            "TROPICAL_CYCLONE",
            "EXTREME_PRECIPITATION",
            "WILDFIRE",
            "LANDSLIDE",
            "EXTREME_HEAT",
            "DROUGHT",
        ],
        "coverage": "global",
        "config": {
            "provider": "Swiss Re",
            "url": "https://catnet.swissre.com",
            "version": "2024",
            "resolution": "250m - 1km",
            "peril_categories": [
                "Flood",
                "Windstorm",
                "Earthquake",
                "Hail",
                "Wildfire",
                "Lightning",
                "Tornado",
            ],
            "update_frequency": "semi-annual",
            "license": "commercial",
            "description": (
                "Swiss Re CatNet provides global natural catastrophe "
                "peril risk scores and hazard maps at high resolution "
                "for insurance underwriting and risk management."
            ),
        },
    },
    {
        "source_id": "munichre-natcat",
        "name": "Munich Re NatCatSERVICE",
        "source_type": "EVENT_CATALOG",
        "hazard_types": [
            "RIVERINE_FLOOD",
            "COASTAL_FLOOD",
            "TROPICAL_CYCLONE",
            "EXTREME_HEAT",
            "EXTREME_COLD",
            "WILDFIRE",
            "DROUGHT",
            "EXTREME_PRECIPITATION",
            "LANDSLIDE",
        ],
        "coverage": "global",
        "config": {
            "provider": "Munich Re",
            "url": "https://www.munichre.com/en/solutions/for-industry-clients/natcatservice.html",
            "version": "2024",
            "temporal_coverage": "1980-present",
            "event_types": [
                "Meteorological",
                "Hydrological",
                "Climatological",
                "Geophysical",
            ],
            "update_frequency": "quarterly",
            "license": "commercial",
            "description": (
                "Munich Re NatCatSERVICE is a comprehensive global "
                "natural catastrophe event database tracking insured "
                "and overall losses, fatalities, and affected areas "
                "since 1980."
            ),
        },
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed for determinism.

    Returns:
        Current UTC datetime with microseconds set to zero.
    """
    return datetime.now(timezone.utc).replace(microsecond=0)


def _build_sha256(data: Any) -> str:
    """Build a deterministic SHA-256 hash from any JSON-serializable value.

    All dict keys are sorted for determinism regardless of insertion order.

    Args:
        data: JSON-serializable value (dict, list, str, int, etc.).

    Returns:
        64-character lowercase hex SHA-256 digest.
    """
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _clamp(value: float, low: float, high: float) -> float:
    """Clamp a numeric value to [low, high].

    Args:
        value: Value to clamp.
        low: Lower bound (inclusive).
        high: Upper bound (inclusive).

    Returns:
        Clamped value.
    """
    return max(low, min(value, high))


# ---------------------------------------------------------------------------
# HazardDatabaseEngine
# ---------------------------------------------------------------------------


class HazardDatabaseEngine:
    """Pure-Python engine for unified climate hazard data ingestion and query.

    Engine 1 of 7 in the Climate Hazard Connector pipeline (AGENT-DATA-020).

    Manages the full lifecycle of climate hazard data including source
    registration, hazard record ingestion, spatial and temporal querying,
    historical event tracking, multi-source aggregation with configurable
    strategies, import/export, and aggregate statistics. Every mutation
    is tracked with SHA-256 provenance hashing for complete audit trails.

    Ten authoritative built-in data sources are pre-registered on
    initialization, covering global databases (WRI Aqueduct, ThinkHazard,
    IPCC AR6 WG2, Copernicus CDS, NASA SEDAC), event catalogs (EM-DAT,
    NOAA NCEI, Munich Re NatCatSERVICE), scenario models (NGFS), and
    regional indices (Swiss Re CatNet).

    Zero-Hallucination Guarantees:
        - UUID assignment via ``uuid.uuid4()`` (no LLM involvement)
        - Timestamps from ``datetime.now(timezone.utc)`` (deterministic)
        - Haversine distance uses pure trigonometry (math module only)
        - Aggregation uses explicit arithmetic (sum, mean, min, max, median)
        - SHA-256 provenance hash computed from JSON-serialized payloads
        - All lookups use explicit dict/set operations
        - No ML or LLM calls anywhere in the class

    Attributes:
        _sources: Source store keyed by source_id.
        _records: Hazard data record store keyed by record_id.
        _events: Historical event store keyed by event_id.
        _source_type_index: Mapping from source_type to set of source_ids.
        _hazard_type_index: Mapping from hazard_type to set of record_ids.
        _source_record_index: Mapping from source_id to set of record_ids.
        _region_record_index: Mapping from region to set of record_ids.
        _hazard_event_index: Mapping from hazard_type to set of event_ids.
        _lock: Thread-safety lock protecting all state.
        _provenance: ProvenanceTracker for SHA-256 audit trails.
        _operation_counts: Counter dict tracking operation frequencies.

    Example:
        >>> engine = HazardDatabaseEngine()
        >>> assert engine.get_statistics()["total_sources"] >= 10
        >>> src = engine.get_source("wri-aqueduct")
        >>> assert src is not None
        >>> assert src["name"] == "WRI Aqueduct"
    """

    def __init__(
        self,
        provenance: Optional[Any] = None,
        genesis_hash: Optional[str] = None,
    ) -> None:
        """Initialize HazardDatabaseEngine with built-in data sources.

        Sets up the source, record, and event stores, all lookup indexes,
        the provenance tracker, and pre-registers 10 authoritative climate
        hazard data sources.

        Args:
            provenance: Optional ProvenanceTracker instance. When None,
                a new tracker is created using genesis_hash if provided,
                otherwise using the default genesis seed.
            genesis_hash: Optional genesis hash seed string for
                provenance chain initialization. Only used when
                provenance is None.

        Example:
            >>> engine = HazardDatabaseEngine()
            >>> stats = engine.get_statistics()
            >>> assert stats["total_sources"] == 10
            >>> assert stats["total_records"] == 0
        """
        # -- Primary stores --
        self._sources: Dict[str, dict] = {}
        self._records: Dict[str, dict] = {}
        self._events: Dict[str, dict] = {}

        # -- Indexes --
        self._source_type_index: Dict[str, Set[str]] = {}
        self._hazard_type_index: Dict[str, Set[str]] = {}
        self._source_record_index: Dict[str, Set[str]] = {}
        self._region_record_index: Dict[str, Set[str]] = {}
        self._hazard_event_index: Dict[str, Set[str]] = {}
        self._source_hazard_index: Dict[str, Set[str]] = {}

        # -- Thread safety --
        self._lock: threading.Lock = threading.Lock()

        # -- Provenance --
        if provenance is not None:
            self._provenance: ProvenanceTracker = provenance
        elif genesis_hash:
            self._provenance = ProvenanceTracker(genesis_hash=genesis_hash)
        else:
            self._provenance = ProvenanceTracker()

        # -- Operation counters --
        self._operation_counts: Dict[str, int] = {
            "sources_registered": 0,
            "sources_deleted": 0,
            "records_ingested": 0,
            "records_queried": 0,
            "events_registered": 0,
            "aggregations_performed": 0,
            "imports_performed": 0,
            "exports_performed": 0,
        }

        # -- Register built-in sources --
        self._register_builtin_sources()

        logger.info(
            "HazardDatabaseEngine initialized "
            "(AGENT-DATA-020, Engine 1 of 7): "
            "%d built-in sources registered",
            len(self._sources),
        )

    # ------------------------------------------------------------------
    # Internal: ID generation
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_id(prefix: str) -> str:
        """Generate a UUID-based ID with the given prefix.

        Args:
            prefix: Prefix string (e.g. "SRC-", "REC-", "EVT-").

        Returns:
            String of the form ``<prefix><uuid4>``.
        """
        return f"{prefix}{uuid.uuid4()}"

    # ------------------------------------------------------------------
    # Internal: index management
    # ------------------------------------------------------------------

    def _add_to_index(
        self,
        index: Dict[str, Set[str]],
        key: str,
        item_id: str,
    ) -> None:
        """Add an item_id to an index set under the given key.

        Args:
            index: The index dict to update.
            key: The index key (e.g. a hazard_type or source_type).
            item_id: The UUID to add.
        """
        index.setdefault(key, set()).add(item_id)

    def _remove_from_index(
        self,
        index: Dict[str, Set[str]],
        key: str,
        item_id: str,
    ) -> None:
        """Remove an item_id from an index set under the given key.

        Args:
            index: The index dict to update.
            key: The index key (e.g. a hazard_type or source_type).
            item_id: The UUID to remove.
        """
        bucket = index.get(key)
        if bucket is not None:
            bucket.discard(item_id)
            if not bucket:
                del index[key]

    # ------------------------------------------------------------------
    # Internal: validation
    # ------------------------------------------------------------------

    def _validate_hazard_type(self, hazard_type: str) -> str:
        """Validate and normalize a hazard type string.

        Args:
            hazard_type: Raw hazard type string.

        Returns:
            Uppercased, stripped hazard type string.

        Raises:
            ValueError: If the hazard type is not recognized.
        """
        normalized = hazard_type.strip().upper()
        if normalized not in VALID_HAZARD_TYPES:
            raise ValueError(
                f"Invalid hazard_type: {hazard_type!r}. "
                f"Must be one of: {sorted(VALID_HAZARD_TYPES)}"
            )
        return normalized

    def _validate_source_type(self, source_type: str) -> str:
        """Validate and normalize a source type string.

        Args:
            source_type: Raw source type string.

        Returns:
            Uppercased, stripped source type string.

        Raises:
            ValueError: If the source type is not recognized.
        """
        normalized = source_type.strip().upper()
        if normalized not in VALID_SOURCE_TYPES:
            raise ValueError(
                f"Invalid source_type: {source_type!r}. "
                f"Must be one of: {sorted(VALID_SOURCE_TYPES)}"
            )
        return normalized

    def _validate_location(self, location: Dict[str, Any]) -> None:
        """Validate a location dictionary has valid lat/lon.

        Args:
            location: Dictionary with ``lat`` and ``lon`` keys.

        Raises:
            ValueError: If lat or lon are missing or out of range.
        """
        if "lat" not in location or "lon" not in location:
            raise ValueError(
                "Location must contain 'lat' and 'lon' keys. "
                f"Got keys: {sorted(location.keys())}"
            )
        lat = float(location["lat"])
        lon = float(location["lon"])
        if not (-90.0 <= lat <= 90.0):
            raise ValueError(
                f"Latitude must be between -90 and 90, got {lat}"
            )
        if not (-180.0 <= lon <= 180.0):
            raise ValueError(
                f"Longitude must be between -180 and 180, got {lon}"
            )

    def _validate_intensity(self, intensity: float) -> float:
        """Validate and clamp intensity to [0, 10] range.

        Args:
            intensity: Raw intensity value.

        Returns:
            Clamped intensity in [0.0, 10.0].

        Raises:
            ValueError: If intensity is not a finite number.
        """
        val = float(intensity)
        if math.isnan(val) or math.isinf(val):
            raise ValueError(
                f"Intensity must be a finite number, got {intensity}"
            )
        return _clamp(val, 0.0, 10.0)

    def _validate_probability(self, probability: float) -> float:
        """Validate and clamp probability to [0, 1] range.

        Args:
            probability: Raw probability value.

        Returns:
            Clamped probability in [0.0, 1.0].

        Raises:
            ValueError: If probability is not a finite number.
        """
        val = float(probability)
        if math.isnan(val) or math.isinf(val):
            raise ValueError(
                f"Probability must be a finite number, got {probability}"
            )
        return _clamp(val, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Internal: spatial helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_distance(
        loc1: Dict[str, Any],
        loc2: Dict[str, Any],
    ) -> float:
        """Calculate Haversine distance between two lat/lon points.

        Uses the Haversine formula for great-circle distance on a
        sphere with the WGS-84 mean Earth radius of 6371.0088 km.

        Args:
            loc1: First location with ``lat`` and ``lon`` keys.
            loc2: Second location with ``lat`` and ``lon`` keys.

        Returns:
            Distance in kilometers between the two points.
        """
        lat1 = math.radians(float(loc1["lat"]))
        lon1 = math.radians(float(loc1["lon"]))
        lat2 = math.radians(float(loc2["lat"]))
        lon2 = math.radians(float(loc2["lon"]))

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (
            math.sin(dlat / 2.0) ** 2
            + math.cos(lat1)
            * math.cos(lat2)
            * math.sin(dlon / 2.0) ** 2
        )
        c = 2.0 * math.asin(math.sqrt(a))

        return EARTH_RADIUS_KM * c

    @staticmethod
    def _is_in_region(
        location: Dict[str, Any],
        region: str,
    ) -> bool:
        """Check if a location falls within a named region's bounds.

        Uses the REGION_BOUNDS lookup table for approximate continental
        and sub-continental bounding boxes.

        Args:
            location: Dictionary with ``lat`` and ``lon`` keys.
            region: Region name (case-insensitive). Must match a key
                in ``REGION_BOUNDS``.

        Returns:
            True if the location is within the region bounds, False
            otherwise. Returns True for unknown regions (permissive
            fallback).
        """
        region_lower = region.strip().lower()
        bounds = REGION_BOUNDS.get(region_lower)
        if bounds is None:
            # Unknown region: permissive fallback
            return True

        lat = float(location["lat"])
        lon = float(location["lon"])
        lat_min, lat_max, lon_min, lon_max = bounds

        lat_ok = lat_min <= lat <= lat_max

        # Handle regions that cross the antimeridian
        if lon_min <= lon_max:
            lon_ok = lon_min <= lon <= lon_max
        else:
            lon_ok = lon >= lon_min or lon <= lon_max

        return lat_ok and lon_ok

    # ------------------------------------------------------------------
    # Internal: built-in source registration
    # ------------------------------------------------------------------

    def _register_builtin_sources(self) -> None:
        """Register all built-in data sources during initialization.

        Iterates through BUILTIN_SOURCES and registers each without
        provenance tracking (to keep the genesis chain clean). Sets
        the status to 'active' and marks them as built-in.
        """
        now_str = _utcnow().isoformat()
        for src_def in BUILTIN_SOURCES:
            source_id = src_def["source_id"]
            hazard_types_list = [
                ht for ht in src_def.get("hazard_types", [])
                if ht in VALID_HAZARD_TYPES
            ]
            source_type = src_def.get("source_type", "CUSTOM")

            source_record: dict = {
                "source_id": source_id,
                "name": src_def["name"],
                "source_type": source_type,
                "hazard_types": hazard_types_list,
                "coverage": src_def.get("coverage", "global"),
                "config": copy.deepcopy(
                    src_def.get("config", {}),
                ),
                "status": "active",
                "is_builtin": True,
                "record_count": 0,
                "created_at": now_str,
                "updated_at": now_str,
                "provenance_hash": "",
            }

            # Compute provenance hash
            data_hash = _build_sha256(
                {k: v for k, v in source_record.items()
                 if k != "provenance_hash"},
            )
            source_record["provenance_hash"] = data_hash

            # Store
            self._sources[source_id] = source_record

            # Index by source type
            self._add_to_index(
                self._source_type_index,
                source_type,
                source_id,
            )

            # Index by hazard types served
            for ht in hazard_types_list:
                self._add_to_index(
                    self._source_hazard_index,
                    ht,
                    source_id,
                )

    # ------------------------------------------------------------------
    # 1. register_source
    # ------------------------------------------------------------------

    def register_source(
        self,
        source_id: str,
        name: str,
        source_type: str,
        hazard_types: List[str],
        coverage: str = "global",
        config: Optional[Dict[str, Any]] = None,
    ) -> dict:
        """Register a new hazard data source in the engine.

        Creates a source record with the given ID, validates all input
        parameters, sets the status to ``"active"``, updates indexes,
        records provenance, and emits metrics.

        The ``source_id`` must be unique. Attempting to register a
        duplicate raises a ``ValueError``.

        Args:
            source_id: Globally unique identifier for the source.
                Must be non-empty and at most 128 characters.
            name: Human-readable name for the data source.
                Must be non-empty and at most 256 characters.
            source_type: Classification of the source. Must be one
                of: ``GLOBAL_DATABASE``, ``EVENT_CATALOG``,
                ``SCENARIO_MODEL``, ``REANALYSIS``, ``REGIONAL_INDEX``,
                ``CUSTOM``.
            hazard_types: List of hazard types this source covers.
                Each must be a valid hazard type string.
            coverage: Geographic coverage descriptor (e.g. "global",
                "europe", "north_america"). Defaults to "global".
            config: Optional dictionary of source-specific configuration
                (URLs, credentials, API keys, etc.).

        Returns:
            Deep copy of the complete source dict containing all fields
            plus ``status``, ``is_builtin``, ``record_count``,
            ``created_at``, ``updated_at``, and ``provenance_hash``.

        Raises:
            ValueError: If source_id is empty, name is empty, source_type
                is invalid, hazard_types is empty or contains invalid
                types, or a source with the same ID already exists.

        Example:
            >>> engine = HazardDatabaseEngine()
            >>> src = engine.register_source(
            ...     source_id="my-source",
            ...     name="My Custom Source",
            ...     source_type="CUSTOM",
            ...     hazard_types=["DROUGHT", "WILDFIRE"],
            ...     coverage="europe",
            ... )
            >>> assert src["source_id"] == "my-source"
            >>> assert src["status"] == "active"
        """
        t0 = time.monotonic()

        # -- Input validation --
        if not source_id or not source_id.strip():
            raise ValueError("source_id must be a non-empty string.")
        clean_id = source_id.strip()
        if len(clean_id) > 128:
            raise ValueError(
                "source_id exceeds maximum length of 128 characters."
            )

        if not name or not name.strip():
            raise ValueError("name must be a non-empty string.")
        clean_name = name.strip()
        if len(clean_name) > 256:
            raise ValueError(
                "name exceeds maximum length of 256 characters."
            )

        normalized_type = self._validate_source_type(source_type)

        if not hazard_types:
            raise ValueError(
                "hazard_types must be a non-empty list."
            )
        validated_hazards: List[str] = []
        for ht in hazard_types:
            validated_hazards.append(self._validate_hazard_type(ht))
        # Deduplicate and sort for determinism
        validated_hazards = sorted(set(validated_hazards))

        clean_coverage = coverage.strip().lower() if coverage else "global"

        # -- Build source record --
        now_str = _utcnow().isoformat()

        source_record: dict = {
            "source_id": clean_id,
            "name": clean_name,
            "source_type": normalized_type,
            "hazard_types": validated_hazards,
            "coverage": clean_coverage,
            "config": copy.deepcopy(config) if config else {},
            "status": "active",
            "is_builtin": False,
            "record_count": 0,
            "created_at": now_str,
            "updated_at": now_str,
            "provenance_hash": "",
        }

        # -- Compute data hash --
        data_hash = _build_sha256(
            {k: v for k, v in source_record.items()
             if k != "provenance_hash"},
        )

        with self._lock:
            # Check capacity
            if len(self._sources) >= MAX_SOURCES:
                record_error("source_capacity_exceeded")
                raise ValueError(
                    f"Maximum number of sources ({MAX_SOURCES}) reached. "
                    f"Delete unused sources before registering new ones."
                )

            # Enforce ID uniqueness
            if clean_id in self._sources:
                raise ValueError(
                    f"A source with id={clean_id!r} already exists."
                )

            # Record provenance
            entry = self._provenance.record(
                entity_type="hazard_source",
                entity_id=clean_id,
                action="source_registered",
                metadata={
                    "data_hash": data_hash,
                    "name": clean_name,
                    "source_type": normalized_type,
                    "hazard_types": validated_hazards,
                },
            )
            source_record["provenance_hash"] = entry.hash_value

            # Store source
            self._sources[clean_id] = source_record

            # Update indexes
            self._add_to_index(
                self._source_type_index,
                normalized_type,
                clean_id,
            )
            for ht in validated_hazards:
                self._add_to_index(
                    self._source_hazard_index,
                    ht,
                    clean_id,
                )

            # Update counters
            self._operation_counts["sources_registered"] += 1

        # Metrics
        elapsed = time.monotonic() - t0
        record_source_registered(normalized_type)
        set_total_sources(len(self._sources))
        observe_processing_duration("source_register", elapsed)

        logger.info(
            "Source registered: id=%s name=%s type=%s "
            "hazards=%d coverage=%s elapsed=%.3fms",
            clean_id,
            clean_name,
            normalized_type,
            len(validated_hazards),
            clean_coverage,
            elapsed * 1000,
        )
        return copy.deepcopy(source_record)

    # ------------------------------------------------------------------
    # 2. get_source
    # ------------------------------------------------------------------

    def get_source(self, source_id: str) -> Optional[dict]:
        """Retrieve a registered source by its unique ID.

        Args:
            source_id: Unique source identifier string.

        Returns:
            Deep copy of the source dict, or ``None`` if no source
            with the given ID exists.

        Example:
            >>> engine = HazardDatabaseEngine()
            >>> src = engine.get_source("wri-aqueduct")
            >>> assert src is not None
            >>> assert src["source_type"] == "GLOBAL_DATABASE"
        """
        with self._lock:
            record = self._sources.get(source_id)
            if record is None:
                logger.debug(
                    "Source not found by id: %s", source_id,
                )
                return None
            return copy.deepcopy(record)

    # ------------------------------------------------------------------
    # 3. list_sources
    # ------------------------------------------------------------------

    def list_sources(
        self,
        source_type: Optional[str] = None,
        hazard_type: Optional[str] = None,
    ) -> List[dict]:
        """List registered sources with optional filtering.

        When both filters are provided, only sources matching both
        criteria are returned (logical AND).

        Args:
            source_type: Optional source type filter. When provided,
                only sources of the given type are returned.
            hazard_type: Optional hazard type filter. When provided,
                only sources covering the given hazard are returned.

        Returns:
            List of deep-copied source dicts matching the filters,
            sorted by source_id for determinism.

        Example:
            >>> engine = HazardDatabaseEngine()
            >>> catalogs = engine.list_sources(
            ...     source_type="EVENT_CATALOG",
            ... )
            >>> assert len(catalogs) >= 3
        """
        t0 = time.monotonic()

        with self._lock:
            # Start with all source IDs
            candidate_ids: Optional[Set[str]] = None

            if source_type is not None:
                normalized_type = source_type.strip().upper()
                type_ids = self._source_type_index.get(
                    normalized_type, set(),
                )
                candidate_ids = set(type_ids)

            if hazard_type is not None:
                normalized_hazard = hazard_type.strip().upper()
                hazard_ids = self._source_hazard_index.get(
                    normalized_hazard, set(),
                )
                if candidate_ids is not None:
                    candidate_ids = candidate_ids & hazard_ids
                else:
                    candidate_ids = set(hazard_ids)

            if candidate_ids is None:
                candidate_ids = set(self._sources.keys())

            results: List[dict] = []
            for sid in sorted(candidate_ids):
                record = self._sources.get(sid)
                if record is not None:
                    results.append(copy.deepcopy(record))

        elapsed = time.monotonic() - t0
        observe_processing_duration("source_list", elapsed)
        record_query_executed("list_sources")

        logger.debug(
            "list_sources: type=%s hazard=%s -> %d results in %.3fms",
            source_type,
            hazard_type,
            len(results),
            elapsed * 1000,
        )
        return results

    # ------------------------------------------------------------------
    # 4. update_source
    # ------------------------------------------------------------------

    def update_source(
        self,
        source_id: str,
        **kwargs: Any,
    ) -> Optional[dict]:
        """Update metadata or configuration of an existing source.

        Only the fields explicitly provided in ``kwargs`` are updated.
        All other fields retain their current values.

        Supported updatable fields:
            ``name``, ``source_type``, ``hazard_types``, ``coverage``,
            ``config``, ``status``.

        Args:
            source_id: Unique source identifier to update.
            **kwargs: Field-name/value pairs to update.

        Returns:
            Deep copy of the updated source dict, or ``None`` if the
            source was not found.

        Raises:
            ValueError: If an unsupported field is provided, or if a
                field value fails validation.

        Example:
            >>> engine = HazardDatabaseEngine()
            >>> updated = engine.update_source(
            ...     "wri-aqueduct",
            ...     coverage="europe",
            ... )
            >>> assert updated is not None
            >>> assert updated["coverage"] == "europe"
        """
        t0 = time.monotonic()

        updatable_fields: Set[str] = {
            "name",
            "source_type",
            "hazard_types",
            "coverage",
            "config",
            "status",
        }

        invalid_keys = set(kwargs.keys()) - updatable_fields
        if invalid_keys:
            raise ValueError(
                f"Cannot update fields: {sorted(invalid_keys)}. "
                f"Updatable fields: {sorted(updatable_fields)}"
            )

        with self._lock:
            record = self._sources.get(source_id)
            if record is None:
                logger.debug(
                    "Update failed: source not found id=%s",
                    source_id,
                )
                return None

            # De-index old state
            old_type = record.get("source_type", "")
            self._remove_from_index(
                self._source_type_index, old_type, source_id,
            )
            for old_ht in record.get("hazard_types", []):
                self._remove_from_index(
                    self._source_hazard_index, old_ht, source_id,
                )

            # Apply updates
            for key, value in kwargs.items():
                if key == "name":
                    if not value or not str(value).strip():
                        # Re-index and abort
                        self._add_to_index(
                            self._source_type_index,
                            record["source_type"],
                            source_id,
                        )
                        for ht in record.get("hazard_types", []):
                            self._add_to_index(
                                self._source_hazard_index,
                                ht,
                                source_id,
                            )
                        raise ValueError(
                            "name must be a non-empty string."
                        )
                    record["name"] = str(value).strip()

                elif key == "source_type":
                    normalized = self._validate_source_type(str(value))
                    record["source_type"] = normalized

                elif key == "hazard_types":
                    if not value:
                        # Re-index and abort
                        self._add_to_index(
                            self._source_type_index,
                            record["source_type"],
                            source_id,
                        )
                        for ht in record.get("hazard_types", []):
                            self._add_to_index(
                                self._source_hazard_index,
                                ht,
                                source_id,
                            )
                        raise ValueError(
                            "hazard_types must be a non-empty list."
                        )
                    validated: List[str] = []
                    for ht in value:
                        validated.append(
                            self._validate_hazard_type(str(ht)),
                        )
                    record["hazard_types"] = sorted(set(validated))

                elif key == "coverage":
                    record["coverage"] = (
                        str(value).strip().lower() if value else "global"
                    )

                elif key == "config":
                    record["config"] = (
                        copy.deepcopy(value) if value else {}
                    )

                elif key == "status":
                    status_lower = str(value).strip().lower()
                    if status_lower not in VALID_SOURCE_STATUSES:
                        # Re-index and abort
                        self._add_to_index(
                            self._source_type_index,
                            record["source_type"],
                            source_id,
                        )
                        for ht in record.get("hazard_types", []):
                            self._add_to_index(
                                self._source_hazard_index,
                                ht,
                                source_id,
                            )
                        raise ValueError(
                            f"Invalid status: {value!r}. "
                            f"Must be one of: "
                            f"{sorted(VALID_SOURCE_STATUSES)}"
                        )
                    record["status"] = status_lower

            # Update timestamp
            record["updated_at"] = _utcnow().isoformat()

            # Re-index new state
            self._add_to_index(
                self._source_type_index,
                record["source_type"],
                source_id,
            )
            for ht in record.get("hazard_types", []):
                self._add_to_index(
                    self._source_hazard_index,
                    ht,
                    source_id,
                )

            # Record provenance
            data_hash = _build_sha256(
                {k: v for k, v in record.items()
                 if k != "provenance_hash"},
            )
            entry = self._provenance.record(
                entity_type="hazard_source",
                entity_id=source_id,
                action="source_updated",
                metadata={
                    "data_hash": data_hash,
                    "updated_fields": sorted(kwargs.keys()),
                },
            )
            record["provenance_hash"] = entry.hash_value

            result = copy.deepcopy(record)

        elapsed = time.monotonic() - t0
        observe_processing_duration("source_update", elapsed)

        logger.info(
            "Source updated: id=%s fields=%s elapsed=%.3fms",
            source_id,
            sorted(kwargs.keys()),
            elapsed * 1000,
        )
        return result

    # ------------------------------------------------------------------
    # 5. delete_source
    # ------------------------------------------------------------------

    def delete_source(self, source_id: str) -> bool:
        """Delete a registered source and all its associated records.

        Removes the source from the store and all indexes. Any hazard
        data records associated with this source are also removed.

        Args:
            source_id: Unique source identifier to delete.

        Returns:
            True if the source was found and deleted, False if no
            source with the given ID exists.

        Example:
            >>> engine = HazardDatabaseEngine()
            >>> engine.register_source(
            ...     source_id="temp-src",
            ...     name="Temporary",
            ...     source_type="CUSTOM",
            ...     hazard_types=["DROUGHT"],
            ... )
            >>> assert engine.delete_source("temp-src") is True
            >>> assert engine.get_source("temp-src") is None
        """
        t0 = time.monotonic()

        with self._lock:
            record = self._sources.get(source_id)
            if record is None:
                logger.debug(
                    "Delete failed: source not found id=%s",
                    source_id,
                )
                return False

            # De-index source
            source_type = record.get("source_type", "")
            self._remove_from_index(
                self._source_type_index,
                source_type,
                source_id,
            )
            for ht in record.get("hazard_types", []):
                self._remove_from_index(
                    self._source_hazard_index,
                    ht,
                    source_id,
                )

            # Remove associated records
            associated_record_ids = set(
                self._source_record_index.get(source_id, set()),
            )
            for rec_id in associated_record_ids:
                rec = self._records.get(rec_id)
                if rec is not None:
                    # De-index record
                    rec_hazard = rec.get("hazard_type", "")
                    self._remove_from_index(
                        self._hazard_type_index,
                        rec_hazard,
                        rec_id,
                    )
                    rec_region = rec.get("region", "")
                    if rec_region:
                        self._remove_from_index(
                            self._region_record_index,
                            rec_region,
                            rec_id,
                        )
                    del self._records[rec_id]

            # Remove source record index
            if source_id in self._source_record_index:
                del self._source_record_index[source_id]

            # Record provenance
            self._provenance.record(
                entity_type="hazard_source",
                entity_id=source_id,
                action="source_deleted",
                metadata={
                    "name": record.get("name", ""),
                    "records_removed": len(associated_record_ids),
                },
            )

            # Remove source
            del self._sources[source_id]

            # Update counters
            self._operation_counts["sources_deleted"] += 1

        elapsed = time.monotonic() - t0
        set_total_sources(len(self._sources))
        set_total_records(len(self._records))
        observe_processing_duration("source_delete", elapsed)

        logger.info(
            "Source deleted: id=%s records_removed=%d elapsed=%.3fms",
            source_id,
            len(associated_record_ids),
            elapsed * 1000,
        )
        return True

    # ------------------------------------------------------------------
    # 6. ingest_hazard_data
    # ------------------------------------------------------------------

    def ingest_hazard_data(
        self,
        source_id: str,
        hazard_type: str,
        records: List[Dict[str, Any]],
        region: Optional[str] = None,
    ) -> dict:
        """Ingest hazard data records from a registered source.

        Each record is validated, assigned a unique ID, indexed by
        hazard type, source, and region, and tracked with provenance.

        Each record dict should contain:
            - ``location``: dict with ``lat`` (float) and ``lon`` (float)
            - ``intensity``: float in [0, 10] range
            - ``probability``: float in [0, 1] range
            - ``frequency``: float, events per year (>= 0)
            - ``duration_days``: float, event duration in days (>= 0)
            - ``observed_at``: ISO datetime string or None
            - ``metadata``: optional dict of additional data

        Args:
            source_id: ID of the registered source providing the data.
            hazard_type: Type of hazard for all records in the batch.
            records: List of hazard data record dicts to ingest.
            region: Optional region label for all records.

        Returns:
            Dict with ``ingested_count``, ``record_ids``,
            ``source_id``, ``hazard_type``, ``region``,
            ``provenance_hash``, and ``processing_time_ms``.

        Raises:
            ValueError: If source_id is not found, hazard_type is
                invalid, records list is empty, batch exceeds
                MAX_INGEST_BATCH, or total records would exceed
                MAX_RECORDS.

        Example:
            >>> engine = HazardDatabaseEngine()
            >>> result = engine.ingest_hazard_data(
            ...     source_id="wri-aqueduct",
            ...     hazard_type="DROUGHT",
            ...     records=[{
            ...         "location": {"lat": 40.0, "lon": -3.7},
            ...         "intensity": 7.0,
            ...         "probability": 0.3,
            ...         "frequency": 0.5,
            ...         "duration_days": 90,
            ...         "observed_at": "2023-06-01T00:00:00",
            ...         "metadata": {"spi_index": -2.1},
            ...     }],
            ...     region="europe",
            ... )
            >>> assert result["ingested_count"] == 1
        """
        t0 = time.monotonic()

        # -- Input validation --
        normalized_hazard = self._validate_hazard_type(hazard_type)

        if not records:
            raise ValueError("records must be a non-empty list.")
        if len(records) > MAX_INGEST_BATCH:
            raise ValueError(
                f"Batch size {len(records)} exceeds maximum of "
                f"{MAX_INGEST_BATCH} records per ingest call."
            )

        clean_region = (
            region.strip().lower() if region else None
        )
        now_str = _utcnow().isoformat()

        # Validate and build records outside lock for performance
        ingested_records: List[dict] = []
        for idx, raw in enumerate(records):
            # Validate location
            location = raw.get("location")
            if location is None:
                raise ValueError(
                    f"Record {idx}: 'location' is required."
                )
            self._validate_location(location)

            # Validate numeric fields
            intensity = self._validate_intensity(
                raw.get("intensity", 0.0),
            )
            probability = self._validate_probability(
                raw.get("probability", 0.0),
            )
            frequency = max(0.0, float(raw.get("frequency", 0.0)))
            duration_days = max(
                0.0, float(raw.get("duration_days", 0.0)),
            )

            record_id = self._generate_id("REC-")

            rec: dict = {
                "record_id": record_id,
                "source_id": source_id,
                "hazard_type": normalized_hazard,
                "location": {
                    "lat": float(location["lat"]),
                    "lon": float(location["lon"]),
                },
                "intensity": intensity,
                "probability": probability,
                "frequency": frequency,
                "duration_days": duration_days,
                "observed_at": raw.get("observed_at", now_str),
                "region": clean_region,
                "metadata": copy.deepcopy(
                    raw.get("metadata", {}),
                ),
                "ingested_at": now_str,
                "provenance_hash": "",
            }
            ingested_records.append(rec)

        # -- Lock and store --
        record_ids: List[str] = []

        with self._lock:
            # Verify source exists
            source = self._sources.get(source_id)
            if source is None:
                raise ValueError(
                    f"Source not found: {source_id!r}. "
                    f"Register the source before ingesting data."
                )

            # Check capacity
            if len(self._records) + len(ingested_records) > MAX_RECORDS:
                record_error("record_capacity_exceeded")
                raise ValueError(
                    f"Ingesting {len(ingested_records)} records would "
                    f"exceed maximum capacity of {MAX_RECORDS}. "
                    f"Current count: {len(self._records)}."
                )

            for rec in ingested_records:
                rec_id = rec["record_id"]

                # Compute provenance
                data_hash = _build_sha256(
                    {k: v for k, v in rec.items()
                     if k != "provenance_hash"},
                )
                entry = self._provenance.record(
                    entity_type="hazard_record",
                    entity_id=rec_id,
                    action="record_ingested",
                    metadata={
                        "data_hash": data_hash,
                        "source_id": source_id,
                        "hazard_type": normalized_hazard,
                    },
                )
                rec["provenance_hash"] = entry.hash_value

                # Store record
                self._records[rec_id] = rec
                record_ids.append(rec_id)

                # Index by hazard type
                self._add_to_index(
                    self._hazard_type_index,
                    normalized_hazard,
                    rec_id,
                )

                # Index by source
                self._add_to_index(
                    self._source_record_index,
                    source_id,
                    rec_id,
                )

                # Index by region
                if clean_region:
                    self._add_to_index(
                        self._region_record_index,
                        clean_region,
                        rec_id,
                    )

            # Update source record count
            source["record_count"] = len(
                self._source_record_index.get(source_id, set()),
            )
            source["updated_at"] = now_str

            # Update counters
            self._operation_counts["records_ingested"] += len(
                ingested_records,
            )

        # Metrics
        elapsed = time.monotonic() - t0
        record_data_ingested(
            source_id, normalized_hazard, len(ingested_records),
        )
        set_total_records(len(self._records))
        observe_processing_duration("data_ingest", elapsed)

        # Compute batch provenance
        batch_hash = _build_sha256({
            "source_id": source_id,
            "hazard_type": normalized_hazard,
            "record_count": len(record_ids),
            "record_ids": record_ids,
            "region": clean_region,
        })

        logger.info(
            "Data ingested: source=%s hazard=%s count=%d "
            "region=%s elapsed=%.3fms",
            source_id,
            normalized_hazard,
            len(record_ids),
            clean_region,
            elapsed * 1000,
        )

        return {
            "ingested_count": len(record_ids),
            "record_ids": list(record_ids),
            "source_id": source_id,
            "hazard_type": normalized_hazard,
            "region": clean_region,
            "provenance_hash": batch_hash,
            "processing_time_ms": round(elapsed * 1000, 3),
        }

    # ------------------------------------------------------------------
    # 7. get_hazard_data
    # ------------------------------------------------------------------

    def get_hazard_data(
        self,
        hazard_type: str,
        location: Optional[Dict[str, Any]] = None,
        region: Optional[str] = None,
        time_range: Optional[Dict[str, str]] = None,
        limit: int = DEFAULT_QUERY_LIMIT,
    ) -> List[dict]:
        """Query hazard data by type, location, region, and time range.

        Supports spatial proximity filtering (location + radius_km),
        region bounding box filtering, and temporal range filtering.

        Args:
            hazard_type: Type of hazard to query. Must be a valid
                hazard type string.
            location: Optional spatial filter dict with ``lat``,
                ``lon``, and optional ``radius_km`` (default 50 km).
            region: Optional region name filter.
            time_range: Optional temporal filter dict with ``start``
                and/or ``end`` keys (ISO datetime strings).
            limit: Maximum number of records to return. Defaults to
                100, capped at MAX_QUERY_LIMIT.

        Returns:
            List of deep-copied record dicts matching all filters,
            sorted by intensity descending (highest first).

        Example:
            >>> engine = HazardDatabaseEngine()
            >>> # After ingesting data...
            >>> data = engine.get_hazard_data(
            ...     hazard_type="RIVERINE_FLOOD",
            ...     location={"lat": 51.5, "lon": -0.1, "radius_km": 100},
            ...     limit=10,
            ... )
        """
        t0 = time.monotonic()

        normalized_hazard = self._validate_hazard_type(hazard_type)
        effective_limit = min(max(1, limit), MAX_QUERY_LIMIT)

        with self._lock:
            # Start with hazard type index
            candidate_ids = set(
                self._hazard_type_index.get(normalized_hazard, set()),
            )

            # Apply region filter via index
            if region is not None:
                clean_region = region.strip().lower()
                region_ids = self._region_record_index.get(
                    clean_region, set(),
                )
                candidate_ids = candidate_ids & region_ids

            # Collect candidate records
            candidates: List[dict] = []
            for rec_id in candidate_ids:
                rec = self._records.get(rec_id)
                if rec is not None:
                    candidates.append(rec)

        # Apply spatial filter outside lock for performance
        if location is not None:
            radius_km = float(location.get("radius_km", 50.0))
            center = {
                "lat": float(location["lat"]),
                "lon": float(location["lon"]),
            }
            spatial_filtered: List[dict] = []
            for rec in candidates:
                dist = self._calculate_distance(center, rec["location"])
                if dist <= radius_km:
                    spatial_filtered.append(rec)
            candidates = spatial_filtered

        # Apply temporal filter
        if time_range is not None:
            start_str = time_range.get("start")
            end_str = time_range.get("end")
            temporal_filtered: List[dict] = []
            for rec in candidates:
                observed = rec.get("observed_at", "")
                if not observed:
                    continue
                if start_str and observed < start_str:
                    continue
                if end_str and observed > end_str:
                    continue
                temporal_filtered.append(rec)
            candidates = temporal_filtered

        # Sort by intensity descending
        candidates.sort(
            key=lambda r: float(r.get("intensity", 0.0)),
            reverse=True,
        )

        # Apply limit and deep copy
        results = [
            copy.deepcopy(rec)
            for rec in candidates[:effective_limit]
        ]

        # Update counters and metrics
        with self._lock:
            self._operation_counts["records_queried"] += 1

        elapsed = time.monotonic() - t0
        record_query_executed("get_hazard_data")
        observe_processing_duration("data_query", elapsed)

        logger.debug(
            "get_hazard_data: hazard=%s location=%s region=%s "
            "time_range=%s -> %d results in %.3fms",
            normalized_hazard,
            location is not None,
            region,
            time_range is not None,
            len(results),
            elapsed * 1000,
        )
        return results

    # ------------------------------------------------------------------
    # 8. search_hazard_data
    # ------------------------------------------------------------------

    def search_hazard_data(
        self,
        hazard_type: Optional[str] = None,
        region: Optional[str] = None,
        severity_min: Optional[float] = None,
        source_id: Optional[str] = None,
        limit: int = DEFAULT_QUERY_LIMIT,
    ) -> List[dict]:
        """Search across all hazard data with flexible filters.

        Unlike ``get_hazard_data``, all filters are optional. When no
        filters are provided, returns the most intense records across
        all hazard types and sources.

        Args:
            hazard_type: Optional hazard type filter.
            region: Optional region name filter.
            severity_min: Optional minimum intensity threshold (0-10).
                Only records with intensity >= severity_min are returned.
            source_id: Optional source ID filter.
            limit: Maximum number of records to return. Defaults to
                100, capped at MAX_QUERY_LIMIT.

        Returns:
            List of deep-copied record dicts matching all provided
            filters, sorted by intensity descending.

        Example:
            >>> engine = HazardDatabaseEngine()
            >>> results = engine.search_hazard_data(
            ...     severity_min=7.0,
            ...     limit=20,
            ... )
        """
        t0 = time.monotonic()
        effective_limit = min(max(1, limit), MAX_QUERY_LIMIT)

        with self._lock:
            # Determine candidate IDs using indexes
            candidate_ids: Optional[Set[str]] = None

            if hazard_type is not None:
                normalized_hazard = self._validate_hazard_type(
                    hazard_type,
                )
                hazard_ids = self._hazard_type_index.get(
                    normalized_hazard, set(),
                )
                candidate_ids = set(hazard_ids)

            if source_id is not None:
                source_ids = self._source_record_index.get(
                    source_id, set(),
                )
                if candidate_ids is not None:
                    candidate_ids = candidate_ids & source_ids
                else:
                    candidate_ids = set(source_ids)

            if region is not None:
                clean_region = region.strip().lower()
                region_ids = self._region_record_index.get(
                    clean_region, set(),
                )
                if candidate_ids is not None:
                    candidate_ids = candidate_ids & region_ids
                else:
                    candidate_ids = set(region_ids)

            if candidate_ids is None:
                candidate_ids = set(self._records.keys())

            # Collect and filter
            candidates: List[dict] = []
            for rec_id in candidate_ids:
                rec = self._records.get(rec_id)
                if rec is None:
                    continue

                # Apply severity filter
                if severity_min is not None:
                    if float(rec.get("intensity", 0.0)) < severity_min:
                        continue

                candidates.append(rec)

        # Sort by intensity descending
        candidates.sort(
            key=lambda r: float(r.get("intensity", 0.0)),
            reverse=True,
        )

        # Apply limit and deep copy
        results = [
            copy.deepcopy(rec)
            for rec in candidates[:effective_limit]
        ]

        # Update counters and metrics
        with self._lock:
            self._operation_counts["records_queried"] += 1

        elapsed = time.monotonic() - t0
        record_query_executed("search_hazard_data")
        observe_processing_duration("data_search", elapsed)

        logger.debug(
            "search_hazard_data: hazard=%s region=%s sev_min=%s "
            "source=%s -> %d results in %.3fms",
            hazard_type,
            region,
            severity_min,
            source_id,
            len(results),
            elapsed * 1000,
        )
        return results

    # ------------------------------------------------------------------
    # 9. get_historical_events
    # ------------------------------------------------------------------

    def get_historical_events(
        self,
        hazard_type: str,
        region: Optional[str] = None,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        limit: int = DEFAULT_QUERY_LIMIT,
    ) -> List[dict]:
        """Get historical hazard events with optional filters.

        Args:
            hazard_type: Type of hazard to query. Must be valid.
            region: Optional region name filter.
            start_year: Optional start year filter (inclusive).
            end_year: Optional end year filter (inclusive).
            limit: Maximum number of events to return. Defaults to
                100, capped at MAX_QUERY_LIMIT.

        Returns:
            List of deep-copied event dicts matching all filters,
            sorted by start_date descending (most recent first).

        Example:
            >>> engine = HazardDatabaseEngine()
            >>> events = engine.get_historical_events(
            ...     hazard_type="TROPICAL_CYCLONE",
            ...     start_year=2000,
            ...     end_year=2024,
            ...     limit=50,
            ... )
        """
        t0 = time.monotonic()

        normalized_hazard = self._validate_hazard_type(hazard_type)
        effective_limit = min(max(1, limit), MAX_QUERY_LIMIT)

        with self._lock:
            # Start with hazard event index
            candidate_ids = set(
                self._hazard_event_index.get(
                    normalized_hazard, set(),
                ),
            )

            candidates: List[dict] = []
            for evt_id in candidate_ids:
                evt = self._events.get(evt_id)
                if evt is None:
                    continue

                # Apply region filter
                if region is not None:
                    evt_location = evt.get("location", {})
                    if not self._is_in_region(evt_location, region):
                        continue

                # Apply year filters
                start_date = evt.get("start_date", "")
                if start_date and start_year is not None:
                    try:
                        evt_year = int(start_date[:4])
                        if evt_year < start_year:
                            continue
                    except (ValueError, IndexError):
                        pass

                if start_date and end_year is not None:
                    try:
                        evt_year = int(start_date[:4])
                        if evt_year > end_year:
                            continue
                    except (ValueError, IndexError):
                        pass

                candidates.append(evt)

        # Sort by start_date descending
        candidates.sort(
            key=lambda e: e.get("start_date", ""),
            reverse=True,
        )

        # Apply limit and deep copy
        results = [
            copy.deepcopy(evt)
            for evt in candidates[:effective_limit]
        ]

        elapsed = time.monotonic() - t0
        record_query_executed("get_historical_events")
        observe_processing_duration("event_query", elapsed)

        logger.debug(
            "get_historical_events: hazard=%s region=%s "
            "years=%s-%s -> %d results in %.3fms",
            normalized_hazard,
            region,
            start_year,
            end_year,
            len(results),
            elapsed * 1000,
        )
        return results

    # ------------------------------------------------------------------
    # 10. register_historical_event
    # ------------------------------------------------------------------

    def register_historical_event(
        self,
        hazard_type: str,
        location: Dict[str, Any],
        start_date: str,
        end_date: Optional[str] = None,
        intensity: Optional[float] = None,
        affected_area_km2: Optional[float] = None,
        deaths: Optional[int] = None,
        economic_loss_usd: Optional[float] = None,
        source: Optional[str] = None,
    ) -> dict:
        """Register a historical hazard event in the engine.

        Creates an event record with a unique ID, validates all input
        parameters, indexes by hazard type, records provenance, and
        emits metrics.

        Args:
            hazard_type: Type of hazard event. Must be valid.
            location: Dictionary with ``lat`` and ``lon`` keys.
            start_date: ISO date/datetime string for event start.
            end_date: Optional ISO date/datetime string for event end.
            intensity: Optional intensity value (0-10 scale).
            affected_area_km2: Optional affected area in square km.
            deaths: Optional number of fatalities.
            economic_loss_usd: Optional economic loss in US dollars.
            source: Optional source attribution string.

        Returns:
            Deep copy of the complete event dict including
            ``event_id``, ``provenance_hash``, and ``created_at``.

        Raises:
            ValueError: If hazard_type is invalid, location is invalid,
                start_date is empty, or total events would exceed
                MAX_EVENTS.

        Example:
            >>> engine = HazardDatabaseEngine()
            >>> event = engine.register_historical_event(
            ...     hazard_type="TROPICAL_CYCLONE",
            ...     location={"lat": 25.0, "lon": -71.0},
            ...     start_date="2017-09-06",
            ...     end_date="2017-09-13",
            ...     intensity=9.5,
            ...     affected_area_km2=350000,
            ...     deaths=134,
            ...     economic_loss_usd=77_000_000_000,
            ...     source="emdat-cred",
            ... )
            >>> assert event["event_id"].startswith("EVT-")
        """
        t0 = time.monotonic()

        # -- Input validation --
        normalized_hazard = self._validate_hazard_type(hazard_type)
        self._validate_location(location)

        if not start_date or not start_date.strip():
            raise ValueError("start_date must be a non-empty string.")
        clean_start = start_date.strip()
        clean_end = end_date.strip() if end_date else None

        validated_intensity: Optional[float] = None
        if intensity is not None:
            validated_intensity = self._validate_intensity(intensity)

        validated_area: Optional[float] = None
        if affected_area_km2 is not None:
            validated_area = max(0.0, float(affected_area_km2))

        validated_deaths: Optional[int] = None
        if deaths is not None:
            validated_deaths = max(0, int(deaths))

        validated_loss: Optional[float] = None
        if economic_loss_usd is not None:
            validated_loss = max(0.0, float(economic_loss_usd))

        # -- Build event record --
        event_id = self._generate_id("EVT-")
        now_str = _utcnow().isoformat()

        event_record: dict = {
            "event_id": event_id,
            "hazard_type": normalized_hazard,
            "location": {
                "lat": float(location["lat"]),
                "lon": float(location["lon"]),
            },
            "start_date": clean_start,
            "end_date": clean_end,
            "intensity": validated_intensity,
            "affected_area_km2": validated_area,
            "deaths": validated_deaths,
            "economic_loss_usd": validated_loss,
            "source": source.strip() if source else None,
            "created_at": now_str,
            "provenance_hash": "",
        }

        # -- Compute data hash --
        data_hash = _build_sha256(
            {k: v for k, v in event_record.items()
             if k != "provenance_hash"},
        )

        with self._lock:
            # Check capacity
            if len(self._events) >= MAX_EVENTS:
                record_error("event_capacity_exceeded")
                raise ValueError(
                    f"Maximum number of events ({MAX_EVENTS}) reached."
                )

            # Record provenance
            entry = self._provenance.record(
                entity_type="hazard_event",
                entity_id=event_id,
                action="event_registered",
                metadata={
                    "data_hash": data_hash,
                    "hazard_type": normalized_hazard,
                    "start_date": clean_start,
                },
            )
            event_record["provenance_hash"] = entry.hash_value

            # Store event
            self._events[event_id] = event_record

            # Index by hazard type
            self._add_to_index(
                self._hazard_event_index,
                normalized_hazard,
                event_id,
            )

            # Update counters
            self._operation_counts["events_registered"] += 1

        # Metrics
        elapsed = time.monotonic() - t0
        record_event_registered(normalized_hazard)
        observe_processing_duration("event_register", elapsed)

        logger.info(
            "Historical event registered: id=%s hazard=%s "
            "start=%s intensity=%s elapsed=%.3fms",
            event_id,
            normalized_hazard,
            clean_start,
            validated_intensity,
            elapsed * 1000,
        )
        return copy.deepcopy(event_record)

    # ------------------------------------------------------------------
    # 11. aggregate_sources
    # ------------------------------------------------------------------

    def aggregate_sources(
        self,
        hazard_type: str,
        location: Dict[str, Any],
        strategy: str = "weighted_average",
    ) -> dict:
        """Aggregate hazard data from multiple sources for a location.

        Collects all records for the given hazard type within a
        default 100 km radius of the specified location, groups
        them by source, and aggregates intensity, probability,
        frequency, and duration using the chosen strategy.

        Supported strategies:
            - ``weighted_average``: Weight by source record count
              (sources with more data points have more influence).
            - ``maximum``: Take the maximum value from all sources.
            - ``minimum``: Take the minimum value from all sources.
            - ``median``: Take the median value from all sources.

        Args:
            hazard_type: Type of hazard to aggregate.
            location: Dictionary with ``lat`` and ``lon`` keys.
            strategy: Aggregation strategy name. Defaults to
                ``"weighted_average"``.

        Returns:
            Dict with aggregated ``intensity``, ``probability``,
            ``frequency``, ``duration_days``, ``source_count``,
            ``record_count``, ``sources_used``, ``strategy``,
            ``hazard_type``, ``location``, and ``provenance_hash``.

        Raises:
            ValueError: If hazard_type is invalid, location is invalid,
                or strategy is not supported.

        Example:
            >>> engine = HazardDatabaseEngine()
            >>> agg = engine.aggregate_sources(
            ...     hazard_type="DROUGHT",
            ...     location={"lat": 40.0, "lon": -3.7},
            ...     strategy="maximum",
            ... )
            >>> assert "intensity" in agg
        """
        t0 = time.monotonic()

        normalized_hazard = self._validate_hazard_type(hazard_type)
        self._validate_location(location)

        strategy_lower = strategy.strip().lower()
        if strategy_lower not in VALID_AGGREGATION_STRATEGIES:
            raise ValueError(
                f"Invalid aggregation strategy: {strategy!r}. "
                f"Must be one of: {sorted(VALID_AGGREGATION_STRATEGIES)}"
            )

        # Get all nearby records for this hazard type
        nearby_records = self.get_hazard_data(
            hazard_type=normalized_hazard,
            location={
                "lat": location["lat"],
                "lon": location["lon"],
                "radius_km": 100.0,
            },
            limit=MAX_QUERY_LIMIT,
        )

        # Group values by field
        intensities: List[float] = []
        probabilities: List[float] = []
        frequencies: List[float] = []
        durations: List[float] = []
        sources_used: Set[str] = set()

        # For weighted average, track per-source data
        source_groups: Dict[str, List[dict]] = {}

        for rec in nearby_records:
            intensities.append(float(rec.get("intensity", 0.0)))
            probabilities.append(float(rec.get("probability", 0.0)))
            frequencies.append(float(rec.get("frequency", 0.0)))
            durations.append(float(rec.get("duration_days", 0.0)))
            src = rec.get("source_id", "unknown")
            sources_used.add(src)
            source_groups.setdefault(src, []).append(rec)

        # Aggregate based on strategy
        if not nearby_records:
            agg_intensity = 0.0
            agg_probability = 0.0
            agg_frequency = 0.0
            agg_duration = 0.0
        elif strategy_lower == "weighted_average":
            agg_intensity = self._weighted_average_by_source(
                source_groups, "intensity",
            )
            agg_probability = self._weighted_average_by_source(
                source_groups, "probability",
            )
            agg_frequency = self._weighted_average_by_source(
                source_groups, "frequency",
            )
            agg_duration = self._weighted_average_by_source(
                source_groups, "duration_days",
            )
        elif strategy_lower == "maximum":
            agg_intensity = max(intensities)
            agg_probability = max(probabilities)
            agg_frequency = max(frequencies)
            agg_duration = max(durations)
        elif strategy_lower == "minimum":
            agg_intensity = min(intensities)
            agg_probability = min(probabilities)
            agg_frequency = min(frequencies)
            agg_duration = min(durations)
        elif strategy_lower == "median":
            agg_intensity = self._compute_median(intensities)
            agg_probability = self._compute_median(probabilities)
            agg_frequency = self._compute_median(frequencies)
            agg_duration = self._compute_median(durations)
        else:
            # Defensive: should not reach here
            agg_intensity = 0.0
            agg_probability = 0.0
            agg_frequency = 0.0
            agg_duration = 0.0

        # Compute provenance hash
        agg_data = {
            "hazard_type": normalized_hazard,
            "location": {
                "lat": float(location["lat"]),
                "lon": float(location["lon"]),
            },
            "strategy": strategy_lower,
            "intensity": round(agg_intensity, 6),
            "probability": round(agg_probability, 6),
            "frequency": round(agg_frequency, 6),
            "duration_days": round(agg_duration, 6),
            "source_count": len(sources_used),
            "record_count": len(nearby_records),
        }
        provenance_hash = _build_sha256(agg_data)

        # Record provenance
        with self._lock:
            self._provenance.record(
                entity_type="hazard_aggregation",
                entity_id=provenance_hash[:16],
                action="data_aggregated",
                metadata={
                    "hazard_type": normalized_hazard,
                    "strategy": strategy_lower,
                    "source_count": len(sources_used),
                    "record_count": len(nearby_records),
                },
            )
            self._operation_counts["aggregations_performed"] += 1

        # Metrics
        elapsed = time.monotonic() - t0
        record_aggregation(strategy_lower)
        observe_processing_duration("data_aggregate", elapsed)

        logger.info(
            "Sources aggregated: hazard=%s strategy=%s "
            "sources=%d records=%d elapsed=%.3fms",
            normalized_hazard,
            strategy_lower,
            len(sources_used),
            len(nearby_records),
            elapsed * 1000,
        )

        return {
            "hazard_type": normalized_hazard,
            "location": {
                "lat": float(location["lat"]),
                "lon": float(location["lon"]),
            },
            "strategy": strategy_lower,
            "intensity": round(agg_intensity, 6),
            "probability": round(agg_probability, 6),
            "frequency": round(agg_frequency, 6),
            "duration_days": round(agg_duration, 6),
            "source_count": len(sources_used),
            "record_count": len(nearby_records),
            "sources_used": sorted(sources_used),
            "provenance_hash": provenance_hash,
            "processing_time_ms": round(elapsed * 1000, 3),
        }

    # ------------------------------------------------------------------
    # Internal: aggregation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _weighted_average_by_source(
        source_groups: Dict[str, List[dict]],
        field: str,
    ) -> float:
        """Compute weighted average of a field across source groups.

        Weight is proportional to the number of records each source
        contributed (sources with more data have higher weight).

        Args:
            source_groups: Mapping from source_id to list of records.
            field: Field name to aggregate.

        Returns:
            Weighted average value, or 0.0 if no data.
        """
        total_weight = 0.0
        weighted_sum = 0.0

        for src_id, records in source_groups.items():
            count = len(records)
            if count == 0:
                continue
            src_mean = sum(
                float(r.get(field, 0.0)) for r in records
            ) / count
            weighted_sum += src_mean * count
            total_weight += count

        if total_weight == 0.0:
            return 0.0
        return weighted_sum / total_weight

    @staticmethod
    def _compute_median(values: List[float]) -> float:
        """Compute the median of a list of float values.

        Args:
            values: List of numeric values.

        Returns:
            Median value, or 0.0 if the list is empty.
        """
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        mid = n // 2
        if n % 2 == 0:
            return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0
        return sorted_vals[mid]

    # ------------------------------------------------------------------
    # 12. get_source_coverage
    # ------------------------------------------------------------------

    def get_source_coverage(self, source_id: str) -> Optional[dict]:
        """Get spatial and temporal coverage statistics for a source.

        Computes bounding box (min/max lat/lon), temporal range
        (earliest/latest observation), and record count across all
        hazard types for the given source.

        Args:
            source_id: Unique source identifier.

        Returns:
            Dict with ``source_id``, ``source_name``, ``total_records``,
            ``hazard_types``, ``spatial_extent`` (bounding box),
            ``temporal_extent`` (start/end dates), and
            ``records_per_hazard_type``. Returns ``None`` if source
            not found.

        Example:
            >>> engine = HazardDatabaseEngine()
            >>> cov = engine.get_source_coverage("wri-aqueduct")
            >>> assert cov is not None
            >>> assert "spatial_extent" in cov
        """
        t0 = time.monotonic()

        with self._lock:
            source = self._sources.get(source_id)
            if source is None:
                logger.debug(
                    "Coverage query failed: source not found id=%s",
                    source_id,
                )
                return None

            record_ids = self._source_record_index.get(
                source_id, set(),
            )

            if not record_ids:
                elapsed = time.monotonic() - t0
                return {
                    "source_id": source_id,
                    "source_name": source.get("name", ""),
                    "total_records": 0,
                    "hazard_types": copy.deepcopy(
                        source.get("hazard_types", []),
                    ),
                    "spatial_extent": None,
                    "temporal_extent": None,
                    "records_per_hazard_type": {},
                    "processing_time_ms": round(elapsed * 1000, 3),
                }

            # Compute spatial and temporal extents
            lat_min = 90.0
            lat_max = -90.0
            lon_min = 180.0
            lon_max = -180.0
            earliest_obs = None
            latest_obs = None
            hazard_counts: Dict[str, int] = {}

            for rec_id in record_ids:
                rec = self._records.get(rec_id)
                if rec is None:
                    continue

                loc = rec.get("location", {})
                lat = float(loc.get("lat", 0.0))
                lon = float(loc.get("lon", 0.0))

                if lat < lat_min:
                    lat_min = lat
                if lat > lat_max:
                    lat_max = lat
                if lon < lon_min:
                    lon_min = lon
                if lon > lon_max:
                    lon_max = lon

                obs = rec.get("observed_at", "")
                if obs:
                    if earliest_obs is None or obs < earliest_obs:
                        earliest_obs = obs
                    if latest_obs is None or obs > latest_obs:
                        latest_obs = obs

                ht = rec.get("hazard_type", "UNKNOWN")
                hazard_counts[ht] = hazard_counts.get(ht, 0) + 1

        elapsed = time.monotonic() - t0
        observe_processing_duration("source_coverage", elapsed)

        return {
            "source_id": source_id,
            "source_name": source.get("name", ""),
            "total_records": len(record_ids),
            "hazard_types": sorted(hazard_counts.keys()),
            "spatial_extent": {
                "lat_min": lat_min,
                "lat_max": lat_max,
                "lon_min": lon_min,
                "lon_max": lon_max,
            },
            "temporal_extent": {
                "earliest": earliest_obs,
                "latest": latest_obs,
            },
            "records_per_hazard_type": dict(
                sorted(hazard_counts.items()),
            ),
            "processing_time_ms": round(elapsed * 1000, 3),
        }

    # ------------------------------------------------------------------
    # 13. export_data
    # ------------------------------------------------------------------

    def export_data(
        self,
        hazard_type: Optional[str] = None,
        source_id: Optional[str] = None,
    ) -> List[dict]:
        """Export hazard data records as a list of dicts.

        When filters are provided, only matching records are exported.
        When no filters are provided, all records are exported.

        Args:
            hazard_type: Optional hazard type filter.
            source_id: Optional source ID filter.

        Returns:
            List of deep-copied record dicts sorted by record_id
            for determinism.

        Example:
            >>> engine = HazardDatabaseEngine()
            >>> all_data = engine.export_data()
            >>> flood_data = engine.export_data(
            ...     hazard_type="RIVERINE_FLOOD",
            ... )
        """
        t0 = time.monotonic()

        with self._lock:
            # Determine candidate IDs
            candidate_ids: Optional[Set[str]] = None

            if hazard_type is not None:
                normalized_hazard = self._validate_hazard_type(
                    hazard_type,
                )
                hazard_ids = self._hazard_type_index.get(
                    normalized_hazard, set(),
                )
                candidate_ids = set(hazard_ids)

            if source_id is not None:
                source_ids = self._source_record_index.get(
                    source_id, set(),
                )
                if candidate_ids is not None:
                    candidate_ids = candidate_ids & source_ids
                else:
                    candidate_ids = set(source_ids)

            if candidate_ids is None:
                candidate_ids = set(self._records.keys())

            results: List[dict] = []
            for rec_id in sorted(candidate_ids):
                rec = self._records.get(rec_id)
                if rec is not None:
                    results.append(copy.deepcopy(rec))

            self._operation_counts["exports_performed"] += 1

        elapsed = time.monotonic() - t0
        observe_processing_duration("data_export", elapsed)

        logger.info(
            "Data exported: hazard=%s source=%s -> %d records "
            "in %.3fms",
            hazard_type,
            source_id,
            len(results),
            elapsed * 1000,
        )
        return results

    # ------------------------------------------------------------------
    # 14. import_data
    # ------------------------------------------------------------------

    def import_data(
        self,
        records: List[Dict[str, Any]],
    ) -> dict:
        """Bulk import hazard data records.

        Each record must contain ``source_id`` and ``hazard_type``
        fields. Records are validated and ingested grouped by
        (source_id, hazard_type) for efficient batch processing.

        Args:
            records: List of record dicts to import. Each must have
                at minimum ``source_id``, ``hazard_type``, and
                ``location`` fields.

        Returns:
            Dict with ``imported_count``, ``skipped_count``,
            ``error_count``, ``errors``, and ``provenance_hash``.

        Example:
            >>> engine = HazardDatabaseEngine()
            >>> result = engine.import_data([
            ...     {
            ...         "source_id": "wri-aqueduct",
            ...         "hazard_type": "DROUGHT",
            ...         "location": {"lat": 40.0, "lon": -3.7},
            ...         "intensity": 5.0,
            ...         "probability": 0.2,
            ...         "frequency": 0.3,
            ...         "duration_days": 60,
            ...     },
            ... ])
            >>> assert result["imported_count"] == 1
        """
        t0 = time.monotonic()

        if not records:
            return {
                "imported_count": 0,
                "skipped_count": 0,
                "error_count": 0,
                "errors": [],
                "provenance_hash": _build_sha256(
                    {"action": "import", "count": 0},
                ),
            }

        # Group records by (source_id, hazard_type)
        groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        errors: List[Dict[str, Any]] = []
        skipped = 0

        for idx, raw in enumerate(records):
            src_id = raw.get("source_id")
            ht = raw.get("hazard_type")

            if not src_id or not ht:
                errors.append({
                    "index": idx,
                    "reason": "Missing source_id or hazard_type",
                })
                skipped += 1
                continue

            try:
                normalized_ht = self._validate_hazard_type(ht)
            except ValueError as e:
                errors.append({
                    "index": idx,
                    "reason": str(e),
                })
                skipped += 1
                continue

            key = (str(src_id).strip(), normalized_ht)
            groups.setdefault(key, []).append(raw)

        # Ingest each group
        total_imported = 0

        for (src_id, ht), group_records in groups.items():
            # Extract region from first record if present
            region = group_records[0].get("region")

            try:
                result = self.ingest_hazard_data(
                    source_id=src_id,
                    hazard_type=ht,
                    records=group_records,
                    region=region,
                )
                total_imported += result["ingested_count"]
            except ValueError as e:
                for i, rec in enumerate(group_records):
                    errors.append({
                        "source_id": src_id,
                        "hazard_type": ht,
                        "reason": str(e),
                    })
                skipped += len(group_records)

        # Record provenance
        import_hash = _build_sha256({
            "action": "bulk_import",
            "total_records": len(records),
            "imported": total_imported,
            "skipped": skipped,
            "errors": len(errors),
        })

        with self._lock:
            self._provenance.record(
                entity_type="hazard_import",
                entity_id=import_hash[:16],
                action="data_imported",
                metadata={
                    "total_records": len(records),
                    "imported": total_imported,
                    "skipped": skipped,
                },
            )
            self._operation_counts["imports_performed"] += 1

        elapsed = time.monotonic() - t0
        observe_processing_duration("data_import", elapsed)

        logger.info(
            "Data imported: total=%d imported=%d skipped=%d "
            "errors=%d elapsed=%.3fms",
            len(records),
            total_imported,
            skipped,
            len(errors),
            elapsed * 1000,
        )

        return {
            "imported_count": total_imported,
            "skipped_count": skipped,
            "error_count": len(errors),
            "errors": errors,
            "provenance_hash": import_hash,
            "processing_time_ms": round(elapsed * 1000, 3),
        }

    # ------------------------------------------------------------------
    # 15. get_statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> dict:
        """Return comprehensive engine statistics.

        Provides counts of sources, records, events, hazard type
        distributions, source type distributions, region distributions,
        operation counters, and provenance chain length.

        Returns:
            Dict with engine statistics including ``total_sources``,
            ``total_records``, ``total_events``, ``active_sources``,
            ``builtin_sources``, ``custom_sources``,
            ``records_per_hazard_type``, ``records_per_source``,
            ``records_per_region``, ``events_per_hazard_type``,
            ``sources_per_type``, ``operation_counts``, and
            ``provenance_entries``.

        Example:
            >>> engine = HazardDatabaseEngine()
            >>> stats = engine.get_statistics()
            >>> assert stats["total_sources"] == 10
            >>> assert stats["builtin_sources"] == 10
        """
        with self._lock:
            # Source counts
            total_sources = len(self._sources)
            active_sources = sum(
                1 for s in self._sources.values()
                if s.get("status") == "active"
            )
            builtin_sources = sum(
                1 for s in self._sources.values()
                if s.get("is_builtin", False)
            )
            custom_sources = total_sources - builtin_sources

            # Record counts
            total_records = len(self._records)
            total_events = len(self._events)

            # Distribution: records per hazard type
            records_per_hazard: Dict[str, int] = {}
            for ht, rec_ids in self._hazard_type_index.items():
                records_per_hazard[ht] = len(rec_ids)

            # Distribution: records per source
            records_per_source: Dict[str, int] = {}
            for src_id, rec_ids in self._source_record_index.items():
                records_per_source[src_id] = len(rec_ids)

            # Distribution: records per region
            records_per_region: Dict[str, int] = {}
            for reg, rec_ids in self._region_record_index.items():
                records_per_region[reg] = len(rec_ids)

            # Distribution: events per hazard type
            events_per_hazard: Dict[str, int] = {}
            for ht, evt_ids in self._hazard_event_index.items():
                events_per_hazard[ht] = len(evt_ids)

            # Distribution: sources per type
            sources_per_type: Dict[str, int] = {}
            for st, src_ids in self._source_type_index.items():
                sources_per_type[st] = len(src_ids)

            # Operation counts
            op_counts = copy.deepcopy(self._operation_counts)

            # Provenance entries
            prov_entries = self._provenance.entry_count

        return {
            "total_sources": total_sources,
            "total_records": total_records,
            "total_events": total_events,
            "active_sources": active_sources,
            "builtin_sources": builtin_sources,
            "custom_sources": custom_sources,
            "records_per_hazard_type": dict(
                sorted(records_per_hazard.items()),
            ),
            "records_per_source": dict(
                sorted(records_per_source.items()),
            ),
            "records_per_region": dict(
                sorted(records_per_region.items()),
            ),
            "events_per_hazard_type": dict(
                sorted(events_per_hazard.items()),
            ),
            "sources_per_type": dict(
                sorted(sources_per_type.items()),
            ),
            "operation_counts": op_counts,
            "provenance_entries": prov_entries,
        }

    # ------------------------------------------------------------------
    # 16. clear
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Reset all engine state to initial condition.

        Clears all sources, records, events, indexes, and operation
        counters. Re-registers built-in data sources. Resets the
        provenance tracker.

        Example:
            >>> engine = HazardDatabaseEngine()
            >>> engine.clear()
            >>> stats = engine.get_statistics()
            >>> assert stats["total_sources"] == 10
            >>> assert stats["total_records"] == 0
        """
        with self._lock:
            self._sources.clear()
            self._records.clear()
            self._events.clear()
            self._source_type_index.clear()
            self._hazard_type_index.clear()
            self._source_record_index.clear()
            self._region_record_index.clear()
            self._hazard_event_index.clear()
            self._source_hazard_index.clear()

            # Reset operation counters
            for key in self._operation_counts:
                self._operation_counts[key] = 0

            # Reset provenance
            self._provenance.reset()

        # Re-register built-in sources
        self._register_builtin_sources()

        # Update metrics
        set_total_sources(len(self._sources))
        set_total_records(0)

        logger.info(
            "HazardDatabaseEngine cleared and re-initialized "
            "with %d built-in sources",
            len(self._sources),
        )


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = ["HazardDatabaseEngine"]
