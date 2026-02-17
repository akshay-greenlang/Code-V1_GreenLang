# -*- coding: utf-8 -*-
"""
Exposure Assessor Engine - AGENT-DATA-020

Engine 4 of 7 in the Climate Hazard Connector Agent SDK.  Performs
asset-level, portfolio-level, and supply-chain exposure analysis
against registered climate hazard types.

The engine maintains an in-memory asset registry, computes composite
exposure scores via a deterministic five-factor formula, classifies
results into five exposure levels, and supports spatial exposure-map
generation over bounding boxes.

Zero-Hallucination: All calculations use deterministic Python arithmetic
(math, hashlib, json).  No LLM calls for numeric computations.  Exposure
scores are produced by a fully transparent, auditable formula.

Exposure Score Formula (0-100):
    proximity_score = max(0, 1 - distance_km / max_radius_km)
    composite = (proximity_score * 0.25
                 + intensity_norm * 0.30
                 + frequency_norm * 0.25
                 + elevation_factor * 0.10
                 + population_factor * 0.10) * 100

Exposure Levels:
    NONE       0 -  10
    LOW       10 -  30
    MODERATE  30 -  55
    HIGH      55 -  80
    CRITICAL  80 - 100

Hazard-specific Max Radius (km):
    RIVERINE_FLOOD: 50, COASTAL_FLOOD: 30, DROUGHT: 500,
    EXTREME_HEAT: 200, EXTREME_COLD: 200, WILDFIRE: 100,
    TROPICAL_CYCLONE: 300, EXTREME_PRECIPITATION: 100,
    WATER_STRESS: 200, SEA_LEVEL_RISE: 20, LANDSLIDE: 10,
    COASTAL_EROSION: 5

8 Asset Types:
    FACILITY, SUPPLY_CHAIN_NODE, AGRICULTURAL_PLOT, INFRASTRUCTURE,
    REAL_ESTATE, NATURAL_ASSET, WATER_SOURCE, COASTAL_ASSET

Example:
    >>> from greenlang.climate_hazard.exposure_assessor import ExposureAssessorEngine
    >>> engine = ExposureAssessorEngine()
    >>> asset = engine.register_asset(
    ...     asset_id="A-001", name="Factory Alpha",
    ...     asset_type="FACILITY",
    ...     location={"latitude": 51.5, "longitude": -0.1},
    ... )
    >>> result = engine.assess_exposure(
    ...     asset_id="A-001", hazard_type="RIVERINE_FLOOD",
    ...     hazard_intensity=0.7, hazard_probability=0.6,
    ...     hazard_frequency=0.5,
    ... )
    >>> assert result["exposure_level"] in (
    ...     "NONE", "LOW", "MODERATE", "HIGH", "CRITICAL",
    ... )

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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple
from uuid import uuid4

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

__all__ = ["ExposureAssessorEngine"]


# ---------------------------------------------------------------------------
# Graceful provenance import
# ---------------------------------------------------------------------------

try:
    from greenlang.climate_hazard.provenance import ProvenanceTracker

    _PROVENANCE_MODULE_AVAILABLE = True
except Exception:  # pragma: no cover  # noqa: BLE001
    _PROVENANCE_MODULE_AVAILABLE = False
    ProvenanceTracker = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Graceful metrics import
# ---------------------------------------------------------------------------

try:
    from greenlang.climate_hazard.metrics import (  # type: ignore[import-untyped]
        record_exposure as _record_exposure_raw,
    )

    _METRICS_AVAILABLE = True
except Exception:  # pragma: no cover  # noqa: BLE001
    _METRICS_AVAILABLE = False
    _record_exposure_raw = None  # type: ignore[assignment]


def _safe_record_exposure(hazard_type: str, exposure_level: str) -> None:
    """Safely record an exposure assessment metric.

    Args:
        hazard_type: The hazard type assessed.
        exposure_level: The computed exposure level.
    """
    if _METRICS_AVAILABLE and _record_exposure_raw is not None:
        try:
            _record_exposure_raw(hazard_type, exposure_level)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Graceful config import
# ---------------------------------------------------------------------------

try:
    from greenlang.climate_hazard.config import get_config as _get_config

    _CONFIG_AVAILABLE = True
except Exception:  # pragma: no cover  # noqa: BLE001
    _CONFIG_AVAILABLE = False
    _get_config = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class AssetType(str, Enum):
    """Supported physical and financial asset types."""

    FACILITY = "FACILITY"
    SUPPLY_CHAIN_NODE = "SUPPLY_CHAIN_NODE"
    AGRICULTURAL_PLOT = "AGRICULTURAL_PLOT"
    INFRASTRUCTURE = "INFRASTRUCTURE"
    REAL_ESTATE = "REAL_ESTATE"
    NATURAL_ASSET = "NATURAL_ASSET"
    WATER_SOURCE = "WATER_SOURCE"
    COASTAL_ASSET = "COASTAL_ASSET"


class HazardType(str, Enum):
    """Climate hazard types with proximity decay radii."""

    RIVERINE_FLOOD = "RIVERINE_FLOOD"
    COASTAL_FLOOD = "COASTAL_FLOOD"
    DROUGHT = "DROUGHT"
    EXTREME_HEAT = "EXTREME_HEAT"
    EXTREME_COLD = "EXTREME_COLD"
    WILDFIRE = "WILDFIRE"
    TROPICAL_CYCLONE = "TROPICAL_CYCLONE"
    EXTREME_PRECIPITATION = "EXTREME_PRECIPITATION"
    WATER_STRESS = "WATER_STRESS"
    SEA_LEVEL_RISE = "SEA_LEVEL_RISE"
    LANDSLIDE = "LANDSLIDE"
    COASTAL_EROSION = "COASTAL_EROSION"


class ExposureLevel(str, Enum):
    """Exposure severity classification (5 levels)."""

    NONE = "NONE"
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Valid asset type strings (for quick membership testing)
VALID_ASSET_TYPES: frozenset[str] = frozenset(
    member.value for member in AssetType
)

# Valid hazard type strings
VALID_HAZARD_TYPES: frozenset[str] = frozenset(
    member.value for member in HazardType
)

# Valid exposure level strings
VALID_EXPOSURE_LEVELS: frozenset[str] = frozenset(
    member.value for member in ExposureLevel
)

# Hazard-specific max radius (km) for proximity decay calculation
HAZARD_MAX_RADIUS_KM: Dict[str, float] = {
    HazardType.RIVERINE_FLOOD.value: 50.0,
    HazardType.COASTAL_FLOOD.value: 30.0,
    HazardType.DROUGHT.value: 500.0,
    HazardType.EXTREME_HEAT.value: 200.0,
    HazardType.EXTREME_COLD.value: 200.0,
    HazardType.WILDFIRE.value: 100.0,
    HazardType.TROPICAL_CYCLONE.value: 300.0,
    HazardType.EXTREME_PRECIPITATION.value: 100.0,
    HazardType.WATER_STRESS.value: 200.0,
    HazardType.SEA_LEVEL_RISE.value: 20.0,
    HazardType.LANDSLIDE.value: 10.0,
    HazardType.COASTAL_EROSION.value: 5.0,
}

# Exposure level thresholds (inclusive lower, exclusive upper)
EXPOSURE_THRESHOLDS: List[Tuple[float, float, str]] = [
    (0.0, 10.0, ExposureLevel.NONE.value),
    (10.0, 30.0, ExposureLevel.LOW.value),
    (30.0, 55.0, ExposureLevel.MODERATE.value),
    (55.0, 80.0, ExposureLevel.HIGH.value),
    (80.0, 100.01, ExposureLevel.CRITICAL.value),
]

# Composite score weights
WEIGHT_PROXIMITY: float = 0.25
WEIGHT_INTENSITY: float = 0.30
WEIGHT_FREQUENCY: float = 0.25
WEIGHT_ELEVATION: float = 0.10
WEIGHT_POPULATION: float = 0.10

# Earth radius for Haversine calculation (km)
EARTH_RADIUS_KM: float = 6371.0

# Default elevation thresholds for elevation factor calculation (metres)
DEFAULT_ELEVATION_SEA_LEVEL_M: float = 0.0
DEFAULT_ELEVATION_HIGH_M: float = 2000.0

# Default population density thresholds (people per km2)
DEFAULT_POP_LOW: float = 0.0
DEFAULT_POP_HIGH: float = 10000.0


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class AssetRecord:
    """Internal representation of a registered asset.

    Attributes:
        asset_id: Unique identifier for the asset.
        name: Human-readable name of the asset.
        asset_type: One of the 8 supported asset types.
        location: Location dict with latitude, longitude, elevation_m.
        sector: Optional industry or business sector.
        value_usd: Optional asset monetary value in USD.
        metadata: Additional key-value metadata.
        registered_at: ISO-8601 UTC timestamp of registration.
        updated_at: ISO-8601 UTC timestamp of last update.
        provenance_hash: SHA-256 hash computed at registration.
    """

    asset_id: str = ""
    name: str = ""
    asset_type: str = ""
    location: Dict[str, Any] = field(default_factory=dict)
    sector: Optional[str] = None
    value_usd: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    registered_at: str = ""
    updated_at: str = ""
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the asset record to a plain dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        return {
            "asset_id": self.asset_id,
            "name": self.name,
            "asset_type": self.asset_type,
            "location": dict(self.location),
            "sector": self.sector,
            "value_usd": self.value_usd,
            "metadata": dict(self.metadata),
            "registered_at": self.registered_at,
            "updated_at": self.updated_at,
            "provenance_hash": self.provenance_hash,
        }


@dataclass
class ExposureAssessment:
    """Result of a single exposure assessment.

    Attributes:
        assessment_id: Unique identifier (EA-<hex12>).
        asset_id: The assessed asset's identifier.
        hazard_type: The climate hazard type assessed.
        exposure_level: Classified exposure level (NONE-CRITICAL).
        proximity_score: Proximity decay score [0, 1].
        intensity_at_location: Normalised hazard intensity [0, 1].
        frequency_exposure: Normalised frequency factor [0, 1].
        elevation_factor: Normalised elevation factor [0, 1].
        population_factor: Normalised population factor [0, 1].
        composite_score: Final weighted composite score [0, 100].
        scenario: Optional climate scenario label (SSP/RCP).
        time_horizon: Optional time horizon label.
        assessed_at: ISO-8601 UTC timestamp of assessment.
        provenance_hash: SHA-256 hash of the assessment.
    """

    assessment_id: str = ""
    asset_id: str = ""
    hazard_type: str = ""
    exposure_level: str = ExposureLevel.NONE.value
    proximity_score: float = 0.0
    intensity_at_location: float = 0.0
    frequency_exposure: float = 0.0
    elevation_factor: float = 0.0
    population_factor: float = 0.0
    composite_score: float = 0.0
    scenario: Optional[str] = None
    time_horizon: Optional[str] = None
    assessed_at: str = ""
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the assessment to a plain dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        return {
            "assessment_id": self.assessment_id,
            "asset_id": self.asset_id,
            "hazard_type": self.hazard_type,
            "exposure_level": self.exposure_level,
            "proximity_score": self.proximity_score,
            "intensity_at_location": self.intensity_at_location,
            "frequency_exposure": self.frequency_exposure,
            "elevation_factor": self.elevation_factor,
            "population_factor": self.population_factor,
            "composite_score": self.composite_score,
            "scenario": self.scenario,
            "time_horizon": self.time_horizon,
            "assessed_at": self.assessed_at,
            "provenance_hash": self.provenance_hash,
        }


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _build_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Sorts dictionary keys and serializes to JSON for reproducibility.

    Args:
        data: Data to hash (dict, list, str, numeric, or other).

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _generate_assessment_id() -> str:
    """Generate a unique exposure assessment ID.

    Returns:
        Assessment ID with EA- prefix and 12-character hex suffix.
    """
    return f"EA-{uuid4().hex[:12]}"


def _generate_asset_id() -> str:
    """Generate a unique asset ID (used when callers supply empty IDs).

    Returns:
        Asset ID with AST- prefix and 12-character hex suffix.
    """
    return f"AST-{uuid4().hex[:12]}"


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp a value to the range [lo, hi].

    Args:
        value: Numeric value to clamp.
        lo: Lower bound.
        hi: Upper bound.

    Returns:
        Clamped value.
    """
    return max(lo, min(hi, value))


def _haversine_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """Compute the great-circle distance between two points using Haversine.

    Uses deterministic Python ``math`` library only.

    Args:
        lat1: Latitude of point 1 in decimal degrees.
        lon1: Longitude of point 1 in decimal degrees.
        lat2: Latitude of point 2 in decimal degrees.
        lon2: Longitude of point 2 in decimal degrees.

    Returns:
        Distance in kilometres.
    """
    r_lat1 = math.radians(lat1)
    r_lat2 = math.radians(lat2)
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)

    a = (
        math.sin(d_lat / 2.0) ** 2
        + math.cos(r_lat1) * math.cos(r_lat2) * math.sin(d_lon / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return EARTH_RADIUS_KM * c


# ===========================================================================
# ExposureAssessorEngine
# ===========================================================================


class ExposureAssessorEngine:
    """Asset and supply-chain exposure assessment engine.

    Manages an in-memory asset registry and performs deterministic
    exposure scoring against 12 climate hazard types.  The engine
    supports single-asset assessment, portfolio-level batching,
    supply-chain tier mapping, hotspot identification, and spatial
    exposure-map generation.

    All arithmetic is deterministic Python (zero-hallucination).
    Every operation produces a SHA-256 provenance hash for audit trail
    tracking.  Thread safety is provided via ``threading.Lock``.

    Attributes:
        _risk_engine: Optional reference to a RiskIndexEngine for
            cross-engine queries.
        _provenance: ProvenanceTracker for SHA-256 chain hashing.
        _lock: Thread-safety lock for concurrent access.
        _assets: In-memory asset registry keyed by asset_id.
        _assessments: In-memory assessment registry keyed by
            assessment_id.
        _assessment_count: Running count of assessments performed.
        _asset_count: Running count of assets registered.

    Example:
        >>> engine = ExposureAssessorEngine()
        >>> asset = engine.register_asset(
        ...     asset_id="A-001", name="Factory Alpha",
        ...     asset_type="FACILITY",
        ...     location={"latitude": 51.5, "longitude": -0.1},
        ... )
        >>> result = engine.assess_exposure(
        ...     asset_id="A-001",
        ...     hazard_type="RIVERINE_FLOOD",
        ...     hazard_intensity=0.7,
        ...     hazard_probability=0.6,
        ...     hazard_frequency=0.5,
        ... )
        >>> assert result["exposure_level"] in (
        ...     "NONE", "LOW", "MODERATE", "HIGH", "CRITICAL",
        ... )
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        risk_engine: Any = None,
        provenance: Any = None,
        genesis_hash: Optional[str] = None,
    ) -> None:
        """Initialize ExposureAssessorEngine.

        Args:
            risk_engine: Optional RiskIndexEngine instance for cross-
                engine queries.  When None the engine operates
                standalone.
            provenance: Optional pre-built ProvenanceTracker instance.
                When None the engine creates its own tracker (if the
                provenance module is importable).
            genesis_hash: Optional genesis anchor string for the
                provenance chain.  Ignored when *provenance* is
                supplied.
        """
        self._risk_engine: Any = risk_engine

        # --- Provenance setup ------------------------------------------------
        if provenance is not None:
            self._provenance: Any = provenance
        elif _PROVENANCE_MODULE_AVAILABLE and ProvenanceTracker is not None:
            gen = genesis_hash or "greenlang-climate-hazard-exposure-genesis"
            self._provenance = ProvenanceTracker(gen)
        else:
            self._provenance = None

        # --- Thread safety ----------------------------------------------------
        self._lock: threading.Lock = threading.Lock()

        # --- In-memory stores -------------------------------------------------
        self._assets: Dict[str, AssetRecord] = {}
        self._assessments: Dict[str, ExposureAssessment] = {}

        # --- Counters ---------------------------------------------------------
        self._asset_count: int = 0
        self._assessment_count: int = 0
        self._portfolio_count: int = 0
        self._supply_chain_count: int = 0
        self._hotspot_count: int = 0
        self._exposure_map_count: int = 0
        self._error_count: int = 0

        logger.info(
            "ExposureAssessorEngine initialized "
            "(risk_engine=%s, provenance=%s)",
            "attached" if risk_engine is not None else "standalone",
            "attached" if self._provenance is not None else "disabled",
        )

    # ==================================================================
    # 1. register_asset
    # ==================================================================

    def register_asset(
        self,
        asset_id: str,
        name: str,
        asset_type: str,
        location: Dict[str, Any],
        sector: Optional[str] = None,
        value_usd: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Register a physical or financial asset for exposure analysis.

        Validates all inputs, creates an :class:`AssetRecord`, records
        provenance, and stores the asset in the in-memory registry.

        Args:
            asset_id: Unique identifier for the asset.  Must not be
                empty.
            name: Human-readable name of the asset.  Must not be empty.
            asset_type: One of the 8 supported asset types (FACILITY,
                SUPPLY_CHAIN_NODE, AGRICULTURAL_PLOT, INFRASTRUCTURE,
                REAL_ESTATE, NATURAL_ASSET, WATER_SOURCE, COASTAL_ASSET).
            location: Dict with ``latitude`` (float, -90 to 90) and
                ``longitude`` (float, -180 to 180).  Optional key
                ``elevation_m`` (float).
            sector: Optional industry or business sector string.
            value_usd: Optional monetary value of the asset in USD.
                Must be non-negative when provided.
            metadata: Optional dict of additional key-value metadata.

        Returns:
            Deep-copied dictionary of the registered asset record.

        Raises:
            ValueError: If any required parameter is missing or invalid.
        """
        start_time = time.monotonic()

        # --- Validation ------------------------------------------------------
        self._validate_asset_id(asset_id)
        self._validate_name(name)
        self._validate_asset_type(asset_type)
        self._validate_location(location)
        if value_usd is not None and value_usd < 0:
            raise ValueError(
                f"value_usd must be non-negative, got {value_usd}"
            )

        now_iso = _utcnow().isoformat()
        safe_metadata = dict(metadata) if metadata else {}

        record = AssetRecord(
            asset_id=asset_id,
            name=name,
            asset_type=asset_type.upper(),
            location=self._normalise_location(location),
            sector=sector,
            value_usd=value_usd,
            metadata=safe_metadata,
            registered_at=now_iso,
            updated_at=now_iso,
            provenance_hash="",
        )

        # --- Provenance hash --------------------------------------------------
        provenance_hash = self._compute_provenance(
            operation="register_asset",
            input_data=record.to_dict(),
            output_data={"asset_id": asset_id},
        )
        record.provenance_hash = provenance_hash

        # --- Store ------------------------------------------------------------
        with self._lock:
            self._assets[asset_id] = record
            self._asset_count += 1

        self._record_provenance_entry(
            entity_type="asset",
            action="register_asset",
            entity_id=asset_id,
            data=record.to_dict(),
        )

        duration = time.monotonic() - start_time
        logger.info(
            "Asset registered: asset_id=%s, type=%s, sector=%s "
            "(%.3fs)",
            asset_id,
            asset_type,
            sector or "N/A",
            duration,
        )

        return copy.deepcopy(record.to_dict())

    # ==================================================================
    # 2. get_asset
    # ==================================================================

    def get_asset(self, asset_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a registered asset by its identifier.

        Args:
            asset_id: The unique asset identifier to look up.

        Returns:
            Deep-copied dictionary of the asset record, or ``None`` if
            the asset does not exist.
        """
        if not asset_id:
            return None

        with self._lock:
            record = self._assets.get(asset_id)

        if record is None:
            return None

        return copy.deepcopy(record.to_dict())

    # ==================================================================
    # 3. list_assets
    # ==================================================================

    def list_assets(
        self,
        asset_type: Optional[str] = None,
        sector: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List registered assets with optional filters.

        Args:
            asset_type: Optional filter by asset type string.
            sector: Optional filter by sector string.
            limit: Maximum number of assets to return.  Defaults to 100.

        Returns:
            List of deep-copied asset record dictionaries, ordered by
            registration time (newest first), limited to *limit*.
        """
        with self._lock:
            records = list(self._assets.values())

        # Apply filters
        if asset_type is not None:
            upper_type = asset_type.upper()
            records = [r for r in records if r.asset_type == upper_type]

        if sector is not None:
            records = [r for r in records if r.sector == sector]

        # Newest first
        records.sort(key=lambda r: r.registered_at, reverse=True)

        # Apply limit
        safe_limit = max(1, limit) if limit else 100
        records = records[:safe_limit]

        return [copy.deepcopy(r.to_dict()) for r in records]

    # ==================================================================
    # 4. update_asset
    # ==================================================================

    def update_asset(
        self,
        asset_id: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Update one or more fields of a registered asset.

        Supported fields: ``name``, ``asset_type``, ``location``,
        ``sector``, ``value_usd``, ``metadata``.  Attempting to
        change the ``asset_id`` is not supported.

        Args:
            asset_id: Identifier of the asset to update.
            **kwargs: Fields to update.

        Returns:
            Deep-copied dictionary of the updated asset record.

        Raises:
            ValueError: If asset_id is empty, the asset does not exist,
                or any supplied value fails validation.
        """
        start_time = time.monotonic()

        if not asset_id or not asset_id.strip():
            raise ValueError("asset_id must not be empty")

        with self._lock:
            record = self._assets.get(asset_id)

        if record is None:
            raise ValueError(f"Asset not found: {asset_id}")

        # --- Apply updates ---------------------------------------------------
        updatable_fields = {
            "name", "asset_type", "location", "sector",
            "value_usd", "metadata",
        }
        applied: Dict[str, Any] = {}

        for key, value in kwargs.items():
            if key not in updatable_fields:
                logger.warning(
                    "Ignoring non-updatable field '%s' for asset %s",
                    key, asset_id,
                )
                continue

            if key == "name":
                self._validate_name(value)
                record.name = value
                applied[key] = value

            elif key == "asset_type":
                self._validate_asset_type(value)
                record.asset_type = value.upper()
                applied[key] = value.upper()

            elif key == "location":
                self._validate_location(value)
                record.location = self._normalise_location(value)
                applied[key] = record.location

            elif key == "sector":
                record.sector = value
                applied[key] = value

            elif key == "value_usd":
                if value is not None and value < 0:
                    raise ValueError(
                        f"value_usd must be non-negative, got {value}"
                    )
                record.value_usd = value
                applied[key] = value

            elif key == "metadata":
                record.metadata = dict(value) if value else {}
                applied[key] = record.metadata

        now_iso = _utcnow().isoformat()
        record.updated_at = now_iso

        # --- Provenance -------------------------------------------------------
        provenance_hash = self._compute_provenance(
            operation="update_asset",
            input_data={"asset_id": asset_id, "updates": applied},
            output_data=record.to_dict(),
        )
        record.provenance_hash = provenance_hash

        with self._lock:
            self._assets[asset_id] = record

        self._record_provenance_entry(
            entity_type="asset",
            action="update_asset",
            entity_id=asset_id,
            data={"updates": applied},
        )

        duration = time.monotonic() - start_time
        logger.info(
            "Asset updated: asset_id=%s, fields=%s (%.3fs)",
            asset_id,
            list(applied.keys()),
            duration,
        )

        return copy.deepcopy(record.to_dict())

    # ==================================================================
    # 5. delete_asset
    # ==================================================================

    def delete_asset(self, asset_id: str) -> bool:
        """Delete a registered asset from the registry.

        Also removes any assessments associated with the asset.

        Args:
            asset_id: Identifier of the asset to delete.

        Returns:
            ``True`` if the asset was found and deleted, ``False``
            otherwise.
        """
        if not asset_id:
            return False

        with self._lock:
            record = self._assets.pop(asset_id, None)
            if record is None:
                return False

            # Remove associated assessments
            to_remove = [
                aid for aid, assess in self._assessments.items()
                if assess.asset_id == asset_id
            ]
            for aid in to_remove:
                del self._assessments[aid]

        self._record_provenance_entry(
            entity_type="asset",
            action="delete_asset",
            entity_id=asset_id,
            data={"deleted_assessments": len(to_remove)},
        )

        logger.info(
            "Asset deleted: asset_id=%s (removed %d assessments)",
            asset_id,
            len(to_remove),
        )
        return True

    # ==================================================================
    # 6. assess_exposure
    # ==================================================================

    def assess_exposure(
        self,
        asset_id: str,
        hazard_type: str,
        hazard_intensity: float,
        hazard_probability: float,
        hazard_frequency: float,
        scenario: Optional[str] = None,
        time_horizon: Optional[str] = None,
        distance_km: Optional[float] = None,
        elevation_factor: Optional[float] = None,
        population_factor: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Assess a single asset's exposure to a specific climate hazard.

        Computes the composite exposure score using the five-factor
        weighted formula and classifies the result into one of five
        exposure levels.

        Formula:
            proximity_score = max(0, 1 - distance_km / max_radius_km)
            composite = (proximity_score * 0.25
                         + intensity_norm * 0.30
                         + frequency_norm * 0.25
                         + elevation_factor * 0.10
                         + population_factor * 0.10) * 100

        Args:
            asset_id: Identifier of the asset to assess.  Must be
                registered.
            hazard_type: One of the 12 supported hazard types.
            hazard_intensity: Hazard intensity normalised to [0, 1].
            hazard_probability: Hazard probability normalised to [0, 1].
                Used as a modifier on the composite score but not a
                direct weight factor in the five-factor formula.
            hazard_frequency: Hazard frequency normalised to [0, 1].
            scenario: Optional climate scenario label (e.g. SSP2-4.5).
            time_horizon: Optional time horizon label (SHORT_TERM,
                MID_TERM, LONG_TERM).
            distance_km: Optional pre-computed distance in km from asset
                to hazard centroid.  When None the proximity score is
                assumed to be 1.0 (asset at hazard epicentre).
            elevation_factor: Optional pre-computed elevation factor
                [0, 1].  When None the engine derives it from the
                asset's ``elevation_m`` if available, else 0.5.
            population_factor: Optional pre-computed population density
                factor [0, 1].  When None defaults to 0.5.

        Returns:
            Deep-copied assessment result dictionary with keys:
            assessment_id, asset_id, hazard_type, exposure_level,
            proximity_score, intensity_at_location, frequency_exposure,
            elevation_factor, population_factor, composite_score,
            scenario, time_horizon, assessed_at, provenance_hash.

        Raises:
            ValueError: If asset_id is empty, asset is not registered,
                hazard_type is unsupported, or numeric inputs are out
                of range.
        """
        start_time = time.monotonic()

        # --- Validation ------------------------------------------------------
        self._validate_asset_id(asset_id)
        self._validate_hazard_type(hazard_type)

        with self._lock:
            record = self._assets.get(asset_id)

        if record is None:
            raise ValueError(
                f"Asset not found: {asset_id}. "
                "Register the asset before assessing exposure."
            )

        hazard_upper = hazard_type.upper()
        intensity_norm = _clamp(float(hazard_intensity), 0.0, 1.0)
        probability_norm = _clamp(float(hazard_probability), 0.0, 1.0)
        frequency_norm = _clamp(float(hazard_frequency), 0.0, 1.0)

        # --- Proximity score --------------------------------------------------
        max_radius = HAZARD_MAX_RADIUS_KM.get(hazard_upper, 100.0)
        if distance_km is not None:
            dist = max(0.0, float(distance_km))
            prox_score = max(0.0, 1.0 - dist / max_radius)
        else:
            # No distance supplied: assume asset is at hazard epicentre
            prox_score = 1.0

        # --- Elevation factor -------------------------------------------------
        if elevation_factor is not None:
            elev_factor = _clamp(float(elevation_factor), 0.0, 1.0)
        else:
            elev_factor = self._derive_elevation_factor(record, hazard_upper)

        # --- Population factor ------------------------------------------------
        if population_factor is not None:
            pop_factor = _clamp(float(population_factor), 0.0, 1.0)
        else:
            pop_factor = self._derive_population_factor(record)

        # --- Composite score --------------------------------------------------
        raw_composite = (
            prox_score * WEIGHT_PROXIMITY
            + intensity_norm * WEIGHT_INTENSITY
            + frequency_norm * WEIGHT_FREQUENCY
            + elev_factor * WEIGHT_ELEVATION
            + pop_factor * WEIGHT_POPULATION
        ) * 100.0

        # Apply probability as a scaling modifier
        composite = raw_composite * probability_norm
        composite = _clamp(composite, 0.0, 100.0)

        # --- Classify ---------------------------------------------------------
        exposure_level = self._classify_exposure_level(composite)

        # --- Build assessment record ------------------------------------------
        now_iso = _utcnow().isoformat()
        assessment_id = _generate_assessment_id()

        assessment = ExposureAssessment(
            assessment_id=assessment_id,
            asset_id=asset_id,
            hazard_type=hazard_upper,
            exposure_level=exposure_level,
            proximity_score=round(prox_score, 6),
            intensity_at_location=round(intensity_norm, 6),
            frequency_exposure=round(frequency_norm, 6),
            elevation_factor=round(elev_factor, 6),
            population_factor=round(pop_factor, 6),
            composite_score=round(composite, 4),
            scenario=scenario,
            time_horizon=time_horizon,
            assessed_at=now_iso,
            provenance_hash="",
        )

        # --- Provenance -------------------------------------------------------
        provenance_hash = self._compute_provenance(
            operation="assess_exposure",
            input_data={
                "asset_id": asset_id,
                "hazard_type": hazard_upper,
                "hazard_intensity": hazard_intensity,
                "hazard_probability": hazard_probability,
                "hazard_frequency": hazard_frequency,
                "distance_km": distance_km,
            },
            output_data={
                "assessment_id": assessment_id,
                "composite_score": composite,
                "exposure_level": exposure_level,
            },
        )
        assessment.provenance_hash = provenance_hash

        # --- Store ------------------------------------------------------------
        with self._lock:
            self._assessments[assessment_id] = assessment
            self._assessment_count += 1

        self._record_provenance_entry(
            entity_type="exposure",
            action="assess_exposure",
            entity_id=assessment_id,
            data=assessment.to_dict(),
        )

        # --- Metrics ----------------------------------------------------------
        _safe_record_exposure(hazard_upper, exposure_level)

        duration = time.monotonic() - start_time
        logger.info(
            "Exposure assessed: asset=%s, hazard=%s, score=%.2f, "
            "level=%s (%.3fs)",
            asset_id,
            hazard_upper,
            composite,
            exposure_level,
            duration,
        )

        return copy.deepcopy(assessment.to_dict())

    # ==================================================================
    # 7. assess_portfolio_exposure
    # ==================================================================

    def assess_portfolio_exposure(
        self,
        asset_ids: List[str],
        hazard_types: List[str],
        hazard_data: Dict[str, Dict[str, float]],
        scenario: Optional[str] = None,
        time_horizon: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Assess exposure for a portfolio of assets against multiple hazards.

        Iterates over every (asset, hazard) pair, calls
        :meth:`assess_exposure` for each, and aggregates the results
        into a portfolio summary with average, maximum, and minimum
        composite scores plus an exposure-level distribution breakdown.

        Args:
            asset_ids: List of registered asset identifiers.
            hazard_types: List of hazard types to assess.
            hazard_data: Mapping of hazard type string to a dict
                containing ``intensity``, ``probability``, and
                ``frequency`` keys (all floats in [0, 1]).
            scenario: Optional climate scenario label passed through
                to each individual assessment.
            time_horizon: Optional time horizon label passed through
                to each individual assessment.

        Returns:
            Dictionary with keys:
                ``per_asset_results`` - list of per-asset result dicts,
                    each containing ``asset_id`` and ``hazard_results``
                    (list of assessment dicts).
                ``portfolio_summary`` - dict with ``avg_score``,
                    ``max_score``, ``min_score``, ``total_assessments``,
                    ``exposure_distribution`` (count per level).
                ``hotspot_count`` - number of assets with at least one
                    HIGH or CRITICAL assessment.
                ``scenario`` - scenario label.
                ``time_horizon`` - time horizon label.
                ``assessed_at`` - ISO-8601 UTC timestamp.
                ``provenance_hash`` - SHA-256 provenance hash.

        Raises:
            ValueError: If asset_ids or hazard_types are empty.
        """
        start_time = time.monotonic()

        if not asset_ids:
            raise ValueError("asset_ids must not be empty")
        if not hazard_types:
            raise ValueError("hazard_types must not be empty")

        per_asset_results: List[Dict[str, Any]] = []
        all_scores: List[float] = []
        exposure_dist: Dict[str, int] = {
            level.value: 0 for level in ExposureLevel
        }
        hotspot_asset_ids: set[str] = set()

        for aid in asset_ids:
            asset_hazard_results: List[Dict[str, Any]] = []

            for ht in hazard_types:
                ht_upper = ht.upper()
                hdata = hazard_data.get(ht_upper, hazard_data.get(ht, {}))

                intensity = float(hdata.get("intensity", 0.0))
                probability = float(hdata.get("probability", 0.0))
                frequency = float(hdata.get("frequency", 0.0))
                dist_km = hdata.get("distance_km")
                elev_f = hdata.get("elevation_factor")
                pop_f = hdata.get("population_factor")

                try:
                    result = self.assess_exposure(
                        asset_id=aid,
                        hazard_type=ht_upper,
                        hazard_intensity=intensity,
                        hazard_probability=probability,
                        hazard_frequency=frequency,
                        scenario=scenario,
                        time_horizon=time_horizon,
                        distance_km=dist_km,
                        elevation_factor=elev_f,
                        population_factor=pop_f,
                    )
                    asset_hazard_results.append(result)
                    score = float(result.get("composite_score", 0.0))
                    all_scores.append(score)

                    level = result.get("exposure_level", ExposureLevel.NONE.value)
                    if level in exposure_dist:
                        exposure_dist[level] += 1

                    if level in (
                        ExposureLevel.HIGH.value,
                        ExposureLevel.CRITICAL.value,
                    ):
                        hotspot_asset_ids.add(aid)

                except (ValueError, KeyError) as exc:
                    logger.warning(
                        "Portfolio assessment skipped for asset=%s "
                        "hazard=%s: %s",
                        aid, ht_upper, str(exc),
                    )
                    self._error_count += 1

            per_asset_results.append({
                "asset_id": aid,
                "hazard_results": asset_hazard_results,
            })

        # --- Portfolio summary ------------------------------------------------
        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
        max_score = max(all_scores) if all_scores else 0.0
        min_score = min(all_scores) if all_scores else 0.0

        portfolio_summary: Dict[str, Any] = {
            "avg_score": round(avg_score, 4),
            "max_score": round(max_score, 4),
            "min_score": round(min_score, 4),
            "total_assessments": len(all_scores),
            "exposure_distribution": dict(exposure_dist),
        }

        now_iso = _utcnow().isoformat()

        provenance_hash = self._compute_provenance(
            operation="assess_portfolio_exposure",
            input_data={
                "asset_count": len(asset_ids),
                "hazard_count": len(hazard_types),
                "scenario": scenario,
            },
            output_data=portfolio_summary,
        )

        with self._lock:
            self._portfolio_count += 1

        self._record_provenance_entry(
            entity_type="exposure",
            action="assess_portfolio",
            entity_id=f"portfolio-{uuid4().hex[:8]}",
            data=portfolio_summary,
        )

        duration = time.monotonic() - start_time
        logger.info(
            "Portfolio exposure assessed: %d assets x %d hazards = "
            "%d assessments, avg=%.2f, max=%.2f (%.3fs)",
            len(asset_ids),
            len(hazard_types),
            len(all_scores),
            avg_score,
            max_score,
            duration,
        )

        return copy.deepcopy({
            "per_asset_results": per_asset_results,
            "portfolio_summary": portfolio_summary,
            "hotspot_count": len(hotspot_asset_ids),
            "scenario": scenario,
            "time_horizon": time_horizon,
            "assessed_at": now_iso,
            "provenance_hash": provenance_hash,
        })

    # ==================================================================
    # 8. assess_supply_chain_exposure
    # ==================================================================

    def assess_supply_chain_exposure(
        self,
        supply_chain_nodes: List[Dict[str, Any]],
        hazard_types: List[str],
        hazard_data: Dict[str, Dict[str, float]],
        scenario: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Assess supply-chain exposure across tiered nodes.

        Each node in *supply_chain_nodes* represents a supplier,
        facility, or logistics hub.  Nodes must include a ``tier``
        key (1, 2, or 3) and standard asset fields (``asset_id``,
        ``name``, ``asset_type``, ``location``).

        The engine auto-registers unregistered nodes, performs
        exposure assessment per (node, hazard) pair, and summarises
        results by supply-chain tier.

        Args:
            supply_chain_nodes: List of node dicts, each containing
                ``asset_id``, ``name``, ``asset_type``, ``location``,
                and ``tier`` (int: 1, 2, or 3).  Optional: ``sector``,
                ``value_usd``, ``metadata``.
            hazard_types: List of hazard types to assess.
            hazard_data: Mapping of hazard type string to a dict
                containing ``intensity``, ``probability``, ``frequency``.
            scenario: Optional climate scenario label.

        Returns:
            Dictionary with keys:
                ``per_node_results`` - list of per-node result dicts
                    containing ``asset_id``, ``tier``,
                    ``hazard_results``.
                ``tier_summary`` - dict mapping tier int to summary
                    with ``avg_score``, ``max_score``, ``node_count``,
                    ``critical_count``.
                ``critical_path_exposure`` - overall max score across
                    all nodes and hazards.
                ``scenario`` - scenario label.
                ``assessed_at`` - ISO-8601 UTC timestamp.
                ``provenance_hash`` - SHA-256 provenance hash.

        Raises:
            ValueError: If supply_chain_nodes or hazard_types are empty.
        """
        start_time = time.monotonic()

        if not supply_chain_nodes:
            raise ValueError("supply_chain_nodes must not be empty")
        if not hazard_types:
            raise ValueError("hazard_types must not be empty")

        per_node_results: List[Dict[str, Any]] = []
        tier_scores: Dict[int, List[float]] = {1: [], 2: [], 3: []}
        tier_critical: Dict[int, int] = {1: 0, 2: 0, 3: 0}
        overall_max: float = 0.0

        for node in supply_chain_nodes:
            aid = node.get("asset_id", "")
            tier = int(node.get("tier", 1))
            tier = _clamp(tier, 1, 3)  # type: ignore[arg-type]
            tier = int(tier)

            # Auto-register if not present
            if not self.get_asset(aid):
                try:
                    self.register_asset(
                        asset_id=aid,
                        name=node.get("name", f"Node-{aid}"),
                        asset_type=node.get(
                            "asset_type", AssetType.SUPPLY_CHAIN_NODE.value
                        ),
                        location=node.get("location", {}),
                        sector=node.get("sector"),
                        value_usd=node.get("value_usd"),
                        metadata=node.get("metadata"),
                    )
                except ValueError as exc:
                    logger.warning(
                        "Supply chain node auto-registration failed "
                        "for %s: %s",
                        aid, str(exc),
                    )
                    self._error_count += 1
                    continue

            node_hazard_results: List[Dict[str, Any]] = []
            node_has_critical = False

            for ht in hazard_types:
                ht_upper = ht.upper()
                hdata = hazard_data.get(ht_upper, hazard_data.get(ht, {}))

                intensity = float(hdata.get("intensity", 0.0))
                probability = float(hdata.get("probability", 0.0))
                frequency = float(hdata.get("frequency", 0.0))
                dist_km = hdata.get("distance_km")
                elev_f = hdata.get("elevation_factor")
                pop_f = hdata.get("population_factor")

                try:
                    result = self.assess_exposure(
                        asset_id=aid,
                        hazard_type=ht_upper,
                        hazard_intensity=intensity,
                        hazard_probability=probability,
                        hazard_frequency=frequency,
                        scenario=scenario,
                        distance_km=dist_km,
                        elevation_factor=elev_f,
                        population_factor=pop_f,
                    )
                    node_hazard_results.append(result)
                    score = float(result.get("composite_score", 0.0))
                    tier_scores.setdefault(tier, []).append(score)
                    if score > overall_max:
                        overall_max = score
                    level = result.get(
                        "exposure_level", ExposureLevel.NONE.value
                    )
                    if level in (
                        ExposureLevel.HIGH.value,
                        ExposureLevel.CRITICAL.value,
                    ):
                        node_has_critical = True
                except (ValueError, KeyError) as exc:
                    logger.warning(
                        "Supply chain assessment skipped for node=%s "
                        "hazard=%s: %s",
                        aid, ht_upper, str(exc),
                    )
                    self._error_count += 1

            if node_has_critical:
                tier_critical[tier] = tier_critical.get(tier, 0) + 1

            per_node_results.append({
                "asset_id": aid,
                "tier": tier,
                "hazard_results": node_hazard_results,
            })

        # --- Tier summary -----------------------------------------------------
        tier_summary: Dict[str, Any] = {}
        for t in [1, 2, 3]:
            scores = tier_scores.get(t, [])
            tier_summary[str(t)] = {
                "avg_score": round(
                    sum(scores) / len(scores), 4
                ) if scores else 0.0,
                "max_score": round(max(scores), 4) if scores else 0.0,
                "node_count": len(
                    [n for n in per_node_results if n.get("tier") == t]
                ),
                "critical_count": tier_critical.get(t, 0),
            }

        now_iso = _utcnow().isoformat()

        provenance_hash = self._compute_provenance(
            operation="assess_supply_chain_exposure",
            input_data={
                "node_count": len(supply_chain_nodes),
                "hazard_count": len(hazard_types),
                "scenario": scenario,
            },
            output_data={
                "tier_summary": tier_summary,
                "critical_path_exposure": overall_max,
            },
        )

        with self._lock:
            self._supply_chain_count += 1

        self._record_provenance_entry(
            entity_type="exposure",
            action="assess_supply_chain",
            entity_id=f"sc-{uuid4().hex[:8]}",
            data=tier_summary,
        )

        duration = time.monotonic() - start_time
        logger.info(
            "Supply chain exposure assessed: %d nodes x %d hazards, "
            "critical_path=%.2f (%.3fs)",
            len(supply_chain_nodes),
            len(hazard_types),
            overall_max,
            duration,
        )

        return copy.deepcopy({
            "per_node_results": per_node_results,
            "tier_summary": tier_summary,
            "critical_path_exposure": round(overall_max, 4),
            "scenario": scenario,
            "assessed_at": now_iso,
            "provenance_hash": provenance_hash,
        })

    # ==================================================================
    # 9. identify_hotspots
    # ==================================================================

    def identify_hotspots(
        self,
        asset_ids: Optional[List[str]] = None,
        hazard_types: Optional[List[str]] = None,
        threshold: float = 55.0,
    ) -> List[Dict[str, Any]]:
        """Identify assets whose exposure scores exceed a threshold.

        Scans stored assessments and returns those with a composite
        score at or above the specified threshold, optionally filtered
        by asset IDs and/or hazard types.

        Args:
            asset_ids: Optional list of asset identifiers to restrict
                the search.  When None, all assets are scanned.
            hazard_types: Optional list of hazard types to filter.
                When None, all hazard types are included.
            threshold: Minimum composite score to qualify as a hotspot.
                Defaults to 55.0 (HIGH level boundary).

        Returns:
            List of hotspot dictionaries sorted by composite_score
            descending.  Each dict contains ``assessment_id``,
            ``asset_id``, ``hazard_type``, ``composite_score``,
            ``exposure_level``, ``assessed_at``.
        """
        start_time = time.monotonic()

        with self._lock:
            assessments = list(self._assessments.values())

        # Normalise filter sets
        aid_set: Optional[frozenset[str]] = None
        if asset_ids is not None:
            aid_set = frozenset(asset_ids)

        ht_set: Optional[frozenset[str]] = None
        if hazard_types is not None:
            ht_set = frozenset(h.upper() for h in hazard_types)

        hotspots: List[Dict[str, Any]] = []

        for assess in assessments:
            # Filter by asset
            if aid_set is not None and assess.asset_id not in aid_set:
                continue

            # Filter by hazard
            if ht_set is not None and assess.hazard_type not in ht_set:
                continue

            # Filter by threshold
            if assess.composite_score < threshold:
                continue

            hotspots.append({
                "assessment_id": assess.assessment_id,
                "asset_id": assess.asset_id,
                "hazard_type": assess.hazard_type,
                "composite_score": assess.composite_score,
                "exposure_level": assess.exposure_level,
                "assessed_at": assess.assessed_at,
            })

        # Sort descending by composite score
        hotspots.sort(
            key=lambda h: h.get("composite_score", 0.0), reverse=True
        )

        with self._lock:
            self._hotspot_count += 1

        self._record_provenance_entry(
            entity_type="exposure",
            action="identify_hotspots",
            entity_id=f"hotspot-{uuid4().hex[:8]}",
            data={
                "threshold": threshold,
                "hotspot_count": len(hotspots),
                "asset_filter_count": len(asset_ids) if asset_ids else 0,
                "hazard_filter_count": len(hazard_types) if hazard_types else 0,
            },
        )

        duration = time.monotonic() - start_time
        logger.info(
            "Hotspot identification complete: threshold=%.1f, "
            "found=%d hotspots (%.3fs)",
            threshold,
            len(hotspots),
            duration,
        )

        return copy.deepcopy(hotspots)

    # ==================================================================
    # 10. get_exposure_map
    # ==================================================================

    def get_exposure_map(
        self,
        hazard_type: str,
        bounding_box: Dict[str, float],
        resolution_km: float = 10.0,
        intensity: float = 0.5,
        probability: float = 0.5,
        frequency: float = 0.5,
        elevation_factor: float = 0.5,
        population_factor: float = 0.5,
    ) -> Dict[str, Any]:
        """Generate a spatial exposure grid over a bounding box.

        Divides the bounding box into cells of approximately
        *resolution_km* width and computes an exposure score at the
        centroid of each cell using the five-factor formula.

        The hazard epicentre is assumed to be the centre of the
        bounding box.  Distance from each grid cell centroid to the
        epicentre is computed via Haversine.

        Args:
            hazard_type: One of the 12 supported hazard types.
            bounding_box: Dict with keys ``min_lat``, ``max_lat``,
                ``min_lon``, ``max_lon``.
            resolution_km: Approximate cell size in km.  Defaults to 10.
            intensity: Uniform hazard intensity [0, 1].  Defaults to 0.5.
            probability: Uniform hazard probability [0, 1].  Defaults to
                0.5.
            frequency: Uniform hazard frequency [0, 1].  Defaults to 0.5.
            elevation_factor: Uniform elevation factor [0, 1].  Defaults
                to 0.5.
            population_factor: Uniform population factor [0, 1].  Defaults
                to 0.5.

        Returns:
            Dictionary with keys:
                ``hazard_type`` - hazard type string.
                ``bounding_box`` - echo of the input bounding box.
                ``resolution_km`` - cell resolution.
                ``grid_cells`` - list of dicts, each with ``latitude``,
                    ``longitude``, ``exposure_score``,
                    ``exposure_level``, ``distance_km``.
                ``cell_count`` - total number of grid cells.
                ``assessed_at`` - ISO-8601 UTC timestamp.
                ``provenance_hash`` - SHA-256 provenance hash.

        Raises:
            ValueError: If hazard_type is unsupported or bounding_box
                is malformed.
        """
        start_time = time.monotonic()

        self._validate_hazard_type(hazard_type)
        hazard_upper = hazard_type.upper()

        # --- Validate bounding box -------------------------------------------
        required_keys = {"min_lat", "max_lat", "min_lon", "max_lon"}
        if not isinstance(bounding_box, dict):
            raise ValueError("bounding_box must be a dict")
        missing = required_keys - set(bounding_box.keys())
        if missing:
            raise ValueError(
                f"bounding_box missing keys: {sorted(missing)}"
            )

        min_lat = float(bounding_box["min_lat"])
        max_lat = float(bounding_box["max_lat"])
        min_lon = float(bounding_box["min_lon"])
        max_lon = float(bounding_box["max_lon"])

        if min_lat >= max_lat:
            raise ValueError(
                f"min_lat ({min_lat}) must be < max_lat ({max_lat})"
            )
        if min_lon >= max_lon:
            raise ValueError(
                f"min_lon ({min_lon}) must be < max_lon ({max_lon})"
            )

        if not (-90.0 <= min_lat <= 90.0 and -90.0 <= max_lat <= 90.0):
            raise ValueError("Latitude must be in [-90, 90]")
        if not (-180.0 <= min_lon <= 180.0 and -180.0 <= max_lon <= 180.0):
            raise ValueError("Longitude must be in [-180, 180]")

        safe_resolution = max(0.1, float(resolution_km))

        # --- Grid generation --------------------------------------------------
        centre_lat = (min_lat + max_lat) / 2.0
        centre_lon = (min_lon + max_lon) / 2.0

        # Approximate degrees per km at the centre latitude
        lat_km = 111.32  # km per degree latitude (approximately constant)
        lon_km = 111.32 * math.cos(math.radians(centre_lat))
        lon_km = max(lon_km, 0.001)  # prevent division by zero at poles

        lat_step = safe_resolution / lat_km
        lon_step = safe_resolution / lon_km

        max_radius = HAZARD_MAX_RADIUS_KM.get(hazard_upper, 100.0)

        intensity_norm = _clamp(float(intensity), 0.0, 1.0)
        probability_norm = _clamp(float(probability), 0.0, 1.0)
        frequency_norm = _clamp(float(frequency), 0.0, 1.0)
        elev_f = _clamp(float(elevation_factor), 0.0, 1.0)
        pop_f = _clamp(float(population_factor), 0.0, 1.0)

        grid_cells: List[Dict[str, Any]] = []
        lat = min_lat
        while lat <= max_lat:
            lon = min_lon
            while lon <= max_lon:
                dist = _haversine_km(lat, lon, centre_lat, centre_lon)
                prox = max(0.0, 1.0 - dist / max_radius)

                raw_comp = (
                    prox * WEIGHT_PROXIMITY
                    + intensity_norm * WEIGHT_INTENSITY
                    + frequency_norm * WEIGHT_FREQUENCY
                    + elev_f * WEIGHT_ELEVATION
                    + pop_f * WEIGHT_POPULATION
                ) * 100.0

                score = _clamp(raw_comp * probability_norm, 0.0, 100.0)
                level = self._classify_exposure_level(score)

                grid_cells.append({
                    "latitude": round(lat, 6),
                    "longitude": round(lon, 6),
                    "exposure_score": round(score, 4),
                    "exposure_level": level,
                    "distance_km": round(dist, 4),
                })

                lon += lon_step
            lat += lat_step

        # --- Safety cap on grid cells -----------------------------------------
        max_cells = 10_000
        if len(grid_cells) > max_cells:
            logger.warning(
                "Exposure map grid capped at %d cells (requested %d). "
                "Increase resolution_km or reduce bounding_box size.",
                max_cells,
                len(grid_cells),
            )
            grid_cells = grid_cells[:max_cells]

        now_iso = _utcnow().isoformat()

        provenance_hash = self._compute_provenance(
            operation="get_exposure_map",
            input_data={
                "hazard_type": hazard_upper,
                "bounding_box": bounding_box,
                "resolution_km": safe_resolution,
            },
            output_data={"cell_count": len(grid_cells)},
        )

        with self._lock:
            self._exposure_map_count += 1

        self._record_provenance_entry(
            entity_type="exposure",
            action="assess_exposure",
            entity_id=f"map-{uuid4().hex[:8]}",
            data={
                "hazard_type": hazard_upper,
                "cell_count": len(grid_cells),
            },
        )

        duration = time.monotonic() - start_time
        logger.info(
            "Exposure map generated: hazard=%s, cells=%d, "
            "resolution=%.1fkm (%.3fs)",
            hazard_upper,
            len(grid_cells),
            safe_resolution,
            duration,
        )

        return copy.deepcopy({
            "hazard_type": hazard_upper,
            "bounding_box": {
                "min_lat": min_lat,
                "max_lat": max_lat,
                "min_lon": min_lon,
                "max_lon": max_lon,
            },
            "resolution_km": safe_resolution,
            "grid_cells": grid_cells,
            "cell_count": len(grid_cells),
            "assessed_at": now_iso,
            "provenance_hash": provenance_hash,
        })

    # ==================================================================
    # 11. get_assessment
    # ==================================================================

    def get_assessment(
        self,
        assessment_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a stored exposure assessment by its identifier.

        Args:
            assessment_id: The unique assessment identifier to look up.

        Returns:
            Deep-copied dictionary of the assessment, or ``None`` if
            the assessment does not exist.
        """
        if not assessment_id:
            return None

        with self._lock:
            assessment = self._assessments.get(assessment_id)

        if assessment is None:
            return None

        return copy.deepcopy(assessment.to_dict())

    # ==================================================================
    # 12. list_assessments
    # ==================================================================

    def list_assessments(
        self,
        asset_id: Optional[str] = None,
        hazard_type: Optional[str] = None,
        exposure_level: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List stored assessments with optional filters.

        Args:
            asset_id: Optional filter by asset identifier.
            hazard_type: Optional filter by hazard type string.
            exposure_level: Optional filter by exposure level string.
            limit: Maximum number of assessments to return.  Defaults
                to 100.

        Returns:
            List of deep-copied assessment dictionaries, newest first,
            limited to *limit*.
        """
        with self._lock:
            assessments = list(self._assessments.values())

        # Apply filters
        if asset_id is not None:
            assessments = [
                a for a in assessments if a.asset_id == asset_id
            ]

        if hazard_type is not None:
            ht_upper = hazard_type.upper()
            assessments = [
                a for a in assessments if a.hazard_type == ht_upper
            ]

        if exposure_level is not None:
            el_upper = exposure_level.upper()
            assessments = [
                a for a in assessments if a.exposure_level == el_upper
            ]

        # Newest first
        assessments.sort(key=lambda a: a.assessed_at, reverse=True)

        # Apply limit
        safe_limit = max(1, limit) if limit else 100
        assessments = assessments[:safe_limit]

        return [copy.deepcopy(a.to_dict()) for a in assessments]

    # ==================================================================
    # 13. get_statistics
    # ==================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Return engine operational statistics.

        Returns:
            Dictionary with counts and summary statistics:
                ``total_assets`` - number of registered assets.
                ``total_assessments`` - number of assessments performed.
                ``total_portfolios`` - number of portfolio assessments.
                ``total_supply_chains`` - number of supply chain runs.
                ``total_hotspot_queries`` - number of hotspot queries.
                ``total_exposure_maps`` - number of exposure maps.
                ``total_errors`` - accumulated error count.
                ``asset_type_distribution`` - count per asset type.
                ``exposure_level_distribution`` - count per level.
                ``hazard_type_distribution`` - count per hazard type.
                ``provenance_entries`` - provenance entry count.
        """
        with self._lock:
            assets = list(self._assets.values())
            assessments = list(self._assessments.values())
            stats: Dict[str, Any] = {
                "total_assets": len(self._assets),
                "total_assessments": self._assessment_count,
                "total_portfolios": self._portfolio_count,
                "total_supply_chains": self._supply_chain_count,
                "total_hotspot_queries": self._hotspot_count,
                "total_exposure_maps": self._exposure_map_count,
                "total_errors": self._error_count,
            }

        # Asset type distribution
        asset_type_dist: Dict[str, int] = {}
        for a in assets:
            at = a.asset_type
            asset_type_dist[at] = asset_type_dist.get(at, 0) + 1
        stats["asset_type_distribution"] = asset_type_dist

        # Exposure level distribution
        exposure_dist: Dict[str, int] = {
            level.value: 0 for level in ExposureLevel
        }
        for assess in assessments:
            level = assess.exposure_level
            if level in exposure_dist:
                exposure_dist[level] += 1
        stats["exposure_level_distribution"] = exposure_dist

        # Hazard type distribution
        hazard_dist: Dict[str, int] = {}
        for assess in assessments:
            ht = assess.hazard_type
            hazard_dist[ht] = hazard_dist.get(ht, 0) + 1
        stats["hazard_type_distribution"] = hazard_dist

        # Provenance entry count
        if self._provenance is not None:
            try:
                stats["provenance_entries"] = len(self._provenance)
            except Exception:
                stats["provenance_entries"] = 0
        else:
            stats["provenance_entries"] = 0

        return copy.deepcopy(stats)

    # ==================================================================
    # 14. clear
    # ==================================================================

    def clear(self) -> None:
        """Reset all engine state to initial (empty) condition.

        Clears the asset registry, assessment store, and provenance
        tracker.  Resets all counters to zero.  Intended for testing
        and teardown.
        """
        with self._lock:
            self._assets.clear()
            self._assessments.clear()
            self._asset_count = 0
            self._assessment_count = 0
            self._portfolio_count = 0
            self._supply_chain_count = 0
            self._hotspot_count = 0
            self._exposure_map_count = 0
            self._error_count = 0

        if self._provenance is not None:
            try:
                self._provenance.reset()
            except Exception:
                pass

        self._record_provenance_entry(
            entity_type="exposure",
            action="clear_engine",
            entity_id="exposure-assessor",
            data={"cleared": True},
        )

        logger.info("ExposureAssessorEngine cleared to initial state")

    # ==================================================================
    # Additional public utilities
    # ==================================================================

    def compute_proximity_score(
        self,
        distance_km: float,
        hazard_type: str,
    ) -> float:
        """Compute the proximity decay score for a distance and hazard.

        Formula: max(0, 1 - distance_km / max_radius_km)

        Args:
            distance_km: Distance from asset to hazard centroid in km.
            hazard_type: One of the 12 supported hazard types.

        Returns:
            Proximity score in [0, 1].  Returns 1.0 when distance is 0
            and 0.0 when distance exceeds the max radius.
        """
        hazard_upper = hazard_type.upper()
        max_radius = HAZARD_MAX_RADIUS_KM.get(hazard_upper, 100.0)
        dist = max(0.0, float(distance_km))
        return max(0.0, 1.0 - dist / max_radius)

    def compute_composite_score(
        self,
        proximity_score: float,
        intensity_norm: float,
        frequency_norm: float,
        elevation_factor: float,
        population_factor: float,
        probability_norm: float = 1.0,
    ) -> float:
        """Compute the composite exposure score from normalised factors.

        Formula:
            raw = (proximity * 0.25 + intensity * 0.30 + frequency * 0.25
                   + elevation * 0.10 + population * 0.10) * 100
            score = raw * probability

        Args:
            proximity_score: Proximity decay [0, 1].
            intensity_norm: Hazard intensity normalised [0, 1].
            frequency_norm: Hazard frequency normalised [0, 1].
            elevation_factor: Elevation factor [0, 1].
            population_factor: Population density factor [0, 1].
            probability_norm: Hazard probability modifier [0, 1].
                Defaults to 1.0.

        Returns:
            Composite exposure score in [0, 100].
        """
        raw = (
            _clamp(proximity_score, 0.0, 1.0) * WEIGHT_PROXIMITY
            + _clamp(intensity_norm, 0.0, 1.0) * WEIGHT_INTENSITY
            + _clamp(frequency_norm, 0.0, 1.0) * WEIGHT_FREQUENCY
            + _clamp(elevation_factor, 0.0, 1.0) * WEIGHT_ELEVATION
            + _clamp(population_factor, 0.0, 1.0) * WEIGHT_POPULATION
        ) * 100.0
        return _clamp(raw * _clamp(probability_norm, 0.0, 1.0), 0.0, 100.0)

    def classify_exposure(self, score: float) -> str:
        """Classify a composite score into an exposure level.

        Args:
            score: Composite exposure score in [0, 100].

        Returns:
            Exposure level string (NONE, LOW, MODERATE, HIGH, CRITICAL).
        """
        return self._classify_exposure_level(score)

    def get_hazard_max_radius(self, hazard_type: str) -> float:
        """Return the maximum proximity radius for a hazard type.

        Args:
            hazard_type: One of the 12 supported hazard types.

        Returns:
            Max radius in km.  Returns 100.0 for unrecognised types.
        """
        return HAZARD_MAX_RADIUS_KM.get(hazard_type.upper(), 100.0)

    def get_supported_asset_types(self) -> List[str]:
        """Return the list of supported asset type strings.

        Returns:
            Sorted list of the 8 supported asset type strings.
        """
        return sorted(VALID_ASSET_TYPES)

    def get_supported_hazard_types(self) -> List[str]:
        """Return the list of supported hazard type strings.

        Returns:
            Sorted list of the 12 supported hazard type strings.
        """
        return sorted(VALID_HAZARD_TYPES)

    def get_supported_exposure_levels(self) -> List[str]:
        """Return the list of supported exposure level strings.

        Returns:
            Ordered list of the 5 exposure levels from NONE to CRITICAL.
        """
        return [
            ExposureLevel.NONE.value,
            ExposureLevel.LOW.value,
            ExposureLevel.MODERATE.value,
            ExposureLevel.HIGH.value,
            ExposureLevel.CRITICAL.value,
        ]

    def haversine_distance(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
    ) -> float:
        """Compute Haversine great-circle distance between two points.

        Public wrapper around the module-level helper for use by
        callers that need distance computation without internal state.

        Args:
            lat1: Latitude of point 1 (decimal degrees).
            lon1: Longitude of point 1 (decimal degrees).
            lat2: Latitude of point 2 (decimal degrees).
            lon2: Longitude of point 2 (decimal degrees).

        Returns:
            Distance in kilometres.
        """
        return _haversine_km(lat1, lon1, lat2, lon2)

    def get_asset_count(self) -> int:
        """Return the number of currently registered assets.

        Returns:
            Integer count of assets in the registry.
        """
        with self._lock:
            return len(self._assets)

    def get_assessment_count(self) -> int:
        """Return the number of stored exposure assessments.

        Returns:
            Integer count of assessments in the store.
        """
        with self._lock:
            return len(self._assessments)

    def get_assessments_for_asset(
        self,
        asset_id: str,
    ) -> List[Dict[str, Any]]:
        """Return all assessments for a given asset.

        Args:
            asset_id: The asset identifier to look up.

        Returns:
            List of deep-copied assessment dictionaries, newest first.
        """
        return self.list_assessments(asset_id=asset_id, limit=10000)

    def get_assessments_for_hazard(
        self,
        hazard_type: str,
    ) -> List[Dict[str, Any]]:
        """Return all assessments for a given hazard type.

        Args:
            hazard_type: The hazard type string to filter by.

        Returns:
            List of deep-copied assessment dictionaries, newest first.
        """
        return self.list_assessments(hazard_type=hazard_type, limit=10000)

    def get_worst_exposure_for_asset(
        self,
        asset_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Return the assessment with the highest composite score for an asset.

        Args:
            asset_id: The asset identifier to look up.

        Returns:
            Deep-copied assessment dictionary, or ``None`` if no
            assessments exist for the asset.
        """
        assessments = self.list_assessments(asset_id=asset_id, limit=10000)
        if not assessments:
            return None
        return max(assessments, key=lambda a: a.get("composite_score", 0.0))

    def batch_register_assets(
        self,
        assets: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Register multiple assets in a single batch call.

        Each element of *assets* must contain the keys required by
        :meth:`register_asset`: ``asset_id``, ``name``, ``asset_type``,
        ``location``.  Optional: ``sector``, ``value_usd``, ``metadata``.

        Failures for individual assets are logged and skipped; the
        remaining assets are still processed.

        Args:
            assets: List of asset definition dicts.

        Returns:
            List of successfully registered asset record dicts.
        """
        results: List[Dict[str, Any]] = []
        for asset_def in assets:
            try:
                result = self.register_asset(
                    asset_id=asset_def.get("asset_id", ""),
                    name=asset_def.get("name", ""),
                    asset_type=asset_def.get("asset_type", ""),
                    location=asset_def.get("location", {}),
                    sector=asset_def.get("sector"),
                    value_usd=asset_def.get("value_usd"),
                    metadata=asset_def.get("metadata"),
                )
                results.append(result)
            except (ValueError, TypeError, KeyError) as exc:
                logger.warning(
                    "Batch registration failed for asset %s: %s",
                    asset_def.get("asset_id", "unknown"),
                    str(exc),
                )
                self._error_count += 1
        return results

    def compute_distance_to_asset(
        self,
        asset_id: str,
        target_lat: float,
        target_lon: float,
    ) -> Optional[float]:
        """Compute Haversine distance from a registered asset to a point.

        Args:
            asset_id: Identifier of the registered asset.
            target_lat: Target latitude in decimal degrees.
            target_lon: Target longitude in decimal degrees.

        Returns:
            Distance in km, or ``None`` if the asset is not found.
        """
        asset = self.get_asset(asset_id)
        if asset is None:
            return None

        loc = asset.get("location", {})
        lat = float(loc.get("latitude", 0.0))
        lon = float(loc.get("longitude", 0.0))
        return _haversine_km(lat, lon, target_lat, target_lon)

    def get_exposure_summary_for_asset(
        self,
        asset_id: str,
    ) -> Dict[str, Any]:
        """Return a summary of all exposure assessments for an asset.

        Args:
            asset_id: The asset identifier.

        Returns:
            Dictionary with ``asset_id``, ``assessment_count``,
            ``avg_score``, ``max_score``, ``min_score``,
            ``exposure_level_distribution``, and ``hazard_types_assessed``.
        """
        assessments = self.list_assessments(asset_id=asset_id, limit=10000)
        scores = [
            float(a.get("composite_score", 0.0)) for a in assessments
        ]

        level_dist: Dict[str, int] = {
            level.value: 0 for level in ExposureLevel
        }
        hazard_types_seen: set[str] = set()

        for a in assessments:
            lvl = a.get("exposure_level", ExposureLevel.NONE.value)
            if lvl in level_dist:
                level_dist[lvl] += 1
            hazard_types_seen.add(a.get("hazard_type", ""))

        return {
            "asset_id": asset_id,
            "assessment_count": len(assessments),
            "avg_score": round(
                sum(scores) / len(scores), 4
            ) if scores else 0.0,
            "max_score": round(max(scores), 4) if scores else 0.0,
            "min_score": round(min(scores), 4) if scores else 0.0,
            "exposure_level_distribution": level_dist,
            "hazard_types_assessed": sorted(hazard_types_seen),
        }

    def export_assessments(self) -> List[Dict[str, Any]]:
        """Export all stored assessments as a list of dictionaries.

        Returns:
            List of deep-copied assessment dictionaries, oldest first.
        """
        with self._lock:
            assessments = list(self._assessments.values())
        assessments.sort(key=lambda a: a.assessed_at)
        return [copy.deepcopy(a.to_dict()) for a in assessments]

    def export_assets(self) -> List[Dict[str, Any]]:
        """Export all registered assets as a list of dictionaries.

        Returns:
            List of deep-copied asset record dictionaries, oldest first.
        """
        with self._lock:
            assets = list(self._assets.values())
        assets.sort(key=lambda a: a.registered_at)
        return [copy.deepcopy(a.to_dict()) for a in assets]

    def import_assets(
        self,
        assets: List[Dict[str, Any]],
    ) -> int:
        """Import assets from a list of dictionaries.

        Existing assets with matching IDs are overwritten.

        Args:
            assets: List of asset record dictionaries.

        Returns:
            Number of assets successfully imported.
        """
        count = 0
        for asset_dict in assets:
            try:
                self.register_asset(
                    asset_id=asset_dict.get("asset_id", ""),
                    name=asset_dict.get("name", ""),
                    asset_type=asset_dict.get("asset_type", ""),
                    location=asset_dict.get("location", {}),
                    sector=asset_dict.get("sector"),
                    value_usd=asset_dict.get("value_usd"),
                    metadata=asset_dict.get("metadata"),
                )
                count += 1
            except (ValueError, TypeError, KeyError) as exc:
                logger.warning(
                    "Import failed for asset %s: %s",
                    asset_dict.get("asset_id", "unknown"),
                    str(exc),
                )
                self._error_count += 1
        return count

    # ==================================================================
    # Dunder methods
    # ==================================================================

    def __repr__(self) -> str:
        """Return a developer-friendly string representation.

        Returns:
            String showing asset and assessment counts.
        """
        with self._lock:
            asset_c = len(self._assets)
            assess_c = len(self._assessments)
        return (
            f"ExposureAssessorEngine("
            f"assets={asset_c}, "
            f"assessments={assess_c}, "
            f"risk_engine={'attached' if self._risk_engine else 'standalone'}, "
            f"provenance={'on' if self._provenance else 'off'})"
        )

    def __len__(self) -> int:
        """Return the total number of stored assessments.

        Returns:
            Integer count of assessments in the store.
        """
        with self._lock:
            return len(self._assessments)

    # ==================================================================
    # Private validation helpers
    # ==================================================================

    @staticmethod
    def _validate_asset_id(asset_id: str) -> None:
        """Validate an asset identifier.

        Args:
            asset_id: The identifier to validate.

        Raises:
            ValueError: If asset_id is empty or whitespace-only.
        """
        if not asset_id or not asset_id.strip():
            raise ValueError("asset_id must not be empty")

    @staticmethod
    def _validate_name(name: str) -> None:
        """Validate an asset name.

        Args:
            name: The name string to validate.

        Raises:
            ValueError: If name is empty or whitespace-only.
        """
        if not name or not name.strip():
            raise ValueError("name must not be empty")

    @staticmethod
    def _validate_asset_type(asset_type: str) -> None:
        """Validate an asset type string.

        Args:
            asset_type: The asset type to validate.

        Raises:
            ValueError: If asset_type is not one of the 8 supported
                types.
        """
        if not asset_type or asset_type.upper() not in VALID_ASSET_TYPES:
            raise ValueError(
                f"asset_type must be one of {sorted(VALID_ASSET_TYPES)}, "
                f"got '{asset_type}'"
            )

    @staticmethod
    def _validate_location(location: Dict[str, Any]) -> None:
        """Validate a location dictionary.

        Args:
            location: Dict with required keys ``latitude`` and
                ``longitude``.

        Raises:
            ValueError: If location is not a dict, or latitude/longitude
                are missing or out of valid range.
        """
        if not isinstance(location, dict):
            raise ValueError("location must be a dict")

        if "latitude" not in location:
            raise ValueError("location must contain 'latitude'")
        if "longitude" not in location:
            raise ValueError("location must contain 'longitude'")

        lat = float(location["latitude"])
        lon = float(location["longitude"])

        if not (-90.0 <= lat <= 90.0):
            raise ValueError(
                f"latitude must be in [-90, 90], got {lat}"
            )
        if not (-180.0 <= lon <= 180.0):
            raise ValueError(
                f"longitude must be in [-180, 180], got {lon}"
            )

    @staticmethod
    def _validate_hazard_type(hazard_type: str) -> None:
        """Validate a hazard type string.

        Args:
            hazard_type: The hazard type to validate.

        Raises:
            ValueError: If hazard_type is not one of the 12 supported
                types.
        """
        if not hazard_type or hazard_type.upper() not in VALID_HAZARD_TYPES:
            raise ValueError(
                f"hazard_type must be one of {sorted(VALID_HAZARD_TYPES)}, "
                f"got '{hazard_type}'"
            )

    @staticmethod
    def _normalise_location(location: Dict[str, Any]) -> Dict[str, Any]:
        """Normalise a location dict to canonical form.

        Args:
            location: Raw location dict.

        Returns:
            Dict with ``latitude``, ``longitude``, and
            ``elevation_m`` (float or None).
        """
        return {
            "latitude": float(location["latitude"]),
            "longitude": float(location["longitude"]),
            "elevation_m": (
                float(location["elevation_m"])
                if "elevation_m" in location and location["elevation_m"] is not None
                else None
            ),
        }

    # ==================================================================
    # Private calculation helpers
    # ==================================================================

    @staticmethod
    def _classify_exposure_level(score: float) -> str:
        """Classify a composite score into an exposure level.

        Threshold boundaries (inclusive lower, exclusive upper):
            NONE       [0, 10)
            LOW       [10, 30)
            MODERATE  [30, 55)
            HIGH      [55, 80)
            CRITICAL  [80, 100]

        Args:
            score: Composite exposure score in [0, 100].

        Returns:
            Exposure level string.
        """
        clamped = _clamp(score, 0.0, 100.0)

        if clamped >= 80.0:
            return ExposureLevel.CRITICAL.value
        if clamped >= 55.0:
            return ExposureLevel.HIGH.value
        if clamped >= 30.0:
            return ExposureLevel.MODERATE.value
        if clamped >= 10.0:
            return ExposureLevel.LOW.value
        return ExposureLevel.NONE.value

    @staticmethod
    def _derive_elevation_factor(
        record: AssetRecord,
        hazard_type: str,
    ) -> float:
        """Derive an elevation factor from an asset's elevation_m.

        For flood and sea-level hazards, lower elevation increases
        exposure.  For other hazards, the factor is a neutral 0.5.

        Args:
            record: The asset record.
            hazard_type: Upper-case hazard type string.

        Returns:
            Elevation factor in [0, 1].
        """
        elev = record.location.get("elevation_m")
        if elev is None:
            return 0.5

        elev_m = float(elev)

        # Flood / coastal hazards: lower elevation = higher factor
        flood_hazards = {
            HazardType.RIVERINE_FLOOD.value,
            HazardType.COASTAL_FLOOD.value,
            HazardType.SEA_LEVEL_RISE.value,
            HazardType.COASTAL_EROSION.value,
        }

        if hazard_type in flood_hazards:
            # 0m => factor 1.0, >= 50m => factor 0.0
            max_elev = 50.0
            if elev_m <= DEFAULT_ELEVATION_SEA_LEVEL_M:
                return 1.0
            if elev_m >= max_elev:
                return 0.0
            return max(0.0, 1.0 - elev_m / max_elev)

        # Landslide: higher elevation = higher factor
        if hazard_type == HazardType.LANDSLIDE.value:
            # 0m => factor 0.0, >= 2000m => factor 1.0
            if elev_m <= DEFAULT_ELEVATION_SEA_LEVEL_M:
                return 0.0
            if elev_m >= DEFAULT_ELEVATION_HIGH_M:
                return 1.0
            return min(1.0, elev_m / DEFAULT_ELEVATION_HIGH_M)

        # Wildfire: moderate elevation = higher factor (bell curve)
        if hazard_type == HazardType.WILDFIRE.value:
            # Peak at 500m, falloff by 1500m
            peak = 500.0
            span = 1500.0
            diff = abs(elev_m - peak)
            if diff >= span:
                return 0.1
            return max(0.1, 1.0 - diff / span)

        # All other hazards: neutral 0.5
        return 0.5

    @staticmethod
    def _derive_population_factor(record: AssetRecord) -> float:
        """Derive a population density factor from asset metadata.

        If the asset's metadata contains ``population_density``, it is
        normalised to [0, 1] using the DEFAULT_POP_HIGH constant.
        Otherwise a neutral 0.5 is returned.

        Args:
            record: The asset record.

        Returns:
            Population density factor in [0, 1].
        """
        pop_density = record.metadata.get("population_density")
        if pop_density is None:
            return 0.5

        density = float(pop_density)
        if density <= DEFAULT_POP_LOW:
            return 0.0
        if density >= DEFAULT_POP_HIGH:
            return 1.0
        return density / DEFAULT_POP_HIGH

    # ==================================================================
    # Private provenance helpers
    # ==================================================================

    def _compute_provenance(
        self,
        operation: str,
        input_data: Any,
        output_data: Any,
    ) -> str:
        """Compute a SHA-256 provenance hash for an operation.

        Combines the operation name, input data, output data, and
        current UTC timestamp into a deterministic hash.

        Args:
            operation: Name of the operation performed.
            input_data: Input data for the operation.
            output_data: Output data from the operation.

        Returns:
            Hex-encoded SHA-256 hash string.
        """
        timestamp = _utcnow().isoformat()
        payload = {
            "operation": operation,
            "input": input_data,
            "output": output_data,
            "timestamp": timestamp,
        }
        serialized = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _record_provenance_entry(
        self,
        entity_type: str,
        action: str,
        entity_id: str,
        data: Any = None,
    ) -> None:
        """Record a provenance entry via the tracker if available.

        Silently no-ops when the provenance tracker is not present or
        an error occurs.

        Args:
            entity_type: Type of entity being tracked.
            action: Action performed.
            entity_id: Unique identifier for the entity instance.
            data: Optional data payload to hash.
        """
        if self._provenance is None:
            return
        try:
            self._provenance.record(
                entity_type=entity_type,
                action=action,
                entity_id=entity_id,
                data=data,
            )
        except Exception as exc:
            logger.debug(
                "Provenance recording failed for %s/%s/%s: %s",
                entity_type,
                action,
                entity_id,
                str(exc),
            )
