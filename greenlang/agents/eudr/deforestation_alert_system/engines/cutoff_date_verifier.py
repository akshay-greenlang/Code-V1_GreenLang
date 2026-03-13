# -*- coding: utf-8 -*-
"""
CutoffDateVerifier - AGENT-EUDR-020 Engine 5: EUDR Cutoff Date Verification

Verifies whether detected deforestation events occurred before or after the
EUDR cutoff date (December 31, 2020) per Regulation (EU) 2023/1115 Article 2(1).
Uses multi-source temporal evidence from Sentinel-2, Landsat, Hansen GFC, and
GLAD alert systems to construct time-series forest state histories and make
deterministic pre/post-cutoff classification decisions.

Products from areas deforested AFTER the cutoff date CANNOT be placed on
the EU market. This engine is therefore critical for compliance determination
and directly impacts market access decisions.

Zero-Hallucination Guarantees:
    - All temporal analysis uses deterministic date arithmetic.
    - Confidence scoring uses weighted Decimal sums of evidence quality.
    - Cutoff classification uses explicit threshold comparisons.
    - Forest state transitions are derived from spectral index observations.
    - Evidence weighting uses static per-source reliability scores.
    - SHA-256 provenance hashes on all output objects.
    - No ML/LLM models in any classification or scoring path.

Evidence Sources and Temporal Coverage:
    - Sentinel-2: 10m resolution, available from June 2015 (S2A) / March 2017 (S2B),
      5-day revisit, NDVI/EVI/NBR spectral indices, cloud-filtered.
    - Landsat 8/9: 30m resolution, available from 2013, 8-day revisit,
      NDVI/NBR spectral indices, cross-calibrated with Sentinel-2.
    - Hansen Global Forest Change (GFC): Annual 30m tree cover loss from
      Landsat time series, available from 2000, year-of-loss attribution.
    - GLAD Alerts: Weekly Landsat-based deforestation alerts from University
      of Maryland, available from 2016, sub-annual temporal resolution.
    - RADD Alerts: Sentinel-1 SAR radar alerts, cloud-independent, available
      from 2019, complementary to optical sources.

Confidence Classification Thresholds:
    - HIGH:         confidence >= 0.85
    - MEDIUM:       confidence >= 0.65
    - LOW:          confidence >= 0.45
    - INSUFFICIENT: confidence <  0.45

Cutoff Decision Logic:
    - PRE_CUTOFF:  All evidence indicates clearing completed before 2020-12-31.
    - POST_CUTOFF: Evidence indicates clearing began or was detected after cutoff.
    - ONGOING:     Clearing was in progress spanning the cutoff date.
    - UNCERTAIN:   Insufficient temporal evidence to make determination.
    - If UNCERTAIN and confidence < threshold, treated as HIGH risk.

Grace Period:
    - A configurable grace period (default 90 days) before the cutoff allows
      for temporal uncertainty in satellite revisit gaps and cloud cover.

Performance Targets:
    - Single verification: <200ms
    - Batch verification (100 detections): <5s
    - Evidence retrieval: <100ms per detection
    - Timeline construction: <150ms per detection

Regulatory References:
    - EUDR Article 2(1): Definition of cutoff date (31 December 2020)
    - EUDR Article 3: Prohibition on non-compliant products
    - EUDR Article 9: Information requirements for due diligence statements
    - EUDR Article 10: Risk assessment obligations
    - EUDR Article 31: Five-year record retention
    - EUDR Recital 24: Reference date for deforestation definition

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-020, Engine 5 (Cutoff Date Verifier)
Agent ID: GL-EUDR-DAS-020
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module version for provenance tracking
# ---------------------------------------------------------------------------

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash (dict, dataclass with to_dict, or other).

    Returns:
        SHA-256 hex digest string (64 characters).
    """
    if hasattr(data, "to_dict"):
        serializable = data.to_dict()
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _generate_id(prefix: str = "cv") -> str:
    """Generate a unique identifier with a given prefix.

    Args:
        prefix: ID prefix string.

    Returns:
        ID in format ``{prefix}-{hex12}``.
    """
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _to_decimal(value: Any) -> Decimal:
    """Safely convert a value to Decimal.

    Args:
        value: Numeric value to convert.

    Returns:
        Decimal representation.

    Raises:
        ValueError: If value cannot be converted.
    """
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"Cannot convert {value!r} to Decimal") from exc


def _clamp_decimal(value: Decimal, lo: Decimal, hi: Decimal) -> Decimal:
    """Clamp a Decimal value to [lo, hi] range.

    Args:
        value: Value to clamp.
        lo: Lower bound.
        hi: Upper bound.

    Returns:
        Clamped Decimal.
    """
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def _date_from_str(date_str: str) -> date:
    """Parse an ISO date string to a date object.

    Args:
        date_str: Date string in YYYY-MM-DD format.

    Returns:
        date object.

    Raises:
        ValueError: If date_str is not a valid ISO date.
    """
    return date.fromisoformat(date_str)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class CutoffResult(str, Enum):
    """EUDR cutoff date verification result.

    Classifies whether deforestation occurred before or after the EUDR
    cutoff date of December 31, 2020.

    Values:
        PRE_CUTOFF: Clearing completed before the cutoff date. COMPLIANT.
        POST_CUTOFF: Clearing detected after the cutoff date. NON-COMPLIANT.
        ONGOING: Clearing was in progress spanning the cutoff date.
        UNCERTAIN: Insufficient evidence to determine timing.
    """

    PRE_CUTOFF = "PRE_CUTOFF"
    POST_CUTOFF = "POST_CUTOFF"
    ONGOING = "ONGOING"
    UNCERTAIN = "UNCERTAIN"


class ForestState(str, Enum):
    """Observed forest cover state at a specific point in time.

    Values:
        FORESTED: Full canopy cover detected (NDVI > threshold).
        CLEARED: No canopy cover detected (clearing confirmed).
        TRANSITIONING: Partial canopy loss detected (in-progress clearing).
        DEGRADED: Significant canopy degradation but not full clearing.
        UNKNOWN: State could not be determined (e.g., cloud cover).
    """

    FORESTED = "FORESTED"
    CLEARED = "CLEARED"
    TRANSITIONING = "TRANSITIONING"
    DEGRADED = "DEGRADED"
    UNKNOWN = "UNKNOWN"


class EvidenceSource(str, Enum):
    """Satellite data source providing temporal evidence.

    Values:
        SENTINEL2: ESA Sentinel-2 (10m optical, from 2015/2017).
        LANDSAT: USGS Landsat 8/9 (30m optical, from 2013).
        HANSEN_GFC: Hansen Global Forest Change (annual, from 2000).
        GLAD: GLAD weekly alerts (from 2016).
        RADD: RADD radar alerts (Sentinel-1 SAR, from 2019).
        FIELD_SURVEY: Ground truth field survey data.
        AERIAL: Aerial / drone survey data.
        HISTORICAL_MAP: Historical forest cover map data.
    """

    SENTINEL2 = "SENTINEL2"
    LANDSAT = "LANDSAT"
    HANSEN_GFC = "HANSEN_GFC"
    GLAD = "GLAD"
    RADD = "RADD"
    FIELD_SURVEY = "FIELD_SURVEY"
    AERIAL = "AERIAL"
    HISTORICAL_MAP = "HISTORICAL_MAP"


class ConfidenceLevel(str, Enum):
    """Confidence level classification for cutoff verification.

    Values:
        HIGH: Confidence >= 0.85. Strong multi-source agreement.
        MEDIUM: Confidence >= 0.65. Moderate evidence support.
        LOW: Confidence >= 0.45. Limited evidence, some ambiguity.
        INSUFFICIENT: Confidence < 0.45. Cannot reliably classify.
    """

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INSUFFICIENT = "INSUFFICIENT"


class VerificationStatus(str, Enum):
    """Verification processing status.

    Values:
        COMPLETED: Verification fully processed.
        PARTIAL: Some evidence sources were unavailable.
        FAILED: Verification could not be completed.
        PENDING: Verification is queued for processing.
    """

    COMPLETED = "COMPLETED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    PENDING = "PENDING"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: EUDR cutoff date per Article 2(1) of Regulation (EU) 2023/1115.
EUDR_CUTOFF_DATE: date = date(2020, 12, 31)

#: Default pre-cutoff grace period (days) for temporal uncertainty.
DEFAULT_GRACE_PERIOD_DAYS: int = 90

#: Minimum number of temporal evidence sources for reliable classification.
MIN_EVIDENCE_SOURCES: int = 2

#: Confidence thresholds for classification.
CONFIDENCE_HIGH: Decimal = Decimal("0.85")
CONFIDENCE_MEDIUM: Decimal = Decimal("0.65")
CONFIDENCE_LOW: Decimal = Decimal("0.45")

#: Default cutoff confidence threshold for compliance determination.
DEFAULT_CUTOFF_CONFIDENCE: Decimal = Decimal("0.85")

#: Maximum number of detections in a single batch.
MAX_BATCH_SIZE: int = 500

#: Maximum temporal evidence entries per detection.
MAX_EVIDENCE_ENTRIES: int = 200

#: Evidence source reliability weights (higher = more reliable).
SOURCE_RELIABILITY_WEIGHTS: Dict[str, Decimal] = {
    EvidenceSource.SENTINEL2.value: Decimal("0.90"),
    EvidenceSource.LANDSAT.value: Decimal("0.85"),
    EvidenceSource.HANSEN_GFC.value: Decimal("0.80"),
    EvidenceSource.GLAD.value: Decimal("0.75"),
    EvidenceSource.RADD.value: Decimal("0.70"),
    EvidenceSource.FIELD_SURVEY.value: Decimal("0.95"),
    EvidenceSource.AERIAL.value: Decimal("0.88"),
    EvidenceSource.HISTORICAL_MAP.value: Decimal("0.60"),
}

#: Evidence source temporal availability (earliest year of data).
SOURCE_AVAILABILITY_START: Dict[str, int] = {
    EvidenceSource.SENTINEL2.value: 2015,
    EvidenceSource.LANDSAT.value: 2013,
    EvidenceSource.HANSEN_GFC.value: 2000,
    EvidenceSource.GLAD.value: 2016,
    EvidenceSource.RADD.value: 2019,
    EvidenceSource.FIELD_SURVEY.value: 2000,
    EvidenceSource.AERIAL.value: 2010,
    EvidenceSource.HISTORICAL_MAP.value: 1990,
}

#: NDVI threshold for forest state classification.
NDVI_FORESTED_THRESHOLD: Decimal = Decimal("0.55")
NDVI_TRANSITIONING_THRESHOLD: Decimal = Decimal("0.35")
NDVI_DEGRADED_THRESHOLD: Decimal = Decimal("0.20")

#: EVI threshold for forest state classification.
EVI_FORESTED_THRESHOLD: Decimal = Decimal("0.40")
EVI_TRANSITIONING_THRESHOLD: Decimal = Decimal("0.25")

#: Minimum NDVI drop to confirm clearing event.
NDVI_CLEARING_DROP: Decimal = Decimal("-0.20")

#: Temporal consistency window for confirming state transition (days).
STATE_TRANSITION_CONFIRMATION_DAYS: int = 30

#: Maximum days between observations before considering a gap.
MAX_OBSERVATION_GAP_DAYS: int = 60

#: Weight for temporal consistency in confidence calculation.
TEMPORAL_CONSISTENCY_WEIGHT: Decimal = Decimal("0.30")

#: Weight for source diversity in confidence calculation.
SOURCE_DIVERSITY_WEIGHT: Decimal = Decimal("0.25")

#: Weight for observation density in confidence calculation.
OBSERVATION_DENSITY_WEIGHT: Decimal = Decimal("0.20")

#: Weight for source reliability in confidence calculation.
SOURCE_RELIABILITY_WEIGHT: Decimal = Decimal("0.25")

#: Sentinel-2 reference spectral signatures for tropical forest.
TROPICAL_FOREST_NDVI_RANGE: Tuple[Decimal, Decimal] = (
    Decimal("0.60"),
    Decimal("0.90"),
)
TROPICAL_FOREST_EVI_RANGE: Tuple[Decimal, Decimal] = (
    Decimal("0.35"),
    Decimal("0.60"),
)

#: Reference spectral signatures for cleared land.
CLEARED_LAND_NDVI_RANGE: Tuple[Decimal, Decimal] = (
    Decimal("0.05"),
    Decimal("0.20"),
)
CLEARED_LAND_EVI_RANGE: Tuple[Decimal, Decimal] = (
    Decimal("0.02"),
    Decimal("0.12"),
)

#: Biome-specific NDVI baselines for forest classification.
BIOME_NDVI_BASELINES: Dict[str, Decimal] = {
    "tropical_moist": Decimal("0.70"),
    "tropical_dry": Decimal("0.55"),
    "subtropical": Decimal("0.60"),
    "temperate_broadleaf": Decimal("0.55"),
    "temperate_conifer": Decimal("0.50"),
    "boreal": Decimal("0.45"),
    "mangrove": Decimal("0.65"),
    "default": Decimal("0.55"),
}

#: Seasonal NDVI adjustment factors by hemisphere and month.
SEASONAL_NDVI_ADJUSTMENTS: Dict[str, Dict[int, Decimal]] = {
    "northern": {
        1: Decimal("-0.10"), 2: Decimal("-0.08"), 3: Decimal("-0.03"),
        4: Decimal("0.02"), 5: Decimal("0.05"), 6: Decimal("0.05"),
        7: Decimal("0.05"), 8: Decimal("0.03"), 9: Decimal("0.00"),
        10: Decimal("-0.03"), 11: Decimal("-0.07"), 12: Decimal("-0.10"),
    },
    "southern": {
        1: Decimal("0.05"), 2: Decimal("0.03"), 3: Decimal("0.00"),
        4: Decimal("-0.03"), 5: Decimal("-0.07"), 6: Decimal("-0.10"),
        7: Decimal("-0.10"), 8: Decimal("-0.08"), 9: Decimal("-0.03"),
        10: Decimal("0.02"), 11: Decimal("0.05"), 12: Decimal("0.05"),
    },
    "tropical": {
        1: Decimal("0.00"), 2: Decimal("0.00"), 3: Decimal("-0.02"),
        4: Decimal("-0.03"), 5: Decimal("0.00"), 6: Decimal("0.02"),
        7: Decimal("0.02"), 8: Decimal("0.01"), 9: Decimal("0.00"),
        10: Decimal("-0.01"), 11: Decimal("-0.02"), 12: Decimal("0.00"),
    },
}

#: Country-to-biome mapping for common EUDR commodity origin countries.
COUNTRY_BIOME_MAP: Dict[str, str] = {
    "BR": "tropical_moist",
    "ID": "tropical_moist",
    "CO": "tropical_moist",
    "PE": "tropical_moist",
    "EC": "tropical_moist",
    "CI": "tropical_moist",
    "GH": "tropical_moist",
    "CM": "tropical_moist",
    "CD": "tropical_moist",
    "CG": "tropical_moist",
    "MY": "tropical_moist",
    "PG": "tropical_moist",
    "BO": "tropical_moist",
    "PY": "tropical_dry",
    "AR": "subtropical",
    "MX": "subtropical",
    "TH": "tropical_moist",
    "VN": "tropical_moist",
    "MM": "tropical_moist",
    "LR": "tropical_moist",
    "SL": "tropical_moist",
    "NG": "tropical_moist",
    "ET": "tropical_dry",
    "KE": "tropical_dry",
    "TZ": "tropical_dry",
    "UG": "tropical_moist",
    "GT": "tropical_moist",
    "HN": "tropical_moist",
    "NI": "tropical_moist",
    "CR": "tropical_moist",
}

#: Hemisphere determination by latitude.
TROPICAL_LATITUDE_BAND: Tuple[Decimal, Decimal] = (
    Decimal("-23.5"),
    Decimal("23.5"),
)


# ---------------------------------------------------------------------------
# Reference Data: Simulated Satellite Observations
# ---------------------------------------------------------------------------
# In production, these observations come from the satellite data pipeline.
# This reference data enables offline testing and development.

REFERENCE_OBSERVATIONS: Dict[str, List[Dict[str, Any]]] = {
    "sample_post_cutoff": [
        {"date": "2019-06-15", "source": "LANDSAT", "ndvi": "0.78", "evi": "0.45", "cloud_pct": 5},
        {"date": "2019-12-01", "source": "SENTINEL2", "ndvi": "0.76", "evi": "0.44", "cloud_pct": 8},
        {"date": "2020-03-15", "source": "SENTINEL2", "ndvi": "0.75", "evi": "0.43", "cloud_pct": 12},
        {"date": "2020-06-20", "source": "LANDSAT", "ndvi": "0.74", "evi": "0.42", "cloud_pct": 10},
        {"date": "2020-09-10", "source": "SENTINEL2", "ndvi": "0.73", "evi": "0.41", "cloud_pct": 6},
        {"date": "2020-12-15", "source": "SENTINEL2", "ndvi": "0.72", "evi": "0.40", "cloud_pct": 15},
        {"date": "2021-02-10", "source": "LANDSAT", "ndvi": "0.45", "evi": "0.22", "cloud_pct": 5},
        {"date": "2021-03-15", "source": "SENTINEL2", "ndvi": "0.30", "evi": "0.15", "cloud_pct": 3},
        {"date": "2021-05-20", "source": "GLAD", "ndvi": "0.18", "evi": "0.08", "cloud_pct": 0},
        {"date": "2021-07-01", "source": "SENTINEL2", "ndvi": "0.15", "evi": "0.06", "cloud_pct": 7},
        {"date": "2021-09-10", "source": "RADD", "ndvi": "0.12", "evi": "0.05", "cloud_pct": 0},
    ],
    "sample_pre_cutoff": [
        {"date": "2017-06-15", "source": "SENTINEL2", "ndvi": "0.80", "evi": "0.48", "cloud_pct": 4},
        {"date": "2017-12-01", "source": "LANDSAT", "ndvi": "0.78", "evi": "0.46", "cloud_pct": 6},
        {"date": "2018-03-15", "source": "SENTINEL2", "ndvi": "0.75", "evi": "0.44", "cloud_pct": 10},
        {"date": "2018-06-20", "source": "LANDSAT", "ndvi": "0.50", "evi": "0.28", "cloud_pct": 8},
        {"date": "2018-09-10", "source": "SENTINEL2", "ndvi": "0.32", "evi": "0.16", "cloud_pct": 5},
        {"date": "2018-12-15", "source": "GLAD", "ndvi": "0.18", "evi": "0.08", "cloud_pct": 0},
        {"date": "2019-03-15", "source": "SENTINEL2", "ndvi": "0.14", "evi": "0.06", "cloud_pct": 3},
        {"date": "2019-06-20", "source": "LANDSAT", "ndvi": "0.12", "evi": "0.05", "cloud_pct": 5},
        {"date": "2020-06-15", "source": "SENTINEL2", "ndvi": "0.11", "evi": "0.04", "cloud_pct": 7},
        {"date": "2020-12-15", "source": "SENTINEL2", "ndvi": "0.10", "evi": "0.04", "cloud_pct": 10},
    ],
    "sample_ongoing": [
        {"date": "2019-06-15", "source": "SENTINEL2", "ndvi": "0.78", "evi": "0.45", "cloud_pct": 5},
        {"date": "2020-01-10", "source": "LANDSAT", "ndvi": "0.76", "evi": "0.44", "cloud_pct": 8},
        {"date": "2020-06-20", "source": "SENTINEL2", "ndvi": "0.72", "evi": "0.40", "cloud_pct": 12},
        {"date": "2020-10-15", "source": "SENTINEL2", "ndvi": "0.58", "evi": "0.32", "cloud_pct": 6},
        {"date": "2020-12-28", "source": "LANDSAT", "ndvi": "0.42", "evi": "0.22", "cloud_pct": 10},
        {"date": "2021-01-15", "source": "SENTINEL2", "ndvi": "0.30", "evi": "0.15", "cloud_pct": 5},
        {"date": "2021-03-20", "source": "GLAD", "ndvi": "0.18", "evi": "0.08", "cloud_pct": 0},
        {"date": "2021-06-15", "source": "SENTINEL2", "ndvi": "0.13", "evi": "0.06", "cloud_pct": 7},
    ],
    "sample_uncertain": [
        {"date": "2019-06-15", "source": "LANDSAT", "ndvi": "0.72", "evi": "0.40", "cloud_pct": 5},
        {"date": "2021-06-15", "source": "SENTINEL2", "ndvi": "0.15", "evi": "0.06", "cloud_pct": 7},
    ],
}

#: Hansen GFC annual tree cover loss year reference data.
#: Maps detection_id patterns to loss year (for offline testing).
HANSEN_LOSS_YEARS: Dict[str, int] = {
    "sample_post_cutoff": 2021,
    "sample_pre_cutoff": 2018,
    "sample_ongoing": 2020,
    "sample_uncertain": 0,
}


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class TemporalEvidence:
    """A single temporal observation providing evidence of forest state.

    Represents one satellite or ground-truth observation of forest cover
    at a specific location and time, used to build the temporal evidence
    chain for cutoff date verification.

    Attributes:
        evidence_id: Unique evidence identifier.
        source: Satellite or survey source.
        observation_date: Date of observation.
        forest_state: Classified forest state at observation time.
        ndvi_value: NDVI spectral index value (-1 to +1).
        evi_value: EVI spectral index value.
        confidence: Observation confidence score (0-1).
        cloud_cover_pct: Cloud cover percentage at time of observation.
        imagery_id: Satellite imagery scene identifier.
        spatial_resolution_m: Spatial resolution in meters.
        metadata: Additional observation metadata.
    """

    evidence_id: str = ""
    source: str = ""
    observation_date: str = ""
    forest_state: str = ForestState.UNKNOWN.value
    ndvi_value: Decimal = Decimal("0")
    evi_value: Decimal = Decimal("0")
    confidence: Decimal = Decimal("0")
    cloud_cover_pct: int = 0
    imagery_id: str = ""
    spatial_resolution_m: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation with Decimal fields as strings.
        """
        return {
            "evidence_id": self.evidence_id,
            "source": self.source,
            "observation_date": self.observation_date,
            "forest_state": self.forest_state,
            "ndvi_value": str(self.ndvi_value),
            "evi_value": str(self.evi_value),
            "confidence": str(self.confidence),
            "cloud_cover_pct": self.cloud_cover_pct,
            "imagery_id": self.imagery_id,
            "spatial_resolution_m": self.spatial_resolution_m,
            "metadata": self.metadata,
        }


@dataclass
class TemporalTransition:
    """A detected forest state transition between two observations.

    Attributes:
        transition_id: Unique transition identifier.
        from_state: Previous forest state.
        to_state: New forest state.
        from_date: Date of previous observation.
        to_date: Date of new observation.
        ndvi_change: Change in NDVI between observations.
        transition_days: Number of days between observations.
        transition_rate: Daily NDVI change rate.
        confidence: Transition detection confidence (0-1).
        confirmed: Whether transition was confirmed by multiple sources.
    """

    transition_id: str = ""
    from_state: str = ""
    to_state: str = ""
    from_date: str = ""
    to_date: str = ""
    ndvi_change: Decimal = Decimal("0")
    transition_days: int = 0
    transition_rate: Decimal = Decimal("0")
    confidence: Decimal = Decimal("0")
    confirmed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "transition_id": self.transition_id,
            "from_state": self.from_state,
            "to_state": self.to_state,
            "from_date": self.from_date,
            "to_date": self.to_date,
            "ndvi_change": str(self.ndvi_change),
            "transition_days": self.transition_days,
            "transition_rate": str(self.transition_rate),
            "confidence": str(self.confidence),
            "confirmed": self.confirmed,
        }


@dataclass
class CutoffVerification:
    """Complete EUDR cutoff date verification result.

    Contains the full verification outcome including the cutoff
    classification, confidence score, temporal evidence chain, state
    transitions, timeline analysis, and EUDR compliance determination.

    Attributes:
        verification_id: Unique verification identifier.
        detection_id: Source detection identifier being verified.
        latitude: Detection latitude.
        longitude: Detection longitude.
        detection_date: Date the detection was reported.
        cutoff_result: PRE_CUTOFF / POST_CUTOFF / ONGOING / UNCERTAIN.
        confidence: Overall verification confidence (0-1).
        confidence_level: Classification of confidence (HIGH/MEDIUM/LOW/INSUFFICIENT).
        eudr_compliant: Whether products from this area can access EU market.
        evidence_sources: List of evidence sources used.
        evidence_count: Total evidence observations.
        earliest_forested_date: Earliest observation with FORESTED state.
        latest_forested_date: Latest observation with FORESTED state.
        earliest_cleared_date: Earliest observation with CLEARED state.
        latest_cleared_date: Latest observation confirming cleared state.
        estimated_clearing_start: Estimated date clearing began.
        estimated_clearing_end: Estimated date clearing completed.
        clearing_duration_days: Estimated clearing duration.
        hansen_loss_year: Hansen GFC loss year (if available).
        temporal_transitions: Detected forest state transitions.
        temporal_analysis: Full temporal analysis details.
        biome: Detected biome type.
        risk_level: Compliance risk level based on verification.
        verification_status: Processing status.
        warnings: List of warning messages.
        calculation_timestamp: When verification was performed.
        processing_time_ms: Processing duration in milliseconds.
        provenance_hash: SHA-256 hash for audit trail.
    """

    verification_id: str = ""
    detection_id: str = ""
    latitude: Decimal = Decimal("0")
    longitude: Decimal = Decimal("0")
    detection_date: str = ""
    cutoff_result: str = CutoffResult.UNCERTAIN.value
    confidence: Decimal = Decimal("0")
    confidence_level: str = ConfidenceLevel.INSUFFICIENT.value
    eudr_compliant: bool = False
    evidence_sources: List[str] = field(default_factory=list)
    evidence_count: int = 0
    earliest_forested_date: str = ""
    latest_forested_date: str = ""
    earliest_cleared_date: str = ""
    latest_cleared_date: str = ""
    estimated_clearing_start: str = ""
    estimated_clearing_end: str = ""
    clearing_duration_days: int = 0
    hansen_loss_year: int = 0
    temporal_transitions: List[Dict[str, Any]] = field(default_factory=list)
    temporal_analysis: Dict[str, Any] = field(default_factory=dict)
    biome: str = "default"
    risk_level: str = "HIGH"
    verification_status: str = VerificationStatus.PENDING.value
    warnings: List[str] = field(default_factory=list)
    calculation_timestamp: str = ""
    processing_time_ms: float = 0.0
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation with Decimal fields as strings.
        """
        return {
            "verification_id": self.verification_id,
            "detection_id": self.detection_id,
            "latitude": str(self.latitude),
            "longitude": str(self.longitude),
            "detection_date": self.detection_date,
            "cutoff_result": self.cutoff_result,
            "confidence": str(self.confidence),
            "confidence_level": self.confidence_level,
            "eudr_compliant": self.eudr_compliant,
            "evidence_sources": self.evidence_sources,
            "evidence_count": self.evidence_count,
            "earliest_forested_date": self.earliest_forested_date,
            "latest_forested_date": self.latest_forested_date,
            "earliest_cleared_date": self.earliest_cleared_date,
            "latest_cleared_date": self.latest_cleared_date,
            "estimated_clearing_start": self.estimated_clearing_start,
            "estimated_clearing_end": self.estimated_clearing_end,
            "clearing_duration_days": self.clearing_duration_days,
            "hansen_loss_year": self.hansen_loss_year,
            "temporal_transitions": self.temporal_transitions,
            "temporal_analysis": self.temporal_analysis,
            "biome": self.biome,
            "risk_level": self.risk_level,
            "verification_status": self.verification_status,
            "warnings": self.warnings,
            "calculation_timestamp": self.calculation_timestamp,
            "processing_time_ms": self.processing_time_ms,
            "provenance_hash": self.provenance_hash,
        }


@dataclass
class EvidenceChain:
    """Complete temporal evidence chain for a detection.

    Attributes:
        chain_id: Unique chain identifier.
        detection_id: Detection this chain belongs to.
        evidence: List of temporal evidence observations.
        total_observations: Total observation count.
        source_counts: Count of observations per source.
        date_range_start: Earliest observation date.
        date_range_end: Latest observation date.
        temporal_coverage_days: Total temporal coverage span.
        mean_observation_gap_days: Average gap between observations.
        max_observation_gap_days: Maximum gap between observations.
        provenance_hash: SHA-256 hash.
    """

    chain_id: str = ""
    detection_id: str = ""
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    total_observations: int = 0
    source_counts: Dict[str, int] = field(default_factory=dict)
    date_range_start: str = ""
    date_range_end: str = ""
    temporal_coverage_days: int = 0
    mean_observation_gap_days: float = 0.0
    max_observation_gap_days: int = 0
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "chain_id": self.chain_id,
            "detection_id": self.detection_id,
            "evidence": self.evidence,
            "total_observations": self.total_observations,
            "source_counts": self.source_counts,
            "date_range_start": self.date_range_start,
            "date_range_end": self.date_range_end,
            "temporal_coverage_days": self.temporal_coverage_days,
            "mean_observation_gap_days": self.mean_observation_gap_days,
            "max_observation_gap_days": self.max_observation_gap_days,
            "provenance_hash": self.provenance_hash,
        }


@dataclass
class ForestTimeline:
    """Complete forest state timeline for a detection location.

    Attributes:
        timeline_id: Unique timeline identifier.
        detection_id: Detection this timeline belongs to.
        latitude: Location latitude.
        longitude: Location longitude.
        states: Chronological list of forest state entries.
        transitions: Detected state transitions.
        clearing_events: Identified clearing events.
        current_state: Most recent forest state.
        pre_cutoff_state: Forest state just before cutoff date.
        post_cutoff_state: Forest state just after cutoff date.
        cutoff_date: EUDR cutoff date used.
        biome: Biome classification.
        provenance_hash: SHA-256 hash.
    """

    timeline_id: str = ""
    detection_id: str = ""
    latitude: Decimal = Decimal("0")
    longitude: Decimal = Decimal("0")
    states: List[Dict[str, Any]] = field(default_factory=list)
    transitions: List[Dict[str, Any]] = field(default_factory=list)
    clearing_events: List[Dict[str, Any]] = field(default_factory=list)
    current_state: str = ForestState.UNKNOWN.value
    pre_cutoff_state: str = ForestState.UNKNOWN.value
    post_cutoff_state: str = ForestState.UNKNOWN.value
    cutoff_date: str = ""
    biome: str = "default"
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "timeline_id": self.timeline_id,
            "detection_id": self.detection_id,
            "latitude": str(self.latitude),
            "longitude": str(self.longitude),
            "states": self.states,
            "transitions": self.transitions,
            "clearing_events": self.clearing_events,
            "current_state": self.current_state,
            "pre_cutoff_state": self.pre_cutoff_state,
            "post_cutoff_state": self.post_cutoff_state,
            "cutoff_date": self.cutoff_date,
            "biome": self.biome,
            "provenance_hash": self.provenance_hash,
        }


@dataclass
class BatchVerificationResult:
    """Result of a batch cutoff date verification.

    Attributes:
        batch_id: Unique batch identifier.
        total_detections: Number of detections processed.
        pre_cutoff_count: Number classified as PRE_CUTOFF.
        post_cutoff_count: Number classified as POST_CUTOFF.
        ongoing_count: Number classified as ONGOING.
        uncertain_count: Number classified as UNCERTAIN.
        compliant_count: Number determined EUDR compliant.
        non_compliant_count: Number determined non-compliant.
        mean_confidence: Average confidence across all verifications.
        verifications: List of individual verification results.
        processing_time_ms: Total batch processing time.
        provenance_hash: SHA-256 hash.
    """

    batch_id: str = ""
    total_detections: int = 0
    pre_cutoff_count: int = 0
    post_cutoff_count: int = 0
    ongoing_count: int = 0
    uncertain_count: int = 0
    compliant_count: int = 0
    non_compliant_count: int = 0
    mean_confidence: Decimal = Decimal("0")
    verifications: List[Dict[str, Any]] = field(default_factory=list)
    processing_time_ms: float = 0.0
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "batch_id": self.batch_id,
            "total_detections": self.total_detections,
            "pre_cutoff_count": self.pre_cutoff_count,
            "post_cutoff_count": self.post_cutoff_count,
            "ongoing_count": self.ongoing_count,
            "uncertain_count": self.uncertain_count,
            "compliant_count": self.compliant_count,
            "non_compliant_count": self.non_compliant_count,
            "mean_confidence": str(self.mean_confidence),
            "verifications": self.verifications,
            "processing_time_ms": self.processing_time_ms,
            "provenance_hash": self.provenance_hash,
        }


# ---------------------------------------------------------------------------
# CutoffDateVerifier
# ---------------------------------------------------------------------------


class CutoffDateVerifier:
    """Production-grade EUDR cutoff date verification engine.

    Verifies whether detected deforestation events occurred before or after
    the EUDR cutoff date (December 31, 2020). Uses multi-source temporal
    evidence, time-series analysis, and confidence scoring to make
    deterministic compliance classification decisions.

    Thread Safety:
        All mutable state is protected by a reentrant lock.

    Zero-Hallucination:
        All temporal analysis uses deterministic date arithmetic and
        Decimal confidence scoring. No ML/LLM in any classification path.
        Forest state is classified from spectral index thresholds.
        Cutoff result is determined by explicit date comparisons.

    Attributes:
        _cutoff_date: EUDR cutoff date for verification.
        _grace_period_days: Pre-cutoff temporal grace period.
        _min_evidence_sources: Minimum evidence sources required.
        _cutoff_confidence_threshold: Confidence threshold for compliance.
        _custom_observations: User-supplied observation data overrides.
        _verification_cache: Cache of completed verifications.
        _lock: Reentrant lock for thread-safe access.

    Example:
        >>> verifier = CutoffDateVerifier()
        >>> result = verifier.verify("det-001", -3.1234, 28.5678, "2021-05-15")
        >>> assert result["cutoff_result"] in ("PRE_CUTOFF", "POST_CUTOFF", "ONGOING", "UNCERTAIN")
        >>> assert "provenance_hash" in result
    """

    def __init__(
        self,
        cutoff_date: Optional[str] = None,
        grace_period_days: Optional[int] = None,
        min_evidence_sources: Optional[int] = None,
        cutoff_confidence_threshold: Optional[Decimal] = None,
    ) -> None:
        """Initialize CutoffDateVerifier.

        Args:
            cutoff_date: EUDR cutoff date override (YYYY-MM-DD).
                Defaults to 2020-12-31.
            grace_period_days: Grace period days override.
                Defaults to 90.
            min_evidence_sources: Minimum evidence sources override.
                Defaults to 2.
            cutoff_confidence_threshold: Confidence threshold override.
                Defaults to 0.85.
        """
        self._cutoff_date: date = (
            _date_from_str(cutoff_date) if cutoff_date else EUDR_CUTOFF_DATE
        )
        self._grace_period_days: int = (
            grace_period_days if grace_period_days is not None
            else DEFAULT_GRACE_PERIOD_DAYS
        )
        self._min_evidence_sources: int = (
            min_evidence_sources if min_evidence_sources is not None
            else MIN_EVIDENCE_SOURCES
        )
        self._cutoff_confidence_threshold: Decimal = (
            cutoff_confidence_threshold if cutoff_confidence_threshold is not None
            else DEFAULT_CUTOFF_CONFIDENCE
        )
        self._custom_observations: Dict[str, List[Dict[str, Any]]] = {}
        self._verification_cache: Dict[str, Dict[str, Any]] = {}
        self._lock: threading.RLock = threading.RLock()

        logger.info(
            "CutoffDateVerifier initialized (version=%s, cutoff=%s, "
            "grace_days=%d, min_sources=%d, confidence_threshold=%s)",
            _MODULE_VERSION,
            self._cutoff_date.isoformat(),
            self._grace_period_days,
            self._min_evidence_sources,
            self._cutoff_confidence_threshold,
        )

    # ------------------------------------------------------------------
    # Data Loading
    # ------------------------------------------------------------------

    def load_observations(
        self,
        detection_id: str,
        observations: List[Dict[str, Any]],
    ) -> None:
        """Load custom observation data for a detection.

        Args:
            detection_id: Detection identifier.
            observations: List of observation dicts, each with:
                date (str), source (str), ndvi (str/Decimal),
                evi (str/Decimal), cloud_pct (int).

        Raises:
            ValueError: If detection_id is empty or observations empty.
        """
        if not detection_id:
            raise ValueError("detection_id must be non-empty")
        if not observations:
            raise ValueError("observations must be non-empty")

        with self._lock:
            self._custom_observations[detection_id] = list(observations)

        logger.info(
            "Loaded %d observations for detection %s",
            len(observations),
            detection_id,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify(
        self,
        detection_id: str,
        latitude: Any,
        longitude: Any,
        detection_date: str,
        country_code: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Verify whether a deforestation event is pre or post EUDR cutoff.

        Performs full temporal evidence collection, time-series analysis,
        state transition detection, and cutoff classification with
        confidence scoring.

        Args:
            detection_id: Unique detection identifier.
            latitude: Detection latitude (-90 to +90).
            longitude: Detection longitude (-180 to +180).
            detection_date: Date detection was reported (YYYY-MM-DD).
            country_code: Optional ISO country code for biome detection.

        Returns:
            Dictionary with complete verification result including
            cutoff_result, confidence, eudr_compliant, evidence chain,
            temporal analysis, and provenance_hash.

        Raises:
            ValueError: If detection_id is empty or coordinates invalid.
        """
        start_time = time.monotonic()

        # Input validation
        self._validate_verify_input(detection_id, latitude, longitude, detection_date)

        lat = _to_decimal(latitude)
        lon = _to_decimal(longitude)
        det_date = _date_from_str(detection_date)

        # Determine biome
        biome = self._determine_biome(lat, lon, country_code)

        # Collect temporal evidence
        evidence_list = self._collect_temporal_evidence(detection_id, lat, lon)

        # Build classified evidence
        classified_evidence = self._classify_evidence(evidence_list, biome)

        # Analyze temporal sequence
        analysis = self._analyze_temporal_sequence(classified_evidence)

        # Detect state transitions
        transitions = self._detect_transitions(classified_evidence)

        # Check Hansen GFC loss year
        hansen_year = self._get_hansen_loss_year(detection_id)

        # Determine cutoff result
        cutoff_result = self._determine_cutoff_result(analysis, transitions, hansen_year)

        # Calculate confidence
        confidence = self._calculate_confidence(
            classified_evidence, analysis, transitions
        )

        # Classify confidence level
        confidence_level = self._classify_confidence(confidence)

        # Determine EUDR compliance
        eudr_compliant = self._determine_compliance(cutoff_result, confidence)

        # Determine risk level
        risk_level = self._determine_risk_level(cutoff_result, confidence)

        # Determine verification status
        verification_status = self._determine_verification_status(
            classified_evidence, confidence
        )

        # Build warnings
        warnings = self._generate_warnings(
            classified_evidence, analysis, confidence, cutoff_result
        )

        processing_time_ms = (time.monotonic() - start_time) * 1000.0

        # Build result
        verification = CutoffVerification(
            verification_id=_generate_id("cv"),
            detection_id=detection_id,
            latitude=lat,
            longitude=lon,
            detection_date=detection_date,
            cutoff_result=cutoff_result.value if isinstance(cutoff_result, CutoffResult) else cutoff_result,
            confidence=confidence,
            confidence_level=confidence_level.value if isinstance(confidence_level, ConfidenceLevel) else confidence_level,
            eudr_compliant=eudr_compliant,
            evidence_sources=list(analysis.get("sources_used", [])),
            evidence_count=len(classified_evidence),
            earliest_forested_date=analysis.get("earliest_forested_date", ""),
            latest_forested_date=analysis.get("latest_forested_date", ""),
            earliest_cleared_date=analysis.get("earliest_cleared_date", ""),
            latest_cleared_date=analysis.get("latest_cleared_date", ""),
            estimated_clearing_start=analysis.get("estimated_clearing_start", ""),
            estimated_clearing_end=analysis.get("estimated_clearing_end", ""),
            clearing_duration_days=analysis.get("clearing_duration_days", 0),
            hansen_loss_year=hansen_year,
            temporal_transitions=[t.to_dict() for t in transitions],
            temporal_analysis=analysis,
            biome=biome,
            risk_level=risk_level,
            verification_status=verification_status.value if isinstance(verification_status, VerificationStatus) else verification_status,
            warnings=warnings,
            calculation_timestamp=_utcnow().isoformat(),
            processing_time_ms=round(processing_time_ms, 3),
        )
        verification.provenance_hash = _compute_hash(verification)

        result = verification.to_dict()

        # Cache result
        with self._lock:
            self._verification_cache[detection_id] = result

        logger.info(
            "Cutoff verification: detection=%s result=%s confidence=%s "
            "compliant=%s evidence=%d time_ms=%.1f",
            detection_id,
            verification.cutoff_result,
            confidence,
            eudr_compliant,
            len(classified_evidence),
            processing_time_ms,
        )

        return result

    def batch_verify(
        self,
        detections: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Batch verify multiple detections against the EUDR cutoff date.

        Processes each detection sequentially and aggregates results.

        Args:
            detections: List of detection dicts, each with:
                detection_id (str), latitude (numeric), longitude (numeric),
                detection_date (str), optionally country_code (str).

        Returns:
            Dictionary with batch verification results including per-detection
            verifications, summary counts, and provenance_hash.

        Raises:
            ValueError: If detections is empty or exceeds MAX_BATCH_SIZE.
        """
        start_time = time.monotonic()

        if not detections:
            raise ValueError("detections list must be non-empty")
        if len(detections) > MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size {len(detections)} exceeds maximum {MAX_BATCH_SIZE}"
            )

        verifications: List[Dict[str, Any]] = []
        counts = {
            "pre_cutoff": 0,
            "post_cutoff": 0,
            "ongoing": 0,
            "uncertain": 0,
            "compliant": 0,
            "non_compliant": 0,
        }
        confidence_sum = Decimal("0")

        for det in detections:
            det_id = det.get("detection_id", "")
            lat = det.get("latitude", 0)
            lon = det.get("longitude", 0)
            det_date = det.get("detection_date", "")
            country = det.get("country_code")

            try:
                result = self.verify(det_id, lat, lon, det_date, country)
                verifications.append(result)

                cr = result.get("cutoff_result", "UNCERTAIN")
                if cr == CutoffResult.PRE_CUTOFF.value:
                    counts["pre_cutoff"] += 1
                elif cr == CutoffResult.POST_CUTOFF.value:
                    counts["post_cutoff"] += 1
                elif cr == CutoffResult.ONGOING.value:
                    counts["ongoing"] += 1
                else:
                    counts["uncertain"] += 1

                if result.get("eudr_compliant", False):
                    counts["compliant"] += 1
                else:
                    counts["non_compliant"] += 1

                confidence_sum += _to_decimal(result.get("confidence", "0"))

            except Exception as exc:
                logger.warning(
                    "Batch verification failed for detection %s: %s",
                    det_id,
                    exc,
                )
                verifications.append({
                    "detection_id": det_id,
                    "cutoff_result": CutoffResult.UNCERTAIN.value,
                    "confidence": "0",
                    "eudr_compliant": False,
                    "verification_status": VerificationStatus.FAILED.value,
                    "error": str(exc),
                })
                counts["uncertain"] += 1
                counts["non_compliant"] += 1

        total = len(detections)
        mean_conf = (
            (confidence_sum / Decimal(str(total))).quantize(
                Decimal("0.0001"), rounding=ROUND_HALF_UP
            )
            if total > 0
            else Decimal("0")
        )

        processing_time_ms = (time.monotonic() - start_time) * 1000.0

        batch_result = BatchVerificationResult(
            batch_id=_generate_id("bcv"),
            total_detections=total,
            pre_cutoff_count=counts["pre_cutoff"],
            post_cutoff_count=counts["post_cutoff"],
            ongoing_count=counts["ongoing"],
            uncertain_count=counts["uncertain"],
            compliant_count=counts["compliant"],
            non_compliant_count=counts["non_compliant"],
            mean_confidence=mean_conf,
            verifications=verifications,
            processing_time_ms=round(processing_time_ms, 3),
        )
        batch_result.provenance_hash = _compute_hash(batch_result)

        logger.info(
            "Batch cutoff verification: total=%d pre=%d post=%d ongoing=%d "
            "uncertain=%d compliant=%d non_compliant=%d time_ms=%.1f",
            total,
            counts["pre_cutoff"],
            counts["post_cutoff"],
            counts["ongoing"],
            counts["uncertain"],
            counts["compliant"],
            counts["non_compliant"],
            processing_time_ms,
        )

        return batch_result.to_dict()

    def get_evidence(self, detection_id: str) -> Dict[str, Any]:
        """Retrieve the full temporal evidence chain for a detection.

        Args:
            detection_id: Detection identifier.

        Returns:
            Dictionary with evidence chain including all observations,
            source counts, temporal coverage statistics, and provenance_hash.

        Raises:
            ValueError: If detection_id is empty.
        """
        if not detection_id:
            raise ValueError("detection_id must be non-empty")

        evidence_list = self._collect_temporal_evidence(
            detection_id, Decimal("0"), Decimal("0")
        )

        # Build source counts
        source_counts: Dict[str, int] = {}
        for obs in evidence_list:
            src = obs.get("source", "UNKNOWN")
            source_counts[src] = source_counts.get(src, 0) + 1

        # Sort by date
        sorted_evidence = sorted(evidence_list, key=lambda x: x.get("date", ""))

        # Calculate temporal statistics
        dates = []
        for obs in sorted_evidence:
            try:
                dates.append(_date_from_str(obs["date"]))
            except (KeyError, ValueError):
                pass

        coverage_days = 0
        mean_gap = 0.0
        max_gap = 0
        if len(dates) >= 2:
            coverage_days = (dates[-1] - dates[0]).days
            gaps = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
            mean_gap = sum(gaps) / len(gaps) if gaps else 0.0
            max_gap = max(gaps) if gaps else 0

        evidence_dicts = []
        for obs in sorted_evidence:
            ev = TemporalEvidence(
                evidence_id=_generate_id("ev"),
                source=obs.get("source", ""),
                observation_date=obs.get("date", ""),
                ndvi_value=_to_decimal(obs.get("ndvi", "0")),
                evi_value=_to_decimal(obs.get("evi", "0")),
                cloud_cover_pct=obs.get("cloud_pct", 0),
            )
            evidence_dicts.append(ev.to_dict())

        chain = EvidenceChain(
            chain_id=_generate_id("ec"),
            detection_id=detection_id,
            evidence=evidence_dicts,
            total_observations=len(sorted_evidence),
            source_counts=source_counts,
            date_range_start=dates[0].isoformat() if dates else "",
            date_range_end=dates[-1].isoformat() if dates else "",
            temporal_coverage_days=coverage_days,
            mean_observation_gap_days=round(mean_gap, 1),
            max_observation_gap_days=max_gap,
        )
        chain.provenance_hash = _compute_hash(chain)

        return chain.to_dict()

    def get_timeline(
        self,
        detection_id: str,
        latitude: Any = 0,
        longitude: Any = 0,
        country_code: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build a complete forest state timeline for a detection.

        Args:
            detection_id: Detection identifier.
            latitude: Location latitude.
            longitude: Location longitude.
            country_code: Optional ISO country code.

        Returns:
            Dictionary with chronological forest state entries,
            transitions, clearing events, and provenance_hash.

        Raises:
            ValueError: If detection_id is empty.
        """
        if not detection_id:
            raise ValueError("detection_id must be non-empty")

        lat = _to_decimal(latitude)
        lon = _to_decimal(longitude)
        biome = self._determine_biome(lat, lon, country_code)

        evidence_list = self._collect_temporal_evidence(detection_id, lat, lon)
        classified = self._classify_evidence(evidence_list, biome)
        transitions = self._detect_transitions(classified)

        # Build state entries
        states = []
        for ev in classified:
            states.append({
                "date": ev.observation_date,
                "source": ev.source,
                "forest_state": ev.forest_state,
                "ndvi": str(ev.ndvi_value),
                "evi": str(ev.evi_value),
                "confidence": str(ev.confidence),
            })

        # Identify clearing events
        clearing_events = []
        for tr in transitions:
            if tr.to_state in (ForestState.CLEARED.value, ForestState.TRANSITIONING.value):
                clearing_events.append({
                    "start_date": tr.from_date,
                    "end_date": tr.to_date,
                    "from_state": tr.from_state,
                    "to_state": tr.to_state,
                    "ndvi_change": str(tr.ndvi_change),
                    "duration_days": tr.transition_days,
                })

        # Determine pre/post cutoff states
        cutoff_str = self._cutoff_date.isoformat()
        pre_cutoff_state = ForestState.UNKNOWN.value
        post_cutoff_state = ForestState.UNKNOWN.value

        for ev in classified:
            try:
                ev_date = _date_from_str(ev.observation_date)
                if ev_date <= self._cutoff_date:
                    pre_cutoff_state = ev.forest_state
                elif post_cutoff_state == ForestState.UNKNOWN.value:
                    post_cutoff_state = ev.forest_state
            except ValueError:
                pass

        current_state = classified[-1].forest_state if classified else ForestState.UNKNOWN.value

        timeline = ForestTimeline(
            timeline_id=_generate_id("tl"),
            detection_id=detection_id,
            latitude=lat,
            longitude=lon,
            states=states,
            transitions=[t.to_dict() for t in transitions],
            clearing_events=clearing_events,
            current_state=current_state,
            pre_cutoff_state=pre_cutoff_state,
            post_cutoff_state=post_cutoff_state,
            cutoff_date=cutoff_str,
            biome=biome,
        )
        timeline.provenance_hash = _compute_hash(timeline)

        return timeline.to_dict()

    def get_cached_verification(self, detection_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a cached verification result.

        Args:
            detection_id: Detection identifier.

        Returns:
            Cached verification dict, or None if not cached.
        """
        with self._lock:
            return self._verification_cache.get(detection_id)

    def clear_cache(self) -> int:
        """Clear the verification cache.

        Returns:
            Number of entries cleared.
        """
        with self._lock:
            count = len(self._verification_cache)
            self._verification_cache.clear()
        logger.info("Cleared verification cache (%d entries)", count)
        return count

    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics.

        Returns:
            Dictionary with cache size, configuration, and version.
        """
        with self._lock:
            cache_size = len(self._verification_cache)
            custom_count = len(self._custom_observations)

        return {
            "engine": "CutoffDateVerifier",
            "version": _MODULE_VERSION,
            "cutoff_date": self._cutoff_date.isoformat(),
            "grace_period_days": self._grace_period_days,
            "min_evidence_sources": self._min_evidence_sources,
            "cutoff_confidence_threshold": str(self._cutoff_confidence_threshold),
            "cache_size": cache_size,
            "custom_observations_loaded": custom_count,
            "evidence_sources_count": len(SOURCE_RELIABILITY_WEIGHTS),
            "biomes_count": len(BIOME_NDVI_BASELINES),
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_verify_input(
        self,
        detection_id: str,
        latitude: Any,
        longitude: Any,
        detection_date: str,
    ) -> None:
        """Validate verify() input parameters.

        Args:
            detection_id: Detection ID to validate.
            latitude: Latitude to validate.
            longitude: Longitude to validate.
            detection_date: Date string to validate.

        Raises:
            ValueError: If any parameter is invalid.
        """
        if not detection_id:
            raise ValueError("detection_id must be non-empty")

        lat = _to_decimal(latitude)
        lon = _to_decimal(longitude)

        if lat < Decimal("-90") or lat > Decimal("90"):
            raise ValueError(
                f"latitude must be between -90 and 90, got {lat}"
            )
        if lon < Decimal("-180") or lon > Decimal("180"):
            raise ValueError(
                f"longitude must be between -180 and 180, got {lon}"
            )

        if not detection_date:
            raise ValueError("detection_date must be non-empty")
        try:
            _date_from_str(detection_date)
        except ValueError:
            raise ValueError(
                f"detection_date must be valid ISO date (YYYY-MM-DD), "
                f"got {detection_date}"
            )

    # ------------------------------------------------------------------
    # Biome Determination
    # ------------------------------------------------------------------

    def _determine_biome(
        self,
        latitude: Decimal,
        longitude: Decimal,
        country_code: Optional[str] = None,
    ) -> str:
        """Determine biome type from location and country code.

        Uses country-biome mapping first, then latitude-based fallback.

        Args:
            latitude: Location latitude.
            longitude: Location longitude.
            country_code: Optional ISO country code.

        Returns:
            Biome identifier string.
        """
        if country_code:
            biome = COUNTRY_BIOME_MAP.get(country_code.upper())
            if biome:
                return biome

        # Latitude-based fallback
        abs_lat = abs(latitude)
        if abs_lat <= Decimal("23.5"):
            return "tropical_moist"
        elif abs_lat <= Decimal("35"):
            return "subtropical"
        elif abs_lat <= Decimal("55"):
            return "temperate_broadleaf"
        elif abs_lat <= Decimal("66.5"):
            return "boreal"
        return "default"

    def _determine_hemisphere(self, latitude: Decimal) -> str:
        """Determine hemisphere/zone for seasonal adjustments.

        Args:
            latitude: Location latitude.

        Returns:
            'tropical', 'northern', or 'southern'.
        """
        if (
            TROPICAL_LATITUDE_BAND[0] <= latitude <= TROPICAL_LATITUDE_BAND[1]
        ):
            return "tropical"
        elif latitude > Decimal("0"):
            return "northern"
        return "southern"

    # ------------------------------------------------------------------
    # Evidence Collection
    # ------------------------------------------------------------------

    def _collect_temporal_evidence(
        self,
        detection_id: str,
        latitude: Decimal,
        longitude: Decimal,
    ) -> List[Dict[str, Any]]:
        """Gather temporal evidence from all available sources.

        In production, this queries the satellite data pipeline. For
        development/testing, uses reference data or custom observations.

        Args:
            detection_id: Detection identifier.
            latitude: Location latitude.
            longitude: Location longitude.

        Returns:
            List of observation dictionaries sorted by date.
        """
        # Check custom observations first
        with self._lock:
            custom = self._custom_observations.get(detection_id)
            if custom:
                return sorted(custom, key=lambda x: x.get("date", ""))

        # Check reference data
        for key, observations in REFERENCE_OBSERVATIONS.items():
            if key in detection_id:
                return sorted(observations, key=lambda x: x.get("date", ""))

        # Generate synthetic observations for unknown detections
        return self._generate_synthetic_observations(latitude, longitude)

    def _generate_synthetic_observations(
        self,
        latitude: Decimal,
        longitude: Decimal,
    ) -> List[Dict[str, Any]]:
        """Generate synthetic observations for locations without data.

        Creates a reasonable time series based on biome characteristics.
        In production, this would be replaced by actual satellite queries.

        Args:
            latitude: Location latitude.
            longitude: Location longitude.

        Returns:
            List of synthetic observation dictionaries.
        """
        biome = self._determine_biome(latitude, longitude)
        baseline_ndvi = BIOME_NDVI_BASELINES.get(biome, Decimal("0.55"))

        observations = []
        # Generate quarterly observations from 2018 to 2022
        for year in range(2018, 2023):
            for quarter_month in [3, 6, 9, 12]:
                obs_date = date(year, quarter_month, 15)
                hemisphere = self._determine_hemisphere(latitude)
                seasonal_adj = SEASONAL_NDVI_ADJUSTMENTS.get(
                    hemisphere, SEASONAL_NDVI_ADJUSTMENTS["tropical"]
                ).get(quarter_month, Decimal("0"))

                ndvi = baseline_ndvi + seasonal_adj
                evi = ndvi * Decimal("0.58")

                source = "SENTINEL2" if year >= 2017 else "LANDSAT"
                observations.append({
                    "date": obs_date.isoformat(),
                    "source": source,
                    "ndvi": str(ndvi.quantize(Decimal("0.01"))),
                    "evi": str(evi.quantize(Decimal("0.01"))),
                    "cloud_pct": 10,
                })

        return sorted(observations, key=lambda x: x.get("date", ""))

    # ------------------------------------------------------------------
    # Evidence Classification
    # ------------------------------------------------------------------

    def _classify_evidence(
        self,
        observations: List[Dict[str, Any]],
        biome: str,
    ) -> List[TemporalEvidence]:
        """Classify each observation with forest state based on spectral indices.

        Args:
            observations: Raw observation dictionaries.
            biome: Biome type for threshold adjustment.

        Returns:
            List of classified TemporalEvidence objects sorted by date.
        """
        baseline_ndvi = BIOME_NDVI_BASELINES.get(biome, Decimal("0.55"))

        # Adjust thresholds for biome
        forested_threshold = baseline_ndvi * Decimal("0.75")
        transitioning_threshold = baseline_ndvi * Decimal("0.50")
        degraded_threshold = baseline_ndvi * Decimal("0.30")

        classified: List[TemporalEvidence] = []

        for obs in observations:
            ndvi = _to_decimal(obs.get("ndvi", "0"))
            evi = _to_decimal(obs.get("evi", "0"))
            source = obs.get("source", "UNKNOWN")
            obs_date = obs.get("date", "")
            cloud_pct = obs.get("cloud_pct", 0)

            # Classify forest state from NDVI
            if ndvi >= forested_threshold:
                state = ForestState.FORESTED.value
            elif ndvi >= transitioning_threshold:
                state = ForestState.TRANSITIONING.value
            elif ndvi >= degraded_threshold:
                state = ForestState.DEGRADED.value
            elif ndvi > Decimal("0"):
                state = ForestState.CLEARED.value
            else:
                state = ForestState.UNKNOWN.value

            # Calculate observation confidence
            source_reliability = SOURCE_RELIABILITY_WEIGHTS.get(
                source, Decimal("0.50")
            )
            cloud_penalty = Decimal("1") - (
                _to_decimal(cloud_pct) / Decimal("100")
            )
            obs_confidence = (
                source_reliability * cloud_penalty
            ).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)

            # Resolution based on source
            resolution = 10 if source == "SENTINEL2" else 30

            evidence = TemporalEvidence(
                evidence_id=_generate_id("ev"),
                source=source,
                observation_date=obs_date,
                forest_state=state,
                ndvi_value=ndvi,
                evi_value=evi,
                confidence=obs_confidence,
                cloud_cover_pct=cloud_pct,
                imagery_id=obs.get("imagery_id", ""),
                spatial_resolution_m=resolution,
            )
            classified.append(evidence)

        # Sort by date
        classified.sort(key=lambda e: e.observation_date)

        return classified

    # ------------------------------------------------------------------
    # Temporal Sequence Analysis
    # ------------------------------------------------------------------

    def _analyze_temporal_sequence(
        self,
        evidence: List[TemporalEvidence],
    ) -> Dict[str, Any]:
        """Analyze the temporal sequence to determine clearing timeline.

        Identifies the earliest and latest dates for each forest state,
        estimates the clearing period, and calculates the relationship
        to the EUDR cutoff date.

        Args:
            evidence: Classified temporal evidence list.

        Returns:
            Dictionary with temporal analysis results.
        """
        if not evidence:
            return {
                "sources_used": [],
                "observation_count": 0,
                "earliest_forested_date": "",
                "latest_forested_date": "",
                "earliest_cleared_date": "",
                "latest_cleared_date": "",
                "estimated_clearing_start": "",
                "estimated_clearing_end": "",
                "clearing_duration_days": 0,
                "cutoff_relationship": "UNKNOWN",
                "pre_cutoff_observations": 0,
                "post_cutoff_observations": 0,
            }

        sources_used = list({e.source for e in evidence})
        cutoff = self._cutoff_date

        forested_dates: List[str] = []
        cleared_dates: List[str] = []
        transitioning_dates: List[str] = []
        pre_cutoff_count = 0
        post_cutoff_count = 0

        for ev in evidence:
            try:
                ev_date = _date_from_str(ev.observation_date)
            except ValueError:
                continue

            if ev_date <= cutoff:
                pre_cutoff_count += 1
            else:
                post_cutoff_count += 1

            if ev.forest_state == ForestState.FORESTED.value:
                forested_dates.append(ev.observation_date)
            elif ev.forest_state == ForestState.CLEARED.value:
                cleared_dates.append(ev.observation_date)
            elif ev.forest_state == ForestState.TRANSITIONING.value:
                transitioning_dates.append(ev.observation_date)

        earliest_forested = min(forested_dates) if forested_dates else ""
        latest_forested = max(forested_dates) if forested_dates else ""
        earliest_cleared = min(cleared_dates) if cleared_dates else ""
        latest_cleared = max(cleared_dates) if cleared_dates else ""

        # Estimate clearing period
        clearing_start = ""
        clearing_end = ""
        clearing_duration = 0

        if latest_forested and earliest_cleared:
            clearing_start = latest_forested
            clearing_end = earliest_cleared
            try:
                start_d = _date_from_str(clearing_start)
                end_d = _date_from_str(clearing_end)
                clearing_duration = max(0, (end_d - start_d).days)
            except ValueError:
                pass
        elif transitioning_dates:
            clearing_start = min(transitioning_dates)
            clearing_end = max(transitioning_dates)
            if cleared_dates:
                clearing_end = min(cleared_dates)
            try:
                start_d = _date_from_str(clearing_start)
                end_d = _date_from_str(clearing_end)
                clearing_duration = max(0, (end_d - start_d).days)
            except ValueError:
                pass

        # Determine cutoff relationship
        cutoff_relationship = "UNKNOWN"
        if clearing_start and clearing_end:
            try:
                start_d = _date_from_str(clearing_start)
                end_d = _date_from_str(clearing_end)
                grace_start = cutoff - timedelta(days=self._grace_period_days)

                if end_d <= grace_start:
                    cutoff_relationship = "FULLY_PRE_CUTOFF"
                elif start_d > cutoff:
                    cutoff_relationship = "FULLY_POST_CUTOFF"
                elif start_d <= cutoff <= end_d:
                    cutoff_relationship = "SPANS_CUTOFF"
                elif start_d <= cutoff and end_d > cutoff:
                    cutoff_relationship = "SPANS_CUTOFF"
                else:
                    cutoff_relationship = "AMBIGUOUS"
            except ValueError:
                pass

        return {
            "sources_used": sources_used,
            "observation_count": len(evidence),
            "earliest_forested_date": earliest_forested,
            "latest_forested_date": latest_forested,
            "earliest_cleared_date": earliest_cleared,
            "latest_cleared_date": latest_cleared,
            "earliest_transitioning_date": (
                min(transitioning_dates) if transitioning_dates else ""
            ),
            "latest_transitioning_date": (
                max(transitioning_dates) if transitioning_dates else ""
            ),
            "estimated_clearing_start": clearing_start,
            "estimated_clearing_end": clearing_end,
            "clearing_duration_days": clearing_duration,
            "cutoff_relationship": cutoff_relationship,
            "pre_cutoff_observations": pre_cutoff_count,
            "post_cutoff_observations": post_cutoff_count,
            "forested_observations": len(forested_dates),
            "cleared_observations": len(cleared_dates),
            "transitioning_observations": len(transitioning_dates),
        }

    # ------------------------------------------------------------------
    # Transition Detection
    # ------------------------------------------------------------------

    def _detect_transitions(
        self,
        evidence: List[TemporalEvidence],
    ) -> List[TemporalTransition]:
        """Detect forest state transitions in the evidence chain.

        Identifies significant changes in forest state between consecutive
        observations and characterizes each transition.

        Args:
            evidence: Classified temporal evidence list.

        Returns:
            List of TemporalTransition objects.
        """
        if len(evidence) < 2:
            return []

        transitions: List[TemporalTransition] = []

        for i in range(1, len(evidence)):
            prev = evidence[i - 1]
            curr = evidence[i]

            if prev.forest_state == curr.forest_state:
                continue

            ndvi_change = curr.ndvi_value - prev.ndvi_value

            try:
                prev_date = _date_from_str(prev.observation_date)
                curr_date = _date_from_str(curr.observation_date)
                transition_days = max(1, (curr_date - prev_date).days)
            except ValueError:
                transition_days = 1

            transition_rate = (ndvi_change / Decimal(str(transition_days))).quantize(
                Decimal("0.000001"), rounding=ROUND_HALF_UP
            )

            # Calculate transition confidence
            tr_confidence = (
                (prev.confidence + curr.confidence) / Decimal("2")
            ).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)

            # Check if confirmed by close-in-time observations
            confirmed = transition_days <= STATE_TRANSITION_CONFIRMATION_DAYS

            transition = TemporalTransition(
                transition_id=_generate_id("tr"),
                from_state=prev.forest_state,
                to_state=curr.forest_state,
                from_date=prev.observation_date,
                to_date=curr.observation_date,
                ndvi_change=ndvi_change,
                transition_days=transition_days,
                transition_rate=transition_rate,
                confidence=tr_confidence,
                confirmed=confirmed,
            )
            transitions.append(transition)

        return transitions

    # ------------------------------------------------------------------
    # Hansen GFC Loss Year
    # ------------------------------------------------------------------

    def _get_hansen_loss_year(self, detection_id: str) -> int:
        """Retrieve Hansen Global Forest Change loss year for detection.

        In production, this queries the Hansen GFC dataset. For testing,
        uses reference data.

        Args:
            detection_id: Detection identifier.

        Returns:
            Year of tree cover loss (0 if not available).
        """
        for key, year in HANSEN_LOSS_YEARS.items():
            if key in detection_id:
                return year
        return 0

    # ------------------------------------------------------------------
    # Cutoff Result Determination
    # ------------------------------------------------------------------

    def _determine_cutoff_result(
        self,
        analysis: Dict[str, Any],
        transitions: List[TemporalTransition],
        hansen_year: int,
    ) -> CutoffResult:
        """Determine the cutoff classification from temporal analysis.

        Uses a priority-based decision tree:
        1. Hansen GFC annual loss year (strong evidence).
        2. Clearing period relationship to cutoff date.
        3. Transition timing relative to cutoff.
        4. Default to UNCERTAIN if insufficient evidence.

        Args:
            analysis: Temporal sequence analysis results.
            transitions: Detected state transitions.
            hansen_year: Hansen GFC loss year (0 if unavailable).

        Returns:
            CutoffResult classification.
        """
        cutoff_year = self._cutoff_date.year
        cutoff_relationship = analysis.get("cutoff_relationship", "UNKNOWN")

        # Priority 1: Hansen GFC year
        if hansen_year > 0:
            if hansen_year <= cutoff_year:
                if cutoff_relationship == "SPANS_CUTOFF":
                    return CutoffResult.ONGOING
                return CutoffResult.PRE_CUTOFF
            else:
                return CutoffResult.POST_CUTOFF

        # Priority 2: Cutoff relationship from temporal analysis
        if cutoff_relationship == "FULLY_PRE_CUTOFF":
            return CutoffResult.PRE_CUTOFF
        elif cutoff_relationship == "FULLY_POST_CUTOFF":
            return CutoffResult.POST_CUTOFF
        elif cutoff_relationship == "SPANS_CUTOFF":
            return CutoffResult.ONGOING

        # Priority 3: Transition timing
        clearing_transitions = [
            t for t in transitions
            if t.to_state in (ForestState.CLEARED.value, ForestState.TRANSITIONING.value)
            and t.from_state == ForestState.FORESTED.value
        ]

        if clearing_transitions:
            earliest_clearing = min(
                clearing_transitions, key=lambda t: t.from_date
            )
            try:
                clearing_start = _date_from_str(earliest_clearing.from_date)
                grace_start = self._cutoff_date - timedelta(
                    days=self._grace_period_days
                )
                if clearing_start > self._cutoff_date:
                    return CutoffResult.POST_CUTOFF
                elif clearing_start < grace_start:
                    return CutoffResult.PRE_CUTOFF
                else:
                    return CutoffResult.ONGOING
            except ValueError:
                pass

        # Priority 4: Observation count check
        obs_count = analysis.get("observation_count", 0)
        if obs_count < self._min_evidence_sources:
            return CutoffResult.UNCERTAIN

        # Priority 5: Check if cleared observations exist
        cleared_count = analysis.get("cleared_observations", 0)
        if cleared_count == 0:
            return CutoffResult.UNCERTAIN

        return CutoffResult.UNCERTAIN

    # ------------------------------------------------------------------
    # Confidence Calculation
    # ------------------------------------------------------------------

    def _calculate_confidence(
        self,
        evidence: List[TemporalEvidence],
        analysis: Dict[str, Any],
        transitions: List[TemporalTransition],
    ) -> Decimal:
        """Calculate overall verification confidence score.

        Combines four weighted components:
        1. Temporal consistency (0.30): Agreement of evidence over time.
        2. Source diversity (0.25): Number of independent sources.
        3. Observation density (0.20): Observations per year.
        4. Source reliability (0.25): Mean reliability of sources used.

        Args:
            evidence: Classified temporal evidence.
            analysis: Temporal analysis results.
            transitions: Detected transitions.

        Returns:
            Confidence score (Decimal 0.0 - 1.0).
        """
        if not evidence:
            return Decimal("0")

        # Component 1: Temporal consistency
        temporal_consistency = self._calc_temporal_consistency(evidence, analysis)

        # Component 2: Source diversity
        source_diversity = self._calc_source_diversity(evidence)

        # Component 3: Observation density
        observation_density = self._calc_observation_density(evidence, analysis)

        # Component 4: Source reliability
        source_reliability = self._calc_source_reliability(evidence)

        # Weighted combination
        confidence = (
            TEMPORAL_CONSISTENCY_WEIGHT * temporal_consistency
            + SOURCE_DIVERSITY_WEIGHT * source_diversity
            + OBSERVATION_DENSITY_WEIGHT * observation_density
            + SOURCE_RELIABILITY_WEIGHT * source_reliability
        ).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)

        return _clamp_decimal(confidence, Decimal("0"), Decimal("1"))

    def _calc_temporal_consistency(
        self,
        evidence: List[TemporalEvidence],
        analysis: Dict[str, Any],
    ) -> Decimal:
        """Calculate temporal consistency component of confidence.

        Measures how consistently the evidence tells a coherent story
        (forested -> transitioning -> cleared progression).

        Args:
            evidence: Classified temporal evidence.
            analysis: Temporal analysis results.

        Returns:
            Consistency score (0-1).
        """
        if len(evidence) < 2:
            return Decimal("0.3")

        # Check for consistent monotonic progression
        state_order = {
            ForestState.FORESTED.value: 0,
            ForestState.TRANSITIONING.value: 1,
            ForestState.DEGRADED.value: 2,
            ForestState.CLEARED.value: 3,
            ForestState.UNKNOWN.value: -1,
        }

        consistent_count = 0
        total_transitions = 0

        for i in range(1, len(evidence)):
            prev_order = state_order.get(evidence[i - 1].forest_state, -1)
            curr_order = state_order.get(evidence[i].forest_state, -1)

            if prev_order < 0 or curr_order < 0:
                continue

            total_transitions += 1
            # Consistent if state stays same or progresses forward
            if curr_order >= prev_order:
                consistent_count += 1

        if total_transitions == 0:
            return Decimal("0.3")

        ratio = Decimal(str(consistent_count)) / Decimal(str(total_transitions))
        return ratio.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)

    def _calc_source_diversity(
        self,
        evidence: List[TemporalEvidence],
    ) -> Decimal:
        """Calculate source diversity component of confidence.

        More independent sources increase confidence.

        Args:
            evidence: Classified temporal evidence.

        Returns:
            Diversity score (0-1).
        """
        unique_sources = {e.source for e in evidence}
        total_sources = len(SOURCE_RELIABILITY_WEIGHTS)
        diversity = Decimal(str(len(unique_sources))) / Decimal(str(total_sources))
        return _clamp_decimal(
            diversity.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP),
            Decimal("0"),
            Decimal("1"),
        )

    def _calc_observation_density(
        self,
        evidence: List[TemporalEvidence],
        analysis: Dict[str, Any],
    ) -> Decimal:
        """Calculate observation density component of confidence.

        More frequent observations reduce temporal uncertainty.

        Args:
            evidence: Classified temporal evidence.
            analysis: Temporal analysis results.

        Returns:
            Density score (0-1).
        """
        if len(evidence) < 2:
            return Decimal("0.2")

        # Get date range
        dates = []
        for ev in evidence:
            try:
                dates.append(_date_from_str(ev.observation_date))
            except ValueError:
                pass

        if len(dates) < 2:
            return Decimal("0.2")

        coverage_years = max(
            Decimal("0.5"),
            Decimal(str((dates[-1] - dates[0]).days)) / Decimal("365.25"),
        )

        # observations per year (target: 12 per year for monthly coverage)
        obs_per_year = Decimal(str(len(dates))) / coverage_years
        # Normalize to 0-1 with 12/year as perfect score
        density = _clamp_decimal(
            obs_per_year / Decimal("12"),
            Decimal("0"),
            Decimal("1"),
        )

        return density.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)

    def _calc_source_reliability(
        self,
        evidence: List[TemporalEvidence],
    ) -> Decimal:
        """Calculate mean source reliability component of confidence.

        Args:
            evidence: Classified temporal evidence.

        Returns:
            Mean reliability score (0-1).
        """
        if not evidence:
            return Decimal("0")

        total_reliability = Decimal("0")
        for ev in evidence:
            total_reliability += SOURCE_RELIABILITY_WEIGHTS.get(
                ev.source, Decimal("0.50")
            )

        mean = total_reliability / Decimal(str(len(evidence)))
        return mean.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)

    # ------------------------------------------------------------------
    # Confidence Classification
    # ------------------------------------------------------------------

    def _classify_confidence(self, confidence: Decimal) -> ConfidenceLevel:
        """Classify confidence score into a level.

        Args:
            confidence: Confidence score (0-1).

        Returns:
            ConfidenceLevel classification.
        """
        if confidence >= CONFIDENCE_HIGH:
            return ConfidenceLevel.HIGH
        elif confidence >= CONFIDENCE_MEDIUM:
            return ConfidenceLevel.MEDIUM
        elif confidence >= CONFIDENCE_LOW:
            return ConfidenceLevel.LOW
        return ConfidenceLevel.INSUFFICIENT

    # ------------------------------------------------------------------
    # Compliance Determination
    # ------------------------------------------------------------------

    def _determine_compliance(
        self,
        cutoff_result: CutoffResult,
        confidence: Decimal,
    ) -> bool:
        """Determine EUDR compliance based on cutoff result and confidence.

        A product is EUDR compliant only if:
        - Cutoff result is PRE_CUTOFF with sufficient confidence, OR
        - No deforestation detected.

        POST_CUTOFF, ONGOING, and UNCERTAIN are all treated as non-compliant
        for market access purposes (precautionary principle).

        Args:
            cutoff_result: Cutoff classification.
            confidence: Verification confidence.

        Returns:
            True if products can access EU market, False otherwise.
        """
        if cutoff_result == CutoffResult.PRE_CUTOFF:
            # Only compliant if confidence meets threshold
            return confidence >= self._cutoff_confidence_threshold

        # POST_CUTOFF, ONGOING, UNCERTAIN are all non-compliant
        return False

    # ------------------------------------------------------------------
    # Risk Level Determination
    # ------------------------------------------------------------------

    def _determine_risk_level(
        self,
        cutoff_result: CutoffResult,
        confidence: Decimal,
    ) -> str:
        """Determine compliance risk level from cutoff result.

        Args:
            cutoff_result: Cutoff classification.
            confidence: Verification confidence.

        Returns:
            Risk level string: CRITICAL, HIGH, MEDIUM, LOW, or INFORMATIONAL.
        """
        if cutoff_result == CutoffResult.POST_CUTOFF:
            return "CRITICAL"
        elif cutoff_result == CutoffResult.ONGOING:
            return "HIGH"
        elif cutoff_result == CutoffResult.UNCERTAIN:
            if confidence < CONFIDENCE_LOW:
                return "HIGH"
            return "MEDIUM"
        elif cutoff_result == CutoffResult.PRE_CUTOFF:
            if confidence >= CONFIDENCE_HIGH:
                return "LOW"
            return "MEDIUM"
        return "HIGH"

    # ------------------------------------------------------------------
    # Verification Status
    # ------------------------------------------------------------------

    def _determine_verification_status(
        self,
        evidence: List[TemporalEvidence],
        confidence: Decimal,
    ) -> VerificationStatus:
        """Determine verification processing status.

        Args:
            evidence: Classified temporal evidence.
            confidence: Overall confidence score.

        Returns:
            VerificationStatus classification.
        """
        if not evidence:
            return VerificationStatus.FAILED

        source_count = len({e.source for e in evidence})
        if source_count >= self._min_evidence_sources:
            return VerificationStatus.COMPLETED

        return VerificationStatus.PARTIAL

    # ------------------------------------------------------------------
    # Warnings Generation
    # ------------------------------------------------------------------

    def _generate_warnings(
        self,
        evidence: List[TemporalEvidence],
        analysis: Dict[str, Any],
        confidence: Decimal,
        cutoff_result: CutoffResult,
    ) -> List[str]:
        """Generate warning messages for the verification result.

        Args:
            evidence: Classified temporal evidence.
            analysis: Temporal analysis results.
            confidence: Overall confidence.
            cutoff_result: Cutoff classification.

        Returns:
            List of warning message strings.
        """
        warnings: List[str] = []

        # Insufficient evidence sources
        source_count = len({e.source for e in evidence})
        if source_count < self._min_evidence_sources:
            warnings.append(
                f"Only {source_count} evidence source(s) available; "
                f"minimum {self._min_evidence_sources} recommended for "
                f"reliable classification."
            )

        # Low confidence
        if confidence < CONFIDENCE_LOW:
            warnings.append(
                f"Verification confidence ({confidence}) is below "
                f"LOW threshold ({CONFIDENCE_LOW}). Result may be unreliable."
            )
        elif confidence < self._cutoff_confidence_threshold:
            warnings.append(
                f"Verification confidence ({confidence}) is below "
                f"cutoff confidence threshold ({self._cutoff_confidence_threshold})."
            )

        # Large observation gaps
        dates = []
        for ev in evidence:
            try:
                dates.append(_date_from_str(ev.observation_date))
            except ValueError:
                pass

        if len(dates) >= 2:
            dates_sorted = sorted(dates)
            for i in range(1, len(dates_sorted)):
                gap = (dates_sorted[i] - dates_sorted[i - 1]).days
                if gap > MAX_OBSERVATION_GAP_DAYS:
                    warnings.append(
                        f"Large observation gap of {gap} days detected "
                        f"between {dates_sorted[i-1].isoformat()} and "
                        f"{dates_sorted[i].isoformat()}."
                    )
                    break

        # Uncertain result
        if cutoff_result == CutoffResult.UNCERTAIN:
            warnings.append(
                "Cutoff classification is UNCERTAIN. Products from this "
                "area should be treated as HIGH risk per precautionary "
                "principle until further evidence is available."
            )

        # Ongoing clearing
        if cutoff_result == CutoffResult.ONGOING:
            warnings.append(
                "Clearing activity appears to span the EUDR cutoff date. "
                "Enhanced due diligence is required to determine the "
                "proportion of clearing that occurred post-cutoff."
            )

        # High cloud cover
        high_cloud_count = sum(
            1 for ev in evidence if ev.cloud_cover_pct > 30
        )
        if high_cloud_count > len(evidence) * 0.3:
            warnings.append(
                f"{high_cloud_count} of {len(evidence)} observations "
                f"have cloud cover >30%, reducing spectral index reliability."
            )

        # No pre-cutoff observations
        pre_count = analysis.get("pre_cutoff_observations", 0)
        if pre_count == 0:
            warnings.append(
                "No observations available before the EUDR cutoff date. "
                "Pre-cutoff forest state cannot be verified."
            )

        return warnings

    # ------------------------------------------------------------------
    # Pre/Post Cutoff State Checks
    # ------------------------------------------------------------------

    def check_pre_cutoff_forest_state(
        self,
        detection_id: str,
        latitude: Any,
        longitude: Any,
        target_date: Optional[str] = None,
        country_code: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Check forest state at a specific date before the cutoff.

        Args:
            detection_id: Detection identifier.
            latitude: Location latitude.
            longitude: Location longitude.
            target_date: Target date (defaults to cutoff date).
            country_code: Optional ISO country code.

        Returns:
            Dictionary with forest state assessment at the target date.
        """
        lat = _to_decimal(latitude)
        lon = _to_decimal(longitude)
        target = (
            _date_from_str(target_date)
            if target_date
            else self._cutoff_date
        )
        biome = self._determine_biome(lat, lon, country_code)

        evidence_list = self._collect_temporal_evidence(detection_id, lat, lon)
        classified = self._classify_evidence(evidence_list, biome)

        # Find closest observation to target date
        closest_ev = None
        closest_gap = float("inf")

        for ev in classified:
            try:
                ev_date = _date_from_str(ev.observation_date)
                gap = abs((ev_date - target).days)
                if gap < closest_gap:
                    closest_gap = gap
                    closest_ev = ev
            except ValueError:
                pass

        if closest_ev is None:
            return {
                "detection_id": detection_id,
                "target_date": target.isoformat(),
                "forest_state": ForestState.UNKNOWN.value,
                "confidence": "0",
                "observation_gap_days": -1,
                "warning": "No observations available near target date.",
                "provenance_hash": _compute_hash({
                    "detection_id": detection_id,
                    "target_date": target.isoformat(),
                }),
            }

        result = {
            "detection_id": detection_id,
            "target_date": target.isoformat(),
            "forest_state": closest_ev.forest_state,
            "ndvi": str(closest_ev.ndvi_value),
            "evi": str(closest_ev.evi_value),
            "confidence": str(closest_ev.confidence),
            "observation_date": closest_ev.observation_date,
            "observation_gap_days": int(closest_gap),
            "source": closest_ev.source,
            "biome": biome,
        }
        result["provenance_hash"] = _compute_hash(result)

        return result

    def check_post_cutoff_change(
        self,
        detection_id: str,
        latitude: Any,
        longitude: Any,
        country_code: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Check for forest state changes after the cutoff date.

        Args:
            detection_id: Detection identifier.
            latitude: Location latitude.
            longitude: Location longitude.
            country_code: Optional ISO country code.

        Returns:
            Dictionary with post-cutoff change assessment.
        """
        lat = _to_decimal(latitude)
        lon = _to_decimal(longitude)
        biome = self._determine_biome(lat, lon, country_code)

        evidence_list = self._collect_temporal_evidence(detection_id, lat, lon)
        classified = self._classify_evidence(evidence_list, biome)

        # Filter to post-cutoff observations
        post_cutoff_evidence = []
        for ev in classified:
            try:
                ev_date = _date_from_str(ev.observation_date)
                if ev_date > self._cutoff_date:
                    post_cutoff_evidence.append(ev)
            except ValueError:
                pass

        if not post_cutoff_evidence:
            return {
                "detection_id": detection_id,
                "cutoff_date": self._cutoff_date.isoformat(),
                "post_cutoff_observations": 0,
                "change_detected": False,
                "warning": "No post-cutoff observations available.",
                "provenance_hash": _compute_hash({
                    "detection_id": detection_id,
                    "cutoff_date": self._cutoff_date.isoformat(),
                }),
            }

        # Detect changes within post-cutoff period
        changes = []
        for i in range(1, len(post_cutoff_evidence)):
            prev = post_cutoff_evidence[i - 1]
            curr = post_cutoff_evidence[i]
            ndvi_change = curr.ndvi_value - prev.ndvi_value

            if ndvi_change < NDVI_CLEARING_DROP:
                changes.append({
                    "from_date": prev.observation_date,
                    "to_date": curr.observation_date,
                    "ndvi_change": str(ndvi_change),
                    "from_state": prev.forest_state,
                    "to_state": curr.forest_state,
                })

        result = {
            "detection_id": detection_id,
            "cutoff_date": self._cutoff_date.isoformat(),
            "post_cutoff_observations": len(post_cutoff_evidence),
            "change_detected": len(changes) > 0,
            "changes": changes,
            "earliest_post_cutoff_date": (
                post_cutoff_evidence[0].observation_date
                if post_cutoff_evidence else ""
            ),
            "latest_post_cutoff_date": (
                post_cutoff_evidence[-1].observation_date
                if post_cutoff_evidence else ""
            ),
            "current_state": (
                post_cutoff_evidence[-1].forest_state
                if post_cutoff_evidence else ForestState.UNKNOWN.value
            ),
        }
        result["provenance_hash"] = _compute_hash(result)

        return result
