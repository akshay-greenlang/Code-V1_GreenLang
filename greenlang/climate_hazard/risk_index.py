# -*- coding: utf-8 -*-
"""
Risk Index Engine - AGENT-DATA-020 (Engine 2 of 7)

Computes composite climate risk indices from raw hazard data combining
probability, intensity, frequency, and duration into a single 0-100
risk score.  Supports single-hazard, multi-hazard, and compound-event
risk calculations with configurable weighting from the Climate Hazard
Connector configuration.

The engine provides five risk classification tiers (NEGLIGIBLE, LOW,
MEDIUM, HIGH, EXTREME), hazard ranking, location comparison, and risk
trend analysis.  All calculations are deterministic (zero-hallucination):
every numeric result derives from explicit Python arithmetic with no LLM
involvement.

Risk Score Formula (0-100):
    risk_score = (probability * w_prob
                  + normalized_intensity * w_int
                  + normalized_frequency * w_freq
                  + normalized_duration * w_dur) * 100

    Where:
        - probability is in [0, 1]
        - normalized_intensity = min(intensity / 10.0, 1.0)
        - normalized_frequency = min(frequency / 10.0, 1.0)
        - normalized_duration  = min(duration_days / 365.0, 1.0)
        - Default weights: w_prob=0.30, w_int=0.30, w_freq=0.25, w_dur=0.15

Risk Levels (5 tiers):
    NEGLIGIBLE:  0 <= score <  20
    LOW:        20 <= score <  40
    MEDIUM:     40 <= score <  60
    HIGH:       60 <= score <  80
    EXTREME:    80 <= score <= 100

Compound Hazard Correlations (default):
    drought + wildfire:                          0.75
    flood + landslide:                           0.70
    extreme_heat + drought:                      0.65
    tropical_cyclone + coastal_flood:            0.80
    extreme_precipitation + riverine_flood:      0.85
    sea_level_rise + coastal_erosion:            0.70

Example:
    >>> from greenlang.climate_hazard.risk_index import RiskIndexEngine
    >>> engine = RiskIndexEngine()
    >>> result = engine.calculate_risk_index(
    ...     hazard_type="riverine_flood",
    ...     location={"lat": 51.5074, "lon": -0.1278, "name": "London"},
    ...     probability=0.4,
    ...     intensity=6.0,
    ...     frequency=2.0,
    ...     duration_days=14.0,
    ... )
    >>> print(result["risk_score"], result["risk_level"])

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
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

__all__ = ["RiskIndexEngine"]


# ---------------------------------------------------------------------------
# Graceful imports -- config
# ---------------------------------------------------------------------------

try:
    from greenlang.climate_hazard.config import get_config as _get_config

    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False
    _get_config = None  # type: ignore[assignment]
    logger.info(
        "climate_hazard.config not available; "
        "RiskIndexEngine will use built-in defaults"
    )


# ---------------------------------------------------------------------------
# Graceful imports -- provenance
# ---------------------------------------------------------------------------

try:
    from greenlang.climate_hazard.provenance import (
        ProvenanceTracker as _ProvenanceTracker,
    )

    _PROVENANCE_AVAILABLE = True
except ImportError:
    _PROVENANCE_AVAILABLE = False
    _ProvenanceTracker = None  # type: ignore[assignment,misc]
    logger.info(
        "climate_hazard.provenance not available; "
        "RiskIndexEngine provenance tracking disabled"
    )


# ---------------------------------------------------------------------------
# Graceful imports -- metrics
# ---------------------------------------------------------------------------

try:
    from greenlang.climate_hazard.metrics import (
        record_risk_calculation as _record_risk_calculation_raw,
    )

    _METRICS_AVAILABLE = True
except ImportError:
    _METRICS_AVAILABLE = False
    _record_risk_calculation_raw = None  # type: ignore[assignment]
    logger.info(
        "climate_hazard.metrics not available; "
        "RiskIndexEngine metrics recording disabled"
    )


# ---------------------------------------------------------------------------
# Safe metrics helper
# ---------------------------------------------------------------------------


def _safe_record_risk_calculation(hazard_type: str, risk_level: str) -> None:
    """Safely record a risk calculation metric.

    Args:
        hazard_type: The hazard type that was calculated.
        risk_level: The resulting risk level classification.
    """
    if _METRICS_AVAILABLE and _record_risk_calculation_raw is not None:
        try:
            _record_risk_calculation_raw(hazard_type, risk_level)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class RiskLevel(str, Enum):
    """Five-tier climate risk classification.

    Attributes:
        NEGLIGIBLE: No significant hazard risk (score 0-20).
        LOW: Minor hazard potential (score 20-40).
        MEDIUM: Moderate hazard potential (score 40-60).
        HIGH: Significant hazard risk (score 60-80).
        EXTREME: Critical or imminent hazard (score 80-100).
    """

    NEGLIGIBLE = "NEGLIGIBLE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


class HazardType(str, Enum):
    """Twelve climate hazard types supported by the connector.

    Attributes:
        RIVERINE_FLOOD: River overflow flooding.
        COASTAL_FLOOD: Storm surge and tidal flooding.
        DROUGHT: Meteorological, hydrological, or agricultural drought.
        EXTREME_HEAT: Heat waves and temperature extremes.
        EXTREME_COLD: Cold waves and frost events.
        WILDFIRE: Forest and brush fires.
        TROPICAL_CYCLONE: Hurricanes, typhoons, and cyclones.
        EXTREME_PRECIPITATION: Heavy rainfall and hail events.
        WATER_STRESS: Water scarcity and depletion.
        SEA_LEVEL_RISE: Chronic sea level rise.
        LANDSLIDE: Rainfall or seismic-induced landslides.
        COASTAL_EROSION: Shoreline retreat.
    """

    RIVERINE_FLOOD = "riverine_flood"
    COASTAL_FLOOD = "coastal_flood"
    DROUGHT = "drought"
    EXTREME_HEAT = "extreme_heat"
    EXTREME_COLD = "extreme_cold"
    WILDFIRE = "wildfire"
    TROPICAL_CYCLONE = "tropical_cyclone"
    EXTREME_PRECIPITATION = "extreme_precipitation"
    WATER_STRESS = "water_stress"
    SEA_LEVEL_RISE = "sea_level_rise"
    LANDSLIDE = "landslide"
    COASTAL_EROSION = "coastal_erosion"


class AggregationStrategy(str, Enum):
    """Strategy for aggregating multiple hazard risk scores.

    Attributes:
        WEIGHTED_AVERAGE: Weight-averaged composite of per-hazard scores.
        MAXIMUM: The single highest hazard score dominates.
        SUM_CAPPED: Sum of all hazard scores, capped at 100.
    """

    WEIGHTED_AVERAGE = "weighted_average"
    MAXIMUM = "maximum"
    SUM_CAPPED = "sum_capped"


class TrendDirection(str, Enum):
    """Direction of risk evolution over successive time horizons.

    Attributes:
        INCREASING: Risk is rising over time.
        DECREASING: Risk is declining over time.
        STABLE: Risk is not changing materially.
    """

    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"


# ---------------------------------------------------------------------------
# Default compound hazard correlation matrix
# ---------------------------------------------------------------------------

# Keys are frozensets of two hazard-type strings (lower-cased) so lookup
# order does not matter.  Values are Pearson-style correlation factors in
# [0, 1] representing how strongly one hazard amplifies another.

_DEFAULT_COMPOUND_CORRELATIONS: Dict[frozenset, float] = {
    frozenset({"drought", "wildfire"}): 0.75,
    frozenset({"flood", "landslide"}): 0.70,
    frozenset({"riverine_flood", "landslide"}): 0.70,
    frozenset({"coastal_flood", "landslide"}): 0.70,
    frozenset({"extreme_heat", "drought"}): 0.65,
    frozenset({"tropical_cyclone", "coastal_flood"}): 0.80,
    frozenset({"extreme_precipitation", "riverine_flood"}): 0.85,
    frozenset({"sea_level_rise", "coastal_erosion"}): 0.70,
}


# ---------------------------------------------------------------------------
# Default risk index weights
# ---------------------------------------------------------------------------

_DEFAULT_WEIGHT_PROBABILITY: float = 0.30
_DEFAULT_WEIGHT_INTENSITY: float = 0.30
_DEFAULT_WEIGHT_FREQUENCY: float = 0.25
_DEFAULT_WEIGHT_DURATION: float = 0.15


# ---------------------------------------------------------------------------
# Default risk level thresholds (lower bounds, inclusive)
# ---------------------------------------------------------------------------

_DEFAULT_THRESHOLD_EXTREME: float = 80.0
_DEFAULT_THRESHOLD_HIGH: float = 60.0
_DEFAULT_THRESHOLD_MEDIUM: float = 40.0
_DEFAULT_THRESHOLD_LOW: float = 20.0


# ---------------------------------------------------------------------------
# Normalisation caps
# ---------------------------------------------------------------------------

_INTENSITY_MAX: float = 10.0
_FREQUENCY_MAX: float = 10.0
_DURATION_MAX_DAYS: float = 365.0


# ---------------------------------------------------------------------------
# Helper: UTC now
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed.

    Returns:
        A timezone-aware datetime object truncated to seconds.
    """
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# Helper: deterministic SHA-256 hash
# ---------------------------------------------------------------------------


def _hash_data(data: Any) -> str:
    """Compute a deterministic SHA-256 hash for arbitrary data.

    Serializes the payload to canonical JSON (sorted keys, ``str``
    fallback for non-serialisable types) and returns the hex digest.

    Args:
        data: Any JSON-serialisable Python object (dict, list, str,
            number, ``None``).

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    if data is None:
        serialized = "null"
    else:
        serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# RiskIndexEngine
# ---------------------------------------------------------------------------


class RiskIndexEngine:
    """Composite climate risk index calculation engine.

    Computes single-hazard, multi-hazard, and compound-event risk
    indices from raw hazard parameters (probability, intensity,
    frequency, duration).  All calculations are deterministic --
    no LLM involvement.

    Risk weights and classification thresholds are loaded from the
    :class:`~greenlang.climate_hazard.config.ClimateHazardConfig`
    singleton when available, otherwise built-in defaults are used.

    Thread-safety is provided via a ``threading.Lock`` that protects
    all mutable state (the in-memory risk index store).

    Attributes:
        _database: Optional reference to a HazardDatabaseEngine for
            integrated hazard data lookups.
        _provenance: ProvenanceTracker instance for SHA-256 audit
            trails (or ``None`` when provenance module is unavailable).
        _weight_probability: Weight for hazard probability in the
            composite risk score.
        _weight_intensity: Weight for normalised hazard intensity.
        _weight_frequency: Weight for normalised hazard frequency.
        _weight_duration: Weight for normalised hazard duration.
        _threshold_extreme: Lower bound for EXTREME risk level.
        _threshold_high: Lower bound for HIGH risk level.
        _threshold_medium: Lower bound for MEDIUM risk level.
        _threshold_low: Lower bound for LOW risk level.
        _indices: In-memory store of computed risk indices keyed by
            index_id.
        _compound_correlations: Default compound hazard correlation
            look-up table.
        _lock: Thread-safety lock protecting all mutable state.
        _total_calculations: Running count of risk calculations
            performed (single-hazard).
        _total_multi_calculations: Running count of multi-hazard
            calculations performed.
        _total_compound_calculations: Running count of compound risk
            calculations performed.
        _created_at: ISO-formatted UTC timestamp of engine creation.

    Example:
        >>> engine = RiskIndexEngine()
        >>> result = engine.calculate_risk_index(
        ...     hazard_type="drought",
        ...     location={"lat": 34.0, "lon": -118.2, "name": "LA"},
        ...     probability=0.6,
        ...     intensity=7.5,
        ...     frequency=1.5,
        ...     duration_days=90,
        ... )
        >>> result["risk_level"]
        'HIGH'
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        database: Any = None,
        provenance: Any = None,
        genesis_hash: Optional[str] = None,
    ) -> None:
        """Initialize the RiskIndexEngine.

        Args:
            database: Optional reference to a ``HazardDatabaseEngine``
                instance for integrated hazard data lookups.  May be
                ``None`` when the engine is used standalone.
            provenance: Optional ``ProvenanceTracker`` instance for
                SHA-256 audit trails.  When ``None`` a new tracker is
                created if the provenance module is available.
            genesis_hash: Optional genesis hash anchor string for the
                provenance chain.  Ignored when a ``provenance``
                instance is explicitly supplied.
        """
        self._database = database
        self._lock = threading.Lock()
        self._created_at: str = _utcnow().isoformat()

        # ---- Provenance tracker -------------------------------------------
        self._provenance = self._init_provenance(provenance, genesis_hash)

        # ---- Risk weights from config or defaults -------------------------
        self._weight_probability: float = _DEFAULT_WEIGHT_PROBABILITY
        self._weight_intensity: float = _DEFAULT_WEIGHT_INTENSITY
        self._weight_frequency: float = _DEFAULT_WEIGHT_FREQUENCY
        self._weight_duration: float = _DEFAULT_WEIGHT_DURATION

        # ---- Risk level thresholds from config or defaults ----------------
        self._threshold_extreme: float = _DEFAULT_THRESHOLD_EXTREME
        self._threshold_high: float = _DEFAULT_THRESHOLD_HIGH
        self._threshold_medium: float = _DEFAULT_THRESHOLD_MEDIUM
        self._threshold_low: float = _DEFAULT_THRESHOLD_LOW

        self._load_config_weights()

        # ---- In-memory risk index store -----------------------------------
        self._indices: Dict[str, Dict[str, Any]] = {}

        # ---- Default compound correlations --------------------------------
        self._compound_correlations: Dict[
            frozenset, float
        ] = dict(_DEFAULT_COMPOUND_CORRELATIONS)

        # ---- Statistics counters ------------------------------------------
        self._total_calculations: int = 0
        self._total_multi_calculations: int = 0
        self._total_compound_calculations: int = 0
        self._total_rankings: int = 0
        self._total_comparisons: int = 0
        self._total_trends: int = 0

        logger.info(
            "RiskIndexEngine initialized: "
            "weights=(prob=%.2f, int=%.2f, freq=%.2f, dur=%.2f), "
            "thresholds=(extreme=%.1f, high=%.1f, med=%.1f, low=%.1f), "
            "provenance=%s, database=%s",
            self._weight_probability,
            self._weight_intensity,
            self._weight_frequency,
            self._weight_duration,
            self._threshold_extreme,
            self._threshold_high,
            self._threshold_medium,
            self._threshold_low,
            self._provenance is not None,
            self._database is not None,
        )

    # ------------------------------------------------------------------
    # Provenance initialisation helper
    # ------------------------------------------------------------------

    def _init_provenance(
        self,
        provenance: Any,
        genesis_hash: Optional[str],
    ) -> Any:
        """Resolve the ProvenanceTracker to use.

        If ``provenance`` is explicitly provided, it is returned as-is.
        Otherwise, if the provenance module is available, a new tracker
        is created with the supplied or default genesis hash.

        Args:
            provenance: Explicitly supplied ProvenanceTracker or None.
            genesis_hash: Optional genesis hash string.

        Returns:
            A ProvenanceTracker instance or ``None``.
        """
        if provenance is not None:
            return provenance
        if _PROVENANCE_AVAILABLE and _ProvenanceTracker is not None:
            genhash = genesis_hash or "greenlang-climate-hazard-risk-index-genesis"
            return _ProvenanceTracker(genesis_hash=genhash)
        return None

    # ------------------------------------------------------------------
    # Config loading helper
    # ------------------------------------------------------------------

    def _load_config_weights(self) -> None:
        """Load risk weights and thresholds from config if available.

        Falls back to module-level defaults when the config module is
        not importable.  Logs the source of the weights at DEBUG level.
        """
        if not _CONFIG_AVAILABLE or _get_config is None:
            logger.debug(
                "RiskIndexEngine: using built-in default weights and thresholds"
            )
            return

        try:
            cfg = _get_config()
            self._weight_probability = cfg.risk_weight_probability
            self._weight_intensity = cfg.risk_weight_intensity
            self._weight_frequency = cfg.risk_weight_frequency
            self._weight_duration = cfg.risk_weight_duration
            self._threshold_extreme = cfg.threshold_extreme
            self._threshold_high = cfg.threshold_high
            self._threshold_medium = cfg.threshold_medium
            self._threshold_low = cfg.threshold_low
            logger.debug(
                "RiskIndexEngine: loaded weights and thresholds from config"
            )
        except Exception as exc:
            logger.warning(
                "RiskIndexEngine: failed to load config, "
                "falling back to defaults: %s",
                exc,
            )

    # ------------------------------------------------------------------
    # Internal: generate index ID
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_index_id() -> str:
        """Generate a unique risk index identifier.

        Format: ``RI-<hex12>`` where ``hex12`` is the first 12
        hex characters of a UUID4.

        Returns:
            Unique index ID string.
        """
        return f"RI-{uuid4().hex[:12]}"

    # ------------------------------------------------------------------
    # Internal: provenance recording
    # ------------------------------------------------------------------

    def _record_provenance(
        self,
        entity_type: str,
        action: str,
        entity_id: str,
        data: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Record a provenance entry if the tracker is available.

        Args:
            entity_type: Provenance entity type (e.g. ``"risk_index"``).
            action: Action label (e.g. ``"calculate_risk"``).
            entity_id: Entity identifier.
            data: Optional data payload to hash.
            metadata: Optional extra metadata dict.

        Returns:
            The chain hash string, or ``None`` if provenance is
            unavailable.
        """
        if self._provenance is None:
            return None
        try:
            entry = self._provenance.record(
                entity_type=entity_type,
                action=action,
                entity_id=entity_id,
                data=data,
                metadata=metadata,
            )
            return entry.hash_value
        except Exception as exc:
            logger.warning(
                "RiskIndexEngine provenance recording failed: %s", exc
            )
            return None

    # ------------------------------------------------------------------
    # Internal: normalisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_intensity(intensity: float) -> float:
        """Normalise hazard intensity to [0, 1].

        Divides intensity by 10 (the maximum severity scale) and clamps
        the result to a maximum of 1.0.

        Args:
            intensity: Raw hazard intensity on a 0-10 scale.

        Returns:
            Normalised intensity in [0, 1].
        """
        return min(max(intensity, 0.0) / _INTENSITY_MAX, 1.0)

    @staticmethod
    def _normalise_frequency(frequency: float) -> float:
        """Normalise hazard frequency (events/year) to [0, 1].

        Caps frequency at 10 events per year and divides by the cap.

        Args:
            frequency: Expected hazard occurrence rate per year.

        Returns:
            Normalised frequency in [0, 1].
        """
        return min(max(frequency, 0.0) / _FREQUENCY_MAX, 1.0)

    @staticmethod
    def _normalise_duration(duration_days: float) -> float:
        """Normalise hazard duration (days) to [0, 1].

        Caps duration at 365 days and divides by the cap.

        Args:
            duration_days: Average hazard event duration in days.

        Returns:
            Normalised duration in [0, 1].
        """
        return min(max(duration_days, 0.0) / _DURATION_MAX_DAYS, 1.0)

    @staticmethod
    def _clamp_probability(probability: float) -> float:
        """Clamp probability to the valid [0, 1] range.

        Args:
            probability: Hazard occurrence probability.

        Returns:
            Probability clamped to [0, 1].
        """
        return max(0.0, min(probability, 1.0))

    # ------------------------------------------------------------------
    # Internal: risk score computation (ZERO-HALLUCINATION)
    # ------------------------------------------------------------------

    def _compute_risk_score(
        self,
        probability: float,
        intensity: float,
        frequency: float,
        duration_days: float,
    ) -> float:
        """Compute the composite risk score from raw hazard parameters.

        This is the core deterministic calculation.  No LLM calls.

        Formula::

            risk_score = (probability * w_prob
                          + norm_intensity * w_int
                          + norm_frequency * w_freq
                          + norm_duration * w_dur) * 100

        Args:
            probability: Hazard occurrence probability in [0, 1].
            intensity: Hazard intensity on a 0-10 scale.
            frequency: Expected events per year (capped at 10).
            duration_days: Average event duration in days (capped at 365).

        Returns:
            Composite risk score in [0, 100].
        """
        prob = self._clamp_probability(probability)
        norm_int = self._normalise_intensity(intensity)
        norm_freq = self._normalise_frequency(frequency)
        norm_dur = self._normalise_duration(duration_days)

        raw_score = (
            prob * self._weight_probability
            + norm_int * self._weight_intensity
            + norm_freq * self._weight_frequency
            + norm_dur * self._weight_duration
        ) * 100.0

        # Clamp to [0, 100]
        return max(0.0, min(round(raw_score, 4), 100.0))

    # ------------------------------------------------------------------
    # Internal: component scores builder
    # ------------------------------------------------------------------

    def _build_component_scores(
        self,
        probability: float,
        intensity: float,
        frequency: float,
        duration_days: float,
    ) -> Dict[str, Any]:
        """Build a dictionary of individual component scores.

        Args:
            probability: Raw probability.
            intensity: Raw intensity.
            frequency: Raw frequency.
            duration_days: Raw duration in days.

        Returns:
            Dictionary containing raw and normalised values for each
            component, plus the corresponding weight.
        """
        return {
            "probability": {
                "raw": round(probability, 6),
                "normalised": round(self._clamp_probability(probability), 6),
                "weight": self._weight_probability,
            },
            "intensity": {
                "raw": round(intensity, 6),
                "normalised": round(self._normalise_intensity(intensity), 6),
                "weight": self._weight_intensity,
            },
            "frequency": {
                "raw": round(frequency, 6),
                "normalised": round(self._normalise_frequency(frequency), 6),
                "weight": self._weight_frequency,
            },
            "duration": {
                "raw_days": round(duration_days, 6),
                "normalised": round(self._normalise_duration(duration_days), 6),
                "weight": self._weight_duration,
            },
        }

    # ------------------------------------------------------------------
    # Internal: location normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_location(
        location: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Normalise a location dict ensuring lat, lon, and name keys.

        Args:
            location: Dictionary that should contain ``lat``, ``lon``,
                and optionally ``name`` keys.

        Returns:
            Normalised copy with guaranteed ``lat``, ``lon``, and
            ``name`` keys.
        """
        return {
            "lat": float(location.get("lat", 0.0)),
            "lon": float(location.get("lon", 0.0)),
            "name": str(location.get("name", "unknown")),
        }

    # ------------------------------------------------------------------
    # Internal: validate risk components
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_risk_components(
        probability: float,
        intensity: float,
        frequency: float,
        duration_days: float,
    ) -> List[str]:
        """Validate risk component values and return error messages.

        Args:
            probability: Hazard probability.
            intensity: Hazard intensity.
            frequency: Hazard frequency.
            duration_days: Hazard duration.

        Returns:
            List of validation error strings; empty if all valid.
        """
        errors: List[str] = []
        if not isinstance(probability, (int, float)):
            errors.append(
                f"probability must be numeric, got {type(probability).__name__}"
            )
        elif math.isnan(probability) or math.isinf(probability):
            errors.append("probability must be finite")
        elif probability < 0.0 or probability > 1.0:
            errors.append(
                f"probability must be in [0, 1], got {probability}"
            )

        if not isinstance(intensity, (int, float)):
            errors.append(
                f"intensity must be numeric, got {type(intensity).__name__}"
            )
        elif math.isnan(intensity) or math.isinf(intensity):
            errors.append("intensity must be finite")
        elif intensity < 0.0:
            errors.append(
                f"intensity must be >= 0, got {intensity}"
            )

        if not isinstance(frequency, (int, float)):
            errors.append(
                f"frequency must be numeric, got {type(frequency).__name__}"
            )
        elif math.isnan(frequency) or math.isinf(frequency):
            errors.append("frequency must be finite")
        elif frequency < 0.0:
            errors.append(
                f"frequency must be >= 0, got {frequency}"
            )

        if not isinstance(duration_days, (int, float)):
            errors.append(
                f"duration_days must be numeric, got {type(duration_days).__name__}"
            )
        elif math.isnan(duration_days) or math.isinf(duration_days):
            errors.append("duration_days must be finite")
        elif duration_days < 0.0:
            errors.append(
                f"duration_days must be >= 0, got {duration_days}"
            )

        return errors

    # ------------------------------------------------------------------
    # Internal: lookup compound correlation
    # ------------------------------------------------------------------

    def _get_compound_correlation(
        self,
        hazard_a: str,
        hazard_b: str,
    ) -> float:
        """Retrieve the compound hazard correlation factor for a pair.

        Looks up the unordered pair (hazard_a, hazard_b) in the
        correlation table.  Returns a default of 0.30 when the pair
        is not found.

        Args:
            hazard_a: First hazard type string.
            hazard_b: Second hazard type string.

        Returns:
            Correlation factor in [0, 1].
        """
        key = frozenset({hazard_a.lower(), hazard_b.lower()})
        return self._compound_correlations.get(key, 0.30)

    # ==================================================================
    # PUBLIC API -- 1. calculate_risk_index
    # ==================================================================

    def calculate_risk_index(
        self,
        hazard_type: str,
        location: Dict[str, Any],
        probability: float,
        intensity: float,
        frequency: float,
        duration_days: float,
        scenario: Optional[str] = None,
        time_horizon: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate a single-hazard composite climate risk index.

        Validates inputs, computes the weighted risk score, classifies
        the risk level, records provenance, and stores the result in
        the in-memory index.

        Zero-Hallucination: all numeric outputs derive from explicit
        Python arithmetic.

        Args:
            hazard_type: Climate hazard identifier (e.g.
                ``"riverine_flood"``, ``"drought"``).
            location: Dictionary with ``lat``, ``lon``, and optionally
                ``name`` keys.
            probability: Hazard occurrence probability in [0, 1].
            intensity: Hazard intensity on a 0-10 scale.
            frequency: Expected events per year.
            duration_days: Average event duration in days.
            scenario: Optional SSP/RCP scenario label (e.g.
                ``"SSP2-4.5"``).
            time_horizon: Optional time horizon label (e.g.
                ``"MID_TERM"``).

        Returns:
            Deep-copied dictionary containing:
                - ``index_id``: Unique risk index identifier.
                - ``hazard_type``: Hazard type string.
                - ``location``: Normalised location dict.
                - ``risk_score``: Composite score in [0, 100].
                - ``risk_level``: Risk classification string.
                - ``component_scores``: Per-component breakdown.
                - ``scenario``: SSP/RCP scenario or ``None``.
                - ``time_horizon``: Time horizon or ``None``.
                - ``calculated_at``: ISO UTC timestamp.
                - ``provenance_hash``: SHA-256 audit hash.

        Raises:
            ValueError: If any risk component is invalid.
        """
        start = time.monotonic()

        # -- Validate inputs ------------------------------------------------
        errors = self._validate_risk_components(
            probability, intensity, frequency, duration_days,
        )
        if not hazard_type or not isinstance(hazard_type, str):
            errors.append("hazard_type must be a non-empty string")
        if not location or not isinstance(location, dict):
            errors.append("location must be a non-empty dictionary")

        if errors:
            raise ValueError(
                "Risk index validation failed: " + "; ".join(errors)
            )

        # -- Normalise location ---------------------------------------------
        norm_location = self._normalise_location(location)
        hazard_str = hazard_type.lower().strip()

        # -- Compute score (ZERO HALLUCINATION) -----------------------------
        risk_score = self._compute_risk_score(
            probability, intensity, frequency, duration_days,
        )
        risk_level = self.classify_risk_level(risk_score)

        # -- Build component scores -----------------------------------------
        component_scores = self._build_component_scores(
            probability, intensity, frequency, duration_days,
        )

        # -- Build result ---------------------------------------------------
        index_id = self._generate_index_id()
        calculated_at = _utcnow().isoformat()

        # -- Provenance -----------------------------------------------------
        provenance_data = {
            "index_id": index_id,
            "hazard_type": hazard_str,
            "location": norm_location,
            "probability": probability,
            "intensity": intensity,
            "frequency": frequency,
            "duration_days": duration_days,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "scenario": scenario,
            "time_horizon": time_horizon,
        }
        provenance_hash = self._record_provenance(
            entity_type="risk_index",
            action="calculate_risk",
            entity_id=index_id,
            data=provenance_data,
            metadata={
                "hazard_type": hazard_str,
                "risk_score": risk_score,
                "risk_level": risk_level,
            },
        ) or _hash_data(provenance_data)

        result: Dict[str, Any] = {
            "index_id": index_id,
            "hazard_type": hazard_str,
            "location": norm_location,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "component_scores": component_scores,
            "scenario": scenario,
            "time_horizon": time_horizon,
            "calculated_at": calculated_at,
            "provenance_hash": provenance_hash,
        }

        # -- Store in memory ------------------------------------------------
        with self._lock:
            self._indices[index_id] = copy.deepcopy(result)
            self._total_calculations += 1

        # -- Metrics --------------------------------------------------------
        _safe_record_risk_calculation(hazard_str, risk_level)

        elapsed = (time.monotonic() - start) * 1000.0
        logger.info(
            "Risk index calculated: id=%s hazard=%s score=%.2f level=%s "
            "location=%s elapsed=%.1fms",
            index_id,
            hazard_str,
            risk_score,
            risk_level,
            norm_location.get("name", "unknown"),
            elapsed,
        )

        return copy.deepcopy(result)

    # ==================================================================
    # PUBLIC API -- 2. calculate_multi_hazard_index
    # ==================================================================

    def calculate_multi_hazard_index(
        self,
        location: Dict[str, Any],
        hazard_risks: List[Dict[str, Any]],
        aggregation: str = "weighted_average",
    ) -> Dict[str, Any]:
        """Calculate a multi-hazard composite risk index for a location.

        Computes per-hazard risk scores then aggregates them using the
        specified strategy into a single composite score.

        Args:
            location: Dictionary with ``lat``, ``lon``, and optionally
                ``name`` keys.
            hazard_risks: List of dicts, each containing:
                - ``hazard_type`` (str): Hazard identifier.
                - ``probability`` (float): Probability [0, 1].
                - ``intensity`` (float): Intensity [0, 10].
                - ``frequency`` (float): Events per year.
                - ``duration_days`` (float): Duration in days.
                - ``weight`` (float, optional): Weight for
                  weighted_average aggregation (default 1.0).
            aggregation: Aggregation strategy -- one of
                ``"weighted_average"``, ``"maximum"``, or
                ``"sum_capped"``.

        Returns:
            Deep-copied dictionary containing:
                - ``multi_hazard_id``: Unique identifier.
                - ``location``: Normalised location.
                - ``multi_hazard_score``: Composite score [0, 100].
                - ``multi_hazard_level``: Risk classification.
                - ``per_hazard_scores``: List of per-hazard results.
                - ``dominant_hazard``: Hazard with the highest score.
                - ``aggregation``: Strategy used.
                - ``hazard_count``: Number of hazards assessed.
                - ``calculated_at``: ISO UTC timestamp.
                - ``provenance_hash``: SHA-256 audit hash.

        Raises:
            ValueError: If ``hazard_risks`` is empty or aggregation
                strategy is unknown.
        """
        start = time.monotonic()

        # -- Validate -------------------------------------------------------
        if not hazard_risks:
            raise ValueError("hazard_risks must contain at least one entry")
        if not location or not isinstance(location, dict):
            raise ValueError("location must be a non-empty dictionary")

        agg_lower = aggregation.lower().strip()
        valid_strategies = {"weighted_average", "maximum", "sum_capped"}
        if agg_lower not in valid_strategies:
            raise ValueError(
                f"aggregation must be one of {sorted(valid_strategies)}, "
                f"got '{aggregation}'"
            )

        # -- Normalise location ---------------------------------------------
        norm_location = self._normalise_location(location)

        # -- Compute per-hazard scores --------------------------------------
        per_hazard: List[Dict[str, Any]] = []
        for hr in hazard_risks:
            hazard_type = hr.get("hazard_type", "unknown")
            prob = float(hr.get("probability", 0.0))
            inten = float(hr.get("intensity", 0.0))
            freq = float(hr.get("frequency", 0.0))
            dur = float(hr.get("duration_days", 0.0))
            weight = float(hr.get("weight", 1.0))

            score = self._compute_risk_score(prob, inten, freq, dur)
            level = self.classify_risk_level(score)
            components = self._build_component_scores(
                prob, inten, freq, dur,
            )

            per_hazard.append({
                "hazard_type": str(hazard_type).lower().strip(),
                "risk_score": score,
                "risk_level": level,
                "weight": round(weight, 6),
                "component_scores": components,
            })

        # -- Aggregate scores -----------------------------------------------
        multi_score = self._aggregate_scores(per_hazard, agg_lower)
        multi_level = self.classify_risk_level(multi_score)

        # -- Identify dominant hazard (highest score) -----------------------
        dominant = self._find_dominant_hazard(per_hazard)

        # -- Build result ---------------------------------------------------
        multi_id = f"MH-{uuid4().hex[:12]}"
        calculated_at = _utcnow().isoformat()

        provenance_data = {
            "multi_hazard_id": multi_id,
            "location": norm_location,
            "multi_hazard_score": multi_score,
            "aggregation": agg_lower,
            "hazard_count": len(per_hazard),
        }
        provenance_hash = self._record_provenance(
            entity_type="risk_index",
            action="calculate_multi_hazard",
            entity_id=multi_id,
            data=provenance_data,
            metadata={
                "multi_hazard_score": multi_score,
                "multi_hazard_level": multi_level,
                "hazard_count": len(per_hazard),
            },
        ) or _hash_data(provenance_data)

        result: Dict[str, Any] = {
            "multi_hazard_id": multi_id,
            "location": norm_location,
            "multi_hazard_score": multi_score,
            "multi_hazard_level": multi_level,
            "per_hazard_scores": per_hazard,
            "dominant_hazard": dominant,
            "aggregation": agg_lower,
            "hazard_count": len(per_hazard),
            "calculated_at": calculated_at,
            "provenance_hash": provenance_hash,
        }

        # -- Store and metrics ----------------------------------------------
        with self._lock:
            self._indices[multi_id] = copy.deepcopy(result)
            self._total_multi_calculations += 1

        _safe_record_risk_calculation("multi_hazard", multi_level)

        elapsed = (time.monotonic() - start) * 1000.0
        logger.info(
            "Multi-hazard index calculated: id=%s score=%.2f level=%s "
            "hazards=%d agg=%s location=%s elapsed=%.1fms",
            multi_id,
            multi_score,
            multi_level,
            len(per_hazard),
            agg_lower,
            norm_location.get("name", "unknown"),
            elapsed,
        )

        return copy.deepcopy(result)

    # ------------------------------------------------------------------
    # Internal: aggregation strategies
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate_scores(
        per_hazard: List[Dict[str, Any]],
        strategy: str,
    ) -> float:
        """Aggregate per-hazard scores using the named strategy.

        Args:
            per_hazard: List of per-hazard score dicts (must contain
                ``risk_score`` and ``weight`` keys).
            strategy: One of ``"weighted_average"``, ``"maximum"``,
                ``"sum_capped"``.

        Returns:
            Aggregated risk score in [0, 100].
        """
        if not per_hazard:
            return 0.0

        scores = [h["risk_score"] for h in per_hazard]

        if strategy == "maximum":
            return max(scores)

        if strategy == "sum_capped":
            return min(sum(scores), 100.0)

        # weighted_average (default)
        weights = [h.get("weight", 1.0) for h in per_hazard]
        total_weight = sum(weights)
        if total_weight <= 0.0:
            return sum(scores) / len(scores) if scores else 0.0

        weighted_sum = sum(
            s * w for s, w in zip(scores, weights)
        )
        result = weighted_sum / total_weight
        return round(max(0.0, min(result, 100.0)), 4)

    @staticmethod
    def _find_dominant_hazard(
        per_hazard: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Find the hazard with the highest risk score.

        Args:
            per_hazard: List of per-hazard score dicts.

        Returns:
            Dictionary with ``hazard_type``, ``risk_score``, and
            ``risk_level`` of the dominant hazard.
        """
        if not per_hazard:
            return {
                "hazard_type": "none",
                "risk_score": 0.0,
                "risk_level": RiskLevel.NEGLIGIBLE.value,
            }

        dominant = max(per_hazard, key=lambda h: h["risk_score"])
        return {
            "hazard_type": dominant["hazard_type"],
            "risk_score": dominant["risk_score"],
            "risk_level": dominant["risk_level"],
        }

    # ==================================================================
    # PUBLIC API -- 3. calculate_compound_risk
    # ==================================================================

    def calculate_compound_risk(
        self,
        location: Dict[str, Any],
        primary_hazard: Dict[str, Any],
        secondary_hazards: List[Dict[str, Any]],
        correlation_factors: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Calculate compound event risk for correlated hazards.

        Models cascading/concurrent hazard events where one hazard
        amplifies another.  The amplified score is:

            compound_score = primary_score * (1 + sum(
                secondary_score_i * correlation_i
            ))

        The result is capped at 100.

        Args:
            location: Dictionary with ``lat``, ``lon``, ``name``.
            primary_hazard: Dict with ``hazard_type``, ``probability``,
                ``intensity``, ``frequency``, ``duration_days``.
            secondary_hazards: List of dicts with the same keys as
                ``primary_hazard``.
            correlation_factors: Optional mapping of secondary hazard
                types to custom correlation factors.  Falls back to
                the built-in correlation table when not supplied.

        Returns:
            Deep-copied dictionary containing:
                - ``compound_id``: Unique identifier.
                - ``location``: Normalised location.
                - ``primary_hazard``: Primary hazard score dict.
                - ``secondary_hazards``: List of secondary score dicts.
                - ``compound_score``: Amplified score [0, 100].
                - ``compound_level``: Risk classification.
                - ``amplification_factor``: Total amplification applied.
                - ``correlation_details``: Per-secondary correlation.
                - ``calculated_at``: ISO UTC timestamp.
                - ``provenance_hash``: SHA-256 audit hash.

        Raises:
            ValueError: If primary_hazard is invalid or empty.
        """
        start = time.monotonic()

        # -- Validate -------------------------------------------------------
        if not primary_hazard or not isinstance(primary_hazard, dict):
            raise ValueError("primary_hazard must be a non-empty dictionary")
        if not location or not isinstance(location, dict):
            raise ValueError("location must be a non-empty dictionary")

        norm_location = self._normalise_location(location)
        custom_corr = correlation_factors or {}

        # -- Primary hazard score -------------------------------------------
        primary_type = str(
            primary_hazard.get("hazard_type", "unknown")
        ).lower().strip()
        primary_score = self._compute_risk_score(
            float(primary_hazard.get("probability", 0.0)),
            float(primary_hazard.get("intensity", 0.0)),
            float(primary_hazard.get("frequency", 0.0)),
            float(primary_hazard.get("duration_days", 0.0)),
        )
        primary_level = self.classify_risk_level(primary_score)
        primary_components = self._build_component_scores(
            float(primary_hazard.get("probability", 0.0)),
            float(primary_hazard.get("intensity", 0.0)),
            float(primary_hazard.get("frequency", 0.0)),
            float(primary_hazard.get("duration_days", 0.0)),
        )

        primary_result: Dict[str, Any] = {
            "hazard_type": primary_type,
            "risk_score": primary_score,
            "risk_level": primary_level,
            "component_scores": primary_components,
        }

        # -- Secondary hazards and correlations -----------------------------
        secondary_results: List[Dict[str, Any]] = []
        correlation_details: List[Dict[str, Any]] = []
        amplification_sum: float = 0.0

        secondary_list = secondary_hazards or []
        for sh in secondary_list:
            sec_type = str(
                sh.get("hazard_type", "unknown")
            ).lower().strip()
            sec_score = self._compute_risk_score(
                float(sh.get("probability", 0.0)),
                float(sh.get("intensity", 0.0)),
                float(sh.get("frequency", 0.0)),
                float(sh.get("duration_days", 0.0)),
            )
            sec_level = self.classify_risk_level(sec_score)
            sec_components = self._build_component_scores(
                float(sh.get("probability", 0.0)),
                float(sh.get("intensity", 0.0)),
                float(sh.get("frequency", 0.0)),
                float(sh.get("duration_days", 0.0)),
            )

            # Determine correlation factor
            if sec_type in custom_corr:
                corr = float(custom_corr[sec_type])
            else:
                corr = self._get_compound_correlation(
                    primary_type, sec_type,
                )

            corr = max(0.0, min(corr, 1.0))
            normalised_sec = sec_score / 100.0
            amplification = normalised_sec * corr
            amplification_sum += amplification

            secondary_results.append({
                "hazard_type": sec_type,
                "risk_score": sec_score,
                "risk_level": sec_level,
                "component_scores": sec_components,
            })

            correlation_details.append({
                "secondary_hazard": sec_type,
                "secondary_score": sec_score,
                "correlation_factor": round(corr, 4),
                "amplification_contribution": round(amplification, 6),
            })

        # -- Compound score calculation -------------------------------------
        amplification_factor = round(1.0 + amplification_sum, 6)
        compound_score = primary_score * amplification_factor
        compound_score = round(max(0.0, min(compound_score, 100.0)), 4)
        compound_level = self.classify_risk_level(compound_score)

        # -- Build result ---------------------------------------------------
        compound_id = f"CR-{uuid4().hex[:12]}"
        calculated_at = _utcnow().isoformat()

        provenance_data = {
            "compound_id": compound_id,
            "location": norm_location,
            "primary_type": primary_type,
            "primary_score": primary_score,
            "compound_score": compound_score,
            "amplification_factor": amplification_factor,
            "secondary_count": len(secondary_results),
        }
        provenance_hash = self._record_provenance(
            entity_type="risk_index",
            action="calculate_compound",
            entity_id=compound_id,
            data=provenance_data,
            metadata={
                "compound_score": compound_score,
                "compound_level": compound_level,
                "amplification_factor": amplification_factor,
            },
        ) or _hash_data(provenance_data)

        result: Dict[str, Any] = {
            "compound_id": compound_id,
            "location": norm_location,
            "primary_hazard": primary_result,
            "secondary_hazards": secondary_results,
            "compound_score": compound_score,
            "compound_level": compound_level,
            "amplification_factor": amplification_factor,
            "correlation_details": correlation_details,
            "calculated_at": calculated_at,
            "provenance_hash": provenance_hash,
        }

        # -- Store and metrics ----------------------------------------------
        with self._lock:
            self._indices[compound_id] = copy.deepcopy(result)
            self._total_compound_calculations += 1

        _safe_record_risk_calculation("compound_hazard", compound_level)

        elapsed = (time.monotonic() - start) * 1000.0
        logger.info(
            "Compound risk calculated: id=%s primary=%s score=%.2f "
            "compound_score=%.2f level=%s amplification=%.4f "
            "secondaries=%d elapsed=%.1fms",
            compound_id,
            primary_type,
            primary_score,
            compound_score,
            compound_level,
            amplification_factor,
            len(secondary_results),
            elapsed,
        )

        return copy.deepcopy(result)

    # ==================================================================
    # PUBLIC API -- 4. rank_hazards
    # ==================================================================

    def rank_hazards(
        self,
        location: Dict[str, Any],
        hazard_risks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Rank hazards by risk score in descending order for a location.

        Computes the risk score for each hazard and returns the sorted
        list from highest to lowest risk.

        Args:
            location: Dictionary with ``lat``, ``lon``, ``name``.
            hazard_risks: List of dicts, each containing:
                - ``hazard_type`` (str)
                - ``probability`` (float)
                - ``intensity`` (float)
                - ``frequency`` (float)
                - ``duration_days`` (float)

        Returns:
            Deep-copied list of dicts sorted descending by
            ``risk_score``, each containing ``rank``,
            ``hazard_type``, ``risk_score``, ``risk_level``,
            ``component_scores``, and ``location``.

        Raises:
            ValueError: If hazard_risks is empty.
        """
        start = time.monotonic()

        if not hazard_risks:
            raise ValueError("hazard_risks must contain at least one entry")
        if not location or not isinstance(location, dict):
            raise ValueError("location must be a non-empty dictionary")

        norm_location = self._normalise_location(location)

        scored: List[Dict[str, Any]] = []
        for hr in hazard_risks:
            hazard_type = str(hr.get("hazard_type", "unknown")).lower().strip()
            prob = float(hr.get("probability", 0.0))
            inten = float(hr.get("intensity", 0.0))
            freq = float(hr.get("frequency", 0.0))
            dur = float(hr.get("duration_days", 0.0))

            score = self._compute_risk_score(prob, inten, freq, dur)
            level = self.classify_risk_level(score)
            components = self._build_component_scores(
                prob, inten, freq, dur,
            )

            scored.append({
                "hazard_type": hazard_type,
                "risk_score": score,
                "risk_level": level,
                "component_scores": components,
                "location": norm_location,
            })

        # Sort descending by risk_score, then alphabetically by hazard_type
        scored.sort(
            key=lambda h: (-h["risk_score"], h["hazard_type"]),
        )

        # Add rank
        for i, entry in enumerate(scored, start=1):
            entry["rank"] = i

        # -- Provenance -----------------------------------------------------
        provenance_data = {
            "location": norm_location,
            "hazard_count": len(scored),
            "top_hazard": scored[0]["hazard_type"] if scored else "none",
        }
        prov_hash = self._record_provenance(
            entity_type="risk_index",
            action="rank_hazards",
            entity_id=f"RANK-{uuid4().hex[:12]}",
            data=provenance_data,
        )

        with self._lock:
            self._total_rankings += 1

        _safe_record_risk_calculation("rank", "n/a")

        elapsed = (time.monotonic() - start) * 1000.0
        logger.info(
            "Hazards ranked: location=%s count=%d top=%s score=%.2f "
            "elapsed=%.1fms",
            norm_location.get("name", "unknown"),
            len(scored),
            scored[0]["hazard_type"] if scored else "none",
            scored[0]["risk_score"] if scored else 0.0,
            elapsed,
        )

        return copy.deepcopy(scored)

    # ==================================================================
    # PUBLIC API -- 5. compare_locations
    # ==================================================================

    def compare_locations(
        self,
        locations_with_risks: List[Dict[str, Any]],
        hazard_type: str,
    ) -> List[Dict[str, Any]]:
        """Compare risk across multiple locations for a specific hazard.

        Calculates the risk score for the given hazard type at each
        location and returns results sorted from highest to lowest risk.

        Args:
            locations_with_risks: List of dicts, each containing:
                - ``location`` (dict): With ``lat``, ``lon``, ``name``.
                - ``probability`` (float)
                - ``intensity`` (float)
                - ``frequency`` (float)
                - ``duration_days`` (float)
            hazard_type: The hazard type to evaluate across all
                locations.

        Returns:
            Deep-copied list of dicts sorted descending by
            ``risk_score``, each containing ``rank``,
            ``location``, ``hazard_type``, ``risk_score``,
            ``risk_level``, and ``component_scores``.

        Raises:
            ValueError: If locations_with_risks is empty or hazard_type
                is invalid.
        """
        start = time.monotonic()

        if not locations_with_risks:
            raise ValueError(
                "locations_with_risks must contain at least one entry"
            )
        if not hazard_type or not isinstance(hazard_type, str):
            raise ValueError("hazard_type must be a non-empty string")

        hazard_str = hazard_type.lower().strip()
        compared: List[Dict[str, Any]] = []

        for entry in locations_with_risks:
            loc = entry.get("location", {})
            norm_location = self._normalise_location(loc)

            prob = float(entry.get("probability", 0.0))
            inten = float(entry.get("intensity", 0.0))
            freq = float(entry.get("frequency", 0.0))
            dur = float(entry.get("duration_days", 0.0))

            score = self._compute_risk_score(prob, inten, freq, dur)
            level = self.classify_risk_level(score)
            components = self._build_component_scores(
                prob, inten, freq, dur,
            )

            compared.append({
                "location": norm_location,
                "hazard_type": hazard_str,
                "risk_score": score,
                "risk_level": level,
                "component_scores": components,
            })

        # Sort descending by risk_score, then by location name
        compared.sort(
            key=lambda c: (-c["risk_score"], c["location"].get("name", "")),
        )

        # Add rank
        for i, entry in enumerate(compared, start=1):
            entry["rank"] = i

        # -- Provenance -----------------------------------------------------
        provenance_data = {
            "hazard_type": hazard_str,
            "location_count": len(compared),
            "highest_risk_location": (
                compared[0]["location"].get("name", "unknown")
                if compared
                else "none"
            ),
        }
        self._record_provenance(
            entity_type="risk_index",
            action="calculate_risk",
            entity_id=f"CMP-{uuid4().hex[:12]}",
            data=provenance_data,
        )

        with self._lock:
            self._total_comparisons += 1

        _safe_record_risk_calculation("compare_locations", "n/a")

        elapsed = (time.monotonic() - start) * 1000.0
        logger.info(
            "Locations compared: hazard=%s count=%d highest=%s "
            "score=%.2f elapsed=%.1fms",
            hazard_str,
            len(compared),
            compared[0]["location"].get("name", "unknown") if compared else "none",
            compared[0]["risk_score"] if compared else 0.0,
            elapsed,
        )

        return copy.deepcopy(compared)

    # ==================================================================
    # PUBLIC API -- 6. get_risk_trend
    # ==================================================================

    def get_risk_trend(
        self,
        hazard_type: str,
        location: Dict[str, Any],
        risk_snapshots: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Analyse risk evolution over successive time horizons.

        Computes the risk score at each snapshot, determines the trend
        direction, and calculates the rate of change.

        Trend Direction:
            - ``increasing``: Last score > first score by > 2 points.
            - ``decreasing``: Last score < first score by > 2 points.
            - ``stable``: Absolute change <= 2 points.

        Args:
            hazard_type: Climate hazard identifier.
            location: Dictionary with ``lat``, ``lon``, ``name``.
            risk_snapshots: Ordered list of dicts (earliest first),
                each containing:
                - ``time_horizon`` (str): Label (e.g. ``"2030"``).
                - ``probability`` (float)
                - ``intensity`` (float)
                - ``frequency`` (float)
                - ``duration_days`` (float)

        Returns:
            Deep-copied dictionary containing:
                - ``trend_id``: Unique identifier.
                - ``hazard_type``: Hazard type string.
                - ``location``: Normalised location.
                - ``direction``: ``"increasing"``/``"decreasing"``/
                  ``"stable"``.
                - ``rate_of_change``: Score change per snapshot
                  interval.
                - ``total_change``: Absolute score difference first
                  to last.
                - ``snapshots``: List of scored snapshots with
                  ``time_horizon``, ``risk_score``, ``risk_level``.
                - ``first_score``: Score at earliest snapshot.
                - ``last_score``: Score at latest snapshot.
                - ``min_score``: Minimum score across all snapshots.
                - ``max_score``: Maximum score across all snapshots.
                - ``calculated_at``: ISO UTC timestamp.
                - ``provenance_hash``: SHA-256 audit hash.

        Raises:
            ValueError: If risk_snapshots is empty.
        """
        start = time.monotonic()

        if not risk_snapshots:
            raise ValueError(
                "risk_snapshots must contain at least one entry"
            )
        if not hazard_type or not isinstance(hazard_type, str):
            raise ValueError("hazard_type must be a non-empty string")
        if not location or not isinstance(location, dict):
            raise ValueError("location must be a non-empty dictionary")

        hazard_str = hazard_type.lower().strip()
        norm_location = self._normalise_location(location)

        # -- Compute score at each snapshot ---------------------------------
        snapshots: List[Dict[str, Any]] = []
        for snap in risk_snapshots:
            horizon = str(snap.get("time_horizon", "unknown"))
            prob = float(snap.get("probability", 0.0))
            inten = float(snap.get("intensity", 0.0))
            freq = float(snap.get("frequency", 0.0))
            dur = float(snap.get("duration_days", 0.0))

            score = self._compute_risk_score(prob, inten, freq, dur)
            level = self.classify_risk_level(score)

            snapshots.append({
                "time_horizon": horizon,
                "risk_score": score,
                "risk_level": level,
                "probability": round(prob, 6),
                "intensity": round(inten, 6),
                "frequency": round(freq, 6),
                "duration_days": round(dur, 6),
            })

        # -- Trend analysis -------------------------------------------------
        scores = [s["risk_score"] for s in snapshots]
        first_score = scores[0]
        last_score = scores[-1]
        total_change = round(last_score - first_score, 4)

        # Rate of change per interval
        num_intervals = len(scores) - 1
        if num_intervals > 0:
            rate_of_change = round(total_change / num_intervals, 4)
        else:
            rate_of_change = 0.0

        # Direction with a 2-point stability threshold
        abs_change = abs(total_change)
        if abs_change <= 2.0:
            direction = TrendDirection.STABLE.value
        elif total_change > 0:
            direction = TrendDirection.INCREASING.value
        else:
            direction = TrendDirection.DECREASING.value

        min_score = round(min(scores), 4)
        max_score = round(max(scores), 4)

        # -- Build result ---------------------------------------------------
        trend_id = f"TR-{uuid4().hex[:12]}"
        calculated_at = _utcnow().isoformat()

        provenance_data = {
            "trend_id": trend_id,
            "hazard_type": hazard_str,
            "location": norm_location,
            "direction": direction,
            "total_change": total_change,
            "snapshot_count": len(snapshots),
        }
        provenance_hash = self._record_provenance(
            entity_type="risk_index",
            action="calculate_risk",
            entity_id=trend_id,
            data=provenance_data,
            metadata={
                "direction": direction,
                "rate_of_change": rate_of_change,
            },
        ) or _hash_data(provenance_data)

        result: Dict[str, Any] = {
            "trend_id": trend_id,
            "hazard_type": hazard_str,
            "location": norm_location,
            "direction": direction,
            "rate_of_change": rate_of_change,
            "total_change": total_change,
            "snapshots": snapshots,
            "first_score": round(first_score, 4),
            "last_score": round(last_score, 4),
            "min_score": min_score,
            "max_score": max_score,
            "calculated_at": calculated_at,
            "provenance_hash": provenance_hash,
        }

        with self._lock:
            self._indices[trend_id] = copy.deepcopy(result)
            self._total_trends += 1

        _safe_record_risk_calculation("trend_analysis", direction)

        elapsed = (time.monotonic() - start) * 1000.0
        logger.info(
            "Risk trend analysed: id=%s hazard=%s direction=%s "
            "change=%.2f rate=%.4f snapshots=%d elapsed=%.1fms",
            trend_id,
            hazard_str,
            direction,
            total_change,
            rate_of_change,
            len(snapshots),
            elapsed,
        )

        return copy.deepcopy(result)

    # ==================================================================
    # PUBLIC API -- 7. classify_risk_level
    # ==================================================================

    def classify_risk_level(self, score: float) -> str:
        """Convert a numeric risk score (0-100) to a risk level string.

        Uses the configured thresholds (defaulting to EXTREME >= 80,
        HIGH >= 60, MEDIUM >= 40, LOW >= 20, NEGLIGIBLE < 20).

        Args:
            score: Composite risk score in [0, 100].

        Returns:
            Risk level string: ``"EXTREME"``, ``"HIGH"``,
            ``"MEDIUM"``, ``"LOW"``, or ``"NEGLIGIBLE"``.
        """
        if score >= self._threshold_extreme:
            return RiskLevel.EXTREME.value
        if score >= self._threshold_high:
            return RiskLevel.HIGH.value
        if score >= self._threshold_medium:
            return RiskLevel.MEDIUM.value
        if score >= self._threshold_low:
            return RiskLevel.LOW.value
        return RiskLevel.NEGLIGIBLE.value

    # ==================================================================
    # PUBLIC API -- 8. get_risk_index
    # ==================================================================

    def get_risk_index(
        self,
        index_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a stored risk index by its unique identifier.

        Args:
            index_id: The identifier of the risk index to retrieve
                (e.g. ``"RI-abcdef123456"``).

        Returns:
            Deep-copied risk index dictionary, or ``None`` if the
            identifier is not found.
        """
        if not index_id:
            return None

        with self._lock:
            record = self._indices.get(index_id)

        if record is None:
            logger.debug(
                "get_risk_index: index_id=%s not found", index_id,
            )
            return None

        return copy.deepcopy(record)

    # ==================================================================
    # PUBLIC API -- 9. list_risk_indices
    # ==================================================================

    def list_risk_indices(
        self,
        hazard_type: Optional[str] = None,
        risk_level: Optional[str] = None,
        location: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List stored risk indices with optional filters.

        Filters are applied with AND semantics.  Results are returned
        in insertion order (newest last).

        Args:
            hazard_type: Optional hazard type filter (case-insensitive).
            risk_level: Optional risk level filter (case-insensitive).
            location: Optional location name substring filter
                (case-insensitive).
            limit: Maximum number of results to return (default 100).

        Returns:
            Deep-copied list of matching risk index dictionaries,
            limited to ``limit`` entries.
        """
        with self._lock:
            indices = list(self._indices.values())

        # -- Apply filters --------------------------------------------------
        if hazard_type:
            ht_lower = hazard_type.lower().strip()
            indices = [
                idx for idx in indices
                if self._match_hazard_type(idx, ht_lower)
            ]

        if risk_level:
            rl_upper = risk_level.upper().strip()
            indices = [
                idx for idx in indices
                if self._match_risk_level(idx, rl_upper)
            ]

        if location:
            loc_lower = location.lower().strip()
            indices = [
                idx for idx in indices
                if self._match_location_name(idx, loc_lower)
            ]

        # -- Limit and return -----------------------------------------------
        limited = indices[:limit] if limit > 0 else indices

        logger.debug(
            "list_risk_indices: hazard_type=%s risk_level=%s "
            "location=%s limit=%d -> %d results",
            hazard_type,
            risk_level,
            location,
            limit,
            len(limited),
        )

        return copy.deepcopy(limited)

    # ------------------------------------------------------------------
    # Internal: filter matchers
    # ------------------------------------------------------------------

    @staticmethod
    def _match_hazard_type(record: Dict[str, Any], ht_lower: str) -> bool:
        """Check if a record matches a hazard type filter.

        Checks the ``hazard_type`` key directly, and also inspects
        compound/multi-hazard records that may store the type
        differently.

        Args:
            record: Risk index record dict.
            ht_lower: Lower-cased hazard type to match.

        Returns:
            True if the record matches.
        """
        # Direct hazard_type field
        if record.get("hazard_type", "").lower() == ht_lower:
            return True

        # Multi-hazard: check per_hazard_scores
        for sub in record.get("per_hazard_scores", []):
            if sub.get("hazard_type", "").lower() == ht_lower:
                return True

        # Compound: check primary_hazard and secondary_hazards
        primary = record.get("primary_hazard", {})
        if primary.get("hazard_type", "").lower() == ht_lower:
            return True
        for sec in record.get("secondary_hazards", []):
            if sec.get("hazard_type", "").lower() == ht_lower:
                return True

        return False

    @staticmethod
    def _match_risk_level(record: Dict[str, Any], rl_upper: str) -> bool:
        """Check if a record matches a risk level filter.

        Checks ``risk_level``, ``multi_hazard_level``, and
        ``compound_level`` keys.

        Args:
            record: Risk index record dict.
            rl_upper: Upper-cased risk level to match.

        Returns:
            True if the record matches.
        """
        if record.get("risk_level", "").upper() == rl_upper:
            return True
        if record.get("multi_hazard_level", "").upper() == rl_upper:
            return True
        if record.get("compound_level", "").upper() == rl_upper:
            return True
        return False

    @staticmethod
    def _match_location_name(
        record: Dict[str, Any],
        loc_lower: str,
    ) -> bool:
        """Check if a record's location name contains the filter substring.

        Args:
            record: Risk index record dict.
            loc_lower: Lower-cased location substring to search for.

        Returns:
            True if the location name contains the substring.
        """
        loc = record.get("location", {})
        name = str(loc.get("name", "")).lower()
        return loc_lower in name

    # ==================================================================
    # PUBLIC API -- 10. get_high_risk_summary
    # ==================================================================

    def get_high_risk_summary(
        self,
        threshold: float = 60.0,
    ) -> Dict[str, Any]:
        """Get a summary of all risk indices above a score threshold.

        Scans all stored indices and returns those with a risk score
        at or above the threshold.

        Args:
            threshold: Minimum score to include (default 60.0).

        Returns:
            Deep-copied dictionary containing:
                - ``threshold``: The threshold used.
                - ``total_indices``: Total indices in store.
                - ``high_risk_count``: Number above threshold.
                - ``high_risk_indices``: List of matching indices.
                - ``by_risk_level``: Count per risk level.
                - ``by_hazard_type``: Count per hazard type.
                - ``highest_score``: Maximum score found.
                - ``average_score``: Mean score of matching indices.
                - ``generated_at``: ISO UTC timestamp.
                - ``provenance_hash``: SHA-256 hash.
        """
        start = time.monotonic()

        with self._lock:
            all_indices = list(self._indices.values())

        high_risk: List[Dict[str, Any]] = []
        for idx in all_indices:
            score = self._extract_score(idx)
            if score >= threshold:
                high_risk.append(idx)

        # -- Counts by risk level -------------------------------------------
        by_level: Dict[str, int] = {}
        for idx in high_risk:
            level = self._extract_level(idx)
            by_level[level] = by_level.get(level, 0) + 1

        # -- Counts by hazard type ------------------------------------------
        by_hazard: Dict[str, int] = {}
        for idx in high_risk:
            ht = self._extract_hazard_type(idx)
            by_hazard[ht] = by_hazard.get(ht, 0) + 1

        # -- Score statistics -----------------------------------------------
        scores = [self._extract_score(idx) for idx in high_risk]
        highest = max(scores) if scores else 0.0
        average = round(sum(scores) / len(scores), 4) if scores else 0.0

        # -- Sort by score descending ---------------------------------------
        high_risk.sort(
            key=lambda h: -self._extract_score(h),
        )

        # -- Build result ---------------------------------------------------
        generated_at = _utcnow().isoformat()

        provenance_data = {
            "threshold": threshold,
            "total_indices": len(all_indices),
            "high_risk_count": len(high_risk),
            "highest_score": highest,
        }
        provenance_hash = self._record_provenance(
            entity_type="risk_index",
            action="calculate_risk",
            entity_id=f"HRS-{uuid4().hex[:12]}",
            data=provenance_data,
        ) or _hash_data(provenance_data)

        result: Dict[str, Any] = {
            "threshold": threshold,
            "total_indices": len(all_indices),
            "high_risk_count": len(high_risk),
            "high_risk_indices": high_risk,
            "by_risk_level": by_level,
            "by_hazard_type": by_hazard,
            "highest_score": round(highest, 4),
            "average_score": average,
            "generated_at": generated_at,
            "provenance_hash": provenance_hash,
        }

        elapsed = (time.monotonic() - start) * 1000.0
        logger.info(
            "High risk summary: threshold=%.1f total=%d high=%d "
            "highest=%.2f avg=%.2f elapsed=%.1fms",
            threshold,
            len(all_indices),
            len(high_risk),
            highest,
            average,
            elapsed,
        )

        return copy.deepcopy(result)

    # ------------------------------------------------------------------
    # Internal: score and level extractors
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_score(record: Dict[str, Any]) -> float:
        """Extract the primary risk score from any record type.

        Checks ``risk_score``, ``multi_hazard_score``, and
        ``compound_score`` keys in priority order.

        Args:
            record: Risk index record dict.

        Returns:
            The extracted risk score, or 0.0 if not found.
        """
        if "risk_score" in record:
            return float(record["risk_score"])
        if "multi_hazard_score" in record:
            return float(record["multi_hazard_score"])
        if "compound_score" in record:
            return float(record["compound_score"])
        return 0.0

    @staticmethod
    def _extract_level(record: Dict[str, Any]) -> str:
        """Extract the primary risk level from any record type.

        Checks ``risk_level``, ``multi_hazard_level``, and
        ``compound_level`` keys in priority order.

        Args:
            record: Risk index record dict.

        Returns:
            The extracted risk level string, or ``"UNKNOWN"``.
        """
        if "risk_level" in record:
            return str(record["risk_level"])
        if "multi_hazard_level" in record:
            return str(record["multi_hazard_level"])
        if "compound_level" in record:
            return str(record["compound_level"])
        return "UNKNOWN"

    @staticmethod
    def _extract_hazard_type(record: Dict[str, Any]) -> str:
        """Extract the primary hazard type string from any record type.

        For multi-hazard records, returns ``"multi_hazard"``.  For
        compound records, returns the primary hazard type.

        Args:
            record: Risk index record dict.

        Returns:
            Hazard type string.
        """
        if "hazard_type" in record:
            return str(record["hazard_type"])
        if "multi_hazard_id" in record:
            return "multi_hazard"
        primary = record.get("primary_hazard", {})
        if "hazard_type" in primary:
            return str(primary["hazard_type"])
        return "unknown"

    # ==================================================================
    # PUBLIC API -- 11. get_statistics
    # ==================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Return engine statistics and operational counters.

        Returns:
            Deep-copied dictionary containing:
                - ``total_indices``: Number of stored indices.
                - ``total_calculations``: Single-hazard calculations.
                - ``total_multi_calculations``: Multi-hazard.
                - ``total_compound_calculations``: Compound risk.
                - ``total_rankings``: Ranking operations performed.
                - ``total_comparisons``: Location comparisons.
                - ``total_trends``: Trend analyses performed.
                - ``by_risk_level``: Count of indices per risk level.
                - ``by_hazard_type``: Count of indices per hazard type.
                - ``score_distribution``: Min, max, mean, median score.
                - ``weights``: Currently active risk weights.
                - ``thresholds``: Currently active risk thresholds.
                - ``provenance_entries``: Provenance entry count.
                - ``created_at``: Engine creation timestamp.
                - ``generated_at``: Stats generation timestamp.
        """
        with self._lock:
            all_indices = list(self._indices.values())
            total_calcs = self._total_calculations
            total_multi = self._total_multi_calculations
            total_compound = self._total_compound_calculations
            total_rankings = self._total_rankings
            total_comparisons = self._total_comparisons
            total_trends = self._total_trends

        # -- Counts by risk level -------------------------------------------
        by_level: Dict[str, int] = {}
        scores: List[float] = []
        by_hazard: Dict[str, int] = {}

        for idx in all_indices:
            level = self._extract_level(idx)
            by_level[level] = by_level.get(level, 0) + 1

            ht = self._extract_hazard_type(idx)
            by_hazard[ht] = by_hazard.get(ht, 0) + 1

            score = self._extract_score(idx)
            scores.append(score)

        # -- Score distribution ---------------------------------------------
        if scores:
            sorted_scores = sorted(scores)
            n = len(sorted_scores)
            min_score = sorted_scores[0]
            max_score = sorted_scores[-1]
            mean_score = round(sum(sorted_scores) / n, 4)
            if n % 2 == 0:
                median_score = round(
                    (sorted_scores[n // 2 - 1] + sorted_scores[n // 2]) / 2.0,
                    4,
                )
            else:
                median_score = sorted_scores[n // 2]
        else:
            min_score = 0.0
            max_score = 0.0
            mean_score = 0.0
            median_score = 0.0

        # -- Provenance count -----------------------------------------------
        provenance_entries = 0
        if self._provenance is not None:
            try:
                provenance_entries = self._provenance.entry_count
            except Exception:
                pass

        result: Dict[str, Any] = {
            "total_indices": len(all_indices),
            "total_calculations": total_calcs,
            "total_multi_calculations": total_multi,
            "total_compound_calculations": total_compound,
            "total_rankings": total_rankings,
            "total_comparisons": total_comparisons,
            "total_trends": total_trends,
            "by_risk_level": by_level,
            "by_hazard_type": by_hazard,
            "score_distribution": {
                "min": round(min_score, 4),
                "max": round(max_score, 4),
                "mean": mean_score,
                "median": round(median_score, 4),
                "count": len(scores),
            },
            "weights": {
                "probability": self._weight_probability,
                "intensity": self._weight_intensity,
                "frequency": self._weight_frequency,
                "duration": self._weight_duration,
            },
            "thresholds": {
                "extreme": self._threshold_extreme,
                "high": self._threshold_high,
                "medium": self._threshold_medium,
                "low": self._threshold_low,
            },
            "provenance_entries": provenance_entries,
            "created_at": self._created_at,
            "generated_at": _utcnow().isoformat(),
        }

        logger.debug(
            "get_statistics: total=%d calcs=%d multi=%d compound=%d "
            "rankings=%d comparisons=%d trends=%d",
            len(all_indices),
            total_calcs,
            total_multi,
            total_compound,
            total_rankings,
            total_comparisons,
            total_trends,
        )

        return copy.deepcopy(result)

    # ==================================================================
    # PUBLIC API -- 12. clear
    # ==================================================================

    def clear(self) -> Dict[str, Any]:
        """Reset all engine state, clearing stored indices and counters.

        Returns:
            Dictionary confirming the operation with ``cleared_count``
            and ``cleared_at`` fields.
        """
        with self._lock:
            count = len(self._indices)
            self._indices.clear()
            self._total_calculations = 0
            self._total_multi_calculations = 0
            self._total_compound_calculations = 0
            self._total_rankings = 0
            self._total_comparisons = 0
            self._total_trends = 0

        # Reset provenance tracker if available
        if self._provenance is not None:
            try:
                self._provenance.reset()
            except Exception as exc:
                logger.warning(
                    "RiskIndexEngine: failed to reset provenance: %s", exc
                )

        cleared_at = _utcnow().isoformat()

        self._record_provenance(
            entity_type="risk_index",
            action="clear_engine",
            entity_id="risk_index_engine",
            metadata={"cleared_count": count},
        )

        logger.info(
            "RiskIndexEngine cleared: %d indices removed at %s",
            count,
            cleared_at,
        )

        return {
            "cleared_count": count,
            "cleared_at": cleared_at,
        }

    # ==================================================================
    # PUBLIC API -- Additional utility methods
    # ==================================================================

    def get_weights(self) -> Dict[str, float]:
        """Return the currently active risk index weights.

        Returns:
            Dictionary with ``probability``, ``intensity``,
            ``frequency``, and ``duration`` weights.
        """
        return {
            "probability": self._weight_probability,
            "intensity": self._weight_intensity,
            "frequency": self._weight_frequency,
            "duration": self._weight_duration,
        }

    def get_thresholds(self) -> Dict[str, float]:
        """Return the currently active risk level thresholds.

        Returns:
            Dictionary with ``extreme``, ``high``, ``medium``,
            and ``low`` thresholds.
        """
        return {
            "extreme": self._threshold_extreme,
            "high": self._threshold_high,
            "medium": self._threshold_medium,
            "low": self._threshold_low,
        }

    def set_weights(
        self,
        probability: Optional[float] = None,
        intensity: Optional[float] = None,
        frequency: Optional[float] = None,
        duration: Optional[float] = None,
    ) -> Dict[str, float]:
        """Update risk index weights.

        Only non-None arguments are applied.  After update, validates
        that all weights are in [0, 1] and that they sum to 1.0
        (with 1e-6 tolerance).

        Args:
            probability: New probability weight.
            intensity: New intensity weight.
            frequency: New frequency weight.
            duration: New duration weight.

        Returns:
            Updated weights dictionary.

        Raises:
            ValueError: If weights violate constraints after update.
        """
        with self._lock:
            if probability is not None:
                self._weight_probability = probability
            if intensity is not None:
                self._weight_intensity = intensity
            if frequency is not None:
                self._weight_frequency = frequency
            if duration is not None:
                self._weight_duration = duration

            # Validate
            weights = [
                self._weight_probability,
                self._weight_intensity,
                self._weight_frequency,
                self._weight_duration,
            ]
            for w in weights:
                if w < 0.0 or w > 1.0:
                    raise ValueError(
                        f"All weights must be in [0, 1]; got {weights}"
                    )
            if abs(sum(weights) - 1.0) > 1e-6:
                raise ValueError(
                    f"Weights must sum to 1.0; got {sum(weights):.6f}"
                )

        logger.info(
            "Risk weights updated: prob=%.2f int=%.2f freq=%.2f dur=%.2f",
            self._weight_probability,
            self._weight_intensity,
            self._weight_frequency,
            self._weight_duration,
        )

        return self.get_weights()

    def set_thresholds(
        self,
        extreme: Optional[float] = None,
        high: Optional[float] = None,
        medium: Optional[float] = None,
        low: Optional[float] = None,
    ) -> Dict[str, float]:
        """Update risk level classification thresholds.

        Only non-None arguments are applied.  After update, validates
        that the ordering extreme > high > medium > low > 0 holds.

        Args:
            extreme: New extreme threshold.
            high: New high threshold.
            medium: New medium threshold.
            low: New low threshold.

        Returns:
            Updated thresholds dictionary.

        Raises:
            ValueError: If thresholds violate ordering constraints.
        """
        with self._lock:
            if extreme is not None:
                self._threshold_extreme = extreme
            if high is not None:
                self._threshold_high = high
            if medium is not None:
                self._threshold_medium = medium
            if low is not None:
                self._threshold_low = low

            # Validate ordering
            if not (
                self._threshold_extreme > self._threshold_high
                > self._threshold_medium > self._threshold_low > 0.0
            ):
                raise ValueError(
                    "Thresholds must satisfy extreme > high > medium > low > 0; "
                    f"got extreme={self._threshold_extreme}, "
                    f"high={self._threshold_high}, "
                    f"medium={self._threshold_medium}, "
                    f"low={self._threshold_low}"
                )

        logger.info(
            "Risk thresholds updated: extreme=%.1f high=%.1f "
            "medium=%.1f low=%.1f",
            self._threshold_extreme,
            self._threshold_high,
            self._threshold_medium,
            self._threshold_low,
        )

        return self.get_thresholds()

    def get_compound_correlations(self) -> Dict[str, float]:
        """Return the current compound hazard correlation table.

        Returns:
            Dictionary mapping ``"hazard_a + hazard_b"`` strings to
            correlation factors.
        """
        result: Dict[str, float] = {}
        with self._lock:
            for key, val in self._compound_correlations.items():
                sorted_types = sorted(key)
                label = " + ".join(sorted_types)
                result[label] = val
        return result

    def set_compound_correlation(
        self,
        hazard_a: str,
        hazard_b: str,
        correlation: float,
    ) -> None:
        """Set or update a compound hazard correlation factor.

        Args:
            hazard_a: First hazard type.
            hazard_b: Second hazard type.
            correlation: Correlation factor in [0, 1].

        Raises:
            ValueError: If correlation is outside [0, 1].
        """
        if correlation < 0.0 or correlation > 1.0:
            raise ValueError(
                f"correlation must be in [0, 1], got {correlation}"
            )

        key = frozenset({hazard_a.lower(), hazard_b.lower()})
        with self._lock:
            self._compound_correlations[key] = correlation

        logger.info(
            "Compound correlation updated: %s + %s = %.4f",
            hazard_a,
            hazard_b,
            correlation,
        )

    def calculate_risk_score_raw(
        self,
        probability: float,
        intensity: float,
        frequency: float,
        duration_days: float,
    ) -> float:
        """Calculate risk score without storing the result.

        Convenience method for callers that need only the numeric
        score without index creation, provenance recording, or
        in-memory storage.

        Args:
            probability: Hazard probability in [0, 1].
            intensity: Hazard intensity on a 0-10 scale.
            frequency: Events per year.
            duration_days: Duration in days.

        Returns:
            Composite risk score in [0, 100].
        """
        return self._compute_risk_score(
            probability, intensity, frequency, duration_days,
        )

    def batch_calculate_risk_indices(
        self,
        hazard_entries: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Calculate risk indices for a batch of hazard entries.

        Each entry in the list must have the same structure as the
        arguments to :meth:`calculate_risk_index`, passed as
        dictionary keys.

        Args:
            hazard_entries: List of dicts, each containing:
                - ``hazard_type`` (str)
                - ``location`` (dict)
                - ``probability`` (float)
                - ``intensity`` (float)
                - ``frequency`` (float)
                - ``duration_days`` (float)
                - ``scenario`` (str, optional)
                - ``time_horizon`` (str, optional)

        Returns:
            List of risk index result dicts corresponding to each
            input entry.  Entries that fail validation are represented
            as error dicts with ``status="error"`` and ``message``.
        """
        start = time.monotonic()
        results: List[Dict[str, Any]] = []

        for i, entry in enumerate(hazard_entries):
            try:
                result = self.calculate_risk_index(
                    hazard_type=str(entry.get("hazard_type", "unknown")),
                    location=entry.get("location", {}),
                    probability=float(entry.get("probability", 0.0)),
                    intensity=float(entry.get("intensity", 0.0)),
                    frequency=float(entry.get("frequency", 0.0)),
                    duration_days=float(entry.get("duration_days", 0.0)),
                    scenario=entry.get("scenario"),
                    time_horizon=entry.get("time_horizon"),
                )
                results.append(result)
            except Exception as exc:
                logger.warning(
                    "batch_calculate_risk_indices: entry[%d] failed: %s",
                    i,
                    exc,
                )
                results.append({
                    "status": "error",
                    "index": i,
                    "message": str(exc),
                    "hazard_type": str(entry.get("hazard_type", "unknown")),
                })

        elapsed = (time.monotonic() - start) * 1000.0
        success_count = sum(
            1 for r in results if r.get("status") != "error"
        )
        logger.info(
            "Batch risk calculation complete: total=%d success=%d "
            "errors=%d elapsed=%.1fms",
            len(hazard_entries),
            success_count,
            len(hazard_entries) - success_count,
            elapsed,
        )

        return results

    def get_risk_level_distribution(self) -> Dict[str, Any]:
        """Get the distribution of stored indices across risk levels.

        Returns:
            Deep-copied dictionary with:
                - ``total``: Total indices.
                - ``distribution``: Dict mapping risk levels to
                  counts and percentages.
                - ``generated_at``: ISO UTC timestamp.
        """
        with self._lock:
            all_indices = list(self._indices.values())

        total = len(all_indices)
        counts: Dict[str, int] = {
            RiskLevel.NEGLIGIBLE.value: 0,
            RiskLevel.LOW.value: 0,
            RiskLevel.MEDIUM.value: 0,
            RiskLevel.HIGH.value: 0,
            RiskLevel.EXTREME.value: 0,
        }

        for idx in all_indices:
            level = self._extract_level(idx)
            if level in counts:
                counts[level] += 1
            else:
                counts[level] = counts.get(level, 0) + 1

        distribution: Dict[str, Dict[str, Any]] = {}
        for level, count in counts.items():
            pct = round((count / total * 100.0) if total > 0 else 0.0, 2)
            distribution[level] = {
                "count": count,
                "percentage": pct,
            }

        return copy.deepcopy({
            "total": total,
            "distribution": distribution,
            "generated_at": _utcnow().isoformat(),
        })

    def get_hazard_type_summary(self) -> Dict[str, Any]:
        """Get a summary of risk scores grouped by hazard type.

        Returns:
            Deep-copied dictionary mapping each hazard type to
            count, min, max, and mean scores.
        """
        with self._lock:
            all_indices = list(self._indices.values())

        by_hazard: Dict[str, List[float]] = {}
        for idx in all_indices:
            ht = self._extract_hazard_type(idx)
            score = self._extract_score(idx)
            if ht not in by_hazard:
                by_hazard[ht] = []
            by_hazard[ht].append(score)

        summary: Dict[str, Dict[str, Any]] = {}
        for ht, scores in by_hazard.items():
            summary[ht] = {
                "count": len(scores),
                "min_score": round(min(scores), 4),
                "max_score": round(max(scores), 4),
                "mean_score": round(sum(scores) / len(scores), 4),
            }

        return copy.deepcopy({
            "hazard_types": summary,
            "total_types": len(summary),
            "generated_at": _utcnow().isoformat(),
        })

    def get_location_risk_profile(
        self,
        location_name: str,
    ) -> Dict[str, Any]:
        """Get all risk indices for a specific location by name.

        Args:
            location_name: Location name to search for
                (case-insensitive substring match).

        Returns:
            Deep-copied dictionary with matching indices, score
            summary, and dominant hazard.
        """
        matching = self.list_risk_indices(
            location=location_name,
            limit=1000,
        )

        scores = [self._extract_score(idx) for idx in matching]
        hazard_types = [self._extract_hazard_type(idx) for idx in matching]

        # Find dominant hazard
        if scores:
            max_idx = scores.index(max(scores))
            dominant = hazard_types[max_idx]
            max_score = max(scores)
            min_score = min(scores)
            avg_score = round(sum(scores) / len(scores), 4)
        else:
            dominant = "none"
            max_score = 0.0
            min_score = 0.0
            avg_score = 0.0

        return copy.deepcopy({
            "location_name": location_name,
            "index_count": len(matching),
            "indices": matching,
            "dominant_hazard": dominant,
            "score_summary": {
                "min": round(min_score, 4),
                "max": round(max_score, 4),
                "mean": avg_score,
            },
            "generated_at": _utcnow().isoformat(),
        })

    def validate_risk_components(
        self,
        probability: float,
        intensity: float,
        frequency: float,
        duration_days: float,
    ) -> Dict[str, Any]:
        """Validate risk component values without calculating a score.

        Useful for pre-flight validation before submitting data.

        Args:
            probability: Hazard probability.
            intensity: Hazard intensity.
            frequency: Hazard frequency.
            duration_days: Hazard duration.

        Returns:
            Dictionary with ``is_valid`` (bool), ``errors`` (list of
            strings), and normalised values.
        """
        errors = self._validate_risk_components(
            probability, intensity, frequency, duration_days,
        )
        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "normalised": {
                "probability": round(
                    self._clamp_probability(probability), 6
                ) if not errors else None,
                "intensity": round(
                    self._normalise_intensity(intensity), 6
                ) if not errors else None,
                "frequency": round(
                    self._normalise_frequency(frequency), 6
                ) if not errors else None,
                "duration": round(
                    self._normalise_duration(duration_days), 6
                ) if not errors else None,
            },
        }

    def delete_risk_index(self, index_id: str) -> bool:
        """Delete a stored risk index by identifier.

        Args:
            index_id: The identifier of the risk index to delete.

        Returns:
            True if the index was found and deleted, False otherwise.
        """
        with self._lock:
            if index_id in self._indices:
                del self._indices[index_id]
                logger.info(
                    "Risk index deleted: id=%s", index_id,
                )
                self._record_provenance(
                    entity_type="risk_index",
                    action="clear_engine",
                    entity_id=index_id,
                    metadata={"action_detail": "single_delete"},
                )
                return True

        logger.debug(
            "delete_risk_index: id=%s not found", index_id,
        )
        return False

    def get_index_count(self) -> int:
        """Return the total number of stored risk indices.

        Returns:
            Integer count of indices in the in-memory store.
        """
        with self._lock:
            return len(self._indices)

    def export_indices(
        self,
        format: str = "dict",
    ) -> Any:
        """Export all stored risk indices.

        Args:
            format: Export format. ``"dict"`` returns a list of
                dictionaries. ``"json"`` returns a JSON string.

        Returns:
            Exported indices in the requested format.
        """
        with self._lock:
            indices = list(self._indices.values())

        exported = copy.deepcopy(indices)

        if format == "json":
            return json.dumps(exported, indent=2, default=str)

        return exported

    def import_indices(
        self,
        indices: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Import risk indices into the engine store.

        Indices must have an ``index_id`` (or ``multi_hazard_id`` or
        ``compound_id``) key.  Existing indices with matching IDs are
        overwritten.

        Args:
            indices: List of risk index dictionaries to import.

        Returns:
            Summary dict with ``imported``, ``skipped``, and ``total``.
        """
        imported = 0
        skipped = 0

        for idx in indices:
            idx_id = (
                idx.get("index_id")
                or idx.get("multi_hazard_id")
                or idx.get("compound_id")
                or idx.get("trend_id")
            )
            if not idx_id:
                skipped += 1
                continue

            with self._lock:
                self._indices[idx_id] = copy.deepcopy(idx)
            imported += 1

        logger.info(
            "Imported %d risk indices, skipped %d", imported, skipped,
        )

        return {
            "imported": imported,
            "skipped": skipped,
            "total": imported + skipped,
        }

    # ==================================================================
    # Dunder methods
    # ==================================================================

    def __len__(self) -> int:
        """Return the number of stored risk indices.

        Returns:
            Integer count of indices in the in-memory store.
        """
        with self._lock:
            return len(self._indices)

    def __repr__(self) -> str:
        """Return a developer-friendly representation.

        Returns:
            String showing index count and configuration summary.
        """
        with self._lock:
            count = len(self._indices)
        return (
            f"RiskIndexEngine("
            f"indices={count}, "
            f"weights=(prob={self._weight_probability:.2f}, "
            f"int={self._weight_intensity:.2f}, "
            f"freq={self._weight_frequency:.2f}, "
            f"dur={self._weight_duration:.2f}), "
            f"thresholds=(ext={self._threshold_extreme:.0f}, "
            f"high={self._threshold_high:.0f}, "
            f"med={self._threshold_medium:.0f}, "
            f"low={self._threshold_low:.0f}), "
            f"provenance={self._provenance is not None})"
        )

    def __contains__(self, index_id: str) -> bool:
        """Check if a risk index exists by identifier.

        Args:
            index_id: The identifier to check.

        Returns:
            True if the index exists in the store.
        """
        with self._lock:
            return index_id in self._indices

    def __iter__(self):
        """Iterate over stored risk index IDs.

        Yields:
            Risk index identifier strings.
        """
        with self._lock:
            keys = list(self._indices.keys())
        return iter(keys)
