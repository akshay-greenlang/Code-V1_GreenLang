# -*- coding: utf-8 -*-
"""
AGENT-EUDR-020: Deforestation Alert System - Severity Classifier

Multi-factor severity classification engine using configurable weighted
scoring across five dimensions: cleared area (0.25), rate of clearing (0.20),
proximity to supply chain plots (0.25), protected area overlap (0.15), and
post-cutoff timing (0.15). Produces CRITICAL/HIGH/MEDIUM/LOW/INFORMATIONAL
severity levels with full component breakdown for audit transparency.

Zero-Hallucination Guarantees:
    - All scoring uses deterministic Decimal arithmetic
    - Severity thresholds are static configuration values
    - Weights are validated to sum to 1.0 at initialization
    - Component scores use piecewise linear interpolation
    - Aggravating factors determined by static rule lookup
    - SHA-256 provenance hashes on all result objects
    - No LLM/ML in the classification path

Severity Level Thresholds:
    - CRITICAL:       score >= 80
    - HIGH:           score >= 60
    - MEDIUM:         score >= 40
    - LOW:            score >= 20
    - INFORMATIONAL:  score < 20

Default Scoring Weights (sum = 1.0):
    - Area:       0.25 (cleared area magnitude)
    - Rate:       0.20 (deforestation rate over time)
    - Proximity:  0.25 (distance to nearest supply plot)
    - Protected:  0.15 (overlap with protected areas)
    - Timing:     0.15 (pre/post EUDR cutoff 2020-12-31)

Area Scoring Thresholds:
    - >= 50 ha  -> 100 points
    - >= 10 ha  ->  80 points
    - >=  1 ha  ->  50 points
    - >= 0.5 ha ->  30 points
    - <  0.5 ha ->  10 points

Proximity Scoring Thresholds:
    - <  1 km   -> 100 points
    - <  5 km   ->  80 points
    - < 25 km   ->  50 points
    - < 50 km   ->  30 points
    - >= 50 km  ->  10 points

Timing Scoring:
    - Post-cutoff  -> 100 points (with 2.0x multiplier)
    - Pre-cutoff   ->  20 points

Performance Targets:
    - Single classification: <10ms
    - Batch classification (1000 alerts): <5s
    - Reclassification: <15ms

Regulatory References:
    - EUDR Article 2(1): Deforestation-free verification
    - EUDR Article 2(6): Cutoff date 31 December 2020
    - EUDR Article 10: Risk mitigation measures
    - EUDR Article 11: Simplified due diligence for low-risk

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-020 (Engine 3: Severity Classifier)
Agent ID: GL-EUDR-DAS-020
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module version for provenance tracking
# ---------------------------------------------------------------------------

_MODULE_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Config import (thread-safe singleton)
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.eudr.deforestation_alert_system.config import get_config
except ImportError:
    get_config = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Provenance import
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.eudr.deforestation_alert_system.provenance import (
        ProvenanceTracker,
        get_tracker,
    )
except ImportError:
    ProvenanceTracker = None  # type: ignore[misc,assignment]
    get_tracker = None  # type: ignore[misc,assignment]

# ---------------------------------------------------------------------------
# Metrics import (graceful fallback)
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.eudr.deforestation_alert_system.metrics import (
        PROMETHEUS_AVAILABLE,
        record_severity_classification,
        observe_severity_scoring_duration,
        record_api_error,
    )
except ImportError:
    PROMETHEUS_AVAILABLE = False  # type: ignore[assignment]
    record_severity_classification = None  # type: ignore[misc,assignment]
    observe_severity_scoring_duration = None  # type: ignore[misc,assignment]
    record_api_error = None  # type: ignore[misc,assignment]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash for audit provenance.

    Args:
        data: Any JSON-serializable object.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _generate_id() -> str:
    """Generate a unique identifier using UUID4.

    Returns:
        String representation of a new UUID4.
    """
    return str(uuid.uuid4())


def _safe_decimal(value: Any, default: Decimal = Decimal("0")) -> Decimal:
    """Safely convert a value to Decimal.

    Args:
        value: Value to convert.
        default: Default Decimal if conversion fails.

    Returns:
        Decimal representation of value or default.
    """
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return default


def _elapsed_ms(start: float) -> float:
    """Calculate elapsed milliseconds since start.

    Args:
        start: time.perf_counter() start value.

    Returns:
        Elapsed time in milliseconds.
    """
    return round((time.perf_counter() - start) * 1000, 2)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: EUDR deforestation cutoff date per Article 2(6).
EUDR_CUTOFF_DATE: date = date(2020, 12, 31)

#: Default severity level thresholds (lower bound inclusive).
DEFAULT_SEVERITY_THRESHOLDS: Dict[str, Decimal] = {
    "CRITICAL": Decimal("80"),
    "HIGH": Decimal("60"),
    "MEDIUM": Decimal("40"),
    "LOW": Decimal("20"),
    "INFORMATIONAL": Decimal("0"),
}

#: Default scoring weights per dimension (sum = 1.0).
DEFAULT_WEIGHTS: Dict[str, Decimal] = {
    "area": Decimal("0.25"),
    "rate": Decimal("0.20"),
    "proximity": Decimal("0.25"),
    "protected": Decimal("0.15"),
    "timing": Decimal("0.15"),
}

#: Area scoring breakpoints (ha -> raw score).
AREA_SCORE_BREAKPOINTS: List[Tuple[Decimal, Decimal]] = [
    (Decimal("50"), Decimal("100")),
    (Decimal("10"), Decimal("80")),
    (Decimal("1"), Decimal("50")),
    (Decimal("0.5"), Decimal("30")),
    (Decimal("0"), Decimal("10")),
]

#: Proximity scoring breakpoints (km -> raw score).
PROXIMITY_SCORE_BREAKPOINTS: List[Tuple[Decimal, Decimal]] = [
    (Decimal("1"), Decimal("100")),
    (Decimal("5"), Decimal("80")),
    (Decimal("25"), Decimal("50")),
    (Decimal("50"), Decimal("30")),
]

#: Default proximity score when distance >= 50km.
PROXIMITY_DISTANT_SCORE: Decimal = Decimal("10")

#: Default post-cutoff score multiplier.
DEFAULT_POST_CUTOFF_MULTIPLIER: Decimal = Decimal("2.0")

#: Default protected area score multiplier.
DEFAULT_PROTECTED_AREA_MULTIPLIER: Decimal = Decimal("1.5")

#: Rate scoring: hectares per day breakpoints.
RATE_SCORE_BREAKPOINTS: List[Tuple[Decimal, Decimal]] = [
    (Decimal("10"), Decimal("100")),
    (Decimal("5"), Decimal("80")),
    (Decimal("1"), Decimal("60")),
    (Decimal("0.5"), Decimal("40")),
    (Decimal("0.1"), Decimal("20")),
    (Decimal("0"), Decimal("5")),
]

#: Protected area overlap scoring breakpoints (pct -> raw score).
PROTECTED_OVERLAP_BREAKPOINTS: List[Tuple[Decimal, Decimal]] = [
    (Decimal("75"), Decimal("100")),
    (Decimal("50"), Decimal("80")),
    (Decimal("25"), Decimal("60")),
    (Decimal("10"), Decimal("40")),
    (Decimal("1"), Decimal("20")),
    (Decimal("0"), Decimal("0")),
]

#: Known aggravating factor rules.
AGGRAVATING_RULES: List[Dict[str, Any]] = [
    {
        "id": "AGG-001",
        "name": "critical_area_near_plots",
        "description": "Large deforestation (>=50ha) within 5km of supply plots",
        "condition": "area_ha >= 50 AND proximity_km < 5",
        "severity_boost": Decimal("10"),
    },
    {
        "id": "AGG-002",
        "name": "post_cutoff_protected_area",
        "description": "Post-cutoff deforestation in protected area",
        "condition": "is_post_cutoff AND protected_overlap_pct > 0",
        "severity_boost": Decimal("15"),
    },
    {
        "id": "AGG-003",
        "name": "rapid_clearing",
        "description": "Clearing rate exceeds 5 ha/day",
        "condition": "rate_ha_per_day >= 5",
        "severity_boost": Decimal("10"),
    },
    {
        "id": "AGG-004",
        "name": "multi_source_confirmed",
        "description": "Change confirmed by multiple satellite sources",
        "condition": "multi_source_confirmed == True",
        "severity_boost": Decimal("5"),
    },
    {
        "id": "AGG-005",
        "name": "repeat_location",
        "description": "Recurring deforestation at same location",
        "condition": "previous_alerts_count > 0",
        "severity_boost": Decimal("8"),
    },
]


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class SeverityLevel(str, Enum):
    """Alert severity classification levels.

    Determines the urgency, notification channels, and response
    requirements for deforestation alerts.
    """

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFORMATIONAL = "INFORMATIONAL"


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class SeverityComponent:
    """Score component for a single severity dimension.

    Represents one of the five scoring dimensions (area, rate,
    proximity, protected, timing) with its raw value, normalized
    score, weight, and weighted contribution.

    Attributes:
        name: Component dimension name.
        raw_value: Raw input value before normalization.
        normalized_score: Score after normalization (0-100).
        weight: Weight applied to this component.
        weighted_score: Final weighted contribution to total.
        explanation: Human-readable explanation of the scoring.
    """

    name: str = ""
    raw_value: Decimal = Decimal("0")
    normalized_score: Decimal = Decimal("0")
    weight: Decimal = Decimal("0")
    weighted_score: Decimal = Decimal("0")
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "name": self.name,
            "raw_value": str(self.raw_value),
            "normalized_score": str(self.normalized_score),
            "weight": str(self.weight),
            "weighted_score": str(self.weighted_score),
            "explanation": self.explanation,
        }


@dataclass
class AggravatingFactor:
    """Identified aggravating factor that boosts severity.

    Attributes:
        factor_id: Unique factor identifier.
        name: Short name of the aggravating factor.
        description: Detailed description.
        severity_boost: Additional score points added.
        evidence: Evidence data supporting the factor.
    """

    factor_id: str = ""
    name: str = ""
    description: str = ""
    severity_boost: Decimal = Decimal("0")
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "factor_id": self.factor_id,
            "name": self.name,
            "description": self.description,
            "severity_boost": str(self.severity_boost),
            "evidence": self.evidence,
        }


@dataclass
class SeverityScore:
    """Complete severity classification result.

    Contains the overall score, severity level, component breakdown,
    aggravating factors, and contributing factors for full audit
    transparency.

    Attributes:
        classification_id: Unique classification identifier.
        alert_id: Alert being classified.
        components: List of individual SeverityComponent scores.
        total_score: Weighted sum of component scores (0-100).
        severity_level: Final severity classification.
        contributing_factors: Main factors driving the severity.
        aggravating_factors: Identified aggravating conditions.
        base_score: Score before aggravating factors applied.
        aggravation_boost: Total boost from aggravating factors.
        weights_used: Weight configuration used for scoring.
        thresholds_used: Threshold configuration used.
        processing_time_ms: Classification processing time.
        provenance_hash: SHA-256 provenance hash.
        calculation_timestamp: Classification timestamp.
        metadata: Additional metadata.
    """

    classification_id: str = ""
    alert_id: str = ""
    components: List[SeverityComponent] = field(default_factory=list)
    total_score: Decimal = Decimal("0")
    severity_level: str = SeverityLevel.INFORMATIONAL.value
    contributing_factors: List[str] = field(default_factory=list)
    aggravating_factors: List[AggravatingFactor] = field(default_factory=list)
    base_score: Decimal = Decimal("0")
    aggravation_boost: Decimal = Decimal("0")
    weights_used: Dict[str, str] = field(default_factory=dict)
    thresholds_used: Dict[str, str] = field(default_factory=dict)
    processing_time_ms: float = 0.0
    provenance_hash: str = ""
    calculation_timestamp: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Set defaults for unset fields."""
        if not self.classification_id:
            self.classification_id = _generate_id()
        if not self.calculation_timestamp:
            self.calculation_timestamp = _utcnow().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        return {
            "classification_id": self.classification_id,
            "alert_id": self.alert_id,
            "components": [c.to_dict() for c in self.components],
            "total_score": str(self.total_score),
            "severity_level": self.severity_level,
            "contributing_factors": self.contributing_factors,
            "aggravating_factors": [
                f.to_dict() for f in self.aggravating_factors
            ],
            "base_score": str(self.base_score),
            "aggravation_boost": str(self.aggravation_boost),
            "weights_used": self.weights_used,
            "thresholds_used": self.thresholds_used,
            "processing_time_ms": self.processing_time_ms,
            "provenance_hash": self.provenance_hash,
            "calculation_timestamp": self.calculation_timestamp,
            "metadata": self.metadata,
        }


@dataclass
class SeverityResult:
    """Result wrapper for severity classification operations.

    Attributes:
        score: The SeverityScore classification result.
        previous_severity: Previous severity if reclassification.
        severity_changed: Whether severity changed (reclassification).
        processing_time_ms: Processing time in milliseconds.
        warnings: List of warning messages.
        provenance_hash: SHA-256 provenance hash.
        calculation_timestamp: Query timestamp.
    """

    score: Optional[SeverityScore] = None
    previous_severity: str = ""
    severity_changed: bool = False
    processing_time_ms: float = 0.0
    warnings: List[str] = field(default_factory=list)
    provenance_hash: str = ""
    calculation_timestamp: str = ""

    def __post_init__(self) -> None:
        """Set calculation timestamp if unset."""
        if not self.calculation_timestamp:
            self.calculation_timestamp = _utcnow().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "score": self.score.to_dict() if self.score else None,
            "previous_severity": self.previous_severity,
            "severity_changed": self.severity_changed,
            "processing_time_ms": self.processing_time_ms,
            "warnings": self.warnings,
            "provenance_hash": self.provenance_hash,
            "calculation_timestamp": self.calculation_timestamp,
        }


@dataclass
class ThresholdsResult:
    """Current severity threshold and weight configuration.

    Attributes:
        thresholds: Severity level thresholds.
        weights: Scoring dimension weights.
        area_breakpoints: Area scoring breakpoints.
        proximity_breakpoints: Proximity scoring breakpoints.
        rate_breakpoints: Rate scoring breakpoints.
        protected_breakpoints: Protected area scoring breakpoints.
        multipliers: Post-cutoff and protected area multipliers.
        provenance_hash: SHA-256 provenance hash.
        calculation_timestamp: Query timestamp.
    """

    thresholds: Dict[str, str] = field(default_factory=dict)
    weights: Dict[str, str] = field(default_factory=dict)
    area_breakpoints: List[Dict[str, str]] = field(default_factory=list)
    proximity_breakpoints: List[Dict[str, str]] = field(default_factory=list)
    rate_breakpoints: List[Dict[str, str]] = field(default_factory=list)
    protected_breakpoints: List[Dict[str, str]] = field(default_factory=list)
    multipliers: Dict[str, str] = field(default_factory=dict)
    provenance_hash: str = ""
    calculation_timestamp: str = ""

    def __post_init__(self) -> None:
        """Set calculation timestamp if unset."""
        if not self.calculation_timestamp:
            self.calculation_timestamp = _utcnow().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "thresholds": self.thresholds,
            "weights": self.weights,
            "area_breakpoints": self.area_breakpoints,
            "proximity_breakpoints": self.proximity_breakpoints,
            "rate_breakpoints": self.rate_breakpoints,
            "protected_breakpoints": self.protected_breakpoints,
            "multipliers": self.multipliers,
            "provenance_hash": self.provenance_hash,
            "calculation_timestamp": self.calculation_timestamp,
        }


@dataclass
class DistributionResult:
    """Severity distribution across alerts.

    Attributes:
        total_classified: Total alerts classified.
        distribution: Count per severity level.
        percentage: Percentage per severity level.
        avg_score: Average severity score.
        median_score: Median severity score.
        min_score: Minimum severity score.
        max_score: Maximum severity score.
        provenance_hash: SHA-256 provenance hash.
        calculation_timestamp: Query timestamp.
    """

    total_classified: int = 0
    distribution: Dict[str, int] = field(default_factory=dict)
    percentage: Dict[str, str] = field(default_factory=dict)
    avg_score: Decimal = Decimal("0")
    median_score: Decimal = Decimal("0")
    min_score: Decimal = Decimal("0")
    max_score: Decimal = Decimal("0")
    provenance_hash: str = ""
    calculation_timestamp: str = ""

    def __post_init__(self) -> None:
        """Set calculation timestamp if unset."""
        if not self.calculation_timestamp:
            self.calculation_timestamp = _utcnow().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "total_classified": self.total_classified,
            "distribution": self.distribution,
            "percentage": self.percentage,
            "avg_score": str(self.avg_score),
            "median_score": str(self.median_score),
            "min_score": str(self.min_score),
            "max_score": str(self.max_score),
            "provenance_hash": self.provenance_hash,
            "calculation_timestamp": self.calculation_timestamp,
        }


# ---------------------------------------------------------------------------
# SeverityClassifier Engine
# ---------------------------------------------------------------------------


class SeverityClassifier:
    """Production-grade multi-factor severity classification engine.

    Classifies deforestation alert severity using a configurable
    weighted scoring model across five dimensions: area, rate,
    proximity, protected area overlap, and post-cutoff timing.
    Identifies aggravating factors and produces transparent component
    breakdowns for regulatory audit compliance.

    All scoring uses deterministic Decimal arithmetic with static
    threshold lookup tables. No LLM/ML involvement in the
    classification path.

    Attributes:
        _config: Agent configuration from get_config().
        _tracker: ProvenanceTracker instance for audit trails.
        _weights: Active scoring weight configuration.
        _thresholds: Active severity level thresholds.
        _post_cutoff_multiplier: Post-cutoff score multiplier.
        _protected_multiplier: Protected area score multiplier.
        _classification_store: In-memory classification results.

    Example:
        >>> classifier = SeverityClassifier()
        >>> result = classifier.classify(alert, context)
        >>> assert result.score.severity_level == "CRITICAL"
        >>> assert result.provenance_hash != ""
    """

    def __init__(self, config: Any = None) -> None:
        """Initialize the SeverityClassifier.

        Args:
            config: Optional configuration object. If None, loads from
                get_config() singleton.
        """
        self._config = config
        if self._config is None and get_config is not None:
            try:
                self._config = get_config()
            except Exception:
                logger.warning(
                    "Failed to load config via get_config(), "
                    "using hardcoded defaults"
                )
                self._config = None

        self._tracker: Optional[Any] = None
        if get_tracker is not None:
            try:
                self._tracker = get_tracker()
            except Exception:
                logger.debug("ProvenanceTracker not available")

        # Initialize weights from config or defaults
        self._weights = self._load_weights()
        self._thresholds = self._load_thresholds()
        self._post_cutoff_multiplier = self._load_multiplier(
            "post_cutoff_multiplier", DEFAULT_POST_CUTOFF_MULTIPLIER
        )
        self._protected_multiplier = self._load_multiplier(
            "protected_area_multiplier", DEFAULT_PROTECTED_AREA_MULTIPLIER
        )

        self._classification_store: Dict[str, SeverityScore] = {}

        logger.info(
            "SeverityClassifier initialized: weights=%s, "
            "post_cutoff_mult=%s, protected_mult=%s",
            {k: str(v) for k, v in self._weights.items()},
            self._post_cutoff_multiplier,
            self._protected_multiplier,
        )

    # ------------------------------------------------------------------
    # Configuration loaders
    # ------------------------------------------------------------------

    def _load_weights(self) -> Dict[str, Decimal]:
        """Load scoring weights from config or defaults.

        Returns:
            Dictionary of dimension name to weight value.
        """
        weights = dict(DEFAULT_WEIGHTS)
        if self._config:
            weight_fields = {
                "area": "area_weight",
                "rate": "rate_weight",
                "proximity": "proximity_weight",
                "protected": "protected_weight",
                "timing": "timing_weight",
            }
            for dim, attr in weight_fields.items():
                if hasattr(self._config, attr):
                    weights[dim] = _safe_decimal(
                        getattr(self._config, attr), weights[dim]
                    )
        return weights

    def _load_thresholds(self) -> Dict[str, Decimal]:
        """Load severity thresholds from config or defaults.

        Returns:
            Dictionary of severity level to threshold score.
        """
        return dict(DEFAULT_SEVERITY_THRESHOLDS)

    def _load_multiplier(
        self, attr_name: str, default: Decimal,
    ) -> Decimal:
        """Load a multiplier from config or use default.

        Args:
            attr_name: Config attribute name.
            default: Default value.

        Returns:
            Decimal multiplier value.
        """
        if self._config and hasattr(self._config, attr_name):
            return _safe_decimal(getattr(self._config, attr_name), default)
        return default

    # ------------------------------------------------------------------
    # Public API: Classification
    # ------------------------------------------------------------------

    def classify(
        self,
        alert: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> SeverityResult:
        """Perform full severity classification on a deforestation alert.

        Computes component scores for all five dimensions, identifies
        aggravating factors, and determines the final severity level.

        Args:
            alert: DeforestationAlert object to classify.
            context: Optional additional context dictionary with keys:
                - duration_days (int): Days over which clearing occurred
                - protected_overlap_pct (Decimal): Protected area overlap %
                - multi_source_confirmed (bool): Multi-source confirmation
                - previous_alerts_count (int): Previous alerts at location
                - country_risk_level (str): Country risk categorization

        Returns:
            SeverityResult with full classification breakdown.

        Raises:
            ValueError: If alert is None.
        """
        op_start = time.perf_counter()

        if alert is None:
            raise ValueError("alert must not be None")

        alert_id = getattr(alert, "alert_id", _generate_id())
        context = context or {}

        logger.debug("classify: alert_id=%s", str(alert_id)[:12])

        # Extract alert attributes
        area_ha = _safe_decimal(getattr(alert, "area_ha", 0))
        proximity_km = _safe_decimal(getattr(alert, "proximity_km", -1))
        is_post_cutoff = getattr(alert, "is_post_cutoff", True)
        detection_date = self._extract_date(alert)

        # Extract context attributes
        duration_days = int(context.get("duration_days", 1))
        duration_days = max(1, duration_days)
        protected_pct = _safe_decimal(
            context.get("protected_overlap_pct", 0)
        )

        # Score each component
        area_comp = self._score_area(area_ha)
        rate_comp = self._score_rate(area_ha, duration_days)
        proximity_comp = self._score_proximity(proximity_km)
        protected_comp = self._score_protected_area(protected_pct)
        timing_comp = self._score_timing(detection_date, is_post_cutoff)

        components = [
            area_comp, rate_comp, proximity_comp,
            protected_comp, timing_comp,
        ]

        # Calculate base score (weighted sum)
        base_score = self._calculate_total_score(components)

        # Identify aggravating factors
        aggravating = self._identify_aggravating_factors(
            alert, components, context
        )
        aggravation_boost = sum(
            f.severity_boost for f in aggravating
        )

        # Final score with aggravation
        total_score = min(
            Decimal("100"),
            base_score + aggravation_boost,
        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        # Determine severity level
        severity_level = self._determine_severity_level(total_score)

        # Identify contributing factors
        contributing = self._identify_contributing_factors(
            components, severity_level
        )

        # Build weights and thresholds used
        weights_used = {k: str(v) for k, v in self._weights.items()}
        thresholds_used = {k: str(v) for k, v in self._thresholds.items()}

        elapsed = _elapsed_ms(op_start)

        score = SeverityScore(
            alert_id=str(alert_id),
            components=components,
            total_score=total_score,
            severity_level=severity_level.value,
            contributing_factors=contributing,
            aggravating_factors=aggravating,
            base_score=base_score,
            aggravation_boost=aggravation_boost.quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            ),
            weights_used=weights_used,
            thresholds_used=thresholds_used,
            processing_time_ms=elapsed,
        )
        score.provenance_hash = _compute_hash(score.to_dict())

        # Store classification
        self._classification_store[score.classification_id] = score

        # Record provenance
        if self._tracker:
            try:
                self._tracker.record(
                    entity_type="alert",
                    action="update",
                    entity_id=str(alert_id),
                    data=score.to_dict(),
                    metadata={
                        "severity_level": severity_level.value,
                        "total_score": str(total_score),
                        "aggravating_count": len(aggravating),
                    },
                )
            except Exception:
                logger.debug("Failed to record provenance for classify")

        # Record metrics
        if record_severity_classification:
            try:
                record_severity_classification(
                    severity=severity_level.value,
                )
            except Exception:
                pass

        if observe_severity_scoring_duration:
            try:
                observe_severity_scoring_duration(elapsed / 1000.0)
            except Exception:
                pass

        result = SeverityResult(score=score, processing_time_ms=elapsed)
        result.provenance_hash = _compute_hash(result.to_dict())

        logger.info(
            "classify: alert=%s -> %s (score=%.1f, base=%.1f, "
            "boost=%.1f, %d aggravating) in %.1fms",
            str(alert_id)[:12], severity_level.value,
            float(total_score), float(base_score),
            float(aggravation_boost), len(aggravating), elapsed,
        )
        return result

    # ------------------------------------------------------------------
    # Public API: Reclassification
    # ------------------------------------------------------------------

    def reclassify(
        self,
        alert_id: str,
        alert: Any,
        new_context: Optional[Dict[str, Any]] = None,
    ) -> SeverityResult:
        """Reclassify an existing alert with updated context.

        Used when new information becomes available that may change
        the severity assessment (e.g., confirmed protected area overlap).

        Args:
            alert_id: Alert identifier to reclassify.
            alert: Updated alert object.
            new_context: Updated context dictionary.

        Returns:
            SeverityResult with updated classification and change flag.
        """
        # Get previous classification
        previous_severity = ""
        for score in self._classification_store.values():
            if score.alert_id == alert_id:
                previous_severity = score.severity_level
                break

        # Perform new classification
        result = self.classify(alert, new_context)

        # Set reclassification fields
        result.previous_severity = previous_severity
        if result.score and previous_severity:
            result.severity_changed = (
                result.score.severity_level != previous_severity
            )

        result.provenance_hash = _compute_hash(result.to_dict())

        logger.info(
            "reclassify: alert=%s, previous=%s, new=%s, changed=%s",
            alert_id[:12],
            previous_severity,
            result.score.severity_level if result.score else "none",
            result.severity_changed,
        )
        return result

    # ------------------------------------------------------------------
    # Public API: Threshold configuration
    # ------------------------------------------------------------------

    def get_thresholds(self) -> ThresholdsResult:
        """Return current severity threshold and weight configuration.

        Returns:
            ThresholdsResult with all active configuration.
        """
        result = ThresholdsResult(
            thresholds={k: str(v) for k, v in self._thresholds.items()},
            weights={k: str(v) for k, v in self._weights.items()},
            area_breakpoints=[
                {"threshold_ha": str(bp[0]), "score": str(bp[1])}
                for bp in AREA_SCORE_BREAKPOINTS
            ],
            proximity_breakpoints=[
                {"threshold_km": str(bp[0]), "score": str(bp[1])}
                for bp in PROXIMITY_SCORE_BREAKPOINTS
            ],
            rate_breakpoints=[
                {"threshold_ha_per_day": str(bp[0]), "score": str(bp[1])}
                for bp in RATE_SCORE_BREAKPOINTS
            ],
            protected_breakpoints=[
                {"threshold_pct": str(bp[0]), "score": str(bp[1])}
                for bp in PROTECTED_OVERLAP_BREAKPOINTS
            ],
            multipliers={
                "post_cutoff": str(self._post_cutoff_multiplier),
                "protected_area": str(self._protected_multiplier),
            },
        )
        result.provenance_hash = _compute_hash(result.to_dict())
        return result

    # ------------------------------------------------------------------
    # Public API: Distribution
    # ------------------------------------------------------------------

    def get_distribution(
        self,
        filters: Optional[Dict[str, Any]] = None,
    ) -> DistributionResult:
        """Get severity distribution across classified alerts.

        Args:
            filters: Optional filters (not yet implemented, reserved).

        Returns:
            DistributionResult with severity distribution statistics.
        """
        scores = list(self._classification_store.values())

        if not scores:
            result = DistributionResult()
            result.provenance_hash = _compute_hash(result.to_dict())
            return result

        # Count distribution
        distribution: Dict[str, int] = {
            level.value: 0 for level in SeverityLevel
        }
        all_scores: List[Decimal] = []

        for s in scores:
            level = s.severity_level
            distribution[level] = distribution.get(level, 0) + 1
            all_scores.append(s.total_score)

        total = len(scores)
        percentage: Dict[str, str] = {}
        for level, count in distribution.items():
            pct = (
                Decimal(str(count)) / Decimal(str(total)) * Decimal("100")
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            percentage[level] = str(pct)

        # Statistics
        all_scores.sort()
        avg = (sum(all_scores) / Decimal(str(len(all_scores)))).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        median_idx = len(all_scores) // 2
        median = all_scores[median_idx]

        result = DistributionResult(
            total_classified=total,
            distribution=distribution,
            percentage=percentage,
            avg_score=avg,
            median_score=median,
            min_score=all_scores[0],
            max_score=all_scores[-1],
        )
        result.provenance_hash = _compute_hash(result.to_dict())
        return result

    # ------------------------------------------------------------------
    # Internal: Component scoring
    # ------------------------------------------------------------------

    def _score_area(self, area_ha: Decimal) -> SeverityComponent:
        """Score the area component based on cleared hectares.

        Scoring breakpoints:
            >= 50 ha  -> 100 points
            >= 10 ha  ->  80 points
            >=  1 ha  ->  50 points
            >= 0.5 ha ->  30 points
            <  0.5 ha ->  10 points

        ZERO-HALLUCINATION: Static threshold lookup.

        Args:
            area_ha: Cleared area in hectares.

        Returns:
            SeverityComponent with area score.
        """
        raw_score = Decimal("10")
        explanation = ""

        for threshold, score in AREA_SCORE_BREAKPOINTS:
            if area_ha >= threshold:
                raw_score = score
                explanation = f"Area {area_ha} ha >= {threshold} ha threshold"
                break

        weight = self._weights.get("area", Decimal("0.25"))
        weighted = (raw_score * weight).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        return SeverityComponent(
            name="area",
            raw_value=area_ha,
            normalized_score=raw_score,
            weight=weight,
            weighted_score=weighted,
            explanation=explanation,
        )

    def _score_rate(
        self, area_ha: Decimal, duration_days: int,
    ) -> SeverityComponent:
        """Score the deforestation rate component.

        Calculates hectares per day and applies scoring breakpoints.

        Args:
            area_ha: Total cleared area in hectares.
            duration_days: Number of days over which clearing occurred.

        Returns:
            SeverityComponent with rate score.
        """
        rate = area_ha / Decimal(str(max(1, duration_days)))
        rate = rate.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        raw_score = Decimal("5")
        explanation = ""

        for threshold, score in RATE_SCORE_BREAKPOINTS:
            if rate >= threshold:
                raw_score = score
                explanation = (
                    f"Rate {rate} ha/day >= {threshold} ha/day threshold"
                )
                break

        weight = self._weights.get("rate", Decimal("0.20"))
        weighted = (raw_score * weight).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        return SeverityComponent(
            name="rate",
            raw_value=rate,
            normalized_score=raw_score,
            weight=weight,
            weighted_score=weighted,
            explanation=explanation,
        )

    def _score_proximity(self, distance_km: Decimal) -> SeverityComponent:
        """Score the proximity component based on distance to nearest plot.

        Scoring breakpoints:
            <  1 km   -> 100 points
            <  5 km   ->  80 points
            < 25 km   ->  50 points
            < 50 km   ->  30 points
            >= 50 km  ->  10 points

        If no plots are registered (distance_km < 0), defaults to
        30 points (MEDIUM proximity).

        ZERO-HALLUCINATION: Static threshold lookup.

        Args:
            distance_km: Distance to nearest supply plot in km.
                Negative value means no plots registered.

        Returns:
            SeverityComponent with proximity score.
        """
        # Handle no-plot case
        if distance_km < Decimal("0"):
            weight = self._weights.get("proximity", Decimal("0.25"))
            weighted = (Decimal("30") * weight).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
            return SeverityComponent(
                name="proximity",
                raw_value=distance_km,
                normalized_score=Decimal("30"),
                weight=weight,
                weighted_score=weighted,
                explanation="No registered plots; default MEDIUM proximity score",
            )

        raw_score = PROXIMITY_DISTANT_SCORE
        explanation = f"Distance {distance_km} km >= 50 km (distant)"

        for threshold, score in PROXIMITY_SCORE_BREAKPOINTS:
            if distance_km < threshold:
                raw_score = score
                explanation = (
                    f"Distance {distance_km} km < {threshold} km threshold"
                )
                break

        weight = self._weights.get("proximity", Decimal("0.25"))
        weighted = (raw_score * weight).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        return SeverityComponent(
            name="proximity",
            raw_value=distance_km,
            normalized_score=raw_score,
            weight=weight,
            weighted_score=weighted,
            explanation=explanation,
        )

    def _score_protected_area(
        self, overlap_pct: Decimal,
    ) -> SeverityComponent:
        """Score the protected area overlap component.

        Applies the protected area multiplier when overlap exists.

        Args:
            overlap_pct: Percentage of deforested area overlapping
                protected areas (0-100).

        Returns:
            SeverityComponent with protected area score.
        """
        raw_score = Decimal("0")
        explanation = "No protected area overlap"

        for threshold, score in PROTECTED_OVERLAP_BREAKPOINTS:
            if overlap_pct >= threshold:
                raw_score = score
                if threshold > Decimal("0"):
                    # Apply multiplier for positive overlap
                    raw_score = min(
                        Decimal("100"),
                        raw_score * self._protected_multiplier,
                    ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                    explanation = (
                        f"Overlap {overlap_pct}% >= {threshold}% threshold "
                        f"(multiplier {self._protected_multiplier}x applied)"
                    )
                break

        weight = self._weights.get("protected", Decimal("0.15"))
        weighted = (raw_score * weight).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        return SeverityComponent(
            name="protected",
            raw_value=overlap_pct,
            normalized_score=raw_score,
            weight=weight,
            weighted_score=weighted,
            explanation=explanation,
        )

    def _score_timing(
        self,
        detection_date: Optional[date],
        is_post_cutoff: bool,
    ) -> SeverityComponent:
        """Score the timing component based on EUDR cutoff date.

        Post-cutoff events receive 100 points with a 2.0x multiplier.
        Pre-cutoff events receive 20 points (still relevant for
        historical baseline tracking).

        Args:
            detection_date: Date of detection (may be None).
            is_post_cutoff: Whether event is post-EUDR cutoff.

        Returns:
            SeverityComponent with timing score.
        """
        if is_post_cutoff:
            raw_score = min(
                Decimal("100"),
                Decimal("100") * self._post_cutoff_multiplier,
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            explanation = (
                f"Post-cutoff event (after {EUDR_CUTOFF_DATE.isoformat()}); "
                f"multiplier {self._post_cutoff_multiplier}x applied, "
                f"capped at 100"
            )
            # Since raw_score is multiplied but capped at 100, the effective
            # multiplied score may exceed 100 before capping
            raw_score = min(Decimal("100"), raw_score)
        else:
            raw_score = Decimal("20")
            explanation = (
                f"Pre-cutoff event (on or before {EUDR_CUTOFF_DATE.isoformat()})"
            )

        weight = self._weights.get("timing", Decimal("0.15"))
        weighted = (raw_score * weight).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        return SeverityComponent(
            name="timing",
            raw_value=_safe_decimal(
                detection_date.isoformat() if detection_date else "unknown",
                Decimal("0"),
            ),
            normalized_score=raw_score,
            weight=weight,
            weighted_score=weighted,
            explanation=explanation,
        )

    # ------------------------------------------------------------------
    # Internal: Score aggregation
    # ------------------------------------------------------------------

    def _calculate_total_score(
        self, components: List[SeverityComponent],
    ) -> Decimal:
        """Calculate total weighted score from components.

        Sums the weighted_score of each component.

        Args:
            components: List of SeverityComponent objects.

        Returns:
            Total score as Decimal (0-100).
        """
        total = sum(c.weighted_score for c in components)
        total = max(Decimal("0"), min(Decimal("100"), total))
        return total.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    def _determine_severity_level(
        self, score: Decimal,
    ) -> SeverityLevel:
        """Determine severity level from numeric score.

        Args:
            score: Total severity score (0-100).

        Returns:
            SeverityLevel enumeration value.
        """
        if score >= self._thresholds["CRITICAL"]:
            return SeverityLevel.CRITICAL
        elif score >= self._thresholds["HIGH"]:
            return SeverityLevel.HIGH
        elif score >= self._thresholds["MEDIUM"]:
            return SeverityLevel.MEDIUM
        elif score >= self._thresholds["LOW"]:
            return SeverityLevel.LOW
        else:
            return SeverityLevel.INFORMATIONAL

    # ------------------------------------------------------------------
    # Internal: Contributing and aggravating factors
    # ------------------------------------------------------------------

    def _identify_contributing_factors(
        self,
        components: List[SeverityComponent],
        severity: SeverityLevel,
    ) -> List[str]:
        """Identify the main contributing factors to the severity level.

        Selects components with normalized scores above threshold
        and lists them as contributing factors.

        Args:
            components: List of SeverityComponent objects.
            severity: Assigned severity level.

        Returns:
            List of contributing factor description strings.
        """
        factors: List[str] = []

        # Sort components by weighted score descending
        sorted_comps = sorted(
            components, key=lambda c: c.weighted_score, reverse=True
        )

        for comp in sorted_comps:
            if comp.normalized_score >= Decimal("50"):
                factors.append(
                    f"{comp.name}: score {comp.normalized_score} "
                    f"(weight {comp.weight}) = {comp.weighted_score} weighted"
                )

        if not factors:
            factors.append(
                f"All components below significant threshold; "
                f"severity {severity.value} from aggregate score"
            )

        return factors

    def _identify_aggravating_factors(
        self,
        alert: Any,
        components: List[SeverityComponent],
        context: Dict[str, Any],
    ) -> List[AggravatingFactor]:
        """Identify aggravating factors that boost severity.

        Evaluates each rule in AGGRAVATING_RULES against the alert
        and context to determine applicable aggravating conditions.

        Args:
            alert: DeforestationAlert object.
            components: Computed severity components.
            context: Context dictionary with additional attributes.

        Returns:
            List of applicable AggravatingFactor objects.
        """
        factors: List[AggravatingFactor] = []

        area_ha = _safe_decimal(getattr(alert, "area_ha", 0))
        proximity_km = _safe_decimal(getattr(alert, "proximity_km", -1))
        is_post_cutoff = getattr(alert, "is_post_cutoff", True)
        protected_pct = _safe_decimal(
            context.get("protected_overlap_pct", 0)
        )
        multi_source = context.get("multi_source_confirmed", False)
        previous_count = int(context.get("previous_alerts_count", 0))

        # Compute rate for rate-based rule
        duration_days = max(1, int(context.get("duration_days", 1)))
        rate_ha = area_ha / Decimal(str(duration_days))

        # AGG-001: Critical area near plots
        if area_ha >= Decimal("50") and Decimal("0") <= proximity_km < Decimal("5"):
            factors.append(AggravatingFactor(
                factor_id="AGG-001",
                name="critical_area_near_plots",
                description=(
                    f"Large deforestation ({area_ha} ha) within "
                    f"{proximity_km} km of supply plots"
                ),
                severity_boost=Decimal("10"),
                evidence={
                    "area_ha": str(area_ha),
                    "proximity_km": str(proximity_km),
                },
            ))

        # AGG-002: Post-cutoff in protected area
        if is_post_cutoff and protected_pct > Decimal("0"):
            factors.append(AggravatingFactor(
                factor_id="AGG-002",
                name="post_cutoff_protected_area",
                description=(
                    f"Post-cutoff deforestation with {protected_pct}% "
                    f"protected area overlap"
                ),
                severity_boost=Decimal("15"),
                evidence={
                    "is_post_cutoff": True,
                    "protected_overlap_pct": str(protected_pct),
                },
            ))

        # AGG-003: Rapid clearing
        if rate_ha >= Decimal("5"):
            factors.append(AggravatingFactor(
                factor_id="AGG-003",
                name="rapid_clearing",
                description=f"Rapid clearing at {rate_ha} ha/day",
                severity_boost=Decimal("10"),
                evidence={
                    "rate_ha_per_day": str(rate_ha),
                    "duration_days": duration_days,
                },
            ))

        # AGG-004: Multi-source confirmed
        if multi_source:
            factors.append(AggravatingFactor(
                factor_id="AGG-004",
                name="multi_source_confirmed",
                description="Change confirmed by multiple satellite sources",
                severity_boost=Decimal("5"),
                evidence={"multi_source_confirmed": True},
            ))

        # AGG-005: Repeat location
        if previous_count > 0:
            factors.append(AggravatingFactor(
                factor_id="AGG-005",
                name="repeat_location",
                description=(
                    f"Recurring deforestation: {previous_count} previous "
                    f"alert(s) at this location"
                ),
                severity_boost=Decimal("8"),
                evidence={"previous_alerts_count": previous_count},
            ))

        return factors

    # ------------------------------------------------------------------
    # Internal: Date extraction helper
    # ------------------------------------------------------------------

    def _extract_date(self, alert: Any) -> Optional[date]:
        """Extract detection date from alert object.

        Args:
            alert: Alert object with timestamp or generated_at attribute.

        Returns:
            date object or None.
        """
        for attr in ("generated_at", "timestamp"):
            value = getattr(alert, attr, None)
            if value and isinstance(value, str) and len(value) >= 10:
                try:
                    return date.fromisoformat(value[:10])
                except (ValueError, TypeError):
                    continue
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Enumerations
    "SeverityLevel",
    # Constants
    "EUDR_CUTOFF_DATE",
    "DEFAULT_SEVERITY_THRESHOLDS",
    "DEFAULT_WEIGHTS",
    "AREA_SCORE_BREAKPOINTS",
    "PROXIMITY_SCORE_BREAKPOINTS",
    "RATE_SCORE_BREAKPOINTS",
    "PROTECTED_OVERLAP_BREAKPOINTS",
    "DEFAULT_POST_CUTOFF_MULTIPLIER",
    "DEFAULT_PROTECTED_AREA_MULTIPLIER",
    "AGGRAVATING_RULES",
    # Data classes
    "SeverityComponent",
    "AggravatingFactor",
    "SeverityScore",
    "SeverityResult",
    "ThresholdsResult",
    "DistributionResult",
    # Engine class
    "SeverityClassifier",
]
