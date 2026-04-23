# -*- coding: utf-8 -*-
"""
CondenserStateClassifier for GL-017 CONDENSYNC

Multimodal condenser health state classification combining Cleanliness Factor (CF),
Terminal Temperature Difference (TTD), approach temperature, vacuum pressure,
and air removal indicators using deterministic weighted scoring.

Standards:
- HEI Standards for Steam Surface Condensers (12th Edition)
- ASME PTC 12.2: Steam Surface Condensers
- EPRI Condenser Performance Monitoring Guidelines

Key Features:
- Multimodal parameter fusion (thermal + vacuum + air removal)
- Late fusion architecture for indicator independence
- Uncertainty quantification with calibrated bounds
- SHAP-compatible feature importance
- Deterministic rule-based classification (zero hallucination)

Zero-Hallucination Guarantee:
All classifications use deterministic weighted scoring.
No LLM or AI inference in any calculation path.
Same inputs always produce identical outputs.

Author: GL-BackendDeveloper
Date: December 2025
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class CondenserState(str, Enum):
    """Condenser operational health states per HEI Standards."""
    NORMAL = "normal"                 # Operating within design parameters
    FOULED = "fouled"                 # Tube fouling reducing heat transfer
    AIR_LEAK = "air_leak"             # Air ingress affecting vacuum
    DEGRADED = "degraded"             # Multiple minor issues
    CRITICAL = "critical"             # Immediate intervention required
    SUBCOOLED = "subcooled"           # Excessive subcooling (waste)
    STARTUP = "startup"               # Transient startup condition
    SHUTDOWN = "shutdown"             # Transient shutdown condition
    UNKNOWN = "unknown"               # Insufficient data


class ConfidenceLevel(str, Enum):
    """Confidence levels for classification."""
    HIGH = "high"           # >= 90%
    MEDIUM = "medium"       # 70-89%
    LOW = "low"             # 50-69%
    VERY_LOW = "very_low"   # < 50%


class SeverityLevel(str, Enum):
    """Severity classification for maintenance prioritization."""
    CRITICAL = "critical"   # Immediate action required
    HIGH = "high"           # Within 24 hours
    MEDIUM = "medium"       # Within 7 days
    LOW = "low"             # Next maintenance window
    NONE = "none"           # No action required


class IndicatorCategory(str, Enum):
    """Categories of condenser health indicators."""
    THERMAL = "thermal"           # CF, TTD, approach temperature
    VACUUM = "vacuum"             # Backpressure, vacuum deviation
    AIR_REMOVAL = "air_removal"   # Air leakage rate, ejector performance
    FLOW = "flow"                 # Cooling water flow, steam flow
    PERFORMANCE = "performance"   # Overall efficiency metrics


class ModalityWeight(str, Enum):
    """Weight profiles for indicator fusion."""
    THERMAL_PRIMARY = "thermal_primary"         # 50% thermal, 30% vacuum, 20% air
    VACUUM_PRIMARY = "vacuum_primary"           # 30% thermal, 50% vacuum, 20% air
    BALANCED = "balanced"                       # 40% thermal, 35% vacuum, 25% air
    AIR_SENSITIVE = "air_sensitive"             # 30% thermal, 30% vacuum, 40% air


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class ClassificationConfig:
    """
    Immutable configuration for condenser state classifier.

    Attributes:
        modality_weight_profile: Weighting scheme for indicator fusion
        confidence_threshold: Minimum confidence for valid classification
        thermal_weight: Weight for thermal indicators (0-1)
        vacuum_weight: Weight for vacuum indicators (0-1)
        air_removal_weight: Weight for air removal indicators (0-1)
        cf_normal_threshold: Cleanliness Factor threshold for normal operation
        ttd_normal_max: Maximum TTD for normal operation (degrees C)
        vacuum_deviation_max: Maximum vacuum deviation for normal (kPa)
        min_indicators_for_classification: Minimum indicators required
    """
    modality_weight_profile: ModalityWeight = ModalityWeight.BALANCED
    confidence_threshold: Decimal = Decimal("0.50")
    thermal_weight: Decimal = Decimal("0.40")
    vacuum_weight: Decimal = Decimal("0.35")
    air_removal_weight: Decimal = Decimal("0.25")
    cf_normal_threshold: float = 0.85
    ttd_normal_max: float = 5.0  # degrees C
    approach_normal_max: float = 3.0  # degrees C
    vacuum_deviation_max: float = 1.5  # kPa
    air_leak_rate_max: float = 15.0  # kg/hr
    min_indicators_for_classification: int = 2

    def __post_init__(self):
        """Validate weights sum to 1.0."""
        total = self.thermal_weight + self.vacuum_weight + self.air_removal_weight
        if abs(float(total) - 1.0) > 0.001:
            raise ValueError(f"Modality weights must sum to 1.0, got {total}")


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass(frozen=True)
class CondenserInput:
    """
    Immutable condenser sensor input data for classification.

    All temperatures in Celsius, pressures in kPa (absolute), flows in kg/hr.

    Attributes:
        condenser_id: Unique condenser identifier
        timestamp: Measurement timestamp
        cleanliness_factor: Cleanliness Factor (0-1), CF = U_actual / U_design
        ttd: Terminal Temperature Difference (C) = T_sat - T_cw_out
        approach_temp: Approach temperature (C) = T_cw_out - T_cw_in
        vacuum_pressure_kpa: Condenser vacuum pressure (kPa absolute)
        design_vacuum_kpa: Design vacuum pressure (kPa absolute)
        air_leak_rate_kg_hr: Air in-leakage rate (kg/hr)
        ejector_capacity_pct: Air ejector operating capacity (%)
        cw_inlet_temp_c: Cooling water inlet temperature (C)
        cw_outlet_temp_c: Cooling water outlet temperature (C)
        cw_flow_rate_kg_s: Cooling water flow rate (kg/s)
        steam_flow_kg_s: Steam flow to condenser (kg/s)
        hotwell_level_pct: Hotwell water level (%)
        condenser_type: Type of condenser (surface, direct_contact)
        tube_material: Tube material (titanium, stainless, admiralty_brass)
        tube_count: Number of condenser tubes
        last_cleaning_days: Days since last tube cleaning
        operating_hours: Total operating hours
    """
    condenser_id: str
    timestamp: datetime
    cleanliness_factor: Optional[float] = None
    ttd: Optional[float] = None
    approach_temp: Optional[float] = None
    vacuum_pressure_kpa: Optional[float] = None
    design_vacuum_kpa: float = 5.0
    air_leak_rate_kg_hr: Optional[float] = None
    ejector_capacity_pct: Optional[float] = None
    cw_inlet_temp_c: Optional[float] = None
    cw_outlet_temp_c: Optional[float] = None
    cw_flow_rate_kg_s: Optional[float] = None
    steam_flow_kg_s: Optional[float] = None
    hotwell_level_pct: Optional[float] = None
    condenser_type: str = "surface"
    tube_material: str = "titanium"
    tube_count: int = 10000
    last_cleaning_days: int = 0
    operating_hours: float = 0.0


@dataclass(frozen=True)
class IndicatorScore:
    """
    Score from a single indicator category.

    Attributes:
        category_name: Name of the indicator category
        state_scores: Scores for each possible state
        available: Whether this indicator had valid input data
        confidence: Confidence in this indicator's assessment
        primary_evidence: Key evidence supporting the assessment
        contributing_factors: List of factors that influenced the score
    """
    category_name: str
    state_scores: Dict[str, float]
    available: bool
    confidence: float
    primary_evidence: str
    contributing_factors: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class FeatureImportance:
    """
    Feature importance for explainability (SHAP-compatible).

    Attributes:
        feature_name: Name of the feature
        importance_score: SHAP-style importance score
        direction: Whether feature pushes toward failure or normal
        value: Actual feature value
        baseline: Expected baseline value
        threshold: Threshold value for the feature
    """
    feature_name: str
    importance_score: float
    direction: str  # "toward_degradation" or "toward_normal"
    value: float
    baseline: float
    threshold: Optional[float] = None


@dataclass(frozen=True)
class ClassificationResult:
    """
    Complete immutable classification result.

    Attributes:
        condenser_id: Condenser identifier
        timestamp: Classification timestamp
        state: Classified condenser state
        confidence_score: Numeric confidence (0-1)
        confidence_level: Qualitative confidence level
        severity: Maintenance severity level
        uncertainty_bounds: 5th/95th percentile bounds
        state_probabilities: Probabilities for all states
        indicator_scores: Scores from each indicator category
        feature_importance: Feature importance for explainability
        recommended_actions: List of recommended maintenance actions
        evidence_summary: Plain-language evidence summary
        performance_impact_pct: Estimated performance impact (%)
        provenance_hash: SHA-256 hash for audit trail
    """
    condenser_id: str
    timestamp: datetime
    state: CondenserState
    confidence_score: float
    confidence_level: ConfidenceLevel
    severity: SeverityLevel
    uncertainty_bounds: Tuple[float, float]
    state_probabilities: Dict[str, float]
    indicator_scores: List[IndicatorScore]
    feature_importance: List[FeatureImportance]
    recommended_actions: List[str]
    evidence_summary: str
    performance_impact_pct: float
    provenance_hash: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "condenser_id": self.condenser_id,
            "timestamp": self.timestamp.isoformat(),
            "state": self.state.value,
            "confidence_score": round(self.confidence_score, 4),
            "confidence_level": self.confidence_level.value,
            "severity": self.severity.value,
            "uncertainty_bounds": {
                "p05": round(self.uncertainty_bounds[0], 4),
                "p95": round(self.uncertainty_bounds[1], 4)
            },
            "state_probabilities": {
                k: round(v, 4) for k, v in self.state_probabilities.items()
            },
            "indicator_scores": [
                {
                    "category": score.category_name,
                    "confidence": round(score.confidence, 4),
                    "evidence": score.primary_evidence,
                    "available": score.available
                }
                for score in self.indicator_scores
            ],
            "feature_importance": [
                {
                    "feature": fi.feature_name,
                    "importance": round(fi.importance_score, 4),
                    "direction": fi.direction,
                    "value": round(fi.value, 4) if fi.value else None
                }
                for fi in self.feature_importance[:5]  # Top 5
            ],
            "recommended_actions": self.recommended_actions,
            "evidence_summary": self.evidence_summary,
            "performance_impact_pct": round(self.performance_impact_pct, 2),
            "provenance_hash": self.provenance_hash
        }


# =============================================================================
# REFERENCE DATA
# =============================================================================

# Steam saturation temperature at various vacuum pressures (kPa absolute -> T_sat C)
SATURATION_TEMPS_KPA: Dict[float, float] = {
    3.0: 24.1, 4.0: 29.0, 5.0: 32.9, 6.0: 36.2, 7.0: 39.0,
    8.0: 41.5, 9.0: 43.8, 10.0: 45.8, 12.0: 49.4, 15.0: 54.0,
    20.0: 60.1, 25.0: 64.9, 30.0: 69.1, 40.0: 75.9, 50.0: 81.3,
}

# CF thresholds by tube material (per HEI Standards)
CF_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "titanium": {
        "excellent": 0.95, "good": 0.90, "acceptable": 0.85,
        "marginal": 0.75, "poor": 0.65
    },
    "stainless": {
        "excellent": 0.93, "good": 0.88, "acceptable": 0.82,
        "marginal": 0.72, "poor": 0.62
    },
    "admiralty_brass": {
        "excellent": 0.90, "good": 0.85, "acceptable": 0.78,
        "marginal": 0.68, "poor": 0.55
    },
    "cupronickel": {
        "excellent": 0.92, "good": 0.87, "acceptable": 0.80,
        "marginal": 0.70, "poor": 0.58
    },
}

# Air leak rate thresholds (kg/hr) by condenser size
AIR_LEAK_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "small": {"normal": 5.0, "elevated": 10.0, "high": 20.0, "critical": 40.0},
    "medium": {"normal": 10.0, "elevated": 20.0, "high": 40.0, "critical": 80.0},
    "large": {"normal": 15.0, "elevated": 30.0, "high": 60.0, "critical": 120.0},
}

# TTD thresholds by cooling water type (seawater vs freshwater)
TTD_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "seawater": {"excellent": 2.0, "good": 3.5, "acceptable": 5.0, "marginal": 7.0, "poor": 10.0},
    "freshwater": {"excellent": 1.5, "good": 2.5, "acceptable": 4.0, "marginal": 6.0, "poor": 9.0},
    "cooling_tower": {"excellent": 2.5, "good": 4.0, "acceptable": 6.0, "marginal": 8.0, "poor": 12.0},
}

# Days since cleaning multipliers for fouling probability
CLEANING_MULTIPLIERS: Dict[int, float] = {
    0: 0.5, 30: 0.6, 60: 0.8, 90: 1.0, 120: 1.2,
    180: 1.5, 270: 2.0, 365: 2.5, 545: 3.5, 730: 5.0
}


# =============================================================================
# MAIN CLASSIFIER
# =============================================================================

class CondenserStateClassifier:
    """
    Multimodal condenser state classifier using late fusion.

    ZERO-HALLUCINATION GUARANTEE:
    - All classifications use deterministic weighted scoring
    - No LLM or ML inference in classification path
    - Same inputs always produce identical outputs
    - Complete provenance tracking with SHA-256 hashes

    Architecture:
    1. Thermal indicators: CF, TTD, approach temperature
    2. Vacuum indicators: Backpressure deviation, vacuum trend
    3. Air removal indicators: Air leak rate, ejector performance
    4. Late Fusion: Weighted combination of indicator scores

    Example:
        >>> classifier = CondenserStateClassifier()
        >>> result = classifier.classify(CondenserInput(
        ...     condenser_id="COND-001",
        ...     timestamp=datetime.now(timezone.utc),
        ...     cleanliness_factor=0.82,
        ...     ttd=6.5,
        ...     vacuum_pressure_kpa=5.2,
        ...     air_leak_rate_kg_hr=18.0
        ... ))
        >>> print(f"State: {result.state.value}")
    """

    def __init__(self, config: Optional[ClassificationConfig] = None):
        """
        Initialize condenser state classifier.

        Args:
            config: Classification configuration (uses defaults if not provided)
        """
        self.config = config or ClassificationConfig()
        self._classification_count = 0
        self._lock = threading.Lock()

        logger.info(
            f"CondenserStateClassifier initialized "
            f"(weights: thermal={self.config.thermal_weight}, "
            f"vacuum={self.config.vacuum_weight}, "
            f"air_removal={self.config.air_removal_weight})"
        )

    def classify(self, condenser_input: CondenserInput) -> ClassificationResult:
        """
        Classify condenser state from sensor inputs.

        ZERO-HALLUCINATION: Uses deterministic weighted scoring.

        Args:
            condenser_input: Condenser input data for classification

        Returns:
            ClassificationResult with complete analysis

        Raises:
            ValueError: If insufficient valid inputs provided
        """
        with self._lock:
            self._classification_count += 1

        timestamp = datetime.now(timezone.utc)

        # Score each indicator category
        thermal_score = self._score_thermal_indicators(condenser_input)
        vacuum_score = self._score_vacuum_indicators(condenser_input)
        air_removal_score = self._score_air_removal_indicators(condenser_input)

        indicator_scores = [thermal_score, vacuum_score, air_removal_score]

        # Check minimum indicators available
        available_count = sum(1 for s in indicator_scores if s.available)
        if available_count < self.config.min_indicators_for_classification:
            logger.warning(
                f"Insufficient indicators for classification: "
                f"{available_count} < {self.config.min_indicators_for_classification}"
            )
            return self._create_unknown_result(condenser_input, timestamp, indicator_scores)

        # Fuse indicator scores
        state_probs = self._fuse_indicator_scores(
            thermal_score, vacuum_score, air_removal_score
        )

        # Determine primary state
        state = CondenserState(max(state_probs.items(), key=lambda x: x[1])[0])
        confidence_score = state_probs[state.value]

        # Calculate confidence level and uncertainty
        confidence_level = self._get_confidence_level(confidence_score)
        uncertainty_bounds = self._calculate_uncertainty_bounds(
            state_probs, state, indicator_scores
        )

        # Determine severity
        severity = self._determine_severity(state, confidence_score, condenser_input)

        # Calculate feature importance
        feature_importance = self._calculate_feature_importance(
            condenser_input, thermal_score, vacuum_score, air_removal_score
        )

        # Calculate performance impact
        performance_impact = self._calculate_performance_impact(
            condenser_input, state, state_probs
        )

        # Generate recommendations and evidence
        recommended_actions = self._generate_recommendations(
            state, severity, condenser_input, indicator_scores
        )
        evidence_summary = self._generate_evidence_summary(
            state, indicator_scores, confidence_score, performance_impact
        )

        # Calculate provenance hash
        provenance_hash = self._calculate_provenance_hash(condenser_input, state_probs)

        return ClassificationResult(
            condenser_id=condenser_input.condenser_id,
            timestamp=timestamp,
            state=state,
            confidence_score=round(confidence_score, 4),
            confidence_level=confidence_level,
            severity=severity,
            uncertainty_bounds=uncertainty_bounds,
            state_probabilities=state_probs,
            indicator_scores=indicator_scores,
            feature_importance=feature_importance,
            recommended_actions=recommended_actions,
            evidence_summary=evidence_summary,
            performance_impact_pct=performance_impact,
            provenance_hash=provenance_hash
        )

    def _score_thermal_indicators(self, inp: CondenserInput) -> IndicatorScore:
        """
        Score condenser state based on thermal indicators.

        Uses CF, TTD, and approach temperature to assess heat transfer performance.
        """
        states = [s.value for s in CondenserState if s not in [CondenserState.UNKNOWN, CondenserState.STARTUP, CondenserState.SHUTDOWN]]
        scores = {s: 0.0 for s in states}
        contributing_factors = []
        available = False
        confidence = 0.0
        evidence = "No thermal data available"

        # Get tube material thresholds
        cf_thresh = CF_THRESHOLDS.get(
            inp.tube_material.lower(),
            CF_THRESHOLDS["titanium"]
        )

        # Score based on Cleanliness Factor
        if inp.cleanliness_factor is not None:
            available = True
            cf = inp.cleanliness_factor

            if cf >= cf_thresh["excellent"]:
                scores[CondenserState.NORMAL.value] += 0.45
                contributing_factors.append(f"CF excellent: {cf:.3f}")
            elif cf >= cf_thresh["good"]:
                scores[CondenserState.NORMAL.value] += 0.35
                scores[CondenserState.DEGRADED.value] += 0.05
                contributing_factors.append(f"CF good: {cf:.3f}")
            elif cf >= cf_thresh["acceptable"]:
                scores[CondenserState.NORMAL.value] += 0.20
                scores[CondenserState.FOULED.value] += 0.15
                scores[CondenserState.DEGRADED.value] += 0.10
                contributing_factors.append(f"CF acceptable: {cf:.3f}")
            elif cf >= cf_thresh["marginal"]:
                scores[CondenserState.FOULED.value] += 0.35
                scores[CondenserState.DEGRADED.value] += 0.15
                contributing_factors.append(f"CF marginal: {cf:.3f}")
            else:
                scores[CondenserState.FOULED.value] += 0.40
                scores[CondenserState.CRITICAL.value] += 0.15
                contributing_factors.append(f"CF poor: {cf:.3f}")

            evidence = f"Cleanliness Factor: {cf:.3f} ({inp.tube_material})"
            confidence += 0.35

        # Score based on TTD (Terminal Temperature Difference)
        if inp.ttd is not None:
            available = True
            ttd = inp.ttd
            ttd_thresh = TTD_THRESHOLDS.get("freshwater", TTD_THRESHOLDS["freshwater"])

            if ttd <= ttd_thresh["excellent"]:
                scores[CondenserState.NORMAL.value] += 0.25
                contributing_factors.append(f"TTD excellent: {ttd:.1f}C")
            elif ttd <= ttd_thresh["good"]:
                scores[CondenserState.NORMAL.value] += 0.20
                contributing_factors.append(f"TTD good: {ttd:.1f}C")
            elif ttd <= ttd_thresh["acceptable"]:
                scores[CondenserState.NORMAL.value] += 0.10
                scores[CondenserState.DEGRADED.value] += 0.10
                contributing_factors.append(f"TTD acceptable: {ttd:.1f}C")
            elif ttd <= ttd_thresh["marginal"]:
                scores[CondenserState.FOULED.value] += 0.20
                scores[CondenserState.DEGRADED.value] += 0.10
                contributing_factors.append(f"TTD marginal: {ttd:.1f}C")
            else:
                scores[CondenserState.FOULED.value] += 0.25
                scores[CondenserState.CRITICAL.value] += 0.10
                contributing_factors.append(f"TTD poor: {ttd:.1f}C")

            if evidence == "No thermal data available":
                evidence = f"TTD: {ttd:.1f}C"
            else:
                evidence += f", TTD: {ttd:.1f}C"
            confidence += 0.30

        # Score based on approach temperature
        if inp.approach_temp is not None:
            available = True
            approach = inp.approach_temp

            if approach <= self.config.approach_normal_max:
                scores[CondenserState.NORMAL.value] += 0.15
                contributing_factors.append(f"Approach normal: {approach:.1f}C")
            elif approach <= self.config.approach_normal_max * 1.5:
                scores[CondenserState.DEGRADED.value] += 0.10
                contributing_factors.append(f"Approach elevated: {approach:.1f}C")
            else:
                scores[CondenserState.FOULED.value] += 0.15
                contributing_factors.append(f"Approach high: {approach:.1f}C")

            confidence += 0.15

        # Check for subcooling
        if inp.ttd is not None and inp.ttd < 0:
            scores[CondenserState.SUBCOOLED.value] += 0.30
            contributing_factors.append(f"Subcooling detected: TTD={inp.ttd:.1f}C")

        # Normalize scores
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}

        return IndicatorScore(
            category_name="thermal",
            state_scores=scores,
            available=available,
            confidence=min(0.85, confidence),
            primary_evidence=evidence,
            contributing_factors=contributing_factors
        )

    def _score_vacuum_indicators(self, inp: CondenserInput) -> IndicatorScore:
        """
        Score condenser state based on vacuum indicators.

        Uses vacuum pressure deviation from design to assess condenser performance.
        """
        states = [s.value for s in CondenserState if s not in [CondenserState.UNKNOWN, CondenserState.STARTUP, CondenserState.SHUTDOWN]]
        scores = {s: 0.0 for s in states}
        contributing_factors = []
        available = False
        confidence = 0.0
        evidence = "No vacuum data available"

        if inp.vacuum_pressure_kpa is not None and inp.design_vacuum_kpa is not None:
            available = True
            actual = inp.vacuum_pressure_kpa
            design = inp.design_vacuum_kpa
            deviation = actual - design  # Positive = worse vacuum (higher pressure)
            deviation_pct = (deviation / design) * 100 if design > 0 else 0

            if deviation <= 0:
                # Better than design vacuum
                scores[CondenserState.NORMAL.value] += 0.50
                contributing_factors.append(f"Vacuum at/below design: {actual:.2f} kPa")
                evidence = f"Vacuum: {actual:.2f} kPa (design: {design:.2f} kPa)"
            elif deviation <= self.config.vacuum_deviation_max:
                # Slight deviation - acceptable
                scores[CondenserState.NORMAL.value] += 0.35
                scores[CondenserState.DEGRADED.value] += 0.10
                contributing_factors.append(f"Vacuum slight deviation: +{deviation:.2f} kPa")
                evidence = f"Vacuum deviation: +{deviation:.2f} kPa ({deviation_pct:.1f}%)"
            elif deviation <= self.config.vacuum_deviation_max * 2:
                # Moderate deviation
                scores[CondenserState.DEGRADED.value] += 0.25
                scores[CondenserState.AIR_LEAK.value] += 0.15
                scores[CondenserState.FOULED.value] += 0.10
                contributing_factors.append(f"Vacuum moderate deviation: +{deviation:.2f} kPa")
                evidence = f"Vacuum deviation: +{deviation:.2f} kPa ({deviation_pct:.1f}%)"
            elif deviation <= self.config.vacuum_deviation_max * 4:
                # Significant deviation
                scores[CondenserState.AIR_LEAK.value] += 0.30
                scores[CondenserState.FOULED.value] += 0.15
                scores[CondenserState.CRITICAL.value] += 0.10
                contributing_factors.append(f"Vacuum significant deviation: +{deviation:.2f} kPa")
                evidence = f"Vacuum deviation HIGH: +{deviation:.2f} kPa ({deviation_pct:.1f}%)"
            else:
                # Severe deviation
                scores[CondenserState.CRITICAL.value] += 0.35
                scores[CondenserState.AIR_LEAK.value] += 0.20
                contributing_factors.append(f"Vacuum severe deviation: +{deviation:.2f} kPa")
                evidence = f"Vacuum deviation CRITICAL: +{deviation:.2f} kPa ({deviation_pct:.1f}%)"

            confidence = 0.80

        # Normalize scores
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}

        return IndicatorScore(
            category_name="vacuum",
            state_scores=scores,
            available=available,
            confidence=confidence,
            primary_evidence=evidence,
            contributing_factors=contributing_factors
        )

    def _score_air_removal_indicators(self, inp: CondenserInput) -> IndicatorScore:
        """
        Score condenser state based on air removal indicators.

        Uses air leak rate and ejector performance to detect air ingress issues.
        """
        states = [s.value for s in CondenserState if s not in [CondenserState.UNKNOWN, CondenserState.STARTUP, CondenserState.SHUTDOWN]]
        scores = {s: 0.0 for s in states}
        contributing_factors = []
        available = False
        confidence = 0.0
        evidence = "No air removal data available"

        # Determine condenser size category
        if inp.tube_count < 5000:
            size = "small"
        elif inp.tube_count < 15000:
            size = "medium"
        else:
            size = "large"

        air_thresh = AIR_LEAK_THRESHOLDS.get(size, AIR_LEAK_THRESHOLDS["medium"])

        # Score based on air leak rate
        if inp.air_leak_rate_kg_hr is not None:
            available = True
            air_rate = inp.air_leak_rate_kg_hr

            if air_rate <= air_thresh["normal"]:
                scores[CondenserState.NORMAL.value] += 0.45
                contributing_factors.append(f"Air leak normal: {air_rate:.1f} kg/hr")
                evidence = f"Air leak rate: {air_rate:.1f} kg/hr (normal)"
            elif air_rate <= air_thresh["elevated"]:
                scores[CondenserState.NORMAL.value] += 0.20
                scores[CondenserState.AIR_LEAK.value] += 0.20
                scores[CondenserState.DEGRADED.value] += 0.10
                contributing_factors.append(f"Air leak elevated: {air_rate:.1f} kg/hr")
                evidence = f"Air leak rate: {air_rate:.1f} kg/hr (elevated)"
            elif air_rate <= air_thresh["high"]:
                scores[CondenserState.AIR_LEAK.value] += 0.40
                scores[CondenserState.DEGRADED.value] += 0.15
                contributing_factors.append(f"Air leak high: {air_rate:.1f} kg/hr")
                evidence = f"Air leak rate: {air_rate:.1f} kg/hr (HIGH)"
            else:
                scores[CondenserState.AIR_LEAK.value] += 0.45
                scores[CondenserState.CRITICAL.value] += 0.20
                contributing_factors.append(f"Air leak critical: {air_rate:.1f} kg/hr")
                evidence = f"Air leak rate: {air_rate:.1f} kg/hr (CRITICAL)"

            confidence += 0.45

        # Score based on ejector capacity
        if inp.ejector_capacity_pct is not None:
            available = True
            ejector = inp.ejector_capacity_pct

            if ejector <= 60:
                scores[CondenserState.NORMAL.value] += 0.20
                contributing_factors.append(f"Ejector capacity normal: {ejector:.0f}%")
            elif ejector <= 80:
                scores[CondenserState.NORMAL.value] += 0.10
                scores[CondenserState.DEGRADED.value] += 0.10
                contributing_factors.append(f"Ejector capacity elevated: {ejector:.0f}%")
            elif ejector <= 95:
                scores[CondenserState.AIR_LEAK.value] += 0.20
                scores[CondenserState.DEGRADED.value] += 0.10
                contributing_factors.append(f"Ejector capacity high: {ejector:.0f}%")
            else:
                scores[CondenserState.AIR_LEAK.value] += 0.25
                scores[CondenserState.CRITICAL.value] += 0.15
                contributing_factors.append(f"Ejector at/over capacity: {ejector:.0f}%")

            confidence += 0.30

        # Normalize scores
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}

        return IndicatorScore(
            category_name="air_removal",
            state_scores=scores,
            available=available,
            confidence=min(0.85, confidence),
            primary_evidence=evidence,
            contributing_factors=contributing_factors
        )

    def _fuse_indicator_scores(
        self,
        thermal: IndicatorScore,
        vacuum: IndicatorScore,
        air_removal: IndicatorScore
    ) -> Dict[str, float]:
        """
        Fuse indicator scores using weighted late fusion.

        Formula: P(state) = sum(weight_i * confidence_i * score_i) / sum(weight_i * confidence_i)
        """
        states = [s.value for s in CondenserState if s not in [CondenserState.UNKNOWN, CondenserState.STARTUP, CondenserState.SHUTDOWN]]
        fused = {s: 0.0 for s in states}

        indicators = [
            (thermal, float(self.config.thermal_weight)),
            (vacuum, float(self.config.vacuum_weight)),
            (air_removal, float(self.config.air_removal_weight))
        ]

        total_weight = 0.0
        for indicator, base_weight in indicators:
            if indicator.available:
                effective_weight = base_weight * indicator.confidence
                total_weight += effective_weight

                for state in states:
                    score = indicator.state_scores.get(state, 0.0)
                    fused[state] += score * effective_weight

        # Normalize
        if total_weight > 0:
            fused = {k: v / total_weight for k, v in fused.items()}
        else:
            fused[CondenserState.UNKNOWN.value] = 1.0

        return {k: round(v, 4) for k, v in fused.items()}

    def _get_confidence_level(self, score: float) -> ConfidenceLevel:
        """Convert numeric confidence to qualitative level."""
        if score >= 0.90:
            return ConfidenceLevel.HIGH
        elif score >= 0.70:
            return ConfidenceLevel.MEDIUM
        elif score >= 0.50:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW

    def _calculate_uncertainty_bounds(
        self,
        state_probs: Dict[str, float],
        predicted: CondenserState,
        indicator_scores: List[IndicatorScore]
    ) -> Tuple[float, float]:
        """Calculate 5th/95th percentile uncertainty bounds."""
        prob = state_probs.get(predicted.value, 0.0)

        # Calculate entropy for uncertainty
        entropy = -sum(
            p * math.log2(p + 1e-10)
            for p in state_probs.values() if p > 0
        )
        max_entropy = math.log2(len(state_probs)) if len(state_probs) > 0 else 1.0
        norm_entropy = entropy / max_entropy if max_entropy > 0 else 1.0

        # Count available indicators
        available_count = sum(1 for s in indicator_scores if s.available)
        indicator_factor = available_count / 3.0

        # Calculate bound width
        uncertainty = (1.0 - prob) * 0.5 + norm_entropy * 0.3 + (1 - indicator_factor) * 0.2
        bound_width = 0.05 + uncertainty * 0.25

        p_low = max(0.0, prob - bound_width)
        p_high = min(1.0, prob + bound_width)

        return (round(p_low, 4), round(p_high, 4))

    def _determine_severity(
        self,
        state: CondenserState,
        confidence: float,
        inp: CondenserInput
    ) -> SeverityLevel:
        """Determine maintenance severity level."""
        if state == CondenserState.NORMAL:
            return SeverityLevel.NONE

        if state == CondenserState.CRITICAL:
            return SeverityLevel.CRITICAL

        if state == CondenserState.AIR_LEAK:
            if inp.air_leak_rate_kg_hr and inp.air_leak_rate_kg_hr > 60:
                return SeverityLevel.CRITICAL
            if confidence > 0.8:
                return SeverityLevel.HIGH
            return SeverityLevel.MEDIUM

        if state == CondenserState.FOULED:
            if inp.cleanliness_factor and inp.cleanliness_factor < 0.65:
                return SeverityLevel.HIGH
            if confidence > 0.8:
                return SeverityLevel.MEDIUM
            return SeverityLevel.LOW

        if state == CondenserState.DEGRADED:
            if confidence > 0.85:
                return SeverityLevel.MEDIUM
            return SeverityLevel.LOW

        if state == CondenserState.SUBCOOLED:
            return SeverityLevel.LOW

        return SeverityLevel.LOW

    def _calculate_feature_importance(
        self,
        inp: CondenserInput,
        thermal: IndicatorScore,
        vacuum: IndicatorScore,
        air_removal: IndicatorScore
    ) -> List[FeatureImportance]:
        """Calculate SHAP-compatible feature importance scores."""
        features = []

        # Cleanliness Factor importance
        if inp.cleanliness_factor is not None:
            cf = inp.cleanliness_factor
            baseline = 0.90  # Expected baseline
            diff = baseline - cf  # Positive = degradation
            importance = min(1.0, abs(diff) / 0.30) * 0.35

            features.append(FeatureImportance(
                feature_name="cleanliness_factor",
                importance_score=importance,
                direction="toward_degradation" if diff > 0.05 else "toward_normal",
                value=cf,
                baseline=baseline,
                threshold=self.config.cf_normal_threshold
            ))

        # TTD importance
        if inp.ttd is not None:
            ttd = inp.ttd
            baseline = 3.0  # Normal baseline
            diff = ttd - baseline  # Positive = degradation
            importance = min(1.0, abs(diff) / 5.0) * 0.30

            features.append(FeatureImportance(
                feature_name="terminal_temperature_difference",
                importance_score=importance,
                direction="toward_degradation" if diff > 1.0 else "toward_normal",
                value=ttd,
                baseline=baseline,
                threshold=self.config.ttd_normal_max
            ))

        # Vacuum deviation importance
        if inp.vacuum_pressure_kpa is not None and inp.design_vacuum_kpa is not None:
            deviation = inp.vacuum_pressure_kpa - inp.design_vacuum_kpa
            baseline = 0.0
            importance = min(1.0, abs(deviation) / 3.0) * 0.25

            features.append(FeatureImportance(
                feature_name="vacuum_deviation_kpa",
                importance_score=importance,
                direction="toward_degradation" if deviation > 0.5 else "toward_normal",
                value=deviation,
                baseline=baseline,
                threshold=self.config.vacuum_deviation_max
            ))

        # Air leak rate importance
        if inp.air_leak_rate_kg_hr is not None:
            air = inp.air_leak_rate_kg_hr
            baseline = 10.0
            diff = air - baseline
            importance = min(1.0, max(0, diff) / 30.0) * 0.20

            features.append(FeatureImportance(
                feature_name="air_leak_rate_kg_hr",
                importance_score=importance,
                direction="toward_degradation" if diff > 5 else "toward_normal",
                value=air,
                baseline=baseline,
                threshold=self.config.air_leak_rate_max
            ))

        # Sort by importance
        features.sort(key=lambda x: x.importance_score, reverse=True)

        return features

    def _calculate_performance_impact(
        self,
        inp: CondenserInput,
        state: CondenserState,
        state_probs: Dict[str, float]
    ) -> float:
        """Calculate estimated performance impact percentage."""
        impact = 0.0

        # Impact from CF degradation
        if inp.cleanliness_factor is not None:
            cf_loss = max(0, 0.95 - inp.cleanliness_factor)
            impact += cf_loss * 5.0  # ~5% impact per 0.10 CF drop

        # Impact from vacuum deviation
        if inp.vacuum_pressure_kpa and inp.design_vacuum_kpa:
            deviation = inp.vacuum_pressure_kpa - inp.design_vacuum_kpa
            if deviation > 0:
                impact += deviation * 1.0  # ~1% per kPa deviation

        # Impact from TTD
        if inp.ttd is not None:
            if inp.ttd > 5.0:
                impact += (inp.ttd - 5.0) * 0.3  # ~0.3% per degree over threshold

        # State-based adjustment
        state_multipliers = {
            CondenserState.NORMAL.value: 0.0,
            CondenserState.FOULED.value: 1.2,
            CondenserState.AIR_LEAK.value: 1.3,
            CondenserState.DEGRADED.value: 1.1,
            CondenserState.CRITICAL.value: 1.5,
            CondenserState.SUBCOOLED.value: 0.5,
        }

        multiplier = state_multipliers.get(state.value, 1.0)
        impact *= multiplier

        return min(15.0, round(impact, 2))  # Cap at 15%

    def _generate_recommendations(
        self,
        state: CondenserState,
        severity: SeverityLevel,
        inp: CondenserInput,
        indicator_scores: List[IndicatorScore]
    ) -> List[str]:
        """Generate maintenance recommendations based on classification."""
        recommendations = []

        if state == CondenserState.NORMAL:
            recommendations.append("Continue routine monitoring per schedule")
            if inp.last_cleaning_days > 180:
                recommendations.append(
                    f"Consider preventive tube cleaning (last: {inp.last_cleaning_days} days ago)"
                )

        elif state == CondenserState.FOULED:
            recommendations.append("Schedule condenser tube cleaning")
            if inp.cleanliness_factor and inp.cleanliness_factor < 0.75:
                recommendations.append("URGENT: CF below 0.75 - prioritize cleaning within 7 days")
            recommendations.append("Inspect tube sheet for macro-fouling")
            recommendations.append("Review cooling water treatment program")

        elif state == CondenserState.AIR_LEAK:
            recommendations.append("Perform air in-leakage test to locate leak source")
            recommendations.append("Check expansion joint integrity")
            recommendations.append("Inspect vacuum breaker valves")
            recommendations.append("Verify steam ejector performance")
            if severity == SeverityLevel.CRITICAL:
                recommendations.append("IMMEDIATE: Consider load reduction until leak isolated")

        elif state == CondenserState.DEGRADED:
            recommendations.append("Schedule comprehensive inspection")
            recommendations.append("Review all indicator trends for root cause")
            recommendations.append("Check for multiple concurrent issues (fouling + air)")

        elif state == CondenserState.CRITICAL:
            recommendations.append("IMMEDIATE ACTION REQUIRED")
            recommendations.append("Consider load reduction to protect equipment")
            recommendations.append("Notify plant engineering and operations")
            recommendations.append("Prepare for emergency maintenance")

        elif state == CondenserState.SUBCOOLED:
            recommendations.append("Reduce cooling water flow if possible")
            recommendations.append("Review hotwell level control")
            recommendations.append("Check for tube bundle bypassing")

        return recommendations

    def _generate_evidence_summary(
        self,
        state: CondenserState,
        indicator_scores: List[IndicatorScore],
        confidence: float,
        performance_impact: float
    ) -> str:
        """Generate plain-language evidence summary."""
        parts = [
            f"Classification: {state.value.replace('_', ' ').title()} "
            f"({confidence * 100:.0f}% confidence)"
        ]

        for indicator in indicator_scores:
            if indicator.available:
                parts.append(f"{indicator.category_name.title()}: {indicator.primary_evidence}")

        if performance_impact > 0:
            parts.append(f"Est. performance impact: {performance_impact:.1f}%")

        return "; ".join(parts)

    def _calculate_provenance_hash(
        self,
        inp: CondenserInput,
        state_probs: Dict[str, float]
    ) -> str:
        """Calculate SHA-256 provenance hash for audit trail."""
        data = {
            "condenser_id": inp.condenser_id,
            "timestamp": inp.timestamp.isoformat(),
            "cf": inp.cleanliness_factor,
            "ttd": inp.ttd,
            "vacuum_kpa": inp.vacuum_pressure_kpa,
            "air_leak": inp.air_leak_rate_kg_hr,
            "probabilities": state_probs
        }
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()

    def _create_unknown_result(
        self,
        inp: CondenserInput,
        timestamp: datetime,
        indicator_scores: List[IndicatorScore]
    ) -> ClassificationResult:
        """Create result for unknown/insufficient data case."""
        state_probs = {s.value: 0.0 for s in CondenserState}
        state_probs[CondenserState.UNKNOWN.value] = 1.0

        provenance_hash = self._calculate_provenance_hash(inp, state_probs)

        return ClassificationResult(
            condenser_id=inp.condenser_id,
            timestamp=timestamp,
            state=CondenserState.UNKNOWN,
            confidence_score=0.0,
            confidence_level=ConfidenceLevel.VERY_LOW,
            severity=SeverityLevel.LOW,
            uncertainty_bounds=(0.0, 1.0),
            state_probabilities=state_probs,
            indicator_scores=indicator_scores,
            feature_importance=[],
            recommended_actions=[
                "Insufficient data for classification",
                "Verify sensor connectivity and data quality",
                "Manual inspection recommended"
            ],
            evidence_summary="Classification: Unknown - Insufficient valid indicators",
            performance_impact_pct=0.0,
            provenance_hash=provenance_hash
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get classifier statistics."""
        with self._lock:
            return {
                "classification_count": self._classification_count,
                "indicator_weights": {
                    "thermal": float(self.config.thermal_weight),
                    "vacuum": float(self.config.vacuum_weight),
                    "air_removal": float(self.config.air_removal_weight)
                },
                "supported_states": [s.value for s in CondenserState],
                "confidence_threshold": float(self.config.confidence_threshold),
                "thresholds": {
                    "cf_normal": self.config.cf_normal_threshold,
                    "ttd_normal_max": self.config.ttd_normal_max,
                    "vacuum_deviation_max": self.config.vacuum_deviation_max,
                    "air_leak_rate_max": self.config.air_leak_rate_max
                }
            }


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "CondenserStateClassifier",
    "ClassificationConfig",
    "ClassificationResult",
    "CondenserInput",
    "IndicatorScore",
    "FeatureImportance",
    "CondenserState",
    "ConfidenceLevel",
    "SeverityLevel",
    "IndicatorCategory",
    "ModalityWeight",
    "CF_THRESHOLDS",
    "TTD_THRESHOLDS",
    "AIR_LEAK_THRESHOLDS",
]
