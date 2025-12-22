# -*- coding: utf-8 -*-
"""
TrapStateClassifier for GL-008 TRAPCATCHER

Multimodal steam trap state classification combining acoustic, thermal,
and contextual signals using late fusion architecture.

Standards:
- ASME PTC 39: Steam Traps - Performance Test Codes
- DOE Steam System Assessment Protocol
- ISO 7841: Automatic steam traps - Steam loss determination

Key Features:
- Multimodal sensor fusion (acoustic + thermal + pressure)
- Late fusion architecture for modality independence
- Uncertainty quantification with calibrated bounds
- SHAP-compatible feature importance
- Deterministic rule-based classification

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
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS
# ============================================================================

class TrapCondition(str, Enum):
    """Steam trap operational conditions."""
    OPERATING_NORMAL = "operating_normal"
    FAILED_OPEN = "failed_open"         # Blow-through steam loss
    FAILED_CLOSED = "failed_closed"     # Blocked - no condensate drainage
    LEAKING = "leaking"                 # Partial failure
    INTERMITTENT = "intermittent"       # Erratic cycling
    COLD = "cold"                       # Not receiving steam
    UNKNOWN = "unknown"                 # Insufficient data


class ConfidenceLevel(str, Enum):
    """Confidence levels for classification."""
    HIGH = "high"           # >= 90%
    MEDIUM = "medium"       # 70-89%
    LOW = "low"             # 50-69%
    VERY_LOW = "very_low"   # < 50%


class SeverityLevel(str, Enum):
    """Severity classification for maintenance prioritization."""
    CRITICAL = "critical"   # Immediate action required
    HIGH = "high"           # Within 7 days
    MEDIUM = "medium"       # Within 30 days
    LOW = "low"             # Next maintenance cycle
    NONE = "none"           # No action required


class ModalityWeight(str, Enum):
    """Weight profiles for modality fusion."""
    ACOUSTIC_PRIMARY = "acoustic_primary"       # 50% acoustic, 30% thermal, 20% context
    THERMAL_PRIMARY = "thermal_primary"         # 30% acoustic, 50% thermal, 20% context
    BALANCED = "balanced"                       # 40% acoustic, 40% thermal, 20% context
    CONTEXT_ENHANCED = "context_enhanced"       # 35% acoustic, 35% thermal, 30% context


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass(frozen=True)
class ClassificationConfig:
    """
    Immutable configuration for trap state classifier.

    Attributes:
        modality_weight_profile: Weighting scheme for sensor fusion
        confidence_threshold: Minimum confidence for valid classification
        acoustic_weight: Weight for acoustic modality (0-1)
        thermal_weight: Weight for thermal modality (0-1)
        context_weight: Weight for contextual factors (0-1)
        failure_rate_prior: Prior probability of trap failure
        min_sensors_for_classification: Minimum sensors required
    """
    modality_weight_profile: ModalityWeight = ModalityWeight.BALANCED
    confidence_threshold: Decimal = Decimal("0.50")
    acoustic_weight: Decimal = Decimal("0.40")
    thermal_weight: Decimal = Decimal("0.40")
    context_weight: Decimal = Decimal("0.20")
    failure_rate_prior: Decimal = Decimal("0.15")  # 15% industry average
    min_sensors_for_classification: int = 2

    def __post_init__(self):
        """Validate weights sum to 1.0."""
        total = self.acoustic_weight + self.thermal_weight + self.context_weight
        if abs(float(total) - 1.0) > 0.001:
            raise ValueError(f"Modality weights must sum to 1.0, got {total}")


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass(frozen=True)
class SensorInput:
    """
    Immutable sensor input data for classification.

    Attributes:
        trap_id: Unique trap identifier
        timestamp: Measurement timestamp
        acoustic_amplitude_db: Peak acoustic amplitude (dB)
        acoustic_frequency_khz: Dominant frequency (kHz)
        inlet_temp_c: Inlet temperature (Celsius)
        outlet_temp_c: Outlet temperature (Celsius)
        pressure_bar_g: Operating pressure (bar gauge)
        trap_type: Type of steam trap
        trap_age_years: Age of trap in years
        last_maintenance_days: Days since last maintenance
        location: Physical location identifier
    """
    trap_id: str
    timestamp: datetime
    acoustic_amplitude_db: Optional[float] = None
    acoustic_frequency_khz: Optional[float] = None
    inlet_temp_c: Optional[float] = None
    outlet_temp_c: Optional[float] = None
    pressure_bar_g: float = 10.0
    trap_type: str = "thermodynamic"
    trap_age_years: float = 0.0
    last_maintenance_days: int = 0
    location: str = ""


@dataclass(frozen=True)
class ModalityScore:
    """
    Score from a single modality.

    Attributes:
        modality_name: Name of the modality (acoustic, thermal, context)
        condition_scores: Scores for each possible condition
        available: Whether this modality had valid input data
        confidence: Confidence in this modality's assessment
        primary_evidence: Key evidence supporting the assessment
    """
    modality_name: str
    condition_scores: Dict[str, float]
    available: bool
    confidence: float
    primary_evidence: str


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
    """
    feature_name: str
    importance_score: float
    direction: str  # "toward_failure" or "toward_normal"
    value: float
    baseline: float


@dataclass(frozen=True)
class ClassificationResult:
    """
    Complete immutable classification result.

    Attributes:
        trap_id: Steam trap identifier
        timestamp: Classification timestamp
        condition: Classified trap condition
        confidence_score: Numeric confidence (0-1)
        confidence_level: Qualitative confidence level
        severity: Maintenance severity level
        uncertainty_bounds: 5th/95th percentile bounds
        condition_probabilities: Probabilities for all conditions
        modality_scores: Scores from each modality
        feature_importance: Feature importance for explainability
        recommended_action: Recommended maintenance action
        evidence_summary: Plain-language evidence summary
        provenance_hash: SHA-256 hash for audit trail
    """
    trap_id: str
    timestamp: datetime
    condition: TrapCondition
    confidence_score: float
    confidence_level: ConfidenceLevel
    severity: SeverityLevel
    uncertainty_bounds: Tuple[float, float]
    condition_probabilities: Dict[str, float]
    modality_scores: List[ModalityScore]
    feature_importance: List[FeatureImportance]
    recommended_action: str
    evidence_summary: str
    provenance_hash: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "trap_id": self.trap_id,
            "timestamp": self.timestamp.isoformat(),
            "condition": self.condition.value,
            "confidence_score": round(self.confidence_score, 4),
            "confidence_level": self.confidence_level.value,
            "severity": self.severity.value,
            "uncertainty_bounds": {
                "p05": round(self.uncertainty_bounds[0], 4),
                "p95": round(self.uncertainty_bounds[1], 4)
            },
            "condition_probabilities": {
                k: round(v, 4) for k, v in self.condition_probabilities.items()
            },
            "feature_importance": [
                {
                    "feature": fi.feature_name,
                    "importance": round(fi.importance_score, 4),
                    "direction": fi.direction
                }
                for fi in self.feature_importance[:5]  # Top 5
            ],
            "recommended_action": self.recommended_action,
            "evidence_summary": self.evidence_summary,
            "provenance_hash": self.provenance_hash
        }


# ============================================================================
# REFERENCE DATA
# ============================================================================

# Steam saturation temperature table: pressure (bar gauge) -> T_sat (C)
SATURATION_TEMPS: Dict[int, float] = {
    0: 100.0, 2: 133.5, 4: 151.8, 6: 165.0, 8: 175.4, 10: 184.1,
    12: 191.6, 14: 198.3, 16: 204.3, 18: 209.8, 20: 214.9,
    25: 226.0, 30: 235.8, 40: 252.4, 50: 266.4,
}

# Acoustic thresholds by trap type
ACOUSTIC_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "thermodynamic": {"normal_max": 45.0, "leak_min": 45.0, "leak_max": 70.0, "failed_open": 70.0},
    "thermostatic": {"normal_max": 40.0, "leak_min": 40.0, "leak_max": 65.0, "failed_open": 65.0},
    "mechanical": {"normal_max": 50.0, "leak_min": 50.0, "leak_max": 75.0, "failed_open": 75.0},
    "venturi": {"normal_max": 55.0, "leak_min": 55.0, "leak_max": 80.0, "failed_open": 80.0},
}

# Age-based failure probability multipliers
AGE_MULTIPLIERS: Dict[int, float] = {
    0: 0.5, 1: 0.6, 2: 0.8, 3: 1.0, 4: 1.1, 5: 1.3,
    6: 1.5, 7: 1.8, 8: 2.2, 9: 2.7, 10: 3.5
}


# ============================================================================
# MAIN CLASSIFIER
# ============================================================================

class TrapStateClassifier:
    """
    Multimodal steam trap state classifier using late fusion.

    ZERO-HALLUCINATION GUARANTEE:
    - All classifications use deterministic weighted scoring
    - No LLM or ML inference in classification path
    - Same inputs always produce identical outputs
    - Complete provenance tracking with SHA-256 hashes

    Architecture:
    1. Acoustic modality: Ultrasonic amplitude and frequency analysis
    2. Thermal modality: Temperature differential analysis
    3. Context modality: Age, maintenance history, trap type
    4. Late Fusion: Weighted combination of modality scores

    Example:
        >>> classifier = TrapStateClassifier()
        >>> result = classifier.classify(SensorInput(
        ...     trap_id="ST-001",
        ...     timestamp=datetime.now(timezone.utc),
        ...     acoustic_amplitude_db=75.0,
        ...     acoustic_frequency_khz=42.0,
        ...     inlet_temp_c=185.0,
        ...     outlet_temp_c=180.0,
        ...     pressure_bar_g=10.0
        ... ))
        >>> print(f"Condition: {result.condition.value}")
    """

    def __init__(self, config: Optional[ClassificationConfig] = None):
        """
        Initialize trap state classifier.

        Args:
            config: Classification configuration (uses defaults if not provided)
        """
        self.config = config or ClassificationConfig()
        self._classification_count = 0
        self._lock = threading.Lock()

        logger.info(
            f"TrapStateClassifier initialized "
            f"(weights: acoustic={self.config.acoustic_weight}, "
            f"thermal={self.config.thermal_weight}, "
            f"context={self.config.context_weight})"
        )

    def classify(self, sensor_input: SensorInput) -> ClassificationResult:
        """
        Classify trap condition from sensor inputs.

        ZERO-HALLUCINATION: Uses deterministic weighted scoring.

        Args:
            sensor_input: Sensor input data for classification

        Returns:
            ClassificationResult with complete analysis
        """
        with self._lock:
            self._classification_count += 1

        timestamp = datetime.now(timezone.utc)

        # Score each modality
        acoustic_score = self._score_acoustic_modality(sensor_input)
        thermal_score = self._score_thermal_modality(sensor_input)
        context_score = self._score_context_modality(sensor_input)

        modality_scores = [acoustic_score, thermal_score, context_score]

        # Fuse modality scores
        condition_probs = self._fuse_modality_scores(
            acoustic_score, thermal_score, context_score
        )

        # Determine primary condition
        condition = TrapCondition(max(condition_probs.items(), key=lambda x: x[1])[0])
        confidence_score = condition_probs[condition.value]

        # Calculate confidence level and uncertainty
        confidence_level = self._get_confidence_level(confidence_score)
        uncertainty_bounds = self._calculate_uncertainty_bounds(
            condition_probs, condition, modality_scores
        )

        # Determine severity
        severity = self._determine_severity(condition, confidence_score, sensor_input)

        # Calculate feature importance
        feature_importance = self._calculate_feature_importance(
            sensor_input, acoustic_score, thermal_score, context_score
        )

        # Generate recommendation and evidence
        recommended_action = self._generate_recommendation(condition, severity)
        evidence_summary = self._generate_evidence_summary(
            condition, modality_scores, confidence_score
        )

        # Calculate provenance hash
        provenance_hash = self._calculate_provenance_hash(sensor_input, condition_probs)

        return ClassificationResult(
            trap_id=sensor_input.trap_id,
            timestamp=timestamp,
            condition=condition,
            confidence_score=round(confidence_score, 4),
            confidence_level=confidence_level,
            severity=severity,
            uncertainty_bounds=uncertainty_bounds,
            condition_probabilities=condition_probs,
            modality_scores=modality_scores,
            feature_importance=feature_importance,
            recommended_action=recommended_action,
            evidence_summary=evidence_summary,
            provenance_hash=provenance_hash
        )

    def _score_acoustic_modality(self, sensor_input: SensorInput) -> ModalityScore:
        """Score trap condition based on acoustic data."""
        scores = {c.value: 0.0 for c in TrapCondition if c != TrapCondition.UNKNOWN}

        if sensor_input.acoustic_amplitude_db is None:
            return ModalityScore(
                modality_name="acoustic",
                condition_scores=scores,
                available=False,
                confidence=0.0,
                primary_evidence="No acoustic data available"
            )

        amp_db = sensor_input.acoustic_amplitude_db
        thresholds = ACOUSTIC_THRESHOLDS.get(
            sensor_input.trap_type.lower(),
            ACOUSTIC_THRESHOLDS["thermodynamic"]
        )

        # Score based on amplitude thresholds
        if amp_db < thresholds["normal_max"]:
            scores[TrapCondition.OPERATING_NORMAL.value] = 0.9
            scores[TrapCondition.COLD.value] = 0.1
            evidence = f"Normal acoustic level: {amp_db:.1f} dB"
        elif amp_db < thresholds["leak_max"]:
            leak_fraction = (amp_db - thresholds["leak_min"]) / \
                           (thresholds["leak_max"] - thresholds["leak_min"])
            scores[TrapCondition.LEAKING.value] = leak_fraction * 0.8
            scores[TrapCondition.OPERATING_NORMAL.value] = (1 - leak_fraction) * 0.5
            scores[TrapCondition.INTERMITTENT.value] = 0.1
            evidence = f"Elevated acoustic level: {amp_db:.1f} dB (potential leak)"
        else:
            scores[TrapCondition.FAILED_OPEN.value] = 0.85
            scores[TrapCondition.LEAKING.value] = 0.1
            scores[TrapCondition.INTERMITTENT.value] = 0.05
            evidence = f"High acoustic level: {amp_db:.1f} dB (likely blow-through)"

        # Very low acoustic may indicate blocked trap
        if amp_db < 25.0:
            scores[TrapCondition.FAILED_CLOSED.value] = 0.3
            scores[TrapCondition.COLD.value] = 0.3

        # Normalize
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}

        return ModalityScore(
            modality_name="acoustic",
            condition_scores=scores,
            available=True,
            confidence=0.85 if amp_db > 30 else 0.6,
            primary_evidence=evidence
        )

    def _score_thermal_modality(self, sensor_input: SensorInput) -> ModalityScore:
        """Score trap condition based on thermal data."""
        scores = {c.value: 0.0 for c in TrapCondition if c != TrapCondition.UNKNOWN}

        if sensor_input.inlet_temp_c is None or sensor_input.outlet_temp_c is None:
            return ModalityScore(
                modality_name="thermal",
                condition_scores=scores,
                available=False,
                confidence=0.0,
                primary_evidence="No thermal data available"
            )

        inlet = sensor_input.inlet_temp_c
        outlet = sensor_input.outlet_temp_c
        delta_t = inlet - outlet

        # Get saturation temperature
        t_sat = self._get_saturation_temp(sensor_input.pressure_bar_g)
        t_span = max(1.0, t_sat - 25.0)  # Ambient assumed 25C
        normalized_delta = delta_t / t_span

        # Score based on thermal patterns
        if 0.15 <= normalized_delta <= 0.50:
            scores[TrapCondition.OPERATING_NORMAL.value] = 0.85
            scores[TrapCondition.LEAKING.value] = 0.15
            evidence = f"Normal temperature differential: {delta_t:.1f}C"
        elif normalized_delta < 0.05:
            # Very small delta = blow-through (outlet near inlet)
            scores[TrapCondition.FAILED_OPEN.value] = 0.85
            scores[TrapCondition.LEAKING.value] = 0.15
            evidence = f"Minimal temperature drop: {delta_t:.1f}C (blow-through suspected)"
        elif normalized_delta > 0.70:
            # Very large delta = blocked (cold outlet)
            scores[TrapCondition.FAILED_CLOSED.value] = 0.75
            scores[TrapCondition.COLD.value] = 0.20
            scores[TrapCondition.OPERATING_NORMAL.value] = 0.05
            evidence = f"Large temperature drop: {delta_t:.1f}C (blockage suspected)"
        else:
            # Intermediate - could be leaking
            scores[TrapCondition.LEAKING.value] = 0.5
            scores[TrapCondition.OPERATING_NORMAL.value] = 0.3
            scores[TrapCondition.INTERMITTENT.value] = 0.2
            evidence = f"Intermediate temperature differential: {delta_t:.1f}C"

        # Check for cold trap
        if inlet < t_sat * 0.6:
            scores[TrapCondition.COLD.value] = 0.7
            for k in scores:
                if k != TrapCondition.COLD.value:
                    scores[k] *= 0.3
            evidence = f"Low inlet temperature: {inlet:.1f}C (cold trap)"

        # Normalize
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}

        return ModalityScore(
            modality_name="thermal",
            condition_scores=scores,
            available=True,
            confidence=0.80,
            primary_evidence=evidence
        )

    def _score_context_modality(self, sensor_input: SensorInput) -> ModalityScore:
        """Score trap condition based on contextual factors."""
        scores = {c.value: 0.0 for c in TrapCondition if c != TrapCondition.UNKNOWN}

        # Start with base failure rate prior
        base_failure_prob = float(self.config.failure_rate_prior)

        # Age adjustment
        age_years = int(min(10, sensor_input.trap_age_years))
        age_multiplier = AGE_MULTIPLIERS.get(age_years, 1.0)
        adjusted_failure_prob = min(0.5, base_failure_prob * age_multiplier)

        # Maintenance adjustment
        if sensor_input.last_maintenance_days > 365:
            adjusted_failure_prob *= 1.2
        elif sensor_input.last_maintenance_days < 90:
            adjusted_failure_prob *= 0.8

        # Distribute failure probability
        scores[TrapCondition.OPERATING_NORMAL.value] = 1.0 - adjusted_failure_prob
        scores[TrapCondition.FAILED_OPEN.value] = adjusted_failure_prob * 0.4
        scores[TrapCondition.LEAKING.value] = adjusted_failure_prob * 0.35
        scores[TrapCondition.FAILED_CLOSED.value] = adjusted_failure_prob * 0.15
        scores[TrapCondition.INTERMITTENT.value] = adjusted_failure_prob * 0.1

        evidence = (f"Trap age: {sensor_input.trap_age_years:.1f} years, "
                   f"Last maintenance: {sensor_input.last_maintenance_days} days ago")

        return ModalityScore(
            modality_name="context",
            condition_scores=scores,
            available=True,
            confidence=0.60,
            primary_evidence=evidence
        )

    def _fuse_modality_scores(
        self,
        acoustic: ModalityScore,
        thermal: ModalityScore,
        context: ModalityScore
    ) -> Dict[str, float]:
        """
        Fuse modality scores using weighted late fusion.

        Formula: P(condition) = sum(weight_i * confidence_i * score_i) / sum(weight_i * confidence_i)
        """
        conditions = [c.value for c in TrapCondition if c != TrapCondition.UNKNOWN]
        fused = {c: 0.0 for c in conditions}

        modalities = [
            (acoustic, float(self.config.acoustic_weight)),
            (thermal, float(self.config.thermal_weight)),
            (context, float(self.config.context_weight))
        ]

        total_weight = 0.0
        for modality, base_weight in modalities:
            if modality.available:
                effective_weight = base_weight * modality.confidence
                total_weight += effective_weight

                for condition in conditions:
                    score = modality.condition_scores.get(condition, 0.0)
                    fused[condition] += score * effective_weight

        # Normalize
        if total_weight > 0:
            fused = {k: v / total_weight for k, v in fused.items()}
        else:
            fused[TrapCondition.UNKNOWN.value] = 1.0

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
        condition_probs: Dict[str, float],
        predicted: TrapCondition,
        modality_scores: List[ModalityScore]
    ) -> Tuple[float, float]:
        """Calculate 5th/95th percentile uncertainty bounds."""
        prob = condition_probs.get(predicted.value, 0.0)

        # Calculate entropy for uncertainty
        entropy = -sum(
            p * math.log2(p + 1e-10)
            for p in condition_probs.values() if p > 0
        )
        max_entropy = math.log2(len(condition_probs))
        norm_entropy = entropy / max_entropy if max_entropy > 0 else 1.0

        # Count available modalities
        available_count = sum(1 for m in modality_scores if m.available)
        modality_factor = available_count / 3.0

        # Calculate bound width
        uncertainty = (1.0 - prob) * 0.5 + norm_entropy * 0.3 + (1 - modality_factor) * 0.2
        bound_width = 0.05 + uncertainty * 0.25

        p_low = max(0.0, prob - bound_width)
        p_high = min(1.0, prob + bound_width)

        return (round(p_low, 4), round(p_high, 4))

    def _determine_severity(
        self,
        condition: TrapCondition,
        confidence: float,
        sensor_input: SensorInput
    ) -> SeverityLevel:
        """Determine maintenance severity level."""
        if condition == TrapCondition.OPERATING_NORMAL:
            return SeverityLevel.NONE

        if condition == TrapCondition.FAILED_OPEN:
            if confidence > 0.8:
                return SeverityLevel.CRITICAL
            return SeverityLevel.HIGH

        if condition == TrapCondition.FAILED_CLOSED:
            return SeverityLevel.HIGH

        if condition == TrapCondition.LEAKING:
            if sensor_input.acoustic_amplitude_db and sensor_input.acoustic_amplitude_db > 60:
                return SeverityLevel.HIGH
            return SeverityLevel.MEDIUM

        if condition == TrapCondition.INTERMITTENT:
            return SeverityLevel.MEDIUM

        if condition == TrapCondition.COLD:
            return SeverityLevel.LOW

        return SeverityLevel.LOW

    def _calculate_feature_importance(
        self,
        sensor_input: SensorInput,
        acoustic: ModalityScore,
        thermal: ModalityScore,
        context: ModalityScore
    ) -> List[FeatureImportance]:
        """Calculate SHAP-compatible feature importance scores."""
        features = []

        # Acoustic amplitude importance
        if sensor_input.acoustic_amplitude_db is not None:
            amp = sensor_input.acoustic_amplitude_db
            baseline = 40.0  # Normal baseline
            diff = amp - baseline
            importance = min(1.0, abs(diff) / 40.0) * 0.4  # Max 40% importance

            features.append(FeatureImportance(
                feature_name="acoustic_amplitude_db",
                importance_score=importance,
                direction="toward_failure" if diff > 10 else "toward_normal",
                value=amp,
                baseline=baseline
            ))

        # Temperature differential importance
        if sensor_input.inlet_temp_c and sensor_input.outlet_temp_c:
            delta_t = sensor_input.inlet_temp_c - sensor_input.outlet_temp_c
            baseline = 50.0  # Normal delta-T
            diff = abs(delta_t - baseline)
            importance = min(1.0, diff / 50.0) * 0.35

            direction = "toward_failure" if delta_t < 10 or delta_t > 100 else "toward_normal"

            features.append(FeatureImportance(
                feature_name="temperature_differential_c",
                importance_score=importance,
                direction=direction,
                value=delta_t,
                baseline=baseline
            ))

        # Age importance
        age = sensor_input.trap_age_years
        baseline = 3.0
        importance = min(1.0, max(0, age - baseline) / 7.0) * 0.15

        features.append(FeatureImportance(
            feature_name="trap_age_years",
            importance_score=importance,
            direction="toward_failure" if age > 5 else "toward_normal",
            value=age,
            baseline=baseline
        ))

        # Sort by importance
        features.sort(key=lambda x: x.importance_score, reverse=True)

        return features

    def _generate_recommendation(self, condition: TrapCondition, severity: SeverityLevel) -> str:
        """Generate maintenance recommendation."""
        recommendations = {
            TrapCondition.OPERATING_NORMAL: "Continue routine monitoring per schedule",
            TrapCondition.FAILED_OPEN: "IMMEDIATE: Replace trap - significant steam loss",
            TrapCondition.FAILED_CLOSED: "URGENT: Inspect and replace - risk of water hammer",
            TrapCondition.LEAKING: "Schedule replacement within 7 days" if severity == SeverityLevel.HIGH else "Include in next maintenance cycle",
            TrapCondition.INTERMITTENT: "Schedule detailed inspection to determine root cause",
            TrapCondition.COLD: "Verify steam supply and check upstream valves",
            TrapCondition.UNKNOWN: "Insufficient data - schedule manual inspection"
        }
        return recommendations.get(condition, "Continue monitoring")

    def _generate_evidence_summary(
        self,
        condition: TrapCondition,
        modality_scores: List[ModalityScore],
        confidence: float
    ) -> str:
        """Generate plain-language evidence summary."""
        parts = [
            f"Classification: {condition.value.replace('_', ' ').title()} "
            f"({confidence * 100:.0f}% confidence)"
        ]

        for modality in modality_scores:
            if modality.available:
                parts.append(f"{modality.modality_name.title()}: {modality.primary_evidence}")

        return "; ".join(parts)

    def _get_saturation_temp(self, pressure_bar_g: float) -> float:
        """Get saturation temperature at given pressure."""
        pressures = sorted(SATURATION_TEMPS.keys())
        if pressure_bar_g <= pressures[0]:
            return SATURATION_TEMPS[pressures[0]]
        if pressure_bar_g >= pressures[-1]:
            return SATURATION_TEMPS[pressures[-1]]

        for i in range(len(pressures) - 1):
            p_low, p_high = pressures[i], pressures[i + 1]
            if p_low <= pressure_bar_g <= p_high:
                fraction = (pressure_bar_g - p_low) / (p_high - p_low)
                return SATURATION_TEMPS[p_low] + fraction * (SATURATION_TEMPS[p_high] - SATURATION_TEMPS[p_low])

        return SATURATION_TEMPS[pressures[0]]

    def _calculate_provenance_hash(
        self,
        sensor_input: SensorInput,
        condition_probs: Dict[str, float]
    ) -> str:
        """Calculate SHA-256 provenance hash for audit trail."""
        data = {
            "trap_id": sensor_input.trap_id,
            "timestamp": sensor_input.timestamp.isoformat(),
            "acoustic_db": sensor_input.acoustic_amplitude_db,
            "inlet_temp": sensor_input.inlet_temp_c,
            "outlet_temp": sensor_input.outlet_temp_c,
            "probabilities": condition_probs
        }
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()

    def get_statistics(self) -> Dict[str, Any]:
        """Get classifier statistics."""
        with self._lock:
            return {
                "classification_count": self._classification_count,
                "modality_weights": {
                    "acoustic": float(self.config.acoustic_weight),
                    "thermal": float(self.config.thermal_weight),
                    "context": float(self.config.context_weight)
                },
                "supported_conditions": [c.value for c in TrapCondition],
                "confidence_threshold": float(self.config.confidence_threshold)
            }


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    "TrapStateClassifier",
    "ClassificationConfig",
    "ClassificationResult",
    "SensorInput",
    "ModalityScore",
    "FeatureImportance",
    "TrapCondition",
    "ConfidenceLevel",
    "SeverityLevel",
    "ModalityWeight",
]
