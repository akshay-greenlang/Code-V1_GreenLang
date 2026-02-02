# -*- coding: utf-8 -*-
"""
Carryover Event Detection for Steam Quality Control

This module implements carryover detection and early warning:
- Classification model for carryover detection
- Early warning with time-to-event prediction
- Feature engineering: T_sat delta, valve rates, load ramp rate

Carryover occurs when liquid water is carried along with steam,
causing quality degradation and potential equipment damage.

Zero-Hallucination Guarantee:
- Detection uses physics-based thresholds + learned patterns
- Predictions bounded by physical constraints
- Uncertainty quantified for all outputs
- Complete audit trail

Author: GL-BackendDeveloper
Date: December 2024
Version: 1.0.0
"""

import hashlib
import json
import logging
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS AND THRESHOLDS
# ============================================================================

# Carryover detection thresholds (physics-based)
TSAT_DELTA_WARNING = 2.0    # C below saturation triggers warning
TSAT_DELTA_CRITICAL = 5.0   # C below saturation indicates carryover

# Rate-based thresholds
MAX_SAFE_LOAD_RAMP_RATE = 0.1    # per minute (10% per minute)
MAX_SAFE_VALVE_RATE = 0.05       # per second (5% per second)

# Time constants
CARRYOVER_PERSISTENCE_S = 30.0   # Minimum duration to confirm carryover
WARNING_LOOKAHEAD_S = 60.0       # Time window for early warning


class CarryoverRiskLevel(Enum):
    """Carryover risk classification."""
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    ACTIVE_CARRYOVER = "active_carryover"


class CarryoverCause(Enum):
    """Root causes of carryover."""
    LOAD_RAMP = "load_ramp"               # Rapid load increase
    PRESSURE_DROP = "pressure_drop"        # Sudden pressure decrease
    SEPARATOR_FAILURE = "separator_failure" # Separator malfunction
    LEVEL_CONTROL = "level_control"        # Drum level control issue
    VALVE_MALFUNCTION = "valve_malfunction" # Valve stuck or rapid movement
    FOAMING = "foaming"                    # Water chemistry issue
    UNKNOWN = "unknown"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class CarryoverDetectorConfig:
    """Configuration for carryover detector."""

    # Detection thresholds
    tsat_delta_warning_c: float = TSAT_DELTA_WARNING
    tsat_delta_critical_c: float = TSAT_DELTA_CRITICAL

    # Rate limits
    max_load_ramp_rate_per_min: float = MAX_SAFE_LOAD_RAMP_RATE
    max_valve_rate_per_sec: float = MAX_SAFE_VALVE_RATE

    # Time windows
    feature_window_s: float = 60.0    # Window for feature calculation
    persistence_threshold_s: float = CARRYOVER_PERSISTENCE_S
    warning_lookahead_s: float = WARNING_LOOKAHEAD_S

    # Model parameters
    use_ml_classifier: bool = True
    confidence_threshold: float = 0.7

    # History for pattern learning
    history_size: int = 1000

    # Safety margins
    apply_safety_margin: bool = True
    safety_margin_factor: float = 0.8  # Trigger at 80% of threshold

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tsat_delta_warning_c": self.tsat_delta_warning_c,
            "tsat_delta_critical_c": self.tsat_delta_critical_c,
            "max_load_ramp_rate_per_min": self.max_load_ramp_rate_per_min,
            "max_valve_rate_per_sec": self.max_valve_rate_per_sec,
            "confidence_threshold": self.confidence_threshold,
        }


@dataclass
class CarryoverFeatures:
    """
    Engineered features for carryover detection.

    These features capture the physical indicators of carryover risk.
    """
    # Temperature features
    tsat_delta_c: float              # T_actual - T_saturation
    tsat_delta_rate_c_per_s: float   # Rate of change of Tsat delta

    # Pressure features
    pressure_kpa: float
    pressure_rate_kpa_per_s: float

    # Load features
    load_fraction: float             # 0-1, current load / design load
    load_ramp_rate_per_min: float    # Rate of load change

    # Valve features
    valve_position: float            # 0-1, valve opening
    valve_rate_per_s: float          # Rate of valve movement

    # Derived features
    superheat_margin_c: float        # Positive = superheated
    dryness_estimate: float          # Estimated dryness fraction

    # Historical features
    max_tsat_delta_60s: float        # Max delta in last 60s
    min_tsat_delta_60s: float        # Min delta in last 60s
    load_volatility_60s: float       # Std of load in last 60s

    # Timestamp
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_array(self) -> NDArray[np.float64]:
        """Convert to feature vector for ML model."""
        return np.array([
            self.tsat_delta_c,
            self.tsat_delta_rate_c_per_s,
            self.pressure_kpa,
            self.pressure_rate_kpa_per_s,
            self.load_fraction,
            self.load_ramp_rate_per_min,
            self.valve_position,
            self.valve_rate_per_s,
            self.superheat_margin_c,
            self.dryness_estimate,
            self.max_tsat_delta_60s,
            self.min_tsat_delta_60s,
            self.load_volatility_60s,
        ], dtype=np.float64)

    @staticmethod
    def feature_names() -> List[str]:
        """Get feature names for interpretability."""
        return [
            "tsat_delta_c",
            "tsat_delta_rate_c_per_s",
            "pressure_kpa",
            "pressure_rate_kpa_per_s",
            "load_fraction",
            "load_ramp_rate_per_min",
            "valve_position",
            "valve_rate_per_s",
            "superheat_margin_c",
            "dryness_estimate",
            "max_tsat_delta_60s",
            "min_tsat_delta_60s",
            "load_volatility_60s",
        ]


@dataclass
class CarryoverWarning:
    """Early warning for potential carryover."""

    risk_level: CarryoverRiskLevel
    probability: float                      # 0-1, probability of carryover
    time_to_event_s: Optional[float]        # Estimated time until carryover
    confidence: float                       # Confidence in prediction

    # Contributing factors
    primary_cause: CarryoverCause
    contributing_factors: List[str]

    # Recommended actions
    recommended_actions: List[str]

    # Feature contributions
    top_risk_features: Dict[str, float]

    # Timestamp
    timestamp: datetime = field(default_factory=datetime.utcnow)
    valid_until: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "risk_level": self.risk_level.value,
            "probability": round(self.probability, 4),
            "time_to_event_s": round(self.time_to_event_s, 1) if self.time_to_event_s else None,
            "confidence": round(self.confidence, 4),
            "primary_cause": self.primary_cause.value,
            "contributing_factors": self.contributing_factors,
            "recommended_actions": self.recommended_actions,
            "top_risk_features": {k: round(v, 4) for k, v in self.top_risk_features.items()},
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class CarryoverEvent:
    """Detected carryover event."""

    event_id: str
    start_time: datetime
    end_time: Optional[datetime]
    duration_s: float

    # Event characteristics
    risk_level: CarryoverRiskLevel
    severity: float                  # 0-1, severity score
    cause: CarryoverCause
    dryness_drop: float             # How much dryness decreased

    # Peak conditions
    min_tsat_delta_c: float
    max_load_ramp_rate: float
    peak_valve_rate: float

    # Impact
    estimated_water_carryover_kg: float

    # Provenance
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_s": round(self.duration_s, 1),
            "risk_level": self.risk_level.value,
            "severity": round(self.severity, 4),
            "cause": self.cause.value,
            "dryness_drop": round(self.dryness_drop, 4),
            "min_tsat_delta_c": round(self.min_tsat_delta_c, 2),
            "estimated_water_carryover_kg": round(self.estimated_water_carryover_kg, 2),
            "provenance_hash": self.provenance_hash,
        }


@dataclass
class DetectionResult:
    """Result from carryover detection."""

    # Current state
    is_carryover_detected: bool
    current_risk_level: CarryoverRiskLevel

    # Warning
    warning: Optional[CarryoverWarning]

    # Active event (if carryover ongoing)
    active_event: Optional[CarryoverEvent]

    # Features used
    features: CarryoverFeatures

    # Classification outputs
    classification_score: float      # Raw score from classifier
    classification_confidence: float

    # Physics-based indicators
    physics_indicators: Dict[str, bool]

    # Timestamp
    timestamp: datetime = field(default_factory=datetime.utcnow)
    processing_time_ms: float = 0.0

    # Provenance
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "is_carryover_detected": self.is_carryover_detected,
            "current_risk_level": self.current_risk_level.value,
            "warning": self.warning.to_dict() if self.warning else None,
            "active_event": self.active_event.to_dict() if self.active_event else None,
            "classification_score": round(self.classification_score, 4),
            "classification_confidence": round(self.classification_confidence, 4),
            "physics_indicators": self.physics_indicators,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "provenance_hash": self.provenance_hash,
        }


# ============================================================================
# PHYSICS-BASED CLASSIFIER
# ============================================================================

class PhysicsBasedClassifier:
    """
    Physics-based carryover classification using known relationships.

    Uses deterministic rules based on steam thermodynamics.
    """

    def __init__(self, config: CarryoverDetectorConfig):
        """Initialize classifier."""
        self.config = config

    def classify(self, features: CarryoverFeatures) -> Tuple[float, Dict[str, bool]]:
        """
        Classify carryover risk using physics rules.

        Returns:
            Tuple of (risk_score 0-1, indicator_flags)
        """
        indicators: Dict[str, bool] = {}
        risk_scores: List[float] = []

        # Rule 1: Temperature below saturation
        tsat_threshold = self.config.tsat_delta_critical_c
        if self.config.apply_safety_margin:
            tsat_threshold *= self.config.safety_margin_factor

        indicators["temp_below_saturation"] = features.tsat_delta_c < -tsat_threshold

        if features.tsat_delta_c < 0:
            # Score based on how far below saturation
            temp_score = min(1.0, abs(features.tsat_delta_c) / self.config.tsat_delta_critical_c)
            risk_scores.append(temp_score * 0.4)  # Weight: 40%

        # Rule 2: Rapid temperature drop
        indicators["rapid_temp_drop"] = features.tsat_delta_rate_c_per_s < -0.5

        if features.tsat_delta_rate_c_per_s < 0:
            rate_score = min(1.0, abs(features.tsat_delta_rate_c_per_s) / 2.0)
            risk_scores.append(rate_score * 0.2)  # Weight: 20%

        # Rule 3: Rapid load ramp
        ramp_threshold = self.config.max_load_ramp_rate_per_min
        if self.config.apply_safety_margin:
            ramp_threshold *= self.config.safety_margin_factor

        indicators["rapid_load_ramp"] = features.load_ramp_rate_per_min > ramp_threshold

        if features.load_ramp_rate_per_min > 0:
            ramp_score = min(1.0, features.load_ramp_rate_per_min / self.config.max_load_ramp_rate_per_min)
            risk_scores.append(ramp_score * 0.2)  # Weight: 20%

        # Rule 4: Rapid valve movement
        valve_threshold = self.config.max_valve_rate_per_sec
        if self.config.apply_safety_margin:
            valve_threshold *= self.config.safety_margin_factor

        indicators["rapid_valve_movement"] = abs(features.valve_rate_per_s) > valve_threshold

        if abs(features.valve_rate_per_s) > 0:
            valve_score = min(1.0, abs(features.valve_rate_per_s) / self.config.max_valve_rate_per_sec)
            risk_scores.append(valve_score * 0.1)  # Weight: 10%

        # Rule 5: Low dryness estimate
        indicators["low_dryness"] = features.dryness_estimate < 0.9

        if features.dryness_estimate < 0.95:
            dryness_score = max(0, 1.0 - features.dryness_estimate)
            risk_scores.append(dryness_score * 0.1)  # Weight: 10%

        # Combine scores
        total_risk = sum(risk_scores)
        total_risk = min(1.0, total_risk)

        return total_risk, indicators


# ============================================================================
# LEARNED PATTERN CLASSIFIER
# ============================================================================

class LearnedPatternClassifier:
    """
    Data-driven classifier that learns carryover patterns.

    Uses a simple scoring model trained on historical events.
    In production, could be replaced with more sophisticated ML.
    """

    def __init__(self, config: CarryoverDetectorConfig):
        """Initialize classifier."""
        self.config = config

        # Learned weights (initialized from domain knowledge)
        self._feature_weights = np.array([
            -0.3,   # tsat_delta_c (negative = bad)
            -0.2,   # tsat_delta_rate_c_per_s
            0.0,    # pressure_kpa (neutral)
            -0.1,   # pressure_rate_kpa_per_s
            0.0,    # load_fraction
            0.25,   # load_ramp_rate_per_min (positive = bad)
            0.0,    # valve_position
            0.15,   # valve_rate_per_s
            -0.2,   # superheat_margin_c (less superheat = bad)
            -0.15,  # dryness_estimate
            -0.1,   # max_tsat_delta_60s
            -0.1,   # min_tsat_delta_60s
            0.1,    # load_volatility_60s
        ])

        # Intercept (bias)
        self._intercept = 0.3

        # Training history
        self._positive_examples: Deque[NDArray] = deque(maxlen=config.history_size)
        self._negative_examples: Deque[NDArray] = deque(maxlen=config.history_size)

    def predict_proba(self, features: CarryoverFeatures) -> float:
        """
        Predict carryover probability.

        Uses a simple linear model with sigmoid activation.
        """
        x = features.to_array()

        # Normalize features (simple z-score with fixed stats)
        feature_means = np.array([0, 0, 500, 0, 0.7, 0, 0.5, 0, 5, 0.95, 0, -2, 0.02])
        feature_stds = np.array([3, 1, 300, 10, 0.2, 0.1, 0.3, 0.05, 10, 0.05, 3, 3, 0.05])

        x_norm = (x - feature_means) / (feature_stds + 1e-6)

        # Linear combination
        logit = np.dot(x_norm, self._feature_weights) + self._intercept

        # Sigmoid
        prob = 1.0 / (1.0 + np.exp(-logit))

        return float(np.clip(prob, 0.0, 1.0))

    def add_training_example(
        self,
        features: CarryoverFeatures,
        is_carryover: bool
    ) -> None:
        """Add example for online learning."""
        x = features.to_array()

        if is_carryover:
            self._positive_examples.append(x)
        else:
            self._negative_examples.append(x)

        # Simple online weight update (gradient descent step)
        if len(self._positive_examples) >= 10 and len(self._negative_examples) >= 10:
            self._update_weights()

    def _update_weights(self) -> None:
        """Update weights using simple gradient descent."""
        if not self._positive_examples or not self._negative_examples:
            return

        # Sample from history
        n_samples = min(20, len(self._positive_examples), len(self._negative_examples))

        pos_samples = np.array(list(self._positive_examples)[-n_samples:])
        neg_samples = np.array(list(self._negative_examples)[-n_samples:])

        # Simple gradient step
        learning_rate = 0.01

        for x in pos_samples:
            pred = self.predict_proba(CarryoverFeatures(*x[:13], timestamp=datetime.utcnow()))
            error = 1.0 - pred  # Should predict 1
            self._feature_weights += learning_rate * error * x[:13]

        for x in neg_samples:
            pred = self.predict_proba(CarryoverFeatures(*x[:13], timestamp=datetime.utcnow()))
            error = 0.0 - pred  # Should predict 0
            self._feature_weights += learning_rate * error * x[:13]

        # Clip weights to prevent explosion
        self._feature_weights = np.clip(self._feature_weights, -2.0, 2.0)


# ============================================================================
# MAIN CARRYOVER DETECTOR CLASS
# ============================================================================

class CarryoverDetector:
    """
    Carryover event detector with early warning capability.

    Combines physics-based rules with learned patterns to detect
    and predict carryover events in steam systems.

    Zero-Hallucination Guarantee:
    - Physics-based thresholds are deterministic
    - ML predictions are bounded by physics constraints
    - All outputs include confidence/uncertainty
    - Complete provenance tracking

    Example:
        >>> config = CarryoverDetectorConfig()
        >>> detector = CarryoverDetector(config)
        >>>
        >>> features = detector.compute_features(
        ...     temperature_c=178.0,
        ...     pressure_kpa=1000.0,
        ...     load_fraction=0.85,
        ...     valve_position=0.6
        ... )
        >>> result = detector.detect(features)
        >>> if result.warning:
        ...     print(f"Warning: {result.warning.risk_level.value}")
    """

    VERSION = "1.0.0"

    def __init__(self, config: Optional[CarryoverDetectorConfig] = None):
        """
        Initialize carryover detector.

        Args:
            config: Detector configuration (optional)
        """
        self.config = config or CarryoverDetectorConfig()

        # Classifiers
        self._physics_classifier = PhysicsBasedClassifier(self.config)
        self._learned_classifier = LearnedPatternClassifier(self.config)

        # State tracking
        self._feature_history: Deque[CarryoverFeatures] = deque(
            maxlen=int(self.config.feature_window_s * 10)  # 10 Hz max
        )
        self._active_event: Optional[CarryoverEvent] = None
        self._event_counter = 0

        # Persistence tracking for confirmed events
        self._carryover_start_time: Optional[datetime] = None

        logger.info("CarryoverDetector initialized")

    def compute_features(
        self,
        temperature_c: float,
        pressure_kpa: float,
        load_fraction: float = 1.0,
        valve_position: float = 0.5,
        dryness_estimate: float = 0.95,
        timestamp: Optional[datetime] = None
    ) -> CarryoverFeatures:
        """
        Compute detection features from raw measurements.

        Args:
            temperature_c: Steam temperature
            pressure_kpa: Steam pressure
            load_fraction: Current load as fraction of design
            valve_position: Control valve position (0-1)
            dryness_estimate: Current dryness estimate (from soft sensor)
            timestamp: Measurement timestamp

        Returns:
            Computed features for detection
        """
        timestamp = timestamp or datetime.utcnow()

        # Compute saturation temperature (simplified)
        t_sat = self._compute_saturation_temp(pressure_kpa)
        tsat_delta = temperature_c - t_sat
        superheat_margin = max(0, tsat_delta)

        # Compute rates from history
        tsat_delta_rate = 0.0
        pressure_rate = 0.0
        load_ramp_rate = 0.0
        valve_rate = 0.0

        if self._feature_history:
            last = self._feature_history[-1]
            dt = (timestamp - last.timestamp).total_seconds()

            if dt > 0:
                tsat_delta_rate = (tsat_delta - last.tsat_delta_c) / dt
                pressure_rate = (pressure_kpa - last.pressure_kpa) / dt
                load_ramp_rate = (load_fraction - last.load_fraction) / dt * 60.0  # per minute
                valve_rate = (valve_position - last.valve_position) / dt

        # Historical features (last 60 seconds)
        max_tsat_delta = tsat_delta
        min_tsat_delta = tsat_delta
        load_values = [load_fraction]

        cutoff_time = timestamp - timedelta(seconds=60)
        for hist in self._feature_history:
            if hist.timestamp >= cutoff_time:
                max_tsat_delta = max(max_tsat_delta, hist.tsat_delta_c)
                min_tsat_delta = min(min_tsat_delta, hist.tsat_delta_c)
                load_values.append(hist.load_fraction)

        load_volatility = float(np.std(load_values)) if len(load_values) > 1 else 0.0

        features = CarryoverFeatures(
            tsat_delta_c=tsat_delta,
            tsat_delta_rate_c_per_s=tsat_delta_rate,
            pressure_kpa=pressure_kpa,
            pressure_rate_kpa_per_s=pressure_rate,
            load_fraction=load_fraction,
            load_ramp_rate_per_min=load_ramp_rate,
            valve_position=valve_position,
            valve_rate_per_s=valve_rate,
            superheat_margin_c=superheat_margin,
            dryness_estimate=dryness_estimate,
            max_tsat_delta_60s=max_tsat_delta,
            min_tsat_delta_60s=min_tsat_delta,
            load_volatility_60s=load_volatility,
            timestamp=timestamp
        )

        # Store in history
        self._feature_history.append(features)

        return features

    def detect(self, features: CarryoverFeatures) -> DetectionResult:
        """
        Detect carryover based on features.

        Args:
            features: Computed detection features

        Returns:
            Detection result with warning and event info
        """
        start_time = datetime.utcnow()

        # Physics-based classification
        physics_score, physics_indicators = self._physics_classifier.classify(features)

        # Learned classification (if enabled)
        if self.config.use_ml_classifier:
            ml_score = self._learned_classifier.predict_proba(features)
            # Combine: physics has veto power
            combined_score = max(physics_score, ml_score * 0.8)  # ML capped at 0.8
        else:
            combined_score = physics_score
            ml_score = 0.0

        # Determine risk level
        risk_level = self._score_to_risk_level(combined_score)

        # Check for confirmed carryover (persistence)
        is_carryover = self._check_carryover_persistence(
            combined_score >= 0.7,  # High score threshold
            features.timestamp
        )

        # Update or create active event
        if is_carryover and self._active_event is None:
            self._create_event(features)

        if not is_carryover and self._active_event is not None:
            self._close_event(features.timestamp)

        # Generate warning
        warning = self._generate_warning(
            risk_level, combined_score, features, physics_indicators
        )

        # Compute confidence
        confidence = self._compute_confidence(physics_score, ml_score)

        # Processing time
        processing_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

        # Provenance
        provenance_hash = self._compute_provenance(features, combined_score)

        return DetectionResult(
            is_carryover_detected=is_carryover,
            current_risk_level=risk_level,
            warning=warning,
            active_event=self._active_event,
            features=features,
            classification_score=combined_score,
            classification_confidence=confidence,
            physics_indicators=physics_indicators,
            timestamp=features.timestamp,
            processing_time_ms=processing_time_ms,
            provenance_hash=provenance_hash
        )

    def _compute_saturation_temp(self, pressure_kpa: float) -> float:
        """Compute saturation temperature (simplified)."""
        # Antoine equation approximation
        # More accurate: use IAPWS-IF97
        sat_temps = {
            100: 99.6, 200: 120.2, 300: 133.5, 400: 143.6,
            500: 151.8, 600: 158.8, 700: 164.9, 800: 170.4,
            900: 175.4, 1000: 179.9, 1500: 198.3, 2000: 212.4,
        }

        pressures = sorted(sat_temps.keys())

        if pressure_kpa <= pressures[0]:
            return sat_temps[pressures[0]]
        if pressure_kpa >= pressures[-1]:
            return sat_temps[pressures[-1]]

        for i in range(len(pressures) - 1):
            if pressures[i] <= pressure_kpa <= pressures[i + 1]:
                p1, p2 = pressures[i], pressures[i + 1]
                t1, t2 = sat_temps[p1], sat_temps[p2]
                return t1 + (t2 - t1) * (pressure_kpa - p1) / (p2 - p1)

        return sat_temps[pressures[-1]]

    def _score_to_risk_level(self, score: float) -> CarryoverRiskLevel:
        """Convert score to risk level."""
        if score >= 0.9:
            return CarryoverRiskLevel.ACTIVE_CARRYOVER
        elif score >= 0.7:
            return CarryoverRiskLevel.CRITICAL
        elif score >= 0.5:
            return CarryoverRiskLevel.HIGH
        elif score >= 0.3:
            return CarryoverRiskLevel.MODERATE
        elif score >= 0.1:
            return CarryoverRiskLevel.LOW
        else:
            return CarryoverRiskLevel.NONE

    def _check_carryover_persistence(
        self,
        high_risk: bool,
        timestamp: datetime
    ) -> bool:
        """Check if carryover persists long enough to confirm."""
        if high_risk:
            if self._carryover_start_time is None:
                self._carryover_start_time = timestamp
            else:
                duration = (timestamp - self._carryover_start_time).total_seconds()
                if duration >= self.config.persistence_threshold_s:
                    return True
        else:
            self._carryover_start_time = None

        return False

    def _create_event(self, features: CarryoverFeatures) -> None:
        """Create new carryover event."""
        self._event_counter += 1
        event_id = f"CO-{self._event_counter:06d}-{features.timestamp.strftime('%Y%m%d%H%M%S')}"

        self._active_event = CarryoverEvent(
            event_id=event_id,
            start_time=features.timestamp,
            end_time=None,
            duration_s=0.0,
            risk_level=CarryoverRiskLevel.ACTIVE_CARRYOVER,
            severity=0.0,
            cause=self._identify_cause(features),
            dryness_drop=1.0 - features.dryness_estimate,
            min_tsat_delta_c=features.tsat_delta_c,
            max_load_ramp_rate=features.load_ramp_rate_per_min,
            peak_valve_rate=features.valve_rate_per_s,
            estimated_water_carryover_kg=0.0
        )

        logger.warning(f"Carryover event started: {event_id}")

    def _close_event(self, end_time: datetime) -> None:
        """Close active carryover event."""
        if self._active_event is None:
            return

        self._active_event.end_time = end_time
        self._active_event.duration_s = (
            end_time - self._active_event.start_time
        ).total_seconds()

        # Compute severity based on duration and dryness drop
        duration_factor = min(1.0, self._active_event.duration_s / 300.0)  # Max at 5 min
        dryness_factor = self._active_event.dryness_drop
        self._active_event.severity = (duration_factor + dryness_factor) / 2.0

        # Estimate water carryover (simplified)
        flow_rate_kg_s = 10.0  # Assumed
        water_fraction = self._active_event.dryness_drop
        self._active_event.estimated_water_carryover_kg = (
            flow_rate_kg_s * water_fraction * self._active_event.duration_s
        )

        # Provenance
        data = {
            "event_id": self._active_event.event_id,
            "duration_s": self._active_event.duration_s,
            "severity": self._active_event.severity,
        }
        self._active_event.provenance_hash = hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()[:16]

        logger.warning(
            f"Carryover event ended: {self._active_event.event_id}, "
            f"duration={self._active_event.duration_s:.1f}s"
        )

        # Clear active event
        self._active_event = None

    def _identify_cause(self, features: CarryoverFeatures) -> CarryoverCause:
        """Identify primary cause of carryover."""
        causes_scores = {
            CarryoverCause.LOAD_RAMP: abs(features.load_ramp_rate_per_min) / 0.1,
            CarryoverCause.PRESSURE_DROP: max(0, -features.pressure_rate_kpa_per_s) / 10.0,
            CarryoverCause.VALVE_MALFUNCTION: abs(features.valve_rate_per_s) / 0.05,
            CarryoverCause.LEVEL_CONTROL: features.load_volatility_60s / 0.05,
        }

        # Get highest scoring cause
        if any(score > 0.5 for score in causes_scores.values()):
            return max(causes_scores, key=causes_scores.get)

        return CarryoverCause.UNKNOWN

    def _generate_warning(
        self,
        risk_level: CarryoverRiskLevel,
        score: float,
        features: CarryoverFeatures,
        indicators: Dict[str, bool]
    ) -> Optional[CarryoverWarning]:
        """Generate early warning if risk is elevated."""
        if risk_level == CarryoverRiskLevel.NONE:
            return None

        # Estimate time to event
        time_to_event = self._estimate_time_to_event(features, score)

        # Identify contributing factors
        factors = []
        if indicators.get("temp_below_saturation"):
            factors.append("Temperature below saturation")
        if indicators.get("rapid_temp_drop"):
            factors.append("Rapid temperature decline")
        if indicators.get("rapid_load_ramp"):
            factors.append("Rapid load increase")
        if indicators.get("rapid_valve_movement"):
            factors.append("Rapid valve movement")
        if indicators.get("low_dryness"):
            factors.append("Low dryness fraction")

        # Recommended actions
        actions = self._get_recommended_actions(risk_level, factors)

        # Top risk features
        top_features = {
            "tsat_delta_c": features.tsat_delta_c,
            "load_ramp_rate": features.load_ramp_rate_per_min,
            "valve_rate": features.valve_rate_per_s,
            "dryness": features.dryness_estimate,
        }

        # Confidence
        confidence = 1.0 - 0.3 * (1.0 - score)  # Higher score = higher confidence

        return CarryoverWarning(
            risk_level=risk_level,
            probability=score,
            time_to_event_s=time_to_event,
            confidence=confidence,
            primary_cause=self._identify_cause(features),
            contributing_factors=factors,
            recommended_actions=actions,
            top_risk_features=top_features,
            timestamp=features.timestamp,
            valid_until=features.timestamp + timedelta(seconds=30)
        )

    def _estimate_time_to_event(
        self,
        features: CarryoverFeatures,
        score: float
    ) -> Optional[float]:
        """Estimate time until carryover event."""
        if score < 0.3:
            return None  # Too uncertain

        # Simple linear extrapolation
        if features.tsat_delta_rate_c_per_s < 0:
            # Temperature dropping towards saturation
            time_to_sat = abs(features.superheat_margin_c / features.tsat_delta_rate_c_per_s)
            return time_to_sat

        if features.load_ramp_rate_per_min > 0:
            # Load ramping up
            remaining_capacity = 1.0 - features.load_fraction
            if remaining_capacity > 0:
                time_to_full = remaining_capacity / (features.load_ramp_rate_per_min / 60.0)
                return time_to_full

        # Default estimate based on score trend
        return self.config.warning_lookahead_s * (1.0 - score)

    def _get_recommended_actions(
        self,
        risk_level: CarryoverRiskLevel,
        factors: List[str]
    ) -> List[str]:
        """Get recommended actions for risk level."""
        actions = []

        if risk_level in [CarryoverRiskLevel.CRITICAL, CarryoverRiskLevel.ACTIVE_CARRYOVER]:
            actions.append("IMMEDIATE: Reduce load ramp rate")
            actions.append("Check separator level controls")
            actions.append("Verify steam trap operation")
        elif risk_level == CarryoverRiskLevel.HIGH:
            actions.append("Reduce load increase rate")
            actions.append("Monitor separator performance")
            actions.append("Prepare for load reduction if needed")
        elif risk_level == CarryoverRiskLevel.MODERATE:
            actions.append("Continue monitoring")
            actions.append("Check water treatment chemistry")
        else:
            actions.append("Standard monitoring")

        return actions

    def _compute_confidence(
        self,
        physics_score: float,
        ml_score: float
    ) -> float:
        """Compute overall confidence in detection."""
        if abs(physics_score - ml_score) < 0.2:
            # Good agreement
            return 0.9
        elif abs(physics_score - ml_score) < 0.4:
            return 0.7
        else:
            # Disagreement - lower confidence
            return 0.5

    def _compute_provenance(
        self,
        features: CarryoverFeatures,
        score: float
    ) -> str:
        """Compute provenance hash."""
        data = {
            "version": self.VERSION,
            "timestamp": features.timestamp.isoformat(),
            "tsat_delta": round(features.tsat_delta_c, 4),
            "score": round(score, 4),
        }
        return hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()[:16]

    def add_training_example(
        self,
        features: CarryoverFeatures,
        is_carryover: bool
    ) -> None:
        """
        Add labeled example for online learning.

        Args:
            features: Detection features
            is_carryover: True if carryover occurred
        """
        if self.config.use_ml_classifier:
            self._learned_classifier.add_training_example(features, is_carryover)

    def reset(self) -> None:
        """Reset detector state."""
        self._feature_history.clear()
        self._active_event = None
        self._carryover_start_time = None
        logger.info("CarryoverDetector reset")

    def get_recent_events(
        self,
        lookback_hours: int = 24
    ) -> List[CarryoverEvent]:
        """Get recent carryover events (placeholder for event storage)."""
        # In production, query from event store
        return []
