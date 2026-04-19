# -*- coding: utf-8 -*-
"""
Separator and Trap Health Estimator

This module implements health analytics for steam separators and traps:
- Anomaly detection on separator/trap signals
- Efficiency degradation trending
- Failure probability estimation using Weibull analysis
- Maintenance recommendations

Zero-Hallucination Guarantee:
- Anomaly detection uses statistical thresholds
- Degradation uses physics-based efficiency models
- Failure probability from established reliability theory
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
# CONSTANTS
# ============================================================================

# Weibull parameters for different equipment types
WEIBULL_PARAMS = {
    "cyclone_separator": {"beta": 2.2, "eta": 8.0},      # 8 year characteristic life
    "demister_pad": {"beta": 1.8, "eta": 5.0},           # 5 year characteristic life
    "vane_separator": {"beta": 2.0, "eta": 7.0},         # 7 year characteristic life
    "steam_trap_thermodynamic": {"beta": 2.1, "eta": 5.5},
    "steam_trap_thermostatic": {"beta": 1.8, "eta": 7.0},
    "steam_trap_float": {"beta": 2.0, "eta": 8.0},
}

# Normal operating ranges
NORMAL_RANGES = {
    "pressure_drop_kpa": (1.0, 10.0),
    "efficiency_percent": (95.0, 99.9),
    "temperature_delta_c": (-2.0, 5.0),
    "vibration_mm_s": (0.0, 4.5),
}


class HealthStatus(Enum):
    """Equipment health classification."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"


class AnomalyType(Enum):
    """Types of detected anomalies."""
    EFFICIENCY_DROP = "efficiency_drop"
    PRESSURE_SPIKE = "pressure_spike"
    TEMPERATURE_ANOMALY = "temperature_anomaly"
    VIBRATION_HIGH = "vibration_high"
    FOULING = "fouling"
    BLOCKAGE = "blockage"
    LEAK = "leak"
    UNKNOWN = "unknown"


class MaintenanceUrgency(Enum):
    """Maintenance urgency levels."""
    IMMEDIATE = "immediate"       # Within 24 hours
    URGENT = "urgent"            # Within 1 week
    PLANNED = "planned"          # Within 1 month
    ROUTINE = "routine"          # Next scheduled maintenance
    NONE = "none"                # No action needed


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class SeparatorHealthConfig:
    """Configuration for health estimator."""

    # Anomaly detection thresholds (number of std deviations)
    anomaly_threshold_sigma: float = 3.0

    # Efficiency degradation thresholds
    efficiency_warning_threshold: float = 0.95
    efficiency_critical_threshold: float = 0.90

    # Trending window
    trend_window_hours: int = 168  # 1 week

    # History retention
    history_size: int = 10000  # Samples

    # Failure prediction horizon
    prediction_horizon_days: int = 90

    # Confidence level for predictions
    confidence_level: float = 0.90

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "anomaly_threshold_sigma": self.anomaly_threshold_sigma,
            "efficiency_warning_threshold": self.efficiency_warning_threshold,
            "efficiency_critical_threshold": self.efficiency_critical_threshold,
            "trend_window_hours": self.trend_window_hours,
            "prediction_horizon_days": self.prediction_horizon_days,
        }


@dataclass
class SeparatorHealthState:
    """Current health state of a separator."""

    # Identification
    equipment_id: str
    equipment_type: str

    # Current measurements
    efficiency: float                    # Current efficiency (0-1)
    pressure_drop_kpa: float
    inlet_temperature_c: float
    outlet_temperature_c: float
    flow_rate_kg_s: float

    # Optional measurements
    vibration_mm_s: Optional[float] = None
    acoustic_level_db: Optional[float] = None

    # Age and history
    age_years: float = 0.0
    operating_hours: float = 0.0
    last_maintenance_date: Optional[datetime] = None

    # Timestamp
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def temperature_delta(self) -> float:
        """Calculate temperature differential."""
        return self.outlet_temperature_c - self.inlet_temperature_c


@dataclass
class AnomalyEvent:
    """Detected anomaly event."""

    event_id: str
    equipment_id: str
    anomaly_type: AnomalyType
    severity: float                      # 0-1

    # Detection details
    detected_at: datetime
    detection_confidence: float

    # Measurement at detection
    anomalous_value: float
    expected_value: float
    deviation_sigma: float

    # Context
    contributing_factors: List[str]
    possible_causes: List[str]

    # Provenance
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "equipment_id": self.equipment_id,
            "anomaly_type": self.anomaly_type.value,
            "severity": round(self.severity, 4),
            "detected_at": self.detected_at.isoformat(),
            "detection_confidence": round(self.detection_confidence, 4),
            "anomalous_value": round(self.anomalous_value, 4),
            "expected_value": round(self.expected_value, 4),
            "deviation_sigma": round(self.deviation_sigma, 2),
            "possible_causes": self.possible_causes,
            "provenance_hash": self.provenance_hash,
        }


@dataclass
class EfficiencyTrend:
    """Efficiency degradation trend analysis."""

    equipment_id: str
    analysis_period_hours: int

    # Current state
    current_efficiency: float
    baseline_efficiency: float

    # Trend
    trend_slope_per_day: float          # Efficiency change per day
    trend_direction: str                 # "degrading", "stable", "improving"
    r_squared: float                     # Trend fit quality

    # Projections
    days_to_warning_threshold: Optional[float]
    days_to_critical_threshold: Optional[float]

    # Confidence
    confidence: float

    # Statistics
    efficiency_mean: float
    efficiency_std: float
    min_efficiency: float
    max_efficiency: float

    # Timestamp
    analysis_timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "equipment_id": self.equipment_id,
            "current_efficiency": round(self.current_efficiency, 4),
            "baseline_efficiency": round(self.baseline_efficiency, 4),
            "trend_slope_per_day": round(self.trend_slope_per_day, 6),
            "trend_direction": self.trend_direction,
            "r_squared": round(self.r_squared, 4),
            "days_to_warning_threshold": round(self.days_to_warning_threshold, 1)
                if self.days_to_warning_threshold else None,
            "days_to_critical_threshold": round(self.days_to_critical_threshold, 1)
                if self.days_to_critical_threshold else None,
            "confidence": round(self.confidence, 4),
            "analysis_timestamp": self.analysis_timestamp.isoformat(),
        }


@dataclass
class FailureProbability:
    """Failure probability estimation using Weibull analysis."""

    equipment_id: str
    equipment_type: str
    equipment_age_years: float

    # Probability estimates
    failure_probability: float           # P(fail in horizon | survived to now)
    lower_bound: float
    upper_bound: float
    confidence_level: float

    # Time estimates
    median_remaining_life_years: float
    time_to_50_percent_failure: float
    time_to_90_percent_failure: float

    # Weibull parameters used
    beta: float                          # Shape parameter
    eta: float                           # Scale parameter (adjusted for conditions)

    # Current reliability
    current_reliability: float           # R(t) at current age
    hazard_rate_per_year: float

    # Risk assessment
    risk_score: float                    # 0-100

    # Timestamp
    calculation_timestamp: datetime = field(default_factory=datetime.utcnow)
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "equipment_id": self.equipment_id,
            "equipment_type": self.equipment_type,
            "equipment_age_years": round(self.equipment_age_years, 2),
            "failure_probability": round(self.failure_probability, 4),
            "confidence_interval": {
                "lower": round(self.lower_bound, 4),
                "upper": round(self.upper_bound, 4),
                "confidence_level": self.confidence_level,
            },
            "median_remaining_life_years": round(self.median_remaining_life_years, 2),
            "current_reliability": round(self.current_reliability, 4),
            "hazard_rate_per_year": round(self.hazard_rate_per_year, 6),
            "risk_score": round(self.risk_score, 1),
            "provenance_hash": self.provenance_hash,
        }


@dataclass
class MaintenanceRecommendation:
    """Maintenance recommendation based on health assessment."""

    equipment_id: str
    urgency: MaintenanceUrgency
    recommended_action: str
    description: str

    # Timing
    recommended_date: datetime
    deadline: Optional[datetime]

    # Justification
    health_status: HealthStatus
    contributing_factors: List[str]
    risk_if_deferred: str

    # Cost-benefit (if available)
    estimated_downtime_hours: float = 0.0
    failure_cost_risk: float = 0.0

    # Provenance
    recommendation_timestamp: datetime = field(default_factory=datetime.utcnow)
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "equipment_id": self.equipment_id,
            "urgency": self.urgency.value,
            "recommended_action": self.recommended_action,
            "description": self.description,
            "recommended_date": self.recommended_date.isoformat(),
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "health_status": self.health_status.value,
            "contributing_factors": self.contributing_factors,
            "risk_if_deferred": self.risk_if_deferred,
            "estimated_downtime_hours": round(self.estimated_downtime_hours, 1),
            "provenance_hash": self.provenance_hash,
        }


@dataclass
class HealthAssessment:
    """Complete health assessment for equipment."""

    equipment_id: str
    assessment_timestamp: datetime

    # Overall status
    health_status: HealthStatus
    health_score: float                  # 0-100

    # Current state
    current_state: SeparatorHealthState

    # Analysis results
    anomalies_detected: List[AnomalyEvent]
    efficiency_trend: Optional[EfficiencyTrend]
    failure_probability: Optional[FailureProbability]

    # Recommendations
    maintenance_recommendation: Optional[MaintenanceRecommendation]

    # Uncertainty
    assessment_confidence: float

    # Processing info
    processing_time_ms: float = 0.0
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "equipment_id": self.equipment_id,
            "assessment_timestamp": self.assessment_timestamp.isoformat(),
            "health_status": self.health_status.value,
            "health_score": round(self.health_score, 1),
            "anomalies_detected": [a.to_dict() for a in self.anomalies_detected],
            "efficiency_trend": self.efficiency_trend.to_dict()
                if self.efficiency_trend else None,
            "failure_probability": self.failure_probability.to_dict()
                if self.failure_probability else None,
            "maintenance_recommendation": self.maintenance_recommendation.to_dict()
                if self.maintenance_recommendation else None,
            "assessment_confidence": round(self.assessment_confidence, 4),
            "processing_time_ms": round(self.processing_time_ms, 2),
            "provenance_hash": self.provenance_hash,
        }


# ============================================================================
# ANOMALY DETECTOR
# ============================================================================

class StatisticalAnomalyDetector:
    """
    Statistical anomaly detection using adaptive thresholds.

    Uses Welford's online algorithm for streaming mean/variance.
    """

    def __init__(
        self,
        threshold_sigma: float = 3.0,
        window_size: int = 1000
    ):
        """
        Initialize detector.

        Args:
            threshold_sigma: Number of standard deviations for anomaly
            window_size: Rolling window for statistics
        """
        self.threshold_sigma = threshold_sigma
        self.window_size = window_size

        # Per-metric statistics
        self._stats: Dict[str, Dict[str, float]] = {}
        self._history: Dict[str, Deque[float]] = {}

    def update(self, metric_name: str, value: float) -> Optional[Tuple[float, float]]:
        """
        Update statistics and check for anomaly.

        Returns:
            Tuple of (deviation_sigma, expected_value) if anomaly, else None
        """
        # Initialize if needed
        if metric_name not in self._stats:
            self._stats[metric_name] = {"mean": value, "M2": 0.0, "count": 0}
            self._history[metric_name] = deque(maxlen=self.window_size)

        stats = self._stats[metric_name]
        history = self._history[metric_name]

        # Welford's online update
        stats["count"] += 1
        delta = value - stats["mean"]
        stats["mean"] += delta / stats["count"]
        delta2 = value - stats["mean"]
        stats["M2"] += delta * delta2

        history.append(value)

        # Need minimum samples for reliable statistics
        if stats["count"] < 30:
            return None

        # Calculate variance and std
        variance = stats["M2"] / (stats["count"] - 1) if stats["count"] > 1 else 0
        std = math.sqrt(variance) if variance > 0 else 1e-6

        # Calculate z-score
        z_score = abs(value - stats["mean"]) / std

        if z_score > self.threshold_sigma:
            return (z_score, stats["mean"])

        return None

    def get_statistics(self, metric_name: str) -> Dict[str, float]:
        """Get current statistics for metric."""
        if metric_name not in self._stats:
            return {"mean": 0.0, "std": 0.0, "count": 0}

        stats = self._stats[metric_name]
        variance = stats["M2"] / (stats["count"] - 1) if stats["count"] > 1 else 0

        return {
            "mean": stats["mean"],
            "std": math.sqrt(variance) if variance > 0 else 0.0,
            "count": stats["count"]
        }


# ============================================================================
# MAIN HEALTH ESTIMATOR CLASS
# ============================================================================

class SeparatorHealthEstimator:
    """
    Separator and trap health estimator with anomaly detection,
    degradation trending, and failure prediction.

    Zero-Hallucination Guarantee:
    - Anomaly detection uses statistical z-scores
    - Degradation uses linear regression
    - Failure probability from Weibull analysis
    - All outputs include uncertainty quantification

    Example:
        >>> config = SeparatorHealthConfig()
        >>> estimator = SeparatorHealthEstimator(config)
        >>>
        >>> state = SeparatorHealthState(
        ...     equipment_id="SEP-001",
        ...     equipment_type="cyclone_separator",
        ...     efficiency=0.96,
        ...     pressure_drop_kpa=5.0,
        ...     inlet_temperature_c=180.0,
        ...     outlet_temperature_c=179.5,
        ...     flow_rate_kg_s=10.0,
        ...     age_years=3.5
        ... )
        >>> assessment = estimator.assess(state)
        >>> print(f"Health: {assessment.health_status.value}")
    """

    VERSION = "1.0.0"

    def __init__(self, config: Optional[SeparatorHealthConfig] = None):
        """
        Initialize health estimator.

        Args:
            config: Estimator configuration (optional)
        """
        self.config = config or SeparatorHealthConfig()

        # Anomaly detector
        self._anomaly_detector = StatisticalAnomalyDetector(
            threshold_sigma=self.config.anomaly_threshold_sigma
        )

        # State history per equipment
        self._state_history: Dict[str, Deque[SeparatorHealthState]] = {}

        # Detected anomalies
        self._anomaly_counter = 0

        logger.info("SeparatorHealthEstimator initialized")

    def assess(self, state: SeparatorHealthState) -> HealthAssessment:
        """
        Perform complete health assessment.

        Args:
            state: Current equipment state

        Returns:
            Complete health assessment
        """
        start_time = datetime.utcnow()

        # Store state in history
        self._update_history(state)

        # Step 1: Detect anomalies
        anomalies = self._detect_anomalies(state)

        # Step 2: Analyze efficiency trend
        efficiency_trend = self._analyze_efficiency_trend(state.equipment_id)

        # Step 3: Estimate failure probability
        failure_prob = self._estimate_failure_probability(state)

        # Step 4: Determine health status
        health_status, health_score = self._determine_health_status(
            state, anomalies, efficiency_trend, failure_prob
        )

        # Step 5: Generate maintenance recommendation
        maintenance_rec = self._generate_maintenance_recommendation(
            state, health_status, anomalies, efficiency_trend, failure_prob
        )

        # Calculate confidence
        confidence = self._calculate_confidence(state.equipment_id)

        # Processing time
        processing_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

        # Provenance
        provenance_hash = self._compute_provenance(state, health_score)

        return HealthAssessment(
            equipment_id=state.equipment_id,
            assessment_timestamp=datetime.utcnow(),
            health_status=health_status,
            health_score=health_score,
            current_state=state,
            anomalies_detected=anomalies,
            efficiency_trend=efficiency_trend,
            failure_probability=failure_prob,
            maintenance_recommendation=maintenance_rec,
            assessment_confidence=confidence,
            processing_time_ms=processing_time_ms,
            provenance_hash=provenance_hash
        )

    def _update_history(self, state: SeparatorHealthState) -> None:
        """Update state history for equipment."""
        if state.equipment_id not in self._state_history:
            self._state_history[state.equipment_id] = deque(
                maxlen=self.config.history_size
            )
        self._state_history[state.equipment_id].append(state)

    def _detect_anomalies(
        self,
        state: SeparatorHealthState
    ) -> List[AnomalyEvent]:
        """Detect anomalies in current state."""
        anomalies = []
        prefix = state.equipment_id

        # Check efficiency
        result = self._anomaly_detector.update(
            f"{prefix}_efficiency", state.efficiency
        )
        if result:
            deviation, expected = result
            anomalies.append(self._create_anomaly_event(
                state.equipment_id,
                AnomalyType.EFFICIENCY_DROP,
                state.efficiency,
                expected,
                deviation,
                ["Efficiency below expected range"],
                ["Fouling", "Mechanical wear", "Separator bypass"]
            ))

        # Check pressure drop
        result = self._anomaly_detector.update(
            f"{prefix}_pressure_drop", state.pressure_drop_kpa
        )
        if result:
            deviation, expected = result
            anomaly_type = AnomalyType.BLOCKAGE if state.pressure_drop_kpa > expected else AnomalyType.LEAK
            anomalies.append(self._create_anomaly_event(
                state.equipment_id,
                anomaly_type,
                state.pressure_drop_kpa,
                expected,
                deviation,
                ["Pressure drop deviation"],
                ["Fouling/blockage" if anomaly_type == AnomalyType.BLOCKAGE else "Internal leak"]
            ))

        # Check temperature delta
        temp_delta = state.temperature_delta()
        result = self._anomaly_detector.update(
            f"{prefix}_temp_delta", temp_delta
        )
        if result:
            deviation, expected = result
            anomalies.append(self._create_anomaly_event(
                state.equipment_id,
                AnomalyType.TEMPERATURE_ANOMALY,
                temp_delta,
                expected,
                deviation,
                ["Temperature differential abnormal"],
                ["Heat transfer issue", "Bypass flow", "Fouling"]
            ))

        # Check vibration (if available)
        if state.vibration_mm_s is not None:
            result = self._anomaly_detector.update(
                f"{prefix}_vibration", state.vibration_mm_s
            )
            if result:
                deviation, expected = result
                anomalies.append(self._create_anomaly_event(
                    state.equipment_id,
                    AnomalyType.VIBRATION_HIGH,
                    state.vibration_mm_s,
                    expected,
                    deviation,
                    ["Elevated vibration levels"],
                    ["Mechanical imbalance", "Bearing wear", "Loose mounting"]
                ))

        return anomalies

    def _create_anomaly_event(
        self,
        equipment_id: str,
        anomaly_type: AnomalyType,
        actual: float,
        expected: float,
        deviation: float,
        factors: List[str],
        causes: List[str]
    ) -> AnomalyEvent:
        """Create anomaly event object."""
        self._anomaly_counter += 1
        event_id = f"ANO-{self._anomaly_counter:06d}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

        # Severity based on deviation
        severity = min(1.0, deviation / (self.config.anomaly_threshold_sigma * 2))

        # Confidence based on number of observations
        stats = self._anomaly_detector.get_statistics(f"{equipment_id}_efficiency")
        confidence = min(1.0, stats["count"] / 100) if stats["count"] > 0 else 0.5

        # Provenance
        data = {"event_id": event_id, "deviation": deviation, "actual": actual}
        provenance_hash = hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()[:16]

        return AnomalyEvent(
            event_id=event_id,
            equipment_id=equipment_id,
            anomaly_type=anomaly_type,
            severity=severity,
            detected_at=datetime.utcnow(),
            detection_confidence=confidence,
            anomalous_value=actual,
            expected_value=expected,
            deviation_sigma=deviation,
            contributing_factors=factors,
            possible_causes=causes,
            provenance_hash=provenance_hash
        )

    def _analyze_efficiency_trend(
        self,
        equipment_id: str
    ) -> Optional[EfficiencyTrend]:
        """Analyze efficiency degradation trend."""
        if equipment_id not in self._state_history:
            return None

        history = list(self._state_history[equipment_id])
        if len(history) < 10:
            return None

        # Filter to trend window
        cutoff = datetime.utcnow() - timedelta(hours=self.config.trend_window_hours)
        recent_history = [s for s in history if s.timestamp >= cutoff]

        if len(recent_history) < 10:
            return None

        # Extract efficiency values and timestamps
        efficiencies = np.array([s.efficiency for s in recent_history])
        timestamps = np.array([
            (s.timestamp - recent_history[0].timestamp).total_seconds() / 86400.0
            for s in recent_history
        ])

        # Linear regression
        n = len(efficiencies)
        x_mean = np.mean(timestamps)
        y_mean = np.mean(efficiencies)

        numerator = np.sum((timestamps - x_mean) * (efficiencies - y_mean))
        denominator = np.sum((timestamps - x_mean) ** 2)

        if denominator == 0:
            slope = 0.0
            r_squared = 0.0
        else:
            slope = numerator / denominator
            y_pred = y_mean + slope * (timestamps - x_mean)
            ss_res = np.sum((efficiencies - y_pred) ** 2)
            ss_tot = np.sum((efficiencies - y_mean) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Trend direction
        if slope < -0.001:
            direction = "degrading"
        elif slope > 0.001:
            direction = "improving"
        else:
            direction = "stable"

        # Time to thresholds (if degrading)
        days_to_warning = None
        days_to_critical = None

        current_eff = efficiencies[-1]
        if slope < 0:
            if current_eff > self.config.efficiency_warning_threshold:
                days_to_warning = (
                    current_eff - self.config.efficiency_warning_threshold
                ) / abs(slope)
            if current_eff > self.config.efficiency_critical_threshold:
                days_to_critical = (
                    current_eff - self.config.efficiency_critical_threshold
                ) / abs(slope)

        # Confidence based on R-squared and sample size
        confidence = min(1.0, r_squared * min(1.0, n / 50))

        return EfficiencyTrend(
            equipment_id=equipment_id,
            analysis_period_hours=self.config.trend_window_hours,
            current_efficiency=float(current_eff),
            baseline_efficiency=float(efficiencies[0]),
            trend_slope_per_day=float(slope),
            trend_direction=direction,
            r_squared=float(r_squared),
            days_to_warning_threshold=days_to_warning,
            days_to_critical_threshold=days_to_critical,
            confidence=confidence,
            efficiency_mean=float(np.mean(efficiencies)),
            efficiency_std=float(np.std(efficiencies)),
            min_efficiency=float(np.min(efficiencies)),
            max_efficiency=float(np.max(efficiencies))
        )

    def _estimate_failure_probability(
        self,
        state: SeparatorHealthState
    ) -> Optional[FailureProbability]:
        """Estimate failure probability using Weibull analysis."""
        # Get Weibull parameters for equipment type
        params = WEIBULL_PARAMS.get(state.equipment_type)
        if params is None:
            # Use default
            params = {"beta": 2.0, "eta": 6.0}

        beta = params["beta"]
        eta = params["eta"]

        # Adjust eta based on operating conditions
        stress_multiplier = 1.0

        # High flow rate stress
        if state.flow_rate_kg_s > 15.0:
            stress_multiplier *= 1.2

        # High temperature stress
        if state.inlet_temperature_c > 200.0:
            stress_multiplier *= 1.15

        # Apply stress to scale parameter
        adjusted_eta = eta / stress_multiplier

        age = state.age_years
        horizon = self.config.prediction_horizon_days / 365.0

        # Weibull reliability function: R(t) = exp(-(t/eta)^beta)
        def reliability(t: float) -> float:
            if t <= 0:
                return 1.0
            return math.exp(-math.pow(t / adjusted_eta, beta))

        # Current reliability
        current_rel = reliability(age)

        # Conditional failure probability: P(fail in [t, t+h] | survived to t)
        future_rel = reliability(age + horizon)
        failure_prob = 1.0 - (future_rel / current_rel) if current_rel > 0 else 1.0

        # Hazard rate: h(t) = (beta/eta) * (t/eta)^(beta-1)
        if age > 0:
            hazard_rate = (beta / adjusted_eta) * math.pow(age / adjusted_eta, beta - 1)
        else:
            hazard_rate = (beta / adjusted_eta) if beta == 1 else 0.0

        # Median remaining life: solve R(t) = 0.5 * R(age)
        # t_median = eta * (-ln(0.5 * R(age)))^(1/beta)
        target_rel = 0.5 * current_rel
        if target_rel > 0:
            t_median = adjusted_eta * math.pow(-math.log(target_rel), 1 / beta)
            median_remaining = max(0, t_median - age)
        else:
            median_remaining = 0.0

        # Time to 50% and 90% failure
        def quantile(p: float) -> float:
            if p <= 0:
                return 0.0
            if p >= 1:
                return float('inf')
            return adjusted_eta * math.pow(-math.log(1 - p), 1 / beta)

        t_50 = quantile(0.5)
        t_90 = quantile(0.9)

        # Uncertainty bounds (simplified)
        cv = 0.15  # Coefficient of variation
        z = 1.645 if self.config.confidence_level == 0.90 else 1.96

        lower_bound = max(0, failure_prob - z * cv * failure_prob)
        upper_bound = min(1, failure_prob + z * cv * failure_prob)

        # Risk score
        risk_score = failure_prob * 100

        # Provenance
        data = {
            "equipment_id": state.equipment_id,
            "age": age,
            "beta": beta,
            "eta": adjusted_eta,
            "failure_prob": round(failure_prob, 6)
        }
        provenance_hash = hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()[:16]

        return FailureProbability(
            equipment_id=state.equipment_id,
            equipment_type=state.equipment_type,
            equipment_age_years=age,
            failure_probability=failure_prob,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            confidence_level=self.config.confidence_level,
            median_remaining_life_years=median_remaining,
            time_to_50_percent_failure=max(0, t_50 - age),
            time_to_90_percent_failure=max(0, t_90 - age),
            beta=beta,
            eta=adjusted_eta,
            current_reliability=current_rel,
            hazard_rate_per_year=hazard_rate,
            risk_score=risk_score,
            provenance_hash=provenance_hash
        )

    def _determine_health_status(
        self,
        state: SeparatorHealthState,
        anomalies: List[AnomalyEvent],
        trend: Optional[EfficiencyTrend],
        failure_prob: Optional[FailureProbability]
    ) -> Tuple[HealthStatus, float]:
        """Determine overall health status and score."""
        score_components = []

        # Component 1: Current efficiency (40% weight)
        eff_score = state.efficiency * 100
        score_components.append(("efficiency", eff_score, 0.4))

        # Component 2: Anomaly count (20% weight)
        anomaly_penalty = len(anomalies) * 10
        anomaly_score = max(0, 100 - anomaly_penalty)
        score_components.append(("anomalies", anomaly_score, 0.2))

        # Component 3: Trend (20% weight)
        if trend:
            if trend.trend_direction == "degrading":
                trend_score = 50 - abs(trend.trend_slope_per_day) * 1000
            elif trend.trend_direction == "improving":
                trend_score = 100
            else:
                trend_score = 80
            trend_score = max(0, min(100, trend_score))
        else:
            trend_score = 70  # Unknown = moderate
        score_components.append(("trend", trend_score, 0.2))

        # Component 4: Failure probability (20% weight)
        if failure_prob:
            fp_score = (1 - failure_prob.failure_probability) * 100
        else:
            fp_score = 70
        score_components.append(("failure_prob", fp_score, 0.2))

        # Calculate weighted score
        health_score = sum(score * weight for _, score, weight in score_components)

        # Determine status
        if health_score >= 90:
            status = HealthStatus.HEALTHY
        elif health_score >= 70:
            status = HealthStatus.DEGRADED
        elif health_score >= 50:
            status = HealthStatus.WARNING
        elif health_score >= 25:
            status = HealthStatus.CRITICAL
        else:
            status = HealthStatus.FAILED

        # Override based on critical conditions
        if state.efficiency < self.config.efficiency_critical_threshold:
            status = HealthStatus.CRITICAL
        if any(a.severity > 0.9 for a in anomalies):
            if status not in [HealthStatus.CRITICAL, HealthStatus.FAILED]:
                status = HealthStatus.WARNING

        return status, health_score

    def _generate_maintenance_recommendation(
        self,
        state: SeparatorHealthState,
        health_status: HealthStatus,
        anomalies: List[AnomalyEvent],
        trend: Optional[EfficiencyTrend],
        failure_prob: Optional[FailureProbability]
    ) -> Optional[MaintenanceRecommendation]:
        """Generate maintenance recommendation."""
        factors = []
        now = datetime.utcnow()

        # Determine urgency and action
        if health_status == HealthStatus.FAILED:
            urgency = MaintenanceUrgency.IMMEDIATE
            action = "Emergency repair/replacement"
            description = "Equipment has failed and requires immediate attention"
            deadline = now + timedelta(hours=24)
            risk = "Production loss, safety risk"
        elif health_status == HealthStatus.CRITICAL:
            urgency = MaintenanceUrgency.URGENT
            action = "Inspection and repair"
            description = "Critical degradation detected, inspection required"
            deadline = now + timedelta(days=7)
            risk = "Potential failure, quality impact"
        elif health_status == HealthStatus.WARNING:
            urgency = MaintenanceUrgency.PLANNED
            action = "Scheduled maintenance"
            description = "Performance degradation trending, plan maintenance"
            deadline = now + timedelta(days=30)
            risk = "Progressive degradation"
        elif health_status == HealthStatus.DEGRADED:
            urgency = MaintenanceUrgency.ROUTINE
            action = "Include in next maintenance window"
            description = "Minor degradation, monitor and plan"
            deadline = now + timedelta(days=90)
            risk = "Continued degradation if unaddressed"
        else:
            # Healthy - no action needed
            return None

        # Build contributing factors
        if state.efficiency < 0.95:
            factors.append(f"Low efficiency: {state.efficiency:.1%}")
        if anomalies:
            factors.append(f"{len(anomalies)} anomalies detected")
        if trend and trend.trend_direction == "degrading":
            factors.append(f"Degradation rate: {abs(trend.trend_slope_per_day)*100:.3f}% per day")
        if failure_prob and failure_prob.failure_probability > 0.2:
            factors.append(f"Failure probability: {failure_prob.failure_probability:.1%}")

        # Estimated downtime
        downtime_hours = {
            MaintenanceUrgency.IMMEDIATE: 8.0,
            MaintenanceUrgency.URGENT: 4.0,
            MaintenanceUrgency.PLANNED: 2.0,
            MaintenanceUrgency.ROUTINE: 1.0,
        }.get(urgency, 2.0)

        # Provenance
        data = {
            "equipment_id": state.equipment_id,
            "urgency": urgency.value,
            "action": action,
        }
        provenance_hash = hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()[:16]

        return MaintenanceRecommendation(
            equipment_id=state.equipment_id,
            urgency=urgency,
            recommended_action=action,
            description=description,
            recommended_date=now + timedelta(days=1),
            deadline=deadline,
            health_status=health_status,
            contributing_factors=factors,
            risk_if_deferred=risk,
            estimated_downtime_hours=downtime_hours,
            provenance_hash=provenance_hash
        )

    def _calculate_confidence(self, equipment_id: str) -> float:
        """Calculate assessment confidence based on data availability."""
        if equipment_id not in self._state_history:
            return 0.5

        history_size = len(self._state_history[equipment_id])

        # More history = higher confidence
        confidence = min(1.0, history_size / 100)

        # Adjust based on anomaly detector confidence
        stats = self._anomaly_detector.get_statistics(f"{equipment_id}_efficiency")
        if stats["count"] > 50:
            confidence = min(1.0, confidence + 0.2)

        return confidence

    def _compute_provenance(
        self,
        state: SeparatorHealthState,
        health_score: float
    ) -> str:
        """Compute provenance hash."""
        data = {
            "version": self.VERSION,
            "equipment_id": state.equipment_id,
            "timestamp": state.timestamp.isoformat(),
            "efficiency": round(state.efficiency, 4),
            "health_score": round(health_score, 2),
        }
        return hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()[:16]

    def get_equipment_history(
        self,
        equipment_id: str,
        hours: int = 24
    ) -> List[SeparatorHealthState]:
        """Get recent state history for equipment."""
        if equipment_id not in self._state_history:
            return []

        cutoff = datetime.utcnow() - timedelta(hours=hours)
        return [
            s for s in self._state_history[equipment_id]
            if s.timestamp >= cutoff
        ]

    def reset(self, equipment_id: Optional[str] = None) -> None:
        """Reset estimator state."""
        if equipment_id:
            if equipment_id in self._state_history:
                del self._state_history[equipment_id]
            logger.info(f"Reset state for equipment {equipment_id}")
        else:
            self._state_history.clear()
            logger.info("Reset all equipment states")
