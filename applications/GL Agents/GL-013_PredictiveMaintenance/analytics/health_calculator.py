# -*- coding: utf-8 -*-
"""
Health Index Calculator for GL-013 PredictiveMaintenance Agent.

Provides composite health index calculation for industrial assets:
- Multi-factor weighted scoring
- Degradation trend analysis
- Health grade assignment
- Comparison with peer assets

Author: GreenLang AI Team
Version: 1.0.0
"""

import logging
import hashlib
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class HealthGrade(str, Enum):
    """Health grades (A to F)."""
    A = "A"  # Excellent: 90-100%
    B = "B"  # Good: 75-90%
    C = "C"  # Fair: 60-75%
    D = "D"  # Poor: 40-60%
    F = "F"  # Critical: 0-40%


class TrendDirection(str, Enum):
    """Direction of health trend."""
    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"
    RAPIDLY_DEGRADING = "rapidly_degrading"


@dataclass
class HealthCalculatorConfig:
    """Configuration for health calculation."""
    # Factor weights (must sum to 1.0)
    weight_vibration: float = 0.25
    weight_temperature: float = 0.20
    weight_operating_hours: float = 0.15
    weight_maintenance_history: float = 0.15
    weight_failure_risk: float = 0.15
    weight_anomaly_score: float = 0.10

    # Thresholds
    grade_a_threshold: float = 0.90
    grade_b_threshold: float = 0.75
    grade_c_threshold: float = 0.60
    grade_d_threshold: float = 0.40

    # Trend analysis
    trend_window_days: int = 30
    degradation_rate_critical: float = 0.02  # 2% per day


@dataclass
class HealthResult:
    """Result of health index calculation."""
    index_id: str
    asset_id: str
    timestamp: datetime

    # Overall health
    overall_health: float  # 0-1 scale
    health_grade: HealthGrade

    # Component scores
    component_scores: Dict[str, float] = field(default_factory=dict)
    component_weights: Dict[str, float] = field(default_factory=dict)

    # Trend analysis
    trend_direction: TrendDirection = TrendDirection.STABLE
    trend_slope: float = 0.0  # Change per day
    days_to_grade_change: Optional[float] = None

    # Benchmarking
    percentile_rank: Optional[float] = None
    peer_group_average: Optional[float] = None

    # Recommendations
    limiting_factor: str = ""
    improvement_potential: float = 0.0

    # Provenance
    provenance_hash: str = ""
    computation_time_ms: float = 0.0


@dataclass
class DegradationModel:
    """Degradation model parameters."""
    model_type: str = "linear"  # linear, exponential, logarithmic
    initial_health: float = 1.0
    current_health: float = 1.0
    degradation_rate: float = 0.0
    time_elapsed_days: float = 0.0

    # Model coefficients
    slope: float = 0.0
    intercept: float = 1.0
    r_squared: float = 0.0

    # Projections
    time_to_threshold_days: Optional[float] = None
    projected_health_30d: float = 0.0
    projected_health_90d: float = 0.0


class HealthCalculator:
    """
    Composite health index calculator for industrial assets.

    Combines multiple health factors:
    - Vibration health (normalized vibration levels)
    - Thermal health (temperature vs limits)
    - Age/usage health (operating hours vs expected life)
    - Maintenance health (maintenance compliance)
    - Risk health (failure probability inverse)
    - Anomaly health (anomaly score inverse)
    """

    def __init__(self, config: Optional[HealthCalculatorConfig] = None):
        """
        Initialize health calculator.

        Args:
            config: Configuration for health calculation
        """
        self.config = config or HealthCalculatorConfig()
        self._validate_weights()
        self._history: Dict[str, List[Tuple[datetime, float]]] = {}

    def _validate_weights(self):
        """Validate that weights sum to 1.0."""
        total = (
            self.config.weight_vibration +
            self.config.weight_temperature +
            self.config.weight_operating_hours +
            self.config.weight_maintenance_history +
            self.config.weight_failure_risk +
            self.config.weight_anomaly_score
        )
        if abs(total - 1.0) > 0.01:
            logger.warning(f"Health factor weights sum to {total:.3f}, normalizing")

    def calculate(
        self,
        asset_id: str,
        vibration_health: float = 1.0,
        temperature_health: float = 1.0,
        operating_hours_health: float = 1.0,
        maintenance_health: float = 1.0,
        failure_risk: float = 0.0,  # 0-1, will be inverted
        anomaly_score: float = 0.0,  # 0-1, will be inverted
    ) -> HealthResult:
        """
        Calculate composite health index.

        Args:
            asset_id: Asset identifier
            vibration_health: Vibration health score (0-1, 1 = healthy)
            temperature_health: Temperature health score (0-1)
            operating_hours_health: Operating hours health (0-1)
            maintenance_health: Maintenance compliance (0-1)
            failure_risk: Failure probability (0-1, will be inverted)
            anomaly_score: Anomaly score (0-1, will be inverted)

        Returns:
            HealthResult with comprehensive health assessment
        """
        start_time = time.time()

        # Clip inputs to valid range
        vibration_health = np.clip(vibration_health, 0, 1)
        temperature_health = np.clip(temperature_health, 0, 1)
        operating_hours_health = np.clip(operating_hours_health, 0, 1)
        maintenance_health = np.clip(maintenance_health, 0, 1)
        failure_risk = np.clip(failure_risk, 0, 1)
        anomaly_score = np.clip(anomaly_score, 0, 1)

        # Convert risk/anomaly to health scores (invert)
        risk_health = 1.0 - failure_risk
        anomaly_health = 1.0 - anomaly_score

        # Component scores
        component_scores = {
            "vibration": float(vibration_health),
            "temperature": float(temperature_health),
            "operating_hours": float(operating_hours_health),
            "maintenance": float(maintenance_health),
            "failure_risk": float(risk_health),
            "anomaly": float(anomaly_health),
        }

        # Weights
        component_weights = {
            "vibration": self.config.weight_vibration,
            "temperature": self.config.weight_temperature,
            "operating_hours": self.config.weight_operating_hours,
            "maintenance": self.config.weight_maintenance_history,
            "failure_risk": self.config.weight_failure_risk,
            "anomaly": self.config.weight_anomaly_score,
        }

        # Calculate weighted average
        overall_health = sum(
            component_scores[k] * component_weights[k]
            for k in component_scores
        )

        # Normalize if weights don't sum to 1
        total_weight = sum(component_weights.values())
        if total_weight > 0:
            overall_health /= total_weight

        # Determine grade
        health_grade = self._determine_grade(overall_health)

        # Find limiting factor
        limiting_factor = min(component_scores, key=component_scores.get)
        limiting_score = component_scores[limiting_factor]

        # Calculate improvement potential
        improvement_potential = 1.0 - overall_health

        # Update history and calculate trend
        timestamp = datetime.utcnow()
        self._update_history(asset_id, timestamp, overall_health)
        trend_direction, trend_slope = self._calculate_trend(asset_id)

        # Calculate days to grade change
        days_to_grade_change = self._calculate_days_to_grade_change(
            overall_health, health_grade, trend_slope
        )

        computation_time = (time.time() - start_time) * 1000

        # Generate provenance hash
        provenance_hash = hashlib.sha256(
            f"{asset_id}{overall_health:.6f}{timestamp}".encode()
        ).hexdigest()

        return HealthResult(
            index_id=f"health_{hashlib.sha256(f'{asset_id}{timestamp}'.encode()).hexdigest()[:12]}",
            asset_id=asset_id,
            timestamp=timestamp,
            overall_health=float(overall_health),
            health_grade=health_grade,
            component_scores=component_scores,
            component_weights=component_weights,
            trend_direction=trend_direction,
            trend_slope=float(trend_slope),
            days_to_grade_change=days_to_grade_change,
            limiting_factor=limiting_factor,
            improvement_potential=float(improvement_potential),
            provenance_hash=provenance_hash,
            computation_time_ms=computation_time,
        )

    def calculate_from_features(
        self,
        asset_id: str,
        features: Dict[str, float],
        limits: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> HealthResult:
        """
        Calculate health from raw feature values.

        Args:
            asset_id: Asset identifier
            features: Dictionary of feature values
            limits: Dictionary of limits for each feature
                    {"feature_name": {"min": 0, "max": 100, "warning": 80}}

        Returns:
            HealthResult
        """
        limits = limits or {}

        # Default feature-to-health mappings
        vibration_features = ["vibration_rms", "vibration_peak", "velocity_mm_s"]
        temperature_features = ["temperature", "bearing_temp", "motor_temp"]

        # Calculate vibration health
        vibration_health = 1.0
        for feat in vibration_features:
            if feat in features and feat in limits:
                value = features[feat]
                limit = limits[feat]
                health = self._value_to_health(
                    value,
                    limit.get("max", 100),
                    limit.get("warning", limit.get("max", 100) * 0.8)
                )
                vibration_health = min(vibration_health, health)

        # Calculate temperature health
        temperature_health = 1.0
        for feat in temperature_features:
            if feat in features and feat in limits:
                value = features[feat]
                limit = limits[feat]
                health = self._value_to_health(
                    value,
                    limit.get("max", 100),
                    limit.get("warning", limit.get("max", 100) * 0.9)
                )
                temperature_health = min(temperature_health, health)

        # Operating hours health
        operating_hours_health = 1.0
        if "operating_hours" in features and "expected_life_hours" in features:
            usage_ratio = features["operating_hours"] / max(features["expected_life_hours"], 1)
            operating_hours_health = max(0, 1 - usage_ratio)

        # Maintenance health (from maintenance_score or compliance)
        maintenance_health = features.get("maintenance_score", 1.0)

        # Failure risk (if provided)
        failure_risk = features.get("failure_probability", 0.0)

        # Anomaly score (if provided)
        anomaly_score = features.get("anomaly_score", 0.0)

        return self.calculate(
            asset_id=asset_id,
            vibration_health=vibration_health,
            temperature_health=temperature_health,
            operating_hours_health=operating_hours_health,
            maintenance_health=maintenance_health,
            failure_risk=failure_risk,
            anomaly_score=anomaly_score,
        )

    def _value_to_health(
        self,
        value: float,
        max_limit: float,
        warning_limit: float,
    ) -> float:
        """
        Convert a measurement value to health score.

        Health is 1.0 when value is 0, decreases linearly to warning,
        then more rapidly to max limit where health is 0.
        """
        if value <= 0:
            return 1.0
        elif value <= warning_limit:
            # Linear decrease to 0.6 at warning
            return 1.0 - 0.4 * (value / warning_limit)
        elif value <= max_limit:
            # Steeper decrease from 0.6 to 0 between warning and max
            progress = (value - warning_limit) / (max_limit - warning_limit)
            return 0.6 * (1 - progress)
        else:
            return 0.0

    def _determine_grade(self, health: float) -> HealthGrade:
        """Determine health grade from score."""
        if health >= self.config.grade_a_threshold:
            return HealthGrade.A
        elif health >= self.config.grade_b_threshold:
            return HealthGrade.B
        elif health >= self.config.grade_c_threshold:
            return HealthGrade.C
        elif health >= self.config.grade_d_threshold:
            return HealthGrade.D
        else:
            return HealthGrade.F

    def _update_history(
        self,
        asset_id: str,
        timestamp: datetime,
        health: float,
    ) -> None:
        """Update health history for an asset."""
        if asset_id not in self._history:
            self._history[asset_id] = []

        self._history[asset_id].append((timestamp, health))

        # Keep only recent history (based on trend window)
        cutoff = timestamp - timedelta(days=self.config.trend_window_days * 2)
        self._history[asset_id] = [
            (t, h) for t, h in self._history[asset_id]
            if t >= cutoff
        ]

    def _calculate_trend(
        self,
        asset_id: str,
    ) -> Tuple[TrendDirection, float]:
        """Calculate health trend from history."""
        if asset_id not in self._history or len(self._history[asset_id]) < 2:
            return TrendDirection.STABLE, 0.0

        history = self._history[asset_id]

        # Filter to trend window
        now = datetime.utcnow()
        cutoff = now - timedelta(days=self.config.trend_window_days)
        recent = [(t, h) for t, h in history if t >= cutoff]

        if len(recent) < 2:
            return TrendDirection.STABLE, 0.0

        # Convert to arrays for linear regression
        times = np.array([(t - recent[0][0]).total_seconds() / 86400 for t, h in recent])
        healths = np.array([h for t, h in recent])

        # Linear regression
        n = len(times)
        if n > 1:
            slope = (n * np.sum(times * healths) - np.sum(times) * np.sum(healths)) / \
                    (n * np.sum(times ** 2) - np.sum(times) ** 2 + 1e-10)
        else:
            slope = 0.0

        # Determine trend direction
        if slope > 0.005:
            direction = TrendDirection.IMPROVING
        elif slope < -self.config.degradation_rate_critical:
            direction = TrendDirection.RAPIDLY_DEGRADING
        elif slope < -0.002:
            direction = TrendDirection.DEGRADING
        else:
            direction = TrendDirection.STABLE

        return direction, float(slope)

    def _calculate_days_to_grade_change(
        self,
        current_health: float,
        current_grade: HealthGrade,
        slope: float,
    ) -> Optional[float]:
        """Calculate days until grade changes."""
        if abs(slope) < 1e-6:
            return None  # No change expected

        # Find next grade threshold
        thresholds = [
            (HealthGrade.A, self.config.grade_a_threshold),
            (HealthGrade.B, self.config.grade_b_threshold),
            (HealthGrade.C, self.config.grade_c_threshold),
            (HealthGrade.D, self.config.grade_d_threshold),
            (HealthGrade.F, 0.0),
        ]

        if slope > 0:
            # Improving - find next higher threshold
            for grade, threshold in thresholds:
                if threshold > current_health:
                    days = (threshold - current_health) / slope
                    if days > 0:
                        return float(days)
        else:
            # Degrading - find next lower threshold
            for grade, threshold in reversed(thresholds):
                if threshold < current_health:
                    days = (current_health - threshold) / abs(slope)
                    if days > 0:
                        return float(days)

        return None

    def get_degradation_model(
        self,
        asset_id: str,
        model_type: str = "linear",
    ) -> DegradationModel:
        """
        Get degradation model for an asset.

        Args:
            asset_id: Asset identifier
            model_type: Type of model (linear, exponential)

        Returns:
            DegradationModel with parameters
        """
        if asset_id not in self._history or len(self._history[asset_id]) < 2:
            return DegradationModel()

        history = self._history[asset_id]

        # Convert to arrays
        times = np.array([(t - history[0][0]).total_seconds() / 86400 for t, h in history])
        healths = np.array([h for t, h in history])

        # Fit linear model
        n = len(times)
        slope = (n * np.sum(times * healths) - np.sum(times) * np.sum(healths)) / \
                (n * np.sum(times ** 2) - np.sum(times) ** 2 + 1e-10)
        intercept = (np.sum(healths) - slope * np.sum(times)) / n

        # Calculate R-squared
        y_pred = slope * times + intercept
        ss_res = np.sum((healths - y_pred) ** 2)
        ss_tot = np.sum((healths - np.mean(healths)) ** 2)
        r_squared = 1 - (ss_res / (ss_tot + 1e-10))

        # Projections
        current_time = times[-1]
        projected_30d = slope * (current_time + 30) + intercept
        projected_90d = slope * (current_time + 90) + intercept

        # Time to threshold (0.4 = grade D boundary)
        threshold = 0.4
        if slope < 0:
            time_to_threshold = (threshold - healths[-1]) / slope
        else:
            time_to_threshold = None

        return DegradationModel(
            model_type=model_type,
            initial_health=float(healths[0]),
            current_health=float(healths[-1]),
            degradation_rate=float(-slope),  # Positive = degrading
            time_elapsed_days=float(times[-1]),
            slope=float(slope),
            intercept=float(intercept),
            r_squared=float(max(0, r_squared)),
            time_to_threshold_days=float(time_to_threshold) if time_to_threshold and time_to_threshold > 0 else None,
            projected_health_30d=float(np.clip(projected_30d, 0, 1)),
            projected_health_90d=float(np.clip(projected_90d, 0, 1)),
        )

    def get_peer_comparison(
        self,
        asset_id: str,
        peer_healths: Dict[str, float],
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Compare asset health with peers.

        Args:
            asset_id: Asset to compare
            peer_healths: Dictionary of peer asset IDs to health scores

        Returns:
            Tuple of (percentile_rank, peer_average)
        """
        if asset_id not in self._history or not self._history[asset_id]:
            return None, None

        current_health = self._history[asset_id][-1][1]

        if not peer_healths:
            return None, None

        peer_values = list(peer_healths.values())
        peer_average = float(np.mean(peer_values))

        # Calculate percentile rank
        n_below = sum(1 for h in peer_values if h < current_health)
        percentile_rank = (n_below / len(peer_values)) * 100

        return float(percentile_rank), peer_average
