"""
PredictiveMaintenanceModel - Equipment Failure Prediction for GL-004 BURNMASTER

This module implements predictive maintenance for combustion equipment.
Uses survival analysis and gradient boosting to predict time-to-failure
and maintenance windows.

Key Features:
    - Time-to-failure prediction
    - Failure mode classification
    - Maintenance window optimization
    - Remaining useful life (RUL) estimation
    - Physics-informed feature engineering
    - Uncertainty quantification

CRITICAL: Predictions are ADVISORY ONLY.
Maintenance decisions should be confirmed by operations personnel.

Example:
    >>> model = PredictiveMaintenanceModel()
    >>> features = MaintenanceFeatures(
    ...     operating_hours=5000,
    ...     start_stop_cycles=150,
    ...     max_flame_temp_c=1650
    ... )
    >>> prediction = model.predict(features)
    >>> print(f"RUL: {prediction.remaining_useful_life_hours} hours")

Author: GreenLang ML Engineering Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# Optional imports with graceful degradation
try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available, using physics-based fallback only")


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================


class FailureMode(str, Enum):
    """Equipment failure modes for combustion systems."""
    NONE = "none"  # No failure predicted
    BURNER_TIP_DEGRADATION = "burner_tip_degradation"
    REFRACTORY_DAMAGE = "refractory_damage"
    IGNITER_FAILURE = "igniter_failure"
    FLAME_SCANNER_DRIFT = "flame_scanner_drift"
    FUEL_VALVE_WEAR = "fuel_valve_wear"
    AIR_DAMPER_MALFUNCTION = "air_damper_malfunction"
    HEAT_EXCHANGER_FOULING = "heat_exchanger_fouling"
    BURNER_MANAGEMENT_SYSTEM = "bms_failure"
    UNKNOWN = "unknown"


class MaintenancePriority(str, Enum):
    """Maintenance priority levels."""
    CRITICAL = "critical"  # Immediate attention required
    HIGH = "high"  # Schedule within 1 week
    MEDIUM = "medium"  # Schedule within 1 month
    LOW = "low"  # Schedule at next planned outage
    ROUTINE = "routine"  # Normal preventive maintenance


class HealthStatus(str, Enum):
    """Equipment health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


# Failure mode typical progression rates (per 1000 hours)
FAILURE_PROGRESSION_RATES = {
    FailureMode.BURNER_TIP_DEGRADATION: 0.02,
    FailureMode.REFRACTORY_DAMAGE: 0.005,
    FailureMode.IGNITER_FAILURE: 0.03,
    FailureMode.FLAME_SCANNER_DRIFT: 0.025,
    FailureMode.FUEL_VALVE_WEAR: 0.015,
    FailureMode.AIR_DAMPER_MALFUNCTION: 0.01,
    FailureMode.HEAT_EXCHANGER_FOULING: 0.008,
    FailureMode.BURNER_MANAGEMENT_SYSTEM: 0.001,
}

# Mean time between failures (MTBF) in hours for each failure mode
MTBF_HOURS = {
    FailureMode.BURNER_TIP_DEGRADATION: 20000,
    FailureMode.REFRACTORY_DAMAGE: 50000,
    FailureMode.IGNITER_FAILURE: 15000,
    FailureMode.FLAME_SCANNER_DRIFT: 18000,
    FailureMode.FUEL_VALVE_WEAR: 25000,
    FailureMode.AIR_DAMPER_MALFUNCTION: 35000,
    FailureMode.HEAT_EXCHANGER_FOULING: 40000,
    FailureMode.BURNER_MANAGEMENT_SYSTEM: 100000,
}


# =============================================================================
# PYDANTIC SCHEMAS
# =============================================================================


class MaintenanceFeatures(BaseModel):
    """Input features for maintenance prediction."""

    # Operating time features
    operating_hours: float = Field(
        default=0.0, ge=0.0,
        description="Total operating hours since last overhaul"
    )
    hours_since_last_maintenance: float = Field(
        default=0.0, ge=0.0,
        description="Hours since last maintenance"
    )
    start_stop_cycles: int = Field(
        default=0, ge=0,
        description="Number of start/stop cycles"
    )

    # Temperature stress features
    max_flame_temp_c: float = Field(
        default=1500.0, ge=0.0, le=2500.0,
        description="Maximum flame temperature observed"
    )
    avg_flame_temp_c: float = Field(
        default=1400.0, ge=0.0, le=2500.0,
        description="Average flame temperature"
    )
    temp_excursion_count: int = Field(
        default=0, ge=0,
        description="Number of temperature excursions above limit"
    )
    thermal_cycles: int = Field(
        default=0, ge=0,
        description="Number of significant thermal cycles"
    )

    # Mechanical stress features
    vibration_rms: float = Field(
        default=0.0, ge=0.0,
        description="RMS vibration level"
    )
    pressure_oscillation_max: float = Field(
        default=0.0, ge=0.0,
        description="Maximum pressure oscillation observed"
    )

    # Performance degradation features
    efficiency_trend: float = Field(
        default=0.0,
        description="Efficiency trend (negative = degradation)"
    )
    co_baseline_drift: float = Field(
        default=0.0, ge=0.0,
        description="CO baseline drift from initial value"
    )
    o2_sensor_drift: float = Field(
        default=0.0,
        description="O2 sensor drift from calibration"
    )

    # Flame quality features
    flame_stability_avg: float = Field(
        default=0.9, ge=0.0, le=1.0,
        description="Average flame stability index"
    )
    flame_scanner_signal_trend: float = Field(
        default=0.0,
        description="Flame scanner signal trend"
    )

    # Environmental factors
    ambient_temp_avg_c: float = Field(
        default=25.0,
        description="Average ambient temperature"
    )
    humidity_avg_percent: float = Field(
        default=50.0, ge=0.0, le=100.0,
        description="Average humidity"
    )

    # Equipment identifiers
    burner_id: str = Field(default="BNR-001", description="Burner ID")
    equipment_age_years: float = Field(
        default=0.0, ge=0.0,
        description="Equipment age in years"
    )

    @field_validator("operating_hours", "hours_since_last_maintenance")
    @classmethod
    def validate_hours(cls, v: float) -> float:
        """Ensure hours are reasonable."""
        if v > 1000000:
            raise ValueError("Hours value seems unreasonably high")
        return v


class MaintenancePrediction(BaseModel):
    """Prediction result for maintenance model."""

    prediction_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique prediction identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Prediction timestamp"
    )

    # Core predictions
    failure_probability: float = Field(
        ..., ge=0.0, le=1.0,
        description="Probability of failure in prediction window"
    )
    remaining_useful_life_hours: float = Field(
        ..., ge=0.0,
        description="Estimated remaining useful life"
    )
    predicted_failure_mode: FailureMode = Field(
        ..., description="Most likely failure mode"
    )

    # Risk assessment
    health_status: HealthStatus = Field(
        ..., description="Overall equipment health status"
    )
    maintenance_priority: MaintenancePriority = Field(
        ..., description="Recommended maintenance priority"
    )
    risk_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Overall risk score"
    )

    # Uncertainty quantification
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Prediction confidence"
    )
    rul_lower_bound: float = Field(
        default=0.0, ge=0.0,
        description="RUL lower bound (95% CI)"
    )
    rul_upper_bound: float = Field(
        default=0.0, ge=0.0,
        description="RUL upper bound (95% CI)"
    )

    # Failure mode probabilities
    failure_mode_probabilities: Dict[str, float] = Field(
        default_factory=dict,
        description="Probability for each failure mode"
    )

    # Contributing factors
    top_risk_factors: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Top contributing risk factors"
    )

    # Recommendations
    recommended_actions: List[str] = Field(
        default_factory=list,
        description="Recommended maintenance actions"
    )
    optimal_maintenance_window: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optimal maintenance timing"
    )

    # Provenance
    model_version: str = Field(default="1.0.0", description="Model version")
    is_physics_fallback: bool = Field(
        default=False,
        description="Whether physics fallback was used"
    )
    provenance_hash: str = Field(default="", description="SHA-256 audit hash")
    computation_time_ms: float = Field(default=0.0, ge=0.0)


# =============================================================================
# PREDICTIVE MAINTENANCE MODEL
# =============================================================================


class PredictiveMaintenanceModel:
    """
    Predictive maintenance model for combustion equipment.

    Uses a combination of ML (gradient boosting) and physics-based
    calculations to predict equipment failure probability and
    remaining useful life.

    CRITICAL: Predictions are ADVISORY ONLY.

    Attributes:
        is_fitted: Whether model has been trained
        model_id: Unique model identifier
        feature_names: Expected feature names

    Example:
        >>> model = PredictiveMaintenanceModel()
        >>> features = MaintenanceFeatures(operating_hours=5000)
        >>> prediction = model.predict(features)
        >>> print(f"Failure probability: {prediction.failure_probability:.1%}")
    """

    FEATURE_NAMES = [
        "operating_hours",
        "hours_since_last_maintenance",
        "start_stop_cycles",
        "max_flame_temp_c",
        "avg_flame_temp_c",
        "temp_excursion_count",
        "thermal_cycles",
        "vibration_rms",
        "pressure_oscillation_max",
        "efficiency_trend",
        "co_baseline_drift",
        "o2_sensor_drift",
        "flame_stability_avg",
        "flame_scanner_signal_trend",
        "equipment_age_years",
    ]

    def __init__(
        self,
        model_path: Optional[Path] = None,
        prediction_window_hours: float = 720,  # 30 days
        random_seed: int = 42
    ):
        """
        Initialize PredictiveMaintenanceModel.

        Args:
            model_path: Path to pre-trained model file
            prediction_window_hours: Time window for failure prediction
            random_seed: Random seed for reproducibility
        """
        self.prediction_window_hours = prediction_window_hours
        self.random_seed = random_seed
        self._model_id = f"maintenance_{uuid4().hex[:8]}"

        self._classifier: Optional[Any] = None  # Failure mode classifier
        self._regressor: Optional[Any] = None  # RUL regressor
        self._scaler: Optional[Any] = None
        self._is_fitted = False
        self._feature_importance: Dict[str, float] = {}

        if model_path and model_path.exists():
            self._load_model(model_path)
        elif SKLEARN_AVAILABLE:
            self._initialize_default_models()

        logger.info(
            f"PredictiveMaintenanceModel initialized: "
            f"id={self._model_id}, fitted={self._is_fitted}"
        )

    def _initialize_default_models(self) -> None:
        """Initialize default model architecture."""
        if not SKLEARN_AVAILABLE:
            return

        self._classifier = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=self.random_seed
        )

        self._regressor = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=self.random_seed
        )

        self._scaler = StandardScaler()

    def predict(self, features: MaintenanceFeatures) -> MaintenancePrediction:
        """
        Predict equipment failure and maintenance needs.

        Args:
            features: Equipment operating features

        Returns:
            Maintenance prediction with RUL and recommendations
        """
        start_time = time.time()

        # Extract feature vector
        feature_vector = self._extract_features(features)

        # Use ML if available, otherwise physics fallback
        if self._is_fitted and SKLEARN_AVAILABLE:
            prediction = self._predict_with_ml(features, feature_vector)
        else:
            prediction = self._predict_with_physics(features)

        # Compute provenance hash
        prediction.provenance_hash = self._compute_provenance_hash(
            features, prediction
        )
        prediction.computation_time_ms = (time.time() - start_time) * 1000

        logger.debug(
            f"Maintenance prediction: failure_prob={prediction.failure_probability:.3f}, "
            f"rul={prediction.remaining_useful_life_hours:.0f}h, "
            f"mode={prediction.predicted_failure_mode.value}"
        )

        return prediction

    def _predict_with_ml(
        self,
        features: MaintenanceFeatures,
        feature_vector: np.ndarray
    ) -> MaintenancePrediction:
        """Make prediction using trained ML model."""
        # Scale features
        if self._scaler and hasattr(self._scaler, "mean_"):
            feature_vector_scaled = self._scaler.transform(feature_vector.reshape(1, -1))
        else:
            feature_vector_scaled = feature_vector.reshape(1, -1)

        # Predict failure probability
        if self._classifier and hasattr(self._classifier, "predict_proba"):
            failure_proba = self._classifier.predict_proba(feature_vector_scaled)
            failure_prob = float(failure_proba[0, 1]) if failure_proba.shape[1] > 1 else 0.5
        else:
            failure_prob = self._calculate_physics_failure_prob(features)

        # Predict RUL
        if self._regressor:
            rul = float(self._regressor.predict(feature_vector_scaled)[0])
            rul = max(0, rul)  # RUL cannot be negative
        else:
            rul = self._calculate_physics_rul(features)

        # Confidence based on model certainty
        confidence = self._calculate_confidence(failure_prob)

        # Calculate failure mode probabilities
        failure_mode_probs = self._calculate_failure_mode_probabilities(features)
        predicted_mode = max(failure_mode_probs.items(), key=lambda x: x[1])[0]

        # Determine health status and priority
        health_status = self._determine_health_status(failure_prob, rul)
        priority = self._determine_priority(failure_prob, rul, health_status)

        # Risk factors
        risk_factors = self._identify_risk_factors(features)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            predicted_mode, failure_prob, rul, risk_factors
        )

        # Calculate uncertainty bounds
        rul_std = rul * 0.2  # 20% standard deviation estimate
        rul_lower = max(0, rul - 1.96 * rul_std)
        rul_upper = rul + 1.96 * rul_std

        return MaintenancePrediction(
            failure_probability=failure_prob,
            remaining_useful_life_hours=rul,
            predicted_failure_mode=FailureMode(predicted_mode),
            health_status=health_status,
            maintenance_priority=priority,
            risk_score=failure_prob,
            confidence=confidence,
            rul_lower_bound=rul_lower,
            rul_upper_bound=rul_upper,
            failure_mode_probabilities={k: round(v, 4) for k, v in failure_mode_probs.items()},
            top_risk_factors=risk_factors[:5],
            recommended_actions=recommendations,
            optimal_maintenance_window=self._calculate_maintenance_window(rul, priority),
            model_version="1.0.0",
            is_physics_fallback=False
        )

    def _predict_with_physics(
        self, features: MaintenanceFeatures
    ) -> MaintenancePrediction:
        """Make prediction using physics-based calculations (DETERMINISTIC)."""

        # Calculate failure probability using Weibull-like model
        failure_prob = self._calculate_physics_failure_prob(features)

        # Calculate RUL using degradation model
        rul = self._calculate_physics_rul(features)

        # Calculate failure mode probabilities
        failure_mode_probs = self._calculate_failure_mode_probabilities(features)
        predicted_mode = max(failure_mode_probs.items(), key=lambda x: x[1])[0]

        # Determine status and priority
        health_status = self._determine_health_status(failure_prob, rul)
        priority = self._determine_priority(failure_prob, rul, health_status)

        # Risk factors
        risk_factors = self._identify_risk_factors(features)

        # Recommendations
        recommendations = self._generate_recommendations(
            predicted_mode, failure_prob, rul, risk_factors
        )

        # Uncertainty (higher for physics model)
        confidence = 0.65
        rul_std = rul * 0.3

        return MaintenancePrediction(
            failure_probability=failure_prob,
            remaining_useful_life_hours=rul,
            predicted_failure_mode=FailureMode(predicted_mode),
            health_status=health_status,
            maintenance_priority=priority,
            risk_score=failure_prob,
            confidence=confidence,
            rul_lower_bound=max(0, rul - 1.96 * rul_std),
            rul_upper_bound=rul + 1.96 * rul_std,
            failure_mode_probabilities={k: round(v, 4) for k, v in failure_mode_probs.items()},
            top_risk_factors=risk_factors[:5],
            recommended_actions=recommendations,
            optimal_maintenance_window=self._calculate_maintenance_window(rul, priority),
            model_version="1.0.0",
            is_physics_fallback=True
        )

    def _calculate_physics_failure_prob(
        self, features: MaintenanceFeatures
    ) -> float:
        """
        Calculate failure probability using physics/empirical model.

        DETERMINISTIC: Uses Weibull-based reliability model.
        """
        # Time-based hazard (Weibull with shape > 1 for wear-out)
        shape = 2.0  # Wear-out mode
        hours = features.operating_hours

        # Find minimum MTBF across failure modes
        min_mtbf = min(MTBF_HOURS.values())
        scale = min_mtbf

        # Base failure probability from Weibull
        if hours > 0 and scale > 0:
            base_prob = 1 - np.exp(-((hours / scale) ** shape))
        else:
            base_prob = 0.0

        # Stress acceleration factors
        temp_factor = 1.0
        if features.max_flame_temp_c > 1700:
            temp_factor = 1 + (features.max_flame_temp_c - 1700) / 300

        cycle_factor = 1.0
        if features.start_stop_cycles > 100:
            cycle_factor = 1 + (features.start_stop_cycles - 100) / 500

        vibration_factor = 1.0
        if features.vibration_rms > 5:
            vibration_factor = 1 + (features.vibration_rms - 5) / 10

        # Combined acceleration
        acceleration = temp_factor * cycle_factor * vibration_factor

        # Accelerated failure probability
        failure_prob = 1 - (1 - base_prob) ** acceleration

        # Efficiency degradation contribution
        if features.efficiency_trend < -0.5:
            failure_prob += abs(features.efficiency_trend) * 0.05

        failure_prob = max(0.0, min(1.0, failure_prob))

        return round(failure_prob, 4)

    def _calculate_physics_rul(self, features: MaintenanceFeatures) -> float:
        """
        Calculate remaining useful life using physics model.

        DETERMINISTIC: Uses linear degradation model with stress factors.
        """
        # Base RUL from MTBF
        base_rul = max(
            0,
            min(MTBF_HOURS.values()) - features.operating_hours
        )

        # Adjust for operating conditions
        temp_factor = 1.0
        if features.max_flame_temp_c > 1600:
            temp_factor = 1600 / features.max_flame_temp_c

        cycle_factor = 1.0
        if features.start_stop_cycles > 50:
            cycle_factor = 1 - min(0.5, (features.start_stop_cycles - 50) / 500)

        stability_factor = features.flame_stability_avg

        # Adjusted RUL
        rul = base_rul * temp_factor * cycle_factor * stability_factor

        # Minimum RUL based on maintenance interval
        if features.hours_since_last_maintenance > 4000:
            rul = min(rul, 8000 - features.hours_since_last_maintenance)

        return max(0, round(rul, 0))

    def _calculate_failure_mode_probabilities(
        self, features: MaintenanceFeatures
    ) -> Dict[str, float]:
        """Calculate probability for each failure mode."""
        probs = {}
        total = 0.0

        for mode in FailureMode:
            if mode in (FailureMode.NONE, FailureMode.UNKNOWN):
                continue

            # Base probability from MTBF
            mtbf = MTBF_HOURS.get(mode, 50000)
            hours = features.operating_hours
            base_prob = min(0.95, hours / mtbf)

            # Mode-specific adjustments
            if mode == FailureMode.BURNER_TIP_DEGRADATION:
                if features.max_flame_temp_c > 1700:
                    base_prob *= 1.5
            elif mode == FailureMode.IGNITER_FAILURE:
                if features.start_stop_cycles > 200:
                    base_prob *= 1.3
            elif mode == FailureMode.FLAME_SCANNER_DRIFT:
                if abs(features.flame_scanner_signal_trend) > 0.1:
                    base_prob *= 1.4
            elif mode == FailureMode.FUEL_VALVE_WEAR:
                if features.thermal_cycles > 100:
                    base_prob *= 1.2
            elif mode == FailureMode.AIR_DAMPER_MALFUNCTION:
                if features.vibration_rms > 8:
                    base_prob *= 1.3
            elif mode == FailureMode.HEAT_EXCHANGER_FOULING:
                if features.efficiency_trend < -0.3:
                    base_prob *= 1.5

            probs[mode.value] = base_prob
            total += base_prob

        # Normalize to sum to 1
        if total > 0:
            probs = {k: v / total for k, v in probs.items()}

        return probs

    def _determine_health_status(
        self, failure_prob: float, rul: float
    ) -> HealthStatus:
        """Determine equipment health status."""
        if failure_prob >= 0.7 or rul < 168:  # 1 week
            return HealthStatus.CRITICAL
        elif failure_prob >= 0.4 or rul < 720:  # 30 days
            return HealthStatus.WARNING
        elif failure_prob >= 0.2 or rul < 2160:  # 90 days
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY

    def _determine_priority(
        self,
        failure_prob: float,
        rul: float,
        health: HealthStatus
    ) -> MaintenancePriority:
        """Determine maintenance priority."""
        if health == HealthStatus.CRITICAL:
            return MaintenancePriority.CRITICAL
        elif health == HealthStatus.WARNING or failure_prob >= 0.5:
            return MaintenancePriority.HIGH
        elif health == HealthStatus.DEGRADED:
            return MaintenancePriority.MEDIUM
        elif rul < 4000:
            return MaintenancePriority.LOW
        else:
            return MaintenancePriority.ROUTINE

    def _identify_risk_factors(
        self, features: MaintenanceFeatures
    ) -> List[Dict[str, Any]]:
        """Identify top risk factors."""
        factors = []

        # Operating hours risk
        if features.operating_hours > 10000:
            factors.append({
                "factor": "operating_hours",
                "value": features.operating_hours,
                "risk_contribution": min(1.0, features.operating_hours / 50000),
                "description": f"High operating hours: {features.operating_hours:.0f}h"
            })

        # Temperature stress
        if features.max_flame_temp_c > 1650:
            factors.append({
                "factor": "temperature_stress",
                "value": features.max_flame_temp_c,
                "risk_contribution": (features.max_flame_temp_c - 1650) / 350,
                "description": f"High flame temperature: {features.max_flame_temp_c:.0f}C"
            })

        # Start-stop cycles
        if features.start_stop_cycles > 100:
            factors.append({
                "factor": "start_stop_cycles",
                "value": features.start_stop_cycles,
                "risk_contribution": min(1.0, features.start_stop_cycles / 500),
                "description": f"High cycle count: {features.start_stop_cycles}"
            })

        # Vibration
        if features.vibration_rms > 5:
            factors.append({
                "factor": "vibration",
                "value": features.vibration_rms,
                "risk_contribution": min(1.0, features.vibration_rms / 15),
                "description": f"Elevated vibration: {features.vibration_rms:.1f} RMS"
            })

        # Efficiency degradation
        if features.efficiency_trend < -0.2:
            factors.append({
                "factor": "efficiency_degradation",
                "value": features.efficiency_trend,
                "risk_contribution": min(1.0, abs(features.efficiency_trend)),
                "description": f"Efficiency declining: {features.efficiency_trend:.2f}%/month"
            })

        # Flame stability
        if features.flame_stability_avg < 0.8:
            factors.append({
                "factor": "flame_instability",
                "value": features.flame_stability_avg,
                "risk_contribution": 1 - features.flame_stability_avg,
                "description": f"Low flame stability: {features.flame_stability_avg:.2f}"
            })

        # Sort by risk contribution
        factors.sort(key=lambda x: x["risk_contribution"], reverse=True)

        return factors

    def _generate_recommendations(
        self,
        failure_mode: str,
        failure_prob: float,
        rul: float,
        risk_factors: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate maintenance recommendations."""
        recommendations = []

        # Priority-based recommendations
        if failure_prob >= 0.7:
            recommendations.append(
                "CRITICAL: Schedule immediate inspection and maintenance"
            )
        elif failure_prob >= 0.4:
            recommendations.append(
                "HIGH: Schedule maintenance within 1 week"
            )

        # Failure mode specific
        mode = FailureMode(failure_mode)
        if mode == FailureMode.BURNER_TIP_DEGRADATION:
            recommendations.append("Inspect burner tip for erosion or fouling")
            recommendations.append("Check flame pattern for asymmetry")
        elif mode == FailureMode.IGNITER_FAILURE:
            recommendations.append("Test igniter spark quality")
            recommendations.append("Inspect igniter electrode gap and condition")
        elif mode == FailureMode.FLAME_SCANNER_DRIFT:
            recommendations.append("Recalibrate flame scanner")
            recommendations.append("Clean scanner lens and check alignment")
        elif mode == FailureMode.FUEL_VALVE_WEAR:
            recommendations.append("Check fuel valve seat and trim")
            recommendations.append("Verify valve stroke and positioner calibration")
        elif mode == FailureMode.HEAT_EXCHANGER_FOULING:
            recommendations.append("Inspect heat exchanger for fouling")
            recommendations.append("Consider chemical cleaning or mechanical cleaning")

        # Risk factor specific
        for factor in risk_factors[:3]:
            if factor["factor"] == "temperature_stress":
                recommendations.append(
                    "Review burner tuning to reduce peak flame temperature"
                )
            elif factor["factor"] == "vibration":
                recommendations.append(
                    "Investigate vibration source - check combustion stability"
                )
            elif factor["factor"] == "efficiency_degradation":
                recommendations.append(
                    "Conduct combustion analysis and optimize air-fuel ratio"
                )

        # RUL-based
        if rul < 500:
            recommendations.append(
                f"Plan for major overhaul within {rul:.0f} operating hours"
            )

        return recommendations[:6]  # Limit to 6 recommendations

    def _calculate_maintenance_window(
        self, rul: float, priority: MaintenancePriority
    ) -> Dict[str, Any]:
        """Calculate optimal maintenance window."""
        now = datetime.now(timezone.utc)

        # Conservative window at 80% of RUL
        optimal_hours = rul * 0.8

        # Convert to calendar time (assuming 80% utilization)
        utilization = 0.8
        calendar_hours = optimal_hours / utilization
        optimal_date = now + timedelta(hours=calendar_hours)

        # Window boundaries
        window_start = now + timedelta(hours=optimal_hours * 0.6 / utilization)
        window_end = now + timedelta(hours=optimal_hours * 1.0 / utilization)

        return {
            "optimal_date": optimal_date.isoformat(),
            "window_start": window_start.isoformat(),
            "window_end": window_end.isoformat(),
            "operating_hours_remaining": optimal_hours,
            "priority": priority.value
        }

    def _calculate_confidence(self, failure_prob: float) -> float:
        """Calculate prediction confidence."""
        # Confidence decreases at extremes (harder to predict)
        if 0.3 <= failure_prob <= 0.7:
            return 0.75
        elif failure_prob < 0.1 or failure_prob > 0.9:
            return 0.85
        else:
            return 0.80

    def _extract_features(self, features: MaintenanceFeatures) -> np.ndarray:
        """Extract feature vector from input."""
        return np.array([
            features.operating_hours,
            features.hours_since_last_maintenance,
            features.start_stop_cycles,
            features.max_flame_temp_c,
            features.avg_flame_temp_c,
            features.temp_excursion_count,
            features.thermal_cycles,
            features.vibration_rms,
            features.pressure_oscillation_max,
            features.efficiency_trend,
            features.co_baseline_drift,
            features.o2_sensor_drift,
            features.flame_stability_avg,
            features.flame_scanner_signal_trend,
            features.equipment_age_years,
        ], dtype=np.float64)

    def fit(
        self,
        X: np.ndarray,
        y_failure: np.ndarray,
        y_rul: np.ndarray,
        validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """
        Train the predictive maintenance model.

        Args:
            X: Feature matrix (n_samples, n_features)
            y_failure: Binary failure labels
            y_rul: RUL values
            validation_split: Validation set fraction

        Returns:
            Training metrics
        """
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn required for training")

        start_time = time.time()

        # Scale features
        X_scaled = self._scaler.fit_transform(X)

        # Train classifier
        self._classifier.fit(X_scaled, y_failure)

        # Train regressor
        self._regressor.fit(X_scaled, y_rul)

        # Store feature importance
        if hasattr(self._classifier, "feature_importances_"):
            self._feature_importance = dict(zip(
                self.FEATURE_NAMES,
                [float(v) for v in self._classifier.feature_importances_]
            ))

        self._is_fitted = True

        elapsed = time.time() - start_time

        return {
            "training_time_s": elapsed,
            "n_samples": len(X),
            "n_features": X.shape[1],
            "feature_importance": self._feature_importance
        }

    def save_model(self, path: Path) -> None:
        """Save model to file."""
        data = {
            "classifier": self._classifier,
            "regressor": self._regressor,
            "scaler": self._scaler,
            "feature_importance": self._feature_importance,
            "is_fitted": self._is_fitted,
            "model_id": self._model_id
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Model saved to {path}")

    def _load_model(self, path: Path) -> None:
        """Load model from file."""
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self._classifier = data.get("classifier")
            self._regressor = data.get("regressor")
            self._scaler = data.get("scaler")
            self._feature_importance = data.get("feature_importance", {})
            self._is_fitted = data.get("is_fitted", False)
            self._model_id = data.get("model_id", self._model_id)
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            if SKLEARN_AVAILABLE:
                self._initialize_default_models()

    def _compute_provenance_hash(
        self,
        features: MaintenanceFeatures,
        prediction: MaintenancePrediction
    ) -> str:
        """Compute SHA-256 hash for audit trail."""
        data = {
            "model_id": self._model_id,
            "features": features.model_dump(),
            "failure_probability": prediction.failure_probability,
            "rul": prediction.remaining_useful_life_hours,
            "failure_mode": prediction.predicted_failure_mode.value,
            "timestamp": prediction.timestamp.isoformat()
        }
        return hashlib.sha256(
            json.dumps(data, sort_keys=True, default=str).encode()
        ).hexdigest()

    @property
    def model_id(self) -> str:
        """Get model identifier."""
        return self._model_id

    @property
    def is_fitted(self) -> bool:
        """Check if model is fitted."""
        return self._is_fitted

    @property
    def feature_importance(self) -> Dict[str, float]:
        """Get feature importance."""
        return self._feature_importance.copy()
