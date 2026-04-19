"""
CombustionAnomalyDetector - Real-Time Anomaly Detection for GL-004 BURNMASTER

This module implements real-time anomaly detection for combustion systems
using Isolation Forest and statistical methods.

Key Features:
    - Real-time anomaly detection using Isolation Forest
    - Multi-variate anomaly scoring
    - Anomaly classification by type
    - Severity assessment
    - Alert generation with root cause hints
    - Online learning for concept drift adaptation
    - Physics-based threshold fallback

CRITICAL: Anomaly alerts are ADVISORY ONLY.
Control decisions use deterministic physics-based calculations.

Example:
    >>> detector = CombustionAnomalyDetector()
    >>> features = CombustionFeatures(
    ...     o2_percent=3.5,
    ...     co_ppm=45,
    ...     flame_temp_c=1450
    ... )
    >>> result = detector.detect(features)
    >>> if result.is_anomaly:
    ...     print(f"Anomaly: {result.anomaly_type} - {result.severity}")

Author: GreenLang ML Engineering Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# Optional imports with graceful degradation
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.covariance import EllipticEnvelope
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available, using statistical fallback only")


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================


class AnomalySeverity(str, Enum):
    """Severity levels for detected anomalies."""
    CRITICAL = "critical"  # Immediate action required
    HIGH = "high"  # Investigate soon
    MEDIUM = "medium"  # Monitor closely
    LOW = "low"  # Minor deviation
    INFO = "info"  # Informational only


class AnomalyType(str, Enum):
    """Types of combustion anomalies."""
    NORMAL = "normal"  # No anomaly
    FLAME_INSTABILITY = "flame_instability"
    LEAN_BLOWOUT_RISK = "lean_blowout_risk"
    RICH_COMBUSTION = "rich_combustion"
    CO_SPIKE = "co_spike"
    NOX_EXCURSION = "nox_excursion"
    TEMPERATURE_ANOMALY = "temperature_anomaly"
    PRESSURE_OSCILLATION = "pressure_oscillation"
    SENSOR_FAULT = "sensor_fault"
    AIR_FUEL_RATIO_DRIFT = "air_fuel_ratio_drift"
    EFFICIENCY_DEGRADATION = "efficiency_degradation"
    UNKNOWN = "unknown"


class DetectionMethod(str, Enum):
    """Anomaly detection method used."""
    ISOLATION_FOREST = "isolation_forest"
    ELLIPTIC_ENVELOPE = "elliptic_envelope"
    STATISTICAL = "statistical"
    THRESHOLD = "threshold"
    ENSEMBLE = "ensemble"


# Threshold definitions for physics-based detection
ANOMALY_THRESHOLDS = {
    "o2_min": 0.5,
    "o2_max": 10.0,
    "co_warning": 150.0,
    "co_critical": 300.0,
    "nox_warning": 75.0,
    "nox_critical": 150.0,
    "flame_temp_min": 1100.0,
    "flame_temp_max": 1950.0,
    "lambda_min": 1.02,
    "lambda_max": 1.5,
    "pressure_var_max": 10.0,
    "stability_min": 0.6,
}

# Feature importance for anomaly classification
FEATURE_WEIGHTS = {
    "o2_percent": 0.15,
    "co_ppm": 0.15,
    "nox_ppm": 0.10,
    "flame_temp_c": 0.15,
    "lambda_value": 0.10,
    "flame_stability": 0.15,
    "pressure_variance": 0.10,
    "efficiency_percent": 0.10,
}


# =============================================================================
# PYDANTIC SCHEMAS
# =============================================================================


class CombustionFeatures(BaseModel):
    """Input features for anomaly detection."""

    # Flue gas composition
    o2_percent: float = Field(
        default=3.0, ge=0.0, le=21.0,
        description="O2 percentage in flue gas"
    )
    co_ppm: float = Field(
        default=50.0, ge=0.0,
        description="CO concentration in ppm"
    )
    co2_percent: Optional[float] = Field(
        default=None, ge=0.0, le=25.0,
        description="CO2 percentage"
    )
    nox_ppm: Optional[float] = Field(
        default=None, ge=0.0,
        description="NOx concentration in ppm"
    )

    # Temperature measurements
    flame_temp_c: float = Field(
        default=1500.0, ge=0.0, le=2500.0,
        description="Flame temperature in Celsius"
    )
    stack_temp_c: Optional[float] = Field(
        default=None, ge=0.0, le=500.0,
        description="Stack temperature in Celsius"
    )
    furnace_temp_c: Optional[float] = Field(
        default=None, ge=0.0, le=1500.0,
        description="Furnace temperature in Celsius"
    )

    # Air-fuel parameters
    lambda_value: float = Field(
        default=1.15, ge=0.5, le=3.0,
        description="Lambda (air-fuel equivalence ratio)"
    )
    excess_air_percent: Optional[float] = Field(
        default=None, ge=-50.0, le=200.0,
        description="Excess air percentage"
    )

    # Flow rates
    fuel_flow_rate: Optional[float] = Field(
        default=None, ge=0.0,
        description="Fuel flow rate"
    )
    air_flow_rate: Optional[float] = Field(
        default=None, ge=0.0,
        description="Air flow rate"
    )

    # Stability indicators
    flame_stability: float = Field(
        default=0.9, ge=0.0, le=1.0,
        description="Flame stability index (0=unstable, 1=stable)"
    )
    pressure_variance: float = Field(
        default=1.0, ge=0.0,
        description="Furnace pressure variance"
    )
    flame_scanner_signal: Optional[float] = Field(
        default=None, ge=0.0, le=100.0,
        description="Flame scanner signal strength"
    )

    # Derived features
    efficiency_percent: Optional[float] = Field(
        default=None, ge=50.0, le=100.0,
        description="Combustion efficiency"
    )

    # Temporal features
    temp_rate_of_change: Optional[float] = Field(
        default=None,
        description="Temperature rate of change (C/min)"
    )
    o2_rate_of_change: Optional[float] = Field(
        default=None,
        description="O2 rate of change (%/min)"
    )

    # Metadata
    burner_id: str = Field(default="BNR-001", description="Burner ID")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Measurement timestamp"
    )


class AnomalyAlert(BaseModel):
    """Alert generated from anomaly detection."""

    alert_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique alert identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Alert timestamp"
    )
    anomaly_type: AnomalyType = Field(..., description="Type of anomaly")
    severity: AnomalySeverity = Field(..., description="Alert severity")
    title: str = Field(..., description="Alert title")
    description: str = Field(..., description="Detailed description")
    affected_variables: List[str] = Field(
        default_factory=list,
        description="Variables with anomalous values"
    )
    current_values: Dict[str, float] = Field(
        default_factory=dict,
        description="Current variable values"
    )
    threshold_values: Dict[str, float] = Field(
        default_factory=dict,
        description="Threshold values exceeded"
    )
    recommended_actions: List[str] = Field(
        default_factory=list,
        description="Recommended corrective actions"
    )
    burner_id: str = Field(default="BNR-001", description="Affected burner")


class AnomalyResult(BaseModel):
    """Result from anomaly detection."""

    result_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique result identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Detection timestamp"
    )

    # Core detection results
    is_anomaly: bool = Field(..., description="Whether anomaly was detected")
    anomaly_score: float = Field(
        ..., ge=-1.0, le=1.0,
        description="Anomaly score (-1=normal, 1=highly anomalous)"
    )
    anomaly_type: AnomalyType = Field(..., description="Classified anomaly type")
    severity: AnomalySeverity = Field(..., description="Anomaly severity")

    # Confidence and method
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Detection confidence"
    )
    detection_method: DetectionMethod = Field(
        ..., description="Detection method used"
    )

    # Feature analysis
    feature_contributions: Dict[str, float] = Field(
        default_factory=dict,
        description="Feature contribution to anomaly"
    )
    deviating_features: List[str] = Field(
        default_factory=list,
        description="Features with significant deviation"
    )

    # Context
    deviation_magnitude: float = Field(
        default=0.0, ge=0.0,
        description="Overall deviation magnitude"
    )
    baseline_comparison: Dict[str, float] = Field(
        default_factory=dict,
        description="Comparison to baseline values"
    )

    # Alerts
    alerts: List[AnomalyAlert] = Field(
        default_factory=list,
        description="Generated alerts"
    )

    # Probable causes
    probable_causes: List[str] = Field(
        default_factory=list,
        description="Probable root causes"
    )
    recommended_actions: List[str] = Field(
        default_factory=list,
        description="Recommended actions"
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
# ANOMALY DETECTOR
# =============================================================================


class CombustionAnomalyDetector:
    """
    Real-time anomaly detection for combustion systems.

    Uses Isolation Forest as the primary detector with statistical
    and threshold-based fallbacks. Supports online learning for
    adaptation to concept drift.

    CRITICAL: Anomaly alerts are ADVISORY ONLY.

    Attributes:
        is_fitted: Whether model has been trained
        model_id: Unique model identifier
        contamination: Expected proportion of anomalies

    Example:
        >>> detector = CombustionAnomalyDetector()
        >>> features = CombustionFeatures(o2_percent=1.5, co_ppm=350)
        >>> result = detector.detect(features)
        >>> if result.is_anomaly:
        ...     for alert in result.alerts:
        ...         print(f"{alert.severity}: {alert.title}")
    """

    FEATURE_NAMES = [
        "o2_percent",
        "co_ppm",
        "nox_ppm",
        "flame_temp_c",
        "lambda_value",
        "flame_stability",
        "pressure_variance",
        "efficiency_percent",
        "temp_rate_of_change",
        "o2_rate_of_change",
    ]

    def __init__(
        self,
        model_path: Optional[Path] = None,
        contamination: float = 0.05,
        n_estimators: int = 100,
        window_size: int = 1000,
        random_seed: int = 42
    ):
        """
        Initialize CombustionAnomalyDetector.

        Args:
            model_path: Path to pre-trained model file
            contamination: Expected proportion of anomalies (0.01-0.5)
            n_estimators: Number of trees in Isolation Forest
            window_size: Size of rolling window for online learning
            random_seed: Random seed for reproducibility
        """
        self.contamination = min(0.5, max(0.01, contamination))
        self.n_estimators = n_estimators
        self.window_size = window_size
        self.random_seed = random_seed
        self._model_id = f"anomaly_{uuid4().hex[:8]}"

        self._isolation_forest: Optional[Any] = None
        self._elliptic_envelope: Optional[Any] = None
        self._scaler: Optional[Any] = None
        self._is_fitted = False

        # Baseline statistics for statistical detection
        self._baseline_mean: Dict[str, float] = {}
        self._baseline_std: Dict[str, float] = {}

        # Rolling window for online updates
        self._feature_buffer: Deque[np.ndarray] = deque(maxlen=window_size)

        # Alert cooldown tracking
        self._recent_alerts: Dict[str, datetime] = {}
        self._alert_cooldown = timedelta(minutes=5)

        if model_path and model_path.exists():
            self._load_model(model_path)
        elif SKLEARN_AVAILABLE:
            self._initialize_default_models()

        logger.info(
            f"CombustionAnomalyDetector initialized: "
            f"id={self._model_id}, contamination={self.contamination}"
        )

    def _initialize_default_models(self) -> None:
        """Initialize default model architecture."""
        if not SKLEARN_AVAILABLE:
            return

        self._isolation_forest = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_seed,
            n_jobs=-1
        )

        self._elliptic_envelope = EllipticEnvelope(
            contamination=self.contamination,
            random_state=self.random_seed,
            support_fraction=0.7
        )

        self._scaler = StandardScaler()

    def detect(self, features: CombustionFeatures) -> AnomalyResult:
        """
        Detect anomalies in combustion data.

        Args:
            features: Current combustion measurements

        Returns:
            AnomalyResult with detection results and alerts
        """
        start_time = time.time()

        # Extract feature vector
        feature_vector = self._extract_features(features)

        # Add to buffer for online learning
        self._feature_buffer.append(feature_vector)

        # Detect anomaly using available method
        if self._is_fitted and SKLEARN_AVAILABLE:
            result = self._detect_with_ml(features, feature_vector)
        else:
            result = self._detect_with_thresholds(features)

        # Generate alerts if anomaly detected
        if result.is_anomaly:
            result.alerts = self._generate_alerts(features, result)

        # Compute provenance hash
        result.provenance_hash = self._compute_provenance_hash(features, result)
        result.computation_time_ms = (time.time() - start_time) * 1000

        logger.debug(
            f"Anomaly detection: is_anomaly={result.is_anomaly}, "
            f"score={result.anomaly_score:.3f}, type={result.anomaly_type.value}"
        )

        return result

    def detect_batch(
        self, features_list: List[CombustionFeatures]
    ) -> List[AnomalyResult]:
        """Detect anomalies in batch."""
        return [self.detect(f) for f in features_list]

    def _detect_with_ml(
        self,
        features: CombustionFeatures,
        feature_vector: np.ndarray
    ) -> AnomalyResult:
        """Detect anomaly using ML models."""
        # Scale features
        if self._scaler and hasattr(self._scaler, "mean_"):
            feature_vector_scaled = self._scaler.transform(
                feature_vector.reshape(1, -1)
            )
        else:
            feature_vector_scaled = feature_vector.reshape(1, -1)

        # Isolation Forest score
        if self._isolation_forest and hasattr(self._isolation_forest, "offset_"):
            if_prediction = self._isolation_forest.predict(feature_vector_scaled)[0]
            if_score = -self._isolation_forest.score_samples(feature_vector_scaled)[0]
            is_anomaly_if = if_prediction == -1
        else:
            if_score = 0.0
            is_anomaly_if = False

        # Threshold-based check as secondary
        threshold_result = self._detect_with_thresholds(features)

        # Ensemble decision
        anomaly_score = if_score
        is_anomaly = is_anomaly_if or (threshold_result.is_anomaly and threshold_result.severity in [AnomalySeverity.CRITICAL, AnomalySeverity.HIGH])

        # Calculate feature contributions
        contributions = self._calculate_feature_contributions(
            features, feature_vector_scaled
        )

        # Classify anomaly type
        anomaly_type, probable_causes = self._classify_anomaly(
            features, contributions, is_anomaly
        )

        # Determine severity
        severity = self._determine_severity(anomaly_score, anomaly_type, features)

        # Calculate confidence
        confidence = self._calculate_confidence(anomaly_score, is_anomaly)

        # Deviating features
        deviating = [
            name for name, contrib in contributions.items()
            if contrib > 0.2
        ]

        # Recommended actions
        actions = self._generate_recommendations(anomaly_type, contributions)

        return AnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_score=round(float(anomaly_score), 4),
            anomaly_type=anomaly_type,
            severity=severity,
            confidence=confidence,
            detection_method=DetectionMethod.ISOLATION_FOREST,
            feature_contributions=contributions,
            deviating_features=deviating,
            deviation_magnitude=sum(contributions.values()),
            probable_causes=probable_causes,
            recommended_actions=actions,
            model_version="1.0.0",
            is_physics_fallback=False
        )

    def _detect_with_thresholds(
        self, features: CombustionFeatures
    ) -> AnomalyResult:
        """
        Detect anomaly using threshold-based rules.

        DETERMINISTIC: Uses predefined thresholds.
        """
        violations = []
        contributions = {}
        max_severity = AnomalySeverity.INFO

        # O2 check
        if features.o2_percent < ANOMALY_THRESHOLDS["o2_min"]:
            violations.append(("o2_low", AnomalySeverity.CRITICAL))
            contributions["o2_percent"] = abs(
                features.o2_percent - ANOMALY_THRESHOLDS["o2_min"]
            )
            max_severity = AnomalySeverity.CRITICAL
        elif features.o2_percent > ANOMALY_THRESHOLDS["o2_max"]:
            violations.append(("o2_high", AnomalySeverity.MEDIUM))
            contributions["o2_percent"] = abs(
                features.o2_percent - ANOMALY_THRESHOLDS["o2_max"]
            )
            if max_severity.value not in ["critical", "high"]:
                max_severity = AnomalySeverity.MEDIUM

        # CO check
        if features.co_ppm > ANOMALY_THRESHOLDS["co_critical"]:
            violations.append(("co_critical", AnomalySeverity.CRITICAL))
            contributions["co_ppm"] = features.co_ppm / ANOMALY_THRESHOLDS["co_critical"]
            max_severity = AnomalySeverity.CRITICAL
        elif features.co_ppm > ANOMALY_THRESHOLDS["co_warning"]:
            violations.append(("co_warning", AnomalySeverity.HIGH))
            contributions["co_ppm"] = features.co_ppm / ANOMALY_THRESHOLDS["co_warning"]
            if max_severity != AnomalySeverity.CRITICAL:
                max_severity = AnomalySeverity.HIGH

        # NOx check
        if features.nox_ppm is not None:
            if features.nox_ppm > ANOMALY_THRESHOLDS["nox_critical"]:
                violations.append(("nox_critical", AnomalySeverity.CRITICAL))
                contributions["nox_ppm"] = features.nox_ppm / ANOMALY_THRESHOLDS["nox_critical"]
                max_severity = AnomalySeverity.CRITICAL
            elif features.nox_ppm > ANOMALY_THRESHOLDS["nox_warning"]:
                violations.append(("nox_warning", AnomalySeverity.HIGH))
                contributions["nox_ppm"] = features.nox_ppm / ANOMALY_THRESHOLDS["nox_warning"]

        # Flame temperature check
        if features.flame_temp_c < ANOMALY_THRESHOLDS["flame_temp_min"]:
            violations.append(("flame_temp_low", AnomalySeverity.HIGH))
            contributions["flame_temp_c"] = abs(
                features.flame_temp_c - ANOMALY_THRESHOLDS["flame_temp_min"]
            ) / 100
        elif features.flame_temp_c > ANOMALY_THRESHOLDS["flame_temp_max"]:
            violations.append(("flame_temp_high", AnomalySeverity.CRITICAL))
            contributions["flame_temp_c"] = abs(
                features.flame_temp_c - ANOMALY_THRESHOLDS["flame_temp_max"]
            ) / 100
            max_severity = AnomalySeverity.CRITICAL

        # Lambda check
        if features.lambda_value < ANOMALY_THRESHOLDS["lambda_min"]:
            violations.append(("lambda_low", AnomalySeverity.CRITICAL))
            contributions["lambda_value"] = abs(
                features.lambda_value - ANOMALY_THRESHOLDS["lambda_min"]
            ) * 5
            max_severity = AnomalySeverity.CRITICAL
        elif features.lambda_value > ANOMALY_THRESHOLDS["lambda_max"]:
            violations.append(("lambda_high", AnomalySeverity.MEDIUM))
            contributions["lambda_value"] = abs(
                features.lambda_value - ANOMALY_THRESHOLDS["lambda_max"]
            ) * 2

        # Stability check
        if features.flame_stability < ANOMALY_THRESHOLDS["stability_min"]:
            violations.append(("stability_low", AnomalySeverity.HIGH))
            contributions["flame_stability"] = (
                ANOMALY_THRESHOLDS["stability_min"] - features.flame_stability
            ) * 2

        # Pressure variance check
        if features.pressure_variance > ANOMALY_THRESHOLDS["pressure_var_max"]:
            violations.append(("pressure_oscillation", AnomalySeverity.HIGH))
            contributions["pressure_variance"] = (
                features.pressure_variance / ANOMALY_THRESHOLDS["pressure_var_max"]
            )

        is_anomaly = len(violations) > 0

        # Calculate anomaly score
        if contributions:
            anomaly_score = min(1.0, sum(contributions.values()) / len(contributions))
        else:
            anomaly_score = 0.0

        # Classify anomaly type
        anomaly_type = AnomalyType.NORMAL
        probable_causes = []

        if violations:
            violation_types = [v[0] for v in violations]

            if "co_critical" in violation_types or "co_warning" in violation_types:
                anomaly_type = AnomalyType.CO_SPIKE
                probable_causes.extend([
                    "Insufficient combustion air",
                    "Poor air-fuel mixing",
                    "Low flame temperature"
                ])
            elif "lambda_low" in violation_types:
                anomaly_type = AnomalyType.RICH_COMBUSTION
                probable_causes.extend([
                    "Excess fuel supply",
                    "Air damper malfunction",
                    "Control system error"
                ])
            elif "o2_low" in violation_types:
                anomaly_type = AnomalyType.LEAN_BLOWOUT_RISK
                probable_causes.extend([
                    "Operating near stability limit",
                    "Air-fuel ratio too lean",
                    "Sudden load change"
                ])
            elif "stability_low" in violation_types:
                anomaly_type = AnomalyType.FLAME_INSTABILITY
                probable_causes.extend([
                    "Poor fuel quality",
                    "Burner tip fouling",
                    "Combustion air preheat issue"
                ])
            elif "flame_temp_high" in violation_types:
                anomaly_type = AnomalyType.TEMPERATURE_ANOMALY
                probable_causes.extend([
                    "Hotspot development",
                    "Refractory damage",
                    "Excess air ratio issue"
                ])
            elif "pressure_oscillation" in violation_types:
                anomaly_type = AnomalyType.PRESSURE_OSCILLATION
                probable_causes.extend([
                    "Combustion instability",
                    "Draft control issue",
                    "Fuel pressure fluctuation"
                ])
            elif "nox_critical" in violation_types or "nox_warning" in violation_types:
                anomaly_type = AnomalyType.NOX_EXCURSION
                probable_causes.extend([
                    "High flame temperature",
                    "Excess air at high load",
                    "Fuel nitrogen content change"
                ])

        # Generate recommendations
        actions = self._generate_recommendations(anomaly_type, contributions)

        return AnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_score=round(anomaly_score, 4),
            anomaly_type=anomaly_type,
            severity=max_severity,
            confidence=0.7 if is_anomaly else 0.9,  # Lower confidence for threshold
            detection_method=DetectionMethod.THRESHOLD,
            feature_contributions={k: round(v, 4) for k, v in contributions.items()},
            deviating_features=list(contributions.keys()),
            deviation_magnitude=sum(contributions.values()),
            probable_causes=probable_causes,
            recommended_actions=actions,
            model_version="1.0.0",
            is_physics_fallback=True
        )

    def _calculate_feature_contributions(
        self,
        features: CombustionFeatures,
        feature_vector_scaled: np.ndarray
    ) -> Dict[str, float]:
        """Calculate contribution of each feature to anomaly score."""
        contributions = {}

        # Use distance from mean as contribution
        for i, name in enumerate(self.FEATURE_NAMES):
            if i < feature_vector_scaled.shape[1]:
                value = abs(feature_vector_scaled[0, i])
                # Weight by feature importance
                weight = FEATURE_WEIGHTS.get(name, 0.1)
                contributions[name] = round(float(value * weight), 4)

        # Normalize
        total = sum(contributions.values())
        if total > 0:
            contributions = {k: v / total for k, v in contributions.items()}

        return contributions

    def _classify_anomaly(
        self,
        features: CombustionFeatures,
        contributions: Dict[str, float],
        is_anomaly: bool
    ) -> Tuple[AnomalyType, List[str]]:
        """Classify anomaly type based on features and contributions."""
        if not is_anomaly:
            return AnomalyType.NORMAL, []

        probable_causes = []

        # Get top contributing features
        top_features = sorted(
            contributions.items(), key=lambda x: x[1], reverse=True
        )[:3]
        top_names = [f[0] for f in top_features]

        # Classify based on top contributors
        if "co_ppm" in top_names and features.co_ppm > 100:
            return AnomalyType.CO_SPIKE, [
                "Incomplete combustion",
                "Insufficient combustion air",
                "Fuel quality issue"
            ]

        if "flame_stability" in top_names and features.flame_stability < 0.7:
            return AnomalyType.FLAME_INSTABILITY, [
                "Operating near lean limit",
                "Poor fuel quality",
                "Burner maintenance required"
            ]

        if "lambda_value" in top_names:
            if features.lambda_value < 1.05:
                return AnomalyType.RICH_COMBUSTION, [
                    "Excess fuel",
                    "Air flow restriction",
                    "Control system error"
                ]
            elif features.lambda_value > 1.4:
                return AnomalyType.LEAN_BLOWOUT_RISK, [
                    "Excess air",
                    "Fuel flow restriction",
                    "Load change"
                ]

        if "flame_temp_c" in top_names:
            return AnomalyType.TEMPERATURE_ANOMALY, [
                "Combustion intensity change",
                "Air preheat variation",
                "Fuel heating value change"
            ]

        if "pressure_variance" in top_names and features.pressure_variance > 5:
            return AnomalyType.PRESSURE_OSCILLATION, [
                "Combustion instability",
                "Draft fluctuation",
                "Fuel pressure variation"
            ]

        if "o2_percent" in top_names:
            return AnomalyType.AIR_FUEL_RATIO_DRIFT, [
                "Air damper position change",
                "Fuel flow change",
                "Sensor calibration drift"
            ]

        return AnomalyType.UNKNOWN, ["Investigate process conditions"]

    def _determine_severity(
        self,
        anomaly_score: float,
        anomaly_type: AnomalyType,
        features: CombustionFeatures
    ) -> AnomalySeverity:
        """Determine anomaly severity."""
        # Critical conditions
        if (features.lambda_value < 1.02 or
            features.co_ppm > 300 or
            features.flame_stability < 0.5):
            return AnomalySeverity.CRITICAL

        # High severity
        if (anomaly_score > 0.7 or
            features.co_ppm > 150 or
            features.flame_stability < 0.65):
            return AnomalySeverity.HIGH

        # Medium severity
        if anomaly_score > 0.4:
            return AnomalySeverity.MEDIUM

        # Low severity
        if anomaly_score > 0.2:
            return AnomalySeverity.LOW

        return AnomalySeverity.INFO

    def _calculate_confidence(
        self, anomaly_score: float, is_anomaly: bool
    ) -> float:
        """Calculate detection confidence."""
        # High confidence at extremes
        if anomaly_score > 0.8 or anomaly_score < 0.1:
            return 0.9
        # Lower confidence in ambiguous region
        elif 0.3 < anomaly_score < 0.6:
            return 0.7
        else:
            return 0.8

    def _generate_recommendations(
        self,
        anomaly_type: AnomalyType,
        contributions: Dict[str, float]
    ) -> List[str]:
        """Generate recommended actions."""
        recommendations = []

        if anomaly_type == AnomalyType.CO_SPIKE:
            recommendations.extend([
                "Increase combustion air flow",
                "Check burner alignment and flame pattern",
                "Verify fuel quality and flow"
            ])
        elif anomaly_type == AnomalyType.FLAME_INSTABILITY:
            recommendations.extend([
                "Reduce excess air gradually",
                "Check fuel pressure stability",
                "Inspect burner tip for fouling"
            ])
        elif anomaly_type == AnomalyType.LEAN_BLOWOUT_RISK:
            recommendations.extend([
                "CAUTION: Reduce load or increase fuel",
                "Verify air damper position",
                "Check fuel supply pressure"
            ])
        elif anomaly_type == AnomalyType.RICH_COMBUSTION:
            recommendations.extend([
                "Increase combustion air",
                "Check air damper operation",
                "Verify fuel flow metering"
            ])
        elif anomaly_type == AnomalyType.TEMPERATURE_ANOMALY:
            recommendations.extend([
                "Check air-fuel ratio",
                "Verify combustion air preheat",
                "Inspect for refractory damage"
            ])
        elif anomaly_type == AnomalyType.PRESSURE_OSCILLATION:
            recommendations.extend([
                "Check combustion stability",
                "Verify draft control",
                "Reduce firing rate temporarily"
            ])
        elif anomaly_type == AnomalyType.NOX_EXCURSION:
            recommendations.extend([
                "Reduce flame temperature if possible",
                "Adjust air staging if equipped",
                "Check for flue gas recirculation issues"
            ])
        else:
            recommendations.append("Monitor closely and investigate trends")

        return recommendations[:5]

    def _generate_alerts(
        self,
        features: CombustionFeatures,
        result: AnomalyResult
    ) -> List[AnomalyAlert]:
        """Generate alerts from anomaly detection."""
        alerts = []

        # Check cooldown
        alert_key = f"{features.burner_id}_{result.anomaly_type.value}"
        now = datetime.now(timezone.utc)

        if alert_key in self._recent_alerts:
            if now - self._recent_alerts[alert_key] < self._alert_cooldown:
                return alerts  # Still in cooldown

        # Create alert
        alert = AnomalyAlert(
            anomaly_type=result.anomaly_type,
            severity=result.severity,
            title=f"{result.anomaly_type.value.replace('_', ' ').title()} Detected",
            description=f"Anomaly score: {result.anomaly_score:.2f}, "
                       f"Confidence: {result.confidence:.1%}",
            affected_variables=result.deviating_features,
            current_values={
                "o2_percent": features.o2_percent,
                "co_ppm": features.co_ppm,
                "flame_temp_c": features.flame_temp_c,
                "lambda_value": features.lambda_value,
                "flame_stability": features.flame_stability,
            },
            threshold_values=ANOMALY_THRESHOLDS,
            recommended_actions=result.recommended_actions,
            burner_id=features.burner_id
        )

        alerts.append(alert)
        self._recent_alerts[alert_key] = now

        return alerts

    def _extract_features(self, features: CombustionFeatures) -> np.ndarray:
        """Extract feature vector from input."""
        return np.array([
            features.o2_percent,
            features.co_ppm,
            features.nox_ppm if features.nox_ppm is not None else 30.0,
            features.flame_temp_c,
            features.lambda_value,
            features.flame_stability,
            features.pressure_variance,
            features.efficiency_percent if features.efficiency_percent is not None else 85.0,
            features.temp_rate_of_change if features.temp_rate_of_change is not None else 0.0,
            features.o2_rate_of_change if features.o2_rate_of_change is not None else 0.0,
        ], dtype=np.float64)

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None  # Labels optional for unsupervised
    ) -> Dict[str, Any]:
        """
        Train the anomaly detector.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Optional labels (not used for Isolation Forest)

        Returns:
            Training metrics
        """
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn required for training")

        start_time = time.time()

        # Scale features
        X_scaled = self._scaler.fit_transform(X)

        # Train Isolation Forest
        self._isolation_forest.fit(X_scaled)

        # Compute baseline statistics
        for i, name in enumerate(self.FEATURE_NAMES):
            if i < X.shape[1]:
                self._baseline_mean[name] = float(np.mean(X[:, i]))
                self._baseline_std[name] = float(np.std(X[:, i]))

        self._is_fitted = True

        elapsed = time.time() - start_time

        return {
            "training_time_s": elapsed,
            "n_samples": len(X),
            "n_features": X.shape[1],
            "contamination": self.contamination,
            "baseline_mean": self._baseline_mean,
            "baseline_std": self._baseline_std
        }

    def partial_fit(self, X: np.ndarray) -> None:
        """
        Online update of the model (pseudo-incremental).

        Note: Isolation Forest doesn't support true incremental learning.
        This retrains on recent window data.

        Args:
            X: New feature samples
        """
        if not SKLEARN_AVAILABLE or not self._is_fitted:
            return

        # Add to buffer
        for row in X:
            self._feature_buffer.append(row)

        # Retrain if buffer is full
        if len(self._feature_buffer) >= self.window_size:
            buffer_array = np.array(list(self._feature_buffer))
            self.fit(buffer_array)
            logger.info("Model updated with recent data window")

    def save_model(self, path: Path) -> None:
        """Save model to file."""
        data = {
            "isolation_forest": self._isolation_forest,
            "scaler": self._scaler,
            "baseline_mean": self._baseline_mean,
            "baseline_std": self._baseline_std,
            "is_fitted": self._is_fitted,
            "model_id": self._model_id,
            "contamination": self.contamination
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Model saved to {path}")

    def _load_model(self, path: Path) -> None:
        """Load model from file."""
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self._isolation_forest = data.get("isolation_forest")
            self._scaler = data.get("scaler")
            self._baseline_mean = data.get("baseline_mean", {})
            self._baseline_std = data.get("baseline_std", {})
            self._is_fitted = data.get("is_fitted", False)
            self._model_id = data.get("model_id", self._model_id)
            self.contamination = data.get("contamination", self.contamination)
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            if SKLEARN_AVAILABLE:
                self._initialize_default_models()

    def _compute_provenance_hash(
        self,
        features: CombustionFeatures,
        result: AnomalyResult
    ) -> str:
        """Compute SHA-256 hash for audit trail."""
        data = {
            "model_id": self._model_id,
            "features": {
                "o2_percent": features.o2_percent,
                "co_ppm": features.co_ppm,
                "flame_temp_c": features.flame_temp_c,
                "lambda_value": features.lambda_value
            },
            "is_anomaly": result.is_anomaly,
            "anomaly_score": result.anomaly_score,
            "anomaly_type": result.anomaly_type.value,
            "timestamp": result.timestamp.isoformat()
        }
        return hashlib.sha256(
            json.dumps(data, sort_keys=True, default=str).encode()
        ).hexdigest()

    def get_baseline_stats(self) -> Dict[str, Dict[str, float]]:
        """Get baseline statistics for each feature."""
        return {
            "mean": self._baseline_mean.copy(),
            "std": self._baseline_std.copy()
        }

    @property
    def model_id(self) -> str:
        """Get model identifier."""
        return self._model_id

    @property
    def is_fitted(self) -> bool:
        """Check if model is fitted."""
        return self._is_fitted
