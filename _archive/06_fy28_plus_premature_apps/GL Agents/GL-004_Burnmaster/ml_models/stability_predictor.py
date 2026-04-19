# -*- coding: utf-8 -*-
"""
FlameStabilityPredictor - Flame Stability and Anomaly Detection

This module implements ML models for predicting flame instability risk,
classifying operating regimes, and detecting combustion anomalies.

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import time
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.isotonic import IsotonicRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


class OperatingRegime(str, Enum):
    """Operating regime classification for burner systems."""
    STABLE = "stable"
    MARGINAL = "marginal"
    UNSTABLE = "unstable"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class AnomalyType(str, Enum):
    """Types of combustion anomalies."""
    FLAME_OSCILLATION = "flame_oscillation"
    LEAN_BLOWOUT_RISK = "lean_blowout_risk"
    FLASHBACK_RISK = "flashback_risk"
    SENSOR_FAULT = "sensor_fault"
    FUEL_QUALITY_CHANGE = "fuel_quality_change"
    AIR_FUEL_RATIO_DRIFT = "air_fuel_ratio_drift"
    UNKNOWN = "unknown"


class ModelType(str, Enum):
    """Supported model types for stability prediction."""
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"


REGIME_THRESHOLDS = {
    OperatingRegime.STABLE: (0.0, 0.15),
    OperatingRegime.MARGINAL: (0.15, 0.40),
    OperatingRegime.UNSTABLE: (0.40, 0.70),
    OperatingRegime.CRITICAL: (0.70, 1.0),
}


class StabilityFeatures(BaseModel):
    """Input features for stability prediction."""
    excess_air_percent: float = Field(..., ge=0.0, le=100.0)
    flame_temperature_c: float = Field(..., ge=0.0, le=2500.0)
    fuel_flow_rate_kg_s: float = Field(..., ge=0.0)
    air_flow_rate_kg_s: float = Field(..., ge=0.0)
    furnace_pressure_kpa: float = Field(default=101.325)
    fuel_pressure_kpa: float = Field(default=200.0, ge=0.0)
    air_preheat_temp_c: float = Field(default=25.0)
    stoichiometric_ratio: Optional[float] = Field(default=None)
    o2_stack_percent: float = Field(default=3.0, ge=0.0, le=21.0)
    co_ppm: float = Field(default=50.0, ge=0.0)
    nox_ppm: Optional[float] = Field(default=None, ge=0.0)
    flame_temp_std_1min: Optional[float] = Field(default=None, ge=0.0)
    pressure_oscillation_hz: Optional[float] = Field(default=None, ge=0.0)
    load_change_rate_percent_min: Optional[float] = Field(default=None)
    burner_id: str = Field(default="BNR-001")
    fuel_type: str = Field(default="natural_gas")


class ConfidenceInterval(BaseModel):
    """Confidence interval for predictions."""
    lower: float
    upper: float
    confidence_level: float = Field(default=0.90)


class ProvenanceRecord(BaseModel):
    """Provenance tracking for audit trails."""
    record_id: str = Field(default_factory=lambda: str(uuid4()))
    calculation_type: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    model_id: str = Field(default="")
    input_hash: str = Field(default="")
    output_hash: str = Field(default="")
    computation_time_ms: float = Field(default=0.0)

    @classmethod
    def create(cls, calculation_type: str, inputs: Dict, outputs: Dict,
               model_id: str = "", computation_time_ms: float = 0.0,
               random_seed: Optional[int] = None) -> "ProvenanceRecord":
        """Create provenance record with computed hashes."""
        return cls(
            calculation_type=calculation_type,
            model_id=model_id,
            input_hash=hashlib.sha256(json.dumps(inputs, sort_keys=True, default=str).encode()).hexdigest(),
            output_hash=hashlib.sha256(json.dumps(outputs, sort_keys=True, default=str).encode()).hexdigest(),
            computation_time_ms=computation_time_ms
        )


class InstabilityPrediction(BaseModel):
    """Prediction result for flame instability risk."""
    prediction_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    instability_risk: float = Field(..., ge=0.0, le=1.0)
    risk_category: str
    confidence_interval: ConfidenceInterval
    prediction_uncertainty: float = Field(..., ge=0.0, le=1.0)
    operating_regime: OperatingRegime
    top_risk_factors: List[Dict[str, float]] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)
    provenance: ProvenanceRecord
    computation_time_ms: float = Field(default=0.0)


class BurnerState(BaseModel):
    """Current state of a burner for regime classification."""
    burner_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    firing_rate_percent: float = Field(..., ge=0.0, le=100.0)
    excess_air_percent: float = Field(..., ge=0.0, le=100.0)
    flame_signal_strength: float = Field(..., ge=0.0, le=100.0)
    flame_stable: bool = Field(default=True)
    pressure_stable: bool = Field(default=True)
    o2_percent: float = Field(..., ge=0.0, le=21.0)
    co_ppm: float = Field(default=0.0)
    flame_temp_c: Optional[float] = Field(default=None)
    stack_temp_c: Optional[float] = Field(default=None)


class Anomaly(BaseModel):
    """Detected combustion anomaly."""
    anomaly_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    anomaly_type: AnomalyType
    severity: str
    anomaly_score: float = Field(..., ge=-1.0, le=1.0)
    affected_variables: List[str] = Field(default_factory=list)
    deviation_magnitude: float = Field(default=0.0)
    probable_causes: List[str] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)
    provenance: Optional[ProvenanceRecord] = Field(default=None)


class FlameStabilityPredictor:
    """Flame stability prediction using gradient boosting with calibrated probabilities."""

    FEATURE_NAMES = [
        'excess_air_percent', 'flame_temperature_c', 'fuel_flow_rate_kg_s',
        'air_flow_rate_kg_s', 'furnace_pressure_kpa', 'fuel_pressure_kpa',
        'air_preheat_temp_c', 'stoichiometric_ratio', 'o2_stack_percent',
        'co_ppm', 'flame_temp_std_1min', 'pressure_oscillation_hz',
        'load_change_rate_percent_min'
    ]

    def __init__(self, model_type: ModelType = ModelType.XGBOOST,
                 model_path: Optional[Path] = None, random_seed: int = 42,
                 calibration_method: str = "isotonic"):
        """Initialize FlameStabilityPredictor."""
        self.model_type = model_type
        self.random_seed = random_seed
        self._classifier = None
        self._calibrator = None
        self._anomaly_detector = None
        self._scaler = None
        self._model_id = f"{model_type.value}_{uuid4().hex[:8]}"
        self._is_fitted = False
        self._feature_importance: Dict[str, float] = {}

        if model_path and model_path.exists():
            self._load_model(model_path)
        else:
            self._initialize_default_models()

    def _initialize_default_models(self) -> None:
        """Initialize default model architectures."""
        if self.model_type == ModelType.XGBOOST and XGBOOST_AVAILABLE:
            self._classifier = xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=self.random_seed)
        elif self.model_type == ModelType.LIGHTGBM and LIGHTGBM_AVAILABLE:
            self._classifier = lgb.LGBMClassifier(n_estimators=100, max_depth=6, random_state=self.random_seed)
        if SKLEARN_AVAILABLE:
            self._anomaly_detector = IsolationForest(n_estimators=100, contamination=0.05, random_state=self.random_seed)
            self._scaler = StandardScaler()

    def predict_instability_risk(self, features: StabilityFeatures) -> InstabilityPrediction:
        """Predict flame instability risk with uncertainty quantification."""
        start_time = time.time()
        feature_vector = np.array([getattr(features, n, 0) or 0 for n in self.FEATURE_NAMES], dtype=np.float64)

        if self._scaler and hasattr(self._scaler, 'mean_'):
            feature_vector_scaled = self._scaler.transform(feature_vector.reshape(1, -1))
        else:
            feature_vector_scaled = feature_vector.reshape(1, -1)

        if self._classifier and self._is_fitted:
            raw_proba = self._classifier.predict_proba(feature_vector_scaled)[0, 1]
            calibrated_proba = float(self._calibrator.predict([[raw_proba]])[0]) if self._calibrator else float(raw_proba)
        else:
            calibrated_proba = self._physics_based_risk_estimate(features)

        uncertainty = min(1.0, 4.0 * calibrated_proba * (1.0 - calibrated_proba) + 0.05)
        z, se = 1.645, uncertainty * 0.25
        conf_interval = ConfidenceInterval(lower=max(0.0, calibrated_proba - z * se), upper=min(1.0, calibrated_proba + z * se))
        regime = next((r for r, (lo, hi) in REGIME_THRESHOLDS.items() if lo <= calibrated_proba < hi), OperatingRegime.CRITICAL)
        risk_cat = "LOW" if calibrated_proba < 0.15 else "MEDIUM" if calibrated_proba < 0.4 else "HIGH" if calibrated_proba < 0.7 else "CRITICAL"
        computation_time_ms = (time.time() - start_time) * 1000

        return InstabilityPrediction(
            instability_risk=calibrated_proba, risk_category=risk_cat, confidence_interval=conf_interval,
            prediction_uncertainty=uncertainty, operating_regime=regime,
            top_risk_factors=[{"feature": k, "importance": v} for k, v in sorted(self._feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]],
            recommended_actions=self._generate_recommendations(regime),
            provenance=ProvenanceRecord.create("instability_risk_prediction", features.dict(), {"risk": calibrated_proba}, self._model_id, computation_time_ms),
            computation_time_ms=computation_time_ms
        )

    def classify_operating_regime(self, state: BurnerState) -> OperatingRegime:
        """Classify the current operating regime based on burner state."""
        features = StabilityFeatures(
            excess_air_percent=state.excess_air_percent, flame_temperature_c=state.flame_temp_c or 1200.0,
            fuel_flow_rate_kg_s=state.firing_rate_percent * 0.01, air_flow_rate_kg_s=state.excess_air_percent * 0.17,
            o2_stack_percent=state.o2_percent, co_ppm=state.co_ppm, burner_id=state.burner_id
        )
        prediction = self.predict_instability_risk(features)
        if not state.flame_stable:
            return OperatingRegime.UNSTABLE
        return prediction.operating_regime

    def detect_anomalies(self, recent_data: pd.DataFrame) -> List[Anomaly]:
        """Detect anomalies in recent combustion data using Isolation Forest."""
        if recent_data.empty:
            return []
        feature_cols = [c for c in self.FEATURE_NAMES if c in recent_data.columns]
        if not feature_cols:
            return []
        X = recent_data[feature_cols].fillna(0).values
        X_scaled = self._scaler.transform(X) if self._scaler and hasattr(self._scaler, 'mean_') else (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)

        if self._anomaly_detector:
            if not hasattr(self._anomaly_detector, 'offset_'):
                self._anomaly_detector.fit(X_scaled)
            predictions, scores = self._anomaly_detector.predict(X_scaled), self._anomaly_detector.score_samples(X_scaled)
        else:
            z_scores = np.abs(X_scaled)
            predictions, scores = np.where(np.any(z_scores > 3, axis=1), -1, 1), -np.max(z_scores, axis=1)

        anomalies = []
        for idx in np.where(predictions == -1)[0]:
            deviations = np.abs(X_scaled[idx])
            top_indices = np.argsort(deviations)[-3:][::-1]
            affected = [feature_cols[i] for i in top_indices]
            atype = AnomalyType.FLAME_OSCILLATION if any(v in affected for v in ['flame_temp_std_1min', 'pressure_oscillation_hz']) else AnomalyType.UNKNOWN
            severity = "CRITICAL" if scores[idx] < -0.5 else "HIGH" if scores[idx] < -0.3 else "MEDIUM" if scores[idx] < -0.15 else "LOW"
            anomalies.append(Anomaly(anomaly_type=atype, severity=severity, anomaly_score=float(scores[idx]), affected_variables=affected, deviation_magnitude=float(np.max(deviations))))
        return anomalies

    def compute_stability_margin(self, current: Dict[str, float], envelope: Dict[str, float]) -> float:
        """Compute stability margin relative to operating envelope."""
        margins = []
        for param, value in current.items():
            if f"{param}_min" in envelope and f"{param}_max" in envelope:
                pmin, pmax = envelope[f"{param}_min"], envelope[f"{param}_max"]
                if pmax > pmin:
                    margins.append(max(0.0, min(1.0, 1.0 - 2.0 * abs((value - pmin) / (pmax - pmin) - 0.5))))
        return min(margins) if margins else 1.0

    def fit(self, X: pd.DataFrame, y: pd.Series, validation_data=None) -> Dict[str, Any]:
        """Train the stability prediction model."""
        feature_cols = [c for c in self.FEATURE_NAMES if c in X.columns]
        X_train = X[feature_cols].fillna(0).values
        if self._scaler:
            X_train = self._scaler.fit_transform(X_train)
        if self._classifier:
            self._classifier.fit(X_train, y.values)
            if SKLEARN_AVAILABLE:
                self._calibrator = IsotonicRegression(out_of_bounds='clip')
                self._calibrator.fit(self._classifier.predict_proba(X_train)[:, 1], y.values)
            if hasattr(self._classifier, 'feature_importances_'):
                self._feature_importance = {feature_cols[i]: float(v) for i, v in enumerate(self._classifier.feature_importances_)}
            self._is_fitted = True
        if self._anomaly_detector:
            self._anomaly_detector.fit(X_train)
        return {"n_samples": len(X_train), "n_features": len(feature_cols)}

    def save_model(self, path: Path) -> None:
        """Save model to disk."""
        with open(path, 'wb') as f:
            pickle.dump({'classifier': self._classifier, 'calibrator': self._calibrator, 'scaler': self._scaler, 'anomaly_detector': self._anomaly_detector, 'feature_importance': self._feature_importance, 'is_fitted': self._is_fitted}, f)

    def _load_model(self, path: Path) -> None:
        """Load model from disk."""
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            self._classifier, self._calibrator, self._scaler = data.get('classifier'), data.get('calibrator'), data.get('scaler')
            self._anomaly_detector, self._feature_importance = data.get('anomaly_detector'), data.get('feature_importance', {})
            self._is_fitted = data.get('is_fitted', False)
        except Exception:
            self._initialize_default_models()

    def _physics_based_risk_estimate(self, features: StabilityFeatures) -> float:
        """Physics-based fallback risk estimate."""
        risk = 0.3 * min(1.0, abs(features.excess_air_percent - 15.0) / 15.0)
        risk += 0.2 * min(1.0, features.co_ppm / 500.0)
        if features.flame_temp_std_1min:
            risk += 0.15 * min(1.0, features.flame_temp_std_1min / 50.0)
        if features.pressure_oscillation_hz:
            risk += 0.15 * min(1.0, features.pressure_oscillation_hz / 100.0)
        return min(1.0, max(0.0, risk))

    def _generate_recommendations(self, regime: OperatingRegime) -> List[str]:
        """Generate recommendations based on operating regime."""
        recs = {
            OperatingRegime.CRITICAL: ["IMMEDIATE: Reduce firing rate", "CHECK: Fuel supply"],
            OperatingRegime.UNSTABLE: ["ALERT: Monitor closely", "VERIFY: Air-fuel ratio"],
            OperatingRegime.MARGINAL: ["CAUTION: Near stability boundary"]
        }
        return recs.get(regime, [])

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
        """Get feature importance dictionary."""
        return self._feature_importance.copy()
