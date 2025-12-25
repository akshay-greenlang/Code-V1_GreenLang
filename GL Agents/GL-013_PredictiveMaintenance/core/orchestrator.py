# -*- coding: utf-8 -*-
"""
Orchestrator for GL-013 PredictiveMaintenance Agent.

Coordinates all predictive maintenance analytics including:
- Failure prediction
- RUL estimation
- Anomaly detection
- Health index computation
- Maintenance recommendations

Author: GreenLang AI Team
Version: 1.0.0
"""

import logging
import hashlib
import time
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
import numpy as np

from .config import (
    PredictiveMaintenanceConfig,
    SeverityLevel,
)
from .schemas import (
    AssetInfo,
    AssetTelemetry,
    PredictionResult,
    FailurePrediction,
    RULEstimate,
    AnomalyDetection,
    HealthIndex,
    DegradationTrend,
    MaintenanceRecommendation,
    UncertaintyQuantification,
    ConfidenceInterval,
    HealthStatus,
    MaintenanceWindow,
)

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorState:
    """Internal state for the orchestrator."""
    last_prediction_time: Dict[str, datetime] = field(default_factory=dict)
    prediction_cache: Dict[str, PredictionResult] = field(default_factory=dict)
    asset_registry: Dict[str, AssetInfo] = field(default_factory=dict)
    active_alerts: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    model_versions: Dict[str, str] = field(default_factory=dict)
    prediction_history: List[Dict[str, Any]] = field(default_factory=list)


class PredictiveMaintenanceOrchestrator:
    """
    Main orchestrator for predictive maintenance operations.

    Coordinates:
    - Multi-modal sensor data processing
    - ML model inference for predictions
    - Uncertainty quantification
    - Alert generation and management
    - Maintenance recommendation generation
    - Integration with external systems
    """

    AGENT_ID = "GL-013"
    VERSION = "1.0.0"

    def __init__(
        self,
        config: PredictiveMaintenanceConfig,
        failure_predictor: Optional[Any] = None,
        rul_estimator: Optional[Any] = None,
        anomaly_detector: Optional[Any] = None,
        health_calculator: Optional[Any] = None,
        signal_processor: Optional[Any] = None,
        explainer: Optional[Any] = None,
        track_provenance: bool = True,
    ):
        """
        Initialize the orchestrator.

        Args:
            config: Agent configuration
            failure_predictor: Failure prediction model
            rul_estimator: RUL estimation model
            anomaly_detector: Anomaly detection model
            health_calculator: Health index calculator
            signal_processor: Signal processing module
            explainer: Explainability module
            track_provenance: Enable provenance tracking
        """
        self.config = config
        self.track_provenance = track_provenance

        # Components (can be injected or created)
        self._failure_predictor = failure_predictor
        self._rul_estimator = rul_estimator
        self._anomaly_detector = anomaly_detector
        self._health_calculator = health_calculator
        self._signal_processor = signal_processor
        self._explainer = explainer

        # Internal state
        self._state = OrchestratorState()

        # Callbacks for external integrations
        self._alert_callbacks: List[Callable] = []
        self._cmms_callbacks: List[Callable] = []

        logger.info(
            f"PredictiveMaintenanceOrchestrator initialized: "
            f"agent_id={self.AGENT_ID}, version={self.VERSION}"
        )

    def register_asset(self, asset: AssetInfo) -> None:
        """Register an asset for monitoring."""
        self._state.asset_registry[asset.asset_id] = asset
        self._state.last_prediction_time[asset.asset_id] = None
        logger.info(f"Registered asset: {asset.asset_id} ({asset.asset_name})")

    def get_asset(self, asset_id: str) -> Optional[AssetInfo]:
        """Get registered asset information."""
        return self._state.asset_registry.get(asset_id)

    def list_assets(self) -> List[AssetInfo]:
        """List all registered assets."""
        return list(self._state.asset_registry.values())

    def predict(
        self,
        asset_id: str,
        telemetry: AssetTelemetry,
        include_failure: bool = True,
        include_rul: bool = True,
        include_anomaly: bool = True,
        include_health: bool = True,
        generate_recommendations: bool = True,
    ) -> PredictionResult:
        """
        Generate comprehensive predictions for an asset.

        Args:
            asset_id: Asset identifier
            telemetry: Current telemetry data
            include_failure: Include failure prediction
            include_rul: Include RUL estimation
            include_anomaly: Include anomaly detection
            include_health: Include health index
            generate_recommendations: Generate recommendations

        Returns:
            PredictionResult with all requested predictions
        """
        start_time = time.time()
        timestamp = datetime.now(timezone.utc)

        # Validate asset
        asset = self._state.asset_registry.get(asset_id)
        if not asset:
            raise ValueError(f"Asset not registered: {asset_id}")

        # Check data quality
        data_quality = self._assess_data_quality(telemetry)
        if data_quality < self.config.safety.min_data_completeness:
            logger.warning(
                f"Low data quality for {asset_id}: {data_quality:.2%}"
            )

        # Extract features from telemetry
        features = self._extract_features(telemetry)

        # Generate predictions
        failure_pred = None
        rul_est = None
        anomaly_det = None
        health_idx = None
        degradation = None
        recommendations = []

        if include_failure:
            failure_pred = self._predict_failure(asset_id, features, timestamp)

        if include_rul:
            rul_est = self._estimate_rul(asset_id, features, timestamp)

        if include_anomaly:
            anomaly_det = self._detect_anomalies(asset_id, features, timestamp)

        if include_health:
            health_idx = self._compute_health_index(
                asset_id, features, failure_pred, rul_est, anomaly_det, timestamp
            )
            degradation = self._analyze_degradation(asset_id, health_idx)

        if generate_recommendations:
            recommendations = self._generate_recommendations(
                asset_id, asset, failure_pred, rul_est, anomaly_det, health_idx
            )

        # Build result
        computation_time_ms = (time.time() - start_time) * 1000

        result = PredictionResult(
            result_id=self._generate_id("pred"),
            asset_id=asset_id,
            timestamp=timestamp,
            failure_prediction=failure_pred,
            rul_estimate=rul_est,
            anomaly_detection=anomaly_det,
            health_index=health_idx,
            degradation_trend=degradation,
            recommendations=recommendations,
            sensor_readings_count=len(telemetry.readings),
            feature_vector_size=len(features) if features else 0,
            data_quality_score=data_quality,
            models_used=self._get_models_used(
                include_failure, include_rul, include_anomaly, include_health
            ),
            total_computation_time_ms=computation_time_ms,
        )

        # Compute provenance
        if self.track_provenance:
            result.provenance_hash = result.compute_provenance_hash()

        # Cache result
        self._state.prediction_cache[asset_id] = result
        self._state.last_prediction_time[asset_id] = timestamp

        # Record in history
        self._record_prediction(result)

        # Check for alerts
        self._check_alerts(asset_id, result)

        logger.info(
            f"Prediction completed for {asset_id} in {computation_time_ms:.1f}ms"
        )

        return result

    def get_latest_prediction(self, asset_id: str) -> Optional[PredictionResult]:
        """Get the latest cached prediction for an asset."""
        return self._state.prediction_cache.get(asset_id)

    def get_prediction_history(
        self,
        asset_id: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get prediction history."""
        history = self._state.prediction_history

        if asset_id:
            history = [h for h in history if h.get("asset_id") == asset_id]

        if since:
            history = [
                h for h in history
                if datetime.fromisoformat(h.get("timestamp", "")) > since
            ]

        return history[-limit:]

    def get_active_alerts(
        self,
        asset_id: Optional[str] = None,
        severity_min: Optional[SeverityLevel] = None,
    ) -> List[Dict[str, Any]]:
        """Get active alerts."""
        if asset_id:
            alerts = self._state.active_alerts.get(asset_id, [])
        else:
            alerts = []
            for asset_alerts in self._state.active_alerts.values():
                alerts.extend(asset_alerts)

        if severity_min:
            severity_order = [s.value for s in SeverityLevel]
            min_idx = severity_order.index(severity_min.value)
            alerts = [
                a for a in alerts
                if severity_order.index(a.get("severity", "S0")) >= min_idx
            ]

        return alerts

    def acknowledge_alert(
        self,
        alert_id: str,
        acknowledged_by: str,
        notes: str = "",
    ) -> bool:
        """Acknowledge an alert."""
        for asset_id, alerts in self._state.active_alerts.items():
            for alert in alerts:
                if alert.get("alert_id") == alert_id:
                    alert["acknowledged"] = True
                    alert["acknowledged_by"] = acknowledged_by
                    alert["acknowledged_at"] = datetime.now(timezone.utc).isoformat()
                    alert["notes"] = notes
                    logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
                    return True
        return False

    def register_alert_callback(self, callback: Callable) -> None:
        """Register a callback for alert notifications."""
        self._alert_callbacks.append(callback)

    def register_cmms_callback(self, callback: Callable) -> None:
        """Register a callback for CMMS integration."""
        self._cmms_callbacks.append(callback)

    def get_maintenance_schedule(
        self,
        asset_ids: Optional[List[str]] = None,
        horizon_days: int = 30,
    ) -> List[Dict[str, Any]]:
        """Get recommended maintenance schedule."""
        schedule = []

        target_assets = asset_ids or list(self._state.asset_registry.keys())

        for asset_id in target_assets:
            result = self._state.prediction_cache.get(asset_id)
            if not result:
                continue

            if result.rul_estimate and result.rul_estimate.recommended_maintenance_date:
                maint_date = result.rul_estimate.recommended_maintenance_date
                if maint_date <= datetime.now(timezone.utc) + timedelta(days=horizon_days):
                    schedule.append({
                        "asset_id": asset_id,
                        "recommended_date": maint_date.isoformat(),
                        "urgency": result.rul_estimate.maintenance_urgency,
                        "rul_days": result.rul_estimate.rul_days,
                        "health_index": (
                            result.health_index.overall_health
                            if result.health_index else None
                        ),
                        "recommendations": [
                            {"title": r.title, "priority": r.priority}
                            for r in result.recommendations[:3]
                        ],
                    })

        # Sort by urgency and date
        urgency_order = {"immediate": 0, "urgent": 1, "soon": 2, "normal": 3}
        schedule.sort(key=lambda x: (
            urgency_order.get(x["urgency"], 3),
            x["recommended_date"]
        ))

        return schedule

    def _predict_failure(
        self,
        asset_id: str,
        features: Dict[str, float],
        timestamp: datetime,
    ) -> FailurePrediction:
        """Generate failure prediction."""
        start_time = time.time()

        # Default values if no model is available
        if self._failure_predictor is None:
            failure_prob = 0.1  # Default low probability
            uncertainty = self._create_default_uncertainty(failure_prob, 0.05)
        else:
            # Use actual model prediction
            failure_prob, uncertainty = self._failure_predictor.predict(features)

        # Determine risk category
        if failure_prob >= self.config.alerts.failure_prob_critical:
            risk_category = "critical"
        elif failure_prob >= self.config.alerts.failure_prob_warning:
            risk_category = "high"
        elif failure_prob >= 0.3:
            risk_category = "medium"
        else:
            risk_category = "low"

        computation_time = (time.time() - start_time) * 1000

        prediction = FailurePrediction(
            prediction_id=self._generate_id("fail"),
            asset_id=asset_id,
            timestamp=timestamp,
            prediction_horizon_hours=self.config.model.failure_horizon_hours,
            failure_probability=failure_prob,
            uncertainty=uncertainty,
            risk_score=failure_prob,
            risk_category=risk_category,
            model_name="failure_predictor",
            model_version=self._state.model_versions.get("failure", "1.0.0"),
            computation_time_ms=computation_time,
        )

        if self.track_provenance:
            prediction.provenance_hash = prediction.compute_provenance_hash()

        return prediction

    def _estimate_rul(
        self,
        asset_id: str,
        features: Dict[str, float],
        timestamp: datetime,
    ) -> RULEstimate:
        """Estimate remaining useful life."""
        start_time = time.time()

        # Default values if no model is available
        if self._rul_estimator is None:
            rul_days = 180.0  # Default 6 months
            rul_std = 30.0
            uncertainty = self._create_default_uncertainty(rul_days, rul_std)
        else:
            rul_days, uncertainty = self._rul_estimator.estimate(features)

        # Calculate derived values
        rul_hours = rul_days * 24

        # Survival probability (simplified)
        survival_prob = min(1.0, rul_days / self.config.model.rul_horizon_days)

        # Hazard rate (simplified exponential model)
        hazard_rate = 1.0 / max(1.0, rul_days)

        # Health index based on RUL
        health_index = min(1.0, rul_days / 365.0)

        # Degradation rate
        degradation_rate = 1.0 / max(1.0, rul_days)

        # Maintenance urgency
        if rul_days <= self.config.alerts.rul_critical_days:
            urgency = "immediate"
        elif rul_days <= self.config.alerts.rul_warning_days:
            urgency = "urgent"
        elif rul_days <= 60:
            urgency = "soon"
        else:
            urgency = "normal"

        # Recommended maintenance date
        safety_margin_days = min(7, rul_days * 0.1)
        recommended_date = timestamp + timedelta(days=max(1, rul_days - safety_margin_days))

        computation_time = (time.time() - start_time) * 1000

        estimate = RULEstimate(
            estimate_id=self._generate_id("rul"),
            asset_id=asset_id,
            timestamp=timestamp,
            rul_days=rul_days,
            rul_hours=rul_hours,
            uncertainty=uncertainty,
            survival_probability=survival_prob,
            hazard_rate=hazard_rate,
            cumulative_hazard=-np.log(max(0.01, survival_prob)),
            current_health_index=health_index,
            degradation_rate_per_day=degradation_rate,
            recommended_maintenance_date=recommended_date,
            maintenance_urgency=urgency,
            model_name="rul_estimator",
            model_version=self._state.model_versions.get("rul", "1.0.0"),
            computation_time_ms=computation_time,
        )

        if self.track_provenance:
            estimate.provenance_hash = hashlib.sha256(
                f"{estimate.estimate_id}{asset_id}{rul_days}".encode()
            ).hexdigest()

        return estimate

    def _detect_anomalies(
        self,
        asset_id: str,
        features: Dict[str, float],
        timestamp: datetime,
    ) -> AnomalyDetection:
        """Detect anomalies in sensor data."""
        start_time = time.time()

        # Default values if no model is available
        if self._anomaly_detector is None:
            anomaly_score = 0.1
            is_anomaly = False
        else:
            anomaly_score, is_anomaly = self._anomaly_detector.detect(features)

        # Determine severity
        if anomaly_score >= self.config.alerts.anomaly_score_critical:
            severity = "critical"
        elif anomaly_score >= self.config.alerts.anomaly_score_warning:
            severity = "high"
        elif anomaly_score >= 0.5:
            severity = "medium"
        else:
            severity = "low"

        computation_time = (time.time() - start_time) * 1000

        detection = AnomalyDetection(
            detection_id=self._generate_id("anom"),
            asset_id=asset_id,
            timestamp=timestamp,
            is_anomaly=is_anomaly,
            anomaly_score=anomaly_score,
            severity=severity,
            detection_method=self.config.model.anomaly_model_type.value,
            computation_time_ms=computation_time,
        )

        if self.track_provenance:
            detection.provenance_hash = hashlib.sha256(
                f"{detection.detection_id}{asset_id}{anomaly_score}".encode()
            ).hexdigest()

        return detection

    def _compute_health_index(
        self,
        asset_id: str,
        features: Dict[str, float],
        failure_pred: Optional[FailurePrediction],
        rul_est: Optional[RULEstimate],
        anomaly_det: Optional[AnomalyDetection],
        timestamp: datetime,
    ) -> HealthIndex:
        """Compute composite health index."""
        # Weight factors for different components
        weights = {
            "failure_risk": 0.3,
            "rul_health": 0.3,
            "anomaly_health": 0.2,
            "sensor_health": 0.2,
        }

        scores = {}

        # Failure risk contribution (inverted - high risk = low health)
        if failure_pred:
            scores["failure_risk"] = 1.0 - failure_pred.failure_probability
        else:
            scores["failure_risk"] = 0.9

        # RUL contribution
        if rul_est:
            scores["rul_health"] = rul_est.current_health_index
        else:
            scores["rul_health"] = 0.8

        # Anomaly contribution (inverted)
        if anomaly_det:
            scores["anomaly_health"] = 1.0 - anomaly_det.anomaly_score
        else:
            scores["anomaly_health"] = 0.9

        # Sensor health (based on data quality)
        scores["sensor_health"] = 0.95  # Default high if no issues

        # Compute weighted average
        overall_health = sum(
            scores.get(k, 0) * w for k, w in weights.items()
        )

        # Determine status
        if overall_health >= 0.8:
            status = HealthStatus.HEALTHY
        elif overall_health >= 0.6:
            status = HealthStatus.DEGRADED
        elif overall_health >= 0.4:
            status = HealthStatus.WARNING
        elif overall_health >= 0.2:
            status = HealthStatus.CRITICAL
        else:
            status = HealthStatus.FAILED

        # Trend analysis (simplified)
        cached = self._state.prediction_cache.get(asset_id)
        if cached and cached.health_index:
            prev_health = cached.health_index.overall_health
            if overall_health > prev_health + 0.02:
                trend = "improving"
                slope = overall_health - prev_health
            elif overall_health < prev_health - 0.02:
                trend = "degrading"
                slope = overall_health - prev_health
            else:
                trend = "stable"
                slope = 0.0
        else:
            trend = "stable"
            slope = 0.0

        health_index = HealthIndex(
            index_id=self._generate_id("health"),
            asset_id=asset_id,
            timestamp=timestamp,
            overall_health=overall_health,
            health_status=status,
            component_scores=scores,
            factor_weights=weights,
            factor_scores=scores,
            trend_direction=trend,
            trend_slope=slope,
        )

        if self.track_provenance:
            health_index.provenance_hash = hashlib.sha256(
                f"{health_index.index_id}{asset_id}{overall_health}".encode()
            ).hexdigest()

        return health_index

    def _analyze_degradation(
        self,
        asset_id: str,
        health_idx: HealthIndex,
    ) -> Optional[DegradationTrend]:
        """Analyze degradation trend."""
        # Get historical health values
        history = [
            h for h in self._state.prediction_history
            if h.get("asset_id") == asset_id and "health_index" in h
        ]

        if len(history) < 3:
            return None

        # Simple linear trend analysis
        timestamps = []
        health_values = []

        for h in history[-30:]:  # Last 30 predictions
            if "health_index" in h:
                timestamps.append(
                    datetime.fromisoformat(h["timestamp"])
                )
                health_values.append(h["health_index"])

        if len(health_values) < 3:
            return None

        # Convert to numeric time (days from first timestamp)
        t0 = timestamps[0]
        x = np.array([(t - t0).total_seconds() / 86400 for t in timestamps])
        y = np.array(health_values)

        # Linear regression
        n = len(x)
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (
            n * np.sum(x**2) - np.sum(x)**2 + 1e-10
        )
        intercept = (np.sum(y) - slope * np.sum(x)) / n

        # R-squared
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - (ss_res / (ss_tot + 1e-10))

        # Project failure date (when health reaches 0.2)
        threshold = 0.2
        if slope < 0:
            days_to_threshold = (threshold - health_idx.overall_health) / slope
            projected_failure = datetime.now(timezone.utc) + timedelta(days=days_to_threshold)
        else:
            days_to_threshold = None
            projected_failure = None

        return DegradationTrend(
            trend_id=self._generate_id("trend"),
            asset_id=asset_id,
            analysis_window_start=timestamps[0],
            analysis_window_end=timestamps[-1],
            trend_type="linear",
            slope=float(slope),
            intercept=float(intercept),
            r_squared=float(max(0, min(1, r_squared))),
            projected_failure_date=projected_failure,
            days_to_threshold=float(days_to_threshold) if days_to_threshold else None,
        )

    def _generate_recommendations(
        self,
        asset_id: str,
        asset: AssetInfo,
        failure_pred: Optional[FailurePrediction],
        rul_est: Optional[RULEstimate],
        anomaly_det: Optional[AnomalyDetection],
        health_idx: Optional[HealthIndex],
    ) -> List[MaintenanceRecommendation]:
        """Generate maintenance recommendations."""
        recommendations = []
        timestamp = datetime.now(timezone.utc)

        # Critical failure risk
        if failure_pred and failure_pred.failure_probability >= 0.7:
            recommendations.append(MaintenanceRecommendation(
                recommendation_id=self._generate_id("rec"),
                asset_id=asset_id,
                timestamp=timestamp,
                recommendation_type="inspect",
                priority=1,
                urgency="immediate",
                title=f"Critical Inspection Required: {asset.asset_name}",
                description=(
                    f"High failure probability detected ({failure_pred.failure_probability:.1%}). "
                    "Immediate inspection recommended to prevent unplanned downtime."
                ),
                rationale="Failure probability exceeds critical threshold",
                confidence_score=1.0 - failure_pred.uncertainty.total_uncertainty,
                recommended_actions=[
                    "Stop equipment if safe to do so",
                    "Perform visual inspection",
                    "Check vibration levels",
                    "Review recent operating parameters",
                ],
                requires_approval=True,
            ))

        # Low RUL
        if rul_est and rul_est.rul_days <= 14:
            recommendations.append(MaintenanceRecommendation(
                recommendation_id=self._generate_id("rec"),
                asset_id=asset_id,
                timestamp=timestamp,
                recommendation_type="repair",
                priority=2,
                urgency="urgent",
                title=f"Schedule Maintenance: {asset.asset_name}",
                description=(
                    f"Remaining useful life estimated at {rul_est.rul_days:.0f} days. "
                    "Schedule maintenance within the next week."
                ),
                rationale=f"RUL below warning threshold of {self.config.alerts.rul_warning_days} days",
                confidence_score=1.0 - rul_est.uncertainty.total_uncertainty,
                recommended_actions=[
                    "Schedule maintenance window",
                    "Order replacement parts",
                    "Prepare maintenance procedures",
                ],
                deadline=rul_est.recommended_maintenance_date,
            ))

        # Anomaly detected
        if anomaly_det and anomaly_det.is_anomaly and anomaly_det.severity in ["high", "critical"]:
            recommendations.append(MaintenanceRecommendation(
                recommendation_id=self._generate_id("rec"),
                asset_id=asset_id,
                timestamp=timestamp,
                recommendation_type="inspect",
                priority=2,
                urgency="soon",
                title=f"Investigate Anomaly: {asset.asset_name}",
                description=(
                    f"Anomalous behavior detected (score: {anomaly_det.anomaly_score:.2f}). "
                    "Investigate root cause to prevent potential issues."
                ),
                rationale="Anomaly score exceeds warning threshold",
                confidence_score=anomaly_det.anomaly_score,
                recommended_actions=[
                    "Review sensor readings",
                    "Check for operational changes",
                    "Compare with historical patterns",
                ],
            ))

        # Degrading health
        if health_idx and health_idx.trend_direction == "degrading" and health_idx.overall_health < 0.6:
            recommendations.append(MaintenanceRecommendation(
                recommendation_id=self._generate_id("rec"),
                asset_id=asset_id,
                timestamp=timestamp,
                recommendation_type="adjust",
                priority=3,
                urgency="normal",
                title=f"Address Degradation: {asset.asset_name}",
                description=(
                    f"Health index declining (current: {health_idx.overall_health:.1%}). "
                    "Consider preventive actions to slow degradation."
                ),
                rationale="Continuous health degradation detected",
                confidence_score=0.8,
                recommended_actions=[
                    "Review operating conditions",
                    "Check lubrication schedule",
                    "Verify alignment and balance",
                ],
            ))

        return recommendations

    def _extract_features(self, telemetry: AssetTelemetry) -> Dict[str, float]:
        """Extract features from telemetry data."""
        features = {}

        # Extract values from sensor readings
        for sensor_id, reading in telemetry.readings.items():
            features[f"{sensor_id}_value"] = reading.value

        # Add computed metrics
        features.update(telemetry.metrics)

        return features

    def _assess_data_quality(self, telemetry: AssetTelemetry) -> float:
        """Assess data quality of telemetry."""
        if not telemetry.readings:
            return 0.0

        good_count = sum(
            1 for r in telemetry.readings.values()
            if r.quality == "good"
        )

        return good_count / len(telemetry.readings)

    def _create_default_uncertainty(
        self,
        point_estimate: float,
        std_error: float,
    ) -> UncertaintyQuantification:
        """Create default uncertainty quantification."""
        return UncertaintyQuantification(
            point_estimate=point_estimate,
            standard_error=std_error,
            confidence_interval=ConfidenceInterval(
                lower_bound=point_estimate - 1.96 * std_error,
                upper_bound=point_estimate + 1.96 * std_error,
                confidence_level=0.95,
            ),
            epistemic_uncertainty=std_error * 0.6,
            aleatoric_uncertainty=std_error * 0.4,
            total_uncertainty=std_error,
            is_high_uncertainty=std_error > self.config.safety.max_prediction_uncertainty,
        )

    def _get_models_used(
        self,
        include_failure: bool,
        include_rul: bool,
        include_anomaly: bool,
        include_health: bool,
    ) -> List[str]:
        """Get list of models used in prediction."""
        models = []
        if include_failure:
            models.append(f"failure_{self.config.model.failure_model_type.value}")
        if include_rul:
            models.append(f"rul_{self.config.model.rul_model_type.value}")
        if include_anomaly:
            models.append(f"anomaly_{self.config.model.anomaly_model_type.value}")
        if include_health:
            models.append("health_composite")
        return models

    def _check_alerts(self, asset_id: str, result: PredictionResult) -> None:
        """Check prediction results and generate alerts if needed."""
        alerts = []
        timestamp = datetime.now(timezone.utc)

        # Check failure probability
        if result.failure_prediction:
            fp = result.failure_prediction
            if fp.failure_probability >= self.config.alerts.failure_prob_critical:
                alerts.append({
                    "alert_id": self._generate_id("alert"),
                    "asset_id": asset_id,
                    "severity": "S2",
                    "type": "failure_risk",
                    "title": "Critical Failure Risk",
                    "message": f"Failure probability: {fp.failure_probability:.1%}",
                    "timestamp": timestamp.isoformat(),
                    "acknowledged": False,
                })

        # Check RUL
        if result.rul_estimate:
            rul = result.rul_estimate
            if rul.rul_days <= self.config.alerts.rul_critical_days:
                alerts.append({
                    "alert_id": self._generate_id("alert"),
                    "asset_id": asset_id,
                    "severity": "S2",
                    "type": "rul_critical",
                    "title": "Critical RUL",
                    "message": f"Remaining useful life: {rul.rul_days:.0f} days",
                    "timestamp": timestamp.isoformat(),
                    "acknowledged": False,
                })

        # Check anomalies
        if result.anomaly_detection and result.anomaly_detection.is_anomaly:
            ad = result.anomaly_detection
            if ad.severity == "critical":
                alerts.append({
                    "alert_id": self._generate_id("alert"),
                    "asset_id": asset_id,
                    "severity": "S2",
                    "type": "anomaly",
                    "title": "Critical Anomaly Detected",
                    "message": f"Anomaly score: {ad.anomaly_score:.2f}",
                    "timestamp": timestamp.isoformat(),
                    "acknowledged": False,
                })

        # Store and notify
        if alerts:
            if asset_id not in self._state.active_alerts:
                self._state.active_alerts[asset_id] = []
            self._state.active_alerts[asset_id].extend(alerts)

            # Notify callbacks
            for callback in self._alert_callbacks:
                try:
                    callback(alerts)
                except Exception as e:
                    logger.error(f"Alert callback error: {e}")

    def _record_prediction(self, result: PredictionResult) -> None:
        """Record prediction in history."""
        record = {
            "timestamp": result.timestamp.isoformat(),
            "asset_id": result.asset_id,
            "result_id": result.result_id,
            "failure_probability": (
                result.failure_prediction.failure_probability
                if result.failure_prediction else None
            ),
            "rul_days": (
                result.rul_estimate.rul_days
                if result.rul_estimate else None
            ),
            "anomaly_score": (
                result.anomaly_detection.anomaly_score
                if result.anomaly_detection else None
            ),
            "health_index": (
                result.health_index.overall_health
                if result.health_index else None
            ),
            "provenance_hash": result.provenance_hash,
        }

        self._state.prediction_history.append(record)

        # Limit history size
        max_history = 10000
        if len(self._state.prediction_history) > max_history:
            self._state.prediction_history = self._state.prediction_history[-max_history:]

    def _generate_id(self, prefix: str) -> str:
        """Generate unique ID with prefix."""
        import uuid
        return f"{prefix}_{uuid.uuid4().hex[:12]}"
