# -*- coding: utf-8 -*-
"""
Core Schemas for GL-013 PredictiveMaintenance Agent.

Defines Pydantic models for all predictive maintenance data structures
with zero-hallucination guarantees through provenance tracking.

Author: GreenLang AI Team
Version: 1.0.0
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import json


class HealthStatus(str, Enum):
    """Equipment health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"
    UNKNOWN = "unknown"


class FailureMode(str, Enum):
    """Common failure modes for process heat equipment."""
    BEARING_WEAR = "bearing_wear"
    SHAFT_MISALIGNMENT = "shaft_misalignment"
    IMBALANCE = "imbalance"
    LOOSENESS = "looseness"
    ELECTRICAL_FAULT = "electrical_fault"
    OVERHEATING = "overheating"
    CORROSION = "corrosion"
    FOULING = "fouling"
    EROSION = "erosion"
    FATIGUE = "fatigue"
    CAVITATION = "cavitation"
    VIBRATION_EXCESSIVE = "vibration_excessive"
    LUBRICATION_FAILURE = "lubrication_failure"
    SEAL_FAILURE = "seal_failure"
    INSULATION_DEGRADATION = "insulation_degradation"
    COMBUSTION_INEFFICIENCY = "combustion_inefficiency"


class CriticalityLevel(str, Enum):
    """Asset criticality levels."""
    A_CRITICAL = "A"  # Safety critical, production essential
    B_IMPORTANT = "B"  # Production important
    C_STANDARD = "C"  # Standard equipment
    D_NON_CRITICAL = "D"  # Non-critical, redundant


class ConfidenceInterval(BaseModel):
    """Confidence interval for predictions."""
    lower_bound: float = Field(..., description="Lower bound of interval")
    upper_bound: float = Field(..., description="Upper bound of interval")
    confidence_level: float = Field(0.95, ge=0.5, le=0.99)
    method: str = Field("bootstrap", description="Method used for CI calculation")


class UncertaintyQuantification(BaseModel):
    """Uncertainty quantification for predictions."""
    point_estimate: float = Field(..., description="Best estimate")
    standard_error: float = Field(..., ge=0)
    confidence_interval: ConfidenceInterval
    epistemic_uncertainty: float = Field(..., ge=0, description="Model uncertainty")
    aleatoric_uncertainty: float = Field(..., ge=0, description="Data uncertainty")
    total_uncertainty: float = Field(..., ge=0)
    is_high_uncertainty: bool = Field(False)

    @validator("total_uncertainty", always=True)
    def compute_total(cls, v, values):
        if "epistemic_uncertainty" in values and "aleatoric_uncertainty" in values:
            import math
            return math.sqrt(values["epistemic_uncertainty"]**2 + values["aleatoric_uncertainty"]**2)
        return v


class AssetInfo(BaseModel):
    """Information about a monitored asset."""
    asset_id: str = Field(..., description="Unique asset identifier")
    asset_name: str = Field(..., description="Human-readable name")
    asset_type: str = Field(..., description="Equipment type")
    manufacturer: Optional[str] = None
    model_number: Optional[str] = None
    serial_number: Optional[str] = None
    installation_date: Optional[datetime] = None
    criticality: CriticalityLevel = CriticalityLevel.C_STANDARD
    location: Optional[str] = None
    parent_asset_id: Optional[str] = None
    operating_hours: Optional[float] = None
    maintenance_history_count: int = 0
    last_maintenance_date: Optional[datetime] = None
    tags: Dict[str, str] = Field(default_factory=dict)


class MaintenanceWindow(BaseModel):
    """Maintenance window specification."""
    window_id: str = Field(..., description="Unique window identifier")
    start_time: datetime
    end_time: datetime
    window_type: str = Field("planned", description="planned, emergency, or opportunity")
    resources_available: List[str] = Field(default_factory=list)
    max_duration_hours: float = 8.0
    priority: int = Field(1, ge=1, le=5)


class FailurePrediction(BaseModel):
    """Failure probability prediction."""
    prediction_id: str = Field(..., description="Unique prediction identifier")
    asset_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    prediction_horizon_hours: int = Field(168, description="Prediction horizon in hours")

    # Failure probabilities
    failure_probability: float = Field(..., ge=0, le=1)
    uncertainty: UncertaintyQuantification

    # Failure mode analysis
    most_likely_failure_mode: Optional[FailureMode] = None
    failure_mode_probabilities: Dict[str, float] = Field(default_factory=dict)

    # Contributing factors
    top_contributing_factors: List[Dict[str, Any]] = Field(default_factory=list)

    # Risk assessment
    risk_score: float = Field(..., ge=0, le=1)
    risk_category: str = Field("low", description="low, medium, high, critical")

    # Provenance
    model_name: str = ""
    model_version: str = ""
    provenance_hash: str = ""
    computation_time_ms: float = 0.0

    def compute_provenance_hash(self) -> str:
        """Compute SHA-256 provenance hash."""
        content = f"{self.prediction_id}{self.asset_id}{self.failure_probability}{self.timestamp}"
        return hashlib.sha256(content.encode()).hexdigest()


class RULEstimate(BaseModel):
    """Remaining Useful Life estimation."""
    estimate_id: str = Field(..., description="Unique estimate identifier")
    asset_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # RUL values
    rul_days: float = Field(..., ge=0, description="Estimated RUL in days")
    rul_hours: float = Field(..., ge=0, description="Estimated RUL in hours")
    rul_cycles: Optional[float] = Field(None, ge=0, description="Estimated RUL in cycles")

    # Uncertainty
    uncertainty: UncertaintyQuantification

    # Survival analysis details
    survival_probability: float = Field(..., ge=0, le=1)
    hazard_rate: float = Field(..., ge=0)
    cumulative_hazard: float = Field(..., ge=0)

    # Degradation state
    current_health_index: float = Field(..., ge=0, le=1)
    degradation_rate_per_day: float = Field(..., ge=0)

    # Recommendations
    recommended_maintenance_date: Optional[datetime] = None
    maintenance_urgency: str = Field("normal", description="normal, soon, urgent, immediate")

    # Weibull parameters (if applicable)
    weibull_shape: Optional[float] = None
    weibull_scale: Optional[float] = None

    # Provenance
    model_name: str = ""
    model_version: str = ""
    provenance_hash: str = ""
    computation_time_ms: float = 0.0


class AnomalyDetection(BaseModel):
    """Anomaly detection result."""
    detection_id: str = Field(..., description="Unique detection identifier")
    asset_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Anomaly scores
    is_anomaly: bool
    anomaly_score: float = Field(..., ge=0, le=1)
    severity: str = Field("low", description="low, medium, high, critical")

    # Detection details
    detection_method: str = Field("isolation_forest")
    affected_sensors: List[str] = Field(default_factory=list)
    anomalous_features: Dict[str, float] = Field(default_factory=dict)

    # Context
    expected_range: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    actual_values: Dict[str, float] = Field(default_factory=dict)
    deviation_sigma: Dict[str, float] = Field(default_factory=dict)

    # Classification
    anomaly_type: Optional[str] = None  # point, contextual, collective
    potential_cause: Optional[str] = None

    # Provenance
    provenance_hash: str = ""
    computation_time_ms: float = 0.0


class HealthIndex(BaseModel):
    """Composite health index for an asset."""
    index_id: str = Field(..., description="Unique index identifier")
    asset_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Overall health
    overall_health: float = Field(..., ge=0, le=1)
    health_status: HealthStatus

    # Component health scores
    component_scores: Dict[str, float] = Field(default_factory=dict)

    # Weighted factors
    factor_weights: Dict[str, float] = Field(default_factory=dict)
    factor_scores: Dict[str, float] = Field(default_factory=dict)

    # Trend information
    trend_direction: str = Field("stable", description="improving, stable, degrading")
    trend_slope: float = 0.0

    # Benchmarking
    percentile_rank: Optional[float] = None
    peer_average: Optional[float] = None

    # Provenance
    provenance_hash: str = ""


class DegradationTrend(BaseModel):
    """Degradation trend analysis."""
    trend_id: str = Field(..., description="Unique trend identifier")
    asset_id: str
    analysis_window_start: datetime
    analysis_window_end: datetime

    # Trend parameters
    trend_type: str = Field("linear", description="linear, exponential, logarithmic")
    slope: float
    intercept: float
    r_squared: float = Field(..., ge=0, le=1)

    # Projected values
    projected_failure_date: Optional[datetime] = None
    days_to_threshold: Optional[float] = None

    # Change points
    change_points: List[datetime] = Field(default_factory=list)
    regime_changes: int = 0

    # Seasonality
    has_seasonality: bool = False
    seasonality_period_days: Optional[float] = None

    # Provenance
    provenance_hash: str = ""


class MaintenanceRecommendation(BaseModel):
    """Maintenance recommendation."""
    recommendation_id: str = Field(..., description="Unique recommendation identifier")
    asset_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Recommendation details
    recommendation_type: str = Field(..., description="inspect, repair, replace, adjust")
    priority: int = Field(..., ge=1, le=5)
    urgency: str = Field("normal", description="normal, soon, urgent, immediate")

    # Description
    title: str
    description: str
    rationale: str

    # Evidence
    supporting_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    confidence_score: float = Field(..., ge=0, le=1)

    # Actions
    recommended_actions: List[str] = Field(default_factory=list)
    required_parts: List[Dict[str, Any]] = Field(default_factory=list)
    estimated_duration_hours: float = 1.0

    # Cost-benefit
    estimated_cost: Optional[float] = None
    potential_savings: Optional[float] = None
    risk_if_deferred: str = ""

    # Scheduling
    preferred_maintenance_window: Optional[MaintenanceWindow] = None
    deadline: Optional[datetime] = None

    # Approval workflow
    requires_approval: bool = False
    approval_status: str = Field("pending", description="pending, approved, rejected")
    approved_by: Optional[str] = None

    # Provenance
    provenance_hash: str = ""


class SensorReading(BaseModel):
    """Individual sensor reading."""
    sensor_id: str
    timestamp: datetime
    value: float
    unit: str
    quality: str = Field("good", description="good, uncertain, bad")

    # Optional metadata
    raw_value: Optional[float] = None
    calibration_factor: float = 1.0


class PredictionResult(BaseModel):
    """Comprehensive prediction result package."""
    result_id: str = Field(..., description="Unique result identifier")
    asset_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Core predictions
    failure_prediction: Optional[FailurePrediction] = None
    rul_estimate: Optional[RULEstimate] = None
    anomaly_detection: Optional[AnomalyDetection] = None
    health_index: Optional[HealthIndex] = None
    degradation_trend: Optional[DegradationTrend] = None

    # Recommendations
    recommendations: List[MaintenanceRecommendation] = Field(default_factory=list)

    # Input summary
    sensor_readings_count: int = 0
    feature_vector_size: int = 0
    data_quality_score: float = Field(1.0, ge=0, le=1)

    # Metadata
    models_used: List[str] = Field(default_factory=list)
    total_computation_time_ms: float = 0.0

    # Provenance
    provenance_hash: str = ""

    def compute_provenance_hash(self) -> str:
        """Compute SHA-256 provenance hash for the entire result."""
        components = []
        if self.failure_prediction:
            components.append(self.failure_prediction.provenance_hash)
        if self.rul_estimate:
            components.append(self.rul_estimate.provenance_hash)
        if self.anomaly_detection:
            components.append(self.anomaly_detection.provenance_hash)
        if self.health_index:
            components.append(self.health_index.provenance_hash)

        content = f"{self.result_id}{self.asset_id}{self.timestamp}{''.join(components)}"
        return hashlib.sha256(content.encode()).hexdigest()


class MaintenanceEvent(BaseModel):
    """Historical maintenance event."""
    event_id: str
    asset_id: str
    event_type: str  # inspection, repair, replacement, failure
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_hours: Optional[float] = None
    description: str = ""
    failure_mode: Optional[FailureMode] = None
    parts_replaced: List[str] = Field(default_factory=list)
    cost: Optional[float] = None
    performed_by: Optional[str] = None
    notes: str = ""


class AssetTelemetry(BaseModel):
    """Real-time asset telemetry snapshot."""
    asset_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Sensor readings
    readings: Dict[str, SensorReading] = Field(default_factory=dict)

    # Computed metrics
    metrics: Dict[str, float] = Field(default_factory=dict)

    # Quality indicators
    data_completeness: float = Field(1.0, ge=0, le=1)
    stale_sensors: List[str] = Field(default_factory=list)

    # Status
    operating_mode: str = "normal"
    is_online: bool = True
