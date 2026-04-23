"""GL-013 PredictiveMaintenance: Prediction Event Schemas - Version 1.0"""
from __future__ import annotations
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

class UncertaintyMethod(str, Enum):
    MONTE_CARLO = "monte_carlo"
    BOOTSTRAP = "bootstrap"
    ENSEMBLE = "ensemble"
    BAYESIAN = "bayesian"
    CONFORMAL = "conformal"

class CalibrationStatus(str, Enum):
    CALIBRATED = "calibrated"
    NEEDS_CALIBRATION = "needs_calibration"
    UNCALIBRATED = "uncalibrated"
    DEGRADED = "degraded"

class ActionSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class ActionType(str, Enum):
    IMMEDIATE_SHUTDOWN = "immediate_shutdown"
    SCHEDULE_MAINTENANCE = "schedule_maintenance"
    INCREASE_MONITORING = "increase_monitoring"
    ADJUST_OPERATING_PARAMS = "adjust_operating_params"
    NO_ACTION = "no_action"

class PredictionConfidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"

class RULPrediction(BaseModel):
    model_config = ConfigDict(frozen=True)
    p10_hours: float = Field(..., ge=0, description="10th percentile RUL")
    p50_hours: float = Field(..., ge=0, description="Median RUL")
    p90_hours: float = Field(..., ge=0, description="90th percentile RUL")
    mean_hours: Optional[float] = Field(default=None, ge=0)
    std_hours: Optional[float] = Field(default=None, ge=0)
    confidence: PredictionConfidence = PredictionConfidence.MEDIUM
    @model_validator(mode="after")
    def validate_percentile_order(self) -> "RULPrediction":
        if self.p10_hours > self.p50_hours or self.p50_hours > self.p90_hours:
            raise ValueError("Percentiles must be ordered: p10 <= p50 <= p90")
        return self

class FailureProbability(BaseModel):
    model_config = ConfigDict(frozen=True)
    horizon_days: int = Field(..., ge=1, le=365)
    probability: float = Field(..., ge=0, le=1)
    confidence_lower: Optional[float] = Field(default=None, ge=0, le=1)
    confidence_upper: Optional[float] = Field(default=None, ge=0, le=1)
    failure_mode: Optional[str] = None

class UncertaintyQuantification(BaseModel):
    model_config = ConfigDict(frozen=True)
    method: UncertaintyMethod
    calibration_status: CalibrationStatus
    calibration_error: Optional[float] = Field(default=None, ge=0, le=1)
    sharpness: Optional[float] = Field(default=None, ge=0)
    coverage_90: Optional[float] = Field(default=None, ge=0, le=1)

class Explainability(BaseModel):
    model_config = ConfigDict(frozen=True)
    top_features: List[Dict[str, Any]] = Field(default_factory=list)
    shap_values: Optional[Dict[str, float]] = None
    feature_importance: Optional[Dict[str, float]] = None
    explanation_text: Optional[str] = None
    contributing_sensors: List[str] = Field(default_factory=list)

class DataQualityReference(BaseModel):
    model_config = ConfigDict(frozen=True)
    input_quality_score: float = Field(..., ge=0, le=100)
    missing_features: List[str] = Field(default_factory=list)
    imputed_features: List[str] = Field(default_factory=list)
    data_freshness_hours: float = Field(..., ge=0)

class RecommendedAction(BaseModel):
    model_config = ConfigDict(frozen=True)
    action_type: ActionType
    severity: ActionSeverity
    description: str = Field(..., min_length=1)
    deadline_hours: Optional[float] = Field(default=None, ge=0)
    estimated_cost_usd: Optional[float] = Field(default=None, ge=0)
    estimated_downtime_hours: Optional[float] = Field(default=None, ge=0)
    parts_required: List[str] = Field(default_factory=list)
    work_order_template_id: Optional[str] = None

class ModelMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)
    model_id: str
    model_version: str
    model_type: str
    training_date: Optional[datetime] = None
    performance_metrics: Dict[str, float] = Field(default_factory=dict)

class PredictionEvent(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, validate_default=True, extra="forbid")
    schema_version: str = Field(default="1.0")
    prediction_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    asset_id: str = Field(..., min_length=1)
    component_id: Optional[str] = None
    timestamp: datetime
    rul: Optional[RULPrediction] = None
    failure_probability_7d: Optional[FailureProbability] = None
    failure_probability_30d: Optional[FailureProbability] = None
    predicted_failure_mode: Optional[str] = None
    health_index: Optional[float] = Field(default=None, ge=0, le=100)
    uncertainty: Optional[UncertaintyQuantification] = None
    explainability: Optional[Explainability] = None
    data_quality: Optional[DataQualityReference] = None
    recommended_action: Optional[RecommendedAction] = None
    model_metadata: Optional[ModelMetadata] = None
    provenance_hash: Optional[str] = None
    @field_validator("timestamp", mode="before")
    @classmethod
    def ensure_utc(cls, v: Any) -> datetime:
        if isinstance(v, str): v = datetime.fromisoformat(v.replace("Z", "+00:00"))
        if isinstance(v, datetime) and v.tzinfo is None: v = v.replace(tzinfo=timezone.utc)
        return v

class PredictionBatch(BaseModel):
    model_config = ConfigDict(frozen=True)
    batch_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    asset_id: str
    predictions: List[PredictionEvent] = Field(..., min_length=1)
    batch_start: datetime
    batch_end: datetime

PREDICTION_SCHEMAS = {"RULPrediction": RULPrediction, "FailureProbability": FailureProbability, "UncertaintyQuantification": UncertaintyQuantification, "Explainability": Explainability, "DataQualityReference": DataQualityReference, "RecommendedAction": RecommendedAction, "ModelMetadata": ModelMetadata, "PredictionEvent": PredictionEvent, "PredictionBatch": PredictionBatch}
__all__ = ["UncertaintyMethod", "CalibrationStatus", "ActionSeverity", "ActionType", "PredictionConfidence", "RULPrediction", "FailureProbability", "UncertaintyQuantification", "Explainability", "DataQualityReference", "RecommendedAction", "ModelMetadata", "PredictionEvent", "PredictionBatch", "PREDICTION_SCHEMAS"]
