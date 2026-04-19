"""
MLOps Schemas - Pydantic models for MLOps components.

This module defines all data models used across the MLOps pipeline framework
for GreenLang Process Heat agents, ensuring type safety and validation.

Example:
    >>> from greenlang.ml.mlops.schemas import ModelInfo, DriftReport
    >>> model_info = ModelInfo(name="heat_predictor", version="1.0.0", stage="production")
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator


# =============================================================================
# Enums
# =============================================================================

class ModelStage(str, Enum):
    """Model deployment stages."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class DriftType(str, Enum):
    """Types of drift detected."""

    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    FEATURE_DRIFT = "feature_drift"
    LABEL_DRIFT = "label_drift"


class DriftSeverity(str, Enum):
    """Severity levels for drift detection."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertLevel(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ExperimentStatus(str, Enum):
    """A/B experiment status."""

    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class RetrainingTriggerType(str, Enum):
    """Types of retraining triggers."""

    DRIFT = "drift"
    SCHEDULE = "schedule"
    DATA_VOLUME = "data_volume"
    MANUAL = "manual"
    PERFORMANCE = "performance"


# =============================================================================
# Model Registry Schemas
# =============================================================================

class ModelMetadata(BaseModel):
    """Metadata for a registered model."""

    training_date: datetime = Field(..., description="When the model was trained")
    training_duration_seconds: Optional[float] = Field(
        None, description="Training duration in seconds"
    )
    metrics: Dict[str, float] = Field(
        default_factory=dict, description="Model performance metrics"
    )
    hyperparameters: Dict[str, Any] = Field(
        default_factory=dict, description="Model hyperparameters"
    )
    data_hash: str = Field(..., description="SHA-256 hash of training data")
    data_version: Optional[str] = Field(None, description="Version of training data")
    features: List[str] = Field(default_factory=list, description="Feature names used")
    target: Optional[str] = Field(None, description="Target variable name")
    framework: Optional[str] = Field(None, description="ML framework used")
    framework_version: Optional[str] = Field(None, description="Framework version")
    description: Optional[str] = Field(None, description="Model description")
    tags: Dict[str, str] = Field(default_factory=dict, description="Custom tags")

    class Config:
        """Pydantic configuration."""

        json_encoders = {datetime: lambda v: v.isoformat()}


class ModelInfo(BaseModel):
    """Information about a registered model."""

    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    stage: ModelStage = Field(
        default=ModelStage.DEVELOPMENT, description="Deployment stage"
    )
    artifact_path: str = Field(..., description="Path to model artifact")
    artifact_hash: str = Field(..., description="SHA-256 hash of model artifact")
    metadata: ModelMetadata = Field(..., description="Model metadata")
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Registration timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow, description="Last update timestamp"
    )
    created_by: Optional[str] = Field(None, description="User who registered the model")

    @validator("version")
    def validate_version(cls, v: str) -> str:
        """Validate semantic version format."""
        parts = v.split(".")
        if len(parts) < 2:
            raise ValueError("Version must be in semantic format (e.g., 1.0.0)")
        return v

    class Config:
        """Pydantic configuration."""

        json_encoders = {datetime: lambda v: v.isoformat()}


class ModelVersionHistory(BaseModel):
    """Version history for a model."""

    name: str = Field(..., description="Model name")
    versions: List[ModelInfo] = Field(
        default_factory=list, description="List of model versions"
    )
    current_production: Optional[str] = Field(
        None, description="Current production version"
    )
    current_staging: Optional[str] = Field(None, description="Current staging version")


# =============================================================================
# Drift Detection Schemas
# =============================================================================

class FeatureDriftResult(BaseModel):
    """Drift detection result for a single feature."""

    feature_name: str = Field(..., description="Name of the feature")
    drift_detected: bool = Field(..., description="Whether drift was detected")
    drift_score: float = Field(
        ..., ge=0.0, le=1.0, description="Drift score (0-1)"
    )
    statistic: float = Field(..., description="Test statistic value")
    p_value: Optional[float] = Field(None, description="P-value of statistical test")
    test_used: str = Field(..., description="Statistical test used")
    reference_mean: Optional[float] = Field(None, description="Mean in reference data")
    current_mean: Optional[float] = Field(None, description="Mean in current data")
    reference_std: Optional[float] = Field(None, description="Std in reference data")
    current_std: Optional[float] = Field(None, description="Std in current data")
    psi: Optional[float] = Field(
        None, description="Population Stability Index"
    )
    kl_divergence: Optional[float] = Field(
        None, description="KL divergence"
    )


class DriftReport(BaseModel):
    """Comprehensive drift detection report."""

    report_id: str = Field(..., description="Unique report identifier")
    model_name: str = Field(..., description="Model being monitored")
    model_version: str = Field(..., description="Model version")
    drift_type: DriftType = Field(..., description="Type of drift analyzed")
    drift_detected: bool = Field(..., description="Whether significant drift detected")
    overall_drift_score: float = Field(
        ..., ge=0.0, le=1.0, description="Overall drift score"
    )
    severity: DriftSeverity = Field(..., description="Drift severity level")
    feature_results: List[FeatureDriftResult] = Field(
        default_factory=list, description="Per-feature drift results"
    )
    drifted_features: List[str] = Field(
        default_factory=list, description="Features with detected drift"
    )
    reference_data_size: int = Field(..., description="Reference dataset size")
    current_data_size: int = Field(..., description="Current dataset size")
    reference_data_hash: str = Field(..., description="Hash of reference data")
    current_data_hash: str = Field(..., description="Hash of current data")
    analysis_timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When analysis was performed"
    )
    recommendations: List[str] = Field(
        default_factory=list, description="Recommended actions"
    )

    class Config:
        """Pydantic configuration."""

        json_encoders = {datetime: lambda v: v.isoformat()}


class ConceptDriftMetrics(BaseModel):
    """Metrics for concept drift detection."""

    accuracy_degradation: float = Field(
        ..., description="Accuracy change from baseline"
    )
    mae_degradation: float = Field(..., description="MAE change from baseline")
    rmse_degradation: float = Field(..., description="RMSE change from baseline")
    error_distribution_shift: float = Field(
        ..., description="Shift in error distribution"
    )
    residual_correlation: float = Field(
        ..., description="Correlation in residuals"
    )


# =============================================================================
# A/B Testing Schemas
# =============================================================================

class ExperimentConfig(BaseModel):
    """Configuration for an A/B experiment."""

    experiment_id: str = Field(..., description="Unique experiment identifier")
    experiment_name: str = Field(..., description="Human-readable name")
    champion_model: str = Field(..., description="Champion model name:version")
    challenger_model: str = Field(..., description="Challenger model name:version")
    traffic_split: float = Field(
        ..., ge=0.0, le=1.0, description="Traffic fraction to challenger (0-1)"
    )
    min_samples: int = Field(
        default=1000, ge=100, description="Minimum samples before analysis"
    )
    max_duration_hours: int = Field(
        default=168, ge=1, description="Maximum experiment duration in hours"
    )
    success_metric: str = Field(
        default="mae", description="Primary metric for comparison"
    )
    significance_level: float = Field(
        default=0.05, ge=0.001, le=0.1, description="Statistical significance level"
    )
    auto_promote: bool = Field(
        default=False, description="Auto-promote challenger if it wins"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Experiment creation time"
    )

    class Config:
        """Pydantic configuration."""

        json_encoders = {datetime: lambda v: v.isoformat()}


class ExperimentOutcome(BaseModel):
    """Single outcome record for an A/B experiment."""

    outcome_id: str = Field(..., description="Unique outcome identifier")
    experiment_id: str = Field(..., description="Experiment identifier")
    model_used: str = Field(..., description="Model that made prediction")
    prediction: float = Field(..., description="Model prediction")
    actual: Optional[float] = Field(None, description="Actual value (when available)")
    features_hash: str = Field(..., description="Hash of input features")
    latency_ms: float = Field(..., ge=0, description="Prediction latency")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Outcome timestamp"
    )


class ModelMetricsComparison(BaseModel):
    """Comparison of metrics between two models."""

    model_name: str = Field(..., description="Model identifier")
    sample_count: int = Field(..., description="Number of samples")
    mae: float = Field(..., description="Mean Absolute Error")
    rmse: float = Field(..., description="Root Mean Square Error")
    mape: Optional[float] = Field(None, description="Mean Absolute Percentage Error")
    accuracy: Optional[float] = Field(None, description="Accuracy (for classification)")
    mean_latency_ms: float = Field(..., description="Mean prediction latency")
    p95_latency_ms: float = Field(..., description="95th percentile latency")


class ABTestResult(BaseModel):
    """Results from an A/B test analysis."""

    experiment_id: str = Field(..., description="Experiment identifier")
    status: ExperimentStatus = Field(..., description="Current experiment status")
    champion_metrics: ModelMetricsComparison = Field(..., description="Champion metrics")
    challenger_metrics: ModelMetricsComparison = Field(
        ..., description="Challenger metrics"
    )
    winner: Optional[str] = Field(None, description="Winning model (if determined)")
    improvement_percent: Optional[float] = Field(
        None, description="Challenger improvement percentage"
    )
    p_value: Optional[float] = Field(
        None, description="Statistical significance p-value"
    )
    is_significant: bool = Field(
        False, description="Whether result is statistically significant"
    )
    confidence_interval: Optional[tuple] = Field(
        None, description="95% confidence interval for improvement"
    )
    recommendation: str = Field(..., description="Recommended action")
    analysis_timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Analysis timestamp"
    )

    class Config:
        """Pydantic configuration."""

        json_encoders = {datetime: lambda v: v.isoformat()}


# =============================================================================
# Retraining Pipeline Schemas
# =============================================================================

class RetrainingTrigger(BaseModel):
    """Configuration for a retraining trigger."""

    trigger_type: RetrainingTriggerType = Field(..., description="Type of trigger")
    enabled: bool = Field(default=True, description="Whether trigger is enabled")
    drift_threshold: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Drift threshold for drift-based trigger"
    )
    schedule: Optional[str] = Field(
        None, description="Cron expression for schedule-based trigger"
    )
    min_samples: Optional[int] = Field(
        None, ge=100, description="Minimum new samples for data volume trigger"
    )
    performance_threshold: Optional[float] = Field(
        None, description="Performance degradation threshold"
    )


class RetrainingConfig(BaseModel):
    """Configuration for model retraining."""

    model_name: str = Field(..., description="Model to retrain")
    triggers: List[RetrainingTrigger] = Field(
        default_factory=list, description="Retraining triggers"
    )
    training_config: Dict[str, Any] = Field(
        default_factory=dict, description="Training hyperparameters"
    )
    validation_split: float = Field(
        default=0.2, ge=0.1, le=0.5, description="Validation data fraction"
    )
    min_improvement: float = Field(
        default=0.01, ge=0.0, description="Minimum improvement required to deploy"
    )
    auto_deploy: bool = Field(
        default=False, description="Auto-deploy if validation passes"
    )
    max_training_time_seconds: int = Field(
        default=3600, ge=60, description="Maximum training time"
    )
    notification_emails: List[str] = Field(
        default_factory=list, description="Emails for notifications"
    )


class RetrainingResult(BaseModel):
    """Result of a retraining run."""

    retraining_id: str = Field(..., description="Unique retraining identifier")
    model_name: str = Field(..., description="Model that was retrained")
    old_version: str = Field(..., description="Previous model version")
    new_version: str = Field(..., description="New model version")
    trigger_type: RetrainingTriggerType = Field(..., description="What triggered retraining")
    trigger_reason: str = Field(..., description="Detailed trigger reason")
    training_started_at: datetime = Field(..., description="Training start time")
    training_completed_at: datetime = Field(..., description="Training completion time")
    training_duration_seconds: float = Field(..., description="Training duration")
    old_model_metrics: Dict[str, float] = Field(..., description="Old model metrics")
    new_model_metrics: Dict[str, float] = Field(..., description="New model metrics")
    improvement: Dict[str, float] = Field(..., description="Metric improvements")
    validation_passed: bool = Field(..., description="Whether validation passed")
    deployed: bool = Field(..., description="Whether new model was deployed")
    deployment_stage: Optional[ModelStage] = Field(
        None, description="Stage deployed to"
    )

    class Config:
        """Pydantic configuration."""

        json_encoders = {datetime: lambda v: v.isoformat()}


# =============================================================================
# Monitoring Schemas
# =============================================================================

class PredictionLog(BaseModel):
    """Log entry for a single prediction."""

    prediction_id: str = Field(..., description="Unique prediction identifier")
    model_name: str = Field(..., description="Model that made prediction")
    model_version: str = Field(..., description="Model version")
    features: Dict[str, Any] = Field(..., description="Input features")
    features_hash: str = Field(..., description="SHA-256 hash of features")
    prediction: float = Field(..., description="Model prediction")
    confidence: Optional[float] = Field(None, description="Prediction confidence")
    latency_ms: float = Field(..., ge=0, description="Prediction latency")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Prediction timestamp"
    )
    actual: Optional[float] = Field(None, description="Actual value (when available)")
    actual_timestamp: Optional[datetime] = Field(
        None, description="When actual value was recorded"
    )

    class Config:
        """Pydantic configuration."""

        json_encoders = {datetime: lambda v: v.isoformat()}


class PerformanceMetrics(BaseModel):
    """Model performance metrics over a time window."""

    model_name: str = Field(..., description="Model name")
    model_version: str = Field(..., description="Model version")
    window: str = Field(..., description="Time window (e.g., '1h', '24h', '7d')")
    start_time: datetime = Field(..., description="Window start time")
    end_time: datetime = Field(..., description="Window end time")
    prediction_count: int = Field(..., description="Number of predictions")
    predictions_with_actuals: int = Field(
        ..., description="Predictions with actual values"
    )
    mae: Optional[float] = Field(None, description="Mean Absolute Error")
    rmse: Optional[float] = Field(None, description="Root Mean Square Error")
    mape: Optional[float] = Field(None, description="Mean Absolute Percentage Error")
    r2_score: Optional[float] = Field(None, description="R-squared score")
    mean_latency_ms: float = Field(..., description="Mean prediction latency")
    p50_latency_ms: float = Field(..., description="Median latency")
    p95_latency_ms: float = Field(..., description="95th percentile latency")
    p99_latency_ms: float = Field(..., description="99th percentile latency")
    max_latency_ms: float = Field(..., description="Maximum latency")
    throughput_per_second: float = Field(..., description="Predictions per second")
    error_count: int = Field(default=0, description="Number of prediction errors")
    error_rate: float = Field(default=0.0, description="Error rate (0-1)")

    class Config:
        """Pydantic configuration."""

        json_encoders = {datetime: lambda v: v.isoformat()}


class Alert(BaseModel):
    """Alert for model issues."""

    alert_id: str = Field(..., description="Unique alert identifier")
    model_name: str = Field(..., description="Model that triggered alert")
    model_version: str = Field(..., description="Model version")
    level: AlertLevel = Field(..., description="Alert severity level")
    alert_type: str = Field(..., description="Type of alert")
    message: str = Field(..., description="Alert message")
    metric_name: Optional[str] = Field(None, description="Metric that triggered alert")
    metric_value: Optional[float] = Field(None, description="Current metric value")
    threshold: Optional[float] = Field(None, description="Threshold that was exceeded")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Alert timestamp"
    )
    acknowledged: bool = Field(default=False, description="Whether alert was acknowledged")
    acknowledged_by: Optional[str] = Field(None, description="Who acknowledged")
    acknowledged_at: Optional[datetime] = Field(None, description="When acknowledged")
    resolved: bool = Field(default=False, description="Whether alert was resolved")
    resolved_at: Optional[datetime] = Field(None, description="When resolved")

    class Config:
        """Pydantic configuration."""

        json_encoders = {datetime: lambda v: v.isoformat()}


class PrometheusMetric(BaseModel):
    """Prometheus-compatible metric export format."""

    name: str = Field(..., description="Metric name")
    help_text: str = Field(..., description="Metric description")
    type: str = Field(..., description="Metric type (gauge, counter, histogram)")
    labels: Dict[str, str] = Field(default_factory=dict, description="Metric labels")
    value: float = Field(..., description="Metric value")
    timestamp_ms: Optional[int] = Field(None, description="Unix timestamp in ms")


# =============================================================================
# Common Schemas
# =============================================================================

class HealthStatus(BaseModel):
    """Health status for MLOps components."""

    component: str = Field(..., description="Component name")
    status: str = Field(..., description="Health status (healthy, degraded, unhealthy)")
    message: Optional[str] = Field(None, description="Status message")
    last_check: datetime = Field(
        default_factory=datetime.utcnow, description="Last health check time"
    )
    details: Dict[str, Any] = Field(
        default_factory=dict, description="Additional details"
    )


class AuditLogEntry(BaseModel):
    """Audit log entry for MLOps operations."""

    log_id: str = Field(..., description="Unique log identifier")
    operation: str = Field(..., description="Operation performed")
    component: str = Field(..., description="Component that performed operation")
    model_name: Optional[str] = Field(None, description="Related model")
    model_version: Optional[str] = Field(None, description="Related model version")
    user: Optional[str] = Field(None, description="User who performed operation")
    details: Dict[str, Any] = Field(default_factory=dict, description="Operation details")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Operation timestamp"
    )
    provenance_hash: str = Field(..., description="SHA-256 hash for audit trail")

    class Config:
        """Pydantic configuration."""

        json_encoders = {datetime: lambda v: v.isoformat()}
