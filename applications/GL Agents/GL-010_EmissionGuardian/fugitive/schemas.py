"""
GL-010 EmissionsGuardian - Fugitive Emissions Detection Schemas

This module defines all data schemas for the fugitive emissions detection
ML module, including sensor readings, detection results, leak events,
and human-in-the-loop review decisions.

Standards Compliance:
    - EPA 40 CFR Part 60 Subpart VVa (LDAR Requirements)
    - EPA Method 21 (Determination of VOC Leaks)

Zero-Hallucination Principle:
    - All data structures include provenance tracking
    - Timestamps in UTC for deterministic processing
    - Decimal types for precise emission rate calculations
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union
import hashlib
import json
import uuid


class SourceType(Enum):
    """Fugitive emission source classification types."""
    VALVE = "valve"
    PUMP = "pump"
    COMPRESSOR = "compressor"
    PRESSURE_RELIEF = "pressure_relief"
    CONNECTOR = "connector"
    FLANGE = "flange"
    TANK = "tank"
    SAMPLING_CONNECTION = "sampling_connection"
    OPEN_ENDED_LINE = "open_ended_line"
    AGITATOR = "agitator"
    UNKNOWN = "unknown"


class RepairStatus(Enum):
    """Leak repair status tracking."""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    DELAY_OF_REPAIR = "delay_of_repair"
    VERIFICATION_REQUIRED = "verification_required"


class ReviewStatus(Enum):
    """Human-in-the-loop review status."""
    PENDING = "pending"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_MORE_INFO = "needs_more_info"
    ESCALATED = "escalated"


class AnomalyType(Enum):
    """Types of anomalies detected."""
    STATISTICAL_OUTLIER = "statistical_outlier"
    CUSUM_DEVIATION = "cusum_deviation"
    MOVING_AVERAGE_DEVIATION = "moving_average_deviation"
    ISOLATION_FOREST = "isolation_forest"
    SEASONAL_ANOMALY = "seasonal_anomaly"
    MULTI_SENSOR_CORRELATION = "multi_sensor_correlation"


class PasquillStabilityClass(Enum):
    """Pasquill-Gifford atmospheric stability classes."""
    A = "A"  # Very unstable
    B = "B"  # Moderately unstable
    C = "C"  # Slightly unstable
    D = "D"  # Neutral
    E = "E"  # Slightly stable
    F = "F"  # Moderately stable


@dataclass
class FugitiveReading:
    """
    Individual fugitive emissions sensor reading.

    Captures all relevant data from a single sensor measurement
    including concentration, meteorological conditions, and location.
    """
    sensor_id: str
    timestamp: datetime
    location_lat: float
    location_lon: float
    concentration_ppm: float
    wind_speed: float  # m/s
    wind_direction: float  # degrees from north (0-360)
    temperature: float  # Celsius
    humidity: float  # percentage (0-100)
    atmospheric_pressure: float  # hPa

    # Optional extended fields
    elevation_m: Optional[float] = None
    sensor_height_m: Optional[float] = None
    measurement_uncertainty_ppm: Optional[float] = None
    quality_flag: Optional[str] = None

    def __post_init__(self):
        """Validate reading parameters."""
        if not -90 <= self.location_lat <= 90:
            raise ValueError(f"Invalid latitude: {self.location_lat}")
        if not -180 <= self.location_lon <= 180:
            raise ValueError(f"Invalid longitude: {self.location_lon}")
        if self.concentration_ppm < 0:
            raise ValueError(f"Concentration cannot be negative: {self.concentration_ppm}")
        if not 0 <= self.wind_direction <= 360:
            raise ValueError(f"Wind direction must be 0-360: {self.wind_direction}")
        if not 0 <= self.humidity <= 100:
            raise ValueError(f"Humidity must be 0-100: {self.humidity}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "sensor_id": self.sensor_id,
            "timestamp": self.timestamp.isoformat(),
            "location_lat": self.location_lat,
            "location_lon": self.location_lon,
            "concentration_ppm": self.concentration_ppm,
            "wind_speed": self.wind_speed,
            "wind_direction": self.wind_direction,
            "temperature": self.temperature,
            "humidity": self.humidity,
            "atmospheric_pressure": self.atmospheric_pressure,
            "elevation_m": self.elevation_m,
            "sensor_height_m": self.sensor_height_m,
            "measurement_uncertainty_ppm": self.measurement_uncertainty_ppm,
            "quality_flag": self.quality_flag,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FugitiveReading":
        """Create from dictionary."""
        data = data.copy()
        if isinstance(data.get("timestamp"), str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)

    def compute_hash(self) -> str:
        """Compute SHA-256 hash for provenance tracking."""
        content = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()


@dataclass
class AnomalyResult:
    """
    Result from anomaly detection algorithm.

    Contains detection metadata, confidence scores, and
    explanatory information for each detected anomaly.
    """
    anomaly_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    anomaly_type: AnomalyType = AnomalyType.STATISTICAL_OUTLIER
    is_anomaly: bool = False
    confidence_score: float = 0.0  # 0.0 to 1.0
    anomaly_score: float = 0.0  # Raw score from algorithm
    threshold_used: float = 0.0

    # Explanation fields
    contributing_features: Dict[str, float] = field(default_factory=dict)
    explanation: str = ""

    # Algorithm metadata
    algorithm_name: str = ""
    algorithm_version: str = ""
    parameters_used: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not 0 <= self.confidence_score <= 1:
            raise ValueError(f"Confidence score must be 0-1: {self.confidence_score}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "anomaly_id": self.anomaly_id,
            "timestamp": self.timestamp.isoformat(),
            "anomaly_type": self.anomaly_type.value,
            "is_anomaly": self.is_anomaly,
            "confidence_score": self.confidence_score,
            "anomaly_score": self.anomaly_score,
            "threshold_used": self.threshold_used,
            "contributing_features": self.contributing_features,
            "explanation": self.explanation,
            "algorithm_name": self.algorithm_name,
            "algorithm_version": self.algorithm_version,
            "parameters_used": self.parameters_used,
        }


@dataclass
class SourceEstimate:
    """
    Estimated source location from plume analysis.

    Contains the estimated emission source coordinates with
    uncertainty bounds based on Gaussian plume modeling.
    """
    estimate_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Estimated location
    latitude: float = 0.0
    longitude: float = 0.0

    # Uncertainty bounds (meters)
    uncertainty_radius_m: float = 0.0
    latitude_uncertainty: float = 0.0
    longitude_uncertainty: float = 0.0

    # Confidence and quality
    confidence_score: float = 0.0
    quality_indicator: str = ""

    # Method metadata
    method_used: str = "gaussian_plume"
    stability_class: Optional[PasquillStabilityClass] = None
    num_sensors_used: int = 0

    # Estimated emission parameters
    estimated_emission_rate_kg_hr: Optional[Decimal] = None
    emission_rate_uncertainty_kg_hr: Optional[Decimal] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "estimate_id": self.estimate_id,
            "timestamp": self.timestamp.isoformat(),
            "latitude": self.latitude,
            "longitude": self.longitude,
            "uncertainty_radius_m": self.uncertainty_radius_m,
            "latitude_uncertainty": self.latitude_uncertainty,
            "longitude_uncertainty": self.longitude_uncertainty,
            "confidence_score": self.confidence_score,
            "quality_indicator": self.quality_indicator,
            "method_used": self.method_used,
            "stability_class": self.stability_class.value if self.stability_class else None,
            "num_sensors_used": self.num_sensors_used,
            "estimated_emission_rate_kg_hr": str(self.estimated_emission_rate_kg_hr) if self.estimated_emission_rate_kg_hr else None,
            "emission_rate_uncertainty_kg_hr": str(self.emission_rate_uncertainty_kg_hr) if self.emission_rate_uncertainty_kg_hr else None,
        }


@dataclass
class DetectionResult:
    """
    Complete detection result for a potential fugitive emission event.

    Aggregates sensor readings, anomaly detection results, source
    classification, and plume analysis into a single result with
    full provenance tracking.
    """
    detection_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Input data
    sensor_readings: List[FugitiveReading] = field(default_factory=list)

    # Detection results
    is_anomaly: bool = False
    confidence_score: float = 0.0
    anomaly_results: List[AnomalyResult] = field(default_factory=list)

    # Classification
    source_type_prediction: SourceType = SourceType.UNKNOWN
    source_type_probabilities: Dict[str, float] = field(default_factory=dict)

    # Plume analysis
    plume_origin_estimate: Optional[SourceEstimate] = None

    # Explanation
    explanation: str = ""
    feature_importances: Dict[str, float] = field(default_factory=dict)

    # Provenance
    provenance_hash: str = ""
    model_version: str = ""
    processing_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Compute provenance hash if not provided."""
        if not self.provenance_hash:
            self.provenance_hash = self._compute_provenance_hash()

    def _compute_provenance_hash(self) -> str:
        """Compute SHA-256 hash of detection inputs for provenance."""
        content = {
            "detection_id": self.detection_id,
            "timestamp": self.timestamp.isoformat(),
            "sensor_readings": [r.to_dict() for r in self.sensor_readings],
            "model_version": self.model_version,
        }
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "detection_id": self.detection_id,
            "timestamp": self.timestamp.isoformat(),
            "sensor_readings": [r.to_dict() for r in self.sensor_readings],
            "is_anomaly": self.is_anomaly,
            "confidence_score": self.confidence_score,
            "anomaly_results": [a.to_dict() for a in self.anomaly_results],
            "source_type_prediction": self.source_type_prediction.value,
            "source_type_probabilities": self.source_type_probabilities,
            "plume_origin_estimate": self.plume_origin_estimate.to_dict() if self.plume_origin_estimate else None,
            "explanation": self.explanation,
            "feature_importances": self.feature_importances,
            "provenance_hash": self.provenance_hash,
            "model_version": self.model_version,
            "processing_metadata": self.processing_metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DetectionResult":
        """Create from dictionary."""
        data = data.copy()
        if isinstance(data.get("timestamp"), str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        if data.get("sensor_readings"):
            data["sensor_readings"] = [FugitiveReading.from_dict(r) for r in data["sensor_readings"]]
        if isinstance(data.get("source_type_prediction"), str):
            data["source_type_prediction"] = SourceType(data["source_type_prediction"])
        return cls(**data)


@dataclass
class LeakEvent:
    """
    Confirmed or suspected leak event for LDAR tracking.

    Represents a leak that requires repair tracking per EPA
    40 CFR Part 60 Subpart VVa requirements.
    """
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    detection_id: str = ""

    # Event status
    confirmed: bool = False
    confirmation_method: str = ""  # e.g., "method_21", "ogi", "manual_inspection"
    confirmation_timestamp: Optional[datetime] = None

    # Emission quantification
    estimated_emission_rate_kg_hr: Decimal = Decimal("0")
    emission_rate_uncertainty_kg_hr: Optional[Decimal] = None

    # Source identification
    source_equipment: str = ""
    equipment_tag: str = ""
    source_type: SourceType = SourceType.UNKNOWN

    # Location
    location_lat: float = 0.0
    location_lon: float = 0.0
    location_description: str = ""

    # Repair tracking
    repair_status: RepairStatus = RepairStatus.PENDING
    repair_deadline: Optional[datetime] = None
    first_attempt_deadline: Optional[datetime] = None
    final_repair_deadline: Optional[datetime] = None
    delay_of_repair_approved: bool = False
    delay_of_repair_reason: str = ""

    # Repair completion
    repair_completed_timestamp: Optional[datetime] = None
    repair_verification_reading_ppm: Optional[float] = None
    repair_technician: str = ""
    repair_notes: str = ""

    # Audit fields
    created_timestamp: datetime = field(default_factory=datetime.utcnow)
    updated_timestamp: datetime = field(default_factory=datetime.utcnow)
    created_by: str = ""
    updated_by: str = ""

    def set_repair_deadlines(self, first_attempt_days: int = 5, final_repair_days: int = 15):
        """Set repair deadlines based on EPA requirements."""
        now = datetime.utcnow()
        self.first_attempt_deadline = now + timedelta(days=first_attempt_days)
        self.final_repair_deadline = now + timedelta(days=final_repair_days)
        self.repair_deadline = self.final_repair_deadline

    def is_overdue(self) -> bool:
        """Check if repair is overdue."""
        if self.repair_status == RepairStatus.COMPLETED:
            return False
        if self.repair_deadline and datetime.utcnow() > self.repair_deadline:
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "detection_id": self.detection_id,
            "confirmed": self.confirmed,
            "confirmation_method": self.confirmation_method,
            "confirmation_timestamp": self.confirmation_timestamp.isoformat() if self.confirmation_timestamp else None,
            "estimated_emission_rate_kg_hr": str(self.estimated_emission_rate_kg_hr),
            "emission_rate_uncertainty_kg_hr": str(self.emission_rate_uncertainty_kg_hr) if self.emission_rate_uncertainty_kg_hr else None,
            "source_equipment": self.source_equipment,
            "equipment_tag": self.equipment_tag,
            "source_type": self.source_type.value,
            "location_lat": self.location_lat,
            "location_lon": self.location_lon,
            "location_description": self.location_description,
            "repair_status": self.repair_status.value,
            "repair_deadline": self.repair_deadline.isoformat() if self.repair_deadline else None,
            "first_attempt_deadline": self.first_attempt_deadline.isoformat() if self.first_attempt_deadline else None,
            "final_repair_deadline": self.final_repair_deadline.isoformat() if self.final_repair_deadline else None,
            "delay_of_repair_approved": self.delay_of_repair_approved,
            "delay_of_repair_reason": self.delay_of_repair_reason,
            "repair_completed_timestamp": self.repair_completed_timestamp.isoformat() if self.repair_completed_timestamp else None,
            "repair_verification_reading_ppm": self.repair_verification_reading_ppm,
            "repair_technician": self.repair_technician,
            "repair_notes": self.repair_notes,
            "created_timestamp": self.created_timestamp.isoformat(),
            "updated_timestamp": self.updated_timestamp.isoformat(),
            "created_by": self.created_by,
            "updated_by": self.updated_by,
        }


@dataclass
class ReviewDecision:
    """
    Human-in-the-loop review decision for ML detections.

    Captures the reviewer's assessment of an ML detection result
    for feedback and model improvement.
    """
    decision_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    detection_id: str = ""

    # Reviewer information
    reviewer_id: str = ""
    reviewer_name: str = ""
    reviewer_role: str = ""

    # Decision
    status: ReviewStatus = ReviewStatus.PENDING
    decision_timestamp: Optional[datetime] = None

    # Assessment
    ml_prediction_correct: Optional[bool] = None
    corrected_source_type: Optional[SourceType] = None
    corrected_is_anomaly: Optional[bool] = None
    confidence_in_decision: float = 0.0  # Reviewer confidence (0-1)

    # Feedback
    notes: str = ""
    additional_context: str = ""
    recommended_action: str = ""

    # Escalation
    escalated_to: str = ""
    escalation_reason: str = ""

    # Audit
    created_timestamp: datetime = field(default_factory=datetime.utcnow)
    updated_timestamp: datetime = field(default_factory=datetime.utcnow)

    # Training feedback
    use_for_retraining: bool = True
    feedback_quality_score: Optional[float] = None  # Quality of feedback for training

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "decision_id": self.decision_id,
            "detection_id": self.detection_id,
            "reviewer_id": self.reviewer_id,
            "reviewer_name": self.reviewer_name,
            "reviewer_role": self.reviewer_role,
            "status": self.status.value,
            "decision_timestamp": self.decision_timestamp.isoformat() if self.decision_timestamp else None,
            "ml_prediction_correct": self.ml_prediction_correct,
            "corrected_source_type": self.corrected_source_type.value if self.corrected_source_type else None,
            "corrected_is_anomaly": self.corrected_is_anomaly,
            "confidence_in_decision": self.confidence_in_decision,
            "notes": self.notes,
            "additional_context": self.additional_context,
            "recommended_action": self.recommended_action,
            "escalated_to": self.escalated_to,
            "escalation_reason": self.escalation_reason,
            "created_timestamp": self.created_timestamp.isoformat(),
            "updated_timestamp": self.updated_timestamp.isoformat(),
            "use_for_retraining": self.use_for_retraining,
            "feedback_quality_score": self.feedback_quality_score,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReviewDecision":
        """Create from dictionary."""
        data = data.copy()
        if isinstance(data.get("status"), str):
            data["status"] = ReviewStatus(data["status"])
        if isinstance(data.get("corrected_source_type"), str):
            data["corrected_source_type"] = SourceType(data["corrected_source_type"])
        if isinstance(data.get("decision_timestamp"), str):
            data["decision_timestamp"] = datetime.fromisoformat(data["decision_timestamp"])
        if isinstance(data.get("created_timestamp"), str):
            data["created_timestamp"] = datetime.fromisoformat(data["created_timestamp"])
        if isinstance(data.get("updated_timestamp"), str):
            data["updated_timestamp"] = datetime.fromisoformat(data["updated_timestamp"])
        return cls(**data)


@dataclass
class ReviewTask:
    """
    Task queued for human review.

    Contains all information needed by a reviewer to assess
    an ML detection result.
    """
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    detection_id: str = ""

    # Detection details
    detection_result: Optional[DetectionResult] = None

    # Task metadata
    created_timestamp: datetime = field(default_factory=datetime.utcnow)
    deadline: Optional[datetime] = None
    priority: int = 0  # Higher = more urgent

    # Assignment
    assigned_to: str = ""
    assigned_timestamp: Optional[datetime] = None

    # Suggested action
    suggested_action: str = ""
    auto_generated_notes: str = ""

    # Status
    status: ReviewStatus = ReviewStatus.PENDING

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "detection_id": self.detection_id,
            "detection_result": self.detection_result.to_dict() if self.detection_result else None,
            "created_timestamp": self.created_timestamp.isoformat(),
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "priority": self.priority,
            "assigned_to": self.assigned_to,
            "assigned_timestamp": self.assigned_timestamp.isoformat() if self.assigned_timestamp else None,
            "suggested_action": self.suggested_action,
            "auto_generated_notes": self.auto_generated_notes,
            "status": self.status.value,
        }


@dataclass
class ReviewResult:
    """
    Result of a completed review.

    Links the review task to the decision made by the reviewer.
    """
    result_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str = ""
    detection_id: str = ""

    # Decision
    decision: Optional[ReviewDecision] = None

    # Timing
    review_duration_seconds: float = 0.0
    completed_timestamp: datetime = field(default_factory=datetime.utcnow)

    # Outcome
    outcome: str = ""  # e.g., "confirmed_leak", "false_positive", "needs_followup"
    followup_actions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "result_id": self.result_id,
            "task_id": self.task_id,
            "detection_id": self.detection_id,
            "decision": self.decision.to_dict() if self.decision else None,
            "review_duration_seconds": self.review_duration_seconds,
            "completed_timestamp": self.completed_timestamp.isoformat(),
            "outcome": self.outcome,
            "followup_actions": self.followup_actions,
        }


@dataclass
class ClassificationResult:
    """
    Result from leak source classification.

    Contains predicted source type with probability distribution
    across all possible classifications.
    """
    classification_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Primary prediction
    predicted_source_type: SourceType = SourceType.UNKNOWN
    confidence_score: float = 0.0

    # Top-k predictions
    top_predictions: List[Tuple[SourceType, float]] = field(default_factory=list)

    # Full probability distribution
    probabilities: Dict[str, float] = field(default_factory=dict)

    # Model metadata
    model_name: str = ""
    model_version: str = ""
    feature_importances: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "classification_id": self.classification_id,
            "timestamp": self.timestamp.isoformat(),
            "predicted_source_type": self.predicted_source_type.value,
            "confidence_score": self.confidence_score,
            "top_predictions": [(s.value, p) for s, p in self.top_predictions],
            "probabilities": self.probabilities,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "feature_importances": self.feature_importances,
        }


@dataclass
class FeatureVector:
    """
    Feature vector extracted from sensor readings.

    Contains all engineered features for ML model input.
    """
    feature_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Feature values
    features: Dict[str, float] = field(default_factory=dict)

    # Metadata
    source_reading_ids: List[str] = field(default_factory=list)
    feature_names: List[str] = field(default_factory=list)

    # Feature groups
    temporal_features: Dict[str, float] = field(default_factory=dict)
    spatial_features: Dict[str, float] = field(default_factory=dict)
    meteorological_features: Dict[str, float] = field(default_factory=dict)

    def to_array(self) -> List[float]:
        """Convert features to array for ML input."""
        return [self.features.get(name, 0.0) for name in self.feature_names]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feature_id": self.feature_id,
            "timestamp": self.timestamp.isoformat(),
            "features": self.features,
            "source_reading_ids": self.source_reading_ids,
            "feature_names": self.feature_names,
            "temporal_features": self.temporal_features,
            "spatial_features": self.spatial_features,
            "meteorological_features": self.meteorological_features,
        }


@dataclass
class WindData:
    """
    Wind data for plume dispersion modeling.
    """
    timestamp: datetime = field(default_factory=datetime.utcnow)
    wind_speed: float = 0.0  # m/s
    wind_direction: float = 0.0  # degrees from north
    wind_speed_std: float = 0.0  # standard deviation
    wind_direction_std: float = 0.0  # standard deviation
    measurement_height_m: float = 10.0
    stability_class: PasquillStabilityClass = PasquillStabilityClass.D

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "wind_speed": self.wind_speed,
            "wind_direction": self.wind_direction,
            "wind_speed_std": self.wind_speed_std,
            "wind_direction_std": self.wind_direction_std,
            "measurement_height_m": self.measurement_height_m,
            "stability_class": self.stability_class.value,
        }


@dataclass
class Explanation:
    """
    ML model explanation for a detection result.

    Contains interpretability information including feature
    importances, SHAP values, and natural language summary.
    """
    explanation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    detection_id: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Feature importances
    feature_importances: Dict[str, float] = field(default_factory=dict)

    # SHAP values
    shap_values: Dict[str, float] = field(default_factory=dict)
    shap_base_value: float = 0.0

    # LIME explanation
    lime_weights: Dict[str, float] = field(default_factory=dict)
    lime_local_prediction: float = 0.0

    # Natural language summary (template-based, LLM-safe)
    natural_language_summary: str = ""

    # Visualization data
    visualization_data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "explanation_id": self.explanation_id,
            "detection_id": self.detection_id,
            "timestamp": self.timestamp.isoformat(),
            "feature_importances": self.feature_importances,
            "shap_values": self.shap_values,
            "shap_base_value": self.shap_base_value,
            "lime_weights": self.lime_weights,
            "lime_local_prediction": self.lime_local_prediction,
            "natural_language_summary": self.natural_language_summary,
            "visualization_data": self.visualization_data,
        }


@dataclass
class SHAPExplanation:
    """SHAP-specific explanation values."""
    shap_values: Dict[str, float] = field(default_factory=dict)
    base_value: float = 0.0
    expected_value: float = 0.0
    feature_names: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "shap_values": self.shap_values,
            "base_value": self.base_value,
            "expected_value": self.expected_value,
            "feature_names": self.feature_names,
        }


@dataclass
class LIMEExplanation:
    """LIME-specific explanation values."""
    feature_weights: Dict[str, float] = field(default_factory=dict)
    local_prediction: float = 0.0
    intercept: float = 0.0
    score: float = 0.0  # R-squared of local model

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_weights": self.feature_weights,
            "local_prediction": self.local_prediction,
            "intercept": self.intercept,
            "score": self.score,
        }
