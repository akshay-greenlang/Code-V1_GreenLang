"""
GL-016 Waterguard Kafka Message Schemas

Defines Avro-compatible message schemas for all Kafka topics used in boiler
water chemistry monitoring. Each schema includes validation, serialization,
and deserialization support.
"""

from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


# =============================================================================
# Enums
# =============================================================================

class QualityCode(str, Enum):
    """OPC-UA quality codes for sensor data."""
    GOOD = "good"
    UNCERTAIN = "uncertain"
    BAD = "bad"
    STALE = "stale"
    SUBSTITUTED = "substituted"


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class CommandType(str, Enum):
    """Types of actuation commands."""
    SETPOINT = "setpoint"
    PULSE = "pulse"
    ENABLE = "enable"
    DISABLE = "disable"
    CALIBRATE = "calibrate"


class AckStatus(str, Enum):
    """Command acknowledgement status."""
    RECEIVED = "received"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class RecommendationType(str, Enum):
    """Types of chemistry recommendations."""
    INCREASE_DOSING = "increase_dosing"
    DECREASE_DOSING = "decrease_dosing"
    MAINTAIN = "maintain"
    BLOWDOWN = "blowdown"
    ALERT_OPERATOR = "alert_operator"
    CALIBRATION_DUE = "calibration_due"


class AuditAction(str, Enum):
    """Audit log action types."""
    COMMAND_ISSUED = "command_issued"
    COMMAND_EXECUTED = "command_executed"
    SETPOINT_CHANGED = "setpoint_changed"
    ALERT_RAISED = "alert_raised"
    ALERT_ACKNOWLEDGED = "alert_acknowledged"
    CONFIG_CHANGED = "config_changed"
    OVERRIDE_APPLIED = "override_applied"
    SAFETY_LOCKOUT = "safety_lockout"


# =============================================================================
# Base Message Schema
# =============================================================================

class BaseKafkaMessage(BaseModel):
    """Base class for all Kafka messages with common metadata."""

    message_id: UUID = Field(default_factory=uuid4, description="Unique message identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Message timestamp (UTC)")
    source: str = Field(..., description="Source system identifier")
    version: str = Field(default="1.0", description="Schema version")
    trace_id: Optional[UUID] = Field(default=None, description="Distributed tracing ID")
    correlation_id: Optional[UUID] = Field(default=None, description="Correlation ID for request/response")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: str
        }

    def to_avro_dict(self) -> Dict[str, Any]:
        """Convert to Avro-compatible dictionary."""
        data = self.model_dump()
        data["timestamp"] = self.timestamp.isoformat()
        data["message_id"] = str(self.message_id)
        if self.trace_id:
            data["trace_id"] = str(self.trace_id)
        if self.correlation_id:
            data["correlation_id"] = str(self.correlation_id)
        return data

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, json_str: str) -> "BaseKafkaMessage":
        """Deserialize from JSON string."""
        return cls.model_validate_json(json_str)


# =============================================================================
# Raw Chemistry Message (boiler.gl016.raw)
# =============================================================================

class SensorReading(BaseModel):
    """Individual sensor reading with quality metadata."""

    tag_id: str = Field(..., description="OPC-UA tag identifier")
    value: float = Field(..., description="Raw sensor value")
    quality: QualityCode = Field(default=QualityCode.GOOD, description="Quality code")
    engineering_units: str = Field(..., description="Engineering units (e.g., 'ppm', 'uS/cm')")
    timestamp: datetime = Field(..., description="Sensor reading timestamp")
    source_quality_bits: Optional[int] = Field(default=None, description="OPC-UA quality bits")

    @field_validator("value")
    @classmethod
    def validate_finite(cls, v: float) -> float:
        """Ensure value is finite (not NaN or Inf)."""
        import math
        if not math.isfinite(v):
            raise ValueError(f"Value must be finite, got {v}")
        return v


class RawChemistryMessage(BaseKafkaMessage):
    """
    Raw chemistry sensor data from OPC-UA.

    Topic: boiler.gl016.raw

    Contains unprocessed sensor readings with full quality metadata.
    Published at sensor scan rate (typically 1-5 seconds).
    """

    boiler_id: str = Field(..., description="Boiler identifier")
    readings: List[SensorReading] = Field(..., description="List of sensor readings")
    scan_time_ms: int = Field(..., description="Scan cycle time in milliseconds")
    plc_timestamp: Optional[datetime] = Field(default=None, description="PLC/DCS timestamp if available")

    @field_validator("readings")
    @classmethod
    def validate_non_empty(cls, v: List[SensorReading]) -> List[SensorReading]:
        """Ensure at least one reading is present."""
        if not v:
            raise ValueError("At least one sensor reading is required")
        return v

    @property
    def has_bad_quality(self) -> bool:
        """Check if any readings have bad quality."""
        return any(r.quality == QualityCode.BAD for r in self.readings)

    def get_reading(self, tag_id: str) -> Optional[SensorReading]:
        """Get a specific reading by tag ID."""
        for reading in self.readings:
            if reading.tag_id == tag_id:
                return reading
        return None


# =============================================================================
# Cleaned Chemistry Message (boiler.gl016.cleaned)
# =============================================================================

class CleanedReading(BaseModel):
    """Cleaned and normalized sensor reading."""

    tag_id: str = Field(..., description="OPC-UA tag identifier")
    raw_value: float = Field(..., description="Original raw value")
    cleaned_value: float = Field(..., description="Cleaned/normalized value")
    quality: QualityCode = Field(..., description="Quality code")
    quality_score: float = Field(..., ge=0.0, le=1.0, description="Quality confidence score (0-1)")
    engineering_units: str = Field(..., description="Engineering units")
    was_interpolated: bool = Field(default=False, description="Value was interpolated")
    was_clamped: bool = Field(default=False, description="Value was clamped to limits")
    interpolation_method: Optional[str] = Field(default=None, description="Interpolation method used")


class CleanedChemistryMessage(BaseKafkaMessage):
    """
    Cleaned and normalized chemistry data.

    Topic: boiler.gl016.cleaned

    Contains processed sensor data with:
    - Unit normalization
    - Outlier removal/clamping
    - Missing value interpolation
    - Quality scoring
    """

    boiler_id: str = Field(..., description="Boiler identifier")
    readings: List[CleanedReading] = Field(..., description="Cleaned readings")
    raw_message_id: UUID = Field(..., description="Reference to source raw message")
    processing_time_ms: int = Field(..., description="Data cleaning processing time")
    overall_quality_score: float = Field(..., ge=0.0, le=1.0, description="Overall data quality score")

    @property
    def good_readings_count(self) -> int:
        """Count of readings with good quality."""
        return sum(1 for r in self.readings if r.quality == QualityCode.GOOD)


# =============================================================================
# Feature Message (boiler.gl016.features)
# =============================================================================

class DerivedFeature(BaseModel):
    """Derived feature from chemistry data."""

    feature_name: str = Field(..., description="Feature identifier")
    value: float = Field(..., description="Computed feature value")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Feature confidence")
    input_tags: List[str] = Field(..., description="Source tags used in calculation")
    calculation_method: str = Field(..., description="Method used for calculation")


class FeatureMessage(BaseKafkaMessage):
    """
    Engineered features for ML models.

    Topic: boiler.gl016.features

    Contains derived features computed from cleaned chemistry data:
    - Rolling statistics (mean, std, rate of change)
    - Cross-parameter features (ratios, correlations)
    - Time-domain features (trends, cycles)
    """

    boiler_id: str = Field(..., description="Boiler identifier")
    features: List[DerivedFeature] = Field(..., description="Computed features")
    cleaned_message_id: UUID = Field(..., description="Reference to cleaned data message")
    window_start: datetime = Field(..., description="Feature window start time")
    window_end: datetime = Field(..., description="Feature window end time")
    window_size_seconds: int = Field(..., description="Feature window size in seconds")


# =============================================================================
# Chemistry State Message (boiler.gl016.chemistry_state)
# =============================================================================

class ChemistryParameter(BaseModel):
    """Current state of a chemistry parameter."""

    parameter_name: str = Field(..., description="Parameter name (e.g., 'phosphate', 'conductivity')")
    current_value: float = Field(..., description="Current value")
    target_value: float = Field(..., description="Target setpoint")
    lower_limit: float = Field(..., description="Lower control limit")
    upper_limit: float = Field(..., description="Upper control limit")
    deviation: float = Field(..., description="Deviation from target (%)")
    trend: str = Field(..., description="Trend direction: rising, falling, stable")
    engineering_units: str = Field(..., description="Engineering units")


class ChemistryStateMessage(BaseKafkaMessage):
    """
    Current chemistry control state.

    Topic: boiler.gl016.chemistry_state

    Provides aggregated view of current boiler chemistry:
    - Current values vs targets
    - Control status
    - Trend indicators
    """

    boiler_id: str = Field(..., description="Boiler identifier")
    parameters: List[ChemistryParameter] = Field(..., description="Chemistry parameters")
    overall_health: str = Field(..., description="Overall chemistry health: good, marginal, poor")
    health_score: float = Field(..., ge=0.0, le=100.0, description="Overall health score (0-100)")
    active_deviations: int = Field(..., description="Number of parameters outside limits")


# =============================================================================
# Recommendation Message (boiler.gl016.recommendations)
# =============================================================================

class Recommendation(BaseModel):
    """Single recommendation for chemistry control."""

    recommendation_id: UUID = Field(default_factory=uuid4, description="Recommendation ID")
    recommendation_type: RecommendationType = Field(..., description="Type of recommendation")
    parameter: str = Field(..., description="Target parameter")
    current_value: float = Field(..., description="Current parameter value")
    target_value: float = Field(..., description="Recommended target")
    action: str = Field(..., description="Recommended action text")
    priority: int = Field(..., ge=1, le=5, description="Priority (1=highest, 5=lowest)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Recommendation confidence")
    reasoning: str = Field(..., description="Explanation for recommendation")
    auto_executable: bool = Field(default=False, description="Can be auto-executed without operator")


class RecommendationMessage(BaseKafkaMessage):
    """
    Chemistry control recommendations.

    Topic: boiler.gl016.recommendations

    Contains AI-generated recommendations for chemistry control:
    - Dosing adjustments
    - Blowdown recommendations
    - Calibration reminders
    - Alert conditions
    """

    boiler_id: str = Field(..., description="Boiler identifier")
    recommendations: List[Recommendation] = Field(..., description="List of recommendations")
    model_version: str = Field(..., description="ML model version")
    state_message_id: UUID = Field(..., description="Reference to chemistry state message")
    requires_operator_approval: bool = Field(default=True, description="Requires operator approval")


# =============================================================================
# Command Message (boiler.gl016.actuation_commands)
# =============================================================================

class CommandMessage(BaseKafkaMessage):
    """
    Actuation command for equipment control.

    Topic: boiler.gl016.actuation_commands

    Contains control commands for:
    - Dosing pump setpoints
    - Blowdown valve positions
    - Analyzer calibration triggers
    """

    command_id: UUID = Field(default_factory=uuid4, description="Unique command ID")
    boiler_id: str = Field(..., description="Target boiler")
    command_type: CommandType = Field(..., description="Type of command")
    target_tag: str = Field(..., description="Target OPC-UA tag")
    target_value: float = Field(..., description="Target value/setpoint")
    previous_value: Optional[float] = Field(default=None, description="Previous value before command")
    ramp_rate: Optional[float] = Field(default=None, description="Ramp rate for setpoint changes")
    timeout_seconds: int = Field(default=30, description="Command timeout")
    requires_ack: bool = Field(default=True, description="Requires acknowledgement")
    operator_id: Optional[str] = Field(default=None, description="Operator who approved")
    recommendation_id: Optional[UUID] = Field(default=None, description="Source recommendation")
    safety_validated: bool = Field(default=False, description="Passed safety validation")

    @model_validator(mode="after")
    def validate_safety(self) -> "CommandMessage":
        """Ensure safety validation for auto commands."""
        if self.operator_id is None and not self.safety_validated:
            raise ValueError("Auto commands require safety_validated=True")
        return self


# =============================================================================
# Acknowledgement Message (boiler.gl016.actuation_ack)
# =============================================================================

class AckMessage(BaseKafkaMessage):
    """
    Command acknowledgement message.

    Topic: boiler.gl016.actuation_ack

    Confirms command receipt and execution status.
    """

    command_id: UUID = Field(..., description="Original command ID")
    ack_status: AckStatus = Field(..., description="Acknowledgement status")
    actual_value: Optional[float] = Field(default=None, description="Actual achieved value")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    execution_time_ms: Optional[int] = Field(default=None, description="Execution time in ms")
    plc_response_code: Optional[int] = Field(default=None, description="PLC response code")

    @model_validator(mode="after")
    def validate_error_on_failure(self) -> "AckMessage":
        """Ensure error message is present on failure."""
        if self.ack_status == AckStatus.FAILED and not self.error_message:
            raise ValueError("error_message required when ack_status is FAILED")
        return self


# =============================================================================
# Alert Message (boiler.gl016.alerts)
# =============================================================================

class AlertMessage(BaseKafkaMessage):
    """
    Chemistry alert notification.

    Topic: boiler.gl016.alerts

    Contains alerts for:
    - Parameter out of limits
    - Equipment failures
    - Communication losses
    - Safety conditions
    """

    alert_id: UUID = Field(default_factory=uuid4, description="Unique alert ID")
    boiler_id: str = Field(..., description="Boiler identifier")
    severity: AlertSeverity = Field(..., description="Alert severity")
    category: str = Field(..., description="Alert category")
    title: str = Field(..., description="Short alert title")
    message: str = Field(..., description="Detailed alert message")
    parameter: Optional[str] = Field(default=None, description="Related parameter")
    current_value: Optional[float] = Field(default=None, description="Current value")
    threshold_value: Optional[float] = Field(default=None, description="Threshold that was crossed")
    recommended_action: Optional[str] = Field(default=None, description="Recommended response")
    auto_acknowledged: bool = Field(default=False, description="Auto-acknowledged")
    escalation_level: int = Field(default=1, description="Escalation level (1-3)")


# =============================================================================
# Audit Message (boiler.gl016.audit)
# =============================================================================

class AuditMessage(BaseKafkaMessage):
    """
    Audit log entry for compliance and traceability.

    Topic: boiler.gl016.audit

    Immutable record of all control actions for:
    - Regulatory compliance
    - Incident investigation
    - Performance analysis
    """

    audit_id: UUID = Field(default_factory=uuid4, description="Unique audit ID")
    boiler_id: str = Field(..., description="Boiler identifier")
    action: AuditAction = Field(..., description="Action type")
    actor: str = Field(..., description="User or system that performed action")
    actor_type: str = Field(..., description="Actor type: operator, system, ai")
    target: str = Field(..., description="Target of action (tag, parameter, etc.)")
    previous_state: Optional[Dict[str, Any]] = Field(default=None, description="State before action")
    new_state: Optional[Dict[str, Any]] = Field(default=None, description="State after action")
    justification: Optional[str] = Field(default=None, description="Reason for action")
    command_id: Optional[UUID] = Field(default=None, description="Related command ID")
    recommendation_id: Optional[UUID] = Field(default=None, description="Related recommendation ID")
    ip_address: Optional[str] = Field(default=None, description="Source IP address")
    session_id: Optional[str] = Field(default=None, description="User session ID")


# =============================================================================
# Schema Registry
# =============================================================================

SCHEMA_REGISTRY: Dict[str, type] = {
    "boiler.gl016.raw": RawChemistryMessage,
    "boiler.gl016.cleaned": CleanedChemistryMessage,
    "boiler.gl016.features": FeatureMessage,
    "boiler.gl016.chemistry_state": ChemistryStateMessage,
    "boiler.gl016.recommendations": RecommendationMessage,
    "boiler.gl016.actuation_commands": CommandMessage,
    "boiler.gl016.actuation_ack": AckMessage,
    "boiler.gl016.alerts": AlertMessage,
    "boiler.gl016.audit": AuditMessage,
}


def get_schema_for_topic(topic: str) -> Optional[type]:
    """Get the schema class for a given topic."""
    return SCHEMA_REGISTRY.get(topic)


def validate_message(topic: str, message: Dict[str, Any]) -> BaseKafkaMessage:
    """
    Validate a message against its topic schema.

    Args:
        topic: Kafka topic name
        message: Message data dictionary

    Returns:
        Validated message instance

    Raises:
        ValueError: If topic is unknown or message is invalid
    """
    schema_class = get_schema_for_topic(topic)
    if schema_class is None:
        raise ValueError(f"Unknown topic: {topic}")
    return schema_class.model_validate(message)


# =============================================================================
# Avro Schema Definitions (for schema registry)
# =============================================================================

def get_avro_schema(topic: str) -> Optional[Dict[str, Any]]:
    """
    Get Avro schema definition for a topic.

    This is used for registering schemas with Confluent Schema Registry.
    """
    schemas = {
        "boiler.gl016.raw": {
            "type": "record",
            "name": "RawChemistryMessage",
            "namespace": "com.greenlang.waterguard",
            "fields": [
                {"name": "message_id", "type": "string"},
                {"name": "timestamp", "type": "string"},
                {"name": "source", "type": "string"},
                {"name": "version", "type": "string"},
                {"name": "trace_id", "type": ["null", "string"], "default": None},
                {"name": "boiler_id", "type": "string"},
                {"name": "scan_time_ms", "type": "int"},
                {
                    "name": "readings",
                    "type": {
                        "type": "array",
                        "items": {
                            "type": "record",
                            "name": "SensorReading",
                            "fields": [
                                {"name": "tag_id", "type": "string"},
                                {"name": "value", "type": "double"},
                                {"name": "quality", "type": "string"},
                                {"name": "engineering_units", "type": "string"},
                                {"name": "timestamp", "type": "string"},
                            ]
                        }
                    }
                }
            ]
        },
        # Additional Avro schemas would be defined here for other topics
    }
    return schemas.get(topic)
