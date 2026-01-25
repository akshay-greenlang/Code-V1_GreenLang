"""
GL-012 SteamQual API Schemas

Pydantic models for request/response validation and serialization.
Defines core data structures for steam quality controller API.

Latency Targets:
- Sensor-to-metric: < 5 seconds
- Event emission: < 10 seconds
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


# =============================================================================
# Enumerations
# =============================================================================

class QualityLevel(str, Enum):
    """Steam quality classification levels."""
    EXCELLENT = "excellent"  # > 99.5% dryness
    GOOD = "good"  # 98-99.5% dryness
    ACCEPTABLE = "acceptable"  # 95-98% dryness
    MARGINAL = "marginal"  # 90-95% dryness
    POOR = "poor"  # < 90% dryness
    CRITICAL = "critical"  # Unacceptable for process


class CarryoverRiskLevel(str, Enum):
    """Risk levels for moisture carryover."""
    LOW = "low"  # < 10% probability
    MODERATE = "moderate"  # 10-30% probability
    HIGH = "high"  # 30-60% probability
    CRITICAL = "critical"  # > 60% probability


class EventSeverity(str, Enum):
    """Event severity levels."""
    INFO = "info"
    WARNING = "warning"
    ALERT = "alert"
    CRITICAL = "critical"


class EventType(str, Enum):
    """Types of quality events."""
    QUALITY_DEGRADATION = "quality_degradation"
    CARRYOVER_DETECTED = "carryover_detected"
    SEPARATOR_EFFICIENCY_DROP = "separator_efficiency_drop"
    DRUM_LEVEL_ANOMALY = "drum_level_anomaly"
    BLOWDOWN_NEEDED = "blowdown_needed"
    LOAD_CHANGE_IMPACT = "load_change_impact"
    QUALITY_RESTORED = "quality_restored"
    THRESHOLD_EXCEEDED = "threshold_exceeded"


class RecommendationType(str, Enum):
    """Types of control recommendations."""
    BLOWDOWN_ADJUSTMENT = "blowdown_adjustment"
    FEEDWATER_TREATMENT = "feedwater_treatment"
    SEPARATOR_MAINTENANCE = "separator_maintenance"
    DRUM_LEVEL_SETPOINT = "drum_level_setpoint"
    LOAD_REDUCTION = "load_reduction"
    STEAM_SUPERHEAT = "steam_superheat"
    INSTRUMENTATION_CHECK = "instrumentation_check"


class RecommendationPriority(str, Enum):
    """Priority levels for recommendations."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"


class MeasurementSource(str, Enum):
    """Source of quality measurements."""
    CALORIMETRIC = "calorimetric"  # Throttling calorimeter
    SEPARATOR = "separator"  # Separator method
    ELECTRICAL = "electrical"  # Conductivity/capacitance
    OPTICAL = "optical"  # Optical sensors
    TRACER = "tracer"  # Tracer method
    CALCULATED = "calculated"  # Derived from other measurements


# =============================================================================
# Base Models
# =============================================================================

class TimestampedModel(BaseModel):
    """Base model with timestamp fields."""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
            Decimal: lambda v: float(v),
        }


# =============================================================================
# Quality Measurement Models
# =============================================================================

class SteamMeasurements(BaseModel):
    """Raw steam measurements for quality estimation."""
    header_id: str = Field(..., min_length=1, max_length=100, description="Steam header identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Primary measurements
    pressure_kpa: float = Field(..., ge=0, description="Steam pressure in kPa")
    temperature_c: float = Field(..., description="Steam temperature in Celsius")

    # Optional measurements for quality calculation
    saturation_temperature_c: Optional[float] = Field(None, description="Saturation temperature at pressure")
    superheat_c: Optional[float] = Field(None, ge=0, description="Degrees of superheat")

    # Separator/drum measurements
    drum_level_percent: Optional[float] = Field(None, ge=0, le=100, description="Drum level percentage")
    separator_dp_kpa: Optional[float] = Field(None, ge=0, description="Separator differential pressure")

    # Conductivity measurements (for TDS carryover)
    steam_conductivity_us_cm: Optional[float] = Field(None, ge=0, description="Steam conductivity")
    feedwater_conductivity_us_cm: Optional[float] = Field(None, ge=0, description="Feedwater conductivity")
    blowdown_conductivity_us_cm: Optional[float] = Field(None, ge=0, description="Blowdown conductivity")

    # Flow measurements
    steam_flow_kg_s: Optional[float] = Field(None, ge=0, description="Steam flow rate")
    blowdown_flow_kg_s: Optional[float] = Field(None, ge=0, description="Blowdown flow rate")

    # Load context
    boiler_load_percent: Optional[float] = Field(None, ge=0, le=100, description="Current boiler load")
    load_change_rate_percent_min: Optional[float] = Field(None, description="Rate of load change")


class QualityEstimateRequest(BaseModel):
    """Request to estimate steam quality from measurements."""
    request_id: UUID = Field(default_factory=uuid4)
    measurements: SteamMeasurements

    # Estimation options
    method: Optional[MeasurementSource] = Field(
        default=MeasurementSource.CALCULATED,
        description="Primary estimation method"
    )
    include_uncertainty: bool = Field(default=True, description="Include uncertainty bounds")
    include_trend: bool = Field(default=True, description="Include historical trend")

    # Reference data
    target_quality_percent: float = Field(default=99.0, ge=90, le=100, description="Target quality")
    alarm_threshold_percent: float = Field(default=95.0, ge=85, le=100, description="Alarm threshold")

    @model_validator(mode="after")
    def validate_thresholds(self):
        """Ensure alarm threshold is below target."""
        if self.alarm_threshold_percent >= self.target_quality_percent:
            raise ValueError("alarm_threshold must be below target_quality")
        return self


class QualityEstimate(BaseModel):
    """Estimated steam quality with confidence bounds."""
    quality_percent: float = Field(..., ge=0, le=100, description="Estimated steam quality (dryness)")
    quality_level: QualityLevel = Field(..., description="Quality classification")

    # Uncertainty bounds (95% confidence)
    quality_lower_bound: Optional[float] = Field(None, ge=0, le=100, description="Lower bound (95% CI)")
    quality_upper_bound: Optional[float] = Field(None, ge=0, le=100, description="Upper bound (95% CI)")
    confidence_score: float = Field(..., ge=0, le=1, description="Estimation confidence")

    # Derived metrics
    moisture_content_percent: float = Field(..., ge=0, le=100, description="Moisture content (100 - quality)")
    specific_enthalpy_kj_kg: Optional[float] = Field(None, description="Specific enthalpy at quality")

    # Trend data
    trend_direction: Optional[str] = Field(None, pattern="^(improving|stable|degrading)$")
    trend_rate_percent_hour: Optional[float] = Field(None, description="Rate of quality change")

    # Estimation metadata
    estimation_method: MeasurementSource
    measurement_count: int = Field(default=1, ge=1)


class QualityEstimateResponse(TimestampedModel):
    """Response with quality estimation results."""
    request_id: UUID
    header_id: str
    success: bool

    # Quality estimate
    estimate: Optional[QualityEstimate] = None

    # Comparison to targets
    target_quality_percent: float
    deviation_from_target: Optional[float] = None
    is_below_alarm_threshold: bool = False

    # Processing metrics (target: < 5s)
    processing_time_ms: float = Field(..., ge=0)
    sensor_timestamp: datetime

    # Provenance
    provenance_hash: str = Field(..., description="SHA-256 hash for audit trail")

    error_message: Optional[str] = None


# =============================================================================
# Carryover Risk Assessment Models
# =============================================================================

class CarryoverRiskRequest(BaseModel):
    """Request to assess moisture carryover risk."""
    request_id: UUID = Field(default_factory=uuid4)
    header_id: str = Field(..., min_length=1, max_length=100)

    # Current operating conditions
    steam_flow_kg_s: float = Field(..., ge=0, description="Current steam flow rate")
    drum_level_percent: float = Field(..., ge=0, le=100, description="Current drum level")
    boiler_load_percent: float = Field(..., ge=0, le=100, description="Current load percentage")

    # Steam conditions
    steam_pressure_kpa: float = Field(..., ge=0)
    steam_temperature_c: float

    # Separator data (if available)
    separator_efficiency_percent: Optional[float] = Field(None, ge=0, le=100)

    # TDS data
    boiler_tds_ppm: Optional[float] = Field(None, ge=0, description="Boiler water TDS")
    steam_tds_ppm: Optional[float] = Field(None, ge=0, description="Steam TDS (if measured)")

    # Operating context
    load_change_occurring: bool = Field(default=False)
    time_since_blowdown_min: Optional[float] = Field(None, ge=0)

    # Thresholds
    risk_threshold: CarryoverRiskLevel = Field(
        default=CarryoverRiskLevel.MODERATE,
        description="Threshold for triggering alerts"
    )


class CarryoverRiskFactors(BaseModel):
    """Individual risk factors contributing to carryover risk."""
    factor_name: str
    factor_value: float
    contribution_score: float = Field(..., ge=0, le=1)
    is_primary_driver: bool = False
    mitigation_action: Optional[str] = None


class CarryoverRiskAssessment(BaseModel):
    """Assessment of carryover risk."""
    risk_level: CarryoverRiskLevel
    risk_probability: float = Field(..., ge=0, le=1, description="Probability of carryover event")
    risk_score: float = Field(..., ge=0, le=100, description="Composite risk score")

    # Contributing factors
    risk_factors: List[CarryoverRiskFactors] = Field(default_factory=list)
    primary_risk_driver: Optional[str] = None

    # Predicted impact
    predicted_quality_impact_percent: Optional[float] = Field(None, description="Expected quality reduction")
    predicted_steam_loss_kg_h: Optional[float] = Field(None, ge=0)
    predicted_energy_loss_kw: Optional[float] = Field(None, ge=0)

    # Time to threshold
    time_to_threshold_min: Optional[float] = Field(None, ge=0, description="Estimated time to exceed threshold")

    # Model confidence
    model_confidence: float = Field(..., ge=0, le=1)


class CarryoverRiskResponse(TimestampedModel):
    """Response with carryover risk assessment."""
    request_id: UUID
    header_id: str
    success: bool

    assessment: Optional[CarryoverRiskAssessment] = None

    # Threshold comparison
    exceeds_threshold: bool = False

    # Processing metrics
    processing_time_ms: float = Field(..., ge=0)
    provenance_hash: str

    error_message: Optional[str] = None


# =============================================================================
# Quality State Models
# =============================================================================

class QualityKPI(BaseModel):
    """Key performance indicator for steam quality."""
    kpi_name: str
    current_value: float
    target_value: Optional[float] = None
    unit: str
    trend: Optional[str] = Field(None, pattern="^(up|down|stable)$")
    is_on_target: Optional[bool] = None
    last_updated: datetime = Field(default_factory=datetime.utcnow)


class QualityStateResponse(TimestampedModel):
    """Current quality state for a steam header."""
    request_id: UUID = Field(default_factory=uuid4)
    header_id: str
    success: bool

    # Current quality
    current_quality_percent: Optional[float] = Field(None, ge=0, le=100)
    quality_level: Optional[QualityLevel] = None

    # Operating conditions
    steam_pressure_kpa: Optional[float] = None
    steam_temperature_c: Optional[float] = None
    steam_flow_kg_s: Optional[float] = None
    boiler_load_percent: Optional[float] = None

    # Quality statistics (rolling window)
    quality_mean_24h: Optional[float] = None
    quality_std_24h: Optional[float] = None
    quality_min_24h: Optional[float] = None
    quality_max_24h: Optional[float] = None

    # Carryover metrics
    carryover_risk_level: Optional[CarryoverRiskLevel] = None
    last_carryover_event: Optional[datetime] = None

    # KPIs
    quality_kpis: List[QualityKPI] = Field(default_factory=list)

    # Alarms
    active_alarms: List[str] = Field(default_factory=list)

    # Last update
    last_measurement_time: Optional[datetime] = None
    data_staleness_seconds: Optional[float] = None

    error_message: Optional[str] = None


# =============================================================================
# Events Models
# =============================================================================

class QualityEvent(BaseModel):
    """A quality-related event."""
    event_id: UUID = Field(default_factory=uuid4)
    event_type: EventType
    severity: EventSeverity

    header_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Event details
    title: str
    description: str

    # Measurements at event time
    quality_at_event: Optional[float] = None
    threshold_value: Optional[float] = None
    deviation: Optional[float] = None

    # Duration (for ongoing events)
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_minutes: Optional[float] = None
    is_active: bool = True

    # Impact
    estimated_steam_loss_kg: Optional[float] = None
    estimated_energy_loss_kwh: Optional[float] = None

    # Resolution
    is_acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None


class EventsRequest(BaseModel):
    """Request for quality events."""
    header_id: Optional[str] = Field(None, description="Filter by header (None = all)")
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    event_types: Optional[List[EventType]] = None
    severity_filter: Optional[List[EventSeverity]] = None
    active_only: bool = False
    limit: int = Field(default=100, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)


class EventsResponse(TimestampedModel):
    """Response with quality events."""
    request_id: UUID = Field(default_factory=uuid4)
    success: bool

    events: List[QualityEvent] = Field(default_factory=list)

    # Summary
    total_count: int
    active_count: int

    # Pagination
    limit: int
    offset: int
    has_more: bool

    # Processing (target: < 10s for event emission)
    processing_time_ms: float = Field(..., ge=0)

    error_message: Optional[str] = None


# =============================================================================
# Recommendations Models
# =============================================================================

class ControlAction(BaseModel):
    """Specific control action within a recommendation."""
    action_name: str
    action_type: RecommendationType

    # Setpoint change
    current_value: Optional[float] = None
    recommended_value: Optional[float] = None
    change_magnitude: Optional[float] = None
    unit: Optional[str] = None

    # Implementation
    is_automated: bool = False
    requires_operator_approval: bool = True
    estimated_implementation_time_min: Optional[float] = None

    # Constraints
    min_value: Optional[float] = None
    max_value: Optional[float] = None


class QualityRecommendation(BaseModel):
    """Control recommendation for quality improvement."""
    recommendation_id: UUID = Field(default_factory=uuid4)
    recommendation_type: RecommendationType
    priority: RecommendationPriority

    header_id: str

    # Description
    title: str
    description: str
    rationale: str

    # Actions
    actions: List[ControlAction] = Field(default_factory=list)

    # Expected impact
    expected_quality_improvement_percent: Optional[float] = Field(None, ge=0)
    expected_time_to_effect_min: Optional[float] = Field(None, ge=0)
    confidence_score: float = Field(..., ge=0, le=1)

    # Cost/benefit
    estimated_energy_savings_kw: Optional[float] = None
    estimated_water_savings_kg_h: Optional[float] = None

    # Validity
    valid_until: Optional[datetime] = None
    is_expired: bool = False

    # Status
    is_implemented: bool = False
    implemented_at: Optional[datetime] = None
    implemented_by: Optional[str] = None

    created_at: datetime = Field(default_factory=datetime.utcnow)


class RecommendationsRequest(BaseModel):
    """Request for control recommendations."""
    request_id: UUID = Field(default_factory=uuid4)
    header_id: str = Field(..., min_length=1, max_length=100)

    # Current state
    current_quality_percent: float = Field(..., ge=0, le=100)
    target_quality_percent: float = Field(default=99.0, ge=90, le=100)

    # Operating conditions
    measurements: Optional[SteamMeasurements] = None

    # Context
    carryover_risk_level: Optional[CarryoverRiskLevel] = None
    active_events: Optional[List[EventType]] = None

    # Options
    max_recommendations: int = Field(default=5, ge=1, le=20)
    include_automated_actions: bool = Field(default=True)
    include_manual_actions: bool = Field(default=True)

    # Constraints
    max_implementation_time_min: Optional[float] = Field(None, ge=0)


class RecommendationsResponse(TimestampedModel):
    """Response with control recommendations."""
    request_id: UUID
    header_id: str
    success: bool

    recommendations: List[QualityRecommendation] = Field(default_factory=list)

    # Summary
    total_recommendations: int
    critical_count: int
    high_priority_count: int

    # Aggregate impact
    total_expected_improvement_percent: Optional[float] = None
    total_expected_energy_savings_kw: Optional[float] = None

    # Processing
    processing_time_ms: float = Field(..., ge=0)
    provenance_hash: str

    error_message: Optional[str] = None


# =============================================================================
# Metrics Models
# =============================================================================

class QualityMetrics(BaseModel):
    """Steam quality metrics and KPIs."""
    header_id: str
    period_start: datetime
    period_end: datetime

    # Quality metrics
    average_quality_percent: float = Field(..., ge=0, le=100)
    min_quality_percent: float = Field(..., ge=0, le=100)
    max_quality_percent: float = Field(..., ge=0, le=100)
    std_quality_percent: float = Field(..., ge=0)

    # Target compliance
    time_on_target_percent: float = Field(..., ge=0, le=100)
    time_below_alarm_percent: float = Field(..., ge=0, le=100)

    # Carryover metrics
    carryover_event_count: int = Field(default=0, ge=0)
    total_carryover_duration_min: float = Field(default=0, ge=0)

    # Energy impact
    estimated_energy_loss_kwh: Optional[float] = Field(None, ge=0)
    estimated_steam_loss_kg: Optional[float] = Field(None, ge=0)

    # Improvement metrics
    quality_trend: str = Field(..., pattern="^(improving|stable|degrading)$")
    improvement_vs_previous_period_percent: Optional[float] = None


class MetricsRequest(BaseModel):
    """Request for quality metrics."""
    header_id: Optional[str] = Field(None, description="Filter by header (None = all)")
    period_hours: int = Field(default=24, ge=1, le=720)
    aggregation: str = Field(default="hourly", pattern="^(hourly|daily|weekly)$")


class MetricsResponse(TimestampedModel):
    """Response with quality metrics."""
    request_id: UUID = Field(default_factory=uuid4)
    success: bool

    metrics: List[QualityMetrics] = Field(default_factory=list)

    # Summary across all headers
    overall_average_quality: Optional[float] = None
    overall_time_on_target_percent: Optional[float] = None
    total_carryover_events: int = Field(default=0)

    # Processing
    processing_time_ms: float = Field(..., ge=0)

    error_message: Optional[str] = None


# =============================================================================
# Common Response Models
# =============================================================================

class ErrorResponse(BaseModel):
    """Standard error response."""
    error_code: str
    error_message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[UUID] = None


class HealthStatus(BaseModel):
    """Health check response."""
    status: str = Field(..., pattern="^(healthy|degraded|unhealthy)$")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str
    service: str = "GL-012 SteamQual"

    # Latency targets
    sensor_to_metric_latency_ms: Optional[float] = None
    event_emission_latency_ms: Optional[float] = None

    # Target compliance
    latency_targets_met: bool = True


class ServiceHealth(BaseModel):
    """Health status of a service component."""
    service_name: str
    status: str = Field(..., pattern="^(healthy|degraded|unhealthy|disabled)$")
    latency_ms: float = Field(..., ge=0)
    last_check: datetime
    error_message: Optional[str] = None


class SystemStatus(BaseModel):
    """Overall system status."""
    status: str = Field(..., pattern="^(healthy|degraded|unhealthy)$")
    version: str
    uptime_seconds: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    services: List[ServiceHealth]
    active_connections: int
    requests_per_minute: float

    # Latency SLA
    sensor_to_metric_p99_ms: Optional[float] = None
    event_emission_p99_ms: Optional[float] = None
