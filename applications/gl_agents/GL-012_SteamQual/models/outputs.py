"""
GL-012_SteamQual - Output Models

Pydantic v2 output models for steam quality monitoring and control.
These models define the data structures for all outputs from the SteamQual agent.

Features:
- Full Pydantic v2 validation
- Provenance tracking with SHA-256 hashes
- JSON serialization support for API and audit trail
- GL-003 UnifiedSteam interface compatibility
- Comprehensive docstrings for API documentation

Standards Reference:
- GreenLang provenance tracking standards
- SI units with explicit unit suffixes

Author: GL-BackendDeveloper
Version: 1.0.0
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
    ConfigDict,
    computed_field,
)

from .domain import (
    AlarmPriority,
    ConstraintType,
    ConsumerClass,
    EstimationMethod,
    EventType,
    RecommendationAction,
    Severity,
    SteamState,
)


# =============================================================================
# Base Output Model
# =============================================================================


class BaseOutputModel(BaseModel):
    """
    Base class for all output models with common configuration.

    Provides:
    - JSON serialization configuration
    - Provenance hash field
    - Common timestamp handling
    """

    model_config = ConfigDict(
        json_schema_extra={
            "description": "GL-012_SteamQual Output Model"
        },
        validate_assignment=True,
        extra="forbid",
    )


# =============================================================================
# Quality Estimate Model
# =============================================================================


class QualityEstimate(BaseOutputModel):
    """
    Steam quality (dryness fraction) estimation result.

    Contains the estimated quality value along with uncertainty
    quantification, confidence metrics, and provenance tracking.

    This is the primary output of the SteamQual quality estimation
    pipeline and feeds into GL-003 optimization constraints.

    Attributes:
        x_est: Estimated steam quality (dryness fraction) 0-1.
        uncertainty: Standard uncertainty in quality estimate.
        confidence: Confidence level in the estimate (0-1).
        method: Estimation method used.
        provenance_hash: SHA-256 hash for audit trail.

    Example:
        >>> estimate = QualityEstimate(
        ...     x_est=0.95,
        ...     uncertainty=0.02,
        ...     confidence=0.85,
        ...     method=EstimationMethod.ENTHALPY,
        ...     provenance_hash="abc123..."
        ... )
    """

    # Core estimate
    x_est: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Estimated steam quality (dryness fraction) 0-1",
        json_schema_extra={"examples": [0.95, 0.98, 0.92]},
    )

    uncertainty: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Standard uncertainty in quality estimate (1-sigma)",
        json_schema_extra={"examples": [0.02, 0.05, 0.01]},
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence level in the estimate (0-1)",
        json_schema_extra={"examples": [0.85, 0.90, 0.95]},
    )

    method: EstimationMethod = Field(
        ...,
        description="Primary estimation method used",
    )

    provenance_hash: str = Field(
        ...,
        min_length=64,
        max_length=64,
        description="SHA-256 provenance hash for audit trail",
    )

    # Additional estimate details
    estimate_id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this estimate",
    )

    header_id: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Header ID this estimate applies to",
    )

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Estimate timestamp (UTC)",
    )

    # Uncertainty bounds
    x_lower: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Lower bound of quality estimate (95% CI)",
    )

    x_upper: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Upper bound of quality estimate (95% CI)",
    )

    # Measurement context
    pressure_kpa: Optional[float] = Field(
        default=None,
        ge=0,
        description="Pressure at which estimate applies (kPa)",
    )

    temperature_c: Optional[float] = Field(
        default=None,
        description="Temperature at which estimate applies (C)",
    )

    flow_kg_s: Optional[float] = Field(
        default=None,
        ge=0,
        description="Flow rate at which estimate applies (kg/s)",
    )

    # Inferred state
    inferred_state: SteamState = Field(
        default=SteamState.SATURATED,
        description="Inferred thermodynamic state",
    )

    # Method details
    alternative_estimates: List[Tuple[float, str]] = Field(
        default_factory=list,
        description="Alternative estimates from other methods: (value, method)",
    )

    evidence: List[str] = Field(
        default_factory=list,
        description="Evidence/reasoning supporting the estimate",
    )

    # Processing metadata
    processing_time_ms: Optional[float] = Field(
        default=None,
        ge=0,
        description="Processing time in milliseconds",
    )

    warnings: List[str] = Field(
        default_factory=list,
        description="Warnings generated during estimation",
    )

    @field_validator("provenance_hash")
    @classmethod
    def validate_provenance_hash_format(cls, v: str) -> str:
        """Validate provenance hash is valid SHA-256 hex string."""
        v = v.lower()
        if len(v) != 64:
            raise ValueError("Provenance hash must be 64 characters (SHA-256)")
        if not all(c in "0123456789abcdef" for c in v):
            raise ValueError("Provenance hash must be hexadecimal")
        return v

    @model_validator(mode="after")
    def validate_bounds_consistency(self) -> "QualityEstimate":
        """Validate uncertainty bounds are consistent."""
        if self.x_lower is not None and self.x_upper is not None:
            if self.x_lower > self.x_upper:
                raise ValueError(
                    f"x_lower ({self.x_lower}) cannot exceed x_upper ({self.x_upper})"
                )
            if not (self.x_lower <= self.x_est <= self.x_upper):
                # Allow small tolerance for numerical precision
                if self.x_est < self.x_lower - 0.001 or self.x_est > self.x_upper + 0.001:
                    raise ValueError(
                        f"x_est ({self.x_est}) must be within bounds "
                        f"[{self.x_lower}, {self.x_upper}]"
                    )
        return self

    @computed_field
    @property
    def moisture_content(self) -> float:
        """Calculate moisture content from quality."""
        return 1.0 - self.x_est

    @computed_field
    @property
    def is_acceptable_for_turbine(self) -> bool:
        """Check if quality is acceptable for turbine use (x >= 0.995)."""
        return self.x_est >= 0.995

    def meets_consumer_requirement(self, consumer_class: ConsumerClass) -> bool:
        """Check if quality meets requirements for given consumer class."""
        return self.x_est >= consumer_class.min_quality


# =============================================================================
# Carryover Risk Assessment Model
# =============================================================================


class ContributingFactor(BaseOutputModel):
    """
    Factor contributing to carryover risk.

    Describes a single factor that increases or decreases
    the likelihood of moisture carryover.
    """

    factor_name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Name of the contributing factor",
    )

    factor_value: float = Field(
        ...,
        description="Current value of the factor",
    )

    reference_value: Optional[float] = Field(
        default=None,
        description="Reference/normal value for comparison",
    )

    impact_score: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Impact on risk (-1 to 1, positive = increases risk)",
    )

    explanation: str = Field(
        default="",
        max_length=500,
        description="Explanation of how this factor affects risk",
    )


class CarryoverRiskAssessment(BaseOutputModel):
    """
    Assessment of liquid carryover risk.

    Evaluates the risk of moisture carryover to downstream
    consumers based on current operating conditions.

    Attributes:
        risk_score: Overall risk score (0-100).
        contributing_factors: Factors affecting risk.
        recommendations: Recommended actions to mitigate risk.

    Example:
        >>> assessment = CarryoverRiskAssessment(
        ...     risk_score=35.0,
        ...     contributing_factors=[...],
        ...     recommendations=["Increase separator drain rate"]
        ... )
    """

    risk_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Overall carryover risk score (0-100)",
        json_schema_extra={"examples": [25.0, 50.0, 75.0]},
    )

    contributing_factors: List[ContributingFactor] = Field(
        default_factory=list,
        description="Factors contributing to the risk assessment",
    )

    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommended actions to mitigate carryover risk",
    )

    # Assessment metadata
    assessment_id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this assessment",
    )

    header_id: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Header ID this assessment applies to",
    )

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Assessment timestamp (UTC)",
    )

    # Risk categorization
    risk_category: str = Field(
        default="low",
        pattern="^(low|medium|high|critical)$",
        description="Risk category classification",
    )

    # Affected consumers
    affected_consumers: List[str] = Field(
        default_factory=list,
        description="Consumer IDs at risk from carryover",
    )

    # Confidence
    assessment_confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence in the risk assessment",
    )

    # Provenance
    provenance_hash: Optional[str] = Field(
        default=None,
        min_length=64,
        max_length=64,
        description="SHA-256 provenance hash for audit trail",
    )

    @field_validator("risk_category", mode="before")
    @classmethod
    def categorize_risk(cls, v: str) -> str:
        """Ensure risk category is lowercase."""
        return v.lower()

    @computed_field
    @property
    def requires_action(self) -> bool:
        """Check if risk level requires operator action."""
        return self.risk_score >= 50.0 or self.risk_category in ("high", "critical")


# =============================================================================
# GL-003 Interface Models
# =============================================================================


class QualityState(BaseOutputModel):
    """
    Quality state for GL-003 UnifiedSteam interface.

    Provides steam quality information in the format expected
    by the GL-003 optimization engine.

    This model is designed for direct consumption by GL-003's
    optimizer and constraint engine.

    Attributes:
        header_id: Steam header identifier.
        quality_x: Current steam quality estimate.
        quality_uncertainty: Uncertainty in quality estimate.
        state: Thermodynamic state classification.
        is_acceptable: Whether quality meets consumer requirements.

    Example:
        >>> state = QualityState(
        ...     header_id="MP-01",
        ...     quality_x=0.95,
        ...     quality_uncertainty=0.02,
        ...     state=SteamState.SATURATED,
        ...     is_acceptable=True
        ... )
    """

    header_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Steam header identifier",
    )

    quality_x: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Current steam quality (dryness fraction)",
    )

    quality_uncertainty: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Uncertainty in quality estimate (1-sigma)",
    )

    state: SteamState = Field(
        ...,
        description="Thermodynamic state classification",
    )

    is_acceptable: bool = Field(
        ...,
        description="Whether quality meets consumer requirements",
    )

    # Operating conditions
    pressure_kpa: float = Field(
        ...,
        ge=0,
        description="Header pressure (kPa)",
    )

    temperature_c: float = Field(
        ...,
        description="Header temperature (C)",
    )

    flow_kg_s: float = Field(
        ...,
        ge=0,
        description="Total header flow (kg/s)",
    )

    # Consumer context
    consumer_class: ConsumerClass = Field(
        default=ConsumerClass.GENERAL,
        description="Primary consumer class for this header",
    )

    min_acceptable_quality: float = Field(
        default=0.90,
        ge=0.0,
        le=1.0,
        description="Minimum acceptable quality for consumers",
    )

    # Timestamp
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="State timestamp (UTC)",
    )

    # Data quality
    data_confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence in the data quality",
    )

    # GL-003 integration metadata
    gl003_version_compatible: str = Field(
        default="1.0",
        description="Compatible GL-003 interface version",
    )

    @computed_field
    @property
    def moisture_content(self) -> float:
        """Calculate moisture content (1 - quality)."""
        return 1.0 - self.quality_x

    @computed_field
    @property
    def quality_margin(self) -> float:
        """Calculate margin above minimum acceptable quality."""
        return self.quality_x - self.min_acceptable_quality

    def to_gl003_dict(self) -> Dict[str, Any]:
        """
        Convert to GL-003 compatible dictionary format.

        Returns dictionary in exact format expected by GL-003.
        """
        return {
            "header_id": self.header_id,
            "steam_quality_x": self.quality_x,
            "steam_quality_uncertainty": self.quality_uncertainty,
            "steam_state": self.state.value,
            "quality_acceptable": self.is_acceptable,
            "pressure_kpa": self.pressure_kpa,
            "temperature_c": self.temperature_c,
            "mass_flow_kg_s": self.flow_kg_s,
            "consumer_class": self.consumer_class.value,
            "timestamp_utc": self.timestamp.isoformat(),
        }


class QualityConstraint(BaseOutputModel):
    """
    Single quality constraint for GL-003 optimizer.

    Defines a constraint on steam quality that must be
    respected by the optimization engine.
    """

    constraint_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique constraint identifier",
    )

    constraint_type: ConstraintType = Field(
        ...,
        description="Type of constraint",
    )

    header_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Header ID this constraint applies to",
    )

    limit_value: float = Field(
        ...,
        description="Constraint limit value",
    )

    is_hard_constraint: bool = Field(
        default=True,
        description="True if constraint cannot be violated",
    )

    priority: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Constraint priority (1=highest)",
    )

    penalty_weight: float = Field(
        default=1.0,
        ge=0,
        description="Penalty weight for soft constraints",
    )

    source: str = Field(
        default="GL-012_SteamQual",
        max_length=100,
        description="Constraint source agent",
    )

    reason: str = Field(
        default="",
        max_length=500,
        description="Reason for this constraint",
    )


class QualityConstraints(BaseOutputModel):
    """
    Collection of quality constraints for GL-003 optimizer.

    Provides all quality-related constraints that should be
    incorporated into GL-003's optimization problem.

    Attributes:
        constraints: List of individual constraints.
        header_id: Header these constraints apply to.
        consumer_class: Consumer class driving requirements.

    Example:
        >>> constraints = QualityConstraints(
        ...     header_id="MP-01",
        ...     consumer_class=ConsumerClass.TURBINE,
        ...     constraints=[
        ...         QualityConstraint(
        ...             constraint_id="Q-MIN-01",
        ...             constraint_type=ConstraintType.QUALITY_MIN,
        ...             header_id="MP-01",
        ...             limit_value=0.995
        ...         )
        ...     ]
        ... )
    """

    constraints: List[QualityConstraint] = Field(
        default_factory=list,
        description="List of quality constraints",
    )

    header_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Header ID these constraints apply to",
    )

    consumer_class: ConsumerClass = Field(
        ...,
        description="Consumer class driving constraint requirements",
    )

    # Constraint set metadata
    constraint_set_id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this constraint set",
    )

    valid_from: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Constraint validity start time",
    )

    valid_until: Optional[datetime] = Field(
        default=None,
        description="Constraint validity end time (None = indefinite)",
    )

    # Generation metadata
    generated_by: str = Field(
        default="GL-012_SteamQual",
        description="Agent that generated these constraints",
    )

    generation_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When constraints were generated",
    )

    # Provenance
    provenance_hash: Optional[str] = Field(
        default=None,
        min_length=64,
        max_length=64,
        description="SHA-256 provenance hash for audit trail",
    )

    def get_quality_min_constraint(self) -> Optional[QualityConstraint]:
        """Get the minimum quality constraint if present."""
        for c in self.constraints:
            if c.constraint_type == ConstraintType.QUALITY_MIN:
                return c
        return None

    def get_hard_constraints(self) -> List[QualityConstraint]:
        """Get all hard (non-violable) constraints."""
        return [c for c in self.constraints if c.is_hard_constraint]

    def to_gl003_format(self) -> Dict[str, Any]:
        """
        Convert to GL-003 optimizer format.

        Returns dictionary in exact format expected by GL-003.
        """
        return {
            "header_id": self.header_id,
            "consumer_class": self.consumer_class.value,
            "constraints": [
                {
                    "id": c.constraint_id,
                    "type": c.constraint_type.value,
                    "limit": c.limit_value,
                    "is_hard": c.is_hard_constraint,
                    "priority": c.priority,
                    "penalty": c.penalty_weight,
                }
                for c in self.constraints
            ],
            "valid_from": self.valid_from.isoformat(),
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
        }


# =============================================================================
# Quality Event Model
# =============================================================================


class QualityEvent(BaseOutputModel):
    """
    Steam quality event notification.

    Generated when quality conditions change significantly,
    thresholds are crossed, or anomalies are detected.

    Attributes:
        event_type: Type of quality event.
        severity: Event severity level.
        evidence: Evidence supporting the event.
        timestamp: When the event occurred.

    Example:
        >>> event = QualityEvent(
        ...     event_type=EventType.LOW_DRYNESS,
        ...     severity=Severity.S2_WARNING,
        ...     evidence=["Quality 0.85 below threshold 0.90"],
        ...     message="Low steam quality detected on MP-01"
        ... )
    """

    event_type: EventType = Field(
        ...,
        description="Type of quality event",
    )

    severity: Severity = Field(
        ...,
        description="Event severity level",
    )

    evidence: List[str] = Field(
        default_factory=list,
        description="Evidence supporting this event",
    )

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the event occurred (UTC)",
    )

    # Event identification
    event_id: UUID = Field(
        default_factory=uuid4,
        description="Unique event identifier",
    )

    # Context
    header_id: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Affected header ID",
    )

    separator_id: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Affected separator ID",
    )

    consumer_id: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Affected consumer ID",
    )

    # Event details
    message: str = Field(
        default="",
        max_length=1000,
        description="Human-readable event message",
    )

    current_value: Optional[float] = Field(
        default=None,
        description="Current value that triggered the event",
    )

    threshold_value: Optional[float] = Field(
        default=None,
        description="Threshold value that was crossed",
    )

    # Alarm integration
    alarm_priority: AlarmPriority = Field(
        default=AlarmPriority.LOW,
        description="Alarm priority for operator notification",
    )

    requires_acknowledgment: bool = Field(
        default=False,
        description="Whether event requires operator acknowledgment",
    )

    # Recommended action
    recommended_action: RecommendationAction = Field(
        default=RecommendationAction.MONITOR,
        description="Recommended response action",
    )

    action_details: str = Field(
        default="",
        max_length=500,
        description="Details for recommended action",
    )

    # State tracking
    is_active: bool = Field(
        default=True,
        description="Whether event is still active",
    )

    cleared_at: Optional[datetime] = Field(
        default=None,
        description="When event was cleared (if applicable)",
    )

    acknowledged_at: Optional[datetime] = Field(
        default=None,
        description="When event was acknowledged (if applicable)",
    )

    acknowledged_by: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Who acknowledged the event",
    )

    # Provenance
    provenance_hash: Optional[str] = Field(
        default=None,
        min_length=64,
        max_length=64,
        description="SHA-256 provenance hash for audit trail",
    )

    @computed_field
    @property
    def is_critical(self) -> bool:
        """Check if event is critical severity."""
        return self.severity == Severity.S3_CRITICAL

    @computed_field
    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate event duration if cleared."""
        if self.cleared_at is not None:
            return (self.cleared_at - self.timestamp).total_seconds()
        return None

    def mark_cleared(self) -> None:
        """Mark the event as cleared."""
        self.is_active = False
        self.cleared_at = datetime.now(timezone.utc)

    def mark_acknowledged(self, acknowledged_by: str) -> None:
        """Mark the event as acknowledged."""
        self.acknowledged_at = datetime.now(timezone.utc)
        self.acknowledged_by = acknowledged_by


# =============================================================================
# Complete Estimation Response Model
# =============================================================================


class QualityEstimationResponse(BaseOutputModel):
    """
    Complete response from the quality estimation pipeline.

    Contains all outputs from a quality estimation request including
    estimates, risk assessments, events, and GL-003 interface data.

    Attributes:
        request_id: Original request identifier.
        estimates: Quality estimates by header.
        risk_assessments: Carryover risk assessments.
        events: Quality events generated.
        quality_states: GL-003 compatible quality states.
        constraints: GL-003 compatible constraints.

    Example:
        >>> response = QualityEstimationResponse(
        ...     request_id=uuid4(),
        ...     estimates=[...],
        ...     quality_states=[...],
        ...     success=True
        ... )
    """

    # Request tracking
    request_id: UUID = Field(
        ...,
        description="Original request identifier",
    )

    # Results
    estimates: List[QualityEstimate] = Field(
        default_factory=list,
        description="Quality estimates by header",
    )

    risk_assessments: List[CarryoverRiskAssessment] = Field(
        default_factory=list,
        description="Carryover risk assessments",
    )

    events: List[QualityEvent] = Field(
        default_factory=list,
        description="Quality events generated",
    )

    # GL-003 interface outputs
    quality_states: List[QualityState] = Field(
        default_factory=list,
        description="Quality states for GL-003 interface",
    )

    constraints: List[QualityConstraints] = Field(
        default_factory=list,
        description="Quality constraints for GL-003 optimizer",
    )

    # Response metadata
    response_id: UUID = Field(
        default_factory=uuid4,
        description="Unique response identifier",
    )

    success: bool = Field(
        ...,
        description="Whether processing was successful",
    )

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Response timestamp (UTC)",
    )

    processing_time_ms: float = Field(
        default=0.0,
        ge=0,
        description="Total processing time in milliseconds",
    )

    # Error handling
    error_message: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Error message if success is False",
    )

    warnings: List[str] = Field(
        default_factory=list,
        description="Warnings generated during processing",
    )

    # Agent metadata
    agent_id: str = Field(
        default="GL-012",
        description="Agent identifier",
    )

    agent_version: str = Field(
        default="1.0.0",
        description="Agent version",
    )

    # Provenance
    computation_hash: Optional[str] = Field(
        default=None,
        min_length=64,
        max_length=64,
        description="SHA-256 hash of entire computation for audit",
    )

    def get_estimate_for_header(self, header_id: str) -> Optional[QualityEstimate]:
        """Get quality estimate for a specific header."""
        for estimate in self.estimates:
            if estimate.header_id == header_id:
                return estimate
        return None

    def get_events_by_severity(self, severity: Severity) -> List[QualityEvent]:
        """Get all events of a specific severity."""
        return [e for e in self.events if e.severity == severity]

    def get_critical_events(self) -> List[QualityEvent]:
        """Get all critical events."""
        return self.get_events_by_severity(Severity.S3_CRITICAL)

    def has_quality_issues(self) -> bool:
        """Check if any quality issues were detected."""
        return any(
            e.event_type in (EventType.LOW_DRYNESS, EventType.HIGH_MOISTURE, EventType.CARRYOVER_DETECTED)
            for e in self.events
        )

    @computed_field
    @property
    def event_count_by_severity(self) -> Dict[str, int]:
        """Count of events by severity level."""
        counts: Dict[str, int] = {}
        for event in self.events:
            key = event.severity.value
            counts[key] = counts.get(key, 0) + 1
        return counts
