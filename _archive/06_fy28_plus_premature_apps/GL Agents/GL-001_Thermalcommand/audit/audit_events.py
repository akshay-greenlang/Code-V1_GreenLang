"""
Audit Event Types for GL-001 ThermalCommand

This module defines strongly-typed audit event models for the ThermalCommand
orchestrator. All events are immutable, timestamped, and include correlation
IDs for distributed tracing across services.

Event Types:
    - DecisionAuditEvent: Optimization decisions and solver outcomes
    - ActionAuditEvent: Recommended and executed control actions
    - SafetyAuditEvent: Safety boundary checks and violations
    - ComplianceAuditEvent: Regulatory compliance verifications

Example:
    >>> from audit_events import DecisionAuditEvent, SolverStatus
    >>> event = DecisionAuditEvent(
    ...     correlation_id="corr-12345",
    ...     asset_id="boiler-001",
    ...     solver_status=SolverStatus.OPTIMAL,
    ...     objective_value=125000.50
    ... )
    >>> event.event_hash
    'a1b2c3d4...'

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator, root_validator


class EventType(str, Enum):
    """Enumeration of all audit event types."""

    DECISION = "DECISION"
    ACTION = "ACTION"
    SAFETY = "SAFETY"
    COMPLIANCE = "COMPLIANCE"
    SYSTEM = "SYSTEM"
    OVERRIDE = "OVERRIDE"


class SolverStatus(str, Enum):
    """MILP solver termination status."""

    OPTIMAL = "OPTIMAL"
    FEASIBLE = "FEASIBLE"
    INFEASIBLE = "INFEASIBLE"
    UNBOUNDED = "UNBOUNDED"
    TIMEOUT = "TIMEOUT"
    ERROR = "ERROR"


class ActionStatus(str, Enum):
    """Control action execution status."""

    RECOMMENDED = "RECOMMENDED"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXECUTED = "EXECUTED"
    FAILED = "FAILED"
    OVERRIDDEN = "OVERRIDDEN"
    PENDING = "PENDING"


class SafetyLevel(str, Enum):
    """Safety event severity levels."""

    INFO = "INFO"
    WARNING = "WARNING"
    ALARM = "ALARM"
    TRIP = "TRIP"
    EMERGENCY = "EMERGENCY"


class ComplianceStatus(str, Enum):
    """Compliance check result status."""

    COMPLIANT = "COMPLIANT"
    NON_COMPLIANT = "NON_COMPLIANT"
    PENDING_REVIEW = "PENDING_REVIEW"
    WAIVER_APPLIED = "WAIVER_APPLIED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class OperatorType(str, Enum):
    """Entity that initiated or approved an action."""

    AGENT = "AGENT"
    OPERATOR = "OPERATOR"
    SUPERVISOR = "SUPERVISOR"
    SYSTEM = "SYSTEM"
    SAFETY_SYSTEM = "SAFETY_SYSTEM"


class ModelVersionInfo(BaseModel):
    """ML model version tracking information."""

    model_name: str = Field(..., description="Model identifier")
    model_version: str = Field(..., description="Semantic version")
    model_hash: str = Field(..., description="SHA-256 hash of model artifacts")
    training_date: Optional[datetime] = Field(None, description="Model training timestamp")
    metrics: Optional[Dict[str, float]] = Field(None, description="Validation metrics")

    class Config:
        frozen = True
        json_encoders = {datetime: lambda v: v.isoformat()}


class InputDatasetReference(BaseModel):
    """Reference to input datasets used in decisions."""

    dataset_id: str = Field(..., description="Unique dataset identifier")
    dataset_type: str = Field(..., description="Type of dataset (sensor, forecast, etc.)")
    schema_version: str = Field(..., description="Schema version used")
    data_hash: str = Field(..., description="SHA-256 hash of dataset content")
    record_count: int = Field(..., ge=0, description="Number of records")
    time_range_start: Optional[datetime] = Field(None, description="Start of data time range")
    time_range_end: Optional[datetime] = Field(None, description="End of data time range")
    source_system: str = Field(..., description="Source system identifier")

    class Config:
        frozen = True
        json_encoders = {datetime: lambda v: v.isoformat()}


class ConstraintInfo(BaseModel):
    """Information about optimization constraints."""

    constraint_id: str = Field(..., description="Unique constraint identifier")
    constraint_name: str = Field(..., description="Human-readable constraint name")
    constraint_type: str = Field(..., description="Type: equality, inequality, bound")
    is_binding: bool = Field(..., description="Whether constraint is active at solution")
    slack_value: Optional[float] = Field(None, description="Slack/surplus value")
    dual_value: Optional[float] = Field(None, description="Shadow price/dual value")

    class Config:
        frozen = True


class RecommendedAction(BaseModel):
    """A recommended control action from the optimizer."""

    action_id: str = Field(..., description="Unique action identifier")
    tag_id: str = Field(..., description="OPC/control tag identifier")
    asset_id: str = Field(..., description="Asset receiving the action")
    current_value: float = Field(..., description="Current setpoint value")
    recommended_value: float = Field(..., description="New recommended setpoint")
    lower_bound: float = Field(..., description="Allowable lower bound")
    upper_bound: float = Field(..., description="Allowable upper bound")
    ramp_rate: Optional[float] = Field(None, description="Max change rate per minute")
    ramp_duration_s: Optional[float] = Field(None, description="Ramp duration in seconds")
    unit: str = Field(..., description="Engineering unit")
    priority: int = Field(1, ge=1, le=10, description="Action priority 1-10")
    rationale: Optional[str] = Field(None, description="Explanation for action")

    class Config:
        frozen = True


class ExpectedImpact(BaseModel):
    """Expected impact of recommended actions."""

    cost_delta_usd: float = Field(..., description="Expected cost change in USD")
    emissions_delta_kg_co2e: float = Field(..., description="Expected emissions change in kg CO2e")
    energy_delta_mmbtu: float = Field(..., description="Expected energy change in MMBtu")
    efficiency_delta_pct: float = Field(..., description="Expected efficiency change in %")
    risk_score_delta: float = Field(..., description="Expected risk score change")
    confidence_interval_lower: float = Field(..., description="95% CI lower bound")
    confidence_interval_upper: float = Field(..., description="95% CI upper bound")
    horizon_minutes: int = Field(..., ge=1, description="Forecast horizon in minutes")

    class Config:
        frozen = True


class ExplainabilityArtifact(BaseModel):
    """SHAP/LIME explainability artifacts."""

    artifact_type: str = Field(..., description="SHAP, LIME, or other method")
    model_name: str = Field(..., description="Model being explained")
    feature_importances: Dict[str, float] = Field(..., description="Feature importance scores")
    base_value: Optional[float] = Field(None, description="Expected value baseline")
    interaction_effects: Optional[Dict[str, Dict[str, float]]] = Field(
        None, description="Feature interaction effects"
    )
    artifact_hash: str = Field(..., description="SHA-256 hash of full artifact")

    class Config:
        frozen = True


class UncertaintyQuantification(BaseModel):
    """Uncertainty quantification intervals."""

    prediction_mean: float = Field(..., description="Point estimate")
    prediction_std: float = Field(..., ge=0, description="Standard deviation")
    ci_lower_95: float = Field(..., description="95% CI lower bound")
    ci_upper_95: float = Field(..., description="95% CI upper bound")
    ci_lower_99: float = Field(..., description="99% CI lower bound")
    ci_upper_99: float = Field(..., description="99% CI upper bound")
    method: str = Field(..., description="UQ method used (ensemble, dropout, etc.)")

    class Config:
        frozen = True


class BaseAuditEvent(BaseModel):
    """
    Base class for all audit events.

    All audit events inherit from this base class which provides:
    - Unique event ID
    - Correlation ID for distributed tracing
    - Timestamps
    - Event hashing for integrity verification
    - Previous hash for chain linking
    """

    event_id: UUID = Field(default_factory=uuid4, description="Unique event identifier")
    correlation_id: str = Field(..., description="Correlation ID for distributed tracing")
    event_type: EventType = Field(..., description="Type of audit event")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Event creation timestamp"
    )
    asset_id: str = Field(..., description="Asset or equipment identifier")
    facility_id: Optional[str] = Field(None, description="Facility/plant identifier")
    agent_id: str = Field(default="GL-001", description="Agent that generated event")
    agent_version: str = Field(default="1.0.0", description="Agent software version")
    operator_id: Optional[str] = Field(None, description="Operator ID if applicable")
    operator_type: OperatorType = Field(
        default=OperatorType.AGENT, description="Type of entity"
    )
    previous_event_hash: Optional[str] = Field(
        None, description="Hash of previous event in chain"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    class Config:
        frozen = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }

    @property
    def event_hash(self) -> str:
        """
        Calculate SHA-256 hash of event for integrity verification.

        Returns:
            Hex-encoded SHA-256 hash of event content.
        """
        # Create deterministic JSON representation
        event_data = self.dict(exclude={"event_hash"})
        # Sort keys for deterministic ordering
        json_str = json.dumps(event_data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()

    def to_chain_dict(self) -> Dict[str, Any]:
        """
        Convert event to dictionary suitable for hash chain linking.

        Returns:
            Dictionary with all fields including computed hash.
        """
        data = self.dict()
        data["event_hash"] = self.event_hash
        return data


class DecisionAuditEvent(BaseAuditEvent):
    """
    Audit event for optimization decisions.

    Captures complete decision context including inputs, solver status,
    constraints, recommended actions, and expected impacts.

    Example:
        >>> event = DecisionAuditEvent(
        ...     correlation_id="corr-12345",
        ...     asset_id="boiler-001",
        ...     solver_status=SolverStatus.OPTIMAL,
        ...     objective_value=125000.50
        ... )
    """

    event_type: EventType = Field(default=EventType.DECISION, const=True)

    # Timestamps
    ingestion_timestamp: datetime = Field(..., description="Data ingestion timestamp")
    decision_timestamp: datetime = Field(..., description="Decision computation timestamp")

    # Input data references
    input_datasets: List[InputDatasetReference] = Field(
        default_factory=list, description="Input datasets used"
    )
    unit_conversion_version: str = Field(
        default="1.0.0", description="Unit conversion library version"
    )

    # Constraint and policy versions
    constraint_set_id: str = Field(..., description="Constraint set identifier")
    constraint_set_version: str = Field(..., description="Constraint set version")
    safety_boundary_policy_version: str = Field(
        ..., description="Safety boundary policy version"
    )

    # Model versions
    demand_model: Optional[ModelVersionInfo] = Field(None, description="Demand forecast model")
    health_model: Optional[ModelVersionInfo] = Field(None, description="Equipment health model")
    anomaly_model: Optional[ModelVersionInfo] = Field(None, description="Anomaly detection model")

    # Explainability
    shap_artifacts: Optional[List[ExplainabilityArtifact]] = Field(
        None, description="SHAP explanation artifacts"
    )
    lime_artifacts: Optional[List[ExplainabilityArtifact]] = Field(
        None, description="LIME explanation artifacts"
    )
    uncertainty_quantification: Optional[Dict[str, UncertaintyQuantification]] = Field(
        None, description="UQ intervals by prediction"
    )

    # Solver results
    solver_status: SolverStatus = Field(..., description="Solver termination status")
    solver_name: str = Field(default="HiGHS", description="Solver used")
    solver_version: str = Field(default="1.5.0", description="Solver version")
    solve_time_ms: float = Field(..., ge=0, description="Solve time in milliseconds")
    mip_gap: Optional[float] = Field(None, ge=0, description="MIP optimality gap")

    # Objective
    objective_value: float = Field(..., description="Objective function value")
    objective_breakdown: Dict[str, float] = Field(
        default_factory=dict, description="Objective components breakdown"
    )

    # Constraints
    binding_constraints: List[ConstraintInfo] = Field(
        default_factory=list, description="Binding constraints at solution"
    )
    constraint_violations: List[ConstraintInfo] = Field(
        default_factory=list, description="Any constraint violations"
    )

    # Recommended actions
    recommended_actions: List[RecommendedAction] = Field(
        default_factory=list, description="Recommended control actions"
    )

    # Expected impact
    expected_impact: Optional[ExpectedImpact] = Field(
        None, description="Expected impact of recommendations"
    )

    @validator("decision_timestamp")
    def decision_after_ingestion(cls, v, values):
        """Validate decision timestamp is after ingestion."""
        if "ingestion_timestamp" in values and v < values["ingestion_timestamp"]:
            raise ValueError("decision_timestamp must be >= ingestion_timestamp")
        return v


class ActionAuditEvent(BaseAuditEvent):
    """
    Audit event for control actions.

    Tracks the lifecycle of recommended actions from recommendation
    through approval, execution, and verification.

    Example:
        >>> event = ActionAuditEvent(
        ...     correlation_id="corr-12345",
        ...     asset_id="boiler-001",
        ...     decision_event_id="dec-67890",
        ...     action=recommended_action,
        ...     action_status=ActionStatus.EXECUTED
        ... )
    """

    event_type: EventType = Field(default=EventType.ACTION, const=True)

    # Reference to decision
    decision_event_id: str = Field(..., description="ID of originating decision event")
    decision_correlation_id: str = Field(..., description="Correlation ID of decision")

    # Action details
    action: RecommendedAction = Field(..., description="The control action")
    action_status: ActionStatus = Field(..., description="Current action status")

    # Timing
    recommended_timestamp: datetime = Field(..., description="When action was recommended")
    actuation_timestamp: Optional[datetime] = Field(
        None, description="When action was executed"
    )
    verification_timestamp: Optional[datetime] = Field(
        None, description="When action was verified"
    )

    # Execution details
    executed_value: Optional[float] = Field(None, description="Actual value executed")
    execution_duration_s: Optional[float] = Field(
        None, ge=0, description="Execution duration in seconds"
    )

    # Operator interaction
    operator_action: Optional[str] = Field(
        None, description="Operator action taken (approve/reject/modify)"
    )
    operator_notes: Optional[str] = Field(None, description="Operator notes or comments")
    modification_reason: Optional[str] = Field(
        None, description="Reason for modification if any"
    )

    # Verification
    verification_status: Optional[str] = Field(
        None, description="Post-execution verification result"
    )
    actual_impact: Optional[ExpectedImpact] = Field(
        None, description="Measured impact after execution"
    )

    @validator("actuation_timestamp")
    def actuation_after_recommended(cls, v, values):
        """Validate actuation is after recommendation."""
        if v and "recommended_timestamp" in values:
            if v < values["recommended_timestamp"]:
                raise ValueError("actuation_timestamp must be >= recommended_timestamp")
        return v


class SafetyAuditEvent(BaseAuditEvent):
    """
    Audit event for safety boundary checks and violations.

    Captures all safety-related events including boundary checks,
    constraint violations, trips, and emergency actions.

    Example:
        >>> event = SafetyAuditEvent(
        ...     correlation_id="corr-12345",
        ...     asset_id="boiler-001",
        ...     boundary_id="TEMP_HIGH_001",
        ...     safety_level=SafetyLevel.ALARM,
        ...     current_value=850.5,
        ...     boundary_value=850.0
        ... )
    """

    event_type: EventType = Field(default=EventType.SAFETY, const=True)

    # Boundary reference
    boundary_id: str = Field(..., description="Safety boundary identifier")
    boundary_name: str = Field(..., description="Human-readable boundary name")
    boundary_version: str = Field(..., description="Boundary definition version")

    # Safety classification
    safety_level: SafetyLevel = Field(..., description="Severity level")
    safety_category: str = Field(..., description="Safety category (temperature, pressure, etc.)")

    # Values
    tag_id: str = Field(..., description="OPC/sensor tag")
    current_value: float = Field(..., description="Current measured value")
    boundary_value: float = Field(..., description="Boundary threshold value")
    unit: str = Field(..., description="Engineering unit")
    deviation_pct: float = Field(..., description="Deviation from boundary as percentage")

    # Status
    is_violation: bool = Field(..., description="Whether boundary was violated")
    violation_duration_s: Optional[float] = Field(
        None, ge=0, description="Duration of violation in seconds"
    )

    # Response
    automatic_response: Optional[str] = Field(
        None, description="Automatic system response taken"
    )
    sis_action: Optional[str] = Field(None, description="SIS action if triggered")
    requires_operator_action: bool = Field(
        False, description="Whether operator action required"
    )

    # Recommendations
    recommended_response: Optional[str] = Field(
        None, description="Recommended operator response"
    )

    # Related events
    related_decision_id: Optional[str] = Field(
        None, description="Related decision event if applicable"
    )
    related_action_ids: List[str] = Field(
        default_factory=list, description="Related action events"
    )

    @validator("deviation_pct")
    def calculate_deviation(cls, v, values):
        """Validate deviation percentage is reasonable."""
        if abs(v) > 1000:
            raise ValueError("Deviation percentage seems unreasonable (>1000%)")
        return v


class ComplianceAuditEvent(BaseAuditEvent):
    """
    Audit event for regulatory compliance checks.

    Tracks compliance with EPA 40 CFR 98, ISO 50001, and other
    applicable regulations and standards.

    Example:
        >>> event = ComplianceAuditEvent(
        ...     correlation_id="corr-12345",
        ...     asset_id="boiler-001",
        ...     regulation_id="EPA_40_CFR_98",
        ...     compliance_status=ComplianceStatus.COMPLIANT
        ... )
    """

    event_type: EventType = Field(default=EventType.COMPLIANCE, const=True)

    # Regulation reference
    regulation_id: str = Field(..., description="Regulation identifier")
    regulation_name: str = Field(..., description="Regulation name")
    regulation_version: str = Field(..., description="Regulation version/year")
    subpart: Optional[str] = Field(None, description="Regulation subpart if applicable")

    # Requirement
    requirement_id: str = Field(..., description="Specific requirement identifier")
    requirement_description: str = Field(..., description="Requirement description")

    # Compliance status
    compliance_status: ComplianceStatus = Field(..., description="Compliance check result")
    compliance_score: Optional[float] = Field(
        None, ge=0, le=100, description="Compliance score 0-100"
    )

    # Check details
    check_type: str = Field(..., description="Type of check (emission limit, reporting, etc.)")
    check_method: str = Field(..., description="Method used for compliance check")
    check_timestamp: datetime = Field(..., description="When check was performed")

    # Values if applicable
    measured_value: Optional[float] = Field(None, description="Measured value")
    limit_value: Optional[float] = Field(None, description="Regulatory limit")
    margin_pct: Optional[float] = Field(None, description="Margin to limit as percentage")
    unit: Optional[str] = Field(None, description="Engineering unit")

    # Reporting period
    reporting_period_start: Optional[datetime] = Field(
        None, description="Start of reporting period"
    )
    reporting_period_end: Optional[datetime] = Field(
        None, description="End of reporting period"
    )

    # Documentation
    evidence_references: List[str] = Field(
        default_factory=list, description="References to supporting evidence"
    )
    evidence_pack_id: Optional[str] = Field(
        None, description="Associated evidence pack ID"
    )

    # Non-compliance details
    non_compliance_details: Optional[str] = Field(
        None, description="Details if non-compliant"
    )
    corrective_action_required: Optional[str] = Field(
        None, description="Required corrective action"
    )
    corrective_action_deadline: Optional[datetime] = Field(
        None, description="Deadline for corrective action"
    )

    # Waiver/exemption
    waiver_id: Optional[str] = Field(None, description="Waiver ID if applicable")
    waiver_expiry: Optional[datetime] = Field(None, description="Waiver expiry date")


class SystemAuditEvent(BaseAuditEvent):
    """
    Audit event for system-level events.

    Captures system state changes, configuration updates,
    and other infrastructure events.
    """

    event_type: EventType = Field(default=EventType.SYSTEM, const=True)

    # System event type
    system_event_type: str = Field(..., description="Type of system event")
    component: str = Field(..., description="System component affected")

    # State change
    previous_state: Optional[str] = Field(None, description="Previous state")
    new_state: str = Field(..., description="New state")

    # Configuration
    configuration_version: Optional[str] = Field(
        None, description="Configuration version if applicable"
    )
    configuration_hash: Optional[str] = Field(
        None, description="Hash of configuration"
    )

    # Details
    details: Dict[str, Any] = Field(
        default_factory=dict, description="Additional event details"
    )

    # Impact
    service_impact: Optional[str] = Field(
        None, description="Impact on service availability"
    )


class OverrideAuditEvent(BaseAuditEvent):
    """
    Audit event for manual overrides.

    Captures all instances where automated recommendations
    were overridden by operators or supervisors.
    """

    event_type: EventType = Field(default=EventType.OVERRIDE, const=True)

    # Reference to original
    original_event_id: str = Field(..., description="ID of original event/action")
    original_event_type: EventType = Field(..., description="Type of original event")

    # Override details
    override_type: str = Field(..., description="Type of override")
    original_value: Any = Field(..., description="Original recommended value")
    override_value: Any = Field(..., description="Overridden value")

    # Authorization
    authorized_by: str = Field(..., description="ID of authorizing entity")
    authorization_level: str = Field(..., description="Authorization level required")
    authorization_timestamp: datetime = Field(..., description="When override was authorized")

    # Justification
    justification: str = Field(..., description="Reason for override")
    supporting_documentation: List[str] = Field(
        default_factory=list, description="Supporting document references"
    )

    # Risk assessment
    risk_assessment: Optional[str] = Field(
        None, description="Risk assessment of override"
    )
    risk_accepted_by: Optional[str] = Field(
        None, description="Who accepted the risk"
    )

    # Expiry
    override_expiry: Optional[datetime] = Field(
        None, description="When override expires"
    )
    review_required: bool = Field(
        True, description="Whether periodic review required"
    )


# Type alias for all audit events
AuditEvent = Union[
    DecisionAuditEvent,
    ActionAuditEvent,
    SafetyAuditEvent,
    ComplianceAuditEvent,
    SystemAuditEvent,
    OverrideAuditEvent,
]


def create_event_from_dict(data: Dict[str, Any]) -> AuditEvent:
    """
    Factory function to create appropriate event type from dictionary.

    Args:
        data: Dictionary containing event data with 'event_type' field.

    Returns:
        Appropriate AuditEvent subclass instance.

    Raises:
        ValueError: If event_type is unknown.
    """
    event_type = data.get("event_type")

    event_classes = {
        EventType.DECISION: DecisionAuditEvent,
        EventType.ACTION: ActionAuditEvent,
        EventType.SAFETY: SafetyAuditEvent,
        EventType.COMPLIANCE: ComplianceAuditEvent,
        EventType.SYSTEM: SystemAuditEvent,
        EventType.OVERRIDE: OverrideAuditEvent,
    }

    if isinstance(event_type, str):
        event_type = EventType(event_type)

    event_class = event_classes.get(event_type)
    if event_class is None:
        raise ValueError(f"Unknown event type: {event_type}")

    return event_class(**data)
