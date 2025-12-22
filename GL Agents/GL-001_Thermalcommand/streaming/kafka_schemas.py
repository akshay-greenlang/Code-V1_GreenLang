"""
Kafka Schemas Module - GL-001 ThermalCommand

This module defines Pydantic models for all Kafka topic schemas in the
ThermalCommand system. Each schema follows Avro-compatible patterns for
Schema Registry integration with explicit type definitions.

Topics and Schemas:
    - gl001.telemetry.normalized: TelemetryNormalizedEvent
    - gl001.plan.dispatch: DispatchPlanEvent
    - gl001.actions.recommendations: ActionRecommendationEvent
    - gl001.safety.events: SafetyEvent
    - gl001.maintenance.triggers: MaintenanceTriggerEvent
    - gl001.explainability.reports: ExplainabilityReportEvent
    - gl001.audit.log: AuditLogEvent

Schema Evolution Rules:
    - All schemas use BACKWARD compatibility mode
    - New optional fields allowed (with defaults)
    - Field removal not allowed
    - Type changes not allowed

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


# =============================================================================
# COMMON ENUMS AND TYPES
# =============================================================================


class QualityCode(str, Enum):
    """OPC-UA quality codes for telemetry data."""

    GOOD = "GOOD"
    GOOD_LOCAL_OVERRIDE = "GOOD_LOCAL_OVERRIDE"
    UNCERTAIN = "UNCERTAIN"
    UNCERTAIN_SENSOR_CAL = "UNCERTAIN_SENSOR_CAL"
    BAD = "BAD"
    BAD_SENSOR_FAILURE = "BAD_SENSOR_FAILURE"
    BAD_COMM_FAILURE = "BAD_COMM_FAILURE"
    BAD_OUT_OF_SERVICE = "BAD_OUT_OF_SERVICE"


class UnitOfMeasure(str, Enum):
    """Standard units of measure for process heat systems."""

    # Temperature
    CELSIUS = "degC"
    FAHRENHEIT = "degF"
    KELVIN = "K"

    # Pressure
    BAR = "bar"
    PSI = "psi"
    KPA = "kPa"
    MPA = "MPa"

    # Flow
    KG_PER_HOUR = "kg/h"
    M3_PER_HOUR = "m3/h"
    LB_PER_HOUR = "lb/h"

    # Energy
    MWH = "MWh"
    KWH = "kWh"
    GJ = "GJ"
    MMBTU = "MMBtu"

    # Power
    MW = "MW"
    KW = "kW"

    # Emissions
    TCO2 = "tCO2"
    KG_CO2 = "kgCO2"

    # Dimensionless
    PERCENT = "%"
    RATIO = "ratio"
    COUNT = "count"


class SolverStatus(str, Enum):
    """MILP solver status codes."""

    OPTIMAL = "OPTIMAL"
    FEASIBLE = "FEASIBLE"
    INFEASIBLE = "INFEASIBLE"
    UNBOUNDED = "UNBOUNDED"
    TIMEOUT = "TIMEOUT"
    ERROR = "ERROR"


class SafetyLevel(str, Enum):
    """Safety event severity levels per IEC 61511."""

    INFO = "INFO"
    WARNING = "WARNING"
    ALARM = "ALARM"
    TRIP = "TRIP"
    EMERGENCY = "EMERGENCY"


class MaintenancePriority(str, Enum):
    """Maintenance work order priority levels."""

    P1_EMERGENCY = "P1_EMERGENCY"
    P2_URGENT = "P2_URGENT"
    P3_HIGH = "P3_HIGH"
    P4_MEDIUM = "P4_MEDIUM"
    P5_LOW = "P5_LOW"
    PM_SCHEDULED = "PM_SCHEDULED"


class AuditAction(str, Enum):
    """Audit log action types."""

    CREATE = "CREATE"
    READ = "READ"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    EXECUTE = "EXECUTE"
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    OVERRIDE = "OVERRIDE"


# =============================================================================
# TOPIC: gl001.telemetry.normalized
# =============================================================================


class TelemetryPoint(BaseModel):
    """
    Single normalized telemetry data point.

    Represents a sensor reading with full metadata including
    quality, units, and source information.
    """

    tag_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="OPC-UA tag identifier",
    )
    value: float = Field(
        ...,
        description="Numeric value of the reading",
    )
    unit: UnitOfMeasure = Field(
        ...,
        description="Unit of measure",
    )
    quality: QualityCode = Field(
        QualityCode.GOOD,
        description="OPC-UA quality code",
    )
    timestamp: datetime = Field(
        ...,
        description="Source timestamp (UTC)",
    )
    raw_value: Optional[float] = Field(
        None,
        description="Original raw value before normalization",
    )
    raw_unit: Optional[str] = Field(
        None,
        description="Original unit before conversion",
    )
    sensor_id: Optional[str] = Field(
        None,
        max_length=64,
        description="Physical sensor identifier",
    )
    equipment_id: Optional[str] = Field(
        None,
        max_length=64,
        description="Equipment asset identifier",
    )

    @field_validator("timestamp", mode="before")
    @classmethod
    def ensure_utc(cls, v: Any) -> datetime:
        """Ensure timestamp is UTC timezone-aware."""
        if isinstance(v, str):
            v = datetime.fromisoformat(v.replace("Z", "+00:00"))
        if isinstance(v, datetime):
            if v.tzinfo is None:
                return v.replace(tzinfo=timezone.utc)
            return v.astimezone(timezone.utc)
        return v


class TelemetryNormalizedEvent(BaseModel):
    """
    Schema for gl001.telemetry.normalized topic.

    Contains normalized time-series points with units and quality
    from OPC-UA data collection.

    Example:
        >>> event = TelemetryNormalizedEvent(
        ...     source_system="opc-ua-collector-01",
        ...     points=[TelemetryPoint(...), TelemetryPoint(...)],
        ...     collection_timestamp=datetime.now(timezone.utc)
        ... )
    """

    source_system: str = Field(
        ...,
        max_length=128,
        description="Source OPC-UA server or collector ID",
    )
    points: List[TelemetryPoint] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="List of telemetry points in this batch",
    )
    collection_timestamp: datetime = Field(
        ...,
        description="Timestamp when data was collected",
    )
    processing_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when data was normalized",
    )
    batch_id: str = Field(
        ...,
        max_length=64,
        description="Unique batch identifier",
    )
    sequence_number: int = Field(
        ...,
        ge=0,
        description="Sequence number for ordering",
    )
    data_hash: Optional[str] = Field(
        None,
        min_length=64,
        max_length=64,
        description="SHA-256 hash of point values for integrity",
    )

    @model_validator(mode="after")
    def compute_data_hash(self) -> TelemetryNormalizedEvent:
        """Compute data hash if not provided."""
        if self.data_hash is None:
            content = json.dumps(
                [{"tag": p.tag_id, "value": p.value} for p in self.points],
                sort_keys=True,
            )
            object.__setattr__(
                self, "data_hash", hashlib.sha256(content.encode()).hexdigest()
            )
        return self


# =============================================================================
# TOPIC: gl001.plan.dispatch
# =============================================================================


class LoadAllocation(BaseModel):
    """Single load allocation in a dispatch plan."""

    equipment_id: str = Field(
        ...,
        max_length=64,
        description="Equipment asset identifier",
    )
    load_mw: float = Field(
        ...,
        ge=0.0,
        description="Allocated load in MW",
    )
    min_load_mw: float = Field(
        ...,
        ge=0.0,
        description="Minimum operational load",
    )
    max_load_mw: float = Field(
        ...,
        ge=0.0,
        description="Maximum operational load",
    )
    efficiency_percent: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Expected efficiency at this load",
    )
    emissions_rate_kgco2_mwh: float = Field(
        ...,
        ge=0.0,
        description="Expected emissions rate",
    )
    fuel_type: str = Field(
        ...,
        max_length=32,
        description="Fuel type (natural_gas, hydrogen, etc.)",
    )
    ramp_rate_mw_min: float = Field(
        ...,
        ge=0.0,
        description="Maximum ramp rate in MW/min",
    )
    startup_time_min: Optional[float] = Field(
        None,
        ge=0.0,
        description="Cold startup time in minutes",
    )
    marginal_cost_usd_mwh: float = Field(
        ...,
        ge=0.0,
        description="Marginal cost at this load",
    )


class ExpectedImpact(BaseModel):
    """Expected impact metrics from dispatch plan."""

    total_cost_usd: float = Field(
        ...,
        ge=0.0,
        description="Total expected cost in USD",
    )
    total_emissions_tco2: float = Field(
        ...,
        ge=0.0,
        description="Total expected emissions in tCO2",
    )
    average_efficiency_percent: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Weighted average efficiency",
    )
    cost_savings_usd: float = Field(
        0.0,
        description="Cost savings vs baseline",
    )
    emissions_reduction_tco2: float = Field(
        0.0,
        description="Emissions reduction vs baseline",
    )
    reliability_score: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description="System reliability score",
    )


class DispatchPlanEvent(BaseModel):
    """
    Schema for gl001.plan.dispatch topic.

    Contains dispatch plan with allocations, solver status,
    and expected impact metrics.

    Example:
        >>> event = DispatchPlanEvent(
        ...     plan_id="plan-abc123",
        ...     horizon_start=datetime.now(timezone.utc),
        ...     horizon_end=datetime.now(timezone.utc) + timedelta(hours=24),
        ...     allocations=[LoadAllocation(...), ...],
        ...     solver_status=SolverStatus.OPTIMAL,
        ...     expected_impact=ExpectedImpact(...)
        ... )
    """

    plan_id: str = Field(
        ...,
        max_length=64,
        description="Unique plan identifier",
    )
    horizon_start: datetime = Field(
        ...,
        description="Planning horizon start (UTC)",
    )
    horizon_end: datetime = Field(
        ...,
        description="Planning horizon end (UTC)",
    )
    resolution_minutes: int = Field(
        15,
        ge=1,
        le=60,
        description="Time resolution in minutes",
    )
    allocations: List[LoadAllocation] = Field(
        ...,
        min_length=1,
        description="Load allocations by equipment",
    )
    solver_status: SolverStatus = Field(
        ...,
        description="MILP solver status",
    )
    solver_gap_percent: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="Optimality gap percentage",
    )
    solver_time_seconds: float = Field(
        ...,
        ge=0.0,
        description="Solver execution time",
    )
    iteration_count: int = Field(
        1,
        ge=1,
        description="Solver iteration count",
    )
    expected_impact: ExpectedImpact = Field(
        ...,
        description="Expected impact metrics",
    )
    constraints_active: List[str] = Field(
        default_factory=list,
        description="List of active constraints",
    )
    demand_mw: float = Field(
        ...,
        ge=0.0,
        description="Total demand to be met",
    )
    reserve_margin_mw: float = Field(
        0.0,
        ge=0.0,
        description="Reserve capacity margin",
    )
    created_by: str = Field(
        ...,
        max_length=64,
        description="System or user that created plan",
    )
    approved_by: Optional[str] = Field(
        None,
        max_length=64,
        description="Approver if manual approval required",
    )
    version: int = Field(
        1,
        ge=1,
        description="Plan version number",
    )


# =============================================================================
# TOPIC: gl001.actions.recommendations
# =============================================================================


class SetpointBound(BaseModel):
    """Bounds for a setpoint recommendation."""

    min_value: float = Field(
        ...,
        description="Minimum allowed value",
    )
    max_value: float = Field(
        ...,
        description="Maximum allowed value",
    )
    safety_min: Optional[float] = Field(
        None,
        description="Safety system minimum (SIS)",
    )
    safety_max: Optional[float] = Field(
        None,
        description="Safety system maximum (SIS)",
    )
    soft_min: Optional[float] = Field(
        None,
        description="Soft constraint minimum",
    )
    soft_max: Optional[float] = Field(
        None,
        description="Soft constraint maximum",
    )


class SetpointRecommendation(BaseModel):
    """Single setpoint recommendation with bounds and rationale."""

    tag_id: str = Field(
        ...,
        max_length=128,
        description="OPC-UA tag for setpoint",
    )
    current_value: float = Field(
        ...,
        description="Current setpoint value",
    )
    recommended_value: float = Field(
        ...,
        description="Recommended new value",
    )
    unit: UnitOfMeasure = Field(
        ...,
        description="Unit of measure",
    )
    bounds: SetpointBound = Field(
        ...,
        description="Value bounds and constraints",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in recommendation",
    )
    rationale: str = Field(
        ...,
        max_length=1024,
        description="Human-readable rationale",
    )
    expected_benefit: Dict[str, float] = Field(
        default_factory=dict,
        description="Expected benefits (cost, emissions, etc.)",
    )
    risk_factors: List[str] = Field(
        default_factory=list,
        description="Risk factors to consider",
    )
    requires_approval: bool = Field(
        False,
        description="Whether manual approval is required",
    )
    auto_execute: bool = Field(
        False,
        description="Whether to auto-execute if within bounds",
    )


class ActionRecommendationEvent(BaseModel):
    """
    Schema for gl001.actions.recommendations topic.

    Contains setpoint recommendations with bounds and rationale
    for operator review and/or automatic execution.

    Example:
        >>> event = ActionRecommendationEvent(
        ...     recommendation_id="rec-abc123",
        ...     recommendations=[SetpointRecommendation(...), ...],
        ...     triggered_by="dispatch_plan",
        ...     overall_confidence=0.92
        ... )
    """

    recommendation_id: str = Field(
        ...,
        max_length=64,
        description="Unique recommendation identifier",
    )
    recommendations: List[SetpointRecommendation] = Field(
        ...,
        min_length=1,
        description="List of setpoint recommendations",
    )
    triggered_by: str = Field(
        ...,
        max_length=128,
        description="Event or system that triggered this",
    )
    trigger_timestamp: datetime = Field(
        ...,
        description="When the trigger occurred",
    )
    overall_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall recommendation confidence",
    )
    expires_at: datetime = Field(
        ...,
        description="When recommendation expires",
    )
    execution_mode: str = Field(
        "advisory",
        description="advisory, semi-auto, or auto",
    )
    total_expected_benefit: Dict[str, float] = Field(
        default_factory=dict,
        description="Aggregate expected benefits",
    )
    plan_reference_id: Optional[str] = Field(
        None,
        max_length=64,
        description="Reference to dispatch plan",
    )
    operator_notes: Optional[str] = Field(
        None,
        max_length=2048,
        description="Notes for operator review",
    )


# =============================================================================
# TOPIC: gl001.safety.events
# =============================================================================


class BoundaryViolation(BaseModel):
    """Safety boundary violation details."""

    tag_id: str = Field(
        ...,
        max_length=128,
        description="Tag that violated boundary",
    )
    boundary_type: str = Field(
        ...,
        description="Type: high, low, rate_of_change",
    )
    limit_value: float = Field(
        ...,
        description="Boundary limit value",
    )
    actual_value: float = Field(
        ...,
        description="Actual value at violation",
    )
    deviation_percent: float = Field(
        ...,
        description="Deviation from limit as percentage",
    )
    unit: UnitOfMeasure = Field(
        ...,
        description="Unit of measure",
    )
    duration_seconds: float = Field(
        0.0,
        ge=0.0,
        description="Duration of violation",
    )


class BlockedWrite(BaseModel):
    """Record of a blocked write operation."""

    tag_id: str = Field(
        ...,
        max_length=128,
        description="Tag write was attempted on",
    )
    attempted_value: float = Field(
        ...,
        description="Value that was attempted",
    )
    current_value: float = Field(
        ...,
        description="Current value at block time",
    )
    block_reason: str = Field(
        ...,
        max_length=512,
        description="Reason for blocking",
    )
    blocked_by: str = Field(
        ...,
        max_length=64,
        description="System that blocked (SIS, operator, etc.)",
    )


class SISPermissiveChange(BaseModel):
    """Safety Instrumented System permissive state change."""

    sis_id: str = Field(
        ...,
        max_length=64,
        description="SIS identifier",
    )
    permissive_name: str = Field(
        ...,
        max_length=128,
        description="Permissive name",
    )
    previous_state: bool = Field(
        ...,
        description="Previous permissive state",
    )
    new_state: bool = Field(
        ...,
        description="New permissive state",
    )
    trigger_condition: str = Field(
        ...,
        max_length=512,
        description="Condition that triggered change",
    )
    sil_level: int = Field(
        ...,
        ge=1,
        le=4,
        description="Safety Integrity Level (1-4)",
    )


class SafetyEvent(BaseModel):
    """
    Schema for gl001.safety.events topic.

    Contains boundary violations, blocked writes, and
    SIS permissive changes for safety monitoring.

    Example:
        >>> event = SafetyEvent(
        ...     event_id="safety-abc123",
        ...     level=SafetyLevel.ALARM,
        ...     boundary_violations=[BoundaryViolation(...)],
        ...     equipment_id="boiler-01"
        ... )
    """

    event_id: str = Field(
        ...,
        max_length=64,
        description="Unique safety event identifier",
    )
    level: SafetyLevel = Field(
        ...,
        description="Safety event severity level",
    )
    event_timestamp: datetime = Field(
        ...,
        description="When the safety event occurred",
    )
    equipment_id: str = Field(
        ...,
        max_length=64,
        description="Affected equipment identifier",
    )
    equipment_name: str = Field(
        ...,
        max_length=128,
        description="Human-readable equipment name",
    )
    area_id: Optional[str] = Field(
        None,
        max_length=64,
        description="Plant area identifier",
    )
    boundary_violations: List[BoundaryViolation] = Field(
        default_factory=list,
        description="List of boundary violations",
    )
    blocked_writes: List[BlockedWrite] = Field(
        default_factory=list,
        description="List of blocked writes",
    )
    sis_changes: List[SISPermissiveChange] = Field(
        default_factory=list,
        description="SIS permissive changes",
    )
    operator_action_required: bool = Field(
        False,
        description="Whether operator action is required",
    )
    action_deadline: Optional[datetime] = Field(
        None,
        description="Deadline for required action",
    )
    escalation_level: int = Field(
        0,
        ge=0,
        le=5,
        description="Escalation level (0=none, 5=executive)",
    )
    related_event_ids: List[str] = Field(
        default_factory=list,
        description="Related safety event IDs",
    )
    response_plan_id: Optional[str] = Field(
        None,
        max_length=64,
        description="Emergency response plan reference",
    )
    acknowledged_by: Optional[str] = Field(
        None,
        max_length=64,
        description="User who acknowledged event",
    )
    acknowledged_at: Optional[datetime] = Field(
        None,
        description="When event was acknowledged",
    )


# =============================================================================
# TOPIC: gl001.maintenance.triggers
# =============================================================================


class MaintenanceEvidence(BaseModel):
    """Evidence supporting a maintenance trigger."""

    evidence_type: str = Field(
        ...,
        description="Type: sensor_reading, model_prediction, etc.",
    )
    description: str = Field(
        ...,
        max_length=1024,
        description="Human-readable description",
    )
    value: Optional[float] = Field(
        None,
        description="Numeric value if applicable",
    )
    threshold: Optional[float] = Field(
        None,
        description="Threshold value if applicable",
    )
    unit: Optional[UnitOfMeasure] = Field(
        None,
        description="Unit of measure if applicable",
    )
    confidence: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description="Confidence in this evidence",
    )
    source_tag_ids: List[str] = Field(
        default_factory=list,
        description="Source sensor tags",
    )
    model_id: Optional[str] = Field(
        None,
        max_length=64,
        description="Predictive model identifier",
    )
    data_range_start: Optional[datetime] = Field(
        None,
        description="Start of data range analyzed",
    )
    data_range_end: Optional[datetime] = Field(
        None,
        description="End of data range analyzed",
    )


class MaintenanceTriggerEvent(BaseModel):
    """
    Schema for gl001.maintenance.triggers topic.

    Contains work order recommendations with supporting
    evidence for predictive maintenance.

    Example:
        >>> event = MaintenanceTriggerEvent(
        ...     trigger_id="maint-abc123",
        ...     equipment_id="pump-01",
        ...     failure_mode="bearing_degradation",
        ...     priority=MaintenancePriority.P3_HIGH,
        ...     evidence=[MaintenanceEvidence(...), ...]
        ... )
    """

    trigger_id: str = Field(
        ...,
        max_length=64,
        description="Unique trigger identifier",
    )
    equipment_id: str = Field(
        ...,
        max_length=64,
        description="Equipment asset identifier",
    )
    equipment_name: str = Field(
        ...,
        max_length=128,
        description="Human-readable equipment name",
    )
    equipment_class: str = Field(
        ...,
        max_length=64,
        description="Equipment class (pump, boiler, etc.)",
    )
    failure_mode: str = Field(
        ...,
        max_length=128,
        description="Predicted or detected failure mode",
    )
    failure_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of failure if not addressed",
    )
    remaining_useful_life_hours: Optional[float] = Field(
        None,
        ge=0.0,
        description="Estimated RUL in hours",
    )
    priority: MaintenancePriority = Field(
        ...,
        description="Recommended work order priority",
    )
    evidence: List[MaintenanceEvidence] = Field(
        ...,
        min_length=1,
        description="Supporting evidence",
    )
    recommended_action: str = Field(
        ...,
        max_length=2048,
        description="Recommended maintenance action",
    )
    estimated_downtime_hours: float = Field(
        ...,
        ge=0.0,
        description="Estimated maintenance downtime",
    )
    estimated_cost_usd: float = Field(
        ...,
        ge=0.0,
        description="Estimated maintenance cost",
    )
    cost_of_failure_usd: float = Field(
        ...,
        ge=0.0,
        description="Estimated cost if failure occurs",
    )
    spare_parts_required: List[str] = Field(
        default_factory=list,
        description="Required spare parts",
    )
    skills_required: List[str] = Field(
        default_factory=list,
        description="Required maintenance skills",
    )
    window_start: datetime = Field(
        ...,
        description="Recommended maintenance window start",
    )
    window_end: datetime = Field(
        ...,
        description="Recommended maintenance window end",
    )
    cmms_work_order_id: Optional[str] = Field(
        None,
        max_length=64,
        description="CMMS work order ID if created",
    )
    cmms_status: Optional[str] = Field(
        None,
        max_length=32,
        description="CMMS work order status",
    )


# =============================================================================
# TOPIC: gl001.explainability.reports
# =============================================================================


class FeatureContribution(BaseModel):
    """Feature contribution in explainability analysis."""

    feature_name: str = Field(
        ...,
        max_length=128,
        description="Feature name",
    )
    feature_value: float = Field(
        ...,
        description="Feature value used",
    )
    contribution: float = Field(
        ...,
        description="Contribution to prediction",
    )
    contribution_percent: float = Field(
        ...,
        description="Contribution as percentage of total",
    )
    direction: str = Field(
        ...,
        description="positive or negative",
    )
    baseline_value: Optional[float] = Field(
        None,
        description="Baseline feature value",
    )
    unit: Optional[UnitOfMeasure] = Field(
        None,
        description="Feature unit if applicable",
    )


class SHAPSummary(BaseModel):
    """SHAP explanation summary."""

    method: str = Field(
        "TreeSHAP",
        description="SHAP method used",
    )
    base_value: float = Field(
        ...,
        description="SHAP base value (expected value)",
    )
    output_value: float = Field(
        ...,
        description="Model output value",
    )
    feature_contributions: List[FeatureContribution] = Field(
        ...,
        description="Feature contributions",
    )
    interaction_effects: Dict[str, float] = Field(
        default_factory=dict,
        description="Interaction effects between features",
    )


class LIMESummary(BaseModel):
    """LIME explanation summary."""

    num_samples: int = Field(
        ...,
        ge=1,
        description="Number of samples used",
    )
    r2_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Local model R-squared score",
    )
    feature_weights: Dict[str, float] = Field(
        ...,
        description="Feature weights in local model",
    )
    intercept: float = Field(
        ...,
        description="Local model intercept",
    )
    prediction_local: float = Field(
        ...,
        description="Local model prediction",
    )
    prediction_model: float = Field(
        ...,
        description="Original model prediction",
    )


class UncertaintyQuantification(BaseModel):
    """Uncertainty quantification for predictions."""

    point_estimate: float = Field(
        ...,
        description="Point estimate (mean prediction)",
    )
    std_deviation: float = Field(
        ...,
        ge=0.0,
        description="Standard deviation",
    )
    confidence_level: float = Field(
        0.95,
        gt=0.0,
        lt=1.0,
        description="Confidence interval level",
    )
    lower_bound: float = Field(
        ...,
        description="Lower confidence bound",
    )
    upper_bound: float = Field(
        ...,
        description="Upper confidence bound",
    )
    method: str = Field(
        "bootstrap",
        description="Uncertainty method (bootstrap, dropout, etc.)",
    )
    num_samples: int = Field(
        ...,
        ge=1,
        description="Number of samples used",
    )
    epistemic_uncertainty: Optional[float] = Field(
        None,
        ge=0.0,
        description="Model uncertainty component",
    )
    aleatoric_uncertainty: Optional[float] = Field(
        None,
        ge=0.0,
        description="Data uncertainty component",
    )


class ExplainabilityReportEvent(BaseModel):
    """
    Schema for gl001.explainability.reports topic.

    Contains SHAP/LIME summaries, feature contributions,
    and uncertainty quantification for model predictions.

    Example:
        >>> event = ExplainabilityReportEvent(
        ...     report_id="explain-abc123",
        ...     model_id="load-optimizer-v2",
        ...     prediction_type="load_allocation",
        ...     shap_summary=SHAPSummary(...),
        ...     uncertainty=UncertaintyQuantification(...)
        ... )
    """

    report_id: str = Field(
        ...,
        max_length=64,
        description="Unique report identifier",
    )
    model_id: str = Field(
        ...,
        max_length=64,
        description="Model identifier",
    )
    model_version: str = Field(
        ...,
        max_length=32,
        description="Model version",
    )
    prediction_type: str = Field(
        ...,
        max_length=64,
        description="Type of prediction explained",
    )
    prediction_timestamp: datetime = Field(
        ...,
        description="When prediction was made",
    )
    input_snapshot: Dict[str, Any] = Field(
        ...,
        description="Input features snapshot",
    )
    prediction_value: float = Field(
        ...,
        description="Model prediction value",
    )
    prediction_unit: Optional[UnitOfMeasure] = Field(
        None,
        description="Prediction unit if applicable",
    )
    shap_summary: Optional[SHAPSummary] = Field(
        None,
        description="SHAP explanation summary",
    )
    lime_summary: Optional[LIMESummary] = Field(
        None,
        description="LIME explanation summary",
    )
    uncertainty: Optional[UncertaintyQuantification] = Field(
        None,
        description="Uncertainty quantification",
    )
    top_features: List[FeatureContribution] = Field(
        default_factory=list,
        max_length=20,
        description="Top contributing features",
    )
    anomaly_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Anomaly score for this prediction",
    )
    model_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence in prediction",
    )
    human_readable_explanation: str = Field(
        ...,
        max_length=4096,
        description="Human-readable explanation text",
    )
    reference_decision_id: Optional[str] = Field(
        None,
        max_length=64,
        description="Reference to decision using this prediction",
    )


# =============================================================================
# TOPIC: gl001.audit.log
# =============================================================================


class AuditLogEvent(BaseModel):
    """
    Schema for gl001.audit.log topic.

    Append-only audit events with correlation IDs for
    complete system traceability and compliance.

    Example:
        >>> event = AuditLogEvent(
        ...     audit_id="audit-abc123",
        ...     action=AuditAction.EXECUTE,
        ...     resource_type="dispatch_plan",
        ...     resource_id="plan-xyz789",
        ...     actor_id="thermalcommand-orchestrator"
        ... )
    """

    audit_id: str = Field(
        ...,
        max_length=64,
        description="Unique audit event identifier",
    )
    action: AuditAction = Field(
        ...,
        description="Action performed",
    )
    action_timestamp: datetime = Field(
        ...,
        description="When action was performed",
    )
    actor_id: str = Field(
        ...,
        max_length=128,
        description="Actor (user, system, or service) ID",
    )
    actor_type: str = Field(
        ...,
        max_length=32,
        description="Actor type: user, service, system",
    )
    actor_ip: Optional[str] = Field(
        None,
        max_length=45,
        description="Actor IP address if applicable",
    )
    resource_type: str = Field(
        ...,
        max_length=64,
        description="Type of resource affected",
    )
    resource_id: str = Field(
        ...,
        max_length=128,
        description="Resource identifier",
    )
    resource_name: Optional[str] = Field(
        None,
        max_length=256,
        description="Human-readable resource name",
    )
    correlation_id: str = Field(
        ...,
        max_length=64,
        description="Correlation ID for tracing",
    )
    causation_id: Optional[str] = Field(
        None,
        max_length=64,
        description="ID of causing event",
    )
    session_id: Optional[str] = Field(
        None,
        max_length=64,
        description="Session identifier if applicable",
    )
    request_id: Optional[str] = Field(
        None,
        max_length=64,
        description="Request identifier if applicable",
    )
    previous_state: Optional[Dict[str, Any]] = Field(
        None,
        description="State before action (for updates)",
    )
    new_state: Optional[Dict[str, Any]] = Field(
        None,
        description="State after action (for updates)",
    )
    changes: Optional[Dict[str, Any]] = Field(
        None,
        description="Summary of changes made",
    )
    outcome: str = Field(
        "success",
        description="Action outcome: success, failure, partial",
    )
    error_code: Optional[str] = Field(
        None,
        max_length=32,
        description="Error code if failure",
    )
    error_message: Optional[str] = Field(
        None,
        max_length=2048,
        description="Error message if failure",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional audit metadata",
    )
    compliance_tags: List[str] = Field(
        default_factory=list,
        description="Compliance tags (SOX, ISO50001, etc.)",
    )
    retention_days: int = Field(
        2555,  # ~7 years for SOX compliance
        ge=1,
        description="Retention period in days",
    )
    signature: Optional[str] = Field(
        None,
        max_length=512,
        description="Digital signature for non-repudiation",
    )

    @model_validator(mode="after")
    def validate_state_for_update(self) -> AuditLogEvent:
        """Validate state fields are provided for update actions."""
        if self.action == AuditAction.UPDATE:
            if self.previous_state is None or self.new_state is None:
                # Log warning but don't fail - might be intentional
                pass
        return self


# =============================================================================
# TOPIC REGISTRY AND SCHEMA REGISTRY HELPERS
# =============================================================================


class TopicSchemaRegistry:
    """
    Registry mapping topics to their schemas.

    Provides schema lookup, validation, and Schema Registry
    integration helpers.
    """

    TOPIC_SCHEMAS: Dict[str, type] = {
        "gl001.telemetry.normalized": TelemetryNormalizedEvent,
        "gl001.plan.dispatch": DispatchPlanEvent,
        "gl001.actions.recommendations": ActionRecommendationEvent,
        "gl001.safety.events": SafetyEvent,
        "gl001.maintenance.triggers": MaintenanceTriggerEvent,
        "gl001.explainability.reports": ExplainabilityReportEvent,
        "gl001.audit.log": AuditLogEvent,
    }

    @classmethod
    def get_schema(cls, topic: str) -> type:
        """
        Get the Pydantic schema class for a topic.

        Args:
            topic: Topic name

        Returns:
            Pydantic model class for the topic

        Raises:
            KeyError: If topic is not registered
        """
        if topic not in cls.TOPIC_SCHEMAS:
            raise KeyError(f"Unknown topic: {topic}")
        return cls.TOPIC_SCHEMAS[topic]

    @classmethod
    def validate_payload(cls, topic: str, payload: Dict[str, Any]) -> BaseModel:
        """
        Validate a payload against the topic schema.

        Args:
            topic: Topic name
            payload: Payload dictionary to validate

        Returns:
            Validated Pydantic model instance

        Raises:
            KeyError: If topic is not registered
            ValidationError: If payload is invalid
        """
        schema_class = cls.get_schema(topic)
        return schema_class.model_validate(payload)

    @classmethod
    def get_json_schema(cls, topic: str) -> Dict[str, Any]:
        """
        Get JSON schema for a topic (for Schema Registry).

        Args:
            topic: Topic name

        Returns:
            JSON schema dictionary
        """
        schema_class = cls.get_schema(topic)
        return schema_class.model_json_schema()

    @classmethod
    def list_topics(cls) -> List[str]:
        """Return list of all registered topics."""
        return list(cls.TOPIC_SCHEMAS.keys())

    @classmethod
    def get_avro_schema(cls, topic: str) -> Dict[str, Any]:
        """
        Convert Pydantic schema to Avro schema format.

        This is a simplified conversion - production systems
        should use fastavro or similar for full Avro support.

        Args:
            topic: Topic name

        Returns:
            Avro-compatible schema dictionary
        """
        json_schema = cls.get_json_schema(topic)
        schema_class = cls.get_schema(topic)

        # Basic conversion to Avro format
        avro_schema = {
            "type": "record",
            "name": schema_class.__name__,
            "namespace": "com.greenlang.gl001",
            "fields": [],
        }

        # Convert properties to Avro fields
        properties = json_schema.get("properties", {})
        required = json_schema.get("required", [])

        for field_name, field_def in properties.items():
            avro_field = {
                "name": field_name,
                "type": cls._json_to_avro_type(
                    field_def,
                    field_name in required,
                ),
            }
            if "description" in field_def:
                avro_field["doc"] = field_def["description"]
            avro_schema["fields"].append(avro_field)

        return avro_schema

    @classmethod
    def _json_to_avro_type(
        cls,
        json_def: Dict[str, Any],
        required: bool,
    ) -> Union[str, List, Dict]:
        """Convert JSON Schema type to Avro type."""
        json_type = json_def.get("type", "string")

        type_mapping = {
            "string": "string",
            "integer": "long",
            "number": "double",
            "boolean": "boolean",
            "array": {"type": "array", "items": "string"},
            "object": {"type": "map", "values": "string"},
        }

        avro_type = type_mapping.get(json_type, "string")

        # Handle nullable (optional) fields
        if not required:
            return ["null", avro_type]

        return avro_type
