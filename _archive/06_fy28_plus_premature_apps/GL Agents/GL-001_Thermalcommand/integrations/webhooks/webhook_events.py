"""
GL-001 ThermalCommand - Webhook Event Definitions

This module defines all webhook event types for the ThermalCommand system.
Events are strongly-typed using Pydantic models with comprehensive validation.

Event Types:
- HeatPlanCreated: Emitted when a new heat optimization plan is generated
- SetpointRecommendation: Emitted when a setpoint change is recommended
- SafetyActionBlocked: Emitted when safety system blocks a recommended action
- MaintenanceTrigger: Emitted when maintenance is recommended for an asset

All events include:
- Unique event ID for idempotency
- Timestamp with timezone
- Correlation ID for request tracing
- SHA-256 provenance hash for audit trail

Example:
    >>> event = HeatPlanCreatedEvent(
    ...     plan_id="plan-001",
    ...     horizon_hours=24,
    ...     expected_cost_usd=15000.0,
    ...     expected_emissions_kg_co2e=1200.0
    ... )
    >>> payload = event.to_webhook_payload()

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import hashlib
import json
import uuid

from pydantic import BaseModel, Field, validator


class WebhookEventType(str, Enum):
    """
    Enumeration of all supported webhook event types.

    Each event type corresponds to a specific domain event in the
    ThermalCommand system that external systems may need to be notified about.
    """

    HEAT_PLAN_CREATED = "heat_plan.created"
    HEAT_PLAN_UPDATED = "heat_plan.updated"
    HEAT_PLAN_CANCELLED = "heat_plan.cancelled"
    SETPOINT_RECOMMENDATION = "setpoint.recommendation"
    SETPOINT_APPLIED = "setpoint.applied"
    SETPOINT_REJECTED = "setpoint.rejected"
    SAFETY_ACTION_BLOCKED = "safety.action_blocked"
    SAFETY_ALARM_RAISED = "safety.alarm_raised"
    SAFETY_ALARM_CLEARED = "safety.alarm_cleared"
    MAINTENANCE_TRIGGER = "maintenance.trigger"
    MAINTENANCE_SCHEDULED = "maintenance.scheduled"
    MAINTENANCE_COMPLETED = "maintenance.completed"
    SYSTEM_STATUS_CHANGED = "system.status_changed"
    AGENT_HEALTH_CHANGED = "agent.health_changed"


class ApplyPolicy(str, Enum):
    """Policy for applying setpoint recommendations."""

    MANUAL = "manual"  # Requires operator approval
    AUTO_SAFE = "auto_safe"  # Auto-apply if within safe bounds
    AUTO_ALL = "auto_all"  # Auto-apply all (requires elevated privileges)
    SUPERVISED = "supervised"  # Auto-apply with supervisor notification


class TriggerLevel(str, Enum):
    """Maintenance trigger severity levels."""

    ADVISORY = "advisory"  # Informational, plan maintenance
    WARNING = "warning"  # Degraded performance, schedule soon
    CRITICAL = "critical"  # Imminent failure, immediate action
    EMERGENCY = "emergency"  # Safety risk, immediate shutdown


class WebhookEvent(BaseModel):
    """
    Base class for all webhook events.

    Provides common fields and methods for all ThermalCommand webhook events.
    All events are immutable once created and include provenance tracking.

    Attributes:
        event_id: Unique identifier for idempotent delivery
        event_type: Type of the event (from WebhookEventType enum)
        event_version: Schema version for forward compatibility
        timestamp: ISO 8601 timestamp when event was created
        source: Source system/agent that generated the event
        correlation_id: Request correlation ID for distributed tracing
        provenance_hash: SHA-256 hash of event payload for audit trail
        metadata: Optional additional metadata
    """

    event_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique event identifier for idempotency"
    )
    event_type: WebhookEventType = Field(
        ...,
        description="Type of webhook event"
    )
    event_version: str = Field(
        default="1.0",
        description="Event schema version"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Event creation timestamp (UTC)"
    )
    source: str = Field(
        default="GL-001-ThermalCommand",
        description="Source system identifier"
    )
    correlation_id: Optional[str] = Field(
        default=None,
        description="Correlation ID for request tracing"
    )
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 hash for audit trail"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional event metadata"
    )

    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    def calculate_provenance_hash(self) -> str:
        """
        Calculate SHA-256 provenance hash for audit trail.

        The hash is calculated over the canonical JSON representation
        of the event payload (excluding the provenance_hash field itself).

        Returns:
            SHA-256 hex digest of the event payload
        """
        # Create a copy without the provenance_hash field
        payload_dict = self.dict(exclude={"provenance_hash"})

        # Canonical JSON serialization (sorted keys, no whitespace)
        canonical_json = json.dumps(
            payload_dict,
            sort_keys=True,
            separators=(",", ":"),
            default=str
        )

        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    def with_provenance(self) -> "WebhookEvent":
        """
        Return a new event instance with calculated provenance hash.

        Returns:
            New WebhookEvent instance with provenance_hash set
        """
        provenance = self.calculate_provenance_hash()
        return self.copy(update={"provenance_hash": provenance})

    def to_webhook_payload(self) -> Dict[str, Any]:
        """
        Convert event to webhook payload dictionary.

        Returns:
            Dictionary suitable for JSON serialization and webhook delivery
        """
        # Ensure provenance hash is calculated
        if self.provenance_hash is None:
            event = self.with_provenance()
        else:
            event = self

        return event.dict()

    def to_json(self) -> str:
        """
        Convert event to JSON string.

        Returns:
            JSON string representation of the event
        """
        return json.dumps(self.to_webhook_payload(), default=str)


class HeatPlanCreatedEvent(WebhookEvent):
    """
    Event emitted when a new heat optimization plan is created.

    This event is triggered when the MILP optimizer generates a new
    heat allocation plan for the facility. External systems can use
    this to prepare for upcoming operational changes.

    Attributes:
        plan_id: Unique identifier for the heat plan
        horizon_hours: Planning horizon in hours
        created_at: Plan creation timestamp
        expected_cost_usd: Expected total cost in USD
        expected_emissions_kg_co2e: Expected emissions in kg CO2e
        num_time_slots: Number of time slots in the plan
        equipment_ids: List of equipment IDs included in the plan
        optimization_objective: Objective function used (cost, emissions, balanced)
        confidence_score: Model confidence in the plan (0-1)

    Example:
        >>> event = HeatPlanCreatedEvent(
        ...     plan_id="plan-2024-001",
        ...     horizon_hours=24,
        ...     expected_cost_usd=15000.0,
        ...     expected_emissions_kg_co2e=1200.0
        ... )
    """

    event_type: WebhookEventType = Field(
        default=WebhookEventType.HEAT_PLAN_CREATED,
        const=True
    )
    plan_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Unique heat plan identifier"
    )
    horizon_hours: int = Field(
        ...,
        ge=1,
        le=168,  # Max 1 week
        description="Planning horizon in hours"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Plan creation timestamp"
    )
    expected_cost_usd: float = Field(
        ...,
        ge=0.0,
        description="Expected total cost in USD"
    )
    expected_emissions_kg_co2e: float = Field(
        ...,
        ge=0.0,
        description="Expected total emissions in kg CO2e"
    )
    num_time_slots: int = Field(
        default=24,
        ge=1,
        le=672,  # Max hourly slots for 4 weeks
        description="Number of time slots in plan"
    )
    equipment_ids: List[str] = Field(
        default_factory=list,
        description="Equipment IDs included in plan"
    )
    optimization_objective: str = Field(
        default="balanced",
        description="Optimization objective (cost, emissions, balanced)"
    )
    confidence_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Model confidence score"
    )
    constraints_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Summary of active constraints"
    )

    @validator("optimization_objective")
    def validate_objective(cls, v: str) -> str:
        """Validate optimization objective is valid."""
        valid = {"cost", "emissions", "balanced", "reliability"}
        if v.lower() not in valid:
            raise ValueError(f"Optimization objective must be one of: {valid}")
        return v.lower()


class SetpointRecommendationEvent(WebhookEvent):
    """
    Event emitted when a setpoint change is recommended.

    This event is triggered when the control system recommends a change
    to an equipment setpoint. The recommendation includes bounds validation,
    rationale, and the apply policy.

    Attributes:
        rec_id: Unique recommendation identifier
        tag: Equipment/sensor tag (e.g., "FIC-101.SP")
        value: Recommended setpoint value
        unit: Engineering unit for the value
        bounds: Min/max bounds for the setpoint
        rationale: Explanation for the recommendation
        apply_policy: How the recommendation should be applied
        current_value: Current setpoint value (if known)
        expected_benefit: Expected benefit description
        urgency: Urgency level (low, medium, high, critical)

    Example:
        >>> event = SetpointRecommendationEvent(
        ...     rec_id="rec-001",
        ...     tag="TIC-201.SP",
        ...     value=1450.0,
        ...     unit="degF",
        ...     bounds={"min": 1200.0, "max": 1600.0},
        ...     rationale="Reduce temperature to lower emissions by 5%"
        ... )
    """

    event_type: WebhookEventType = Field(
        default=WebhookEventType.SETPOINT_RECOMMENDATION,
        const=True
    )
    rec_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Unique recommendation identifier"
    )
    tag: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Equipment/sensor tag identifier"
    )
    value: float = Field(
        ...,
        description="Recommended setpoint value"
    )
    unit: str = Field(
        default="",
        max_length=32,
        description="Engineering unit for the value"
    )
    bounds: Dict[str, float] = Field(
        ...,
        description="Min/max bounds for the setpoint"
    )
    rationale: str = Field(
        ...,
        min_length=1,
        max_length=1024,
        description="Explanation for the recommendation"
    )
    apply_policy: ApplyPolicy = Field(
        default=ApplyPolicy.MANUAL,
        description="Policy for applying the recommendation"
    )
    current_value: Optional[float] = Field(
        default=None,
        description="Current setpoint value"
    )
    expected_benefit: Optional[str] = Field(
        default=None,
        max_length=512,
        description="Expected benefit description"
    )
    urgency: str = Field(
        default="medium",
        description="Urgency level"
    )
    confidence_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence in recommendation"
    )
    related_plan_id: Optional[str] = Field(
        default=None,
        description="Related heat plan ID if applicable"
    )

    @validator("bounds")
    def validate_bounds(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Validate bounds dictionary has required keys."""
        if "min" not in v or "max" not in v:
            raise ValueError("Bounds must contain 'min' and 'max' keys")
        if v["min"] >= v["max"]:
            raise ValueError("Bounds 'min' must be less than 'max'")
        return v

    @validator("urgency")
    def validate_urgency(cls, v: str) -> str:
        """Validate urgency level."""
        valid = {"low", "medium", "high", "critical"}
        if v.lower() not in valid:
            raise ValueError(f"Urgency must be one of: {valid}")
        return v.lower()


class SafetyActionBlockedEvent(WebhookEvent):
    """
    Event emitted when the safety system blocks a recommended action.

    This is a critical event that indicates a recommendation was rejected
    because it would violate safety boundaries. All blocked actions are
    logged for audit and investigation purposes.

    Attributes:
        rec_id: ID of the blocked recommendation
        reason: Human-readable reason for blocking
        boundary_id: ID of the safety boundary that was violated
        boundary_type: Type of boundary (temperature, pressure, emissions, etc.)
        current_state_snapshot_ref: Reference to state snapshot at block time
        violated_limit: The limit that would have been violated
        recommended_value: The value that was recommended
        severity: Severity of the potential violation
        required_action: Any required follow-up action

    Example:
        >>> event = SafetyActionBlockedEvent(
        ...     rec_id="rec-001",
        ...     reason="Recommended temperature exceeds SIL-3 high limit",
        ...     boundary_id="TAHH-201",
        ...     current_state_snapshot_ref="snapshot-2024-001"
        ... )
    """

    event_type: WebhookEventType = Field(
        default=WebhookEventType.SAFETY_ACTION_BLOCKED,
        const=True
    )
    rec_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="ID of the blocked recommendation"
    )
    reason: str = Field(
        ...,
        min_length=1,
        max_length=1024,
        description="Human-readable reason for blocking"
    )
    boundary_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="ID of the violated safety boundary"
    )
    boundary_type: str = Field(
        default="unknown",
        description="Type of safety boundary"
    )
    current_state_snapshot_ref: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Reference to state snapshot at block time"
    )
    violated_limit: Optional[float] = Field(
        default=None,
        description="The safety limit that would be violated"
    )
    recommended_value: Optional[float] = Field(
        default=None,
        description="The recommended value that was blocked"
    )
    severity: str = Field(
        default="high",
        description="Severity of the potential violation"
    )
    safety_integrity_level: str = Field(
        default="SIL_3",
        description="Safety Integrity Level of the boundary"
    )
    required_action: Optional[str] = Field(
        default=None,
        max_length=512,
        description="Required follow-up action"
    )
    equipment_id: Optional[str] = Field(
        default=None,
        description="Affected equipment ID"
    )
    operator_notification_required: bool = Field(
        default=True,
        description="Whether operator must be notified"
    )

    @validator("severity")
    def validate_severity(cls, v: str) -> str:
        """Validate severity level."""
        valid = {"low", "medium", "high", "critical"}
        if v.lower() not in valid:
            raise ValueError(f"Severity must be one of: {valid}")
        return v.lower()

    @validator("safety_integrity_level")
    def validate_sil(cls, v: str) -> str:
        """Validate SIL level."""
        valid = {"NONE", "SIL_1", "SIL_2", "SIL_3", "SIL_4"}
        if v.upper() not in valid:
            raise ValueError(f"SIL must be one of: {valid}")
        return v.upper()


class MaintenanceTriggerEvent(WebhookEvent):
    """
    Event emitted when maintenance is triggered for an asset.

    This event is generated when predictive maintenance algorithms or
    condition monitoring systems detect a need for maintenance action.

    Attributes:
        asset_id: Unique identifier for the asset
        trigger_level: Severity/urgency of the maintenance trigger
        evidence_refs: References to evidence supporting the trigger
        recommended_task: Description of recommended maintenance task
        predicted_failure_hours: Predicted time to failure in hours
        confidence_score: Confidence in the prediction
        maintenance_type: Type of maintenance (predictive, preventive, corrective)
        estimated_duration_hours: Estimated maintenance duration
        estimated_cost_usd: Estimated maintenance cost

    Example:
        >>> event = MaintenanceTriggerEvent(
        ...     asset_id="FURN-001",
        ...     trigger_level=TriggerLevel.WARNING,
        ...     evidence_refs=["vibration-trend-001", "temperature-anomaly-002"],
        ...     recommended_task="Replace burner nozzle and inspect refractory"
        ... )
    """

    event_type: WebhookEventType = Field(
        default=WebhookEventType.MAINTENANCE_TRIGGER,
        const=True
    )
    asset_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Asset/equipment identifier"
    )
    trigger_level: TriggerLevel = Field(
        ...,
        description="Maintenance trigger severity level"
    )
    evidence_refs: List[str] = Field(
        ...,
        min_items=1,
        description="References to supporting evidence"
    )
    recommended_task: str = Field(
        ...,
        min_length=1,
        max_length=2048,
        description="Recommended maintenance task description"
    )
    predicted_failure_hours: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Predicted time to failure in hours"
    )
    confidence_score: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence in the prediction"
    )
    maintenance_type: str = Field(
        default="predictive",
        description="Type of maintenance"
    )
    estimated_duration_hours: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Estimated maintenance duration"
    )
    estimated_cost_usd: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Estimated maintenance cost"
    )
    asset_criticality: str = Field(
        default="medium",
        description="Asset criticality level"
    )
    affected_systems: List[str] = Field(
        default_factory=list,
        description="Systems affected by this maintenance"
    )
    spare_parts_required: List[str] = Field(
        default_factory=list,
        description="Required spare parts"
    )
    cmms_work_order_id: Optional[str] = Field(
        default=None,
        description="CMMS work order ID if already created"
    )

    @validator("maintenance_type")
    def validate_maintenance_type(cls, v: str) -> str:
        """Validate maintenance type."""
        valid = {"predictive", "preventive", "corrective", "emergency"}
        if v.lower() not in valid:
            raise ValueError(f"Maintenance type must be one of: {valid}")
        return v.lower()

    @validator("asset_criticality")
    def validate_criticality(cls, v: str) -> str:
        """Validate asset criticality."""
        valid = {"low", "medium", "high", "critical"}
        if v.lower() not in valid:
            raise ValueError(f"Asset criticality must be one of: {valid}")
        return v.lower()


# Type alias for any webhook event
AnyWebhookEvent = Union[
    HeatPlanCreatedEvent,
    SetpointRecommendationEvent,
    SafetyActionBlockedEvent,
    MaintenanceTriggerEvent,
    WebhookEvent,
]


def create_event(
    event_type: WebhookEventType,
    **kwargs: Any
) -> WebhookEvent:
    """
    Factory function to create webhook events by type.

    Args:
        event_type: Type of event to create
        **kwargs: Event-specific parameters

    Returns:
        Appropriate WebhookEvent subclass instance

    Raises:
        ValueError: If event_type is not recognized

    Example:
        >>> event = create_event(
        ...     WebhookEventType.HEAT_PLAN_CREATED,
        ...     plan_id="plan-001",
        ...     horizon_hours=24,
        ...     expected_cost_usd=15000.0,
        ...     expected_emissions_kg_co2e=1200.0
        ... )
    """
    event_class_map = {
        WebhookEventType.HEAT_PLAN_CREATED: HeatPlanCreatedEvent,
        WebhookEventType.SETPOINT_RECOMMENDATION: SetpointRecommendationEvent,
        WebhookEventType.SAFETY_ACTION_BLOCKED: SafetyActionBlockedEvent,
        WebhookEventType.MAINTENANCE_TRIGGER: MaintenanceTriggerEvent,
    }

    event_class = event_class_map.get(event_type)

    if event_class is None:
        # Fall back to base WebhookEvent for other types
        return WebhookEvent(event_type=event_type, **kwargs)

    return event_class(**kwargs)
