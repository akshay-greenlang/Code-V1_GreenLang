"""
Agent Event Types for Kafka Streaming.

This module defines the standard event types and data classes for agent
events in the GreenLang Agent Factory Kafka streaming infrastructure.

Features:
- Type-safe event definitions with Pydantic
- SHA-256 provenance hashing for audit trails
- Event versioning support
- Correlation and causation tracking
- Full event lifecycle (created, validated, processed)

Event Types:
- AgentCalculationCompleted: Calculation results with provenance
- AgentAlertRaised: Alert events with severity levels
- AgentRecommendationGenerated: Optimization recommendations
- AgentHealthCheck: Health status events
- AgentConfigurationChanged: Configuration change tracking

Usage:
    from connectors.kafka.events import (
        AgentEvent,
        AgentCalculationCompleted,
        AgentAlertRaised,
    )

    event = AgentCalculationCompleted(
        agent_id="gl-001-carbon-emissions",
        calculation_type="scope1_emissions",
        input_data={"fuel_type": "natural_gas", "quantity": 1000},
        output_data={"emissions_kgco2e": 2500.0},
    )
"""

import hashlib
import json
import logging
import uuid
from abc import ABC
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


# =============================================================================
# Event Type Enumeration
# =============================================================================


class EventType(str, Enum):
    """Standard event types for GreenLang agents."""

    # Calculation events
    CALCULATION_COMPLETED = "agent.calculation.completed"
    CALCULATION_STARTED = "agent.calculation.started"
    CALCULATION_FAILED = "agent.calculation.failed"

    # Alert events
    ALERT_RAISED = "agent.alert.raised"
    ALERT_ACKNOWLEDGED = "agent.alert.acknowledged"
    ALERT_RESOLVED = "agent.alert.resolved"

    # Recommendation events
    RECOMMENDATION_GENERATED = "agent.recommendation.generated"
    RECOMMENDATION_ACCEPTED = "agent.recommendation.accepted"
    RECOMMENDATION_REJECTED = "agent.recommendation.rejected"

    # Health events
    HEALTH_CHECK = "agent.health.check"
    HEALTH_DEGRADED = "agent.health.degraded"
    HEALTH_RECOVERED = "agent.health.recovered"

    # Configuration events
    CONFIG_CHANGED = "agent.config.changed"
    CONFIG_VALIDATED = "agent.config.validated"
    CONFIG_ROLLBACK = "agent.config.rollback"

    # Execution events
    EXECUTION_STARTED = "agent.execution.started"
    EXECUTION_COMPLETED = "agent.execution.completed"
    EXECUTION_FAILED = "agent.execution.failed"
    EXECUTION_TIMEOUT = "agent.execution.timeout"

    # Compliance events
    COMPLIANCE_CHECK = "agent.compliance.check"
    COMPLIANCE_VIOLATION = "agent.compliance.violation"
    COMPLIANCE_CLEARED = "agent.compliance.cleared"


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class HealthStatus(str, Enum):
    """Agent health status values."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class RecommendationPriority(str, Enum):
    """Recommendation priority levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class RecommendationCategory(str, Enum):
    """Recommendation categories."""

    ENERGY_EFFICIENCY = "energy_efficiency"
    EMISSIONS_REDUCTION = "emissions_reduction"
    COST_OPTIMIZATION = "cost_optimization"
    COMPLIANCE = "compliance"
    SAFETY = "safety"
    MAINTENANCE = "maintenance"
    PROCESS_IMPROVEMENT = "process_improvement"


# =============================================================================
# Event Metadata
# =============================================================================


class EventMetadata(BaseModel):
    """Metadata for event tracking and correlation."""

    # Correlation and causation
    correlation_id: Optional[str] = Field(
        None,
        description="ID for correlating related events across services",
    )
    causation_id: Optional[str] = Field(
        None,
        description="ID of the event that caused this event",
    )

    # Source information
    source_system: str = Field(
        "gl-agent-factory",
        description="System that generated the event",
    )
    source_version: str = Field(
        "1.0.0",
        description="Version of the source system",
    )

    # Multi-tenancy
    tenant_id: Optional[str] = Field(
        None,
        description="Tenant identifier for multi-tenant deployments",
    )
    user_id: Optional[str] = Field(
        None,
        description="User who triggered the event",
    )

    # Tracing
    trace_id: Optional[str] = Field(
        None,
        description="Distributed tracing ID",
    )
    span_id: Optional[str] = Field(
        None,
        description="Span ID within trace",
    )

    # Additional context
    tags: Dict[str, str] = Field(
        default_factory=dict,
        description="Additional tags for filtering and routing",
    )


# =============================================================================
# Base Agent Event
# =============================================================================


class AgentEvent(BaseModel):
    """
    Base class for all agent events.

    This class provides:
    - Unique event identification
    - Timestamp with ISO 8601 format
    - Provenance hash for audit trails
    - Event versioning
    - Correlation tracking

    All agent events should inherit from this class.

    Attributes:
        event_id: Unique event identifier (UUID)
        event_type: Type of event from EventType enum
        event_version: Schema version for evolution
        agent_id: Identifier of the agent that generated the event
        timestamp: When the event occurred (ISO 8601)
        payload: Event-specific data
        provenance_hash: SHA-256 hash for verification
        metadata: Additional event metadata

    Example:
        event = AgentEvent(
            event_type=EventType.CALCULATION_COMPLETED,
            agent_id="gl-001-carbon",
            payload={"result": 2500.0},
        )
    """

    event_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique event identifier",
    )
    event_type: str = Field(
        ...,
        description="Event type",
    )
    event_version: str = Field(
        "1.0",
        description="Schema version for event evolution",
    )
    agent_id: str = Field(
        ...,
        description="Agent that generated the event",
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Event timestamp (UTC)",
    )
    payload: Dict[str, Any] = Field(
        default_factory=dict,
        description="Event-specific payload data",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash of event data for verification",
    )
    metadata: EventMetadata = Field(
        default_factory=EventMetadata,
        description="Event metadata",
    )

    @model_validator(mode="after")
    def compute_provenance_hash(self) -> "AgentEvent":
        """Compute provenance hash if not provided."""
        if self.provenance_hash is None:
            self.provenance_hash = self._calculate_hash()
        return self

    def _calculate_hash(self) -> str:
        """
        Calculate SHA-256 hash of event data.

        Returns:
            SHA-256 hex digest
        """
        # Create deterministic representation
        hash_data = {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "agent_id": self.agent_id,
            "timestamp": self.timestamp.isoformat(),
            "payload": self.payload,
        }

        json_str = json.dumps(hash_data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()

    def verify_provenance(self) -> bool:
        """
        Verify that the provenance hash matches the event data.

        Returns:
            True if hash is valid, False otherwise
        """
        calculated = self._calculate_hash()
        return calculated == self.provenance_hash

    def get_partition_key(self) -> str:
        """
        Get the partition key for this event.

        Default implementation uses agent_id for ordering guarantees
        per agent. Override in subclasses for different partitioning.

        Returns:
            Partition key string
        """
        return self.agent_id

    def to_kafka_headers(self) -> List[tuple]:
        """
        Convert metadata to Kafka headers.

        Returns:
            List of (key, value) tuples for Kafka headers
        """
        headers = [
            ("event_type", self.event_type.encode("utf-8")),
            ("event_version", self.event_version.encode("utf-8")),
            ("agent_id", self.agent_id.encode("utf-8")),
            ("timestamp", self.timestamp.isoformat().encode("utf-8")),
            ("provenance_hash", self.provenance_hash.encode("utf-8")),
        ]

        if self.metadata.correlation_id:
            headers.append(("correlation_id", self.metadata.correlation_id.encode("utf-8")))
        if self.metadata.tenant_id:
            headers.append(("tenant_id", self.metadata.tenant_id.encode("utf-8")))
        if self.metadata.trace_id:
            headers.append(("trace_id", self.metadata.trace_id.encode("utf-8")))

        return headers

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


# =============================================================================
# Calculation Events
# =============================================================================


class CalculationInput(BaseModel):
    """Input data for a calculation."""

    parameters: Dict[str, Any] = Field(
        ...,
        description="Calculation input parameters",
    )
    input_hash: str = Field(
        ...,
        description="SHA-256 hash of input parameters",
    )


class CalculationOutput(BaseModel):
    """Output data from a calculation."""

    result: Dict[str, Any] = Field(
        ...,
        description="Calculation result",
    )
    output_hash: str = Field(
        ...,
        description="SHA-256 hash of output",
    )
    unit: Optional[str] = Field(
        None,
        description="Unit of measurement (QUDT)",
    )


class AgentCalculationCompleted(AgentEvent):
    """
    Event emitted when an agent completes a calculation.

    This event captures:
    - Input data with hash
    - Output data with hash
    - Calculation type and formula
    - Processing metrics
    - Full provenance chain

    Example:
        event = AgentCalculationCompleted(
            agent_id="gl-001-carbon-emissions",
            calculation_type="scope1_stationary_combustion",
            formula_id="ghg.scope1.fuel.v1",
            input_data=CalculationInput(
                parameters={"fuel_type": "natural_gas", "quantity_kg": 1000},
                input_hash="abc123...",
            ),
            output_data=CalculationOutput(
                result={"emissions_kgco2e": 2500.0},
                output_hash="def456...",
                unit="kg CO2e",
            ),
        )
    """

    event_type: str = Field(
        EventType.CALCULATION_COMPLETED.value,
        const=True,
    )
    calculation_type: str = Field(
        ...,
        description="Type of calculation performed",
    )
    formula_id: str = Field(
        ...,
        description="Identifier of the formula used",
    )
    formula_version: str = Field(
        "1.0",
        description="Version of the formula",
    )
    input_data: CalculationInput = Field(
        ...,
        description="Calculation input with hash",
    )
    output_data: CalculationOutput = Field(
        ...,
        description="Calculation output with hash",
    )
    calculation_chain_hash: str = Field(
        ...,
        description="Hash of complete calculation chain",
    )
    processing_time_ms: float = Field(
        ...,
        description="Processing time in milliseconds",
    )
    emission_factor_source: Optional[str] = Field(
        None,
        description="Source of emission factors used (EPA, IPCC, etc.)",
    )
    uncertainty_percent: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Uncertainty percentage in result",
    )

    @classmethod
    def create(
        cls,
        agent_id: str,
        calculation_type: str,
        formula_id: str,
        input_params: Dict[str, Any],
        output_result: Dict[str, Any],
        processing_time_ms: float,
        unit: Optional[str] = None,
        emission_factor_source: Optional[str] = None,
        uncertainty_percent: Optional[float] = None,
        metadata: Optional[EventMetadata] = None,
    ) -> "AgentCalculationCompleted":
        """
        Factory method to create a calculation completed event.

        Args:
            agent_id: Agent identifier
            calculation_type: Type of calculation
            formula_id: Formula identifier
            input_params: Input parameters
            output_result: Calculation result
            processing_time_ms: Processing time
            unit: Unit of measurement
            emission_factor_source: Emission factor source
            uncertainty_percent: Result uncertainty
            metadata: Event metadata

        Returns:
            AgentCalculationCompleted event
        """
        # Compute hashes
        input_hash = hashlib.sha256(
            json.dumps(input_params, sort_keys=True, default=str).encode()
        ).hexdigest()
        output_hash = hashlib.sha256(
            json.dumps(output_result, sort_keys=True, default=str).encode()
        ).hexdigest()

        # Compute chain hash
        chain_data = {
            "input_hash": input_hash,
            "formula_id": formula_id,
            "output_hash": output_hash,
        }
        chain_hash = hashlib.sha256(
            json.dumps(chain_data, sort_keys=True).encode()
        ).hexdigest()

        return cls(
            agent_id=agent_id,
            calculation_type=calculation_type,
            formula_id=formula_id,
            input_data=CalculationInput(
                parameters=input_params,
                input_hash=input_hash,
            ),
            output_data=CalculationOutput(
                result=output_result,
                output_hash=output_hash,
                unit=unit,
            ),
            calculation_chain_hash=chain_hash,
            processing_time_ms=processing_time_ms,
            emission_factor_source=emission_factor_source,
            uncertainty_percent=uncertainty_percent,
            metadata=metadata or EventMetadata(),
        )


# =============================================================================
# Alert Events
# =============================================================================


class AlertContext(BaseModel):
    """Context information for an alert."""

    threshold: Optional[float] = Field(
        None,
        description="Threshold that was exceeded",
    )
    actual_value: Optional[float] = Field(
        None,
        description="Actual value that triggered the alert",
    )
    measurement_unit: Optional[str] = Field(
        None,
        description="Unit of measurement",
    )
    related_entity_id: Optional[str] = Field(
        None,
        description="ID of related entity (equipment, process, etc.)",
    )
    location: Optional[str] = Field(
        None,
        description="Physical or logical location",
    )


class AgentAlertRaised(AgentEvent):
    """
    Event emitted when an agent raises an alert.

    This event captures:
    - Alert severity and category
    - Alert message and code
    - Contextual information
    - Recommended actions

    Example:
        event = AgentAlertRaised(
            agent_id="gl-045-combustion-efficiency",
            severity=AlertSeverity.WARNING,
            alert_code="COMB_EFF_LOW",
            title="Combustion Efficiency Below Threshold",
            description="Boiler B1 combustion efficiency at 78%, below 85% threshold",
            context=AlertContext(
                threshold=85.0,
                actual_value=78.0,
                measurement_unit="%",
                related_entity_id="boiler-b1",
            ),
        )
    """

    event_type: str = Field(
        EventType.ALERT_RAISED.value,
        const=True,
    )
    severity: AlertSeverity = Field(
        ...,
        description="Alert severity level",
    )
    alert_code: str = Field(
        ...,
        description="Unique alert code for categorization",
    )
    title: str = Field(
        ...,
        description="Short alert title",
    )
    description: str = Field(
        ...,
        description="Detailed alert description",
    )
    context: AlertContext = Field(
        default_factory=AlertContext,
        description="Alert context information",
    )
    recommended_actions: List[str] = Field(
        default_factory=list,
        description="List of recommended actions",
    )
    auto_resolvable: bool = Field(
        False,
        description="Whether the alert can auto-resolve",
    )
    expiry_time: Optional[datetime] = Field(
        None,
        description="When the alert expires if not resolved",
    )
    related_alerts: List[str] = Field(
        default_factory=list,
        description="IDs of related alerts",
    )

    def get_partition_key(self) -> str:
        """Partition by severity for priority handling."""
        if self.severity == AlertSeverity.CRITICAL:
            return f"critical-{self.agent_id}"
        return self.agent_id


# =============================================================================
# Recommendation Events
# =============================================================================


class RecommendationImpact(BaseModel):
    """Estimated impact of implementing a recommendation."""

    energy_savings_percent: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Estimated energy savings (%)",
    )
    emissions_reduction_kgco2e: Optional[float] = Field(
        None,
        ge=0,
        description="Estimated emissions reduction (kg CO2e)",
    )
    cost_savings_usd: Optional[float] = Field(
        None,
        description="Estimated cost savings (USD)",
    )
    payback_period_months: Optional[int] = Field(
        None,
        ge=0,
        description="Estimated payback period (months)",
    )
    implementation_cost_usd: Optional[float] = Field(
        None,
        ge=0,
        description="Estimated implementation cost (USD)",
    )
    confidence_score: float = Field(
        0.8,
        ge=0,
        le=1,
        description="Confidence in the estimate (0-1)",
    )


class AgentRecommendationGenerated(AgentEvent):
    """
    Event emitted when an agent generates a recommendation.

    This event captures:
    - Recommendation details and rationale
    - Priority and category
    - Estimated impact
    - Implementation steps

    Example:
        event = AgentRecommendationGenerated(
            agent_id="gl-041-waste-heat-recovery",
            priority=RecommendationPriority.HIGH,
            category=RecommendationCategory.ENERGY_EFFICIENCY,
            title="Install Economizer on Boiler B2",
            description="Installing an economizer could recover 15% of waste heat",
            rationale="Flue gas temperature of 280C indicates significant recoverable heat",
            impact=RecommendationImpact(
                energy_savings_percent=12,
                emissions_reduction_kgco2e=50000,
                cost_savings_usd=35000,
                payback_period_months=18,
            ),
        )
    """

    event_type: str = Field(
        EventType.RECOMMENDATION_GENERATED.value,
        const=True,
    )
    priority: RecommendationPriority = Field(
        ...,
        description="Recommendation priority",
    )
    category: RecommendationCategory = Field(
        ...,
        description="Recommendation category",
    )
    title: str = Field(
        ...,
        description="Short recommendation title",
    )
    description: str = Field(
        ...,
        description="Detailed recommendation description",
    )
    rationale: str = Field(
        ...,
        description="Rationale for the recommendation",
    )
    impact: RecommendationImpact = Field(
        ...,
        description="Estimated impact of implementing",
    )
    implementation_steps: List[str] = Field(
        default_factory=list,
        description="Steps to implement the recommendation",
    )
    prerequisites: List[str] = Field(
        default_factory=list,
        description="Prerequisites for implementation",
    )
    related_equipment_ids: List[str] = Field(
        default_factory=list,
        description="IDs of related equipment",
    )
    supporting_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Supporting data for the recommendation",
    )
    expiry_date: Optional[datetime] = Field(
        None,
        description="When the recommendation expires",
    )


# =============================================================================
# Health Check Events
# =============================================================================


class HealthMetrics(BaseModel):
    """Health metrics for an agent."""

    response_time_ms: float = Field(
        ...,
        ge=0,
        description="Response time in milliseconds",
    )
    memory_usage_mb: Optional[float] = Field(
        None,
        ge=0,
        description="Memory usage in megabytes",
    )
    cpu_usage_percent: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="CPU usage percentage",
    )
    error_rate_percent: float = Field(
        0,
        ge=0,
        le=100,
        description="Error rate percentage",
    )
    throughput_per_minute: Optional[float] = Field(
        None,
        ge=0,
        description="Requests processed per minute",
    )
    queue_depth: Optional[int] = Field(
        None,
        ge=0,
        description="Current queue depth",
    )
    last_successful_execution: Optional[datetime] = Field(
        None,
        description="Timestamp of last successful execution",
    )


class DependencyHealth(BaseModel):
    """Health status of a dependency."""

    dependency_name: str = Field(
        ...,
        description="Name of the dependency",
    )
    status: HealthStatus = Field(
        ...,
        description="Health status",
    )
    latency_ms: Optional[float] = Field(
        None,
        ge=0,
        description="Latency in milliseconds",
    )
    error_message: Optional[str] = Field(
        None,
        description="Error message if unhealthy",
    )


class AgentHealthCheck(AgentEvent):
    """
    Event emitted for agent health monitoring.

    This event captures:
    - Overall health status
    - Performance metrics
    - Dependency health
    - Configuration validity

    Example:
        event = AgentHealthCheck(
            agent_id="gl-001-carbon-emissions",
            status=HealthStatus.HEALTHY,
            metrics=HealthMetrics(
                response_time_ms=45,
                memory_usage_mb=256,
                error_rate_percent=0.1,
            ),
        )
    """

    event_type: str = Field(
        EventType.HEALTH_CHECK.value,
        const=True,
    )
    status: HealthStatus = Field(
        ...,
        description="Overall health status",
    )
    metrics: HealthMetrics = Field(
        ...,
        description="Health metrics",
    )
    dependencies: List[DependencyHealth] = Field(
        default_factory=list,
        description="Health of dependencies",
    )
    agent_version: str = Field(
        ...,
        description="Agent version",
    )
    config_valid: bool = Field(
        True,
        description="Whether configuration is valid",
    )
    issues: List[str] = Field(
        default_factory=list,
        description="List of issues if unhealthy",
    )
    last_config_update: Optional[datetime] = Field(
        None,
        description="Last configuration update timestamp",
    )


# =============================================================================
# Configuration Events
# =============================================================================


class ConfigChange(BaseModel):
    """Details of a configuration change."""

    field_path: str = Field(
        ...,
        description="Path to the changed field",
    )
    old_value: Any = Field(
        None,
        description="Previous value",
    )
    new_value: Any = Field(
        ...,
        description="New value",
    )
    change_type: str = Field(
        "update",
        description="Type of change: create, update, delete",
    )


class AgentConfigurationChanged(AgentEvent):
    """
    Event emitted when agent configuration changes.

    This event captures:
    - Configuration changes with diff
    - Who made the change
    - Reason for the change
    - Validation status

    Example:
        event = AgentConfigurationChanged(
            agent_id="gl-045-combustion-efficiency",
            changes=[
                ConfigChange(
                    field_path="thresholds.min_efficiency",
                    old_value=80,
                    new_value=85,
                ),
            ],
            changed_by="admin@company.com",
            change_reason="Increased minimum efficiency threshold per new regulations",
        )
    """

    event_type: str = Field(
        EventType.CONFIG_CHANGED.value,
        const=True,
    )
    changes: List[ConfigChange] = Field(
        ...,
        min_length=1,
        description="List of configuration changes",
    )
    config_version: str = Field(
        ...,
        description="New configuration version",
    )
    previous_config_version: Optional[str] = Field(
        None,
        description="Previous configuration version",
    )
    changed_by: str = Field(
        ...,
        description="User or system that made the change",
    )
    change_reason: str = Field(
        ...,
        description="Reason for the change",
    )
    validated: bool = Field(
        True,
        description="Whether the new config passed validation",
    )
    validation_errors: List[str] = Field(
        default_factory=list,
        description="Validation errors if any",
    )
    rollback_available: bool = Field(
        True,
        description="Whether rollback is available",
    )
    effective_from: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the change becomes effective",
    )


# =============================================================================
# Event Factory
# =============================================================================


class AgentEventFactory:
    """Factory for creating agent events with proper defaults."""

    @staticmethod
    def calculation_completed(
        agent_id: str,
        calculation_type: str,
        formula_id: str,
        input_params: Dict[str, Any],
        output_result: Dict[str, Any],
        processing_time_ms: float,
        **kwargs,
    ) -> AgentCalculationCompleted:
        """Create a calculation completed event."""
        return AgentCalculationCompleted.create(
            agent_id=agent_id,
            calculation_type=calculation_type,
            formula_id=formula_id,
            input_params=input_params,
            output_result=output_result,
            processing_time_ms=processing_time_ms,
            **kwargs,
        )

    @staticmethod
    def alert_raised(
        agent_id: str,
        severity: AlertSeverity,
        alert_code: str,
        title: str,
        description: str,
        **kwargs,
    ) -> AgentAlertRaised:
        """Create an alert raised event."""
        return AgentAlertRaised(
            agent_id=agent_id,
            severity=severity,
            alert_code=alert_code,
            title=title,
            description=description,
            **kwargs,
        )

    @staticmethod
    def recommendation_generated(
        agent_id: str,
        priority: RecommendationPriority,
        category: RecommendationCategory,
        title: str,
        description: str,
        rationale: str,
        impact: RecommendationImpact,
        **kwargs,
    ) -> AgentRecommendationGenerated:
        """Create a recommendation generated event."""
        return AgentRecommendationGenerated(
            agent_id=agent_id,
            priority=priority,
            category=category,
            title=title,
            description=description,
            rationale=rationale,
            impact=impact,
            **kwargs,
        )

    @staticmethod
    def health_check(
        agent_id: str,
        status: HealthStatus,
        metrics: HealthMetrics,
        agent_version: str,
        **kwargs,
    ) -> AgentHealthCheck:
        """Create a health check event."""
        return AgentHealthCheck(
            agent_id=agent_id,
            status=status,
            metrics=metrics,
            agent_version=agent_version,
            **kwargs,
        )

    @staticmethod
    def config_changed(
        agent_id: str,
        changes: List[ConfigChange],
        config_version: str,
        changed_by: str,
        change_reason: str,
        **kwargs,
    ) -> AgentConfigurationChanged:
        """Create a configuration changed event."""
        return AgentConfigurationChanged(
            agent_id=agent_id,
            changes=changes,
            config_version=config_version,
            changed_by=changed_by,
            change_reason=change_reason,
            **kwargs,
        )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enumerations
    "EventType",
    "AlertSeverity",
    "HealthStatus",
    "RecommendationPriority",
    "RecommendationCategory",
    # Metadata
    "EventMetadata",
    # Base event
    "AgentEvent",
    # Calculation events
    "CalculationInput",
    "CalculationOutput",
    "AgentCalculationCompleted",
    # Alert events
    "AlertContext",
    "AgentAlertRaised",
    # Recommendation events
    "RecommendationImpact",
    "AgentRecommendationGenerated",
    # Health events
    "HealthMetrics",
    "DependencyHealth",
    "AgentHealthCheck",
    # Configuration events
    "ConfigChange",
    "AgentConfigurationChanged",
    # Factory
    "AgentEventFactory",
]
