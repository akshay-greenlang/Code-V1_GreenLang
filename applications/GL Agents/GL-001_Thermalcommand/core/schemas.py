"""
GL-001 ThermalCommand Orchestrator - Schema Definitions

This module defines all Pydantic models for inputs, outputs, workflows,
and status reporting for the ThermalCommand Orchestrator.
"""

from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Union
import uuid

from pydantic import BaseModel, Field, validator


# =============================================================================
# ENUMS
# =============================================================================

class WorkflowType(Enum):
    """Types of orchestration workflows."""
    OPTIMIZATION = auto()
    MONITORING = auto()
    COMPLIANCE = auto()
    MAINTENANCE = auto()
    EMERGENCY = auto()
    CALIBRATION = auto()
    REPORTING = auto()


class WorkflowStatus(Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class TaskStatus(Enum):
    """Individual task status."""
    QUEUED = "queued"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class AgentHealthStatus(Enum):
    """Agent health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"


class Priority(Enum):
    """Task/workflow priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5


# =============================================================================
# INPUT SCHEMAS
# =============================================================================

class OrchestratorInput(BaseModel):
    """Input data for orchestrator operations."""

    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique request identifier"
    )
    request_type: str = Field(..., description="Type of orchestrator request")
    priority: Priority = Field(
        default=Priority.NORMAL,
        description="Request priority"
    )
    source_agent: Optional[str] = Field(
        default=None,
        description="Source agent ID if from another agent"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Request timestamp"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Request parameters"
    )
    constraints: Dict[str, Any] = Field(
        default_factory=dict,
        description="Execution constraints"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )

    class Config:
        use_enum_values = True


class WorkflowSpec(BaseModel):
    """Specification for a workflow execution."""

    workflow_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique workflow identifier"
    )
    workflow_type: WorkflowType = Field(..., description="Type of workflow")
    name: str = Field(..., description="Workflow name")
    description: str = Field(default="", description="Workflow description")
    priority: Priority = Field(
        default=Priority.NORMAL,
        description="Workflow priority"
    )
    tasks: List["TaskSpec"] = Field(
        default_factory=list,
        description="Tasks in workflow"
    )
    dependencies: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Task dependencies (task_id -> [dependency_ids])"
    )
    timeout_s: float = Field(
        default=300.0,
        ge=1.0,
        le=86400.0,
        description="Workflow timeout in seconds"
    )
    retry_count: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Number of retries on failure"
    )
    rollback_enabled: bool = Field(
        default=True,
        description="Enable rollback on failure"
    )
    checkpoint_enabled: bool = Field(
        default=True,
        description="Enable workflow checkpointing"
    )
    required_agents: Set[str] = Field(
        default_factory=set,
        description="Required agent types"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Workflow parameters"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )

    class Config:
        use_enum_values = True


class TaskSpec(BaseModel):
    """Specification for a workflow task."""

    task_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique task identifier"
    )
    task_type: str = Field(..., description="Task type identifier")
    name: str = Field(..., description="Task name")
    description: str = Field(default="", description="Task description")
    target_agent_type: str = Field(
        ...,
        description="Target agent type (e.g., GL-002)"
    )
    target_agent_id: Optional[str] = Field(
        default=None,
        description="Specific agent ID (None for any of type)"
    )
    priority: Priority = Field(
        default=Priority.NORMAL,
        description="Task priority"
    )
    timeout_s: float = Field(
        default=60.0,
        ge=1.0,
        le=3600.0,
        description="Task timeout"
    )
    retry_count: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Task retry count"
    )
    inputs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Task input parameters"
    )
    expected_outputs: List[str] = Field(
        default_factory=list,
        description="Expected output fields"
    )
    validation_rules: List[str] = Field(
        default_factory=list,
        description="Output validation rules"
    )

    class Config:
        use_enum_values = True


# =============================================================================
# OUTPUT SCHEMAS
# =============================================================================

class OrchestratorOutput(BaseModel):
    """Output from orchestrator operations."""

    request_id: str = Field(..., description="Original request ID")
    response_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Response identifier"
    )
    status: str = Field(..., description="Operation status")
    success: bool = Field(..., description="Operation success flag")
    result: Optional[Any] = Field(default=None, description="Operation result")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Response timestamp"
    )
    processing_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Processing time in milliseconds"
    )
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 provenance hash"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )


class WorkflowResult(BaseModel):
    """Result from workflow execution."""

    workflow_id: str = Field(..., description="Workflow identifier")
    status: WorkflowStatus = Field(..., description="Workflow status")
    start_time: datetime = Field(..., description="Workflow start time")
    end_time: Optional[datetime] = Field(
        default=None,
        description="Workflow end time"
    )
    duration_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Execution duration"
    )
    tasks_completed: int = Field(default=0, ge=0, description="Completed task count")
    tasks_failed: int = Field(default=0, ge=0, description="Failed task count")
    tasks_total: int = Field(default=0, ge=0, description="Total task count")
    task_results: Dict[str, "TaskResult"] = Field(
        default_factory=dict,
        description="Results per task"
    )
    final_output: Optional[Any] = Field(
        default=None,
        description="Final workflow output"
    )
    error: Optional[str] = Field(default=None, description="Error if failed")
    checkpoints: List[str] = Field(
        default_factory=list,
        description="Checkpoint IDs"
    )
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 provenance hash"
    )

    class Config:
        use_enum_values = True


class TaskResult(BaseModel):
    """Result from task execution."""

    task_id: str = Field(..., description="Task identifier")
    status: TaskStatus = Field(..., description="Task status")
    assigned_agent: Optional[str] = Field(
        default=None,
        description="Agent that executed task"
    )
    start_time: Optional[datetime] = Field(
        default=None,
        description="Task start time"
    )
    end_time: Optional[datetime] = Field(
        default=None,
        description="Task end time"
    )
    duration_ms: float = Field(default=0.0, ge=0.0, description="Task duration")
    output: Optional[Any] = Field(default=None, description="Task output")
    error: Optional[str] = Field(default=None, description="Error if failed")
    retry_count: int = Field(default=0, ge=0, description="Retries attempted")
    metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Task metrics"
    )

    class Config:
        use_enum_values = True


# =============================================================================
# STATUS SCHEMAS
# =============================================================================

class AgentStatus(BaseModel):
    """Status of a registered agent."""

    agent_id: str = Field(..., description="Agent identifier")
    agent_type: str = Field(..., description="Agent type (e.g., GL-002)")
    name: str = Field(..., description="Agent name")
    health: AgentHealthStatus = Field(..., description="Health status")
    version: str = Field(..., description="Agent version")
    last_heartbeat: datetime = Field(..., description="Last heartbeat time")
    active_tasks: int = Field(default=0, ge=0, description="Active task count")
    completed_tasks: int = Field(default=0, ge=0, description="Completed tasks")
    failed_tasks: int = Field(default=0, ge=0, description="Failed tasks")
    capabilities: Set[str] = Field(
        default_factory=set,
        description="Agent capabilities"
    )
    metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Agent metrics"
    )

    class Config:
        use_enum_values = True


class SystemStatus(BaseModel):
    """Overall system status."""

    orchestrator_id: str = Field(..., description="Orchestrator identifier")
    orchestrator_name: str = Field(..., description="Orchestrator name")
    orchestrator_version: str = Field(..., description="Orchestrator version")
    status: str = Field(..., description="System status")
    uptime_seconds: float = Field(default=0.0, ge=0.0, description="Uptime in seconds")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Status timestamp"
    )

    # Agent status
    registered_agents: int = Field(default=0, ge=0, description="Registered agent count")
    healthy_agents: int = Field(default=0, ge=0, description="Healthy agent count")
    agents: List[AgentStatus] = Field(
        default_factory=list,
        description="Individual agent statuses"
    )

    # Workflow status
    active_workflows: int = Field(default=0, ge=0, description="Active workflow count")
    pending_workflows: int = Field(default=0, ge=0, description="Pending workflows")
    completed_workflows: int = Field(default=0, ge=0, description="Completed workflows today")

    # Safety status
    safety_level: str = Field(default="SIL_3", description="Active safety level")
    safety_status: str = Field(default="normal", description="Safety system status")
    esd_armed: bool = Field(default=True, description="ESD system armed")
    active_alarms: int = Field(default=0, ge=0, description="Active alarm count")

    # Performance metrics
    cpu_usage_percent: float = Field(default=0.0, ge=0.0, le=100.0)
    memory_usage_percent: float = Field(default=0.0, ge=0.0, le=100.0)
    task_queue_depth: int = Field(default=0, ge=0, description="Task queue depth")
    avg_task_latency_ms: float = Field(default=0.0, ge=0.0, description="Average task latency")

    # Integration status
    integrations: Dict[str, str] = Field(
        default_factory=dict,
        description="Integration connection status"
    )


# =============================================================================
# EVENT SCHEMAS
# =============================================================================

class OrchestratorEvent(BaseModel):
    """Event emitted by the orchestrator."""

    event_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Event identifier"
    )
    event_type: str = Field(..., description="Event type")
    source: str = Field(..., description="Event source (orchestrator ID)")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Event timestamp"
    )
    priority: Priority = Field(
        default=Priority.NORMAL,
        description="Event priority"
    )
    payload: Dict[str, Any] = Field(
        default_factory=dict,
        description="Event payload"
    )
    correlation_id: Optional[str] = Field(
        default=None,
        description="Correlation ID"
    )

    class Config:
        use_enum_values = True


class SafetyEvent(BaseModel):
    """Safety-related event."""

    event_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Event identifier"
    )
    event_type: str = Field(..., description="Safety event type")
    severity: str = Field(..., description="Severity (info, warning, critical)")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Event timestamp"
    )
    source_equipment: Optional[str] = Field(
        default=None,
        description="Source equipment"
    )
    description: str = Field(..., description="Event description")
    threshold_value: Optional[float] = Field(
        default=None,
        description="Threshold that was exceeded"
    )
    actual_value: Optional[float] = Field(
        default=None,
        description="Actual value that triggered event"
    )
    response_action: Optional[str] = Field(
        default=None,
        description="Recommended/taken action"
    )
    acknowledged: bool = Field(default=False, description="Event acknowledged")
    acknowledged_by: Optional[str] = Field(
        default=None,
        description="User who acknowledged"
    )


# Update forward references
WorkflowSpec.update_forward_refs()
WorkflowResult.update_forward_refs()
