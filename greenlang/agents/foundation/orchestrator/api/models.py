# -*- coding: utf-8 -*-
"""
GreenLang Orchestrator Control Plane API - Request/Response Models
==================================================================

Pydantic models for the FastAPI Control Plane API.

This module defines all request and response models for:
- Pipeline Management (register, list, get, delete)
- Run Operations (submit, list, get, cancel, logs, audit)
- Health and Metrics (health, metrics, ready)
- Agent Registry (list, get)

All models include comprehensive validation and OpenAPI documentation.

Author: GreenLang Team
Version: 1.0.0
GL-FOUND-X-001: Control Plane API Models
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# ENUMS
# =============================================================================


class RunStatus(str, Enum):
    """Status of a pipeline run."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"
    PAUSED = "paused"


class StepStatus(str, Enum):
    """Status of a pipeline step."""
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELED = "canceled"


class LogLevel(str, Enum):
    """Log severity levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class HealthStatus(str, Enum):
    """Service health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


# =============================================================================
# ERROR RESPONSE MODELS
# =============================================================================


class ErrorDetail(BaseModel):
    """Detailed error information."""
    code: str = Field(..., description="GreenLang error code (GL-E-*)")
    message: str = Field(..., description="Human-readable error message")
    field: Optional[str] = Field(None, description="Field that caused the error")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")


class ErrorResponse(BaseModel):
    """Standard error response format."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: List[ErrorDetail] = Field(default_factory=list, description="Detailed errors")
    trace_id: Optional[str] = Field(None, description="Request trace ID for debugging")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")

    model_config = {
        "json_schema_extra": {
            "example": {
                "error": "validation_error",
                "message": "Request validation failed",
                "details": [
                    {
                        "code": "GL-E-PARAM-MISSING",
                        "message": "Required parameter 'pipeline_id' is missing",
                        "field": "pipeline_id"
                    }
                ],
                "trace_id": "trace-abc123",
                "timestamp": "2026-01-27T10:30:00Z"
            }
        }
    }


# =============================================================================
# PIPELINE MODELS
# =============================================================================


class StepDefinitionRequest(BaseModel):
    """Step definition in a pipeline registration request."""
    id: str = Field(..., min_length=1, max_length=63, description="Unique step identifier")
    agent: str = Field(..., description="Agent ID (format: GL-CATEGORY-TYPE-NUMBER)")
    depends_on: List[str] = Field(default_factory=list, alias="dependsOn", description="Step dependencies")
    with_params: Dict[str, Any] = Field(default_factory=dict, alias="with", description="Step parameters")
    timeout_seconds: int = Field(default=900, ge=1, le=86400, alias="timeoutSeconds")
    retries: int = Field(default=0, ge=0, le=10, description="Number of retry attempts")
    condition: Optional[str] = Field(None, description="Conditional execution expression")
    continue_on_error: bool = Field(default=False, alias="continueOnError")

    model_config = {"populate_by_name": True}


class PipelineSpecRequest(BaseModel):
    """Pipeline specification in a registration request."""
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Pipeline parameters schema")
    steps: List[StepDefinitionRequest] = Field(..., min_length=1, description="Pipeline steps")
    concurrency: int = Field(default=10, ge=1, le=100, description="Max concurrent steps")
    timeout_seconds: int = Field(default=3600, ge=60, le=86400, alias="timeoutSeconds")

    model_config = {"populate_by_name": True}


class PipelineMetadataRequest(BaseModel):
    """Pipeline metadata in a registration request."""
    name: str = Field(..., min_length=1, max_length=253, description="Pipeline name")
    namespace: str = Field(default="default", min_length=1, max_length=63, description="Namespace")
    description: Optional[str] = Field(None, max_length=1000, description="Pipeline description")
    version: str = Field(default="1.0.0", description="Pipeline version")
    owner: Optional[str] = Field(None, description="Pipeline owner")
    team: Optional[str] = Field(None, description="Owning team")
    labels: Dict[str, str] = Field(default_factory=dict, description="Labels for filtering")
    annotations: Dict[str, str] = Field(default_factory=dict, description="Annotations")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")


class PipelineRegisterRequest(BaseModel):
    """Request to register a new pipeline."""
    api_version: str = Field(default="greenlang/v1", alias="apiVersion", description="API version")
    kind: str = Field(default="Pipeline", description="Resource kind")
    metadata: PipelineMetadataRequest = Field(..., description="Pipeline metadata")
    spec: PipelineSpecRequest = Field(..., description="Pipeline specification")

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "apiVersion": "greenlang/v1",
                "kind": "Pipeline",
                "metadata": {
                    "name": "carbon-footprint-pipeline",
                    "namespace": "production",
                    "description": "Calculate Scope 1-3 emissions",
                    "version": "1.0.0",
                    "owner": "sustainability-team",
                    "labels": {"domain": "mrv", "priority": "high"}
                },
                "spec": {
                    "parameters": {
                        "input_uri": {"type": "string", "required": True},
                        "reporting_year": {"type": "integer", "default": 2025}
                    },
                    "steps": [
                        {
                            "id": "ingest",
                            "agent": "GL-DATA-X-001",
                            "with": {"uri": "{{ params.input_uri }}"}
                        },
                        {
                            "id": "calculate",
                            "agent": "GL-MRV-X-001",
                            "dependsOn": ["ingest"],
                            "with": {"year": "{{ params.reporting_year }}"}
                        }
                    ]
                }
            }
        }
    }


class PipelineResponse(BaseModel):
    """Response for pipeline operations."""
    pipeline_id: str = Field(..., description="Unique pipeline identifier")
    name: str = Field(..., description="Pipeline name")
    namespace: str = Field(..., description="Pipeline namespace")
    version: str = Field(..., description="Pipeline version")
    description: Optional[str] = Field(None, description="Pipeline description")
    owner: Optional[str] = Field(None, description="Pipeline owner")
    team: Optional[str] = Field(None, description="Owning team")
    labels: Dict[str, str] = Field(default_factory=dict, description="Labels")
    tags: List[str] = Field(default_factory=list, description="Tags")
    step_count: int = Field(..., description="Number of steps")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    content_hash: str = Field(..., description="SHA-256 hash of pipeline content")

    model_config = {
        "json_schema_extra": {
            "example": {
                "pipeline_id": "pipe-abc123",
                "name": "carbon-footprint-pipeline",
                "namespace": "production",
                "version": "1.0.0",
                "description": "Calculate Scope 1-3 emissions",
                "owner": "sustainability-team",
                "team": "climate-tech",
                "labels": {"domain": "mrv"},
                "tags": ["emissions", "scope3"],
                "step_count": 5,
                "created_at": "2026-01-27T10:00:00Z",
                "updated_at": "2026-01-27T10:00:00Z",
                "content_hash": "abc123def456..."
            }
        }
    }


class PipelineListResponse(BaseModel):
    """Response for listing pipelines."""
    pipelines: List[PipelineResponse] = Field(..., description="List of pipelines")
    total: int = Field(..., description="Total count of matching pipelines")
    offset: int = Field(default=0, description="Pagination offset")
    limit: int = Field(default=100, description="Pagination limit")


class PipelineDetailResponse(PipelineResponse):
    """Detailed pipeline response including spec."""
    spec: Dict[str, Any] = Field(..., description="Full pipeline specification")
    execution_order: List[str] = Field(..., description="Topologically sorted step IDs")
    dependency_graph: Dict[str, List[str]] = Field(..., description="Step dependency graph")


# =============================================================================
# RUN MODELS
# =============================================================================


class RunSubmitRequest(BaseModel):
    """Request to submit a new pipeline run."""
    pipeline_id: str = Field(..., description="Pipeline ID to execute")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Run parameters")
    tenant_id: str = Field(..., description="Tenant identifier")
    user_id: Optional[str] = Field(None, description="User initiating the run")
    labels: Dict[str, str] = Field(default_factory=dict, description="Run labels")
    priority: int = Field(default=5, ge=1, le=10, description="Run priority (1=lowest, 10=highest)")
    dry_run: bool = Field(default=False, description="Validate without executing")
    skip_policy_check: bool = Field(default=False, description="Skip policy enforcement (dev only)")
    notification_url: Optional[str] = Field(None, description="Webhook URL for notifications")
    timeout_seconds: Optional[int] = Field(None, ge=60, le=86400, description="Run timeout override")

    model_config = {
        "json_schema_extra": {
            "example": {
                "pipeline_id": "pipe-abc123",
                "parameters": {
                    "input_uri": "s3://data-bucket/emissions-2025.csv",
                    "reporting_year": 2025
                },
                "tenant_id": "tenant-xyz",
                "user_id": "user-123",
                "labels": {"env": "production", "priority": "high"},
                "priority": 7,
                "dry_run": False
            }
        }
    }


class StepStatusResponse(BaseModel):
    """Status of a single pipeline step."""
    step_id: str = Field(..., description="Step identifier")
    agent_id: str = Field(..., description="Agent ID executing the step")
    status: StepStatus = Field(..., description="Current step status")
    started_at: Optional[datetime] = Field(None, description="Start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    duration_ms: Optional[float] = Field(None, description="Execution duration in milliseconds")
    attempts: int = Field(default=0, description="Number of execution attempts")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    output_uri: Optional[str] = Field(None, description="Output artifact URI")
    progress_percent: Optional[float] = Field(None, ge=0, le=100, description="Progress percentage")


class RunResponse(BaseModel):
    """Response for run operations."""
    run_id: str = Field(..., description="Unique run identifier")
    pipeline_id: str = Field(..., description="Pipeline ID")
    pipeline_name: str = Field(..., description="Pipeline name")
    tenant_id: str = Field(..., description="Tenant identifier")
    user_id: Optional[str] = Field(None, description="User who initiated the run")
    status: RunStatus = Field(..., description="Current run status")
    created_at: datetime = Field(..., description="Run creation timestamp")
    started_at: Optional[datetime] = Field(None, description="Run start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Run completion timestamp")
    duration_ms: Optional[float] = Field(None, description="Total duration in milliseconds")
    progress_percent: float = Field(default=0.0, ge=0, le=100, description="Overall progress")
    steps_total: int = Field(..., description="Total number of steps")
    steps_completed: int = Field(default=0, description="Completed steps count")
    steps_failed: int = Field(default=0, description="Failed steps count")
    steps_running: int = Field(default=0, description="Currently running steps")
    labels: Dict[str, str] = Field(default_factory=dict, description="Run labels")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    trace_id: str = Field(..., description="Distributed trace ID")

    model_config = {
        "json_schema_extra": {
            "example": {
                "run_id": "run-xyz789",
                "pipeline_id": "pipe-abc123",
                "pipeline_name": "carbon-footprint-pipeline",
                "tenant_id": "tenant-xyz",
                "user_id": "user-123",
                "status": "running",
                "created_at": "2026-01-27T10:30:00Z",
                "started_at": "2026-01-27T10:30:05Z",
                "progress_percent": 45.0,
                "steps_total": 5,
                "steps_completed": 2,
                "steps_failed": 0,
                "steps_running": 1,
                "labels": {"env": "production"},
                "trace_id": "trace-abc123def456"
            }
        }
    }


class RunListResponse(BaseModel):
    """Response for listing runs."""
    runs: List[RunResponse] = Field(..., description="List of runs")
    total: int = Field(..., description="Total count of matching runs")
    offset: int = Field(default=0, description="Pagination offset")
    limit: int = Field(default=100, description="Pagination limit")


class RunDetailResponse(RunResponse):
    """Detailed run response including step statuses."""
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Run parameters")
    steps: List[StepStatusResponse] = Field(default_factory=list, description="Step statuses")
    plan_id: Optional[str] = Field(None, description="Execution plan ID")
    dry_run: bool = Field(default=False, description="Whether this was a dry run")
    outputs: Optional[Dict[str, Any]] = Field(None, description="Final outputs")
    lineage: Optional[Dict[str, Any]] = Field(None, description="Data lineage information")


class RunCancelRequest(BaseModel):
    """Request to cancel a running run."""
    reason: Optional[str] = Field(None, max_length=500, description="Cancellation reason")
    force: bool = Field(default=False, description="Force immediate cancellation")


class RunCancelResponse(BaseModel):
    """Response for run cancellation."""
    run_id: str = Field(..., description="Run identifier")
    status: RunStatus = Field(..., description="New run status")
    canceled_at: datetime = Field(..., description="Cancellation timestamp")
    reason: Optional[str] = Field(None, description="Cancellation reason")


# =============================================================================
# LOG MODELS
# =============================================================================


class LogEntry(BaseModel):
    """A single log entry."""
    timestamp: datetime = Field(..., description="Log timestamp")
    level: LogLevel = Field(..., description="Log level")
    step_id: Optional[str] = Field(None, description="Associated step ID")
    agent_id: Optional[str] = Field(None, description="Associated agent ID")
    message: str = Field(..., description="Log message")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")


class RunLogsResponse(BaseModel):
    """Response for run logs."""
    run_id: str = Field(..., description="Run identifier")
    logs: List[LogEntry] = Field(..., description="Log entries")
    total: int = Field(..., description="Total log count")
    has_more: bool = Field(default=False, description="Whether more logs are available")
    next_cursor: Optional[str] = Field(None, description="Cursor for pagination")


# =============================================================================
# AUDIT MODELS
# =============================================================================


class AuditEvent(BaseModel):
    """A single audit event."""
    event_id: str = Field(..., description="Unique event identifier")
    event_type: str = Field(..., description="Event type (e.g., RUN_SUBMITTED, STEP_COMPLETED)")
    timestamp: datetime = Field(..., description="Event timestamp")
    step_id: Optional[str] = Field(None, description="Associated step ID")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Event payload")
    prev_event_hash: str = Field(..., description="Previous event hash (for chain verification)")
    event_hash: str = Field(..., description="This event's hash")


class RunAuditResponse(BaseModel):
    """Response for run audit trail."""
    run_id: str = Field(..., description="Run identifier")
    events: List[AuditEvent] = Field(..., description="Audit events")
    chain_valid: bool = Field(..., description="Whether hash chain is valid")
    event_count: int = Field(..., description="Total event count")
    exported_at: datetime = Field(..., description="Export timestamp")


# =============================================================================
# HEALTH AND METRICS MODELS
# =============================================================================


class ComponentHealth(BaseModel):
    """Health status of a system component."""
    name: str = Field(..., description="Component name")
    status: HealthStatus = Field(..., description="Component health status")
    latency_ms: Optional[float] = Field(None, description="Latency in milliseconds")
    message: Optional[str] = Field(None, description="Status message")
    last_checked: datetime = Field(..., description="Last health check timestamp")


class HealthResponse(BaseModel):
    """Response for health check endpoint."""
    status: HealthStatus = Field(..., description="Overall health status")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(..., description="Check timestamp")
    components: List[ComponentHealth] = Field(default_factory=list, description="Component health")

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "timestamp": "2026-01-27T10:30:00Z",
                "components": [
                    {
                        "name": "database",
                        "status": "healthy",
                        "latency_ms": 5.2,
                        "last_checked": "2026-01-27T10:30:00Z"
                    },
                    {
                        "name": "k8s_executor",
                        "status": "healthy",
                        "latency_ms": 12.5,
                        "last_checked": "2026-01-27T10:30:00Z"
                    }
                ]
            }
        }
    }


class ReadinessResponse(BaseModel):
    """Response for readiness probe."""
    ready: bool = Field(..., description="Whether service is ready")
    message: Optional[str] = Field(None, description="Status message")


class MetricsResponse(BaseModel):
    """Response for metrics endpoint (Prometheus format or JSON)."""
    runs_total: int = Field(..., description="Total runs processed")
    runs_active: int = Field(..., description="Currently active runs")
    runs_succeeded: int = Field(..., description="Successful runs")
    runs_failed: int = Field(..., description="Failed runs")
    steps_total: int = Field(..., description="Total steps executed")
    steps_succeeded: int = Field(..., description="Successful steps")
    steps_failed: int = Field(..., description="Failed steps")
    avg_run_duration_ms: float = Field(..., description="Average run duration")
    avg_step_duration_ms: float = Field(..., description="Average step duration")
    policy_evaluations: int = Field(..., description="Total policy evaluations")
    policy_denials: int = Field(..., description="Policy denials")
    uptime_seconds: float = Field(..., description="Service uptime")
    timestamp: datetime = Field(..., description="Metrics timestamp")


# =============================================================================
# AGENT REGISTRY MODELS
# =============================================================================


class AgentCapabilityResponse(BaseModel):
    """Agent capability information."""
    name: str = Field(..., description="Capability name")
    category: str = Field(..., description="Capability category")
    description: str = Field(..., description="Capability description")
    input_types: List[str] = Field(default_factory=list, description="Supported input types")
    output_types: List[str] = Field(default_factory=list, description="Output types")


class AgentResponse(BaseModel):
    """Response for agent registry operations."""
    agent_id: str = Field(..., description="Agent identifier")
    name: str = Field(..., description="Agent name")
    description: str = Field(..., description="Agent description")
    version: str = Field(..., description="Agent version")
    layer: str = Field(..., description="Agent layer (e.g., foundation, data, mrv)")
    sectors: List[str] = Field(default_factory=list, description="Applicable sectors")
    capabilities: List[AgentCapabilityResponse] = Field(default_factory=list, description="Capabilities")
    execution_mode: str = Field(..., description="Execution mode (glip_v1, legacy_http, hybrid)")
    health_status: str = Field(..., description="Agent health status")
    is_glip_compatible: bool = Field(..., description="Whether agent supports GLIP v1")
    is_idempotent: bool = Field(..., description="Whether agent is fully idempotent")
    deterministic: bool = Field(..., description="Whether agent produces deterministic outputs")
    tags: List[str] = Field(default_factory=list, description="Agent tags")

    model_config = {
        "json_schema_extra": {
            "example": {
                "agent_id": "GL-MRV-X-001",
                "name": "Emissions Calculator",
                "description": "Calculate Scope 1-3 greenhouse gas emissions",
                "version": "1.2.0",
                "layer": "mrv",
                "sectors": ["energy", "industrials"],
                "capabilities": [
                    {
                        "name": "emissions_calculation",
                        "category": "calculation",
                        "description": "Calculate GHG emissions",
                        "input_types": ["activity_data"],
                        "output_types": ["emissions_report"]
                    }
                ],
                "execution_mode": "glip_v1",
                "health_status": "healthy",
                "is_glip_compatible": True,
                "is_idempotent": True,
                "deterministic": True,
                "tags": ["emissions", "ghg", "mrv"]
            }
        }
    }


class AgentListResponse(BaseModel):
    """Response for listing agents."""
    agents: List[AgentResponse] = Field(..., description="List of agents")
    total: int = Field(..., description="Total count of matching agents")
    offset: int = Field(default=0, description="Pagination offset")
    limit: int = Field(default=100, description="Pagination limit")


class AgentDetailResponse(AgentResponse):
    """Detailed agent response with full metadata."""
    resource_profile: Optional[Dict[str, Any]] = Field(None, description="Resource requirements")
    container_spec: Optional[Dict[str, Any]] = Field(None, description="Container specification")
    dependencies: List[Dict[str, Any]] = Field(default_factory=list, description="Agent dependencies")
    variants: List[Dict[str, Any]] = Field(default_factory=list, description="Agent variants")
    registered_at: datetime = Field(..., description="Registration timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


# =============================================================================
# QUERY PARAMETERS (for use with FastAPI Query)
# =============================================================================


class PipelineListParams(BaseModel):
    """Query parameters for listing pipelines."""
    namespace: Optional[str] = Field(None, description="Filter by namespace")
    owner: Optional[str] = Field(None, description="Filter by owner")
    team: Optional[str] = Field(None, description="Filter by team")
    label: Optional[str] = Field(None, description="Filter by label (key=value)")
    search: Optional[str] = Field(None, description="Search in name/description")
    offset: int = Field(default=0, ge=0, description="Pagination offset")
    limit: int = Field(default=100, ge=1, le=1000, description="Pagination limit")


class RunListParams(BaseModel):
    """Query parameters for listing runs."""
    status: Optional[RunStatus] = Field(None, description="Filter by status")
    pipeline_id: Optional[str] = Field(None, description="Filter by pipeline ID")
    tenant_id: Optional[str] = Field(None, description="Filter by tenant ID")
    user_id: Optional[str] = Field(None, description="Filter by user ID")
    created_after: Optional[datetime] = Field(None, description="Filter by creation date (after)")
    created_before: Optional[datetime] = Field(None, description="Filter by creation date (before)")
    label: Optional[str] = Field(None, description="Filter by label (key=value)")
    offset: int = Field(default=0, ge=0, description="Pagination offset")
    limit: int = Field(default=100, ge=1, le=1000, description="Pagination limit")


class LogsQueryParams(BaseModel):
    """Query parameters for logs."""
    level: Optional[LogLevel] = Field(None, description="Filter by log level")
    step_id: Optional[str] = Field(None, description="Filter by step ID")
    since: Optional[datetime] = Field(None, description="Logs after this timestamp")
    until: Optional[datetime] = Field(None, description="Logs before this timestamp")
    search: Optional[str] = Field(None, description="Search in log messages")
    cursor: Optional[str] = Field(None, description="Pagination cursor")
    limit: int = Field(default=100, ge=1, le=1000, description="Max entries to return")


class AgentListParams(BaseModel):
    """Query parameters for listing agents."""
    layer: Optional[str] = Field(None, description="Filter by layer")
    sector: Optional[str] = Field(None, description="Filter by sector")
    capability: Optional[str] = Field(None, description="Filter by capability")
    execution_mode: Optional[str] = Field(None, description="Filter by execution mode")
    glip_compatible: Optional[bool] = Field(None, description="Filter GLIP v1 compatible agents")
    health_status: Optional[str] = Field(None, description="Filter by health status")
    search: Optional[str] = Field(None, description="Search in name/description")
    offset: int = Field(default=0, ge=0, description="Pagination offset")
    limit: int = Field(default=100, ge=1, le=1000, description="Pagination limit")


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "RunStatus",
    "StepStatus",
    "LogLevel",
    "HealthStatus",
    # Error models
    "ErrorDetail",
    "ErrorResponse",
    # Pipeline models
    "StepDefinitionRequest",
    "PipelineSpecRequest",
    "PipelineMetadataRequest",
    "PipelineRegisterRequest",
    "PipelineResponse",
    "PipelineListResponse",
    "PipelineDetailResponse",
    # Run models
    "RunSubmitRequest",
    "StepStatusResponse",
    "RunResponse",
    "RunListResponse",
    "RunDetailResponse",
    "RunCancelRequest",
    "RunCancelResponse",
    # Log models
    "LogEntry",
    "RunLogsResponse",
    # Audit models
    "AuditEvent",
    "RunAuditResponse",
    # Health models
    "ComponentHealth",
    "HealthResponse",
    "ReadinessResponse",
    "MetricsResponse",
    # Agent models
    "AgentCapabilityResponse",
    "AgentResponse",
    "AgentListResponse",
    "AgentDetailResponse",
    # Query params
    "PipelineListParams",
    "RunListParams",
    "LogsQueryParams",
    "AgentListParams",
    # FR-074: Checkpoint and retry models
    "CheckpointStatusEnum",
    "StepCheckpointResponse",
    "RunCheckpointResponse",
    "RunRetryRequest",
    "RunRetryResponse",
    "NonIdempotentStepWarning",
    "CheckpointClearResponse",
]


# =============================================================================
# APPROVAL MODELS (FR-043)
# =============================================================================


class ApprovalStatusEnum(str, Enum):
    """Status of an approval request."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


class ApprovalDecisionEnum(str, Enum):
    """Decision made by an approver."""
    APPROVED = "approved"
    REJECTED = "rejected"


class ApprovalSubmitRequest(BaseModel):
    """Request to submit an approval decision."""
    decision: ApprovalDecisionEnum = Field(..., description="APPROVED or REJECTED")
    reason: Optional[str] = Field(None, max_length=2000, description="Explanation for decision")
    signature: str = Field(..., description="Base64-encoded Ed25519 signature")
    public_key: str = Field(..., description="Base64-encoded public key")
    approver_name: Optional[str] = Field(None, description="Human-readable name")
    approver_role: Optional[str] = Field(None, description="Approver role")

    model_config = {
        "json_schema_extra": {
            "example": {
                "decision": "approved",
                "reason": "Verified calculation methodology is correct",
                "signature": "base64-encoded-ed25519-signature",
                "public_key": "base64-encoded-public-key",
                "approver_name": "John Smith",
                "approver_role": "Manager"
            }
        }
    }


class ApprovalAttestationResponse(BaseModel):
    """Response containing signed attestation details."""
    approver_id: str = Field(..., description="Approver identifier")
    approver_name: Optional[str] = Field(None, description="Approver name")
    approver_role: Optional[str] = Field(None, description="Approver role")
    decision: ApprovalDecisionEnum = Field(..., description="Decision made")
    reason: Optional[str] = Field(None, description="Decision reason")
    timestamp: datetime = Field(..., description="Attestation timestamp")
    signature: str = Field(..., description="Truncated signature for display")
    attestation_hash: str = Field(..., description="SHA-256 hash of attestation")
    signature_valid: Optional[bool] = Field(None, description="Whether signature verified")


class ApprovalRequestResponse(BaseModel):
    """Response for an approval request."""
    request_id: str = Field(..., description="Unique request identifier")
    run_id: str = Field(..., description="Associated run ID")
    step_id: str = Field(..., description="Step requiring approval")
    approval_type: str = Field(..., description="Type of approval required")
    reason: str = Field(..., description="Why approval is required")
    requested_by: Optional[str] = Field(None, description="Who requested approval")
    requested_at: datetime = Field(..., description="Request timestamp")
    deadline: datetime = Field(..., description="Approval deadline")
    status: ApprovalStatusEnum = Field(..., description="Current status")
    attestation: Optional[ApprovalAttestationResponse] = Field(None, description="Signed attestation if decided")
    provenance_hash: str = Field(..., description="Provenance hash for audit")

    model_config = {
        "json_schema_extra": {
            "example": {
                "request_id": "apr-abc123def456",
                "run_id": "run-xyz789",
                "step_id": "step-calculate",
                "approval_type": "manager",
                "reason": "High-value calculation requires manager approval",
                "requested_by": "system",
                "requested_at": "2026-01-28T10:00:00Z",
                "deadline": "2026-01-29T10:00:00Z",
                "status": "pending",
                "attestation": None,
                "provenance_hash": "abc123..."
            }
        }
    }


class ApprovalListResponse(BaseModel):
    """Response for listing approvals."""
    approvals: List[ApprovalRequestResponse] = Field(..., description="List of approvals")
    total: int = Field(..., description="Total count")
    run_id: str = Field(..., description="Run ID filter")


class ApprovalSubmitResponse(BaseModel):
    """Response after submitting an approval."""
    request_id: str = Field(..., description="Approval request ID")
    status: ApprovalStatusEnum = Field(..., description="New status")
    attestation: ApprovalAttestationResponse = Field(..., description="Signed attestation")
    message: str = Field(..., description="Confirmation message")


# =============================================================================
# CHECKPOINT AND RETRY MODELS (FR-074)
# =============================================================================


class CheckpointStatusEnum(str, Enum):
    """Status of a step checkpoint."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELED = "canceled"


class StepCheckpointResponse(BaseModel):
    """Response for a single step checkpoint state."""
    step_id: str = Field(..., description="Step identifier")
    status: CheckpointStatusEnum = Field(..., description="Checkpoint status")
    outputs: Dict[str, Any] = Field(default_factory=dict, description="Step outputs")
    artifacts: List[str] = Field(default_factory=list, description="Artifact URIs")
    idempotency_key: str = Field(..., description="Idempotency key for deduplication")
    attempt: int = Field(default=1, ge=1, description="Execution attempt number")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    started_at: Optional[datetime] = Field(None, description="Execution start time")
    completed_at: Optional[datetime] = Field(None, description="Execution completion time")

    model_config = {
        "json_schema_extra": {
            "example": {
                "step_id": "step-calculate",
                "status": "completed",
                "outputs": {"emissions_total": 1500.5, "unit": "tCO2e"},
                "artifacts": ["s3://bucket/artifacts/emissions-report.json"],
                "idempotency_key": "abc123def456...",
                "attempt": 1,
                "started_at": "2026-01-28T10:00:00Z",
                "completed_at": "2026-01-28T10:05:00Z"
            }
        }
    }


class RunCheckpointResponse(BaseModel):
    """Response for run checkpoint state."""
    run_id: str = Field(..., description="Run identifier")
    plan_id: str = Field(..., description="Execution plan ID")
    plan_hash: str = Field(..., description="SHA-256 hash of the execution plan")
    pipeline_id: str = Field(..., description="Pipeline identifier")
    step_checkpoints: Dict[str, StepCheckpointResponse] = Field(
        default_factory=dict,
        description="Checkpoint state for each step"
    )
    last_successful_step: Optional[str] = Field(None, description="Last successfully completed step")
    retry_count: int = Field(default=0, ge=0, description="Number of retry attempts")
    parent_run_id: Optional[str] = Field(None, description="Parent run ID if this is a retry")
    created_at: datetime = Field(..., description="Checkpoint creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    expires_at: Optional[datetime] = Field(None, description="Checkpoint expiration timestamp")
    is_expired: bool = Field(default=False, description="Whether checkpoint has expired")
    completed_steps: List[str] = Field(default_factory=list, description="List of completed step IDs")
    failed_steps: List[str] = Field(default_factory=list, description="List of failed step IDs")
    state_hash: str = Field(..., description="SHA-256 hash of checkpoint state for integrity")

    model_config = {
        "json_schema_extra": {
            "example": {
                "run_id": "run-xyz789",
                "plan_id": "plan-abc123",
                "plan_hash": "sha256-abc123...",
                "pipeline_id": "pipe-abc123",
                "step_checkpoints": {
                    "step-ingest": {
                        "step_id": "step-ingest",
                        "status": "completed",
                        "outputs": {"records_processed": 1000},
                        "idempotency_key": "key-123",
                        "attempt": 1
                    }
                },
                "last_successful_step": "step-ingest",
                "retry_count": 0,
                "created_at": "2026-01-28T10:00:00Z",
                "updated_at": "2026-01-28T10:05:00Z",
                "is_expired": False,
                "completed_steps": ["step-ingest"],
                "failed_steps": [],
                "state_hash": "sha256-def456..."
            }
        }
    }


class RunRetryRequest(BaseModel):
    """Request to retry a failed run from checkpoint."""
    from_checkpoint: bool = Field(
        default=True,
        description="Whether to resume from checkpoint (skip completed steps)"
    )
    skip_succeeded: bool = Field(
        default=True,
        description="Skip steps that succeeded in the original run"
    )
    force_rerun_steps: List[str] = Field(
        default_factory=list,
        description="List of step IDs to force re-run even if they succeeded"
    )
    new_parameters: Optional[Dict[str, Any]] = Field(
        None,
        description="Override parameters for the retry run"
    )
    reason: Optional[str] = Field(
        None,
        max_length=1000,
        description="Reason for retry (for audit trail)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "from_checkpoint": True,
                "skip_succeeded": True,
                "force_rerun_steps": ["step-calculate"],
                "new_parameters": {"reporting_year": 2026},
                "reason": "Recalculating with updated emission factors"
            }
        }
    }


class NonIdempotentStepWarning(BaseModel):
    """Warning about a non-idempotent step that may have side effects."""
    step_id: str = Field(..., description="Step identifier")
    warning_message: str = Field(..., description="Warning description")
    idempotency_behavior: str = Field(..., description="Step's idempotency behavior")
    recommendation: str = Field(..., description="Recommended action")


class RunRetryResponse(BaseModel):
    """Response for run retry operation."""
    new_run_id: str = Field(..., description="New run identifier for the retry")
    original_run_id: str = Field(..., description="Original run that is being retried")
    skipped_steps: List[str] = Field(
        default_factory=list,
        description="Steps that will be skipped (using checkpoint outputs)"
    )
    steps_to_execute: List[str] = Field(
        default_factory=list,
        description="Steps that will be executed"
    )
    retry_count: int = Field(..., ge=1, description="Retry attempt number")
    from_checkpoint: bool = Field(..., description="Whether resuming from checkpoint")
    non_idempotent_warnings: List[NonIdempotentStepWarning] = Field(
        default_factory=list,
        description="Warnings about non-idempotent steps being re-run"
    )
    schema_compatible: bool = Field(
        default=True,
        description="Whether the current plan is compatible with checkpoint"
    )
    created_at: datetime = Field(..., description="Retry submission timestamp")
    message: str = Field(..., description="Status message")

    model_config = {
        "json_schema_extra": {
            "example": {
                "new_run_id": "run-retry-001",
                "original_run_id": "run-xyz789",
                "skipped_steps": ["step-ingest", "step-validate"],
                "steps_to_execute": ["step-calculate", "step-report"],
                "retry_count": 1,
                "from_checkpoint": True,
                "non_idempotent_warnings": [],
                "schema_compatible": True,
                "created_at": "2026-01-28T12:00:00Z",
                "message": "Retry submitted successfully. Resuming from step 'step-calculate'."
            }
        }
    }


class CheckpointClearResponse(BaseModel):
    """Response for checkpoint clear operation."""
    run_id: str = Field(..., description="Run identifier")
    cleared: bool = Field(..., description="Whether checkpoint was successfully cleared")
    message: str = Field(..., description="Status message")
