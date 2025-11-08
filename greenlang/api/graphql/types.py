"""
GraphQL Type Definitions using Strawberry
Comprehensive type system for GreenLang GraphQL API
"""

from __future__ import annotations
import strawberry
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


# ==============================================================================
# Enum Types
# ==============================================================================

@strawberry.enum
class PermissionAction(Enum):
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    EXECUTE = "execute"
    LIST = "list"
    APPROVE = "approve"
    ADMIN = "admin"
    ALL = "*"


@strawberry.enum
class ResourceType(Enum):
    PIPELINE = "pipeline"
    PACK = "pack"
    DATASET = "dataset"
    MODEL = "model"
    AGENT = "agent"
    WORKFLOW = "workflow"
    TENANT = "tenant"
    USER = "user"
    ROLE = "role"
    API_KEY = "api_key"
    CLUSTER = "cluster"
    NAMESPACE = "namespace"
    SECRET = "secret"
    CONFIG = "config"
    ALL = "*"


@strawberry.enum
class ExecutionStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@strawberry.enum
class SortOrder(Enum):
    ASC = "asc"
    DESC = "desc"


@strawberry.enum
class AgentStatus(Enum):
    ENABLED = "enabled"
    DISABLED = "disabled"
    ARCHIVED = "archived"


@strawberry.enum
class WorkflowStepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@strawberry.enum
class SubscriptionEvent(Enum):
    CREATED = "created"
    UPDATED = "updated"
    DELETED = "deleted"
    STATUS_CHANGED = "status_changed"
    PROGRESS_UPDATE = "progress_update"


@strawberry.enum
class FilterOperator(Enum):
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    GREATER_THAN_OR_EQUAL = "gte"
    LESS_THAN_OR_EQUAL = "lte"
    IN = "in"
    NOT_IN = "not_in"
    IS_NULL = "is_null"
    IS_NOT_NULL = "is_not_null"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"


@strawberry.enum
class AggregationType(Enum):
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    COUNT = "count"


# ==============================================================================
# Scalar Types
# ==============================================================================

JSON = strawberry.scalar(
    Dict[str, Any],
    serialize=lambda v: v,
    parse_value=lambda v: v,
    description="JSON scalar type",
)


# ==============================================================================
# Input Types
# ==============================================================================

@strawberry.input
class PaginationInput:
    page: int = 1
    page_size: int = 20
    offset: Optional[int] = None
    limit: Optional[int] = None


@strawberry.input
class SortInput:
    field: str
    order: SortOrder = SortOrder.ASC


@strawberry.input
class FilterInput:
    field: str
    operator: FilterOperator
    value: JSON


@strawberry.input
class AgentFilterInput:
    name: Optional[str] = None
    version: Optional[str] = None
    status: Optional[AgentStatus] = None
    tags: Optional[List[str]] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None


@strawberry.input
class WorkflowFilterInput:
    name: Optional[str] = None
    version: Optional[str] = None
    tags: Optional[List[str]] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None


@strawberry.input
class ExecutionFilterInput:
    status: Optional[ExecutionStatus] = None
    workflow_id: Optional[strawberry.ID] = None
    agent_id: Optional[strawberry.ID] = None
    user_id: Optional[strawberry.ID] = None
    started_after: Optional[datetime] = None
    started_before: Optional[datetime] = None
    completed_after: Optional[datetime] = None
    completed_before: Optional[datetime] = None


@strawberry.input
class CreateAgentInput:
    name: str
    description: str
    version: str = "0.0.1"
    enabled: bool = True
    parameters: Optional[JSON] = None
    resource_paths: Optional[List[str]] = None
    log_level: str = "INFO"
    tags: Optional[List[str]] = None
    metadata: Optional[JSON] = None


@strawberry.input
class UpdateAgentInput:
    name: Optional[str] = None
    description: Optional[str] = None
    version: Optional[str] = None
    enabled: Optional[bool] = None
    parameters: Optional[JSON] = None
    resource_paths: Optional[List[str]] = None
    log_level: Optional[str] = None
    tags: Optional[List[str]] = None
    metadata: Optional[JSON] = None


@strawberry.input
class WorkflowStepInput:
    name: str
    agent_id: strawberry.ID
    description: Optional[str] = None
    input_mapping: Optional[JSON] = None
    output_key: Optional[str] = None
    condition: Optional[str] = None
    on_failure: str = "stop"
    retry_count: int = 0
    timeout: Optional[int] = None


@strawberry.input
class CreateWorkflowInput:
    name: str
    description: str
    version: str = "0.0.1"
    steps: List[WorkflowStepInput]
    output_mapping: Optional[JSON] = None
    metadata: Optional[JSON] = None
    tags: Optional[List[str]] = None


@strawberry.input
class UpdateWorkflowInput:
    name: Optional[str] = None
    description: Optional[str] = None
    version: Optional[str] = None
    steps: Optional[List[WorkflowStepInput]] = None
    output_mapping: Optional[JSON] = None
    metadata: Optional[JSON] = None
    tags: Optional[List[str]] = None


@strawberry.input
class ExecuteWorkflowInput:
    workflow_id: strawberry.ID
    input_data: JSON
    context: Optional[JSON] = None
    tags: Optional[List[str]] = None


@strawberry.input
class ExecuteSingleAgentInput:
    agent_id: strawberry.ID
    input_data: JSON
    context: Optional[JSON] = None
    tags: Optional[List[str]] = None


@strawberry.input
class PermissionInput:
    resource: str
    action: str
    scope: Optional[str] = None
    conditions: Optional[JSON] = None


@strawberry.input
class CreateRoleInput:
    name: str
    description: Optional[str] = None
    permissions: List[PermissionInput]
    parent_roles: Optional[List[str]] = None
    metadata: Optional[JSON] = None


@strawberry.input
class UpdateRoleInput:
    description: Optional[str] = None
    permissions: Optional[List[PermissionInput]] = None
    parent_roles: Optional[List[str]] = None
    metadata: Optional[JSON] = None


@strawberry.input
class AssignRoleInput:
    user_id: strawberry.ID
    role_names: List[str]


@strawberry.input
class CreateAPIKeyInput:
    name: str
    description: Optional[str] = None
    scopes: Optional[List[str]] = None
    expires_in: Optional[int] = None
    allowed_ips: Optional[List[str]] = None
    allowed_origins: Optional[List[str]] = None
    rate_limit: Optional[int] = None


@strawberry.input
class TimeRangeInput:
    start: datetime
    end: datetime


# ==============================================================================
# Object Types
# ==============================================================================

@strawberry.type
class PageInfo:
    has_next_page: bool
    has_previous_page: bool
    start_cursor: Optional[str]
    end_cursor: Optional[str]
    total_count: int
    total_pages: int
    current_page: int


@strawberry.type
class Permission:
    resource: str
    action: str
    scope: Optional[str]
    conditions: Optional[JSON]


@strawberry.type
class Role:
    name: str
    description: Optional[str]
    permissions: List[Permission]
    parent_roles: List[str]
    metadata: Optional[JSON]
    created_at: datetime
    updated_at: datetime


@strawberry.type
class User:
    id: strawberry.ID
    tenant_id: strawberry.ID
    username: str
    email: str
    active: bool
    roles: List[Role]
    permissions: List[Permission]
    created_at: datetime
    last_login: Optional[datetime]
    metadata: Optional[JSON]


@strawberry.type
class APIKey:
    key_id: strawberry.ID
    name: str
    description: Optional[str]
    display_key: str
    scopes: List[str]
    tenant_id: strawberry.ID
    created_at: datetime
    expires_at: Optional[datetime]
    last_used_at: Optional[datetime]
    use_count: int
    active: bool
    allowed_ips: List[str]
    allowed_origins: List[str]
    rate_limit: Optional[int]


@strawberry.type
class AgentMetrics:
    execution_time_ms: float
    input_size: int
    output_size: int
    records_processed: int
    cache_hits: int
    cache_misses: int
    custom_metrics: Optional[JSON]


@strawberry.type
class AgentStats:
    executions: int
    successes: int
    failures: int
    success_rate: float
    total_time_ms: float
    avg_time_ms: float
    custom_counters: Optional[JSON]
    custom_timers: Optional[JSON]


@strawberry.type
class Agent:
    id: strawberry.ID
    name: str
    description: str
    version: str
    enabled: bool
    parameters: Optional[JSON]
    resource_paths: List[str]
    log_level: str
    tags: List[str]
    metadata: Optional[JSON]
    stats: AgentStats
    created_at: datetime
    updated_at: datetime


@strawberry.type
class WorkflowStep:
    name: str
    agent_id: strawberry.ID
    description: Optional[str]
    input_mapping: Optional[JSON]
    output_key: Optional[str]
    condition: Optional[str]
    on_failure: str
    retry_count: int
    timeout: Optional[int]


@strawberry.type
class Workflow:
    id: strawberry.ID
    name: str
    description: str
    version: str
    steps: List[WorkflowStep]
    output_mapping: Optional[JSON]
    metadata: Optional[JSON]
    tags: List[str]
    created_at: datetime
    updated_at: datetime


@strawberry.type
class ExecutionStepResult:
    step_name: str
    agent_id: strawberry.ID
    status: WorkflowStepStatus
    result: Optional[JSON]
    error: Optional[str]
    metrics: Optional[AgentMetrics]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    duration: Optional[float]
    attempts: int


@strawberry.type
class Execution:
    id: strawberry.ID
    execution_id: str
    workflow_id: Optional[strawberry.ID]
    agent_id: Optional[strawberry.ID]
    user_id: Optional[strawberry.ID]
    status: ExecutionStatus
    input_data: JSON
    output_data: Optional[JSON]
    context: Optional[JSON]
    errors: List[JSON]
    tags: List[str]
    step_results: List[ExecutionStepResult]
    total_duration: Optional[float]
    started_at: datetime
    completed_at: Optional[datetime]
    metadata: Optional[JSON]
    created_at: datetime
    updated_at: datetime


@strawberry.type
class ExecutionResult:
    success: bool
    execution: Execution
    errors: List[str]


# ==============================================================================
# Connection Types
# ==============================================================================

@strawberry.type
class AgentEdge:
    node: Agent
    cursor: str


@strawberry.type
class AgentConnection:
    edges: List[AgentEdge]
    nodes: List[Agent]
    page_info: PageInfo
    total_count: int


@strawberry.type
class WorkflowEdge:
    node: Workflow
    cursor: str


@strawberry.type
class WorkflowConnection:
    edges: List[WorkflowEdge]
    nodes: List[Workflow]
    page_info: PageInfo
    total_count: int


@strawberry.type
class ExecutionEdge:
    node: Execution
    cursor: str


@strawberry.type
class ExecutionConnection:
    edges: List[ExecutionEdge]
    nodes: List[Execution]
    page_info: PageInfo
    total_count: int


@strawberry.type
class RoleEdge:
    node: Role
    cursor: str


@strawberry.type
class RoleConnection:
    edges: List[RoleEdge]
    nodes: List[Role]
    page_info: PageInfo
    total_count: int


# ==============================================================================
# System Types
# ==============================================================================

@strawberry.type
class HealthCheck:
    name: str
    status: str
    message: Optional[str]
    latency: Optional[float]


@strawberry.type
class SystemHealth:
    status: str
    version: str
    uptime: float
    agent_count: int
    workflow_count: int
    execution_count: int
    timestamp: datetime
    checks: List[HealthCheck]


@strawberry.type
class Metric:
    name: str
    value: float
    unit: Optional[str]
    timestamp: datetime
    labels: Optional[JSON]


# ==============================================================================
# Subscription Types
# ==============================================================================

@strawberry.type
class ExecutionUpdate:
    event: SubscriptionEvent
    execution: Execution
    timestamp: datetime


@strawberry.type
class ExecutionStatusUpdate:
    execution_id: strawberry.ID
    old_status: Optional[ExecutionStatus]
    new_status: ExecutionStatus
    timestamp: datetime


@strawberry.type
class ExecutionProgress:
    execution_id: strawberry.ID
    current_step: Optional[str]
    completed_steps: int
    total_steps: int
    progress: float
    estimated_time_remaining: Optional[float]
    timestamp: datetime


@strawberry.type
class AgentUpdate:
    event: SubscriptionEvent
    agent: Agent
    timestamp: datetime


@strawberry.type
class WorkflowUpdate:
    event: SubscriptionEvent
    workflow: Workflow
    timestamp: datetime


@strawberry.type
class SystemMetrics:
    cpu_usage: float
    memory_usage: float
    active_executions: int
    requests_per_second: float
    timestamp: datetime
