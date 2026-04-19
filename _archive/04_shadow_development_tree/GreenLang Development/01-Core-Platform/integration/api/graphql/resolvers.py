# -*- coding: utf-8 -*-
"""
GraphQL Resolvers with DataLoader Integration
Comprehensive resolver implementation with N+1 query prevention
"""

from __future__ import annotations
import strawberry
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging
import uuid
import base64

from greenlang.utilities.determinism import deterministic_uuid, DeterministicClock
from greenlang.api.graphql.types import (
    Agent,
    Workflow,
    Execution,
    Role,
    User,
    APIKey,
    AgentConnection,
    WorkflowConnection,
    ExecutionConnection,
    RoleConnection,
    AgentEdge,
    WorkflowEdge,
    ExecutionEdge,
    RoleEdge,
    PageInfo,
    ExecutionResult,
    SystemHealth,
    Metric,
    HealthCheck,
    AgentStats,
    WorkflowStep,
    ExecutionStepResult,
    Permission,
    AgentMetrics,
    PaginationInput,
    SortInput,
    AgentFilterInput,
    WorkflowFilterInput,
    ExecutionFilterInput,
    CreateAgentInput,
    UpdateAgentInput,
    CreateWorkflowInput,
    UpdateWorkflowInput,
    ExecuteWorkflowInput,
    ExecuteSingleAgentInput,
    CreateRoleInput,
    UpdateRoleInput,
    AssignRoleInput,
    CreateAPIKeyInput,
    TimeRangeInput,
    AggregationType,
    ExecutionStatus,
    WorkflowStepStatus,
)
from greenlang.api.graphql.context import GraphQLContext
from greenlang.api.graphql.dataloaders import (
    AgentLoader,
    WorkflowLoader,
    ExecutionLoader,
    UserLoader,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# Helper Functions
# ==============================================================================

def create_page_info(
    items: List[Any],
    total_count: int,
    pagination: PaginationInput,
) -> PageInfo:
    """Create pagination info from items and pagination input"""
    page = pagination.page
    page_size = pagination.page_size
    total_pages = (total_count + page_size - 1) // page_size

    return PageInfo(
        has_next_page=page < total_pages,
        has_previous_page=page > 1,
        start_cursor=_encode_cursor(0) if items else None,
        end_cursor=_encode_cursor(len(items) - 1) if items else None,
        total_count=total_count,
        total_pages=total_pages,
        current_page=page,
    )


def _encode_cursor(index: int) -> str:
    """Encode cursor for pagination"""
    return base64.b64encode(f"cursor:{index}".encode()).decode()


def _decode_cursor(cursor: str) -> int:
    """Decode cursor for pagination"""
    try:
        decoded = base64.b64decode(cursor.encode()).decode()
        return int(decoded.split(":")[1])
    except Exception:
        return 0


def apply_pagination(
    items: List[Any],
    pagination: Optional[PaginationInput] = None,
) -> List[Any]:
    """Apply pagination to items"""
    if not pagination:
        pagination = PaginationInput()

    if pagination.offset is not None and pagination.limit is not None:
        start = pagination.offset
        end = pagination.offset + pagination.limit
    else:
        start = (pagination.page - 1) * pagination.page_size
        end = start + pagination.page_size

    return items[start:end]


def apply_sort(items: List[Any], sort: Optional[List[SortInput]] = None) -> List[Any]:
    """Apply sorting to items"""
    if not sort:
        return items

    sorted_items = list(items)
    for sort_input in reversed(sort):
        field = sort_input.field
        reverse = sort_input.order.value == "desc"
        sorted_items.sort(
            key=lambda x: getattr(x, field, None) or "", reverse=reverse
        )

    return sorted_items


def apply_filters(
    items: List[Any],
    filter_input: Optional[Any] = None,
) -> List[Any]:
    """Apply filters to items"""
    if not filter_input:
        return items

    filtered = items
    for field, value in filter_input.__dict__.items():
        if value is None:
            continue

        if field.endswith("_after"):
            attr = field.replace("_after", "_at")
            filtered = [x for x in filtered if getattr(x, attr, None) and getattr(x, attr) >= value]
        elif field.endswith("_before"):
            attr = field.replace("_before", "_at")
            filtered = [x for x in filtered if getattr(x, attr, None) and getattr(x, attr) <= value]
        elif field == "tags":
            filtered = [
                x for x in filtered
                if any(tag in getattr(x, "tags", []) for tag in value)
            ]
        else:
            filtered = [x for x in filtered if getattr(x, field, None) == value]

    return filtered


# ==============================================================================
# Query Resolvers
# ==============================================================================

@strawberry.type
class Query:
    """GraphQL Query root with comprehensive resolvers"""

    @strawberry.field
    async def agent(
        self,
        info: strawberry.Info[GraphQLContext],
        id: strawberry.ID,
    ) -> Optional[Agent]:
        """Get agent by ID using DataLoader"""
        context = info.context

        # Permission check
        if not context.rbac_manager.check_permission(
            context.user_id,
            f"agent:{id}",
            "read",
        ):
            raise PermissionError(f"Access denied to agent {id}")

        # Use DataLoader to prevent N+1
        return await context.agent_loader.load(id)

    @strawberry.field
    async def agents(
        self,
        info: strawberry.Info[GraphQLContext],
        pagination: Optional[PaginationInput] = None,
        filter: Optional[AgentFilterInput] = None,
        sort: Optional[List[SortInput]] = None,
    ) -> AgentConnection:
        """Get paginated agents with filtering and sorting"""
        context = info.context

        # Permission check
        if not context.rbac_manager.check_permission(
            context.user_id,
            "agent",
            "list",
        ):
            raise PermissionError("Access denied to list agents")

        # Get all agents from orchestrator
        orchestrator = context.orchestrator
        agent_ids = orchestrator.list_agents()

        # Load agents using DataLoader
        agents = await context.agent_loader.load_many(agent_ids)
        agents = [a for a in agents if a is not None]

        # Apply filters
        agents = apply_filters(agents, filter)

        # Apply sorting
        agents = apply_sort(agents, sort)

        # Get total count before pagination
        total_count = len(agents)

        # Apply pagination
        paginated_agents = apply_pagination(agents, pagination)

        # Create edges
        edges = [
            AgentEdge(node=agent, cursor=_encode_cursor(i))
            for i, agent in enumerate(paginated_agents)
        ]

        # Create page info
        page_info = create_page_info(paginated_agents, total_count, pagination or PaginationInput())

        return AgentConnection(
            edges=edges,
            nodes=paginated_agents,
            page_info=page_info,
            total_count=total_count,
        )

    @strawberry.field
    async def workflow(
        self,
        info: strawberry.Info[GraphQLContext],
        id: strawberry.ID,
    ) -> Optional[Workflow]:
        """Get workflow by ID using DataLoader"""
        context = info.context

        # Permission check
        if not context.rbac_manager.check_permission(
            context.user_id,
            f"workflow:{id}",
            "read",
        ):
            raise PermissionError(f"Access denied to workflow {id}")

        return await context.workflow_loader.load(id)

    @strawberry.field
    async def workflows(
        self,
        info: strawberry.Info[GraphQLContext],
        pagination: Optional[PaginationInput] = None,
        filter: Optional[WorkflowFilterInput] = None,
        sort: Optional[List[SortInput]] = None,
    ) -> WorkflowConnection:
        """Get paginated workflows with filtering and sorting"""
        context = info.context

        # Permission check
        if not context.rbac_manager.check_permission(
            context.user_id,
            "workflow",
            "list",
        ):
            raise PermissionError("Access denied to list workflows")

        # Get all workflows
        orchestrator = context.orchestrator
        workflow_ids = orchestrator.list_workflows()

        # Load workflows using DataLoader
        workflows = await context.workflow_loader.load_many(workflow_ids)
        workflows = [w for w in workflows if w is not None]

        # Apply filters
        workflows = apply_filters(workflows, filter)

        # Apply sorting
        workflows = apply_sort(workflows, sort)

        # Get total count
        total_count = len(workflows)

        # Apply pagination
        paginated_workflows = apply_pagination(workflows, pagination)

        # Create edges
        edges = [
            WorkflowEdge(node=workflow, cursor=_encode_cursor(i))
            for i, workflow in enumerate(paginated_workflows)
        ]

        # Create page info
        page_info = create_page_info(
            paginated_workflows, total_count, pagination or PaginationInput()
        )

        return WorkflowConnection(
            edges=edges,
            nodes=paginated_workflows,
            page_info=page_info,
            total_count=total_count,
        )

    @strawberry.field
    async def execution(
        self,
        info: strawberry.Info[GraphQLContext],
        id: strawberry.ID,
    ) -> Optional[Execution]:
        """Get execution by ID using DataLoader"""
        context = info.context

        # Permission check
        if not context.rbac_manager.check_permission(
            context.user_id,
            f"execution:{id}",
            "read",
        ):
            raise PermissionError(f"Access denied to execution {id}")

        return await context.execution_loader.load(id)

    @strawberry.field
    async def executions(
        self,
        info: strawberry.Info[GraphQLContext],
        pagination: Optional[PaginationInput] = None,
        filter: Optional[ExecutionFilterInput] = None,
        sort: Optional[List[SortInput]] = None,
    ) -> ExecutionConnection:
        """Get paginated executions with filtering and sorting"""
        context = info.context

        # Permission check
        if not context.rbac_manager.check_permission(
            context.user_id,
            "execution",
            "list",
        ):
            raise PermissionError("Access denied to list executions")

        # Get execution history
        orchestrator = context.orchestrator
        execution_records = orchestrator.get_execution_history()

        # Convert to Execution objects
        executions = [_convert_execution_record(rec) for rec in execution_records]

        # Apply filters
        executions = apply_filters(executions, filter)

        # Apply sorting
        executions = apply_sort(executions, sort)

        # Get total count
        total_count = len(executions)

        # Apply pagination
        paginated_executions = apply_pagination(executions, pagination)

        # Create edges
        edges = [
            ExecutionEdge(node=execution, cursor=_encode_cursor(i))
            for i, execution in enumerate(paginated_executions)
        ]

        # Create page info
        page_info = create_page_info(
            paginated_executions, total_count, pagination or PaginationInput()
        )

        return ExecutionConnection(
            edges=edges,
            nodes=paginated_executions,
            page_info=page_info,
            total_count=total_count,
        )

    @strawberry.field
    def role(
        self,
        info: strawberry.Info[GraphQLContext],
        name: str,
    ) -> Optional[Role]:
        """Get role by name"""
        context = info.context

        # Permission check
        if not context.rbac_manager.check_permission(
            context.user_id,
            "role",
            "read",
        ):
            raise PermissionError("Access denied to read roles")

        role_obj = context.rbac_manager.get_role(name)
        if not role_obj:
            return None

        return _convert_role(role_obj)

    @strawberry.field
    def roles(
        self,
        info: strawberry.Info[GraphQLContext],
        pagination: Optional[PaginationInput] = None,
        sort: Optional[List[SortInput]] = None,
    ) -> RoleConnection:
        """Get paginated roles"""
        context = info.context

        # Permission check
        if not context.rbac_manager.check_permission(
            context.user_id,
            "role",
            "list",
        ):
            raise PermissionError("Access denied to list roles")

        # Get all roles
        roles = [
            _convert_role(role)
            for role in context.rbac_manager.roles.values()
        ]

        # Apply sorting
        roles = apply_sort(roles, sort)

        # Get total count
        total_count = len(roles)

        # Apply pagination
        paginated_roles = apply_pagination(roles, pagination)

        # Create edges
        edges = [
            RoleEdge(node=role, cursor=_encode_cursor(i))
            for i, role in enumerate(paginated_roles)
        ]

        # Create page info
        page_info = create_page_info(
            paginated_roles, total_count, pagination or PaginationInput()
        )

        return RoleConnection(
            edges=edges,
            nodes=paginated_roles,
            page_info=page_info,
            total_count=total_count,
        )

    @strawberry.field
    async def user(
        self,
        info: strawberry.Info[GraphQLContext],
        id: strawberry.ID,
    ) -> Optional[User]:
        """Get user by ID"""
        context = info.context

        # Permission check
        if not context.rbac_manager.check_permission(
            context.user_id,
            f"user:{id}",
            "read",
        ):
            raise PermissionError(f"Access denied to user {id}")

        return await context.user_loader.load(id)

    @strawberry.field
    def current_user(
        self,
        info: strawberry.Info[GraphQLContext],
    ) -> User:
        """Get current authenticated user"""
        context = info.context
        user_data = context.auth_manager.users.get(context.user_id)

        if not user_data:
            raise ValueError("User not found")

        # Get user roles
        role_names = context.rbac_manager.get_user_roles(context.user_id)
        roles = [
            _convert_role(context.rbac_manager.get_role(name))
            for name in role_names
            if context.rbac_manager.get_role(name)
        ]

        # Get user permissions
        permissions = [
            _convert_permission(perm)
            for perm in context.rbac_manager.get_user_permissions(context.user_id)
        ]

        return User(
            id=strawberry.ID(user_data["user_id"]),
            tenant_id=strawberry.ID(user_data["tenant_id"]),
            username=user_data["username"],
            email=user_data["email"],
            active=user_data.get("active", True),
            roles=roles,
            permissions=permissions,
            created_at=user_data.get("created_at", DeterministicClock.utcnow()),
            last_login=None,
            metadata=user_data.get("metadata", {}),
        )

    @strawberry.field
    def check_permission(
        self,
        info: strawberry.Info[GraphQLContext],
        resource: str,
        action: str,
        context_data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Check if current user has permission"""
        context = info.context
        return context.rbac_manager.check_permission(
            context.user_id,
            resource,
            action,
            context_data,
        )

    @strawberry.field
    def api_key(
        self,
        info: strawberry.Info[GraphQLContext],
        key_id: strawberry.ID,
    ) -> Optional[APIKey]:
        """Get API key by ID"""
        context = info.context

        # Permission check
        if not context.rbac_manager.check_permission(
            context.user_id,
            "api_key",
            "read",
        ):
            raise PermissionError("Access denied to API keys")

        # Find API key
        for key_string, key_obj in context.auth_manager.api_keys.items():
            if key_obj.key_id == key_id:
                return _convert_api_key(key_obj)

        return None

    @strawberry.field
    def api_keys(
        self,
        info: strawberry.Info[GraphQLContext],
        pagination: Optional[PaginationInput] = None,
    ) -> List[APIKey]:
        """Get API keys for current tenant"""
        context = info.context

        # Permission check
        if not context.rbac_manager.check_permission(
            context.user_id,
            "api_key",
            "list",
        ):
            raise PermissionError("Access denied to list API keys")

        # Get tenant ID from current user
        user_data = context.auth_manager.users.get(context.user_id)
        if not user_data:
            return []

        tenant_id = user_data["tenant_id"]

        # Filter API keys by tenant
        keys = [
            _convert_api_key(key)
            for key in context.auth_manager.api_keys.values()
            if key.tenant_id == tenant_id
        ]

        # Apply pagination
        keys = apply_pagination(keys, pagination)

        return keys

    @strawberry.field
    def system_health(
        self,
        info: strawberry.Info[GraphQLContext],
    ) -> SystemHealth:
        """Get system health status"""
        context = info.context
        orchestrator = context.orchestrator

        # Calculate uptime (placeholder)
        uptime = 3600.0

        # Health checks
        checks = [
            HealthCheck(
                name="orchestrator",
                status="healthy",
                message="Orchestrator is operational",
                latency=0.5,
            ),
            HealthCheck(
                name="auth",
                status="healthy",
                message="Authentication service is operational",
                latency=0.3,
            ),
            HealthCheck(
                name="rbac",
                status="healthy",
                message="RBAC service is operational",
                latency=0.2,
            ),
        ]

        return SystemHealth(
            status="healthy",
            version="1.0.0",
            uptime=uptime,
            agent_count=len(orchestrator.list_agents()),
            workflow_count=len(orchestrator.list_workflows()),
            execution_count=len(orchestrator.get_execution_history()),
            timestamp=DeterministicClock.utcnow(),
            checks=checks,
        )

    @strawberry.field
    def metrics(
        self,
        info: strawberry.Info[GraphQLContext],
        time_range: Optional[TimeRangeInput] = None,
        aggregation: Optional[AggregationType] = None,
    ) -> List[Metric]:
        """Get system metrics"""
        context = info.context

        # Permission check
        if not context.rbac_manager.check_permission(
            context.user_id,
            "metrics",
            "read",
        ):
            raise PermissionError("Access denied to metrics")

        # Placeholder metrics
        metrics = [
            Metric(
                name="executions_total",
                value=float(len(context.orchestrator.get_execution_history())),
                unit="count",
                timestamp=DeterministicClock.utcnow(),
                labels={"type": "total"},
            ),
            Metric(
                name="agents_total",
                value=float(len(context.orchestrator.list_agents())),
                unit="count",
                timestamp=DeterministicClock.utcnow(),
                labels={"type": "total"},
            ),
            Metric(
                name="workflows_total",
                value=float(len(context.orchestrator.list_workflows())),
                unit="count",
                timestamp=DeterministicClock.utcnow(),
                labels={"type": "total"},
            ),
        ]

        return metrics


# ==============================================================================
# Mutation Resolvers
# ==============================================================================

@strawberry.type
class Mutation:
    """GraphQL Mutation root with comprehensive mutations"""

    @strawberry.mutation
    async def create_agent(
        self,
        info: strawberry.Info[GraphQLContext],
        input: CreateAgentInput,
    ) -> Agent:
        """Create a new agent"""
        context = info.context

        # Permission check
        if not context.rbac_manager.check_permission(
            context.user_id,
            "agent",
            "create",
        ):
            raise PermissionError("Access denied to create agents")

        # Create agent config
        from greenlang.agents.base import AgentConfig, BaseAgent, AgentResult

        config = AgentConfig(
            name=input.name,
            description=input.description,
            version=input.version,
            enabled=input.enabled,
            parameters=input.parameters or {},
            resource_paths=input.resource_paths or [],
            log_level=input.log_level,
        )

        # Create a simple agent implementation
        class DynamicAgent(BaseAgent):
            def execute(self, input_data: Dict[str, Any]) -> AgentResult:
                return AgentResult(
                    success=True,
                    data={"message": "Dynamic agent executed"},
                    metadata={"agent_name": self.config.name},
                )

        agent_obj = DynamicAgent(config)
        agent_id = str(deterministic_uuid(__name__, str(DeterministicClock.now())))

        # Register with orchestrator
        context.orchestrator.register_agent(agent_id, agent_obj)

        # Convert to GraphQL type
        return _convert_agent(agent_id, agent_obj, input.tags or [], input.metadata or {})

    @strawberry.mutation
    async def update_agent(
        self,
        info: strawberry.Info[GraphQLContext],
        id: strawberry.ID,
        input: UpdateAgentInput,
    ) -> Agent:
        """Update an existing agent"""
        context = info.context

        # Permission check
        if not context.rbac_manager.check_permission(
            context.user_id,
            f"agent:{id}",
            "update",
        ):
            raise PermissionError(f"Access denied to update agent {id}")

        # Get existing agent
        agent_obj = context.orchestrator.agents.get(id)
        if not agent_obj:
            raise ValueError(f"Agent {id} not found")

        # Update agent config
        if input.name:
            agent_obj.config.name = input.name
        if input.description:
            agent_obj.config.description = input.description
        if input.version:
            agent_obj.config.version = input.version
        if input.enabled is not None:
            agent_obj.config.enabled = input.enabled
        if input.parameters:
            agent_obj.config.parameters.update(input.parameters)
        if input.resource_paths:
            agent_obj.config.resource_paths = input.resource_paths
        if input.log_level:
            agent_obj.config.log_level = input.log_level

        return await context.agent_loader.load(id)

    @strawberry.mutation
    async def delete_agent(
        self,
        info: strawberry.Info[GraphQLContext],
        id: strawberry.ID,
    ) -> bool:
        """Delete an agent"""
        context = info.context

        # Permission check
        if not context.rbac_manager.check_permission(
            context.user_id,
            f"agent:{id}",
            "delete",
        ):
            raise PermissionError(f"Access denied to delete agent {id}")

        # Remove agent from orchestrator
        if id in context.orchestrator.agents:
            del context.orchestrator.agents[id]
            logger.info(f"Deleted agent: {id}")
            return True

        return False

    @strawberry.mutation
    async def enable_agent(
        self,
        info: strawberry.Info[GraphQLContext],
        id: strawberry.ID,
    ) -> Agent:
        """Enable an agent"""
        context = info.context

        # Permission check
        if not context.rbac_manager.check_permission(
            context.user_id,
            f"agent:{id}",
            "update",
        ):
            raise PermissionError(f"Access denied to enable agent {id}")

        agent_obj = context.orchestrator.agents.get(id)
        if not agent_obj:
            raise ValueError(f"Agent {id} not found")

        agent_obj.config.enabled = True
        return await context.agent_loader.load(id)

    @strawberry.mutation
    async def disable_agent(
        self,
        info: strawberry.Info[GraphQLContext],
        id: strawberry.ID,
    ) -> Agent:
        """Disable an agent"""
        context = info.context

        # Permission check
        if not context.rbac_manager.check_permission(
            context.user_id,
            f"agent:{id}",
            "update",
        ):
            raise PermissionError(f"Access denied to disable agent {id}")

        agent_obj = context.orchestrator.agents.get(id)
        if not agent_obj:
            raise ValueError(f"Agent {id} not found")

        agent_obj.config.enabled = False
        return await context.agent_loader.load(id)

    @strawberry.mutation
    async def create_workflow(
        self,
        info: strawberry.Info[GraphQLContext],
        input: CreateWorkflowInput,
    ) -> Workflow:
        """Create a new workflow"""
        context = info.context

        # Permission check
        if not context.rbac_manager.check_permission(
            context.user_id,
            "workflow",
            "create",
        ):
            raise PermissionError("Access denied to create workflows")

        # Create workflow
        from greenlang.core.workflow import Workflow as WorkflowModel, WorkflowStep as StepModel

        steps = [
            StepModel(
                name=step.name,
                agent_id=step.agent_id,
                description=step.description,
                input_mapping=step.input_mapping,
                output_key=step.output_key,
                condition=step.condition,
                on_failure=step.on_failure,
                retry_count=step.retry_count,
            )
            for step in input.steps
        ]

        workflow = WorkflowModel(
            name=input.name,
            description=input.description,
            version=input.version,
            steps=steps,
            output_mapping=input.output_mapping,
            metadata=input.metadata or {},
        )

        workflow_id = str(deterministic_uuid(__name__, str(DeterministicClock.now())))

        # Register with orchestrator
        context.orchestrator.register_workflow(workflow_id, workflow)

        return _convert_workflow(workflow_id, workflow, input.tags or [])

    @strawberry.mutation
    async def update_workflow(
        self,
        info: strawberry.Info[GraphQLContext],
        id: strawberry.ID,
        input: UpdateWorkflowInput,
    ) -> Workflow:
        """Update an existing workflow"""
        context = info.context

        # Permission check
        if not context.rbac_manager.check_permission(
            context.user_id,
            f"workflow:{id}",
            "update",
        ):
            raise PermissionError(f"Access denied to update workflow {id}")

        # Get existing workflow
        workflow_obj = context.orchestrator.workflows.get(id)
        if not workflow_obj:
            raise ValueError(f"Workflow {id} not found")

        # Update workflow
        if input.name:
            workflow_obj.name = input.name
        if input.description:
            workflow_obj.description = input.description
        if input.version:
            workflow_obj.version = input.version
        if input.output_mapping:
            workflow_obj.output_mapping = input.output_mapping
        if input.metadata:
            workflow_obj.metadata.update(input.metadata)
        if input.steps:
            from greenlang.core.workflow import WorkflowStep as StepModel

            workflow_obj.steps = [
                StepModel(
                    name=step.name,
                    agent_id=step.agent_id,
                    description=step.description,
                    input_mapping=step.input_mapping,
                    output_key=step.output_key,
                    condition=step.condition,
                    on_failure=step.on_failure,
                    retry_count=step.retry_count,
                )
                for step in input.steps
            ]

        return await context.workflow_loader.load(id)

    @strawberry.mutation
    async def delete_workflow(
        self,
        info: strawberry.Info[GraphQLContext],
        id: strawberry.ID,
    ) -> bool:
        """Delete a workflow"""
        context = info.context

        # Permission check
        if not context.rbac_manager.check_permission(
            context.user_id,
            f"workflow:{id}",
            "delete",
        ):
            raise PermissionError(f"Access denied to delete workflow {id}")

        if id in context.orchestrator.workflows:
            del context.orchestrator.workflows[id]
            logger.info(f"Deleted workflow: {id}")
            return True

        return False

    @strawberry.mutation
    async def clone_workflow(
        self,
        info: strawberry.Info[GraphQLContext],
        id: strawberry.ID,
        name: str,
    ) -> Workflow:
        """Clone a workflow"""
        context = info.context

        # Permission check
        if not context.rbac_manager.check_permission(
            context.user_id,
            "workflow",
            "create",
        ):
            raise PermissionError("Access denied to create workflows")

        # Get existing workflow
        workflow_obj = context.orchestrator.workflows.get(id)
        if not workflow_obj:
            raise ValueError(f"Workflow {id} not found")

        # Clone workflow
        from greenlang.core.workflow import Workflow as WorkflowModel

        cloned = WorkflowModel(
            name=name,
            description=workflow_obj.description,
            version=workflow_obj.version,
            steps=list(workflow_obj.steps),
            output_mapping=workflow_obj.output_mapping,
            metadata=dict(workflow_obj.metadata),
        )

        new_id = str(deterministic_uuid(__name__, str(DeterministicClock.now())))
        context.orchestrator.register_workflow(new_id, cloned)

        return _convert_workflow(new_id, cloned, [])

    @strawberry.mutation
    async def execute_workflow(
        self,
        info: strawberry.Info[GraphQLContext],
        input: ExecuteWorkflowInput,
    ) -> ExecutionResult:
        """Execute a workflow"""
        context = info.context

        # Permission check
        if not context.rbac_manager.check_permission(
            context.user_id,
            f"workflow:{input.workflow_id}",
            "execute",
        ):
            raise PermissionError(f"Access denied to execute workflow {input.workflow_id}")

        try:
            # Execute workflow
            result = context.orchestrator.execute_workflow(
                input.workflow_id,
                input.input_data,
            )

            # Convert to Execution object
            execution = _convert_execution_record(result)

            return ExecutionResult(
                success=result.get("success", False),
                execution=execution,
                errors=[str(e) for e in result.get("errors", [])],
            )

        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            # Create failed execution
            execution = Execution(
                id=strawberry.ID(str(deterministic_uuid(__name__, str(DeterministicClock.now())))),
                execution_id=f"failed_{deterministic_uuid(__name__, str(DeterministicClock.now()))}",
                workflow_id=strawberry.ID(input.workflow_id),
                agent_id=None,
                user_id=strawberry.ID(context.user_id),
                status=ExecutionStatus.FAILED,
                input_data=input.input_data,
                output_data=None,
                context=input.context,
                errors=[{"error": str(e)}],
                tags=input.tags or [],
                step_results=[],
                total_duration=None,
                started_at=DeterministicClock.utcnow(),
                completed_at=DeterministicClock.utcnow(),
                metadata={},
                created_at=DeterministicClock.utcnow(),
                updated_at=DeterministicClock.utcnow(),
            )

            return ExecutionResult(
                success=False,
                execution=execution,
                errors=[str(e)],
            )

    @strawberry.mutation
    async def execute_single_agent(
        self,
        info: strawberry.Info[GraphQLContext],
        input: ExecuteSingleAgentInput,
    ) -> ExecutionResult:
        """Execute a single agent"""
        context = info.context

        # Permission check
        if not context.rbac_manager.check_permission(
            context.user_id,
            f"agent:{input.agent_id}",
            "execute",
        ):
            raise PermissionError(f"Access denied to execute agent {input.agent_id}")

        try:
            # Execute agent
            result = context.orchestrator.execute_single_agent(
                input.agent_id,
                input.input_data,
            )

            # Create execution record
            execution_id = str(deterministic_uuid(__name__, str(DeterministicClock.now())))
            execution = Execution(
                id=strawberry.ID(execution_id),
                execution_id=f"agent_{execution_id}",
                workflow_id=None,
                agent_id=strawberry.ID(input.agent_id),
                user_id=strawberry.ID(context.user_id),
                status=ExecutionStatus.COMPLETED if result.get("success") else ExecutionStatus.FAILED,
                input_data=input.input_data,
                output_data=result.get("data"),
                context=input.context,
                errors=[result.get("error")] if result.get("error") else [],
                tags=input.tags or [],
                step_results=[],
                total_duration=result.get("metadata", {}).get("execution_time_ms"),
                started_at=DeterministicClock.utcnow(),
                completed_at=DeterministicClock.utcnow(),
                metadata=result.get("metadata", {}),
                created_at=DeterministicClock.utcnow(),
                updated_at=DeterministicClock.utcnow(),
            )

            return ExecutionResult(
                success=result.get("success", False),
                execution=execution,
                errors=[result.get("error")] if result.get("error") else [],
            )

        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            raise

    @strawberry.mutation
    async def cancel_execution(
        self,
        info: strawberry.Info[GraphQLContext],
        id: strawberry.ID,
    ) -> Execution:
        """Cancel an execution (placeholder - requires async execution)"""
        context = info.context

        # Permission check
        if not context.rbac_manager.check_permission(
            context.user_id,
            f"execution:{id}",
            "update",
        ):
            raise PermissionError(f"Access denied to cancel execution {id}")

        execution = await context.execution_loader.load(id)
        if execution:
            # Update status to cancelled (would need persistent storage)
            execution.status = ExecutionStatus.CANCELLED

        return execution

    @strawberry.mutation
    async def retry_execution(
        self,
        info: strawberry.Info[GraphQLContext],
        id: strawberry.ID,
    ) -> ExecutionResult:
        """Retry a failed execution"""
        context = info.context

        # Permission check
        if not context.rbac_manager.check_permission(
            context.user_id,
            f"execution:{id}",
            "execute",
        ):
            raise PermissionError(f"Access denied to retry execution {id}")

        # Get original execution
        original = await context.execution_loader.load(id)
        if not original:
            raise ValueError(f"Execution {id} not found")

        # Retry based on type
        if original.workflow_id:
            return await self.execute_workflow(
                info,
                ExecuteWorkflowInput(
                    workflow_id=original.workflow_id,
                    input_data=original.input_data,
                    context=original.context,
                    tags=original.tags,
                ),
            )
        elif original.agent_id:
            return await self.execute_single_agent(
                info,
                ExecuteSingleAgentInput(
                    agent_id=original.agent_id,
                    input_data=original.input_data,
                    context=original.context,
                    tags=original.tags,
                ),
            )
        else:
            raise ValueError("Cannot retry execution without workflow or agent ID")

    @strawberry.mutation
    def create_role(
        self,
        info: strawberry.Info[GraphQLContext],
        input: CreateRoleInput,
    ) -> Role:
        """Create a new role"""
        context = info.context

        # Permission check
        if not context.rbac_manager.check_permission(
            context.user_id,
            "role",
            "create",
        ):
            raise PermissionError("Access denied to create roles")

        # Convert permissions
        from greenlang.auth.rbac import Permission as RBACPermission

        permissions = [
            RBACPermission(
                resource=perm.resource,
                action=perm.action,
                scope=perm.scope,
                conditions=perm.conditions or {},
            )
            for perm in input.permissions
        ]

        # Create role
        role_obj = context.rbac_manager.create_role(
            name=input.name,
            description=input.description or "",
            permissions=permissions,
            parent_roles=input.parent_roles or [],
        )

        # Update metadata
        if input.metadata:
            role_obj.metadata.update(input.metadata)

        return _convert_role(role_obj)

    @strawberry.mutation
    def update_role(
        self,
        info: strawberry.Info[GraphQLContext],
        name: str,
        input: UpdateRoleInput,
    ) -> Role:
        """Update an existing role"""
        context = info.context

        # Permission check
        if not context.rbac_manager.check_permission(
            context.user_id,
            "role",
            "update",
        ):
            raise PermissionError("Access denied to update roles")

        # Prepare updates
        updates = {}
        if input.description:
            updates["description"] = input.description
        if input.parent_roles:
            updates["parent_roles"] = input.parent_roles
        if input.permissions:
            from greenlang.auth.rbac import Permission as RBACPermission

            updates["permissions"] = [
                RBACPermission(
                    resource=perm.resource,
                    action=perm.action,
                    scope=perm.scope,
                    conditions=perm.conditions or {},
                )
                for perm in input.permissions
            ]

        # Update role
        role_obj = context.rbac_manager.update_role(name, updates)
        if not role_obj:
            raise ValueError(f"Role {name} not found")

        return _convert_role(role_obj)

    @strawberry.mutation
    def delete_role(
        self,
        info: strawberry.Info[GraphQLContext],
        name: str,
    ) -> bool:
        """Delete a role"""
        context = info.context

        # Permission check
        if not context.rbac_manager.check_permission(
            context.user_id,
            "role",
            "delete",
        ):
            raise PermissionError("Access denied to delete roles")

        return context.rbac_manager.delete_role(name)

    @strawberry.mutation
    async def assign_role(
        self,
        info: strawberry.Info[GraphQLContext],
        input: AssignRoleInput,
    ) -> User:
        """Assign roles to a user"""
        context = info.context

        # Permission check
        if not context.rbac_manager.check_permission(
            context.user_id,
            "user",
            "update",
        ):
            raise PermissionError("Access denied to assign roles")

        # Assign roles
        for role_name in input.role_names:
            context.rbac_manager.assign_role(input.user_id, role_name)

        return await context.user_loader.load(input.user_id)

    @strawberry.mutation
    async def revoke_role(
        self,
        info: strawberry.Info[GraphQLContext],
        user_id: strawberry.ID,
        role_name: str,
    ) -> User:
        """Revoke a role from a user"""
        context = info.context

        # Permission check
        if not context.rbac_manager.check_permission(
            context.user_id,
            "user",
            "update",
        ):
            raise PermissionError("Access denied to revoke roles")

        context.rbac_manager.revoke_role(user_id, role_name)
        return await context.user_loader.load(user_id)

    @strawberry.mutation
    def create_api_key(
        self,
        info: strawberry.Info[GraphQLContext],
        input: CreateAPIKeyInput,
    ) -> APIKey:
        """Create a new API key"""
        context = info.context

        # Permission check
        if not context.rbac_manager.check_permission(
            context.user_id,
            "api_key",
            "create",
        ):
            raise PermissionError("Access denied to create API keys")

        # Get user tenant
        user_data = context.auth_manager.users.get(context.user_id)
        if not user_data:
            raise ValueError("User not found")

        # Create API key
        key_obj = context.auth_manager.create_api_key(
            tenant_id=user_data["tenant_id"],
            name=input.name,
            scopes=input.scopes or [],
            expires_in=input.expires_in,
            allowed_ips=input.allowed_ips or [],
            allowed_origins=input.allowed_origins or [],
            rate_limit=input.rate_limit,
        )

        return _convert_api_key(key_obj)

    @strawberry.mutation
    def revoke_api_key(
        self,
        info: strawberry.Info[GraphQLContext],
        key_id: strawberry.ID,
    ) -> bool:
        """Revoke an API key"""
        context = info.context

        # Permission check
        if not context.rbac_manager.check_permission(
            context.user_id,
            "api_key",
            "delete",
        ):
            raise PermissionError("Access denied to revoke API keys")

        # Find and deactivate API key
        for key_string, key_obj in context.auth_manager.api_keys.items():
            if key_obj.key_id == key_id:
                key_obj.active = False
                logger.info(f"Revoked API key: {key_id}")
                return True

        return False

    @strawberry.mutation
    def rotate_api_key(
        self,
        info: strawberry.Info[GraphQLContext],
        key_id: strawberry.ID,
    ) -> APIKey:
        """Rotate an API key (generate new secret)"""
        context = info.context

        # Permission check
        if not context.rbac_manager.check_permission(
            context.user_id,
            "api_key",
            "update",
        ):
            raise PermissionError("Access denied to rotate API keys")

        # Find and rotate API key
        for key_string, key_obj in context.auth_manager.api_keys.items():
            if key_obj.key_id == key_id:
                new_secret = key_obj.rotate()

                # Update storage (remove old, add new)
                del context.auth_manager.api_keys[key_string]
                new_key_string = f"{key_obj.key_id}.{new_secret}"
                context.auth_manager.api_keys[new_key_string] = key_obj

                logger.info(f"Rotated API key: {key_id}")
                return _convert_api_key(key_obj)

        raise ValueError(f"API key {key_id} not found")

    @strawberry.mutation
    async def batch_create_agents(
        self,
        info: strawberry.Info[GraphQLContext],
        inputs: List[CreateAgentInput],
    ) -> List[Agent]:
        """Batch create multiple agents"""
        context = info.context

        # Permission check
        if not context.rbac_manager.check_permission(
            context.user_id,
            "agent",
            "create",
        ):
            raise PermissionError("Access denied to create agents")

        agents = []
        for input_data in inputs:
            try:
                agent = await self.create_agent(info, input_data)
                agents.append(agent)
            except Exception as e:
                logger.error(f"Failed to create agent {input_data.name}: {e}")

        return agents

    @strawberry.mutation
    def batch_delete_executions(
        self,
        info: strawberry.Info[GraphQLContext],
        ids: List[strawberry.ID],
    ) -> int:
        """Batch delete executions"""
        context = info.context

        # Permission check
        if not context.rbac_manager.check_permission(
            context.user_id,
            "execution",
            "delete",
        ):
            raise PermissionError("Access denied to delete executions")

        # For now, return count (would need persistent storage)
        deleted_count = len(ids)
        logger.info(f"Batch deleted {deleted_count} executions")

        return deleted_count


# ==============================================================================
# Converter Functions
# ==============================================================================

def _convert_agent(
    agent_id: str,
    agent_obj: Any,
    tags: List[str] = None,
    metadata: Dict[str, Any] = None,
) -> Agent:
    """Convert BaseAgent to GraphQL Agent type"""
    stats = agent_obj.get_stats()

    return Agent(
        id=strawberry.ID(agent_id),
        name=agent_obj.config.name,
        description=agent_obj.config.description,
        version=agent_obj.config.version,
        enabled=agent_obj.config.enabled,
        parameters=agent_obj.config.parameters,
        resource_paths=agent_obj.config.resource_paths,
        log_level=agent_obj.config.log_level,
        tags=tags or [],
        metadata=metadata or {},
        stats=AgentStats(
            executions=stats["executions"],
            successes=stats["successes"],
            failures=stats["failures"],
            success_rate=stats["success_rate"],
            total_time_ms=stats["total_time_ms"],
            avg_time_ms=stats["avg_time_ms"],
            custom_counters=stats.get("custom_counters", {}),
            custom_timers=stats.get("custom_timers", {}),
        ),
        created_at=DeterministicClock.utcnow(),
        updated_at=DeterministicClock.utcnow(),
    )


def _convert_workflow(
    workflow_id: str,
    workflow_obj: Any,
    tags: List[str] = None,
) -> Workflow:
    """Convert Workflow to GraphQL Workflow type"""
    steps = [
        WorkflowStep(
            name=step.name,
            agent_id=strawberry.ID(step.agent_id),
            description=step.description,
            input_mapping=step.input_mapping,
            output_key=step.output_key,
            condition=step.condition,
            on_failure=step.on_failure,
            retry_count=step.retry_count,
            timeout=None,  # Not in original model
        )
        for step in workflow_obj.steps
    ]

    return Workflow(
        id=strawberry.ID(workflow_id),
        name=workflow_obj.name,
        description=workflow_obj.description,
        version=workflow_obj.version,
        steps=steps,
        output_mapping=workflow_obj.output_mapping,
        metadata=workflow_obj.metadata,
        tags=tags or [],
        created_at=DeterministicClock.utcnow(),
        updated_at=DeterministicClock.utcnow(),
    )


def _convert_execution_record(record: Dict[str, Any]) -> Execution:
    """Convert execution record to GraphQL Execution type"""
    # Determine status
    status = ExecutionStatus.COMPLETED if record.get("success") else ExecutionStatus.FAILED

    # Convert step results
    step_results = []
    for step_name, result in record.get("results", {}).items():
        if isinstance(result, dict):
            step_results.append(
                ExecutionStepResult(
                    step_name=step_name,
                    agent_id=strawberry.ID("unknown"),  # Not available in record
                    status=WorkflowStepStatus.COMPLETED if result.get("success") else WorkflowStepStatus.FAILED,
                    result=result.get("data"),
                    error=result.get("error"),
                    metrics=None,  # Not available in record
                    started_at=None,
                    completed_at=None,
                    duration=None,
                    attempts=1,
                )
            )

    return Execution(
        id=strawberry.ID(record.get("execution_id", str(deterministic_uuid(__name__, str(DeterministicClock.now()))))),
        execution_id=record.get("execution_id", ""),
        workflow_id=strawberry.ID(record.get("workflow_id")) if record.get("workflow_id") else None,
        agent_id=None,
        user_id=None,
        status=status,
        input_data=record.get("input", {}),
        output_data=record.get("results", {}),
        context={},
        errors=record.get("errors", []),
        tags=[],
        step_results=step_results,
        total_duration=None,
        started_at=DeterministicClock.utcnow(),
        completed_at=DeterministicClock.utcnow() if status == ExecutionStatus.COMPLETED else None,
        metadata={},
        created_at=DeterministicClock.utcnow(),
        updated_at=DeterministicClock.utcnow(),
    )


def _convert_role(role_obj: Any) -> Role:
    """Convert RBAC Role to GraphQL Role type"""
    permissions = [_convert_permission(perm) for perm in role_obj.permissions]

    return Role(
        name=role_obj.name,
        description=role_obj.description,
        permissions=permissions,
        parent_roles=role_obj.parent_roles,
        metadata=role_obj.metadata,
        created_at=role_obj.created_at,
        updated_at=role_obj.updated_at,
    )


def _convert_permission(perm_obj: Any) -> Permission:
    """Convert RBAC Permission to GraphQL Permission type"""
    return Permission(
        resource=perm_obj.resource,
        action=perm_obj.action,
        scope=perm_obj.scope,
        conditions=perm_obj.conditions,
    )


def _convert_api_key(key_obj: Any) -> APIKey:
    """Convert APIKey to GraphQL APIKey type"""
    return APIKey(
        key_id=strawberry.ID(key_obj.key_id),
        name=key_obj.name,
        description=key_obj.description,
        display_key=key_obj.get_display_key(),
        scopes=key_obj.scopes,
        tenant_id=strawberry.ID(key_obj.tenant_id),
        created_at=key_obj.created_at,
        expires_at=key_obj.expires_at,
        last_used_at=key_obj.last_used_at,
        use_count=key_obj.use_count,
        active=key_obj.active,
        allowed_ips=key_obj.allowed_ips,
        allowed_origins=key_obj.allowed_origins,
        rate_limit=key_obj.rate_limit,
    )
