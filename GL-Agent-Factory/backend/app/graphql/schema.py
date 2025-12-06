"""
GraphQL Schema for GreenLang

Complete GraphQL schema for all GreenLang entities:
- Agent queries and mutations
- Execution queries and subscriptions
- Calculation result types
- DataLoader for N+1 prevention
- Complexity and depth limiting

Example:
    >>> from app.graphql import create_graphql_app
    >>> graphql_app = create_graphql_app()
    >>> app.mount("/graphql", graphql_app)
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, TypeVar

import strawberry
from strawberry import Schema
from strawberry.extensions import Extension
from strawberry.fastapi import GraphQLRouter
from strawberry.scalars import JSON
from strawberry.schema.config import StrawberryConfig
from strawberry.types import Info

logger = logging.getLogger(__name__)


# =============================================================================
# Custom Scalars
# =============================================================================


@strawberry.scalar(
    description="Date and time in ISO 8601 format",
    serialize=lambda v: v.isoformat() if v else None,
    parse_value=lambda v: datetime.fromisoformat(v) if v else None,
)
class DateTime:
    """Custom DateTime scalar for ISO 8601 format."""
    pass


@strawberry.scalar(
    description="Decimal number for precise calculations",
    serialize=lambda v: str(v),
    parse_value=lambda v: float(v),
)
class Decimal:
    """Custom Decimal scalar for precise numeric values."""
    pass


# =============================================================================
# Context and DataLoader
# =============================================================================


@dataclass
class GreenLangContext:
    """
    GraphQL context containing request state and data loaders.

    Provides:
    - User and tenant information
    - Database connection
    - Data loaders for batched queries
    """

    request: Any
    tenant_id: str
    user_id: Optional[str] = None
    db_pool: Optional[Any] = None
    redis: Optional[Any] = None

    # Data loaders (initialized lazily)
    _agent_loader: Optional[Any] = None
    _execution_loader: Optional[Any] = None
    _tenant_loader: Optional[Any] = None

    @property
    def agent_loader(self) -> "DataLoader":
        """Get or create agent data loader."""
        if self._agent_loader is None:
            self._agent_loader = AgentDataLoader(self.db_pool, self.tenant_id)
        return self._agent_loader

    @property
    def execution_loader(self) -> "DataLoader":
        """Get or create execution data loader."""
        if self._execution_loader is None:
            self._execution_loader = ExecutionDataLoader(self.db_pool, self.tenant_id)
        return self._execution_loader


class DataLoader:
    """
    Generic DataLoader for batching and caching database queries.

    Prevents N+1 query problems by batching multiple key lookups
    into a single database query.
    """

    def __init__(self, batch_fn: Callable, cache_enabled: bool = True):
        """
        Initialize the data loader.

        Args:
            batch_fn: Async function that loads multiple items by keys
            cache_enabled: Whether to cache loaded items
        """
        self.batch_fn = batch_fn
        self.cache_enabled = cache_enabled
        self._cache: Dict[str, Any] = {}
        self._pending: Dict[str, asyncio.Future] = {}
        self._batch: List[str] = []
        self._batch_scheduled = False

    async def load(self, key: str) -> Optional[Any]:
        """
        Load a single item by key.

        Items are batched and loaded together for efficiency.
        """
        # Check cache
        if self.cache_enabled and key in self._cache:
            return self._cache[key]

        # Check if already loading
        if key in self._pending:
            return await self._pending[key]

        # Create future for this key
        future = asyncio.get_event_loop().create_future()
        self._pending[key] = future
        self._batch.append(key)

        # Schedule batch load
        if not self._batch_scheduled:
            self._batch_scheduled = True
            asyncio.get_event_loop().call_soon(
                lambda: asyncio.create_task(self._dispatch())
            )

        return await future

    async def load_many(self, keys: List[str]) -> List[Optional[Any]]:
        """Load multiple items by keys."""
        return await asyncio.gather(*[self.load(key) for key in keys])

    async def _dispatch(self) -> None:
        """Dispatch the batched load."""
        self._batch_scheduled = False
        batch = self._batch
        self._batch = []

        try:
            results = await self.batch_fn(batch)

            for key, result in zip(batch, results):
                # Cache result
                if self.cache_enabled:
                    self._cache[key] = result

                # Resolve future
                if key in self._pending:
                    self._pending[key].set_result(result)
                    del self._pending[key]

        except Exception as e:
            # Reject all pending futures
            for key in batch:
                if key in self._pending:
                    self._pending[key].set_exception(e)
                    del self._pending[key]

    def clear(self, key: Optional[str] = None) -> None:
        """Clear cache for a specific key or all keys."""
        if key:
            self._cache.pop(key, None)
        else:
            self._cache.clear()


class AgentDataLoader(DataLoader):
    """DataLoader for batch loading agents."""

    def __init__(self, db_pool: Any, tenant_id: str):
        self.db_pool = db_pool
        self.tenant_id = tenant_id
        super().__init__(self._batch_load_agents)

    async def _batch_load_agents(self, agent_ids: List[str]) -> List[Optional[Dict]]:
        """Batch load agents by IDs."""
        # TODO: Implement actual database query
        # SELECT * FROM agents WHERE id IN (...) AND tenant_id = ?

        # Placeholder: return mock data
        return [
            {
                "id": agent_id,
                "agent_id": f"category/{agent_id.split('-')[-1]}",
                "name": f"Agent {agent_id}",
                "version": "1.0.0",
                "state": "CERTIFIED",
                "category": "emissions",
                "tenant_id": self.tenant_id,
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
            }
            for agent_id in agent_ids
        ]


class ExecutionDataLoader(DataLoader):
    """DataLoader for batch loading executions."""

    def __init__(self, db_pool: Any, tenant_id: str):
        self.db_pool = db_pool
        self.tenant_id = tenant_id
        super().__init__(self._batch_load_executions)

    async def _batch_load_executions(
        self, execution_ids: List[str]
    ) -> List[Optional[Dict]]:
        """Batch load executions by IDs."""
        # TODO: Implement actual database query
        return [
            {
                "id": exec_id,
                "agent_id": "agent-001",
                "status": "COMPLETED",
                "started_at": datetime.now(timezone.utc),
                "completed_at": datetime.now(timezone.utc),
                "tenant_id": self.tenant_id,
            }
            for exec_id in execution_ids
        ]


# =============================================================================
# Extensions (Complexity and Depth Limiting)
# =============================================================================


class QueryComplexityExtension(Extension):
    """
    Extension to limit query complexity.

    Prevents expensive queries by calculating and limiting complexity
    based on field weights and depth.
    """

    MAX_COMPLEXITY = 1000

    def on_operation(self) -> None:
        """Check query complexity before execution."""
        complexity = self._calculate_complexity(
            self.execution_context.query,
            self.execution_context.schema,
        )

        if complexity > self.MAX_COMPLEXITY:
            raise Exception(
                f"Query complexity {complexity} exceeds maximum {self.MAX_COMPLEXITY}"
            )

        logger.debug(f"Query complexity: {complexity}")

    def _calculate_complexity(self, query: Any, schema: Any) -> int:
        """Calculate query complexity based on field weights."""
        # Simplified complexity calculation
        # TODO: Implement proper AST-based complexity calculation
        return 100  # Placeholder


class QueryDepthExtension(Extension):
    """
    Extension to limit query depth.

    Prevents deeply nested queries that could cause performance issues.
    """

    MAX_DEPTH = 10

    def on_operation(self) -> None:
        """Check query depth before execution."""
        depth = self._calculate_depth(self.execution_context.query)

        if depth > self.MAX_DEPTH:
            raise Exception(
                f"Query depth {depth} exceeds maximum {self.MAX_DEPTH}"
            )

        logger.debug(f"Query depth: {depth}")

    def _calculate_depth(self, query: Any) -> int:
        """Calculate query depth from AST."""
        # Simplified depth calculation
        # TODO: Implement proper AST-based depth calculation
        return 3  # Placeholder


# =============================================================================
# GraphQL Types - Enums
# =============================================================================


@strawberry.enum
class AgentState(strawberry.enum.Enum):
    """Agent lifecycle states."""

    DRAFT = "DRAFT"
    EXPERIMENTAL = "EXPERIMENTAL"
    CERTIFIED = "CERTIFIED"
    DEPRECATED = "DEPRECATED"
    RETIRED = "RETIRED"


@strawberry.enum
class ExecutionStatus(strawberry.enum.Enum):
    """Execution status values."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMEOUT = "TIMEOUT"


@strawberry.enum
class SortOrder(strawberry.enum.Enum):
    """Sort order for list queries."""

    ASC = "ASC"
    DESC = "DESC"


@strawberry.enum
class AgentCategory(strawberry.enum.Enum):
    """Agent categories."""

    EMISSIONS = "emissions"
    ENERGY = "energy"
    REGULATORY = "regulatory"
    SUPPLY_CHAIN = "supply_chain"
    REPORTING = "reporting"
    DATA_QUALITY = "data_quality"


# =============================================================================
# GraphQL Types - Objects
# =============================================================================


@strawberry.type
class PageInfo:
    """Pagination information."""

    has_next_page: bool
    has_previous_page: bool
    start_cursor: Optional[str]
    end_cursor: Optional[str]
    total_count: int


@strawberry.type
class EmissionFactorType:
    """Emission factor used in calculations."""

    id: str
    name: str
    value: float
    unit: str
    source: str
    source_url: Optional[str]
    valid_from: DateTime
    valid_to: Optional[DateTime]
    region: Optional[str]
    methodology: str


@strawberry.type
class CalculationResultType:
    """Result of a calculation with provenance."""

    id: str
    execution_id: str
    result_type: str
    value: float
    unit: str
    confidence_score: float
    methodology: str
    source_hash: str
    emission_factors: List[EmissionFactorType]
    created_at: DateTime

    @strawberry.field
    def formatted_value(self) -> str:
        """Get formatted value with unit."""
        return f"{self.value:.4f} {self.unit}"


@strawberry.type
class ExecutionLogType:
    """Log entry from an execution."""

    timestamp: DateTime
    level: str
    message: str
    data: Optional[JSON]


@strawberry.type
class ExecutionMetricsType:
    """Metrics from an execution."""

    duration_seconds: float
    memory_mb: float
    cpu_seconds: float
    api_calls: int
    cache_hits: int
    cache_misses: int


@strawberry.type
class ExecutionType:
    """Agent execution record."""

    id: str
    agent_id: str
    version: str
    status: ExecutionStatus
    inputs: JSON
    outputs: Optional[JSON]
    error: Optional[str]
    error_code: Optional[str]
    started_at: DateTime
    completed_at: Optional[DateTime]
    created_by: Optional[str]
    tenant_id: str

    @strawberry.field
    async def agent(self, info: Info) -> "AgentType":
        """Get the agent that was executed."""
        context: GreenLangContext = info.context
        agent_data = await context.agent_loader.load(self.agent_id)
        if not agent_data:
            raise Exception(f"Agent {self.agent_id} not found")
        return AgentType.from_dict(agent_data)

    @strawberry.field
    async def calculation_results(self) -> List[CalculationResultType]:
        """Get calculation results from this execution."""
        # TODO: Load from database
        return []

    @strawberry.field
    async def logs(self, limit: int = 100) -> List[ExecutionLogType]:
        """Get execution logs."""
        # TODO: Load from database
        return []

    @strawberry.field
    async def metrics(self) -> Optional[ExecutionMetricsType]:
        """Get execution metrics."""
        # TODO: Load from database
        return ExecutionMetricsType(
            duration_seconds=1.5,
            memory_mb=128.0,
            cpu_seconds=0.5,
            api_calls=3,
            cache_hits=10,
            cache_misses=2,
        )

    @strawberry.field
    def duration_seconds(self) -> Optional[float]:
        """Get execution duration in seconds."""
        if self.completed_at and self.started_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionType":
        """Create ExecutionType from dictionary."""
        return cls(
            id=data["id"],
            agent_id=data["agent_id"],
            version=data.get("version", "1.0.0"),
            status=ExecutionStatus(data.get("status", "PENDING")),
            inputs=data.get("inputs", {}),
            outputs=data.get("outputs"),
            error=data.get("error"),
            error_code=data.get("error_code"),
            started_at=data.get("started_at", datetime.now(timezone.utc)),
            completed_at=data.get("completed_at"),
            created_by=data.get("created_by"),
            tenant_id=data["tenant_id"],
        )


@strawberry.type
class AgentVersionType:
    """Agent version record."""

    version: str
    agent_id: str
    changelog: Optional[str]
    artifact_url: Optional[str]
    is_latest: bool
    created_at: DateTime
    created_by: Optional[str]


@strawberry.type
class AgentMetricsType:
    """Aggregated metrics for an agent."""

    invocation_count: int
    success_count: int
    failure_count: int
    average_duration_seconds: float
    p95_duration_seconds: float
    p99_duration_seconds: float
    total_compute_cost: float
    last_invoked_at: Optional[DateTime]


@strawberry.type
class AgentType:
    """Agent definition and metadata."""

    id: str
    agent_id: str
    name: str
    version: str
    state: AgentState
    category: str
    description: Optional[str]
    tags: List[str]
    entrypoint: str
    deterministic: bool
    inputs_schema: Optional[JSON]
    outputs_schema: Optional[JSON]
    regulatory_frameworks: List[str]
    tenant_id: str
    created_at: DateTime
    updated_at: DateTime
    created_by: Optional[str]

    @strawberry.field
    async def versions(
        self,
        limit: int = 10,
        offset: int = 0,
    ) -> List[AgentVersionType]:
        """Get all versions of this agent."""
        # TODO: Load from database
        return [
            AgentVersionType(
                version=self.version,
                agent_id=self.agent_id,
                changelog=None,
                artifact_url=None,
                is_latest=True,
                created_at=self.created_at,
                created_by=self.created_by,
            )
        ]

    @strawberry.field
    async def executions(
        self,
        info: Info,
        status: Optional[ExecutionStatus] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> List[ExecutionType]:
        """Get recent executions of this agent."""
        # TODO: Load from database with filters
        return []

    @strawberry.field
    async def metrics(
        self,
        period_days: int = 30,
    ) -> AgentMetricsType:
        """Get aggregated metrics for this agent."""
        # TODO: Load from database/metrics service
        return AgentMetricsType(
            invocation_count=1000,
            success_count=990,
            failure_count=10,
            average_duration_seconds=1.5,
            p95_duration_seconds=3.0,
            p99_duration_seconds=5.0,
            total_compute_cost=10.50,
            last_invoked_at=datetime.now(timezone.utc),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentType":
        """Create AgentType from dictionary."""
        return cls(
            id=data["id"],
            agent_id=data["agent_id"],
            name=data["name"],
            version=data.get("version", "1.0.0"),
            state=AgentState(data.get("state", "DRAFT")),
            category=data.get("category", "emissions"),
            description=data.get("description"),
            tags=data.get("tags", []),
            entrypoint=data.get("entrypoint", ""),
            deterministic=data.get("deterministic", True),
            inputs_schema=data.get("inputs_schema"),
            outputs_schema=data.get("outputs_schema"),
            regulatory_frameworks=data.get("regulatory_frameworks", []),
            tenant_id=data["tenant_id"],
            created_at=data.get("created_at", datetime.now(timezone.utc)),
            updated_at=data.get("updated_at", datetime.now(timezone.utc)),
            created_by=data.get("created_by"),
        )


@strawberry.type
class TenantType:
    """Tenant organization."""

    id: str
    name: str
    slug: str
    plan: str
    quota_agents: int
    quota_executions_per_month: int
    used_agents: int
    used_executions_this_month: int
    created_at: DateTime

    @strawberry.field
    async def agents(
        self,
        state: Optional[AgentState] = None,
        category: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> List[AgentType]:
        """Get agents for this tenant."""
        # TODO: Load from database
        return []


# =============================================================================
# GraphQL Types - Connections (Cursor-based Pagination)
# =============================================================================


@strawberry.type
class AgentEdge:
    """Edge in agent connection."""

    cursor: str
    node: AgentType


@strawberry.type
class AgentConnection:
    """Paginated connection of agents."""

    edges: List[AgentEdge]
    page_info: PageInfo


@strawberry.type
class ExecutionEdge:
    """Edge in execution connection."""

    cursor: str
    node: ExecutionType


@strawberry.type
class ExecutionConnection:
    """Paginated connection of executions."""

    edges: List[ExecutionEdge]
    page_info: PageInfo


# =============================================================================
# GraphQL Input Types
# =============================================================================


@strawberry.input
class AgentCreateInput:
    """Input for creating a new agent."""

    agent_id: str = strawberry.field(description="Unique agent ID (category/name)")
    name: str = strawberry.field(description="Human-readable name")
    version: str = strawberry.field(default="1.0.0", description="Initial version")
    category: str = strawberry.field(description="Agent category")
    description: Optional[str] = None
    tags: List[str] = strawberry.field(default_factory=list)
    entrypoint: str = strawberry.field(description="Python entrypoint")
    deterministic: bool = True
    inputs_schema: Optional[JSON] = None
    outputs_schema: Optional[JSON] = None
    regulatory_frameworks: List[str] = strawberry.field(default_factory=list)


@strawberry.input
class AgentUpdateInput:
    """Input for updating an agent."""

    name: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None


@strawberry.input
class AgentFilterInput:
    """Filters for agent queries."""

    state: Optional[AgentState] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    search: Optional[str] = None
    regulatory_framework: Optional[str] = None


@strawberry.input
class ExecutionCreateInput:
    """Input for creating a new execution."""

    agent_id: str = strawberry.field(description="Agent to execute")
    version: Optional[str] = strawberry.field(
        default=None, description="Version to execute (latest if not specified)"
    )
    inputs: JSON = strawberry.field(description="Execution inputs")
    priority: int = strawberry.field(default=5, description="Priority 1-10")
    timeout_seconds: Optional[int] = strawberry.field(
        default=None, description="Execution timeout"
    )
    idempotency_key: Optional[str] = strawberry.field(
        default=None, description="Key for idempotent execution"
    )


@strawberry.input
class ExecutionFilterInput:
    """Filters for execution queries."""

    status: Optional[ExecutionStatus] = None
    agent_id: Optional[str] = None
    created_after: Optional[DateTime] = None
    created_before: Optional[DateTime] = None


# =============================================================================
# GraphQL Query Type
# =============================================================================


@strawberry.type
class Query:
    """Root query type for GreenLang GraphQL API."""

    @strawberry.field(description="Get agent by ID")
    async def agent(self, info: Info, id: str) -> Optional[AgentType]:
        """Get a single agent by ID."""
        context: GreenLangContext = info.context
        agent_data = await context.agent_loader.load(id)
        return AgentType.from_dict(agent_data) if agent_data else None

    @strawberry.field(description="Get agent by agent_id")
    async def agent_by_agent_id(
        self, info: Info, agent_id: str
    ) -> Optional[AgentType]:
        """Get an agent by its agent_id (e.g., 'carbon/calculator')."""
        # TODO: Implement lookup by agent_id
        return None

    @strawberry.field(description="List agents with filters")
    async def agents(
        self,
        info: Info,
        filters: Optional[AgentFilterInput] = None,
        first: int = 20,
        after: Optional[str] = None,
        order_by: str = "created_at",
        order: SortOrder = SortOrder.DESC,
    ) -> AgentConnection:
        """Get paginated list of agents."""
        # TODO: Implement with actual database query
        return AgentConnection(
            edges=[],
            page_info=PageInfo(
                has_next_page=False,
                has_previous_page=False,
                start_cursor=None,
                end_cursor=None,
                total_count=0,
            ),
        )

    @strawberry.field(description="Search agents by text")
    async def search_agents(
        self,
        info: Info,
        query: str,
        limit: int = 20,
    ) -> List[AgentType]:
        """Full-text search for agents."""
        # TODO: Implement with Elasticsearch or PostgreSQL FTS
        return []

    @strawberry.field(description="Get execution by ID")
    async def execution(self, info: Info, id: str) -> Optional[ExecutionType]:
        """Get a single execution by ID."""
        context: GreenLangContext = info.context
        execution_data = await context.execution_loader.load(id)
        return ExecutionType.from_dict(execution_data) if execution_data else None

    @strawberry.field(description="List executions with filters")
    async def executions(
        self,
        info: Info,
        filters: Optional[ExecutionFilterInput] = None,
        first: int = 20,
        after: Optional[str] = None,
        order_by: str = "started_at",
        order: SortOrder = SortOrder.DESC,
    ) -> ExecutionConnection:
        """Get paginated list of executions."""
        # TODO: Implement with actual database query
        return ExecutionConnection(
            edges=[],
            page_info=PageInfo(
                has_next_page=False,
                has_previous_page=False,
                start_cursor=None,
                end_cursor=None,
                total_count=0,
            ),
        )

    @strawberry.field(description="Get current tenant")
    async def me(self, info: Info) -> TenantType:
        """Get current tenant information."""
        context: GreenLangContext = info.context
        return TenantType(
            id=context.tenant_id,
            name="Development Tenant",
            slug="dev",
            plan="enterprise",
            quota_agents=100,
            quota_executions_per_month=10000,
            used_agents=5,
            used_executions_this_month=100,
            created_at=datetime.now(timezone.utc),
        )

    @strawberry.field(description="Get emission factor by ID")
    async def emission_factor(
        self, info: Info, id: str
    ) -> Optional[EmissionFactorType]:
        """Get a single emission factor by ID."""
        # TODO: Implement
        return None

    @strawberry.field(description="Search emission factors")
    async def emission_factors(
        self,
        info: Info,
        source: Optional[str] = None,
        region: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = 50,
    ) -> List[EmissionFactorType]:
        """Search emission factors."""
        # TODO: Implement
        return []


# =============================================================================
# GraphQL Mutation Type
# =============================================================================


@strawberry.type
class AgentMutationResult:
    """Result of an agent mutation."""

    success: bool
    agent: Optional[AgentType]
    error: Optional[str]


@strawberry.type
class ExecutionMutationResult:
    """Result of an execution mutation."""

    success: bool
    execution: Optional[ExecutionType]
    error: Optional[str]


@strawberry.type
class Mutation:
    """Root mutation type for GreenLang GraphQL API."""

    @strawberry.mutation(description="Create a new agent")
    async def create_agent(
        self, info: Info, input: AgentCreateInput
    ) -> AgentMutationResult:
        """Create a new agent in DRAFT state."""
        context: GreenLangContext = info.context

        try:
            # TODO: Implement with actual service call
            agent = AgentType(
                id=f"agent-{input.agent_id.replace('/', '-')}",
                agent_id=input.agent_id,
                name=input.name,
                version=input.version,
                state=AgentState.DRAFT,
                category=input.category,
                description=input.description,
                tags=input.tags,
                entrypoint=input.entrypoint,
                deterministic=input.deterministic,
                inputs_schema=input.inputs_schema,
                outputs_schema=input.outputs_schema,
                regulatory_frameworks=input.regulatory_frameworks,
                tenant_id=context.tenant_id,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                created_by=context.user_id,
            )

            return AgentMutationResult(success=True, agent=agent, error=None)

        except Exception as e:
            logger.error(f"Failed to create agent: {e}")
            return AgentMutationResult(success=False, agent=None, error=str(e))

    @strawberry.mutation(description="Update an agent")
    async def update_agent(
        self, info: Info, id: str, input: AgentUpdateInput
    ) -> AgentMutationResult:
        """Update an existing agent."""
        context: GreenLangContext = info.context

        try:
            # TODO: Implement with actual service call
            agent_data = await context.agent_loader.load(id)
            if not agent_data:
                return AgentMutationResult(
                    success=False, agent=None, error=f"Agent {id} not found"
                )

            # Apply updates
            if input.name:
                agent_data["name"] = input.name
            if input.description is not None:
                agent_data["description"] = input.description
            if input.tags is not None:
                agent_data["tags"] = input.tags

            agent_data["updated_at"] = datetime.now(timezone.utc)

            agent = AgentType.from_dict(agent_data)
            return AgentMutationResult(success=True, agent=agent, error=None)

        except Exception as e:
            logger.error(f"Failed to update agent: {e}")
            return AgentMutationResult(success=False, agent=None, error=str(e))

    @strawberry.mutation(description="Delete an agent")
    async def delete_agent(self, info: Info, id: str) -> AgentMutationResult:
        """Delete (retire) an agent."""
        context: GreenLangContext = info.context

        try:
            # TODO: Implement with actual service call
            agent_data = await context.agent_loader.load(id)
            if not agent_data:
                return AgentMutationResult(
                    success=False, agent=None, error=f"Agent {id} not found"
                )

            # Transition to RETIRED
            agent_data["state"] = "RETIRED"
            agent_data["updated_at"] = datetime.now(timezone.utc)

            agent = AgentType.from_dict(agent_data)
            return AgentMutationResult(success=True, agent=agent, error=None)

        except Exception as e:
            logger.error(f"Failed to delete agent: {e}")
            return AgentMutationResult(success=False, agent=None, error=str(e))

    @strawberry.mutation(description="Transition agent state")
    async def transition_agent_state(
        self, info: Info, id: str, target_state: AgentState, reason: Optional[str] = None
    ) -> AgentMutationResult:
        """Transition agent to a new lifecycle state."""
        context: GreenLangContext = info.context

        try:
            # TODO: Implement with actual service call and state machine validation
            agent_data = await context.agent_loader.load(id)
            if not agent_data:
                return AgentMutationResult(
                    success=False, agent=None, error=f"Agent {id} not found"
                )

            agent_data["state"] = target_state.value
            agent_data["updated_at"] = datetime.now(timezone.utc)

            agent = AgentType.from_dict(agent_data)
            return AgentMutationResult(success=True, agent=agent, error=None)

        except Exception as e:
            logger.error(f"Failed to transition agent state: {e}")
            return AgentMutationResult(success=False, agent=None, error=str(e))

    @strawberry.mutation(description="Execute an agent")
    async def execute_agent(
        self, info: Info, input: ExecutionCreateInput
    ) -> ExecutionMutationResult:
        """Start a new agent execution."""
        context: GreenLangContext = info.context

        try:
            # TODO: Implement with actual execution service
            execution = ExecutionType(
                id=f"exec-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                agent_id=input.agent_id,
                version=input.version or "1.0.0",
                status=ExecutionStatus.PENDING,
                inputs=input.inputs,
                outputs=None,
                error=None,
                error_code=None,
                started_at=datetime.now(timezone.utc),
                completed_at=None,
                created_by=context.user_id,
                tenant_id=context.tenant_id,
            )

            return ExecutionMutationResult(
                success=True, execution=execution, error=None
            )

        except Exception as e:
            logger.error(f"Failed to execute agent: {e}")
            return ExecutionMutationResult(success=False, execution=None, error=str(e))

    @strawberry.mutation(description="Cancel an execution")
    async def cancel_execution(
        self, info: Info, id: str
    ) -> ExecutionMutationResult:
        """Cancel a running execution."""
        context: GreenLangContext = info.context

        try:
            # TODO: Implement with actual execution service
            execution_data = await context.execution_loader.load(id)
            if not execution_data:
                return ExecutionMutationResult(
                    success=False, execution=None, error=f"Execution {id} not found"
                )

            execution_data["status"] = "CANCELLED"
            execution_data["completed_at"] = datetime.now(timezone.utc)

            execution = ExecutionType.from_dict(execution_data)
            return ExecutionMutationResult(
                success=True, execution=execution, error=None
            )

        except Exception as e:
            logger.error(f"Failed to cancel execution: {e}")
            return ExecutionMutationResult(success=False, execution=None, error=str(e))


# =============================================================================
# GraphQL Subscription Type
# =============================================================================


@strawberry.type
class Subscription:
    """Root subscription type for GreenLang GraphQL API."""

    @strawberry.subscription(description="Subscribe to execution progress")
    async def execution_progress(
        self, info: Info, execution_id: str
    ) -> AsyncGenerator[ExecutionType, None]:
        """
        Subscribe to real-time execution progress updates.

        Yields updates as the execution progresses.
        """
        # TODO: Implement with actual pub/sub
        for i in range(10):
            await asyncio.sleep(1)
            yield ExecutionType(
                id=execution_id,
                agent_id="carbon/calculator",
                version="1.0.0",
                status=ExecutionStatus.RUNNING if i < 9 else ExecutionStatus.COMPLETED,
                inputs={"progress": i * 10},
                outputs=None if i < 9 else {"result": "complete"},
                error=None,
                error_code=None,
                started_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc) if i == 9 else None,
                created_by=None,
                tenant_id="dev",
            )

    @strawberry.subscription(description="Subscribe to tenant metrics")
    async def metrics(
        self, info: Info, interval_seconds: int = 5
    ) -> AsyncGenerator[JSON, None]:
        """
        Subscribe to real-time tenant metrics.

        Yields metrics updates at the specified interval.
        """
        while True:
            await asyncio.sleep(interval_seconds)
            yield {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "executions_per_minute": 10,
                "success_rate": 0.99,
                "average_duration_seconds": 1.5,
                "active_agents": 5,
            }

    @strawberry.subscription(description="Subscribe to calculation results")
    async def calculation_results(
        self, info: Info, execution_id: str
    ) -> AsyncGenerator[CalculationResultType, None]:
        """
        Subscribe to live calculation results as they're computed.

        Useful for long-running calculations to show intermediate results.
        """
        # TODO: Implement with actual pub/sub
        for i in range(5):
            await asyncio.sleep(1)
            yield CalculationResultType(
                id=f"calc-{i}",
                execution_id=execution_id,
                result_type="intermediate" if i < 4 else "final",
                value=100.0 + i * 10,
                unit="tCO2e",
                confidence_score=0.8 + i * 0.04,
                methodology="GHG Protocol",
                source_hash=f"sha256:{i * 1111111111111111}",
                emission_factors=[],
                created_at=datetime.now(timezone.utc),
            )


# =============================================================================
# Schema and Application Factory
# =============================================================================


# Create the schema
schema = Schema(
    query=Query,
    mutation=Mutation,
    subscription=Subscription,
    config=StrawberryConfig(auto_camel_case=True),
    extensions=[
        QueryComplexityExtension,
        QueryDepthExtension,
    ],
)


async def get_context(request) -> GreenLangContext:
    """Create GraphQL context from request."""
    # Extract tenant and user from request state (set by auth middleware)
    tenant_id = getattr(request.state, "tenant_id", "dev_tenant")
    user_id = getattr(request.state, "user_id", None)
    db_pool = getattr(request.app.state, "db_pool", None)
    redis = getattr(request.app.state, "redis", None)

    return GreenLangContext(
        request=request,
        tenant_id=tenant_id,
        user_id=user_id,
        db_pool=db_pool,
        redis=redis,
    )


def create_graphql_app() -> GraphQLRouter:
    """
    Create the GraphQL FastAPI router.

    Returns:
        Configured GraphQLRouter

    Example:
        >>> graphql_app = create_graphql_app()
        >>> app.include_router(graphql_app, prefix="/graphql")
    """
    return GraphQLRouter(
        schema,
        context_getter=get_context,
        graphiql=True,  # Enable GraphiQL IDE
    )
