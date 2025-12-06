"""
GraphQL Schema Builder for GreenLang

This module provides GraphQL schema building and resolver
management for GreenLang services.

Features:
- Schema-first and code-first approaches
- Automatic type generation from Pydantic models
- DataLoader integration
- Subscription support
- Federation support

Example:
    >>> builder = GraphQLSchemaBuilder()
    >>> builder.add_query("emissions", EmissionType, resolver)
    >>> schema = builder.build()
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, Type, TypeVar, Union
from uuid import uuid4

from pydantic import BaseModel, Field

try:
    import strawberry
    from strawberry import Schema, type as strawberry_type, field as strawberry_field
    from strawberry.dataloader import DataLoader
    STRAWBERRY_AVAILABLE = True
except ImportError:
    STRAWBERRY_AVAILABLE = False
    strawberry = None
    Schema = None
    strawberry_type = None
    strawberry_field = None
    DataLoader = None

logger = logging.getLogger(__name__)

T = TypeVar("T")


class GraphQLOperationType(str, Enum):
    """GraphQL operation types."""
    QUERY = "query"
    MUTATION = "mutation"
    SUBSCRIPTION = "subscription"


@dataclass
class GraphQLSchemaConfig:
    """Configuration for GraphQL schema."""
    enable_introspection: bool = True
    enable_federation: bool = False
    max_depth: int = 10
    enable_tracing: bool = True
    enable_validation: bool = True
    enable_dataloader: bool = True


class FieldDefinition(BaseModel):
    """GraphQL field definition."""
    name: str = Field(..., description="Field name")
    return_type: str = Field(..., description="Return type")
    arguments: Dict[str, str] = Field(default_factory=dict)
    description: Optional[str] = Field(default=None)
    resolver_name: Optional[str] = Field(default=None)
    deprecated: bool = Field(default=False)
    deprecation_reason: Optional[str] = Field(default=None)


class TypeDefinition(BaseModel):
    """GraphQL type definition."""
    name: str = Field(..., description="Type name")
    fields: List[FieldDefinition] = Field(default_factory=list)
    description: Optional[str] = Field(default=None)
    interfaces: List[str] = Field(default_factory=list)
    is_input: bool = Field(default=False)
    is_interface: bool = Field(default=False)


class QueryDefinition(BaseModel):
    """Query field definition."""
    name: str = Field(..., description="Query name")
    return_type: str = Field(..., description="Return type")
    arguments: Dict[str, Any] = Field(default_factory=dict)
    description: Optional[str] = Field(default=None)
    resolver: Optional[Callable] = Field(default=None, exclude=True)

    class Config:
        arbitrary_types_allowed = True


class MutationDefinition(BaseModel):
    """Mutation field definition."""
    name: str = Field(..., description="Mutation name")
    return_type: str = Field(..., description="Return type")
    input_type: str = Field(..., description="Input type")
    description: Optional[str] = Field(default=None)
    resolver: Optional[Callable] = Field(default=None, exclude=True)

    class Config:
        arbitrary_types_allowed = True


class SubscriptionDefinition(BaseModel):
    """Subscription field definition."""
    name: str = Field(..., description="Subscription name")
    return_type: str = Field(..., description="Return type")
    arguments: Dict[str, Any] = Field(default_factory=dict)
    description: Optional[str] = Field(default=None)
    resolver: Optional[Callable] = Field(default=None, exclude=True)

    class Config:
        arbitrary_types_allowed = True


class DataLoaderRegistry:
    """
    Registry for GraphQL DataLoaders.

    Manages DataLoader instances for batching and caching.
    """

    def __init__(self):
        """Initialize DataLoader registry."""
        self._loaders: Dict[str, Callable] = {}
        self._instances: Dict[str, Any] = {}

    def register(
        self,
        name: str,
        batch_load_fn: Callable[[List[str]], List[Any]]
    ) -> None:
        """
        Register a DataLoader.

        Args:
            name: Loader name
            batch_load_fn: Batch loading function
        """
        self._loaders[name] = batch_load_fn
        logger.debug(f"Registered DataLoader: {name}")

    def get(self, name: str) -> Optional[Any]:
        """
        Get a DataLoader instance.

        Args:
            name: Loader name

        Returns:
            DataLoader instance
        """
        if name not in self._instances and name in self._loaders:
            if STRAWBERRY_AVAILABLE and DataLoader:
                self._instances[name] = DataLoader(load_fn=self._loaders[name])

        return self._instances.get(name)

    def clear(self) -> None:
        """Clear all DataLoader instances."""
        self._instances.clear()


class GraphQLSchemaBuilder:
    """
    GraphQL schema builder for GreenLang.

    Builds GraphQL schemas from type definitions and resolvers
    using Strawberry GraphQL.

    Attributes:
        config: Schema configuration
        types: Registered types
        queries: Registered queries

    Example:
        >>> builder = GraphQLSchemaBuilder()
        >>> @builder.type
        >>> class Emission:
        ...     id: str
        ...     co2_value: float
        >>> @builder.query
        >>> async def get_emission(id: str) -> Emission: ...
        >>> schema = builder.build()
    """

    def __init__(self, config: Optional[GraphQLSchemaConfig] = None):
        """
        Initialize GraphQL schema builder.

        Args:
            config: Schema configuration
        """
        self.config = config or GraphQLSchemaConfig()
        self._types: Dict[str, TypeDefinition] = {}
        self._queries: Dict[str, QueryDefinition] = {}
        self._mutations: Dict[str, MutationDefinition] = {}
        self._subscriptions: Dict[str, SubscriptionDefinition] = {}
        self._resolvers: Dict[str, Callable] = {}
        self._dataloader_registry = DataLoaderRegistry()
        self._strawberry_types: Dict[str, Any] = {}

        logger.info("GraphQLSchemaBuilder initialized")

    def add_type(
        self,
        type_def: TypeDefinition
    ) -> None:
        """
        Add a type definition.

        Args:
            type_def: Type definition
        """
        self._types[type_def.name] = type_def
        logger.debug(f"Added type: {type_def.name}")

    def add_type_from_pydantic(
        self,
        model: Type[BaseModel],
        name: Optional[str] = None,
        exclude_fields: Optional[List[str]] = None
    ) -> str:
        """
        Add a type from a Pydantic model.

        Args:
            model: Pydantic model class
            name: Optional type name
            exclude_fields: Fields to exclude

        Returns:
            Type name
        """
        type_name = name or model.__name__
        exclude = exclude_fields or []

        fields = []
        for field_name, field_info in model.__fields__.items():
            if field_name in exclude:
                continue

            field_type = self._map_python_type_to_graphql(
                field_info.outer_type_
            )

            fields.append(FieldDefinition(
                name=field_name,
                return_type=field_type,
                description=field_info.field_info.description,
            ))

        type_def = TypeDefinition(
            name=type_name,
            fields=fields,
            description=model.__doc__,
        )

        self.add_type(type_def)
        return type_name

    def _map_python_type_to_graphql(self, python_type: Type) -> str:
        """Map Python type to GraphQL type."""
        type_map = {
            str: "String",
            int: "Int",
            float: "Float",
            bool: "Boolean",
            datetime: "DateTime",
        }

        # Handle basic types
        if python_type in type_map:
            return type_map[python_type]

        # Handle Optional
        origin = getattr(python_type, "__origin__", None)
        if origin is Union:
            args = python_type.__args__
            if type(None) in args:
                non_none = [a for a in args if a is not type(None)][0]
                return self._map_python_type_to_graphql(non_none)

        # Handle List
        if origin is list:
            inner = python_type.__args__[0]
            inner_type = self._map_python_type_to_graphql(inner)
            return f"[{inner_type}]"

        # Default to String
        return "String"

    def add_query(
        self,
        name: str,
        return_type: str,
        resolver: Callable,
        arguments: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None
    ) -> None:
        """
        Add a query field.

        Args:
            name: Query name
            return_type: Return type
            resolver: Resolver function
            arguments: Query arguments
            description: Query description
        """
        self._queries[name] = QueryDefinition(
            name=name,
            return_type=return_type,
            arguments=arguments or {},
            description=description,
            resolver=resolver,
        )
        self._resolvers[f"Query.{name}"] = resolver
        logger.debug(f"Added query: {name}")

    def add_mutation(
        self,
        name: str,
        return_type: str,
        input_type: str,
        resolver: Callable,
        description: Optional[str] = None
    ) -> None:
        """
        Add a mutation field.

        Args:
            name: Mutation name
            return_type: Return type
            input_type: Input type
            resolver: Resolver function
            description: Mutation description
        """
        self._mutations[name] = MutationDefinition(
            name=name,
            return_type=return_type,
            input_type=input_type,
            description=description,
            resolver=resolver,
        )
        self._resolvers[f"Mutation.{name}"] = resolver
        logger.debug(f"Added mutation: {name}")

    def add_subscription(
        self,
        name: str,
        return_type: str,
        resolver: Callable,
        arguments: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None
    ) -> None:
        """
        Add a subscription field.

        Args:
            name: Subscription name
            return_type: Return type
            resolver: Async generator resolver
            arguments: Subscription arguments
            description: Subscription description
        """
        self._subscriptions[name] = SubscriptionDefinition(
            name=name,
            return_type=return_type,
            arguments=arguments or {},
            description=description,
            resolver=resolver,
        )
        self._resolvers[f"Subscription.{name}"] = resolver
        logger.debug(f"Added subscription: {name}")

    def register_dataloader(
        self,
        name: str,
        batch_load_fn: Callable[[List[str]], List[Any]]
    ) -> None:
        """
        Register a DataLoader.

        Args:
            name: Loader name
            batch_load_fn: Batch loading function
        """
        self._dataloader_registry.register(name, batch_load_fn)

    def build(self) -> Any:
        """
        Build the GraphQL schema.

        Returns:
            Strawberry Schema instance
        """
        if not STRAWBERRY_AVAILABLE:
            raise ImportError(
                "Strawberry is required for GraphQL. "
                "Install with: pip install strawberry-graphql"
            )

        # Build Query type
        query_type = self._build_query_type()

        # Build Mutation type if mutations exist
        mutation_type = None
        if self._mutations:
            mutation_type = self._build_mutation_type()

        # Build Subscription type if subscriptions exist
        subscription_type = None
        if self._subscriptions:
            subscription_type = self._build_subscription_type()

        # Create schema
        schema = strawberry.Schema(
            query=query_type,
            mutation=mutation_type,
            subscription=subscription_type,
        )

        logger.info(
            f"Built GraphQL schema: {len(self._queries)} queries, "
            f"{len(self._mutations)} mutations, "
            f"{len(self._subscriptions)} subscriptions"
        )

        return schema

    def _build_query_type(self) -> Type:
        """Build the Query type."""
        fields = {}

        for name, query_def in self._queries.items():
            fields[name] = strawberry.field(
                resolver=query_def.resolver,
                description=query_def.description,
            )

        # Add default health query
        async def health() -> str:
            return "healthy"

        fields["health"] = strawberry.field(
            resolver=health,
            description="Health check"
        )

        Query = type("Query", (), fields)
        return strawberry.type(Query)

    def _build_mutation_type(self) -> Type:
        """Build the Mutation type."""
        fields = {}

        for name, mutation_def in self._mutations.items():
            fields[name] = strawberry.field(
                resolver=mutation_def.resolver,
                description=mutation_def.description,
            )

        Mutation = type("Mutation", (), fields)
        return strawberry.type(Mutation)

    def _build_subscription_type(self) -> Type:
        """Build the Subscription type."""
        fields = {}

        for name, sub_def in self._subscriptions.items():
            fields[name] = strawberry.subscription(
                resolver=sub_def.resolver,
                description=sub_def.description,
            )

        Subscription = type("Subscription", (), fields)
        return strawberry.type(Subscription)

    def generate_sdl(self) -> str:
        """
        Generate SDL (Schema Definition Language) string.

        Returns:
            SDL string representation
        """
        lines = []

        # Types
        for name, type_def in self._types.items():
            if type_def.is_input:
                lines.append(f"input {name} {{")
            elif type_def.is_interface:
                lines.append(f"interface {name} {{")
            else:
                lines.append(f"type {name} {{")

            for field in type_def.fields:
                lines.append(f"  {field.name}: {field.return_type}")

            lines.append("}")
            lines.append("")

        # Query type
        if self._queries:
            lines.append("type Query {")
            for name, query_def in self._queries.items():
                args = ", ".join(
                    f"{k}: {v}" for k, v in query_def.arguments.items()
                )
                if args:
                    lines.append(f"  {name}({args}): {query_def.return_type}")
                else:
                    lines.append(f"  {name}: {query_def.return_type}")
            lines.append("}")
            lines.append("")

        # Mutation type
        if self._mutations:
            lines.append("type Mutation {")
            for name, mutation_def in self._mutations.items():
                lines.append(
                    f"  {name}(input: {mutation_def.input_type}): "
                    f"{mutation_def.return_type}"
                )
            lines.append("}")
            lines.append("")

        # Subscription type
        if self._subscriptions:
            lines.append("type Subscription {")
            for name, sub_def in self._subscriptions.items():
                args = ", ".join(
                    f"{k}: {v}" for k, v in sub_def.arguments.items()
                )
                if args:
                    lines.append(f"  {name}({args}): {sub_def.return_type}")
                else:
                    lines.append(f"  {name}: {sub_def.return_type}")
            lines.append("}")

        return "\n".join(lines)

    def get_dataloader(self, name: str) -> Optional[Any]:
        """
        Get a DataLoader instance.

        Args:
            name: Loader name

        Returns:
            DataLoader instance
        """
        return self._dataloader_registry.get(name)

    def type(self, cls: Type[T]) -> Type[T]:
        """
        Decorator to register a GraphQL type.

        Args:
            cls: Class to register

        Returns:
            Decorated class
        """
        if STRAWBERRY_AVAILABLE:
            decorated = strawberry.type(cls)
            self._strawberry_types[cls.__name__] = decorated
            return decorated
        return cls

    def input(self, cls: Type[T]) -> Type[T]:
        """
        Decorator to register a GraphQL input type.

        Args:
            cls: Class to register

        Returns:
            Decorated class
        """
        if STRAWBERRY_AVAILABLE:
            decorated = strawberry.input(cls)
            self._strawberry_types[cls.__name__] = decorated
            return decorated
        return cls

    def query(self, fn: Callable) -> Callable:
        """
        Decorator to register a query resolver.

        Args:
            fn: Resolver function

        Returns:
            Decorated function
        """
        name = fn.__name__
        self._queries[name] = QueryDefinition(
            name=name,
            return_type="Unknown",  # Would need type introspection
            resolver=fn,
            description=fn.__doc__,
        )
        self._resolvers[f"Query.{name}"] = fn
        return fn

    def mutation(self, fn: Callable) -> Callable:
        """
        Decorator to register a mutation resolver.

        Args:
            fn: Resolver function

        Returns:
            Decorated function
        """
        name = fn.__name__
        self._mutations[name] = MutationDefinition(
            name=name,
            return_type="Unknown",
            input_type="Unknown",
            resolver=fn,
            description=fn.__doc__,
        )
        self._resolvers[f"Mutation.{name}"] = fn
        return fn


# ============================================================================
# PROCESS HEAT AGENTS - GRAPHQL SCHEMA IMPLEMENTATION
# ============================================================================

class AgentStatus(str, Enum):
    """Agent execution status."""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


class JobStatus(str, Enum):
    """Calculation job status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ReportType(str, Enum):
    """Compliance report type."""
    GHG_EMISSIONS = "ghg_emissions"
    ENERGY_AUDIT = "energy_audit"
    EFFICIENCY_ANALYSIS = "efficiency_analysis"
    REGULATORY_COMPLIANCE = "regulatory_compliance"
    PREDICTIVE_MAINTENANCE = "predictive_maintenance"


class ComplianceStatus(str, Enum):
    """Compliance status."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    UNDER_REVIEW = "under_review"
    PENDING_REMEDIATION = "pending_remediation"


if STRAWBERRY_AVAILABLE and strawberry is not None:
    from datetime import date

    # ========================================================================
    # STRAWBERRY TYPES - EMISSION RESULT
    # ========================================================================

    @strawberry.type
    class EmissionResult:
        """GHG emission calculation result."""
        id: str = strawberry.field(description="Result ID")
        facility_id: str = strawberry.field(description="Facility identifier")
        co2_tonnes: float = strawberry.field(description="CO2 emissions in tonnes")
        ch4_tonnes: float = strawberry.field(description="CH4 emissions in tonnes")
        n2o_tonnes: float = strawberry.field(description="N2O emissions in tonnes")
        total_co2e_tonnes: float = strawberry.field(
            description="Total CO2 equivalent in tonnes"
        )
        provenance_hash: str = strawberry.field(
            description="SHA-256 provenance hash for audit trail"
        )
        calculation_method: str = strawberry.field(
            description="Calculation methodology used"
        )
        timestamp: datetime = strawberry.field(
            description="Calculation timestamp"
        )
        confidence_score: float = strawberry.field(
            description="Confidence score (0.0-1.0)"
        )

    # ========================================================================
    # STRAWBERRY TYPES - AGENT METRICS
    # ========================================================================

    @strawberry.type
    class AgentMetricsType:
        """Agent performance metrics."""
        execution_time_ms: float = strawberry.field(
            description="Execution time in milliseconds"
        )
        memory_usage_mb: float = strawberry.field(
            description="Memory usage in megabytes"
        )
        records_processed: int = strawberry.field(
            description="Number of records processed"
        )
        processing_rate: float = strawberry.field(
            description="Records per second processing rate"
        )
        cache_hit_ratio: float = strawberry.field(
            description="Cache hit ratio (0.0-1.0)"
        )
        error_count: int = strawberry.field(description="Number of errors")

    # ========================================================================
    # STRAWBERRY TYPES - PROCESS HEAT AGENT
    # ========================================================================

    @strawberry.type
    class ProcessHeatAgent:
        """Process heat agent with monitoring and control capabilities."""
        id: str = strawberry.field(description="Agent ID")
        name: str = strawberry.field(description="Agent name")
        agent_type: str = strawberry.field(
            description="Agent type (e.g., GL-001, GL-002)"
        )
        status: str = strawberry.field(description="Current agent status")
        enabled: bool = strawberry.field(description="Whether agent is enabled")
        version: str = strawberry.field(description="Agent software version")
        last_run: Optional[datetime] = strawberry.field(
            description="Timestamp of last run"
        )
        next_run: Optional[datetime] = strawberry.field(
            description="Scheduled next run"
        )
        metrics: AgentMetricsType = strawberry.field(
            description="Agent performance metrics"
        )
        error_message: Optional[str] = strawberry.field(
            description="Last error message if any"
        )
        created_at: datetime = strawberry.field(
            description="Agent creation timestamp"
        )
        updated_at: datetime = strawberry.field(
            description="Last update timestamp"
        )

    # ========================================================================
    # STRAWBERRY TYPES - CALCULATION JOB
    # ========================================================================

    @strawberry.type
    class CalculationJob:
        """Asynchronous calculation job."""
        id: str = strawberry.field(description="Job ID")
        status: str = strawberry.field(description="Job status")
        progress_percent: int = strawberry.field(
            description="Progress percentage (0-100)"
        )
        agent_id: str = strawberry.field(description="Associated agent ID")
        input_summary: str = strawberry.field(
            description="Summary of input data"
        )
        results: Optional[List[EmissionResult]] = strawberry.field(
            description="Calculation results"
        )
        error_details: Optional[str] = strawberry.field(
            description="Error details if failed"
        )
        execution_time_ms: float = strawberry.field(
            description="Total execution time"
        )
        created_at: datetime = strawberry.field(
            description="Job creation timestamp"
        )
        started_at: Optional[datetime] = strawberry.field(
            description="Job start timestamp"
        )
        completed_at: Optional[datetime] = strawberry.field(
            description="Job completion timestamp"
        )

    # ========================================================================
    # STRAWBERRY TYPES - COMPLIANCE FINDING
    # ========================================================================

    @strawberry.type
    class ComplianceFinding:
        """Single compliance finding."""
        id: str = strawberry.field(description="Finding ID")
        category: str = strawberry.field(description="Compliance category")
        severity: str = strawberry.field(
            description="Severity level (critical, high, medium, low)"
        )
        description: str = strawberry.field(description="Finding description")
        remediation_action: Optional[str] = strawberry.field(
            description="Recommended remediation action"
        )
        deadline: Optional[date] = strawberry.field(
            description="Remediation deadline"
        )

    # ========================================================================
    # STRAWBERRY TYPES - COMPLIANCE REPORT
    # ========================================================================

    @strawberry.type
    class ComplianceReport:
        """Regulatory compliance report."""
        id: str = strawberry.field(description="Report ID")
        report_type: str = strawberry.field(
            description="Type of compliance report"
        )
        status: str = strawberry.field(
            description="Compliance status"
        )
        period_start: date = strawberry.field(
            description="Reporting period start"
        )
        period_end: date = strawberry.field(
            description="Reporting period end"
        )
        findings: List[ComplianceFinding] = strawberry.field(
            description="Compliance findings"
        )
        summary: str = strawberry.field(
            description="Executive summary"
        )
        action_items_count: int = strawberry.field(
            description="Number of action items"
        )
        generated_at: datetime = strawberry.field(
            description="Report generation timestamp"
        )

    # ========================================================================
    # STRAWBERRY TYPES - DATE RANGE INPUT
    # ========================================================================

    @strawberry.input
    class DateRangeInput:
        """Date range filter input."""
        start_date: date = strawberry.field(description="Start date")
        end_date: date = strawberry.field(description="End date")

    # ========================================================================
    # STRAWBERRY TYPES - CALCULATION INPUT
    # ========================================================================

    @strawberry.input
    class CalculationInput:
        """Input for calculation jobs."""
        agent_id: str = strawberry.field(description="Target agent ID")
        facility_id: str = strawberry.field(description="Facility identifier")
        date_range: DateRangeInput = strawberry.field(
            description="Calculation date range"
        )
        parameters: Optional[str] = strawberry.field(
            description="JSON parameters for calculation"
        )
        priority: str = strawberry.field(
            default="normal",
            description="Job priority (low, normal, high)"
        )

    # ========================================================================
    # STRAWBERRY TYPES - AGENT CONFIG INPUT
    # ========================================================================

    @strawberry.input
    class AgentConfigInput:
        """Agent configuration update input."""
        enabled: Optional[bool] = strawberry.field(
            description="Enable/disable agent"
        )
        execution_interval_minutes: Optional[int] = strawberry.field(
            description="Execution interval in minutes"
        )
        parameters: Optional[str] = strawberry.field(
            description="JSON agent parameters"
        )

    # ========================================================================
    # STRAWBERRY TYPES - REPORT PARAMS INPUT
    # ========================================================================

    @strawberry.input
    class ReportParamsInput:
        """Report generation parameters."""
        facility_ids: List[str] = strawberry.field(
            description="Target facility IDs"
        )
        date_range: DateRangeInput = strawberry.field(
            description="Report date range"
        )
        include_recommendations: bool = strawberry.field(
            default=True,
            description="Include recommendations"
        )

    # ========================================================================
    # STRAWBERRY TYPES - JOB PROGRESS EVENT
    # ========================================================================

    @strawberry.type
    class JobProgressEvent:
        """Job progress subscription event."""
        job_id: str = strawberry.field(description="Job ID")
        progress_percent: int = strawberry.field(
            description="Progress percentage"
        )
        status: str = strawberry.field(description="Current status")
        message: str = strawberry.field(description="Status message")
        timestamp: datetime = strawberry.field(
            description="Event timestamp"
        )

    # ========================================================================
    # STRAWBERRY TYPES - AGENT ALERT EVENT
    # ========================================================================

    @strawberry.type
    class AlertEvent:
        """Agent alert subscription event."""
        agent_id: str = strawberry.field(description="Agent ID")
        alert_type: str = strawberry.field(
            description="Alert type (warning, error, critical)"
        )
        message: str = strawberry.field(description="Alert message")
        metric_name: Optional[str] = strawberry.field(
            description="Associated metric name"
        )
        metric_value: Optional[float] = strawberry.field(
            description="Metric value that triggered alert"
        )
        timestamp: datetime = strawberry.field(
            description="Alert timestamp"
        )

    # ========================================================================
    # STRAWBERRY QUERY TYPE
    # ========================================================================

    @strawberry.type
    class Query:
        """GraphQL Query root type for Process Heat operations."""

        @strawberry.field
        async def agents(
            self,
            status: Optional[str] = None
        ) -> List[ProcessHeatAgent]:
            """Query for process heat agents.

            Args:
                status: Filter by agent status

            Returns:
                List of process heat agents
            """
            # Mock implementation - replace with actual agent registry
            agents_list = [
                ProcessHeatAgent(
                    id="agent-001",
                    name="Thermal Command",
                    agent_type="GL-001",
                    status=status or "idle",
                    enabled=True,
                    version="1.0.0",
                    last_run=datetime.now(),
                    next_run=None,
                    metrics=AgentMetricsType(
                        execution_time_ms=1234.5,
                        memory_usage_mb=256.2,
                        records_processed=15000,
                        processing_rate=1250.0,
                        cache_hit_ratio=0.85,
                        error_count=0
                    ),
                    error_message=None,
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                )
            ]

            if status:
                agents_list = [a for a in agents_list if a.status == status]

            return agents_list

        @strawberry.field
        async def agent(self, id: str) -> Optional[ProcessHeatAgent]:
            """Query a specific agent by ID.

            Args:
                id: Agent ID

            Returns:
                ProcessHeatAgent or None
            """
            # Mock implementation
            if id == "agent-001":
                return ProcessHeatAgent(
                    id=id,
                    name="Thermal Command",
                    agent_type="GL-001",
                    status="idle",
                    enabled=True,
                    version="1.0.0",
                    last_run=datetime.now(),
                    next_run=None,
                    metrics=AgentMetricsType(
                        execution_time_ms=1234.5,
                        memory_usage_mb=256.2,
                        records_processed=15000,
                        processing_rate=1250.0,
                        cache_hit_ratio=0.85,
                        error_count=0
                    ),
                    error_message=None,
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                )
            return None

        @strawberry.field
        async def emissions(
            self,
            facility_id: str,
            date_range: Optional[DateRangeInput] = None
        ) -> List[EmissionResult]:
            """Query emissions for a facility.

            Args:
                facility_id: Facility ID
                date_range: Optional date range filter

            Returns:
                List of emission results
            """
            # Mock implementation
            return [
                EmissionResult(
                    id=f"emission-{i}",
                    facility_id=facility_id,
                    co2_tonnes=1250.0 + i * 10,
                    ch4_tonnes=5.2 + i * 0.1,
                    n2o_tonnes=0.8 + i * 0.02,
                    total_co2e_tonnes=1256.5 + i * 10.2,
                    provenance_hash="sha256_hash_" + str(i).zfill(64),
                    calculation_method="IPCC AR6",
                    timestamp=datetime.now(),
                    confidence_score=0.95
                )
                for i in range(3)
            ]

        @strawberry.field
        async def jobs(
            self,
            status: Optional[str] = None
        ) -> List[CalculationJob]:
            """Query calculation jobs.

            Args:
                status: Filter by job status

            Returns:
                List of calculation jobs
            """
            # Mock implementation
            jobs_list = [
                CalculationJob(
                    id=f"job-{i}",
                    status=status or "completed",
                    progress_percent=100 if status != "running" else 75,
                    agent_id="agent-001",
                    input_summary="1 facility, 30 days",
                    results=[
                        EmissionResult(
                            id=f"result-{i}",
                            facility_id="facility-1",
                            co2_tonnes=1250.0,
                            ch4_tonnes=5.2,
                            n2o_tonnes=0.8,
                            total_co2e_tonnes=1256.5,
                            provenance_hash="hash_" + str(i).zfill(64),
                            calculation_method="IPCC",
                            timestamp=datetime.now(),
                            confidence_score=0.95
                        )
                    ],
                    error_details=None,
                    execution_time_ms=5432.1,
                    created_at=datetime.now(),
                    started_at=datetime.now(),
                    completed_at=datetime.now(),
                )
                for i in range(3)
            ]

            if status:
                jobs_list = [j for j in jobs_list if j.status == status]

            return jobs_list

        @strawberry.field
        async def compliance_reports(
            self,
            report_type: Optional[str] = None
        ) -> List[ComplianceReport]:
            """Query compliance reports.

            Args:
                report_type: Filter by report type

            Returns:
                List of compliance reports
            """
            from datetime import date as date_class
            # Mock implementation
            reports = [
                ComplianceReport(
                    id=f"report-{i}",
                    report_type=report_type or "ghg_emissions",
                    status="compliant",
                    period_start=date_class(2025, 1, 1),
                    period_end=date_class(2025, 3, 31),
                    findings=[
                        ComplianceFinding(
                            id="finding-1",
                            category="emissions_tracking",
                            severity="low",
                            description="Minor tracking deviation",
                            remediation_action="Update measurement method",
                            deadline=date_class(2025, 6, 30)
                        )
                    ],
                    summary="Facility is 98% compliant",
                    action_items_count=1,
                    generated_at=datetime.now(),
                )
                for i in range(2)
            ]

            if report_type:
                reports = [r for r in reports if r.report_type == report_type]

            return reports

    # ========================================================================
    # STRAWBERRY MUTATION TYPE
    # ========================================================================

    @strawberry.type
    class Mutation:
        """GraphQL Mutation root type for Process Heat operations."""

        @strawberry.mutation
        async def run_calculation(
            self,
            input: CalculationInput
        ) -> CalculationJob:
            """Start an asynchronous calculation job.

            Args:
                input: Calculation parameters

            Returns:
                CalculationJob instance
            """
            from uuid import uuid4

            job_id = str(uuid4())

            return CalculationJob(
                id=job_id,
                status="pending",
                progress_percent=0,
                agent_id=input.agent_id,
                input_summary=f"Facility {input.facility_id}, date range provided",
                results=None,
                error_details=None,
                execution_time_ms=0.0,
                created_at=datetime.now(),
                started_at=None,
                completed_at=None,
            )

        @strawberry.mutation
        async def update_agent_config(
            self,
            id: str,
            config: AgentConfigInput
        ) -> ProcessHeatAgent:
            """Update agent configuration.

            Args:
                id: Agent ID
                config: New configuration

            Returns:
                Updated ProcessHeatAgent
            """
            return ProcessHeatAgent(
                id=id,
                name="Thermal Command",
                agent_type="GL-001",
                status="idle",
                enabled=config.enabled if config.enabled is not None else True,
                version="1.0.0",
                last_run=datetime.now(),
                next_run=None,
                metrics=AgentMetricsType(
                    execution_time_ms=1234.5,
                    memory_usage_mb=256.2,
                    records_processed=15000,
                    processing_rate=1250.0,
                    cache_hit_ratio=0.85,
                    error_count=0
                ),
                error_message=None,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )

        @strawberry.mutation
        async def generate_report(
            self,
            report_type: str,
            params: ReportParamsInput
        ) -> ComplianceReport:
            """Generate a compliance report.

            Args:
                report_type: Type of report to generate
                params: Report generation parameters

            Returns:
                ComplianceReport instance
            """
            from uuid import uuid4

            report_id = str(uuid4())

            return ComplianceReport(
                id=report_id,
                report_type=report_type,
                status="compliant",
                period_start=params.date_range.start_date,
                period_end=params.date_range.end_date,
                findings=[],
                summary="Report generated successfully",
                action_items_count=0,
                generated_at=datetime.now(),
            )

    # ========================================================================
    # STRAWBERRY SUBSCRIPTION TYPE
    # ========================================================================

    @strawberry.type
    class Subscription:
        """GraphQL Subscription root type for real-time updates."""

        @strawberry.subscription
        async def job_progress(
            self,
            job_id: str
        ) -> JobProgressEvent:
            """Subscribe to job progress updates.

            Args:
                job_id: Job ID to monitor

            Yields:
                JobProgressEvent instances
            """
            # Mock implementation - yields periodic updates
            for progress in range(0, 101, 25):
                yield JobProgressEvent(
                    job_id=job_id,
                    progress_percent=progress,
                    status="running" if progress < 100 else "completed",
                    message=f"Processing at {progress}%",
                    timestamp=datetime.now(),
                )
                await asyncio.sleep(1)

        @strawberry.subscription
        async def agent_alerts(
            self,
            agent_ids: List[str]
        ) -> AlertEvent:
            """Subscribe to agent alerts.

            Args:
                agent_ids: Agent IDs to monitor

            Yields:
                AlertEvent instances
            """
            # Mock implementation - yields sample alerts
            alerts = [
                AlertEvent(
                    agent_id=agent_ids[0] if agent_ids else "agent-001",
                    alert_type="warning",
                    message="High memory usage detected",
                    metric_name="memory_usage_mb",
                    metric_value=512.5,
                    timestamp=datetime.now(),
                ),
                AlertEvent(
                    agent_id=agent_ids[0] if agent_ids else "agent-001",
                    alert_type="error",
                    message="Calculation error in emission estimation",
                    metric_name="error_count",
                    metric_value=1.0,
                    timestamp=datetime.now(),
                ),
            ]

            for alert in alerts:
                yield alert
                await asyncio.sleep(2)


def create_process_heat_schema() -> Any:
    """Create and return the Process Heat GraphQL schema.

    Returns:
        Strawberry Schema instance for Process Heat agents

    Raises:
        ImportError: If Strawberry is not installed
    """
    if not STRAWBERRY_AVAILABLE or strawberry is None:
        raise ImportError(
            "Strawberry is required for GraphQL. "
            "Install with: pip install strawberry-graphql[fastapi]"
        )

    schema = strawberry.Schema(
        query=Query,
        mutation=Mutation,
        subscription=Subscription,
    )

    logger.info(
        "Process Heat GraphQL schema created successfully with "
        "Query, Mutation, and Subscription types"
    )

    return schema
