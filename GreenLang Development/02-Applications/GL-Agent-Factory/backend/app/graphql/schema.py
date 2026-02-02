"""
GreenLang Process Heat GraphQL Schema

Complete GraphQL schema for all 143 Process Heat agents.
Integrates with the agent registry for queries, mutations, and subscriptions.

Features:
- Query all Process Heat agents by ID, category, type, priority
- Execute agents with validated input
- Configure agent parameters
- Real-time subscriptions for events and progress
- Full provenance tracking for compliance
- JWT/API key authentication
- Request logging and audit trails

Example:
    >>> from app.graphql import create_graphql_app
    >>> graphql_app = create_graphql_app()
    >>> app.include_router(graphql_app, prefix="/graphql")

GraphQL Schema:
    type Query {
        agent(id: String!): Agent
        agents(category: String, type: String): [Agent!]!
        agentHealth(id: String!): HealthStatus
        calculation(id: String!): CalculationResult
        registryStats: RegistryStats
        searchAgents(query: String!, limit: Int): [Agent!]!
    }

    type Mutation {
        runAgent(id: String!, input: JSON!): CalculationResult!
        configureAgent(id: String!, config: JSON!): Agent!
    }

    type Subscription {
        agentEvents(agentId: String): AgentEvent!
        calculationProgress(calculationId: String!): Progress!
        systemEvents: SystemEvent!
    }
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import AsyncGenerator, List, Optional

import strawberry
from strawberry import Schema
from strawberry.fastapi import GraphQLRouter
from strawberry.scalars import JSON
from strawberry.schema.config import StrawberryConfig
from strawberry.types import Info

# Import types
from app.graphql.types.agent import (
    ProcessHeatAgentType,
    AgentStatusEnum,
    HealthStatusType,
    AgentConnection,
    AgentFilterInput,
    RegistryStatsType,
    PageInfo,
)
from app.graphql.types.calculation import (
    CalculationResultType,
    CalculationInputType,
)
from app.graphql.types.events import (
    AgentEventType,
    CalculationProgressType,
    SystemEventType,
    GenericEventType,
    EventFilterInput,
)

# Import resolvers
from app.graphql.resolvers.agents import (
    get_agent,
    get_agents,
    get_agent_health,
    run_agent,
    configure_agent,
    get_registry_stats,
    search_agents,
    get_calculation,
)
from app.graphql.resolvers.subscriptions import (
    agent_events_generator,
    calculation_progress_generator,
    system_events_generator,
)

# Import middleware
from app.graphql.middleware.auth import (
    AuthMiddleware,
    AuthContext,
    get_context_with_auth,
    Permission,
    require_permission,
)
from app.graphql.middleware.logging import (
    LoggingMiddleware,
    get_metrics,
)

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


# =============================================================================
# GraphQL Context
# =============================================================================


class GreenLangContext:
    """
    GraphQL context containing request state.

    Provides:
    - Authentication context
    - Tenant information
    - Request metadata
    """

    def __init__(
        self,
        request,
        auth: Optional[AuthContext] = None,
    ):
        self.request = request
        self.auth = auth or AuthContext()

    @property
    def tenant_id(self) -> str:
        return self.auth.tenant_id

    @property
    def user_id(self) -> Optional[str]:
        return self.auth.user_id

    @property
    def authenticated(self) -> bool:
        return self.auth.authenticated


async def get_context(request) -> GreenLangContext:
    """
    Create GraphQL context from HTTP request.

    Extracts authentication and sets up request context.
    """
    auth_context = await get_context_with_auth(request)
    return GreenLangContext(request=request, auth=auth_context)


# =============================================================================
# Query Type
# =============================================================================


@strawberry.type
class Query:
    """
    Root query type for GreenLang Process Heat GraphQL API.

    Provides access to all 143 Process Heat agents, calculations,
    and system information.
    """

    @strawberry.field(description="Get a single agent by ID (e.g., 'GL-022')")
    async def agent(
        self,
        info: Info,
        id: str,
    ) -> Optional[ProcessHeatAgentType]:
        """
        Get a single Process Heat agent by ID.

        Args:
            id: Agent ID (e.g., "GL-022") or name (e.g., "SUPERHEAT-CTRL")

        Returns:
            ProcessHeatAgentType or None if not found

        Example:
            query {
                agent(id: "GL-022") {
                    id
                    name
                    category
                    type
                    status
                    healthScore
                }
            }
        """
        return await get_agent(info, id)

    @strawberry.field(description="List agents with optional filtering")
    async def agents(
        self,
        info: Info,
        category: Optional[str] = None,
        type: Optional[str] = None,
        filters: Optional[AgentFilterInput] = None,
        first: int = 20,
        after: Optional[str] = None,
    ) -> AgentConnection:
        """
        Get paginated list of Process Heat agents.

        Args:
            category: Filter by category (e.g., "Steam Systems")
            type: Filter by agent type (e.g., "Optimizer")
            filters: Additional filter criteria
            first: Number of items to return (max 100)
            after: Cursor for pagination

        Returns:
            Paginated connection of agents

        Example:
            query {
                agents(category: "Steam Systems", first: 10) {
                    edges {
                        node {
                            id
                            name
                            status
                        }
                        cursor
                    }
                    pageInfo {
                        hasNextPage
                        endCursor
                        totalCount
                    }
                }
            }
        """
        # Limit first to prevent excessive queries
        first = min(first, 100)
        return await get_agents(
            info,
            category=category,
            type=type,
            filters=filters,
            first=first,
            after=after,
        )

    @strawberry.field(description="Get agent health status")
    async def agent_health(
        self,
        info: Info,
        id: str,
    ) -> Optional[HealthStatusType]:
        """
        Get detailed health status for an agent.

        Args:
            id: Agent ID

        Returns:
            HealthStatusType with health metrics

        Example:
            query {
                agentHealth(id: "GL-022") {
                    level
                    score
                    lastCheck
                    responseTimeMs
                    errorRate
                    availability
                }
            }
        """
        return await get_agent_health(info, id)

    @strawberry.field(description="Get a calculation result by ID")
    async def calculation(
        self,
        info: Info,
        id: str,
    ) -> Optional[CalculationResultType]:
        """
        Get a calculation result by ID.

        Args:
            id: Calculation ID

        Returns:
            CalculationResultType with results and provenance

        Example:
            query {
                calculation(id: "calc-abc123") {
                    id
                    status
                    result {
                        value
                        unit
                    }
                    provenance {
                        inputHash
                        outputHash
                        chainHash
                    }
                }
            }
        """
        return await get_calculation(info, id)

    @strawberry.field(description="Get registry statistics")
    async def registry_stats(self, info: Info) -> RegistryStatsType:
        """
        Get overall statistics for the agent registry.

        Returns:
            Statistics including agent counts by category, type, priority

        Example:
            query {
                registryStats {
                    totalAgents
                    byCategory {
                        category
                        count
                    }
                    totalAddressableMarketBillions
                }
            }
        """
        return await get_registry_stats(info)

    @strawberry.field(description="Search agents by text")
    async def search_agents(
        self,
        info: Info,
        query: str,
        limit: int = 20,
    ) -> List[ProcessHeatAgentType]:
        """
        Full-text search for agents.

        Searches agent names, IDs, categories, and descriptions.

        Args:
            query: Search query
            limit: Maximum results (max 100)

        Returns:
            List of matching agents

        Example:
            query {
                searchAgents(query: "steam optimizer", limit: 10) {
                    id
                    name
                    category
                    description
                }
            }
        """
        limit = min(limit, 100)
        return await search_agents(info, query, limit)

    @strawberry.field(description="Get API metrics")
    async def metrics(self, info: Info) -> JSON:
        """
        Get current API metrics.

        Returns metrics including request counts, latencies, and error rates.
        """
        return get_metrics().to_dict()


# =============================================================================
# Mutation Type
# =============================================================================


@strawberry.type
class MutationResult:
    """Generic mutation result."""
    success: bool
    message: Optional[str] = None
    error: Optional[str] = None


@strawberry.type
class AgentMutationResult:
    """Result of an agent mutation."""
    success: bool
    agent: Optional[ProcessHeatAgentType] = None
    error: Optional[str] = None


@strawberry.type
class CalculationMutationResult:
    """Result of a calculation mutation."""
    success: bool
    calculation: Optional[CalculationResultType] = None
    error: Optional[str] = None


@strawberry.type
class Mutation:
    """
    Root mutation type for GreenLang Process Heat GraphQL API.

    Provides operations to execute and configure agents.
    """

    @strawberry.mutation(description="Execute an agent with input data")
    async def run_agent(
        self,
        info: Info,
        id: str,
        input: JSON,
    ) -> CalculationMutationResult:
        """
        Execute a Process Heat agent.

        Args:
            id: Agent ID to execute (e.g., "GL-022")
            input: Agent input parameters as JSON

        Returns:
            CalculationMutationResult with execution results

        Example:
            mutation {
                runAgent(
                    id: "GL-022",
                    input: {
                        "steamPressure": 150,
                        "feedwaterTemp": 220,
                        "fuelType": "natural_gas"
                    }
                ) {
                    success
                    calculation {
                        id
                        status
                        result {
                            value
                            unit
                        }
                    }
                    error
                }
            }
        """
        try:
            result = await run_agent(info, id, input)
            return CalculationMutationResult(
                success=True,
                calculation=result,
                error=None,
            )
        except Exception as e:
            logger.error(f"run_agent mutation failed: {e}")
            return CalculationMutationResult(
                success=False,
                calculation=None,
                error=str(e),
            )

    @strawberry.mutation(description="Configure an agent")
    async def configure_agent(
        self,
        info: Info,
        id: str,
        config: JSON,
    ) -> AgentMutationResult:
        """
        Configure an agent's parameters.

        Args:
            id: Agent ID to configure
            config: Configuration parameters as JSON

        Returns:
            AgentMutationResult with updated agent

        Example:
            mutation {
                configureAgent(
                    id: "GL-022",
                    config: {
                        "timeout_seconds": 600,
                        "retry_count": 5,
                        "cache_enabled": true
                    }
                ) {
                    success
                    agent {
                        id
                        name
                    }
                    error
                }
            }
        """
        try:
            agent = await configure_agent(info, id, config)
            return AgentMutationResult(
                success=True,
                agent=agent,
                error=None,
            )
        except Exception as e:
            logger.error(f"configure_agent mutation failed: {e}")
            return AgentMutationResult(
                success=False,
                agent=None,
                error=str(e),
            )


# =============================================================================
# Subscription Type
# =============================================================================


@strawberry.type
class Subscription:
    """
    Root subscription type for GreenLang Process Heat GraphQL API.

    Provides real-time streaming of agent events and calculation progress.
    """

    @strawberry.subscription(description="Subscribe to agent events")
    async def agent_events(
        self,
        info: Info,
        agent_id: Optional[str] = None,
    ) -> AsyncGenerator[AgentEventType, None]:
        """
        Subscribe to real-time agent events.

        Args:
            agent_id: Optional agent ID to filter events

        Yields:
            AgentEventType instances as events occur

        Example:
            subscription {
                agentEvents(agentId: "GL-022") {
                    eventId
                    eventType
                    timestamp
                    message
                    data
                }
            }
        """
        async for event in agent_events_generator(info, agent_id):
            yield event

    @strawberry.subscription(description="Subscribe to calculation progress")
    async def calculation_progress(
        self,
        info: Info,
        calculation_id: str,
    ) -> AsyncGenerator[CalculationProgressType, None]:
        """
        Subscribe to real-time calculation progress updates.

        Args:
            calculation_id: Calculation ID to track

        Yields:
            CalculationProgressType instances with progress updates

        Example:
            subscription {
                calculationProgress(calculationId: "calc-abc123") {
                    calculationId
                    status
                    progress {
                        percent
                        currentStep
                        estimatedRemainingSeconds
                    }
                    intermediateValue
                    intermediateUnit
                }
            }
        """
        async for progress in calculation_progress_generator(info, calculation_id):
            yield progress

    @strawberry.subscription(description="Subscribe to system events")
    async def system_events(
        self,
        info: Info,
        filters: Optional[EventFilterInput] = None,
    ) -> AsyncGenerator[SystemEventType, None]:
        """
        Subscribe to system health and status events.

        Args:
            filters: Optional event filters

        Yields:
            SystemEventType instances for system events

        Example:
            subscription {
                systemEvents {
                    eventId
                    eventType
                    component
                    healthScore
                    message
                }
            }
        """
        async for event in system_events_generator(info, filters):
            yield event


# =============================================================================
# Schema Extensions
# =============================================================================


class QueryComplexityExtension:
    """Extension to limit query complexity."""

    MAX_COMPLEXITY = 1000

    def on_operation(self) -> None:
        """Check query complexity before execution."""
        # Simplified - in production would analyze AST
        pass


class QueryDepthExtension:
    """Extension to limit query depth."""

    MAX_DEPTH = 10

    def on_operation(self) -> None:
        """Check query depth before execution."""
        # Simplified - in production would analyze AST
        pass


# =============================================================================
# Schema Creation
# =============================================================================


# Create the GraphQL schema
schema = Schema(
    query=Query,
    mutation=Mutation,
    subscription=Subscription,
    config=StrawberryConfig(auto_camel_case=True),
    extensions=[
        LoggingMiddleware,
    ],
)


def create_graphql_app(
    graphiql: bool = True,
    subscription_protocols: List[str] = None,
) -> GraphQLRouter:
    """
    Create the GraphQL FastAPI router.

    Args:
        graphiql: Enable GraphiQL IDE (default True)
        subscription_protocols: WebSocket protocols for subscriptions

    Returns:
        Configured GraphQLRouter

    Example:
        >>> from app.graphql.schema import create_graphql_app
        >>> graphql_app = create_graphql_app()
        >>> app.include_router(graphql_app, prefix="/graphql")
    """
    if subscription_protocols is None:
        subscription_protocols = ["graphql-ws", "graphql-transport-ws"]

    return GraphQLRouter(
        schema,
        context_getter=get_context,
        graphiql=graphiql,
        subscription_protocols=subscription_protocols,
    )


# =============================================================================
# Schema Export for Tools
# =============================================================================


def export_schema_sdl() -> str:
    """
    Export the GraphQL schema in SDL format.

    Returns:
        Schema Definition Language string

    Example:
        >>> sdl = export_schema_sdl()
        >>> with open("schema.graphql", "w") as f:
        ...     f.write(sdl)
    """
    return str(schema)


def get_schema() -> Schema:
    """
    Get the GraphQL schema instance.

    Returns:
        Strawberry Schema instance
    """
    return schema


# =============================================================================
# Health Check
# =============================================================================


async def graphql_health_check() -> dict:
    """
    Perform health check on GraphQL service.

    Returns:
        Health status dictionary
    """
    return {
        "status": "healthy",
        "schema_loaded": schema is not None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics": get_metrics().to_dict(),
    }
