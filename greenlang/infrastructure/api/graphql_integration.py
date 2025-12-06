"""
GraphQL Integration for Process Heat Agents

This module provides FastAPI integration for the Process Heat GraphQL schema,
including subscription support via WebSockets and query execution utilities.

Features:
- FastAPI GraphQL endpoint
- WebSocket subscription support
- Query/mutation execution helpers
- Error handling and logging
- Authentication integration points

Example:
    >>> from fastapi import FastAPI
    >>> from greenlang.infrastructure.api.graphql_integration import setup_graphql
    >>>
    >>> app = FastAPI()
    >>> setup_graphql(app)
    >>>
    >>> # Query example:
    >>> # query {
    >>> #   agents(status: "idle") {
    >>> #     id
    >>> #     name
    >>> #     status
    >>> #     metrics {
    >>> #       execution_time_ms
    >>> #       memory_usage_mb
    >>> #     }
    >>> #   }
    >>> # }
"""

import logging
import json
from typing import Any, Dict, Optional, Callable, Awaitable
from datetime import datetime

try:
    from fastapi import FastAPI, WebSocket, HTTPException, Depends
    from strawberry.fastapi import GraphQLRouter
    from strawberry import Schema
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = None
    WebSocket = None
    HTTPException = Exception
    GraphQLRouter = None
    Schema = None

from greenlang.infrastructure.api.graphql_schema import (
    create_process_heat_schema,
    STRAWBERRY_AVAILABLE,
)

logger = logging.getLogger(__name__)


class GraphQLIntegrationError(Exception):
    """Raised when GraphQL integration fails."""
    pass


class QueryExecutor:
    """
    Executes GraphQL queries and mutations.

    Provides high-level query execution with error handling and logging.

    Attributes:
        schema: Strawberry GraphQL schema
    """

    def __init__(self, schema: Any):
        """
        Initialize query executor.

        Args:
            schema: Strawberry Schema instance
        """
        self.schema = schema
        logger.debug("QueryExecutor initialized")

    async def execute_query(
        self,
        query: str,
        variables: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a GraphQL query.

        Args:
            query: GraphQL query string
            variables: Query variables
            context: Execution context

        Returns:
            Query result with data and errors

        Raises:
            GraphQLIntegrationError: If query execution fails
        """
        try:
            logger.debug(f"Executing query: {query[:100]}...")

            result = await self.schema.execute(
                query,
                variable_values=variables,
                context_value=context
            )

            if result.errors:
                logger.warning(f"Query execution errors: {result.errors}")
                return {
                    "data": result.data,
                    "errors": [str(e) for e in result.errors]
                }

            logger.debug(f"Query executed successfully")
            return {"data": result.data}

        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}", exc_info=True)
            raise GraphQLIntegrationError(f"Query execution failed: {str(e)}") from e

    async def execute_mutation(
        self,
        mutation: str,
        variables: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a GraphQL mutation.

        Args:
            mutation: GraphQL mutation string
            variables: Mutation variables
            context: Execution context

        Returns:
            Mutation result with data and errors

        Raises:
            GraphQLIntegrationError: If mutation execution fails
        """
        return await self.execute_query(mutation, variables, context)


class SubscriptionHandler:
    """
    Handles GraphQL subscription connections.

    Manages WebSocket subscriptions with cleanup and error handling.

    Attributes:
        schema: Strawberry GraphQL schema
    """

    def __init__(self, schema: Any):
        """
        Initialize subscription handler.

        Args:
            schema: Strawberry Schema instance
        """
        self.schema = schema
        self._subscriptions: Dict[str, Any] = {}
        logger.debug("SubscriptionHandler initialized")

    async def handle_subscription(
        self,
        websocket: WebSocket,
        subscription_id: str,
        query: str,
        variables: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Handle a GraphQL subscription connection.

        Args:
            websocket: WebSocket connection
            subscription_id: Unique subscription ID
            query: GraphQL subscription query
            variables: Subscription variables

        Raises:
            GraphQLIntegrationError: If subscription setup fails
        """
        try:
            await websocket.accept()
            logger.info(f"Subscription {subscription_id} accepted")

            # Store subscription reference
            self._subscriptions[subscription_id] = {
                "websocket": websocket,
                "query": query,
                "variables": variables,
                "started_at": datetime.now()
            }

            # Execute subscription
            async for result in self.schema.subscribe(
                query,
                variable_values=variables
            ):
                if result.errors:
                    await websocket.send_json({
                        "id": subscription_id,
                        "type": "error",
                        "payload": [str(e) for e in result.errors]
                    })
                else:
                    await websocket.send_json({
                        "id": subscription_id,
                        "type": "data",
                        "payload": result.data
                    })

        except Exception as e:
            logger.error(
                f"Subscription {subscription_id} failed: {str(e)}",
                exc_info=True
            )
            try:
                await websocket.send_json({
                    "id": subscription_id,
                    "type": "error",
                    "payload": str(e)
                })
            except Exception:
                pass  # Connection may already be closed
            raise GraphQLIntegrationError(f"Subscription failed: {str(e)}") from e

        finally:
            # Clean up subscription
            if subscription_id in self._subscriptions:
                del self._subscriptions[subscription_id]
                logger.info(f"Subscription {subscription_id} cleaned up")

    def get_active_subscriptions(self) -> int:
        """
        Get count of active subscriptions.

        Returns:
            Number of active subscriptions
        """
        return len(self._subscriptions)


class GraphQLConfig:
    """Configuration for GraphQL integration."""

    def __init__(
        self,
        path: str = "/graphql",
        enable_schema_introspection: bool = True,
        enable_playground: bool = True,
        max_query_depth: int = 10,
        timeout_seconds: float = 30.0
    ):
        """
        Initialize GraphQL configuration.

        Args:
            path: GraphQL endpoint path
            enable_schema_introspection: Enable introspection queries
            enable_playground: Enable GraphQL playground
            max_query_depth: Maximum query depth
            timeout_seconds: Query timeout in seconds
        """
        self.path = path
        self.enable_schema_introspection = enable_schema_introspection
        self.enable_playground = enable_playground
        self.max_query_depth = max_query_depth
        self.timeout_seconds = timeout_seconds


def setup_graphql(
    app: FastAPI,
    config: Optional[GraphQLConfig] = None,
    authentication_handler: Optional[Callable[[Any], Awaitable[bool]]] = None
) -> None:
    """
    Set up GraphQL integration with FastAPI application.

    This function configures the GraphQL endpoint, enables playground,
    and sets up subscription support via WebSockets.

    Args:
        app: FastAPI application instance
        config: GraphQL configuration
        authentication_handler: Optional async function for authentication

    Raises:
        ImportError: If required libraries are not installed
        GraphQLIntegrationError: If setup fails

    Example:
        >>> from fastapi import FastAPI
        >>> from greenlang.infrastructure.api.graphql_integration import (
        ...     setup_graphql, GraphQLConfig
        ... )
        >>>
        >>> app = FastAPI()
        >>> config = GraphQLConfig(path="/graphql")
        >>> setup_graphql(app, config)
        >>>
        >>> # Start with: uvicorn main:app --reload
        >>> # Access GraphQL playground at http://localhost:8000/graphql
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError(
            "FastAPI is required for GraphQL integration. "
            "Install with: pip install fastapi strawberry-graphql[fastapi]"
        )

    if not STRAWBERRY_AVAILABLE:
        raise ImportError(
            "Strawberry is required for GraphQL. "
            "Install with: pip install strawberry-graphql[fastapi]"
        )

    config = config or GraphQLConfig()

    try:
        # Create schema
        schema = create_process_heat_schema()
        logger.info("Process Heat GraphQL schema created")

        # Create GraphQL router
        graphql_router = GraphQLRouter(
            schema,
            path=config.path,
            allow_queries_via_get=True,
            allow_mutations_via_get=False,
        )

        # Include router in app
        app.include_router(graphql_router)
        logger.info(f"GraphQL endpoint configured at {config.path}")

        # Create executor for helper usage
        executor = QueryExecutor(schema)

        # Create subscription handler
        subscription_handler = SubscriptionHandler(schema)

        # Add to app state for access in route handlers
        app.state.graphql_executor = executor
        app.state.graphql_schema = schema
        app.state.subscription_handler = subscription_handler

        # Add GraphQL status endpoint
        @app.get("/graphql/status")
        async def graphql_status() -> Dict[str, Any]:
            """Get GraphQL service status."""
            return {
                "status": "operational",
                "endpoint": config.path,
                "active_subscriptions": subscription_handler.get_active_subscriptions(),
                "introspection_enabled": config.enable_schema_introspection,
                "playground_enabled": config.enable_playground,
            }

        logger.info("GraphQL integration setup complete")

    except Exception as e:
        logger.error(f"GraphQL setup failed: {str(e)}", exc_info=True)
        raise GraphQLIntegrationError(f"GraphQL setup failed: {str(e)}") from e


async def query_agents(
    executor: QueryExecutor,
    status: Optional[str] = None
) -> Dict[str, Any]:
    """
    Query process heat agents.

    Helper function for querying agents from application code.

    Args:
        executor: GraphQL executor instance
        status: Optional status filter

    Returns:
        Query result with agents list
    """
    query = """
        query getAgents($status: String) {
            agents(status: $status) {
                id
                name
                agentType
                status
                enabled
                version
                lastRun
                metrics {
                    executionTimeMs
                    memoryUsageMb
                    recordsProcessed
                    processingRate
                    cacheHitRatio
                    errorCount
                }
            }
        }
    """

    variables = {"status": status} if status else None
    return await executor.execute_query(query, variables)


async def query_emissions(
    executor: QueryExecutor,
    facility_id: str,
    start_date: str,
    end_date: str
) -> Dict[str, Any]:
    """
    Query emissions for a facility.

    Helper function for querying emissions from application code.

    Args:
        executor: GraphQL executor instance
        facility_id: Target facility ID
        start_date: Start date (ISO format)
        end_date: End date (ISO format)

    Returns:
        Query result with emissions list
    """
    query = """
        query getEmissions($facilityId: String!, $dateRange: DateRangeInput!) {
            emissions(facilityId: $facilityId, dateRange: $dateRange) {
                id
                facilityId
                co2Tonnes
                ch4Tonnes
                n2oTonnes
                totalCo2eTonnes
                provenanceHash
                calculationMethod
                timestamp
                confidenceScore
            }
        }
    """

    variables = {
        "facilityId": facility_id,
        "dateRange": {
            "startDate": start_date,
            "endDate": end_date
        }
    }

    return await executor.execute_query(query, variables)


async def run_calculation(
    executor: QueryExecutor,
    agent_id: str,
    facility_id: str,
    start_date: str,
    end_date: str,
    priority: str = "normal"
) -> Dict[str, Any]:
    """
    Start a calculation job mutation.

    Helper function for starting jobs from application code.

    Args:
        executor: GraphQL executor instance
        agent_id: Target agent ID
        facility_id: Target facility ID
        start_date: Start date (ISO format)
        end_date: End date (ISO format)
        priority: Job priority (low, normal, high)

    Returns:
        Mutation result with job details
    """
    mutation = """
        mutation startCalculation($input: CalculationInput!) {
            runCalculation(input: $input) {
                id
                status
                progressPercent
                agentId
                inputSummary
                executionTimeMs
                createdAt
            }
        }
    """

    variables = {
        "input": {
            "agentId": agent_id,
            "facilityId": facility_id,
            "dateRange": {
                "startDate": start_date,
                "endDate": end_date
            },
            "priority": priority
        }
    }

    return await executor.execute_mutation(mutation, variables)


__all__ = [
    "GraphQLIntegrationError",
    "QueryExecutor",
    "SubscriptionHandler",
    "GraphQLConfig",
    "setup_graphql",
    "query_agents",
    "query_emissions",
    "run_calculation",
]
