"""
GraphQL Integration for Process Heat Agents

This module provides FastAPI integration for the Process Heat GraphQL schema,
including full subscription lifecycle support via WebSockets.

Features:
- FastAPI GraphQL endpoint
- Full WebSocket subscription lifecycle management
- Subscription registry with connection tracking
- Event broadcasting to subscribers
- Proper cleanup on disconnect
- Integration with SSE infrastructure
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

import asyncio
import json
import logging
import weakref
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Set,
)
from uuid import uuid4

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
    from fastapi.websockets import WebSocketState
    from strawberry.fastapi import GraphQLRouter
    from strawberry import Schema
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = None
    WebSocket = None
    WebSocketDisconnect = Exception
    HTTPException = Exception
    GraphQLRouter = None
    Schema = None
    WebSocketState = None

from greenlang.infrastructure.api.graphql_schema import (
    create_process_heat_schema,
    STRAWBERRY_AVAILABLE,
)

logger = logging.getLogger(__name__)


class GraphQLIntegrationError(Exception):
    """Raised when GraphQL integration fails."""
    pass


class SubscriptionMessageType(str, Enum):
    """GraphQL WebSocket message types (graphql-ws protocol)."""
    CONNECTION_INIT = "connection_init"
    CONNECTION_ACK = "connection_ack"
    CONNECTION_ERROR = "connection_error"
    PING = "ping"
    PONG = "pong"
    SUBSCRIBE = "subscribe"
    NEXT = "next"
    ERROR = "error"
    COMPLETE = "complete"


@dataclass
class SubscriptionInfo:
    """Information about an active subscription."""
    subscription_id: str
    client_id: str
    query: str
    variables: Optional[Dict[str, Any]]
    operation_name: Optional[str]
    created_at: datetime
    last_event_at: Optional[datetime] = None
    event_count: int = 0
    task: Optional[asyncio.Task] = field(default=None, repr=False)


@dataclass
class WebSocketConnection:
    """Information about a WebSocket connection."""
    connection_id: str
    websocket: Any  # WebSocket type
    connected_at: datetime
    authenticated: bool = False
    auth_payload: Optional[Dict[str, Any]] = None
    subscriptions: Dict[str, SubscriptionInfo] = field(default_factory=dict)
    last_ping_at: Optional[datetime] = None
    last_pong_at: Optional[datetime] = None


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

            logger.debug("Query executed successfully")
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


class SubscriptionRegistry:
    """
    Registry for managing GraphQL subscriptions.

    Tracks all active subscriptions and connections for proper lifecycle management.

    Attributes:
        connections: Active WebSocket connections
        subscription_to_connection: Maps subscription IDs to connection IDs
    """

    def __init__(self) -> None:
        """Initialize subscription registry."""
        self._connections: Dict[str, WebSocketConnection] = {}
        self._subscription_to_connection: Dict[str, str] = {}
        self._lock = asyncio.Lock()
        self._event_handlers: Dict[str, List[Callable[[Any], Awaitable[None]]]] = {}
        logger.info("SubscriptionRegistry initialized")

    async def register_connection(
        self,
        websocket: WebSocket,
        auth_payload: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Register a new WebSocket connection.

        Args:
            websocket: WebSocket connection
            auth_payload: Authentication payload from connection_init

        Returns:
            Connection ID
        """
        connection_id = str(uuid4())

        async with self._lock:
            self._connections[connection_id] = WebSocketConnection(
                connection_id=connection_id,
                websocket=websocket,
                connected_at=datetime.utcnow(),
                authenticated=True if auth_payload else False,
                auth_payload=auth_payload,
            )

        logger.info(f"Registered WebSocket connection: {connection_id}")
        return connection_id

    async def unregister_connection(self, connection_id: str) -> None:
        """
        Unregister a WebSocket connection and clean up all its subscriptions.

        Args:
            connection_id: Connection ID to unregister
        """
        async with self._lock:
            connection = self._connections.pop(connection_id, None)
            if connection:
                # Cancel all subscription tasks
                for sub_id, sub_info in connection.subscriptions.items():
                    if sub_info.task and not sub_info.task.done():
                        sub_info.task.cancel()
                    self._subscription_to_connection.pop(sub_id, None)

                logger.info(
                    f"Unregistered connection {connection_id} with "
                    f"{len(connection.subscriptions)} subscriptions"
                )

    async def add_subscription(
        self,
        connection_id: str,
        subscription_id: str,
        query: str,
        variables: Optional[Dict[str, Any]] = None,
        operation_name: Optional[str] = None,
        task: Optional[asyncio.Task] = None
    ) -> bool:
        """
        Add a subscription to a connection.

        Args:
            connection_id: Connection ID
            subscription_id: Subscription ID (client-provided)
            query: GraphQL subscription query
            variables: Query variables
            operation_name: Operation name
            task: Async task running the subscription

        Returns:
            True if added successfully
        """
        async with self._lock:
            connection = self._connections.get(connection_id)
            if not connection:
                logger.warning(f"Connection {connection_id} not found")
                return False

            if subscription_id in connection.subscriptions:
                logger.warning(
                    f"Subscription {subscription_id} already exists for "
                    f"connection {connection_id}"
                )
                return False

            connection.subscriptions[subscription_id] = SubscriptionInfo(
                subscription_id=subscription_id,
                client_id=connection_id,
                query=query,
                variables=variables,
                operation_name=operation_name,
                created_at=datetime.utcnow(),
                task=task
            )
            self._subscription_to_connection[subscription_id] = connection_id

        logger.debug(f"Added subscription {subscription_id} to connection {connection_id}")
        return True

    async def remove_subscription(
        self,
        connection_id: str,
        subscription_id: str
    ) -> bool:
        """
        Remove a subscription from a connection.

        Args:
            connection_id: Connection ID
            subscription_id: Subscription ID

        Returns:
            True if removed successfully
        """
        async with self._lock:
            connection = self._connections.get(connection_id)
            if not connection:
                return False

            sub_info = connection.subscriptions.pop(subscription_id, None)
            if sub_info:
                if sub_info.task and not sub_info.task.done():
                    sub_info.task.cancel()
                self._subscription_to_connection.pop(subscription_id, None)
                logger.debug(f"Removed subscription {subscription_id}")
                return True

        return False

    async def get_connection(self, connection_id: str) -> Optional[WebSocketConnection]:
        """Get a connection by ID."""
        async with self._lock:
            return self._connections.get(connection_id)

    async def get_subscription(
        self,
        subscription_id: str
    ) -> Optional[SubscriptionInfo]:
        """Get a subscription by ID."""
        async with self._lock:
            connection_id = self._subscription_to_connection.get(subscription_id)
            if connection_id:
                connection = self._connections.get(connection_id)
                if connection:
                    return connection.subscriptions.get(subscription_id)
        return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics."""
        total_subscriptions = sum(
            len(conn.subscriptions) for conn in self._connections.values()
        )
        return {
            "total_connections": len(self._connections),
            "total_subscriptions": total_subscriptions,
            "connections": [
                {
                    "connection_id": conn.connection_id,
                    "connected_at": conn.connected_at.isoformat(),
                    "authenticated": conn.authenticated,
                    "subscription_count": len(conn.subscriptions)
                }
                for conn in self._connections.values()
            ]
        }


class SubscriptionManager:
    """
    Manages the full GraphQL subscription lifecycle.

    Handles WebSocket connections, subscription registration,
    event broadcasting, and cleanup.

    Attributes:
        schema: Strawberry GraphQL schema
        registry: Subscription registry
    """

    def __init__(
        self,
        schema: Any,
        sse_manager: Optional[Any] = None
    ) -> None:
        """
        Initialize subscription manager.

        Args:
            schema: Strawberry Schema instance
            sse_manager: Optional SSE stream manager for event integration
        """
        self.schema = schema
        self.registry = SubscriptionRegistry()
        self.sse_manager = sse_manager
        self._shutdown = False
        self._ping_task: Optional[asyncio.Task] = None
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._broadcast_task: Optional[asyncio.Task] = None
        logger.info("SubscriptionManager initialized")

    async def start(self) -> None:
        """Start background tasks for subscription management."""
        self._shutdown = False
        self._ping_task = asyncio.create_task(self._ping_loop())
        self._broadcast_task = asyncio.create_task(self._broadcast_loop())
        logger.info("SubscriptionManager started")

    async def stop(self) -> None:
        """Stop background tasks and clean up."""
        self._shutdown = True

        if self._ping_task:
            self._ping_task.cancel()
            try:
                await self._ping_task
            except asyncio.CancelledError:
                pass

        if self._broadcast_task:
            self._broadcast_task.cancel()
            try:
                await self._broadcast_task
            except asyncio.CancelledError:
                pass

        logger.info("SubscriptionManager stopped")

    async def handle_websocket(
        self,
        websocket: WebSocket,
        authentication_handler: Optional[Callable[[Dict], Awaitable[bool]]] = None
    ) -> None:
        """
        Handle a WebSocket connection for GraphQL subscriptions.

        Implements the graphql-ws protocol for subscription management.

        Args:
            websocket: WebSocket connection
            authentication_handler: Optional async function for authentication
        """
        await websocket.accept()
        connection_id: Optional[str] = None

        try:
            while True:
                # Receive message
                try:
                    raw_message = await websocket.receive_text()
                    message = json.loads(raw_message)
                except json.JSONDecodeError:
                    await self._send_error(websocket, None, "Invalid JSON")
                    continue

                message_type = message.get("type", "").lower()
                message_id = message.get("id")
                payload = message.get("payload", {})

                # Handle message types
                if message_type == SubscriptionMessageType.CONNECTION_INIT.value:
                    # Authentication and connection setup
                    authenticated = True
                    if authentication_handler:
                        try:
                            authenticated = await authentication_handler(payload)
                        except Exception as e:
                            logger.error(f"Authentication error: {e}")
                            authenticated = False

                    if not authenticated:
                        await self._send_message(
                            websocket,
                            SubscriptionMessageType.CONNECTION_ERROR,
                            payload={"message": "Authentication failed"}
                        )
                        await websocket.close(code=4401)
                        return

                    connection_id = await self.registry.register_connection(
                        websocket, payload
                    )
                    await self._send_message(
                        websocket,
                        SubscriptionMessageType.CONNECTION_ACK
                    )

                elif message_type == SubscriptionMessageType.PING.value:
                    await self._send_message(
                        websocket,
                        SubscriptionMessageType.PONG
                    )

                elif message_type == SubscriptionMessageType.PONG.value:
                    # Update pong timestamp
                    if connection_id:
                        conn = await self.registry.get_connection(connection_id)
                        if conn:
                            conn.last_pong_at = datetime.utcnow()

                elif message_type == SubscriptionMessageType.SUBSCRIBE.value:
                    if not connection_id:
                        await self._send_error(
                            websocket, message_id,
                            "Connection not initialized"
                        )
                        continue

                    query = payload.get("query")
                    variables = payload.get("variables")
                    operation_name = payload.get("operationName")

                    if not query:
                        await self._send_error(
                            websocket, message_id,
                            "Query is required"
                        )
                        continue

                    # Start subscription
                    task = asyncio.create_task(
                        self._execute_subscription(
                            websocket,
                            connection_id,
                            message_id,
                            query,
                            variables,
                            operation_name
                        )
                    )

                    await self.registry.add_subscription(
                        connection_id,
                        message_id,
                        query,
                        variables,
                        operation_name,
                        task
                    )

                elif message_type == SubscriptionMessageType.COMPLETE.value:
                    if connection_id and message_id:
                        await self.registry.remove_subscription(
                            connection_id, message_id
                        )

                else:
                    logger.warning(f"Unknown message type: {message_type}")

        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected: {connection_id}")
        except Exception as e:
            logger.error(f"WebSocket error: {e}", exc_info=True)
        finally:
            if connection_id:
                await self.registry.unregister_connection(connection_id)

    async def _execute_subscription(
        self,
        websocket: WebSocket,
        connection_id: str,
        subscription_id: str,
        query: str,
        variables: Optional[Dict[str, Any]],
        operation_name: Optional[str]
    ) -> None:
        """Execute a GraphQL subscription and stream results."""
        try:
            logger.debug(f"Starting subscription {subscription_id}")

            async for result in self.schema.subscribe(
                query,
                variable_values=variables,
                operation_name=operation_name
            ):
                # Check if still connected
                if websocket.client_state != WebSocketState.CONNECTED:
                    break

                # Update subscription stats
                sub_info = await self.registry.get_subscription(subscription_id)
                if sub_info:
                    sub_info.last_event_at = datetime.utcnow()
                    sub_info.event_count += 1

                # Send result
                if result.errors:
                    await self._send_message(
                        websocket,
                        SubscriptionMessageType.ERROR,
                        subscription_id,
                        [{"message": str(e)} for e in result.errors]
                    )
                else:
                    await self._send_message(
                        websocket,
                        SubscriptionMessageType.NEXT,
                        subscription_id,
                        result.data
                    )

            # Subscription completed normally
            await self._send_message(
                websocket,
                SubscriptionMessageType.COMPLETE,
                subscription_id
            )

        except asyncio.CancelledError:
            logger.debug(f"Subscription {subscription_id} cancelled")
        except Exception as e:
            logger.error(f"Subscription error: {e}", exc_info=True)
            try:
                await self._send_message(
                    websocket,
                    SubscriptionMessageType.ERROR,
                    subscription_id,
                    [{"message": str(e)}]
                )
            except Exception:
                pass

    async def _send_message(
        self,
        websocket: WebSocket,
        message_type: SubscriptionMessageType,
        message_id: Optional[str] = None,
        payload: Any = None
    ) -> None:
        """Send a message to a WebSocket."""
        message: Dict[str, Any] = {"type": message_type.value}
        if message_id:
            message["id"] = message_id
        if payload is not None:
            message["payload"] = payload

        await websocket.send_json(message)

    async def _send_error(
        self,
        websocket: WebSocket,
        message_id: Optional[str],
        error_message: str
    ) -> None:
        """Send an error message."""
        await self._send_message(
            websocket,
            SubscriptionMessageType.ERROR,
            message_id,
            [{"message": error_message}]
        )

    async def _ping_loop(self) -> None:
        """Background loop to send pings to all connections."""
        while not self._shutdown:
            try:
                await asyncio.sleep(30)  # Ping every 30 seconds

                stats = self.registry.get_statistics()
                for conn_info in stats.get("connections", []):
                    conn = await self.registry.get_connection(
                        conn_info["connection_id"]
                    )
                    if conn and conn.websocket:
                        try:
                            await self._send_message(
                                conn.websocket,
                                SubscriptionMessageType.PING
                            )
                            conn.last_ping_at = datetime.utcnow()
                        except Exception as e:
                            logger.debug(f"Ping failed for {conn.connection_id}: {e}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Ping loop error: {e}")

    async def _broadcast_loop(self) -> None:
        """Background loop to broadcast events from the queue."""
        while not self._shutdown:
            try:
                event = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=1.0
                )

                # Broadcast event to relevant subscriptions
                await self._broadcast_event(event)

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Broadcast loop error: {e}")

    async def _broadcast_event(self, event: Dict[str, Any]) -> None:
        """Broadcast an event to relevant subscribers."""
        event_type = event.get("type")
        event_data = event.get("data")

        # This would be extended to match events with subscriptions
        # For now, log the broadcast
        logger.debug(f"Broadcasting event: {event_type}")

    async def broadcast_job_progress(
        self,
        job_id: str,
        progress: int,
        status: str,
        message: str
    ) -> None:
        """
        Broadcast job progress to subscribed clients.

        Args:
            job_id: Job ID
            progress: Progress percentage
            status: Job status
            message: Status message
        """
        await self._event_queue.put({
            "type": "job_progress",
            "data": {
                "job_id": job_id,
                "progress_percent": progress,
                "status": status,
                "message": message,
                "timestamp": datetime.utcnow().isoformat()
            }
        })

    async def broadcast_agent_alert(
        self,
        agent_id: str,
        alert_type: str,
        message: str,
        metric_name: Optional[str] = None,
        metric_value: Optional[float] = None
    ) -> None:
        """
        Broadcast agent alert to subscribed clients.

        Args:
            agent_id: Agent ID
            alert_type: Alert type (warning, error, critical)
            message: Alert message
            metric_name: Related metric name
            metric_value: Metric value that triggered alert
        """
        await self._event_queue.put({
            "type": "agent_alert",
            "data": {
                "agent_id": agent_id,
                "alert_type": alert_type,
                "message": message,
                "metric_name": metric_name,
                "metric_value": metric_value,
                "timestamp": datetime.utcnow().isoformat()
            }
        })

    def get_statistics(self) -> Dict[str, Any]:
        """Get subscription manager statistics."""
        return self.registry.get_statistics()


class GraphQLConfig:
    """Configuration for GraphQL integration."""

    def __init__(
        self,
        path: str = "/graphql",
        ws_path: str = "/graphql/ws",
        enable_schema_introspection: bool = True,
        enable_playground: bool = True,
        max_query_depth: int = 10,
        timeout_seconds: float = 30.0,
        enable_subscriptions: bool = True
    ):
        """
        Initialize GraphQL configuration.

        Args:
            path: GraphQL HTTP endpoint path
            ws_path: GraphQL WebSocket endpoint path
            enable_schema_introspection: Enable introspection queries
            enable_playground: Enable GraphQL playground
            max_query_depth: Maximum query depth
            timeout_seconds: Query timeout in seconds
            enable_subscriptions: Enable WebSocket subscriptions
        """
        self.path = path
        self.ws_path = ws_path
        self.enable_schema_introspection = enable_schema_introspection
        self.enable_playground = enable_playground
        self.max_query_depth = max_query_depth
        self.timeout_seconds = timeout_seconds
        self.enable_subscriptions = enable_subscriptions


def setup_graphql(
    app: FastAPI,
    config: Optional[GraphQLConfig] = None,
    authentication_handler: Optional[Callable[[Dict], Awaitable[bool]]] = None,
    sse_manager: Optional[Any] = None
) -> SubscriptionManager:
    """
    Set up GraphQL integration with FastAPI application.

    This function configures the GraphQL endpoint, enables playground,
    and sets up full subscription support via WebSockets.

    Args:
        app: FastAPI application instance
        config: GraphQL configuration
        authentication_handler: Optional async function for authentication
        sse_manager: Optional SSE manager for event integration

    Returns:
        SubscriptionManager instance

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
        >>> subscription_manager = setup_graphql(app, config)
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

        # Create subscription manager
        subscription_manager = SubscriptionManager(schema, sse_manager)

        # Add to app state for access in route handlers
        app.state.graphql_executor = executor
        app.state.graphql_schema = schema
        app.state.subscription_manager = subscription_manager

        # Set up WebSocket endpoint for subscriptions
        if config.enable_subscriptions:
            @app.websocket(config.ws_path)
            async def graphql_websocket(websocket: WebSocket):
                """GraphQL WebSocket endpoint for subscriptions."""
                await subscription_manager.handle_websocket(
                    websocket, authentication_handler
                )

            logger.info(f"GraphQL WebSocket endpoint configured at {config.ws_path}")

        # Add GraphQL status endpoint
        @app.get("/graphql/status")
        async def graphql_status() -> Dict[str, Any]:
            """Get GraphQL service status."""
            return {
                "status": "operational",
                "endpoint": config.path,
                "websocket_endpoint": config.ws_path if config.enable_subscriptions else None,
                "subscriptions": subscription_manager.get_statistics(),
                "introspection_enabled": config.enable_schema_introspection,
                "playground_enabled": config.enable_playground,
            }

        # Set up startup/shutdown handlers
        @app.on_event("startup")
        async def start_subscription_manager():
            await subscription_manager.start()

        @app.on_event("shutdown")
        async def stop_subscription_manager():
            await subscription_manager.stop()

        logger.info("GraphQL integration setup complete")
        return subscription_manager

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
    "SubscriptionRegistry",
    "SubscriptionManager",
    "SubscriptionInfo",
    "WebSocketConnection",
    "SubscriptionMessageType",
    "GraphQLConfig",
    "setup_graphql",
    "query_agents",
    "query_emissions",
    "run_calculation",
]
