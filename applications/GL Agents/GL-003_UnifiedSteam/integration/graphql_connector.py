"""
GL-003 UNIFIEDSTEAM - GraphQL Connector

GraphQL client connector for real-time data integration with:
- Query and mutation support
- Subscription handling for real-time updates
- Query batching and caching
- Authentication (API key, OAuth2, Bearer token)
- Retry logic with exponential backoff
- Circuit breaker pattern for resilience

Uses:
- httpx for async HTTP requests
- gql for GraphQL operations (optional)
- websockets for subscriptions
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple, Union
from collections import OrderedDict
import asyncio
import json
import logging
import hashlib
import random
import time
import base64

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class AuthType(Enum):
    """Authentication types."""
    NONE = "none"
    API_KEY = "api_key"
    BEARER = "bearer"
    OAUTH2 = "oauth2"
    BASIC = "basic"


class RetryStrategy(Enum):
    """Retry strategies."""
    NONE = "none"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    EXPONENTIAL_JITTER = "exponential_jitter"


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class OAuth2Config:
    """OAuth2 configuration."""
    token_url: str
    client_id: str
    client_secret: str  # Retrieved from vault, never hardcoded
    scopes: List[str] = field(default_factory=list)
    grant_type: str = "client_credentials"


@dataclass
class RetryConfig:
    """Retry configuration."""
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_JITTER
    max_retries: int = 3
    base_delay_ms: int = 100
    max_delay_ms: int = 10000
    jitter_factor: float = 0.3

    def get_delay(self, attempt: int) -> float:
        """Calculate retry delay."""
        if self.strategy == RetryStrategy.NONE:
            return 0

        if self.strategy == RetryStrategy.LINEAR:
            delay_ms = self.base_delay_ms * (attempt + 1)
        elif self.strategy in (RetryStrategy.EXPONENTIAL, RetryStrategy.EXPONENTIAL_JITTER):
            delay_ms = self.base_delay_ms * (2 ** attempt)
        else:
            delay_ms = self.base_delay_ms

        delay_ms = min(delay_ms, self.max_delay_ms)

        if self.strategy == RetryStrategy.EXPONENTIAL_JITTER:
            jitter = delay_ms * self.jitter_factor * random.random()
            delay_ms = delay_ms + jitter

        return delay_ms / 1000.0


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5
    success_threshold: int = 3
    timeout_seconds: float = 30.0
    half_open_max_calls: int = 3


@dataclass
class CacheConfig:
    """Query cache configuration."""
    enabled: bool = True
    max_size: int = 1000
    ttl_seconds: int = 300  # 5 minutes


@dataclass
class GraphQLConfig:
    """GraphQL connector configuration."""
    endpoint: str
    ws_endpoint: Optional[str] = None  # WebSocket endpoint for subscriptions

    # Authentication
    auth_type: AuthType = AuthType.NONE
    api_key: Optional[str] = None
    api_key_header: str = "X-API-Key"
    bearer_token: Optional[str] = None
    basic_username: Optional[str] = None
    basic_password: Optional[str] = None
    oauth2_config: Optional[OAuth2Config] = None

    # Request settings
    timeout_seconds: float = 30.0
    max_connections: int = 10
    max_keepalive_connections: int = 5

    # Retry and circuit breaker
    retry_config: RetryConfig = field(default_factory=RetryConfig)
    circuit_breaker_config: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)

    # Caching
    cache_config: CacheConfig = field(default_factory=CacheConfig)

    # Batching
    batch_enabled: bool = True
    batch_max_size: int = 10
    batch_timeout_ms: int = 50


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class GraphQLResult:
    """Result from GraphQL operation."""
    data: Optional[Dict[str, Any]]
    errors: Optional[List[Dict[str, Any]]]
    extensions: Optional[Dict[str, Any]] = None
    success: bool = True
    latency_ms: float = 0.0
    cached: bool = False

    def has_errors(self) -> bool:
        return self.errors is not None and len(self.errors) > 0

    def get_data(self, path: Optional[str] = None) -> Any:
        """Get data by dot-notation path."""
        if self.data is None:
            return None
        if path is None:
            return self.data

        result = self.data
        for key in path.split("."):
            if isinstance(result, dict):
                result = result.get(key)
            else:
                return None
        return result


@dataclass
class CacheEntry:
    """Cache entry with TTL."""
    value: GraphQLResult
    expires_at: datetime
    hits: int = 0


@dataclass
class ConnectorMetrics:
    """Connector performance metrics."""
    queries_sent: int = 0
    queries_failed: int = 0
    mutations_sent: int = 0
    mutations_failed: int = 0
    subscriptions_active: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    retry_count: int = 0
    circuit_breaker_trips: int = 0


# =============================================================================
# Circuit Breaker
# =============================================================================

class CircuitBreaker:
    """Circuit breaker pattern for resilient operations."""

    def __init__(self, config: CircuitBreakerConfig) -> None:
        self.config = config
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._half_open_calls = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitBreakerState:
        return self._state

    async def can_execute(self) -> bool:
        """Check if request can be executed."""
        async with self._lock:
            if self._state == CircuitBreakerState.CLOSED:
                return True

            if self._state == CircuitBreakerState.OPEN:
                if self._last_failure_time:
                    elapsed = (datetime.now(timezone.utc) - self._last_failure_time).total_seconds()
                    if elapsed >= self.config.timeout_seconds:
                        self._state = CircuitBreakerState.HALF_OPEN
                        self._half_open_calls = 0
                        logger.info("Circuit breaker transitioning to HALF_OPEN")
                        return True
                return False

            if self._state == CircuitBreakerState.HALF_OPEN:
                if self._half_open_calls < self.config.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False

            return False

    async def record_success(self) -> None:
        """Record successful operation."""
        async with self._lock:
            if self._state == CircuitBreakerState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._state = CircuitBreakerState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    logger.info("Circuit breaker CLOSED after successful recovery")
            else:
                self._failure_count = 0

    async def record_failure(self) -> None:
        """Record failed operation."""
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = datetime.now(timezone.utc)

            if self._state == CircuitBreakerState.HALF_OPEN:
                self._state = CircuitBreakerState.OPEN
                self._success_count = 0
                logger.warning("Circuit breaker OPEN after failure in HALF_OPEN state")
            elif self._failure_count >= self.config.failure_threshold:
                self._state = CircuitBreakerState.OPEN
                logger.warning(f"Circuit breaker OPEN after {self._failure_count} failures")


# =============================================================================
# Query Cache (LRU)
# =============================================================================

class QueryCache:
    """LRU cache for GraphQL queries."""

    def __init__(self, config: CacheConfig) -> None:
        self.config = config
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()

    def _compute_key(self, query: str, variables: Optional[Dict]) -> str:
        """Compute cache key from query and variables."""
        key_data = f"{query}:{json.dumps(variables or {}, sort_keys=True)}"
        return hashlib.sha256(key_data.encode()).hexdigest()

    async def get(self, query: str, variables: Optional[Dict]) -> Optional[GraphQLResult]:
        """Get cached result if available and not expired."""
        if not self.config.enabled:
            return None

        key = self._compute_key(query, variables)

        async with self._lock:
            if key not in self._cache:
                return None

            entry = self._cache[key]

            # Check expiration
            if datetime.now(timezone.utc) > entry.expires_at:
                del self._cache[key]
                return None

            # Update LRU order
            self._cache.move_to_end(key)
            entry.hits += 1

            return entry.value

    async def set(self, query: str, variables: Optional[Dict], result: GraphQLResult) -> None:
        """Cache query result."""
        if not self.config.enabled:
            return

        key = self._compute_key(query, variables)

        async with self._lock:
            # Evict oldest if at capacity
            while len(self._cache) >= self.config.max_size:
                self._cache.popitem(last=False)

            entry = CacheEntry(
                value=result,
                expires_at=datetime.now(timezone.utc) + timedelta(seconds=self.config.ttl_seconds),
            )
            self._cache[key] = entry

    async def invalidate(self, pattern: Optional[str] = None) -> int:
        """Invalidate cache entries."""
        async with self._lock:
            if pattern is None:
                count = len(self._cache)
                self._cache.clear()
                return count

            # Pattern-based invalidation (simple substring match)
            keys_to_remove = [
                k for k in self._cache.keys()
                if pattern in k
            ]
            for k in keys_to_remove:
                del self._cache[k]
            return len(keys_to_remove)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_hits = sum(e.hits for e in self._cache.values())
        return {
            "size": len(self._cache),
            "max_size": self.config.max_size,
            "total_hits": total_hits,
            "enabled": self.config.enabled,
        }


# =============================================================================
# GraphQL Connector
# =============================================================================

class GraphQLConnector:
    """
    GraphQL client connector for real-time data integration.

    Features:
    - Query and mutation support
    - Subscription handling for real-time updates
    - Query batching and caching
    - Multiple authentication methods
    - Retry logic with exponential backoff
    - Circuit breaker pattern for resilience

    Example:
        config = GraphQLConfig(
            endpoint="https://api.example.com/graphql",
            ws_endpoint="wss://api.example.com/graphql",
            auth_type=AuthType.BEARER,
            bearer_token="your-token",
        )

        connector = GraphQLConnector(config)
        await connector.connect()

        # Query
        result = await connector.query('''
            query GetAsset($id: ID!) {
                asset(id: $id) {
                    id
                    name
                    status
                }
            }
        ''', variables={"id": "asset-123"})

        # Mutation
        result = await connector.mutation('''
            mutation UpdateSetpoint($input: SetpointInput!) {
                updateSetpoint(input: $input) {
                    success
                    message
                }
            }
        ''', variables={"input": {"assetId": "asset-123", "value": 150.0}})

        # Subscription
        async for event in connector.subscribe('''
            subscription OnSensorUpdate($assetId: ID!) {
                sensorUpdate(assetId: $assetId) {
                    tag
                    value
                    timestamp
                }
            }
        ''', variables={"assetId": "asset-123"}):
            print(f"Sensor update: {event}")
    """

    def __init__(
        self,
        config: GraphQLConfig,
        vault_client: Optional[Any] = None,
    ) -> None:
        """
        Initialize GraphQL connector.

        Args:
            config: Connector configuration
            vault_client: Optional vault client for credential retrieval
        """
        self.config = config
        self._vault_client = vault_client

        # Retrieve credentials from vault (NEVER hardcode)
        if vault_client:
            self._load_credentials_from_vault()

        self._client = None
        self._ws_client = None
        self._connected = False

        # OAuth2 token management
        self._oauth2_token: Optional[str] = None
        self._oauth2_expires_at: Optional[datetime] = None

        # Circuit breaker
        self._circuit_breaker = CircuitBreaker(config.circuit_breaker_config)

        # Cache
        self._cache = QueryCache(config.cache_config)

        # Batch buffer
        self._batch_buffer: List[Tuple[str, Optional[Dict], asyncio.Future]] = []
        self._batch_lock = asyncio.Lock()
        self._batch_task: Optional[asyncio.Task] = None

        # Active subscriptions
        self._subscriptions: Dict[str, asyncio.Task] = {}

        # Metrics
        self._metrics = ConnectorMetrics()

        logger.info(f"GraphQLConnector initialized: {config.endpoint}")

    def _load_credentials_from_vault(self) -> None:
        """Load credentials from vault."""
        try:
            if self.config.auth_type == AuthType.API_KEY:
                self.config.api_key = self._vault_client.get_secret("graphql/api_key")
            elif self.config.auth_type == AuthType.BEARER:
                self.config.bearer_token = self._vault_client.get_secret("graphql/bearer_token")
            elif self.config.auth_type == AuthType.BASIC:
                self.config.basic_password = self._vault_client.get_secret("graphql/basic_password")
            elif self.config.auth_type == AuthType.OAUTH2 and self.config.oauth2_config:
                self.config.oauth2_config.client_secret = self._vault_client.get_secret(
                    "graphql/oauth2_client_secret"
                )
        except Exception as e:
            logger.warning(f"Failed to load credentials from vault: {e}")

    async def connect(self) -> None:
        """Connect to GraphQL endpoint."""
        try:
            # In production, use httpx:
            # import httpx
            # self._client = httpx.AsyncClient(
            #     base_url=self.config.endpoint,
            #     timeout=self.config.timeout_seconds,
            #     limits=httpx.Limits(
            #         max_connections=self.config.max_connections,
            #         max_keepalive_connections=self.config.max_keepalive_connections,
            #     ),
            # )

            self._connected = True

            # Start batch processor if enabled
            if self.config.batch_enabled:
                self._batch_task = asyncio.create_task(self._batch_processor())

            logger.info(f"Connected to GraphQL endpoint: {self.config.endpoint}")

        except Exception as e:
            logger.error(f"Failed to connect to GraphQL endpoint: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from GraphQL endpoint."""
        # Cancel batch processor
        if self._batch_task:
            self._batch_task.cancel()
            try:
                await self._batch_task
            except asyncio.CancelledError:
                pass

        # Cancel all subscriptions
        for sub_id, task in self._subscriptions.items():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._subscriptions.clear()

        # Close HTTP client
        if self._client:
            # await self._client.aclose()
            self._client = None

        # Close WebSocket client
        if self._ws_client:
            # await self._ws_client.close()
            self._ws_client = None

        self._connected = False
        logger.info("Disconnected from GraphQL endpoint")

    @property
    def is_connected(self) -> bool:
        """Check connection status."""
        return self._connected

    async def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers."""
        headers = {"Content-Type": "application/json"}

        if self.config.auth_type == AuthType.API_KEY:
            headers[self.config.api_key_header] = self.config.api_key or ""

        elif self.config.auth_type == AuthType.BEARER:
            headers["Authorization"] = f"Bearer {self.config.bearer_token}"

        elif self.config.auth_type == AuthType.BASIC:
            credentials = f"{self.config.basic_username}:{self.config.basic_password}"
            encoded = base64.b64encode(credentials.encode()).decode()
            headers["Authorization"] = f"Basic {encoded}"

        elif self.config.auth_type == AuthType.OAUTH2:
            token = await self._get_oauth2_token()
            headers["Authorization"] = f"Bearer {token}"

        return headers

    async def _get_oauth2_token(self) -> str:
        """Get OAuth2 access token, refreshing if needed."""
        # Check if current token is still valid
        if self._oauth2_token and self._oauth2_expires_at:
            if datetime.now(timezone.utc) < self._oauth2_expires_at - timedelta(minutes=5):
                return self._oauth2_token

        # Request new token
        oauth_config = self.config.oauth2_config
        if not oauth_config:
            raise ValueError("OAuth2 config not provided")

        token_data = {
            "grant_type": oauth_config.grant_type,
            "client_id": oauth_config.client_id,
            "client_secret": oauth_config.client_secret,
        }
        if oauth_config.scopes:
            token_data["scope"] = " ".join(oauth_config.scopes)

        try:
            # In production:
            # response = await self._client.post(oauth_config.token_url, data=token_data)
            # response.raise_for_status()
            # token_response = response.json()
            # self._oauth2_token = token_response["access_token"]
            # expires_in = token_response.get("expires_in", 3600)
            # self._oauth2_expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in)

            # Simulate for framework
            self._oauth2_token = "simulated-oauth2-token"
            self._oauth2_expires_at = datetime.now(timezone.utc) + timedelta(hours=1)

            logger.info("OAuth2 token refreshed successfully")
            return self._oauth2_token

        except Exception as e:
            logger.error(f"Failed to get OAuth2 token: {e}")
            raise

    async def _execute_request(
        self,
        query: str,
        variables: Optional[Dict] = None,
        operation_name: Optional[str] = None,
    ) -> GraphQLResult:
        """Execute GraphQL request with retry and circuit breaker."""
        retry_config = self.config.retry_config

        for attempt in range(retry_config.max_retries + 1):
            # Check circuit breaker
            if not await self._circuit_breaker.can_execute():
                logger.warning("Circuit breaker OPEN, request rejected")
                self._metrics.circuit_breaker_trips += 1
                return GraphQLResult(
                    data=None,
                    errors=[{"message": "Circuit breaker open"}],
                    success=False,
                )

            try:
                start_time = time.perf_counter()

                headers = await self._get_auth_headers()

                payload = {
                    "query": query,
                    "variables": variables or {},
                }
                if operation_name:
                    payload["operationName"] = operation_name

                # In production:
                # response = await self._client.post(
                #     self.config.endpoint,
                #     headers=headers,
                #     json=payload,
                # )
                # response.raise_for_status()
                # response_data = response.json()

                # Simulate for framework
                await asyncio.sleep(0.01)
                response_data = {
                    "data": {"simulated": True},
                    "errors": None,
                }

                latency = (time.perf_counter() - start_time) * 1000

                await self._circuit_breaker.record_success()
                self._update_latency_metrics(latency)

                return GraphQLResult(
                    data=response_data.get("data"),
                    errors=response_data.get("errors"),
                    extensions=response_data.get("extensions"),
                    success=response_data.get("errors") is None,
                    latency_ms=latency,
                )

            except Exception as e:
                logger.warning(f"Request attempt {attempt + 1} failed: {e}")
                await self._circuit_breaker.record_failure()
                self._metrics.retry_count += 1

                if attempt < retry_config.max_retries:
                    delay = retry_config.get_delay(attempt)
                    logger.info(f"Retrying in {delay:.2f}s...")
                    await asyncio.sleep(delay)
                else:
                    logger.error("All retries exhausted")
                    return GraphQLResult(
                        data=None,
                        errors=[{"message": str(e)}],
                        success=False,
                    )

        return GraphQLResult(data=None, errors=[{"message": "Unknown error"}], success=False)

    def _update_latency_metrics(self, latency_ms: float) -> None:
        """Update latency metrics."""
        n = self._metrics.queries_sent + self._metrics.mutations_sent
        if n > 0:
            self._metrics.avg_latency_ms = (
                (self._metrics.avg_latency_ms * (n - 1) + latency_ms) / n
            )
        else:
            self._metrics.avg_latency_ms = latency_ms
        self._metrics.max_latency_ms = max(self._metrics.max_latency_ms, latency_ms)

    async def query(
        self,
        query: str,
        variables: Optional[Dict] = None,
        operation_name: Optional[str] = None,
        use_cache: bool = True,
    ) -> GraphQLResult:
        """
        Execute GraphQL query.

        Args:
            query: GraphQL query string
            variables: Query variables
            operation_name: Optional operation name
            use_cache: Whether to use query cache

        Returns:
            GraphQLResult with data and/or errors
        """
        # Check cache
        if use_cache:
            cached_result = await self._cache.get(query, variables)
            if cached_result:
                self._metrics.cache_hits += 1
                cached_result.cached = True
                return cached_result
            self._metrics.cache_misses += 1

        # Execute query
        result = await self._execute_request(query, variables, operation_name)

        if result.success:
            self._metrics.queries_sent += 1
            # Cache successful results
            if use_cache:
                await self._cache.set(query, variables, result)
        else:
            self._metrics.queries_failed += 1

        return result

    async def mutation(
        self,
        mutation: str,
        variables: Optional[Dict] = None,
        operation_name: Optional[str] = None,
    ) -> GraphQLResult:
        """
        Execute GraphQL mutation.

        Args:
            mutation: GraphQL mutation string
            variables: Mutation variables
            operation_name: Optional operation name

        Returns:
            GraphQLResult with data and/or errors
        """
        result = await self._execute_request(mutation, variables, operation_name)

        if result.success:
            self._metrics.mutations_sent += 1
        else:
            self._metrics.mutations_failed += 1

        return result

    async def batch_query(
        self,
        query: str,
        variables: Optional[Dict] = None,
        operation_name: Optional[str] = None,
    ) -> GraphQLResult:
        """
        Add query to batch buffer.

        Queries are automatically batched and sent together for efficiency.

        Args:
            query: GraphQL query string
            variables: Query variables
            operation_name: Optional operation name

        Returns:
            GraphQLResult when batch is executed
        """
        if not self.config.batch_enabled:
            return await self.query(query, variables, operation_name)

        future: asyncio.Future = asyncio.get_event_loop().create_future()

        async with self._batch_lock:
            self._batch_buffer.append((query, variables, future))

            # Execute immediately if batch is full
            if len(self._batch_buffer) >= self.config.batch_max_size:
                await self._execute_batch()

        return await future

    async def _batch_processor(self) -> None:
        """Background task to process batched queries."""
        while True:
            try:
                await asyncio.sleep(self.config.batch_timeout_ms / 1000.0)
                await self._execute_batch()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in batch processor: {e}")

    async def _execute_batch(self) -> None:
        """Execute all queued batch queries."""
        async with self._batch_lock:
            if not self._batch_buffer:
                return

            batch = self._batch_buffer.copy()
            self._batch_buffer.clear()

        # Execute each query (in production, use actual batching)
        for query, variables, future in batch:
            try:
                result = await self._execute_request(query, variables)
                if not future.done():
                    future.set_result(result)
            except Exception as e:
                if not future.done():
                    future.set_exception(e)

    async def subscribe(
        self,
        subscription: str,
        variables: Optional[Dict] = None,
        operation_name: Optional[str] = None,
    ) -> AsyncIterator[GraphQLResult]:
        """
        Subscribe to GraphQL subscription.

        Args:
            subscription: GraphQL subscription string
            variables: Subscription variables
            operation_name: Optional operation name

        Yields:
            GraphQLResult for each subscription event
        """
        subscription_id = f"sub_{hash(subscription)}_{time.time()}"

        try:
            self._metrics.subscriptions_active += 1
            logger.info(f"Starting subscription: {subscription_id}")

            # In production, use websockets with graphql-ws protocol:
            # async with websockets.connect(self.config.ws_endpoint) as ws:
            #     await ws.send(json.dumps({
            #         "type": "connection_init",
            #         "payload": await self._get_auth_headers(),
            #     }))
            #
            #     await ws.send(json.dumps({
            #         "id": subscription_id,
            #         "type": "subscribe",
            #         "payload": {
            #             "query": subscription,
            #             "variables": variables,
            #             "operationName": operation_name,
            #         },
            #     }))
            #
            #     async for message in ws:
            #         data = json.loads(message)
            #         if data["type"] == "next":
            #             yield GraphQLResult(
            #                 data=data["payload"].get("data"),
            #                 errors=data["payload"].get("errors"),
            #                 success=data["payload"].get("errors") is None,
            #             )

            # Simulate subscription events for framework
            for i in range(5):
                await asyncio.sleep(1.0)
                yield GraphQLResult(
                    data={"subscriptionEvent": {"index": i, "timestamp": datetime.now(timezone.utc).isoformat()}},
                    errors=None,
                    success=True,
                )

        finally:
            self._metrics.subscriptions_active -= 1
            logger.info(f"Subscription ended: {subscription_id}")

    async def invalidate_cache(self, pattern: Optional[str] = None) -> int:
        """
        Invalidate cache entries.

        Args:
            pattern: Optional pattern to match cache keys

        Returns:
            Number of invalidated entries
        """
        return await self._cache.invalidate(pattern)

    def get_metrics(self) -> Dict[str, Any]:
        """Get connector metrics."""
        return {
            "queries_sent": self._metrics.queries_sent,
            "queries_failed": self._metrics.queries_failed,
            "mutations_sent": self._metrics.mutations_sent,
            "mutations_failed": self._metrics.mutations_failed,
            "subscriptions_active": self._metrics.subscriptions_active,
            "cache_hits": self._metrics.cache_hits,
            "cache_misses": self._metrics.cache_misses,
            "cache_hit_rate": (
                self._metrics.cache_hits / (self._metrics.cache_hits + self._metrics.cache_misses)
                if (self._metrics.cache_hits + self._metrics.cache_misses) > 0 else 0
            ),
            "avg_latency_ms": round(self._metrics.avg_latency_ms, 3),
            "max_latency_ms": round(self._metrics.max_latency_ms, 3),
            "retry_count": self._metrics.retry_count,
            "circuit_breaker_state": self._circuit_breaker.state.value,
            "circuit_breaker_trips": self._metrics.circuit_breaker_trips,
            "connected": self._connected,
            **self._cache.get_stats(),
        }


# =============================================================================
# Factory Functions
# =============================================================================

def create_graphql_connector(
    endpoint: str,
    auth_type: AuthType = AuthType.NONE,
    api_key: Optional[str] = None,
    bearer_token: Optional[str] = None,
    ws_endpoint: Optional[str] = None,
) -> GraphQLConnector:
    """
    Create a GraphQL connector with common configuration.

    Args:
        endpoint: GraphQL HTTP endpoint
        auth_type: Authentication type
        api_key: API key (if auth_type is API_KEY)
        bearer_token: Bearer token (if auth_type is BEARER)
        ws_endpoint: WebSocket endpoint for subscriptions

    Returns:
        Configured GraphQLConnector instance
    """
    config = GraphQLConfig(
        endpoint=endpoint,
        ws_endpoint=ws_endpoint,
        auth_type=auth_type,
        api_key=api_key,
        bearer_token=bearer_token,
    )

    return GraphQLConnector(config)


def create_oauth2_graphql_connector(
    endpoint: str,
    token_url: str,
    client_id: str,
    client_secret: str,
    scopes: Optional[List[str]] = None,
    ws_endpoint: Optional[str] = None,
) -> GraphQLConnector:
    """
    Create a GraphQL connector with OAuth2 authentication.

    Args:
        endpoint: GraphQL HTTP endpoint
        token_url: OAuth2 token endpoint
        client_id: OAuth2 client ID
        client_secret: OAuth2 client secret
        scopes: OAuth2 scopes
        ws_endpoint: WebSocket endpoint for subscriptions

    Returns:
        Configured GraphQLConnector instance
    """
    oauth2_config = OAuth2Config(
        token_url=token_url,
        client_id=client_id,
        client_secret=client_secret,
        scopes=scopes or [],
    )

    config = GraphQLConfig(
        endpoint=endpoint,
        ws_endpoint=ws_endpoint,
        auth_type=AuthType.OAUTH2,
        oauth2_config=oauth2_config,
    )

    return GraphQLConnector(config)
