"""
Base Connector Framework
========================

Abstract base class and infrastructure for building ERP and API connectors.

Author: GL-DataIntegrationEngineer
Version: 1.0.0
Created: 2025-12-04
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, TypeVar, Generic
from enum import Enum
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import asyncio
import logging
import time
from contextlib import asynccontextmanager

from pydantic import BaseModel, Field, SecretStr
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ConnectorStatus(str, Enum):
    """Connector status states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATING = "authenticating"
    AUTHENTICATED = "authenticated"
    ERROR = "error"
    RATE_LIMITED = "rate_limited"


class AuthenticationType(str, Enum):
    """Authentication types supported."""
    NONE = "none"
    API_KEY = "api_key"
    BASIC = "basic"
    OAUTH2_CLIENT_CREDENTIALS = "oauth2_client_credentials"
    OAUTH2_AUTH_CODE = "oauth2_auth_code"
    SAML = "saml"
    JWT = "jwt"


class ConnectorConfig(BaseModel):
    """Base configuration for connectors."""
    name: str = Field(..., description="Connector name")
    base_url: str = Field(..., description="Base URL for API")
    auth_type: AuthenticationType = Field(default=AuthenticationType.API_KEY)
    api_key: Optional[SecretStr] = Field(None, description="API key (from vault)")
    client_id: Optional[str] = Field(None, description="OAuth2 client ID")
    client_secret: Optional[SecretStr] = Field(None, description="OAuth2 client secret")
    token_url: Optional[str] = Field(None, description="OAuth2 token endpoint")
    username: Optional[str] = Field(None, description="Basic auth username")
    password: Optional[SecretStr] = Field(None, description="Basic auth password")
    timeout_seconds: int = Field(default=30, ge=1, le=300)
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_delay_seconds: int = Field(default=2, ge=1, le=60)
    rate_limit_requests_per_minute: int = Field(default=60, ge=1, le=10000)
    connection_pool_size: int = Field(default=10, ge=1, le=100)
    verify_ssl: bool = Field(default=True)
    headers: Dict[str, str] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        use_enum_values = True


@dataclass
class ConnectionMetrics:
    """Metrics for connector operations."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_bytes_received: int = 0
    total_bytes_sent: int = 0
    average_response_time_ms: float = 0.0
    last_request_time: Optional[datetime] = None
    last_error: Optional[str] = None
    rate_limit_hits: int = 0


class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, requests_per_minute: int):
        self.rate = requests_per_minute
        self.tokens = requests_per_minute
        self.max_tokens = requests_per_minute
        self.last_refill = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self) -> bool:
        """Acquire a token for making a request."""
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_refill

            # Refill tokens based on elapsed time
            new_tokens = elapsed * (self.rate / 60.0)
            self.tokens = min(self.max_tokens, self.tokens + new_tokens)
            self.last_refill = now

            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False

    async def wait_for_token(self, timeout: float = 60.0) -> bool:
        """Wait for a token to become available."""
        start = time.time()
        while time.time() - start < timeout:
            if await self.acquire():
                return True
            await asyncio.sleep(0.1)
        return False


class CircuitBreaker:
    """Circuit breaker for fault tolerance."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 30,
        half_open_requests: int = 3
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_requests = half_open_requests
        self.failures = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "closed"  # closed, open, half_open
        self._lock = asyncio.Lock()

    async def can_execute(self) -> bool:
        """Check if request can be executed."""
        async with self._lock:
            if self.state == "closed":
                return True

            if self.state == "open":
                # Check if recovery timeout has passed
                if self.last_failure_time:
                    elapsed = (datetime.utcnow() - self.last_failure_time).total_seconds()
                    if elapsed >= self.recovery_timeout:
                        self.state = "half_open"
                        return True
                return False

            if self.state == "half_open":
                return True

            return False

    async def record_success(self):
        """Record a successful request."""
        async with self._lock:
            self.failures = 0
            self.state = "closed"

    async def record_failure(self):
        """Record a failed request."""
        async with self._lock:
            self.failures += 1
            self.last_failure_time = datetime.utcnow()

            if self.failures >= self.failure_threshold:
                self.state = "open"
                logger.warning("Circuit breaker opened due to failures")


class ConnectionPool:
    """Connection pool for managing HTTP clients."""

    def __init__(self, pool_size: int = 10):
        self.pool_size = pool_size
        self._connections: List[Any] = []
        self._available: asyncio.Queue = asyncio.Queue(maxsize=pool_size)
        self._lock = asyncio.Lock()
        self._initialized = False

    async def initialize(self, client_factory: callable):
        """Initialize the connection pool."""
        async with self._lock:
            if self._initialized:
                return

            for _ in range(self.pool_size):
                client = await client_factory()
                self._connections.append(client)
                await self._available.put(client)

            self._initialized = True

    @asynccontextmanager
    async def get_connection(self):
        """Get a connection from the pool."""
        connection = await self._available.get()
        try:
            yield connection
        finally:
            await self._available.put(connection)

    async def close_all(self):
        """Close all connections."""
        async with self._lock:
            for conn in self._connections:
                try:
                    await conn.aclose()
                except:
                    pass
            self._connections.clear()
            self._initialized = False


class BaseConnector(ABC, Generic[T]):
    """
    Abstract base class for ERP and API connectors.

    Provides:
    - OAuth2 authentication with token refresh
    - Rate limiting
    - Retry logic with exponential backoff
    - Circuit breaker for fault tolerance
    - Connection pooling
    - Metrics collection
    """

    def __init__(self, config: ConnectorConfig):
        """Initialize connector."""
        self.config = config
        self.status = ConnectorStatus.DISCONNECTED
        self.metrics = ConnectionMetrics()

        # Token management
        self.access_token: Optional[str] = None
        self.token_expires_at: Optional[datetime] = None
        self.refresh_token: Optional[str] = None

        # Rate limiting and circuit breaker
        self.rate_limiter = RateLimiter(config.rate_limit_requests_per_minute)
        self.circuit_breaker = CircuitBreaker()

        # Connection pool
        self.pool = ConnectionPool(config.connection_pool_size)

        logger.info(f"Connector {config.name} initialized")

    async def connect(self) -> bool:
        """
        Establish connection and authenticate.

        Returns:
            True if connection successful
        """
        self.status = ConnectorStatus.CONNECTING

        try:
            # Initialize connection pool
            await self.pool.initialize(self._create_client)

            # Authenticate
            self.status = ConnectorStatus.AUTHENTICATING
            await self.authenticate()

            self.status = ConnectorStatus.AUTHENTICATED
            logger.info(f"Connector {self.config.name} connected successfully")
            return True

        except Exception as e:
            self.status = ConnectorStatus.ERROR
            self.metrics.last_error = str(e)
            logger.error(f"Connection failed: {e}")
            raise

    @abstractmethod
    async def _create_client(self) -> Any:
        """Create an HTTP client instance."""
        pass

    async def authenticate(self) -> str:
        """
        Perform authentication based on auth type.

        Returns:
            Access token
        """
        auth_type = AuthenticationType(self.config.auth_type) if isinstance(
            self.config.auth_type, str
        ) else self.config.auth_type

        if auth_type == AuthenticationType.NONE:
            return ""

        elif auth_type == AuthenticationType.API_KEY:
            if self.config.api_key:
                self.access_token = self.config.api_key.get_secret_value()
            return self.access_token or ""

        elif auth_type == AuthenticationType.OAUTH2_CLIENT_CREDENTIALS:
            return await self._oauth2_client_credentials()

        elif auth_type == AuthenticationType.BASIC:
            # Basic auth is handled per-request
            return ""

        else:
            raise ValueError(f"Unsupported auth type: {auth_type}")

    async def _oauth2_client_credentials(self) -> str:
        """Perform OAuth2 client credentials flow."""
        # Check if token is still valid
        if self.access_token and self.token_expires_at:
            if datetime.utcnow() < self.token_expires_at - timedelta(minutes=5):
                return self.access_token

        if not self.config.token_url:
            raise ValueError("token_url required for OAuth2")

        if not self.config.client_id or not self.config.client_secret:
            raise ValueError("client_id and client_secret required for OAuth2")

        # Request new token
        async with self.pool.get_connection() as client:
            token_data = {
                'grant_type': 'client_credentials',
                'client_id': self.config.client_id,
                'client_secret': self.config.client_secret.get_secret_value(),
            }

            response = await client.post(self.config.token_url, data=token_data)
            response.raise_for_status()

            token_response = response.json()
            self.access_token = token_response['access_token']
            expires_in = token_response.get('expires_in', 3600)
            self.token_expires_at = datetime.utcnow() + timedelta(seconds=expires_in)

            if 'refresh_token' in token_response:
                self.refresh_token = token_response['refresh_token']

            logger.info("OAuth2 authentication successful")
            return self.access_token

    async def _ensure_authenticated(self):
        """Ensure we have a valid authentication token."""
        auth_type = AuthenticationType(self.config.auth_type) if isinstance(
            self.config.auth_type, str
        ) else self.config.auth_type

        if auth_type in [AuthenticationType.OAUTH2_CLIENT_CREDENTIALS, AuthenticationType.OAUTH2_AUTH_CODE]:
            # Check token expiration
            if self.token_expires_at and datetime.utcnow() >= self.token_expires_at - timedelta(minutes=5):
                await self.authenticate()

    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for requests."""
        headers = dict(self.config.headers)
        auth_type = AuthenticationType(self.config.auth_type) if isinstance(
            self.config.auth_type, str
        ) else self.config.auth_type

        if auth_type == AuthenticationType.API_KEY and self.access_token:
            headers['Authorization'] = f'Bearer {self.access_token}'
            # Some APIs use X-API-Key header
            headers['X-API-Key'] = self.access_token

        elif auth_type in [AuthenticationType.OAUTH2_CLIENT_CREDENTIALS, AuthenticationType.OAUTH2_AUTH_CODE]:
            if self.access_token:
                headers['Authorization'] = f'Bearer {self.access_token}'

        elif auth_type == AuthenticationType.BASIC:
            import base64
            if self.config.username and self.config.password:
                credentials = f"{self.config.username}:{self.config.password.get_secret_value()}"
                encoded = base64.b64encode(credentials.encode()).decode()
                headers['Authorization'] = f'Basic {encoded}'

        return headers

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Make an authenticated HTTP request with retry logic.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (appended to base_url)
            params: Query parameters
            data: Form data
            json_data: JSON body data

        Returns:
            Response data as dictionary
        """
        # Check circuit breaker
        if not await self.circuit_breaker.can_execute():
            raise ConnectionError("Circuit breaker is open")

        # Wait for rate limit token
        if not await self.rate_limiter.wait_for_token(timeout=30):
            self.metrics.rate_limit_hits += 1
            self.status = ConnectorStatus.RATE_LIMITED
            raise ConnectionError("Rate limit exceeded")

        await self._ensure_authenticated()

        url = f"{self.config.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        headers = self._get_auth_headers()
        headers['Accept'] = 'application/json'

        start_time = time.time()
        self.metrics.total_requests += 1

        try:
            async with self.pool.get_connection() as client:
                response = await client.request(
                    method=method,
                    url=url,
                    params=params,
                    data=data,
                    json=json_data,
                    headers=headers,
                    timeout=self.config.timeout_seconds,
                )

                response.raise_for_status()

                # Update metrics
                elapsed_ms = (time.time() - start_time) * 1000
                self._update_metrics(success=True, response_time_ms=elapsed_ms)

                await self.circuit_breaker.record_success()

                return response.json()

        except Exception as e:
            self._update_metrics(success=False)
            await self.circuit_breaker.record_failure()
            self.metrics.last_error = str(e)
            logger.error(f"Request failed: {e}")
            raise

    def _update_metrics(self, success: bool, response_time_ms: float = 0):
        """Update connection metrics."""
        if success:
            self.metrics.successful_requests += 1
        else:
            self.metrics.failed_requests += 1

        self.metrics.last_request_time = datetime.utcnow()

        # Update rolling average response time
        if response_time_ms > 0:
            total_successful = self.metrics.successful_requests
            if total_successful == 1:
                self.metrics.average_response_time_ms = response_time_ms
            else:
                current_avg = self.metrics.average_response_time_ms
                self.metrics.average_response_time_ms = (
                    current_avg * (total_successful - 1) + response_time_ms
                ) / total_successful

    @abstractmethod
    async def fetch_data(self, **kwargs) -> List[T]:
        """
        Fetch data from the source system.

        Returns:
            List of data records
        """
        pass

    async def close(self):
        """Close connector and release resources."""
        await self.pool.close_all()
        self.status = ConnectorStatus.DISCONNECTED
        logger.info(f"Connector {self.config.name} closed")

    def get_metrics(self) -> Dict[str, Any]:
        """Get connector metrics."""
        return {
            "name": self.config.name,
            "status": self.status.value,
            "total_requests": self.metrics.total_requests,
            "successful_requests": self.metrics.successful_requests,
            "failed_requests": self.metrics.failed_requests,
            "success_rate": (
                self.metrics.successful_requests / self.metrics.total_requests * 100
                if self.metrics.total_requests > 0 else 0
            ),
            "average_response_time_ms": round(self.metrics.average_response_time_ms, 2),
            "rate_limit_hits": self.metrics.rate_limit_hits,
            "last_error": self.metrics.last_error,
            "circuit_breaker_state": self.circuit_breaker.state,
        }
