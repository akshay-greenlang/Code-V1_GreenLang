"""
Base ERP Connector Framework for GreenLang

Provides abstract base class with enterprise features:
- Authentication management (OAuth2, API keys, SAML)
- Connection pooling and HTTP client management
- Rate limiting with configurable thresholds
- Retry logic with exponential backoff
- Circuit breaker pattern for fault tolerance
- Data transformation pipeline
- Comprehensive error handling and logging
- Metrics tracking for monitoring
"""

import asyncio
import hashlib
import json
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging

import httpx
from pydantic import BaseModel, Field, HttpUrl, validator
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)
import jwt


# Configure logging
logger = logging.getLogger(__name__)


class AuthType(Enum):
    """Supported authentication types."""
    OAUTH2 = "oauth2"
    API_KEY = "api_key"
    BASIC = "basic"
    SAML = "saml"
    JWT = "jwt"
    CUSTOM = "custom"


class ConnectorState(Enum):
    """Connector operational states."""
    READY = "ready"
    CONNECTING = "connecting"
    AUTHENTICATED = "authenticated"
    RATE_LIMITED = "rate_limited"
    CIRCUIT_OPEN = "circuit_open"
    ERROR = "error"
    CLOSED = "closed"


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    requests_per_second: float = 10
    requests_per_minute: int = 100
    requests_per_hour: int = 1000
    burst_size: int = 20
    retry_after_seconds: int = 60


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5
    recovery_timeout: int = 60  # seconds
    expected_exception_types: Tuple[type, ...] = (httpx.HTTPStatusError,)
    success_threshold: int = 2  # successes needed to close circuit


@dataclass
class ConnectionPoolConfig:
    """HTTP connection pool configuration."""
    max_connections: int = 100
    max_keepalive_connections: int = 20
    keepalive_expiry: int = 300  # seconds
    connect_timeout: float = 10.0
    read_timeout: float = 30.0
    write_timeout: float = 10.0
    pool_timeout: float = 5.0


class BaseConnectorConfig(BaseModel):
    """Base configuration for all ERP connectors."""

    # Connection settings
    base_url: HttpUrl
    api_version: str = "v1"

    # Authentication
    auth_type: AuthType
    auth_config: Dict[str, Any] = Field(default_factory=dict)

    # Timeouts and retries
    timeout_seconds: float = 30.0
    max_retries: int = 3
    retry_backoff_factor: float = 2.0

    # Rate limiting
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)

    # Circuit breaker
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)

    # Connection pool
    connection_pool: ConnectionPoolConfig = Field(default_factory=ConnectionPoolConfig)

    # Features
    enable_caching: bool = True
    cache_ttl_seconds: int = 300
    enable_metrics: bool = True
    enable_audit_log: bool = True

    class Config:
        arbitrary_types_allowed = True


class ConnectorMetrics:
    """Track connector performance metrics."""

    def __init__(self):
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_retries = 0
        self.circuit_breaker_trips = 0
        self.rate_limit_hits = 0
        self.total_response_time = 0.0
        self.last_request_time = None
        self.errors_by_type = {}

    def record_request(self, success: bool, response_time: float, error: Optional[Exception] = None):
        """Record request metrics."""
        self.total_requests += 1
        self.total_response_time += response_time
        self.last_request_time = datetime.utcnow()

        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
            if error:
                error_type = type(error).__name__
                self.errors_by_type[error_type] = self.errors_by_type.get(error_type, 0) + 1

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        avg_response_time = (
            self.total_response_time / self.total_requests
            if self.total_requests > 0 else 0
        )

        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": (
                self.successful_requests / self.total_requests
                if self.total_requests > 0 else 0
            ),
            "average_response_time": avg_response_time,
            "total_retries": self.total_retries,
            "circuit_breaker_trips": self.circuit_breaker_trips,
            "rate_limit_hits": self.rate_limit_hits,
            "errors_by_type": self.errors_by_type,
            "last_request_time": self.last_request_time
        }


class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.tokens = config.burst_size
        self.max_tokens = config.burst_size
        self.refill_rate = config.requests_per_second
        self.last_refill = time.monotonic()
        self.minute_window = []
        self.hour_window = []
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> bool:
        """Acquire tokens for request."""
        async with self._lock:
            now = time.monotonic()

            # Refill tokens based on time elapsed
            elapsed = now - self.last_refill
            tokens_to_add = elapsed * self.refill_rate
            self.tokens = min(self.max_tokens, self.tokens + tokens_to_add)
            self.last_refill = now

            # Check minute and hour windows
            current_time = datetime.utcnow()
            self._clean_windows(current_time)

            if len(self.minute_window) >= self.config.requests_per_minute:
                return False

            if len(self.hour_window) >= self.config.requests_per_hour:
                return False

            # Check token bucket
            if self.tokens >= tokens:
                self.tokens -= tokens
                self.minute_window.append(current_time)
                self.hour_window.append(current_time)
                return True

            return False

    def _clean_windows(self, current_time: datetime):
        """Clean old entries from time windows."""
        minute_ago = current_time - timedelta(minutes=1)
        hour_ago = current_time - timedelta(hours=1)

        self.minute_window = [t for t in self.minute_window if t > minute_ago]
        self.hour_window = [t for t in self.hour_window if t > hour_ago]

    async def wait_if_needed(self) -> float:
        """Calculate wait time if rate limited."""
        if self.tokens < 1:
            wait_time = (1 - self.tokens) / self.refill_rate
            return wait_time
        return 0


class CircuitBreaker:
    """Circuit breaker for fault tolerance."""

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.state = ConnectorState.READY
        self._lock = asyncio.Lock()

    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        async with self._lock:
            if self.state == ConnectorState.CIRCUIT_OPEN:
                if self._should_attempt_reset():
                    self.state = ConnectorState.READY
                else:
                    raise Exception("Circuit breaker is open")

        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except self.config.expected_exception_types as e:
            await self._on_failure()
            raise e

    async def _on_success(self):
        """Handle successful call."""
        async with self._lock:
            self.failure_count = 0
            self.success_count += 1

            if (self.state == ConnectorState.CIRCUIT_OPEN and
                self.success_count >= self.config.success_threshold):
                self.state = ConnectorState.READY
                logger.info("Circuit breaker closed after successful calls")

    async def _on_failure(self):
        """Handle failed call."""
        async with self._lock:
            self.failure_count += 1
            self.success_count = 0
            self.last_failure_time = datetime.utcnow()

            if self.failure_count >= self.config.failure_threshold:
                self.state = ConnectorState.CIRCUIT_OPEN
                logger.warning(f"Circuit breaker opened after {self.failure_count} failures")

    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset."""
        if self.last_failure_time:
            elapsed = (datetime.utcnow() - self.last_failure_time).total_seconds()
            return elapsed >= self.config.recovery_timeout
        return True


class DataTransformer:
    """Transform data between ERP and GreenLang formats."""

    @staticmethod
    def normalize_date(date_value: Any) -> Optional[str]:
        """Normalize date to ISO format."""
        if isinstance(date_value, str):
            # Handle various date formats
            for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y%m%d"]:
                try:
                    dt = datetime.strptime(date_value, fmt)
                    return dt.isoformat()
                except ValueError:
                    continue
        elif isinstance(date_value, datetime):
            return date_value.isoformat()
        return None

    @staticmethod
    def normalize_currency(amount: Any, currency: str = "USD") -> Dict[str, Any]:
        """Normalize currency values."""
        try:
            if isinstance(amount, str):
                amount = float(amount.replace(",", "").replace("$", ""))
            return {
                "amount": float(amount),
                "currency": currency,
                "formatted": f"{currency} {amount:,.2f}"
            }
        except (ValueError, TypeError):
            return {"amount": 0.0, "currency": currency, "formatted": f"{currency} 0.00"}

    @staticmethod
    def clean_text(text: Any) -> str:
        """Clean and normalize text fields."""
        if text is None:
            return ""
        text = str(text).strip()
        # Remove excessive whitespace
        text = " ".join(text.split())
        return text


class BaseERPConnector(ABC):
    """
    Abstract base class for ERP connectors.

    Provides:
    - Authentication management
    - HTTP client with connection pooling
    - Rate limiting
    - Circuit breaker
    - Retry logic
    - Data transformation
    - Metrics tracking
    - Audit logging
    """

    def __init__(self, config: BaseConnectorConfig):
        """Initialize connector with configuration."""
        self.config = config
        self.state = ConnectorState.READY
        self.metrics = ConnectorMetrics()

        # Initialize components
        self.rate_limiter = RateLimiter(config.rate_limit)
        self.circuit_breaker = CircuitBreaker(config.circuit_breaker)
        self.transformer = DataTransformer()

        # HTTP client with connection pooling
        self.client = None
        self._client_lock = asyncio.Lock()

        # Authentication
        self.access_token = None
        self.token_expires_at = None
        self.refresh_token = None

        # Caching
        self.cache = {} if config.enable_caching else None

        # Audit log
        self.audit_log = [] if config.enable_audit_log else None

        logger.info(f"Initialized {self.__class__.__name__} connector")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()

    async def connect(self):
        """Establish connection to ERP system."""
        async with self._client_lock:
            if self.client is None:
                limits = httpx.Limits(
                    max_connections=self.config.connection_pool.max_connections,
                    max_keepalive_connections=self.config.connection_pool.max_keepalive_connections
                )

                timeout = httpx.Timeout(
                    connect=self.config.connection_pool.connect_timeout,
                    read=self.config.connection_pool.read_timeout,
                    write=self.config.connection_pool.write_timeout,
                    pool=self.config.connection_pool.pool_timeout
                )

                self.client = httpx.AsyncClient(
                    base_url=str(self.config.base_url),
                    limits=limits,
                    timeout=timeout,
                    follow_redirects=True
                )

                self.state = ConnectorState.CONNECTING
                logger.info(f"Established connection to {self.config.base_url}")

    async def disconnect(self):
        """Close connection to ERP system."""
        async with self._client_lock:
            if self.client:
                await self.client.aclose()
                self.client = None
                self.state = ConnectorState.CLOSED
                logger.info("Disconnected from ERP system")

    @abstractmethod
    async def authenticate(self) -> bool:
        """
        Authenticate with ERP system.

        Must be implemented by subclasses for specific auth flows.
        """
        pass

    @abstractmethod
    async def validate_connection(self) -> bool:
        """
        Validate connection to ERP system.

        Must be implemented by subclasses.
        """
        pass

    async def _ensure_authenticated(self):
        """Ensure valid authentication before API calls."""
        if self.token_expires_at:
            if datetime.utcnow() >= self.token_expires_at - timedelta(minutes=5):
                logger.info("Token expiring, refreshing authentication")
                await self.authenticate()
        elif self.access_token is None:
            await self.authenticate()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(httpx.HTTPStatusError),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None
    ) -> httpx.Response:
        """
        Make HTTP request with retry logic and rate limiting.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            headers: Optional headers
            params: Optional query parameters
            json_data: Optional JSON body
            data: Optional form data

        Returns:
            HTTP response

        Raises:
            httpx.HTTPStatusError: On HTTP errors
            Exception: On other errors
        """
        # Ensure authenticated
        await self._ensure_authenticated()

        # Check rate limit
        if not await self.rate_limiter.acquire():
            wait_time = await self.rate_limiter.wait_if_needed()
            self.metrics.rate_limit_hits += 1
            logger.warning(f"Rate limited, waiting {wait_time:.2f} seconds")
            await asyncio.sleep(wait_time)
            await self.rate_limiter.acquire()

        # Prepare request
        if headers is None:
            headers = {}

        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"

        # Make request with circuit breaker
        start_time = time.monotonic()

        try:
            response = await self.circuit_breaker.call(
                self.client.request,
                method=method,
                url=endpoint,
                headers=headers,
                params=params,
                json=json_data,
                data=data
            )

            response.raise_for_status()

            # Record metrics
            elapsed = time.monotonic() - start_time
            self.metrics.record_request(True, elapsed)

            # Audit log
            if self.audit_log is not None:
                self.audit_log.append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "method": method,
                    "endpoint": endpoint,
                    "status": response.status_code,
                    "response_time": elapsed,
                    "success": True
                })

            return response

        except Exception as e:
            elapsed = time.monotonic() - start_time
            self.metrics.record_request(False, elapsed, e)

            if self.audit_log is not None:
                self.audit_log.append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "method": method,
                    "endpoint": endpoint,
                    "error": str(e),
                    "response_time": elapsed,
                    "success": False
                })

            logger.error(f"Request failed: {method} {endpoint} - {str(e)}")
            raise

    async def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """Make GET request."""
        response = await self._make_request("GET", endpoint, params=params, **kwargs)
        return response.json()

    async def post(self, endpoint: str, json_data: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """Make POST request."""
        response = await self._make_request("POST", endpoint, json_data=json_data, **kwargs)
        return response.json()

    async def put(self, endpoint: str, json_data: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """Make PUT request."""
        response = await self._make_request("PUT", endpoint, json_data=json_data, **kwargs)
        return response.json()

    async def delete(self, endpoint: str, **kwargs) -> bool:
        """Make DELETE request."""
        response = await self._make_request("DELETE", endpoint, **kwargs)
        return response.status_code in [200, 204]

    async def paginate(
        self,
        endpoint: str,
        page_size: int = 100,
        max_pages: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Paginate through API results.

        Args:
            endpoint: API endpoint
            page_size: Items per page
            max_pages: Maximum pages to fetch
            params: Additional query parameters

        Returns:
            List of all records
        """
        if params is None:
            params = {}

        all_records = []
        page = 1

        while True:
            # Add pagination parameters
            page_params = {**params, "page": page, "page_size": page_size}

            # Fetch page
            response = await self.get(endpoint, params=page_params)

            # Extract records (implementation specific to each ERP)
            records = self._extract_records_from_response(response)

            if not records:
                break

            all_records.extend(records)

            # Check if we've reached max pages
            if max_pages and page >= max_pages:
                break

            # Check if there are more pages
            if not self._has_more_pages(response):
                break

            page += 1

        logger.info(f"Paginated {len(all_records)} records from {endpoint}")
        return all_records

    def _extract_records_from_response(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract records from paginated response."""
        # Default implementation - override in subclasses
        if isinstance(response, list):
            return response
        elif "data" in response:
            return response["data"]
        elif "results" in response:
            return response["results"]
        elif "records" in response:
            return response["records"]
        else:
            return []

    def _has_more_pages(self, response: Dict[str, Any]) -> bool:
        """Check if there are more pages."""
        # Default implementation - override in subclasses
        if "has_more" in response:
            return response["has_more"]
        elif "next" in response:
            return response["next"] is not None
        elif "next_page" in response:
            return response["next_page"] is not None
        else:
            # If we can't determine, assume no more pages
            return False

    async def batch_process(
        self,
        items: List[Any],
        processor_func,
        batch_size: int = 50,
        max_workers: int = 5
    ) -> List[Any]:
        """
        Process items in batches with concurrency control.

        Args:
            items: Items to process
            processor_func: Async function to process each item
            batch_size: Items per batch
            max_workers: Maximum concurrent workers

        Returns:
            Processed results
        """
        results = []
        semaphore = asyncio.Semaphore(max_workers)

        async def process_item(item):
            async with semaphore:
                return await processor_func(item)

        # Process in batches
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_results = await asyncio.gather(
                *[process_item(item) for item in batch],
                return_exceptions=True
            )

            # Filter out exceptions
            for result in batch_results:
                if not isinstance(result, Exception):
                    results.append(result)
                else:
                    logger.error(f"Batch processing error: {result}")

        return results

    def get_metrics(self) -> Dict[str, Any]:
        """Get connector metrics."""
        return self.metrics.get_summary()

    def get_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get audit log entries."""
        if self.audit_log is not None:
            return self.audit_log[-limit:]
        return []

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on connector."""
        health = {
            "status": "healthy",
            "state": self.state.value,
            "metrics": self.get_metrics(),
            "issues": []
        }

        # Check connection
        try:
            if await self.validate_connection():
                health["connection"] = "ok"
            else:
                health["connection"] = "failed"
                health["status"] = "unhealthy"
                health["issues"].append("Connection validation failed")
        except Exception as e:
            health["connection"] = "error"
            health["status"] = "unhealthy"
            health["issues"].append(f"Connection error: {str(e)}")

        # Check circuit breaker
        if self.circuit_breaker.state == ConnectorState.CIRCUIT_OPEN:
            health["status"] = "degraded"
            health["issues"].append("Circuit breaker is open")

        # Check rate limiting
        if self.metrics.rate_limit_hits > 10:
            health["status"] = "degraded"
            health["issues"].append(f"High rate limit hits: {self.metrics.rate_limit_hits}")

        return health