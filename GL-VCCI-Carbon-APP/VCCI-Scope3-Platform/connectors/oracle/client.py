"""
Oracle Fusion Cloud REST Client
GL-VCCI Scope 3 Platform

REST client for Oracle Fusion Cloud API with pagination, error handling,
and retry logic.

Version: 1.0.0
Phase: 4 (Weeks 22-24)
Date: 2025-11-06
"""

import logging
import time
from typing import Dict, List, Any, Optional, Generator
from urllib.parse import urljoin, urlencode
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry as URLRetry

from .config import OracleConnectorConfig, RESTEndpoint
from .auth import OracleAuthHandler, get_auth_handler
from .exceptions import (
    OracleConnectionError,
    OracleAuthenticationError,
    OracleRateLimitError,
    OracleDataError,
    OracleTimeoutError,
    get_exception_for_status_code
)

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Simple token bucket rate limiter.

    In production, this would be replaced with Redis-based distributed
    rate limiting for multi-instance deployments.
    """

    def __init__(self, requests_per_minute: int, burst_size: int = 5):
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests per minute
            burst_size: Maximum burst size
        """
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.tokens = burst_size
        self.last_update = time.time()
        self.interval = 60.0 / requests_per_minute  # Seconds per request

    def acquire(self, timeout: float = 60.0) -> bool:
        """
        Acquire permission to make a request.

        Args:
            timeout: Maximum time to wait for permission (seconds)

        Returns:
            True if permission granted, False if timeout

        Raises:
            OracleRateLimitError: If rate limit would be exceeded
        """
        start = time.time()

        while True:
            # Refill tokens
            now = time.time()
            elapsed = now - self.last_update
            self.tokens = min(
                self.burst_size,
                self.tokens + elapsed / self.interval
            )
            self.last_update = now

            # Check if we have tokens
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return True

            # Check timeout
            if time.time() - start > timeout:
                return False

            # Wait for next token
            sleep_time = min(self.interval, timeout - (time.time() - start))
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                return False


class RESTQueryBuilder:
    """Helper class for building Oracle REST API query parameters."""

    def __init__(self):
        """Initialize query builder."""
        self.params: Dict[str, Any] = {}

    def q(self, query_expr: str) -> "RESTQueryBuilder":
        """
        Add q parameter for filtering.

        Args:
            query_expr: Query expression (e.g., "LastUpdateDate >= '2024-01-01T00:00:00'")

        Returns:
            Self for chaining
        """
        self.params["q"] = query_expr
        return self

    def fields(self, *field_names: str) -> "RESTQueryBuilder":
        """
        Add fields parameter to limit response fields.

        Args:
            field_names: Fields to include in response

        Returns:
            Self for chaining
        """
        self.params["fields"] = ",".join(field_names)
        return self

    def limit(self, count: int) -> "RESTQueryBuilder":
        """
        Add limit parameter for pagination.

        Args:
            count: Maximum number of records

        Returns:
            Self for chaining
        """
        self.params["limit"] = count
        return self

    def offset(self, count: int) -> "RESTQueryBuilder":
        """
        Add offset parameter for pagination.

        Args:
            count: Number of records to skip

        Returns:
            Self for chaining
        """
        self.params["offset"] = count
        return self

    def orderby(self, *fields: str, descending: bool = False) -> "RESTQueryBuilder":
        """
        Add orderBy parameter.

        Args:
            fields: Fields to order by
            descending: Whether to sort descending

        Returns:
            Self for chaining
        """
        order_expr = ",".join(fields)
        if descending:
            order_expr += ":desc"
        self.params["orderBy"] = order_expr
        return self

    def finder(self, finder_name: str, **params) -> "RESTQueryBuilder":
        """
        Add finder parameter with custom parameters.

        Args:
            finder_name: Name of the finder
            params: Finder parameters

        Returns:
            Self for chaining
        """
        # Finder format: ?finder=FinderName;param1=value1;param2=value2
        finder_params = ";".join([f"{k}={v}" for k, v in params.items()])
        if finder_params:
            self.params["finder"] = f"{finder_name};{finder_params}"
        else:
            self.params["finder"] = finder_name
        return self

    def expand(self, *child_resources: str) -> "RESTQueryBuilder":
        """
        Add expand parameter to include child resources.

        Args:
            child_resources: Child resources to expand

        Returns:
            Self for chaining
        """
        self.params["expand"] = ",".join(child_resources)
        return self

    def build(self) -> Dict[str, Any]:
        """
        Build query parameters.

        Returns:
            Dictionary of query parameters
        """
        return self.params.copy()


class OracleRESTClient:
    """
    REST client for Oracle Fusion Cloud.

    Provides methods for querying REST services with automatic pagination,
    authentication, retry logic, and error handling.
    """

    def __init__(self, config: OracleConnectorConfig):
        """
        Initialize REST client.

        Args:
            config: Oracle connector configuration
        """
        self.config = config
        self.auth_handler = get_auth_handler(
            oauth_config=config.oauth,
            environment=config.environment
        )

        # Set up rate limiter
        self.rate_limiter = RateLimiter(
            requests_per_minute=config.rate_limit.requests_per_minute,
            burst_size=config.rate_limit.burst_size
        ) if config.rate_limit.enabled else None

        # Set up HTTP session with connection pooling
        self.session = self._create_session()

        logger.info(
            f"Initialized Oracle REST client for environment: {config.environment}"
        )

    def _create_session(self) -> requests.Session:
        """
        Create HTTP session with connection pooling and retry logic.

        Returns:
            Configured requests.Session
        """
        session = requests.Session()

        # Configure connection pooling
        adapter = HTTPAdapter(
            pool_connections=self.config.pool_connections,
            pool_maxsize=self.config.pool_maxsize,
            max_retries=URLRetry(
                total=0,  # We handle retries manually
                connect=0,
                read=0,
                redirect=3,
                status_forcelist=[]
            )
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        # Set default headers
        session.headers.update({
            "Accept": "application/json",
            "Content-Type": "application/json"
        })

        return session

    def _get_headers(self, additional_headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """
        Get request headers including auth.

        Args:
            additional_headers: Additional headers to include

        Returns:
            Complete headers dictionary
        """
        headers = self.auth_handler.get_auth_header()

        if additional_headers:
            headers.update(additional_headers)

        return headers

    def _make_request(
        self,
        method: str,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        retry_count: int = 0
    ) -> requests.Response:
        """
        Make HTTP request with retry logic.

        Args:
            method: HTTP method (GET, POST, PATCH)
            url: Request URL
            params: Query parameters
            data: Request body data
            retry_count: Current retry attempt

        Returns:
            HTTP response

        Raises:
            OracleConnectionError: On connection failure
            OracleAuthenticationError: On auth failure
            OracleRateLimitError: On rate limit
            OracleTimeoutError: On timeout
        """
        # Apply rate limiting
        if self.rate_limiter and not self.rate_limiter.acquire():
            raise OracleRateLimitError(
                endpoint=url,
                limit=self.config.rate_limit.requests_per_minute
            )

        try:
            # Get headers with auth
            headers = self._get_headers()

            # Make request
            start_time = time.time()
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                json=data,
                headers=headers,
                timeout=(
                    self.config.timeout.connect_timeout,
                    self.config.timeout.read_timeout
                )
            )
            elapsed_ms = (time.time() - start_time) * 1000

            logger.debug(
                f"{method} {url} - Status: {response.status_code}, "
                f"Time: {elapsed_ms:.2f}ms"
            )

            # Handle HTTP errors
            if response.status_code >= 400:
                return self._handle_error_response(
                    response, url, method, params, data, retry_count
                )

            return response

        except requests.exceptions.Timeout as e:
            logger.error(f"Request timeout: {url}")
            raise OracleTimeoutError(
                endpoint=url,
                timeout_seconds=int(self.config.timeout.total_timeout)
            )

        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error: {url} - {e}")

            # Retry on connection errors
            if retry_count < self.config.retry.max_retries:
                return self._retry_request(
                    method, url, params, data, retry_count, str(e)
                )

            raise OracleConnectionError(
                endpoint=url,
                reason="Connection failed after retries",
                original_exception=e
            )

        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {url} - {e}")
            raise OracleConnectionError(
                endpoint=url,
                reason=str(e),
                original_exception=e
            )

    def _handle_error_response(
        self,
        response: requests.Response,
        url: str,
        method: str,
        params: Optional[Dict[str, Any]],
        data: Optional[Dict[str, Any]],
        retry_count: int
    ) -> requests.Response:
        """
        Handle HTTP error responses.

        Args:
            response: HTTP response
            url: Request URL
            method: HTTP method
            params: Query parameters
            data: Request body
            retry_count: Current retry attempt

        Returns:
            Response if recoverable

        Raises:
            Appropriate Oracle exception
        """
        status_code = response.status_code

        # Check if we should retry
        if (
            status_code in self.config.retry.retry_on_status_codes
            and retry_count < self.config.retry.max_retries
        ):
            return self._retry_request(
                method, url, params, data, retry_count,
                f"HTTP {status_code}"
            )

        # Handle 401 - try refreshing token once
        if status_code == 401 and retry_count == 0:
            logger.warning("Received 401, refreshing token and retrying")
            self.auth_handler.invalidate_token()
            return self._make_request(method, url, params, data, retry_count + 1)

        # Parse error message
        error_msg = self._parse_rest_error(response)

        # Raise appropriate exception
        raise get_exception_for_status_code(status_code, url, error_msg)

    def _retry_request(
        self,
        method: str,
        url: str,
        params: Optional[Dict[str, Any]],
        data: Optional[Dict[str, Any]],
        retry_count: int,
        reason: str
    ) -> requests.Response:
        """
        Retry request with exponential backoff.

        Args:
            method: HTTP method
            url: Request URL
            params: Query parameters
            data: Request body
            retry_count: Current retry attempt
            reason: Reason for retry

        Returns:
            HTTP response from retry
        """
        retry_count += 1
        delay = min(
            self.config.retry.base_delay * (self.config.retry.backoff_multiplier ** retry_count),
            self.config.retry.max_delay
        )

        logger.info(
            f"Retrying request (attempt {retry_count}/{self.config.retry.max_retries}) "
            f"after {delay:.2f}s - Reason: {reason}"
        )

        time.sleep(delay)
        return self._make_request(method, url, params, data, retry_count)

    def _parse_rest_error(self, response: requests.Response) -> str:
        """
        Parse Oracle REST error response.

        Args:
            response: HTTP response

        Returns:
            Human-readable error message
        """
        try:
            error_data = response.json()

            # Oracle REST API error format
            if "title" in error_data:
                title = error_data.get("title", "")
                detail = error_data.get("detail", "")
                return f"{title}: {detail}" if detail else title

            # Alternative error format
            if "error" in error_data:
                error = error_data["error"]
                if isinstance(error, dict):
                    message = error.get("message", str(error))
                    return message
                return str(error)

            return response.text[:200]

        except Exception:
            return f"HTTP {response.status_code}: {response.text[:200]}"

    def get(
        self,
        endpoint_name: str,
        query_params: Optional[Dict[str, Any]] = None,
        resource_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute GET request to REST endpoint.

        Args:
            endpoint_name: Name of the endpoint (from config)
            query_params: REST query parameters
            resource_id: Optional resource ID for single resource retrieval

        Returns:
            Parsed JSON response

        Raises:
            ValueError: If endpoint not found
            OracleDataError: If response parsing fails
        """
        # Get endpoint config
        endpoint = self.config.get_endpoint_config(endpoint_name)
        if not endpoint:
            raise ValueError(f"Unknown endpoint: {endpoint_name}")

        # Build URL
        base_url = self.config.get_full_endpoint_url(endpoint_name)
        if resource_id:
            url = f"{base_url}/{resource_id}"
        else:
            url = base_url

        # Make request
        response = self._make_request("GET", url, params=query_params)

        # Parse response
        try:
            return response.json()
        except Exception as e:
            logger.error(f"Failed to parse response: {e}")
            raise OracleDataError(
                data_type=endpoint.name,
                reason="Failed to parse JSON response",
                original_exception=e
            )

    def post(
        self,
        endpoint_name: str,
        data: Dict[str, Any],
        resource_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute POST request to REST endpoint.

        Args:
            endpoint_name: Name of the endpoint
            data: Request body data
            resource_id: Optional resource ID for nested resources

        Returns:
            Parsed JSON response

        Raises:
            ValueError: If endpoint not found
            OracleDataError: If response parsing fails
        """
        # Get endpoint config
        endpoint = self.config.get_endpoint_config(endpoint_name)
        if not endpoint:
            raise ValueError(f"Unknown endpoint: {endpoint_name}")

        # Build URL
        base_url = self.config.get_full_endpoint_url(endpoint_name)
        if resource_id:
            url = f"{base_url}/{resource_id}"
        else:
            url = base_url

        # Make request
        response = self._make_request("POST", url, data=data)

        # Parse response
        try:
            return response.json()
        except Exception as e:
            logger.error(f"Failed to parse response: {e}")
            raise OracleDataError(
                data_type=endpoint.name,
                reason="Failed to parse JSON response",
                original_exception=e
            )

    def patch(
        self,
        endpoint_name: str,
        resource_id: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute PATCH request to REST endpoint.

        Args:
            endpoint_name: Name of the endpoint
            resource_id: Resource ID to update
            data: Request body data (partial update)

        Returns:
            Parsed JSON response

        Raises:
            ValueError: If endpoint not found
            OracleDataError: If response parsing fails
        """
        # Get endpoint config
        endpoint = self.config.get_endpoint_config(endpoint_name)
        if not endpoint:
            raise ValueError(f"Unknown endpoint: {endpoint_name}")

        # Build URL
        base_url = self.config.get_full_endpoint_url(endpoint_name)
        url = f"{base_url}/{resource_id}"

        # Make request
        response = self._make_request("PATCH", url, data=data)

        # Parse response
        try:
            return response.json()
        except Exception as e:
            logger.error(f"Failed to parse response: {e}")
            raise OracleDataError(
                data_type=endpoint.name,
                reason="Failed to parse JSON response",
                original_exception=e
            )

    def query(
        self,
        endpoint_name: str,
        builder: Optional[RESTQueryBuilder] = None
    ) -> List[Dict[str, Any]]:
        """
        Query endpoint and return all results (handles pagination).

        Args:
            endpoint_name: Name of the endpoint
            builder: REST query builder with filters/parameters

        Returns:
            List of all matching items

        Raises:
            ValueError: If endpoint not found
        """
        params = builder.build() if builder else {}
        results = []

        for batch in self.query_paginated(endpoint_name, params):
            results.extend(batch)

        return results

    def query_paginated(
        self,
        endpoint_name: str,
        query_params: Optional[Dict[str, Any]] = None
    ) -> Generator[List[Dict[str, Any]], None, None]:
        """
        Query endpoint with automatic pagination.

        Args:
            endpoint_name: Name of the endpoint
            query_params: REST query parameters

        Yields:
            Batches of results

        Raises:
            ValueError: If endpoint not found
        """
        # Get endpoint config
        endpoint = self.config.get_endpoint_config(endpoint_name)
        if not endpoint:
            raise ValueError(f"Unknown endpoint: {endpoint_name}")

        # Set default batch size if not specified
        params = query_params.copy() if query_params else {}
        if "limit" not in params:
            params["limit"] = endpoint.batch_size

        # Build initial URL
        url = self.config.get_full_endpoint_url(endpoint_name)

        while url:
            # Make request
            response = self._make_request("GET", url, params=params)

            # Parse response
            try:
                data = response.json()
            except Exception as e:
                logger.error(f"Failed to parse response: {e}")
                raise OracleDataError(
                    data_type=endpoint.name,
                    reason="Failed to parse JSON response",
                    original_exception=e
                )

            # Extract results (Oracle format: {"items": [...], "count": N, "hasMore": bool, "links": [...]})
            items = data.get("items", [])
            if items:
                yield items

            # Check for next page via links
            has_more = data.get("hasMore", False)
            if has_more:
                links = data.get("links", [])
                next_link = None

                # Find "next" link
                for link in links:
                    if link.get("rel") == "next":
                        next_link = link.get("href")
                        break

                if next_link:
                    # Next link is usually absolute
                    url = next_link
                    params = None  # Params are in the next link URL
                else:
                    # No next link found, stop pagination
                    break
            else:
                break

    def close(self):
        """Close HTTP session and clean up resources."""
        if self.session:
            self.session.close()
            logger.info("Oracle REST client session closed")


def create_query() -> RESTQueryBuilder:
    """
    Create new REST query builder.

    Returns:
        RESTQueryBuilder instance
    """
    return RESTQueryBuilder()
