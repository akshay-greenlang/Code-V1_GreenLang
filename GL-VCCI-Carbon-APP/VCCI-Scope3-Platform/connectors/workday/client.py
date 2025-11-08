"""
Workday RaaS (Report as a Service) Client
GL-VCCI Scope 3 Platform

RaaS client for Workday with pagination, error handling, and retry logic.
Reuses rate limiting utilities from SAP connector for consistency.

SECURITY: Uses defusedxml to prevent XXE (XML External Entity) attacks.

Version: 1.0.0
Phase: 4 (Weeks 24-26)
Date: 2025-11-06
Security Update: 2025-11-08
"""

import logging
import time
try:
    # Use defusedxml for secure XML parsing (prevents XXE attacks)
    import defusedxml.ElementTree as ET
    DEFUSEDXML_AVAILABLE = True
except ImportError:
    # Fallback to standard library with warning
    import xml.etree.ElementTree as ET
    DEFUSEDXML_AVAILABLE = False

from typing import Dict, List, Any, Optional, Generator
from urllib.parse import urlencode
from datetime import date, datetime
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry as URLRetry

from .config import WorkdayConnectorConfig, RaaSReport
from .auth import WorkdayAuthHandler, get_auth_handler
from .exceptions import (
    WorkdayConnectionError,
    WorkdayAuthenticationError,
    WorkdayRateLimitError,
    WorkdayDataError,
    WorkdayTimeoutError,
    get_exception_for_status_code
)

logger = logging.getLogger(__name__)

# Warn if defusedxml is not available
if not DEFUSEDXML_AVAILABLE:
    logger.warning(
        "defusedxml not available - using standard xml.etree.ElementTree. "
        "Install defusedxml for XXE attack protection: pip install defusedxml>=0.7.1"
    )


class RateLimiter:
    """
    Simple token bucket rate limiter.

    Reused from SAP connector for consistency.
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
            WorkdayRateLimitError: If rate limit would be exceeded
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


class WorkdayRaaSClient:
    """
    RaaS client for Workday.

    Provides methods for retrieving reports with automatic pagination,
    authentication, retry logic, and error handling.
    """

    def __init__(self, config: WorkdayConnectorConfig):
        """
        Initialize RaaS client.

        Args:
            config: Workday connector configuration
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
            f"Initialized Workday RaaS client for environment: {config.environment}"
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
        retry_count: int = 0
    ) -> requests.Response:
        """
        Make HTTP request with retry logic.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: Request URL
            params: Query parameters
            retry_count: Current retry attempt

        Returns:
            HTTP response

        Raises:
            WorkdayConnectionError: On connection failure
            WorkdayAuthenticationError: On auth failure
            WorkdayRateLimitError: On rate limit
            WorkdayTimeoutError: On timeout
        """
        # Apply rate limiting
        if self.rate_limiter and not self.rate_limiter.acquire():
            raise WorkdayRateLimitError(
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
                    response, url, method, params, retry_count
                )

            return response

        except requests.exceptions.Timeout as e:
            logger.error(f"Request timeout: {url}")
            raise WorkdayTimeoutError(
                endpoint=url,
                timeout_seconds=int(self.config.timeout.total_timeout)
            )

        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error: {url} - {e}")

            # Retry on connection errors
            if retry_count < self.config.retry.max_retries:
                return self._retry_request(
                    method, url, params, retry_count, str(e)
                )

            raise WorkdayConnectionError(
                endpoint=url,
                reason="Connection failed after retries",
                original_exception=e
            )

        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {url} - {e}")
            raise WorkdayConnectionError(
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
        retry_count: int
    ) -> requests.Response:
        """
        Handle HTTP error responses.

        Args:
            response: HTTP response
            url: Request URL
            method: HTTP method
            params: Query parameters
            retry_count: Current retry attempt

        Returns:
            Response if recoverable

        Raises:
            Appropriate Workday exception
        """
        status_code = response.status_code

        # Check if we should retry
        if (
            status_code in self.config.retry.retry_on_status_codes
            and retry_count < self.config.retry.max_retries
        ):
            return self._retry_request(
                method, url, params, retry_count,
                f"HTTP {status_code}"
            )

        # Handle 401 - try refreshing token once
        if status_code == 401 and retry_count == 0:
            logger.warning("Received 401, refreshing token and retrying")
            self.auth_handler.invalidate_token()
            return self._make_request(method, url, params, retry_count + 1)

        # Parse error message
        error_msg = self._parse_error_response(response)

        # Raise appropriate exception
        raise get_exception_for_status_code(status_code, url, error_msg)

    def _retry_request(
        self,
        method: str,
        url: str,
        params: Optional[Dict[str, Any]],
        retry_count: int,
        reason: str
    ) -> requests.Response:
        """
        Retry request with exponential backoff.

        Args:
            method: HTTP method
            url: Request URL
            params: Query parameters
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
        return self._make_request(method, url, params, retry_count)

    def _parse_error_response(self, response: requests.Response) -> str:
        """
        Parse Workday error response.

        Args:
            response: HTTP response

        Returns:
            Human-readable error message
        """
        try:
            # Try JSON first
            error_data = response.json()
            if "error" in error_data:
                error = error_data["error"]
                if isinstance(error, dict):
                    message = error.get("message", str(error))
                    return str(message)
                return str(error)
            return response.text[:200]

        except Exception:
            # Try XML if JSON fails
            try:
                root = ET.fromstring(response.text)
                error_msg = root.find(".//message")
                if error_msg is not None:
                    return error_msg.text
            except Exception:
                pass

            return f"HTTP {response.status_code}: {response.text[:200]}"

    def get_report(
        self,
        report_name: str,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        additional_params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get report data with automatic pagination.

        Args:
            report_name: Name of the report (from config)
            from_date: Start date for report filtering
            to_date: End date for report filtering
            additional_params: Additional query parameters

        Returns:
            List of all report records

        Raises:
            ValueError: If report not found
        """
        results = []

        for batch in self.get_report_paginated(
            report_name, from_date, to_date, additional_params
        ):
            results.extend(batch)

        return results

    def get_report_paginated(
        self,
        report_name: str,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        additional_params: Optional[Dict[str, Any]] = None
    ) -> Generator[List[Dict[str, Any]], None, None]:
        """
        Get report data with pagination.

        Args:
            report_name: Name of the report (from config)
            from_date: Start date for report filtering
            to_date: End date for report filtering
            additional_params: Additional query parameters

        Yields:
            Batches of report records

        Raises:
            ValueError: If report not found
        """
        # Get report config
        report = self.config.get_report_config(report_name)
        if not report:
            raise ValueError(f"Unknown report: {report_name}")

        # Build URL
        url = self.config.get_raas_url(report_name)
        if not url:
            raise ValueError(f"Could not build URL for report: {report_name}")

        # Build query parameters
        params = {
            "format": report.format,
            "limit": report.batch_size,
            "offset": 0
        }

        # Add date filters
        if from_date:
            params["From_Date"] = from_date.isoformat()
        if to_date:
            params["To_Date"] = to_date.isoformat()

        # Add additional parameters
        if additional_params:
            params.update(additional_params)

        # Paginate through results
        while True:
            # Make request
            response = self._make_request("GET", url, params=params)

            # Parse response
            try:
                if report.format == "json":
                    data = response.json()
                    # Workday RaaS returns data in Report_Entry array
                    results = data.get("Report_Entry", [])
                else:
                    # Parse XML response
                    results = self._parse_xml_response(response.text)

            except Exception as e:
                logger.error(f"Failed to parse response: {e}")
                raise WorkdayDataError(
                    data_type=report.name,
                    reason="Failed to parse response",
                    original_exception=e
                )

            # Yield results
            if results:
                yield results

            # Check if there are more results
            if len(results) < report.batch_size:
                # Last page
                break

            # Move to next page
            params["offset"] += report.batch_size

    def _parse_xml_response(self, xml_text: str) -> List[Dict[str, Any]]:
        """
        Parse XML response to list of dictionaries.

        Args:
            xml_text: XML response text

        Returns:
            List of dictionaries
        """
        results = []
        try:
            root = ET.fromstring(xml_text)
            for entry in root.findall(".//Report_Entry"):
                record = {}
                for child in entry:
                    # Strip namespace from tag
                    tag = child.tag.split('}')[-1]
                    record[tag] = child.text
                results.append(record)
        except Exception as e:
            logger.error(f"Failed to parse XML: {e}")

        return results

    def close(self):
        """Close HTTP session and clean up resources."""
        if self.session:
            self.session.close()
            logger.info("Workday RaaS client session closed")
