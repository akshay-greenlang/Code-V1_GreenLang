"""
Connector Error Taxonomy
========================

Structured error hierarchy for connector operations.

Follows the pattern from greenlang/intelligence/providers/errors.py
with connector-specific error types.

Design principles:
- Base ConnectorError with structured context
- Specific error types for classification
- Machine-readable error codes
- Remediation hints for common issues
- Integration with policy/security errors
"""

from typing import Optional, Dict, Any, List


class ConnectorError(Exception):
    """
    Base exception for all connector errors

    Provides structured error information similar to ProviderError pattern:
    - message: Human-readable error description
    - connector: Which connector raised the error
    - status_code: HTTP status code (if applicable)
    - request_id: Request identifier for debugging
    - url: Target URL (if applicable)
    - context: Additional error context
    - original_error: Wrapped exception (if any)
    """

    def __init__(
        self,
        message: str,
        connector: str,
        status_code: Optional[int] = None,
        request_id: Optional[str] = None,
        url: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        self.message = message
        self.connector = connector
        self.status_code = status_code
        self.request_id = request_id
        self.url = url
        self.context = context or {}
        self.original_error = original_error

        # Format detailed message
        parts = [f"[{connector}] {message}"]

        if status_code:
            parts.append(f"(HTTP {status_code})")
        if request_id:
            parts.append(f"(request: {request_id})")
        if url:
            parts.append(f"(URL: {url})")

        super().__init__(" ".join(parts))

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for serialization"""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "connector": self.connector,
            "status_code": self.status_code,
            "request_id": self.request_id,
            "url": self.url,
            "context": self.context,
            "original_error": str(self.original_error) if self.original_error else None
        }


class ConnectorConfigError(ConnectorError):
    """
    Configuration error

    Raised when connector configuration is invalid or incomplete.

    Example:
        raise ConnectorConfigError(
            "Missing API key",
            connector="grid/electricitymaps",
            context={"required_env": "ELECTRICITYMAPS_API_KEY"}
        )
    """
    pass


class ConnectorAuthError(ConnectorError):
    """
    Authentication/authorization error

    Raised when authentication fails or credentials are invalid.

    Common causes:
    - Invalid API key
    - Expired token
    - Insufficient permissions
    - Rate limit exceeded (can also use ConnectorRateLimit)
    """
    pass


class ConnectorNetworkError(ConnectorError):
    """
    Network communication error

    Raised when network request fails.

    Common causes:
    - Connection timeout
    - DNS resolution failure
    - TLS/SSL errors
    - Network unreachable
    """
    pass


class ConnectorTimeoutError(ConnectorError):
    """
    Request timeout error

    Raised when request exceeds timeout threshold.
    """
    pass


class ConnectorRateLimit(ConnectorError):
    """
    Rate limit exceeded

    Raised when API rate limit is hit.

    Includes retry information when available.
    """

    def __init__(
        self,
        message: str,
        connector: str,
        retry_after: Optional[int] = None,  # Seconds until retry allowed
        limit: Optional[int] = None,  # Rate limit value
        **kwargs
    ):
        super().__init__(message, connector, **kwargs)
        self.retry_after = retry_after
        self.limit = limit

        if retry_after:
            self.context["retry_after"] = retry_after
        if limit:
            self.context["limit"] = limit


class ConnectorNotFound(ConnectorError):
    """
    Resource not found error

    Raised when requested resource doesn't exist.

    Common causes:
    - Invalid region code
    - Non-existent data for date range
    - Missing dataset
    """
    pass


class ConnectorBadRequest(ConnectorError):
    """
    Bad request error (client error)

    Raised when request is malformed or invalid.

    Common causes:
    - Invalid query parameters
    - Malformed date format
    - Unsupported region code
    - Schema validation failure
    """
    pass


class ConnectorServerError(ConnectorError):
    """
    Server error (5xx)

    Raised when upstream service has internal error.

    Usually retryable.
    """
    pass


class ConnectorReplayRequired(ConnectorError):
    """
    Replay mode requires snapshot

    Raised when connector is in replay mode but no snapshot available.

    This is a security/determinism enforcement error.

    Remediation:
    1. Switch to record mode to create snapshot
    2. Provide snapshot file path
    3. Use golden snapshot for testing
    """

    def __init__(
        self,
        message: str,
        connector: str,
        query_hash: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, connector, **kwargs)
        self.query_hash = query_hash

        # Add remediation hint
        self.context["hint"] = (
            "Either (1) switch to record mode to create snapshot, "
            "or (2) provide snapshot file path, "
            "or (3) use golden snapshot for testing"
        )


class ConnectorSnapshotNotFound(ConnectorError):
    """
    Snapshot file not found

    Raised when snapshot path specified but file doesn't exist.
    """

    def __init__(
        self,
        message: str,
        connector: str,
        snapshot_path: str,
        **kwargs
    ):
        super().__init__(message, connector, **kwargs)
        self.snapshot_path = snapshot_path
        self.context["snapshot_path"] = snapshot_path


class ConnectorSnapshotCorrupt(ConnectorError):
    """
    Snapshot data corrupted

    Raised when snapshot exists but is invalid or corrupted.

    Common causes:
    - Hash mismatch
    - Invalid JSON
    - Missing required fields
    - Version incompatibility
    """
    pass


class ConnectorSecurityError(ConnectorError):
    """
    Security policy violation

    Raised when operation violates security policy.

    Common causes:
    - Egress blocked (default deny)
    - Domain not in allowlist
    - TLS required but not used
    - Metadata endpoint access blocked
    """

    def __init__(
        self,
        message: str,
        connector: str,
        policy_violated: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, connector, **kwargs)
        self.policy_violated = policy_violated

        if policy_violated:
            self.context["policy_violated"] = policy_violated


class ConnectorValidationError(ConnectorError):
    """
    Data validation error

    Raised when payload data fails validation.

    Common causes:
    - Pydantic validation failure
    - Schema mismatch
    - Invalid data types
    - Missing required fields
    """

    def __init__(
        self,
        message: str,
        connector: str,
        validation_errors: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> None:
        super().__init__(message, connector, **kwargs)
        self.validation_errors = validation_errors or []

        if validation_errors:
            self.context["validation_errors"] = validation_errors


def classify_connector_error(
    error: Exception,
    connector: str,
    url: Optional[str] = None
) -> ConnectorError:
    """
    Classify generic exception as specific ConnectorError type

    Similar to classify_provider_error in intelligence/providers/errors.py

    Attempts to map common exceptions to appropriate ConnectorError subclasses.

    Args:
        error: Original exception
        connector: Connector identifier
        url: Request URL if applicable

    Returns:
        ConnectorError subclass instance

    Example:
        try:
            response = requests.get(url, timeout=30)
        except Exception as e:
            raise classify_connector_error(e, "grid/electricitymaps", url)
    """
    error_str = str(error).lower()

    # Network errors
    if "timeout" in error_str or "timed out" in error_str:
        return ConnectorTimeoutError(
            f"Request timed out: {error}",
            connector=connector,
            url=url,
            original_error=error
        )

    if any(x in error_str for x in ["connection", "network", "dns", "unreachable"]):
        return ConnectorNetworkError(
            f"Network error: {error}",
            connector=connector,
            url=url,
            original_error=error
        )

    # HTTP errors (if using requests/httpx)
    if hasattr(error, 'response'):
        response = error.response
        status_code = getattr(response, 'status_code', None)

        if status_code == 401 or status_code == 403:
            return ConnectorAuthError(
                f"Authentication failed: {error}",
                connector=connector,
                status_code=status_code,
                url=url,
                original_error=error
            )

        if status_code == 404:
            return ConnectorNotFound(
                f"Resource not found: {error}",
                connector=connector,
                status_code=status_code,
                url=url,
                original_error=error
            )

        if status_code == 429:
            retry_after = None
            if hasattr(response, 'headers'):
                retry_after = response.headers.get('Retry-After')
                if retry_after:
                    try:
                        retry_after = int(retry_after)
                    except ValueError:
                        retry_after = None

            return ConnectorRateLimit(
                f"Rate limit exceeded: {error}",
                connector=connector,
                status_code=status_code,
                retry_after=retry_after,
                url=url,
                original_error=error
            )

        if status_code is not None and 400 <= status_code < 500:
            return ConnectorBadRequest(
                f"Bad request: {error}",
                connector=connector,
                status_code=status_code,
                url=url,
                original_error=error
            )

        if status_code is not None and 500 <= status_code < 600:
            return ConnectorServerError(
                f"Server error: {error}",
                connector=connector,
                status_code=status_code,
                url=url,
                original_error=error
            )

    # Pydantic validation errors
    if error.__class__.__name__ == 'ValidationError':
        validation_errors = []
        if hasattr(error, 'errors'):
            validation_errors = error.errors()

        return ConnectorValidationError(
            f"Validation failed: {error}",
            connector=connector,
            validation_errors=validation_errors,
            original_error=error
        )

    # Default to base ConnectorError
    return ConnectorError(
        str(error),
        connector=connector,
        url=url,
        original_error=error
    )
