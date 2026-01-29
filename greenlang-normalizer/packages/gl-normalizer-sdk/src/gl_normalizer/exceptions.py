"""
GL-FOUND-X-003: GreenLang Normalizer SDK - Exceptions

This module defines the exception hierarchy for the GreenLang Normalizer SDK.
All exceptions follow the GLNORM error code taxonomy for consistency with
the backend API.

Example:
    >>> from gl_normalizer.exceptions import ValidationError
    >>> try:
    ...     raise ValidationError("GLNORM-E100", "Unit parse failed", path="/unit")
    ... except ValidationError as e:
    ...     print(f"Error {e.code}: {e.message}")
"""

from typing import Any, Dict, Optional


class NormalizerError(Exception):
    """
    Base exception for all GreenLang Normalizer SDK errors.

    All SDK exceptions inherit from this class, allowing for broad
    exception handling when needed.

    Attributes:
        message: Human-readable error description.
        code: GLNORM error code (e.g., "GLNORM-E100").
        details: Additional error context and metadata.

    Example:
        >>> try:
        ...     client.normalize(100, "invalid")
        ... except NormalizerError as e:
        ...     print(f"Normalization failed: {e}")
    """

    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize NormalizerError.

        Args:
            message: Human-readable error description.
            code: Optional GLNORM error code.
            details: Optional additional error context.
        """
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}

    def __str__(self) -> str:
        """Return string representation with code if available."""
        if self.code:
            return f"[{self.code}] {self.message}"
        return self.message

    def __repr__(self) -> str:
        """Return detailed representation for debugging."""
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"code={self.code!r}, "
            f"details={self.details!r})"
        )


class ConfigurationError(NormalizerError):
    """
    Exception raised for SDK configuration errors.

    Raised when the SDK is configured with invalid parameters,
    such as missing API key or invalid base URL.

    Example:
        >>> client = NormalizerClient(api_key="")
        ConfigurationError: API key is required
    """

    pass


class ValidationError(NormalizerError):
    """
    Exception raised for input validation failures.

    Corresponds to GLNORM-E1xx (unit parsing) and GLNORM-E2xx (dimension)
    error codes from the API.

    Attributes:
        path: JSON path to the invalid field.
        expected: What was expected.
        actual: What was provided.
        hint: Suggestion for fixing the error.

    Example:
        >>> try:
        ...     result = client.normalize(100, "kg", expected_dimension="energy")
        ... except ValidationError as e:
        ...     print(f"Path: {e.path}, Hint: {e.hint}")
    """

    def __init__(
        self,
        code: str,
        message: str,
        path: Optional[str] = None,
        expected: Optional[Any] = None,
        actual: Optional[Any] = None,
        hint: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize ValidationError.

        Args:
            code: GLNORM error code (E1xx or E2xx).
            message: Human-readable error description.
            path: JSON path to the invalid field.
            expected: What was expected.
            actual: What was provided.
            hint: Suggestion for fixing the error.
            details: Additional error context.
        """
        super().__init__(message, code, details)
        self.path = path
        self.expected = expected
        self.actual = actual
        self.hint = hint


class ConversionError(NormalizerError):
    """
    Exception raised for unit conversion failures.

    Corresponds to GLNORM-E3xx error codes from the API.

    Attributes:
        from_unit: Source unit that failed to convert.
        to_unit: Target unit for conversion.
        path: JSON path to the field that failed.
        hint: Suggestion for fixing the error.

    Example:
        >>> try:
        ...     result = client.normalize(100, "Nm3")  # Missing reference conditions
        ... except ConversionError as e:
        ...     print(f"Missing: {e.hint}")
    """

    def __init__(
        self,
        code: str,
        message: str,
        from_unit: Optional[str] = None,
        to_unit: Optional[str] = None,
        path: Optional[str] = None,
        hint: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize ConversionError.

        Args:
            code: GLNORM error code (E3xx).
            message: Human-readable error description.
            from_unit: Source unit.
            to_unit: Target unit.
            path: JSON path to the field.
            hint: Suggestion for fixing the error.
            details: Additional error context.
        """
        super().__init__(message, code, details)
        self.from_unit = from_unit
        self.to_unit = to_unit
        self.path = path
        self.hint = hint


class ResolutionError(NormalizerError):
    """
    Exception raised for entity resolution failures.

    Corresponds to GLNORM-E4xx error codes from the API.

    Attributes:
        entity_type: Type of entity that failed resolution (fuel, material, process).
        raw_name: Original name that could not be resolved.
        candidates: List of potential matches if ambiguous.
        confidence: Confidence score if low confidence match.
        path: JSON path to the field.
        hint: Suggestion for fixing the error.

    Example:
        >>> try:
        ...     result = client.resolve_entity("Unknown Fuel", entity_type="fuel")
        ... except ResolutionError as e:
        ...     if e.candidates:
        ...         print(f"Did you mean: {e.candidates}")
    """

    def __init__(
        self,
        code: str,
        message: str,
        entity_type: Optional[str] = None,
        raw_name: Optional[str] = None,
        candidates: Optional[list[Dict[str, Any]]] = None,
        confidence: Optional[float] = None,
        path: Optional[str] = None,
        hint: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize ResolutionError.

        Args:
            code: GLNORM error code (E4xx).
            message: Human-readable error description.
            entity_type: Type of entity.
            raw_name: Original name.
            candidates: Potential matches.
            confidence: Confidence score.
            path: JSON path.
            hint: Suggestion for fixing.
            details: Additional context.
        """
        super().__init__(message, code, details)
        self.entity_type = entity_type
        self.raw_name = raw_name
        self.candidates = candidates or []
        self.confidence = confidence
        self.path = path
        self.hint = hint


class VocabularyError(NormalizerError):
    """
    Exception raised for vocabulary-related failures.

    Corresponds to GLNORM-E5xx error codes from the API.

    Attributes:
        vocabulary_version: Version that caused the error.
        path: JSON path to the field.
        hint: Suggestion for fixing the error.

    Example:
        >>> try:
        ...     result = client.normalize(100, "kWh", vocabulary_version="invalid")
        ... except VocabularyError as e:
        ...     print(f"Vocabulary error: {e.code}")
    """

    def __init__(
        self,
        code: str,
        message: str,
        vocabulary_version: Optional[str] = None,
        path: Optional[str] = None,
        hint: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize VocabularyError.

        Args:
            code: GLNORM error code (E5xx).
            message: Human-readable error description.
            vocabulary_version: Version that caused error.
            path: JSON path.
            hint: Suggestion for fixing.
            details: Additional context.
        """
        super().__init__(message, code, details)
        self.vocabulary_version = vocabulary_version
        self.path = path
        self.hint = hint


class AuditError(NormalizerError):
    """
    Exception raised for audit-related failures.

    Corresponds to GLNORM-E6xx error codes from the API.
    These are typically server-side issues that may be retriable.

    Example:
        >>> try:
        ...     result = client.normalize(100, "kWh")
        ... except AuditError as e:
        ...     print(f"Audit failed: {e.code}")
    """

    pass


class APIError(NormalizerError):
    """
    Exception raised for HTTP API errors.

    Raised when the API returns an unexpected HTTP status code
    or the response cannot be parsed.

    Attributes:
        status_code: HTTP status code.
        response_body: Raw response body if available.

    Example:
        >>> try:
        ...     result = client.normalize(100, "kWh")
        ... except APIError as e:
        ...     print(f"HTTP {e.status_code}: {e.message}")
    """

    def __init__(
        self,
        message: str,
        status_code: int,
        code: Optional[str] = None,
        response_body: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize APIError.

        Args:
            message: Human-readable error description.
            status_code: HTTP status code.
            code: Optional GLNORM error code.
            response_body: Raw response body.
            details: Additional context.
        """
        super().__init__(message, code, details)
        self.status_code = status_code
        self.response_body = response_body


class RateLimitError(APIError):
    """
    Exception raised when API rate limit is exceeded.

    Corresponds to GLNORM-E900 (LIMIT_EXCEEDED) and HTTP 429.

    Attributes:
        retry_after: Number of seconds to wait before retrying.

    Example:
        >>> try:
        ...     result = client.normalize(100, "kWh")
        ... except RateLimitError as e:
        ...     time.sleep(e.retry_after)
        ...     result = client.normalize(100, "kWh")  # Retry
    """

    def __init__(
        self,
        message: str,
        retry_after: Optional[float] = None,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize RateLimitError.

        Args:
            message: Human-readable error description.
            retry_after: Seconds to wait before retry.
            code: GLNORM error code.
            details: Additional context.
        """
        super().__init__(message, status_code=429, code=code or "GLNORM-E900", details=details)
        self.retry_after = retry_after or 60.0


class TimeoutError(NormalizerError):
    """
    Exception raised when a request times out.

    Corresponds to GLNORM-E901 (TIMEOUT).

    Attributes:
        timeout: The timeout value that was exceeded.

    Example:
        >>> try:
        ...     result = client.normalize_batch(large_batch)
        ... except TimeoutError as e:
        ...     print(f"Request timed out after {e.timeout}s")
    """

    def __init__(
        self,
        message: str,
        timeout: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize TimeoutError.

        Args:
            message: Human-readable error description.
            timeout: The timeout value that was exceeded.
            details: Additional context.
        """
        super().__init__(message, code="GLNORM-E901", details=details)
        self.timeout = timeout


class ServiceUnavailableError(APIError):
    """
    Exception raised when the service is unavailable.

    Corresponds to GLNORM-E903 (SERVICE_UNAVAILABLE) and HTTP 503.

    Example:
        >>> try:
        ...     result = client.normalize(100, "kWh")
        ... except ServiceUnavailableError as e:
        ...     # Implement circuit breaker or fallback
        ...     pass
    """

    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize ServiceUnavailableError.

        Args:
            message: Human-readable error description.
            code: GLNORM error code.
            details: Additional context.
        """
        super().__init__(
            message, status_code=503, code=code or "GLNORM-E903", details=details
        )


class ConnectionError(NormalizerError):
    """
    Exception raised for network connectivity issues.

    Raised when the SDK cannot establish a connection to the API.

    Example:
        >>> try:
        ...     result = client.normalize(100, "kWh")
        ... except ConnectionError as e:
        ...     print(f"Connection failed: {e}")
    """

    pass


class JobError(NormalizerError):
    """
    Exception raised for async job processing failures.

    Raised when an async job fails or is cancelled.

    Attributes:
        job_id: The ID of the failed job.
        job_status: Final status of the job.

    Example:
        >>> try:
        ...     job = client.create_job(requests)
        ...     job.wait()
        ... except JobError as e:
        ...     print(f"Job {e.job_id} failed: {e.message}")
    """

    def __init__(
        self,
        message: str,
        job_id: Optional[str] = None,
        job_status: Optional[str] = None,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize JobError.

        Args:
            message: Human-readable error description.
            job_id: The failed job ID.
            job_status: Final job status.
            code: GLNORM error code.
            details: Additional context.
        """
        super().__init__(message, code, details)
        self.job_id = job_id
        self.job_status = job_status


def raise_for_error_response(response_data: Dict[str, Any], status_code: int) -> None:
    """
    Raise appropriate exception based on API error response.

    This function maps API error responses to SDK exceptions based on
    the GLNORM error code taxonomy.

    Args:
        response_data: Parsed JSON response from the API.
        status_code: HTTP status code.

    Raises:
        ValidationError: For E1xx and E2xx errors.
        ConversionError: For E3xx errors.
        ResolutionError: For E4xx errors.
        VocabularyError: For E5xx errors.
        AuditError: For E6xx errors.
        RateLimitError: For rate limit (429) responses.
        ServiceUnavailableError: For service unavailable (503) responses.
        APIError: For other HTTP errors.

    Example:
        >>> response = httpx.get(url)
        >>> if response.status_code >= 400:
        ...     raise_for_error_response(response.json(), response.status_code)
    """
    # Extract error details from response
    errors = response_data.get("errors", [])
    error = errors[0] if errors else response_data

    code = error.get("code", "")
    message = error.get("message", "Unknown error")
    path = error.get("path")
    hint_data = error.get("hint", {})
    hint = hint_data.get("suggestion") if isinstance(hint_data, dict) else None
    expected = error.get("expected")
    actual = error.get("actual")
    details = error.get("details", {})

    # Rate limit
    if status_code == 429:
        retry_after = response_data.get("retry_after")
        raise RateLimitError(message, retry_after=retry_after, code=code, details=details)

    # Service unavailable
    if status_code == 503:
        raise ServiceUnavailableError(message, code=code, details=details)

    # Map by error code category
    if code.startswith("GLNORM-E1") or code.startswith("GLNORM-E2"):
        raise ValidationError(
            code=code,
            message=message,
            path=path,
            expected=expected,
            actual=actual,
            hint=hint,
            details=details,
        )
    elif code.startswith("GLNORM-E3"):
        raise ConversionError(
            code=code,
            message=message,
            from_unit=actual.get("unit") if isinstance(actual, dict) else None,
            to_unit=expected.get("unit") if isinstance(expected, dict) else None,
            path=path,
            hint=hint,
            details=details,
        )
    elif code.startswith("GLNORM-E4"):
        raise ResolutionError(
            code=code,
            message=message,
            entity_type=actual.get("entity_type") if isinstance(actual, dict) else None,
            raw_name=actual.get("raw_name") if isinstance(actual, dict) else None,
            candidates=response_data.get("candidates"),
            confidence=response_data.get("confidence"),
            path=path,
            hint=hint,
            details=details,
        )
    elif code.startswith("GLNORM-E5"):
        raise VocabularyError(
            code=code,
            message=message,
            vocabulary_version=response_data.get("vocabulary_version"),
            path=path,
            hint=hint,
            details=details,
        )
    elif code.startswith("GLNORM-E6"):
        raise AuditError(message, code=code, details=details)
    elif code.startswith("GLNORM-E9"):
        if code == "GLNORM-E901":
            raise TimeoutError(message, details=details)
        elif code == "GLNORM-E903":
            raise ServiceUnavailableError(message, code=code, details=details)
        else:
            raise APIError(message, status_code=status_code, code=code, details=details)
    else:
        raise APIError(message, status_code=status_code, code=code, details=details)
