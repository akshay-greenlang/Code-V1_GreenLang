"""
Provider Error Taxonomy

Normalized error types across all LLM providers:
- ProviderAuthError: API key invalid or expired
- ProviderRateLimit: Rate limit exceeded (429)
- ProviderTimeout: Request timeout
- ProviderServerError: Provider infrastructure error (5xx)
- ProviderBadRequest: Invalid request parameters (4xx)

Enables:
- Consistent error handling across providers
- Retry logic classification (which errors to retry)
- Error attribution (which provider failed)
"""

from __future__ import annotations
from typing import Optional


class ProviderError(Exception):
    """
    Base class for all provider errors

    Attributes:
        message: Error description
        provider: Provider name (openai, anthropic, etc.)
        status_code: HTTP status code (if applicable)
        request_id: Provider's request ID (for support)
        original_error: Underlying exception
    """

    def __init__(
        self,
        message: str,
        provider: str,
        status_code: Optional[int] = None,
        request_id: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.status_code = status_code
        self.request_id = request_id
        self.original_error = original_error

    def __str__(self) -> str:
        parts = [f"[{self.provider}] {self.message}"]
        if self.status_code:
            parts.append(f"Status: {self.status_code}")
        if self.request_id:
            parts.append(f"Request ID: {self.request_id}")
        return " | ".join(parts)


class ProviderAuthError(ProviderError):
    """
    Authentication error

    Causes:
    - API key missing
    - API key invalid
    - API key expired
    - Insufficient permissions

    HTTP Status: 401, 403

    Transient: NO (don't retry)
    """

    pass


class ProviderRateLimit(ProviderError):
    """
    Rate limit exceeded

    Causes:
    - Too many requests per minute/hour
    - Token quota exceeded
    - Concurrent request limit

    HTTP Status: 429

    Transient: YES (retry with backoff)

    Attributes:
        retry_after: Seconds to wait before retrying (from Retry-After header)
    """

    def __init__(
        self,
        message: str,
        provider: str,
        retry_after: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(message, provider, status_code=429, **kwargs)
        self.retry_after = retry_after

    def __str__(self) -> str:
        s = super().__str__()
        if self.retry_after:
            s += f" | Retry after: {self.retry_after}s"
        return s


class ProviderTimeout(ProviderError):
    """
    Request timeout

    Causes:
    - Network latency
    - Provider overloaded
    - Long generation (hit timeout limit)

    HTTP Status: 408, 504

    Transient: YES (retry)

    Attributes:
        timeout_seconds: Timeout duration that was exceeded
    """

    def __init__(
        self,
        message: str,
        provider: str,
        timeout_seconds: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(message, provider, **kwargs)
        self.timeout_seconds = timeout_seconds


class ProviderServerError(ProviderError):
    """
    Provider infrastructure error

    Causes:
    - Provider service down
    - Internal server error
    - Gateway timeout

    HTTP Status: 500, 502, 503, 504

    Transient: YES (retry)
    """

    pass


class ProviderBadRequest(ProviderError):
    """
    Invalid request parameters

    Causes:
    - Invalid model name
    - Malformed prompt
    - Invalid tool schema
    - Parameter validation failed

    HTTP Status: 400, 422

    Transient: NO (don't retry, fix code)
    """

    pass


class ProviderContentFilter(ProviderError):
    """
    Content filtered by safety system

    Causes:
    - Prompt contains prohibited content
    - Response contains prohibited content
    - Safety threshold exceeded

    HTTP Status: 400 (sometimes)

    Transient: NO (don't retry, modify prompt)
    """

    pass


def classify_provider_error(
    error: Exception,
    provider: str,
    status_code: Optional[int] = None,
    error_message: Optional[str] = None,
) -> ProviderError:
    """
    Classify a raw provider exception into normalized error type

    Args:
        error: Original exception from provider SDK
        provider: Provider name
        status_code: HTTP status code (if available)
        error_message: Error message (if different from exception message)

    Returns:
        Normalized ProviderError subclass

    Example:
        try:
            openai.ChatCompletion.create(...)
        except Exception as e:
            # Classify into normalized error
            normalized = classify_provider_error(
                error=e,
                provider="openai",
                status_code=getattr(e, 'status_code', None)
            )

            # Now can handle consistently
            if isinstance(normalized, ProviderRateLimit):
                await asyncio.sleep(normalized.retry_after or 60)
                # retry...
    """
    message = error_message or str(error)

    # Classify by status code
    if status_code:
        if status_code == 401 or status_code == 403:
            return ProviderAuthError(message, provider, status_code=status_code)
        elif status_code == 429:
            # Try to extract retry_after from error
            retry_after = getattr(error, "retry_after", None)
            return ProviderRateLimit(
                message, provider, retry_after=retry_after, status_code=status_code
            )
        elif status_code == 408 or status_code == 504:
            return ProviderTimeout(message, provider, status_code=status_code)
        elif status_code >= 500:
            return ProviderServerError(message, provider, status_code=status_code)
        elif status_code == 400 or status_code == 422:
            # Check if content filter
            if any(
                keyword in message.lower()
                for keyword in ["content", "filter", "safety", "policy"]
            ):
                return ProviderContentFilter(message, provider, status_code=status_code)
            return ProviderBadRequest(message, provider, status_code=status_code)

    # Classify by error message patterns
    message_lower = message.lower()

    if any(
        keyword in message_lower
        for keyword in ["auth", "api key", "unauthorized", "forbidden"]
    ):
        return ProviderAuthError(message, provider, original_error=error)

    if any(keyword in message_lower for keyword in ["rate limit", "429", "quota"]):
        return ProviderRateLimit(message, provider, original_error=error)

    if any(
        keyword in message_lower for keyword in ["timeout", "timed out", "deadline"]
    ):
        return ProviderTimeout(message, provider, original_error=error)

    if any(
        keyword in message_lower for keyword in ["server error", "500", "503", "502"]
    ):
        return ProviderServerError(message, provider, original_error=error)

    if any(keyword in message_lower for keyword in ["content", "filter", "safety"]):
        return ProviderContentFilter(message, provider, original_error=error)

    # Default to bad request
    return ProviderBadRequest(message, provider, original_error=error)
