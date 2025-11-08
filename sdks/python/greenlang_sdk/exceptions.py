"""
GreenLang SDK Exceptions

Custom exception classes for the GreenLang SDK.
"""


class GreenLangException(Exception):
    """Base exception for all GreenLang SDK errors"""

    def __init__(self, message: str, **kwargs):
        self.message = message
        self.details = kwargs
        super().__init__(self.message)

    def __str__(self):
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


class APIException(GreenLangException):
    """
    Raised when API returns an error response

    Attributes:
        message: Error message
        status_code: HTTP status code
        response: Response data
    """

    def __init__(self, message: str, status_code: int = None, response: dict = None):
        super().__init__(message, status_code=status_code)
        self.status_code = status_code
        self.response = response or {}


class AuthenticationException(APIException):
    """
    Raised when authentication fails

    This typically means your API key is invalid or missing.
    """

    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, status_code=401)


class AuthorizationException(APIException):
    """
    Raised when authorization fails

    This means you don't have permission to access the resource.
    """

    def __init__(self, message: str = "Access denied"):
        super().__init__(message, status_code=403)


class NotFoundException(APIException):
    """
    Raised when a resource is not found

    This typically means the ID you provided doesn't exist.
    """

    def __init__(self, message: str = "Resource not found"):
        super().__init__(message, status_code=404)


class ValidationException(APIException):
    """
    Raised when request validation fails

    This means your request data is invalid.
    """

    def __init__(self, message: str = "Validation failed", field_errors: dict = None):
        super().__init__(message, status_code=422)
        self.field_errors = field_errors or {}


class RateLimitException(APIException):
    """
    Raised when rate limit is exceeded

    Wait before making more requests.

    Attributes:
        retry_after: Seconds to wait before retrying
    """

    def __init__(self, message: str = "Rate limit exceeded", retry_after: int = None):
        super().__init__(message, status_code=429)
        self.retry_after = retry_after


class TimeoutException(GreenLangException):
    """
    Raised when a request times out
    """

    def __init__(self, message: str = "Request timeout"):
        super().__init__(message)


class ConnectionException(GreenLangException):
    """
    Raised when connection to API fails
    """

    def __init__(self, message: str = "Connection error"):
        super().__init__(message)


class StreamingException(GreenLangException):
    """
    Raised when streaming encounters an error
    """

    def __init__(self, message: str = "Streaming error"):
        super().__init__(message)
