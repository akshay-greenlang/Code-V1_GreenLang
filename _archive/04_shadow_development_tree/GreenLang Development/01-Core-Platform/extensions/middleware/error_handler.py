# -*- coding: utf-8 -*-
"""Error Handling Middleware for GreenLang.

Provides centralized error handling, formatting, and logging for all GreenLang
exceptions across agents, workflows, and API endpoints.

Features:
- Catches and formats GreenLang custom exceptions
- Provides consistent error response format
- Logs errors with rich context for debugging
- Supports retry logic for retriable errors
- Integrates with monitoring and alerting

Example:
    >>> from greenlang.middleware.error_handler import ErrorHandler
    >>> handler = ErrorHandler()
    >>>
    >>> try:
    ...     # Agent execution
    ...     result = agent.run(payload)
    ... except Exception as e:
    ...     error_response = handler.handle_error(e)
    ...     return error_response

Author: GreenLang Framework Team
Date: October 2025
Status: Production Ready
"""

from typing import Any, Dict, Optional, Callable
from datetime import datetime
import logging
import sys
import traceback

from greenlang.utilities.determinism import DeterministicClock
from greenlang.exceptions import (
    GreenLangException,
    AgentException,
    WorkflowException,
    DataException,
    ValidationError,
    ExecutionError,
    TimeoutError,
    format_exception_chain,
    is_retriable,
)


logger = logging.getLogger(__name__)


# ==============================================================================
# Error Response Format
# ==============================================================================

class ErrorResponse:
    """Standardized error response format.

    Provides consistent error structure for API responses, logging, and monitoring.

    Attributes:
        success: Always False for errors
        error: Error details dictionary
        timestamp: When error occurred
        request_id: Optional request ID for tracing
    """

    def __init__(
        self,
        error_type: str,
        error_code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ):
        """Initialize error response.

        Args:
            error_type: Type of error (ValidationError, ExecutionError, etc.)
            error_code: Unique error code (GL_AGENT_VALIDATION_001, etc.)
            message: Human-readable error message
            details: Additional error details and context
            request_id: Optional request ID for tracing
        """
        self.success = False
        self.error = {
            "type": error_type,
            "code": error_code,
            "message": message,
            "details": details or {},
        }
        self.timestamp = DeterministicClock.now().isoformat()
        self.request_id = request_id

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        response = {
            "success": self.success,
            "error": self.error,
            "timestamp": self.timestamp,
        }
        if self.request_id:
            response["request_id"] = self.request_id
        return response

    def __repr__(self) -> str:
        return f"ErrorResponse(type={self.error['type']}, code={self.error['code']})"


# ==============================================================================
# Error Handler
# ==============================================================================

class ErrorHandler:
    """Centralized error handler for GreenLang.

    Handles error catching, formatting, logging, and recovery for all GreenLang
    exceptions.

    Features:
    - Consistent error formatting
    - Rich error logging with context
    - Retry logic for retriable errors
    - Integration with monitoring/alerting
    - Graceful degradation for unknown errors
    """

    def __init__(
        self,
        log_level: int = logging.ERROR,
        include_traceback: bool = True,
        alert_on_critical: bool = False,
    ):
        """Initialize error handler.

        Args:
            log_level: Logging level for errors (default: ERROR)
            include_traceback: Include full traceback in logs (default: True)
            alert_on_critical: Send alerts for critical errors (default: False)
        """
        self.log_level = log_level
        self.include_traceback = include_traceback
        self.alert_on_critical = alert_on_critical
        self.logger = logging.getLogger(__name__)

    def handle_error(
        self,
        exception: Exception,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ErrorResponse:
        """Handle exception and return formatted error response.

        Args:
            exception: Exception to handle
            request_id: Optional request ID for tracing
            context: Additional context for error

        Returns:
            ErrorResponse with formatted error details
        """
        # Log the error
        self._log_error(exception, request_id, context)

        # Format error response
        if isinstance(exception, GreenLangException):
            response = self._handle_greenlang_exception(exception, request_id)
        else:
            response = self._handle_unknown_exception(exception, request_id, context)

        # Alert if critical
        if self.alert_on_critical and self._is_critical(exception):
            self._send_alert(exception, request_id)

        return response

    def _handle_greenlang_exception(
        self,
        exception: GreenLangException,
        request_id: Optional[str] = None,
    ) -> ErrorResponse:
        """Handle GreenLang custom exception.

        Args:
            exception: GreenLang exception
            request_id: Optional request ID

        Returns:
            Formatted error response
        """
        return ErrorResponse(
            error_type=exception.__class__.__name__,
            error_code=exception.error_code,
            message=exception.message,
            details={
                "agent_name": exception.agent_name,
                "context": exception.context,
                "timestamp": exception.timestamp.isoformat(),
                "retriable": is_retriable(exception),
            },
            request_id=request_id,
        )

    def _handle_unknown_exception(
        self,
        exception: Exception,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ErrorResponse:
        """Handle unknown/unhandled exception.

        Args:
            exception: Unknown exception
            request_id: Optional request ID
            context: Additional context

        Returns:
            Formatted error response
        """
        return ErrorResponse(
            error_type=exception.__class__.__name__,
            error_code="GL_UNKNOWN_ERROR",
            message=str(exception),
            details={
                "context": context or {},
                "exception_module": exception.__class__.__module__,
                "retriable": False,
            },
            request_id=request_id,
        )

    def _log_error(
        self,
        exception: Exception,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log error with rich context.

        Args:
            exception: Exception to log
            request_id: Optional request ID
            context: Additional context
        """
        log_data = {
            "error_type": exception.__class__.__name__,
            "message": str(exception),
            "request_id": request_id,
        }

        # Add GreenLang exception details
        if isinstance(exception, GreenLangException):
            log_data.update({
                "error_code": exception.error_code,
                "agent_name": exception.agent_name,
                "context": exception.context,
                "retriable": is_retriable(exception),
            })

        # Add additional context
        if context:
            log_data["additional_context"] = context

        # Add traceback if enabled
        if self.include_traceback:
            log_data["traceback"] = traceback.format_exc()

        # Log at appropriate level
        if isinstance(exception, ValidationError):
            self.logger.warning("Validation error: %s", log_data)
        elif isinstance(exception, (TimeoutError, ResourceError)):
            self.logger.warning("Retriable error: %s", log_data)
        elif isinstance(exception, (AgentException, WorkflowException, DataException)):
            self.logger.error("GreenLang error: %s", log_data)
        else:
            self.logger.error("Unknown error: %s", log_data)

    def _is_critical(self, exception: Exception) -> bool:
        """Check if exception is critical and requires alerting.

        Args:
            exception: Exception to check

        Returns:
            True if critical
        """
        # Critical errors that need immediate attention
        critical_types = (
            ExecutionError,
            WorkflowException,
            DataAccessError,
        )

        return isinstance(exception, critical_types)

    def _send_alert(
        self,
        exception: Exception,
        request_id: Optional[str] = None,
    ) -> None:
        """Send alert for critical error.

        Args:
            exception: Critical exception
            request_id: Optional request ID
        """
        alert_message = f"CRITICAL ERROR: {exception.__class__.__name__}: {str(exception)}"
        if request_id:
            alert_message += f" (Request ID: {request_id})"

        self.logger.critical(alert_message)

    def with_error_handling(
        self,
        func: Callable,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Decorator for automatic error handling.

        Args:
            func: Function to wrap with error handling
            request_id: Optional request ID
            context: Additional context

        Returns:
            Function result or ErrorResponse on error
        """
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                return self.handle_error(e, request_id, context)

        return wrapper


# ==============================================================================
# Error Handler Decorator
# ==============================================================================

def handle_errors(
    log_level: int = logging.ERROR,
    include_traceback: bool = True,
):
    """Decorator for automatic error handling.

    Example:
        >>> @handle_errors()
        ... def my_function():
        ...     raise ValidationError("Invalid input")
        ...
        >>> result = my_function()
        >>> assert result.success is False

    Args:
        log_level: Logging level for errors
        include_traceback: Include full traceback in logs

    Returns:
        Decorated function with automatic error handling
    """
    handler = ErrorHandler(
        log_level=log_level,
        include_traceback=include_traceback,
    )

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                return handler.handle_error(e)

        return wrapper

    return decorator


# ==============================================================================
# Retry Logic
# ==============================================================================

class RetryHandler:
    """Retry handler for retriable errors.

    Implements exponential backoff retry logic for retriable exceptions.
    """

    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 30.0,
        backoff_factor: float = 2.0,
    ):
        """Initialize retry handler.

        Args:
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds
            backoff_factor: Exponential backoff multiplier
        """
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor

    def retry_with_backoff(
        self,
        func: Callable,
        *args,
        **kwargs,
    ) -> Any:
        """Execute function with retry logic.

        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            Exception if all retries exhausted
        """
        import time

        delay = self.initial_delay
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e

                # Check if retriable
                if not is_retriable(e):
                    raise

                # Last attempt - don't sleep
                if attempt == self.max_retries:
                    break

                # Log retry
                logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries + 1} failed, "
                    f"retrying in {delay}s: {e}"
                )

                # Sleep with exponential backoff
                time.sleep(delay)
                delay = min(delay * self.backoff_factor, self.max_delay)

        # All retries exhausted
        raise last_exception


# ==============================================================================
# Global Error Handler Instance
# ==============================================================================

# Global error handler instance for convenience
_global_error_handler = ErrorHandler()


def handle_error(
    exception: Exception,
    request_id: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
) -> ErrorResponse:
    """Handle error using global error handler.

    Convenience function for quick error handling without creating ErrorHandler instance.

    Args:
        exception: Exception to handle
        request_id: Optional request ID
        context: Additional context

    Returns:
        ErrorResponse with formatted error
    """
    return _global_error_handler.handle_error(exception, request_id, context)
