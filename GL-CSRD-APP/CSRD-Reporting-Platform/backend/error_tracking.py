"""
CSRD Reporting Platform - Error Tracking with Sentry
====================================================

Production-grade error tracking and monitoring with Sentry integration.

Features:
- Automatic error capture and reporting
- Performance monitoring
- ESRS/CSRD context in error reports
- User feedback collection
- Release tracking
- Custom error grouping

Author: GreenLang Operations Team (Team B3)
Date: 2025-11-08
"""

import os
import logging
from typing import Dict, Any, Optional, Callable
from functools import wraps
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
from sentry_sdk.integrations.redis import RedisIntegration
from sentry_sdk.integrations.logging import LoggingIntegration
from sentry_sdk.integrations.threading import ThreadingIntegration


# ============================================================================
# SENTRY CONFIGURATION
# ============================================================================

def init_sentry(
    dsn: Optional[str] = None,
    environment: str = "production",
    release: Optional[str] = None,
    traces_sample_rate: float = 0.1,
    profiles_sample_rate: float = 0.1,
    enable_tracing: bool = True
):
    """
    Initialize Sentry error tracking.

    Args:
        dsn: Sentry DSN (Data Source Name). If not provided, reads from SENTRY_DSN env var
        environment: Environment name (production, staging, development)
        release: Release version. If not provided, reads from RELEASE_VERSION env var
        traces_sample_rate: Percentage of transactions to trace (0.0 to 1.0)
        profiles_sample_rate: Percentage of transactions to profile (0.0 to 1.0)
        enable_tracing: Enable performance monitoring
    """
    # Get DSN from environment if not provided
    if dsn is None:
        dsn = os.getenv('SENTRY_DSN')

    if not dsn:
        logging.warning("Sentry DSN not provided. Error tracking is disabled.")
        return

    # Get release version from environment if not provided
    if release is None:
        release = os.getenv('RELEASE_VERSION', 'csrd-platform@1.0.0')

    # Configure logging integration
    logging_integration = LoggingIntegration(
        level=logging.INFO,        # Capture info and above as breadcrumbs
        event_level=logging.ERROR  # Send errors and above as events
    )

    # Initialize Sentry
    sentry_sdk.init(
        dsn=dsn,
        environment=environment,
        release=release,
        traces_sample_rate=traces_sample_rate if enable_tracing else 0,
        profiles_sample_rate=profiles_sample_rate if enable_tracing else 0,
        integrations=[
            FastApiIntegration(),
            SqlalchemyIntegration(),
            RedisIntegration(),
            logging_integration,
            ThreadingIntegration(propagate_hub=True)
        ],
        # Send default PII (Personally Identifiable Information)
        send_default_pii=False,  # Set to False for GDPR compliance
        # Attach stack traces
        attach_stacktrace=True,
        # Maximum breadcrumbs
        max_breadcrumbs=50,
        # Before send hook for filtering/modifying events
        before_send=before_send_handler,
        # Before breadcrumb hook for filtering/modifying breadcrumbs
        before_breadcrumb=before_breadcrumb_handler,
        # Debug mode (set to False in production)
        debug=False
    )

    logging.info(
        f"Sentry initialized - Environment: {environment}, Release: {release}"
    )


def before_send_handler(event: Dict[str, Any], hint: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Hook called before sending event to Sentry.

    Use this to filter out events, scrub sensitive data, or modify events.

    Args:
        event: Event data
        hint: Additional information about the event

    Returns:
        Modified event or None to drop the event
    """
    # Add custom logic here
    # Example: Filter out certain exceptions
    if 'exc_info' in hint:
        exc_type, exc_value, tb = hint['exc_info']
        # Don't send certain exception types
        if isinstance(exc_value, KeyboardInterrupt):
            return None

    # Scrub sensitive data from event
    if 'request' in event:
        if 'headers' in event['request']:
            # Remove sensitive headers
            sensitive_headers = ['Authorization', 'X-API-Key', 'Cookie']
            for header in sensitive_headers:
                if header in event['request']['headers']:
                    event['request']['headers'][header] = '[Filtered]'

    return event


def before_breadcrumb_handler(crumb: Dict[str, Any], hint: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Hook called before adding breadcrumb.

    Use this to filter or modify breadcrumbs.

    Args:
        crumb: Breadcrumb data
        hint: Additional information about the breadcrumb

    Returns:
        Modified breadcrumb or None to drop it
    """
    # Filter out noisy breadcrumbs
    if crumb.get('category') == 'httplib':
        # Don't log health check requests as breadcrumbs
        if '/health' in crumb.get('data', {}).get('url', ''):
            return None

    return crumb


# ============================================================================
# CONTEXT HELPERS
# ============================================================================

def set_esrs_context(
    esrs_standard: Optional[str] = None,
    company_id: Optional[str] = None,
    reporting_period: Optional[str] = None,
    data_points: Optional[int] = None
):
    """
    Set ESRS/CSRD context for error tracking.

    Args:
        esrs_standard: ESRS standard (e.g., "E1", "S1", "G1")
        company_id: Company identifier
        reporting_period: Reporting period (e.g., "2024")
        data_points: Number of data points being processed
    """
    context = {}

    if esrs_standard:
        context['esrs_standard'] = esrs_standard
    if company_id:
        context['company_id'] = company_id
    if reporting_period:
        context['reporting_period'] = reporting_period
    if data_points:
        context['data_points'] = data_points

    if context:
        sentry_sdk.set_context('esrs', context)


def set_user_context(
    user_id: Optional[str] = None,
    username: Optional[str] = None,
    email: Optional[str] = None,
    company_id: Optional[str] = None
):
    """
    Set user context for error tracking.

    Args:
        user_id: User identifier
        username: Username
        email: User email
        company_id: Company identifier
    """
    user_data = {}

    if user_id:
        user_data['id'] = user_id
    if username:
        user_data['username'] = username
    if email:
        user_data['email'] = email
    if company_id:
        user_data['company_id'] = company_id

    if user_data:
        sentry_sdk.set_user(user_data)


def set_agent_context(
    agent_name: str,
    operation: Optional[str] = None,
    input_size: Optional[int] = None,
    version: Optional[str] = None
):
    """
    Set agent context for error tracking.

    Args:
        agent_name: Name of the agent
        operation: Operation being performed
        input_size: Size of input data
        version: Agent version
    """
    context = {
        'agent_name': agent_name
    }

    if operation:
        context['operation'] = operation
    if input_size:
        context['input_size'] = input_size
    if version:
        context['version'] = version

    sentry_sdk.set_context('agent', context)


def add_breadcrumb(
    message: str,
    category: str = 'default',
    level: str = 'info',
    data: Optional[Dict[str, Any]] = None
):
    """
    Add a breadcrumb for debugging context.

    Args:
        message: Breadcrumb message
        category: Category (e.g., 'auth', 'query', 'validation')
        level: Level ('debug', 'info', 'warning', 'error', 'fatal')
        data: Additional data
    """
    sentry_sdk.add_breadcrumb(
        message=message,
        category=category,
        level=level,
        data=data or {}
    )


# ============================================================================
# ERROR CAPTURE HELPERS
# ============================================================================

def capture_exception(
    exception: Exception,
    level: str = 'error',
    tags: Optional[Dict[str, str]] = None,
    extra: Optional[Dict[str, Any]] = None
) -> str:
    """
    Capture an exception and send to Sentry.

    Args:
        exception: Exception to capture
        level: Severity level ('fatal', 'error', 'warning', 'info', 'debug')
        tags: Additional tags
        extra: Additional extra data

    Returns:
        Event ID
    """
    with sentry_sdk.push_scope() as scope:
        scope.level = level

        if tags:
            for key, value in tags.items():
                scope.set_tag(key, value)

        if extra:
            for key, value in extra.items():
                scope.set_extra(key, value)

        event_id = sentry_sdk.capture_exception(exception)

    return event_id


def capture_message(
    message: str,
    level: str = 'info',
    tags: Optional[Dict[str, str]] = None,
    extra: Optional[Dict[str, Any]] = None
) -> str:
    """
    Capture a message and send to Sentry.

    Args:
        message: Message to capture
        level: Severity level ('fatal', 'error', 'warning', 'info', 'debug')
        tags: Additional tags
        extra: Additional extra data

    Returns:
        Event ID
    """
    with sentry_sdk.push_scope() as scope:
        scope.level = level

        if tags:
            for key, value in tags.items():
                scope.set_tag(key, value)

        if extra:
            for key, value in extra.items():
                scope.set_extra(key, value)

        event_id = sentry_sdk.capture_message(message)

    return event_id


# ============================================================================
# DECORATORS
# ============================================================================

def monitor_errors(
    agent_name: Optional[str] = None,
    operation: Optional[str] = None,
    capture_args: bool = False
):
    """
    Decorator to monitor function errors and send to Sentry.

    Args:
        agent_name: Name of the agent
        operation: Operation being performed
        capture_args: Whether to capture function arguments

    Usage:
        @monitor_errors(agent_name="intake", operation="process_data")
        def process_data(data):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Set agent context if provided
            if agent_name:
                set_agent_context(agent_name=agent_name, operation=operation)

            # Add breadcrumb
            add_breadcrumb(
                message=f"Calling {func.__name__}",
                category='function_call',
                data={
                    'function': func.__name__,
                    'agent': agent_name,
                    'operation': operation
                }
            )

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                # Capture function arguments if enabled
                extra_data = {}
                if capture_args:
                    extra_data['args'] = str(args)[:500]  # Limit size
                    extra_data['kwargs'] = str(kwargs)[:500]  # Limit size

                # Capture exception
                capture_exception(
                    exception=e,
                    tags={
                        'function': func.__name__,
                        'agent': agent_name or 'unknown',
                        'operation': operation or 'unknown'
                    },
                    extra=extra_data
                )
                raise

        return wrapper
    return decorator


def monitor_performance(operation_name: Optional[str] = None):
    """
    Decorator to monitor function performance.

    Args:
        operation_name: Name of the operation

    Usage:
        @monitor_performance(operation_name="data_validation")
        def validate_data(data):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__

            with sentry_sdk.start_transaction(op=op_name, name=func.__name__) as transaction:
                try:
                    result = func(*args, **kwargs)
                    transaction.set_status('ok')
                    return result
                except Exception as e:
                    transaction.set_status('internal_error')
                    raise

        return wrapper
    return decorator


# ============================================================================
# VALIDATION ERROR TRACKING
# ============================================================================

def track_validation_error(
    esrs_standard: str,
    error_type: str,
    error_message: str,
    company_id: Optional[str] = None,
    data_point: Optional[str] = None,
    severity: str = 'warning'
):
    """
    Track a validation error.

    Args:
        esrs_standard: ESRS standard
        error_type: Type of validation error
        error_message: Error message
        company_id: Company ID
        data_point: Specific data point that failed
        severity: Severity level
    """
    set_esrs_context(esrs_standard=esrs_standard, company_id=company_id)

    capture_message(
        message=f"Validation error in {esrs_standard}: {error_message}",
        level=severity,
        tags={
            'validation_error': error_type,
            'esrs_standard': esrs_standard,
            'error_type': error_type
        },
        extra={
            'data_point': data_point,
            'company_id': company_id
        }
    )


def track_compliance_issue(
    issue_type: str,
    description: str,
    company_id: str,
    deadline: Optional[str] = None,
    severity: str = 'error'
):
    """
    Track a compliance issue.

    Args:
        issue_type: Type of compliance issue
        description: Issue description
        company_id: Company ID
        deadline: Compliance deadline
        severity: Severity level
    """
    capture_message(
        message=f"Compliance issue: {description}",
        level=severity,
        tags={
            'compliance_issue': issue_type,
            'company_id': company_id
        },
        extra={
            'deadline': deadline,
            'issue_type': issue_type
        }
    )


# ============================================================================
# PERFORMANCE MONITORING
# ============================================================================

class PerformanceMonitor:
    """Context manager for performance monitoring."""

    def __init__(self, operation: str, description: Optional[str] = None):
        """
        Initialize performance monitor.

        Args:
            operation: Operation name
            description: Operation description
        """
        self.operation = operation
        self.description = description
        self.transaction = None

    def __enter__(self):
        """Start transaction."""
        self.transaction = sentry_sdk.start_transaction(
            op=self.operation,
            name=self.description or self.operation
        )
        self.transaction.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End transaction."""
        if exc_type is not None:
            self.transaction.set_status('internal_error')
        else:
            self.transaction.set_status('ok')

        self.transaction.__exit__(exc_type, exc_val, exc_tb)

    def add_span(self, operation: str, description: Optional[str] = None):
        """
        Add a span to the transaction.

        Args:
            operation: Span operation
            description: Span description

        Returns:
            Span context manager
        """
        return self.transaction.start_child(
            op=operation,
            description=description or operation
        )


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    # Initialize Sentry
    init_sentry(
        dsn="https://your-sentry-dsn@sentry.io/project-id",
        environment="development",
        release="csrd-platform@1.0.0"
    )

    # Set ESRS context
    set_esrs_context(
        esrs_standard="E1",
        company_id="comp-123",
        reporting_period="2024"
    )

    # Add breadcrumb
    add_breadcrumb(
        message="Processing climate data",
        category="data_processing",
        level="info",
        data={"records": 150}
    )

    # Capture exception
    try:
        raise ValueError("Invalid data point")
    except ValueError as e:
        capture_exception(
            exception=e,
            tags={"esrs_standard": "E1"},
            extra={"data_point": "emissions_scope_1"}
        )

    # Track validation error
    track_validation_error(
        esrs_standard="E1",
        error_type="missing_data",
        error_message="Scope 1 emissions data missing",
        company_id="comp-123",
        data_point="emissions_scope_1"
    )

    # Performance monitoring
    with PerformanceMonitor("data_validation", "Validate ESRS E1 data"):
        # ... do validation work ...
        pass

    print("Error tracking examples completed!")
