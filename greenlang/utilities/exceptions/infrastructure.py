"""
GreenLang Infrastructure Exceptions - Database, Cache, Storage, and Resilience Errors

This module provides exception classes for infrastructure-related errors
including database operations, caching, object storage, message queues,
and resilience patterns (circuit breaker, bulkhead, retry).

Features:
- Database connection and query errors
- Cache read/write failures
- Object storage access errors
- Message queue failures
- Circuit breaker open state
- Bulkhead saturation
- Retry exhaustion tracking

Author: GreenLang Team
Date: 2026-04-02
"""

from typing import Any, Dict, Optional

from greenlang.exceptions.base import GreenLangException


class InfrastructureException(GreenLangException):
    """Base exception for infrastructure-related errors.

    Raised when infrastructure components (database, cache, storage, queue)
    encounter operational failures.
    """
    ERROR_PREFIX = "GL_INFRA"


class DatabaseError(InfrastructureException):
    """Database operation failed.

    Raised when a database connection, query, or transaction fails.
    Covers PostgreSQL, TimescaleDB, and pgvector operations.

    Example:
        >>> raise DatabaseError(
        ...     message="Connection pool exhausted",
        ...     database="emissions_db",
        ...     operation="SELECT",
        ...     cause=original_exception
        ... )
    """

    def __init__(
        self,
        message: str,
        database: Optional[str] = None,
        operation: Optional[str] = None,
        cause: Optional[Exception] = None,
        agent_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Initialize database error.

        Args:
            message: Error message
            database: Database name or connection identifier
            operation: Operation that failed (SELECT, INSERT, UPDATE, DELETE)
            cause: Original database exception
            agent_name: Name of agent
            context: Error context
        """
        context = context or {}
        if database:
            context["database"] = database
        if operation:
            context["operation"] = operation
        if cause:
            context["cause"] = str(cause)
            context["cause_type"] = type(cause).__name__
        super().__init__(message, agent_name=agent_name, context=context)


class CacheError(InfrastructureException):
    """Cache operation failed.

    Raised when a Redis cache read, write, or eviction operation fails.

    Example:
        >>> raise CacheError(
        ...     message="Redis connection timeout",
        ...     cache_key="emission_factor:natural_gas:US",
        ...     operation="GET"
        ... )
    """

    def __init__(
        self,
        message: str,
        cache_key: Optional[str] = None,
        operation: Optional[str] = None,
        agent_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Initialize cache error.

        Args:
            message: Error message
            cache_key: Cache key involved in the failure
            operation: Operation that failed (GET, SET, DEL, EXPIRE)
            agent_name: Name of agent
            context: Error context
        """
        context = context or {}
        if cache_key:
            context["cache_key"] = cache_key
        if operation:
            context["operation"] = operation
        super().__init__(message, agent_name=agent_name, context=context)


class StorageError(InfrastructureException):
    """Object storage operation failed.

    Raised when S3 or compatible object storage read, write, or delete
    operations fail.

    Example:
        >>> raise StorageError(
        ...     message="Failed to upload report to S3",
        ...     bucket="greenlang-reports",
        ...     object_key="reports/2026/Q1/cbam-report.pdf",
        ...     operation="PUT"
        ... )
    """

    def __init__(
        self,
        message: str,
        bucket: Optional[str] = None,
        object_key: Optional[str] = None,
        operation: Optional[str] = None,
        agent_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Initialize storage error.

        Args:
            message: Error message
            bucket: Storage bucket name
            object_key: Object key or path
            operation: Operation that failed (GET, PUT, DELETE, LIST)
            agent_name: Name of agent
            context: Error context
        """
        context = context or {}
        if bucket:
            context["bucket"] = bucket
        if object_key:
            context["object_key"] = object_key
        if operation:
            context["operation"] = operation
        super().__init__(message, agent_name=agent_name, context=context)


class QueueError(InfrastructureException):
    """Message queue operation failed.

    Raised when message publishing, consuming, or acknowledgement fails.

    Example:
        >>> raise QueueError(
        ...     message="Failed to publish message to queue",
        ...     queue_name="emissions-processing",
        ...     operation="publish"
        ... )
    """

    def __init__(
        self,
        message: str,
        queue_name: Optional[str] = None,
        operation: Optional[str] = None,
        agent_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Initialize queue error.

        Args:
            message: Error message
            queue_name: Name of the queue
            operation: Operation that failed (publish, consume, ack, nack)
            agent_name: Name of agent
            context: Error context
        """
        context = context or {}
        if queue_name:
            context["queue_name"] = queue_name
        if operation:
            context["operation"] = operation
        super().__init__(message, agent_name=agent_name, context=context)


class CircuitBreakerOpenError(InfrastructureException):
    """Circuit breaker is in open state.

    Raised when a call is rejected because the circuit breaker has tripped
    due to too many consecutive failures.

    Example:
        >>> raise CircuitBreakerOpenError(
        ...     message="Circuit breaker open for SAP connector",
        ...     circuit_name="sap_erp",
        ...     failure_count=5,
        ...     reset_timeout_seconds=60
        ... )
    """

    def __init__(
        self,
        message: str,
        circuit_name: Optional[str] = None,
        failure_count: Optional[int] = None,
        reset_timeout_seconds: Optional[float] = None,
        agent_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Initialize circuit breaker open error.

        Args:
            message: Error message
            circuit_name: Name of the circuit breaker
            failure_count: Number of failures that triggered the open state
            reset_timeout_seconds: Seconds until circuit half-opens
            agent_name: Name of agent
            context: Error context
        """
        context = context or {}
        if circuit_name:
            context["circuit_name"] = circuit_name
        if failure_count is not None:
            context["failure_count"] = failure_count
        if reset_timeout_seconds is not None:
            context["reset_timeout_seconds"] = reset_timeout_seconds
        super().__init__(message, agent_name=agent_name, context=context)


class BulkheadFullError(InfrastructureException):
    """Bulkhead concurrency limit reached.

    Raised when a call is rejected because the bulkhead partition has
    reached its maximum concurrent execution limit.

    Example:
        >>> raise BulkheadFullError(
        ...     message="Bulkhead full for emission calculations",
        ...     bulkhead_name="emission_calc",
        ...     max_concurrent=10
        ... )
    """

    def __init__(
        self,
        message: str,
        bulkhead_name: Optional[str] = None,
        max_concurrent: Optional[int] = None,
        agent_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Initialize bulkhead full error.

        Args:
            message: Error message
            bulkhead_name: Name of the bulkhead partition
            max_concurrent: Maximum concurrent executions allowed
            agent_name: Name of agent
            context: Error context
        """
        context = context or {}
        if bulkhead_name:
            context["bulkhead_name"] = bulkhead_name
        if max_concurrent is not None:
            context["max_concurrent"] = max_concurrent
        super().__init__(message, agent_name=agent_name, context=context)


class RetryExhaustedError(InfrastructureException):
    """All retry attempts exhausted.

    Raised when an operation has failed all configured retry attempts.

    Example:
        >>> raise RetryExhaustedError(
        ...     message="Database connection failed after 3 retries",
        ...     max_retries=3,
        ...     last_error=original_exception
        ... )
    """

    def __init__(
        self,
        message: str,
        max_retries: Optional[int] = None,
        last_error: Optional[Exception] = None,
        agent_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Initialize retry exhausted error.

        Args:
            message: Error message
            max_retries: Maximum number of retries attempted
            last_error: The last exception encountered
            agent_name: Name of agent
            context: Error context
        """
        context = context or {}
        if max_retries is not None:
            context["max_retries"] = max_retries
        if last_error:
            context["last_error"] = str(last_error)
            context["last_error_type"] = type(last_error).__name__
        super().__init__(message, agent_name=agent_name, context=context)


__all__ = [
    'InfrastructureException',
    'DatabaseError',
    'CacheError',
    'StorageError',
    'QueueError',
    'CircuitBreakerOpenError',
    'BulkheadFullError',
    'RetryExhaustedError',
]
