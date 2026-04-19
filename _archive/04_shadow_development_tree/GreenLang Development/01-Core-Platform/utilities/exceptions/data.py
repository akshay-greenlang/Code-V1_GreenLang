"""
GreenLang Data Exceptions - Data Access and Validation Errors

This module provides exception classes for data-related errors.

Features:
- Schema validation errors
- Missing data errors
- Data corruption detection
- Data access failures

Author: GreenLang Team
Date: 2025-11-21
"""

from typing import Any, Dict, Optional

from greenlang.exceptions.base import GreenLangException


class DataException(GreenLangException):
    """Base exception for data-related errors.

    Raised when data access, validation, or processing fails.
    """
    ERROR_PREFIX = "GL_DATA"


class InvalidSchema(DataException):
    """Data schema is invalid.

    Raised when data does not conform to expected schema.

    Example:
        >>> raise InvalidSchema(
        ...     message="Input data does not match schema",
        ...     context={
        ...         "expected_schema": {"fuel_type": "string", "amount": "number"},
        ...         "actual_data": {"fuel_type": 123, "amount": "invalid"},
        ...         "errors": ["fuel_type: expected string, got int"]
        ...     }
        ... )
    """

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        expected_schema: Optional[Dict[str, Any]] = None,
        actual_data: Optional[Dict[str, Any]] = None,
        schema_errors: Optional[list] = None,
    ):
        """Initialize schema error.

        Args:
            message: Error message
            context: Error context
            expected_schema: Expected schema definition
            actual_data: Actual data that failed validation
            schema_errors: List of schema validation errors
        """
        context = context or {}
        if expected_schema:
            context["expected_schema"] = expected_schema
        if actual_data:
            context["actual_data"] = actual_data
        if schema_errors:
            context["schema_errors"] = schema_errors
        super().__init__(message, context=context)


class MissingData(DataException):
    """Required data is missing.

    Raised when required data fields or resources are not found.

    Example:
        >>> raise MissingData(
        ...     message="Required emission factor not found",
        ...     context={
        ...         "data_type": "emission_factor",
        ...         "query": {"fuel_type": "natural_gas", "country": "US"},
        ...         "available_factors": ["coal", "diesel"]
        ...     }
        ... )
    """

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        data_type: Optional[str] = None,
        missing_fields: Optional[list] = None,
    ):
        """Initialize missing data error.

        Args:
            message: Error message
            context: Error context
            data_type: Type of missing data
            missing_fields: List of missing field names
        """
        context = context or {}
        if data_type:
            context["data_type"] = data_type
        if missing_fields:
            context["missing_fields"] = missing_fields
        super().__init__(message, context=context)


class CorruptedData(DataException):
    """Data is corrupted or malformed.

    Raised when data integrity checks fail or data is malformed.

    Example:
        >>> raise CorruptedData(
        ...     message="Checksum verification failed",
        ...     context={
        ...         "file_path": "/data/emissions.json",
        ...         "expected_checksum": "abc123",
        ...         "actual_checksum": "def456"
        ...     }
        ... )
    """

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        data_source: Optional[str] = None,
        corruption_details: Optional[Dict[str, Any]] = None,
    ):
        """Initialize corrupted data error.

        Args:
            message: Error message
            context: Error context
            data_source: Source of corrupted data
            corruption_details: Details about the corruption
        """
        context = context or {}
        if data_source:
            context["data_source"] = data_source
        if corruption_details:
            context["corruption_details"] = corruption_details
        super().__init__(message, context=context)


class DataAccessError(DataException):
    """Data access failed.

    Raised when data cannot be accessed (permissions, network, etc.).

    Example:
        >>> raise DataAccessError(
        ...     message="Failed to access database",
        ...     context={
        ...         "database": "emissions_db",
        ...         "operation": "SELECT",
        ...         "error": "Connection timeout"
        ...     }
        ... )
    """

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        data_source: Optional[str] = None,
        operation: Optional[str] = None,
        cause: Optional[Exception] = None,
    ):
        """Initialize data access error.

        Args:
            message: Error message
            context: Error context
            data_source: Data source that failed
            operation: Operation that failed (read, write, delete)
            cause: Original exception
        """
        context = context or {}
        if data_source:
            context["data_source"] = data_source
        if operation:
            context["operation"] = operation
        if cause:
            context["cause"] = str(cause)
            context["cause_type"] = type(cause).__name__
        super().__init__(message, context=context)


__all__ = [
    'DataException',
    'InvalidSchema',
    'MissingData',
    'CorruptedData',
    'DataAccessError',
]
