# -*- coding: utf-8 -*-
"""
Constraint Validator for GL-FOUND-X-002.

This module implements constraint validation (ranges, patterns, enums)
against compiled schema IR. It provides comprehensive validation for:

- Numeric constraints: min/max, exclusive bounds, multipleOf
- String constraints: pattern (regex), minLength/maxLength, format
- Array constraints: minItems/maxItems, uniqueItems, contains
- Enum validation

All validations use the GLSCHEMA-E2xx error code range for constraint violations.

Example:
    >>> from greenlang.schema.validator.constraints import ConstraintValidator
    >>> validator = ConstraintValidator(ir, options)
    >>> findings = validator.validate_numeric(42, constraints, "/value")

Author: GreenLang Framework Team
Version: 0.1.0
GL-FOUND-X-002: Schema Compiler & Validator - Task 2.2
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import signal
import threading
from datetime import datetime
from ipaddress import IPv4Address, IPv6Address
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Tuple, Union
from urllib.parse import urlparse
from uuid import UUID

from ..compiler.ir import (
    ArrayConstraintIR,
    CompiledPattern,
    NumericConstraintIR,
    SchemaIR,
    StringConstraintIR,
)
from ..constants import MAX_FINDINGS, REGEX_TIMEOUT_MS
from ..models.config import ValidationOptions
from ..models.finding import Finding, FindingHint, Severity


logger = logging.getLogger(__name__)


# =============================================================================
# FORMAT VALIDATION FUNCTIONS
# =============================================================================


def _validate_email(value: str) -> bool:
    """
    Validate email address format.

    Uses RFC 5322 simplified pattern for practical validation.

    Args:
        value: String to validate as email

    Returns:
        True if valid email format, False otherwise
    """
    # RFC 5322 simplified pattern for practical email validation
    email_pattern = re.compile(
        r"^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@"
        r"[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?"
        r"(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$"
    )
    if not value or len(value) > 254:  # RFC 5321 length limit
        return False
    return bool(email_pattern.match(value))


def _validate_uri(value: str) -> bool:
    """
    Validate URI format per RFC 3986.

    Args:
        value: String to validate as URI

    Returns:
        True if valid URI format, False otherwise
    """
    try:
        result = urlparse(value)
        # Must have scheme and netloc (or path for file://)
        return bool(result.scheme) and bool(result.netloc or result.path)
    except Exception:
        return False


def _validate_date(value: str) -> bool:
    """
    Validate full-date format per RFC 3339 (YYYY-MM-DD).

    Args:
        value: String to validate as date

    Returns:
        True if valid date format, False otherwise
    """
    if not value or len(value) != 10:
        return False
    try:
        datetime.strptime(value, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def _validate_datetime(value: str) -> bool:
    """
    Validate date-time format per RFC 3339.

    Supports formats:
    - 2023-01-15T10:30:00Z
    - 2023-01-15T10:30:00+00:00
    - 2023-01-15T10:30:00.123Z

    Args:
        value: String to validate as datetime

    Returns:
        True if valid datetime format, False otherwise
    """
    if not value:
        return False

    # Common ISO 8601 / RFC 3339 patterns
    datetime_patterns = [
        r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$",  # 2023-01-15T10:30:00Z
        r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}$",  # With offset
        r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z$",  # With milliseconds
        r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+[+-]\d{2}:\d{2}$",  # Both
        r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$",  # Space separator
    ]

    for pattern in datetime_patterns:
        if re.match(pattern, value):
            # Also validate the actual date/time values
            try:
                # Extract date portion and validate
                date_part = value[:10]
                datetime.strptime(date_part, "%Y-%m-%d")
                return True
            except ValueError:
                return False

    return False


def _validate_time(value: str) -> bool:
    """
    Validate full-time format per RFC 3339 (HH:MM:SS).

    Args:
        value: String to validate as time

    Returns:
        True if valid time format, False otherwise
    """
    if not value:
        return False

    # Time with optional timezone
    time_patterns = [
        r"^\d{2}:\d{2}:\d{2}$",  # HH:MM:SS
        r"^\d{2}:\d{2}:\d{2}Z$",  # With Z
        r"^\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}$",  # With offset
        r"^\d{2}:\d{2}:\d{2}\.\d+$",  # With fractional seconds
        r"^\d{2}:\d{2}:\d{2}\.\d+Z$",  # With fractional and Z
    ]

    for pattern in time_patterns:
        if re.match(pattern, value):
            # Validate actual time values
            try:
                time_part = value[:8]
                datetime.strptime(time_part, "%H:%M:%S")
                return True
            except ValueError:
                return False

    return False


def _validate_uuid(value: str) -> bool:
    """
    Validate UUID format per RFC 4122.

    Args:
        value: String to validate as UUID

    Returns:
        True if valid UUID format, False otherwise
    """
    try:
        UUID(value)
        return True
    except (ValueError, AttributeError):
        return False


def _validate_ipv4(value: str) -> bool:
    """
    Validate IPv4 address format.

    Args:
        value: String to validate as IPv4 address

    Returns:
        True if valid IPv4 format, False otherwise
    """
    try:
        IPv4Address(value)
        return True
    except (ValueError, TypeError):
        return False


def _validate_ipv6(value: str) -> bool:
    """
    Validate IPv6 address format.

    Args:
        value: String to validate as IPv6 address

    Returns:
        True if valid IPv6 format, False otherwise
    """
    try:
        IPv6Address(value)
        return True
    except (ValueError, TypeError):
        return False


def _validate_hostname(value: str) -> bool:
    """
    Validate hostname format per RFC 1123.

    Args:
        value: String to validate as hostname

    Returns:
        True if valid hostname format, False otherwise
    """
    if not value or len(value) > 253:
        return False

    # Remove trailing dot for FQDN
    if value.endswith("."):
        value = value[:-1]

    # Each label must be 1-63 characters, alphanumeric or hyphen
    # Cannot start or end with hyphen
    hostname_pattern = re.compile(
        r"^(?!-)[A-Za-z0-9-]{1,63}(?<!-)(\.[A-Za-z0-9-]{1,63}(?<!-))*$"
    )
    return bool(hostname_pattern.match(value))


def _validate_regex(value: str) -> bool:
    """
    Validate that value is a valid regex pattern.

    Args:
        value: String to validate as regex pattern

    Returns:
        True if valid regex, False otherwise
    """
    try:
        re.compile(value)
        return True
    except re.error:
        return False


def _validate_json_pointer(value: str) -> bool:
    """
    Validate JSON Pointer format per RFC 6901.

    Args:
        value: String to validate as JSON Pointer

    Returns:
        True if valid JSON Pointer format, False otherwise
    """
    # Empty string is valid (root document)
    if value == "":
        return True

    # Must start with /
    if not value.startswith("/"):
        return False

    # Simple validation - more complex validation would check escaping
    json_pointer_pattern = re.compile(r"^(/[^/]*)*$")
    return bool(json_pointer_pattern.match(value))


def _validate_uri_reference(value: str) -> bool:
    """
    Validate URI-reference format per RFC 3986.

    URI-reference includes both URIs and relative-references.

    Args:
        value: String to validate as URI reference

    Returns:
        True if valid URI reference, False otherwise
    """
    try:
        # URI reference can be relative
        result = urlparse(value)
        # At minimum, should be parseable
        return bool(result.scheme or result.path or result.fragment)
    except Exception:
        return False


def _validate_iri(value: str) -> bool:
    """
    Validate IRI format per RFC 3987.

    IRIs are internationalized URIs that can contain Unicode characters.

    Args:
        value: String to validate as IRI

    Returns:
        True if valid IRI format, False otherwise
    """
    # For simplicity, we accept both ASCII and non-ASCII characters
    # Full IRI validation would require more comprehensive checking
    try:
        # Encode to URI and validate
        result = urlparse(value)
        return bool(result.scheme) and bool(result.netloc or result.path)
    except Exception:
        return False


# Registry of format validators
FORMAT_VALIDATORS: Dict[str, Callable[[str], bool]] = {
    "email": _validate_email,
    "uri": _validate_uri,
    "uri-reference": _validate_uri_reference,
    "iri": _validate_iri,
    "iri-reference": _validate_uri_reference,  # Same validation
    "date": _validate_date,
    "date-time": _validate_datetime,
    "time": _validate_time,
    "uuid": _validate_uuid,
    "ipv4": _validate_ipv4,
    "ipv6": _validate_ipv6,
    "hostname": _validate_hostname,
    "idn-hostname": _validate_hostname,  # Simplified
    "regex": _validate_regex,
    "json-pointer": _validate_json_pointer,
    "relative-json-pointer": _validate_json_pointer,  # Simplified
}


# =============================================================================
# CONSTRAINT VALIDATOR CLASS
# =============================================================================


class ConstraintValidator:
    """
    Validates values against schema constraints.

    This validator handles all constraint validation for the GreenLang Schema
    Validator, including numeric ranges, string patterns, array constraints,
    and enum validation.

    Attributes:
        ir: Compiled schema Intermediate Representation
        options: Validation options controlling behavior
        _findings: Internal list of accumulated findings

    Example:
        >>> from greenlang.schema.validator.constraints import ConstraintValidator
        >>> validator = ConstraintValidator(ir, options)
        >>> findings = validator.validate(payload, "")
        >>> for f in findings:
        ...     print(f"{f.code}: {f.message}")
    """

    def __init__(self, ir: SchemaIR, options: ValidationOptions):
        """
        Initialize the constraint validator.

        Args:
            ir: Compiled schema IR with constraint definitions
            options: Validation options controlling strictness and behavior
        """
        self.ir = ir
        self.options = options
        self._findings: List[Finding] = []

    def validate(
        self,
        payload: Dict[str, Any],
        path: str = ""
    ) -> List[Finding]:
        """
        Validate all constraints in payload.

        Recursively traverses the payload and validates each value against
        its corresponding constraints from the schema IR.

        Args:
            payload: The payload data to validate
            path: JSON Pointer path prefix (default: root)

        Returns:
            List of validation findings (errors and warnings)

        Example:
            >>> findings = validator.validate({"value": 150}, "")
            >>> assert len(findings) == 0  # If valid
        """
        self._findings = []
        self._validate_recursive(payload, path)
        return self._findings

    def _validate_recursive(
        self,
        value: Any,
        path: str
    ) -> None:
        """
        Recursively validate constraints on a value.

        Dispatches to appropriate validation methods based on value type
        and schema constraints.

        Args:
            value: The value to validate
            path: JSON Pointer path to this value
        """
        # Check if we've hit the maximum findings limit
        if len(self._findings) >= self.options.max_errors > 0:
            return

        # Skip None values (handled by structural validator)
        if value is None:
            return

        # Check for enum constraint at this path
        enum_values = self.ir.get_enum(path)
        if enum_values is not None:
            findings = self.validate_enum(value, enum_values, path)
            self._findings.extend(findings)

        # Dispatch based on value type
        if isinstance(value, dict):
            # Validate nested object properties
            for key, nested_value in value.items():
                nested_path = f"{path}/{key}" if path else f"/{key}"
                self._validate_recursive(nested_value, nested_path)

        elif isinstance(value, list):
            # Validate array constraints
            array_constraint = self.ir.get_array_constraint(path)
            if array_constraint is not None:
                findings = self.validate_array(value, array_constraint, path)
                self._findings.extend(findings)

            # Validate array items recursively
            for index, item in enumerate(value):
                item_path = f"{path}/{index}"
                self._validate_recursive(item, item_path)

        elif isinstance(value, str):
            # Validate string constraints
            string_constraint = self.ir.get_string_constraint(path)
            if string_constraint is not None:
                findings = self.validate_string(value, string_constraint, path)
                self._findings.extend(findings)

        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            # Validate numeric constraints
            numeric_constraint = self.ir.get_numeric_constraint(path)
            if numeric_constraint is not None:
                findings = self.validate_numeric(value, numeric_constraint, path)
                self._findings.extend(findings)

    def validate_numeric(
        self,
        value: Union[int, float],
        constraints: NumericConstraintIR,
        path: str
    ) -> List[Finding]:
        """
        Validate numeric value against constraints.

        Checks:
        - minimum/maximum (inclusive bounds)
        - exclusiveMinimum/exclusiveMaximum
        - multipleOf

        Args:
            value: The numeric value to validate
            constraints: Numeric constraint IR with bounds
            path: JSON Pointer path for error reporting

        Returns:
            List of validation findings

        Example:
            >>> constraints = NumericConstraintIR(path="/temp", minimum=0, maximum=100)
            >>> findings = validator.validate_numeric(150, constraints, "/temp")
            >>> assert len(findings) == 1  # Value exceeds maximum
        """
        findings: List[Finding] = []

        # Check inclusive minimum
        if constraints.minimum is not None:
            if value < constraints.minimum:
                findings.append(self._create_range_finding(
                    path=path,
                    message=f"Value {value} is less than minimum {constraints.minimum}",
                    expected={"minimum": constraints.minimum},
                    actual=value,
                    constraint_type="minimum"
                ))

        # Check inclusive maximum
        if constraints.maximum is not None:
            if value > constraints.maximum:
                findings.append(self._create_range_finding(
                    path=path,
                    message=f"Value {value} exceeds maximum {constraints.maximum}",
                    expected={"maximum": constraints.maximum},
                    actual=value,
                    constraint_type="maximum"
                ))

        # Check exclusive minimum
        if constraints.exclusive_minimum is not None:
            if value <= constraints.exclusive_minimum:
                findings.append(self._create_range_finding(
                    path=path,
                    message=f"Value {value} must be greater than {constraints.exclusive_minimum}",
                    expected={"exclusiveMinimum": constraints.exclusive_minimum},
                    actual=value,
                    constraint_type="exclusiveMinimum"
                ))

        # Check exclusive maximum
        if constraints.exclusive_maximum is not None:
            if value >= constraints.exclusive_maximum:
                findings.append(self._create_range_finding(
                    path=path,
                    message=f"Value {value} must be less than {constraints.exclusive_maximum}",
                    expected={"exclusiveMaximum": constraints.exclusive_maximum},
                    actual=value,
                    constraint_type="exclusiveMaximum"
                ))

        # Check multipleOf
        if constraints.multiple_of is not None:
            if not self._is_multiple_of(value, constraints.multiple_of):
                findings.append(Finding(
                    code="GLSCHEMA-E205",
                    severity=Severity.ERROR,
                    path=path,
                    message=f"Value {value} is not a multiple of {constraints.multiple_of}",
                    expected={"multipleOf": constraints.multiple_of},
                    actual=value,
                    hint=FindingHint(
                        category="multiple_of_violation",
                        suggested_values=[
                            constraints.multiple_of * round(value / constraints.multiple_of)
                        ]
                    )
                ))

        return findings

    def _is_multiple_of(self, value: Union[int, float], divisor: float) -> bool:
        """
        Check if value is a multiple of divisor with floating-point tolerance.

        Uses a relative tolerance for floating-point comparison to handle
        precision issues.

        Args:
            value: The value to check
            divisor: The divisor to check against

        Returns:
            True if value is a multiple of divisor
        """
        if divisor == 0:
            return False

        # For integers, use exact division
        if isinstance(value, int) and isinstance(divisor, int):
            return value % divisor == 0

        # For floats, use tolerance-based comparison
        remainder = abs(value % divisor)
        # Check if remainder is effectively 0 or effectively equal to divisor
        tolerance = 1e-10 * max(abs(value), abs(divisor), 1)
        return remainder < tolerance or abs(remainder - abs(divisor)) < tolerance

    def _create_range_finding(
        self,
        path: str,
        message: str,
        expected: Dict[str, Any],
        actual: Any,
        constraint_type: str
    ) -> Finding:
        """
        Create a range violation finding.

        Args:
            path: JSON Pointer path
            message: Error message
            expected: Expected constraint value
            actual: Actual value
            constraint_type: Type of constraint violated

        Returns:
            Finding for the range violation
        """
        return Finding(
            code="GLSCHEMA-E200",
            severity=Severity.ERROR,
            path=path,
            message=message,
            expected=expected,
            actual=actual,
            hint=FindingHint(
                category="range_violation",
                suggested_values=[],
                docs_url=None
            )
        )

    def validate_string(
        self,
        value: str,
        constraints: StringConstraintIR,
        path: str
    ) -> List[Finding]:
        """
        Validate string value against constraints.

        Checks:
        - minLength/maxLength
        - pattern (regex with timeout protection)
        - format (email, uri, date, etc.)

        Args:
            value: The string value to validate
            constraints: String constraint IR
            path: JSON Pointer path for error reporting

        Returns:
            List of validation findings

        Example:
            >>> constraints = StringConstraintIR(path="/email", format="email")
            >>> findings = validator.validate_string("invalid", constraints, "/email")
            >>> assert len(findings) == 1  # Invalid email format
        """
        findings: List[Finding] = []

        # Check minimum length
        if constraints.min_length is not None:
            if len(value) < constraints.min_length:
                findings.append(Finding(
                    code="GLSCHEMA-E203",
                    severity=Severity.ERROR,
                    path=path,
                    message=f"String length {len(value)} is less than minimum {constraints.min_length}",
                    expected={"minLength": constraints.min_length},
                    actual=len(value),
                    hint=FindingHint(
                        category="length_violation",
                        suggested_values=[]
                    )
                ))

        # Check maximum length
        if constraints.max_length is not None:
            if len(value) > constraints.max_length:
                findings.append(Finding(
                    code="GLSCHEMA-E203",
                    severity=Severity.ERROR,
                    path=path,
                    message=f"String length {len(value)} exceeds maximum {constraints.max_length}",
                    expected={"maxLength": constraints.max_length},
                    actual=len(value),
                    hint=FindingHint(
                        category="length_violation",
                        suggested_values=[value[:constraints.max_length]] if constraints.max_length > 0 else []
                    )
                ))

        # Check pattern (regex)
        if constraints.pattern is not None:
            pattern_matched = self._match_pattern_with_timeout(
                constraints.pattern_compiled if constraints.pattern_compiled else CompiledPattern(
                    pattern=constraints.pattern,
                    is_safe=True
                ),
                value
            )

            if pattern_matched is False:
                findings.append(Finding(
                    code="GLSCHEMA-E201",
                    severity=Severity.ERROR,
                    path=path,
                    message=f"Value '{self._truncate_value(value)}' does not match pattern '{constraints.pattern}'",
                    expected={"pattern": constraints.pattern},
                    actual=value,
                    hint=FindingHint(
                        category="pattern_mismatch",
                        suggested_values=[]
                    )
                ))
            elif pattern_matched is None:
                # Timeout occurred during regex matching
                logger.warning(f"Regex pattern match timed out at path {path}")
                findings.append(Finding(
                    code="GLSCHEMA-E809",
                    severity=Severity.ERROR,
                    path=path,
                    message=f"Pattern matching timed out for pattern '{constraints.pattern}'",
                    expected={"pattern": constraints.pattern},
                    actual=None,
                    hint=FindingHint(
                        category="timeout",
                        suggested_values=[]
                    )
                ))

        # Check format
        if constraints.format is not None:
            format_finding = self._validate_format(value, constraints.format, path)
            if format_finding is not None:
                findings.append(format_finding)

        return findings

    def _truncate_value(self, value: str, max_length: int = 50) -> str:
        """
        Truncate a value for display in error messages.

        Args:
            value: The value to truncate
            max_length: Maximum length before truncation

        Returns:
            Truncated value with ellipsis if needed
        """
        if len(value) <= max_length:
            return value
        return value[:max_length - 3] + "..."

    def validate_array(
        self,
        value: List,
        constraints: ArrayConstraintIR,
        path: str
    ) -> List[Finding]:
        """
        Validate array against constraints.

        Checks:
        - minItems/maxItems
        - uniqueItems

        Args:
            value: The array value to validate
            constraints: Array constraint IR
            path: JSON Pointer path for error reporting

        Returns:
            List of validation findings

        Example:
            >>> constraints = ArrayConstraintIR(path="/items", min_items=1)
            >>> findings = validator.validate_array([], constraints, "/items")
            >>> assert len(findings) == 1  # Array is empty
        """
        findings: List[Finding] = []

        # Check minimum items
        if constraints.min_items is not None:
            if len(value) < constraints.min_items:
                findings.append(Finding(
                    code="GLSCHEMA-E203",
                    severity=Severity.ERROR,
                    path=path,
                    message=f"Array has {len(value)} items, minimum required is {constraints.min_items}",
                    expected={"minItems": constraints.min_items},
                    actual=len(value),
                    hint=FindingHint(
                        category="length_violation",
                        suggested_values=[]
                    )
                ))

        # Check maximum items
        if constraints.max_items is not None:
            if len(value) > constraints.max_items:
                findings.append(Finding(
                    code="GLSCHEMA-E203",
                    severity=Severity.ERROR,
                    path=path,
                    message=f"Array has {len(value)} items, maximum allowed is {constraints.max_items}",
                    expected={"maxItems": constraints.max_items},
                    actual=len(value),
                    hint=FindingHint(
                        category="length_violation",
                        suggested_values=[]
                    )
                ))

        # Check unique items
        if constraints.unique_items:
            duplicate_pairs = self._check_unique_items(value)
            if duplicate_pairs:
                # Get indices of all duplicates
                indices = set()
                for i, j in duplicate_pairs:
                    indices.add(i)
                    indices.add(j)

                findings.append(Finding(
                    code="GLSCHEMA-E204",
                    severity=Severity.ERROR,
                    path=path,
                    message=f"Array contains duplicate items at indices {sorted(indices)}",
                    expected={"uniqueItems": True},
                    actual={"duplicate_indices": sorted(indices)},
                    hint=FindingHint(
                        category="unique_violation",
                        suggested_values=[]
                    )
                ))

        return findings

    def validate_enum(
        self,
        value: Any,
        allowed_values: List[Any],
        path: str
    ) -> List[Finding]:
        """
        Validate value is in enum.

        Args:
            value: The value to validate
            allowed_values: List of allowed values
            path: JSON Pointer path for error reporting

        Returns:
            List of findings if value not in enum

        Example:
            >>> allowed = ["low", "medium", "high"]
            >>> findings = validator.validate_enum("critical", allowed, "/severity")
            >>> assert len(findings) == 1  # Value not in enum
        """
        findings: List[Finding] = []

        if not self._value_in_enum(value, allowed_values):
            # Format allowed values for display
            displayed_values = self._format_enum_values(allowed_values)

            findings.append(Finding(
                code="GLSCHEMA-E202",
                severity=Severity.ERROR,
                path=path,
                message=f"Value '{self._format_value(value)}' is not one of allowed values: {displayed_values}",
                expected={"enum": allowed_values},
                actual=value,
                hint=FindingHint(
                    category="enum_violation",
                    suggested_values=allowed_values[:10]  # Limit suggestions
                )
            ))

        return findings

    def _value_in_enum(self, value: Any, allowed_values: List[Any]) -> bool:
        """
        Check if value is in the allowed enum values.

        Uses deep equality comparison for objects.

        Args:
            value: The value to check
            allowed_values: List of allowed values

        Returns:
            True if value is in allowed values
        """
        for allowed in allowed_values:
            if self._values_equal(value, allowed):
                return True
        return False

    def _values_equal(self, a: Any, b: Any) -> bool:
        """
        Check if two values are equal using JSON semantics.

        Args:
            a: First value
            b: Second value

        Returns:
            True if values are equal
        """
        # Handle type differences
        if type(a) != type(b):
            # Allow int/float comparison
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                return float(a) == float(b)
            return False

        # For dictionaries and lists, use JSON serialization for comparison
        if isinstance(a, (dict, list)):
            try:
                return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
            except (TypeError, ValueError):
                return a == b

        return a == b

    def _format_enum_values(self, values: List[Any], max_display: int = 5) -> str:
        """
        Format enum values for display in error messages.

        Args:
            values: List of enum values
            max_display: Maximum number of values to display

        Returns:
            Formatted string of enum values
        """
        if len(values) <= max_display:
            return ", ".join(repr(v) for v in values)

        displayed = ", ".join(repr(v) for v in values[:max_display])
        return f"{displayed}, ... ({len(values) - max_display} more)"

    def _format_value(self, value: Any, max_length: int = 50) -> str:
        """
        Format a value for display in error messages.

        Args:
            value: The value to format
            max_length: Maximum length for string representation

        Returns:
            Formatted string representation
        """
        if isinstance(value, str):
            if len(value) > max_length:
                return value[:max_length - 3] + "..."
            return value
        return repr(value)

    def _match_pattern_with_timeout(
        self,
        pattern: CompiledPattern,
        value: str
    ) -> Optional[bool]:
        """
        Match regex with timeout protection.

        Protects against ReDoS attacks by implementing a timeout on regex
        matching operations.

        Args:
            pattern: CompiledPattern with pattern string and safety metadata
            value: String to match against pattern

        Returns:
            True if matches, False if not, None if timeout occurred
        """
        # If pattern is marked as unsafe, return None to skip
        if not pattern.is_safe:
            logger.warning(f"Skipping unsafe pattern: {pattern.pattern}")
            return None

        # Get the compiled regex
        compiled = pattern.get_compiled()
        if compiled is None:
            logger.warning(f"Failed to compile pattern: {pattern.pattern}")
            return None

        # Use timeout for matching
        timeout_seconds = pattern.timeout_ms / 1000.0

        result: List[Optional[bool]] = [None]
        exception_holder: List[Optional[Exception]] = [None]

        def match_with_timeout():
            try:
                if compiled.search(value):
                    result[0] = True
                else:
                    result[0] = False
            except Exception as e:
                exception_holder[0] = e

        # Create thread for matching
        thread = threading.Thread(target=match_with_timeout)
        thread.daemon = True
        thread.start()
        thread.join(timeout=timeout_seconds)

        if thread.is_alive():
            # Timeout occurred
            logger.warning(
                f"Regex matching timed out after {timeout_seconds}s "
                f"for pattern '{pattern.pattern[:50]}...'"
            )
            return None

        if exception_holder[0] is not None:
            logger.warning(
                f"Regex matching failed: {exception_holder[0]}"
            )
            return None

        return result[0]

    def _validate_format(
        self,
        value: str,
        format_name: str,
        path: str
    ) -> Optional[Finding]:
        """
        Validate string format (email, uri, date, etc.).

        Args:
            value: The string to validate
            format_name: The format name to validate against
            path: JSON Pointer path for error reporting

        Returns:
            Finding if format validation fails, None otherwise
        """
        validator = FORMAT_VALIDATORS.get(format_name)

        if validator is None:
            # Unknown format - log warning but don't fail
            logger.debug(f"Unknown format '{format_name}' at path {path}")
            return None

        try:
            if not validator(value):
                return Finding(
                    code="GLSCHEMA-E206",
                    severity=Severity.ERROR,
                    path=path,
                    message=f"Value '{self._truncate_value(value)}' does not match format '{format_name}'",
                    expected={"format": format_name},
                    actual=value,
                    hint=FindingHint(
                        category="format_violation",
                        suggested_values=[]
                    )
                )
        except Exception as e:
            logger.warning(f"Format validation failed for '{format_name}': {e}")
            return Finding(
                code="GLSCHEMA-E206",
                severity=Severity.ERROR,
                path=path,
                message=f"Format validation failed for '{format_name}': {str(e)}",
                expected={"format": format_name},
                actual=value,
                hint=FindingHint(
                    category="format_violation",
                    suggested_values=[]
                )
            )

        return None

    def _add_finding(
        self,
        code: str,
        severity: Severity,
        path: str,
        message: str,
        expected: Any = None,
        actual: Any = None
    ) -> None:
        """
        Add a validation finding.

        Helper method to create and add findings to the internal list.

        Args:
            code: Error code (GLSCHEMA-*)
            severity: Severity level
            path: JSON Pointer path
            message: Error message
            expected: Expected value (optional)
            actual: Actual value (optional)
        """
        # Check findings limit
        if len(self._findings) >= self.options.max_errors > 0:
            return

        finding = Finding(
            code=code,
            severity=severity,
            path=path,
            message=message,
            expected={"expected": expected} if expected is not None else None,
            actual=actual
        )
        self._findings.append(finding)

    def _check_unique_items(
        self,
        array: List[Any]
    ) -> List[Tuple[int, int]]:
        """
        Check for duplicate items in array.

        Uses hash-based comparison with JSON serialization for complex objects.

        Args:
            array: Array to check for duplicates

        Returns:
            List of (index1, index2) tuples for duplicate pairs
        """
        duplicates: List[Tuple[int, int]] = []
        seen: Dict[str, int] = {}

        for i, item in enumerate(array):
            # Create a hashable representation
            try:
                if isinstance(item, (dict, list)):
                    key = json.dumps(item, sort_keys=True)
                elif isinstance(item, (int, float, str, bool)) or item is None:
                    key = json.dumps(item)
                else:
                    # For other types, use repr
                    key = repr(item)
            except (TypeError, ValueError):
                # If item is not JSON serializable, use repr
                key = repr(item)

            if key in seen:
                duplicates.append((seen[key], i))
            else:
                seen[key] = i

        return duplicates


# =============================================================================
# MODULE EXPORTS
# =============================================================================


__all__ = [
    # Main validator class
    "ConstraintValidator",
    # Format validators
    "FORMAT_VALIDATORS",
    # Individual format validators (for testing and extension)
    "_validate_email",
    "_validate_uri",
    "_validate_date",
    "_validate_datetime",
    "_validate_time",
    "_validate_uuid",
    "_validate_ipv4",
    "_validate_ipv6",
    "_validate_hostname",
    "_validate_regex",
    "_validate_json_pointer",
    "_validate_uri_reference",
    "_validate_iri",
]
