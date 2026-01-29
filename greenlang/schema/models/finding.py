# -*- coding: utf-8 -*-
"""
Validation Finding Models
=========================

Pydantic models for validation findings (errors, warnings, info).

A Finding represents a single issue found during validation, with:
- Error code (GLSCHEMA-* prefix)
- Severity level (error/warning/info)
- JSON Pointer path to the problematic location
- Human-readable message
- Expected vs actual values for debugging
- Optional hints for resolution

Example:
    >>> finding = Finding(
    ...     code="GLSCHEMA-E100",
    ...     severity=Severity.ERROR,
    ...     path="/energy_consumption",
    ...     message="Missing required field"
    ... )
    >>> print(finding.is_error())
    True

Author: GreenLang Framework Team
Version: 0.1.0
GL-FOUND-X-002: Schema Compiler & Validator
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# JSON Pointer pattern (RFC 6901)
JSON_POINTER_PATTERN = re.compile(r"^(/[^/]*)*$")

# GreenLang error code pattern
ERROR_CODE_PATTERN = re.compile(r"^GLSCHEMA-[EWI]\d{3}$")


class Severity(str, Enum):
    """
    Severity level for validation findings.

    Defines how severe a finding is:
    - ERROR: Validation failure, payload is invalid
    - WARNING: Potential issue, but payload may still be usable
    - INFO: Informational note, no action required

    Example:
        >>> severity = Severity.ERROR
        >>> print(severity.value)
        error
    """

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

    def is_error(self) -> bool:
        """Check if this is an error severity."""
        return self == Severity.ERROR

    def is_warning(self) -> bool:
        """Check if this is a warning severity."""
        return self == Severity.WARNING

    def is_info(self) -> bool:
        """Check if this is an info severity."""
        return self == Severity.INFO

    def numeric_level(self) -> int:
        """
        Get numeric severity level (higher = more severe).

        Returns:
            Integer severity level: ERROR=3, WARNING=2, INFO=1
        """
        levels = {
            Severity.ERROR: 3,
            Severity.WARNING: 2,
            Severity.INFO: 1,
        }
        return levels[self]


class FindingHint(BaseModel):
    """
    Hint for resolving a validation finding.

    Provides additional context to help users understand and fix
    validation issues.

    Attributes:
        category: Category of the hint (e.g., "type_mismatch", "missing_field").
        suggested_values: List of valid values or suggestions.
        docs_url: URL to relevant documentation.

    Example:
        >>> hint = FindingHint(
        ...     category="enum_violation",
        ...     suggested_values=["solar", "wind", "hydro"],
        ...     docs_url="https://docs.greenlang.dev/schemas/energy"
        ... )
    """

    category: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Category of the hint"
    )

    suggested_values: List[Any] = Field(
        default_factory=list,
        max_length=20,
        description="List of suggested/valid values"
    )

    docs_url: Optional[str] = Field(
        default=None,
        max_length=2048,
        description="URL to relevant documentation"
    )

    model_config = {
        "frozen": False,
        "extra": "forbid",
    }

    @field_validator("docs_url")
    @classmethod
    def validate_docs_url(cls, v: Optional[str]) -> Optional[str]:
        """
        Validate docs_url is a valid URL format.

        Args:
            v: The URL string (may be None).

        Returns:
            The validated URL string.

        Raises:
            ValueError: If URL format is invalid.
        """
        if v is None:
            return v

        if not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError(
                f"Invalid docs_url '{v}'. Must start with http:// or https://"
            )

        return v


class Finding(BaseModel):
    """
    A single validation finding (error, warning, or info).

    Represents an issue found during schema validation. Each finding
    includes all information needed to understand and resolve the issue.

    Attributes:
        code: GreenLang error code (e.g., "GLSCHEMA-E100").
        severity: Severity level (error/warning/info).
        path: JSON Pointer (RFC 6901) to the problematic location.
        message: Human-readable description of the issue.
        expected: What the schema expected (for debugging).
        actual: What was actually found (for debugging).
        hint: Optional hint for resolution.

    Example:
        >>> finding = Finding(
        ...     code="GLSCHEMA-E100",
        ...     severity=Severity.ERROR,
        ...     path="/energy_consumption",
        ...     message="Missing required field",
        ...     expected={"type": "object", "required": True},
        ...     hint=FindingHint(
        ...         category="missing_required",
        ...         docs_url="https://docs.greenlang.dev/errors/E100"
        ...     )
        ... )
        >>> print(finding.is_error())
        True
    """

    code: str = Field(
        ...,
        min_length=1,
        max_length=32,
        description="GreenLang error code (e.g., 'GLSCHEMA-E100')"
    )

    severity: Severity = Field(
        ...,
        description="Severity level"
    )

    path: str = Field(
        ...,
        max_length=4096,
        description="JSON Pointer (RFC 6901) to the issue location"
    )

    message: str = Field(
        ...,
        min_length=1,
        max_length=2048,
        description="Human-readable description of the issue"
    )

    expected: Optional[Dict[str, Any]] = Field(
        default=None,
        description="What the schema expected"
    )

    actual: Optional[Any] = Field(
        default=None,
        description="What was actually found"
    )

    hint: Optional[FindingHint] = Field(
        default=None,
        description="Optional hint for resolution"
    )

    model_config = {
        "frozen": False,
        "extra": "forbid",
        "json_schema_extra": {
            "examples": [
                {
                    "code": "GLSCHEMA-E100",
                    "severity": "error",
                    "path": "/energy_consumption",
                    "message": "Missing required field",
                    "expected": {"type": "object", "required": True}
                },
                {
                    "code": "GLSCHEMA-E301",
                    "severity": "error",
                    "path": "/fuel_type",
                    "message": "Unit 'kg' incompatible with dimension 'volume'",
                    "expected": {"dimension": "volume", "units": ["L", "gallon", "m3"]},
                    "actual": {"dimension": "mass", "unit": "kg"},
                    "hint": {
                        "category": "unit_mismatch",
                        "suggested_values": ["L", "gallon", "m3"],
                        "docs_url": "https://docs.greenlang.dev/units"
                    }
                },
                {
                    "code": "GLSCHEMA-W700",
                    "severity": "warning",
                    "path": "/emmisions",
                    "message": "Unknown field. Did you mean 'emissions'?",
                    "hint": {
                        "category": "typo",
                        "suggested_values": ["emissions"]
                    }
                }
            ]
        }
    }

    @field_validator("code")
    @classmethod
    def validate_code(cls, v: str) -> str:
        """
        Validate error code format.

        Args:
            v: The error code string.

        Returns:
            The validated error code.

        Raises:
            ValueError: If code doesn't match GLSCHEMA-* pattern.
        """
        if not ERROR_CODE_PATTERN.match(v):
            raise ValueError(
                f"Invalid error code '{v}'. Must match pattern GLSCHEMA-[EWI]XXX "
                "(e.g., 'GLSCHEMA-E100', 'GLSCHEMA-W700')."
            )
        return v

    @field_validator("path")
    @classmethod
    def validate_path(cls, v: str) -> str:
        """
        Validate JSON Pointer format (RFC 6901).

        Args:
            v: The JSON Pointer path.

        Returns:
            The validated path.

        Raises:
            ValueError: If path is not a valid JSON Pointer.
        """
        # Empty string is valid (root)
        if v == "":
            return v

        if not JSON_POINTER_PATTERN.match(v):
            raise ValueError(
                f"Invalid JSON Pointer path '{v}'. Must follow RFC 6901 format "
                "(e.g., '/field', '/parent/child', '/array/0')."
            )
        return v

    def is_error(self) -> bool:
        """
        Check if this finding is an error.

        Returns:
            True if severity is ERROR.
        """
        return self.severity == Severity.ERROR

    def is_warning(self) -> bool:
        """
        Check if this finding is a warning.

        Returns:
            True if severity is WARNING.
        """
        return self.severity == Severity.WARNING

    def is_info(self) -> bool:
        """
        Check if this finding is informational.

        Returns:
            True if severity is INFO.
        """
        return self.severity == Severity.INFO

    def error_category(self) -> str:
        """
        Extract the error category from the code.

        Returns:
            Error category (e.g., "E1" for structural, "E3" for unit errors).

        Example:
            >>> finding = Finding(code="GLSCHEMA-E301", ...)
            >>> finding.error_category()
            'E3'
        """
        # Extract the letter and first digit (e.g., "E3" from "E301")
        match = re.match(r"GLSCHEMA-([EWI])(\d)", self.code)
        if match:
            return f"{match.group(1)}{match.group(2)}"
        return "UNKNOWN"

    def is_structural_error(self) -> bool:
        """Check if this is a structural error (E1xx)."""
        return self.code.startswith("GLSCHEMA-E1")

    def is_constraint_error(self) -> bool:
        """Check if this is a constraint error (E2xx)."""
        return self.code.startswith("GLSCHEMA-E2")

    def is_unit_error(self) -> bool:
        """Check if this is a unit error (E3xx)."""
        return self.code.startswith("GLSCHEMA-E3")

    def is_rule_error(self) -> bool:
        """Check if this is a rule error (E4xx)."""
        return self.code.startswith("GLSCHEMA-E4")

    def is_schema_error(self) -> bool:
        """Check if this is a schema error (E5xx)."""
        return self.code.startswith("GLSCHEMA-E5")

    def is_deprecation_warning(self) -> bool:
        """Check if this is a deprecation warning (W6xx)."""
        return self.code.startswith("GLSCHEMA-W6")

    def is_lint_warning(self) -> bool:
        """Check if this is a lint warning (W7xx)."""
        return self.code.startswith("GLSCHEMA-W7")

    def format_short(self) -> str:
        """
        Format finding as a short one-line message.

        Returns:
            Short formatted string.

        Example:
            >>> finding.format_short()
            'ERROR GLSCHEMA-E100 at /energy: Missing required field'
        """
        severity_str = self.severity.value.upper()
        path_str = self.path if self.path else "(root)"
        return f"{severity_str} {self.code} at {path_str}: {self.message}"

    def format_detailed(self) -> str:
        """
        Format finding with full details.

        Returns:
            Multi-line detailed formatted string.
        """
        lines = [self.format_short()]

        if self.expected:
            lines.append(f"  Expected: {self.expected}")

        if self.actual is not None:
            lines.append(f"  Actual: {self.actual}")

        if self.hint:
            if self.hint.suggested_values:
                suggestions = ", ".join(str(v) for v in self.hint.suggested_values[:5])
                lines.append(f"  Suggestions: {suggestions}")
            if self.hint.docs_url:
                lines.append(f"  Docs: {self.hint.docs_url}")

        return "\n".join(lines)

    def __str__(self) -> str:
        """Return string representation."""
        return self.format_short()

    def __repr__(self) -> str:
        """Return detailed representation."""
        return f"Finding(code='{self.code}', severity={self.severity}, path='{self.path}')"
