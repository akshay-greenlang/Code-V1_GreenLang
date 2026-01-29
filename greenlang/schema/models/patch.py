# -*- coding: utf-8 -*-
"""
JSON Patch and Fix Suggestion Models
====================================

Pydantic models for JSON Patch operations (RFC 6902) and fix suggestions.

This module provides:
- JSONPatchOp: A single JSON Patch operation (add/remove/replace/move/copy/test)
- PatchSafety: Enum for patch safety classification
- FixSuggestion: A complete fix suggestion with preconditions and safety

Example:
    >>> patch_op = JSONPatchOp(
    ...     op="replace",
    ...     path="/energy/value",
    ...     value=100
    ... )
    >>> suggestion = FixSuggestion(
    ...     patch=[patch_op],
    ...     confidence=0.95,
    ...     safety=PatchSafety.SAFE,
    ...     rationale="Converting string to number"
    ... )

Author: GreenLang Framework Team
Version: 0.1.0
GL-FOUND-X-002: Schema Compiler & Validator
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# JSON Pointer pattern (RFC 6901)
JSON_POINTER_PATTERN = re.compile(r"^(/[^/]*)*$")


class JSONPatchOp(BaseModel):
    """
    A single JSON Patch operation (RFC 6902).

    Represents one operation in a JSON Patch document that can
    be used to modify a JSON document.

    Attributes:
        op: The operation type (add/remove/replace/move/copy/test).
        path: JSON Pointer (RFC 6901) to the target location.
        value: The value for add/replace/test operations.
        from_: Source path for move/copy operations (aliased as "from" in JSON).

    Example:
        >>> # Add operation
        >>> JSONPatchOp(op="add", path="/energy", value={"value": 100, "unit": "kWh"})

        >>> # Replace operation
        >>> JSONPatchOp(op="replace", path="/energy/value", value=100)

        >>> # Move operation
        >>> JSONPatchOp(op="move", from_="/old_field", path="/new_field")

        >>> # Test operation (precondition)
        >>> JSONPatchOp(op="test", path="/version", value="1.0")
    """

    op: Literal["add", "remove", "replace", "move", "copy", "test"] = Field(
        ...,
        description="The operation type"
    )

    path: str = Field(
        ...,
        max_length=4096,
        description="JSON Pointer (RFC 6901) to target location"
    )

    value: Optional[Any] = Field(
        default=None,
        description="Value for add/replace/test operations"
    )

    from_: Optional[str] = Field(
        default=None,
        alias="from",
        max_length=4096,
        description="Source path for move/copy operations"
    )

    model_config = {
        "frozen": False,
        "extra": "forbid",
        "populate_by_name": True,
        "json_schema_extra": {
            "examples": [
                {
                    "op": "add",
                    "path": "/energy",
                    "value": {"value": 100, "unit": "kWh"}
                },
                {
                    "op": "replace",
                    "path": "/energy/value",
                    "value": 100
                },
                {
                    "op": "remove",
                    "path": "/deprecated_field"
                },
                {
                    "op": "move",
                    "from": "/old_name",
                    "path": "/new_name"
                },
                {
                    "op": "test",
                    "path": "/version",
                    "value": "1.0"
                }
            ]
        }
    }

    @field_validator("path")
    @classmethod
    def validate_path(cls, v: str) -> str:
        """
        Validate path is a valid JSON Pointer.

        Args:
            v: The path string.

        Returns:
            The validated path.

        Raises:
            ValueError: If path is not a valid JSON Pointer.
        """
        if v == "":
            return v  # Empty string is valid (root)

        if not JSON_POINTER_PATTERN.match(v):
            raise ValueError(
                f"Invalid JSON Pointer path '{v}'. Must follow RFC 6901 format "
                "(e.g., '/field', '/parent/child', '/array/0')."
            )
        return v

    @field_validator("from_")
    @classmethod
    def validate_from(cls, v: Optional[str]) -> Optional[str]:
        """
        Validate from_ is a valid JSON Pointer if provided.

        Args:
            v: The from path string (may be None).

        Returns:
            The validated path or None.

        Raises:
            ValueError: If path is not a valid JSON Pointer.
        """
        if v is None:
            return v

        if v == "":
            return v  # Empty string is valid (root)

        if not JSON_POINTER_PATTERN.match(v):
            raise ValueError(
                f"Invalid JSON Pointer 'from' path '{v}'. Must follow RFC 6901 format."
            )
        return v

    @model_validator(mode="after")
    def validate_operation_requirements(self) -> "JSONPatchOp":
        """
        Validate operation has required fields.

        Returns:
            The validated operation.

        Raises:
            ValueError: If required fields for operation are missing.
        """
        # add, replace, test require value
        if self.op in ("add", "replace", "test"):
            # Note: value can be None (valid JSON value), so we only validate
            # that it's explicitly provided when needed
            pass

        # remove doesn't use value
        if self.op == "remove" and self.value is not None:
            # Allow value on remove for idempotency checking, but it's not required
            pass

        # move, copy require from_
        if self.op in ("move", "copy"):
            if self.from_ is None:
                raise ValueError(
                    f"Operation '{self.op}' requires 'from' field"
                )

        return self

    def is_additive(self) -> bool:
        """
        Check if this operation adds data.

        Returns:
            True if operation is add or copy.
        """
        return self.op in ("add", "copy")

    def is_destructive(self) -> bool:
        """
        Check if this operation removes data.

        Returns:
            True if operation is remove.
        """
        return self.op == "remove"

    def is_modification(self) -> bool:
        """
        Check if this operation modifies existing data.

        Returns:
            True if operation is replace or move.
        """
        return self.op in ("replace", "move")

    def is_test(self) -> bool:
        """
        Check if this is a test/precondition operation.

        Returns:
            True if operation is test.
        """
        return self.op == "test"

    def to_rfc6902(self) -> dict:
        """
        Convert to RFC 6902 compliant dictionary.

        Returns:
            Dictionary representation per RFC 6902.

        Example:
            >>> op = JSONPatchOp(op="move", from_="/old", path="/new")
            >>> op.to_rfc6902()
            {'op': 'move', 'from': '/old', 'path': '/new'}
        """
        result = {"op": self.op, "path": self.path}

        if self.op in ("add", "replace", "test"):
            result["value"] = self.value
        elif self.op in ("move", "copy"):
            result["from"] = self.from_

        return result

    def __str__(self) -> str:
        """Return string representation."""
        if self.op in ("move", "copy"):
            return f"{self.op} {self.from_} -> {self.path}"
        elif self.op == "remove":
            return f"{self.op} {self.path}"
        else:
            value_repr = repr(self.value)
            if len(value_repr) > 50:
                value_repr = value_repr[:47] + "..."
            return f"{self.op} {self.path} = {value_repr}"


class PatchSafety(str, Enum):
    """
    Safety classification for fix suggestions.

    Defines how safe it is to automatically apply a patch:
    - SAFE: Can be auto-applied without review
    - NEEDS_REVIEW: Should be reviewed by a human before applying
    - UNSAFE: Should not be auto-applied, requires explicit confirmation

    Example:
        >>> safety = PatchSafety.SAFE
        >>> print(safety.value)
        safe
    """

    SAFE = "safe"
    NEEDS_REVIEW = "needs_review"
    UNSAFE = "unsafe"

    def allows_auto_apply(self) -> bool:
        """Check if this safety level allows automatic application."""
        return self == PatchSafety.SAFE

    def requires_review(self) -> bool:
        """Check if this safety level requires human review."""
        return self in (PatchSafety.NEEDS_REVIEW, PatchSafety.UNSAFE)

    def numeric_level(self) -> int:
        """
        Get numeric safety level (lower = safer).

        Returns:
            Integer safety level: SAFE=1, NEEDS_REVIEW=2, UNSAFE=3
        """
        levels = {
            PatchSafety.SAFE: 1,
            PatchSafety.NEEDS_REVIEW: 2,
            PatchSafety.UNSAFE: 3,
        }
        return levels[self]


class FixSuggestion(BaseModel):
    """
    A complete fix suggestion for a validation finding.

    Contains one or more JSON Patch operations that would fix
    a validation error, along with safety information and rationale.

    Attributes:
        patch: List of JSON Patch operations to apply.
        preconditions: Test operations that must pass before applying.
        confidence: Confidence score for this suggestion (0.0 to 1.0).
        safety: Safety classification (safe/needs_review/unsafe).
        rationale: Human-readable explanation of the fix.

    Example:
        >>> suggestion = FixSuggestion(
        ...     patch=[
        ...         JSONPatchOp(op="replace", path="/energy/value", value=100)
        ...     ],
        ...     preconditions=[
        ...         JSONPatchOp(op="test", path="/energy/value", value="100")
        ...     ],
        ...     confidence=0.95,
        ...     safety=PatchSafety.SAFE,
        ...     rationale="Converting string '100' to integer 100"
        ... )
    """

    patch: List[JSONPatchOp] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of JSON Patch operations to apply"
    )

    preconditions: List[JSONPatchOp] = Field(
        default_factory=list,
        max_length=50,
        description="Test operations that must pass before applying"
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score (0.0 to 1.0)"
    )

    safety: PatchSafety = Field(
        ...,
        description="Safety classification"
    )

    rationale: str = Field(
        ...,
        min_length=1,
        max_length=2048,
        description="Human-readable explanation of the fix"
    )

    model_config = {
        "frozen": False,
        "extra": "forbid",
        "json_schema_extra": {
            "examples": [
                {
                    "patch": [
                        {"op": "replace", "path": "/energy/value", "value": 100}
                    ],
                    "preconditions": [
                        {"op": "test", "path": "/energy/value", "value": "100"}
                    ],
                    "confidence": 0.95,
                    "safety": "safe",
                    "rationale": "Converting string '100' to integer 100"
                },
                {
                    "patch": [
                        {"op": "move", "from": "/emmisions", "path": "/emissions"}
                    ],
                    "preconditions": [],
                    "confidence": 0.85,
                    "safety": "needs_review",
                    "rationale": "Correcting likely typo: 'emmisions' -> 'emissions'"
                }
            ]
        }
    }

    @field_validator("preconditions")
    @classmethod
    def validate_preconditions(cls, v: List[JSONPatchOp]) -> List[JSONPatchOp]:
        """
        Validate all preconditions are test operations.

        Args:
            v: List of precondition operations.

        Returns:
            The validated list.

        Raises:
            ValueError: If any precondition is not a test operation.
        """
        for op in v:
            if op.op != "test":
                raise ValueError(
                    f"Precondition must be a 'test' operation, got '{op.op}'"
                )
        return v

    def is_safe(self) -> bool:
        """
        Check if this suggestion is safe to auto-apply.

        Returns:
            True if safety level is SAFE.
        """
        return self.safety == PatchSafety.SAFE

    def is_high_confidence(self, threshold: float = 0.9) -> bool:
        """
        Check if this suggestion has high confidence.

        Args:
            threshold: Minimum confidence threshold (default 0.9).

        Returns:
            True if confidence meets or exceeds threshold.
        """
        return self.confidence >= threshold

    def should_auto_apply(self, min_confidence: float = 0.9) -> bool:
        """
        Check if this suggestion should be auto-applied.

        Args:
            min_confidence: Minimum required confidence.

        Returns:
            True if safe and high confidence.
        """
        return self.is_safe() and self.is_high_confidence(min_confidence)

    def operation_count(self) -> int:
        """
        Get total number of operations (patch + preconditions).

        Returns:
            Total operation count.
        """
        return len(self.patch) + len(self.preconditions)

    def affected_paths(self) -> List[str]:
        """
        Get all paths affected by this suggestion.

        Returns:
            List of unique JSON Pointer paths.
        """
        paths = set()
        for op in self.patch:
            paths.add(op.path)
            if op.from_:
                paths.add(op.from_)
        return sorted(paths)

    def to_patch_document(self) -> List[dict]:
        """
        Convert to RFC 6902 patch document.

        Returns:
            List of operations in RFC 6902 format.
            Preconditions are prepended as test operations.

        Example:
            >>> suggestion.to_patch_document()
            [
                {'op': 'test', 'path': '/energy/value', 'value': '100'},
                {'op': 'replace', 'path': '/energy/value', 'value': 100}
            ]
        """
        document = []

        # Add preconditions first
        for op in self.preconditions:
            document.append(op.to_rfc6902())

        # Then add patch operations
        for op in self.patch:
            document.append(op.to_rfc6902())

        return document

    def format_summary(self) -> str:
        """
        Format a human-readable summary.

        Returns:
            Summary string.
        """
        op_count = len(self.patch)
        paths = ", ".join(self.affected_paths()[:3])
        if len(self.affected_paths()) > 3:
            paths += f", ... (+{len(self.affected_paths()) - 3} more)"

        return (
            f"[{self.safety.value.upper()}] {op_count} op(s) at {paths} "
            f"({self.confidence*100:.0f}% confidence): {self.rationale}"
        )

    def __str__(self) -> str:
        """Return string representation."""
        return self.format_summary()

    def __repr__(self) -> str:
        """Return detailed representation."""
        return (
            f"FixSuggestion(ops={len(self.patch)}, "
            f"safety={self.safety.value}, confidence={self.confidence:.2f})"
        )
