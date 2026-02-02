# -*- coding: utf-8 -*-
"""
Type Coercion Engine for GL-FOUND-X-002.

This module implements safe type coercions from strings to typed values,
following the zero-hallucination principle with exact, reversible coercions.

Safe Coercions:
    - "42" -> 42 (string to integer, exact match only)
    - "3.14" -> 3.14 (string to float, exact parsing)
    - "true"/"false" -> True/False (string to boolean, case insensitive)
    - "null" -> None (string to null)
    - 42 -> 42.0 (integer to float, if needed)

Aggressive Coercions (policy=AGGRESSIVE only):
    - 0/1 -> False/True (integer to boolean)
    - "" -> None (empty string to null)

Design Principles:
    - All coercions are exact (no rounding, no precision loss)
    - All coercions are reversible and auditable
    - Coercion records track original values for provenance
    - Policy controls which coercions are allowed

Example:
    >>> from greenlang.schema.normalizer.coercions import CoercionEngine
    >>> from greenlang.schema.models.config import CoercionPolicy
    >>> engine = CoercionEngine(policy=CoercionPolicy.SAFE)
    >>> result = engine.coerce("42", "integer", "/value")
    >>> if result.success:
    ...     print(result.value)  # 42
    ...     print(result.record.reversible)  # True

Author: GreenLang Framework Team
Version: 0.1.0
GL-FOUND-X-002: Schema Compiler & Validator - Task 3.1
"""

from __future__ import annotations

import logging
import re
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field

from ..models.config import CoercionPolicy

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# JSON Schema type names
JSON_TYPE_STRING = "string"
JSON_TYPE_NUMBER = "number"
JSON_TYPE_INTEGER = "integer"
JSON_TYPE_BOOLEAN = "boolean"
JSON_TYPE_NULL = "null"
JSON_TYPE_OBJECT = "object"
JSON_TYPE_ARRAY = "array"

# Boolean string values (case insensitive)
BOOLEAN_TRUE_STRINGS = frozenset({"true"})
BOOLEAN_FALSE_STRINGS = frozenset({"false"})

# Null string value
NULL_STRING = "null"

# Regex pattern for valid integer string
INTEGER_PATTERN = re.compile(r"^-?(?:0|[1-9]\d*)$")

# Regex pattern for valid number string (including scientific notation)
NUMBER_PATTERN = re.compile(
    r"^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?$"
)


# =============================================================================
# COERCION TYPES
# =============================================================================


class CoercionType(str, Enum):
    """
    Types of coercion operations.

    Each coercion type describes a specific transformation from
    one type to another.
    """

    STRING_TO_INTEGER = "string_to_integer"
    STRING_TO_NUMBER = "string_to_number"
    STRING_TO_BOOLEAN = "string_to_boolean"
    STRING_TO_NULL = "string_to_null"
    INTEGER_TO_NUMBER = "integer_to_number"
    INTEGER_TO_BOOLEAN = "integer_to_boolean"
    FLOAT_TO_INTEGER = "float_to_integer"
    EMPTY_STRING_TO_NULL = "empty_string_to_null"
    NONE_COERCION = "none"


# =============================================================================
# COERCION RECORD MODEL
# =============================================================================


class CoercionRecord(BaseModel):
    """
    Record of a coercion operation.

    Captures complete information about a type coercion for audit
    and provenance tracking. All coercions are recorded to enable
    reversibility verification and debugging.

    Attributes:
        path: JSON Pointer path to the coerced value (RFC 6901)
        original_value: The value before coercion
        original_type: JSON Schema type name of original value
        coerced_value: The value after coercion
        coerced_type: JSON Schema type name of coerced value
        reversible: Whether the coercion can be reversed without loss
        coercion_type: The type of coercion performed

    Example:
        >>> record = CoercionRecord(
        ...     path="/energy",
        ...     original_value="42",
        ...     original_type="string",
        ...     coerced_value=42,
        ...     coerced_type="integer",
        ...     reversible=True,
        ...     coercion_type="string_to_integer"
        ... )
    """

    path: str = Field(
        ...,
        description="JSON Pointer path to the coerced value"
    )
    original_value: Any = Field(
        ...,
        description="The value before coercion"
    )
    original_type: str = Field(
        ...,
        description="JSON Schema type name of original value"
    )
    coerced_value: Any = Field(
        ...,
        description="The value after coercion"
    )
    coerced_type: str = Field(
        ...,
        description="JSON Schema type name of coerced value"
    )
    reversible: bool = Field(
        ...,
        description="Whether the coercion can be reversed without loss"
    )
    coercion_type: str = Field(
        ...,
        description="The type of coercion performed"
    )

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the coercion record.
        """
        return {
            "path": self.path,
            "original_value": self.original_value,
            "original_type": self.original_type,
            "coerced_value": self.coerced_value,
            "coerced_type": self.coerced_type,
            "reversible": self.reversible,
            "coercion_type": self.coercion_type,
        }


# =============================================================================
# COERCION RESULT MODEL
# =============================================================================


class CoercionResult(BaseModel):
    """
    Result of a coercion attempt.

    Encapsulates the outcome of a coercion operation, including
    success status, the resulting value, and optional error details.

    Attributes:
        success: Whether the coercion succeeded
        value: The coerced value (or original if no coercion needed/failed)
        record: Coercion record if coercion was performed
        error: Error message if coercion failed

    Example:
        >>> result = engine.coerce("42", "integer", "/value")
        >>> if result.success:
        ...     print(result.value)
        ... else:
        ...     print(result.error)
    """

    success: bool = Field(
        ...,
        description="Whether the coercion succeeded"
    )
    value: Any = Field(
        ...,
        description="The coerced value or original if failed"
    )
    record: Optional[CoercionRecord] = Field(
        default=None,
        description="Coercion record if coercion was performed"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if coercion failed"
    )

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    @classmethod
    def success_result(
        cls,
        value: Any,
        record: Optional[CoercionRecord] = None
    ) -> "CoercionResult":
        """
        Create a successful coercion result.

        Args:
            value: The coerced value
            record: Optional coercion record

        Returns:
            CoercionResult with success=True
        """
        return cls(success=True, value=value, record=record, error=None)

    @classmethod
    def failure_result(
        cls,
        original_value: Any,
        error: str
    ) -> "CoercionResult":
        """
        Create a failed coercion result.

        Args:
            original_value: The original value that couldn't be coerced
            error: Error message describing the failure

        Returns:
            CoercionResult with success=False
        """
        return cls(success=False, value=original_value, record=None, error=error)

    @classmethod
    def no_coercion_needed(cls, value: Any) -> "CoercionResult":
        """
        Create a result indicating no coercion was needed.

        Args:
            value: The original value (already correct type)

        Returns:
            CoercionResult with success=True and no record
        """
        return cls(success=True, value=value, record=None, error=None)


# =============================================================================
# COERCION ENGINE
# =============================================================================


class CoercionEngine:
    """
    Engine for safe type coercions.

    Performs type coercions according to the configured policy, ensuring
    that all coercions are exact, reversible, and auditable. Follows
    the zero-hallucination principle by only performing deterministic
    transformations.

    Policies:
        - OFF: No coercion (validation fails on type mismatch)
        - SAFE: Only lossless, exact coercions
        - AGGRESSIVE: More permissive coercions (use with caution)

    Attributes:
        policy: The coercion policy to apply

    Example:
        >>> engine = CoercionEngine(policy=CoercionPolicy.SAFE)
        >>> result = engine.coerce("42", "integer", "/value")
        >>> if result.success:
        ...     print(f"Coerced to: {result.value}")
        ...     if result.record:
        ...         print(f"Original: {result.record.original_value}")
    """

    def __init__(self, policy: CoercionPolicy = CoercionPolicy.SAFE):
        """
        Initialize the coercion engine.

        Args:
            policy: Coercion policy controlling which coercions are allowed.
                   Defaults to SAFE for zero-hallucination compliance.
        """
        self.policy = policy
        self._records: List[CoercionRecord] = []
        logger.debug(f"CoercionEngine initialized with policy: {policy.value}")

    # -------------------------------------------------------------------------
    # Main Coercion Method
    # -------------------------------------------------------------------------

    def coerce(
        self,
        value: Any,
        target_type: str,
        path: str
    ) -> CoercionResult:
        """
        Coerce value to target type if safe.

        Attempts to coerce the given value to the target JSON Schema type.
        The coercion is performed only if:
        1. The policy allows coercion (not OFF)
        2. The coercion is safe according to the policy
        3. The coercion is exact (no data loss)

        Args:
            value: The value to coerce
            target_type: Expected JSON Schema type
                        ("string", "number", "integer", "boolean", "null")
            path: JSON Pointer path for provenance tracking

        Returns:
            CoercionResult with:
            - success=True and coerced value if coercion succeeded
            - success=True and original value if no coercion needed
            - success=False and error if coercion failed

        Example:
            >>> result = engine.coerce("42", "integer", "/energy")
            >>> assert result.success
            >>> assert result.value == 42
        """
        # Policy OFF means no coercion allowed
        if self.policy == CoercionPolicy.OFF:
            return self._check_type_match(value, target_type, path)

        # Get current type of value
        current_type = get_python_type_name(value)

        # If already correct type, no coercion needed
        if self._types_match(current_type, target_type):
            return CoercionResult.no_coercion_needed(value)

        # Dispatch to appropriate coercion method
        if target_type == JSON_TYPE_INTEGER:
            return self.coerce_to_integer(value, path)
        elif target_type == JSON_TYPE_NUMBER:
            return self.coerce_to_number(value, path)
        elif target_type == JSON_TYPE_BOOLEAN:
            return self.coerce_to_boolean(value, path)
        elif target_type == JSON_TYPE_STRING:
            return self.coerce_to_string(value, path)
        elif target_type == JSON_TYPE_NULL:
            return self.coerce_to_null(value, path)
        else:
            # Unsupported target type (object, array, etc.)
            return CoercionResult.failure_result(
                value,
                f"Cannot coerce to type '{target_type}' - not a primitive type"
            )

    # -------------------------------------------------------------------------
    # Type-Specific Coercion Methods
    # -------------------------------------------------------------------------

    def coerce_to_integer(
        self,
        value: Any,
        path: str
    ) -> CoercionResult:
        """
        Coerce to integer.

        Safe coercions:
        - String "42" -> 42 (exact match only, no leading zeros except "0")
        - Float 42.0 -> 42 (only if no decimal part)

        Args:
            value: The value to coerce
            path: JSON Pointer path for record

        Returns:
            CoercionResult with integer value or error

        Example:
            >>> result = engine.coerce_to_integer("42", "/value")
            >>> assert result.success and result.value == 42
            >>> result = engine.coerce_to_integer("42.5", "/value")
            >>> assert not result.success  # Has decimal part
        """
        current_type = get_python_type_name(value)

        # Already an integer
        if current_type == JSON_TYPE_INTEGER:
            return CoercionResult.no_coercion_needed(value)

        # String to integer
        if current_type == JSON_TYPE_STRING:
            return self._coerce_string_to_integer(value, path)

        # Float to integer (only if no decimal part)
        if current_type == JSON_TYPE_NUMBER:
            return self._coerce_float_to_integer(value, path)

        return CoercionResult.failure_result(
            value,
            f"Cannot coerce {current_type} to integer"
        )

    def coerce_to_number(
        self,
        value: Any,
        path: str
    ) -> CoercionResult:
        """
        Coerce to number (float).

        Safe coercions:
        - String "3.14" -> 3.14 (exact parsing)
        - String "42" -> 42.0 (integer string to float)
        - Integer 42 -> 42.0 (always exact)

        Args:
            value: The value to coerce
            path: JSON Pointer path for record

        Returns:
            CoercionResult with number value or error

        Example:
            >>> result = engine.coerce_to_number("3.14", "/value")
            >>> assert result.success and result.value == 3.14
            >>> result = engine.coerce_to_number(42, "/value")
            >>> assert result.success and result.value == 42.0
        """
        current_type = get_python_type_name(value)

        # Already a number (float)
        if current_type == JSON_TYPE_NUMBER:
            return CoercionResult.no_coercion_needed(value)

        # Integer to number (always safe)
        if current_type == JSON_TYPE_INTEGER:
            return self._coerce_integer_to_number(value, path)

        # String to number
        if current_type == JSON_TYPE_STRING:
            return self._coerce_string_to_number(value, path)

        return CoercionResult.failure_result(
            value,
            f"Cannot coerce {current_type} to number"
        )

    def coerce_to_boolean(
        self,
        value: Any,
        path: str
    ) -> CoercionResult:
        """
        Coerce to boolean.

        Safe coercions:
        - String "true"/"false" -> True/False (case insensitive)

        Aggressive coercions (policy=AGGRESSIVE only):
        - Integer 0/1 -> False/True

        Args:
            value: The value to coerce
            path: JSON Pointer path for record

        Returns:
            CoercionResult with boolean value or error

        Example:
            >>> result = engine.coerce_to_boolean("true", "/flag")
            >>> assert result.success and result.value is True
            >>> result = engine.coerce_to_boolean("TRUE", "/flag")
            >>> assert result.success and result.value is True
        """
        current_type = get_python_type_name(value)

        # Already a boolean
        if current_type == JSON_TYPE_BOOLEAN:
            return CoercionResult.no_coercion_needed(value)

        # String to boolean (safe mode)
        if current_type == JSON_TYPE_STRING:
            return self._coerce_string_to_boolean(value, path)

        # Integer to boolean (aggressive mode only)
        if current_type == JSON_TYPE_INTEGER:
            if self.policy.is_aggressive():
                return self._coerce_integer_to_boolean(value, path)
            return CoercionResult.failure_result(
                value,
                "Integer to boolean coercion requires aggressive policy"
            )

        return CoercionResult.failure_result(
            value,
            f"Cannot coerce {current_type} to boolean"
        )

    def coerce_to_string(
        self,
        value: Any,
        path: str
    ) -> CoercionResult:
        """
        Coerce to string.

        Safe coercions:
        - Number -> string representation (e.g., 42 -> "42", 3.14 -> "3.14")
        - Boolean -> "true"/"false"
        - Null -> "null"

        Args:
            value: The value to coerce
            path: JSON Pointer path for record

        Returns:
            CoercionResult with string value or error

        Example:
            >>> result = engine.coerce_to_string(42, "/value")
            >>> assert result.success and result.value == "42"
            >>> result = engine.coerce_to_string(True, "/flag")
            >>> assert result.success and result.value == "true"
        """
        current_type = get_python_type_name(value)

        # Already a string
        if current_type == JSON_TYPE_STRING:
            return CoercionResult.no_coercion_needed(value)

        # Number to string
        if current_type in (JSON_TYPE_NUMBER, JSON_TYPE_INTEGER):
            return self._coerce_number_to_string(value, path)

        # Boolean to string
        if current_type == JSON_TYPE_BOOLEAN:
            return self._coerce_boolean_to_string(value, path)

        # Null to string
        if current_type == JSON_TYPE_NULL:
            return self._coerce_null_to_string(value, path)

        return CoercionResult.failure_result(
            value,
            f"Cannot coerce {current_type} to string"
        )

    def coerce_to_null(
        self,
        value: Any,
        path: str
    ) -> CoercionResult:
        """
        Coerce to null.

        Safe coercions:
        - String "null" -> None (case insensitive)

        Aggressive coercions (policy=AGGRESSIVE only):
        - Empty string "" -> None

        Args:
            value: The value to coerce
            path: JSON Pointer path for record

        Returns:
            CoercionResult with None or error

        Example:
            >>> result = engine.coerce_to_null("null", "/value")
            >>> assert result.success and result.value is None
        """
        current_type = get_python_type_name(value)

        # Already null
        if current_type == JSON_TYPE_NULL:
            return CoercionResult.no_coercion_needed(value)

        # String to null
        if current_type == JSON_TYPE_STRING:
            return self._coerce_string_to_null(value, path)

        return CoercionResult.failure_result(
            value,
            f"Cannot coerce {current_type} to null"
        )

    # -------------------------------------------------------------------------
    # Private Coercion Implementation Methods
    # -------------------------------------------------------------------------

    def _coerce_string_to_integer(
        self,
        value: str,
        path: str
    ) -> CoercionResult:
        """
        Coerce string to integer with exact matching.

        Only succeeds if:
        - String matches integer pattern (no leading zeros except "0")
        - String represents exact integer (not float)
        - Round-trip conversion matches original

        Args:
            value: String value
            path: JSON Pointer path

        Returns:
            CoercionResult with integer or error
        """
        if not self._is_exact_integer_string(value):
            return CoercionResult.failure_result(
                value,
                f"String '{value}' is not an exact integer"
            )

        try:
            int_value = int(value)

            # Verify round-trip (ensures no precision loss)
            if str(int_value) != value.strip():
                return CoercionResult.failure_result(
                    value,
                    f"Integer conversion not exact: '{value}' vs '{int_value}'"
                )

            record = self._create_record(
                path=path,
                original_value=value,
                original_type=JSON_TYPE_STRING,
                coerced_value=int_value,
                coerced_type=JSON_TYPE_INTEGER,
                reversible=True,
                coercion_type=CoercionType.STRING_TO_INTEGER.value
            )

            return CoercionResult.success_result(int_value, record)

        except (ValueError, OverflowError) as e:
            return CoercionResult.failure_result(
                value,
                f"Failed to convert string to integer: {e}"
            )

    def _coerce_string_to_number(
        self,
        value: str,
        path: str
    ) -> CoercionResult:
        """
        Coerce string to number (float) with exact parsing.

        Only succeeds if:
        - String matches number pattern
        - No special values (inf, nan)
        - Can be parsed as valid float

        Args:
            value: String value
            path: JSON Pointer path

        Returns:
            CoercionResult with float or error
        """
        if not self._is_exact_number_string(value):
            return CoercionResult.failure_result(
                value,
                f"String '{value}' is not a valid number"
            )

        try:
            # Use Decimal for exact parsing, then convert to float
            decimal_value = Decimal(value.strip())
            float_value = float(decimal_value)

            # Check for infinity (overflow)
            if float_value in (float("inf"), float("-inf")):
                return CoercionResult.failure_result(
                    value,
                    f"Number '{value}' overflows to infinity"
                )

            record = self._create_record(
                path=path,
                original_value=value,
                original_type=JSON_TYPE_STRING,
                coerced_value=float_value,
                coerced_type=JSON_TYPE_NUMBER,
                reversible=True,
                coercion_type=CoercionType.STRING_TO_NUMBER.value
            )

            return CoercionResult.success_result(float_value, record)

        except (ValueError, InvalidOperation, OverflowError) as e:
            return CoercionResult.failure_result(
                value,
                f"Failed to convert string to number: {e}"
            )

    def _coerce_string_to_boolean(
        self,
        value: str,
        path: str
    ) -> CoercionResult:
        """
        Coerce string to boolean.

        Accepts "true" and "false" (case insensitive).

        Args:
            value: String value
            path: JSON Pointer path

        Returns:
            CoercionResult with boolean or error
        """
        value_lower = value.strip().lower()

        if value_lower in BOOLEAN_TRUE_STRINGS:
            record = self._create_record(
                path=path,
                original_value=value,
                original_type=JSON_TYPE_STRING,
                coerced_value=True,
                coerced_type=JSON_TYPE_BOOLEAN,
                reversible=True,
                coercion_type=CoercionType.STRING_TO_BOOLEAN.value
            )
            return CoercionResult.success_result(True, record)

        if value_lower in BOOLEAN_FALSE_STRINGS:
            record = self._create_record(
                path=path,
                original_value=value,
                original_type=JSON_TYPE_STRING,
                coerced_value=False,
                coerced_type=JSON_TYPE_BOOLEAN,
                reversible=True,
                coercion_type=CoercionType.STRING_TO_BOOLEAN.value
            )
            return CoercionResult.success_result(False, record)

        return CoercionResult.failure_result(
            value,
            f"String '{value}' is not a valid boolean (expected 'true' or 'false')"
        )

    def _coerce_string_to_null(
        self,
        value: str,
        path: str
    ) -> CoercionResult:
        """
        Coerce string to null.

        Safe mode: Only accepts "null" (case insensitive)
        Aggressive mode: Also accepts empty string ""

        Args:
            value: String value
            path: JSON Pointer path

        Returns:
            CoercionResult with None or error
        """
        value_stripped = value.strip()
        value_lower = value_stripped.lower()

        # "null" string -> None
        if value_lower == NULL_STRING:
            record = self._create_record(
                path=path,
                original_value=value,
                original_type=JSON_TYPE_STRING,
                coerced_value=None,
                coerced_type=JSON_TYPE_NULL,
                reversible=True,
                coercion_type=CoercionType.STRING_TO_NULL.value
            )
            return CoercionResult.success_result(None, record)

        # Empty string -> None (aggressive mode only)
        if value_stripped == "" and self.policy.is_aggressive():
            record = self._create_record(
                path=path,
                original_value=value,
                original_type=JSON_TYPE_STRING,
                coerced_value=None,
                coerced_type=JSON_TYPE_NULL,
                reversible=False,  # Not reversible (could have been empty string)
                coercion_type=CoercionType.EMPTY_STRING_TO_NULL.value
            )
            return CoercionResult.success_result(None, record)

        return CoercionResult.failure_result(
            value,
            f"String '{value}' cannot be coerced to null (expected 'null')"
        )

    def _coerce_integer_to_number(
        self,
        value: int,
        path: str
    ) -> CoercionResult:
        """
        Coerce integer to number (float).

        This is always safe as integers can be exactly represented as floats
        (within reasonable ranges).

        Args:
            value: Integer value
            path: JSON Pointer path

        Returns:
            CoercionResult with float
        """
        float_value = float(value)

        # Verify exact conversion (for very large integers)
        if int(float_value) != value:
            return CoercionResult.failure_result(
                value,
                f"Integer {value} cannot be exactly represented as float"
            )

        record = self._create_record(
            path=path,
            original_value=value,
            original_type=JSON_TYPE_INTEGER,
            coerced_value=float_value,
            coerced_type=JSON_TYPE_NUMBER,
            reversible=True,
            coercion_type=CoercionType.INTEGER_TO_NUMBER.value
        )

        return CoercionResult.success_result(float_value, record)

    def _coerce_float_to_integer(
        self,
        value: float,
        path: str
    ) -> CoercionResult:
        """
        Coerce float to integer (only if no decimal part).

        Only succeeds if value has no fractional part (e.g., 42.0 -> 42).

        Args:
            value: Float value
            path: JSON Pointer path

        Returns:
            CoercionResult with integer or error
        """
        # Check if value is a whole number
        if value != int(value):
            return CoercionResult.failure_result(
                value,
                f"Float {value} has decimal part, cannot coerce to integer"
            )

        int_value = int(value)

        record = self._create_record(
            path=path,
            original_value=value,
            original_type=JSON_TYPE_NUMBER,
            coerced_value=int_value,
            coerced_type=JSON_TYPE_INTEGER,
            reversible=True,
            coercion_type=CoercionType.FLOAT_TO_INTEGER.value
        )

        return CoercionResult.success_result(int_value, record)

    def _coerce_integer_to_boolean(
        self,
        value: int,
        path: str
    ) -> CoercionResult:
        """
        Coerce integer to boolean (aggressive mode only).

        Only 0 -> False and 1 -> True are accepted.

        Args:
            value: Integer value
            path: JSON Pointer path

        Returns:
            CoercionResult with boolean or error
        """
        if value == 0:
            record = self._create_record(
                path=path,
                original_value=value,
                original_type=JSON_TYPE_INTEGER,
                coerced_value=False,
                coerced_type=JSON_TYPE_BOOLEAN,
                reversible=True,
                coercion_type=CoercionType.INTEGER_TO_BOOLEAN.value
            )
            return CoercionResult.success_result(False, record)

        if value == 1:
            record = self._create_record(
                path=path,
                original_value=value,
                original_type=JSON_TYPE_INTEGER,
                coerced_value=True,
                coerced_type=JSON_TYPE_BOOLEAN,
                reversible=True,
                coercion_type=CoercionType.INTEGER_TO_BOOLEAN.value
            )
            return CoercionResult.success_result(True, record)

        return CoercionResult.failure_result(
            value,
            f"Integer {value} cannot be coerced to boolean (only 0 or 1 allowed)"
        )

    def _coerce_number_to_string(
        self,
        value: Union[int, float],
        path: str
    ) -> CoercionResult:
        """
        Coerce number to string representation.

        Args:
            value: Numeric value
            path: JSON Pointer path

        Returns:
            CoercionResult with string
        """
        original_type = get_python_type_name(value)

        # For integers, use simple string conversion
        if isinstance(value, int) and not isinstance(value, bool):
            str_value = str(value)
        else:
            # For floats, handle special cases
            if value == int(value):
                # Whole number float -> integer string
                str_value = str(int(value))
            else:
                str_value = str(value)

        record = self._create_record(
            path=path,
            original_value=value,
            original_type=original_type,
            coerced_value=str_value,
            coerced_type=JSON_TYPE_STRING,
            reversible=True,
            coercion_type=CoercionType.STRING_TO_NUMBER.value
        )

        return CoercionResult.success_result(str_value, record)

    def _coerce_boolean_to_string(
        self,
        value: bool,
        path: str
    ) -> CoercionResult:
        """
        Coerce boolean to string ("true" or "false").

        Args:
            value: Boolean value
            path: JSON Pointer path

        Returns:
            CoercionResult with string
        """
        str_value = "true" if value else "false"

        record = self._create_record(
            path=path,
            original_value=value,
            original_type=JSON_TYPE_BOOLEAN,
            coerced_value=str_value,
            coerced_type=JSON_TYPE_STRING,
            reversible=True,
            coercion_type=CoercionType.STRING_TO_BOOLEAN.value
        )

        return CoercionResult.success_result(str_value, record)

    def _coerce_null_to_string(
        self,
        value: None,
        path: str
    ) -> CoercionResult:
        """
        Coerce null to string "null".

        Args:
            value: None value
            path: JSON Pointer path

        Returns:
            CoercionResult with string "null"
        """
        record = self._create_record(
            path=path,
            original_value=value,
            original_type=JSON_TYPE_NULL,
            coerced_value="null",
            coerced_type=JSON_TYPE_STRING,
            reversible=True,
            coercion_type=CoercionType.STRING_TO_NULL.value
        )

        return CoercionResult.success_result("null", record)

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _is_exact_integer_string(self, s: str) -> bool:
        """
        Check if string represents exact integer.

        Valid formats:
        - "0", "42", "-123"
        - No leading zeros (except "0" itself)
        - No decimal point
        - No scientific notation

        Args:
            s: String to check

        Returns:
            True if string represents an exact integer
        """
        s = s.strip()
        if not s:
            return False
        return bool(INTEGER_PATTERN.match(s))

    def _is_exact_number_string(self, s: str) -> bool:
        """
        Check if string represents exact number.

        Valid formats:
        - Integers: "0", "42", "-123"
        - Decimals: "3.14", "-0.5"
        - Scientific notation: "1e10", "1.5e-3"

        Invalid:
        - "inf", "-inf", "infinity"
        - "nan"
        - Empty string

        Args:
            s: String to check

        Returns:
            True if string represents an exact number
        """
        s = s.strip().lower()
        if not s:
            return False

        # Reject special values
        if s in ("inf", "-inf", "infinity", "-infinity", "nan", "+inf", "+infinity"):
            return False

        return bool(NUMBER_PATTERN.match(s))

    def _types_match(self, current_type: str, target_type: str) -> bool:
        """
        Check if current type matches target type.

        Handles the case where integer is a subtype of number.

        Args:
            current_type: Current JSON type
            target_type: Target JSON type

        Returns:
            True if types match (no coercion needed)
        """
        if current_type == target_type:
            return True

        # Integer is a valid number
        if current_type == JSON_TYPE_INTEGER and target_type == JSON_TYPE_NUMBER:
            return True

        return False

    def _check_type_match(
        self,
        value: Any,
        target_type: str,
        path: str
    ) -> CoercionResult:
        """
        Check if value matches target type (when coercion is OFF).

        Args:
            value: Value to check
            target_type: Expected type
            path: JSON Pointer path

        Returns:
            CoercionResult with success if types match, error otherwise
        """
        current_type = get_python_type_name(value)

        if self._types_match(current_type, target_type):
            return CoercionResult.no_coercion_needed(value)

        return CoercionResult.failure_result(
            value,
            f"Type mismatch: expected {target_type}, got {current_type} "
            f"(coercion disabled)"
        )

    def _create_record(
        self,
        path: str,
        original_value: Any,
        original_type: str,
        coerced_value: Any,
        coerced_type: str,
        reversible: bool,
        coercion_type: str
    ) -> CoercionRecord:
        """
        Create and store a coercion record.

        Args:
            path: JSON Pointer path
            original_value: Original value
            original_type: Original JSON type
            coerced_value: Coerced value
            coerced_type: Coerced JSON type
            reversible: Whether coercion is reversible
            coercion_type: Type of coercion performed

        Returns:
            Created CoercionRecord
        """
        record = CoercionRecord(
            path=path,
            original_value=original_value,
            original_type=original_type,
            coerced_value=coerced_value,
            coerced_type=coerced_type,
            reversible=reversible,
            coercion_type=coercion_type
        )

        self._records.append(record)
        logger.debug(
            f"Coercion at {path}: {original_type} -> {coerced_type} "
            f"({original_value!r} -> {coerced_value!r})"
        )

        return record

    # -------------------------------------------------------------------------
    # Record Management
    # -------------------------------------------------------------------------

    def get_records(self) -> List[CoercionRecord]:
        """
        Get all coercion records.

        Returns a copy of the internal records list to prevent modification.

        Returns:
            List of all CoercionRecord objects created by this engine
        """
        return list(self._records)

    def clear_records(self) -> None:
        """
        Clear coercion records.

        Removes all stored coercion records. Call this between
        processing different payloads if reusing the engine.
        """
        self._records.clear()
        logger.debug("Coercion records cleared")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def can_coerce(
    value: Any,
    target_type: str,
    policy: CoercionPolicy
) -> bool:
    """
    Quick check if coercion is possible.

    Performs a fast check without actually performing the coercion.
    Useful for validation planning and optimization.

    Args:
        value: The value to check
        target_type: Target JSON Schema type
        policy: Coercion policy to apply

    Returns:
        True if coercion is possible under the given policy

    Example:
        >>> can_coerce("42", "integer", CoercionPolicy.SAFE)
        True
        >>> can_coerce("hello", "integer", CoercionPolicy.SAFE)
        False
    """
    # Policy OFF means no coercion
    if policy == CoercionPolicy.OFF:
        current_type = get_python_type_name(value)
        if current_type == target_type:
            return True
        if current_type == JSON_TYPE_INTEGER and target_type == JSON_TYPE_NUMBER:
            return True
        return False

    # Create a temporary engine to test coercion
    engine = CoercionEngine(policy=policy)
    result = engine.coerce(value, target_type, "")
    return result.success


def get_python_type_name(value: Any) -> str:
    """
    Get JSON Schema type name for Python value.

    Maps Python types to their JSON Schema type equivalents:
    - str -> "string"
    - int (not bool) -> "integer"
    - float -> "number"
    - bool -> "boolean"
    - None -> "null"
    - list -> "array"
    - dict -> "object"

    Args:
        value: Python value to check

    Returns:
        JSON Schema type name as string

    Example:
        >>> get_python_type_name(42)
        'integer'
        >>> get_python_type_name(3.14)
        'number'
        >>> get_python_type_name("hello")
        'string'
    """
    if value is None:
        return JSON_TYPE_NULL

    if isinstance(value, bool):
        # Must check bool before int (bool is subclass of int)
        return JSON_TYPE_BOOLEAN

    if isinstance(value, int):
        return JSON_TYPE_INTEGER

    if isinstance(value, float):
        return JSON_TYPE_NUMBER

    if isinstance(value, str):
        return JSON_TYPE_STRING

    if isinstance(value, list):
        return JSON_TYPE_ARRAY

    if isinstance(value, dict):
        return JSON_TYPE_OBJECT

    # Unknown type - return Python type name
    return type(value).__name__


# =============================================================================
# MODULE EXPORTS
# =============================================================================


__all__ = [
    # Constants
    "JSON_TYPE_STRING",
    "JSON_TYPE_NUMBER",
    "JSON_TYPE_INTEGER",
    "JSON_TYPE_BOOLEAN",
    "JSON_TYPE_NULL",
    "JSON_TYPE_OBJECT",
    "JSON_TYPE_ARRAY",
    # Enums
    "CoercionType",
    # Models
    "CoercionRecord",
    "CoercionResult",
    # Engine
    "CoercionEngine",
    # Utility functions
    "can_coerce",
    "get_python_type_name",
]
