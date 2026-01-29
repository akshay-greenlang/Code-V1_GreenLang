# -*- coding: utf-8 -*-
"""
Schema Linter for GL-FOUND-X-002.

This module implements non-blocking lint checks that produce warnings
and informational messages rather than validation errors. Lint findings
do not fail validation but provide helpful suggestions for improving
data quality, consistency, and correctness.

Lint Checks:
    - Unknown fields with close matches (typo detection via Levenshtein distance)
    - Deprecated field usage warnings
    - Non-canonical casing (snake_case vs camelCase inconsistency)
    - Unit formatting suggestions
    - Suspicious patterns (empty strings, zero values, etc.)

Warning Codes:
    - GLSCHEMA-W600: DEPRECATED_FIELD
    - GLSCHEMA-W601: RENAMED_FIELD
    - GLSCHEMA-W700: SUSPICIOUS_KEY (typo detection)
    - GLSCHEMA-W701: NONCOMPLIANT_CASING
    - GLSCHEMA-W702: UNIT_FORMAT_STYLE

Example:
    >>> from greenlang.schema.validator.linter import SchemaLinter, lint_payload
    >>> from greenlang.schema.compiler.ir import SchemaIR
    >>> from greenlang.schema.models.config import ValidationOptions
    >>>
    >>> linter = SchemaLinter(ir, ValidationOptions())
    >>> findings = linter.lint({"emmisions": 100})  # typo: "emmisions"
    >>> # Returns warning: "Unknown field 'emmisions'. Did you mean 'emissions'?"

Author: GreenLang Framework Team
Version: 0.1.0
GL-FOUND-X-002: Schema Compiler & Validator - Task 2.6
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from greenlang.schema.compiler.ir import SchemaIR
from greenlang.schema.constants import (
    MAX_TYPO_EDIT_DISTANCE,
    MIN_KEY_LENGTH_FOR_TYPO_CHECK,
    NAMING_CONVENTIONS,
    DEFAULT_NAMING_CONVENTION,
)
from greenlang.schema.errors import ErrorCode
from greenlang.schema.models.config import ValidationOptions
from greenlang.schema.models.finding import Finding, FindingHint, Severity


logger = logging.getLogger(__name__)


# =============================================================================
# CASING DETECTION CONSTANTS
# =============================================================================

# Casing style identifiers
CASING_SNAKE_CASE = "snake_case"
CASING_CAMEL_CASE = "camelCase"
CASING_PASCAL_CASE = "PascalCase"
CASING_KEBAB_CASE = "kebab-case"
CASING_SCREAMING_SNAKE = "SCREAMING_SNAKE_CASE"
CASING_UNKNOWN = "unknown"


# =============================================================================
# CASING DETECTION HELPERS
# =============================================================================


def is_snake_case(s: str) -> bool:
    """
    Check if string is snake_case.

    snake_case format:
    - Lowercase letters and underscores
    - Words separated by underscores
    - Must start with a letter
    - Numbers allowed after first character

    Args:
        s: String to check.

    Returns:
        True if string matches snake_case pattern.

    Example:
        >>> is_snake_case("energy_consumption")
        True
        >>> is_snake_case("energyConsumption")
        False
    """
    if not s:
        return False
    return bool(re.match(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$", s))


def is_camel_case(s: str) -> bool:
    """
    Check if string is camelCase.

    camelCase format:
    - First character lowercase
    - Subsequent words capitalized
    - No underscores or hyphens

    Args:
        s: String to check.

    Returns:
        True if string matches camelCase pattern.

    Example:
        >>> is_camel_case("energyConsumption")
        True
        >>> is_camel_case("energy_consumption")
        False
    """
    if not s:
        return False
    # camelCase: starts lowercase, contains uppercase, no underscores/hyphens
    if "_" in s or "-" in s:
        return False
    if not s[0].islower():
        return False
    # Must have at least one uppercase letter (otherwise it's just lowercase)
    return bool(re.match(r"^[a-z][a-zA-Z0-9]*$", s)) and any(c.isupper() for c in s)


def is_pascal_case(s: str) -> bool:
    """
    Check if string is PascalCase.

    PascalCase format:
    - First character uppercase
    - Subsequent words capitalized
    - No underscores or hyphens

    Args:
        s: String to check.

    Returns:
        True if string matches PascalCase pattern.

    Example:
        >>> is_pascal_case("EnergyConsumption")
        True
        >>> is_pascal_case("energyConsumption")
        False
    """
    if not s:
        return False
    if "_" in s or "-" in s:
        return False
    if not s[0].isupper():
        return False
    return bool(re.match(r"^[A-Z][a-zA-Z0-9]*$", s))


def is_kebab_case(s: str) -> bool:
    """
    Check if string is kebab-case.

    kebab-case format:
    - Lowercase letters and hyphens
    - Words separated by hyphens
    - Must start with a letter

    Args:
        s: String to check.

    Returns:
        True if string matches kebab-case pattern.

    Example:
        >>> is_kebab_case("energy-consumption")
        True
        >>> is_kebab_case("energy_consumption")
        False
    """
    if not s:
        return False
    return bool(re.match(r"^[a-z][a-z0-9]*(-[a-z0-9]+)*$", s))


def is_screaming_snake_case(s: str) -> bool:
    """
    Check if string is SCREAMING_SNAKE_CASE.

    SCREAMING_SNAKE_CASE format:
    - Uppercase letters and underscores
    - Words separated by underscores
    - Must start with a letter

    Args:
        s: String to check.

    Returns:
        True if string matches SCREAMING_SNAKE_CASE pattern.

    Example:
        >>> is_screaming_snake_case("ENERGY_CONSUMPTION")
        True
        >>> is_screaming_snake_case("energy_consumption")
        False
    """
    if not s:
        return False
    return bool(re.match(r"^[A-Z][A-Z0-9]*(_[A-Z0-9]+)*$", s))


def to_snake_case(s: str) -> str:
    """
    Convert string to snake_case.

    Handles conversion from:
    - camelCase
    - PascalCase
    - kebab-case
    - SCREAMING_SNAKE_CASE

    Args:
        s: String to convert.

    Returns:
        String converted to snake_case.

    Example:
        >>> to_snake_case("energyConsumption")
        'energy_consumption'
        >>> to_snake_case("EnergyConsumption")
        'energy_consumption'
        >>> to_snake_case("energy-consumption")
        'energy_consumption'
    """
    if not s:
        return s

    # Handle kebab-case
    if "-" in s:
        return s.replace("-", "_").lower()

    # Handle SCREAMING_SNAKE_CASE
    if is_screaming_snake_case(s):
        return s.lower()

    # Handle camelCase and PascalCase
    # Insert underscore before each uppercase letter
    result = re.sub(r"([A-Z])", r"_\1", s)
    # Remove leading underscore and convert to lowercase
    result = result.lstrip("_").lower()
    # Clean up multiple underscores
    result = re.sub(r"_+", "_", result)

    return result


def to_camel_case(s: str) -> str:
    """
    Convert string to camelCase.

    Handles conversion from:
    - snake_case
    - kebab-case
    - PascalCase
    - SCREAMING_SNAKE_CASE

    Args:
        s: String to convert.

    Returns:
        String converted to camelCase.

    Example:
        >>> to_camel_case("energy_consumption")
        'energyConsumption'
        >>> to_camel_case("energy-consumption")
        'energyConsumption'
    """
    if not s:
        return s

    # First convert to snake_case as intermediate form
    snake = to_snake_case(s)

    # Split on underscores
    words = snake.split("_")

    if not words:
        return s

    # First word lowercase, rest capitalized
    result = words[0].lower()
    for word in words[1:]:
        if word:
            result += word.capitalize()

    return result


def to_pascal_case(s: str) -> str:
    """
    Convert string to PascalCase.

    Args:
        s: String to convert.

    Returns:
        String converted to PascalCase.

    Example:
        >>> to_pascal_case("energy_consumption")
        'EnergyConsumption'
    """
    if not s:
        return s

    camel = to_camel_case(s)
    if camel:
        return camel[0].upper() + camel[1:]
    return camel


# =============================================================================
# SCHEMA LINTER CLASS
# =============================================================================


class SchemaLinter:
    """
    Non-blocking lint checks for payload quality.

    The SchemaLinter analyzes payloads and generates warnings/info findings
    without failing validation. It detects:
    - Typos in field names (using Levenshtein distance)
    - Deprecated field usage
    - Inconsistent key casing
    - Unit formatting issues
    - Suspicious values

    All findings from the linter are warnings or info level (non-blocking).

    Attributes:
        ir: Compiled schema Intermediate Representation.
        options: Validation options controlling lint behavior.
        _findings: Accumulated lint findings.

    Example:
        >>> linter = SchemaLinter(ir, ValidationOptions())
        >>> findings = linter.lint({"emmisions": 100})
        >>> for f in findings:
        ...     print(f"{f.code}: {f.message}")
        GLSCHEMA-W700: Unknown field 'emmisions'. Did you mean 'emissions'?
    """

    def __init__(self, ir: SchemaIR, options: Optional[ValidationOptions] = None):
        """
        Initialize SchemaLinter.

        Args:
            ir: Compiled schema IR with property and deprecation info.
            options: Validation options. Defaults to standard options if None.
        """
        self.ir = ir
        self.options = options or ValidationOptions()
        self._findings: List[Finding] = []
        self._max_edit_distance = MAX_TYPO_EDIT_DISTANCE

    def lint(
        self,
        payload: Dict[str, Any],
        path: str = ""
    ) -> List[Finding]:
        """
        Run lint checks on payload.

        All findings are warnings or info (non-blocking).

        Args:
            payload: The payload to lint.
            path: Current JSON Pointer path (for recursive calls).

        Returns:
            List of Finding objects with warnings and info.

        Example:
            >>> findings = linter.lint({"emmisions": 100})
            >>> len([f for f in findings if f.is_warning()]) > 0
            True
        """
        self._findings = []

        if not isinstance(payload, dict):
            logger.debug("Payload is not a dict, skipping lint")
            return self._findings

        # Get known keys from the IR
        known_keys = self._get_known_keys_for_path(path)

        # Run all lint checks
        self._check_unknown_fields(payload, known_keys, path)
        self._check_deprecated_fields(payload, path)
        self._check_casing_consistency(payload, path)
        self._check_suspicious_values(payload, path)

        # Recursively lint nested objects
        for key, value in payload.items():
            child_path = f"{path}/{key}"

            if isinstance(value, dict):
                # Lint nested object
                nested_findings = self.lint(value, child_path)
                self._findings.extend(nested_findings)

            elif isinstance(value, list):
                # Lint array items
                self._lint_array(value, child_path)

            else:
                # Check unit formatting for primitive values
                self._check_unit_formatting(value, child_path)

        return self._findings

    def _get_known_keys_for_path(self, path: str) -> Set[str]:
        """
        Get known property keys for a given path from the IR.

        Args:
            path: JSON Pointer path.

        Returns:
            Set of known property names at this path.
        """
        known_keys: Set[str] = set()

        # Get all properties from the IR and filter by path prefix
        for prop_path in self.ir.properties.keys():
            # Extract the immediate child key name for the given path
            if path == "":
                # Root level - get first segment
                if prop_path.startswith("/"):
                    parts = prop_path.split("/")
                    if len(parts) >= 2:
                        known_keys.add(parts[1])
            else:
                # Nested level - check prefix match
                prefix = path + "/"
                if prop_path.startswith(prefix):
                    remaining = prop_path[len(prefix):]
                    parts = remaining.split("/")
                    if parts:
                        known_keys.add(parts[0])

        return known_keys

    def _lint_array(self, arr: List[Any], path: str) -> None:
        """
        Lint array items recursively.

        Args:
            arr: Array to lint.
            path: JSON Pointer path to array.
        """
        for idx, item in enumerate(arr):
            item_path = f"{path}/{idx}"

            if isinstance(item, dict):
                nested_findings = self.lint(item, item_path)
                self._findings.extend(nested_findings)
            elif isinstance(item, list):
                self._lint_array(item, item_path)
            else:
                self._check_unit_formatting(item, item_path)

    def _check_unknown_fields(
        self,
        obj: Dict[str, Any],
        known_keys: Set[str],
        path: str
    ) -> None:
        """
        Check for unknown fields and suggest corrections.

        Uses Levenshtein distance to find close matches for potential typos.

        Args:
            obj: Object to check.
            known_keys: Set of valid key names.
            path: Current JSON Pointer path.
        """
        if not known_keys:
            # No schema info, skip unknown field checking
            return

        for key in obj.keys():
            if key not in known_keys:
                # Key is unknown, check for close matches
                field_path = f"{path}/{key}" if path else f"/{key}"

                # Only check for typos if key is long enough
                if len(key) >= MIN_KEY_LENGTH_FOR_TYPO_CHECK:
                    close_matches = self._find_close_matches(
                        key, known_keys, self._max_edit_distance
                    )

                    if close_matches:
                        # Found close match(es) - likely a typo
                        best_match, distance = close_matches[0]
                        suggestions = [m[0] for m in close_matches[:3]]

                        self._add_finding(
                            code=ErrorCode.SUSPICIOUS_KEY.value,
                            severity=Severity.WARNING,
                            path=field_path,
                            message=(
                                f"Unknown field '{key}'. "
                                f"Did you mean '{best_match}'?"
                            ),
                            expected={"valid_fields": list(known_keys)[:10]},
                            actual=key,
                            hint_category="typo",
                            hint_suggestions=suggestions,
                        )

    def _check_deprecated_fields(
        self,
        obj: Dict[str, Any],
        path: str
    ) -> None:
        """
        Check for usage of deprecated fields.

        Args:
            obj: Object to check.
            path: Current JSON Pointer path.
        """
        for key in obj.keys():
            field_path = f"{path}/{key}" if path else f"/{key}"

            # Check if field is deprecated
            deprecation_info = self.ir.get_deprecation_info(field_path)

            if deprecation_info:
                since_version = deprecation_info.get("since_version", "unknown")
                message = deprecation_info.get("message", "This field is deprecated")
                replacement = deprecation_info.get("replacement")
                removal_version = deprecation_info.get("removal_version")

                hint_msg = message
                if replacement:
                    hint_msg = f"Use '{replacement}' instead. {hint_msg}"
                if removal_version:
                    hint_msg = f"{hint_msg} Will be removed in version {removal_version}."

                self._add_finding(
                    code=ErrorCode.DEPRECATED_FIELD.value,
                    severity=Severity.WARNING,
                    path=field_path,
                    message=(
                        f"Field '{key}' is deprecated since version {since_version}"
                    ),
                    expected={"replacement": replacement} if replacement else None,
                    actual=key,
                    hint_category="deprecated",
                    hint_suggestions=[replacement] if replacement else [],
                )

            # Check if field has been renamed
            new_name = self.ir.get_renamed_to(field_path)

            if new_name:
                self._add_finding(
                    code=ErrorCode.RENAMED_FIELD.value,
                    severity=Severity.WARNING,
                    path=field_path,
                    message=(
                        f"Field '{key}' has been renamed to '{new_name}'"
                    ),
                    expected={"new_name": new_name},
                    actual=key,
                    hint_category="renamed",
                    hint_suggestions=[new_name.split("/")[-1] if "/" in new_name else new_name],
                )

    def _check_casing_consistency(
        self,
        obj: Dict[str, Any],
        path: str
    ) -> None:
        """
        Check for inconsistent key casing.

        Detects when keys don't follow the expected naming convention
        (typically snake_case for GreenLang schemas).

        Args:
            obj: Object to check.
            path: Current JSON Pointer path.
        """
        # Detect the dominant casing style in the object
        detected_styles: Dict[str, int] = {
            CASING_SNAKE_CASE: 0,
            CASING_CAMEL_CASE: 0,
            CASING_PASCAL_CASE: 0,
            CASING_KEBAB_CASE: 0,
            CASING_SCREAMING_SNAKE: 0,
        }

        for key in obj.keys():
            style = self._detect_casing_style(key)
            if style in detected_styles:
                detected_styles[style] += 1

        # Find the dominant style (most common)
        if not detected_styles or max(detected_styles.values()) == 0:
            return

        dominant_style = max(detected_styles, key=lambda k: detected_styles[k])

        # Default convention is snake_case (GreenLang standard)
        expected_convention = DEFAULT_NAMING_CONVENTION

        # Check each key for compliance
        for key in obj.keys():
            key_style = self._detect_casing_style(key)
            field_path = f"{path}/{key}" if path else f"/{key}"

            # Skip single-character keys or keys with only lowercase letters
            if len(key) < 2:
                continue

            # Check if key follows expected convention
            if expected_convention == CASING_SNAKE_CASE and not is_snake_case(key):
                # Only flag if key clearly follows a different style
                if key_style in (CASING_CAMEL_CASE, CASING_PASCAL_CASE):
                    suggested_name = to_snake_case(key)
                    self._add_finding(
                        code=ErrorCode.NONCOMPLIANT_CASING.value,
                        severity=Severity.WARNING,
                        path=field_path,
                        message=(
                            f"Field '{key}' does not follow {expected_convention} "
                            f"naming convention"
                        ),
                        expected={"convention": expected_convention},
                        actual={"key": key, "detected_style": key_style},
                        hint_category="casing",
                        hint_suggestions=[suggested_name],
                    )

    def _check_unit_formatting(
        self,
        value: Any,
        path: str
    ) -> None:
        """
        Suggest unit formatting improvements.

        Args:
            value: Value to check.
            path: JSON Pointer path to value.
        """
        # Check if this path has a unit spec
        unit_spec = self.ir.get_unit_spec(path)

        if not unit_spec:
            # No unit specification for this path
            return

        # Check if value is a string that might contain a unit
        if isinstance(value, str):
            # Pattern: "123 kWh" or "123kWh"
            unit_pattern = r"^(-?\d+\.?\d*)\s*([a-zA-Z]+.*)$"
            match = re.match(unit_pattern, value.strip())

            if match:
                numeric_part, unit_part = match.groups()

                # Suggest the canonical object format
                canonical = unit_spec.canonical

                self._add_finding(
                    code=ErrorCode.UNIT_FORMAT_STYLE.value,
                    severity=Severity.INFO,
                    path=path,
                    message=(
                        f"Unit format could be improved. Consider using object format: "
                        f'{{ "value": {numeric_part}, "unit": "{canonical}" }}'
                    ),
                    expected={"format": "object", "canonical_unit": canonical},
                    actual={"format": "string", "value": value},
                    hint_category="unit_format",
                    hint_suggestions=[
                        f'{{"value": {numeric_part}, "unit": "{canonical}"}}'
                    ],
                )

    def _check_suspicious_values(
        self,
        obj: Dict[str, Any],
        path: str
    ) -> None:
        """
        Check for suspicious values (empty strings, zeros, etc.).

        Args:
            obj: Object to check.
            path: Current JSON Pointer path.
        """
        for key, value in obj.items():
            field_path = f"{path}/{key}" if path else f"/{key}"

            # Check for empty strings
            if isinstance(value, str) and value == "":
                # Check if this is a required field or has constraints
                prop_ir = self.ir.get_property(field_path)

                if prop_ir and prop_ir.required:
                    self._add_finding(
                        code=ErrorCode.SUSPICIOUS_KEY.value,
                        severity=Severity.WARNING,
                        path=field_path,
                        message=(
                            f"Field '{key}' has an empty string value. "
                            f"This may be unintentional."
                        ),
                        expected={"non_empty": True},
                        actual="",
                        hint_category="empty_value",
                        hint_suggestions=["Provide a non-empty value or remove the field"],
                    )

            # Check for zero values in numeric fields that typically shouldn't be zero
            elif isinstance(value, (int, float)) and value == 0:
                # Check if this path has a unit spec (likely a measurement field)
                unit_spec = self.ir.get_unit_spec(field_path)

                if unit_spec:
                    # Measurement fields with zero are suspicious
                    self._add_finding(
                        code=ErrorCode.SUSPICIOUS_KEY.value,
                        severity=Severity.INFO,
                        path=field_path,
                        message=(
                            f"Field '{key}' has a zero value. "
                            f"Verify this is intentional for a {unit_spec.dimension} measurement."
                        ),
                        expected={"non_zero": True},
                        actual=0,
                        hint_category="zero_value",
                        hint_suggestions=["Verify this zero value is intentional"],
                    )

    def _find_close_matches(
        self,
        key: str,
        known_keys: Set[str],
        max_distance: int = 2
    ) -> List[Tuple[str, int]]:
        """
        Find keys within edit distance threshold.

        Returns list of (key, distance) tuples sorted by distance.

        Args:
            key: Unknown key to find matches for.
            known_keys: Set of valid keys to compare against.
            max_distance: Maximum Levenshtein distance to consider.

        Returns:
            List of (matching_key, distance) tuples, sorted by distance ascending.

        Example:
            >>> matches = linter._find_close_matches("emmisions", {"emissions", "energy"})
            >>> matches[0]
            ('emissions', 1)
        """
        matches: List[Tuple[str, int]] = []

        for known_key in known_keys:
            distance = self._levenshtein_distance(key.lower(), known_key.lower())

            if distance <= max_distance:
                matches.append((known_key, distance))

        # Sort by distance (closest first), then alphabetically
        matches.sort(key=lambda x: (x[1], x[0]))

        return matches

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """
        Calculate Levenshtein edit distance between two strings.

        The Levenshtein distance is the minimum number of single-character
        edits (insertions, deletions, substitutions) required to change
        one string into the other.

        Uses dynamic programming with O(m*n) time and O(min(m,n)) space.

        Args:
            s1: First string.
            s2: Second string.

        Returns:
            Edit distance as integer.

        Example:
            >>> linter._levenshtein_distance("emissions", "emmisions")
            1
            >>> linter._levenshtein_distance("cat", "dog")
            3
        """
        # Ensure s1 is the shorter string for space optimization
        if len(s1) > len(s2):
            s1, s2 = s2, s1

        m, n = len(s1), len(s2)

        # Edge cases
        if m == 0:
            return n
        if n == 0:
            return m

        # Use two rows for space optimization
        prev_row = list(range(m + 1))
        curr_row = [0] * (m + 1)

        for j in range(1, n + 1):
            curr_row[0] = j

            for i in range(1, m + 1):
                # Cost is 0 if characters match, 1 otherwise
                cost = 0 if s1[i - 1] == s2[j - 1] else 1

                # Minimum of three operations:
                # 1. Delete from s1 (prev_row[i] + 1)
                # 2. Insert into s1 (curr_row[i-1] + 1)
                # 3. Substitute (prev_row[i-1] + cost)
                curr_row[i] = min(
                    prev_row[i] + 1,      # deletion
                    curr_row[i - 1] + 1,  # insertion
                    prev_row[i - 1] + cost  # substitution
                )

            # Swap rows
            prev_row, curr_row = curr_row, prev_row

        return prev_row[m]

    def _detect_casing_style(self, key: str) -> str:
        """
        Detect casing style: snake_case, camelCase, PascalCase, etc.

        Args:
            key: Key name to analyze.

        Returns:
            String identifier for the detected casing style.

        Example:
            >>> linter._detect_casing_style("energy_consumption")
            'snake_case'
            >>> linter._detect_casing_style("energyConsumption")
            'camelCase'
        """
        if not key:
            return CASING_UNKNOWN

        if is_snake_case(key):
            return CASING_SNAKE_CASE
        elif is_screaming_snake_case(key):
            return CASING_SCREAMING_SNAKE
        elif is_camel_case(key):
            return CASING_CAMEL_CASE
        elif is_pascal_case(key):
            return CASING_PASCAL_CASE
        elif is_kebab_case(key):
            return CASING_KEBAB_CASE
        else:
            return CASING_UNKNOWN

    def _add_finding(
        self,
        code: str,
        severity: Severity,
        path: str,
        message: str,
        expected: Optional[Dict[str, Any]] = None,
        actual: Optional[Any] = None,
        hint_category: Optional[str] = None,
        hint_suggestions: Optional[List[Any]] = None,
    ) -> None:
        """
        Add a lint finding.

        Args:
            code: Error code (GLSCHEMA-W*).
            severity: Finding severity (WARNING or INFO).
            path: JSON Pointer path.
            message: Human-readable message.
            expected: Expected value/format.
            actual: Actual value found.
            hint_category: Category for the hint.
            hint_suggestions: List of suggested fixes.
        """
        hint = None
        if hint_category or hint_suggestions:
            hint = FindingHint(
                category=hint_category or "lint",
                suggested_values=hint_suggestions or [],
            )

        finding = Finding(
            code=code,
            severity=severity,
            path=path,
            message=message,
            expected=expected,
            actual=actual,
            hint=hint,
        )

        self._findings.append(finding)
        logger.debug("Lint finding: %s at %s: %s", code, path, message)


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================


def lint_payload(
    payload: Dict[str, Any],
    ir: SchemaIR,
    options: Optional[ValidationOptions] = None
) -> List[Finding]:
    """
    Convenience function for linting a payload.

    Creates a SchemaLinter instance and runs lint checks.

    Args:
        payload: The payload to lint.
        ir: Compiled schema IR.
        options: Optional validation options.

    Returns:
        List of lint findings.

    Example:
        >>> findings = lint_payload({"emmisions": 100}, ir)
        >>> print(findings[0].message)
        Unknown field 'emmisions'. Did you mean 'emissions'?
    """
    linter = SchemaLinter(ir, options)
    return linter.lint(payload)


# =============================================================================
# MODULE EXPORTS
# =============================================================================


__all__ = [
    # Main class
    "SchemaLinter",
    # Convenience function
    "lint_payload",
    # Casing detection helpers
    "is_snake_case",
    "is_camel_case",
    "is_pascal_case",
    "is_kebab_case",
    "is_screaming_snake_case",
    "to_snake_case",
    "to_camel_case",
    "to_pascal_case",
    # Casing constants
    "CASING_SNAKE_CASE",
    "CASING_CAMEL_CASE",
    "CASING_PASCAL_CASE",
    "CASING_KEBAB_CASE",
    "CASING_SCREAMING_SNAKE",
    "CASING_UNKNOWN",
]
