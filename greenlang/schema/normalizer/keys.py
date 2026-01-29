# -*- coding: utf-8 -*-
"""
Key Canonicalizer for GL-FOUND-X-002.

This module implements key canonicalization for payload normalization,
including alias resolution, casing normalization, and stable key ordering.

Key Features:
    - Resolves known aliases from schema definitions
    - Normalizes casing according to schema requirements (snake_case, camelCase, etc.)
    - Applies stable key ordering for reproducibility
    - Records all renames for audit trail

Design Principles:
    - Zero-hallucination: All transformations are deterministic and schema-driven
    - Provenance tracking: Every rename is recorded with reason
    - Idempotency: canonicalize(canonicalize(x)) == canonicalize(x)

Example:
    >>> from greenlang.schema.normalizer.keys import KeyCanonicalizer
    >>> canonicalizer = KeyCanonicalizer(schema_ir)
    >>> result, renames = canonicalizer.canonicalize({"energyConsumption": 100})
    >>> print(result)
    {"energy_consumption": 100}
    >>> print(renames[0].reason)
    RenameReason.CASING

Author: GreenLang Framework Team
Version: 0.1.0
GL-FOUND-X-002: Schema Compiler & Validator - Task 3.3
"""

from __future__ import annotations

import logging
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, ConfigDict, Field

from ..compiler.ir import SchemaIR

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Default casing style when not specified by schema
DEFAULT_CASING_STYLE = "snake_case"

# Maximum recursion depth to prevent stack overflow on deeply nested payloads
MAX_RECURSION_DEPTH = 100


# =============================================================================
# RENAME REASON ENUM
# =============================================================================


class RenameReason(str, Enum):
    """
    Reason for a key rename operation.

    Attributes:
        ALIAS: Schema-defined alias was resolved to canonical name
        CASING: Key casing was normalized to match schema convention
        TYPO_CORRECTION: Close match correction (user opt-in feature)
    """

    ALIAS = "alias"
    CASING = "casing"
    TYPO_CORRECTION = "typo"


# =============================================================================
# KEY RENAME MODEL
# =============================================================================


class KeyRename(BaseModel):
    """
    Record of a key rename operation.

    This model provides a complete audit trail for key transformations,
    including the path, original key, canonical key, and reason.

    Attributes:
        path: JSON Pointer to the parent object containing the renamed key
        original_key: The original key name before canonicalization
        canonical_key: The canonical key name after canonicalization
        reason: The reason for the rename (alias, casing, or typo)

    Example:
        >>> rename = KeyRename(
        ...     path="/data",
        ...     original_key="energyConsumption",
        ...     canonical_key="energy_consumption",
        ...     reason=RenameReason.CASING
        ... )
    """

    path: str = Field(
        ...,
        description="JSON Pointer to the parent object"
    )
    original_key: str = Field(
        ...,
        description="The original key name before canonicalization"
    )
    canonical_key: str = Field(
        ...,
        description="The canonical key name after canonicalization"
    )
    reason: RenameReason = Field(
        ...,
        description="The reason for the rename"
    )

    model_config = ConfigDict(frozen=True, extra="forbid")

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary representation."""
        return {
            "path": self.path,
            "original_key": self.original_key,
            "canonical_key": self.canonical_key,
            "reason": self.reason.value,
        }


# =============================================================================
# CASING CONVERSION UTILITIES
# =============================================================================


def to_snake_case(s: str) -> str:
    """
    Convert string to snake_case.

    Handles camelCase, PascalCase, and mixed formats.

    Args:
        s: String to convert

    Returns:
        String in snake_case format

    Examples:
        >>> to_snake_case("energyConsumption")
        'energy_consumption'
        >>> to_snake_case("EnergyConsumption")
        'energy_consumption'
        >>> to_snake_case("HTTPResponse")
        'http_response'
        >>> to_snake_case("already_snake_case")
        'already_snake_case'
    """
    if not s:
        return s

    # Handle consecutive uppercase letters (e.g., HTTP -> http)
    # First pass: insert underscore between uppercase and lowercase
    s = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', s)

    # Second pass: insert underscore between lowercase/digit and uppercase
    s = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s)

    # Convert to lowercase
    result = s.lower()

    # Clean up multiple consecutive underscores
    result = re.sub(r'_+', '_', result)

    # Remove leading/trailing underscores
    result = result.strip('_')

    return result


def to_camel_case(s: str) -> str:
    """
    Convert string to camelCase.

    Handles snake_case, PascalCase, and mixed formats.

    Args:
        s: String to convert

    Returns:
        String in camelCase format

    Examples:
        >>> to_camel_case("energy_consumption")
        'energyConsumption'
        >>> to_camel_case("EnergyConsumption")
        'energyConsumption'
        >>> to_camel_case("already_camelCase")
        'alreadyCamelCase'
    """
    if not s:
        return s

    # First convert to snake_case to normalize
    snake = to_snake_case(s)

    # Split on underscores
    parts = snake.split('_')

    if not parts:
        return s

    # First part lowercase, rest title case
    result = parts[0].lower()
    for part in parts[1:]:
        if part:
            result += part.title()

    return result


def to_pascal_case(s: str) -> str:
    """
    Convert string to PascalCase.

    Handles snake_case, camelCase, and mixed formats.

    Args:
        s: String to convert

    Returns:
        String in PascalCase format

    Examples:
        >>> to_pascal_case("energy_consumption")
        'EnergyConsumption'
        >>> to_pascal_case("energyConsumption")
        'EnergyConsumption'
        >>> to_pascal_case("already_PascalCase")
        'AlreadyPascalCase'
    """
    if not s:
        return s

    # First convert to snake_case to normalize
    snake = to_snake_case(s)

    # Split on underscores and title case each part
    parts = snake.split('_')

    return ''.join(part.title() for part in parts if part)


def detect_casing(s: str) -> str:
    """
    Detect the casing style of a string.

    Args:
        s: String to analyze

    Returns:
        Casing style: "snake_case", "camelCase", "PascalCase", or "unknown"

    Examples:
        >>> detect_casing("energy_consumption")
        'snake_case'
        >>> detect_casing("energyConsumption")
        'camelCase'
        >>> detect_casing("EnergyConsumption")
        'PascalCase'
        >>> detect_casing("ENERGY")
        'unknown'
    """
    if not s:
        return "unknown"

    # Check for snake_case: contains underscores and is lowercase
    if '_' in s:
        # Has underscores - check if it's snake_case
        without_underscores = s.replace('_', '')
        if without_underscores.islower() or without_underscores.isupper():
            return "snake_case"
        return "unknown"

    # No underscores - check for camelCase vs PascalCase
    if s[0].isupper():
        # Starts with uppercase
        if len(s) > 1 and any(c.islower() for c in s):
            return "PascalCase"
        return "unknown"

    if s[0].islower():
        # Starts with lowercase
        if any(c.isupper() for c in s):
            return "camelCase"
        # All lowercase, no underscores - could be single word
        return "unknown"

    return "unknown"


def normalize_to_casing(key: str, target_casing: str) -> str:
    """
    Normalize a key to the target casing style.

    Args:
        key: Key to normalize
        target_casing: Target casing style ("snake_case", "camelCase", "PascalCase")

    Returns:
        Key normalized to target casing

    Raises:
        ValueError: If target_casing is not supported

    Examples:
        >>> normalize_to_casing("energyConsumption", "snake_case")
        'energy_consumption'
        >>> normalize_to_casing("energy_consumption", "camelCase")
        'energyConsumption'
    """
    target = target_casing.lower().replace("_", "")

    if target == "snakecase":
        return to_snake_case(key)
    elif target == "camelcase":
        return to_camel_case(key)
    elif target == "pascalcase":
        return to_pascal_case(key)
    else:
        # Unknown casing style - return original
        logger.warning(
            f"Unknown casing style '{target_casing}', returning original key"
        )
        return key


# =============================================================================
# KEY CANONICALIZER
# =============================================================================


class KeyCanonicalizer:
    """
    Canonicalizes object keys in payloads.

    This class implements key canonicalization according to schema definitions:
    1. Resolves known aliases from schema
    2. Normalizes casing if schema demands
    3. Applies stable key ordering for reproducibility
    4. Records all renames for audit trail

    The canonicalizer is designed to be idempotent - applying it multiple
    times to the same payload produces the same result.

    Attributes:
        ir: Compiled schema intermediate representation
        expected_casing: Expected casing style from schema (default: snake_case)
        enable_typo_correction: Whether to attempt typo correction (default: False)
        typo_threshold: Maximum edit distance for typo correction (default: 2)

    Example:
        >>> ir = SchemaIR(...)
        >>> canonicalizer = KeyCanonicalizer(ir)
        >>> result, renames = canonicalizer.canonicalize({
        ...     "energyConsumption": 100,
        ...     "old_field_name": "value"
        ... })
        >>> print(result)
        {"energy_consumption": 100, "new_field_name": "value"}
    """

    def __init__(
        self,
        ir: SchemaIR,
        expected_casing: Optional[str] = None,
        enable_typo_correction: bool = False,
        typo_threshold: int = 2,
    ):
        """
        Initialize the KeyCanonicalizer.

        Args:
            ir: Compiled schema intermediate representation
            expected_casing: Expected casing style (default: schema-defined or snake_case)
            enable_typo_correction: Enable typo correction (requires user opt-in)
            typo_threshold: Maximum edit distance for typo correction
        """
        self.ir = ir
        self.expected_casing = expected_casing or DEFAULT_CASING_STYLE
        self.enable_typo_correction = enable_typo_correction
        self.typo_threshold = typo_threshold

        # Internal rename tracking
        self._renames: List[KeyRename] = []

        # Build alias lookup from schema
        self._alias_to_canonical = self._build_alias_map()

        # Build known keys set from schema properties
        self._known_keys = self._build_known_keys()

        logger.debug(
            f"KeyCanonicalizer initialized with {len(self._alias_to_canonical)} aliases, "
            f"{len(self._known_keys)} known keys, casing={self.expected_casing}"
        )

    def _build_alias_map(self) -> Dict[str, str]:
        """
        Build alias to canonical name mapping from schema IR.

        The schema IR provides renamed_fields which maps old names to new names.
        Additionally, property extensions may define aliases.

        Returns:
            Dictionary mapping aliases to their canonical names
        """
        alias_map: Dict[str, str] = {}

        # Add renamed fields (old_name -> new_name)
        if self.ir.renamed_fields:
            for old_name, new_name in self.ir.renamed_fields.items():
                alias_map[old_name] = new_name
                logger.debug(f"Added alias: '{old_name}' -> '{new_name}'")

        # Check property extensions for additional aliases
        for path, prop in self.ir.properties.items():
            if prop.gl_extensions and isinstance(prop.gl_extensions, dict):
                # Check for aliases in extensions
                aliases = prop.gl_extensions.get("aliases", [])
                if aliases:
                    # Extract the key name from the path
                    key_name = path.rsplit("/", 1)[-1] if "/" in path else path
                    for alias in aliases:
                        if alias and alias != key_name:
                            alias_map[alias] = key_name
                            logger.debug(
                                f"Added alias from extensions: '{alias}' -> '{key_name}'"
                            )

        return alias_map

    def _build_known_keys(self) -> Set[str]:
        """
        Build set of known keys from schema properties.

        Returns:
            Set of known key names from schema
        """
        known_keys: Set[str] = set()

        for path in self.ir.properties.keys():
            # Extract the key name from the path
            if "/" in path:
                key_name = path.rsplit("/", 1)[-1]
            else:
                key_name = path

            if key_name:
                known_keys.add(key_name)

        return known_keys

    def canonicalize(
        self,
        payload: Dict[str, Any],
        path: str = "",
    ) -> Tuple[Dict[str, Any], List[KeyRename]]:
        """
        Canonicalize all keys in payload.

        This method applies alias resolution, casing normalization, and stable
        ordering to all keys in the payload, including nested objects.

        Args:
            payload: The payload dictionary to canonicalize
            path: JSON Pointer path prefix (default: root)

        Returns:
            Tuple of (canonicalized payload, list of KeyRename records)

        Raises:
            ValueError: If payload is not a dictionary
            RecursionError: If nesting exceeds MAX_RECURSION_DEPTH

        Example:
            >>> result, renames = canonicalizer.canonicalize({"OldName": 100})
            >>> print(result)
            {"new_name": 100}
        """
        if not isinstance(payload, dict):
            raise ValueError(
                f"Payload must be a dictionary, got {type(payload).__name__}"
            )

        # Clear previous renames
        self._renames = []

        # Process the payload
        start_time = _get_time_ms()

        try:
            result = self._canonicalize_object(payload, path, depth=0)
        except RecursionError:
            logger.error(
                f"Recursion depth exceeded {MAX_RECURSION_DEPTH} during canonicalization"
            )
            raise

        # Apply stable ordering
        result = self._apply_stable_ordering(result, path)

        elapsed_ms = _get_time_ms() - start_time
        logger.debug(
            f"Canonicalized payload in {elapsed_ms:.2f}ms with {len(self._renames)} renames"
        )

        return result, self._renames.copy()

    def _canonicalize_object(
        self,
        obj: Dict[str, Any],
        path: str,
        depth: int = 0,
    ) -> Dict[str, Any]:
        """
        Canonicalize keys in a single object.

        Processes each key through:
        1. Alias resolution
        2. Casing normalization
        3. Recursive processing for nested values

        Args:
            obj: The object dictionary to process
            path: Current JSON Pointer path
            depth: Current recursion depth

        Returns:
            Canonicalized object dictionary
        """
        if depth > MAX_RECURSION_DEPTH:
            raise RecursionError(
                f"Maximum recursion depth {MAX_RECURSION_DEPTH} exceeded at path '{path}'"
            )

        result: Dict[str, Any] = {}

        for original_key, value in obj.items():
            # Determine canonical key
            canonical_key = original_key
            rename_reason: Optional[RenameReason] = None

            # Step 1: Check for alias
            alias_result = self._resolve_alias(original_key, path)
            if alias_result is not None:
                canonical_key = alias_result
                rename_reason = RenameReason.ALIAS

            # Step 2: Normalize casing (if not already renamed by alias)
            if rename_reason is None:
                casing_result = self._normalize_casing(
                    canonical_key, self.expected_casing, path
                )
                if casing_result is not None:
                    canonical_key = casing_result
                    rename_reason = RenameReason.CASING

            # Step 3: Typo correction (opt-in feature)
            if (
                rename_reason is None
                and self.enable_typo_correction
                and canonical_key not in self._known_keys
            ):
                typo_result = self._correct_typo(canonical_key, path)
                if typo_result is not None:
                    canonical_key = typo_result
                    rename_reason = RenameReason.TYPO_CORRECTION

            # Record rename if key changed
            if canonical_key != original_key:
                self._record_rename(path, original_key, canonical_key, rename_reason)

            # Process nested values recursively
            child_path = f"{path}/{canonical_key}"
            canonicalized_value = self._canonicalize_value(value, child_path, depth)

            result[canonical_key] = canonicalized_value

        return result

    def _canonicalize_value(
        self,
        value: Any,
        path: str,
        depth: int,
    ) -> Any:
        """
        Canonicalize a value, recursing into nested structures.

        Args:
            value: The value to process
            path: Current JSON Pointer path
            depth: Current recursion depth

        Returns:
            Canonicalized value
        """
        if isinstance(value, dict):
            return self._canonicalize_object(value, path, depth + 1)
        elif isinstance(value, list):
            return [
                self._canonicalize_value(item, f"{path}/{i}", depth + 1)
                for i, item in enumerate(value)
            ]
        else:
            # Primitive value - no canonicalization needed
            return value

    def _resolve_alias(
        self,
        key: str,
        path: str,
    ) -> Optional[str]:
        """
        Check if key is an alias and return canonical name.

        Uses aliases defined in schema IR (renamed_fields and property aliases).

        Args:
            key: The key to check
            path: Current JSON Pointer path (for context-aware resolution)

        Returns:
            Canonical name if key is an alias, None otherwise

        Example:
            >>> canonicalizer._resolve_alias("old_field_name", "")
            "new_field_name"
        """
        # Direct lookup in alias map
        if key in self._alias_to_canonical:
            canonical = self._alias_to_canonical[key]
            logger.debug(f"Resolved alias '{key}' -> '{canonical}' at path '{path}'")
            return canonical

        return None

    def _normalize_casing(
        self,
        key: str,
        expected_casing: str,
        path: str,
    ) -> Optional[str]:
        """
        Normalize key casing if needed.

        Args:
            key: The key to normalize
            expected_casing: Expected casing style ("snake_case", "camelCase", etc.)
            path: Current JSON Pointer path

        Returns:
            Normalized key if casing changed, None if no change needed

        Example:
            >>> canonicalizer._normalize_casing("energyConsumption", "snake_case", "")
            "energy_consumption"
        """
        # Convert key to expected casing
        normalized = normalize_to_casing(key, expected_casing)

        # Only return if actually changed
        if normalized != key:
            logger.debug(
                f"Normalized casing '{key}' -> '{normalized}' "
                f"(expected: {expected_casing}) at path '{path}'"
            )
            return normalized

        return None

    def _correct_typo(
        self,
        key: str,
        path: str,
    ) -> Optional[str]:
        """
        Attempt to correct a typo in key name.

        Uses Levenshtein distance to find close matches in known keys.
        This is an opt-in feature requiring enable_typo_correction=True.

        Args:
            key: The unknown key to check
            path: Current JSON Pointer path

        Returns:
            Corrected key if a close match found, None otherwise
        """
        if not self._known_keys:
            return None

        best_match: Optional[str] = None
        best_distance = self.typo_threshold + 1

        for known_key in self._known_keys:
            distance = _levenshtein_distance(key.lower(), known_key.lower())
            if distance <= self.typo_threshold and distance < best_distance:
                best_distance = distance
                best_match = known_key

        if best_match is not None:
            logger.info(
                f"Typo correction: '{key}' -> '{best_match}' "
                f"(distance={best_distance}) at path '{path}'"
            )
            return best_match

        return None

    def _apply_stable_ordering(
        self,
        obj: Dict[str, Any],
        path: str,
    ) -> Dict[str, Any]:
        """
        Apply stable key ordering for reproducibility.

        Order priority:
        1. Required fields first (in schema order)
        2. Optional fields (alphabetically)
        3. Unknown fields (alphabetically)

        Args:
            obj: The object to reorder
            path: Current JSON Pointer path

        Returns:
            Object with stable key ordering
        """
        # Build ordered key list
        required_keys: List[str] = []
        optional_keys: List[str] = []
        unknown_keys: List[str] = []

        for key in obj.keys():
            full_path = f"{path}/{key}" if path else f"/{key}"

            if full_path in self.ir.required_paths:
                required_keys.append(key)
            elif full_path in self.ir.properties or f"/{key}" in self.ir.properties:
                optional_keys.append(key)
            else:
                unknown_keys.append(key)

        # Sort each group
        # Required keys maintain schema order (preserve insertion order)
        # Optional and unknown keys sorted alphabetically
        optional_keys.sort()
        unknown_keys.sort()

        # Build ordered result
        ordered_keys = required_keys + optional_keys + unknown_keys
        result = {key: obj[key] for key in ordered_keys}

        # Recursively apply ordering to nested objects
        for key, value in result.items():
            if isinstance(value, dict):
                child_path = f"{path}/{key}"
                result[key] = self._apply_stable_ordering(value, child_path)
            elif isinstance(value, list):
                result[key] = [
                    self._apply_stable_ordering(item, f"{path}/{key}/{i}")
                    if isinstance(item, dict)
                    else item
                    for i, item in enumerate(value)
                ]

        return result

    def _record_rename(
        self,
        path: str,
        original: str,
        canonical: str,
        reason: Optional[RenameReason],
    ) -> None:
        """
        Record a key rename in the audit trail.

        Args:
            path: JSON Pointer to parent object
            original: Original key name
            canonical: Canonical key name
            reason: Reason for the rename
        """
        if reason is None:
            reason = RenameReason.ALIAS  # Default reason

        rename = KeyRename(
            path=path,
            original_key=original,
            canonical_key=canonical,
            reason=reason,
        )

        self._renames.append(rename)

        logger.debug(
            f"Recorded rename: '{original}' -> '{canonical}' "
            f"(reason={reason.value}) at path '{path}'"
        )

    def get_renames(self) -> List[KeyRename]:
        """
        Get all rename records from the last canonicalization.

        Returns:
            List of KeyRename records
        """
        return self._renames.copy()

    def clear_renames(self) -> None:
        """Clear all rename records."""
        self._renames = []
        logger.debug("Cleared rename records")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate Levenshtein (edit) distance between two strings.

    Uses dynamic programming for O(n*m) time complexity.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Minimum edit distance between strings

    Examples:
        >>> _levenshtein_distance("energy", "enregy")
        2
        >>> _levenshtein_distance("same", "same")
        0
    """
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    # Use only two rows for space efficiency
    previous_row = list(range(len(s2) + 1))
    current_row = [0] * (len(s2) + 1)

    for i, c1 in enumerate(s1):
        current_row[0] = i + 1

        for j, c2 in enumerate(s2):
            # Insertions, deletions, substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (0 if c1 == c2 else 1)

            current_row[j + 1] = min(insertions, deletions, substitutions)

        previous_row, current_row = current_row, previous_row

    return previous_row[len(s2)]


def _get_time_ms() -> float:
    """Get current time in milliseconds."""
    import time
    return time.time() * 1000


# =============================================================================
# MODULE EXPORTS
# =============================================================================


__all__ = [
    # Enums
    "RenameReason",
    # Models
    "KeyRename",
    # Main class
    "KeyCanonicalizer",
    # Utility functions
    "to_snake_case",
    "to_camel_case",
    "to_pascal_case",
    "detect_casing",
    "normalize_to_casing",
]
